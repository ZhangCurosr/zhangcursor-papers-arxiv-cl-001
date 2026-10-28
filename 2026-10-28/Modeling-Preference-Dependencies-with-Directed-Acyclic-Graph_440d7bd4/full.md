# Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation

Mingzhe Lu<sup>1,2</sup> Jian Qin<sup>1,2</sup> Ruize Qin<sup>3</sup> Jinhao Liu<sup>1,2</sup> Yunpeng Li<sup>1,2,\*</sup> Hang Su<sup>4,\*</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China

<sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China

<sup>3</sup>Faculty of Engineering and IT, The University of Melbourne, Melbourne, Australia

<sup>4</sup>Independent Researcher

Corresponding authors: liyunpeng@iie.ac.cn; hangsucuhk@gmail.com.

## Abstract

Narrative-driven recommendation allows users to express multifaceted needs through free-form text where stated preferences naturally qualify and condition one another. Existing methods typically treat narratives as flat keyword combinations or unconstrained generative queries, failing to capture how complex preferences hierarchically refine and contextualize each other. We propose the Preference Dependency Graph Framework (PDG-F), a narrativedriven recommendation framework centered on explicit preference dependencies. Specifically, PDG-A (PDG-Agent) first parses unstructured requests to organize user preferences and their mutual qualifications into a directed acyclic Preference Dependency Graph (PDG). Through this graph, candidate evidence from the open Web is acquired via directed path retrieval, and final recommendations are determined by topological path ranking. Comprehensive experiments across three diverse domains spanning English and Chinese contexts confirm that PDG-F consistently achieves superior recommendation performance over competitive baselines, opendomain retrieval agents, and frontier language models. Further diagnostic analyses show that our dependency-guided retrieval markedly raises the attainable candidate-pool ceiling, demonstrating the fundamental value of modeling preference dependencies in narrative-driven recommendation.

## 1 Introduction

Relevance to Web. Traditional recommender systems on Web platforms largely rely on historical interactions or structured profiles to approximate user intent [1, 19]. However, these signals often fall short as modern Web users increasingly turn to free-form natural language narratives [14, 17, 21]. While the emergence of large lan guage models (LLMs) ofers new opportunities to interpret such narratives [12, 27, 31], retrieving candidates from open Web resources and evaluating their suitability require preserving how users’ preferences condition and constrain one another. Against this backdrop, we study narrative-driven recommendation (NDR) in open-web environments, aiming to provide personalized recommendations by deciphering rich, unstructured user requests [2, 4].

Existing eforts have explored diferent paradigms to transform narratives into recommendation evidence. Early work primarily relied on explicit clues. For example, Bogers and Koolen [5] combined example items with narrative context. Eberhard et al. [10] used annotated genres and keywords for post-filtering. Subsequent work further explored automated element extraction to reduce manual annotation [7]. Another line explores semantic matching between narratives and items through embedding representations [9]. Recent work further explores the potential of directly generating recommendations from narratives with LLMs [8, 13]. To further empower LLMs with external evidence, Zhu et al. [32] introduce collaborative retrieval to provide behavioral signals. Shi et al. [28] apply open-domain retrieval and attribute completion to rank candidates. Kanchi et al. [16] utilize generated plot tags for fine-grained reranking. However, preferences expressed in narratives often mutually qualify each other [24], a dimension still insuficiently modeled in existing methods.

User preferences in narratives are not merely flat collections of independent elements, but rather hierarchical and dependency structures formed through conditional qualifications. Consider the following movie recommendation request:

“I’d like fantasy with romance as a subplot, rather than the main attraction. [. . . ] I need believable individual characters and a relationship shown through conversation, humour, and handling disagreements, not mostly sex.”

In this query, “subplot” constrains the prominence of romance within the overarching narrative, while conversation, humour, and handling disagreements further qualify the specific manner in which this relationship unfolds. Neglecting such qualifying dependencies can easily cause a film that superficially covers fantasy, romance, and humor to be falsely identified as a match, such as The Lord ofthe Rings. Yet the humor belongs to camaraderie instead of romance, failing the qualification. In contrast, Howl’s Moving Castle develops the romance through bickering and everyday interaction, satisfying the requested dynamic. This contrast reveals the necessity of explicitly modeling qualification and dependency structures among narrative preferences to guide the retrieval and evaluation of Web evidence for recommendation.

We propose PDG-F (Preference Dependency Graph Framework), a narrative-driven recommendation framework comprising a graph construction agent, directed path retrieval, and topological path ranking. Specifically, the graph construction agent, PDG-A, extracts diverse elements from the narrative description and unfolds them into discrete nodes through a two-layer structure of perspective and instantiation. On this basis, PDG-A evaluates the qualification relationships between node pairs within the context of the entire narrative to form directed edges. After resolving cycles, it generates the concrete preference for each current node in topological order conditioned on the preference context of its ancestral nodes, al lowing upstream preferences to be progressively refined with each additional qualification to complete the preference dependency graph (PDG). Relying on this PDG, we first acquire candidates and their associated evidence from the open Web via directed path retrieval, and then compute candidate scores through topological path ranking, ultimately delivering recommendation results that better fit the specific preferences of the user.

The main contributions of this work are summarized as follows.

• We propose PDG-F (Preference Dependency Graph Framework), a narrative-driven recommendation framework that organizes user qualifying constraints into a progressively refined directed dependency structure, providing a principled structural representation for complex composite preferences.

• We develop a structure-guided decision mechanism built directly upon the constructed PDG (Preference Dependency Graph), comprising directed path retrieval to acquire candidate evidence from the Web under specific qualification constraints and topological path ranking to compute candidate scores, thereby unifying evidence gathering and final recommendation under a coherent structure.

• We conduct comprehensive experiments across three distinct domains encompassing movies, books, and anime in both English and Chinese contexts, demonstrating that PDG-F consistently achieves state-of-the-art performance against competitive baselines as well as modern commercial retrieval engines alongside frontier models and LLM agents, with ablations and diagnostic analyses verifying the eficacy of dependency modeling.

## 2 Related Work

Translating natural narratives into accurate item recommendations hinges on efectively understanding and operationalizing the mul tifaceted needs woven into unstructured text. Guided by varying depths of textual interpretation and representation paradigms, existing eforts have broadly evolved across three paradigms: explicit feature matching, representation learning, and LLM-based recommendation with evidence augmentation.

Explicit feature matching focuses on distilling concrete recommendation signals from unstructured narratives through signal integration, request element extraction, and narrative trope generation. Early work by Bogers and Koolen [5] combined narrative context with referenced example items to exploit the complementarity between explicit exemplars and descriptive intent. To obtain more specific recommendation cues, Eberhard et al. [7] automated the extraction of request elements into discrete recommendation cues, while Kanchi et al. [16] generated narrative trope tags from book descriptions to incorporate specific story patterns into candidate reranking. However, these methods primarily organize recommendation around feature matching, still lacking explicit representations for the modifier and qualification relationships among preference elements.

Representation learning approaches compute relevance by learning semantic correspondences between narrative requests and items, expanding from embedding-based matching [18, 26] to task adaptation, data augmentation, and general-purpose representation transfer. Early work by Eberhard et al. [9] leveraged embedding representations to analyze the predictive utility of distinct request elements, while Chaganty et al. [6] adopted contrastive learning to train dual-encoder retrieval models between conversational context and songs. Addressing the sparsity of narrative supervision, Mysore et al. [23] utilized LLMs to synthesize narrative queries from user interactions, providing supervision for dense retrieval models. To better capture domain semantics, Epure et al. [11] fine-tuned models to align natural language requests with high-level track descriptions. Broadening the scope to general recommendation and search, Attimonelli et al. [3] validated the zero-shot capabilities of general-purpose text embeddings. Although these eforts substantially advance semantic relevance modeling, they still lack explicit characterization of the qualification scope and inter-dependencies among diverse preference criteria.

LLM-based methods leverage powerful language understanding and reasoning capabilities to interpret narrative intents, evolving from direct generation relying on internal parametric knowledge toward evidence-augmented paradigms that integrate external retrieval and structured attributes [20]. Eberhard et al. [8] systematically evaluated diferent models and prompting strategies, bench marking the feasibility and boundaries of directly generating recommendations from parametric memory. To alleviate the limitations of static parametric knowledge, subsequent eforts focus on equipping LLMs with external grounding evidence: Zhu et al. [32] integrated collaborative retrieval into conversational recommendation to inject behavioral signals for candidate generation, while Shi et al. [28] utilized dynamic attribute templates, open-domain retrieval, and attribute completion to construct structured profiles for fine-grained candidate comparison. Despite enriching the grounding evidence for candidates, these approaches still do not explicitly leverage preference dependencies to coordinate retrieval and candidate evaluation.

## 3 Methodology

## 3.1 Problem Formulation

Given a user’s natural language narrative request �, narrativedriven recommendation aims to return a ranked list of top-� items from the item space I that satisfy the articulated needs. The request � can incorporate descriptions of the target item, reference items, and user evaluations. The system output is an ordered recommendation list:

$$
\begin{array} { r } { \mathcal { R } _ { K } ( q ) = [ i _ { 1 } , i _ { 2 } , . . . , i _ { K } ] , \qquad i _ { j } \in \mathcal { I } . } \end{array}\tag{1}
$$

Notably, NDR does not presuppose a fixed candidate item pool; candidates are typically derived from web retrieval or the parametric knowledge of LLMs. Given the request �, the system first acquires a candidate set $C _ { q } \subseteq { \cal T }$ along with its supporting evidence $\varepsilon _ { q } ,$ and subsequently scores and ranks the candidates:

$$
( C _ { q } , \mathcal { E } _ { q } ) = \mathrm { A c q u i r e } ( q ) , \qquad \mathcal { R } _ { K } ( q ) = \mathrm { T o p K } _ { i \in C _ { q } } S ( i \mid q , \mathcal { E } _ { q } ) ,\tag{2}
$$

where Acquire(·) denotes the process of obtaining candidate items and associated evidence from the aforementioned sources, and $S ( i \mid q , \mathcal { E } _ { q } )$ evaluates the extent to which candidate item � satisfies request � based on evidence $\varepsilon _ { q } .$

## 3.2 Preference Graph Construction

PDG-A, an autonomous decision-making agent, first determines interpretive perspectives and their concrete instantiations based on narrative �, expanding the user’s preference descriptions into a discrete node set $V _ { q } ,$ , where each node � has an initial representation $c _ { v } ^ { 0 } .$ It then identifies qualification relationships between nodes within the narrative context to construct directed edges $( u , v ) \in E _ { q } .$ . Finally, it resolves cycles and generates a concrete preference description $f _ { v }$ for each node in topological order, conditioned on the preferences expressed by ancestor nodes, completing the node representation as $\boldsymbol { c _ { v } } = \left( c _ { v } ^ { 0 } , f _ { v } \right)$ . This yields a request-specific preference dependency graph $\mathcal { G } _ { q } = ( V _ { q } , E _ { q } , \mathcal { F } _ { q } )$ , where $\mathcal { F } _ { q } = \{ f _ { v } \} _ { v \in V _ { q } }$

![](images/81f0abfcefc3a945be4158334dcd9a5b43bd7a57070a4368832fafdaa1095442.jpg)  
Figure 1: Overview of PDG-F. PDG-A constructs the preference dependency graph, which guides directed path retrieval and topological path ranking.

3.2.1 Node: Preference Instantiation. A user narrative � typically intertwines referenced items, subjective feedback, and explicit criteria for recommendations. PDG-A first distills these into a set of preference atoms $\mathcal { A } _ { q } = \{ a _ { i } \} _ { i = 1 } ^ { M }$ , and subsequently decomposes each atom along a two-tiered hierarchy comprising interpretive perspectives and their concrete instantiations. Formally, for each atom $a _ { i } \in \mathcal { A } _ { q } \colon$

• An interpretive perspective � ∈ $\mathcal { P } _ { i }$ specifies the semantic dimension along which the atom is interpreted (e.g., genre, director, or cast);

• A concrete instantiation $z \in \mathcal { T } _ { i , p }$ anchors the specific realization within that perspective.

Each admissible tuple $( a _ { i } , p , z )$ instantiates a distinct preference node � $\in V _ { q } .$ The initial state of node � is formalized as a quadruplet:

$$
c _ { v } ^ { 0 } = \langle \mathrm { a t o m } : a _ { i } , \mathrm { p e r s p e c t i v e } : p , \mathrm { i n s t a n t i a t i o n } : z , \mathrm { f e a t u r e } : \emptyset \rangle .\tag{3}
$$

Here, the feature field is initialized to ∅, reserved for the contextaware preference description generated later via topological propagation over ancestral constraints.

For example, given the raw user input: “I am looking for a movie similar to Harry Potter. . . ”, PDG-A extracts the preference atom $a _ { i } = H a r r y$ Potter. Conditioned on this atom, the interpretive perspectives determined by the agent can include genre, director, or actor, among others. Taking the perspective � = actor as an exam ple, the concrete instantiation can be � = Emma Watson, which forms distinct nodes alongside other instances such as Daniel Radclife and Rupert Grint. Both the perspective set $\mathcal { P } _ { i }$ and instantiation set ${ \mathcal { T } } _ { i , p }$ are dynamically determined by PDG-A conditioned on the current atom and the full narrative context, ultimately yielding $\begin{array} { r } { \vert V _ { q } \vert = \sum _ { a _ { i } \in \mathcal A _ { q } } \sum _ { p \in \mathcal P _ { i } } \vert \mathcal T _ { i , p } \vert } \end{array}$ nodes.

3.2.2 Edge: Qualification Dependencies. Independent node representations do not yet determine their semantics in a compositional context, where the same element may qualify diferent preference targets. To explicitly capture such relationships, we define a directed edge (�, �) indicating that node � imposes a concrete qualification upon the base preference expressed by node �. The edge direction is determined by the semantic structure of narrative �, reflecting a dependency directed from a foundational interpretation toward further refinement.

For any two distinct nodes, PDG-A jointly considers the complete narrative �, the initial representations $c _ { u } ^ { 0 } , c _ { v } ^ { 0 } ,$ , and their original atoms to determine whether one node qualifies the preference target, presentation manner, or applicability conditions of the other:

$$
d _ { q } ( u , v ) \in \{ ( u , v ) , ( v , u ) , \emptyset \} ,\tag{4}
$$

where (�, �) indicates that � further qualifies �, (�, �) indicates qualification in the reverse direction, and ∅ denotes the absence of any qualification dependency. The full narrative provides the overarching context, while the two nodes specify the concrete preference interpretations under examination.

For example, if � involves “romance gradually develops through conflict”, let � denote the romance node and � the conflict interaction node; then $d _ { q } ( u , v ) = ( u , v )$ , establishing the dependency � → � to signify that conflict interactions further qualify how the romance unfolds. Conversely, if � merely demands romance and conflict as independent requirements, then $d _ { q } ( u , v ) = \emptyset$ , and no edge is created.

PDG-A evaluates all unordered node pairs and aggregates the non-empty decisions into the initial edge set:

$$
{ \cal E } _ { q } ^ { 0 } = \left\{ d _ { q } ( u , v ) \ \big | \ \{ u , v \} \subseteq { \cal V } _ { q } , \ u \neq v , \ d _ { q } ( u , v ) \neq \emptyset \right\} .\tag{5}
$$

This yields the initial directed graph $\mathcal { G } _ { q } ^ { 0 } = ( V _ { q } , E _ { q } ^ { 0 } )$ . A node can possess multiple successors, representing divergent refinement branches of the same preference, as well as multiple predecessors, preserving its dependencies across multiple upstream contexts.

3.2.3 Graph: Cycle Resolution. Local judgments between node pairs may produce directed cycles, creating circular qualifications that prevent preferences from being progressively refined in a consistent order. PDG-A therefore revisits cyclic dependencies within the complete narrative context to resolve conflicts among local judgments.

Given the current edge set $E _ { q } ^ { t } { \mathrm { : } }$ we identify a strongly connected component $S _ { t }$ containing a directed cycle. Based on narrative $q ,$ the relevant nodes and their original atoms, the internal dependencies, and the external connections of the component, PDG-A selects the internal edge $e _ { t }$ least supported by the narrative for removal:

$$
\begin{array} { r } { e _ { t } \in E _ { q } ^ { t } \cap ( S _ { t } \times S _ { t } ) , \qquad E _ { q } ^ { t + 1 } = E _ { q } ^ { t } \setminus \{ e _ { t } \} . } \end{array}\tag{6}
$$

Strongly connected components are recomputed after each removal until no directed cycle remains.

For example, suppose � expresses that “romance gradually develops through conflict and deepens through subsequent reconciliation.” Let � = romance, � = conflict interactions, and $\begin{array} { r l } { w } & { { } = } \end{array}$ reconciliation after conflict. If local judgments establish dependencies $d _ { q } ( u , v ) , d _ { q } ( v , w )$ , and $d _ { q } ( w , u )$ , they form a directed cycle $u \ \to \ v \ \to \ w \ \to \ u .$ . Within the narrative context, the first two dependencies progressively qualify how the romance develops, whereas (�, �) returns from reconciliation to the broader romantic preference. PDG-A therefore removes $\boldsymbol { e } _ { t } ~ = ~ ( w , u )$ , retaining the refinement path $u  v  w$

Once the graph is acyclic, we apply transitive reduction. If an indirect path $u  \cdot \cdot \cdot  v$ already exists, the direct shortcut edge $( u , v )$ is removed, eliminating transitive redundancy while preserving reachability and direct refinement dependencies. This yields the final edge set $E _ { q }$ and a valid topological ordering $\pi _ { q }$ satisfying $\pi _ { q } ( u ) < \pi _ { q } ( v )$ for all $( u , v ) \in E _ { q } ,$ , providing a dependency consistent order for subsequent preference propagation.

3.2.4 Preference: Contextual Refinement. The initial node representation $c _ { v } ^ { 0 }$ specifies a concrete direction for interpreting a preference, while the dependency structure determines its meaning within the compositional context. PDG-A completes node representations in topological order $\pi _ { q } ,$ using upstream preferences as context and incorporating the current node’s qualification to generate a concrete preference description.

For node $v ,$ let $\operatorname { A n c } ( v )$ denote its ancestor set. PDG-A jointly considers the complete narrative $q ,$ the current representation $c _ { v } ^ { 0 } ,$ the preference descriptions already generated for ancestor nodes in Anc(�), and the dependencies among these nodes to generate $f _ { v } .$ This description fills the reserved field, yielding $c _ { v } = c _ { v } ^ { 0 } $ [feature ← $f _ { v } ]$ . The topological order ensures that all ancestor descriptions are determined prior to generating $f _ { v } .$ For nodes without ancestors, the description is determined by their own initial representations and the original narrative.

For example, node � represents $^ { * 6 } \mathbf { a }$ concealed motive to evade accountability behind an appearance of innocence,” while node � represents $^ { * 6 } \mathrm { a }$ lawyer’s loyalty to a client and the associated moral dilemma.” Along $u  v ,$ PDG-A refines the preference at � into $f _ { v } = \mathrm { ^ { \ast } t h e }$ lawyer’s duty to protect a client may instead enable the client to evade accountability.” The upstream concealed motive thus reshapes the interpretation of professional loyalty, explicitly manifesting the moral conflict between professional responsibility and its actual consequences.

When multiple dependency paths converge at a node, PDG-A synthesizes the path structure and the original narrative to capture the corresponding compositional or alternative relations within a single $f _ { v } .$ The resulting preference descriptions across all nodes yield the complete preference dependency graph:

$$
\begin{array} { r } { G _ { q } = ( V _ { q } , E _ { q } , \mathcal { F } _ { q } ) , \qquad \mathcal { F } _ { q } = \{ f _ { v } \} _ { v \in V _ { q } } , } \end{array}\tag{7}
$$

providing concrete preference criteria for subsequent retrieval and candidate evaluation.

## 3.3 Directed Path Retrieval

3.3.1 Rewriter. Preference dependencies represent user needs as structural relationships between nodes, whereas Web retrieval requires concrete search expressions capable of capturing these compositional qualifications. Accordingly, for each retained dependency edge $( u , v ) \in E _ { q } ,$ PDG-A jointly examines the complete narrative $q$ and the initial endpoint representations $c _ { u } ^ { 0 } , c _ { v } ^ { 0 }$ to interpret the base preference under the qualification imposed by the target node, reformulating it into an efective search query $r _ { u v }$ following Google Search query syntax. Each query aligns with the language of the user narrative and translates the qualification relationship into targeted keywords and search operators, steering retrieval toward the composed preference semantics.

For a node � without predecessors, its search intent is governed independently by the narrative $q$ and its own initial representation $c _ { \upsilon } ^ { 0 } .$ To unify query formulation, we introduce a virtual root node ROOT and generate an entry query �<sub>ROO</sub> $_ { \mathrm { ~ T ~ } , v }$ for each such predecessor-free node. Let $E _ { q } ^ { + }$ denote the edge set augmented with these virtual entry edges. The resulting queries collectively constitute the retrieval query set:

$$
Q _ { q } = \{ r _ { u v } \ | \ ( u , v ) \in E _ { q } ^ { + } \} .\tag{8}
$$

3.3.2 Query. We execute the query set $Q _ { q }$ through the Serper API and collect URLs from the returned organic search results. We filter these URLs by resource type, removing invalid $\mathrm { { U R L s } , }$ social media websites, PDF documents, videos, and other excluded resource types.

Let $U ( r _ { u v } )$ denote the filtered URL set returned for query $r _ { u v } .$ Through the dependency edge $( u , v ) \in E _ { q } ^ { + }$ , these URLs are associated with the target node �, providing sources for subsequent candidate recall under its specific preference. Each node aggregates and deduplicates the URLs returned by all its incoming-edge queries:

$$
\mathcal { U } _ { v } = \bigcup _ { u : ( u , v ) \in E _ { q } ^ { + } } U ( r _ { u v } ) .\tag{9}
$$

The same URL may be associated with multiple nodes, allowing candidate items and relevant descriptions to be extracted separately according to each node’s preference.

3.3.3 Recall. For the URL set ${ { \mathcal { U } } _ { v } }$ associated with node �, we fetch the webpage text, resolve redirection targets, and inspect MIME resource types, pruning failed requests, empty bodies, and exceptionally large pages exceeding 200,000 characters. For each node, we then select at most the top 3 pages sorted by text length in descending order, and segment the contents into chunks of up to 12,000 characters with a 2,000-character overlap between adjacent passages, forming the text corpus $\mathcal { B } _ { v }$ . When identical URLs are shared across multiple nodes, the parsed text and chunks are reused while preserving their individual node assignments.

For each chunk $b \in \mathcal { B } _ { v }$ , the model extracts candidate items � along with their associated textual evidence $x _ { v , i } ^ { ( b ) }$ , guided by the node’s refined preference description $f _ { v } .$ Each extracted entry is formalized as ⟨name, year, $x _ { v , i } ^ { ( b ) } \rangle$ , recording the item name, release year if explicitly mentioned (or empty otherwise), and concrete descriptions pertinent to the active preference. A candidate record is formed only when a chunk simultaneously provides both unambiguous item identification and substantive content. Crucially, the extraction faithfully retains statements whether they support or contradict the preference, decoupling evidence extraction from subsequent preference verification. Because extraction is strictly conditioned on node $v ,$ extracted candidates and their evidence naturally inherit node provenance, allowing a single text chunk to yield distinct records under diferent node specifications $f _ { v } .$

Following extraction, we identify and exclude reference items already mentioned in the initial request � using name and year metadata, and then deduplicate candidates within each node. For the same item, identical descriptions are removed, and distinct descriptions are sequentially concatenated into a consolidated text $x _ { v , i }$ , with their source URLs retained, yielding the node candidate set $C _ { v }$ . Cross-node descriptions are maintained independently to ensure that the factual evidence for an item across diferent preference facets remains unentangled. This produces the global candidate set and the node-grounded evidence set for the narrative request:

$$
C _ { q } = \bigcup _ { v \in V _ { q } } C _ { v } , \qquad \mathcal { E } _ { q } = \left\{ ( v , i , x _ { v , i } ) \ \big | \ v \in V _ { q } , \ i \in C _ { v } \right\} ,\tag{10}
$$

where $x _ { v , i }$ encapsulates the web-grounded descriptions aligned with preference $f _ { v } ,$ providing fine-grained evidence for candidate evaluation along the preference dependency paths.

## 3.4 Topological Path Ranking

3.4.1 Initial Scoring. We collect the preference descriptions of the current node � and all its ancestors into ${ \mathcal { F } } _ { v } = \{ f _ { u } \ | \ u \in { \mathrm { A n c } } ( v ) \cup$ $\{ v \} \}$ , where Anc(�) denotes the ancestor set of $v ,$ and $f _ { u }$ is the concrete preference description of node �. For each item � recalled at node $v ,$ we hold its retrieved description $x _ { v , i }$ fixed and compare it against every preference description $f _ { u } \in \mathcal { F } _ { v }$ , assessing how the same textual evidence supports the active and upstream preferences.

PDG-A assigns a relation label based solely on the text pair $( x _ { v , i } , f _ { u } )$ , which the system then maps to an initial score:

$$
s ( x _ { v , i } , f _ { u } ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { e x p l i c i t ~ c o n t r a d i c t i o n , } } \\ { 1 , } & { \mathrm { w e a k ~ b u t ~ m e a n i n g f u l ~ r e l a t i o n , } } \\ { 2 , } & { \mathrm { r e l a t i o n ~ t h r o u g h ~ r e a s o n a b l e ~ i n f e r e n c e , } } \\ { 3 , } & { \mathrm { e x p l i c i t ~ s u p p o r t , } } \\ { 4 , } & { \mathrm { p a r a p h r a s e . } } \end{array} \right.\tag{11}
$$

The judgment respects the qualification scope, preference strength, and permitted alternatives expressed in $f _ { u } .$ . Missing details alone do not establish an explicit contradiction, and permitted alternatives need not hold simultaneously. An explicit contradiction assigns zero only to the current pairwise comparison, without removing the candidate or altering its other comparison scores. These initial scores provide the foundation for subsequent dependency-path computation.

3.4.2 Topological Path Computation. The initial scores quantify how an item description supports individual node preferences. We further aggregate these scores along dependency paths to assess how the same description supports progressively refined preferences. For an item � recalled at node $v ,$ we hold its retrieved description $x _ { v , i }$ fixed and traverse the subgraph induced by node � and all its ancestors $\operatorname { A n c } ( v )$ according to the topological ordering $\pi _ { q } .$ . Let $\mathrm { P a } ( u ) = \{ p \mid ( p , u ) \in E _ { q } \}$ denote the immediate predecessor set of node �, and let $H ( x _ { v , i } , u )$ denote the optimal path score ending at $u ,$ computed using this fixed description. The recurrence is defined as:

$$
H ( x _ { v , i } , u ) = \left\{ \begin{array} { l l } { s ( x _ { v , i } , f _ { u } ) , } & { \mathrm { P a } ( u ) = \emptyset , } \\ { s ( x _ { v , i } , f _ { u } ) + \gamma \displaystyle \operatorname* { m a x } _ { p \in \mathrm { P a } ( u ) } H ( x _ { v , i } , p ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{12}
$$

where $s ( x _ { v , i } , f _ { u } )$ is the initial score defined in Eq. (11), and $\gamma \in ( 0 , 1 ]$ is a discount factor (set to 0.7 in our implementation). A node without predecessors initializes its path score with its own initial score. For a node with predecessors, its initial score is combined with the discounted optimal path score from its predecessors. The topological order guarantees that all predecessor path scores are determined before evaluating node �.

When multiple predecessors converge at a node, the maximum operator isolates the highest-scoring refinement path. Along this trajectory, each additional edge traced upstream introduces an extra factor of $\dot { \gamma } ,$ exponentially decaying the contribution of more distant ancestors while prioritizing preferences proximal to the current node. The recurrence consistently evaluates the identical evidence $x _ { v , i }$ and ultimately yields $H ( x _ { v , i } , v )$ , the path score for item � under node �. The virtual root used for retrieval is excluded from scoring.

The same candidate item may be recalled across multiple nodes with distinct node-specific descriptions. Let ${ \mathcal { V } } _ { i } = \{ v \in V _ { q } \mid i \in C _ { v } \}$ denote the set of nodes where item � is retained. We aggregate its path scores across these nodes and apply a logarithmic compression to obtain the final graph-grounded score:

$$
S _ { \mathrm { g r a p h } } ( i \mid q ) = \log \left( 1 + \sum _ { v \in \mathcal { V } _ { i } } H ( x _ { v , i } , v ) \right) .\tag{13}
$$

Each node evaluates its path score independently using its own extracted description, and evidence from multiple webpages mapped to the same node is consolidated to contribute only once. An ancestor preference participating in the path computation of a descendant candidate does not spuriously introduce an extra candidate record at that ancestor node.

## 4 Experiment

## 4.1 Experimental Setup

Datasets. Following the experimental setup of OCG-Agent [28], we first adopt Reddit MovieSuggestions, a widely used benchmark for narrative-driven recommendation [7–10, 28]. We seek to further validate the capability of PDG-F across diverse tasks. Given that the proprietary education [28] dataset is not publicly accessible, we instead collect two datasets from public online communities, namely Book with 30 English queries from Reddit and Anime with 30 Chinese queries from Xiaohongshu and Zhihu.

Table 1: Main comparison results. Best values are bold; second-best values are underlined. For ties, only the last occurrence in each column is marked.
<table><tr><td>Method</td><td colspan="3">Movie</td><td colspan="3">Book</td><td colspan="3">Anime</td></tr><tr><td></td><td>P@10</td><td>R@10</td><td>NDCG@10</td><td>P@10</td><td>R@10</td><td>NDCG@10</td><td>P@10</td><td>R@10</td><td>NDCG@10</td></tr><tr><td>qwen3.8-27B</td><td>0.2730</td><td>0.1645</td><td>0.3280</td><td>0.1100</td><td>0.0989</td><td>0.1492</td><td>0.0267</td><td>0.0383</td><td>0.0429</td></tr><tr><td>minimax-m3</td><td>0.3070</td><td>0.1885</td><td>0.3647</td><td>0.1900</td><td>0.1589</td><td>0.2606</td><td>0.0633</td><td>0.0791</td><td>0.0979</td></tr><tr><td>deepseek-v4-pro</td><td>0.3170</td><td>0.1954</td><td>0.3753</td><td>0.1867</td><td>0.1648</td><td>0.2526</td><td>0.0567</td><td>0.0736</td><td>0.0884</td></tr><tr><td>OCG-Agent</td><td>0.2410</td><td>0.1448</td><td>0.2647</td><td>0.2200</td><td>0.2299</td><td>0.3264</td><td>0.1133</td><td>0.1356</td><td>0.1901</td></tr><tr><td>OCG-Agent (+Year)</td><td>0.2880</td><td>0.1693</td><td>0.3284</td><td>0.2200</td><td>0.2299</td><td>0.3264</td><td>0.1167</td><td>0.1389</td><td>0.1922</td></tr><tr><td>Perplexity AI</td><td>0.2700</td><td>0.1656</td><td>0.3206</td><td>0.1733</td><td>0.1396</td><td>0.2182</td><td>0.1667</td><td>0.1692</td><td>0.2028</td></tr><tr><td>GPT-6-Astra (Codex)</td><td>0.3210</td><td>0.1938</td><td>0.3771</td><td>0.1800</td><td>0.1410</td><td>0.2078</td><td>0.1467</td><td>0.1411</td><td>0.1888</td></tr><tr><td>Ours</td><td>0.3010</td><td>0.1865</td><td>0.3631</td><td>0.4233</td><td>0.3480</td><td>0.5101</td><td>0.2467</td><td>0.2722</td><td>0.3025</td></tr><tr><td>Ours (+Year)</td><td>0.3650</td><td>0.2229</td><td>0.4511</td><td>0.4233</td><td>0.3480</td><td>0.5101</td><td>0.2467</td><td>0.2722</td><td>0.3085</td></tr></table>

We adapt the curation methodology of Reddit MovieSuggestions [10] to construct these two benchmarks. For each sample, the input query is obtained by faithfully summarizing an informative recommendation post while preserving user preferences, constraints, and contextual details. The ground-truth items are scored based on community support, defined by the peak engagement achieved across recommending replies, namely the net vote score on Reddit or the like count on Xiaohongshu and Zhihu. When reliable engagement scores are unavailable for a query, we quantify community support by recommendation frequency within the collected replies.

Evaluation. We evaluate recommendation performance using Precision@10, Recall@10, and NDCG@10 [15, 22]. For each query, we follow the original setting of OCG-Agent [28] by performing retrieval once and running RankGPT reranking [29] three times, reporting the averaged results. For our method, PDG-F, ranking scores are computed deterministically from the constructed graph and collected evidence, so a single ranking run is suficient.

Baselines. Given the continuing improvements in LLM capabili ties, we rerun experiments with frontier models, including three leading Chinese models, Qwen-3.8-27B, DeepSeek-V4-Pro, and MiniMax-M3, as well as GPT-6-Astra, a leading international model. We also include Perplexity, a widely used commercial AI search engine. We set the reasoning efort to high for all models except GPT-6-Astra, which uses xhigh. For OCG-Agent [28], we reproduce its pipeline using Qwen-3.8-27B as the LLM backbone. Our method, PDG-F, uses the same Qwen-3.8-27B backbone, which we deploy locally on four H20 GPUs with 140 GB of memory each. Since the remaining model API calls and search services incur monetary costs, we follow OCG-Agent [28] and randomly sample 100 Movie queries for evaluation.

## 4.2 Result Analysis

Overall Performance. Table 1 summarizes the performance comparison across the three benchmark datasets. Overall, our proposed PDG-F (+Year) achieves the best performance on all three datasets across all evaluation metrics. Specifically, on the Movie dataset, PDG-F (+Year) outperforms the strongest competing baselines by 13.7%, 14.1%, and 19.6% in P@10, R@10, and NDCG@10, respectively. The gains are even larger on the Book and Anime benchmarks, where PDG-F (+Year) achieves a 92.4% improvement in P@10 on Book and a 60.9% gain in R@10 on Anime over the best competing methods. These consistent improvements across diverse domains substantiate the value of explicitly modeling preference dependencies and integrating supporting evidence for narrativedriven recommendation.

Further Analysis. The performance disparities across domains and model variants may partly reflect temporal retrieval biases and diferences in knowledge coverage. First, web search often favors recent and popular content, whereas the Movie benchmark frequently targets historical or classic titles. Accordingly, the (+Year) variant applies rule-based temporal comparison and filtering using the extracted release year metadata to exclude candidates that violate temporal constraints. Incorporating year information helps counter this mismatch by reducing the prominence of newer candidates, yielding substantial gains on Movie while gains on Anime remain more limited. Second, classic movie plots and item profiles are likely well represented in LLM pretraining corpora, helping direct prompting remain competitive on Movie. Newer or less documented content in Book and Anime may be less well covered by parametric knowledge, making external retrieval more valuable in these domains.

## 5 Ablation Study

## 5.1 Candidate Recall Analysis

We compare the candidate acquisition of PDG-F and OCG-Agent across overall pool coverage, relevant item overlap, and rankinglevel Recall@�.

5.1.1 Candidate Coverage. Table 2 compares candidate acquisition against the strongest retrieval baseline, OCG-Agent. Across the three datasets, PDG-F constructs substantially broader candidate pools, averaging roughly 250 to 690 items per request, compared with around 20 for OCG-Agent. Crucially, this expansion does not merely introduce noise, as pool recall approximately doubles or more across the three domains, increasing from 0.2102 to 0.4858 on Movie, from 0.2854 to 0.5481 on Book, and nearly tripling from 0.1672 to 0.4734 on Anime. This confirms that decom posing narrative preferences across dependency nodes efectively retrieves diverse, high-utility ground-truth items, establishing a much stronger candidate foundation for subsequent topological ranking. Importantly, because PDG-F scores candidates through deterministic graph accumulation rather than prompting-based listwise rerankers such as RankGPT, such candidate pool expansion avoids the pitfalls of context window limits [25], position bias [30], and prohibitive inference latency.

![](images/6de38ea402e5fe1a5310c100773ea6dae19fd123ecffffb1ed7fc21f9675f7c7.jpg)  
Figure 2: Recall at cutofs � ∈ [1, 50]. Shaded bands denote 95% bootstrap confidence intervals. The dashed PDG-F curves largely overlap with the solid PDG-F (+Year) curves on Book and Anime.

Table 2: Candidate-pool size and relevant-item coverage.
<table><tr><td>Dataset</td><td>Method</td><td>Avg. candidates</td><td>Pool recall</td></tr><tr><td rowspan="2">Movie</td><td>PDG-F (+Year)</td><td>251.84</td><td>0.4858</td></tr><tr><td>OCG-Agent</td><td>22.52</td><td>0.2102</td></tr><tr><td rowspan="2">Book</td><td>PDG-F (+Year)</td><td>687.33</td><td>0.5481</td></tr><tr><td>OCG-Agent</td><td>17.63</td><td>0.2854</td></tr><tr><td rowspan="2">Anime</td><td>PDG-F (+Year)</td><td>366.30</td><td>0.4734</td></tr><tr><td>OCG-Agent</td><td>17.33</td><td>0.1672</td></tr></table>

5.1.2 Recall across Recommendation Lengths. Figure 2 illustrates how broader candidate acquisition translates into ranking recall across varying recommendation lengths. As � grows from 1 to 50, the recall trajectories exhibit markedly diferent saturation behaviors. Due to its constrained candidate pool, OCG-Agent rapidly plateaus around � = 20 across all datasets, yielding minimal gains as the list extends further. In contrast, PDG-F with year maintains a continuous upward trajectory, consistently surfacing additional relevant items throughout longer lists. Remarkably, its Recall@10 alone surpasses the entire complete-pool recall of OCG-Agent across all three benchmarks. On Anime, OCG-Agent exhibits a slight advantage at the extreme top position, with a Recall@1 of

0.0492 versus 0.0359 for PDG-F with year, yet PDG-F rapidly overtakes it as � reaches 10 and beyond. This demonstrates that the candidate coverage advantages of PDG-F become increasingly pronounced when presenting multi-item recommendation lists.

5.1.3 Complementarity ofRetrieved Items. Figure 3 further examines where the additional coverage comes from. Relevant items found exclusively by PDG-F account for 30.41%, 33.57%, and 38.85% of all ground-truth items on Movie, Book, and Anime, respectively, while those found exclusively by OCG-Agent account for 2.86%, 7.29%, and 8.24%. These proportions are computed per query and then macro-averaged. Thus, PDG-F’s coverage advantage includes a substantial number of relevant works absent from the baseline pools, illustrating the additional coverage obtained from candidate acquisition under diferent preference perspectives. OCG-Agent also retrieves some relevant items missed by PDG-F, indicating that the candidate sources remain partially complementary.

![](images/a753749892842d3bc8173f7e063353d460fb60b02aebd7630126443e015edb5e.jpg)  
Figure 3: Ground truth item overlap between candidate pools.

## 5.2 Ranking Analysis

We analyze node aggregation strategies, score discriminability, oracle performance gaps, and the role of dependency modeling.

Table 3: Evaluation of candidate ranking strategies.
<table><tr><td>Strategy</td><td>Movie</td><td>Book</td><td>Anime</td></tr><tr><td>Random</td><td>0.0641</td><td>0.0163</td><td>0.0242</td></tr><tr><td>Mean node</td><td>0.1506</td><td>0.0178</td><td>0.0667</td></tr><tr><td>Max node</td><td>0.3205</td><td>0.1997</td><td>0.1255</td></tr><tr><td>Full strategy</td><td>0.4511</td><td>0.5101</td><td>0.3085</td></tr></table>

5.2.1 Ranking Strategy Comparison. Table 3 compares node aggregation strategies on identical candidate pools against a random ranking baseline. Mean aggregation degrades severely across all datasets because averaging dilutes decisive signals across nodes. Taking the maximum node score provides a viable baseline, con firming that prominent local matches ofer useful discrimination. Notably, this maximum node score already benefits from our propagation strategy by inheriting contextual preference constraints from ancestor nodes. Even with this contextual advantage, the full additive strategy consistently achieves higher NDCG@10 across all domains, and even more notably, it more than doubles the performance on Book and Anime, demonstrating the efectiveness of our cumulative evidence design.

![](images/3af287f041839932b76c9aafb6bf4d99c79f98a581a09b3a29dc88b897f26856.jpg)  
Figure 4: Relevance rates across score deciles.

5.2.2 Graph Scores and Relevance. Figure 4 evaluates the discriminative power of the computed graph scores within the expanded candidate pools. Partitioning candidates into score deciles reveals that ground-truth relevant items are heavily concentrated in the top decile, achieving relevance rates of 17.97%, 8.01%, and 7.92% on Movie, Book, and Anime, respectively, compared with only 1.87%, 0.10%, and 0.18% in the bottom decile. This sharp separation directly reflects the scoring quality because candidates assigned high graph scores are significantly more likely to be truly relevant, whereas irrelevant candidates are successfully suppressed into lower deciles. The cross-domain consistency of this trend confirms that the graph score functions as a reliable relevance indicator, validating that evaluating evidence along interdependent preference paths efectively captures whether an item satisfies the user’s composite needs.

![](images/123a2d38569e2acbcfef00200855e870804100f9e6c705e3b2a5aa514fe14296.jpg)

![](images/3b87d6b508c6e6c7dccdd5582740a82d56ab5f9e80bb995eabbca5aeefffb4b8.jpg)

![](images/4bbca209ea722387635d779c9515fc940012f8f075d45f72588248db88277a40.jpg)  
Figure 5: Decomposition of ideal NDCG@10 into pool and ranking gaps.

5.2.3 Candidate Acquisition and Ranking Gaps. We finally examine Figure 5 to decompose the gap to ideal NDCG@10 into candidatepool coverage and within-pool ranking accuracy. Across all domains, PDG-F achieves markedly higher pool-oracle upper bounds than OCG-Agent, confirming that multi-path acquisition builds a far stronger candidate foundation. At the same time, the visible ranking gaps highlight that while candidate retrieval raises the performance ceiling, ranking retrieved relevant items into top positions remains the primary bottleneck and leaves substantial room for future improvement.

5.2.4 Dependency Modeling Analysis. The empirical comparison between PDG-F and OCG-Agent naturally constitutes an ablation over dependency modeling. Specifically, our preference nodes refine the dynamic attribute representations of OCG-Agent through redesigned extraction sources and field specifications, while the overarching innovation lies in establishing directed qualification graphs. In the absence of directional dependency edges, candidate acquisition reduces to uncoordinated retrieval across isolated preference nodes. Furthermore, lacking the underlying graph structure precludes our topological path ranking, forcing candidate evaluation to revert to standard prompting-based reranking via RankGPT. Under this double degradation, the framework essentially degenerates into the baseline pipeline of OCG-Agent. The substantial performance margins across all evaluation domains, as shown in Table 1, therefore confirm that directional qualification modeling serves as the decisive driver for enhanced candidate acquisition and relevance discrimination.

## 6 Conclusion

We presented PDG-F, a framework for narrative-driven recommendation that captures how user preferences qualify one another through directed acyclic graphs. By carrying these qualification relationships into candidate acquisition and topological evidence accumulation, the framework connects narrative understanding with evidence-based recommendation. Experiments across three domains demonstrate broader relevant-item coverage and improved recommendation performance over competitive baselines. Oracle analysis further shows that this broader coverage raises the attainable performance ceiling, while the remaining ranking gap motivates more accurate evidence scoring to realize that potential.

## References

[1] Gediminas Adomavicius and Alexander Tuzhilin. 2005. Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions. IEEE Transactions on Knowledge and Data Engineering 17, 6 (2005), 734–749. doi:10.1109/TKDE.2005.99

[2] Jafar Afzali, Aleksander Mark Drzewiecki, and Krisztian Balog. 2021. POINTREC: A Test Collection for Narrative-driven Point of Interest Recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’21). Association for Computing Machinery, 2478–2484. doi:10.1145/3404835.3463243

[3] Matteo Attimonelli, Alessandro De Bellis, Claudio Pomo, Dietmar Jannach, Eugenio Di Sciascio, and Tommaso Di Noia. 2025. Do We Really Need Spe cialization? Evaluating Generalist Text Embeddings for Zero-Shot Recom mendation and Search. In Proceedings ofthe 19th ACM Conference on Recommender Systems (RecSys ’25). Association for Computing Machinery, 575–580. doi:10.1145/3705328.3748040

[4] Toine Bogers and Marijn Koolen. 2017. Defining and Supporting Narrativedriven Recommendation. In Proceedings of the Eleventh ACM Conference on Recommender Systems (RecSys ’17). Association for Computing Machinery, 238– 242. doi:10.1145/3109859.3109893

[5] Toine Bogers and Marijn Koolen. 2018. “I’m looking for something like ...”: Combining Narratives and Example Items for Narrative-driven Book Recommen dation. In Proceedings of the Workshop on Knowledge-aware and Conversational Recommender Systems (CEUR Workshop Proceedings, Vol. 2290). CEUR-WS.org, 35–43. https://ceur-ws.org/Vol-2290/kars2018\_paper7.pdf

[6] Arun Tejasvi Chaganty, Megan Leszczynski, Shu Zhang, Ravi Ganti, Krisztian Balog, and Filip Radlinski. 2023. Beyond Single Items: Exploring User Preferences in Item Sets with the Conversational Playlist Curation Dataset. In Proceedings ofthe 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’23). Association for Computing Machinery, 2754– 2764. doi:10.1145/3539618.3591881

[7] Lukas Eberhard, Kristina Popova, Simon Walk, and Denis Helic. 2024. Computing recommendations from free-form text. Expert Systems with Applications 236, Article 121268 (2024), 21 pages. doi:10.1016/j.eswa.2023.121268

[8] Lukas Eberhard, Thorsten Ruprechter, and Denis Helic. 2025. Large Language Models as Narrative-Driven Recommenders. In Proceedings ofthe ACM on Web Conference 2025 (WWW ’25). Association for Computing Machinery, 4543–4561. doi:10.1145/3696410.3714668

[9] Lukas Eberhard, Simon Walk, and Denis Helic. 2020. Tell Me What You Want: Embedding Narratives for Movie Recommendations. In Proceedings ofthe 31st ACM Conference on Hypertext and Social Media (HT ’20). Association for Computing Machinery, 301–306. doi:10.1145/3372923.3404818

[10] Lukas Eberhard, Simon Walk, Lisa Posch, and Denis Helic. 2019. Evaluating Narrative-Driven Movie Recommendations on Reddit. In Proceedings ofthe 24th International Conference on Intelligent User Interfaces (IUI ’19). Association for Computing Machinery, 1–11. doi:10.1145/3301275.3302287

[11] Elena V. Epure, Gabriel Meseguer Brocal, Darius Afchar, and Romain Hennequin. 2024. Harnessing High-Level Song Descriptors towards Natural Language-Based Music Recommendation. In Proceedings ofthe 3rd Workshop on NLPfor Music and Audio (NLP4MusA). Association for Computational Linguistics, Oakland, USA, 17–24. https://aclanthology.org/2024.nlp4musa-1.4/

[12] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). In Proceedings ofthe 16th ACM Conference on Recommender Systems. Association for Computing Machinery, 299–315. doi:10. 1145/3523227.3546767

[13] Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. 2023. Large Language Models as Zero-Shot Conversational Recommenders. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 720–730. doi:10.1145/3583780.3614949

[14] Dietmar Jannach, Ahtsham Manzoor, Wanling Cai, and Li Chen. 2021. A Survey on Conversational Recommender Systems. Comput. Surveys 54, 5, Article 105 (2021), 36 pages. doi:10.1145/3453154

[15] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated Gain-Based Evaluation of IR Techniques. ACM Transactions on Information Systems 20, 4 (2002), 422–446. doi:10.1145/582415.582418

[16] Ritu Rajesh Kanchi, Ruhul Amin Hazarika, and T. Gopalakrishnan Thirumoorthy. 2026. BookForYou: Leveraging BART-Generated Narrative Tropes for Content Based Book Recommendations. In Proceedings ofthe 20th ACM Conference on Recommender Systems (RecSys ’26). Association for Computing Machinery, 647– 652. doi:10.1145/3773078.3831814

[17] Jie Kang, Kyle Condif, Shuo Chang, Joseph A. Konstan, Loren Terveen, and F. Maxwell Harper. 2017. Understanding How People Use Natural Language to Ask for Recommendations. In Proceedings ofthe Eleventh ACM Conference on Recommender Systems. Association for Computing Machinery, 229–237. doi:10.

1145/3109859.3109873

[18] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 6769–6781. doi:10.18653/v1/2020.emnlp-main.550

[19] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (2009), 30–37. doi:10. 1109/MC.2009.263

[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 6b493230205f780e1bc26945df7481e5-Paper.pdf

[21] Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Lau rent Charlin, and Chris Pal. 2018. Towards Deep Conversational Recommendations. In Advances in Neural Information Processing Systems, Vol. 31. Cur ran Associates, Inc. https://proceedings.neurips.cc/paper\_files/paper/2018/file/ 800de15c79c8d840f4e78d3af937d4d4-Paper.pdf

[22] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008. Introduction to Information Retrieval. Cambridge University Press. doi:10.1017/ cbo9780511809071

[23] Sheshera Mysore, Andrew McCallum, and Hamed Zamani. 2023. Large Language Model Augmented Narrative Driven Recommendations. In Proceedings ofthe 17th ACM Conference on Recommender Systems (RecSys ’23). Association for Computing Machinery, 777–783. doi:10.1145/3604915.3608829

[24] Barbara H. Partee. 1984. Compositionality. In Varieties ofFormal Semantics, Fred Landman and Frank Veltman (Eds.). Foris, Dordrecht, 281–311.

[25] Zhen Qin, RolfJagerman, Kai Hui, Honglei Zhuang, Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Bendersky. 2024. Large Language Models are Efective Text Rankers with Pairwise Ranking Prompting. In Findings of the Association for Computational Linguistics: NAACL 2024. Association for Computational Linguistics, Mexico City, Mexico, 1504–1518. doi:10.18653/v1/2024.findings-naacl.97

[26] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings ofthe 2019Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, Hong Kong, China, 3982–3992. doi:10.18653/v1/D19-1410

[27] Scott Sanner, Krisztian Balog, Filip Radlinski, Ben Wedin, and Lucas Dixon. 2023. Large Language Models are Competitive Near Cold-start Recommenders for Language- and Item-based Preferences. In Proceedings ofthe 17th ACM Conference on Recommender Systems. Association for Computing Machinery, 890–896. doi:10. 1145/3604915.3608845

[28] Yunxiao Shi, Haoning Shang, Xing Zi, Wujiang Xu, Yue Feng, and Min Xu. 2025. Answering Narrative-Driven Recommendation Queries via a Retrieve– Rank Paradigm and the OCG-Agent. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Suzhou, China, 13181–13202. doi:10.18653/v1/2025.emnlp-main.667

[29] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investi gating Large Language Models as Re-Ranking Agents. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 14918–14937. doi:10.18653/v1/2023.emnlp-main.923

[30] Raphael Tang, Crystina Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Ture. 2024. Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, Mexico City, Mexico, 2327–2340. doi:10.18653/v1/2024.naacl-long. 129

[31] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, Hui Xiong, and Enhong Chen. 2024. A survey on large language models for recommendation. World Wide Web 27, 5, Article 60 (2024). doi:10.1007/s11280-024-01291-2

[32] Yaochen Zhu, Chao Wan, Harald Steck, Dawen Liang, Yesu Feng, Nathan Kallus, and Jundong Li. 2025. Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems. In Proceedings ofthe ACM on Web Conference 2025 (WWW ’25). Association for Computing Machinery, 3323–3334. doi:10.1145/3696410.3714908