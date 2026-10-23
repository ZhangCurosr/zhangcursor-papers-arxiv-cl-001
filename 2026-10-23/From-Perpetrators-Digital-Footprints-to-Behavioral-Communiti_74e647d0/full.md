# From Perpetrators’ Digital Footprints to Behavioral Communities: An Ontology-Driven Framework for Bengali Cyberbullying Analysis— FULL VERSION

Md. Nesarul Hoque<sup>a,b</sup>, Md Hanif Seddiqui<sup>a</sup>, Miskatul Anwar<sup>a</sup> and Rudra Pratap Deb Nath<sup>a,∗</sup>

<sup>a</sup>Big Data, Information and Knowledge Engineering Lab, Department ofComputer Science and Engineering, University of Chittagong, Chattogram, 4331, Bangladesh

<sup>b</sup>Department of Computer Science and Engineering, Gopalganj Science and Technology University, Gopalganj, 8105, Bangladesh

## A R T I C L E I N F O

Keywords:   
Cyberbullying   
Ontology   
Knowledge graph   
Behavioral modeling   
Network analysis   
Influence maximization.

## A BS T RA C T

Cyberbullying involves behavioral and contextual patterns beyond isolated abusive messages. However, existing approaches remain fragmented across content detection, semantic representation, and network analysis, with limited interpretable perpetrator representations, multi-scale structural modeling, and explanations of behavior-driven similarity and influence propagation. These limitations hinder the identification of persistent perpetrators, characterization of behavioral communities, and assessment of potentially influential users. This paper proposes Ontology-Driven Cyberbullying Behavior and Network Analysis (ODCBNA), a perpetrator-centric framework integrating requirementsdriven ontology design, knowledge graph construction, behavioral modeling, community analysis, and influence modeling. Using the Bengali Contextual and Diverse Cyberbullying (BenCoDiC) corpus as a use case, ODCBNA identifies 1,063 users with at least three bullying comments or replies as perpetrators and represents each through a 64-dimensional behavioral and contextual feature vector. Cosine similarity and �-nearest-neighbor modeling produce a weighted network with 27,554 edges. Louvain analysis identifies six communities with a modularity of approximately 0.495 and 27 sub-communities. Feature-contribution analysis identifies the Video, Impact, and User Profile groups as the strongest contributors to behavioral similarity and community separation, while contextual dimensions support interpretation. Community-constrained influence analysis shows that IM Greedy achieves small positive gains in all six communities, although diferent seed-selection methods produce comparable difusion spreads despite limited seed overlap. These findings provide an interpretable basis for understanding perpetrator behavior, community organization, and potential influence patterns in Bengali online environments.

## 1. Introduction

Cyberbullying is a critical concern on online social platforms because harmful online communication can lead to serious psychological and social consequences. Prior studies have associated cyberbullying victimization with depression, anxiety, suicidal ideation, and related adverse outcomes [22, 16]. A widely documented public example is the online harassment directed at Olympic boxer Imane Khelif during the Paris 2024 Olympics, where false claims and identity-based attacks spread across social media and led to a legal complaint for online harassment [1]. Such incidents illustrate that cyberbullying may appear through individual posts or comments, but the underlying behavior often involves repeated participation, recurring targets, and interaction with other users. Its meaning and severity may also depend on language, discussion context, topic, and social environment. Therefore, understanding cyberbullying requires attention not only to abusive content but also to the behavioral and contextual patterns surrounding abusive interactions [17, 32].

This need becomes especially clear in politically sensitive Bengali discussions on YouTube. A classifier may identify an abusive comment under a news video, but that decision says little about whether the same user repeatedly posts harmful content, targets similar individuals or groups, reacts to similar political contexts, or belongs to a cluster of users with comparable behavior. The comment is therefore only the visible trace of a wider digital footprint. A contentlevel approach can detect the harmful message, but it cannot explain how abusive users are behaviorally connected, how their activity varies across contexts, or which users may become structurally important within a community. This motivates a perpetrator-centric analysis that connects abusive content with user behavior, contextual exposure, community organization, and influence patterns.

Recent studies and surveys show that content-based cyberbullying detection has expanded across machine learning, deep learning, transformer-based models, feature engineering, multilingual datasets, and platform-specific settings [25, 20, 26, 31]. Despite these advances, most content-centered approaches remain focused on determining whether individual posts, comments, or messages contain abusive content. They therefore provide limited insight into persistent perpetrators, multiple bullying-related interactions, contextual exposure, or the position of abusive users within broader behavioral structures. Contextual ambiguity and variations in user behavior further limit approaches that rely primarily on message content [14, 17].

Alongside content-centered detection, ontology-driven and semantic representation methods organize heterogeneous social-media entities, attributes, and relationships within explicit domain models [15, 6]. Such representations provide a structured way to connect digital artifacts, user-related attributes, contextual information, and cyberviolence concepts within a consistent semantic framework. However, semantic representation alone does not reveal how abusive users are behaviorally similar, how they organize into communities, or how influence may propagate within those communities.

Graph- and network-aware studies address the relational side of cyberbullying and harmful online behavior. Usercentric analysis shows that hateful or abusive users may difer in activity patterns, vocabulary, and network structure [32]. Work on online trolling further shows that discussion context and exposure to prior harmful posts can increase subsequent antisocial behavior [5]. Recent cyberbullying studies also explore fine-grained role identification and GNN-based detection by incorporating behavioral features or graph-structured relations among text instances [35, 24]. These studies demonstrate the value of relational and behavioral signals, but they remain mainly oriented toward detection, role identification, or representation learning.

Overall, existing research remains divided among content analysis, semantic representation, and network modeling. As a result, cyberbullying is rarely examined through a single framework that connects meaning, perpetrator behavior, structural organization, and influence processes. This separation makes it dificult to explain how abusive users can be represented, compared, grouped, and examined for their potential influence. We identify four research gaps (RGs):

RG1 Fragmented modeling paradigms: Content classification, graph analysis, and semantic representation are usually studied separately [14, 32]. The lack of integration limits multi-level analysis of cyberbullying.

RG2 Lack of interpretable behavioral abstraction: Most methods rely on textual features or basic user statistics rather than interpretable, multi-dimensional representations of perpetrators. This limits the analysis of abusive activity across multiple interactions [17].

RG3 Limited multi-scale structural modeling: Previous studies examine user networks and interaction patterns, but similarity-based network construction and hierarchical community organization remain insuficiently explored [32]. Consequently, the structure of cyberbullying behavior across diferent scales is not well understood.

RG4 Lack of interpretable behavior-driven similarity and difusion analysis: Network-based studies indicate that abusive behavior may spread through interactions [5], but they rarely clarify how behavioral attributes contribute to user similarity or how different users influence propagation. This limits the interpretability of network findings and their use in targeted intervention.

To address these gaps, we propose Ontology-Driven Cyberbullying Behavior and Network Analysis (ODCBNA), a perpetrator-centric framework for YouTube cyberbullying analysis. Guided by analytical requirements and competency questions, ODCBNA defines a cyberbullying application ontology as the semantic foundation for knowledge graph construction, behavioral modeling, community analysis, and influence modeling. The framework is instantiated using the Bengali Contextual and Diverse Cyberbullying (BenCoDiC) corpus, which provides labeled YouTube political-news interactions and associated behavioral and contextual metadata [19]. By integrating these analytical components, OD-CBNA supports interpretable investigation of perpetrator behavior, community organization, and potential influence. The following research contributions (RCs) correspond directly to RG1–RG4, respectively:

RC1 Ontology-driven analytical framework: We develop an integrated pipeline connecting competency-questionguided ontology design, knowledge graph construction, heterogeneous graph transformation, and network analysis. This integration preserves semantic relationships while supporting behavioral, contextual, and structural analysis within a unified framework.

RC2 Perpetrator-centric behavioral–contextual representation: We construct an interpretable 64-dimensional representation covering User Profile, Video, Impact, Language, Target, Party, Time, and Channel. The representation enables consistent comparison among perpetrators and supports analysis of the behavioral and contextual dimensions associated with community structure.

RC3 Multi-scale network modeling: We construct a weighted behavioral-similarity network and identify six communities and 27 sub-communities. This hierarchical structure reveals perpetrator organization at multiple scales and supports finer analysis within the detected communities.

RC4 Interpretable similarity and influence analysis: We develop a behavioral contribution framework and a community-constrained influence maximization procedure. The results show that the Video, Impact, and User Profile groups provide the strongest contributions to similarity, while largely distinct seed sets achieve closely comparable difusion outcomes under the adopted IC settings.

Together, these contributions connect semantic representation, perpetrator behavior, community structure, and influence analysis, providing a broader account of cyberbullying beyond isolated content.

The rest of this paper is organized as follows. Section 2 reviews related studies, Section 3 describes the proposed framework, and Section 4 reports the experimental analysis and findings. Section 5 concludes the paper.

## 2. Related Work

Cyberbullying research broadly follows three directions: content-based detection, ontology-driven semantic modeling, and graph- or network-based analysis. Each direction contributes useful methods, but they are often examined separately. This separation limits eforts to connect the semantic, behavioral, and structural aspects of cyberbullying. This section reviews representative studies from these areas and clarifies the position of the proposed framework.

## 2.1. Cyberbullying Detection

Much of the existing research treats cyberbullying detection as a supervised text classification task. Recent surveys show that machine learning, deep learning, and transformer models are widely used and perform well on benchmark datasets [31, 26]. Hybrid architectures that combine CNNs, recurrent networks, transformer representations, attention, ensemble learning, and adversarial training have also improved accuracy, robustness, and cross-domain performance [28, 7, 37].

Research has also expanded to low-resource and multilingual settings. Atif et al. [2] use user-level signals for Roman Urdu cyberbullying detection, whereas Hoque et al. [18] develop a transformer-stacking framework for Bengali. Dataset-oriented studies have adopted broader definitions of cyberbullying by considering repetition, peerness, and intent to harm [13].

Even with this progress, most methods remain centered on content detection or user classification. They ofer limited evidence about how abusive users form behavioral communities, how their interaction patterns develop, or how cyberbullying operates as a networked social process.

## 2.2. Ontology-Driven Semantic Approaches

Another line of research uses ontology-driven modeling to organize heterogeneous social media data and represent domain knowledge. Ontologies define entities, relationships, and contextual information in a structured form, allowing information from diferent sources to be linked semantically [9].

Grigaliunas et al. [¯ 15] propose an ontology-driven digital profiling framework for linking social media evidence, while Chun et al. [6] develop a comprehensive ontology for cyber violence victimization, capturing participants, behavioral characteristics, and outcomes.

Although these approaches support semantic representation and structured profiling, their integration with perpetratorlevel behavioral similarity networks, hierarchical community analysis, and influence propagation remains limited.

## 2.3. Graph- and Network-Aware Cyberbullying Analysis

To address the relational nature of cyberbullying, recent work incorporates graph-based and network-aware modeling. Surveys highlight increasing use of interaction networks, contextual signals, and relational structures for cyberbullying analysis [26, 31].

Some studies move beyond content classification toward behavioral modeling. Wang et al. [35] propose a multilevel role identification framework that captures fine-grained user roles and their evolution. Other approaches explore distributed and privacy-preserving cyberbullying detection through transfer learning and federated learning frameworks [23]. More recently, graph neural network (GNN)- based approaches have been explored to capture interaction structures and contextual dependencies in cyberbullying data [24].

Although these approaches incorporate contextual, behavioral, or relational information, they primarily emphasize detection or role identification. Their network representations do not systematically combine perpetrator-level behavioral similarity, hierarchical community analysis, and influence propagation within a unified analytical framework.

Summary and Discussion. Existing studies have advanced cyberbullying detection, semantic representation, behavioral profiling, and relational modeling. However, these capabilities are generally addressed separately, with limited integration of interpretable perpetrator representations, multi-scale behavioral networks, and influence propagation analysis. Consequently, a unified approach connecting semantic knowledge, perpetrator behavior, structural organization, and difusion remains insuficiently explored.

Table 1 compares representative studies with ODCBNA using semantic, behavioral, structural, and analytical criteria. Behavioral representation corresponds to RG2, similarity and community modeling to RG3, and explainability and influence modeling to RG4. RG1 is reflected in the integration of these capabilities rather than a separate criterion. ODCBNA integrates competency-question-guided ontology design, knowledge graph construction, perpetratorlevel behavioral modeling, hierarchical community analysis, and interpretable similarity and influence modeling within a unified framework.

## 3. Methodology

This section presents the Ontology-Driven Cyberbullying Behavior and Network Analysis (ODCBNA) framework for analyzing cyberbullying from a perpetrator-centric perspective. The problem is to identify users exhibiting multiple bullying-labeled actions, characterize their observed behavioral patterns, examine their community organization, and model potential influence propagation. Guided by these analytical requirements and the corresponding competency questions (CQs), ODCBNA begins with the design of a cyberbullying application ontology that provides the semantic foundation for knowledge graph construction, behavioral modeling, community analysis, and influence modeling. Fig. 1 illustrates the overall workflow of ODCBNA. The framework is instantiated and evaluated using the Bengali Contextual and Diverse Cyberbullying (BenCoDiC) corpus as a use case [19].

## 3.1. Semantic Knowledge Representation

The Semantic Knowledge Representation stage establishes the semantic foundation for perpetrator-centric analysis through three components: Cyberbullying Application Ontology, Use-Case Dataset and Formalization, and Knowledge Graph Construction. First, a competency-questionguided application ontology defines the concepts, properties, and relationships required for the analysis. Next, the BenCoDiC corpus is introduced as a use case to formalize observed interactions and their associated behavioral and contextual information. Finally, the ontology is instantiated using the corpus records to construct a cyberbullying knowledge graph for subsequent behavioral modeling and network analysis.

## 3.1.1. Cyberbullying Application Ontology

Perpetrator-centric analysis of YouTube cyberbullying requires a structured representation of users, channel content, interactions, and their associated behavioral and contextual information. If these entities and attributes are processed as independent records or isolated features, their semantic relationships may be lost. For example, a bullying comment is not only a text instance. It is authored by a user, associated with channel content, expressed in a language, published at a time, and may target a particular identity or receive engagement responses. Therefore, an explicit semantic model is required to represent these entities and their relationships in a consistent and machine-processable form.

Comparative analysis of existing studies and ODCBNA across semantic, behavioral, structural, and analytical dimensions.
<table><tr><td rowspan="2">Ref.</td><td colspan="2">Semantic / Structural Basis</td><td rowspan="2">RG2</td><td colspan="2">RG3</td><td colspan="2">RG4</td></tr><tr><td>Ontology</td><td>Perp-Graph</td><td>Behav. Rep. Similarity</td><td>Community</td><td>Explainability</td><td>Influence</td></tr><tr><td>Ejaz et al. [13]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Harshitha et al. [28]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Cuzzocrea et al. [7]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Xiong et al. [37]</td><td>x</td><td>X</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Atif et al. [2]</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Hoque et al. [18]</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Wang et al. [35]</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Kumari and Kaur [23]</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Maity et al. [24]</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Grigaliūnas et al. [15]</td><td>√</td><td>x</td><td>√</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Chun et al. [6]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>x</td></tr><tr><td>Proposed ODCBNA framework</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Notes: Perp-Graph = Perpetrator-level graph; Behav. Rep. = User-level behavioral representation. Similarity includes explicit text- or user-level similarity modeling. Explainability includes model explanations, feature attribution, or interpretable semantic profiling. Influence refers to network difusion or influence modeling. ✓ indicates a reported capability, while ✗ indicates that the capability is not a principa component of the study.

![](images/cd990e972208ec5a1ffc2159ef825dc0324dd8a21e2656e94f012f99636181a5.jpg)  
Figure 1: Overview of the ODCBNA framework.

To meet this need, we define a domain-specific application ontology guided by the analytical requirements of ODCBNA. The ontology serves as a semantic schema for representing entities, attributes, and relationships relevant to perpetrator-centric cyberbullying analysis. Its explicit definitions preserve semantic relationships among users, interactions, textual content, channel content, and contextual information. These relationships can be represented through RDF triples, providing a consistent foundation for knowledge graph construction and subsequent behavioral and network analyses.

The analytical requirements are expressed through the following competency questions, which define the intended scope of the framework:

CQ1 Which users exhibit multiple bullying-labeled actions across interactions?

CQ2 What behavioral characteristics define diferent types of perpetrators?

CQ3 What structural patterns characterize the perpetrator similarity network at multiple scales?

CQ4 How are perpetrators structurally organized into communities and sub-communities based on behavioral similarity?

CQ5 How are behavioral patterns associated with contextual environments across communities?

CQ6 Which behavioral dimensions contribute most to similarity and community formation?

CQ7 Which users are most influential within each community under a difusion process?

CQ8 How does influence propagate across structurally similar perpetrators?

These CQs guide the selection of ontology classes and properties by identifying the information required for perpetrator-centric analysis. In particular, the ontology must represent user behavior, textual labels, interaction context, content exposure, target identity, discussion subjects, temporal information, and engagement impact. While the ontology establishes the semantic foundation, the subsequent behavioral and network analysis components address the computational aspects of these questions.

Fig. 2 shows the high-level structure of the proposed cyberbullying application ontology. The ontology combines a domain-specific vocabulary in the cb: namespace with selected terms from established vocabularies, including Schema.org (schema:) and RDF Schema (rdfs:). In semantic knowledge representation, a knowledge base is commonly described using two components: the TBox and the ABox [3]. The TBox, or terminological component, defines the domain schema in terms of concepts, properties, and their semantic relationships, whereas the ABox, or assertional component, contains instance-level assertions that instantiate the TBox. Accordingly, the proposed ontology defines the TBox of the cyberbullying knowledge graph, while observed data provide the corresponding ABox through RDF triples.

The central concept in the ontology is cb:InteractionConten which represents an online interaction such as a comment or reply. The cb:publishedBy property links each interaction to its authoring cb:User, while cb:belongsToVideo links it to the relevant cb:Video. Video uploader accounts are connected through cb:uploadedFrom. To represent channel-published content more generally, cb:ChannelContent is defined as a parent class of cb:Video through rdfs:subClassOf. This hierarchy supports extensibility to other YouTube content types while preserving the video-specific relationships used in the present analysis. These relationships also support the distinction between abusive participants and content producers during perpetrator-centric subgraph construction.

The textual content associated with an interaction is represented by cb:TextualContent, with its cyberbullying annotation stored through the cb:hasBullyLabel datatype property. This separation keeps the label attached to the textual content while preserving its broader interaction context. The label indicates whether bullying is present, while the associated interaction identifies who posted the content, where it appeared, and its surrounding context.

![](images/b442423b63f017d1fd5b4c61e7ca55dbb043a4f78eefdf6314811b7a6a893415.jpg)  
Figure 2: High-level structure of the cyberbullying application ontology (TBox), including subclass relationships and selected datatype properties.

The ontology also represents the contextual and behavioral dimensions required for downstream analysis. The schema:Language concept captures the linguistic form of an interaction, while cb:TargetIdentity represents the identity category targeted by the content. The cb:DiscussionSubject concept represents subjects associated with channel content through cb:relatedTo, with schema:PoliticalParty defined as its subclass to represent political discussion contexts. Temporal information is represented by cb:Time through cb:publishedAt, while audience response is represented by cb:ImpactScore through cb:impactScore. In addition, cb:Contex captures interaction-level contextual information and is linked to cb:InteractionContent through cb:context.

Together, these concepts and relationships provide a semantic foundation for organizing cyberbullying-related information and supporting subsequent behavioral modeling and network analysis.

## 3.1.2. Use-Case Dataset and Formalization

To instantiate and evaluate the proposed ODCBNA framework, we employ the Bengali Contextual and Diverse Cyberbullying (BenCoDiC) corpus as a use case [19]. BenCoDiC is a context-rich Bengali cyberbullying resource developed from political news discourse on YouTube. The corpus is particularly suitable for perpetrator-centric analysis because it provides persistent user-level information together with multidimensional behavioral and contextual metadata covering users, videos, textual content, interaction context, time, engagement, political-party context, target identities, and language categories. It comprises 50,250 textual instances associated with 100 YouTube videos and 37,813 user accounts. To protect user privacy, original user IDs, handlers, and channel titles are anonymized, while persistent anonymized identifiers preserve user-level continuity across interactions without exposing the original identities.

Let  denote the set of interaction instances,  the set of user accounts, V the set of videos, TC the set of textual content instances, ${ \mathcal { C } } _ { \mathrm { t x } }$ the set of contextual records,  the set of temporal records, IS the set of impact-score records, PP the set of political-party records, L the set of language records, and TI the set of target-identity records. In this use case, video and political-party records correspond to instances of cb:Video and schema:PoliticalParty, which are subclasses of cb:ChannelContent and cb:DiscussionSubject, respectively. Since each labeled textual instance is linked to its corresponding interaction and associated contextual records, the set of observed record combinations is represented as

$$
\mathcal { X } \subseteq T \times \mathcal { V } \times \mathcal { V } \times \mathcal { T } C \times C _ { \mathrm { t x } } \times \mathcal { T } \times T S \times \mathcal { P } \mathcal { P } \times \mathcal { L } \times T  { \boldsymbol { \mathrm { \Pi } } } .
$$

The binary label-assignment mapping is represented as

$$
f : \mathcal { X }  \{ b u l l y , n o t B u l l y \} .
$$

Here, $( i , u , v , t c , c , t , i s , p p , l , t i ) \in \mathcal { X }$ denotes the interaction, user, video, textual content, context, time, impactscore, political-party, language, and target-identity records, respectively. Accordingly, $f ( i , u , v , t c , c , t , i s , p p , l , t i )$ denotes the binary cyberbullying label associated with textual instance �� and its linked records. The annotation decision is based on the textual content together with its available conversational context, while the remaining dimensions preserve the associated information required for subsequent behavioral analysis. For later use, $\ell ( t c )$ denotes the label associated with textual instance �� through �.

The final BenCoDiC labels were obtained through a twotier human annotation process supported by sociolinguistguided training. Three annotators independently labeled each sample at Level-1, followed by verification by three NLP domain experts at Level-2. The final annotations achieved Fleiss’ $\kappa \ = \ 0 . 8 6$ and Krippendorf’s $\alpha ~ = ~ 0 . 8 6$ indicating strong inter-annotator agreement.

These linked records and annotations provide the instancelevel information required for ontology-guided knowledge graph construction, supporting subsequent perpetrator-centric behavioral representation, community analysis, and influence modeling.

## 3.1.3. Knowledge Graph Construction

Based on the application ontology introduced in Section 3.1.1, the BenCoDiC corpus is transformed into a cyberbullying knowledge graph. The ontology provides the TBox, i.e., the schema-level concepts and properties, while the corpus provides the observed entities, relationships, and literal values required to construct the ABox. Accordingly, this step generates corpus-derived ABox triples by mapping BenCoDiC records and metadata fields to the corresponding classes and properties defined in the TBox, using Resource Description Framework (RDF) representation.

Let �, �, and � denote the sets of Internationalized Resource Identifiers, blank nodes, and literals, respectively. An RDF triple is represented as $( s , p , o )$ , where $s \in ( \mathtt { I U B } )$ is a subject resource, $p \in \mathtt { I }$ is a predicate, and � $\in ( \mathrm { I } \cup \mathsf { B } \cup \mathsf { L } )$ is an object resource, blank node, or literal value [8]. Accordingly, the cyberbullying knowledge graph is represented as a finite set of RDF triples:

$$
\mathcal G _ { K G } \subseteq ( \mathbf { I } \cup \mathbf { B } ) \times \mathbf { I } \times ( \mathbf { I } \cup \mathbf { B } \cup \mathsf { L } ) .
$$

Within $\mathcal { G } _ { K G } .$ , the schema-level triples correspond to the TBox, while the corpus-derived instance-level triples constitute the ABox.

Fig. 3 illustrates the ontology-driven knowledge graph construction pipeline. The process starts with the cyberbullying dataset and the application ontology. The ontology provides the target semantic schema, while the dataset provides the observed entities, relationships, and literal values. Through ontology-guided mapping, corpus records are converted into RDF triples that instantiate the TBox [11, 10].

![](images/eb70462bb11ad5b0dd42619c5992012fe7b4d1c24148830cc2e983c28cb88a52.jpg)  
Figure 3: Ontology-driven knowledge graph construction pipeline.

In this mapping process, dataset entities such as users, videos, interaction content instances, textual content instances, languages, target identities, political contexts, temporal records, and impact scores are assigned resource identifiers and instantiated according to the corresponding TBox classes. Relationships among these entities are encoded using object properties defined in the TBox. For example, an interaction is linked to the user who published it, the video under which it appears, its textual content, its language, its target identity, its publication time, and its impactrelated information. Literal values such as textual content, timestamps, bullying labels, and engagement statistics are encoded using datatype properties.

The generated ABox triples, together with the RDF representation of the TBox, form the cyberbullying knowledge graph. This graph provides a unified multi-relational representation of cyberbullying interactions by explicitly linking abusive textual content with users, videos, contextual metadata, and engagement signals. The resulting knowledge graph serves as the semantic foundation for the subsequent RDF-to-heterogeneous-graph transformation and perpetrator-centric behavioral modeling.

## 3.2. Perpetrator Behavioral Modeling

This stage transforms the ontology-grounded knowledge graph into a perpetrator-centric behavioral representation suitable for network analysis. It includes heterogeneousgraph transformation, perpetrator-centric subgraph construction, behavioral and contextual feature construction, and similarity modeling to produce the weighted perpetrator similarity network.

## 3.2.1. RDF-to-Heterogeneous-Graph Transformation

While RDF provides a flexible semantic representation, downstream graph learning and network analysis require a computationally eficient heterogeneous graph with explicit typing and indexing [33]. Therefore, the RDF knowledge graph is transformed into a heterogeneous graph through an intermediate tabular representation consisting of typed node tables, typed edge tables, and type-specific identifier mappings.

Let $G _ { h } = ( \mathcal { N } _ { h } , \mathcal { E } _ { h } )$ denote the resulting heterogeneous graph, where $\mathcal { N } _ { h }$ is the set of typed nodes and $\mathcal { E } _ { h }$ is the set of typed edges. Each typed edge is represented as $( n _ { s } , n _ { t } , r ) \in$ $\mathcal { E } _ { h }$ , where $n _ { s } ~ \in ~ \mathcal { N } _ { h }$ is the source node, $n _ { t } \ \in \ \mathcal N _ { h }$ is the target node, and � denotes the relation type derived from the ontology.

The transformation proceeds in three stages. First, RDF triples are loaded from the Turtle files, and node instances are identified based on ontology classes and URI patterns. For each node type, relevant literal attributes and relational references are extracted and organized into node-specific tables.

Second, type-specific identifier mappings are constructed to convert URI-based entities into contiguous integer indices, enabling eficient storage and computation in graphbased frameworks.

Third, typed edge tables are generated by extracting relations defined in the ontology between compatible node types. These tables encode multi-relational connections while preserving the semantic structure of the original RDF graph.

The resulting heterogeneous graph includes node types such as User, Video, InteractionContent, TextualContent,

![](images/8140c23cbd93e797dab0f06c476ed4f4d1f8e612a56975f712b2dab5481a28ba.jpg)  
Figure 4: RDF-to-heterogeneous-graph transformation process.

Language, TargetIdentity, PoliticalParty, Context, Time, and ImpactScore, along with edge types such as publishedBy, belongsToVideo, hasInteractionContext, language, targetTo, publishedAt, impactScore, and relatedTo. Fig. 4 illustrates the overall transformation process.

This transformation bridges semantic knowledge representation and graph-based computation by converting ontology-grounded RDF data into a structured heterogeneous graph while preserving node heterogeneity and relation semantics. The resulting graph serves as the foundation for constructing the perpetrator-centric subgraph and subsequent behavioral modeling.

## 3.2.2. Perpetrator-Centric Subgraph Construction

The heterogeneous graph constructed in Section 3.2.1 represents all user accounts and interactions, including nonabusive behavior. However, the objective of this work is to analyze the structural organization of user-level patterns of cyberbullying activity. To this end, we induce a perpetratorcentric subgraph by restricting the full graph to users exhibiting multiple bullying-labeled actions and their associated interaction context.

A user is defined as a perpetrator if the user has posted at least three bullying comments or replies. Let  denote the set of user accounts and $ { \tau } _ { c r } ( u )$ denote the set of comment and reply textual instances authored by user �. Since each textual instance already has a cyberbullying label assigned through the context-aware annotation process defined in Section 3.1.2, the bullying action count of user $u \in \mathcal V$ is computed as

$$
b _ { u } = \left| \left\{ t c \in \mathcal { T } _ { c r } ( u ) \ : | \ : \ell ( t c ) = b u l l y \right\} \right| ,\tag{1}
$$

where $b _ { u }$ denotes the number of bullying comment or reply actions associated with user �. The set of perpetrators is then defined as

$$
\mathcal { V } _ { p } = \{ u \in \mathcal { V } \mid b _ { u } \geq \tau \} ,\tag{2}
$$

where the threshold is set to $\tau \ = \ 3$ . Repetition is recognized as an important characteristic of cyberbullying, although its conceptualization and operationalization vary across studies [30]. The threshold used here does not operationalize victim-specific repetition. Instead, it serves as a study-specific user-level activity criterion for identifying users who have produced multiple bullying-labeled comments or replies across the observed interactions. Accordingly, $\tau \ = \ 3$ is used to reduce the inclusion of users represented by only one or two bullying-labeled actions. This criterion is not intended as a universal definition of a cyberbullying perpetrator. Each element $p \in \mathcal { V } _ { p }$ is subsequently treated as a perpetrator user for behavioral representation, similarity modeling, community detection, and influence analysis.

Fig. 5 illustrates the perpetrator-centric subgraph induction process. Starting from the heterogeneous knowledge graph, bullying interactions are first filtered, users with at least � = 3 bullying comments or replies are identified as perpetrators, and the corresponding interaction context is retained to induce the perpetrator-centric graph.

The subgraph is induced through a three-step filtering and retention process. First, textual content instances are filtered to retain only comments and replies whose annotation is bully. Second, users associated with at least � such bullying actions are identified as perpetrators. Third, a heterogeneous subgraph is constructed by retaining: (i) InteractionContent nodes authored by perpetrators, (ii) linked TextualContent nodes restricted to comments and replies, (iii) associated Video nodes, and (iv) connected contextual nodes of types Language, TargetIdentity, Context, Time, ImpactScore, and PoliticalParty.

![](images/e22b0479eb4d7a71b7b73272b0c28a5374eb8b22438b9008ffe9d690136d974f.jpg)  
Figure 5: Perpetrator-centric subgraph induction from the heterogeneous knowledge graph.

Although commenters and video uploaders are represented by nodes of type User in the full heterogeneous graph, they are distinguished by role in the induced subgraph. Users who post bullying comments or replies are represented as PerpUser nodes, whereas uploader accounts associated with videos are represented as ChannelUser nodes. Let the resulting perpetrator-centric graph be denoted by $G _ { p } = ( \mathcal { N } _ { p } , \mathcal { E } _ { p } )$ where $\mathcal { N } _ { p }$ denotes the selected PerpUser, ChannelUser, InteractionContent, TextualContent, Video, and contextual nodes, and $\mathcal { E } _ { p }$ denotes the typed edges connecting these nodes.

This subgraph induction process reduces irrelevant structure and concentrates the representation on behaviorally meaningful interactions. By preserving heterogeneous node and relation types while filtering non-abusive activity, the resulting graph provides a focused and semantically consistent basis for subsequent behavioral feature construction and network analysis.

## 3.2.3. Behavioral and Contextual Feature

## Construction

The perpetrator-centric subgraph captures heterogeneous behavioral and contextual information associated with abusive users. However, similarity modeling and subsequent network analysis require a fixed-dimensional numerical representation. To address this, we construct a joint behavioral– contextual feature vector for each perpetrator through an aggregation and encoding process over the perpetratorcentric subgraph.

The feature construction process summarizes user attributes, interaction context, content exposure, engagement signals, temporal activity, and channel information into a unified representation. Aggregation is performed over all interaction-comment nodes associated with each perpetrator. Categorical attributes are accumulated as count vectors and normalized, while continuous attributes are summarized using statistical aggregation functions and discretized into comparable numerical groups.

For each perpetrator �, the resulting feature representation is defined as

$$
\begin{array} { r l } & { \mathbf { x } _ { p } = \big [ \mathbf { x } _ { p } ^ { ( \mathrm { p r o f } ) } , \mathbf { x } _ { p } ^ { ( \mathrm { v i d e o } ) } , \mathbf { x } _ { p } ^ { ( \mathrm { i m p a c t } ) } , \mathbf { x } _ { p } ^ { ( \mathrm { l a n g } ) } , } \\ & { \qquad \mathbf { x } _ { p } ^ { ( \mathrm { t a r g e t } ) } , \mathbf { x } _ { p } ^ { ( \mathrm { p a r t y } ) } , \mathbf { x } _ { p } ^ { ( \mathrm { t i m e } ) } , \mathbf { x } _ { p } ^ { ( \mathrm { c h a n n e l } ) } \big ] \in \mathbb { R } ^ { 6 4 } , } \end{array}\tag{3}
$$

where each component corresponds to a distinct behavioral or contextual dimension derived from the perpetrator-centric subgraph. The dimensionality of each feature group is determined from ontology-derived attributes that are consistently observable across perpetrators.

![](images/e70567a0e98502b7bf37765b26e34e36eaedcfd180030f539a4e7c6e095867b3.jpg)  
Figure 6: Behavioral and contextual feature construction over the perpetrator-centric subgraph.

Specifically, $\mathbf { x } _ { p } ^ { ( \mathrm { p r o f } ) }$ represents User Profile features, $\mathbf { x } _ { p } ^ { ( \mathrm { v i d e o } ) }$ represents Video features, $\mathbf { x } _ { p } ^ { ( \mathrm { i m p a c t } ) }$ represents Impact features, $\mathbf { x } _ { p } ^ { ( \mathrm { l a n g } ) }$ represents Language features, $\mathbf { x } _ { p } ^ { ( \mathrm { t a r g e t } ) }$ represents Target features, $\mathbf { x } _ { p } ^ { ( \mathrm { p a r t y } ) }$ represents Party features, $\mathbf { x } _ { p } ^ { ( \mathrm { t i m e } ) }$ represents Time features, and $\mathbf { x } _ { p } ^ { ( \mathrm { c h a n n e l } ) }$ represents Channel features.

As illustrated in Fig. 6, the final representation consists of 64 features organized into eight groups: 12 user-profile features, 12 video features, 5 language features, 6 targetidentity features, 6 political-context features, 6 temporal features, 9 impact features, and 8 channel-exposure features. For brevity, subsequent analyses and visualizations refer to these groups using the labels User Profile, Video, Impact, Language, Target, Party, Time, and Channel, respectively.

Categorical attributes are represented as normalized frequency distributions over observed interactions. Let $\mathbf { c } _ { u } \in \mathbb { R } ^ { d }$ denote a categorical count vector associated with perpetrator �. The normalized representation is computed as

$$
\hat { \mathbf { c } } _ { u } = \frac { \mathbf { c } _ { u } } { \Vert \mathbf { c } _ { u } \Vert _ { 2 } + \epsilon } ,\tag{4}
$$

where $\epsilon$ is a small constant introduced for numerical stability.

Continuous attributes, including subscriber count, account age, content popularity, and engagement statistics, are first transformed using

$$
x ^ { \prime } = \log _ { 1 0 } ( 1 + x ) ,\tag{5}
$$

to reduce heavy-tailed skewness commonly observed in social media data. The transformed values are subsequently encoded using a strict three-bin one-hot discretization scheme. The encoder first applies value-based tertile thresholds; if the thresholds collapse due to ties or zero inflation, a zero-aware fallback strategy separates zero-valued observations from positive values and partitions positive values using their median. This design guarantees that identical values are never assigned to diferent bins.

For interpretability, user-profile, video, and impact features are conceptually associated with activity intensity, content exposure, and engagement response, respectively. These abstractions are used in subsequent analyses to characterize behavioral patterns at the community and network levels.

Overall, the aggregation and encoding process transforms the heterogeneous perpetrator-centric subgraph into a compact and expressive joint behavioral–contextual representation. The resulting feature space supports scalable similarity computation, interpretable behavioral comparison, and subsequent multi-scale network analysis while preserving the heterogeneous nature of cyberbullying behavior and its surrounding contextual environment.

## 3.2.4. Perpetrator Similarity Modeling

To identify perpetrators with comparable behavioral signatures, we model pairwise similarity over the feature vectors defined in Eq. (3). Let $\textit { p } = \textit { v } _ { p }$ denote the set of perpetrator nodes. For perpetrator nodes $p , q \in { \mathcal { P } } .$ , cosine similarity is adopted because it captures directional alignment between feature vectors while reducing sensitivity to magnitude diferences [34]:

$$
\sin ( p , q ) = { \frac { \mathbf { x } _ { p } ^ { \mathsf { T } } \mathbf { x } _ { q } } { \| \mathbf { x } _ { p } \| _ { 2 } \left\| \mathbf { x } _ { q } \right\| _ { 2 } } } ,\tag{6}
$$

where $\mathbf { X } _ { p }$ and $\mathbf { x } _ { q }$ denote the behavioral representations of perpetrator nodes $p$ and $q ,$ respectively.

Cosine similarity is particularly suitable in this setting because the feature representation consists of heterogeneous blocks with diferent semantics and scales. The objective is to capture similarity in behavioral patterns rather than absolute activity magnitude, allowing perpetrators with diferent activity levels to be compared based on relative behavioral structure.

Directed kNN Similarity Graph: Computing similarity for all pairs of perpetrators would result in a dense graph that is not suitable for network analysis. To obtain a sparse and structurally meaningful representation, we construct a directed �-nearest-neighbor (kNN) graph. To avoid ambiguity with the influence seed budget used later, the nearestneighbor parameter is denoted as $k _ { \mathrm { N N } } .$ . For each perpetrator node �, the $\mathrm { t o p } { - } k _ { \mathrm { N N } }$ most similar neighbors are retained, forming the set $\mathcal { N } _ { k _ { \mathrm { N N } } } ( p )$

This step yields a directed weighted graph, since the $\mathrm { t o p } { - } k _ { \mathrm { N N } }$ neighbor relation is not necessarily symmetric even though cosine similarity itself is symmetric. Each directed edge $( p , q )$ indicates that $q \in \mathcal { N } _ { k _ { \mathrm { N N } } } ( p )$ , and the associated edge weight is given by $\sin ( p , q )$ . The kNN construction enforces locality by preserving only the strongest behavioral relations, thereby preventing the graph from becoming fully connected.

Symmetrized Weighted Similarity Network: Although cosine similarity is symmetric, the kNN construction yields a directed graph because the neighbor relation is not necessarily reciprocal. We therefore convert the directed kNN graph into an undirected weighted similarity network by averaging reciprocal edge weights when both directions exist and retaining the available weight otherwise.

The resulting similarity network is denoted as $\begin{array} { l l } { G _ { s } } & { = } \end{array}$ $( \mathcal { P } , E _ { s } , W _ { s } )$ , where $_ { \mathcal { P } }$ represents the set of perpetrator nodes, $E _ { s }$ denotes the set of undirected similarity edges, and $W _ { s }$ denotes the associated behavioral-similarity weights. This network preserves local similarity neighborhoods while ensuring compatibility with subsequent community detection, structural analysis, and difusion modeling.

## 3.3. Behavioral Community and Influence

## Analysis

This stage analyzes the perpetrator similarity network to identify multi-scale behavioral organization, interpret community-level similarity patterns, and examine influence propagation. It includes community and sub-community detection, behavioral profiling and contribution analysis, and community-constrained influence maximization.

## 3.3.1. Community Detection

The perpetrator similarity network $G _ { s }$ is expected to contain groups of perpetrator nodes with internally coherent behavioral patterns. To identify such groups, we apply modularity-based community detection using the Louvain algorithm [4].

The modularity objective is defined as [27]

$$
Q = \frac { 1 } { 2 m } \sum _ { p , q \in \cal P } \left[ w _ { p q } - \frac { d _ { p } d _ { q } } { 2 m } \right] \delta ( c ( p ) , c ( q ) ) ,\tag{7}
$$

where $w _ { p q }$ denotes the similarity weight between perpetrator nodes $p$ and $\begin{array} { r } { q , d _ { p } = \sum _ { q \in \mathcal { P } } w _ { p q } } \end{array}$ is the weighted degree of node $\begin{array} { r } { p , m = \frac { 1 } { 2 } \sum _ { p , q \in \mathcal { P } } w _ { p q } } \end{array}$ is the total similarity weight in the network, and $\delta ( c ( p ) , c ( q ) )$ is an indicator function equal to 1 if � and $q$ are assigned to the same community and 0 otherwise.

The Louvain algorithm is selected due to its scalability on weighted networks and its efectiveness in identifying meso-scale structures. The resulting partition is denoted as $C = \{ C _ { 1 } , C _ { 2 } , \dots , C _ { K } \}$ , where � is the number of detected communities and each $C _ { r } \subseteq { \mathcal { P } }$ , for $r = 1 , \ldots , K$ , represents a group of perpetrators with similar behavioral characteristics.

This step transforms pairwise behavioral similarity into an interpretable meso-scale organization of the perpetrator network. The detected communities serve as the primary structural units for subsequent behavioral profiling and community-constrained difusion analysis.

Community detection identifies broad behavioral groups, but it may overlook variations within them. The next subsection addresses this issue by refining each community into sub-communities.

## 3.3.2. Sub-community Detection

To reveal finer behavioral structures, community detection is repeated within each previously identified community. For every $C _ { r } \in { \mathcal { C } }$ , we construct the induced subgraph $G _ { s } [ C _ { r } ]$ and apply the Louvain algorithm independently [4] to obtain sub-communities.

This step produces local clusters within each parent community and exposes behavioral diferences that may remain hidden at the global level. A higher resolution is used for finer partitioning, while decomposition is limited to suficiently large communities to maintain structural stability.

The resulting structure defines a two-level hierarchy in which each perpetrator node $p \in { \mathcal { P } }$ is associated with both a community label and a sub-community label. This hierarchical organization captures both meso-scale community-level and micro-scale sub-community-level behavioral patterns.

Sub-community detection complements the global community structure by revealing internal variations within communities, thereby supporting more fine-grained structural and behavioral analysis in subsequent sections. Each perpetrator node � is thus associated with a pair $( c ( p ) , s ( p ) )$ , where $c ( p )$ denotes the community label and $s ( p )$ denotes the corresponding sub-community label.

## 3.3.3. Community Behavioral Profiling and

## Contribution Analysis

Following community detection, we apply two complementary analyses to interpret the behavioral organization of the perpetrator network. Lift-based profiling identifies attributes that are relatively prominent within each community, while cosine-similarity decomposition quantifies how individual features and feature groups contribute to behavioral similarity.

Lift-Based Community Profiling: Let $C _ { r } \subseteq \mathcal { P }$ denote community $r ,$ and let $x _ { p , k }$ denote the value of feature � for perpetrator node $p .$ The lift of feature � in community $C _ { r }$ is defined as

$$
L _ { r , k } = \frac { \displaystyle \frac { 1 } { | C _ { r } | } \sum _ { p \in C _ { r } } x _ { p , k } } { \displaystyle \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } x _ { p , k } + \epsilon } ,\tag{8}
$$

where � is a small constant used to avoid division by zero. A value of $L _ { r , k } \ > \ 1$ indicates that feature � is over-represented in community $C _ { r }$ relative to the overall perpetrator population.

The same calculation is applied at the feature-group level by averaging the features indexed by $\kappa _ { g } .$ , where $\kappa _ { g }$ denotes the set of features belonging to group $g .$ Feature-level lift is used to identify dominant and distinctive attributes, while group-level lift summarizes the relative prominence of the eight feature groups defined in Section 3.2.3. For behavioral fingerprint visualization, the selected lift values are standardized column-wise across communities.

Behavioral Contribution Analysis: The behavioral representation in Eq. (3) and cosine similarity in Eq. (6) allow the similarity between perpetrator nodes to be decomposed into feature-wise contributions. For $p , q \in \mathcal { P }$

$$
\sin ( p , q ) = \sum _ { k = 1 } ^ { 6 4 } \phi _ { k } ( p , q ) ,\tag{9}
$$

where

$$
\phi _ { k } ( p , q ) = { \frac { x _ { p , k } x _ { q , k } } { \| \mathbf { x } _ { p } \| _ { 2 } \| \mathbf { x } _ { q } \| _ { 2 } } }\tag{10}
$$

denotes the contribution of feature � to the similarity between � and $q .$

For interpretation at the feature-group level, the featurewise contributions are aggregated as

$$
\Phi _ { g } ( p , q ) = \sum _ { k \in { \mathcal { K } } _ { g } } \phi _ { k } ( p , q ) ,\tag{11}
$$

where $\kappa _ { g }$ is the index set of features belonging to group �.

To characterize the contribution of feature group � within community $C _ { r } ,$ , we average the absolute pairwise

contributions over all unique perpetrator pairs in that community:

$$
\overline { { \Phi } } _ { g } ^ { ( C _ { r } ) } = \frac { 1 } { | \mathcal { Q } _ { r } | } \sum _ { ( p , q ) \in \mathcal { Q } _ { r } } | \Phi _ { g } ( p , q ) | ,\tag{12}
$$

where

$$
\mathcal { Q } _ { r } = \{ ( p , q ) \ | \ p , q \in C _ { r } , \ p < q \}\tag{13}
$$

denotes the set of unique perpetrator pairs within community $C _ { r }$ .

To compare feature-group contributions across the full network, the group-level discriminative score is defined as

$$
\Delta _ { g } = \mathbb { E } \left[ | \Phi _ { g } ( p , q ) | \mid c ( p ) = c ( q ) \right] - \mathbb { E } \left[ | \Phi _ { g } ( p , q ) | \mid c ( p ) \neq c ( q ) \right] ,\tag{14}
$$

where $c ( p )$ denotes the community label of perpetrator node �. A positive value indicates that group � contributes more strongly to similarity among perpetrators in the same community than among perpetrators assigned to diferent communities.

The corresponding discriminative score for an individual feature is

$$
\Delta _ { k } = \mathbb { E } \left[ \vert \phi _ { k } ( p , q ) \vert \mid c ( p ) = c ( q ) \right] - \mathbb { E } \left[ \vert \phi _ { k } ( p , q ) \vert \mid c ( p ) \neq c ( q ) \right] .\tag{15}
$$

Lift and contribution decomposition provide diferent forms of interpretation. Lift identifies attributes that are relatively prominent within a community, whereas $\overline { { \Phi } } _ { g } ^ { ( C _ { r } ) } , \Delta _ { g } ,$

and $\Delta _ { k }$ measure how feature groups and individual features contribute to similarity.

## 3.3.4. Community-Constrained Influence

## Maximization

The detected communities define structurally coherent regions in which cyberbullying behavior may propagate. Rather than performing influence maximization on the full network [21], we formulate the problem independently within each community to preserve local behavioral structure and avoid dominance by large communities.

For a given community �, let

$$
{ \mathcal { P } } _ { C } = \{ p \in { \mathcal { P } } \mid c ( p ) = C \}\tag{16}
$$

denote the set of perpetrator nodes assigned to community $C ,$ , where $c ( p )$ is the community label of perpetrator node $p .$

Difusion Model (Independent Cascade): Behavioral influence propagation is modeled using the Independent Cascade (IC) process [21, 12], a widely adopted difusion model for influence analysis in social and information networks. The IC model is well suited to the proposed framework because cyberbullying interactions are modeled as stochastic peer-to-peer influence events occurring over weighted behavioral-similarity connections. Unlike threshold-based collective activation models, the IC process explicitly captures localized probabilistic transmission along network edges, enabling interpretable modeling of behavioral difusion and eficient influence maximization.

When a perpetrator node � becomes active, it has a single chance to activate each inactive neighboring perpetrator

node � with probability

$$
\rho ( p , q ) = \operatorname* { m i n } ( \alpha w _ { p q } , \rho _ { \operatorname* { m a x } } ) ,\tag{17}
$$

where $w _ { p q }$ denotes the behavioral-similarity weight between perpetrator nodes � and �, � is a scaling parameter controlling difusion strength, and $\rho _ { \mathrm { m a x } }$ is an upper bound on activation probability.

To emphasize harmful users, each perpetrator node is assigned a bullying-intensity weight

$$
\begin{array} { r } { \omega ( p ) = \log ( 1 + b _ { p } ) , } \end{array}\tag{18}
$$

where $b _ { p }$ denotes the number of bullying actions performed by perpetrator �.

The influence of a seed set $S \subseteq { \mathcal { P } } _ { C }$ is defined as the expected weighted number of activated perpetrator nodes under the IC process:

$$
\sigma ( S ) = \mathbb { E } \left[ \sum _ { q \in { \cal A } ( S ) } \omega ( q ) \right] ,\tag{19}
$$

where $\boldsymbol { \mathcal { A } } ( \boldsymbol { S } )$ denotes the set of perpetrator nodes activated during difusion.

Greedy Influence Maximization: The goal is to identify a seed set of size $k _ { s }$ that maximizes $\sigma ( S )$ . Since this objective does not admit a closed-form solution, it is estimated via Monte Carlo simulation, and a greedy algorithm is used for optimization [21]. At each step, the perpetrator node providing the largest marginal gain is selected:

$$
p ^ { * } = \arg \operatorname* { m a x } _ { p \in \mathcal { P } _ { C } \backslash S } \left[ \sigma ( S \cup \{ p \} ) - \sigma ( S ) \right] .\tag{20}
$$

This approach leverages the approximately submodular structure of the influence objective to obtain a near-optimal solution using greedy optimization [21].

Baseline Methods: To evaluate the efectiveness of the proposed approach, we consider three baseline seed-selection strategies:

(i) Bully-Only: Perpetrator nodes are ranked based on bullying intensity $b _ { p } ,$ and the top- $\mathbf { \nabla } \cdot k _ { s }$ nodes are selected.

(ii) Weighted Degree: Perpetrator nodes are ranked based on their total similarity strength, defined as $\begin{array} { r } { \sum _ { q \in \mathcal { N } ( p ) } w _ { p q } , } \end{array}$ where $\mathcal { N } ( p )$ denotes the neighborhood of perpetrator node �.

(iii) PageRank: Perpetrator nodes are ranked using PageRank scores [29] computed over the similarity graph, capturing global structural importance.

These baselines capture complementary aspects of influence, including individual aggressiveness, local connectivity, and global centrality.

Community-Constrained Formulation: The influence maximization procedure is applied independently within each community �. For each community-specific perpetrator set $\mathcal { P } _ { C }$ , a seed set of size $k _ { s }$ is selected, and the final seed set is obtained by aggregating the selected perpetrator nodes across communities.

This formulation supports localized interventions and distributes seeds across structurally distinct behavioral groups. Section 4.6 evaluates the methods using both standard and bullying-weighted influence spread.

## 4. Experimental Evaluation

This section evaluates the ODCBNA framework across multiple analytical dimensions, including network structure, behavioral organization, contextual interaction, difusion dynamics, and interpretability. The analyses are designed to address the competency questions outlined in Section 3.1.1, thereby validating the efectiveness of the ontology-driven knowledge graph and the subsequent modeling pipeline.

The evaluation begins with the experimental setup, followed by multi-scale network structure analysis and community-level behavioral and contextual characterization. Subsequent analyses examine behavioral contribution, influence propagation, and difusion dynamics, culminating in an integrated interpretation of the cyberbullying ecosystem.

## 4.1. Experimental Setup

This subsection outlines the evaluation network, difusion settings, parameter-selection rationale, baseline comparison methods, and measures used in the experiments.

Network Construction and Scope: The experiments use the perpetrator similarity network $G _ { s }$ constructed from the behavioral representations described in Section 3.2.3. It contains $\vert \mathcal { P } \vert ~ = ~ 1 , 0 6 3$ perpetrator nodes connected by weighted similarity edges obtained through a �-nearestneighbor strategy. To distinguish the neighborhood size from the influence seed budget, we denote it by $k _ { \mathrm { N N } }$ and set $k _ { \mathrm { N N } } =$ 40. The directed kNN graph is subsequently symmetrized to obtain the undirected weighted network $G _ { s }$ .

Difusion Parameters: Influence propagation follows the Independent Cascade (IC) process introduced in Section 3.3.4. We use a seed budget of $k _ { s } = 3$ , difusion-strength parameter $\alpha = 0 . 5 ,$ upper bound on the activation probability $\rho _ { \mathrm { m a x } } = 0 . 2 5 , R _ { \mathrm { s e l } } = 1 2 0$ Monte Carlo runs during seed selection, and $R _ { \mathrm { e v a l } } = 3 0 0$ runs for final influence evaluation.

Parameter Selection Rationale: The adopted parameter values are treated as study-specific experimental settings rather than universally optimal choices. For network construction, $k _ { \mathrm { N N } } = 4 0$ is selected to maintain a connected similarity network while preserving locality by restricting each perpetrator to a limited set of behaviorally similar neighbors. The seed budget $k _ { s } = 3$ represents a constrained intervention in which only a small number of perpetrator nodes can be selected within each community. In the IC model, � scales the behavioral-similarity weights used to determine activation probabilities, while $\rho _ { \mathrm { m a x } }$ bounds these probabilities to avoid excessively large activation likelihoods. Monte Carlo simulation follows established practice [21]. A smaller Monte Carlo budget, $R _ { \mathrm { s e l } } ,$ is used during greedy seed selection because influence spread must be estimated repeatedly across candidate nodes, whereas the larger budget, $R _ { \mathrm { e v a l } } ,$ is used for final evaluation to obtain more reliable spread estimates. All seed-selection methods are evaluated using the same difusion parameters and seed budget to ensure comparability. These settings define the operating point of the present experiments rather than a claim of parameter optimality.

Baseline Methods: We compare the greedy influence maximization strategy with three baseline seed-selection strategies introduced in Section 3.3.4, namely Bully-Only, Weighted Degree, and PageRank [29].

Evaluation Measures: Diferent measures are used for each analytical task. Network structure is examined through density, average degree, clustering coeficient, average shortest path length, and modularity. Community profiles are described using dominant attributes and lift-based distinctive features. Influence maximization is evaluated through expected spread, relative gain over the strongest baseline, and overlap among selected seed sets. Feature interpretability is assessed by comparing contribution values for withincommunity and between-community perpetrator pairs at both group and individual feature levels.

## 4.2. Multi-scale Network Structure

This subsection examines the perpetrator similarity network at three structural scales: the global network, communities, and sub-communities. It characterizes network connectivity and the hierarchical organization of similaritybased behavioral groupings.

Global Network Structure: Table 2 summarizes the perpetrator, community, and sub-community networks. The perpetrator similarity network contains 1,063 nodes and 27,554 undirected edges, with an average degree of 51.84 and a density of 0.0488. This low density reflects the sparse connectivity produced by the �-nearest-neighbor construction.

## Table 2

Structural properties of the perpetrator, community, and subcommunity networks.
<table><tr><td>Network</td><td>#Nodes #Edges Avg Deg</td><td></td><td></td><td>Density Clust</td><td>Path</td></tr><tr><td>Perpetrator</td><td>1063</td><td>27554</td><td>51.84</td><td>0.0488</td><td>0.1939 3.4192</td></tr><tr><td>Community</td><td>6</td><td>15</td><td>5.00</td><td>1.0000 1.0000 0.0015</td><td></td></tr><tr><td>Sub-community</td><td>27</td><td>311</td><td>23.04</td><td>0.88600.8932 0.0156</td><td></td></tr></table>

Notes: Avg Deg = Average degree; Clust = Average clustering coeficient; Path = Average weighted shortest-path distance in the largest connected component.

The perpetrator network has an average clustering coeficient of 0.1939 and an average weighted shortest-path distance of 3.4192. These values indicate some local grouping and relatively short paths within the largest connected component, consistent with characteristics associated with small-world networks [36]. A comparison with an appropriate random-network baseline would be required for formal confirmation.

At the aggregated levels, the community network is complete, while the sub-community network remains highly connected, with a density of 0.8860 and a clustering coeficient of 0.8932.

Interactive visualizations of the perpetrator similarity network are available at https://cyberbly-na.vercel.app /visualizations/perpetrator-similarity.

Community-Level Structure: The Louvain resolution parameter was selected by examining candidate values between 0.70 and 1.20. Among the tested values, 0.85 yielded the highest modularity of approximately 0.495, identifying six communities with sizes ranging from 104 to 237 perpetrators. Table 3 reports the internal structural properties and meta-level connectivity of each community.

Table 3  
Table 4  
Community-level structural properties of the perpetrator network.
<table><tr><td></td><td>Comm. #Nodes #Edges</td><td></td><td>Density</td><td>Avg Deg</td><td>Clust</td><td>Meta Deg</td></tr><tr><td>CO</td><td>213</td><td>3214</td><td>0.1424</td><td>30.18</td><td>0.2676</td><td>5</td></tr><tr><td>C1</td><td>141</td><td>2135</td><td>0.2163</td><td>30.28</td><td>0.2984</td><td>5</td></tr><tr><td>C2</td><td>184</td><td>3728</td><td>0.2214</td><td>40.52</td><td>0.3208</td><td>5</td></tr><tr><td>C3</td><td>104</td><td>1514</td><td>0.2827</td><td>29.12</td><td>0.3621</td><td>5</td></tr><tr><td>C4</td><td>237</td><td>4010</td><td>0.1434</td><td>33.84</td><td>0.2570</td><td>5</td></tr><tr><td>C5</td><td>184</td><td>3556</td><td>0.2112</td><td>38.65</td><td>0.3157</td><td>5</td></tr></table>

Notes: Avg Deg = Average internal degree; Clust = Average weighted clustering coeficient; Meta Deg = Degree in the community meta-network.

C3, the smallest community, has the highest internal density of 0.2827, whereas C0 and C4, the two largest communities, have the lowest densities of 0.1424 and 0.1434, respectively. This indicates that the smaller community contains a greater proportion of possible internal links, while the two largest communities exhibit more dispersed connectivity.

Average degree follows a diferent pattern. C2 has the highest value of 40.52, followed by C5 at 38.65, whereas C3 has the lowest at 29.12. Weighted clustering ranges from 0.2570 to 0.3621, with C3 showing the strongest local clustering and C4 the weakest. These variations indicate that community size, internal connectivity, and local clustering capture diferent aspects of the network structure.

All communities have a meta-degree of 5, meaning that each connects to every other community in the metanetwork. They therefore difer internally but remain interconnected at the aggregated level.

Interactive visualizations of the community network are available at https://cyberbly-na.vercel.app/visualization s/perpetrator-community.

Sub-community (Hierarchical) Structure: Hierarchical Louvain partitioning was applied within the six parent communities to examine finer-grained behavioral groupings. Similar to community detection, candidate resolution values between 0.70 and 1.20 were examined, with 1.02 selected for sub-community detection. At this resolution, the analysis identifies 27 sub-communities, with an edgeweighted average modularity of approximately 0.314. This value represents the average of the within-parent modularity scores, weighted by the number of internal edges in each parent community. Table 4 summarizes the sub-community distribution within each parent community.

Hierarchical sub-community structure within the six parent communities.
<table><tr><td>Parent</td><td>#Subcomms</td><td>Avg Size</td><td>Largest Size</td></tr><tr><td>CO</td><td>4</td><td>53.25</td><td>64</td></tr><tr><td>C1</td><td>4</td><td>35.25</td><td>51</td></tr><tr><td>C2</td><td>6</td><td>30.67</td><td>42</td></tr><tr><td>C3</td><td>4</td><td>26.00</td><td>38</td></tr><tr><td>C4</td><td>5</td><td>47.40</td><td>63</td></tr><tr><td>C5</td><td>4</td><td>46.00</td><td>57</td></tr></table>

The number of sub-communities varies from four to six per parent community. C2 contains the highest number of sub-communities (six), followed by C4 with five, while the remaining communities each contain four. C0 has the largest average sub-community size (53.25), whereas C3 has the smallest (26.00). The largest individual subcommunities occur in C0 and C4, containing 64 and 63 perpetrators, respectively. These diferences indicate that the parent communities exhibit varying levels of internal subdivision, revealing finer-grained groupings within the broader behavioral communities.

Interactive visualizations of the sub-community network are available at https://cyberbly-na.vercel.app/visualiza tions/perpetrator-subcommunity.

Table 5  
Condensed community-level behavioral profiles using dominant attributes from each feature group.
<table><tr><td>Comm.</td><td>Perp.</td><td>Act.</td><td> ${ \sf A c t . } / { \sf P }$ </td><td>Bully</td><td>B/P</td><td>Language</td><td>Target</td><td>Party</td><td>Time</td><td>Channel</td><td>User Profile</td><td>Video</td><td>Impact</td></tr><tr><td rowspan="2">CO</td><td rowspan="2">213</td><td rowspan="2">1099</td><td rowspan="2">5.16</td><td rowspan="2">864</td><td rowspan="2">4.06</td><td rowspan="2">Bengali</td><td>Individual /</td><td> $\mathsf { A L } ~ /$  Other</td><td>Afternoon /</td><td>Rtv News /</td><td>High Video /</td><td>High Avg. Comment /</td><td>High % Reply /</td></tr><tr><td></td><td>General People</td><td>Morning</td><td>Ekattor TV</td><td>High View</td><td>High Avg. Like</td><td>High % Like</td></tr><tr><td rowspan="2">C1</td><td rowspan="2">141</td><td rowspan="2">774</td><td rowspan="2">5.49</td><td rowspan="2">615</td><td rowspan="2">4.36</td><td></td><td>People Group /</td><td> ${ \mathsf { N C P \ } } /$ </td><td> $\mathsf { N i g h t } \ j$ </td><td> $\mathsf { D B C } \mathsf { N e w s } /$ </td><td> $\mathsf { L o w \ A c c . \ A g e } \ /$ </td><td> $\mathsf { L o w \ A v g . \mathsf { V i e w / \Omega } }$ </td><td>Mid p95 Likes /</td></tr><tr><td></td><td>Male</td><td> $\mathsf { J a P a }$ </td><td>Late Night</td><td>Channel 24</td><td>Low View</td><td> $\mathsf { L o w \ A v g . \mathsf { L i k e } }$ </td><td>High % Reply</td></tr><tr><td rowspan="2">C2</td><td rowspan="2">184</td><td rowspan="2">749</td><td rowspan="2">4.07</td><td rowspan="2">663</td><td rowspan="2">3.60</td><td>Romanized</td><td>Organization</td><td>JaPa</td><td>Early Morning /</td><td>DBC News</td><td>Low Subscribe /</td><td> $\mathsf { L o w \ A v g . \mathsf { L i k e } / }$ </td><td>Low p95 Likes /</td></tr><tr><td>Bengali</td><td></td><td></td><td>Late Night</td><td>Independent TV</td><td>Low Video</td><td> $\mathsf { L o w \ A v g . \mathsf { V i e w } }$ </td><td>Low % Like</td></tr><tr><td rowspan="2">C3</td><td rowspan="2">104</td><td rowspan="2">441</td><td rowspan="2">4.24</td><td rowspan="2">387</td><td rowspan="2">3.72</td><td>Romanized Bengali</td><td>Organization /</td><td> $\mathsf { J a P a } \ /$ </td><td>Late Night /</td><td> $\mathsf { l n d e p e n d e n t ~ T V } /$ </td><td> $\mathsf { L o w A c c . A g e } /$ </td><td> $\mathsf { M i d \ A v g . \ L i k e \ / }$ </td><td>Low % Like /</td></tr><tr><td></td><td>Female</td><td> $\scriptstyle \mathtt { O t h e r }$ </td><td>Morning</td><td>Ekattor TV</td><td>Low View</td><td> $\mathsf { M i d ~ A v g . ~ } \mathsf { V i e w }$ </td><td>Low p95 Likes</td></tr><tr><td rowspan="2"></td><td rowspan="2">237</td><td rowspan="2">1351</td><td rowspan="2">5.70</td><td rowspan="2">1058</td><td rowspan="2">4.46</td><td>English</td><td>Female / Male</td><td> $\mathsf { J a m a a t } /$   $\mathsf { B N P }$ </td><td>Evening /</td><td>Jamuna  $\mathsf { T V } /$  Somoy TV</td><td>Low Video /</td><td> $\mathsf { H i g h \ A v g . \ L i k e } \ /$ </td><td>Mid % Like /</td></tr><tr><td>Romanized</td><td></td><td> $\mathsf { A L } ~ /$ </td><td> $\mathsf { E a r l y \mathsf { M o r n i n g } }$ </td><td> $\mathsf { J a m u n a } \mathsf { T v } /$ </td><td>Low View</td><td> $\mathsf { M i d \ A v g . \nabla i e w }$  Low Subscribe / High Avg. View /</td><td>Mid % Reply Low % Like /</td></tr><tr><td rowspan="2"></td><td rowspan="2">184</td><td rowspan="2">745</td><td rowspan="2">4.05</td><td rowspan="2">684</td><td rowspan="2">3.72</td><td>Bengali</td><td>Female /  $\mathsf { P e o p l e \mathsf { G r o u p } }$ </td><td></td><td>Night /</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $\mathsf { J a m a a t }$ </td><td> $\mathsf { E v e n i n g }$ </td><td> $\mathsf { S o m o y ~ T V }$ </td><td>Low Video</td><td>High Avg. Like</td><td>Low p95 Likes</td></tr></table>

Notes: Comm. = Community; Perp. = Number of perpetrators; Act. = Total interaction actions; $\mathsf { A c t . } / \mathsf { P } = \mathsf { A c t i o n s }$ per perpetrator; Bully = Bullying actions; $\mathsf { B } / \mathsf { P } = \mathsf { B u l l y i n g }$ actions per perpetrator. Each feature-group column reports at most the two attributes with the larges lift values for the corresponding community. – indicates that no dominant attribute was reported for that group.

Overall, the network of 1,063 perpetrators exhibits a hierarchical structure comprising six communities and 27 subcommunities. The community and sub-community partitions reveal progressively finer groupings within the broader similarity network.

## 4.3. Community Behavior

This subsection characterizes the six communities using condensed profiles, distinctive attributes, and behavioral fingerprints. The analysis considers the eight feature groups introduced in Section 3.2.3: User Profile, Video, Impact, Language, Target, Party, Time, and Channel.

Community-Level Behavior: Table 5 summarizes community size, interaction activity, bullying activity, and the most prominent attributes from each feature group. C4 and C1 record more than 5.4 actions per perpetrator, followed by C0 with 5.16. The corresponding values for C2, C3, and C5 remain close to 4.0. C4 is both the largest and the most active community, containing 237 perpetrators and 1,351 interaction actions. It also has the highest number of bullying actions per perpetrator at 4.46, followed by C1 at 4.36.

C2 and C3 display similar contextual patterns, including Romanized Bengali, organization-focused targets, and JaParelated discussions. Both are associated with lower userprofile activity attributes, although their video characteristics difer. C2 is linked to low average video views and likes, whereas C3 shows mid-range values. C0 and C1 exhibit higher interaction activity but distinct behavioral profiles. C0 is associated with high user video and view counts, together with stronger reply- and like-based impact attributes. In contrast, C1 is characterized by lower user-profile and video engagement attributes.

C4 combines the highest interaction activity with English content, female and male targets, Jamaat- and BNPrelated discussions, and comparatively stronger mid-range impact attributes. C5 is associated with Romanized Bengali, female and people-group targets, AL- and Jamaat-related discussions, and high average video views and likes. However, its interaction-level impact attributes remain predominantly low. This contrast indicates that greater video engagement does not necessarily correspond to stronger responses to individual interactions.

Distinctive Attributes: Table 6 lists the three attributes with the largest lift values in each community. Lift, defined in Eq. (8), compares the mean value of an attribute within a community with its mean across all perpetrators. Values greater than 1 indicate relative over-representation.

Most of the leading attributes belong to the User Profile, Video, and Impact groups. C1 and C2 are distinguished by low average video views and likes, with additional diferences in peak-like activity. C3, in contrast, is characterized by mid-range average views and likes, together with a low proportion of liked interactions. C5 combines high average video views and likes with a low proportion of liked interactions.

Table 6  
Top three distinctive attributes of each community based on lift.
<table><tr><td>Comm.</td><td>Top Distinctive Attributes</td></tr><tr><td>CO</td><td>High Video (4.64), High View (4.55), Mid Video (2.77)</td></tr><tr><td>C1</td><td>Low Avg. View (2.61), Low Avg. Like (2.44), Mid p95 Likes (2.22)</td></tr><tr><td>C2</td><td>Low Avg. Like (2.75), Low Avg. View (2.59), Low p95 Likes (2.02)</td></tr><tr><td>C3</td><td>Mid Avg. Like (3.00), Mid Avg. View (2.74), Low % Like (1.93)</td></tr><tr><td>C4</td><td>Mid % Like (2.35), Mid % Reply (2.10), Mid p95 Likes (2.05)</td></tr><tr><td>C5</td><td>High Avg. View (2.61), High Avg. Like (2.53), Low % Like (2.03)</td></tr></table>

The strongest lift values occur in C0 and relate to User Profile, particularly high video and view counts. C4 is diferentiated mainly by interaction-response attributes, including mid-range liked and replied proportions and peak likes. These patterns indicate that community distinctions arise from diferent combinations of account activity, video exposure, and engagement response.

Behavioral Fingerprint Analysis: Fig. 7 provides a compact comparison of the community profiles. It includes the aggregate profile of each feature group together with up to two representative attributes from that group. The selected lift values are standardized column-wise across communities before plotting. The colors therefore represent relative variation within each dimension rather than diferences in their original scales.

The fingerprints reveal several contrasts. C3 shows strong positive intensity in mid-range average video likes and views, whereas C2 and C5 exhibit weaker signals for these attributes. C4 has stronger aggregate Language, Target, Party, Time, and Channel signals. Language-related diferences include Romanized Bengali in C5 and emojionly content in C3. Target-related variation is particularly visible in organization-focused activity in C2 and peoplegroup associations in C5.

Additional variation appears across Party, Time, and Channel. C1 is associated with NCP-related discussions, while C2 shows stronger JaPa and early-morning signals. C3 is distinguished by late-night activity and associations with Independent TV and Ekattor TV. C4 exhibits relatively strong aggregate Party, Time, and Channel intensities, indicating broader contextual diferences among the communities.

Impact patterns also separate the communities. Low liked-interaction and low peak-like attributes are more prominent in C2, C3, and C5, but less prominent in C1 and C4. These diferences reflect the relative representation of specific engagement-response categories rather than absolute interaction intensity. Taken together, the fingerprints reveal variation in video engagement, interaction response, language use, target identity, political context, temporal activity, and channel exposure.

## 4.4. Community Interaction

To complement the behavioral profiles in Section 4.3, we construct two aggregated views: a Community–Behavioral Attribute Interaction Network and a Community–Context Interaction Network. These networks summarize community associations with behavioral and contextual attributes. Edge weights represent perpetrator counts in the behavioral network and interaction counts in the contextual network, while edge intensity indicates the corresponding bullying proportion.

Community–Behavioral Attribute Interaction Network: The network connects communities with attributes from the User Profile, Video, and Impact groups. An interactive visualization is available at https://cyberbly-na.verc el.app/visualizations/community-behavior.

![](images/658feb8ae157383115e6d1512850b59084f33d84cd6b5853fcd732b690072347.jpg)  
Figure 7: Community behavioral fingerprints based on column-wise standardized lift values.

For the User Profile group, high subscriber count is prominent in C0, whereas low view and video counts dominate several other communities. However, the highest bullying proportions occur in diferent categories, including low subscriber count in C0, mid view count in C2–C4, and high video count in C5.

The Video group reveals diferences in content exposure and engagement. High average comment count is prominent in C0, low average views or likes in C1 and C2, mid average likes in C3, and high average likes or views in C4 and C5. The highest bullying proportions often correspond to diferent attributes, such as low video duration in C0, mid average views in C2, and low average views in C4.

For the Impact group, low percentage of replied interactions is prominent in C0, C1, C3, and C4, while low peaklike count is prominent in C2 and C5. In contrast, high percentage of replied interactions records the highest bullying proportion in C0, C1, and C4, whereas low percentage of liked interactions is associated with the highest proportion in C2.

Overall, attributes associated with more perpetrators do not necessarily exhibit higher bullying proportions, highlighting diferences in behavioral patterns among communities.

Community–Context Interaction Network: The network connects communities with attributes from the Language, Target, Party, Time, and Channel groups, comparing interaction volume and bullying proportion. An interactive visualization is available at https://cyberbly-na.vercel.app /visualizations/community-context.

For the Language group, bengali accounts for the greatest interaction volume in all communities. However, romanized bengali records the highest bullying proportion in C0, C2, and C4, while code-mixed content leads in C5, with all 26 interactions labeled as bullying.

The Target group shows that male targets dominate interaction volume in C0, C1, C4, and C5, whereas organizationfocused interactions lead in C2 and C3. Higher bullying proportions are associated with female targets in C0, people groups in C1, C2, and C4, general people in C3, and organizations in C5.

For the Party group, NCP accounts for the greatest interaction volume in C0, C1, and C4, JaPa in C2 and C3, and Jamaat in C5. In contrast, AL records the highest bullying proportion in four communities, while JaPa leads in C2 and C5.

For the Time group, afternoon dominates interaction volume in every community, whereas the highest bullying proportions occur mainly at night, except in C2 (early morning) and C5 (evening).

The Channel group reveals further diferences. Rtv News leads interaction volume in C0, Channel 24 in C1, Independent Television in C2 and C3, and Jamuna TV in C4 and C5. However, Jamuna TV records the highest bullying proportion in C0, C1, C2, and C4, followed by DBC NEWS in C3 and Ekattor TV in C5.

Overall, greater interaction volume does not necessarily correspond to higher bullying proportions, highlighting contextual diferences in community activity and bullyinglabeled interactions.

## 4.5. Behavioral Contribution Analysis

The preceding analyses describe the structural organization and behavioral patterns of perpetrators, but they do not show which dimensions contribute most to the observed similarity structure. We therefore examine feature contributions at the group, individual, and community levels using the framework defined in Section 3.3.3.

Group-Level Contribution Analysis: Fig. 8 reports the group-level discriminative scores computed using Eq. (14). Each score represents the diference between the mean contribution of a feature group for within-community pairs and that for between-community pairs. A larger positive value indicates a stronger contribution to within-community similarity relative to between-community similarity.

![](images/9863ed5c76da1881c294dda9e6a86208eb969e934ade4dd09ee1bc7bd0917597.jpg)  
Figure 8: Group-level diferences between within-community and between-community similarity contributions.

The Video group records the largest discriminative contribution, followed closely by the Impact group and then the User Profile group. Their scores are substantially greater than those of the Party, Channel, Target, Time, and Language groups, whose contributions remain comparatively small.

These results indicate that community separation is associated mainly with video exposure, engagement response, and user-profile characteristics. The contextual feature groups contribute less directly to distinguishing withincommunity similarity from between-community similarity.

Individual-Level Contribution: Fig. 9 presents the 10 features with the largest discriminative scores computed using Eq. (15). The ranking considers all 64 features, while only the leading subset is shown for readability. A larger positive score indicates a stronger contribution to withincommunity similarity relative to between-community similarity.

The leading features come entirely from the Impact, Video, and User Profile groups. The strongest scores are associated with low peak-like count and low percentage of liked interactions, followed by low average video likes. Low account-level video and view counts also rank among the leading attributes. These results indicate that lower engagement response, video exposure, and user activity contribute prominently to distinguishing within-community similarity from between-community similarity.

![](images/4e88a79b720d38f68fb11d6b5690d40862f2b13f2ad6fb965c4866d74e4759ae.jpg)  
Figure 9: Top 10 individual features based on withincommunity and between-community contribution diferences.

The remaining leading features include high average video likes, low and high average video views, high peak-like count, and high percentage of liked interactions. Together, they indicate that both lower and higher ranges of video exposure and engagement response contribute to community diferentiation.

The Language, Target, Party, Time, and Channel groups are absent from the displayed subset, indicating smaller individual discriminative contributions than the leading behavioral features.

Community-Specific Contribution Patterns: Fig. 10 presents the mean within-community contribution of each feature group. Pairwise group contributions are computed using Eq. (11) and averaged within each community according to Eq. (12).

The User Profile group provides the largest contribution in C0, C1, and C4. In C0, it is followed by the Impact and Video groups, whereas the Video group ranks second in C1 and C4. In contrast, the Impact group dominates C2, C3, and C5, followed by the Video and User Profile groups.

![](images/9d48efa8f7597dde26c99a46d563a63bd043ea911b4d5778ada7d4ca2d045d6c.jpg)  
Figure 10: Community-wise mean contributions of behavioral and contextual feature groups to within-community similarity.

In all six communities, the User Profile, Video, and Impact groups contribute substantially more than the Language, Target, Party, Time, and Channel groups. Their relative prominence varies among communities, indicating that within-community similarity reflects diferent combinations of account characteristics, video exposure, and engagement response.

## 4.6. Influence Propagation

This subsection evaluates the community-constrained influence maximization framework from three perspectives: spread performance, seed-selection behavior, and difusion dynamics.

Influence Spread Performance: Fig. 11 compares the average number of activated nodes achieved by the greedy method and the three baseline strategies in the six communities.

The four methods produce closely comparable spreads in every community. The greedy method achieves marginally higher spread than the baselines in all six communities, although the diferences remain small. C4 records the highest overall spread, followed by C0, while C3 exhibits the lowest. The diferences among methods are considerably smaller than those among communities, indicating that distinct seedselection strategies achieve similar difusion reach under the adopted IC parameters and seed budget.

![](images/ba9ac4345e4d16dc60ba736962221877574ae2e29cbe44f69908c8d03c6f0331.jpg)  
Figure 11: Average influence spread of the seed-selection methods across communities for a seed budget of $k _ { s } = 3$

![](images/4f7d05f72e7231db3151bb174b7a32e913b029ff8e3a3c0c9ed7414099c861a9.jpg)  
Figure 12: Percentage diference between the greedy method and the strongest baseline across communities.

Relative Gain Analysis: Fig. 12 reports the percentage diference between the greedy method and the strongest baseline in each community.

The greedy method records positive gains in all six communities. The improvement is largest in C4 at approximately 0.77%, followed by C1 at 0.66% and C3 at 0.62%. Gains in C2 and C5 remain below 0.4%, while C0 records the smallest improvement at approximately 0.27%.

Overall, the baseline strategies already achieve spreads close to those obtained by the greedy method. Direct optimization provides consistent but modest improvements across the six communities under the adopted IC settings.

Seed Selection Divergence: Table 7 compares the greedy seed sets with those selected by the three baseline strategies for $k _ { s } = 3 .$

Seed overlap between the greedy method and baseline strategies for $k _ { s } = 3$
<table><tr><td>Comm.</td><td>IMnB</td><td>IMnWD</td><td>IMnPR</td><td>B (%)</td><td>WD (%) PR (%)</td></tr><tr><td>CO</td><td>0</td><td>0</td><td>0</td><td>0.00</td><td>0.00 0.00</td></tr><tr><td>C1</td><td>0</td><td>0</td><td>0</td><td>0.00</td><td>0.00 0.00</td></tr><tr><td>C2</td><td>1</td><td>0</td><td>0</td><td>33.33</td><td>0.00 0.00</td></tr><tr><td>C3</td><td>1</td><td>0</td><td>0</td><td>33.33</td><td>0.00 0.00</td></tr><tr><td>C4</td><td>0</td><td>0</td><td>0</td><td>0.00</td><td>0.00 0.00</td></tr><tr><td>C5</td><td>0</td><td>0</td><td>0</td><td>0.00</td><td>0.00 0.00</td></tr></table>

Notes: Comm. = Community; IM = Greedy influence maximization; $\mathsf { B } = \mathsf { B u l l y - O n l y }$ ; WD = Weighted Degree; PR = PageRank. Overlap percentages are calculated relative to the seed budget.

The greedy seed sets have limited overlap with those of the baselines. In C2 and C3, the greedy and Bully-Only methods share one seed each, corresponding to 33.33% of the three-node seed set. No overlap with Weighted Degree or PageRank appears in any community. An additional evaluation with $k _ { s } = 5$ shows Bully-Only overlap of 20% in C2 and 40% in C3, with no overlap elsewhere or with the other baselines.

Despite selecting largely diferent nodes, the methods achieve closely comparable influence spreads for $k _ { s } \ = \ 3 .$ This indicates that distinct seed configurations can provide similar difusion reach under the adopted IC setting, suggesting that influence propagation is not dependent on a single seed configuration.

Difusion Dynamics: Difusion dynamics were examined for all six communities. As an illustrative example, Fig. 13 presents C5, comparing the representative difusion trajectories of the greedy method and the Bully-Only baseline.

Diffusion Dynamics Comparison in Community 5  
![](images/af0096f38f061c4a4f23ca5131f3e4673f1b08d598e5eefd35987c324b572fc5.jpg)

![](images/3368328e5bcd853981b1c85e4445e98e4a834c0acda6add336d19f7e5e9d0f62.jpg)  
Figure 13: Representative difusion dynamics of IM Greedy and Bully-Only in community C5.

Both methods exhibit similar activation patterns, with newly activated nodes peaking at step 2. The Bully-Only baseline activates more nodes at step 1, whereas the greedy method reaches a higher peak at step 2, activating 96 nodes compared with 85 for the baseline. Activation subsequently declines for both methods. The greedy method reaches a difusion depth of four steps, while the baseline continues for five steps.

The cumulative curves remain close throughout the process. The baseline leads initially, but the greedy method overtakes it at step 2 and finishes with 182 activated nodes compared with 181 for Bully-Only. Their average spreads are also comparable, at 181.84 and 181.17 activated nodes, respectively. These results indicate that diferent seed configurations can achieve similar difusion reach while exhibiting diferences in activation timing and difusion depth.

Visualizations of influence propagation in all six communities under the four seed-selection methods are available at https://cyberbly-na.vercel.app/visualizations/diffus ion-all-methods.

Together with the limited seed overlap, the similar diffusion trajectories observed in C5 suggest that distinct seed configurations can achieve comparable difusion reach under the adopted IC setting.

## 4.7. Discussion

The findings reported in Sections 4.2–4.6 are discussed collectively to explain how network structure, behavioral similarity, contextual variation, and influence propagation characterize perpetrator behavior within the ODCBNA framework.

Behavioral Organization: The six communities exhibit diferences in size, internal connectivity, and behavioral profiles, with further variation revealed by the 27 subcommunities. Community-level profiles and fingerprints show diferences in user characteristics, video exposure, and engagement response, with the prominence of these dimensions varying among communities. These findings indicate that perpetrators with similar behavioral patterns form distinguishable groups while retaining internal structural and behavioral variation.

Contextual Embedding: The Language, Target, Party, Time, and Channel groups contribute relatively little to behavioral similarity and community separation, but help characterize the settings in which bullying-labeled interactions occur. The community–context interaction network further shows that attributes associated with greater interaction volume do not necessarily exhibit higher bullying proportions. Thus, behavioral similarity and contextual variation provide diferent but related perspectives on perpetrator activity and community characteristics.

Behavioral Sources of Similarity: At the group level, the Video group records the largest discriminative contribution, followed closely by the Impact and User Profile groups. The leading individual features also belong entirely to these three groups, while the contextual groups have substantially smaller discriminative scores. Community-specific analysis reveals variation in their relative contributions, with the User Profile group most prominent in C0, C1, and C4, and the Impact group in C2, C3, and C5. These findings indicate that behavioral similarity reflects diferent combinations of video exposure, engagement response, and account-level characteristics rather than any single feature group or contextual attribute.

Influence Propagation: The greedy method achieves small positive gains in all six communities, although the strongest baselines remain competitive. The seed sets show limited overlap, with only one shared seed between IM Greedy and Bully-Only in each of C2 and C3. The illustrative analysis of C5 further shows similar difusion trajectories despite diferences in peak activation and difusion depth. These findings suggest that distinct seed configurations can achieve comparable difusion reach under the adopted IC parameters.

Overall Interpretation: The findings reveal a multidimensional system linking network structure, behavioral similarity, contextual variation, and simulated influence propagation. Communities reflect diferent combinations of user characteristics, video exposure, and engagement response, while contextual analysis identifies settings associated with community activity and bullying concentration. Influence analysis further shows that distinct seed configurations can produce comparable difusion reach under the adopted IC model.

Methodological Implications: The findings support integrating competency-question-guided ontology design with behavioral and network analysis. The ontology preserves semantic relationships among users, interactions, channel content, discussion subjects, and contextual entities, providing a consistent basis for knowledge graph construction, behavioral representation, community detection, contribution analysis, and influence modeling. ODCBNA thus connects observed corpus records with interpretable network-level findings beyond text-only analysis.

Practical Implications: The findings suggest that monitoring and mitigation strategies should not rely solely on highly active users or a single centrality measure. Diferent seed-selection methods identify largely distinct perpetrators while achieving comparable simulated difusion reach. Combining behavioral similarity, contextual information, network position, and engagement response may therefore help identify potentially influential users within each community.

Competency Question Alignment: Table 8 summarizes how the experimental findings address the competency questions introduced in Section 3.1.1, linking the ontology requirements with the analytical outcomes of ODCBNA.

## 4.8. Limitations

Several limitations should be considered when interpreting the findings.

Behavioral Representation: The analysis uses a manually designed 64-dimensional behavioral–contextual feature vector and cosine similarity, which may not capture latent or nonlinear behavioral relationships. The contribution results depend on the selected attributes, preprocessing decisions, and feature-group definitions. Learned or hybrid representations may capture additional patterns while preserving interpretability.

Table 8  
Summary of competency questions and corresponding findings.
<table><tr><td>CQ</td><td>Competency Question (Short)</td><td>Key Findings</td></tr><tr><td></td><td>CQ1 Identification of perpetrator users</td><td>Applying the threshold τ = 3 identifies 1,063 users with at least three bullying-labeled comments or replies as perpetrators (Section 3.2.2).</td></tr><tr><td></td><td>CQ2 Behavioral characteristics of perpetra- tors</td><td>Community profiles reveal differences in user characteristics, video exposure, engagement response, language use, target identity, political context, temporal activity, and channel exposure (Section 4.3).</td></tr><tr><td></td><td>CQ3 Multi-scale network structure</td><td>The perpetrator similarity network contains 1,063 nodes and 27,554 edges, with an average degree of 51.84, clustering coefficient of 0.1939, and average weighted shortest-path distance of 3.4192 in the largest connected component (Section 4.2).</td></tr><tr><td></td><td>CQ4 Community and sub-community struc- ture</td><td>Louvain detection identifies six communities with a modularity of approximately 0.495. Hierarchical analysis further reveals 27 sub-communities, exposing finer structural divisions within the parent communities (Section 4.2).</td></tr><tr><td></td><td>CQ5 Behavior-context association</td><td>Communities differ in behavioral profiles and contextual settings. Attributes associated with greater interaction volume do not necessarily exhibit higher bullying proportions (Section 4.4).</td></tr><tr><td></td><td>CQ6 Contributions to similarity and commu- nity separation</td><td>The Video group records the largest discriminative contribution, followed closely by the Impact and User Profile groups. These three groups also dominate the leading individual features, while their relative contributions vary among communities (Section 4.5).</td></tr><tr><td></td><td>CQ7 Influential users within communities</td><td>IM Greedy achieves small positive gains in all six communities. Limited seed overlap with the baselines indicates that different methods identify largely distinct potentially influential users (Section 4.6).</td></tr><tr><td></td><td>CQ8 Influence propagation behavior</td><td>Distinct seed sets achieve comparable diffusion reach under the adopted IC settings. Representative trajectories in C5 show similar final activation counts (182 versus 181) but different diffusion depths (four versus five steps) for IM Greedy and Bully-Only (Section 4.6).</td></tr></table>

Analytical Granularity: Behavioral, contextual, and contribution analyses are conducted mainly at the community level. Although 27 sub-communities are identified, these analyses are not systematically extended to that finer scale. Local behavioral variations within the parent communities may therefore remain underrepresented.

Network and Model Parameters: The perpetrator set, similarity network, community structure, and difusion outcomes depend on methodological choices such as the perpetrator threshold �, neighborhood size $k _ { \mathrm { N N } } ,$ Louvain resolution, seed budget $k _ { s } ,$ and IC propagation parameters. The findings therefore reflect the adopted experimental settings rather than parameter-optimal configurations. Further sensitivity analysis is needed to assess the stability of these results under alternative parameter values.

Temporal and Difusion Assumptions: The similarity network is treated as static, and the IC model uses fixed edge-level activation probabilities. However, user behavior and interaction patterns may evolve over time. Therefore, the difusion results represent simulated outcomes under the adopted network and parameter settings rather than direct estimates of real-world propagation.

Dataset Scope and Generalizability: The framework is instantiated using video-based Bengali political-news interactions collected from selected YouTube channels during a specific period. The findings may depend on the language, topic, collection window, and source-selection criteria of BenCoDiC. Evaluation using other YouTube content types, languages, topics, and time periods is needed to assess broader generalizability.

These limitations motivate future work on learned or hybrid representations, sub-community-level analysis, parameter sensitivity, temporal network modeling, and crossdomain evaluation.

## 5. Conclusion

This paper presented an Ontology-Driven Cyberbullying Behavior and Network Analysis (ODCBNA) framework for perpetrator-centric cyberbullying analysis on YouTube. Guided by analytical requirements and competency questions, ODCBNA integrates application ontology design, knowledge graph construction, heterogeneous graph transformation, behavioral similarity modeling, community analysis, and influence propagation within a unified pipeline.

Using BenCoDiC as a use case, the framework identified 1,063 perpetrators and constructed a weighted similarity network with 27,554 edges. The network revealed six communities with a modularity of approximately 0.495 and 27 sub-communities, demonstrating organization at multiple structural levels. The Video group recorded the largest discriminative contribution, followed closely by the Impact and User Profile groups. These groups also dominated the leading individual features, while the Language, Target, Party, Time, and Channel groups remained useful for contextual interpretation.

The influence experiments produced closely comparable spreads among the four seed-selection methods. IM Greedy achieved small positive gains in all six communities, despite limited seed overlap with the baselines. These findings suggest that distinct seed configurations can achieve comparable difusion reach under the adopted IC settings.

Overall, ODCBNA establishes a traceable connection between heterogeneous corpus records, semantic relationships, behavioral representations, network structure, and influence propagation. The findings suggest that cyberbullying analysis and intervention should consider behavioral, contextual, and structural evidence rather than relying solely on textual content, user activity, or individual centrality measures.

Future work will examine learned and hybrid representations, parameter sensitivity, temporal network modeling, and sub-community-level analysis. It will also incorporate semantically linked external data descriptions to extend the analytical capabilities of ontologies and knowledge graphs. Further evaluation using other languages, topics, time periods, and social platforms will assess the framework’s broader generalizability.

## CRediT authorship contribution statement

Md. Nesarul Hoque: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Md Hanif Seddiqui: Writing – review & editing, Supervision, Resources, Methodology, Conceptualization. Miskatul Anwar: Writing – review & editing, Visualization, Validation, Software. Rudra Pratap Deb Nath: Writing – review & editing, Writing – original draft, Supervision, Resources, Project administration, Methodology, Formal analysis, Investigation, Funding acquisition, Conceptualization.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

[1] Associated Press, 2024. French prosecutors investigate gender-based cyber harassment of algerian olympic champ imane khelif. https: //apnews.com/article/paris- olympics-khelif-cyber- harassment-b oxing-algeria-investigation-cfc710d52ec0830936b0b3b0758fbf0f. Accessed: 2026-07-10.

[2] Atif, A., Zafar, A., Wasim, M., Waheed, T., Ali, A., Ali, H., Shah, Z., 2024. Cyberbullying detection and abuser profile identification on social media for roman urdu. IEEE Access 12, 123339–123351.

URL: http://dx.doi.org/10.1109/ACCESS.2024.3445288, doi:10.1109/ access.2024.3445288.

[3] Baader, F., McGuinness, D., Nardi, D., Patel-Schneider, P., 2007. The Description Logic Handbook: Theory, Implementation and Applications. volume 2. Cambridge University Press. URL: http://dx.doi.o rg/10.1017/CBO9780511711787, doi:10.1017/cbo9780511711787.

[4] Blondel, V.D., Guillaume, J.L., Lambiotte, R., Lefebvre, E., 2008. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment 2008, P10008. URL: http: //dx.doi.org/10.1088/1742-5468/2008/10/P10008, doi:10.1088/17 42-5468/2008/10/p10008.

[5] Cheng, J., Bernstein, M., Danescu-Niculescu-Mizil, C., Leskovec, J., 2017. Anyone can become a troll: Causes of trolling behavior in online discussions, in: Proceedings of the 2017 ACM Conference on Computer Supported Cooperative Work and Social Computing, ACM. p. 1217–1230. URL: http://dx.doi.org/10.1145/2998181.299 8213, doi:10.1145/2998181.2998213.

[6] Chun, J., Lee, S., Kim, J., 2024. Ontology development for cyber violence victimization in korean adolescents. Sage Open 14. URL: http://dx.doi.org/10.1177/21582440241250152, doi:10.1177/21582440 241250152.

[7] Cuzzocrea, A., Akter, M.S., Shahriar, H., García Bringas, P., 2025. Cyberbullying detection, prevention, and analysis on social media via trustable lstm-autoencoder networks over synthetic data: The tla-net approach. Future Internet 17, 84. URL: http://dx.doi.org/10.3390/f i17020084, doi:10.3390/fi17020084.

[8] Cyganiak, R., Wood, D., Lanthaler, M., 2014. RDF 1.1 Concepts and Abstract Syntax. W3C Recommendation. World Wide Web Consortium. URL: https://www.w3.org/TR/rdf11-concepts/. available online.

[9] Deb Nath, R.P., 2020. Aspects of semantic etl. Aalborg University.

[10] Deb Nath, R.P., Hose, K., Pedersen, T.B., Romero, O., 2017. Setl: A programmable semantic extract-transform-load framework for semantic data warehouses. Information Systems 68, 17–43. URL: http://dx.doi.org/10.1016/j.is.2017.01.005, doi:10.1016/j.is.2 017.01.005.

[11] Deb Nath, R.P., Romero, O., Pedersen, T.B., Hose, K., 2021. Highlevel etl for semantic data warehouses. Semantic Web 13, 85–132. URL: http://dx.doi.org/10.3233/SW-210429, doi:10.3233/sw-210429.

[12] Easley, D., Kleinberg, J., 2010. Networks, Crowds, and Markets: Reasoning about a Highly Connected World. Cambridge University

Press. URL: http://dx.doi.org/10.1017/CBO9780511761942, doi:10.1017/cbo9780511761942.

[13] Ejaz, N., Razi, F., Choudhury, S., 2024. Towards comprehensive cyberbullying detection: A dataset incorporating aggressive texts, repetition, peerness, and intent to harm. Computers in Human Behavior 153, 108123. URL: http://dx.doi.org/10.1016/j.chb.2023.108123, doi:10.1016/j.chb.2023.108123.

[14] Fortuna, P., Nunes, S., 2018. A survey on automatic detection of hate speech in text. ACM Computing Surveys 51, 1–30. URL: http://dx.doi.org/10.1145/3232676, doi:10.1145/3232676.

[15] Grigaliunas, Š., Br ¯ uzgien ¯ e, R., Venčkauskas, A., 2023. Ontology- ˙ driven digital profiling for identification and linking evidence across social media platform. IEEE Access 11, 111672–111691. URL: http://dx.doi.org/10.1109/ACCESS.2023.3322162, doi:10.1109/ac cess.2023.3322162.

[16] Hinduja, S., Patchin, J.W., 2010. Bullying, cyberbullying, and suicide. Archives of Suicide Research 14, 206–221. URL: http://dx.doi.org /10.1080/13811118.2010.494133, doi:10.1080/13811118.2010.494133.

[17] Hoque, M.N., Chakraborty, P., Seddiqui, M.H., 2023. The challenges and approaches during the detection of cyberbullying text for low-resource language: A literature review. ECTI Transactions on Computer and Information Technology (ECTI-CIT) 17, 192–214. doi:10.37936/ecti-cit.2023172.248039.

[18] Hoque, M.N., Deb Nath, R.P., Chy, A.N., Ghose, D., Seddiqui, M.H., 2026a. Advancing cyberbullying detection in low-resource languages: a transformer- stacking framework for bengali. Frontiers in Artificial Intelligence 8. URL: http://dx.doi.org/10.3389/frai.2025.1679962, doi:10.3389/frai.2025.1679962.

[19] Hoque, M.N., Nath, R.P.D., Salma, U., Chakraborty, S., Seddiqui, M.H., 2026b. Bencodic: A large-scale contextual cyberbullying corpus for bengali political news discourse full version. URL: https: //arxiv.org/abs/2610.13121, doi:10.48550/ARXIV.2610.13121.

[20] Hoque, M.N., Seddiqui, M.H., 2024. Exploring transformer ensemble approach to classify cyberbullying text for the low-resource bengali language, in: 2024 International Conference on Advances in Computing, Communication, Electrical, and Smart Systems (iCACCESS), IEEE. p. 1–6. URL: http://dx.doi.org/10.1109/iCACCESS61735.2024. 10499469, doi:10.1109/icaccess61735.2024.10499469.

[21] Kempe, D., Kleinberg, J., Tardos, É., 2003. Maximizing the spread of influence through a social network, in: Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and data

mining, ACM. p. 137–146. URL: http://dx.doi.org/10.1145/95675 0.956769, doi:10.1145/956750.956769.

[22] Kowalski, R.M., Giumetti, G.W., Schroeder, A.N., Lattanner, M.R., 2014. Bullying in the digital age: A critical review and meta-analysis of cyberbullying research among youth. Psychological Bulletin 140, 1073–1137. URL: http://dx.doi.org/10.1037/a0035618, doi:10.1037/ a0035618.

[23] Kumari, C., Kaur, M., 2025. Towards enhanced cyberbullying detection: A unified framework with transfer and federated learning. Systems 13, 818. URL: http://dx.doi.org/10.3390/systems13090818, doi:10.3390/systems13090818.

[24] Maity, K., Sen, T., Saha, S., Bhattacharyya, P., 2024. Mtbullygnn: A graph neural network-based multitask framework for cyberbullying detection. IEEE Transactions on Computational Social Systems 11, 849–858. URL: http://dx.doi.org/10.1109/TCSS.2022.3230974, doi:10.1109/tcss.2022.3230974.

[25] Mali, M.K., Pawar, R.R., Shinde, S.A., Kale, S.D., Mulik, S.V., Jagtap, A.A., Tambewagh, P.A., Rajput, P.U., 2025. Automatic detection of cyberbullying behaviour on social media using stacked bi-gru attention with bert model. Expert Systems with Applications 262, 125641. URL: http://dx.doi.org/10.1016/j.eswa.2024.125641, doi:10.1016/j.eswa.2024.125641.

[26] Mishra, A., Sinha, S., George, C.P., 2024. Shielding against online harm: A survey on text analysis to prevent cyberbullying. Engineering Applications of Artificial Intelligence 133, 108241. URL: http: //dx.doi.org/10.1016/j.engappai.2024.108241, doi:10.1016/j.en gappai.2024.108241.

[27] Newman, M.E.J., 2006. Modularity and community structure in networks. Proceedings of the National Academy of Sciences 103, 8577–8582. URL: http://dx.doi.org/10.1073/pnas.0601602103, doi:10.1073/pnas.0601602103.

[28] Nitya Harshitha, T., Prabu, M., Suganya, E., Sountharrajan, S., Bavirisetti, D.P., Gadde, N., Uppu, L.S., 2024. Protect: a hybrid deep learning model for proactive detection of cyberbullying on social media. Frontiers in Artificial Intelligence 7. doi:10.3389/frai.2024. 1269366.

[29] Page, L., Brin, S., Motwani, R., Winograd, T., 1999. The pagerank citation ranking : Bringing order to the web, in: The Web Conference. URL: https://api.semanticscholar.org/CorpusID:1508503.

[30] Patchin, J.W., Hinduja, S., 2015. Measuring cyberbullying: Implications for research. Aggression and Violent Behavior 23, 69–74. URL:

http://dx.doi.org/10.1016/j.avb.2015.05.013, doi:10.1016/j.avb.20 15.05.013.

[31] Philipo, A.G., Sarwatt, D.S., Ding, J., Daneshmand, M., Ning, H., 2026. Cyberbullying detection: Exploring datasets, technologies, and approaches on social media platforms. ACM Computing Surveys 58, 1–35. URL: http://dx.doi.org/10.1145/3785654, doi:10.1145/378565 4.

[32] Ribeiro, M., Calais, P., Santos, Y., Almeida, V., Meira Jr., W., 2018. Characterizing and detecting hateful users on twitter. Proceedings of the International AAAI Conference on Web and Social Media 12. URL: h t tp : / / d x . d o i . o r g / 1 0 . 1 6 0 9 / i c w s m . v 1 2 i 1 . 1 5 0 57, doi:10.1609/icwsm.v12i1.15057.

[33] Shi, C., Li, Y., Zhang, J., Sun, Y., Yu, P.S., 2017. A survey of heterogeneous information network analysis. IEEE Transactions on Knowledge and Data Engineering 29, 17–37. URL: http://dx.doi.o rg/10.1109/TKDE.2016.2598561, doi:10.1109/tkde.2016.2598561.

[34] Singhal, A., et al., 2001. Modern information retrieval: A brief overview. IEEE Data Engineering Bulletin 24, 35–43.

[35] Wang, R., Lu, T., Zhang, P., Gu, N., 2025. Role identification based method for cyberbullying analysis in social edge computing. Tsinghua Science and Technology 30, 1659–1684. URL: http://dx.doi.org/1 0.26599/TST.2024.9010066, doi:10.26599/tst.2024.9010066.

[36] Watts, D.J., Strogatz, S.H., 1998. Collective dynamics of ‘smallworld’ networks. Nature 393, 440–442. URL: http://dx.doi.org /10.1038/30918, doi:10.1038/30918.

[37] Xiong, S., Liu, W., Wang, B., Che, Y., Shi, L., 2025. Topic adversarial neural network for cross-topic cyberbullying detection. PeerJ Computer Science 11, e2942. URL: http://dx.doi.org/10.7717/pee rj-cs.2942, doi:10.7717/peerj-cs.2942.