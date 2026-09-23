# Discovery-Driven Integration of Disjoint Tables via Text

Md Ataur Rahman   
UPC, BarcelonaTech   
Barcelona, Spain   
md.ataur.rahman@upc.edu

Dimitris Sacharidis Université Libre de Bruxelles Brussels, Belgium dimitris.sacharidis@ulb.be

Oscar Romero   
UPC, BarcelonaTech   
Barcelona, Spain   
oscar.romero@upc.edu

Sergi Nadal UPC, BarcelonaTech Barcelona, Spain sergi.nadal@upc.edu

## Abstract

Integrating heterogeneous datasets within data lakes is a critical challenge, particularly for semantically related tables that lack the explicit attributes needed to be joined. We study Discovery-Driven Integration, where the relevant sources and their missing relational structure must be discovered before integration. In this setting, unstructured text provides the evidence that connects otherwise disjoint tables. The fundamental challenge is to discover the relationships at a fine-grained level that connect individual rows from diferent tables through specific sentences. We formalize this task as Text-Mediated Join Path Discovery and propose a horizontal bidirectional cross-attention architecture called LOKI (Latent-space Optimization for Knowledge Integration)<sup>1</sup> that learns contextualized representations of table rows and sentences. Through a global table-text contrastive objective, fine-grained rowsentence associations emerge without explicit local supervision. Existing multi-modal discovery methods largely retrieve coarsegrained column-text associations, whereas integration systems assume supplied row-text links, schemas, or queries. LOKI instead transforms these implicit associations into explicit, interpretable join paths, organizes them into relation-consistent groups, and materializes them as typed integrated tables with sentence-level provenance. Comprehensive evaluations on real-world benchmarks demonstrate that LOKI consistently outperforms state-of-the-art multi-modal data discovery approaches, and materializes typed integrated tables with 0.982 macro typed-pair precision while being up to 40 times cheaper in LLM API cost than direct prompting.

## 1 Introduction

Data lakes serve as centralized repositories consolidating large volumes of heterogeneous data, often containing thousands of datasets with structured tables and large collections of unstructured text documents. Information embedded in text contains valuable insights that remain inaccessible unless extracted and integrated with structured data: in a hospital data lake, for example, discharge summaries may link patient records to medication treatments. Extracting and integrating such information makes it available for analysis.

Recently, several methods have been proposed for text-table querying, allowing users to retrieve information from both modalities, e.g., [19, 30, 33, 34]. However, these methods share a key limitation: they assume that the integration between the two modalities is already known. In particular, they require knowledge of (1) the row-text links, $\mathrm { i . e . , }$ , which part of the text is associated to each table row, and (2) the text schema, i.e., what information is contained in this part of the text. Essentially, they perform query-time extraction over a predefined text-table integration. For example, ELEET [33] processes queries that specify a latent attribute to be extracted from text for each table row, while ThalamusDB [19] processes queries containing a Boolean predicate expressed in natural language over a text document. The latent attribute in the former and the predicate in the latter specify text schema, while the association between text and table rows is assumed to be known.

Motivating Example. Consider the hospital data lake in Fig. 1, with tables such as the Patients table $T _ { 1 }$ and the Medications table $T _ { 2 } ,$ and text documents such as the Discharge Summaries $D _ { 1 }$ and $D _ { 2 } .$ Assume that (1) $D _ { 1 }$ concerns patient $I _ { 1 }$ and $D _ { 2 }$ concerns patient $I _ { 2 } ,$ and (2) Diagnosis information appears in the discharge summaries and must be extracted per patient. Under these assumptions, the Diagnosis table $T _ { a }$ can be extracted from the text and linked to the Patients table $T _ { 1 } ,$ enabling querying across the text and tables.

To fully extract value from data lakes, we must remove these two assumptions and enable text-table integration, where neither rowtext associations nor text schemas are known beforehand. Existing multi-modal discovery and retrieval techniques [2, 8, 10, 11] identify only coarse-grained associations between columns (or tables) and text, missing fine-grained row-text links. This is insuficient for integration in two ways. It can collapse semantically distinct relationships: a discharge summary may state that a patient was prescribed aspirin, had an adverse reaction, and later discontinued it (three relationships a coarse-grained association cannot distinguish). It can also introduce spurious ones (e.g., linking a patient to every medication merely discussed rather than prescribed).

Running Example. On the example above, coarse-grained discovery may identify, for instance, that the $T _ { 1 }$ .���� column of the Patients table $T _ { 1 }$ is related to the discharge summaries $D _ { 1 }$ and $D _ { 2 } .$ Fine-grained discovery, in contrast, identifies row-text links, such as the links between the first row of $T _ { 1 ; }$ , corresponding to patient $I _ { 1 } ,$ and sentences $s _ { 1 } - s _ { 5 }$ in $D _ { 1 } .$

To the best of our knowledge, no existing method automatically discovers both the fine-grained row-text associations and the schemas needed to integrate text with existing tables. In this work, we present LOKI, a system that discovers such associations, starting from fine-grained row-text links: two links with overlapping text spans and rows from diferent tables indicate that the text expresses a relationship between them. We call the resulting sequence of a table row, a supporting text span, and another table row a textmediated join path. LOKI collects the join paths for each pair of tables, examining their supporting text to determine whether they express one relationship or several, each defining a text schema.

![](images/8e9bfde441f3802f3dd002b93a9f6b34c8ccac8febc481b7a222fc8b981d6d63.jpg)  
Figure 1: Discovery-driven integration within a multi-modal data lake. Coarse-grained approaches retrieve table-text pairs, and integration systems require supplied row-text links and predefined text schema. LOKI instead derives document relevance from fine-grained row-sentence associations and uses them to disambiguate and materialize evidence-backed cross-table relations without a predefined relationship type or target schema.

Overview of LOKI. At the heart of LOKI is a cross-modal encoder that takes a table and a text document as input and produces embeddings for each table row and document sentence. The encoder combines a frozen text encoder with a trainable bidirectional attention mechanism that contextualizes each row embedding with information from the document and each sentence embedding with information from the table: rows attend to sentences in the forward direction, and sentences attend to rows in the reverse. LOKI uses the resulting contextualized embeddings to compute a row-sentence pair-score matrix that captures fine-grained associations between rows and sentences. LOKI does not require costly row-sentence annotations for training. Instead, it relies on a small number of potentially noisy coarse table-document annotations, which can be obtained from weak signals such as syntactic or embedding-based similarity, or weak-supervision techniques [31], as done in [8, 36].

LOKI trains the encoder contrastively on triples $( T , D ^ { + } , D ^ { - } )$ with two objectives: a global one that ranks $D ^ { + }$ above �<sup>−</sup> for table � from the aggregated pair scores, and a local one that, drawing on multiple-instance learning [18, 27], separates a sampled row’s bestmatching sentence in $D ^ { + }$ from its best in $D ^ { - }$ . Together they induce fine-grained associations from coarse supervision. Once trained,

LOKI discovers row-text links and text-mediated join paths: for each document, it identifies coarsely related tables, considers pairs of them, and uses the overlapping supporting text spans of their row-text links to construct join paths. LOKI encodes and groups the resulting paths using density-based clustering, where each cluster represents a distinct relationship between a pair of tables. An LLM names each relationship based on the evidence sentences, and LOKI creates one table per cluster, populated with the join paths.

Running Example. Continuing the example, fine-grained discovery identifies that sentences such as � in � relate patient �<sub>1</sub> in �<sub>1</sub> and medication �<sub>1</sub> in �<sub>2</sub>, revealing the textmediated join path $( I _ { 1 } , D _ { 1 } . s _ { 1 } , M _ { 1 } ) \colon$ a relationship between $I _ { 1 }$ and $M _ { 1 }$ mediated by $D _ { 1 } . s _ { 1 }$ . LOKI collects such join paths per table pair and groups them by the relationships expressed in their supporting text. For $T _ { 1 }$ and $T _ { 2 }$ , this reveals three relationships between patients and medications: $( I _ { 1 } , D _ { 1 } . s _ { 1 } , M _ { 1 } )$ expresses a Prescribed\_For, $\left( I _ { 1 } , D _ { 1 } . s _ { 2 } , M _ { 1 } \right)$ an Adverse\_Efect, and $\left( I _ { 1 } , D _ { 1 } . s _ { 3 } , M _ { 1 } \right)$ a Discontinued relationship. LOKI therefore creates three distinct text schemas, one per relationship, and populates the tables with the discovered instances.

Contributions. This work makes the following contributions to multi-modal discovery and querying:

• Novel problem. We introduce Discovery-Driven Integration, the setting in which relationships between tables mediated by text documents are unknown, and formalize its core task as Text-Mediated Join Path Discovery. To the best of our knowledge, no prior method jointly discovers, disambiguates, and materializes latent row-sentence-row relationships, into integrated tables.

• Joint representation-learning architecture. We introduce the LOKI encoder, which learns contextualized row-text representations from coarse table-text supervision alone, enabling finegrained associations to emerge without explicit row-sentence annotations. Unlike existing column-based approaches that use vertical attention within a table column [8, 36], LOKI uses horizontal bidirectional cross-attention between rows and sentences.

• SOTA Performance. We evaluate LOKI on an end-to-end integration task using real-world datasets, bridging disjoint tables through fine-grained row-sentence-row evidence and materializing typed relations at a fraction of the LLM token cost. LOKI also outperforms existing cross-modal discovery and representationlearning methods at coarse-grained discovery.

## 2 Related Work

We situate LOKI at the intersection of multi-modal data discovery and integration: existing discovery systems retrieve related tables or table-document pairs, while integration systems operate over predefined links, schemas, or queries. LOKI bridges the two through discovery-driven integration, using text to discover and materialize relationships between disjoint tables.

Join-Path Discovery. Our definition of Text-Mediated Join Path introduces a row-to-row association mediated by natural language sentences, whereas prior work defined join paths primarily at the table or schema level. In systems such as Metam [12], Data Civilizer [25], Aurum [9], and SemDisc [28], join paths denote ordered chains ofjoinable datasets used for data discovery or augmentation, but these remain purely structural, without textual mediation or record-level grounding. Ver [6, 14] extends this notion to schemaagnostic view discovery with no predefined foreign keys. The novelty of our formulated join path lies in its formulation as an explicit fine-grained transitive bridge through mediating sentences as evidence of latent relationships. This also difers from multi-modal discovery systems that stop at one-hop table-document retrieval [8].

Semantic Table Discovery. Current research moves beyond syntactic value overlaps to capture semantic relatedness between tables. DeepJoin [7] and Snoopy [15] fine-tune Transformer-based models to encode column values, identifying joinable columns based on semantic proximity rather than exact matches. FREYJA [26] combines data profiling with predictive modeling to approximate semantics-aware joins, while OmniMatch [24] constructs a graph of columns to propagate similarity signals. SANTOS [22] retrieves unionable tables using semantic relationships between column pairs from an external or synthesized knowledge base, while KGLiDS [16] captures dataset semantics with learned data profiles and knowledge graphs. Despite their diferent mechanisms, these approaches derive their discovery signals from table values, profiles, metadata, or semantics constructed from tabular artifacts, and cannot discover relationships between disjoint tables when the connecting evidence exists only in external text, which LOKI instead exploits as a semantic bridge to discover row-level relationships.

Joint Representation Learning. To bridge the gap between tables and text, prior work learns a unified embedding space. SEMPROP [11] and Termite [10] project structured and unstructured data into a shared latent space to retrieve related items based on vector proximity. TABERT [36] and PNEUMA [4] linearize tables to learn joint representation with text, often for Question Answering (QA) or Retrieval-Augmented Generation (RAG) tasks. These approaches operate at a coarse schema or document granularity: they function as semantic search engines rather than integration systems and do not extract explicit, row-level join paths, whereas LOKI resolves fine-grained row-sentence associations, transforming black-box retrieval scores into interpretable, transitive join paths.

Multi-Modal Data Discovery. Several systems retrieve related data across structured and unstructured data. OTTER [17] jointly retrieves tables and text, but its retrieval unit is a table row prelinked to text passages via entity linking, relying on existing links rather than discovering them. CMDL [8] and Tri-Encoder [23] support data-lake queries over structured and unstructured data, returning ranked table- or column-document associations. Tab-STAR [3] jointly represents structured attributes and text fields for tabular prediction, also supporting cross-modal retrieval. These systems narrow the candidate sources, but their coarse-grained outputs do not determine which rows connect, which sentences support the connection, or which relationships they express; using them directly for integration may produce spurious joins, e.g., linking a patient to every medication in the same summary. LOKI instead composes fine-grained row-sentence associations into evidencebacked row-sentence-row paths for relation-specific integration.

Schema/Query-Driven Integration. Existing multi-modal integration systems assume a predefined relational structure. THOR [30] populates values for a target concept from related text. ELEET [33] extracts registered attributes into latent tables and composes them with relational operators. Its multi-modal join assumes table tuples are already linked to documents. ThalamusDB [19] evaluates explicit natural-language predicates within SQL queries over a known multi-modal schema, whereas iDataLake [34] uses an analytical query to select sources and construct an execution plan. ConnectionLens [2] integrates heterogeneous sources by extracting and linking entities from text and other sources. These systems operate once the required links, schemas, or queries are supplied, but do not determine which cross-table relations should exist when both connections and relationship types are unknown. LOKI, in contrast, discovers row-sentence-row paths, disambiguates their semantics, and materializes them as evidence-backed tables without a predefined schema.

## 3 The LOKI Encoder

We present the LOKI encoder (Fig. 2) and discuss its input, its contextualization of rows and sentences, its output, and its training.

![](images/e5e9f42712833c1d9c06d353d3a68229bbfb85a99a8c08a01fea3ae5a0280a95.jpg)  
Figure 2: LOKI Encoder for multi-modal representation learning. Given a candidate table-document pair (�, �), the frozen encoder � embeds the linearized table rows and document sentences as R and S. Bidirectional cross-attention, parameterized by shared LoRA-adapted projections, produces contextualized embeddings eR and eS and the attention distributions $\mathbf { A } _ { f }$ and $\mathbf { A } _ { r } .$ Their cosine similarities form the pair-score matrix $\mathbf { P } ,$ whose strongest entries capture fine-grained row-sentence associations and whose Top-� aggregation yields the global similarity Sim(�, �).

## 3.1 Input and Initial Embeddings

The LOKI encoder takes as input a text � and a table �, where � is converted into text by a row-based linearization. Specifically, each row �<sub>�</sub> from table � is first transformed into a natural language string. The linearization concatenates the column names and their cell values for that row:

$$
\operatorname { l i n e a r i z e } ( T ) = [ ^ { \ast } c _ { 1 } { : } v _ { i 1 } ; c _ { 2 } { : } v _ { i 2 } ; \cdot \cdot \cdot ; c _ { \ell } { : } v _ { i \ell } { } ^ { , } ] _ { i = 1 } ^ { n }
$$

where ℓ is the number of columns of�. For instance, the first row of the Patients table $T _ { 1 }$ is serialized into the string: “PatientID: $I _ { 1 } ;$ Name: Alice; Age: 34; Sex: F”. This lets LOKI treat structured data as natural language, consumable by attention modules alongside documents. LOKI feeds text � and linearized table � into a pre-trained sentence encoder, �(·). While token-level encoders return a contextualized vector per token, sentence encoders such as SBERT [32] aggregate these into a single fixed-dimensional vector per linearized row and sentence:

$$
\mathbf { R } = [ \xi ( \mathrm { l i n e a r i z e } ( T ) ) ] ^ { \top } \in \mathbb { R } ^ { n \times d } ; \quad \mathbf { S } = [ \xi ( s _ { 1 } ) , \ldots , \xi ( s _ { m } ) ] ^ { \top } \in \mathbb { R } ^ { m \times d } .
$$

Here � is the number of rows of�, � the number of sentences in $D ,$ and � the embedding dimension of �. This yields two matrices R and S of initial, non-contextualized embeddings that capture the general meaning of each table row (e.g., the record for Alice) and each sentence in isolation, without awareness of the other modality. The weights of the base encoder � are kept frozen to leverage its general-purpose linguistic knowledge.

## 3.2 Contextualized Embeddings

The distinguishing characteristic of the LOKI encoder is its bidirectional cross-attention module. Unlike traditional unidirectional attention, it is designed around the premise that cross-modal association requires a symmetric exchange of information: a table row’s meaning is grounded by the sentences that mention its entities, and a sentence’s relevance is determined by the referenced tabular data. This stage produces contextualized cross-modal representations.

The module consists of two parallel cross-attention blocks, stabilized by a gated attention mechanism [29] that helps prevent attention collapse. We retain the Forward Attention Matrix $\mathbf { A } _ { f }$ and Reverse Attention Matrix $\mathbf { A } _ { r }$ as the internal attention distributions of these blocks. For a given set of queries (Q), keys (K), and values (V), the attention matrix A and its output H are computed as:

$$
\begin{array} { l } { \displaystyle { \bf A } ( { \bf Q } , { \bf K } ) = \mathrm { s o f t m a x } \left( \frac { ( { \bf Q } { \bf W } _ { Q } ) ( { \bf K } { \bf W } _ { K } ) ^ { \top } } { \sqrt { d _ { k } } } \right) ; } \\ { \displaystyle { \bf H } ( { \bf Q } , { \bf K } , { \bf V } ) = { \bf A } ( { \bf Q } , { \bf K } ) ( { \bf V } { \bf W } _ { V } ) } \end{array}
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { K }$ , and $\mathbf { W } _ { V }$ are the query, key, and value projections, shared between the forward and reverse cross-attention blocks, and $d _ { k }$ is the dimensionality of the projected keys. For parametereficient adaptation, we apply Low-Rank Adaptation (LoRA) to the shared projections while keeping the base encoder $\xi$ frozen, so cross-attention learns task-specific row-sentence interactions without modifying the encoder’s general-purpose representations.

Forward Attention (Rows → Sentences). As depicted in Fig. 2 (top), table rows act as queries to seek information from sentences, which serve as keys and values. This produces contextualized row representations eR, enriched by the relevant sentence context:

$$
\mathbf { A } _ { f } = \mathbf { A } ( \mathbf { R } , \mathbf { S } ) ; \quad \mathbf { H } _ { f } = \mathbf { H } ( \mathbf { R } , \mathbf { S } , \mathbf { S } ) ; \quad \widetilde { \mathbf { R } } = \mathbf { R } + \mathbf { H } _ { f } \odot \sigma ( \mathbf { R } \mathbf { W } _ { g } ^ { f } )
$$

This operation contextualizes each row using the sentences most relevant to it. In our running example, the Alice row attends most strongly to the sentence $s _ { 1 }$ describing her Vancomycin prescription, producing a sentence-aware representation in eR. The residual connection preserves the row’s original identity.

Reverse Attention (Sentences → Rows). As depicted in Fig. 2 (bottom), it uses sentences as queries to attend to the table rows. This process produces the contextualized sentence representations ${ \widetilde { \mathsf { S } } } ,$ enriched by the relevant row context:

$$
\mathbf { A } _ { r } = \mathbf { A } ( \mathbf { S } , \mathbf { R } ) ; \quad \mathbf { H } _ { r } = \mathbf { H } ( \mathbf { S } , \mathbf { R } , \mathbf { R } ) ; \quad \widetilde { \mathbf { S } } = \mathbf { S } + \mathbf { H } _ { r } \odot \sigma ( \mathbf { S } \mathbf { W } _ { g } ^ { r } )
$$

where $\sigma ( \cdot )$ is the logistic sigmoid and ⊙ denotes element-wise multiplication. Here, $\mathbf { W } _ { g } ^ { f }$ and $\mathbf { W } _ { g } ^ { r }$ are separately learned gating parameters for the forward and reverse directions. This operation contextualizes each sentence with the rows it references. The sentence $s _ { 1 }$ describing Alice’s Vancomycin prescription attends most strongly to the corresponding patient and medication rows, producing a row-aware representation in ${ \widetilde { \mathsf { S } } } .$ Together, the two directions reduce ambiguity by conditioning each modality on the other.

## 3.3 Output

The primary outputs of the LOKI encoder are the contextualized representations eR and ${ \widetilde { \mathsf { S } } } ,$ from which LOKI computes two secondary outputs: a fine-grained row-sentence similarity matrix and a coarsegrained table-text similarity score.

Row-Sentence Similarity Matrix. The fine-grained association between individual rows and sentences is captured in the matrix $\mathbf { P } \in \mathbb { R } ^ { n \times m }$ , computed using cosine similarity between their contextualized representations, where $r \in \{ 1 , \ldots , n \}$ indexes the table rows and $s \in \{ 1 , \ldots , m \}$ the sentences:

$$
P _ { r s } = \frac { \langle \widetilde { \bf R } _ { r } , \widetilde { \bf S } _ { s } \rangle } { \Vert \widetilde { \bf R } _ { r } \Vert _ { 2 } \cdot \Vert \widetilde { \bf S } _ { s } \Vert _ { 2 } }
$$

The pair-score matrix preserves the row-sentence granularity required for join-path extraction. In our running example, the Alice row receives a high score for sentence $s _ { 1 }$ describing her prescription and lower scores with sentences concerning unrelated patients.

Table-Text Similarity. For the global retrieval task and loss calculation, P is aggregated into a scalar similarity score, Sim(�, �). Empirically, we adopt Top-� sum pooling with $K _ { \mathrm { p o o l } } = 5$ by default:

$$
\mathrm { S i m } ( T , D ) = \sum _ { ( r , s ) \in \mathrm { T o p K } ( \mathbf { P } ; K _ { \mathrm { p o o l } } ) } P _ { r s }
$$

Here, $\mathrm { T o p K } ( \mathbf { P } ; K _ { \mathrm { p o o l } } )$ denotes the index pairs (�, �) corresponding to the $K _ { \mathrm { p o o l } }$ largest entries of P. Aggregating only the strongest rowsentence links avoids relying on a single pair, as in max-pooling, prevents dilution by mean-pooling, and concentrates the global ranking gradient on the most relevant local interactions, directly coupling table-text retrieval with fine-grained row-sentence association.

## 3.4 Training

LOKI’s training (Fig. 3) uses a composite loss over triplets $( T , D ^ { + } , D ^ { - } )$ where $D ^ { + }$ is a relevant document for table � and $D ^ { - }$ is an unrelated or hard-negative document. The total loss jointly optimizes five objectives:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { g l o b } } \mathcal { L } _ { \mathrm { g l o b } } + \lambda _ { \mathrm { l o c } } \mathcal { L } _ { \mathrm { l o c } } + \lambda _ { \mathrm { d i s t } } \mathcal { L } _ { \mathrm { d i s t } } + \lambda _ { \mathrm { s i g } } \mathcal { L } _ { \mathrm { s i g } } + \lambda _ { \mathrm { s i n k } } \mathcal { L } _ { \mathrm { s i n k } }
$$

Here, $\mathcal { L } _ { \mathrm { g l o b } }$ drives coarse-grained table-document retrieval from aggregated pair scores, $\mathcal { L } _ { \mathrm { l o c } }$ sharpens row-sentence separation within the pair-score matrix, and $\mathcal { L } _ { \mathrm { d i s t } }$ aligns LOKI’s pair-score distributions with the frozen encoder’s. $\mathcal { L } _ { \mathrm { s i g } }$ and $\mathcal { L } _ { \mathrm { s i n k } }$ regularize the contextualized representations and attention maps, and the � coefficients weight each term.

![](images/77cb58527326682a6bd4781365b85953012b58fc6c087291e149610d1d83c6c0.jpg)  
Figure 3: LOKI training pipeline, from coarse table-document supervision to fine-grained row-sentence association.

Global Table-Text Loss $( \mathcal { L } _ { \mathrm { g l o b } } ) .$ . The global objective encourages the positive table-document pair to receive a higher aggregated similarity than the negative pair:

$$
\mathcal { L } _ { \mathrm { g l o b } } = - \log \frac { \exp ( \operatorname { S i m } ( T , D ^ { + } ) / \tau ) } { \exp ( \operatorname { S i m } ( T , D ^ { + } ) / \tau ) + \exp ( \operatorname { S i m } ( T , D ^ { - } ) / \tau ) }
$$

where � is a temperature parameter that controls the sharpness of the contrastive comparison between positive and negative tabledocument pairs; we used $\tau = 0 . 2$ by default. Since Sim(�, �) is aggregated from the pair-score matrix, this objective propagates table-text supervision to the strongest row-sentence interactions.

Local Row-Sentence Loss $( \mathcal { L } _ { \mathrm { l o c } } ) .$ . To sharpen fine-grained association, we contrast a sentence $s ^ { + }$ from the positive document with a sentence $s ^ { - }$ from the negative document, and expect the former to have a stronger similarity to a row � compared to the latter. Let $\Omega = \{ r , s ^ { + } , s ^ { - } \}$ denote the constrasting triples; the set is constructed as follows. We first sample a row � that has the highest similarity with some sentence in $D ^ { + }$ . Then, we sample a sentence $s ^ { + }$ from $D ^ { + }$ among those with the highest similarity to �. Finally, we sample a sentence �<sup>−</sup> from �<sup>−</sup>. We define the following local objective:

$$
\mathcal { L } _ { \mathrm { l o c } } = \frac { 1 } { | \Omega | } \sum _ { ( r , s ^ { + } , s ^ { - } ) \in \Omega } \operatorname* { m a x } \left. 0 , m _ { \mathrm { l o c } } - \left( P _ { r s ^ { + } } ^ { + } - P _ { r s ^ { - } } ^ { - } \right) \right.
$$

Using a local margin of $m _ { \mathrm { l o c } } = 0 . 3$ by default, the loss encourages row-sentence interactions in the positive context to score higher than those in the negative context by at least this margin. The resulting loss is averaged over the training triplets. This term sharpens the final pair-score matrix without requiring any ground-truth rowsentence association labels and complements the global ranking loss by enforcing local discriminability between positive and negative contexts.

Distillation Loss $( { \mathcal { L } } _ { \mathrm { d i s t } } )$ . Because the global ranking objective uses Top-� aggregation, its supervision is concentrated on the strongest row-sentence pairs. We therefore distill the broader similarity structure of the frozen encoder into LOKI’s pair-score matrix $\mathbf { P } ;$ the teacher scores are simply the cosine similarities between the original non-contextualized embeddings R and S, requiring no separate teacher model. To reduce the influence of generic sentences that are similar to many rows, we center each teacher score by the mean

score received by its sentence:

$$
P _ { r s } ^ { \xi } = \frac { \langle \mathbf { R } _ { r } , \mathbf { S } _ { s } \rangle } { \| \mathbf { R } _ { r } \| \cdot \| \mathbf { S } _ { s } \| } ; \qquad \widehat { P } _ { r s } ^ { \xi } = P _ { r s } ^ { \xi } - \frac { 1 } { n } \sum _ { r ^ { \prime } = 1 } ^ { n } P _ { r ^ { \prime } s } ^ { \xi }
$$

We then construct two conditional distributions from the centered teacher scores and the contextualized student scores:

$$
\mathbf { D } _ { \mathbf { R }  \mathbf { S } } ^ { \xi } = { \mathrm { s o f t m a x } } _ { s } ( \frac { \widehat { \mathbf { P } } ^ { \xi } } { \tau _ { t } } ) ; \quad \mathbf { D } _ { \mathbf { R }  \mathbf { S } } ^ { \mathrm { L O K I } } = { \mathrm { s o f t m a x } } _ { s } ( \frac { \mathbf { P } } { \tau _ { s } } ) ;
$$

$$
\mathbf { D } _ { \mathsf { S } \to \mathrm { R } } ^ { \xi } = \mathsf { s o f t m a x } _ { r } \left( \frac { \widehat { \mathbf { P } } ^ { \xi } } { \tau _ { t } } \right) ; \quad \mathbf { D } _ { \mathsf { S } \to \mathrm { R } } ^ { \mathrm { L O K I } } = \mathsf { s o f t m a x } _ { r } \left( \frac { \mathbf { P } } { \tau _ { s } } \right)
$$

Here, softma $\mathbf { X } _ { S }$ normalizes over sentences for each row, whereas softmax normalizes over rows for each sentence. We use teacher and student temperatures $\tau _ { t } = 0 . 5$ and $\tau _ { s } = 0 . 1$ , respectively. LOKI matches the teacher and student distributions in both normalization directions using Jensen-Shannon divergence:

$$
\mathcal { L } _ { \mathrm { d i s t } } = \frac { 1 } { 2 } [ \mathrm { J S } ( \mathbf { D } _ { \mathbf { R }  \mathbf { S } } ^ { \xi } \mid \mid \mathbf { D } _ { \mathbf { R }  \mathbf { S } } ^ { \mathrm { L O K I } } ) + \mathrm { J S } ( \mathbf { D } _ { \mathbf { S }  \mathbf { R } } ^ { \xi } \mid \mid \mathbf { D } _ { \mathbf { S }  \mathbf { R } } ^ { \mathrm { L O K I } } ) ]
$$

The first term preserves each row’s relative association across sentences, while the second preserves each sentence’s relevance across rows. Computed for both positive and negative contexts and averaged, this loss preserves associations beyond the strongest Top- $- K$ pairs without directly supervising the attention matrices $\mathbf { A } _ { f } , \mathbf { A } _ { r }$

Regularization Losses $( \mathcal { L } _ { \mathrm { s i g } } , \mathcal { L } _ { \mathrm { s i n k } } )$ . We use two complementary regularizers to prevent representation and attention collapse. First, the SIGReg regularizer of [5] encourages diversity and approximate isotropy in the contextualized embeddings by penalizing deviations of their random projections from a Gaussian distribution:

$$
\mathcal { L } _ { \mathrm { s i g } } = \frac { 1 } { 4 } \bigg [ \mathrm { S I G R e g } ( \widetilde { \mathbf { R } } ^ { + } ) + \mathrm { S I G R e g } ( \widetilde { \mathbf { S } } ^ { + } ) + \mathrm { S I G R e g } ( \widetilde { \mathbf { R } } ^ { - } ) + \mathrm { S I G R e g } ( \widetilde { \mathbf { S } } ^ { - } ) \bigg ]
$$

Second, a Sinkhorn-inspired marginal constraint [13] discourages hub-like attention, in which a small number of rows or sentences absorb most of the attention mass:

$$
\begin{array} { r l r } {  { \mathrm { S k } ( \mathbf { A } ) = \frac { 1 } { N _ { k } } \| \mathrm { c o l \_ s u m } ( \mathbf { A } ) - \frac { N _ { q } } { N _ { k } } \mathbf { 1 } \| _ { 2 } ^ { 2 } + \mathrm { V a r } ( \mathrm { c o l \_ s u m } ( \mathbf { A } ) ) ; } } \\ & { } & { \mathcal { L } _ { \mathrm { s i n k } } = \frac { 1 } { 4 } \bigg [ \mathrm { S k } ( \mathbf { A } _ { f } ^ { + } ) + \mathrm { S k } ( \mathbf { A } _ { r } ^ { + } ) + \mathrm { S k } ( \mathbf { A } _ { f } ^ { - } ) + \mathrm { S k } ( \mathbf { A } _ { r } ^ { - } ) \bigg ] \quad } \end{array}
$$

Here, $\operatorname { c o l } _ { - } s \mathbf { u m } ( \mathbf { A } )$ denotes the column sums of the row-normalized attention matrix, while $N _ { q }$ and $N _ { k }$ denote the numbers of valid queries and keys, respectively. The target column sum $N _ { q } / N _ { k }$ corresponds to distributing the attention mass uniformly across the keys. Together, these regularizers maintain diverse contextualized representations and balanced bidirectional attention distributions.

## 4 Text-Table Integration

We now discuss how LOKI discovers text-mediated join paths and uses them to integrate tables. Let $\mathcal { T } = \{ T _ { k } \} _ { k = } ^ { N }$ be the collection of 1 tables in a raw multi-modal data lake and $\mathcal { D } = \mathrm { \bar { \{ } }  D _ { \ell } \} _ { \ell = 1 } ^ { M }$ its collection of documents.

Step 1: Candidate Discovery. We assume that LOKI has been trained and denote the trained model by $\operatorname { L O K I } _ { \theta } ,$ , where $\theta$ represents its learned parameters while the sentence encoder $\xi$ remains frozen. We first discover coarsely associated tables and documents directly from the encoder’s outputs: for every table–document pair $( T _ { k } , D _ { \ell } )$

LOKI<sub>�</sub> aggregates its pair-score matrix P into the global similarity $q _ { k \ell } = \mathrm { S i m } ( T _ { k } , D _ { \ell } )$ . We retain the suficiently similar pairs as the set A of candidate table-document associations:

$$
\begin{array} { r } { \mathcal { A } = \{ ( T _ { k } , D _ { \ell } ) \in \mathcal { T } \times \mathcal { D } \mid q _ { k \ell } \geq \delta _ { \mathrm { r e t } } \} , } \end{array}
$$

where the retrieval threshold $\delta _ { \mathrm { { r e t } } }$ is selected on the validation set and fixed during inference to bound the number of combinations subsequently considered for join-path extraction. Candidate combinations are then constructed by pairing distinct tables associated with the same document:

$$
C = \{ ( T _ { A } , D , T _ { B } ) \mid ( T _ { A } , D ) \in \mathcal { A } , ( T _ { B } , D ) \in \mathcal { A } \}
$$

Each candidate combination $( T _ { A } , D , T _ { B } ) \in C$ therefore contains two tables independently relevant to $D ;$ ; documents with fewer than two retained tables yield no candidate.

## 4.1 Text-Mediated Join Path Discovery

Having constructed C, LOKI performs fine-grained discovery within each candidate combination. For simplicity, let $( T _ { A } , D , T _ { B } ) \in C { \mathrm { ~ d e } } .$ note an arbitrary candidate, where $T _ { A }$ and $T _ { B }$ are schematically disjoint tables independently identified as relevant to $D .$ Their coarse associations with � do not yet establish how their rows are related. For each candidate, $\mathrm { L O K I } _ { \theta }$ jointly contextualizes the rows from both tables with the sentences in $D$ while preserving their table of origin: we linearize and encode their rows before concatenating the resulting sequences in a fixed order:

$$
\mathbf { R } _ { 0 } = \xi { \big ( } \operatorname { l i n e a r i z e } ( T _ { A } ) { \big ) } \oplus \xi { \big ( } \operatorname { l i n e a r i z e } ( T _ { B } ) { \big ) } ; \quad \mathbf { S } _ { 0 } = \xi { \big ( } { \mathrm { s e n t e n c e s } } ( D ) { \big ) }
$$

where sentences(·) segments a document into sentences, ⊕ denotes ordered concatenation, and $n _ { A }$ and $n _ { B }$ are the row counts of $\mathit { T _ { A } }$ and $T _ { B } ,$ so that the first $n _ { A }$ rows originate from $T _ { A }$ and the following �<sub>�</sub> from $T _ { B } . \mathrm { A }$ single joint forward pass then produces the contextualized representations and pair-score matrix:

$$
( \widetilde { \mathbf { R } } , \widetilde { \mathbf { S } } , \mathbf { P } ) = \mathrm { L O K I } _ { \theta } ( \mathbf { R } _ { 0 } , \mathbf { S } _ { 0 } ) .
$$

These outputs provide a shared row-sentence context from which sparse, evidence-backed join paths can be extracted. Algorithm 1 summarizes this join-path extraction process.

Step 2: Atomic Link Identification. The pair-score matrix P is dense because every row in the ordered sequence $\mathbf { R } _ { 0 }$ receives an afinity score against every sentence in �. Since only a small subset provides useful evidence for text-mediated join paths, $\operatorname { L O K I } _ { \theta }$ converts P into a sparse set of high-confidence atomic links. For each candidate combination $( T _ { A } , D , T _ { B } )$ , we compute an adaptive threshold � rather than using a globally fixed cutof. For each row $r ,$ let $q _ { r } = \operatorname* { m a x } _ { s } ( P _ { r s } )$ denote its highest score across all sentences, and let $P _ { 7 5 }$ denote the 75th percentile of $\left\{ q _ { r } \right\} _ { r = 1 } ^ { n _ { A } + n _ { B } }$ . Furthermore, �<sub>P</sub> and �<sub>P</sub> denote the mean and standard deviation of all entries in P. The adaptive threshold is given by:

$$
\gamma = \operatorname* { m a x } ( \operatorname* { m i n } ( P _ { 7 5 } , \mu _ { \mathrm { P } } + 2 s _ { \mathrm { P } } ) , \gamma _ { \mathrm { m i n } } )
$$

where $\gamma _ { \mathrm { m i n } } = 0 . 1 5$ is the default minimum score threshold. This makes the extraction stricter when the example contains clear rowsentence associations, while still preventing the threshold from collapsing on noisier examples. We then apply mutual Top-� filtering. For each row $r _ { i } ,$ the filter retains the indices of its $k _ { r }$ highest-scoring sentences, whereas for each sentence $s _ { t } .$ , it retains the indices of its $k _ { s }$ highest-scoring rows:

$$
\begin{array} { r } { N _ { k _ { r } } ^ { \mathrm { r o w } } ( i ) = \mathrm { T o p } _ { k _ { r } } ( P _ { i , : } ) ; \quad N _ { k _ { s } } ^ { \mathrm { s e n t } } ( t ) = \mathrm { T o p } _ { k _ { s } } ( P _ { : , t } ) } \end{array}
$$

where $\mathrm { T o p } _ { k } ( \cdot )$ returns the indices of the � largest entries. The sentence-side operation is applied independently to the row partitions of $T _ { A }$ and �<sub>�</sub>. We use $k _ { r } = 3 2$ and $k _ { s } = 1 0$ by default. The row-side filter prevents a row from attaching to too many vague sentences, while the sentence-side filter suppresses hub sentences that align weakly with many rows but do not provide discriminative evidence for a specific join path. The resulting atomic-link set is:

$$
\mathcal { T } = \Big \{ ( i , t ) ~ \Big | ~ P _ { i t } \geq \gamma ~ \wedge ~ t \in N _ { k _ { r } } ^ { \mathrm { r o w } } ( i ) ~ \wedge ~ i \in N _ { k _ { s } } ^ { \mathrm { s e n t } } ( t ) \Big \}
$$

Since Step 3 must connect rows across two schematically disjoint tables, we partition these links by table origin:

$$
\begin{array} { r l } & { \mathcal { T } _ { A } = \{ ( i , t ) \in \mathcal { T } : 1 \leq i \leq n _ { A } \} } \\ & { \mathcal { T } _ { B } = \{ ( j , t ) : ( n _ { A } + j , t ) \in \mathcal { T } , \ 1 \leq j \leq n _ { B } \} } \end{array}
$$

Here, $\mathcal { T } _ { A }$ and $\mathcal { T } _ { B }$ preserve table origin while retaining the sentence supporting each atomic link. These table-specific links provide the evidence anchors for cross-table join paths.

Algorithm 1: Join-path extraction (Steps 2–3) for one candidate   
combination $( T _ { A } , D , T _ { B } )$   
Input: Candidate $( T _ { A } , D , T _ { B } )$ , frozen encoder $\xi ,$ trained model   
$\mathrm { L O K I } _ { \theta } ,$ threshold floor $\gamma _ { \mathrm { { m i n } } } ,$ , and Top-� parameters $k _ { r } , k _ { s }$   
Output: Atomic links $\mathcal { T } _ { A } , \mathcal { T } _ { B }$ and candidate join paths $\mathcal { P } _ { \mathrm { c a n d } }$   
// Joint row-sentence encoding   
1 $\mathbf { R } _ { 0 }  \xi ( \mathrm { l i n e a r i z e } ( T _ { A } )$ ⊕ linearize $\left( T _ { B } \right) )$   
2 $\mathbf { S } _ { 0 } \gets \boldsymbol { \dot { \xi } }$ (sentences(�) )   
3 $( \widetilde { \mathbf { R } } , \widetilde { \mathbf { S } } , \mathbf { P } ) \gets \mathrm { L O K I } _ { \theta } ( \mathbf { R } _ { 0 } , \mathbf { S } _ { 0 } )$   
// Compute adaptive score floor   
4 � ← max(min $( P _ { 7 5 } , \mu _ { \mathrm { P } } + 2 s _ { \mathrm { P } } ) , \gamma _ { \mathrm { m i n } } )$   
// Extract table-specific atomic links   
5 $\begin{array} { r } { \Lambda _ { k _ { r } } ^ { \mathrm { r o w } } ( i ) \gets \mathrm { T o p } _ { k _ { r } } ( P _ { i ; \cdot } ) , \quad \mathcal { N } _ { k _ { s } } ^ { \mathrm { s e n t } } ( t ) \gets \mathrm { T o p } _ { k _ { s } } ( P _ { : , t } ) } \end{array}$   
6 $\mathcal { T }  \{ ( i , t ) \mid P _ { i t } \geq \gamma \wedge t \in N _ { k _ { r } } ^ { \mathrm { r o w } } ( i ) \wedge i \in N _ { k _ { s } } ^ { \mathrm { s e n t } } ( t ) \}$   
7 $\mathcal { T } _ { A }  \{ ( i , t ) \in \mathcal { I } : 1 \leq i \leq n _ { A } \}$   
8 $\mathcal { T } _ { B } \gets \{ ( j , t ) : ( n _ { A } + j , t ) \in \mathcal { T } , \ 1 \leq j \leq n _ { B } \}$   
// Construct shared-sentence join paths   
9 $\mathcal { P } _ { \mathrm { c a n d } }  \{ ( r _ { i } ^ { A } , s _ { t } , r _ { j } ^ { B } ) \ \middle | \ ( i , t ) \in \mathcal { T } _ { A } \wedge ( j , t ) \in \mathcal { T } _ { B } \}$   
10 foreach sentence and each row pair do   
11 retain only the path with maximal $\nu _ { i j t }$   
12 end   
13 return $\mathcal { T } _ { A } , \mathcal { T } _ { B } , \mathcal { P } _ { \mathrm { c a n d } }$

Step 3: Transitive Join-Path Construction. Because $T _ { A }$ and $T _ { B }$ are schematically disjoint, LOKI does not assume that a row from one table can be joined directly with a row from the other table. Instead, the connection must be mediated by text. After atomiclink identification, a row $r _ { i } ^ { A } \in T _ { A }$ and a row $\dot { r } _ { j } ^ { B } \in T _ { B }$ become join candidates only when they share at least one mediating sentence $s _ { t } .$ . For each shared sentence, LOKI constructs a text-mediated join path and assigns it a bridge weight:

$$
\ p _ { i j t } = ( r _ { i } ^ { A } , s _ { t } , r _ { j } ^ { B } ) , w _ { i j t } = { \frac { P _ { i t } + P _ { n _ { A } + j , t } } { 2 } } 
$$

Here, $\mathbf { \nabla } \phi _ { i j t }$ is a single-evidence instance of the general text-mediated join path $\boldsymbol { p } = ( r _ { i } ^ { A } , S _ { p } , r _ { j } ^ { B } )$ , with $S _ { p _ { i j t } } = \{ s _ { t } \}$ . Paths connecting the same row pair and relationship type are consolidated by collecting their mediating sentences into $S _ { p }$ during materialization. For simplicity, each path in our running example contains a single sentence. Because a candidate path is formed only when �<sub>�</sub> has retained atomic links to both rows, $w _ { i j t }$ summarizes their joint compatibility and serves as the bridge confidence used to rank candidate paths and weight the contextualized sentence representation during materialization. The candidate path set is then:

$$
\mathcal { P } _ { \mathrm { c a n d } } = \left\{ \{ p _ { i j t } = { \left( r _ { i } ^ { A } , s _ { t } , r _ { j } ^ { B } \right) } \ \middle | \ ( i , t ) \in \mathcal { I } _ { A } \wedge ( j , t ) \in \mathcal { J } _ { B } \right\}
$$

To avoid over-representing repeated or generic evidence, LOKI retains only the strongest paths per sentence and per row pair. The resulting $\mathcal { P } _ { \mathrm { c a n d } }$ is therefore a compact set of explicit, evidencebacked bridges. In the running example, the sentences $\{ s _ { 1 } , . . . , s _ { 5 } \}$ produce the following paths: $p _ { 1 } = ( I _ { 1 } , \{ s _ { 1 } \in D _ { 1 } \} , M _ { 1 } ) , p _ { 2 } = ( I _ { 1 } , \{ s _ { 2 } \in$ $D _ { 1 } \} , M _ { 1 } ) , \ p _ { 3 } \ = \ ( I _ { 1 } , \{ s _ { 3 } \in D _ { 1 } \} , M _ { 1 } ) , \ p _ { 4 } \ = \ ( I _ { 1 } , \{ s _ { 4 } \in D _ { 1 } \} , M _ { 2 } )$ , and ${ p } _ { 5 } = ( I _ { 1 } , \{ s _ { 5 } \in D _ { 1 } \} , M _ { 3 } )$

## 4.2 Integration

Discovery-driven integration requires mapping candidate join paths to latent relationship types R and materializing them as typed integrated tables. Each candidate path $\bar { p } _ { i j t } = ( \bar { r _ { i } ^ { A } } , s _ { t } , r _ { j } ^ { B } ) \in \bar { \mathcal { P } } _ { \mathrm { c a n d } }$ d is treated as a distinct evidence unit, allowing diferent sentences connecting the same row pair to express diferent relationships. In the running example, $p _ { 1 } = ( I _ { 1 } , \{ s _ { 1 } \in D _ { 1 } \} , M _ { 1 } )$ and $p _ { 2 } = ( I _ { 1 } , \{ s _ { 2 } \in$ $D _ { 1 } \} , M _ { 1 } )$ share the same row pair but express Prescribed\_For and Adverse\_Efect, respectively.

Step 4: Path Representation and Clustering. To group candidate paths by the relationship expressed in their mediating evidence, LOKI constructs a path-specific representation that combines the contextualized sentence with the strength of its attachment to both table rows. For a candidate path $p _ { i j t } = ( r _ { i } ^ { A } , s _ { t } , r _ { j } ^ { B } )$ , let

$$
\alpha _ { i t } = P _ { i t } , \beta _ { j t } = P _ { n _ { A } + j , t } , w _ { i j t } = \frac { \alpha _ { i t } + \beta _ { j t } } { 2 }
$$

Here, �<sub>��</sub> and $\beta _ { j t }$ measure how strongly sentence �<sub>�</sub> supports rows $r _ { i } ^ { A }$ and $r _ { j } ^ { B } ,$ , respectively, and $\boldsymbol { w } _ { i j t }$ is their average, representing the confidence of the complete bridge. We �<sub>2</sub>-normalize the contextualized embeddings so that the three components begin on the same scale and, since an unweighted concatenation would treat them as equally important even when one underlying link is considerably weaker, we use the association scores as multiplicative gates:

$$
\phi ^ { \mathrm { c t x } } ( \phi _ { i j t } ) = \left[ \alpha _ { i t } \widetilde { \mathbf { r } } _ { i } ^ { A } \parallel \boldsymbol { w } _ { i j t } \widetilde { \mathbf { s } } _ { t } \parallel \beta _ { j t } \widetilde { \mathbf { r } } _ { j } ^ { B } \right] \in \mathbb { R } ^ { 3 d }
$$

The ordered concatenation preserves the distinct roles of the two table rows and the mediating sentence, while the weights modulate each component according to the strength of its supporting evidence, preserving potentially asymmetric attachment strengths. Thus, paths involving the same sentence remain distinguishable when they connect diferent row pairs or exhibit diferent attachment strengths. Finally, the complete path vector is $L _ { 2 }$ -normalized before clustering, preventing its overall magnitude from dominating the clustering distance:

$$
{ \mathcal { K } } = \{ C _ { 1 } , . . . , C _ { M } \} = \operatorname { H D B S C A N } \left( \left\{ \phi ^ { \mathrm { c t x } } ( p ) ~ | ~ p \in { \mathcal { P } } _ { \mathrm { c a n d } } \right\} \right)
$$

![](images/3c6fe9310d3510e15f73576e6f1fc6b41c1afaddc29f193f86954a88cdb1ed8e.jpg)  
Figure 4: LOKI inference pipeline. (a) Given a retrieved candidate $( T _ { A } , D , T _ { B } )$ , a single joint forward pass through LOKI<sub>�</sub> produces the pair-score matrix P, whose high-confidence atomic links are composed into candidate join paths, (b) clustered into the illustrative groups $\{ C _ { 1 } , C _ { 3 } \}$ . (c) The Generalization Function � assigns the relationship labels $\{ R _ { 1 } , R _ { 3 } \}$ , (d) and the labeled clusters are materialized as relation-specific integrated tables with supporting evidence.

Each cluster $C _ { k } \in \mathcal K$ therefore groups text-mediated join paths whose row-conditioned evidence expresses similar relational semantics. In the running example, the paths $\mathcal { P } 1$ and $\phi _ { 4 }$ are expected to be grouped together because they represent a Prescribed\_For relationship, while $\mathcal { P } 3$ and $\mathcal { P } 5$ form another group because they represent a Discontinued relationship.

Step 5: Relationship Materialization. After clustering, each cluster $C _ { k }$ contains text-mediated join paths whose row-conditioned evidence expresses a coherent but unnamed relationship. To make this relationship interpretable, we use a Generalization Function � that employs an LLM to summarize the shared evidence within the cluster and assign a descriptive relationship label $R _ { k } \colon$

$$
G : C _ { k } \xrightarrow { \mathrm { L L M } } R _ { k }
$$

Each labeled cluster is then materialized as an integrated table:

$$
T _ { \mathrm { i n t e g r a t e d } } ^ { ( R _ { k } ) } = \left\{ ( r _ { i } ^ { A } , r _ { j } ^ { B } , s _ { t } ) ~ { \textstyle | } ~ p _ { i j t } = ( r _ { i } ^ { A } , s _ { t } , r _ { j } ^ { B } ) \in C _ { k } \right\}
$$

Repeating this process for each labeled cluster produces the final set of relation-specific integrated tables, where every entry combines one record from $T _ { A }$ with one record from $T _ { B }$ and preserves the supporting evidence for the relationship $R _ { k } .$ . Fig. 4 illustrates the materialization of a subset of the paths in our running example, first represented in the latent space and clustered into relationconsistent groups. The paths $\mathcal { P } 1$ and $\mathcal { P } 3$ remain close because they share the same row pair $( I _ { 1 } , M _ { 1 } )$ , but they are assigned to diferent clusters because they represent diferent relationships. The Generalization Function then labels $C _ { 1 }$ as $R _ { 1 }$ : Prescribed\_For and $C _ { 3 }$ as $R _ { 3 }$ : Discontinued. Finally, each labeled cluster is materialized as a separate integrated table with sentence-level provenance.

## 5 Experimental Evaluation

We empirically evaluate LOKI through three experiments that examine successive stages of the same discovery-driven integration pipeline: (i) architectural evaluation of representation learning for implicit association, (ii) multi-modal data discovery, and (iii) data integration. Together, the three experiments establish the progression from association to discovery towards integration; Table 1 summarizes this evaluation strategy, and each experiment details its objective in the corresponding subsection. All datasets, scripts, and experimental results are publicly available in our repository.<sup>2</sup>

Table 1: Progressive evaluation of LOKI.
<table><tr><td>Exp.</td><td>Capability</td><td>Output</td><td>Validates</td></tr><tr><td>Exp.1</td><td>Association</td><td>Row ↔ Sentence</td><td>Representation Learning</td></tr><tr><td>Exp. 2</td><td>Discovery</td><td>Table ↔ Document</td><td>Cross-modal Retrieval</td></tr><tr><td>Exp. 3</td><td>Integration</td><td>Integrated Relation</td><td>Discovery-driven Integration</td></tr></table>

## 5.1 Datasets

No publicly available benchmark natively provides the typed rowsentence-row annotations required by our Text-Mediated Join Path Discovery task; existing benchmarks target generation, question answering, or coarse discovery instead. We therefore constructed ground truth over four complementary datasets, summarized in Table 2 and used throughout the three experiments; all of them are publicly released.<sup>3</sup> FEVEROUS [1] is an open-domain Wikipedia fact-verification benchmark whose evidence combines table cells with sentences, from which we derive row-sentence associations. We manually annotated ProTrix [35] using the original questionanswer pairs. Although Pharma [8] primarily targets table-text discovery, we repurposed it for row-sentence silver annotation using term frequency–inverse document frequency (TF-IDF) similarity and keyword matching.

Among the datasets we consider, MIMIC-IV [20, 21] provides the most comprehensive granular links between rows of the two schematically disjoint tables Diagnosis and Medications through Clinical Notes; it is the only dataset that supports our complete endto-end integration setting. We constructed a validated benchmark from it covering 382 hospital admissions: three frontier LLM annotators independently generated candidate annotations, consolidated through majority voting and verified by a human annotator. All fine-grained annotations are used exclusively for evaluation; the full re-construction procedure, annotation guidelines, and validation protocol are included in the released artifact.

## 5.2 Experiment 1: Architectural Evaluation

The first experiment investigates whether a model trained exclusively using coarse table-text associations can simultaneously learn: (i) global table-text relevance, required for multi-modal discovery, and (ii) fine-grained row-sentence associations, required for join path extraction. This setting presents a fundamental supervision dilemma. Coarse table-document associations are comparatively easy to obtain, whereas explicitly annotating every supporting row-sentence pair is expensive and domain-dependent, and such annotations are scarce. Standard sentence encoders represent rows and sentences independently and provide no mechanism for iden tifying which local interactions explain the relevance of an entire table-document pair. LOKI instead derives the global score from a pair-score matrix over mutually contextualized representations, so we can ask whether local association emerges from the global objective alone.

Table 2: Datasets and task coverage. Size denotes total finegrained (F) and coarse-grained (C) examples.
<table><tr><td>Dataset</td><td>Domain</td><td>Native unit</td><td>Size (F/C)</td><td>Exp.</td></tr><tr><td>FEVEROUS[1]</td><td>Wiki</td><td>Cell-Sentence</td><td>500/13,304</td><td>1</td></tr><tr><td>ProTrix[35]</td><td>QA</td><td>Table-Paragraph</td><td> $4 4 6 / 3 , 1 5 7$ </td><td>1</td></tr><tr><td>Pharma[8]</td><td>BioMed</td><td> $\mathrm { T a b l e s } ( 8 ) – \mathrm { T e x t }$ </td><td>140/926</td><td>1,2</td></tr><tr><td>MIMIC-IV[20]</td><td>Clinical</td><td> $\mathrm { T a b l e s } ( 2 ) – \mathrm { N o t e s }$ </td><td> $3 8 2 / 2 4 { , } 4 1 0$ </td><td> $1 , 2 , 3$ </td></tr></table>

Investigated Architectures. We compare five configurations. The Baseline uses the Frozen Encoder $\xi$ alone to independently embed linearized rows and document sentences, scoring each rowsentence pair by cosine similarity without task-specific training or cross-attention. FT-Encoder fine-tunes $\xi$ using the global tabletext triplet objective. Uni (R→S) contextualizes rows with sentences, whereas Uni $\mathbf { ( S \mathrm { \partial } \vec { \mathbf { \sigma } } \mathbf { \cdot } \mathbf { R } ) }$ contextualizes sentences with table rows, while keeping � frozen. LOKI uses the complete bidirectional architecture to jointly contextualize both modalities before constructing the pair-score matrix and aggregated global score (Fig. 2).

Evaluation Protocol. We evaluate all five architectures on all datasets of Table 2. The fine-grained row-sentence annotations are used only for evaluation. We report the best validation accuracy (Acc.) for global table-text matching and track test average precision (AP) and macro-�1 across epochs for local row-sentence association. AP is the primary local metric under class imbalance, while macro-�1 captures the precision-recall balance. We additionally report ranking quality, training time, and peak VRAM/RAM consumption.

Cross-dataset Architectural Comparison. Table 3 shows that LOKI’s architecture generalizes across benchmarks and transfers across granularities, attaining the best local AP on every dataset and demonstrating that coarse supervision can induce stronger local association. The smaller local gain on Pharma reflects its silver annotations, derived from TF-IDF and keyword matching, which favor explicit lexical matches already captured by the Baseline and provide limited scope for measuring deeper semantic association. Overall, bidirectional contextualization transfers table-text supervision to row-sentence association more efectively than encoder fine-tuning or unidirectional attention; next we examine this capability in detail.

MIMIC-IV is more challenging than the other benchmarks: its long, noisy clinical notes express relationships only implicitly, beyond lexical overlap, and its validated fine-grained annotations support the complete end-to-end integration task. It is also where

Table 3: Table-Text accuracy (Acc.) vs. row-sentence average precision (AP) across datasets, measuring implicit learning from the coarse training task to fine-grained association.
<table><tr><td>Configuration FEVEROUS</td><td></td><td>ProTrix</td><td>Pharma</td><td>MIMIC-IV</td></tr><tr><td>Baseline</td><td> $\overline { { 0 . 9 6 \mathrm { ~ / ~ } 0 . 3 4 } }$ </td><td> $0 . 8 6 / 0 . 1 7$ </td><td> $\overline { { 0 . 9 1 \mathrm { ~ / ~ } 0 . 4 0 } }$ </td><td> $\overline { { 0 . 8 0 / 0 . 4 2 } }$ </td></tr><tr><td>FT-Encoder</td><td> $0 . 9 6 / \ : 0 . 3 4$ </td><td> $0 . 9 3 / 0 . 5 8$ </td><td> $0 . 9 3 \mathrm { ~ / ~ } 0 . 4 0$ </td><td> $0 . 8 1 / 0 . 4 4$ </td></tr><tr><td>Uni (R→S)</td><td> $0 . 7 7 / \ 0 . 3 5$ </td><td> $0 . 8 3 / 0 . 6 2$ </td><td> $1 . 0 / 0 . 4 0$ </td><td> $0 . 7 3 / 0 . 4 2$ </td></tr><tr><td> $\mathrm { U n i } \left( \mathrm { S \to R } \right)$ </td><td> $0 . 8 2 / \ : 0 . 3 3$ </td><td> $0 . 5 5 / 0 . 3 9$ </td><td> $1 . 0 / 0 . 3 9$ </td><td> $0 . 7 2 / 0 . 4 1$ </td></tr><tr><td>LOKI</td><td> $\mathbf { 0 . 9 8 } / \mathbf { 0 . 4 4 }$ </td><td> $\mathbf { 0 . 9 4 } / \mathbf { 0 . 6 7 }$ </td><td> $\mathbf { 0 . 9 9 } / \mathbf { 0 . 4 2 }$ </td><td> $\mathbf { 0 . 9 4 } / \mathbf { 0 . 5 4 }$ </td></tr></table>

the architectural diferences become clearest. Although the Baseline uses a domain-specific encoder (MedEmbed),<sup>4</sup> fine-tuning it transfers poorly from the global objective to local association, and the unidirectional variants retain near-Baseline AP while degrading global accuracy. LOKI outperforms all of them on both granularities.

![](images/199169e18d1c1d6dc51438f29c6362b4eaca074563ea107ccd89bf41b913e187.jpg)  
Figure 5: Relative improvement over the Baseline on global Acc. and local macro-� 1 and AP across training epochs.

Architectural Benefits. Fig. 5 tracks whether optimization of the coarse table-text task induces fine-grained association, reporting the relative improvement over the Baseline, $\Delta _ { \mathrm { r e l } } ( M ) = \left( M _ { \mathrm { m o d e l } } - \frac { \ d } { \ d t } \right.$ $M _ { \mathrm { b a s e l i n e } } ) / M _ { \mathrm { b a s e l i n e } } \times 1 0 0 \%$ , per metric �. FT-Encoder produces only small local gains, while the unidirectional variants show uneven transfer; LOKI improves all three metrics, nearly doubling macro-� 1. All configurations converge within a few epochs, reflecting their strong pre-trained initialization rather than low task complexity: the variants plateau at clearly diferent levels, and only LOKI’s bidirectional contextualization converges to substantially stronger local association using only table-text signals.

![](images/6f8900a1525431bd010f7fff6213fe10bc7f30d3fc265b1f425937071cde6a33.jpg)

![](images/bf71076eef7d645d497e90c34cb885de24692cc3c418d344ceac01334292081c.jpg)  
Figure 6: Training time and VRAM/Memory consumption.

Compute Requirements. Fig. 6 shows that LOKI trains approximately 30× faster than full encoder fine-tuning while achieving higher AP/� 1. On a consumer-grade GPU with 16 GB of VRAM, it also has the lowest peak VRAM and RAM footprint among the trained variants, yielding the best efectiveness-eficiency trade-of.

![](images/bf6e5860689f342a3a2e5c407a868bc8bd559023f1cedc9920a9452a46d833b5.jpg)  
Figure 7: MIMIC-IV row-sentence ranking across cutof-based metrics, precision-recall, and mean rank.

Ranking Analysis. Fig. 7 confirms that LOKI’s AP gain reflects consistently stronger rankings. LOKI achieves the highest P@�, F1@�, and normalized discounted cumulative gain (NDCG@�) across all evaluated values of � and maintains the strongest precision-recall curve. Its normalized mean rank is approximately 34% lower than the Baseline, denoting valid evidence appears earlier in the ranking.

Summary. Exp. 1 shows that of-the-shelf encoder fine-tuning and unidirectional attention do not reliably transfer coarse table-text supervision to fine-grained row-sentence association. LOKI finds better associations in both granularities without local supervision, while remaining more eficient than full encoder fine-tuning.

## 5.3 Experiment 2: Multi-Modal Data Discovery

The second experiment evaluates whether LOKI’s global table-text score, derived from fine-grained row-sentence associations, supports accurate and scalable multi-modal data discovery. First, we evaluate whether LOKI learns both retrieval directions: unlike existing systems that are typically designed for a fixed query direction, LOKI learns a shared bidirectional representation, and we test whether a single trained instance can serve both document-totable and table-to-document retrieval without retraining. Second, we compare it against state-of-the-art systems operating at the coarser table-document level and examine whether its bottom-up aggregation remains efective as the candidate pool grows.

Compared Systems. No existing system targets our fine-grained row-sentence discovery task, but several state-of-the-art approaches operate at the coarser table-document level. We therefore evaluate whether LOKI identifies the same information as these systems on their own coarse-grained task, while additionally providing the row-sentence traceability that enables the integration task of Exp. 3. We compare against three representative systems. CMDL [8] targets cross-modal data discovery and was primarily evaluated on the Pharma benchmark for document-to-table retrieval, motivating our use of the same dataset. TaBERT [36] represents joint tabletext representation learning, while TabSTAR [3] represents recent text-aware tabular foundation models.

Pharma Benchmark Construction. To ensure a fair comparison with CMDL [8], we reconstruct Pharma from its 926 PubMed abstracts and 82 DrugBank tables. In the original dataset, 8 tables, spanning 15 ground-truth columns, are positively associated with every document, while the remaining 74 tables form the negative pool. We horizontally partition the source tables into row-disjoint candidate tables, ensuring that training, validation, and test splits do not reuse the same rows. For each query, the associated tables form the positive candidates and are paired with an equal number of randomly sampled negative candidates. We additionally inject non-matching rows into positive candidates to prevent models from relying only on isolated lexical matches. The test split contains 140 query documents and 2,240 candidate tables in the Full retrieval pool. This construction requires models to infer query-specific table relevance rather than memorize stable table or column associations.

Evaluation Protocol. Following the objective, we first evaluate whether LOKI learns both directions: we report cross-dataset Mean Average Precision (MAP) on Pharma and MIMIC-IV for documentto-table and table-to-document retrieval, and then evaluate crossdirection generalization on Pharma. The architecture is directionagnostic; only the training triplets are anchored on one modality. We call a configuration direction-matched when the training anchor and the retrieval query use the same modality, and directionmismatched otherwise; the latter tests whether a single trained instance can serve either query modality without retraining. These comparisons use a fixed pool of 50 candidates per query to control retrieval dificulty. Second, following CMDL’s native document-totable setting, we progressively increase the candidate-table pool and report F1@�, NDCG@�, mean reciprocal rank (MRR@�), MAP, mean rank, and inference time to assess retrieval quality and scalability under realistic data-lake conditions.

Table 4: LOKI’s cross-dataset table-text discovery (MAP).
<table><tr><td>Dataset</td><td>Doc→Table Table→Doc</td><td></td></tr><tr><td>Pharma</td><td>0.78</td><td>0.20</td></tr><tr><td>MIMIC-IV</td><td>0.42</td><td>0.54</td></tr></table>

Cross-dataset Bidirectional Discovery. On both datasets, LOKI supports both retrieval directions without retraining (Table 4), with direction-dependent behavior reflecting the distinct discovery workflows of each benchmark. On MIMIC-IV, it reaches 0.42 MAP for document-to-table and 0.54 for table-to-document retrieval, whereas the latter is noticeably weaker on Pharma. This follows from Pharma’s construction: every document is associated with the same eight source tables, so a table query faces a large, difuse set of relevant abstracts with limited discriminative signal. Existing systems are typically specialized for one query direction and require separate training or retrieval pipelines for the reverse task. LOKI instead provides a single representation that supports both query modalities without retraining.

Cross-Direction Generalization under Direction Mismatch. Table 5 crosses the modality used as the training anchor with the modality used as the retrieval query on Pharma dataset. LOKI achieves the highest MAP in all four settings. Under direction mismatch (grey cells), it remains more than 50% above the strongest baselines, although document-to-table retrieval is more afected by the mismatch. The main result is therefore not direction-invariant performance, but that a single trained LOKI instance can support either retrieval direction without retraining.

Table 5: Cross-direction generalization on Pharma (MAP), crossing the training anchor modality with the retrieval query modality. Grey cells mark direction-mismatched training configurations for retrieval.
<table><tr><td rowspan="2">System</td><td colspan="2">Doc→Table</td><td colspan="2">Table→Doc</td></tr><tr><td>Doc-trained</td><td>Table-trained</td><td>Doc-trained</td><td>Table-trained</td></tr><tr><td>CMDL</td><td>0.25</td><td>0.20</td><td>0.09</td><td>0.09</td></tr><tr><td>TaBERT</td><td>0.52</td><td>0.23</td><td>0.11</td><td>0.11</td></tr><tr><td>TabSTAR</td><td>0.20</td><td>0.20</td><td>0.09</td><td>0.09</td></tr><tr><td>LOKI</td><td>0.78</td><td>0.35</td><td>0.18</td><td>0.20</td></tr></table>

![](images/71b26cbe61b20c2c8f3a34134ee4c91aa6699ce59138e38c00cd7985fe7ec781.jpg)  
Figure 8: Impact of scaling on data discovery; candidate pool size denotes the number of tables per query document.

Scalability Analysis. Fig. 8 evaluates document-to-table retrieval as the candidate-table pool increases from 50 to the Full set of 2,240 candidates. We report metrics at � = 8 because each Pharma query document is associated with eight relevant tables. LOKI consistently achieves the strongest retrieval quality across all pool sizes and degrades less drastically than the competing systems. At the Full scale it roughly triples the F1@8 and MAP of the second-strongest TaBERT, achieves the lowest mean rank by placing relevant tables approximately 56% earlier, and completes full-pool retrieval 32× faster than TaBERT. Although CMDL is faster still due to its ANNbased approximate retrieval, its retrieval quality collapses at scale, with MAP approximately 35× lower than LOKI’s.

Summary. Exp. 2 demonstrates that LOKI’s fine-grained rowsentence associations provide a strong basis for multi-modal discovery: a single trained instance serves either retrieval direction without retraining, and remains the strongest method as the candidate pool grows.

## 5.4 Experiment 3: Data Integration

The third experiment evaluates whether fine-grained row-sentence associations underlying table-document discovery can be transformed into explicit text-mediated join paths and subsequently materialized as relationship-specific Typed Tables, completing the progression from association to discovery and finally integration.

Evaluation Protocol. We evaluate end-to-end integration on 382 MIMIC-IV admissions, each constituting an independent integration problem over the schematically disjoint Diagnosis and Medications tables and their associated Clinical Notes. For each admission, LOKI transforms the discovered text-mediated join paths into provenance-preserving Typed Tables. Since the Generalization Function � produces open-world relationship names, we ground them to the canonical relation types R before comparing the predicted Typed Tables with their ground-truth counterparts.

Compared Systems. We instantiate � with two interchangeable open-weight local LLMs, GPT-OSS 20B and Qwen-3.6, denoted as LOKI (GPT-OSS) and LOKI (Qwen-3.6), respectively. We compare LOKI against direct LLM-based integration using the same local Qwen-3.6, and additionally include Qwen-3.7 as a stronger API-based frontier baseline. The direct LLMs receive the relevant Diagnosis table, Medications table, and Clinical Notes directly and infer typed row-pair relationships for materialization as Typed Tables under the same canonical schema, thereby evaluating relation inference over known inputs rather than table-document discovery or text-mediated join-path recovery.

Evaluation Metrics. We evaluate the outputs at two complementary levels. At the typed-pair level, Best-Match Typed-Pair precision (P), recall (R), and F1 measure row-pair correctness and coverage after each predicted Typed Table is associated with its best-matching ground-truth relationship type. The reported summary scores are macro-averaged over admissions. At the Typed Table level, Type Acc. measures whether the predicted table is assigned the correct canonical relationship type, while Typed Table P, R, and F1 measure the final relationship-specific materialization quality and are likewise macro-averaged over admissions. For both levels we additionally report pooled counts and the corresponding micro scores, computed on the same evaluation surface as the macro metrics. For eficiency we report average runtime per admission, LLM token consumption, and estimated API-equivalent cost over the 382-admission test set.

We organize the evaluation around two questions:

RQ1: How accurately are relationship semantics disambiguated and materialized as Typed Tables?

RQ2: How practical is discovery-driven integration?

## RQ1: How accurately are relationship semantics disambiguated and materialized as Typed Tables?

Recovering a row pair alone is insuficient for semantic integration because the relationship expressed between the participating rows must also be distinguished. We therefore evaluate disambiguation through its final database-level consequence: whether the discovered relationships are assigned to the correct relationship-specific Typed Tables and whether their row-pair contents are faithfully materialized. Table 6 summarizes the main results.

Table 6: Comparison of LOKI and direct LLM baselines on relationship disambiguation and Typed Table materialization. P, R, and F1 are macro-averaged over admissions.
<table><tr><td></td><td colspan="3">Best-Match Typed-Pair</td><td colspan="4">Typed Table Materialization</td></tr><tr><td>System</td><td>P</td><td>R</td><td>F1</td><td>Type Acc.</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>LOKI (GPT-OSS)</td><td>0.982</td><td>0.486</td><td>0.627</td><td>0.840</td><td>0.755</td><td>0.755</td><td>0.742</td></tr><tr><td>LOKI (Qwen-3.6)</td><td>0.982</td><td>0.515</td><td>0.652</td><td>0.807</td><td>0.750</td><td>0.732</td><td>0.726</td></tr><tr><td>Qwen-3.6 (Local)</td><td>0.993</td><td>0.540</td><td>0.678</td><td>0.920</td><td>0.932</td><td>0.929</td><td>0.930</td></tr><tr><td>Qwen-3.7 (API)</td><td>0.997</td><td>0.717</td><td>0.817</td><td>0.958</td><td>0.966</td><td>0.963</td><td>0.964</td></tr></table>

All systems exhibit very high typed-pair precision, with LOKI remaining above 0.98 and within two percentage points of the direct LLMs. Their main diference is therefore coverage. Relative to LOKI (GPT-OSS), direct Qwen-3.6 improves pair recall by approximately 11%, while Qwen-3.7 improves it by approximately 48%. This higher coverage propagates to the final Typed Tables, where the direct Qwen-3.6 and Qwen-3.7 baselines achieve approximately 25% and 30% higher macro-F1 than LOKI (GPT-OSS), respectively. In contrast, the two LOKI configurations remain within approximately 2% of each other in table-level F1, indicating that the discovered relational structure can be materialized consistently using diferent implementations of the replaceable function �.

Table 7: Raw counts underlying the Best-Match Typed-Pair and Typed Table materialization results.
<table><tr><td>System</td><td>Pred.</td><td>GT</td><td>TP</td><td>FP</td><td>FN</td><td>Micro P</td><td>Micro R</td><td>Micro F1</td></tr><tr><td colspan="9">Best-Match Typed-Pair</td></tr><tr><td>LOKI (GPT-OSS)</td><td>2882</td><td>6441</td><td>2823</td><td>59</td><td>3618</td><td>0.980</td><td>0.438</td><td>0.606</td></tr><tr><td>LOKI (Qwen-3.6)</td><td>3051</td><td>6454</td><td>2990</td><td>61</td><td>3464</td><td>0.980</td><td>0.463</td><td>0.629</td></tr><tr><td>Qwen-3.6 (Local)</td><td>3103</td><td>6468</td><td>3086</td><td>17</td><td>3382</td><td>0.995</td><td>0.477</td><td>0.645</td></tr><tr><td>Qwen-3.7 (API)</td><td>4149</td><td>6468</td><td>4135</td><td>14</td><td>2333</td><td>0.997</td><td>0.639</td><td>0.779</td></tr><tr><td colspan="9">Typed Table Materialization</td></tr><tr><td>LOKI (GPT-OSS)</td><td>2018</td><td>2018</td><td>1696</td><td>322</td><td>322</td><td>0.840</td><td>0.840</td><td>0.840</td></tr><tr><td>LOKI (Qwen-3.6)</td><td>2011</td><td>2113</td><td>1705</td><td>306</td><td>408</td><td>0.848</td><td>0.807</td><td>0.827</td></tr><tr><td>Qwen-3.6 (Local)</td><td>523</td><td>523</td><td>481</td><td>42</td><td>42</td><td>0.920</td><td>0.920</td><td>0.920</td></tr><tr><td>Qwen-3.7 (API)</td><td>548</td><td>548</td><td>525</td><td>23</td><td>23</td><td>0.958</td><td>0.958</td><td>0.958</td></tr></table>

The raw counts are pooled over each system’s evaluable outputs, so GT denominators may difer slightly despite the same 382-admission test set.

LOKI’s pair-level errors are dominated by false negatives rather than false positives (Table 7): 2,823 of its 2,882 predicted pairs are correct, but 3,618 ground-truth pairs remain unrecovered. Qwen-3.7 improves primarily by expanding this coverage, reducing missed pairs by roughly one third rather than by improving an already high precision. At the typed-pair level, LOKI’s main limitation is therefore coverage rather than precision.

## RQ2: How practical is discovery-driven integration?

Table 8 reports the computational and monetary cost of end-toend integration. LOKI reduces the downstream LLM workload by approximately 67–69%, requiring only about one third as many tokens as direct prompting. With GPT-OSS as �, the complete pipeline is approximately 2× faster than direct local Qwen-3.6 and 3.9× faster than the frontier API baseline. Under the corresponding API-equivalent pricing assumptions, it is also approximately 11× less expensive than direct Qwen-3.6 and 44× less expensive than the frontier API baseline. Comparing LOKI (�=Qwen-3.6) against direct Qwen-3.6 separates token eficiency from backend latency. LOKI requires approximately 3× fewer LLM tokens and reduces the corresponding token-priced cost by approximately 3.1× relative to direct Qwen-3.6, despite this particular implementation of � taking

approximately 4.4× longer. The additional latency therefore stems from the selected downstream implementation rather than a larger LLM reasoning workload.  
Table 8: Runtime, LLM token consumption, and estimated API-equivalent cost for Experiment 3.
<table><tr><td>System</td><td>Time/Adm. (s)</td><td>Tokens/Adm. (K)</td><td>Total Tokens (M)</td><td>Est. Cost ($)</td></tr><tr><td>LOKI (GPT-OSS)</td><td>44.8</td><td>7.2†</td><td>2.75†</td><td>~0.70</td></tr><tr><td>LOKI (Qwen-3.6)</td><td>391.3</td><td>7.2†</td><td>2.75†</td><td>~2.53</td></tr><tr><td>Qwen-3.6 (Local)</td><td>89.9</td><td>22.1</td><td>8.43</td><td>~7.76</td></tr><tr><td>Qwen-3.7 (API)</td><td>175.9</td><td>23.1</td><td>8.82</td><td>~30.60</td></tr></table>

<sup>†</sup>LOKI tokens cover only the downstream Generalization Function �; <sup>‡</sup>Costs are estimated over 382 admissions using provider token prices.

Summary. Exp. 3 shows that LOKI transforms discovered rowsentence associations into text-mediated join paths, disambiguates their relationships, and materializes them as provenance-preserving Typed Tables. Direct LLMs achieve stronger final materialization, primarily through higher coverage, whereas LOKI additionally performs the preceding discovery while requiring only about one third of the LLM token workload. By restricting LLM reasoning to compact evidence-backed relationship descriptions, the replaceable function � can be selected according to latency, cost, and deployment requirements.

## 6 Conclusion

This paper studies the discovery of latent relational structure between disjoint tables and unstructured text, uncovering not only which sources are relevant but how they are connected. We formalized Text-Mediated Join Path Discovery as the task of identifying and organizing row-sentence associations into coherent relational structures, and introduced LOKI, a horizontal bidirectional cross-attention model that, from coarse-grained supervision alone, induces fine-grained row-sentence associations, composes them into join paths, and organizes these into relation-consistent groups supporting provenance-aware integration.

Candidate discovery currently requires scoring all table-document pairs, with a cost proportional to |T||D|; although these evaluations can be batched, an indexable task-specific retrieval representation remains an important direction for scaling to substantially larger data lakes. A second limitation is coverage: LOKI recovers typed row pairs with high precision but lower recall compared to direct LLM prompting, so improving recall of the underlying rowsentence associations is the clearest path to stronger end-to-end integration. A third limitation concerns relationship labeling: as a joint table-text representation model, LOKI delegates canonical relationship naming to the Generalization Function �. This concerns only downstream ontology-specific labeling, not the discovery of the underlying relational structure; future work aims to improve � via ontology-aware calibration and learnable naming.

Our evaluation across four complementary datasets demonstrates this progression from representation learning to multi-modal discovery and, ultimately, integration, at roughly one third of the LLM token cost of competing approaches. More broadly, unstructured text can serve as explicit relational evidence that makes disconnected data interpretable, auditable, and queryable.

## References

[1] Rami Aly, Zhijiang Guo, Michael Sejr Schlichtkrull, James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Oana Cocarascu, and Arpit Mittal. 2021. FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information. In NeurIPS Datasets and Benchmarks

[2] Angelos-Christos G. Anadiotis, Oana Balalau, Catarina Conceição, Helena Galhardas, Mhd Yamen Haddad, Ioana Manolescu, Tayeb Merabti, and Jingmao You. 2022. Graph integration of structured, semistructured and unstructured data for data journalism. Inf. Syst. 104 (2022), 101846.

[3] Alan Arazi, Eilam Shapira, and Roi Reichart. 2025. TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields. In NeurIPS

[4] Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, Yue Gong, Adila Krisnadhi, and Raul Castro Fernandez. 2025. Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System. Proc. ACM Manag. Data 3, 3 (2025), 200:1–200:28.

[5] Randall Balestriero and Yann LeCun. 2025. LeJEPA: Provable and Scalable Self Supervised Learning Without the Heuristics. CoRR abs/2511.08544 (2025).

[6] Kevin Dharmawan, Chirag A. Kawediya, Yue Gong, Zaki Indra Yudhistira, Zhiru Zhu, Sainyam Galhotra, Adila Alfa Krisnadhi, and Raul Castro Fernandez. 2024. Demonstration of Ver: View Discovery in the Wild. In SIGMOD Conference Companion. ACM, 428–431.

[7] Yuyang Dong, Chuan Xiao, Takuma Nozawa, Masafumi Enomoto, and Masafumi Oyamada. 2023. DeepJoin: Joinable Table Discovery with Pre-trained Language Models. Proc. VLDB Endow. 16, 10 (2023), 2458–2470.

[8] Mohamed Y. Eltabakh, Mayuresh Kunjir, Ahmed K. Elmagarmid, and Mohammad Shahmeer Ahmad. 2023. Cross Modal Data Discovery over Structured and Unstructured Data Lakes. Proc. VLDB Endow. 16, 11 (2023), 3377–3390.

[9] Raul Castro Fernandez, Ziawasch Abedjan, Famien Koko, Gina Yuan, Samuel Madden, and Michael Stonebraker. 2018. Aurum: A Data Discovery System. In ICDE. IEEE Computer Society, 1001–1012

[10] Raul Castro Fernandez and Samuel Madden. 2019. Termite: a system for tunneling through heterogeneous data. In aiDM@SIGMOD. ACM, 7:1–7:8

[11] Raul Castro Fernandez, Essam Mansour, Abdulhakim Ali Qahtan, Ahmed K. Elmagarmid, Ihab F. Ilyas, Samuel Madden, Mourad Ouzzani, Michael Stonebraker, and Nan Tang. 2018. Seeping Semantics: Linking Datasets Using Word Embeddings for Data Discovery. In ICDE. IEEE Computer Society, 989–1000.

[12] Sainyam Galhotra, Yue Gong, and Raul Castro Fernandez. 2023. Metam: Goal-Oriented Data Discovery. In ICDE. IEEE, 2780–2793.

[13] Aude Genevay, Gabriel Peyré, and Marco Cuturi. 2018. Learning Generative Models with Sinkhorn Divergences. In AISTATS (Proceedings ofMachine Learning Research), Vol. 84. PMLR, 1608–1617.

[14] Yue Gong, Zhiru Zhu, Sainyam Galhotra, and Raul Castro Fernandez. 2023. Ver: View Discovery in the Wild. In ICDE. IEEE, 503–516.

[15] Yuxiang Guo, Yuren Mao, Zhonghao Hu, Lu Chen, and Yunjun Gao. 2025. Snoopy: Efective and Eficient Semantic Join Discovery via Proxy Columns. IEEE Trans. Knowl. Data Eng. 37, 5 (2025), 2971–2985.

[16] Mossad Helali, Niki Monjazeb, Shubham Vashisth, Philippe Carrier, Ahmed Helal, Antonio Cavalcante, Khaled Ammar, Katja Hose, and Essam Mansour. 2024. KGLiDS: A Platform for Semantic Abstraction, Linking, and Automation of Data Science. In ICDE. IEEE, 179–192.

[17] Junjie Huang, Wanjun Zhong, Qian Liu, Ming Gong, Daxin Jiang, and Nan Duan. 2022. Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in OpenQA. In EMNLP (Findings). Association for Computational Linguistics, 4117–4129.

[18] Maximilian Ilse,Jakub M. Tomczak, and Max Welling. 2018. Attention-based Deep Multiple Instance Learning. In Proceedings ofthe 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018 (Proceedings of Machine Learning Research), Jennifer G. Dy and Andreas Krause (Eds.), Vol. 80. PMLR, 2132–2141. http://proceedings.mlr.press/v80 ilse18a.html

[19] Saehan Jo and Immanuel Trummer. 2024. ThalamusDB: Approximate Query Processing on Multi-Modal Data. Proc. ACM Manag. Data 2, 3 (2024), 186.

[20] Alistair Johnson, Lucas Bulgarelli, Tom Pollard, Brian Gow, Benjamin Moody, Steven Horng, Leo Anthony Celi, and Roger Mark. 2024. MIMIC-IV. PhysioNet (Oct. 2024). https://doi.org/10.13026/kpb9-mt58 Version 3.1.

[21] Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. 2023. MIMIC-IV, a freely accessible electronic health record dataset. Scientific data 10, 1 (2023), 1.

[22] Aamod Khatiwada, Grace Fan, Roee Shraga, Zixuan Chen, Wolfgang Gatterbauer, Renée J. Miller, and Mirek Riedewald. 2023. SANTOS: Relationship-based Semantic Table Union Search. Proc. ACM Manag. Data 1, 1 (2023), 9:1–9:25.

[23] Bogdan Kostić, Julian Risch, and Timo Möller. 2021. Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models. In Proceedings ofthe 3rd Workshop on Machine Reading for Question Answering, Adam Fisch, Alon Talmor, Danq Chen, Eunsol Choi, Minjoon Seo, Patrick Lewis, Robin Jia, and Sewon Min (Eds.). Association for Computational Linguistics, Punta Cana, Dominican Republic,

82–91. https://doi.org/10.18653/v1/2021.mrqa-1.8

[24] Christos Koutras, Jiani Zhang, Xiao Qin, Chuan Lei, Vasileios Ioannidis, Christos Faloutsos, George Karypis, and Asterios Katsifodimos. 2025. OmniMatch: Joinability Discovery in Data Products. Proc. VLDB Endow. 18, 11 (Sept. 2025), 4588–4601. https://doi.org/10.14778/3749646.3749715

[25] Essam Mansour, Dong Deng, Raul Castro Fernandez, Abdulhakim Ali Qahtan, Wenbo Tao, Ziawasch Abedjan, Ahmed K. Elmagarmid, Ihab F. Ilyas, Samuel Madden, Mourad Ouzzani, Michael Stonebraker, and Nan Tang. 2018. Building Data Civilizer Pipelines with an Advanced Workflow Engine. In ICDE. IEEE Computer Society, 1593–1596.

[26] Marc Maynou, Sergi Nadal, Raquel Panadero, Javier Flores, Oscar Romero, and Anna Queralt. 2024. FREYJA: Eficient Join Discovery in Data Lakes. CoRR abs/2412.06637 (2024).

[27] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. 2020. End-to-End Learning of Visual Representations From Uncurated Instructional Videos. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. Computer Vision Foundation / IEEE, 9876–9886. https://doi.org/10.1109/CVPR42600.2020. 00990

[28] Mir Mahathir Mohammad and El Kindi Rezig. 2026. Qualitative Join Discovery in Data Lakes using Examples. Proc. ACM Manag. Data 4, 1 (SIGMOD) (May 2026), 68:1–68:28.

[29] Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. CoRR abs/2505.06708 (2025).

[30] Md. Ataur Rahman, Sergi Nadal, Oscar Romero, and Dimitris Sacharidis. 2024. Mitigating Data Sparsity in Integrated Data through Text Conceptualization. In ICDE. IEEE, 3490–3504.

[31] Alexander J. Ratner, Christopher De Sa, Sen Wu, Daniel Selsam, and Christo pher Ré. 2016. Data Programming: Creating Large Training Sets, Quickly. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett (Eds.). 3567–3575. https://proceedings.neurips.cc/paper/2016/hash 6709e8d64a5f47269ed5cea9f625f7ab-Abstract.html

[32] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In EMNLP/IJCNLP (1). Association for Computational Linguistics, 3980–3990.

[33] Matthias Urban and Carsten Binnig. 2024. ELEET: Eficient Learned Query Execution over Text and Tables. Proc. VLDB Endow. 17, 13 (2024), 4867–4880.

[34] Jiayi Wang, Guoliang Li, and Jianhua Feng. 2025. iDataLake: An LLM-Powered Analytics System on Data Lakes. IEEE Data Eng. Bull. 49, 1 (2025), 57–69.

[35] Zirui Wu and Yansong Feng. 2024. ProTrix: Building Models for Planning and Reasoning over Tables with Sentence Context. In EMNLP (Findings) (Findings of ACL), Vol. EMNLP 2024. Association for Computational Linguistics, 4378–4406.

[36] Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020. TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data. In ACL. Association for Computational Linguistics, 8413–8426.