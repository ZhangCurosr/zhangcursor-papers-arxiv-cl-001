# DiscoSign: Discourse-Aware Text to Sign Language Gloss Translation

Vasileios Baltatzis \* Apple

Mert Inan \* <sup>†</sup> Northeastern University

Raja Kushalnagar <sup>†</sup> Gallaudet University

Lorna Quandt <sup>†</sup> Gallaudet University

Connor Gillis Apple

Leah Findlater Apple

Colin Lea Apple

## Abstract

Sign language processing systems have traditionally operated at the sentence level, ignoring critical discourse phenomena fundamental to sign language comprehension. We introduce DiscoSign, a computational approach for discourse-aware text to sign language gloss translation grounded in linguistic research. We address three key phenomena within our modular Large Language Model (LLM)-based translation framework: (i) spatial coreference resolution, where entities maintain consistent spatial locations throughout discourse; (ii) Question-Answer Clauses (QACs), pseudocleft structures serving specific discourse functions; and (iii) concept-gloss consistency, ensuring stable mappings between English concepts and American Sign Language (ASL) signs. Traditional translation metrics fail to capture discourse-level quality, so we introduce a suite of novel evaluation metrics designed to assess each dimension of discourse coherence addressed by our framework. Experiments on sentence-level and discourse-level datasets show that our approach for discourse-aware processing significantly improves spatial consistency and entity tracking relative to sentence-only translation, while maintaining competitive single-sentence gloss translation quality. Our work establishes the first systematic framework for discourse-level text to sign language gloss translation with corresponding evaluation methodology.

## 1 Introduction

Current approaches in text-to-sign language generation and translation systems predominantly operate at the sentence level. Each sentence is translated independently, without mechanisms to track how entities, concepts, and rhetorical choices should remain consistent across a multi-sentence text. As a result, these systems fail to capture discourse-level phenomena that are essential for coherent communication, which can have direct consequences for practical sign translation systems.

![](images/cb9eebc67c841f13d40aa39114f22f034f681dc1f9bd16d0d1e250eb381e6cf0.jpg)  
Figure 1: Sentence-level translation violates spatial coreference and concept-gloss consistency, and misses a context where a QAC is appropriate. Our framework maintains discourse coherence through consistent spatial indices, stable concept-gloss mappings, and appropriate QAC usage.

Most such systems rely on text-to-gloss translation as an intermediate step. Glosses are written labels denoting individual signs (e.g., CAT, BIRD), providing a representation that can be mapped to sign dictionaries for downstream sign production. The available gloss-sign dictionary defines the vocabulary constraints for the text-to-gloss task and generated glosses must correspond to entries in the dictionary. Discourse-level errors introduced at the gloss stage propagate through the entire pipeline.

Signed languages, such as American Sign Language (ASL), employ sophisticated grammatical structures including spatial indexing and rhetorical questions, which have discourse-level consistency requirements. Spatial coreference relies on a complex spatial reference system where entities are assigned specific spatial coordinates (e.g., ASL glosses of: IX-3p:i, IX-3p:j referring to third person objects/subjects)<sup>1</sup> that must remain consistent throughout discourse (Winston, 1991). A signer may establish several referents in a scene and later reference them by pointing to the physical locations that were previously established. Sentence-level systems have no mechanism to track these assignments as a referent established as IX-3p:i in one sentence may be arbitrarily reassigned to a different location in the next. Question-Answer Clauses (QACs), or pseudoclefts, are sentence structures used to emphasize information, starting with a rhetorical question word like "what" or "where" and then providing the answer. The English declarative "I want coffee" could be phrased as "What I want is coffee," and in ASL gloss as "I WANT WHAT? COFFEE." QAC usage varies depending on discourse context and information structure; with out access to this context, systems cannot identify when such restructuring is appropriate. Conceptgloss consistency requires that the same concept map to the same gloss throughout discourse to maintain cohesion. For example, the English word “computer” appearing five times should be rendered with the same ASL sign throughout, rather than alternating between COMPUTER, MACHINE, and TECHNOLOGY. Sentence-level systems, lacking this context, may render the same concept inconsistently across sentences.

In this paper, we introduce, DiscoSign, the first framework of discourse-aware text-to-gloss translation. This work makes four primary contributions to computational sign language processing:

1. Discourse-aware text-to-gloss translation framework: We present the first systematic approach to sign language gloss translation from text that maintains cross-sentence coherence through explicit state registries and programmatic constraint enforcement for spatial coreference resolution, pseudocleft usage, and concept-gloss consistency.

2. ASL-centric instruction design methodology: We develop a systematic approach to embedding specialized ASL linguistic expertise into Large Language Model (LLM) prompts, demonstrating how complex discourse phenomena can be addressed through careful instruction design.

3. Comprehensive evaluation framework: We establish novel evaluation metrics for discourse-level text-to-gloss translation including spatial indexing consistency measurement, pseudocleft appropriateness assessment, and concept-gloss mapping consistency evaluation through cross-sentence reasoning tasks.

4. Systematic instruction component analysis: We provide detailed ablation studies demonstrating the individual and combined contributions of different instruction components, establishing the relative importance of spatial indexing consistency, and discourse phenomena for translation quality, and we show that these gains hold for both a proprietary and an open-weight LLM backbone.

## 2 Related Work

Sign Language Linguistics and Processing Sign languages exhibit rich linguistic structures that present unique challenges for computational modeling (Yin et al., 2021b). Prior work has addressed various linguistic phenomena including coreference resolution with spatial agreement (Yin et al., 2021a) and non-manual markers such as facial expressions and prosodic intensification (Zhang et al., 2025; Viegas et al., 2023; Inan et al., 2022), alongside system-level and productionside considerations for signing avatars (Dougherty, 2025; Baltatzis et al., 2024). The challenges facing sign language NLP are compounded by broader issues affecting low-resource languages (Court and Elsner, 2024; Zhong et al., 2025) and demographic biases in model performance (Atwell et al., 2024). Discourse-relevant phenomena were also identified in earlier computational work on sign language translation: Stein et al. (2007) showed that the spatial position of deictic signs carries referential information that helps disambiguate pronouns when translating from a signed language into English, and Lugaresi and Di Eugenio (2013) analyzed how Italian discourse connectives are realized in Italian Sign Language, observing that inter-sentential connectives are frequently dropped. Both establish that reference and discourse relations behave differently in signed languages and must be handled explicitly. Our work takes up the complementary computational problem of tracking and enforcing such discourse coherence across multi-sentence texts, through shared state that persists between sentences and metrics that measure whether coherence is maintained. The pragmatics of signed discourse presents particularly rich phenomena: Caponigro and Davidson (2011) analyzed Question-Answer Clauses (QACs) as topic-comment structures governed by Question Under Discussion theory, while Wilbur (1994) identified foregrounding structures where Q-constituents convey background information and A-constituents introduce new information. Advances in multilingual discourse prediction (Eichin et al., 2025) suggest promising directions for modeling such phenomena. Building on this rich linguistic background, we develop the first discourse-aware sign language generation framework.

LLMs for Sign Language Translation The integration of large language models into sign language processing has emerged as a promising research direction. Recent work has demonstrated that LLMs possess inherent capabilities for sign language translation (Gong et al., 2024), with approaches ranging from gloss-free translation via direct visual-to-text mapping (Wong et al., 2024; Hwang et al., 2025) to vocabulary sharing strategies for cross-modal knowledge transfer (Lee et al., 2023). Training strategies have focused on semantically aware label smoothing (Fayyazsanavi et al., 2024) and techniques for low-resource scenarios (Bulla et al., 2025). More recently, Inan et al. (2025) introduced SignAlignLM, the first framework to natively integrate multimodal sign language processing into LLMs, in addition to practical deployment in large-scale dialogue systems (Inan et al., 2024). While these advances have improved translation accuracy, existing approaches predominantly focus on sentence-level translation and neglect discourse-level pragmatic phenomena essential for natural signed communication.

Evaluation of Sign Language Systems Evaluation methods for sign language systems have historically relied on metrics borrowed from machine translation, which fail to capture visual-spatial and discourse-level properties of signed languages (Yin et al., 2021b). Standard metrics such as BLEU (Papineni et al., 2002) or ROUGE (Lin, 2004) are insufficient for assessing sign language translation quality, as they focus on lexical overlap and penalize translations that exhibit various nuances of sign languages (e.g. flexible word order, omission of function words, and spatial-visual elements) (Müller et al., 2023; Lea et al., 2026). Metrics such as chrF (Popovic´, 2015) which compares character n-grams and is inherently more robust to word order differences, or COMET (Rei et al., 2020), which captures semantic similarity and aligns with human judgment, are better suited. Imai et al. (2025) introduced SilverScore, a semantically aware embedding-based metric demonstrating robustness to semantic and prosodic variations. However, existing metrics cannot evaluate pragmatic appropriateness or discourse coherence, failing to distinguish semantically correct but pragmatically awkward translations, such as using declarative word order where QAC structures are preferred (Caponigro and Davidson, 2011; Wilbur, 1994). Our proposed evaluation suite (Section 4) addresses these gaps through novel metrics for discourse-level phenomena specific to signed languages.

## 3 Discourse-Aware Sign Language Gloss Translation Framework

We introduce DiscoSign, the first computational framework for discourse-aware text to sign language gloss translation. Section 3.1 presents the linguistic foundations underlying our approach, detailing the three discourse phenomena that current systems fail to address: spatial coreference, question-answer clauses, and concept-gloss consistency. Section 3.2 then describes our modular framework, which operationalizes each phenomenon through a dedicated module with explicit state representations and constraint enforcement.

## 3.1 Linguistic Foundations

This section presents our framework’s linguistic foundations, formalizing three phenomena central to sign language discourse coherence: spatial coreference, question-answer clauses, and concept-gloss consistency. We illustrate these using the example in Figure 1.

Spatial Coreference Resolution Sign language discourse requires a unified spatial reference system that maintains both entity tracking and spatial consistency (Yin et al., 2021a). Unlike spoken languages that rely on morphological agreement, sign languages use spatial indexing where entities are assigned consistent spatial locations (IX-3p:i, IX-3p:j) and directional relationships (i:GIVE:j, 1p:MEET:i) throughout discourse. This system must coordinate entity coreference resolution with spatial grammar, ensuring that pronouns, definite descriptions, and repeated mentions resolve to correct antecedents through appropriate spatial index coordination across sentence boundaries. In Figure 1, Alice is established at IX-3p:i and Bob at IX-3p:j; subsequent pronouns must reference these locations consistently, and the possessive "his" must be rendered as POSS-j to maintain the link to Bob.

Question-Answer Clauses (QACs) QACs are topic-comment structures that are widely-occurring grammatical components of many sign languages (Wilbur, 1994; Caponigro and Davidson, 2011). In American Sign Language, this topic-comment structure is specialized into QACs, which consist of a preamble question that is directly answered by a following declarative statement (Caponigro and Davidson, 2011). From a syntactic perspective, QACs are argued to be pseudocleft structures with silent copula (Wilbur, 1994).

QACs serve specific discourse functions and must align with contextual requirements for natural sign language discourse. They are appropriate in contexts such as causal explanations, topic introductions, or emphasis, but should not appear at discourse beginnings or as direct responses to genuine questions. Awkward QAC usage can disrupt discourse flow and reduce comprehension (Wilbur, 1994). Identifying when QAC restructuring is appropriate therefore requires access to discourse context that sentence-level systems lack. In Figure 1, the English clause "as he would take the bus" expresses Bob’s reason for agreeing; the discourse-aware translation restructures this as a QAC that topicalizes the causal relationship: "IX-3p:j AGREE WHY? IX-3p:j TAKE BUS".

Concept-Gloss Consistency Discourse coherence requires stable concept-to-sign mappings to reduce cognitive load and maintain semantic clarity (Halliday and Hasan, 1976; Frederiksen, 2019). The same English concepts should consistently translate to the same ASL glosses throughout discourse while respecting vocabulary constraints. In Figure 1 the English passage uses both car and vehicle to refer to the same object. A discourseaware system recognizes this coreference and renders both as CAR; a sentence-level system might produce CAR and VEHICLE, creating ambiguity about whether two objects are being discussed.

![](images/d37ad93cef168741a270257d1c78d80a955820f214d69757d845e9eb7a429f50.jpg)  
Figure 2: The DiscoSign pipeline. Each sentence is translated in a single LLM call whose prompt states the accumulated discourse registries as hard constraints. A deterministic post-processor corrects any remaining violations, and the verified metadata updates the registries for the next sentence.

These phenomena interact throughout discourse: spatial coreference and concept-gloss consistency both require entity tracking, while QAC usage depends on information spanning multiple clauses. The following sections describe how our framework addresses each through dedicated modules.

## 3.2 DiscoSign Modules

We operationalize these linguistic phenomena through three interconnected modules, each with an explicit state representation and a set of constraints. Before giving the formal definitions, we describe the end-to-end pipeline, shown in Figure 2.

The framework translates a text one sentence at a time, carrying discourse state forward in three registries: S for spatial assignments, L for conceptgloss mappings, and $\mathcal { Q }$ for QAC decisions. For each sentence $s _ { t } .$ , the framework (i) retrieves the registries accumulated over sentences $s _ { 1 } \ldots s _ { t - 1 } ;$ (ii) constructs a single prompt containing $s _ { t } ,$ a context window of previous sentence–translation pairs, and the registry contents stated as hard constraints; (iii) receives a JSON response with the gloss translation and structured metadata (spatial mappings, concept mappings, and a QAC flag); (iv) runs a deterministic post-processor that checks the output against the registries and corrects violations by string replacement, without re-querying the model; and (v) updates the registries from the verified metadata, so that they constrain $s _ { t + 1 }$ . The modules are therefore not separate model calls: each is a structured prompt section together with its registry and verification logic, all within a single LLM call per sentence.

The rest of this section formalizes each module’s state and constraints. Prompt templates are given in Appendix A and the verification procedure in Appendix A.3.

Spatial Coreference Module (SCM) Given a new sentence, the SCM identifies entity mentions, resolves coreference with previously established entities, and either retrieves existing spatial assignments or assigns new indices for novel entities. The module outputs spatially-indexed glosses (e.g., $I X  – 3 p { : } i , P O S S  – j )$ that maintain referential consistency throughout the discourse. We formalize this by defining the module’s state as tracking spatial assignments and their corresponding entities:

$$
\boldsymbol { \mathcal { S } } = \{ ( e , i , t ) : e \in \mathcal { E } , i \in \mathcal { T } , t \in \mathcal { T } \}\tag{1}
$$

where e is an entity, i is a spatial index, and t is the sentence where the assignment was established. The SCM enforces two primary constraints:

• Spatial Consistency: For any entity e, if $( e , i , t ) \in \mathcal { S } _ { }$ , then all subsequent references to e in sentences $t ^ { \prime } > t$ use spatial index i.

• Coreference Correspondence: For any spatial reference claiming entity e uses index i, there must exist a coreference chain $\mathcal { C } _ { e }$ in the English text that includes the claimed mention.

For directional verbs $i _ { s o u r c e } : V E R B : i _ { t a r g e t } ,$ both source and target indices must satisfy these constraints independently.

Question-Answer Clause Module (QACM) Given a new sentence and its discourse context, the QACM analyzes whether the sentence contains information suitable for topicalization, such as causal explanations, contrastive elements, or focal points. If appropriate, the module selects the corresponding QAC type and restructures the output accordingly, ensuring that QAC usage patterns remain consistent with ASL discourse conventions throughout the text. We formalize this by defining the module’s state as a registry of QAC decisions across the discourse:

$$
\mathcal { Q } = \{ ( t , q ) : t \in \mathcal { T } , q \in \{ 0 , 1 \} \}\tag{2}
$$

where t is the sentence index and q indicates whether a QAC structure was used. The QACM enforces two constraints:

• Discourse Position: QACs should not appear at discourse beginnings (t=0) or as responses to genuine questions in the source text.

• Contextual Appropriateness: QAC insertion depends on information structure spanning multiple clauses; the module evaluates preceding discourse to determine whether topicalization is warranted.

Concept-Gloss Consistency Module (CGCM) Given a new sentence, the CGCM identifies content words and checks whether they correspond to concepts already established in prior context. For known concepts, the module retrieves and applies the existing gloss mapping. For novel concepts, it selects an appropriate gloss from the available vocabulary and registers the new mapping. This process ensures that synonyms or paraphrases in the English source (e.g., “car” and “vehicle”) resolve to a single consistent gloss throughout discourse. We formalize the module’s state as a concept registry L that tracks established mappings:

$$
\mathcal { L } = \{ ( c , g , t ) : c \in \mathcal { C } , g \in \mathcal { V } , t \in \mathcal { T } \}\tag{3}
$$

where c is an English concept, g is an ASL gloss from vocabulary V, and t is the sentence where the mapping was established. The CGCM enforces:

• Mapping Consistency: For any concept c, if $( c , g , t ) \in \mathcal { L }$ , then all subsequent translations of c in sentences $t ^ { \prime } > t$ must use gloss g.

• Vocabulary Compliance: All glosses must satisfy $g \in \mathcal { V } ,$ ensuring mappings respect the constraints imposed by available dictionaries.

Integrated Processing The framework coordinates all modules through a unified state $u \ =$ $( S , { \mathcal { L } } , { \mathcal { Q } } )$ . For each input sentence, the three modules operate in coordination. Each module’s decisions are informed by and constrained by the others. For instance, a QAC restructuring must preserve the spatial indices established by the SCM.

The unified state U is updated after each sentence: the structured output (spatial mappings, concept mappings, QAC usage) is extracted and used to update the registries for subsequent sentences.

Constraint Enforcement Module constraints are enforced programmatically rather than relying solely on LLM instruction-following. Before each LLM call, the prompt is constructed as a deterministic function of the current registry state and sentence position. After each call, a verification step checks the output against the registries: spatial index and concept-gloss violations are corrected via string replacement throughout the translation. All enforcement is deterministic and requires no re-querying (Appendix A.3).

## 4 Evaluation Suite

We introduce novel metrics aligned with each module of our framework. Each metric evaluates a specific aspect of discourse coherence that traditional translation metrics cannot capture, providing the first systematic evaluation of cross-sentence phenomena in text to sign language gloss translation.

## 4.1 Spatial Coreference Accuracy (SCA)

SCA evaluates the SCM by measuring how accurately spatial indexing in ASL translations preserves coreference relationships from the source English discourse. Intuitively, if Alice is initially assigned to location IX-3p:i, all subsequent references to Alice must use the same location. We evaluate spatial references excluding first definitions, where entities are initially assigned to spatial locations. For directional verbs $( i ; V E R B ; j )$ , both source and target indices are evaluated independently. For each spatial reference r claiming entity e at index i in sentence t, we perform two checks.

The spatial consistency check verifies that entity e has not been previously assigned a different spatial index:

$$
\begin{array} { r } { \mathrm { s p a t i a l \_ c o n s i s t e n t } ( r ) = \nexists ( e , i ^ { \prime } , t ^ { \prime } ) \in S : } \\ { i ^ { \prime } \neq i \land t ^ { \prime } < t } \end{array}\tag{4}
$$

The coreference correspondence check verifies that the claimed entity e has a mention in the aligned English sentence according to the coreference chain $\mathcal { C } _ { e }$ :

$$
\begin{array} { c } { \mathrm { c o r e f e r e n c e \_ v a l i d } ( r ) = \exists m \in \mathcal { C } _ { e } : } \\ { m \mathrm { a p p e a r s i n s e n t e n c e } t } \end{array}\tag{5}
$$

SCA is calculated as the proportion of spatial references passing both checks:

$$
\mathrm { S C A } = \frac { 1 } { | R | } \sum _ { { r \in { \cal R } } } [ \mathrm { s p a t i a l \_ c o n s i s t e n t } ( r ) \wedge\tag{6}
$$

where R represents all spatial references in the ASL translation. SCA addresses the core challenge of ASL discourse translation: maintaining consistent spatial-entity mappings across sentences while preserving referential relationships. This metric captures phenomena that traditional word-level metrics cannot assess, providing the first systematic evaluation of ASL discourse coherence.

## 4.2 Question-Answer Clause Appropriateness (QAC<sub>Ap</sub>)

${ \mathrm { Q A C } } _ { \mathrm { A p } }$ evaluates the QACM by measuring whether QACs appear in linguistically suitable contexts. For instance, a QAC is appropriate when translating “He agreed because he would take the bus” (causal relationship) but not when directly answering a genuine question like “Why did he agree?” in the source text. For each sentence $t \in \mathcal T$ where a QAC is used, we evaluate:

$$
\begin{array} { c } { { \mathrm { a p p r o p r i a t e } ( t ) = \mathbb { I } [ \neg \mathrm { a n s w e r s Q u e s t i o n } ( t - 1 ) ] \wedge } } \\ { { \mathbb { I } [ \mathrm { h a s C o n t e x t } ( t ) ] } } \end{array}\tag{7}
$$

where I[·] is the indicator function that returns 1 if the condition is true and 0 otherwise, hasContext(t) indicates the presence of causal, topical, or emphasis discourse markers, and the question-answering check applies only when $t > 1$ The overall QAC appropriateness is calculated as:

$$
\mathrm { Q A C _ { A p } } = \frac { \sum _ { t \in \mathcal { T } _ { Q A C } } \mathrm { a p p r o p r i a t e } ( t ) } { | \mathcal { T } _ { Q A C } | }\tag{8}
$$

where $\mathcal { T } _ { Q A C } \subseteq \mathcal { T }$ represents the set of sentences where QACs were used. This metric focuses on precision rather than recall, evaluating whether QAC usage decisions were contextually appropriate rather than penalizing missed opportunities, as QACs are optional discourse enhancement tools.

## 4.3 Concept-Gloss Consistency (CGC)

CGC evaluates the CGCM by measuring whether English concepts receive consistent ASL gloss translations throughout discourse. Inconsistent mappings, such as rendering “car” as CAR in one sentence and VEHICLE in another, can disrupt a signers’s ability to track entities across discourse. For each concept c appearing in sentence t as gloss g, we verify consistency with prior occurrences:

$$
\begin{array} { r } { \mathrm { c o n c e p t \_ c o n s i s t e n t } ( c , g , t ) = \nexists ( c , g ^ { \prime } , t ^ { \prime } ) \in \mathcal { L } : } \\ { g ^ { \prime } \ne g \wedge t ^ { \prime } < t } \end{array}\tag{9}
$$

CGC is calculated as the proportion of repeated

<table><tr><td>Dataset</td><td>Examples</td><td>Sents./Ex.</td><td>Gold</td><td>Disc.</td></tr><tr><td>ASL STEM Wiki</td><td>500</td><td>1</td><td>√</td><td>x</td></tr><tr><td>Licensed Dataset</td><td>2121</td><td>1</td><td>√</td><td>x</td></tr><tr><td>Aesop&#x27;s Fables</td><td>284</td><td>5</td><td>x</td><td>√</td></tr></table>

Table 1: Evaluation datasets. Examples: sentence pairs (ASL STEM Wiki (Yin et al., 2024), Licensed Dataset) or texts (Aesop’s Fables (Aesop, 1912)). Sents./Ex.: average sentences per example. Gold: expert-annotated reference glosses available. Disc.: contains discourselevel phenomena.

concepts that maintain consistent mappings:

$$
{ \mathrm { C G C } } = { \frac { \sum _ { c \in { \mathcal { C } } _ { r e p e a t e d } } { \mathrm { c o n c e p t \Pi } } _ { - } { \mathrm { c o n s i s t e n t } } ( c ) } { | { \mathcal { C } } _ { r e p e a t e d } | } }\tag{10}
$$

where $\mathcal { C } _ { r e p e a t e d }$ represents concepts that appear in multiple sentences. CGC evaluates conceptgloss consistency across all discourse-level concept repetitions, ensuring that gloss choices support discourse coherence. This validation captures the fundamental requirement that identical concepts should receive identical gloss translations throughout discourse.

## 5 Experiments and Findings

We conduct comprehensive experiments to validate DiscoSign across two complementary evaluation paradigms: sentence-level translation accuracy and discourse-level coherence assessment.

## 5.1 Experimental Setup

Datasets. We evaluate on three datasets spanning sentence-level and discourse-level phenomena (Table 1). ASL STEM Wiki and a licensed dataset provide sentence-level evaluation with expert-annotated ground truth glosses. The licensed dataset pairs English sentences describing everyday activities with ASL glosses, and was collected and annotated by a team of Deaf signers fluent in both ASL and English. ASL STEM Wiki contains scientific terminology requiring substantial fingerspelling. Aesop’s Fables provides discourse-level evaluation with multi-sentence narratives containing coreference chains, causal relationships, and topic transitions.

System Configurations. We compare three system configurations: (1) Sentence-level: Gemini 2.5 Pro with access only to the current sentence and no discourse-specific instructions, (2) Context-Aware: the same model with access to discourse context (i.e. previous sentences) but no specialized discourse instructions, and (3) Proposed Framework: Our proposed framework with all three discourse modules (SCM, QACM, CGCM) implemented as structured prompt sections with explicit state registries and post-processing verification. All systems use the ASLLRP SignBank (Neidle et al., 2022) lexicon (2,859 glosses) as vocabulary constraints and deterministic sampling (temperature=0).

<table><tr><td colspan="2"></td><td colspan="2">Gloss</td><td colspan="2">Back-translation</td></tr><tr><td>Dataset</td><td>Approach</td><td>chrF</td><td>COMET</td><td>chrF</td><td>COMET</td></tr><tr><td rowspan="2">ASL STEM Wiki</td><td>Sentence-level</td><td>28.2</td><td>0.54</td><td>41.9</td><td>0.73</td></tr><tr><td>Proposed</td><td>30.1</td><td>0.51</td><td>54.8</td><td>0.81</td></tr><tr><td rowspan="2">Licensed Dataset</td><td>Sentence-level</td><td>39.6</td><td>0.59</td><td>65.2</td><td>0.88</td></tr><tr><td>Proposed</td><td>39.1</td><td>0.57</td><td>67.2</td><td>0.90</td></tr></table>

Table 2: Sentence-level translation results on ASL STEM Wiki and the licensed dataset.

Evaluation Details. Every LLM call, both in the translation pipeline and in evaluation, uses Gemini 2.5 Pro unless stated otherwise; §5.3 additionally reports the framework with an open-weight backbone. For back-translation evaluation, we prompt Gemini 2.5 Pro without discourse-specific instructions to translate generated glosses back to English, then compare against the original source. For discourse-level evaluation, the English coreference chains required by SCA were extracted with Gemini 2.5 Pro and then validated by a human expert. Similarly, the structured outputs generated during translation (spatial mappings, concept-gloss mappings) were verified by a human expert for faithfulness to the ASL translations. All prompts and configurations are provided in Appendix A.

## 5.2 Sentence-Level Translation Quality

We first validate that our discourse-aware framework maintains competitive sentence-level translation quality on datasets where discourse phenomena are absent. Table 2 presents results on both ASL STEM Wiki and the licensed dataset, providing both gloss-level and back-translation evaluation.

Discourse Integration Does Not Compromise Sentence-Level Quality. Our proposed framework demonstrates competitive or superior performance across both datasets, confirming that discourseaware capabilities do not degrade sentence-level translation. The lower absolute scores on ASL STEM Wiki reflect its challenging scientific vocabulary requiring extensive fingerspelling, compared to the licensed dataset’s everyday language.

Notably, the framework shows consistent improvements in back-translation quality (ASL STEM Wiki chrF: 54.8 vs 41.9, COMET: 0.81 vs 0.73; licensed dataset chrF: 67.2 vs 65.2, COMET: $0 . 9 0 \mathrm { v s } 0 . 8 8 )$ , suggesting that the enhanced linguistic reasoning developed for discourse processing also benefits sentence-level semantic preservation.

Gloss-Level and Back-Translation Metrics Capture Different Aspects. Gloss-level evaluation yields lower scores than back-translation due to two factors: different datasets employ different gloss conventions, directly impacting string-matching metrics; and gloss metrics measure strict string similarity while back-translation tolerates surfacelevel variation. The higher back-translation scores indicate both approaches preserve meaning despite surface-level differences from gloss labels.

## 5.3 Discourse-Level Translation Quality

We evaluate discourse-aware translation on Aesop’s Fables using our proposed metrics alongside traditional MT metrics. Table 3 presents comparative results across all system configurations and both LLM backbones.

Traditional Metrics Show Limited Sensitivity. All three Gemini-based approaches achieve comparable chrF (39.4–41.7) and COMET (0.76–0.78) scores despite dramatically different discourse coherence, confirming that traditional MT metrics cannot capture spatial consistency, coreference resolution, or pseudocleft structure requirements fundamental to sign language discourse comprehension.

Explicit Discourse Modeling Outperforms Context Access Alone. While providing context improves over no-context processing (SCA: 0.29 → $0 . 4 8 ; \mathrm { C G C : 0 . 5 9 }  0 . 6 8 ; \mathrm { Q A C _ { A p } : 0 . 7 0  0 . 7 2 ) }$ our structured framework achieves substantially higher performance across all discourse metrics (SCA: 0.84; CGC: 0.97; $\mathrm { \Delta Q A C _ { A p } : }$ 0.76). This demonstrates that access to prior sentences is helpful but insufficient. Explicit modeling of spatial coreference resolution, pseudocleft structures, and concept-gloss consistency is necessary for discourse-aware translation. The dramatic improvements captured by SCA and CGC, coupled with the limited sensitivity of MT metrics, validate both our framework’s effectiveness and the need for discourse-specific evaluation methodologies.

Backbone Generalization. Our framework is LLM-agnostic by construction: any instructionfollowing model that can produce the specified

<table><tr><td>Backbone</td><td>Approach</td><td>SCA</td><td>CGC</td><td> $\bf { Q } \bf { A } \bf { C } _ { \bf { A p } }$ </td><td>chrF</td><td>COMET</td></tr><tr><td rowspan="3">Gemini 2.5 Pro</td><td>Sentence-Level</td><td>0.29</td><td>0.59</td><td>0.70</td><td>40.9</td><td>0.77</td></tr><tr><td>Context-Aware</td><td>0.48</td><td>0.68</td><td>0.72</td><td>39.4</td><td>0.76</td></tr><tr><td>Proposed</td><td>0.84</td><td>0.97</td><td>0.76</td><td>41.7</td><td>0.78</td></tr><tr><td rowspan="3">Qwen3.6-35B-A3B</td><td>Sentence-Level</td><td>0.12</td><td>0.73</td><td>0.73</td><td>40.1</td><td>0.73</td></tr><tr><td>Context-Aware</td><td>0.19</td><td>0.83</td><td>0.72</td><td>41.2</td><td>0.74</td></tr><tr><td>Proposed</td><td>0.61</td><td>0.86</td><td>0.75</td><td>40.3</td><td>0.74</td></tr></table>

Table 3: Discourse-level translation results for both LLM backbones; the Qwen block is discussed in §5.3. With Gemini 2.5 Pro, all differences between Proposed and both baselines are statistically significant (p<0.05, paired bootstrap, 10k samples) except $\mathrm { Q A C _ { A p } }$ vs Context-aware (p=0.066) and COMET vs Sentencelevel (p=0.345).

JSON schema can serve as the backbone. We repeat the Aesop’s Fables experiment with Qwen3.6-35B-A3B, an open-weight mixture-of-experts model (35B total, 3B active parameters), holding prompts and post-processing fixed. The ordering of the three conditions is preserved (Table 3): our framework again scores highest on the discourse metrics, improving SCA by +0.49 and CGC by +0.13 over the sentence-level baseline. Absolute scores are lower for the smaller model, most visibly on SCA. Traditional metrics again separate the conditions poorly, as under Qwen, the context-aware baseline attains the highest chrF despite the second-lowest SCA, reinforcing that our contribution lies in discourse coherence rather than surface-level quality. Qualitative Analysis. Table 4 illustrates these differences concretely. The sentence-level baseline produces $\mathrm { ^ { 6 6 } F O X ^ { \prime } }$ in S1 but “fs-W-E-A-S-E-L” in S2 for the same entity, and assigns inconsistent spatial indices (IX-3p:j in S1, IX-3p:i in S2). Our framework maintains both concept-gloss consistency (“fs-W-E-A-S-E-L” throughout) and spatial consistency (IX-3p:j for Weasel in both sentences).

## 5.4 Ablation Studies

Each module plays a role. Table 5 evaluates individual module contributions by selectively disabling each of SCM, QACM and CGCM. Each module primarily affects its corresponding metric: disabling SCM drops SCA from 0.81 to 0.71; disabling CGCM drops CGC from 0.97 to 0.64; with ${ \mathrm { Q A C } } _ { \mathrm { A p } }$ dropping from 0.76 to 0.71–0.72 when disabled. Notably, the presence of QACM is not as consequential on the translation metrics as its impact is mostly stylistic. The combination of all modules provides the most balanced performance across discourse-specific metrics, demonstrating that each module addresses distinct aspects of discourse coherence in ASL translation.

English: English:   
S1: A Bat fell to the ground and was caught by a Weasel,   
and was just going to be killed and eaten when it begged   
to be let go.   
S2: The Weasel said he couldn’t do that because he was   
an enemy of all birds on principle.   
Sentence-level Baseline:   
S1: BAT IX-3p:i FALL GROUND IX-loc:k. FOX IX-3p:j   
CATCH IX-3p:i. IX-3p:j ALMOST KILL EAT IX-3p:i. IX-  
3p:i BEG IX-3p:j. FOR-FOR? FREE.   
S2: FS-W-E-A-S-E-L IX-3p:i SAY CANNOT DO THAT   
WHY IX-3p:i ALWAYS OPPOSITE+AGENT ALL BIRD   
Proposed:   
S1: BAT IX-3p:i FALL GROUND FS-W-E-A-S-E-L   
IX-3p:j CATCH IX-3p:i ALMOST KILL EAT WHEN IX-  
3p:i BEG FREE   
S2: FS-W-E-A-S-E-L IX-3p:j SAY CANNOT FREE IX-  
3p:i WHY? IX-3p:j OPPOSITE+AGENT ALL BIRD RULE

##

Table 4: Qualitative examples. Filled boxes = concepts/glosses; unfilled boxes = spatial indices. Red = error; $\mathrm { G r e e n } = \mathrm { c o r r e c t } .$
<table><tr><td>SCM</td><td>CGCM</td><td>QACM</td><td>SCA</td><td>CGC</td><td> $\bf { Q } \bf { A } \bf { C } \bf { _ { A p } }$ </td><td>chrF</td><td>COMET</td></tr><tr><td>√</td><td>x</td><td>x</td><td>0.81</td><td>0.64</td><td>0.71</td><td>41.8</td><td>0.78</td></tr><tr><td>x</td><td>√</td><td>x</td><td>0.71</td><td>0.95</td><td>0.72</td><td>42.5</td><td>0.79</td></tr><tr><td>√</td><td>√</td><td>x</td><td>0.84</td><td>0.97</td><td>0.72</td><td>42.5</td><td>0.77</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.84</td><td>0.97</td><td>0.76</td><td>41.7</td><td>0.78</td></tr></table>

Table 5: Ablation study on each module’s contribution.

Context Window Size. Figure 3 shows the impact of context. Increasing context progressively improves all discourse metrics, while traditional metrics remain unchanged (chrF: ∼42, COMET: ∼0.78 across all conditions). SCA shows the steepest improvement $( 0 . 4 0  0 . 8 4 )$ , CGC follows a similar pattern (0.78 → 0.97), and $\mathrm { Q A C _ { A p } }$ plateaus at two sentences (0.76). Comparing against baselines without discourse-aware instructions, our framework outperforms the sentence-level baseline even without context (SCA: 0.45 vs 0.29), and with just one prior sentence exceeds the contextaware baseline with full access (SCA: 0.57 vs 0.48). This demonstrates that structured modeling provides greater benefit than context quantity alone.

## 5.5 Human Evaluation

To validate our proposed metrics, two ASL-fluent evaluators rated 32 stories (152 sentences) selected via stratified sampling across automated score ranges, drawn from both the proposed framework and sentence-level baseline. Sentences were rated on a 1–5 Likert scale for SCA, CGC, and ${ \mathrm { Q A C } } _ { \mathrm { A p } }$

SCA shows the strongest validation, with significant story-level correlation for both evaluators individually $( \rho = 0 . 5 0 , p = 0 . 0 0 5$ and $\rho = 0 . 4 2 , p = 0 . 0 2 2 ;$ combined $\rho = 0 . 5 2 , p = 0 . 0 0 3 )$ and moderate interrater agreement (weighted $\kappa = 0 . 4 3 .$ $\rho { = } 0 . 4 9 )$ . CGC shows significant correlation $( \rho = 0 . 4 8 , p = 0 . 0 0 7 )$ with both evaluators individually trending in the same direction $( \rho = 0 . 4 8 , \ p = 0 . 0 0 7$ and $\rho { = } 0 . 2 4$ $\textstyle p = 0 . 1 9 5 )$ . ${ \mathrm { Q A C } } _ { \mathrm { A p } }$ correlations are non-significant. However, QAC usage agreement between the system’s self-reported output and human judgments of QAC presence is strong for one evaluator $\textstyle ( \rho = 0 . 8 5 ,$ $p < 0 . 0 0 1 )$ but moderate for the other $( \rho = 0 . 3 9 $ $p { = } 0 . 0 3 5 )$ . This suggests the non-significant appropriateness correlation reflects disagreement among raters about what constitutes a QAC rather than a failure of the automated metric (see Appendix B for details).

![](images/83efadbc1228fe8402111df412870d008e954a4e62f8a89b75d4ee363ceda135.jpg)  
Figure 3: Impact of context window size on discourse metrics. Unfilled markers show SCA for baselines without discourse-aware instructions.

## 6 Conclusion

This work establishes discourse-aware text to sign language gloss translation as a novel task and presents the first comprehensive framework to address it. We formalize three discourse phenomena: spatial coreference, question-answer clauses, and concept-gloss consistency, and develop specialized modules (SCM, QACM, CGCM) with corresponding evaluation metrics (SCA, $\mathrm { Q A C _ { A p } , }$ CGC). Our experiments demonstrate that explicit discourse modeling substantially outperforms both sentencelevel and context-only approaches, establishing a foundation for more linguistically authentic sign language translation systems.

## 7 Limitations

Our framework currently relies on large language models to implement each module, which introduces dependencies on LLM instruction-following capabilities and may limit reproducibility as model versions change. However, the modular framework we propose is agnostic to the specific implementation of each component, as our results with an openweight backbone show (§5.3). SCM, QACM, and CGCM could alternatively be instantiated through rule-based systems, trained neural models, or human experts, depending on application requirements. Our evaluation focuses exclusively on English-to-ASL gloss translation. However, the linguistic phenomena we consider are not specific to ASL but are present across signed languages. Additionally, the Aesop’s Fables dataset lacks ground truth ASL annotations, requiring us to rely on automatic metrics and back-translation for discourselevel evaluation; a reference-annotated discourselevel dataset would allow standard reference-based evaluation alongside our metrics, and we regard building one as a substantial annotation contribution in its own right. The QACM also enforces QAC decisions through deterministic trigger rules, which cannot represent the optionality of QAC usage in contexts where both a QAC and a non-QAC rendering are natural; we discuss learned alternatives in Appendix B.

Finally, our framework addresses manual sign production through glosses but does not incorporate non-manual markers (NMMs) such as facial expressions and prosody, which carry critical grammatical and affective information in sign languages. We note, however, that the state each module maintains is closely related to the information a downstream visual generation system would need in order to produce NMMs: the QACM identifies question and answer constituents, the SCM assigns spatial indices to referents, and the CGCM tracks which referents are already established. Deriving NMM annotations from this registry state is a natural next step for connecting discourse-aware gloss translation to sign production.

## 8 Ethical Considerations

Sign language processing requires careful consideration of the perspectives and needs of those who use sign languages daily. We recognize that developing such systems without meaningful engagement from the Deaf and Hard-of-Hearing communities risks producing technology that does not serve their interests. Throughout the development of this work, we have collaborated with members of the signing community to ensure our approach remains grounded in authentic linguistic practices.

We use three datasets: ASL STEM Wiki (Yin et al., 2024), a licensed dataset, and Aesop’s Fables (Aesop, 1912). These datasets consist of educational, scientific, and traditional narrative content, and do not contain personally identifiable information or offensive material.

## References

Aesop. 1912. Aesop’s Fables. Project Gutenberg. EBook #11339. Available at https://www. gutenberg.org/ebooks/11339.

Katherine Atwell, Danielle Bragg, and Malihe Alikhani. 2024. Studying and mitigating biases in sign language understanding models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 268–283. Association for Computational Linguistics.

Vasileios Baltatzis, Rolandos Alexandros Potamias, Evangelos Ververas, Guanxiong Sun, Jiankang Deng, and Stefanos Zafeiriou. 2024. Neural sign actors: A diffusion model for 3d sign language production from text. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1985–1995.

Luana Bulla, Gabriele Tuccio, Misael Mongiovì, and Aldo Gangemi. 2025. Leveraging large language models for accurate sign language translation in lowresource scenarios. In European Conference on Artificial Intelligence (ECAI 2025), volume 413 of Frontiers in Artificial Intelligence and Applications, pages 2089–2096. IOS Press.

Ivano Caponigro and Kathryn Davidson. 2011. Ask, and tell as well: Question-answer clauses in ASL. Natural Language Semantics, 19(4):323–371.

Sara Court and Micha Elsner. 2024. Shortcomings of LLMs for low-resource translation: Retrieval and understanding are both the problem. In Proceedings of the Ninth Conference on Machine Translation, pages 1332–1354, Miami, Florida, USA. Association for Computational Linguistics.

Travis Dougherty. 2025. Development stages and classification of asl avatars and recognition models.

Florian Eichin, Yang Janet Liu, Barbara Plank, and Michael A. Hedderich. 2025. Probing LLMs for multilingual discourse generalization through a unified label set. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18665–18684. Association for Computational Linguistics.

Pooya Fayyazsanavi, Antonios Anastasopoulos, and Jana Kosecka. 2024. Gloss2Text: Sign language gloss translation using LLMs and semantically aware label smoothing. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 16162–16171, Miami, Florida, USA. Association for Computational Linguistics.

Anne Therese Frederiksen. 2019. Referential Cohesion in American Sign Language: Modality-Specific and Modality-General Influences. Ph.D. thesis, University of California, San Diego.

Jia Gong, Lin Geng Foo, Yixuan He, Hossein Rahmani, and Jun Liu. 2024. Llms are good sign language translators. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18362–18372.

Michael A. K. Halliday and Ruqaiya Hasan. 1976. Cohesion in English. Longman, London.

Eui Jun Hwang, Sukmin Cho, Junmyeong Lee, and Jong C. Park. 2025. An efficient gloss-free sign language translation using spatial configurations and motion dynamics with LLMs. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3901–3920, Albuquerque, New Mexico. Association for Computational Linguistics.

Saki Imai, Mert Inan, Anthony B. Sicilia, and Malihe Alikhani. 2025. SiLVERScore: Semantically-aware embeddings for sign language generation evaluation. In Proceedings ofthe 15th International Conference on Recent Advances in Natural Language Processing - Natural Language Processing in the Generative AI Era, pages 452–461, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria.

Mert Inan, Katherine Atwell, Anthony Sicilia, Lorna Quandt, and Malihe Alikhani. 2024. Generating signed language instructions in large-scale dialogue systems. In Proceedings ofthe 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 140–154. Association for Computational Linguistics.

Mert Inan, Anthony Sicilia, and Malihe Alikhani. 2025. SignAlignLM: Integrating multimodal sign language processing into large language models. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 3691–3706, Vienna, Austria. Association for Computational Linguistics.

Mert Inan, Yang Zhong, Sabit Hassan, Lorna Quandt, and Malihe Alikhani. 2022. Modeling intensification for sign language generation: A computational approach. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2897–2911. Association for Computational Linguistics.

Colin Lea, Vasileios Baltatzis, Connor Gillis, Raja Kushalnagar, Lorna Quandt, and Leah Findlater. 2026. Bootstrapping sign language annotations with sign language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 3630– 3640.

Huije Lee, Jung-Ho Kim, Eui Jun Hwang, Jaewoo Kim, and Jong C. Park. 2023. Leveraging large language

models with vocabulary sharing for sign language translation. 2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), page 1–5.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Camillo Lugaresi and Barbara Di Eugenio. 2013. Translating italian connectives into italian sign language. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 270–280.

Mathias Müller, Zifan Jiang, Amit Moryossef, Annette Rios Gonzales, and Sarah Ebling. 2023. Considerations for meaningful sign language machine translation based on glosses. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 682–693.

Carol Neidle, Augustine Opoku, and Dimitris Metaxas. 2022. Asl video corpora & sign bank: Resources available through the american sign language linguistic research project (asllrp). arXiv preprint arXiv:2201.07899.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting ofthe Associationfor Computational Linguistics, pages 311–318.

Maja Popovic. 2015. chrf: character n-gram f-score for ´ automatic mt evaluation. In Proceedings ofthe tenth workshop on statistical machine translation, pages 392–395.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685–2702, Online. Association for Computational Linguistics.

Daniel Stein, Philippe Dreuw, Hermann Ney, Sara Morrissey, and Andy Way. 2007. Hand in hand: automatic sign language to english translation. In Proceedings of the 11th Conference on Theoretical and Methodological Issues in Machine Translation ofNatural Languages: Papers.

Carla Viegas, Mert Inan, Lorna Quandt, and Malihe Alikhani. 2023. Including facial expressions in contextual embeddings for sign language generation. In Proceedings of the 12th Joint Conference on Lexical and Computational Semantics (\*SEM 2023), pages 1–10. Association for Computational Linguistics.

Ronnie B. Wilbur. 1994. Foregrounding structures in American Sign Language. Journal of Pragmatics, 22:647–672.

Elizabeth A. Winston. 1991. Spatial referencing and cohesion in an American Sign Language text. Sign Language Studies, 73:397–410.

Ryan Wong, Necati Cihan Camgoz, and Richard Bowden. 2024. Sign2gpt: Leveraging large language models for gloss-free sign language translation. In International Conference on Learning Representations, volume 2024, pages 18157–18174.

Kayo Yin, Kenneth DeHaan, and Malihe Alikhani. 2021a. Signed coreference resolution. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 4950–4961, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kayo Yin, Amit Moryossef, Julie Hochgesang, Yoav Goldberg, and Malihe Alikhani. 2021b. Including signed languages in natural language processing. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7347– 7360, Online. Association for Computational Linguistics.

Kayo Yin, Chinmay Singh, Fyodor O. Minakov, Vanessa Milan, Hal Daumé III, Cyril Zhang, Alex Xijie Lu, and Danielle Bragg. 2024. ASL STEM Wiki: Dataset and benchmark for interpreting STEM articles. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 14474–14490, Miami, Florida, USA. Association for Computational Linguistics.

Han Zhang, Rotem Shalev-Arkushin, Vasileios Baltatzis, Connor Gillis, Gierad Laput, Raja Kushalnagar, Lorna C Quandt, Leah Findlater, Abdelkareem Bedri, and Colin Lea. 2025. Towards ai-driven sign language generation with non-manual markers. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems, CHI ’25, New York, NY, USA. Association for Computing Machinery.

Tianyang Zhong, Zhenyuan Yang, Zhengliang Liu, Ruidong Zhang, Yiheng Liu, Haiyang Sun, Yi Pan, Yiwei Li, Yifan Zhou, Hanqi Jiang, Junhao Chen, and Tianming Liu. 2025. Opportunities and challenges of large language models for low-resource languages in humanities research. Preprint, arXiv:2412.04497.

## A Translation Prompts

This appendix presents the LLM prompts used in each experimental condition. Unless stated otherwise, all experiments use Gemini 2.5 Pro as the underlying model; the backbone comparison in §5.3 uses Qwen3.6-35B-A3B with these same prompts. All three conditions require the model to return a JSON object containing: (1) the ASL gloss translation, (2) spatial index–entity mappings with corresponding English words, (3) directional verb source/target mappings, (4) whether a rhetorical question (QAC) was used, and (5) concept-to-gloss mappings for nouns. The output format specification is shared across conditions and is listed separately in Section A.4.

A vocabulary constraint is enforced in all conditions: every gloss must come from a provided vocabulary list, with fingerspelling (fs-WORD) as the only fallback. For all conditions, sentences within a story are processed sequentially (causally). The system and user message portions of the prompt are concatenated into a single prompt string.

## A.1 Sentence-Level Baseline

The sentence-level baseline translates each sentence independently with no prior context and minimal instructions.

You are an ASL translator. Translate English sentences to ASL gloss format.

VOCABULARY CONSTRAINT: Use only glosses from this vocabulary: {vocabulary}

ASL GRAMMAR BASICS:

– Use spatial indices for entities (IX-3p:i, IX-3p:j, IX-loc:k)

– Use directional verbs when appropriate (i:GIVE:j, 1p-MEET:i)

– You may use rhetorical questions (QAC) if the sentence structure suggests it

Translate: “{current\_sentence}”

## A.2 Context-Aware Baseline

The context-aware baseline uses the same minimal instructions as the sentence-only baseline but adds a listing of all previously translated sentences and their ASL translations, without explicit guidance on how to use this context.

You are an ASL translator. Translate English sentences to ASL gloss format using discourse context. VOCABULARY CONSTRAINT: Use only glosses from this vocabulary: {vocabulary}

ASL GRAMMAR BASICS:

– Use spatial indices for entities (IX-3p:i, IX-3p:j, IX-loc:k)

– Use directional verbs when appropriate (i:GIVE:j, 1p-MEET:i)

– You may use rhetorical questions (QAC) if appro  
priate   
PREVIOUS CONTEXT:   
Sentence 1: “{sentence<sub>1</sub>}” → {translation<sub>1</sub>}   
Sentence 2: “{sentence<sub>2</sub>}” → {translation<sub>2</sub>}   
CURRENT SENTENCE: “{current\_sentence}”

## A.3 Proposed Approach

The proposed approach uses a detailed prompt with explicit discourse-level translation principles and module-specific state registries. It is composed of two parts: a system message with translation guidelines, and a per-sentence user message that includes (1) a configurable context window of w previous sentence–translation pairs, and (2) structured registries corresponding to the three modules described in §3.2. The registries are accumulated from the LLM’s own structured output across previous sentences and passed back as explicit constraints. Below we present the combined prompt with redundancies between the two parts removed for brevity.

You are an expert ASL (American Sign Language) translator specializing in discourse-level translation. You process sentences sequentially, maintaining consistency with previous translations while having no knowledge of future sentences.

CRITICAL VOCABULARY CONSTRAINT: Every gloss in your output must exist in the provided vocabulary. This is a hard requirement—do not use any glosses outside the vocabulary list. If no suitable vocabulary gloss exists for a concept, use fingerspelling with the format (fs-WORD) as the only fallback.

Available vocabulary: {vocabulary}

CORE ASL GRAMMAR PRINCIPLES:

1. Use Topic-Comment structure when appropriate

2. Omit copula verbs (“is”, “are”, “was”, “were”)

3. Use ASL word order (often SOV or Topic-Comment)

4. Maintain spatial consistency for referents throughout discourse

5. Use appropriate non-manual markers and classifiers

SPATIAL INDEXING SYSTEM:

– First person: IX-1p (I/me), POSS-1p (my/mine)

– Second person: IX-2p (you), POSS-2p (your), IX-2p-pl-arc (you plural)

– Third person: IX-3p:i, IX-3p:j, IX-3p:k (different spatial locations)

– Locations: IX-loc:i, IX-loc:j, IX-loc:k (specific places)

– Directional verbs: 1p-GIVE:i, i:GIVE:j,

i:MEET:j, etc.

For each spatial index, specify which entity it represents and which English word/phrase in the current sentence it corresponds to. If carried over from previous context, use “implicit”.

RHETORICAL QUESTIONS (QAC) USAGE:

Use rhetorical questions for:

1. Cause-Effect: Replace “because”, “so that”, “in order to”

Example: “I study because I want good grades” → “IX-1p STUDY WHY? GRADE GOOD WANT”

2. Topic Introduction: Introduce new information or shift focus

3. Emphasis: Highlight important information

Do not use rhetorical questions at discourse beginning, when directly answering a genuine question, or for simple statements that don’t need emphasis.

CONCEPT-GLOSS MAPPING: For each sentence, identify all nouns in the English sentence and specify which ASL gloss you used to translate each noun. Only include nouns that appear in your translation.

PREVIOUS DISCOURSE CONTEXT:

Sentence 1: “{sentence<sub>1</sub>}”

ASL Translation 1: {translation<sub>1</sub>}

Sentence 2: “{sentence<sub>2</sub>}”

ASL Translation 2: {translation<sub>2</sub>}

SCM — SPATIAL INDEX REGISTRY S (established assignments):

Spatial Consistency constraint: You MUST use the same spatial index for the same entity.

Coreference Correspondence constraint: Each spatial reference must correspond to an entity mention in the English source.

Established entity-index assignments:

– IX-3p:i → fox

– IX-3p:j → rabbit

– IX-loc:k → forest

Assign new spatial indices only for entities not yet in this registry.

Established directional verb patterns:

– i:GIVE:j: source=fox, target=rabbit

Both source and target indices in directional verbs must satisfy the spatial consistency constraint.

QACM — QAC DECISION REGISTRY Q (prior QAC usage):

Discourse Position constraint: QACs should NOT appear at discourse beginning or as responses to genuine questions.

Contextual Appropriateness constraint: Use QACs only when the information structure warrants topicalization (causal explanations, contrastive elements, emphasis).

Prior QAC decisions:

– Sentence 1: no QAC

– Sentence 2: QAC used

CGCM — CONCEPT-GLOSS REGISTRY L (established mappings):

Mapping Consistency constraint: You MUST use the established gloss for any concept already in this registry.

Vocabulary Compliance constraint: All glosses must come from the provided vocabulary.

Established concept-gloss mappings:

– “fox” → FOX (established in sentence 1)

– “rabbit” → RABBIT (established in sentence 1)

– “forest” → fs-F-O-R-E-S-T (established in sentence 1)

For NEW concepts not in this registry, select an appropriate gloss from the vocabulary. It will be registered for future sentences.

CURRENT SENTENCE TO TRANSLATE: “{cur-rent\_sentence}”

The three registries are accumulated from the LLM’s own structured output across previous sentences within the context window: spatial\_mappings and directional\_verbs populate S, qac\_used populates Q, and concept\_mappings populates L. For concept mappings, the first occurrence of each concept within the window establishes the canonical gloss; for spatial indices, the most recent assignment per index is used. This ensures that each module’s constraints (§3.2–3.2) are implemented as explicit prompt-level constraints rather than relying on the

LLM to infer consistency from raw translation context.

The context window parameter w controls how many previous sentence–translation pairs are included, and the module registries are bounded by the same window. When w = 0, no context or registry state is provided; when w = n, the registries reflect the n most recent sentences; when w is unset, all previous sentences contribute to both context and registries.

Post-Processing Verification In addition to the prompt-level registry constraints, a programmatic verification step runs after each LLM call to enforce module constraints as hard constraints. The verification checks each module’s output against the accumulated registries and applies corrections where violations are detected:

• SCM: If an entity’s spatial index differs from its registry assignment, the index is corrected via string replacement throughout the translation, and all affected spatial mappings and directional verbs are updated accordingly.

• CGCM: If a concept’s gloss differs from the registry’s established mapping, the gloss is replaced throughout the translation and the concept mappings are updated.

• QACM: The discourse position constraint (t=0) is enforced via pre-processing: the prompt for the first sentence explicitly prohibits QAC usage. As a fallback, if the LLM still reports a QAC at t=0, the metadata is corrected.

All corrections are applied deterministically without re-querying the LLM. The SCM and CGCM operate via post-processing string replacement; the QACM operates via pre-processing prompt construction based on sentence position.

## A.4 Output Format

All three conditions include the following output format specification in their prompts.

Return a JSON with the following fields:

– “translation”: the ASL gloss translation

– “spatial\_mappings”: a mapping from each spatial index to its entity name and corresponding English word, e.g., {“IX-3p:i”: {“entity”: “fox”, “english\_word”: “he”}}

– “directional\_verbs”: a mapping from each directional verb to its source and target entities with corresponding English words, e.g., {“i:GIVE:j”: {“source”: “fox”, “target”: “rabbit”, “source\_word”: “he”, “target\_word”: “her”}}

– “qac\_used”: boolean indicating whether a rhetorical question structure was used

– “concept\_mappings”: a mapping from English nouns to their ASL glosses, e.g., {“vehicle”: “CAR”}

## A.5 Back-Translation Prompt

Back-translation is used to produce English reconstructions from ASL gloss output for computing chrF and COMET scores. Sentences are translated causally: each sentence is back-translated with access to all previous ASL sentences in the story but no future sentences, mirroring the forward translation setup.

You are an expert ASL interpreter with deep knowledge of ASL discourse structure, spatial referencing, and narrative coherence. You are translating ASL gloss sequences back to natural English while maintaining discourse consistency.

TASK: Translate the TARGET ASL sentence back to natural English, using the previous story context to ensure proper discourse coherence, pronoun resolution, and narrative flow.

Previous ASL sentences in this story:

1. {translation<sub>1</sub>}

2. {translation<sub>2</sub>}

Now translate the next sentence:

TARGET ASL SENTENCE TO TRANSLATE:

{sentence\_index}. {target\_asl}

TRANSLATION GUIDELINES:

1. Use the previous story context to resolve spatial references (IX-3p:i, IX-3p:j, etc.) to appropriate English pronouns or noun phrases

2. Maintain narrative coherence with previous sentences

3. Translate to natural, fluent English that continues the story flow

4. Preserve the meaning and discourse function of the ASL sentence

5. Use appropriate English discourse markers and connectives when needed

6. Ensure pronoun antecedents are clear from the established context

7. Maintain consistency with spatial reference assignments from previous sentences

Return ONLY a JSON: {“translation”:   
“your\_english\_translation”}

## A.6 Coreference Chain Extraction Prompt

English coreference chains are extracted from source stories to serve as the ground-truth entity structure for computing SCA. Unlike the causal translation and back-translation prompts, this prompt receives the entire story at once, as coreference resolution requires full document context.

You are an expert in coreference resolution. Analyze the text for coreference chains and provide detailed output.

Pay special attention to:

– First-person pronouns (I, me, my, mine) and their referents

– Narrator identity in first-person narratives

– All pronouns and their antecedents

– Ambiguous pronouns like “it” that might refer to different entities

Sentences:

1. {sentence<sub>1</sub>}

2. {sentence<sub>2</sub>}

. . .

Return your analysis as JSON:

{“chains”: [{“chain\_id”: 1, “canonical\_entity”: “first\_mention”,

“mentions”: [“mention1”, “mention2”, . . . ],

“sentence\_locations”: [1, 2, . . . ]}]}

## B Human Evaluation Details

Study Design We selected 32 stories (152 sentences) from Aesop’s Fables via stratified sampling across automated SCA, CGC, and QAC score ranges (low: <0.33, medium: 0.33–0.67, high: ≥0.67), drawing equally from the proposed framework and sentence-level baseline outputs. Two ASL-fluent evaluators independently rated each sentence on a 1–5 Likert scale for spatial coreference accuracy (SCA), concept-gloss consistency (CGC), and QAC appropriateness $\mathrm { ( Q A C _ { A p } ) }$ , with an N/A option when QAC was not applicable.

Inter-Rater Agreement For SCA, inter-rater agreement is moderate: weighted $\kappa { = } 0 . 4 3$ (quadratic), Spearman $\rho { = } 0 . 4 9 \ ( p { < } 0 . 0 0 1 )$ , with 64% within-one-point agreement across 144 paired ratings. For CGC and QAC, both evaluators assigned near-ceiling ratings (CGC means: 4.72 and 4.96; QAC means: 4.58 and 5.00), resulting in insufficient variance for meaningful agreement statistics.

Correlation Analysis We report story-level Spearman correlations (mean aggregation across sentences), as sentences within a story are not independent. Table 6 presents per-evaluator and combined results.

<table><tr><td>Metric</td><td>Evaluator</td><td>ρ</td><td>p</td></tr><tr><td>SCA</td><td>Evaluator 1 Evaluator 2 Combined</td><td>0.50 0.42 0.52</td><td>0.005 0.022 0.003</td></tr><tr><td>CGC</td><td>Evaluator 1 Evaluator 2 Combined</td><td>0.48 0.24 0.48</td><td>0.007 0.195 0.007</td></tr><tr><td> $\mathrm { Q A C _ { A p } }$ </td><td>Evaluator 1 Evaluator 2 Combined</td><td>-0.12 0.28 -0.09</td><td>0.561 0.301 0.659</td></tr></table>

Table 6: Story-level Spearman correlations $( \rho )$ between automated metrics and human ratings. Combined averages both evaluators’ ratings per sentence before correlating.

QAC Subjectivity $\mathrm { Q A C _ { A p } }$ correlations are nonsignificant for both evaluators. We attribute this primarily to the inherently subjective nature of QAC appropriateness judgments. Our automated metric evaluates QAC usage against specific discourse criteria: the presence of causal relationships, topicalization, or emphasis markers in the English source. However, native ASL signers may find QAC restructuring acceptable even in the absence of such markers. For example, the English sentence “There are two sides to every question” can be rendered as either EACH QUESTION HAVE TWO SIDE or EACH QUESTIONHAVE WHAT? TWO SIDE. The latter employs a QAC for topicalization that is natural in ASL discourse but has no explicit discourse trigger in the English source. This gap between the metric’s rule-based criteria and the broader range of acceptable human judgments makes strong automated-human agreement challenging for this phenomenon. The evaluators noted that some QAC usages were acceptable even when not matching their personal signing preferences. We further examined agreement on QAC usage (binary presence/absence, as opposed to appropriateness). At the story level, one evaluator shows strong agreement with the system’s self-reported qac\_used field $( \rho { = } 0 . 8 5 , p { < } 0 . 0 0 1 )$ , while the other shows moderate agreement $( \rho = 0 . 3 9 , p = 0 . 0 3 5 )$ . Inter-rater agreement on QAC presence is similarly moderate $( \rho = 0 . 4 1 , \ p = 0 . 0 0 1 )$ . The primary source of disagreement is that one evaluator rated 28 sentences as not containing a QAC (N/A) where both the system and the other evaluator identified one, suggesting differing thresholds for what constitutes a QAC structure.

One direction for future work is to replace the QACM’s deterministic trigger rules with a learned trigger model, trained on annotated data where QAC structures are identifiable from gloss patterns (a wh-constituent followed by an answer clause). Emitting a confidence score rather than a binary decision would allow restructuring to be applied only in high-confidence contexts, with mid-range cases treated as stylistic variants. A further extension would condition on signer preference and treat QAC frequency as a register parameter, analogous to formality in text generation.

CGC Ceiling Effects CGC shows significant combined correlation $( \rho = 0 . 4 8 , p = 0 . 0 0 7 )$ , driven primarily by Evaluator 1 $( \rho = 0 . 4 8 , p = 0 . 0 0 7 )$ . Evaluator 2 shows a positive but non-significant trend $( \rho = 0 . 2 4 , \ p = 0 . 1 9 5 )$ , likely attenuated by nearceiling ratings (95% rated 5). Both evaluators assigned high CGC ratings overall (94–95% rated 4 or 5), which limits discriminative power. We note that CGC evaluates cross-sentence gloss consistency (whether the same concept receives the same gloss throughout the discourse), which is a subtler judgment than per-sentence translation accuracy and may be difficult to assess reliably without explicit side-by-side comparison tools.

The ceiling effect appears on the human-rating side rather than in the metric itself. CGC remains discriminative on the translation side: it is 0.59 for the sentence-level baseline (Table 3) and moves from 0.64 to 0.97 as the CGCM is enabled (Table 5). Validating CGC more sharply would require sampling translations with a wider spread of consistency violations than the systems we evaluate produce.