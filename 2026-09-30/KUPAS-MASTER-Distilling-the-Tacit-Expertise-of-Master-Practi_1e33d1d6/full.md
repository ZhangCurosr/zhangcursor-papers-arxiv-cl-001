# KUPAS MASTER: Distilling the Tacit Expertise of Master Practitioners into Agent-Ready Experience Corpora

KUPAS MASTER Team<sup>1</sup>

Shanghai Kupas Technology Co., Ltd. and Tongji University Technical Report<sup>2</sup>, September 2026

## Abstract

In every field, experienced professionals know more than just facts and conclusions. They know which cues matter, why a judgment is reasonable, which action to take, and when a familiar approach no longer applies. Routine work records often leave out this tacit knowledge, making it dificult for large language model (LLM) agents to use professional experience efectively. We introduce KUPAS MASTER, an experience engineering platform built around nine-layer cognitive corpus construction. It turns heterogeneous work records and practitioner interviews into traceable, reusable experience corpora for agents. Six case elements preserve the task process: context, cues, judgment, action, boundaries, and outcomes. Nine-layer cognitive corpus construction organizes tacit experience along nine extraction dimensions and stores the resulting assets in six libraries: rules, constraints, best practices, negative examples, corner cases, and skills. Semantic alignment, individual experience distillation, organizational consolidation, and crossreview preserve source evidence, conditions of use, and unresolved disagreements. The platform packages these assets into callable skills with explicit inputs, steps, dependencies, and stopping conditions, connecting experience collection to task execution and evaluation feedback. Using authorized samples from 20 randomly selected practitioners, the platform processed 1,576 source files into 23,024 individual experience records and 13,113 organizational assets. The evaluation spans 177 questions and 531 responses across multiple professional domains. Under common task inputs and scoring criteria, the base model, raw-corpus retrieval-augmented generation (RAG), and KUPAS MASTER agent scored 70.63, 79.75, and 89.58, respectively. The KUPAS MASTER agent improved on raw-corpus RAG by 9.83 points, with gains in all seven scoring dimensions. The systematic comparison demonstrates the efectiveness of the platform and its core nine-layer method in professional tasks, delivering better task quality, deeper professional judgment, and efective experience reuse. The platform provides a practical path from individual tacit experience to organizational knowledge and agent capabilities.

Keywords: expert experience; tacit knowledge; experience corpora; knowledge acquisition; agent skills; organizational knowledge; AI for engineering

## 1 Introduction

As agents enter enterprise and scientific workflows, tasks that depend on professional experience require them to understand context, make sound judgments, and choose appropriate actions. Much of the experience they need is spread across practitioners and work records. Turning it into traceable, reusable corpora strengthens task execution and preserves critical know-how. Major national initiatives reflect the international importance of this need. China’s AI Plus initiative calls for reusable expert knowledge and high-quality datasets [1]. In the United States, the Genesis Mission calls for an integrated AI platform that uses federal scientific datasets to develop scientific foundation models and agents for hypothesis testing and research automation [2]. Together, these initiatives highlight a shared priority: making domain knowledge usable by AI systems. By turning professional experience into reusable assets for agents, KUPAS MASTER addresses a problem of global significance for industrial productivity and scientific innovation.

Organizing professional experience for AI systems is therefore an important problem in enterprise knowledge management. A maintenance expert may combine an unusual sound, recent repair records, and operating load to diagnose a fault. A mediator may first verify disputed facts before discussing responsibility. In both tasks, useful experience includes the evidence behind a decision, the alternatives considered, the conditions for an action, and the reasons to stop or change course. Without those conditions, another practitioner or agent may apply a useful rule to the wrong case.

Organizations already keep manuals, work orders, incident reports, recordings, and case reviews. Yet terminology varies, intermediate decisions go unrecorded, and observations made at the time may be mixed with later explanations. A record of a successful intervention may omit the conditions that made it work. Rare but serious failures may survive only in conversation and remain unavailable to text retrieval. Knowledge ac quisition research has long studied how to recover such information from actual work. The Critical Decision Method, for example, uses structured questions about specific incidents to identify the cues and reasoning that experienced practitioners relied on [3–5].

Large language models (LLMs) ofer new ways to organize and use these materials. Retrieval-augmented generation (RAG) brings external text into inference [6], graph retrieval uses relationships across records [7], and agent frameworks connect reasoning with tool use [8, 9]. These methods help retrieve and use existing information. Several questions remain about the experience itself: how to fill gaps, express conditions of use, handle conflicting accounts, and check whether an extracted procedure can actually be followed. Answering them requires attention to how the underlying experience was collected, organized, and reviewed.

KUPAS MASTER calls this process experience engineering: acquiring, structuring, validating, and maintaining experience so that it becomes reusable, reviewable knowledge. Here, an experienced practitioner is someone who can contribute practical knowledge about a specific task and context. The platform links that knowledge to its task, conditions, and supporting evidence so that others can understand and review the judgment before reusing it. It also records additions and revisions through version control, supporting expert review, source tracing, and later agent use.

For this report, we randomly selected 20 authorized platform users from diferent professional domains and examined their corpora, assets, and run records. The sample contains 1,576 source files, 30,762 corpus chunks, 23,024 individual experience records, and 13,113 organizational assets. These counts correspond to source collection, parsing, experience extraction, and organizational consolidation.

We compare three agent configurations on the same tasks under a shared scoring rubric: a base-model agent (A), an agent using raw-corpus RAG (B), and an agent using KUPAS MASTER experience assets and skills (C). The evaluation contains 177 questions and 531 responses. We average the platform-reported practitioner scores equally across the 20 practitioners; each score combines seven weighted dimensions. The scores are 70.63, 79.75, and 89.58 for A, B, and C. The results demonstrate the efectiveness of nine-layer cognitive corpus construction, with consistent improvements in task quality, professional judgment, and execution. Later sections explain the evaluation and the role of experience assets in these gains.

The platform makes four main contributions. A structured representation of expert experience connects six case elements, nine extraction dimensions, and six asset types in a common format for collection, review, tracing, and execution. An experience corpus construction workflow converts heterogeneous work records into candidate assets while retaining their evidence, context, and review status. Organizational consolidation compares and combines experience under explicit task conditions. It keeps context-dependent alternatives separate and flags conflicts, disagreements, and insuficient evidence for review. Finally, skill construction and evaluation turns these assets into callable procedures and compares their use against the base model and raw-corpus RAG.

![](images/cd5d9de6739f4fce72bc1276c262d4ae821a2c9ebedc0a95475f9f3893bf3362.jpg)  
Figure 1 System overview of KUPAS MASTER and its architecture for using professional experience.

The following sections introduce task scope, experience representation, corpus construction, and organizational consolidation, with examples from the authorized sample. Section 8 describes the evaluation design, including controls for model configuration, input information, and runtime resources.

## 2 System Overview and Task Scope

## 2.1 Using the system

The KUPAS MASTER workflow starts with a clearly scoped task. The task description identifies the practitioner’s role, the object of work, the environment, available tools, triggers, and the extent of a single case record. It also defines completion criteria and the decisions the agent is allowed to make. A broad label such as “industrial maintenance” is too vague to guide experience collection. A more useful definition names a component and its operating conditions, then limits the task to gathering information, analyzing the problem, and recommending repairs within the authorized scope.

Once the scope is clear, the practitioner and collection assistant prepare a task description and evidence collection plan. They record key observations, judgments, decisions, exceptions, and outcomes. Missing information prompts targeted follow-up. If a diagnosis lacks an important measurement, for example, the next step is to locate the measurement record or clarify it in an interview. This preserves a clear link between facts, judgments, and outcomes.

One practitioner in our sample mediates workplace injury disputes. Their tasks include assessing the nature and causes of an injury, calculating compensation items, negotiating disputes, and reviewing agreements. Relevant evidence includes responsibility findings, medical records, attendance records, and proof of third-party payments. The task specification also defines the mediator’s authority: mediation and advice do not replace judicial decisions. Matters outside that authority, or disputes that remain unresolved, follow the appropriate referral procedure. These requirements give experience collection a concrete focus on evidence, reasoning, steps, and conditions of use.

Figure 1 shows the platform architecture. Its seven-step workflow is grouped in Figure 2. The platform defines the task, collects evidence, and aligns terms, entities, and events. It then extracts candidate assets from indi vidual cases, compares and consolidates experience across practitioners, and reviews evidence, applicability, and wording. We call this stage cross-review. The platform label “cross-validation” refers to this content review, rather than k-fold cross-validation. Reviewed assets then support agent evaluation. Evaluation findings feed back into collection and revision.

Two distinct records run through the workflow. A case record describes what happened during a particular task, including its context, observations, actions, and outcome. An asset record captures reusable experience drawn from one or more cases. Their relationship is many-to-many: one case may support several assets, and one asset may draw on several cases. Keeping these records separate preserves the original events while allowing interpretations and generalizations to be revised.

![](images/4a67d871296869160e04298911c49d371ad0c7a95a37461afe1c8e7958eb9d4e.jpg)  
Retained throughout: evidence, applicability, review status, and versions  
Figure 2 The seven-step KUPAS MASTER workflow, grouped into data collection, experience construction, and agent evaluation. Solid arrows show processing order. The dashed loop returns evaluation findings to corpus revision. Each stage retains source references, conditions of use, review status, and version identifiers.

## 2.2 Platform responsibilities

The platform acquires, structures, reviews, selects, and maintains experience. The language model handles task understanding and reasoning, while domain tools obtain external observations or perform authorized actions. Model services connect through a common interface to reduce the efect of model diferences on the surrounding workflow.

To make a run traceable and reviewable, its record should include the task specification, model and model ver sion, asset versions, retrieval configuration, and tool permissions. A model change should trigger evaluation on the same fixed tasks with other conditions held constant. Section 10 describes deployment and imple mentation configurations. The next sections explain how nine-layer cognitive corpus construction supports representation, corpus building, consolidation, and agent integration.

Table 1 Case elements and recording requirements. Facts, judgments, and execution states remain distinct, and missing fields are explicit.
<table><tr><td>Element</td><td>Content</td><td>Distinctions to retain</td></tr><tr><td>Context</td><td>Task, role, object, environment, time, and available resources.</td><td>Conditions observed in this case versus conditions assumed for reuse.</td></tr><tr><td>Cues</td><td>Signals, measurements, statements, and changes noticed during work.</td><td>Direct observations versus reported or inferred observations.</td></tr><tr><td>Judgment</td><td>Interpretations, supporting reasons, alternatives, and uncertainty.</td><td>Practitioner accounts versus model-generated explanations.</td></tr><tr><td>Action</td><td>Selected operations, order, parameters, and stopping conditions.</td><td>Planned, attempted, and completed actions.</td></tr><tr><td>Boundaries</td><td>Prohibitions, scope, missing prerequisites, and referral conditions.</td><td>Mandatory constraints versus preferences or usual practice.</td></tr><tr><td>Outcome</td><td>Immediate results, later verification, and unresolved effects.</td><td>Expected, reported, and independently verified outcomes.</td></tr></table>

## 3 Structured Representation of Expert Experience

KUPAS MASTER represents experience through case records, extraction dimensions, and asset types. These describe the task process, capture the basis of expert judgment, and organize reusable content, respectively (Figure 3). Their connections are many-to-many: a case can involve several dimensions, and a skill can draw on several asset types.

![](images/bb3eb811d98a73675d6e4c4ffac2db1d9ba1ac6da942e2fa823d744c8ad322e8.jpg)  
Figure 3 The nine extraction dimensions, grouped by function. Case elements record the task process, and reviewed statements can become six types of experience asset. Dimensions and asset types have a many-to-many relationship: a skill may combine attention, decision, habit, and constraint information.

## 3.1 Task cases

Following the case elements in Table 1, a task case is represented as

$$
\boldsymbol { e } = ( c , u , j , a , b , o ) ,\tag{1}
$$

Table 2 Extraction questions and representative objects in the nine-layer framework. Outputs become assets after evidence and applicability review.
<table><tr><td>ID</td><td>Dimension</td><td>Extraction question</td><td>Representative objects</td></tr><tr><td>L1</td><td>Knowledge</td><td>Which facts and concepts were used?</td><td>Entities, terms, relations, and domain materials.</td></tr><tr><td>L2</td><td>Attention</td><td>Which observations received priority?</td><td>Diagnostic cues, ignored distractions, and shifts of focus.</td></tr><tr><td>L3</td><td>Judgment</td><td>What supports this interpretation?</td><td>Conditional judgments, thresholds, alternatives, and uncertainty.</td></tr><tr><td>L4</td><td>Decision</td><td>How was an action selected?</td><td>Prerequisites, selection criteria, and stopping conditions.</td></tr><tr><td>L5</td><td>Association</td><td>Which other cases or concepts were relevant?</td><td>Analogies, recalled precedents, and links across cases.</td></tr><tr><td>L6</td><td>Anticipation</td><td>What consequences were expected?</td><td>Predicted effects, time horizons, and follow-up checks.</td></tr><tr><td>L7</td><td>Monitoring</td><td>When was another check needed?</td><td>Knowledge gaps, uncertainty, and reasons to revisit a judgment.</td></tr><tr><td>L8</td><td>Habits</td><td>Which procedures recur across cases?</td><td>Repeated action sequences, communication routines, and preferences.</td></tr><tr><td>L9</td><td>Constraints</td><td>What must not be done?</td><td>Prohibitions, scope limits, and referral conditions.</td></tr></table>

where c is context, u cues, j judgment, a action, b boundaries, and o outcome. These elements provide a common record while allowing diferent execution orders. Repeated judgments and actions can be recorded as timestamped or partially ordered events, retaining their order and dependencies. Unresolved cases keep their outcome unverified; missing reasons remain explicit and can be added later.

Several independent accounts of the same event can coexist. If practitioners disagree about an observation or fact, each account retains its source rather than being merged into a single certain statement. Earlier failed actions also remain in the record even when a later action solves the problem. They show how feedback changed the practitioner’s judgment or strategy. Keeping only the successful path would lose that part of the experience.

For workplace injury mediation, the context might be an injured construction worker requesting compensation mediation. Cues include responsibility findings, medical records, proof of employment, and insurance documents. Judgment concerns the relationship between work injury benefits and third-party payments. Actions include compensation calculations, explanations of policy and responsibility, negotiation, and agreement review. Boundaries specify that the mediator cannot replace a judicial decision and must refer unresolved dis putes to arbitration or litigation. An outcome may be an agreement or a record of unresolved issues and completed referral. The six fields let reviewers examine facts, professional judgments, actions, authority, and results separately.

## 3.2 The nine-layer framework and extraction dimensions

The nine-layer cognitive framework supplies a common vocabulary for extraction and annotation (Table 2). L1– L9 follow the platform’s established terminology and denote dimensions that work together and can overlap. Figure 3 groups them into evidence and context, judgment and action, and practice and monitoring. A single statement can span several dimensions or groups.

The dimensions guide questions such as what a practitioner noticed, which evidence supported a judgment, what action followed, and how feedback changed the next step. Together with the asset types, they form the complete experience configuration evaluated on professional tasks.

![](images/96e5e6832f0a563fd6c64487f2023dd5258d5b17ec6f32a9e4f8a644dcaad0c3.jpg)  
Figure 4 Shared dependencies and cross-layer links among the nine extraction pipelines. The arrow from L1 to the L2--L9 group denotes a shared anchor map for object identity and source tracing. Each pipeline also reads its own required source records. Other arrows denote evidence references, candidate additions, or review constraints supported by sources.

Attention extraction requires direct or indirect evidence of what the practitioner actually attended to, such as an explanation, inspection sequence, or activity log. A model’s attention weights alone do not establish a practitioner attention pattern [10]. Anticipation records pair a prediction with its horizon and later outcome, separating prior judgment from subsequent fact. Repeated behavior supports a habit description but does not make it mandatory. Monitoring records uncertainty about judgments, evidence suficiency, and applicability, preserving the limits on use.

## 3.3 Nine-layer cognitive corpus construction

This section explains how the nine dimensions turn factual accounts, reasoning, and action records into traceable experience units. The approach organizes linked extraction pipelines over narrative materials, explicit reasoning, action logs, and feedback, using entity-relation and event-dependency graphs. It first aligns events across sources in time, merges duplicate records only when their objects and meanings are compatible, and keeps conflicting accounts separate. Changes in meaning, task phase, or feedback state define cognitive seg ments with common fields for objects, evidence, actions, and outcomes. A segment may enter several pipelines. Figure 4 shows their shared dependencies and cross-layer links. The methods below are selected according to the available data. Scores and thresholds require calibration on annotated data, and all outputs begin as candidates for review.

L1: Knowledge. Coreference resolution, entity linking, and relation classification align concepts, objects, and relations with a graph. For segment $c _ { i }$ and graph version $\nu ,$ let the anchor mapping be $\mathcal { M } _ { \nu } ( c _ { i } ) = ( E _ { i } , R _ { i } , \Lambda _ { i } )$ where $E _ { i }$ and $R _ { i }$ contain entities and relations, and $\Lambda _ { i }$ records source locations and event times. Changing entity attributes retain temporal versions, and uncertain links remain candidates. Explicit references, use in judgments or actions, and retractions distinguish a mere mention from evidence of use. Frequency ranking does not automatically remove rare but critical knowledge. The outputs are knowledge units, attribute versions, and anchor mappings. L2–L9 share these mappings for object alignment and source tracing while also reading their own required records.

L2: Attention. Attention cues in language, dependency parsing, and inspection or selection logs identify focus objects, yielding a time-ordered record $\mathcal { F } = \{ ( t _ { k } , f _ { k } , \ell _ { k } ) \} _ { k = 1 } ^ { m }$ . Here, $f _ { k }$ is a focus object or set of objects and $\ell _ { k }$ a source anchor. A transition is recorded only when both adjacent observations are valid. Descriptions and selection changes can then calibrate a switching score; unobserved features remain missing. Across cases, attention frequency is measured relative to opportunities to observe the object in comparable tasks, and patterns are summarized after review. Outputs retain focus objects, transition paths, and evidence types. An unmentioned object is not automatically an ignored object, and model attention weights do not replace practitioner evidence.

L3: Judgment. Explicit reasoning provides judgment targets, supporting and opposing evidence, and exclu sions. Anchors connect conditions to conclusions, and path compression retains intermediate premises. For a record x whose prerequisites hold, exclusions do not apply, and evidence can be scored, let $q _ { r } ( x ) \in [ 0 , 1 ]$ be the calibrated support score for rule r. A review recommendation can be written as

$$
\mathrm { s t a t u s } _ { r } ( x ) = \left\{ \begin{array} { l l } { \mathrm { s u p p o r t e d } , } & { q _ { r } ( x ) \geq \tau _ { + } , } \\ { \mathrm { r e v i e w ~ n e e d e d } , } & { \tau _ { - } \leq q _ { r } ( x ) < \tau _ { + } , } \\ { \mathrm { d o ~ n o t ~ a c c e p t ~ t h i s ~ j u d g m e n t } , } & { q _ { r } ( x ) < \tau _ { - } , } \end{array} \right. \quad 0 \leq \tau _ { - } < \tau _ { + } \leq 1 .\tag{2}
$$

Support is not a probability of correctness. Missing evidence does not receive a zero score, and an opposite conclusion requires its own evidence. With explicit likelihoods and priors, Markov chain Monte Carlo (MCMC) can sample posterior distributions over numerical domain thresholds. Review thresholds $\tau _ { - }$ and $\tau _ { + }$ instead require calibration on labeled examples. Low support alone does not revoke an asset. Change-point detection applies only to ordered data that meet its piecewise distribution assumptions. The output is a judgment rule with applicability, validity and failure boundaries, and evidence references.

L4: Decision. The pipeline extracts candidate actions, reasons for rejection, and trigger, stopping, and fallback conditions, then links actions to operation nodes. Let $\mathcal C ( x )$ be the candidates in context $x .$ . Let $K ( x , a )$ mean that relevant facts and local-rule applicability have been checked, $F ( x , a )$ denote local feasibility, and $H _ { r } ( x , a )$ denote satisfaction of an applicable local restriction $r \in \mathcal { R } _ { \mathrm { l o c } } ( x )$ . Then

$$
\mathcal { C } _ { \mathrm { l o c } } ( x ) = \left\{ a \in \mathcal { C } ( x ) \bigg \vert K ( x , a ) \wedge F ( x , a ) \wedge \bigwedge _ { r \in \mathcal { R } _ { \mathrm { l o c } } ( x ) } H _ { r } ( x , a ) \right\} .\tag{3}
$$

Only actions with all conditions confirmed enter this set. Unknown cases await review. L9 jointly checks required actions, mutual exclusions, and timing across the full plan, so local acceptance does not establish overall compliance. When $| { \mathcal { C } } ( x ) | > 0$ , the ratio $\rho ( x ) = 1 - | \mathcal { C } _ { \mathrm { l o c } } ( x ) | / | \mathcal { C } ( x )$ | measures the reduction to the confirmed set. Excluded candidates are not necessarily infeasible. A single remaining candidate still needs evidence for selection; an empty set calls for clarification or referral.

Condition-action pairs across cases can train a pruned C4.5 decision tree for testing on held-out cases. With explicit states, actions, transitions, and reward features, maximum entropy inverse reinforcement learning can estimate a candidate reward function. Textual comparisons first establish a partial order over criteria Experts can complete pairwise comparisons and check consistency before applying the Analytic Hierarchy Process (AHP) to compute weights. Outputs distinguish candidate, planned, selected, and performed actions A candidate reward function is not treated as the uniquely true preference.

L5: Association. Explicit association statements identify a source concept $u ,$ target concept $v ,$ and a connect ing cue. Graph shortest-path distance $d _ { G } ( u , v )$ and cosine similarity between nonzero embeddings, $\cos ( \mathbf { h } _ { u } , \mathbf { h } _ { v } )$ describe structural distance and semantic similarity. Language cues, the basis of an analogy, and shared attributes help filter candidates while preserving graph versions and scoring evidence. PrefixSpan can mine frequent subsequences from operation logs, but a pattern is labeled as a practitioner association only with ver bal or other process evidence. Outputs include supported associations and links awaiting verification, which can suggest explanations or alternative actions. These associations still need careful interpretation: a missing graph edge does not establish a cognitive leap, and association does not establish causation.

L6: Anticipation. The pipeline extracts the current state, candidate action, expected consequence, and time horizon. It creates prediction-event nodes and freezes the information, rule version, and timestamp available at prediction time. To verify a conditional prediction, it first checks that the action and trigger actually occurred, then matches the object, outcome definition, and observation window. An unexecuted plan is not paired directly with an observed outcome. For a verified pair $i ,$ let $( \widehat { y } _ { i } , y _ { i } ) , ( \widehat { \kappa } _ { i } , \kappa _ { i } )$ , and $( \widehat { t } _ { i } , t _ { i } )$ denote predicted and b bactual numerical values, categories, and event times. Record errors by type:

$$
e _ { i } ^ { \mathrm { n u m } } = \widehat { y } _ { i } - y _ { i } , \qquad e _ { i } ^ { \mathrm { c a t } } = { \bf 1 } [ \widehat { \kappa } _ { i } \neq \kappa _ { i } ] , \qquad e _ { i } ^ { \mathrm { t i m e } } = \widehat { t } _ { i } - t _ { i } .\tag{4}
$$

The indicator 1[·] is 1 when its condition holds and 0 otherwise. Errors retain their own units and are not added across types. If an event does not occur within a complete observation window, an occurrence prediction re ceives a negative outcome and timing error is undefined. An incomplete window or record remains unverified. Probabilistic predictions require calibration checks on independent samples with suficient follow-up labels. Physical tasks may also retain states and contact or support relations. Counterfactual analysis requires particular care: causal mechanisms and identification assumptions must be explicit, and diferences in simulated outcomes are not verified action efects. Outputs contain the prediction, its valid horizon, and verification records.

L7: Monitoring. Sequence labeling identifies uncertainty, requests for verification, revisions, and statements about role limits, linking each to a proposition p and its evidence. The monitoring state $z _ { t } ( p )$ records transitions such as doubt, verification requests, retraction, and reinstatement after checking, together with the evidence that triggered them. A topic change alone is not a state change. Stated confidence, evidence suficiency, graph gaps, and role boundaries remain separate. Linguistic confidence is not converted directly into a probability of correctness. Outputs identify knowledge gaps, unresolved checks, and review conditions for the judgment and decision pipelines.

L8: Habits. Logs are grouped by practitioner, task type, and comparable context, then normalized into objectaction sequences. For $N > 0$ valid case sequences $S _ { i } ,$ , define the case-level support of a nonempty pattern s and a normalized Levenshtein distance with unit insertion, deletion, and substitution costs:

$$
\operatorname { s u p p } ( s ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ s \preceq S _ { i } ] , \qquad d _ { \operatorname { n o r m } } ( S _ { i } , S _ { j } ) = \frac { d _ { \operatorname { e d i t } } ( S _ { i } , S _ { j } ) } { \operatorname* { m a x } \{ 1 , | S _ { i } | , | S _ { j } | \} } .\tag{5}
$$

Here $s \preceq S _ { i }$ denotes an ordered, not necessarily contiguous subsequence, and $| S _ { i } |$ is sequence length. Repeated occurrences within one case count once. Stability checks consider recurrence over time, sample size, and exceptions alongside support and distance. Direct quotations can become wording templates after object and role slots are introduced and semantically equivalent expressions are grouped. Outputs include atomic operations, compound procedures, and optional wording. Performed actions verified in L4 logs can support patterns across cases, and sequence candidates can inform L5, where connecting cues or process evidence are still required. These links reuse corpus content; they are not an online execution loop. An observed habit does not by itself establish competence or a mandatory requirement.

L9: Constraints. Negation-scope parsing and checks of normative sources recover prerequisites, prohibitions, required actions, responsibility boundaries, and alternatives. Within a task and agreed time window, an action identifier a includes its actor, object, and occurrence. Boolean variable $X _ { a }$ records whether the action occurs, with time $T _ { a }$ added when needed. Let $P _ { r } ( x )$ be the applicability condition of rule $r ,$ and $\mathcal { R } ^ { - }$ and $\mathcal { R } ^ { + }$ the prohibition and obligation sets. One constraint formula is

$$
\Phi _ { x } = \Phi _ { \mathrm { t a s k } } ( x ) \wedge \bigwedge _ { r \in { \mathcal R } ^ { - } } \left( P _ { r } ( x ) \Rightarrow \neg X _ { a _ { r } } \right) \wedge \bigwedge _ { r \in { \mathcal R } ^ { + } } \left( P _ { r } ( x ) \Rightarrow X _ { a _ { r } } \right) .\tag{6}
$$

Here $a _ { r }$ is the action governed by rule $r ,$ and $\Phi _ { \mathrm { t a s k } }$ encodes verified facts, prerequisites, mutual exclusions, and timing. Time constraints apply only to performed actions, and diferent scopes are modeled separately. A satisfiability modulo theories (SMT) solver can check $\Phi _ { x } .$ Satisfiability means only that the encoded conditions admit a consistent solution. Unknown applicability still requires evidence and review, even if the formula is satisfiable; a solver assignment is not an observed fact. Unsatisfiable cases and cases for which the solver returns unknown go to human review. Active hard constraints must be satisfied before risk signals determine whether to request review, reduce output detail, or suppress output. Scores cannot override hard constraints. Missing logs do not prove compliance. Outputs retain their basis, scope, and review status.

All nine pipelines save case and segment identifiers, graph anchors, content, conditions, sources, review status, and versions. Cross-layer mapping first checks task, object, and time scope, then uses explicit references, temporal proximity, and content agreement to find and rank candidate links. Source evidence determines whether a link is an evidence reference, candidate addition, or review constraint. Exceeding a score threshold does not by itself establish a relation type or a causal claim. Reviewed reusable content is organized into six asset types while retaining its case and graph links. Quality control checks structure, evidence, conditions, and task use. It does not require every segment to cover all nine dimensions or treat more links as higher quality. The next section discusses the framework’s functional analogies with brain systems.

## 3.4 Functional correspondences with brain systems

The nine-layer approach provides a common vocabulary for extracting, reviewing, and reusing experience from heterogeneous sources. Its design draws inspiration from the organization of cognitive functions in neuroscience. These correspondences guide extraction questions and give the dimensions a coherent functional basis.

Figure 5 connects L1–L9 to functions described in the neuroscience literature. This is an analytical framework for organizing expert experience, with many-to-many functional references to cooperating brain regions and networks. The anterior temporal lobe and distributed association cortex support semantic representation and integration, providing references for L1 knowledge and L5 association [11]. Predictive representations in hippocampal systems connect existing relationships to possible successor states, informing L5 association and L6 anticipation [12]. The frontal eye fields and intraparietal sulcus participate in goal-directed selection, which resembles the selective processing considered in L2 attention [13, 14].

The lateral prefrontal cortex contributes to context-dependent cognitive control, while the orbitofrontal cortex represents and compares values. These functions provide references for judgment, decision, and rule maintenance [15, 16]. Anterior prefrontal and anterior cingulate functions related to self-monitoring, conflict detection, and adjustment ofer analogies for monitoring experience [17, 18]. Striatal circuits involved in goaldirected behavior and habitual responses inform L4 and L8 [19]. Right inferior frontal involvement in response inhibition provides a partial reference for stopping and control in L4 and L9 [20]. Together, these functional analogies organize experience extraction, while explicit records, rules, and review govern permissions and applicability.

## 3.5 Six experience asset libraries

Experience assets package reusable content from case records, practitioner statements, and authoritative procedures for a defined task. The six core types are rules, constraints, best practices, negative examples, corner cases, and skills. “Negative examples” follows the platform’s terminology for failures, inefective practices, and corrections; it does not mean negative-class training examples. The corner-case library records unusual situations in which routine rules may fail or need adjustment, including cases far from a numerical threshold.

Dimensions and asset types are connected many-to-many. L1 facts, entities, and relations generally remain in case records or the shared graph. L6 predictions stay with their horizons and observed outcomes and may also become applicability conditions for a rule, skill, or case. Several dimensions can therefore support one asset.

(a) Brain regions and dimensions  
(b) Functions and supporting literature
<table><tr><td>Region / network</td><td>Function</td><td></td><td>Dimensions</td></tr><tr><td>(ATL)</td><td>A Anterior temporal cortex</td><td>Semantic integration [N1]</td><td>L1, L5</td></tr><tr><td>network</td><td>B Frontoparietal attention (FEF / IPS)</td><td>Goal-directed selection [N2][N3]</td><td>L2</td></tr><tr><td></td><td>C Lateral prefrontal cortex (LPFC)</td><td>Contextual rule maintenance; action control [N4]</td><td>L3, L4, L9</td></tr><tr><td>A ATL L1, L5</td><td>D Orbitofrontal cortex (OFC)</td><td>Value comparison and choice [N5]</td><td>L3, L4</td></tr><tr><td></td><td>E Hippocampus (HPC)</td><td>Relational links and predictive state representations [N6]</td><td>L5, L6</td></tr><tr><td></td><td>F Anterior prefrontal cortex (aPFC)</td><td>Metacognitive accuracy [N7]</td><td>L7</td></tr><tr><td>H Striatum L4, L8</td><td>G Dorsal anterior cingulate cortex (dACC)</td><td>Conflict monitoring; control adjustment [N8]</td><td>L4, L7</td></tr><tr><td></td><td>H Dorsal striatum</td><td>Action selection and habits [N9]</td><td>L4, L8</td></tr><tr><td>D OFC L3, L4</td><td>I Right inferior frontal gyrus (rIFG)</td><td>Response inhibition [N10]</td><td>L4, L9</td></tr></table>

L1 Knowledge L2 Attention L3 Judgment L4 Decision L5 Association L6 Anticipation L7 Monitoring L8 Habits L9 Constraints

Figure 5 Many-to-many functional correspondences between brain systems and the nine-layer framework, based on neuroscience studies. The upper drawing shows the right lateral hemisphere; the lower drawing projects medial and deep structures. Letters match the table rows, and arrows indicate approximate locations. References: N1 [11]; N2 [13]; N3 [14]; N4 [15]; N5 [16]; N6 [12]; N7 [17]; N8 [18]; N9 [19]; N10 [20].

Table 3 Asset types, reusable content, and common source dimensions. Assets can combine information across dimen sions.
<table><tr><td>Asset</td><td>Reusable content</td><td>Common dimensions</td></tr><tr><td>Rules</td><td>Scoped conditions and recommended judgments or actions, including exceptions.</td><td>L3, L4</td></tr><tr><td>Constraints</td><td>Prohibitions or required prerequisites, their basis, and allowed alternatives.</td><td>L7, L9</td></tr><tr><td>Best practices</td><td>Supported successful procedures and conditions for considering reuse.</td><td>L1, L4, L5, L6</td></tr><tr><td>Negative examples</td><td>Inappropriate actions or adverse outcomes, context, and reviewed corrections.</td><td>L3, L4, L7, L9</td></tr><tr><td>Corner cases</td><td>Unusual contexts where a routine rule may fail or need adjustment.</td><td>L2, L3, L5, L7</td></tr><tr><td>Skills</td><td>Callable procedures with inputs, outputs, dependencies, checks, and evidence.</td><td>L2, L3, L4, L8, L9</td></tr></table>

Some content can be included as extensions. Wording styles and role preferences may be optional skill settings. Knowledge gaps remain explicit metadata that trigger collection or review, without requiring another top-level library. Common fields define the core assets, while extensions support domain-specific types in commercial deployments.

The sample used in this report contains 13,113 organizational assets: 3,550 rules, 2,892 best practices, 1,955 skills, 1,948 corner cases, 1,623 constraints, and 1,145 negative examples. Rules and best practices guide judgments and operations. Constraints, negative examples, and corner cases supply applicability conditions, risk information, and exceptions. Skills organize this experience into callable task procedures.

Among the 53 questions with individual tool-call records, C retrieved rules in 46 runs, corner cases in 35, best practices in 33, constraints in 28, and negative examples in 18. It called skills in 46 runs. These records show how libraries are combined during retrieval, reasoning, and execution.

Negative examples and corner cases serve diferent purposes. Negative examples preserve observed mistakes, failures, and corrections to help prevent repetition. Corner cases identify situations that require adjustment even if no failure has yet occurred. Best practices retain both outcomes and conditions for reuse. A skill combines these asset types under a common specification for inputs, steps, state transitions, and outputs.

## 3.6 Evidence, review, and uncertainty

Every transformation should retain the source type: direct observation, practitioner account, or model pro posal. A model may suggest missing relations or reconstruct possible reasons from behavior, but these remain candidate explanations for review and are stored separately from observations and practitioner statements. In this report, distillation means extracting, organizing, and consolidating experience from materials, rather than training a student model from a teacher.

Review status and confidence are separate. Extraction confidence concerns whether parsing recovered the original statement accurately. Evidence suficiency concerns whether sources support it. Applicability concerns whether it can be used in a particular task or context. The system stores these separately so reviewers can inspect each question.

An asset may move through candidate, reviewed, active, suspended, and retired states. Review examines content, evidence, and declared scope. Activation also requires deployment and runtime checks. A correction creates a new version with a replacement link to the old one. Even when a generalization is revoked or found invalid, its original cases remain available to trace how the experience changed.

The organizational assets in this sample retain library versions and entry identifiers, allowing references and version relationships to be checked. Evaluation records should also include the corpus, asset versions, and runtime configuration actually used, supporting review and regression tests after changes.

## 3.7 Graph structures and requirements for agent use

Entity graphs organize task objects and relations; event and dependency graphs describe task progression and links between steps. Edge types distinguish temporal order, procedural dependency, association, and explicitly proposed causal hypotheses. Event order within a case can form a directed acyclic structure. Repeated operations or decisions can instead use repeated event instances or a separate state machine. Explicit edge types keep their meanings distinct.

An asset is ready for agent use when a consumer can understand its content, applicability, evidence, and permitted uses, and respond appropriately to missing prerequisites. The interface therefore needs machine-readable fields for task scope, exclusions, sources, review status, and versions. Skills also require inputs, outputs, and ex ecution conditions. Interface validation checks that required fields and constraints are present; task evaluation checks whether the agent selects and uses the right asset in context.

The following sections explain how corpus construction and consolidation produce traceable assets under these requirements. Section 7 then shows their retrieval and use in recorded workplace injury mediation tasks.

## 4 Experience Acquisition and Corpus Construction

## 4.1 Collecting heterogeneous evidence

Experience collection draws on existing materials and additional records of actual work. Existing sources include case files, work logs, incident reviews, training materials, and demonstration videos. Additional collec tion may involve interviews, task observation, application logs, sensors, audio, and video. The right method depends on the task. An information-heavy review process may already have detailed digital records, whereas physical work often requires observation and follow-up interviews to explain changes in action. Materials should cover key decisions, outcomes, and exceptions. Interviewees should have relevant experience, be able to explain important cues, and provide cases that can be checked.

Process records should preserve the order of work and distinguish information available at the time from information learned later. Interviews can recover omitted observations, rejected alternatives, and conditions under which the usual procedure fails. These accounts should be marked as retrospective. Measurements or observations that remain unavailable should be listed explicitly for later collection.

The authorized sample contains 1,576 files in 13 formats: 567 DOCX, 351 PDF, 172 JSON, 154 Markdown, 131 DOC, and 95 XLSX files, plus text, webpages, images, presentations, and audio. The materials include case files, procedures, standards, training materials, interviews, question-answer records, tabular data, and de-identified records. Processing therefore needs to handle narrative text, tables, and multimodal attachments while keeping related content linked.

![](images/7d0b938ddf6416f5b664d4240bfd928609666fe143ce737b8f73c42e85aa2439.jpg)  
Figure 6 Four representations in corpus construction. Source materials (S0) become case narratives (S1), which are analyzed for decision elements (S2) and converted into structured candidate records (S3). Dashed links preserve source traceability across stages.

## 4.2 Semantic alignment

The platform retains original wording while normalizing identifiers, terms, event boundaries, and units. Abbreviations, colloquial or local expressions, and ambiguous references require context. A mapping record should preserve the original phrase, normalized term, supporting context, and plausible alternatives that remain un resolved. A phrase such as “slightly hot” should become a numerical range only when measurements or a reviewed domain definition support that mapping.

Multimodal observations are aligned with task events and time intervals. Transcriptions and image recognition outputs retain uncertainty and source locations. Sensor records include units, sampling methods, and available calibration information. Because recognition errors can afect later rules and skills, every derived claim should remain traceable to the original record.

The hypertension practitioner provides a concrete example. Their materials comprise two 2024 Chinese hypertension guidelines and XML field definitions for a hypertension follow-up dataset. The guidelines describe diagnosis, monitoring, and follow-up. The XML specifies fields such as systolic and diastolic blood pressure, body mass index, medication adherence, adverse reactions, referral reasons, and the next follow-up date. Semantic alignment connects guideline conditions to these fields, units, and value ranges while retaining guideline ver sions and source locations. Later follow-up entries can then use consistent fields and refer back to the relevant guideline passages. The XML supplies the data schema; actual follow-up data are entered during subsequent clinical work.

Table 4 Core transformation operators and reasons to reject an output or leave it unresolved. Every operator records input and output versions and source locations.
<table><tr><td>Operator and input</td><td>Transformation and output</td><td>Reject or leave unresolved when</td></tr><tr><td>Term alignment: wording and context</td><td>Propose a standard term while retaining aliases, units, and source spans.</td><td>Multiple interpretations remain plausible, or a unit conversion lacks support.</td></tr><tr><td>Claim extraction: case narrative</td><td>Separate observations, judgments, and planned actions, each with its own source.</td><td>The extracted claim lacks supporting content.</td></tr><tr><td>Condition preservation: conditional statement</td><td>Recover triggers, recommendations, prerequisites, and exceptions as a candidate rule.</td><td>Compression changes the action, drops an exception, or alters a prerequisite.</td></tr><tr><td>Conflict classification: comparable asset pair</td><td>Check scope overlap and action compatibility to distinguish agreement different conditions, and conflict.</td><td>Scope overlap is unknown or evidence is insufficient to resolve the conflict.</td></tr></table>

## 4.3 Narrative reconstruction and structured extraction

The S0–S3 stages in Figure 6 separate case reconstruction from experience generalization. S0 retains source materials. S1 organizes the case by roles, observations, actions, and outcomes. S2 identifies decision elements, including cues, explanations, alternatives, plan changes, and uncertainty. S3 converts supported elements into structured records and graphs. All stages remain linked to their sources.

The extractors then apply the nine-layer approach in Section 3.3. Knowledge extraction links terms and entities. Attention extraction identifies priorities supported by practitioner statements or behavioral evidence. Judgment extraction recovers conditions and conclusions. Decision extraction records actions, selection criteria, and stopping conditions. Association extraction captures related cases that the practitioner mentions, without treating similarity as causal evidence. Anticipation extraction records predictions and time horizons. Monitoring extraction records when more information, review, or referral is needed. Habit extraction requires repetition across cases. Constraint extraction distinguishes explicit prohibitions from preferences.

An extractor may combine language model prompts, rule parsing, graph processing, and domain rules. If it involves model training, the training data, objective, and results on an independent validation set should also be recorded. Regardless of implementation, candidate assets need structured fields and supporting evidence that a reviewer can inspect.

A model can propose candidates for practitioners to review against these criteria. Each stage retains its input span, proposed output, revisions, and review decision. Consider the statement “During initial intake, request verification before assigning a cause, except when a separate emergency procedure applies.” A valid rule must retain both the intake prerequisite and the emergency exception. Dropping either fails the condition preservation check.

Preserving conditions during transformation. For a rule $r ,$ let $P _ { r }$ denote its prerequisites, $E _ { r }$ its exclusions, and $a _ { r }$ its recommended action. Within a predefined set of task states $\Omega ,$ , its allowed scope is

$$
D ( r ) = \{ x \in \Omega : P _ { r } ( x ) = { \mathrm { t r u e ~ } } \land E _ { r } ( x ) = { \mathrm { f a l s e } } \} .\tag{7}
$$

Unknown values do not establish applicability. For a source rule $r _ { s }$ and an extracted rule $r _ { e } ,$ define the added and lost scope as

$$
B ^ { + } ( r _ { e } , r _ { s } ) = D ( r _ { e } ) \setminus D ( r _ { s } ) , \qquad B ^ { - } ( r _ { e } , r _ { s } ) = D ( r _ { s } ) \setminus D ( r _ { e } ) .\tag{8}
$$

Normalization should leave both sets empty while preserving the action’s meaning and required exceptions. A nonempty $B ^ { + }$ introduces unsupported situations; a nonempty $B ^ { - }$ removes situations covered by the source. An

intentional scope change requires a separate review. These sets define the consistency target, which targeted task tests check through conditions and applicability.

## 4.4 Quality control and corpus versions

Quality checks operate at three levels. Structural checks verify identifiers, required fields, valid references, and types. Semantic review checks whether evidence supports the content and its stated scope. Task review checks whether an asset helps make a decision or follow a procedure without hiding missing prerequisites. Review by other practitioners can expose ambiguities and assumptions that the original contributor takes for granted, and help distinguish personal habits from procedures suitable for wider use.

Duplicate detection should consider derivation as well as text similarity. Several summaries of the same event are not independent evidence. Each experiment or deployment should fix a data version and record its source files, processing programs, active assets, and review decisions. Earlier versions must remain available after updates so that reported results can be checked.

Review status of the sample materials. The authorized source materials occupy about 1.73 GB. The file inven tory marks 1,564 files as Approved and 4 as Rejected, with 8 lacking a review status. These are file counts. Task sets, corpus versions, and processing configurations are recorded separately for evaluation review.

In the comparative evaluation, adding raw-corpus retrieval raised the composite mean from 70.63 in A to 79.75 in B. The evidence suficiency and accuracy score rose from 62.18 to 77.27. These results demonstrate that parsing, indexing, and retrieving source materials strengthen evidence-grounded answers and improve task performance.

## 5 Organizational Experience Consolidation

## 5.1 From individual accounts to organizational assets

Practitioners may agree, difer because they work under diferent conditions, or recommend diferent actions under the same conditions. Simply pooling their libraries can hide these distinctions. Before combining assets, KUPAS MASTER compares their applicability, supporting evidence, and review status.

The platform first aligns task identifiers, entities, terms, and asset types. Semantic similarity and scope overlap then identify entries worth comparing. Similarity is a way to find candidates, not suficient evidence for a merge. The comparison asks whether the assets address the same decision, whether their conditions can hold together, and whether their judgments or actions are compatible.

## 5.2 Agreement, diferent conditions, and conflicts

Figure 7 shows four consolidation outcomes. When accounts agree under the same conditions, the combined entry retains all independent sources and their derivation links. When diferent conditions explain diferent recommendations, both procedures remain available with explicit conditions. A routine inspection and a pro cedure for abnormal operation should not be averaged into one sequence.

When comparing alternatives, the platform organizes historical outcomes by case dificulty, operating environment, resources, and outcome definition. Historical records preserve practical evidence; controlled compar isons help assess diferences between procedures under comparable conditions.

An unresolved contradiction remains visible as linked alternatives for review or adjudication. It should not become an unconditionally active rule. Reviewers may narrow its scope, request more cases, or decide that neither alternative is ready for use. They also distinguish formal requirements from personal preferences. A requirement grounded in a standard or procedure cannot be settled by a simple vote or a compromise with preferences.

![](images/6bb068ebcdc942dba863aa874eb726f2e62e580f3f113967523c7fe52c91ca9f.jpg)  
Figure 7 Four outcomes of organizational consolidation and their evidence requirements. Agreement retains independent support. Context-dependent alternatives remain separate. Outcome comparison requires comparable cases. Unresolved contradictions remain inactive until review. Consolidated assets retain both supporting evidence and conditions of use.

Using the scope in Eq. (7), define the overlap and conflict regions of two rules as

$$
O _ { i j } = D ( r _ { i } ) \cap D ( r _ { j } ) , \qquad C _ { i j } = \{ x \in O _ { i j } : \mathrm { c o m p a t i b l e } ( a _ { i } , a _ { j } , x ) = \mathsf { f a l s e } \} .\tag{9}
$$

Compatibility depends on task order, resources, and procedural requirements. Diferent actions need not conflict. A nonempty $C _ { i j }$ requires adjudication. Unknown overlap or compatibility remains unresolved. If the overlap is empty, both rules can be retained within the checked task scope. Domain review is still needed to establish the relevant conditions and action compatibility.

For example, one intake procedure may apply when there is no emergency and another when there is an emergency. If emergency status is known, their scopes are disjoint and both can be kept. If both apply to non-emergency intake but require incompatible next actions, the platform records a conflict with the asset versions, shared conditions, and unresolved conclusion. If emergency status is unknown, the agent first asks for the missing information.

The workplace injury mediation sample also requires separate procedures for ordinary compensation calcu lations, third-party commercial insurance, and statutory work injury insurance. The platform must record the insurance type, liable party, eligible compensation items, and mediation authority. A record that merely says “insurance exists” leaves applicability unknown. The agent must verify the type and beneficiary relation ship before using the relevant experience. Separate conditions make it possible to retrieve the procedure that matches the case.

## 5.3 Cross-review and version management

Other practitioners can review candidate assets and how they were produced. They check sources, conditions, exceptions, consistency with other assets, and possible misuse. Tests that change one important condition, such as whether a measurement is available or referral is required, help check whether applicability is clear.

Consolidation produces both organizational assets and decision records: which inputs were merged, which were kept separate, why, and what remains unresolved. Each update creates a new version. Dependencies identify skills that need revalidation after a rule changes. Version records preserve original contributions and the exact assets used in a run.

Review records in the sample. In the sample collected for this report, organizational assets are stored by library and version. Records for 8 practitioners contain 37 completed cross-review jobs across the six libraries, covering 818 candidates. “Completed” means that comparison, scoring, and record storage finished. Human confirmation is recorded separately: 9 candidates had completed human verification and 331 were marked as awaiting it. Separating automated results from human review status makes the processing state of each candidate clearer.

For the workplace injury mediator, two rule-library review jobs used the same source version. Each involved 37 source assets and 21 candidates, with processing status retained. These records make experience traceable from its source through review and consolidation. The three-configuration evaluation demonstrates the value of the resulting organizational assets and skills in professional task execution.

## 6 Skill Construction and Agent Integration

## 6.1 Skill invocation and execution requirements

![](images/c034d48c206051e1f62e8e43b290d75ac3862732497842d5e12e63463752913e.jpg)  
Figure 8 Skill selection and execution. Retrieval retains active, authorized assets with true or unknown applicability. Invocation requires satisfied prerequisites and action checks. Missing information prompts clarification and another state check; a rejected action may lead to escalation. Actions and escalations are logged. Mandatory constraints apply independently of retrieval ranking.

A skill organizes assets produced through nine-layer cognitive corpus construction into a procedure an agent can call. Its specification includes applicability, inputs, output structure, ordered steps, tool dependencies, constraints, stopping conditions, evidence references, and versions. Together, these fields make the procedure’s inputs explicit and its execution and outputs checkable.

Skill construction starts from reviewed assets. Rules support judgments and state transitions. Best practices supply candidate procedures. Constraints and negative examples define checks. Corner cases identify when to pause or adjust the usual procedure. Each skill combines these elements into a complete procedure, with suggested wording where useful. A dependency list links each part of the procedure to its supporting assets, so corrections can trigger targeted revalidation.

The system retrieves and calls external assets during inference. Rules, constraints, and skills can therefore be updated without retraining the base model. Regression evaluation should check that a correction takes efect and that unrelated tasks do not degrade.

## 6.2 Selection and execution

Let A be the asset set, $\sigma ( r )$ an asset’s status, $\phi _ { r } ( x )$ its three-valued applicability after considering prerequisites and exclusions, and access $( r , x )$ its access permission. For task state $x ,$ retrieval keeps active, authorized assets whose applicability is true or unknown. Invocation uses a stricter subset:

$$
{ \mathcal { R } } _ { x } = \{ r \in A : \sigma ( r ) = { \mathrm { a c t i v e } } , { \mathrm { a c c e s s } } ( r , x ) = { \mathrm { a l l o w e d } } , \phi _ { r } ( x ) \neq { \mathrm { f a l s e } } \} ,\tag{10}
$$

$$
{ \mathcal { X } } _ { x } = \{ r \in { \mathcal { R } } _ { x } : \phi _ { r } ( x ) = \operatorname { t r u e } \} .\tag{11}
$$

Retrieval and relevance ranking operate within $\mathcal { R } _ { x }$ . A relevant asset with unknown prerequisites can prompt a targeted question or a request for an observation. Only assets in $\mathcal { X } _ { x }$ are eligible for invocation, which still requires dependency and action checks. False applicability excludes an asset. Applicability uses strong threevalued logic: a missing value is unknown, conjunction is false if any operand is false and true only if all are true, and negating unknown remains unknown. Unknown cannot be treated as either false or satisfied. Ranking may consider relevance, evidence, and recency, and its policy should be included in the evaluation record.

The agent binds known values to skill inputs, identifies missing prerequisites, and proposes a next step (Fig ure 8). Before an action with material consequences, it checks constraints and tool permissions. Mandatory checks operate independently of relevance ranking and top-k truncation. An unknown mandatory condition blocks the action. Clarification targets fields whose values could change the next decision, and the updated state is checked again. Tool outputs are observations, not instructions that can override the task or platform controls. Output validation checks the required schema, reported evidence, and stopping conditions. A failed check can lead to a bounded retry, more information, a revised plan, or human referral.

Execution need not follow a fixed chain. Skills can branch, but branch conditions and allowed actions must be checkable. A new observation can invalidate a previously satisfied prerequisite, so checks must occur before action rather than only at task entry. The task record preserves the actual sequence, including failed attempts and human intervention.

Runtime logs record asset and skill calls for tracing and branch testing. The next section examines how ex perience processed by nine-layer cognitive corpus construction supports judgments, boundary checks, and procedures in actual runs.

## 7 Case Study and Task Runs

## 7.1 Comparing the three configurations in workplace injury mediation

To examine the use of extracted experience in mediation, we compared the base model, raw-corpus RAG, and KUPAS MASTER configurations for a workplace injury mediator. This practitioner supplied just 2 interview files, which produced 167 experience assets, and completed rule-library cross-review jobs.

One representative task asked whether a trafic accident during a detour to collect a child after work could qualify as a workplace injury. The answer also had to identify evidence about the route, timing, and purpose, and set an order for addressing disputed issues in mediation. The base model provided a general framework and evidence checklist but cited no specific regulation and did not clearly distinguish mediation from formal injury determination. Raw-corpus RAG cited the Regulations on Work-Related Injury Insurance and proposed mediation steps. The KUPAS MASTER agent additionally called the “Mediation Request Assessment and Focus” skill and consulted rule and corner-case libraries. It distinguished the boundaries of injury determination, mediation authority, and high-risk issues, and generated an output file. It was rated best on all 8 comparison items, including 1 tie with raw-corpus RAG.

A second recorded question concerned a work-related injury with grade-ten disability, part-time employment, and several compensation amounts. The base model focused on legal principles and inferring amounts. Rawcorpus RAG added practical calculations and mediation advice. The KUPAS MASTER agent called a mediation skill and used rules and best practices to organize wage evidence, compensation calculations, reasons for employment termination, and judicial confirmation into a complete process. It also generated a “Preliminary Analysis of a Workplace Injury Mediation Case” file.

Table 5 Three-configuration comparison for workplace injury mediation. All configurations assess an accident during a detour to collect a child after work, the evidence needed, and the order of mediation steps.
<table><tr><td>Item</td><td>A: Base model</td><td>B: Raw-corpus RAG</td><td>C: KUPAS MASTER</td></tr><tr><td>Approach</td><td>General analysis and an evidence checklist.</td><td>Cites work injury insurance regulations and proposes mediation steps.</td><td>Calls the mediation assessment skill and combines rules with corner cases.</td></tr><tr><td>Boundaries</td><td>Does not clearly separate mediation from formal injury determination.</td><td>Identifies some procedural and risk boundaries.</td><td>States that mediation does not replace formal determination and distinguishes accident</td></tr><tr><td>Output</td><td>General principles and evidence suggestions.</td><td>Legal grounds and process advice.</td><td>scenarios and risks. A structured process and an output file.</td></tr><tr><td>Comparison</td><td>Rated lowest on all 8 items.</td><td>Ties with C on 1 item.</td><td>Rated best or tied for best on all 8 items.</td></tr></table>

On this practitioner’s 8 evaluation questions, A, B, and C scored 73.34, 80.80, and 90.58, respectively. C exceeded B by 9.78 points. Mean end-to-end times were 13, 31, and 53 seconds. Structured experience improves the completeness and usefulness of professional analysis by supplying task boundaries, judgment conditions, and executable procedures. The comparison shows the added value of organizing source knowledge into experience assets and skills. The next two sections describe the evaluation setup and results across domains.

## 8 Evaluation Design

We systematically compare three configurations on a common evaluation protocol to assess the efectiveness of nine-layer cognitive corpus construction and use run records to explain how assets and skills contribute to task performance. The evaluation uses authorized samples from 20 randomly selected practitioners and covers 177 questions and 531 responses. A uses the base model, B retrieves the raw corpus, and C uses KUPAS MASTER experience assets and skills.

![](images/9b77f1c75776475398344aa9d43e0eaab11d2d85726f9755fede4c63e32bdd17.jpg)  
Figure 9 The three-configuration comparison. Direct inference provides a model-only reference. Raw-corpus RAG measures the benefit of access to source records. The KUPAS MASTER agent evaluates the complete experience-guided configuration. All three answer the same tasks under common input requirements and scoring criteria.

## 8.1 Evaluation objectives

The evaluation measures the KUPAS MASTER agent’s improvement over the base model and raw-corpus RAG. We analyze the gains alongside case representation, experience collection, consolidation, skill construction, and boundary controls to connect task results with the platform workflow.

Tasks should have clear inputs, bounded goals, and assessable outputs. AgentBench and benchmarks for web, enterprise, desktop, and software tasks illustrate environment-based evaluation [21–25]. Benchmarks for tool and user interaction and scientific workflows further examine dialogue consistency and output validation [26– 28]. Mediation tasks can test whether an agent identifies missing facts, selects the next information-gathering step, checks a response against reviewed criteria, or chooses an escalation path. Ofline evaluation measures the accuracy and usefulness of advice. Evaluation in actual use can additionally track outcomes such as dispute resolution. Studies of professional writing and human-AI collaboration suggest focusing on task outcomes and comparing against the stronger of human-only and AI-only performance [29, 30].

## 8.2 Design principles

Figure 9 shows the comparison. A receives the base model, task prompt, and observations available at task entry, with no domain retrieval. B adds retrieval from raw domain records, with the parsing and indexing needed to support it. C uses assets and skills constructed from those records. Across the experiments, each configuration answers independently under the same tasks, input requirements, and scoring rubric. Model settings, sampling parameters, tool permissions, and runtime resources follow a common protocol, as do the rules for providing task clarification. This consistent setup makes the results directly comparable.

B and C draw on the same authorized source materials, including practitioner interviews. B retrieves the original content, while C uses experience assets and skills derived from it. This shared information base allows the comparison to measure the value added by experience structuring and skill construction. Retrieval, skill calls, and end-to-end time are retained to analyze quality and resource use. Configuration records cover B’s parsing, chunking, indexing, and retrieval, and C’s corpus snapshot, experience retrieval, and skill versions.

## 8.3 Experimental setup and data

Sample processing. KUPAS MASTER is commercially deployed and has accumulated experience from many practitioners. This report uses a random sample of 20 authorized users. Their 1,576 source files yield 30,762 retrievable chunks, 23,024 individual experience records, and 13,113 organizational assets after consolidation. The corresponding Elasticsearch indices contain 43,882 documents.

Shared identifiers link processing stages. The file inventory records sources, formats, and review status. Individual records link to contributors and cases. Organizational assets add types, versions, and source relationships. Index documents support retrieval and agent use. The sources span 13 formats, and 17 of the 20 practitioners have all six asset types. These records support analysis of corpus composition, asset distributions, and use across tasks.

Tasks and run configurations. The 177 evaluation questions each receive an independent answer from A, B, and C, producing 531 responses. Domains include finance and accounting, community governance, engineering quality, safety oversight, healthcare, mediation, and emergency management. Table 6 summarizes the setup. The comparison evaluates experience use during inference: raw-corpus retrieval and structured experi ence assets improve task performance without retraining the base model. Reviewed experience also provides reusable material for subsequent model adaptation.

Scoring and run records. An LLM applies a shared rubric to score A, B, and C on the same questions across seven dimensions: result correctness, output actionability, depth of professional judgment, evidence suficiency and accuracy, appropriate tool and skill use, accuracy in understanding requirements, and boundaries and compliance. Their weights are 22, 15, 15, 15, 11, 11, and 11, summing to 100. Main-text score means average the platform-reported practitioner summaries equally. These summaries are rounded at source; question-level distributions and bootstrap intervals are recomputed from the weighted dimension scores. Small diferences in the last decimal can arise from that rounding. Displayed aggregates use decimal round-half-up at the stated precision.

Table 6 Experimental setup and data composition.
<table><tr><td>Item</td><td>Configuration and scale</td></tr><tr><td>Sample</td><td>20 randomly selected authorized practitioners, covering finance and accounting, community governance, engineering quality, safety oversight, healthcare, mediation, and emergency management.</td></tr><tr><td>Tasks</td><td>177 questions, independently answered by A, B, and C, for 531 responses.</td></tr><tr><td>Difficulty</td><td>8 easy, 83 medium, and 86 hard questions.</td></tr><tr><td>Configurations</td><td>A: base model; B: raw-corpus RAG; C: KUPAS MASTER agent.</td></tr><tr><td>Scoring</td><td>A shared seven-dimension rubric applied to independent answers to the same tasks.</td></tr><tr><td>Run traces</td><td>Individual records for 5 practitioners and 53 questions, covering retrieval, library calls, and skill execution.</td></tr></table>

The evaluation results support paired and stratified analysis by practitioner, domain, dificulty, and task form. Detailed run traces are available for 5 practitioners and 53 questions, showing knowledge-base retrieval, rawcorpus access, asset-library calls, and skill execution. In C, 52 runs retrieved from the knowledge base and 46 called skills. Rules, corner cases, best practices, constraints, and negative examples all appear in the traces. These records connect final scores to actual asset use.

Evaluation materials, run outputs, and summaries share a common data structure. Practitioner summaries retain question counts, configuration scores, failed questions, end-to-end time, and dimension scores along side question-level results. Task records remain linked to the assets derived from their source materials. In workplace injury mediation, for example, the agent calls the “Mediation Request Assessment and Focus” skill and combines rules and corner cases to address procedural boundaries, fact verification, and risks. The dataset thus supports both outcome comparisons and analysis of the path from materials to assets, skill calls, and task outputs.

## 8.4 Metrics and judgment

Task performance is measured by a composite score, the weighted sum of the seven dimension scores, and a score pass rate, the fraction of questions scoring at least 60 points. The framework also defines task success for subsequent workflow evaluations as meeting a predefined domain rubric and all mandatory boundary conditions. Each run then has a binary outcome. For K prespecified stochastic runs, let $s _ { i m } = K ^ { - 1 } \sum _ { k = 1 } ^ { K }$ s<sub>imk</sub> be the mean success rate of method m on case $i ,$ with $s _ { i m k } \in \{ 0 , 1 \}$ . Its paired diference from baseline b is

$$
\widehat { \Delta } _ { m , b } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( s _ { i m } - s _ { i b } ) .\tag{12}
$$

For these workflow evaluations, the protocol fixes the rubric, mandatory failure conditions, repeat count, and stratum weights before testing, and reports stratum results, macro-averages, and deployment-weighted scores. Each original case is the statistical unit, with all its prespecified runs included in the case mean. Case-level resampling estimates intervals for paired diferences while preserving within-case dependence. For the score gains reported here, practitioner-cluster resampling retains dependence among a practitioner’s questions and produces 95% intervals.

Secondary measures include factual support, procedural completeness, applicability errors, prohibited actions, unnecessary refusals, and escalation quality. Evidence faithfulness measures how well a cited passage supports the corresponding claim. Prior work studies claim-level factuality and citation verification [31, 32]; RAGAs and RAGChecker distinguish retrieval quality from generation quality [33, 34]. The evaluation framework distinguishes citation coverage from factual support, checks claims against sources, and measures latency and token use under a common runtime budget. Collection and expert review costs form a separate part of the cost analysis.

Extensions to human evaluation will use randomized output order, masked system identities where feasible, a fixed rubric, and independent review of a subset, with reviewer qualifications, agreement, and adjudication recorded. Human checks complement model scoring for calibration and scale [35]. Workflow evaluations will connect these assessments to business outcomes over a defined observation period.

## 9 Experimental Results and Analysis

Table 7 shows gains from both raw-corpus RAG and the KUPAS MASTER agent. Calculated before rounding the aggregate means, B improves on A by 9.13 points, and C improves on B by 9.83 points. C gains across all scoring dimensions, including 11.1 points in evidence suficiency and accuracy and 9.0 in boundaries and compliance (Figure 12). The 95% practitioner-cluster bootstrap interval for C minus B is [9.20, 10.45], confirming a consistently positive score gain.

Table 7 Overall performance of the three configurations. Higher scores and pass rates are better; lower latency is better. Scores and latency weight practitioners equally; pass rates weight the 177 questions equally, with a pass threshold of 60.
<table><tr><td>Method</td><td>Composite score</td><td>Score pass rate (%)</td><td>Evidence sufficiency and accuracy</td><td>Boundaries and compliance</td><td>Time (s)</td></tr><tr><td>A: Base model</td><td>70.63</td><td>87.57</td><td>62.18</td><td>74.52</td><td>16.2</td></tr><tr><td>B: Raw-corpus RAG</td><td>79.75</td><td>98.87</td><td>77.27</td><td>79.19</td><td>34.0</td></tr><tr><td>C: KUPAS MASTER</td><td>89.58</td><td>100.00</td><td>88.32</td><td>88.23</td><td>57.1</td></tr></table>

## 9.1 Composite scores

Figure 10 shows the composite score for each practitioner. The mean rises from 70.63 for A to 79.75 for B and 89.58 for C. Diferences computed before rounding the aggregate means are 9.13 points for B minus A, 9.83 for C minus B, and 18.96 for C minus A. All 20 practitioner means follow $A < B < C$ . At the question level, C exceeds both A and B on 176 of 177 questions, demonstrating consistent improvement across professional tasks.

## 9.2 Performance by dimension

Figure 11 compares result correctness, output actionability, depth of professional judgment, evidence sufi ciency and accuracy, appropriate tool and skill use, accuracy in understanding requirements, and boundaries and compliance. C’s mean scores range from 87.7 to 91.2, and all 20 practitioners score higher in C than B on every dimension. The gains therefore cover output quality, evidence, judgment, execution, and compliance boundaries.

## 9.3 Gains at each stage

Figure 12 separates the change from A to C into two stages. B minus A measures the gain from raw-corpus RAG.   
C minus B measures the additional gain from assets and skills constructed through the nine-layer approach.

![](images/e120753e5f68f9d3980ed05b48741471a8b46f4d68597421e3924a433fd565a7.jpg)  
Figure 10 Composite scores for the 20 practitioners. Equal weighting across practitioners gives means of 70.63, 79.75, and 89.58 for A, B, and C, with $C - B = + 9 . 8 3$ and $C - A = + 1 8 . 9 6$ . C exceeds both baselines for every practitioner.

The overall gains, 9.13 and 9.83 points, are similar in size. Both access to source records and the structured experience configuration improve scores under the shared evaluation process.

The practitioner means, seven dimensions, and staged comparisons all favor C. The next analysis uses asset calls, skills, and run traces to examine how the experience configuration participates in task execution.

## 9.4 Sources of gains and runtime behavior

We examine organizational asset counts, run logs, and representative tasks to understand C’s gains over rawcorpus RAG. The analysis considers asset organization, retrieval, skill use, source formats, and corpus size.

The 20 practitioners contributed 23,024 published individual entries, which became 13,113 organizational assets after consolidation. The process combines duplicate or similar experience and organizes individual entries into shared assets. Consolidation adapts to each practitioner’s source material, with the reduction reflecting diferences in repetition, content structure, and organization. Retrieval then operates over structured rules, best practices, constraints, negative examples, and corner cases.

For the 5 practitioners and 53 questions with complete individual traces, C performs knowledge-base retrieval in 52 runs and calls skills in 46. Rules, corner cases, best practices, constraints, and negative examples appear in 46, 35, 33, 28, and 18 runs, respectively. Compared with B’s main reliance on raw-corpus retrieval, C combines several asset types and calls skills where needed. Table 8 summarizes their observed roles.

![](images/ef139169bdaa628ea621b7aae89d399d5b5b5342af547af173ef6b1166eea18f.jpg)  
Figure 11 Mean scores in seven dimensions, weighting the 20 practitioners equally. C scores range from 87.7 to 91.2. For every practitioner, C exceeds B in every dimension.

![](images/cd3468ca6e4065fddbf6e50a45bf055550fd216e4b79da655664d3588f5e22df.jpg)  
Figure 12 Score gains by dimension. Orange shows the gain from raw-corpus retrieval (B A); green shows the additional gain from experience assets and skills (C B). Labels above the bars give the total (C A). All dimensions weight the 20 practitioners equally.

Table 8 Asset and skill use in 53 C runs for 53 questions. Counts are by run and can overlap.
<table><tr><td>Asset type</td><td>Runs</td><td>Observed role and example tasks</td></tr><tr><td>Rules</td><td>46</td><td>Express judgment conditions as rules, thresholds, and requirements, including flood dispatch decisions and engineering quality escalation criteria.</td></tr><tr><td>Best practices</td><td>33</td><td>Supply established procedures and steps that turn experience into a task process.</td></tr><tr><td>Corner cases</td><td>35</td><td>Add exceptions, high-risk situations, and business boundaries that routine rules may miss.</td></tr><tr><td>Constraints</td><td>28</td><td>Identify prohibited actions or required prerequisites for procedural and compliance checks.</td></tr><tr><td>Negative examples</td><td>18</td><td>Provide recurring errors and inappropriate responses to help recognize risky decisions and exceptions.</td></tr><tr><td>Skills</td><td>46</td><td>Organize retrieved experience into steps, checks, responsible roles, and structured deliverables.</td></tr></table>

Examples make these roles concrete. In flood and typhoon response, rules, best practices, and corner cases support judgments about water levels, dispatch order, and de-escalation conditions. In safety management, negative examples and corner cases flag hazards such as unplanned rescue attempts in confined spaces and distinctions between ordinary hazards and major accident risks. In social insurance auditing, skills organize experience into plans, checklists, review forms, and ledgers. Assets thus contribute conditions, risk boundaries, and procedures as well as retrieved text.

The gains span appropriate tool and skill use, evidence suficiency and accuracy, depth of professionaljudgment, output actionability, and boundaries and compliance. Removing the appropriate tool and skill use dimension and renormalizing the remaining weights to 100 leaves an average C-minus-B gap of about 9.3 points, close to the full-score gap of 9.83. The diference extends to evidence use, judgment, and task outputs.

All 20 practitioners gain across diferent corpus sizes and formats. Raw chunk counts, organizational asset counts, and library sizes show no stable relationship with C’s gain over B. Spreadsheets can produce many cell fragments, JSON contains structural fields, and documents usually provide more continuous text. The platform organizes these sources into a common representation. The consistent gains across these varied corpora demonstrate the practical value of organizing heterogeneous experience for task use.

The traces show a connected process: experience extraction produces assets, retrieval combines relevant assets for the current question, and skills organize them into steps, checks, and deliverables. This workflow turns experience into stronger evidence, clearer judgments, actionable outputs, and explicit boundary checks, connecting the score improvements to concrete task behavior.

## 9.5 Eficiency

Across 20 practitioners and 177 questions, mean end-to-end latency, weighting practitioners equally, is 16.2 s for A, 34.0 s for B, and 57.1 s for C. The means of practitioner-level 90th-percentile (P90) latencies are 21.1, 41.1, and 68.4 s. C’s mean latency is about 3.5 times A’s and 1.7 times B’s.

In the detailed-trace subset of 5 practitioners and 53 questions, C performs knowledge-base retrieval in 52 runs and calls skills in 46. Library usage is 46 for rules, 35 for corner cases, 33 for best practices, 28 for constraints, and 18 for negative examples. Mean latency on this same subset is 17.4, 36.2, and 59.1 s for A, B, and C.

The platform delivers stronger professional judgments, better supporting evidence, and actionable outputs in about a minute on average. This response time accommodates experience retrieval, asset checks, and skill execution within a practical workflow for business analysis, planning, and professional review. The result is a substantial quality improvement with a response window suited to these tasks.

## 10 Commercial Deployment

KUPAS MASTER is a commercial platform with cloud and local deployment options. It separates corpus production, asset management, agent execution, and evaluation records, and uses permissions, versions, and run records to manage access across these stages. The experiments use the release dated September 16, 2026. The platform continues to evolve.

## 10.1 Customized local deployment

The platform supports deployment of corpus processing, asset management, retrieval, and agent execution on customer-owned servers or a private cloud. Domain terms, library structures, review rules, and skill workflow can be configured for an industry’s requirements and business processes.

For applications with strict confidentiality requirements, local model inference and internal tool services can keep source materials, assets, task inputs, outputs, and logs within the customer environment. In this con figuration, business data need not pass through external model APIs or third-party services. Authentication, role-based permissions, access isolation, and audit logs govern collection, sharing, and agent use. Customers can control where data are stored, how they are used, and who can access them, while retaining a record of operations.

## 11 Related Work

Acquiring and organizing professional experience. The Critical Decision Method and Applied Cognitive Task Analysis use questions about specific incidents to recover cues, judgments, and strategies missing from routine records [3–5]. Organizational knowledge creation also studies how individual knowledge becomes shared practice [36]. KUPAS MASTER applies these ideas to the design of records for agents. Its nine dimensions organize expert judgments, actions, and feedback into experience that can be extracted, reviewed, and reused.

Retrieval and structured corpora. Dense Passage Retrieval and Fusion-in-Decoder study learned retrieval and generation from multiple passages [37, 38]. Self-RAG adds retrieval and self-critique decisions, while RAP-TOR organizes recursive summaries [39, 40]. GraphRAG and LightRAG retrieve linked information through graphs [7, 41], and HippoRAG 2 studies nonparametric continual learning [42]. These approaches ofer alternatives to a simple raw-corpus index. A longer context alone does not ensure that a model uses the relevant evidence [43]. Our focus is the experience contributed by practitioners and the conditions under which it applies.

Memory and experience reuse. CoALA distinguishes memory and action components in language agents [44]. Generative Agents uses experience records and reflection to guide simulated behavior [45]. Reflexion and ExpeL derive reusable feedback from agent attempts [46, 47]. A-Mem and Mem0 study structured or consolidated memory, while ACE and LangMem support evolving context and persistent memory [48–51]. KUPAS MASTER focuses on externally collected practitioner accounts, claim-level evidence, conditional disagreements, and re view before use. This approach brings practitioner expertise and its supporting evidence into agent memory and reuse.

Skills and execution frameworks. Toolformer studies learned tool use, and ReAct combines reasoning with environment interaction [8, 52]. Voyager maintains executable skills, and Agent Workflow Memory derives reusable workflows from past trajectories [53, 54]. AutoGen, MetaGPT, DSPy, and AgentScope provide orchestration or program construction mechanisms [9, 55–57]. KUPAS MASTER can supply reviewed content to these systems. Our focus is the evidence and dependency requirements of the procedures passed to an executor, building on existing orchestration work.

AI in professional and scientific work. AMIE studies conversational diagnosis, and subsequent work extends conversational AI to disease management [58, 59]. Agents for scientific instruments and Co-Scientist study tool-based workflows and scientist-guided discovery, respectively [60, 61]. These studies inform evidence organization, expert assessment, and task validation in professional settings. The platform improves agents through external experience corpora accessed during inference, without requiring base-model retraining.

## 12 Conclusion

We presented KUPAS MASTER, an experience engineering platform built around nine-layer cognitive corpus construction. It turns practitioners’ tacit experience into structured assets that can be traced and reused. Six case elements, nine extraction dimensions, and six asset types connect case records to experience and callable skills. Semantic alignment, individual distillation, organizational consolidation, and cross-review re tain sources, conditions, and disagreements. Explicit inputs, steps, dependencies, and stopping conditions connect these assets to task execution and evaluation feedback.

Using authorized samples from 20 randomly selected practitioners, the platform processed 1,576 source files into 23,024 individual records and 13,113 organizational assets. Across 177 questions and 531 responses produced under a common task protocol and evaluated with a shared rubric, the base model, raw-corpus RAG, and KUPAS MASTER agent scored 70.63, 79.75, and 89.58 when practitioner means were weighted equally. The KUPAS MASTER agent improved on raw-corpus RAG by 9.83 points and gained in all seven dimensions. Run records confirm that assets and skills suppliedjudgment conditions, identified risk boundaries, and organized task procedures. These results demonstrate the efectiveness of nine-layer cognitive corpus construction, with consistent advantages in task quality, evidence use, and boundary handling. The platform turns individual tacit experience into organizational knowledge and agent capabilities through a complete engineering workflow, from experience capture to practical application.

By bringing expert judgment, practical methods, and execution boundaries into agent workflows, KUPAS MAS-TER provides a foundation for staf development, business collaboration, and professional services. Its trace able assets preserve organizational know-how and make professional experience reusable across tasks. Fu ture development will expand the asset base and applications, streamline experience capture and reuse, and strengthen sharing, review, updates, and application feedback. These advances will help organizations build lasting knowledge assets and apply AI to increasingly demanding decisions and tasks.

[1] State Council. Opinions on Deepening the Implementation of the “AI+” Initiative. Policy document 国发〔2025 11 号, State Council, 2025. In Chinese.

[2] The White House. Launching the Genesis Mission. Executive Order 14363, The White House, November 2025.

[3] Gary A Klein, Roberta Calderwood, and Donald Macgregor. Critical decision method for eliciting knowledge. IEEE Transactions on systems, man, and cybernetics, 19(3):462–472, 1989.

[4] Robert R Hofman, Beth Crandall, and Nigel Shadbolt. Use of the critical decision method to elicit expert knowledge: A case study in the methodology of cognitive task analysis. Humanfactors, 40(2):254–276, 1998.

[5] Laura G Militello and Robert JB Hutton. Applied cognitive task analysis (acta): a practitioner’s toolkit for understanding cognitive task demands. Ergonomics, 41(11):1618–1641, 1998.

[6] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[7] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

[8] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

[9] Dawei Gao, Zitao Li, Yuexiang Xie, Weirui Kuang, Liuyi Yao, Bingchen Qian, Zhijian Ma, Yue Cui, Haohao Luo, Shen Li, et al. Agentscope 1.0: A developer-centric framework for building agentic applications. arXiv preprint arXiv:2508.16279, 2025.

[10] Sarthak Jain and Byron C Wallace. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019.

[11] Matthew A Lambon Ralph, Elizabeth Jeferies, Karalyn Patterson, and Timothy T Rogers. The neural and computational bases of semantic cognition. Nature reviews neuroscience, 18(1):42–55, 2017.

[12] Kimberly L Stachenfeld, Matthew M Botvinick, and Samuel J Gershman. The hippocampus as a predictive map. Nature neuroscience, 20(11):1643–1653, 2017.

[13] Maurizio Corbetta and Gordon L Shulman. Control of goal-directed and stimulus-driven attention in the brain. Nature reviews neuroscience, 3(3):201–215, 2002.

[14] Tirin Moore and Katherine M Armstrong. Selective gating of visual signals by microstimulation of frontal cortex. Nature, 421(6921):370–373, 2003.

[15] Etienne Koechlin, Chrystele Ody, and Frédérique Kouneiher. The architecture of cognitive control in the human prefrontal cortex. Science, 302(5648):1181–1185, 2003.

[16] Sébastien Ballesta, Weikang Shi, Katherine E Conen, and Camillo Padoa-Schioppa. Values encoded in orbitofronta cortex are causally related to economic choices. Nature, 588(7838):450–453, 2020.

[17] Stephen M Fleming, Rimona S Weil, Zoltan Nagy, Raymond J Dolan, and Geraint Rees. Relating introspective accuracy to individual diferences in brain structure. Science, 329(5998):1541–1543, 2010.

[18] John G Kerns, Jonathan D Cohen, Angus W MacDonald III, Raymond Y Cho, V Andrew Stenger, and Cameron S Carter. Anterior cingulate conflict monitoring and adjustments in control. Science, 303(5660):1023–1026, 2004.

[19] Henry H Yin and Barbara J Knowlton. The role of the basal ganglia in habit formation. Nature reviews neuroscience, 7(6):464–476, 2006.

[20] Adam R Aron, Paul C Fletcher, Ed T Bullmore, Barbara J Sahakian, and Trevor W Robbins. Stop-signal inhibition disrupted by damage to right inferior frontal gyrus in humans. Nature neuroscience, 6(2):115–116, 2003.

[21] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, pages 52989–53046, 2024.

[22] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606, 2024.

[23] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.

[24] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

[25] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024.

[26] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ -bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

[27] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.

[28] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, et al. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery. In International Conference on Learning Representations, volume 2025, pages 96934–96990, 2025.

[29] Shakked Noy and Whitney Zhang. Experimental evidence on the productivity efects of generative artificial intelligence. Science, 381(6654):187–192, 2023.

[30] Michelle Vaccaro, Abdullah Almaatouq, and Thomas Malone. When combinations of humans and ai are useful: A systematic review and meta-analysis. Nature Human Behaviour, 8(12):2293–2303, 2024.

[31] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 12076–12100, 2023.

[32] Nelson F Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 7001–7025, 2023.

[33] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th conference of the european chapter of the association for computational linguistics: system demonstrations, pages 150–158, 2024.

[34] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, et al. Ragchecker: A fine-grained framework for diagnosing retrieval-augmented generation. Advances in Neural Information Processing Systems, 37:21999–22027, 2024.

[35] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

[36] Ikujiro Nonaka. A dynamic theory of organizational knowledge creation. Organization science, 5(1):14–37, 1994.

[37] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781, 2020.

[38] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th conference of the european chapter of the association for computational linguistics: main volume, pages 874–880, 2021.

[39] Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, 2024.

[40] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In International Conference on Learning Representations, volume 2024, pages 32628–32649, 2024.

[41] Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation. In EMNLP (Findings), pages 10746–10761, 2025.

[42] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025.

[43] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.

[44] Theodore R Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L Grifiths. Cognitive architectures for language agents. arXiv preprint arXiv:2309.02427, 2023.

[45] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

[46] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

[47] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

[48] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2026.

[49] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

[50] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations, volume 2026, pages 86069–86100, 2026.

[51] LangChain. LangMem: Introduction and documentation, 2025. Accessed September 12, 2026.

[52] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.

[53] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

[54] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024.

[55] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

[56] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Steven Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, volume 2024, pages 23247–23275, 2024.

[57] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Saiful Haq, Ashutosh Sharma, Thomas Joshi, Hanna Moazam, Heather Miller, et al. Dspy: Compiling declarative language model calls into state-of-the-art pipelines. In International Conference on Learning Representations, volume 2024, pages 54928–54958, 2024.

[58] Tao Tu, Mike Schaekermann, Anil Palepu, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Yong Cheng, et al. Towards conversational diagnostic artificial intelligence. Nature, 642(8067): 442–450, 2025.

[59] Valentin Liévin, Anil Palepu, Wei-Hung Weng, et al. Towards conversational artificial intelligence for disease management. Nature, 655(8125):1292–1299, 2026. doi: 10.1038/s41586-026-10764-5.

[60] Aikaterini Vriza, Michael H Prince, Tao Zhou, Henry Chan, and Mathew J Cherukara. Operating advanced scientific instruments with ai agents that learn on the job. npj Computational Materials, 12(1):160, 2026.

[61] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, et al. Accelerating scientific discovery with co-scientist. Nature, pages 1–3, 2026.

## A Team Members

The project and research teams of KUPAS MASTER are listed below.

## A.1 Project Leaders

<table><tr><td>Name</td><td>Affiliation</td><td>Title</td></tr><tr><td>Changmian Wang</td><td>KUPAS</td><td>CTO</td></tr><tr><td>Yuchao Ma</td><td>KUPAS</td><td>Director of Innovative Products; Technical Architect; Senior FDE</td></tr></table>

## A.2 Project Members

<table><tr><td>Name</td><td>Affiliation</td><td>Title</td></tr><tr><td>Xuchao Lu</td><td>KUPAS</td><td>Algorithm Expert; Senior FDE</td></tr><tr><td>Chen Zhang</td><td>KUPAS</td><td>AI Product Manager; Senior FDE</td></tr><tr><td>Ping Sun</td><td>KUPAS</td><td>AI Engineer; Senior FDE</td></tr><tr><td>Jiazheng Wang</td><td>KUPAS</td><td>AI Evaluation Expert; Senior FDE</td></tr><tr><td>Shan Wang</td><td>KUPAS</td><td>Senior FDE</td></tr></table>

## A.3 Research Leaders

<table><tr><td>Name</td><td>Affiliation</td><td>Title</td></tr><tr><td>Qinghua Zheng</td><td>Tongji University</td><td>Party Secretary of Tongji University; Academician of the Chinese Academy of Engineering</td></tr><tr><td>Xian-Sheng Hua</td><td>Tongji University</td><td>Executive Dean, Institute of AI for Engineering; Tenured Distinguished Professor</td></tr></table>

## A.4 Research Members

<table><tr><td>Name</td><td>Affiliation</td><td>Title</td></tr><tr><td>Hongzhi Li</td><td>Tongji University</td><td>Tenured Distinguished Professor; Former Principal Researcher and Chief Architect, Microsoft (USA)</td></tr><tr><td>Jianqiang Huang</td><td>Tongji University</td><td>Tenured Professor; Former Head of Alibaba City Brain</td></tr><tr><td>Kaihua Tang</td><td>Tongji University</td><td>Tenured Associate Professor; World&#x27;s Top 2% Scientists, 2024--2025</td></tr><tr><td>Ziqing Xia</td><td>Tongji University</td><td>Assistant Professor, Human Factors Expert</td></tr><tr><td>Ziyu Lu</td><td>Tongji University</td><td>PhD Student</td></tr><tr><td>Yihe Sun</td><td>Tongji University</td><td>Master&#x27;s Student</td></tr><tr><td>Xuanwen Chen</td><td>Tongji University</td><td>Assistant Research Fellow</td></tr></table>

## B Case Study: System Workflow

Figure 13 illustrates how a hypertension practitioner assistant is built and used for community follow-up. The user first defines the task, inputs, and deliverables, imports guidelines and follow-up field definitions, and records sources, versions, and missing information. Semantic alignment connects the materials. Extraction then recovers judgment criteria, action conditions, and exceptions as candidates linked to evidence. Experi ence from diferent sources is consolidated and reviewed by domain physicians, then organized into skills with explicit inputs, steps, and stopping conditions. The agent uses these skills to produce follow-up outputs with citations and clear verification steps. Evaluation fixes the questions, model, and inputs, compares configura tions, and feeds errors back into corpus, asset, or skill revisions. The XML in the figure provides follow-up field definitions for subsequent data entry.

![](images/63c11ff80f8baae915de03f1de07aba3ac69d5ee811b12af2954034bd5a8d001.jpg)  
Figure 13 Building, using, and evaluating a hypertension practitioner assistant.

## C Detailed Experimental Analysis

This appendix brings together configuration scores, asset statistics, and run records to explain the platform’s performance, experience organization, skill execution, boundary handling, domain coverage, and eficiency. Table 9 summarizes ten aspects of the analysis and their supporting evidence. All configuration comparisons follow the common evaluation protocol described in Section 8.

Table 9 Ten aspects of platform performance and supporting evidence (20 practitioners, 19 platform industry labels, 177 questions, and 531 responses).
<table><tr><td>ID</td><td>Analysis focus</td><td>Supporting evidence</td></tr><tr><td>E1</td><td>Overall improvement in task outcomes</td><td>Configuration scores, paired questions, resampling intervals, and below-threshold scores in the random authorized sample.</td></tr><tr><td>E2</td><td>Added value from structuring the same information</td><td>Overall gain of the experience-and-skill configuration built from the same sources.</td></tr><tr><td>E3</td><td>Useful, traceable experience representation</td><td>Asset fields, source relationships, automated checks, and separately recorded human review.</td></tr><tr><td>E4</td><td>Consolidation into organizational assets</td><td>Individual-to-organizational asset counts, library versions, and consolidation records.</td></tr><tr><td>E5</td><td>Skills in task execution</td><td>Skill calls, file generation, and task scores.</td></tr><tr><td>E6</td><td>Boundary handling and reliability</td><td>Boundary score gains and associations between asset use and boundary-judgment rows.</td></tr><tr><td>E7</td><td>Performance across corpus sizes</td><td>Performance across sample sizes and rank correlations for the two stages.</td></tr><tr><td>E8</td><td>Traceable updates and repeated execution</td><td>Library versions, repeated-question reports, and retrieval records across runs.</td></tr><tr><td>E9</td><td>Consistent performance across professional domains</td><td>Task performance in multiple professional settings, each using domain-specific corpora.</td></tr><tr><td>E10</td><td>Runtime cost and efficiency</td><td>End-to-end latency and percentile statistics by practitioner and configuration.</td></tr></table>

## Analysis and supporting evidence

• E1: Overall task performance. The comparison uses authorized samples from 20 randomly selected practitioners, covering 177 questions and 531 responses. Scores rise from 70.63 for A to 79.75 for B and 89.58 for C. C gains 18.96 points over A and 9.83 over B, equivalent to relative increases of 26.8% and 12.3%. All practitioner means follow $A < B < C .$ . C wins 176/177 paired question comparisons against B and 177/177 against A. Counts of responses scoring below 60 fall from 22 to 2 to 0. The 95% interval is [9.20, 10.45] for the practitioner-weighted gap of 9.83 and [9.02, 10.45] for the question-weighted gap of 9.72, using 100,000 cluster bootstrap samples with seed 20260926.

• E2: Structuring the same source information. The raw-corpus RAG and structured-asset configurations use the same uploaded materials. C adds 9.83 points (12.3%) over B, with the same direction of change across all 20 practitioners and all 7 dimensions. The raw retrieval stage contributes 9.13 points over A. The shared-source comparison demonstrates the added value of turning raw information into structured experience and executable skills.

• E3: Representation and reliability. The six libraries contain 13,113 organizational assets. Their fields include triggers, exceptions, judgment logic, evidence spans, source links, and confidence, supporting checkable citations. Across the six library types, 37 cross-review jobs cover 818 candidates, of which 486 are marked as passed (59.4%). The best-practice pass rate is 70%, and the mean of the 37 job-level scores is 79.5. Human review status is recorded independently. The ratio of organizational assets to retrievable chunks ranges from 0.02 to 11.9, reflecting diferent source forms, from tables to large case collections.

• E4: Organizational consolidation. The platform converts 23,024 published individual entries into 13,113 organizational assets through consolidation of agreement, disambiguation, and context annotation. All six libraries are present for 17/20 practitioners. Organizational libraries are rebuilt with version identifiers and enter the shared index of 43,882 Elasticsearch documents. Entries retain a consolidation method field, making each asset’s processing history traceable.

• E5: Skills and task execution. Skills organize extracted assets into executable procedures. Among 217 three-configuration comparison reports, C calls skills in 166 (76.5%) and generates files in 116 (53.5%); A and B make no skill calls. In the 53 C runs with detailed logs, 46 call skills and 46 retrieve rules. C gains 14.38 points in appropriate tool and skill use and 9.89 in output actionability over B. The records show experience being used to produce deliverables as well as advice.

• E6: Boundary assets and reliability. The boundaries and compliance score increases by 9.03 points from B to C, about 1.9 times the 4.67-point gain from A to B. For 19 of 20 practitioners, the C-minus-B gain on this dimension exceeds the B-minus-A gain. Reports that use corner cases, constraints, or negative examples have 95 C-only wins in 99 boundary-judgment rows, compared with 94/127 (74.0%) in reports without those calls. Boundary rows are identified by judgment-item names mentioning boundaries, risk, or applicability; library use is counted at the report level. Run records show how these assets supply concrete checks, such as verifying the grounds for judging a transfer unlawful and checking compliance evidence beyond proof of delivery.

• E7: Data scale and usefulness. Small, focused collections deliver strong performance. The community Party secretary has 7 files and 36 assets, with a composite score of 91.36, ranking second. The workplace injury mediator has 2 interviews and 167 assets, scoring 90.58. The Spearman correlation between asset count and the C-minus-B gain is only 0.07. Raw chunk count correlates with the B-minus-A gain at 0.61 (p ≈ 0.004), suggesting diferent relationships at the two stages. All 20 practitioners improve across corpora spanning two orders of magnitude.

• E8: Updates and version management. The libraries support batch rebuilding and retain library-level versions. Current versions for 13 practitioners share a timestamp prefix. The organizational libraries are updated as their corpora evolve. In 19 repeated runs of an accountant’s equity-method investment in come question, every comparison row includes C among the winners. A wealth-adviser question retrieves between 1 and 5 library types, including the raw corpus, across runs. Version records make experience updates traceable, while repeated runs document how the agent retrieves and applies experience across executions.

• E9: Consistent gains across industries. The sample covers 19 platform industry labels. All 20 practitioners improve in all 7 dimensions, and C exceeds B on 176/177 questions. Applications include firefighting, energy, trafic policing, finance and accounting, community governance, mediation, ports, economic crime investigation, rail transit, emergency management, social insurance funds, auditing, healthcare, environmental management, and construction. The platform consistently improves task performance across these professional settings by turning their varied source materials into usable domain experience.

• E10: Runtime cost and eficiency. Practitioner-weighted mean latencies are 16.2/34.0/57.1 s for A/B/C. The corresponding means of practitioner P90 values are 21.1/41.1/68.4 s. Ratios of unrounded mean times put C at 3.53× A and 1.68× B. C’s practitioner means range from 43.6 to 78.3 s, and the mean of practitioner-level 95th-percentile (P95) latencies is 71.6 s. Latency is positively associated with the number of libraries retrieved, skill calls, and file generation. The platform combines these response times with higher accuracy and more complete outputs for professional analysis, planning, and review.

## D Additional Results and Plots

This appendix visualizes the evaluation data in more detail. Score distributions and dificulty groups weight the 177 questions equally. Practitioner-level composite scores, dimension means, and mean latency weight the 20 practitioners equally. The captions identify the aggregation used.

Figure 14 shows the distributions shifting toward higher scores. Question-weighted means are 70.9, 79.9, and 89.6 for A, B, and C. At a pass threshold of 60, pass rates rise from 87.57% to 98.87% to 100.00%. The fractions scoring at least 80 are 7.3%, 58.8%, and 99.4%. C reduces low-scoring answers and brings nearly all answers above 80.

In Figure 15, each point pairs a practitioner’s composite score with mean end-to-end time for one configura tion. Equal weighting across practitioners gives A/B/C scores of 70.6/79.8/89.6 and times of 16.2/34.0/57.1 seconds; the legend rounds time to 16/34/57 seconds. Ratios calculated from unrounded means give C/A

![](images/168d747459c182e0d750f34be114bba4594726fd5480b7096d6a089658fdcf8b.jpg)  
Figure 14 Distribution of 531 scores for 177 questions, with 177 scores per configuration. $\mathrm { A / B / C }$ question-weighted means are $7 0 . 9 / 7 9 . 9 / 8 9 . 6 .$ pass rates are 87.57%/98.87%/100.00%, and fractions scoring 80 are 7.3%/58.8%/99.4%. Counts below the score threshold are $2 2 / 2 / 0$

of 3.53 and C/B of 1.68. C achieves a score of 89.6 at a mean time of 57.1 seconds, delivering high-quality professional analysis within a practical response window.  
![](images/0bd813c22cee2b9dec9e47586dc1515b3c92acf7571d8e9debed135941dbe6e2.jpg)  
Figure 15 Latency and score, with one point per practitioner and configuration. Mean $\mathrm { A / B / C }$ times are $1 6 . 2 / 3 4 . 0 / 5 7 . 1$ s, rounded to $1 6 / 3 4 / 5 7$ s in the legend. Ratios of mean times are $C / A = 3 . 5 3$ and $C / B = 1 . 6 8$

Figure 16 counts 13,113 organizational assets across the 20 practitioners. Rules and best practices account for 49.1%; the remaining assets supply skills, constraints, failure experience, and unusual situations. Counts and type distributions vary substantially. Seventeen practitioners have all six types, showing how the same structure accommodates diferent professional settings.

![](images/89b6cc368aa49a2641871bfe4ff102d6108e03dd9245fffdd4bb6db41c0e21d6.jpg)  
Organizational assets (six libraries, log scale)  
Figure 16 The six libraries contain 13,113 assets: 3,550 rules, 2,892 best practices, 1,955 skills, 1,948 corner cases, 1,623 constraints, and 1,145 negative examples. All six types are present for 17/20 practitioners. The cumulative horizontal positions use a linear scale up to 200 assets and a logarithmic scale above 200; segment widths therefore do not represent type proportions.

Figure 17 shows 1,576 source files in 13 formats, with 2 to 378 files per practitioner. Formats and storage sizes vary widely. The platform organizes documents, tables, structured data, and multimedia while preserving source relationships for extraction and consolidation.

The left panel of Figure 18 uses 62 reports with four standard judgment categories. Each category has 123 comparison rows, and C wins every row. Wins include ties, so a row can count for more than one configuration. The right panel uses all 217 retained reports and 1,345 comparison rows: C wins alone on 1,109 rows (82.5%) and ties on 214 (15.9%); B wins alone on 11 (0.8%), and A-only wins or other outcomes account for 11 (0.8%). C calls skills in 166 reports and generates files in 116. These report-level comparisons supplement the 177-question scored evaluation.

Figure 19 shows a weak relationship between organizational asset count and the C-minus-B gain across 20 practitioners, with Spearman correlation 0.07. Raw chunk count has a positive correlation of 0.61 with the Bminus-A gain. The relationship between scale and score gain therefore difers between stages, with experience organization and use adding value beyond raw retrieval.

Figure 20 groups the questions by their recorded dificulty labels: 8 easy, 83 medium, and 86 hard. C’s mean scores are 90.8, 89.3, and 89.8, above both baselines in every group. Its gains over B are about 8.7, 10.0, and 9.5 points. The figure reports the size of each group, including 8 easy questions.

![](images/ce9da6571897b45c02246895beb155bc32913d42334bdd7bca908d40da69af4c.jpg)

![](images/c5c4b117219089c4f2f704412dae4f68029b2df3d31f010ecfddb0cdaa07700f.jpg)  
Figure 17 The uploaded corpus contains 1,576 files, 1.73 GB, and 13 formats. Individual file counts range from 2 to 378, covering documents, spreadsheets, structured data, images, and recordings. MB labels are rounded to whole decimal megabytes; 0 MB denotes less than 0.5 MB, not an empty corpus.  
Figure 18 Judgments in three-configuration comparison reports. Left: wins including ties across 123 rows per category in 62 reports. Right: outcomes across 217 retained reports and 1,345 rows, with 1,109 C-only wins (82.5%) and 214 C ties (15.9%).

![](images/a8799381681323cca275182a6baf28bd8999cff1798cb5cdcf858888752c1003.jpg)

![](images/123536eba2c5a9d05b726c23a82e3e876905dab274bf68e51f7e702c89002b80.jpg)  
Figure 19 Scale and score gains (n = 20). Left: the Spearman correlation between organizational asset count and $C - B$ is 0.07. Right: the correlation between raw chunk count and $B - A$ is 0.61.

![](images/7284f2de19a7236ae4e32654bad87d3ed5af5892f5975d8e87d942b3e20e69cd.jpg)  
Figure 20 Question-weighted composite scores by dificulty. $\mathrm { A } / \mathrm { B } / \mathrm { C }$ means are $7 5 . 5 / 8 2 . 1 / 9 0 . 8$ for easy questions $( n = 8 )$ $6 9 . 0 / 7 9 . 3 / 8 9 . 3$ for medium questions $( n = 8 3 )$ , and 72.3/80.3/89.8 for hard questions $( n = 8 6 )$ . C exceeds both baselines and scores above 89 in all groups.