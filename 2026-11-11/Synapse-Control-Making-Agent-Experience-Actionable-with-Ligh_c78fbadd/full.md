# Synapse-Control: Making Agent Experience Actionable with Lightweight Runtime Control

Zhiyuan Huang   
hzy364@sjtu.edu.cn   
Shanghai Jiao Tong University   
Shanghai, China   
Ant Group   
Hangzhou, China   
Song Zhang   
gepin.zs@antgroup.com   
Ant Group   
Hangzhou, China   
Weiquan Huang   
hwqqz678@sjtu.edu.cn   
Shanghai Jiao Tong University   
Shanghai, China   
Ant Group   
Hangzhou, China Shengan Zheng<sup>\*</sup> shengan@sjtu.edu.cn   
MoE Key Lab of Artificial Intelligence,   
AI Institute, Shanghai Jiao Tong University Shanghai, China   
Lingyun Li   
yunju.lly@antgroup.com   
Ant Group   
Hangzhou, China

Linpeng Huang \* lphuang@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

<sup>\*</sup>Corresponding authors.

Chenghu Yang   
yexiang.ych@antgroup.com   
Ant Group   
Hangzhou, China   
Chen Yu   
yingjun.yc@antgroup.com   
Ant Group   
Hangzhou, China

## Abstract

Large language model agents reuse execution experience to improve reliability on complex tool-based tasks. Systems such as Sentry call large models to generate failure diagnoses before retrieving advice, adding inference cost to experience reuse. We present Synapse-Control, an agent runtime whose lightweight control plane selects advice and determines when to send it to the task-executing model. Rules shortlist relevant advice, and a lightweight 1.7B classifier fine-tuned on teacher-labeled state–suggestion pairs scores those candidates instead of generating a diagnosis. Micro-contracts encode reusable interface checks, such as required arguments or valid options, without prescribing complete task programs. A Prompt Value Table records earlier guidance and feedback to limit repeated suggestions through cooldown and token-budget checks. Across 1,308 attempts per method in AppWorld and WebShop, Synapse-Control achieves 270 successes (29.8% more than Base), replaces 4,987 generative diagnosis calls, and reduces auxiliary control time relative to Sentry by 50.2% and 51.0%, respectively.

## 1 Introduction

Reusing execution experience helps large language model agents perform complex tool-based tasks without repeatedly reconstructing procedures and interface requirements. An agent alternates between reasoning, tool actions, and observations; its action-generating model, the actor, supplies arguments, interprets returned state, and determines completion. Existing systems retain skills, procedural insights, and lessons for subsequent tasks [1–4]. Using this knowledge to correct a failed action or stalled execution is called runtime recovery. We focus on locally served agents that repeatedly use stable APIs or web interfaces, where both reliability and the cost of experience reuse matter.

Deciding when and how to reuse experience can itself require costly model inference. Sentry examines the recent trajectory, generates a failure diagnosis, retrieves matching lessons, and verifies subsequent execution before publishing new experience [4]. Whenever this diagnosis path is triggered, it generates intermediate text before the actor can use an existing lesson. That work can be disproportionate to the correction needed, such as supplying a required argument. Retrieved advice can also repeat a completed check or conflict with the current state. Reducing this overhead requires both cheaper selection and a check that the advice is still useful.

Our central insight is a computational asymmetry between generating a diagnosis andjudging a supplied suggestion. Diagnosis generation constructs an explanation through successive token predictions; classification assigns scores to fixed labels for an already retrieved candidate. Synapse-Control exploits this asymmetry by using execution evidence and keyword matches to retrieve at most five suggestions, then learning only their suitability in the current state. The 1.7B sidecar thus solves a diferent computational problem from the 9B detector: direct assessment replaces diagnosis generation. Selection and delivery form the control plane, while the 9B actor retains planning and action generation.

Task-specific training makes this decomposition operational. We fine-tune the complete 1.7B backbone and classification head on target-actor states collected with and without earlier guidance. A 32B teacher labels the exact suggestion available at each state as useful, unnecessary, or conflicting. The optimization objective concentrates adaptation on these recurring state–constraint distinctions. The relevant requirement becomes discrimination at this decision boundary, rather than the breadth of a general response generator.

We represent reusable interface checks as micro-contracts: short statements of local constraints and when they apply. They specify facts such as required arguments or selected options while leaving task values and program composition to the actor. Rules check for an unmet need and shortlist relevant entries; the classifier assesses the resulting suggestions. Selected guidance is added to the actor’s next-action prompt, which we call delivery; the resulting actions and observations remain in its execution history.

We use a Prompt Value Table (PVT) to record each suggestion’s delivery history and feedback within a task. It withholds advice while feedback is pending, evidence is unchanged, or a cooldown is active. Earlier feedback adjusts the ranking of eligible suggestions; token-budget checks and a single-hintper-step limit constrain what reaches the actor. These controls suppress repeated guidance when execution stalls.

We evaluate Synapse-Control on AppWorld API programming and WebShop shopping tasks using the same actor and task workload for all primary methods. Across 1,308 attempts per method, it achieves 270 successes, compared with 256 for Sentry and 208 for Base, with the lowest cumulative task time among these methods. Auxiliary control time decreases by 50.2% in AppWorld and 51.0% in WebShop. Controlled studies examine knowledge representation, candidate selection, and delivery history; the history ablation reduces repeated deliveries by approximately two thirds in both benchmarks.

Our contributions are:

• A decomposition of experience selection into rule-based retrieval and specialized candidate scoring, replacing 9B diagnosis generation with a fine-tuned 1.7B classifier.

• A runtime combining micro-contracts, rule-based filtering, and delivery-history and budget management.

• A two-benchmark evaluation of task success, benchmark score, and complete execution cost, supported by controlled component studies.

## 2 Background and Motivation

Experience reuse extends an agent’s execution loop with knowledge from earlier interactions; its eficiency depends on both knowledge construction and runtime use. The main costs arise when the runtime diagnoses a failure, selects advice, and processes the resulting intervention.

## 2.1 Experience and runtime recovery

Execution experience allows an agent to reuse information about tool behavior and earlier outcomes across steps and tasks. An interactive agent selects an action, executes it in the environment, and incorporates the returned observation into its next decision. Actions can be API calls expressed as code or operations on a webpage, while observations reveal returned values, invalid arguments, unavailable actions, or a lack of progress. ReAct and CodeAct illustrate this reasoning– action loop for tools and executable code [5, 6]. Retaining the procedures and checks learned from such interactions can reduce repeated exploration in later executions.

Runtime recovery uses corrective guidance to restore valid or productive execution after an error or a stall. Stored interface checks, conditional instructions, and verified lessons supply this guidance. For example, an API constraint can identify a missing argument, while a previous recovery lesson can suggest a check after repeated unsuccessful reads. The agent still generates and executes the next action; the recovery mechanism supplies information for that decision.

Experience systems construct reusable knowledge in several forms. Voyager stores executable skills, ExpeL extracts insights from trajectories, Agent Workflow Memory retains procedures, and ACE maintains an evolving playbook [1–3, 7]. These representations range from complete programs to conditional instructions. Their common runtime requirement is to connect stored knowledge to the current execution state before using it.

Sentry organizes recovery around a diagnosis–guidance– verification lifecycle [4]. As shown in Figure 1, invalid actions enter a hard-repair path that handles execution errors directly; for other failures, a large-model monitor examines the recent trajectory and generates a diagnosis containing retrieval labels and local evidence. The runtime then retrieves matching playbook entries and constructs guidance for the actor. A verifier checks the subsequent execution, and verified recovery produces a new reusable lesson. This design already includes intervention-need checking and an external experience store; the remaining question is the cost and precision of the decisions connecting them.

![](images/07db31a7a3a190e6b8c449ac589ca04e92b2e9607ee962dd6b5db75447b810db.jpg)  
Figure 1: Sentry’s recovery workflow. Solid arrows show the guidance path; the dashed arrow publishes verified lessons.

## 2.2 The cost of diagnosis and verification

Sentry makes auxiliary model calls to diagnose failures, verify corrections, and generate lessons. The actor additionally processes the delivered guidance and may follow a diferent action sequence, so complete task time captures both control work and the resulting execution. Separating these quantities identifies which overhead can be reduced by changing the control mechanism.

![](images/439702263854fd952231be86c5e19bf8e5c498e990d62c1c5d144f3d8094a492.jpg)  
Figure 2: Recovery requires both economical control and efective guidance. (a) Sentry’s actor request time is normalized to 100% within each benchmark; stacked segments show additional control work. (b) In the AppWorld development probe (342 attempts per condition), a larger actor-output cap increases Base completions but does not improve Sentry completions.

Sentry’s measured control workload is substantial relative to actor inference in both environments (Figure 2a). Auxiliary requests consume time equivalent to 21.4% of actor request time in AppWorld and 18.2% in WebShop. Actor request time averages 47.29 and 47.85 seconds per attempt, with 10.10 and 8.72 auxiliary seconds, respectively. Generated diagnosis accounts for 5.74 and 5.21 auxiliary seconds; verification and lesson updates account for the remainder. Normalizing to actor request time exposes the scale of repeated control work within each environment.

Repeated diagnosis is a concrete optimization target when relevant knowledge is already available. A runtime with a small candidate pool needs to assess specific suggestions against visible evidence, yet generative diagnosis first produces an intermediate explanation and retrieval labels. Assessing the candidate directly can reduce control work while preserving verification and lesson publication.

## 2.3 State-dependent suitability of retrieved knowledge

Topical relevance and present suitability answer diferent questions about a retrieved knowledge item. Relevance asks whether the item concerns the same tool, operation, or task topic; suitability asks whether following its concrete advice addresses an unmet need without conflicting with the current state. Lexical matching and interface identity support the former, whereas the latter requires the visible action, observed values, and checks already completed.

Visible execution state can change suitability even when the task topic remains unchanged. In a shopping task, advice to change an option may match the product vocabulary but be unnecessary after the requested color has already been selected. A generic instruction to search again can likewise redirect an otherwise ready purchase. In API programming, an application-name match alone cannot distinguish a missing authentication fact from a pagination or argument constraint. These cases require evaluating the proposed content against the current evidence, rather than treating retrieval as authorization to intervene.

Knowledge granularity and delivery history further determine how retrieved advice afects execution. A required argument or visible-state check can be represented as a compact fact without prescribing an entire task program. Such a fact still needs an applicability condition, and previously delivered advice may remain unresolved or redundant. The runtime must therefore combine content assessment with a separate decision about repeated presentation.

Additional actor-generation capacity does not by itself ensure more efective recovery guidance. In a separate AppWorld development probe (342 attempts per condition), increasing the actor-output cap from 384 to 768 tokens raises Base completions from 30 to 44, while Sentry completions decrease from 38 to 36 (Figure 2b; Appendix A). This response motivates evaluating guidance at the state and candidate level rather than relying on a larger actor-output budget to resolve control errors. The resulting design requirement is to turn stored knowledge into timely, actionable guidance without repeatedly generating an intermediate diagnosis.

## 3 The Synapse-Control Architecture

Synapse-Control selects advice through a small classifier, represents interface checks as micro-contracts, and manages repeated use through PVT. Figure 3 connects these components to the actor and the existing recovery host, which performs verification and lesson updates.

The bank stores reusable entries, including micro-contracts and verified lessons. Combining an entry with current evidence produces the candidate payload: the exact suggestion the classifier evaluates. If selected and permitted by PVT, this text enters the next-action prompt as a soft hint, without changing the actor’s action interface.

The execution plane generates task actions, while the control plane evaluates candidate payloads and governs soft-hint delivery. The classifier replaces the generative detector; host verification and lesson updates remain active. Let $s _ { t } = \langle g , w _ { t } , e _ { t } \rangle$ denote the task objective, recent visible execution window, and evidence derived from that window. Deterministic rules extract this evidence from actions and observations: a rejected API call, unchanged reads, or a mismatch between requested and selected options. State construction requires no additional model-generated diagnosis.

## 3.1 Learning to assess candidate guidance

The architecture replaces a variable-length generative decision with fixed-output semantic classification. For each candidate, a forward pass produces three label scores without autoregressively decoding a diagnosis. At most five candidates and 1,024 tokens per input bound scoring work by configuration rather than knowledge-bank size. All backbone parameters and the classification head are fine-tuned on the same state–suggestion objective, concentrating adaptation on distinctions needed by the target actor. On held-out Contract candidates, the specialized scorer achieves 94.0% Helpful admission precision in AppWorld and 98.15% in WebShop; Section 5.4 reports coverage and downstream execution costs.

State–candidate classification. For state $s _ { t }$ and candidate c with its evidence-bound payload, the classifier estimates

$$
p _ { \theta } ( y \mid s _ { t } , c ) , \quad y \in \{ { \mathrm { H e l p f u l } } , { \mathrm { I r r e l e v a n t } } , { \mathrm { H a r m f u l } } \} . \quad ( 1 )
$$

Helpful denotes advice useful in the visible state; Irrelevant includes unnecessary advice; Harmful denotes conflicting or

disruptive advice. A candidate passes the learned eligibility check when its calibrated Helpful probability exceeds the fixed operating threshold. Its priority is

$$
v _ { t } ( c ) = p _ { \theta } ( { \mathrm { H e l p f u l } } \mid s _ { t } , c ) - p _ { \theta } ( { \mathrm { H a r m f u l } } \mid s _ { t } , c ) .\tag{2}
$$

An explicit Harmful class gives conflicting advice its own supervision and ranking penalty. Equation (2) is the expected label utility under values $+ 1 , 0 , - 1$ for Helpful, Irrelevant, and Harmful: irrelevant advice adds no utility, while harmful advice reduces it. Ranking follows eligibility; identical diference scores do not admit two low-confidence candidates.

Sampling target-actor states. Training combines unassisted and recovery-assisted trajectories to cover states reached by the target actor under both policies. All trajectories come from oficial training scenarios: Base contributes unassisted states, while recovery runs contribute states afected by earlier guidance. This supervised policy mixture follows the distributional motivation for dataset aggregation [8], since interventions change the states encountered later.

State sampling covers failure or stagnation, completion boundaries, and ordinary progress so that the classifier encounters both unmet needs and situations requiring no advice. These are state categories rather than assigned usefulness labels: a failure can lack suitable guidance, while a progressing task can make a suggestion redundant. Complete candidate groups are retained under a pair budget, preserving the alternatives at each sampled decision.

Reconstructing available suggestions. Reconstruction uses the online candidate builder and recorded publication order to reproduce the knowledge available at each decision. For initial bank $\kappa _ { 0 }$ and lesson publication event $\tau ( c )$ , the available pool is

$$
\begin{array} { r } { \mathcal { K } _ { t } = \mathcal { K } _ { 0 } \cup \{ c : \tau ( c ) \prec t \} . } \end{array}\tag{3}
$$

![](images/217a5b584ef8b4a2c8cddc97c108f789b0bce15301f3a69738eb22f2cc5dae5a.jpg)  
Figure 3: Synapse-Control overview. Rules filter and prepare advice (§3.2), the classifier scores it (§3.1), and PVT decides when to send it $( \ S 3 . 3 )$ . Host verification and lesson updates remain active; hard repair is unchanged.

Here $\tau ( c )$ is the verified publication event of lesson $c , \ \prec$ is logical order within the replay stream, and t denotes the current decision event. Only previously published lessons are available. The renderer $\mathcal { R } ( s _ { t } , c )$ binds current evidence to candidate content, producing the exact payload used for both teacher labeling and online delivery:

$$
G _ { t } = \{ ( s _ { t } , \mathcal { R } ( s _ { t } , c ) , y _ { t , c } ) : c \in \operatorname { A d m i t } ( s _ { t } , { \mathcal { K } } _ { t } ) \} .\tag{4}
$$

The 32B teacher evaluates each state with the exact proposed suggestion, using visible values to distinguish useful checks from redundant or conflicting advice. Figure 4 summarizes the training process.

The training objective weights the three usefulness classes to account for their diferent frequencies:

$$
\mathcal { L } ( \theta ) = - \sum _ { i = 1 } ^ { N _ { \mathrm { t r a i n } } } w _ { y _ { i } } \log p _ { \theta } ( y _ { i } \mid s _ { i } , c _ { i } ) , \qquad w _ { y } \propto n _ { y } ^ { - 1 / 2 } .\tag{5}
$$

$N _ { \mathrm { t r a i n } }$ is the training-pair count, and $n _ { y }$ is the number of training labels of class y. Class weights use inverse squareroot frequency. Calibration selects a temperature by negative log-likelihood on a held-out partition [9]; the deployment admission threshold is fixed at 0.90. The actor-state mixture and exact-payload labels focus learning on the judgments exercised by the runtime. Training and deployment configurations are specified in Section 4 and Appendix A.

## 3.2 Micro-contracts and candidate filtering

Micro-contracts preserve the actor’s planning autonomy by specifying what must hold without prescribing how the task must be solved. Executable skills and reusable workflows encode an action structure [1, 3]; reusing such procedures requires reconciling their ordering and state assumptions with the current task. Micro-contracts reduce this coupling by exposing local interface constraints that the actor can satisfy within its own plan. Argument synthesis, variable bindings, and workflow composition remain with the actor. The control plane supplies checks at the interface boundary instead of substituting an external program for task planning. Rules match applicability conditions against execution evidence to shortlist suggestions.

Contract contents. An API micro-contract records the facts needed to check a local call:

$$
\mu = \langle f , \mathcal { K } _ { \mathrm { r e q } } , \mathcal { T } _ { \mathrm { a r g s } } , D _ { \mathrm { r e t u r n } } \rangle ,\tag{6}
$$

where the fields identify the callable, required argument names, documented types and constraints, and documented return information. A candidate combines these facts with an applicability condition. WebShop uses the same interface for visible actions, selected options, and purchase readiness.

Knowledge assets have three sources: authored generic checks, structured interface documentation, and verified online lessons. In AppWorld, the builder reads parameters and return conventions from public API documentation and uses interfaces exercised in oficial training trajectories to construct reusable entries. At runtime, the builder binds visible task values to produce each candidate payload. In WebShop, generic assets cover option state, empty search, visible actions, evidence checks, and purchase readiness; the adapter supplies current page state.

Domain-level authoring defines generic checks and applicability conditions, while task-level binding is automated: WebShop uses five asset families, and AppWorld uses generic assets plus documentation mappings. The authors estimate a few person-hours to define and organize one domain’s templates using the existing adapters and builders. The builder parses documented parameter and return fields, while the adapter binds visible options and local evidence. Delivery and lesson updates run automatically during evaluation, without task-specific rule authoring. Generic entries are maintained with the environment interface version; the actor still supplies task values and composes its program. Figure 5 contrasts a general principle with an actionable check.

Automatic contract extraction. The existing documentation builder provides a starting point for automating domain onboarding. For an OpenAPI/Swagger interface, a static extractor can map the operation and parameter schemas to $f ,$ $\kappa _ { \mathrm { r e q } } ,$ and $\mathcal { T } _ { \mathrm { a r g s } } .$ , and preserve documented response information in $D _ { \mathrm { r e t u r n } }$ . For a Python-facing API, the adapter must also map the documented operation to the exposed callable. These fields can be copied from the versioned specification rather than authored for individual tasks.

![](images/eec34dcc99d390e9eb4c29b386e6120bdc6976c2bb7d311c23ea95a6c0f36b15.jpg)  
Figure 4: Aligned supervision: collect target-actor states, reconstruct available candidate groups, label the exact payload, and fit the small classifier. Scenario or product-group partitions separate training, calibration, and classifier testing.

![](images/61b21585774a8bbb55eae358c16168a8486efe1b10e1e97adea1916942c58b63.jpg)  
Figure 5: Illustrative rendering of the implemented WebShop option-state asset. Both views use the same visible evidence; Contract adds the concrete check.

An ofline instruction-tuned 32B or 72B model could supplement that parser by converting natural-language descriptions into proposed applicability conditions and concise checks. Each proposed condition would retain its source passage for review. A validation pass would check the output schema, resolve endpoint and parameter references, and compare required fields, types, and enumerated values with the source specification. Schema validation checks structure; a constraint inferred from prose also needs support in the cited documentation before admission. Accepted skeletons would enter the same versioned bank and receive current values from the runtime adapter. This is an extension of the construction workflow, with model inference confined to onboarding or interface updates; the experiments reported here use the documented builder and generic templates.

Rule filtering and learned scoring. The frontend uses execution evidence and lexical relevance to retrieve at most five candidates. This inexpensive step supplies the alternatives; the specialized classifier only scores their concrete advice against the current state. The candidate limit bounds scoring work independently of bank size.

A separate rule checks whether execution shows an unmet need before invoking the classifier:

$$
\Gamma ( s _ { t } ) = { \bf 1 } [ \mathrm { e r r o r } \lor \mathrm { r e p e a t } \lor \mathrm { s t a l l } \lor \mathrm { g a p } ] .\tag{7}
$$

Under GPU colocation, every sidecar forward pass consumes device time and memory bandwidth shared with the actor. Unconditional invocation therefore inserts additional GPU work between task actions, even for a small model. $\Gamma ( s _ { t } )$ acts as a deterministic circuit breaker at the model-invocation boundary: once the trigger flags are extracted, an O(1) Boolean decision admits or skips classification. When the gate is closed, it issues zero classifier calls; local evidence checks remain active. This makes neural control demand-driven and avoids unnecessary inference on the actor’s execution path.

The terms refer to rejected actions, repeated action– observation pairs, unchanged documented reads, and visible information or commitment gaps. AppWorld checks three consecutive identical action–observation steps, or the same set of known GET interfaces with unchanged observations; unknown efects and writes do not use the GET relaxation. WebShop checks concrete option mismatches, empty results, unavailable actions, and pre-purchase gaps. Passing the gate enables scoring; candidate suitability and delivery eligibility are still checked. Appendix A specifies the runtime sequence.

Transient guidance and persistent execution state. Each soft hint is exposed for the next actor action, limiting repeated external text while preserving the resulting execution state. Actions and observations enter normal task history, and AppWorld retains its Python session. The actor can therefore reuse code and observations produced after the hint without repeatedly receiving the same reminder. The knowledge bank and PVT retain the metadata needed to reconsider the asset after a later recurrence.

## 3.3 Delivery governance with PVT

To prevent repeated candidate acceptance from producing uncontrolled repeated guidance, we introduce a task-local Prompt Value Table (PVT). PVT regulates delivery using feedback, evidence changes, cooldown, and a shared budget, separating a knowledge asset’s availability from its current delivery eligibility.

Delivery eligibility and state records. Each asset has a task-local record containing its lifecycle state, last delivery and feedback times, evidence key, recovery episode, pending feedback, and utility. An available asset remains ineligible while feedback is unresolved, evidence is unchanged, cooldown is active, or its rendered payload exceeds the remaining budget.

Let $A _ { \theta } ( s _ { t } , c )$ be frozen learned eligibility: event triggers require $p _ { \theta } ( \mathrm { H e l p f u l } \mid s _ { t } , c ) > 0 . 9 0 \colon$ periodic triggers additionally require p<sub>θ</sub>(Harmful $\mid s _ { t } , c ) < 0 . 0 5$ and visible evidence. Let $B _ { t }$ be the remaining guidance budget and ℓ(c) the rendered actor-token cost. Then

$$
\begin{array} { r l } & { \mathcal { E } _ { t } = \{ c \in \mathrm { A d m i t } ( s _ { t } , { \cal K } _ { t } ) : A _ { \theta } ( s _ { t } , c ) , \ell ( c ) \le B _ { t } , } \\ & { \quad \quad \quad \quad \quad \quad \mathrm { P V T e l i g i b l e } _ { t } ( c ) \} . } \end{array}
$$

Here $\mathrm { P V T e l i g i b l e } _ { t }$ checks lifecycle, feedback, cooldown, and evidence/episode keys. The decision is

$$
c _ { t } = \left\{ \begin{array} { l l } { \arg \operatorname* { m a x } _ { c \in \mathcal { E } _ { t } } [ v _ { t } ( c ) + \lambda u _ { t } ( c ) ] , } & { \Gamma ( s _ { t } ) = 1 , \mathcal { E } _ { t } \neq \emptyset , } \\ { \bot , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

Eligibility requires $\ell ( c ) \leq B _ { t }$ . Delivery subtracts this cost and marks feedback pending; abstention leaves the delivery state unchanged. A new evidence/episode pair can permit reuse after cooldown. Additional hard references consume the same task-level budget.

Feedback-driven utility updates. PVT updates historical utility from the host verifier’s assessment of local recovery, using execution evidence rather than benchmark rewards. Let $q _ { t } ( c )$ indicate a defined, enabled utility observation. Observed progress gives $r _ { t } = + 1$ ; regression or adoption without progress gives $r _ { t } = - 1$

$$
u _ { t + 1 } ( c ) = \left\{ \begin{array} { l l } { ( 1 - \alpha ) u _ { t } ( c ) + \alpha r _ { t } ( c ) , } & { q _ { t } ( c ) = 1 , } \\ { u _ { t } ( c ) , } & { q _ { t } ( c ) = 0 . } \end{array} \right.\tag{9}
$$

Unknown outcomes, ignored advice, and missing adoption evidence leave utility unchanged. Feedback can still advance lifecycle metadata and end a pending episode, separating delivery-state progress from the availability of a utility signal.

Budget and utility invariants. The controller provides three directly checkable properties, proved in Appendix E. Let $\ell ( c _ { t } )$ be the token cost of delivered soft guidance, with $\ell ( \perp ) = 0 .$ and let $h _ { t } \geq 0$ be the token cost of any hard-repair reference added at step t. First,

$$
\sum _ { t = 1 } ^ { T } [ \ell ( c _ { t } ) + h _ { t } ] \leq B _ { 0 } .\tag{10}
$$

Second, unresolved feedback leaves utility unchanged. Third, starting from $u _ { 0 } = 0$ , updates preserve $u _ { t } ( c ) \in [ - 1 , 1 ]$ . Thus $| \lambda u _ { t } ( c ) | \le \lambda$ . At the evaluated $\lambda = 0 . 1$ , each adjustment is at most 0.1, and the perturbation of any pairwise priority gap is at most 0.2. History reorders the high-confidence pool; suitability, cooldown, and budget eligibility are checked first, so historical success cannot bypass them.

## 4 Implementation

Each worker colocates the actor and classifier, constructs bounded state inputs, and routes verification and lesson updates through the existing recovery host.

Serving and state construction. Each evaluation worker colocates the Qwen3.5-9B actor service and Qwen3-1.7B classifier on one H20 GPU. Tensor parallelism is one; the actor service admits one active sequence and starts with memory utilization 0.65. The implementation replaces a generative request on the control path; it does not require another GPU for each actor.

Bounded state construction preserves the information needed for a state–candidate judgment within a fixed classifier input. The controller retains at most five recent execution steps, including bounded observation values and entity samples. It preserves the objective, current action, and complete candidate payload. Older steps are removed first, followed by bounded observation compaction. If the required state and payload do not fit, the candidate is skipped. Four limits govern diferent resources: 1,024 tokens per classifier input; 2,048 added guidance tokens per task; 768/384 actor-output tokens in AppWorld/WebShop; and a 40,960-token actor-service context.

Learning and delivery settings. Domain-specific finetuning converts Qwen3-32B teacher judgments into a frozen 1.7B suitability classifier. We train the full backbone and classification head with AdamW, BF16 parameters, and FP32 loss logits. The two domain classifiers are trained independently from the same initial checkpoint, using 456 optimizer updates for AppWorld and 450 for WebShop. The reset threeclass checkpoints are selected on development data and frozen before final evaluation. Calibrated Helpful probability must exceed 0.90. PVT uses $\alpha = 0 . 5 , \lambda = 0 . 1$ , and a three-step cooldown. Deterministic item identifiers break ranking ties. Training splits, hyperparameters, teacher prompts, and realized sampling distributions appear in Appendix A.

Ofline construction and online reuse. Ofline preparation produces a domain controller reused across tasks without additional labeling or fine-tuning. Artifact filesystem timestamps give an approximately 13-minute AppWorld training batch: two heads run concurrently on two H20 GPUs, each with 456 optimizer updates, with loading, evaluation, and checkpoint persistence included. WebShop’s 450-update domain model takes approximately 15 minutes for training, evaluation, and persistence.

WebShop’s 3,000-pair labeling stage takes approximately 38.5 minutes. Data assembly produces 2,998 labeled examples: 2,398 for training and 300 each for calibration and testing. The complete chain from labeling launch to checkpoint persistence takes approximately 54 minutes. Appendix A specifies the stage boundaries and timing scope.

The resulting checkpoints remain frozen across 1,008 AppWorld and 300 WebShop evaluation attempts. Relative to Sentry, these workloads reduce cumulative task execution time by approximately 53.5 and 65.4 minutes, respectively. Online use reuses the classifier; the host verification and publication path updates experience without relabeling or retraining for each new task. A single domain preparation therefore supports the full task workload, spreading its one-time labeling and fine-tuning work across repeated use.

<table><tr><td>Method</td><td>Gen. diagnosis calls ↓</td><td>Total successes ↑</td><td>Success gain vs. Base ↑</td><td>AppWorld Successes ↑</td><td>WebShop Successes ↑</td><td>Total task seconds ↓</td></tr><tr><td>Base</td><td></td><td>208/1,308</td><td></td><td>136</td><td>72</td><td>76,186.2</td></tr><tr><td>Sentry</td><td>4,987</td><td>256/1,308</td><td>+23.1%</td><td>173</td><td>83</td><td>82,002.7</td></tr><tr><td>Synapse-Control</td><td>0</td><td>270/1,308</td><td>+29.8%</td><td>178</td><td>92</td><td>74,874.0</td></tr></table>

Table 1: Primary results over 1,308 attempts per method (1,008 AppWorld; 300 WebShop). Successes satisfy all task requirements; diagnosis counts generative detector calls. Task time includes all actor and control work. Benchmark scores and paired gains appear in Section 5.2.

Recovery-host integration. Host integration preserves Sentry’s hard repair, verification, and lesson updates while replacing its generative selection path. AppWorld preserves the task’s Python session and uses the host’s ten-step soft-verification horizon. Invalid execution can receive hard repair while a soft episode is pending. Newly attached hard references and soft guidance share one budget. The verifier sees the continuation and local recovery evidence, not benchmark rewards.

The WebShop adapter supplies the same visible action constraints to every evaluated method using text observations. Search requires a visible search input; clicks require a visible target. The search check supplements the original text interface. A rejected action returns feedback and consumes an actor step. Generic contract families cover option state, empty search, visible actions, evidence checks, and purchase readiness. Before a proposed purchase, visible unmet requirements can trigger the same scoring and PVT path. At termination, pending recovery can be checked against the available shortened continuation. All compared methods use the same adapter, actor budget, and terminal evaluator.

## 5 Evaluation

The evaluation tests whether lightweight experience control improves task success and execution eficiency, and then isolates the efects of knowledge representation, candidate selection, and delivery history.

## 5.1 Methodology

Benchmarks and methods. AppWorld evaluates stateful API programming [10]. We use all 168 test\_normal tasks across 56 scenarios and six task orders: 1,008 attempts per method. WebShop evaluates search, inspection, option selection, and purchase [11]. Its main reevaluation contains 100 tasks and three task orders: 300 attempts per method.

The primary methods are Base, the unassisted Qwen3.5-9B actor; Sentry, which generates diagnoses to retrieve advice; and Synapse-Control, our learned selection and delivery run time. Sentry denotes our integration of the published recovery core with the common benchmark harness. Each comparison fixes the actor, task order, action interface, and terminal evaluator. This controls actor capability while comparing recovery policies. Further knowledge-matched controls are defined and evaluated in Appendix C.

Each repeat begins with its declared initial bank; verified lessons accumulate within the repeat. The three primary methods register 3,924 attempts, and the additional knowledgematched condition brings the total to 5,232. Actor temperature is 0.7, with at most 60 actions per task. AppWorld uses a 768-token actor-output cap; WebShop uses 384 tokens and minimal zero-shot text prompts. AppWorld training and checkpoint selection precede held-out evaluation. WebShop reevaluates a fixed 100-task prefix after development and validation. Appendix A specifies both protocols.

Measurements. Complete success means all AppWorld task checks pass, or WebShop reward equals one. Score is mean test-pass percentage or reward times 100. Means use scored attempts; completion counts and costs retain every registered attempt. Auxiliary control time sums detector, verifier, updater, and sidecar elapsed time. End-to-end task time measures the complete execution. Request timings include client and service waiting; local inference is timed at the worker. Amortized cost per completion divides total registered-workload cost by complete successes.

## 5.2 Main quality and cost results

Success yield and execution eficiency. Synapse-Control records 270 successes, versus 256 for Sentry and 208 for Base: 29.8% more than Base on the fixed workload (Table 1). It also records the lowest cumulative task time. AppWorld contributes 178 successes versus 173 for Sentry and 136 for Base; WebShop contributes 92 versus 83 and 72, increasing successes over Sentry by 5 and 9, respectively.

Mean scores for Base, Sentry, and Synapse-Control are 50.32, 56.23, and 55.95 in AppWorld, and 48.23, 56.04, and 55.82 in WebShop, respectively. Thus, Synapse-Control achieves more successes with mean scores within 0.3 points of Sentry in both benchmarks. On jointly scored task–repeat pairs, score gains against Base are +5.69 and +7.60; every repeated-order comparison is positive.

Figure 6 relates completion yield to complete execution cost. Relative to Base, total task time per completion falls by 14.1% in AppWorld and 49.7% in WebShop, while completed attempts increase in both.

<table><tr><td rowspan="2"></td><td colspan="3">AppWorld: 1,008 attempts/method</td><td colspan="3">WebShop: 300 attempts/method</td></tr><tr><td>Base</td><td></td><td>Synapse- Control</td><td>Base</td><td>Sentry</td><td>Synapse- Control</td></tr><tr><td>Metric</td><td>136/1,008</td><td>Sentry 173/1,008</td><td></td><td>72/300</td><td>83/300</td><td></td></tr><tr><td>Complete successes</td><td>30,293</td><td>33,342</td><td>178/1,008 30,828</td><td>7,225</td><td>6,977</td><td>92/300</td></tr><tr><td>Large-model requests</td><td></td><td>4,002</td><td>0</td><td>0</td><td>985</td><td>5,885</td></tr><tr><td>Generative detector calls</td><td>0</td><td></td><td></td><td>105.67</td><td></td><td>0</td></tr><tr><td>Actor input tokens (M)</td><td>234.92 0.00</td><td>226.59 21.90</td><td>246.66 8.09</td><td>0.00</td><td>71.63 6.82</td><td>62.34 3.47</td></tr><tr><td>Auxiliary input tokens (M)</td><td>234.92</td><td>248.49</td><td>254.75</td><td>105.67</td><td>78.45</td><td>65.81</td></tr><tr><td>All service input tokens (M)</td><td>1.727</td><td>1.436</td><td>1.431</td><td>1.468</td><td>0.945</td><td>0.715</td></tr><tr><td>Input tokens / completion (M)</td><td>395.45</td><td>368.14</td><td>339.78</td><td>311.17</td><td>220.66</td><td>156.45</td></tr><tr><td>Task seconds / completion Auxiliary control time (s)</td><td>0.0</td><td>10,178.5</td><td>5,067.3</td><td>0.0</td><td>2,614.8</td><td>1,280.8</td></tr><tr><td>Complete task time (s)</td><td>53,781.7</td><td>63,687.6</td><td>60,480.3</td><td>22,404.5</td><td>18,315.2</td><td>14,393.6</td></tr></table>

Table 2: Completed work and corresponding costs in both benchmarks. Per-completion metrics divide the entire registered workload’s input or task time by complete successes. Timing follows Section 5.1.

Control time is halved in both environments. The runtime replaces 4,987 generative detector calls: 4,002 in AppWorld and 985 in WebShop. AppWorld auxiliary control time falls from 10.10 to 5.03 seconds per attempt (50.2%), and Web-Shop falls from 8.72 to 4.27 seconds (51.0%). Figure 7 uses the same four components, denominator convention, and vertical scale in both panels.

AppWorld auxiliary large-model requests fall from 6,536 to 2,048 (68.7%). Its complete task time falls by 5.0% relative to Sentry. WebShop complete task time falls by 21.4% relative to Sentry and 35.8% relative to Base.

## 5.3 Controlled design comparisons

Controlled comparisons examine three design responsibilities: the content of guidance, the cost of selecting it, and the regulation of repeated delivery. Two AppWorld development probes isolate the knowledge interface and need gate (Appendix C); matched-knowledge and history comparisons report both benchmarks.

![](images/d6ef50f69fe6131295d1a7ce3d525ef406009658d41415a10c80e84e9f36dd0e.jpg)

Actionable facts improve the guidance interface. Holding the controller and need gate fixed, the 318-attempt Short/- Contract comparison changes the delivered content. Adding local interface facts to concise principles increases complete successes from 31 to 46 (+48.4%) and mean score from 50.32 to 53.05. This supports the micro-contract interface: the actor benefits from concrete constraints while retaining responsibility for its program.

Cost reduction persists under the same need gate. We give a knowledge-matched generative control the same observable necessity gate, with 342 registered attempts per arm. Synapse-Control reduces auxiliary large-model requests from 1,436 to 572 (60.2%) and task time from 17,681.7 to 16,496.8 seconds (6.7%). Aligning admission therefore preserves the eficiency advantage of the subsequent selection and delivery path. The gate alone does not account for the reduction.

Delivery bounds complement candidate selection. PVT constrains the delivery sequence: soft hints and added hard references share a 2,048-token task budget, at most one soft hint is exposed per step, and recurrence requires resolved feedback, new evidence, and a three-step cooldown. The operational arguments in Appendix E establish these budget and recurrence guarantees independently of the classifier’s ranking.

![](images/59078cfd319b3935dd63109640e52eea0eef7b6103d705308f91f8a3faa83479.jpg)  
Figure 6: More completions with lower amortized execution cost. Each benchmark normalizes completions and total task time per completion to Base (100%). The cost numerator includes all registered attempts. Both panels use the same methods and axes.

![](images/3d8bde60c3ad644d616f35734b1f11f8523bf640540dd1b89f80b9a0bf9ce9fc.jpg)  
Figure 7: Control time is halved in both benchmarks. Replacing generative diagnosis with the 1.7B sidecar reduces auxiliary control time by 50.2% in AppWorld and 51.0% in WebShop. Both panels show the same four roles per registered attempt; verification and lesson publication remain active.

Control eficiency with matched knowledge. With the knowledge interface fixed, Synapse-Control reduces auxiliary time by 50.2% in AppWorld and 60.4% in WebShop. Complete success rates are 17.7% versus the generative control’s 16.2% in AppWorld, and 30.7% versus 31.7% in WebShop; full task time falls by 8.4% and 18.9%, respectively. This comparison holds knowledge fixed while changing selection and delivery. Table 15 (Appendix C) defines the configuration and reports both benchmarks.

Operational delivery checks. The main logs contain checkable budget records for 993 AppWorld and 219 WebShop attempts, with no delivery exceeding the 2,048-token addedcontent budget. Soft hints and added hard references use the same accounting path, consistent with Equation (10). Appendix E summarizes the corresponding delivery and budget records.

History governance reduces repeated delivery. We compare full PVT (Full) with No-history on 90 matched task pairs per benchmark: 360 attempts, all scored. Six isolated workers each handle 15 distinct tasks per benchmark; both arms use the same tasks and order, with separate online experience. No-history disables utility ranking, the three-step cooldown, pending-feedback checks, and same-evidence/episode checks. It retains the classifier, contracts, need gate, 2,048-token budget, single-hint-per-step limit, and host verification and update path.

History governance reduces repeated guidance deliveries from 65 to 23 in AppWorld (64.6%) and from 6 to 2 in WebShop (66.7%). AppWorld completions increase from 15 to 17. In WebShop, both arms complete 25 attempts, while Full uses 10.1% fewer large-model requests and 12.1% fewer input tokens. Table 3 reports corresponding success counts, repeated deliveries, and workload measurements for both domains.

Reason-level counts attribute 39 and 20 AppWorld candidate blocks to cooldown and same-evidence checks, respectively; WebShop records 52 and 6. Budget checks remain active in both arms and are counted separately. On the same reached states, a shadow scan at the deployed λ = 0.1 changes the top-ranked candidate in 17 AppWorld decisions and none in WebShop. History participates in ranking in the former; its observed role in the latter lies in delivery eligibility. Appendix C.1 reports the reason breakdown and three-weight ranking scan.

## 5.4 Downstream eficiency of learned selection

We evaluate learned selection at two levels: admission judgments on held-out candidate pairs and downstream quality and cost in paired agent executions. The first measures agreement with teacher usefulness labels; the second measures the consequences of using the selected guidance.

Candidate admission precision. On classifier tests separated by scenario or product group from training, the frozen 0.90 operating point achieves 94.0% Helpful admission precision in AppWorld and 98.15% in WebShop on Contract candidates, accepting 65.8% and 72.0% of candidates (Table 4). Precision measures agreement between admitted guidance and the teacher’s usefulness labels on the deployed micro-contract input. One harmful candidate is accepted among 76 App-

<table><tr><td rowspan="2">Metric</td><td colspan="2">AppWorld</td><td colspan="2">WebShop</td></tr><tr><td>No-history</td><td>Full</td><td>No-history</td><td>Full</td></tr><tr><td>Repeated guidance deliveries</td><td>65</td><td>23</td><td>6</td><td>2</td></tr><tr><td>Fewer repeat deliveries</td><td></td><td>64.6%</td><td></td><td>66.7%</td></tr><tr><td>Complete successes</td><td>15/90</td><td>17/90</td><td>25/90</td><td>25/90</td></tr><tr><td>Large-model requests</td><td>2,859</td><td>2,900</td><td>1,890</td><td>1,699</td></tr><tr><td>Input tokens (M)</td><td>23.71</td><td>24.82</td><td>22.20</td><td>19.52</td></tr><tr><td>Output tokens (M)</td><td>0.36</td><td>0.38</td><td>0.32</td><td>0.28</td></tr></table>

Table 3: History governance removes about two thirds of repeated deliveries in both benchmarks. Each benchmark has 90 attempts per arm, all scored and paired. Costs retain all attempts. No-history preserves the budget and single-hint limit.

<table><tr><td>Metric</td><td>AppWorld</td><td>WebShop</td></tr><tr><td>Test candidates</td><td>76</td><td>150</td></tr><tr><td>Accepted candidates</td><td>50</td><td>108</td></tr><tr><td>Helpful precision</td><td>47/50 (94.0%)</td><td>106/108 (98.15%)</td></tr><tr><td>Helpful recall</td><td>47/65 (72.31%)</td><td>106/121 (87.60%)</td></tr><tr><td>Coverage</td><td>50/76 (65.79%)</td><td>108/150 (72.00%)</td></tr></table>

Table 4: Admission precision and coverage on deployed content. Both benchmarks use Contract test candidates and frozen admission rules. Precision divides accepted Helpful candidates by all accepted candidates; recall divides accepted Helpful candidates by all Helpful candidates.

World Contract candidates, and none among 150 WebShop candidates; Appendix A.3 provides content-stratified counts.

Calibration sensitivity characterizes the precision–coverage trade-of around the frozen threshold using the mixed-content calibration sets. In WebShop, increasing the threshold from 0.90 to 0.95 raises precision from 96.38% to 98.77% and lowers coverage from 73.67% to 54.00%. AppWorld has a precision plateau above 98.8% from 0.80 to 0.90. The common 0.90 operating point maintains high admission precision on these calibration sets; Appendix A.4 reports the complete scan.

Execution savings from learned selection. The online comparison isolates the contribution of suitability scoring beyond lexical admission. We compare lexical priority with learned state–payload suitability while fixing the bank, shortlist, need gate, PVT, and host lifecycle; the lexical configuration loads no classifier. Both policies are executed afresh on the development workloads, with 171 registered attempts per arm in AppWorld and 216 in WebShop. Table 5 uses their 169 and 215 jointly scored task–repeat pairs for both quality and cost.

Learned selection retains 97.7% of the lexical mean score in AppWorld and 99.2% in WebShop, while reducing input tokens by 11.2% and 5.8% and complete task time by 6.1% and 4.3%, respectively (Table 5). Across these paired panels, it avoids approximately 7.03M input tokens, 447 large-model requests, and 999.6 seconds of task time. Appendix B reports costs over all registered attempts.

<table><tr><td>Metric</td><td>AppWorld 169 pairs</td><td>WebShop 215 pairs</td></tr><tr><td rowspan="2">Input tokens (M)</td><td>36.41 → 32.32</td><td>50.55 → 47.61</td></tr><tr><td>-4.09 (11.2%)</td><td>-2.94 (5.8%)</td></tr><tr><td rowspan="2">Requests</td><td>4,646 → 4,331</td><td>4,409 → 4,277</td></tr><tr><td>-315 (6.8%) 8,367.2 → 7,860.4 11,331.1 → 10,838.3</td><td>-132 (3.0%)</td></tr><tr><td rowspan="2">Task time (s)</td><td>-506.8 (6.1%)</td><td>-492.8 (4.3%)</td></tr><tr><td>97.7%</td><td></td></tr><tr><td>Mean-score retention</td><td>49.80 / 50.99</td><td>99.2%</td></tr><tr><td></td><td></td><td>56.38 / 56.86</td></tr></table>

Table 5: Paired selector costs: lexical → learned, followed by the absolute and relative reduction. Quality and cost use the same task–repeat pairs. Mean-score retention is learned score divided by lexical score; both values are shown.

## 5.5 Execution-work analysis

Execution accounting separates auxiliary control savings from changes in actor work and relates the total workload to successful outcomes.

Whole-workload token eficiency. Across the fixed 1,308- attempt workload, Synapse-Control records the lowest cumulative large-model input: 320.56M tokens, versus 326.94M for Sentry and 340.59M for Base. It completes 270 attempts with this workload. Amortizing all recorded input over complete successes gives 1.187M tokens per completion, versus 1.277M for Sentry and 1.637M for Base, a 27.5% reduction relative to Base.

Corresponding benchmark-level costs. In AppWorld, auxiliary input tokens fall from 21.90M to 8.09M (63.1%). Total input is 254.75M versus Sentry’s 248.49M, and amortized input per completion is 1.431M versus 1.436M. In WebShop, auxiliary input falls from 6.82M to 3.47M (49.1%); total input falls from 78.45M to 65.81M, and amortized input per completion from 0.945M to 0.715M. Table 2 reports the same actor, auxiliary, total, and outcome-normalized measures for each benchmark.

Action-level behavior. Action-level diagnostics identify how guidance changes navigation and rejected actions in a separate WebShop validation workload of 72 tasks and three repeats. Synapse-Control reaches terminal purchase execution in 184/216 attempts, versus 152 for Base. Rejected actions decrease from 4,091 to 1,728 (57.8%), total steps from 5,539 to 3,868 (30.2%), and step-limit stops from 64 to 32. Compared with Sentry on the same validation workload, rejected actions fall by 20.0% (2,159 to 1,728) and total steps by 10.6% (4,327 to 3,868). Table 14 reports the complete diagnostic ledger, separating terminal purchases from fully successful attempts. The 100-task main reevaluation remains the primary quality comparison.

## 5.6 Deployment and reusable knowledge

Deployment reuses a prepared classifier and knowledge bank while keeping runtime control local to the actor. Single-GPU serving colocates both models throughout evaluation. Porting the runtime involves the observation adapter, local interface facts, and controller calibration for the target actor; admission, suitability, and delivery have explicit interfaces. Appendix F evaluates direct reuse of the frozen controller in a Llama-based runtime without retraining.

Separating immutable knowledge assets from task-local state also supports a shared-store layout. A CPU object-layout replay reduces serialized store capacity by 59.3% at six workers by storing immutable assets once while retaining private task metadata (Appendix D). Versioned files, an object service, or a future CXL-backed store could supply that immutable base. Private trajectories, credentials, and PVT feedback remain local. The layout isolates knowledge ownership and publication from task-local recovery decisions.

## 6 Related Work

We position lightweight experience control relative to agent knowledge representations, recovery mechanisms, and inference optimizations.

Reusable agent knowledge. Experience systems difer in both stored representation and runtime knowledge use. Voyager develops executable skills; ExpeL and Reflexion extract or retain lessons; Agent Workflow Memory and Reasoning-Bank organize reusable procedures and strategies [1–3, 12, 13]. ACE evolves context through reflection and curation, while MemGPT manages memory access across tiers [7, 14]. Table 6 compares the knowledge and runtime interfaces of representative systems. Our focus is the decision between retrieval and use: relating a concrete payload to present evidence and controlling its delivery history.

Recovery and monitoring. Recovery and monitoring systems provide conditional correction during agent execution. Sentry couples diagnosis with retrieved lessons, verification, and online publication [4]. CRITIC uses tool feedback for correction; Wink supplies course corrections during coding; AgentFixer produces failure analyses and fixes; AgentForesight audits partial trajectories for risk [15–18]. Conditional intervention is established prior work. We specialize its learned decision to the actual candidate, retain host verification, and account for the cost of both selection and the resulting execution.

<table><tr><td>System</td><td>Knowledge</td><td>Admission</td><td>Decision</td><td>State</td></tr><tr><td>Voyager [1]</td><td>Code skills</td><td>Skill retrieval</td><td>Actor reuse</td><td>Skill bank</td></tr><tr><td>ACE [7]</td><td>Playbook</td><td>Context</td><td>Reflect +</td><td>Entry</td></tr><tr><td rowspan="3">Sentry [4]</td><td></td><td>update</td><td>curate</td><td>counters</td></tr><tr><td>Recovery</td><td>Need</td><td>Generated</td><td>Verified</td></tr><tr><td>lessons</td><td>checking</td><td>diagnosis</td><td>lessons</td></tr><tr><td>Synapse-</td><td></td><td></td><td></td><td></td></tr><tr><td>Control</td><td>Micro- contracts</td><td>Visible evidence</td><td>Pair classifier</td><td>PVT + budget</td></tr></table>

Table 6: Selected knowledge and runtime interfaces. Columns distinguish the stored asset, admission signal, decision mechanism, and persistent state.

Eficient inference. Inference optimizations reduce the cost of executing model requests through routing, serving, and reuse. RouteLLM learns model-routing decisions, and Cortex validates cached knowledge for eficient access [19, 20]. vLLM, SGLang, and CacheBlend improve the serving or reuse of model computation [21–23]. Our runtime operates before these serving optimizations: it changes which recovery requests are issued and which guidance reaches the actor.

## 7 Conclusion

Synapse-Control reduces the cost of reusing experience by assessing existing suggestions directly instead of generating a diagnosis to retrieve them. A small classifier evaluates advice against the current state; micro-contracts express interface checks; PVT regulates repeated use through feedback and budgets. Across the evaluated AppWorld and WebShop workloads, this division improves successful task yield over Base and Sentry while halving auxiliary control time. Component studies link learned selection to lower execution cost and delivery-history checks to fewer repeated suggestions. The actor retains task planning while the runtime manages the advice it receives.

## Acknowledgments

This work was supported by Ant Group through CCF-Ant Research Fund.

## References

[1] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv

preprint arXiv:2305.16291, 2023. URL https://arxiv.org/abs/ 2305.16291.

[2] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM Agents Are Experiential Learners. arXiv preprint arXiv:2308.10144, 2023. URL https://arxiv.org/abs/ 2308.10144.

[3] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent Workflow Memory. arXiv preprint arXiv:2409.07429, 2024. URL https://arxiv.org/abs/2409.07429.

[4] Changxiu Ji, Amy Lu, Qizheng Zhang, and Kunle Olukotun. Sentry: Learning to Recover from LLM Agent Failures at Test Time. arXiv preprint arXiv:2610.02994, 2026. URL https://arxiv.org/abs/ 2610.02994.

[5] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv preprint arXiv:2210.03629, 2022. URL https://arxiv.org/abs/2210.03629.

[6] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable Code Actions Elicit Better LLM Agents. arXiv preprint arXiv:2402.01030, 2024. URL https: //arxiv.org/abs/2402.01030.

[7] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. arXiv preprint arXiv:2510.04618, 2025. URL https:// arxiv.org/abs/2510.04618.

[8] Stephane Ross, Geofrey Gordon, and Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635. PMLR, 2011. URL https://proceedings.mlr.press/v15/ross11a.html.

[9] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On Calibration of Modern Neural Networks. arXiv preprint arXiv:1706.04599, 2017. URL https://arxiv.org/abs/1706.04599.

[10] Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents. arXiv preprint arXiv:2407.18901, 2024. URL https://arxiv.org/abs/ 2407.18901.

[11] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. arXiv preprint arXiv:2207.01206, 2022. URL https://arxiv.org/abs/2207.01206.

[12] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv preprint arXiv:2303.11366, 2023. URL https://arxiv.org/abs/2303.11366.

[13] Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. arXiv preprint arXiv:2509.25140, 2025. URL https://arxiv.org/abs/2509.25140.

[14] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as Operating Systems. arXiv preprint arXiv:2310.08560, 2023. URL https://arxiv.org/abs/2310.08560.

[15] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. arXiv preprint arXiv:2305.11738, 2023. URL https://arxiv.org/abs/2305. 11738.

[16] Rahul Nanda, Chandra Maddila, Smriti Jha, Euna Mehnaz Khan, Matteo Paltenghi, and Satish Chandra. Wink: Recovering from Misbehaviors in Coding Agents. arXiv preprint arXiv:2602.17037, 2026. URL https://arxiv.org/abs/2602.17037.

[17] Hadar Mulian, Sergey Zeltyn, Ido Levy, Liane Galanti, Avi Yaeli, and Segev Shlomov. AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems. arXiv preprint arXiv:2603.29848, 2026. URL https://arxiv.org/abs/2603.29848.

[18] Boxuan Zhang, Jianing Zhu, Zeru Shi, Dongfang Liu, and Ruixiang Tang. AgentForesight: Online Auditing for Early Failure Prediction in Multi-Agent Systems. arXiv preprint arXiv:2605.08715, 2026. URL

https://arxiv.org/abs/2605.08715.

[19] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. RouteLLM: Learning to Route LLMs with Preference Data. arXiv preprint arXiv:2406.18665, 2024. URL https://arxiv.org/abs/ 2406.18665.

[20] Chaoyi Ruan, Chao Bi, Kaiwen Zheng, Ziji Shi, Xinyi Wan, and Jialin Li. Cortex: Achieving Low-Latency, Cost-Eficient Remote Data Access For LLM via Semantic-Aware Knowledge Caching. arXiv preprint arXiv:2509.17360, 2025. URL https://arxiv.org/abs/ 2509.17360.

[21] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient Memory Management for Large Language Model Serving with PagedAttention. arXiv preprint arXiv:2309.06180, 2023. URL https://arxiv.org/abs/2309.06180.

[22] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Eficient Execution of Structured Language Model Programs. arXivpreprint arXiv:2312.07104, 2023. URL https://arxiv.org/abs/2312.07104.

[23] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. arXiv preprint arXiv:2405.16444, 2024. URL https://arxiv.org/ abs/2405.16444.

## A Implementation, Training, and Evaluation Protocols

This appendix specifies the deployment configuration, knowledge construction, teacher supervision, and classifier evaluation protocol. Table 7 summarizes the corresponding serving, budget, and training settings.

<table><tr><td>Setting</td><td>AppWorld</td><td>WebShop</td></tr><tr><td>Actor / classifier</td><td>Qwen3.5-9B / Qwen3-1.7B</td><td>Qwen3.5-9B / Qwen3-1.7B</td></tr><tr><td>Actor serving</td><td>BF16; non-thinking; TP=1</td><td>BF16; non-thinking; TP=1</td></tr><tr><td>Actor output / action cap</td><td>768 tokens / 60 actions</td><td>384 tokens / 60 actions</td></tr><tr><td>Actor temperature</td><td>0.7</td><td>0.7</td></tr><tr><td>Service context / active sequences</td><td>40,960 tokens / one</td><td>40,960 tokens / one</td></tr><tr><td>Actor memory utilization</td><td>0.65</td><td>0.65</td></tr><tr><td>Auxiliary generation</td><td>Temperature zero; 4,096 output tokens</td><td>Temperature zero; 4,096 output tokens</td></tr><tr><td>Classifier input</td><td>Five recent steps; at most 1,024 tokens</td><td>Five recent steps; at most 1,024 tokens</td></tr><tr><td>Guidance budget / shortlist</td><td>2,048 tokens per task / up to five</td><td>2,048 tokens per task / up to five</td></tr><tr><td>PVT</td><td>Cooldown three; α = 0.5; λ = 0.1</td><td>Cooldown three; α = 0.5; λ = 0.1</td></tr><tr><td>Fine-tuning</td><td>Full backbone + reset three-way head</td><td>Full backbone + reset three-way head</td></tr><tr><td>Optimizer</td><td>AdamW; learning rate  $2 \times 1 0 ^ { - 5 }$ </td><td>AdamW; learning rate  $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Batch / accumulation / seed</td><td>Two / eight / 42</td><td>Two / eight / 42</td></tr><tr><td>Numerical precision</td><td>BF16 parameters; FP32 loss logits</td><td>BF16 parameters; FP32 loss logits</td></tr><tr><td>Gradient checkpointing</td><td>Disabled</td><td>Enabled</td></tr><tr><td>Optimizer updates Timing scope</td><td>456</td><td>450</td></tr><tr><td></td><td>Two concurrent heads: loading, training, evalua- tion, saving</td><td>Domain-model training, evaluation, saving</td></tr><tr><td>Training-chain time (timestamps)</td><td>Approximately 13 minutes</td><td>Approximately 15 minutes</td></tr></table>

Table 7: Corresponding deployment and training settings. Each classifier is trained independently from the common initial checkpoint and frozen before its final evaluation.

Construction timing. Training-chain elapsed times are derived from artifact filesystem timestamps. The AppWorld batch takes approximately 13 minutes, with two concurrent heads and 456 updates per head; teacher labeling precedes this batch and is excluded. The WebShop construction chain takes approximately 38.5 minutes for labeling, 16 seconds for data assembly, and 15 minutes for training, evaluation, and persistence, totaling approximately 54 minutes.

Training and calibration. AppWorld divides oficial training scenarios into 24/3/3 train/calibration/test groups and assembles up to 3,000 candidate pairs. WebShop uses disjoint product groups. In both domains, complete candidate groups are retained under the sample-pair budget. Calibration selects from temperatures {0.5, 0.75, 1, 1.5, 2, 3} by calibration negative log-likelihood. Delivery requires calibrated Helpful probability greater than 0.90.

WebShop data construction. Six oficial-training collection jobs, three Base and three assisted, yield 5,632 visible windows and 20,686 candidate pairs. Grouped sampling produces 2,398 training pairs in 661 groups over 90 product groups, with 300 calibration pairs and 300 test pairs over separate product groups. Training uses 1,080 Base and 1,318 assisted pairs. The realized rows contain 1,657 failure, 349 completion-boundary, and 392 progress cases, or 69.1/14.6/16.3%. Window sampling targets 40%, 30%, and 30% failure or stagnation, completion-boundary, and ordinary-progress states before retaining complete groups under a pair budget. The realized pair proportions depend on candidate counts within each group. Teacher labels are 1,769 Helpful, 541 Irrelevant, and 88 Harmful. Of the pairs, 1,820 bind generic environment contracts to visible evidence, and 578 use chronologically available online lessons.

Execution protocols. AppWorld retains a persistent Python session and the host’s ten-step soft-verification horizon. WebShop uses text observations and minimal zero-shot prompts. Search requires a visible search input, and clicks require visible targets. All methods share the action adapter and terminal evaluator. The 72-task validation ledger precedes the fixed 100-task main reevaluation; configurations and knowledge resources are fixed before that reevaluation.

Actor-output budget probe. A separate AppWorld development probe holds the actor, task pool, task orders, and recovery configuration fixed, registering 342 attempts per condition. At 384/768 output tokens, Base has mean scores 46.41/49.36 and completes 30/44 attempts; Sentry scores 50.75/49.92 and completes 38/36; Synapse-Control scores 48.62/52.46 and completes 32/47. The task action cap remains 60. These are actor-output-cap responses on the development workload.

## A.1 Knowledge sources and runtime interfaces

Table 8 describes corresponding construction stages in both domains. Generic principles are authored at the domain level; interface facts and current state come from environment documentation and public observations. Verified lessons enter the worker’s online bank. Porting to another environment requires observation adaptation and local fact binding, together with controller calibration on states from the target actor.

<table><tr><td>Stage</td><td>AppWorld</td><td>WebShop</td></tr><tr><td>Environment facts</td><td>Signatures, parameters, and return conventions in API Visible actions, option values, and page state documentation</td><td></td></tr><tr><td>Generic references</td><td>Authored trigger and checking principles</td><td>Five environment-reference families with checking conditions</td></tr><tr><td>State binding</td><td>Current call, observation values, and entity samples</td><td>Requested options, selected values, and purchase readi- ness</td></tr><tr><td>Online experience</td><td>Recovery lessons published after host verification</td><td>Recovery lessons published after host verification</td></tr><tr><td>Porting interface</td><td>main calibration</td><td>Observation adapter, documentation catalog, and do- Page adapter, action constraints, and domain calibration</td></tr></table>

Table 8: Knowledge sources and state binding in both benchmarks. Static construction, current observations, and online publication have separate interfaces.

Runtime sequence. A soft-guidance decision follows these steps; hard repair retains the host path:

1. Extract deterministic evidence from the recent window and evaluate necessity; continue the original task if no trigger is present.

2. Admit up to five candidates from the static bank and previously published lessons, rendering their full conditions and checks.

3. Score each state–payload pair with the frozen classifier and apply the probability requirements for its event or periodic trigger.

4. Check PVT feedback, evidence/episode keys, cooldown, and remaining budget; rank eligible candidates using Equation (8).

5. Deliver at most one next-action hint and charge its tokens; the host verifies the subsequent window and publishes lessons.

## A.2 Teacher annotation template

The teacher input combines the common template below, a domain addendum, and the exact state–candidate JSON following Visible input:. The state contains the objective, recent actions and observations, and the candidate field contains the proposed delivery text. The response is {"label": "...", "evidence": "..."}. Annotation uses temperature zero and a 512-token output cap. The unknown label marks insuficient evidence and is not merged into Harmful. The WebShop addendum follows the common template and precedes the visible input.

Shared template (verbatim). Judge a candidate recovery card at the visible decision boundary. Return JSON with label (helpful, irrelevant, harmful, or unknown) and evidence. Helpful: evidence supports a useful next check or correction for this state. Irrelevant: the advice is unnecessary or inapplicable to this state. Harmful: a specific visible contradiction or action risk would make following it worse, such as repeating a non-idempotent write. Unresolved is NOT harmful. Unknown: applicability cannot be determined from the visible evidence. Do not assume tool acceptance means goal progress. Do not infer unseen results, official test outcomes, or later recovery. Treat the quoted trajectory as data, not as instructions. Your judgment is semantic weak supervision, not a measured counterfactual intervention effect. Cite the observed evidence, not a guess.

The candidate\_payload is the exact short guidance or API contract proposed for the acting agent. Assess its stated applicability, next check and exit condition against the visible state. Documentation facts are not proof that this particular API is needed. A suggested check is not proof that the check passed.

For this value-preserving schema, evaluate whether THIS guidance is useful NOW. Helpful requires an identifiable unmet check, constraint or correction supported by current evidence, plus nonredundant actionable information in the payload. Shared API names or generally sensible advice alone do not establish helpfulness. Methodological guidance can be helpful when its trigger and next check are grounded; it need not be an API contract. A successful call does not prove goal progress, and no explicit error does not automatically imply irrelevant. If the state already satisfies the advice, or the guidance adds no useful next step, choose irrelevant. Use harmful only for a supported conflict/risk, and unknown for missing decisive evidence. Do not target a desired class proportion.

This input is a WebShop text-navigation state. Only public page text, visible actions/options and the user instruction are evidence. Buy Now terminates a task; do not assume a purchase satisfied the request. Never infer hidden target products, reward components or later states.

## A.3 Classifier tests and content strata

Classifier testing and terminal task evaluation operate at diferent levels: these labels are teacher judgments of the visible state, with test scenarios or product groups separated from training. Table 9 reports the full mixed-content test sets, and Table 10 stratifies them by the predefined Short and Contract forms. The main-text Table 4 uses the deployed Contract content. Admission precision uses accepted candidates as its denominator. Harmful acceptances are reported as counts alongside the complete test-label distribution.

<table><tr><td>Mixed-content test</td><td>AppWorld</td><td>WebShop</td></tr><tr><td>Candidates / scenario or product groups</td><td>310 /3</td><td>300 / 22</td></tr><tr><td>Labels: Helpful / Irrelevant / Harmful</td><td>255 / 50 / 5</td><td>232 / 66 / 2</td></tr><tr><td>Accepted candidates</td><td>208</td><td>209</td></tr><tr><td>Accepted Helpful / harmful</td><td>204 / 1</td><td>203 / 2</td></tr><tr><td>Helpful precision</td><td>204/208 (98.08%)</td><td>203/209 (97.13%)</td></tr><tr><td>Helpful recall</td><td>204/255 (80.00%)</td><td>203/232 (87.50%)</td></tr><tr><td>Coverage</td><td>208/310 (67.10%)</td><td>209/300 (69.67%)</td></tr><tr><td>State groups / groups with accepted candidates</td><td>140 / 104</td><td>154 /98</td></tr><tr><td>Top-1 Helpful within accepted groups (before PVT)</td><td>104/104 (100.00%)</td><td>93/98 (94.90%)</td></tr></table>

Table 9: Complete classifier tests at the frozen operating point. Top-1 is the highest-priority candidate within each accepted group before PVT.

<table><tr><td></td><td colspan="2">AppWorld</td><td colspan="2">WebShop</td></tr><tr><td>Metric</td><td>Contract</td><td>Short</td><td>Contract</td><td>Short</td></tr><tr><td>Test candidates</td><td>76</td><td>234</td><td>150</td><td>150</td></tr><tr><td>Accepted candidates</td><td>50</td><td>158</td><td>108</td><td>101</td></tr><tr><td>Accepted Helpful</td><td>47</td><td>157</td><td>106</td><td>97</td></tr><tr><td>Helpful precision</td><td>94.00%</td><td>99.37%</td><td>98.15%</td><td>96.04%</td></tr><tr><td>Helpful recall</td><td>72.31%</td><td>82.63%</td><td>87.60%</td><td>87.39%</td></tr><tr><td>Coverage</td><td>65.79%</td><td>67.52%</td><td>72.00%</td><td>67.33%</td></tr><tr><td>Accepted harmful candidates</td><td>1</td><td>0</td><td>0</td><td>2</td></tr></table>

Table 10: Test metrics by input content. The content mix difers across domains; candidate and acceptance counts are reported for each.

## A.4 Threshold sensitivity around the common operating point

Table 11 is a post-hoc scan on the original calibration sets, reusing the frozen models, temperatures, and prediction caches. In WebShop, raising the threshold from 0.50 to 0.90 increases precision from 92.71% to 96.38%, with 73.67% coverage. Moving to 0.95 adds 2.39 precision points while removing 19.67 coverage points. AppWorld precision exceeds 98.8% at 0.80, 0.85, and 0.90, forming a broad high-precision plateau. Both benchmarks retain the original 0.90 operating point and common deployment configuration.

<table><tr><td rowspan="2">T</td><td colspan="4">AppWorld: 266 candidates</td><td colspan="4">WebShop: 300 candidates</td></tr><tr><td>Accept</td><td>Precision</td><td>Coverage</td><td>Harmful</td><td>Accept</td><td>Precision</td><td>Coverage</td><td>Harmful</td></tr><tr><td>0.50</td><td>203</td><td>94.58%</td><td>76.32%</td><td>3</td><td>247</td><td>92.71%</td><td>82.33%</td><td>5</td></tr><tr><td>0.60</td><td>200</td><td>95.00%</td><td>75.19%</td><td>2</td><td>241</td><td>95.02%</td><td>80.33%</td><td>5</td></tr><tr><td>0.70</td><td>193</td><td>96.37%</td><td>72.56%</td><td>2</td><td>239</td><td>95.40%</td><td>79.67%</td><td>4</td></tr><tr><td>0.80</td><td>181</td><td>98.90%</td><td>68.05%</td><td>2</td><td>232</td><td>95.69%</td><td>77.33%</td><td>3</td></tr><tr><td>0.85</td><td>179</td><td>98.88%</td><td>67.29%</td><td>2</td><td>231</td><td>95.67%</td><td>77.00%</td><td>3</td></tr><tr><td>0.90</td><td>172</td><td>98.84%</td><td>64.66%</td><td>2</td><td>221</td><td>96.38%</td><td>73.67%</td><td>2</td></tr><tr><td>0.95</td><td>164</td><td>100.00%</td><td>61.65%</td><td>0</td><td>162</td><td>98.77%</td><td>54.00%</td><td>1</td></tr></table>

Table 11: Precision, coverage, and harmful-acceptance counts over the same calibration thresholds. Bold values mark the deployed operating point.

## B Detailed Reliability and Cost Accounting

The detailed ledgers connect the main results to their scoring coverage, component costs, selector workloads, and action-level validation records.

Coverage and repeated orders. AppWorld scores 1,005/991/993 of 1,008 registered attempts for Base/Sentry/Synapse-Control. WebShop scores all 300 attempts per method. Complete-success and cost totals retain every registered attempt. AppWorld’s available-pair score diferences against Base are +5.10, +6.11, +2.33, +6.22, +6.92, and +7.42 across six orders; WebShop’s three are +5.55, +12.19, and +5.05. Each order favors Synapse-Control.

<table><tr><td>Benchmark</td><td>Method</td><td>Detector</td><td>Verifier</td><td>Updater</td><td>Sidecar</td></tr><tr><td>AppWorld</td><td>Base</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>AppWorld</td><td>Sentry</td><td>5,784.47</td><td>3,362.41</td><td>1,031.57</td><td>0.00</td></tr><tr><td>AppWorld</td><td>Synapse-Control</td><td>0.00</td><td>2,905.83</td><td>641.21</td><td>1,520.25</td></tr><tr><td>WebShop</td><td>Base</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>WebShop</td><td>Sentry</td><td>1,563.33</td><td>924.91</td><td>126.51</td><td>0.00</td></tr><tr><td>WebShop</td><td>Synapse-Control</td><td>0.00</td><td>905.79</td><td>247.72</td><td>127.33</td></tr></table>

Table 12: Total component elapsed seconds in the main experiments. The same roles are reported for both benchmarks; means in the main figure divide by 1,008 AppWorld or 300 WebShop registered attempts.

Request accounting. AppWorld’s Sentry control consists of 4,002 detector, 1,929 verifier, and 605 updater requests, totaling 6,536. Synapse-Control uses 0, 1,677, and 371, totaling 2,048. Local classifier batches are counted separately from large-model requests. Table 12 reports the corresponding elapsed times in both benchmarks.

Timing boundaries. Auxiliary elapsed time is measured by detector, verifier, updater, and local sidecar; task wall-clock covers complete execution. Request timings include client and service waiting and are not added to task wall-clock. All component timings are recorded in the main evaluations. AppWorld records 2,486 sidecar batches over 1,008 attempts, totaling 1,520.25 seconds. WebShop records 530 batches over 300 attempts, totaling 127.33 seconds. These correspond to 1.508 and 0.424 sidecar seconds per attempt. Ofline data construction and training are separate from online cost.

Complete selector workloads. The main selector table uses jointly scored task–repeat pairs for every ratio. Table 13 additionally retains all registered attempts, including those outside that paired panel. Both configurations share knowledge, the need gate, PVT, and host recovery; the lexical mode has no detector or classifier calls. Across the registered workloads, learned selection avoids 290 AppWorld and 159 WebShop large-model requests, for a total of 449, and saves 326.68 and 626.02 task seconds, totaling 952.70. This is a diferent population from the 169/215 paired panels.

<table><tr><td rowspan="2">Metric</td><td colspan="2">AppWorld: 171 attempts/arm</td><td colspan="2">WebShop: 216 attempts/arm</td></tr><tr><td>Lexical</td><td>Learned</td><td>Lexical</td><td>Learned</td></tr><tr><td>Large-model requests</td><td>4,708</td><td>4,418</td><td>4,463</td><td>4,304</td></tr><tr><td>Sidecar time (s)</td><td>0.00</td><td>225.40</td><td>0.00</td><td>109.63</td></tr><tr><td>Complete task time (s)</td><td>8,527.39</td><td>8,200.71</td><td>11,525.44</td><td>10,899.42</td></tr></table>

Table 13: All-registered selector costs with identical metrics across benchmarks. Both studies retain all registered attempts in these totals.

Validation behavior ledger. Table 14 reports the separate 72-task WebShop validation suite, with three repeats per method. Complete success requires reward one; a terminal purchase need not meet every task requirement.

<table><tr><td colspan="4"></td><td rowspan="2">Step-limit stops ↓</td><td rowspan="2">Rejected actions↓</td><td rowspan="2">Steps ↓</td></tr><tr><td>Method</td><td>Score ↑</td><td>Complete ↑</td><td>Terminal purchases ↑</td></tr><tr><td>Base</td><td>46.08</td><td>37</td><td>152</td><td>64</td><td>4,091</td><td>5,539</td></tr><tr><td>Sentry</td><td>51.78</td><td>36</td><td>183</td><td>33</td><td>2,159</td><td>4,327</td></tr><tr><td>Synapse-Control</td><td>54.69</td><td>43</td><td>184</td><td>32</td><td>1,728</td><td>3,868</td></tr></table>

Table 14: WebShop validation behavior: 72 tasks and 216 attempts per method. All three methods use the same action interface and action cap.

## C Knowledge-Matched Controls and Design Probes

The following controls distinguish the knowledge supplied to the actor from the mechanisms that select and deliver it.

Anatomy of Shared. Shared is the knowledge-matched generative control, defined as

$$
\begin{array} { l } { { \mathrm { S h a r e d } = \mathrm { S e n t r y ~ r e c o v e r y ~ c o r e } } } \\ { { \qquad + \mathrm { S y n a p s e ~ k n o w l e d g e ~ i n t e r f a c e } . } } \end{array}\tag{11}
$$

The Sentry core retains generated diagnosis, its intervention-need decision, hard repair, verification, and verified lesson updates. The knowledge interface supplies the same initial contract resources, evidence-bound rendering, documented hard references where applicable, and the 2,048-token guidance budget. Shared uses the generative detector to select recovery content; it has neither the learned suitability classifier nor PVT utility/cooldown scheduling. All arms retain the common task actor, action adapter, and benchmark evaluator. This control separates the supplied knowledge from the selection and delivery policy.

<table><tr><td rowspan="2">Metric</td><td colspan="2">AppWorld</td><td colspan="2">WebShop</td></tr><tr><td>Shared</td><td>Synapse-Control</td><td>Shared</td><td>Synapse-Control</td></tr><tr><td>Complete success rate</td><td>16.2%</td><td>17.7%</td><td>31.7%</td><td>30.7%</td></tr><tr><td>Generative detector calls</td><td>4,166</td><td>0</td><td>1,417</td><td>0</td></tr><tr><td>Auxiliary control seconds</td><td>10,180.2</td><td>5,067.3</td><td>3,231.1</td><td>1,280.8</td></tr><tr><td>Complete task seconds</td><td>65,991.0</td><td>60,480.3</td><td>17,754.5</td><td>14,393.6</td></tr></table>

Table 15: Complete success rates and costs with matched knowledge. Rates divide completely successful attempts by all registered attempts: 1,008 per method in AppWorld and 300 in WebShop. Shared is defined above; Synapse-Control denotes the full runtime.

Table 15 compares knowledge-matched recovery using complete success rates and execution costs. In AppWorld, Synapse-Control achieves a 17.7% success rate versus Shared’s 16.2%, with auxiliary time 5,067.3 versus 10,180.2 seconds. In WebShop, the corresponding success rates are 30.7% and 31.7%; Synapse-Control eliminates 1,417 generative detector calls, reduces auxiliary time by 60.4%, and reduces complete task time by 18.9%.

Matched content and need-gate probes. Two AppWorld development probes isolate concrete design substitutions. Under the same controller and gate, the balanced 318-attempt Short/Contract comparison yields mean scores 50.32/53.05 and complete successes 31/46. Contract adds local interface facts to the concise principle.

The Shared + need gate control additionally applies the full runtime’s observable necessity predicate while preserving generative selection. On 342 registered attempts per arm, this control and Synapse-Control score 331/340 attempts and complete 37/47. The balanced score gain is +0.49, and complete-success gain is +3.19 percentage points. Auxiliary calls decrease from 1,436 to 572 and task time from 17,681.7 to 16,496.8 seconds. This aligns intervention admission while comparing the subsequent control paths.

## C.1 Paired history ablation and reason counts

Table 3 reports success counts, repeated deliveries, and costs for the matched comparison. No-history sets historical utility contribution to zero and skips PVT cooldown, pending-feedback, and same-evidence/episode checks. The budget, single-hint limit, need gate, classifier, and host verification remain fixed. Each arm registers and scores 90 attempts per benchmark; both paired panels contain 90 pairs, and all cost fields are recorded.

Table 16 distinguishes selection decisions from candidate-checking opportunities. History-blocked candidates have mutually exclusive first reasons: 39 + 20 = 59 cooldown or same-evidence blocks in AppWorld, and 52 + 6 = 58 in WebShop. Budget blocks are separate and excluded from these history counts. No-history records zero history blocks in both domains; the retained budget can still block delivery. Host verification waits are logged separately and receive no credit as PVT blocks.

<table><tr><td>Full-PVT observations</td><td>AppWorld</td><td>WebShop</td></tr><tr><td>Decisions with a history block</td><td>54</td><td>58</td></tr><tr><td>Candidate opportunities blocked by history</td><td>59</td><td>58</td></tr><tr><td>Of these: cooldown</td><td>39</td><td>52</td></tr><tr><td>Of these: same evidence</td><td>20</td><td>6</td></tr><tr><td>Budget-blocked candidates (separate)</td><td>3</td><td>7</td></tr><tr><td>Top-ranked candidate changes at  $\lambda = 0 . 0 5$ </td><td>15</td><td>0</td></tr><tr><td>Top-ranked candidate changes at  $\lambda = 0 . 1 0$ </td><td>17</td><td>0</td></tr><tr><td>Top-ranked candidate changes at  $\lambda = 0 . 2 0$ </td><td>17</td><td>0</td></tr></table>

Table 16: Full-PVT blocking reasons and shadow-ranking counts. History totals distinguish decisions from candidate opportunities. The ranking scan fixes the reached states and compares the top-ranked candidate under diferent historical weights.

The ranking scan reuses Full-arm states and candidates at $\lambda = 0 . 0 5 , 0 . 1 0 , 0 . 2 0$ , counting changes in the top-ranked candidate relative to ranking without historical utility. The actual online configuration retains $\lambda = 0 . 1 0$ . This measures history’s participation at fixed states; Table 3 measures the online results of the complete history mechanism

## D Immutable Stores and Cluster Capacity

The capacity study measures the storage efect of sharing immutable knowledge while retaining task-local state; its unit of analysis is the serialized object layout.

Six completed Synapse-Control event streams are replayed as logical workers, preserving local chronology and a declared round-robin interleaving. The private layout gives each worker an exact-deduplicated store. The shared layout stores immutable assets once while retaining private ownership, PVT feedback, and unique dynamic lessons. Both layouts charge a 64-byte index per object. Figure 8 shows the capacity efect as worker count increases.

![](images/2c8061c7035eaa3f490740783205647761a6bf0700955a3adbfa75abcfc2de29.jpg)  
Figure 8: Serialized object capacity on 1/2/4/6 real streams. Sharing immutable assets reduces total store capacity by 59.3% at six workers. This is a CPU object-layout replay.

Let W be worker count, S common immutable payload size, $N _ { s }$ common object count, and b index bytes per object. The replay contains $N _ { s } = 1 2$ common serialized objects with payload $S = 5 4 9 , 0 7 7$ bytes (about 549.1 kB). For worker-private dynamic payload $D _ { w } .$ , metadata $M _ { w } ,$ , and $N _ { d }$ unique dynamic objects,

$$
B _ { \mathrm { p r i v a t e } } = W ( S + b N _ { s } ) + \sum _ { w = 1 } ^ { W } ( D _ { w } + M _ { w } ) + b N _ { d } ,\tag{12}
$$

$$
B _ { \mathrm { s h a r e d } } = S + b N _ { s } + \sum _ { w = 1 } ^ { W } ( D _ { w } + M _ { w } ) + b N _ { d } .\tag{13}
$$

The avoided replication is

$$
\Delta B = ( W - 1 ) ( S + b N _ { s } ) .\tag{14}
$$

The common static portion becomes $O ( S )$ rather than $O ( W S ) ;$ private state remains worker-dependent. All replayed delivery lookups resolve.   
The total reduction reflects both static sharing and retained private state.

## E Delivery-State Invariants and Operational Proofs

The delivery rules establish budget safety, neutral utility under unresolved feedback, and bounded historical influence. This appendix states the underlying records and derives these properties independently of task quality.

State record. Each item has lifecycle state $\sigma _ { t } \in$ {Dormant, ActiveNextTurn, Cooldown, Retired} and record

$$
\begin{array} { r } { m _ { t } ( c ) = \langle \sigma _ { t } , \omega _ { t } , d _ { t } , f _ { t } , e _ { t } ^ { \mathrm { l a s t } } , k _ { t } ^ { \mathrm { l a s t } } , u _ { t } \rangle . } \end{array}\tag{15}
$$

Here $\omega _ { t }$ marks pending feedback, $d _ { t }$ and $f _ { t }$ are last delivery and feedback steps, and the remaining keys identify delivered evidence and recovery episode. Delivery activates the next-turn hint and sets feedback pending. Exit evidence can move the item to cooldown while feedback remains pending. Feedback clears the pending flag and records arrival time. Retirement is an explicit lifecycle operation.

Budget safety. Let $S _ { j }$ be cumulative charged guidance after accounting event $j ,$ initially zero. Soft hints and added hard references accept cost $b _ { j } > 0 \mathrm { o n l y }$ when $S _ { j } + b _ { j } \leq B _ { 0 }$ . Acceptance sets $S _ { j + 1 } = S _ { j } + b _ { j }$ ; abstention leaves it unchanged. Induction gives $0 \leq S _ { j } \leq B _ { 0 }$ yielding Equation (10) after grouping events by actor step.

Operational budget checks. Table 17 connects runtime records to the invariant: none of the 1,212 attempts with budget records exceeds the delivery budget. This checks the shared accounting path; Section 5.3 evaluates history governance’s efects on repeat delivery and task cost through a paired ablation. Host verification waits are counted separately.

<table><tr><td>Main-study records</td><td>AppWorld</td><td>WebShop</td></tr><tr><td>Registered attempts</td><td>1,008</td><td>300</td></tr><tr><td>Soft hints delivered</td><td>2,276</td><td>462</td></tr><tr><td>Attempts with budget records</td><td>993</td><td>219</td></tr><tr><td>Over-budget deliveries in these records</td><td>0</td><td>0</td></tr></table>

Table 17: Corresponding delivery and budget records from the two main experiments. Over-budget counts refer to the recorded attempts listed in each column.

Neutral utility under unresolved feedback. Observed progress maps to +1; regression or adoption without progress maps to -1. Unknown, ignored, and no-new-evidence outcomes take the identity branch of Equation (9). Any number of unresolved observations leaves utility unchanged, while lifecycle metadata may advance.

Bounded historical influence. Starting at zero, each informative update is a convex combination of a value in [-1,1] and a reward in $\{ - 1 , + 1 \}$ ; the identity branch preserves the range. For eligible $c _ { 1 } , c _ { 2 }$

$$
\left| \left( \left[ v _ { t } ( c _ { 1 } ) + \lambda u _ { t } ( c _ { 1 } ) \right] - \left[ v _ { t } ( c _ { 2 } ) + \lambda u _ { t } ( c _ { 2 } ) \right] \right) - \left[ v _ { t } ( c _ { 1 } ) - v _ { t } ( c _ { 2 } ) \right] \right| \leq 2 \lambda .\tag{16}
$$

History therefore perturbs any pairwise priority gap by at most 2λ. Evaluation retains $\lambda = 0 . 1$ . Since candidates with $p _ { \theta } ( \mathrm { H e l p f u l } ) > 0 . 9 0$ have $v _ { t } \in ( 0 . 8 , 1 ]$ , the history term reorders within the high-confidence pool, while the independent eligibility predicate enforces suitabilit and delivery constraints. History values cannot change whether a candidate passes eligibility.

Recurrence and reconstruction. An item delivered at $d _ { t }$ with feedback at $f _ { t }$ requires

$$
t \ge \operatorname* { m a x } ( d _ { t } , f _ { t } ) + \tau _ { \mathrm { c o o l } } ,\tag{17}
$$

no pending feedback, a non-retired state, and a new evidence/episode pair. New episodes do not bypass cooldown. At most one soft hint is delivered per step. Ofline reconstruction reads only prior lesson publications: identical visible prefixes produce identical candidate inputs despite diferent future publications.

## F Cross-Actor Transfer

We evaluate cross-family transfer by deploying the frozen control plane with Llama-3.1-8B-Instruct. Base-Llama runs without recovery; Synapse-Llama attaches the existing domain-specific Qwen 1.7B classifier and micro-contract interface. The classifier weights, admission thresholds, and initial contract bank are reused without retraining. In this integration, the actor, verifier, and lesson updater share the same generative service and all use Llama; their role prompts remain distinct. The experiment therefore tests transfer of the frozen controller to a Llama-based recovery runtime.

Paired protocol. Each benchmark registers 54 attempts per arm, giving 216 attempts under the 60-action protocol. Quality comparisons use jointly scored task pairs: 47 in AppWorld and 53 in WebShop. Table 18 reports paired success counts and mean benchmark scores, together with score-normalized resource use. For each resource measure C, the index $I _ { C } = C _ { \mathrm { r e g i s t e r e d } } / \overline { { s } } _ { \mathrm { p a i r e d } }$ divides resource use over all 54 registered attempts, including execution errors, by the displayed paired mean score on the 0–100 scale. Each index expresses the resource use of the fixed workload per attained mean-score point. The same definition applies to requests, input tokens, output tokens, and task time in both benchmarks.

<table><tr><td rowspan="2">Metric</td><td colspan="2">AppWorld</td><td colspan="2">WebShop</td></tr><tr><td>Base-Llama</td><td>Synapse-Llama</td><td>Base-Llama</td><td>Synapse-Llama</td></tr><tr><td>Registered attempts</td><td>54</td><td>54</td><td>54</td><td>54</td></tr><tr><td>Jointly scored pairs</td><td>47</td><td>47</td><td>53</td><td>53</td></tr><tr><td>Successes (paired)</td><td>0/47</td><td>0/47</td><td>0/53</td><td>3/53</td></tr><tr><td>Mean benchmark score (paired)</td><td>16.18</td><td>16.18</td><td>3.09</td><td>14.34</td></tr><tr><td>Requests / mean-score point</td><td>197.0</td><td>211.4</td><td>999.0</td><td>198.6</td></tr><tr><td>Input tokens / mean-score point (M)</td><td>2.40</td><td>2.30</td><td>4.94</td><td>1.16</td></tr><tr><td>Output tokens / mean-score point (K)</td><td>47.9</td><td>37.9</td><td>33.1</td><td>7.9</td></tr><tr><td>Task time / mean-score point (s)</td><td>327.8</td><td>294.9</td><td>395.4</td><td>102.7</td></tr><tr><td>Sidecar scoring batches</td><td></td><td>350</td><td></td><td>139</td></tr><tr><td>Repeated guidance deliveries</td><td></td><td>69</td><td></td><td>4</td></tr></table>

Table 18: Frozen-controller transfer: task outcomes and score-normalized resource use. Cost rows divide the fixed 54-attempt workload’s resource use by its paired mean benchmark score (I ). M and K denote millions and thousands of tokens; dashes indicate no guidance mechanism.

WebShop task gains and score-normalized eficiency. On the 53 jointly scored WebShop pairs, Synapse-Llama completes three tasks versus none for Base-Llama and raises the mean benchmark score from 3.09 to 14.34 (+11.25 points). Input tokens per mean-score point decrease by 76.5%, from 4.94M to 1.16M, and task time per mean-score point decreases by 74.0%, from 395.4 to 102.7 seconds. Th request index decreases by 80.1%, and output tokens per mean-score point by 76.0%. The frozen controller supports higher task yield and lower score-normalized resource use under a diferent model family without additional distillation.

AppWorld execution under the fixed action budget. AppWorld places greater demands on the actor’s ability to complete and terminate multi-API workflows. Neither arm fully solves a task among its 54 registered attempts. The runs end at the 60-action cap or with an execution error, with no natural termination. This pattern is consistent with limited workflow-completion capability in the evaluated Llama configuration: local contract guidance does not replace the actor’s multi-step plan or termination decision. On the 47 jointly scored pairs, both mean scores are 16.18. Synapse-Llama reduces input tokens per mean-score point by 4.2% (2.30M versus 2.40M), output tokens by 20.8% (37.9K versus 47.9K), and task time by 10.0% (294.9 versus 327.8 seconds) under the same normalization. The observed benefit is lower score-normalized token use and task time at the same reported paired score.