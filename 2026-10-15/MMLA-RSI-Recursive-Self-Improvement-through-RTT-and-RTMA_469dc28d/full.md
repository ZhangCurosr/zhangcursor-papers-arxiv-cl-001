# MMLA-RSI: Recursive Self-Improvement through RTT and RTMA

Technical Report of MMLA

Junyi Zou1,\* • Avrova Donz1,2,\*

<sup>1</sup>MMLA-org <sup>2</sup>Communication University of China (CUC), No. 1 Dingfuzhuang East Street, Chaoyang District, Beijing 100024, China <sup>\*</sup>Equal contribution

Email: lior.j.zou@gmail.com · avrovadonz@icloud.com

October 14, 2026

## ABSTRACT

An agent can improve its behavior within an unresolved problem by changing how it acts and which experience it retains. We propose MMLA-RSI, a recursive self-improvement scheme with two implementations in Memory-Mediated Learning Architecture: Reasoning-Time Training (RTT) changes the policy state Φ, while Reasoning-Time Memory Adaptation (RTMA) changes authoritative memory �. Each route also makes its adaptation method an object of improvement. A current numerical updater trains the policy proposing its successor; a current memory manager chooses the records available to the process proposing its successor. An independently admitted successor takes over the same role in the next generation.

We give bounded executable rule languages, shared task/proposal interfaces, and one algorithm for either route or their composition. The RTT instance changes scalar schedules in a clipped-momentum update. The RTMA instance chooses NULL or a complete-row write with possible eviction. A fixed interpreter, trusted committer, and independent evaluator bound what can change. State versions, read closures, and conditional properties describe causal execution, accepted transitions, and finite work

Existing numerical-component studies motivate the construction. In four-candidate program selection, retaining LARC feedback updates reduces expected error by 24.65 and 36.65 percentage points for two trained initializations. In chronological continuous-integration replay, feedback binding is positive while retention increases loss. Completed SQL studies improve substantially over an unadapted substrate but do not establish the required additional online benefit. These results characterize their declared components. The two recursive routes and their joint realization are specified here; their beneficial recursive chains and discovery eficiency remain to be evaluated.

## MMLA-RSI: learning through policy and memory

![](images/6b6346b4ad910c355c52a776f8a70b23134629d9658409a401744fcf8bd4dbab.jpg)  
Figure 1: Two state paths, two recursive implementations. RTT changes Φ through a numerical updater; RTMA changes � through a memory manager. In each route, the current rule changes the proposal process that produces its successor. Admission installs that successor for the next generation. The routes can operate separately or share an unresolved problem.

## 1 Introduction

An unsuccessful attempt supplies information that the next attempt may need. A program fails an execution check, a query violates a visible constraint, or a candidate learning rule performs poorly on a development task. The agent must decide both how this feedback changes its behavior and which parts remain available later. Memory-Mediated Learning Architecture (MMLA) makes these two learning objects explicit: policy state Φ and authoritative memory � [30].

Reasoning-Time Training (RTT) updates Φ while a problem remains unresolved, then uses that state in a later attempt. Reasoning-Time Memory Adaptation (RTMA) commits a non-NULL successor of � that a later attempt reads. The distinction follows the MMLA v4 state account. Here both paths are central. Learning a useful memory is one level of adaptation; improving the method that selects and maintains that memory is another. The same distinction holds between updating a policy state and improving its updater.

MMLA-RSI connects each adaptation method to the process producing its own successor. In the RTT implementation, the current numerical updater trains a private proposal policy. In the RTMA implementation, the current manager chooses the authoritative records that the proposal process can read. A locked successor is evaluated independently and, if admitted, replaces its parent in the same proposal-learning role:

$$
U _ { \eta _ { k } ^ { \Phi } } \longrightarrow \Phi _ { j + 1 } ^ { \mathrm { p r o p } } \longrightarrow \eta _ { k + 1 } ^ { \Phi } \longrightarrow \mathrm { n e x t } \mathrm { p r o p o s a l } \mathrm { l e a r n i n g } \mathrm { w i t h } U _ { \eta _ { k + 1 } ^ { \Phi } } ,
$$

$$
V _ { \eta _ { k } ^ { M } } \longrightarrow M _ { j + 1 } ^ { \mathrm { p r o p } } \longrightarrow \eta _ { k + 1 } ^ { M } \longrightarrow \mathrm { n e x t p r o p o s a l ~ l e a r n i n g ~ w i t h ~ } V _ { \eta _ { k + 1 } ^ { M } } .\tag{1}
$$

The middle arrows include generation, selection, and independent admission. The inherited objects are the rules $\eta ^ { \Phi }$ and $\eta ^ { M }$ proposal states may restart from fixed initializations each generation. A joint instance carries a pair of rules through the same cycle. Changing a state alone is adaptation. A recursive improvement chain additionally requires the changed adaptation method to participate functionally in producing a later improved method.

This dependency has direct precedents. Self-referential meta learning studies systems that modify their learning behavior [9]; STOP executes an improver on itself, and Darwin Gödel Machine uses descendant agents in further self-modification [25, 27]. Our construction specifies the dependency through MMLA’s two state paths: bounded numerical update programs and bounded memory-management programs, their task/proposal interfaces, and independent admission before role replacement. It makes both the changed object and its later use inspectable.

MMLA is mixer-agnostic. Low-Rank Adaptive Residual Connections (LARC) provide the current numerical carrier in front of a reused frozen mixer [29]. The Transformer mixer and low-rank factors are implementation choices. Φ denotes a policy-state role, not a synonym for adapter parameters; another declared fast state can fill it. � is the committed authority for retained records, not an undiferentiated name for prompts, logs, or caches. These distinctions let the same architecture support policy adaptation, memory adaptation, and their composition.

The method contribution is a complete recursive construction with two concrete bounded domains. The RTT domain varies feedback-dependent schedules in a clipped-momentum template. The RTMA domain varies the rule selecting NULL or a complete-row write under finite memory capacity. We specify their state, feedback, failure, read, and version semantics, then give a common algorithm and its conditional properties. Existing component studies motivate separate measurements of functional participation, task benefit, and discovery eficiency. The proposed method, an observed beneficial recursive chain, and an advantage over fixed search are distinct levels of evidence

## 2 The MMLA State, RTT, and RTMA

## 2.1 Three timescales and two learning objects

It is helpful to separate an attempt, a problem, and an improvement generation. Attempt � produces a proposal for a fixed problem $( X , I )$ , where � is its input and � is its immutable task identity. Feedback from that attempt may change the private state before attempt $j + 1$ . An improvement generation � binds an adaptation-rule version $\eta _ { k }$ across a collection of such problems. It can end by installing a new version for future problems.

The process state extends the state account in the MMLA report:

$$
\begin{array} { r } { S _ { j } = \big ( X , I , C _ { j } , M _ { j } , W _ { j } , \Phi _ { j } , \Xi _ { j } ^ { \mathrm { r e a d } } , \Xi _ { j } ^ { \mathrm { a u x } } , b _ { j } \big ) , \qquad \Psi _ { j } ^ { \Phi } = \big ( \Phi _ { j } , \Xi _ { j } ^ { \mathrm { r e a d , \Phi } } \big ) . } \end{array}\tag{2}
$$

Here � is working context, � is authoritative memory, � is external world state, and � is remaining resource budget. $\Xi ^ { \mathrm { r e a d } }$ contains auxiliary state capable of afecting a later action, including an optimizer when it participates in a subsequent update, a random-number state, a tool cache, or a pending callback. $\Xi ^ { \mathrm { a u x } }$ contains records that cannot influence the measured continuation. Calling a log “auxiliary” requires establishing that the continuation cannot read it.

The corresponding memory closure is $\Psi ^ { M } = ( M , \Xi ^ { \mathrm { r e a d , M } } )$ . The joint closure includes the union of dependencies, counted once when shared. Each controlled kernel below uses the appropriate declared closure Ψ. Slow initialization $\rho$ and initial memory $M _ { 0 }$ say where a learner begins; $\eta ^ { \Phi }$ and $\eta ^ { M }$ say how feedback changes it. A run fixes its mutable domain to the RTT

rule, the RTMA rule, or their pair. The pretrained substrate, language interpreters, memory authority checks, and evaluator remain bound outside that domain.
<table><tr><td>Object</td><td>Operation and lifetime</td><td>Restoration or transition</td></tr><tr><td> $\theta _ { 0 } , \omega$ </td><td>Bound pretrained substrate and shared offline adaptation</td><td>Verify original identity; never recreate by retraining.</td></tr><tr><td> $\rho$ </td><td>Slow initialization of the numerical carrier</td><td>Copied at problem start; bound for each comparison.</td></tr><tr><td> $\Phi$ </td><td>Private state changed by feedback in the current problem</td><td>Reset to the problem&#x27;s bound  $\rho .$ </td></tr><tr><td> $\eta ^ { \Phi } , \eta ^ { M }$ </td><td>Numerical updater and memory-management rule</td><td>The selected domain changes only by admitted version replacement.</td></tr><tr><td>M</td><td>Bounded authoritative records with verified complete-row commits</td><td>Starts from bound  $M _ { 0 } ;$  independently retained, reset, or held fixed.</td></tr><tr><td>read</td><td>Every additional dependency of a future action</td><td>Restored with the intervention&#x27;s declared closure.</td></tr><tr><td>W, b</td><td>External effects and resource consumption</td><td>Effects and consumed resources survive a rollback.</td></tr></table>

Table 1: Ownership and lifetime. Policy learning, memory learning, and changes to their adaptation methods are distinct operations.

## 2.2 Feedback must arrive before it can teach

Let an attempt finish at $\epsilon _ { j } ,$ its feedback become visible at $\nu _ { j } ,$ and an update commit at $u _ { j } .$ Let $\sigma _ { j + 1 }$ be the start of the next fresh proposal and � the resolution time of the current problem. The required event order is

$$
\epsilon _ { j } \leq \nu _ { j } \leq u _ { j } < \sigma _ { j + 1 } < \tau .\tag{3}
$$

An update after resolution may improve another problem, but it does not establish the within-problem RTT or RTMA path for this one. Reranking candidates generated before an update does not show that their content was generated by the updated state. A newly computed selection action can nevertheless depend functionally on that state; its action space is candidate selection rather than candidate-content generation.

Let $( \mathcal { F } _ { t } ) _ { t }$ be the filtration of legally available information. The updater has the form

$$
\begin{array} { r l r } { ( \Phi _ { j + 1 } , s _ { j + 1 } ) = \Pi _ { \mathcal { K } ( b _ { j } ) } U _ { \eta _ { k } ^ { \Phi } } ( \Phi _ { j } , s _ { j } , r _ { j } ; \zeta _ { j } ) , } & { } & { r _ { j } \mathrm { ~ i s ~ } \mathcal { F } _ { \nu _ { j } \mathrm { - } } \mathrm { m e a s u r a b l e } , } \end{array}\tag{4}
$$

where $r _ { j }$ is the arrived record, $s _ { j }$ is private update state, $\zeta _ { j }$ is permitted randomness, and $\Pi _ { \mathcal { K } ( b _ { j } ) }$ enforces the declared feasible update set. The record can contain an attempt, its execution response, and a derived gradient. It cannot contain the answer to a later independent evaluation. An ofline teacher can supply targets when training $\rho ;$ its presence must not be mistaken for online feedback available to $U _ { \eta }$ at deployment.

![](images/097621960c5f6fb4e6a42a1fb47599a20f0118ca42f566a3cbad5751ee48cc1b.jpg)  
Figure 2: The shared timing condition. The unresolved task persists from attempt to feedback, commit, and a later pinned read. RTT commits a policy-state update; RTMA commits a non-NULL memory successor. Independent future audit outcomes remain outside both paths.

The subsequent action kernel averages over a declared source of continuation randomness. To distinguish this object from a replay, write $F _ { \bar { \Psi } } ( h , Z )$ for the executed action, where ℎ fixes the reached task history and arrived inputs, $\bar { \Psi }$ fixes the non-random read state, and $Z \sim \nu$ is the continuation random tape. RNG components of $\Xi ^ { \mathrm { r e a d } }$ are conditioned on in a replay or marginalized under the declared law in a policy comparison; the same random tape is not simultaneously fixed and marginalized. Thus

$$
K ( A \mid h , \bar { \Psi } ) = \int \mathbf { 1 } \{ F _ { \bar { \Psi } } ( h , z ) \in A \} \nu ( d z ) .
$$

At a reached history, functional use in the marginal action-distribution sense requires

$$
\mathrm { T V } \big ( K ( \cdot \ | \ h , \bar { \Psi } _ { j + 1 } ) , K ( \cdot \ | \ h , \bar { \Psi } _ { j } ) \big ) > 0 ,\tag{5}
$$

with non-target inputs and the random-tape law held fixed. The action space must also be declared: a token, a complete proposal, and a selected program define diferent kernels.

A common-random-number replay compares the conditional execution mappings $F _ { \bar { \Psi } _ { i + 1 } } ( h , z )$ and $F _ { \bar { \Psi } _ { i } } ( h , z )$ . A changed output witnesses sensitivity at that fixed tape. It does not in general prove Equation 5, because for this coupling

$$
\mathrm { T V } ( K _ { j + 1 } , K _ { j } ) \ \le \ \operatorname * { P r } _ { Z \sim \nu } \Bigl [ F _ { \bar { \Psi } _ { j + 1 } } ( h , Z ) \not = F _ { \bar { \Psi } _ { j } } ( h , Z ) \Bigr ] \ .
$$

Thus replay disagreement alone does not establish a marginal diference.<sup>1</sup> When the evaluated configuration is deterministic and � is a point mass, diferent outputs do prove a diference between those conditional kernels.

Evidence for the marginal criterion can instead compare exact action probabilities on a finite action space, or a fixed measurable event whose probability difers with appropriate uncertainty. At a fixed generation prefix, a changed next-token probability is one such event for the corresponding continuation kernel. Replay sensitivity, estimated distributional dependence, and exact distributional dependence are reported under their respective objects. The saved SQL output changes below remain replay observations.

## 2.3 What an RTT witness must record

Table 2 summarizes the parent report’s eight obligations for an inspectable RTT transition [30].

Witness Required evidence   
W1 The input and task identity remain the same and the problem is unresolved.   
W2 Feedback is bound to a completed attempt and has arrived before the update.   
W3 A genuine learning-state change is committed, with a version and read closure.   
W4 The update is causal, belongs to the current epoch, and does not use future information.   
W5 The commit precedes a fresh proposal and both occur before resolution.   
W6 The next proposal actually loads the committed state version.   
W7 A controlled intervention shows functional action-kernel dependence at a reached history.   
W8 Update, read, cost, lifetime, reset, and late-callback behavior are recorded.  
Table 2: The RTT witness. W1–W6 and W8 establish the process; W7 adds functional use. Task benefit is a separate empirical quantity.

The distinction between functional use and benefit is essential. An update can change the next output and make it worse. Real feedback can be less damaging than permuted feedback while both are worse than resetting. Conversely, a strong starting policy can solve a task without requiring a useful fast update. Section 7 contains examples of all three issues in the component research program.

## 2.4 Authoritative memory and the RTMA path

The memory path follows MMLA v4's definitions of RTMA and dual-state composition [30, Section 72, Definitions 72.2–72.3]. Its transition separates a proposed action from an authorized commit:

$$
a _ { j } ^ { M } = V _ { \eta _ { k } ^ { M } } ( M _ { j } , d _ { j } ) , \qquad ( M _ { j + 1 } , \nu _ { j + 1 } ^ { M } ) = \mathrm { C o m m i t } ( M _ { j } , a _ { j } ^ { M } , d _ { j } ) .\tag{6}
$$

The fixed assembler constructs the candidate record $d _ { j }$ from arrived evidence. The manager chooses an allowed action. The trusted committer checks its type, provenance, capacity, expected predecessor, and epoch before installing a complete successor. NULL leaves authoritative content and its version unchanged and produces an explicit no-op receipt. In the minimal instance, a non-NULL action writes one complete row into a selected slot, evicting its previous row if occupied. The later attempt pins the actual successor version. A failed pin, an unread write, and a NULL receipt do not establish the positive RTMA path.

Authority concerns which committed records the reader treats as its stored evidence. It does not turn a stored hypothesis into a fact. A row can authoritatively record that an attempt received a particular execution response, with the response’s scope and provenance, without claiming the attempted answer was correct. The manager cannot rewrite the evaluator, forge provenance, or enlarge its own access permissions.

The two paths have independent receipts. RTT requires the policy update/read and functional-use obligations in Table 2. RTMA requires a non-NULL complete-row commit and later same-problem read; functional memory dependence is additionally tested by intervening on the memory closure with Φ fixed. A joint interval includes both paths and their interleaving. Neither classification implies positive task benefit.

A KV cache belongs to the state role determined by its production and use. A cache deterministically rebuilt from an unchanged transcript is working context or a read cache. An explicitly updated, versioned policy-state carrier may serve as Φ if it satisfies the declared update/read semantics. It is not authoritative � merely because it stores tensors, and changing it does not automatically establish RTT.

## 3 Numerical Carrier and Initialization

## 3.1 The RTT carrier: a slow initialization and a private fast residual

The present carrier inserts a low-rank residual before a frozen mixer:

$$
T _ { \Phi } ( h ) = h + B _ { \Phi } A _ { \Phi } h , \quad A _ { \Phi } \in \mathbb { R } ^ { r \times d } , \quad B _ { \Phi } \in \mathbb { R } ^ { d \times r } , \quad f _ { \theta , \Phi } ( x ) = F _ { \theta } \big ( T _ { \Phi } ( E _ { \theta } ( x ) ) \big ) .\tag{7}
$$

The factorization follows LoRA [8]; the input-side placement, slow/fast state lifetime, and feedback experiments are described in the LARC report [29]. With $d = 1 5 3 6$ and $r = 4 .$ , the two factors have 12,288 entries. Random nonzero � and zero � give an initial zero residual. Setting both factors to zero would remove the first task-gradient path to both factors. At the intended initialization, the first task gradient of � is zero while the gradient of � can be nonzero.

After slow training, $\rho = ( A _ { \rho } , B _ { \rho } , \nu _ { \rho } )$ is the starting state, where $\nu _ { \rho }$ identifies the initialization version. A new problem copies these factors into Φ. Reset restores this trained state, rather than a newly randomized � or a bare zero �. A complete resume additionally restores the bound updater, private optimizer, random streams, and input cursor. Equal products �� need not imply equal subsequent factor-space updates; storing the product alone therefore cannot reproduce a learning trajectory.

Frozen weights still participate in the derivative with respect to an input residual. Wrapping the entire downstream mixer in a no-gradient region would break this path. Similarly, a cached mixer output generally depends on Φ even though the mixer weights are frozen. Only representations before the first Φ-dependent operation can be reused without including the fast-state version in the cache identity. Batched problems share the same slow version but own separate fast factors, optimizer state, and random streams; attention masks and positions must preserve those boundaries.

## 3.2 Two ways to train the starting state

For an episode with support � and an outer-training query �, let $\Phi ^ { ( 0 ) } = \rho$ and let $\Phi ^ { ( m ) } = U _ { \eta } ^ { m } ( \rho ; S )$ . Two initialization objectives are

$$
J _ { \mathrm { s t a t i c } } ( \rho ) = \mathbb { E } _ { S , Q } \mathcal { L } _ { Q } ( \rho ) ,\tag{8}
$$

$$
J _ { \mathrm { a d a p t e d } } ( \rho ) = \mathbb { E } _ { S , Q } \mathscr { L } _ { Q } \big ( U _ { \eta } ^ { m } ( \rho ; S ) \big ) .\tag{9}
$$

In the paired program-selection experiment, both arms have the same residual placement, support, query supervision, and three paired initializations. The adapted arm uses a first-order MAML-style estimator [6, 13]:

$$
\widehat { g } _ { \rho } = \left. \nabla _ { \Phi } \mathcal { L } _ { Q } ( \Phi ) \right| _ { \Phi = U _ { \eta } ^ { m } ( \rho ; S ) } .\tag{10}
$$

This assigns the adapted query gradient to corresponding slow factors, approximating the inner-update Jacobian by the identity. It is neither diferentiation through the exact full inner trajectory nor a claim that a detached fast gradient automatically updates $\rho .$ The implementation must explicitly aggregate episode gradients and apply them to the bound outer parameters.

Learning $\rho$ with a fixed $U _ { \eta }$ prepares an adaptable starting point. It does not change the updater. Even learning � using a fixed external meta-optimizer need not create the recursive dependency in Equation 1. The proposed next layer uses the current $U _ { \eta }$ in the process that proposes its replacement. Keeping these objects separate prevents an ordinary initialization experiment from being counted as an experiment on recursive improvement.

## 4 Measuring Feedback and Retention

## 4.1 The complete read closure

A clean intervention specifies everything that can reach the measured continuation. Removing feedback from the visible prompt is insuficient if a selector reads a feedback log, an optimizer resumes with feedback-dependent momentum, or an asynchronous tool response arrives after a reset. The read closure includes all such paths. For a Φ-only comparison, every other causal input must be identical across the paired reads. If auxiliary state is changed together with Φ, the estimand is the efect of the bundle $\Psi ^ { \Phi }$ in Equation 2.

The program-selection study uses a particularly narrow read: the next query sees the bound model and residual but no support-feedback text, inner optimizer, or post-update cache. Memory is disabled. Later SQL studies use broader task-matched reads, including shared support and feedback context and selectors. Their Φ efects are conditional on these channels. Disabling � alone does not make such a continuation the memory-disabled, feedback-text-free read of the earlier study.

![](images/f7f4a4964d7bfef9bfb8a36f3b8287abb105b0255ab95b58254ac98695abe990.jpg)  
Figure 3: Two controlled state paths and their read closure. Purple marks the policy path; teal marks authoritative memory. Context, caches, selectors, random streams, and callbacks are additional dependencies. Each intervention specifies its target and the other paths held fixed.

## 4.2 Policy retention and feedback binding

Let $L _ { f k }$ be expected loss when feedback binding is $f \in \{ \mathrm { r e a l } , \mathrm { s h a m } \}$ and retention is $k \in \{ \mathrm { k e e p } , \mathrm { r e s e t } \}$ . Sham feedback independently applies a uniformly sampled permutation over the declared eligible feedback positions. It retains the feedback values while breaking their binding; an identity permutation and an uninformative record remain legitimate draws. Both cells perform the registered updates. Reset occurs after those updates and before the paired continuation, restoring the arm’s own $\rho$ and the specified closure.

$$
\begin{array} { r } { G = L _ { \mathrm { r e a l , r e s e t } } - L _ { \mathrm { r e a l , k e e p } } , } \end{array}\tag{11}
$$

$$
D = G - \left( L _ { \mathrm { s h a m , r e s e t } } - L _ { \mathrm { s h a m , k e e p } } \right) .\tag{12}
$$

� measures the value of retaining the real-feedback state relative to resetting. � measures the additional retention efect attributable to the real binding. If both reset distributions coincide, $D = L _ { \mathrm { s h a m , k e e p } } - L _ { \mathrm { r e a l , k e e p } }$ . For a higher-is-better score, all loss contrasts are reversed consistently. Positive � does not imply positive �.

Complete reset equality is a useful implementation consequence. If the restored causal state, future exogenous-input distribution, and action kernels are identical, the continuation distributions agree. Pathwise equality additionally requires the same coupled randomness and deterministic realization of the remaining operations. This equality concerns the declared continuation, not external world efects that an earlier action has already caused. A transaction can restore numerical state without undoing a sent request or refunding its computation.

## 4.3 Memory efects and joint task efects

For memory, compare a committed write with NULL from the same reached state, with the same arrived information and continuation budget. Recompute the complete future task outcome under each branch: the write can add useful information and remove useful old information. A memory reset restores the bound pre-intervention snapshot or $M _ { 0 }$ , as registered, together with its read caches and index versions. It is a diferent intervention from rejecting one proposed write. A memory-rule comparison gives an exact selector or strong fixed heuristic the same legal information, action set, capacity, and total computation allowance.

A joint evaluation uses policy keep/reset and memory keep/reset as separate factors, rather than relabeling the real/sham × policy-retention matrix. For the local retention probe defined here, branch from one common unresolved checkpoint after the registered feedback updates and before the next read. Keep uses that checkpoint’s carrier state. Policy reset restores the bound $\rho$ and its initial private optimizer state; memory reset restores the task’s bound $M _ { 0 }$ and its index state. Invalidate or rebuild dependent caches for each resulting pair and verify its new joint pin. Hold the other reached inputs, external state, continuation random-tape law, and resource allowance fixed. Freeze further Φ and � adaptation during the declared finite continuation and evaluate the same endpoint in all four cells. Let $L _ { p m }$ be its loss with policy retention $p \in \{ 0 , 1 \}$ and memory retention � ∈ {0, 1} under one declared feedback condition. Then

$$
G _ { \Phi } ( m ) = L _ { 0 m } - L _ { 1 m } , \qquad G _ { M } ( p ) = L _ { p 0 } - L _ { p 1 } , \qquad I = L _ { 0 1 } + L _ { 1 0 } - L _ { 1 1 } - L _ { 0 0 } .
$$

These are state-retention efects conditional on the reached checkpoint. Resetting one carrier does not remove its earlier influence on the other carrier, context, or world. A probe that resumes both adaptation rules defines a diferent continuation estimand; its efects can propagate through subsequent cross-state updates. Disabling a route from initialization instead defines a complete-trajectory efect. Each comparison binds its own reset and continuation specification.

� is the extra loss reduction of keeping both beyond the sum of their separate reductions; equivalently, $\begin{array} { r } { I = G _ { \Phi } ( 1 ) - G _ { \Phi } ( 0 ) = } \end{array}$ $G _ { M } ( 1 ) - G _ { M } ( 0 )$ . Joint usefulness is judged by the complete task endpoint against same-information strong and frozen methods; neither two positive main efects nor a positive interaction is required merely to define the composed mechanism. Feedback-binding interventions can be added as a separately declared factor. The component values of � below remain policy binding efects in their original read conditions, not estimates of �.

If Φ, its updater, or the memory reader changes, a cached memory-value estimate may no longer describe the new continuation. Such caches must bind all causal reader identities and be invalidated or recomputed after a mismatch. Repricing, retrieval, write selection, eviction, and restoration enter the task cost. The minimal memory manager below uses directly observed row features rather than a learned value cache; a predictive value estimator is a possible larger domain, not a prerequisite for the RTMA construction.

## 4.4 From a component efect to an RSI efect

An adapted initialization can outperform another initialization without gaining anything from its own feedback. The objective increment and each arm’s real-feedback increment therefore appear separately in the results. The recursive evaluation likewise reports parent participation and accepted task improvement separately from discovery eficiency; Section 5.9 gives the corresponding interventions and comparisons.

Each comparison uses the original unit of generalization. Multiple seeds applied to one task group are repeated measurements within that group; they are not additional independent tasks. Selection uses development groups, and a locked candidate is evaluated on separate audit groups. Once an audit outcome influences a successor’s design, that group becomes part of the development history for subsequent statistical claims. New accepted generations require fresh audit information under the stated admission assumptions.

## 5 Two Recursive Implementations and Their Composition

Fix the mutable domain before a run: $\mathcal { H } _ { \Phi }$ for RTT, $\mathcal { H } _ { M }$ for RTMA, or $\mathcal { H } _ { \Phi } \times \mathcal { H } _ { M }$ for a joint instance. We use $\eta$ for the selected rule or pair. In every case the parent changes a state read by the process proposing its successor, and the admitted successor takes over that same role. The numerical and memory languages below instantiate this construction with finite, explicit operations.

## 5.1 RTT: a bounded numerical updater

In the next three RTT subsections, � abbreviates $\eta ^ { \Phi }$ and H abbreviates $\mathcal { H } _ { \Phi }$ . The updater is a bounded program with a typed input and output:

$$
\begin{array} { r } { \big ( \Delta \Phi _ { j } , s _ { j + 1 } \big ) = \mathrm { E v a l } _ { \mathcal { H } } ( \eta ; \Phi _ { j } , s _ { j } , g _ { j } , z _ { j } ) , \qquad \Phi _ { j + 1 } = \Pi _ { \mathcal { K } } ( \Phi _ { j } + \Delta \Phi _ { j } ) . } \end{array}\tag{13}
$$

$g _ { j }$ is a feedback-derived gradient and $z _ { j }$ contains arrived scalar statistics. The minimal instance below varies feedback-dependent step sizes, gradient-clipping thresholds, and momentum coeficients within a fixed clipped-momentum template. The state layout, gradient input, projection, and interpreter remain fixed. This gives a concrete bounded domain for the recursive construction.

Flatten the chosen LARC factors in row-major order, first � and then �, into $\Phi \in \mathbb { R } ^ { q }$ . The language’s private state is one momentum vector $s \in \mathbb { R } ^ { q }$ in the same coordinate order, initially zero. A program is the canonical prefix serialization of three scalar expression trees, $\boldsymbol { \eta } = \left( e _ { a } , e _ { \kappa } , e _ { \mu } \right)$ . Each tree has the grammar

$$
e : : = \nu \mid z _ { i } \mid ( e + e ) \mid ( e - e ) \mid ( e e ) \mid \operatorname* { m i n } ( e , e ) \mid \operatorname* { m a x } ( e , e ) \mid \operatorname { s d i v } _ { \varepsilon } ( e , e ) ,
$$

where � belongs to a fixed finite set $_ \mathrm { ~  }$ of finite FP32 constants containing 0 and 1, and $i \in \{ 1 , . . . , 4 \}$ . Constants are serialized as indices into a bound ordered table of their exact FP32 bit patterns; operator names, delimiters, and index encodings are fixed. There are no free-form numerical literals. The product $_ { e e }$ denotes scalar multiplication. Safe division is

$$
\begin{array} { r } { \mathrm { s d i v } _ { \varepsilon } ( x , y ) = x / \big ( \mathrm { s i g n } _ { + } ( y ) \operatorname* { m a x } \{ | y | , \varepsilon \} \big ) , \quad \mathrm { s i g n } _ { + } ( 0 ) = 1 . } \end{array}
$$

The interpreter traverses trees in fixed left-to-right postorder. All scalar and vector operations, including norm reduction order and rounding, use the bound numerical implementation. Every intermediate is checked for finiteness. The constructor bounds total syntax nodes by $N _ { \mathrm { A S T } } \geq 4 _ { \mathrm { \Omega } }$ , each expression depth by $D \geq 0 ,$ and the tokenized serialization length by $L _ { \mathrm { A S T } } \geq 1$ Admission requires a nonempty grammar satisfying all three limits and containing the parent.

Let $\ell _ { j } \in [ 0 , 1 ]$ be the arrived feedback loss. Each learner starts with $j = 0$ and $\ell _ { - 1 } = 0$ . Here $j$ indexes feedback-bearing attempts, including invalid or incomplete actions assigned a failure loss; it does not count only successful numerical updates.

The task runner binds a cap $T ^ { \mathrm { t a s k } } \geq 1$ per task, while proposal learning uses $T ^ { \mathrm { p r o p } } = N$ attempts per generation and at most � − 1 updates. The same fixed statistics map uses the current role’s $T \colon$

$$
z _ { j } = \left( \ell _ { j } , \mathrm { \ c l i p } _ { [ - 1 , 1 ] } ( \ell _ { j } - \ell _ { j - 1 } ) , \mathrm { \ m i n } \{ 1 , \| g _ { j } \| _ { 2 } / G \} , \mathrm { \ m i n } \{ 1 , j / T \} \right) , \qquad G > 0 .
$$

The loss history and opportunity index belong to each learner’s arrived record and reset with that learner. The updater cannot read raw task identities, future targets, memory, or evaluation internals. The frozen parser and bounds turn the three expressions into coeficients:

$$
\begin{array} { r } { a _ { j } = \mathrm { c l i p } _ { [ 0 , a _ { \mathrm { m a x } } ] } e _ { a } ( z _ { j } ) , \quad \kappa _ { j } = \mathrm { c l i p } _ { [ \kappa _ { \mathrm { m i n } } , \kappa _ { \mathrm { m a x } } ] } e _ { \kappa } ( z _ { j } ) , \quad \mu _ { j } = \mathrm { c l i p } _ { [ 0 , \mu _ { \mathrm { m a x } } ] } e _ { \mu } ( z _ { j } ) . } \end{array}
$$

Here $0 < a _ { \mathrm { m a x } } < \infty , 0 < \kappa _ { \mathrm { m i n } } \leq \kappa _ { \mathrm { m a x } } < \infty , 0 \leq \mu _ { \mathrm { m a x } } < 1$ , and $\varepsilon > 0$ are fixed algorithm parameters. For $S _ { \mathrm { m a x } } , R _ { \Phi } \in ( 0 , \infty )$ the deterministic execution rule is

$$
\begin{array} { r l } & { \tilde { g } _ { j } = g _ { j } \operatorname* { m i n } \{ 1 , \kappa _ { j } / ( \| g _ { j } \| _ { 2 } + \varepsilon ) \} , } \\ & { \quad s ^ { + } = \mathrm { c l i p } _ { [ - S _ { \operatorname* { m a x } } , S _ { \operatorname* { m a x } } ] } \big ( \mu _ { j } s _ { j } + ( 1 - \mu _ { j } ) \tilde { g } _ { j } \big ) , } \\ & { \Delta \Phi _ { j } = - a _ { j } s ^ { + } , \qquad \Phi ^ { + } = \mathrm { c l i p } _ { [ \rho - R _ { \Phi } , \rho + R _ { \Phi } ] } ( \Phi _ { j } + \Delta \Phi _ { j } ) . } \end{array}
$$

Clipping of a vector is coordinatewise. $\rho$ is the bound starting vector for the current learner role. Evaluation commits $( \Phi ^ { + } , s ^ { + } )$ together. A nonfinite intermediate, malformed tree, exceeded instruction cap, or failed output check rejects the whole update and restores both $\Phi _ { j }$ and $s _ { j } ; $ neither tentative tensor can reach a later read. The deterministic updater consumes no random draws. Sampling and tool work already performed retain their RNG progression, observations, and charges. Reset of an episode and transactional rejection of one update are therefore diferent operations.

## 5.2 RTT: one interface for task and proposal learning

The task learner and proposal learner use the same factor layout and dimension �, with separately bound initializations $\rho ^ { \mathrm { t a s k } }$ and $\rho ^ { \mathrm { p r o p } }$ and private momentum vectors. The same syntax tree is interpreted in both roles without a role-dependent branch. Only the gradient and arrived feedback come from diferent problems. No momentum or fast factors are shared between task executions or between the two roles.

For a complete minimal instantiation, fix a task runner that samples an action $y _ { j } ^ { \mathrm { t a s k } }$ , receives a legal arrived feedback loss $\ell _ { j } \in [ 0 , 1 ]$ , and returns

$$
g _ { j } ^ { \mathrm { t a s k } } = \operatorname { s g } ( \ell _ { j } ) \nabla _ { \Phi ^ { \mathrm { t a s k } } } \log p _ { \Phi ^ { \mathrm { t a s k } } } ( y _ { j } ^ { \mathrm { t a s k } } \mid h _ { j } ) .
$$

This is the zero-baseline score-function estimate for minimizing that feedback loss; the probability is that of the actual sampling law. A task may instead bind a diferentiable feedback loss, but that replacement defines a diferent fixed runner. The runner specifies its observation space, resolution condition, maximum $T ^ { \mathrm { t a s k } }$ feedback opportunities, and final task loss in [0, 1]. Invalid or incomplete actions receive the runner’s fixed failure loss one and retain their actual sampling probabilities. These outcomes, ordinary validation failures, and tool-reported errors provide legal feedback: the runner continues its feedback/update cycle unless its resolution condition or $T ^ { \mathrm { t a s k } }$ cap has been reached. Candidate updaters never receive the independent final audit answer as within-task feedback.

The proposal learner receives the development evaluation of a candidate program and uses Equation 14. Its $\ell _ { j }$ is the negative of the bounded proposal reward defined below. Both roles thus return the same typed tuple $( \Phi , s , g , z )$ to $U _ { \eta }$ . Type compatibility follows by construction; useful transfer of an updater between the two gradient distributions is an empirical question. The constant-tree program gives clipped momentum SGD, while feedback-dependent trees change its step size, clipping threshold, or momentum according to the arrived record. Its task utility is evaluated by the bound runner.

## 5.3 RTT: how the updater participates in successor generation

At generation �, form a proposal task: produce a legal updater program with better task performance under the common budget. Its input includes $\eta _ { k }$ , the fixed interface, and permitted development summaries. Begin a private proposal-policy state $\Phi _ { 0 } ^ { \mathrm { p r o p } } = \bar { \rho } ^ { \mathrm { p r o p } }$ . The model samples a bounded token rollout $y _ { j } \in \Omega _ { \mathcal { H } }$ using a fixed grammar mask. This space includes terminated legal serializations and incomplete or invalid outcomes of the bounded sampler. Decoding returns a candidate $c _ { j } \in { \mathcal { H } }$ or the failure symbol ⊥. Candidate execution or its fixed failure branch produces an arrived bounded reward $r _ { j } ^ { \mathrm { p r o p } }$ Algorithm 1 specifies its value.

Candidate execution is discrete. We therefore use a score-function estimate [19] for the proposal-policy loss:

$$
\begin{array} { r } { g _ { j } ^ { \mathrm { p r o p } } = - \operatorname { s g } \left( r _ { j } ^ { \mathrm { p r o p } } - \beta _ { j } \right) \nabla _ { \Phi ^ { \mathrm { p r o p } } } \log p _ { \Phi ^ { \mathrm { p r o p } } } ^ { \mathcal { H } } ( y _ { j } \mid x _ { j } ) . } \end{array}\tag{14}
$$

The log probability is the probability of the actual sampled rollout $y _ { j }$ under the declared grammar mask and stopping rule. It is not conditioned on successful parsing or renormalized over surviving candidates. A truncated rollout is scored through its actual sampled prefix; forced cap termination is deterministic. The failure reward therefore participates in the same bounded sampling objective. Equation 14 targets candidate development evaluation conditional on the pre-sample history; it is not an unbiased meta-gradient of the entire selection–audit–recursion trajectory. The reward is stopped from gradient propagation. Baseline $\beta _ { j }$ is fixed from the pre-sample history or an independent sample; it must not covertly use the sampled candidate’s own reward while claiming the usual unbiased score-function identity. Finite-precision estimates, masks, truncation, and reward clipping are part of the implementation identity.

Crucially, the proposal task then invokes its current updater:

$$
( \Phi _ { j + 1 } ^ { \mathrm { p r o p } } , s _ { j + 1 } ^ { \mathrm { p r o p } } ) = U _ { \eta _ { k } } ( \Phi _ { j } ^ { \mathrm { p r o p } } , s _ { j } ^ { \mathrm { p r o p } } , g _ { j } ^ { \mathrm { p r o p } } , z _ { j } ) , \qquad y _ { j + 1 } \sim p _ { \Phi _ { j + 1 } ^ { \mathrm { p r o p } } } ^ { \mathcal { H } } .\tag{15}
$$

The fresh rollout $y _ { j + 1 }$ and its decoded candidate are generated after feedback and the committed update, while the proposal task is unresolved. The transition records the actual factor and private-state changes, then the subsequent read. A zero current gradient can still produce an update when momentum is nonzero. Conversely, changed momentum alone need not change the next proposal if the reader uses only Φ. Functional use is classified from the declared read and action kernel, not from the gradient norm or a version counter. A rejected update restores both factors and private updater state from a snapshot taken immediately before the updater call; already consumed sampling randomness, feedback records, and costs are retained.

Write the resulting generator as $\begin{array} { r l } {  { { \mathcal { G } } _ { \eta } = } } \end{array}$ fixed proposal framework $\circ \operatorname { R T T } ( U _ { \eta } )$ . This expression says exactly what changes: the framework’s code stays fixed, while the learning process it calls changes with �. The common typed interface makes the same update program executable in both learners. Whether it learns well on both gradient distributions is measured separately from its type and state compatibility.

## 5.4 RTMA: a bounded memory-management program

The memory route makes the management rule, rather than just the retained contents, a candidate for improvement. Its minimal mutable domain chooses which complete arrived evidence records to retain. It leaves record meaning, authority checks, representation, and reading fixed. No numerical gradient is required for this branch.

Records, capacity, and authority. Bind $S \geq 1$ slots, a maximum serialized row size $B _ { M } \geq 1$ , and an initially valid $M _ { 0 }$ for each learner role. The authoritative capacity is at most $S B _ { M }$ bytes, including row metadata and any stored derived view. A row records a bounded attempted action $y ,$ its arrived feedback loss $\ell \in [ 0 , 1 ]$ , normalized charged cost $c \in [ 0 , 1 ]$ , a completion/failure tag, an immutable source identifier, and the role-local opportunity index �. In the proposal role, � is the actual sampled program rollout, including a failed or capped prefix; its evaluation loss is $- r ^ { \mathrm { p r o p } }$ and its cost is the development execution’s ${ \bar { C } } .$ If parsing or bounded sampling produces no legal candidate, no development execution starts: set $\bar { C } = \bar { 0 }$ , loss one, and the explicit invalid-or-incomplete tag. Sampling, parsing, and failure handling still enter the total cost ledger. In the task role, � is the sampled task action, ℓ is the runner’s legal feedback loss, and � is incurred attempt cost divided by its bound attempt allowance. All strings, integer encodings, tag vocabularies, token counts, and numeric representations are fixed and finite. There is no missing-value inference: malformed or oversized evidence is an operation failure; a legal failed attempt ha an explicit tag and loss one.

The candidate payload � is assembled once from those observations. For each target slot �, a fixed assembler constructs a complete replacement row $R _ { i } ( d ; M )$ , including slot and object identifiers, expected predecessor version, new row version, provenance binding, canonical payload, and its fixed-size derived view. The derived view may be a fixed encoding of the canonical fields; the bound deterministic encoder and consistency check are part of the carrier identity. The validator checks bytewise agreement with this encoding before publication and at verified reads. Reader access to the canonical view does not authorize omitting required integrity fields. Object identifiers are unique within the carrier, and an attempted duplicate insertion with incompatible lineage is rejected. The trusted validator determines eligible slots from the fixed protection and eviction rules; the minimal configuration marks all slots evictable. The manager cannot change those rules. Stored and temporary representations, the isolated archive, and construction costs are accounted for separately from resident capacity.

A shared scoring language. First validate the payload's common schema, size, and provenance requirements. A malformed payload returns an operation failure with � unchanged, before action features or scores are computed. A correctly formed record of a failed attempt remains a valid payload. Only after this common check passes are complete rows assembled and validated against the slot-specific constraints. A valid payload with no eligible write slot leaves NULL as the normal choice. The resulting legal action set is $\mathcal { A } ( M , d ) = \{ \mathrm { N U L L } \} \cup \{ \mathrm { W R I T E } _ { i } : R _ { i } ( d ; M )$ passes validation}. An empty slot gives insertion; an occupied slot gives eviction followed by insertion. Let $o _ { i }$ indicate occupancy and let $( \ell _ { i } , c _ { i } , t _ { i } )$ be the current row’s fields. Empty-slot fields are zero. NULL has $o _ { i } = \ell _ { i } = c _ { i } = t _ { i } = 0$ and its age feature is zero. At opportunity � with role cap $T ,$ define the fixed action features

$$
\begin{array} { r } { f ( M , d , a ) = \bigl ( { \bf 1 } _ { a = \mathrm { N U L } , \textit { o } _ { i } } , \boldsymbol { \ell } , \boldsymbol { c } , \ \ell _ { i } , \ { c } _ { i } , \ \mathrm { a g e } _ { i } , \ \mathrm { o c c } ( M ) / S \bigr ) , \quad \mathrm { a g e } _ { i } = \left\{ \begin{array} { l l } { \mathrm { m i n } \{ 1 , ( j - t _ { i } ) / T \} , } & { a = \mathrm { W R I T E } _ { i } , \ o _ { i } = 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

Initial occupied rows have $t _ { i } = 0 ;$ subsequent writes store the current $j .$ Each learner starts at $j = 0 ;$ , advances its read-only opportunity counter after a feedback cycle, and resets it at problem start. Time passing does not mutate rows. Features are in [0, 1] and have the same units in both roles, with $T = T ^ { \mathrm { t a s k } }$ or � and independently bound cost denominators. They include information about the proposed row and the displaced row, allowing the rule to trade new evidence against retained evidence. They are observations, not oracle estimates of future utility.

A manager $\eta ^ { M } = e _ { M }$ is one scalar expression tree using the RTT grammar’s finite constants and arithmetic operators, with leaves $f _ { 1 } , \ldots , f _ { 8 }$ instead of $z _ { 1 } , \ldots , z _ { 4 }$ . Its own node, depth, instruction, and serialization bounds are fixed. The same finite-precision evaluation, safe division, and intermediate finiteness checks apply. The chosen action is

$$
V _ { \eta ^ { M } } ( M , d ) = \operatorname * { a r g m a x } _ { a \in \mathcal { A } ( M , d ) } e _ { M } { \bigl ( } f ( M , d , a ) { \bigr ) } .\tag{16}
$$

Ties use the fixed order NULL, then ascending slot index. Infeasible slots are omitted; NULL is always available. A scoring failure rejects the entire operation without changing �. A non-NULL selection is revalidated against the pinned predecessor and committed as a complete row; any failed validation is recorded as failure, not a silent NULL. The interpreter permits no I/O or access to payload text beyond the declared scalar features. It has no hidden recurrent state. The all-zero tree selects NULL; other trees can prefer an empty slot or replace an occupied high-loss record. These examples show the action-dependent domain, not an assumption that low-loss records are always most valuable.

NULL leaves all authoritative bits, versions, and indexes unchanged. Its receipt and cost go to the isolated execution record A non-NULL commit changes the selected row and its authorized version/index metadata; it does not rewrite unrelated records. The later read is a verified pin of this successor. If all decisions are NULL, that run contains no positive RTMA memory path.

The memory seen by the next attempt. The fixed reader serializes every occupied canonical row in slot order, with unambiguous field delimiters and explicit evidence tags. It reads no omitted payload through an external pointer. A preflight check ensures the immutable header plus the maximum permitted memory serialization fits the model’s context and read allowance. The header contains the current problem, parent program, and fixed interface, but no changing evaluation history. Each new attempt rebuilds its input and input-dependent KV cache from that header and its pinned $M ;$ evicted records cannot survive in an old cache. Archived attempts remain available to the final selector and auditor only. The proposal model and memory manager cannot read that archive. Cache keys include the pinned memory, policy, reader, and header identities when reuse is implemented.

## 5.5 RTMA: learning memory and improving its manager

Task and proposal learners use the same row schema, action features, expression interpreter, and commit/read interface, with independent $M _ { 0 } ^ { \mathrm { t a s k } }$ and $M _ { 0 } ^ { \mathrm { p r o p } }$ . The task runner fixes the unresolved problem, legal action distribution, feedback, terminal endpoint, and opportunity limit. Between unresolved attempts it supplies a complete arrived payload to the active manager and then pins the resulting memory for the next attempt. In RTMA-only runs the policy state is fixed. The two roles share field semantics; whether one manager works well across their feedback distributions is evaluated, not inferred from type compatibility.

In generation $k ,$ the proposal problem is to produce a better manager within the fixed interface and budget. The model samples a candidate $c _ { j } \in \mathcal { H } _ { M }$ or an invalid bounded rollout. A legal candidate is evaluated as a manager in fresh task learners; the fixed development block returns its arrived loss and cost. The proposal learner then uses its current parent to manage this evaluation record:

$$
\begin{array} { r l } & { d _ { j } ^ { \mathsf { p r o p } } = \mathrm { A s s e m b l e P a y l o a d } ( y _ { j } , - r _ { j } ^ { \mathsf { p r o p } } , \bar { C } _ { j } , \mathrm { p r o v e n a n c e } ) , } \\ & { M _ { j + 1 } ^ { \mathsf { p r o p } } = \mathrm { C o m m i t } \big ( M _ { j } ^ { \mathsf { p r o p } } , V _ { \eta _ { k } ^ { M } } ( M _ { j } ^ { \mathsf { p r o p } } , d _ { j } ^ { \mathsf { p r o p } } ) , d _ { j } ^ { \mathsf { p r o p } } \big ) , } \\ & { y _ { j + 1 } \sim p _ { \rho ^ { \mathrm { p r o p } } } ^ { \mathcal { H } _ { M } } ( \cdot \mid \mathrm { h e a d e r } , \mathrm { R e a d } ( M _ { j + 1 } ^ { \mathsf { p r o p } } ) ) . } \end{array}\tag{17}
$$

The notation suppresses the commit receipt and version returned by Equation 6. The candidate’s task executions end before their development summary arrives, but the enclosing proposal problem is still unresolved. Thus the write and next proposa occur in the same problem. Candidate managers never manage the current proposal memory before admission.

This route contains two learning levels. Within a generation, feedback changes the retained memory under $\eta _ { k } ^ { M }$ . Across generations, a candidate with independently improved task performance can replace that management rule. Generation $k + 1$ resets proposal memory to the same bound $M _ { 0 } ^ { \mathrm { p r o p } }$ and uses $\eta _ { k + 1 } ^ { M }$ to choose the evidence for further proposals. The rule, rather than an ever-growing transcript, carries the cross-generation change. Merely storing a candidate’s program in memory does not establish this dependency; the admitted program must execute as the next manager.

## 5.6 A joint RTT–RTMA instance

For the joint domain, a candidate is a canonical pair $( \eta ^ { \Phi } , \eta ^ { M } )$ . Its task evaluation executes both rules. The proposal learner also uses both parent rules on the same arrived development feedback: first compute the score-function gradient and commit the numerical update, then assemble the feedback payload and commit the selected memory action. Memory scores use only Equation 16’s observed features; they do not secretly query the updated policy or a value cache. The next attempt pins the compatible pair $( \nu ^ { \Phi } , \nu ^ { M } )$ and reads both states. The sampler grammar, serialization cap, and candidate identity cover the whole rule pair.

Each carrier operation is atomic, but the pair is not a single transaction. A rejected numerical update restores its own factors and private momentum and terminates this minimal attempt before memory management. If it succeeds and memory selection returns NULL, retain the new Φ and unchanged �. If memory construction or commit fails, retain the already committed Φ, preserve the receipts for the successful policy commit and failed memory operation, and end the task or generation without a further proposal. An incompatible version pair likewise prevents the next read; it does not undo successful commits. Costs, feedback already observed, and consumed random streams remain recorded in every case.

Admission of a joint candidate targets the task risk of the rule pair $( \eta ^ { \Phi } , \eta ^ { M } )$ . It certifies neither component-wise improvement nor a beneficial interaction, and an accepted edge need not change both rule components. The joint construction does not require both states to change after every feedback event. Its positive dual-path evidence requires actual policy and non-NULL memory paths with later reads. Each route can afect complete task utility conditionally on the other; their individual efects and interaction are measured using Section 4, rather than inferred from the fact that two modules ran.

## 5.7 Lock, audit, adopt, and use

After a finite proposal budget, choose at most the registered number of candidates using development evidence and lock their exact identities. Each locked candidate and its parent run on paired, previously unused audit tasks with the same starting state, information, and resource envelope. A fixed evaluator measures the primary task endpoint and cost. Its answers cannot update the proposal state that produced the candidate under audit.

If a candidate satisfies the fixed admission rule, commit $\eta _ { k + 1 }$ and begin the next generation with a fresh proposal state. That next generation uses the admitted $U _ { \eta _ { k + 1 } ^ { \Phi } } , V _ { \eta _ { k + 1 } ^ { M } }$ , or their pair in the same proposal-learning role. Otherwise retain $\eta _ { k }$ or terminate according to the registered finite contract. A failed candidate remains in the record, with its development, audit, serialization, and restoration costs. Restoring a previous version is a state transition; it is not a new budget allowance.

![](images/be46e1f4528773f58d9be2fc1e520768c1eb8ff42f9fbae5034df7f749676af8.jpg)  
A failed admission retains the parent and ends the finite run; all charges remain  
Figure 4: Two realizations of parent participation. The RTT parent updates the numerical proposal state. The RTMA parent manages the authoritative evidence read by the next proposal. Selection and independent audit are shared; an admitted rule returns to its own role in the next generation. Joint runs carry both rules.

## 5.8 The complete recursive algorithm

Algorithm 1 collects the construction into one execution order. Its parameters fix a finite run without fixing experimenta hyperparameters in this report. In addition to the language bounds, they include a generation cap $K \geq 1$ , a proposal cap $N \geq 2$ per generation, a nonempty finite development task block, fresh audit blocks, the task runner, a cost weight $\lambda \in [ 0 , \infty )$ , a deterministic tie order, and finite resource reservations for every call. At most one candidate is locked and audited in each generation. Failure of admission ends this minimal run.

For a candidate $^ { c , }$ execute the fixed task runner with its rule or rule pair from private copies of $\rho ^ { \mathrm { t a s k } }$ and $M _ { 0 } ^ { \mathrm { t a s k } }$ on the development block. Return its mean final loss $\bar { L } ( c ) \in [ 0 , 1 ]$ and normalized charged cost $\bar { C } ( c ) \in [ 0 , 1 ]$ , using the fixed per-execution cost allowance as denominator. For a legal program, a runtime exception that prevents legal feedback, a rejected numerical update or malformed memory operation, or exhaustion of a task-local allowance is a terminal task failure. It ends only that task with final loss one and its actual incurred cost; ordinary action, validation, and tool-error feedback follows the continuing runner described above. Evaluation then continues with fresh state on the remaining fixed tasks; means use the entire fixed block, never only completed successes. The same rule applies within an audit block. Exhausting the global reservation stops the generation without admission. The proposal reward is

$$
r ^ { \mathrm { p r o p } } ( c ) = - \operatorname* { m i n } \{ 1 , \bar { L } ( c ) + \lambda \bar { C } ( c ) \} .
$$

This reward intentionally caps the combined loss-and-cost penalty at the unit failure level. Candidates above that level have the same proposal reward, so their within-region quality diferences do not afect the score-function reward; the recorded cost still breaks selection ties. This is the chosen saturated objective, not an order-preserving normalization of the uncapped sum. The saturated reward guides development proposal learning and selection. Admission instead tests the bound task risk $R ( \eta )$ in Equation 18 together with deterministic resource checks; a better development reward alone does not admit a successor. The minimal instantiation fixes $\beta _ { j } = 0$ in Equation 14. The development task block and task random tapes are paired across candidates; each execution begins from fresh private state. Audit tasks are disjoint from development and from prior audits. A bound stream table assigns independent tapes to proposal sampling and task execution; paired candidate–parent audit runs share task tapes, not learner state.

The proposal input $x _ { j }$ is the immutable header followed by the fixed read of $M _ { j } ^ { \mathrm { p r o p } }$ . For RTT-only runs memory stays at $M _ { 0 } ^ { \mathrm { p r o p } }$ ; feedback afects later proposals through Φ. For RTMA-only runs Φ stays at $\rho ^ { \mathrm { p r o p } }$ ; feedback reaches them through the selected memory rows. Joint runs expose both paths. This common input contract has no additional full candidate-history prompt. The bounded archive used for final selection is inaccessible to the generator and its adaptation rules. Fixed header and row bounds must fit the model context without truncating a committed row.

Generation uses the fixed grammar mask, temperature, token cap, and normalized sampling law for the selected domain. Initialize each attempt record with loss one, development cost $\begin{array} { r } { \bar { C } _ { j } = 0 . } \end{array}$ , its source identifier, and an invalid-or-incomplete tag. A legal proposal executes the full fixed block and replaces these defaults with its measured loss, normalized cost, and completed-evaluation tag, including task-failure indicators. Each terminated legal proposal is scored once; incomplete or illegal output retains reward −1 and is ineligible for admission. Its actual bounded rollout can still supply RTT’s score-function loss or RTMA’s record of a failed attempt. Parsing does not renormalize the sampling law over successful candidates.

## Algorithm 1: MMLA-RSI with one locked successor per generation

Inputs. A domain � ∈ {RTT, RTMA, joint} and legal �<sub>0</sub>; bound substrate, role initializations and memories; languages, typed record schema, reader and committer; task/development/audit partitions; stream table; �, �; fixed cost and admission rules.

1. Bind and reserve. Check identities, partitions, language membership, context/capacity bounds, numerical implementation, and the finite ledger. Set $k = 0$ and the admitted-version list to [�<sub>0</sub>].

2. Start generation $k < K .$ . Reserve its envelope, including one complete paired audit and closure work. Bind the parent rule or pair. Set $j = 0 , \Phi _ { 0 } ^ { \mathrm { p r o p } } = \rho ^ { \mathrm { p r o p } } , s _ { 0 } ^ { \mathrm { p r o p } } \stackrel { \cdot } { = } 0 , M ^ { \mathrm { p r o p } } = M _ { 0 } ^ { \mathrm { p r o p } }$ , and $\ell _ { - 1 } ^ { \mathrm { p r o p } } = 0$ . Bind $T ^ { \mathrm { p r o p } } = N$ , initialize the isolated selection archive, and keep the proposal problem unresolved

3. Generate attempt $j < N .$ Pin compatible state versions. Sample a bounded rollout from the declared header and memory read with the current Φ. Record its probability, parent and read versions, random-tape reference, and times.

4. Obtain development feedback. Initialize the typed attempt record as above, then decode and check the candidate. A legal candidate runs in fresh private task learners and replaces the defaults with its block results. Aggregate task-local failures as specified; invalid or incomplete proposals retain reward −1 and development cost zero. Charge all actual work to the ledger. Append the record to the isolated selection archive.

5. Adapt before another attempt. If $j + 1 = N _ { : }$ , go to step 6. Otherwise invoke the parent, not the candidate: RTT computes Equation 14 and calls $U _ { \eta _ { k } ^ { \Phi } } ;$ ; RTMA assembles $d _ { j }$ and calls $V _ { \eta _ { k } ^ { M } }$ plus the trusted committer; joint executes RTT then RTMA from the same arrived feedback. Record actual changes, NULL/rejection receipts, and the next read versions. On a failed operation, end the generation without admission using the rollback rules in the text. Otherwise advance the preceding-loss record and �, then return to step 3.

6. Lock one candidate. Eligibility requires a preceding committed parent numerical update for RTT, a preceding non-NULL parent memory commit for RTMA, and both for joint; subsequent attempts must read their recorded descendant states. Among eligible legal proposals choose the highest development reward, lowest cost, then lexicographically smallest canonical program; duplicates use the earliest eligible attempt. If none exists, terminate. Lock the candidate, settings, and selection provenance.

7. Audit and decide. Run the candidate and parent on paired fresh audit tasks from their bound task states. Apply Equation 18 and resource checks. Audit outcomes do not enter this generation’s proposal state. An incomplete audit or failed admission terminates with the parent retained.

8. Adopt and recur. Commit the admitted rule or pair $\eta _ { k + 1 }$ and its version edge. Close the problem, release private state, and increment �. If $k < K$ , return to step 2: reset the proposal states but use the new rule in the same adaptation role.

9. Deliver on every exit. Save the last admitted version, all candidates and failures, both state/read receipt chains, selection/audit identities, and cumulative charges. Exhaustion or absence of a successor ends the run without an automatic retry.

The full construction. The branch-specific update/read path makes a later candidate eligible; it does not by itself certify functional participation or an accepted recursive-improvement chain. Those observations are classified separately using Equation 5 and Definition 5.1.

Functional probes bind an actual update/read transition to the candidate-producing history and receive their own resource reservations. The resulting evidence determines whether the admitted versions form the chain defined below.

Definition 5.1 (Finite MMLA recursive improvement). A finite accepted chain $\eta _ { 0 }  \eta _ { 1 }  \eta _ { 2 }$ has the proposed recursive dependency when each child is produced through functional participation of its parent adaptation rule, each accepted edge has independent improvement evidence under the declared task-and-cost criterion, and the accepted $\eta _ { 1 }$ actually participates in generating $\eta _ { 2 }$ . Participation follows the RTT policy path, the RTMA memory path, or their declared joint closure. A claim that both routes participate additionally needs separate path witnesses. Longer chains repeat these obligations within the fixed mutable domain.

One independently improved and accepted rule demonstrates a useful change to an adaptation method. A second accepted edge, with the first successor now acting in the proposal process, establishes the finite recursive chain. A mere hash diference, a version label in a prompt, or selection by an unchanged external optimizer does not substitute for functional parent participation.

## 5.9 Interventions and practical comparisons

For RTT, two interventions address diferent parts of the dependency. Disabling the update replaces $U _ { \eta }$ by the identity on both Φ and its private state, while holding the reached prefix, arrived feedback, and other inputs fixed. It asks whether numerical updating participates in the next proposal. Replacing the updater runs the evolved $U _ { \eta _ { k } }$ and original $U _ { \eta _ { 0 } }$ on the same pre-update $( \Phi , s , g , z )$ within their common state space, then compares the subsequent read. It asks whether the evolved rule, beyond merely having an update, changes successor generation. With nonzero � this is a local common-state intervention; a complete trajectory comparison starts both arms from their bound initial states.

For RTMA, disabling the operation forces NULL while preserving the arrived record, state, and budget accounting Replacing the manager evaluates $V _ { \eta _ { k } ^ { M } }$ and $V _ { \eta _ { 0 } ^ { M } }$ on the same $( M , d )$ , then compares the resulting pinned reads and proposal kernels. Changed scores alone are not participation evidence: the selected action and read must matter. A joint probe intervenes on each rule and state path independently, then on the pair. A joint efect alone cannot be assigned to both routes.

These interventions use the action-distribution criterion or the explicitly conditional replay object specified in Section 2. Fixing one random tape is useful for debugging a transition, but does not silently convert a replay diference into a diference of marginalized kernels. Whole-history probes must also specify which past updates are replaced; changing only the current call and replacing a parent over the entire generation answer diferent questions.

A separate performance comparison runs complete proposal trajectories under the evolving rule or pair, its original fixed version, and a fixed external search method in the same candidate language and common total budget. It measures whether the evolving process finds useful successors more efectively. Trajectories can diverge after the intervention; the locked outputs receive independent evaluation. Such an advantage concerns discovery quality and economic value, rather than the existence of the recursive structure or the definition of a finite accepted chain.

The external search method receives the same legitimate development information, representation, evaluator, and total allocation, with the same memory/read capacity and archive-access restrictions. Generation, candidate execution, backward passes, failed proposals, audit, and recovery all count. If external search wins, the observed implementation may be less practical while retaining its specified recursive dependency.

## 6 Conditional Properties of the Construction

The method is the adaptation-rule-to-proposal construction in Section 5. The following properties describe its causal execution, accepted transitions, and finite resource use under the stated assumptions.

Proposition 6.1 (Causal composition). Suppose each generation begins with an adaptation-rule version measurable with respect to its admittedpast; each within-task transition is measurable with respect to arrived records andpermitted randomness; and an audit is opened only after the candidate is locked. Ifcallbacks are epoch-bound and cannot modify a closed task, every action in afinite sequence ofgenerations is measurable with respect to the information available at its execution time.

Proof. Condition on the admitted state at the start of a generation. Composition of the causal proposal kernel, arrived-feedback map, and selected adaptation operation yields a causal next state. Induction covers the finite attempts in that generation. Candidate locking precedes the audit; admission is therefore a function of information available at the generation boundary. This state supplies the induction hypothesis for the next generation. Epoch checks exclude a late write from a closed task. □

Proposition 6.2 (Finite admission on a fixed task distribution). Let $R ( \eta ) \in [ 0 , 1 ]$ be the loss ofa bound learner using the selected rule or rule pair � on a fixed task distribution and resource rule. For every locked candidate and its parent, suppose audit intervals cover their risks simultaneously with probability at least 1 − �. Admit an edge only if

$$
U _ { k } ( \eta ^ { \prime } ) + \delta \le L _ { k } ( \eta _ { k } ) , \qquad \delta > 0 ,\tag{18}
$$

and its deterministic resource checks pass. On the simultaneous coverage event, every accepted edge reduces risk by at least �;   
hence there are at most $\lfloor R ( \eta _ { 0 } ) / \delta \rfloor$ accepted edges.

Proof. Coverage gives $R ( \eta ^ { \prime } ) \leq U _ { k } ( \eta ^ { \prime } ) \leq L _ { k } ( \eta _ { k } ) - \delta \leq R ( \eta _ { k } ) - \delta$ . Sum along accepted edges and use $R \geq 0$ . A rejected proposal does not change the bound learner’s admitted rule, though it consumes resources. □

One way to obtain the stated event is to allocate conditional error probabilities to all candidate–parent comparisons, with their sum at most $\alpha ,$ and use fresh audits with valid conditional intervals. Reusing an ordinary bootstrap interval after repeatedly selecting on the same audit data does not establish this assumption. The component bootstrap intervals reported later are their original finite-study summaries; they are not being promoted to sequential admission guarantees.

Proposition 6.3 (Finite work). Ifthere are at most � generations, at most $N _ { k }$ proposals per generation, afinite maximum instruction count and state sizefor every adaptation call, and an enforced reservationfor every generation, execution, update and audit operation, total work is bounded by the sum ofthe reserved costs.

Proof. There are finitely many operations and each has an enforced finite bound. Summing these bounds covers successful and failed operations alike. An operation without a suficient reservation cannot begin. □

A decreasing budget variable alone is insuficient if a loop can perform infinitely many nominally zero-cost operations. The finite generation and instruction bounds are therefore part of the protocol, not only accounting metadata.

The admission result applies to accepted edges on the fixed distribution, and the work bound applies whether candidates are accepted or rejected. Existence of an admissible successor and performance after a distribution change are empirical questions outside those conditions

## 7 Component Evidence and Design Motivation

The completed studies summarized here establish implementations of the numerical carrier and its update, retention, read, and restoration operations. They do not evaluate the new recursive memory-manager domain or a joint recursive chain. They also expose dificulties that a recursive updater must address. Program selection and CI reuse the LARC experiments [29]; the SQL studies are subsequent completed evaluations. The four populations and endpoints are distinct. Appendix B retains their full results and uncertainty, and Appendix C retains all reported cost scopes. No new experiment was run for this report.

<table><tr><td>Component study</td><td>Main observation</td><td>Strong comparison or boundary</td></tr><tr><td>Four-program selection</td><td>Retention reduces expected error by 24.65 and 36.65 points for static and adapted initializations.</td><td>Real-keep error 29.302% / 24.611%; support-loss selection 0.78125%.</td></tr><tr><td>Chronological CI replay</td><td>G = −0.05337 while D = +0.14969: real binding helps relative to sham, retention harms.</td><td>Half-Brier loss rises from 0.12744 to 0.18081; HEDGE4 reaches 0.10243.</td></tr><tr><td>SParC interaction</td><td>Whole-interaction success 18.42% versus frozen six-proposal 20.18%; paired gain –1.75 points.</td><td>97.5% interval [-4.39, 0.88]; strict prefix-read task gains are zero.</td></tr><tr><td>CoSQL correction transfer</td><td>Final-query success 46.965% versus frozen 47.944%; paired gain -0.979 points.</td><td>97.5% interval [-4.247, 1.759]; neither arm passes its online Φ requirements.</td></tr></table>

Table 3: Existing component evidence, not an evaluation of the proposed recursive routes. Percentage-point diferences and half-Brier diferences have diferent units; the studies are not pooled.

The program-selection result is a development evaluation over 16 parameter groups and three paired training seeds; its final audit remained unopened. It concerns four supplied candidate programs rather than free generation. Both learned initializations benefit from feedback, although a direct same-information rule is much more accurate. Expected and greedy decisions also order the two arms diferently, and the fixed stability rule selects the static arm.

CI uses a repository-local retrospective queue, with equal repository weights over one NumPy commit and 71 pandas commits. Its three-seed means are descriptive rather than population intervals. The negative retention efect and positive binding efect show that correctly associated feedback can still drive harmful learning. The later same-batch non-descent diagnostic does not uniquely identify the cause.

The SQL studies provide task-level evaluations with seeds averaged inside tasks and databases as the resampling units. In SParC, saved replay states change some proposal tokens and selected queries without changing the paired task endpoint. CoSQL’s trained task models greatly improve over the original substrate (7.295% final-query success), but their additional online increment fails the registered requirements. The copy-edit, retrieval, and search rules reach 1.824%, 5.348%, and 0.961%; selection remains the development-chosen copy-edit rule. Both SQL endpoints measure reference execution concordance on fixed databases, not universal SQL equivalence. Shared feedback-text and selection paths remain part of their declared reads.

These findings motivate searching over the learning rule while retaining separate measurements of feedback binding, state use, and task benefit. They provide component foundations, rather than a test of the recursive construction’s success or failure. A numerical update that helps a task, an updater that improves task learning, and an updater that helps produce a useful successor are observations at diferent levels.

## 8 Relation to Other Self-Improvement Routes

MMLA-RSI connects policy and memory adaptation with recursive changes to their respective methods. Its distinction from other routes is clearest along three axes: what changes, when feedback becomes a learning signal, and how the changed object participates in the next improvement. These axes overlap; they do not partition all systems into mutually exclusive families.

RRSI regularizes the evolution of an agent harness: prompts, control flow, tools, memory, and context management around a frozen backbone [20]. The current harness produces traces and feedback, proposal uses the evolution history with a constrained edit budget, and screening and cost-aware selection determine the next incumbent. The inspected algorithms keep the outer proposal and selection rules prescribed. This is an agent-system route to RSI. MMLA-RSI specifies two narrower mutable domains and their proposal-state pathways: the current numerical updater changes proposal Φ, and the current memory manager changes proposal �. The RTMA route overlaps memory and context-management search; its defining construction is the explicit reuse of the candidate management rule in managing the evidence that generates its successor.

MemEvolve jointly evolves experiential knowledge and the memory architecture through programmatic encoding, storage, retrieval, and management modules [26]. Its outer process selects architectures using performance, cost, and latency, diagnoses their execution trajectories, and redesigns permitted modules. It is a direct precedent for improving the memory mechanism itself. The RTMA instance here fixes the representation and reader while varying a bounded retention-and-replacement rule. Equation 17 places that rule in a particular proposal interface: the current manager controls which arrived evaluation records the successor generator can read, and an admitted successor takes over that proposal-memory role in the next generation. The comparison concerns this parent-to-proposal dependency and the mutable domain.

On-policy distillation (OPD) supervises a student’s own rollout prefixes with teacher distributions. Agarwal et al. develop this approach through Generalized Knowledge Distillation [1]; the Thinking Machines account provides a later practical treatment [11]. It describes a sampling and supervision relation; it does not by itself specify updates inside an unresolved answer or changes to the optimizer. In On-Policy Self-Distillation (OPSD), a teacher from the same model family receives privileged context; the reported experiments keep that teacher fixed at the initial policy [28]. The more recent DCE+SRCL framework reuses the updated checkpoint for both student and privileged teacher in the next round [21]. This is a real recursive dependence: learned ability changes subsequent supervision. Its specified numerical optimization rule remains fixed. MMLA-RSI’s RTT branch changes that numerical rule and carries it into successor generation; its RTMA branch changes evidence retention through a memory-management rule. An OPD objective could be used inside an RTT system if its supervision were legally available at the required time.

The temporal distinction from TTT also needs care. Recent aTTT updates an episode-specific LoRA from trajectory text during a live episode, then uses the adapted state in later turns [18]. Its repetition-aware token weighting limits reinforcement of repeated text. It therefore directly overlaps RTT’s within-episode numerical update timing. The separate Agentic-TTT work learns a policy for choosing adaptation tools and reusing skills across queries [10]. MMLA’s RTT account contributes an explicit same-problem state lifetime, read closure, and functional feedback witness; the MMLA-RSI RTT extension additionally uses a changed updater in generating its successor. Whether another implementation satisfies these obligations depends on it actual process, not its acronym.

<table><tr><td>Route</td><td>Main changed object</td><td>Dependence carried into later improvement</td></tr><tr><td>aTTT</td><td>Episode-specific numerical adapter</td><td>Updated policy changes later trajectory text under a prescribed update rule.</td></tr><tr><td>RRSI</td><td>Agent harness around a frozen model</td><td>Successor harness produces the next round&#x27;s evidence and evolution history.</td></tr><tr><td>MemEvolve</td><td>Programmable memory architecture</td><td>Architecture-dependent execution traces inform selection, diagnosis, and redesign of memory modules.</td></tr><tr><td>OPSD</td><td>Student policy</td><td>Student sampling changes; the experimental teacher remains fixed.</td></tr><tr><td>DCE+SRCL</td><td>Student and privileged teacher checkpoints</td><td>Updated ability changes the next round&#x27;s supervision.</td></tr><tr><td>STOP / DGM</td><td>Improver program / self-modifying agent</td><td>Successor can act in producing a later descendant.</td></tr><tr><td>MMLA-RSI (proposed)</td><td>Numerical updater, memory manager, or their pair</td><td>Successor adapts the proposal policy or memory used to produce the next rule.</td></tr></table>

Table 4: Diferent realizations of self-improvement. The table identifies demonstrated dependencies in the cited methods and the specified dependency of the proposed MMLA-RSI system. It is not a performance ranking.

Self-Refine and Reflexion use feedback to revise outputs or maintain textual reflections [12, 15]. Such information can improve later attempts without a numerical update. They provide relevant same-information context and memory controls. Retaining a reflection under a fixed manager and recursively changing that manager are distinct operations; both may be useful. Test-time training learns from test inputs [17], while TTT layers treat a learnable hidden state as a sequence-model component [16]. Discover uses reinforcement learning at test time to search for solutions to open problems [24]. These works show why learning time, mutable object, and feedback source should be compared separately.

MAML and its first-order approximation learn an initialization from which task adaptation can proceed [6, 13]. Learned optimizers instead learn an update rule [2]. Both are relevant ingredients. Kirsch and Schmidhuber study self-referential meta learning through shared neural functionality, with fitness-monotonic execution allocating computation according to observed performance [9]. That work directly addresses learning mechanisms that participate in changing themselves. MMLA-RSI specifies bounded symbolic rules, policy or memory adaptation of the proposal process under those rules, and independen admission before version replacement.

Chen et al. formulate optimization-algorithm discovery as program search and discover Lion [4]. Representing and searching update programs is therefore an established approach. In the RTT branch, the current update program additionally trains the proposal policy producing its successor. Its bounded scalar-schedule language gives one concrete domain; the RTMA language gives a separate memory-management domain with the same parent-participation requirement.

STOP is a direct precedent for procedure recursion: its update has the form $I _ { t } = I _ { t - 1 } ( \hat { u } , I _ { t - 1 } , L )$ , so the current improver produces its successor [25]. The authors’ qualification that their frozen language model is not full RSI uses a diferent boundary from improvement-process recursion. Darwin Gödel Machine retains self-modifying agents in an archive, allowing a descendant to become a parent of a later modification [27]. Its frozen foundation model and fixed archive management do not erase that recursive dependence. Promptbreeder changes mutation prompts and applies them to task-prompt mutation [5];

the inspected method uses an explicit higher-level prompt for mutation-prompt evolution, so its levels should not be collapsed into unrestricted recursion.

Gödel machines take a diferent route: self-modification is justified by proofs of utility under a formalized system [14]. Equation 18 is an empirical finite-audit condition with stated statistical assumptions, not a replacement for such a proof system or a universal self-improvement theorem. The practical comparison is between the objects that can change, the evidence tha admits a change, and the role the changed object plays in the next improvement.

## 9 Implementation, Cost, and Reproducibility

## 9.1 A version is a complete computational identity

A reproducible learner binds the substrate revision, shared ofline adaptation, slow factors, updater implementation, tokenizer, numerical configuration, and task reader. Each study uses its own complete computational identity: the program study binds its SFT lineage and always-enabled ofline component, while the SQL studies bind their task adapters. Loading verifies the original assets against that identity; detailed version relationships remain in the accompanying lineage records.

Each episode package should save the actual factor tensors, bound slow version, update count, necessary optimizer and random-number state, task and reader identity, and input cursor. Each generation package adds the domain and parent rules, generated candidate programs, selection record, audit split identity, acceptance decision, and cumulative cost. RTMA packages also bind $M _ { 0 } ,$ complete committed rows, provenance and predecessor receipts, capacity, row serializer, trusted committer, and pinned read/index identities. Joint packages preserve the exact update interleaving. A functional probe and a successful save/load operation answer diferent questions and receive separate records.

The RTT, authoritative-memory, and dual-state modules are part of MMLA v4, and the numerical carrier report is available as LARC [30, 29]. The companion source package for this report contains the manuscript, editable vector figures, rendered figure PDFs, bibliography, aggregate figure data, and a manifest binding the completed evidence inputs. The full private research workspace, raw trajectories, and model weights are not included in this writing package. Their availability is separate from the public paper sources. No public URL is asserted for an unpublished recursive implementation.

Detailed component costs, failed-entry charges, and physical/efective work are retained in Appendix C. The same accounting scope applies to the proposed recursive algorithm: discovering, auditing, and executing an adaptation rule or rule pair are charged together, including memory construction, validation, and reading.

## 10 Discussion

MMLA-RSI develops two ways for an adaptation method to participate in producing its successor. RTT changes the policy state through numerical learning. RTMA changes the authoritative evidence available to the next attempt. Their recursive layers change the numerical updater and memory manager, respectively, then return the admitted rule to that same role in the next generation. The joint construction retains both state paths instead of treating memory as an accessory to policy learning.

The component evidence identifies practical dificulties: feedback can change behavior without improving it, a strong direct rule can outperform a learned carrier, and ofline training can explain most gains attributed to an online system. These observations motivate measurements of complete task performance, state use, and discovery cost. They neither provide nor rule out an efective recursive chain in the proposed domains.

The minimal languages make limited parts of the adaptation process mutable. RTT varies scalar schedules within a fixed numerical template. RTMA varies which complete evidence rows are retained, including the option to write nothing. The trusted memory authority, record semantics, and evaluators stay fixed. More expressive memory transformations or numerical optimizers would be separate domains with their own bounds and comparisons.

The empirical question is whether a rule admitted for improved task behavior then helps produce another admitted rule, and whether that discovery process is useful relative to fixed rules and external search at the same information and total cost. Either route can be evaluated first. A joint claim additionally measures the complete task and both state paths, including the cost of invalidating or recomputing memory values when the reader changes.

## A Illustrative RTT and RTMA Generation Traces

Consider a legal member of the proposed updater language with fixed tensor shapes:

$$
\tilde { g } _ { j } = g _ { j } \operatorname* { m i n } \{ 1 , \kappa / ( \lVert g _ { j } \rVert + \varepsilon ) \} ,\tag{19}
$$

$$
s _ { j + 1 } = \mathrm { c l i p } _ { [ - S _ { \mathrm { m a x } } , S _ { \mathrm { m a x } } ] } \left( \mu s _ { j } + ( 1 - \mu ) \tilde { g } _ { j } \right) , \qquad \Delta \Phi _ { j } = - a _ { j } s _ { j + 1 } .\tag{20}
$$

Here the three expression trees return bounded $a _ { j } , \kappa , \mu$ as specified in Section 5, and the factor update uses its declared projection. This instance changes the scalar schedules within the fixed clipped-momentum template. A comparison fixes the initial factors and optimizer state so that its efect is attributable to the updater.

In generation zero, the proposal learner starts from $\rho ^ { \mathrm { p r o p } }$ and samples a first program. Its development execution returns a reward; Equation 14 gives a gradient; $U _ { \eta _ { 0 } }$ updates the proposal state. The next sampled program is therefore a fresh attempt generated after the update. After a bounded number of attempts, development selection locks one candidate. Independent audit may admit it as $\eta _ { 1 }$ . Generation one resets proposal factors to the same bound initialization and now uses $U _ { \eta _ { 1 } }$ to process proposal feedback. If a resulting $\eta _ { 2 }$ is independently better and parent participation is verified, the trace contains the finite recursive chain defined in Section 5.

This trace requires two distinct uses of candidate evaluation. Development executions teach and select; audit executions estimate utility after selection. A successful audit answer cannot be silently fed back into the proposal trajectory whose independence it was meant to establish. It can enter the declared history of a later generation, which then needs fresh audit tasks.

## A memory-management trace

Take a capacity-two proposal memory, initially empty. When the fixed syntax bounds admit it, one legal score tree is

$$
e _ { M } ( f ) = ( 1 - f _ { 1 } ) \big [ ( 1 - f _ { 2 } ) ( 1 - f _ { 3 } ) + f _ { 2 } ( f _ { 5 } - f _ { 3 } ) \big ] .
$$

It uses only the existing features, constants 0, 1, and arithmetic operations. NULL scores zero; insertion into an empty slot scores $1 - \ell ;$ replacement of an occupied slot scores $\ell _ { i } - \ell .$ With NULL winning ties, this illustrative rule retains lower-loss records when their score is positive and otherwise leaves memory unchanged. It is a concrete program in the stated language, not a measured result or a claim that low-loss records always have greater future value. A first candidate with development feedback loss $\ell < 1$ supplies a complete provenance-bound row. The parent manager selects an empty slot; the trusted committer installs the row, and the next proposal reads its exact contents. After later evaluations fill the slots, a write can replace an older record or select NULL. The generator receives only the header and current rows, so evicted evidence does not persist in a separate prompt history.

A candidate successor manager is executed on fresh development task memories. If selected and independently admitted, it becomes the manager of the next generation’s proposal memory, starting again from the same $M _ { 0 } ^ { \mathrm { p r o p } }$ . Its diferent scoring program can retain a diferent set of evaluation records and thereby afect the successor proposals. This describes the specified dependency, not an observed beneficial run. With joint rules, the same arrived evaluations also update the proposal residua through the parent numerical updater, with both pinned versions recorded.

A parent manager that always selects NULL, such as the all-zero score tree, makes no non-NULL commit. RTMA-only and joint runs then have no eligible successor under step 6 of Algorithm 1 and terminate, even if an independently sampled program has better task performance. Thus a legal initial rule need not provide the state transition required to start a recursive chain.

## B Detailed Component Evidence

We use completed aggregate artifacts to examine what the existing learning state can do and where the proposed recursive layer would need to help. The program-selection and continuous-integration (CI) studies are reused from the LARC report [29]; the two SQL studies are subsequent completed task-level evaluations. They have diferent populations, read closures, and endpoints. Their scores are not pooled. No result from the separate evaluation still running at the evidence cutof is included

## B.1 Program selection: useful adaptation with a stronger direct rule

The program task presents four complete candidate programs and supplies their execution feedback on support examples. The learner selects a program; it does not freely generate one. Both training objectives use the same input LARC4 position on the bound MiniCPM5-1B-SFT substrate with the shared ofline component enabled. Two support-gradient steps use SGD with step size 0.1, no momentum, no weight decay, and no inner clipping. The numerical query predicts a distribution over the four candidates, and expected execution error is the primary endpoint.

The saved development evaluation covers 16 parameter groups, four recipes per group, and three paired training seeds. Its 384 repeated rows are not 384 independent tasks. The reported 99% intervals resample parameter groups within the declared family strata, conditional on those three trained seeds. The separate final audit remained unopened in this study.

<table><tr><td>Initialization</td><td>Real keep</td><td>Real reset</td><td>Sham keep</td><td>Sham reset</td></tr><tr><td>Static objective</td><td>29.302</td><td>53.948</td><td>57.666</td><td>53.948</td></tr><tr><td>Adapted objective</td><td>24.611</td><td>61.259</td><td>60.379</td><td>61.259</td></tr></table>

Table 5: Expected program-selection error (%, lower is better). Reset restores each arm’s own trained initialization. The support-loss selector reaches 0.78125% error.

Both arms benefit from real-feedback retention. The adapted objective has the lower mean post-update error, while its between-objective diferences vary across seeds: +1.45, −10.46, and +23.09 percentage points in its favor. The predeclared stability rule selects the static initialization because its worst-seed real-feedback gain is 20.16 points, versus 18.64 for the adapted initialization. Selection therefore preserves the distinction between mean objective superiority and stability across the three trained instances.

<table><tr><td>Contrast</td><td>Mean (percentage points)</td><td>99% group interval</td></tr><tr><td>Static retention G</td><td>24.65</td><td>[19.39, 29.26]</td></tr><tr><td>Static binding D</td><td>28.36</td><td>[21.85, 34.93]</td></tr><tr><td>Adapted retention G</td><td>36.65</td><td>[31.65, 41.20]</td></tr><tr><td>Adapted binding D</td><td>35.77</td><td>[30.33, 41.01]</td></tr><tr><td>Static error – adapted error, real keep</td><td>4.69</td><td>[1.19, 8.88]</td></tr></table>

Table 6: Within-arm feedback efects and between-objective increment answer separate questions. Intervals are conditional on the three paired training seeds.

The larger adaptation increment of the adapted arm also has two sources. Its $3 6 . 6 5 - 2 4 . 6 5 = 1 2 . 0 0$ point larger � decomposes into a 7.31 point worse reset error and a 4.69 point better retained error. Reporting only the 12.00 point contrast would conflate a worse starting point with a better final policy. Greedy decisions tell another part of the story: static keep has 15.49% greedy error and adapted keep has 16.04%, despite the ordering of their expected errors. Appendix D preserves the corresponding secondary table.

A direct rule evaluates each candidate’s support loss and selects the best. Its 0.78125% error is far below either numerical learner’s error. Thus this study establishes useful feedback adaptation within the specified learner, alongside a much stronger same-information rule. It does not establish an advantage in open-ended program synthesis or a reason to deploy the learner instead of the rule.

## B.2 CI: feedback binding and retention can disagree

The CI study replays public job outcomes in repository-local chronological order. It contains 3,321 jobs across 72 commits: 83 NumPy jobs from one commit and 3,238 pandas jobs from 71 commits. Jobs are aggregated to commits, commits are weighted equally within each repository, and the two repositories are weighted equally. Consequently the single NumPy commit has 71 times the weight of one pandas commit. This is a retrospective repository-balanced replay, not a global prospective deployment sample.

The score is half-Brier loss [3],

$$
\ell _ { \mathrm { H B } } ( p , y ) = \frac { 1 } { 2 } \sum _ { c } ( p _ { c } - 1 \{ y = c \} ) ^ { 2 } .\tag{21}
$$

HEDGE4 is this study’s four-expert exponential-weights combination, based on the HEDGE framework [7]. The name does not import a delayed-feedback regret guarantee for this implementation.

<table><tr><td>Method, with recorded memory condition</td><td>Mean half-Brier loss</td></tr><tr><td>Shared real/sham reset read, real M</td><td>0.12744</td></tr><tr><td>Real feedback, keep Φ, real M</td><td>0.18081</td></tr><tr><td>Permuted feedback, keep Φ, real M</td><td>0.33050</td></tr><tr><td>HEDGE4</td><td>0.10243</td></tr><tr><td>Real retention G</td><td>-0.05337</td></tr><tr><td>Feedback binding D</td><td>+0.14969</td></tr></table>

Table 7: CI replay. The three-seed means are descriptive summaries on two recorded repositories, not population confidence intervals. HEDGE4 has the lowest mean loss among the methods shown.

Both logical reset cells use the same P0M1 prediction, evaluated from each seed’s task-trained initialization $\rho$ and the same real-memory prompt. The permutation changes only fast-update labels; its private random-number state is not read by the probability scorer. The implementation evaluates this common reset read once, rather than separately rolling back and reevaluating both updated branches. Thus the registered contrast is

$$
D = ( L _ { \mathrm { r e s e t } } - L _ { \mathrm { r e a l , k e e p } } ) - ( L _ { \mathrm { r e s e t } } - L _ { \mathrm { s h a m , k e e p } } ) = L _ { \mathrm { s h a m , k e e p } } - L _ { \mathrm { r e a l , k e e p } } \approx 0 . 1 4 9 6 9 .
$$

The equality concerns the per-job prediction readout under fixed real memory. The update histories, private random-number states, and subsequent learning trajectories need not coincide.

Retaining real updates is worse than resetting in every seed, while real binding is better than permuted binding in every seed. With � disabled, the mean retention efect remains negative, −0.04400. Correct feedback binding therefore does not establish useful online learning. A subsequent fixed diagnostic in the LARC study records same-batch cross-entropy non-descent and inconsistent future benefit from shrinking updates; it does not uniquely identify a single root cause [29].

This experiment adapts across jobs. It informs the stability of a numerical learning carrier and the interpretation of � and �, but it is not a same-unresolved-problem RTT witness. The unit and timing of feedback matter as much as the existence of a backward pass.

## Two diferent outcomes of feedback adaptation

![](images/76024e840eab6bb119fe07fecedc14e21e90eede9aa0dfad339ab2858f6ae314.jpg)

![](images/055495e73b0f2a2d6479f225852a71d14ecaaa13013fc63a516c67a6595af394.jpg)  
Figure 5: Binding is not the same as benefit. Left: program-selection retention and binding efects with original 99% development-group intervals. Right: CI efects in half-Brier units, shown as three-seed means without population intervals. Positive values favor the named efect in both panels; the units and populations difer.

## B.3 SQL interaction: state changes without endpoint gains

The first completed task-level SQL study uses grouped interactions from SParC [23], with 56 development interactions across 28 databases and 75 confirmation interactions across 19 databases. Its primary endpoint is whole-interaction success: every turn must produce a query execution agreeing with the reference on the fixed database. This is an operational reference-concordance score, rather than a proof that two SQL programs are equivalent on all databases. Seeds are averaged within the interaction and then databases are weighted equally.

The adapted real-feedback method reaches 18.42% confirmation success, compared with 20.18% for the static six-proposal frozen method. The paired diference is −1.75 percentage points, with a 97.5% database-bootstrap interval of [−4.39, 0.88]. The three registered non-neural rule methods each score zero on this confirmation population; the development-selected rule remains the comparator. Beating those particular rules does not establish an advantage over general-purpose SQL systems.

The saved-state audit is informative even though the main comparison fails. Across development and confirmation, 393 paired cases contain 1,965 fast-state packages. The audit verifies nonnull gradients and state changes in 786 keep packages and exact initialization, optimizer, and random-state restoration in 786 reset packages. On 225 confirmation task–seed pairs, real keep versus reset changes 14 third-proposal token sequences and three selected SQL outputs, while whole-interaction and last-turn scores remain unchanged in every paired case. Real versus sham produces diferent actual rewards and fast states in only six of these 225 pairs. This is evidence of local saved-output sensitivity and weak efective sham separation, not positive task benefit.

The ofline SQL adaptation includes a rank-16 task adapter and the rank-4 input residual, totaling 700,416 trainable entries. Gains over the original model cannot be attributed to the fast residual alone. The strict saved-prefix retention and binding contrasts are zero at the task endpoints despite the verified state and output changes.

## B.4 CoSQL correction transfer: online increment below the prespecified requirements

The next completed study uses CoSQL [22] under its own fixed correction-transfer protocol. The evaluation has 358 development episodes across 28 databases and 210 confirmation episodes across 19 databases, with three paired seeds and 43 method–seed cells over 568 episodes. All 24,424 logical records are complete. The endpoint is final-query execution concordance with the reference, macro-averaged by database. It is distinct from the previous study’s whole-interaction success and is not presented as an oficial benchmark leaderboard score.

The adapted real-feedback procedure difers from the static frozen comparator by −0.979 percentage points, with a 97.5% interval of [−4.247, 1.759]. The registered task-level gate requires at least five points and a positive 97.5% lower bound against both the frozen method and the development-selected rule. It fails against the frozen method.

The final audit checks 13,632 prefix bindings and 3,408 reset/query-closure cases. Memory is disabled, but shared support text, proposal history, feedback, tools, and selectors remain in the declared read. The results therefore concern the additional numerical efect conditional on those channels. The substantial improvement over the original frozen substrate demonstrates a better task model. The prespecified deployment requirements—the task endpoint and the online Φ requirements—remain unmet for this numerical recipe; the completed models and failed comparisons are retained.

<table><tr><td>Confirmation method</td><td>Final-query success (%)</td></tr><tr><td>Original frozen substrate, six proposals</td><td>7.295</td></tr><tr><td>Development-selected copy-edit rule</td><td>1.824</td></tr><tr><td>Retrieval rule</td><td>5.348</td></tr><tr><td>Search rule</td><td>0.961</td></tr><tr><td>Static task model, six frozen proposals</td><td>47.944</td></tr><tr><td>Static task model, real-feedback procedure Adapted task model, real-feedback procedure</td><td>48.230 46.965</td></tr></table>

Table 8: Completed CoSQL correction-transfer evaluation. The copy-edit rule remains the selected rule even though retrieval scores higher on confirmation. The current, separate contrast-training evaluation is not included.

<table><tr><td>Arm and controlled effect</td><td>Mean (percentage points)</td><td>95% database interval</td></tr><tr><td>Static retention</td><td>0.775</td><td>[-0.638,2.930]</td></tr><tr><td>Static binding</td><td>0.387</td><td>[-0.196,0.971]</td></tr><tr><td>Adapted retention</td><td>0.580</td><td>[0.000, 1.565]</td></tr><tr><td>Adapted binding</td><td>0.423</td><td>[-0.125,1.260]</td></tr></table>

Table 9: CoSQL controlled fast-state efects. Each arm requires both binding and retention lower bounds strictly above zero; neither meets that requirement. The strict checks use the stored unrounded bounds; the adapted-retention lower bound is exactly zero in the saved aggregate. These paired prefix reads difer from the complete procedures in Table 8.

## C Measured Costs and Accounting Scope

<table><tr><td>Study</td><td>Accounting scope</td><td>Recorded quantity</td><td>Value</td></tr><tr><td rowspan="3">Program selection</td><td>Full entry, including production preflight</td><td>Wall seconds</td><td>1,634.50</td></tr><tr><td>Contract charge, including data work</td><td>Seconds</td><td>1,640.27</td></tr><tr><td>Physical / effective token positions</td><td>Millions</td><td>49.026 / 19.521</td></tr><tr><td>CI</td><td>Qualification, task fit, and repeated evaluation</td><td>Native wall hours</td><td>3.675</td></tr><tr><td rowspan="3">SQL interaction R5</td><td>This successful R5 invocation</td><td>GPU residency upper hours</td><td>4.819</td></tr><tr><td>Budgeted process-tree CPU</td><td>CPU hours</td><td>5.641</td></tr><tr><td>Native wall snapshot</td><td>Wall hours</td><td>5.266</td></tr><tr><td rowspan="3">CoSQL transfer</td><td>Cumulative finite recipe, including failed work</td><td>GPU residency hours</td><td>14.359</td></tr><tr><td>Cumulative process-tree CPU</td><td>CPU hours</td><td>21.739</td></tr><tr><td>Original origin, including pauses and review</td><td>Wall hours</td><td>24.888</td></tr></table>

Table 10: Recorded component costs. Wall time, process-tree CPU, GPU residency, and efective computation are separate quantities. They must not be summed as though they were one resource measure.

Program-selection physical positions decompose as 49,025,792 = 47,435,776 mixer positions +1,585,920 selected-head positions +4,096 native positions. Its 19,521,408 efective positions count a diferent computational quantity. The entry includes 1,536 formal outer updates and two production-preflight outer updates; it is not simply the nominal training matrix multiplied by a forward cost.

The CI total includes a 57.84-second failed qualification, 6,626.08 seconds of task-fit work, and 6,545.88 seconds of repeated evaluation. The SQL R5 cost does not include the preceding four failed entries, whose separate GPU residency totals 2,760.68 seconds and CPU budget charges total 3,541.26 seconds. The CoSQL cumulative record includes failed work and the redone third-seed evaluation; its separate result-collection call uses 24.02 CPU seconds and 23.97 wall seconds. These boundaries preserve the actual cost of arriving at a result without folding unrelated experiments into it.

The recursive design must use the same discipline. A fast updater can be expensive to discover; a model that improves after more attempts may lose under an equal complete-task budget. Proposal development, audit evaluation, optimizer computation, and restoration costs all belong to the recursive process.

## D Secondary Program Decisions and CI Weighting

<table><tr><td>Program arm/read</td><td>Expected error (%)</td><td>Greedy error (%)</td><td>All-query pass (%)</td></tr><tr><td>Static reset</td><td>53.95</td><td>47.76</td><td>42.71</td></tr><tr><td>Static keep</td><td>29.30</td><td>15.49</td><td>81.77</td></tr><tr><td>Adapted reset</td><td>61.26</td><td>61.72</td><td>30.21</td></tr><tr><td>Adapted keep</td><td>24.61</td><td>16.04</td><td>80.73</td></tr></table>

Table 11: Saved secondary program-selection summaries, reused from LARC. A probability-weighted endpoint and a greedy decision need not order the trained arms in the same way.
<table><tr><td>Repository</td><td>Jobs</td><td>Commits</td><td>Weight per commit</td></tr><tr><td>NumPy</td><td>83</td><td>1</td><td>1/2</td></tr><tr><td>pandas</td><td>3,238</td><td>71</td><td>1/142</td></tr></table>

Table 12: CI population and aggregation. The 71-fold diference in commit weight follows directly from equal repository weighting. Jobs inside each commit contribute to that commit’s score.

## E Evidence Identities and Cutof

The manuscript draws its theory from the local MMLA v4 source, including its RTT, authoritative-memory, and dual-state modules; the public citation is fixed at arXiv:2606.28876v4. The local report source has SHA256

$$
\mathrm { d } 4 \mathrm { f } 2 4 3 0 5 \mathrm { d } \mathrm { c } \mathrm { d } 0 0 0 0 \mathrm { a } 2 0 3 \mathrm { d } \mathrm { c } \mathrm { b } \mathrm { e } 2 4 \mathrm { a } 4 \mathrm { d } \mathrm { c } \mathrm { d } \mathrm { b } 1 2 4 9 4 \mathrm { a } 7 \mathrm { a } 4 \mathrm { d } \mathrm { e } 5 \mathrm { d } \mathrm { e } { \mathrm { a } } 3 \mathrm { e } \mathrm { b } 0 6 \mathrm { e } \mathrm { b } \mathrm { b } 1 \mathrm { a } 1 \mathrm { c } 9 1 \mathrm { c } \mathrm { a } 8 9 9 \mathrm { a } 8 \mathrm { a } 2 \mathrm { c } \mathrm { b } 0 0 \mathrm { b } 0 \mathrm { c } \mathrm { b } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 \mathrm { d } \mathrm { c } \mathrm { b } 0 0 \mathrm { b } 0 \mathrm { c } \mathrm { b } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 \mathrm { a } 7 \mathrm { a } 8 \mathrm { b } 0 \mathrm { c } \mathrm { b } 0 \mathrm { b } 0 \mathrm { c } \mathrm { b } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 \mathrm { a } 9 9 \mathrm { a } 8 \mathrm { b } 0 \mathrm { b } 0 \mathrm { c } \mathrm { b } 0 0 \mathrm { b } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 \mathrm { a } 9 9 \mathrm { a } 8 \mathrm { b } 0 \mathrm { c } \mathrm { b } 0 0 \mathrm { b } 0 \mathrm { c } \mathrm { b } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 \mathrm { a } 8 9 9 \mathrm { a } 8 \mathrm { b } 0 \mathrm { b } 0 \mathrm { c } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 \mathrm { a } 8 7 \mathrm { a } 8 0 \mathrm { b } 0 \mathrm { c } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 0 \mathrm { b } 0 \mathrm { c } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 0 \mathrm { b } 0 \mathrm { c } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 0 \mathrm { b } 0 \mathrm { c } 1 \mathrm { a } 1 \mathrm { c } 9 \mathrm { a } 8 0 \mathrm { b } 0 \mathrm 
$$

The locally supplied LARC report source has SHA256

$$
\mathrm { f 7 c c f e 0 a f 9 5 1 6 8 f 2 e 4 f 4 0 e 2 0 8 b 3 0 3 3 f 8 4 2 f 1 2 c 0 6 6 1 b 8 4 4 6 6 6 1 e 3 9 7 1 1 b e 0 3 a b 9 a b 8 4 2 5 c o n s i n t e r m e s s e }
$$

and the public citation is arXiv:2609.40063v1. A local source identity is not a claim of byte identity with an arXiv-generated PDF. These report sources and the versioned aggregate records fix what was available for this synthesis.

The program-selection substrate is openbmb/MiniCPM5-1B-SFT, fixed at revision

$$
\mathrm { a 6 0 b 3 7 f 1 f c 4 0 9 c 5 4 e 1 e 3 3 7 b 0 7 2 3 a a a c 6 f 9 2 d f e c 0 }
$$

Its $\theta _ { 0 }$ tensor digest and always-enabled � byte and tensor digests are preserved in the accompanying model-lineage.json. That identity refers to the program-selection realization; it does not replace the separate task adapters of the SQL studies. The original assets must be checked on load rather than reconstructed by substitute training.

The completed SQL interaction result audit binds candidate

$$
\mathrm { c 4 c 1 f 3 3 c 6 6 d 3 2 9 a d 4 l c b 5 2 a c 4 e d b f 5 8 d 2 a 6 8 6 e c 2 }
$$

and the completed CoSQL correction-transfer result audit binds candidate

$$
1 6 \mathrm { c } 3 4 8 \mathrm { c } 5 8 6 5 \mathrm { a c } 9 \mathrm { e } \mathrm { e } 3 \mathrm { c } 2 0 0 \mathrm { e } \mathrm { d } 4 3 \mathrm { a } 0 2 6 7 8 8 \mathrm { d } 0 9 \mathrm { d } \mathrm { a c } 0 9 \mathrm { d } \mathrm { a e } \mathrm { f } 2 6 5 \mathrm { a } 1 \mathrm { c } \mathrm { d } 8 9 \mathrm { b } 9 1 \mathrm { a f a f b e d } 3 \mathrm { d } \mathrm { a c } 0 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { d } \mathrm { b } 9 \mathrm { b } 9 \mathrm { d } \mathrm { a c } 1 \mathrm { a b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { d } \mathrm { b } 9 \mathrm { b } 9 \mathrm { d } \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b } 9 \mathrm { b }
$$

The latter covers the closed correction-transfer recipe, not the separate contrast-training candidate still under evaluation at this report’s cutof. A source review, numerical save/restore check, statistical result review, and observed scientific efect remain separate forms of evidence.

The companion aggregate data preserve the values used to render the figures, and the evidence manifest records original relative paths, byte sizes, and SHA256 hashes. The source archive is self-contained for document compilation. Access to private experiment inputs and trajectories remains subject to their existing access scope; no new public release of those materials is implied by this report.

## References

[1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. arXiv:2306.13649v3, 2024. Version fixed at 2024-01-17.

[2] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hofman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. arXiv:1606.04474v2, 2016. Version fixed at 2016-11-30.

[3] Glenn W. Brier. Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review, 78(1):1–3, 1950. https: //doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

[4] Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le. Symbolic Discovery of Optimization Algorithms. arXiv:2302.06675v4, 2023. Version fixed at 2023-05-08.

[5] Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution. arXiv:2309.16797v1, 2023. Version fixed at 2023-09-28.

[6] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. arXiv:1703.03400v3, 2017. Version fixed at 2017-07-18.

[7] Yoav Freund and Robert E. Schapire. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. Journal ofComputer and System Sciences, 55(1):119–139, 1997. https://doi.org/10.1006/jcss.1997.1504.

[8] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685v2, 2021. Version fixed at 2021-10-16.

[9] Louis Kirsch and Jürgen Schmidhuber. Eliminating Meta Optimization Through Self-Referential Meta Learning. arXiv:2212.14392v1, 2022. Version fixed at 2022-12-29.

[10] Jiahao Lu and Mohan Kankanhalli. Agentic-TTT: Training test-time policy for test-time training. arXiv:2610.12002v1, 2026. Version fixed at 2026-10-08.

[11] Kevin Lu and Thinking Machines Lab. On-Policy Distillation. Thinking Machines Lab: Connectionism, 2025. October 2025; doi:10.64434/tml.20251026; accessed October 14, 2026.

[12] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegrefe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative Refinement with Self-Feedback. arXiv:2303.17651v2, 2023. Version fixed at 2023-05-25.

[13] Alex Nichol, Joshua Achiam, and John Schulman. On First-Order Meta-Learning Algorithms. arXiv:1803.02999v3, 2018. Version fixed at 2018-10-22.

[14] Juergen Schmidhuber. Goedel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements arXiv:cs/0309048v5, 2006. Version fixed at 2006-12-17.

[15] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366v4, 2023. Version fixed at 2023-10-10.

[16] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620v4, 2025. Version fixed at 2025-08-31.

[17] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. arXiv:1909.13231v3, 2020. Version fixed at 2020-07-01.

[18] Yanbo Wang, Jinhua Hao, Yuze Shi, Kun Yuan, and Ming Sun. No Time Like the Present: Agentic Test-Time Training for LLM Agents. arXiv:2607.03441v1, 2026. Version fixed at 2026-07-03.

[19] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256, 1992.

[20] Peng Xia, Rujun Han, Zifeng Wang, Yanfei Chen, Yufan Zhuang, Yoonho Lee, Chengsong Huang, Han Yu, Zhongying CuiZhu, Yifei Ming, Huaxiu Yao, Burak Gokturk, Tomas Pfister, and Chen-Yu Lee. RRSI: Regularized Recursive Self-Improvement of Agent Harnesses. arXiv:2609.24972v3, 2026. Version fixed at 2026-10-04.

[21] Shangjian Yin, Zehao Zhao, Kavosh Asadi, Rui Liu, Yuchen Lu, Shike Mei, Hang Cui, Luke Simon, Zhouxing Shi, and Hamed Firooz Recursive Self-Improvement via On-Policy Distillation for Reasoning. arXiv:2609.30652v1, 2026. Version fixed at 2026-09-25.

[22] Tao Yu, Rui Zhang, He Yang Er, Suyi Li, Eric Xue, Bo Pang, Xi Victoria Lin, Yi Chern Tan, Tianze Shi, Zihan Li, Youxuan Jiang, Michihiro Yasunaga, Sungrok Shim, Tao Chen, Alexander Fabbri, Zifan Li, Luyao Chen, Yuwen Zhang, Shreya Dixit, Vincent Zhang, Caiming Xiong, Richard Socher, Walter S Lasecki, and Dragomir Radev. CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain Natural Language Interfaces to Databases. arXiv:1909.05378v1, 2019. Version fixed at 2019-09-11.

[23] Tao Yu, Rui Zhang, Michihiro Yasunaga, Yi Chern Tan, Xi Victoria Lin, Suyi Li, Heyang Er, Irene Li, Bo Pang, Tao Chen, Emily Ji, Shreya Dixit, David Proctor, Sungrok Shim, Jonathan Kraft, Vincent Zhang, Caiming Xiong, Richard Socher, and Dragomir Radev. SParC: Cross-Domain Semantic Parsing in Context. arXiv:1906.02285v1, 2019. Version fixed at 2019-06-05.

[24] Mert Yuksekgonul, Daniel Koceja, Xinhao Li, Federico Bianchi, Jed McCaleb, Xiaolong Wang, Jan Kautz, Yejin Choi, James Zou, Carlos Guestrin, and Yu Sun. Learning to Discover at Test Time. arXiv:2601.16175v2, 2026. Version fixed at 2026-02-05.

[25] Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation. arXiv:2310.02304v3, 2024. Version fixed at 2024-08-16.

[26] Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. MemEvolve: Meta-Evolution of Agent Memory Systems. arXiv:2512.18746v1, 2025. Version fixed at 2025-12-21.

[27] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. arXiv:2505.22954v3, 2026. Version fixed at 2026-03-12.

[28] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv:2601.18734v3, 2026. Version fixed at 2026-03-20.

[29] Junyi Zou and Avrova Donz. LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models. arXiv:2609.40063v1, 2026. Version fixed at 2026-09-30.

[30] Junyi Zou and Avrova Donz. MMLA: Memory-Mediated Learning Architecture for Predictive Dual-State Adaptation. arXiv:2606.28876v4, 2026. Version fixed at 2026-09-14.