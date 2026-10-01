Evaluated instance: rank 4 hidden width 1,536 12,288 FP32 factor entries

# LARC: Low-Rank Adaptive Residual Connections for Learning in Frozen Models

Technical Report of MMLA

Junyi Zou1,\* • Avrova Donz1,2,\*

<sup>1</sup>MMLA-org <sup>2</sup>Communication University of China (CUC), No. 1 Dingfuzhuang East Street, Chaoyang District, Beijing 100024, China <sup>\*</sup>Equal contribution

Email: lior.j.zou@gmail.com · avrovadonz@icloud.com

September 29, 2026

## ABSTRACT

Low-Rank Adaptive Residual Connections (LARC) give a frozen model a compact numerical state that can learn from feedback. The map $h + B A h$ adds a low-rank correction to a hidden representation. A slow state � learns starting factors across tasks; a private fast state Φ copies them, changes with feedback, and resets to the trained initialization. This report specifies an input-side realization of the numerical policy carrier in Memory-Mediated Learning Architecture and examines its factor-space dynamics and learning lifetime.

We study a rank-4 input residual with 12,288 trainable parameters on a frozen MiniCPM5-1B-SFT substrate. In a fourcandidate program-selection task, two feedback-gradient steps reduce expected query execution error by 24.65 and 36.65 percentage points relative to resetting to the respective trained static and post-adaptation initializations. These development results cover 16 parameter groups and three paired training seeds. A direct support-loss selection rule is much more accurate, reaching 0.78125% error. In a repository-balanced chronological replay of public continuous-integration jobs, retaining online updates raises half-Brier loss from 0.1274 to 0.1808. A fixed follow-up intervention records same-batch non-descent and inconsistent future benefit from shrinking updates. Together, the algebra and measurements distinguish residual capacity, adaptation relative to a starting point, and usefulness on later decisions.

## LARC: a small residual with an explicit learning lifetime

![](images/2e92e90f360fea1a1b01d09335fcf950d98fc2921a7ab7886a7bcb8c79d29d4a.jpg)  
Figure 1: LARC at a glance. Slow training determines the starting factors; arrived feedback updates a private fast state; later predictions read that state through a frozen mixer. Copy, reset, and resume give these numerical changes a defined lifetime.

## 1 Introduction

Feedback often arrives while a task is still in progress. A program execution reveals that a proposed transformation is wrong. A completed workflow provides evidence about a later job. The next decision can use this observation through additional context, a stored record, or a change in the parameters that participate in the computation. LARC implements this last path through a small residual attached to a hidden representation. The question is whether retaining its feedback updates improves a later read, and under which learning conditions.

Consider a learner given four complete programs, one of which is the target. It assigns a probability to each candidate, and an executor returns all four errors on the available examples. A gradient update can reduce the probability-weighted error even when the next prompt and the most likely candidate remain unchanged. This setting makes a numerical feedback efect measurable without requiring program generation. It also raises concrete implementation questions: where is the change stored, how does it reach the next prediction, how long is it retained, and what behavior returns when it is removed? A learning state needs an initialization, an update process, a reader, and a lifetime.

Residual connections preserve a reference computation through an identity path [8]. $\mathrm { L A R C ^ { 1 } }$ uses LoRA’s two-factor low-rank parameterization [10] as a hidden-state correction $h \mapsto h + B A h$ . Its factors start from a learned slow state $\rho$ and are updated as a fast state $\Phi .$ Ending an episode or applying a reset restores the bound $\rho$

The separation between $\rho$ and Φ gives the learner two timescales, following the distinction between slow parameters and fast weights [1]. Slow training shapes a starting representation across tasks. Fast updates respond to observations in the current episode. Resetting to the trained $\rho$ removes those fast updates while preserving what slow training has learned. For this input residual, learning requires a gradient through the downstream mixer: its weights stay fixed, but its response to a changed input determines how the factors should change.

MMLA separates a numerical policy carrier from an authoritative memory carrier and assigns them distinct writers and reset domains [17]. LARC supplies a concrete realization of the policy carrier. Its interface is hidden-to-hidden; the mixer that follows it is a separate architectural choice. The present implementation reuses a Transformer language-model mixer as the frozen substrate. The residual interface and its learning lifetime carry the definition, while this mixer supplies the computation used in the experiments.

Our central empirical comparison trains two initializations using the same feedback, queries, residual placement, and paired random seeds. One objective evaluates queries at the slow initialization. The other evaluates them after two feedback updates, using a first-order MAML-style estimator. For each trained seed, retaining real-feedback updates reduces expected query error on held-out development parameter groups. The advantage of one training objective over the other is less consistent across seeds, and the direct symbolic rule remains much more accurate than either arm. We then follow the static initialization into a task-matched continuous-integration study, where preserving online updates is harmful relative to reset on the recorded queue. The contrast shows why feedback responsiveness, the quality of the starting policy, and sustained online benefit need separate measurements.

This report specifies and implements a rank-4 input residual with slow initialization and episode-local fast state, including its gradient path through a frozen mixer and its reset and resume operations. It compares static and post-adaptation training with paired initializations, then measures real and sham feedback crossed with keep and reset reads. The factor-space analysis explains how an input residual changes the computation and why identical factor products can have diferent subsequent updates. Six episodic initializations and three task-matched initializations were trained and saved along this sequence.

Reading guide. Section 2 describes the executable mechanism and learning procedure. Section 3 explains its factor-space dynamics, and Section 4 defines the feedback comparisons. Sections 5–7 follow the experiments in order. The implementation section and appendices give the configuration, task semantics, and restoration details.

## 2 LARC as a Learning State

## 2.1 Residual interface

Let $h \in \mathbb { R } ^ { d }$ be a hidden vector at an insertion boundary and let $F _ { \theta }$ denote the downstream frozen computation. A rank-� LARC applies

$$
\begin{array} { r } { T _ { \Phi } ( h ) = h + B _ { \Phi } A _ { \Phi } h , \quad \quad A _ { \Phi } \in \mathbb { R } ^ { r \times d } , \quad B _ { \Phi } \in \mathbb { R } ^ { d \times r } , \quad \quad f _ { \theta , \Phi } ( x ) = F _ { \theta } ( T _ { \Phi } ( E _ { \theta } ( x ) ) ) . } \end{array}\tag{1}
$$

The first map, $A _ { \Phi } .$ , reads � learned linear combinations of the hidden coordinates. The second, $B _ { \Phi }$ , writes their contribution back into the �-dimensional representation. Adding the original ℎ preserves the identity path. The correction lies in the column space of $B _ { \Phi }$ , although changing that correction can alter many outputs of the nonlinear mixer that follows it. Low rank describes the inserted linear correction, rather than the rank of the complete input-to-output computation.

For a sequence, the same factors act tokenwise. The factors are shared within one episode and isolated across episodes. We use $r = 4 , d = 1 5 3 6$ , and a fixed residual scale of one. There is no bias or nonlinearity inside this residual. The 12,288 factor entries require 49,152 bytes in FP32, excluding optimizers and runtime activations. Applying the two linear maps use $2 d r = 1 2 { , } 2 8 8$ multiply-accumulates per token, in addition to the frozen model computation and the residual addition.

Definition 2.1 (LARC state). The slow state $\rho = ( A _ { \rho } , B _ { \rho } , \nu )$ contains the learned initialization and its version �. The fast state $\Phi _ { e , k } = ( A _ { e , k } , B _ { e , k } )$ is a private copy for episode � after � feedback updates. Each episode binds one version of $\rho ,$ one update rule, and private optimizer and random-number state. Its reset restores that bound initialization.

This definition fixes the reference against which adaptation is measured. Before training, we sample $A _ { \rho }$ from $N ( 0 , 0 . 0 2 ^ { 2 } )$ and set $B _ { \rho } = 0$ . After training, both factors may be nonzero. Starting a new episode copies the trained factors, so the initial episode behavior includes slow learning. Numerical state can be retained across feedback steps while the observed prompt remains fixed.

## A trainable residual before the frozen mixer

Rank limits the correction subspace. The mixer still processes the complete modified representation.

![](images/e6f53b7804d67ca65163807ea90d6bb0fec8c34bf543a0210b828622f19b2eb2.jpg)  
Figure 2: The input residual and its learning path. The two factors read a four-dimensional projection and write a correction back into the full hidden representation. The orange path denotes backpropagation to the factors through the frozen downstream computation. Zero � makes the initial forward map the identity; an episode reset later restores the trained factors.

## 2.2 What changes, and on which timescale?

Three parameter collections participate in the evaluated learner. The common substrate contains the pretrained weights and the always-enabled semantic adapter �. Their values are fixed throughout the study. The slow LARC factors $\rho$ change across outer training batches. The fast factors Φ change within an episode or online session. A factor tensor belongs to exactly one of these roles at a time: a fast update operates on an owned copy, rather than modifying the slow reference in place.

<table><tr><td>State</td><td>What it contains</td><td>When it changes</td><td>What a fast reset does</td></tr><tr><td>Frozen substrate</td><td>Backbone and semantic adapter ω</td><td>Fixed during these experiments</td><td>Leaves it unchanged</td></tr><tr><td>Slow  $\rho _ { \nu }$ </td><td>Learned factors  $( A _ { \rho } , B _ { \rho } )$  and version</td><td>One outer update after a batch closes</td><td>Uses it as the reference</td></tr><tr><td>Fast  $\Phi _ { e , k }$ </td><td>Episode factors and private update state</td><td>Arrived-feedback update</td><td>Copies the bound  $\rho _ { \nu }$ </td></tr><tr><td>Explicit memory M</td><td>Completed cases in the CI study</td><td>Real-label arrival under its own rule</td><td>Keeps its separate lifetime</td></tr></table>

Table 1: The states read by the evaluated system. Explicit memory is disabled in the episodic study. In CI it provides the same causal case history to the compared policy states; resetting the numerical residual does not erase that history.

This distinction is useful even without meta-learning. An ordinarily trained residual can be copied into an episode and adapted from feedback. Meta-learning changes how the reference is obtained; it does not create the distinction between a reference and a temporary state. Conversely, a successful outer training loss does not show that keeping subsequent fast updates helps. That question requires the keep/reset comparison developed in Section 4.

## 2.3 Two timescales and two outer objectives

We use an initialization-based meta-learning formulation [6] to compare two outer objectives. For support feedback $S _ { e }$ and an outer query $Q _ { e }$ , the fast update is

$$
\Phi _ { e , 0 } = \mathrm { c o p y } ( \rho _ { \nu } ) , \qquad \Phi _ { e , k + 1 } = \Phi _ { e , k } - \eta \nabla _ { \Phi _ { e , k } } \mathcal { L } _ { S _ { e } } ( \Phi _ { e , k } ) .\tag{2}
$$

Our episodic experiments use two steps of SGD, $\eta = 0 . 1$ , with no momentum, decay, or inner gradient clipping. All episodes in an outer batch start from the same version of $\rho .$ Their factors have distinct storage and random-number streams. Only after every episode finishes do we aggregate query losses and update the slow factors.

We compare

$$
J _ { \mathrm { s t a t i c } } ( \rho ) = \mathbb { E } _ { e } \bigl [ \mathcal { L } _ { Q _ { e } } ( \rho ) \bigr ] ,\tag{3}
$$

$$
J _ { \mathrm { a d a p t e d } } ( \rho ) = \mathbb { E } _ { e } \left[ \mathcal { L } _ { Q _ { e } } ( U ^ { 2 } ( \rho , S _ { e } ) ) \right] .\tag{4}
$$

Static training computes the same support updates for matched work and observations, but its query loss is evaluated at $\rho .$ Adapted training evaluates the query at $\Phi _ { e , 2 }$ . We implement its first-order gradient through

$$
\widetilde { \Phi } _ { e , 2 } = \rho + \mathsf { s g } ( \Phi _ { e , 2 } - \rho ) , \qquad \widehat { \nabla _ { \rho } J } _ { \mathrm { a d a p t e d } } = \frac { 1 } { | \mathcal { B } | } \sum _ { e \in \mathcal { B } } \nabla _ { \Phi } \mathcal { L } _ { Q _ { e } } ( \Phi ) \big | _ { \Phi = \Phi _ { e , 2 } } .\tag{5}
$$

Equation (5) implements a first-order MAML-style estimator [12]. Here sg is stop-gradient: the forward value is the adapted state, and the backward Jacobian to $\rho$ is the identity. The support trajectory determines where the query gradient is measured; that gradient is passed to the corresponding slow factor entries without diferentiating through the two support updates. An exact MAML gradient would include their Jacobians. Detaching the fast state without the identity connection would instead disconnect the outer gradient from $\rho .$

Both objectives use AdamW [11] with learning rate $1 0 ^ { - 3 } , ( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 ) , \epsilon = 1 0 ^ { - 8 }$ , zero weight decay, and outer gradient-norm clipping at one. Each outer batch contains two episodes. We train exactly 256 outer updates for each of three paired initializations in each objective, retaining all six final states.

## One slow version, independent fast trajectories

The evaluated outer batch contains two episodes; each episode takes two support updates.

![](images/49ff2e3dfb54b875c19202a1ec4a2d943209d7d0680b464a108901c5d987150d.jpg)  
Figure 3: Slow learning and fast adaptation. Every episode in a batch begins at the same slow version and follows its own support trajectory. The objective chooses whether its query is evaluated at $\rho _ { \nu }$ or at the adapted factors. Query gradients are then aggregated into one slow update. At use time, the trained slow state stays fixed; fast resets and checkpoint restoration perform diferent operations.

## 2.4 Gradient flow, batching, and restoration

The model weights are frozen while the downstream function remains diferentiable with respect to its input. With $z = T _ { \Phi } ( h )$ an input-side gradient contains the Jacobian $\partial F _ { \theta } ( z ) / \partial z$ . Wrapping this computation in a no-gradient region removes the learning path. Our implementation caches only the frozen embeddings before LARC and uses non-reentrant activation checkpointing for the downstream mixer. The output head is evaluated at answer positions. Padding masks and position indice remain specific to each sequence.

An episodic physical batch has eight rows: two independent episodes, each with four candidate continuations. Each row selects its episode’s own factors. A permutation of episode rows must permute the corresponding predictions without changing them. This isolation is also required for optimizers and random-number state; ordinary batching must not turn independent learners into one shared learner.

A slow checkpoint stores actual factor values, version, original initialization, outer optimizer, random-number state, model identity, and configuration. A live-session checkpoint additionally stores the fast factors, feedback cursor, pending feedback, and any separate explicit memory. Reset and resume serve diferent purposes. Reset restores the bound slow initialization; resume restores the current fast trajectory. We tested midpoint restoration and final reload as part of the recorded training and replay runs.

For example, an episode saved after its first support update must resume at $\Phi _ { e , 1 }$ before taking its second update. Replacing that saved state with $\rho _ { \nu }$ would replay a diferent trajectory. A reset intervention deliberately makes this replacement just before the designated query. In a delayed-feedback session, restoration must also retain which jobs have already been predicted and which labels have arrived, because those determine the information available to the next update.

## 3 Properties of the Residual and Its Updates

The following statements describe the real-valued residual in Equation (1). They explain the parameterization and initialization, while the implemented model combines FP32 factors with BF16 activations.

## 3.1 Relation to LoRA: placement and learning lifetime

LoRA parameterizes a weight correction [10], writing a linear layer as $( W + \Delta W ) I$ ℎ with $\Delta W = U V$ . LARC uses the same low-rank factorization to transform its input. If the next operation is a linear map $W \in \mathbb { R } ^ { m \times d }$ , then

$$
W T _ { \Phi } ( h ) = W ( h + B _ { \Phi } A _ { \Phi } h ) = W h + ( W B _ { \Phi } ) A _ { \Phi } h .\tag{6}
$$

The efective correction $\Delta W = ( W B _ { \Phi } ) A _ { \Phi }$ has rank at most �, and its columns lie in the image of �. Conversely, for any rank-at-most-� correction with columns in that image, choose a rank factorization $\Delta W = U V$ whose left factor lies in the image, padding with zeros to width � if needed. Solving � $B _ { \Phi } = U$ and setting $A _ { \Phi } = V$ realizes that correction. When � has full row rank, every rank-at-most-� weight correction is representable. A square invertible � is a special case, with $B _ { \Phi } = W ^ { - 1 } U$ . When � is not full row rank, corrections outside its image remain inaccessible. These statements concern forward representations; the factor coordinates and their gradient updates can still difer.

Our evaluated residual precedes the full nonlinear mixer. The mixer must be evaluated on the changed representation; the linear identity above does not generally merge this transformation into an arbitrary internal LoRA site. Table 2 separates the placement of the correction from its learning lifetime.

<table><tr><td>Choice</td><td>Evaluated input LARC</td><td>Weight-space LoRA</td></tr><tr><td>Forward correction</td><td> $F _ { \theta } \big ( ( I + B A ) h \big )$ </td><td> $( W + U V ) h$  at selected linear maps</td></tr><tr><td>Linear overlap</td><td> $\Delta W = ( W B ) A$ </td><td>Free factored  $\Delta W = U V$ </td></tr><tr><td>Placement here</td><td>One hidden-to-hidden map before the full mixer</td><td>Depends on the chosen weight sites</td></tr><tr><td>Learning lifetime</td><td>Trained  $\rho ;$  private, adapted Φ; reset to ρ</td><td>Can use the same initialization, adaptation, and reset procedure</td></tr></table>

Table 2: Placement and lifetime are separate choices. The linear overlap is given by Equation (6); the evaluated LARC precedes the full nonlinear mixer.

## 3.2 Identity initialization and the first learning step

Let $C = B A$ and write the diferentiable task loss as ℓ(�), with matrix gradien $G _ { C } = \nabla _ { C } \ell ( C )$ under the Frobenius inner product. The chain rule gives

$$
\nabla _ { \boldsymbol { A } } \boldsymbol { \ell } = \boldsymbol { B } ^ { \top } \boldsymbol { G } _ { C } , \qquad \nabla _ { \boldsymbol { B } } \boldsymbol { \ell } = \boldsymbol { G } _ { C } \boldsymbol { A } ^ { \top } .\tag{7}
$$

Proposition 3.1 (First update from an identity residual). With simultaneous SGD, initial $B _ { 0 } = 0 ,$ , and initial $A _ { 0 } ,$ the first update satisfies

$$
A _ { 1 } = A _ { 0 } , \qquad B _ { 1 } = - \eta G _ { C , 0 } A _ { 0 } ^ { \top } , \qquad C _ { 1 } = - \eta G _ { C , 0 } A _ { 0 } ^ { \top } A _ { 0 } .\tag{8}
$$

$H A _ { 0 }$ has independent zero-mean entries of variance $\sigma ^ { 2 } ,$ , and $G _ { C , 0 }$ depends on the factors only through $C _ { 0 } = 0 ;$ , then $\mathbb { E } [ C _ { 1 } ] = - \eta r \sigma ^ { 2 } G _ { C , 0 } .$

□

Proof. Substitute $B _ { 0 } = 0$ into Equation (7). Then multiply the updated factors and use $\mathbb { E } [ A _ { 0 } ^ { \top } A _ { 0 } ] = r \sigma ^ { 2 } I .$

The first step learns within the row space selected by $A _ { 0 } .$ . In expectation, that initial update is proportional to the full matrix gradient, although each realization remains rank constrained. The zero first-step gradient of � is therefore expected. A nonzero � gradient additionally requires $G _ { C , 0 } A _ { 0 } ^ { \top } \ne 0 ;$ random initialization does not replace checking the actual task gradient. Setting both factors to zero yields zero task gradients for both and leaves this bilinear branch stationary under plain SGD.

This calculation concerns the first task-gradient update. The outer optimizer, clipping, and mixed-precision arithmetic can change the numerical step. Once slow training has made $B _ { \rho }$ nonzero, both factors can receive a task gradient at the start of a new episode.

## 3.3 Rank controls capacity, not displacement

Proposition 3.2 (Rank and norm). At any state, rank $( B A ) \leq r$ and

$$
\| T _ { \Phi } ( h ) - h \| _ { 2 } \leq \| B \| _ { 2 } \| A \| _ { 2 } \| h \| _ { 2 } .\tag{9}
$$

For two states with rank-at-most-� residuals, the rank of their diference is at most $2 r .$ This upper bound is attainable when $d \geq 2 r$

Proof. The first two assertions follow from the rank inequality for products and submultiplicativity of the operator norm. For the diference, use rank $( C _ { 1 } - C _ { 0 } ) \le \mathrm { r a n k } ( C _ { 1 } ) + \mathrm { r a n k } ( C _ { 0 } )$ . Two rank-� diagonal matrices supported on disjoint sets of coordinates attain 2�. □

Rank four therefore bounds each instantaneous residual to rank four, while the change between two learned residuals can have rank eight. Multiplying one factor by an arbitrarily large constant preserves its rank and increases its norm. A small state is compatible with a large representation displacement. If the downstream map is �-Lipschitz on the relevant segment, Equation (9) yields an output bound multiplied by �. Neither a uniform � nor bounded factor norms follows from the rank choice alone.

## 3.4 Equal forward maps can have diferent learning dynamics

For any invertible $R \in \mathbb { R } ^ { r \times r }$ , the factors $( R A , B R ^ { - 1 } )$ represent the same �. Euclidean gradient steps generally depend on this choice of coordinates. With a positive scalar �, consider $A ^ { \prime } = c A , B ^ { \prime } = B / c$ . At the common forward map, one simultaneous SGD step changes � by

$$
C _ { + } ^ { \prime } - C = - \eta \left( c ^ { - 2 } B B ^ { \top } G _ { C } + c ^ { 2 } G _ { C } A ^ { \top } A \right) + \eta ^ { 2 } G _ { C } A ^ { \top } B ^ { \top } G _ { C } .\tag{10}
$$

These factor-space properties also apply to other bilinear low-rank updates, including LoRA. The first-order terms depend on � even though the current prediction does not. Initialization scale, factor norms, optimizer state, and step size are therefore part of the learning specification. Saving only the product �� is enough to reconstruct a forward map, but is generally insuficient to reproduce the next factor-space update.

For an �-smooth factor-space objective, the standard descent bound is

$$
\ell ( \Phi - \eta \nabla \ell ( \Phi ) ) \leq \ell ( \Phi ) - \eta \left( 1 - \frac { L \eta } { 2 } \right) \| \nabla \ell ( \Phi ) \| ^ { 2 } .\tag{11}
$$

It requires smoothness along the step and $0 < \eta < 2 / L$ to obtain a strict decrease for a nonzero gradient. Rank supplies neither condition. Moreover, a decrease on the arrived support batch does not determine the loss on later jobs. The experiments measure both quantities.

## 4 Measuring Feedback Adaptation

We evaluate numerical adaptation using a feedback intervention crossed with a state-retention intervention. Real feedback preserves its binding to candidate programs. Sham feedback independently permutes the four observed losses uniformly at each inner step, including natural fixed points. Both branches start from the same trained initialization, take the same number of updates, and then either keep the fast factors or reset to their own $\rho$ immediately before the query read.

Let $R _ { z , a }$ be the query risk for feedback condition $z \in$ {real, sham} and read state $a \in \{ \mathrm { k e e p } , \mathrm { r e s e t } \}$ . Define

$$
\begin{array} { r } { G = R _ { \mathrm { r e a l , r e s e t } } - R _ { \mathrm { r e a l , k e e p } } , } \end{array}\tag{12}
$$

$$
D = G - \left( R _ { \mathrm { s h a m , r e s e t } } - R _ { \mathrm { s h a m , k e e p } } \right) .\tag{13}
$$

Positive $G$ measures benefit from retaining real-feedback state relative to reset. Positive � measures the corresponding diference from the permutation intervention. The two reset reads are identical for a deterministic query with the same substrate and initialization, so they can share one physical evaluation. Their logical identities remain distinct.

The query read receives the public prompt, candidate programs, frozen model, and designated factor state. It receives no feedback transcript, inner optimizer, live inner random stream, cached post-residual hidden representation, or stale key–value cache. Explicit memory is disabled in the episodic study. These conditions make retention of Φ the manipulated path from feedback to the query. A text-feedback baseline is evaluated separately with the feedback rendered in its prompt.

## Identify what the feedback writes into the fast state

![](images/01aa231699ce8479031b43c21a15b80c0ca83e0a64b6a1acee5b9e750c6cebff.jpg)  
The query receives its prompt, candidates, frozen substrate, and designated factors. The feedback transcript and fast-dependent caches have no separate path into this read.  
Figure 4: Feedback binding crossed with state retention. Both feedback branches execute two updates. The reset read then restores its own trained initialization immediately before the query. With the remaining read inputs fixed, � measures the efect of retaining real-feedback state, while � compares this efect with independently permuted feedback.

Under the declared query-read controls, these interventions estimate retention and feedback-binding efects. Keeping the state tests the efect of carrying the numerical update into the query. Permuting feedback tests the efect of its binding to the candidate programs. An update can respond to the correct feedback and still move farther from a good initial policy. The reset risk therefore remains the reference in �, with � reported alongside it.

The outer-objective contrast is

$$
\Delta _ { \mathrm { o b j } } = R _ { \mathrm { r e a l , k e e p } } ^ { \mathrm { s t a t i c } } - R _ { \mathrm { r e a l , k e e p } } ^ { \mathrm { a d a p t e d } } .\tag{14}
$$

It asks a diferent question from � and �. Similar outer objectives can both yield useful feedback adaptation, and a positive diference between real and permuted updates can coexist with harmful real updates. We retain all three comparisons in the results.

## 5 Grouped Program-Selection Study

## 5.1 Model and task construction

All six models share MiniCPM5-1B-SFT revision a60b37f1fc409c54e1e337b0723aaac6f92dfec0 and an always-enabled, previously trained semantic adapter �. This frozen adapter uses rank-16 query/value additions in layers 16–23, with scale 32/16. It is part of the common substrate. The input LARC has rank four and is the only component trained in this study. Thus the rank-16 semantic adapter supplies fixed task semantics, while the rank-4 input residual is the learning state under evaluation.

Each episode asks the model to choose among four complete public programs in either an arithmetic or a list-processing family. Programs are instantiated from the public configuration using four fixed recipes per family. The learner receives four input/output examples and an executor returns each candidate’s support error fraction. A query contains 20 new inputs to the same transformation, disjoint from support. Query outcomes supervise outer training or subsequent scoring; they are absent from the inner feedback and policy prompt.

We group arithmetic tasks by the active pair (scale, ofset) and list tasks by (minimum, limit). All four recipes for a parameter group stay in the same split. A conservative inventory excludes historical active pairs and the reachable support of the earlier generator. The arithmetic candidate space contains 350 pairs, of which 245 remain after exclusion. The list space contains 78 pairs, of which 34 remain. We allocate 34 per family: 16 training, eight development, eight reserved final, and two preflight groups. Development consequently has 16 independent parameter groups and 64 recipe episodes. The eight recipes are familiar; the held-out variation is in their active parameters.

The policy scores each complete candidate and its end-of-sequence token using mean token log probability, then applies a temperature-one softmax over the four scores:

$$
s _ { j } ( \Phi ) = \frac { 1 } { | a _ { j } | } \sum _ { t = 1 } ^ { | a _ { j } | } \log p _ { \theta , \Phi } ( a _ { j , t } \mid x , a _ { j , < t } ) , \qquad \pi _ { j } ( \Phi ) = \frac { e ^ { s _ { j } ( \Phi ) } } { \sum _ { m = 1 } ^ { 4 } e ^ { s _ { m } ( \Phi ) } } .\tag{15}
$$

For externally supplied execution losses $\ell _ { j } ( S )$

$$
{ \mathcal L } _ { S } ( \Phi ) = \sum _ { j = 1 } ^ { 4 } { \pi } _ { j } ( \Phi ) \ell _ { j } ( S ) .\tag{16}
$$

The query loss uses the same expectation with query execution losses. Exact enumeration gives a diferentiable finite-policy objective without sampling programs or passing gradients through a discrete executor. This is a program-selection evaluation.

## 5.2 A concrete feedback update

For illustration, take arithmetic scale 2, ofset 3, and clipping to [−16, 16]. The four candidate programs are

$$
\begin{array} { l } { { f _ { 0 } ( x ) = 2 x + 3 , } } \\ { { f _ { 2 } ( x ) = 2 \operatorname { c l i p } ( x ) + 3 , } } \end{array}
$$

$$
\begin{array} { l } { { f _ { 1 } ( x ) = 2 ( x + 3 ) , } } \\ { { f _ { 3 } ( x ) = \mathrm { c l i p } ( 2 x ) + 3 . } } \end{array}
$$

Suppose the target is $f _ { 0 }$ and the four support inputs are 2, 24, −20, 5. The target outputs are 7, 51, −37, 13. Candidate $f _ { 1 }$ disagrees on all four inputs; $f _ { 2 }$ and $f _ { 3 }$ each disagree on the two large-magnitude inputs. The executor therefore supplies the loss vector $( 0 , 1 , { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } )$ , together with its candidate binding. The policy uses that vector through Equation (16).

The derivative with respect to a candidate energy is particularly simple:

$$
\frac { \partial \mathcal { L } _ { S } } { \partial s _ { j } } = \pi _ { j } \big ( \ell _ { j } ( S ) - \mathcal { L } _ { S } \big ) .\tag{17}
$$

In energy coordinates, candidates with below-average execution error are favored. The actual optimizer changes � and $B ,$ so its step is also shaped by the model Jacobian from those factors to all four energies. A finite factor update need not realize an independent increase or decrease of each energy. This connection between an execution error and the factor-space gradient is the learning mechanism tested here.

On the query read, the updated factors change candidate probabilities while the support losses remain outside the prompt. The evaluator then applies the candidate programs to held-out inputs; for $x = 3 ,$ the target output is 9. Resetting the factors first instead reads the trained starting policy. These are the real/keep and real/reset cells. The symbolic baseline uses the same support vector directly to select $f _ { 0 }$ , which also explains why that baseline is strong on this finite task.

## 5.3 Analysis and selection rule

The primary endpoint is real/keep expected query error. We average recipes within each parameter group, weight the two task families equally, and pair the three training seeds. The seeds are 2026092811, 2026092812, and 2026092813, shortened to 2811–2813 in tables. Each model completes all 256 outer updates before development predictions are generated. Development query scores are opened after predictions have been saved. Reserved final cases are not constructed or read in this study.

We use 10,000 paired bootstrap resamples of complete parameter groups, stratified by family, keeping all recipes, seeds, objectives, and intervention cells together. Five 99% percentile intervals cover $\Delta _ { \mathrm { o b j } }$ and the two $G , D$ pairs, with nominal Bonferroni family coverage of 95%. The materiality rule requires a point estimate of at least three percentage points and a lower interval bound above zero. Feedback eligibility additionally requires positive � and � for each training seed. These intervals quantify task-group variation conditional on the three trained seeds.

Among eligible objectives, the preregistered selection rule maximizes the worst-seed value of min(�, �). Measured training-plus-adaptation cost breaks a tie within $1 0 ^ { - 6 }$ , followed by the static objective if cost also ties. This rule chooses an objective’s complete seed group rather than its best individual model.

## 5.4 Results

The adapted objective achieves 24.61% mean error versus 29.30% for the static objective (Table 3). Its paired improvement is 4.69 percentage points with a 99% task-group interval of [1.19, 8.88], satisfying the registered development rule. Across seeds, the improvements are 1.45, −10.46, and 23.09 points. Thus the mean objective advantage is accompanied by a reversal for one initialization and a larger seed standard deviation.

Both objectives yield positive feedback benefits (Table 4). Static training obtains $G = 2 4 . 6 5$ and $D = 2 8 . 3 6$ points; adapted training obtains $G = 3 6 . 6 5$ and $D = 3 5 . 7 7$ points. Every seed has positive �, �, and all four contrasts satisfy the materiality and interval rule. The selected static group has a worst-seed min $( G , D )$ of 20.16 points, compared with 18.64 for the adapted group. Selection therefore follows stability of the feedback gain, even though adapted training has lower mean error.

<table><tr><td>Objective</td><td>2811</td><td>2812</td><td>2813</td><td>Mean</td><td>Seed SD</td></tr><tr><td>Static</td><td>21.90</td><td>31.47</td><td>34.53</td><td>29.30</td><td>6.59</td></tr><tr><td>Adapted</td><td>20.46</td><td>41.93</td><td>11.44</td><td>24.61</td><td>15.66</td></tr></table>

Table 3: Post-adaptation development error (%), lower is better. Each column evaluates the same 16 parameter groups. Seed SD is the sample standard deviation across the three trained initializations.

<table><tr><td>Contrast</td><td>Mean</td><td>99% group interval</td><td>2811</td><td>2812</td><td>2813</td></tr><tr><td> $\Delta _ { \mathrm { o b j } }$ </td><td>4.69</td><td>[1.19, 8.88]</td><td>1.45</td><td>-10.46</td><td>23.09</td></tr><tr><td>Static G</td><td>24.65</td><td>[19.39, 29.26]</td><td>31.43</td><td>20.16</td><td>22.35</td></tr><tr><td>Static D</td><td>28.36</td><td>[21.85, 34.93]</td><td>37.76</td><td>26.78</td><td>20.55</td></tr><tr><td>Adapted G</td><td>36.65</td><td>[31.65, 41.20]</td><td>40.52</td><td>18.64</td><td>50.78</td></tr><tr><td>Adapted D</td><td>35.77</td><td>[30.33, 41.01]</td><td>40.21</td><td>19.97</td><td>47.13</td></tr></table>

Table 4: Paired gains in percentage points, positive is favorable. The bootstrap unit is a complete parameter group. Training seeds are kept paired within each resample.

The larger � of the adapted objective includes a diferent starting point. Its reset error is 61.26%, versus 53.95% for static training. The 12.00-point diference between their feedback gains decomposes as

$$
G _ { \mathrm { a d a p t e d } } - G _ { \mathrm { s t a t i c } } = \underbrace { ( 6 1 . 2 6 - 5 3 . 9 5 ) } _ { \mathrm { r e s e t ~ d i f f e r e n c e } 7 . 3 1 } + \underbrace { ( 2 9 . 3 0 - 2 4 . 6 1 ) } _ { \mathrm { e n d p o i n t ~ a d v a n t a g e } 4 . 6 9 } \mathrm { p e r c e n t a g e ~ p o i n t s } ,\tag{18}
$$

with the displayed values rounded. The adapted objective starts worse and ends better on average. The objective contrast measures the latter advantage; it cannot be replaced by the full diference in feedback gains.
<table><tr><td>Objective</td><td>Real/keep</td><td>Real/reset</td><td>Sham/keep</td><td>Sham/reset</td><td>Text</td><td>Rule</td></tr><tr><td>Static</td><td>29.30</td><td>53.95</td><td>57.67</td><td>53.95</td><td>61.78</td><td>0.78</td></tr><tr><td>Adapted</td><td>24.61</td><td>61.26</td><td>60.38</td><td>61.26</td><td>61.03</td><td>0.78</td></tr></table>

Table 5: Descriptive development errors (%). Text renders the program-to-feedback mapping in the prompt at fixed $\rho .$ Rule chooses the lowest-support-error program, with first-index tie breaking. All methods receive the same external support information.

The symbolic support rule reaches 0.78125% error, considerably below either learned policy (Table 5). Exact enumeration makes that rule a strong comparator: it can directly use the same four execution losses that drive the gradient. LARC produces a numerical feedback efect in this setting, while program selection itself has little remaining error for a learned system to remove relative to this baseline. Figure 5 places the objective comparison next to the retained-state and feedback-binding gains.

# Episodic adaptation: objective choice and feedback benefit

16 held-out parameter groups three paired training seeds saved development results

(a) Error after real feedback  
![](images/b97d5d212f797fcc32133f0a9ceaccdb030ab9620735cffa203926deafb60fea.jpg)

(b) Paired gains with 99% group intervals  
![](images/e59de790c0abf459a8e0ec9fc383d26115122049330b3d4baeeef3191722701a.jpg)  
Intervals resample task groups and retain all three seeds.

Figure 5: Two questions in the episodic study. Left: each line joins one paired initialization under static and adapted training; the dashed black line joins their means. Right: the five recorded gains and 99% task-group intervals. The adapted objective improves the mean query error, while both objectives support useful real-feedback adaptation. The intervals retain the three trained seeds in every resample.
<table><tr><td>Objective</td><td>Read state</td><td>Expected error</td><td>Greedy error</td><td>All query cases correct</td></tr><tr><td>Static</td><td>Reset</td><td>53.95</td><td>47.76</td><td>42.71</td></tr><tr><td>Static</td><td>Adapted state</td><td>29.30</td><td>15.49</td><td>81.77</td></tr><tr><td>Adapted</td><td>Reset</td><td>61.26</td><td>61.72</td><td>30.21</td></tr><tr><td>Adapted</td><td>Adapted state</td><td>24.61</td><td>16.04</td><td>80.73</td></tr></table>

Table 6: Additional descriptive development measures (%), reconstructed from saved predictions. Greedy error evaluates the highest-probability supplied program; the last column is the fraction of episodes where that program passes all 20 query inputs. Group/family weights and paired seeds match the primary analysis. These are secondary measures, not free-form generation.

The expected-risk endpoint also difers from a greedy decision. After feedback, the static and adapted models have greedy query errors of 15.49% and 16.04%, respectively (Table 6). Their ordering is diferent from expected error. This distinction matters because changing probability mass across the four complete programs can improve expected risk without changing the most probable program.

## 6 Delayed Feedback in Public CI Workflows

## 6.1 Chronological replay and task-matched training

The second study predicts whether a public continuous-integration job will finish as success, failure, cancelled, or other. A job is predicted at its recorded start time. Its label becomes available at completion. At a timestamp with both starts and completions, predictions precede feedback processing. Updates use the prompt saved at prediction time. This ordering prevents later outcomes or memory contents from entering an earlier prediction.

The training queue has 2,678 jobs from 60 repository–commit groups. The audit queue has 3,321 jobs from 72 groups: 83 NumPy jobs from one commit and 3,238 pandas jobs from 71 commits. The NumPy training and audit dates are May 14 and May 16, 2026; the pandas dates are September 17 and September 19. Time is ordered within each repository. Because the shared training pool includes September pandas jobs, this is a repository-local retrospective evaluation, rather than a global-calendar prospective deployment.

Each selected static initialization receives one epoch of task-matched supervised training, using both no-memory and causal-memory views of each job. The loss is categorical cross-entropy on the four candidate-label energies, weighted equally by repository, commit within repository, job within commit, and the two views. AdamW uses the episodic outer hyperparameters. Each seed completes 670 additional updates, moving from slow version 256 to 926. The resulting three slow states are fixed during audit.

The online policy uses SGD with step size 0.1 on arrived-label cross-entropy, with independent fast states for each path and repository. A separate memory stores at most 16 completed cases within 16 KiB and exposes at most two, preferring the same workflow and job name. It is updated from real completed labels in every relevant branch. The memory uses FIFO eviction;

reading prefers the latest matching cases and fills any remaining slot with recent cases. Commit identifiers are exposed in the prompt, but do not impose a hard version filter. The policy-state intervention keeps online factors $( P _ { 1 } )$ or reads at the trained initialization $( P _ { 0 } )$ . The memory intervention provides or omits these case records $( M _ { 1 }$ or $M _ { 0 } )$ . A permuted-label path draws a fresh, independent uniform permutation of the four labels for each arrived job using its private random stream, then applies the inverse permutation to that job’s target. It retains the same real memory. There is no single fixed permutation shared by a completion batch.

All jobs completing at the same timestamp form one feedback batch. Stable identifiers determine serialization within a timestamp, without implying finer temporal resolution. For each active path, gradients are the mean cross-entropy over its arrived jobs; physical chunks accumulate that same mean. They are not divided by the number of paths or seeds and do not use the commit weights of the outer training objective. Each path then takes one functional SGD step with no momentum, decay, or gradient clipping. All factor updates finish before real cases are committed to memory.

At audit start, task-trained policy paths copy their own version-926 . Case memory is initialized from each repository’s completed training history. The original and calibrated reference policies first process the pooled training labels through their own fast updates, then fork into independent audit sessions. HEDGE4 also processes the same pooled training history before the audit. Historical supervision is therefore available to every comparator through its declared learning mechanism.

HEDGE4 is the four-expert comparator constructed for this study, using an exponential-weights mixture [7]. Its experts are a smoothed workflow-frequency prior, a last-label predictor for the workflow/job key, a last-label predictor for the commit/workflow/job key, and hashed-feature logistic regression. It updates mixture weights using prediction-time losses after labels arrive. We evaluate this delayed-feedback combination by its observed prediction loss. Every method receives the same externally available history; their computation costs difer.

This replay extends the lifetime of the fast state. In the episodic study, two updates precede one designated query. Here each completed batch can update the same factors that will be used for many later jobs. The distribution of arriving labels can difer from the distribution of jobs currently starting, and their order is set by completion times. A locally fitted update can therefore carry evidence from a narrow or delayed part of the workload into a diferent prediction population.

## 6.2 Prediction quality and retained state

The primary prediction loss is the Brier score [5] with a factor of $1 / 2$ normalization (half-Brier),

$$
b ( p , y ) = { \textstyle \frac { 1 } { 2 } } \sum _ { c = 1 } ^ { 4 } ( p _ { c } - { \bf 1 } [ c = y ] ) ^ { 2 } .\tag{19}
$$

We average jobs within commits, commits within repositories, and the two repositories equally. Seed means and sample standard deviations are reported separately. The primary success rule requires HEDGE4 minus $P _ { 1 } M _ { 1 }$ to average at least 0.0 and be positive for all three seeds.

<table><tr><td>Path</td><td>2811</td><td>2812</td><td>2813</td><td>Mean</td><td>Seed SD</td></tr><tr><td>POMO</td><td>0.1291</td><td>0.1196</td><td>0.1372</td><td>0.1286</td><td>0.0088</td></tr><tr><td>P1M0</td><td>0.1667</td><td>0.1542</td><td>0.1970</td><td>0.1726</td><td>0.0220</td></tr><tr><td>P0M1</td><td>0.1162</td><td>0.1174</td><td>0.1487</td><td>0.1274</td><td>0.0184</td></tr><tr><td>P1M1</td><td>0.1839</td><td>0.1658</td><td>0.1927</td><td>0.1808</td><td>0.0137</td></tr><tr><td>PERM_M1</td><td>0.3193</td><td>0.2867</td><td>0.3854</td><td>0.3305</td><td>0.0503</td></tr><tr><td>ORIGINAL</td><td>0.1681</td><td>0.1646</td><td>0.1633</td><td>0.1653</td><td>0.0025</td></tr><tr><td>CALIBRATED</td><td>0.1576</td><td>0.1594</td><td>0.1534</td><td>0.1568</td><td>0.0030</td></tr><tr><td>HEDGE4</td><td>0.1024</td><td>0.1024</td><td>0.1024</td><td>0.1024</td><td>0.0000</td></tr></table>

Table 7: Full CI replay, grouped half-Brier loss. PERM\_M1 uses permuted fast-update labels and real memory. ORIGINAL retains the pre-CI slow initialization; CALIBRATED adds a four-class bias fitted on historical data. Both controls also receive historical and online fast updates.

The full path scores 0.1808 compared with HEDGE4’s 0.1024, a gain of −0.0784 that is negative for every seed. Reading at the trained initialization with real memory scores 0.1274. Consequently, retaining the real-feedback numerical state has $G = - 0 . 0 5 3 4$ . Real feedback still outperforms permuted feedback by $D = 0 . 1 4 9 7$ . These measurements separate two observations: correct feedback is less harmful than permuted feedback, while both continuous numerical updating and the full path remain worse than their respective comparators.

Without explicit memory, retention also increases loss, from 0.1286 to 0.1726. The memory efect and policy-by-memory interaction vary in sign across seeds. NumPy contributes one all-success commit; the equal-repository average gives that single group half the weight. A pandas commit receives weight 1/142, so the single NumPy commit carries 71 times its weight We report this queue descriptively, without treating training seeds as additional independent jobs or attaching a population confidence interval to the two-repository result.

The decomposition locates the retained-state harm primarily in pandas (Table 8). Keep slightly improves the one all-success NumPy commit, from 0.0300 to 0.0273. For pandas it raises commit-weighted loss from 0.2248 to 0.3343. Its job-weighted

<table><tr><td>Repository</td><td>Label</td><td>Jobs</td><td>Commits</td><td>Reset</td><td>Keep</td><td>Permuted</td><td>HEDGE4</td></tr><tr><td>NumPy</td><td>All</td><td>83</td><td>1</td><td>0.0300</td><td>0.0273</td><td>0.2656</td><td>0.0277</td></tr><tr><td>NumPy</td><td>Success</td><td>83</td><td>1</td><td>0.0300</td><td>0.0273</td><td>0.2656</td><td>0.0277</td></tr><tr><td>pandas</td><td>All</td><td>3238</td><td>71</td><td>0.2248</td><td>0.3343</td><td>0.3954</td><td>0.1772</td></tr><tr><td>pandas</td><td>Success</td><td>2310</td><td>63</td><td>0.0401</td><td>0.2048</td><td>0.3926</td><td>0.0702</td></tr><tr><td>pandas</td><td>Failure</td><td>32</td><td>12</td><td>0.9386</td><td>0.7644</td><td>0.3842</td><td>0.7128</td></tr><tr><td>pandas</td><td>Cancelled</td><td>704</td><td>18</td><td>0.8475</td><td>0.6987</td><td>0.4050</td><td>0.5847</td></tr><tr><td>pandas</td><td>Other</td><td>192</td><td>57</td><td>0.2692</td><td>0.7327</td><td>0.4179</td><td>0.0574</td></tr></table>

Table 8: Saved CI half-Brier losses by repository and observed label, averaged over the three seeds. Rows labeled All weight commits equally within a repository; individual-label rows weight jobs equally within that label. Reset, Keep, and Permuted all use real memory Label strata are descriptive and overlap in their commit membership.

losses are lower on failures and cancellations but higher on successes and other outcomes. These strata describe where the error occurs; they do not identify which update caused it.
<table><tr><td>Aggregation</td><td>Reset</td><td>Keep</td><td>Permuted</td><td>HEDGE4</td></tr><tr><td>Repository then commit (primary)</td><td>0.1274</td><td>0.1808</td><td>0.3305</td><td>0.1024</td></tr><tr><td>Commit (descriptive)</td><td>0.2221</td><td>0.3300</td><td>0.3936</td><td>0.1751</td></tr><tr><td>Job (descriptive)</td><td>0.2329</td><td>0.3410</td><td>0.3934</td><td>0.1836</td></tr></table>

Table 9: Weighting sensitivity on the same saved 3,321 predictions per seed. The first row remains the registered primary measure. The other rows are descriptive; neither introduces new jobs, retraining, or a replacement success rule.

The absolute scores depend on repository weighting, but the ordering HEDGE4, reset, keep, permuted is preserved under all three aggregations (Table 9). Thus the observed harm of continuous retention is not removed by weighting commits or jobs globally.

## 7 Separating Update Magnitude from Parameter History

The CI replay records when predictions begin to degrade but does not retain every intermediate gradient or support loss. We therefore conducted one fixed follow-up intervention using the same trained slow states and a declared early prefix. It is a mechanism study on the already observed audit queue.

The intervention crosses two choices. CARRY preserves the fast parameters across feedback batches; LATEST restores them to the bound $\rho$ before each batch. The second factor either applies the original SGD increment or multiplies its actual FP32 parameter diference by 0.1. We denote these settings by 1 and TENTH. The latter is defined on the realized parameter diference, rather than assumed to be bitwise identical to another learning-rate implementation. Memory, arrived feedback, and private random streams retain their prescribed lifetimes. RESET always reads the bound slow initialization.

The prefix includes all 83 NumPy jobs and 653 pandas jobs, totaling 736 predictions per seed. By its cutof, 422 pandas labels have arrived in 128 batches; 231 remain pending. The pandas prefix intersects 13 commits, of which six are complete. We restrict the original full-queue job weights to this prefix and renormalize within each repository. This produces a prefix measure, not a new full-group test set. Saved HEDGE4 predictions are reused on the same jobs. The repeated CARRY\_1 and RESET probabilities exactly match the original saved probabilities for all seeds.

<table><tr><td>Update path</td><td>2811</td><td>2812</td><td>2813</td><td>Mean</td><td>Seed SD</td></tr><tr><td>CARRY_1</td><td>0.3577</td><td>0.1925</td><td>0.3400</td><td>0.2967</td><td>0.0907</td></tr><tr><td>CARRY_TENTH</td><td>0.2847</td><td>0.3384</td><td>0.3382</td><td>0.3204</td><td>0.0310</td></tr><tr><td>LATEST_1</td><td>0.2666</td><td>0.2488</td><td>0.3115</td><td>0.2756</td><td>0.0323</td></tr><tr><td>LATEST_TENTH</td><td>0.2223</td><td>0.1748</td><td>0.2713</td><td>0.2228</td><td>0.0482</td></tr><tr><td>RESET</td><td>0.1775</td><td>0.2214</td><td>0.2302</td><td>0.2097</td><td>0.0282</td></tr><tr><td>HEDGE4</td><td>0.1679</td><td>0.1679</td><td>0.1679</td><td>0.1679</td><td>0.0000</td></tr></table>

Table 10: Fixed-prefix diagnostic, half-Brier loss. The primary comparison is CARRY\_1 minus CARRY\_TENTH. All paths use the same prefix. Their losses use a diferent evaluation population from Table 7.

Reducing the carried update has a mean gain of −0.0237, with per-seed gains 0.0730, −0.1459, and 0.0017. It fails the fixed rule of a mean gain of at least 0.01 with all seeds positive. At the smaller increment, removing accumulated parameter history improves loss by 0.0976 on average and for every seed. At the original increment, the history-removal efect changes sign across seeds. All four active paths have higher mean loss than RESET, and each is worse than HEDGE4. LATEST\_TENTH has the lowest mean loss among the four active paths, followed by LATEST\_1, CARRY\_1, and CARRY\_TENTH.

## 7.1 What happens on the first feedback batch?

The first pandas feedback batch contains eight jobs labeled other. Three more other labels arrive in the next batch. At 01:16:06 UTC, after those two updates, all three original-update seeds have a local retained-state gain below −0.01 on the same 37 newly starting jobs. Those jobs later resolve to 24 successes and 13 cancellations. This is a descriptive local marker; the first cumulative crossing occurs after 2, 421, and 25 feedback batches for the three seeds, respectively.

<table><tr><td>Seed</td><td>Before update</td><td>Original increment</td><td>Tenth increment</td><td>Original increment norm</td></tr><tr><td>2811</td><td>0.513218</td><td>3.007320</td><td>0.102798</td><td>1.590172</td></tr><tr><td>2812</td><td>2.503274</td><td>2.838060</td><td>2.805870</td><td>7.709818</td></tr><tr><td>2813</td><td>0.721480</td><td>1.350831</td><td>0.139986</td><td>2.632058</td></tr></table>

Table 11: Cross-entropy on the same first pandas feedback batch, before and after the numerical update. Increment norm is the Euclidean norm across the two factor tensors.

The original update increases same-batch cross-entropy for all three seeds (Table 11). At that point, the feedback itself is not being fit better. The observation identifies local non-descent under the implemented update, which is distinct from a support-loss decrease followed by poor generalization. The smaller increment reduces same-batch loss for two seeds and increases it for the third. The pre- and post-update losses use the same saved prompts, candidate labels, masks, positions, and per-path batch-mean normalization. The pre-update gradient traverses the mixer with non-reentrant activation checkpointing; the post-update measurement uses a no-gradient forward. The substrate stays in evaluation mode, with BF16 computation and FP32 factors. These are matched loss definitions, not a guarantee of identical numerical kernel paths. Step size, factor geometry, model curvature, and finite-precision arithmetic have not been independently separated by this intervention.

Across the full diagnostic prefix, the smaller CARRY path has lower average support loss before and after updates, yet worse mean future half-Brier loss. The selected update batch and the later prediction population thus answer diferent questions. Feedback delay and class composition ofer plausible mechanisms for this divergence, but their individual causal contributions require diferent interventions. A support-loss decrease alone is insuficient evidence that an online update is useful.

## Online updates: retained-state harm, local fitting, and later loss

Each panel uses its stated population and loss. The diagnostic prefix is drawn from the observed replay.

(a) Full replay: 3,321 jobs  
![](images/880a59e38da2181ff87441dfb74be8b14e1a8bb32db8070ddcca99007afd1ea7.jpg)

(b) First feedback batch  
![](images/8cb55c7bc296953934d7df2b3cc91ec7002c11328e7d3eb87de3156659a13363.jpg)

(c) Prefix: 736 jobs  
![](images/3bab27f848d227544485b55951ec0369c1adc77ac2a7a390034c9b89a052ec07.jpg)  
Figure 6: Three views of online learning. (a) The full 3,321-job replay compares real-memory paths at reset, retained real-feedback state, permuted-feedback state, and HEDGE4. (b) Same-batch CE change on the first eight pandas feedback jobs: positive values indicate non-descent. (c) The fixed 736-job diagnostic prefix: original carried-update loss minus the smaller carried-update loss, separately for each seed. The latter two panels describe a diagnostic on the observed queue; their populations and losses are distinct from panel (a).

Read together, the panels in Figure 6 separate three stages of an update’s efect. The full replay measures the consequence of retaining a learning trajectory. The first-batch measurement asks whether the update improves the batch that generated its gradient. The prefix intervention changes the realized step magnitude while preserving the comparison population. The original step fails locally at the first batch; shrinking it changes that local behavior without producing a consistent future-risk improvement across seeds. This is the specific failure pattern a subsequent update controller has to address.

## 8 Implementation, Cost, and Reproducibility

The episodic run uses PyTorch 2.11.0, Transformers 5.12.1, CUDA 12.8, and one RTX 4090. The substrate is BF16 and LARC factors are FP32. A physical batch has eight sequences of length 512, including padding. Prompt and answer limits are 384 and 32 tokens, respectively, and programs are scored with their complete end-of-sequence token. The mean-token scoring convention is used in both studies.
<table><tr><td>Study</td><td>Slow updates</td><td>Entry wall time (s)</td><td>Saved learning state</td></tr><tr><td>Episodic training and evaluation</td><td>1,536</td><td>1,634.503</td><td>Six slow initializations, version 256</td></tr><tr><td>CI task fit and complete replay</td><td>2,010</td><td>13,229.793</td><td>Three slow initializations, version 926; six online states</td></tr><tr><td>CI prefix intervention</td><td>0</td><td>1,058.808</td><td>Six sessions containing all four active fast states</td></tr></table>

Table 12: Recorded execution costs. Wall times include model loading, evaluation, CPU work inside the entry, saving, and cleanup. The CI total includes two stopped preflights, an interrupted partial replay, and its completed evaluation. They are not pure GPU kernel times.

The episodic preflight also performed two temporary slow updates that were discarded before production. Its total operation ledger contains 11,581 mixer forwards, 6,195 head forwards, and 5,386 backward calls, including preflight, recomputation, and evaluation. The original ledger names its physical and efective position totals physical\_tokens and effective\_tokens: 49,025,792 and 19,521,408. They add positions charged by diferent operations. Table 13 accounts for the physical total exactly. The efective total uses non-padding positions under the same operation accounting. These totals are neither mixer-input tokens alone nor FLOP-equivalent work or independent training examples.

<table><tr><td>Operation</td><td>Calls</td><td>Positions per call</td><td>Ledger contribution</td></tr><tr><td>Mixer, including recomputation</td><td>11,581</td><td> $8 \times 5 1 2$ </td><td>47,435,776</td></tr><tr><td>Selected-answer head</td><td>6,195</td><td> $8 \times 3 2$ </td><td>1,585,920</td></tr><tr><td>Native reference</td><td>1</td><td> $8 \times 5 1 2$ </td><td>4,096</td></tr><tr><td>Total</td><td></td><td></td><td>49,025,792</td></tr></table>

Table 13: Decomposition of the episodic physical-position ledger. Head and native-reference charges explain the 1,590,016 positions beyond the mixer subtotal. Backward calls are counted separately.

Data construction costs and source-review costs are recorded separately from the execution entries.

The CI run encountered a pending-feedback capacity stop after task-fit training. The completed replay reused the saved slow models and restarted the audit from its beginning, counting the repeated evaluation in full. Pending records were compacted to store the two distinct prompt strings, plain and remembered, once per job. The completed configuration allows 2 MiB of pending feedback and 4 MiB of total logical live state per session, including factors, private random streams, memory, and pending metadata. No replacement training was performed to reconstruct their identities. Audit sessions were saved and restored after at least min(128, max(1, $\lfloor N _ { \mathrm { r e p o } } / 2 \rfloor ) $ ) audit labels had arrived. Restoration checked the complete pending queue, numerical states, memory, counters, and random states; final sessions were also reloaded. The prefix intervention performed 2,472 fast-state updates and zero slow updates. Its same-batch post-update measurements are included in its recorded cost.

The reproducibility package records the slow substrate, semantic adapter, input-residual placement, trained factors, optimizer state, and random-number state separately. Source manifests and checkpoint identities disambiguate the several historical residual implementations. The manuscript’s numerical tables are generated from saved result artifacts. The paper and source archive are accompanied by an evidence map; access to the frozen base model and the original semantic adapter is required to execute the trained residuals.

## 8.1 The state boundary in an implementation

A minimal implementation needs three operations. Begin copies the factors from the requested slow version and initializes the episode’s private update state. Update consumes only feedback that has arrived, computes its declared loss through the frozen reader, and replaces that episode’s fast factors. Read evaluates the current prompt and candidates using the designated factors. Keeping these operations distinct makes the update lifetime visible in the code and in the resulting checkpoint.

The inner optimizer here is functional SGD, so there are no momentum tensors to carry between steps. Factors, step count, and the private random stream still belong to the episode. The outer AdamW optimizer owns diferent state attached to �. Slow versions prevent an episode from silently resetting to an initialization that changed after it began. This is especially useful when many independent episodes share a physical batch.

At an input boundary, cached embeddings are reusable only while their tokenization and frozen substrate match. Once LARC has changed the representation, the downstream activations depend on Φ. A cached mixer output or key–value state from another factor version would evaluate a diferent function from Equation (1). The implementation therefore recomputes the mixer with the current factors, preserves attention masks and positions, and uses activation checkpointing to trade computation for memory during backpropagation.

In the delayed-feedback setting, the pending record is also part of the learning state. It retains the prompt used for the original prediction and the information needed to score the eventual label against that prediction. Rebuilding the prompt from current memory at completion would train on a diferent observation. Saving only the factor matrices would likewise leave restoration unable to determine which feedback belongs to which earlier read.

## 9 Discussion

The program-selection study shows that feedback written into this trained rank-4 input residual reduces expected query execution error with the query read otherwise fixed. The target is already among four supplied programs, and the direct support-loss rule remains much more accurate. Within that setting, the retained factors provide a measurable path from feedback to a better probability distribution. Comparing final errors separately from gains over reset also clarifies why both objectives support adaptation, while the fixed worst-seed stability rule selects static training despite its higher mean endpoint error.

The continuous-integration study changes the learning regime from two updates on a small controlled support set to accumulated updates with delayed labels. In this repository-local retrospective queue, the repository-balanced result favors the trained reset reference over retained online state, and HEDGE4 remains stronger than the full neural path. The follow-up intervention records both same-batch non-descent and a lack of consistent future benefit from shrinking updates. It makes update size and parameter history concrete design questions, while leaving the causes of failure unresolved. Rank alone controls neither local descent nor usefulness on later predictions.

The available evidence covers one frozen language-model substrate, one rank, finite candidate decisions, and three training seeds. Comparisons with untrained initialization on the same task groups, alternative residual ranks and insertion boundaries, and matched weight-space LoRA or nonlinear adapters remain unrun. The results characterize the evaluated LARC realization and do not measure a performance advantage over those alternatives. The program results are development evidence; the CI intervention reuses an observed audit population.

A learning-state specification must describe both the current prediction and the next update. Resuming factor-space learning requires the factors and any optimizer state; the product �� describes the current linear correction alone. A trained rese reference measures the efect of retaining updates, and a strong direct-feedback comparator measures how useful the resulting policy is. Within MMLA, these findings motivate evaluating a controller that uses arrived-batch fitting and later outcomes to decide whether to retain an update. Such a controller remains a next study, with independent task groups and the existing comparators needed to assess it.

## 10 Relation to Existing Methods

Residual learning and compact adaptation. Residual networks learn a correction relative to a reference path [8]. ReZero initializes residual branches through a zero-valued scalar gate [2]. Adapter modules train compact additions to a frozen pretrained network [9]. LoRA parameterizes weight corrections with low-rank factors [10]; Section 3 gives the linear relationship to an input-side residual and identifies the insertion boundary used here.

Fast weights and learning an initialization. Fast weights provide an intermediate timescale between neural activities and slowly trained parameters [1]. MAML trains initial parameters for rapid task adaptation [6], and its first-order approximation omits support-loss Hessians [12]. Equation (5) applies that estimator to the input-residual factors, with ordinary query-risk training as the matched static objective.

Block et al. [4] study retraining objectives that prepare model weights for later low-rank adaptation and give performance guarantees in a linear setting. ABMLL [16] uses low-rank forms to model global and task-specific parameter distributions for Bayesian adaptation across datasets. Here the pretrained substrate is fixed, the learned initialization is a deterministic pair of input-residual factors, and the experiments measure feedback-retention efects with real/sham and keep/reset interventions. These diferences specify the objects trained and evaluated in the respective studies.

Learning during inference. Test-time training adapts model parameters through a self-supervised objective at test time [15] TTT layers instead make the sequence state a model that is updated by learning [14]. Titans develops neural memory that learns historical context [3]. Our experiments use external execution feedback or completed-job labels, with state resets that isolate their efect. Within MMLA, LARC occupies the numerical policy plane; explicit stored records retain a separate memory lifetime.

## 11 Conclusion

LARC realizes a numerical policy state with the low-rank map $h + B A h$ , a trained starting point, private feedback updates, and defined reset and resume operations. In the four-candidate development task, two gradient steps reduce expected query execution error relative to each trained reset, while a direct support-loss rule remains much more accurate. In the CI replay, retained updates increase later loss, and the diagnostic also records local non-descent. These measurements connect the residual’s factor-space dynamics to the practical question of when a small numerical learning state should be retained.

## References

[1] Jimmy Ba, Geofrey Hinton, Volodymyr Mnih, Joel Z. Leibo, and Catalin Ionescu. Using Fast Weights to Attend to the Recent Past. arXiv:1610.06258, 2016.

[2] Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Garrison W. Cottrell, and Julian McAuley. ReZero is All You Need: Fast Convergence at Large Depth. arXiv:2003.04887, 2020.

[3] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to Memorize at Test Time. arXiv:2501.00663v1, 2024. Version 1, first submitted December 31, 2024.

[4] Jacob L. Block, Sundararajan Srinivasan, Liam Collins, Aryan Mokhtari, and Sanjay Shakkottai. Provable Meta-Learning with Low-Rank Adaptations. arXiv:2410.22264v2, 2025. Version 2, October 22, 2025.

[5] Glenn W. Brier. Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review, 78(1):1–3, 1950. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

[6] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. arXiv:1703.03400, 2017.

[7] Yoav Freund and Robert E. Schapire. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. Journal ofComputer and System Sciences, 55(1):119–139, 1997. https://doi.org/10.1006/jcss.1997.1504.

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015.

[9] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-Eficient Transfer Learning for NLP. arXiv:1902.00751, 2019.

[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021.

[11] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv:1711.05101v3, 2019. Version 3, January 4, 2019.

[12] Alex Nichol, Joshua Achiam, and John Schulman. On First-Order Meta-Learning Algorithms. arXiv:1803.02999, 2018

[13] NVIDIA. Layer-wise Adaptive Rate Control (LARC). OpenSeq2Seq oficial optimizer documentation. https: //nvidia.github.io/OpenSeq2Seq/html/optimizers.html; accessed September 30, 2026.

[14] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620v4, 2025. Version 4, August 31, 2025.

[15] Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. arXiv:1909.13231v3, 2020. Version 3, July 1, 2020.

[16] Liyi Zhang, Jake Snell, and Thomas L. Grifiths. Meta-Learning at Scale for Large Language Models via Low-Rank Amortized Bayesian Meta-Learning. arXiv:2508.14285v3, 2026. Version 3, April 1, 2026.

[17] Junyi Zou and Avrova Donz. MMLA: Memory-Mediated Learning Architecture for Predictive Dual-State Adaptation. arXiv:2606.28876v4, 2026. Version 4, September 14, 2026.

## A Derivation of Factor-Space Dynamics

For a diferentiable scalar loss, $d \ell = \langle G _ { C } , d C \rangle _ { F }$ and $d C = ( d B ) A + B ( d A )$ . Cyclic permutation in the trace gives Equation (7). A simultaneous gradient step at any $( A , B )$ yields

$$
\begin{array} { r } { A _ { + } = A - \eta B ^ { \top } G _ { C } , \quad B _ { + } = B - \eta G _ { C } A ^ { \top } , } \end{array}\tag{20}
$$

$$
B _ { + } A _ { + } = B A - \eta B B ^ { \top } G _ { C } - \eta G _ { C } A ^ { \top } A + \eta ^ { 2 } G _ { C } A ^ { \top } B ^ { \top } G _ { C } .\tag{21}
$$

Substituting $( c A , B / c )$ proves Equation (10). The efective matrix step has two state-dependent preconditioning terms. In particular, two states with identical �� need not have identical subsequent products. Optimizer moments introduce further coordinate dependence.

For the first-step expectation, write $\begin{array} { r } { ( A _ { 0 } ^ { \top } A _ { 0 } ) _ { i j } = \sum _ { q = 1 } ^ { r } A _ { q i } A _ { q j } } \end{array}$ . Independence and zero means make of-diagonal expectations zero, while each diagonal expectation is $r \sigma ^ { 2 }$ . Because the initial residual is exactly zero, the downstream matrix gradient is independent of the sampled factors whenever the loss depends on them only through �. Factor regularizers or factor-dependent stochastic operations would require a separate treatment.

Equation (11) follows from �-smoothness: $\begin{array} { r } { \ell ( \Phi + u ) \le \ell ( \Phi ) + \langle \nabla \ell ( \Phi ) , u \rangle + \frac { L } { 2 } \| u \| ^ { 2 } } \end{array}$ , with $\boldsymbol { u } = - \eta \nabla \ell ( \Phi )$ . The observed non-descent in Table 11 concerns the actual finite step and numerical implementation; it does not contradict this conditional bound.

## B Episodic Learning Procedure

1. Initialize paired slow factors with the same nonzero Gaussian � and zero � for each training seed. Fix the substrate, data split, candidate order, losses, optimizers, and update counts.

2. For each outer batch, bind the current version of $\rho$ and create two private fast states. Obtain the same support feedback and query targets for both training objectives.

3. Evaluate all four candidate programs and perform two support-risk SGD updates per fast state. Preserve episode isolation across sequence rows and random-number streams.

4. Evaluate the static query loss at $\rho$ or the adapted query loss at Equation (5). Aggregate the two episodes, apply one clipped AdamW update to $\rho ,$ and increment its version.

5. Save actual factors and optimizer/random state at declared boundaries. Complete all six trajectories before producing development predictions.

6. Evaluate real/sham feedback crossed with keep/reset. Save predictions before opening development query targets. Compute group-weighted risks and paired group intervals, then apply the fixed selection rule.

## C Task and Protocol Details

## C.1 Finite parameter support

Arithmetic scales range over $\{ - 7 , \ldots , - 1 , 1 , \ldots , 7 \}$ and ofsets over $\{ - 1 2 , \ldots , 1 2 \}$ . Clipping bounds are fixed at [−16, 16]. List minima range over $\{ - 6 , \ldots , 6 \}$ and limits over $\{ 1 , \ldots , 6 \}$ . A sufix is assigned by a fixed group-specific random seed, but it does not define a new independent active-parameter group. Arithmetic inputs are integers in [−48, 48]; list inputs have six entries in [−12, 12]. Support and query inputs are sampled without duplicate inputs within an episode and with disjoint support/query sets.

Table 14 gives the complete ordered candidate set. Operations are applied from left to right. For arithmetic, mul multiplies by scale, add adds ofset, and clip clamps to the fixed lower and upper bounds. For lists, filter\_min preserves entries greater than or equal to minimum in their original order, sort sorts ascending, take retains the first limit entries, and append adds sufix as one final entry. Each parameter group contributes four episodes, one for each possible target recipe; candidate order stays fixed. Support and query outputs are generated by that target program. The target index is withheld from the learner.

<table><tr><td>Index</td><td>Arithmetic recipe</td><td>List recipe</td></tr><tr><td>0</td><td>mul, add</td><td>filter_min, take</td></tr><tr><td>1</td><td>add, mul</td><td>take, filter_min</td></tr><tr><td>2</td><td>clip, mul, add</td><td>filter_min, sort, take, append</td></tr><tr><td>3</td><td>mul, clip, add</td><td>append, filter_min, sort, take</td></tr></table>

Table 14: All eight fixed recipes. A family’s four recipes instantiate the four candidate programs using its public parameter values.

The historical exclusion operates on the efective parameter tuple, rather than a newly assigned episode identifier. This prevents relabeling an old transformation as a new task. Model-pretraining overlap is unknown.

## C.2 What reset removes

In the episodic intervention, reset is applied after the second inner update and before the deterministic query read. It restore the episode’s trained $\rho ,$ clears inner optimizer and cached fast-dependent representations, and restores the private episode random state. Real/reset and sham/reset therefore have equal predictions. The trained $B _ { \rho }$ is retained.

In CI, $P _ { 0 } M _ { 1 }$ reads the task-trained initialization with the same causal case memory as $P _ { 1 } M _ { 1 }$ . The LATEST diagnostic resets only fast parameter history before each feedback batch; it preserves memory and the prescribed random stream. Its post-update state is then read until the next batch. This difers from the always-reset control and from resetting explicit memory.

## C.3 CI observations, labels, and baseline parameters

The prediction interface contains exactly seven fields: repository name, workflow identifier, workflow path, event type, job name, commit identifier, and recorded start time. The prompt asks for the eventual status and serializes these fields with the selected completed-history cases. Each case exposes its workflow identifier, job name, commit identifier, and arrived result. Current-job completion time and conclusion are absent. Four literal answer strings, success, failure, cancelled, and other, are scored with their end-of-sequence token using the mean-log-probability policy.

The source conclusions success, failure, and cancelled map to labels 0, 1, and 2. Conclusions timed\_out, action\_required, neutral, skipped, and stale map to label 3, other. Noncompleted jobs, unrecognized conclusions, and invalid or missing timestamps are excluded. Commit groups shared across training and audit or exposed in the earlier queue are excluded as groups. The public input is the same for the neural predictors and HEDGE4. The neural path scores label strings, whereas HEDGE4 directly predicts the four class probabilities.

HEDGE4 uses 4,096 keys per bounded statistics map, 1,024 hashed unigram/bigram features, logistic learning rate 0.1, and logistic weight decay $\bar { 1 0 ^ { - 4 } }$ . Mixture weights are proportional to $\exp ( - L _ { j } )$ , where $L _ { j }$ is the expert’s accumulated prediction-time half-Brier loss observed through arrived feedback. A workflow prior uses Laplace smoothing. The latest workflow/job label receives probability 0.7 and each alternative 0.1; the exact commit/workflow/job label receives 0.97 and each alternative 0.01. Missing keys fall back to the prior. Updates sharing a completion timestamp are processed as a batch.

The calibrated neural control fits four fixed additive label biases by weighted historical cross-entropy with an $\ell _ { 2 }$ coeficient of $1 0 ^ { - 4 }$ . It uses CPU float64 L-BFGS, at most 100 iterations and 125 function evaluations, then subtracts the mean bias. Fast updates for that control use the calibrated energies; the bias persists across fast-state replacements. This control tests a fitted class bias together with its prescribed adaptation path.

## C.4 Data and artifact access

CI observations come from public GitHub Actions run and job metadata for numpy/numpy and pandas-dev/pandas. The study uses recorded job outcomes and timing, without modifying either repository or executing their workflows. Repository– commit groups are the task units. The source archive includes all table fragments, figures, and artifact-identities.json, which records source revisions, checkpoint versions, and full SHA-256 identities. Table 15 distinguishes the downloadable upstream model from the project artifacts. The program-selection initialization package contains the three selected static $\rho _ { 2 5 6 }$ tensors and the original $\omega ,$ with loader code, lineage, environment, evaluation summary, and an AGPL-3.0 license. The package is assembled locally and is not yet publicly hosted; no Hugging Face repository is available for it at the time of this revision. The later $\rho 9 2 6$ models are distinct original checkpoints. Scientific source and full continuation states remain in the project’s private archive; readers can contact the authors at the addresses on the first page for access. Earlier code and records use the name LRARC for this residual. The identities support exact matching, while public end-to-end rerunning also requires access to those artifacts.

<table><tr><td>Artifact</td><td>Exact identity in accompanying manifest</td><td>Access at report preparation</td></tr><tr><td>MiniCPM5-1B-SFT</td><td>Revision a60b37f1. . . ; base file and tensor hashes</td><td>Public upstream revision</td></tr><tr><td>Frozen semantic ω</td><td>Original and exported file hashes; tensor identity</td><td>Local initialization package; not publicly hosted</td></tr><tr><td>Three static  $\rho _ { 2 5 6 }$ </td><td>Seed-specific original/export hashes</td><td>Local initialization package; not publicly hosted</td></tr><tr><td>Three adapted  $\rho _ { 2 5 6 }$ </td><td>Original checkpoint hashes and versions</td><td>Project archive; author access</td></tr><tr><td>Three CI ρ926</td><td>Original checkpoint hashes; parent lineage</td><td>Project archive; author access</td></tr><tr><td>Scientific implementation</td><td>Program-selection source revision 7a8e7b77.. . ; CI source revision f8a8ed95. . .</td><td>Private project repository; author access</td></tr><tr><td>Paper and evidence summaries</td><td>Self-contained LaTeX, generated tables, identities</td><td>Accompanying source archive</td></tr></table>

Table 15: Artifact availability and identity. An identity record does not imply that a private artifact is publicly hosted. The frozen upstream model retains its own distribution terms.