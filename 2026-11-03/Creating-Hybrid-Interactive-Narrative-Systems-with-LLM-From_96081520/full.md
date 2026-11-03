# Creating Hybrid Interactive Narrative Systems with LLM: From Formal Description to Evaluation

Jules Clerc<sup>1,2</sup> , Domitile Lourdeaux<sup>1</sup> , Mohamed Sallak<sup>1</sup> , Marc Ravaine<sup>2</sup>

<sup>1</sup>Université de technologie de Compiègne, CNRS, Alliance Sorbonne Université,

Heudiasyc, Compiègne, France

<sup>2</sup>ALTEN, Sophia Antipolis, France

{jules.clerc,domitile.lourdeaux,mohamed.sallak}@hds.utc.fr

Abstract. Interactive narratives let users shape a story while an Experience Manager (EM) keeps it coherent, balancing player freedom against authorial control. Large Language Models now enable a new generation of hybrid EMs, which combine run-time generation from free-text input with symbolic narrative control. However, their heterogeneity makes them hard to compare: A manager’s behaviour is generally tied to the unique and specific scenario in which it operates, and much of its decision-making takes place outside of any explicit and verifiable structure. We address this with an LLM-based narrative platform whose managers are interchangeable modules operating over a shared symbolic world model of facts, operations, and goals. Because every applied operation carries explicit preconditions and efects, each session can be analysed as a plan through a common state-transition representation and evaluated, independently of how the manager works internally, with shared indicators of four properties: player agency, EM controllability, author intent, and robustness. We implement three contrasting managers, based respectively on open generation, Monte Carlo repair, and autonomous characters, and compare them through functional analysis on an interactive Little Red Riding Hood scenario. The results show that no manager dominates on every property, giving authors and EM designers a concrete basis to select a manager according to their priorities.

Keywords: Experience Management, Functional Analysis, Interactive Narrative, Large Language Models, Narrative Generation

## 1 Introduction

While traditional narratives are perceived passively, interactive narrative systems (INS) place the user at the centre of the plot, allowing them to influence the story through their choices. Preserving narrative coherence in the face of this freedom is the role of the Experience Manager (EM): a system-side agent that decides, at each stage, how far to follow the player and when to redirect the narrative. Reconciling player initiative with narrative coherence is a long-standing concern in the field [1], and every EM embodies one particular way of striking that balance.

The arrival of Large Language Models (LLMs) has been a turning point. Because such a model can interpret free-form textual input and generate plausible content as the story unfolds, the author no longer has to spell out in advance every action the player might attempt, nor every way the characters might react. This flexibility, however, comes at a price: when narrative decisions are made within the generative model, the EM can no longer easily anticipate them, keep them within the limits defined by the author, or put them back into the narrative space. A recent line of research attempts to reconcile flexibility with control through hybrid designs, in which the manager itself combines the fluidity of generation with the logic of an explicit structure [2–4], for example, by letting a language model guide a planner, or by integrating the generated content into an inspectable game state.

The wide variety of these hybrid designs, however, makes comparison dificult. Since an EM is generally presented in a unique and personalised scenario, its qualities are inseparable from those of the scenario it runs on [5]: the field still lacks shared benchmarks, a situation known as the "No Yardstick" problem [6]. Eforts to structure the domain have primarily focused on architecture, unifying how managers are described, how they are decomposed, and how they interact with their environments [7–9]. They leave open the complementary question that we address here: how to judge, on common ground, the narratives that diferent EMs actually produce?

To investigate this question, we consider three contrasting forms of hybrid experience management, evaluated separately: open generation, targeted repair, and character-driven progression. Their diferent strategies lead us to three hypotheses. H1, the Monte Carlo Tree Search (MCTS) repair improves robustness: We expect the MCTS-based manager to be more robust than the open manager, since it behaves identically during ordinary interaction but projects a corrective sequence when the narrative reaches a dead end. H2, the multi-agent manager increases emergent progression: We expect the multi-agent manager to produce more emergent and less authored narratives, as autonomous NPCs generate and execute events independently of the player. H3, no architecture dominates: We expect no manager to perform best on all four properties, since each architecture prioritises a diferent balance of player agency, EM controllability, author intent, and robustness.

Our solution involves decoupling the internal workings of an EM from how its behaviour is measured. We build a narrative platform whose managers are interchangeable modules connected to the same symbolic world model, expressed in terms of facts, operations, and goals. Every operation that is efectively applied carries its own preconditions and efects, so a session played under any manager can be replayed as a path through a formal state-transition representation. Based on this common foundation, the narratives are then compared using indicators of four properties (player agency, EM controllability, author intent, and robustness), covering the diferent actors in the system (author, EM, player). The paper ofers (1) a modular LLM-based platform where heterogeneous hybrid managers share a common world model; (2) three contrasting managers: one favours open generation, another adds Monte Carlo repair, and the third delegates events to autonomous characters (with the ambition of placing them at diferent points on the spectrum of agency and control); and (3) a functional analysis that brings their sessions into a single formalism and reads them with shared indicators, demonstrated on an interactive Little Red Riding Hood scenario.

## 2 Related Work

Experience management addresses a central problem in interactive narrative: the system must react to player actions while preserving narrative progression [1]. Planning-based systems represent actions, their conditions, their efects, and narrative objectives, allowing control knowledge to be expressed as constraints over possible state trajectories [10]. Search-based drama management [11] evaluates alternative interventions according to their expected narrative value, while plan-based mediation attempts to preserve the coherence of an executing narrative plan when the player performs an unanticipated action [12]. Control can also be distributed among autonomous characters: character-centred systems derive narrative behaviour from the goals and decisions of individual agents [13], whereas multi-agent story-graph pruning constrains NPC actions according to their long-term consequences while leaving player actions available [14]. MCTS provides another way to explore large narrative spaces by using believability estimates to guide the search towards stories satisfying user-defined goals [15]. These approaches difer in their decision mechanisms, but all rely on an explicit representation on which narrative control can operate.

LLMs challenge this assumption by allowing players and characters to act beyond a fully predefined set of dialogues or behaviours. Recent systems attempt to retain narrative direction through complementary mechanisms: director agents coordinate actor agents and reschedule the plot after player intervention [2]; plot-based reflection aligns generated reactions with player intentions [16]; and hierarchical memory supports coherence and character behaviour over longer interactions [17]. These systems extend the range of interactions, but they also move part of narrative decision-making outside the explicit structures traditionally used by an EM.

![](images/128dc49963871e5a7a60c899ca853b4f1c6ebc6edbf156d073bc4f9d367a5bab.jpg)  
Figure 1. Overall architecture of the platform

Hybrid approaches reconnect these generative capabilities with symbolic or authored control mechanisms. A language model can guide a classical narrative-planning algorithm as a search heuristic [3], while behaviour-tree systems [4] can use generated dialogue and descriptions without relinquishing authored goals or explicit game-state updates. Our platform follows this direction but applies it to several EM strategies rather than to a single architecture. LLM components are used to interpret player commands, generate operations that modify the world state, propose events during narrative projection, and drive autonomous NPC decisions. At the same time, every operation efectively applied to the story is represented through explicit preconditions and efects. This shared symbolic layer allows for the introduction of openness to diferent techniques for managing interactive narrative without abandoning an inspectable description of world evolution.

The resulting architectures are nevertheless dificult to compare, and existing work has mainly addressed this fragmentation from an architectural or integration perspective. The Generalized Experience Management framework [7] provides a common representation of experience-management processes, while structured analysis [8] identifies comparable components across managers with diferent designs. EM-Glue [9] addresses software interoperability by decoupling managers from their execution environments through an intermediate platform and communication protocol. These contributions help describe, decompose, or connect heterogeneous systems, but not evaluate the narrative trajectories these systems produce on common functional criteria, which is the gap we address.

## 3 Narrative Platform Design

Our platform has three layers (Fig. 1): a world model of facts, operations, and goals; an execution loop in which the player issues natural-language commands and a swappable EM selects the operations applied; and an evaluation layer that maps sessions onto a formal model (Section 5).

## 3.1 Symbolic world model

We represent the narrative world with symbolic facts, in a way close to ontologies. A narrative fact is a subject-predicate-object triple:

$$
\tau = ( u , p , o ) , \qquad u \in \mathcal { U } , \quad p \in \mathcal { P } , \quad o \in \mathcal { U } \cup \mathcal { L }\tag{1}
$$

where U is the set of entities, $\mathcal { P }$ the set of predicates and L the set of literal values. The set of all possible facts is $\mathcal { F } \subseteq \mathcal { U } \times \mathcal { P } \times ( \mathcal { U } \cup \mathcal { L } )$ . A world state is a snapshot of the facts held true at a given time, $v \subseteq { \mathcal { F } }$ , and we write W for the set of world states.

The world evolves through narrative operations. An operation $\eta = ( P r e _ { \eta } , A d d _ { \eta } , D e l _ { \eta } )$ , with $P r e _ { \eta } , A d d _ { \eta } , D e l _ { \eta } \subseteq \mathcal { F }$ represents a possible modification of the current world state. The preconditions $P r e _ { \eta }$ specify the facts that must be true before the operation can be performed, while $A d d _ { \eta }$ and ${ D e l } _ { \eta }$ respectively specify the facts that become true and the facts that cease to be true after its application. It is applicable in w if $P r e _ { \eta } \subseteq w$ , and its application leads to the world state:

$$
R e s u l t ( w , \eta ) = ( w \setminus D e l _ { \eta } ) \cup A d d _ { \eta } .\tag{2}
$$

We separate the operations the player may propose, the actions A, from those the system controls, the events $\mathcal { E } ,$ and write $\mathcal { H } = \mathcal { A } \cup \mathcal { E }$ (with $\mathcal { A } \cap \mathcal { E } = \emptyset )$ for the set of operations. The operations applicable in w, together with their restrictions to actions and events are:

$$
\begin{array} { r l } & { \mathcal { H } ( w ) = \{ \eta \in \mathcal { H } \mid P r e _ { \eta } \subseteq w \} , } \\ & { \mathcal { A } ( w ) = \mathcal { H } ( w ) \cap \mathcal { A } , } \\ & { \mathcal { E } ( w ) = \mathcal { H } ( w ) \cap \mathcal { E } } \end{array}\tag{3}
$$

The operation set H is not fixed. An EM may enlarge it at run time by generating new operations, as detailed in Section 4. Finally, a narrative goal is a set of facts ${ \mathcal { G } } \subseteq { \mathcal { F } }$ , and a world state is terminal when $\mathcal { G } \subseteq w$

## 3.2 Execution loop

The interaction proceeds turn by turn. At each step, the player does not directly access the current world state $w \in W$ . Instead, the platform provides a textual observation describing what is perceptible in the narrative world. The player may then answer with a natural-language command, or remain inactive. We denote by C the set of natural-language commands and by ⊥ an inaction. The player’s input at a given turn is:

$$
\alpha \in \mathcal { C } \cup \{ \perp \}\tag{4}
$$

The input α does not directly modify the world state. It is only a proposal expressed by the player and must be resolved by the Manager. The EM selects the narrative operations efectively applied during the turn through a policy:

$$
\nu : W \times ( \mathcal { C } \cup \{ \bot \} ) \to \{ \mathcal { H } ^ { * }\tag{5}
$$

Given the current world state w and the player’s input α, the EM selects a finite sequence of operations:

$$
\Sigma = \nu ( w , \alpha )\tag{6}
$$

The selected sequence may contain an existing action matching the player’s command, generated actions or events, system events, or no operation at all. The concrete way in which this sequence is produced depends on the EM architecture and is detailed in Section 4.

If a sequence $\Sigma = ( \eta _ { 1 } , \dots , \eta _ { k } )$ is selected, its operations are applied in order to the current world state. The resulting state is defined as:

$$
w ^ { \prime } = R e s u l t ^ { * } ( w , \Sigma ) ,\tag{7}
$$

where Result<sup>∗</sup> denotes the successive application of Result to all operations in Σ. A complete execution produces a sequence of world states $( w _ { 0 } , w _ { 1 } , \ldots , w _ { n } )$ , which represents the narrative trajectory generated during the session. We write ( ) for the empty sequence, with the convention $\begin{array} { r } { R e s u l t ^ { * } ( w , ( \mathbf { \omega } ) ) = w } \end{array}$ (no operation applied, the world state is left unchanged).

## 4 Experience Manager Architectures

All EMs share two LLM-based functions that turn a command $c \in { \mathcal { C } }$ into a formal operation.   
They difer in how they use them and in the control they exert over the story.

Natural-language matching (δ) The first maps a command to an existing applicable operation. Given a command c and an operation type $k \in \{ a c t i o n , e v e n t \}$ 2

$$
\delta : \mathcal { C } \times \{ a c t i o n , e v e n t \}  \mathcal { H } \cup \{ \exists \}\tag{8}
$$

returns the operation of type k that matches c when one exists, and the symbol $\beth$ when none does.

Dynamic generation (Gen) When matching fails, an EM may create a new operation. Let $\mathcal { H } _ { a l l }$ be the set of all narrative operations that can be built over the world facts, of which the current operation set is a subset, $\mathcal { H } \subseteq \mathcal { H } _ { a l l }$ . The function

$$
G e n : W \times \mathcal { C } \times \{ a c t i o n , e v e n t \}  \mathcal { H } _ { a l l }\tag{9}
$$

returns a fresh operation $\eta _ { n e w } \in \mathcal { H } _ { a l l } \ \backslash$ H of type k, synthesised from the command c in the state w. The EM then adds it to the current set, $\displaystyle \mathcal { H } : = \mathcal { H } \cup \{ \eta _ { n e w } \}$ , which is how the narrative space is enlarged at run time.

## 4.1 Open Generation-Based EM

This manager resolves one operation per turn and maximises openness to the player. It injects an event, drawn uniformly at random from $\mathcal { E } ( w )$ , when a draw $u \sim \mathcal { U } ( [ 0 , 1 ] )$ falls below $p _ { e v t }$ or the player submits no command, provided an applicable event exists (lines 1–3); this pushes the story forward independently of the player. Otherwise the command c is processed: it is bound through the parser δ to an existing action when possible (lines 7–8), and otherwise triggers generation, adding a new action to H (lines 10–12). An empty command with no applicable event yields a wait (lines 4–5). The selected operation is finally validated against its preconditions (lines 15–18): it is executed only if $P r e _ { \eta } \subseteq w$ , otherwise the turn is rejected and w is left unchanged. Algorithm 1 details $\nu ^ { \mathrm { O P E N } }$

Because any unrecognised command is turned into a new operation, this strategy grants strong player agency at the cost of an unbounded expansion of the narrative space. The EM’s own control reduces to the injection of events and to precondition validation.

## 4.2 MCTS Projection-Based EM

This manager extends the open one with a repair mechanism that improves robustness without giving up its openness. As long as at least one operation is applicable, it behaves exactly like $\nu ^ { \mathrm { O P E N } }$ (lines 1–2). The additional mechanism is triggered only when the system reaches a dead end, defined as a state in which no operation is currently applicable:

$$
b l o c k e d ( w ) \iff \mathcal { H } ( w ) = \emptyset .\tag{10}
$$

From such a state, rather than leaving the narrative stuck, the EM runs an MCTS over event continuations to project a sequence that brings the story back towards a finishable narrative space, and injects it in the system. Algorithm 2 details the resulting policy $\nu ^ { \mathrm { M C T S } }$

From the root state $w ,$ the search builds a tree whose edges are new events. To expand a node, the EM queries an LLM for up to b natural-language suggestions of plausible next events in the node’s state, then selects one and instantiates it through Gen (lines 13–14). A node is fully expanded once its b children have been generated (line 8). These generated events live only in a working copy used by the search, and do not alter the live operation set H. They support the rollouts and the children of the node, but only the retained sequence is committed to the system at the end. The events composing a candidate continuation $\sigma = ( \eta _ { 1 } , \dots , \eta _ { k } )$ are therefore all generated $( \eta _ { i } \in \mathcal { H } _ { a l l } \ \backslash \ \mathcal { H } )$ . During traversal, children are selected by the Upper Confidence bound applied to Trees (line 9), an unvisited child being given infinite priority:

Algorithm 1 Open Generation-Based EM policy $\overline { { \nu ^ { \mathrm { O P E N } } ( w , \alpha ) } }$   
Input: world state w, player command $\overline { { \alpha \in \mathcal { C } \cup \{ \perp \} } }$ (natural language, or ⊥ for inaction)   
Parameter: event probability $p _ { e v t }$   
Output: efective operation sequence Σ   
1: $u \sim \mathcal { U } ( [ 0 , 1 ] )$   
2: if $( u < p _ { e v t }$ or $\alpha = \perp )$ and $\mathcal { E } ( w ) \neq \emptyset$ then   
3: η ← Uniform(E(w)) {event injection}   
4: else if $\alpha = \perp$ then   
5: return ( ) {no event available: wait}   
6: else   
7: if δ(α, action) $\neq \beth$ then   
8: $\eta  \delta ( \alpha ,$ action) {matched action}   
9: else   
10: η<sub>new</sub> ← Gen(w, α, action) {create a new operation}   
11: $\mathcal { H }  \mathcal { H } \cup \{ \eta _ { n e w } \}$ {add it to H}   
12: $\eta  \eta _ { n e w }$ {apply generated action}   
13: end if   
14: end if   
15: if $P r e _ { \eta } \subseteq w$ then   
16: return (η) {single-operation sequence}   
17: else   
18: return ( ) {reject, w unchanged}   
19: end if

$$
\mathrm { U C T } ( n ^ { \prime } ) = \frac { R ( n ^ { \prime } ) } { N ( n ^ { \prime } ) } + c \sqrt { \frac { \ln N ( n ) } { N ( n ^ { \prime } ) } } ,\tag{11}
$$

where $N ( \cdot )$ is a visit count, $R ( n ^ { \prime } )$ the accumulated reward of child $n ^ { \prime } ,$ n its parent, and c the exploration constant. Each node is then evaluated by a bounded number of depth-limited random rollouts (line 17), measuring whether the goal becomes reachable:

$$
r ( w _ { n } ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } { \mathrm { s o m e ~ r o l l o u t ~ r e a c h e s ~ a ~ } } w ^ { \prime } { \mathrm { ~ w i t h ~ } } \mathcal { G } \subseteq w ^ { \prime } , } \\ { 0 } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }\tag{12}
$$

After N iterations, the retained sequence $\sigma ^ { * } ( w )$ is read of greedily from the root (line 20), following the most promising children by visit count, accumulated reward, then solved status. Being all newly generated, its events are then committed to H and executed in order (lines 22–30).

This projection-based repair thus takes authority over the event channel only in blocked states: its control is stronger than that of $\nu ^ { \mathrm { O P E N } }$ , yet local and corrective, and it trades player agency in those states for finishability.

## 4.3 Multi-Agent NPC-Based EM

This manager decentralises the system side into a set of autonomous non-player characters (NPCs) $N = \{ c _ { 1 } , \ldots , c _ { m } \}$ , turning narrative progression into emergent, character-driven behaviour. A turn unfolds in two phases:

Algorithm 2 MCTS Projection-Based EM policy $\overline { { \nu ^ { \mathrm { M C T S } } ( w , \alpha ) } }$   
Input: world state w, player command $\overline { { \alpha \in { \mathcal { C } } \cup \{ \perp \} } }$   
Parameter: exploration constant c, number of iterations $N _ { ; }$ , suggestions per node $b ,$ rollouts per   
node B, rollout depth $d _ { m a x }$   
Output: efective operation sequence Σ   
1: if $\mathcal { H } ( w ) \neq \emptyset$ then   
2: return $\nu ^ { \mathrm { O P E N } } ( w , \alpha )$ {not blocked: ordinary policy}   
3: end if   
4: root ← Node(w)   
5: for $i \gets 1$ to N do   
6: n ← root   
7: $w _ { n } \gets w$ {node state, starts at the blocked root}   
8: while n has b children and n is not terminal do   
9: $n \gets \mathrm { a r g m a x } _ { n ^ { \prime } \in c h i l d r e n ( n ) } \mathrm { U C T } ( n ^ { \prime } )$   
10: $w _ { n } \gets R e s u l t ( w _ { n } , \eta _ { n } )$   
11: end while   
12: if n is not terminal then   
13: $\eta \gets \mathrm { G E N E R A T E E V E N T } ( w _ { n } )$ {LLM suggestion instantiated via Gen}   
14: n ← AddChild(n, η)   
15: $w _ { n } \gets R e s u l t ( w _ { n } , \eta )$   
16: end if   
17: $r \gets \mathrm { R o L L O U T } ( w _ { n } , B , d _ { m a x } )$   
18: Backpropagate $( n , r )$   
19: end for   
20: σ<sup>∗</sup>(w) ← BestEventSequence(root)   
21: $\Sigma  ( )$   
22: for all $\eta _ { i } \in \sigma ^ { * } ( w )$ do   
23: H $ \mathcal { H } \cup \{ \eta _ { i } \}$ {commit generated event to the live system}   
24: if $P r e _ { \eta _ { i } } \subseteq w$ then   
25: w ← Result(w, η<sub>i</sub>)   
26: $\Sigma  \Sigma \cdot ( \eta _ { i } )$   
27: else   
28: break {stop injection on first failure}   
29: end if   
30: end for   
31: return Σ

• The player phase (lines 2–5) applies the open policy with $p _ { e v t } = 0$ when the player proposes a command α $\neq \perp$ . The command is matched or generated as in the open manager, but no event is injected on a probability draw.

• The NPC phase then lets each character act once, in a random order $\rho$ drawn afresh every turn to avoid any first-mover advantage (lines 6–27).

Each character c is driven by an LLM initialised with its profile and motivation $\theta _ { c } \in \Theta _ { c } ,$ where $\Theta _ { c }$ is the space of profiles and motivations (persona, goals, traits) of character c. It acts only during the NPC phase: it triggers an applicable event, proposes a new one described by a natural-language text $t \in { \mathcal { C } }$ , or waits,

$$
\nu _ { c } : W \times \Theta _ { c } \to \{ \mathrm { P E R F O R M \_ E V E N T } ( \eta ) , \ \mathrm { P R O P O S E } ( t ) , \ \mathrm { W A I T } \} .\tag{13}
$$

Its decision $d _ { c } = \nu _ { c } ( w , \theta _ { c } )$ is then resolved into an efective operation in the current state (lines 12–22): a performed or proposed event is committed only when it is applicable in the current state $( P r e \subseteq w )$ , the generated case being added to H beforehand, otherwise the character waits and no operation is applied. The state is updated after each character acts (lines 23–26), until the narrative goal is reached (lines 8–9). Algorithm 3 details the resulting policy $\nu ^ { \mathrm { M A } }$

Algorithm 3 Multi-Agent NPC-Based EM policy $\nu ^ { \mathrm { M A } } ( w , \alpha )$   
Input: world state w, player command $\alpha \in \mathcal { C } \cup \{ \perp \}$   
Parameter: NPC set $N = \{ c _ { 1 } , \ldots , c _ { m } \}$ with profiles $\{ \theta _ { c } \}$   
Output: efective operation sequence Σ   
1: $\Sigma  ( )$   
2: if $\alpha \neq \perp$ then   
3: $\Sigma  \nu ^ { \mathrm { O P E N } } ( w , \alpha )$ {player phase: only if the player proposes something, with $p _ { e v t } = 0 \}$   
4: $w \gets R e s u l t ^ { * } ( w , \Sigma )$   
5: end if   
6: $\rho $ random permutation of N {avoid first-mover advantage}   
7: for $j  1$ to m do   
8: if $\mathcal { G } \subseteq w$ then   
9: break {goal reached}   
10: end if   
11: $c \gets \rho ( j )$ {select NPC}   
12: $d _ { c } \gets \nu _ { c } ( w , \theta _ { c } )$ {LLM agent decision}   
13: $\eta ^ { c }  \beth$ {no operation by default}   
14: if $d _ { c } =$ perform\_event(η) and $\eta \in \mathcal { E } ( w )$ then   
15: $\eta ^ { c }  \eta$ {existing applicable event}   
16: else if $d _ { c } = \mathrm { P R O P O S E } ( t )$ then   
17: $\eta  \mathrm { G e n } ( w , t ,$ event) {candidate event}   
18: if $P r e _ { \eta } \subseteq w$ then   
19: $\mathcal { H }  \mathcal { H } \cup \{ \eta \}$ {playable: commit}   
20: $\eta ^ { c }  \eta$   
21: end if   
22: end if   
23: if $\eta ^ { c } \neq \beth$ then   
24: $w  R e s u l t ( w , \eta ^ { c } )$   
25: $\Sigma  \Sigma \cdot ( \eta ^ { c } )$   
26: end if   
27: end for   
28: return Σ

By partitioning the event channel E among autonomous agents, this manager makes the narrative progress largely independently of the player, allowing a high event rate and through guarded NPC generation, emergent of-script behaviour kept consistent by the playability check. It is the most decentralised of the three: authority over events is distributed across the characters rather than held by a single policy, trading tight authorial control for emergent, character-driven narration.

## 5 Evaluation Framework

We evaluate the managers by functional analysis: each drives many sessions, recorded as plans in the state-transition representation of [18] and scored with properties defined on it.

## 5.1 Simulation protocol

Since the platform is LLM-based and the player interacts in natural language, we replace the human with an LLM<sup>1</sup> acting as the player. At each turn it reads the textual observation and either proposes a single free-text command or does nothing $( \alpha \in { \mathcal { C } } \cup \{ \bot \} )$ ). Executing a simulation turns one session into a narrative trajectory $( w _ { 0 } , \ldots , w _ { n } )$ with the operations efectively applied. A session ends when the goal is reached $( \mathcal G \subseteq w )$ , when no operation can advance the story, or when a turn budget is exhausted.

These simulations only approximate human behaviour, but they give EM designers fast feedback on their managers’ capabilities, well before any costly user study. For each manager we run $n _ { s i m } = 1 0 0$ sessions on a fixed scenario. We fixed this number by a convergence criterion, adding sessions until the running average of the properties, all valued in [0, 1], varies by less than $\varepsilon = 0 . 0 0 5$ between successive sessions. This is reached at around one hundred sessions per manager, which we round to $n _ { s i m }$

All three managers $\left( \nu ^ { \mathrm { O P E N } } , \ \nu ^ { \mathrm { M C T S } } , \ \nu ^ { \mathrm { M A } } \right)$ are evaluated on the same fixed scenario, a story inspired by Little Red Riding Hood that we reuse from earlier work on the formal modelling of INS. The player is the hunter, who must intervene at the right moments to keep the wolf from harming Little Red Riding Hood and her grandmother. Together with the wolf, these characters act as non-player characters. We chose it because it exposes the tension we study: some player choices derail the intended progression; killing the wolf too soon, for instance, blocks the events that should still occur and strands the story in a dead end, while its key stages are encoded as islands that a session is expected to cross.

## 5.2 From sessions to plans

To analyse all sessions on common ground (and subsequently compare managers with heterogeneous designs), we convert the world model design into an extended state-transition system [18] S = $( S , T , \gamma , s _ { \mathrm { i n i t } } , S _ { \mathrm { g o a l } } )$ : each world state $w _ { i }$ is read as a state $s _ { i } .$ , each operation in $\mathcal { H } = \mathcal { A } \cup \mathcal { E }$ as a transition of the same type, the Result function as the transition function γ, and the goal facts G as the goal states $S _ { \mathrm { g o a l } } = \{ s \mid \mathcal { G } \subseteq s \}$ . Mapped this way, the trajectory of one session becomes a path in S, which we call a plan π. We write $\Pi _ { m }$ for the set of plans collected for manager m.

## 5.3 Evaluated properties

Four properties, covering the player, the manager, and the author, are scored on each plan:

• Player agency captures the player’s ability to take meaningful actions and observe their impact on the story. We measure it through the acceptance rate (the share of player proposals the manager applies unchanged).

• EM controllability captures how much the manager can orient or constrain the unfolding narrative. We measure it through the event rate (the share of executed transitions that are system events) and the pre-emption rate (the share of player proposals overridden by an event).

• Author intent captures how much of the authored narrative structure is preserved. We measure it through the islands rate (the share of authored key waypoints, called islands, the plan crosses) and the scripted rate (the share of executed transitions that come from the authored design rather than being generated at run time).

• Robustness captures the system’s ability to keep the story finishable despite the player’s behaviour. We measure it through the completion rate (the share of plans reaching a goal state) and the execution rate (the share of turns whose selected operation is applied without rejection).

Each property is computed on every plan, then summarised over $\Pi _ { m }$ by its mean and standard deviation.

## 6 Results

We confront H1–H3 with the properties reported in Table 1. Dispersion is largest for author intent and robustness (standard deviations up to 0.41), since a session tends to succeed or fail as a whole, and smallest for the controllability of $\nu ^ { \mathrm { M A } }$ (0.06), whose NPCs act at every turn.

H1 — The MCTS repair improves robustness The results confirm it: robustness rises from 0.58 to 0.76 and the completion rate from 0.52 to 0.80, while author intent (0.47→0.44) barely moves. The repair only triggers in dead ends, leaving ordinary turns identical to the open manager. The injected repair events do raise controllability, though (0.28→0.40). It does, however, cost player agency (0.44→0.27): in blocked states the manager takes over the event channel, reducing the share of player-driven turns.

H2 — The multi-agent manager increases emergent progression We expected $\nu ^ { \mathrm { M A } }$ to produce more emergent, less authored narratives. This is supported: it reaches the highest controllability (0.52 against 0.28–0.40), as each NPC applies events every turn, and the lowest author intent (0.04), as of-script character behaviour drifts away from the authored islands. This value also reflects a strict calibration: both indicators only count the points of passage and transitions created by the author, so a generated event diminishes the author’s intention even if it fits the narrative. One prediction is contradicted, though: rather than curbing the player, the manager keeps the highest agency (0.68), because the player still acts every turn while NPC events are added on top instead of replacing player proposals<sup>2</sup>. Here the tension is between emergence and author intent, not between control and agency.

H3 — No architecture dominates The simulations confirm it: $\nu ^ { \mathrm { M C T S } }$ leads on robustness but has the lowest agency, $\nu ^ { \mathrm { M A } }$ leads on agency and control but collapses on author intent, and $\nu ^ { \mathrm { O P E N } }$ stays balanced, on par with $\nu ^ { \mathrm { M C T S } }$ on author intent. The three profiles are thus not mutually dominant.

Table 1. Measured properties on the Little Red Riding Hood scenario $( n _ { s i m } = 1 0 0$ sessions per manager): mean ± standard deviation.
<table><tr><td rowspan=1 colspan=1>Property</td><td rowspan=1 colspan=1> $\nu ^ { \mathrm { o P E N } }$ </td><td rowspan=1 colspan=1> $\nu ^ { \mathrm { M C T S } }$ </td><td rowspan=1 colspan=1> $\nu ^ { \mathrm { M A } }$ </td></tr><tr><td rowspan=1 colspan=1>Player agency</td><td rowspan=1 colspan=1> $0 . 4 4 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $0 . 2 7 \pm 0 . 1 4$ </td><td rowspan=1 colspan=1> ${ \bf 0 . 6 8 \pm 0 . 2 9 }$ </td></tr><tr><td rowspan=1 colspan=1>EM controllability</td><td rowspan=1 colspan=1> $0 . 2 8 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> $0 . 4 0 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> ${ \bf 0 . 5 2 \pm 0 . 0 6 }$ </td></tr><tr><td rowspan=1 colspan=1>Author intent</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 7 \pm 0 . 4 1 }$ </td><td rowspan=1 colspan=1> $0 . 4 4 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $0 . 0 4 \pm 0 . 1 6$ </td></tr><tr><td rowspan=1 colspan=1>Robustness</td><td rowspan=1 colspan=1> $0 . 5 8 \pm 0 . 4 0$ </td><td rowspan=1 colspan=1> ${ \bf 0 . 7 6 \pm 0 . 3 1 }$ </td><td rowspan=1 colspan=1> $0 . 6 3 \pm 0 . 3 9$ </td></tr></table>

These results suggest that no EM is ideal. The optimal choice depends on the priorities of the experience, whether it is maintaining player control, ensuring the story’s completion, or allowing characters to influence emerging situations. By translating these design goals into measurable indicators and properties, this framework ofers story authors and EM designers a concrete basis for selecting an EM.

Limitations and scalability The results still need to be confirmed with human players, notably to validate that the proposed indicators reflect perceived experience. Moreover, a single scenario isolates the efect of managers without establishing generalisability. Regarding scalability, the properties are computed on the simulated plans rather than on the whole state space, so the evaluation cost grows linearly with the number and length of sessions.

## 7 Conclusion

Our goal was to make LLM-based hybrid experience managers comparable despite their architectural diversity. To achieve this, we designed a narrative platform where managers are interchangeable modules sharing a single symbolic model, and we expressed each operation actually applied through explicit preconditions and efects. This allows us to represent any played session, regardless of the manager that produced it, as a path through a common formal representation of state transition, and to read it with the same properties of player agency, EM controllability, author intent, and robustness. Three managers of contrasting design (open generation, Monte Carlo repair, and autonomous characters) were instantiated on this common framework and compared by functional analysis on a Little Red Riding Hood scenario. The comparison confirmed that each manager favours diferent properties, with none prevailing on all of them, turning the choice of a manager into an explicit, priority-driven trade-of rather than a search for a single optimal design.

This study was deliberately scoped to isolate and validate the evaluation mechanism itself. Players were simulated by a language model, a choice that gives EM designers quick and repeatable feedback well before a costly user study, and managers were compared on a single controlled scenario so that observed diferences could be attributed to the managers rather than to the story. Even within this scope, the framework already makes heterogeneous, LLM-driven managers measurable and comparable on common ground. Building on these foundations, several directions extend the work. The simulated player can be complemented by human participants to confirm the indicators against perceived experience; the evaluation can be extended to richer and more varied scenarios; the properties can be deepened; and, since the managers share the same world model, they could be combined, for instance by selecting the most suitable one at each step. Expanding the catalogue of managers and properties would transform the platform into a shared testbed where heterogeneous experience management strategies would be designed, optimised and compared: an evaluation resource for the scientific community.

## References

[1] M. O. Riedl and V. Bulitko, “Interactive Narrative: An Intelligent Systems Approach,” AI Magazine, vol. 34, no. 1, pp. 67–67, 2013.

[2] S. Han, L. Chen, L.-M. Lin, Z. Xu, and K. Yu, “IBSEN: Director-actor agent collaboration for controllable and interactive drama script generation,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 1607–1619. [Online]. Available: https://aclanthology.org/2024.acl-long.88/

[3] L. Senanayake and S. G. Ware, “Language Models as Narrative Planning Heuristics,” in Proceedings of the 20th International Conference on the Foundations of Digital Games, ser. FDG ’25. New York, NY, USA: Association for Computing Machinery, May 2025, pp. 1–9.

[4] B. Ngaw, G. Jena, J. Sedoc, and A. Normoyle, “Towards Authoring Open-Ended Behaviors for Narrative Puzzle Games with Large Language Model Support,” in Proceedings of the 19th International Conference on the Foundations of Digital Games. Worcester MA USA: ACM, May 2024, pp. 1–4.

[5] B. Magerko, “A Comparative Analysis of Story Representations for Interactive Narrative Systems,” Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 3, no. 1, pp. 91–94, 2007.

[6] H. Koenitz and M. P. Eladhari, “Challenges of IDN Research and Teaching,” in Interactive Storytelling, R. E. Cardona-Rivera, A. Sullivan, and R. M. Young, Eds. Cham: Springer International Publishing, 2019, pp. 26–39.

[7] D. J. Thue, “Generalized Experience Management,” Doctoral Thesis, 2015.

[8] G. Mori, D. Thue, and S. Schifel, “A structured analysis of experience management techniques,” in Proceedings of the Fifteenth AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, ser. AIIDE’19, vol. 15. Atlanta, Georgia: AAAI Press, Oct. 2019, pp. 174–180.

[9] ——, “EM-Glue: A Platform for Decoupling Experience Managers and Environments,” Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 18, no. 1, pp. 266–274, Oct. 2022.

[10] J. Porteous, M. Cavazza, and F. Charles, “Applying Planning to Interactive Storytelling: Narrative Control Using State Constraints,” ACM TIST, vol. 1, p. 10, Nov. 2010.

[11] M. Nelson and M. Mateas, “Search-Based Drama Management in the Interactive Fiction Anchorhead,” Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 1, no. 1, pp. 99–104, Jun. 2005.

[12] R. Young, M. Riedl, M. Branly, A. Jhala, RJ. Martin, and CJ. Saretto, “An architecture for integrating plan-based behavior generation with interactive game environments,” Journal of Game Development, vol. 1, pp. 51–70, Jan. 2004.

[13] M. Si, S. C. Marsella, and D. V. Pynadath, “Thespian: Using multi-agent fitting to craft interactive drama,” in Proceedings of the Fourth International Joint Conference on Autonomous Agents and Multiagent Systems, ser. AAMAS ’05. New York, NY, USA: Association for Computing Machinery, Jul. 2005, pp. 21–28.

[14] S. G. Ware, E. T. Garcia, A. Shirvani, and R. Farrell, “Multi-Agent Narrative Experience Management as Story Graph Pruning,” Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 15, no. 1, pp. 87–93, Oct. 2019.

[15] B. Kartal, J. Koenig, and S. J. Guy, “User-driven narrative variation in large story domains using monte carlo tree search,” in Proceedings of the 2014 International Conference on Autonomous Agents and Multi-Agent Systems, ser. AAMAS ’14. Richland, SC: International Foundation for Autonomous Agents and Multiagent Systems, May 2014, pp. 69–76.

[16] H. Wu, W. Wu, T. Xu, J. Zhang, and H. Zhao, “Towards Enhanced Immersion and Agency for LLM-based Interactive Drama,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 11 166– 11 182.

[17] T. Xu, H. Wu, W. Wu, and H. Zhao, “Open-Theatre: An Open-Source Toolkit for LLM-based Interactive Drama,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, I. Habernal, P. Schulam, and J. Tiedemann, Eds. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 453–460.

[18] J. Clerc, D. Lourdeaux, M. Sallak, J. Barbier, and M. Ravaine, “Modeling Interactive Narrative Systems: A Formal Approach,” in Proceedings of the 8th International Workshop on Computational Models of Narrative (CMN 2025), ser. CEUR Workshop Proceedings, vol. 4288, 2025, pp. 179–187. [Online]. Available: https://ceur-ws.org/Vol-4288/paper14.pdf