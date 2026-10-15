# MakSafe: A Spatiotemporal Semantic Framework for Formal Specification and Compliance Checking of Road Vehicle Safety Rules

Maksym Labzhaniia, Julien Deantoni, Marie-Agnes Peraldi-Frati, and Fr\` ed´ eric Mallet´

Universite C´ ote d’Azur, Inria, CNRS, I3S, Kairosˆ

Sophia Antipolis, France

maksym.labzhaniia@etu.univ-cotedazur.fr, julien.deantoni@inria.fr, {marie-agnes.peraldi, frederic.mallet}@univ-cotedazur.fr

Abstract—Scenario-based validation is essential for automated road vehicles but cannot cover the full variability of openworld traffic. We present MakSafe, a formal spatiotemporal framework for specifying and checking road vehicle safety rules through semantic zones, action-based forecasting, and trace-tree reasoning, avoiding uniform time discretization. We formalize the road network, vehicle states, and occupancy zones, then build executable rules for front-vehicle identification and timeto-collision (TTC) verification. The evaluation combines a tracedependent case study with a scalability benchmark over traffic density (N), horizon (H), and branch factor (B). MakSafe yields branch-specific verdicts, deriving safe and unsafe outcomes from the same initial scene, and re-evaluates a scenario after a control update. A single rule check stays well below a millisecond across the reported sweep, and longer horizons slow the TTC check.

Index Terms—Autonomous vehicles, SOTIF, spatiotemporal reasoning, safety rules.

## I. INTRODUCTION

Advanced Driver Assistance Systems (ADAS) and increasingly autonomous driving functionalities require rigorous mechanisms for specifying and assessing safety. Standards such as ISO 21448 SOTIF and ISO 26262 define development and validation processes for automotive systems, yet the operational validation of driving behaviour remains difficult due to the combinatorial complexity of traffic situations.

Scenario-based validation is today the dominant industrial practice. However, a scenario only captures a finite sequence of scenes (snapshots of the environment and all vehicle states) and consequently, even large scenario libraries cannot guarantee coverage of all future traffic evolutions that may emerge from a given scene. This limitation is especially problematic for systems that must reason about future evolution and interactions among dynamic agents.

Safety rules constitute a complementary abstraction. Instead of validating isolated scenarios, safety rules specify general constraints over acceptable vehicle behaviour. Typical examples include maintaining a minimum time headway, ensuring safe lane-change distances, or preserving collision-free occupancy over a given horizon.

Many existing formal approaches rely on discretized temporal traces and evaluate safety at uniformly sampled timestamps. While effective, such approaches can incur large com-979-8-3195-3997-7/26/\$31.00 ©2026 IEEE

putational overheads and may require fine-grained sampling to capture semantically relevant transitions between sampling instants. In this paper, we present MakSafe, a spatiotemporal semantic framework for specification and compliance checking of road vehicle safety rules. MakSafe relies on semantic zones representing relevant spatial occupancy over road networks and performs safety reasoning around decision points induced by actions rather than uniformly discretizing time. The framework supports prospective safety assessment, enabling interpretable, machine-checkable verdicts for predicted future behaviour.

The main contributions of this paper are:

• A formal semantic framework for representing road networks, vehicles, actions, and spatial zones;

• A trace-tree forecasting mechanism that supports actiondependent prospective safety reasoning and branch-specific verdict generation;

• A Haskell implementation with two evaluations: (i) a laneend time-to-collision (TTC) case study with trace-specific verdicts, and (ii) a scalability benchmark.

We organize the remainder of the paper as follows. Section II summarizes the background and related work. Section III introduces the formal foundations and verification procedures of MakSafe. Section IV presents the lane-end TTC case study and trace-dependent evaluation results. Section V discusses implications and limitations. Section VI concludes.

## II. BACKGROUND AND RELATED WORK

Frameworks such as GeoScenario [1] and SceML [2], together with industrial standards such as OpenSCENARIO [3], support the structured specification of driving situations and testing workflows. Menzel et al. [4] discussed the central role of scenarios for development and validation of automated vehicles. More recently, Babikian et al. [5] proposed optimizationbased concretization mechanisms for abstract traffic-scene specifications. While these approaches are effective for test generation and simulation, they primarily focus on concrete scenario instantiation rather than semantic reasoning about general traffic-safety properties.

Several works investigated machine-interpretable formalization of traffic laws. Rizaldi and Althoff [6] formalized traffic rules for accountability of autonomous vehicles, while

Rizaldi et al. [7] introduced a formally verified monitoring framework in Isabelle/HOL. Maierhofer et al. [8] expressed interstate traffic rules using temporal logic, and Esterle et al. [9] investigated machine-interpretable representations of traffic regulations. Rulebook-oriented approaches [10], [11] define prioritized behavioural constraints for autonomousdriving decision making. LawBreaker [12] combines trafficlaw specification with fuzz testing for autonomous vehicles. These works provide important formal foundations for trafficlaw reasoning. However, many evaluate over discretized traces and check logical compliance, without modelling continuous spatial occupancy.

Formal verification of cyber-physical systems frequently relies on temporal logic and hybrid-system models. Timed and hybrid automata [13], [14] provide rigorous models combining continuous dynamics and discrete transitions, while tools such as UPPAAL [15] support model checking of timed systems. Signal Temporal Logic (STL) [16] expresses temporal properties over continuous signals and has become popular for autonomous-driving verification. MakSafe shares conceptual similarities with hybrid-system approaches since vehicle evolution combines continuous motion and discrete behavioural actions. However, unlike classical hybrid automata, MakSafe centres its reasoning on semantic spatial zones and actionoriented trace exploration rather than global state-space discretization.

Road topology representation is another important aspect of autonomous-driving verification. Lanelets [17] provide an efficient representation for structured road geometries and are common in autonomous-driving systems. MakSafe currently relies on semantic lane segments and connectivity relations. The framework is compatible with richer geometric representations and future work will investigate integration with laneletbased infrastructures and OpenDRIVE [18] road descriptions.

MakSafe complements these lines of work by combining semantic occupancy zones with action-based forecasting.

## III. MAKSAFE: A FRAMEWORK FOR SPATIOTEMPORAL SAFETY REASONING

MakSafe is a formal spatiotemporal framework for specifying and checking road vehicle safety rules. Reasoning about semantic zones and action-triggered decision points, rather than uniformly sampled time traces, yields interpretable verdicts and bounded exploration under rule-derived horizons.

## A. Static Road Network Encoding

MakSafe models the road network as a set of lanes. We represent a lane as

$$
\ell = ( \rho , s _ { 0 } , s _ { 1 } , \lambda , \delta ) ,\tag{1}
$$

where $\rho \in \mathbb { N }$ is the road identifier, $s _ { 0 } , s _ { 1 } \in \mathbb { R }$ are longitudinal start and end coordinates with $s _ { 0 } < s _ { 1 } , \lambda \in \mathbb { Z }$ is the lateral lane index, and $\delta \in \mathsf { \Gamma } \{ \mathsf { F o r w a r d }$ , Backward} is the driving direction. We denote the set of all lanes ${ \mathcal { L } } .$

A lane segment is

$$
\mathsf { s e g } = ( \ell , x _ { 1 } , x _ { 2 } ) , \quad x _ { 1 } , x _ { 2 } \in [ s _ { 0 } , s _ { 1 } ] , \ x _ { 1 } \leq x _ { 2 } .\tag{2}
$$

Directed links model connectivity:

$$
e = ( \ell _ { 1 } , \ell _ { 2 } ) ,\tag{3}
$$

where $\ell _ { 1 } , \ell _ { 2 } \in \mathcal { L }$ are the source and target lanes. We denote the set of all links E. A lane graph $G = ( \mathcal { L } , \mathcal { E } )$ is the topological layer, the substrate for zone propagation and reachability computation.

## B. Participants and Zones

We represent a vehicle as

$$
v = ( i , n a m e , L e n , u _ { \mathrm { m a x } } ) ,\tag{4}
$$

where i is the vehicle identifier, name is its name, $L e n \in \mathbb { R } _ { > 0 }$ is the longitudinal length, and $u _ { \mathrm { m a x } } \in \mathbb { R } _ { \geq 0 }$ is the maximum speed. We define a position as $p = ( \ell , x )$ where $\ell \in \mathcal L$ and $x \in [ s _ { 0 } , s _ { 1 } ]$ is a longitudinal coordinate. A dynamic vehicle state is

$$
\sigma = ( p , u , a , \delta _ { v } , A , L ) ,\tag{5}
$$

where $\boldsymbol { p } ^ { \mathrm { ~ ~ } } = { ~ } ( \ell , x )$ is the current reference position (the longitudinal centre of the vehicle) on lane $\ell , u \in [ 0 , u _ { \mathrm { m a x } } ]$ is the current speed, a is the acceleration, $\delta _ { v }$ is the movement direction, A is the set of lanes being assumed (entered but reference point not yet fully inside), and L is the set of lanes being left (exited but part of vehicle body remains). In the current implementation, prospective propagation inside one action interval, a maximal action-free span of a forecast trace, uses constant longitudinal acceleration (possibly zero), with speed saturation at physical bounds.

A zone is a finite set of lane segments:

$$
Z = \{ \mathsf { s e g } _ { 1 } , \ldots , \mathsf { s e g } _ { n } \} .\tag{6}
$$

We use fusion (i.e., the minimal union of overlapping or adjacent segments in a zone), union, and intersection operators over zones; we define overlap as non-empty intersection.

Let $Z _ { \sigma }$ collect one full-lane segment for each lane the vehicle occupies in state $\sigma { : }$ the current lane and the lanes in A and L. We define the collision zone, the vehicle body’s footprint, as a one-dimensional metric ball of radius $L e n / 2$ centred at the position $p ,$ intersected with $Z _ { \sigma }$

$$
\begin{array} { r } { \mathsf { C o l l i s i o n Z o n e } ( v , \sigma ) = \mathsf { b a l l } \left( p , \frac { L e n } { 2 } \right) \cap Z _ { \sigma } . } \end{array}\tag{7}
$$

A reachability zone is a semantic zone comprising the positions a vehicle’s shape can occupy under the assumed forecast, starting from a given state and extending over a specified time horizon. Over the horizon, we assume the vehicle preserves its acceleration profile and performs no optional manoeuvres, while forced linkage events fire as they become feasible.

$$
{ \mathsf { R e a c h Z o n e } } ( v , \sigma , t ) = { \mathsf { f u s e } } \left( \bigcup _ { 0 \leq \tau \leq t } { \mathsf { C o l l i s i o n Z o n e } } ( v , \sigma _ { \tau } ) \right) ,\tag{8}
$$

the footprint swept as the vehicle advances over the horizon, where the union also spans each $\sigma _ { \tau } ~ \in ~ f o r e c a s t ( \sigma , \tau )$ , the states the forecast admits at time τ by advancing σ along the lane graph by the travelled distance $d ( \tau )$ , turning multi-valued at links, and updating the lanes $Z _ { \sigma _ { \tau } }$ covers. In the current implementation, $d ( t )$ is acceleration-aware and piecewise: while acceleration is active,

$$
\begin{array} { r } { d ( t ) = u t + \frac { 1 } { 2 } a t ^ { 2 } , } \end{array}\tag{9}
$$

and after saturation (full stop for braking or $u _ { \mathrm { m a x } }$ for acceleration), propagation continues at the corresponding constant limiting speed.

MakSafe computes reachability zones without sampling intermediate instants, propagating the vehicle’s reference position along the lane graph’s feasible links by the distance travelled over the horizon and accounting for the vehicle length.

A scene q is a snapshot of the road network and all vehicle states at a given instant. Formally, it consists of a lane graph together with the states of all vehicles. Scenes are the atomic units for both rule evaluation and scenario construction; a scenario is a sequence of scenes over time.

## C. Action Modelling and Forecasting

Starting from a single scene, multiple future evolutions may emerge through different action choices. Accordingly, MakSafe performs prospective safety verification by generating a branching set of future traces and evaluating safety conditions independently on each forecast branch, producing trace-specific verdicts. MakSafe models vehicle behaviour through actions, which can be either forced or unforced. Forced actions correspond to linkage events, while unforced actions are optional manoeuvres. We use forced actions for lane-transition events: AssumeLane $( \ell _ { 1 } , \ell _ { 2 } )$ , LeaveLane(ℓ), ExecuteTransition $( \ell _ { 2 } )$

Applying an action α to a state σ yields a successor state $\sigma ^ { \prime } .$ . The forecast function repeatedly applies admissible actions and stores the resulting evolutions in a trace tree, where nodes represent scenes and edges represent forecasting in time. Mak-Safe explores the tree up to the rule-derived horizon, producing reachable future states and their corresponding action prefixes. Figure 1 illustrates this structure: initial scenes come from sensing or simulated environments and form the roots; forecast branches encode alternative action sequences (for example linking to a lane or another at a specific forecasted time).

At the specification level, users compose rules from these semantic primitives (zones, actions, forecast horizons, and geometric predicates) through executable Haskell combinators. In this paper, we focus on this reasoning core rather than introducing a separate external surface grammar.

## D. Illustrative MakSafe Constructs

To demonstrate how MakSafe supports the construction of safety-relevant executable definitions, we present two illustrative rules: front-vehicle identification and time-to-collision verification. Both rules operate in a prospective mode from an initial scene (provided either by sensing or a simulator), and the framework forecasts the near-future evolution of all participants before evaluating the safety condition on each reachable path. The output of the first rule is therefore a traceindexed set of front vehicles, not a static vehicle list.

![](images/c78ae343b120672130c7e04b6e1bc4c09a0d3728347e5e8c2b6981c72fb499ae.jpg)  
Fig. 1. Trace-tree structure for prospective rule evaluation. Observed scenes seed a forecast tree of scenes $q _ { i }$ with actions as edges; MakSafe evaluates each branch independently up to the horizon, yielding per-trace verdicts. Forecasting advances on actions, not uniform time, so one look-ahead step k occurs at different forecast times across branches (q , q sit at k=2, yet 2.1 and 1.2 s ahead).

1) Front-Vehicle(s) Identification: Front-vehicle(s) identification is itself a spatiotemporal and horizon-relative rule: the answer may change with time, with the action prefix the ego vehicle takes, and with the chosen horizon. As specified in Algorithm 1, for each ego trace, MakSafe identifies vehicles that are (i) ahead in the current lane or (ii) the earliest oppositedirection entrants into the ego lane before horizon expiration or ego lane change. The rule therefore reads as a tree-structured forecast: the same initial scene can yield different front-vehicle sets on different branches, with different qualifying times and even different horizon cutoffs. Concretely, the algorithm must forecast either to test whether another vehicle may enter the ego lane before the horizon expires, or to extend the ego trace through the next reachable forward lane set when no front vehicle has qualified yet. Operationally, only ego actions label trace branches: MakSafe forecasts each surrounding vehicle deterministically (with constant acceleration in the current implementation), though a forecast that crosses a link still multiplies the propagated state combinations. For each ego branch and its associated time window, the algorithm queries each surrounding vehicle to extract local predicates (current occupancy or earliest lane-entry time into the ego lane), and records the corresponding qualifying time on that same ego branch. This ego-centric design keeps the branching factor bounded even when a forecast turns multi-valued, as in the benchmark at $H { = } 6 \ \mathrm { s } { : }$ the extra states multiply zone checks within a branch, never the branch count.

Algorithm 1: Trace-Dependent Front Vehicle Identifi  
cation for an Ego Vehicle e Algorithm 2: Time-to-Collision Threshold Verification   
Input: Scene $q ,$ Ego Vehicle $e ,$ Time Horizon $\overline { { T } }$ Input: Scene $q ,$ Ego Vehicle $e ,$ TTC Threshold T   
Output: Set of trace-indexed front vehicles $\mathcal { F }$ with Output: Set of violations V with front vehicle, ego   
ego trace $\pi _ { e }$ and qualifying time $t _ { v }$ trace, and qualifying time   
${ \mathcal { F } } \gets \emptyset ;$ $\nu  \varnothing ;$   
$Q  \{ \begin{array} { r l r l } \end{array} $ {(empty trace, current lane of $e , t = 0 ) \}$ ; $\mathcal { F } $ FrontVehicleIdentification $( q , e , T ) ;$   
while $Q \neq \emptyset$ do foreach front vehicle $( v , \pi _ { e } , t _ { v } )$ in $\mathcal { F }$ do   
$( \pi _ { e } , \ell _ { e } , t _ { e } ) \gets$ dequeue from $Q ;$ $T ^ { \prime } \gets T - t _ { v } ;$   
foreach vehicle $v \neq e$ in q do $r z _ { e } ( t ) \gets$ ReachZone(e, apply $( \sigma _ { e } , \pi _ { e } ) , t ) ;$   
if v is ahead of e in lane $\ell _ { e }$ at time $t _ { e }$ then if ego and front vehicle move in the same direction   
Add $( v , \pi _ { e } , t _ { e } )$ to ${ \mathcal { F } } ;$ then   
else if v may enter lane $\ell _ { e }$ from opposite $\{ t _ { k } \} \gets$ the defined closest-approach,   
direction before T or before ego lane change speed-saturation, and eventual-overtake   
then instants, clamped to $T ^ { \prime } { \mathrm { ; } }$   
$t _ { v } \gets$ earliest predicted entry time of v into $h i t \gets \exists t _ { k } \exists \sigma ^ { \prime } \in f o r e c a s t ( \sigma _ { v } , t _ { k } )$   
$\ell _ { e }$ within the current ego time window; $r z _ { e } ( t _ { k } )$ ∩ CollisionZone $( v , \sigma ^ { \prime } ) \neq \emptyset ;$   
Add $( v , \pi _ { e } , t _ { v } )$ to ${ \mathcal { F } } ;$ else   
$r z _ { v } \gets \mathrm { R e a c h Z o n e } ( v , \sigma _ { v } , T ^ { \prime } ) ;$   
if this node’s set is empty and linkage actions can $h i t \gets r z _ { e } ( T ^ { \prime } ) \cap r z _ { v } \neq \emptyset ;$   
extend $\pi _ { e }$ then   
foreach next reachable forward lane $\ell _ { e } ^ { \prime }$ do if hit then Add $( v , \pi _ { e } , t _ { v } )$ to $\nu ;$   
if $t _ { e } ^ { \prime } \leq T$ then Enqueue $( \pi _ { e } ^ { \prime } , \ell _ { e } ^ { \prime } , t _ { e } ^ { \prime } )$ into $Q ;$ return V   
return $\mathcal { F }$

2) Formalization of the Time-to-Collision Rule: Building on front-vehicle identification, we formalize the rule that the ego vehicle must maintain at least a two-second timeto-collision (TTC) with front vehicles. Prospection is again central: MakSafe computes the ego reachability zone under forecasted dynamics, so the check reflects whether a collision would occur, on a specific ego trace, within the safety horizon, not whether one is already occurring. The horizon is closed, so an overlap arising exactly at the two-second bound already counts as a violation, which places the check on the conservative side of the threshold.

Starting from an initial global state, reachable configurations at time $\bar { T }$ are

$$
\mathsf { R e a c h } ( q _ { 0 } , T ) = \prod _ { i = 1 } ^ { n } \mathsf { R e a c h } _ { v _ { i } } ( \sigma _ { i } , T ) ,\tag{10}
$$

where each Reach $\qquad \quad _ { | v _ { i } } ( \sigma _ { i } , T )$ collects the states $v _ { i }$ can reach from $\sigma _ { i }$ within horizon T. A collision occurs when some reachable configuration has overlapping zones:

$$
\begin{array} { r l } & { \exists ( \sigma _ { 1 } ^ { \prime } , \dots , \sigma _ { n } ^ { \prime } ) \in \mathsf { R e a c h } ( q _ { 0 } , T ) , \ \exists i \neq j : } \\ & { \quad \mathsf { C o l l i s i o n Z o n e } ( v _ { i } , \sigma _ { i } ^ { \prime } ) \cap \mathsf { C o l l i s i o n Z o n e } ( v _ { j } , \sigma _ { j } ^ { \prime } ) \neq \emptyset . } \end{array}\tag{11}
$$

As specified in Algorithm 2, we forecast both the ego and the front vehicle over the reduced horizon. For a samedirection candidate we test overlap between the ego reachability zone and the front vehicle’s forecast collision zones at finitely many critical instants clamped to the horizon endpoint. Opposite-direction candidates instead intersect the two reachability zones.

## E. Complexity

We characterize the computational cost of rule evaluation by separating ego-branch exploration from per-vehicle query processing.

If $A _ { e }$ is the number of admissible ego actions per node and $D$ the bounded exploration depth, ego-trace exploration is approximately

$$
O ( A _ { e } ^ { D } ) .\tag{12}
$$

The end-to-end cost also scales with the number of surrounding vehicles N and the per-vehicle query cost $C _ { \mathrm { { q u e r y } } }$ used for occupancy/entry-time checks, yielding $O ( A _ { e } ^ { D } \cdot \dot { N } \cdot \mathbf { \dot { C } _ { \mathrm { q u e r y } } } )$ in the typical ego-anchored workflow. Two design choices bound this cost: evaluating at action points rather than at $N _ { t }$ uniformly sampled time steps removes the time-sampling factor, and ego-only branch labels avoid the $O ( \prod _ { i } A _ { v _ { i } } ^ { D } )$ blowup of enumerating every agent’s choices whenever surrounding forecasts stay single-valued within the horizon. The cost is coverage: MakSafe does not explore non-deterministic surrounding behaviour, so verdicts hold only under the assumed forecast.

The artifact exposes this specification model as a hostlanguage DSL in Haskell rather than as a standalone surface language. A dedicated end-user concrete syntax, such as a BNF-defined surface language, remains future work.

## IV. SPATIOTEMPORAL SAFETY ASSESSMENT

## A. Verification Under TTC Constraints

We evaluate MakSafe on a scenario that demonstrates its core contribution: assessing safety under spatiotemporal constraints where the trace tree determines different front vehicles for different action choices, yielding divergent safety verdicts. Specifically, the ego vehicle approaching a lane transition must evaluate whether each lane choice violates the TTC rule, a spatiotemporal constraint requiring the ego to maintain enough distance from the front vehicle such that a collision would occur more than 2 seconds in the future (i.e., a time-tocollision above 2 s is safe). This assessment is trace-dependent: the choice of lane determines which vehicle becomes the front vehicle, and thus which verdict applies. The Haskell implementation directly encodes and executes the scenario.

1) Road Network and Initial Scene: The road network, shown in Fig. 2, consists of three lanes: the First lane: road 1, lateral index 1, longitudinal span [0, 80] m; the Left lane: road 2, lateral index 1, longitudinal span [0, 200] m; the Right lane: road 2, lateral index −1, longitudinal span [0, 200] m. Two links connect the first lane to the left and right lanes respectively.

![](images/2d53774658646d9e4e20e8b2f17e0357d9a265706c9d5eef6b7226cdb1705023.jpg)  
Fig. 2. Road topology for the evaluation scenario.

The initial scene (Fig. 3) contains three vehicles, each with a length of 3 m. The ego vehicle V1 starts at 73.5 m in the first lane and travels at 5 m/s. Because it is close to the lane end, it must imminently take one of the two available links. Vehicle V2 sits at 2 m in the left successor lane and also travels at 5 m/s, while vehicle V3 lies at 2 m in the right successor lane and travels at 2 m/s.

2) Identification of Front Vehicles: A key property of MakSafe is that the safety verdict is trace-dependent, not a global Boolean. The trace tree, rooted at the initial scene, branches according to the ego’s admissible actions. Here, the ego has two lane-assumption actions at t = 1.0 s: assume the left lane or assume the right lane. As illustrated in Fig. 4, each action branch identifies a different front vehicle. The framework then evaluates the TTC rule independently for each trace, resulting in opposing verdicts.

![](images/a821e6a593b4f9d0e734e5dc0121cad3c8f79f620e5b33e74bc67629716f0fc8.jpg)  
Fig. 3. Initial scene: ego V1 (blue) in the first lane, V2 (red) in the left lane, V3 (green) in the right lane.

![](images/e74859efd7c3a70ce21c53fdbfc0cfba2745bc550f87070eab482d53589e5935.jpg)  
Fig. 4. Trace tree for the TTC rule. Assuming the left lane is safe (π : q<sub>0</sub> → q<sub>1</sub>); assuming the right lane violates TTC with V3 (π : q<sub>0</sub> → q<sub>2</sub>).

The front-vehicle identification algorithm returns both V2 and V3 as front vehicles, each associated with the trace that leads the ego toward its lane.

Since the ego speed is 5 m/s and V3 moves at only 2 m/s, under a two-second horizon the ego reachability zone overlaps V3’s collision zone if the ego takes the right-lane transition, giving a TTC violation. The left-lane assumption is safe because the relative longitudinal speed between the ego and V2 is zero, so no TTC violation occurs within the horizon.

3) Scenario-Level Check: Beyond a single scene, MakSafe evaluates safety over a scenario, i.e. a timed sequence of scenes. We define a two-scene scenario: a control update decelerates the ego from 5 m/s to 2 m/s. After this update, MakSafe re-evaluates the rule from the same reference position: the slower ego no longer reaches the lane transition within the two-second horizon, so no front vehicle qualifies and MakSafe reports no violation. The deceleration restores safety.

## B. Performance Evaluation

To complement the qualitative case study, we evaluate runtime with the benchmark harness in the released artifact. The benchmark characterizes implementation-level computational behaviour rather than establishing asymptotic complexity bounds for all rule classes. For each configuration, we report front-vehicle identification time, TTC-check time, and their sum (total time), where $t _ { \mathrm { t o t a l } } = t _ { \mathrm { f r o n t V e h i c l e } } + t _ { \mathrm { T T C } }$ . The TTC check identifies front vehicles internally, so $t _ { \mathrm { T T C } }$ already covers that work and $t _ { \mathrm { t o t a l } }$ upper-bounds a single end-to-end check rather than measuring one. Table I reports the recorded baseline’s means; the README relates them to a rerun.

The measured campaign covers $N \in \{ 1 0 , 2 5 , 5 0 \}$ surrounding vehicles, horizons $H ~ \in ~ \{ 2 , 6 \} ~ \mathrm { s } .$ , and branch factors $B \in \{ 2 , 4 \}$ . The benchmark places surrounding vehicles deterministically from the run seed (lane, offset, speed in 2– 7 m/s, acceleration in $[ - 0 . 5 , 0 . 5 ] \ \mathrm { m } / \mathrm { s } ^ { 2 } )$ and fixes the ego at 610 m on a 650 m lane at 4 m/s with acceleration 0.5 m/s<sup>2</sup>. It reports one System.CPUTime delta spanning 80 inner iterations, then averages over 5 repeats, forcing each result, including its rendered textual report, to weak-head normal form through an integer checksum.

The benchmark generator parameterizes topology by B: one approach lane splits into B split lanes, then into B branch lanes, which merge back into a downstream corridor of three consecutive lanes. The resulting synthetic network contains 2B + 4 lanes and 3B + 2 links. The ego reaches no link within either horizon, yet B raises the TTC-check timings in Table I: it widens the admissible-link scan, and a front vehicle reaching a link within H forecasts one state per outgoing link.

TABLE I  
PERFORMANCE RESULTS (AGGREGATED MEANS)
<table><tr><td>N</td><td>B</td><td>H (s)</td><td>frontVehicle (ms)</td><td>TTC check (ms)</td><td>total (ms)</td></tr><tr><td>10</td><td>2</td><td>2</td><td>0.787</td><td>1.049</td><td>1.836</td></tr><tr><td>10</td><td>2</td><td>6</td><td>0.583</td><td>1.455</td><td>2.038</td></tr><tr><td>10</td><td>4</td><td>2</td><td>0.755</td><td>1.118</td><td>1.873</td></tr><tr><td>10</td><td>4</td><td>6</td><td>0.760</td><td>1.877</td><td>2.638</td></tr><tr><td>25</td><td>2</td><td>2</td><td>0.803</td><td>1.297</td><td>2.101</td></tr><tr><td>25</td><td>2</td><td>6</td><td>0.808</td><td>1.801</td><td>2.610</td></tr><tr><td>25</td><td>4</td><td>2</td><td>0.979</td><td>1.663</td><td>2.641</td></tr><tr><td>25</td><td>4</td><td>6</td><td>0.887</td><td>3.087</td><td>3.974</td></tr><tr><td>50</td><td>2</td><td>2</td><td>1.242</td><td>1.840</td><td>3.082</td></tr><tr><td>50</td><td>2</td><td>6</td><td>1.249</td><td>2.260</td><td>3.509</td></tr><tr><td>50</td><td>4</td><td>2</td><td>1.322</td><td>2.257</td><td>3.579</td></tr><tr><td>50</td><td>4</td><td>6</td><td>1.306</td><td>3.562</td><td>4.869</td></tr></table>

Overall, runtimes remain within a few milliseconds across all reported configurations (about 1.8 to 4.9 ms in Table I), which bounds one end-to-end check at roughly 23 to 61 µs. Total time generally increases with N and with horizon H, with the largest mean in this sweep at $( N , B , H ) = ( 5 0 , 4 , 6 )$

A key point is that runtime is not strictly monotone in each individual component. Front-vehicle identification performs identical work at both horizons in this sweep, so its variation across H stays within run-to-run noise, while the TTC check grows with the horizon and dominates the total trend.

The evaluation in this paper uses the open-source implementation of MakSafe, available at [19], with all code, scenarios, and instructions for reproducing the results.

## V. DISCUSSION

MakSafe separates the topological (lanes, links, connectivity), geometric (positions, zones, reachability), and dynamical (actions, evolution over time) layers, enabling modularity and reuse of abstract definitions across traffic systems.

MakSafe evaluates safety at semantic action points rather than at uniformly sampled timestamps, reasoning over the spatial reachability zones of Section III. We design the check to be conservative for the modelled longitudinal dynamics under the stated assumptions: constant-acceleration forecasting of surrounding traffic and conflicts encoded through shared lane segments. Verdicts are thus relative to this forecast, not a guarantee over all possible evolutions, and a machine-checked soundness statement remains future work, including a proof that the computed zones match their definitions in Section III.

The framework does not yet model continuous acceleration variation within an action interval, jerk-bounded profiles, curved geometry, or probabilistic prediction.

## VI. CONCLUSION

This paper presented MakSafe, a spatiotemporal semantic framework for specifying and checking road vehicle safety rules. The framework combines semantic zones, trace-tree forecasting, and action-based reasoning to support machinecheckable prospective safety analysis without uniform time discretization. Our evaluation supports two conclusions. First, the case study confirms interpretability: MakSafe identifies safe and unsafe branches from the same scene and localizes the triggering action sequence. Second, the performance study (Table I) places a single check in the tens of microseconds across the reported configurations, with the TTC check growing as the horizon does.

## ACKNOWLEDGMENT

This work has been supported by the French government, through the France 2030 investment plan managed by the Agence Nationale de la Recherche, as part of the “UCA DS4H” project, reference ANR-17-EURE-0004.

UNIVERSITÉ SYSTEMES NUMERIQUES CÔTE D'AZUR POUR L'HUMAIN

[1] R. Queiroz, T. Berger, and K. Czarnecki, “GeoScenario: An open DSL for autonomous driving scenario representation,” in IEEE Intelligent Vehicles Symposium (IV). IEEE, 2019, pp. 287–294.

[2] B. Schutt, T. Braun, S. Otten, and E. Sax, “SceML: A graphical¨ modeling framework for scenario-based testing of autonomous vehicles,” in ACM/IEEE International Conference on Model Driven Engineering Languages and Systems (MODELS). ACM, 2020, pp. 114–120.

[3] ASAM, “OpenSCENARIO standard,” https://www.asam.net, 2024.

[4] T. Menzel, G. Bagschik, and M. Maurer, “Scenarios for development, test and validation of automated vehicles,” in IEEE Intelligent Vehicles Symposium (IV). IEEE, 2018, pp. 1821–1827.

[5] A. A. Babikian, O. Semerath, and D. Varr´ o, “Concretization of abstract´ traffic scene specifications using metaheuristic search,” IEEE Transactions on Software Engineering, vol. 50, no. 1, pp. 48–68, 2024.

[6] A. Rizaldi and M. Althoff, “Formalising traffic rules for accountability of autonomous vehicles,” in IEEE 18th International Conference on Intelligent Transportation Systems. IEEE, 2015, pp. 1658–1665.

[7] A. Rizaldi, J. Keinholz, M. Huber, J. Feldle, F. Immler, M. Althoff, E. Hilgendorf, and T. Nipkow, “Formalising and monitoring traffic rules for autonomous vehicles in Isabelle/HOL,” in Integrated Formal Methods (IFM), ser. LNCS, vol. 10510. Springer, 2017, pp. 50–66.

[8] S. Maierhofer, A.-K. Rettinger, E. C. Mayer, and M. Althoff, “Formalization of interstate traffic rules in temporal logic,” in IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020, pp. 752–759.

[9] K. Esterle, L. Gressenbuch, and A. C. Knoll, “Formalizing traffic rules for machine interpretability,” in IEEE Connected and Automated Vehicles Symposium (CAVS). IEEE, 2020, pp. 1–7.

[10] A. Censi, K. Slutsky, T. Wongpiromsarn, D. S. Yershov, S. Pendleton, J. G. M. Fu, and E. Frazzoli, “Liability, ethics, and culture-aware behavior specification using rulebooks,” in IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 8536–8542.

[11] A. Collin, A. Bilka, S. Pendleton, and R. J. Duintjer Tebbens, “Safety of the intended driving behavior using rulebooks,” in IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020, pp. 136–143.

[12] Y. Sun, C. M. Poskitt, J. Sun, Y. Chen, and Z. Yang, “LawBreaker: An approach for specifying traffic laws and fuzzing autonomous vehicles,” in IEEE/ACM International Conference on Automated Software Engineering (ASE). ACM, 2022, pp. 62:1–62:12.

[13] R. Alur and D. L. Dill, “A theory of timed automata,” Theoretical Computer Science, vol. 126, no. 2, pp. 183–235, 1994.

[14] T. A. Henzinger, “The theory of hybrid automata,” in Verification of Digital and Hybrid Systems, M. K. Inan and R. P. Kurshan, Eds. Springer, 2000, pp. 265–292.

[15] G. Behrmann, A. David, and K. G. Larsen, “A tutorial on UPPAAL,” in Formal Methods for the Design of Real-Time Systems. Springer, 2004, pp. 200–236.

[16] O. Maler and D. Nickovic, “Monitoring temporal properties of continuous signals,” in Formal Techniques, Modelling and Analysis of Timed and Fault-Tolerant Systems. Springer, 2004, pp. 152–166.

[17] P. Bender, J. Ziegler, and C. Stiller, “Lanelets: Efficient map representation for autonomous driving,” in IEEE Intelligent Vehicles Symposium (IV). IEEE, 2014, pp. 420–425.

[18] ASAM, “OpenDRIVE standard,” https://www.asam.net, 2024.

[19] M. Labzhaniia, J. Deantoni, M.-A. Peraldi-Frati, and F. Mallet, “MakSafe experiments and benchmark,” Zenodo, Jul. 2026. [Online]. Available: https://doi.org/10.5281/zenodo.21710396