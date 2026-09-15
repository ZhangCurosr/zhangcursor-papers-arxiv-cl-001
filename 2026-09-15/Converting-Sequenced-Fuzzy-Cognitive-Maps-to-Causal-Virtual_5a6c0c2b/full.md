# Converting Sequenced Fuzzy Cognitive Maps to Causal Virtual Worlds with Large Video Generators

Akash Kumar Panda<sup>1</sup> and Olaoluwa Adigun<sup>2</sup> and Bart Kosko<sup>3</sup>

Abstract— We show how users can create and manipulate causal virtual worlds with large-language-model (LLM) and large-video-model agents. The approach uses feedback fuzzy cognitive maps (FCMs) both to model the granular causal structure of the virtual world and to guide its causal evolution. The local causal rules are partial or fuzzy while the FCM’s feedback structure produces global equilibria that define causal scenarios. A sequence of dynamical meta-rules of the form “If then ” define the causal scenes of the virtual-world video. The if-part causal pattern  perturbs the FCM’s virtual world at the user’s or agent’s discretion. The FCM’s transient feedback dynamics define the metarule’s causal arrow of implication. The then-part is the resulting equilibrium attractor such as a FCM limit cycle or fixed point. Our algorithm extracts these meta-rules from the FCM and guides the LLM agent to write a script based on the FCM meta-rule sequence. The large video generator converts the meta-rule into a video scene in accord with the flow of the dynamics. We applied the agent-based technique to a simple FCM that describes an undersea world of dolphins and sharks. Google’s Gemini 3.1 generated the script and Google’s Veo 3.1 generated the dolphin-shark video. The approach is general and can scale by mixing larger FCMs and AI agents to produce more immersive virtual worlds.

## I. Agentic AI Generation of Virtual Worlds from Sequenced Fuzzy Cognitive Maps

We show how to generate realistic causal virtual-world videos by combining feedback Fuzzy Cognitive Maps (FCMs) with large-video-generator agents.

This new virtual-world technique is general and scales for arbitrary mixed causal-graph FCMs combined with controlled AI agents. As OpenAI has stated: “scaling video generation models is a promising path towards building general purpose simulators of the physical world.” [1] Sequenced FCM dynamics ofers a practical way to approximate such models.

Figure 1 shows the system flow of stimulated FCM causal patterns that produce an AI-generated 1-minute video from a simple 5-node FCM. The FCM defines a coarse-grained undersea world of dolphins and sharks. We use this simple dolphin FCM because over the decades it has become a type of small-scale test and teaching model of an FCM [2]. The FCM itself is a nonlinear feedback dynamical system whose causal graph is signed and weighted or partial to reflect partial or fuzzy causality [3], [4], [4]–[10]. The local causal rule structure of the FCM helps make the nonlinear system interpretable or explainable [11], [12].

The FCM-generated video shows how a pod of playful dolphins responds to the threat of a large tiger shark. The video uses a sequence of 8 scenes as we explain in detail in the next two sections. Figure 2 shows further snapshots from the video.

Figure 3 shows the 5-node dolphin-shark FCM and its corresponding 5-by-5 causal edge matrix E. This paper uses just this small FCM to illustrate the process of generating virtual worlds although we stress again that the process is completely general for any size or mixture of FCMs.

Figure 4 shows the sequence of 8 scenes as 8 inputoutput pairs of stimulated FCM states that map to equilibrium limit cycles or fixed-point attractors. This gives a type of dynamical system “storyboard” of the FCM-based video or virtual world. Users can pick the stimulus patterns directly or ask the AI agent to pick the nearest-matching patterns for them. Users can also allow an agent to pick the stimulus patterns itself.

Figures 5-8 show frames from the first 4 scenes of the final video. Figure 9 shows all sequenced limit cycles used.

The key insight is that a FCM dynamical system defines a set of causal meta-rules of the form $\mathcal { A } _ { j }  \mathcal { B } _ { j }$ We “sequence” or impose a time order on these FCM meta-rules. We then use the sequenced meta-rules to drive the time-ordered scenes of the generated video or virtual world.

The nonlinear FCM dynamical system F itself is a time-varying mapping $\mathcal { F } : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ from a causal pattern space such as R<sup>n</sup> back into itself. A pulsed or sustained (“clamped”) input vector $x \in \mathbb { R } ^ { n }$ produces a sequence of transient causal states that ends in an equilibrium attractor that can range from a fixed point to a limit cycle to aperiodic chaos. The simple binary-state dolphin FCM in Figure 3 always ends in a limit cycle or a fixed-point attractor that lives inside the binary 5-cube $\{ 0 , 1 \} ^ { 5 }$ . The fuzzy causal edge values $e _ { i j }$ of the directed causal edge from concept node $C _ { i }$ to node $C _ { j }$ takes values in the bipolar interval [−1, 1] in general where positive values $e _ { i j } > 0$ denote causal increase and negative values $e _ { i j } < 0$ denote causal decrease and $e _ { i j } = 0$ denotes no causal efect.

The global meta-rule $\mathcal { A } _ { j }  \mathcal { B } _ { j }$ asks and answers its own “what-if” question: What dynamical equilibrium $B _ { j }$ results if we perturb the FCM causal system with new input x or otherwise knock it into a new state of disequilibrium $\mathbf { \nabla } \mathcal { A } _ { j } ?$

So the implication arrow “→” in the meta-rule $A _ { j } $ $B _ { j }$ is not a classical logical or material-implication operator. The arrow is instead a causal implicant that depends on system transients and hence depends on time in general. These are unconstrained rules. The user can add constraint rules that guide or extend the equilibria and so modify the FCM video.

The resulting set of such dynamical meta-rules $A _ { j } $ $B _ { j }$ endows the FCM system a with a global level of explainability or XAI. This still holds in the general case when we take convex combinations $w _ { 1 } \mathcal { F } _ { 1 } + \cdot \cdot \cdot + w _ { K } \mathcal { F } _ { K }$ of K-many FCMs $\mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { K }$ for nonnegative mixing weights $w _ { j }$ that sum to unity and that can depend on x. Mixing FCM causal edge matrices $E _ { 1 } , \dots , E _ { K }$ always produces a new causal edge matrix and thus always produces an FCM F. Appropriate zero-padding of rows and columns ensures that the mixed causal edge matrices are conformable for matrix addition.

This mixing-closure property greatly extends the application reach and knowlege-representation power of FCMs. Users can combine and uncombine and adapt the underlying mixed FCM at will. Users can also pick the mixed meta rules and especially the disequilibrium stimuli $A _ { j }$ at will to create and change the virtual world. Agentic AI systems can further assist in this immersive world-creating process.

The next section reviews FCMs and show how to sequence and fire their meta-rules. It shows how the dolphin FCM’s 8 causal meta-rules $\mathcal { A } _ { 1 }  \mathcal { B } _ { 1 } , . . . , \mathcal { A } _ { 8 } $ $B _ { 8 }$ become the 8 scene prompts to the AI video generator.

The last section explains how to convert these FCM scenes to prompts that control Google’s Veo 3.1 largevideo generator by way of Algorithm 1 below. We here focus on user-driven choices of meta-rules $\mathcal { A } _ { j }  \mathcal { B } _ { j }$ and resulting scenes but recent work shows that we can easily use large language models (LLMs) to generate FCM knowledge graphs from text or vice versa [13] in more complex agentic AI systems.

## II. Sequenced Fuzzy Cognitive Maps

## A. Fuzzy Cognitive Maps (FCM)

FCMs model causal dynamical systems as directed weighted graphs [3], [4], [4]–[10]. The concept nodes describe the causal variables in the dynamical system and the directed edges describe the causal relationships between those nodes. FCMs allow feedback and therefore converge to non-trivial equilibria like limit cycles. FCMs model dynamical systems by approximating their underlying maps from inputs to equilibria.

## B. Causal Edge Matrix

The directed edges of the FCM describe the causal relationships between concept nodes. An edge $e _ { i j }$ from the $i ^ { \mathrm { t h } }$ concept node $C _ { i }$ to the $j ^ { \mathrm { t h } }$ concept node $C _ { j }$ means ${ } ^ { 6 6 } C _ { i }$ causes $C _ { j } { } ^ { \mathfrak { N } } .$ . The fuzzy edge weights describe partial causality and we use the same symbol for this value—it can also vary with time. The causal edge weight $e _ { i j } \in [ - 1 , 1 ]$ on the ijth edge states the degree to which $C _ { i }$ causes $C _ { j }$

$$
e _ { i j } = D e g r e e ( C _ { i } \to C _ { j } ) .\tag{1}
$$

A positive $e _ { i j }$ means that $C _ { j }$ increases when $C _ { i }$ increases and a negative $e _ { i j }$ means that $C _ { j }$ decreases if $C _ { i }$ increases. The weight $e _ { i j }$ is zero when there is no causal edge between $C _ { i }$ and $C _ { j }$ . The magnitude of $e _ { i j }$ is high if there is a strong causal relationship between $C _ { i }$ and $C _ { j }$ . A low magnitude of $e _ { i j }$ describes a weak causal relationship between $C _ { i }$ and $C _ { j }$

An n × n matrix E describes all the directed weighted edges of a n-node FCM. The causal edge weight $e _ { i j }$ corresponds to the matrix element on the $i ^ { \mathrm { t h } }$ row and the $j ^ { \mathrm { t h } }$ column. The matrix element is zero if there is no edge between the corresponding node-pair.

Figure 3 shows the dolphin FCM with 5 nodes: “Herd Clustering”, “Fatigue”, “Rest”, “Survival Threat”, and “Run Away”. The figure also shows its corresponding causal edge matrix. The matrix element $e _ { 4 5 }$ on the $4 ^ { \mathrm { t h } }$ row and the $5 ^ { \mathrm { t h } }$ column is 1 because there is a positive edge with weight 1 from the $4 ^ { \mathrm { t h } }$ concept node “Survival Threat” to the $5 ^ { \mathrm { t h } }$ concept node “Run Away”. This says that the dolphins “run away” when a “survival threat” like a shark is present.

## C. FCM Evolution

A n-dimensional row vector $C ( t ) \in [ 0 , 1 ] ^ { n }$ describes the state of the FCM’s concept nodes at time t. The $i ^ { \mathrm { t h } }$ node is “active” or $^ { 6 6 } \mathrm { o n } ^ { \dag }$ at time t if the $i ^ { \mathrm { t h } }$ component $C _ { i } ( t )$ of the state vector $C ( t )$ is equal to or close to one. The $i ^ { \mathrm { t h } }$ node is “inactive” or “of” at time t if $C _ { i } ( t )$ is equal to or close to zero. A node is partially active otherwise. The causal variables corresponding to the active nodes are present in the system and those corresponding to the inactive nodes are absent. The causal factors are partially present in the system if their corresponding nodes are partially active.

Consider the dolphin FCM in figure 3. If $C ( t ) \ =$ $\left( 0 \quad 0 \quad 0 \quad 1 \quad 0 \right)$ then the $4 ^ { \mathrm { t h } }$ node “Survival Threat” is $^ { 6 6 } \mathrm { o n } ^ { \dag }$ and all other nodes are $ { { } ^ { 6 6 } } \mathrm { O f f } ^ { 9 }$ at time t. This means there is a “threat” like a shark lurking around.

FCMs evolve in discrete time through vector-matrix multiplication and nonlinear compression–at least in the simple FCM we use for this video creation. The state $C _ { j } ( t + 1 )$ of the $j ^ { \mathrm { t h } }$ concept node $C _ { j }$ at discrete time step t + 1 is:

$$
C _ { j } ( t + 1 ) = \Phi { \bigg ( } \sum _ { i = 1 } ^ { n } C _ { i } ( t ) e _ { i j } { \bigg ) }\tag{2}
$$

![](images/949235ddadc72311b17d7a3558bf9fcf59a586729f2994ddd795a5616030caa2.jpg)  
Fig. 1: Frames from a video of the virtual world generated from a 5-node Fuzzy Cognitive Map (FCM) that models dolphin behavior in presence of a “Survival Threat”. The video shows a pod of dolphins running away from a shark over 8 scenes. The input stimulus for each scene is above the frame corresponding to the scene. The figures on the left and right of the video frames show the respective state of the FCM at the beginning and the end of the video. The figure highlights clamped node “Herd Clustering” in green.

if Φ is a nonlinear function bounded between zero and one.

The sum $\textstyle \sum _ { i = 1 } ^ { n } C _ { i } ( t ) e _ { i j }$ is the matrix product between the state row-vector C(t) and the edge matrix E. The nonlinear function Φ then compresses this product between zero and one. We stress that Φ is an arbitrary but bounded nonlinear function in general even though here we use the simplest case of a threshold nonlinearity.

This process repeats itself to give the discrete-time evolution of the FCM. The FCM starts with the initial state C(0) at time $t ~ = ~ 0$ and then goes through the states $C ( 1 ) , C ( 2 ) , C ( 3 )$ , and so on in order. The active nodes in this state-vector sequence qualitatively describe the trajectory of the dynamical system that the FCM models.

Consider the dolphin FCM with an initial state $C ( 0 ) =$ $\left( 0 \quad 0 \quad 0 \quad 1 \quad 0 \right)$ . Equation 2 and figure 3(b) give $C ( 1 ) =$ $\left( 1 \quad 0 \quad 0 \quad 0 \quad 1 \right)$ . “Survival Threat” was active at $t = 0$ and “Herd Clustering” and “Run Away” were active at $t = 1$ . This shows the dolphins’ behavior in the presence of a shark. They group themselves together in herd clusters and run away from the shark.

## D. FCM Equilibria

The equilibria characterize a dynamical system. The equilibrium behavior of the FCM depends on the limiting behavior of the state-vector sequence. The FCM converges to a “fixed point” if the state-vector sequence converges to a constant vector. The FCM converges to a K-step “limit cycle” for an integer $K \ > \ 1$ if $C ( t { + } K ) = C ( t )$ somewhere in the state-vector sequence.

Then the FCM converges to an equilibrium where K state vectors repeat themselves over and over in the same order. The FCM may also converge to a chaotic attractor where there are no repeating patterns in the state-vector sequence.

Consider the dolphin FCM with same initial state $\begin{array} { r l r } { C ( 0 ) } & { { } = } & { \left( 0 \quad 0 \quad 0 \quad 1 \quad 0 \right) } \end{array}$ . Equation 2 and figure $3 ( \mathrm { b } ) \mathrm { g i v e } C ( 1 ) = ( 1 0 0 0 0 1 ) , C ( 2 ) = $ $( 0 , 1 \quad 0 \quad 0 \quad 0 ) , C ( 3 ) = ( 0 \quad 0 \quad 1 \quad 0 \quad 0 )$ , and $C ( 4 ) =$ ${ \bigl ( } 0 \quad 0 \quad 0 \quad 1 \quad 0 { \bigr ) } = C ( 0 )$ . The state vectors C(0), C(1), C(2), and C(3) repeat themselves over and over in the same order. The dolphin FCM converges to a 4-state limit cycle. The active nodes in these states respectively are “Survival Threat”, “Herd Clustering” and “Run Away”, “Fatigue”, and “Rest”. So dolphins run away from threats like sharks in herd clusters. They then get tired and need to rest. The shark then catches up to the dolphins while they rest. This starts the cycle all over again.

The set of all initial conditions C(0) that lead to a given equilibrium describes the “basin of attraction” for that equilibrium. The FCM describes a map from these basins to their corresponding equilibrium attractors. The basins of the FCM’s equilibria partition the FCM’s input space. The FCM models a dynamical system by approximating its corresponding basin-to-equilibrium map.

## E. Causal State Activation: Clamping vs. Pulsing

“Clamping” the $k ^ { \mathrm { t h } }$ node $C _ { k }$ to a value $c _ { k } \in [ 0 , 1 ]$ forces $C _ { k } ( t )$ to equal $c _ { k }$ irrespective of the values of other nodes at time t−1. The $k ^ { \mathrm { t h } }$ node is clamped $^ { 6 6 } \mathrm { o n } ^ { \dag }$ if $c _ { k }$ ≈ 1 and it is clamped $ { { } ^ { 6 6 } } \mathrm { O f f } ^ { 9 }$ $c _ { k } \approx 0$ . Clamping acts as a forcing function and pushes the FCM into new equilibria. FCMs can clamp one or more nodes at a time. Clamping is often a way to implement policies and answer What-if questions regarding their efects.

![](images/e9481825dbf93351a60e808b5d05a6b702fd4b0d3ff9da92151b4e81214232d1.jpg)  
(a) Scene 1  
(b) Scene 2

![](images/991a191f712bf9e9fb7d142954398f8a3c0d6bc3546ff00e6821f41447e1b5c8.jpg)

![](images/511bf20ba785afd211ea2a9b4dc14404e2aaa2197814266f9d935e5e2faf8aff.jpg)

(c) Scene 3  
![](images/5c3ac1198a866c6d93558f0dd23b53c2da8c3e263f183b5fb8c8d74c752491de.jpg)  
(d) Scene 4

![](images/9809ed7c38d9412014657e8c4671e08717ba0c8e182328af753c49b318d254df.jpg)  
(e) Scene 5

![](images/241344f8026713d8c5c22e84b96d7e74858f4924ff27ae2d7a9ba37abe6ee536.jpg)

![](images/f6680ce1347f0c02dc68a678523052c45e5df60bb031653da6eb6a6dacb47fd0.jpg)  
(g) Scene 7

(f) Scene 6  
![](images/e1063f0e226d7f78dad795e58d862ec9b9ab0a13f50a036240c98e9bb28f4266.jpg)  
(h) Scene 8  
Fig. 2: Limit cycle 1: Interaction of a dolphin pod and a tiger shark in a water. (a). Scene 1: The dolphin pod rests and plays. (b). Scene 2: The dolphin pod runs away from a threat, gets tired, and rests. (c). Scene 3: The dolphins swim in a herd cluster to avoid a larg tiger shark. (d). Scene 4: The tiger shark breaks the dolphins’ herd-cluster apart as the dolphins tire. (e). Scene 5: The tired dolphins stop running away and regroup in a tighter herd-cluster. (f). Scene 6: The dolphins push through their fatigue and swim away from the large tiger shark. (g). Scene 7: The dolphin pod gets away from the large tiger shark. (h). Scene 8: The dolphin goes back to playing and resting until their next shark encounter.

![](images/4f9d1c1768d117f8ea2590614c909c8b4ba3221caf961da885bcfd6875b0275f.jpg)

![](images/9633a6c38ce65dc435059b61e16e61d9c2acddc96ce80af15b21cf4371db579a.jpg)  
(b)  
Fig. 3: The 5-node Dolphin FCM with trivalent causal edges. (a) The directed weighted graph for the dolphin FCM. The directed edge $e _ { i j }$ is trivalent because $e _ { i j } \in \{ - 1 , 0 , 1 \}$ but in general can be properly fuzzy and take on any directed causal degree in the bipolar interval [−1, 1]. The positive edges are in blue and the negative edges are in red. (b) The causal edge matrix E that corresponds to the dolphin FCM. The 5 concept nodes $C _ { 1 ^ { - } } C _ { 5 }$ index both the rows and the columns. The matrix elements give the edge weights $e _ { i j } .$ The positive edge values $e _ { i j } > 0$ are in blue and the negative edge values $e _ { i j } < 0$ are in red.

Clamping on the “Survival Threat” node changes the dolphin FCM’s limit cycle from $\left( 0 \quad 0 \quad 0 \quad 1 \quad 0 \right)$ $ \quad ( 1 \quad 0 \quad 0 \quad 0 \quad 1 ) \quad  \quad ( 0 \quad 1 \quad 0 \quad 0 \quad 0 ) \quad $ $\left( 0 \quad 0 \quad 1 \quad 0 \quad 0 \right) \quad \to \quad \left( 0 \quad 0 \quad 0 \quad 1 \quad 0 \right)$ to a new 3- state limit cycle $( 0 \quad { \mathrm { ~ 1 ~ ~ 0 ~ 1 ~ ~ 0 ~ } } ) \to { \ ' } ( 1 \quad 0 \quad 0 \quad 1 \quad 0 )$ $ \ ( 1 \mathrm { ~  ~ 1 ~ } 0 1 0 )  \ ( 0 1 0 1 ) \ 0 )$ . The active nodes in this new limit cycles respectively are $^ { \mathrm { 6 6 } } \mathrm { F a t i g u e } ^ { \mathrm { 3 } }$ and “Survival Threat”; “Herd Clustering” and “Survival Threat”; and “Herd Clustering”, “Fatigue”, and “Survival Threat”. The shark is always present in this case. The dolphins cluster and get tired over and over. The nodes “Rest” and “Run Away” always of because the dolphins do not get a chance to rest and they never escape the shark.

FCMs can also ‘pulse’ a node instead of clamping it. ‘Pulsing’ the $k ^ { \mathrm { t h } }$ node $C _ { k }$ to a value $c _ { k }$ at time $t _ { 0 }$ forces $C _ { k } ( t _ { 0 } )$ to equal $c _ { k }$ irrespective of the values of other nodes at time $t _ { 0 } - 1$ . This only afects the FCM at time $t _ { 0 }$ and the FCM reverts back to its unperturbed dynamics after $t _ { 0 }$ . The node $C _ { k }$ pulses $\cdot _ { \mathrm { o n } } \cdot$ at time $t _ { 0 }$ if $C _ { k } ( t _ { 0 } ) \approx 1$ and it pulses ‘of’ at time $t _ { 0 }$ if $C _ { k } ( t _ { 0 } ) \approx 0$ . FCMs can also pulse more than one node at a time.

The unperturbed dolphin FCM may converge to its fixed point $\left( 0 \quad 0 \quad 0 \quad 0 \quad 0 \quad 0 \right)$ . “Survival Threat” ‘on’- pulse can drive the FCM into a 4-step limit cycle instead.

## F. FCM Sequencing

FCM sequencing stimulates the FCM with a sequence of clamping patterns $\mathcal { A } _ { k }$ to push it through a corresponding sequence of equilibria $\boldsymbol { B } _ { k }$ . This map from stimulus patterns $\mathcal { A } _ { k }$ to their corresponding FCM equilibria $\boldsymbol { B } _ { k }$ define a set of causal ‘if-then’ meta-rules $\mathcal A _ { k } ~ \to ~ \mathcal B _ { k }$ The causal if-part $\mathcal { A } _ { k }$ stimulates the FCM at the user’s discretion through clamps and pulses. The arrow describes the transient dynamics that connect $\mathcal { A } _ { k }$ to its corresponding equilibrium $\boldsymbol { B } _ { k }$ . The stimulus pattern destroys the FCM’s prior equilibrium and before pushing it into a new one.

Figure 4 shows this process for a sequence of 8 if-parts $\boldsymbol { A } _ { 1 } \ – A _ { 8 }$ to the dolphin FCM. The figure also shows the corresponding then-parts $B _ { 1 } { - } B _ { 8 }$ . The transition from the if-part stimulus to the then-part equilibrium defines each of the 8 scenes in the virtual world video.

Scene 1 clamps on “Herd Clustering” and shows the FCM converging to a 4-stage limit cycle. The dolphins play together as a group, get tired, rest, and then go back to playing. Scene 2 disrupts this equilibrium $\boldsymbol { B } _ { 1 }$ by unclamping “Herd Clustering and ‘on’-pulsing “Survival Threat”. This moves the system from $\boldsymbol { B } _ { 1 }$ to another 4- stage limit cycle $B _ { 2 }$ where the dolphins come across a shark, run away, get tired, rest, and encounter the shark again. Scene 3 then interrupts this equilibrium through stimulus $\mathcal { A } _ { 3 }$ . This process then repeats 5 more times with stimuli $\mathcal { A } _ { 4 } { - } \mathcal { A } _ { 8 }$ . The FCM in turn converges to 2 more limit cycles $B _ { 4 }$ and $B _ { 8 }$ and 3 more fixed points ${ \cal B } _ { 5 } – { \cal B } _ { 7 }$ The 8 scenes together tell the story of a pod of dolphins that gets chased by a persistent shark and has to push through fatigue to escape.

Figures 5-8 show the equilibria $B _ { 1 } – B _ { 4 }$ in detail. They show one frame from each state of the limit cycles. They also show the active nodes in those states. The limit cycle for $B _ { 8 }$ is the same as $B _ { 2 }$ . The FCM equilibria ${ \cal B } _ { 5 } – { \cal B } _ { 7 }$ are fixed points and thus have only one state. Figure 2(e)-(g) show the video frames that best represent those fixedpoint states.

## III. Agentic Control of Large Video Generator Scene Prompts

We present the agentic control for generating the causal virtual world for an FCM. This explains how to generate a video from a sequence of equilibria for an FCM with a large video generator. The sequence of equilibria comes from the concatenation of equilibria that result from the combination of input pulse signals and node clamping operation. We tested our method with Google’s Veo 3.1 video generator.

![](images/e834b1f9c6173e3d17acce482fe78a789ddbe12b46d131bb0c6308b87184ba40.jpg)  
Fig. 4: Sequenced dolphin FCM Equilibria: A sequence of 8 input patterns and clamping conditions $\mathcal { A } _ { k }$ stimulate the dolphin FCM into their respective equilibria $\boldsymbol { B } _ { k }$ . The scene k stimulates the FCM withA . This breaks the FCM’s prior equilibrium $B _ { k - 1 }$ and pushes it into its new equilibrium $\boldsymbol { B } _ { k }$

Algorithm 1: Causal Virtual World Generation   
Input: $\scriptstyle { \mathcal { G } } = \left( \nu , \mathbf { E } \right)$ : FCM with $\mathbf { E } { \in } \mathbb { R } ^ { n \times n } ;$   
$\mathbf { c } ^ { ( 0 ) } \in \{ 0 , 1 \} ^ { n } \colon$ initial state and clamping   
conditions; $T \colon$ time steps; $\varphi ( \cdot ) \colon$ threshold   
function; $\mathcal { P } _ { \mathrm { s c r i p t } } \colon$ LLM scripting prompt;   
$\mathcal { P } _ { \mathrm { v i d e o } } \colon$ LVM generation prompt; $\mathrm { L L M } ( \cdot )$   
LVM(·): language and video models   
Output: D: causal virtual world documentary   
// Step 1: FCM Simulation   
1 Initialize $\mathbf { C } { \gets } \{ \ \} , { \ } { \mathbf { c } } { \gets } \mathbf { c } ^ { ( 0 ) }$   
2 for t = 0 to $T - 1$ do   
3 Store $\mathbf { C } [ \cdot , t ] \gets \mathbf { c } .$ , then update $\mathbf { c } \gets \varphi ( \mathbf { E } \cdot \mathbf { c } ) ;$   
4 end   
5 Detect equilibrium: set eq ← FixedPoint or   
LimitCycle;   
// Step 2: LLM Script Generation   
6 Analyze C for START, STOP, PERSIST transitions   
across all t;   
7 ${ \mathcal { S C } } \gets \mathrm { L L M } ( { \mathcal { P } } _ { \mathrm { s c r i p t } } , { \mathcal { V } } , \mathbf { C } , \mathrm { e q } ) \ ; \ / /$ yields title +   
T scene descriptors $\{ p _ { t } , l t _ { t } \}$   
// Step 3: LVM Clip Generation and Assembly   
8 $\mathcal { D }  \mathrm { L V M } ( \mathcal { P } _ { \mathrm { v i d e o } } , \{ p _ { t } \}$ , {lt<sub>t</sub>}, SC.title) ;   
$/ /$ generate and concatenate clips   
9 return $\mathcal { D } ;$

Our method involves two main steps: screenwriting and movie production The screenwriting step converts an input sequence of equilibria into a movie script. The script comprises multiple scenes. Each script can capture the causal information of a single state, a sequence of states, or a single equilibrium state. The choice depends on multiple factors: the number of FCM nodes, the degree of information granularity in the visual world, the maximum video length of the video generator agent, and so on.

![](images/34f769a8b5afd28a2fd15dc23f4db245407df0e8244b68e7083e8e3424414b29.jpg)  
Fig. 5: Scene 1 from the virtual world video. $\mathcal { A } _ { 1 }$ stimulates the dolphin FCM by clamping on the “Herd Clustering” node. The dolphin FCM the converges to a 4-stage limit cycle $\bar { B _ { 1 } }$ . The figure shows one frame from each stage of the limit cycle and also mentions their corresponding active nodes next to the frame.

An LLM agent with screenwriting prompt $\mathcal { P } _ { \mathrm { s c r i p t } }$ turns a list of active nodes in an FCM state vector into a sentence that describes the state based on the node names. The agent turns the equilibrium state sequence into a paragraph that describes the FCM equilibrium. The agent condenses that 1-paragraph description into a sentence that describes a scene. These scene descriptions form the script for the video generator.

The movie production feeds the movie script into the video generator agent and prompts it with $\mathcal { P } _ { \mathrm { v i d e o } }$ to create the video. This video generation can be a single continuous shot or a sequence of shots. The single continuous shot generates all the scripts once. The maximum supported duration of the video generator agent constrains this approach. It may truncate important information. The sequential approach decomposes the main script into multiple sub-scripts. The system first generates a video segment for the initial sub-script. Then it produces the second video segment conditioned on the first segment. Each subsequent generation conditions on the cumulative output of prior video segments. This recursive cycle continues until the sub-script sequence concludes. The sequential method enables longer video durations than one-shot approaches. Algorithm 1 shows the complete method for script and video generation.

![](images/a799d4287a94f21fdc0dfcb42ed35a7d81d02951b7ff34297d8943086c866252.jpg)  
Fig. 6: Scene 2 from the virtual world video. $A _ { 2 }$ stimulates the dolphin FCM by pulsing on the “Survival Threat” node. The dolphin FCM the converges to a 4-stage limit cycle $B _ { 2 } .$ The figure shows one frame from each stage of the limit cycle and also mentions their corresponding active nodes next to the frame.

![](images/aa9ca84c55d3d7cff40955c6c208c2c0432308dd8866242e129685a10db17673.jpg)  
Fig. 7: Scene 3 from the virtual world video. $\mathcal { A } _ { 3 }$ stimulates the dolphin FCM by clamping on the “Survival Threat” node. The dolphin FCM the converges to a 3-stage limit cycle $B _ { 3 } .$ The figure shows one frame from each stage of the limit cycle and also mentions their corresponding active nodes next to the frame.

We present two agentic control methods based on autonomy levels. These include semi-autonomous and autonomous generations. Both methods utilize the twostep process of screenwriting and movie production. But each method varies in autonomy level during the

![](images/0374134c188612b6e55512ba6e6cd36b3b95308e6414b6c97e43b55838b6ef7b.jpg)  
Fig. 8: Scene 4 from the virtual world video. $\mathcal { A } _ { 4 }$ stimulates the dolphin FCM by clamping on the “Survival Threat” node as well as clamping of the “Herd Clustering Node. The dolphin FCM then converges to a 4-stage limit cycle $B _ { 4 } .$ . The figure shows one frame from each stage of the limit cycle and also mentions their corresponding active nodes next to the frame.

screenwriting step.

## A. Semi-autonomous agentic control

An LLM agent executes the screenwriting phase of this agentic control method. This operator converts a sequence of equilibria into a script. Script scenes align with state transitions within the equilibria sequence. Each scene details granular information derived from active state variables or nodes. The agent reads the vector sequence directly or employs an intermediate visualization. The agent inputs the script into the video generator agent. This agent transforms the script into a video or storyboard.

Let us consider the 5-node dolphin FCM in Figure 3. Figure 4 shows the links between a sequence of the dolphin FCM equilibria with pulsing and clamping. The sequence comprises of 8 equilibrium cases from the FCM. Figure 9 visualizes the 8 equilibria and their transitions. Each scene corresponds to an equlibrium state.

An LLM agent annotated the sequenced equilibria into a movie script. The movie script is divided into three parts: task description, summary, and the scene description for each equilibrium state. The task descrip tion captures the high-level description of the movie. The summary provides more specific information. The scene descriptions translates the state variables into visual elements, dialogue and character interactions that advance the central narrative as laid out in the summary.

Each scene in Figure 9 captures the causal transition from the causal if-part $\mathcal { A } _ { k }$ to its corresponding equilibrium $\boldsymbol { B } _ { k }$ . Consider the first pair $( A _ { 1 } , \ B _ { 1 } )$ . The first column $\mathbf { C } ( 0 ) = \left[ C _ { 1 } ( 0 ) , C _ { 2 } ( 0 ) , C _ { 3 } ( 0 ) , C _ { 4 } ( 0 ) , C _ { 5 } ( 0 ) \right]$ . The if-part corresponds to clamping on the “Herd Clustering” node while other nodes start in the OFF state. The description of Scene 1 follow from the causal interaction from C(0) to C(3).

![](images/525cacbe43e9f13d27ae83fea0953e87a45b7734057950cb556b928f88b587f1.jpg)  
Fig. 9: Sequence of all FCM equilibria in the video: Clamped-on states are in red and clamped-of states are in blue.

Here is a snippet of our prompt to the LLM for script generation.

A snippet from the script-generation prompt   
$\mathcal { P } _ { \mathrm { s c r i p t } }$   
Task Guidelines:   
1. Analyze the Data:   
(a) Step $t = 0$ (The Opening): Establish the scene based   
on which nodes are active (1) in the first column.   
(b) Transitions (t = 1 to $t \doteq T )$ : Compare column t with   
column t 1.   
(i) Identify what Started (0 to 1).   
(ii) Identify what Stopped (1 to 0).   
(iii) Identify what Persisted (1 to 1).   
(c) Interpret the causal logic: If Node A is active at   
t = 1 and Node B activates at $t = 2 ,$ imply that A might   
be causing B.   
2. Documentary Structure:   
(a) Title: Generate a thematic title based on the Node   
List concepts.   
(b) Lower Thirds: Provide short text overlays   
summarizing the state logic (e.g., ”Rainfall triggers   
River Surge”).   
(c) Narrative (Voiceover): A 1-2 sentence script   
connecting the current state to the previous one.   
(d) Conclusion: Explicitly address the ”Equilibrium   
Behavior” in the final scene.

Google’s Veo 3.1 generative agent converted the LLMgenerated script into a movie. The maximum allowed duration for a Veo 3.1 generated video is 8 seconds. We used the multiple-shot approach to generate the video as a sequence of shots. Here is a snippet of the instruction we use for generating the video:

An snippet of the Veo 3.1 instruction $\mathcal { P } _ { \mathrm { v i d e o } }$   
Style: Cinematic 4K, underwater photography, realistic   
Pacific Bottlenose dolphins, deep ocean blue color   
grade, naturalistic lighting, fluid camera movements,   
high-fidelity water physics.   
Prompt: Generate A pod of dolphins is seen from a   
side-angle tracking shot, swimming at maximum speed

through the deep blue ocean. Intense cavitation bubbles trail from their fins. The camera moves rapidly alongside them. The scene ends with the dolphins beginning to slow down.

Transition Note: End shot with decelerating tail beats.

We sequentially prompted the video generator with the scene descriptions from the LLM annotated movie script. The agent rendered the Scene 1 description as an 8-second clip. The model extended the output to 16 seconds using the Scene 2 description. This recursive extension continued through all 8 scene descriptions. The procedure yielded a 57-second video encapsulating all scenes.

Figure 2 shows frames from the generated video. Figure 2a shows a frame from Scene 1 and corresponds to its equilibrium state in Figure 9. This scene runs from t = 0 to t = 3 with “Herd Clustering” node clamped ON and the “Survival Threat” node inactive: dolphins play and rest without a survival threat. Here is the annotation for this scene: A pod of dolphin is resting and playing. Figure 2a depicts this. Figure 2b - 2h similarly capture the corresponding causal virtual world of Scene 2-8 in Figure 9.

Again we can use LLM agents to extract FCMs from text or vice versa [13], [14] and exploit the Bayesian de-chunking structure [15]. This allows AI agents to partially or fully automate the selection of scenes or meta-rules $\mathcal { A } _ { j }  \mathcal { B } _ { j }$

## IV. Conclusions

Sequenced FCMs can combine with agentic control of large video generators to produce causal virtual worlds or videos. Users can define or modify the FCMs or oversee LLMs that generate them from text or from transcribed speech. Users can also modify the virtual world midstream if they impose new causal patterns or if they adapt the FCM’s local causal structure. These techniques scale and favor hardware implementation both because FCM causal propagation involves only simple vector-matrix multiplication of often sparse causal edge matrices and because mixing FCMs always produces a new FCM. Large-scale versions ofer in principle a way to design realistic immersive virtual worlds that allow multiple users to interact.

## References

[1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.

[2] Julie A Dickerson and Bart Kosko. Virtual worlds as fuzzy cognitive maps. Presence: Teleoperators & Virtual Environments, 3(2):173–189, 1994.

[3] Bart Kosko. Fuzzy cognitive maps. International journal of man-machine studies, 24(1):65–75, 1986.

[4] Bart Kosko. Hidden patterns in combined and adaptive knowledge networks. International Journal of Approximate Reasoning, 2(4):377–393, 1988.

[5] Osonde A Osoba and Bart Kosko. Fuzzy cognitive maps of public support for insurgency and terrorism. The Journal of Defense Modeling and Simulation, 14(1):17–32, 2017.

[6] Guy Ziv, Elizabeth Watson, Dylan Young, David C Howard, Shaun T Larcom, and Andrew J Tanentzap. The potential impact of Brexit on the energy, water and food nexus in the uk: A fuzzy cognitive mapping approach. Applied Energy, 210:487–498, 2018.

[7] Michael Glykas. Fuzzy cognitive maps: Advances in theory, methodologies, tools and applications, volume 247. Springer, 2010.

[8] Elpiniki I Papageorgiou. Fuzzy cognitive maps for applied sciences and engineering: from fundamentals to extensions and learning algorithms, volume 54. Springer Science & Business Media, 2013.

[9] Wojciech Stach, Lukasz Kurgan, and Witold Pedrycz. A divide and conquer method for learning large fuzzy cognitive maps. Fuzzy Sets and Systems, 161(19):2515–2532, 2010.

[10] Rod Taber, Ronald R Yager, and Cathy M Helgason. Quantization Efects on the Equilibrium Behavior of Combined Fuzzy Cognitive Maps. International Journal of Intelligent Systems, 22(2):181–202, 2007.

[11] A. Adadi and M. Berrada. Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI). IEEE Access, 6:52138–52160, 2018.

[12] Wojciech Samek. Explainable AI: Interpreting, explaining and visualizing deep learning, volume 11700. Springer Nature, 2019.

[13] Akash Kumar Panda, Olaoluwa Adigun, and Bart Kosko. Causal autoencoder-like generation of feedback fuzzy cognitive maps with an LLM agent. In 2025 International Conference on Machine Learning and Applications (ICMLA-2025), pages 1234–1241. IEEE, 2025.

[14] Akash Kumar Panda, Olaoluwa Adigun, and Bart Kosko. The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with Mixed Large Language Models. In International Conference on Large Language Models (CSCI-2025). Springer, 2025.

[15] Akash Kumar Panda, Olaoluwa Adigun, and Bart Kosko. Agentic Chunking and Bayesian De-chunking of AI Generated Fuzzy Cognitive Maps: A Model of the Thucydides Trap. In IFIP International Conference on Artificial Intelligence Applications and Innovations, pages 17–32. Springer, 2026.