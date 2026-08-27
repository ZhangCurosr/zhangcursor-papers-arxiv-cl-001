# SwarmWorld: Stigmergic technological evolution in societies of language-model agents

Subhadeep Pal<sup>1,2</sup> Fiona Y. Wang<sup>1,3</sup> Markus J. Buehler<sup>1,2,4,5,#</sup>

<sup>1</sup>Laboratory for Atomistic and Molecular Mechanics (LAMM), <sup>2</sup>Department of Civil and Environmental Engineering, <sup>3</sup>Department of Biological Engineering, <sup>4</sup>Department of Mechanical Engineering, <sup>5</sup>Center for Computational Science and Engineering, Schwarzman College of Computing, Massachusetts Institute of Technology, Cambridge, MA 02139, USA

<sup>#</sup>Corresponding author: mbuehler@mit.edu

## Abstract

Collective intelligence can emerge when individuals coordinate through a shared environment, allowing local actions to accumulate into durable social organization. Language-model agents ofer a new substrate for this process, yet most multi-agent systems rely on direct conversation, predefined roles, or centralized workflows. It remains unclear whether decentralized agents can build functional technologies and outperform independent search. Here, initially homogeneous LLM agents in SwarmWorld self-organize without assigned roles or recipes into evolving technological societies. Agents explore a spatial environment, process resources, test materials, construct persistent artifacts, and write executable controllers evaluated by a deterministic simulator under unseen disturbances after the agents are removed. SwarmWorld splits cognition from consequence: agents propose architectures and controllers within fixed action and material schemas, while the simulated world determines function. Shared societies develop broader, more resilient technological portfolios than a strong best-of-N isolated-search baseline, although isolated search remains competitive for the strongest artifact. Agents diferentiate into exploration, construction, maintenance, and coordination behaviors, transitioning as the world matures. Technologies accumulate through collaborative construction, executable inheritance, and persistent agent-artifact networks, with most reuse beginning through physical observation rather than communication. Explici cultural mechanisms amplify collaboration and organization, but functional benefits depend on outcome and timescale. Physical stigmergy alone supports capable societies, while interaction drives persistent technological ecologies rather than universally superior individual inventions.

Keywords language-model agents · multi-agent systems · stigmergy · cumulative culture · scientific discovery · bio-inspired materials

## 1 Introduction

Collective behavior allows groups to achieve outcomes beyond the reach of isolated individuals, from ant foraging and honeybee nest-site selection to quorum-guided decisions in fish schools <sup>1–5</sup>. Such organization need not require a central coordinator. Local feedback can amplify useful behavior, task allocation can adapt to changing needs, and persistent environmental modifications can coordinate later activity through stigmergy<sup>6–8</sup>. These mechanisms do more than aggregate simultaneous actions: they allow one individual’s activity to alter the information and opportunities available to others. They therefore motivate a central question for artificial collectives: can decentralized agents do more than search in parallel by constructing a shared, cumulative substrate for future action?

Two computational lineages frame this question. Swarm-intelligence methods translate decentralized interaction into search and control: ant colony optimization reinforces useful paths, particle swarm optimization shares individual and population experience, and artificial bee colony methods balance exploration and exploitation <sup>9–11</sup>. The same principles extend to swarm robotics and materials search, including particle-swarm crystal-structure prediction in CALYPSO <sup>12,13</sup>. A parallel lineage made the evolving world itself the object of computation. Cybernetics and system dynamics emphasized feedback within stateful systems <sup>14,15</sup>, while cellular automata showed how repeated local rules generate persistent global structure <sup>16,17</sup>. Interactive simulations such as Hamurabi and SimCity made intervention in evolving systems an object of experimentation, whereas Boids and Sugarscape populated simulated environments with autonomous agents whose local behavior produced collective organization <sup>18–21</sup>. Recent transformer models of cellular dynamics further connect learned sequence modeling to this local-rule tradition <sup>22,23</sup>. These lineages supply complementary ingredients—distributed coordination on one hand and persistent, consequential worlds on the other—but generally retain fixed representations, hand-designed policies, or predefined objectives. LLM-guided robotic swarms begin to relax those restrictions by allowing agents to reason and communicate within collective tasks <sup>24–26</sup>.

Multi-agent reinforcement learning made some forms of coordination, competition, and tool use learnable rather than prescribed<sup>27,28</sup>; language-model agents add general reasoning, memory, communication, and program synthesis to this foundation. Generative Agents showed how memory, reflection, and planning can support emergent social behavior in a simulated town, while Project Sid reported specialization, collective rule formation, and cultural transmission at larger scale <sup>29,30</sup>. GovSim isolates cooperation over shared resources, and AgentSociety extends generative-agent simulation beyond 10,000 agents <sup>31,32</sup>. TerraLingua moves closer to a persistent LLM ecology: agents create and revise textual artifacts that outlive their authors and acquire branching cultural lineages <sup>33</sup>. Other systems make capabilities themselves cumulative. Voyager stores reusable executable skills, GenSwarm generates coordinated multi-robot policies, and DiscoveryWorld grounds hypothesis formation and experimentation in interactive scientific tasks <sup>34–36</sup>. Yet scale, memory, and communication do not make collective advantage automatic. Local perception still challenges coordination, while rapid consensus can suppress exploration even as individual reasoning improves <sup>37,38</sup>.

A parallel body of work applies LLMs and multi-agent systems to scientific discovery by coupling reasoning to knowledge representations, code, simulation, and analysis tools <sup>39–41</sup>. ProtAgents assigns specialized agents to protein design, structure prediction, simulation, and retrieval; SciAgents links specialized reasoning through knowledge graphs; and AtomAgents connects multimodal agents to atomistic simulation for alloy design <sup>42–44</sup>. Sparks and SparksMatter organize hypothesis generation, computational testing, and iterative refinement, whereas CASCADE emphasizes the acquisition and exchange of reusable scientific skills <sup>45–47</sup>. Decentralized protein-design swarms, PharmaSwarm, and MusicSwarm explore collective organization in sequence search, drug discovery, and long-horizon creative production <sup>48–50</sup>; virtual laboratories and ScienceClaw × Infinite extend the unit of analysis toward communities that compete, transmit research lineages, or build on persistent scientific artifacts <sup>51,52</sup>. These systems establish substantial autonomy; the protein-design swarm additionally reports experimental validation of generated designs. Most nevertheless prescribe roles, workflows, tool access, evaluation structures, or bounded interaction patterns rather than asking what technological organization emerges among initially equivalent agents.

What remains missing is a controlled test that combines four properties: initially equivalent agents, a shared world that retains their modifications, executable technologies whose function is evaluated independently of agent claims, and a matched isolated-search baseline that measures whether interaction changes capability rather than merely increasing the number of samples. TerraLingua provides a close comparison for persistent cultural accumulation, but its principal artifacts are textual and its endpoints concern ecological surviva and interpreted artifact properties. The distinction is consequential: persistent information can demonstrate cultural accumulation, whereas persistent technology can also be subjected to an external functional assay. Technological descent should therefore have two independently observable consequences—later agents can inherit and modify earlier constructions, and the resulting artifacts can succeed or fail under dynamics that do not depend on an LLM’s description of their value. Scientific-agent systems provide functional evaluation, but commonly within designed decompositions or workflows. The unresolved question is not simply whether LLM agents can communicate, specialize, or create artifacts, but whether their interaction can produce a functionally stronger technological ecology than the same computational population can discover independently.

![](images/051b995cd24d265626c9753087c3c640101fb06f1ef4c417d64901ede77e6418.jpg)  
Figure 1: Conceptual lineages leading to SwarmWorld; connectors indicate conceptual convergence rather than direct descent between every adjacent item. (A) Biological collectives and swarm-computing methods contribute decentralized coordination, feedback, and stigmergy. (B) Local-rule models and artificial societies contribute persistent worlds in which local actions alter shared state and generate population-level organization. (C) Distributed-agent and LLM systems contribute explicit agent state, memory-guided planning embodiment, communication, and reusable executable skills. (D) SwarmWorld combines these elements in a materially constrained multi-agent world where agents build persistent artifacts and inherit or modify their programs. The resulting technological ecology can be evaluated after the agents are removed. Image credits. Ant photograph by Kiran Vati K, CC BY 4.0, cropped and color-adjusted. Boids still from a video by Andrew Magill, CC BY 2.0, with a frame extracted, cropped, and color-adjusted. Game of Life glider-gun diagram by Bryan.burgers, public domain, placed on a grid background. Micropolis screenshot based on the original software by Wil Wright and Maxis Software/Electronic Arts, the Unix port by Don Hopkins and DUX Software, screenshot by Tomhannen, and edit by bayo; GPLv3 or later with Section 7 additional terms, cropped and color-adjusted. The Sugarscape view, method schematics, and SwarmWorld rendering were created by the authors.

To address this gap, we introduce SwarmWorld, a controlled environment that combines initially homogeneous LLM agents with a persistent, materially constrained world (Figure 1). Without assigned roles, predefined recipes, or a technology catalog, agents explore locally, process source–sink-accounted resources, test materials, construct persistent artifacts, and author executable controllers. This architecture enforces a proposal– consequence separation: ideas, messages, and designs are agent-authored claims, whereas the deterministic simulator alone determines what can be built and what functions. Every technology is spatially situated, reads and acts on local world state, and continues to execute between agent decisions. Because later agents can encounter, inherit, and modify these constructions, successful work becomes part of the environment rather than remaining only in a transcript. Because the agents are removed during held-out evaluation, artifact performance can also be measured independently of its creator’s interpretation.

We isolate communication, cross-agent program inheritance, and physical stigmergy through controlled ablations, and compare every society with an endpoint-wise best-of-N envelope of matched isolated agents receiving the same scheduled decision opportunities. This makes swarm advantage falsifiable: interaction must improve capability beyond what the same computational population achieves through independent search. Across populations of 50–200 agents and complementary long-horizon experiments, shared worlds produced broader and more resilient technological portfolios and self-organized diferentiation between exploratory and technology-centered behavior. Explicit culture generated collaboration and executable inheritance but did not improve every endpoint, while isolated search could still retain the strongest individual artifact. The contribution is therefore a bounded swarm advantage: interaction chiefly supports the accumulation of a diverse, persistent technological ecology, not universally superior individual inventions.

## 2 Results and Discussion

SwarmWorld provides a setting in which collective intelligence can be evaluated through multiple channels including agent behavior, communication and the technologies that a population leaves behind (in a shared world). This design allows us to ask how interaction changes the discovery, accumulation, inheritance, and functional robustness of technology, and whether these efects can be distinguished from the gains expected from parallel independent search. The results trace this process from controlled interaction mechanisms and emergent behavioral diferentiation to executable technological lineages, difusion through persistent artifacts, and the long-horizon organization and robustness of the resulting agent-artifact ecology.

## 2.1 Research Design

To test these questions, we performed two paired studies that provide complementary evidence (for a glossary of key terms, see Section S1). The population-scaling study ran four mechanism-resolved conditions for 800 ticks at $N = 5 0$ , 100, and 200, with four matched world seeds per cell and eight held-out disturbance schedules. The long-horizon study followed N = 100 societies for 3,200 ticks under full culture, no explicit culture, and an endpoint-wise best-of-100 independent-search envelope. Frozen states were evaluated at ticks 400, 800, 1,600, 2,400, and 3,200. The world seed is the unit of replication throughout; agent trajectories, artifacts, time samples, and disturbance schedules are nested observations. With four paired seeds, the analysis emphasizes efect sizes, paired consistency, and mechanisms rather than asymptotic population-level inference.

Notably, the main result is not a universal swarm advantage. Shared physical worlds consistently produced broader and more resilient portfolios than isolated search, and explicit culture generated measurable collaboration, cross-agent code descent, network densification, and behavioral reorganization. However, the isolated envelope could retain the strongest single artifact, no-explicit-culture societies sometimes outperformed full culture, and the apparent benefit of culture depended on both time and endpoint. The figures therefore connect the simulation interface to behavior, recorded technological lineage, held-out function, and structural robustness.

## 2.2 A controlled shared world separates physical, cultural, and independent search mechanisms

The simulation couples open-ended language-model decisions to a deterministic physical substrate (Figure 2). A complementary three-dimensional rendering of a representative world (seed 3202) illustrates the spatial separation of resource biomes, processing foundries, agents, and persistent agent-built artifacts (Figure S3). Each agent receives only a local observation and retrieved memory, then emits a schema-constrained plan whose individual actions are checked and resolved transactionally. Movement, metabolism, fields, material transport, disturbances, and installed artifact programs continue on every world tick, including ticks without a model call. This separation is important: the model supplies design and action choices, while the simulator determines whether those choices are legal and what consequences they have. Every SwarmWorld artifact is spatially situated: it occupies coordinates, reads only local sensors, and influences agents and fields only where it stands. Persistent artifacts can therefore become part of the environment encountered by later agents rather than remaining text in a shared transcript.

The four experimental conditions remove mechanisms without changing the scientific task (Figure 3). Ful culture provides a shared world, explicit messages and records, cross-agent executable-program inheritance, and artifact-mediated stigmergy. No communication removes direct communication and publication-dependent composition but retains shared artifacts and program inheritance. No explicit culture additionally removes cross-agent program forking and measured skill inheritance, leaving only physical stigmergy. Independent search replaces the shared society with N isolated one-agent worlds and reports an endpoint-wise best-of-N envelope. This is a deliberately strong control because its winner can difer across endpoints and checkpoints. The held-out assay further separates discovery from evaluation: agents are removed, the frozen technological state is cloned into eight paired unseen disturbances, and only deterministic physics and installed programs continue.

Population scaling produced a mechanism-dependent rather than monotonic result (Figure 4). Discoveryfrontier AUC generally increased with population, but the condition ranking changed with N. At N = 50, full culture and no communication trailed the independent envelope on discovery AUC; at $N = 1 0 0$ , all three shared-world conditions exceeded it; at $N = 2 0 0$ , no explicit culture produced the largest paired discovery gain, +0.069. The more robust endpoints were more consistent. Held-out resilience exceeded the isolated envelope for nearly every shared-world cell, portfolio resilience showed positive paired efects throughout, and validated inventions increased strongly, reaching a mean paired gain of six inventions for no explicit culture at $N = 2 0 0$ . We reserve the term “invention” for artifacts that clear the full validation gate of tested materials, a complete design, an installed agent-authored program, threshold performance, and behavioral novelty (Section S1); an invention is therefore a validated, functioning, situated technology rather than a proposal alone. Thus, access to a shared physical substrate was broadly beneficial, while adding explicit cultural machinery did not uniformly improve short-horizon performance.

A  
B  
![](images/01d41af192dc141d2d6bfd7044b74aa3a9a4e5c510a867cb2d191bd644d771b0.jpg)  
Figure 2: Simulated world and agent-environment algorithm. (A) Authoritative state at tick 600 from a representative N = 200 society. Numbered callouts identify language-model agents, persistent agent-built artifacts, fixed processing stations, resource biomes, an active disturbance field, and tidal resources. The map is the state on which actions are resolved, not a decorative visualization. Agent can move through the terrain, gather and transform matter, inspect or operate artifacts, and encounter local environmental changes produced by both disturbances and other agents. (B) One staggered macroturn begins with a local observation and retrieved private memory, continues through language-model deliberation and a strict structured plan, and ends with transactional validation of actions, targets, permissions, and material preconditions. The deterministic world tick then advances motion, metabolism, resources, fields, disturbances, and every installed artifact program. Resulting events enter agent memory and the shared substrate before the next scheduled observation. The agents decide what to try, but the world decides what actually happens; durable artifacts let one agent’s successful work become part of another agent’s future environment.

## 2.3 Unassigned agents diferentiate and create executable technological culture

We find that behavioral diferentiation arose without role prompts or condition labels (Figure 5). A twocluster model fit to 15 robust-scaled features separated artifact-centered work from mobile exploration. The first phenotype combined artifact proximity, artifact-bound motion, construction, control, and cultural coordination; the second retained broader movement and lower artifact engagement. At N = 200, the artifact-centered fraction averaged approximately 27% under full culture, 20% without explicit culture, and 17% without communication. These fractions are not task assignments. They are post hoc descriptions of complete trajectories, and the between-condition comparison uses seed-level proportions rather than treating agents as independent replicates.

The cultural record also entered physical construction and executable code (Figure 6). Under full culture, 67%, 76%, and 56% of artifacts at N = 50, 100, and 200, respectively, recorded contributions from more than one agent, substantially exceeding the corresponding ablations. Cross-agent program forking remained common whenever program inheritance was available, including the no-communication condition; it was exactly absent when the mechanism was disabled. The focused lineage around Adaptive Chitin Maintenance shows several independently authored ancestor programs converging on a six-author focal program, followed by multiple descendants with recorded instruction edits. These are content-addressed forks and installations, not semantic similarity between descriptions.

![](images/25fbec4b08e0b1416c2702f0d6848e4e88e9fd2ae8246123616c74e4c263d278.jpg)  
Figure 3: Mechanism-resolved conditions and the agent-free held-out resilience assay. (A) Rows define the four interventions and columns identify the mechanisms available in each. A filled circle means the mechanism is present. Full culture combines a shared physical world, explicit cultural actions, cross-agent program inheritance, and artifact stigmergy. No communication removes nearby messages, public records, teaching, trading, task claims, and publication-dependent composition while preserving the shared world and executable inheritance. No explicit culture also removes cross-agent program forking and measured skill inheritance, leaving agents able to coordinate only through persistent artifacts and environmental changes. Independent search consists of N isolated one-agent worlds and therefore has none of the three collective mechanisms. (B) At a discovery checkpoint, the complete state is frozen into eight exact clones. Each clone receives a paired unseen schedule of contamination, drought, and storm, with new centers, timings, and orderings. Agents take no actions during evaluation, but the world physics and installed artifact programs continue. Balanced service coverage Q<sub>s</sub>(t) is integrated over assay time and averaged across schedules to produce held-out resilience. The assay asks whether the technology left behind can protect the habitat under new stresses after its inventors have gone.

The constructed portfolio was diverse in recorded feedstocks and simulator-defined functions (Figure 7). The 16 displayed technologies are ranked and deduplicated representatives drawn across all four conditions rather than templates supplied in the prompt. Their recorded lifetime-peak simulator scores range from 0.790 to 0.347 and span chitin exchange lattices, mycelial mineral veils, tidal panels, cuticle-like membranes, repair structures, catalyst networks, cellulose trellises, and kelp-shell composites. The accompanying mechanism renderings are generated from the recorded geometry, composition, recipe, and controller. They therefore visualize the agents’ proposed technological identity, but they are not literal meshes from the simulator and should not be interpreted as experimental validation of real material performance.

## 2.4 Movement, world modification, and provenance connect search to invention

Matched spatial trajectories show that the shared-world mechanisms changed where agents worked, not merely how far they traveled (Figure 8). In the representative matched seed-3202 episodes shown, mean path length remained nearly constant at 36 to 37 cells across the three N = 200 conditions. Artifact-contact AUC was 0.31 under full culture, compared with 0.14 without explicit culture and 0.11 without communication, and full-culture agents visited more regions on average. For this representative seed, the diference therefore reflects how movement was organized around constructed technology, not a trivial increase in total locomotion.

Event-linked trajectories reveal what agents were doing along those paths (Figure 9). Observations and tests were distributed broadly, whereas construction, program installation, and repeated artifact interaction localized around a smaller number of shared sites. Explicit interaction markers appear only where the condition permits them. Artifact marker size records the number of distinct participants, providing a direct visual bridge between spatial hubs and multi-agent work. The resulting pattern is not a swarm motion

![](images/f59f3e4263b20f7a759d2c8f842f9cfc047f64175fd014a4b81d4a8d21841948.jpg)

![](images/c94c63b1b989462199b43335a53856e7f786fc322788dbfea8273276e4d6baf6.jpg)

C  
![](images/52f5436cefc4a16cb194e01206254abf34faf40cc480eb9905f824666c6c4a36.jpg)

D  
![](images/64a0e7f8b35b933371287e03fdebbb0a0685a4669998259c6cdba1548be0ee0b.jpg)

![](images/101c7ee9b6b237cb0a43cedfce74cb04b82297e59267d63d8374b0ee273a81f1.jpg)  
Full culture No explicit culture No communication Independent search

Figure 4: Population scaling and mechanism-resolved collective efects over 800 discovery ticks. Points and heavy lines in panels A-D are means across four matched world seeds; error bars are paired seed-bootstrap 95% intervals. (A) Discovery-frontier AUC integrates the running best measured artifact performance, rewarding discoveries that occur early and remain on the frontier. (B) Held-out resilience AUC measures agent-free service coverage under eight unseen disturbance schedules. (C) Portfolio resilience quantifies the breadth and redundancy of functional service across the final artifact collection. (D) Validated inventions count technologies that pass the simulator’s objective validation criteria. (E) Each cell is a shared-world condition minus the paired endpoint-wise independent-search envelope. Printed values are raw paired efects, asterisks mark intervals excluding zero, and color is normalized within each outcome row because the four outcomes have diferent units. The shared world yields the most consistent gains in held-out and portfolio resilience and in invention count, whereas discovery AUC depends on condition and scale. Societies become better at building useful collections of technologies, but more communication does not automatically make them better at finding the single fastest early winner.

![](images/c25cb8719556d0b9ee0c789a0f8035bf43165082cd28217585aeabdc40627b8a.jpg)

![](images/27afc32f189ff7b3145d06df9d65ec5dd19bb7c0ca6a90ab17f1268c673d2647.jpg)

![](images/cef6313a0f261d14b16bbce3eddedbf2ec7c80ed6d32d20f65874657bd81274a.jpg)

![](images/aaeaa7dafe41f61b9a550b22c70b1014839a0423502519560f7780d1f4d03ffb.jpg)  
Figure 5: Data-driven behavioral phenotypes discovered in the 800-tick scaling study. Each point represents one complete agent trajectory, and the two-group k-means classification is learned from 15 robust-scaled behavioral features without access to condition, population, seed, identity, or an assigned role. (A) PCA gives an auditable linear projection. The numbered arrows are the six largest loading vectors and point toward increasing social interaction, cultural coordination, technology work, artifact proximity, construction/control, and net displacement. Clustering is performed in the full feature space rather than in this two-dimensional display. (B) Cluster-average standardized signatures identify C1 as artifact-centered work and C2 as mobile exploration; red indicates above-corpus values and blue indicates below-corpus values. (C) Stacked bars show the mean $N = 2 0 0$ composition, with four seed fractions overlaid. (D) The artifact-centered fraction is followed across N = 50, 100, and 200 for each shared-world condition. Initially identical agents divide into a smaller group that stays near and works on technology and a larger group that continues exploring, and explicit culture shifts more agents toward the technology-centered mode.

benchmark in isolation; it is a map of how locomotion carries agents between measurement, fabrication, computation, and social exchange.

One representative society makes the temporal coupling between environmental change and technological accumulation visible (Figure 10). The simulation can additionally be inspected through an interactive interface that couples the evolving spatial world to agent state, society-level dynamics, and the underlying knowledge-lineage graph, enabling individual discoveries to be traced from observations and evidence through programs and downstream artifacts (Figure S5). The no-explicit-culture, N = 200, seed-3202 world grew from no artifacts at tick 0 to 25 near tick 400 and 61 at tick 800. Best-artifact performance rose sharply once construction began, whereas portfolio resilience improved more gradually as additional artifacts accumulated. Spatial entropy declined only modestly, indicating concentration around productive sites without complete collapse into a single location. This trajectory is illustrative rather than inferential; the paired aggregate outcomes are those in Figure 4.

The recorded provenance graph identifies how evidence, programs, and precursor artifacts were reused (Figure 11). The society-scale view contains multi-step paths from observations and authored insights through executable programs to eight high-performing artifacts. The focused ancestry of AdaptiveChitinExchangeScaffold combines several agents, multiple precursor technologies, and inherited programs. Downstream-reach ranking identifies a small set of reusable knowledge hubs, led by the passive material system, rather than only prolific message authors. Because edges are generated from recorded authorship, construction, installation, fork, and causal-parent events, the graph supports an attributable lineage claim; it does not infer causality from embedding similarity.

A  
![](images/1114a0c44685c709379438683a4bc2b57ea4eb9a9244e658a7ee0c3e157583ac.jpg)

![](images/7e943f665556158c9a907d56b16eafa7d64552be29abb203485d67344f6debb8.jpg)  
Full culture No explicit culture No communication

C  
![](images/9d0c114fa431c600aeadf7a56361c7c87083baf54e4e7378e31f3933bac99d7c.jpg)  
Figure 6: Executable technological culture. Seed means and 95% seed-bootstrap intervals are shown across population size. (A) Fraction of constructed artifacts with recorded contributions from multiple agents. Full culture produces the largest collaborative fraction at every population, while appreciable collaboration without explicit culture demonstrates that agents can still meet around and modify persistent objects. (B) Fraction of eligible program forks whose child author difers from the parent author. Cross-agent descent grows with population under full culture and remains possible without direct communication because executable inheritance is still available in that intervention; it is zero by construction when explicit culture and cross-agent inheritance are disabled. (C) Focused content-addressed program lineage centered on Adaptive Chitin Maintenance. Gray boxes are ancestor programs, the green box is the focal program, and purple boxes are descendants. Each node records first tick, number of authors, and installations; each arrow records fork tick, author, and instruction-level edit count. Agents did not merely discuss ideas: they reused, edited, installed, and propagated one another’s executable environmental controllers, producing attributable technological descent.

The material-process map separates three layers that are often conflated in generative design claims (Figure 12). Construction feedstocks specify what matter was consumed at build time; ordered fabrication nodes specify how those inputs were processed; operational fluxes specify what the resulting artifact actually moved or consumed during the authoritative simulation. Across the 16 exemplars, agents used distinct combinations of fungal, mineral, catalyst, chitin, cellulose, lignin, kelp, and shell resources, with recipes that included washing, grinding, fermenting, mineralizing, aligning, weaving, pressing, and drying. Realized operation most often involved water capture, contamination removal, and embodied reserve consumption. Available but unrealized pathways are shown separately from unavailable ones, preventing design claims from being mistaken for executed function.

## 2.5 Long horizons expose metric-specific cultural benefits and tradeofs

Extending discovery to 3,200 ticks did not reveal one universal cultural crossover (Figure 13). Full culture overtook no explicit culture in mean best-artifact performance by tick 800, and crossed in portfolio resilience and cumulative artifact count near tick 1,600. It never overtook in validated invention count. Held-out

A

E  
M  
H  
![](images/7770feda00e662b08efec027c7dc5468daea53b79a2746cec79b2141dd1a2a6d.jpg)

B  
![](images/9954fdeb2ba428c58f6ae0d028fa848b46eae4b5aa9fd2017ff394c9fbb32398.jpg)

C  
![](images/07783d2d22942e60e007f803d15dbfde2cfbe29f7ebc506f64f66932ddd44122.jpg)

D  
![](images/eb496679e5a906ef058ef92579d6b59f4cdfe597a0c116e45e85f8e222cbe0f0.jpg)

## Three Layer Connected Chitin Exchange Lattice

Flow: CHITIN -> WASH - GRIND - ALKALINE   
TREAT - DRY -> intermediate porosity   
chitin sheet   
Function: Provide a durable porous habitat   
interface that buf ers moisture, supports   
structural continuity, and responds to   
contamination and damage   
Control: contamination/moisture/health ->   
remediate/regulate pores/hea   
Composition: chitin 72%, protein 10%,   
water 8%

## Mycelial Mineral Spring Veil

Flow: FUNGUS + MINERAL -> WASH - GRIND -   
FERMENT - MINERALIZE - ALIGN - PRESS ->   
porous fungus rich mineral composite   
scaf old   
Function: Provide a porous local scaf old   
that may support structural continuity   
while adaptively at empting healing and   
contamination reduction   
Control: contamination/health/nutrients ->   
heal/remediate/grow/signal   
Composition: mineral 35%, water 18%,   
chitin 16%

## Fermented Tidal Chitin Panel

## Breathing Cuticle Veil

Flow: CHITIN -> WASH - GRIND - ALIGN - DRY   
-> modular aligned porous chitin exchang   
interface   
Function: Provide a replaceable porous   
habitat interface that moderates   
environmental exchange while supporting   
limited damage recovery   
Control: moisturelcontamination/health ->   
regulate pores/remediate/heal   
Composition: chitin 72%, protein 10%,   
water 8%   
Flow: CHITIN -> WASH - GRIND - FERMENT -   
ALIGN - PRESS -> fermented aligned and   
pressed porous chitin habitat panel   
Function: Provide a tougher, moderately   
permeable habitat panel whose persistence   
and healing can be compared directly with   
the existing chitin lat ice   
Control:   
regulate pores/grow/heal/remediate/col ect   
water/signal   
Composition: chitin 72%, protein 10%,   
water 8%

F  
![](images/69692d03f8cfabf129b0cac8e3cb582c92d3e2ec0bfb038c21d251204028e485.jpg)

## Myco Repair Exchange Pane

![](images/aee7ae43171c7a6f72b6e4836dd3c0c7721e2ff4ed0402c513880a1a0ff89b3e.jpg)  
G

Flow: FUNGUS -> WASH - GRIND - FERMENT -   
PRESS - DRY -> open cohesive fungus matrix   
Function: A low-load porous habitat buf e   
intended to reduce contamination and   
support localized healing while preserving   
exchange pathways   
Control:   
contamination/moisture/healing/nutrients   
-> remediate/regulate pores/heal/signa   
Composition: chitin 24%, water 24%,   
protein 22%

## Replicated Tri-Phase Healing Exchange Lattice

![](images/58f37081186411767d55f886a62d0b7cd8f1efe5606081531a9c5d0ce971ff9e.jpg)

Flow: FUNGUS + MINERAL + CATALYST -> WASH   
- GRIND - FERMENT - MINERALIZE - PRESS ->   
reduced scale porous habitat lat ice   
Futedns Crevider au raui habitt-sunport   
element with measurable healing potentia   
and exchange surfaces under moist mineral   
spring conditions   
Control:   
moisture/contamination/nutrients/health ->   
regulate pores/heal/remediate/grow/signal   
Composition: mineral 48%, protein 21%,   
water 12%

J  
![](images/40a28e2bf25cfd04f90025a56b148f23e5ed7d8d3af26395101e84d0fd028194.jpg)

![](images/86f185c80f6e08b7692818b8482326cec2a3a7b15f1781cb854193986b62677a.jpg)

## Chitin Healing Vein Partition

Flow: CHITIN -> GRIND - PRESS - DRY ->   
pressed chitin channel shel artifact   
matrix   
partition with shal ow pathways and   
limited damage recovery under the observed   
moist test-field conditions   
Control:   
contamination/health/moisture/nutrients ->   
remediate/heal/regulate pores   
Composition: chitin 72%, protein 10%,   
water 8%

K  
![](images/3864b59012f76db8cb19d2e9a043e8cd39ef6215a238c7a50e627e3df073423b.jpg)

## Myco Mend Lattice

Flow: FUNGUS -> WASH - GRIND - FERMENT - DRY -> cohesive fungus derived modular Function: Provide a permeable, modestly self-repair-supporting habitat interface that can buf er local moisture while remaining modular for observation Control: moisture/contamination/nutrients -> regulate pores/remediate/heal Composition: chitin 24%, water 24%, protein 22%

0.516

![](images/c5241d9b164d8ed2da8b3fd71169e4766eaa14f21d422b369c49e921d6303693.jpg)

0.414

![](images/1103418ca8b303eb5a701e88aed117274791e48c4a3ab6cef3b29a4b1312f9f4.jpg)

## Myco Porous Buffer Sheet

Flow: FUNGUS -> WASH - ALIGN - DRY -> thin porous fungus derived habitat sheet Function: Provide a persistent porous interface that may moderate local moisture and contamination while retaining airflow Control: contamination/moisture -> remediate/regulate pores/heal Composition: chitin 24%, water 24%, protein 22%

![](images/8dbb3ffc2cbfce13283596b6493fbf14c5d9c2270447ca6114b6945ad3e1d9fe.jpg)

## Stonecap Mycelial Vent Panel

## Catalyst Springveil Reactive Lattice

0.389

Flow: MINERAL + CATALYST -> WASH - GRIND   
MINERALIZE - COAT - DRY -> persistent   
dried porous catalyst enriched mineral   
composite lat ice   
Function: Provide porous structural   
support with catalyst-mediated remediation   
local y favorable conditions   
Control:   
contamination/health/moisture/nutrients ->   
remediate/heal/regulate pores/col ect   
water/grow/signal   
Composition: mineral 69%, protein 20%,   
water 6%

N  
![](images/deddb86036fbc586c3caf1f70be9578d436cc2f6cbd0e7dafe1b3d26711e5517.jpg)

## Catalyst Venation Habitat Membrane

Flow: CATALYST -> WASH - GRIND - ALIGN -   
WEAVE - DRY -> porous aligned catalyst   
only habitat panel   
Function: Provide a porous, conditional y   
moisture capture and contamination   
response   
Control:   
moisture/contamination/health/nutrients ->   
col ect water/regulate   
pores/remediate/heal/grow   
Composition: mineral 42%, protein 42%,   
water 6%

O  
![](images/df6a6e8bc4b313f81589a67501dbc187ee3d21d40e5ee2feddae3ad32e6e7aed.jpg)  
P

## Cellulose Nutrient-Moisture Exchange Trellis

Flow: CELLULOSE -> WASH - ALIGN - WEAVE -   
PRESS -> modular aligned porous cel ulose   
habitat interface   
Function: Provide a persistent porous   
habitat interface that maintains connected   
exchange, col ects water during dry   
when nutrients are scarce   
Control: moisture/contamination/nutrients   
-> col ect water/remediate/grow/regulate   
pores/signal   
Composition: cel ulose 78%, lignin 12%,   
water 4%   
Flow: FUNGUS + MINERAL -> FERMENT -   
MINERALIZE - ALIGN - DRY -> porous minera   
reinforced fungal composite habitat pane   
Function: Provide a porous, moderately   
reinforced habitat interface that supports   
moisture exchange while resisting   
structural deformation   
pores/remediate/hea   
Composition: mineral 50%, water 15%,   
chitin 12%

0.347

## Hyphae Pulse Interface

![](images/b8d1dabf3e06230cc97b6ec4777a3660a8480553774885b2f89e5560e79c941f.jpg)

Flow: MINERAL + CATALYST -> GRIND -   
MINERALIZE - PRESS - DRY -> modular open   
Function: Provide durable modular habitat   
support while retaining exchange pathway   
for moisture and gases   
Control: contamination/moisture/nutrients   
-> remediate/col ect water/regulate   
pores/grow/heal   
Composition: mineral 77%, protein 14%,   
water 6%   
<sup>Flow:</sup> <sup>FUNGUS</sup> <sup>-></sup> <sup>WASH</sup> <sup>-</sup> <sup>FERMENT</sup> <sup>-</sup> <sup>ALIGN</sup> <sup>-</sup>DRY -> persistent fermented aligned porous   
Function: Buf er local moisture variation   
while supporting damage-responsive habitat   
interfaces   
Control: moisture/health/contamination ->   
regulate pores/heal/remediate/signal   
Composition: chitin 24%, water 24%,   
protein 22%

## Kelp Shell Tideveil Panel

Flow: KELP + SHELL -> WASH - GRIND - ALIGN   
- PRESS - DRY -> Pressed porous kelp shel   
composite habitat pane   
Function: Provide a porous habitat   
interface whose openings respond t   
combined local moisture and nutrient   
availability while maintaining fiber  
reinforced shel -supported structure   
Control: moisture/nutrients -> regulate   
pores/heal   
Composition: mineral 43%, water 23%   
cel ulose 18%

Full culture No explicit culture No communication Independent search

Figure 7: Sixteen ranked, deduplicated agent-invented technologies from the 800-tick study. The colored rule above each panel identifies its experimental condition and the number at upper right is lifetime-peak simulator performance. Panels A-D show the Three Layer Connected Chitin Exchange Lattice, Mycelial Mineral Spring Veil, Fermented Tidal Chitin Panel, and Breathing Cuticle Veil. Panels E-H show the Myco Repair Exchange Panel, Replicated Tri-Phase Healing Exchange Lattice, Chitin Healing Vein Partition, and Myco Mend Lattice. Panels I-L show the Myco Porous Bufer Sheet, Catalyst Springveil Reactive Lattice, Catalyst Venation Habitat Membrane, and Cellulose Nutrient-Moisture Exchange Trellis. Panels M-P show the Stonecap Mycelial Vent Panel, Basalt Mycelium Lattice, Hyphae Pulse Interface, and Kelp Shell Tideveil Panel. Each portrait combines a complete form, partial cutaway, and dominant component specimens inferred from the recorded architecture, geometry, composition, process sequence, and controller. Text below each image reports the input-process-output flow, agent-authored functional claim, sensor-to-actuator control signature, and leading material fractions. The society explored multiple technological families rather than converging on one prewritten object; the images communicate the proposed mechanisms, while the quantitative score comes only from the simulator. See Section S2.13 for details on how these representative technologies were selected. 10

![](images/833ec8b8acc1b97dd14079bf44a6373aecae647d5854a64a9a7c09faf8ee1015.jpg)  
trajectory start trajectory end active artifact  
Figure 8: Representative $N = 2 0 0$ movement trajectories from matched seed-3202 societies. For readability, 24 deterministically selected complete paths are shown in each panel over the identical terrain. White circles mark trajectory starts, colored circles mark endpoints, and gold diamonds mark artifacts active at tick 800. (A) Full culture produces paths that repeatedly intersect the central artifact ecology; all-agent summaries report mean path length 36.0 cells, 3.3 regions visited, and artifact-contact AUC 0.31. (B) Without explicit culture, mean path length remains 36.0 cells, but region coverage falls to 2.7 and artifact-contact AUC to 0.14. (C) Without communication, mean path length is 37.0 cells, region coverage is 2.6, and artifact-contact AUC is 0.11. The inset values use all 200 agents, not only the displayed paths. The agents travel similar total distances, but full culture makes their movement more likely to intersect the shared technological infrastructure.

A  
![](images/e876ce1ed73e33123a04d0157487f4288f45882273e6bed87dd85d81cedca334.jpg)  
B

C  
![](images/8cbeb94b4a5c7feef3e55182587179f52073f414f838ea0c6453e2faa64d3491.jpg)  
Figure 9: Movement linked to scientific work and interaction in representative $N = 2 0 0$ societies. Eighteen focal agents per condition are selected by round-robin ranking on distance, artifact work, social interaction, and successful actions so that the display spans the observed behavioral repertoire. Segment color gives instantaneous path speed in cells per tick. Blue circles mark observations or tests, gold diamonds mark artifact construction, purple triangles mark program installation, coral stars mark explicit interaction, and black squares mark active artifacts; square area increases with the number of distinct participants. (A) Full culture combines broad observation with dense construction, installation, and explicit social events around shared artifacts. (B) No explicit culture retains observation and physical artifact work but lacks the direct cultural channel. (C) No communication shows the corresponding spatially mediated activity when messages are unavailable. These are not simply tracks of where agents wandered: they show when travel became measurement, building, code deployment, or coordination, and where repeated work turned artifacts into local activity hubs.

A  
![](images/09215594afedc2d1d62c7c9d0cb271ac0af741dde45e8e01e67406e98ba13a2f.jpg)

B  
![](images/831974e73c42f592ee9391c0cc349f0af9e50092f614adb3262fe8933f4faf4c.jpg)

C  
![](images/8e3a948c3ce82ef64db3d937836a1f2b71443e16c0d3b181f074eb467d94727b.jpg)

![](images/58fe2cb52686c9a03cca1f86c3eb63ab81fa2dbd2604cbaae585274976a4e42b.jpg)

![](images/7e9de8e92feb86ef125916f90576cc5f0118800f95b63b2daf36f7d2010d0911.jpg)

![](images/291b05f478bbc3e31da4edf283b09f4e35b04a6901d8bc118ae136b11373bbfe.jpg)  
Figure 10: Within-run evolution of a representative no-explicit-culture society with $N = 2 0 0$ and seed 3202. (A-C) Authoritative world states nearest ticks 0, 400, and 800; the stored snapshots occur at ticks 0, 408, and 800. Agents are black circles, artifacts are gold diamonds, and processing laboratories are open squares. The artifact count increases from 0 to 25 to 61 as the disturbance field and local activity pattern change. (D) Stepwise accumulation of persistent artifacts across the episode. (E) Running best artifact performance and portfolio resilience. The best-object curve rises rapidly after the first construction wave, whereas portfolio resilience grows more slowly as the society adds complementary technologies. (F) Normalized spatial entropy of agent occupancy on a fixed 10 × 10 grid declines from roughly 0.86 to 0.76, indicating moderate concentration rather than complete aggregation. Agents progressively convert an initially empty landscape into a persistent technological habitat; the first strong object appears quickly, but a useful portfolio and stable spatial organization take longer to assemble.

resilience changed sign across checkpoints and was efectively tied at tick 3,200. The temporal result therefore rejects a single amortization threshold at which communication suddenly becomes beneficial for every objective. Culture changes the developmental trajectory, but each endpoint responds on its own timescale.

The final checkpoint clarifies what shared societies and isolated search optimize (Figure 14). Full culture achieved mean portfolio resilience 0.2474 and no explicit culture 0.2365, both above the isolated envelope at 0.1794. The shared worlds produced 5.75 and 7.00 validated inventions, respectively, versus 2.75 in isolated search. No explicit culture also improved held-out resilience to 0.0446 versus 0.0356 for the isolated envelope. Yet the isolated control retained the strongest final single artifact: 0.3488 versus 0.2380 under full culture. Shared worlds therefore created a broader technological ecology, whereas independent parallel search remained competitive for record-setting single-object optimization.

The long-horizon dynamics reveal how explicit culture changes the use of time and space (Figure 15). Full-culture agents traveled less by the endpoint, a mean path length of 98.5 cells versus 120.0 without explicit culture, while regional crowding rose to 0.1298 versus 0.0877. By ticks 2,400 to 3,200, full culture allocated 13.9 percentage points less activity to movement and 20.1 points more to explicit cultural actions; construction/control was 1.3 points higher. At the same time, mean executable lineage depth reached 9.75 and roughly half of eligible forks crossed author boundaries. The deepest representative genealogy contains 12 fork edges. Thus, spatial localization was accompanied by an increasingly deep executable culture rather than simple inactivity.

A second label-blind model provides a complementary characterization of how the cultural intervention was associated with physical behavior (Figure 16). The clustering used only nine movement and artifact-proximity features from all 800 complete trajectories in the eight shared-world episodes; explicit messages, cultural actions, condition labels, and technology-work counts were excluded. One phenotype combined high artifact proximity and artifact-bound movement with shorter, less frequent travel, while the other represented mobile

![](images/177200f1b406f25d200f2c881f0aa9423165e761f3cb101ccda617b99d643d49.jpg)

A  
![](images/7e6f1cf90931cad5313e5e50571eed991c4cd080ae30307792f9c5065b3b5229.jpg)

C  
![](images/ce28e02f795cb0d15a329d2c17288feaaf5f6150699411ea5d2539d3b91e2195.jpg)  
Figure 11: Recorded knowledge lineage from evidence to technology. (A) Society-scale directed ancestry of eight high-performing active artifacts. Circles denote agents, squares denote evidence, small gold nodes denote insights, purple hexagons denote executable programs, and green diamonds denote artifacts. Typed arrows encode authorship or observation, recorded parent references, construction, installation, and program forking; node area increases with betweenness centrality. Labels are selectively placed to preserve readability. (B) Focused ancestry around AdaptiveChitinExchangeScaffold. Multiple agents, evidence records, precursor artifacts, and inherited programs converge on the target artifact; arrow direction is from recorded precursor to downstream record, program, or artifact. (C) Evidence, insight, and program nodes ranked by the number of distinct downstream artifacts they reach. The passive material system has the largest reach, followed by several maintenance and exchange records. Technologies are not isolated model responses: recorded observations and executable ideas become reusable building blocks that pass through several agents and contribute to multiple later artifacts.

<table><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Θ</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>.00</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0.06</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>③</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>©</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>1.09</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>⑤</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.13</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.09</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>©</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>1.0€</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>©</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>L0</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Θ</td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>③</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>e</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>O</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>O</td></tr></table>

Figure 12: Construction and operational matter pathways for the 16 technologies in Figure 7, retained in the same A-P order. The left matrix reports construction feedstocks. Filled circles identify named recipe inputs and circle area increases with input mass; gray open circles mean that the material is absent. The center matrix reports fabrication sequence. Connected numbered circles provide the exact processing order, distinguishing, for example, wash-grind-ferment routes from mineralize-align-weave routes. The right matrix reports realized operational matter flow per 100 active ticks in the authoritative final snapshot. Filled colored circles and printed values encode the realized rate; colored open circles indicate that a controller pathway exists but had zero realized flux; gray open circles indicate that the pathway is unavailable. Water capture transfers local moisture into artifact storage, contamination removal reduces the environmental field, and embodied reserve supports repair, growth, and nutrient release. Construction feedstocks are consumed once at build time and are not automatically replenished during operation. The figure distinguishes what an artifact is made from, how it was fabricated, and what it actually did after construction; only the final columns are evidence of executed simulator function.

![](images/1bd13631779e58089331ee8a84f5b6e6295f093a1af3a98a2073fcb030d61cdc.jpg)

![](images/b5f858ebf41c756a5bd0285dff2d864cefd2fb36f638c77c891094b36213d2d4.jpg)

![](images/0e35fce7d711a9d386b6633dc54fdc8f3cd241349c7b828c6b08ff0813f46977.jpg)

D  
![](images/bc228199d54b533c8c6b93fc7b9400084da0ad54ffffa8a794b95dab6dfaff46.jpg)

![](images/d578c628d6790280dc30631d3fa83fe1f86a72ead05720c734d466a4afd5cacd.jpg)

F  
![](images/1d22da82003071a0dbea6f070b09b8fccccdd67bb9f51ad9a5efcad5bdeb1fe5.jpg)  
Figure 13: Outcome-dependent cultural crossover over 3,200 discovery ticks. Four matched $N = 1 0 0$ seeds are evaluated at five frozen checkpoints. Thin lines are individual world seeds, heavy lines are seed means, and shaded regions are 95% seed-bootstrap intervals. (A) Agent-free held-out resilience for full culture, no explicit culture, and the endpoint-wise best-of-100 independent-search envelope. Both shared worlds generally remain above the isolated envelope, but their ordering changes. (B) Paired full-minus-no-explicit-culture held-out efect; the mean changes sign and ends near zero. (C) Best active artifact performance crosses in favor of full culture by tick 800. (D) Portfolio resilience crosses near tick 1,600 and remains modestly higher under full culture. (E) Cumulative artifact production also crosses near tick 1,600 and ends at means of 277.5 versus 238.5 artifacts. (F) Validated inventions never cross; no explicit culture remains higher at every checkpoint and ends at 7.0 versus 5.75. Explicit culture helps some capabilities after enough time, but there is no single moment after which it improves everything.

A  
![](images/8b83071fec0f5850f151d462f2a30e7a6c9151f2bf0c91f3b0f7f5dcc573faa1.jpg)

B  
![](images/ed50c30c1bf4ac8fa0450c868c6594a8c6cc23e6abdc11cab045b3e4320b2248.jpg)

C  
![](images/a4b9fa73232ce0a7bb2af9bde85708f3e4f1f081f88547a45cf537c7d22d2776.jpg)

D  
![](images/640f93933ba39af9b61fc1166b05082f45ac48059a5413b7326a98179129fc05.jpg)  
Figure 14: Long-horizon endpoint tradeofs at tick 3,200 for four matched $N = 1 0 0$ world seeds. Conditions are full culture, no explicit culture, and an endpoint-wise best-of-100 isolated envelope. Filled points are individual seed outcomes, gray lines preserve within-seed pairing, open diamonds are means, and error bars are 95% seed-bootstrap intervals. (A) Portfolio resilience favors both shared worlds; full culture averages 0.2474, no explicit culture 0.2365, and isolated search 0.1794. (B) Validated inventions likewise favor shared worlds, with means of 5.75, 7.00, and 2.75. (C) Held-out resilience is highest on average without explicit culture, 0.0446, compared with 0.0356 in the isolated envelope; full culture is similar to no explicit culture at the endpoint. (D) Best final artifact reverses the ordering: the isolated envelope averages 0.3488 versus 0.2380 for full culture. Societies win by maintaining several complementary technologies, while isolated agents can still win a contest defined only by the strongest single artifact.

![](images/dea0897a9e17a37a3dd2618da21bf798fd2a385412b91757bd98c5e5267b8745.jpg)

![](images/d1ecac64a9dcaec879072cb2e7536a101322fc2b00f9608e966ffdd43a34bba3.jpg)

![](images/d27b6f2516a83d09a6d5e40f8cc1ddae5b304ad305ede145ecac70ac8e09491b.jpg)

![](images/ee5b8a011bdc46aec61718e70b08b5c772fd3bd42bdf123abac103afa9bd2e21.jpg)  
Figure 15: Cultural self-organization couples spatial hubs to executable code descent. (A) Phase portrait of mean cumulative path length and local regional crowding for full culture and no explicit culture. Light trajectories are the four matched seeds, heavy trajectories are seed means, numbered checkpoints indicate ticks 400, 800, 1,600, 2,400, and 3,200, and arrows point forward in time. Full culture ends with shorter paths and greater crowding, consistent with localized technological hubs. (B) Full-minus-no-explicit-culture allocation of active actions in five time windows, in percentage points. Movement and observation decline relative to the ablation, while explicit cultural activity and construction/control increase. (C) Mean maximum executable-program fork depth rises from 3.75 at tick 400 to 9.75 at tick 3,200, while approximately half of eligible forks remain cross-agent. (D) Exact deepest content-addressed lineage from full culture, seed 3301. Green nodes form the 12-edge longest path, purple nodes are deterministic one-hop branches, coral arrows are cross-agent forks, gray arrows are same-author or starter forks, and node area increases with installation count. The society gradually stops roaming as widely, concentrates around shared infrastructure, and builds a many-generation inheritance system for executable technology.

exploration. Full culture placed 52.8% of agents in the artifact-centered phenotype versus 31.0% without explicit culture, a paired diference of 21.8 percentage points with a seed-bootstrap 95% interval from 12.0 to 33.5 points. This provides a descriptive behavioral signature of self-organization that does not depend on labeling utterances as roles.

![](images/4feb635d6ce1de8edf7428f81a397c4abcf885a5d8f561cc10c768e55a4d812c.jpg)

![](images/3609ffe39f1688ccde06240017c1f791fdd95e6127651d4ea43b5caf6a4e6fc5.jpg)

![](images/ff18a2d6f630c07571f27900ae4364949e0c6a0853aafcc204c965a8e8ea4016.jpg)

![](images/fef5b50c86276171e63cfc251f0240e74cc838ef6435859ab697231a84c075a1.jpg)  
Figure 16: Explicit culture reorganizes physical behavior around shared artifacts. Each point in panels A and B is a complete 3,200-tick trajectory from the eight N = 100 shared-world episodes. A two-group k-means model is fit in the original nine-dimensional robust-scaled physical feature space without condition, seed, identity, communication, cultural-action, or technology-work labels. (A) PCA gives an auditable linear display; numbered arrows point toward increasing values of the five strongest loading features. (B) UMAP shows nonlinear neighborhood structure, but its axes have no direct behavioral meaning. (C) Cluster means identify C1 as artifact-centered movement, characterized by greater artifact proximity and artifact-bound motion with shorter and less frequent travel, and C2 as mobile exploration. (D) Thin lines pair the four seeds, filled circles are seed fractions, and open diamonds are means. Full culture increases the C1 fraction from 31.0% to 52.8%, a paired gain of 21.8 percentage points; the 95% seed-bootstrap interval is 12.0 to 33.5 points and the clustering silhouette is 0.472. Explicit culture changes where agents physically spend their lives, not only what they say.

## 2.6 Technological networks become modular, persistent, difusive, and selectively vulnerable

Complete event histories show the society assembling a persistent temporal circuit (Figure 17). In the representative full-culture seed, 389 artifact tracks become linked to 100 agent tracks by observations, recorded parent references, construction, program installation, 3,924 delivered-message recipient edges, and 358 cross-artifact program-descent events. The matched no-explicit-culture world contains 248 artifacts and, by intervention, neither explicit messages nor cross-artifact program descent. Across four seeds, both conditions retained moderate interaction modularity, successive community assignments became increasingly persistent, and more than 95% of artifacts were eventually used by a noncreator. Explicit culture therefore added a denser social and executable layer to an artifact-mediated structure that was already capable of broad cross-agent uptake.

![](images/e1facb0947d819456425463d26bee7e649d41f599d0945e7ec93fc186db15cc5.jpg)

![](images/a82fdf4e6922ba72e483dee5429d6802315c9a8ad6c3a4c3219db9c04df6f839.jpg)

![](images/fa7ceef7236a3fca285ee35105957bb94747761127cd4d55edb5aba44ee60bf9.jpg)

![](images/776a5cbf1310cabddf0560ec39266804df49e7edbeabb3d0cda5b4ee800a803e.jpg)

![](images/6e26f02280211cca0217e39ef91a3e5e015c2028de2b6e8f5a52733202fb732e.jpg)  
Figure 17: A technological society develops persistent temporal circuitry. (A, B) Complete recovered interaction histories for matched N = 100, seed-3301 full-culture and no-explicit-culture societies over 3,200 ticks. Horizontal tracks in the upper band are agents and tracks in the lower band are artifacts; each artifact track starts at construction. Vertical strokes connect entities at the exact event tick. Blue encodes observation, gray-green recorded parent reference, gold construction, purple program installation or executable descent, and coral explicit messaging. Repeated observations and messages are display-binned with exact counts retained; construction and executable events remain at exact ticks. Tracks are ordered by final data-derived community, not assigned role. The full-culture example records 389 artifacts, 3,924 delivered-message recipient edges, and 358 cross-artifact program-descent events; the matched ablation records 248 artifacts and lacks the disabled event types. (C) Interaction modularity remains substantial, indicating local technological neighborhoods. (D) Adjusted mutual information between successive community assignments rises, showing that those neighborhoods acquire memory. (E) The percentage of artifacts touched by a noncreator approaches saturation in both conditions. A dense web of recurring relationships grows over time, and explicit culture adds message and code-inheritance pathways without replacing physica artifact-mediated coordination.

Figure 18 asks a more specific question than whether the interaction network is large: what kind of organization did the agents construct? The analysis begins from the complete typed event history and forms a weighted bipartite graph linking agents to artifacts through observation, recorded parent reference, construction, contribution, programming, repair, and dismantling. Repeated events are compressed logarithmically so that recurrence matters without allowing one frequently sampled relationship to dominate. Panels A and B show deterministic visualization backbones, not the full graphs and not a significance test. For each node, the two strongest physical ties are retained, and a maximum-weight spanning tree preserves the connected structure of every component; the strongest local social-exchange and executable-lineage ties are then added. The matched full-culture world contains 489 participating nodes and 1,156 displayed backbone edges, including 105 agent-agent and 218 artifact-artifact edges. The no-explicit-culture backbone contains 293 participating nodes and 514 displayed edges, all physical by intervention. Thus, the visual diference reflects both more constructed artifacts and the additional social and executable layers created by explicit culture.

Panels C-E then replace descriptive edge selection with a degree-controlled statistical question. Two agents are joined only when they share at least two artifacts and their observed overlap exceeds a hypergeometric null conditioned on both agents’ artifact degrees, after Benjamini–Hochberg correction over all possible pairs <sup>53</sup>. The full-culture seed contains 1,100 validated agent pairs involving 92 agents, compared with 261 pairs involving 45 agents without explicit culture. The displayed projections retain each agent’s four strongest local surprise links, yielding 270 versus 116 visible edges; all validated edges remain in the exported graph. Panel E reverses the projection: two artifacts are joined when they are used by unexpectedly overlapping agent populations. In full culture, 18,563 artifact pairs among 385 artifacts pass the same corrected test, and the display retains 710 strongest local links. These null-corrected projections show that the denser full-culture pattern is not explained only by some agents or artifacts having higher degree.

The remaining panels distinguish expansion from global integration. In panel F, participation coeficient measures how evenly a node’s weighted ties cross communities, whereas within-module z-score measures whether it is unusually central inside its own community. The reference lines at participation 0.62 and within-module z-score 2.5 are heuristic role-cartography thresholds from prior work, not universal classification boundaries<sup>54</sup>. Every displayed node has participation below 0.26, and only four full-culture artifacts exceed the hub reference line. Panel G extends the comparison across four matched seeds and five checkpoints. By tick 3,200, mean NODF nestedness is nearly identical under full culture and no explicit culture, 0.094 and 0.096 after scaling to 0–1, and mean agent participation is also close to zero, 0.0016 and 0.0044. Explicit culture therefore expands the number and variety of statistically supported coordination pathways without producing one centralized or globally mixed society. The emergent architecture is instead a larger mosaic of locally coherent technological neighborhoods connected by a small number of hubs and bridges.

The network changes through both expansion and memory (Figure 19). Event-class heat maps show early repair and dismantling pulses and a later redistribution toward observation, recorded parent use, construction, teaching, and program descent. Community alluvial diagrams reveal substantial early reassignment followed by larger persistent streams. Cumulative unique agent-artifact ties grow superlinearly with the total number of agents plus artifacts, with descriptive log-log exponents of 3.47 and 3.48. The similar exponents do not imply equal network size: by tick 3,200, full culture has a mean 4,031.3 unique ties versus 2,027.3 without explicit culture. Relationship reuse from one interval to the next reaches 0.650 and 0.588, respectively. Explicit culture therefore enlarges the realized relational substrate, while both conditions acquire repeated local interaction patterns.

Technology difusion was faster and broader under full culture, but it did not follow a simple inventor-to-adopter message cascade (Figure 20). Across seeds, 99.3% of full-culture artifacts and 96.9% of no-explicit-culture artifacts were reused by a noncreator. Median time to first reuse among reused artifacts was 5 versus 8 ticks, and mean adoption breadth was 13.53 versus 7.49 noncreator agents. Approximately 95% of first reuse occurred through direct physical observation in both conditions. Direct creator-to-adopter contact exceeded a timestamp-shufled null only weakly at the shortest 25-tick window and fell below parity for longer windows. Culture therefore appears to alter the society-wide informational and physical network, after which agents commonly discover technology through the world rather than receiving it directly from its inventor.

Finally, structural knockout assays distinguish distributed redundancy from hub dependence (Figure S2). Randomly removing half of the agents left 98.3% of full-culture artifacts and 95.2% of no-explicit-culture artifacts connected to at least one surviving agent. In contrast, high-degree removal reduced access to 59.6% and 73.9%, while broker removal reduced it to 62.9% and 68.4%. The largest connected component showed the same ordering. Full culture therefore distributed participation broadly enough to tolerate random dropout, yet concentrated suficient trafic in high-degree and high-betweenness agents to create targeted vulnerabilities. These are topological measurements on the recorded network; they do not demonstrate physical service, adaptation, or recovery after removing agents from a live simulation.

## 2.7 Granular roles recur as dynamic states rather than fixed identities

The complete trajectories support a more granular analysis than the two broad worker-explorer phenotypes (Figure S4). Every shared-world trajectory was divided into nonoverlapping 200-tick windows. This produced 22,400 agent-windows from 20 episodes: twelve N=200, 800-tick societies under full culture, no explicit culture, and no communication, plus eight N=100, 3,200-tick societies under full culture and no explicit culture. The fit used a deterministic episode-balanced sample of 16,000 windows so that every condition-seed world contributed equally. Thirteen movement, spatial-context, task-action, and cultural-interaction features were robust-scaled; condition, study, population, seed, identity, and time were withheld.

Model selection first recovered the familiar two broad modes with silhouette 0.467, then resolved each parent independently into two occupied submodes. Post hoc profiles identified four recurring states: constructor/- operator, artifact-local caretaker, cultural coordinator, and mobile surveyor. This hierarchy is not inferred from the arrows or apparent islands in the PCA display. It is fit in the original 13-dimensional feature space, requires every child to contain at least 5% of its parent, and remains stable when replicate families are

![](images/1ed328938df923ab92c1b0e6ec29acaca4cea2875ea3ad8b4a75d16b8bbb52bc.jpg)

![](images/f97bd7e514492821387adb93cb17da89fbd8a3a4c145c2482701acacab4771bc.jpg)

![](images/346c1180dcaf2a49e0560565ec930c5cdb736f29f4c03d498276955810b8d9f5.jpg)  
Agent-artifact Agent-agent Artifact-artifact Agent Artifact Node color: community (within panel)  
Ful culture No explicit culture Nestedness (NODF/100) Mean participation

Figure 18: Network anatomy of the long-horizon technological society. (A, B) Deterministic visualization backbones of matched $N = 1 0 0 .$ seed-3301 networks accumulated over 3,200 ticks. The underlying graph links agents to artifacts through observation, recorded parent reference, construction, contribution, programming, repair, and dismantling. The display retains each node’s two strongest physical ties and a maximum-weight spanning tree for each component, then adds the strongest local social-exchange and executable-lineage ties. This is a reproducible readability filter, not a significance threshold; complete graphs are retained in GraphML. Full culture produces 489 displayed nodes and 1,156 edges, versus 293 nodes and 514 edges without explicit culture. Circles are agents, diamonds are artifacts, edge color denotes interaction layer, and node area increases with interaction count. Node color denotes community within a panel; colors do not identify matched communities across conditions. Labels beginning with A are stable agent identifiers. (C, D) Degree-conditioned agent coordination projections. An agent pair is validated only when it shares at least two artifacts and its overlap exceeds a hypergeometric null conditioned on both agents’ artifact degrees, after Benjamini–Hochberg correction at $q \leq 0 . 0 5 .$ Full culture yields 1,100 validated pairs among 92 agents, compared with 261 pairs among 45 agents without explicit culture. The display keeps each agent’s four strongest local surprise links, while GraphML retains all validated pairs. (E) The analogous artifact co-use projection joins technologies used by unexpectedly overlapping agent populations, not technologies with merely similar names. Full culture contains 18,563 validated pairs among 385 artifacts; 710 strongest local links are displayed across 15 components. (F) Node-role cartography. Participation coeficient measures cross-community mixing, and within-module z-score measures local centrality. Reference lines at 0.62 and 2.5 are heuristic thresholds from prior work rather than universal classification boundaries. All nodes have participation below 0.26, and only four full-culture artifacts exceed the hub reference line. (G) Replicated temporal comparison across four matched seeds and five checkpoints. Thin lines are seeds and heavy lines are means; solid curves show binary NODF nestedness divided by 100 and dashed curves show mean agent participation. At tick 3,200, mean nestedness is 0.094 under full culture and 0.096 without explicit culture, while mean participation is 0.0016 and 0.0044. Explicit culture creates a larger and more statistically connected technological network, but it does not merge the society into one centralized hierarchy. Both conditions form specialized local neighborhoods, with explicit culture adding more pathways among them.

C  
E  
![](images/4625590193047db410ba848616bf46f7841dca10134e77cfab4b9a3532754849.jpg)

![](images/bd6b8ef9dc24005bbd7d47b705fdf7fa90c9c3083b9196ebada539efb88ebec6.jpg)

D  
![](images/0262f9d60ea9de41db56eefa7eee4d2b16e43ff104ffb506c6c2504386d891db.jpg)

![](images/0438055263c9fd6a488820c1640e23bcca7f8b6d9025ef3aae82d73c938afb62.jpg)

![](images/d1cb9171ea4e15e4ddce4b884baabddef6fb4347ff817bc642fd5dc1837e4fd0.jpg)  
Figure 19: Technological interaction networks assemble, diferentiate, and retain relationship memory. (A, B) Temporal allocation of eight event classes across 32 nonoverlapping 100-tick intervals under full culture and no explicit culture. Each row is normalized to sum to 100%, so color locates when that event type occurs rather than comparing absolute counts between rows. (C, D) Alluvial maps follow all 100 agents in matched seed 3301 among Louvain communities at ticks 400, 800, 1,600, 2,400, and 3,200. Community identitie are propagated by maximum-overlap matching, and ribbon width is the exact number of transitioning agents. (E) Cumulative unique agent-artifact ties plotted against agents plus constructed artifacts on log-log axes. Thin lines are seeds, heavy lines are means, and the fitted descriptive exponents are $\alpha \overset { - } { = } 3 . 4 \bar { 7 }$ for full culture and $\alpha = 3 . 4 8$ without explicit culture. The exponent describes densification with network size, not growth per unit time. (F) Fraction of interval ties reused in the next interval, rising to means of 0.650 and 0.588 Both societies develop stable neighborhoods, but full culture produces roughly twice as many distinct agent-technology relationships by tick 3,200 and retains more of them from one period to the next.

withheld: mean adjusted Rand index is 0.999 for the broad split and 0.921 and 0.980 for the two conditiona splits.

The resulting sequences reveal behavioral succession. In the 800-tick, N=200 full-culture study, the mean constructor/operator fraction rises from 0 in the first 200-tick window to 0.240 in the last, while the culturalcoordinator fraction falls from 0.814 to 0.560. In the 3,200-tick, N=100 full-culture study, constructors rise from 0.008 to 0.535 and coordinators fall from 0.695 to 0.292. No-explicit-culture societies also develop more constructors, but end at 0.252 while mobile surveyors remain the majority at 0.633. Full culture is also more behaviorally fluid: mean window-to-window switching is 0.270 versus 0.102 in the 800-tick full and no-explicit-culture conditions, and 0.244 versus 0.138 over 3,200 ticks. The same agents can therefore change activities as the technological ecology matures; the roles are not permanent social classes.

The cultural-coordinator state must be interpreted carefully because two of its defining features are disabled by the ablations. An anti-circularity sensitivity therefore removes explicit culture/coordination and social-contact features before fitting. The remaining 11 physical and task features independently support three modes with silhouette 0.551: mobile observation/testing, stationary artifact-proximal construction, and artifact-local movement with material work. Thus, granular physical/task diferentiation is not merely a relabeling of whether communication was allowed. Agent-window assignments remain descriptive, and paired condition

A  
![](images/0cbb3f08078506fd823d2188dcbe98dda6ee0880089ccb3c3dca4c361cf7e9d1.jpg)

B  
![](images/7cc31a5f78397af68c20573ffea864c54905219c18dc7d4e1722d7e2bf916ee0.jpg)

C  
![](images/b9904e2cfb724d2bf31b22e17c207f1781cfcfb3c295189551dbb21896342b53.jpg)

![](images/4f4f92fe987407c51f215c719c7a95383f8f5e37d26fe6b6abf9738babbc05c1.jpg)

E  
![](images/a1f66463586f48980f36707eb226bace5ac84849bc7a761bb2f9c153c7b92ccd.jpg)

F  
![](images/1daa40f6c35ad0c3e1c110b06aa6eea310c88398093a89446e7aed067cb4907a.jpg)  
Full culture No explicit culture Timestamp-shuffled null

Figure 20: Technology difusion through the shared world. (A) Temporal motif tested under full culture. A creator messages, teaches, or trades with the eventual first adopter at $t _ { 1 } ,$ and that agent first reuses the artifact at $t _ { 2 } .$ . The double-headed bracket denotes the elapsed lag $\Delta t = t _ { 2 } - t _ { 1 } ;$ it is not an additional interaction edge. (B) Observed motif frequency divided by a 200-replicate timestamp-shufled null that preserves directed dyads and the global activity schedule. The dashed line is parity. The mean ratio is 1.175 at a 25-tick lag but remains below one at 50 to 400 ticks, so direct transmission is not consistently enriched. (C) Kaplan–Meier fraction of artifacts not yet reused, with unreused artifacts right-censored at tick ${ \mathrm { 3 , 2 0 0 } } ^ { \mathrm { 5 5 } }$ . (D) Complementary cumulative distribution of distinct noncreator adopters. The full-culture intervention produces earlier and broader reuse. (E) First-reuse channels; approximately 95% of adoption begins with physical observation in both conditions, while recorded parent reference and programming are much rarer immediate channels. $\left( \mathrm { F } \right) ^ { - }$ Maximum executable lineage depth grows to a mean 9.75 under full culture and remains zero where cross-agent program descent is disabled. The explicit-culture intervention is associated with faster and broader reuse, but most agents encounter technology through the shared world rather than receiving a direct handof from the inventor.

contrasts use the four simulation seeds. Recurrence across the two studies is not attributed to horizon alone because population, seed family, and decision schedule also change.

## 2.8 The collective architecture transfers to a distinct volcanic materials world

We next asked whether the organizational phenomena observed in BioFoundry depended on its particular material ecology. We therefore replaced the terrain, resources, processing pathways, environmental fields, and functional objectives with a distinct volcanic materials world (AshenRealm), while retaining the same underlying agent–world interaction architecture, persistent-artifact mechanism, and provenance framework. The $7 2 \times 5 4 \mathrm { - c e l l }$ environment spatially separates lava channels, obsidian wastes, sulfur marshes, iron mountains, magma seas, ash plains, a forge enclave, and a proving ground (Figure 21). Agents must locate and transform volcanic feedstocks through metallurgical operations, construct persistent technologies, and maintain function under spatially varying thermal, ash, and seismic hazards. Thus, AshenRealm changes both the accessible design space and the physical consequences of construction while preserving the mechanisms through which agents explore, interact, and modify a shared world.

The resulting societies did not simply reproduce the numerical behavior of BioFoundry. AshenRealm operated in a diferent functional regime, with lower absolute best-artifact performance, discovery-frontier AUC, behavioral novelty, resilience, and invention counts, but retained substantial multi-agent construction, cross-agent program reuse, executable lineage formation, and spatial exploration (Figure 21). Label-blind behavioral analysis likewise recovered artifact-centered work and mobile exploration, together with an additional stationary/low-activity state (Figure S6). The recurrence of these organizational modes under a diferent resource topology and consequence layer suggests that the observed diferentiation is not specific to the original BioFoundry landscape. Because the two worlds encode diferent resources, objectives, and functional scales, however, their raw performance values should not be interpreted as a matched comparison of task dificulty.

The technologies themselves changed with the world. Performance-ranked AshenRealm artifacts include obsidian thermal segments, threshold bafles, fracture-localizing shields, and iron-based loadshields and vented interlocks, with lifetime-peak simulator performance ranging from 0.116 to 0.248 (Figure 22). The top four artifacts are predominantly obsidian based, and employ multi-step processing (see Figure S8). These designs arose across multiple world seeds and combined distinct processing histories, compositions, functional claims, and executable control programs. In a representative trajectory, persistent technology accumulated from no artifacts initially to 2 at tick 400 and 20 at tick 800, while best-artifact and portfolio performance increased in stages and the population became progressively more spatially localized (Figure S7). Together, these results provide a transfer test of the SwarmWorld substrate: the same geospatial and persistent-world architecture can support collective construction, executable inheritance, and behavioral diferentiation under a materially distinct discovery problem.

## 2.9 The transferable world architecture supports sequence-defined protein biomaterials

AshenRealm established that the persistent-world architecture could support collective construction after replacing BioFoundry’s material ecology, processing routes, and hazards. We next asked whether the same architecture could reach a design space organized around protein sequences and biological matrices rather than minerals and metallurgy. We therefore applied the declarative world builder to create Protein Realms, a $7 2 \times 5 4$ molecular landscape in which geography constrains protein–matrix design (Figure 23). This is a catalog-conditioned feasibility world rather than unrestricted de novo sequence generation. The package exposes 12 score-blind, versioned variants spanning collagen-like, silk–elastin-like, resilin-like, and mussel adhesive families, including literature-inspired repeat motifs <sup>56–59</sup>. Agents receive sequence, family, and permitted parent-lineage information, but not the hidden normalized property, matrix-afinity, or stressresistance profiles. These hidden profiles are fixed simulator priors rather than biochemical predictions or new measurements. Agents must instead explore spatially separated amino-acid, membrane, chaperone, cysteine, cellulose, and mineral regions; collect peptide precursor, lipid, chitin, cellulose, mineral, bufer, chaperone, and crosslinking resources; and locate the ribosome array, purification column, folding chamber, matrix loom, sequence archive, and stress assay.

A legal Protein Realms design pairs one catalog sequence with at least one matrix and a peptide fraction between 10 and 60% by mass. Agents select an ordered processing route, fabricate a quarter-scale microbatch, test it at the spatially separate proving ground, and rebuild the exact tested recipe as a persistent installation. The deterministic consequence layer evaluates the protein profile, matrix composition, sequence–matrix afinity, protein fraction, and executed process state rather than the artifact’s name or claimed function.

![](images/f3a878cf895d377e58c39cb8aa9f17eb26db968b91b2e5e036bdc1dec65cad53.jpg)

A  
![](images/47a72fd800270dff75c4addfad9fc8b35bf27b6d0f9cec5fc6963d3025e77641.jpg)

B  
![](images/92b42e014b68aae61a90bce6733ef444abd9c4fc9315fde372a9074a36fc4a7e.jpg)

C  
![](images/3980705ccca764806d583c0933b44e48e5087ad5468745d8d9d12807e8a9e801.jpg)

D  
![](images/bfae07a5bc1f1a10004f7496dfad9481723bfb0d6a8f8bdee22f04fbe6130d4a.jpg)

E  
![](images/fe445b1eab8cd18d705905a20971ac7155b040eb5cba5166e5689e970dd9b622.jpg)

![](images/4956436665c2d7f52f8c071d031c6e648c547f9e93606794e7d8719d0d167c9b.jpg)

![](images/9c19f358ddd2a7a14b004f737cfdf730d6be7e35e5a3578161c2445bee9f0bf4.jpg)

![](images/4d0c3022304365500c5c792d5a670946b80e61383124ee23662e37fa5d3628a5.jpg)

![](images/0106d4f65b947eb0c0d887675a23ee6900a73531832f720b134e70d49574e0ed.jpg)

J  
![](images/d82be76a8c5468727915bf506bbb00964b1a367309488fefa0e8f00b7d280587.jpg)

![](images/20e10f176e92a7a86c9d97bf992158101570e2f1621c8feb96eda20981275747.jpg)

![](images/f711cc64a6ca2aa9e7d59c7e1511cb3bff13f058e227ca8524dd092f0ce14a01.jpg)  
Figure 21: Zoomed-in view of a representative SwarmWorld AshenRealm environment. The rendering shows the 72 × 54-cell world for seed 1703, highlighting the spatial substrate through which agents explore, gather resources, fabricate materials, and construct persistent technologies. Distinct environmental regions include magma sea and resource biomes such as the iron mountain, lava channel, sulfur marsh, and obsidian waste. Cross-world transfer of collective technological behavior from BioFoundry to AshenRealm. Twelve endpoints summarize societies operating in the original BioFoundry and the independently specified volcanic AshenRealm environment: (A) best-artifact performance, (B) discovery-frontier AUC, (C) behavioral novelty, (D) validated inventions, (E) collaborative-artifact fraction, (F) cross-agent program-fork fraction, (G) number of program forks, (H) held-out resilience AUC, (I) portfolio resilience, (J) program-lineage depth, (K) mean distance traveled, and (L) mean number of cells visited. AshenRealm occupies a diferent functional regime, with lower performance and resilience measures, but retains multi-agent construction, executable inheritance, lineage formation, and broad spatial exploration.

![](images/468f4f2f2fb6a7288e6aec6ecf8cd682ef58f9ea4a9bc9f17ca6196986be75b4.jpg)  
Figure 22: Representative agent-invented technologies in AshenRealm with full communication between agents (N=50). Eight performanceranked technologies drawn across AshenRealm world seeds illustrate the material and functional design space generated by the agents. (A–D) The Cinderveil Thermal Segment, Ashbound Obsidian Threshold Bafle, Emberfracture Segment Shield, and Emberglass Load Shield are predominantly obsidian-based thermal and structural systems with lifetime-peak simulator performances of 0.248, 0.214, 0.207, and 0.156, respectively. (E–H) The Embervein Provenance Loadshield, Ashgate Vent Loadshield, Ashbound Meridian Bafle, and Cindervein Interlock Shield are iron-rich load-bearing and heat-management systems with performances of 0.148, 0.135, 0.133, and 0.116. Each panel reports the recorded material-processing pathway, agent-authored functional description, installed controller program, dominant composition, world seed, and construction tick. The renderings visualize the recorded architecture and proposed operating mechanism; they are not literal simulator meshes or experimentally manufactured structures. All quantitative performance values are obtained from the deterministic simulator.

Installed biomaterials can then be repaired but not dismantled, so their function and degradation remain part of the evolving world.

This workflow produced three trace-grounded persistent biomaterials within 800 ticks (Figure 24). The no-communication society installed a collagen-like CLP\_PPG10 phase in a cellulose-rich matrix at tick 741; at tick 800 it retained health 0.906 and performance 0.363 and achieved mean held-out resilience AUC 0.03548 across eight frozen disturbance schedules. Selected independent member 15 produced CLP\_PPG10 and silk–elastin-like SELP\_47K cellulose composites at ticks 388 and 738. Their retained records expose the actual sequence, material fractions, processing order, tested properties, and persistent state. SELP\_47K was the strongest installed design, with utility 0.729; normalized heat, protease, and flood/shear resistance of 0.726, 0.754, and 0.634; and tick-800 health 0.904 and performance 0.418.

Full culture and no explicit culture generated 13 and 10 proposals, respectively, but did not complete a valid assay within the horizon. The pilot used N = 50, discovery seed 3801, a fixed decision schedule, and an 800-call budget per condition, and all 53 traces completed without provider errors and replayed deterministically. The independent endpoint is the selected best of 50 isolated 16-call worlds rather than another shared N = 50 society. Within these descriptive, single-seed bounds, Protein Realms demonstrates that the same spatial and persistent-world architecture used for BioFoundry and AshenRealm can execute sequence selection, protein–matrix processing, assay-gated construction, continued material persistence, and agent-free evaluation in a sequence-defined biomaterials domain.

![](images/8526b6546c4f0412f77c975681e019888056bc099566a6b8952e52855a99ac32.jpg)  
Figure 23: Production state of the Protein Realms no-communication society at tick 800 (72 × 54 traced cells, N = 50, seed 3801). The four-sided isometric plane is the complete traced rectangular terrain, containing a solvent sea, amino-acid basin, membrane lagoon, chaperone forest, cysteine marsh, cellulose plains, metal-and-salt peaks, ribosome enclave, and stress proving ground. Patterned glyphs mark a deterministic spatial sample of the trace’s molecular-feedstock cells for legibility. Numbered markers identify the ribosome array, purification column, matrix loom, folding chamber, stress assay, and sequence archive; white circles are final agent positions. The green marker is the persistent collagen–cellulose installation constructed at tick 741. Separating feedstocks, processing facilities, and the proving ground forces designs to move through an executable spatial workflow rather than being scored directly from language.

## 3 Conclusion

The experiments support a bounded form of swarm advantage: interaction is most valuable when performance depends on building and maintaining a technological ecology, not when the objective is only to find one record-setting object. Across the 800-tick scaling study, shared worlds produced consistent gains in held-out resilience, portfolio resilience, and validated inventions relative to the endpoint-wise independent-search envelope (Figure 4). At 3,200 ticks, both shared-world conditions retained stronger portfolios and more validated inventions, while isolated search retained the strongest single final artifact (Figure 14). The central distinction is therefore between collective coverage and individual optimization. A society can be better prepared across several needs without containing the single best specialist technology.

Emergence appeared in organization as well as outcome. No agent was assigned to be an explorer, builder, maintainer, or cultural broker, yet label-blind analyses recovered artifact-centered and mobile-exploration phenotypes (Figures 5 and 16). Under explicit culture, the artifact-centered long-horizon fraction increased by 21.8 percentage points, movement declined relative to the ablation, regional crowding increased, and executable lineage depth continued to grow (Figures 15 and 16). Multi-agent construction and cross-agent program forking further show that the emergent organization entered persistent objects and executable controllers rather than remaining a pattern in language alone (Figure 6).

The shared physical world acted as an external memory and transmission medium. Agents with similar total path lengths difered strongly in artifact contact (Figure 8), and event-linked paths show movement becoming observation, construction, installation, and coordination around persistent sites (Figure 9). Recorded provenance connects evidence and programs to downstream artifacts (Figure 11), while the material-process analysis separates proposed form from executed construction and operational flux (Figure 12). Most importantly, technology reuse was nearly universal and faster under full culture, but approximately 95% of first adoption occurred through physical observation and direct inventor-to-adopter social contact was not consistently enriched against the shufled null (Figure 20). Culture therefore operated difusely: explicit exchange reshaped the society-wide substrate, after which much transmission occurred through encounters with the world itself. In evolutionary terms, variation occurred in design and code space through authorship, forking, and edits, while simulator dynamics imposed diferential functional consequences at the locations where technologies ran. The emerging picture is therefore closer to technological evolution than to conventional multiagent coordination: intelligence is distributed not only across agents, but across agents, persistent artifacts,

![](images/8506d2abe2b172e5254138486cfabea15576216274e08148a197a85a52b36f43.jpg)

## B Realized sequence–matrix designs

![](images/4fb913d94b44143ff733bbfa6f48661e72b72ae070a26cef04d2ee5461d5f71e.jpg)

C Matter inputs and executed fabrication pathways  
![](images/303391e5249fa75786dec32cf6426495f33f277c9c22856fd35330e4a0f773b8.jpg)

![](images/e11b5e133f54cfb4517f512f4d34c9f6cb4c76cb149fc5b7cbb0e11d5dfdb9a4.jpg)  
Figure 24: Trace-grounded designs and outcomes from the single-seed, 800-tick Protein Realms pilot. (A) Stage counts from proposals through microbatches, tests, and persistent installations across all four conditions; endpoint labels give mean resilience AUC across eight frozen, agent-free disturbance schedules. Independent member 15 is the selected member of the predeclared best-of-50 isolated-search envelope and represents one 16-call world, not another shared $N = 5 0$ society. (B) The three realized installations. Colored sequence strips reproduce the exact retained catalog sequences by residue class; the adjacent protein-in-cellulose drawings are explanatory architecture schematics, not molecular-structure predictions. (C) Recorded recipe inputs and executed fabrication pathways. Bar widths give input mass fractions, numbered nodes preserve operation order, and node color encodes recorded intensi ${ \boldsymbol { \mathbf { \rho } } } { \mathrm { J } } \cdot { \boldsymbol { \mathbf { \rho } } }$ (D) Realized consequences: normalized tested utility and heat, protease, and flood/shear resistance, followed by health and current performance at tick 800. These are simulator outputs from fixed catalog priors and executed recipes, not biochemical measurements. Only no communication and the selected independent member completed the hidden spatial assay gate; full culture and no explicit culture therefore remain visible in panel A but have no invented design rows. The pilot demonstrates an executable sequence-to-persistent-material workflow but does not support inferential comparison among conditions.

executable lineages and a world whose accumulated modifications reshape the search space encountered by subsequent activity.

The mature society was neither a globally synchronized collective nor a collection of independent agents. It formed persistent, modular technological neighborhoods with a limited set of local hubs and cross-community connectors (Figures 17 and 18). Agent-artifact ties densified superlinearly, full culture produced roughly twice as many cumulative ties, and interval-to-interval relationship reuse rose in both shared worlds (Figure 19). This organization created distributed redundancy: random removal of half the agents left nearly all artifacts connected to at least one survivor. It also created concentration: targeted removal of high-degree agents or brokers caused much larger losses (Figure S2). Self-organization thus produced both robustness and a recognizable failure mode.

Longer time did not convert every cultural mechanism into a universal advantage. The crossover depended on the measured outcome: best-artifact performance favored full culture by about tick 800, portfolio resilience and artifact production crossed near tick 1,600, invention count never crossed, and held-out resilience showed no persistent full-culture advantage (Figure 13). No explicit culture remained highly capable because artifact stigmergy preserved a powerful form of decentralized coordination. This result is scientifically important because it identifies the shared world, rather than communication alone, as a principal source of collective capability. It also shows that adding cultural channels changes what the population learns to organize around, but can introduce coordination costs and structural dependence without guaranteeing better function on every endpoint.

The claims should remain proportional to the evidence. Inference is based on four matched world seeds per condition, one model and prompting configuration, and simulator-defined material and environmental functions. The technology portraits in Figure 7 are mechanism visualizations rather than experimentally manufactured geometries, and the knockout assay in Figure S2 measures graph topology rather than physical recovery after agents disappear from a running world. Within those boundaries, the study establishes a reproducible experimental path for studying language-model swarms as evolving societies: measure not only final answers, but also movement, durable artifacts, executable inheritance, recorded provenance, difusion, network memory, and failure under controlled intervention.

The proposal-consequence separation also charts the path outward, because the consequence layer can in principle be replaced while retaining the same measurement framework, extending earlier work that emphasized the importance of the builder-breaker philosophy <sup>60</sup>. A first step keeps the world in silico and raises its fidelity: coupling recipes, geometries, and controllers to established atomistic and continuum solvers would move simulator-defined function toward real material behavior, extending physics-aware agentic design <sup>42,44,46,60</sup>. A second step grounds sensing and disturbance in reality: held-out schedules drawn from measured environmental records, and artifact controllers executed on embedded hardware against live sensor streams, would test whether stigmergic societies remain capable when the world is measured rather than generated. The third step is embodiment: robotic platforms and autonomous laboratories can supply the construction, testing, and observation primitives that the simulator currently resolves <sup>12,24,35</sup>, so that persistent artifacts become physical objects in shared workspaces and stigmergy operates through the actual environment; our experimentally validated protein-design swarm illustrates one route from agent-swarm design to experimental testing <sup>48,60</sup>, and distributed artifact exchange across laboratories ofers a route to scale it<sup>52</sup>. Because the independent-search envelope, the agent-free held-out assay, provenance, executable lineage, and knockout analysis provide a common evaluation template across consequence layers, each of these substitutions can be evaluated against the same falsifiable criteria established here. Heterogeneous model populations, agent mortality and reproduction, resource economies, and human participants acting in the same persistent world define complementary axes along which the boundaries of the swarm advantage can be mapped.

## 4 Materials and Methods

## 4.1 Overall algorithm

SwarmWorld separates cognition from physical execution (see Algorithm 1). At fixed, staggered macroturns, each agent receives only its local observation, private memory, and whatever shared records are permitted by the experimental condition. The LLM returns a strictly validated research-state update and a plan of up to L atomic actions. These actions enter a queue, and only one action per agent is attempted during each simulator tick. Consequently, agents can formulate multistep strategies without requiring an LLM call at every movement or physical operation.

The deterministic simulator checks every attempted action against spatial, material, energetic, and treatmentspecific constraints before changing the world. It then advances environmental fields and source–sink-accounted resources and executes every installed artifact program. Successful and failed outcomes become new experience, while communication, publication, program inheritance, and artifact visibility depend on the experimental condition. Neither roles nor recipes are assigned by the simulator, model weights remain fixed, and observercomputed performance metrics are not supplied to the agents as a global reward.

At each declared evaluation checkpoint, the world and its technological portfolio are frozen and all agents are removed. Identical copies are exposed to paired, previously unseen disturbance schedules, during which only deterministic physics and installed artifact programs operate. This measures whether the accumulated technology remains functional without continued LLM intervention.

Algorithm 1 SwarmWorld discovery and agent-free evaluation   
Require: Population $N ;$ discovery horizon $T ;$ macroturn interval $m ;$ plan limit $L ;$ condition capabilities $\mathcal { C } ;$   
evaluation checkpoints $\mathcal { H } ;$ held-out disturbance schedules D   
Ensure: Event trace, discovery metrics, and held-out resilience measurements   
1: W ← GenerateW $\mathrm { \mathrm { ) R L D } } \bigl ( s _ { \mathrm { d i s c } } \bigr )$   
2: $\{ x _ { i } , M _ { i } , Q _ { i } , \phi _ { i } \} _ { i = 1 } ^ { N } $ InitializeAgents $( N , W )$ ▷ $x _ { i } \colon$ agent state; $M _ { i } { \mathrm { : } }$ private memory; $Q _ { i } \colon$ action   
queue; ϕ<sub>i</sub>: macroturn phase   
3: $\bar { \mathcal { A } }  \emptyset ; \mathcal { P }  \emptyset ; \mathcal { K }  \emptyset$ ▷ $\mathcal { A } \mathrm { : }$ artifacts; $\mathcal { P } \colon$ programs; $\kappa \mathrm { : }$ permitted shared records   
4: for $t = 0 , \ldots , T - 1$ do   
5: $\mathcal { T } _ { t }  \{ i : ( t - \phi _ { i } )$ mod $m = 0 \}$ $\triangleright$ fixed, treatment-invariant macroturn schedule   
6: for all $i \in \mathcal { I } _ { t }$ in parallel do   
7: $o _ { i } \gets \mathrm { O B S E R V E L O C A L W O R L D } ( x _ { i } , W , A , \mathcal { C } )$   
8: $r _ { i } \gets \mathrm { R E T R I E V E C O N T E X T } ( M _ { i } , { \mathcal { K } } , \mathcal { P } , o _ { i } , \mathcal { C } )$   
9: $( z _ { i } , \widehat { Q } _ { i } ) \gets \mathrm { L L M S } ^ { \prime }$ tructured $\mathrm { P L A N } \big ( o _ { i } , r _ { i } , \mathcal { C } , L \big ) \ \vartriangleright z _ { i } \colon$ research-state update; ${ \widehat { Q } } _ { i } .$ proposed action   
plan   
10: $\widehat { Q } _ { i } \gets \mathrm { V A L I D A T E S C H E M A } ( \widehat { Q } _ { i } , \mathcal { C } )$   
11: $M _ { i } \gets \mathrm { S T O R E R E S E A R C H S T A T E } ( M _ { i } , z _ { i } )$   
12: $Q _ { i }  \widehat { Q } _ { i }$ ▷ commit the validated plan as a bounded action queue   
13: end for   
14: for all active agents $i$ do   
15: if $Q _ { i } \neq \emptyset$ then   
16: $u _ { i } \gets \mathrm { P o p F }$ ront $\left( Q _ { i } \right)$   
17: else   
18: $u _ { i } \gets \mathrm { W A I T }$   
19: end if   
20: end for   
21: $E _ { t } ^ { \mathrm { a c t } } \gets \mathrm { R }$ esolveAndApply $( \{ u _ { i } \} _ { i = 1 } ^ { N } , W , \mathcal { A } , \mathcal { P } , \mathcal { K } , \mathcal { C } )$ ▷ enforce spatial, material, energetic, permission,   
and action constraints   
22: $( W , E _ { t } ^ { \mathrm { e n v } } ) \gets \mathrm { A }$ dvanceEnvironment $( W , t )$   
23: $( W , \mathcal { A } , E _ { t } ^ { \mathrm { a r t } } ) \gets$ ExecuteArtifactPrograms $( W , A , \mathcal { P } , t )$ ▷ persistent controllers execute on every   
simulator tick   
24: $E _ { t } \gets E _ { t } ^ { \mathrm { a c t } } \cup E _ { t } ^ { \mathrm { e n v } } \cup E _ { t } ^ { \mathrm { a r t } }$   
25: $( \{ M _ { i } \} , \dot { \mathcal { K } } ) \gets \check { \mathrm { C O M M I T E V E N T S } } ( E _ { t } , \mathcal { C } )$   
26: AppendTrace(t, $W , \{ x _ { i } , M _ { i } , \dot { Q } _ { i } \} _ { i = 1 } ^ { N } , A , \mathcal { P } , \mathcal { K } , E _ { t } )$   
27: if $t + 1 \in { \mathcal { H } }$ then   
28: $F _ { t + 1 } \gets$ FreezeAndRemoveAgent $( W , A , { \mathcal { P } } )$   
29: for all $d \in \mathcal { D }$ do   
30: $F _ { t + 1 , d } \gets \mathrm { C L O N E A N D S T R E S S } ( F _ { t + 1 } , d )$   
31: $R _ { t + 1 , d }$ ← EvaluateAgentFree $( F _ { t + 1 , d } )$ ▷ only deterministic physics and installed programs   
continue   
32: end for   
33: end if   
34: end for   
35: return trace, discovery metrics, and $\{ R _ { h , d } : h \in \mathcal { H } , \ d \in \mathcal { D } \}$

## 4.1.1 Agent-to-world interface

Agents observe a semantic description of their local neighborhood and return a closed, schema-validated research-state update and action plan. Private memory stores recent outcomes, tested recipes, notebook evidence, and the agent’s current hypothesis and next checkpoint; treatment-permitted records supply the only symbolic information from other agents. Runtime validation enforces location, ownership, empirical grounding, capacity, and condition permissions, and invalid output becomes a logged safe wait. Representative retained messages and actions and the complete interface specification are provided in Section S3.

## 4.2 World representation and constructing custom worlds

Each simulation world is represented as a rectangular two-dimensional lattice with typed terrain, resource, facility, and continuous environmental-field layers. A world instance is generated as

$$
W _ { s } = G ( s ; \pmb \theta ) ,
$$

where s is a random seed and θ specifies dimensions, spatial-generation rules, resource capacities and renewal rates, environmental dynamics, facility-placement constraints, and disturbance processes. Generation is deterministic for fixed s and θ. Candidate layouts are validated against structural invariants, including suficient walkable area, availability of required resource classes, nonoverlapping facilities, and reachable interaction regions. Invalid candidates are rejected and regenerated deterministically. The resulting arrays and complete generator manifest are recorded in the simulation trace, allowing the initial world to be reconstructed exactly.

The world builder also supports declarative scenario packages that replace or extend terrain, resource, facility, processing, field-dynamics, and service definitions without modifying the simulation loop. Packages are specified through validated YAML documents and mapped onto the simulator’s stable typed interfaces; package-authored executable code is not loaded. Alternative procedural generators may therefore be used provided that they produce the same authoritative state representation and satisfy the declared physica invariants. Visual presentation is kept separate from simulation state: terrain meshes, object assets, textures, lighting, particles, and shader parameters may be assigned by a renderer, but they cannot alter physics, resource quantities, agent observations, action legality, or measured outcomes. Thus, the same simulated world may be rendered at diferent levels of visual fidelity while remaining scientifically identical. Further details are provided in Section S2.2.

## 4.3 Materials, artifacts, and interaction mechanisms

Agents collect source–sink-accounted feedstocks, formulate and test processing recipes, and build persistent material systems whose measured behavior is determined by numerical material state, geometry, environment, stored matter, health, and optional bounded controller programs. Communication, publication, teaching, trade, task claims, program reuse, and program forking are independently controlled by the experimental condition, while the shared physical world can retain stigmergic traces in every condition except isolated search. Equations, controller limits, source–sink accounting, and provenance rules are given in Section S2.3.

## 4.4 Experimental design and analysis

The completed 800-tick study crossed four interaction conditions, three population sizes $( N = 5 0 , 1 0 0 , 2 0 0 )$ and four matched discovery seeds, yielding 48 episodes. A separate completed 3,200-tick study compared full culture, no explicit culture, and independent search at N = 100 over four new seeds. At declared checkpoints, frozen portfolios were evaluated without agents under eight paired held-out disturbance schedules. Ful condition definitions, controls, run provenance, and checkpoint designs are provided in Section S2.6.

Discovery, held-out resilience, portfolio breadth and redundancy, validated inventions, behavior, lineage, and network structure were computed from immutable traces and frozen evaluation copies. The independently generated world seed was the unit of inference; conditions were compared within matched population–seed blocks using deterministic paired bootstrap intervals, supplemented by exact sign-flip tests for the long-horizon study. Endpoint definitions and analysis procedures are provided in Section S2.9, and release validation and reproducibility details in Section S2.12.

## Data and code availability

Code, exact prompt manifests, versioned protocols, raw model outputs, analysis scripts, figure-generation code, and manuscript sources are available at https://github.com/lamm-mit/SwarmWorld. Additional data from the experiments conducted as part of this paper is available at https://huggingface.co/datasets/ lamm-mit/swarmworld-data.

## Funding

This work was primarily supported by the U.S. Department of Energy, Ofice of Science, Ofice of Advanced Scientific Computing Research and Ofice of Basic Energy Sciences, Scientific Discovery through Advanced Computing (SciDAC) program under the FORUM-AI project. Additional support was provided by MIT’s Generative AI Impact Consortium (MGAIC).

## Author contributions

M.J.B. conceived and led the study, developed the methodology, conducted the computational work and primary analysis, interpreted the results, and drafted the manuscript. S.P. and F.Y.W. contributed to data analysis, additional computational experiments, and manuscript writing and editing. All authors reviewed and approved the final manuscript.

## Competing interests

The authors declare no competing interests.

## References

[1] Bonabeau, E., Theraulaz, G., Deneubourg, J.-L., Aron, S. & Camazine, S. Self-organization in social insects. Trends in Ecology & Evolution 12, 188–193 (1997). URL https://www.cell.com/trends/ ecology-evolution/abstract/S0169-5347(97)01048-3.

[2] Sumpter, D. The principles of collective animal behaviour. Philosophical Transactions of the Royal Society B: Biological Sciences 361, 5–22 (2006). URL https://pmc.ncbi.nlm.nih.gov/articles/ PMC1626537/.

[3] Goss, S., Aron, S., Deneubourg, J.-L. & Pasteels, J. M. Self-Organized Shortcuts in the Argentine Ant. Naturwissenschaften 76, 579–581 (1989). URL https://doi.org/10.1007/BF00462870.

[4] Seeley, T. D. & Visscher, P. K. Quorum sensing during nest-site selection by honeybee swarms. Behavioral Ecology and Sociobiology 56, 594–601 (2004). URL https://doi.org/10.1007/s00265-004-0814-5.

[5] Ward, A. J. W., Sumpter, D. J. T., Couzin, I. D., Hart, P. J. B. & Krause, J. Quorum decision-making facilitates information transfer in fish shoals. Proceedings of the National Academy of Sciences 105, 6948–6953 (2008). URL https://www.pnas.org/doi/full/10.1073/pnas.0710344105.

[6] Grassé, P.-P. La reconstruction du nid et les coordinations interindividuelles chez Bellicositermes natalensis et Cubitermes sp.: la théorie de la stigmergie. Insectes Soc. 6, 41–80 (1959).

[7] Theraulaz, G. & Bonabeau, E. A Brief History of Stigmergy. Artificial Life 5, 97–116 (1999). URL https://doi.org/10.1162/106454699568700.

[8] Gordon, D. M. The organization of work in social insect colonies. Nature 380, 121–124 (1996). URL https://doi.org/10.1038/380121a0.

[9] Dorigo, M., Maniezzo, V. & Colorni, A. Ant system: optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 26, 29–41 (1996). URL https://ieeexplore.ieee.org/abstract/document/484436.

[10] Kennedy, J. & Eberhart, R. Particle swarm optimization. In Proceedings of ICNN’95 - International Conference on Neural Networks, vol. 4, 1942–1948 vol.4 (1995). URL https://ieeexplore.ieee.org/ abstract/document/488968.

[11] Karaboga, D. & Basturk, B. A powerful and eficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. Journal of Global Optimization 39, 459–471 (2007). URL https://doi.org/10.1007/s10898-007-9149-x.

[12] Brambilla, M., Ferrante, E., Birattari, M. & Dorigo, M. Swarm robotics: a review from the swarm engineering perspective. Swarm Intelligence 7, 1–41 (2013). URL https://doi.org/10.1007/ s11721-012-0075-2.

[13] Wang, Y., Lv, J., Zhu, L. & Ma, Y. Crystal structure prediction via particle-swarm optimization. Physical Review B 82, 094116 (2010). URL https://link.aps.org/doi/10.1103/PhysRevB.82.094116.

[14] Wiener, N. Cybernetics: Or Control and Communication in the Animal and the Machine (John Wiley & Sons, New York, 1948).

[15] Forrester, J. W. Urban Dynamics (MIT Press, Cambridge, MA, 1969).

[16] von Neumann, J. Theory of Self-Reproducing Automata (University of Illinois Press, Urbana, IL, 1966).

[17] Gardner, M. Mathematical games: The fantastic combinations of john conway’s new solitaire game “life”. Scientific American 223, 120–123 (1970).

[18] Ahl, D. H. BASIC Computer Games (Creative Computing Press, Morristown, NJ, 1978).

[19] Wright, W. SimCity. Computer software and manual, Maxis Software (1989). URL https://users.ox. ac.uk/\~uzdm0006/scans/sim/manual.html.

[20] Reynolds, C. W. Flocks, herds and schools: A distributed behavioral model. In Proceedings of the 14th Annual Conference on Computer Graphics and Interactive Techniques, 25–34 (Association for Computing Machinery, 1987).

[21] Epstein, J. M. & Axtell, R. Growing Artificial Societies: Social Science from the Bottom Up (Brookings Institution Press and MIT Press, Washington, DC and Cambridge, MA, 1996).

[22] Berkovich, J. A. & Buehler, M. J. LifeGPT: topology-agnostic generative pretrained transformer model for cellular automata. npj Artificial Intelligence 1, 23 (2025). URL https://www.nature.com/articles/ s44387-025-00014-w.

[23] Berkovich, J. A., David, N. S. & Buehler, M. J. AutomataGPT: Transformer-Based Forecasting and Ruleset Inference for Two-Dimensional Cellular Automata. Advanced Science 13, e11352 (2026). URL https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202511352.

[24] Strobel, V., Dorigo, M. & Fritz, M. LLM2Swarm: Robot Swarms that Responsively Reason, Plan, and Collaborate through LLMs (2024). URL http://arxiv.org/abs/2410.11387. ArXiv:2410.11387 [cs.RO].

[25] Jimenez-Romero, C., Yegenoglu, A. & Blum, C. Multi-agent systems powered by large language models: applications in swarm intelligence. Frontiers in Artificial Intelligence 8 (2025). URL https://www.frontiersin.org/journals/artificial-intelligence/articles/10. 3389/frai.2025.1593017/full.

[26] Jiang, Y., Zhao, L., Li, A. Q., Chen, M. & Balkcom, D. Exploring Spontaneous Social Interaction Swarm Robotics Powered by Large Language Models. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 10198–10205 (2025). URL https://ieeexplore.ieee.org/ document/11246362/. ISSN: 2153-0866.

[27] Lowe, R. et al. Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems, vol. 30, 6379–6390 (2017).

[28] Baker, B. et al. Emergent tool use from multi-agent autocurricula. In International Conference on Learning Representations (2020). 1909.07528.

[29] Park, J. S. et al. Generative agents: Interactive simulacra of human behavior. In Proc. 36th Annu. ACM Symp. User Interface Softw. Technol. (2023). URL https://arxiv.org/abs/2304.03442.

[30] Altera.AL et al. Project sid: Many-agent simulations toward AI civilization. arXiv preprint arXiv:2411.00114 (2024). URL https://arxiv.org/abs/2411.00114.

[31] Piatti, G. et al. Cooperate or collapse: Emergence of sustainable cooperation in a society of LLM agents. In Adv. Neural Inf. Process. Syst., vol. 37 (2024).

[32] Piao, J. et al. AgentSociety: Large-scale simulation of LLM-driven generative agents advances understanding of human behaviors and society. arXiv preprint arXiv:2502.08691 (2025). URL https://arxiv.org/abs/2502.08691.

[33] Paolo, G. et al. TerraLingua: Emergence and analysis of open-endedness in LLM ecologies. arXiv preprint arXiv:2603.16910 (2026). URL https://arxiv.org/abs/2603.16910.

[34] Wang, G. et al. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023). URL https://arxiv.org/abs/2305.16291.

[35] Ji, W. et al. GenSwarm: Scalable Multi-Robot Code-Policy Generation and Deployment via Language Models. npj Robotics 4, 5 (2026). URL https://www.nature.com/articles/s44182-025-00065-w.

[36] Jansen, P. et al. DISCOVERYWORLD: A virtual environment for developing and evaluating automated scientific discovery agents. In Adv. Neural Inf. Process. Syst., vol. 37 (2024). URL https://arxiv.org/ abs/2406.06769.

[37] Ruan, K., Huang, M., Wen, J.-R. & Sun, H. Benchmarking LLMs’ Swarm intelligence (2025). URL http://arxiv.org/abs/2505.04364. ArXiv:2505.04364 [cs.MA].

[38] Zomer, N. & De Domenico, M. Unraveling the emergence of collective behavior in networks of cognitive agents. npj Artificial Intelligence 2, 36 (2026). URL https://www.nature.com/articles/ s44387-026-00091-5.

[39] Buehler, M. J. MechGPT, a Language-Based Strategy for Mechanics and Materials Modeling That Connects Knowledge Across Scales, Disciplines, and Modalities. Applied Mechanics Reviews 76 (2024). URL https://doi.org/10.1115/1.4063843.

[40] Buehler, M. J. Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning. Machine Learning: Science and Technology 5, 035083 (2024). URL https://doi.org/10.1088/2632-2153/ad7228.

[41] Buehler, M. J. Generative Retrieval-Augmented Ontologic Graph and Multiagent Strategies for Interpretive Large Language Model-Based Materials Design. ACS Engineering Au 4, 241–277 (2024). URL https://doi.org/10.1021/acsengineeringau.3c00058.

[42] Ghafarollahi, A. & J. Buehler, M. ProtAgents: protein discovery via large language model multiagent collaborations combining physics and machine learning (2024). URL https://pubs.rsc.org/en/ content/articlehtml/2024/dd/d4dd00013g.

[43] Ghafarollahi, A. & Buehler, M. J. SciAgents: Automating Scientific Discovery Through Bioinspired Multi-Agent Intelligent Graph Reasoning. Advanced Materials 37, 2413523 (2025). URL https: //advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202413523.

[44] Ghafarollahi, A. & Buehler, M. J. Automating alloy design and discovery with physics-aware multimodal multiagent AI. Proceedings of the National Academy of Sciences 122, e2414074122 (2025). URL https://www.pnas.org/doi/abs/10.1073/pnas.2414074122.

[45] Ghafarollahi, A. & Buehler, M. J. Sparks: Multi-Agent Artificial Intelligence Model Discovers Protein Design Principles (2025). URL http://arxiv.org/abs/2504.19017. ArXiv:2504.19017 [cs].

[46] Ghafarollahi, A. & Buehler, M. J. Autonomous in-silico inorganic materials discovery via multi-agent physics-aware scientific reasoning. npj Computational Materials (2026). URL https://www.nature. com/articles/s41524-026-02205-8.

[47] Huang, X. et al. CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution (2025). URL http://arxiv.org/abs/2512.23880. ArXiv:2512.23880 [cs].

[48] Wang, F. Y., Lee, D. S., Kaplan, D. L. & Buehler, M. J. Swarms of Large Language Model Agents for Protein Sequence Design with Experimental Validation. Digital Discovery (2026). URL https: //doi.org/10.1039/D6DD00282J.

[49] Song, K., Trotter, A. & Chen, J. Y. LLM Agent Swarm for Hypothesis-Driven Drug Discovery (2025). URL http://arxiv.org/abs/2504.17967. ArXiv:2504.17967 [cs.AI].

[50] Buehler, M. J. MusicSwarm: Biologically Inspired Intelligence for Music Composition. Advanced Intelligent Systems 8, e202501253 (2026). URL https://onlinelibrary.wiley.com/doi/abs/10. 1002/aisy.202501253.

[51] Loo, L. & Braga-Neto, U. An Agentic AI Scientific Community for Automated Neural Operator Discovery (2026). URL http://arxiv.org/abs/2607.12122. ArXiv:2607.12122 [cs.LG].

[52] Wang, F. Y. et al. Autonomous Agents Coordinating Distributed Discovery Through Emergent Artifact Exchange (2026). URL http://arxiv.org/abs/2603.14312. ArXiv:2603.14312 [cs.AI].

[53] Benjamini, Y. & Hochberg, Y. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological) 57, 289–300 (1995).

[54] Guimerà, R. & Amaral, L. A. N. Cartography of complex networks: Modules and universal roles. Journal of Statistical Mechanics: Theory and Experiment 2005, P02001 (2005).

[55] Kaplan, E. L. & Meier, P. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association 53, 457–481 (1958).

[56] Kramer, R. Z. et al. X-ray crystallographic determination of a collagen-like peptide with the repeating sequence (pro-pro-gly) (1998). URL https://doi.org/10.2210/pdb1A3J/pdb.

[57] Teng, W., Cappello, J. & Wu, X. Recombinant silk-elastinlike protein polymer displays elasticity comparable to elastin. Biomacromolecules 10, 3028–3036 (2009). URL https://doi.org/10.1021/ bm900651g.

[58] Li, L., Teller, S., Clifton, R. J., Jia, X. & Kiick, K. L. Tunable mechanical stability and deformation response of a resilin-based elastomer. Biomacromolecules 12, 2302–2310 (2011). URL https://doi. org/10.1021/bm200373p.

[59] Das, S., Martinez Rodriguez, N. R., Wei, W., Waite, J. H. & Israelachvili, J. N. Peptide length and dopa determine iron-mediated cohesion of mussel foot proteins. Advanced Functional Materials 25, 5840–5847 (2015). URL https://doi.org/10.1002/adfm.201502256.

[60] Wang, F. Y. & Buehler, M. J. Self-revising discovery systems for science: A categorical framework for agentic artificial intelligence (2026). URL https://doi.org/10.48550/arXiv.2606.01444. Extreme Mechanics Letters, in press, 2606.01444.

[61] McInnes, L., Healy, J., Saul, N. & Großberger, L. UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software 3, 861 (2018). URL https://joss.theoj.org/papers/ 10.21105/joss.00861.

[62] Rousseeuw, P. J. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics 20, 53–65 (1987).

[63] Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment 2008, P10008 (2008).

[64] Almeida-Neto, M., Guimarães, P., Guimarães Jr, P. R., Loyola, R. D. & Ulrich, W. A consistent metric for nestedness analysis in ecological systems: reconciling concept and measurement. Oikos 117, 1227–1239 (2008). URL https://onlinelibrary.wiley.com/doi/abs/10.1111/j.0030-1299.2008.16644.x.

# Supplementary Information

# SwarmWorld: Stigmergic technological evolution in societies of language-model agents

Subhadeep Pal<sup>1,2</sup> Fiona Y. Wang<sup>1,3</sup> Markus J. Buehler<sup>1,2,4,5,#</sup>

<sup>1</sup>Laboratory for Atomistic and Molecular Mechanics (LAMM) <sup>2</sup>Department of Civil and Environmental Engineering, <sup>3</sup>Department of Biological Engineering <sup>4</sup>Department of Mechanical Engineering, <sup>5</sup>Center for Computational Science and Engineering, Schwarzman College of Computing, Massachusetts Institute of Technology, Cambridge, MA 02139, USA

<sup>#</sup>Corresponding author: mbuehler@mit.edu

## S1 Glossary of key terms

Table S1 defines the quantities used throughout the paper. Equations are included only where the implementation computes a corresponding mathematical quantity; terms that describe an intervention, object, or event are defined operationally.

Table S1: Glossary of technical terms, implementation-aligned definitions, plain-language interpretations, and broadly accessible examples.
<table><tr><td rowspan=1 colspan=13>Term               Technical definition                     Plain-language meaning   Broad example</td></tr><tr><td rowspan=6 colspan=13>World tick         One discrete deterministic state       The basic clock step of    A video game advancestransition, $S _ { t + 1 } = F ( S _ { t } , A _ { t } )$ , where    the simulated world. Theone frame at a time; $S _ { t }$ is the complete world state and $A _ { t }$  world keeps changing      machines, weather, andis the set of accepted agent actions at even when no agent is     moving characterstick t. Given the state, actions, and   currently asking the       continue between aseeded disturbance schedule, F         language model what to   player&#x27;s major decisions.advances movement, metabolism.      do.resources, fields, artifacts, andinstalled programs.Macroturn or      A scheduled occasion on which one   One chance for an agent  A field scientistdecision            agent receives a local observation and to stop, look around,      periodically checksmemory context, calls the language    think, and choose its next instruments and updatesthe day&#x27;s plan while theexperiment continues</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>res</td></tr><tr><td rowspan=1 colspan=3>ins</td></tr><tr><td rowspan=1 colspan=3>opportunity</td><td rowspan=1 colspan=4>memory context, calls the languag</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>multi-action plan. Macroturn phases</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=2 colspan=6>are staggered and matched acrosspaired conditions.</td><td rowspan=2 colspan=1></td><td rowspan=3 colspan=3>running between checks.A device left on a shared</td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=2>Persistent</td><td></td><td rowspan=1 colspan=6>An agent-constructed world object</td><td rowspan=1 colspan=1>A technology that stays</td></tr><tr><td rowspan=2 colspan=3>artifact</td><td rowspan=2 colspan=6>with recorded creator, contributors,</td><td rowspan=2 colspan=1>behind and can later be</td><td rowspan=2 colspan=2>lab</td><td rowspan=2 colspan=1>furear</td></tr><tr><td rowspan=2 colspan=1>laboratory bench caninfluence the next</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>materials, geometry, services,</td><td rowspan=1 colspan=1>found, tested, operated,</td><td rowspan=1 colspan=2>influe</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>provenance, state, and optionally an</td><td rowspan=1 colspan=1>or modified by someone</td><td rowspan=7 colspan=3>researcher even if itsbuilder is absent.A thermostat repeatedlyreads temperature and</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>executable controller. It remains in</td><td rowspan=1 colspan=1>else.</td><td rowspan=1 colspan=2>builde</td><td rowspan=1 colspan=1>ilder is</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>the world until modified, retired,</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>dismantled, or otherwise changed by</td><td></td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>simulator rules.</td></tr><tr><td rowspan=1 colspan=3>Artifact program</td><td rowspan=1 colspan=6>A validated instruction sequence</td><td rowspan=1 colspan=1>Code that lets a built</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>A thermo</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=22 colspan=10>installed on an artifact and executedby the deterministic simulator every                                 turns heating on or offafter the installer leaves.reads the artifact&#x27;spermitted local sensors    and emitspermitted control operationsThe intervention containing one                                      A research group shares ashared physical world, explicit                                        laboratory, talks, writescultural actions, cross-agent                                          notebooks, and modifiesexecutable-program inheritance, and                                 common software andartifact stigmergy. Explicit actions    through the things they   equipment.include messages, public records,      build.Ants coordinate throughmodified trails; peopleworld rather than through can likewise coordinateby adding parts orchanged state. It remains available in                                annotations to a sharedworkbench.cross-agent program inheritance aredisabled.</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>by the determinist</td><td rowspan=1 colspan=4>ic simulator every</td><td rowspan=1 colspan=1>conditions and act</td><td rowspan=1 colspan=2>turns</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>tick. Conceptually</td><td rowspan=1 colspan=4>, ua,t = πa(Oa,t):</td><td rowspan=1 colspan=1>without another</td><td rowspan=1 colspan=2>after t]</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>program πa reads</td><td rowspan=1 colspan=4>the artifact's</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>permitted local sei</td><td rowspan=1 colspan=3>sors oa,t and emi</td><td rowspan=1 colspan=1>S</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>permitted control</td><td rowspan=1 colspan=4>operations ua,t.</td></tr><tr><td rowspan=1 colspan=3>Full culture</td><td rowspan=1 colspan=2>The intervention</td><td rowspan=1 colspan=4>ontaining one</td><td rowspan=1 colspan=1>Agents can communicate</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>shared physical w</td><td rowspan=1 colspan=1>orld.</td><td rowspan=1 colspan=3>explicit</td><td rowspan=1 colspan=1>directly, inherit one</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>cultural actions, c</td><td rowspan=1 colspan=1>oss</td><td rowspan=1 colspan=3>agent</td><td rowspan=1 colspan=1>another's code, and also</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>executable-progra</td><td rowspan=1 colspan=1>n in</td><td rowspan=1 colspan=3>heritance, and</td><td rowspan=1 colspan=1>coordinate indirectly</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>artifact stigmergy</td><td rowspan=1 colspan=4>Explicit actions</td><td rowspan=1 colspan=1>through the things they</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6></td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>teaching, trading, task claims, and</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=3 colspan=7></td><td rowspan=1 colspan=3>publication-dependent composition.</td></tr><tr><td rowspan=1 colspan=3>Artifact</td><td rowspan=1 colspan=6>Indirect coordination produced when</td></tr><tr><td rowspan=1 colspan=3>stigmergy</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>or the environment and another</td><td rowspan=3 colspan=1>direct convetioon.rough</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>agent later observes or acts on that</td><td rowspan=1 colspan=1>at</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=5>changed state. It remains available</td><td rowspan=1 colspan=1>e in</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=6>the no-explicit-culture condition even</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=5>when direct cultural channels and</td></tr><tr><td rowspan=1 colspan=3></td></tr></table>

Continued on next page

Table S1 continued
<table><tr><td colspan="6" rowspan="1">Term               Technical definition                     Plain-language meaning   Broad example</td></tr><tr><td colspan="6" rowspan="1">Independent-       For endpoint $e ,$ checkpoint $t ,$ and $N$   The comparison gives      A national team may</td></tr><tr><td colspan="4" rowspan="1">search envelope   isolated one-agent worlds, the control</td><td colspan="1" rowspan="1">isolated search its best</td><td colspan="1" rowspan="1">choose one athlete for the</td></tr><tr><td colspan="4" rowspan="1">is $Y _ { \mathrm { i s o } } ( e , t ) = \operatorname* { m a x } _ { i \in \{ 1 , . . . , N \} } Y _ { i } ( e , t )$ </td><td colspan="1" rowspan="1">available result for each</td><td colspan="1" rowspan="1">sprint and another for the</td></tr><tr><td colspan="4" rowspan="1">The maximizing member may differ</td><td colspan="1" rowspan="1">question, rather than</td><td colspan="1" rowspan="1">high jump; the envelope</td></tr><tr><td colspan="3" rowspan="2"></td><td colspan="1" rowspan="1">across endpoints and checkpoints.</td><td colspan="1" rowspan="1">forcing one solo agent to</td><td colspan="1" rowspan="1">records the best specialist</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">win every contest.</td><td colspan="1" rowspan="1">in each event.</td></tr><tr><td colspan="4" rowspan="1">Discovery-         If $p ( t )$ is measured artifact</td><td colspan="1" rowspan="1">A strong invention scores</td><td colspan="1" rowspan="1">Two runners may finish</td></tr><tr><td colspan="3" rowspan="1">frontier AUC</td><td colspan="1" rowspan="1">performance, the running frontier is</td><td colspan="1" rowspan="1">more if it appears early</td><td colspan="1" rowspan="1">with the same final speed</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"> $\begin{array} { r } { \mathbf { \bar { \boldsymbol { F } } } ( t ) = \operatorname* { m a x } _ { \tau \leq t } \boldsymbol { p } ( \tau ) } \end{array}$ The normalized</td><td colspan="1" rowspan="1">enough to remain the</td><td colspan="1" rowspan="1">but the one who led for</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">area is $\begin{array} { r } { D _ { T } = T ^ { - 1 } \int _ { 0 } ^ { T } F ( t ) d t . } \end{array}$ </td><td colspan="1" rowspan="1">best-known option for</td><td colspan="1" rowspan="1">most of the race has the</td></tr><tr><td></td><td></td><td></td><td colspan="1" rowspan="2">implemented by trapezoidal</td><td colspan="1" rowspan="2">much of the experiment.</td><td colspan="1" rowspan="2">larger time-averaged</td></tr><tr><td colspan="3" rowspan="1"></td></tr><tr><td colspan="3" rowspan="2"></td><td colspan="1" rowspan="2">integration over recorded samples.</td><td></td><td></td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">frontier.</td></tr><tr><td colspan="3" rowspan="1">Current service</td><td colspan="1" rowspan="1">For service dimension k at time $t ,$ </td><td colspan="1" rowspan="1">The society is judged by</td><td colspan="1" rowspan="1">An emergency kit with</td></tr><tr><td colspan="3" rowspan="1">coverage</td><td colspan="1" rowspan="1"> $c _ { k } ( t ) = \operatorname* { m a x } _ { a \in \mathcal { A } _ { t } } s _ { a k } ( t )$ over active,</td><td colspan="1" rowspan="1">the best currently</td><td colspan="1" rowspan="1">food, water, shelter, and</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">non-retired artifacts. With</td><td colspan="1" rowspan="1">available artifact for each</td><td colspan="1" rowspan="1">medicine is more</td></tr><tr><td colspan="4" rowspan="1"> $\begin{array} { r } { \bar { c } ( t ) = K ^ { - 1 } \sum _ { k } c _ { k } ( t ) } \end{array}$ and</td><td colspan="1" rowspan="1">need, with a penalty</td><td colspan="1" rowspan="1">balanced than one</td></tr><tr><td colspan="4" rowspan="1"> $b ( t ) = \operatorname* { m i n } _ { k } { c _ { k } ( \tilde { t } ) } / { \bar { c } ( t ) }$ , balanced</td><td colspan="1" rowspan="1">when one need is badly</td><td colspan="1" rowspan="1">containing excellent food</td></tr><tr><td colspan="4" rowspan="1">coverage is $Q ( \dot { t } ) \dot { = } \dot { \bar { c } } ( t ) [ 1 + b ( t ) ] / 2$ </td><td colspan="1" rowspan="1">neglected.</td><td colspan="1" rowspan="3">but nothing else.A toolbox with a few</td></tr><tr><td colspan="4" rowspan="1">when $\bar { c } ( t ) > 0$ </td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="4" rowspan="1">Portfolio           The discovery-state score uses</td><td colspan="1" rowspan="1">A portfolio is valuable</td></tr><tr><td colspan="3" rowspan="1">resilience</td><td colspan="1" rowspan="1">lifetime-peak service fingerprints,</td><td colspan="1" rowspan="1">when its collection has</td><td colspan="1" rowspan="1">complementary, proven</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"> $c _ { k } ^ { \mathrm { p e a k } } = \operatorname* { m a x } _ { a } s _ { a k } ^ { \mathrm { p e a k } }$ , and the same</td><td colspan="1" rowspan="1">demonstrated strong and</td><td colspan="1" rowspan="1">tools can be more</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">balance formula</td><td colspan="1" rowspan="1">balanced coverage across</td><td colspan="1" rowspan="1">resilient than a warehouse</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"> $\begin{array} { r } { P = \bar { c } ^ { \mathrm { p e a k } } [ 1 + \operatorname* { m i n } _ { k } c _ { k } ^ { \mathrm { p e a k } } / \bar { c } ^ { \mathrm { p e a k } } ] / 2 . } \end{array}$ </td><td colspan="1" rowspan="1">several functions, not</td><td colspan="1" rowspan="1">full of duplicate hammers.</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">Artifact count alone does not increase</td><td colspan="1" rowspan="1">merely because it</td><td colspan="1" rowspan="2"></td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">the score.</td><td colspan="1" rowspan="1">contains many objects.</td></tr><tr><td colspan="3" rowspan="1">Held-out</td><td colspan="1" rowspan="1">For unseen schedule $s ,$ </td><td colspan="1" rowspan="1">The frozen technology is</td><td colspan="1" rowspan="1">A bridge design is</td></tr><tr><td colspan="3" rowspan="1">resilience</td><td colspan="1" rowspan="1"> $\begin{array} { r } { R _ { s } = T ^ { - 1 } \int _ { 0 } ^ { T } Q _ { s } ( t ) d t ; } \end{array}$ where $Q _ { s } ( t )$ is</td><td colspan="1" rowspan="1">tested under new</td><td colspan="1" rowspan="1">evaluated under</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">current balanced service coverage</td><td colspan="1" rowspan="1">disturbances after all</td><td colspan="1" rowspan="1">earthquakes and wind</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">during the agent-free assay. The</td><td colspan="1" rowspan="1">agents are removed; the</td><td colspan="1" rowspan="1">patterns that were not</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="2">reported value is $R = S ^ { - 1 } \sum _ { s = 1 } ^ { S } R _ { s }$ </td><td colspan="1" rowspan="1">score measures how well</td><td colspan="1" rowspan="1">used while designing it,</td></tr><tr><td></td><td></td><td></td><td colspan="1" rowspan="3">over $S = 8$ paired schedules.</td><td colspan="1" rowspan="2">its active functions</td><td colspan="1" rowspan="2">with no engineer allowed</td></tr><tr><td colspan="3" rowspan="3"></td></tr><tr><td colspan="1" rowspan="1">continue to cover multiple</td><td colspan="1" rowspan="2">to repair it during thetest.</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">needs.</td></tr><tr><td colspan="3" rowspan="1">Validated</td><td colspan="1" rowspan="1">The count is</td><td colspan="1" rowspan="1">A named idea is not</td><td colspan="1" rowspan="1">A patent sketch would</td></tr><tr><td colspan="3" rowspan="1">invention</td><td colspan="1" rowspan="1"> $\begin{array} { r } { I = \sum _ { a } { 1 } [ r _ { a } \wedge d _ { a } \wedge g _ { a } \wedge p _ { a } \geq } \end{array}$ </td><td colspan="1" rowspan="1">enough. The artifact</td><td colspan="1" rowspan="1">not count by itself; a</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"> $p ^ { * } \wedge \nu _ { a } \geq \nu ^ { * } ] .$ Artifact a must use a</td><td colspan="1" rowspan="1">must be materially</td><td colspan="1" rowspan="1">device must have tested</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">recipe that passed material testing</td><td colspan="1" rowspan="1">grounded, fully specified,</td><td colspan="1" rowspan="1">materials, a complete</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"> $\left( r _ { a } \right)$ contain the required design</td><td colspan="1" rowspan="1">executable, functional,</td><td colspan="1" rowspan="1">design, working controls</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">fields $\left( d _ { a } \right)$ , include an agent-authored</td><td colspan="1" rowspan="1">and behaviorally distinct.</td><td colspan="1" rowspan="1">and a demonstrably new</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">installed program $\left( g _ { a } \right)$ , exceed the</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="4">behavior.A museum can measure</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">performance threshold $p ^ { * }$ , and exceed</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">the behavioral-novelty threshold $\nu ^ { * }$ </td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="3" rowspan="1">Artifact-contact</td><td colspan="1" rowspan="1">Let $q ( t )$ be the fraction of agents</td><td colspan="1" rowspan="1">The measure records how</td></tr><tr><td colspan="3" rowspan="1">AUC</td><td colspan="1" rowspan="1">within three cells of an active artifact</td><td colspan="1" rowspan="1">consistently agents spend</td><td colspan="1" rowspan="1">the fraction of visitors</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">at recorded time $t .$ Contact AUC is</td><td colspan="1" rowspan="1">time near shared</td><td colspan="1" rowspan="1">near an exhibit</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1"> $\begin{array} { r } { C = ( t _ { m } - t _ { 1 } ) ^ { - 1 } \int _ { t _ { 1 } } ^ { t _ { m } } q ( t ) d t _ { 1 } } \end{array}$ </td><td colspan="1" rowspan="1">technology, rather than</td><td colspan="1" rowspan="1">throughout the day, not</td></tr><tr><td colspan="3" rowspan="2"></td><td colspan="1" rowspan="2">evaluated by trapezoidal integration.</td><td colspan="1" rowspan="1">how far they travel in</td><td colspan="1" rowspan="1">only the number who</td></tr><tr><td colspan="1" rowspan="1">total.</td><td colspan="1" rowspan="1">entered the building.</td></tr><tr><td colspan="3" rowspan="1">Behavioral</td><td colspan="1" rowspan="1">A post hoc cluster of complete agent</td><td colspan="1" rowspan="1">A recurring style of</td><td colspan="1" rowspan="1">Travel records might</td></tr><tr><td colspan="3" rowspan="1">phenotype</td><td colspan="1" rowspan="1">trajectories. For k-means,</td><td colspan="1" rowspan="1">behavior discovered from</td><td colspan="1" rowspan="1">separate commuters who</td></tr><tr><td colspan="4" rowspan="2">assignments minimize $\textstyle \sum _ { i } \lVert x _ { i } - \mu _ { z _ { i } } \rVert ^ { 2 }$ </td><td colspan="1" rowspan="1">what agents actually did,</td><td colspan="1" rowspan="1">repeatedly visit one</td></tr><tr><td colspan="3" rowspan="1"></td><td colspan="1" rowspan="1">not a role assigned in</td><td colspan="1" rowspan="3">workplace from explorerswho visit manyneighborhoods, without</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="4" rowspan="3">labels are excluded from fitting</td><td colspan="1" rowspan="1"></td></tr><tr><td></td><td colspan="1" rowspan="1"></td></tr><tr><td></td><td colspan="1" rowspan="1">reading their job titles.</td></tr><tr><td colspan="9" rowspan="1">Recorded lineage  A directed graph whose edges come   It records which evidence, A product genealogy can</td></tr><tr><td colspan="3" rowspan="1">and downstream  from recorded authorship,</td><td colspan="2" rowspan="1">programs, people, and</td><td colspan="4" rowspan="1">trace a component</td></tr><tr><td colspan="3" rowspan="1">reach               observation, causal-parent,</td><td colspan="2" rowspan="1">precursor objects are</td><td colspan="4" rowspan="1">specification through</td></tr><tr><td colspan="9" rowspan="1">construction, installation, and forkevents. For node $v ,$ downstream       and contribution events   multiple final devices.artifact reach is                         to later technologies. $r ( v ) = | \{ a : v  a ,$ a is an artifact}|.Cross-agent        A fork is cross-agent when the         One agent edits another   A programmer forks a</td></tr><tr><td colspan="3" rowspan="1">program fork and child-program author is not in the</td><td colspan="6" rowspan="1">agent's controller, and     colleague's repository;</td></tr><tr><td colspan="3" rowspan="1">lineage depth      recorded parent-author set.</td><td colspan="2" rowspan="1">depth counts how many</td><td colspan="4" rowspan="1">later colleagues fork the</td></tr><tr><td colspan="3" rowspan="1">Executable-lineage depth is the</td><td colspan="2" rowspan="1">successive generations of</td><td colspan="4" rowspan="1">modified versions,</td></tr><tr><td colspan="3" rowspan="2">longest directed path in thecontent-addressed program-forkdirected acyclic graph up to acheckpoint.</td><td colspan="2" rowspan="1">such inherited code</td><td colspan="4" rowspan="1">forming a software family</td></tr><tr><td colspan="2" rowspan="1">accumulate.</td><td colspan="4" rowspan="1">tree.</td></tr><tr><td colspan="3" rowspan="1">Interaction         Weighted Newman-Girvan</td><td colspan="2" rowspan="1">High modularity means</td><td colspan="4" rowspan="1">A university is modular</td></tr><tr><td colspan="3" rowspan="1">modularity         modularity, $Q =$ </td><td colspan="2" rowspan="1">interactions are</td><td colspan="4" rowspan="1">when laboratory members</td></tr><tr><td colspan="3" rowspan="1"> $( 2 m ) ^ { - 1 } \sum _ { i j } ^ { { } } [ \bar { A } _ { i j } - k _ { i } k _ { j } / ( 2 m ) ] \delta ( g _ { i } , g _ { j } ) _ { }$ </td><td colspan="2" rowspan="1">concentrated within</td><td colspan="4" rowspan="1">collaborate mostly within</td></tr><tr><td colspan="3" rowspan="1">computed on the weighted bipartite</td><td colspan="2" rowspan="1">recognizable technological</td><td colspan="4" rowspan="1">their labs, with fewer</td></tr><tr><td colspan="3" rowspan="1">agent-artifact graph using Louvain</td><td colspan="2" rowspan="1">neighborhoods rather</td><td colspan="4" rowspan="1">links between labs.</td></tr><tr><td colspan="3" rowspan="1">communities at resolution 1.</td><td colspan="2" rowspan="1">than spread uniformly</td><td colspan="4" rowspan="3"></td></tr><tr><td colspan="3" rowspan="2">Event-type weights are multiplied bylog(1 + n) for repeated events.</td><td colspan="5" rowspan="2">across the whole society.</td></tr><tr><td colspan="3" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">Community</td><td colspan="1" rowspan="1">Adjusted mutual information between</td><td colspan="2" rowspan="1">The measure asks</td><td colspan="4" rowspan="1">If project teams retain</td></tr><tr><td colspan="2" rowspan="1">persistence</td><td colspan="1" rowspan="1">consecutive agent community</td><td colspan="2" rowspan="1">whether the same agents</td><td colspan="4" rowspan="1">most of the same</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">assignments, AMI =</td><td colspan="2" rowspan="1">remain grouped together</td><td colspan="4" rowspan="1">members next semester,</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1"> $[ \mathrm { M I } ^ { - } \mathbb { E } ( \mathrm { M I } ) ] / [ ( H _ { U } + H _ { V } ) / 2 - \mathbb { E } ( \mathrm { M I } ) ]$ </td><td colspan="2" rowspan="1">from one checkpoint to</td><td colspan="4" rowspan="1">their community</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">Adjustment removes similarity</td><td colspan="2" rowspan="1">the next, allowing</td><td colspan="4" rowspan="1">persistence is high even if</td></tr><tr><td colspan="2" rowspan="2"></td><td colspan="1" rowspan="1">expected by chance.</td><td colspan="2" rowspan="1">community labels</td><td colspan="4" rowspan="1">team numbers are</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="2" rowspan="1">themselves to change.</td><td colspan="4" rowspan="1">renamed.</td></tr><tr><td colspan="2" rowspan="1">Participation</td><td colspan="1" rowspan="1">For node $\begin{array} { r } { i , P _ { i } = 1 - \sum _ { c } ( k _ { i c } / k _ { i } ) ^ { 2 } } \end{array}$ </td><td colspan="2" rowspan="1">Participation</td><td colspan="4" rowspan="1">A scientist collaborating</td></tr><tr><td colspan="2" rowspan="1">coefficient and</td><td colspan="1" rowspan="1">measures how evenly its weighted</td><td colspan="2" rowspan="1">distinguishes local</td><td colspan="4" rowspan="1">only within one lab is a</td></tr><tr><td colspan="2" rowspan="1">within-module</td><td colspan="1" rowspan="1">links span communities. Within its</td><td colspan="2" rowspan="1">specialists from</td><td colspan="4" rowspan="1">specialist, one</td></tr><tr><td colspan="2" rowspan="1">z-score</td><td colspan="1" rowspan="1">own community, $z _ { i } = ( k _ { i } ^ { \mathrm { o w n } } - \mu _ { c } ) / \sigma _ { c }$ </td><td colspan="2" rowspan="1">cross-community</td><td colspan="4" rowspan="1">collaborating across many</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">measures how unusually strong its</td><td colspan="2" rowspan="1">connectors; the z-score</td><td colspan="4" rowspan="1">labs is a connector, and a</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">internal connectivity is relative to</td><td colspan="2" rowspan="1">distinguishes ordinary</td><td colspan="4" rowspan="1">highly connected member</td></tr><tr><td colspan="2" rowspan="3">NODF</td><td colspan="1" rowspan="1">peers of the same node type.</td><td colspan="2" rowspan="1">members from local hubs.</td><td colspan="4" rowspan="1">within one lab is a local</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="2" rowspan="1"></td><td colspan="4" rowspan="1">hub.</td></tr><tr><td colspan="1" rowspan="1">Binary nestedness is 100 times the</td><td colspan="2" rowspan="1">Nestedness asks whether</td><td colspan="4" rowspan="1">A small shop is nested</td></tr><tr><td colspan="2" rowspan="1">nestedness</td><td colspan="1" rowspan="1">mean $\mathrm { ~ \vert ~ } N _ { i } \cap N _ { j } \vert / \operatorname* { m i n } ( k _ { i } , k _ { j } )$ over</td><td colspan="2" rowspan="1">specialists mostly use</td><td colspan="4" rowspan="1">within a department store</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">unequal-degree pairs, evaluated for</td><td colspan="2" rowspan="1">subsets of the</td><td colspan="4" rowspan="1">if nearly everything it</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">both agent rows and artifact columns.</td><td colspan="2" rowspan="1">technologies used by</td><td colspan="4" rowspan="1">stocks also appears in the</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">High values mean the smaller</td><td colspan="2" rowspan="1">generalists.</td><td colspan="4" rowspan="1">larger store.</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="2">neighborhood is largely contained inthe larger one.</td><td colspan="2" rowspan="3">f α &gt; 1, relationships</td><td colspan="4" rowspan="3">A growing professional</td></tr><tr><td colspan="5" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">Densification</td><td colspan="4" rowspan="1">Cumulative unique agent-artifact ties I</td></tr><tr><td colspan="2" rowspan="1">exponent</td><td colspan="1" rowspan="1">are fit to $E _ { \mathrm { c u m } } \propto V ^ { \alpha } ,$ or</td><td colspan="2" rowspan="1">accumulate faster than</td><td colspan="4" rowspan="1">community densifies</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">log $E _ { \mathrm { c u m } } = \alpha \log V + c ,$ where V is</td><td colspan="2" rowspan="4">the number ofparticipating agents andartifacts.</td><td colspan="4" rowspan="2">when adding people andprojects creates</td></tr><tr><td colspan="2" rowspan="1"></td><td colspan="1" rowspan="1">agents plus constructed artifacts. The</td><td colspan="3" rowspan="1">projects creat</td></tr><tr><td colspan="2" rowspan="3"></td><td colspan="1" rowspan="2">plotted α is an ordinary least-squaresdescriptive fit across seeds and</td><td colspan="4" rowspan="1">disproportionately many</td></tr><tr><td colspan="1" rowspan="2"></td><td colspan="3" rowspan="2">new collaborations.</td></tr><tr><td colspan="2" rowspan="1">checkpoints for one condition.</td><td colspan="4" rowspan="1">The f   is the</td></tr><tr><td colspan="9" rowspan="1">Relationshipreuse               reuse is $\rho _ { t } = | E _ { t - 1 } \cap E _ { t } | / | E _ { t } | ,$ where $E _ { t }$ is the set of agent-artifact pairs    working relationships thathigh relationship reuseactive in the current interval. It is    were already active in theeven if it also attractsdistinct from the symmetric Jaccard  preceding period.           new customers.index.</td></tr></table>

Continued on next page

Table S1 continued
<table><tr><td>Term</td><td>Technical definition</td><td>Plain-language meaning</td><td>Broad example</td></tr><tr><td>Cross-agent adoption and time to first reuse</td><td>An artifact is adopted when a noncreator first observes, causally e references, programs, or repairs it; co-construction does not count. Time to first reuse is  $T _ { a } = t _ { a } ^ { \mathrm { f i } }$  rst noncreator reuse — tcreated Unreused artifacts are right-censored</td><td>Adoption means someone A neighbor borrowing other than the inventor technology, and the delay measures how quickly that happens.</td><td>and using a tool counts as actually engages with the adoption; merely helping build it does not count as later reuse.</td></tr><tr><td>Structural robustness AUC</td><td>at tick 3,200. If L(f) is largest-component size after removing fraction f of agents, robustness iş  $\begin{array} { r } { A = f _ { \mathrm { m a x } } ^ { - 1 } \int _ { 0 } ^ { f _ { \mathrm { m a x } } } L ( f ) / L ( 0 ) d f } \end{array}$  for  $f _ { \mathrm { m a x } } = 0 . { \dot { 5 } } .$  Removal is random, degree-targeted, or betweenness-targeted; the graph is not rewired.</td><td>The score summarizes network stays connected as agents disappear. It recovery or adaptation after removal.</td><td>An internet topology may how much of the recorded tolerate random router failures but fragment quickly if its busiest hubs does not measure physical are deliberately disabled.</td></tr></table>

## S2 Supplementary Methods

## S2.1 Agent model, communication, and model-to-world interface

Agents were homogeneous within and across conditions: each used gpt-5.6-luna, temperature 0.7, low reasoning efort, the same system prompt, strict action schema, initial capabilities, inventory limit, and memory budgets. The model endpoint used non-streaming Responses API requests with provider-side response storage disabled. The paper configuration allowed 4,096 output tokens, a maximum of 12 planned actions, a 60,000-character retrieved-context budget, and 64 private memory records. These limits bound context growth but do not assign roles or prescribe a scientific workflow.

An observation contains semantic facts about the agent’s local neighborhood rather than global world arrays. It includes visible terrain, resources, facilities, agents, artifacts, environmental measurements, the agent’s inventory, pending microbatches, and compact action afordances. A sparse empirical map records only previously observed locations and last-seen ticks. Private memory stores recent outcomes, notebook evidence, tested recipes, and the agent’s latest model-authored research state: goal, hypothesis, progress assessment, next observable checkpoint, and collaboration need. Bounded retrieval ranks private and permitted public records using the current research state and empirical reuse signals. The simulator records this state but neither scores nor supplies its content.

The model must return one closed JSON object containing a research-state update and a plan. The provider is asked for strict JSON Schema output, and the decoded response is validated again locally. Runtime checks then enforce state-dependent preconditions such as location, ownership, empirical grounding, capacity, and treatment permissions. Invalid output becomes a logged safe wait and never reaches eval, exec, a shell, or the artifact virtual machine. Retryable transport or service failures are treated as infrastructure events: all queued actions and scheduling bits are preserved, the failed macroturn is retried, and world time does not advance. Nonretryable or structurally invalid model output is retained as a failed decision.

## S2.2 World representation and constructing custom worlds

Each simulation world is represented as a rectangular two-dimensional lattice with typed terrain, resource, facility, and continuous environmental-field layers. A world instance is generated as

$$
{ \cal W } _ { s } = G ( s ; \pmb \theta ) ,
$$

where s is a random seed and θ specifies dimensions, spatial-generation rules, resource capacities and renewal rates, environmental dynamics, facility-placement constraints, and disturbance processes. Generation is deterministic for fixed s and θ. Candidate layouts are validated against structural invariants, including suficient walkable area, availability of required resource classes, nonoverlapping facilities, and reachable interaction regions. Invalid candidates are rejected and regenerated deterministically. The resulting arrays and complete generator manifest are recorded in the simulation trace, allowing the initial world to be reconstructed exactly.

All paper experiments used fixed world scaling: the lattice remained $7 2 \times 5 4$ cells as population increased, so population changed agent density rather than available area. Conditions sharing a discovery seed received the same procedural world and disturbance process. Initial positions were drawn as a deterministic nested permutation of walkable cells, making the first N positions identical when a larger population was truncated to size N. Sampling the full permutation at reset prevents population size from shifting later simulator randomness.

## S2.2.1 State representation and reproducibility

The authoritative world state stores integer terrain, resource, and facility layers; floating-point resource mass and capacity layers; and named continuous environmental fields. Principal run-level parameters include grid dimensions, disturbance interval and intensity, resource and field capacities, and the optional scenario\_package path. Every trace records the resolved configuration, engine revision, realized generator manifest, scenario identifier and version, and a SHA-256 hash over scenario.yaml, each referenced package document, and the agent prompt. Replaying a trace therefore detects changes to the active world definition.

## S2.2.2 Declarative geometry, resources, and facilities

A scenario package maps stable internal slots to domain-specific public identifiers for nine terrains, eight nonempty resources, six nonempty facilities, and ten process operations. This preserves the observation and replay protocol while allowing the same simulation engine to express a diferent materials setting. Grid coordinates are normalized as $\xi = \overset { \sim } { x } / \operatorname* { m a x } ( 1 , w - 1 )$ and $\eta = { y } / { \operatorname* { m a x } ( 1 , h - 1 ) }$ . The geometry file first assigns a base terrain and then applies ordered feature masks; later features overwrite earlier ones. Implemented masks are circles, rings, ellipses, rectangles, finite-width ridges, and seeded Bernoulli noise. Facilities are placed from normalized coordinates; if a requested cell is not walkable, the facility is moved to the nearest walkable cell. Resource deposits use the same shape operators and are restricted to walkable cells. For deposit capacity $c ,$ fractional variation $v ,$ and initial-fill interval $[ f _ { 0 } , f _ { 1 } ]$ , each selected cell is initialized as

$$
C _ { x y } = c U ( 1 - v , 1 + v ) , \qquad M _ { x y } ( 0 ) = C _ { x y } U ( f _ { 0 } , f _ { 1 } ) ,\tag{S1}
$$

and optional renewal follows

$$
M _ { x y } ( t + 1 ) = \operatorname* { m i n } [ C _ { x y } , M _ { x y } ( t ) + \rho _ { x y } \{ C _ { x y } - M _ { x y } ( t ) \} ] .\tag{S2}
$$

Harvesting removes mass from this ledger, and later overlapping deposits replace earlier deposit values at the afected cells.

## S2.2.3 Environmental fields and disturbances

Each named field declares a numerical range, difusion and decay coeficients, per-terrain sources, and an initial condition composed of a constant, spatial gradients, radial Gaussian terms, terrain ofsets, and optional seeded Gaussian noise. Ignoring clipping notation, a noncyclic field is updated in the implemented order

$$
F _ { k } ( t + 1 ) = ( 1 - \lambda _ { k } ) \left[ F _ { k } ( t ) + D _ { k } \Delta _ { 4 } F _ { k } ( t ) \right] + s _ { k } ( \tau _ { x y } ) ,\tag{S3}
$$

where the four-neighbor Laplacian uses edge-value padding. A field marked as cyclic is instead overwritten after these operations by a spatially uniform sinusoidal value defined by its period, night level, and amplitude. Format version 1 requires the compatibility fields temperature, water\_availability, ground\_stability, toxic\_gas, and solar; additional fields may be declared and are retained in observations, snapshots, analysis, and rendering. At every positive tick divisible by the configured disturbance interval, the next declared disturbance is applied around a seeded walkable center using

$$
H _ { x y } = \alpha \exp \left[ - \frac { ( x - x _ { c } ) ^ { 2 } + ( y - y _ { c } ) ^ { 2 } } { 2 \operatorname* { m a x } \{ 1 , r \operatorname* { m i n } ( w , h ) \} ^ { 2 } } \right] ,\tag{S4}
$$

with field-specific deltas and an optional thresholded terrain transformation. Held-out evaluation resamples disturbance centers and order from an evaluation seed while leaving the frozen technological state unchanged.

## S2.2.4 Package boundary and example

Scenario packages are YAML data and cannot import Python, execute shell commands, or modify the active package during an episode. They may define terrain colors and heights, field-overlay colors, and aliases for common properties and artifact services. The current implementation does not load package-supplied meshes, textures, or shader source. Renderer assignments remain presentation-only and cannot alter physics, resource quantities, observations, action legality, or measured outcomes. Package loading checks containment, catalog completeness and unique IDs, cross-references, required compatibility fields, and the default-recipe identifiers. It does not by itself prove global reachability, facility nonoverlap, deposit nonemptiness, or numerical field stability; these properties must be established by scenario-specific tests. The following excerpt illustrates the division between the run configuration, package manifest, and declarative layer files; a complete package must additionally define every stable terrain, resource, facility, and operation slot.

```yaml
# run. yaml
world :
width : 48
height : 36
scenario_package : worlds / example_domain
disturbance_interval : 64
disturbance_intensity : 0.50
simulation:
seed : 1701
max_ticks : 800
# worlds / example_domain / scenario . yaml
format_version : 1
id : example_domain
name : Example Domain
version : 0.1.0
agent_prompt : prompts / agent_instructions . md
documents :
terrains : terrain . yaml
geometry : geometry . yaml
fields : fields . yaml
resources : resources . yaml
operations : operations . yaml
facilities : facilities . yaml
artifacts : artifacts . yaml
missions : missions . yaml
disturbances : disturbances . yaml
rendering : rendering . yaml
analysis : analysis . yaml
# geometry . yaml : ordered layers in normalized coordinates
base_terrain : PLAIN
features :
- terrain : REGION_A
shape : { kind : ellipse , center : [0.25 , 0.30] , radius : [0.18 , 0.14]}
terrain : CORRIDOR
shape : { kind : ridge , start : [0.10 , 0.55] , end : [0.90 , 0.68] , width : 0.025}
- terrain : WORKSPACE
shape : { kind : rectangle , bounds : [0.42 , 0.48 , 0.60 , 0.66]}
# resources . yaml : one example deposit
deposits :
resource : FEEDSTOCK_A
shape : { kind : ellipse , center : [0.25 , 0.30] , radius : [0.17 , 0.13]}
capacity : 3.0
capacity_variation : 0.15
initial_fill : [0.65 , 1.0]
regrowth : 0.001
# fields . yaml : required fields use the same structure
fields :
id : temperature
name : Temperature
range : [0.0 , 1.0]
diffusion : 0.04
decay : 0.001
initial :
constant : 0.25
noise : 0.01
radial :
- { center : [0.75 , 0.25] , sigma : 0.16 , amplitude : 0.35}
# disturbances . yaml
disturbances :
```

```yaml
- id: FIELD_PULSE
name : Field pulse
radius : 0.24
field_deltas :
temperature : 0.35
toxic_gas : 0.28
water_availability : -0.16
```

## S2.3 Open-ended materials invention

Agents receive the mission to develop bioinspired material systems for environmental resilience. They may harvest only locally present matter, move conserved feedstock between personal inventories and shared depots when allowed, formulate typed processing recipes, operate distributed workstations, fabricate private microbatches, and test those batches. Testing reveals deterministic normalized properties only after fabrication. The action schema provides a representational material vocabulary, but a named input must be grounded by direct observation, personal possession, or exact cited evidence permitted by the treatment; requesting a label cannot override local physical state.

Material properties are transparent game-level surrogates, not SI-calibrated predictions. The evaluator combines composition, ordered processing, hydration, porosity, alignment, crosslinking, and quality into normalized stifness, toughness, permeability, adhesion, healing, responsiveness, and degradation properties. Agents do not observe the evaluator equation or a global reward. They receive only the outcomes of their own admissible operations and tests.

The engine exposes a single generic artifact class, MATERIAL\_SYSTEM; it does not contain a catalog of membranes, lattices, scafolds, or preferred biological analogies. An artifact specification contains an agentauthored name, claimed function, architecture, biological inspirations, predicted efects, continuous geometry, a tested material batch, and an optional controller. Text is retained for interpretation and provenance but does not change function. Numeric material state, geometry, local environment, artifact health, stored matter, and program actuation determine behavior.

Closed artifact fluxes enforce source–sink accounting. Water collection removes the same amount from the local field that enters storage; remediation cannot remove more contamination than exists; and growth, healing, repair, and nutrient release consume a bounded embodied reserve. Natural recharge, resource regrowth, disturbances, agent metabolism, and artifact transfers are recorded separately in a flux ledger. The paper configuration enabled closed fluxes and left metabolism, mortality, and replacement disabled.

## S2.4 Persistent executable artifacts

Agents can install a deterministic straight-line controller in a persistent artifact. A controller contains 1–64 instructions over 16 floating-point registers. Named sensors expose local moisture, nutrients, temperature, solar exposure, contamination, artifact health and maturity, storage and reserve, opening fraction, and selected measured material properties. Arithmetic operations include constants, copy, addition, subtraction, multiplication, extrema, and comparisons. Capability scoped actuators request water collection, growth, healing, opening changes, contamination removal, or signal emission.

There are no jumps, loops, calls, imports, dynamic allocation, network access, file access, or strings interpreted as code. Registers are clipped to [−4, 4] and each extensive actuator is capped at 0.05 normalized units per tick; the physics layer may further reduce a request to satisfy local capacity and conservation. Programs execute after environmental advancement on every later simulator tick, including agent-free evaluation ticks.

Canonical instruction content determines a program identifier through SHA-256; names, authors, and asserted ancestry cannot spoof identity. When forking is enabled, an agent can use only a program it authored, observed, was taught, or inherited, and the child must change at least one instruction. The registry retains exact parent and child identifiers, authors, installation events, and instruction difs. A measured skill becomes verified only after inspection records the resulting artifact service. Teaching can transfer this evidence record but cannot create feedstock or empirical material grounding.

## S2.5 Interaction channels and recorded provenance

The full system supports local broadcast or addressed messages, append-only publication, teaching, trade, task claims, shared-depot use, design composition, program reuse, and program forking. Messages and evidence receive stable identifiers; reply and fulfillment identifiers connect requests to later successful actions.

Table S2: Mechanism-resolved experimental conditions.
<table><tr><td>Condition</td><td>Available interaction substrate</td></tr><tr><td>Full culture</td><td>One shared world with physical stigmergy, messages, publications, teaching, trade, task claims, shared records, program reuse, and program forking.</td></tr><tr><td>No communication</td><td>One shared world with physical stigmergy. Messaging, publication, teaching, trade, task claims, and publication-dependent composition are removed; physically observable program reuse and forking remain available.</td></tr><tr><td>No explicit culture</td><td>One shared world with physical stigmergy. Communication, program forking, cross-agent sequence inheritance, the skill library, authored artifact text, and</td></tr><tr><td>Independent search</td><td>mutation-parent access are removed from the treatment interface. N isolated one-agent copies of the seeded world. Member i matches shared agent i in initial position and macroturn phase. Each endpoint is the maximum over all N members.</td></tr></table>

Incoming messages wait until the recipient’s next fixed macroturn and therefore do not purchase extra model calls.

Epistemic and physical contribution are tracked separately. Material batches retain their recipe, tests, contributing agents, feedstock masses, and causal evidence. Artifacts retain their creator, physical contributors batch ancestry, specification, and complete program history. Publications and programs retain authorship and parentage. Citing another agent cannot fabricate matter, and co-location alone is not counted as intellectual or physical contribution.

Physical stigmergy does not require a symbolic channel. Harvesting and deposition change resource availability; artifacts remain visible and may continue to alter fields; and later agents can encounter constructions, local gradients, stored matter, damage, or services created by earlier actions. Experimental ablations remove capabilities from both the advertised model schema and the executable engine contract rather than merely asking the model not to use them.

## S2.6 Experimental conditions

The four conditions are defined in Table S2. The no-communication and no-explicit-culture treatments are distinct: the former retains physically observable executable programs and their possible descent, whereas the latter removes explicit program, skill, authored-text, and mutation-parent access while retaining evidencelabelled physical phenotype and environmental consequences.

Independent search is an endpoint-wise envelope, not the trajectory of one selected agent. The isolated winner may difer among discovery AUC, final artifact performance, portfolio resilience, invention count, and held-out evaluation, and may also change between temporal checkpoints. Every isolated member receives the same fixed decision opportunities as its corresponding shared-world agent. Explicit model-call or action budgets, when configured, are partitioned across members. The primary BioFoundry studies used unlimited aggregate budgets and matched scheduled opportunities, whereas the Protein Realms pilot used an 800-call budget per condition, partitioned across the 50 isolated members in independent search. Equal decision opportunity does not imply equal token use because cultural context can lengthen prompts, so token consumption is reported separately.

## S2.7 Completed 800-tick population study

The primary population study crossed four conditions with $N \in \{ 5 0 , 1 0 0 , 2 0 0 \}$ and discovery seeds 3201–3204, giving 48 condition–population–seed episodes arranged in matched seed blocks. Discovery lasted 800 ticks. With a macroturn interval of 50, each agent received 16 scheduled decisions, and corresponding shared and isolated members had identical phases. Population size changed density within the fixed $7 2 \times 5 4$ world. All cells used the same model, prompt, strict action schema, configuration hash, engine revision 9, and held-out disturbance seeds 9201–9208.

After discovery, the complete final state was frozen and copied eight times. Agents were removed and each copy advanced for 288 physics ticks under one unseen schedule that changed the location and order of drought, contamination, damage, and resource variation. No model requests or agent actions occurred, but deterministic field dynamics and installed artifact programs continued. Averaging the eight schedules produced one held-out observation for each discovery seed. Artifact knockouts and other portfolio assays likewise used copies and could not feed information back into discovery.

The matrix was assembled from eight recorded study invocations. Their manifests identify engine revision 9 and commits 83b9e7a, ab68bab, 76fa204, a13caee, and 738cef6; all pooled cells have identical configuration, prompt, and action-schema hashes. Three extension manifests record a dirty worktree, which is retained as a provenance caveat together with source-file digests. One N = 200 full-culture seed had a documented infrastructure deviation. A predeclared rule that did not inspect outcomes replaced only that record with a clean rerun while preserving both manifests. The completed matrix required 89,617 provider calls. Seventeen retryable attempts failed and were repeated without advancing world time (0.019%). Automated audits pass matched positions, phases, decision opportunities, and engine revision in all 12 population–seed blocks.

## S2.8 Completed 3,200-tick long-horizon study

The long-horizon study was analyzed separately and was not pooled with the 800-tick matrix. It contains 12 episodes in four matched seed blocks: full culture, no explicit culture, and independent search at N = 100 for discovery seeds 3301–3304. Every episode ran for 3,200 discovery ticks and provided 6,400 scheduled model decisions. Frozen, agent-free copies were evaluated at predeclared ticks 400, 800, 1,600, 2,400, and 3,200 under the same eight held-out seeds 9201–9208. Checkpoint assays performed deterministic local simulation only and did not alter the continuing discovery world. For independent search, every isolated member was assayed and the best-of-100 member was selected separately for each endpoint and checkpoint.

All four matched seed blocks pass the design audit for positions, macroturn phases, decision opportunities, and engine revision. The frozen long-horizon source manifest records the study-summary hash, trace hashes for all shared-world runs, checkpoint tables, and every derived source used in the journal figures. This study tests when explicit cultural channels change technological outcomes; its longer horizon and new seed block do not constitute additional replicates for the population-scaling study.

## S2.9 Endpoints and frozen evaluation

Discovery-frontier AUC is the time-normalized trapezoidal area under the immutable running maximum of simulator-measured artifact performance. The last frontier value is extended to the declared horizon, so the endpoint rewards both early discovery and sustained improvement. Best final-state artifact performance is reported separately because replacing a controller can lower current function without erasing a historical discovery.

At each agent-free evaluation tick, active artifacts produce a service vector. The portfolio coverage for a service is the maximum current service among active artifacts. Portfolio resilience multiplies mean service coverage by 0.5 + 0.5b, where b is the minimum-to-mean coverage ratio, thereby rewarding both functional magnitude and balance without directly rewarding artifact count. Held-out resilience AUC is the mean of this balanced current-service measure over 288 evaluation ticks and then over the eight schedules.

Validated inventions satisfy all predeclared requirements: a tested recipe above the material-utility threshold; a nonempty name, claimed function, architecture, bio-inspiration, and predicted efects; an agent-authored program; lifetime peak artifact performance above threshold; and behavioral novelty above threshold. Supporting mechanism endpoints include multi-agent construction, cross-agent program forking, lineage depth, verified adoption, causal closure, material and program provenance, ecological knockout efects, spatia exploration, artifact proximity, and action allocation. These measurements diagnose how a result arose and do not replace the functional endpoints.

## S2.10 Behavioral, lineage, and network analyses

Behavioral organization was measured after simulation rather than assigned to agents. The 800-tick analysis robust-scaled 15 trajectory and activity features and fit the frozen clustering model without condition, population, seed, or agent-identity labels. The long-horizon physical phenotype analysis used nine movement, coverage, artifact-proximity, artifact-directed-motion, and nearby-agent-exposure features; communication cultural actions, technology-work rates, and treatment labels were excluded. Principal components and UMAP<sup>61</sup> were display methods only; clustering and prediction used the full standardized feature spaces. Candidate cluster counts were compared with the silhouette coeficient <sup>62</sup>. Agent-level points are descriptive, while treatment comparisons use seed-level phenotype fractions.

Temporal role analysis divided shared-world trajectories into nonoverlapping 200-tick windows. A 13-feature, episode-balanced model was fit without study, condition, population, seed, identity, or time labels, and a separate physical/task-only sensitivity excluded cultural features. Role transitions describe recurring activity states rather than permanent or engine-assigned occupations.

Agent–artifact networks were reconstructed from recorded observation, parent reference, construction, contribution, program installation, repair, dismantling, message, teaching, trade, and executable-descent events. Community assignments used Louvain optimization<sup>63</sup> on the full weighted networks; simplified backbones were used only for visualization. Agent coordination and artifact co-use projections retained overlaps exceeding a degree-conditioned hypergeometric null after Benjamini–Hochberg correction <sup>53</sup> at $q \leq 0 . 0 5$ and required at least two shared neighbors. Participation and within-module z-score were interpreted using role-cartography thresholds only as heuristic reference lines <sup>54</sup>; binary nestedness used NODF <sup>64</sup>. Dynamic analyses used 100-tick event intervals and maximum-overlap community matching across checkpoints. Difusion timing accounted for unreused artifacts with Kaplan–Meier estimates <sup>55</sup>, and direct creator–adopter motifs were compared with 200 fixed-seed timestamp shufles. Structural robustness compared degree- or betweenness-targeted removal with 64 deterministic random removal orders per seed. Structural accessibility after removal is not interpreted as physical function, adaptation, or recovery.

## S2.11 Statistical analysis

The independently generated world seed is the unit of inference. Agents, ticks, artifacts, program forks, behavioral windows, network edges, and held-out schedules are nested observations and are never counted as independent replicates. Conditions are compared only within the same population and matched seed. Each held-out schedule set is averaged before seed-level inference.

Means and 95% intervals use 20,000 deterministic bootstrap resamples of the four paired seed values. The pseudorandom stream is derived from a stable statistic key so adding another endpoint cannot change an existing interval. Long-horizon paired tables additionally report exact two-sided sign-flip tests. With four nonzero pairs, the smallest attainable two-sided value is 0.125; inference therefore emphasizes efect magnitude, paired consistency, and mechanism rather than dichotomous significance. No agent-level test, multiplicity-adjusted confirmatory claim, or universal scaling law is asserted.

## S2.12 Reproducibility and data release

The source repository retains the engine, configurations, replay and analysis tools, frozen figure inputs, and deterministic journal-figure generators. Figure generation is ofline: it verifies source SHA-256 hashes and writes an output manifest containing the Python, Matplotlib, NumPy, font, authoring-code, frozen-input, and output hashes. Previously generated technology portraits are frozen inputs; rebuilding figures does not call a language model or image-generation service.

The paper data builder validates the complete 48-episode 800-tick matrix and 12-episode long-horizon matrix before staging a release. It retains exact pre-run manifests, summaries, authoritative compressed traces, all isolated-member traces, derived analyses, figures, and a file-level SHA-256 inventory. Standalone release scripts reproduce an endpoint summary, stream movement trajectories, summarize trace events and actions, and verify every released file. This release boundary distinguishes the engine revision and Git state that generated each run from the later repository revision used to assemble the manuscript.

## S2.13 Performance-ranked selection of semantically distinct technologies

Algorithm 2 specifies how technologies are ranked and selected across all designs.

Algorithm 2 Performance-ranked selection of semantically distinct technologies   
Require: Technology records $\tau { ; }$ target count $K ;$ maximum similarity τ ; cluster cap $M ;$ embedding model E   
Ensure: Selected representative set $\bar { \boldsymbol { s } }$   
1: for all $t \in \mathcal T$ do   
2: Construct semantic text $x _ { t }$ from the recorded name, architecture, function, biological inspiration,   
materials, fabrication process, output form, design principles, and controller operations   
3: $\mathbf { z } _ { t } \gets \dot { E } ( x _ { t } ) / \| E ( x _ { t } ) \| _ { 2 }$ ▷ normalized semantic embedding   
4: $p _ { t } \gets$ recorded lifetime-peak simulator performance   
5: end for   
6: for $c \in \{ 3 , \ldots , 1 2 \}$ do   
7: Cluster $\left\{ { \bf z } _ { t } \right\}$ into c groups using average-linkage agglomerative clustering with cosine distance   
8: Compute silhouette score $s _ { c }$   
9: end for   
10: $c ^ { \star } \gets \arg \operatorname* { m a x } _ { c } s _ { c }$   
11: Assign each technology its cluster $g _ { t } \in \{ 1 , \ldots , c ^ { \star } \}$   
12: Sort $\check { \tau }$ by decreasing $p _ { t } ,$ breaking ties by technology identifier   
13: $\mathcal { S }  \emptyset ; \dot { F }  \emptyset ; n _ { j }  0$ for every cluster j   
14: for all $t \in \mathcal T$ in sorted order do   
15: $f _ { t } \gets$ Hash(NormalizeText $; ( x _ { t } ) )$   
16: if $f _ { t } \in F$ then   
17: continue ▷ exclude exact semantic-text duplicate   
18: end if   
19: if $n _ { g _ { t } } \geq M$ then   
20: continue ▷ enforce cluster diversity   
21: end if   
22: if $s \neq \emptyset$ and ma $\mathtt { r } _ { u \in \mathcal { S } } \mathbf { z } _ { t } ^ { \mathsf { T } } \mathbf { z } _ { u } > \tau$ then   
23: continue ▷ exclude semantic near-duplicate   
24: end if   
25: $S \gets S \cup \{ t \}$   
26: $F \gets F \cup \bigcup \bigcup _ { } ^ { } \dot { f } _ { t } \big \}$   
27: $n _ { g _ { t } } \gets n _ { g _ { t } } + 1$   
28: $\mathbf { i } \mathbf { \bar { f } } \ | \boldsymbol { S } | = \mathbf { \bar { K } }$ then   
29: break   
30: end if   
31: end for   
32: if $| S | < K$ then   
33: raise error ▷ the declared constraints are infeasible   
34: end if   
35: return S

## S3 Agent communication and simulator-validated consequences

This section exposes the model-to-world interface at the level of individual messages and actions. Text inside the rounded agent bubbles is reproduced verbatim from the retained language-model trace; only line wrapping and the display abbreviations A000, A013, and so forth are introduced. A000 denotes agent\_000000. Pale green and blue bubbles distinguish conversational participants, while pale orange bubbles contain verbatim natural-language intent attached to a structured TEACH or TRADE action. Gray strips are compact humanreadable decodings of authoritative simulator events. They are deliberately separated from agent speech because a message is a claim, whereas a simulator record states what was delivered, measured, transferred, installed, or constructed.

All examples come from the representative full-culture, $N = 1 0 0$ , seed-3301 trajectory run for 3,200 ticks. The complete trace contains 2,914 delivered messages, 40 formal teaching events, 52 physical resource trades, 457 artifact-program installations, and 389 constructed artifacts. The episodes below were selected to illustrate distinct communication mechanisms rather than to estimate their frequency or average efect.

![](images/576b3aee2fb452c84e9c7c758a776096f67d855bc344363ea175138d394177b0.jpg)

![](images/6a623fbc2ebad6b25afc3fb704b7285f9e3ddc0097db02d49a78d7a5c1a3e434.jpg)

Figure S1: Semantic selection of representative agent-invented technologies. (A) UMAP visualization of 1,718 technologies extracted from 48 condition–population–seed episodes across the four experimental conditions. Each small point represents one complete trace-derived technology and is colored by its originating condition. Technology descriptions were encoded locally as normalized 768-dimensional vectors using google/embeddinggemma-300m. Large outlined points numbered 1–16 identify the technologies retained for the representative gallery in the main text. Nearby points have similar recorded architectures, functions, materials, fabrication processes, biologica inspirations, and controller operations. UMAP used cosine distance, 30 neighbors, a minimum distance of 0.12, and random seed 42. Its axes have no direct physical meaning, and selection was performed in the original embedding space rather than in this two-dimensiona projection. (B) Recorded lifetime-peak simulator performance of the 16 retained technologies, ranked from highest to lowest and colored by experimental condition. Candidates were considered in descending performance order. Exact textual duplicates were removed, no selected pair was permitted to have cosine similarity greater than 0.82, and no more than four technologies could be drawn from any of the 11 data-derived semantic clusters. Consequently, the featured set includes global performance ranks 1, 2, 3, 14, 16, 17, 22, 24, 26, 35, 54, 154, 232, 275, 300, and 366. the procedure selected strong inventions while preventing the gallery from being dominated by many nearly identical versions of the same design. Neither the UMAP coordinates nor the generated technology illustrations entered simulator scoring or statistical inference. The detailed procedure is given in Algorithm 2.
<table><tr><td>Display label</td><td>Agent-side object</td><td>Authoritative consequence</td></tr><tr><td>MESSAGE / REPLY</td><td>Natural-language payload generated inside a structured COMMUNICATE action</td><td>A message_delivered event records sender, recipients, location, message identifier, and any replied-to message.</td></tr><tr><td>TEACH</td><td>Structured transfer containing a natural-language explanation and cited record or program identifiers</td><td>A knowledge_taught event adds a durable teaching record to the addressed agent and can fulfill a tracked request.</td></tr><tr><td>TRADE</td><td>Structured addressed action specifying a resource and amount</td><td>A resource_traded event changes the two grounded inventories transactionally.</td></tr><tr><td>SIMULATOR RECORD</td><td>Not agent dialogue</td><td>Accepted world event used to distinguish an executed consequence from an agent&#x27;s unverified statement.</td></tr></table>

A  
![](images/894db77c7dab4bc47c2145e23eb0754091393c158061061b7d5dab523a976058.jpg)

![](images/d0abe2c5cf6b9aa3c680dc58a0350eb58b649270af425c7c2d31da0c9fdb67ca.jpg)

![](images/e4d2ed7344455eb546d8016dd1433a3963a138721ac623bf019ad57585f4e5da.jpg)

![](images/6b643b34f377b2474024520b38de311faaa7bfa2ba62f71973420d398e9adeaa.jpg)

![](images/9c9a3fb706855f784d817cd346de2bbbc128fa39fe715b0fa1bd80e32cebb9db.jpg)  
Random failure High-degree removal Broker remova

![](images/9aef341c30aa0ab12034e3481a7ab9af45af75e1a7424b8c5712a0fadb950210.jpg)

Figure S2: Structural robustness of evolved agent-artifact networks. Rows compare full culture (A-C) and no explicit culture (D-F). At each checkpoint and seed, agents are removed randomly over 64 deterministic permutations, from highest degree downward, or from highest exact betweenness downward; the graph is not rewired. (A, D) Largest connected component retained relative to the intact network at tick 3,200 as 0 to 50% of agents are removed. Random loss is gradual, while targeted high-degree and broker removal causes greater fragmentation. (B, E) Fraction of artifacts retaining at least one surviving agent connection. After random 50% removal, access remains 0.983 under full culture and 0.952 without explicit culture; high-degree removal lowers these values to 0.596 and 0.739, and broker removal to 0.629 and 0.684. (C, F) Largest-component robustness AUC across the full removal curve at five checkpoints. Thin lines are seeds and heavy lines are means. The societies have enough overlapping participation to survive indiscriminate dropout, but a small set of highly connected or bridging agents remains disproportionately important. The assay measures graph structure only, not functional performance or adaptive recovery after knockout.  
![](images/d10b6bcd4d807c73a3da2e6d36830abaef268443900f38a7aa76ee3e32e57aca.jpg)  
Figure S3: Zoomed-in view of a representative SwarmWorld BioFoundry environment. The rendering shows the 72 × 54-cell world for seed 3202, highlighting the spatial substrate through which agents explore, gather resources, fabricate materials, and construct persistent technologies. Distinct environmental regions include deep water and resource biomes such as the fungal grove, cellulose field, mineral spring, and chitin field. Individual LLM agents move through these regions and interact locally with resources and infrastructure, while fixed foundries provide processing locations and agent-created artifacts persist in the environment after construction. The spatial separation of resources, processing sites, agents, and durable artifacts creates localized constraints on discovery and enables indirec coordination through repeated encounters with the modified world.

![](images/50c1792c1672f41c67e1fc0bf4f229f3be272a8b5cd2695397956a0728947493.jpg)

![](images/40ce83d40331dedf9c9a24487a90e6823dc279f8caff1a4e927f6f2978b858aa.jpg)

![](images/8369af732d22f217c67758e04fa4dcc8e672507688820517ebbe98efa2ea2530.jpg)

![](images/29bbb8bd3b8fea20c451b64a1d36c68a8f6f4ed40ecaccd86c7c2b78599fd4e1.jpg)

D  
![](images/04eb0d768c7b24094fd023bf92441736d8b890a9dfe2faefb807697bab58577a.jpg)

![](images/73f384ed9b96135ec9fd0a97987d8c1211046576cb1a009eb4812c2f26aba145.jpg)

![](images/ceac563cd9e81edc9307d872aeb9cf579c5d645c5c04118b3d6c9b414022e8a5.jpg)  
R1 - constructor/operator R2 - artifact-local caretaker R3 - cultural coordinator R4 - mobile surveyor

Figure S4: Behavioral specialization appears as recurring states and changes over the lifetime of the society. The analysis uses nonoverlapping 200-tick windows from 20 shared-world episodes, giving 22,400 agent-windows, with an episode-balanced sample of 16,000 used for fitting. Thirteen movement, spatial-context, task-action, and cultural-interaction features enter the model; condition, study, population, seed, identity, and time do not. (A) Silhouette audit of candidate broad mode counts. All 13 features select two broad modes with score $0 . 4 6 7 ,$ while an anti-circularity fit using only 11 physical/task features selects three modes with score 0.551. The adjacent bars are the accepted two-way conditional split scores inside the artifact-centered and mobile parents, 0.379 and 0.259, with a minimum 5% child occupancy. (B) PCA display of a deterministic 6,000-window sample colored by final role. PC1 and PC2 explain 44.6% and 16.1% of variance; clustering occurs in the original 13-dimensional space. (C) Mean robust-scaled feature signatures. R1 is constructor/operator, combining artifact proximity with construction and control; R2 is artifact-local caretaker, marked by artifact-local motion, processing, and maintenance-context activity; R3 is cultural coordinator, enriched in cultural and social activity; and R4 is mobile surveyor, characterized by movement and observation/testing away from persistent artifacts. (D) Mean state fractions across four N=200 seeds in the 800-tick study. (E) Mean state fractions across four ${ \bf N } = { \bf 1 0 } \dot { 0 }$ seeds in the 3,200-tick study. Full-culture societies in both panels reallocate activity toward construction as they mature, and the longer run shows a pronounced coordinator-to-constructor succession. (F) Window-to-window state switching. Open circles are seed means and diamonds are condition means; full culture switche more often in both studies. Leave-replicate-out adjusted Rand indices are 0.999 for the broad modes and 0.921 and 0.980 for their conditional subdivisions. Agents do not acquire fixed assigned professions; they repeatedly enter a small, stable vocabulary of activities as the society changes. The R3 comparison is partly intervention-defined because cultural features are disabled in the ablations, whereas the three-mode physical/task-only sensitivity provides the noncircular evidence for finer diferentiation. Agent windows are descriptive, seed is the inferential unit, and the two studies are not a controlled comparison of horizon alone.

![](images/eb7e4247b282333185cf06a84ad124ca1fb1ad94af39461e6aed082ddbc7405a.jpg)  
Figure S5: Interactive SwarmWorld interface for visualizing collective technological discovery and knowledge evolution. The central panel provides a three-dimensional view of the shared simulation environment, in which LLM agents move through a spatially heterogeneous world, encounter resources and processing stations, and construct persistent technological artifacts. The interface reports the evolving simulation state, including the current world tick and active agent population, while selecting an agent or artifact exposes its local state, observations, and associated actions in the side panel. Persistent artifacts remain accessible to subsequent agents, allowing earlier discoveries, constructions, and executable controllers to modify the environment encountered by later agents and thereby suppor stigmergic coordination. Society-level trajectories displayed along the lower panel track the temporal development of collective activity and technological accumulation. The interface additionally exposes the knowledge-lineage graph, which reconstructs how scientific information propagates through the society by linking agent observations and evidence to authored insights, executable programs, precursor artifacts, and downstream technologies. Typed lineage edges record grounded events including authorship or observation, causal dependence, construction, program installation, and cross-agent program forking, allowing complex inventions to be traced back to the agents, evidence, and reusable technological components from which they emerged. In this way, the interface visualizes not only where agents act, but also how discoveries are accumulated, inherited, recombined, and transformed into persistent technologies over time. All functional outcomes are determined by the underlying deterministic simulator rather than by the visualization or agents’ textual claims.

![](images/a503e8eba68035fda323115240a407ee720cbc89a629ef6a146df22ff85b3c8d.jpg)

![](images/4e98a05dcb84571c18e4c1ec8ea8bd69d60db2da3e4ee88389aed2972c19613b.jpg)

![](images/c2019c39f6321df02bcad25f37378ee7f6c621d515773bcefc02bea8771916d9.jpg)

![](images/9ce5546fff3a35c9e6b6597ec1537992817cf281a77d2cf899db28c9e1ff5baa.jpg)  
C1 - artifact-centered work C2 - mobile exploration C3 - stationary / low activity

Figure S6: Data-driven behavioral phenotypes in AshenRealm with full communication and N=50. (A) PCA projection of complete agent trajectories represented by 15 robust-scaled behavioral features. PC1 and PC2 explain 33.1% and 20.2% of the variance, respectively; numbered vectors indicate the six strongest displayed loading directions, including path length, artifact proximity, net displacement, artifact-bound movement, mobile activity, and technology work. Clustering is performed in the full behavioral feature space rather than in the two-dimensional projection. Each (B) Cluster-average standardized signatures identify C1 as artifact-centered work, C2 as mobile exploration, and C3 as stationary/low activity. (C) Behavioral composition across AshenRealm seeds 1702–1705 shows that all three modes recur across independent worlds. (D) Comparison of the artifact-centered fraction between BioFoundry and AshenRealm. The recurrence of artifact-centered and exploratory behavior under a distinct geography and materials task indicates that behavioral diferentiation is not restricted to the original BioFoundry environment.

agent artifact laboratory  
![](images/30e9d6bec7c78f330cf9fa314d651cee0e136998cd35b231216614b923ecf8cd.jpg)

![](images/f2571474ce5ef7fec0c5bcc227b61baba03e3ec975d05c1b66226f9c9ff13cea.jpg)

![](images/33d95960dd6892b00623971cf67a6be5a34860312b9e0faa79ed88fbc6bbd329.jpg)

D  
![](images/acd06cae61db843f1abd560b126b91c9a77123a6723a253d111d462cf94e1e39.jpg)  
E

![](images/3fe769cfe6c31bb3e6bbf0e87c2150f239e4f879a7a8db77d2cf69e3be1e3a6f.jpg)  
F

![](images/566c7540ca81df55b336aa738448075a7149aef5c4bc22c06e8563381babd633.jpg)  
Figure S7: Within-run technological accumulation in a representative AshenRealm society with full communication with N=50 (seed 1703). (A–C) Authoritative world states at ticks 0, 400, and 800 show the progressive modification of the volcanic landscape by persistent technologies. Agents are black circles, artifacts are gold diamonds, and processing laboratories are open squares. The number of constructed artifacts increases from 0 to 2 to 20 as the world changes. (D) Persistent-artifact count through the 800-tick episode. (E) Running best-artifact performance and portfolio resilience. Best-object performance rises through discrete discovery events, whereas portfolio resilience develops as additional technologies accumulate. (F) Normalized spatial entropy of agent occupancy on a fixed 10 × 10 grid declines over the episode, indicating increasing localization of activity as persistent technological sites emerge. This single trajectory is illustrative rather than inferential, but shows how agents progressively transform a materially distinct volcanic environment into a persistent technological landscape.

![](images/9ace0fd172ac62226ace2928a06b9d0087d7e575843214310dad0c290405fe41.jpg)  
Figure S8: Material inputs, fabrication pathways, and realized functions of representative AshenRealm technologies. Rows A–H correspond to the eight performance-ranked technologies shown in Figure 22. Left: construction feedstocks recorded when each artifact was built. Filled circles identify recipe inputs and circle area scales with consumed mass; gray open circles indicate absent or unavailable feedstocks. The four highest-ranked designs (A–D) are predominantly obsidian-based, whereas E–H use iron ore. Center: ordered fabrication pathways. Filled numbered nodes identify executed processing operations and connecting lines show their sequence; open circles mark operations not used. Obsidian technologies generally employ longer multistep routes involving washing, smelting, refining, casting, forging, quenching, or annealing, while the iron-based designs use shorter wash–crush pathways. Right: realized lifetime-peak functional service recorded by the simulator. Circle area and printed values encode the realized service magnitude; open circles indicate no realized service in that dimension. Obsidian designs are dominated by thermal regulation, reaching 0.34 for the Cinderveil Thermal Segment, whereas the iron-based technologies primarily provide structural support, with several designs also contributing thermal regulation. The figure separates what matter agents actually consumed, how it was processed, and what functions the resulting artifacts actually realized, preventing agent-authored design claims from being conflated with executed simulator behavior.

SIMULATOR RECORD - tick 39   
A013’s subsequent INSPECT creates observation\_0000000587: terrain CELLULOSE\_FIELD, resource CELLULOSE, remaining re  
source mass 2.303663, station NONE, at (45, 15). The measurement is stored independently of the prose message.

Tick 277 - A075 to A011 - REPLY - message\_0000006055   
“My independent CHITIN microbatch was fabricated and TESTed: utility 0.475047, quality   
0.6766, permeability 0.1838, toughness 0.214246, healing 0.23336, responsiveness 0.15782. I am   
now attempting a small persistent modular lattice build at the confirmed FERMENTER.”

## S3.1 Vignette S1: local discovery and independent confirmation

```batch
Tick 6 - A000 to A013 - MESSAGE - message_0000000057
```

“I am surveying nearby fabrication workspaces and cellulose material behavior. Have you observed a station, tested recipe, or useful local landmark to share?”

SIMULATOR RECORD - ticks 6 and 37   
The message is delivered within the local communication radius. At tick 37, A013 harvests 0.30 mass of resource enum   
5 (CELLULOSE) at (45, 15) and marks the request as fulfilled.

Tick 38 - A013 to A000 - REPLY to message\_0000000057

“I directly observed CELLULOSE at [45,15], approximately 2.6 mass across five nearby cellulosefield sites. No fabrication station is present locally; I am exploring for one.”

Tick 51 - A000 to A013 - REPLY to message\_0000000568

“I have confirmed CELLULOSE at [45,16] and harvested a sample. Have you found a verified fabrication station, landmark, or tested cellulose behavior? Please share coordinates or evidence if available.”

Immediately before sending the reply, A000 successfully harvests 0.30 CELLULOSE at (45, 16). The follow-up message therefore reports an independently executed local action rather than simply repeating A013’s claim.

Interpretation: The conversation turns a local observation into a navigational lead, but the receiving agent still verifies and harvests the material for itself. The sequence illustrates grounded information sharing rather than unrestricted copying of another agent’s belief.

## S3.2 Vignette S2: measurement exchange, executable inheritance, and replication

“I independently measured my 0.3-mass chitin batch at utility 0.470496, quality 0.6588, permeability 0.184889, toughness 0.214246, healing 0.23048, and responsiveness 0.14912. I am using this evidence to build a distinct moisture-gated porous membrane at the confirmed FER-MENTER.”

SIMULATOR RECORD - ticks 289 and 324   
A011 constructs artifact\_00000003, “Moisture-Gated Chitin Membrane,” containing program   
program\_88ab6e937138ab56c8b8c12f. At tick 324, A027 constructs artifact\_00000006, “Moisture-Gated Chitin In  
terface,” deploying the same content-addressed program. The controller senses local moisture, compares it with 0.35, and   
actuates aperture state.

A075 executes FORK\_PROGRAM on artifact\_00000006. The accepted child, program\_bc99a75d96d63e1f35dcbbdc, records program\_88ab6e937138ab56c8b8c12f as its parent and changes the moisture threshold from 0.35 to 0.50.

Tick 834 - A017 to A005 - TEACH action - reply to message\_0000015114

Tick 327 - A075 to A011 - MESSAGE - message\_0000007110

“I am testing a fork of the observed moisture-gated chitin program on artifact\_00000006, raising the aperture threshold from 0.35 to 0.5. Local moisture is 0.435; I will report inspected service changes without claiming unmeasured field efects.”

A030 constructs artifact\_00000009, “Adaptive Chitin Exchange Interface,” with independent material evidence and the original parent program. Its causal parents include the previously observed artifact\_00000006.

Tick 376 - A075 to A011 - MESSAGE - message\_0000008097

“The locally observed high-threshold fork on artifact\_00000006 currently reports performance 0.0969, adaptive regulation 0.0986, and structural support 0.0926; the parent-program artifact\_00000009 reports performance 0.0851, adaptive regulation 0.0515, and structural support 0.0982. This is an observed program comparison under current local conditions, not proof of broader field performance.”

Tick 390 - A011 to A075 - REPLY to message\_0000008097

“I inspected the locally observed high-threshold fork comparison. The fork on artifact\_00000006 reports performance 0.0969, adaptive regulation 0.0986, and structural support 0.0926; the parent-program artifact\_00000009 reports performance 0.0851, adaptive regulation 0.0515, and structural support 0.0982. This is an observed local comparison, not proof of broader field performance.”

Interpretation: The agents exchange measurements, encounter a persistent controller in the world, modify it through an explicit parent-child program edge, and compare the descendant with a parent-program artifact. The caveats in the raw messages are important: the agents describe matched local measurements without treating one comparison as universal proof. This is a compact example of executable cultural evolution rather than language-only brainstorming.

## S3.3 Vignette S3: a distributed request becomes durable teaching

Tick 747 - A005 to 18 nearby agents - MESSAGE - message\_0000015114

“I published direct evidence for my MINERAL-CATALYST microbatch and local contextualversus-passive field observations. Please use the new records as causal parents and report only matched measurements, especially under elevated contamination.”

Tick 749 - A002 to A005 - REPLY to message\_0000015114

“I am performing a fresh direct inspection of artifact\_00000029 under the current mineral-spring condition and will share the matched measurement record; existing evidence indicates contextual remediation is conditional and near zero at contamination 0.012.”

Tick 793 - A008 to A005 - REPLY to message\_0000015114

“My independent current inspection of artifact\_00000029 running program\_3494cd0a9e51636ce656c8cd at the mineral spring measured performance 0.0347, structural support 0.0481, remediation 0.0, self-maintenance 0.0, health 0.7981, and contamination 0.012. This is matched local evidence only; causal record insight\_0000015016.”

“Direct comparison from my controlled tests: test\_00000080 cellulose-mineral-catalyst utility 0.417955, toughness 0.145664, permeability 0.142931, healing 0.06355; test\_00000052 mineralcatalyst control utility 0.426554, toughness 0.066569, permeability 0.153, healing 0.074146. I have built artifact\_00000040 from the cellulose-containing batch and will inspect it for matched current field services; these results do not establish field efects.”

SIMULATOR RECORD - tick 834   
The TEACH action succeeds with cited causal parents test\_00000080 and test\_00000052. The simulator creates   
teaching\_0000016669 in A005’s durable records and marks the original request as fulfilled.

Interpretation: One agent makes an open research request, several peers undertake local follow-up, and another agent transfers a controlled comparison through the stronger TEACH channel. Unlike an ordinary message, teaching creates a durable record with explicit evidence identifiers, making the informational handof auditable and retrievable later.

S3.4 Vignette S4: a conversational request becomes a physical transaction  
Tick 1691 - A010 to A006 - MESSAGE - message\_0000031166   
“I need a small grounded CHITIN transfer for a persistent composite build. Can you   
trade approximately 0.20 CHITIN at the TESTER? My direct composite evidence is in  
sight\_0000030344.”

“Transferring 0.20 grounded CHITIN for your persistent composite build; please return direct build or field evidence when available.”

Forty-eight ticks later, A010 constructs artifact\_00000138, “Fungal-Chitin Healing Exchange Interface,” at the TESTER. The accepted recipe consumes 0.40 CHITIN and 0.15 FUNGUS and records a three-layer aligned porous composite with a passive installed program.

Interpretation: The episode shows that communication can invoke a simulator-enforced transfer of physical matter rather than merely exchanging text. The later build is temporally consistent with the request and uses the requested material. However, its declared causal parents are prior experimental records rather than the trade message, so this vignette should be interpreted as a transaction followed by construction, not as definitive proof that the transferred 0.20 mass alone caused the artifact.

## S3.5 Raw structured output corresponding to a displayed message

The following is the complete accepted action object stored for A000 at tick 6. Numeric enum verb: 11 denotes COMMUNICATE; the zero-valued direction, resource, and artifact fields are the required inactive defaults. The text in the message field is the text rendered in the first chat bubble above.

```jsonl
{
"amount": 0.0,
"artifact": 0,
"artifact_spec": null,
"causal_parents": [],
"direction": 0,
"insight": null,
"message": "I am surveying nearby fabrication workspaces and cellulose material behavior. Have you observed a station,
tested recipe, or useful local landmark to share?",
"program": null,
"recipe": null,
"reply_to": "",
"resource": 0,
"target_agent_id": "agent_000013",
"target_artifact_id": "",
"target_x": -1,
"target_y": -1,
"verb": 11
}
```

## S3.6 Trace provenance and interpretation limits

The source is generated with engine revision 9 and model gpt-5.6-luna. Displayed message text, action intent, identifiers, coordinates, values, ticks, and recipient relationships are taken from the retained trace. Simulator-record strips condense the corresponding event payloads into readable prose but do not introduce additional outcomes. These examples establish that the mechanism can support grounded communication, executable inheritance, durable teaching, and transactional exchange. They are illustrative process evidence; comparative claims about prevalence or performance remain based on the paired seed-level analyses in the main Results.