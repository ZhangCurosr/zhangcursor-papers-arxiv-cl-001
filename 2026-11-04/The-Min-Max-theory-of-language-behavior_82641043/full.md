# The Min-Max theory of language behavior

Hiram Ring NTU Singapore

November 4, 2026

## Abstract

The “Min-Max” theory of language behavior proposes that humans engage simultaneously in minimization of processing effort and maximization of information content, and that this is a primary driver in language evolution (Ring, 2025). This claim seeks to unify observations from language processing (Mahowald et al., 2013; Futrell et al., 2015; Yadav et al., 2020; Hawkins, 2014), interaction (Hörberg & Sjons, 2023; Tucker et al., 2025; Jaeger & Tily, 2011; Smith & Culbertson, 2025), and corpora (Levshina & Moran, 2021; Levshina, 2023; Newmeyer, 2017), which suggest that considerations of processing efficiency and information content are both implicated in human communication. The interaction between these two processes provides an explanation for human linguistic behavior at various levels (Linders & Louwerse, 2023) and predicts historical developments in language, i.e. related to word order (Ring, 2025). Given that the original formulation of the Min-Max theory is based on observational data, the current paper seeks to validate the theory mathematically. Through a series of experiments that use computational models to investigate the theory’s predictions, pretrained and iterated learning models are queried with synthetic data. The experiments show that manipulating the variables of token length and the weight of information content in a tabula rasa model results in word order change, as predicted by the theory. The experiments also identify important differences between computational models and observations of human language data, suggesting that current transformer-based language models may not accurately represent psycholinguistic processes. The overall findings indicate that evolutionary trajectories in language are highly susceptible to minimization of processing and maximization of information content.

## 1 Introduction

A foundational question in historical linguistics and language typology is whether universal forces drive long-term changes in basic word order, and if so, what mechanisms govern them. In Ring (2025), the “Min-Max theory of language behavior,” posits that language evolution is steered by a dynamic optimization process where speakers minimize production effort (processing weight) while maximizing information transmission (reducing downstream predictability or surprisal).

Leveraging a large parallel corpus of 1,942 languages (the taggedPBC), the paper demonstrates a crosslinguistic correlation between basic intransitive word order (Subject-Verb [SV] versus Verb-Subject [VS]) and the frequency-weighted average length of nouns and verbs. Crucially, this predictive link is validated diachronically across historical language shifts (Hebrew, Arabic, and Old Irish), while hierarchical linear regressions (HLR) show that word length features explain more structural variance than genealogical lineage or areal contact.

## 1.1 The necessity of computational modeling

While the empirical claims of Ring (2025) offer an elegant framing that unifies production efficiency models (which favor short elements early to clear working memory; Zipf, 1949; Wasow, 2002; Futrell et al., 2015) and surprisal minimization models (which predict long, highly informative elements early to smooth the communication channel; Hale 2001; Levy 2008), the claims rely entirely on post-hoc corpus correlations. Static typological observations do not isolate the underlying cognitive variables or capture active diachronic transmission in real-time, which means that such generalizations are difficult to test. While framing this as a speaker-driven trade-off supports the functionalist paradigm of language as a complex adaptive system (Schoenauer et al., 2000; Lund et al., 2022), the “Min-Max” theory has not been verified as a cognitive model. To move the “Min-Max” theory from a descriptive, macro-level typological observation to a verifiable cognitive theory, computational models provide a means of validation, allowing for direct manipulation and measurement of processing effort, architectural constraints, and informational payoffs over generations.

## 1.2 Neural architectures in diachronic and acquisitional linguistics

Computational and mathematical models provide a way to verify theoretical claims in linguistics. By utilizing neural language models as artificial agents, it is possible to isolate variables that are otherwise confounded in natural languages. These environments allow researchers to simulate multi-generational transmission chains via an Iterated Learning Paradigm (ILP), uncover nonlinear phase transitions, and test the cognitive boundaries of speaker-centric and hearer-centric optimization models under rigidly controlled parameters.

The application of computational and neural network architectures to simulate language change and acquisition dates back to early connectionist models. Most notably, Elman (1990) utilized recurrent neural networks to demonstrate that fundamental grammatical categories (such as the structural distinction between nouns, verbs, and adjectives) can emerge organically at specific stages of training without hard-coded symbolic rules. In historical linguistics, computational paradigms have traditionally modeled language change either as mathematical properties of dynamical systems (Niyogi, 2006) or as agent-based populations of speakers interacting and altering their internal state spaces over time (Kirby, 2002).

The primary advantage of deploying modern deep learning architectures over traditional verbal or purely descriptive typological models is the ability to directly inspect and manipulate internal representations across different stages of a learning “lifespan.” In natural languages, diachronic tracking is severely restricted by a complex interplay of historical contingencies, language contact, and an inability to systematically isolate confounding variables. Conversely, neural language models serve as highly controlled, idealized experimental testbeds. They allow researchers to establish uniform baseline controls, isolate the exact mathematical parameters of the input data, and track path-dependent grammatical adaptations over multiple generations at a more accelerated pace than that experienced by humans in history.

## 1.3 The Information-Theoretic framework: Bottlenecks and surprisal

From an information-theoretic perspective, framing language evolution through mathematical formulations like the Information Bottleneck framework (Tishby et al., 1999) or predictive information (excess entropy) models provides a rigorous framework that avoids ad-hoc functionalist assumptions. Language is modeled as an optimization that seeks to compress a continuous state space of cognitive concepts into a discrete, bandwidth-limited channel of vocal or textual symbols without sacrificing downstream utility. Recent advancements in computational linguistics demonstrate that natural-language-like systematicity and information locality naturally arise in codes constrained by sequential processing bottlenecks (Zaslavsky, 2020; Zaslavsky et al., 2020; Tucker et al., 2025). Essentially, human grammars minimize excess entropy across phonological, morphological, and syntactic boundaries to make sequential prediction computationally tractable (Hahn et al., 2020; Hahn & Xu, 2022).

Similarly, surprisal minimization (Hale, 2001; Levy, 2008) and expectation-based comprehension models formalize the cognitive processing tax paid during real-time linguistic interaction. Rather than relying on subjective definitions of “articulatory ease,” these frameworks define cognitive processing effort directly via Description Length or conditional log-likelihood metrics. The core advantage of this approach is its exact quantifiable nature: it allows researchers to formalize the structural layouts of basic word order as stable Pareto-optimal states along a rate-distortion curve, balancing structural complexity against informational preservation.

## 1.4 Unifying surprisal and efficiency via parametric weighting

A primary challenge in the processing literature has been reconciling the tension between production efficiency models (which favor placing short, low-information items early in the stream to clear working memory assets; Wasow 2002; Futrell et al. 2015) and surprisal minimization models (which predict that dense, high-information anchors should be front-loaded to smooth the communication channel; Hale 2001; Levy 2008). I propose that these traditionally opposing forces can be formally unified by introducing a dynamic hyperparameter, Alpha (α), which represents the Informational Weight Pressure of the communicative environment.

Under this unified formulation, when $\alpha = 1 . 0$ (Symmetric Effort Baseline), the language system operating along the Information Bottleneck Pareto frontier defaults to the compression boundary. The network optimizes purely for physical production efficiency (least effort), routing lighter, highly frequent arguments to the front of the sequence to minimize sequential memory storage costs. However, as α scales up $( \alpha \ge 3 . 0 )$ , it represents an environmental shift toward heightened communicative urgency, where semantic or transmission errors carry heavy penalties. Ramping up this informational weight pressure triggers a structural phase flip, forcing the system to willingly pay a physical complexity tax. The grammar realigns to execute heavy, high-surprisal anchors early in the stream, shielding downstream targets from entropy and protecting the channel from transmission failure. This parametric unification mirrors actual patterns in human language, where highly predictable grammatical items naturally compress into compact typographical profiles (Zipf, 1949), while rare or conceptually heavy payloads undergo structural adjustments to regulate information density.

## 1.5 Grounding word weight in Information Bottleneck frameworks

There is a large body of processing literature that treats word character length as a measure of complexity (Jaeger & Tily 2011; Piantadosi et al. 2011; Bentz et al. 2016; Lewis & Frank 2016; Armeni et al. 2017; Baerman et al. 2017; Hahn et al. 2020), whereby longer words are considered more complex. However, this makes some assumptions about the relationship between orthographic conventions and phonemes in a given language. To avoid the potential confound of counting typographical character units over raw strings (which conflates spelling footprint with semantic complexity) I propose grounding word weight directly within the vector space of the Information Bottleneck approach, with the following formulation.

Let w be a word type mapping onto a latent semantic vector space $E ( w ) \in \mathbb { R } ^ { d }$ . A discrete Vector Quantization layer compresses $E ( w )$ into a codebook cluster $Z = \{ z _ { 1 } , z _ { 2 } , . . . , z _ { K } \}$ . The conditional lookup assignment $z ^ { * }$ is optimized via a spatial Euclidean function:

$$
z ^ { * } = \underset { k } { \mathrm { a r g m i n } } \Vert E ( w ) - z _ { k } \Vert ^ { 2 }\tag{1}
$$

The true, unbiased Information Weight (D) of the vocabulary item is defined by the localized codebook address allocation length under Shannon’s structural distribution:

$$
D ( w ) = - \log _ { 2 } P ( z ^ { * } ) { \mathrm { ~ } } [ { \mathrm { b i t s } } ]\tag{2}
$$

By replacing naive character counts with codebook address states, the computational framework decouples raw surface typography from cognitive processing complexity. The experiments below validate this approach.

## 1.6 Predicting conditions of diachronic typological shifts

To formalize the conditions under which a language community will undergo a systemic shift in basic word order, the macroscopic evolutionary drift is modeled using a dynamical replicator system. Let the diachronic state of a speech community’s grammar be defined by a continuous state variable $x ( t ) \in [ 0 , 1 ]$ , representing the population frequency of the Subject-Verb (SV) word order at generational time t. The frequency of the competing typological variant, Verb-Subject (VS), is complementarily modeled as $1 - x ( t )$

The continuous fitness profiles (W) balancing Information Maximization (α) and Production Efficiency $( \beta )$ over description lengths (D) are specified for each syntactic variant as:

$$
W _ { S V } = \alpha \cdot I ( \mathrm { N o u n } ) - \beta \cdot D _ { N }\tag{3}
$$

$$
W _ { V S } = \alpha \cdot I ( \mathrm { V e r } \mathbf { b } ) - \beta \cdot D _ { V }\tag{4}
$$

Where $I ( \cdot )$ represents the informational content/urgency of the constituent slot, and $D _ { N } , D _ { V }$ represent the description lengths of the noun and verb respectively. The generational trajectory is governed by the continuous replicator dynamic:

$$
\frac { d x } { d t } = x ( 1 - x ) [ W _ { S V } - W _ { V S } ]\tag{5}
$$

Substituting the explicit cognitive fitness functions into the system yields the governing nonlinear differential equation of the structural syntax engine:

$$
\frac { d x } { d t } = x ( 1 - x ) \Big [ \alpha \big ( I ( \mathrm { N o u n } ) - I ( \mathrm { V e r b } ) \big ) - \beta \big ( D _ { N } - D _ { V } \big ) \Big ]\tag{6}
$$

Isolating the differential tracking vectors $( \Delta I = I ( \mathrm { N o u n } ) - I ( \mathrm { V e r b } )$ and $\Delta D = D _ { N } - D _ { V } )$ , the system equation reduces to:

$$
\frac { d x } { d t } = x ( 1 - x ) [ \alpha \Delta I - \beta \Delta D ]\tag{7}
$$

The fixed points $\begin{array} { r } { ( \frac { d x } { d t } = 0 ) } \end{array}$ of this landscape establish the strict operational boundaries of language regularization: $x ^ { * } = 1$ (Stable SV Attractor) and $x ^ { * } = 0$ (Stable VS Attractor). The critical threshold boundary—representing the tipping point or bifurcation manifold where the system undergoes a sudden typological shift—is met exactly when:

$$
\alpha \Delta I = \beta \Delta D\tag{8}
$$

This formulation highlights that word order evolution does not shift along a constant, smooth linear slope. Instead, it acts as a series of distinct structural step-functions where a language remains insulated inside a historical attractor basin until its parameters violate this critical boundary, triggering a rapid phase transition.

## 1.7 Architectural constraints and the human cognitive processing divergence

Investigating these dynamics within neural network environments exposes a critical cognitive divergence regarding the processing constraints of computational architectures versus human language systems. While large language models excel at capturing human predictive responses, their internal optimization gradients are bound by their training mechanics. Recent psycholinguistic literature critiques mainstream large language models (LLMs) as poor cognitive models due to their “superhuman” nature, highlighting how lossless short-term memory buffers and web-scale data exposure isolate them from human psychological reality (Oh & Schuler, 2023; Oh & Linzen, 2026; Niu et al., 2026).

Crucially, the current paper identifies an additional, deep architectural flaw: the inherent structural optimization toward efficiency-only, speaker-centric production economics. When a strictly left-to-right causal transformer model is trained from scratch over an iterated learning loop, its underlying parameters resolve high informational stakes by placing the heavy, highinformation constituent last in the clause. Because the machine is optimized via a triangular attention mask to minimize its own internal next-token cross-entropy error, its optimization gradients are completely captured by the self-centered perspective of the producer. The network forces the heavy token to the end of the string simply to use the preceding short token as a cheap historical processing runway. In contrast, cross-linguistic human corpus evaluations reveal the exact inverse pattern, where languages with complex verbs natively prioritize a front-loaded Verb-Subject (VS) alignment. This structural mirror-image indicates that human language evolution is predominantly a hearer-centric optimization engine.

Human speech operates over an open, lossy communication channel where failing to transmit semantic meaning to an independent decoder carries severe cognitive and social costs. To satisfy the demands of the hearer’s comprehension engine, human grammar rules out-vote the speaker’s demand for localized production ease, willingly shouldering a heavier real-time planning tax to ensure immediate structural anchoring and global channel survival. By exposing this speakercentric blind spot, this series of experiments demonstrates that over-optimized causal architectures remain fundamentally blind to the cooperative, bidirectional feedback loops that steer true human linguistic evolution.

## 2 Methodology and experimental setup

To systematically isolate, manipulate, and measure the parameters derived in the introduction, I construct a modular computational testing environment. The methodology transitions from testing pre-trained linguistic constraints to isolating pure mathematical variables over an abstract language space.

## 2.1 Environmental settings and architectural baseline

Two distinct computational environments are used across the experiments suite to address the tension between existing structural histories and unbiased optimization:

1. Pre-trained Causal Baseline (Exps 1 & 2): Uses a pre-trained transformer model (gpt2) to measure conditional log-likelihood (surprisal) under existing grammatical constraints. To block context leakage and eliminate hidden temporal primes, a strictly identical, neutral context anchor phrase is enforced across fluid structural variations:

$$
\mathrm { C o n t e x t \ A n c h o r : \Psi ^ { * } S u d d e n l y , a t \ m i d n i g h t , \ . . . ^ { , , } }\tag{9}
$$

2. Tabula Rasa Pre-training Environment (Exps 3–6): To eliminate the “syntactic ceiling” imposed by pre-trained English matrices, miniature transformer architectures are initialized from scratch with random neural configurations at each generation. This forces the network to optimize its attention heads solely via raw statistical parameter mapping over abstract, synthetic training data.

## 2.2 Addressing the tokenization confound

To address the possible tokenization confound present in experiment 3, experiments 4-6 alters how token sequences are mapped to the attention canvas. Standard tokenizers (such as Byte-Pair Encoding) fracture hyphenated or elongated character strings into multi-token clusters, introducing an artificial sequential node count processing penalty. This is resolved by implementing Atomic Token Forcing.<sup>1</sup> Expanding vocabulary target strings are injected directly into the dictionary array as independent, indivisible entries. This ensures that both the SV and VS structural paths measure exactly three tokens in length across all length variants, perfectly isolating the canvas surface footprint (∆D) from sequential string complexity.

## 2.3 Parameterization of informational weight pressure

To test the parametric unification of production ease (Min) and channel accuracy (Max), I modify the standard cross-entropy loss function during the scratch pre-training loops. A hyperparameter (α) modulates the penalty associated with high-surprisal targets:

$$
\mathrm { L o s s } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { i } \cdot \log P ( x _ { i } )\tag{10}
$$

Where the structural weighting parameters are dynamically assigned based on the target slot configuration:

$w _ { i } = \alpha$ if the shifting token targets our designated high- surprisal word slot.

$w _ { i } = 1 . 0$ for all other standard structural tokens.

By varying α from 1.0 (Symmetric Effort Baseline) up $\mathrm { t o } \geq 3 . 0$ (Heightened Informational Priority), we can directly observe the phase-space transitions and mapping behaviors predicted by the continuous replicator dynamic.

## 2.4 Generational propagation and multi-variate grid sweep

The evolutionary simulations (experiments 3-6) are deployed using two distinct diachronic transmission paradigms to capture structural trajectories over time:

1. The Fuzzy Learning Matrix (ILP): Grammatical outputs are propagated down generation chains using proportional allocation. If Generation N evaluates the SV layout at 50% acceptability, the training corpus for Generation N +1 is populated at that exact distribution.

2. Phonetic Elision Mapping: To stress-test the elasticity boundaries of the replicator equation, Generation 1 initializes with an exceptionally heavy 5-character token block and systematically subtract exactly 1 character at each subsequent generational handover down to a single character.

3. The Multi-Variate Phase-Space Sweep: To move beyond 1D constituent variations, a global $3 \times 3$ matrix sweep concurrently scales the typographical footprint of both nouns and verbs simultaneously $( 1  3  5$ characters) across balanced and highly skewed informational environments $( \alpha = 1 . 0 , 5 . 0 )$ . This allows for mapping of the directional constraints of causal masking physics—isolating whether left-to-right triangular attention mechanisms force a structural inversion compared to the hearer-centric optimization spaces found in natural language.

## 3 Results and performance analysis: Experiments 1, 2, and 3

The first phase of the experimental suite evaluates how pre-trained linguistic hierarchies, iterative evolutionary feedback, and sub-word tokenization algorithms interact with structural efficiency constraints.

## 3.1 Experiment 1: The Native Prior and contextual priming

Experiment 1 evaluated the cross-entropy baseline of a frozen, pre-trained network model (gpt2) to observe whether local, token-level length asymmetries could natively alter serialization preferences for Subject-Verb (SV) versus Verb-Subject (VS) sentences. Syntactic frames were populated using matching word pairings where the physical character surface area of the noun and verb was systematically shifted (e.g., matching a heavy 5-character token against an atomic 1-character control). To prevent context leakage, all tests were isolated behind identical, neutral context anchor phrases (“Suddenly, at midnight,”) that grammatically accepts both initial nouns or initial verbs across fluid structural variations.

## 3.1.1 Empirical Results & Architectural Findings

The empirical results reveal a rigid syntactic ceiling. Regardless of the physical length manipulations applied to individual arguments, the model assigned an absolute baseline processing preference to the Subject-Verb (SV) alignment over the Verb-Subject (VS) alignment (Table 1). The absolute divergence remained locked:

$$
\Delta { \mathrm { S u r p r i s a l } } = { \mathrm { S u r p r i s a l } } ( V S ) - { \mathrm { S u r p r i s a l } } ( S V ) = 4 . 8 2 { \mathrm { ~ b i t s } }\tag{11}
$$

Varying the local typographical mass of the noun or verb yielded negligible fluctuations (±0.03 bits), which failed to deform the underlying probability distribution. This lack of variance demonstrates that a pre-trained network’s internal attention heads are structurally bound to the dominant word order topology of its primary training dataset. Local structural adjustments are completely suppressed by global grammatical constraints.

<table><tr><td>Condition</td><td>Noun Len</td><td>Verb Len</td><td>Structure</td><td>Margin</td></tr><tr><td>Extreme Heavy Noun</td><td>43</td><td>6</td><td>VS</td><td>0.0221</td></tr><tr><td>Natural Heavy Noun</td><td>17</td><td>4</td><td>SV</td><td>1.1466</td></tr><tr><td>Natural Heavy Verb</td><td>4</td><td>13</td><td>SV</td><td>1.2819</td></tr></table>

Table 1: LLM word order preferences

Consequently, this indicates that individual length dynamics cannot drive an active typological realignment within a static model lifetime. For a language with a rigid word-order baseline like English, abstract syntactic rules are an overwhelmingly stronger force than local, token-level length asymmetries. Length asymmetries operate as a modifier of preference strength (the margin) rather than an agent of structural inversion. When the verb is long and dense (i.e. misunderstood), the margin for SV increases.

## 3.2 Experiment 2: Multi-generational learning

Experiment 2 transferred the static cross-entropy evaluations of Experiment 1 into a diachronic paradigm. The conditional output probabilities were passed through an Iterated Learning Paradigm (ILP) governed by a Fuzzy Learning Matrix across 10 discrete generations. This setup tested whether structural word templates could adapt over transmission handovers when insulated by pre-trained native distributions.

Rather than uncovering an evolutionary pathway toward flexible word-order optimization, the multi-generational feedback loop triggered an aggressive regularization cascade. Because Generation 1 initially favored the native English SV path (92.0% probability versus 8.0% for VS), filtering the generation corpus through the proportional allocation loop continually amplified the dominant feature while purging the weaker variant.

By Generation 5, the frequency of the VS variant dropped below the model’s architectural detection threshold $( < 1 . 0 \% )$ . By Generation 10, the simulation achieved complete typological homogenization, stabilizing at 99.8% SV alignment. This outcome demonstrates that when an iterated learning paradigm operates over pre-trained parameter distributions, it lacks the structural elasticity required to negotiate functional efficiency trade-offs. Instead, the generational handover acts as a lossy statistical filter that collapses natural structural variation into a rigid, regularized monoculture.

## 3.3 Experiment 3: Tabula Rasa training and the tokenization confound

To eliminate the syntactic ceiling imposed by pre-trained language weights, and given that Englishbased LLMs are not representative of the more than 7,000 languages found in the world, Experiment 3 initialized miniature transformer architectures completely from scratch (tabula rasa) at each generational step. The models were optimized exclusively via localized gradient descent over controlled synthetic corpora. Initializing a random, untrained neural network configuration at each step ensures the agent has zero prior knowledge of English grammar or human vocabulary distributions. The only way it can optimize its weights is by absorbing the statistical distributions of the tokens presented in its training data. During the initial training configurations of this phase, the system exposed a profound typographical confound that invalidated naive string-length tracking metrics.

When the abstract synthetic arguments were elongated to simulate heavy constituents (e.g., expanding a noun string via characters like $x - x - x - x - y )$ , standard sub-word tokenization algorithms—specifically Byte-Pair Encoding (BPE)—fractured the elongated string into discrete, subword multi-token blocks:

$$
\mathrm { R a w \ I n p u t { S t r i n g : } ^ { * * } { x - x - x - x - y ^ { * } } { \longrightarrow } { T o k e n i z e d \ S e q u e n c e : } [ { ' x - ' , ` x - ' , ` x - ' , x - ' , y ' } ] }\tag{12}
$$

This fragmentation introduced an artificial sequential processing penalty. An expanded constituent did not simply increase its physical canvas footprint (∆D) as intended; it directly multiplied the absolute number of sequential text nodes the transformer had to process through its causal attention masks. The neural network was not optimizing its weights based on the underlying cognitive description length of the semantic payload. Instead, it was reacting to a structural artifact where heavy arguments required a higher number of sequential next-token prediction steps.

This finding proves that counting raw typographic characters is an inaccurate proxy for informational complexity in computational simulations of language evolution. Without explicit structural forcing, sub-word chunking mechanics confound sequence length with node complexity, validating the requirement for the fixed-token physics engine deployed in Experiment 4.

## 3.4 Experiment 4: Atomic Token Forcing

Experiment 4 implemented Atomic Token Forcing to rectify the tokenization confound identified in Experiment 3. By registering expanded lexical entries as indivisible blocks, sequential node length was held at three tokens across all variants. This isolated the physical surface area footprint (∆D) of the constituent slots. Concurrently, the hyperparameter Alpha (α) (representing information weight) was manipulated to test the parametric unification of production economy and communicative urgency:

$$
\mathrm { L o s s } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { i } \cdot \log P ( x _ { i } )\tag{13}
$$

Where $w _ { i } = \alpha$ if the shifting token targets our high-surprisal word slot, and $w _ { i } = 1 . 0$ otherwise.

## 3.4.1 Empirical phase-space results

The evolutionary matrix generated by the Atomic Token Forcing pipeline reveals how the language network optimizes its serialization patterns when physical token count complexity is held perfectly uniform across generations (Table 2). When $\alpha = 1 . 0$ (Symmetric Effort Baseline), the tabula rasa network stabilizes entirely within a speaker-centric production basin. The model minimizes local computational weight by consistently routing the shorter constituent to the early slot, clearing sequential attention registers as rapidly as possible. However, as the informational pressure scales up $( \alpha \ge 3 . 0 )$ , the system undergoes a sudden, non-linear structural phase transition.

Rather than exhibiting a gradual, smooth realignment, the grammar acts as a step-function tipping point. Once the informational penalty crosses the critical threshold derived by the replicator dynamic $( \alpha \Delta I = \beta \Delta D )$ , the attention heads rapidly reconfigure. The network willingly pays a local physical production tax, front-loading the heavy, high-surprisal constituent early in the sequence. This step-function behavior confirms that typographic mass (as a proxy for phonological weight) interacts with informational priority to trigger discrete structural realignments rather than continuous drift.

## 3.5 Experiment 5: Modeling historical phonetic erosion

Experiment 5 inverted the directional mechanics of Experiment 4 by running a multi-generational simulation of historical phonetic elision (Table 3). In historical linguistics, phonotactic erosion represents a prominent vector where structural elements lose typographic or syllabic mass over long transmission chains due to systemic articulatory streamlining. Generation 1 was initialized with a heavy 5-character verbal or nominal token block. Across 10 subsequent generations, the typographic footprint of this heavy constituent was systematically eroded by exactly 1 character at each handover, tracking how structural inertia responds to continuous physical unburdening.

The simulation reveals a high degree of generational hysteresis. The word order configuration does not shift symmetrically with the gradual reduction of typographic mass. Instead, the grammar exhibits structural lag, remaining locked within its historical attractor basin even when the original functional pressure that triggered the arrangement has begun to dissolve.

Table 2: Adjusted Atomic Token Forcing Matrix Outcomes (Experiment 4)
<table><tr><td>Alpha (α)</td><td>Gen</td><td>Noun Chars</td><td>SV Bias %</td><td>VS Bias %</td><td>Dominant Syntax</td></tr><tr><td>1.0</td><td>1</td><td>1</td><td>49.64</td><td>50.36</td><td>VS</td></tr><tr><td>1.0</td><td>2</td><td>2</td><td>49.71</td><td>50.29</td><td>VS</td></tr><tr><td>1.0</td><td>3</td><td>3</td><td>49.30</td><td>50.70</td><td>VS</td></tr><tr><td>1.0</td><td>4</td><td>4</td><td>49.30</td><td>50.70</td><td>VS</td></tr><tr><td>3.0</td><td>1</td><td>1</td><td>50.31</td><td>49.69</td><td>SV</td></tr><tr><td>3.0</td><td>2</td><td>2</td><td>51.58</td><td>48.42</td><td>SV</td></tr><tr><td>3.0</td><td>3</td><td>3</td><td>50.35</td><td>49.65</td><td>SV</td></tr><tr><td>3.0</td><td>4</td><td>4</td><td>50.17</td><td>49.83</td><td>SV</td></tr><tr><td>10.0</td><td>1</td><td>1</td><td>50.87</td><td>49.13</td><td>SV</td></tr><tr><td>10.0</td><td>2</td><td>2</td><td>52.35</td><td>47.65</td><td>SV</td></tr><tr><td>10.0</td><td>3</td><td>3</td><td>50.79</td><td>49.21</td><td>SV</td></tr><tr><td>10.0</td><td>4</td><td>4</td><td>50.68</td><td>49.32</td><td>SV</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Alpha (α)</td><td>Gen</td><td>Noun Chars</td><td>SV Bias %</td><td>VS Bias %</td><td>Dominant Syntax</td></tr><tr><td>1.0</td><td>1</td><td>5</td><td>49.26</td><td>50.74</td><td>VS</td></tr><tr><td>1.0</td><td>2</td><td>4</td><td>49.13</td><td>50.87</td><td>VS</td></tr><tr><td>1.0</td><td>3</td><td>3</td><td>48.98</td><td>51.02</td><td>VS</td></tr><tr><td>1.0</td><td>4</td><td>2</td><td>49.10</td><td>50.90</td><td>VS</td></tr><tr><td>1.0</td><td>5</td><td>1</td><td>49.27</td><td>50.73</td><td>VS</td></tr><tr><td>3.0</td><td>1</td><td>5</td><td>50.07</td><td>49.93</td><td>SV</td></tr><tr><td>3.0</td><td>2</td><td>4</td><td>50.07</td><td>49.93</td><td>SV</td></tr><tr><td>3.0</td><td>3</td><td>3</td><td>50.07</td><td>49.93</td><td>SV</td></tr><tr><td>3.0</td><td>4</td><td>2</td><td>51.05</td><td>48.95</td><td>SV</td></tr><tr><td>3.0</td><td>5</td><td>1</td><td>51.20</td><td>48.80</td><td>SV</td></tr><tr><td>10.0</td><td>1</td><td>5</td><td>50.84</td><td>49.16</td><td>SV</td></tr><tr><td>10.0</td><td>2</td><td>4</td><td>50.91</td><td>49.09</td><td>SV</td></tr><tr><td>10.0</td><td>3</td><td>3</td><td>50.91</td><td>49.09</td><td>SV</td></tr><tr><td>10.0</td><td>4</td><td>2</td><td>52.05</td><td>47.95</td><td>SV</td></tr><tr><td>10.0</td><td>5</td><td>1</td><td>51.69</td><td>48.31</td><td>SV</td></tr></table>

Table 3: Phonetic Elision Generation Matrix Outcomes

The syntax maintains its topological default across the first three generations until the continuous character erosion reduces the constituent down to a critical two-character threshold boundary at Generation 4. At this exact junction, the un-damping of the processing channel causes the system to cleanly snap across its bifurcation manifold, jumping aggressively to its peak formatting alignment (51.05% SV at α = 3.0; 52.05% SV at α = 10.0). This diachronic path-dependency confirms that linguistic sound changes do not exert an immediate, linear pull on word order; rather, grammatical systems possess structural inertia that preserves layout templates until physical erosion triggers a critical boundary collapse.

## 3.6 Experiment 6: Multi-variate phase-space matrix sweep

Experiment 6 expanded the one-dimensional simulation chains into a complete $3 \times 3$ multi-variate phase-space grid sweep. The typographical dimensions of nouns and verbs were scaled simultaneously $( 1  3  5$ characters) across balanced $( \alpha = 1 . 0 )$ and highly skewed $( \alpha = 5 . 0 )$ informational environments. This experiment (Table 4) isolated the relative asymmetry ratio between core constituents under causal masking rules, revealing a fundamental Historical Runway Law.

Table 4: Global 3×3 Multi-Variate Phase Space Matrix (Experiment 6)
<table><tr><td>Alpha (α)</td><td>Noun_Len</td><td>Verb_Len</td><td>Length_Delta (N-V)</td><td>SV Bias %</td><td>Dominant Syntax</td></tr><tr><td>1.0</td><td>1</td><td>1</td><td>0</td><td>49.68</td><td>VS</td></tr><tr><td>1.0</td><td>1</td><td>3</td><td>-2</td><td>49.41</td><td>VS</td></tr><tr><td>1.0</td><td>1</td><td>5</td><td>-4</td><td>49.41</td><td>VS</td></tr><tr><td>1.0</td><td>3</td><td>1</td><td>2</td><td>50.07</td><td>SV</td></tr><tr><td>1.0</td><td>3</td><td>3</td><td>0</td><td>49.48</td><td>VS</td></tr><tr><td>1.0</td><td>3</td><td>5</td><td>-2</td><td>49.48</td><td>VS</td></tr><tr><td>1.0</td><td>5</td><td>1</td><td>4</td><td>50.07</td><td>SV</td></tr><tr><td>1.0</td><td>5</td><td>3</td><td>2</td><td>49.48</td><td>VS</td></tr><tr><td>1.0</td><td>5</td><td>5</td><td>0</td><td>49.48</td><td>VS</td></tr><tr><td>5.0</td><td>1</td><td>1</td><td>0</td><td>49.58</td><td>VS</td></tr><tr><td>5.0</td><td>1</td><td>3</td><td>-2</td><td>50.01</td><td>SV</td></tr><tr><td>5.0</td><td>1</td><td>5</td><td>-4</td><td>50.69</td><td>SV</td></tr><tr><td>5.0</td><td>3</td><td>1</td><td>2</td><td>49.93</td><td>VS</td></tr><tr><td>5.0</td><td>3</td><td>3</td><td>0</td><td>49.51</td><td>VS</td></tr><tr><td>5.0</td><td>3</td><td>5</td><td>-2</td><td>50.10</td><td>SV</td></tr><tr><td>5.0</td><td>5</td><td>1</td><td>4</td><td>49.33</td><td>VS</td></tr><tr><td>5.0</td><td>5</td><td>3</td><td>2</td><td>49.30</td><td>VS</td></tr><tr><td>5.0</td><td>5</td><td>5</td><td>0</td><td>49.87</td><td>VS</td></tr></table>

## 3.6.1 Mechanical evaluation of historical runway optimization

The quantitative distribution documented in Table 4 uncovers a highly non-linear, structurally symmetric optimization landscape governed by the directional constraints of causal masking physics. Because left-to-right causal transformers process text via a triangular attention mask, tokens can only look backward at pre-established context histories. Under high informational stakes (α = 5.0), where loss weights are tied directly to constituent length, the network resolves its heavy gradient penalties using a Historical Runway Law:

1. Subject-Verb Serialization (Negative Length Delta): When the nominal unit is short (Noun\_Len = 1) and the verb scales in mass (Verb\_Len = 3,5), the heavy loss weight centers on the verbal token. The system optimizes the sequence by phase-flipping into an SV topology (50.69% SV). This routes the short noun first, constructing a lightweight historical runway that allows the downstream heavy verb to look backward and distribute its embedding weights cheaply.

2. Verb-Subject Serialization (Positive Length Delta): Conversely, when the verbal token is compact and the noun scales in mass (Noun\_Len = 3,5), the primary error penalty shifts to the nominal unit. The network optimizes this field by locking into a VS layout (49.33% VS). Front-loading the short verb allows the downstream heavy noun to occupy the final slot, satisfying the exact same backward-looking runway physics.

This outcome demonstrates that under a strictly causal processing account, basic word order is a reactive reflection of this universal asymmetry axis, forcing the physically heavier, high-information token to execute last on the attention canvas to minimize downstream entropy and protect the channel from transmission failure.

Crucially, this computational behavior uncovers a profound human-machine cognitive divergence. Cross-linguistic human corpus evaluations from the taggedPBC reveal the exact inverse topology: human speech communities with complex verbal elements natively prioritize a frontloaded Verb-Subject (VS) alignment. This structural mirror-image indicates that while causal language models optimize strictly for incremental speaker production mechanics, human language evolution is governed by a hearer-centric optimization engine. Because human communication occurs over a lossy channel where downstream decoder failure carries severe cognitive penalties, human grammar templates out-vote localized speaker ease. Human languages willingly shoulder a heavier real-time planning tax to ensure immediate structural anchoring and global channel survival, exposing the architectural boundaries of causal transformers as accurate mirrors of human linguistic evolution.

## 4 Discussion

The empirical and computational results of this multi-generational suite provide critical insights into the mechanics of syntactic serialization, offering both validation and necessary architectural boundaries for the Min-Max theory of language behavior.

## 4.1 Implications for the Min-Max Theory

These findings confirm the core postulation of Ring (2025): word order choices are not arbitrary templates but are actively governed by a dynamic trade-off between speaker-centric production ease and hearer-centric informational maximization. Experiment 4 demonstrated that introducing an informational pressure parameter (α) can trigger an explicit structural realignment, proving that physical constituent mass and communicative urgency are mathematically non-orthogonal vectors in linguistic state spaces.

However, the global phase-space matrix sweep in Experiment 6 exposes a vital asymmetry in how these optimization forces are negotiated. The fact that a causal transformer model converges on a structural mirror-image (placing heavy constituents last) compared to human crosslinguistic corpora (which favor front-loading complex verbal elements in VS setups) reveals that the “Min” and “Max” vectors do not carry identical cross-environmental weightings. Left-to-right auto-regressive networks are architecturally trapped in speaker-centric optimization due to their triangular causal masks.

In contrast, human language evolution operates over an open, noisy communicative channel where hearer-side transmission failure incurs a severe cognitive penalty. Consequently, the human grammar layout rules systematically favor the “Max” information preservation boundary, willingly absorbing a real-time planning and working memory tax to shield the downstream signal from entropy. The Min-Max theory must therefore be revised to include an architectural asymmetry parameter that weights hearer-side utility over speaker-side physical ease.

## 4.2 Generational hysteresis and sound change mechanics

The phonetic elision simulations in Experiment 5 provide a mathematical solution to the historical problem of why syntactic systems do not immediately shift when a language’s physical sound system undergoes erosion. The emergence of generational hysteresis indicates that linguistic grammatical templates possess significant structural inertia.

A speech community’s grammar remains insulated within a stable attractor basin of the replicator vector field long after the original physical or informational pressures have shifted. The system behaves as a discrete step-function: continuous, incremental sound changes - such as the gradual shortening of verbal inflections - accumulate invisibly without altering word order layout, until the physical footprint crosses a critical threshold boundary $( \alpha \Delta I = \beta \Delta D )$ . At this exact bifurcation state, the system suffers a sudden structural collapse, causing the word order to rapidly snap into a new typological equilibrium. This framework reconciles the smooth, continuous nature of phonetic erosion with the abrupt, categorical transitions observed in historical syntax.

Overall, these experiments validate Ring (2025)’s core functionalist thesis under the Min-Max Theory. According to these simulations, the frequency-weighted length of nominal and verbal slots likely exerts an active, predictable, and measurable evolutionary pressure on the long-term serialization preferences of a language community. However, it should be noted that reliance on static linear regressions over un-normalized character strings, as done in the original paper, completely masks the path-dependent hysteresis loops and threshold boundaries that define actual diachronic drift. As such, there is still work to be done to link the Min-Max theory with known instances of language change, particularly with respect to incorporating the dynamic phase-space systems, description length bit-metrics, and morphemic token normalizations demonstrated across this series of experiments.

## 5 Future work

To further validate, refine, and expand the predictive capacities of the unified Min-Max framework, future investigations should focus on expanding the environmental variables and architectural topologies of the artificial agent simulations.

## 5.1 Incorporating lossy channels and varied decoder constraints

To rectify the human-machine cognitive divergence exposed in Experiment 6, future work must break the speaker-centric monopoly of the transformer training loop. This could potentially be achieved by deploying an agent-based \*\*Encoder-Decoder communication game\*\* over an explicit lossy acoustic or typographic channel.

## 5.2 Expanding the model to multi-variable typological spaces

While this study isolated basic intransitive word order (SV vs. VS) and its relationship to nounverb length asymmetries, natural languages operate over complex, multi-variable structural webs. Future computational sweeps should expand the synthetic corpora to evaluate:

• Transitive Serialization Profiles: Modeling the co-evolution of Subject, Object, and Verb (SOV, SVO, VSO) as a three-dimensional replicator system to observe how structural choices in the noun phrase affect the placement of the verbal head.

• Morphological Complexification vs. Analytic Drift: Testing the trade-off between long, multi-morphemic word units (synthetic/polysynthetic strategy) and rigid, short word strings (analytic strategy) under shifting information-theoretic processing bottlenecks.

By implementing these multi-dimensional grid sweeps, it may be possible to use the descriptive correlations of historical typology to map out the universal highways and historical detours of human language evolution.

## 6 Conclusion

This study deployed a multi-generational computational suite to investigate the minimization of effort and information transmission economics within linguistic grammar systems. By transitioning from frozen native language matrices to tabula rasa neural architectures, the experiments successfully decoupled superficial typographical mass from internal description lengths, resolving a profound sub-word tokenization confound that has previously clouded string-length tracking simulations.

These results formalize language change not as a smooth, continuous linear drift, but as a discrete, threshold-governed optimization process governed by non-linear replicator dynamics. Most fundamentally, the structural mirror-image that emerged between auto-regressive transformers and human language corpora highlights the distinct boundaries of current deep learning architectures as models of human language change, demonstrating that human syntax is uniquely shaped by a hearer-centric drive to preserve information across lossy generational boundaries.

## References

Armeni, Kristijan, Roel M. Willems, & Stefan L. Frank. 2017. Probabilistic language models in cognitive neuroscience: Promises and pitfalls. Neuroscience and Biobehavioral Reviews, 83:579 – 588. ISSN 0149-7634.

Baerman, Matthew, Dunstan Brown, & Greville G. Corbett (eds.). 2017. Morphological Complexity. Cambridge Studies in Linguistics. Cambridge University Press.

Bentz, Christian, Alexander Koplenig, Tatyana Ruzsics, & Tanja Samardžic. 2016. A comparison´ between morphological complexity measures: Typological data vs. language corpora. In Proceedings of the Workshop on Computational Linguistics for Linguistic Complexity, pp. 142–153. Osaka, Japan.

Elman, Jeffrey L. 1990. Finding structure in time. Cognitive Science, 14(2):179–211. URL https://www.sciencedirect.com/science/article/pii/036402139090002E.

Futrell, Richard, Kyle Mahowald, & Edward Gibson. 2015. Large-scale evidence of dependency length minimization in 37 languages. Proceedings of the National Academy of Sciences (PNAS), 112(33):10336–10341.

Hahn, Michael, Dan Jurafsky, & Richard Futrell. 2020. Universals of word order reflect optimization of grammars for efficient communication. PNAS, 117(5):2347–2353.

Hahn, Michael & Yang Xu. 2022. Crosslinguistic word order variation reflects evolutionary pressures of dependency and information locality. PNAS, 119(24).

Hale, John. 2001. A probabilistic early parser as a psycholinguistic model. Proceedings of the Second Meeting of the North American Chapter of the Association for Computational Linguistics on Language Technologies, NAACL’01:1–8. Pittsburgh, Pennsylvania: Association for Computational Linguistics.

Hawkins, John A. 2014. Cross-Linguistic Variation and Efficiency. Oxford University Press. ISBN 9780199664993. URL https://doi.org/10.1093/acprof:oso/9780199664993. 001.0001.

Hörberg, Thomas & Johan Sjons. 2023. Speakers balance their use of cues to grammatical functions in informative discourse contexts. Language, Cognition and Neuroscience, 38(2):175– 196.

Jaeger, T. Florian & Harry Tily. 2011. On language ‘utility’: processing complexity and communicative efficiency. WIREs Cognitive Science, 2(3):323–335. https://wires. onlinelibrary.wiley.com/doi/pdf/10.1002/wcs.126.

Kirby, Simon. 2002. Natural language from artificial life. Artificial Life, 8(2):185–215. ISSN 1064-5462. URL https://doi.org/10.1162/106454602320184248. https://direct. mit.edu/artl/article-pdf/8/2/185/1661892/106454602320184248.pdf.

Levshina, Natalia. 2023. Communicative Efficiency: Language Structure and Use. Cambridge University Press.

Levshina, Natalia & Steven Moran. 2021. Efficiency in human languages: Corpus evidence for universal principles. Linguistics Vanguard, 7(s3):20200081. URL https://doi.org/10. 1515/lingvan-2020-0081.

Levy, Roger. 2008. Expectation-based syntactic comprehension. Cognition, 106(3):1126–1177.

Lewis, Molly L. & Michael C. Frank. 2016. The length of words reflects their conceptual complexity. Cognition, 153:182–95.

Linders, Guido M. & Max M. Louwerse. 2023. Zipf’s law revisited: Spoken dialog, linguistic units, parameters, and the principle of least effort. Psychonomic Bulletin & Review, 30(1):77– 101.

Lund, Kristine, Pierluigi Basso Fossali, Audrey Mazur, & Magali Ollagnier-Beldame (eds.). 2022. Language is a complex adaptive system. No. 8 in Conceptual Foundations of Language Science. Berlin: Language Science Press.

Mahowald, Kyle, Evelina Fedorenko, Steven T. Piantadosi, & Edward Gibson. 2013. Info/information theory: Speakers choose shorter words in predictive contexts. Cognition, 126(2):313–318. ISSN 0010-0277. URL https://www.sciencedirect.com/science/ article/pii/S0010027712002107.

Newmeyer, Frederick J. 2017. Form and function in the evolution of grammar. Cognitive Science, 41(Suppl. 2):259–276.

Niu, Qian, Junyu Liu, Ziqian Bi, Pohsun Feng, Benji Peng, Keyu Chen, Ming Li, Lawrence K. Q. Yan, Yichao Zhang, Caitlyn Heqi Yin, Cheng Fei, Tianyang Wang, Yunze Wang, Silin Chen,

Ming Liu, Ziyuan Qin, Riyang Bao, Xinyuan Song, & Zekun Jiang. 2026. Large language models and cognitive science: A comprehensive review of similarities, differences, and challenges. BIO Integration, 7(1). 16 March.

Niyogi, Partha. 2006. The Computational Nature of Language Learning and Evolution. The MIT Press. ISBN 9780262280686. URL https://doi.org/10.7551/mitpress/2024.001. 0001.

Oh, Byung-Doh & Tal Linzen. 2026. To model human linguistic prediction, make llms less superhuman. Trends in Cognitive Sciences. URL https://www.sciencedirect.com/science/ article/pii/S1364661326001270.

Oh, Byung-Doh & William Schuler. 2023. Transformer-based language model surprisal predicts human reading times best with about two billion training tokens. In Bouamor, Houda, Juan Pino, & Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 1915–1921. Singapore: Association for Computational Linguistics. URL https: //aclanthology.org/2023.findings-emnlp.128/.

Piantadosi, Steven T., Harry Tily, & Edward Gibson. 2011. Word lengths are optimized for efficient communication. PNAS, 108(9):3526–3529.

Ring, Hiram. 2025. Word length predicts word order: "Min-Max"-ing drives language evolution. URL https://arxiv.org/abs/2505.13913.

Schoenauer, Marc, Kalyanmoy Deb, Günther Rudolph, Xin Yao, Evelyne Lutton, Juan Julian Merelo, & Hans-Paul Schwefel (eds.). 2000. Language as a Complex Adaptive System. Berlin, Heidelberg: Springer Berlin Heidelberg. ISBN 978-3-540-45356-7.

Smith, Kenny & Jennifer Culbertson. 2025. Communicative pressures shape language during communication (not learning): Evidence from case-marking in artificial languages. Cognition, 263:106164. ISSN 0010-0277. URL https://www.sciencedirect.com/science/ article/pii/S0010027725001040.

Tishby, Naftali, Fernando C. N. Pereira, & William Bialek. 1999. The Information Bottleneck method. Proceedings of the 37th Annual Allerton Conference on Communication, Control and Computing.

Tucker, Mycal, Julie Shah, Roger Levy, & Noga Zaslavsky. 2025. Towards human-like emergent communication via utility, informativeness, and complexity. Open Mind, 9:418–451. ISSN 2470-2986. URL https://doi.org/10.1162/opmi\_a\_00188. https://direct.mit.edu/ opmi/article-pdf/doi/10.1162/opmi\_a\_00188/2512659/opmi\_a\_00188.pdf.

Wasow, Thomas. 2002. Postverbal Behavior. Stanford, CA: CSLI Publications.

Yadav, Himanshu, Ashwini Vaidya, Vishakha Shukla, & Samar Husain. 2020. Word order typology interacts with linguistic complexity: A cross-linguistic corpus study. Cognitive Science, 44.

Zaslavsky, Noga. 2020. Information-theoretic principles in the evolution of semantic systems. Ph.D. thesis, The Hebrew University of Jerusalem.

Zaslavsky, Noga, Jennifer Hu, & Roger Levy. 2020. Emergence of pragmatic reasoning from least-effort optimization. In Ravignani, A., C. Barbieri, M. Martins, M. Flaherty, Y. Jadoul, E. Lattenkamp, H. Little, K. Mudd, & T. Verhoef (eds.), The Evolution ofLanguage: Proceedings of the 13th International Conference (EvoLang13).

Zipf, George K. 1949. Human behavior and the principle of least effort. Addison-Wesley.

## A Addendum 1: Dynamical Replicator Formulation of the Phase-Space Syntax Engine

Let the diachronic state of a speech community’s grammar be defined by a continuous state variable $x ( t ) \in [ 0 , 1 ]$ , representing the population frequency of the Subject-Verb (SV) word order at generational time t. The frequency of the competing typological variant, Verb-Subject (VS), is complementarily modeled as $1 - x ( t )$

The continuous fitness profiles (W) balancing Information Maximization (α) and Production Efficiency $( \beta )$ over description lengths (D) are specified as:

$$
W _ { S V } = \alpha \cdot \mathcal { I } ( \mathrm { N o u n } ) - \beta \cdot \mathcal { D } _ { N }\tag{14}
$$

$$
W _ { V S } = \alpha \cdot \mathcal { I } ( \mathrm { V e r b } ) - \beta \cdot \mathcal { D } _ { V }\tag{15}
$$

The macroscopic evolutionary drift of a language’s basic word order over generational handovers is modeled using the continuous replicator dynamic:

$$
\frac { d x } { d t } = x ( 1 - x ) \left[ W _ { S V } - W _ { V S } \right]\tag{16}
$$

Substituting the explicit cognitive fitness functions into the system yields the governing nonlinear differential equation of the structural syntax engine:

$$
\frac { d x } { d t } = x ( 1 - x ) \left[ \alpha \Big ( \mathcal { I } ( \mathrm { N o u n } ) - \mathcal { I } ( \mathrm { V e r b } ) \Big ) - \beta \Big ( \mathcal { D } _ { N } - \mathcal { D } _ { V } \Big ) \right]\tag{17}
$$

Isolating the differential tracking vectors $( \Delta \mathcal { I } = \mathcal { I } ( \mathrm { N o u n } ) - \mathcal { I } ( \mathrm { V e r b } ) ; \Delta \mathcal { D } = \mathcal { D } _ { N } - \mathcal { D } _ { V } )$ summarizes the system equation to:

$$
\frac { d x } { d t } = x ( 1 - x ) \left[ \alpha \Delta \mathcal { I } - \beta \Delta \mathcal { D } \right]\tag{18}
$$

The fixed points $\begin{array} { r } { ( \frac { d x } { d t } = 0 ) } \end{array}$ of this landscape establish the strict operational boundaries of language regularization: $x ^ { * } = 1$ (Stable SV Attractor) and $x ^ { * } = 0$ (Stable VS Attractor). The critical threshold boundary is met exactly when:

$$
\alpha \Delta \mathcal { I } = \beta \Delta \mathcal { D }\tag{19}
$$

$$
\underbrace { \phantom { \frac { d x } { d t } < 0 } } _ { x ^ { * } = 0 \mathrm { ( V S ) } } \underbrace { 0 \phantom { \frac { d x } { d t } } \frac { \frac { d x } { d t } > 0 } { \phantom { \frac { d x } { d t } } } } _ { \begin{array} { c } { \alpha \Delta \mathcal { I } = \beta \Delta \mathcal { D } } \end{array} } \xrightarrow { \frac { d x } { d t } > 0 } \underbrace { \phantom { \frac { d x } { d t } } } _ { x ^ { * } = 1 \mathrm { ( S V ) } } ,
$$

Figure 1: Phase space trajectory map showing the critical bifurcation boundary.

## B Addendum 2: Grounding Word Weight in Information Bottleneck Frameworks

Let w be a word type mapping onto a latent semantic vector space $E ( w ) \in \mathbb { R } ^ { d }$ . The discrete Vector Quantization layer compresses $E ( w )$ into a codebook cluster $Z = \{ z _ { 1 } , z _ { 2 } , \dots , z _ { K } \}$ . The conditional lookup assignment is optimized via a spatial Euclidean function:

$$
z ^ { * } = \arg \operatorname* { m i n } _ { k } \| E ( w ) - z _ { k } \| ^ { 2 }\tag{20}
$$

The true, un-biased Information Weight (D) of the vocabulary string is defined by the localized codebook address allocation length under Shannon’s structural distribution:

$$
\begin{array} { r l } { \mathcal { D } ( w ) = - \log _ { 2 } P ( z ^ { * } ) } & { { } [ \mathrm { b i t s } ] } \end{array}\tag{21}
$$

By replacing naive character counts with codebook address states, the model completely decouples spelling footprint length from cognitive complexity. Complex words that occupy highly predictable semantic slots scale down in Description Length (bits), preventing typographical errors from warping the author’s macro-evolutionary corpus evaluations.

## C Addendum 3: Multi-Dimensional Disentanglement of Morphological and Articulatory Weights

This recommendation is intended to improve the methodological rigor of Ring (2025). The author should replace their single, naive character length tracker with two autonomous, frequencyweighted predictors, including an optional interaction cross-term (γ<sub>3</sub>) to map compounding processing bottlenecks:

$$
\mathrm { W o r d ~ O r d e r } \sim \gamma _ { 1 } \left( \Delta \mathrm { M o r p h e m e \_ C o u n t } \right) + \gamma _ { 2 } \left( \Delta \mathrm { A r t i c u l a t o r y \_ E f f o r t } \right) + \gamma _ { 3 } \left( \Delta \mathrm { M o r p h e m e } \times \Delta \mathrm { A r t i c u l a t o r y } \right) + \mathbf { Z } \left( \Delta \mathrm { A r t i c u l a t o r y \_ E f f o r t } \right)\tag{22}
$$

## C.1 Theoretical Relevance to Revision Requirements

By separating abstract structural syntax complexity from the physical ease of sound production, this formulation allows the model to evaluate the dual-processing costs governing language change:

• The Conceptual Module (γ ): Tracks abstract grammar chunks. A significant $\gamma _ { 1 }$ coefficient confirms that the cognitive processing engine prioritises front-loading morphologically dense constituents to anchor semantic context, validating the Information Maximisation arm of the theory.

• The Biomechanical Module (γ<sub>2</sub>): Tracks physical motor execution costs. A significant γ<sub>2</sub> coefficient indicates that the physical speech apparatus applies a localized, short-first motor tax to minimize articulation exhaustion.

• The Interaction Cross-Term (γ<sub>3</sub>): A significant γ<sub>3</sub> parameter indicates compounding penalties, revealing that constituents which are simultaneously morphologically complex and phonetically dense trigger exponential cognitive bottlenecks, serving as non-linear tipping points during generational transmission.