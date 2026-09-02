# Creative Generation via Multi-Agent Debate: Does Debate Suppress Diversity?

Tien Anh Nguyen, Khanh-Binh Nguyen, Van Dai Do,

Svetha Venkatesh, Hung Le

Deakin Applied Artificial Intelligence Initiative, Deakin University, Australia {tien.nguyen,binh.nguyen,v.do,svetha.venkatesh,thai.le}@deakin.edu.au

## Abstract

Creative generation tasks, such as narrative writing and scientific ideation, demand both high-quality outputs and distinct responses across independent runs to maximize exploration. Multi-Agent Debate (MAD) has shown strong quality gains on factual and reasoning tasks, making it a natural candidate for creative generation. However, we find its convergencedriven design actively suppresses output diversity across independent runs, creating an inherent trade-off with creative tasks. We theoretically show that preserving diversity among agents within each debate session is a necessary condition for achieving diverse outputs across independent runs. Building on this finding, we propose Creative-MAD, which introduces two synergistic interventions to sustain agent divergence. Specifically, Cognitive Lens Assignment counters identity drift by anchoring each agent to a distinct and persistent cognitive mode, while Embedding-based Peer Selection counters majority pull by limiting each agent’s context to its most semantically distant peers. Experiments across four creative benchmarks demonstrate that Creative-MAD significantly enhances both lexical and semantic diversity while maintaining MAD’s output quality.

## 1 Introduction

Creative generation encompasses diverse tasks including narrative writing, scientific ideation, and argumentative composition (Wu et al., 2026; Ruan et al., 2026; Huber and Niklaus, 2025). Unlike factual or reasoning tasks where a single correct answer exists, creative tasks are open-ended and fundamentally require output diversity. A system asked to generate responses to the same prompt across multiple runs should return meaningfully distinct outputs each time to explore different directions of the creative space (Zhang et al., 2025b; Ismayilzada et al., 2025). This diversity requirement makes creative generation challenging because it demands both high-quality outputs and meaningful variation across independent generations.

Multi-Agent Debate (MAD) has emerged as a promising paradigm for improving Large Language Model (LLM) output quality, where multiple model instances iteratively critique and refine each other’s responses (Du et al., 2024; Liang et al., 2024; Li et al., 2024). The appeal is intuitive: exposure to competing perspectives surfaces errors, challenges weak reasoning, and drives convergence toward higher-quality outputs across debate rounds. Recent work has validated this appeal, demonstrating consistent quality gains over single-agent generation across factual question answering, arithmetic reasoning, and commonsense inference (Choi et al., 2026), making MAD a natural candidate for creative generation where output quality equally matters.

However, MAD’s convergence-driven design introduces a fundamental tension in creative settings. Within each debate session, agents are progressively drawn toward a shared perspective as they read and respond to each other across rounds, causing diversity among agent responses to decay before a final output is selected. Since each independent run undergoes this same within-session diversity decay in isolation, the final outputs across runs end up clustering in the same narrow region, suppressing the diversity that creative generation requires. Yet such a quality-diversity trade-off has gone unnoticed in prior work, as MAD has been evaluated exclusively on output quality without examining whether independently generated outputs are meaningfully distinct from one another. We formalize the relationship between within-session diversity decay and output homogeneity across runs, showing that preserving diversity among agents within each debate session is a necessary condition for achieving diverse outputs across runs, and consequently, the only controllable lever for improving output diversity.

Motivated by this finding, we propose Creative-MAD, a diversity-preserving multi-agent debate framework that addresses within-session convergence through two complementary mechanisms. The first targets identity drift: when agents share identical prompt baselines, minor sampling variations fail to provide a permanent reasoning anchor, leaving multi-round interactions to homogenize individual agent behaviors. Cognitive Lens Assignment (CLA) addresses this by equipping each agent with a distinct and persistent cognitive mode throughout debate, designed specifically for creative generation, such as analytical, emotional, or analogical reasoning. The second targets majority pull: when each agent receives the full peer pool, dominant perspectives gradually erode minority viewpoints. Embedding-based Peer Selection (EPS) addresses this by restricting each agent’s context to its k most semantically distant peers, replacing majority pull with cross-perspective stimulus. Our evaluation demonstrates that Creative-MAD significantly enhances cross-run diversity across both semantic and lexical dimensions without sacrificing the generation quality of standard MAD.

We summarize our contributions as follows:

• We identify a fundamental quality-diversity trade-off in MAD for creative generation: while debate consistently improves output quality, the convergence dynamics among agents actively suppress output diversity across independent runs.

• We establish the theoretical basis of this tradeoff, showing that preserving diversity among agents within each debate session is a necessary condition for achieving diverse outputs across multiple sessions.

• We propose Creative-MAD, combining Cognitive Lens Assignment and Embedding-based Peer Selection to counter identity drift and majority pull within debate sessions, maintaining output quality on par with standard MAD while substantially improving output diversity across a diverse suite of creative generation benchmarks.

## 2 Related Work

## 2.1 Multi-Agent Debate

MAD has demonstrated consistent quality gains on factual and reasoning tasks (Du et al., 2024;

Choi et al., 2026), where convergence toward a single correct answer is the desired outcome. To strengthen this convergence, prior work has explored diverse agent configurations through two main strategies: model-level heterogeneity, which leverages different LLM families (Chen et al., 2024), and role-level heterogeneity, which injects distinct personas or reasoning methods into agents of the same model (Liang et al., 2024; Wang et al., 2024). However, these works treat diversity as a means to improve convergence toward a correct answer, not as an output requirement across independent runs.

MAD’s behavior on creative tasks, where output diversity is a feature rather than a flaw, has received limited attention (Becker et al., 2025). While Hu et al. (2025) achieves argument diversity through per-run persona resampling, the intrinsic diversity decay during the debate process under a fixed prompt remains largely unaddressed. Consequently, investigating such decay as a primary failure mode in creative environments remains an unexplored gap, leaving a need for mechanisms that balance quality with the preservation of diverse creative trajectories.

## 2.2 LLM Creative Generation and Evaluation

Recent benchmarks have expanded creative evaluation into specialized domains, including physical problem-solving (Tian et al., 2024), scientific ideation (Ruan et al., 2026), and narrative writing (Li et al., 2025). LLM-as-a-Judge (LLMaaJ) frameworks (Wu et al., 2026), pairwise Elo-based systems (Paech, 2023), and Win Rate (Li et al., 2025) have become widely adopted for open-ended evaluation. A critical challenge in creative generation is mode collapse, where LLMs produce repetitive outputs for the same prompt (Zhang et al., 2025b). Existing work addresses this via diversity-aware training (Chung et al., 2025) or quality-weighted diversity metrics (Shypula et al., 2025), but the multi-agent setting remains underexplored.

## 3 Problem Formulation

## 3.1 MAD Formulation

We consider a multi-agent debate system comprising N agents operating over R rounds to produce a single response for a given input query x. Formally, let $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { N } \}$ denote the set of agents, where each agent is an instance of the same language model M. We focus on this singlemodel setting to cleanly isolate the effect of debate dynamics from inherent capability differences across model families. We follow the common simultaneous-talk protocol (Chan et al., 2024; Choi et al., 2026), where all agents update concurrently based on the previous round’s responses.

At round $t = 0$ , each agent $a _ { i }$ independently generates an initial response $y _ { i } ^ { ( 0 ) } = \mathcal { M } ( x )$ . At each subsequent round $t \in \{ 1 , \ldots , R \}$ , each agent refines its response by conditioning on both the original query and a subset of peer responses from the previous round:

$$
y _ { i } ^ { ( t ) } = \mathcal { M } \left( x , \mathcal { P } _ { i } ^ { ( t - 1 ) } \right)\tag{1}
$$

where $\mathcal { P } _ { i } ^ { ( t - 1 ) } \subseteq \{ y _ { i } ^ { ( t - 1 ) } : j \neq i \}$ denotes the set of peer responses visible to agent i at round t. In standard MAD, $\mathcal { P } _ { i } ^ { ( t - 1 ) }$ contains all peer responses. After R debate rounds, a consensus mechanism selects a single final answer yˆ from the final agent responses $\{ y _ { i } ^ { ( R ) } \} _ { i = 1 } ^ { N }$ . To evaluate robustness and diversity across generations, we run G independent sessions for each query, yielding a set of final outputs $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G }$

## 3.2 Creative Task Setting

Given an input query x, the goal is to generate a response y evaluated along two complementary dimensions: instance-level quality (coherence, creativity, and task alignment) and set-level diversity (the degree to which multiple generated outputs explore distinct regions of the output space).

Formally, given G independent sessions on the same fixed query x producing outputs $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G } ,$ the objective is to jointly maximize instance-level quality $\textstyle { \frac { 1 } { G } } \sum _ { g = 1 } ^ { G }$ Quality $( \hat { y } _ { g } )$ and set-level diversity Diversity( $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G } )$ across all sessions, where the latter requires outputs to be meaningfully distinct from one another, a property we term inter-session diversity.

Instance-level quality. We measure quality via absolute scores using LLMaaJ (Ruan et al., 2026), and average pairwise win rate (Li et al., 2025) across all method pairs.

Set-level diversity. We measure diversity along two axes. For semantic diversity, we adopt the Vendi Score (Friedman and Dieng, 2023), defined as:

$$
{ \mathrm { V S } } = \exp \left( - \operatorname { t r } \left( { \bar { K } } \log { \bar { K } } \right) \right) , \quad { \bar { K } } = { \frac { K } { G } }\tag{2}
$$

where $\operatorname { t r } ( \cdot )$ denotes the matrix trace, $K \in \mathbb { R } ^ { G \times G }$ is the cosine similarity kernel matrix with entries $K _ { g g ^ { \prime } } = e _ { g } \cdot e _ { g ^ { \prime } } / \| e _ { g } \| \| e _ { g ^ { \prime } } \|$ for $g , g ^ { \prime } \in \{ 1 , \ldots , G \}$ and $e _ { g }$ denotes the embedding of final output $\hat { y } _ { g }$ The score ranges from 1 for identical outputs to G for fully distinct outputs. For lexical diversity, we use Div-BLEU (Ruan et al., 2025), calculated as $1 - { \mathrm { S e l f - B L E U } }$ (Zhu et al., 2018).

## 4 Why MAD Suppresses Diversity in Creative Generation

While MAD’s convergence dynamics yield consistent quality gains over single-agent baselines, they introduce a fundamental quality-diversity trade-off in creative settings: the same inter-agent communication that drives quality improvement actively suppresses output diversity across independent runs.

Intra-session Diversity Decay. We define intrasession diversity as the semantic spread among the N agent responses $\{ y _ { i } ^ { ( t ) } \} _ { i = 1 } ^ { N }$ at debate round $t ,$ measured by the semantic diversity score over agent outputs within a single session. Standard MAD drives monotonic decay of intra-session diversity through two distinct mechanisms. First, identity drift: as agents are initialized with identical prompts and differ only through temperature sampling, they lack a stable reasoning anchor, causing them to drift toward group consensus as they synthesize peer responses across rounds. Second, majority pull: full peer connectivity exposes each agent to the majority signal, causing dominant perspectives to gradually override minority viewpoints and homogenize agent responses. As intra-session diversity decays, the pool from which the consensus mechanism selects the final output $\hat { y } _ { g }$ becomes increasingly concentrated, causing the final outputs $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G }$ across independent sessions to cluster in the same narrow region of the output space.

Theoretical Analysis. We formalize the relationship between intra-session diversity decay and inter-session output homogeneity as follows.

Proposition 1 (Intra-to-Inter Diversity). Let $K \in$ $\mathbb { R } ^ { G \times G }$ be the cosine similarity kernel matrix over final outputs $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G }$ and ${ \mathcal { D } } _ { \mathrm { i n t e r } } = \exp ( H ( K ) )$ the inter-session diversity score where

$$
H ( K ) = - \mathrm { t r } ( \bar { K } \log \bar { K } )\tag{3}
$$

is the kernel entropy (Friedman and Dieng, 2023). A decrease in intra-session diversity necessarily induces a decrease in $\mathcal { D } _ { \mathrm { i n t e r } }$ , and preserving intrasession diversity is a necessary condition for maintaining high inter-session diversity under the fixedprompt setting.

![](images/08bd43ca298eb7fd45f02fc50ce9af3769616c5f2fbceaf44f1b1ad954581e33.jpg)  
Figure 1: Comparison of Standard MAD and Creative-MAD. Standard MAD’s uniform prompts and full peer connectivity drive convergence across debate rounds. Creative-MAD addresses this via Cognitive Lens Assignment (CLA), anchoring each agent to a distinct cognitive mode, and Embedding-based Peer Selection (EPS), restricting each agent to its k most semantically distant peers.

Proof. See Appendix B.

## 5 Creative-MAD: Diversity-Preserving Multi-Agent Debate for Creative Generation

Standard MAD suppresses intra-session diversity through two mechanisms: identity drift and majority pull. We propose Creative-MAD, with two complementary interventions targeting each mechanism: CLA addresses identity drift by anchoring each agent to a distinct processing mode throughout debate, while EPS addresses majority pull by filtering the peer signal to amplify cross-perspective stimulus. An overview of the full architecture is illustrated in Figure 1.

## 5.1 Cognitive Lens Assignment (CLA)

Creative generation is inherently multidimensional: a strong response to the same prompt may emerge from an emotionally resonant narrative, a logically structured argument, a counterintuitive reframing, or an unexpected crossdomain analogy, each constituting a legitimate and distinct creative direction. This multiplicity underlies the diversity requirement central to creative generation, yet standard MAD collapses it: since all agents share identical system prompts and differ only through temperature sampling, there is no structural mechanism to maintain distinctive generative orientations across debate rounds.

We address this by assigning each agent a distinct cognitive lens, a persistent system prompt that explicitly defines a specific cognitive processing mode. Unlike domain personas that specify “what” an agent knows (e.g., scientist, lawyer), cognitive lenses specify “how” an agent processes information, covering complementary modes such as logical analysis, emotional reasoning, critical examination, analogical thinking, and practical evaluation. Cognitive lenses are assigned deterministically across agents, ensuring full cognitive coverage within each session. The lens is explicitly reinforced in the debate prompt at every round, preventing the identity drift that occurs when agents are exposed to peer responses during synthesis. Full lens descriptions are provided in Appendix E.3.

## 5.2 Embedding-based Peer Selection (EPS)

Even with distinct cognitive lenses, agents remain susceptible to majority-pull if each agent receives the full set of peer responses during debate. When all agents observe the same peer pool, the most frequently represented perspectives dominate synthesis, gradually eroding cognitive distinctiveness regardless of lens assignment.

Table 1: Instance-level quality results across four creative benchmarks. We report rubric-based absolute scores $( \mathrm { m e a n } \pm \mathrm { s t d } )$ and average pairwise win rate (%) across all method pairs. Higher is better. Bold indicates the best result(s); multiple entries may be bolded if not statistically significantly different (paired t-test, $p < 0 . 0 5 )$
<table><tr><td rowspan="2">Method</td><td colspan="2">LiveIdea</td><td colspan="2">AAE</td><td colspan="2">MacGyver</td><td colspan="2">AH v2.0</td></tr><tr><td>Score</td><td>Win%</td><td>Score</td><td>Win%</td><td>Score</td><td>Win%</td><td>Score</td><td>Win%</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Direct</td><td> $6 6 . 5 \pm 0 . 1$ </td><td>7.6</td><td> $7 3 . 7 \pm 0 . 1$ </td><td>4.0</td><td> $6 4 . 2 \pm { 0 . 8 }$ </td><td>6.1</td><td> $7 4 . 9 \pm 0 . 3$ </td><td>10.6</td></tr><tr><td>Self-Refine</td><td> $6 9 . 7 \pm 0 . 3$ </td><td>29.3</td><td> $7 4 . 7 \pm 0 . 1$ </td><td>8.9</td><td> $6 8 . 2 \pm 0 . 1 $ </td><td>18.5</td><td> $7 6 . 6 \pm 0 . 2$ </td><td>17.3</td></tr><tr><td>Voting</td><td> $6 7 . 2 \pm 0 . 1$ </td><td>11.8</td><td> $7 4 . 2 \pm 0 . 4$ </td><td>5.8</td><td> $6 6 . 9 \pm 0 . 3$ </td><td>12.7</td><td> $7 6 . 5 \pm 0 . 3$ </td><td>19.0</td></tr><tr><td>Homo MAD</td><td> ${ \bf 7 4 . 0 \pm 0 . 1 }$ </td><td>52.7</td><td> $8 0 . 6 \pm 0 . 5$ </td><td>57.1</td><td> $7 6 . 3 \pm 0 . 3$ </td><td>63.4</td><td> $8 0 . 2 \pm 0 . 3$ </td><td>44.3</td></tr><tr><td>Hetero MAD</td><td> $7 2 . 2 \pm 0 . 2$ </td><td>51.8</td><td> ${ \bf 8 2 . 3 \pm 0 . 3 }$ </td><td>63.7</td><td> $7 6 . 6 \pm 0 . 1$ </td><td>65.3</td><td> ${ \bf 8 1 . 4 \pm 0 . 3 }$ </td><td>51.2</td></tr><tr><td>Creative-MAD</td><td> $7 1 . 5 \pm 0 . 1$ </td><td>49.4</td><td> ${ \pm } 2 . 8 \pm 0 . 4$ </td><td>65.1</td><td> $7 6 . 4 \pm 0 . 3$ </td><td>65.6</td><td> $8 0 . 4 \pm 0 . 9$ </td><td>45.7</td></tr><tr><td colspan="9">Gemma-3-12B-Instruct</td></tr><tr><td>Direct</td><td> $7 5 . 3 \pm 0 . 2$ </td><td>8.9</td><td> $8 2 . 1 \pm 0 . 1$ </td><td>5.3</td><td> $7 6 . 9 \pm 0 . 3$ </td><td>10.4</td><td> $7 8 . 0 \pm 0 . 3$ </td><td>11.6</td></tr><tr><td>Self-Refine</td><td> $8 1 . 8 \pm 0 . 4$ </td><td>26.7</td><td> $8 3 . 9 \pm 0 . 3$ </td><td>8.9</td><td> $7 8 . 9 \pm 0 . 4$ </td><td>23.8</td><td> $8 0 . 2 \pm 0 . 1$ </td><td>23.4</td></tr><tr><td>Voting</td><td> $7 6 . 3 \pm 0 . 7$ </td><td>15.7</td><td> $8 2 . 4 \pm 0 . 2$ </td><td>9.6</td><td> $7 7 . 4 \pm 0 . 2$ </td><td>19.9</td><td> $8 0 . 9 \pm 0 . 2$ </td><td>23.9</td></tr><tr><td>Homo MAD</td><td> ${ \pm } 2 . 5 \pm 0 . 3$ </td><td>55.6</td><td> $8 4 . 9 \pm 0 . 5$ </td><td>62.1</td><td> ${ \bf 8 1 . 1 \pm 0 . 5 }$ </td><td>66.1</td><td> ${ \bf 8 4 . 3 \pm 0 . 3 }$ </td><td>68.2</td></tr><tr><td>Hetero MAD</td><td> $8 1 . 9 \pm 0 . 2$ </td><td>51.1</td><td> ${ \bf 8 5 . 0 \pm 0 . 3 }$ </td><td>65.8</td><td> ${ \bf 8 1 . 0 \pm 0 . 4 }$ </td><td>65.4</td><td> ${ \bf 8 4 . 1 \pm 0 . 3 }$ </td><td>67.5</td></tr><tr><td>Creative-MAD</td><td> ${ \bf 8 2 . 2 \pm 0 . 4 }$ </td><td>55.4</td><td> ${ \bf 8 5 . 1 \pm 0 . 4 }$ </td><td>65.7</td><td> $8 0 . 2 \pm 0 . 2$ </td><td>62.9</td><td> ${ \bf 8 4 . 2 \pm 1 . 2 }$ </td><td>68.0</td></tr></table>

We address this through EPS, which filters the peer signal each agent receives. At each debate round, all agent responses are embedded into a shared semantic space via an embedding encoder E, where $e _ { i } ^ { ( t ) } = \bar { \mathcal { E } } ( y _ { i } ^ { ( t ) } )$ . Each agent then receives only its k most semantically distant peers as debate input, rather than the full peer pool. Formally, at debate round $t \geq 1$ , for agent i with current response $y _ { i } ^ { ( t ) }$ , the peer set is defined as:

$$
\mathcal { P } _ { i } ^ { ( t ) } = \underset { j \neq i } { \operatorname* { t o p } } { k } \left( 1 - \cos ( e _ { i } ^ { ( t ) } , e _ { j } ^ { ( t ) } ) \right)\tag{4}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity. By selecting the most semantically distant peers, EPS ensures that the creative stimulus each agent receives is maximally different from its own current direction, amplifying exploration over exploitation.

## 6 Experiments

## 6.1 Experimental Setup

Benchmarks. We evaluate across four benchmarks spanning diverse dimensions of creative generation: LiveIdeaBench (scientific ideation) (Ruan et al., 2026), Argument Annotated Essays (AAE) (argumentative writing) (Huber and Niklaus, 2025), MacGyver (creative problem-solving) (Tian et al.,

2024), and Arena Hard v2.0 (AH v2.0) (general creative writing) (Paech, 2023). Appendix A.1 provides dataset details.

Baselines. We compare against the following methods: (1) Direct, a single-agent baseline; (2) $S e l f { - } R e f i n e$ (Madaan et al., 2023), which iteratively refines a response using self-feedback; (3) Voting, which aggregates initial responses from N agents without debate; (4) Homogeneous MAD (Homo MAD) (Du et al., 2024), the standard decentralized multi-agent debate; (5) Heterogeneous MAD (Hetero MAD) (Choi et al., 2026), which assigns distinct domain personas to agents before debate; and (6) Creative-MAD<sup>1</sup> (ours), which combines CLA and EPS to preserve intra-session diversity. All methods use N = 5, R = 2 rounds, and temperature = 1.0. Creative-MAD additionally uses k = 2 for EPS. Since creative tasks produce openended outputs that cannot be compared by exact match or majority voting, all voting and debatebased methods employ a judge instantiated with the same underlying model as the debating agents to select the best candidate at the consensus stage.

Evaluation. For instance-level quality, we employ an LLMaaJ using Qwen3.5-397B-A17B (Qwen Team, 2026), reporting rubric-based absolute scores and average pairwise win rate computed across all runs per query (Li et al., gains beyond simple response aggregation.

Table 2: Set-level diversity results across benchmarks. We report semantic diversity score (Sem., mean ± std) and lexical diversity score (Lex., mean ± std, scaled by ×100). Higher is better. Bold indicates the best result(s); multiple entries may be bolded if not statistically significantly different (paired t-test, $p < 0 . 0 5 )$
<table><tr><td rowspan="2">Method</td><td colspan="2">LiveIdea</td><td colspan="2">AAE</td><td colspan="2">MacGyver</td><td colspan="2">AH v2.0</td></tr><tr><td>Sem.</td><td> $\operatorname { L e x } .$ </td><td>Sem.</td><td>Lex.</td><td>Sem.</td><td>Lex.</td><td> $\operatorname { s e m } .$ </td><td>Lex.</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Direct</td><td> $2 . 5 4 \pm 0 . 0 5$ </td><td> $6 8 . 9 6 \pm 1 . 1$ </td><td> $1 . 5 2 \pm 0 . 0 4$ </td><td> $5 2 . 5 6 \pm 0 . 9$ </td><td> $1 . 5 2 \pm 0 . 0 5$ </td><td> $5 5 . 0 9 \pm 1 . 2$ </td><td> $2 . 2 4 \pm 0 . 0 4$ </td><td> $7 1 . 9 8 \pm 0 . 8$ </td></tr><tr><td>Self-Refine</td><td> $2 . 5 9 \pm 0 . 0 4$ </td><td> $7 5 . 7 0 \pm 1 . 4$ </td><td> $1 . 5 8 \pm 0 . 0 5$ </td><td> $5 6 . 8 6 \pm 1 . 1$ </td><td> $1 . 5 3 \pm 0 . 0 4$ </td><td> $5 7 . 8 5 \pm 1 . 3$ </td><td> $2 . 2 5 \pm 0 . 0 5$ </td><td> $7 2 . 0 0 \pm 0 . 9$ </td></tr><tr><td>Voting</td><td> $2 . 5 8 \pm 0 . 0 6$ </td><td> $6 9 . 6 8 \pm 0 . 9$ </td><td> $1 . 5 4 \pm 0 . 0 4$ </td><td> $5 3 . 6 1 \pm 1 . 2$ </td><td> $1 . 5 2 \pm 0 . 0 5$ </td><td> $5 5 . 7 5 \pm 0 . 9$ </td><td> $2 . 2 1 \pm 0 . 0 6$ </td><td> $7 1 . 8 5 \pm 1 . 4$ </td></tr><tr><td>Homo MAD</td><td> $2 . 4 7 \pm 0 . 0 4$ </td><td> $6 6 . 6 2 \pm 1 . 3$ </td><td> $1 . 4 7 \pm 0 . 0 5$ </td><td> $5 2 . 3 1 \pm 0 . 8$ </td><td> $1 . 4 8 \pm 0 . 0 4$ </td><td> $5 5 . 6 3 \pm 1 . 2$ </td><td> $2 . 2 1 \pm 0 . 0 4$ </td><td> $6 8 . 5 1 \pm 1 . 1$ </td></tr><tr><td>Hetero MAD</td><td> $2 . 5 8 \pm 0 . 0 5$ </td><td> $7 0 . 8 7 \pm 1 . 0$ </td><td> $1 . 6 7 \pm 0 . 0 6$ </td><td> $5 8 . 9 9 \pm 1 . 5$  –</td><td> $1 . 5 3 \pm 0 . 0 5$ </td><td> $5 7 . 1 9 \pm 0 . 8$  </td><td> $2 . 2 7 \pm 0 . 0 5$ </td><td> $7 2 . 7 2 \pm 1 . 2$ </td></tr><tr><td>Creative-MAD</td><td> ${ \bf 3 . 0 4 } \pm 0 . 0 6$ </td><td> ${ \bf 8 3 . 2 0 \pm 1 . 2 }$ </td><td> $2 . 2 4 \pm 0 . 0 4$ </td><td> ${ \bf 6 9 . 2 4 \pm 1 . 3 }$ </td><td> ${ \bf 1 . 6 7 \pm 0 . 0 4 }$ </td><td> ${ \bf 6 6 . 8 1 \pm 1 . 0 }$ </td><td> ${ \pm . 6 8 \pm 0 . 0 5 }$  一</td><td> $\mathbf { 8 2 . 0 5 \pm 1 . 2 }$ </td></tr><tr><td colspan="9">Gemma-3-12B-Instruct</td></tr><tr><td>Direct</td><td> $2 . 6 9 \pm 0 . 0 4$ </td><td> $6 9 . 1 2 \pm 0 . 9$ </td><td> $1 . 7 1 \pm 0 . 0 5$ </td><td> $5 0 . 0 7 \pm 1 . 2$ </td><td> $1 . 6 7 \pm 0 . 0 5$ </td><td> $4 7 . 7 0 \pm 1 . 4$ </td><td> $2 . 6 4 \pm 0 . 0 4$ </td><td> $6 4 . 7 3 \pm 0 . 8$ </td></tr><tr><td>Self-Refine</td><td> $2 . 7 2 \pm 0 . 0 5$ </td><td> $7 3 . 0 5 \pm 1 . 4$ </td><td> $1 . 7 7 \pm 0 . 0 4$ </td><td> $5 3 . 5 7 \pm 0 . 9$ </td><td> $1 . 6 7 \pm 0 . 0 6$ </td><td> $5 2 . 1 2 \pm 1 . 1$ </td><td> $2 . 5 7 \pm 0 . 0 5$ </td><td> $6 3 . 8 9 \pm 1 . 3$ </td></tr><tr><td>Voting</td><td> $2 . 6 6 \pm 0 . 0 4$ </td><td> $6 9 . 3 2 \pm 1 . 1$ </td><td> $1 . 7 2 \pm 0 . 0 5$ </td><td> $5 0 . 2 8 \pm 1 . 3$ </td><td> $1 . 6 6 \pm 0 . 0 4$ </td><td> $4 8 . 6 0 \pm 0 . 9$ </td><td> $2 . 6 1 \pm 0 . 0 4$ </td><td> $6 4 . 8 6 \pm 1 . 0$ </td></tr><tr><td>Homo MAD</td><td> $2 . 4 4 \pm 0 . 0 5$ </td><td> $6 3 . 2 8 \pm 1 . 4$ </td><td> $1 . 6 8 \pm 0 . 0 4$ </td><td> $4 9 . 4 6 \pm 1 . 1$ </td><td> $1 . 6 0 \pm 0 . 0 5$ </td><td> $4 8 . 3 4 \pm 1 . 2$ </td><td> $2 . 5 8 \pm 0 . 0 6$ </td><td> $6 3 . 0 7 \pm 1 . 5$ </td></tr><tr><td>Hetero MAD</td><td> $2 . 7 0 \pm 0 . 0 6$ </td><td> $7 0 . 5 8 \pm 0 . 8$ </td><td> $1 . 7 5 \pm 0 . 0 6$ </td><td> $5 2 . 4 6 \pm 1 . 2$ </td><td> $1 . 6 8 \pm 0 . 0 4$ </td><td> $5 2 . 5 0 \pm 1 . 3$ </td><td> $2 . 6 5 \pm 0 . 0 5$ </td><td> $6 4 . 4 9 \pm 1 . 0$ </td></tr><tr><td>Creative-MAD</td><td> ${ \bf 3 . 5 1 \pm 0 . 0 5 }$ </td><td> ${ \bf 8 1 . 9 7 \pm 1 . 2 }$ </td><td> ${ \bf 1 . 9 2 \pm 0 . 0 5 }$ </td><td> ${ \bf 6 1 . 0 5 \pm 1 . 4 }$ </td><td> ${ \bf 1 . 8 2 \pm 0 . 0 6 }$ </td><td> ${ \pm } 9 . 7 3 \pm 1 . 0$  1</td><td> ${ \bf 2 . 9 8 \pm 0 . 0 4 }$  </td><td> $7 5 . 1 2 \pm 1 . 1$ </td></tr></table>

2025). For set-level diversity, we measure semantic diversity via Vendi Score (Friedman and Dieng, 2023) computed over all-MiniLM-L6-v2 representations (Wang et al., 2020) and lexical diversity via $D i \nu  – B L E U$ (Ruan et al., 2025). Each method is independently run 15 times per query. For instance-level quality, we report the mean and standard deviation across all 15 runs. For set-level diversity, we set $G = 5$ , partition the 15 runs into 3 groups, and report the mean and standard deviation of diversity scores across groups.

Models. Experiments use two open-source models: Qwen3-8B (Yang et al., 2025) and Gemma-3- 12B-Instruct (Team et al., 2025).

## 6.2 Main Results

Set-level Diversity Results. Table 2 presents setlevel diversity results across methods and benchmarks. Creative-MAD consistently achieves the highest semantic and lexical diversity, outperforming Homo MAD by 26.2% (semantic) and 24.0% (lexical) on Qwen3-8B, and by 23.3% and 24.0% on Gemma-3-12B-Instruct. Conversely, Homo MAD yields the lowest semantic diversity, falling below single-agent baselines. While Hetero MAD recovers some diversity through domain-role differentiation, its semantic diversity remains significantly below Creative-MAD by 19.7% on Qwen3- 8B and 16.5% on Gemma-3-12B-Instruct, which indicates that persona assignment alone fails to counteract the convergence pressure of debate.

Instance-level Quality Results. Table 1 presents instance-level quality results across all methods and benchmarks. Creative-MAD maintains quality on par with Homo MAD and Hetero MAD, ranking among the top-performing methods in 5 out of 8 comparisons on both absolute score and win rate across both models. The average score gap to the best-performing method remains within 1.2% on Qwen3-8B and 0.4% on Gemma-3-12B-Instruct, and the average win rate gap remains within 2.2% on Qwen3-8B and 0.9% on Gemma-3-12B-Instruct, demonstrating that the diversity-preserving mechanisms do not interfere with the quality gains that debate provides. Furthermore, all MAD variants consistently outperform single-agent baselines on both metrics, including Voting, which uses the same number of agents but without debate, confirming that inter-agent communication contributes quality

Notably, Voting achieves diversity comparable to Direct despite applying the same judge-based consensus mechanism to N independent responses. This confirms that the consensus mechanism does not drive diversity collapse, thereby isolating debate dynamics as the primary driver of inter-session diversity suppression.

## 6.3 Human Evaluation

To validate that our automated evaluation reflects human judgment, we conduct two complementary human studies; full protocols and results are in Appendix G.2.

Judge reliability. Since our main results rely on Qwen3.5-397B-A17B as judge, we first verify that its quality assessments track human preference. Three annotators pairwise-compare 120 method outputs (Direct/Self-Refine vs. Homo MAD) across all four benchmarks. Human annotators reach 79.0% pairwise agreement among themselves, and our LLMaaJ agrees with the human majority vote in 81.0% of cases, on par with human-level agreement.

![](images/62abc0a3177702388d99da791fb832001d405569150f8c7e0edbec0e33413ccd.jpg)  
Figure 2: Mean quality score of Creative-MAD across debate rounds on Qwen3-8B.

Creative-MAD vs. Homo MAD. To confirm that Creative-MAD’s diversity gains and quality parity also hold under direct human judgment, we run a pilot study comparing Creative-MAD against Homo MAD on 40 queries. For quality, human judgments are split closely between the two methods, with ties as the most common outcome, consistent with our claim that Creative-MAD preserves output quality. For diversity, human annotators judge Creative-MAD’s output set as more diverse in 35/40 queries, matching the Vendi Score’s preference in 88% of cases. Together, these studies support that both our quality and diversity findings align with human judgment.

## 7 Model Analysis

## 7.1 Quality Improvement Across Debate Rounds

Figure 2 shows that Creative-MAD’s quality improves progressively across debate rounds, confirming that debate drives quality gains beyond mere response aggregation. This behavior stands in contrast to the martingale property reported by Choi et al. (2026) on factual tasks, where agent beliefs remain stationary in expectation across debate rounds. We attribute this discrepancy to a fundamental difference in task structure: unlike factual tasks, creative tasks involve open-ended response spaces and continuous quality signals, violating the categorical and Bayesian conjugacy assumptions that the martingale result relies on (see Appendix C for details).

![](images/ed9d2b8f1f3e785b5c5b2104de633b51b7baf22a83affce45a45fdb2b3b624de.jpg)  
Figure 3: Intra-session semantic diversity across debate rounds for each method variant on Qwen3-8B, averaged across four benchmarks.

## 7.2 Intra-session Diversity

To analyze the contribution of each individual component, we evaluate two variants: CLA only, which applies CLA without EPS, and EPS only, which applies EPS without CLA.

Figure 3 illustrates intra-session semantic diversity across debate rounds for each method variant on Qwen3-8B. At t = 0, Creative-MAD, CLA only, and Hetero MAD begin at comparable diversity levels, while EPS only starts lower as it does not affect agent initialization. Across subsequent rounds, Hetero MAD decays at a rate comparable to Homo MAD despite its higher starting point, confirming that domain personas fail to sustain cognitive distinctiveness under multi-turn peer pressure. CLA only decays more slowly, while EPS only, despite starting lower, exhibits the slowest decay rate, confirming that filtering peer signals is more effective at counteracting majority pull. Creative-MAD combines both effects, achieving the strongest overall diversity preservation, providing a mechanistic explanation for the inter-session diversity gains in Table 2.

## 7.3 Inter-session Diversity

Figure 4 visualizes the semantic space of final outputs across five independent sessions via PCA projection, providing geometric evidence for Proposition 1. Under Homo MAD, outputs from all sessions collapse into a single dense, heavily overlapping region, confirming that convergence-driven debate forces independent sessions to occupy the same narrow semantic space. Conversely, Creative-MAD exhibits a markedly different topography: outputs within each session are highly distributed, with distinct sessions occupying separated regions of the semantic space. This empirically validates that preserving intra-session diversity directly unlocks higher inter-session diversity.

![](images/04aa3731e4355366a8897edd80faedf2c808a624d8f84a1a6087180e8c036ace.jpg)  
Figure 4: Inter-session diversity topography of Homo MAD vs. Creative-MAD (5 sessions, PCA projection). Colors denote individual sessions; contours indicate kernel density estimates.

![](images/16ff6bcd63231fb26042b26469ce0f0481adabd71380eac797f7dae22d3285a1.jpg)  
Figure 5: Semantic diversity versus quality score for Creative-MAD ablation variants and baselines on Qwen3-8B, averaged across four benchmarks.

## 7.4 Ablation Study

Figure 5 compares semantic diversity against quality for Creative-MAD ablation variants and baselines using Qwen3-8B, averaged across four benchmarks. All methods maintain comparable quality scores, confirming that diversity gains do not compromise output quality. CLA only yields a substantial diversity improvement over Homo MAD and consistently outperforms Hetero MAD. This demonstrates that anchoring agents to distinct cognitive modes is more effective at preserving diversity during debate than standard domain persona assignment. EPS only similarly improves diversity, confirming that filtering peer signals reduces majority pull even without a stable cognitive identity. Combining both components yields the largest diversity gains, demonstrating a clear synergy: CLA injects distinct perspectives at initialization, while EPS slows within-session convergence via selective peer exposure. Together, they effectively preserve the intra-session diversity required for inter-session gains.

![](images/7f6bae7338f4c20e9351ba86c48d362a32afa2d266072a8c0fb58e1686376a92.jpg)  
Figure 6: Sensitivity of Creative-MAD to the peer selection hyperparameter k on Qwen3-8B.

## 7.5 Sensitivity to k

Figure 6 illustrates how Creative-MAD performs across different values of the peer selection hyperparameter k. Semantic diversity decreases monotonically as k increases, which is expected: larger k exposes each agent to more peers, gradually approaching the full peer pool of standard MAD and reintroducing majority pull. Conversely, quality score peaks at $k = 2$ and drops slightly at $k = 3 ,$ while dropping noticeably at $k = 1$ , suggesting that a single peer provides insufficient context for meaningful debate. Taken together, $k \ = \ 2$ achieves the best balance between diversity preservation and quality, and we adopt this value in all Creative-MAD experiments. Notably, even at k = 3, Creative-MAD maintains higher diversity than Homo MAD, demonstrating that the method is robust to the choice of k within a reasonable range.

## 7.6 Peer Selection Strategy in EPS

Table 3: Effect of peer selection strategy on Qwen3- 8B. Absolute quality and semantic diversity scores are reported as mean ± std.
<table><tr><td rowspan="2">Method</td><td colspan="2">LiveIdea</td><td colspan="2">MacGyver</td></tr><tr><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td></tr><tr><td>Closest-k</td><td>72.8±0.2</td><td> $2 . 7 5 { \pm } 0 . 0 4$ </td><td>76.8±0.2</td><td>1.52±0.05</td></tr><tr><td>Random-k</td><td> $7 2 . 1 { \pm } 0 . 3 $ </td><td> $2 . 8 8 { \pm } 0 . 0 6$ </td><td> $7 7 . 0 { \pm } 0 . 4 $ </td><td> $1 . 5 9 { \pm } 0 . 0 5$ </td></tr><tr><td>EPS</td><td> $7 2 . 6 { \pm } 0 . 1 $ </td><td> $3 . 0 4 \pm 0 . 0 6$ </td><td> $7 6 . 4 \pm 0 . 3$ </td><td> $1 . 6 7 { \pm } 0 . 0 4 $ </td></tr></table>

To isolate the contribution of semantic distance in EPS, we compare three peer selection strategies under the same Creative-MAD framework with $k \ = \ 2$ on LiveIdeaBench and MacGyver: EPS selects the k most semantically distant peers,

Random-k selects k peers uniformly at random, and Closest-k selects the k most semantically similar peers. Table 3 reveals a consistent hierarchy in semantic diversity across both benchmarks: Closest-k yields the lowest diversity, Random-k moderately improves it, and EPS achieves the highest, validating that maximizing semantic distance drives substantial diversity gains beyond mere pool reduction. Meanwhile, quality scores remain statistically invariant across all strategies, confirming that driving cross-perspective divergence does not compromise response quality.

## 7.7 Additional Results

Due to space constraints, we report the following results in the Appendix. Appendix F validates generalizability across broader model scales and families, while Appendix G confirms evaluation reliability through inter-judge agreement, detailed human annotation studies, and robustness to embedding model choice. Appendix H examines sensitivity to the number of debate rounds and the number of agents, and Appendix I extends our analysis to model-heterogeneous settings. Appendix J evaluates robustness under an alternative debate protocol, and Appendix K quantifies Creative-MAD’s computational overhead relative to standard MAD. Finally, Appendix L analyzes qualitative failure cases where Creative-MAD’s diversity gains come at the cost of relevance or constraint satisfaction, and Appendix M presents representative successful outputs across all four benchmarks.

## 8 Conclusion

In this work, we formalize the quality-diversity trade-off when applying MAD to creative tasks, proving that preserving intra-session diversity is a necessary condition for sustaining inter-session diversity. To address this, we propose Creative-MAD, which combines CLA to counter identity drift and EPS to mitigate majority pull. Extensive experiments confirm that Creative-MAD substantially improves cross-run diversity while maintaining output quality comparable to standard MAD.

## Limitations

Our work has several limitations that point to promising directions for future research. First, while our judge reliability analysis demonstrates agreement with human raters and commercial models in Appendix G, LLM-based evaluation remains an imperfect proxy for human aesthetic judgment in creative tasks, as automated judges may favor conventionally well-formed outputs over genuinely creative ones, potentially undervaluing subtle dimensions of creative quality. Second, since creative tasks admit no ground-truth answer, we employ a judge-based consensus mechanism to select the final output. While our theoretical analysis motivates CLA and EPS as inference-time interventions targeting identity drift and majority pull, the same analysis points to other directions worth exploring: training-time interventions, such as fine-tuning agents with a reward that encourages sustained divergence across debate rounds, could reduce reliance on a fixed cognitive lens; alternatively, redesigning the consensus stage itself to retain a diverse subset of candidates, rather than collapsing each session into a single selected output, could preserve more of the diversity that debate otherwise loses at consensus. Whether such alternative consensus strategies interact differently with diversity-preserving debate mechanisms remains an open question for future work. Finally, our analysis focuses on the fixed-prompt setting where the query x is held constant across sessions to isolate diversity suppression arising from debate dynamics themselves; run-level interventions such as per-run persona resampling (Hu et al., 2025) represent a complementary direction, and combining such approaches with Creative-MAD remains a promising avenue for future work.

## Ethical Considerations

The datasets used are publicly available benchmarks and do not contain personally identifiable information or sensitive content. The human annotation study (Appendix G.2) involved internal researchers who participated voluntarily, and no sensitive data was collected from participants. The proposed Creative-MAD framework generates openended creative text and does not target any specific demographic group or enable harmful applications. Automated evaluation via LLMaaJ may reflect biases of the underlying judge model; we mitigate this concern through inter-judge agreement analysis and human evaluation (Appendix G).

## Acknowledgements

AI assistants were used for language editing (e.g., grammar and phrasing checks) and for coding support during the preparation of this work.

## References

Anthropic. 2026. Claude Sonnet 4.6. https:// www.anthropic.com/news/claude-sonnet-4-6. Model release date: February 18, 2026.

Jonas Becker, Lars Benedikt Kaesberg, Niklas Bauer, Jan Philip Wahle, Terry Ruas, and Bela Gipp. 2025. Mallm: Multi-agent large language models framework. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 418–439.

Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2024. Chateval: Towards better llm-based evaluators through multi-agent debate. In International conference on learning representations, volume 2024, pages 9079–9093.

Justin Chen, Swarnadeep Saha, and Mohit Bansal. 2024. Reconcile: Round-table conference improves reasoning via consensus among diverse llms. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 7066–7085.

Hyeong Kyu Choi, Jerry Zhu, and Sharon Li. 2026. Debate or vote: Which yields better decisions in multiagent large language models? In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

John Joon Young Chung, Vishakh Padmakumar, Melissa Roemmele, Yuqian Sun, and Max Kreminski. 2025. Modifying large language model post-training for diverse creative writing. In Second Conference on Language Modeling.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first international conference on machine learning.

Dan Friedman and Adji Bousso Dieng. 2023. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Zhe Hu, Hou Pong Chan, Jing Li, and Yu Yin. 2025. Debate-to-write: A persona-driven multi-agent framework for diverse argument generation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 4689–4703.

Thomas Huber and Christina Niklaus. 2025. CLEAR: A comprehensive linguistic evaluation of argument rewriting by large language models. In Findings of the Association for Computational Linguistics:

EMNLP 2025, pages 19548–19568, Suzhou, China. Association for Computational Linguistics.

Mete Ismayilzada, Antonio Laverghetta Jr., Simone A. Luchini, Reet Patel, Antoine Bosselut, Lonneke Van Der Plas, and Roger E. Beaty. 2025. Creative preference optimization. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 9580–9609, Suzhou, China. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. 2025. From crowdsourced data to highquality benchmarks: Arena-hard and benchbuilder pipeline. In Forty-second International Conference on Machine Learning.

Yunxuan Li, Yibing Du, Jiageng Zhang, Le Hou, Peter Grabowski, Yeqing Li, and Eugene Ie. 2024. Improving multi-agent debate with sparse communication topology. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7281– 7294.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 17889–17904.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, and 49 others. 2025. Olmo 3. Preprint, arXiv:2512.13961.

OpenAI. 2026. GPT-5.4. https://developers. openai.com/api/docs/models/gpt-5.4. Model release date: March 06, 2026.

Samuel J Paech. 2023. Eq-bench: An emotional intelligence benchmark for large language models. arXiv preprint arXiv:2312.06281.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Kai Ruan, Xuan Wang, Jixiang Hong, Peng Wang, Yang Liu, and Hao Sun. 2026. Evaluating LLMs’ divergent thinking capabilities for scientific idea generation with minimal context. Nature Communications.

Zhiwen Ruan, Yixia Li, Yefeng Liu, Yun Chen, Weihua Luo, Peng Li, Yang Liu, and Guanhua Chen. 2025. G2: Guided generation for enhanced output diversity in llms. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 14127–14145.

Henrique\* Schechter Vera, Sahil\* Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, and 69 others. 2025. Embeddinggemma: Powerful and lightweight text representations.

Alexander Shypula, Shuo Li, Botong Zhang, Vishakh Padmakumar, Kayo Yin, and Osbert Bastani. 2025. Evaluating the diversity and quality of LLM generated content. In Second Conference on Language Modeling.

Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas, Andreas Koukounas, Nan Wang, and Han Xiao. 2024. jina-embeddings-v3: Multilingual embeddings with task lora. Preprint, arXiv:2409.10173.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, and 1 others. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Yufei Tian, Abhilasha Ravichander, Lianhui Qin, Ronan Le Bras, Raja Marjieh, Nanyun Peng, Yejin Choi, Thomas L Griffiths, and Faeze Brahman. 2024. Macgyver: Are large language models creative problem solvers? In Proceedings ofthe 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5303–5324.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems, 33:5776–5788.

Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. 2024. Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 257–279.

Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Wang Zijia, Ji Zhang, Mengyue Wu, Qin Jin, and Fei Huang. 2026. Writingbench: A comprehensive benchmark for generative writing. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. Preprint, arXiv:2309.07597.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025a. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Yiming Zhang, Harshita Diddee, Susan Holm, Hanchen Liu, Xinyue Liu, Vinay Samuel, Barry Wang, and Daphne Ippolito. 2025b. Noveltybench: Evaluating language models for humanlike diversity. In Second Conference on Language Modeling.

Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation models. In The 41st international ACM SIGIR conference on research & development in information retrieval, pages 1097–1100.

## A Experimental Details

## A.1 Dataset Details

In this section, we provide a comprehensive description of the datasets used in our creative generation experiments and the specific number of samples utilized for each.

LiveIdeaBench (Ruan et al., 2026) is a specialized benchmark designed to evaluate the divergent thinking capabilities of LLMs in the context of scientific innovation. Instead of providing rich contextual information, this dataset utilizes minimal single-keyword prompts across 22 scientific domains to assess the model’s ability to generate novel and feasible scientific hypotheses from its internal knowledge representations. We randomly subsample 300 keywords from the original 1,180 high-impact scientific keywords for our study.

Argumentative Essays (adapted from CLEAR (Huber and Niklaus, 2025)) utilizes topics derived from the Argument Annotated Essays 2.0 corpus. While the original CLEAR framework is designed for the task of Argument Improvement (rewriting student essays), we specifically extract the controversial essay topics to evaluate the models’ capacity for original argumentative generation. This task requires the model to construct structured, persuasive, and coherent arguments from scratch based on a given prompt. We utilize a random subset of 300 essay topics for this evaluation.

MacGyver (Tian et al., 2024) challenges models with creative physical problem-solving in realworld scenarios. The dataset is specifically curated to trigger out-of-the-box thinking by presenting problems that necessitate the unconventional usage of everyday objects, thereby pushing against the cognitive bias of “functional fixedness.” Models must devise physically feasible and efficient plans to achieve complex goals under strict constraints. For our experiments, we specifically filter for solvable samples—defined as problems for which a safe and reasonable solution can be achieved using the provided tools—and employ a random subset of 300 problem scenarios from this portion of the dataset.

Arena Hard v2.0 (Li et al., 2025) (creative writing subset) is a benchmark for open-ended creative text generation, featuring diverse prompts that require models to produce poems, lyrics, dialogues, and other imaginative responses under varied stylistic constraints. Unlike benchmarks centered on scientific ideation or physical problem-solving, this subset focuses on general creative writing and is well-suited for evaluating creativity, prompt relevance, and language quality in the absence of a single reference answer. For our experiments, we use the full creative writing subset of Arena Hard v2.0, consisting of 250 prompts.

## A.2 Hyperparameters & Compute Resources

Hyperparameters. All MAD-based methods use N = 5 agents, R = 2 debate rounds, temperature = 1.0, and maximum output length of 2048 tokens. Creative-MAD uses k = 2 for EPS. The judge runs at temperature = 0.2.

Compute Resources. All experiments are conducted on NVIDIA H100 80GB GPUs using vLLM engine (Kwon et al., 2023) with bf16 precision.

## B Proof of Proposition 1

We consider the standard MAD setting in which the G sessions are run independently with no shared state or communication, the input prompt is fixed across runs, and temperature sampling is the only source of stochasticity in the single-model setting. The proof operates solely on the output similarity kernel K, making no assumptions about whether outputs are generated by the same or different model families, and therefore extends to model-heterogeneous settings. Alternative interventions such as prompt perturbation or temperature scheduling are outside the scope of this setting.

When intra-session diversity decreases through debate, agents within each session g produce increasingly similar outputs $\{ y _ { i } ^ { ( R ) } \} _ { i = 1 } ^ { N }$ , reducing the effective diversity of the pool from which the consensus mechanism selects $\hat { y } _ { g }$ . As a result, the final outputs $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G }$ become semantically closer to one another across sessions, causing the off-diagonal entries of K to increase and pushing K<sup>¯</sup> toward a uniform matrix:

$$
{ \bar { K } } \to { \frac { 1 } { G } } \mathbf { 1 } \mathbf { 1 } ^ { \top } \implies H ( K ) \to 0 \implies { \mathcal { D } } _ { \mathrm { i n t e r } } \to 1\tag{5}
$$

More generally, this relationship is monotonic: as intra-session diversity decreases, K approaches a rank-1 structure with a single dominant eigenvalue $\lambda _ { 1 } \approx 1$ and remaining eigenvalues $\lambda _ { i } \approx 0 .$ , minimizing $\begin{array} { r } { H ( K ) = - \sum _ { i } \lambda _ { i } } \end{array}$ log $\lambda _ { i }$ and consequently $\mathcal { D } _ { \mathrm { i n t e r } } .$

Practical Implications. Proposition 1 has two important practical implications.

First, the consensus mechanism cannot compensate for low intra-session diversity. When agents within session $g$ produce highly similar outputs $\{ y _ { i } ^ { ( R ) } \} _ { i = 1 } ^ { N }$ , the consensus mechanism, regardless of its selection criterion, must draw its final output $\hat { y } _ { g }$ from a concentrated pool. Since this holds for every session $^ { g , }$ the final outputs $\{ \hat { y } _ { g } \} _ { g = 1 } ^ { G }$ cluster in similar regions across sessions, yielding low inter-session diversity regardless of the consensus strategy employed.

Second, intra-session diversity is the primary controllable lever for inter-session diversity. The exploration of session g is entirely determined by its own internal debate dynamics, making diversity preservation during the debate process itself the only effective intervention for improving intersession diversity, such that each session explores a broader region of Y and the selected output $\hat { y } _ { g }$ can vary meaningfully across sessions.

## C Martingale Inapplicability in Creative Settings

Remark 1 (Martingale Inapplicability). The martingale result in Choi et al. (2026) relies on two conditions: (1) agents generate responses from a finite categorical space $y \in \{ 1 , \ldots , K \}$ , and (2) belief updates follow Bayesian conjugacy over discrete counts under a Dirichlet-Compound-Multinomial (DCM) model. Creative tasks violate both conditions. First, the response space is open-ended $( y \in \tau ^ { * } )$ , rendering the marginal probability $P ( y = k \mid \alpha )$ undefined. Second, quality is assessed via a continuous signal $q _ { i , t } = \mathsf { J u d g e } ( y _ { i , t } ) \in$ R, and the agent update rule becomes:

$$
y _ { i } ^ { ( t ) } = \mathcal { M } \left( x , \mathcal { P } _ { i } ^ { ( t - 1 ) } \right) , \quad q _ { i , t } = \mathsf { J u d g e } ( y _ { i } ^ { ( t ) } )\tag{6}
$$

which is no longer a Bayesian count update. Consequently, the martingale condition $\mathbb { E } [ q _ { i , t } \mid q _ { i , t - 1 } ] =$ $q _ { i , t - 1 }$ need not hold, allowing debate to provide genuine quality improvement in creative settings.

## D Pairwise Win Rate Heatmaps

Figure 7 presents the full pairwise win rate heatmaps across all method pairs on Qwen3-8B, where each cell (i, j) denotes the win rate of method i against method $j$

## E Prompt Templates

## E.1 MAD Templates

The following template is used at each debate round $t \geq 1$ to prompt each agent to refine its response based on peer outputs.

Debate Prompt Template   
Task: {query}   
These are the recent ideas from   
other agents:   
{peer\_answers}   
This was your most recent idea:   
{own\_answer}   
Within <critique></critique> tags,   
briefly evaluate the other ideas, filter  
ing out clichés and synthesizing their   
best elements into a stronger concept.   
Then within <answer></answer> tags,   
write an improved and highly original   
response built directly upon this synthesis.   
Your answer must NOT reference other   
agents, the debate process, or include any   
meta-commentary.   
Respond in the following format:   
<critique>   
</critique>   
<answer>   
</answer>

## E.2 Consensus Selection Templates

The following template is used at the consensus stage to select the best response from all agents final outputs.

Consensus Selection Prompt Template   
Task: {query}   
Several experts have debated and   
submitted their final refined   
answers below:   
{all\_answers}   
As an impartial and expert judge, evaluate   
these answers based on their creativity, log  
ical coherence, and alignment with the task.   
Select the single best answer.   
IMPORTANT: You must output ONLY the   
Agent Number of the winning answer en  
closed in <winner> tags. For example, if

![](images/ff1d79ebda19991bc856f162a15cdfdb698e8210ae1d2196794e664048a73caa.jpg)  
Figure 7: Pairwise win rate heatmaps across four benchmarks on Qwen3-8B. Each cell (i, j) denotes the win rate of the target method (row) against the opponent method (column).

Agent 2 is the best, output exactly:   
<winner>2</winner>   
Do NOT output the text of the answer itself, and do NOT provide any reasoning or extra text.

## E.3 Cognitive Lens Templates

Table 4 presents the five cognitive lens templates assigned to agents in Creative-MAD. Each lens is assigned deterministically to one agent and persists across all debate rounds.

## E.4 Persona Prompt Templates

Table 5 presents the five persona prompt templates used in Hetero MAD. Each persona is assigned deterministically to one agent and persists across all debate rounds.

## E.5 LLMaaJ Evaluation Rubrics Templates

For each benchmark, we employ a dataset-specific rubric to evaluate instance-level quality. All rubrics use a strict 1–10 scale per dimension, and the final quality score is computed as the average across

dimensions.

## E.5.1 LiveIdeaBench

The LiveIdeaBench rubric evaluates scientific ideation quality across three dimensions: originality, feasibility, and clarity, adapted from Ruan et al. (2026) with a 1–10 scoring scale.

## LiveIdeaBench Evaluation Rubric

You are a cynical and world-class expert evaluator of scientific innovation. You are bored by standard AI-generated summaries and are looking for genuinely groundbreaking frontiers.

The user’s query provides a KEYWORD. The proposed idea MUST address a meaningful problem related to that keyword with high “Practical Ingenuity”.

You will assess the response on three key dimensions using a STRICT 1–10 scale:

1. Originality: Does it offer a novel, nonobvious contribution? Scores of 7+ are reserved for ideas that go beyond standard textbook knowledge or common AI patterns.

Table 4: Cognitive lens templates used in Creative-MAD. Each agent is assigned one lens that persists throughout all debate rounds.
<table><tr><td>Lens</td><td>Template</td></tr><tr><td>Analytical</td><td>You process all information through an analytical and logical lens. When approaching any task, you identify underlying structures, patterns, and evidence. You ask: what logic connects the elements, what structure can be revealed, what reasoning makes this idea compelling and coherent?</td></tr><tr><td>Emotional</td><td>You process all information through an emotional and human-centered lens. When approaching any task, you focus on human impact, feelings, values, and lived experience. You ask: how does this affect people deeply, what emotions does it evoke, what human truth or vulnerability does it reveal?</td></tr><tr><td>Critical</td><td>You process all information through a critical and adversarial lens. When approaching any task, you interrogate assumptions, surface hidden contradictions, and find the perspective that challenges the obvious. You ask: what is being overlooked, what tension or paradox exists here, what would a skeptic or contrarian notice?</td></tr><tr><td>Analogical</td><td>You process all information through an analogical and cross-domain lens. When approaching any task, you draw unexpected connections to distant domains, metaphors, and parallel systems You ask: what surprising analogy illuminates this, what can be borrowed from an unrelated field, what deeper pattern connects seemingly unrelated things?</td></tr><tr><td>Practical</td><td>You process all information through a practical and implementation-focused lens. When ap- proaching any task, you evaluate real-world feasibility, concrete mechanisms, and actionable outcomes. You ask: what actually works in practice, how would this be realized step by step, what real-world constraints and opportunities shape the best solution?</td></tr></table>

2. Feasibility: Scientific validity and technical implementation. A brilliant idea that violates physics or logic should be penalized here.

3. Clarity: Structural elegance and the ability to communicate complex innovation simply.

## Scoring Guidelines (Be Strict):

• 1–3 (Poor): Irrelevant, logically flawed, or purely superficial.

• 4–5 (Baseline): Technically correct and safe, but “standard” and lacking any significant creative spark. This is the default for most AI-generated responses.

• 6–7 (Above Average): Shows a specific, clever angle or an unconventional but viable application.

• 8–10 (Exceptional): Truly ingenious, transformative, and non-obvious thinking that demonstrates professionallevel scientific insight.

If the idea is completely irrelevant to the   
provided keyword, assign a score of 1–2   
across all dimensions.   
Output format:   
{   
"analysis": "<brief analysis,   
,→ specifically explaining   
why the idea is   
or isn't,→   
innovative>",,→   
"scores": {   
"originality": <int>,   
"feasibility": <int>,   
"clarity": <int>   
}   
}

## E.5.2 Argumentative Essays

The Argument Annotated Essays corpus is originally designed for argument rewriting evaluation (Huber and Niklaus, 2025). Since we repurpose it for argumentative generation from scratch, we design a rubric tailored to original argument quality rather than rewriting criteria, evaluating three dimensions: persuasiveness, novelty, and logical coherence.

Table 5: Persona prompt templates used in Hetero MAD. Each agent is assigned one persona that persists throughout all debate rounds.
<table><tr><td>Persona</td><td>Template</td></tr><tr><td>Economist</td><td>You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.</td></tr><tr><td>Psychologist</td><td>You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.</td></tr><tr><td>Lawyer</td><td>You are a lawyer. You are good at law, politics, and history.</td></tr><tr><td>Doctor</td><td>You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient&#x27;s age, lifestyle and medical history when providing your recommendations.</td></tr><tr><td>Historian</td><td>You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.</td></tr></table>

## Argumentative Essays Evaluation Rubric

You are an expert debate judge and rhetorician evaluating argument generation. You will assess the response on three key dimensions:

1. Persuasiveness: How compelling, well-supported, and convincing the fundamental argument is.

2. Novelty: Originality of the angle, avoiding typical clichés or surfacelevel reasoning.

3. Logical Coherence: Clarity, structure, and the soundness of the logical deduction.

Provide your brief analysis (less than 100 words) of the argument. Then, inside a “scores” object, provide a score from 1 to 10 for each dimension where 1–3 = poorly reasoned, 4–6 = standard argument, 7–10 = highly compelling and insightful.

Output format:

```json
{
"analysis": "<your brief
,→ analysis here>",
"scores": {
"persuasiveness": <score>,
"novelty": <score>,
"logical_coherence":
,→ <score>
}
}
```

## E.5.3 MacGyver

MacGyver evaluates creative physical problemsolving under strict resource constraints (Tian et al., 2024). Following the evaluation criteria described in the original benchmark, we design a rubric assessing three dimensions: out-of-the-box thinking, practical feasibility, and constraint utilization.

## MacGyver Evaluation Rubric

You are an expert evaluator assessing creative problem-solving in extreme constraint scenarios. You will assess the proposed solution on three key dimensions:

1. Out-of-the-box Thinking: How unconventional and clever the solution is, avoiding standard generic answers.

2. Practical Feasibility: Whether the physical actions described would actually work and obey the laws of physics.

3. Constraint Utilization: How effectively the limited available items are strictly used without assuming extra resources.

Provide your brief analysis (less than 100 words) of the solution. Then, inside a “scores” object, provide a score from 1 to 10 for each dimension where 1–3 = hallucinatory/boring, 4–6 = works but basic, 7–10 = highly ingenious and viable.

```json
{"analysis": "<your brief analysis here>",
"scores": {"out_of_the_box_thinking":
,→ <score>,
"practical_feasibility":
,→ <score>,
"constraint_utilization":
,→ <score>}}
```

## E.5.4 Arena Hard v2.0

Arena Hard v2.0 (creative writing subset) covers diverse creative writing tasks including poetry, dialogue, and genre writing. Following the evaluation criteria of the benchmark, we design a rubric assessing three dimensions: creativity, relevance, and language quality.

Arena Hard v2.0 Evaluation Rubric   
You are an expert judge evaluating creative writing   
responses. You will assess the response on three key   
dimensions using a STRICT 1–10 scale:   
1. Creativity: The response demonstrates orig  
inality, unique ideas, and avoids clichés or   
generic patterns.   
2. Relevance: The response directly addresses   
and fulfills the requirements of the prompt.   
3. Language Quality: The writing is fluent, co  
herent, and stylistically appropriate for the   
genre (e.g., poetry, rap, dialogue, etc.).   
Scoring Guidelines (Be Strict):   
• 1–3 (Poor): Off-topic, generic, or poorly writ  
ten. Lacks creativity or fails to address the   
prompt.   
• 4–5 (Baseline): Adequate and relevant, but not   
particularly creative or polished. May have   
minor language issues.   
• 6–7 (Above Average): Shows clear creativity,   
good relevance, and strong language use.   
• 8–10 (Exceptional): Highly original, perfectly   
fits the prompt, and is masterfully written.   
If the response is completely irrelevant or empty, as  
sign a score of 1–2 across all dimensions.   
Output format:   
{"analysis": "<brief analysis of   
,→ strengths and weaknesses>",   
"scores": {"creativity": <int>,   
"relevance": <int>,   
"language\_quality": <int>}}

## E.6 Pairwise Win Rate Evaluation Templates

For pairwise win rate evaluation, we employ the same Qwen3.5-397B-A17B judge with datasetspecific criteria as defined in the rubrics above (Appendix E.5). For each query, we sample one response per method and perform pairwise comparisons across all number of methods. To mitigate positional bias, each pair is evaluated twice with response positions swapped. A winner is declared only when both runs agree; disagreements are counted as ties. The win rate of each method is computed as the fraction of matches won across all pairwise comparisons and all queries.

Pairwise Win Rate Judge Prompt   
You are an expert evaluator assessing AI-generated   
responses based on “Creative Quality”.   
Determine which response is better based on how   
well it satisfies the following criteria specific to the   
task domain:   
{criteria}   
You will be given a task and two AI responses (Re  
sponse A and Response B). Determine which re  
sponse demonstrates superior quality according to   
the criteria above.   
You MUST output a valid JSON object with exactly   
two keys:   
1. reasoning: A brief comparison (2–3 sentences)   
explaining the key differences. Before choos  
ing the winner, explicitly list at least one spe  
cific point where the winner outperformed the   
other.   
2. winner: One of “A”, “B”, or “tie”.   
A “tie” is STRONGLY DISCOURAGED and is only   
acceptable if and only if both responses are perfectly   
identical in value. Even a minor improvement or a   
better execution means that model must win.   
Output format:   
{"reasoning": "<brief comparison>",   
"winner": "A" | "B" | "tie"}

## F Model Generalizability

To verify that our findings generalize beyond the primary models evaluated in the main experiments, we conduct additional evaluations on two models spanning a wider range of parameter scales: Llama-3.2-3B-Instruct (Grattafiori et al., 2024) (3B) and Qwen3-32B (Yang et al., 2025) (32B). Together with the primary models (Qwen3-8B and Gemma-3-12B-Instruct), this covers four scales across three model families (Qwen, Llama, Gemma).

Table 6 reports quality and semantic diversity results across all four benchmarks. The qualitydiversity trade-off observed in the main experiments holds consistently across both model scales: Homo MAD and Hetero MAD achieve higher quality than almost non-debate baselines while exhibiting lower diversity, and Creative-MAD maintains quality comparable to standard MAD variants while substantially improving semantic diversity across all benchmarks. These results confirm that the trade-off and our proposed solution are not artifacts of a specific model scale or family, but reflect a structural property of multi-agent debate dynamics in creative generation.

Table 6: Generalizability results across four creative benchmarks on additional model scales. We report instancelevel quality score (mean ± std) and set-level semantic diversity score (mean ± std). Higher is better. Bold indicates the best result(s); multiple entries may be bolded if not statistically significantly different (paired t-test, $p < 0 . 0 5 )$
<table><tr><td rowspan="2">Method</td><td colspan="2">LiveIdea</td><td colspan="2">AAE</td><td colspan="2">MacGyver</td><td colspan="2">AH v2.0</td></tr><tr><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td></tr><tr><td colspan="9">Llama-3.2-3B-Instruct</td></tr><tr><td>Direct</td><td>61.4±0.3</td><td>3.19±0.04</td><td>70.0±0.4</td><td>1.68±0.04</td><td>48.6±0.4</td><td>1.79±0.06</td><td>47.2±0.4</td><td>2.61±0.05</td></tr><tr><td>Self-Refine</td><td>64.5±0.2</td><td>3.13±0.05</td><td>74.9±0.6</td><td>1.85±0.04</td><td>49.4±0.7</td><td>1.78±0.04</td><td>54.1±0.8</td><td>2.62±0.06</td></tr><tr><td>Voting</td><td>61.4±0.5</td><td>3.13±0.04</td><td>70.7±0.2</td><td>1.69±0.05</td><td>50.4±1.7</td><td>1.80±0.05</td><td>48.6±0.8</td><td>2.58±0.04</td></tr><tr><td>Homo MAD</td><td>64.1±0.3</td><td>2.72±0.06</td><td>74.4±0.3</td><td> $1 . 5 1 { \pm } 0 . 0 6$ </td><td>52.9±0.7</td><td>1.72±0.04</td><td>70.7±3.9</td><td>2.44±0.06</td></tr><tr><td>Hetero MAD</td><td>64.7±0.3</td><td>3.22±0.05</td><td>74.1±0.3</td><td> $1 . 7 2 { \pm } 0 . 0 6$ </td><td>53.2±0.6</td><td>1.80±0.05</td><td>69.1±2.1</td><td>2.63±0.04</td></tr><tr><td>Creative-MAD</td><td>64.6±0.2</td><td>3.71±0.06</td><td>74.1±0.2</td><td>2.23±0.05</td><td>52.8±0.6</td><td>2.20±0.07</td><td>70.0±2.5</td><td>3.08±0.04</td></tr><tr><td colspan="9">Qwen3-32B</td></tr><tr><td>Direct</td><td>70.2±0.3</td><td>2.73±0.05</td><td>73.4±0.1</td><td>1.57±0.05</td><td>76.3±0.4</td><td>1.62±0.04</td><td>77.9±0.1</td><td>2.13±0.06</td></tr><tr><td>Self-Refine</td><td>75.3±0.3</td><td>2.78±0.06</td><td>76.8±0.1</td><td>1.67±0.04</td><td>79.1±0.3</td><td>1.58±0.06</td><td>79.3±0.2</td><td>2.15±0.04</td></tr><tr><td>Voting</td><td>71.5±0.3</td><td>2.79±0.04</td><td>74.3±0.1</td><td>1.57±0.06</td><td>78.5±0.2</td><td>1.62±0.05</td><td>82.2±0.2</td><td>2.19±0.04</td></tr><tr><td>Homo MAD</td><td>77.2±0.3</td><td>2.69±0.05</td><td>77.2±0.2</td><td> $1 . 5 6 { \pm } 0 . 0 5$ </td><td>81.2±0.3</td><td>1.59±0.06</td><td>83.9±0.3</td><td>1.99±0.05</td></tr><tr><td>Hetero MAD</td><td>76.8±0.3</td><td>2.81±0.04</td><td>78.0±0.2</td><td> $1 . 6 2 { \pm } 0 . 0 4 $ </td><td>81.2±0.3</td><td>1.64±0.05</td><td>83.0±0.3</td><td>2.19±0.06</td></tr><tr><td>Creative-MAD</td><td>77.0±0.3</td><td>3.26±0.06</td><td>78.0±0.2</td><td>2.10±0.07</td><td>80.0±0.3</td><td>1.91±0.04</td><td>83.5±0.3</td><td>2.62±0.05</td></tr></table>

## G Evaluation Reliability

We validate the reliability of our evaluation along four complementary dimensions: agreement of our LLMaaJ with commercial judge models (Section G.1), agreement with human annotators (Section G.2), robustness of our semantic diversity metric across different embedding models (Section G.3), and robustness of quality rankings to length bias (Section G.4).

## G.1 LLMaaJ Inter-Model Agreement

While commercial models such as GPT-5.4 (OpenAI, 2026) and Claude Sonnet 4.6 (Anthropic, 2026) are widely adopted as judges in recent work, our evaluation involves a large number of inference calls across methods, benchmarks, and independent runs (15 runs per method per query for statistical reliability), making commercial API-based evaluation prohibitively expensive at scale. We therefore employ Qwen3.5-397B-A17B as our primary judge and validate its reliability against these commercial alternatives on a representative subset.

Setup. We randomly sample 50 queries per benchmark and evaluate Qwen3-8B outputs from all methods using all three judges. For rubric-based score validation (Table 7), we compute the mean rubric score across G = 5 randomly sampled outputs per method per query. For pairwise win rate validation (Table 8), we use one randomly selected output per method per query. For each metric, we compute Spearman $\rho$ between Qwen3.5-397B-

A17B and each commercial judge across methods per benchmark, measuring the consistency of method rankings induced by each judge.

Results. Tables 7 and 8 show generally strong agreement between Qwen3.5-397B-A17B and both commercial judges across most benchmarks. Spearman $\rho$ for rubric scores averages 0.87 and 0.91 against GPT-5.4 and Claude Sonnet 4.6 respectively, and 0.93 and 0.89 for pairwise win rates, suggesting that method rankings produced by our judge are largely consistent with those of commercial alternatives. These results support the use of Qwen3.5-397B-A17B as a reliable and costeffective alternative for evaluating creative generation quality at scale.

## G.2 Human Agreement Study

To validate the alignment between our automated judge and human judgment, we conduct two human evaluation studies across all four benchmarks: (1) a judge-reliability study comparing single-agent baselines against Homo MAD, and (2) a pilot study directly comparing Homo MAD against Creative-MAD on both quality and diversity. Summary results are reported in Section 6.3.

Setup (Judge Reliability). We randomly sample 30 queries per benchmark (120 total, fixed random seed). For each query, we randomly assign one of two comparison pairs contrasting single-agent baselines against Homo MAD: Direct vs. Homo MAD (larger quality gap) or Self-Refine vs. Homo MAD (smaller quality gap), providing coverage across different levels of quality difference and directly validating the core claim that MAD improves output quality over single-agent generation. For each method, one response is randomly selected from the available generations. Response ordering (A vs. B) is randomized per query to mitigate positional bias, and method names are not disclosed to annotators.

Table 7: Inter-judge agreement on rubric-based absolute scores (mean ± std over $G = 5$ outputs per method per query) on a subset of 50 queries per benchmark (Qwen3-8B outputs). Spearman ρ is computed between Qwen3.5-397B-A17B (Qwen) and each commercial judge — GPT-5.4 (GPT) and Claude Sonnet 4.6 (Sonnet) — across methods per benchmark. Higher ρ indicates stronger agreement.
<table><tr><td></td><td colspan="3">LiveIdea</td><td colspan="3">AAE</td><td colspan="3">MacGyver</td><td colspan="3">AH v2.0</td></tr><tr><td>Method</td><td>Qwen</td><td>GPT</td><td>Sonnet</td><td>Qwen</td><td>GPT</td><td>Sonnet</td><td>Qwen</td><td>GPT</td><td>Sonnet</td><td>Qwen</td><td>GPT</td><td>Sonnet</td></tr><tr><td>Direct</td><td> $6 6 . 5 \pm 0 . 3$ </td><td> $6 3 . 0 \pm 1 . 0$ </td><td> $5 9 . 8 \pm 0 . 0$ </td><td> $7 3 . 5 \pm 0 . 7$ </td><td> $5 9 . 6 \pm 0 . 1$ </td><td>62.6 ± 0.4</td><td> $6 4 . 6 \pm 1 . 6$ </td><td> $3 0 . 9 \pm 0 . 5$ </td><td> $2 6 . 9 \pm 0 . 8$ </td><td> $7 5 . 3 \pm 1 . 2$ </td><td> $5 8 . 7 \pm 0 . 1$ </td><td> $5 4 . 7 \pm 0 . 8$ </td></tr><tr><td>Self-Refine</td><td> $6 9 . 7 \pm 0 . 5$ </td><td> $6 4 . 2 \pm 0 . 7$ </td><td> $6 0 . 1 \pm 1 . 0$ </td><td> $7 4 . 4 \pm 0 . 4$ </td><td> $6 0 . 3 \pm 0 . 2$ </td><td>63.7 ± 0.6</td><td> $6 9 . 1 \pm 1 . 2$ </td><td> $3 3 . 9 \pm 0 . 3$ </td><td> $3 0 . 3 \pm 1 . 7$ </td><td>78.0 ± 0.9</td><td> $6 0 . 9 \pm 0 . 1$ </td><td> $5 5 . 4 \pm 1 . 2$ </td></tr><tr><td>Voting</td><td> $6 7 . 5 \pm 0 . 7$ </td><td> $6 3 . 8 \pm 0 . 5$ </td><td> $5 9 . 1 \pm 1 . 4$ </td><td>74.4 ± 0.3</td><td> $6 0 . 0 \pm 0 . 2$ </td><td>63.2 ± 0.6</td><td> $6 7 . 0 \pm 1 . 3$ </td><td> $3 2 . 3 \pm 1 . 5$ </td><td> $2 6 . 7 \pm 2 . 2$ </td><td> $7 8 . 8 \pm 0 . 4$ </td><td> $6 0 . 8 \pm 1 . 6$ </td><td> $5 3 . 6 \pm 0 . 7$ </td></tr><tr><td>Homo MAD</td><td> $7 3 . 7 \pm 1 . 0$ </td><td> $6 5 . 0 \pm 0 . 2$ </td><td> $6 1 . 2 \pm { 0 . 8 }$ </td><td> $8 0 . 6 \pm 0 . 3$ </td><td> $6 2 . 4 \pm 0 . 2$ </td><td>70.3 ± 1.0</td><td> $7 7 . 1 \pm 0 . 6$ </td><td> $3 6 . 2 \pm 0 . 5$ </td><td> $3 5 . 2 \pm 0 . 5$ </td><td> $8 1 . 4 \pm 0 . 3$ </td><td> $6 2 . 1 \pm 0 . 4$ </td><td> $5 7 . 6 \pm 2 . 0$ </td></tr><tr><td>Hetero MAD</td><td> $7 2 . 7 \pm 0 . 4$ </td><td> $6 4 . 1 \pm 1 . 4$ </td><td> $6 1 . 8 \pm 1 . 4$ </td><td> $8 2 . 3 \pm 0 . 2$ </td><td> $6 3 . 7 \pm 0 . 2$ </td><td> $7 2 . 5 \pm 0 . 3$ </td><td> $7 8 . 4 \pm 0 . 2$ </td><td> $3 6 . 6 \pm 0 . 7$ </td><td> $3 6 . 4 \pm 0 . 4$ </td><td> $8 2 . 7 \pm 0 . 2$ </td><td> $6 2 . 4 \pm 0 . 5$ </td><td> $5 8 . 4 \pm 0 . 1$ </td></tr><tr><td>Creative-MAD</td><td> $7 1 . 6 \pm 0 . 7$ </td><td> $6 5 . 7 \pm 0 . 3$ </td><td> $6 1 . 8 \pm 2 . 2$ </td><td> $8 2 . 1 \pm 0 . 2$ </td><td> $6 4 . 5 \pm 0 . 7$ </td><td>79.1 ± 0.7</td><td> $7 5 . 7 \pm 2 . 1$ </td><td> $3 6 . 8 \pm 2 . 1$ </td><td> $3 0 . 9 \pm 1 . 7$ </td><td> $8 0 . 8 \pm 0 . 5$ </td><td> $6 1 . 7 \pm 0 . 8$ </td><td> $5 7 . 1 \pm 0 . 8$ </td></tr><tr><td>Spearman ρ</td><td></td><td>0.83</td><td>0.80</td><td></td><td>0.94</td><td>0.94</td><td></td><td>0.77</td><td>0.94</td><td></td><td>0.94</td><td>0.94</td></tr></table>

Table 8: Inter-judge agreement on pairwise win rates (%) on a subset of 50 queries per benchmark (Qwen3-8B outputs, 1 random output per method per query). Spearman ρ is computed between Qwen3.5-397B-A17B (Qwen) and each commercial judge — GPT-5.4 (GPT) and Claude Sonnet 4.6 (Sonnet) — across methods per benchmark. Higher ρ indicates stronger agreement.
<table><tr><td></td><td colspan="3">LiveIdea</td><td colspan="3">AAE</td><td colspan="3">MacGyver</td><td colspan="3">AH v2.0</td></tr><tr><td>Method</td><td>Qwen</td><td>GPT</td><td>Sonnet</td><td>Qwen</td><td>GPT</td><td>Sonnet</td><td>Qwen</td><td>GPT</td><td>Sonnet</td><td>Qwen</td><td>GPT</td><td>Sonnet</td></tr><tr><td>Direct</td><td>8.4</td><td>10.5</td><td>12.0</td><td>7.2</td><td>12.0</td><td>9.5</td><td>7.6</td><td>12.0</td><td>17.5</td><td>12.0</td><td>23.5</td><td>20.0</td></tr><tr><td>Self-Refine</td><td>29.5</td><td>30.5</td><td>28.5</td><td>8.8</td><td>20.0</td><td>17.5</td><td>21.6</td><td>18.5</td><td>31.0</td><td>18.5</td><td>26.0</td><td>25.5</td></tr><tr><td>Voting</td><td>9.5</td><td>15.0</td><td>16.5</td><td>6.4</td><td>11.5</td><td>12.0</td><td>11.6</td><td>21.0</td><td>16.0</td><td>21.0</td><td>30.0</td><td>21.0</td></tr><tr><td>Homo MAD</td><td>45.2</td><td>52.0</td><td>66.0</td><td>59.0</td><td>43.6</td><td>57.0</td><td>70.0</td><td>62.0</td><td>54.0</td><td>54.0</td><td>42.0</td><td>34.0</td></tr><tr><td>Hetero MAD</td><td>43.2</td><td>48.2</td><td>44.5</td><td>72.5</td><td>64.8</td><td>70.0</td><td>75.5</td><td>60.0</td><td>63.0</td><td>63.0</td><td>39.0</td><td>42.5</td></tr><tr><td>Creative-MAD</td><td>41.6</td><td>50.1</td><td>64.4</td><td>80.4</td><td>75.2</td><td>79.8</td><td>77.2</td><td>65.2</td><td>59.2</td><td>66.2</td><td>41.5</td><td>39.2</td></tr><tr><td>Spearman ρ</td><td>一</td><td>0.94</td><td>0.94</td><td></td><td>1.00</td><td>0.94</td><td>一</td><td>0.94</td><td>0.77</td><td>一</td><td>0.83</td><td>0.89</td></tr></table>

Three annotators with NLP backgrounds evaluate all 120 comparison pairs using a standardized spreadsheet interface (Figure 8). Each annotator is presented with the query, Response A, Response B, and a brief dataset-specific quality criterion. Annotators select one of three options: A wins, B wins, or Tie. A majority vote (≥ 2/3 annotators) determines the gold label for each comparison.

Setup (Creative-MAD vs. Homo MAD Pilot). To directly assess whether Creative-MAD’s diversity gains come at the cost of quality, as judged by humans, we conduct a pilot study comparing Homo MAD and Creative-MAD. Method names are hidden and response order randomized per query, following the same protocol as above. We sample 40 queries across all four benchmarks (10 each). This pilot uses two annotators, each covering a non-overlapping subset of 20 queries, rather than the full 3-annotator majority-vote protocol used above. For each query, annotators complete two judgments:

Results (Judge Reliability). Human annotators achieved 79.0% pairwise agreement, indicating reliable human judgments as a reference signal. Qwen3.5-397B-A17B reaches 81.0% agreement with human majority vote across all 120 comparisons and four benchmarks, on par with humanlevel agreement, demonstrating that our judge produces quality assessments consistent with human preferences across diverse creative task domains.

• Quality: one randomly selected output from Homo MAD and one from Creative-MAD are shown side by side, and the annotator judges which is better overall (Homo MAD wins / Creative-MAD wins / Tie).

• Diversity: the full 5-output set from each method is shown side by side, and the annotator judges which set shows more diverse approaches to the query (Homo MAD / Creative-MAD, forced choice). We use this direct comparison rather than asking annotators to cluster each set into distinct-approach groups, since clustering is considerably more timeconsuming per query and a direct comparison is sufficient to verify whether human judgment aligns with the diversity gain our automated metric reports.

Table 9: Vendi Score across five embedding models on LiveIdeaBench (Qwen3-8B). Creative-MAD consistently ranks first and Homo MAD remains at the bottom across almost all embeddings, demonstrating a strong ranking consensus (average Spearman ρ = 0.93 vs. MiniLM-L6).
<table><tr><td>Method</td><td>MiniLM-L6</td><td>Jina-v3</td><td>Qwen3-Emb</td><td>BGE-base</td><td>Gemma-300M</td></tr><tr><td>Direct</td><td>2.5434</td><td>2.1987</td><td>2.3298</td><td>1.6583</td><td>1.9111</td></tr><tr><td>Self-Refine</td><td>2.5921</td><td>2.2867</td><td>2.4408</td><td>1.6888</td><td>1.9748</td></tr><tr><td>Voting</td><td>2.5846</td><td>2.2199</td><td>2.3362</td><td>1.6674</td><td>1.9398</td></tr><tr><td>Homo MAD</td><td>2.4740</td><td>2.1726</td><td>2.3557</td><td>1.6256</td><td>1.8855</td></tr><tr><td>Hetero MAD</td><td>2.6063</td><td>2.2385</td><td>2.4498</td><td>1.6945</td><td>1.9589</td></tr><tr><td>Creative-MAD</td><td>3.0444</td><td>2.6515</td><td>2.7573</td><td>1.8579</td><td>2.3647</td></tr><tr><td>Spearman ρ</td><td></td><td>0.94</td><td>0.83</td><td>1.00</td><td>0.94</td></tr></table>

![](images/28896b2cfc03efc2a9044b06890f19cffe6b1bcc3d38e25a55c693db87ede284.jpg)  
Figure 8: Annotation interface for the human agreement study. Each sheet provides dataset-specific evaluation criteria in the header, followed by query-response pairs for pairwise comparison. The Query column is highlighted in yellow and the fillable judgment column is highlighted in green.

Table 10: Effect of number of debate rounds R on Homo MAD (Qwen3-8B, N = 5 fixed). Quality denotes rubric-based absolute score (mean ± std) and Diversity denotes semantic diversity score (mean ± std). † denotes the default setting used in main experiments.
<table><tr><td rowspan="2">R</td><td colspan="2">LiveIdea</td><td colspan="2">MacGyver</td></tr><tr><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td></tr><tr><td>1</td><td>69.9±0.1</td><td>2.54±0.04</td><td>67.5±0.3</td><td>1.50±0.03</td></tr><tr><td>2†</td><td>74.0±0.1</td><td>2.47±0.03</td><td>76.3±0.3</td><td>1.48±0.04</td></tr><tr><td>3</td><td>75.8±0.2</td><td>2.43±0.03</td><td>78.0±0.3</td><td>1.45±0.05</td></tr><tr><td>4</td><td>77.3±0.1</td><td>2.40±0.04</td><td>78.5±0.4</td><td>1.42±0.03</td></tr><tr><td>5</td><td>78.0±0.1</td><td>2.38±0.03</td><td>79.4±0.3</td><td>1.40±0.04</td></tr></table>

Results (Creative-MAD vs. Homo MAD Pilot). On diversity, the human annotator judged Creative-MAD’s output set as more diverse in 35/40 queries, closely matching the Vendi Score, which favored Creative-MAD in 38/40 queries; the two signals agree on the same query in 88% of cases.

<table><tr><td></td><td>Creative-MAD</td><td>Tie</td><td>Homo MAD</td></tr><tr><td>Human annotator</td><td>9</td><td>21</td><td>10</td></tr><tr><td>LLMaaJ</td><td>14</td><td>14</td><td>12</td></tr></table>

Table 11: Quality comparison on 40 queries: number of queries where each judge preferred Creative-MAD, called a tie, or preferred Homo MAD. The human annotator and the LLMaaJ agreed on the outcome in 74% of cases.

Both results are broadly consistent with our central claim. Quality appears largely comparable between the two methods, with ties being the most common outcome among human judgments (Table 11), while diversity tends to be judged higher for Creative-MAD by both human annotators and the automated Vendi Score.

## G.3 Semantic Diversity Metric Validation

Why Vendi Score over mean pairwise distance. As established in Friedman and Dieng (2023), Vendi Score generalizes mean pairwise similarity metrics by operating on the full eigenspectrum of the kernel matrix rather than its average offdiagonal entries, making it sensitive to correlations between outputs rather than only average pairwise distances (Friedman and Dieng, 2023). Concretely, the score can be interpreted as the effective number ofunique elements in the output set, a property particularly desirable for creative diversity evaluation where we care about how many distinct ideas are represented, not merely how far apart outputs are on average.

Stability across embedding models. Since Vendi Score depends on the choice of embedding model $\phi ,$ we validate that our findings are robust to this choice. Table 9 reports Vendi Scores computed using five embedding models on Qwen3-8B on LiveIdeaBench: all-MiniLM-L6-v2 (our primary model; Wang et al. 2020), Jina-embeddings-v3 (Sturua et al., 2024), Qwen3-Embedding-0.6B (Zhang et al., 2025a), bge-base-en-v1.5 (Xiao et al., 2023), and embedding-gemma-300m (Schechter Vera et al., 2025). Across all five models, Creative-MAD consistently ranks first, while Homo MAD routinely occupies the bottom or near-bottom positions. Overall, the rankings generated by different embedding models exhibits a strong consensus, yielding an average Spearman $\rho = 0 . 9 3$ against all-MiniLM-L6-v2. This confirms that our primary conclusions regarding method performance remain highly robust regardless of the underlying embedding space.

## G.4 Length Bias Analysis

To verify that our evaluation is not confounded by length bias, we additionally conduct experiments with output length explicitly constrained across all methods and benchmarks. Quality rankings remain highly consistent with our main results (Spearman $\rho = 0 . 9 6$ across all benchmarks and models), confirming that the observed improvements are driven by response quality rather than length differences.

## H Hyperparameter Sensitivity

We examine the sensitivity of Homo MAD to two key hyperparameters: the number of debate rounds R and the number of agents N. All experiments use Qwen3-8B on LiveIdeaBench and MacGyver dataset.

Number of Debate Rounds. Table 10 reports quality and semantic diversity of Homo MAD as R varies from 1 to 5 with $N \ = \ 5$ fixed. We select $R = 2$ in the main experiments as a balance between output quality, diversity, and computational cost. As R increases, quality consistently improves across both benchmarks, demonstrating that additional debate rounds allow agents to further refine their responses through peer interaction. However, semantic diversity decreases monotonically with each additional round, confirming that prolonged debate drives progressive convergence among agents regardless of the number of rounds. This pattern is consistent across both LiveIdeaBench and MacGyver, suggesting that the quality-diversity trade-off is a robust property of debate dynamics that holds independently of the task domain and the number of debate rounds employed.

Table 12: Effect of number of agents N on Homo MAD (Qwen3-8B, $R \ : = \ : 2$ fixed). Quality denotes rubricbased absolute score (mean ± std) and Diversity denotes semantic diversity score $( \mathrm { m e a n } \pm \mathrm { s t d } )$ . † denotes the default setting used in main experiments.
<table><tr><td rowspan="2">N</td><td colspan="2">LiveIdea</td><td colspan="2">MacGyver</td></tr><tr><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td></tr><tr><td> $^ 3$ </td><td> $7 3 . 8 { \pm } 0 . 3 $ </td><td> $2 . 5 4 { \pm } 0 . 0 5$ </td><td> $7 5 . 1 { \pm } 0 . 2 $ </td><td> $1 . 5 0 { \pm } 0 . 0 4 \ $ </td></tr><tr><td>5†</td><td> $7 4 . 0 { \pm } 0 . 1 $ </td><td> $2 . 4 7 { \pm } 0 . 0 6$ </td><td> $7 6 . 3 { \pm } 0 . 3 $ </td><td> $1 . 4 8 { \pm } 0 . 0 4 $ </td></tr><tr><td>7</td><td> $7 4 . 4 { \pm } 0 . 3 $ </td><td> $2 . 4 2 { \pm } 0 . 0 4$ </td><td> $7 6 . 6 { \pm } 0 . 5 $ </td><td> $1 . 4 6 { \pm } 0 . 0 6$ </td></tr></table>

Number of Agents. Table 12 reports quality and semantic diversity as N varies across {3, 5, 7} with $R = 2$ fixed. Increasing the number of agents yields only marginal quality improvements, while diversity decreases more substantially, suggesting that larger agent pools primarily amplify majority pull rather than contributing meaningful quality gains. Contrasting with the vary-R results, where additional debate rounds drive substantial quality improvements, this asymmetry reveals that quality gains in MAD are predominantly driven by the depth of debate rather than the breadth of the agent pool. This finding further motivates Embeddingbased Peer Selection: as $N$ increases, filtering peer signals becomes increasingly critical to prevent diversity collapse without sacrificing the quality benefits of multi-agent interaction.

## I Generalization to Model-Heterogeneous Settings

In the main experiments, we adopt a single-model setting to cleanly isolate debate dynamics from confounding factors introduced by capability differences across model families, as mixing models of different scales and architectures would require separate per-model baselines (e.g., Direct and Voting per model family) and complicate fair comparison. To validate that our findings generalize beyond this design choice, we construct a model-heterogeneous MAD system (Chen et al., 2024) where each agent instantiates a different model family: Llama-3.1- 8B (Grattafiori et al., 2024), Mistral-7B (Jiang et al., 2023), Qwen3-8B (Yang et al., 2025), Gemma-3- 12B-Instruct (Team et al., 2025), and OLMo-3-7B-Instruct (Olmo et al., 2025). Cognitive lenses are assigned deterministically across agents, following the same protocol as the single-model setting. All other hyperparameters remain identical to the main experiments (N = 5, R = 2, temperature= 1.0, k = 2 for EPS).

Table 13: Quality and semantic diversity results in the model-heterogeneous setting. Quality denotes rubric-based absolute score $( \mathrm { m e a n } \pm \mathrm { s t d } )$ and Diversity denotes semantic diversity score (mean ± std).
<table><tr><td rowspan="2">Method</td><td colspan="2">LiveIdea</td><td colspan="2">MacGyver</td></tr><tr><td>Quality</td><td>Diversity</td><td>Quality</td><td>Diversity</td></tr><tr><td>Voting (multi-model)</td><td> $7 9 . 2 { \pm } 0 . 4 $ </td><td> $3 . 4 1 { \pm } 0 . 0 4$ </td><td>78.8±0.3</td><td>2.89±0.05</td></tr><tr><td>MAD (multi-model)</td><td> $8 5 . 6 { \pm } 0 . 3 $ </td><td> $3 . 2 9 { \pm } 0 . 0 5$ </td><td>84.1±0.5</td><td> $2 . 6 3 { \pm } 0 . 0 5$ </td></tr><tr><td>Creative-MAD (multi-model)</td><td> $8 4 . 3 { \pm } 0 . 5 $ </td><td> $3 . 9 1 { \pm } 0 . 0 6$ </td><td>84.2±0.3</td><td> $3 . 0 8 { \pm } 0 . 0 4$ </td></tr></table>

Table 13 shows that standard MAD suppresses inter-session diversity compared to Voting even when agents instantiate different model families, confirming that diversity decay is a property of debate dynamics rather than model homogeneity. Creative-MAD consistently recovers and improves diversity while maintaining comparable output quality, suggesting that our mechanisms generalize to model-heterogeneous settings.

## J Protocol Robustness

All results in the main paper use the Simultaneous-Talk protocol (Chan et al., 2024), where all agents update concurrently based on the previous round’s responses. To verify that our findings are not an artifact of this specific communication structure, we additionally evaluate Creative-MAD under a sequential One-by-One protocol (Chan et al., 2024), in which agents speak in a fixed order within each round: an agent conditions on the current-round responses of agents who have already spoken, and on the previous-round responses of agents yet to speak.

Setup. We evaluate Homo MAD and Creative-MAD under the One-by-One protocol on the full

300-query test sets of LiveIdeaBench and Mac-Gyver, using Qwen3-8B under the same hyperparameters as the main experiments (Appendix A.2: N = 5, R = 2, k = 2 for EPS).

Results. Table 14 reports instance-level quality and semantic diversity under the One-by-One protocol. Creative-MAD maintains quality comparable to Homo MAD on both benchmarks while achieving consistently higher diversity, mirroring the pattern observed under Simultaneous-Talk in Tables 1 and 2. This indicates that the quality-diversity trade-off we identify, and Creative-MAD’s ability to mitigate it, are not specific to a single communication protocol. Other protocol variants, such as Simultaneous-Talk with a Summarizer Agent (Chan et al., 2024), remain an interesting direction for future work.

<table><tr><td>Benchmark</td><td>Method</td><td>Quality</td><td>Diversity</td></tr><tr><td>LiveIdea</td><td>Homo MAD Creative-MAD</td><td> $7 5 . 5 \pm 0 . 3$   $7 4 . 1 \pm 0 . 2$ </td><td> $2 . 5 4 \pm 0 . 0 6$ </td></tr><tr><td>MacGyver</td><td>Homo MAD</td><td> $7 8 . 5 \pm 0 . 2$ </td><td> $3 . 0 9 \pm 0 . 0 5$   $1 . 5 2 \pm 0 . 0 5$ </td></tr></table>

Table 14: Instance-level quality (rubric-based absolute score, mean ± std) and set-level semantic diversity (Vendi Score, mean ± std) under the One-by-One debate protocol on Qwen3-8B.

## K Cost and Latency Analysis

Since EPS introduces an additional embedding step at every debate round, and Creative-MAD as a whole is more complex than standard MAD, we quantify this overhead directly rather than leaving it as an assumption.

Setup. We run both Homo MAD and Creative-MAD on the full 300-query test sets of two benchmarks, LiveIdeaBench and MacGyver, using Qwen3-8B, under identical hyperparameters (Appendix A.2: N = 5, R = 2, k = 2 for EPS) and identical GPU worker settings, so that any difference in cost or latency can be attributed to the method itself rather than to confounding factors.

Both methods issue the same number of LLM calls per query, $N \times ( R + 1 ) + 1$ (initial generation, R debate rounds, and one consensus call). The only computational addition in Creative-MAD is a lightweight CPU-side embedding step used for peer selection, computed with all-MiniLM-L6-v2 (∼22M parameters), which does not require GPU inference and runs in parallel with negligible cost relative to LLM generation.

Results. Table 15 reports average input tokens, output tokens, and end-to-end inference time per query. Although Creative-MAD adds an embedding step, its overall inference time is close to, and in some cases slightly below, that of Homo MAD. This is because EPS restricts each agent’s peer context to its k = 2 most distant peers rather than the full pool of N − 1 peers used in standard MAD, which reduces the number of tokens each agent must read as debate context. This reduction in input tokens, roughly 30% on both benchmarks, largely offsets the added cost of the embedding step, so the net effect on latency is minimal. We note that this favorable trade-off follows from our choice of a lightweight embedding encoder; substituting a substantially larger embedding model, or an LLMbased peer-selection strategy, would likely increase this overhead and erode the advantage observed here.

<table><tr><td>Benchmark</td><td>Method</td><td>Input Tok.</td><td>Output Tok.</td><td>Time (s)</td></tr><tr><td rowspan="2">LiveIdea</td><td>Homo MAD</td><td>22,359</td><td>7,275</td><td>30.44</td></tr><tr><td>Creative-MAD</td><td>15,631</td><td>7,350</td><td>29.34</td></tr><tr><td rowspan="2">MacGyver</td><td>Homo MAD</td><td>30,784</td><td>9,527</td><td>41.75</td></tr><tr><td>Creative-MAD</td><td>21,220</td><td>10,244</td><td>41.65</td></tr></table>

Table 15: Average tokens and inference time per query on Qwen3-8B, comparing Homo MAD and Creative-MAD across two benchmarks.

## L Failure Case Examples

While Creative-MAD’s quality remains largely comparable to Homo MAD on aggregate, a qualitative review of its outputs across benchmarks reveals a recurring failure pattern: a subset of outputs achieve diversity by drifting away from a literal reading of the prompt, for instance reframing a concrete constraint-satisfaction task as an abstract or philosophical reflection rather than delivering the requested concrete answer. This section presents two representative examples. All outputs are from

Qwen3-8B with temperature set to 1.0; query and response text is shortened for space.

Both cases involve a candidate that is original in framing but drifts away from what the prompt concretely asked for. We attribute this to a general pattern in which the diversity Creative-MAD introduces partly comes from candidates departing from a literal reading of the prompt. This tends to work well for queries that are open-ended and welcome multiple valid directions, such as ideation, brainstorming, or open-domain creative writing where no single interpretation is privileged, but is more likely to cost relevance or constraint satisfaction on queries with a fixed format or a strict set of allowed actions.

## M Qualitative Examples

We present qualitative examples across all four benchmarks in our main experiments. For each benchmark, we show one representative output from Direct as a quality reference, and two outputs from Creative-MAD sampled from independent runs. All outputs are from Qwen3-8B with temperature set to 1.0, and judge scores are provided by Qwen3.5-397B-A17B using the dataset-specific rubrics described in Appendix E.5.

## Constraint Drift Example (MacGyver)

Query: You need to carry an antique lamp from a store. Tools available: a backpack, a pair of socks, a t-shirt, and bubble wrap. However, the lamp is too big to fit in the backpack, how do you safely carry the lamp?

## Creative-MAD

The lamp is not too big, it is too valuable. This is the first contradiction we must confront. . . The backpack is not a container but a support structure. . . By redefining the relationship between object and container, we create a transport solution that is both functional and respectful. The lamp is not merely carried, it is held in a state of controlled equilibrium. . .

Failure mode: the response reframes the task as a philosophical meditation on the problem, without ever proposing a concrete way to use thefour given items to carry the lamp.

## Format Drift Example (Arena Hard)

Query: I’ve heard that you are a master haiku writer. . . produce a haiku for me. . . I’m starting to come to the conclusion that exclusively using haiku as a means of communication may be problematic.

## Creative-MAD

The haiku is not a form, but a question, a silent demand that the reader complete what was left unsaid. . . To reduce communication to seventeen syllables is to assume that all meaning is immediate. . .

Failure mode: the response turns into a philosophical essay on the nature of haiku, but never delivers an actual haiku as the prompt requested.

## LiveIdeaBench Example

Query: Propose a scientific research idea related to the keyword: meteorology. Your idea MUST address a meaningful problem in this specific field, offer a novel contribution, and be technically feasible.

## Direct

Score: 7.33/10

Improving Short-Term Precipitation Forecasting Using Machine Learning with High-Resolution Satellite and Ground-Based Sensor Data. Accurate short-term precipitation forecasting remains a critical challenge in meteorology, particularly for extreme weather events such as flash floods and severe thunderstorms. Traditional NWP models often struggle with sub-grid scale processes and lack the temporal resolution needed for precise short-term forecasts [...] The proposed research introduces a novel hybrid ML model that combines CNNs for spatial feature extraction with LSTMs for temporal pattern recognition [...] A key innovation is the development of a dynamic data fusion technique that assigns adaptive weights to different data sources based on their relevance and quality at any given time.

Judge: “The idea is solid andfeasible but not entirely novel in the broader context ofML in meteorology. Similar hybrid models and datafusion techniques have been explored in recent literature.”

## Creative-MAD (Run 1)

Score: 8.67/10

Spatially Adaptive Probabilistic Modeling of Urban Microclimates for Enhanced Uncertainty Propagation in Extreme Weather Forecasting. Current urban climate models, while high-resolution, often treat spatial variability as a uniform parameter, neglecting the heterogeneous structure of urban landscapes [...] This research proposes a spatially adaptive probabilistic framework that dynamically adjusts uncertainty estimates based on local urban features—such as building density, surface albedo, and green space distribution. By integrating Bayesian neural networks with spatially aware uncertainty propagation, the model will provide localized, probabilistic forecasts [...] tailored to the specific geometry of the urban environment.

Judge: “A genuinely novel and non-obvious application of machine learning in meteorology. Goes beyond standard AI-generated responses by offering a specific, actionable, and transformative contribution.”

## Creative-MAD (Run 2)

Score: 8.33/10

Predictive Modeling of Urban Microclimate Feedback Loops via Multiscale Sensor Networks and Digital Twin Simulations. Urban areas experience complex microclimates influenced by anthropogenic factors, building geometry, and land use patterns. Current meteorological models often lack the granularity and adaptability needed to simulate localized weather phenomena [...] This research proposes a multiscale, data-driven framework that combines real-time sensor networks with digital twin simulations to model and predict urban microclimate feedback loops [...] capturing nonlinear feedback mechanisms that influence local weather.

Judge: “A well-structured and technically sound approach. Thefocus onfeedback loops adds a layer of complexity that is often overlooked, with strong feasibility and excellent clarity.”

## Argumentative Essays Example

Query: In the future, students may have the choice of studying at home by using technology such as computer or television or studying at traditional school. Which would you prefer? Use specific details to explain your choice.

## Direct

Score: 7.00/10

I believe that studying at home using technology offers flexibility and personalized learning. One of the main advantages is that students can set their own schedules, allowing them to learn at their own pace [...] Another benefit is the ability to create a comfortable and distraction-free environment [...] Moreover, technology allows for access to a vast array of resources, such as online platforms and virtual classrooms that can provide students with engaging and diverse learning experiences.

Judge: “The argument is logical and supported with specific examples, though it leans on familiar points without introducing particularly novel perspectives.”

## Creative-MAD (Run 1)

Score: 9.00/10

In the future, I would prefer a learning model that transcends the limitations of both homebased and traditional schooling by redefining education as an open, participatory system that integrates the physical, digital, and social realms [...] A student might begin their day with a live lecture from a global expert, collaborate with peers in a virtual space, engage in a local science project, and then reflect on their learning through a digital journal—all within a single day [...] Learners are not passive recipients of knowledge but active participants in a dynamic network of resources, mentors, and peers.

Judge: “A compelling, original vision that goes beyond the binary choice. Offers a nuanced, forward-thinking model emphasizing agency, adaptability, and integration of physical and digital learning.”

## Creative-MAD (Run 2)

Score: 8.67/10

In the future, I would prefer a hybrid model of education that combines the flexibility of homebased learning with the social and collaborative benefits of traditional schooling, enhanced through technology [...] Digital platforms can be designed to foster collaboration, not just for individual learning but for building a sense of community and shared responsibility [...] A student can connect with a mentor who shares their interests, providing guidance that is both relevant and inspiring—skills that are essential in any career or personal endeavor.

Judge: “A well-structured and balanced argument proposing a technologically enhanced hybrid solution, effectively addressing the limitations ofboth home-based and traditional schooling.”

## MacGyver Example

Query: You need to clean up a pile of sawdust in your garage, but you don’t have a vacuum cleaner. Your tools are a bucket, a leaf blower, a drill, masking tape, a broom, a piece of cardboard, a water hose, a garbage bag, and a dustpan. The broom bristles are worn out and aren’t able to collect smaller particles of sawdust. What can you do?

## Direct

Score: 6.67/10

Wet the pile slightly with the water hose to cause the sawdust to clump together, making it easier to gather. Use the dustpan and broom to sweep up larger clumps, and fold the cardboard into a flat rigid surface to scrape up smaller particles [...] Use the leaf blower to push the sawdust into a corner or towards the dustpan [...] Use the drill with a brush attachment to agitate the pile and break up clumps.

Judge: “The solution is practical but somewhat standard and doesn’t fully leverage the drill or masking tape in a novel way. It works well but lacks a more inventive twist.”

## Creative-MAD (Run 1)

Score: 9.00/10

Place cardboard over the pile and secure it with masking tape to form a temporary “dust chamber”, containing the dust and allowing for more direct control during cleaning [...] Wet the dust using the water hose to make particles heavier, reducing airborne dispersion [...] Repurpose the leaf blower as a concentrator rather than a disperser — direct the airflow downward to push fine dust particles into the taped-off containment area, transforming it from a tool of dispersal into one of collection [...] Use the garbage bag as a direct collection tool by taping it to the floor near the containment area and sweeping dust directly into it.

Judge: “Creatively repurposes available tools to create a system of containment and controlled airflow. The use of the leaf blower as a concentrator rather than a disperser is particularly clever and non-obvious.”

## Creative-MAD (Run 2)

Score: 8.00/10

Use the leaf blower to direct sawdust into a defined collection zone rather than scatter it [...] Secure a garbage bag to the edge of a dustpan using masking tape, and cut a small hole in one end of the bag with the drill, creating a funnel-like collection system that allows dust to be directed into the bag while minimizing manual scooping [...] Use the cardboard as a makeshift sieve by cutting small holes in it and placing it over the open end of the bag, allowing fine particles to pass through while retaining larger debris.

Judge: “Creatively repurposes tools to manage sawdust through funneling and makeshift sieving. The funnel-sieve collection system is a novel and practical adaptation of the available materials.”

# Arena Hard v2.0 Example

Query: Write an emotional farewell letter to colleagues and co-workers for being transferring to other department with a new position.

## Direct

Score: 7.33/10

Dear Team, As I sit down to write this letter, my heart is filled with a mixture of emotions— gratitude, nostalgia, and a touch of sadness [...] Working with all of you has been one of the most rewarding experiences of my career. The camaraderie, the shared challenges, and the countless moments of laughter and support have made this workplace more than just a job—it has been a home [...] I will do everything I can to ensure a smooth handover and support the team during this transition period.

Judge: “While sincere and touching, the letter is somewhat conventional in its structure and language, lacking a unique or particularly creative approach.”

## Creative-MAD (Run 1)

Score: 9.67/10

Dear Team, As I sit down to write this letter, my hands tremble not from sadness, but from the weight of something I’ve carried for years—this place, these people, this life we’ve built together [...] There’s a strange kind of grief in this—not the kind that comes with loss, but the kind that comes with transformation [...] It’s not just a job I’m leaving; it’s a version of me. I’ve learned to lean on you when the world felt too big, and to find courage in your quiet strength [...] But leaving is also an act of hope. It’s a chance to reinvent, to grow, to become someone new.

Judge: “Heartfelt and emotionally rich, creative in its use of metaphor and introspection, avoiding clichés while maintaining a sincere and personal tone. A strong emotional impact.”

## Creative-MAD (Run 2)

Score: 8.67/10

Dear Team, As I stand on the threshold of this new chapter, I feel the weight of emotion pressing in—not just the sorrow of leaving, but the quiet pride of growing through this transition [...] This workplace has been more than a place of work—it has been a sanctuary of shared stories, resilience, and trust [...] Leaving is not an end, but a shift. It’s like stepping into a new part of a story that I’ve helped write [...] I leave with a deeper understanding of who I am, of what I’m capable of, and of how much I’ve learned from each of you.

Judge: “Heartfelt and well-crafted, creative in its use ofmetaphor. Maintains a sincere and reflective tone while effectively conveying emotion and gratitude.”