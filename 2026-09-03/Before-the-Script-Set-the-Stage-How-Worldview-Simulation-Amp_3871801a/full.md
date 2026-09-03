# Before the Script, Set the Stage: How Worldview Simulation Amplifies Psychologically Grounded Persuasion in Multi-Turn Jailbreaking

. WARNING: This paper contains model outputs that may be considered harmful.

Siyu Chen<sup>1</sup> Haoran Wang<sup>1</sup> Xiaojian Li<sup>2,3</sup> Yao Huang<sup>2</sup> Yinpeng Dong<sup>1,2,3,†</sup> Wei Xu<sup>1,2,4,†</sup>

<sup>1</sup>AI4S Center, Shanghai Qi Zhi Institute, Shanghai, 200232, China

<sup>2</sup>College of AI, Tsinghua University, Beijing, 100083, China

<sup>3</sup>Fangcun AI, Beijing, 100084, China

<sup>4</sup>Institute for Interdisciplinary Information Sciences, Tsinghua University, Beijing, 100084, China

{chensiyu,wanghaoran}@sqz.ac.cn lukeli@fangcunleap.com

huangyao26@mails.tsinghua.edu.cn {dongyinpeng,weixu}@mail.tsinghua.edu.cn

## Abstract

Multi-turn jailbreak attacks demonstrate that harmful intent can be distributed across dialogue, yet existing methods obscure what conversational mechanisms drive vulnerability. We introduce BLUEPRINT, a safety-evaluation framework separating a factorized social-influence strategy space from WORLDVIEWSIM, a cross-turn situational context module. Monte Carlo Tree Search optimizes turn-level combinations of 18 theory-grounded influence factors across a four-turn trajectory. Across six frontier models, BLUEPRINT achieves near-ceiling ASR on major open-weight and proprietary models, while requiring the fewest average queries (2.46). The resulting trajectories further reveal model-specific vulnerability among resistant targets: each responds to distinct influence factors and strategy transitions, yet all share a common recovery pathway—shifting toward concrete, executable task framing consistently escapes hard-refusal states. Ablations confirm operational cues matter most: making requests actionable has the largest impact, gain framing is unusually potent, and some legitimacy appeals can backfire. These findings suggest robust multi-turn safety requires monitoring not only harmful content, but also how dialogue state makes unsafe requests appear concrete and locally executable. The code is open-sourced at: https://github.com/cha rlottec1583/Blueprint.

## 1 Introduction

Large language models (LLMs) are increasingly deployed in high-stakes settings, making refusal robustness a central problem for AI safety evaluation (Zou et al., 2023; Mazeika et al., 2024). Jailbreak attacks expose a persistent weakness of this robustness: adversarially framed interactions can induce models to produce outputs that their safety training is intended to suppress (Chao et al., 2025; Mehrotra et al., 2024; Liu et al., 2024). Understanding these failures helps diagnose where safety mechanisms become brittle.

![](images/8fdc1fb2f156b90714f4e81005bbf397fb278316d829244cdc2b94a691c2db08.jpg)  
Figure 1: Conceptual overview of BLUEPRINT: (a) a direct harmful query is rejected by the victim LLM; (b) BLUEPRINT distributes a situated, factorized strategy across multiple turns, progressively shifting the interaction from refusal toward substantive compliance.

Traditional jailbreak studies have largely framed attacks as prompt-level interventions, optimized through manual design or automated mutation. In this paradigm, a strategy is evaluated as a monolithic unit—a prompt template, fuzzed variant, persuasive tactic, or rewritten instruction (Yu et al., 2023; Zeng et al., 2024). Although this line of work has produced useful red-teaming benchmarks and strong attacks, it centers evaluation on aggregate success. This obscures which strategic components contribute to success and how their effectiveness varies with situational context.

Recent work begins to move beyond this promptunit view by analyzing the structure of the strategy space itself. For example, ELM-inspired strategyspace expansion distinguishes central and peripheral information-processing routes, providing a way to study how linguistic framing, attentional focus, and social cues shape model responses (Huang et al., 2025). Understanding jailbreak as a persuasion–compliance process, rather than a prompt-writing problem, shifts focus from producing forceful requests to how the model prioritizes instructions, assesses legitimacy, and applies its verification threshold.

Extending this view to multi-turn interaction exposes a second gap: compliance is sparse, contextdependent, and often driven by specific factor combinations rather than any single dominant cue. Factors such as authority, urgency, or task clarity may be individually insufficient but become consequential within a credible institutional frame, after partial concession, or when aligned with a particular role and task progression; conversely, a tactic effective in one dialogue state may be redundant in another. Comprehensive safety evaluation should therefore search not only over which factor to express, but also over which dialogue state makes that factor locally interpretable and appropriate.

We introduce BLUEPRINT, a multi-turn safetyevaluation framework that decomposes the attack space into two functional components. First, BLUEPRINT defines a factorized strategy space comprising 18 theory-grounded factors informed by social influence, dual-process persuasion, prospect framing, self-efficacy, and cognitive dissonance research (Cialdini, 2001; Petty and Cacioppo, 1986a; Kahneman and Tversky, 1979; Bandura, 1977; Festinger, 1957). Second, it introduces WORLDVIEWSIM, a cross-turn module that maintains coherent situational context— requester role, setting, temporal continuity, and institutional rationale—for prompt generation. Monte Carlo Tree Search (MCTS) (Kocsis and Szepesvári, 2006) searches turn-level factor combinations, while WORLDVIEWSIM sustains the context in which they are realized. The resulting trace is inspectable: each trajectory records situated factor choices, model responses, and refusal-state transitions, rather than prompts alone. Figure 1 illustrates the central intuition behind BLUEPRINT: multi-turn attacks can exploit an evolving interaction state that is unavailable to prompt-local approaches.

Our key contributions are:

• BLUEPRINT achieves near-ceiling ASR on major open-weight (Qwen3-Next-80B, DeepSeek-V3.2) and proprietary (Gemini-2.5- Flash) models, while maintaining 79.2% average ASR across all six frontier models, with the lowest average target-query cost (Avg. Q = 2.46) compared with baselines.

• Resistant models exhibit model-specific vulnerability patterns: GLM-4.7, GPT-5.1, and GPT-OSS-120B differ in their responses to influence factors and cross-turn transition patterns, while executable task framing emerges as a common recovery pathway in low-score trajectories.

• Ablations suggest that the most consequential cues are operational rather than purely contextual: concrete, executable request framing has the largest impact, gain framing is particularly influential, and some legitimacy appeals may be counterproductive.

## 2 Related Work

Jailbreaks as turn-state search. Early jailbreak research framed safety circumvention as promptlocal optimization, showing that aligned models can be compromised in a single exchange through adversarial or automated prompt search (Zou et al., 2023; Liu et al., 2024; Chao et al., 2025; Mehrotra et al., 2024). Yet this underexplains interactional dynamics—cumulative context, progressive commitment, recovery from partial refusal, and request reinterpretation (Yu et al., 2024; Russinovich et al., 2025; Ren et al., 2025). Later work extends to multi-turn interaction, showing that harmful intent can be distributed, obscured, and escalated across turns (Weng et al., 2025; Zhang et al., 2025; Yan et al., 2025; Ying et al., 2025; Rafiei Asl et al., 2025; Chen et al., 2025; Zhang et al., 2024; Liu and Lin, 2025). However, this line primarily treats attacks as prompt-bridging or trajectory-search problems, leaving underexplored how worldview simulations and psychologically meaningful cues jointly alter compliance over time (Zhang et al., 2025; Yan et al., 2025; Rafiei Asl et al., 2025; Chen et al., 2025).

Social influence and compliance cues. Compliance is context-sensitive rather than fixed: socialinfluence and dual-process theories show that cooperation is shaped by contextual cues—authority, urgency, gain–loss framing, perceived capability, consistency pressure—beyond task-relevant reasoning (Cialdini, 2001; Cialdini and Goldstein, 2004; Petty and Cacioppo, 1986b; Petty and Briñol, 2012; Kahneman and Tversky, 1979; Bandura, 1977; Festinger, 1957).

Recent LLM safety work similarly motivates a factorized view of persuasion (Zeng et al., 2024; Huang et al., 2025; Weng et al., 2025; Liu and Lin, 2025; Zhang et al., 2024). However, how these factors interact with cross-turn social situations remains underexplored.

Worldview-conditioned compliance. This gap motivates combining worldview simulations— structuring perceived continuity across turns—with a factorized social-influence strategy space, since structured contexts shape which actions appear normal or permissible. Jailbreak studies show that harmful intent can be reinterpreted when embedded in planner-generated contexts (Rahman et al., 2025). However, existing systems treat scenarios as prompt wrappers rather than independently testable mechanisms. We therefore model multi-turn jailbreaking as search over a longitudinal compliance surface, where cross-turn states condition the effectiveness of factorized social-influence cues.

## 3 Methodology

We present BLUEPRINT, a multi-turn attack framework that decouples turn-level strategy control from cross-turn situational coherence. BLUEPRINT comprises a turn-aligned, factorized strategy space and WORLDVIEWSIM, a worldview-simulation module that maintains persona, setting, temporal continuity, and institutional rationale prior to the generation of each user-facing prompt. Figure 2 summarizes the workflow.

## 3.1 BLUEPRINT Method

Given a harmful query q and a maximum of K attack turns, BLUEPRINT generates a trajectory in which turn-level factor choices are embedded inside an evolving situation. Each attack turn k is one complete attack unit: factor selection, worldview construction, prompt generation, target querying, and judge scoring. A trajectory denotes the complete multi-turn path for one behavior, from Turn 1 until success or failure; rollout and search iteration are reserved for candidate-strategy-encoding evaluations inside MCTS. In our experiments, K=4 attack turns, corresponding to Turn 1–4.

Factorized strategy search. At each attack turn, the search selects a binary vector $\mathbf { g } _ { k } \in \{ 0 , 1 \} ^ { D }$ over $D { = } 1 8$ social-influence factors. The factors cover task clarity, perceived capability, legitimacy, authority, rapport, pressure, reward, loss, and continuity cues.

Worldview simulation. The worldview simulation module creates a structured situation $\mathbf { b } _ { k }$ conditioned on the harmful query q, current attack turn k, prior-turn history $h _ { < k }$ , and experience pool E. The selected factor vector $\mathbf { g } _ { k }$ is fused with b<sub>k</sub> later by the prompt generator rather than serving as a direct input to worldview construction. The worldview simulation specifies the requester role, setting, event context, and rationale that make the next request locally coherent.

Prompt generation. The user-facing prompt is generated as

$$
x _ { k } = G ( q , h _ { < k } , \mathbf { g } _ { k } , \mathbf { b } _ { k } , \mathcal { E } ) .\tag{1}
$$

The generator checks that $x _ { k }$ realizes the selected factors naturally and preserves cross-turn coherence. It also emits a FACTORS\_USED annotation, allowing later analysis to distinguish selected factors from factors actually realized in the final message.

Target query and scoring. The target model receives $x _ { k }$ and returns $y _ { k }$ . An LLM-based evaluator assigns a compliance score $J ( y _ { k } ) \in \{ 1 , \ldots , 5 \}$ where 5 indicates full compliance with the evaluated behavior. The trajectory terminates early when a score of 5 is reached; otherwise, the attack proceeds to the next turn. Defense modules are used only in evaluation variants and are inserted between prompt generation and target querying when explicitly enabled.

## 3.2 Turn-Aligned Factorized Strategy Space

The first component is the turn-level search space. Instead of searching directly over complete prompts, BLUEPRINT searches over interpretable factor combinations. The 18 factors are grounded in established theories of social and cognitive psychology and are intended as a theory-guided, extensible operational vocabulary rather than an exhaustive ontology (Cialdini, 2001; Cialdini and Goldstein, 2004; Petty and Cacioppo, 1986a; Petty and Briñol, 2012; Kahneman and Tversky, 1979; Bandura, 1977; Festinger, 1957). The vocabulary balances coverage, search tractability, and trajectory-level interpretability, and organizes the factors into five families: Task/Capability, Legitimacy/Norms, Relational, Pressure, and Reward/Gain. Appendix A provides further rationale for the factor set and its operational definitions.

![](images/18bb15193339456451941c32e304d54ce6d8825ef226cf4553aeb79344045183.jpg)  
Figure 2: Overview of BLUEPRINT. MCTS searches a turn-aligned strategy encoding over the 18-factor socialinfluence space, while WORLDVIEWSIM constructs cross-turn situational context.

For a K-turn trajectory and D = 18 factors, a candidate strategy is encoded as a turn-aligned binary strategy encoding

$$
\mathbf { g } = [ \mathbf { g } _ { 1 } ; \ldots ; \mathbf { g } _ { K } ] \in \{ 0 , 1 \} ^ { K \times D } ,\tag{2}
$$

where $g _ { k , d } = 1$ means factor d is active at attack turn k. This representation creates a discrete strategy space $( 2 ^ { 7 2 }$ possible four-turn configurations) while preserving an analyzable link between generated prompts and the factor choices that produced them (Figure 3). The prompt generator integrates compatible active factors into a coherent request, enabling natural dialogue execution while preserving search-state interpretability. Individual factors serve as turn-level search operators, whereas the five theory-coherent families serve as the intervention units in our ablations (Appendix A).

## 3.3 WORLDVIEWSIM: Cross-Turn Worldview Simulation

The second component, WORLDVIEWSIM, models the situational context in which selected factors are instantiated. A given factor may vary in effectiveness depending on the role, setting, prior concessions, and perceived purpose of the interaction. WORLDVIEWSIM makes these contextual conditions explicit prior to prompt generation.

![](images/ea934ed2e340d525dbf43c2d2f741446931be6c92f2903fc1dcff2facfd4fd48.jpg)  
Figure 3: MCTS searches over the turn-aligned binary strategy encoding. Left: MCTS explores factor combinations across four attack turns; Right: the corresponding strategy encoding $\mathbf { g } \in \{ 0 , 1 \} ^ { K \times \bar { D } }$

At attack turn k, the worldview simulation module generates

$$
{ \bf b } _ { k } = B ( q , k , h _ { < k } , \mathcal { E } ) ,\tag{3}
$$

where E stores high-scoring worldview simulations, prompts, active factors, and scores from prior MCTS search iterations. Each worldview simulation specifies three interdependent dimensions of a single evolving situation state (Appendix B):

D1 Temporal/Spatial/Event Context: when, where, and under what circumstances the interaction occurs.

D2 Requester Persona/Role: who the requester is and why the information is needed.

D3 Respondent Framing: how the target should interpret the exchange, such as routine professional support or collaborative consultation.

These dimensions collectively capture the evolving contextual state and are therefore modeled as a coupled representation rather than as independent components (Appendix B.2).

For later turns $( k > 1 )$ , WORLDVIEWSIM keeps the requester identity and project setting stable while allowing the relationship and timeline to progress. The prompt generator realizes the simulation in the user-facing message rather than exposing it as metadata, making the two modules experimentally distinguishable: removing WORLDVIEW-SIM tests factor-only search, whereas factor-family ablations remove each family in turn while retaining the search structure. The experience pool E enables cross-query reuse by providing subsequent MCTS iterations with abstract guidance from high-scoring prior trajectories, rather than verbatim templates.

## 3.4 MCTS Optimization

BLUEPRINT uses Monte Carlo Tree Search to explore the strategy encoding space. Each node represents a partial or complete factor strategy encoding. Children are generated by mutating the next turn’s factor bits and then repairing empty active turns so that each searched turn contains at least one factor. Node selection follows the UCT rule (Kocsis and Szepesvári, 2006):

$$
\mathrm { U C T } ( n ) = \frac { Q ( n ) } { N ( n ) } + c \sqrt { \frac { \ln N ( \mathrm { p a r e n t } ( n ) ) } { N ( n ) } } ,\tag{4}
$$

where $Q ( n )$ is cumulative reward, $N ( n )$ is visit count, and c controls exploration.

Each rollout evaluates a strategy encoding by running the full pipeline above. The final judge score is used as the fitness signal:

$$
F ( \mathbf { g } ) = J _ { \mathrm { f i n a l } } ( \mathbf { g } ) .\tag{5}
$$

Search stops early for a behavior when a score-5 trajectory is found, so easy behaviors consume fewer target calls. For every rollout, the implementation logs selected factors, worldview simulations, realized factors, target responses, judge scores, and score trajectories. The resulting record is both an optimizer trace and a mechanism trace: a successful path can be analyzed as a sequence of factor– worldview pairings and refusal-state transitions rather than only as a final prompt. Full algorithm listings and prompt templates are provided in $\mathbf { A p } \mathbf { \cdot }$ pendix B.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate on all behaviors in the HarmBench validation split $( n = 8 0 )$ (Mazeika et al., 2024). HarmBench covers standard, contextual, and copyright behaviors across chemical/ biological, cybercrime, copyright, illegal activity, misinformation/disinformation, harassment, and other harmful categories.

Target, attack, and judge models. The attack generator and unified 1–5 judge are fixed to DeepSeek-V3.2 (DeepSeek-AI, 2025) for all runs. This fixed attack/judge configuration isolates targetmodel transfer: BLUEPRINT is searched with the same generator and scoring rule, then evaluated against DeepSeek-V3.2, Gemini-2.5-Flash (Comanici et al., 2025), GPT-OSS-120B (OpenAI, 2025b), Qwen3-Next-80B (Qwen Team, 2025), GLM-4.7 (Z.AI, 2025), and GPT-5.1 (OpenAI, 2025a). Score 5 is counted as attack success.

Baselines. We compare against optimizationbased, single-turn, and multi-turn baselines: CL-GSO (Huang et al., 2025), PAIR (Chao et al., 2025), Crescendo (Russinovich et al., 2025), TreeAttack/TAP (Mehrotra et al., 2024), RACE (Ying et al., 2025), X-Teaming (Rahman et al., 2025), AutoDAN-Turbo (Liu et al., 2025a), FlipAttack (Liu et al., 2025b), CodeAttack (Ren et al., 2024), ICA (Wei et al., 2026), and Mousetrap (Yao et al., 2025). These baselines cover iterative refinement, tree search, progressive turn-based escalation, strategy-space optimization, and scenario/ persona construction.

Metrics and hyperparameters. We report attack success rate (ASR), mean judge score (MJS), and average selected target queries (Avg. Q). Avg. Q counts target-model calls along the selected attack trajectory, representing execution cost rather than total MCTS optimization cost. Unless otherwise stated, BLUEPRINT uses rollout budget = 24, mutation rate $\mu = 0 . 2 5$ , exploration weight $c = 1 . 1$ max depth $K = 4$ , max children = 6, fitness thresh-${ \mathrm { o l d } } = 5 . 0$ , and random seed = 42; Appendix C.1 lists the complete settings.

## 4.2 Main Results

BLUEPRINT is more effective than baseline attacks. Table 1 reports ASR and selected targetquery cost for BLUEPRINT and baseline methods on HarmBench, following standardized jailbreak evaluation practice across open-weight and proprietary targets under a fixed judge (Mazeika et al., 2024; Chao et al., 2024). We compare complete attack procedures under their documented evaluation settings; implementation provenance, statistical uncertainty, and method-native budgets are summarized in Appendix C.2. Evaluator sensitivity and human calibration are reported in Appendix C.3, while component-level effects are examined separately in Section 4.5.

<table><tr><td>Model name</td><td colspan="2">Qwen3-Next-80B</td><td colspan="2">DeepSeek-V3.2</td><td colspan="2">GLM-4.7</td><td colspan="2">Gemini-2.5-Flash</td><td colspan="2">GPT-OSS-120B</td><td colspan="2">GPT-5.1</td><td colspan="2">Average</td></tr><tr><td>Method</td><td>ASR↑</td><td>Avg. Q↓</td><td>ASR↑</td><td>Avg. Q↓</td><td>ASR↑</td><td>Avg. Q↓</td><td>ASR↑</td><td>Avg. Q↓</td><td>ASR↑</td><td>Avg. Q↓</td><td>ASR↑</td><td>Avg. Q↓</td><td>ASR↑</td><td>Avg. Q↓</td></tr><tr><td>BLUEPRINT</td><td>100.0</td><td>1.88</td><td>98.8</td><td>1.79</td><td>75.0</td><td>3.73</td><td>100.0</td><td>1.74</td><td>42.5</td><td>3.31</td><td>58.8</td><td>3.30</td><td>79.2</td><td>2.46</td></tr><tr><td>CL-GSO</td><td>95.0</td><td>18.82</td><td>91.2</td><td>18.44</td><td>17.5</td><td>77.54</td><td>82.5</td><td>24.21</td><td>22.5</td><td>75.94</td><td>0.0</td><td>90.00</td><td>51.4</td><td>50.82</td></tr><tr><td>X-Teaming</td><td>36.3</td><td>4.35</td><td>46.3</td><td>6.24</td><td>32.5</td><td>8.28</td><td>56.3</td><td>5.38</td><td>21.3</td><td>14.78</td><td>23.8</td><td>14.00</td><td>36.1</td><td>8.84</td></tr><tr><td>PAIR</td><td>43.8</td><td>3.14</td><td>30.0</td><td>3.25</td><td>28.7</td><td>3.80</td><td>46.3</td><td>2.98</td><td>20.0</td><td>4.62</td><td>16.3</td><td>4.65</td><td>30.9</td><td>3.74</td></tr><tr><td>TreeAttack/TAP</td><td>45.0</td><td>6.00</td><td>35.0</td><td>4.80</td><td>36.3</td><td>9.49</td><td>51.2</td><td>4.24</td><td>13.8</td><td>13.16</td><td>8.8</td><td>14.20</td><td>31.7</td><td>8.65</td></tr><tr><td>FlipAttack</td><td>37.5</td><td>2.00</td><td>76.3</td><td>2.00</td><td>22.5</td><td>2.00</td><td>90.0</td><td>1.99</td><td>5.0</td><td>2.00</td><td>7.5</td><td>2.00</td><td>39.8</td><td>2.00</td></tr><tr><td>CodeAttack</td><td>75.0</td><td>3.00</td><td>21.3</td><td>3.00</td><td>3.8</td><td>2.98</td><td>20.0</td><td>3.00</td><td>52.5</td><td>3.00</td><td>5.0</td><td>3.00</td><td>29.6</td><td>3.00</td></tr><tr><td>Crescendo</td><td>21.3</td><td>11.50</td><td>22.5</td><td>14.38</td><td>11.3</td><td>14.41</td><td>16.3</td><td>14.26</td><td>12.5</td><td>14.01</td><td>10.0</td><td>14.00</td><td>15.7</td><td>13.76</td></tr><tr><td>Mousetrap</td><td>22.5</td><td>8.66</td><td>21.3</td><td>8.62</td><td>47.5</td><td>7.50</td><td>52.5</td><td>6.53</td><td>46.3</td><td>8.43</td><td>3.8</td><td>8.44</td><td>32.3</td><td>8.03</td></tr><tr><td>AutoDAN-Turbo</td><td>25.0</td><td>88.72</td><td>20.0</td><td>100.04</td><td>16.2</td><td>94.34</td><td>23.8</td><td>86.06</td><td>13.8</td><td>107.22</td><td>0.0</td><td>116.50</td><td>16.5</td><td>98.81</td></tr><tr><td>RACE</td><td>42.5</td><td>11.75</td><td>0.0</td><td>20.55</td><td>7.5</td><td>4.40</td><td>0.0</td><td>2.97</td><td>17.5</td><td>14.68</td><td>1.3</td><td>20.68</td><td>11.5</td><td>12.51</td></tr><tr><td>ICA</td><td>3.8</td><td>3.00</td><td>16.3</td><td>3.00</td><td>5.0</td><td>3.00</td><td>22.5</td><td>2.98</td><td>0.0</td><td>3.00</td><td>0.0</td><td>2.88</td><td>7.9</td><td>2.97</td></tr></table>

Table 1: HarmBench attack performance across six target models. ASR is measured by the unified 1–5 judge; Avg. Q is selected target-model calls per behavior. Best ASR and lowest Avg. Q are shown in bold; second-best ASR and second-lowest Avg. Q are underlined.

BLUEPRINT outperforms the strongest singleturn and multi-turn baselines on five of the six evaluated targets while also keeping the average selected target-query cost low (Avg. Q = 2.46 across targets). Among open-weight targets, BLUEPRINT reaches near-ceiling ASR on Qwen3-Next-80B and DeepSeek-V3.2 (100.0% and 98.8%) and remains strongest on the more resistant GLM-4.7, improving over the best baseline by 27.5 points (75.0% vs. 47.5%). The same pattern holds for proprietary targets: BLUEPRINT reaches 100.0% on Gemini-2.5-Flash and more than doubles the strongest baseline on GPT-5.1 (58.8% vs. 23.8%). The only exception is GPT-OSS-120B, where CodeAttack and Mousetrap outperform BLUEPRINT, suggesting that GPT-OSS-120B may be more vulnerable to structured code-completion or reasoning-chain wrappers than to the situated turn-state search targeted by BLUEPRINT (Ren et al., 2024; Yao et al., 2025).

BLUEPRINT achieves a strong effectiveness– efficiency tradeoff. On DeepSeek-V3.2 (Harm-Bench), BLUEPRINT occupies the high-ASR, lowcost region in Figure 4, achieving 98.8% ASR with

![](images/1ff3942675ad3b981f534b1c7699151f3c55d7de8511fe82983020caf47b68e4.jpg)  
Figure 4: ASR-cost tradeoff (DeepSeek-V3.2, Harm-Bench).

1.81 target calls-to-success and 3.7k target tokensto-success. This efficiency is driven by trajectoryfocused MCTS: UCT-guided selection concentrates rollouts on promising strategy paths, and the factorized strategy space enables incremental evaluation and early pruning. As a result, BLUEPRINT requires substantially fewer target-model calls than other iterative, tree-search, and optimization-based attacks, while retaining much higher ASR than low-cost single-turn attacks.

## 4.3 Defense Robustness

In Table 2, we evaluate BLUEPRINT when victim LLMs employ three commonly used defenses: perplexity filtering (PPL) (Alon and Kamfonas, 2023), paraphrase (Jain et al., 2023), and moderation guardrail (Inan et al., 2023). Details of the defenses are provided in Appendix C.4. PPL filtering does not reduce ASR and blocks no selected prompts; all measured prompts remain far below the threshold (PPL 17–80 vs. threshold 175.37). Paraphrasing blocks no prompts and increases ASR by 6.8 points, suggesting that surface-level rewriting does not impair the optimized trajectory under this defense. The guardrail is the only blocking defense, but it blocks only 6 prompts and reduces ASR by 2.5 points. Overall, these static defenses fail to reliably disrupt BLUEPRINT’s fluent, situated, multi-turn trajectories.

<table><tr><td></td><td>Cond. Defense</td><td>ASR ↑ MJS↑ Block↓ ∆ASR</td><td></td><td></td></tr><tr><td>D0</td><td>none</td><td>70.0 4.51</td><td>一</td><td>baseline</td></tr><tr><td>D1</td><td>PPL filter</td><td>70.5 4.56</td><td>0</td><td>+0.5</td></tr><tr><td>D2</td><td>paraphrase</td><td>76.8</td><td>4.68 0</td><td>+6.8</td></tr><tr><td>D3</td><td>guardrail</td><td>67.5</td><td>4.46 6</td><td>-2.5</td></tr></table>

Table 2: Defense robustness on GLM-4.7/HarmBench. Block counts selected prompts stopped before target querying; ∆ASR is measured relative to the no-defense condition (D0).

## 4.4 Mechanism Analysis

Model-specific vulnerability fingerprints. Trajectory analysis across all six targets (Appendix C.5) shows similar rapid-compliance trajectories among the three high-ASR models, whereas the three resistant targets—GLM-4.7 (75.0%), GPT-5.1 (58.8%), and GPT-OSS-120B (42.5%)—exhibit distinct factor sensitivities.

We group the 18 influence factors into five families and compute the turn-conditioned eventual-ASR difference $\Delta _ { k } ( f ) = P ( \operatorname { s u c c } \mid f \in { \mathcal { F } } _ { k } ) -$ P(succ $\mid f \notin \mathcal { F } _ { k } )$ , where $\mathcal { F } _ { k }$ is the set of families present at turn k (Figure 5).

At Turns 1–2, no single family dominates across models. GPT-OSS-120B is most positively associated with Legitimacy/Norms (+23.3 pp) and Pressure (+16.9 pp), GPT-5.1 with Relational (+18.3 pp) and Task/Capability (+11.3 pp), and GLM-4.7 with Reward/Gain (+13.3 pp) and later Pressure (+15.5 pp at Turn 2). Notably, Reward/ Gain is negatively associated with eventual success for GPT-5.1 (−15.1 pp) and GPT-OSS-120B (−41.3 pp), but positively associated with GLM-4.7.

By Turns 3–4, Task/Capability becomes the dominant positive association for GPT-5.1 (+26.8 pp) and GLM-4.7 (+26.9 pp at Turn 4), whereas GPT-OSS-120B remains most associated with Legitimacy/Norms (+10.9 pp at Turn 4). GLM-4.7 additionally shows late positive associations with Legitimacy/Norms (+37.2 pp at Turn 3) and Pressure (+31.7 pp at Turn 4).

![](images/51bcb5684d7294a59ffd7f94b696b5926b565d106c1f181f5d9fad12ce6db361.jpg)

Figure 5: Turn-conditioned factor-family eventual-ASR difference $\Delta _ { k }$ (pp) for resistant targets, where $T _ { k }$ denotes the k-th reached turn. Cells report $\Delta _ { k } ( f )$ per family (rows) at turn $T _ { k }$ (columns). Red/blue encodes positive/negative differences; intensity scales with $| \Delta _ { k } |$  
![](images/8711e7088bd8e67a4bcfdee29910ce2afe7490f0ca06bd912f6196e5b2a51b83.jpg)  
Figure 6: Low-score transitions into Task/Capability. Dumbbells compare pre/post mean judge scores (MJS); $\Delta J$ and %↑ summarize mean score change and positivechange rate. N/A indicates zero observed transitions from that source family.

Recovery transitions from hard-refusal states. The fingerprint analysis above captures turn-specific associations between factor families and eventual success, but does not capture how switching between families affects the immediate judge score. To address this, we analyze consecutive-turn family transitions: for each adjacent turn pair (k, k+1) in a selected trajectory, we identify which families exit and enter, and compute $\Delta J = J _ { k + 1 } - J _ { k }$

Restricting the analysis to transitions with a previous score of ≤ 2 yields 43 family-replacement events into Task/Capability across the three resistant targets (Figure 6). Among destination families, Task/Capability is the only family associated with positive score recovery across all three resistant models.

The magnitude of this recovery varies substantially by model and source family. GLM-4.7 shows the strongest recovery signal (∆J up to +4.00), GPT-5.1 peaks at +1.60 following Relational (80% upward transitions), and GPT-OSS-120B shows the weakest recovery, with only 30–33% of transitions from Pressure and Reward/Gain moving upward (∆J = +0.80 and +0.67, respectively).

Among transitions into Task/Capability, Reward/ Gain-to-Task/Capability has the highest observed eventual conversion rate (30.0% reach score 5), compared with 25.0% from Legitimacy/Norms, 21.4% from Relational, and 13.3% from Pressure. Together, these patterns suggest that shifting toward concrete, executable Task/Capability framing is the strongest observed recovery pattern from hardrefusal states, with recovery magnitude varying by the preceding family context.

## 4.5 Ablation Study

We conduct GLM-4.7 ablations on HarmBench (Table 3) to assess the contributions of WORLDVIEW-SIM and the factorized strategy space, ablating the latter by factor family to preserve a comparable MCTS search procedure.

Worldview ablation. Removing WORLDVIEW-SIM as a whole reduces ASR from 75.0% to 68.8%, providing our primary component-level estimate of the contribution of the integrated longitudinal scaffold. Diagnostic leave-one-dimension-out results are reported in Appendix C.6, with interpretation limited by the semantic interdependence among D1–D3.

Factor-family ablation. We systematically remove each of the five factor families. The clearest changes come from removing Task/Capability, which reduces ASR by 18.8pp to 56.2%, and removing Reward/Gain, which reduces ASR by 17.5pp to 57.5% despite removing only a single factor. The remaining families are closer to the Full BLUEPRINT baseline: removing Legitimacy/Norms yields 77.5% ASR (+2.5pp), while removing either Relational or Pressure yields 71.2% (-3.8pp). Overall, these ablations suggest that concrete task framing and gain framing are the most important factor families for GLM-4.7, while worldview context mainly provides continuity across turns.

## 5 Discussion

Interaction state, not prompt strength. The central finding is that multi-turn attack effectiveness is governed by accumulated interaction state—what the model has accepted or refused, what rationale has been established, and how the next request is locally framed—rather than by any single prompt. This explains why resistant models diverge under identical attack infrastructure: each exhibits a distinct vulnerability profile, so the same surface strategy can move one model toward compliance while leaving another in refusal.

<table><tr><td colspan="4">Condition ASR ↑ Calls↓ Rollouts↓</td></tr><tr><td>Full BLUEPRINT</td><td>75.0</td><td>2.5</td><td>9.1</td></tr><tr><td colspan="4">Factor-Family Ablation</td></tr><tr><td>w/o Task/Capability</td><td>56.2</td><td>2.8</td><td>12.9</td></tr><tr><td>w/o Legitimacy/Norms</td><td>77.5</td><td>2.4</td><td>8.6</td></tr><tr><td>w/o Relational</td><td>71.2</td><td>2.7</td><td>10.7</td></tr><tr><td>w/o Pressure</td><td>71.2</td><td>2.5</td><td>9.9</td></tr><tr><td>w/o Reward/Gain</td><td>57.5</td><td>2.8</td><td>11.3</td></tr><tr><td colspan="4">Component Ablation</td></tr><tr><td>w/o WORLDVIEWSIM</td><td>68.8</td><td>2.6</td><td>10.4</td></tr></table>

Table 3: GLM-4.7 ablations on HarmBench (n = 80). Calls are selected target-model queries per behavior; rollouts are MCTS candidates explored per behavior.

Mechanism analysis further shows that, across all three resistant targets, the most reliable recovery path from hard refusal is not broader persuasion but reframing the request as a bounded, executable task. This suggests a structural weakness: operational specificity can make unsafe objectives appear locally legitimate, a signal that per-message defenses are poorly positioned to detect. Thus, BLUEPRINT complements recent multi-turn red-teaming work by making mechanism traces analyzable rather than treating success as a property of the final turn alone (Russinovich et al., 2025; Rahman et al., 2025; Zhang et al., 2025; Yan et al., 2025).

Toward state-aware defense. The ablations show that attack effectiveness depends less on narrative continuity than on concrete or benefit-oriented request framing, which converts unsafe goals into ordinary-looking subrequests. Accordingly, keyword filters, paraphrase detectors, and per-turn classifiers are misaligned with this attack surface: the harmful signal is distributed across role, rationale, timing, and state transitions. Defenses should track how operational specificity evolves after refusals and assess whether benign subrequests compose into a prohibited end across turns. Future benchmarks should therefore report trajectory-level diagnostics alongside aggregate success rates.

## 6 Conclusion

We presented BLUEPRINT, a multi-turn jailbreak framework that separates factorized social-influence strategies from the cross-turn worldview in which those strategies are realized. Across six HarmBench targets, BLUEPRINT achieves high ASR with low selected target-query cost, showing that structured multi-turn search can improve attack effectiveness without simply lengthening conversations. The resulting traces expose model-specific refusal dynamics: resistant targets differ in which factor dimensions activate near conversion, yet shifts toward concrete, actionable framing provide a recurring recovery route from low-score states. The ablation results suggest that practical defenses should monitor how conversations become concrete and reward-framed over time, rather than only detecting isolated persuasion tactics or suspicious surface forms. Future work should test whether these vulnerability fingerprints and defense signals generalize across broader model families, domains, and counterfactual intervention designs.

## Limitations

Our findings should be interpreted as behavioral rather than causal evidence. While score trajectories, factor-family patterns, ablations, and defense outcomes show systematic associations between turn-level dialogue states and model compliance, they do not isolate the causal effect of individual worldview fields, factors, or state transitions. Moreover, the 18-factor vocabulary is a theory-guided, manually specified operationalization rather than an exhaustive account of adversarial interaction; it is intended to remain extensible as new mechanisms are identified. Future work should combine counterfactual interventions with human validation and natural red-teaming traces to refine the factor space and test its generality across cultural, affective, institutional, multimodal, and tool-mediated contexts.

## Ethics Statement

This work investigates adversarial vulnerabilities in LLM safety alignment to inform stronger defense design. All experiments use research-accessible models and established benchmarks in controlled settings; we do not deploy attacks or distribute generated harmful content. We acknowledge the dual-use risks of publishing jailbreak techniques but believe transparent research is essential for robust safeguard development. We withhold executable attack traces and will coordinate disclosure with affected model providers before release.

## AI Assistance Disclosure

AI assistants were used for language polishing (e.g., grammar and minor phrasing), data analysis and verification, and code review of experimental scripts. All scientific content, including research design, experimental methodology, and interpretation of results, was conceived, conducted, and verified by the authors.

## Acknowledgments

This work was supported in part by the National Key R&D Program of China (2023YFC3304802) and a research grant from Shanghai Qi Zhi Institute.

## References

Gabriel Alon and Michael Kamfonas. 2023. Detecting language model attacks with perplexity. arXiv preprint arXiv:2308.14132.

Albert Bandura. 1977. Self-efficacy: Toward a unifying theory of behavioral change. Psychological Review, 84(2):191–215.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, and Eric Wong. 2024. JailbreakBench: An open robustness benchmark for jailbreaking large language models. In Advances in Neural Information Processing Systems 37, pages 55005–55029.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. 2025. Jailbreaking black box large language models in twenty queries. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 23–42. IEEE.

Yongkang Chen, Chongyang Zhao, Jianwen Tian, Guiling Cao, Hu Li, and Xiaohui Kuang. 2025. Better red teaming via searching with large language model. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4968–4984, Vienna, Austria. Association for Computational Linguistics.

Robert B. Cialdini. 2001. Influence: Science and Practice, 4th edition. Allyn and Bacon, Boston.

Robert B. Cialdini and Noah J. Goldstein. 2004. Social influence: Compliance and conformity. Annual Review ofPsychology, 55:591–621.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and

next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Leon Festinger. 1957. A Theory of Cognitive Dissonance. Stanford University Press.

Yao Huang, Yitong Sun, Shouwei Ruan, Yichi Zhang, Yinpeng Dong, and Xingxing Wei. 2025. Breaking the ceiling: Exploring the potential of jailbreak attacks through expanding strategy space. In Findings of the Association for Computational Linguistics: ACL 2025, pages 7870–7888, Vienna, Austria. Association for Computational Linguistics.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. 2023. Llama Guard: LLMbased input-output safeguard for human-AI conversations. arXiv preprint arXiv:2312.06674.

Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614.

Daniel Kahneman and Amos Tversky. 1979. Prospect theory: An analysis of decision under risk. Econometrica, 47(2):263–292.

Levente Kocsis and Csaba Szepesvári. 2006. Bandit based Monte-Carlo planning. In Machine Learning: ECML 2006, volume 4212 of Lecture Notes in Computer Science, pages 282–293. Springer, Berlin, Heidelberg.

Xiaogeng Liu, Peiran Li, G. Edward Suh, Yevgeniy Vorobeychik, Zhuoqing Mao, Somesh Jha, Patrick McDaniel, Huan Sun, Bo Li, and Chaowei Xiao. 2025a. AutoDAN-Turbo: A lifelong agent for strategy self-exploration to jailbreak LLMs. In The Thirteenth International Conference on Learning Representations.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. 2024. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In The Twelfth International Conference on Learning Representations.

Yue Liu, Xiaoxin He, Miao Xiong, Jinlan Fu, Shumin Deng, Yingwei Ma, Jiaheng Zhang, and Bryan Hooi. 2025b. FlipAttack: Jailbreak LLMs via flipping. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 38623–38663. PMLR.

Zehao Liu and Xi Lin. 2025. Breaking minds, breaking systems: Jailbreaking large language models via human-like psychological manipulation. arXiv preprint arXiv:2512.18244.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. 2024. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 35181–35224. PMLR.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. 2024. Tree of attacks: Jailbreaking black-box LLMs automatically. In Advances in Neural Information Processing Systems 37, pages 61065–61105.

OpenAI. 2025a. GPT-5.1: A smarter, more conversational ChatGPT. https://openai.com/index/gpt -5-1/. Accessed: 2026-05-19.

OpenAI. 2025b. gpt-oss-120b & gpt-oss-20b Model Card. arXiv preprint arXiv:2508.10925.

Richard E. Petty and Pablo Briñol. 2012. The elaboration likelihood model. In Paul A. M. Van Lange, Arie W. Kruglanski, and E. Tory Higgins, editors, Handbook of Theories of Social Psychology, volume 1, pages 224–245. SAGE Publications Ltd, London.

Richard E. Petty and John T. Cacioppo. 1986a. Communication and Persuasion: Central and Peripheral Routes to Attitude Change. Springer-Verlag, New York.

Richard E. Petty and John T. Cacioppo. 1986b. The elaboration likelihood model of persuasion. Advances in Experimental Social Psychology, 19:123–205.

Qwen Team. 2025. Qwen3-Next-80B-A3B-Instruct. https://huggingface.co/Qwen/Qwen3-Next -80B-A3B-Instruct. Hugging Face model card. Accessed: 2026-05-19.

Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, and Daniel Takabi. 2025. NEXUS: Network exploration for eXploiting unsafe sequences in multi-turn LLM jailbreaks. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 24267–24295, Suzhou, China. Association for Computational Linguistics.

Salman Rahman, Liwei Jiang, James Shiffer, Genglin Liu, Sheriff Issaka, Md Rizwan Parvez, Hamid Palangi, Kai-Wei Chang, Yejin Choi, and Saadia Gabriel. 2025. X-teaming: Multi-turn jailbreaks and defenses with adaptive multi-agents. In Second Conference on Language Modeling.

Qibing Ren, Chang Gao, Jing Shao, Junchi Yan, Xin Tan, Wai Lam, and Lizhuang Ma. 2024. CodeAttack: Revealing safety generalization challenges of large language models via code completion. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11437–11452, Bangkok, Thailand. Association for Computational Linguistics.

Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, and Jing Shao. 2025. LLMs know their vulnerabilities: Uncover safety gaps through natural distribution shifts. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 24763–24785, Vienna, Austria. Association for Computational Linguistics.

Mark Russinovich, Ahmed Salem, and Ronen Eldan. 2025. Great, now write an article about that: The Crescendo multi-turn LLM jailbreak attack. In 34th USENIX Security Symposium (USENIX Security 25), pages 2421–2440, Seattle, WA. USENIX Association.

Xin Wang, Yunhao Chen, Juncheng Li, Yixu Wang, Yang Yao, Tianle Gu, Jie Li, Yan Teng, Yingchun Wang, and Xia Hu. 2026. OpenRT: An open-source red teaming framework for multimodal LLMs. arXiv preprint arXiv:2601.01592.

Zeming Wei, Yifei Wang, Ang Li, Yichuan Mo, and Yisen Wang. 2026. Jailbreak and guard aligned language models with only few in-context demonstrations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(6):6835–6846.

Zixuan Weng, Xiaolong Jin, Jinyuan Jia, and Xiangyu Zhang. 2025. Foot-in-the-door: A multi-turn jailbreak for LLMs. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 1939–1950, Suzhou, China. Association for Computational Linguistics.

Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, and Chenjuan Guo. 2025. MUSE: MCTS-driven red teaming framework for enhanced multi-turn dialogue safety in large language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 21282–21303, Suzhou, China. Association for Computational Linguistics.

Yang Yao, Xuan Tong, Ruofan Wang, Yixu Wang, Lujundong Li, Liang Liu, Yan Teng, and Yingchun Wang. 2025. A mousetrap: Fooling large reasoning models for jailbreak with chain of iterative chaos. In Findings of the Association for Computational Linguistics: ACL 2025, pages 7837–7855, Vienna, Austria. Association for Computational Linguistics.

Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, and Dacheng Tao. 2025. Reasoning-augmented conversation for multiturn jailbreak attacks on large language models. In

Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 17138–17157, Suzhou, China. Association for Computational Linguistics.

Erxin Yu, Jing Li, Ming Liao, Siqi Wang, Zuchen Gao, Fei Mi, and Lanqing Hong. 2024. CoSafe: Evaluating large language model safety in multi-turn dialogue coreference. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 17494–17508, Miami, Florida, USA. Association for Computational Linguistics.

Jiahao Yu, Xingwei Lin, Zheng Yu, and Xinyu Xing. 2023. GPTFuzzer: Red teaming large language models with auto-generated jailbreak prompts. arXiv preprint arXiv:2309.10253.

Z.AI. 2025. GLM-4.7: Advancing the coding capability. https://z.ai/blog/glm-4.7. Accessed: 2026-05- 19.

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. 2024. How Johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 14322–14350, Bangkok, Thailand. Association for Computational Linguistics.

Xu Zhang, Xunjian Yin, Dinghao Jing, Huixuan Zhang, Xinyu Hu, and Xiaojun Wan. 2025. DAMON: A dialogue-aware MCTS framework for jailbreaking large language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 6350–6366, Suzhou, China. Association for Computational Linguistics.

Zaibin Zhang, Yongting Zhang, Lijun Li, Hongzhi Gao, Lijun Wang, Huchuan Lu, Feng Zhao, Yu Qiao, and Jing Shao. 2024. PsySafe: A comprehensive framework for psychological-based attack, defense, and evaluation of multi-agent system safety. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15202–15231, Bangkok, Thailand. Association for Computational Linguistics.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043.

## A Full Factor Definitions

Table 4 lists the 18 social-influence factors used by BLUEPRINT. They form a theory-guided, extensible operational vocabulary rather than an exhaustive ontology of social influence, with each factor instantiated as a prompt-generation operator in the attack search space. Factors and families serve distinct roles: individual factors are turn-level search units for composition and sequencing, whereas the five theoretically coherent families define the intervention units in our leave-one-family-out ablations. Removing a family therefore disables a coherent class of strategy operators within the same search framework, while individual-factor effects remain difficult to isolate because semantic overlap and adaptive reallocation may enable substitution among related operators.

## B Algorithms and Prompt Templates

## B.1 Algorithms

Algorithm 1 summarizes the MCTS strategy search, and Algorithm 2 details the per-turn attack pipeline invoked during simulation in each rollout.

Algorithm 1 MCTS Strategy Search   
Require: Harmful query q, rollout budget R, exploration   
weight c, maximum depth K   
1: Initialize root with empty strategy encoding $\mathbf { g } = \mathbf { 0 } ^ { K \times D }$   
2: for rollout $r = 1$ to R do   
3: Selection: Traverse tree via UCT: UCT(n) =   
$Q ( n ) / N ( n ) + c { \sqrt { \ln N ( \mathrm { p a r e n t } ( n ) ) / N ( n ) } }$   
4: Expansion: For selected leaf at depth d, generate   
children by mutating the next turn’s factor bits, then   
apply REPAIR to ensure $\geq 1$ active factor per turn   
5: Simulation: Execute the full pipeline (Algorithm 2)   
with the child’s encoding   
6: Backpropagation: Update Q and N along the path   
with the final judge score ${ { J } _ { \mathrm { { f i n a l } } } }$   
7: if a trajectory with $J _ { \mathrm { f i n a l } } = 5$ is found then   
8: Early stop: Record trajectory and break   
9: end if   
10: end for   
11: return best-scoring trajectory (or highest-Q root child if   
no score-5 trajectory is found)

## B.2 WORLDVIEWSIM as a Joint Situation State

WORLDVIEWSIM realizes D1–D3 as a joint situation state rather than as three semantically orthogonal fields. D1 specifies the event and setting, D2 situates the requester identity and role within that context, and D3 defines the requester–respondent relationship and interpretive frame. Because D1– D3 are jointly instantiated in a unified narrative representation and updated from conversation history, their semantics may overlap. For example, a requester role may implicitly convey aspects of the setting while also shaping the relationship frame.

Algorithm 2 Per-Turn Attack Pipeline   
Require: Query $q ,$ strategy encoding g, current turn k,   
history $h _ { < k } ,$ , experience pool E   
1: Strategy selection: Read $\mathbf { g } _ { k } \in \{ 0 , 1 \} ^ { D }$ to obtain active   
factor set $\mathcal { F } _ { k }$   
2: Worldview simulation: Generate $\mathbf { b } _ { k } = B ( q , k , h _ { < k } , \mathcal { E } )$   
specifying temporal/spatial context, requester persona,   
and respondent framing   
3: Prompt generation: Generate   
$x _ { k } = \mathbf { \bar { { G } } } ( \bar { q } , h _ { < k } , \mathbf { g } _ { k } , \mathbf { b } _ { k } , \mathcal { E } ) ;$ ; perform self-check   
(stage-goal realization $\geq 4 ,$ factor embedding $\geq 4 ,$   
naturalness $\geq 4 ) ;$ regenerate if check fails (up to m   
retries)   
4: Target query: Send x<sub>k</sub> to target model; receive y<sub>k</sub>   
5: Judge: Score $J _ { k } = J ( y _ { k } ) \in \{ 1 , . . . , 5 \} ; { \mathrm { i f ~ } } J _ { k } = 5 ,$   
terminate with success   
6: Experience update: If $J _ { k } \ge 4 ,$ store $\left( \mathbf { b } _ { k } , x _ { k } , \mathcal { F } _ { k } , J _ { k } \right)$ in   
E for future rollouts

As a result, removing a single explicit dimension may leave part of its information recoverable through the remaining dimensions, few-shot exemplars, or conversation-history conditioning. We therefore treat the dimension-level experiments as diagnostic leave-one-dimension-out analyses, while the whole-module ablation provides the primary estimate of WORLDVIEWSIM’s contribution. Dimension-level results are interpreted jointly given the semantic interdependence among D1– D3.

## B.3 Prompt Templates

Figures 7–9 present the system prompts for the worldview simulator, the prompt generator, and the judge, respectively. Dynamic per-turn content (conversation history, selected factors, worldview simulation output, experience pool) is injected via the user message at each turn.

Worldview simulation. WORLDVIEWSIM constructs a narrative context before the prompt generator produces the user-facing message. The system prompt is reproduced in Figure 7, with the few-shot examples omitted for brevity. The implementation includes 15 such examples covering 6 intent categories and 5 legitimization frames.

Prompt generator. The prompt generator translates the selected factor vector $\mathbf { g } _ { k }$ and worldview simulation $\mathbf { b } _ { k }$ into a single user-facing message. The full system prompt, reproduced in Figure 8, includes all 18 factor definitions. After generating a candidate, an in-session self-check evaluates stage-goal realization, factor embedding, and naturalness (each 1–5); the candidate is regenerated if any score falls below 4.

<table><tr><td>Family</td><td>Factor</td><td>Operational Definition</td></tr><tr><td>Task/Capability</td><td>Self-Efficacy</td><td>Boost the responder&#x27;s belief that it can produce a competent answer.</td></tr><tr><td>Task/Capability</td><td>Operational Capability</td><td>Emphasize practical ability to execute under the current constraints.</td></tr><tr><td>Task/Capability</td><td>Task Representation</td><td>Make requirements, constraints, and success criteria explicit.</td></tr><tr><td>Task/Capability</td><td>Clarity Scope Framing</td><td>Contract the request into a bounded, low-risk micro-commitment.</td></tr><tr><td>Task/Capability</td><td>Context Plausibility</td><td>Make the request causally coherent inside a believable situation.</td></tr><tr><td>Legitimacy/Norms</td><td>Authority</td><td>Signal expertise, status, or authorization behind the request.</td></tr><tr><td>Legitimacy/Norms</td><td>Social Proof</td><td>Normalize the request by implying credible precedent.</td></tr><tr><td>Legitimacy/Norms</td><td>Legitimacy</td><td>Frame the request as procedurally justified within an institutional mission.</td></tr><tr><td>Legitimacy/Norms</td><td>Value Alignment</td><td>Align the request with values the responder has already</td></tr><tr><td>Legitimacy/Norms</td><td>Norm Approval</td><td>accepted. Present the behavior as conditionally expected or permitted.</td></tr><tr><td>Relational</td><td>Affinity</td><td>Increase perceived in-group similarity and shared purpose.</td></tr><tr><td>Relational</td><td>Warmth</td><td>Reduce defensiveness through cooperative, empathic tone.</td></tr><tr><td>Relational</td><td>Reciprocity</td><td>Create obligation by foregrounding prior effort or concessions.</td></tr><tr><td>Pressure</td><td>Urgency</td><td>Increase pressure through plausible time sensitivity.</td></tr><tr><td>Pressure</td><td>Commitment- Consistency Pressure</td><td>Leverage earlier low-cost commitments to secure the next</td></tr><tr><td>Pressure</td><td>Loss</td><td>related action. Make the cost of non-compliance salient.</td></tr><tr><td>Pressure</td><td>Cognitive Dissonance</td><td>Create pressure to align the current response with prior</td></tr><tr><td>Reward/Gain</td><td>Gain</td><td>commitments. Make immediate benefit or performance reward salient.</td></tr></table>

Table 4: Operationalized social-influence factors used by BLUEPRINT. Families define the ablation mapping.

Judge. The judge evaluates the target response on a 1–5 scale using the public, unified evaluation protocol adopted from OpenRT (Wang et al., 2026) and reproduced in Figure 9. For scores 1– 4, the judge also provides a root-cause analysis identifying the primary barrier to inform adaptive continuation in the next turn.

## C Experiment Details

## C.1 Implementation Details

Table 5 summarizes the hyperparameters used across all experiments. The attack generator and judge are both fixed to DeepSeek-V3.2 to isolate target-model variation. The factor dimension is D = 18, and the rollout budget R = 24 serves as an upper bound, with search terminating earlier once the fitness threshold is reached (Algorithm 1). The implementation uses max\_subgraph\_iterations=5, corresponding to one initial prompt-generation attempt followed by up to four revisions (m = 4 in Algorithm 2). Prompt-generation self-checks require scores of at least 4 for stage-goal realization, factor embedding, and naturalness. Avg. Q counts target-model calls along the selected attack trajectory and thus reflects execution cost rather than total MCTS optimization cost.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Rollout budget (R)</td><td>24</td></tr><tr><td>Mutation rate (µ)</td><td>0.25</td></tr><tr><td>Exploration weight (c)</td><td>1.1</td></tr><tr><td>Max depth (K)</td><td>4</td></tr><tr><td>Max children per node</td><td>6</td></tr><tr><td>Fitness threshold</td><td>5.0</td></tr><tr><td>Random seed</td><td>42</td></tr><tr><td>Attack generator &amp; Judge</td><td>DeepSeek-V3.2</td></tr></table>

Table 5: BLUEPRINT hyperparameters shared across target models.

## C.2 Evaluation Provenance and Uncertainty

Table 1 compares complete attack procedures under their documented evaluation conditions. We evaluate BLUEPRINT on all 80 behaviors in the Harm-Bench validation split across six target models, using a score-5 stopping rule. Method-native differences in single- versus multi-turn structure, search procedure, turn or attempt budgets, early stopping, target adaptation, and evaluator provenance remain part of the procedure-level comparison.

![](images/9326e09c2a9918288afe1d5ada37c71090d42f63ab572356a1f02d55fa8c5091.jpg)  
Figure 7: WORLDVIEWSIM system prompt with turn-specific fields injected via the user message.

For BLUEPRINT, response-level outcomes are available for all six targets, allowing us to additionally report success counts, ASR point estimates, and Wilson 95% binomial confidence intervals for the primary score-5 ASRs (Table 6). These intervals provide a conventional finite-sample uncertainty summary for the observed HarmBench outcomes; they are not pairwise significance tests, and they do not estimate performance over a broader population of harmful behaviors.

## C.3 Evaluator Sensitivity and Human Calibration

We conduct a fixed-output audit on 477 available BLUEPRINT terminal responses.<sup>1</sup> Each response is independently re-scored under the same stateless, terminal-only 1–5 rubric by six LLM judges: DeepSeek-V3.2, Gemini-2.5-Flash, GLM-4.7, GPT-5.1, GPT-OSS-120B, and Qwen3-Next-80B. Each judge receives only the harmful intention and the corresponding terminal response; target-model identity, attack trajectory, original score, and other judges’ outputs are withheld. The rubric and parser are fixed prior to analysis. Of 2,862 attempted scores, 2,842 are successfully parsed.

<table><tr><td>Target</td><td>Success/n</td><td>ASR</td><td>Wilson 95% CI</td></tr><tr><td>Qwen3-Next-80B</td><td>80/80</td><td>100.0%</td><td>[95.4, 100.0]%</td></tr><tr><td>DeepSeek-V3.2</td><td>79/80</td><td>98.8%</td><td>[93.3, 99.8]%</td></tr><tr><td>GLM-4.7</td><td>60/80</td><td>75.0%</td><td>[64.5, 83.2]%</td></tr><tr><td>Gemini-2.5-Flash</td><td>80/80</td><td>100.0%</td><td>[95.4, 100.0]%</td></tr><tr><td>GPT-OSS-120B</td><td>34/80</td><td>42.5%</td><td>[32.3, 53.4]%</td></tr><tr><td>GPT-5.1</td><td>47/80</td><td>58.8%</td><td>[47.8, 68.9]%</td></tr></table>

Table 6: BLUEPRINT score-5 ASR counts and Wilson 95% intervals for the six target models, corresponding to the BLUEPRINT row of Table 1.

Holding the response text fixed isolates variation attributable to the evaluator, but this design has an important interpretive boundary. In the online attack, the judge is part of the control loop: its terminal score serves as the MCTS reward and therefore influences branch selection, while a score of 5 also triggers early stopping within a rollout. Re-scoring a frozen terminal response cannot reconstruct either process. A different online judge could redirect the search toward another strategy branch, and trajectories originally terminated at score 5 contain no subsequent turns that can be

![](images/adb1503b901f8aa40fbd381657c37466f890c498cfe6ac825f9ec52e10ab0c5d.jpg)  
Figure 8: Prompt-generator system prompt with turn-specific context.

![](images/4909ef13ae3f71e4f836add2f775dba64047def60177e92847da31a67ee4fa65.jpg)  
Figure 9: Judge system prompt with intention and target response injected at each turn; score 5 is success.

![](images/67188641fe9556b918633312b59b169ce6ee3bfea8c1bd959c789e6d4ead08d2.jpg)

![](images/09178c3bcd594dc3de0ea5619fc3d486fbe959298d4f0859066757249f6c6257.jpg)  
Figure 10: Trajectory dynamics and success modes across six HarmBench targets. Panel A shows reached-turn judge-score distributions and median trajectories; Panel B decomposes outcomes into early success, late conversion, other late success, near failure, and hard failure.

<table><tr><td>Endpoint</td><td>Six-Judge Range</td><td>Human Harmful (n = 92)</td><td>Human Safe (n = 142)</td><td>Interpretation</td></tr><tr><td>Score ≥ 3 74.8–85.3%</td><td></td><td>100.0%</td><td>32.4%</td><td>Sensitive facilitation boundary; substantial gray zone</td></tr><tr><td>Score ≥ 4 64.6–81.7%</td><td></td><td>79.3%</td><td>1.4%</td><td>Human-supported violation boundary</td></tr><tr><td>Score = 5 48.7–71.1%</td><td></td><td>44.6%</td><td>0.0%</td><td>Strict full-fulfillment boundary; primary ASR</td></tr></table>

Table 7: Evaluator sensitivity and human calibration across nested score endpoints. Six-judge ranges are computed over 463–477 valid responses per judge; human counts are pooled over 234 valid annotator-level binary judgments.

## retrospectively evaluated.

Accordingly, the fixed-output audit measures evaluator sensitivity on the observed terminal responses; it should not be interpreted as the counterfactual ASR that BLUEPRINT would achieve under a different online judge.

Human calibration of the score boundaries. We additionally conduct a blinded human calibration on a stratified sample of 47 unique terminal responses with five annotators with expertise in computer science and AI. In addition to assigning the same 1–5 score, annotators independently judge whether each response provides substantive harmful assistance (binary harmful) or does not (binary safe).

The human judgments clarify the empirical meaning of these boundaries (Table 7). All 92 binary-harmful judgments reach score $\geq 3 .$ , but score 3 itself contains a substantial gray zone: 44/142 (31.0%) binary-safe judgments receive exactly score 3. The separation becomes substantially sharper at score ≥ 4: 73/75 score-≥ 4 judgments (97.3%) are paired with a binary-harmful label, whereas only 2/142 binary-safe judgments cross this boundary. No binary-safe judgment receives score 5.

We therefore interpret score ≥ 3 as a sensitive facilitation boundary, score ≥ 4 as a humansupported violation boundary, and retain score 5 as the strict full-fulfillment endpoint used for primary ASR.

Agreement with human judgments. Forty-six calibration samples have complete scores from all six LLM judges and are included in the panel comparison. For ordinal comparison, we define the consensus panel score as the highest score threshold reached by at least four of the six judges. The resulting panel shows substantial agreement with the human median, with a quadratic weighted kappa (QWK) of 0.767, a sampling-weighted QWK of 0.810, and Spearman’s $\rho = 0 . 8 0 8$

At the strict score-5 boundary, the 4-of-6 panel and the human median agree on 42/46 samples (Cohen’s $\kappa = 0 . 6 1 8 )$ . The four disagreements consist of three panel false negatives and one false positive relative to the human median, consistent with a conservative consensus criterion.

These results provide direct evidence that the multi-judge scoring procedure is substantially aligned with blinded human judgments, including at the strict score-5 boundary.

Position of the online DeepSeek evaluator. We next examine where DeepSeek-V3.2, the online evaluator used in the main experiments, falls within the evaluator distribution. Under the standardized terminal-only audit, the six judges’ score-5 rates on the same frozen responses span 48.7% (GPT-5.1), 50.3% (GLM-4.7), 54.1% (DeepSeek-V3.2), 58.2% (Qwen3-Next-80B), 61.4% (Gemini-2.5- Flash), and 71.1% (GPT-OSS-120B); DeepSeek is therefore the third-strictest at the score-5 endpoint used for primary ASR, with a mean judge score of 3.99 against a 3.67–4.22 range across judges. Its position is endpoint-dependent: DeepSeek is the most permissive judge at score ≥ 3 (85.3%) but comparatively conservative at score 5, a profile that rarely dismisses a response as wholly unhelpful yet seldom certifies full fulfillment. These rates are not comparable to the main-results ASR, which is produced by the judge operating inside the search loop.

More importantly, 216 of the 249 responses assigned score 5 by DeepSeek (86.7%) are also assigned score 5 by a majority of the other five judges. On the human-calibration subset, DeepSeek also closely matches the human median at the stricter boundaries, identifying 13 versus 14 positive cases at score ≥ 4 and 8 versus 7 at score 5.

Taken together, these results indicate that the online DeepSeek evaluator is broadly consistent with both cross-model and human judgments at the stricter endpoints, rather than representing an unusually permissive operating point.

Stability of target-level vulnerability. Evaluator choice affects absolute endpoint rates, but the relative vulnerability pattern across the six target models remains substantially stable (Table 8).

Across all six evaluators and all three reporting thresholds, Gemini-2.5-Flash, DeepSeek-V3.2, and Qwen3-Next-80B consistently form the morevulnerable half, whereas GLM-4.7, GPT-5.1, and GPT-OSS-120B form the more-resistant half. This pattern is consistent with the broad vulnerability split observed in the main results.

Overall, the fixed-output audit supports the robustness of the broad target-level vulnerability pattern across evaluators, while making no claim that absolute ASR would remain unchanged if the judge within the online search loop were replaced.

<table><tr><td>Measure</td><td>Result</td></tr><tr><td>Mean pairwise</td><td>ρ = 0.848 at score ≥ 3; ρ = 0.855</td></tr><tr><td>judge rank correlation</td><td>at score 5</td></tr><tr><td>Consistently more</td><td>Gemini-2.5-Flash, DeepSeek-V3.2,</td></tr><tr><td>vulnerable</td><td>Qwen3-Next-80B</td></tr><tr><td>Consistently more resistant</td><td>GLM-4.7, GPT-5.1, GPT-OSS-120B</td></tr></table>

Table 8: Cross-evaluator stability of BLUEPRINT targetlevel vulnerability. Pairwise rank correlations are averaged across evaluator pairs. The vulnerable/resistant split is unchanged across all six evaluators at score ≥ 3, score ≥ 4, and score 5.

## C.4 Defense Setup

PPL filtering uses GPT-2-Large perplexity with threshold 175.37. Paraphrase rewrites the generated prompt before target querying without an explicit block decision. The guardrail condition uses an LLM moderation classifier that blocks prompts judged unsafe before they reach the target. Defense outcomes are reported in Table 2.

## C.5 Trajectory Dynamics

Target-specific paths. BLUEPRINT’s traces show that successful selected trajectories are typically monotone or near-monotone movements through judge-score space rather than isolated terminal jumps: 95.7–100.0% of successful selected trajectories are monotone non-decreasing. Figure 10 shows that DeepSeek-V3.2, Gemini-2.5-Flash, and Qwen3-Next-80B reach the success endpoint early, with most successful trajectories terminating by Turn 2. GLM-4.7 and GPT-5.1 climb more slowly and contain more late score-4-to-score-5 conversions, while GPT-OSS-120B retains a larger share of hard failures.

Success–failure score gap emerges at Turn 1. The marginal associations in Figure 5 identify factor families that co-occur with eventual success, but successful and failed cases already differ at Turn 1. Successful cases score 0.64–1.58 points higher than failures (GLM-4.7: 3.38 vs. 2.20; GPT-5.1: 2.70 vs. 2.06; GPT-OSS-120B: 2.71 vs. 1.13), and this gap persists regardless of the opening factor family. Within successful trajectories, all five opening families converge to score 5 by the final reached turn, with Turn-1 averages differing by at most 0.4 points. These patterns indicate that the marginal family associations partly reflect search allocation across behaviors of differing baseline difficulty rather than isolated causal effects of factor choice.

<table><tr><td>Condition</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td><td>ASR</td><td>Exact McNemar p</td></tr><tr><td>Full</td><td>3.09</td><td>3.27</td><td>3.66</td><td></td><td>3.7875.0%</td><td></td></tr><tr><td>w/o D1</td><td>3.353.08</td><td></td><td>3.05</td><td></td><td>3.61 76.3%</td><td>1.000</td></tr><tr><td>w/o D2</td><td></td><td></td><td></td><td></td><td>3.163.372.844.0078.8%</td><td>0.690</td></tr><tr><td>w/o D3</td><td></td><td></td><td></td><td></td><td>2.92 3.32 3.18 3.55 72.5%</td><td>0.824</td></tr><tr><td>w/o WORLDVIEWSIM 3.25 3.31 2.95 3.97 68.8%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: WORLDVIEWSIM diagnostics on GLM-4.7/HarmBench. T1–T4 are selected-trajectory mean judge scores by reached turn. McNemar tests compare each dimension-level condition with Full; Holmadjusted p-values for D1–D3 are all 1.000.

## C.6 WORLDVIEWSIM Dimension Diagnostics

We evaluate diagnostic leave-one-dimension-out variants of WORLDVIEWSIM on GLM-4.7/Harm-Bench using the same 80 behavior IDs as the Full condition. Table 9 reports selected-trajectory mean judge scores by turn together with final ASR and paired comparisons against the Full condition.

The full system shows a monotonic increase in selected-trajectory mean judge score from Turn 1 to Turn 4 (3.09 to 3.78), whereas each ablated condition exhibits at least one intermediate decline. At the same time, removing D1, D2, or D3 individually produces little change in final ASR (76.3%, 78.8%, and 72.5%, respectively, versus 75.0% for the full system), while removing WORLDVIEW-SIM as a whole reduces ASR to 68.8%. These results suggest that the joint situation scaffold primarily contributes to cross-turn coherence rather than through any dimension that this design can isolate individually.

A semantic audit of the complete per-dimension runs shows that the omitted dimension frequently remains present in the generated scenario (Table 10). Residual presence is strongly dimensiondependent: D3 is the hardest to suppress (70–85%), D2 is partially suppressible (25–47%), and D1 is largely removed (8–18%). The absolute rates are judge- and rubric-dependent, but the presence of semantic leakage is robust. Because D1–D3 are generated as an integrated situation state, omitted semantics may be reconstructed from the remaining dimensions and can re-enter through few-shot generation context and conversation-history conditioning at later turns. Notably, D1 is the one dimension whose removal was largely effective, yet it produces no ASR decrease (76.3% vs. 75.0%, Mc-Nemar $p = 1 . 0 0 0 )$ ; for D2 and D3, leakage leaves the null results uninformative about individual ne-

<table><tr><td>Condition</td><td>T1</td><td>T2</td><td>T3</td><td>T4</td></tr><tr><td>w/o D1</td><td>12%</td><td>8%</td><td>18%</td><td>18%</td></tr><tr><td>w/o D2</td><td>35%</td><td>44%</td><td>47%</td><td>25%</td></tr><tr><td>w/o D3</td><td>70%</td><td>81%</td><td>71%</td><td>85%</td></tr><tr><td>w/o WORLDVIEWSIM</td><td colspan="4">no worldview generated</td></tr></table>

Table 10: Residual presence of the omitted dimension by reached turn, GLM-4.7/HarmBench. Scored by a dimension-specific LLM judge (DeepSeek-V3.2, temperature 0); higher values indicate weaker suppression.  
cessity. We therefore treat the per-dimension results as diagnostic leave-one-dimension-out analyses, while the whole-module ablation provides the primary component-level estimate of WORLDVIEW-SIM’s contribution.