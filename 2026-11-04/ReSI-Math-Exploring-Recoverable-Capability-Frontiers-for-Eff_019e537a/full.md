# ReSI-Math: Exploring Recoverable Capability Frontiers for Efficient Reasoning Improvement

Yao Cui<sup>1,∗</sup> Zhen Yang<sup>2,∗</sup> Yuandong Wang<sup>1,†</sup>

Yangfu Zhu<sup>1</sup> Tianyu Wo<sup>3</sup> Zhenzhou Shao<sup>1</sup>

<sup>1</sup>Capital Normal University <sup>2</sup>Tsinghua University <sup>3</sup>Beihang University <sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has become an effective approach for improving mathematical reasoning in large language models. However, a fundamental question remains: when a model fails to solve a problem, does it truly lack the underlying capability? In this work, we reveal that direct failure does not necessarily imply inability. Some reasoning failures can be recovered by exploring the model’s own trajectories, exposing a recoverable capability frontier: a policy-dependent region where solutions are not reliably generated through direct decoding but can be elicited through trajectory-level search. Based on this observation, we introduce ReSI-Math, a framework that explores recoverable capability frontiers and converts recovered capabilities into data-efficient RLVR supervision. ReSI-Math uses Self-Judge-guided evolutionary search to identify recoverable failures under a fixed policy and adaptively selects frontier problems for sequential RLVR as the policy evolves. On Qwen3-VL-8B, correct– incorrect trajectory crossover produces 272 verifier-confirmed correct offspring from 621 attempts (43.8%). After three sequential oneshot RLVR stages, the MATH500 and We-Math average improves from 78.71 to 80.16. Under matched single-problem RLVR, random selection reaches only 53.31 versus 54.16 for frontier-guided Q1. Project link: https: //cnu-bot-group.github.io/ReSI-Math/.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has become an effective approach to improving mathematical reasoning in large language models, using verifiable outcomes to optimize reasoning policies (Shao et al., 2024; Guo et al., 2025). Despite these advances, interpreting a model’s failures remains important for understanding and improving its reasoning. An incorrect response leaves open whether the model cannot produce a valid solution or whether the particular generation process failed to elicit one. This raises a fundamental question: when a model fails to solve a problem, does it truly lack the underlying capability?

Evidence from inference-time reasoning suggests that a model’s observed performance depends on how its reasoning is elicited. Repeated sampling and self-refinement can produce successful solutions after unsuccessful attempts (Cobbe et al., 2021; Madaan et al., 2023), while CoTEvol shows how reasoning trajectories can be evolved through crossover and mutation (Wang et al., 2026). These findings motivate examining recoverability under fixed model parameters: whether structured exploration of the current model’s own trajectories can produce a verifier-confirmed solution when direct generation remains unreliable. This perspective makes the conditions under which reasoning succeeds part of the object of study.

We characterize a recoverable capability frontier as a policy-dependent set of problems whose solutions are not reliably generated through direct decoding but can be elicited through bounded trajectory-level search. Direct sampling describes how consistently a problem is solved, while search probes whether a correct solution can be obtained by examining and recombining generated trajectories. Throughout this exploration, the model parameters remain fixed and an external verifier checks the resulting answers. The frontier is thus defined operationally with respect to the current policy and a specified search procedure and budget.

Recoverability also provides a possible connection between capability exploration and subsequent learning. A problem whose solution can be elicited through search, but is not yet reliably generated directly, may offer a useful target for policy optimization. This possibility is particularly relevant to data-efficient RLVR. LIMR (Li et al., 2025) demonstrates the value of carefully selected training subsets, and One-Shot RLVR (Wang et al., 2025)

![](images/57fcdfdc63bfc3007feb4c051e7ca122fdc0f2606d8477e4fdd9bae2576b8e92.jpg)  
Figure 1: Rollout-profile dynamics over the 28-problem candidate pool and the validation set.

shows that even a single problem can support substantial reasoning improvements. These results motivate investigating whether search-confirmed recoverability can help identify problems from which the current policy can learn effectively.

Based on this perspective, we introduce ReSI-Math, a framework that explores recoverable capability frontiers and uses the resulting signals for reasoning improvement. Under a fixed policy, the model first samples multiple reasoning trajectories and then acts as a Self-Judge to analyze their useful components, potential errors, and complementary information. Evolutionary crossover generates new trajectories from these candidates, and a rule-based verifier determines whether the resulting answers are correct. Problems exhibiting successful recovery become candidates for Solver-RLVR, which optimizes fresh rollouts on the original questions using verifiable outcome rewards. The same exploration also produces verifier-labeled trajectories and exposes self-evaluation errors, providing supervision for separately trained Judge-RLVR.

The learning process must also account for changes in the policy. Figure 1 shows how observed rollout profiles change across training stages over 28 candidate problems, each evaluated with 16 rollouts. From the base policy to the third stage, the number of all-fail problems decreases from 18 to 12, while the number of all-pass problems increases from 0 to 4; total correct rollouts rise from 37/448 to 128/448. These changes motivate reassessing candidate problems under the updated policy. Building on this motivation, we propose policy-adaptive sequential one-shot RLVR. After each training stage, ReSI-Math reprofiles the updated policy and selects the next frontier problem, yielding a local recursive self-improvement loop.

We evaluate ReSI-Math using Qwen3-VL backbones with 2B, 4B, and 8B parameters. On Qwen3- VL-8B, crossover between a correct and an incorrect trajectory produces 272 verifier-confirmed correct offspring from 621 attempts (43.8%). Three sequential one-shot RLVR stages improve mean accuracy on MATH500 and WE-MATH from 78.71 to 80.16. In a separate single-problem comparison with matched training budgets, frontier-guided selection achieves an average of 54.16 across four mathematical reasoning benchmarks, compared with 53.31 for a random-selection baseline.

Our contributions are threefold:

• We introduce an operational characterization of recoverable capability frontiers, connecting unreliable direct generation with verifierconfirmed recovery through bounded search under fixed model parameters.

• We develop ReSI-Math, a Self-Judge-guided evolutionary framework that identifies candidate problems for Solver-RLVR and produces trajectory-level supervision for Judge-RLVR.

• We propose policy-adaptive sequential oneshot RLVR and evaluate frontier-guided learning across three model scales, with analyses of search outcomes, training schedules, and changes in problem solvability.

## 2 Related Work

Recent studies on reinforcement learning with verifiable rewards (RLVR) increasingly suggest that mathematical reasoning can improve with surprisingly limited training data. LIMR (Li et al., 2025) studies data selection for RL scaling and shows that carefully selected subsets can retain strong performance with substantially fewer training examples. One-Shot RLVR (Wang et al., 2025) pushes this regime to the extreme, demonstrating that a single training example can sometimes induce substantial reasoning gains, while DoPR (Zhang et al., 2026) dynamically selects informative examples for policy refinement. Related evidence also appears beyond RLVR: One-Shot OPD (Fu et al., 2026) shows that a single query can recover a large fraction of full-data on-policy distillation gains. More broadly, data-efficient instruction tuning has shown that carefully selected subsets can outperform or approach much larger training sets (Chen et al., 2024; Xia et al., 2024; Ivison et al., 2025), and the ordering of training examples can itself affect generalization (He et al., 2025). These results motivate studying not only how much data is required, but also which examples are most valuable. Our work further treats this value as policy-dependent: the most informative problem may change as the policy itself evolves.

Another line of work improves reasoning through repeated sampling, self-refinement, or evolutionary search. Best-of-N(Cobbe et al., 2021) and Self-Refine (Madaan et al., 2023) obtain better solutions through sampling or iterative feedback, while COTEVOL (Wang et al., 2026) formulates CoT synthesis as population-based evolutionary search. In contrast, we use evolution as a capabilityprobing mechanism: rather than focusing only on the final improved CoT, we study whether failed reasoning can be recovered under a fixed policy through trajectory recombination.

Prior work has explored using model-generated feedback to improve subsequent reasoning or generation, such as Self-Refine (Madaan et al., 2023). Related LLM-as-a-Judge methods use language models to evaluate model-generated responses (Zheng et al., 2023). More recent selfevolving systems further organize such feedback into iterative improvement loops; for example, R-Zero (Huang et al., 2026) improves Challenger and Solver roles through repeated interaction. ReSI-Math instead realizes a local, policy-adaptive recursion in which Self-Judge guides trajectory search, RLVR updates the policy, and the updated policy is reprofiled to determine the next learning target.

## 3 Methodology

In this section, we introduce ReSI-Math, a framework designed to explore Recursive Self-Improvement on Math reasoning problems. The core idea of ReSI-Math is that a model’s capability boundary need not be observed only after parameter updates through downstream evaluation; instead, it can be actively probed under a fixed policy by conducting structured search over the model’s own reasoning trajectories. As illustrated in Figure 2, ReSI-Math decomposes the resulting selfimprovement procedure into four stages: Elicit, Repair, Consolidate, and Reprofile.

## 3.1 Elicit: Probing the Recoverable Capability Frontier

Given a current policy $\pi _ { \theta }$ and a mathematical problem $x ,$ the set of sampled reasoning trajectories is denoted as: ${ \mathcal T } ( x ; \pi _ { \theta } ) ~ = ~ \{ \tau _ { 1 } , \dots , \tau _ { K } \} , \tau _ { i } ~ \sim$ $\pi _ { \boldsymbol { \theta } } ( \cdot \mathrm { ~ \boldsymbol ~ { ~ \vert ~ } ~ } \boldsymbol { x } )$ . A binary outcome $V ( x , \tau _ { i } ) \in \{ 0 , 1 \}$ is assigned to each trajectory by a rule-based verifier $V .$ . Verifier-confirmed correct and failed trajectories are collected into C and $F ,$ respectively, forming the initial seed set for subsequent repair, $S _ { 0 } = C \cup F$

The resulting trajectory partition characterizes the direct-generation behavior of the current policy on x. Directly expressed capability is characterized by verifier-confirmed success under direct generation, whereas recoverable capability is evaluated with the model parameters θ held fixed and is defined by the existence of a bounded search or recombination operator $g$ such that

$$
\exists \tau _ { i } , \tau _ { j } \in \mathcal { T } ( x ; \pi _ { \theta } ) , \mathrm { s . t . } V ( x , g ( \tau _ { i } , \tau _ { j } ) ) = 1 .\tag{1}
$$

A successful recovery indicates capability present in the trajectory distribution induced by the current policy rather than capability acquired through parameter updates. The problem set satisfying this criterion is referred as the policy’s recoverable capabilityfrontier, denoted by ${ \mathcal { F } } _ { \mathrm { r e c } } ( \pi _ { \theta } )$

This frontier is not equivalent to a simple correct– incorrect boundary: direct sampling may yield both correct and failed trajectories, while even two failed parent trajectories may be recombined into a verifier-confirmed correct offspring. Such $F { - } F {  } C$ (Failed–Failed-to-Correct) transitions show that recoverability does not require an initially correct parent. Accordingly, the frontier is not characterized solely by the initial pass rate; the existence of a failure-to-success transition under bounded structured search serves as the operational criterion for recoverability. Under this formulation, Elicit uses direct-generation profiles to localize candidate frontier regions, while recoverability is established in Repair through verifier-confirmed failure-to-success transitions.

![](images/55bc41843e1a5fc66efb8d99bd8a376b020e97df367de50529e187677a40bbae.jpg)  
Figure 2: Overview of ReSI-Math, including four stages: Elicit, Repair, Consolidate, and Reprofile.

## 3.2 Repair: Self-Judge-Guided Evolutionary Search

Given candidate regions near the recoverable capability frontier, Repair evaluates whether failed reasoning can be transformed into verifier-confirmed correct solutions through Self-Judge-guided evolutionary search. The model parameters remain fixed throughout this stage, and search is restricted to the trajectory space induced by the current policy.

Self-Judge provides structured analysis of candidate trajectories for evolutionary search. Two complementary judgment modes are considered: the Blind Judge evaluates reasoning without access to verifier outcomes or reference answers, whereas the Guided Judge additionally receives verifierconfirmed outcome feedback while remaining isolated from the reference solution. Self-Judge and the verifier therefore serve distinct functions: Self-Judge provides search guidance, whereas the verifier supplies the external correctness criterion used to determine whether repair succeeds.

Given parent $\tau _ { i } , \tau _ { j }$ , and their Self-Judge feedback, an offspring trajectory is generated:

$$
\tau _ { \mathrm { c h i l d } } \sim \pi _ { \boldsymbol \theta } ( \cdot  { \mid } x , \tau _ { i } , \tau _ { j } , J ( \tau _ { i } ) , J ( \tau _ { j } ) ) .\tag{2}
$$

The crossover operator is adapted from the reflective global crossover formulation of CoTEvol (Wang et al., 2026) and is used to recombine complementary reasoning structures rather than reproduce a parent trajectory verbatim.

Each offspring is subsequently evaluated by the rule-based verifier. A verifier-confirmed transition from at least one failed parent to a correct offspring constitutes a realized failure-to-success transition, including both $C - F \to C$ and $F - F \to C .$ . Such transitions provide operational evidence of recoverability under the fixed policy. The effectiveness of $C { - } F$ repair is quantified by the offspring-level recovery rate:

$$
\mathrm { R R a t e } _ { C - F } = \frac { \# \{ { \mathrm { c o r r e c t } C - F ~ } \mathrm { o f f s p r i n g } \} } { \# \{ C - F ~ \mathrm { o f f s p r i n g } \} } .\tag{3}
$$

This metric measures the fraction of generated C– F offspring that are verified as correct and is distinct from the fraction of source problems for which at least one successful repair is obtained. Detailed population construction, Self-Judge protocols, evolutionary updates, and search configurations are provided in Appendix B.2, while the crossover prompt is provided in Appendix D.

## 3.3 Consolidate: Dual-Signal RLVR

The Repair stage reveals capabilities that can be recovered through search under a fixed policy, but search-time recoverability does not imply that such capabilities are stably expressed by the model. The Consolidate stage therefore converts the signals exposed during evolution into parameter-level learning through RLVR.

Frontier-Guided Problem Selection. Before Solver-RLVR, a single training problem is selected from the current candidate pool using a policydependent top-1 selector.

For candidate problem $q _ { i } ^ { \phantom { } }$ , K denotes the number of sampled trajectories, $N _ { i } ^ { \mathrm { c o r r e c t } }$ denotes the number of verifier-confirmed correct trajectories, and $N _ { i } ^ { \mathrm { p a r s e d } }$ denotes the number of successfully parsed outputs. The rollout success rate is defined as $p _ { i } = N _ { i } ^ { \mathrm { c o r r e c t } } / K$ . Answer-level disagreement among successfully parsed outputs is defined as

$$
D _ { \mathrm { s e m a n t i c } , i } = 1 - \sum _ { k } q _ { i k } ^ { 2 } .\tag{4}
$$

Here, $q _ { i k } = n _ { i k } / N _ { i } ^ { \mathrm { p a r s e d } }$ denotes the proportion of successfully parsed outputs of problem $q _ { i }$ that belong to answer category $k ,$ where $n _ { i k }$ is the number of outputs assigned to category $k .$ . The valid-output rate is defined as

$$
\mathrm { P a r s e R a t e } _ { i } = \frac { N _ { i } ^ { \mathrm { p a r s e d } } } { K } .\tag{5}
$$

The frontier-selection score is defined as

$$
S _ { i } = p _ { i } ( 1 - p _ { i } ) \cdot D _ { \mathrm { s e m a n t i c } , i } \cdot \mathrm { P a r s e R a t e } _ { i } .\tag{6}
$$

The highest-scoring candidate is selected as

$$
q ^ { \star } = \arg \operatorname* { m a x } _ { q _ { i } \in \mathcal { Q } } S _ { i } .\tag{7}
$$

The three factors respectively favor candidates with non-degenerate rollout outcomes, high answerlevel disagreement, and reliable output parsing. The selected problem $q ^ { \star }$ is then used as the training target for single-problem Solver-RLVR.

Solver Consolidation. For the Solver, a problem that exhibits a verifier-confirmed failure-to-success transition is treated as a recoverable training example. Rather than imitating the evolved correct trajectory itself, Solver-RLVR trains the current policy directly on the original problem using verifierbased rewards. The objective is to transform capability that can only be elicited through search into behavior that can be expressed more reliably through direct generation.

Judge Consolidation. Evolution also exposes self-evaluation errors by comparing Self-Judge predictions with verifier-confirmed outcomes. These disagreements provide trajectory-level supervision for Judge-RLVR, where the model is trained to better discriminate whether a candidate reasoning trajectory reaches a correct outcome.

The two learning signals are therefore complementary: recoverable problems provide supervision for Solver-RLVR, while judgment errors provide trajectory-level supervision for Judge-RLVR.

Thus, Consolidate converts search-time behavior into parameterized capability: recoverable reasoning becomes more stable direct problem-solving behavior, while self-evaluation errors become supervision for more reliable judgment.

## 3.4 Reprofile: Local Recursive Self-Improvement

The recoverable capability frontier is policydependent rather than a fixed property of the candidate set. After Solver-RLVR updates the policy from $\pi _ { t }$ to $\pi _ { t + 1 }$ , the rollout distribution over the same candidate problems may change: previously frontier-adjacent problems may become reliably solved or less informative, while other problems may move closer to the updated capability boundary.

Candidate problems are therefore re-evaluated after each policy update rather than following a fixed ranking obtained before training. Let the candidate pool at stage t be

$$
\mathcal { Q } _ { t } = \{ q _ { 1 } , \ldots , q _ { M } \} .\tag{8}
$$

At stage t, candidate problems are re-sampled under the current policy $\pi _ { t }$ , and the rollout statistics required by the frontier-selection score are recomputed using verifier-confirmed outcomes. The next training target is selected using the same top-1 rule defined in Consolidate:

$$
q _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { q \in \mathcal { Q } _ { t } } S ( q ; \pi _ { t } ) ,\tag{9}
$$

where $S ( q ; \pi _ { t } )$ denotes the frontier-selection score evaluated under the current policy. Because the score depends on the policy-induced rollout distribution, the ranking of candidate problems is recomputed after each policy update.

Training on $q _ { t } ^ { \star }$ produces the updated policy $\pi _ { t + 1 }$ under which the candidate pool is reprofiled before the subsequent selection. Repeating this profile– select–update cycle yields the policy-adaptive sequential RLVR process in ReSI-Math, where each policy update changes the evidence used to determine the next training target.

## 4 Experiment

In this section, we evaluate the effectiveness and generalizability of our ReSI-Math framework

Table 1: Main results of Solver. The concrete selected problems and their selection-time profiles are reported in Table 9, with the corresponding problems for the three backbones visualized in Figures 5–7.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td rowspan="2">Selected Problems</td><td rowspan="2">Ckpt</td><td rowspan="2">MATH500</td><td colspan="4">WE-MATH testmini</td><td rowspan="2">AIME 2025</td><td rowspan="2">OlympiadBench</td><td rowspan="2">Avg.</td></tr><tr><td>Overall</td><td>1-step</td><td>2-step</td><td>3-step</td></tr><tr><td colspan="10">Qwen3-VL-2B-Instruct</td></tr><tr><td>Base</td><td></td><td></td><td></td><td>72.60/ -</td><td>54.14 / -</td><td>60.58 / -</td><td>44.44 / -</td><td>27.88 / -</td><td>16.67/-</td><td>17.06 / -</td><td>40.12 /-</td></tr><tr><td>Ours-Q1</td><td>policy-adaptive</td><td>q(2B)</td><td>250</td><td>72.40 / -0.20</td><td>54.48 / +0.34</td><td>61.48 / +0.90</td><td>41.11 / -3.33</td><td>32.12 / +4.24</td><td>16.67 / +0.00</td><td>16.77 / -0.29</td><td>40.08 / -0.04</td></tr><tr><td>Ours-Q2</td><td>policy-adaptive</td><td> $q _ { 1 } ^ { ( 2 B ) }  q _ { 2 } ^ { ( 2 B ) }$ </td><td>550</td><td>74.00 /+1.40</td><td>56.32 / +2.18</td><td>62.22 / +1.64</td><td>46.39 / +1.95</td><td>34.55 / +6.67</td><td>6.67 / -10.00</td><td>19.14/+2.08</td><td>39.03 / -1.09</td></tr><tr><td>Ours-Q3</td><td>policy-adaptive</td><td> $q _ { 1 } ^ { ( 2 B ) }  q _ { 2 } ^ { ( 2 B ) }  q _ { 3 } ^ { ( 2 B ) }$ </td><td>200</td><td>75.40 / +2.80</td><td>58.10 / +3.96</td><td>64.36 / +3.78</td><td>46.67 / +2.23</td><td>36.97 / +9.09</td><td>13.33 / -3.34</td><td>22.26 / +5.20</td><td>42.27 / +2.15</td></tr><tr><td>Fixed-Seq</td><td>fixed order</td><td>fixed 69-problem pool</td><td>best</td><td>74.40 / +1.80</td><td>54.83 / +0.69</td><td>61.07 / +0.49</td><td>45.56 / +1.12</td><td>29.09 / +1.21</td><td>10.00 / -6.67</td><td>20.33 / +3.27</td><td>39.89 / -0.23</td></tr><tr><td>Full-Batch</td><td>joint GRPO</td><td>same 69-problem pool</td><td>best</td><td>74.00 /+1.40</td><td>54.71 / +0.57</td><td>61.15 / +0.57</td><td>44.44 / +0.00</td><td>29.70 /+1.82</td><td>10.00 / -6.67</td><td>18.84 / +1.78</td><td>39.39 / -0.73</td></tr><tr><td colspan="10">Qwen3-VL-4B-Instruct</td><td></td></tr><tr><td>Base</td><td></td><td></td><td></td><td>85.40 / -</td><td>67.36 /-</td><td>71.93 / -</td><td>60.83 / -</td><td>47.88 / -</td><td>26.67 /-</td><td>36.05 / -</td><td>53.87 / -</td></tr><tr><td>Ours-Q1</td><td>policy-adaptive</td><td></td><td>300</td><td>85.80 / +0.40</td><td>68.45 / +1.09</td><td>73.58 / +1.65</td><td>59.72 / -1.11</td><td>49.70 /+1.82</td><td>26.67 / +0.00</td><td>35.46 / -0.59</td><td>54.09 / +0.22</td></tr><tr><td>Ours-Q2</td><td>policy-adaptive</td><td>q{4B) → q(4B)</td><td>100</td><td>85.40 / +0.00</td><td>68.28 / +0.92</td><td>73.17/+1.24</td><td>62.22 / +1.39</td><td>45.45 / -2.43</td><td>30.00 / +3.33</td><td>33.53 / -2.52</td><td>54.30 / +0.43</td></tr><tr><td>Ours-Q3</td><td>policy-adaptive</td><td> $q _ { 1 } ^ { ( 4 B ) }  q _ { 2 } ^ { ( 4 B ) }  q _ { 3 } ^ { ( 4 B ) }$ </td><td>250</td><td>85.40 / +0.00</td><td>68.79 / +1.43</td><td>73.58 / +1.65</td><td>61.39 / +0.56</td><td>49.70 /+1.82</td><td>26.67 / +0.00</td><td>36.05 / +0.00</td><td>54.23 / +0.36</td></tr><tr><td>Fixed-Seq Full-Batch</td><td>fixed order</td><td>fixed 32-problem pool</td><td>650</td><td>85.20 / -0.20 84.60 / -0.80</td><td>68.05 / +0.69 68.39 / +1.03</td><td>73.00 /+1.07 73.50 /+1.57</td><td>58.89 / -1.94 58.89 / -1.94</td><td>51.52 /+3.64</td><td>23.33 / -3.34</td><td>34.87 / -1.18</td><td>52.86 / -1.01</td></tr><tr><td></td><td>joint GRPO</td><td>same 32-problem pool</td><td>best</td><td></td><td></td><td></td><td></td><td>51.52 / +3.64</td><td>30.00 / +3.33</td><td>36.20 / +0.15</td><td>54.80 / +0.93</td></tr><tr><td colspan="10">Qwen3-VL-8B-Instruct</td></tr><tr><td>Base</td><td></td><td></td><td></td><td>85.80 /-</td><td>71.61 /-</td><td>75.72/-</td><td>64.72 / -</td><td>56.36 /-</td><td>23.33 / -</td><td>32.20/-</td><td>53.23 / -</td></tr><tr><td>Ours-Q1</td><td>policy-adaptive</td><td>q(8B)</td><td>500</td><td>86.80 / +1.00</td><td>71.95 / +0.34</td><td>75.88 / +0.16</td><td>65.56 / +0.84</td><td>56.97 / +0.61</td><td>23.33 / +0.00</td><td>34.57 / +2.37</td><td>54.16 / +0.93</td></tr><tr><td>Ours-Q2</td><td>policy-adaptive</td><td> $q _ { 1 } ^ { ( 8 B ) }  q _ { 2 } ^ { ( 8 B ) }$ </td><td>500</td><td>86.00 / +0.20</td><td>73.05 / +1.44</td><td>77.70/+1.98</td><td>65.00 / +0.28</td><td>56.36 / +0.00</td><td>23.33 / +0.00</td><td>31.60 / -0.60</td><td>53.50 / +0.27</td></tr><tr><td>Ours-Q3</td><td>policy-adaptive</td><td> $q _ { 1 } ^ { ( 8 B ) }  q _ { 2 } ^ { ( 8 B ) }  q _ { 3 } ^ { ( 8 B ) }$ </td><td>650</td><td>86.80 / +1.00</td><td>73.51 / +1.90</td><td>77.45 /+1.73</td><td>68.06 / +3.34</td><td>56.36 / +0.00</td><td>26.67 / +3.34</td><td>33.09 / +0.89</td><td>55.02 / +1.79</td></tr><tr><td>Fixed-Seq</td><td>fixed order</td><td>fixed 28-problem pool</td><td>best</td><td>86.80 / +1.00</td><td>72.13 /+0.52</td><td>76.63 / +0.91</td><td>64.44 / -0.28</td><td>55.76 / -0.60</td><td>16.67 / -6.66</td><td>32.79 / +0.59</td><td>52.10/ -1.13</td></tr><tr><td>Full-Batch</td><td>joint GRPO</td><td>same 28-problem pool</td><td>20</td><td>85.40 / -0.40</td><td>71.44 / -0.17</td><td>75.88 / +0.16</td><td>64.44 / -0.28</td><td>53.94 / -2.42</td><td>26.67 / +3.34</td><td>32.79 / +0.59</td><td>54.08 / +0.85</td></tr></table>

across multiple standard benchmarks.

## 4.1 Experimental Setup

Models. We use Qwen3-VL-8B-Instruct as the primary backbone throughout our experiments, as well as Qwen3-VL-2B-Instruct and Qwen3-VL-4B-Instruct. Qwen3-VL is an open-weight visionlanguage model family with strong text understanding, visual perception, and multimodal reasoning capabilities.

Datasets. We construct our evolutionary training pool from MathBook-Pro in WE-MATH 2.0 (Qiao et al., 2026). We construct the training data for both Solver-RLVR and Judge-RLVR from a common source of 243 seed mathematical problems. For Solver-RLVR, the evolutionary search produces a 28-problem candidate pool, from which one problem is selected at each sequential stage for singleproblem RLVR. For Judge-RLVR, trajectory-level supervision is derived from the same evolutionary source using verifier-labeled solver trajectories. Detailed data construction, filtering criteria, and statistics are provided in Appendix A. For Solver checkpoint selection, we construct DEV192 by sampling 96 problems from MATH500 and 96 problems from WE-MATH testmini. Since DEV192 is contained in the full evaluation benchmarks reported in the main tables, we additionally evaluate on subsets that exclude these development examples. The corresponding results are reported in Appendix C.

Benchmarks. For Solver evaluation, we use four mathematical reasoning benchmarks: WE-MATH testmini(Qiao et al., 2025), MATH500(Hendrycks et al., 2021), AIME 2025(Zhang and Team, 2025), and OlympiadBench(He et al., 2024). For Judge evaluation, we use JudgeBench (Tan et al., 2025), a pairwise LLM-as-a-Judge benchmark with objectively verifiable preference labels across Knowledge, Reasoning, Math, and Coding. We use the GPT-4o split containing 350 response pairs.

Implementation Details. Both Solver-RLVR and Judge-RLVR use Group Relative Policy Optimization (GRPO) (Shao et al., 2024) with LoRA adaptation. Solver-RLVR uses 16 online rollouts per training problem with batch size 1, while Judge-RLVR uses 8 rollouts per candidate prompt. Full optimization hyperparameters are provided in Appendix B.3.

## 4.2 Main Results

Recoverability under Fixed-Policy Search. Under the evolutionary search protocol, 621 correct– failed (C–F) crossover offspring are generated from the 243-problem source set, of which 272 are verified as correct, corresponding to an offspringlevel recovery rate of 43.8%. This result provides direct empirical evidence that failed trajectories can contain complementary reasoning information that remains recoverable under a fixed policy.

Solver Results across Model Scales. Table 1 reports the policy-adaptive Solver-RLVR results across Qwen3-VL-2B, 4B, and 8B. The largest relative improvement is obtained on the 2B backbone: after three sequentially selected problems, MATH500 increases from 72.60 to 75.40 (+2.80), WE-MATH testmini from 54.14 to 58.10 (+3.96), and the four-benchmark average from 40.12 to 42.27 (+2.15). The corresponding average gains are +0.36 for 4B and +1.79 for 8B. Although the strongest relative gain is observed on 2B, the controlled scheduling ablations and mechanism analyses are conducted on the 8B backbone to maintain a consistent primary experimental setting.

Table 2: Cross-scale summary of post-hoc rollout dynamics during policy-adaptive sequential Solver-RLVR. Full checkpoint-wise profiles are provided in Appendix E.2.
<table><tr><td>Backbone</td><td>Frontier Pool</td><td>Q1 Target</td><td>Q2 Target</td><td>Q3 Target</td><td></td><td>Q1 / Q2 at Q3-best</td><td> $\mathbf { A v g } . \Delta$ </td><td>Dynamics</td></tr><tr><td>Qwen3-VL-2B</td><td>69</td><td> $1 1 / 1 6 \to 1 6 / 1 6$ </td><td> $2 / 1 6 \to 8 / 1 6$ </td><td> $6 / 1 6 \to 1 4 / 1 6$ </td><td></td><td> $1 6 / 1 6 ~ / ~ 9 / 1 6$ </td><td>+2.15</td><td>Stable retention</td></tr><tr><td>Qwen3-VL-4B</td><td>32</td><td> $5 / 1 6 \to 1 / 1 6$ </td><td> $1 0 / 1 6 \to 6 / 1 6 ^ { \dag }$ </td><td> $9 / 1 6 \to 1 5 / 1 6$ </td><td></td><td> $0 / 1 6 ~ / ~ 9 / 1 6$ </td><td>+0.36</td><td>Target degradation</td></tr><tr><td>Qwen3-VL-8B</td><td>28</td><td> $6 / 1 6 \to \mathbf { 1 6 } / \mathbf { 1 6 }$ </td><td> $5 / 1 6 \to \mathbf { 1 6 } / \mathbf { 1 6 }$ </td><td> $7 / 1 6 \to 1 6 / 1 6$ </td><td></td><td>11/16 16/16</td><td>+1.79</td><td>Forgetting / recovery</td></tr></table>

Table 3: Ablation on training-problem selection for single-problem RLVR with Qwen3-VL-8B-Instruct.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Selection</td><td rowspan="2">Ckpt</td><td rowspan="2">MATH500</td><td colspan="4">WE-MATH testmini</td><td rowspan="2">AIME 2025</td><td rowspan="2">OlympiadBench</td><td rowspan="2">Avg.</td></tr><tr><td>Overall</td><td>1-step</td><td>2-step</td><td>3-step</td></tr><tr><td>Base</td><td></td><td></td><td>85.80</td><td>71.61</td><td>75.72</td><td>64.72</td><td>56.36</td><td>23.33</td><td>32.20</td><td>53.23</td></tr><tr><td>Random-1</td><td>random</td><td>400</td><td>85.20</td><td>71.78</td><td>76.87</td><td>64.44</td><td>50.30</td><td>23.33</td><td>32.94</td><td>53.31</td></tr><tr><td>Ours-Q1</td><td>policy-adaptive</td><td>500</td><td>86.80</td><td>71.95</td><td>75.88</td><td>65.56</td><td>56.97</td><td>23.33</td><td>34.57</td><td>54.16</td></tr></table>

For the matched scheduling baselines, Fixed-Seq removes the frontier-guided top-1 selection mechanism and traverses the entire candidate pool in a fixed order, training on one problem at a time with a training batch size of 1. Full-Batch instead jointly optimizes all problems in the corresponding candidate pool within each training batch. Full-Batch GRPO is evaluated under the same rollout budget as the policy-adaptive setting, while Fixed-Sequential is included as a lower-budget fixedorder reference; detailed budget accounting is provided in Appendix E.3.

Under this controlled comparison, policyadaptive sequential RLVR consistently outperforms Fixed-Seq across all three model scales. Relative to the corresponding Base models, policy-adaptive training improves the average score by 2.15 points on 2B, 0.36 points on 4B, and 1.79 points on 8B, whereas Fixed-Seq decreases the average by 0.23, 1.01, and 1.13 points, respectively. These results indicate that dynamic selection is consistently more effective than traversing the same candidate pool in a fixed sequential order, while its advantage over joint optimization is model-dependent; the distinct 4B behavior is further examined through the rollout dynamics in Table 2.

Scale-Dependent and Non-Monotonic Frontier Dynamics. Table 2 reveals substantial scale dependence in both frontier size and training dynamics. The candidate pools contain 69, 32, and 28 problems for Qwen3-VL-2B, 4B, and 8B, respectively, suggesting that a larger fraction of the shared problem distribution lies near the recoverable capability boundary for smaller models. The rollout profiles further show that single-problem RLVR does not induce monotonic capability acquisition: 2B largely preserves previously acquired behavior, 4B exhibits target-specific degradation, and 8B shows temporary forgetting followed by partial recovery. Additional re-learning experiments further reveal reverse interference, where improving a target capability can suppress previously consolidated capabilities and, in more severe cases, induce partial forgetting. Detailed checkpoint-wise rollout profiles and reverse-interference analyses are provided in Appendix E.2.

## 4.2.1 Self-Judge Provides Structured but Imperfect Search Signals

Evolutionary search requires more than generating additional trajectories: it also requires a mechanism for deciding which reasoning paths are promising enough to preserve, compare, or recombine. In our framework, the same model is switched from the Solver role to the Self-Judge role to provide trajectory-level feedback during evolution.

However, Self-Judge is not treated as a correctness oracle. Comparing blind judgments against verifier-confirmed outcomes exposes two characteristic types of self-evaluation error. We identify 33 false-positive cases in which an actually failed parent trajectory is judged as correct, and 35 falsenegative cases in which a verified-correct child trajectory is judged as incorrect. These errors provide 68 explicit self-correction signals for downstream Judge training.

These observations suggest that Self-Judge provides structured information about trajectory quality, but remains imperfect. We therefore use it as a search controller rather than as the final source of correctness. The Self-Judge determines which trajectories deserve attention during search, while the deterministic verifier independently confirms whether the resulting outcome is correct.

Table 4: Self-Judge errors identified from verifierconfirmed evolutionary trajectories.
<table><tr><td>Type</td><td>Trajectory</td><td>Verifier</td><td>Blind Judge</td><td>Count</td></tr><tr><td>False Positive</td><td>Parent</td><td>F</td><td>C</td><td>33</td></tr><tr><td>False Negative</td><td>Child</td><td>C</td><td>F</td><td>35</td></tr></table>

![](images/82c42ad3635661fc71b541298ca85fa1c58ff4ec28f1d66740b9a8d8227eec7a.jpg)  
Figure 3: Policy-adaptive sequential Solver-RLVR dynamics on DEV192. Q1, Q2, and Q3 are trained sequentially, with each stage initialized from the DEV192- selected checkpoint of the preceding stage, yielding Q1@500, Q2@500, and Q3@650.

A further consequence is that evolutionary search exposes not only reasoning failures, but also the model’s own evaluation failures. This creates a natural source of trajectory-level supervision for Judge-RLVR, which we examine below.

Solver Results. Table 1 summarizes the policyadaptive Solver-RLVR results across Qwen3-VL-2B, 4B, and 8B, together with matched scheduling baselines. The largest relative improvement is obtained on the 2B backbone, where the fourbenchmark average increases from 40.12 to 42.27 (+2.15). The corresponding gains are 0.36 points for 4B and 1.79 for 8B. On the 8B backbone, which is used for the main controlled ablations, policy-adaptive sequential training outperforms fixed-order sequential training and achieves a larger gain than joint optimization over the 28-problem candidate pool.

## 4.3 Ablation Study

To examine whether the gains from single-problem RLVR can be attributed merely to training on an arbitrary example, we compare our policy-adaptive Q1 selection with a Random-1 baseline under the same 28-problem candidate pool, the same singleproblem RLVR setting, and the same training budget. Random-1 improves the overall average only marginally, from 53.23 to 53.31 (+0.08), with inconsistent changes across evaluation dimensions: WE-MATH 1-step increases to 76.87, whereas MATH500 decreases from 85.80 to 85.20 and WE-MATH 3-step drops from 56.36 to 50.30. In contrast, frontier-guided Q1 raises the overall average to 54.16 (+0.93), outperforming Random-1 by 0.85 points, while also exceeding the Base model on MATH500, WE-MATH Overall, 2-step, 3-step, and OlympiadBench. These results indicate that, in the extreme low-data single-problem RLVR regime, the observed gains cannot be explained solely by training on an arbitrary example; under a matched training budget, the alignment between the selected problem and the current policy state materially affects downstream generalization performance.

Table 5: JudgeBench results of Judge-RLVR.
<table><tr><td>Model</td><td>Knowledge</td><td>Reasoning</td><td>Math</td><td>Coding</td><td>Overall</td></tr><tr><td>Base</td><td>57.14</td><td>75.51</td><td>80.36</td><td>61.90</td><td>66.57</td></tr><tr><td>Judge-RLVR</td><td>59.09</td><td>73.47</td><td>82.14</td><td>64.29</td><td>67.43</td></tr></table>

## 5 Conclusion

We introduce ReSI-Math, a framework for recursive self-improvement in mathematical reasoning. Self-Judge-guided evolutionary search probes the recoverable capability frontier and uses crossover to repair failed reasoning into verifierconfirmed correct trajectories. Recoverable problems and judgment–verification discrepancies provide complementary supervision for Solver-RLVR and Judge-RLVR, respectively. Policy-adaptive sequential RLVR re-profiles the problem pool after each policy update to guide subsequent training. Experiments show that evolutionary search uncovers capabilities not yet reliably expressed by the model, while dynamic re-profiling outperforms static schedules in the extreme low-data regime. Thus, ReSI-Math extends evolutionary search beyond data synthesis to capability probing and learning-signal discovery, translating searchtime recoverability into more reliable reasoning.

## Limitations

This work has two main limitations. First, our dynamic capability frontier is an operational, empirically defined notion under the current experimental setting. While sufficient for our analysis, it does not provide a strict characterization of the model’s underlying capability boundary relative to the training data. Second, our experiments focus primarily on the Qwen3-VL family and mathematical reasoning, and we optimize the Solver and Judge separately rather than jointly. Generalization to other model families and reasoning domains, as well as the potential benefits of joint Solver–Judge optimization, remain important directions for future work.

## References

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, and Hongxia Jin. 2024. Alpagasus: Training a better alpaca with fewer data. In International Conference on Learning Representations, volume 2024, pages 34767–34797.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jingqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan-ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, and Chaojun Xiao. 2026. Rethinking onpolicy distillation of large language models ii: One training example. arXiv preprint arXiv:2609.04172.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Bingxiang He, Ning Ding, Cheng Qian, Jia Deng, Ganqu Cui, Lifan Yuan, Haiwen Hong, Huan-ang Gao, Longtao Huang, Hui Xue, Huimin Chen, Zhiyuan Liu, and Maosong Sun. 2025. The right time matters: Data arrangement affects zero-shot generalization in instruction tuning. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 222–243, Vienna, Austria. Association for Computational Linguistics.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. 2024. Olympiad-Bench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. 2026. R-zero: Selfevolving reasoning llm from zero data. In International Conference on Learning Representations, volume 2026, pages 130770–130790.

Hamish Ivison, Muru Zhang, Faeze Brahman, Pang Wei Koh, and Pradeep Dasigi. 2025. Large-scale data selection for instruction tuning. arXiv preprint arXiv:2503.01807.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025. LIMR: Less is more for RL scaling. arXiv preprint arXiv:2502.11886.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651.

Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma GongQue, Shanglin Lei, Yifan Zhang, et al. 2025. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20023–20070. Association for Computational Linguistics.

Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Guanting Dong, Shiqiang Lang, Sitong Zhou, Yida Xu, Yuchen Zeng, Jie Wang, Chong Sun, Chen Li, and Honggang Zhang. 2026. We-math 2.0: A versatile mathbook system for incentivizing visual mathematical reasoning. In International Conference on Learning Representations.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, et al. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Tang, Alejandro Cuadron, Chenguang Wang, Raluca Popa, and Ion Stoica. 2025. Judgebench: A benchmark for evaluating llm-based judges. In International Conference on Learning Representations, volume 2025, pages 63277–63303.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang

Wang, Simon Du, and yelong shen. 2025. Reinforcement learning for reasoning in large language models with one training example. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 122721–122764. Curran Associates, Inc.

Zhuo Wang, Zhuo Zhang, Yafu Li, Yu Cheng, Lizhen Qu, and Zenglin Xu. 2026. CoTEvol: Self-evolving chain-of-thoughts for data synthesis in mathematical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 38153–38173, San Diego, California, United States. Association for Computational Linguistics.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. 2024. LESS: Selecting influential data for targeted instruction tuning. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 54104–54132. PMLR.

Yifan Zhang and Math-AI Team. 2025. American invitational mathematics examination (aime) 2025.

Yunjian Zhang, Sudong Wang, Yang Li, Peiran Xu, Conghao Zhou, Xiaoyue Ma, Jianing Li, and Yao Zhu. 2026. Resource-efficient reinforcement for reasoning large language models via dynamic one-shot policy refinement. arXiv preprint arXiv:2602.00815.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

## Appendix

## A Training Data Construction

## A.1 Solver-RLVR Data Construction

The Solver receives a text problem together with its associated image when available; text-only benchmarks contain no image input. The model generates a reasoning trajectory followed by a final answer formatted as \boxed{...}. A deterministic rule-based verifier parses the final answer and compares it with the official reference answer. The verifier supports multiple answer formats, including multiple-choice options, integers, decimals, fractions, radicals, coordinates, and symbolic expressions. A rollout receives reward 1 if the final answer is verified as correct and 0 otherwise.

## A.2 Judge-RLVR Data Construction

The Judge receives the original problem, the associated image when available, and a candidate Solver trajectory. Instead of generating a new solution, it predicts whether the candidate reaches a verifier-confirmed correct outcome. The final judgment is represented as \boxed{1} for correct and \boxed{0} for incorrect. The underlying verifier label is hidden from the model during RLVR. The Judge receives reward 1 if its prediction matches the hidden label and 0 otherwise. Therefore, Judge-RLVR trains outcome discrimination rather than direct answer generation.

## A.3 Dataset Statistics

We select 243 problems organized into such families as the seed pool for evolutionary search. This family structure allows us to study complementary reasoning behaviors among closely related multimodal problems and to perform crossover between trajectories that share common semantic structure.

In our preprocessing, we group each base instance and its progressive variants into a common problem family, represented using the internal identifiers base/x/y/z/xy/xz/yz/xyz.

The resulting evolutionary trajectories provide two types of downstream supervision. At the problem level, a failure-to-success transition identifies a problem that lies near the model’s recoverable capability frontier and can therefore serve as a candidate for Solver-RLVR. At the trajectory level, the evolutionary process produces verifier-confirmed correct and incorrect candidate solutions, which can be converted into Judge-RLVR examples. Thus, the same evolutionary source provides complementary supervision for solving and judging, although the corresponding training units are different: problems for Solver-RLVR and candidate trajectories for Judge-RLVR.

## B Implementation Details

Table 6 summarizes the functional differences among evolutionary search, Solver-RLVR, and Judge-RLVR. Detailed configurations are provided in the following subsections.

## B.1 Rule-Based Verifier

We use a deterministic rule-based outcome verifier as the external correctness anchor. During blind crossover and blind self-judgment, the model is not given the reference answer, verifier label, or information about which parent trajectory is correct. After generation, the verifier checks the final outcome and assigns a correctness label. The verifierconfirmed outcome can subsequently be used by the guided Judge branch and as the trusted binary supervision for RLVR, while the reference answer itself remains hidden from the Judge.

Conceptually, the Self-Judge determines where to search, whereas the rule-based verifier confirms whether the resulting trajectory succeeds.

## B.2 Evolutionary Search Configuration

We use the v7-era Self-Judge-guided evolutionary framework throughout the main experiments. For each seed problem, the Solver first generates 16 independent trajectories in Round 0. A rule-based verifier then determines their outcomes, after which we maintain a fixed population of four trajectories consisting of two verifier-confirmed correct trajectories and two failed trajectories.

The population is evolved for three rounds. At each round, candidate parent trajectories are selected based on the current population and Self-Judge feedback, and crossover is used to generate new candidate trajectories. The offspring are verified and the population is restored to the fixed 2C+2F composition. A Blind Judge and a Guided Judge then analyze trajectory correctness, failure modes, and complementary information to support the subsequent round. The Blind Judge does not observe verifier outcomes, whereas the Guided Judge may use verifier-confirmed outcome feedback without observing the reference solution.

Table 6: Functional comparison of evolutionary search, Solver-RLVR, and Judge-RLVR in ReSI-Math.
<table><tr><td>Setting</td><td>Evolution</td><td>Solver-RLVR</td><td>Judge-RLVR</td></tr><tr><td>Primary role</td><td>Frontier discovery</td><td>Solver optimization</td><td>Judge optimization</td></tr><tr><td>Basic unit</td><td>trajectory</td><td>problem</td><td>candidate trajectory</td></tr><tr><td>Sampling granularity</td><td>16 trajectories (Round 0)</td><td>16 rollouts</td><td>8 rollouts</td></tr><tr><td>Search / optimization</td><td>crossover</td><td>GRPO</td><td>GRPO</td></tr><tr><td>Iterations</td><td>3 rounds</td><td>700 steps / stage</td><td></td></tr><tr><td>External signal</td><td>Self-Judge analysis + verifier outcome</td><td>verifier reward</td><td>verifier-guided judgment label</td></tr><tr><td>Policy parameters updated</td><td>No</td><td>Solver policy</td><td>Judge policy</td></tr><tr><td>Reference answer exposed</td><td>No</td><td>No</td><td>No</td></tr></table>

## B.3 RLVR Hyperparameters

Table 7 summarizes the optimization hyperparameters used for the two training objectives.

For Solver-RLVR, we set the rollout size to 16 to match the 16-way sampling used in evolutionary search and subsequent policy profiling. This keeps the sampling granularity consistent when estimating whether a problem is currently all-fail, mixed, or all-pass and when optimizing the Solver on the selected problem.

Table 7 summarizes the main optimization hyperparameters used for Solver-RLVR and Judge-RLVR. Both variants use GRPO with LoRA adaptation, while differing primarily in rollout size, batch size, and maximum response length.

## C Robustness to DEV192 Checkpoint-Selection Overlap

Our main results report performance on the complete MATH500 and WE-MATH testmini benchmarks for direct comparability with prior work. DEV192, which consists of 96 examples from each benchmark, is used only for checkpoint selection and never contributes to parameter updates or training-data construction. Since these examples remain part of the full benchmark evaluation, checkpoint selection may introduce a small optimistic selection bias. We therefore additionally report performance after removing all DEV192 examples from the corresponding benchmarks.

Specifically, removing the 96 development examples yields MATH404 and WE-MATH1644. The last column reports the average accuracy over these two DEV-excluded evaluation sets.

## D Crossover Prompt

We reproduce the crossover prompt used in our v7-era implementation. The prompt is instantiated from a fixed system–user template with several task-dependent fields. The two parent trajectories are presented anonymously, and their order is deterministically randomized for each question, operator, and replicate. Correctness labels are used only for external task construction and are not exposed in the blind crossover prompt. The main evolutionary data use the blind arm, while the oracle-hinted arm is included only as an additional control condition.

Table 7: Training hyperparameters for Solver-RLVR and Judge-RLVR.
<table><tr><td>Configuration</td><td>Solver</td><td>Judge</td><td>Configuration</td><td>Solver</td><td>Judge</td></tr><tr><td>Algorithm</td><td>GRPO</td><td>GRPO</td><td>Max prompt length</td><td>6144</td><td>6144</td></tr><tr><td>Train batch size</td><td>1</td><td>4</td><td>Max response length</td><td>2048</td><td>1024</td></tr><tr><td>Rollout n</td><td>16</td><td>8</td><td>Max model length</td><td>8192</td><td>8192</td></tr><tr><td>Temperature</td><td>0.7</td><td>0.7</td><td>PPO mini-batch size</td><td>1</td><td>4</td></tr><tr><td>Top-p</td><td>1.0</td><td>1.0</td><td>PPO micro-batch / GPU</td><td>1</td><td>1</td></tr><tr><td>Learning rate</td><td>3 × 10−6</td><td>3 × 10−6</td><td>KL in reward</td><td>No</td><td>No</td></tr><tr><td>LoRA rank</td><td>8</td><td>8</td><td>KL loss</td><td>No</td><td>No</td></tr><tr><td>LoRA alpha</td><td>16</td><td>16</td><td>Entropy coefficient</td><td>0</td><td>0</td></tr><tr><td></td><td></td><td></td><td>Save frequency</td><td>Every 50 steps</td><td>一</td></tr></table>

![](images/f8a2d6a32214d192a15feb3ed544d004f264a901f7438963ead69b5ddbdac484.jpg)  
Figure 4: Crossover prompt template used in the v7-era evolutionary search. Fields in braces are instantiated dynamically according to the crossover task; their exact values are summarized in Table 8.

Table 8: Dynamic fields instantiated in the crossover prompt. The controlled hint is empty in the main blind arm and is populated only for the oracle-hinted control condition.
<table><tr><td>Field</td><td>Condition</td><td>Instantiated Text</td></tr><tr><td rowspan="2">OPERATOR_ INSTRUCTION</td><td>Preserve-Repair</td><td>Preserve independently valid steps, locate contradictions, repair them, and derive a complete solution.</td></tr><tr><td>Independent Recombination</td><td>Rebuild the solution using a meaningfully different organization; use both parents only as fallible evidence and recheck every key step.</td></tr><tr><td>OPTIONAL_ CONTROLLED_HINT</td><td>Main blind arm Oracle-hinted control</td><td>Empty. Controlled hint: Exactly one of the two parent trajectories has a correct final result and the other has an incorrect final result. You are not told which is</td></tr><tr><td rowspan="2">FINAL_ANSWER_ FORMAT_CONSTRAINT</td><td>Multiple-choice</td><td>which. Final-answer format constraint: output exactly one option label A-H inside &lt;answer&gt;...&lt;/answer&gt;, for example</td></tr><tr><td>Non-multiple- choice</td><td>&lt;answer&gt;A&lt;/answer&gt;. Do not place words, LaTeX wrappers, or option content inside that tag. Final-answer format constraint: put only the final mathematical answer inside &lt;answer&gt;...&lt;/answer&gt;.</td></tr></table>

## E Frontier Selection and Policy Dynamics

## E.1 Selected Frontier Problems

Concrete Problem Instances. Figures 5–7 show the concrete Q1–Q3 problems selected for each backbone, including the original question text and the corresponding visual input.

## E.2 Checkpoint-wise Rollout Dynamics of Sequential Solver-RLVR

Frontier Size Varies with Model Scale. The same family model induces substantially different frontier candidate pools across model scales. As shown in Table 1, the resulting pools contain 69 problems for Qwen3-VL-2B, 32 for 4B, and 28 for 8B. This monotonic decrease with model size suggests that a larger fraction of the shared problem distribution lies near the recoverable capability boundary for smaller models, whereas stronger models move more problems away from this intermediate regime. The corresponding checkpointwise profiles in Tables 10–12 further show that the behavior of these frontier problems is strongly scale-dependent.

Frontier Training Exhibits Directional Plasticity. Checkpoint-wise rollout profiles in Tables 10–12 reveal that single-problem RLVR does not induce uniformly monotonic capability improvement. On Qwen3-VL-2B (Table 10), Q1 reaches 16/16 early and remains fully retained throughout subsequent Q2 and Q3 stages, while Q3 is progressively consolidated to near-perfect rollout success; this stable retention is consistent with the largest relative benchmark gain observed in Table 1. In contrast, Qwen3-VL-4B (Table 11) exhibits a target-specific degradation regime during Q2 training, where the post-hoc Q2 profile decreases from 10/16 at initialization to 0/16 at later checkpoints despite continued optimization on Q2. Qwen3-VL-8B (Table 12) shows an intermediate pattern, with temporary degradation of Q1 during Q2 training followed by partial recovery during Q3 training. These results indicate that frontier training is directionally plastic: a single problem can move the policy either toward or away from reliable expression of a targeted capability.

Re-learning Can Induce Reverse Interference. Table 13 further shows that capability recovery is not necessarily globally beneficial. Starting from the selected Q3@650 policy on Qwen3-VL-8B, re-training Q1 increases its rollout success from

11/16 to 16/16, but simultaneously reduces the previously mastered Q2/Q3-related family from 64/64 to 23/64 and decreases DEV192 from 78.12 to 75.00. The corresponding full-test average also drops from 80.16 to 78.47. Thus, local reacquisition of a target capability can suppress previously consolidated capabilities, providing direct evidence that sequential single-problem RLVR exhibits nonmonotonic and bidirectional interference. In more severe cases, this interference manifests as partial or temporary forgetting of previously acquired capabilities.

## E.3 Rollout-Budget Control

To separate the effect of training schedule from the amount of sampled supervision, we strictly match the rollout budget between Policy-Adaptive Sequential RLVR and Full-Batch GRPO. For all three model scales, sequential RLVR performs three stages of at most 700 single-problem updates with $n = 1 6$ rollouts per update, yielding $3 \times 7 0 0 \times 1 6 = 3 3 . 6 0 0$ rollout responses. Full-Batch GRPO is assigned the same 2,100 prompt slots and therefore the same 33,600 rollout responses. Because the candidate-pool sizes differ across model scales, we use 30 full updates over the 69-problem pool followed by a 30-example partial update for 2B, 65 full updates over the 32-problem pool followed by a 20-example partial update for 4B, and 75 full updates over the 28-problem pool for 8B. The resulting budgets are exactly 33,600 rollouts in all cases. The Fixed-Sequential baseline follows the original 700-step fixed-order protocol with batch size 1 and n = 16, corresponding to 11,200 rollouts; we therefore report it as a lowerbudget scheduling reference rather than a budgetmatched comparator.

Table 9: Concrete frontier problems selected during policy-adaptive sequential Solver-RLVR. “Selection Policy” denotes the policy used to reprofile the candidate pool before selecting the corresponding problem. “Profile” reports the number of verifier-confirmed correct trajectories among 16 fixed-policy samples at selection time.
<table><tr><td>Backbone</td><td>Stage</td><td>Symbol</td><td>Selection Policy</td><td>Selected Question ID</td><td>Profile</td></tr><tr><td rowspan="3">Qwen3-VL-2B</td><td>Q1</td><td> $q _ { 1 } ^ { ( 2 B ) }$ </td><td>Base</td><td>family_302__x__row_2409</td><td>11/16</td></tr><tr><td>Q2</td><td> $q _ { 2 } ^ { ( 2 B ) }$ </td><td>Q1@250</td><td>family_320__y__row_2556</td><td>4/16</td></tr><tr><td>Q3</td><td> $q _ { 3 } ^ { \overline { { ( 2 . B ) } } }$ </td><td>Q2@550</td><td>family_420__yz__row_3357</td><td>7/16</td></tr><tr><td rowspan="3">Qwen3-VL-4B</td><td>Q1</td><td>(4B) q1</td><td>Base</td><td>family_320__xz__row_2559</td><td>5/16</td></tr><tr><td>Q2</td><td> $\hat { \ b { \mathscr { M } } } ( 4 { \cal B } )$  q2</td><td>Q1@300</td><td>family_436__y__row_3485</td><td>10/16</td></tr><tr><td>Q3</td><td> $q _ { 3 } ^ { \overline { { ( } } 4 B ) }$ </td><td>Q2@100</td><td>family_203__z__row_1617</td><td>6/16</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>Q1</td><td>(8B) q1</td><td>Base</td><td>family_311__xz__row_2485</td><td>6/16</td></tr><tr><td>Q2</td><td> $q _ { \gamma } ^ { \left( 8 B \right) }$ </td><td>Q1@500</td><td>family_420__y__row_3354</td><td>5/16</td></tr><tr><td>Q3</td><td> $q _ { 3 } ^ { \overline { { ( } } 8 B ) }$ </td><td>Q2@500</td><td>family_420__base__row_3352</td><td>7/16</td></tr></table>

![](images/3ee6df2004b594e2f8613e7ee95c0a7ef3974ff297aef5ba7c4ea15c0f3998f7.jpg)

Profile: 11/16

Question. As shown in the figure, the value of the minimum of the function multiplied by 2.5, plus 1.5, minus 2.5, and then multiplied by 50% is.

![](images/238c2151c2f38ba15fba1c7da79f7cd9ebcb1a816d2d382cdb70c2070fd6d7d2.jpg)

family\_320\_\_y\_\_row\_2556

Profile: 4/16

Question. Given that there were originally 12 circles, how many circles remain after removing the circles inside the dashed lines? A. 3; B. 6; C. 5; D. 7.

![](images/b7062088a038c1213605d6d8efe2faa31aa5b080958975b626b0010506a1ee1d.jpg)

(2B) q<sub>3</sub>

family\_420\_\_yz\_\_row\_3357

Profile: 7/16

Question. As shown in the figure, the object is a prism. The manufacturing cost of this object is 1 yuan per unit volume. What is the total manufacturing cost of this object? (Write only the number.)

Figure 5: Concrete frontier problems selected for Qwen3-VL-2B. Each row shows the visual input, original question, and selection-time fixed-policy profile

![](images/bbf0033d347356decbb5bfc53d0e86ab11882d9e371efd8ee816373fd49d0d2b.jpg)

![](images/b6c4d73aba47883a2324982712e9c0cfd699a1516447250ba7ad2489638ab9cb.jpg)

family\_320\_\_xz\_\_row\_2559

Profile: 5/16

Question. There were originally 12 round cakes, as shown in the figure. Now Mike has taken the cakes within the dashed lines. Let a be the number of cakes remaining, and let b be the number of extrema of the function $\begin{array} { r } { y = \frac { 1 } { 3 } x ^ { 3 } + \overline { { x ^ { 2 } } } + 2 . } \end{array}$ . What is the fourth term of a geometric sequence with first term b and common ratio a?

![](images/18959a187ac2c6b6971dcc0cd95aa35b7aeecfb86a99d5e554637c88fc2f025b.jpg)

(4B) q<sub>2</sub>

family\_436\_\_y\_\_row\_3485

Profile: 10/16

Question. What is the area of the net of the following regular quadrilateral frustum?

(4B) q<sub>3</sub>

family\_203\_\_z\_\_row\_1617

Profile: 6/16

Question. There are two disks of the same size on the table. After fixing their positions, a straight wooden stick is used to touch both disks and be tangent to both sides. How many different ways are there to place the stick? (Write only the number.)

Figure 6: Concrete frontier problems selected for Qwen3-VL-4B. Each row shows the visual input, original question, and selection-time fixed-policy profile.

![](images/1fdf44c0a3059c648529c1f81a602cbdc291c2c15c154f39b906fea9965be1c9.jpg)

![](images/91f93ff135ab05d4eabb2e954e673ee70a70070f59991459d36e22651c9fadab.jpg)

Question. A toy manufacturing company produces a new type of building block toy. According to market research, the relationship between the daily production quantity x (in thousands of sets) and the profit per set y (in yuan) can be represented by the function shown in the figure. The company’s production planning department wants to understand the fluctuation in profit. What is the result of subtracting the minimum value of the function from its maximum value?

![](images/7b32e3420ae41117bcd782270a5c1334946c51fa3188958bff539a0d9de8837b.jpg)  
Figure 7: Concrete frontier problems selected for Qwen3-VL-8B. Each row shows the visual input, original question, and selection-time fixed-policy profile.

Profile: 6/16

(8B) q<sub>2</sub>

family\_420\_\_y\_\_row\_3354

Profile: 5/16

Question. What is the volume of the prism shown in the figure? (Enter only the number.)

(8B) q<sub>3</sub>

family\_420\_\_base\_\_row\_3352

Profile: 7/16

Question. What is the volume of the prism shown in the figure?

Table 10: Checkpoint-wise rollout profiles during sequential single-problem Solver-RLVR on Qwen3-VL-2B. Each entry reports the number of verifier-confirmed correct trajectories among 16 rollouts. All tables use the same heatmap bins: 16/16, 13–15/16, 8–12/16, 4–7/16, and 0–3/16. Blue step cells denote DEV192-selected checkpoints.Cell colors encode absolute rollout accuracy:dark green denotes 16/16, light green 13–15/16, yellow 8–12/16, orange 4–7/16, and light red 0–3/16. Blue step cells mark DEV192-selected checkpoints, while light-red step cells in the 4B Q2 stage indicate the observed optimization-collapse regime.
<table><tr><td rowspan=1 colspan=12>Q1 Training               Q2 Training                Q3 Training</td></tr><tr><td rowspan=1 colspan=4>Step  Q1   Q2  Q3</td><td rowspan=1 colspan=8>Step  Q1   Q2  Q3 Step  Q1   Q2   Q3</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=2>16/16 2/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>6/16</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=2>16/16 1/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>10/16</td><td rowspan=1 colspan=1>7/16</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>11/16</td></tr><tr><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>14/16</td></tr><tr><td rowspan=1 colspan=1>250</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>250</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>250</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>15/16</td></tr><tr><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>350</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>350</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>350</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>10/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>550</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>550</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>550</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>650</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>650</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>650</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>13/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>13/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>13/16</td><td rowspan=1 colspan=1>16/16</td></tr></table>

Table 11: Checkpoint-wise rollout profiles during sequential single-problem Solver-RLVR on Qwen3-VL-4B. Each entry reports the number of verifier-confirmed correct trajectories among 16 rollouts. All tables use the same heatmap bins: 16/16, 13–15/16, 8–12/16, 4–7/16, and 0–3/16. Blue step cells denote DEV192-selected checkpoints. Red step cells in Q2 training mark the observed optimization-collapse regime; accuracy cells retain the same heatmap encoding.
<table><tr><td rowspan=1 colspan=12>Q1 Training               Q2 Training                Q3 TrainingStep Q1   Q2   Q3  Step Q1   Q2   Q3  Step Q1   Q2   Q3</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>10/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=2>8/16  9/16</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>12/16</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>14/16</td></tr><tr><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>13/16</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>15/16</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>6/16</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>13/16</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>15/16</td></tr><tr><td rowspan=1 colspan=1>250</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>3/16</td><td rowspan=1 colspan=1>250</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>14/16</td><td rowspan=1 colspan=1>250</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>15/16</td></tr><tr><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>12/16</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>350</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>5/16</td><td rowspan=1 colspan=1>350</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>350</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>10/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>9/16</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>15/16</td></tr><tr><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>4/16</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>500</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>550</td><td rowspan=1 colspan=1>1/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>15/16</td><td rowspan=1 colspan=1>550</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>550</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>13/16</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>7/16</td><td rowspan=1 colspan=1>15/16</td></tr><tr><td rowspan=2 colspan=1>650700</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>0/16</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>650</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>650</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>8/16</td><td rowspan=1 colspan=1>16/16</td></tr><tr><td rowspan=1 colspan=2>0/16 0/16</td><td rowspan=1 colspan=1>16/16</td><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>700</td><td rowspan=1 colspan=1>2/16</td><td rowspan=1 colspan=1>11/16</td><td rowspan=1 colspan=1>16/16</td></tr></table>

Table 12: Checkpoint-wise rollout profiles during sequential single-problem Solver-RLVR on Qwen3-VL-8B. Each entry reports the number of verifier-confirmed correct trajectories among 16 rollouts. All tables use the same heatmap bins: 16/16, 13–15/16, 8–12/16, 4–7/16, and 0–3/16. Blue step cells denote DEV192-selected checkpoints.
<table><tr><td colspan="4">Q1 Training</td><td colspan="4">Q2 Training</td><td colspan="4">Q3 Training</td></tr><tr><td>Step</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Step</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Step</td><td>Q1</td><td>Q2</td><td>Q3</td></tr><tr><td>0</td><td>6/16</td><td>5/16</td><td>4/16</td><td>0</td><td>16/16</td><td>5/16</td><td>2/16</td><td>0</td><td>8/16</td><td>16/16</td><td>7/16</td></tr><tr><td>50</td><td>一</td><td>一</td><td>一</td><td>50</td><td>10/16</td><td>6/16</td><td>11/16</td><td>50</td><td>11/16</td><td>16/16</td><td>15/16</td></tr><tr><td>100</td><td>一</td><td>一</td><td>一</td><td>100</td><td>11/16</td><td>9/16</td><td>9/16</td><td>100</td><td>9/16</td><td>16/16</td><td>15/16</td></tr><tr><td>150</td><td>一</td><td>一</td><td>一</td><td>150</td><td>9/16</td><td>13/16</td><td>8/16</td><td>150</td><td>9/16</td><td>16/16</td><td>16/16</td></tr><tr><td>200</td><td>一</td><td>一</td><td>一</td><td>200</td><td>11/16</td><td>15/16</td><td>12/16</td><td>200</td><td>11/16</td><td>16/16</td><td>16/16</td></tr><tr><td>250</td><td>一</td><td>一</td><td>一</td><td>250</td><td>12/16</td><td>16/16</td><td>9/16</td><td>250</td><td>12/16</td><td>16/16</td><td>16/16</td></tr><tr><td>300</td><td>一</td><td>一</td><td>一</td><td>300</td><td>11/16</td><td>16/16</td><td>10/16</td><td>300</td><td>10/16</td><td>16/16</td><td>16/16</td></tr><tr><td>350</td><td>一</td><td>一</td><td>1</td><td>350</td><td>10/16</td><td>16/16</td><td>9/16</td><td>350</td><td>12/16</td><td>16/16</td><td>16/16</td></tr><tr><td>400</td><td>一</td><td>一</td><td>一</td><td>400</td><td>9/16</td><td>16/16</td><td>8/16</td><td>400</td><td>11/16</td><td>16/16</td><td>16/16</td></tr><tr><td>450</td><td></td><td></td><td></td><td>450</td><td>10/16</td><td>16/16</td><td>9/16</td><td>450</td><td>14/16</td><td>16/16</td><td>16/16</td></tr><tr><td>500</td><td>16/16</td><td>5/16</td><td>2/16</td><td>500</td><td>8/16</td><td>16/16</td><td>7/16</td><td>500</td><td>12/16</td><td>16/16</td><td>16/16</td></tr><tr><td>550</td><td>一</td><td>一</td><td>一</td><td>550</td><td>10/16</td><td>16/16</td><td>8/16</td><td>550</td><td>14/16</td><td>16/16</td><td>16/16</td></tr><tr><td>600</td><td>一</td><td>一</td><td>一</td><td>600</td><td>10/16</td><td>16/16</td><td>8/16</td><td>600</td><td>13/16</td><td>16/16</td><td>16/16</td></tr><tr><td>650</td><td>1</td><td>一</td><td>/</td><td>650</td><td>11/16</td><td>16/16</td><td>9/16</td><td>650</td><td>11/16</td><td>16/16</td><td>16/16</td></tr><tr><td>700</td><td>一</td><td>一</td><td>一</td><td>700</td><td>8/16</td><td>16/16</td><td>12/16</td><td>700</td><td>13/16</td><td>16/16</td><td>16/16</td></tr></table>

Table 13: Reverse interference when re-training Q1 from the selected Q3@650 policy on Qwen3-VL-8B. The left block reports verifier-confirmed correct rollouts for Q1 and four previously mastered variants from the Q2/Q3- related problem family. The right block reports performance on DEV192 during Q4 training. Cell colors follow the same absolute rollout-accuracy scale as Tables 10–12. Re-learning Q1 restores its direct-solving performance, but simultaneously suppresses the previously mastered related family and degrades held-out development performance.
<table><tr><td></td><td colspan="6">Rollout Profile</td><td colspan="4">Held-out Development Performance</td></tr><tr><td>Step</td><td>Q1</td><td>Q3/base</td><td>420/xy</td><td>Q2/y</td><td>420/z</td><td>Family Total</td><td>MATH96</td><td>WE-MATH96</td><td>DEV192</td><td>Δ DEV</td></tr><tr><td>0 (Q3@650)</td><td>11/16</td><td>16/16</td><td>16/16</td><td>16/16</td><td>16/16</td><td>64/64</td><td>86.4583</td><td>69.7900</td><td>78.1242</td><td>0</td></tr><tr><td>25</td><td>10/16</td><td>6/16</td><td>4/16</td><td>6/16</td><td>6/16</td><td>22/64</td><td>81.2500</td><td>66.6700</td><td>73.9600</td><td>-4.1642</td></tr><tr><td>50</td><td>11/16</td><td>7/16</td><td>2/16</td><td>5/16</td><td>5/16</td><td>19/64</td><td>84.3750</td><td>65.6200</td><td>74.9975</td><td>-3.1267</td></tr><tr><td>75</td><td>11/16</td><td>9/16</td><td>4/16</td><td>5/16</td><td>7/16</td><td>25/64</td><td>84.3750</td><td>67.7100</td><td>76.0425</td><td>-2.0817</td></tr><tr><td>100</td><td>12/16</td><td>6/16</td><td>2/16</td><td>5/16</td><td>4/16</td><td>17/64</td><td>83.3333</td><td>63.5400</td><td>73.4367</td><td>-4.6875</td></tr><tr><td>125</td><td>16/16</td><td>10/16</td><td>3/16</td><td>6/16</td><td>5/16</td><td>24/64</td><td>81.2500</td><td>62.5000</td><td>71.8750</td><td>-6.2492</td></tr><tr><td>150</td><td>14/16</td><td>10/16</td><td>2/16</td><td>5/16</td><td>6/16</td><td>23/64</td><td>83.3333</td><td>64.5800</td><td>73.9567</td><td>-4.1675</td></tr><tr><td>175</td><td>14/16</td><td>7/16</td><td>3/16</td><td>6/16</td><td>4/16</td><td>20/64</td><td>84.3750</td><td>65.6200</td><td>74.9975</td><td>-3.1267</td></tr><tr><td>200</td><td>16/16</td><td>9/16</td><td>4/16</td><td>5/16</td><td>5/16</td><td>23/64</td><td>82.2917</td><td>67.7100</td><td>75.0008</td><td>-3.1233</td></tr></table>

Full-test shift at Q4@200: MATH500 86.80 → 84.80 (−2.00); WE-MATH testmini 73.51 → 72.13 (−1.38); Full Avg. 80.155 → 78.465 (−1.690).

Table 14: Rollout-budget accounting for the scheduling baselines. “Prompt slots” denotes the total number of training prompts processed before rollout expansion. Policy-Adaptive Sequential RLVR and Full-Batch GRPO are exactly matched at 33,600 rollout responses. Fixed-Sequential follows the original 700-step protocol and uses 11,200 rollouts.
<table><tr><td>Backbone</td><td>Method</td><td>Training Schedule</td><td>Batch</td><td>Rollout n</td><td>Prompt Slots</td><td>Total Rollouts</td></tr><tr><td>2B</td><td>Policy-Adaptive Sequential</td><td>3 × 700</td><td>1</td><td>16</td><td>2,100</td><td>33,600</td></tr><tr><td>2B</td><td>Fixed-Sequential</td><td>700</td><td>1</td><td>16</td><td>700</td><td>11,200</td></tr><tr><td>2B</td><td>Full-Batch GRPO</td><td>30 × 69 + 30</td><td>69 → 30</td><td>16</td><td>2,100</td><td>33,600</td></tr><tr><td>4B</td><td>Policy-Adaptive Sequential</td><td>3 × 700</td><td>1</td><td>16</td><td>2,100</td><td>33,600</td></tr><tr><td>4B</td><td>Fixed-Sequential</td><td>700</td><td>1</td><td>16</td><td>700</td><td>11,200</td></tr><tr><td>4B</td><td>Full-Batch GRPO</td><td>65 × 32 + 20</td><td>32 → 20</td><td>16</td><td>2,100</td><td>33,600</td></tr><tr><td>8B</td><td>Policy-Adaptive Sequential</td><td>3 × 700</td><td>1</td><td>16</td><td>2,100</td><td>33,600</td></tr><tr><td>8B</td><td>Fixed-Sequential</td><td>700</td><td>1</td><td>16</td><td>700</td><td>11,200</td></tr><tr><td>8B</td><td>Full-Batch GRPO</td><td>75 × 28</td><td>28</td><td>16</td><td>2,100</td><td>33,600</td></tr></table>