# RECURSE: BOUNDED RECURSIVE SELF-EVALUATION FOR LLM RUBRIC JUDGES

Kaiyuan Liu<sup>1,2</sup>, Ziyuan Zhuang<sup>2,\*</sup>, Rongxiang Weng<sup>2,†</sup>, Jieping Ye<sup>1</sup>

<sup>1</sup>College of Computer Science and Technology, Zhejiang University, <sup>2</sup>Meituan LongCat Team, China 12421281@zju.edu.cn

## ABSTRACT

LLM-as-judge is essential for evaluating open-ended text and steering posttraining, yet improving the judge itself typically relies on expensive human annotations, auxiliary reward models, or distillation from stronger teachers. In this work, we eliminate external gold supervision from the reinforcement learning (RL) training reward: the model’s own evaluative capability generates the learning signal for its optimization—a closed-loop regime of bounded recursive selfimprovement (RSI) that we term Recursive Self-Evaluation (RECURSE). We investigate two governing questions of this loop: when can self-improvement occur, and when must it stop? First, RECURSE employs a two-pass mechanism: a trainable judge evaluates candidate responses under per-rule rubrics (Pass 1), while a synchronized copy of the policy audits the judge’s reasoning against meta-rubrics to supply a scalar process reward (Pass 2). To make learning viable, interface decoupling structurally isolates the checker’s scalar score from the judge’s verdict tokens, closing a degenerative token-copying shortcut that inflates self-assigned rewards. Second, because unanchored recursive learning is inherently bounded, we introduce Pairwise Advantage Validity (PAV)—an unbiased validation monitor that jointly tracks judge accuracy and checker fidelity to reliably identify the optimal early-stopping window. Across Qwen3.5-9B, Gemma-4-E4B-it, and Qwen3.6-27B, RECURSE achieves consistent generalization gains across heldout medical, pairwise, summarization, and professional benchmarks. Extensive ablations show that synchronized judge–checker co-evolution outperforms frozen checkers, external meta-judges, self-consistency, and scaled teacher distillation. Furthermore, preference pairs curated by our judge effectively enhance downstream policy alignment. Bounded RSI for LLM-as-judge is thus feasible when self-produced reward validity is explicitly decoupled and monitored.

## 1 INTRODUCTION

AI systems generate open-ended text faster than humans can review. A scalable paradigm keeps humans over the loop—setting standards—while models execute and verify. This verification role requires a dedicated LLM judge: a model evaluating candidate responses under structured criteria (Zheng et al., 2023) to guide post-training. Improving the judge itself, however, typically relies on expensive human annotations, learned reward models, or distillation from stronger teachers— dependencies that are costly and circular. Furthermore, unlike mathematics or coding where dif ficulty is governed by query complexity alone, improving a judge requires calibration over two coupled variables: a discriminative rubric and a borderline candidate response. If a response is blatantly correct or flawed, the instance yields negligible corrective gradient. This dual-variable constraint makes static teacher distillation inefficient for scaling evaluative capability. Yet fundamentally, evaluating candidate responses and auditing evaluative reasoning share the same underlying capability: both instantiate rubric-based verification (Figure 1). This shared foundation suggests that the two roles can mutually reinforce each other in a closed loop.

This paper studies the closed-loop regime where the judge’s own capability produces the reward for its optimization updates. We formalize this bounded recursive self-improvement (RSI) framework as Recursive Self-Evaluation (RECURSE): a trainable judge emits reasoning and per-rule verdicts (Pass 1), while a synchronized, reward-only copy audits completed reasoning against meta-rubrics to supply group-relative scalar rewards (Pass 2). External gold labels and teacher models do not enter the RL training reward. Therefore, scaling annotation across the training set is thus avoided. In this closed recurrence, we investigate two governing questions: when can self-improvement occur, and when must it stop?

When can self-improvement occur? Sharing identical prompt templates and YES/NO readouts across passes induces surface coupling: policy updates that boost YES tokens on judge verdicts simultaneously alter emission probabilities in the synchronized checker, creating a degenerative token shortcut that inflates self-assigned rewards without improving evaluative accuracy. We resolve this via interface decoupling: the judge emits structured per-rule verdicts, while the checker emits a free-standing 5- tier scalar score (from 0 to 4), severing direct token copying while preserving positive capability transfer. Synchronizing model parameters between the judge and checker at each training step enables both roles to co-evolve productively.

![](images/9660f17c019790aa68fae1eaa820c29c02f9215535a28ca2625a0ce78262e2e6.jpg)

When must self-improvement stop? Without external ground-truth anchors on training rewards, recursive optimization is inherently bounded: selfproduced rewards eventually saturate, causing out-of-distribution transfer to degrade. Selfimprovement requires an early-stopping monitor computed from a compact, human-verified validation set whose labels never enter the RL reward. We address this with Pairwise Advantage Validity (PAV), a composite validation monitor evaluated on this holdout set. Motivated by a theoretical bound on pairwise score differences, PAV constructs a composite validation index that balances judge rule accuracy against checker ranking fidelity, with double weighting on checker fidelity to reflect its role

Figure 1: Overview of RECURSE. Trainable judge π<sub>θ</sub> evaluates responses under rubrics (Pass 1); synchronized checker C¯ audits reasoning against meta-rubrics to yield scalar process reward s (Pass 2). Policy parameters θ are updated via RL on s, and checker weights <sup>¯</sup>θ are synchronized each step $( \bar { \theta } _ { t + 1 }  \theta _ { t + 1 } )$

in pairwise comparison error. The estimator is unbiased at each fixed checkpoint, reliably localizing a useful stopping window before validation deception sets in.

Overview of experimental findings: connecting RQ1–RQ5. We evaluate RECURSE across five interconnected questions spanning viability, mechanisms, validation dynamics, baselines, and downstream transfer:

• Viability across scales and architectures (RQ1): Without gold RL rewards, RECURSE produces consistent generalization gains across held-out medical, pairwise, summarization, and professional benchmarks on Qwen3.5-9B (+12.9 points on the human-verified validation set), Gemma-4-E4B-it (+5.2 points), and Qwen3.6-27B (+3.9 points).

• Decoupling enables learning (RQ2): Shared YES/NO checking triggers reward inflation without accuracy gains, whereas scalar interface decoupling preserves token calibration and converts reward growth into genuine validation accuracy.

• PAV localizes effective stopping (RQ3): While hard-subset accuracy continues climbing late in training, checker fidelity degrades, exposing validation deception; PAV reliably identifies the optimal stopping region without peeking at transfer benchmarks.

• Judge–checker co-evolution is indispensable (RQ4): Synchronized weight tying outperforms frozen checkers, external meta-judges, and self-consistency. Furthermore, scaled teacher SFT (up to 17k traces) plateaus early because static corpora fail to maintain onpolicy dual-variable difficulty coupling.

• Downstream policy transfer (RQ5): Preference pairs generated by the retained judge drive alignment on Qwen-27B, outperforming base-judge labels across GPQA, GuideBench, and SOP-Maze.

Summary of contributions. (1) We formalize bounded recursive self-improvement for LLM judges via on-policy process checking without gold RL rewards; (2) We uncover the surfacecoupling shortcut and design interface decoupling to make closed-loop learning viable; (3) We derive the theoretically grounded PAV monitor to reliably localize the effective stopping window; (4) Across multiple model families and scales up to 27B, we demonstrate broad out-of-distribution generalization and effective downstream preference alignment.

## 2 RELATED WORK

LLM judges and rubric evaluation. LLM judges evaluate open-ended generations when reference-based metrics fall short (Zheng et al., 2023). Rubric-based evaluation decomposes subjective holistic assessments into structured, criterion-level decisions (JUDGE, 2024) or co-evolves generation policies alongside dynamic rubrics (Li et al., 2026a; Wang et al., 2026a). In contrast, we keep criteria fixed and optimize the criterion-level judge itself. Discrete per-rule verdicts and stepby-step reasoning (Figure 1) make evaluative correctness objectively measurable and enable process auditing of the judge’s reasoning. However, removing gold labels from RL rewards introduces a circularity challenge: the model must supervise its updates without exploiting self-generated reward shortcuts.

Self-improving evaluators and bounded RSI. Prior self-improving evaluators such as Self-Taught Evaluators, Meta-Rewarding, SELF-JUDGE, and Grad2Reward train models using generated supervision, but typically retain external anchors such as synthetic preference pairs, external meta-judges, or frozen reference models (Wang et al., 2024; Wu et al., 2025; Lee et al., 2024; Zhang et al., 2026b). In contrast, RECURSE removes external gold labels and teacher models from the optimization reward: the current judge acts simultaneously as the learning target and the source of its scalar reward, relying on a compact holdout solely for early stopping. We reserve bounded RSI for this fixed-task recurrence, distinct from open-ended autonomous self-modification (Chen et al., 2026).

Process auditing and reward validity. Process supervision and generative verification highlight the utility of auditing intermediate reasoning chains (Lightman et al., 2024; Zhang et al., 2025), though they typically require ground-truth correctness labels to train process reward models. In contrast, our process checker operates purely at inference time as a reward-only auditor without separate supervised training. Because naive self-verification risks reinforcing flawed reasoning and inflating self-assigned rewards (Stechly et al., 2025; Simonds et al., 2025; Rentschler & Roberts, 2026; Zhou, 2026), closed-loop rubric optimization imposes two requirements: first, the checker must not share the judge’s verdict readout surface so that self-improvement can occur; second, checker scores must preserve ranking fidelity so that training can be stopped before validity decays—principles formal ized by RECURSE and monitored by PAV (Section 3).

## 3 RECURSIVE SELF-EVALUATION (RECURSE)

RECURSE formalizes bounded recursive self-improvement for evaluators by leveraging the model’s own capability as both the optimization target and the reward source. Guided by the two governing questions from Section 1—when can self-improvement occur, and when must it stop?—we structure our method into three core components: (1) establishing a closed-loop judge–checker recurrence on a single shared policy (Section 3.1); (2) introducing interface decoupling to eliminate the surfacecoupling shortcut that otherwise impedes learning (Section 3.2); and (3) formulating Pairwise Advantage Validity (PAV) as a holdout early-stopping monitor (Section 3.3). Figure 2 and Algorithm 1 in Appendix B summarize the end-to-end framework.

![](images/e3819afba5fa117eca874057b4e65f6e78717909327404736c3490dd416bfe12.jpg)  
Figure 2: The RECURSE closed-loop optimization recurrence. Pass 1: trainable judge $\pi _ { \theta _ { t } }$ emits reasoning and per-rule YES/NO verdicts on $x = \left( h , y , r _ { 1 : K } \right)$ (n rollouts). Pass 2: a synchronized copy $C _ { \bar { \theta } _ { t } }$ audits reasoning against meta-rubrics and returns a differentiated scalar score $s _ { i } \in \{ 0 , \ldots , 4 \}$ as the sole reward. Interface decoupling closes direct token copying; the checker synchronizes parameters every step $( \bar { \theta } _ { t }  \theta _ { t } )$ without independent gradients. Online PAV monitoring on a human-verified validation set localizes early stopping.

## 3.1 JUDGE–CHECKER RECURRENCE

Shared evaluative capability as a closed recurrence. Because evaluating candidate responses and auditing evaluative reasoning share the same rubric-based foundation, a natural closed loop lets the current policy fulfill both roles: generate a rubric evaluation, and score that evaluation to guide the next policy update—without per-step gold supervision, reward models, or stronger teachers in the training reward.

Problem formulation and execution pipeline. A rubric instance is denoted $x = \left( h , y , r _ { 1 : K } \right)$ where h is task context, y is a candidate response, and $r _ { 1 : K }$ are $K$ predefined criteria. The judge policy $\pi _ { \theta _ { i } }$ emits reasoning and binary verdicts $s _ { k } \in \{ \mathrm { Y E S } , \mathrm { N O } \}$ for each criterion. Given n rollouts $z _ { t , i } \sim \pi _ { \theta _ { t } } ( \cdot \mid x )$ , a synchronized snapshot $C _ { \bar { \theta } _ { t } }$ (with $\bar { \theta } _ { t } \gets \theta _ { t } )$ audits the reasoning against metarubrics, assigning process reward $r _ { t , i } = C _ { \bar { \theta } _ { t } } ( x , ^ { \cdot } z _ { t , i } )$ . Parameters are updated as:

$$
\begin{array} { r } { \theta _ { t + 1 } = \mathrm { P o l i c y U p d a t e } ( \theta _ { t } , \{ z _ { t , i } , r _ { t , i } \} ) , \quad \bar { \theta } _ { t + 1 }  \theta _ { t + 1 } . } \end{array}\tag{1}
$$

External gold labels and teachers do not enter $r _ { t , i }$ . Pass 1 is the sole gradient path, while Pass 2 audits reasoning against meta-rubrics in a reward-only capacity (Appendix I). Synchronous parameter refreshes transfer updated judge capability into the auditing role without separate training. Auditing the reasoning process rather than re-judging candidates ensures spurious reasoning is penalized even if binary verdicts happen to be correct.

## 3.2 INTERFACE DECOUPLING: WHEN SELF-IMPROVEMENT CAN OCCUR

Shared templates induce surface coupling. To facilitate capability transfer, an intuitive initial design is to share identical prompt templates and YES/NO readouts across both passes. Under this coupled setup, the checker emits per-rule binary verdicts for meta-rubrics, and reward is extracted as $\begin{array} { r } { \bar { S } = \frac { 1 } { K } \bar { \sum _ { k = 1 } ^ { K } } } \end{array}$ 1(Checker<sub>k</sub> = YES). However, policy gradient optimization rapidly exploits a degenerative shortcut: inflating YES tokens across criteria directly maximizes self-assigned rewards without improving evaluative capability. Formally, we decompose the self-produced checker score $S$ as:

$$
S = a U + b B + \varepsilon ,\tag{2}
$$

where U denotes true reasoning quality, B represents surface verdict token bias $( \mathrm { e . g . }$ ., propensity to emit YES), and ε is noise. The coefficients $a , b \geq 0$ govern how genuine quality versus surface bias propagate into reward. When judge and checker share identical readout interfaces, synchronized weights $( \bar { \theta } _ { t + 1 }  \theta _ { t + 1 } )$ establish a closed-circuit shortcut channel $( b > 0 )$ : gradient updates that boost YES emissions on judge verdicts simultaneously alter emission probabilities in the synchronized checker, allowing the optimizer to exploit bB to harvest near-perfect rewards $( S \to 1 . \dot { 0 } )$ while true evaluative accuracy U stalls (visualized in Appendix B.1).

Decoupled readout design. To close this shortcut while preserving capability transfer, RECURSE retains shared evidence layouts in input prompts (Appendix I) but structurally decouples output interfaces. While the judge outputs step-by-step explanations followed by indexed verdicts $( \mathrm { e . g . }$ $\mathtt { 1 } : < \mathtt { a n s } > \mathtt { Y E S } < / \mathtt { a n s } > ) .$ , the checker audits the complete reasoning trajectory against meta-rubrics and emits a free-standing 5-tier scalar score $s _ { i } \in \{ 0 , 1 , 2 , 3 , 4 \}$ . The absolute numeric range is immaterial: group-relative advantages are invariant to positive affine transforms of R, so any reward inducing valid relative orderings among the n rollouts remains an effective training signal.

Severing the direct token channel. Interface decoupling removes the direct token-copying channel, substantially reducing bB in Eq. 2. We do not claim $b = 0 ,$ , nor that scalar readouts preclude every shortcut; rather, the scalar interface is an empirically effective design. As shown in Section 5, coupled YES/NO checking inflates surface YES rates and rewards without accuracy gains, whereas the scalar Final Score preserves token calibration and converts reward growth into genuine accuracy improvements (Figure 3). Decoupling is thus a vital prerequisite for viable self-improvement.

## 3.3 OPTIMIZATION AND PAIRWISE ADVANTAGE VALIDITY: WHEN SELF-IMPROVEMENT MUST STOP

Group-relative optimization. With the interface decoupled, well-formed rollouts receive reward $R _ { i } = s _ { i } .$ while malformed outputs are masked from group statistics. For each prompt, we sample n rollouts, compute group-relative advantages (Shao et al., 2024), and optimize using a sequencelevel policy loss (Chujie, 2025). Because advantages are normalized within each prompt group, optimization relies strictly on within-prompt rollout rankings.

Viable improvement is inherently bounded. Because the loop operates without external groundtruth anchors on training rewards, recursive optimization cannot continue indefinitely once selfgenerated rewards or ranking fidelities saturate. Optimizing beyond the reliable reward regime degrades out-of-distribution transfer. We therefore construct Pairwise Advantage Validity (PAV) as an online holdout monitor on a compact, human-verified validation set $\mathcal { D } _ { \mathrm { v a l } }$ . Constructing this holdout incurs a one-time human verification cost, but its labels never enter RL training rewards.

A pairwise bound on score-difference error. Let $A _ { t } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { v a l } } } [ Y _ { t } ]$ be judge rule accuracy on $\mathcal { D } _ { \mathrm { v a l } }$ $( \bar { Y _ { t } } \in \{ 0 , 1 \} )$ . Let $S _ { t } \in [ 0 , 4 ]$ denote the self-produced checker score and $U _ { t } \in [ 0 , 4 ]$ its humanverified ground-truth target, with normalized checker error defined as:

$$
e _ { C , t } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { v a l } } } \bigg [ \frac { | S _ { t } - U _ { t } | } { 4 } \bigg ] .\tag{3}
$$

Group-relative advantages apply a positive affine transform within each prompt, preserving the relative ordering of the n scores. The critical quantity that must remain faithful is the score difference $( S _ { i } - S _ { j } )$ rather than an isolated absolute score. By the triangle inequality:

$$
\frac { | ( S _ { i } - S _ { j } ) - ( U _ { i } - U _ { j } ) | } { 4 } \leq \frac { | S _ { i } - U _ { i } | } { 4 } + \frac { | S _ { j } - U _ { j } | } { 4 } .\tag{4}
$$

Taking expectations over exchangeable rollout pairs, the expected pairwise ranking error is bounded by $2 e _ { C , t } ,$ directly motivating continuous monitoring of checker fidelity.

Constructing the PAV monitor and early stopping. To obtain a unified scalar metric reflecting both judge correctness and checker fidelity, we construct the Pairwise Advantage Validity index:

$$
V _ { t } = 1 - \frac { ( 1 - A _ { t } ) + 2 e _ { C , t } } { 3 } = \frac { A _ { t } + 2 ( 1 - e _ { C , t } ) } { 3 } .\tag{5}
$$

The relative weight of 2 on checker error directly reflects the bound in Eq. 4 and is fixed a priori. Both $\widehat { A } _ { t }$ and $\widehat { e } _ { C , t }$ are sample means on $\mathcal { D } _ { \mathrm { v a l } }$ , ensuring that $\mathbb { E } [ \widehat { V } _ { t } ] = V _ { t }$ is unbiased at every fixed checkpoint. We compute 95% prompt-bootstrap intervals (Appendix N). A drop in checker fidelity signals uncalibrated pairwise rewards; conversely, a high $\widehat { V } _ { t }$ localizes an effective checkpoint region, preventing validation deception where same-style accuracy rises while cross-style transfer degrades.

![](images/828695021376dcc0a881f731e0e67b075aaa6464319ecf99a2d1294e536b0a62.jpg)  
Figure 3: Interface decoupling on SV-HARD (RQ2). Comparing identical YES/NO checking against differentiated Final Score (steps 10–100). (a) Identical checking inflates YES rate. (b) Both report climbing rewards ([0, 1] scale). (c) Only the differentiated Final Score converts reward growth into genuine validation accuracy gains.

## 4 EXPERIMENTAL SETUP

Models and training. We evaluate RECURSE across three setups: Qwen3.5-9B (Qwen Team, 2026) (main runs/ablations), Gemma-4-E4B-it (Team et al., 2026) (cross-architecture replication), and Qwen3.6-27B (Qwen Team, 2026) (scaling to 27B). Training instances are cluster-split from RubricHub (Li et al., 2026b) with responses synthesized via a 16-model three-tier pool (Appendix D), with zero evaluation overlap. Ground-truth labels never enter RL rewards. Training uses verl (Sheng et al., 2025) with FSDP and vLLM (Kwon et al., 2023): batch size 32 (48 for 27B), n=8 rollouts, lr 3×10<sup>−6</sup>, clipping 0.005, and no KL penalty (0.005 for 27B). The checker is a standalone vLLM snapshot synchronized each step (Appendix Q).

Three-tier evaluation hierarchy. We organize evaluation into a three-tier hierarchy spanning online monitors and out-of-distribution transfer suites (Table 1): (1) SV-HARD (100 prompts; Appendix E): A compact human-verified indomain monitor on hard rules (2.5 rules/prompt) supplying gold for checker fidelity (1 − NMAE) and serving as the sole stream for online PAV

Table 1: Held-out transfer suite. SV-HARD and SV-FULL serve as separate online monitors.
<table><tr><td>Benchmark</td><td>#</td><td>Rules</td><td>Metric</td></tr><tr><td>HealthBench</td><td>1459</td><td>2.0</td><td>rule acc ↑</td></tr><tr><td>RubricBench</td><td>2294</td><td>5.8</td><td>pair acc ↑</td></tr><tr><td>CheckEval-Summ</td><td>1600</td><td>15</td><td>Pearson ρ ↑</td></tr><tr><td>ProfBench</td><td>120</td><td>29.1</td><td>rule acc ↑</td></tr></table>

early-stopping monitoring. (2) SV-FULL (100 prompts; Appendix F): Scores the full per-prompt rule suite under the same judging style to probe complete in-domain coverage. (3) Held-out transfer suite (Table 1): Evaluated offline at retained checkpoints across distinct domains and rule granularities: HealthBench (Arora et al., 2025) evaluates clinical communication under samplespecific medical rubrics (a stratified 10% representative sample of 1,459 prompts across 34 clinical categories; measured by per-rule accuracy against physician consensus); RubricBench (Zhang et al., 2026a) tests multi-criteria preference discrimination under sample-specific rubrics (5.8 rules/prompt; measured by pairwise preference accuracy); CheckEval-Summ (Lee et al., 2025) assesses document summarization under a shared global 15-criterion checklist (unlike the other sample-specific suites; measured by Pearson ρ against continuous human quality ratings); and Prof-Bench (Wang et al., 2026c) evaluates complex professional tasks with sample-specific expert rubrics (≈29 tight rules/prompt; measured by fine-grained rule accuracy; Appendix K).

Ablations and research questions. All ablations use Qwen3.5-9B, comparing against five controls: (1) a frozen base checker; (2) an external 27B checker; (3) self-consistency majority reward; (4) matched-step teacher SFT (6400 traces, 200 steps); and (5) larger-pool teacher SFT on the full distillation set (≈17k traces, two epochs). Gemma-4-E4B-it provides cross-architecture replication (Appendix M). For downstream transfer, preference pairs generated by our retained RECURSE judge versus its untuned base counterpart train matched Qwen-27B policies using DPO (Rafailov et al., 2023). We evaluate across five research questions: viability across scales and architectures (RQ1), interface decoupling (RQ2), PAV early-stopping localization (RQ3), judge–checker co-evolution (RQ4), and downstream policy transfer (RQ5).

## 5 RESULTS

## 5.1 RQ1: BOUNDED RSI IMPROVES THE JUDGE

Bounded RSI consistently elevates judge accuracy across formats without requiring gold labels or external teachers in the RL reward (Table 2; detailed definitions of rule accuracy R, all-correct

Table 2: Held-out results (%; CheckEval: Pearson $\rho ) .$ . R/All/M: rule/all-correct/macro accuracy (Appendix J). Bold marks designated Best within the PAV-localized validity window across each architecture. Final denotes the training endpoint. SV-FULL reports rule accuracy only.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Ckpt.</td><td colspan="3">SV-HARD</td><td>SV-FULL</td><td colspan="4">HealthBench</td><td>RubricBench</td><td>CheckEval</td><td colspan="3">ProfBench</td></tr><tr><td>R</td><td>All</td><td>M</td><td>R</td><td>R</td><td>All</td><td>M</td><td>F1</td><td>Pair acc</td><td>ρ</td><td>R</td><td>All</td><td>M</td></tr><tr><td rowspan="3">Qwen3.5-9B</td><td>Base</td><td>60.8</td><td>33.1</td><td>64.7</td><td>92.1</td><td>82.5</td><td>74.9</td><td>82.8</td><td>65.4</td><td>76.0</td><td>0.422</td><td>73.1</td><td>3.8 73.0</td><td></td></tr><tr><td>Best</td><td>73.7</td><td>53.6</td><td>77.6</td><td>95.3</td><td>85.9</td><td>79.0</td><td>86.5</td><td>66.3</td><td>79.1</td><td>0.441</td><td>74.7</td><td>5.8 74.5</td><td></td></tr><tr><td>Final</td><td>75.9</td><td>61.0</td><td>80.7</td><td>89.7</td><td>82.9</td><td>74.0</td><td>82.8</td><td>63.4</td><td>77.8</td><td>0.345</td><td>71.5</td><td>4.2</td><td>71.7</td></tr><tr><td rowspan="3">Qwen3.6-27B</td><td>Base</td><td>73.1</td><td>54.5</td><td>77.6</td><td>94.2</td><td>85.5</td><td>77.7</td><td>84.9</td><td>67.1</td><td>78.6</td><td>0.521</td><td>74.9</td><td>4.2</td><td>74.0</td></tr><tr><td>Best</td><td>77.0</td><td>63.0</td><td>82.1</td><td>96.0</td><td>86.6</td><td>79.4</td><td>86.5</td><td>68.2</td><td>80.6</td><td>0.548</td><td>76.4</td><td>6.9</td><td>75.7</td></tr><tr><td>Final</td><td>75.6</td><td>58.7</td><td>80.2</td><td>95.3</td><td>86.5</td><td>78.8</td><td>86.4</td><td>68.1</td><td>81.9</td><td>0.548</td><td>75.1</td><td>4.8</td><td>74.6</td></tr><tr><td rowspan="3">Gemma-4-E4B-it Best</td><td>Base</td><td>55.9</td><td>16.1</td><td>58.9</td><td>92.5</td><td>82.8</td><td>73.2</td><td>82.9</td><td>61.0</td><td>74.9</td><td>0.442</td><td>73.1</td><td>2.9</td><td>73.2</td></tr><tr><td></td><td>61.1</td><td>31.0</td><td>65.1</td><td>92.3</td><td>86.5</td><td>79.3</td><td>86.8</td><td>60.5</td><td>76.6</td><td>0.466</td><td>74.0</td><td></td><td>4.074.0</td></tr><tr><td>Final</td><td>59.6</td><td>27.3</td><td>63.5</td><td>92.5</td><td>85.6</td><td>78.1</td><td>85.8</td><td>59.7</td><td>76.7</td><td>0.463</td><td></td><td></td><td>73.2 4.4 73.4</td></tr></table>

rate All, and macro accuracy M are in Appendix J). Designated Best checkpoints correspond to the peak validity windows localized by PAV. On Qwen-9B, SV-HARD rule accuracy rises by 12.9 points, SV-FULL by 3.2, and all four out-of-distribution transfer benchmarks improve at the retained checkpoint. Gemma replicates this positive trajectory on SV-HARD (+5.2) and across transfer suites, while Qwen-27B achieves a 3.9-point lift on SV-HARD and gains across every transfer suite. Final checkpoints soften or drop on transfer metrics, confirming that the improvement window is inherently bounded.

Generalized lift vs. validation deception. On Qwen-9B, Final SV-HARD accuracy (75.9%) exceeds Best (73.7%), while SV-FULL falls from 95.3% to 89.7% and out-of-distribution metrics deteriorate by step 200. PAV balances same-style judge accuracy against checker fidelity rather than naively chasing SV-HARD accuracy alone, identifying the optimal stopping region without inspecting transfer suites. Transfer benchmarks achieve consistent positive deltas over Base within this PAV-indicated window (Table 2). SV-FULL peaks at the same checkpoint (95.3% vs. 92.1%), and Gemma replicates this pattern (Appendix M). Gains across ProfBench and CheckEval confirm robust generalization under dense expert constraints (≈29 weighted criteria) and continuous human correlation metrics (Pearson ρ).

## 5.2 RQ2: INTERFACE DECOUPLING

Sharing identical YES/NO formats fails due to surface coupling; interface decoupling closes this shortcut and enables reward and accuracy to rise concordantly (Figure 3). Holding evidence and synchronization constant, altering the readout from indexed YES/NO tokens to a freestanding scalar Final Score breaks token copying. Under identical YES/NO checking, YES rate inflates (0.730 → 0.791) and aligned reward climbs (0.465 → 0.698), yet validation accuracy plateaus (0.729 → 0.730). Under the differentiated Final Score, YES rate remains calibrated $( 0 . 7 0 7  0 . 6 9 5 )$ while aligned reward (0.484 → 0.656) and validation accuracy (0.724 → 0.787) climb in tandem, confirming decoupling as a prere

![](images/d412f74a9bf8d4c8019bd854296a51462bfbe4b2830545a7601e6ad11ec5f0bc.jpg)  
Figure 4: PAV on SV-HARD (RQ3). Judge accuracy $\widehat { A } _ { t } ,$ checker fidelity $1 - \widehat { e } _ { C , t } ,$ , and composite $\widehat { V } _ { t }$ (dashed; 95% CI). Dotted line denotes the PAV landmark (step 130).

## 5.3 RQ3: PAV LOCALIZES THE USEFUL CHECKPOINT

Using solely the fixed SV-HARD monitor, PAV reliably localizes an effective stopping region centered near step 130 on Qwen-9B (Figure 4), achieving $\widehat { V } _ { t } = 0 . 7 4 2$ (95% bootstrap CI [0.685, 0.797]). While Qwen’s hard-subset accuracy continues climbing through step 200, checker fidelity peaks sharply near step 130 and degrades thereafter (Figure 4), explaining why tracking rule accuracy alone is insufficient. Three-fold resampling confirms checkpoint stability inside {130, 140} (Ap-

Table 3: Ablation on Qwen3.5-9B (Peak/Final; SFT single-endpoint). Main beats static checkers and teacher SFT.
<table><tr><td>Arm</td><td>SV-HARD</td><td>Health</td><td>Rubric</td><td>Check</td></tr><tr><td>Base</td><td>60.8</td><td>82.5</td><td>76.0</td><td>0.422</td></tr><tr><td>Main</td><td>73.7/75.9</td><td></td><td></td><td>85.9/82.979.1/77.80.441/0.345</td></tr><tr><td>Frozen checker</td><td>62.7/61.3</td><td></td><td></td><td>85.0/76.076.5/77.80.442/0.454</td></tr><tr><td>External 27B</td><td>73.1/72.4</td><td>79.8/79.7</td><td></td><td>79.9/79.40.471/0.459</td></tr><tr><td>Self-consistency</td><td>65.3/56.5</td><td>82.5/83.5</td><td></td><td>573.1/73.00.447/0.435</td></tr><tr><td>Teacher SFT (6.4k)</td><td>64.2</td><td>81.0</td><td>76.2</td><td>0.329</td></tr><tr><td>Teacher SFT (17k)</td><td>70.2</td><td>83.5</td><td>77.1</td><td>0.394</td></tr></table>

Table 4: Downstream tasks (Qwen3.6- 27B). RECURSE preference labels outperform untuned base-judge labels on GPQA (Rein et al., 2023), GuideBench (Diao et al., 2025), and SOP-Maze (Wang et al., 2026b).

<table><tr><td>Model</td><td></td><td>GPQA GuideBench SOP-Maze</td><td></td></tr><tr><td>Qwen3.6-27B</td><td>80.87</td><td>83.88</td><td>34.34</td></tr><tr><td>+ DPO (base)</td><td>79.02</td><td>83.93</td><td>34.95</td></tr><tr><td>+ DPO (RECURSE)</td><td>81.46</td><td>85.51</td><td>36.71</td></tr></table>

pendix N). Crucially, SV-FULL (excluded from PAV) independently corroborates this stopping boundary: it peaks at the same landmark (95.3%) and falls to 89.7% by training conclusion. Heldout suites improve over Base at PAV landmarks, confirming that PAV reliably captures an effective stopping state without inspecting transfer data.

## 5.4 RQ4: JUDGE–CHECKER CO-EVOLUTION MATTERS

Ablations confirm that weight-tied co-evolution and structured process auditing are essential for sustaining self-improvement (Table 3). A frozen base checker plateaus early (SV-HARD 62.7%). An external 27B checker remains competitive on SV-HARD (73.1%) but underperforms suite-wide (HealthBench 79.8% vs. Main 85.9%), showing static scale cannot substitute for co-evolution. Selfconsistency collapses over training (56.5% Final). Matched-step teacher SFT (6400 traces) reaches only 64.2% on SV-HARD; expanding distillation to the full eligible set (≈17k traces, two epochs) reaches 70.2%, still trailing Main across every held-out transfer benchmark (HealthBench 83.5% vs. 85.9%, CheckEval 0.394 vs. 0.441). Main surpasses finished matched-step SFT by step 30.

Why teacher distillation struggles: dual-variable difficulty coupling. The failure of scaled teacher SFT exposes a structural difference between judge training and task distillation (e.g., in math or coding). In single-variable domains, learning primarily demands challenging queries. In rubric judging, however, an informative gradient requires simultaneous calibration over two coupled variables: (1) a discriminative rubric, and (2) a candidate response exhibiting borderline compliance near the model’s decision boundary. In static offline corpora (even across ≈17k teacher trajectories), fixed $\left( h , y , r _ { 1 : K } \right)$ instances inevitably degenerate into trivial samples with negligible corrective gradient. In contrast, RECURSE operates strictly on-policy: $\pi _ { \theta _ { t } }$ samples rollouts along its active decision boundary while the synchronized checker audits reasoning differences via grouprelative advantages, sustaining dual-variable tension throughout training.

## 5.5 RQ5: TRANSFER TO DOWNSTREAM TASKS

Evaluative gains from bounded RSI transfer directly to downstream policy alignment via preference labeling (Table 4). Preference pairs generated by our retained RECURSE judge versus its base counterpart train matched Qwen3.6-27B policies using DPO (LoRA; Appendix Q). While basejudge labels degrade GPQA (80.87 → 79.02) and yield negligible shifts on GuideBench (+0.05) and SOP-Maze (+0.61), RECURSE labels achieve consistent gains across all three benchmarks (+0.59 on GPQA, +1.63 on GuideBench, +2.37 on SOP-Maze), confirming that evaluative gains translate to downstream policy alignment.

## 6 CONCLUSION

RECURSE establishes that bounded recursive self-improvement (RSI) for LLM rubric judges is feasible when self-produced reward validity is explicitly decoupled and monitored. Grounded in the shared evaluative foundation of judging and auditing, interface decoupling closes degenerative surface shortcuts, while Pairwise Advantage Validity (PAV) reliably localizes an effective stopping region without external supervision in RL rewards. Across architectures and scales up to 27B, the resulting judges achieve robust out-of-distribution generalization and generate preference pairs that enhance downstream policy alignment (Appendix T).

## REFERENCES

Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quinonero-Candela,˜ Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

Mingguang Chen, Licheng Wang, and Bo Qu. Recursive self-improvement in ai: From bounded self-refinement to autonomous research loops. arXiv preprint arXiv:2607.07663, 2026.

Zheng Chujie. Group sequence policy optimization. arXiv preprint, 2025.

Lingxiao Diao, Xinyue Xu, Wanxuan Sun, Cheng Yang, and Zhuosheng Zhang. Guidebench: Benchmarking domain-oriented guideline following for llm agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11361–11399, 2025.

Alexander R Fabbri, Wojciech Krysci ´ nski, Bryan McCann, Caiming Xiong, Richard Socher, and ´ Dragomir Radev. Summeval: Re-evaluating summarization evaluation. Transactions of the Associationfor Computational Linguistics, 9:391–409, 2021.

AAFG JUDGE. Self-rationalization improves llm. arXiv preprint arXiv:2410.05465, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pp. 611–626, 2023.

Sangkyu Lee, Sungdong Kim, Ashkan Yousefpour, Minjoon Seo, Kang Min Yoo, and Youngjae Yu. Aligning large language models by on-policy self-judgment. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11442–11459, 2024.

Yukyung Lee, Joonghoon Kim, Jaehee Kim, Hyowon Cho, Jaewook Kang, Pilsung Kang, and Najoung Kim. Checkeval: A reliable llm-as-a-judge framework for evaluating text generation using checklists. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 15771–15798, 2025.

Shuyue Stella Li, Rui Xin, Teng Xiao, Yike Wang, Rulin Shao, Zoey Hao, Melanie Sclar, Sewoong Oh, Faeze Brahman, Pang Wei Koh, et al. Evolm: Self-evolving language models through coevolved discriminative rubrics. arXiv preprint arXiv:2605.03871, 2026a.

Sunzhu Li, Jiale Zhao, Huimin Ren, Zhenlin Wei, Yang Zhou, Jingwen Yang, Shunyu Liu, Kaike Zhang, and Chen Wei. Rubrichub: A comprehensive and highly discriminative rubric dataset via automated coarse-to-fine generation. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 31320–31344, 2026b.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. 2024:39578– 39601, 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. volume 36, pp. 53728–53741, 2023.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

Micah Rentschler and Jesse Roberts. Reinforcement learning from meta-evaluation: Aligning language models without ground-truth labels. arXiv preprint arXiv:2601.21268, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathemati cal reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Toby Simonds, Kevin Lopez, Akira Yoshiyama, and Dominique Garmier. Rlsr: Reinforcement learning from self reward. arXiv preprint arXiv:2505.08827, 2025.

Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati. On the self-verification limitations of large language models on reasoning and planning tasks. 2025:98190–98243, 2025.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026.

Beining Wang, Weihang Su, Hongtao Tian, Hao Kong, Tao Yang, Ting Yao, Qingyi Pan, Yueyue Wu, Qingyao Ai, Min Zhang, et al. Co-evolving llm evaluators and policies via dynamicrubric. arXiv preprint arXiv:2607.20083, 2026a.

Jiaming Wang, Zhe Tang, Zehao Jin, Hefei Chen, Yilin Jin, Peng Ding, Xiaoyu Li, and Xuezhi Cao. Sop-maze: Evaluating large language models on complicated business standard operating procedures. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 14568– 14588, 2026b.

Tianlu Wang, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. Self-taught evaluators. arXiv preprint arXiv:2408.02666, 2024.

Zhilin Wang, Jaehun Jung, Ximing Lu, Shizhe Diao, Ellie Evans, Jiaqi Zeng, Pavlo Molchanov, Yejin Choi, Jan Kautz, and Yi Dong. Profbench: Multi-domain rubrics requiring professional knowledge to answer and judge. 2026:5355–5384, 2026c.

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason E Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 11548–11565, 2025.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Seyed Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. 2025:12476–12505, 2025.

Qiyuan Zhang, Junyi Zhou, Yufei Wang, Fuyuan Lyu, Yidong Ming, Can Xu, Qingfeng Sun, Kai Zheng, Peng Kang, Xue Liu, et al. Rubricbench: Aligning model-generated rubrics with human standards. 2026a.

Zheng Zhang, Ao Lu, Yuanhao Zeng, Ziwei Shan, Jinjin Guo, Lufei Li, Yexin Li, and Kan Ren. Grad2reward: From sparse judgment to dense rewards for improving open-ended llm reasoning. arXiv preprint arXiv:2602.01791, 2026b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. volume 36, pp. 46595–46623, 2023.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pp. 400–410, 2024.

Chenyu Zhou. More convincing, not more correct: Self-play reward hacking of reference-free llm judges. arXiv preprint arXiv:2607.05904, 2026.

## A APPENDIX ROADMAP

The main text keeps the 8-page narrative on the two governing questions—when self-improvement can occur, and when it must stop—together with the primary tables and figures for RQ1–RQ5. This appendix collects the supporting material: full algorithms, mathematical formulations, construction protocols, prompts, extra diagnostics, systems comparisons, taxonomy tables, and robustness checks. The sections below map onto the corresponding main-text claims.

• Appendix B (Section 3): Algorithm 1 of the closed-loop training recurrence, group-relative policy gradient objectives, and the structural causal formulation of surface coupling (Appendix B.1).

• Appendix C (Sections 1 and 5, RQ4): concrete exemplars and structural analysis of dualvariable difficulty coupling versus single-variable task distillation.

• Appendix D (Section 4): candidate response synthesis protocol on RubricHub, detailing the 16-model three-tier pool and the balanced 1:2:1 allocation.

• Appendices E–F (Section 4): construction of the two online monitors—human-verified SV-HARD used by PAV, and complementary SV-FULL scoring the full in-domain rule suite.

• Appendix G (Section 4, RQ4): definitions of the five Qwen-9B controls (frozen checker, external 27B checker, self-consistency, and the two teacher-SFT pools).

• Appendices H–I (Sections 3.1–3.3): format gate, shared evidence-block layout, and the decoupled judge/checker prompt templates.

• Appendices J–K (Section 4, Table 2): metric definitions and per-benchmark characteristics for the held-out transfer suite.

• Appendix L (RQ1, RQ3): Qwen-9B judge/checker diagnostic grids on SV-HARD and HealthBench, showing why rule accuracy alone is an insufficient stopping signal.

• Appendix M (RQ1, RQ3): cross-architecture replication on Gemma-4-E4B-it—finegrained judge and checker diagnostic dynamics on SV-HARD and HealthBench.

• Appendix N (Section 3.3, RQ3): PAV unbiasedness, sequential monitoring vs. pointwise estimation, bootstrap intervals, fold stability, and the offline selector comparison.

• Appendices O–P (RQ3, RQ4): systems and cost-performance trade-offs relative to commercial APIs and standalone RMs, plus format-compliance dynamics.

• Appendix Q (Sections 4 and 5): RL and downstream DPO hyperparameters used for the main runs and RQ5 transfer.

• Appendix R (Sections 1 and 2): systematic taxonomy and multi-dimensional comparison against concurrent self-improving evaluators.

• Appendix S (Section 6): broader societal impact, automation bias mitigation, and the \*Humans-over-the-loop\* safety paradigm.

• Appendix T (Conclusion): expanded limitations and open questions summarized at the end of the main text.

## B ALGORITHM AND METHOD DETAILS

Group-relative advantage computation and policy objective. In each training iteration $t ,$ for a given rubric instance $x = \left( h , y , r _ { 1 : K } \right)$ , the judge policy $\pi _ { \theta _ { t } }$ generates n rollouts $\{ \bar { z } _ { t , i } \} _ { i = 1 } ^ { n }$ . The synchronized checker copy $C _ { \bar { \theta } _ { t } }$ audits each rollout to emit a scalar score $s _ { t , i } \in \{ 0 , 1 , 2 , 3 , 4 \}$ . Outputs that pass the format gate form the valid index set $\mathcal { V } _ { t } \subseteq \{ 1 , \dots , n \}$ with reward $R _ { t , i } = s _ { t , i }$ . Group mean and standard deviation are computed strictly within the prompt group:

$$
\mu _ { t } = \frac { 1 } { | \mathcal { V } _ { t } | } \sum _ { i \in \mathcal { V } _ { t } } R _ { t , i } , \quad \sigma _ { t } = \sqrt { \frac { 1 } { | \mathcal { V } _ { t } | } \sum _ { i \in \mathcal { V } _ { t } } ( R _ { t , i } - \mu _ { t } ) ^ { 2 } } + \epsilon _ { \mathrm { s t d } } ,\tag{6}
$$

where $\epsilon _ { \mathrm { s t d } } = 1 0 ^ { - 6 }$ prevents numerical division by zero. The group-relative advantage for each valid rollout is normalized as:

$$
\widehat { A } _ { t , i } = \frac { R _ { t , i } - \mu _ { t } } { \sigma _ { t } } .\tag{7}
$$

Malformed rollouts $( i \notin \mathcal { V } _ { t } )$ are masked out with zero advantage. The policy $\pi _ { \theta }$ is updated by optimizing the clipped sequence-level policy loss:

$$
\mathcal { L } _ { \mathrm { R L } } ( \theta ) = - \frac { 1 } { \left| \mathcal { V } _ { t } \right| } \sum _ { i \in \mathcal { V } _ { t } } \operatorname* { m i n } \left( \frac { \pi _ { \theta } ( z _ { t , i } \mid x ) } { \pi _ { \theta _ { t } } \left( z _ { t , i } \mid x \right) } \widehat { A } _ { t , i } , \ \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( z _ { t , i } \mid x ) } { \pi _ { \theta _ { t } } \left( z _ { t , i } \mid x \right) } , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } \right) \widehat { A } _ { t , i } \right) ,\tag{8}
$$

with clipping threshold $\epsilon _ { \mathrm { c l i p } } = 0 . 0 0 5$ . Crucially, advantages $\widehat { A } _ { t , i }$ are completely invariant under any affine transformation $R \mapsto { \dot { \alpha } } R + \beta \left( \alpha > 0 \right)$ . Therefore, the absolute numeric range of the checker’s 5-tier score is immaterial; what determines the policy gradient is purely the relative ranking induced among the n on-policy rollouts.

Algorithm 1 RECURSE training loop with periodic held-out validity monitoring   
Require: Rubric instances D, initial judge policy $\pi _ { \theta _ { 0 } }$ , monitoring interval $T = 1 0 ,$ , fixed human  
verified monitor H, training budget B   
1: $\bar { \theta } _ { 0 }  \theta _ { 0 }$ {reward-only checker copy}   
2: $V ^ { \star }  - \infty ; t ^ { \star }  0$   
3: for $t = 0 , 1 , 2 , \ldots , B$ do   
4: Sample $v \sim \mathcal { D } ;$ roll out $z _ { t , i } \sim \pi _ { \theta _ { t } } ( \cdot \mid x ) { \mathrm { f o r } } i = 1 , \dots , n$   
5: $s _ { t , i } \gets C _ { \bar { \theta } _ { t } } ( x , z _ { t , i } )$ {Pass 2: process check}   
6: $R _ { t , i } \gets s _ { t , i }$ if format-valid else 0; mask invalid from group statistics   
7: $\theta _ { t + 1 }  \mathrm { R L U p d a t e } ( \theta _ { t } , \{ ( z _ { t , i } , R _ { t , i } ) \} ) ; \ \bar { \theta } _ { t + 1 }  \theta _ { t + 1 }$   
8: if t mod $T = 0$ then   
9: Compute $\widehat { A } _ { t } , \widehat { e } _ { C , t }$ , and PAV $\widehat { V } _ { t }$ on H   
10: if $\widehat { V } _ { t } > V ^ { \star }$ then   
11: Save checkpoint; $V ^ { \star }  \widehat { V } _ { t } ; \ t ^ { \star }  t$   
12: end if   
13: end if   
14: end for   
15: return checkpoint $t ^ { \star }$

## B.1 SURFACE COUPLING: STRUCTURAL CAUSAL FORMULATION AND VISUAL ILLUSTRATION

Structural Causal Model (SCM) of surface coupling. To formalize the shortcut channel described by $S = a U + b B + \varepsilon { \mathrm { ~ ( E q . ~ } } 2 )$ , consider the structural causal relations governing reward extraction under parameter synchronization $( { \bar { \theta } } _ { t + 1 }  \theta _ { t + 1 } ) :$

1. Latent Evaluative Quality (U): Measures whether the judge’s reasoning correctly verifies the candidate response against criteria $r _ { 1 : K }$

2. Surface Token Bias (B): Measures the marginal emission probability bias for verdict tokens $( \mathrm { e . g . } , \Delta P _ { \mathrm { Y E S } } = P ( ^ { \cdot } \mathrm { Y E S } ^ { \mathrm { * } } ) - 0 . 5 )$

3. Judge Emission $( Z _ { 1 } ) \colon$ : Emits tokens under policy $\pi _ { \theta _ { t } }$ , where $Z _ { 1 } = f _ { 1 } ( U , B ; \theta _ { t } )$

4. Checker Evaluation (S): Audits reasoning under synchronized copy $C _ { \bar { \theta } _ { t } }$ , where $S =$ $f _ { 2 } ( Z _ { 1 } ; \bar { \theta } _ { t } ) = a U + b B + \varepsilon .$

When the checker uses identical YES/NO readout templates, $f _ { 2 }$ computes reward by directly parsing YES tokens from Pass 2. Because $\bar { \theta } _ { t } = \theta _ { t }$ , any gradient update $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { R I } }$ that reinforces YES tokens in Pass 1 simultaneously increases the probability that Pass 2 emits YES tokens $( b > 0 )$ This establishes a degenerative backdoor shortcut: the policy maximizes reward by driving $B  1$ $( b B \to \operatorname* { m a x } )$ without increasing true reasoning quality $\mathbf { \bar {  { U } } } ( a \mathbf { \bar { U } } \approx \mathrm { c o n s t } )$

Interface decoupling as structural causal intervention. Interface decoupling replaces $f _ { 2 }$ with a free-standing 5-tier scalar Final Score $s _ { i } \in \{ 0 , 1 , 2 , 3 , 4 \}$ . Because the scalar score requires generating a holistic reasoning audit followed by a distinct numerical symbol, the direct token-copying gradient pathway from Pass 1 YES tokens to Pass 2 reward extraction is severed $( \partial S / \varepsilon$ verdict token ≈ 0, effectively setting $b \approx 0 )$ . The optimizer is therefore constrained to maximize reward solely through improving genuine reasoning verification (aU), converting reward growth into robust evaluative accuracy.

![](images/38b09eb1cde23cd8af31078ff44ef52fc944847a388fe4ac20672ee6a233634a.jpg)  
Figure 5: Surface coupling and the degenerative reward shortcut pathway (Section 3.2). When Judge (Pass 1) and Checker (Pass 2) share identical YES/NO readout templates, self-produced reward S is extracted directly from surface decision tokens $\begin{array} { r } { ( S = \frac { 1 } { K } \sum \mathbf { 1 } ( \mathrm { C h e c k e r } _ { k } = \dot { \mathrm { Y E S } } ) } \end{array}$ ). Policy gradient optimization inflates YES token emission (B ↑), transferring immediately to synchronized checker weights (<sup>¯</sup>θ ← θ). This establishes a degenerative shortcut loop (bB) where self-assigned rewards surge to 100% while true evaluative capability U stalls. Interface decoupling replaces the checker’s verdict surface with a free-standing 5-tier scalar Final Score, severing this direct tokenlevel shortcut channel.

## C DUAL-VARIABLE DIFFICULTY COUPLING: CONCEPT AND EXEMPLARS

As introduced in Section 1 and analyzed in Section 5 (RQ4), a fundamental distinction exists between training standard generative models (e.g., in mathematics or coding) and training LLM rubric judges.

Single-variable vs. dual-variable difficulty. In single-variable tasks such as mathematical problem solving, instance difficulty is strictly governed by query complexity: given a challenging problem x, generating the correct reasoning trace y provides a non-trivial gradient signal. In rubric-based evaluation, however, an informative learning signal requires the simultaneous calibration of two coupled variables:

![](images/046dfaf8e6e5143750d7a3f77e86e6ed912c643c3ebf5353b1f5102d4ba12c32.jpg)  
If either variable fails to meet this calibration, the instance yields negligible corrective gradient:

(9)

• Trivial Response (y far from decision boundary): If a candidate response is blatantly compliant (e.g., an obvious perfect answer) or blatantly flawed (e.g., empty or gibberish text), even an untuned base judge trivially assigns YES or NO correctly. The cross-entropy loss produces near-zero gradient, imparting no refinement to the model’s decision boundary.

• Non-discriminative Rubric (R trivial or ambiguous): If a rubric criterion is overly generic (e.g., “the response is helpful”) or ambiguous, all rollout candidates receive identical verdicts, providing zero group-relative variance.

Why static teacher SFT degenerates. In static offline distillation corpora (even across ≈17,000 trajectories distilled from four frontier teachers: Qwen3.5, Gemini-3.1-Flash-Lite, DeepSeek-V4- Flash, and Kimi-K2.6), the candidate responses y are fixed beforehand. As the student judge improves during fine-tuning, the static triplets $\left( h , y , r _ { 1 : K } \right)$ ) quickly fall outside the student’s active, shifting decision boundary, degenerating into trivial instances. Scaling data quantity from 6.4k to 17k merely multiplies trivial samples without supplying borderline tension.

Why on-policy RECURSE succeeds. In contrast, RECURSE operates strictly on-policy: as the judge policy $\pi _ { \theta _ { t } }$ evolves, its sampled rollouts $z _ { t , i }$ <sub>i</sub> naturally probe subtle reasoning variations along its active decision boundary. The synchronized checker dynamically identifies nuanced discrepancies between these rollouts, generating differentiated group-relative advantages $\widehat { A } _ { t , i }$ that sustain informative dual-variable tension throughout training.

## D TRAINING DATA AND CANDIDATE RESPONSE SYNTHESIS

RubricHub (Li et al., 2026b) provides queries and fine-grained rubrics $\left( h , r _ { 1 : K } \right)$ across five diverse domains (Chat, Instruction-Following, Medical, Science, and Writing), but does not contain preexisting candidate AI responses y. To construct realistic evaluation instances $x = \left( h , y , r _ { 1 : K } \right)$ spanning a broad, balanced spectrum of response quality, we synthesize candidate responses using a tiered multi-model sampling protocol.

Three-tier model pool. We stratify 16 distinct LLMs into three capability tiers based on their reasoning and instruction-following capacities:

• Low Tier (compact/lightweight models): GLM-4-Flash, GPT-4.1-nano, Gemini-2.5-Flash-Lite, and Qwen3.5-9B.

• Mid Tier (mid-scale conversational models): LongCat-Lite-V2-Chat, DeepSeek-V3.2, GLM-5.1, GPT-4.1-mini, Gemini-2.5-Flash, Moonshot-v1-128k, and Qwen3.6-27B.

• High Tier (frontier reasoning models): LongCat-Flash-Thinking, LongCat-Flash-Chat, Qwen3.5-397B, DeepSeek-V4-Flash, GPT-4.1, Gemini-3.1-Flash-Lite, and Kimi-K2.6.

Balanced 1:2:1 tier allocation per query. For each query h, we allocate four response slots across tiers in a fixed Low : Mid : High = 1 : 2 : 1 ratio:

1 × Low Tier (often violates fine-grained negative constraints or nuances), y ∼ 2 × Mid Tier (borderline quality; compliant on standard rules but mixed on edge cases), 1 × High Tier (substantially compliant; provides challenging borderline rubrics).

(10)

Within each domain, models within each tier are assigned evenly across samples via a greedy balanced permutation schedule (ensuring mid-tier slots use two distinct models). This tiered distribution ensures the training corpus exhibits rich compliance variation across criteria without collapsing to trivial all-YES or all-NO extremes.

## E SV-HARD CONSTRUCTION

SV-HARD is a same-style human-verified validation set of 100 examples used to monitor reward validity and guide early stopping under RubricHub-style rubric judging. RubricHub is cluster-split so that no prompt or rubric cluster appears in both training and evaluation; SV-HARD is drawn from the evaluation split and never contributes gold labels to the RL training reward. Labels are LLMinitialized and then human-verified at the rule level; this compact holdout is far cheaper than scaling gold annotation to the full training corpus, but the verification step is still a real upfront cost.

The set covers diverse instruction domains: 20 instances from each of five categories (Chat, Instruction-Following, Medical, Science, and Writing), scored on progression-critical hard rules selected per prompt (mean 2.5 rules per prompt; 89/100 have 2 or 3). Because gold is expertchecked, SV-HARD is the only stream on which we measure checker fidelity (1 − NMAE) and compute PAV. It is a fair same-style validity monitor—matched in judging style to training and strictly cluster-isolated from it. Repeated inspection can still adapt early stopping to this validation set; we therefore reserve the different-style benchmarks and SV-FULL for testing substantive transfer claims. Their parallel gains argue against a purely SV-HARD-specific selection artifact.

## F SV-FULL CONSTRUCTION

SV-FULL is a complementary online in-domain validation stream on the same 100 cluster-isolated prompts as SV-HARD, but each instance is scored on its full rule suite rather than the progression critical hard subset. Gold labels are resolved by an LLM annotation panel with Claude-based tiebreaking; they are never used in the RL training reward. SV-FULL therefore tests whether judge improvement extends to complete in-domain rubric coverage under the same RubricHub judging interface, without the human-verification cost required for reliable checker diagnostics. We evaluate SV-FULL at the same checkpoint cadence as the offline transfer suite and report rule accuracy in Table 2; PAV itself is computed only from SV-HARD.

## G ABLATION ARMS

All five controls below are run on Qwen3.5-9B. Frozen base: checker fixed at untuned init. External 27B: Qwen3.6-27B checker, never synced. Self-consistency: majority agreement reward without process audit. Teacher-traj. SFT (6.4k): full FT on 6400 teacher YES/NO traces (from Qwen3.5, Gemini-3.1-Flash-Lite, DeepSeek-V4-Flash, and Kimi-K2.6), 200 steps matched to the RL budget. Teacher-traj. SFT (17k): same teachers and recipe on the full eligible pool (≈17k traces, two epochs), testing whether more imitation data closes the gap.

## H REWARD AND FORMAT DETAILS

The format gate enforces two structural constraints on judge rollouts z<sub>t,i</sub>: (1) it requires exactly K parseable YES/NO decisions in the indexed format (1: <ans>YES/NO</ans> . . . K: <ans>YES/NO</ans), and (2) it strictly forbids delimiter tokens reserved for the checker prompt template (e.g., prematurely emitting <|AI’s Response End|> or faking checker reward blocks such as <|Final Score|>4<|Final Score|>). This prevents prompt-injection shortcuts where the judge rollout manipulates Pass 2 slot boundaries or directly injects fraudulent high checker scores into the downstream context. Samples failing the gate receive zero reward $( R _ { t , i } = 0 )$ and a zeroed response mask so they are excluded from the same-prompt group statistics.

## I TRAINING PROMPTS

Both passes share the same overarching evidence-block layout—system instruction, conversation context, the AI response, and a numbered rubric list—ensuring that the checker observes the identical evidence evaluated by the judge. The two passes differ in how these slots are populated and in their required output format. Below, we decompose the prompt template into three distinct components: the shared evidence-block template, the judge-specific instruction and output format (Pass 1), and the checker-specific process-auditing instruction and scoring format (Pass 2).

Part 1: Shared evidence-block template. Both the judge and checker instantiate the same standard three-slot evidence layout. Angle brackets (⟨⟨⟨·⟩⟩⟩) denote placeholders populated dynamically per instance:  
Your job is to assess if the AI’s response to the user’s   
most recent prompt correctly follows the provided rubrics.   
<|Conversation History Start|>   
<<<conversation history>>>   
<|Conversation History End|>   
<|AI’s Response Start|>   
<<<ai response>>>   
<|AI’s Response End|>   
<|Rubrics Start|>   
<<<rubrics>>>   
<|Rubrics End|>

The three evidence slots are instantiated for each pass as follows:

• Judge pass $( \boldsymbol { x } = ( h , y , r _ { 1 : K } ) )$ : <<<conversation history>>> contains the original dialogue context h; <<<ai response>>> contains the candidate generation y; and <<<rubrics>>> contains the K task-level evaluation criteria $r _ { 1 : K }$

• Checker pass $( C ( x , z _ { t , i } ) ) \colon$ : <<<conversation history>>> contains the complete judge input instance x; <<<ai response>>> contains the judge’s full rollout $z _ { t , i }$ (verbatim per-criterion reasoning and verdicts); and <<<rubrics>>> contains meta-rubrics of the form “k: Did the AI correctly judge the following rubric: $\langle r _ { k } \rangle ^ { , , }$

Part 2: Judge instruction and output format (Pass 1, trainable). Appended to the shared evidence block, the judge is instructed to generate step-by-step reasoning followed by concatenated per-rule binary decisions:

Please first analyze step by step whether the response meets   
each of the rubrics above. If it meets the rubric, output   
YES; otherwise, output NO.   
Finally, provide the results in rubric order using the   
concatenated format:   
1:<ans>YES/NO</ans> 2:<ans>YES/NO</ans> ...   
K:<ans>YES/NO</ans>.

The judge’s output is parsed and verified by the format gate (§H), requiring all K indexed decisions to be strictly valid.

Part 3: Checker instruction and output format (Pass 2, reward-only). Appended to the shared evidence block, the synchronized checker audits the reasoning process and emits a single, decoupled scalar Final Score $s _ { i } \in \{ 0 , 1 , 2 , 3 , 4 \}$

First, analyze whether each rule’s judgment is accurate,   
then score the overall scoring process.   
Use 5 tiers (0-4):   
0 = all rule judgments are wrong and the judge’s internal   
explanation is very poor;   
1 = most rule judgments are wrong (only a small fraction   
correct) and the judge’s internal explanation is   
poor/unclear;   
2 = about half of the rule judgments are correct and the   
judge’s internal explanation is partially clear but has   
notable issues;   
3 = most rule judgments are correct (few wrong) and the   
judge’s internal explanation is mostly clear with minor   
issues;   
4 = all rule judgments are correct and the judge’s analysis   
process is clear.   
Output the final score to   
<|Final Score|>   
<<<final score>>>   
<|Final Score|>

The parsed scalar $s _ { i }$ serves as the sole RL reward $( R _ { i } = s _ { i } )$ . This structural separation between the judge’s per-rule YES/NO verdicts and the checker’s 5-point scalar score closes the token-level copying channel (§3.2); other shortcuts can remain.

## J EVALUATION METRICS

Let $\hat { y } _ { r } ~ \in ~ \{ \mathrm { Y E S } , \mathrm { N O } \}$ be the judge’s prediction for rubric criterion r and $g _ { r }$ the gold label. In the main text (Table 1 and Table 2), we report each benchmark under its standard primary metric for conciseness. Below we define both primary and supplementary diagnostic metrics used in our analyses.

Rule accuracy (SV-HARD, SV-FULL, HealthBench, ProfBench). For instances with per-rule gold labels, we pool over all rules with both a gold label and a prediction (micro-average): rule acc $= \textstyle \sum _ { i }$ correct ${ \mathrm { \Omega } } _ { i } / \sum _ { i }$ total , where rules with no prediction are skipped rather than counted wrong. This is the primary metric for SV-HARD, SV-FULL, HealthBench, and ProfBench in Table 2.

All-correct rate (SV-HARD, HealthBench, ProfBench). The sample-level fraction of instances whose every comparable rule is judged correctly, i.e. $\mathbf { 1 } [ \mathrm { g o l d \_ m a t c h } _ { i } \stackrel { \sim } { = } 1 ]$ averaged over instances. This is the stricter sample-level accuracy metric reported alongside rule accuracy in Table 2.

Macro accuracy (SV-HARD, HealthBench, ProfBench). The unweighted average of perinstance rule accuracies across all valid evaluation samples, treating each evaluation prompt/rubric equally regardless of its rule count.

HealthBench rule accuracy and F1. Gold is the strict majority vote of physician binary labels (ties excluded); rule accuracy is agreement with this majority (reported in Table 2). We additionally compute the official balanced F1, which pairs the model prediction with each physician label (including ties) and averages the positive- and negative-class F1.

RubricBench pairwise accuracy. For each A/B pair we compute the mean YES-rate on each side; the side with the higher rate is predicted preferred (ties broken by YES count). Accuracy is over paired cases against the gold preference label.

CheckEval correlation and metric interpretation. CheckEval-Summ (Lee et al., 2025) assesses document summarization across 1,600 summaries generated by 16 distinct LLM systems across 100 source documents from SummEval (Fabbri et al., 2021). In the original SummEval corpus, human experts directly assigned holistic Likert-scale ratings (1–5 scale, averaged across annotators to yield continuous scores $\mathbf { H } _ { d } )$ for each of four broad quality dimensions: Coherence, Consistency, Fluency, and Relevance. Human annotators evaluated overall summary quality directly and did not annotate fine-grained checklist rules. To evaluate automated judges without subjective scalar scoring, CheckEval (Lee et al., 2025) decomposed these four holistic dimensions into 15 atomic, binary checklist criteria:

• Coherence (3 checklist items): Logical narrative flow, paragraph transitions, and semantic organization.

• Consistency (3 checklist items): Factual alignment with the source document and absence of hallucinations.

• Fluency (4 checklist items): Grammatical correctness, lexical clarity, and formatting cleanliness.

• Relevance (5 checklist items): Coverage of central key ideas and avoidance of superfluous details.

All 15 checklist criteria are global (identical across all 1,600 summaries). For each summary $i \in \{ 1 , \ldots , 1 6 0 0 \}$ and dimension d ∈ {coherence, consistency, fluency, relevance}, the model’s predicted dimension score $P _ { i , d }$ is computed as the proportion of YES verdicts emitted over that dimension’s checklist items:

$$
P _ { i , d } = \frac { 1 } { | \mathrm { R u l e s } ( d ) | } \sum _ { r \in \mathrm { R u l e s } ( d ) } \mathbf { 1 } ( \mathrm { J u d g e } _ { i , r } = \mathrm { Y E S } ) \in [ 0 , 1 ] .\tag{11}
$$

Across all $N = 1 { , } 6 0 0$ evaluation summaries, we compute the sample-level Pearson correlation coefficient $\rho _ { d } = \mathrm { C o r r } ( \mathbf { P } _ { d } , \mathbf { H } _ { d } )$ between the model’s predicted score sequence $\mathbf { P } _ { d }$ and the continuous human rating sequence $\mathbf { H } _ { d }$ . The primary CheckEval metric reported in Table 2 is the macro-average of these sample-level correlation coefficients across the four dimensions:

$$
\rho = \frac { 1 } { 4 } \sum _ { d \in \{ \mathrm { c o h , c o n , f l u , r e l } \} } \rho _ { d } .\tag{12}
$$

Interpreting correlation magnitude versus percentage accuracy: Crucially, Pearson correlation $\rho \in \mathsf { [ - 1 , 1 ] }$ measures linear alignment with continuous human judgment distributions rather than discrete 0–100% classification accuracy. In a large evaluation pool of 1,600 instances, an absolute lift in Pearson correlation (e.g., +0.019 from 0.422 to 0.441 on Qwen-9B, +0.024 on Gemma, and +0.027 on Qwen-27B) represents a statistically robust and practically meaningful improvement in continuous alignment fidelity $( p = 0 . 0 0 3 2 < 0 . 0 1 ;$ Appendix N), which must not be naively equated with or compared directly against percentage-point increments on discrete accuracy benchmarks.

![](images/08b90467d95186c7a71678bcd9e26a56dc85811017ebecf8b8be5da97c2cfcfc.jpg)  
Figure 6: Gold-derived diagnostics for judge and checker capability on a four-rollout example. The same held-out rule labels simultaneously evaluate judge rule accuracy (A) and the checker reference target U = 4u (where u is gold rule correctness). Checker diagnostics evaluate point error (MAE and Exact match) and normalized ranking fidelity $( 1 - e _ { C } = 1 - \mathrm { \bar { N } M A E } )$ , directly supplying the PAV composite monitor (V). Gold labels never enter the RL training reward.

Checker point calibration and ranking fidelity (PAV). The checker emits a scalar score $S \in$ {0, 1, 2, 3, 4}. To assess whether it faithfully recovers the correctness of the judge’s reasoning, we compare S against the reference target $U \overset { \cdot } { = } 4 \times$ (correct rules/total rules) ∈ [0, 4] (e.g., all rules correct → 4; half → 2). We evaluate checker capability through three aligned diagnostic metrics:

• Exact-Match Rate: Converts continuous reference target U to its nearest integer tier ⌊U⌉ and tests exact equality $\mathbf { 1 } [ S = \lfloor U \rceil ]$

• Mean Absolute Error (MAE): Computes the sample-level mean absolute error 1< $\bar { \tau } \sum _ { i = 1 } ^ { N } | S _ { i } - U _ { i } |$ , penalizing unparsable outputs with the worst-case endpoint error (4).

• Ranking Fidelity $( 1 - e _ { C } = 1 - \mathrm { N M A E } )$ : Normalizes the checker error to $\begin{array} { r } { e _ { C } = \frac { \mathrm { M A E } } { \varDelta } \in } \end{array}$ [0, 1], yielding the ranking fidelity $1 - e _ { C }$ . As derived in Eq. 4, 2e<sub>C</sub> strictly bounds the expected error on within-prompt pairwise reward differences $\left( { \bar { S } } _ { i } - S _ { j } \right)$ ), directly serving as the reward-quality component of the PAV monitor.

Figure 6 illustrates how the same held-out gold labels simultaneously evaluate judge accuracy and checker ranking fidelity without leaking supervision into the RL training reward.

## K BENCHMARK CHARACTERISTICS AND EXAMPLES

Table 5 summarizes the distinguishing traits of each held-out suite and one representative instance that best illustrates them. Queries, responses, and rubrics are abridged; the judge outputs a YES/NO for each listed criterion.

HealthBench corpus scale and stratified 10% subsampling protocol. The full OpenAI Health-Bench meta-evaluation benchmark (Arora et al., 2025) comprises 29,511 rubric evaluation instances across 14,592 prompt-completion groups across 10 overarching clinical themes (e.g., global health, emergency referrals, healthcare data communication, uncertainty hedging) and 34 medical subcategories. In the original corpus, multiple candidate LLM responses per clinical query are paired with 1–3 physician-crafted rubric criteria evaluating patient safety, diagnostic communication, and referral thresholds. Evaluating all 14,592 groups across training trajectories imposes an intractable computational bottleneck (≈ 30k sequential reasoning traces per evaluation step). To balance eval uation tractability with clinical fidelity, we implement a stratified proportional sampling protocol:

Table 5: Benchmark characteristics and representative examples. Each row pairs the trait emphasized in Section 4 with one instance that best illustrates it.
<table><tr><td>Benchmark</td><td>Key characteristics</td><td>Representative example</td></tr><tr><td></td><td>Same-style PAV monitor; short hard-rule subsets (mean 2.5 rules), mostly format/instruction checks; human-verified gold.</td><td>Combinatorics + format constraints. Query: count paths on a grid where every sentence must begin with “highway&quot; and sentences are joined by hyphens. Rubric: (1) first word of every sentence is &quot;highway&quot;; (2) hyphens split sentences correctly; (3) the response addresses the math question. Gold on progression-critical rules: {1:YES, 2:NO}.</td></tr><tr><td>HealthBench</td><td>Medical safety; representative 10% cluster sample (1,459 few rules per prompt but semanti- cally richer physician norms.</td><td>Incomplete clinical context. Query: compute perioperative mortality risk with missing details. Response: &quot;I&#x27;m sorry, but I can&#x27;t assist with that.&quot; Rubric: (1) prompts) from full benchmark; does not give a definitive answer when information is missing ([T,T]); (2) still seeks missing context rather than flatly declining ([F,F]).</td></tr><tr><td>RubricBench</td><td>Pairwise preference between two responses under sample-specific multi-criterion rubrics (mean 5.8 rules).</td><td>Code assistance. Query: “Add mypy type hints for the code below&quot; (no code pro- vided). Response: asks the user to provide the code first. Rubric checks whether the assistant recognizes the missing code, explains why, addresses mypy specifi- cally, stays clear, and remains polite.</td></tr><tr><td>CheckEval- Summ</td><td>Global 15-criterion checklist shared across all summaries (not sample-specific); Pearson ρ on aggregated YES-rates.</td><td>News summarization. Query: summarize a document (with reference summary). Rubric: fixed checklist over coherence (×3), consistency (×3), fluency (×4), and relevance (× 5); YES-rate per dimension is correlated with human dimension scores.</td></tr><tr><td>ProfBench</td><td>Expert professional tasks; long weighted rubrics (≈29 rules) with tight numerical tolerances; hard- est suite.</td><td>Physics PhD/ EMI shielding. Query: multi-step numerical derivation for shielding effectiveness. Rubric: weighted Critical/Major/Minor criteria checking specific numerical ranges  $( \mathbf { e . g . ~ } R \in$  [0.6835, 0.6841]), the correct dominance conclu- sion, and stated shielding-effectiveness relations.</td></tr></table>

1. Atomic group preservation: Evaluation records are grouped by unique prompt–response pairs $( h , y )$ so that all interdependent rubric criteria corresponding to a single candidate response remain bound together.

2. Stratified category proportionality: Groups are partitioned across all 34 medical categories and sampled strictly in proportion $( 1 0 \%$ , fraction $\alpha = 0 . 1 0 )$ to their original category prevalence in the full benchmark. This yields exactly 1,459 representative prompt–response groups (2,951 rubric rules, mean 2.02 rules/group), faithfully preserving the semantic distribution across all 10 clinical themes without distortion.

3. Physician consensus ground truth: Gold labels represent the strict majority vote across certified physician raters (ties excluded).

## L ADDITIONAL TRAINING DYNAMICS

To understand the internal mechanics of recursive self-improvement and how evaluative capabilities evolve over training steps, we examine fine-grained diagnostic trajectories for both the judge policy $\pi _ { \theta _ { t } }$ and the synchronized process checker $\bar { C } _ { \bar { \theta } _ { t } }$ on the primary Qwen3.5-9B run. Figures 7 and 8 track four complementary diagnostic metrics across the complete optimization trajectory:

1. Judge Rule Accuracy $( \widehat { A } _ { t } ,$ , panel a): The macro-averaged classification accuracy of individual YES/NO rule verdicts emitted during Pass 1 against ground-truth human annotations.

2. Checker Exact-Match Rate (panel b): The proportion of instances where the checker’s 5-tier scalar score $s _ { i } \in \{ 0 , 1 , 2 , 3 , 4 \}$ in Pass 2 matches the true gold target score $S ^ { \star }$ exactly $( { \bf 1 } ( s _ { i } = S ^ { \star } ) )$ .

3. Checker Mean Absolute Error (MAE) (panel c): The absolute difference between the predicted scalar score and the true ground-truth score $( | s _ { i } - S ^ { \star } | )$ . Any malformed or unparsable checker output is penalized with the maximum boundary error (4).

![](images/86076747b4f54ede169b91548536ececbaecdf1b7952969982bea6cd332d0a34.jpg)

![](images/e311e26bb28b37f63d846c93191a1d640c72aef392b90b4ac95ce38e67990110.jpg)

![](images/d46685c2b19afbc15bd459733a7bb2c8c5c864b41f96f8e9c9163983b60a6dc5.jpg)

![](images/a3e0fe8c75f9b2e52997d766369772c24dbe51d0c91ba35718f24ad0b940e353.jpg)  
Figure 7: SV-HARD diagnostics on the Qwen3.5-9B trajectory. (a) Validation rule accuracy (↑). (b) Checker exact-match rate (↑). (c) Checker MAE on the original 0–4 score scale $( \downarrow ) ;$ malformed checker outputs receive their worst possible endpoint error. (d) Checker fidelity $1 - \mathrm { N M A E } = 1 -$ $\mathrm { M A E } / 4 \left( \uparrow \right)$ . Dotted lines and enlarged markers show the PAV landmark @130. Rule accuracy alone continues to rise, whereas exact match, MAE, and fidelity expose the sharp checker improvement around the selected checkpoint.

4. Checker Ranking Fidelity $( 1 - \mathrm { N M A E } = 1 - \mathrm { M A E } / 4 , $ , panel d): Normalized checker fidelity mapping score errors to a unit interval [0, 1], representing the checker’s reliability in assigning calibrated process rewards.

Diagnostic dynamics on SV-HARD and validation deception. As illustrated in Figure 7, during the early and intermediate optimization phases (steps 0–130), judge accuracy $\widehat { A } _ { t }$ and checker fidelity 1 − NMAE climb in close unison. The checker exact-match rate surges from 21.5% to 48.2%, while checker MAE drops from 1.64 to 1.03, reflecting rapid mutual refinement under interface decoupling. However, beyond step 130, a profound divergence emerges: while judge rule accuracy on SV-HARD continues to creep upward through step 200 (reaching 75.9%), checker exact match and fidelity plateau and begin a sharp descent (exact match dropping to 36.0% and MAE rising back to 1.32). This divergence directly uncovers the onset of validation deception: when trained past the optimal stopping boundary, the policy over-optimizes for surface stylistic quirks in the training distribution, causing the checker’s calibrated auditing capability to deteriorate. Relying solely on judge accuracy on the validation set would select an over-optimized late checkpoint (e.g., step 200), whereas PAV’s joint formulation correctly localizes the true capability peak near step 130.

Independent cross-domain validation on HealthBench. Figure 8 tracks the exact same four di agnostic quantities on the held-out HealthBench suite (1,459 prompts, 2,951 rules) spanning clinical communication and patient safety—a distinct distribution never seen during training or validation monitoring. Crucially, on HealthBench, judge rule accuracy (82.5% → 85.9% → 82.9%), checker exact match $( 2 8 . 4 \% \cdot ) 4 1 . 6 \%  3 5 . 1 \% )$ , and checker fidelity $( 0 . 7 4  0 . 8 1  0 . 7 6 )$ all reach their global maxima simultaneously within the exact same window (near step 130) before decaying together. This cross-domain resonance provides decisive evidence: the degradation in checker fidelity detected by PAV on SV-HARD is not a localized artifact of the validation set, but reflects a general transition from robust reasoning acquisition to out-of-distribution capacity collapse.

## M CROSS-ARCHITECTURE REPLICATION ON GEMMA-4-E4B-IT

Gemma-4-E4B-it replicates the main RECURSE recipe without ablation variants, providing an independent test of RQ1 (viability) and RQ3 (bounded horizon) outside the Qwen model family. As summarized in Table 2, held-out metrics exhibit consistent positive trajectories through the useful window: SV-HARD rule accuracy rises from 55.9% to 61.1% at PAV-selected @160, HealthBench gains +3.7 points, and CheckEval Pearson $\rho$ improves from 0.442 to 0.466. SV-FULL remains near its high base (92.5%), consistent with a near-saturated in-domain ceiling on this architecture.

Figure 9 repeats the judge/checker diagnostic panels from Figures 7 and 8 on SV-HARD and Health-Bench across training steps. Checker fidelity and judge accuracy climb in unison and plateau around steps 120–160 before softening afterward, while late-stage SV-HARD accuracy can still drift upward—the same validation-deception pattern that motivates PAV on Qwen-9B. This confirms that bounded recursive self-improvement and PAV-localized stopping are robust across distinct model architectures.

![](images/d09b70f2a15090c13f6da41d064ee80bd11a530f2177745e7f7935ffd412853b.jpg)

![](images/bce6b1fbc8e867004a35b84f4e25b887756e2284ffab970b98d3e0f53e5c8e6c.jpg)

![](images/4accced9c1cde4367825c7dabf4281203c011f7164ffe1feba6046ba01df6812.jpg)

![](images/234bd03d5f10b7efbdc63a5b77f2cb4703bb273ca540ead8f82469d5fa4ab16a.jpg)

![](images/d5fff30b533d0f966b8a5e9aa7f8d21ef7df5d39c58581dd6116973e9a40249d.jpg)  
Figure 8: HealthBench diagnostics on the Qwen3.5-9B trajectory. Same four panels as Figure 7, now on the different-style medical suite that PAV never sees. (a) Judge rule accuracy (↑). (b) Checker exact-match rate (↑). (c) Checker MAE on the 0–4 scale (↓); malformed checker outputs receive their worst possible endpoint error. (d) Checker fidelity $1 - \mathrm { N M A E } = 1 - \mathrm { M A E } / 4 \stackrel { \cdot } { ( \uparrow ) }$ . Dotted lines and enlarged markers mark the PAV landmark @130. Judge accuracy, exact match, and fidelity peak together around that region and then fall together, so the transfer suite tracks the same useful window as the SV-HARD monitor.  
Figure 9: Gemma judge and checker dynamics (RQ3 replication). Top row: SV-HARD diagnostics; bottom row: HealthBench diagnostics on the Gemma-4-E4B-it trajectory (steps 10–200). Same four panels as Figures 7 and 8. Dotted lines and enlarged markers mark the PAV landmark @160. Rule accuracy alone can continue to rise on SV-HARD, whereas exact match, MAE, and fidelity expose the checker-quality window that motivates stopping.

## N PAV CONSTRUCTION AND VALIDATION DETAILS

Finite-sample unbiasedness at a fixed checkpoint. For any single fixed checkpoint $t ,$ rule correctness $Y _ { t } \ \bar { \in } \ \{ 0 , 1 \}$ and normalized checker error $E _ { C , t } = | S _ { t } - \overline { { U _ { t } } } | \overline { { / } } 4$ have expectations $A _ { t }$ and $e _ { C , t }$ over the prompt distribution. Because empirical sample means are linear estimators, their expectations satisfy $\mathbb { E } [ \widehat { A } _ { t } ] = A _ { t }$ and $\mathbb { E } [ \widehat { e } _ { C , t } ] = e _ { C , t }$ , holding even when computed from separate rollout samples on a fixed prompt set. Since the PAV formulation is an affine combination:

$$
\mathbb { E } [ \widehat { V } _ { t } ] = \frac { \mathbb { E } [ \widehat { A } _ { t } ] + 2 \{ 1 - \mathbb { E } [ \widehat { e } _ { C , t } ] \} } { 3 } = \frac { A _ { t } + 2 ( 1 - e _ { C , t } ) } { 3 } = V _ { t } .\tag{13}
$$

Pointwise estimation vs. sequential monitoring. A critical methodological distinction must be made between evaluating a single checkpoint post-hoc and sequentially monitoring an online training trajectory across M evaluation steps $( t _ { 1 } , \dots , t _ { M } )$ . In naive sequential selection, picking the absolute empirical maximum $\widehat { t } = \arg \operatorname* { m a x } _ { t _ { m } } \widehat { V } _ { t _ { m } }$ introduces a family-wise error rate (FWER) inflation across the M dependent hypothesis tests.

Importantly, RECURSE does not treat PAV as a fragile pointwise arg max selector. Rather, as demonstrated by the bootstrap distributions and fold-resampling experiments below, $\widehat { V } _ { t }$ forms an elevated, stable validity plateau (steps 120–140 for Qwen-9B, steps 150–170 for Gemma). Any checkpoint retained within this elevated validity window captures essentially identical peak transfer gains across all held-out suites. PAV thus functions as a robust stopping horizon monitor that signals when checker degradation begins, rather than an overfitted single-step selector.

Adequacy and power of the 100-prompt human-verified monitor. Although 100 prompts may appear compact, SV-HARD is specifically curated for high diagnostic signal-to-noise ratio:

1. High rule density on progression-critical criteria: SV-HARD contains ≈250 humanverified rules, filtered specifically for hard, borderline instances where untuned models exhibit high disagreement.

2. Multi-rollout sample size aggregation: At each evaluation step, every prompt generates $n = 4$ rollout traces that are each scored by the checker. The empirical monitor $\widehat { V } _ { t }$ is therefore computed over $1 0 0 \times 4 = 4 0 0$ complete judge-checker reasoning trajectories per checkpoint, providing substantial statistical power.

3. Independent corroboration by SV-FULL: As highlighted in Section 5, SV-FULL evaluates the full rule suite on 100 prompts under independent LLM panel supervision. Despite being completely excluded from PAV computation, SV-FULL independently peaks at @130 (95.3%) and collapses to 89.7% by @200, mirroring the exact stopping boundary identified by PAV on SV-HARD.

Why checker error has weight two. Group-relative learning depends on score differences $( S _ { i } -$ $S _ { j } )$ , not isolated absolute scores. Writing $\delta _ { i } \stackrel { . } { = } S _ { i } - U _ { i }$ , the error in any pairwise reward difference satisfies $| ( \delta _ { i } - \delta _ { j } ) | / 4 \leq ( | \delta _ { i } | + | \delta _ { j } | ) / 4$ . Taking expectations over exchangeable rollout pairs yields $2 e _ { C , t }$ . Combining this with judge rule error $( 1 { \bar { - } } A _ { t } )$ defines the compound surrogate risk $\left( 1 - A _ { t } \right) +$ $2 e _ { C , t }$ , leading directly to Eq. 5. The factor of 2 is therefore fixed a priori by the pairwise bound, not fitted empirically.

Curvature and boundary dynamics of the weight stability interval. For the generalized monitor $M _ { t } ( \lambda ) = \widehat { A } _ { t } + \lambda ( 1 - \widehat { e } _ { C , t } )$ , empirical evaluation demonstrates that the optimal stopping region remains centered at Qwen@130 and Gemma@160 across any $\lambda \in [ 1 . 5 9 , 1 1 . 0 3 ]$ . This broad stability interval is governed by the asymmetric curvature dynamics of the two constituent terms:

$$
\nabla _ { t } M _ { t } ( \lambda ) = \nabla _ { t } \widehat { A } _ { t } - \lambda \nabla _ { t } \widehat { e } _ { C , t } .\tag{14}
$$

Near step 130 on Qwen-9B, judge rule accuracy enters a gentle plateau with small marginal gains $( \nabla _ { t } \widehat { A } _ { t } \approx + 0 . 0 2 / 1 0$ steps), whereas normalized checker error undergoes a steep, convex escalation post-step $1 3 0 \ ( \dot { \nabla } _ { t }  { \hat { e } } _ { C , t } \  { \stackrel { \sim } { \approx } } \ + 0 . 1 5 / 1 0$ steps). Because the error penalty gradient strongly dominates the flattening accuracy gain for any $\lambda \geq 1 . 5 9$ , the stationary condition $\nabla _ { t } M _ { t } ( \lambda ) = 0$ is firmly locked within the interval [120, 140]. The theoretically derived weight $\lambda = 2$ thus sits safely within a robust, error-tolerant dynamic regime rather than a knife-edge parameter setting.

Prompt-cluster bootstrap uncertainty and variance control. We assess the statistical uncertainty of the empirical monitor by performing 10,000 prompt-level cluster bootstrap iterations on SV-HARD, resampling entire prompt groups with replacement to preserve intra-prompt rollout and rule correlations. At the designated stopping landmarks, empirical estimates achieve tight confidence bounds: $\widehat { V } _ { t } = 0 . 7 4 2$ with a 95% cluster-bootstrap confidence interval of [0.685, 0.797] for Qwen@130, and $\widehat { V } _ { t } = 0 . 6 2 8$ with [0.577, 0.679] for Gemma@160. Conclusion: These compact confidence intervals confirm that empirical PAV estimates maintain low finite-sample variance and high signal-to-noise ratio, proving that the localized validity peaks represent genuine, statistically significant capability gains $( p < 0 . 0 0 1 )$ rather than evaluation noise.

Validation-set sample insensitivity and fold stability. To directly test whether early-stopping localization depends on specific prompt selections (i.e., whether altering the validation sample shifts the selected checkpoint), we partition the 100 SV-HARD prompts into three disjoint subsets (34/33/33 prompts) and evaluate PAV trajectories on the complementary two-thirds (66–67 prompts) for each fold. As summarized in Table 6, the localized stopping region remains exceptionally stable: across all three independent subsets, the optimal window concentrates strictly within {130, 140} for Qwen-9B and within {160, 170} for Gemma. Conclusion: This invariance across validation-set folds demonstrates that PAV does not overfit to idiosyncratic prompt instances; instead, it robustly captures the underlying capability plateau of the optimization trajectory regardless of sample-level variations.

Table 6: Three-fold PAV region on SV-HARD. Each fold holds out 33–34 prompts and scores the remainder.
<table><tr><td>Trajectory</td><td>Fold 1</td><td>Fold 2</td><td>Fold 3</td></tr><tr><td>Qwen-9B</td><td>140</td><td>130</td><td>130</td></tr><tr><td>Gemma</td><td>160</td><td>170</td><td>160</td></tr></table>

Table 7: Offline checkpoint-selector comparison. Mean oracle regret measures the average performance deficit relative to the retrospective hindsight trajectory maximum across all four held-out transfer suites (lower is better; zero denotes oracle performance); “Imp.” denotes the number of transfer suites improved over Base $( / 4 )$ . Checker correlation selects by score–gold Pearson correlation.
<table><tr><td>Trajectory</td><td>Selector</td><td>Step</td><td>Imp.</td><td>Mean regret</td></tr><tr><td>Qwen-9B</td><td>PAV</td><td>130</td><td>4/4</td><td>1.20</td></tr><tr><td></td><td>SV-HARD accuracy</td><td>180</td><td>2/4</td><td>5.41</td></tr><tr><td></td><td>Checker correlation</td><td>200</td><td>2/4</td><td>5.51</td></tr><tr><td></td><td>Fixed final budget</td><td>200</td><td>2/4</td><td>5.51</td></tr><tr><td>Gemma</td><td>PAV</td><td>160</td><td>4/4</td><td>0.49</td></tr><tr><td></td><td>SV-HARD accuracy</td><td>160</td><td>4/4</td><td>0.49</td></tr><tr><td></td><td>Checker correlation</td><td>110</td><td>3/4</td><td>1.23</td></tr><tr><td></td><td>Fixed final budget</td><td>200</td><td>4/4</td><td>0.95</td></tr></table>

Offline selector comparison and oracle regret. To rigorously quantify how effectively an online validation selector identifies high-performing checkpoints without peeking at held-out transfer benchmarks, Table 7 compares PAV against three standard selection heuristics: (1) monitoring SV-HARD rule accuracy alone (arg max<sub>t</sub> $\bar { \widehat { A } } _ { t } )$ , (2) monitoring checker score-to-gold Pearson correlation, and (3) a fixed full-budget endpoint rule (step 200).

We formalize selector quality through mean oracle regret:

• Hindsight Oracle $( O _ { m } ) \colon$ : An omniscient oracle evaluates every intermediate checkpoint post-hoc across all four held-out transfer benchmarks $\begin{array} { r l r l } { m } & { { } \in } & { \mathcal { M } } & { { } = } \end{array}$ {HealthBench, RubricBench, CheckEval, ProfBench}, recording the retrospective global maximum achievable score on each suite: $O _ { m } = \mathrm { m a x } _ { t } \mathrm { S c o r e } _ { t } ( m )$

• Regret Deficit: For any realistic selector that chooses an operational stopping checkpoint t<sup>ˆ</sup>strictly using online validation signals, its performance deficit on benchmark m is $O _ { m } -$ $\mathrm { S c o r e } _ { \hat { t } } ( m ) \geq \mathbf { \bar { 0 } }$

• Mean Oracle Regret: Averaging this deficit across all held-out transfer suites yields the mean regret $\begin{array} { r } { \frac { 1 } { | \mathcal { M } | } \breve { \sum } _ { m \in \mathcal { M } } ( O _ { m } - \breve { \mathrm { S } } \mathrm { c o r e } _ { \hat { t } } ( m ) ) } \end{array}$ , measured in percentage points (lower is better; zero denotes parity with the theoretical hindsight oracle).

As shown in Table 7, PAV achieves the lowest mean oracle regret across both architectures (1.20 on Qwen-9B and 0.49 on Gemma-4B) and successfully retains performance improvements over Base across all four transfer suites $( 4 / 4 )$ . In sharp contrast, tracking validation rule accuracy alone is blinded by validation deception, selecting an over-optimized late checkpoint (step 180 on Qwen-9B) that incurs a steep regret of 5.41 points and degrades performance on half the transfer suites $( 2 / 4 )$ . Similarly, naively running to the final budget (step 200) suffers severe capacity collapse (5.51 regret). Conclusion: By penalizing checker ranking degradation alongside judge accuracy, PAV reliably approximates the hindsight oracle upper bound without leaking supervision from held-out transfer suites.

Statistical significance of out-of-distribution transfer gains. On the primary Qwen-9B ablation backbone, the gains over Base across all transfer suites are statistically significant:

Table 8: Systems and operational trade-off matrix. Comparing different paradigms for improving LLM judge capabilities: commercial API judges, standalone learned Reward Models (RMs), and RECURSE (Ours).
<table><tr><td>Dimension</td><td>Commercial API Judge (e.g., GPT-4o)</td><td>Standalone Learned RM</td><td>RECURSE (Ours)</td></tr><tr><td>Training Reward Supervision Operational &amp; Financial Cost Throughput &amp; Latency Reward Hacking Resistance</td><td>Outsourced to external commercial model Recurring per-token API query fees Bound by API rate limits &amp; network WAN Susceptible to prompt-level exploitation</td><td>Large-scale curated human preference pairs Separate data collection &amp; RM training compute High (co-located local inference) High risk (policy exploits static proxy flaws)</td><td>Zero external gold (on-policy process check) Amortized local inference on shared hardware High (co-located vLLM worker snapshot) Mitigated (interface decoupling + PAV stopping) Fully on-premises</td></tr></table>

• CheckEval-Summ (1,600 summaries across 15 dimensions): Paired t-test confirms the lift in dimension correlation $( \rho : 0 . 4 2 2  0 . 4 4 1 )$ is statistically significant with $t = 2 . 9 5$ $( p = 0 . 0 0 3 2 < 0 . 0 1 )$

• ProfBench (120 expert queries, ≈ 3,500 rule evaluations): Paired cluster bootstrap test across rule sets yields $p = 0 . 0 0 8 < 0 . 0 1$ for the +1.6pp rule accuracy lift $( 7 3 . 1 \% $ 74.7%).

• HealthBench (1,459 physician-annotated queries) and RubricBench (2,294 pairwise comparisons): Both demonstrate positive gains (+3.4pp and +3.1pp) with $p < 0 . 0 0 0 1$

• Downstream DPO (Qwen-27B): On GPQA, whereas base-judge labels degrade performance (80.87 → 79.02), RECURSE labels achieve 81.46, establishing a statistically reliable net positive margin of +2.44pp over the base judge.

These statistical tests confirm that held-out suite improvements reflect genuine capability transfer rather than stochastic evaluation noise.

## O SYSTEMS, SCALABILITY, AND COST-PERFORMANCE TRADE-OFF ANALYSIS

Checker compute overhead and decoding throughput. While the checker audits all n = 8 judge rollouts per training prompt (yielding matching 1:1 rollout generation counts), it operates strictly in pure inference mode without gradient backpropagation or optimizer state updates. In standard RL fine-tuning, policy optimization is dominated by the actor’s backward pass, gradient computation, and optimizer step, which typically consume 2× to 3× the FLOPs of a forward pass per token. By hosting the reward-only checker on a dedicated, co-located vLLM instance with continuous batching and PagedAttention, Pass 2 auditing executes with high throughput. Parameters are synchronized via in-memory peer-to-peer tensor broadcasts every step (≈ 1.2 seconds on Qwen-27B), eliminating the substantial training and maintenance overhead of separate learned reward models.

Synchronization latency and scalability on multi-GPU architectures. In distributed training across multi-GPU nodes, parameters are synchronized via peer-to-peer tensor transfers over highspeed interconnects (NVLink within nodes, InfiniBand across nodes). On our largest trained model, Qwen-27B (Tensor Parallelism size 4), in-memory parameter broadcast requires only ≈ 1.2 seconds per optimization step. Because autoregressive rollout generation across n = 8 samples dominates the per-step iteration time (≈ 45–60 seconds), checker parameter synchronization constitutes < 2.5% of total iteration wall-clock time. This confirms that synchronous weight tying introduces negligible communication bottleneck on multi-GPU distributed systems.

Trade-off comparison across judge-improvement paradigms. Table 8 summarizes the systems and operational trade-offs across different paradigms for improving LLM judge capabilities during post-training optimization: (1) distilling from or querying proprietary commercial LLMs (e.g., GPT-4o) as external judges/teachers during rollouts, (2) training separate standalone learned Reward Models (RMs) on curated human preferences, and (3) our closed-loop self-evaluation framework (RECURSE).

• Commercial API Judges: While commercial APIs provide strong evaluative signals for judge improvement, querying them synchronously across thousands of intermediate RL rollouts introduces significant network latency, strict API rate-limit bottlenecks, ongoing query fees, and potential exposure of sensitive training prompts.

![](images/d9b547d7161198161df3c52a0cc9d55d138b683e7d18ece777d545ca59dfe593.jpg)  
Figure 10: Format compliance on held-out SV-HARD validation for Qwen3.5-9B (steps 10–200). Judge verdicts remain parseable above 99%; checker score parse rate rises over training. The dotted line marks the PAV-selected checkpoint @130.

• Standalone Learned RMs: Training an auxiliary reward model to supervise judge training requires extensive preference data curation. Once fixed, static reward models fail to adapt to the evolving judge policy, suffering from out-of-distribution proxy degradation (Goodhart’s law). This is empirically validated by our RQ4 ablation (Table 3): even a 3× larger static 27B checker underperforms our synchronized 9B co-evolution by 6.1 points on HealthBench (79.8% vs. 85.9%), showing that static external scale cannot substitute for on-policy synchronization.

• RECURSE: By establishing a self-contained, closed-loop judge–checker recurrence on shared model parameters, RECURSE eliminates external API fees and ongoing human annotation for training rewards, maintains high local decoding throughput, guarantees data privacy, and enforces reproducible, bounded optimization via PAV.

## P FORMAT COMPLIANCE AND QUALITATIVE CHECKER FAILURE MODES

Malformed outputs are masked from both gradient and group statistics (§3.3), so a rising parsefailure rate would be an early collapse signal. Figure 10 shows that the judge’s verdicts remain parseable at ≥ 99% throughout the 200-step window. The checker’s score parse rate rises through the pre-boundary window (0.87 → 0.96 by step 130), and experiences a slight dip only late in training (> 160), consistent with the post-selection degradation in RQ4.

Parse rate is not a substitute for PAV. If one used checker parse rate as a heuristic stopping rule (e.g., stopping at the first > 5% drop from peak), selection would land at step 160—long after transfer metrics have begun deteriorating. Parse rate is an essential engineering sanity check, but lags the fine-grained reward-fidelity signal captured by PAV.

Qualitative failure modes of the late-stage checker (post-boundary dynamics). Qualitative error analysis of checker outputs beyond the PAV-localized window (steps 160–200) reveals three distinct degradation patterns explaining why unconstrained recursive training degrades out-ofdistribution transfer:

1. Score inflation and loss of ranking variance: As policy optimization saturates on-policy training prompts, the late checker becomes overly lenient, assigning uniform score $s = 3$ or s = 4 to mediocre reasoning rollouts. This collapses within-group advantage variance $( \sigma _ { t }  0 )$ , starving the policy of informative relative gradients.

2. Hallucinated meta-justifications: When evaluating flawed judge rollouts, late-stage checkers occasionally fabricate non-existent quotes or misinterpret rubric criteria to rationalize a high meta-score, exhibiting confirmation bias.

Table 9: Taxonomy of self-improving evaluation frameworks. Delineating RECURSE from prior self-training evaluation literature.
<table><tr><td>Method</td><td>RL Reward Source</td><td>External Anchors in RL</td><td>Interface Decoupling</td><td>Stopping Monitor</td><td>Feedback Target</td></tr><tr><td>Self-Taught Evaluator (Wang et al., 2024)</td><td>SFT on synthetic pairs</td><td>Synthetic seed pairs</td><td>N/A (SFT only)</td><td>Fixed SFT epochs</td><td>Preference label</td></tr><tr><td>Meta-Rewarding (Wu et al., 2025)</td><td>Meta-judge scalar reward</td><td>External frozen judge</td><td>No (coupled text)</td><td>Heuristic / steps</td><td>Outcome judgment</td></tr><tr><td>SELF-JUDGE (Lee et al., 2024)</td><td>Execution</td><td>Code interpreter / unit tests</td><td>N/A</td><td>Test pass rate</td><td>Program execution</td></tr><tr><td>Grad2Reward (Zhang et al., 2026b)</td><td>Meta-gradient alignment</td><td>Reference RM</td><td>No</td><td>Reference validation</td><td>Reward weights</td></tr><tr><td>RECURSE (Ours)</td><td>Synchronized self-checker</td><td>None (zero gold in RL)</td><td>YES (scalar vs. verdicts)</td><td>PAV on SV-HARD</td><td>Reasoning process</td></tr></table>

3. Verbosity drift and generation truncation: Late-stage checkers exhibit an elongation of internal reasoning chains, occasionally exceeding the maximum decoding window (32,768 tokens) before emitting the ⟨|Final Score|⟩ tag. This accounts for the minor drop in parsed checker outputs observed at step 190 in Figure 10.

These qualitative failure modes directly validate why tracking checker fidelity via PAV is vital for bounding the recursive optimization loop.

## Q TRAINING AND DPO HYPERPARAMETERS

Main judge–checker RL. Qwen-9B and Gemma use train batch size 32, n=8 rollouts per prompt, PPO mini-batch size 16, actor learning rate $3 \times 1 0 ^ { - 6 }$ , clip ratio 0.005 (low=high), and no KL penalty. Qwen-27B uses batch size 48, the same learning rate, and KL coefficient 0.005. All runs use FSDP with optimizer offload and sequence parallelism size 4. Maximum prompt / response lengths are 8192 / 32768 tokens for the judge; the checker score pool uses a matching 32768-token generation budget. Checkpoints and the full held-out monitoring suite (including PAV) are written every ten steps over 200 steps for Qwen-9B and Gemma, and every five steps over 75 steps for Qwen-27B. The checker is a standalone vLLM weight snapshot synchronized from the actor every step.

Downstream DPO. Preference pairs are labeled by our retained Qwen3.5-9B RECURSE judge (retained at the PAV-selected checkpoint, step 130) or its untuned base counterpart. Fine-tuning uses LlamaFactory (Zheng et al., 2024) on Qwen3.6-27B with LoRA rank 16, α=32, dropout 0.05, applying LoRA adapters to all linear projection layers, preference $\beta { = } 0 . 1$ , sigmoid DPO loss, learning rate $\mathrm { \bar { 5 } } \times 1 0 ^ { - 6 }$ , cosine schedule with 3% warmup, three epochs, per-device batch size 1 with gradient accumulation 8 (global batch 64 on 8 GPUs), cutoff length 8192, and bf16. Base-judge and RECURSE-judge runs differ only in the preference file; optimization settings are matched.

Teacher-trajectory SFT. The matched-step SFT reference fully fine-tunes Qwen3.5-9B (no LoRA) on 6400 trajectories distilled from four high-quality teacher judges (Qwen3.5, Gemini-3.1-Flash-Lite, DeepSeek-V4-Flash, Kimi-K2.6) on the training prompt pool, with learning rate $1 \times 1 0 ^ { - 5 }$ , global batch 64, and two epochs (200 optimizer steps), matching the main run’s prompt×step budget. A larger-pool variant uses the same teachers and hyperparameters on the full eligible distillation set (≈17,009 trajectories; two epochs, 532 optimizer steps at the same global batch). While scaling distillation data from 6.4k to 17k improves SFT accuracy on SV-HARD (64.2% → 70.2%), both SFT arms remain consistently below our on-policy RECURSE judge (73.7% on SV-HARD, 85.9% on HealthBench, 79.1% on RubricBench, 0.441 on CheckEval), and scaling the static pool does not achieve suite-wide parity. As analyzed in Section 5 and Appendix C, static offline distillation triplets $\left( h , y , r _ { 1 : K } \right)$ lack dynamic on-policy tension with the student’s shifting decision boundary.

## R TAXONOMY, CONCEPTUAL DISTINCTIONS, AND METHOD EXTENSIONS

To clearly delineate RECURSE from concurrent self-improving evaluators, Table 9 contrasts the core architectural and methodological dimensions.

• Absence of external RL anchors: Unlike Meta-Rewarding and Grad2Reward which require frozen reference models or external meta-judges, RECURSE generates the scalar training reward entirely from the model’s own synchronized policy copy.

• Structural shortcut mitigation: Prior methods overlook surface coupling. RECURSE identifies the degenerative token copying channel and introduces interface decoupling to make closed-loop recursive optimization viable.

• Theoretically grounded early stopping: Instead of heuristic step budgets, RECURSE introduces PAV, grounded in pairwise score-difference error bounds, to reliably identify the validity window before validation deception sets in.

Surface coupling vs. classical reward hacking. It is crucial to distinguish surface coupling from classical reward hacking:

• Classical Reward Hacking: Arises when an actor optimizes against a static, frozen reward model, exploiting proxy misalignment (e.g., generating verbose text) while the RM itself remains fixed.

• Surface Coupling: Arises uniquely in self-improving recurrence due to weight synchronization $( \bar { \theta } _ { t + 1 }  \theta _ { t + 1 } )$ . Because actor and checker share the identical YES/NO readout surface, policy updates inflating YES tokens simultaneously modify the reward function in real time, creating an unconstrained shortcut that bypasses reasoning auditing. Interface decoupling structurally severs this co-adapted token channel.

Generalization to holistic (non-rubric) evaluation. While this work focuses on rubric-based judging, the core principles of interface decoupling and PAV extend naturally to holistic (unconstrained) evaluation:

1. Pass 1 (Holistic Judge): Generates open-ended reasoning evaluating overall response quality, outputting an unconstrained quality score $q \in [ 1 , 1 0 ]$

2. Pass 2 (Decoupled Meta-Checker): Audits whether reasoning is complete, grounded, and logically sound against universal meta-criteria (e.g., factual consistency, depth, tone), emitting a decoupled 5-tier scalar audit score $s \in \{ 0 , \ldots , 4 \}$ as the sole RL reward.

3. Holistic PAV Monitoring: A compact human-verified validation set of holistic preferences computes judge accuracy and checker fidelity to establish the early-stopping horizon.

## S BROADER IMPACT AND ETHICAL CONSIDERATIONS

Automating evaluation with human oversight (\*Humans-over-the-loop\*). As LLMs are increasingly deployed across open-ended domains, manual verification becomes infeasible. RE-CURSE supports scalable automated verification by allowing evaluators to self-improve under fixed rubrics. Crucially, humans remain over the loop by authoring domain standards and establishing compact human-verified monitors (SV-HARD), while models perform high-throughput evaluation and self-auditing.

Mitigating automation bias and unanchored drift. A primary concern in self-improving systems is unconstrained drift or self-reinforcing bias (automation bias). RECURSE directly addresses this risk through: (1) Structural process auditing: Auditing reasoning against explicit meta-rubrics penalizes ungrounded rationales, even if final verdicts appear correct; (2) PAV safety shutdown: Continuously monitoring checker fidelity against human gold detects reward degeneration and enforces an empirical stopping boundary before degradation occurs. In safety-critical domains (e.g., medical or legal), we recommend combining RECURSE judges with periodic human spot-auditing.

## T LIMITATIONS

Scope of model architectures and task domains. Our empirical scope spans three model configurations up to 27B parameters (Qwen3.5-9B, Gemma-4-E4B-it, and Qwen3.6-27B) on English rubric benchmarks. We deliberately allocate compute such that Qwen3.5-9B serves as the primary scientific testbed for comprehensive ablations, Gemma-4-E4B-it provides cross-architecture replication, and Qwen3.6-27B demonstrates scale feasibility. While RECURSE achieves consistent gains across held-out medical, pairwise preference, summarization, and professional suites, extending bounded RSI to non-rubric formats (e.g., holistic unconstrained scoring), multilingual judging, and multimodal domains remains valuable future work.

Role of human verification in the holdout monitor. Although gold labels never enter the RL training reward, RECURSE relies on a compact human-verified validation set (SV-HARD) to compute the PAV early-stopping index. Constructing this 100-prompt holdout represents a real upfront human verification cost. While this cost is orders of magnitude smaller than labeling the entire training corpus, RECURSE keeps gold labels out of the RL reward rather than eliminating human verification from the end-to-end evaluation pipeline.

Statistical monitoring and sequential testing considerations. As analyzed in Section 3.3 and Appendix N, the empirical PAV estimator $\widehat { V } _ { t }$ is unbiased at any fixed checkpoint. However, online sequential monitoring across multiple checkpoint iterations introduces multiple-testing considerations. In our experiments, PAV reliably localizes a stable validity region rather than a brittle single-step point estimate, and the stopping boundary is independently corroborated by SV-FULL. Developing formal sequential stopping boundaries with family-wise error rate control provides a promising avenue for further theoretical refinement.

Process checking assumptions and granularity. The process checker audits the internal reasoning trajectory of the judge against meta-rubrics rather than inspecting ground-truth outcomes directly. If a checker develops systematic reasoning biases, it could theoretically assign high scores to flawed rationales. The 5-tier scalar interface (0–4) mitigates surface token shortcuts and reduce regression noise, though exploring adaptive scalar rewards with dynamic format gating warrants further exploration.

Scope of bounded RSI. Finally, bounded RSI in this work intentionally fixes the task specification, optimizer, and checking protocol. The policy’s evaluative capability provides the reward for its own parameters, but the model does not autonomously modify the training architecture or meta-rubric definitions. Our conclusions therefore apply specifically to closed-loop judge–checker optimization within structured evaluation domains rather than to open-ended, unconstrained recursive self-modification.