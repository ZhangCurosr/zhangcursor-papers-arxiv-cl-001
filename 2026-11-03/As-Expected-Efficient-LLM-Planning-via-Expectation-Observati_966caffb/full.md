# As Expected? Efficient LLM Planning via Expectation–Observation Consistency Verification

Qingfu Zhu Huanyang Zheng Ting Liu Wanxiang Che Harbin Institute of Technology qfzhu@ir.hit.edu.cn huanyzheng@ir.hit.edu.cn

## Abstract

LLM planning often relies on reactive executors (e.g., ReAct) that generate a single action at each step to remain grounded in the latest observations, incurring substantial inference cost. Batched executors reduce this cost by predicting multiple actions in a single model invocation, yet they are prone to compounding errors, as intermediate observations are ignored. We propose an expectation– observation consistency verification (EOCV) mechanism for batched executors, preserving grounding while improving efficiency. The mechanism generates expectations of action outcomes and compares them with the resulting observations. Consistency permits continued execution, whereas mismatches trigger early termination and replanning from the current observation. Experiments on text-based and multimodal benchmarks across multiple backbones show broadly comparable average task performance between EOCV and reactive executors, with a 29–35% reduction in average weighted token usage. Moreover, the mechanism is model-agnostic and remains effective when integrated into state-of-the-art planners.

## 1 Introduction

Planning aims to generate action sequences to achieve goals under constraints and uncertainty (Ghallab et al., 2004). In practice, plans need to be grounded in execution feedback (Castaman et al., 2021; Guo et al., 2023), motivating an iterative observation–action loop. ReAct (Yao et al., 2022) exemplifies this paradigm and has emerged as a foundational approach.

Existing LLM planning methods often adopt ReAct as the executor and equip it with auxiliary controller-level techniques such as task decomposition (Erdogan et al., 2025), search (Hao et al., 2023), and reflection (Shinn et al., 2023; Kim et al.,

![](images/c4da1404a4a06f69b85e0eb3b63f1cec600f4d097a5317eface501d0d7b80416.jpg)  
Figure 1: Executor paradigms for LLM planning. Given the query “Put apple in fridge” and initial observation O<sub>1</sub>: “The fridge is closed. You have an apple”, the reactive executor remains grounded but inefficient, whereas the batched executor is efficient but error-prone. Ours verifies expectations against observations to continue or replan batched actions selectively. Intermediate reasoning steps are omitted for clarity.

2025). Nevertheless, the underlying ReAct execution paradigm remains unchanged: at each iteration, the agent selects a single next action given the latest observation (as shown in Figure 1). We refer to this class of step-wise executors as reactive executors. While this reactive paradigm improves robustness by remaining closely grounded in execution feedback, it causes inference costs to scale with the action sequence length, thereby limiting planning efficiency.

A natural alternative is the batched executor (Singh et al., 2022; Liu et al., 2024). It generates multiple actions per model invocation, reducing repeated context prefilling and the number of internal reasoning phases. However, batched executors weaken step-wise grounding because feedback from earlier actions is unavailable at generation time. As a result, later actions can become inconsistent with the latest observation, increasing the risk of compounding errors.

We propose an expectation–observation consistency verification (EOCV) mechanism that combines efficient batched generation with reactive grounding. The mechanism augments action batches with expectations of their execution outcomes. These expectations are verified against observations as the actions are executed. Consistency permits continued execution within the batch, whereas a mismatch triggers early termination and regeneration from the latest observation.

EOCV is model-agnostic. It can be integrated into state-of-the-art (SOTA) planners and remains compatible with controller techniques such as task decomposition, search, and reflection.<sup>1</sup> We evaluate EOCV on text-based and multimodal benchmarks, comparing against strong reactive baselines and representative batched variants. Across multiple model backbones, EOCV reduces average weighted token usage by 29–35%, with task performance broadly comparable to that of reactive baselines on average. The contributions of the paper are summarized as follows:

• We propose EOCV, a step-wise verification mechanism that connects batched action generation with reactive grounding.

• EOCV is model-agnostic and compatible with SOTA planners augmented with various controller-level techniques.

• Experiments on text-based and multimodal benchmarks demonstrate weighted token savings across backbones, with broadly comparable average task performance.

## 2 Preliminaries

We study a planning executor that maps a task query q to executable actions in an interactive environment. The query can be a user instruction or a subtask produced by a planning controller. At step $t ,$ the executor takes as input the current observation $O t$ and an execution context $c _ { t } .$ , and outputs action $a _ { t }$ under policy π:

$$
a _ { t } = \pi ( q , o _ { t } , c _ { t } ) .\tag{1}
$$

The context $c _ { t }$ summarizes the interaction history and auxiliary signals up to step t. It includes past action–observation pairs and may also contain internal states such as reasoning traces. We consider two executor paradigms: reactive executors and batched executors.

## 2.1 Reactive Executor

A reactive executor produces one action per model invocation and updates its context after each action– observation transition. ReAct (Yao et al., 2022) is a representative instance that maintains intermediate reasoning traces within the context. This design grounds action selection in the latest observations, but requires context prefilling and a new internal reasoning phase for each action, increasing inference cost with sequence length.

## 2.2 Batched Executor

A batched executor generates an action sequence within a single model invocation. At step $t ,$ it conditions on the query $q ,$ the observation $o _ { t } .$ , and the context $c _ { t }$ , and generates a batch of actions autoregressively. The context is expanded with the previously generated actions in the same batch.

$$
\begin{array} { r l r } { a _ { j } = \pi ( q , o _ { t } , c _ { j } ) , } & { { } } & { } \\ { c _ { j + 1 } = c _ { t } \oplus a _ { t : j } , } & { { } } & { j = t , \ldots , t + B - 1 , } \end{array}\tag{2}
$$

where $a _ { t : j } = ( a _ { t } , \dotsc , a _ { j } )$ denotes the action prefix generated so far, and ⊕ denotes context concatenation. B is a batch budget that upper-bounds the number of generated actions. In practice, the realized sequence length may be determined dynamically at generation time when future environment feedback is difficult to anticipate.

Although actions are generated autoregressively, each batch uses one context prefill and one internal reasoning phase, reducing inference cost relative to reactive execution. However, the entire batch is generated before execution begins and is then executed sequentially. When generating $a _ { j }$ the executor cannot condition on the intermediate observations $O t + 1 { : } j$ , which only become available after executing $a _ { t : j - 1 }$ . Later actions may therefore become inconsistent with the latest observation, leading to compounding errors.

![](images/ba9a58d803c4382abc36a001c566f5dea2fa437eb044cd20ded13f8f6a642e7b.jpg)  
Figure 2: An overview of EOCV. The executor generates future action batches with expectations, executes them sequentially, and verifies expectations against the resulting observations. Consistency permits continued execution within the batch, whereas a mismatch triggers early termination and replanning from the latest observation.

## 3 Method

To leverage the efficiency of batched executors while preserving the step-wise grounding characteristic of reactive executors, we introduce an expectation–observation consistency verification (EOCV) mechanism. As illustrated in Figure 2, EOCV augments a batched executor by associating actions with explicit expectations of their execution outcomes. These expectations are verified against the observations returned during execution. Verification is performed step by step: success enables execution to proceed within the batch, whereas mismatches trigger early termination and regeneration from the latest observation. We next describe expectation generation (§3.1) and expectation–observation verification (§3.2).

## 3.1 Expectation Generation

EOCV augments the batched executor (§2.2) with an explicit expectation for action $a _ { j }$

$$
\hat { o } _ { j + 1 } = \pi ( q , o _ { t } , c _ { j } , a _ { j } ) ,\tag{3}
$$

where $\hat { o } _ { j + 1 }$ denotes the expectation associated with action $a _ { j }$ . We represent this expectation as a compact textual description of the action’s anticipated outcome. This expectation has the same form in both text-based and multimodal environments; verification compares it with the returned text or image observation (see §3.2 for details). Expectation generation uses only information available before execution, preserving compatibility with batched execution.

Expectation generation is unnecessary for actions whose observations are inherently unpredictable. For information-seeking actions (e.g., look or examine), generating specific expectations increases generation and verification cost without providing a reliable consistency signal. EOCV therefore adopts Expectation Skipping: these actions are executed, but their expectation generation and verification are omitted. In the main experiments, we identify such actions with a small predefined set, which is practical for the modest action spaces in our benchmarks. We also evaluate an automatic variant in which the executor emits an action-is-skippable tag to invoke skipping. This automatic variant closely matches the predefined strategy in selecting skippable actions, downstream task performance, and efficiency. Detailed results are reported in §4.6.

## 3.2 Expectation–Observation Verification

Expectation–observation verification evaluates whether the realized environment feedback matches the expectation predicted during batched generation. After executing action $a _ { j }$ , the environment returns an observation $o _ { j + 1 }$ , which is compared against the corresponding expectation $\hat { o } _ { j + 1 }$ whenever an expectation is available:

$$
\boldsymbol { v } _ { j + 1 } = \boldsymbol { V } ( o _ { j + 1 } , \hat { o } _ { j + 1 } ) ,\tag{4}
$$

where the function $V ( \cdot )$ returns a scalar consistency score. Verification succeeds above threshold $\tau _ { \ast }$ whose effect is analyzed in Appendix B.4.

If verification succeeds, execution proceeds to the next action in the batch without another executor invocation. Otherwise, execution is terminated immediately, and the remaining actions are discarded. This intervention limits further execution based on expectations that no longer match the environment. EOCV benefits from batched execution by allowing multiple actions to share the same prefilled context and reasoning. Appendix A.1 analyzes these token savings.

Information-seeking actions bypass verification. Any subsequent non-information-seeking actions in the same batch are discarded, as they were generated without access to the new information.

The implementation of V depends on the observation modality. For text observations, we use cosine similarity between expectation and observation embeddings. For visual observations, we prompt a pretrained VLM to assess consistency.

This verification procedure assesses whether action outcomes are consistent with their expectations. Such consistency, however, does not guarantee strategy correctness or prevent irreversible errors. Reactive and batched baselines provide neither guarantee.

## 4 Experiments

This section evaluates the effectiveness, efficiency, and generality of EOCV in interactive planning. We first describe the benchmarks, baselines, and implementation details (§4.1–§4.3), and then present the main results and analyses (§4.5–§4.6).

## 4.1 Benchmarks

We evaluate EOCV on a benchmark suite that covers both text-based and multimodal interactive planning. ALFWorld (Shridhar et al., 2021) and ScienceWorld (Wang et al., 2022) provide established text-based environments with partial observability and multi-step action sequences. EB-Habitat (Yang et al., 2025) extends the evaluation to multimodal household operation, where the agent must act from visual observations. The suite spans tasks from short interactions to long-horizon execution, supporting evaluation across different levels of planning complexity.

## 4.2 Baselines

Following the executor–controller formulation in §2, we evaluate baselines at two levels. At the executor level, we evaluate two primary baselines: a ReAct-based reactive executor (Yao et al., 2022) and a batched executor. The batched executor generates multiple actions per model invocation. Both baselines share the executor implementation used by EOCV, enabling a direct comparison of execution strategies. We additionally evaluate AdaPlanner (Sun et al., 2023), which uses code assertions to check subgoal completion and triggers plan refinement when these checks fail. Its implementation and shared in-context example are detailed in Appendices D and E.2. We also evaluate a batched variant that replans when the environment reports an action failure, without generating expectations (Fail-RP). At the planner level, we use SOTA planner baselines that pair reactive executors with three representative controller techniques: ReflAct, Reflexion, and MPO. These baselines allow us to evaluate whether EOCV remains effective in stronger planning pipelines.

ReflAct. This controller technique provides an inline enhancement for reactive executors by shifting executor reasoning from merely planning the next action to continuously reflecting on the agent’s state relative to its goal (Kim et al., 2025).

Reflexion. This controller technique augments reactive executors with a memory mechanism that stores feedback from previous episodes to refine subsequent executions (Shinn et al., 2023).

MPO. This controller technique decomposes high-level tasks into structured subgoals and provides step-level guidance to the executor (Xiong et al., 2025).

## 4.3 Implementation Details

The main executor comparisons use Qwen3.5- 4B, Qwen3.5-9B, and DeepSeek-V3.2 (Liu et al., 2025) on ALFWorld; GPT-5.5, DeepSeek-V4.1- Flash, and DeepSeek-V3.2 on ScienceWorld; and Qwen3.6-Flash and Qwen3.6-Plus on EB-Habitat. We use greedy decoding with reasoning disabled for the DeepSeek models, Qwen3.6-Flash, and Qwen3.6-Plus. For GPT-5.5, we set the reasoning effort to low. For Qwen3.5-4B and Qwen3.5- 9B, we set enable-thinking to on. For verification (§3.2), text-based experiments use the all-mpnetbase-v2 sentence-transformer model (Reimers and Gurevych, 2019) with a cosine threshold of 0.6, and multimodal experiments use the executor backbone as the prompted VLM verifier. Batched and EOCV-enhanced executors generate at most five future actions per model invocation. Reflexion uses five iterations on ALFWorld and three each on ScienceWorld and EB-Habitat.

<table><tr><td>Method</td><td colspan="3">ALFWorld</td><td colspan="2">ScienceWorld</td><td colspan="2">EB-Habitat</td></tr><tr><td></td><td>Reward</td><td>Tokens</td><td>Time</td><td>Reward</td><td>Tokens</td><td>Reward</td><td>Tokens</td></tr><tr><td>Reactive</td><td>88.81</td><td>1.00×</td><td>1.00×</td><td>89.37</td><td>1.00×</td><td>66.94</td><td>1.00×</td></tr><tr><td>Batched</td><td>83.58</td><td>0.57×</td><td>0.53×</td><td>82.56</td><td>0.41×</td><td>56.56</td><td>0.38×</td></tr><tr><td>Ours</td><td>88.56</td><td>0.65×</td><td>0.59×</td><td>88.87</td><td>0.69×</td><td>67.77</td><td>0.71×</td></tr></table>

Table 1: Average executor results on ALFWorld, ScienceWorld, and EB-Habitat. Reward is the mean score on [0, 100]; Tokens denotes weighted token usage; Time is summed per-task wall-clock time. Token and time ratios are normalized to Reactive (1.00×) within each backbone before averaging. ALFWorld time averages cover the locally deployed Qwen3.5-4B and Qwen3.5-9B models.

We report wall-clock time only for Qwen3.5- 4B and Qwen3.5-9B, each deployed locally using vLLM on a single NVIDIA A100-SXM4-80GB GPU. Local deployment enables more controlled timing comparisons by avoiding network and API server variability.

We follow the standard evaluation protocol for each benchmark. For ALFWorld, we evaluate the full unseen set of 134 tasks with a budget of 50 actions per task. For ScienceWorld, we follow Lin et al. (2023) and evaluate the first 10 test instances of each task type, or all available instances when fewer than 10 are provided, yielding 271 instances with a budget of 70 actions per instance. For EB-Habitat, we evaluate 50 tasks in each of the six categories, for a total of 300 tasks. For MPO, we restrict evaluation to instances with meta-plans available in the official repository: all ALFWorld tasks and 211 of the 271 ScienceWorld instances. We do not evaluate MPO on EB-Habitat because no meta-plans are available. Appendix E provides the prompts and in-context examples for these benchmarks.

## 4.4 Evaluation Metrics

Task performance is reported as reward on a unified [0, 100] scale. The underlying environment metrics are binary success for ALFWorld, graded progress for ScienceWorld, and task-progress score for EB-Habitat, with the latter rescaled to [0, 100]. We normalize scores where necessary and report averages over evaluation episodes. Execution efficiency is measured by a weighted token count. We aggregate prompt and completion tokens across all evaluation episodes and weight them by their pricing. Detailed weights are listed in Appendix A.

## 4.5 Main Results

Table 1 compares executor variants without controllers. Across ALFWorld, ScienceWorld, and EB-Habitat, the reactive executor remains well grounded but uses the most tokens, whereas the batched executor lowers cost with a clear performance penalty. EOCV achieves broadly comparable average task performance to the reactive executor while reducing average weighted token usage to 0.65×, 0.69×, and 0.71× on the three benchmarks, respectively. ALFWorld wall-clock time falls to 0.59× on average across the two locally deployed Qwen models.

![](images/13874905b65c87208bd038ce14b60cbf3a5914718f3c087accf61a1539d056df.jpg)  
Figure 3: EOCV with different planner controllers. The x-axis shows the proportional reduction in weighted token usage, and the y-axis shows the change in reward, both relative to the corresponding planner with a reactive executor. Colors indicate controllers, and marker shapes indicate backbone models. Hollow, filled, and halffilled markers indicate ALFWorld, ScienceWorld, and EB-Habitat, respectively.

Figure 3 evaluates EOCV in combination with planner controllers across text-based and multimodal benchmarks. Across the evaluated controllers, EOCV consistently reduces weighted token usage while maintaining broadly comparable task performance overall. These results suggest that EOCV can serve as an executor component across planners without modifying their controllers. Table 11 in Appendix B.5.2 reports detailed reward and token usage for the controller comparisons across all three benchmarks.

Across 28 equally weighted paired configurations, the mean reward difference between

<table><tr><td rowspan="2">Verifier</td><td colspan="4">ALFWorld</td><td colspan="4">ScienceWorld</td><td colspan="2">EB-Habitat</td></tr><tr><td>Prec.</td><td>Rec.</td><td>Reward</td><td>Tokens</td><td>Prec.</td><td>Rec.</td><td>Reward</td><td>Tokens</td><td>Prec.</td><td>Rec.</td></tr><tr><td>Embedding</td><td>100.00</td><td>100.00</td><td>94.78</td><td>1.00×</td><td>96.46</td><td>100.00</td><td>85.42</td><td>1.00×</td><td></td><td></td></tr><tr><td>LLM/VLM</td><td>100.00</td><td>100.00</td><td>94.78</td><td>1.06×</td><td>97.32</td><td>100.00</td><td>85.93</td><td>1.15×</td><td>89.38</td><td>87.07</td></tr></table>

Table 2: Verifier comparison. Precision (Prec.) and recall (Rec.) use 300 randomly sampled, manually labeled pairs per benchmark. Pairs are collected with DeepSeek-V3.2 on the text benchmarks, while Qwen3.6-Flash is used for EB-Habitat. Tokens are normalized to embedding verification within each text benchmark; dashes mark inapplicable entries.

![](images/9de211f0b590a534aaa2a57e68beca6b7f3bb8e8b0e275cd56821c216a08c29d.jpg)  
Figure 4: Executor comparisons on (a) ALFWorld with Qwen3.5-9B and (b) ScienceWorld with DeepSeek-V4.1-Flash. Reward is the mean score on [0, 100]; Tokens denotes weighted token usage normalized to ReAct (1.00×) within each setting. AdaPlanner is evaluated only on ALFWorld. Dashed lines mark the corresponding ReAct baseline.

EOCV and the corresponding reactive baselines is +0.0458 points (95% confidence interval based on the paired t procedure: [−0.3406, +0.4321]). A two-sided paired t-test finds no statistically significant difference in mean reward $\left( p = 0 . 8 0 9 8 \right)$ . The near-zero observed mean difference is consistent with broadly comparable average task performance across the evaluated configurations. Details are provided in Appendix B.5.1.

## 4.6 Analysis

The analysis first assesses the effectiveness and reliability of EOCV’s joint expectation–verification mechanism. It then examines the efficiency gains from expectation skipping and component contributions through progressive ablation. Finally, case studies illustrate execution behavior, and sensitivity analysis evaluates parameter effects.

Effectiveness and Efficiency. Figure 4 evaluates explicit expectation–verification across representations and granularities in text-based environments. Fail-RP relies solely on environment failure signals, whereas AdaPlanner encodes expectations and verification as executable assertions for subgoals that may span multiple actions. EOCV combines natural-language expectations with semantic verification at the action level. It uses more weighted tokens than Fail-RP but increases reward by 3.73 and 5.21 points on ALFWorld and Science-World, respectively, reaching performance comparable to ReAct. This performance is achieved with 41% and 30% fewer weighted tokens than ReAct, respectively, offering a favorable balance between reward and token usage. On ALFWorld, EOCV also achieves higher reward with fewer tokens than AdaPlanner. These baselines depend on explicit failure feedback or states accessible to code assertions; EOCV supports verification from visual observations when such information is unavailable.

Verifier Reliability. Verifier judgments determine whether EOCV continues a batch or replans. Averaged across backbone–benchmark settings, observations are judged consistent with their corresponding expectations for 66.78% of non-skipped classified actions in text-based environments and 28.55% in multimodal environments. Observations judged inconsistent trigger early termination and replanning (Appendix B.2).

To assess verifier reliability, two graduate students independently label 300 randomly sampled expectation–observation pairs per benchmark and resolve initial disagreements (approximately 0.67% of all pairs) through discussion. On text benchmarks, embedding verification achieves precision, recall, and task reward comparable to the Qwen3.5- Flash LLM verifier with lower weighted token usage (Table 2). These results favor embedding verification as an efficient alternative to LLM verification for text observations and support VLM verification for visual observations. Appendix B.3 provides further analysis of verification difficulty on EB-Habitat.

Occasional verifier errors do not necessarily lead to task failure. Rejecting a consistent outcome triggers additional replanning, whereas accepting an inconsistent outcome can delay it. Subsequent verification may still detect deviations and prompt replanning from the latest observation. This loop provides opportunities to limit the effects of individual verification errors on subsequent execution.

<table><tr><td></td><td colspan="2">Batched Executor</td><td colspan="3">Ours</td></tr><tr><td></td><td>Step Action</td><td>Observation</td><td>Action</td><td>Expectation</td><td>Observation</td></tr><tr><td>t</td><td>move(glass cup, oven)</td><td>That can&#x27;t be moved there, because the oven isn&#x27;t open.</td><td>move(glass cup, oven)</td><td>You move the glass cup to the</td><td>That can&#x27;t be moved there, because the oven isn&#x27;t open.</td></tr><tr><td></td><td>t+1 activate(oven)</td><td>The oven is now activated.</td><td>close(oven)</td><td>oven. The oven is now closed.</td><td>Terminate &amp; Replan</td></tr></table>

Table 3: ScienceWorld case study with DeepSeek-V4.1-Flash. Steps are aligned at each executor’s failed move. Red denotes batched actions executed after the failure; gray denotes actions withheld by EOCV. The full batches are shown in Table 12.
<table><tr><td colspan="5">Batched Executor</td><td colspan="4">Ours</td></tr><tr><td></td><td>Step Action</td><td></td><td>Observation Annotation</td><td></td><td>Action</td><td>Expectation</td><td>Observation Annotation</td><td></td></tr><tr><td>t</td><td>pick up the pear</td><td></td><td>1 nothing is picked up</td><td>pick up the pear</td><td></td><td>The robot picks up a pear from the left counter.</td><td></td><td>nothing is picked up</td></tr><tr><td></td><td>t+1 navigate to the sofa</td><td></td><td></td><td>agent moves to the sofa navigate to the</td><td>sofa</td><td>Skip verification</td><td>Terminate &amp; Replan</td><td></td></tr></table>

Table 4: Multimodal EB-Habitat case study with Qwen3.6-Flash. The batched executor continues with a stale plan after the shared pick-up step, whereas EOCV detects an expectation–observation mismatch and triggers replanning. Annotations describe visual observations for human readers and are not accessible to the agent. The full trajectory is reported in Table 15; dashes indicate not applicable.

<table><tr><td>Setting</td><td>Reward ↑</td><td>Tokens (M) ↓</td></tr><tr><td>Full Model</td><td>94.78</td><td>2.45</td></tr><tr><td>w/o Expectation Skipping</td><td>94.78</td><td>3.96</td></tr><tr><td>w/o Verification</td><td>90.30</td><td>1.73</td></tr></table>

Table 5: Progressive ablation on ALFWorld with DeepSeek-V3.2. Components are removed cumulatively; w/o Verification omits both expectation generation and verification.

Skipping Strategy. Expectation skipping avoids unnecessary expectation generation and verification for information-seeking actions. The main experiments use predefined rules because these actions are readily identifiable within the benchmarks small action spaces. Skipping rates average 64.0% in text-based environments and 59.0% in multimodal environments across backbone–benchmark settings. Appendix B.2 provides detailed distributions.

The executor can also predict these skipping decisions to reduce manual specification. On 400 randomly sampled actions from DeepSeek-V3.2 runs on ALFWorld, automatic skipping matches rulebased decisions with precision and recall above 99%. At the task level, the same strategy yields a similar reward of 94.03 versus 94.78 while using 0.92 times as many weighted tokens. These results support automatic skipping as an alternative in this setting. Appendix B.1 provides detailed results and a skip-only ablation.

Ablation Study. Table 5 reports a progressive ablation following component dependencies. Disabling expectation skipping increases weighted token usage while reward remains unchanged, indicating that skipping improves efficiency. Expectation generation and verification are then ablated as a unified mechanism because the generated expectations provide the reference for consistency verification. Token usage decreases but reward drops, indicating that this mechanism helps maintain task performance during batched execution.

Case Study. Tables 3 and 4 show representative text-based and multimodal failure modes of batched executors. After move(glass cup, oven) fails because the oven is closed, the batched executor continues with its planned activation, waiting, and inspection actions. EOCV discards the remaining wait and inspection actions and replans from the latest observation. A similar failure pattern appears in the EB-Habitat example with visual observations. The executor initially attempts to pick up the pear, but the following image observation contradicts the corresponding expectation, since the pear is not present in the scene. The EOCVenhanced executor detects this mismatch, discards the remaining actions, and replans from the latest observation. Complete versions of both examples are provided in Appendix C.

Hyperparameter Sensitivity. Average reward remains largely stable across the evaluated verification thresholds and batch budgets. Larger budgets reduce token usage with diminishing gains, while threshold choice has a modest effect on token usage (Appendix B.4).

## 5 Related Work

## 5.1 Reactive Executor

Reactive executors follow a closed-loop reason– act–feedback paradigm. SayCan (Ahn et al., 2022) samples candidate robot actions from an LLM and selects one using a value function. Inner Monologue (Huang et al., 2023) incorporates environment descriptions into its reasoning as action feedback. ReAct (Yao et al., 2022) makes reasoning explicit in this execution loop.

Selecting each next action requires another model invocation, repeating context prefilling and internal reasoning and increasing inference cost.

## 5.2 Batched Executor

Batched executors were initially explored in question-answering settings and later extended to interactive and embodied tasks. ReWOO (Xu et al., 2023) separates reasoning from tool observations, while Plan-and-Solve Prompting (Wang et al., 2023) decomposes tasks before sequential execution. ProgPrompt (Singh et al., 2022) uses conditional programs for embodied tasks.

To retain grounding during multi-step execution, AdaPlanner (Sun et al., 2023) uses code assertions to verify subgoals that may contain multiple actions, drawing on textual feedback and HTML states. HVR (Petruzzellis et al., 2025) validates plans and monitors execution using symbolic states and PDDL action conditions. More recently, SPACE (Yang et al., 2026) learns to adapt action-chunk length using boundary supervision from programmatic skills, while CheckVLA (Liu et al., 2026) verifies vision–language–action execution with a trained action-conditioned world model. EOCV shares the principle of revising plans from execution feedback with receding horizon planning (Castaman et al., 2021). It extends this principle to batched LLM agents by coupling expectation generation with semantic consistency verification. Explicit expectations provide the reference for verification. The verifier compares these expectations with textual or visual observations to determine whether to continue the batch or replan. This verification does not require a manually specified symbolic transition model.

## 5.3 Planning Controller

Controllers complement executors through task decomposition, search, and reflection. Multi-agent approaches such as Reason–Plan–ReAct (Molinari and Ciravegna, 2025) introduce a high-level controller responsible for subtask segmentation. Graph-based methods including HyperTree Planning (Gui et al., 2025) leverage structured data representations to support dynamic task expansion and iterative refinement. Context management methods such as AutoManuals (Cai et al., 2025) provide the executor with detailed environment exploration traces. The self-improvement framework Meta-Policy Reflexion (Wu et al., 2025) stores and applies reflective feedback through a dual-track policy architecture.

Despite their architectural diversity, most existing controllers rely on similar reactive executors, allowing EOCV to replace the executor without modifying the controller.

## 6 Conclusion

This work introduces EOCV to combine efficient batched generation with reactive grounding through expectation–observation consistency verification. Verification determines whether to continue the batch or replan from the latest observation. Across ALFWorld, ScienceWorld, and EB-Habitat, EOCV reduces average weighted token usage by 29–35% while maintaining average task performance comparable to reactive execution. Token savings persist with different planner controllers across language and vision-language backbones, supporting EOCV as a model-agnostic approach to efficient interactive planning.

## Limitations

Verification assesses consistency between observed action outcomes and their corresponding expectations. Consistency alone does not establish the suitability of the underlying strategy for the task. Verification occurs after action execution. Replanning can therefore revise subsequent actions but cannot reverse the effects of irreversible actions. Addressing these limitations would require additional mechanisms for assessing progress toward the task goal and evaluating action consequences before execution.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, and 1 others. 2022. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691.

Kuntai Cai, Juncheng Liu, Xianglin Yang, Zhaojie Niu, Xiaokui Xiao, and Xing Chen. 2025. Beyond manuals and tasks: Instance-level context learning for llm agents. arXiv preprint arXiv:2510.02369.

Nicola Castaman, Enrico Pagello, Emanuele Menegatti, and Alberto Pretto. 2021. Receding horizon task and motion planning in changing environments. Robotics and Autonomous Systems, 145:103863.

Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. 2025. Plan-and-act: Improving planning of agents for long-horizon tasks. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 15419–15462. PMLR.

Malik Ghallab, Dana Nau, and Paolo Traverso. 2004. Automated Planning: theory and practice. Elsevier.

Runquan Gui, Zhihai Wang, Jie Wang, Chi Ma, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Defu Lian, Enhong Chen, and Feng Wu. 2025. HyperTree planning: Enhancing LLM reasoning via hierarchical thinking. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 20694–20722. PMLR.

Huihui Guo, Fan Wu, Yunchuan Qin, Ruihui Li, Keqin Li, and Kenli Li. 2023. Recent trends in task and motion planning for robotics: A survey. ACM Computing Surveys, 55(13s):1–36.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world

model. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 8154–8173, Singapore. Association for Computational Linguistics.

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Noah Brown, Tomas Jackson, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. 2023. Inner monologue: Embodied reasoning through planning with language models. In Proceedings ofThe 6th Conference on Robot Learning.

Jeonghye Kim, Sojeong Rhee, Minbeom Kim, Dohyung Kim, Sangmook Lee, Youngchul Sung, and Kyomin Jung. 2025. ReflAct: World-grounded decision making in LLM agents via goal-state reflection. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 33421– 33453, Suzhou, China. Association for Computational Linguistics.

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. 2023. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. In Advances in Neural Information Processing Systems, volume 36, pages 23813–23825. Curran Associates, Inc.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, and 1 others. 2025. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Siyuan Liu, Jiawei Du, Sicheng Xiang, Zibo Wang, and Dingsheng Luo. 2024. Long-horizon embodied planning with implicit logical inference and hallucination mitigation. arXiv preprint arXiv:2409.15658.

Yushan Liu, Peibo Sun, Xintao Chao, Zhenyang Yang, Yifan Xie, Lingfeng Zhang, Shoujie Li, Chenyu Tang, Fang Chen, Xiao-Ping Zhang, and 1 others. 2026. Checkvla: Execution-time verification with actionconditioned world model for long-horizon mobile manipulation. arXiv preprint arXiv:2607.26789.

Gianni Molinari and Fabio Ciravegna. 2025. Reasonplan-react: A reasoner-planner supervising a react executor for complex enterprise tasks. arXiv preprint arXiv:2512.03560.

Flavio Petruzzellis, Cristina Cornelio, and Pietro Lio. 2025. Hierarchical planning for complex tasks with knowledge graph-RAG and symbolic verification. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 49105–49127. PMLR.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings ofthe 2019 Conference on Empirical

Methods in Natural Language Processing. Association for Computational Linguistics.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652. Curran Associates, Inc.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In Proceedings ofthe International Conference on Learning Representations.

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2022. Progprompt: Generating situated robot task plans using large language models. In Workshop on Language and Robotics at CoRL 2022.

Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. 2023. Adaplanner: Adaptive planning from feedback with language models. In Advances in Neural Information Processing Systems, volume 36, pages 58202–58245. Curran Associates, Inc.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Planand-solve prompting: Improving zero-shot chain-ofthought reasoning by large language models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2609–2634, Toronto, Canada. Association for Computational Linguistics.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. 2022. ScienceWorld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Chunlong Wu, Ye Luo, Zhibo Qu, and Min Wang. 2025. Meta-policy reflexion: Reusable reflective memory and rule admissibility for resource-efficient llm agent. arXiv preprint arXiv:2509.03990.

Weimin Xiong, Yifan Song, Qingxiu Dong, Bingchan Zhao, Feifan Song, XWang, and Sujian Li. 2025. MPO: Boosting LLM agents with meta plan optimization. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 3914– 3935, Suzhou, China. Association for Computational Linguistics.

Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, and Dongkuan Xu. 2023. Rewoo: Decoupling reasoning from observations for efficient augmented language models. CoRR, abs/2305.18323.

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. 2025. EmbodiedBench: Comprehensive benchmarking multi-modal large language models for visiondriven embodied agents. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 70576–70631. PMLR.

Yanting Yang, Can Jin, Jinman Zhao, Jiahao Wu, Yang Zhou, Zhepeng Wang, Zhendong Wang, Mu Zhou, and Dimitris N Metaxas. 2026. Act more, decide less: Skill-guided adaptive action chunking for long-horizon llm agents. arXiv preprint arXiv:2609.02042.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. In Proceedings ofthe International Conference on Learning Representations.

## A Token Accounting and Efficiency Analysis

We measure token cost by weighting input and output token counts according to their relative prices and summing the contributions. Table 6 lists these weights, with the output-token weight normalized to 1.

## A.1 Sources of Token Savings

EOCV gains efficiency by allowing multiple actions to share one executor call. A reactive executor generates one action per call, repeatedly receiving the task instructions and interaction history and, when reasoning is enabled, starting a new reasoning phase. EOCV instead generates up to five actions per call and executes them until the batch ends or replanning is needed. Actions within a batch share the same context prefill and internal reasoning, reducing repeated input and potentially reducing reasoning-token generation.

Batching also adds output costs: EOCV generates expectations, and some proposed actions may be discarded when execution stops early. These tokens are counted even when the actions are never executed. EOCV therefore gains a net advantage when input and reasoning savings outweigh the additional action-response output and any modelbased verification cost.

## A.2 ALFWorld Case Study

Figure 5 compares weighted token usage of executor-only ReAct and EOCV runs using

![](images/f43c43e822e2a88ac002023b56725a2b0dfb557228b0951e2767ea156a1d7d08.jpg)

Figure 5: ALFWorld token-cost decomposition without caching. Bar segment heights show price-weighted token costs, not raw token counts. Input, RT, and AT have weights 0.125, 1, and 1. Each model’s ReAct total cost is normalized to 100%.
<table><tr><td>Model</td><td>Input Weight</td><td>Output Weight</td></tr><tr><td>Qwen3.5-4B/9B</td><td>1/8</td><td>1</td></tr><tr><td>GPT-5.5</td><td>1/6</td><td>1</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>1/4</td><td>1</td></tr><tr><td>DeepSeek-V3.2</td><td>2/3</td><td>1</td></tr><tr><td>Qwen3.6-Plus/Flash</td><td>1/6</td><td>1</td></tr></table>

Table 6: Token weights used to compute weighted token usage. The output-token weight is normalized to 1; each input-token weight is the model’s input-to-output price ratio.

Qwen3.5-4B and Qwen3.5-9B on the same 134 ALFWorld tasks. We separate output into reported internal reasoning tokens (RT) and all remaining action-response tokens (AT), including visible thoughts, commands, expectations, formatting, and unexecuted suffixes. In both pairs, EOCV’s savings in input and RT costs outweigh its increased AT costs, reducing weighted usage to 0.703× ReAct for 4B and 0.588× for 9B. RT contributes most of the savings.

## A.3 Effect of Prefix Caching

Prefix caching reuses stored key–value states for matching input prefixes, reducing repeated prefill computation and the price of cached input tokens. We recompute token costs for the same runs by assigning weights of 0.025 and 0.125 to cached and uncached input tokens, respectively, while keeping output-token weights unchanged.

Figure 6 gives EOCV costs of 0.650× and

![](images/32f6517d1fb161b9e03803f31f3dbfd2264197834587ba0ed8a19bb24b21101b.jpg)  
Figure 6: ALFWorld token-cost decomposition with caching. Bar segment heights show price-weighted token costs, not raw token counts. Cached, uncached Input, RT, and AT have weights 0.025, 0.125, 1, and 1. Each model’s ReAct total cost is normalized to 100

0.525× ReAct for 4B and 9B. Caching discounts input but leaves reasoning-token generation costs unchanged, making RT a larger share of ReAct’s total cost. EOCV’s RT savings therefore remain valuable, and its relative cost advantage increases in these comparisons.

## B Additional Experimental Results

This appendix complements the main experimental section with detailed results and analysis omitted from the main text.

## B.1 Expectation Skipping

The main experiments use a predefined rule set to skip expectation generation and verification for information-seeking actions. The actions themselves are still executed. For the automatic variant, the executor is prompted at each step to output an action-is-skippable tag. When this tag is emitted, EOCV skips both expectation prediction and verification for that action. Table 7 reports the precision and recall of the automatic tag, together with the downstream reward and token usage when the tag replaces the predefined rule set.

Skipping Alone. We further evaluate a skip-only ablation on ALFWorld with DeepSeek-V3.2. It executes information-seeking actions without expectation generation or verification, but limits each batch to at most one non-information-seeking action and terminates the batch immediately after executing it. It achieves a reward of 94.78, matching both Reactive and full EOCV, while using 3.23M weighted tokens, compared with 3.79M for Reactive and 2.45M for full EOCV (Table 9). This amounts to approximately 15% fewer tokens than Reactive but 32% more than full EOCV. Skipping alone therefore cannot recover the token savings enabled by verified batched execution.

<table><tr><td>Strategy</td><td>Prec.</td><td>Rec.</td><td>Reward</td><td>Tokens</td></tr><tr><td>Predefined</td><td></td><td></td><td>94.78</td><td>1.00×</td></tr><tr><td>Automatic</td><td>99.68</td><td>99.36</td><td>94.03</td><td>0.92×</td></tr></table>

Table 7: Predefined and automatic expectation skipping on ALFWorld with DeepSeek-V3.2. Precision (Prec.) and recall (Rec.) are computed on 400 labeled actions for the automatic action-is-skippable tag. Token usage is normalized to predefined skipping; dashes indicate that a metric is not applicable.
<table><tr><td>Backbone</td><td>Skip</td><td>Pass</td><td>Fail</td></tr><tr><td>ALFWorld</td><td></td><td></td><td></td></tr><tr><td>DeepSeek-V3.2</td><td>75.55</td><td>11.60</td><td>12.85</td></tr><tr><td>Qwen3.5-4B</td><td>80.39</td><td>8.64</td><td>10.96</td></tr><tr><td>Qwen3.5-9B</td><td>77.00</td><td>10.78</td><td>12.23</td></tr><tr><td>ScienceWorld</td><td></td><td></td><td></td></tr><tr><td>DeepSeek-V3.2</td><td>49.56</td><td>33.06</td><td>17.38</td></tr><tr><td>GPT-5.5</td><td>45.65</td><td>44.44</td><td>9.91</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>55.73</td><td>35.81</td><td>8.46</td></tr><tr><td>EB-Habitat</td><td></td><td></td><td></td></tr><tr><td>Qwen3.6-Flash</td><td>60.42</td><td>9.99</td><td>29.59</td></tr><tr><td>Qwen3.6-Plus</td><td>57.63</td><td>13.41</td><td>28.97</td></tr></table>

Table 8: Distribution (%) of EOCV execution outcomes. Skip denotes actions for which expectation prediction and verification are omitted.

## B.2 Verification Outcomes

Table 8 reports three verification outcomes: Skip, where expectation generation and verification are bypassed for information-seeking actions; Pass, where the expectation is judged consistent with the observation; and Fail, where it is judged inconsistent. Actions without a recorded verification outcome are excluded.

Skipping accounts for a substantial share of actions across settings, reducing verification overhead. For verified actions, EOCV continues execution after a pass and interrupts it after a failure to limit error propagation.

## B.3 Verifier Difficulty Analysis

We analyze the same 300 randomly sampled, manually labeled expectation–observation pairs used in the EB-Habitat precision and recall evaluation in Table 2. Each sample includes an action, its expectation, and the resulting observation from a Qwen3.6-Flash execution trajectory; the same model also serves as the verifier. We use action execution outcomes as a proxy for verification difficulty. We classify failed actions as Easy cases because they leave the environment state unchanged, making the absence of an expected effect relatively easy to identify. Successful actions change the environment state, but execution success alone does not guarantee that the expected effect has occurred. These Hard cases require the verifier to interpret changes in object states or relations and determine whether they match the expectation, placing greater demands on its reasoning.

![](images/a22968c9a28c29f518bad992e3e8c3df101f1c56b924f41c71472cb9bdfa97a9.jpg)  
Figure 7: Analysis of verification threshold τ. Token usage is measured in millions (M).

We define accuracy within each group as the fraction of verifier decisions that agree with the manually assigned consistency labels. The verifier correctly classifies 202 of 232 Easy cases and 44 of 68 Hard cases, giving accuracies of 87.07% and 64.71%, respectively.

## B.4 Hyperparameter Sensitivity

These analyses use ALFWorld with DeepSeek-V3.2. Given ALFWorld’s binary rewards, we report mean reward and weighted token usage across four independent runs for each ablation setting to obtain more stable estimates.

Verification Threshold. Figure 7 studies how the verification threshold τ in expectation–observation verification affects the efficiency–reward trade-off. Across a broad range of thresholds, average reward remains largely stable, indicating that the verification criterion is not overly sensitive and does not require fine-grained tuning. Token consumption exhibits a slight U-shaped trend: overly permissive thresholds admit more inconsistent steps, increasing downstream correction cost, whereas overly strict thresholds trigger frequent early termination and regeneration, also increasing token usage. This stability motivates our use of an intermediate threshold.

![](images/d0aa38aec473e72a2720f2c7cd8698b0839c83dda41a0996ec8735e63d377dd2.jpg)  
Figure 8: Analysis of batch budget B. Token usage is measured in millions (M).

Batch Budget. Figure 8 examines the impact of the batch budget B. As B increases, token usage consistently decreases, reflecting improved efficiency from generating more actions within a single model invocation. At the same time, average reward remains largely stable across a broad range of B. When B becomes large, efficiency gains gradually saturate while reward shows only minor fluctuations, suggesting diminishing returns from further increasing the batch budget. We therefore use a moderate budget, which captures most of the efficiency gain before the saturation regime.

## B.5 EOCV Results

## B.5.1 Hypothesis Testing

We compare EOCV with Reactive across 28 configurations matched by benchmark, backbone, and controller setting. ALFWorld and ScienceWorld each contribute 12 pairs: three backbones evaluated without a controller or with ReflAct, Reflexion, or MPO. EB-Habitat contributes four pairs: Qwen3.6- Flash and Qwen3.6-Plus without controllers, plus Qwen3.6-Flash with ReflAct or Reflexion.

For configuration i, let $d _ { i } = r _ { i } ^ { \mathrm { E O C V } } - r _ { \mathrm { ' } }$ Reactive on the 0–100 reward scale, with each pair weighted equally irrespective of task count. A two-sided paired t-test of $H _ { 0 } : \mathbb { E } [ d _ { i } ] = 0$ at $\alpha = 0 . 0 5$ gives a mean difference of +0.0458 points (95% pairedt confidence interval: $\left[ - 0 . 3 4 0 6 , + 0 . 4 3 2 1 \right] )$ , with $t ( 2 7 ) = 0 . 2 4 3 0 ( p = 0 . 8 0 9 8 )$ . This analysis provides an exploratory assessment of the average reward difference across the evaluated configurations, whose shared backbones and evaluation tasks may induce dependence.

## B.5.2 Additional Results

Table 9 compares Reactive, Batched, and EOCV executors without controllers across all three benchmarks. Table 10 further breaks down EB-Habitat results by task category and vision-language backbone. For EB-Habitat, rewards are averaged across the six categories, while token totals are summed across categories and include vision-language verification costs; token ratios use these totals.

Table 11 pairs Reactive and EOCV executors with ReflAct, MPO, and Reflexion on each text benchmark, plus ReflAct and Reflexion on EB-Habitat.

## C Additional Case Studies

These additional case studies illustrate EOCV’s execution dynamics and the sources of its performance and efficiency gains.

The ScienceWorld examples in Tables 12 and 13 use DeepSeek-V4.1-Flash and GPT-5.5, respectively. In the first case, a cup cannot be placed in a closed oven, yet the naive batched executor continues with heating and inspection actions. EOCV detects the mismatch, terminates the planned suffix, and opens the oven before retrying. In the second case, both executors generate the same five-action plan to make sugar water. When collecting water fails, the naive executor continues adding sugar and mixing an incomplete set of ingredients, whereas EOCV interrupts the batch and replans how to fill the cup. These cases help explain how the EOCVenhanced batched executor avoids cascading errors that can arise in naive batched execution.

The Qwen3.5-4B example in Table 14 shows how the EOCV executor efficiently explores ALFWorld. EOCV executes four consecutive information-seeking actions while skipping verification. With new observations, the next executor call successfully picks up the watch. EOCV executes the five actions shown from t to t+4 in two executor calls, compared with five for ReAct. This reduces weighted token usage from 2,554 to 1,432.

In the multimodal EB-Habitat benchmark, the environment returns image observations rather than textual feedback alone. The Qwen3.6-Flash cases in Tables 15 (expanding Table 4) and 16 illustrate similar execution behavior with visual observations: EOCV verifies expected effects and replans when observations do not match expectations.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Executor</td><td rowspan="2">Reward</td><td rowspan="2">Input (M)</td><td rowspan="2">Output (M)</td><td colspan="2">Weighted Tokens</td><td colspan="2">Time</td></tr><tr><td>(M)</td><td>Ratio</td><td>(s)</td><td>Ratio</td></tr><tr><td>ALFWorld</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>Reactive</td><td>79.85</td><td>7.15</td><td>1.50</td><td>2.39</td><td>1.00×</td><td>17394.41</td><td>1.00×</td></tr><tr><td>Batched</td><td>73.88</td><td>5.44</td><td>1.03</td><td>1.71</td><td>0.71×</td><td>9603.92</td><td>0.55×</td></tr><tr><td>Ours</td><td>79.10</td><td>6.12</td><td>0.92</td><td>1.68</td><td>0.70×</td><td>11170.33</td><td>0.64×</td></tr><tr><td rowspan="3">Qwen3.5-9B</td><td>Reactive</td><td>91.79</td><td>6.15</td><td>1.95</td><td>2.72</td><td>1.00×</td><td>25 984.66</td><td>1.00×</td></tr><tr><td>Batched</td><td>86.57</td><td>4.56</td><td>0.87</td><td>1.44</td><td>0.53×</td><td>12973.72</td><td>0.50×</td></tr><tr><td>Ours</td><td>91.79</td><td>5.36</td><td>0.93</td><td>1.60</td><td>0.59×</td><td>13 804.76</td><td>0.53×</td></tr><tr><td rowspan="3">DeepSeek-V3.2</td><td>Reactive</td><td>94.78</td><td>5.52</td><td>0.11</td><td>3.79</td><td>1.00×</td><td></td><td></td></tr><tr><td>Batched</td><td>90.30</td><td>2.39</td><td>0.13</td><td>1.73</td><td>0.46×</td><td></td><td>1</td></tr><tr><td>Ours</td><td>94.78</td><td>3.31</td><td>0.24</td><td>2.45</td><td>0.65×</td><td></td><td>一</td></tr><tr><td>ScienceWorld</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">GPT-5.5</td><td>Reactive</td><td>94.30</td><td>27.17</td><td>0.48</td><td>5.01</td><td>1.00×</td><td></td><td></td></tr><tr><td>Batched</td><td>92.88</td><td>11.42</td><td>0.36</td><td>2.27</td><td>0.45×</td><td></td><td></td></tr><tr><td>Ours</td><td>94.81</td><td>15.69</td><td>0.50</td><td>3.11</td><td>0.62×</td><td></td><td></td></tr><tr><td rowspan="3">DS-V4.1-Flash</td><td>Reactive</td><td>86.46</td><td>32.45</td><td>0.97</td><td>9.08</td><td>1.00×</td><td></td><td></td></tr><tr><td>Batched</td><td>76.00</td><td>10.97</td><td>1.23</td><td>3.98</td><td>0.44×</td><td></td><td></td></tr><tr><td>Ours</td><td>86.39</td><td>21.11</td><td>1.06</td><td>6.34</td><td>0.70×</td><td></td><td></td></tr><tr><td rowspan="3">DeepSeek-V3.2</td><td>Reactive</td><td>87.36</td><td>29.81</td><td>0.55</td><td>20.42</td><td>1.00×</td><td></td><td>1</td></tr><tr><td>Batched</td><td>78.80</td><td>9.68</td><td>0.53</td><td>6.98</td><td>0.34×</td><td></td><td></td></tr><tr><td>Ours</td><td>85.42</td><td>20.99</td><td>1.06</td><td>15.06</td><td>0.74×</td><td></td><td></td></tr><tr><td>EB-Habitat</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Qwen3.6-Flash</td><td>Reactive</td><td>58.10</td><td>28.57</td><td>1.66</td><td>6.42</td><td>1.00×</td><td></td><td></td></tr><tr><td>Batched</td><td>48.88</td><td>7.41</td><td>0.62</td><td>1.85</td><td>0.29×</td><td></td><td></td></tr><tr><td>Ours</td><td>59.49</td><td>13.73</td><td>1.78</td><td>4.06</td><td>0.63×</td><td></td><td></td></tr><tr><td rowspan="3">Qwen3.6-Plus</td><td>Reactive</td><td>75.78</td><td>21.13</td><td>1.34</td><td>4.87</td><td>1.00×</td><td></td><td></td></tr><tr><td>Batched</td><td>64.24</td><td>6.67</td><td>1.17</td><td>2.28</td><td>0.47×</td><td></td><td></td></tr><tr><td>Ours</td><td>76.04</td><td>12.31</td><td>1.78</td><td>3.84</td><td>0.79×</td><td></td><td></td></tr></table>

Table 9: Executor results by benchmark and backbone. Reward is on [0, 100]; token totals are in millions (M). Qwen3.5 times are summed task durations in seconds. Ratios are relative to Reactive within each backbone. EB-Habitat totals sum the six categories in Table 10. Dashes denote unavailable values. DS-V4.1-Flash denotes DeepSeek-V4.1-Flash.

<table><tr><td rowspan="2">Category</td><td colspan="2">Reactive</td><td colspan="3">Batched</td><td colspan="3">Ours</td></tr><tr><td>Reward</td><td>Tokens (M)</td><td>Reward</td><td>Tokens (M)</td><td>Ratio</td><td>Reward</td><td>Tokens (M)</td><td>Ratio</td></tr><tr><td>Qwen3.6-Flash</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>87.25</td><td>0.56</td><td>79.75</td><td>0.14</td><td>0.25×</td><td>89.75</td><td>0.27</td><td>0.49×</td></tr><tr><td>Common Sense</td><td>44.00</td><td>1.17</td><td>31.47</td><td>0.47</td><td>0.40×</td><td>45.00</td><td>0.69</td><td>0.59×</td></tr><tr><td>Complex Instr.</td><td>54.00</td><td>1.06</td><td>52.07</td><td>0.30</td><td>0.28×</td><td>56.00</td><td>0.61</td><td>0.58×</td></tr><tr><td>Long Horizon</td><td>33.42</td><td>1.36</td><td>19.71</td><td>0.32</td><td>0.24×</td><td>34.50</td><td>0.74</td><td>0.55×</td></tr><tr><td>Spatial Rel.</td><td>68.00</td><td>1.23</td><td>65.83</td><td>0.33</td><td>0.26×</td><td>70.67</td><td>0.91</td><td>0.73×</td></tr><tr><td>Visual App.</td><td>61.92</td><td>1.05</td><td>44.48</td><td>0.30</td><td>0.29×</td><td>61.00</td><td>0.84</td><td>0.80×</td></tr><tr><td>Average</td><td>58.10</td><td>1.07</td><td>48.88</td><td>0.31</td><td>0.29×</td><td>59.49</td><td>0.68</td><td>0.63×</td></tr><tr><td>Qwen3.6-Plus</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>96.00</td><td>0.31</td><td>88.75</td><td>0.15</td><td>0.47×</td><td>95.50</td><td>0.25</td><td>0.81×</td></tr><tr><td>Common Sense</td><td>72.50</td><td>0.87</td><td>54.00</td><td>0.45</td><td>0.51×</td><td>71.50</td><td>0.63</td><td>0.73×</td></tr><tr><td>Complex Instr.</td><td>84.00</td><td>0.76</td><td>73.50</td><td>0.38</td><td>0.50×</td><td>87.50</td><td>0.64</td><td>0.84×</td></tr><tr><td>Long Horizon</td><td>58.33</td><td>0.98</td><td>42.17</td><td>0.36</td><td>0.36×</td><td>59.00</td><td>0.64</td><td>0.66×</td></tr><tr><td>Spatial Rel.</td><td>66.17</td><td>1.22</td><td>60.75</td><td>0.69</td><td>0.56×</td><td>64.25</td><td>1.08</td><td>0.89×</td></tr><tr><td>Visual App.</td><td>77.67</td><td>0.73</td><td>66.25</td><td>0.27</td><td>0.37×</td><td>78.50</td><td>0.58</td><td>0.80×</td></tr><tr><td>Average</td><td>75.78</td><td>0.81</td><td>64.24</td><td>0.38</td><td>0.47×</td><td>76.04</td><td>0.64</td><td>0.79×</td></tr><tr><td rowspan="2">Controller</td><td rowspan="2">Executor</td><td rowspan="2">Reward</td><td rowspan="2">Input (M)</td><td rowspan="2">Output (M)</td><td colspan="2">Weighted Tokens</td><td colspan="2">Time</td></tr><tr><td>(M)</td><td>Ratio</td><td>(s)</td><td>Ratio</td></tr><tr><td>ALFWorld Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive Ours</td><td>81.34 81.34</td><td>7.89 5.45</td><td>3.48 2.22</td><td>4.46 2.90</td><td>1.00× 0.65×</td><td>40540.94 25 778.94</td><td>1.00× 0.64×</td></tr><tr><td>MPO</td><td>Reactive Ours</td><td>88.06 89.55</td><td>5.11 6.18</td><td>1.54 1.02</td><td>2.18 1.79</td><td>1.00× 0.82×</td><td>14173.30 9647.12</td><td>1.00× 0.68×</td></tr><tr><td>Reflexion</td><td>Reactive Ours</td><td>90.30 88.81</td><td>13.59 12.78</td><td>3.97 2.35</td><td>5.67 3.95</td><td>1.00× 0.70×</td><td>36263.58 22008.90</td><td>1.00× 0.61×</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive Ours</td><td>91.04 91.04</td><td>7.58 4.88</td><td>1.51 0.90</td><td>2.46 1.51</td><td>1.00× 0.62×</td><td>22 534.70 13 605.20</td><td>1.00× 0.60×</td></tr><tr><td>MPO</td><td>Reactive Ours</td><td>90.30 89.55</td><td>6.03 4.89</td><td>2.38 1.37</td><td>3.13 1.98</td><td>1.00× 0.63×</td><td>34745.77 20 177.39</td><td>1.00× 0.58×</td></tr><tr><td>Reflexion</td><td>Reactive Ours</td><td>98.51 97.01</td><td>7.99 9.36</td><td>2.75 2.11</td><td>3.75 3.28</td><td>1.00× 0.87×</td><td>36 062.00 28 296.39</td><td>1.00× 0.78×</td></tr><tr><td>DeepSeek-V3.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive Ours</td><td>88.81 87.31</td><td>4.76 3.13</td><td>0.07 0.19</td><td>3.25 2.28</td><td>1.00× 0.70×</td><td></td><td></td></tr><tr><td>MPO</td><td>Reactive</td><td>94.78 95.52</td><td>3.50 2.35</td><td>0.06 0.15</td><td>2.40 1.72</td><td>1.00×</td><td></td><td></td></tr><tr><td>Reflexion</td><td>Ours Reactive Ours</td><td>98.51 99.25</td><td>6.04 3.59</td><td>0.10 0.27</td><td>4.13 2.66</td><td>0.72× 1.00× 0.64×</td><td></td><td></td></tr><tr><td>ScienceWorld GPT-5.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive Ours</td><td>94.18 95.35</td><td>18.91 9.29</td><td>0.68 0.51</td><td>3.83 2.06</td><td>1.00× 0.54×</td><td></td><td></td></tr><tr><td>MPO</td><td>Reactive Ours</td><td>93.20 94.44</td><td>12.45 6.02</td><td>0.36 0.33</td><td>2.44 1.34</td><td>1.00× 0.55×</td><td></td><td></td></tr><tr><td>Reflexion</td><td>Reactive Ours</td><td>97.18 96.75</td><td>20.91 14.31</td><td>0.63 0.73</td><td>4.11 3.12</td><td>1.00× 0.76×</td><td></td><td></td></tr><tr><td>DS-V4.1-Flash</td><td></td><td>78.20</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive Ours</td><td>78.76</td><td>31.34 21.11</td><td>3.30 0.81</td><td>11.14 6.09</td><td>1.00× 0.55×</td><td></td><td></td></tr><tr><td>MPO</td><td>Reactive Ours</td><td>80.59 80.26</td><td>18.82 11.73</td><td>0.46 0.68</td><td>5.17 3.61</td><td>1.00× 0.70×</td><td></td><td></td></tr><tr><td>Reflexion</td><td>Reactive Ours</td><td>87.08 85.39</td><td>84.12 61.16</td><td>2.03 2.35</td><td>23.06 17.64</td><td>1.00× 0.76×</td><td></td><td></td></tr><tr><td>DeepSeek-V3.2</td><td></td><td>81.41</td><td>14.76</td><td></td><td>10.12</td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive Ours</td><td>82.47 86.12</td><td>13.47</td><td>0.28 0.52</td><td>9.50</td><td>1.00× 0.94×</td><td></td><td></td></tr><tr><td>MPO</td><td>Reactive Ours</td><td>87.19</td><td>16.22 11.33</td><td>0.21 0.44</td><td>11.02 7.99</td><td>1.00× 0.73×</td><td></td><td>一</td></tr><tr><td>Reflexion</td><td>Reactive Ours</td><td>88.31 89.11</td><td>34.53 26.98</td><td>0.51 0.96</td><td>23.53 18.95</td><td>1.00× 0.81×</td><td></td><td></td></tr><tr><td>EB-Habitat Qwen3.6-Flash</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReflAct</td><td>Reactive</td><td>58.49 59.10</td><td>29.03 16.48</td><td>1.29 1.81</td><td>6.13 4.56</td><td>1.00× 0.74×</td><td></td><td></td></tr><tr><td></td><td>Ours</td><td>68.82</td><td>93.43</td><td>5.30</td><td>20.87</td><td>1.00×</td><td></td><td></td></tr><tr><td>Reflexion</td><td>Reactive Ours</td><td>68.88</td><td>46.75</td><td>6.35</td><td>14.14</td><td>0.68×</td><td></td><td></td></tr></table>

Table 10: Detailed EB-Habitat results by task category and vision-language backbone. Rewards are rescaled to [0, 100]; tokens are price-weighted totals in millions (M). Ratios are relative to Reactive within each category. Average rows report category means; their token ratios divide each mean by the corresponding Reactive mean.

Table 11: Controller composition across three benchmarks with Reactive and EOCV (Ours) executors. Reward is on [0, 100]; token totals are in millions (M). Qwen3.5 times are summed task durations in seconds. Ratios are relative to Reactive within each backbone and controller. Dashes denote unavailable values. DS-V4.1-Flash denotes DeepSeek-V4.1-Flash.

<table><tr><td></td><td colspan="2">Batched Executor</td><td colspan="3">Ours</td></tr><tr><td></td><td>Step Action</td><td>Observation</td><td>Action</td><td>Expectation</td><td>Observation</td></tr><tr><td>t</td><td>move(cup, oven)</td><td>That can&#x27;t be moved there, because the oven isn&#x27;t open.</td><td>move(cup, oven)</td><td>You move the glass cup to the oven.</td><td>That can&#x27;t be moved there, because the oven</td></tr><tr><td></td><td>t+1 activate(oven)</td><td>The oven is now activated.</td><td>close(oven)</td><td>The oven is now closed.</td><td>isn&#x27;t open. Terminate &amp; Replan</td></tr><tr><td></td><td>t+2 wait()</td><td>You decide to wait for 10 activate(oven) iterations.</td><td></td><td>The oven is now activated.</td><td></td></tr><tr><td></td><td>t+3 look_at(oven)</td><td>An oven, which is turned wait() on. The oven door is closed.</td><td></td><td>You decide to wait for 10 — iterations.</td><td></td></tr></table>

Table 12: ScienceWorld case with DeepSeek-V4.1-Flash, expanding Table 3. Slots preserve the order within each independently generated batch. Red actions continue after the failed move; gray actions are proposed but not executed. Dashes indicate no observation.
<table><tr><td rowspan="2"></td><td colspan="2">Batched Executor</td><td colspan="3">Ours</td></tr><tr><td>Step Action</td><td>Observation</td><td>Action</td><td>Expectation</td><td>Observation</td></tr><tr><td>t</td><td>activate(sink)</td><td>The sink is now activated. activate(sink)</td><td></td><td></td><td>The sink is now activated. The sink is now activated</td></tr><tr><td></td><td>t+1 dunk(glass cup, sink)</td><td>The sink does not contain dunk(glass cup, any liquids to dunk into. sink)</td><td></td><td>You dunk the glass cup in The sink does not contain the sink.</td><td>any liquids to dunk into.</td></tr><tr><td></td><td>t+2 move(sugar, glass</td><td>You move the sugar to the move(sugar, glass</td><td></td><td>You move the sugar to the Terminate &amp; Replan</td><td></td></tr><tr><td></td><td>cup) t+3 mix(glass cup)</td><td>glass cup. There is only one thing</td><td>cup) mix(glass cup)</td><td>glass cup. Sugar and water mix to</td><td></td></tr><tr><td></td><td>t+4 focus_on(sugar water)</td><td>(sugar) No known action matches focus_on(sugar that input.</td><td>water)</td><td>produce sugar water. You focus on the sugar water.</td><td></td></tr></table>

Table 13: ScienceWorld case with GPT-5.5. Slots preserve the order within each independently generated batch. Red actions continue after the failed move; gray actions are proposed but not executed. Dashes indicate no observation.
<table><tr><td></td><td colspan="2">ReAct 5 executor calls; 2,554 weighted tokens</td><td colspan="3">Ours 2 executor calls; 1,432 weighted tokens</td></tr><tr><td>Step Action</td><td></td><td>Observation</td><td>Action</td><td>Expectation</td><td>Observation</td></tr><tr><td></td><td></td><td></td><td></td><td>EOCV executor call 1</td><td></td></tr><tr><td>t</td><td>goto(drawer 4)</td><td>You arrive at drawer 4. The drawer 4 is closed.</td><td>goto(drawer 4)</td><td>Skip verification</td><td>You arrive at drawer 4. The drawer 4 is closed.</td></tr><tr><td></td><td>t+1 open(drawer 4)</td><td>You open the drawer 4. The drawer 4 is open. In</td><td>open(drawer 4)</td><td>Skip verification</td><td>You open the drawer 4. The drawer 4 is open. In</td></tr><tr><td></td><td>t+2 goto(drawer 5)</td><td>it, you see nothing. You arrive at drawer 5.</td><td>goto(drawer 5)</td><td>Skip verification</td><td>it, you see nothing. You arrive at drawer 5.</td></tr><tr><td></td><td>t+3 open(drawer 5)</td><td>The drawer 5 is closed. You open the drawer 5. The drawer 5 is open. In</td><td>open(drawer 5)</td><td>Skip verification</td><td>The drawer 5 is closed. You open the drawer 5. The drawer 5 is open. In</td></tr><tr><td></td><td></td><td>it, you see a watch 1.</td><td></td><td></td><td>it, you see a watch 1.</td></tr><tr><td></td><td></td><td>5)</td><td></td><td>EOCV executor call 2: first action shown</td><td></td></tr><tr><td>5)</td><td>t+4 take(watch 1, drawer</td><td>You pick up the watch 1 from the drawer 5.</td><td></td><td>take(watch 1, drawer You pick up the watch 1</td><td>You pick up the watch 1</td></tr></table>

Table 14: ALFWorld case with Qwen3.5-4B. Token totals cover five complete ReAct calls and two complete EOCV calls, including EOCV actions omitted from the table.

## D AdaPlanner Implementation Details

We adapt the authors’ ALFWorld implementation of AdaPlanner (Sun et al., 2023) to modern LLMs. Complete prompts and output contracts appear in Appendix E.2.

Prompts and decoding. We migrate the completion-based prompts to chat templates. For a fair comparison, AdaPlanner uses the same in-context learning (ICL) example as the other executors, rewritten in AdaPlanner’s format.

<table><tr><td colspan="4"></td><td colspan="4">Ours</td></tr><tr><td></td><td>Step Action</td><td></td><td>Observation Annotation</td><td>Action</td><td>Expectation</td><td>Observation Annotation</td><td></td></tr><tr><td></td><td>t pick up the pear</td><td>2</td><td>nothing is picked up</td><td>pick up the pear</td><td>The robot picks up a pear from the left</td><td>2</td><td>nothing is picked up</td></tr><tr><td></td><td>t+1 navigate to the sofa</td><td></td><td>agent moves to the sofa</td><td>navigate to the sofa Skip verification</td><td></td><td>Terminate &amp; Replan</td><td></td></tr><tr><td></td><td>t+2 place at the sofa</td><td></td><td>nothing can be placed</td><td>place at the sofa</td><td>The pear is placed on the sofa.</td><td></td><td></td></tr></table>

Table 15: EB-Habitat case with Qwen3.6-Flash, expanding Table 4. The batched executor continues executing the red actions after the visual observation contradicts the plan. EOCV detects the mismatch and withholds the gray actions. Annotations describe the images for the reader and are not available to the agent. Dashes indicate that an observation or annotation is not applicable.
<table><tr><td></td><td colspan="4">ReAct 3 executor calls; 2,476 weighted tokens</td><td colspan="4">Ours 1 executor call; 968 weighted tokens</td></tr><tr><td></td><td>Step Action</td><td></td><td>Observation Annotation</td><td>Action</td><td>Expectation</td><td></td><td></td><td>Observation Annotation</td></tr><tr><td></td><td>t place at table 2</td><td>百</td><td></td><td>block is placed place at table 2</td><td></td><td>The block is placed on table 2.</td><td></td><td>block is placed</td></tr><tr><td></td><td>t+1 pick up the hammer</td><td></td><td>hammer is picked up</td><td></td><td>pick up the hammer The robot is holding the hammer.</td><td></td><td></td><td>hammer is picked up</td></tr><tr><td></td><td>t+2 navigate to table 1</td><td></td><td>agent reaches table 1</td><td></td><td>navigate to table 1 Skip verification</td><td></td><td></td><td>agent reaches table 1</td></tr></table>

Table 16: EB-Habitat efficiency case study with Qwen3.6-Flash. Token totals cover three complete ReAct calls and one complete EOCV calls.

Evaluation settings. We evaluate the same 134 ALFWorld tasks on Qwen3.5-9B with reasoning enabled. Each task permits at most 6 plan executions and 50 cumulative environment actions.

Token accounting. Totals include all model calls, including syntax checks, refinement, resume selection, and budget failures. Reasoning is counted once as part of output tokens.

## E Prompts

This appendix presents the prompts used in our experiments for reproducibility.

## E.1 ALFWorld

## E.1.1 Instructions

Instructions for ALFWorld EOCV, Reactive, Batched   
Interact with a household to solve a task. Imagine you are an intelligent agent in a household   
environment and your target is to perform actions to complete the task goal.   
At the beginning of your interactions, you will be given the detailed description of the current   
environment and your goal to accomplish.   
The interaction is turn-based. Each turn is given in the following format:   
\`\`\`python   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action>   
# Observation: the feedback of your action   
The available actions are:   
inventory()   
Description: check your current inventory.   
Example:   
inventory() -> "You are carrying: a cup 4."   
goto(receptacle)   
Description: move to a receptacle and observe what's in/on it.   
Parameters:   
receptacle (str): the receptacle to go to.   
Example:   
goto("sinkbasin 1") -> "You arrive at sinkbasin 1, on the sinkbasin 1, you see <   
receptacles>."   
open(receptacle)   
Description: open a receptacle.   
Parameters:   
receptacle (str): the receptacle to open.   
Example:   
open("drawer 2") -> "You open the drawer 2. The drawer2 is open. In it, you see a   
creditcard 1, a dishsponge 1, and a spatula 2."   
close(receptacle)   
Description: close a receptacle.   
Parameters:   
receptacle (str): the receptacle to close.   
Example:   
close("drawer 2") -> "You close the drawer 2."   
take(object, receptacle)   
Description: take an object from a nearby receptacle.   
Parameters:   
object (str): the object to take.   
receptacle (str): the receptacle to take the object from.   
Example:   
take("spatula 1", "drawer 2") -> "You pick up the spatula 1 from the drawer 2."   
Hint:   
You can only take one thing with you at once.   
You must be near the receptacle to take something from it. That is, you should first "   
goto('receptacle A')" before "take(object, 'receptacle A')".   
If the receptacle is closed, you need to open it first.

move(object, receptacle)   
Description: place an object in or on a receptacle.   
Parameters:   
object (str): the object to move.   
receptacle (str): the receptacle to move the object to.   
Example:   
move("spatula 1", "drawer 2") -> "You move the spatula 1 to the drawer 2."   
Hint:   
You must be near the receptacle to move something to it.   
If the receptacle is closed, you need to open it first.   
turn\_on(object)   
Description: turn on an object.   
Parameters:   
object (str): the object to turn on.   
Example:   
turn\_on('desklamp 1') -> "You turn on the desklamp 1."   
heat(object, receptacle)   
Description: heat an object using a receptacle.   
Parameters:   
object (str): the object to heat.   
receptacle (str): the receptacle to use for heating.   
Example:   
heat("tomato 1", "microwave 1") -> "You heat the tomato 1 using the microwave 1."   
clean(object, receptacle)   
Description: clean an object using a receptacle.   
Parameters:   
object (str): the object to clean.   
receptacle (str): the receptacle to use for cleaning.   
Example:   
clean("cup 1", "sinkbasin 1") -> "You clean the cup 1 using the sinkbasin 1."   
cool(object, receptacle)   
Description: cool an object using a receptacle.   
Parameters:   
object (str): the object to cool.   
receptacle (str): the receptacle to use for cooling.   
Example:   
cool("cup 1", "fridge 1") -> "You cool the cup 1 using the fridge 1."   
slice(object, tool)   
Description: slice an object using a sharp object.   
Parameters:   
object (str): the object to slice.   
tool (str): the sharp object to use for slicing.   
Example:   
slice("apple 1", "knife 1") -> "You slice the apple 1 using the knife 1."   
Reminders:   
1. There are several locations, each location can contain several objects. You can only take one   
object with you at once.   
2. Actions are used to interact with the objects. The actions must be chosen from the given   
available actions. Any actions except provided available actions will be regarded as illegal.   
Furthermore, you can only interact with objects in the current location or the object you take.   
3. The task uses strict matching. This means you must find exactly what the task requires, but   
not use a similar thing to replace it. For instance, if the task want a "knife", you cannot use a   
"butterknife" to replace it. Instead, you must find the object whose name is exactly "knife".

```markdown
4. You can only take and keep one object in your inventory, so you have to put down the object
you are holding before taking another one.
=EOCV=
Now, given the interaction history, you should give out a future action sequence towards the task
goal.
Your response must strictly follow this format:
```python
# Thought: your thoughts about the current condition and plan for your future actions to complete
the current subgoal.
<your action 1>
# Observation: <expected observation of your action>
# Thought: your thoughts about the current condition and plan for your future actions to complete
the current subgoal.
<your action 2>
# Observation: <expected observation of your action>
# Thought: your thoughts about the current condition and plan for your future actions to complete
the current subgoal.
<your action 3>
# Observation: <expected observation of your action>
<... repeat several times ...>
In order to verify if your purpose matches the actual environment observation, you are required
to provide the expected observation of each action you take in the format above. For actions
whose observations are unpredictable(e.g. "look", "examine", "goto", "open"). Simply write "<
unpredictable observation>" as their expected observation.
In order to reduce cost, you are encouraged to generate a sequence of future actions. However,
generating future actions without intermediate observations can make you deviate from the actual
environment state. Therefore, stop extending the sequence when the environment state becomes
unpredictable.
=======Reactive=======
Now, given the interaction history, you should give out your next action towards the task goal.
Your response must strictly follow this format:
```python
# Thought: your thoughts about the current condition and plan for your future actions to complete
the current subgoal.
<your action>
=Batched
Now, given the interaction history, you should give out a future action sequence towards the task
goal.
Your response must strictly follow this format:
```python
# Thought: your thoughts about the current condition and plan for your future actions to complete
the current subgoal.
<your action 1>
# Thought: your thoughts about the current condition and plan for your future actions to complete
the current subgoal.
<your action 2>
```

# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 3>   
<... repeat several times ...>   
Here is an example:   
{icl\_example}   
=======Reactive=======   
Now, here is the real task, think carefully about the task, then give out your next action(give   
only one action) in the same format as the example above.   
=====EOCV/Batched=====   
Now, here is the real task, think carefully about the task, then give out a future action   
sequence in the same format as the example above.   
You are required to plan at most {max\_planning\_steps} future actions per generation step in this   
task. If you are not sure about the environment state, stop extending the action sequence.   
{interact\_history}

## E.1.2 ICL Examples

ICL Example for ALFWorld EOCV

\`\`\`python   
# Task Description: -= Welcome to TextWorld, ALFRED! =-   
You are in the middle of a room. Looking quickly around you, you see a cabinet 11, a cabinet 10,   
a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a   
cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a   
drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a   
stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\n\nYour task   
is to: put a hot bread in garbagecan.   
# Thought: I need to find bread, which is likely to be found in a fridge.   
goto("fridge 1")   
# Observation: You arrive at fridge 1. The fridge 1 is closed.   
# Thought: I should open the fridge to see if there is any bread inside.   
open("fridge 1")   
# Observation: You open the fridge 1. The fridge 1 is open. In it, you see a cup 2, a egg 3, a   
egg 2, a egg 1, a lettuce 3, a mug 3, and a tomato 2.   
# Thought: There is no bread in the fridge. I should look around the countertops and see if I can   
find bread.   
goto("countertop 1")   
# Observation: You arrive at countertop 1. On the countertop 1, you see a bread 1, a dishsponge   
1, a knife 1, a mug 1, a peppershaker 1, a potato 1, and a spoon 1.   
# Thought: I see bread 1 on countertop 1. I will pick up bread 1 and heat it using the microwave.   
take("bread 1", "countertop 1")   
# Observation: You pick up the bread 1 from the countertop 1.   
# Thought: I've picked up bread 1. Now I need to go to the microwave to heat it.   
goto("microwave 1")   
# Observation: You arrive at microwave 1. The microwave 1 is closed.   
# Thought: I need to open the microwave first before heating the bread.   
open("microwave 1")   
# Observation: You open the microwave 1. The microwave 1 is open. In it, you see nothing.   
Your response can be:

\`\`\`python   
# Thought: Now I should heat the bread using the microwave.   
heat("bread 1", "microwave 1")   
# Observation: You heat the bread 1 using the microwave 1.   
# Thought: The bread is now hot. I need to put the hot bread in the garbagecan to complete the   
task.   
goto("garbagecan 1")   
# Observation: <unpredictable observation>

## ICL Example for ALFWorld Reactive

\`\`\`python   
# Task Description: -= Welcome to TextWorld, ALFRED! =-   
You are in the middle of a room. Looking quickly around you, you see a cabinet 11, a cabinet 10,   
a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a   
cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a   
drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a   
stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\n\nYour task   
is to: put a hot bread in garbagecan.   
# Thought: I need to find bread, which is likely to be found in a fridge.   
goto("fridge 1")   
# Observation: You arrive at fridge 1. The fridge 1 is closed.   
# Thought: I should open the fridge to see if there is any bread inside.   
open("fridge 1")   
# Observation: You open the fridge 1. The fridge 1 is open. In it, you see a cup 2, a egg 3, a   
egg 2, a egg 1, a lettuce 3, a mug 3, and a tomato 2.   
# Thought: There is no bread in the fridge. I should look around the countertops and see if I can   
find bread.   
goto("countertop 1")   
# Observation: You arrive at countertop 1. On the countertop 1, you see a bread 1, a dishsponge   
1, a knife 1, a mug 1, a peppershaker 1, a potato 1, and a spoon 1.   
# Thought: I see bread 1 on countertop 1. I will pick up bread 1 and heat it using the microwave.   
take("bread 1", "countertop 1")   
# Observation: You pick up the bread 1 from the countertop 1.   
# Thought: I've picked up bread 1. Now I need to go to the microwave to heat it.   
goto("microwave 1")   
# Observation: You arrive at microwave 1. The microwave 1 is closed.   
# Thought: I need to open the microwave first before heating the bread.   
open("microwave 1")   
# Observation: You open the microwave 1. The microwave 1 is open. In it, you see nothing.   
# Thought: Now I should heat the bread using the microwave.   
heat("bread 1", "microwave 1")   
# Observation: You heat the bread 1 using the microwave 1.   
Your response can be:   
\`\`\`python   
# Thought: The bread is now hot. I need to put the hot bread in the garbagecan to complete the   
task.   
goto("garbagecan 1")

ICL Example for ALFWorld Batched   
\`\`\`python   
# Task Description: -= Welcome to TextWorld, ALFRED! =-   
You are in the middle of a room. Looking quickly around you, you see a cabinet 11, a cabinet 10,   
a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a   
cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a   
drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a   
stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\n\nYour task   
is to: put a hot bread in garbagecan.   
# Thought: I need to find bread, which is likely to be found in a fridge.   
goto("fridge 1")   
# Observation: You arrive at fridge 1. The fridge 1 is closed.   
# Thought: I should open the fridge to see if there is any bread inside.   
open("fridge 1")   
# Observation: You open the fridge 1. The fridge 1 is open. In it, you see a cup 2, a egg 3, a   
egg 2, a egg 1, a lettuce 3, a mug 3, and a tomato 2.   
# Thought: There is no bread in the fridge. I should look around the countertops and see if I can   
find bread.   
goto("countertop 1")   
# Observation: You arrive at countertop 1. On the countertop 1, you see a bread 1, a dishsponge   
1, a knife 1, a mug 1, a peppershaker 1, a potato 1, and a spoon 1.   
# Thought: I see bread 1 on countertop 1. I will pick up bread 1 and heat it using the microwave.   
take("bread 1", "countertop 1")   
# Observation: You pick up the bread 1 from the countertop 1.   
# Thought: I've picked up bread 1. Now I need to go to the microwave to heat it.   
goto("microwave 1")   
# Observation: You arrive at microwave 1. The microwave 1 is closed.   
# Thought: I need to open the microwave first before heating the bread.   
open("microwave 1")   
# Observation: You open the microwave 1. The microwave 1 is open. In it, you see nothing.   
Your response can be:   
\`\`\`python   
# Thought: Now I should heat the bread using the microwave.   
heat("bread 1", "microwave 1")   
# Thought: The bread is now hot. I need to put the hot bread in the garbagecan to complete the   
task.   
goto("garbagecan 1")

## E.2 AdaPlanner on ALFWorld

These are the prompts for the AdaPlanner implementation in Appendix D.

## E.2.1 Instruction and ICL example

The Instruction serves as system prompt, and are used by the following syntax checking, plan refinement, resume selection and resume selection progress.

Instruction for Adaplanner   
You are a household agent. Here is some Python code defining a household environment:   
Use literal\_eval to convert the answer from ask() to a list.   
literal\_eval is provided by the runtime.   
In the environment, you can ask questions to an assistant by ask():

```python
for example: You have a list of receptacles,
and you want to sort them by the likelihood of a soapbar appearing in them.
You can do this by asking the assistant:
receptacles = ['countertop 1', 'garbagecan 1', 'sinkbasin 2', 'sinkbasin 1', 'toilet 1',
toiletpaperhanger 1', 'towelholder 1']
answer = ask(f'Sort the list of receptacles, starting from the one a soapbar is most likely to
appear: {receptacles}. You should return a Python list.')
# answer = ['sinkbasin 1', 'sinkbasin 2', 'countertop 1', 'towelholder 1', 'toiletpaperhanger 1',
'garbagecan 1', 'toilet 1']
Agent class represents the state of the agent, including its location,
what it's holding as well as the actions it can take.
class Agent:
def __init__(self, receptacles):
self.location = None
self.holding = None
self.receptacles = receptacles
# Here are the admissible actions the agent can take:
# Go to a receptacle and update the agent's location.
# For example, 'On the countertop 1, you see a candle 1, a cloth 2, and a soapbar 1.' = goto
('countertop 1')
# For example, 'On the sidetable 2, you see nothing.' = goto('sidetable 2')
def goto(self, receptacle):
# Take an object from a receptacle if the agent is not holding anything.
# For example, 'You pick up the soapbar 1 from the towelholder 1.' = take('soapbar 1',
towelholder 1')
def take(self, object, receptacle):
# Put an object in or on a receptacle if the agent is holding it.
# For example, 'You put the soapbar 1 in/on the cabinet 1.' = put('soapbar 1', 'cabinet 1')
def put(self, object, receptacle):
...
# Open a receptacle and observe its contents.
# For example, 'You open the cabinet 1. The cabinet 1 is open. In it, you see a cloth 1.' =
open_receptacle('cabinet 1')
def open_receptacle(self, receptacle):
# Clean an object with a receptacle.
# For example, 'You clean the soapbar 1 using the sinkbasin 1.' = clean('soapbar 1', '
sinkbasin 1')
def clean(self, object, receptacle):
# Heat an object with a receptacle.
# For example, 'You heat the tomato 1 using the microwave 1.' = heat('tomato 1', 'microwave
1')
def heat(self, object, receptacle):
# Cool an object with a receptacle.
# For example, 'You cool the pan 2 using the fridge 1.' = cool('pan 2', 'fridge 1')
def cool(self, object, receptacle):
# Turn on an object.
# For example, 'You turn on the desklamp 1.' = turn_on('desklamp 1')
def turn_on(self, object):
...
```

```python
# Report agent's current state, including its location, what it's holding, and last action
and observation.
# This function should only be used in assertion.
def report(self):
```

```elixir
ICL example and task description for Adaplanner
Here is an example of a solution to the task:
# -= Welcome to TextWorld, ALFRED! =-
#
# You are in the middle of a room. Looking quickly around you, you see a cabinet 11, a cabinet
10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a
cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a
drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a
stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
#
# Your task is to: put a hot bread in garbagecan.
# Here is a solution:
def solution(agent, start_from=1):
# General plan: find bread, take it, heat it in the microwave,
# and put the hot bread in the garbagecan.
if start_from <= 1:
print("[Step 1] Find bread")
# Bread may be in the fridge; if it is absent, check the countertop.
for receptacle in ["fridge 1", "countertop 1"]:
observation = agent.goto(receptacle)
assert agent.location == receptacle, f"Error in [Step 1]: {agent.report()}"
if "closed" in observation:
observation = agent.open_receptacle(receptacle)
if "bread " in observation:
break
assert "bread " in observation, f"Error in [Step 1]: Bread was not found. {agent.report()
}"
if start_from <= 2:
print("[Step 2] Take the bread")
# Use the observed identifier to pick up the bread that was found.
answer = ask(f"From this observation, what is the identifier of the bread? {observation}
Return only the number.")
bread = f"bread {answer}"
observation = agent.take(bread, receptacle)
assert agent.holding == bread, f"Error in [Step 2]: {agent.report()}"
if start_from <= 3:
print("[Step 3] Heat the bread")
# Go to the microwave and open it if closed, then heat the held bread.
observation = agent.goto("microwave 1")
assert agent.location == "microwave 1", f"Error in [Step 3]: {agent.report()}"
if "closed" in observation:
observation = agent.open_receptacle("microwave 1")
assert "microwave 1 is open" in observation, f"Error in [Step 3]: {agent.report()}"
observation = agent.heat(bread, "microwave 1")
assert f"You heat the {bread} using the microwave 1." in observation, f"Error in [Step 3]:
{agent.report()}"
if start_from <= 4:
print("[Step 4] Put the hot bread in the garbagecan")
# The goal requires placing the heated bread in the garbagecan.
observation = agent.goto("garbagecan 1")
assert agent.location == "garbagecan 1", f"Error in [Step 4]: {agent.report()}"
observation = agent.put(bread, "garbagecan 1")
assert f"You put the {bread} in/on the garbagecan 1." in observation, f"Error in [Step 4]:
{agent.report()}"
```

Here is the actual task:   
receptacles = <RECEPTACLES>   
<TASK\_DESCRIPTION>   
Write only the complete Python solution function for this task

## E.2.2 Syntax Checking

Check the syntax of a Python solution function.   
Return {"decision":"No"} if there are no syntax errors.   
If there are syntax errors, return   
{"decision":"Yes","code":["def solution(agent, start\_from=1):"," ..."]}.   
Each code element is exactly one Python source line;   
preserve leading indentation spaces. Check Python syntax only.   
The runtime provides agent, ask, literal\_eval and common Python builtins;   
missing imports or runtime names are not syntax errors.   
For No, omit code and do not copy the function.   
Keep the algorithm and assertions unchanged. No other prose.   
Here is the python code:   
<SOLUTION\_FUNCTION>   
Question 1: Are there any syntax errors present in the code? Answer Yes/No.   
Answer Question 1 as a JSON object with decision equal to "Yes" or "No".   
Question 2 applies ONLY if there are syntax errors:   
output the entire corrected function as an array of source lines in code,   
preserving indentation.   
Otherwise output only {"decision":"No"}.

## E.2.3 Plan Refinement

<PREVIOUS\_SOLUTION>   
You have generated code of solution() to solve the task. However, you executed the solution()   
function and get an error message:   
<EXECUTION\_ERROR>   
[Current agent state]   
<CURRENT\_STATE>   
[Saved local variables]   
<SAVED\_LOCALS>   
[Actual interactions]   
<ACTION\_OBSERVATION\_TRAJECTORY>   
Referring to the successful case and the error message,   
revise the plan and return only the complete corrected Python function,   
preserving newlines and indentation.   
The environment has NOT been reset.

## E.2.4 Resume Selection

Previously, you generated some code defining a solution function as in [Previous solution].   
The previous code is executed and outputs some error.   
Now you just revised the code as in [Revised solution].   
Determine from which step these two version differs.   
Return only a JSON object with the positive integer field start\_from.   
Choose a safe numbered step for resuming a revised plan in the current environment.   
Return JSON {"start\_from": N} with a positive integer N.

Resume no later than the failed step, taking saved local variables into account.   
If unsure, return {"start\_from":1}.   
[Previous solution]   
<PREVIOUS\_SOLUTION>   
[Revised solution]   
<REVISED\_SOLUTION>   
[Execution error]   
<EXECUTION\_ERROR>   
[Saved locals]   
<SAVED\_LOCALS>

## E.2.5 In-Plan Reasoning

Note that this progress does not adapt the general instruction as system prompt.

In-plan reasoning for AdaPlanner   
You are the household agent's in-plan reasoning assistant.   
Answer the question using only the supplied observations and receptacles.   
Return a JSON object with exactly one field, answer.   
Interpret output-format requests in the question as applying to the answer value;   
the runtime converts arrays to Python list literals and IDs to strings for ask().   
No Markdown fences or additional explanation.   
<IN\_PLAN\_QUESTION>

## E.3 ScienceWorld

## E.3.1 Instructions

Instruction for ScienceWorld EOCV, Reactive, Batched   
Interact with a household to solve a task. Imagine you are an intelligent agent in a household   
environment and your target is to perform actions to complete the task goal.   
At the beginning of your interactions, you will be given the detailed description of the current   
environment and your goal to accomplish.   
The interaction is turn-based. Each turn is given in the following format:   
\`\`\`python   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action>   
# Observation: the feedback of your action   
The available actions are:   
activate(obj: str)   
Description: Activate a device.   
Parameters:   
obj (str): The name of the device to activate.   
Example:   
activate('sink') -> "The sink is now activated."   
choose(number: int)   
Description: When an ambiguous request occurs, the environment feedback will be "This is an   
ambiguous request. Please use the 'choose(number)' action to select the   
specific action you intended."(followed by multiple possible actions). Use "choose" to select the   
specific action, instead of directly outputting the number.It is prohibited   
to use "choose" at other times.   
Example:   
choose(1) -> [Executes the first action from the ambiguous options]

```yaml
close(obj: str)
Description: Close a container.
Parameters:
obj (str): The name of the container to close.
Example:
close('door to hallway') -> "The door is now closed."
deactivate(obj: str)
Description: Deactivate a device.
Parameters:
obj (str): The name of the device to deactivate.
Example:
deactivate('sink') -> "The sink is now deactivated."
dunk(obj: str, target: str)
Description: transferring liquid from target to the obj.
Parameters:
obj (str): The empty container.
target (str): Another container to dunk into(containing liquid).
Example:
dunk('glass cup', 'sink') -> "You dunk the glass cup in the sink."
focus_on(obj: str)
Description: Signal intent on a task object. This action is used to evaluate your performance,
it should be called only if the task description requires you to focus
something. Focus on wrong/non-existence object can cause the failure of current task, so it is
strongly recommended to use "look_around" to see what's in the environment
before use "focus_on".
Parameters:
obj (str): The task-related object to focus on.
Example:
focus_on('red light bulb') -> "You focus on the red light bulb."
look_around()
Description: Describe the current room.
Example:
look_around() -> "[description of surroundings including furniture and objects]"
look_at(obj: str)
Description: Describe a container's contents.
Special: "look at inventory" to check your inventory.
Parameters:
obj (str): The container to look inside.
Example:
look_at('cupboard') -> "The cupboard contains a plate and a bowl."
mix(obj: str)
Description: Chemically mix a container(move the ingredients to the container before mixing).
Parameters:
obj (str): The container to mix.
Example:
mix('bowl') -> "Blue and yellow paint mix to produce green paint."
move(obj: str, target: str)
Description: Move an object to a container. (Can't be used to move object to another room)
Parameters:
obj (str): The object to move.
target (str): The destination container.
Example:
move('metal pot', 'sink') -> "You move the metal pot to the sink."
```

open(obj: str)   
Description: Open a container(You should use look\_around later to check what's in the   
container).   
Parameters:   
obj (str): The name of the container to open.   
Example:   
open('cupboard') -> "The cupboard is now open."   
pick\_up(obj: str)   
Description: Move an object in the current room to the inventory.   
Parameters:   
obj (str): The object to pick up.   
Example:   
pick\_up('thermometer') -> "You move the thermometer to the inventory."   
pour(obj: str, target: str)   
Description: Pour the liquid in one container into another container.   
Parameters:   
obj (str): The container to pour(Should be moveable).   
target (str): Another container to pour into.   
Example:   
pour('jug', 'flower pot 3') -> "You pour the contents of the jug into the flower pot 3."   
read(obj: str)   
Description: Read a note or book.   
Parameters:   
obj (str): The object to read.   
Example:   
read('recipe') -> "The recipe reads: ...")   
teleport(loc: str)   
Description: Teleport to a specific room or outside.   
Parameters:   
loc (str): The name of the room to teleport to.   
Example:   
teleport('kitchen') -> "You teleport to the kitchen."   
use(obj: str, target: str = None)   
Description: Use a tool(on the target if it exists). Note that this action is for high-level   
actions such as dig a hole or plant a seed. You should use other actions   
first(such as move, activate, pour, etc.) unless there is no other available actions to complete   
the what you want to do.   
Parameters:   
obj (str): The tool to use.   
target (str, optional): The target object to use the tool on.   
Example:   
use('thermometer', 'glass cup') -> "the thermometer measures a temperature of 107 degrees   
celsius"   
wait()   
Description: Do nothing for 10 steps.   
Example:   
wait() -> "You decide to wait for 10 iterations."   
wait1()   
Description: Do nothing for 1 step.   
Example:   
wait1() -> "You decide to wait for 1 iterations."   
Reminders:

1. The environment includes the following rooms: "art studio", "greenhouse", "workshop", "kitchen   
", "bedroom", "living room", "hallway", "foundry", and "outside". Use "teleport" to teleport to a   
specific room.   
2. There will be several objects in each room to interact with. And you will have an inventory to   
keep some objects.   
3. Actions are used to interact with the objects. The actions must be chosen from the given   
available actions. Any actions except provided available actions will be regarded as illegal.   
Furthermore, you can only interact with objects in the current room or your inventory.   
4. The environment automatically shuts down when the task is completed, so whenever you are   
required to perform more actions, the task is not done.   
Now, given the interaction history, you should give out a future action sequence towards the task   
goal.   
Your response must strictly follow this format:   
==EOCV=   
\`python   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 1>   
# Observation: <expected observation of your action>   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 2>   
# Observation: <expected observation of your action>   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 3>   
# Observation: <expected observation of your action>   
<... repeat several times ...>   
  
In order to verify if your purpose matches the actual environment observation, you are required   
to provide the expected observation of each action you take in the format above. For actions   
whose observations are unpredictable(e.g. "look\_around", "look\_at", "choose", "read"), simply   
write "<unpredictable observation>" as their expected observations.   
In order to reduce cost, you are encouraged to generate a sequence of future actions. However,   
generating future actions without intermediate observations can make you deviate from the actual   
environment state. Therefore, stop extending the sequence when the environment state becomes   
unpredictable.   
=Reactive   
\`python   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action>   
=Batched=   
Your response must strictly follow this format:   
\`\`python   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 1>   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 2>   
# Thought: your thoughts about the current condition and plan for your future actions to complete   
the current subgoal.   
<your action 3>

<... repeat several times ...>   
Here is an example:   
<icl\_example>   
=======Reactive=======   
Now, here is the real task, think carefully about the task, then give out your next action(give   
only one action) in the same format as the example above. Take experiment or use your common   
sense to solve the task are all acceptable. Keep in mind that you only have 70 step action budget   
to complete the task.   
=====EOCV/Batched=====   
Now, here is the real task, think carefully about the task, then give out a future action   
sequence in the same format as the example above. Take experiment or use your common sense to   
solve the task are all acceptable. Keep in mind that you only have 70 step action budget to   
complete the task.   
You should plan no more than {max\_planning\_steps} future actions per generation step in this task   
, but you do not have to generate that many every time. If you are not sure about the environment   
state, stop extending the action sequence.   
{interact\_history}

## E.3.2 ICL Examples

## ICL Example for ScienceWorld EOCV ICL Example for ScienceWorld EOCV

\`\`\`python   
# Task Description: Your task is to use chemistry to create green paint. When you are done, focus   
on the green paint.   
# Thought: I need to find materials to create green paint, which are likely to be found in an art   
studio.   
look\_around()   
# Observation: This room is called the bedroom. In it, you see:   
the agent   
a substance called air   
a bed. On the bed is: a mattress. On the mattress is: a white pillow..   
a closet. The closet door is open. In the closet is: nothing.   
a picture   
a table. On the table is: nothing.   
# Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and   
painting supplies are likely to be found, such as an art studio.   
teleport("art studio")   
# Observation: You teleport to the art studio.   
# Thought: Now that I'm in the art studio, I should look around to find the necessary materials   
to create green paint.   
look\_around()   
# Observation: This room is called the art studio. In it, you see:   
the agent   
a substance called air   
a large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood   
cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing   
red paint).   
a table. On the table is: a glass cup (containing nothing).   
a wood cup (containing yellow paint)   
a wood cup (containing blue paint)   
a wood cup (containing red paint)

Your response can be:   
\`\`\`python   
# Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I   
should start by pouring it into an empty container to mix with another color.   
pour("cup containing blue paint", "glass cup")   
# Observation: You pour the contents of the wood cup into the glass cup.   
# Thought: Yellow paint is the other primary color needed to mix with blue to create green. I   
should add the yellow paint to the container with blue paint.   
pour("cup containing yellow paint", "glass cup")   
# Observation: You pour the contents of the wood cup into the glass cup.   
# Thought: I need to mix the blue and yellow paint to create green paint.   
mix("glass cup")   
# Observation: Blue and yellow paint mix to produce green paint.   
# Thought: The task is to create green paint and then focus on it. Now that the green paint is   
made. Before I focus on the green paint, I should look at the glass cup to confirm it contains   
green paint.   
look\_at("glass cup")   
# Observation: <unpredictable observation>   
If the \`look\_at("glass cup")\` observes that "The glass cup contains green paint.", then your next   
response can be:   
\`\`\`python   
# Thought: The task is to create green paint and then focus on it. Now that the green paint is   
made, I should concentrate on the final product.   
focus\_on("green paint")   
# Observation: You focus on the green paint.

## ICL Example for ScienceWorld Reactive

\`\`\`python   
# Task Description: Your task is to use chemistry to create green paint. When you are done, focus   
on the green paint.   
# Thought: I need to find materials to create green paint, which are likely to be found in an art   
studio.   
look\_around()   
# Observation: This room is called the bedroom. In it, you see:   
the agent   
a substance called air   
a bed. On the bed is: a mattress. On the mattress is: a white pillow..   
a closet. The closet door is open. In the closet is: nothing.   
a picture   
a table. On the table is: nothing.   
# Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and   
painting supplies are likely to be found, such as an art studio.   
teleport("art studio")   
# Observation: You teleport to the art studio.   
# Thought: Now that I'm in the art studio, I should look around to find the necessary materials   
to create green paint.   
look\_around()   
# Observation: This room is called the art studio. In it, you see:   
the agent   
a substance called air   
a large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood   
cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing   
red paint).   
a table. On the table is: a glass cup (containing nothing).   
a wood cup (containing yellow paint)

a wood cup (containing blue paint)   
a wood cup (containing red paint)   
# Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I   
should start by pouring it into an empty container to mix with another color.   
pour("cup containing blue paint", "glass cup")   
# Observation: You pour the contents of the wood cup into the glass cup.   
# Thought: Yellow paint is the other primary color needed to mix with blue to create green. I   
should add the yellow paint to the container with blue paint.   
pour("cup containing yellow paint", "glass cup")   
# Observation: You pour the contents of the wood cup into the glass cup.   
# Thought: Now that I have both blue and yellow paint in the same container, I need to mix them   
together to create green paint.   
mix("glass cup")   
# Observation: Blue and yellow paint mix to produce green paint.   
# Thought: The task is to create green paint and then focus on it. Now that the green paint is   
made. Before I focus on the green paint, I should look at the glass cup to confirm it contains   
green paint.   
look\_at("glass cup")   
# Observation: The glass cup contains green paint.   
Your response can be:   
\`\`\`python   
# Thought: The task is to create green paint and then focus on it. Now that the green paint is   
made, I should concentrate on the final product.   
focus\_on("green paint")

ICL Example for ScienceWorld Batched   
\`\`\`python   
# Task Description: Your task is to use chemistry to create green paint. When you are done, focus   
on the green paint.   
# Thought: I need to find materials to create green paint, which are likely to be found in an art   
studio.   
look\_around()   
# Observation: This room is called the bedroom. In it, you see:   
the agent   
a substance called air   
a bed. On the bed is: a mattress. On the mattress is: a white pillow..   
a closet. The closet door is open. In the closet is: nothing.   
a picture   
a table. On the table is: nothing.   
# Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and   
painting supplies are likely to be found, such as an art studio.   
teleport("art studio")   
# Observation: You teleport to the art studio.   
# Thought: Now that I'm in the art studio, I should look around to find the necessary materials   
to create green paint.   
look\_around()   
# Observation: This room is called the art studio. In it, you see:   
the agent   
a substance called air   
a large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood   
cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing   
red paint).   
a table. On the table is: a glass cup (containing nothing).   
a wood cup (containing yellow paint)   
a wood cup (containing blue paint)

Instruction for EB-Habitat EOCV, Reactive, Batched   
## You are a robot operating in a home. Given a task, you must accomplish the task using a   
defined set of actions to achieve the desired outcome.   
## Action Descriptions and Validity Rules   
- Navigation: Parameterized by the name of the receptacle to navigate to. This skill is always   
valid.   
- Pick: Parameterized by the name of the object to pick. Only valid if the robot is close to the   
object, not holding another object, and the object is not inside a closed receptacle. Note that   
some object can be out of observation when you are close to it, so always try to pick the object   
up after navigating to somewhere.   
- Place: Parameterized by the name of the receptacle to place the object on. Only valid if the   
robot is close to the receptacle and is holding an object.   
- Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is   
closed and the robot is close to the receptacle.   
- Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is   
open and the robot is close to the receptacle.   
## The available action id (0 \~ <action\_count>) and action names are: <action\_list>.   
<icl\_example>   
## Guidelines   
0. \*\*Reflect on yourself\*\*: You should first reflect in one sentence on the agent's state in   
relation to the task goal, and then output your next actions.   
1. \*\*Output Plan\*\*: The task is not done yet, avoid generating empty plan.   
2. \*\*Searching for objects\*\*: When searching for some object, navigate around the room, and try   
to pick up specific object. Note that some object can be out of observation when you are close to   
it, so always try to pick the object up after navigating to somewhere even if you can't see it   
in the observation. The robot is using a suction cup robotic arm, so the picked up object will be   
floating below the arm after being picked up.   
3. \*\*Action Validity\*\*: Make sure match the action name and its corresponding action id in the   
output.\n Avoid performing actions that do not meet the defined validity criteria.   
4. \*\*Prevent Repeating Action Sequences\*\*: Do not repeatedly execute the same action or sequence   
of actions.\n Try to modify the action sequence because previous actions do not lead to success.   
5. \*\*Multiple Instances\*\*: There may be multiple instances of the same object, distinguished by   
an index following their names, e.g., cabinet 2, cabinet 3. You can explore these instances if   
you do not find the desired object in the current receptacle.   
6. \*\*Reflection on History and Feedback\*\*: Use interaction history and feedback from the   
environment to refine and enhance your current strategies and actions. If the last action is

a wood cup (containing red paint)   
Your response can be:   
\`\`\`python   
# Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I   
should start by pouring it into an empty container to mix with another color.   
pour("cup containing blue paint", "glass cup")   
# Thought: Yellow paint is the other primary color needed to mix with blue to create green. I   
should add the yellow paint to the container with blue paint.   
pour("cup containing yellow paint", "glass cup")   
# Thought: Now that I have both blue and yellow paint in the same container, I need to mix them   
together to create green paint.   
mix("glass cup")   
# Thought: The task is to create green paint and then focus on it. Now that the green paint is   
made, I should concentrate on the final product.   
focus\_on("green paint")

## E.4 EB-Habitat

## E.4.1 Instructions

invalid, reflect on the reason, such as not adhering to action rules or missing preliminary   
actions, and adjust your plan accordingly.   
=======Reactive=======   
You are required to generate only 1 action for the plan.   
=====Batched=======   
You should generate at most 5 actions in the plan.   
=====EOCV=========   
You should generate at most 5 actions in the plan. Additionally, in order to verify if your plan   
is executable, you need to output your expected observation(described in text) after executing   
each action in the plan. The expected visual state should be concise and only include key   
information such as object locations and states, without detailed descriptions.

## E.4.2 ICL Examples

Note that ICL examples in EB-Habitat serves solely to demonstrate the execution flow of the benchmark to the executor; during actual execution, the executors rely on visual observations, and the EOCV executor is forbidden to perform non-information-seeking actions after skipping information-seeking actions.

ICL Example for EB-Habitat EOCV   
Example 1:   
Human instruction: Put both a toy airplane and a bowl onto the black table.   
Output: {'language\_plan': 'To achieve the goal, the robot must locate the toy airplane and the   
bowl in the room and then move each item to the black table. There are two tables in the room,   
and the robot identifies table 1 as the black table. Therefore, the objective is to place both   
objects on table 1. The plan is as follows: first, navigate to the sofa, pick up the airplane,   
move to table 1, and place the airplane there. Then, proceed to table 2, where the bowl might be,   
pick up the bowl, return to table 1, and set the bowl there.', 'executable\_plan': [{'action\_id':   
12, 'action\_name': 'navigate to the sofa', 'expected\_observation': 'None'}, {'action\_id': 47,   
action\_name': 'pick up the toy airplane', 'expected\_observation': 'The toy airplane is picked up   
by the robot.'}, {'action\_id': 6, 'action\_name': 'navigate to the table 1', 'expected\_observation   
': 'None'}, {'action\_id': 50, 'action\_name': 'place at the table 1', 'expected\_observation': 'The   
toy airplane is placed on the table 1.'}, {'action\_id': 7, 'action\_name': 'navigate to the table   
2', 'expected\_observation': 'None'}]}   
Example 2:   
Human instruction: I made a mistake and left the fridge open. Can you assist me by closing it?   
Output: {'language\_plan': 'The objective is for the robot to close the refrigerator. To do so,   
the robot first navigates to the refrigerator and then closes it.', 'executable\_plan': [{'   
action\_id': 13, 'action\_name': 'navigate to the refrigerator', 'expected\_observation': 'None'},   
{'action\_id': 61, 'action\_name': 'close the refrigerator', 'expected\_observation': 'The   
refrigerator is closed.'}]}

ICL Example for EB-Habitat Reactive/Batched   
Example 1:   
Human instruction: Put both a toy airplane and a bowl onto the black table.   
Output: {'language\_plan': 'To achieve the goal, the robot must locate the toy airplane and the   
bowl in the room and then move each item to the black table. There are two tables in the room,   
and the robot identifies table 1 as the black table. Therefore, the objective is to place both   
objects on table 1. The plan is as follows: first, navigate to the sofa, pick up the airplane,   
move to table 1, and place the airplane there. Then, proceed to table 2, where the bowl might be,   
pick up the bowl, return to table 1, and set the bowl there.', 'executable\_plan': [{'action\_id':   
12, 'action\_name': 'navigate to the sofa'}, {'action\_id': 47, 'action\_name': 'pick up the toy   
airplane'}, {'action\_id': 6, 'action\_name': 'navigate to the table 1'}, {'action\_id': 50, '   
action\_name': 'place at the table 1'}, {'action\_id': 7, 'action\_name': 'navigate to the table   
2'}]}   
Example 2:   
Human instruction: I made a mistake and left the fridge open. Can you assist me by closing it?

Output: {'language\_plan': 'The objective is for the robot to close the refrigerator. To do so, the robot first navigates to the refrigerator and then closes it.', 'executable\_plan': [{ action\_id': 13, 'action\_name': 'navigate to the refrigerator'}, {'action\_id': 61, 'action\_name': 'close the refrigerator'}]}