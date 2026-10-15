# Selective Reasoning for Typed Decision Models: When to Start, Read, and Stop

Bin Zheng∗ Zheng Chen∗ University of Electronic Science and Technology of China zchen@uestc.edu.cn

## Abstract

Typed decision models need an interface that can use additional reasoning while retaining candidate outputs. We train a shared candidateanswer reader on a frozen Qwen3.5-4B backbone to unify direct decisions with on-demand, interruptible reasoning. The reader returns distributions from empty and partial reasoning prefixes. A benefit gate starts reasoning, a natural-state trigger requests explicit reads, and an entropy rule accepts acquired answers and cancels generation. Measurements distinguish answer-query work, generated output, and pipeline time. Across 768 questions from ProofWriter, LogiQA 2.0, and ARC-Challenge, accuracy rises from 70.96% for direct reading to 84.38%. Relative to 512- token terminal reading, accuracy improves by 7.29 percentage points with 2.25% more mean output work. Relative to 1,024-token terminal reading, pipeline time falls by 46.30% with an accuracy difference of −1.17 points. Exactprefix controls isolate the saving from interruption, while read-scheduling controls distinguish reduced observation work from shorter response time.

## 1 Introduction

Typed decision models return a choice from supplied candidates, such as truth labels or answer indices, without generating an explanation. This interface suits repeated judgments within larger workflows. Kev implements it through a trained decision head on a language-model backbone (Palmer, 2026). Explicit reasoning can construct inferences needed for some decisions (Wei et al., 2022). We study how one candidate-answer interface can support immediate decisions, reasoning on demand, and accepted answers returned before reasoning finishes.

Reasoning can correct an initial answer or turn a correct decision into an error. Its expected benefit depends on both changes. Input-dependent budgets and selective-thinking methods adapt computation to the question (Snell et al., 2025; Zhang et al., 2025; Fang et al., 2025). Once reasoning starts, intermediate answers offer opportunities to return early (Liu and Wang, 2025; Boppana et al., 2026).

![](images/a76ef152608822f726b0f7cc8f276b981dcdfa01de30c637a87304ca6875ab24.jpg)  
Figure 1: Accuracy and mean pipeline time for ten configurations on the same 768 questions. The star marks Ours at $\tau = 0 . 7 0$ . Parenthesized numbers are reasoning-token caps; Fixed budget denotes terminal candidate reading.

Answer availability and answer acquisition are different operations. A natural generation state can indicate an opportunity to read, but obtaining a candidate distribution requires an explicit backbone query with an answer cue. Acceptance acts on this acquired distribution. Frequent reads incur input-processing work, whereas sparse reads can delay acceptance and consume more generation (Fu et al., 2025; Xiang et al., 2026; Dong et al., 2026). Keeping the candidate reader fixed lets us measure these costs separately from changes to answer generation or parsing.

We train a shared candidate-answer reader on a frozen Qwen3.5-4B backbone (Qwen Team, 2026) using empty and partial reasoning prefixes. A benefit gate estimates correction minus regression to select questions for reasoning. A natural-state trigger requests an explicit read, and an entropy rule accepts its answer and cancels active generation. The backbone retains its original generation policy until cancellation. DEER already separates transition monitoring, trial answers, and confidencebased exit (Yang et al., 2025); our controller connects initial candidate decisions and partial-prefix reads through the same trained interface. Its trigger selects when to acquire a distribution rather than directly deciding when to exit.

Across 768 questions, selective reasoning improves candidate decisions at comparable output work and offers shorter pipeline time than 1,024- token terminal reading, as shown in Figure 1. Separate matched controls verify interruption on identical token prefixes: acceptance and cancellation save 34.09% of time with two fewer correct answers. Holding the gate fixed, learned reading reduces answer queries but increases time relative to fixed reads. These controls distinguish candidateanswer quality, observation work, and the time required to return a decision.

Our contributions are:

• Unified candidate decisions. One trained reader preserves the typed output interface across immediate decisions and decisions enhanced by interruptible reasoning.

• Controlled execution evidence. Matched token-prefix controls isolate the runtime saving from accepting an intermediate candidate answer and cancelling its continuing trajectory.

• Acquisition trade-offs. Fixed-reader comparisons expose how reading schedules exchange query and input-processing work for generation and response time.

## 2 Related Work

Candidate decision interfaces. Kev returns supplied candidate selections through a trained decision head (Palmer, 2026). Chain-of-thought prompting provides intermediate computation, and self-consistency aggregates sampled paths (Wei et al., 2022; Wang et al., 2023). We train one reader across empty and partial prefixes on frozen Qwen3.5-4B (Qwen Team, 2026), retaining the same candidate outputs for direct and reasoningenhanced decisions.

Reasoning initiation and budgets. Adapt-Think, Thinkless, and AutoThink select direct or reasoning-assisted responses (Zhang et al., 2025; Fang et al., 2025; Tu et al., 2025). Sonata uses prefill states to predict the fraction of correct sampled non-thinking answers, termed self-consistency in that work, and extends this prediction to thinking gain (Li et al., 2026). Compute allocation, budget forcing, and shorter-chain selection control how much reasoning is used (Snell et al., 2025; Muennighoff et al., 2025; Hassid et al., 2025). Our gate estimates corrections minus regressions from paired outcomes of the shared candidate reader. Initial routing connects to that reader’s intermediate acceptance and cancellation policy.

Intermediate answers and early stopping. Answer Convergence and Dynasor assess trialanswer consistency or certainty (Liu and Wang, 2025; Fu et al., 2025). DEER monitors transitions, induces trial answers, and accepts confident answers, with branch-parallel decoding to overlap trials and continued reasoning (Yang et al., 2025). DTSR screens reflection signals before requesting a sufficiency score (Xiang et al., 2026). Our reader acquires a fixed-candidate distribution that a separate rule tests for acceptance. The same reader supplies initial decisions and intermediate answers. Reads use separate engine requests, whereas DEER’s branch-parallel implementation uses a specialized attention mask and dynamic KV management.

Internal signals and execution cost. Reasoning Theater probes answer availability (Boppana et al., 2026); NEAT and BLADE use internal signals to control exit or reflection (Liu et al., 2026; Fu et al., 2026). LYNX trains on forced-exit outcomes and applies a hidden-state probe at natural cues to decide exit (Akgül et al., 2025). LearnStop predicts stopping from probed answers and studies query density and cache costs (Dong et al., 2026). Our natural-state trigger selects acquisition before a candidate distribution is available. Acceptance acts on the subsequent read. We hold the candidate reader fixed and measure online acquisition and cancellation, with exact-prefix controls isolating interruption from reasoning trajectory changes.

![](images/23c0816686b78137dc192190db02b1596fda94255f98a2e3222fbcc5bdfd825a.jpg)  
Figure 2: On-demand, interruptible reasoning through a shared candidate interface. The abridged ProofWriter example shows reasoning correcting a direct answer and intermediate acceptance avoiding later generation. The start gate also permits direct return. $P , T _ { i } ,$ , and C denote the prompt, thought fragments, and candidate answer. Grey thoughts mark avoided generation.

## 3 Unified Candidate Decisions with Interruptible Reasoning

A shared candidate-answer reader supports direct decisions from an empty reasoning prefix and enhanced decisions from partial reasoning. The decision process can return immediately, start reasoning on demand, or return an accepted intermediate answer and cancel unfinished generation. Both paths retain the same candidate output interface on a frozen backbone.

The framework separates three decisions: whether to initiate reasoning, when to acquire an answer, and whether to accept it. Our implementation uses a benefit gate, a trigger based on natural generation states and prefix length, and entropybased acceptance. Figure 2 summarizes these decisions and the direct-return alternative. The reader and two learned control modules are fitted separately; backbone weights remain frozen.

## 3.1 A Shared Candidate-Answer Reader

Input x contains a question, accompanying material, and K candidate answers. A frozen Qwen3.5- 4B backbone (Qwen Team, 2026) supports autoregressive reasoning and an explicit answer reader. Let $z _ { 1 : t }$ be a generated prefix, with $z _ { \mathrm { 1 : 0 } }$ empty. The reader uses the empty prefix for direct decisions and a generated prefix for reasoning-enhanced decisions. A read processes x and this prefix, appending </think> if the prefix lacks a thinking closure, followed by a <FINAL> answer cue. $\mathbf { A }$ bias-free linear head maps the state $\tilde { h } _ { t }$ at the cue’s last token to candidate probabilities:

$$
p _ { t } = \mathrm { s o f t m a x } ( W _ { 1 : K } \tilde { h } _ { t } ) , \quad a _ { t } = \arg \operatorname* { m a x } _ { j \leq K } p _ { t } ( j ) .\tag{1}
$$

Here $W \in \mathbb { R } ^ { 4 \times 2 5 6 0 }$ has 10,240 parameters, and Eq. 1 applies when $K \ \leq \ 4 .$ For more candidates, we normalize the native language-model logits over candidate-index tokens at the same position. Both interfaces require the backbone to process the answer query. Each query emits one auxiliary native token to complete the request, while the prediction is obtained from $p _ { t }$

An initial read supplies $a _ { 0 } , p _ { 0 }$ , and $\tilde { h } _ { 0 }$ . The controller either returns $a _ { 0 }$ or starts reasoning up to budget B. The natural state $h _ { t }$ is the final backbone state passed to the vocabulary projection at the last token of $z _ { 1 : t }$ . It is saved from decoding and indexed by prefix length. The trigger uses this state when retained; obtaining $\tilde { h } _ { t }$ requires an explicit answer query. Retention and availability follow the state interface in Appendix B.

Training fits the reader across both prefix types while keeping the backbone frozen. It minimizes

$$
\mathcal { L } _ { \mathrm { r e a d } } = - \mathbb { E } _ { i , t } \left[ \log p _ { i , t } ( y _ { i } ) \right] ,\tag{2}
$$

where the expectation is uniform over training questions i and checkpoints $\begin{array} { r l r l } { t } & { { } \in } & { \tau } & { { } = } \end{array}$ {0, 128, 256, 512, 1024}. Positions are capped at each trace’s available length. The selected head is fixed for online evaluation; optimization details appear in Appendix H.

## 3.2 When to Start: Expected Corrective Benefit

The initiation decision selects direct return or a reasoning trajectory. Our implementation estimates corrective benefit from paired initial and terminal answers under cap B, using the same reader. For training supervision, let $c _ { 0 } = \mathbb { I } [ a _ { 0 } = y ]$ and $c _ { B } = \mathbb { I } [ a _ { B } = y ]$ denote their correctness. Correctness can increase by one, decrease by one, or remain unchanged. Thus,

$$
\begin{array} { c } { { \mathbb { E } [ c _ { B } - c _ { 0 } \mid \phi _ { 0 } ] = \operatorname* { P r } ( c _ { 0 } = 0 , c _ { B } = 1 \mid \phi _ { 0 } ) } } \\ { { - \operatorname* { P r } ( c _ { 0 } = 1 , c _ { B } = 0 \mid \phi _ { 0 } ) . } } \end{array}\tag{3}
$$

We estimate the two events with logistic models:

$$
\begin{array} { r } { g ( x ) = \sigma ( w _ { g } ^ { \top } \phi _ { 0 } + b _ { g } ) , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { d ( x ) = \sigma ( w _ { d } ^ { \top } \phi _ { 0 } + b _ { d } ) . } \end{array}\tag{5}
$$

The score $u ( x ) \ = \ g ( x ) - d ( x )$ estimates corrective benefit. Input ϕ<sub>0</sub> concatenates the initial answer state with maximum probability, top-two margin, normalized entropy $H ( p _ { 0 } ) / \log K$ , and $\log ( 1 + | x | )$ , where |x| is the input token count. Feature standardization is fitted on gate-training groups and folded into deployed weights. The controller starts when $u ( x ) > \theta ;$ otherwise it returns a<sub>0</sub>.

Each event model minimizes L2-regularized binary cross-entropy using out-of-fold reader predictions from training material groups. The deployed score uses the raw logistic probabilities. A separate threshold split sets θ for a target reasoning fraction of 0.75. Actual routing rates and token costs can vary on new questions. The gate targets the $B = 1 0 2 4$ terminal endpoint, and its combination with early stopping is evaluated as a complete policy. Appendix A specifies regularization and fitting settings.

## 3.3 When to Read: Selective Answer Queries

The acquisition decision schedules explicit candidate-answer queries. Our implementation screens prefixes using the natural state $h _ { t }$ and prefix length before paying for a read. Its deployed computation is

$$
r _ { t } = \sigma ( w ^ { \top } h _ { t } + \alpha t / 1 0 2 4 + b ) .\tag{6}
$$

The trigger uses no task identity. At inference, $p _ { t }$ is computed only after a read is requested: $r _ { t }$ predicts a reading opportunity, and the read supplies the answer and certainty.

Reading opportunities come from completesentence boundaries. With generation active and the reader idle, the trigger considers the latest pending prefix when its absolute sentence ordinal satisfies s mod $4 = 0$ and its indexed $h _ { t }$ is available. Each scored event is consumed once, requesting a read if $r _ { t } \geq 0 . 2 5$ . New sentence events replace the pending prefix, so coalescing can skip checkpoints. A submitted read remains bound to its selected token prefix while generation continues. Boundaries align to original token IDs and exclude the next sentence’s content. Appendix B gives the waiting, replacement, and completion rules.

Training supervision identifies prefixes with correct, concentrated reader predictions. Define entropy certainty by

$$
q _ { t } = 1 - \frac { H ( p _ { t } ) } { \log K } .\tag{7}
$$

At training time, an out-of-fold reader supplies

$$
v _ { i , t } = \mathbb { I } [ a _ { i , t } = y _ { i } \land q _ { i , t } \geq 0 . 8 5 ] .\tag{8}
$$

A logistic trigger minimizes L2-regularized weighted binary cross-entropy for $v _ { i , t }$ , assigning equal total weight to each question. Its training features are the natural state projected onto 64 training-fitted principal components and t/1024. Standardization and projection fold into Eq. 6; Appendix A gives the fitting configuration.

## 3.4 When to Stop: Accepting an Observed Answer

Acceptance acts on an acquired candidate distribution. Our implementation accepts the prefixbound answer $a _ { t }$ when its entropy certainty satisfies $q _ { t } \geq \tau$ and cancels generation if it is active. The representative policy uses $\tau = 0 . 7 0 ; 0 . 8 5$ and 0.95 provide additional operating points with the same gate and trigger. The trigger’s training target stays at 0.85. Entropy certainty measures concentration of the answer distribution; its threshold does not specify a probability of correctness.

A rejected read leaves generation active and allows the latest pending event to be considered. At natural completion or the budget limit, the controller finishes any outstanding read. If no answer is accepted, a terminal read supplies the answer without a certainty requirement. A prefix answer accepted after generation ends is recorded separately from cancellation that saves generation.

## 3.5 Online Execution and Cost

Generation and answer requests share one backbone and one vLLM engine (Kwon et al., 2023), with at most one answer query in flight. Starting reasoning submits a separate request on x. It remains scheduled while a read processes a separate prefix plus answer cue; the cue never enters the reasoning trace. Overlap shares GPU resources and can affect generation throughput. The configuration retains autoregressive KV state within a request and disables cross-request prefix caching, so generation and each answer query incur their own input-processing work.

For the complete controller, answer requests count one initial read, all intermediate reads, and a terminal read if needed. Trigger calls count actual evaluations of $r _ { t }$ . Output tokens include natural generation and one auxiliary token per read. The input-token counts in Table 10 sum the generation prompt and every answer-query input.

For an accepted read, let t be the queried prefix length, d the generated length when the decision arrives, and c the length when cancellation takes effect. We record c − t and c − d as subsets of the natural output already counted. Pipeline time spans initial reading, routing, generation, sentence detection, state capture, triggers, answer reading, and cancellation under the event-driven procedure in Table 5. Component durations describe overlapping work and are not summed into a latency estimate.

## 4 Experimental Setup

## 4.1 Tasks and Model Fitting

Evaluation groups. We evaluate 256 questions each from ProofWriter, LogiQA 2.0, and ARC-Challenge, covering open-world logical judgments, passage-based reasoning, and scientific knowledge questions (Tafjord et al., 2021; Liu et al., 2023; Clark et al., 2018). ProofWriter and LogiQA contribute one question per reserved official-training material group, disjoint from fitting, development, and earlier component evaluation. Groups connect shared source identities and identical or near-duplicate material. ARC uses an official-test subset with development overlap excluded. Candidate order is preserved. ProofWriter and LogiQA require material-grounded answers, while ARC permits scientific knowledge and tests transfer of the fitted components.

Fitting and selection. Fitting uses 1,024 training and 256 development questions from ProofWriter and LogiQA under the data-isolation protocol in Appendix A. Reader selection minimizes development negative log-likelihood on a frozen backbone. Gate fitting and threshold selection use separate training-material partitions. Out-of-fold reader predictions supervise the gates and trigger, with equal trigger-training weight per question. The reader, control parameters, and three acceptance thresholds are fixed before primary evaluation.

## 4.2 Comparison Policies

Our comparisons address three questions. Sharedreader policies test computation allocation under a fixed candidate interface. Generated-answer baselines compare complete configurations with different answer and stopping interfaces. Matched controls explain the roles of initiation, reading, and intermediate acceptance within our controller.

Eight shared-reader configurations compare fixed budgets with selective allocation using identical backbone and reader weights. Direct reading in Table 1 establishes initial accuracy, and terminal reading at caps of 512, 1,024, and 2,048 tokens measures reasoning gains. The simple controller combines margin routing, reading after at least eight additional sentences, and entropy acceptance at 0.85. Our controller uses acceptance thresholds 0.70, 0.85, and 0.95, with their operating points reported in Table 6. Both start thresholds are calibrated to a target reasoning fraction of 0.75.

Generated-answer policies compare alternative answer and stopping interfaces using the settings in Appendix B: forced terminal answering at 1,024 tokens, Answer Consistency (AC), a candidateanswer adaptation of Answer Convergence using ten consecutive matching trials (Liu and Wang, 2025), Qwen-adapted DEER at 1,024 and 2,048 tokens (Yang et al., 2025), and native generation at 8,192 tokens. All start reasoning, with natural completion allowed before the cap. Trial and forced-final outputs count additionally, while the native cap includes the final answer.

The supplementary comparisons in Appendix F.1 select simple reading intervals and DEER trial thresholds on 48 development questions disjoint from evaluation, then compare controllers at corresponding acceptance thresholds. These development questions select policies;

<table><tr><td>Method</td><td>Cap</td><td>PW (%)</td><td>LQ (%)</td><td>ARC (%)</td><td>Avg. (%)</td><td>Time (s)</td><td>Output tok.</td><td>Queries</td></tr><tr><td>Shared-reader baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Direct read</td><td>0</td><td>58.20</td><td>66.41</td><td>88.28</td><td>70.96</td><td>0.080</td><td>1.0</td><td>1.00</td></tr><tr><td>Terminal read</td><td>512</td><td>67.97</td><td>68.75</td><td>94.53</td><td>77.08</td><td>4.773</td><td>513.0</td><td>1.00</td></tr><tr><td>Terminal read</td><td>1,024</td><td>83.59</td><td>76.95</td><td>96.09</td><td>85.55</td><td>9.410</td><td>1,022.7</td><td>1.00</td></tr><tr><td>Terminal read</td><td>2,048</td><td>85.55</td><td>78.91</td><td>96.88</td><td>87.11</td><td>17.414</td><td>1,887.7</td><td>1.00</td></tr><tr><td>Generated-answer baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Forced terminal answer</td><td>1,024</td><td>80.86</td><td>76.95</td><td>96.09</td><td>84.64</td><td>9.446</td><td>1,025.1</td><td>1.00</td></tr><tr><td>Answer Consistency (k = 10)</td><td>1,024</td><td>41.02</td><td>71.88</td><td>91.02</td><td>67.97</td><td>4.362</td><td>307.0</td><td>12.28</td></tr><tr><td>DEER adaptation</td><td>1,024</td><td>80.86</td><td>78.91</td><td>96.09</td><td>85.29</td><td>9.582</td><td>1,026.7</td><td>1.76</td></tr><tr><td>DEER adaptation</td><td>2,048</td><td>86.33</td><td>80.47</td><td>96.09</td><td>87.63</td><td>18.500</td><td>1,902.7</td><td>4.43</td></tr><tr><td>Native answer</td><td>8,192</td><td>79.69</td><td>60.16</td><td>84.38</td><td>74.74</td><td>38.213</td><td>4,141.7</td><td>0.00</td></tr><tr><td>Ours (τ = 0.70)</td><td>1,024</td><td>83.98</td><td>75.39</td><td>93.75</td><td>84.38</td><td>5.053</td><td>524.5</td><td>2.43</td></tr></table>

Table 1: Accuracy and per-question costs on 768 questions, 256 per task. PW, LQ, and ARC denote ProofWriter, LogiQA 2.0, and ARC-Challenge; Avg. is task-mean accuracy. Cap bounds reasoning tokens, including the native final answer. Output includes auxiliary tokens; Queries counts answer/trial requests. Time is mean pipeline seconds.

the separate 48-question timing subset comes from evaluation and plays no role in selection.

## 4.3 Execution and Measurement

All configurations use Qwen3.5-4B on A100 80GB with BF16 vLLM, greedy decoding, compiled execution, and disabled cross-request prefix caching.

We report per-task and task-equal accuracy, and mean pipeline latency. Invalid generated answers count as errors. Synchronized timing covers initial reading, routing, generation, sentence detection, state capture and synchronization, triggers, queries, and cancellation. Loading, warm-up, tokenization, cache resets, and record writes are outside the timer. Input and output tokens, answer/trial queries, and trigger calls characterize computational work. Outputs include auxiliary tokens and generation beyond an accepted prefix.

Principal estimates use first executions on the same 768 questions for all 13 configurations. Figure 3 plots accuracy and both costs for seven shared-reader configurations. Matched controls interleave policy executions within each question. Table 7 reports first-execution timing on 48 evaluation questions, 16 per task. Two additional executions support repeated timing analyses, averaging within question. Pointwise 95% intervals use 10,000 paired material-group bootstrap resamples, stratified by ProofWriter subtype, LogiQA, and ARC, as specified in Appendix C.

## 4.4 Matched Mechanism Analysis

We compare Ours with three component removals and two policy replacements on the same 768 questions. All share the reader and 1,024-token cap, with τ = 0.70 for intermediate acceptance when enabled. Policies interleave within each question against a local complete-controller execution. Table 2 reports the removals; Table 9 includes both removals and replacements with complete work metrics.

Always reason forces generation after retaining the initial read and gate computation, testing selective initiation. No early stop retains intermediate queries but disables intermediate acceptance and cancellation, testing interruptible return. Gate + terminal read jointly removes intermediate reads and early acceptance, testing the full process policy against routing followed by terminal reading.

Margin gate replaces benefit scoring with candidate-probability margin routing, while retaining the downstream reading and acceptance policies. It tests corrective-benefit modeling against uncertainty-based allocation. Fixed reads requests an answer after at least eight additional sentences, while retaining the start gate, asynchronous latestprefix handling, and acceptance rule. It tests how a reading schedule exchanges query overhead for answer-acceptance timing.

Routing diagnostics compare initial answers with the deployed Always reason policy’s outcomes, whereas gate training targets terminal-1,024 correctness. Stopping analysis checks exact token-prefix agreement and distinguishes cancellation from acceptance after generation ends.

Supplementary studies in Appendix F cross both gates and reading schedules and compare serial with concurrent execution, reusing the observed evaluation cohort and original controller.

![](images/64e287d0907308b404ae11b88c48f8863912f2ea47f53e13e00d7f86b4d77f5f.jpg)  
Figure 3: Accuracy and cost for seven shared-reader configurations on the same 768 questions and A100. Output work and pipeline time include auxiliary queries. Whiskers are pointwise 95% material-group bootstrap intervals. Terminal labels give reasoning caps; shading marks the predefined τ = 0.70 policy.

## 5 Results and Analysis

## 5.1 Unified Direct and Reasoning-Enhanced Decisions

Optional reasoning strengthens decisions through a unified candidate-answer interface. Across 768 questions, the predefined $\tau ~ = ~ 0 . 7 0$ controller raises accuracy from 70.96% for direct reading to 84.38%: 648 rather than 545 correct answers, or 103 additional correct decisions and a gain of 13.41 percentage points, as reported in Table 1 and Figure 3. The same candidate-answer reader supports immediate decisions and answers acquired from partial reasoning. Paired 95% confidence intervals accompany the primary and matched contrasts in Table 11.

Selective allocation also obtains more correct answers at comparable average output work. The 512-token terminal reader answers 592 questions correctly, reaching 77.08% accuracy, while the controller adds 56 correct answers, improving accuracy by 7.29 percentage points. Mean output rises from 513.0 to 524.5 tokens, or 2.25%, and mean pipeline time from 4.773 to 5.053 seconds, or 5.87%.

Relative to 1,024-token terminal reading, the controller reduces pipeline time by 46.30% and output work by 48.71%. Accuracy changes from 85.55% to 84.38%, a paired difference of −1.17 points.

The predefined simple controller in Appendix F.1 reaches 85.16% at 5.834 seconds; its development-calibrated variant reaches 648/768 at $\tau = 0 . 7 0$ . DEER reaches 85.29% at 1,024 tokens and 87.63% at 2,048.

<table><tr><td>Variant</td><td>Acc. (%)</td><td>∆Acc. (pp)</td><td>∆Time (%)</td></tr><tr><td>Ours</td><td>84.90</td><td>一</td><td></td></tr><tr><td>No early stop</td><td>85.16</td><td>+0.26</td><td>+51.71</td></tr><tr><td>Gate + terminal read</td><td>83.98</td><td>-0.91</td><td>+43.30</td></tr><tr><td>Always reason</td><td>85.55</td><td>+0.65</td><td>+23.04</td></tr></table>

Table 2: Component removals on 768 questions with cap 1,024 and $\tau = 0 . 7 0$ . Matched Ours answers 652 correctly in 4.960 s. Deltas subtract Ours; time changes are normalized by its mean. Gate + terminal read removes intermediate queries and early acceptance. Table 9 reports complete costs and policy replacements.

## 5.2 Interruptible Reasoning through Intermediate Answer Acceptance

Holding the start gate and reading trigger fixed, intermediate answer acceptance and cancellation reduce mean time from 7.524 to 4.960 seconds, saving 34.09%, while correct answers change from 654 to 652. The controls in Table 2 use a separate matched execution on the same 768 questions, with 652 correct answers for the completecontroller reference, compared with 648 in the primary execution in Table 1.

Exact-prefix controls support attributing this saving to interrupting the reasoning trajectory. All 369 cancelled traces are exact token prefixes of the no-stop traces, and the other 399 traces are identical. Initial answers and start decisions agree on every question. Figure 4 separates wrong exits from errors recoverable by continuation: of 25 wrong exits, 23 remain wrong and two are corrected. Those two recovered answers represent 0.26 percentage points over all 768 questions.

Further reasoning can correct a wrong intermediate answer along the same trajectory. The paired control measures these forgone corrections alongside the computation avoided.

![](images/a5ccc506a7211015c2a42a52e9b4f0d3d411d546dc371d4e53862f033df13a28.jpg)

![](images/f7c1465e46f2db88f7eeca4dd6f0a6999e011d219a5b45d1308e14e3eed6e41b.jpg)  
Figure 4: Stopping outcomes and pipeline time against the matched no-early-stop control. Panel (a) compares 369 cancelled answers with continuation to natural completion or 1,024 tokens. Panel (b) averages all 768 questions. Whiskers show pointwise 95% paired material-group bootstrap intervals.

Against routing followed by terminal reading, the controller answers seven more questions correctly and saves 30.22% of time. This comparison jointly measures intermediate acquisition and early acceptance.

## 5.3 Selective Initiation of Reasoning

Selective starting allows the framework to return direct decisions without launching generation on 191 questions. Compared with Always reason, it reduces reasoning invocations from 768 to 577 and saves 18.72% of pipeline time, with correct answers changing from 657 to 652. This allocates additional computation to a subset of questions while retaining the direct-answer path.

The choice of start score determines which correction and regression opportunities receive reasoning. The routing diagnostics in Appendix D show that benefit routing avoids more regressions but selects fewer corrections than margin routing. Benefit-minus-margin accuracy differs by −0.39 points, and time by +0.121 seconds. Neither these contrasts nor the factorial controls in Appendix F.2 establish an advantage for benefit over margin.

## 5.4 Read Scheduling and Execution Cost

Reading less often reduces observation work while changing when an answer can be accepted. Under the same gate and stopping rule, learned reading reduces queries from 4.105 to 2.427 per question, or 40.88%. Correct answers change from 643 to 652, a difference of +1.17 points. Mean time rises from 4.554 to 4.960 seconds, a paired increase of 0.406 seconds. Fewer reads can defer acceptance and allow more generation.

An intermediate read processes a fixed prefix while generation can continue on the same engine. Its response delay can allow generation to extend beyond that prefix, linking query frequency to observation workload and return time.

The calibrated simple policy provides a second view of this trade-off. $\mathrm { A t } \tau = 0 . 7 0$ , both complete controllers answer 648/768 correctly, while the learned controller makes 42.75% fewer queries. Repeated timing in Appendix F.1 resolves no latency advantage. This separates reduced observation work from the time required to return an answer.

At the trigger’s original training target, the cached development diagnostic in Appendix G finds better pooled answer-availability prediction from natural states, without improved withinquestion temporal ranking. Follow-up controls examine trigger-target alignment and bounded waiting in Appendix E, and serial versus concurrent reading in Appendix F.3. Read scheduling must balance query costs with the generation consumed before acceptance: query count alone does not determine response latency.

## 6 Conclusion

One shared candidate-answer reader unifies direct decisions with on-demand, interruptible reasoning on a frozen backbone. Empty and partial reasoning prefixes retain the same typed output interface, allowing the controller to allocate computation without changing how answers are returned. Across three tasks, this unified process improves accuracy at comparable output work and reduces time relative to longer terminal reading.

Matched token-prefix controls isolate the saving from accepting an intermediate answer and cancelling unfinished generation, with the preceding trajectory held fixed. Read-scheduling controls show that fewer queries reduce observation work but can delay acceptance and consume more generation. Efficient candidate decisions therefore require joint control over initiation, answer acquisition, and acceptance, with query costs and computation before return measured together.

## Limitations

We evaluate one frozen Qwen3.5-4B backbone, fitting on ProofWriter and LogiQA and testing transfer on ARC-Challenge. Material groups separate fitting and evaluation. Controller supervision differs from online acceptance, and entropy certainty measures answer concentration.

Our measurements cover single-question pipeline time and token work. Energy and service throughput remain unmeasured. Separate greedy runs can produce different trajectories, so comparisons retain local paired executions. Follow-up policies reuse observed questions; choosing a revised controller requires evaluation on fresh data.

## Ethics Statement

This work uses public reasoning benchmarks and model inference, without a user study or deployment in consequential decisions. Early-stopping errors can propagate through a downstream workflow; applications with unequal error costs need task-specific acceptance criteria. Dataset and model use and any redistribution must follow the original licenses. We report accuracy alongside computational cost so that efficiency gains do not obscure changes in answer quality.

## References

Ömer Faruk Akgül, Yusuf Hakan Kalaycı, Rajgopal Kannan, Willie Neiswanger, and Viktor Prasanna. 2025. LYNX: Learning dynamic exits for confidence-controlled reasoning. arXiv preprint arXiv:2512.05325. Preprint, arXiv:2512.05325.

Siddharth Boppana, Annabel Ma, Max Loeffler, Raphael Sarfati, Eric Bigelow, Atticus Geiger, Owen Lewis, and Jack Merullo. 2026. Reasoning theater: Disentangling model beliefs from chain-of-thought. arXiv preprint arXiv:2603.05488, version 4.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Zhe Dong, Fang Qin, and Manish Shah. 2026. When does learning to stop help? a cost-aware study of early exits in reasoning models. arXiv preprint arXiv:2606.30852, version 2.

Gongfan Fang, Xinyin Ma, and Xinchao Wang. 2025. Thinkless: LLM learns when to think. In Advances in Neural Information Processing Systems, volume 38, pages 167994–168021.

Keshu Fu, Keqin Peng, Jun Bai, Shuhan Qin, Chen Li, Junzhu Liang, Yefei Chen, Jiaqi Li, and Yuanxin Ouyang. 2026. BLADE: Boundary-expanded and layer-adaptive dynamic exit for efficient LLM reasoning. arXiv preprint arXiv:2607.28966. Preprint, arXiv:2607.28966.

Yichao Fu, Junda Chen, Siqi Zhu, Zheyu Fu, Zhongdongming Dai, Yonghao Zhuang, Yian Ma, Aurick Qiao, Tajana Rosing, Ion Stoica, and Hao Zhang. 2025. Efficiently scaling LLM reasoning programs with Certaindex. In Advances in Neural Information Processing Systems, volume 38, pages 157566– 157604.

Michael Hassid, Gabriel Synnaeve, Yossi Adi, and Roy Schwartz. 2025. Don’t overthink it. preferring shorter thinking chains for improved LLM reasoning. Preprint, arXiv:2505.17813. Version 2, revised February 3, 2026.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, pages 611–626. Association for Computing Machinery.

Pingzhi Li, Bairu Hou, Yun Zhu, Yihao Feng, Ke Ye, Tao Lei, Zhifeng Chen, Tianlong Chen, and Xianzhi Du. 2026. Adaptive thinking: Large language models know when to think in latent space. In The Fourteenth International Conference on Learning Representations.

Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. 2023. LogiQA 2.0—an improved dataset for logical reasoning in natural language understanding. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2947–2962.

Kang Liu, YongKang Liu, Xiaocui Yang, Peidong Wang, Wen Zhang, Shi Feng, Yifei Zhang, and Daling Wang. 2026. NEAT: Neuron-based early exit for large reasoning models. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 24616–24627. Association for Computational Linguistics.

Xin Liu and Lu Wang. 2025. Answer convergence as a signal for early stopping in reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17896– 17907. Association for Computational Linguistics.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321. Association for Computational Linguistics.

Jared Palmer. 2026. Kev. Software repository. Source commit 0fe8fc9; accessed 2026-10-11.

Qwen Team. 2026. Qwen3.5-4B. Model card. Accessed 2026-10-11.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations.

Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634. Association for Computational Linguistics.

Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. 2025. Learning when to think: Shaping adaptive reasoning in R1-style models via multi-stage RL. In Advances in Neural Information Processing Systems, volume 38, pages 16989–17015.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Yang Xiang, Yixin Ji, Ruotao Xu, Dan Qiao, Zheming Yang, Juntao Li, and Min Zhang. 2026. When is thinking enough? early exit via sufficiency assessment for efficient reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23541–23556. Association for Computational Linguistics.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. 2025. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, version 3.

Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. 2025. AdaptThink: Reasoning models can learn when to think. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 3716–3730. Association for Computational Linguistics.

## A Reproducibility Specification

The supplementary materials at https://github. com/binzheng-joey/selective-reasoning contain code, reader weights, control parameters, evaluation manifests, and per-question measurements. A CPU script recomputes the reported statistics from these measurements; model inference requires the external model, source data, and feature caches described in the README. Table 3 summarizes the experimental studies.

## A.1 Data and Material Isolation

ProofWriter uses the open-world, depth-5 files of release V2020.12.3; LogiQA 2.0 and ARC-Challenge use the source revisions recorded in the archive. ProofWriter and LogiQA evaluation questions come from official training data, and ARC from official test data. Reader fitting and observed development contain 512 and 128 questions per task from ProofWriter and LogiQA.

Material groups are connected components sharing source IDs within a dataset, normalized material, normalized sentence sets, or five-wordshingle Jaccard similarity of at least 0.90. Normalization lowercases and joins \w+ matches; sentence sets split at [.!?]+ and deduplicate normalized sentences. Training excludes previously observed components, components spanning official splits, and conflicting source IDs. All prefixes inherit their question’s component.

Primary evaluation selects one question per unused component after propagating exclusions from all recorded training, development, and earlier evaluations, including the earlier 1,024-question native confirmation pool. Deterministic hash ranking selects 64 components per ProofWriter subtype and 256 LogiQA components. The archive retains the ranking rules and selected IDs.

ARC groups questions by case-folded text with collapsed whitespace and selects 256 groups by deterministic hash ranking. Development excludes official training or validation questions whose normalized text appears in official test data. Source labels and candidate order are preserved. The evaluation contains 255 four-option items and one fiveoption item, TIMSS\_1995\_8\_N4.

The archive README locates data splits, material-group manifests, reader folds, deployed weights, and aggregate results. arr\_formal\_run.py defines prompt projection, arr\_integrated\_online.py implements the controller, and sentence\_online.py provides the reader backend. The evaluation manifest records all 768 question IDs and strata; separate protocols identify matched and supplementary executions.

<table><tr><td>Study</td><td>Questions/tasks</td><td>Answer interface</td><td>Token cap</td><td>Evidence role and backend</td></tr><tr><td>Primary comparison</td><td>768; PW/LQ/ARC</td><td>Current answer-position head; generated-answer</td><td>0-8,192</td><td>Frozen online policies; com- piled A100 vLLM</td></tr><tr><td>mentary policies</td><td>fixed 48 for repeated coefficients</td><td>adaptations Matched and supple- Same observed 768; Current head and deployed 1,024</td><td></td><td>Local component controls and follow-up; A100 vLLM</td></tr><tr><td>ment</td><td>time served development; answer-cue heads PW/LQ</td><td>Reader/trigger develop- 1,024 train; 256 ob- Matched native-state and 1,024</td><td></td><td>Fitting and fixed-trajectory analysis; cached readouts</td></tr><tr><td>Earlier native confirma- Separate 768; PW/LQ tion</td><td></td><td>Earlier linear head and na- 1,024/8,192 tive candidate logits</td><td></td><td>Frozen component compar- isons; A100 vLLM</td></tr><tr><td>Kev routing</td><td>Separate 512; PW/LQ</td><td>Kev Base head; external na- 8,192 tive generator</td><td></td><td>Two-model component study; A100 vLLM</td></tr></table>

Table 3: Study identities and evidence roles. PW, LQ, and ARC denote ProofWriter, LogiQA 2.0, and ARC-Challenge.

## A.2 Generation and Answer-Query Templates

All policies start from the same task-specific generation prompt. ProofWriter material is the original theory, its task is the selected statement, and its candidates are True, False, and Unknown, in that order. LogiQA material, question, and options retain their source fields. ARC material is the science question and its task is the exact string Select the candidate that best answers this science question.

The system message for ProofWriter and LogiQA is:

Use only the supplied material. Do not invent missing facts. Keep the original candidate order. Follow the requested output format.

## The ARC system message is:

Use the supplied question and relevant scientific knowledge. Keep the original candidate order. Follow the requested output format.

The user-message template has one numbered line per candidate. Braced fields are substitutions; the displayed ellipsis expands to the intervening candidates, and blank lines are two newline characters.

Candidates (in the given order):   
1. {candidate 1}   
{K}. {candidate K}   
{generation instruction}

For ProofWriter and LogiQA, the generation instruction is:

Reason through the task using only the material. Then give exactly one final candidate index from 1 to {K} in the exact format <FINAL>index</FINAL>. Place this final tag after your reasoning, outside any thinking block. Do not append text after the tag.

For ARC, it is:

Reason through the question using the supplied information and relevant scientific knowledge. Then give exactly one final candidate index from 1 to {K} in the exact format <FINAL>index</FINAL>. Place this final tag after your reasoning, outside any thinking block. Do not append text after the tag.

The checkpoint’s chat template uses tokenize= True, add\_generation\_prompt=True, and enable\_thinking=True. Selected inputs are not truncated; the longest has 449 tokens, within the 20,000-token context with the largest output allowance.

An answer query concatenates original prompt and prefix token IDs with a suffix encoded using add\_special\_tokens=False: \n\n<FINAL> after a thinking closure, or \n</think>\n\n<FINAL> otherwise. Here \n denotes a newline. The prefix is not re-encoded. The reader uses the suffix’s final state rather than its auxiliary output token.

Candidate $j$ maps to head row and stored prediction $j - 1$ . For $K \leq 4 .$ , inference casts the state to FP32 and applies the first K head rows and softmax in FP32. All shared-reader policies use native logits over candidate-index tokens for the fiveoption ARC item, whose token IDs are 16–20.

Forced final completions use the same suffix and a 64-token cap, stopping at ${ < } / { \mathsf { F I N A L } } >$ with the stop string included. Parsing prepends the injected <FINAL> cue and accepts a candidate index in final tags, \boxed{...}, or a sole numeric response. Conflicting or out-of-range indices are invalid. Native parsing uses text after the last thinking closure and removes trailing chat EOS; a missing closure or valid answer is an error.

DEER trials append \n\*\*Final Answer\*\*\n and The final candidate index is \boxed to the original prompt and re-encoded history, excluding the triggering Wait. Rejection discards trial output, appends Wait to the saved history, and resumes reasoning. A trial ends at </think>; confidence is zero unless this closure is present and the output has at least two tokens. Otherwise confidence is the geometric mean of probabilities after the first token, including the closure. Acceptance requires confidence strictly above the threshold. A separate forced completion supplies the final answer after acceptance or terminal reasoning.

## A.3 Selected Reader and Control Parameters

The seed-29 reader has 10,240 FP32 parameters, fitted with AdamW, weight decay 0.01, 16 questions per batch, and norm-1 gradient clipping. Each question has equal weight across capped positions 0, 128, 256, 512, 1024. Development negative log-likelihood selects epoch 20 and learning rate $3 \times 1 0 ^ { - 4 }$ from epochs 5, 10, and 20 and rates $3 \times 1 0 ^ { - 4 }$ and $1 0 ^ { - 3 }$ . The backbone is frozen.

Four deterministic material-group folds supply out-of-fold training readouts. Each fold holds out 128 questions per task; its reader trains on the other 768 questions with seed 29, 20 epochs, and learning rate $3 \times 1 0 ^ { - 4 }$ . The full-data reader serves development and online evaluation.

The benefit gate has 2,564 inputs defined in Section 3. Per task, 384 training components fit the event models, 64 support probability-calibrated development variants, and 64 set routing thresholds. The two L2 logistic models use fitting-only standardization, $C = 0 . 0 1$ , LBFGS, 3,000 maximum iterations, tolerance $1 0 ^ { - 7 }$ , and seed 20261009. Standardization is folded into deployed coeffi-

cients.

The 0.75 routing threshold is the midpoint between descending held-out scores 96 and 97 among 128 questions. The raw-benefit threshold is 0.002275481505484662; the negative-margin threshold is −0.8598166108131409. Both use strict greater-than comparisons. The deployed benefit score uses uncalibrated logistic probabilities.

Trigger fitting samples at most two completesentence prefixes per question in each of four 256- token bands. An event receives weight $n _ { b } / ( m _ { b } n )$ where $n _ { b }$ and $m _ { b }$ are available and sampled band counts and n is the question’s available-event count. Terminal-only events are excluded; common observable positions coalesce to the latest complete prefix, with weights renormalized within question. Development retains every available sentence event.

Training-only standardization and randomized PCA retain 64 natural-state dimensions with seed 20261009. The taskless trigger uses weighted L2 logistic regression at $C ~ = ~ 0 . 0 1$ with the gate’s solver and fitting limits. Its features are the projected state and $t / 1 0 2 4$ ; its out-of-fold target is a correct answer with certainty at least 0.85. Folding the transformations gives Eq. 6. Online evaluation clips the logit to $[ - 8 0 , 8 0 ]$ , requests reads at $r _ { t } \geq 0 . 2 5$ , and applies stopping certainty 0.70, 0.85, or 0.95 separately.

## A.4 Causal Sentence Events and State Availability

The incremental detector reports each newly observable sentence boundary once. Chinese sentence punctuation ends a sentence immediately; English ., !, and ?, optionally followed by closing quotes, require following whitespace. Filters exclude decimals, ellipses, enumerations, and abbreviations. Decoding preserves special tokens and spacing, deferring incomplete characters. Boundaries align to the earliest covering original token and exclude tokens containing subsequent nonwhitespace content.

Each boundary increments the absolute sentence ordinal and replaces the pending event. The learned schedule tests s mod 4 = 0; scored events are consumed once. New events can replace a pending prefix while a submitted read stays bound to its original prefix. State waiting, completion, and cancellation follow Table 5.

## A.5 Software, Precision, and Execution Limits

Online execution uses Python 3.11.13, vLLM 0.17.1, PyTorch 2.10.0/CUDA 12.8, Transformers 4.57.6, Triton 3.6.0, and NumPy 2.2.6 on A100 80GB PCIe GPUs. The backbone uses BF16, the reader and trigger FP32, and TF32 matrix multiplication is disabled. Fitting replay uses BF16 SDPA with FP32 readout logits.

<table><tr><td>Engine field</td><td>Value</td></tr><tr><td>Tensor parallel size</td><td>1</td></tr><tr><td>Maximum model length</td><td>20,000</td></tr><tr><td>Maximum scheduled sequences</td><td>2</td></tr><tr><td>Maximum batched tokens</td><td>2,048</td></tr><tr><td>GPU memory utilization</td><td>0.30</td></tr><tr><td>Attention backend</td><td>TRITON_ATTN</td></tr><tr><td>Executor backend</td><td>uni</td></tr><tr><td>Language-model-only mode</td><td>Enabled</td></tr><tr><td>Asynchronous scheduler</td><td>Disabled</td></tr><tr><td>Eager execution</td><td>Disabled</td></tr><tr><td>CUDA graph capture sizes</td><td>1,2</td></tr><tr><td>Cross-request prefix cache</td><td>Disabled</td></tr><tr><td>Generation configuration</td><td>vllm</td></tr><tr><td>Engine and request seed</td><td>20261009</td></tr></table>

Table 4: Common online engine configuration. Generation and reading remain separate scheduled requests despite the disabled asynchronous engine scheduler.

Requests use temperature 0, top-p 1, top-k −1, zero presence and frequency penalties, and repetition penalty 1. Token IDs are returned without engine detokenization. Natural EOS can precede the policy cap. Reader requests emit one token; Answer Consistency trials and forced final completions have separate 64-token caps, and DEER permits at most ten 20-token trials. All emitted tokens count as output. The controller has a 300- second per-question limit.

One question–policy pair runs at a time, with at most one reasoning request and one answer query sharing the engine. Request-local KV state is retained, and between-policy cache resets precede timing. Appendix C defines the synchronized timing interval, repeated-run aggregation, and work counters.

## B Comparison Policy Details

Answer interfaces and online adaptations. Shared-reader policies use the answer-position reader of Section 3. Forced terminal answering and Answer Consistency trials generate completions after <FINAL>. The native 8,192-token reference generates reasoning and its final answer without a forced cue. Invalid answers count as errors.

Answer Consistency (AC) adapts Answer Convergence to candidate answers, using k = 10 consecutive matching trials at causal sentence boundaries. It pauses, queries, and resumes serially; repeated prefix processing and trial tokens are charged. Invalid or conflicting answers reset agreement. The DEER adaptation uses Wait transitions, geometric-mean trial confidence, a thinking-end requirement, and rollback after rejection. Greedy trials use threshold 0.95, at most ten 20-token trials, and a separate 64-token final-answer cap. This adaptation executes trials serially, without DEER’s parallel branch variant.

Baseline execution behavior. At caps of 1,024 and 2,048, primary DEER executions make 587 and 2,638 trial queries across 768 questions. The largest trial confidence is 0.8608, below the 0.95 acceptance threshold. Their 655 and 673 correct answers therefore come from terminal completion; development-selected thresholds are evaluated in Appendix F.1.

Answer Consistency stops early on 765/768 questions and answers 522 correctly, including 105/256 on ProofWriter. Native generation answers 574/768 correctly; 147 traces lack a thinking closure and five lack a valid final choice. In total, 151 executions reach the output cap. All 152 format failures remain in the accuracy denominator.

Matched controls. The six conditions share a 1,024-token cap, reader, and $\tau = 0 . 7 0$ and interleave within question on the same GPU. Always reason retains initial reading and gate evaluation before forcing generation. Margin gate replaces the benefit score. Fixed reads replaces the learned schedule with a gap of at least eight sentences since the last submitted read, initializing its ordinal to zero and retaining latest-prefix handling. No early stop retains intermediate queries but rejects early acceptance. Gate + terminal read retains routing and removes intermediate reads. Each comparison uses its matched complete-controller execution.

State and query interfaces. The decoding hook indexes the 2,560-dimensional natural state $h _ { t }$ by prefix length t and removes states more than 64 tokens behind the latest captured position. A sentence boundary may precede its state’s availability. An eligible event waits for that state until a newer event replaces it or generation ends. Capture, retention, and synchronization are included in pipeline time.

A query snapshots the original prompt and selected prefix token IDs, then appends the answer cue specified in Appendix A. Its distribution and one auxiliary output token remain bound to that prefix while generation continues.

Algorithm: selective reasoning with concurrent reads   
1. Read $( a _ { 0 } , p _ { 0 } , \tilde { h } _ { 0 } )$ ; return $a _ { 0 }$ if $u ( x ) \leq \theta .$   
2. Start generation with cap $B ;$ initialize an empty pending   
prefix and no answer query.   
3. On each sentence event, replace the pending prefix by   
$( z _ { 1 : t } , s )$ . Generation remains scheduled.   
4. If generation is active, the reader is idle, and a prefix is   
pending, discard it when s mod 4 $\neq 0 .$ Otherwise retain it   
until its indexed $h _ { t }$ is available or a newer event replaces it.   
5. With generation active and the reader idle, consume an   
eligible pending event once $h _ { t }$ is available. Evaluate $r _ { t }$ once;   
if $\dot { \boldsymbol { r } } _ { t } \geq \mathbf { \bar { 0 } } . 2 5$ , submit a read bound to that prefix, otherwise   
await a new event.   
6. On read completion, accept if it is terminal or $q _ { t } \geq \tau .$   
Cancel unfinished generation and return its $a _ { t } ;$ otherwise con  
tinue with the latest pending event.   
7. After EOS or budget exhaustion, finish any outstanding   
read. If no answer is accepted, request a terminal read and   
return through step 6.

Table 5: The generator and reader are separate requests to one frozen backbone. Pending prefixes coalesce; at most one answer request runs at a time. A terminal read is accepted unconditionally.

## C Timing and Computational Work

Primary costs average complete single-A100 executions on 768 questions under common model and engine settings. The timing check in Table 7 uses first executions on 48 of these questions for both accuracy and time. Table 8 supplements the mean costs with input processing, trigger calls, and latency tails.

Timing scope. The synchronized timer spans the first read through the final decision and cancellation, including routing, generation, sentence detection, state capture, triggers, and answer or trial requests. Tokenization, loading, warm-up, cache resets, and record writing are excluded. Requestlocal KV state is retained and cross-request prefix caching is disabled, so input counts include repeated processing. Overlapping generation and read durations cannot be added to reconstruct latency.

<table><tr><td>T</td><td>Acc. (%)</td><td>Time (s)</td><td>Output tok.</td><td>Queries</td></tr><tr><td>0.70</td><td>84.38</td><td>5.053</td><td>524.5</td><td>2.43</td></tr><tr><td>0.85</td><td>84.38</td><td>5.845</td><td>604.8</td><td>3.12</td></tr><tr><td>0.95</td><td>84.64</td><td>7.012</td><td>721.2</td><td>4.38</td></tr></table>

Table 6: Acceptance-threshold operating points for the primary shared-reader controller on the same 768 questions and A100. The predefined primary threshold is 0.70; all three thresholds were fixed before evaluation. Accuracy is in percent; costs are per-question means including auxiliary queries.

An in-flight read accepted after EOS adds no fallback. Natural-generation requests are separate from answer requests; generation beyond the accepted prefix and after the stop decision is included in actual output.

Uncertainty and repeated executions. Online comparisons use pointwise 95% paired materialgroup bootstrap intervals with 10,000 resamples, stratified by ProofWriter subtype, LogiQA, and ARC. Repeated timing averages three executions within each of 48 fixed questions before resampling. Repetitions reuse resident engines and contribute no additional independent accuracy subjects. Controls and timing checks retain withinseries same-GPU pairing.

## D Mechanism and Stability Details

Routing by task. The matched direct and always-reason endpoints answer 541 and 657 questions correctly: 147 corrections, 31 regressions, 510 correct under both, and 80 wrong under both. Replaying each gate’s choices over these endpoints reproduces its 768 deployed answers in Table 12. These diagnostic endpoints differ from the terminal-1,024 endpoint used for gate training.

Benefit alone selects 160 questions: two corrections, one regression, and 157 unchanged outcomes. Margin alone selects 80: ten corrections, six regressions, and 64 unchanged outcomes. On ARC, 107 of the 108 questions uniquely selected by benefit have unchanged correctness, locating its excess allocation.

Reading opportunities. Post-hoc inspection matches 1,077 of 4,229 rejected trigger positions to exact-prefix reads from the fixed schedule. Among these matches, 121 questions have a correct rejected opportunity with certainty at least 0.70; 70 fall below the 0.85 training target. These observations motivate target alignment and bounded waiting.

<table><tr><td>Method</td><td>Acc. (%) [95% CI]</td><td>Mean s [95% CI]</td><td>P95 s</td></tr><tr><td>Direct read</td><td>68.75 [56.25, 79.17]</td><td>0.08 [0.08, 0.08]</td><td>0.09</td></tr><tr><td>Terminal read (512)</td><td>77.08 [66.67, 87.50]</td><td>4.77 [4.76, 4.79]</td><td>4.85</td></tr><tr><td>Terminal read (1,024)</td><td>87.50 [79.17, 95.83]</td><td>9.41 [9.40, 9.43]</td><td>9.53</td></tr><tr><td>Terminal read (2,048)</td><td>87.50 [79.17, 95.83]</td><td>17.62 [16.98, 18.18]</td><td>19.08</td></tr><tr><td>Predefined simple controller</td><td>85.42 [75.00, 93.75]</td><td>5.50 [4.47, 6.54]</td><td>10.19</td></tr><tr><td>Ours (τ = 0.70)</td><td>89.58 [81.25, 95.83]</td><td>4.70 [3.84, 5.50]</td><td>9.87</td></tr><tr><td>Ours (τ = 0.85)</td><td>89.58 [81.25, 95.83]</td><td>5.55 [4.62, 6.44]</td><td>9.93</td></tr><tr><td>Ours (τ = 0.95)</td><td>89.58 [81.25, 95.83]</td><td>7.55 [6.42, 8.61]</td><td>10.43</td></tr><tr><td>Forced terminal answer</td><td>87.50 [79.17, 95.83]</td><td>9.46 [9.43, 9.49]</td><td>9.66</td></tr><tr><td>Answer Consistency (k = 10)</td><td>66.67 [56.25, 77.08]</td><td>4.22 [3.72, 4.72]</td><td>8.45</td></tr><tr><td>DEER adaptation (1,024)</td><td>83.33 [72.92, 91.67]</td><td>9.68 [9.60, 9.76]</td><td>10.18</td></tr><tr><td>DEER adaptation (2,048)</td><td>87.50 [79.17, 95.83]</td><td>18.98 [18.17, 19.75]</td><td>22.64</td></tr><tr><td>Native answer (8,192)</td><td>77.08 [66.67, 87.50]</td><td>33.96 [28.41, 39.89]</td><td>76.16</td></tr></table>

Table 7: First-execution accuracy and timing on 48 preselected questions, 16 per task, with paired policy executions. Parenthesized numbers identify the policy caps. Pointwise 95% bootstrap intervals quantify question sampling; three-run timing is analyzed separately.
<table><tr><td>Method</td><td>P95 s Input tok.</td><td></td><td>Triggers</td></tr><tr><td>Direct read</td><td>0.09</td><td>256</td><td>0.00</td></tr><tr><td>Terminal read (512)</td><td>4.81</td><td>1,019</td><td>0.00</td></tr><tr><td>Terminal read (1,024)</td><td>9.48</td><td>1,528</td><td>0.00</td></tr><tr><td>Terminal read (2,048)</td><td>18.99</td><td>2,393</td><td>0.00</td></tr><tr><td>Predefined simple controller</td><td>10.27</td><td>3,968</td><td>0.00</td></tr><tr><td>Ours (τ = 0.70)</td><td>9.92</td><td>1,673</td><td>6.73</td></tr><tr><td>Ours (τ = 0.85)</td><td>10.15</td><td>2,348</td><td>7.76</td></tr><tr><td>Ours (τ = 0.95)</td><td>10.40</td><td>3,613</td><td>9.27</td></tr><tr><td>Forced terminal answer</td><td>9.82</td><td>1,528</td><td>0.00</td></tr><tr><td>Answer Consistency (k = 10)</td><td>9.05</td><td>10,506</td><td>0.00</td></tr><tr><td>DEER adaptation (1,024)</td><td>10.21</td><td>3,093</td><td>0.00</td></tr><tr><td>DEER adaptation (2,048)</td><td>22.72</td><td>13,546</td><td>0.00</td></tr><tr><td>Native answer (8,192)</td><td>76.34</td><td>250</td><td>0.00</td></tr></table>

Table 8: Tail latency and observation work on all 768 questions per method. P95 is the 95th percentile of pipeline seconds. Input tokens and trigger evaluations are per-question means; inputs include repeated prefix processing. Parenthesized numbers identify the policy caps.

zero.

Early-exit errors. Among 369 cancelled questions, 25 exits are wrong (6.78%), but continued reasoning corrects only two (0.54% of cancellations; 0.26% of all 768 questions). Those two ARC executions stop at 104 and 237 tokens, compared with 1,024 and 972 without early stopping. The other 23 erroneous exits remain wrong after continuation.

Observation and cancellation work. The matched controller’s initial query averages 0.0828 seconds; subsequent read-response durations sum to 0.1095 seconds per question while overlapping generation. Mean input processing is 1,666.18 tokens. Generation beyond an accepted prefix averages 0.961 tokens over all 768 questions; recorded tokens after the stop decision average

Repeated and independent executions. All 13 primary configurations and six controls preserve answers and routing across their three executions on fixed 48-question subsets; one DEER-2,048 trace changes. Repeated timing gives a 45.48% saving from early stopping, with a 95% interval of 38.59–52.52%, compared with the full-cohort estimate of 34.09%.

The primary and matched controller executions answer 648 and 652 questions correctly. Their 35 final-answer, 11 routing, and 457 generationsequence differences reflect independent runs. The margin/every-eight policy likewise answers 638 questions correctly in the factorial series and 648 in the calibrated series, despite the same policy and fitted components. Benefit-minus-margin correct counts differ by −3 in the matched comparison and +2 in the factorial comparison. Each reported contrast therefore uses its own local paired executions.

## E Follow-up on Reading Opportunities

We evaluate target alignment, a reading deadline, and their combination. Alignment refits the logistic trigger on the original training prefixes using $\mathbb { I } [ a _ { t } = y \wedge q _ { t } \geq 0 . 7 0 ]$ , with the original projection and $C = 0 . 0 1$ . The deadline requests the latest complete prefix after the first eight sentences, or a gap of 16 sentences or 256 tokens since the last read. The reader, gate, 1,024-token cap, trigger threshold 0.25, and stopping threshold 0.70 remain fixed.

Table 13 evaluates accuracy on the already observed 768 questions. The original and fixed schedules reuse their matched-control runs; the three revisions are new executions. All five schedules are rerun in interleaved order on the same GPU for repeated timing on 48 fixed questions, three times per question.

<table><tr><td>Policy</td><td>Correct</td><td>Routed (%)</td><td>Time (s)</td><td>Output tok.</td><td>Reads</td><td>Input tok.</td><td>Triggers</td></tr><tr><td>Ours</td><td>652</td><td>75.13</td><td>4.960</td><td>516.4</td><td>2.43</td><td>1,666.2</td><td>6.66</td></tr><tr><td>No early stop</td><td>654</td><td>75.13</td><td>7.524</td><td>773.3</td><td>5.25</td><td>4,514.2</td><td>10.02</td></tr><tr><td>Gate + terminal read</td><td>645</td><td>75.13</td><td>7.107</td><td>770.0</td><td>1.75</td><td>1,405.2</td><td>0.00</td></tr><tr><td>Always reason</td><td>657</td><td>100.00</td><td>6.102</td><td>636.5</td><td>2.82</td><td>2,005.8</td><td>8.09</td></tr><tr><td>Margin gate</td><td>655</td><td>64.71</td><td>4.838</td><td>503.6</td><td>2.43</td><td>1,686.9</td><td>6.52</td></tr><tr><td>Fixed reads</td><td>643</td><td>75.13</td><td>4.554</td><td>462.6</td><td>4.11</td><td>2,894.4</td><td>0.00</td></tr></table>

Table 9: Full matched component controls, including the margin-gate and fixed-reading replacements. Correct counts are out of 768; routed denotes the reasoning invocation rate. Time is mean pipeline seconds and Reads counts answer queries. Work metrics are per-question means. Inputs include repeated prefix processing; triggers count actual evaluations of the learned reading score.
<table><tr><td>Counter</td><td>Definition for one complete-controller ex- ecution</td></tr><tr><td>Trigger calls</td><td>Actual evaluations of rt on eligible events with an available state. Sentence events and waiting steps do not increment this</td></tr><tr><td>Answer</td><td>counter.  $N _ { \mathrm { r e a d } } = 1 + N _ { \mathrm { m i d } } + I _ { \mathrm { f b } } ;$  initial read,</td></tr><tr><td>requests</td><td>all submitted intermediate reads, and an optional terminal fallback.</td></tr><tr><td>Input tokens</td><td>The generation prompt, if invoked, plus every answer-query input, including re- peated prefixes and query suffixes.</td></tr><tr><td>Output tokens</td><td> $| z _ { 1 : c } | + N _ { \mathrm { r e a d } } .$  where c is the actual natu- ral output length and each read emits one auxiliary token.</td></tr></table>

Table 10: Computational-work counters with crossrequest prefix caching disabled. $I _ { \mathrm { f b } }$ indicates a terminal fallback and $N _ { \mathrm { m i d } }$ counts intermediate reads. Token counts describe work separately from measured latency.

The original trigger leaves 104/577 routed questions without an intermediate read; alignment reduces this count to 30 and both deadline variants to zero. Among questions receiving a read, its first position averages 397 tokens for the original, 173 for fixed reads, 284 for alignment, 163 for the deadline, and 133 for their combination. These conditional populations differ; the revisions route 579 questions, compared with 577 in the original controls.

The combined revision and fixed reads each answer 643 questions correctly, with 17 corrections and 17 regressions between predictions and a 95% accuracy-difference interval of [−1.43, +1.43] points. The revision uses 437.5 rather than 462.6 output tokens and 4.27 rather than 4.11 reads; repeated time falls by 9.56% (95% interval: 2.61–

16.80%). Against the original trigger, it saves 15.07% of repeated-subset time while full-cohort accuracy changes by −1.17 points (95% interval: [−2.73, +0.26]). Earlier acquisition changes both cost and the answers returned.

## F Supplementary Policy Comparisons

These supplementary comparisons retain the fitted backbone, reader, and controller weights and reuse the observed three-task cohort.

## F.1 Development-Selected Operating Points

The primary simple controller at cap 1,024 and τ = 0.85 reaches 81.64% on ProofWriter, 79.30% on LogiQA 2.0, 94.53% on ARC-Challenge, and 85.16% overall, with mean time 5.834 seconds, 591.2 output tokens, and 5.10 queries.

Operating-point selection uses 48 development questions, 16 per task and four per ProofWriter subtype. ProofWriter and LogiQA questions come from reader development; ARC questions come from official validation after excluding normalized groups appearing in official test data. All development IDs and recorded material groups are disjoint from evaluation.

At acceptance thresholds 0.70, 0.85, and 0.95, the simple margin controller selects reading gaps from four, eight, or sixteen sentences by development correct count, breaking ties by latency. Selected gaps are eight, eight, and sixteen. DEER tests confidence thresholds 0.80, 0.85, 0.90, and 0.95; both the fastest and highest-accuracy development points with an early exit select 0.80. Retaining the learned thresholds, default DEER, and forced terminal answering gives nine evaluation policies.

Each question’s nine policies run on one GPU under common model and engine settings. Repeated timing averages three same-GPU executions within each of 48 preselected questions.

<table><tr><td>Comparison</td><td>Metric</td><td>Estimate</td><td>95% confidence interval</td></tr><tr><td colspan="4">Primary executions: predefined controller versus shared-reader baselines</td></tr><tr><td>Ours versus direct read</td><td>Accuracy difference (pp)</td><td>+13.41</td><td> $+ 1 0 . 4 2 \ \mathrm { t o } + 1 6 . 2 8$ </td></tr><tr><td>Ours versus terminal-512</td><td>Accuracy difference (pp)</td><td>+7.29</td><td>+4.69 to +9.90</td></tr><tr><td>Ours versus terminal-1,024</td><td>Accuracy difference (pp)</td><td>-1.17</td><td>−2.60 to +0.26</td></tr><tr><td>Ours versus terminal-1,024</td><td>Time saved (%)</td><td>46.30</td><td>43.84 to 48.79</td></tr><tr><td colspan="4">Matched executions: local complete-controller and component controls</td></tr><tr><td>Complete versus no early stop</td><td>Time saved (%)</td><td>34.09</td><td>31.88 to 36.28</td></tr><tr><td>Benefit versus margin gate</td><td>Accuracy difference (pp)</td><td>-0.39</td><td>-1.56 to +0.78</td></tr><tr><td>Benefit versus margin gate</td><td>Time difference (s)</td><td>+0.121</td><td>-0.096 to +0.332</td></tr><tr><td>Learned versus fixed reads</td><td>Accuracy difference (pp)</td><td>+1.17</td><td>-0.39 to +2.73</td></tr><tr><td>Learned versus fixed reads</td><td>Time difference (s)</td><td>+0.406</td><td>+0.275 to +0.539</td></tr></table>

Table 11: Paired comparisons from the first executions on all 768 questions. The primary controller answers 648 questions correctly; the independently executed matched complete controller answers 652. Each comparison pairs executions on the same A100 within its series. Accuracy and time differences subtract the second policy from the first; pp denotes percentage points. Positive time savings indicate the reduction relative to the second policy. Intervals use the paired material-group bootstrap protocol in Appendix C.
<table><tr><td></td><td colspan="2">Benefit gate</td><td colspan="2">Margin gate</td></tr><tr><td>Task</td><td>Calls</td><td>Correct</td><td>Calls</td><td>Correct</td></tr><tr><td>ProofWriter</td><td>240</td><td>211</td><td>235</td><td>210</td></tr><tr><td>LogiQA 2.0</td><td>162</td><td>199</td><td>179</td><td>202</td></tr><tr><td>ARC-Challenge</td><td>175</td><td>242</td><td>83</td><td>243</td></tr><tr><td>All</td><td>577</td><td>652</td><td>497</td><td>655</td></tr></table>

Table 12: Matched gate replacement on 256 questions per task, with the same GPU, downstream schedule, and stopping rule. Calls count reasoning invocations; correct counts include direct and reasoning-assisted answers.

At τ = 0.70, learned and simple controllers both answer 648/768 correctly. Learned control reduces reads by 42.75%, from 4.264 to 2.441, and input processing from 3,087 to 1,685 tokens, while output rises by 8.66%, from 481.3 to 523.0. Their 95% accuracy-difference interval is [−1.69, +1.69] points. Simple-minus-learned repeated time differs by −0.142 seconds (95% interval: [−0.997, +0.800]), leaving no resolved latency advantage.

DEER at 0.80 exits early on eleven questions, compared with zero at 0.95; both answer 649/768 correctly in this series. Its repeated time exceeds learned 0.70 by 5.431 seconds (95% interval: 4.702–6.200). Across 432 policy–question repetition groups, one changes its final answer and 54 change generated sequences.

## F.2 Gate and Reading Factorial

Four combinations cross benefit versus margin routing with learned versus every-eight-sentence reading, using the same reader and acceptance 0.70. All four interleave within question on the same GPU. Benefit obtains two additional correct answers under either reading schedule: +0.26 points, with 95% intervals [−0.91, +1.43] under learned reading and [−0.78, +1.30] under fixed reading. Neither time interval resolves a difference.

With benefit routing fixed, learned reading obtains thirteen more correct answers and 40.24% fewer reads, but time rises by 0.406 seconds (95% interval: 0.277–0.539) and generation by 55.54 tokens. Margin routing shows the same direction of observation and generation changes. These factorial executions remain separate from primary and calibrated results.

## F.3 Serial and Concurrent Execution

The controller executes on 48 questions with three same-GPU runs per mode. Serial reading aborts generation, reads the selected prefix, and resumes from the exact token prefix after rejection, charging each restored prefill. Concurrent reading keeps generation active while processing the snapshot.

Serial and concurrent execution answer 40/48 and 41/48 correctly on the first run. Their repeated serial-minus-concurrent time difference is −0.103 seconds (95% interval: [−0.277, +0.055]). Serial input work rises by 21.54%, with thirty restores across fifteen questions. Final sequences differ on 35/48 questions and the answer on one, so the comparison includes prefix-recovery and trajectory effects.

<table><tr><td>Schedule</td><td>Correct/768</td><td>Output tok.</td><td>Queries</td><td>Time saved (%) [95% CI]</td></tr><tr><td>Original trigger</td><td>652</td><td>516.4</td><td>2.43</td><td>Reference</td></tr><tr><td>Fixed every 8 sentences</td><td>643</td><td>462.6</td><td>4.11</td><td>6.10 [-1.76, 13.76]</td></tr><tr><td>Aligned target</td><td>642</td><td>456.4</td><td>3.52</td><td>13.56 [7.20, 20.56]</td></tr><tr><td>Read deadline</td><td>646</td><td>464.6</td><td>3.62</td><td>11.62 [5.90, 17.38]</td></tr><tr><td>Aligned target + deadline</td><td>643</td><td>437.5</td><td>4.27</td><td>15.07 [7.80, 22.78]</td></tr></table>

Table 13: Post-diagnostic schedule revisions on the observed cohort. Correct counts and work use 768 questions; the two original schedules reuse the matched-control runs. Time savings use a fresh, interleaved five-policy comparison on 48 questions, averaging three executions per question before paired bootstrap. All schedules retain the start gate, 1,024-token cap, and stopping threshold 0.70. These revisions are separate from the primary policies in Table 1.

<table><tr><td colspan="6">(a) Accuracy and work: 768 questions</td></tr><tr><td>Method</td><td>Stop criterion</td><td>Correct/768</td><td>∆ acc. (pp) [95% CI]</td><td>Output tok.</td><td>Reads</td></tr><tr><td>Benefit gate + learned</td><td> $q _ { t } \geq 0 . 7 0$ </td><td>648</td><td>Reference</td><td>523.0</td><td>2.44</td></tr><tr><td>Benefit gate + learned</td><td> $q _ { t } \geq 0 . 8 5$ </td><td>648</td><td>0.00 [-0.39, 0.39]</td><td>597.5</td><td>3.09</td></tr><tr><td>Benefit gate + learned</td><td> $q _ { t } \geq 0 . 9 5$ </td><td>650</td><td>0.26 [-0.26, 0.78]</td><td>709.6</td><td>4.33</td></tr><tr><td>Margin gate + every 8</td><td> $q _ { t } \geq 0 . 7 0$ </td><td>648</td><td>0.00 [-1.69, 1.69]</td><td>481.3</td><td>4.26</td></tr><tr><td>Margin gate + every 8</td><td> $\bar { q } _ { t } \ge 0 . 8 5$ </td><td>652</td><td>0.52 [-1.17, 2.34]</td><td>587.0</td><td>5.09</td></tr><tr><td>Margin gate + every 16</td><td> $q _ { t } \geq 0 . 9 5$ </td><td>648</td><td>0.00 [-1.69, 1.82]</td><td>656.1</td><td>3.49</td></tr><tr><td>DEER</td><td> $\tau _ { \mathrm { t r i a l } } = 0 . 8 0$ </td><td>649</td><td>0.13 [-1.69, 2.08]</td><td>1021.9</td><td>1.74</td></tr><tr><td>DEER</td><td> $\tau _ { \mathrm { t r i a l } } = 0 . 9 5$ </td><td>649</td><td>0.13 [-1.69, 2.08]</td><td>1026.4</td><td>1.74</td></tr><tr><td>Forced terminal answer</td><td>Terminal</td><td>645</td><td>-0.39 [-2.34, 1.56]</td><td>1026.2</td><td>1.00</td></tr></table>

<table><tr><td colspan="4">(b) Repeated timing: 48 questions</td></tr><tr><td>Method</td><td>Stop criterion</td><td>Mean s</td><td>∆ mean s [95% CI]</td></tr><tr><td>Benefit gate + learned</td><td> $q _ { t } \geq 0 . 7 0$ </td><td>4.123</td><td>Reference</td></tr><tr><td>Benefit gate + learned</td><td> $q _ { t } \geq 0 . 8 5$ </td><td>5.188</td><td>1.065 [0.542, 1.653]</td></tr><tr><td>Benefit gate + learned</td><td> $q _ { t } \geq 0 . 9 5$ </td><td>7.123</td><td>3.000 [2.253, 3.724]</td></tr><tr><td>Margin gate + every 8</td><td> $q _ { t } \geq 0 . 7 0$ </td><td>3.981</td><td>-0.142 [-0.997, 0.800]</td></tr><tr><td>Margin gate + every 8</td><td> $q _ { t } \geq 0 . 8 5$ </td><td>5.411</td><td>1.288 [0.193, 2.437]</td></tr><tr><td>Margin gate + every 16</td><td> $q _ { t } \geq 0 . 9 5$ </td><td>6.227</td><td>2.104 [0.921, 3.303]</td></tr><tr><td>DEER</td><td> $\tau _ { \mathrm { t r i a l } } = 0 . 8 0$ </td><td>9.554</td><td>5.431 [4.702, 6.200]</td></tr><tr><td>DEER</td><td> $\tau _ { \mathrm { t r i a l } } = 0 . 9 5$ </td><td>9.553</td><td>5.430 [4.699, 6.200]</td></tr><tr><td>Forced terminal answer</td><td>Terminal</td><td>9.396</td><td>5.273 [4.570, 6.024]</td></tr></table>

Table 14: Stopping-threshold comparisons reusing the 768 questions in Table 1. Benefit and margin denote the start gates; learned reads use the natural-state trigger, and every 8 or 16 denotes the sentence interval. All reasoning caps are 1,024 tokens. $q _ { t } = 1 - H ( p _ { t } ) / \log K$ measures answer-distribution concentration; $\mathrm { D E E R } ^ { \prime } \mathrm { s } \ \tau _ { \mathrm { t r i a l } }$ is its separate trial threshold. Panel (a) uses the first execution on all 768 questions for accuracy and per-question work, including auxiliary outputs. Panel (b) uses 48 preselected questions, 16 per task, on one A100, averaging three executions within each question for time. The two panels have different denominators. Differences are method minus benefit gate with learned reads at $q _ { t } \geq 0 . 7 0$ within the respective cohorts. Intervals are pointwise 95% paired stratified material-group bootstrap intervals.

## G Natural-State Information beyond Position

Saved trajectories from 1,024 training and 256 development questions in ProofWriter and LogiQA 2.0 compare the frozen taskless trigger with a position-only logistic model and a 64-token lookup. Both controls fit training data using the original out-of-fold target $\mathbb { I } [ a _ { t } = y \wedge q _ { t } \geq 0 . 8 5 ]$ and question weights. Logistic regression uses $t / 1 0 2 4 , C = 0 . 0 1$ , and the trigger’s fitting limits; lookup smooths weighted bin mass as $( m _ { + } +$

1)/(m + 2).

The cache contains 8,189 sampled training events and 14,012 complete development events. Training sample indices are not sentence ordinals. Development ordinals come from the complete event list before coalescing observable positions; excluding terminal-only events and retaining s mod $\textrm { \scriptsize 4 } = \textrm { \scriptsize 0 }$ gives 3,407 candidates across 256 questions. Pooled weights normalize within question; 95% intervals use 1,000 paired material-group bootstrap draws within task. Within-question analysis was added after inspect-

<table><tr><td colspan="7">(a) Gate and read-schedule combinations: 768 questions</td></tr><tr><td>ID</td><td>Start gate</td><td>Read schedule</td><td>Correct/768</td><td>Mean s</td><td>Output tok.</td><td>Reads</td></tr><tr><td>A</td><td>Benefit</td><td>Learned</td><td>653</td><td>5.037</td><td>525.4</td><td>2.50</td></tr><tr><td>B</td><td>Benefit</td><td>Every 8 sentences</td><td>640</td><td>4.631</td><td>471.6</td><td>4.18</td></tr><tr><td>C</td><td>Margin</td><td>Learned</td><td>651</td><td>4.949</td><td>516.4</td><td>2.50</td></tr><tr><td>D</td><td>Margin</td><td>Every 8 sentences</td><td>638</td><td>4.779</td><td>487.1</td><td>4.30</td></tr></table>

(b) Paired differences with 95% confidence intervals
<table><tr><td>Contrast</td><td>∆ acc. (pp)</td><td>∆ reads</td><td> $\Delta$  mean s 768 questions</td><td> $\Delta$  mean s 48 questions × 3</td></tr><tr><td> $A - B$ </td><td>1.69 [0.13, 3.39]</td><td>-1.681 [-1.823, -1.539]</td><td>0.406 [0.277, 0.539]</td><td>0.295 [-0.080, 0.689]</td></tr><tr><td> $A - C$ </td><td>0.26 [-0.91, 1.43]</td><td>0.000 [-0.083, 0.078]</td><td>0.088 [-0.128, 0.300]</td><td>0.490 [-0.244, 1.151]</td></tr><tr><td> $A - D$ </td><td>1.95 [0.13, 3.91]</td><td>-1.805 [-1.977, -1.637]</td><td>0.258 [0.019, 0.488]</td><td>0.689 [-0.096, 1.416]</td></tr><tr><td> $B - C$ </td><td>-1.43 [-3.39, 0.52]</td><td>1.681 [1.517, 1.842]</td><td>-0.318 [-0.534, -0.114]</td><td>0.195 [-0.592, 0.916]</td></tr><tr><td> $B - D$ </td><td> $0 . 2 6 \ [ - 0 . 7 8 , 1 . 3 0 ]$ </td><td>-0.124 [-0.250, -0.005]</td><td>-0.148 [-0.335, 0.026]</td><td>0.393 [-0.276, 0.994]</td></tr><tr><td> $C - D$ </td><td>1.69 [0.00, 3.39]</td><td>-1.805 [-1.949, -1.661]</td><td>0.169 [0.051, 0.287]</td><td>0.199 [-0.157, 0.584]</td></tr></table>

Table 15: Factorial comparison of start gates and read schedules at $q _ { t } \geq 0 . 7 0$ and a 1,024-token reasoning cap. The four policies run on the same 768 questions from Table 1 and physical A100 with per-question interleaving. Panel (a) reports first-execution correct counts and per-question means; output tokens include auxiliary requests. Panel (b) pairs the 768 first executions for accuracy, reads, and time, and also reports timing on 48 preselected questions, 16 per task, averaging three executions within each question before pairing. Each contrast subtracts the second policy from the first. Intervals are pointwise 95% paired stratified material-group bootstrap intervals.

<table><tr><td colspan="4">(a) First execution: 48 questions</td></tr><tr><td>Execution</td><td>Acc. (%) [95% CI]</td><td>Mean s</td><td>Input tok.</td></tr><tr><td>Concurrent</td><td>85.42 [75.00, 93.75]</td><td>4.792</td><td>1664.2</td></tr><tr><td>Serial</td><td>83.33 [72.92, 91.67]</td><td>4.691</td><td>2022.7</td></tr></table>

<table><tr><td colspan="2">(b) Serial minus concurrent</td></tr><tr><td>Metric</td><td>Difference [95% CI]</td></tr><tr><td>Accuracy (pp)</td><td>-2.08 [-6.25, 0.00]</td></tr><tr><td>Time (s), first run</td><td>-0.101 [-0.276, 0.056]</td></tr><tr><td>Time (s), three-run mean</td><td>-0.103 [-0.277, 0.055]</td></tr><tr><td>Input tokens</td><td>358.5 [16.4, 695.5]</td></tr><tr><td>Reads</td><td>-0.062 [-0.312, 0.104]</td></tr></table>

Table 16: Serial and concurrent execution of the same controller on 48 preselected questions, 16 per task, with cap 1,024 and $\tau = 0 . 7 0$ . Panel (a) reports firstexecution accuracy and mean costs. Panel (b) reports serial-minus-concurrent differences; the three-run time averages executions within question. Input tokens include serial prefix restoration. Intervals are pointwise 95% stratified material-group bootstrap intervals, paired for differences.

ing pooled results.

At target 0.85, pooled AUROC is 0.805 for natural states, 0.671 for position logistic, and 0.665 for lookup. State-minus-lookup AUROC differs by 0.141 (95% interval: 0.100–0.178) and log loss by −0.077 ([−0.096, −0.059]). Gains occur within both tasks. Conditioning all sentence events on 64-token bands gives AUROC 0.788 for states and 0.500 for lookup, supporting predictive information beyond position on this cache.

Within-question AUROC favors the position controls. Among 110 questions with both target classes, mean AUROC is 0.746 for states, 0.793 for position logistic, and 0.782 for lookup. Stateminus-logistic differs by −0.047 (95% interval: $[ - 0 . 0 9 6 9 , - 0 . 0 0 0 1 ] )$ and state-minus-lookup by $- 0 . 0 3 7 \ ( [ - 0 . 0 8 8 , + 0 . 0 1 4 ] )$ . Evaluation at target 0.70 gives the same ordering. Pooled prediction and within-question temporal ranking measure different properties of the reading signal.

Fixed-trajectory reading outcomes. At stopping certainty 0.85 and trigger threshold 0.25, states and lookup both answer 203/256 correctly. State screening uses 2.543 rather than 3.465 reads, a difference of −0.922 (95% interval: $[ - 1 . 1 9 5 , - 0 . 6 3 3 ] )$ , and 851.94 rather than 947.66 ideal tokens, a difference of −95.72 $( [ - 1 2 1 . 5 6 , - 6 9 . 3 0 ] )$ . Position logistic requests no intermediate reads at this threshold and falls back to terminal reading. Equal thresholds yield different query budgets.

Replay assumes instantaneous decisions on fixed prefixes, without an initial gate. Sentence distributions come from RTX3080 readouts and terminal distributions from the earlier A100 replay. These ideal-token outcomes omit online scheduling, state access, and cancellation costs; they are distinct from the three-task pipeline-time measurements.

<table><tr><td>Evaluation certainty</td><td>Score</td><td>Log loss</td><td>Pooled AUROC</td><td>AP</td><td>Within-question AUROC</td></tr><tr><td>0.85</td><td>Natural state + position</td><td>0.371</td><td>0.805</td><td>0.527</td><td>0.746</td></tr><tr><td>0.85</td><td>Position logistic</td><td>0.466</td><td>0.671</td><td>0.289</td><td>0.793</td></tr><tr><td>0.85</td><td>Position lookup</td><td>0.448</td><td>0.665</td><td>0.269</td><td>0.782</td></tr><tr><td>0.70</td><td>Natural state + position</td><td>0.582</td><td>0.758</td><td>0.640</td><td>0.756</td></tr><tr><td>0.70</td><td>Position logistic</td><td>0.663</td><td>0.670</td><td>0.457</td><td>0.843</td></tr><tr><td>0.70</td><td>Position lookup</td><td>0.626</td><td>0.666</td><td>0.439</td><td>0.832</td></tr></table>

Table 17: Position-control diagnostic on 3,407 absolute modulo-four sentence checkpoints from 256 observed development questions, with 128 per task. The target is a correct answer read with entropy certainty at least the evaluation threshold. Every score is trained with the original 0.85 target; 0.70 is an additional evaluation without refitting. Pooled metrics give each question equal total weight. Within-question AUROC averages trajectories containing both target classes: 110/256 at 0.85 and 130/256 at 0.70. Position logistic is independently refitted on the original training questions; lookup uses fixed 64-token bins and training-only weighted rates. These cached events do not measure online latency.

## H Training Details

Readers. The native Qwen3.5- 4B checkpoint uses ModelScope revision f09dfc75214c5f2f4940 ae6a26c126ce93828a9b. Its weight files match Hugging Face revision 851bf6e806efd8d0a36b 00ddf55e13ccb7b8cd0a. Bias-free linear heads map 2,560 dimensions to four candidates; pooling adds a learned 2,560-dimensional query. Seeds are 17, 29, and 43; learning rates are $\{ 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ . AdamW uses weight decay 0.01, batches of 16 questions, and norm-1 gradient clipping. Training lasts 20 epochs, with selection from epochs 5, 10, and 20 by development mean negative log-likelihood. Variation is standard deviation over selected seed runs.

Component gates. The Kev gate uses standardized 2,560-dimensional decision-token states and L2 logistic regression with LBFGS, 2,000 maximum iterations, and tolerance 10−<sup>7</sup>. Mean calibration Brier score across correction and regression heads selects C = 0.001 from {0.001, 0.01, 0.1, 1}.

The native-reader gate study fixes reader seed 29 and concatenates its initial answer state with maximum probability, margin, normalized entropy, and log(1 + input length). Gain, harm, and initial-error models use C = 0.01 and 3,000 maximum iterations. Standardization uses fitting groups; global and task-conditional probability calibration and routing thresholds use separate groups.

Fixed-grid stopping pilot. Four material-group folds produce training predictions from seed-29 readers fitted on the other 768 questions for 20 epochs at $3 ~ \times ~ 1 0 ^ { - 4 }$ Stop models use standardized features and L2 logistic regression with C = 1. Features comprise candidate probabilities, maximum probability, margin, normalized entropy, budget fraction, candidate count, and performed-read history: previous confidence, confidence change, answer agreement, and whether a previous read exists.

Thresholds are {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99} and continuation prices are {0, 0.025, 0.05, 0.1, 0.2}. The joint target compares current and terminal correctness.

Sentence-trigger development. The development trigger uses out-of-fold answer distributions at sampled sentence prefixes, with equal total weight per question. Training-fitted standardization and 64-component PCA transform natural final-layer states. Logistic regression with C = 0.01 receives this projection, generated-token fraction, and a binary task indicator. Standardization and projection fold into the affine trigger. The schedule study fixes entropy certainty at 0.85 and trigger probability at 0.25.

The three-task controller refits the trigger without the task indicator. Its training target remains correctness with entropy certainty at least 0.85, while the subsequent acceptance threshold varies independently.

## I Component Study Protocols

Tasks and data separation. Component studies use separate ProofWriter and LogiQA 2.0 cohorts balanced between tasks (Tafjord et al., 2021; Liu et al., 2023). Splits separate material groups. The Kev, native-component, and reader-development studies differ in endpoints, checkpoints, and backends; the three-task primary evaluation uses a distinct reserved cohort.

Kev routing study. The benefit gate uses 1,024 training and 256 calibration questions, then 512 questions from previously unused official-training material groups excluding historical project material and the reserved native-component cohort. Kev applies its trained adapter and decision head to Qwen3.5-4B-Base; an external post-trained Qwen3.5-4B generates reasoning before a second Kev decision. Generation uses BF16 vLLM and decisions use FP32. The cap is 8,192 tokens, reached by 106/512 traces.

Native component confirmation. A separate evaluation measures frozen native-backbone readers, routing rules, and fixed-grid stopping on 768 reserved material groups, 384 per task. The first 24 complete 32-question blocks were selected from the planned 1,024 before labels were accessed, retaining all 35 configurations and paired comparisons. The remaining 256 labels were unopened. These observations subsequently informed component development.

Native configurations use one A100 with BF16 vLLM and prefix caching disabled. Native reading normalizes language-model logits over valid answer-label tokens at the cue; linear reading applies a checkpoint distinct from the development reader, computing FP32 scores rounded to BF16 before normalization. Each question–policy pair runs separate generation and reading requests in seeded shuffled order. Checks use positions 0, 128, 256, 512, and 1,024. The long baseline generates to natural completion or 8,192 tokens, then performs a native answer read.

Matched readers and unified routing. Reader development uses 1,024 training and 256 observed development questions. Original generation token IDs are replayed through frozen Qwen3.5-4B using BF16 Hugging Face SDPA and FP32 head parameters and logits. Heads share labels and checkpoints at 0, 128, 256, 512, and 1,024 tokens. Mean development negative log-likelihood selects checkpoints under the three-seed, two-learningrate search in Appendix H.

Development routing uses the same seed-29 answer-position head at 0 and 1,024 tokens. A fixed hash orders material groups within task, assigning 768 training groups to gate fitting, 128 to probability calibration, and 128 to threshold selection. Targets use out-of-fold reader predictions. The 256 development questions evaluate the complete-data reader and also inform reader selection.

Stopping and sentence scheduling. The fixedgrid pilot fixes reader seed 29 and uses four material-group folds for training readouts. All development questions enter reasoning, with checks at 128, 256, and 512 tokens and the terminal point. Sentence scheduling reuses natural 1,024-token trajectories and trains triggers on training prefixes, fixing entropy certainty at 0.85 and trigger probability at 0.25. Replay policies share candidate readouts obtained on an A100.

Joint replay crosses five initial gates and ten routing thresholds with terminal, every-eightsentence, gap-4, and tick-4 schedules on aligned development traces. Each question incurs an initial read and follows its schedule if routed. These 200 configurations examine routing–stopping interactions on the observed development set.

Cost and statistical reporting. Replay reports reasoning calls, reads, triggers, and ideal generation tokens, assuming instantaneous decisions once a prefix becomes observable. Online output additionally includes generation while reads finish and auxiliary query tokens.

The native-component evaluation measures GPU-synchronized pipeline time including generation, reads, branching, decisions, and exit. Inputs are pre-encoded; loading and between-trial bookkeeping are excluded. Pointwise 95% intervals use 10,000 paired material-group bootstrap resamples stratified by the four ProofWriter subsets and LogiQA, with no noninferiority margin. The primary evaluation protocol appears in Section 4.

## J Component Studies

## J.1 Measured Component Trade-offs

On the separate native-component cohort, the trained linear reader raises correctness from 588 to 607/768 at a 1,024-token budget and approximately 9.44 seconds per question (Table 18). It corrects 44 answers and introduces 25 errors, gaining 2.47 percentage points (95% CI: [0.39, 4.56]). ProofWriter gains 24 correct answers while LogiQA loses five.

Initial confidence routing at 0.8 answers 236 questions directly and sends 532 to reasoning, yielding 605 correct in 6.6186 seconds. Relative to always reasoning with the linear reader, time decreases by 29.92% (95% CI: [27.07, 32.74]), while accuracy changes by −0.26 points ([−1.17, 0.65]).

<table><tr><td>Policy</td><td>Acc. (%)</td><td>PW</td><td>LQ</td><td>Time (s)</td><td>Output tok.</td></tr><tr><td>Native read, no reasoning</td><td>49.74</td><td>115</td><td>267</td><td>0.0805</td><td>1.00</td></tr><tr><td>Linear read, no reasoning</td><td>65.76</td><td>252</td><td>253</td><td>0.0803</td><td>1.00</td></tr><tr><td>Native read, always reason</td><td>76.56</td><td>309</td><td>279</td><td>9.4488</td><td>1024.68</td></tr><tr><td>Linear read, always reason</td><td>79.04</td><td>333</td><td>274</td><td>9.4440</td><td>1024.68</td></tr><tr><td>Linear, initial confidence 0.8</td><td>78.78</td><td>335</td><td>270</td><td>6.6186</td><td>710.80</td></tr><tr><td>Linear, confidence 0.8 at all checks</td><td>78.26</td><td>332</td><td>269</td><td>5.9578</td><td>613.52</td></tr><tr><td>Linear, initial 0.8 / later 0.95</td><td>78.78</td><td>334</td><td>271</td><td>6.8394</td><td>706.76</td></tr><tr><td>Linear, segmented; initial 0.8 only</td><td>78.78</td><td>334</td><td>271</td><td>6.8943</td><td>713.10</td></tr><tr><td>Linear, segmented; never stop early</td><td>78.91</td><td>332</td><td>274</td><td>9.9169</td><td>1028.86</td></tr><tr><td>Native read, long reasoning (8,192)</td><td>84.64</td><td>355</td><td>295</td><td>43.9222</td><td>4704.13</td></tr></table>

Table 18: Frozen native-backbone component comparisons on 768 reserved questions. All reasoning budgets are 1,024 tokens except the long baseline. PW and LQ give correct counts out of 384 each. Seconds and output tokens are means from actual executions; output tokens include auxiliary reading requests. Segmented controls retain the fixed-grid reading and continuation procedure. These configurations precede the sentence-trigger experiments.
<table><tr><td>Policy</td><td>Correct / 512</td><td>Reasoning calls</td><td>Generated tokens</td><td>Corrected</td><td>Regressed</td></tr><tr><td>Direct Kev</td><td>311</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Native generation, parsed answer</td><td>443</td><td>512</td><td>2,416,895</td><td></td><td></td></tr><tr><td>Always reason, then Kev</td><td>461</td><td>512</td><td>2,416,895</td><td>160</td><td>10</td></tr><tr><td>Confidence routing</td><td>450</td><td>364</td><td>1,840,852</td><td>149</td><td>10</td></tr><tr><td>Margin routing</td><td>452</td><td>371</td><td>1,863,812</td><td>151</td><td>10</td></tr><tr><td>Entropy routing</td><td>453</td><td>368</td><td>1,853,026</td><td>152</td><td>10</td></tr><tr><td>Hidden initial-error gate</td><td>452</td><td>379</td><td>1,884,419</td><td>151</td><td>10</td></tr><tr><td>Hidden benefit gate</td><td>455</td><td>367</td><td>1,743,003</td><td>150</td><td>6</td></tr><tr><td>Four-feature benefit gate</td><td>456</td><td>373</td><td>1,843,623</td><td>155</td><td>10</td></tr><tr><td>Four-feature + hidden benefit gate</td><td>456</td><td>369</td><td>1,752,851</td><td>151</td><td>6</td></tr><tr><td>Always on PW; confidence on LQ</td><td>457</td><td>376</td><td>1,867,037</td><td>156</td><td>10</td></tr><tr><td>Endpoint oracle (uses labels)</td><td>471</td><td>160</td><td>768,115</td><td>160</td><td>0</td></tr></table>

Table 19: Frozen routing on 512 new questions, balanced across tasks. Gates use the 75% calibration threshold; actual calls vary. Tokens come from paired trajectory replay capped at 8,192 per question. Corrected and regressed count changes relative to direct Kev. Parsed answers reuse the same native trajectories; 22 answers fail parsing.

Requiring 0.95 confidence at intermediate checks also yields 605 correct, but takes 6.8394 seconds; nine of the 532 reasoning trajectories stop early. It preserves every prediction from the segmented initial-only control and saves 0.80% of time (95% CI: [0.27, 1.41]), yet runs 3.33% slower than unsegmented initial routing. Lowering all thresholds to 0.8 yields 601 correct in 5.9578 seconds. Usable exit opportunities and checking overhead jointly determine the trade-off.

The long native baseline reaches 650/768 in 43.9222 seconds, with 173 traces truncated at 8,192 tokens. Its endpoint adds an answer read; parsing generated final answers instead gives 529 correct and 174 invalid formats. Longer reasoning reaches higher accuracy at substantially greater cost.

## J.2 Allocating Reasoning

The Kev study uses an external generator capped at 8,192 tokens. Reasoning followed by Kev yields 461/512 correct, compared with 311 for direct Kev, comprising 160 corrections and ten regressions (Table 19). These two events define the benefit target.

At the 75% calibration setting, the hidden benefit gate invokes reasoning 367 times and answers 455 correctly: 28.3% fewer calls and 27.9% fewer generated tokens than always reasoning, with six fewer correct answers. It captures 150/160 corrections and incurs six regressions. Against confidence routing, it gains five correct answers and saves 97,849 tokens; the paired accuracy-difference interval spans zero (95% CI: [−0.78, 2.73] points).

Correction AUROC varies by task: benefit obtains 0.762 on ProofWriter and 0.668 on LogiQA, versus confidence’s 0.597 and 0.819. Of the benefit gate’s 367 calls, 211 leave endpoint correctness unchanged and consume 56.8% of generated

<table><tr><td>Gate</td><td>Correct/256</td><td>Calls</td><td>Corrected</td><td>Regressed</td></tr><tr><td>Hidden + four-feature benefit</td><td>200</td><td>209</td><td>48</td><td>12</td></tr><tr><td>Initial-error prediction</td><td>201</td><td>185</td><td>49</td><td>12</td></tr><tr><td>Confidence</td><td>203</td><td>195</td><td>51</td><td>12</td></tr><tr><td>Margin</td><td>203</td><td>194</td><td>51</td><td>12</td></tr><tr><td>Entropy</td><td>203</td><td>197</td><td>51</td><td>12</td></tr><tr><td>Benefit, global calibration</td><td>195</td><td>193</td><td>41</td><td>10</td></tr><tr><td>Benefit, task calibration</td><td>202</td><td>201</td><td>46</td><td>8</td></tr><tr><td>Task mean benefit</td><td>193</td><td>197</td><td>40</td><td>11</td></tr></table>

Table 20: Unified native-reader routing on 256 observed development questions. All gates use the 75% thresholdcalibration setting; realized calls differ. Direct reading gives 164 correct answers, always reasoning gives 203, and the label-based endpoint oracle gives 217. Calls count reasoning invocations; each replay uses a 1,024-token cap. Corrected and regressed are relative to direct reading.
<table><tr><td>Reader</td><td>0</td><td>128</td><td>256</td><td>512</td><td>1,024</td></tr><tr><td>Answer position</td><td> $6 5 . 1 0 \pm 0 . 9 8$ </td><td> $6 5 . 8 9 \pm 1 . 1 3$ </td><td> $6 4 . 4 5 \pm 0 . 6 8$ </td><td> $7 0 . 1 8 \pm 1 . 9 3$ </td><td> $7 8 . 7 8 \pm 0 . 4 5$ </td></tr><tr><td>Natural final state</td><td> $3 9 . 8 4 \pm 2 . 0 7$ </td><td> $3 0 . 7 3 \pm 2 . 5 4$ </td><td> $3 0 . 6 0 \pm 1 . 5 8$ </td><td> $3 8 . 5 4 \pm 2 . 9 3 $ </td><td> $4 1 . 1 5 \pm 2 . 1 5$ </td></tr><tr><td>Natural mean state</td><td> $3 3 . 2 0 \pm 4 . 8 8$ </td><td> $3 4 . 2 4 \pm 5 . 6 9$ </td><td> $3 3 . 0 7 \pm 4 . 9 5$ </td><td> $3 6 . 0 7 \pm 5 . 5 6$ </td><td> $4 5 . 1 8 \pm 4 . 8 2$ </td></tr><tr><td>Natural attention pool</td><td> $3 5 . 6 8 \pm 5 . 8 8$ </td><td> $3 8 . 4 1 \pm 7 . 0 1$ </td><td> $3 8 . 4 1 \pm 7 . 3 8$ </td><td> $4 2 . 7 1 \pm 7 . 8 9$ </td><td> $5 1 . 8 2 \pm 6 . 3 3$ </td></tr></table>

Table 21: Matched heads on a frozen native backbone at the indicated reasoning-token budgets. The answerposition, final-state, and mean-state heads each have 10,240 parameters; attention pooling has 12,800. Development accuracy is reported in percent as mean ± standard deviation across three seeds. The 256 development questions also select checkpoints. Answer-position reading adds a closure and cue; natural-state interfaces use existing prefix states. Measurements use teacher-forced replay.

tokens.

Aligning the native endpoints. The nativereader development comparison uses the same reader at both endpoints, yielding 164/256 direct and 203 terminal successes, with a label-based selection upper bound of 217 (Table 20). At the 75% calibration setting, margin reaches 203 correct with 194 reasoning calls; benefit using the hidden state and four features reaches 200 with 209 calls. Global and task-conditional probability calibration reach 195 and 202. Benefit modeling does not improve on margin in this configuration.

## J.3 Extracting Answers from Prefixes

At 1,024 tokens, answer-position reading reaches 78.78% mean development accuracy across three seeds, versus 41.15%, 45.18%, and 51.82% for natural final-state, mean-state, and attention-pool heads (Table 21). Heads share labels, prefix budgets, and optimization search; answer-position reading additionally processes a closure and answer cue.

This extraction gap motivates using natural states to select reading opportunities while retaining explicit answer queries.

<table><tr><td>Schedule</td><td>Tokens</td><td>Reads</td><td>Triggers</td></tr><tr><td>Terminal only</td><td>1024.0</td><td>1.000</td><td>0.00</td></tr><tr><td>Every 4 sentences</td><td>788.2</td><td>10.719</td><td>0.00</td></tr><tr><td>Every 8 sentences</td><td>830.9</td><td>5.762</td><td>0.00</td></tr><tr><td>Trigger every sentence</td><td>779.7</td><td>6.363</td><td>41.31</td></tr><tr><td>Trigger, gap-4</td><td>813.5</td><td>3.594</td><td>32.66</td></tr><tr><td>Trigger, tick-4</td><td>855.0</td><td>2.570</td><td>11.07</td></tr><tr><td>Trigger, tick-8</td><td>888.6</td><td>1.770</td><td>5.58</td></tr></table>

Table 22: Sentence scheduling on 256 observed development questions, with the same reader and entropy stopping threshold. All schedules yield 203/256 correct answers; tokens, reads, and triggers are per-question means. Token counts assume instantaneous decisions; this is trajectory replay, not an online timing comparison. Each early-stopping row loses one terminal success and recovers one terminal error.

## J.4 Reading Frequency and Stopping

All displayed sentence schedules yield 203/256 correct (Table 22). Each early-stopping schedule uses the same entropy threshold, recovering one terminal error and losing one terminal success. Reading every four sentences uses 788.2 ideal generation tokens and 10.719 reads per question; every eight uses 830.9 tokens and 5.762 reads.

A learned gap-4 trigger uses 813.5 tokens, 3.594 reads, and 32.66 trigger calls. Tick-8 reduces reads to 1.770 and triggers to 5.58 while delaying stopping to 888.6 ideal tokens. Sparser observation exchanges fewer queries for longer generation.

<table><tr><td>Rule</td><td>T</td><td>Correct</td><td>Reads</td><td>Tokens</td></tr><tr><td>Terminal only</td><td></td><td>203</td><td>1.00</td><td>1,024.0</td></tr><tr><td>Raw confidence</td><td>.90</td><td>198</td><td>3.04</td><td>703.0</td></tr><tr><td>Raw confidence</td><td>.95</td><td>202</td><td>3.50</td><td>852.0</td></tr><tr><td>Learned correctness</td><td>.90</td><td>199</td><td>3.73</td><td>928.0</td></tr><tr><td>Learned correctness</td><td>.95</td><td>203</td><td>3.98</td><td>1,018.5</td></tr><tr><td>Joint, λ = 0</td><td>.90</td><td>203</td><td>3.98</td><td>1,017.0</td></tr><tr><td>Joint, λ = .05</td><td>.90</td><td>199</td><td>3.69</td><td>919.5</td></tr><tr><td>Joint, λ = .05</td><td>.95</td><td>202</td><td>3.96</td><td>1,009.0</td></tr></table>

Table 23: Stopping pilot on 256 observed development questions with a fixed answer-position reader. All questions enter reasoning. Tokens assume instantaneous decisions; reads count performed calls. Displayed settings were specified before stop-model fitting.

In the fixed-grid pilot, confidence at 0.95 yields 202/256 correct, 852 ideal tokens, and 3.50 reads (Table 23). The joint correctness–continuation rule at the same threshold and λ = 0.05 also yields 202, but uses 1,009 tokens and 3.96 reads. Checking confidence only at 128 and 512 preserves the dense grid’s predictions with 863 tokens and 2.656 reads, improving observation cost over the tested joint target.

Interaction with initial routing. On the same development traces, margin routing sends 194 questions to reasoning and yields 203/256 correct. Every-eight-sentence stopping reduces mean ideal generation from 776.0 to 694.3 tokens per input. Gap-4 uses 691.9 tokens and lowers total reads from 5.891 to 3.996, adding 28.32 triggers; tick-4 uses 702.8 tokens, 3.078 reads, and 9.22 triggers. Reads include the initial query, and all three schedules preserve routed questions’ terminal correctness.

## J.5 Online Component Execution

The online development study compares terminal reading, every-eight-sentence reading, and gap-4, tick-4, and tick-8 triggers on 32 observed questions with three repetitions. All questions enter reasoning; policies run in shuffled order on the same A100 for each question, with fresh generation per policy. This eager vLLM configuration enables cross-request prefix caching, unlike the compiled, cache-disabled primary evaluation.

Terminal reading yields 22/32 correct in 60.593 seconds; every-eight-sentence reading yields 23 in 48.651 seconds, gap-4 yields 25 in 48.351 seconds, and tick-4 yields 23 in 47.839 seconds. The three intermediate-reading policies change generated prefixes on 24, 16, and 13 questions, respectively. These are complete-policy comparisons with trajectory changes.