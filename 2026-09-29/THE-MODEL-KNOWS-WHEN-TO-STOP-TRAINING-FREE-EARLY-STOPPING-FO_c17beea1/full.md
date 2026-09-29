# THE MODEL KNOWS WHEN TO STOP: TRAINING-FREE EARLY STOPPING FOR LONG-CONTEXT READING

Muath Alyobi<sup>1</sup> Mohamed Eltahir<sup>1</sup> Almoayyad Abuljdail<sup>1</sup>

Riyadh Almutawa<sup>1</sup> Tanveer Hussain<sup>2‡</sup> Naeemullah Khan<sup>1§</sup>

<sup>1</sup>King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia <sup>2</sup>Department of Computer Science, Edge Hill University, Ormskirk, England

{muath.alyobi, mohamed.hamid, almoayyad.abuljdail, {riyadh.almutawa, naeemullah.khan}@kaust.edu.sa hussaint@edgehill.ac.uk

![](images/06f9f8ac78d5bfe2f6774fc124c0c609dcd4fa2b8e63e7b22570e059a21ded4f.jpg)

![](images/c6ebbdb9b9104403de3febee3a6a4557c1b152ffc2de00dc914fd9173ac46355.jpg)  
Figure 1: Evidence-aligned stopping on S-NIAH. Left: cross-model premature-stop behavior under the same shared setting across five models. Right: within-run stopping behavior relative to the evidence position. The oracle stops exactly at the evidence-containing chunk, so zero denotes the oracle stop, positive values indicate over-reading, and negative values indicate premature stopping.

## ABSTRACT

Language models often process long inputs sequentially in chunks, but continuing to read after sufficient evidence has been acquired wastes computation. Existing stopping mechanisms either learn sufficiency from internal activations or train an exit gate, while a simpler alternative asks the model whether it has read enough. We introduce Answer-Convergence Stopping (ACS), a training-free stopping rule that measures rather than asks. After each chunk, it probes the frozen model’s current answer state and stops when that state is both confident and stable. The rule requires only output-side generation and token log probabilities, has no trained components, and uses one shared configuration across models and benchmarks. Because a stopping policy can save computation simply by stopping too early, we evaluate the stopping decision itself using evidence position where available. On the full LongBench-v2 with two frontier models, ACS is the only stopping policy that matches or exceeds full-reading accuracy. Furthermore, across 250 S-NIAH questions, the premature stopping rate for ACS across five models from two families ranges from 0% to 12%, compared to 8.4% to 45.6% for the verbalized gate. Taken together, ACS reveals that by properly utilizing the output signals of frozen models, we can achieve favorable behaviors like adaptive stopping without the need for additional training.

## 1 INTRODUCTION

Language models are increasingly used for tasks involving long inputs that exceed their context window. A common strategy is to split the input into chunks and process them sequentially, carrying forward information from earlier chunks rather than presenting the full context in a single call. This pattern underlies agent-based readers (Zhang et al., 2024), memory agents trained with reinforcement learning (Yu et al., 2025; Sheng et al., 2026), and programmatic runtimes that recursively decompose long inputs (Zhang et al., 2025; Roy et al., 2026). However, incremental processing still incurs substantial cumulative cost and does not by itself determine when enough has been read. As GRU-Mem (Sheng et al., 2026) puts it, the loop “lacks an exit mechanism, leading to unnecessary computation after even sufficient evidence is collected”.

Recent work has therefore explored how to determine when enough context has been processed. Dynamic context cutoff (Xie et al., 2025) finds that specific attention heads encode sufficiency and monitors them with lightweight classifiers trained on labels derived from ground-truth answer locations. This provides a strong learned sufficiency signal, but requires access to internal activations and training a classifier from sufficiency-labeled examples. GRU-Mem (Sheng et al., 2026) adds an exit gate to a memory agent and trains it through an explicit exit reward. DCC (Xie et al., 2025) also considers a training-free alternative: directly prompting the model to judge whether the accumulated context is sufficient. Their results show a clear dependence on model scale: self-prompting improves as model size increases, while activation-based probes remain stronger across the evaluated models.

Rather than asking the model to judge whether it has read enough or probing its internal activations, we measure how its answer state evolves as more context is read. After each chunk, we probe the frozen model’s current answer state. We hypothesize that this evolving answer state is sufficiently informative to serve as a signal for when further reading is no longer necessary. We call this a measured signal, in contrast to latent signals extracted from internal activations by trained probes and asked signals obtained through self-report. ACS (Figure 2) turns this signal into a stopping rule with two criteria. The confidence criterion measures whether the belief is sufficiently committed. The stability criterion measures whether that belief has stopped changing. The second criterion is important because confidence alone can be transient. Our default evaluation uses one shared conservative configuration across models and benchmarks, while we separately study the confidence threshold as an accuracy-efficiency operating point. No stopping component is trained.

A stopping rule should be judged on where it stops, not only on the cost it saves, because a policy can save by stopping before the evidence and guessing. On needle benchmarks, the evidence position is known by construction, providing an explicit reference for whether a stopping decision occurs before or after the required evidence. We use this information only for evaluation, reporting premature-stop rate, over-read, accuracy regret, and savings capture relative to an oracle that stops at the evidence. This allows us to distinguish evidence-aligned savings from savings obtained simply by truncating the reading process early.

Figure 1 summarizes both the within-run and cross-model stopping behavior. On 250 S-NIAH instances with Qwen3-14B (Qwen Team, 2025), the right panel shows that ACS stops before the evidence on only 1.6% of questions, compared with 8.4% for the verbalized gate, while still capturing 60.7% of the savings available to an oracle that stops exactly at the evidence. The left panel shows that this behavior extends across models: premature stopping for ACS remains between 0.0% and 12.0% across five models, while the verbalized gate ranges from 8.4% to 45.6%. Together, the two views show that ACS produces evidence-aligned stopping while maintaining low premature-stop rates across models.

Contributions. (1) A training-free stopping rule for chunked reading, defined for both optionconstrained and open-ended answers, using only output-side generation and log probabilities. (2) An evidence-aware evaluation protocol that measures stopping decisions against known evidence position using premature-stop rate, over-read, oracle-normalized regret, and savings capture. (3) A cross-setting evaluation showing that the same confidence-stability formulation remains effective across model families, scales, benchmarks, and answer formats, with a shared conservative default and separately studied accuracy-efficiency operating points.

## 2 RELATED WORK

Reading in pieces. Long-context systems often process inputs incrementally, either by carrying information forward across segments or by selectively decomposing and navigating the input. Chain-of-Agents (Zhang et al., 2024) assigns successive text segments to worker agents that pass a communication unit forward, followed by a manager that produces the final answer. MemAgent (Yu et al., 2025) maintains a fixed-length memory that is overwritten after each segment and trains the memory-update policy with reinforcement learning, yielding linear processing complexity. RLM (Zhang et al., 2025) lets the model write code to inspect and decompose the input, whereas λ-RLM (Roy et al., 2026) replaces free-form control code with typed, pre-verified combinators. MemWalker (Chen et al., 2023) constructs a tree of summaries and uses prompted reasoning and navigation actions to search for relevant content, answering once sufficient information has been gathered. These methods primarily address how to make long inputs tractable through sequential processing, decomposition, or navigation. However, none takes the stopping decision itself as the central problem.

Knowing when to stop. Dynamic context cutoff (Xie et al., 2025) processes cumulative prefixes and halts when a sufficiency classifier fires. The classifier is trained on sufficiency labels derived from ground-truth answer locations and operates over cumulatively expanding prefixes containing all previously processed tokens. In DCC, self-prompting improves substantially with model scale, while activation-based probes remain stronger across the evaluated models. GRU-Mem (Sheng et al., 2026) adds a text-controlled exit gate to the MemAgent loop and trains correct exit behavior with an explicit reinforcement-learning reward. Thus, DCC obtains the stopping signal by learning to decode latent sufficiency, whereas GRU-Mem learns the exit behavior itself. Neither evaluates whether an output-side, training-free signal derived from the model’s evolving answer state can control sequential reading.

Stopping on other axes. CALM (Schuster et al., 2022) exits transformer depth, adaptive selfconsistency (Aggarwal et al., 2023) stops sampling, and FLARE (Jiang et al., 2023) triggers retrieval when predicted tokens have low confidence. Uncertainty estimators such as verbalized confidence (Xiong et al., 2024), P(True) (Kadavath et al., 2022), and semantic entropy (Kuhn et al., 2023) characterize uncertainty in model answers rather than control how much input is read. Grid-Probe (Eltahir et al., 2026) uses answer-space posterior probes over subsets of video frames to allocate test-time compute adaptively. These works show that confidence, uncertainty, and answer-space measurements can support adaptive computation along axes other than sequential input reading.

Across these lines of work, prior methods either focus on processing or navigating long inputs, learn sufficiency or exit behavior, or use confidence, uncertainty, and answer-space measurements to control other forms of computation. What these lines of work do not establish is whether the evolving answer state of a frozen model can itself serve as a training-free termination signal for sequential long-context reading. The next section develops this idea.

## 3 ANSWER-CONVERGENCE STOPPING (ACS)

## 3.1 SETUP

A question q is paired with a document D too long to read in one call. The reader splits D into chunks $C _ { 1 } , \ldots , C _ { T }$ of at most L characters in document order and performs a sequential fold. At step $t ,$ a frozen model receives $q ,$ the running notes $N _ { t - 1 }$ , and chunk $C _ { t } .$ , and generates an updated note state $N _ { t }$ . If the update exceeds the cap of B characters, only its most recent B characters are retained. We initialize $N _ { 0 }$ as empty.

Each fold step requires one note-update call. ACS then issues a separate probe on $( q , N _ { t } )$ to obtain the current answer state. A stopping policy maps the fold to a stop step $s \in \{ 1 , \ldots , T \}$ . The prediction derived from the probe at step s is used directly as the final answer; no additional answer-generation call is required. Full reading processes all chunks and returns the prediction from the final probe on $N _ { T }$

Reported token and latency costs include all calls required by each policy. Thus, ACS is charged for both the note-update and probe calls at every processed step, whereas full reading is charged for all note updates and only the final probe. Comparator policies are likewise charged for the calls required by their stopping signals. We assume that the serving stack supports output generation and token log probabilities.

![](images/7cfe58cc34db90e8d7316710f984949359c098442b41a4534776ce7b263120db.jpg)  
Figure 2: Overview of ACS. The document is processed sequentially in chunks while a frozen model maintains running notes. After each chunk, a separate probe produces the current answer state $b _ { t }$ and confidence $c _ { t } .$ , and consecutive answer states define the change $\delta _ { t }$ . The stopping rule halts at the first step where $c _ { t } \geq \theta$ and the mean change $\Delta _ { t }$ over the last min $( t , w )$ ) answer states satisfies $\Delta _ { t } \leq \varepsilon$

## 3.2 THE MEASURED SIGNAL

After each fold step, we issue one probe call on $( q , N _ { t } )$ with a fixed template and obtain an answer state $b _ { t }$ and a confidence $c _ { t }$ . Within each answer format, the probe template is fixed across steps, models, and benchmarks.

For questions with a finite option set ${ \mathcal { A } } ,$ generation is constrained to the option letters and the answer state is the posterior over options,

$$
p _ { t } ( a ) \ = \ \frac { \exp \ell _ { t } ( a ) } { \sum _ { a ^ { \prime } \in \mathcal { A } } \exp \ell _ { t } ( a ^ { \prime } ) } , \qquad c _ { t } \ = \ \operatorname* { m a x } _ { a \in \mathcal { A } } p _ { t } ( a ) , \qquad b _ { t } = p _ { t } ,\tag{1}
$$

where $\ell _ { t } ( a )$ is the log probability assigned to the token corresponding to option a. For open-ended questions, the probe decodes a short draft answer $d _ { t }$ greedily. The answer state is the draft, $b _ { t } = d _ { t }$ and confidence is the geometric mean of its generated-token probabilities,

$$
\begin{array} { r } { c _ { t } \ = \ \exp \Big ( \frac { 1 } { | d _ { t } | } \sum _ { i = 1 } ^ { | d _ { t } | } \log p \big ( d _ { t , i } \ | \ d _ { t , < i } , q , N _ { t } \big ) \Big ) . } \end{array}\tag{2}
$$

Empty or abstaining drafts, such as “I do not know” or “cannot determine,” are prevented from triggering a stop. For both branches, let $\delta _ { t }$ denote the change between consecutive answer states: Jensen–Shannon divergence $\mathrm { J S D } ( p _ { t - 1 } , p _ { t } )$ for posteriors, and $1 - \operatorname { F 1 } ( d _ { t - 1 } , d _ { t } )$ for drafts, where token F1 is computed over normalized token multisets. Both are 0 for identical answer states. With natural-log JSD, the MCQ change lies in [0, ln 2], while the token-F1 change lies in [0, 1] and equals 1 when the drafts share no tokens.

## 3.3 THE STOPPING RULE

ACS halts at the first step that passes both a confidence test and a stability test. It has three constants: a confidence threshold θ, a stability tolerance ε, and a window size w. At step $t \geq 2$ , the stability statistic is the mean change over the last $m _ { t } = \operatorname* { m i n } ( t , w )$ answer states,

$$
\Delta _ { t } = \frac { 1 } { m _ { t } - 1 } \sum _ { i = t - m _ { t } + 2 } ^ { t } \delta _ { i } .\tag{3}
$$

For open-ended questions, let $A _ { t } = 1$ when the probe produces a non-empty, non-abstaining draft and $A _ { t } = 0$ otherwise, and let $A _ { t } = 1$ for option-constrained questions. The stopping step is

$$
s = \operatorname* { m i n } \left\{ t \in \left\{ 2 , \dots , T \right\} : c _ { t } \geq \theta \wedge \Delta _ { t } \leq \varepsilon \wedge A _ { t } = 1 \right\} ,\tag{4}
$$

with $s = T$ if no step qualifies, so the document is read in full.

The returned answer is arg ma $\mathrm { x } _ { a } p _ { s } ( a )$ for option-constrained questions or the draft $d _ { s }$ for openended questions, with confidence $c _ { s }$ . Confidence alone can trigger on transient high-confidence states, while stability alone cannot distinguish a settled answer state from a settled but incorrect one. Together, the two criteria require the answer state to be both confident and temporally stable rather than merely exhibiting a transient confidence peak. The threshold θ sets how much confidence a stable answer state needs before it can trigger a stop. If a model’s probe confidence generally remains below θ, the policy approaches full reading and offers little compute saving.

## 3.4 EVALUATING THE STOPPING DECISION

On needle benchmarks, the needle contains the gold answer, so its known character offset identifies the evidence-containing chunk exactly; we use this information only for evaluation. Let e be the index of the chunk containing the evidence and s the stop step of a policy. The premature rate is the fraction of questions with $s < e .$ Over-read is the mean of $s - e$ over questions with $s \geq e$ Accuracy regret is the accuracy under oracle stopping, $s = e ,$ , minus the accuracy of the evaluated policy. Savings capture is the fraction of oracle-available savings recovered without stopping before the evidence,

$$
{ \mathrm { C a p t u r e } } = { \frac { \sum _ { i } \mathbf { 1 } [ s _ { i } \geq e _ { i } ] ( T _ { i } - s _ { i } ) } { \sum _ { i } ( T _ { i } - e _ { i } ) } } .\tag{5}
$$

Premature stops therefore contribute zero captured savings. Oracle stopping, $s = e ,$ , is the earliest evidence-aligned stop and serves as the reference for both regret and savings capture.

## 4 EXPERIMENTS

## 4.1 SETUP

Benchmarks. LongBench-v2 (Bai et al., 2025) is the full 503-question set, multiple choice across six domains, with contexts ranging up to 2M words. S-NIAH (Hsieh et al., 2024) is RULER’s single-needle retrieval task. We evaluate 250 instances, with 50 at each context length in {8, 16, 32, 64, 128}K tokens, and uniformly placed needles. Because the needle position is known exactly, S-NIAH supports our oracle stopping evaluation. RULER-HotpotQA (Hsieh et al., 2024) extends HotpotQA to long contexts by inserting its supporting paragraphs among distractors. We evaluate 100 questions at five context lengths, yielding 500 trajectories. Unlike S-NIAH, answering can require combining multiple supporting facts. BrowseComp-Plus (Chen et al., 2025) is an 830-query open-ended deep-research benchmark built around a fixed, curated web corpus.

Models. Qwen3.5-397B-A17B (Qwen Team, 2026) and Kimi K2.5 (Kimi Team, 2026) are served through the OpenRouter API with log-probability access, while Qwen3-14B (Qwen Team, 2025) is served with vLLM (Kwon et al., 2023) with thinking disabled. Qwen2.5-7B (Qwen Team, 2024), Qwen3-32B, Gemma-3-12B, and Gemma-3-27B (Gemma Team, 2025) extend the S-NIAH evaluation to five models, with 250 questions per model. L = 24,000 and B = 6,000 characters. The constants are $\varepsilon = 0 . 0 5 , w = 3 .$ and $\mathbf { \bar { \theta } = 0 . 9 9 \bar { 5 } }$ . Appendix E evaluates the robustness of this fixed configuration against model-specific cost-aware tuning on a held-out S-NIAH split. The best-threshold rows of Table 1 select the highest-accuracy θ on the prespecified grid, breaking ties by lower token cost. Because selection uses the reported evaluation questions, these rows are post-hoc points rather than part of the shared-default evaluation.

Baselines. All direct baselines use the same underlying reader, chunking, and note-update procedure as ACS; only the stopping mechanism changes. Full reading processes all T chunks and serves as the exhaustive baseline, while random stop samples the stopping step uniformly from $\{ 1 , \ldots , T \}$ and averages over 200 draws. We also compare against two asked stopping signals: a verbalized gate, which asks the model to report from 0 to 100 how confident it is that the current notes are sufficient to answer and stops at 99.5, and an END/CONTINUE gate, which directly asks whether enough information has been collected and stops when the model returns END. We restrict direct comparisons to stopping rules that can be applied to the same frozen reader without additional training. DCC and GRU-Mem require learned stopping components tied to their respective model or reader setups and are therefore discussed as related methods rather than used as direct baselines.

Table 1: Accuracy, token efficiency, and total elapsed time across benchmarks. Token and time savings are measured relative to full reading; elapsed time is the sum of per-example processing time under the corresponding serving setup. ACS, fixed uses the shared θ = 0.995, while best θ reports the highest-accuracy post-hoc operating point on the prespecified threshold grid, with ties broken by lower token cost.
<table><tr><td></td><td></td><td colspan="5">Qwen3.5-397B-A17B</td><td colspan="5">Kimi K2.5</td></tr><tr><td>Benchmark</td><td>Policy</td><td>Acc.</td><td>Tokens</td><td>Tok. save</td><td>Time</td><td>Time save</td><td>Acc.</td><td>Tokens</td><td>Tok. save</td><td>Time</td><td>Time save</td></tr><tr><td rowspan="6">LongBench-v2</td><td>full reading</td><td>0.535</td><td>382,971</td><td>0%</td><td>86.0 h</td><td>0%</td><td>0.517</td><td>340,761</td><td>0%</td><td>36.3 h</td><td>0%</td></tr><tr><td>random stop</td><td>0.474</td><td>185,632</td><td>52%</td><td>44.2 h</td><td>48.7%</td><td>0.481</td><td>168,707</td><td>50%</td><td>18.7 h</td><td>48.7%</td></tr><tr><td>verbalized gate</td><td>0.533</td><td>267,179</td><td>30.2%</td><td>54.9 h</td><td>36.2%</td><td>0.517</td><td>290,354</td><td>14.8%</td><td>27.6 h</td><td>24.1%</td></tr><tr><td>END gate</td><td>0.507</td><td>134,569</td><td>65%</td><td>28.3 h</td><td>67.1%</td><td>0.525</td><td>183,604</td><td>46%</td><td>18.1 h</td><td>50.3%</td></tr><tr><td>ACS, fixed</td><td>0.541</td><td>190,966</td><td>50%</td><td>40.4 h</td><td>53.1%</td><td>0.519</td><td>331,070</td><td>3%</td><td>31.2 h</td><td>14.1%</td></tr><tr><td>ACS, best θ</td><td>0.541</td><td>190,966</td><td>50%</td><td>40.4 h</td><td>53.1%</td><td>0.535</td><td>172,648</td><td>49%</td><td>17.2 h</td><td>52.6%</td></tr><tr><td rowspan="6">S-NIAH</td><td>full reading</td><td>1.000</td><td>57,396</td><td>0%</td><td>4.2 h</td><td>0%</td><td>0.996</td><td>54,632</td><td>0%</td><td>3.5 h</td><td>0%</td></tr><tr><td>random stop</td><td>0.675</td><td>34,098</td><td>41%</td><td>2.4 h</td><td>44.2%</td><td>0.673</td><td>32,564</td><td>40%</td><td>2.0 h</td><td>44.2%</td></tr><tr><td>verbalized gate</td><td>0.944</td><td>37,564</td><td>34.6%</td><td>2.6 h</td><td>37.8%</td><td>0.996</td><td>39,426</td><td>27.8%</td><td>2.4 h</td><td>32.5%</td></tr><tr><td>END gate</td><td>1.000</td><td>36,816</td><td>36%</td><td>2.5 h</td><td>39.7%</td><td>0.996</td><td>32,386</td><td>41%</td><td>1.9 h</td><td>46.5%</td></tr><tr><td>ACS, fixed</td><td>1.000</td><td>41,734</td><td>27%</td><td>3.0 h</td><td>28.4%</td><td>0.996</td><td>39,559</td><td>28%</td><td>2.5 h</td><td>29.6%</td></tr><tr><td>ACS, best θ</td><td>1.000</td><td>41,663</td><td>27%</td><td>3.0 h</td><td>28.4%</td><td>0.996</td><td>39,237</td><td>28%</td><td>2.5 h</td><td>30.0%</td></tr><tr><td rowspan="6">RULER-HotpotQA</td><td>full reading</td><td>0.722</td><td>65,374</td><td>0%</td><td>16.4 h</td><td>0%</td><td>0.760</td><td>55,095</td><td>0%</td><td>8.8 h</td><td>0%</td></tr><tr><td>random stop</td><td>0.548</td><td>36,357</td><td>44%</td><td>9.2 h</td><td>44.3%</td><td>0.590</td><td>31,046</td><td>44%</td><td>4.9 h</td><td>44.3%</td></tr><tr><td>verbalized gate</td><td>0.722</td><td>46,291</td><td>29.2%</td><td>10.5 h</td><td>36.4%</td><td>0.744</td><td>42,201</td><td>23.4%</td><td>6.3 h</td><td>29.1%</td></tr><tr><td>END gate</td><td>0.716</td><td>43,741</td><td>33%</td><td>9.8 h</td><td>40.2%</td><td>0.736</td><td>37,490</td><td>32%</td><td>5.5 h</td><td>37.6%</td></tr><tr><td>ACS, fixed</td><td>0.714</td><td>54,867</td><td>16%</td><td>12.8 h</td><td>22.1%</td><td>0.758</td><td>53,716</td><td>3%</td><td>8.1 h</td><td>7.9%</td></tr><tr><td>ACS, best θ</td><td>0.714</td><td>52,971</td><td>19%</td><td>12.4 h</td><td>24.7%</td><td>0.758</td><td>53,716</td><td>3%</td><td>8.1 h</td><td>7.9%</td></tr><tr><td rowspan="6">BrowseComp-Plus</td><td>full reading</td><td>0.881</td><td>56,169</td><td>0%</td><td>34.9 h</td><td>0%</td><td>0.895</td><td>45,420</td><td>0%</td><td>23.7 h</td><td>0%</td></tr><tr><td>random stop</td><td>0.753</td><td>31,598</td><td>44%</td><td>20.1 h</td><td>42.3%</td><td>0.772</td><td>25,716</td><td>43%</td><td>13.7 h</td><td>42.3%</td></tr><tr><td>verbalized gate</td><td>0.886</td><td>47,853</td><td>15%</td><td>25.6 h</td><td>26.7%</td><td>0.889</td><td>40,168</td><td>12%</td><td>18.9 h</td><td>20.3%</td></tr><tr><td>END gate</td><td>0.873</td><td>35,679</td><td>36%</td><td>18.7 h</td><td>46.5%</td><td>0.890</td><td>33,923</td><td>25%</td><td>15.5 h</td><td>34.5%</td></tr><tr><td>ACS, fixed</td><td>0.869</td><td>33,368</td><td>41%</td><td>19.1 h</td><td>45.2%</td><td>0.895</td><td>39,795</td><td>12%</td><td>19.1 h</td><td>19.3%</td></tr><tr><td>ACS, best θ</td><td>0.869</td><td>33,368</td><td>41%</td><td>19.1 h</td><td>45.2%</td><td>0.895</td><td>39,795</td><td>12%</td><td>19.1 h</td><td>19.3%</td></tr></table>

## 4.2 ACS MATCHES FULL READING AT HALF THE COST

Table 1 shows the central efficiency result: on LongBench-v2, ACS is the only stopping method that matches or exceeds full-reading accuracy on both frontier models. On Qwen3.5, the shared setting improves accuracy from 0.535 to 0.541 while using 50% fewer tokens and 53.1% less elapsed time. The highest-accuracy threshold is the shared θ = 0.995 itself. On Kimi, the shared θ is conservative, yielding only 3% token savings and 14.1% time savings. Lowering the operating point to θ = 0.92 raises accuracy from 0.517 under full reading to 0.535 while saving 49% of tokens and 52.6% of elapsed time.

The asked gates do not achieve this accuracy-efficiency trade-off consistently: on Qwen3.5, the verbalized gate retains near-full accuracy but saves only 30.2% of tokens, whereas the more aggressive END gate saves 65% but loses 2.8 accuracy points. At nearly the same token budgets, random stopping loses 6.1 and 3.6 accuracy points relative to full reading on Qwen3.5 and Kimi, respectively, whereas ACS at the corresponding matched-budget operating points gains 0.6 and 1.8 points. The benefit therefore comes from selecting where to stop, not merely from processing less context.

All reported token and time savings include the calls required by each stopping policy and therefore already account for ACS’s probing overhead. Appendix F, Table 10 isolates this overhead on LongBench-v2 by comparing execution with and without probe calls. The fact that stopping can outperform exhaustive reading also suggests that additional processing is not always harmless. Repeated note rewriting may perturb an already settled answer state. Savings occur across all six LongBench-v2 domains, with the largest reductions on code repositories, the longest domain at 150.2 chunks on average, where they reach 62% on Qwen3.5 and 67% on Kimi (Appendix C).

Table 2: Evidence-relative stopping on S-NIAH, where the evidence-containing chunk is known exactly. Premature measures stopping before the evidence, over-read measures additional chunks read after it, and capture measures the fraction of oracle-available savings realized without premature stopping.
<table><tr><td colspan="8">S-NIAH (N = 250 per model)</td></tr><tr><td>Model</td><td>Policy</td><td>Premature ↓</td><td>Over-read</td><td>Acc.</td><td>Regret</td><td>Capture</td></tr><tr><td rowspan="8">Qwen3-14B</td><td>fixed at 25%</td><td>56.4%</td><td>0.58</td><td>0.436</td><td>+0.560</td><td>50.2%</td></tr><tr><td>random stop</td><td>32.3%</td><td>2.25</td><td>0.676</td><td>+0.320</td><td>37.2%</td></tr><tr><td>verbalized gate</td><td>8.4%</td><td>0.00</td><td>0.916</td><td>+0.080</td><td>92.0%</td></tr><tr><td>END gate</td><td>5.2%</td><td>0.00</td><td>0.948</td><td>+0.048</td><td>95.2%</td></tr><tr><td>ACS, fixed</td><td>1.6%</td><td>1.51</td><td>0.980</td><td>+0.016</td><td>60.7%</td></tr><tr><td>full reading</td><td>0.0%</td><td>4.10</td><td>0.996</td><td>+0.000</td><td>0.0%</td></tr><tr><td>oracle stop</td><td>0.0%</td><td>0.00</td><td>0.996</td><td>+0.000</td><td>100%</td></tr><tr><td>fixed at 25%</td><td>56.4%</td><td>0.58</td><td>0.436</td><td>+0.564</td><td>50.2%</td></tr><tr><td rowspan="8">Qwen3.5-397B</td><td>random stop</td><td>32.3%</td><td>2.25</td><td>0.677</td><td>+0.323</td><td>37.2%</td></tr><tr><td>verbalized gate</td><td>5.6%</td><td>1.30</td><td>0.944</td><td>+0.056</td><td>66.5%</td></tr><tr><td>END gate</td><td>0.0%</td><td>0.67</td><td>1.000</td><td>+0.000</td><td>83.6%</td></tr><tr><td>ACS, fixed</td><td>0.0%</td><td>1.48</td><td>1.000</td><td>+0.000</td><td>63.9%</td></tr><tr><td>full reading</td><td>0.0%</td><td>4.10</td><td>1.000</td><td>+0.000</td><td>0.0%</td></tr><tr><td>oracle stop</td><td>0.0%</td><td>0.00</td><td>1.000</td><td>+0.000</td><td>100%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>fixed at 25%</td><td>56.4%</td><td>0.58</td><td>0.436</td><td>+0.560</td><td>50.2%</td></tr><tr><td rowspan="8">Kimi K2.5</td><td>random stop</td><td>32.3%</td><td>2.25</td><td>0.674</td><td>+0.322</td><td>37.2%</td></tr><tr><td>verbalized gate</td><td>0.0%</td><td>1.30</td><td>0.996</td><td>+0.000</td><td>68.3%</td></tr><tr><td>END gate</td><td>0.0%</td><td>0.10</td><td>0.996</td><td>+0.000</td><td>97.6%</td></tr><tr><td>ACS, fixed</td><td>0.0%</td><td>1.40</td><td>0.996</td><td>+0.000</td><td>65.8%</td></tr><tr><td>full reading</td><td>0.0%</td><td>4.10</td><td>0.996</td><td>+0.000</td><td>0.0%</td></tr><tr><td>oracle stop</td><td>0.0%</td><td>0.00</td><td>0.996</td><td>+0.000</td><td>100%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

On S-NIAH, ACS exactly matches full-reading accuracy on both models while saving 27–28% of tokens and about 29% of elapsed time. Changing θ provides no meaningful additional gain. The asked gates can also work well here. END matches full reading on both models, while the verbalized gate drops 5.6 points on Qwen3.5 but matches full reading on Kimi, reinforcing that asked stopping can be effective but is model-dependent.

On RULER-HotpotQA, ACS remains close to full reading, trailing by only 0.8 points on Qwen3.5 and 0.2 points on Kimi. It is more conservative here, saving 16% and 3% of tokens under the shared setting, but stays closer to full-reading accuracy across both models: the verbalized gate matches full reading on Qwen3.5 yet loses 1.6 points on Kimi, while END loses 0.6 and 2.4 points, respectively.

On BrowseComp-Plus, ACS trades 1.2 accuracy points for 41% fewer tokens and 45.2% less elapsed time on Qwen3.5, while exactly matching full reading on Kimi with 12% token and 19.3% time savings. The best threshold changes nothing on either model. Random stopping at similar or larger token savings loses 12.8 and 12.3 accuracy points, again showing that the gain depends on where reading stops rather than on truncation alone.

## 4.3 IT STOPS ONCE THE EVIDENCE HAS BEEN READ

Table 2 evaluates the stopping decision directly against the evidence position. On S-NIAH, the evidence appears at a mean normalized depth of 0.55, with its chunk ranging from 1 to 21 and a median of 3, making a fixed cutoff a poor substitute for adaptive stopping. On Qwen3-14B, ACS stops before the evidence on only 1.6% of questions, compared with 8.4% for the verbalized gate, 5.2% for END, and 32.3% for random stopping. Thus, ACS reduces premature stopping by 81% relative to the verbalized gate and 69% relative to END, and attains the highest stopping-policy accuracy at 0.980, closing the gap to full/oracle reading to just 1.6 points. For every policy, accuracy after reaching the evidence matches full-reading accuracy, while answers produced before the evidence are incorrect; the residual errors on this benchmark are therefore timing errors. ACS over-reads only 1.51 chunks after the evidence on average while capturing 60.7% of the savings available to an oracle that stops exactly at the evidence. The verbalized and END gates capture more oracle savings, 92.0% and 95.2%, but do so with substantially higher premature-stop rates and corresponding accuracy regret.

Table 3: One setting across models. S-NIAH, all 250 questions per model, $\theta = 0 . 9 9 5 , \epsilon = 0 . 0 5 .$ , and w = 3.
<table><tr><td></td><td colspan="2">ACS</td><td colspan="2">Confidence only</td><td colspan="2">Verbalized gate</td></tr><tr><td>Model</td><td>Acc.</td><td>Premature</td><td>Acc.</td><td>Premature</td><td>Acc.</td><td>Premature</td></tr><tr><td>Qwen2.5-7B</td><td>0.948</td><td>3.2%</td><td>0.892</td><td>9.2%</td><td>0.700</td><td>29.2%</td></tr><tr><td>Qwen3-14B</td><td>0.980</td><td>1.6%</td><td>0.928</td><td>6.8%</td><td>0.916</td><td>8.4%</td></tr><tr><td>Qwen3-32B</td><td>1.000</td><td>0.0%</td><td>1.000</td><td>0.0%</td><td>0.592</td><td>40.8%</td></tr><tr><td>Gemma-3-12B</td><td>0.980</td><td>1.6%</td><td>0.872</td><td>12.8%</td><td>0.544</td><td>45.6%</td></tr><tr><td>Gemma-3-27B</td><td>0.876</td><td>12.0%</td><td>0.520</td><td>23.2%</td><td>0.876</td><td>12.0%</td></tr><tr><td>Mean</td><td>0.957</td><td>3.7%</td><td>0.890</td><td>10.4%</td><td>0.726</td><td>27.2%</td></tr></table>

Table 4: Stopping relative to the located answer proxy on RULER-HotpotQA, restricted to the 455 of 500 trajectories with a located nontrivial literal answer occurrence.
<table><tr><td>Model</td><td>Policy</td><td>Pre-proxy ↓</td><td>Over-read</td><td>Acc.</td></tr><tr><td rowspan="7">Qwen3.5-397B</td><td>fixed at 25%</td><td>37.8%</td><td>0.66</td><td>0.420</td></tr><tr><td>random stop</td><td>24.9%</td><td>2.71</td><td>0.522</td></tr><tr><td>verbalized gate</td><td>3.3%</td><td>2.03</td><td>0.708</td></tr><tr><td>END gate</td><td>4.4%</td><td>1.74</td><td>0.701</td></tr><tr><td>ACS, fixed</td><td>2.4%</td><td>3.20</td><td>0.697</td></tr><tr><td>full reading</td><td>0.0%</td><td>5.01</td><td>0.705</td></tr><tr><td>fixed at 25%</td><td>37.8%</td><td>0.66</td><td>0.470</td></tr><tr><td rowspan="5">Kimi K2.5</td><td>random stop</td><td>24.9%</td><td>2.71</td><td>0.565</td></tr><tr><td>verbalized gate</td><td>2.9%</td><td>2.62</td><td>0.725</td></tr><tr><td>END gate</td><td>4.4%</td><td>1.91</td><td>0.714</td></tr><tr><td>ACS, fixed</td><td>0.7%</td><td>4.29</td><td>0.741</td></tr><tr><td>full reading</td><td>0.0%</td><td>5.01</td><td>0.743</td></tr><tr><td rowspan="6">Qwen3-14B</td><td>fixed at 25%</td><td>35.1%</td><td>0.77</td><td></td></tr><tr><td>random stop</td><td>23.1%</td><td></td><td>0.446</td></tr><tr><td>verbalized gate, θ = 0.995</td><td>17.7%</td><td>2.85</td><td>0.513</td></tr><tr><td></td><td></td><td>1.69</td><td>0.572</td></tr><tr><td>END gate ACS, fixed</td><td>17.5%</td><td>1.53</td><td>0.578</td></tr><tr><td>full reading</td><td>8.8% 0.0%</td><td>3.10 5.25</td><td>0.606 0.645</td></tr></table>

At frontier scale, ACS’s premature timing errors disappear: it exactly matches full-reading accuracy on both Qwen3.5 and Kimi K2.5, achieving 1.000 and 0.996 while capturing 63.9% and 65.8% of the oracle-available savings. The verbalized gate is premature on 5.6% of Qwen3.5 questions but reaches 0% on Kimi, whereas ACS remains within a narrow 0–1.6% premature-stop range across all three models under the same shared configuration. The END gate shows the same model dependence: it incurs 5.2% premature stopping on Qwen3-14B but none on Qwen3.5 or Kimi, indicating that an asked gate can work well on some models without providing the same consistency across models.

The same pattern holds in the broader five-model replication shown in Table 3. Using the same θ = 0.995 configuration, ACS remains within a narrow 0–12% premature-stop range across models from both Qwen and Gemma families, while the verbalized gate varies from 8.4% to 45.6%. This broader replication reinforces that the measured signal is substantially more consistent across models than the asked stopping signal.

The evidence-alignment pattern also extends beyond the exact single-needle setting. On RULER-HotpotQA, where the located answer position provides only a lower-bound evidence proxy, Table 4 shows that ACS nevertheless has the lowest pre-proxy stopping rate across all three models. This suggests that the evidence-aligned stopping behavior is not specific to single-needle retrieval, but persists in a harder multi-hop setting where the true completion point of the evidence is not known exactly.

## 5 ABLATION STUDY

Table 5: Effect of rule components. S-NIAH, Qwen3- 14B, $N = 2 5 0 , \theta = 0 . 9 9 5$ . Five-model rows report means over the replication set.
<table><tr><td>Axis</td><td>Configuration</td><td>Acc.</td><td>Premature</td><td>Tokens</td></tr><tr><td rowspan="2">Stopping signal</td><td>verbalized gate</td><td>0.916</td><td>8.4%</td><td>27,019</td></tr><tr><td>END gate ACS</td><td>0.948</td><td>5.2%</td><td>28,403</td></tr><tr><td rowspan="2">Stability test, five models</td><td></td><td>0.980</td><td>1.6%</td><td>37,696</td></tr><tr><td>confidence only both tests (default)</td><td>0.890 0.957</td><td>10.4% 3.7%</td><td>37,901 44,768</td></tr><tr><td rowspan="2">Chunk size L</td><td>12K</td><td></td><td></td><td></td></tr><tr><td>24K (default)</td><td>0.968 0.980</td><td>3.2% 1.6%</td><td>34,390 37,696</td></tr><tr><td rowspan="2"></td><td>48K</td><td>0.976</td><td>2.4%</td><td>42,018</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Notes cap B</td><td>3K</td><td>0.992</td><td>0.4%</td><td>38,159</td></tr><tr><td>6K (default)</td><td>0.980</td><td>1.6%</td><td>37,696</td></tr><tr><td>12K</td><td>0.980</td><td>1.6%</td><td>38,020</td></tr></table>

Table 6: Effect of the confidence threshold. LongBench-v2, $N = 5 0 3$ , and S-NIAH, $N =$ 250; ε = 0.05, w = 3.
<table><tr><td></td><td colspan="2">Qwen3.5, LB-v2</td><td colspan="2">Kimi, LB-v2</td><td colspan="2">Needle premature</td></tr><tr><td>θ</td><td>Acc.</td><td>Saving</td><td>Acc.</td><td>Saving</td><td>Qwen3.5</td><td>Kimi</td></tr><tr><td>0.80</td><td>0.479</td><td>79%</td><td>0.521</td><td>58%</td><td>30.4%</td><td>0.4%</td></tr><tr><td>0.90</td><td>0.513</td><td>73%</td><td>0.531</td><td>51%</td><td>28.0%</td><td>0.4%</td></tr><tr><td>0.92</td><td>0.517</td><td>72%</td><td>0.535</td><td>49%</td><td>24.8%</td><td>0.0%</td></tr><tr><td>0.95</td><td>0.527</td><td>68%</td><td>0.533</td><td>44%</td><td>10.4%</td><td>0.0%</td></tr><tr><td>0.98</td><td>0.533</td><td>61%</td><td>0.523</td><td>29%</td><td>1.2%</td><td>0.0%</td></tr><tr><td>0.99</td><td>0.529</td><td>54%</td><td>0.519</td><td>19%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>0.995</td><td>0.541</td><td>50%</td><td>0.519</td><td>3%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>full reading</td><td>0.535</td><td>0%</td><td>0.517</td><td>0%</td><td>0.0%</td><td>0.0%</td></tr></table>

The stability test carries the rule. Removing stability raises the five-model mean premature-stop rate from 3.7% to 10.4% and lowers mean accuracy from 0.957 to 0.890. The effect is especially pronounced on Gemma-3-27B, where accuracy falls from 0.876 with stability to 0.520 without it (Table 3).

The threshold controls the safety–efficiency trade-off. $\mathrm { O n } \mathrm { Q w e n } 3 . 5$ , increasing θ sharply reduces premature stopping, from 30.4% at $\theta = 0 . 8 0$ to zero at $\theta \ge 0 . 9 9$ , while the shared $\mathsf { \bar { \theta } } = 0 . 9 9 5$ also gives the highest LongBench-v2 accuracy, 0.541. Kimi behaves differently: every tested threshold remains at or above its full-reading accuracy of 0.517, while token savings range from 58% at $\theta = 0 . 8 0$ to 3% at $\theta = 0 . 9 9 5$ ; here, θ primarily controls how aggressively the model stops. This difference reflects a shift in confidence scale: the median full-reading confidence is 0.992 for Qwen3.5 but only 0.873 for Kimi. Consistent with this shift, Kimi at $\theta = 0 . 9 2$ reaches a very similar LongBench-v2 accuracy–efficiency operating point to Qwen3.5 at $\theta = 0 . 9 9 5$ , with accuracies of 0.535 and 0.541 and token savings of 49% and 50%, respectively. On S-NIAH, Qwen3.5 still stops prematurely on 1.2% of questions at $\theta = 0 . 9 8$ , with premature stopping disappearing only at $\theta \geq 0 . 9 9$ . Among the evaluated operating points, $\theta = 0 . 9 9 5$ is the only one that both exceeds full-reading accuracy on Qwen3.5 and incurs no premature stops on either S-NIAH model, although this conservatism reduces Kimi’s token savings from 49% at $\theta = 0 . 9 2$ to only 3%.

The remaining constants show limited sensitivity in the tested ranges. On S-NIAH, no observed stability change falls between 0 and 0.05, so every tested $\varepsilon \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 \}$ yields identical stopping decisions (Appendix D). On LongBench-v2, varying the stability window from $w = 2$ to $w = 5$ changes accuracy by at most 0.8 points on either model, while savings vary more moderately. The default $w = 3$ gives the highest accuracy on Kimi and ties for the highest on Qwen3.5 while retaining greater savings than the larger windows (Appendix F: Table 13).

The fold constants are not hidden thresholds. On S-NIAH, halving or doubling either the chunk size L or the notes cap B changes accuracy and premature stopping only modestly, showing that stopping behavior is not driven by these fold parameters. LongBench-v2 is more sensitive, consistent with its longer and more heterogeneous contexts placing greater demands on the running notes across repeated updates. There, the default $L = 2 4 \mathrm { K }$ and $B \bar { = } \bar { 6 } \mathsf { K }$ give the highest accuracy (0.538), while moving either parameter in either direction reduces accuracy, with different accuracy–cost trade-offs (Appendix F: Table 12).

## 6 CONCLUSION

We introduced ACS, a training-free stopping rule for chunked long-context reading that halts when the frozen model’s answer state is both confident and stable, using one shared configuration across models and benchmarks. On the full LongBench-v2, ACS is the only stopping policy that matches or exceeds full-reading accuracy on both frontier models; on Qwen3.5, it does so with 50% fewer tokens and 53.1% less elapsed time than full reading. On S-NIAH, it reduces premature stopping to 1.6% on Qwen3-14B and remains within a narrow 0–12% range across five models, while the verbalized gate varies from 8.4% to 45.6%. These results show that a useful stopping signal can be obtained from the model’s evolving output-side answer state without training a separate controller or accessing internal activations. The ablations further show that confidence alone is insufficient: stability substantially reduces premature stopping, while the confidence threshold exposes an explicit safety–efficiency trade-off whose operating point depends on the model’s confidence scale. Ultimately, ACS shifts the long-context problem from how to read more to how to read efficiently.

## 7 LIMITATIONS

ACS requires token log probabilities, which some inference endpoints do not expose. Its efficiency gains are also workload-dependent: when the answer state does not become sufficiently confident and stable until late in the context, ACS continues reading and can approach the cost of full reading. This reflects the method’s intended safety-efficiency trade-off rather than an assumption that substantial savings are always available.

## 8 ACKNOWLEDGMENT

We are grateful to the KAUST Academy for its generous support, and especially to Prof. Sultan Albarakati who made this work possible. For compute time, this research used Ibex managed by the Supercomputing Core Laboratory at King Abdullah University of Science & Technology (KAUST) in Thuwal, Saudi Arabia.

## REFERENCES

Pranjal Aggarwal, Aman Madaan, Yiming Yang, and Mausam. Let’s sample step by step: Adaptiveconsistency for efficient reasoning and coding with llms. arXiv preprint arXiv:2305.11860, 2023.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 3639–3664, 2025. doi: 10.18653/v1/2025.acl-long.183.

Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. Walking down the memory maze: Beyond context limit through interactive reading. arXiv preprint arXiv:2310.05029, 2023.

Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv preprint arXiv:2508.06600, 2025.

Mohamed Eltahir, Lama Ayash, Ali Habibullah, Tanveer Hussain, and Naeemullah Khan. Gridprobe: Posterior-probing for adaptive test-time compute in long-video vlms. arXiv preprint arXiv:2605.10762, 2026.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? In Conference on Language Modeling (COLM), 2024.

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In EMNLP, 2023.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Kimi Team. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In ICLR, 2023.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In SOSP, 2023.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Amartya Roy, Rasul Tutunov, Xiaotong Ji, Matthieu Zimmer, and Haitham Bou-Ammar. The Ycombinator for LLMs: Solving long-context rot with λ-calculus. arXiv preprint arXiv:2603.20105, 2026.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Q. Tran, Yi Tay, and Donald Metzler. Confident adaptive language modeling. NeurIPS, 2022.

Leheng Sheng, Yongtao Zhang, Wenchang Ma, Yaorui Shi, Ting Huang, Xiang Wang, An Zhang, Ke Shen, and Tat-Seng Chua. When to memorize and when to stop: Gated recurrent memory for long-context reasoning. arXiv preprint arXiv:2602.10560, 2026.

Roy Xie, Junlin Wang, Paul Rosu, Chunyuan Deng, Bolun Sun, Zihao Lin, and Bhuwan Dhingra. Knowing when to stop: Efficient context processing via latent sufficiency signals. In Advances in Neural Information Processing Systems, volume 38, 2025.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In ICLR, 2024.

Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.

Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models. arXiv preprint arXiv:2512.24601, 2025.

Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan O. Arik. Chain of agents: Large language models collaborating on long-context tasks. arXiv preprint arXiv:2406.02818, 2024.

## APPENDIX

## A IMPLEMENTATION DETAILS

Fold. Chunks are fixed-width slices of L = 24,000 characters in document order. The notes prompt instructs the model to retain information relevant to the question, preserve exact names and numbers, and remain within the B = 6,000-character cap. Every prompt is reproduced verbatim in Appendix B.

Probe. For multiple-choice questions, the probe ends in Answer: and generation is constrained to the option letters, with top-k = 20 token log probabilities returned. For open-ended questions, the probe greedily decodes at most 32 tokens. Answer labels such as Answer: are stripped before drafts are compared, since models may alternate between labeled and bare answers. A draft matching a fixed list of abstention phrases is treated as an abstention and cannot trigger stopping.

Baselines. Full reading processes every chunk before probing. Random stopping samples a stop uniformly from 1 to T and is averaged over 200 draws. The verbalized gate asks for a 0–100 confidence that the current notes suffice and stops at 99.5. The END/CONTINUE gate asks the model whether enough information has been collected and stops on END. All stopping policies are evaluated over the same recorded reader trajectories because stopping decisions do not modify note updates or subsequent reader inputs. Replay provides a paired comparison in which policies differ only in when they stop; each policy is charged for its own required calls.

Constants. The shared configuration is θ = 0.995, ε = 0.05, and w = 3, and is fixed for all main experiments.

BrowseComp-Plus construction. We reconstruct all 830 BrowseComp-Plus test questions deterministically using seed 0. All evidence and gold documents are retained without truncation. Unique corpus distractors of at most 16,000 characters are added until each context contains at least ten documents; questions requiring more mandatory documents retain all of them. Document order is then shuffled deterministically. The query and corpus revisions are pinned for reproducibility.

## B PROMPTS

All prompts are fixed across steps, models, and benchmarks. Placeholders in braces are filled per call: {qblock} is the question, with the four options appended for multiple choice, {notes} the running notes, {chunk} the current chunk, {cap} the notes cap B, and {draft} the current draft answer.

## Fold, system prompt, multiple choice.

You maintain running NOTES that gather every piece of information relevant to answering a multiple-choice question about a long document, read chunk by chunk. Update the notes with relevant facts from the new chunk, keep prior facts, stay under {cap} characters, output ONLY the updated notes.

## Fold, system prompt, open-ended.

You maintain compact running NOTES needed to answer an open-ended question about a long context read chunk by chunk. Update and rewrite the notes using relevant information from the new chunk while preserving useful evidence from earlier chunks. Adapt the working state to the question: preserve exact names, facts, numbers, relationships, unresolved candidates, and contradictions; maintain counts or calculations for aggregation and evidence chains for multi-step questions. Remove only irrelevant or superseded material. Do not narrate the reading process, make unsupported guesses, or treat missing evidence as a negative answer. Place the most important current state near the end, stay under {cap} characters, and output ONLY the updated notes.

Fold, user turn.   
QUESTION:   
{qblock}   
CURRENT NOTES:   
{notes}   
NEW CHUNK ({i}/{n}):   
{chunk}   
UPDATED NOTES:   
Probe, multiple choice. Generation is constrained to the option letters.   
QUESTION:   
{qblock}   
NOTES SO FAR:   
{notes}   
Based only on the notes, answer with a single letter (A, B,   
C, or D).   
Answer:   
Probe, open-ended.   
QUESTION:   
{qblock}   
NOTES SO FAR:   
{notes}   
Based only on the notes, give your best current answer.   
Reply with ONLY the answer.   
Answer:   
Verbalized gate.   
QUESTION:   
{qblock}   
NOTES SO FAR:   
{notes}   
How confident are you (0-100) that the notes are sufficient   
to answer correctly? Reply with ONLY a number.   
Confidence:   
END gate.   
QUESTION:   
{qblock}   
NOTES SO FAR:   
{notes}   
Decide whether the notes contain enough information to   
answer the question. ONLY when enough information is   
collected, return <next>end</next>. Otherwise return   
<next>continue</next>.   
Decision:

## C LONGBENCH-V2 BY DOMAIN

Per-domain accuracy differences between ACS and full reading are small and inconsistent across the two models, so we do not interpret them as domain-level accuracy effects. The clearest efficiency result is on code repositories, which have by far the longest contexts at 150.2 chunks on average and yield the largest savings: 62% on Qwen3.5 and 67% on Kimi.

Table 7: ACS token savings on LongBench-v2 by domain at each model’s best θ, ordered by mean chunks per document.
<table><tr><td>Domain</td><td>n</td><td>Mean chunks</td><td>Saving, Qwen3.5</td><td>Saving, Kimi</td></tr><tr><td>Long-dialogue history</td><td>39</td><td>13.9</td><td>31%</td><td>31%</td></tr><tr><td>Single-document QA</td><td>175</td><td>19.1</td><td>39%</td><td>28%</td></tr><tr><td>Multi-document QA</td><td>125</td><td>22.4</td><td>31%</td><td>32%</td></tr><tr><td>Long in-context learning</td><td>81</td><td>36.3</td><td>50%</td><td>45%</td></tr><tr><td>Long structured data</td><td>33</td><td>51.8</td><td>37%</td><td>42%</td></tr><tr><td>Code repository</td><td>50</td><td>150.2</td><td>62%</td><td>67%</td></tr></table>

Table 8: Accuracy and mean token cost on S-NIAH, Qwen3-14B, all 250 questions, as θ varies. Every ϵ ∈ {0.005, 0.01, 0.02, 0.05} produces the same results.
<table><tr><td>θ</td><td>0.5</td><td>0.8</td><td>0.9</td><td>0.95</td><td>0.97</td><td>0.98</td><td>0.99</td><td>0.995</td></tr><tr><td>Acc.</td><td>0.764</td><td>0.788</td><td>0.856</td><td>0.872</td><td>0.900</td><td>0.912</td><td>0.956</td><td>0.980</td></tr><tr><td>Tokens (K)</td><td>26.1</td><td>27.1</td><td>31.1</td><td>31.8</td><td>33.1</td><td>33.6</td><td>36.5</td><td>37.7</td></tr></table>

## D SENSITIVITY

Confidence scales. The highest-accuracy LongBench-v2 operating points, $\theta = 0 . 9 9 5$ for Qwen3.5 and $\theta = 0 . 9 2$ for Kimi, lie at similar positions in their respective full-reading confidence distributions: the 54th and 55th percentiles, respectively, despite median confidences of 0.992 and 0.873. This similarity does not yield a general percentile-based rule. On S-NIAH, the corresponding percentile maps to 1.000 for models whose confidence saturates, while for Qwen3-32B it gives $\theta = 0 . 9 4 2$ which stops before the evidence on 30% of questions. Thus, normalizing θ by a model’s fullreading confidence distribution does not replace the conservative shared threshold used in the main experiments.

## E ROBUSTNESS OF THE SHARED OPERATING POINT

We examine whether a single operating point can be used across model families without per-model tuning. The analysis uses the complete S-NIAH trajectories: 250 questions for each of five models. Because all models are evaluated on the same question IDs, we apply one deterministic split shared across models. Questions whose integer MD5 hash is even form the development set $( n = 1 3 5 )$ , and the remaining questions form the held-out test set (n = 115).

On the development split, we search

$$
\theta \in \{ 0 . 5 0 , 0 . 6 0 , 0 . 7 0 , 0 . 8 0 , 0 . 9 0 , 0 . 9 2 , 0 . 9 4 , 0 . 9 5 , 0 . 9 6 , 0 . 9 7 , 0 . 9 8 , 0 . 9 9 , 0 . 9 9 5 \}
$$

and

$$
\varepsilon \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 \} ,
$$

with the stability window fixed at $w = 3$ . For each model, let $A _ { \mathrm { d e v } } ^ { \star }$ denote the highest development accuracy observed on the grid. We then select the lowest-cost configuration satisfying

$$
A _ { \mathrm { d e v } } \geq A _ { \mathrm { d e v } } ^ { \star } - 0 . 0 2 ,
$$

breaking ties by higher development accuracy and then by the more conservative threshold. This produces a model-specific cost-aware operating point that we compare against the shared fixed configuration $( \theta , \varepsilon , \bar { w } ) = ( 0 . 9 9 5 , 0 . 0 5 , 3 )$ on the same held-out questions.

Table 9: Model-specific cost-aware tuning versus the shared fixed operating point on S-NIAH. All results use the same 115-question held-out split. The shared configuration is $( \theta , \varepsilon , w ) ~ =$ (0.995, 0.05, 3).
<table><tr><td>Model</td><td>Tuned (θ, ε)</td><td>Tuned Acc.</td><td>Fixed Acc.</td><td>Fixed Premature ↓</td></tr><tr><td>Qwen2.5-7B</td><td>(0.98,0.05)</td><td>0.878</td><td>0.922</td><td>6.1%</td></tr><tr><td>Qwen3-14B</td><td>(0.995, 0.05)</td><td>0.965</td><td>0.965</td><td>2.6%</td></tr><tr><td>Qwen3-32B</td><td>(0.80, 0.05)</td><td>0.965</td><td>1.000</td><td>0.0%</td></tr><tr><td>Gemma-3-12B</td><td>(0.995,0.05)</td><td>0.983</td><td>0.983</td><td>1.7%</td></tr><tr><td>Gemma-3-27B</td><td>(0.50, 0.02)</td><td>0.791</td><td>0.835</td><td>16.5%</td></tr><tr><td>Mean</td><td></td><td>0.917</td><td>0.941</td><td>5.4%</td></tr></table>

The cost-aware tuned operating points vary substantially across models, from θ = 0.50 to 0.995, whereas the shared conservative configuration retains equal or higher held-out accuracy on every model and yields a higher mean held-out accuracy, 0.941 versus 0.917. This supports the robustness of the fixed $\theta = 0 . 9 9 { \bar { 5 } }$ operating point across models.

## F ADDITIONAL RESULTS

Table 10: Probe-overhead accounting on LongBench-v2, all 503 questions. Savings are relative to full reading: 382,971 tokens for Qwen3.5 and 340,761 for Kimi. “Without probes” is a counterfactual accounting ablation that preserves the same stopping decisions and accuracy while removing intermediate probe calls; the final probe used to produce the answer is retained.
<table><tr><td>Model</td><td>Configuration</td><td>Acc.</td><td>With probes</td><td>Probe cost</td><td>Without probes</td><td>Saving w/o probes</td></tr><tr><td>Qwen3.5</td><td>fixed/best</td><td>0.541</td><td>190,966</td><td>24,079</td><td>166,887</td><td>56.4%</td></tr><tr><td>Kimi</td><td>fixed</td><td>0.519</td><td>331,070</td><td>37,988</td><td>293,082</td><td>14.0%</td></tr><tr><td>Kimi</td><td>best θ</td><td>0.535</td><td>172,648</td><td>19,968</td><td>152,680</td><td>55.2%</td></tr></table>

Probe calls introduce a measurable token cost, but this overhead is already included in all main results; removing it counterfactually would increase the available savings to 56.4% on Qwen3.5 and up to 55.2% on Kimi.

Table 11: Threshold sweep on RULER-HotpotQA, all 500 trajectories, $\epsilon = 0 . 0 5 , w = 3$ . Pre-proxy stopping is measured on the 455 trajectories with a located nontrivial literal answer mention. Accuracy uses the official RULER substring criterion, and saving is relative to full reading.
<table><tr><td rowspan="2">θ</td><td colspan="2">Qwen3.5</td><td colspan="2">Kimi</td><td colspan="2">Qwen3-14B</td><td colspan="3">Premature</td></tr><tr><td>Acc.</td><td>Saving</td><td>Acc.</td><td>Saving</td><td>Acc.</td><td>Saving</td><td>Qwen3.5</td><td>Kimi</td><td>Qwen3-14B</td></tr><tr><td>0.80</td><td>0.660</td><td>36%</td><td>0.730</td><td>32%</td><td>0.532</td><td>53%</td><td>10.3%</td><td>7.9%</td><td>20.7%</td></tr><tr><td>0.90</td><td>0.682</td><td>31%</td><td>0.742</td><td>26%</td><td>0.556</td><td>48%</td><td>8.4%</td><td>6.6%</td><td>18.1%</td></tr><tr><td>0.92</td><td>0.686</td><td>30%</td><td>0.750</td><td>24%</td><td>0.556</td><td>48%</td><td>7.7%</td><td>5.1%</td><td>17.7%</td></tr><tr><td>0.95</td><td>0.698</td><td>26%</td><td>0.752</td><td>19%</td><td>0.562</td><td>46%</td><td>6.2%</td><td>3.5%</td><td>16.3%</td></tr><tr><td>0.98</td><td>0.708</td><td>21%</td><td>0.754</td><td>11%</td><td>0.580</td><td>40%</td><td>4.0%</td><td>1.8%</td><td>12.2%</td></tr><tr><td>0.99</td><td>0.714</td><td>19%</td><td>0.756</td><td>6%</td><td>0.590</td><td>37%</td><td>3.3%</td><td>1.3%</td><td>11.0%</td></tr><tr><td>0.995</td><td>0.714</td><td>16%</td><td>0.758</td><td>3%</td><td>0.608</td><td>32%</td><td>2.4%</td><td>0.7%</td><td>8.8%</td></tr><tr><td>full reading</td><td>0.722</td><td>0%</td><td>0.760</td><td>0%</td><td>0.646</td><td>0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr></table>

Across all three models, increasing θ reduces pre-proxy stopping and moves accuracy toward full reading at the cost of lower savings. Qwen3.5 reaches its highest accuracy at $\theta = 0 . 9 9$ , while Kimi and Qwen3-14B continue improving through $\theta = 0 . 9 9 5$ . The effect is especially pronounced for Qwen3-14B, where the stricter threshold reduces premature stopping from 20.7% to 8.8% while increasing accuracy from 0.532 to 0.608.

Table 12: LongBench-v2 chunk-size and notes-cap ablations with Kimi K2.5 on the prespecified 80-question subset, using $\theta = 0 . 9 2 , \epsilon = 0 . 0 5$ , and w = 3. Regret is measured relative to full reading under the same L or B configuration.
<table><tr><td>Axis</td><td>Configuration</td><td>Acc.</td><td>Regret↓</td><td>Tokens</td></tr><tr><td rowspan="3">Chunk size L</td><td>12K</td><td>0.425</td><td>3.8%</td><td>209,470</td></tr><tr><td>24K (default)</td><td>0.538</td><td>1.3%</td><td>168,160</td></tr><tr><td>48K</td><td>0.500</td><td>1.3%</td><td>151,960</td></tr><tr><td rowspan="3">Notes cap B</td><td>3K</td><td>0.525</td><td>2.5%</td><td>98,755</td></tr><tr><td>6K (default)</td><td>0.538</td><td>1.3%</td><td>168,160</td></tr><tr><td>12K</td><td>0.475</td><td>2.5%</td><td>188,747</td></tr></table>

On this LongBench-v2 subset, the default $L = 2 4 \mathrm { K }$ and $B = 6 \mathsf { K }$ give the highest accuracy, while moving either parameter exposes different accuracy–cost trade-offs rather than improving both simultaneously.

Table 13: Stability-window ablation on LongBench-v2, all 503 questions. Qwen3.5 uses $\theta = 0 . 9 9 5$ and Kimi uses $\theta \overset { \cdot } { = } 0 . 9 2 ; \varepsilon = 0 . 0 5$ . Saving is relative to full reading.
<table><tr><td></td><td colspan="3">Qwen3.5-397B-A17B</td><td colspan="3">Kimi K2.5</td></tr><tr><td>w</td><td>Acc.</td><td>Tokens</td><td>Saving</td><td>Acc.</td><td>Tokens</td><td>Saving</td></tr><tr><td>2</td><td>0.533</td><td>185,722</td><td>51.5%</td><td>0.529</td><td>168,644</td><td>50.5%</td></tr><tr><td>3 (default)</td><td>0.541</td><td>190,966</td><td>50.0%</td><td>0.535</td><td>172,648</td><td>49.0%</td></tr><tr><td>4</td><td>0.537</td><td>202,054</td><td>47.2%</td><td>0.533</td><td>180,912</td><td>46.9%</td></tr><tr><td>5</td><td>0.541</td><td>206,716</td><td>46.0%</td><td>0.527</td><td>194,035</td><td>43.1%</td></tr><tr><td>full reading</td><td>0.535</td><td>382,971</td><td>0%</td><td>0.517</td><td>340,761</td><td>0%</td></tr></table>

Accuracy varies by at most 0.8 points across $w = 2 – 5$ on either model. The default w = 3 gives the highest accuracy on Kimi and ties for the highest on Qwen3.5, while preserving greater savings than the larger windows.

Table 14: Per-model verbalized-threshold ablation on S-NIAH, with 250 questions per model. Thresholds rescale the verbalized 0–100 output to [0, 1]. The selected 0.995 setting is the strictest point on the prespecified grid.
<table><tr><td>Model</td><td>Metric</td><td>0.80</td><td>0.90</td><td>0.92</td><td>0.95</td><td>0.98</td><td>0.99</td><td>0.995</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>Acc.</td><td>0.568</td><td>0.596</td><td>0.596</td><td>0.596</td><td>0.700</td><td>0.700</td><td>0.700</td></tr><tr><td>Premature ↓</td><td>43.2%</td><td>40.4%</td><td>40.4%</td><td>40.4%</td><td>29.2%</td><td>29.2%</td><td>29.2%</td></tr><tr><td rowspan="2">Qwen3-14B</td><td>Acc.</td><td>0.916</td><td>0.916</td><td>0.916</td><td>0.916</td><td>0.916</td><td>0.916</td><td>0.916</td></tr><tr><td>Premature ↓</td><td>8.4%</td><td>8.4%</td><td>8.4%</td><td>8.4%</td><td>8.4%</td><td>8.4%</td><td>8.4%</td></tr><tr><td rowspan="2">Qwen3-32B</td><td>Acc.</td><td>0.588</td><td>0.592</td><td>0.592</td><td>0.592</td><td>0.592</td><td>0.592</td><td>0.592</td></tr><tr><td>Premature ↓</td><td>41.2%</td><td>40.8%</td><td>40.8%</td><td>40.8%</td><td>40.8%</td><td>40.8%</td><td>40.8%</td></tr><tr><td rowspan="2">Gemma-3-12B</td><td>Acc.</td><td>0.544</td><td>0.544</td><td>0.544</td><td>0.544</td><td>0.544</td><td>0.544</td><td>0.544</td></tr><tr><td>Premature ↓</td><td>45.6%</td><td>45.6%</td><td>45.6%</td><td>45.6%</td><td>45.6%</td><td>45.6%</td><td>45.6%</td></tr><tr><td rowspan="2">Gemma-3-27B</td><td>Acc.</td><td>0.808</td><td>0.816</td><td>0.840</td><td>0.840</td><td>0.876</td><td>0.876</td><td>0.876</td></tr><tr><td>Premature ↓</td><td>18.8%</td><td>18.0%</td><td>15.6%</td><td>15.6%</td><td>12.0%</td><td>12.0%</td><td>12.0%</td></tr></table>

Raising the verbalized threshold helps some models but has little or no effect on others. We therefore use 0.995, the strictest prespecified value, as a single conservative threshold across models. It attains or ties the best observed operating point for every model, yet verbalized stopping remains highly premature on Qwen3-32B and Gemma-3-12B, showing that threshold adjustment alone does not resolve its model dependence.