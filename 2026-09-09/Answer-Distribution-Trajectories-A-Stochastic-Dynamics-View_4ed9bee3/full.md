# Answer-Distribution Trajectories: A Stochastic-Dynamics View of LLM Reasoning

Mar Gonzàlez I Català University of Cambridge mg2211@cam.ac.uk

Carola Bibiane-Schönlieb University of Cambridge cbs31@cam.ac.uk

Haitz Sáez de Ocáriz Borde University of Cambridge hs788@cam.ac.uk

Davide Murari University of Cambridge dm2011@cam.ac.uk

Pietro Liò University of Cambridge pl219@cam.ac.uk

George D. Montañez Harvey Mudd College gmontanez@g.hmc.edu

## Abstract

Chain-of-thought reasoning provides a structured computation between a model’s input and final answer. Yet it is often evaluated through endpoint accuracy, which ignores the path taken to reach that answer. An emerging line of work addresses this limitation using entropy profiles, which track how uncertainty evolves over the reasoning process but do not reveal which competing hypotheses account for that uncertainty. We introduce answer-distribution trajectories, a stochastic-dynamicsinspired representation that tracks the model’s full predictive distribution over answers as reasoning unfolds. As a strictly finer representation than endpoint and entropy summaries, answer-distribution trajectories enable us to characterize a trace through a dynamical reasoning profile spanning exploration, revision, motion, and commitment, and to distinguish different dynamical mechanisms of reasoning success and failure. Across sixteen open-weight language models and four reasoning benchmarks, we show that traces with the same endpoint and similar entropy profiles can exhibit substantially different reasoning dynamics. We further find substantial variation in these dynamics both within and across models and tasks, with different objectives favoring different dynamical profiles. Additionally, we show that training and inference choices systematically reshape these profiles. Our results suggest that answer-distribution trajectories provide a rich framework for analysing and evaluating the dynamics of LLM reasoning.

## 1 Introduction

Chain-of-thought (CoT) reasoning first emerged as a powerful capability of large language models: when prompted to produce intermediate natural-language steps, sufficiently capable models could substantially improve performance on multi-step reasoning tasks [21, 42]. What began as a prompting technique has since become integrated into model training and post-training, leading to modern reasoning models that are explicitly optimized to generate extended intermediate computations before producing an answer [16, 28]. In this setting, CoT is part of the model’s inference-time computation, providing an observable trace of the model’s reasoning. Despite having access to this trace, reasoning models are still mostly evaluated by whether the final answer is correct, without accounting for the reasoning that produced it [26, 31, 44, 38].

Recent work has therefore begun to study CoT traces and signals derived from them to better understand how a model arrives at its answer [22, 25, 41]. In particular, a growing line of work examines entropy profiles, tracking how uncertainty over the next token or final answer evolves during CoT generation. These profiles have been used to guide exploration and early stopping [34,

![](images/65ec6bedd1486044bb1f4c12d9c8713c64b704c77b6297e6219a9b3de2071862.jpg)  
Figure 1: Answer-distribution trajectories reveal how predictive mass moves among competing hypotheses during reasoning. Here, the answer 12 initially dominates before losing support as probability shifts toward the answer 18. This switch in the dominant answer is not recoverable from the final answer or entropy alone.

47], identify critical decision points [32, 39], and detect reasoning failures [13, 35]. However, entropy is a limited signal: it captures how probability mass is distributed, but not which hypotheses carry that probability mass [20, 43].

To address these limitations, we propose analyzing reasoning through answer-distribution trajectories, which track a model’s predictive distribution over possible final answers as the CoT is generated (Figure 1). For a given trace, we estimate this distribution at selected prefixes by conditioning on the reasoning generated so far and sampling independent continuations. The resulting trajectory contains strictly more information than final-answer accuracy or entropy, as it preserves which hypotheses are supported, how that support evolves, and where it ultimately settles.

Answer-distribution trajectories allow us to define trajectory-level metrics that capture different aspects of the reasoning dynamics. We group these metrics along four dimensions: exploration, revision, motion, and commitment, which together form a dynamical profile of a given reasoning trajectory. We then track the dominance of the correct answer along each trajectory to distinguish different mechanisms of success and failure. We apply this framework to a diverse collection of openweight language models and four discrete-answer benchmarks and find substantial heterogeneity in reasoning dynamics, even among traces with the same endpoint and similar entropy profiles. We further show that different objectives favor different dynamical profiles, and that training and inference choices systematically reshape these profiles.

Our contributions focus on the following three questions:

• What do conventional summaries of reasoning miss? We show both theoretically and empirically that similar endpoint accuracy and entropy profiles can arise from different answerdistribution trajectories.

• Do models exhibit systematic differences in reasoning dynamics? We construct dynamical profiles of reasoning trajectories and use them to reveal systematic differences in how final answers are reached within and across model families and tasks.

• How do training and inference choices reshape these dynamics? We analyze how changes in training checkpoint, scale, and decoding temperature affect dynamical profiles.

## 2 Related work

Studying reasoning beyond the final answer. Final-answer accuracy does not reveal how a model arrived at its prediction, motivating work that studies the reasoning process itself [31, 44]. A natural observable signal is Chain-of-Thought (CoT), which exposes a sequence of intermediate natural-language steps before the final answer [21, 42]. CoT’s textual content can be monitored for undesirable behavior [3, 12, 22], trace length provides a signal of reasoning confidence [11], agreement across independently sampled chains is a measure of solution robustness [40], and steplevel verifier scores have been developed as signals of local reasoning validity [38, 25]. However,

CoT may not reflect the actual reasons that determined the final answer [37, 1, 23]. Complementary work instead studies the model’s internal computations directly, probing hidden representations for information about truthfulness, hallucination, future outputs, and latent reasoning concepts [2, 27, 6, 29, 17], or intervening on these representations to steer model behavior [49, 36]. In our work, we propose to study the reasoning process by tracking the evolution of the model’s predictive distribution over candidate final answers as reasoning unfolds.

Information-theoretic views of reasoning. A growing line of work studies information-theoretic signals at different points in the reasoning process. Information gain, mutual information, and token entropy have been used to identify important points of the reasoning process [35, 32, 39]. Entropy-based signals have been used to adapt exploration depth [47], compress redundant reasoning steps [24], and inform early stopping [34, 19]. Other work studies the temporal shape of uncertainty: intermediate-answer confidence exhibits different dynamics in correct and incorrect rollouts [19], early entropy trajectories characterize different reasoning regimes [45], and theoretical work relates conditional answer-entropy dynamics to the accumulation of answer-relevant information [5]. However, entropy is limited because it does not capture which hypotheses carry the probability mass. To address this limitation, we instead study the full predictive distribution over candidate answers throughout reasoning.

## 3 Answer-distribution trajectories: a stochastic-dynamics view of reasoning

In this section, we formalize answer-distribution trajectories and establish how they relate to endpoint prediction and entropy profiles.

## 3.1 Answer-distribution trajectories

We first define a reasoning model’s predictive distribution over reasoning traces and final answers.

Definition 1 (Model predictive distribution). Given a query $Q ,$ , a reasoning model with parameters θ generates a sequence of intermediate reasoning tokens $C _ { 1 : K } = ( C _ { 1 } , \dots , C _ { K } )$ followed by an answer sequence $A _ { 1 : T } = ( A _ { 1 } , \ldots , A _ { T } )$ , where K and T are stochastic sequence lengths. The corresponding autoregressive distribution over reasoning traces is given by $\begin{array} { r } { p _ { \theta } ( C _ { 1 : K } ~ \mid ~ Q ) \ = \ \prod _ { k = 1 } ^ { K } p _ { \theta } ( C _ { k } ~ \mid ~ } \end{array}$ $Q , C _ { 1 : k - 1 } )$ . Conditioned on a complete reasoning trace $C _ { 1 : K }$ , the answer distribution factorizes as $\begin{array} { r } { p _ { \theta } ( A _ { 1 : T } \mid Q , C _ { 1 : K } ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( A _ { t } \mid Q , C _ { 1 : K } , A _ { 1 : t - 1 } ) } \end{array}$

In our empirical analysis, we apply a deterministic parser that maps each generated answer sequence $A _ { 1 : T }$ to a single discrete answer label $Y \in A ,$ , where A denotes the discrete answer space. We use this parsed answer label in the definitions that follow.

Definition 2 (Prefix-conditioned predictive distribution). For a prefix $C _ { 1 : k : }$ , with $k > 0 ,$ the prefixconditioned predictive distribution over discrete answer labels is $p _ { k } ( a ) = p _ { \theta } ( Y = a \mid Q , C _ { 1 : k } )$ $a \in { \mathcal { A } }$ . This distribution marginalizes over all possible future reasoning continuations: $p _ { k } ( a ) =$ $\begin{array} { r } { \sum _ { c > \ k } p _ { \theta } ( C _ { > k } = c _ { > k } \ | \ Q , C _ { 1 : k } ^ { \top } ) p _ { \theta } ( Y = a \ | \ Q , \stackrel { . . } { C _ { 1 : k } } , \bar { C _ { > k } } = c > _ { k } ) } \end{array}$ , where $C _ { > k }$ denotes a variablelength future reasoning continuation.

As the reasoning trace unfolds, these prefix-conditioned distributions form a sequence of distributions over the final answer. We refer to this sequence as the model’s answer-distribution trajectory.

Definition 3 (Answer-distribution trajectory). For a question q and realized reasoning trace $c _ { 1 : K }$ we define the answer-distribution trajectory as $\mathcal { T } ( q , c ) = ( p _ { 0 } , p _ { 1 } , . . . , p _ { K } )$ , where $p _ { k }$ denotes the prefix-conditioned predictive distribution associated with the reasoning prefix $c _ { 1 : k }$ and the query q, as defined in Definition 2.

## 3.2 Answer-distribution trajectories provide a finer representation

The answer-distribution trajectory retains substantially more structure than commonly used summaries of reasoning such as endpoint predictions or scalar uncertainty. We formalize this relationship below. Proofs of the theoretical results in this subsection are provided in Appendix C.

For an answer-distribution trajectory $\mathcal { T } = ( p _ { 0 } , p _ { 1 } , \dotsc \dotsc , p _ { K } )$ , we define its endpoint prediction as $E ( \mathcal { T } ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } p _ { K } ( a )$ , and its entropy trajectory as ${ \mathcal { H } } ( { \mathcal { T } } ) = \left( H ( p _ { 0 } ) , H ( p _ { 1 } ) , \dots , H ( p _ { K } ) \right)$ ， where $\begin{array} { r } { H ( p ) = - \sum _ { a \in \mathcal { A } } p ( a ) \log p ( a ) } \end{array}$

Proposition 1 (Trajectory determines endpoint and entropy). Both E and H are deterministic functions ofthe answer-distribution trajectory T.

The converse, however, does not hold.

Proposition 2 (Endpoint and entropy trajectory are non-identifying). Neither endpoint prediction nor entropy trajectory identifies the underlying answer-distribution dynamics. In particular:

1. The endpoint map E is non-injective: there exist distinct answer-distribution trajectories $\mathcal { T } \neq \mathcal { T } ^ { \prime }$ such that $E ( \mathcal { T } ) \overset { \cdot } { = } \mathinner { E \mathopen { \left( \mathcal { T } ^ { \prime } \right) } }$ . Hence, endpoint prediction does not determine the sequence of predictive states that produced it.

2. For $| { \mathcal { A } } | \geq 2 ,$ the entropy map H is non-injective: there exist distinct answer-distribution trajectories $\mathcal { T } \neq \mathcal { T } ^ { \prime }$ such that $\mathcal { H } \dot { ( } T ) = \mathcal { H } ( T ^ { \prime } )$ . Hence, an entropy trajectory does not identify the underlying sequence of predictive states.

Corollary 1 (Answer-distribution trajectories are a strictly finer representation). The answerdistribution trajectory is a strictly finer representation of the reasoning process than either endpoint prediction or entropy trajectory. Both are deterministic functions of the trajectory by Proposition 1, while Proposition 2 shows that neither is sufficient, in general, to recover the underlying trajectory.

The distinction is useful conceptually. Endpoint evaluation preserves the destination of reasoning while discarding its path. Entropy preserves the temporal evolution of concentration while discarding the identity of the hypotheses between which probability mass moves. Answer-distribution trajectories retain both: they record which hypotheses remain plausible at each point in the trace and how probability mass is transported between them.

## 4 Trajectory-level diagnostics

Answer-distribution trajectories allow us to define trajectory-level metrics that summarize different aspects of how predictive answer distributions evolve during reasoning. Taken together, these metrics form a dynamical profile of a reasoning trajectory, providing a common representation for comparing reasoning behavior across models and tasks. By additionally orienting the trajectory relative to the correct answer, we can distinguish different dynamical mechanisms of reasoning success and failure.

## 4.1 Dynamical reasoning profiles

We summarize each trajectory along four complementary dimensions: exploration, revision, motion, and commitment. These values constitute the trajectory’s dynamical profile, providing a common coordinate system for characterizing reasoning dynamics.

Different dynamical patterns may arise across reasoning trajectories, and which profiles are preferable will depend on what we are optimizing for. Dynamical profiles make this heterogeneity explicit and measurable, allowing us to compare reasoning strategies and study which profiles are better suited to different objectives. A first property of a reasoning trajectory is the breadth of its predictive state.

Exploration: How many hypotheses remain in contention? We measure this by the effective support of the predictive distribution, $\begin{array} { r } { S _ { k } ^ { \mathrm { e f f } } = \exp \left( - \sum _ { a } p _ { k } ( a ) \log p _ { k } ( a ) \right) = \exp ( H ( \mathbf { \bar { \rho } } _ { b _ { k } } ) ) } \end{array}$ , which is the base-e perplexity of the predictive answer distribution and can be interpreted as the effective number of plausible answers after k CoT tokens. $S _ { k } ^ { \mathrm { e f f } } \approx .$ 1 indicates concentration on a single hypothesis, while larger values indicate broader competition. To summarize exploration over the full trajectory, we use the mean effective support across the CoT tokens, $\begin{array} { r } { S _ { \mathrm { e f f } } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } S _ { k _ { j } } ^ { \mathrm { e f f } } } \end{array}$

Breadth, however, does not tell us whether reasoning actually changes which hypothesis the model prefers. This motivates a second question.

Revision: Do the leading hypotheses change, and are earlier hypotheses revisited? Let $D _ { k } = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } p _ { k } ( a )$ denote the set of dominant answers after the first k CoT tokens. We first measure how often dominance changes, $\begin{array} { r } { N _ { \mathrm { s w i t c h } } ~ = ~ \sum _ { k = 0 } ^ { K - 1 } \mathbf { 1 } [ D _ { k + 1 } ~ \neq ~ D _ { k } ] } \end{array}$ . To capture whether these switches return to previously considered hypotheses, we also measure recurrence, $\begin{array} { r } { N _ { \mathrm { r e t u r n } } = \sum _ { a \in \mathcal { A } } \# \left\{ k \in \{ 1 , . . . , \dot { K } \} : a \in \dot { D } _ { k } , a \notin D _ { k - 1 } , \ \exists j < k \right. } \end{array}$ such that $a \in D _ { j } \}$

Revision records which answers are on top, but discards how probability mass moves underneath.   
Thus, we next consider the evolution of the full predictive distribution.

Motion: How much does reasoning change the predictive state? We measure consecutive distributional change using total variation, $\begin{array} { r } { J _ { k } = \mathrm { T V } ( p _ { k } , p _ { k + 1 } ) = \frac { 1 } { 2 } \sum _ { a } \left| p _ { k + 1 } ( a ) - p _ { k } ( a ) \right| } \end{array}$ . The total path length, $\begin{array} { r } { V \stackrel { \smile } { = } \sum _ { k } \breve { J _ { k } } } \end{array}$ , measures cumulative distributional movement. Its temporal concentration, $\begin{array} { r } { \mathrm { P R } = ( \sum _ { k } J _ { k } ) ^ { 2 } / ( K \sum _ { k } J _ { k } ^ { 2 } ) } \end{array}$ , distinguishes trajectories that evolve gradually from those whose movement is concentrated in a small number of steps. To quantify movement directness, we compare start-to-end displacement with total path length, $\begin{array} { r } { \dot { R } _ { \mathrm { d i r e c t } } = \frac { T V \dot { ( } p _ { 0 } , p _ { K } ) } { V } } \end{array}$ . For $V > 0 , R _ { \mathrm { d i r e c t } } \in [ 0 , 1 ]$ with higher values indicating more direct trajectories and lower values indicating greater backtracking.

Finally, motion does not tell us when the predictive state settles. Our final dimension captures this.

Commitment: When does competition resolve? We measure when a hypothesis becomes both confident and stable. For a threshold $\tau ~ \in ~ ( 0 . 5 , 1 ]$ , we define $k _ { \mathrm { c o m m i t } } ( \tau ) ~ =$ min $\{ k : \exists a \in { \mathcal { A } }$ such that $\begin{array} { r } { p _ { j } ( a ) \geq \tau ~ \forall j \geq k \} , t _ { \mathrm { c o m m i t } } = \frac { k _ { \mathrm { c o m m i t } } } { K } } \end{array}$ . Small $t _ { \mathrm { c o m m i t } }$ indicates early lock-in, whereas large values indicate persistent competition. Commitment is undefined if no hypothesis remains above the threshold through the end of the trajectory.

These four dimensions describe how broadly probability is distributed across alternatives, how often the dominant answer changes, how much the predictive distribution shifts, and when it stabilizes. By turning these qualitative reasoning behaviors into measurable properties, dynamical profiles provide candidate targets for shaping reasoning behavior through training or inference-time interventions.

## 4.2 Mechanisms of success and failure

We now combine the evolution of the gold answer with the trajectory’s outcome to define various dynamical mechanisms of success and failure. Let $a ^ { \star }$ denote the gold answer and $D _ { k }$ the set of dominant answers at reasoning position k. We say that the gold answer is dominant at k whenever $a ^ { \star } \in D _ { k }$ , including when it is tied with another answer. We quantify the prevalence of these mechanisms across models and tasks in Section 3.2.

Success mechanisms. Conditioning on realized endpoint correctness, correct traces separate into four dynamical mechanisms:

• Stable success. The gold answer remains dominant throughout the observed trajectory.

• Rescue. The gold answer is not initially dominant but is dominant at the end of the trajectory.

• Detour success. The gold answer is dominant initially and at the end of the trajectory, but temporarily loses dominance.

• Fragile success. The realized answer is correct even though the gold answer is not dominant at the end of the trajectory.

Failure mechanisms. Incorrect traces separate into four complementary failure mechanisms:

• Never discovered. The gold answer is never dominant throughout the observed trajectory.

• Escape. The gold answer is initially dominant but is not dominant at the end of the trajectory.

• Failed rescue. The gold answer is not initially dominant, becomes dominant at some point during the trajectory, but is not dominant at the end.

• Stochastic miss. The realized answer is incorrect even though the gold answer is dominant at the end of the trajectory.

Several of these mechanisms are informative about overthinking [7, 48, 4]: escape and failed rescue capture cases in which continued reasoning moves the model away from a state where the correct answer was dominant, while detour success captures cases in which this degradation is temporary.

![](images/b7dfb2d63c9c2f4450d4af90582292e6e687a2f5dae3a1a79b95113aa6c83eeb.jpg)  
(a) Success mechanisms among correct traces.

![](images/81cfa163cc539f818965cc475baa122748541aec4ce19f0994bd3798c0ddc485.jpg)  
(b) Failure mechanisms among incorrect traces.  
Figure 2: Success and failure mechanisms by model, with rows sorted by endpoint accuracy. Models with similar endpoint accuracy can exhibit substantially different mixtures of these mechanisms.

An important observation is that the same dynamical pattern can produce different outcomes depending on which hypotheses the trajectory explores and ultimately favors. Nevertheless, some dynamical strategies may increase the likelihood of success for a given model, task, or objective. Characterizing both profiles and mechanisms therefore provides a vocabulary for identifying advantageous reasoning behaviors and, ultimately, for encouraging them through training or inference-time interventions.

## 5 Results

In this section, we conduct a broad set of experiments to determine whether answer-distribution trajectories capture information missed by conventional summaries of reasoning, and what additional insights they provide. We organize our empirical validation section around three questions: (i) can traces with the same endpoint or similar entropy profiles exhibit different answer-distribution dynamics, (ii) how do reasoning dynamics vary across traces, models, and tasks, and (iii) how do instruction tuning, scale, and decoding temperature reshape these dynamics?

We evaluate sixteen models across four datasets (GSM8K, ARC, SVAMP, and MATH) spanning base, instruction-tuned, and RL-trained regimes. We estimate all entropy quantities via Monte Carlo rollouts under stochastic decoding. Full evaluation details are provided in Appendix A and full reproduction details for figures and tables are provided in Appendix B.

## 5.1 Coarse summaries collapse distinct reasoning dynamics

Our theoretical analysis in Section 3.2 showed that answer-distribution trajectories determine both endpoint predictions and entropy profiles, whereas the converse does not hold. We now test whether this non-identifiability arises in practice.

Same endpoint, different dynamics. Figure 2 compares the prevalence of the success and failure mechanisms defined in Section 4.2 across models, with models ordered by endpoint accuracy. Models with similar accuracy can nevertheless exhibit markedly different mixtures of success and failure mechanisms, as seen among neighboring rows in the accuracy-ordered panels. Endpoint performance therefore does not identify the dynamical processes through which successes and failures arise.

Same entropy profile, different dynamics. Figure 3 matches trajectories for the same question according to the similarity of their entropy profiles and measures disagreement in properties of the corresponding answer-distribution trajectories. Even for traces with similar entropy profiles, trajectories can disagree about which answers are dominant, whether the gold answer is dominant, and which success or failure mechanism the trajectory instantiates. Entropy therefore does not identify the underlying answer-distribution dynamics.

These results confirm that answer-distribution trajectories retain information lost by conventional summaries, motivating their use as a richer diagnostic of reasoning dynamics. In the following sections, we use them to characterize variation across traces, models, and tasks, and to study how training and inference choices reshape that variation.

![](images/4af74cc0eb021bfadefb5225efa868abc0558c8a8d5ff65e0e01f03d3c3fe790.jpg)  
Figure 3: We compare cross-model traces for the same question using the RMSE between their entropy profiles over normalized reasoning time. The displayed range up to $\mathrm { R M S E } = 0 . 2 0$ corresponds to 12% of the largest same-question cross-model RMSE observed in our data. Similar entropy evolution does not imply similar answer-distribution dynamics.

Table 1: Magnitude and sources of variation in trajectory-level reasoning metrics. Entries report root variance components in the natural units of each metric. The total variation is obtained by summing the squared components and taking the square root. Model-task pairs are weighted equally.
<table><tr><td>Source of variation</td><td> $S _ { \mathrm { e f f } }$ </td><td> $N _ { \mathrm { s w i t c h } } / \mathrm { s t e p }$ </td><td> $N _ { \mathrm { r e t u r n } } / \mathrm { s t e p }$ </td><td> $V / \mathrm { s t e p }$ </td><td>PR</td><td> $R _ { \mathrm { d i r e c t } }$ </td><td> $t _ { \mathrm { c o m m i t } } ( 0 . 8 )$ </td></tr><tr><td>Model</td><td>0.529</td><td>0.160</td><td>0.021</td><td>0.164</td><td>0.211</td><td>0.159</td><td>0.160</td></tr><tr><td>Task</td><td>0.323</td><td>0.079</td><td>0.017</td><td>0.068</td><td>0.070</td><td>0.087</td><td>0.065</td></tr><tr><td>Model x task</td><td>0.466</td><td>0.129</td><td>0.034</td><td>0.103</td><td>0.092</td><td>0.104</td><td>0.167</td></tr><tr><td>Within model-task</td><td>1.09</td><td>0.270</td><td>0.123</td><td>0.204</td><td>0.254</td><td>0.299</td><td>0.322</td></tr><tr><td>Total</td><td>1.34</td><td>0.348</td><td>0.131</td><td>0.290</td><td>0.350</td><td>0.365</td><td>0.402</td></tr></table>

## 5.2 Reasoning dynamics vary within and across models and tasks

The dynamical profiles introduced in Section 4.1 provide a coordinate system for comparing reasoning trajectories. We now ask whether traces exhibit distinct profiles within and across models and tasks, and whether this variation indicates which reasoning dynamics are better suited to different objectives.

Individual traces exhibit substantial variation. Reasoning trajectories can differ substantially in their dynamical profiles. To identify the sources of this variation, in Table 1 we decompose each trajectory-level metric into four components: variation across model-level means, variation across task-level means, model-task interaction, and variation across individual traces within a fixed model-task condition. Every reported variance component is nonzero, indicating heterogeneity in each metric. Notably, the within-condition component is the largest for every metric, which indicates that a model-task pair does not correspond to a single characteristic reasoning profile. This suggests that the metrics capture meaningful variation at the level of individual reasoning traces and are not limited to distinguishing between models or tasks.

Which dynamical profile is favorable depends on the objective being optimized. As shown in Table 2, model-task pairs with the highest endpoint accuracy tend to exhibit more focused reasoning dynamics: they maintain narrower predictive support, switch dominant hypotheses less frequently, undergo less local distributional movement, and commit earlier than the task average. In contrast, when the goal is efficiency, the shortest-CoT traces exhibit broader predictive support, more frequent hypothesis switching, greater and more temporally distributed local distributional movement, and later commitment. Thus, no single reasoning profile is universally favorable.

These examples illustrate why dynamical profiles can be useful beyond describing heterogeneity. By turning qualitative reasoning behaviors into measurable properties, they provide coordinates to evaluate which dynamical profiles are advantageous under a particular model, task, and objective.

Table 2: Different objectives favor different dynamical profiles. Standardized mean profiles for high-accuracy and short-CoT model-task pairs. Values denote deviations from the task mean in standard deviation units.
<table><tr><td>Objective</td><td> $S _ { \mathrm { e f f } }$ </td><td> $N _ { \mathrm { s w i t c h } } / \mathrm { s t e p }$ </td><td> $N _ { \mathrm { r e t u r n } } / \mathrm { s t e p }$ </td><td> $V / \mathrm { s t e p }$ </td><td>PR</td><td> $R _ { \mathrm { d i r e c t } }$ </td><td> $t _ { \mathrm { c o m m i t } } ( 0 . 8 )$ </td><td>Acc.</td><td>CoT length</td></tr><tr><td>High accuracy</td><td>-0.83</td><td>-0.68</td><td>-0.50</td><td>-0.85</td><td>-0.92</td><td>-0.14</td><td>-0.53</td><td>78.8%</td><td>240</td></tr><tr><td>Short CoT</td><td>+1.09</td><td>+1.03</td><td>-0.45</td><td>+1.23</td><td>+1.27</td><td>+1.30</td><td>+0.76</td><td>18.6%</td><td>56</td></tr></table>

![](images/d235efc862e2e2ccdebad089e2572b98173be5ba1cd9d73aee8b642809e6ce29.jpg)  
Figure 4: Training and inference choices reshape reasoning dynamics. Standardized changes in endpoint accuracy and trajectory-level metrics under matched comparisons of instruction tuning, increasing model scale, and increasing decoding temperature. Each point represents a matched model-task comparison; diamonds show mean effects and error bars denote 95% confidence intervals. Changes are standardized by the across-condition standard deviation of the corresponding metric.

## 5.3 Training and inference choices reshape reasoning dynamics

The previous section shows that traces can present vastly different reasoning profiles and that the desirability of each of them depends on what we seek to optimize. We next ask whether choices made during training or inference time can steer models towards presenting specific dynamical profiles.

Figure 4 reports the change in each trajectory-level metric for every matched comparison, together with the aggregate effect. Instruction tuning produces the clearest dynamical shift, yielding narrower predictive support $( S _ { \mathrm { e f f } } )$ , fewer hypothesis switches $( N _ { \mathrm { s w i t c h } } / \mathrm { s t e p } )$ , less distributional movement $( V / \mathrm { s t e p } )$ , earlier commitment $( t _ { \mathrm { c o m m i t } } ( 0 . 8 ) )$ , and higher endpoint accuracy. Increasing model scale shows a weaker, less uniform tendency toward shifts in the same direction, with improved accuracy. In contrast, increasing decoding temperature broadens predictive support $( S _ { \mathrm { e f f } } )$ , increases distributional movement $( V / \mathrm { s t e p } )$ , and delays commitment $( t _ { \mathrm { c o m m i t } } ( 0 . 8 ) )$ , while its effect on hypothesis switching $( N _ { \mathrm { s w i t c h } } / \mathrm { s t e p } )$ is smaller and accuracy changes are model-dependent. These results suggest that training and inference-time interventions induce systematic average shifts in the dynamical profiles of the generated reasoning traces.

## 6 Conclusion and Open Questions

This work introduces answer-distribution trajectories as a framework for studying how a language model’s predictive distribution over final answers evolves throughout CoT reasoning. Unlike endpoint predictions or entropy profiles, these trajectories preserve both which hypotheses are supported and how probability mass moves among them. We derive trajectory-level measures of exploration, revision, motion, and commitment and combine them into dynamical profiles. We use these profiles to characterize reasoning dynamics across sixteen open-weight language models and four reasoning benchmarks, finding variation in how final answers are reached both within and across model-task pairs. We also show that different objectives favor different dynamical profiles, and that training and inference choices systematically reshape them. These results position the evolution of the full predictive distribution as an informative characterization of LLM reasoning, and stochastic dynamics as a natural mathematical language for describing the paths by which LLMs arrive at their answers.

Some open questions remain. Our analysis is computationally expensive, developing cheaper approximations of answer-distribution trajectories is therefore an important direction for future work. Moreover, the comparisons in Section 5.3 are only descriptive, and future controlled studies could better isolate the effects of architecture, scale, and post-training. Lastly, whether the information that answer-distribution trajectories provide can translate into practical gains remains an open question. Promising directions include improving error prediction and adaptive test-time compute, diagnosing and mitigating overthinking, and informing reasoning-model optimization and post-training.

## Acknowledgements

Mar Gonzàlez I Català acknowledges that this project was supported by G-Research and by the Qualcomm Innovation Fellowship Europe. Davide Murari acknowledges support from the EPSRC grant EP/Y028783/1. George D. Montañez acknowledges support from the William Whewell Centre for Science and Natural Theology and the Global Scholars Foundation.

## References

[1] Iván Arcuschin et al. “Chain-of-Thought Reasoning in the Wild is not Always Faithful”. In: Workshop on Reasoning and Planning for Large Language Models. 2025. URL: https: //openreview.net/forum?id=L8094Whth0.

[2] Amos Azaria and Tom Mitchell. “The Internal State of an LLM Knows When It’s Lying”. In: The 2023 Conference on Empirical Methods in Natural Language Processing. 2023. URL: https://openreview.net/forum?id=y2V6YgLaW7.

[3] Bowen Baker et al. “Monitoring reasoning models for misbehavior and the risks of promoting obfuscation”. In: arXiv preprint arXiv:2503.11926 (2025).

[4] Simone Caldarella et al. Thinking Past the Answer: Evaluating Harmful Overthinking in Large Reasoning Models. 2026. arXiv: 2606.02835 [cs.AI]. URL: https://arxiv.org/abs/ 2606.02835.

[5] Mar Gonzàlez I Català et al. The Stepwise Informativeness Assumption: Why are Entropy Dynamics and Reasoning Correlated in LLMs? 2026. arXiv: 2604.06192 [cs.CL]. URL: https://arxiv.org/abs/2604.06192.

[6] Chao Chen et al. INSIDE: LLMs’ Internal States Retain the Power ofHallucination Detection. 2024. arXiv: 2402.03744 [cs.CL]. URL: https://arxiv.org/abs/2402.03744.

[7] Xingyu Chen et al. “Do NOT Think That Much for 2+3=? On the Overthinking of Long Reasoning Models”. In: Proceedings of the 42nd International Conference on Machine Learning. Ed. by Aarti Singh et al. Vol. 267. Proceedings of Machine Learning Research. PMLR, 13–19 Jul 2025, pp. 9487–9499. URL: https://proceedings.mlr.press/v267/chen25bx.html.

[8] Peter Clark et al. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. 2018. arXiv: 1803.05457 [cs.AI]. URL: https://arxiv.org/abs/1803. 05457.

[9] Karl Cobbe et al. Training Verifiers to Solve Math Word Problems. 2021. arXiv: 2110.14168 [cs.LG]. URL: https://arxiv.org/abs/2110.14168.

[10] DeepSeek-AI et al. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism. 2024. arXiv: 2401.02954 [cs.CL]. URL: https://arxiv.org/abs/2401.02954.

[11] Siddartha Devic et al. Trace Length is a Simple Uncertainty Signal in Reasoning Models. 2025. arXiv: 2510.10409 [cs.AI]. URL: https://arxiv.org/abs/2510.10409.

[12] Scott Emmons et al. When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors. 2025. arXiv: 2507.05246 [cs.AI]. URL: https://arxiv.org/abs/ 2507.05246.

[13] Sebastian Farquhar et al. “Detecting hallucinations in large language models using semantic entropy”. In: Nature 630.8017 (2024), pp. 625–630.

[14] Gemma Team et al. Gemma 2: Improving Open Language Models at a Practical Size. 2024. arXiv: 2408.00118 [cs.CL]. URL: https://arxiv.org/abs/2408.00118.

[15] Aaron Grattafiori et al. The Llama 3 Herd of Models. 2024. arXiv: 2407.21783 [cs.AI]. URL: https://arxiv.org/abs/2407.21783.

[16] Daya Guo et al. “DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning”. In: Nature 645.8081 (Sept. 2025), pp. 633–638. ISSN: 1476-4687. DOI: 10 . 1038 / s41586-025-09422-z. URL: http://dx.doi.org/10.1038/s41586-025-09422-z.

[17] Wes Gurnee et al. “Verbalizable Representations Form a Global Workspace in Language Models”. In: Transformer Circuits Thread (2026). URL: https://transformer-circuits. pub/2026/workspace/index.html.

[18] Dan Hendrycks et al. Measuring Mathematical Problem Solving With the MATH Dataset. 2021. arXiv: 2103.03874 [cs.LG]. URL: https://arxiv.org/abs/2103.03874.

[19] Parsa Hosseini et al. “Early Stopping for Large Reasoning Models via Confidence Dynamics”. In: arXiv preprint arXiv:2604.04930 (2026).

[20] Eyke Hüllermeier and Willem Waegeman. “Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods”. In: Machine learning 110.3 (2021), pp. 457–506.

[21] Takeshi Kojima et al. “Large language models are zero-shot reasoners”. In: Advances in neural information processing systems 35 (2022), pp. 22199–22213.

[22] Tomek Korbak et al. “Chain of thought monitorability: A new and fragile opportunity for ai safety”. In: arXiv preprint arXiv:2507.11473 (2025).

[23] Tamera Lanham et al. “Measuring faithfulness in chain-of-thought reasoning”. In: arXiv preprint arXiv:2307.13702 (2023).

[24] Zeju LI et al. “Making Slow Thinking Faster: Compressing LLM Chain-of-Thought via Step Entropy”. In: International Conference on Learning Representations. Ed. by C. Vondrick et al. Vol. 2026. 2026, pp. 38331–38349. URL: https://proceedings.iclr.cc/paper\_files/ paper/2026/file/40b2844d3c1451a90e0afb193bd735fc-Paper-Conference.pdf.

[25] Hunter Lightman et al. “Let’s verify step by step”. In: International Conference on Learning Representations. Vol. 2024. 2024, pp. 39578–39601.

[26] Yujun Mao, Yoon Kim, and Yilun Zhou. “Champ: A competition-level dataset for fine-grained analyses of llms’ mathematical reasoning capabilities”. In: Findings of the Association for Computational Linguistics: ACL 2024. 2024, pp. 13256–13274.

[27] Samuel Marks and Max Tegmark. The Geometry ofTruth: Emergent Linear Structure in Large Language Model Representations ofTrue/False Datasets. 2024. arXiv: 2310.06824 [cs.AI]. URL: https://arxiv.org/abs/2310.06824.

[28] OpenAI et al. OpenAI o1 System Card. 2026. arXiv: 2412.16720 [cs.AI]. URL: https: //arxiv.org/abs/2412.16720.

[29] Koyena Pal et al. “Future lens: Anticipating subsequent tokens from a single hidden state”. In: Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL). 2023, pp. 548–560.

[30] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP Models really able to Solve Simple Math Word Problems? 2021. arXiv: 2103.07191 [cs.CL]. URL: https://arxiv. org/abs/2103.07191.

[31] Archiki Prasad et al. “Receval: Evaluating reasoning chains via correctness and informativeness”. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. 2023, pp. 10066–10086.

[32] Chen Qian et al. Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning. 2025. arXiv: 2506.02867 [cs.AI]. URL: https://arxiv.org/abs/2506.02867.

[33] Qwen et al. Qwen2.5 Technical Report. 2025. arXiv: 2412.15115 [cs.CL]. URL: https: //arxiv.org/abs/2412.15115.

[34] Aman Sharma and Paras Chopra. Think Just Enough: Sequence-Level Entropy as a Confidence Signal for LLM Reasoning. 2025. arXiv: 2510.08146 [cs.LG]. URL: https://arxiv.org/ abs/2510.08146.

[35] Jean-Francois Ton, Muhammad Faaiz Taufiq, and Yang Liu. Understanding Chain-of-Thought in LLMs through Information Theory. 2025. arXiv: 2411.11984 [cs.CL]. URL: https: //arxiv.org/abs/2411.11984.

[36] Alexander Matt Turner et al. Steering Language Models With Activation Engineering. 2024. arXiv: 2308.10248 [cs.CL]. URL: https://arxiv.org/abs/2308.10248.

[37] Miles Turpin et al. “Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 74952–74965.

[38] Jonathan Uesato et al. “Solving math word problems with process-and outcome-based feedback”. In: arXiv preprint arXiv:2211.14275 (2022).

[39] Shenzhi Wang et al. “Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning”. In: Advances in Neural Information Processing Systems. Ed. by D. Belgrave et al. Vol. 38, Main Conference. Curran Associates, Inc., 2025, pp. 115452–115486. DOI: 10 . 52202 / 085713 - 3850. URL: https : / / proceedings . neurips . cc / paper \_ files / paper / 2025 / file / a797c2d2e0c1fdabf4d1ab8cd0b465c6-Paper-Conference.pdf.

[40] Xuezhi Wang et al. “Self-Consistency Improves Chain of Thought Reasoning in Language Models”. In: The Eleventh International Conference on Learning Representations. 2023. URL: https://openreview.net/forum?id=1PL1NIMMrw.

[41] Zihan Wang et al. “Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision”. In: Findings ofthe Associationfor Computational Linguistics: EMNLP 2024. 2024, pp. 7309–7319.

[42] Jason Wei et al. “Chain-of-thought prompting elicits reasoning in large language models”. In: Advances in neural information processing systems 35 (2022), pp. 24824–24837.

[43] Lisa Wimmer et al. “Quantifying aleatoric and epistemic uncertainty in machine learning: Are conditional entropy and mutual information appropriate measures?” In: Uncertainty in artificial intelligence. PMLR. 2023, pp. 2282–2292.

[44] Shijie Xia et al. “Evaluating mathematical reasoning beyond accuracy”. In: Proceedings ofthe AAAI Conference on Artificial Intelligence. Vol. 39. 26. 2025, pp. 27723–27730.

[45] Wei Xia et al. “When Do LLMs Reason? A Dynamical Systems View via Entropy Phase Transitions”. In: arXiv preprint arXiv:2605.22873 (2026).

[46] 01. AIand Alex Young et al. Yi: Open Foundation Models by 01.AI. 2025. arXiv: 2403.04652 [cs.CL]. URL: https://arxiv.org/abs/2403.04652.

[47] Jinghan Zhang et al. “Entropy-based exploration conduction for multi-step reasoning”. In: Findings of the Association for Computational Linguistics: ACL 2025. 2025, pp. 3895–3906.

[48] Shu Zhou et al. “When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling”. In: Findings ofthe Associationfor Computational Linguistics: ACL 2026. Ed. by Maria Liakata et al. San Diego, California, United States: Association for Computational Linguistics, July 2026, pp. 23967–23977. ISBN: 979-8-89176-395-1. DOI: 10.18653/v1/ 2026 . findings - acl . 1199. URL: https : / / aclanthology . org / 2026 . findings - acl.1199/.

[49] Andy Zou et al. Representation Engineering: A Top-Down Approach to AI Transparency. 2025. arXiv: 2310.01405 [cs.LG]. URL: https://arxiv.org/abs/2310.01405.

## A Experimental Setup

## A.1 Tasks and datasets

We focus on reasoning tasks with a discrete answer space A. Each example consists of a question $Q \in \mathcal { Q }$ and a ground-truth answer $A \in { \mathcal { A } }$ . We evaluate on the following datasets:

• GSM8K [9]: grade-school mathematical word problems with numeric answers.

• ARC [8]: multiple-choice science questions.

• SVAMP [30]: arithmetic word problems designed to test robustness to linguistic variation.

• MATH [18]: competition-level mathematics (we use algebra track).

For all datasets, we use the official test splits and apply deterministic answer normalization and parsing to map model outputs to discrete answer labels (e.g., numeric normalization for GSM8K, SVAMP and MATH, letter-to-option mapping for ARC). Invalid or unparsable outputs are mapped to a special null answer category.

## A.2 Models

We evaluate a diverse set of sixteen open-weight LLMs corresponding to different training regimes:

• Gemma-2-2B and Gemma-2-9B [14]: base and instruction-tuned variants for each.

• LLaMA-3.2-3B and LLaMA-3.1-8B [15]: base and instruction-tuned variants for each.

• Qwen-2.5-3B and Qwen-2.5-14B [33]: base and instruction-tuned variants for each.

• Qwen-2.5-Math-1.5B [33]: SFT-trained specialized on math problems.

• DeepSeek-Chat-7B [10]: SFT-trained chat model.

• DeepSeek-R1-distilled-7B [16]: reasoning-specialized RL model.

• Yi-1.5-34B [46]: base variant.

Base models correspond to pretrained LLMs without supervised or reinforcement fine-tuning. Instruction-tuned (IT) models are supervised fine-tuned on instruction-following data. RL-trained models are optimized using reinforcement learning from human or synthetic feedback.

## A.3 Generation procedure

For each question $Q = q .$ , we sample M independent reasoning trajectories from the model under a fixed stochastic decoding configuration (temperature, nucleus sampling, and maximum generation length). Concretely, for each $i \in \{ 1 , \ldots , M \}$ we draw

$$
C _ { 1 : K ^ { ( i ) } } ^ { ( i ) } \sim p _ { \theta } ( \cdot \mid q ) ,
$$

where $K ^ { ( i ) }$ denotes the generated reasoning length (up to a fixed truncation limit). We treat each sampled trajectory $C _ { 1 : K ^ { ( i ) } } ^ { ( i ) }$ as one realization of the model’s reasoning process for the given query. Unless otherwise specified, decoding uses:

• temperature $T = 0 . 7$

• nucleus sampling with $p = 0 . 9$

• a maximum generation length of 600 tokens

Each trajectory is treated as one realization of the model’s reasoning process for the given query. All continuation rollouts used to estimate prefix-conditioned answer distributions use the same decoding configuration to ensure comparability.

## A.4 Monte-Carlo estimation of prefix-conditioned answer distributions

Given a fixed query $Q = q$ and a realized reasoning prefix $C _ { 1 : k } = c _ { 1 : k }$ , the model induces a predictive distribution over discrete final-answer labels,

$$
p _ { k } ( a ) = p _ { \theta } ( Y = a \mid q , c _ { 1 : k } ) , \qquad a \in \mathcal { A } ,
$$

where $Y$ denotes the discrete answer label obtained by applying the deterministic answer parser to a generated continuation.

Because this distribution marginalizes over all possible future reasoning continuations, we approximate it using Monte-Carlo sampling. For each fixed prefix $\left( q , c _ { 1 : k } \right)$ , we draw N independent stochastic continuations from the model,

$$
C _ { > k } ^ { ( i ) } , A ^ { ( i ) } \sim p _ { \theta } ( \cdot  { | } \ q , c _ { 1 : k } ) , \quad \quad i = 1 , \dots , N ,
$$

using the same decoding configuration as the original reasoning trajectory. Each generated answer sequence $A ^ { ( i ) }$ is mapped by the deterministic parser to a discrete answer label $Y ^ { ( i ) } \in { \mathcal { A } }$ , including the special null category for invalid or unparsable outputs.

The samples define the empirical prefix-conditioned answer distribution

$$
\widehat { p } _ { k } ( a ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \Big \{ Y ^ { ( i ) } = a \Big \} , \qquad a \in \mathcal { A } .
$$

We use $\widehat { p } _ { k }$ as the empirical approximation to the prefix-conditioned predictive state $p _ { k }$ throughout our trajectory analysis. All trajectory-level quantities are computed from these empirical predictive distributions.

All continuation rollouts are performed in evaluation mode without gradient computation, and sampling parameters are held fixed across models and prefixes. In practice, we use $N ~ = ~ 1 6$ independent continuations per prefix unless otherwise stated.

## A.5 Checkpointed prefix evaluation

Estimating conditional answer entropy at every token is computationally expensive. We therefore evaluate at checkpoint positions

$$
{ \mathcal { I } } = \{ j _ { 1 } , j _ { 2 } , \ldots , j _ { m } \} \subseteq \{ 0 , 1 , \ldots , K \} ,
$$

spaced uniformly at stride $s = 1 6 ,$ , and always including the final prefix length of the trajectory $( j _ { m } = K )$ . The checkpoint at position 0 corresponds to the empty prefix.

## B Reproduction details for figures and tables

This section documents the exact construction of the data-driven figures and tables in the main text.

Common trace processing. The analyses use the complete $4 \times 1 6$ design consisting of ARC, GSM8K, MATH, and SVAMP crossed with the sixteen paper models. Answer parsing, normalization, and correctness evaluation follow the procedure described in Section A.1.

Trajectory-level metrics are defined in Section 4.1. In all analyses below, these metrics are computed from the observed answer-distribution states at checkpoint positions with a nonempty predictive distribution. $R _ { \mathrm { d i r e c t } }$ is undefined for trajectories with zero total path length and is omitted from summaries of that metric. Commitment uses the threshold $\tau = 0 . 8$ and is undefined when no hypothesis remains above this threshold through the end of the observed trajectory; such traces are omitted from summaries of $t _ { \mathrm { c o m m i t } }$

Figure 2: success and failure mechanisms. Each trace is first split by realized endpoint correctness. Correct and incorrect traces are assigned to the success and failure mechanisms defined in Section 4.2 based on the trajectory of gold-answer dominance. At each checkpoint, the gold answer is considered dominant whenever it belongs to the set of probability-maximizing answers, including in the case of an exact tie. For each model, the left panel reports the fraction of correct traces in each success mechanism, and the right panel reports the corresponding fractions among incorrect traces. The rows are ordered by the model’s pooled endpoint accuracy across the four default tasks. The percentage displayed next to each model name is pooled accuracy in the success panel and pooled failure rate in the failure panel.

Figure 3: near-matched entropy profiles. We compare reasoning traces from different models only when they correspond to the same dataset example. Because traces can have different lengths and checkpoints at different positions, we rescale reasoning progress from 0 (start) to 1 (end) and interpolate each entropy profile onto the same 101 evenly spaced points. We then measure how similar two entropy profiles are using root mean squared error (RMSE), where smaller values indicate more similar profiles. At each normalized position, the dominant set and gold-dominance indicator are taken from the most recent observed checkpoint.

For every cross-model pair, we record three disagreement measures: the fraction of normalized-time grid points with different dominant sets, the fraction with different gold-dominance indicators, and an indicator for whether the two traces instantiate different success or failure mechanisms. Mechanism disagreement is evaluated only when the two traces have the same realized outcome and both receive a mechanism label. We group non-identical trace pairs into non-overlapping RMSE bins of width 0.01, from (0, 0.01] through (0.19, 0.20], so that each pair contributes to exactly one bin. Exact entropy-profile matches, defined as $\mathrm { R M } \bar { \mathrm { S } } \mathrm { E } \leq 1 0 ^ { - 1 2 }$ , are reported separately at zero. Each plotted point shows the mean disagreement among the pairs in that RMSE bin, expressed as a percentage, and is positioned at the midpoint of the corresponding bin.

Table 2: variance decomposition. Values are computed separately for each of the seven trajectorylevel profile metrics. Let $\mu _ { m t }$ denote the mean of a metric in model m and task $t ,$ let $\mu _ { m }$ and $\mu . _ { t }$ be the corresponding marginal means, and let $\mu$ be the grand mean across the 64 model-task cells. The between-condition components are

$$
\sigma _ { \mathrm { m o d e l } } ^ { 2 } = { \frac { 1 } { M } } \sum _ { m } ( \mu _ { m } . - \mu ) ^ { 2 } , \qquad \sigma _ { \mathrm { t a s k } } ^ { 2 } = { \frac { 1 } { T } } \sum _ { t } ( \mu . _ { t } - \mu ) ^ { 2 } ,
$$

$$
\sigma _ { \mathrm { i n t e r a c t i o n } } ^ { 2 } = { \frac { 1 } { M T } } \sum _ { m , t } ( \mu _ { m t } - \mu _ { m } . - \mu . t + \mu ) ^ { 2 } .
$$

The within-condition component is the equally weighted average of the population variances within the 64 model-task cells,

$$
\sigma _ { \mathrm { w i t h i n } } ^ { 2 } = { \frac { 1 } { M T } } \sum _ { m , t } \operatorname { V a r } ( X \mid m , t ) .
$$

The total variance is the sum of these four components. The table reports the square root of each component so that every entry remains in the natural units of the corresponding metric. Each model-task cell receives equal weight regardless of the number of traces in that cell.

Table 2: objective-conditioned dynamical profiles. The trace-level data are first aggregated to one row for each of the 64 default model-task conditions. Mean CoT length is the mean number of generated tokens. CoT length is log transformed before standardization. Endpoint accuracy, log CoT length, and each profile metric are then standardized separately within each task across the sixteen model conditions using population standard deviation.

The High accuracy group is the 25% of all model-task pairs with the largest within-task standardized accuracy. The Short $C o T$ group is the 25% with the smallest within-task standardized log CoT length. The reported profile entries are the average standardized values of the seven trajectory metrics within each selected group, where each metric is expressed relative to the mean and standard deviation for that task. The final two columns report unstandardized mean endpoint accuracy and unstandardized mean CoT length for interpretability.

Figure 4: training and inference interventions. Figure 4 operates on model-task condition means. For instruction tuning, we compare six matched base and instruction-tuned model pairs: Gemma-2-2B vs. Gemma-2-2B-IT, Gemma-2-9B vs. Gemma-2-9B-IT, LLaMA-3.2-3B vs. LLaMA-3.2-3B-IT, LLaMA-3.1-8B vs. LLaMA-3.1-8B-IT, Qwen-2.5-3B vs. Qwen-2.5-3B-IT, and Qwen-2.5-14B vs.

Qwen-2.5-14B-IT. Each pair is evaluated on all four tasks. For model scale, we compare Gemma-2- 2B vs. Gemma-2-9B, Gemma-2-2B-IT vs. Gemma-2-9B-IT, LLaMA-3.2-3B vs. LLaMA-3.1-8B, LLaMA-3.2-3B-IT vs. LLaMA-3.1-8B-IT, Qwen-2.5-3B vs. Qwen-2.5-14B, and Qwen-2.5-3B-IT vs. Qwen-2.5-14B-IT, again evaluated on all four tasks. For decoding temperature, we use GSM8K runs from DeepSeek-R1, Gemma-2-2B-IT, Qwen-2.5-3B, and Qwen-2.5-3B-IT, comparing $T = 0 . 2$ with $T = 1 . 0 .$

For every task-specific matched comparison and metric, the raw change is defined as the intervened condition minus the reference condition. To place metrics with different units on a common horizontal scale, each raw change is divided by the sample standard deviation of that metric across the 64 defaultdecoding model-task condition means. For instruction tuning and model scale, these standardized changes are then averaged across the four tasks within each matched model pair. Each plotted point therefore represents one matched model-pair comparison averaged across tasks. For temperature, each point represents one model comparison on GSM8K.

The diamond is the arithmetic mean across matched model comparisons. Error bars are two-sided 95% confidence intervals computed as the mean plus or minus the Student-t critical value times the standard error, with degrees of freedom $n - 1$ . Thus, $n = 6$ for the instruction-tuning and model-scale panels and $n = 4$ for the temperature panel. Vertical jitter is visual only and is controlled by a fixed random seed. The three panels share a symmetric x-axis range so that effect magnitudes are visually comparable across instruction tuning, scale, and temperature.

## C Proofs

Proof of Proposition 1. The endpoint prediction $E ( \mathcal T )$ depends only on the terminal state $p _ { K }$ , while each component of $\mathcal { H } ( \mathcal { T } )$ depends only on the corresponding state $p _ { k }$ . Therefore both are uniquely determined by $\tau$ 口

Proof of Proposition 2. Both claims can be witnessed on the binary answer space $\mathcal { A } = \{ a , b \}$

For the first claim, consider

$$
\begin{array} { r } { \mathcal { T } _ { 1 } : ( 0 . 9 , 0 . 1 )  ( 0 . 9 , 0 . 1 )  ( 0 . 9 , 0 . 1 ) , } \end{array}
$$

and

$$
\begin{array} { r } { \mathcal { T } _ { 2 } : ( 0 . 1 , 0 . 9 )  ( 0 . 4 , 0 . 6 )  ( 0 . 9 , 0 . 1 ) . } \end{array}
$$

Both terminate with a as the dominant answer, and therefore

$$
E ( \mathcal { T } _ { 1 } ) = E ( \mathcal { T } _ { 2 } ) = a .
$$

Nevertheless, $\mathcal { T } _ { 1 }$ maintains the same dominant hypothesis throughout, whereas $\mathcal { T } _ { 2 }$ revises from b to a.   
Hence the endpoint does not identify the preceding distribution dynamics.

For the second claim, take

$$
p = ( 0 . 9 , 0 . 1 ) , \qquad q = ( 0 . 1 , 0 . 9 ) ,
$$

and consider the constant trajectories

$$
\begin{array} { r } { T _ { 1 } : p \to p \to p , \qquad \mathcal { T } _ { 2 } : q \to q \to q . } \end{array}
$$

Since entropy is invariant under permutation of coordinates,

$$
H ( p ) = H ( q ) ,
$$

and therefore

$$
\begin{array} { r } { \mathcal { H } ( \mathcal { T } _ { 1 } ) = \mathcal { H } ( \mathcal { T } _ { 2 } ) . } \end{array}
$$

However, $\mathcal { T } _ { 1 } \neq \mathcal { T } _ { 2 }$ , since $p \neq q$ . Hence entropy profiles do not identify which hypotheses carry the predictive mass. □

## D Licenses

We do not introduce or release any new datasets or model checkpoints. All experiments use publicly released benchmark datasets and open-weight model checkpoints under their respective licenses. We use these artifacts only for academic evaluation and do not redistribute the original dataset contents or model weights. Table 3 summarizes the licenses and usage terms for all artifacts used in our experiments.

Table 3: Licenses and usage terms for datasets and open-weight model checkpoints used in our experiments.
<table><tr><td>Artifact</td><td>Source</td><td>License / terms</td></tr><tr><td>GSM8K</td><td>OpenAI</td><td>MIT License</td></tr><tr><td>ARC</td><td>AllenAI AI2 ARC</td><td>CC BY-SA 4.0</td></tr><tr><td>SVAMP</td><td>Patel et al. / HF mirror</td><td>MIT License</td></tr><tr><td>MATH</td><td>Hendrycks et al.</td><td>MIT License</td></tr><tr><td>Gemma-2-2B, Gemma-2-9B</td><td>Google</td><td>Gemma Terms of Use</td></tr><tr><td>LLaMA-3.2-3B</td><td>Meta</td><td>Llama 3.2 Community License</td></tr><tr><td>LLaMA-3.1-8B</td><td>Meta</td><td>Llama 3.1 Community License</td></tr><tr><td>Qwen-2.5-3B</td><td>Qwen</td><td>Qwen Research</td></tr><tr><td>Qwen-2.5-14B</td><td>Qwen</td><td>Apache 2.0</td></tr><tr><td>Qwen-2.5-Math-1.5B</td><td>Qwen</td><td>Apache 2.0</td></tr><tr><td>DeepSeek-Chat-7B</td><td>DeepSeek</td><td>DeepSeek Model License; code under MIT</td></tr><tr><td>DeepSeek-R1-distilled-7B</td><td>DeepSeek</td><td>MIT License</td></tr><tr><td>Yi-1.5-34B</td><td>01.AI</td><td>Apache 2.0</td></tr></table>

## E Impact Statement

This paper aims to advance the field of Machine Learning. While our work has potential societal implications, we do not identify any specific concerns that require particular emphasis at this stage.