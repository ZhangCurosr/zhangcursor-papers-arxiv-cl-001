# Difficulty-Adaptive Tree-Structured Policy Optimization for Expanding Reasoning Coverage in RLVR

Youngjun Yu Sanghwan Jang Hwanjo Yu Pohang University of Science and Technology (POSTECH) {colin31472, s.jang, hwanjoyu}@postech.ac.kr

## Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has been central to the recent success of Large Reasoning Models. However, while RLVR significantly improves singlesample accuracy, it often fails to expand the model’s intrinsic reasoning coverage (pass@k) due to limited exploration during training. To address this, we optimize the structural design of train-time rollouts to enhance pass@k. Our analysis identifies three key design principles: (1) difficulty-adaptive rollout can play an important role in expanding pass@k, beyond serving as an efficiency heuristic; (2) tree-based rollout outperforms parallel sampling in discovering correct answers; and (3) sentenceentropy-guided forking overcomes the localization phenomenon of token-level branching to maximize semantic diversity. Building on these insights, we propose DATPO (Difficulty-Adaptive Sentence-entropy-guided Tree-structured Policy Optimization). DATPO integrates difficulty-adaptive tree search with a sibling-diversity advantage term, explicitly promoting semantic diversity to expand reasoning coverage during training. Experiments on mathematical reasoning benchmarks demonstrate that DATPO outperforms baselines especially in pass@k, which directly translates to superior test-time scaling performance.<sup>1</sup>

## 1 Introduction

The paradigm of Large Reasoning Models (LRMs) has achieved remarkable success in eliciting rigorous problem-solving capabilities from foundation models (Jaech et al., 2024; Team et al., 2025). A key mechanism behind this is Reinforcement Learning with Verifiable Rewards (RLVR), exemplified by DeepSeek-R1 (Guo et al., 2025) through its successful application of the Group Relative Policy Optimization (GRPO) (Shao et al., 2024).

In parallel, LRM research has increasingly emphasized test-time scaling strategies, including CoT (Wei et al., 2022), majority voting (Wang et al., 2022), best-of-N (Stiennon et al., 2020), and Monte Carlo Tree Search (MCTS) (Ha et al., 2025). Fundamentally, the effectiveness of these methods is limited by the model’s intrinsic reasoning coverage—the breadth of valid reasoning paths the model can explore, typically measured by pass@k. Without sufficient reasoning coverage, even large candidate sets are unlikely to include a correct reasoning path, causing test-time scaling to aggregate or select among recurring errors (Brown et al., 2024; Zhao et al., 2025).

However, recent studies (Yue et al., 2025; Dang et al., 2025; Wu et al., 2025) argue that RLVR often fails to unlock new reasoning capabilities beyond the reasoning coverage of the base model. This limitation stems from a lack of explicit exploration strategies that steer the learning or sampling process toward underexplored regions. Without such guidance, the model merely exploits known solutions rather than discovering novel ones.

While recent studies introduce exploration strategies to expand reasoning coverage (Walder and Karkhanis, 2025; Yao et al., 2025; Wang et al., 2025), they largely focus on optimization-level interventions, leaving the train-time rollout as a fixed parallel structure. Tree-based frameworks (Hou et al., 2025; Liu et al., 2025a) have begun to move beyond standard parallel sampling. However, these methods primarily utilize tree structures for credit assignment or computational efficiency. Consequently, it remains underexplored which structural choices in train-time rollouts actually expand the model’s reasoning coverage.

To bridge this gap, we conduct a systematic empirical analysis of train-time rollouts, revealing three core design principles to maximize reasoning coverage, extending beyond improvements in single-sample accuracy. First, difficulty-adaptive rollout is an important factor in expanding pass@k, rather than merely an efficiency heuristic. While increasing the rollout budget improves pass@k when training on hard problems, we observe that it can actually degrade pass@k when training on easy problems. Second, tree-based rollout outperforms parallel sampling in discovering correct answers by efficiently utilizing limited computational budgets. Finally, we observe that conventional token-guided forking suffers from localization, where high-entropy tokens cluster in narrow segments and trap exploration. To overcome this, forking decisions must be elevated to a broader semantic level to discover genuinely diverse reasoning paths.

![](images/ebc624de57a502542d6014c801300c7db66e523706024a23ee5e74cb8970b739.jpg)  
(a) Models trained on Easy dataset

![](images/be88cbaea78909a3d611a1a69f0bbe23b48e920190269b8df5b006df5ed14be3.jpg)  
(b) Models trained on Medium dataset

![](images/c9ba934cbad106119cb596a90d296d965d442be5d998754b50120fdf31765fa0.jpg)  
(c) Models trained on Hard dataset  
Figure 1: avg@256 and pass@256 performance across different rollout budgets for models trained on different difficulty subsets.

Building on these insights, we propose DATPO (Difficulty-Adaptive Sentence-entropyguided Tree-structured Policy Optimization). For train-time exploration, DATPO combines the structural advantages of tree-based search with difficulty-adaptive rollout to expand reasoning coverage. Forking points are selected using sentencelevel entropy signals, broadening the granularity of uncertainty estimation to avoid localization. Furthermore, to maximize coverage-expanding benefits, we augment the advantage function with an annealed sibling-diversity term. This explicitly rewards semantically diverse reasoning branches, encouraging the model to explore a wider range of valid reasoning paths.

Extensive experiments on mathematical reasoning benchmarks demonstrate the effectiveness of DATPO, which achieves the highest average pass@k among the compared methods. Furthermore, this expanded reasoning coverage directly translates to superior test-time scaling with majority voting (maj@k).

In summary, our main contributions are as follows:

• We reveal key structural principles for expanding reasoning coverage: difficulty-adaptive rollout is a key factor, and tree-structured rollouts inherently outperform parallel sampling, with sentence-entropy forking further amplifying this structural advantage.

• We propose DATPO, a novel RLVR framework that integrates these structural insights with a diversity-augmented advantage to maximize coverage-expanding benefits. Experiments on mathematical reasoning benchmarks validate our approach, showing notable improvements in pass@k.

## 2 Analyzing the Structural Design of Train-time Rollouts

Despite extensive research on expanding reasoning coverage, the structural design of train-time rollouts remains underexplored. This section investigates optimal rollout structures to maximize not only single-sample accuracy (avg@k) but also broader reasoning coverage (pass@k). Our analysis is driven by three primary research questions: (1) Difficulty-Adaptive Allocation: How should compute budgets be allocated across problems of varying difficulty to expand reasoning coverage?; (2) Topology of Search Space: Which search space structure (tree-based or parallel rollout) more efficiently discovers correct reasoning trajectories?; and (3) Optimal Forking Strategy: Where should forking (branching) occur in tree-based rollouts to effectively promote diverse reasoning paths?

## 2.1 Necessity of Difficulty-Adaptive Rollout

Standard RLVR algorithms, notably GRPO (Shao et al., 2024) and its variants (Yu et al., 2025b; Liu et al., 2025b), typically generate a uniform number of rollouts regardless of problem difficulty. While existing difficulty-adaptive rollout strategies prioritize computational efficiency (Liu et al., 2025a; Liao et al., 2025; Zhang et al., 2025; Zheng et al., 2025a; Li et al., 2025b), we challenge this perspective by demonstrating that adaptive allocation is not merely a resource-saving heuristic but a crucial factor in expanding reasoning coverage.

## 2.1.1 Empirical Analysis

We first partitioned the training dataset into Easy, Medium, and Hard subsets based on the accuracy of the base model. Using GRPO (Shao et al., 2024), we conduct a total of 9 training runs. For each difficulty subset, we vary the rollout budget by adjusting the GRPO group size, $G \in \{ 4 , 8 , 1 6 \}$ . The resulting 9 models are evaluated on 3 different benchmarks, reporting both avg@256 and pass@256 metrics (detailed in Appendix B).

The aggregated results, as illustrated in Figure 1, reveal two key insights. First, avg@256 shows consistent, yet marginal, improvements with increased rollout budgets (G) across all difficulties. We attribute this to more stable exploitation; larger GRPO group sizes reduce variance in advantage estimation and yield more frequent meaningful learning signals, effectively solidifying the policy distribution around correct solutions.

Second, the impact of increasing rollouts on pass@256 shifts from detrimental to beneficial depending on the difficulty of the training dataset. For models trained on the Easy dataset (Figure 1a), larger budgets degrade pass@256. This occurs because the model over-exploits specific, easy templates, overfitting to familiar problems while failing entirely on harder ones. Since pass@256 requires only a single correct path, generating redundant successes for easy problems provides no coverage benefit. Consequently, despite marginal gains in avg@256, pass@256 decreases as the rollout budget increases. Conversely, for models trained on the Hard dataset (Figure 1c), where valid paths are scarce, larger budgets promote broader exploration without overfitting to narrow patterns, consistently improving pass@256. Between these opposing behaviors, models trained on the Medium dataset (Figure 1b) exhibit a nonmonotonic trend, peaking at G = 8 by effectively balancing exploration and over-exploitation.

![](images/f4c1e7b4838eee92146d61a4c4f12bd63be047b78e3aaeec3b95381c9460e431.jpg)  
Figure 2: Comparison of PassRate against inference token consumption between parallel and two treestructured sampling using different forking strategies.

These findings suggest that difficulty-adaptive rollout is not merely a heuristic for compute efficiency, but a strategic necessity for balancing the exploitation-exploration trade-off and maximizing the model’s reasoning coverage. Comprehensive theoretical analysis is provided in Appendix A.

## 2.2 Tree vs. Parallel Rollout

In RLVR, the primary challenge in expanding reasoning coverage is the scarcity of positive learning signals on complex problems. These signals emerge only when the model discovers a correct reasoning path. To address this, we investigate the properties of train-time rollout structures by comparing the conventional parallel rollout with a tree-structured rollout strategy, aiming to determine which topology more efficiently discovers correct trajectories.

## 2.2.1 Tree Rollout Algorithm

We employ a generalized and simplified variant of the two-phase tree rollout strategy originally proposed by Hou et al. (2025). In the first phase, the model generates N independent base rollouts in parallel. Subsequently, in the second phase, we apply a forking point selection algorithm to identify K forking points within each base trajectory. From each selected forking point, we generate B additional branch rollouts, thereby expanding the exploration scope. Detailed algorithmic procedures are provided in Appendix C.1.

## 2.2.2 Empirical Analysis

To analyze cost-efficiency during inference, we compare standard parallel sampling against two tree-structured rollout variants by analyzing PassRate—the probability of obtaining at least one correct solution—as a function of token consumption during generation. These variants include fixed-seg, which selects the K forking points at equidistant intervals, and random, which selects them uniformly at random (detailed in Appendix C.2).

As shown in Figure 2, both tree-structured variants achieve higher token efficiency than parallel sampling. Specifically, their performance curves demonstrate a steeper growth rate, yielding a higher PassRate for the same number of generated tokens. This enhanced efficiency stems from prefix sharing in tree structures, which avoids redundant regeneration of identical early segments and enables the exploration of alternative continuations within the same token budget (Tran et al., 2025; Hou et al., 2025).

Furthermore, the choice of forking strategy also affects performance within tree-structured methods. While both fixed-seg and random outperform parallel sampling, fixed-seg consistently achieves a higher PassRate under comparable token budgets. This demonstrates that the effectiveness of treebased exploration depends not only on its structure but also on the placement of branching points.

## 2.3 Forking Point Selection in Tree Search

Section 2.2 shows that the performance of treestructured rollouts varies substantially across different forking strategies. In this section, we investigate which forking strategies most effectively identify correct solutions and promote semantically diverse reasoning paths.

## 2.3.1 The Localization Phenomenon and Sentence-level Forking

Previous studies primarily select top-k highentropy tokens as forking points (Hou et al., 2025; Zheng et al., 2025b; Cao et al., 2026), as they capture highly uncertain words that act as pivotal branching points (Wang et al., 2025; Cheng et al., 2025). However, we identify a critical limitation in this approach: the localization phenomenon (Figure 8). High-entropy tokens tend to densely cluster within narrow, highly uncertain segments of the reasoning trajectory. Consequently, the search budget is monopolized by repeated resampling within a localized cluster, which limits the structural reach of the search tree.

To mitigate localization, we propose a sentencelevel forking strategy (sent-entropy), which expands the granularity of entropy estimation. In this approach, sentence entropy is calculated as the average entropy of its tokens, and the starting points of the top-k high-entropy sentences are selected as forking points. This strategy naturally evades localization while still targeting the most uncertain regions. We compare this approach with the conventional token-level method (tok-entropy) and three baselines: random, fixed-segment selection (fixed-seg), and Attention-based Tree Branching (ATB) (Liu et al., 2025a), which branches at steps receiving the highest attention weights. Detailed algorithmic procedures are provided in Appendix D.1.

<table><tr><td>Method</td><td>PassRate</td><td>SibDiv</td></tr><tr><td>random</td><td>10.0</td><td>0.0796</td></tr><tr><td>fixed-seg</td><td>13.3</td><td>0.0853</td></tr><tr><td>ATB</td><td>14.7</td><td>0.0874</td></tr><tr><td>tok-entropy</td><td>12.0</td><td>0.1065</td></tr><tr><td>sent-entropy</td><td>17.3</td><td>0.1049</td></tr></table>

Table 1: Comparison of forking point selection strategies when applied during inference. Best results are in bold, and second-best results are underlined.

## 2.3.2 Metrics for Measuring Diversity

While PassRate effectively captures task success, comprehensively evaluating these forking strategies also requires measuring whether the generated paths are semantically diverse. To directly quantify exploration diversity, we introduce an embeddingbased metric, Sibling Diversity (SibDiv). We first partition the reasoning tree into contiguous text blocks bounded by forking points or the end of the trajectory. SibDiv then computes the average pairwise cosine distance (i.e., 1−cosine similarity) between the embeddings of sibling blocks originating from the same forking point. By aggregating these values, this metric effectively evaluates how well the forking points promote semantically diverse reasoning branches. Formal definitions are provided in Appendix D.3.

## 2.3.3 Empirical Analysis

We evaluated each forking strategy by generating reasoning trees under a fixed inference budget, measuring PassRate and SibDiv (see Appendix D.4 for detailed experimental setups).

The results are summarized in Table 1. First, tok-entropy achieves the highest SibDiv because performing additional sampling at the model’s most uncertain points naturally generates diverse immediate sibling blocks. However, due to the localization phenomenon, this high SibDiv is strictly confined to a narrow segment of the reasoning path. The search fails to expand into meaningful structural differences across the entire reasoning tree, resulting in a low PassRate. By simply expanding the granularity of the entropy estimation, sent-entropy shows the highest PassRate while incurring minimal loss in SibDiv. In contrast, baselines such as fixed-seg and ATB inherently avoid localization, yielding higher PassRates than tok-entropy, but exhibit lower SibDiv since they do not utilize uncertainty signals. Ultimately, sent-entropy emerges as the most effective strategy by successfully translating high semantic diversity into a high PassRate.

![](images/33a80a17d3f78467fa6923e3edf14064b78964848025ad03ba13fe4916d2e168.jpg)

## 3 Methodology

Building upon the empirical insights from Section 2, we propose DATPO (Difficulty-Adaptive Sentence-entropy-guided Tree-structured Policy Optimization) to expand a model’s intrinsic reasoning coverage (pass@k). An overview of the proposed framework is illustrated in Figure 3.

## 3.1 Difficulty-Adaptive Tree Search at Train-time

Building on the tree algorithm proposed in Section 2.2.1, we introduce a difficulty-adaptive tree search to dynamically allocate the search budget. The procedure operates in two phases. First, we generate N independent base rollouts and estimate the empirical difficulty of the prompt through their average verifiable reward $V ( { \mathrm { r o o t } } ) \in [ 0 , 1 ]$

Second, we scale the tree expansion proportionally to this difficulty, where a lower $V ( \mathrm { r o o t } )$ indicates a more challenging problem, thus allocating more search budget. Given maximum budgets for forking points $K _ { \mathrm { m a x } }$ and branch rollouts $B _ { \mathrm { m a x } }$ , the adaptive parameters are computed as:

$$
\hat { K } = \lceil K _ { \mathrm { m a x } } ( 1 - V ( \mathrm { r o o t } ) ) \rceil\tag{1}
$$

$$
\hat { B } = \lceil B _ { \mathrm { m a x } } ( 1 - V ( \mathrm { r o o t } ) ) \rceil\tag{2}
$$

Using the sent-entropy (Section 2.3.1), we identify K<sup>ˆ</sup> forking points within each base trajectory and generate B<sup>ˆ</sup> branches from each point.

This mechanism entirely bypasses expansion only when $V ( \mathrm { r o o t } ) = 1 $ . For harder problems, it expands the search space up to $N ( 1 + { \hat { K } } { \hat { B } } )$ leaves. By scaling expansion based on difficulty, this strategy concentrates exploration on unsolved problems to maximize reasoning coverage.

![](images/007921fa0b48042cda9fd1d3736692a2f96150f3f9fb8491ea295e7c03ad1db4.jpg)  
Figure 3: Overview of the proposed DATPO framework.

## 3.2 Block-level Diversity-augmented Advantage Estimation

To optimize the policy, we partition trajectories into contiguous blocks bounded by forking points or the end of the trajectory. Advantages are computed and assigned at this block level.

Since our tree search generates multiple branches from intermediate states, we reliably estimate state values using Monte Carlo (MC) returns (Kazemnejad et al., 2024). For a state s at any forking point, $\hat { V } _ { M C } ( s )$ is the average verifiable reward of all its descending terminal blocks. For a terminal state $s _ { T }$ without further rollouts, $\hat { V } _ { M C } ( s _ { T } ) = 0$

For a block b spanning from $s _ { \mathrm { s t a r t } } ^ { ( b ) }$ to $s _ { \mathrm { e n d } } ^ { ( b ) }$ , the base advantage is formulated as:

$$
\hat { A } _ { \mathrm { b a s e } } ( b ) = r ( b ) + \hat { V } _ { M C } ( s _ { \mathrm { e n d } } ^ { ( b ) } ) - \hat { V } _ { M C } ( s _ { \mathrm { s t a r t } } ^ { ( b ) } )\tag{3}
$$

where the verifiable reward $r ( b ) \in \{ 0 , 1 \}$ is assigned exclusively to terminal blocks; otherwise, it is 0. All tokens within b share this identical advantage. This block-level advantage assigns precise credit to intermediate steps, effectively facilitating implicit process supervision.

To maximize the coverage-expanding benefits, we augment the base advantage with a siblingdiversity term $\mathrm { D i v _ { s i b } } ( b )$ , inspired by the SibDiv metric. Specifically, sibling blocks refer to the child blocks generated from a shared forking point. $\mathrm { D i v _ { s i b } } ( b )$ is defined as the average cosine distance between the embedding of block b and its sibling blocks, effectively encouraging the model to discover distinct reasoning paths. Crucially, we apply this bonus exclusively to blocks with positive base advantages to avoid incentivizing the exploration of incorrect paths. The augmented advantage is:

$$
\hat { A } ( b ) = \hat { A } _ { \mathrm { b a s e } } ( b ) + \mathbb { I } ( \hat { A } _ { \mathrm { b a s e } } ( b ) > 0 ) \cdot \alpha \cdot \mathrm { D i v } _ { \mathrm { s i b } } ( b )\tag{4}
$$

<table><tr><td rowspan="2">Method</td><td colspan="2">MATH500</td><td colspan="2">AIME26</td><td colspan="2">AIME25</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td><td colspan="2">Average</td></tr><tr><td>avg@8</td><td>pass@8</td><td>avg@64</td><td>pass@64</td><td>avg@64</td><td>pass@64</td><td>avg@64</td><td>pass@64</td><td>avg@64</td><td>pass@64</td><td>avg@k</td><td>c pass@k</td></tr><tr><td colspan="10">Qwen2.5-3B-Base</td><td></td><td></td><td></td></tr><tr><td>Base</td><td>26.3</td><td>64.7</td><td>0.4</td><td>8.9</td><td>0.2</td><td>12.2</td><td>1.0</td><td>22.2</td><td>10.0</td><td>78.3</td><td>7.6</td><td>37.3</td></tr><tr><td>GRPO</td><td>61.8</td><td>79.8</td><td>1.7</td><td>21.1</td><td>0.7</td><td>24.4</td><td>3.6</td><td>30.0</td><td>35.9</td><td>85.8</td><td>20.7</td><td>48.2</td></tr><tr><td>Dr.GRPO</td><td>61.3</td><td>78.7</td><td>3.0</td><td>23.3</td><td>1.4</td><td>27.8</td><td>5.1</td><td>27.8</td><td>37.9</td><td>83.3</td><td>21.7</td><td>48.2</td></tr><tr><td>TreeRL</td><td>61.7</td><td>80.7</td><td>2.4</td><td>22.2</td><td>1.7</td><td>20.0</td><td>4.2</td><td>24.4</td><td>38.7</td><td>83.3</td><td>21.7</td><td>46.1</td></tr><tr><td>AttnRL</td><td>62.0</td><td>82.1</td><td>1.9</td><td>32.2</td><td>1.5</td><td>25.6</td><td>5.2</td><td>34.4</td><td>35.7</td><td>90.8</td><td>21.3</td><td>53.0</td></tr><tr><td>DATPO (Ours)</td><td>63.5</td><td>81.7</td><td>2.4</td><td>33.3</td><td>1.6</td><td>33.3</td><td>4.8</td><td>35.6</td><td>39.8</td><td>90.8</td><td>22.4</td><td>54.9</td></tr><tr><td colspan="10">Qwen3-4B-Base</td><td colspan="2"></td></tr><tr><td>Base</td><td>39.9</td><td>79.4</td><td>1.7</td><td>21.1</td><td>1.1</td><td>32.2</td><td>2.8</td><td>38.9</td><td>17.2</td><td>84.2</td><td>12.5</td><td>51.2</td></tr><tr><td>GRPO</td><td>75.5</td><td>87.4</td><td>7.1</td><td>32.2</td><td>10.3</td><td>38.9</td><td>9.6</td><td>41.1</td><td>47.9</td><td>91.7</td><td>30.1</td><td>58.3</td></tr><tr><td>Dr.GRPO</td><td>75.2</td><td>86.2</td><td>7.8</td><td>25.6</td><td>7.7</td><td>35.6</td><td>9.7</td><td>40.0</td><td>46.6</td><td>90.8</td><td>29.4</td><td>55.6</td></tr><tr><td>TreeRL</td><td>75.3</td><td>86.2</td><td>7.9</td><td>28.9</td><td>9.4</td><td>36.7</td><td>9.4</td><td>41.1</td><td>49.4</td><td>90.8</td><td>30.3</td><td>56.7</td></tr><tr><td>AttnRL</td><td>74.7</td><td>87.5</td><td>7.1</td><td>28.9</td><td>8.2</td><td>36.7</td><td>11.0</td><td>42.2</td><td>52.5</td><td>91.7</td><td>30.7</td><td>57.4</td></tr><tr><td>DATPO (Ours)</td><td>76.4</td><td>87.9</td><td>6.5</td><td>33.3</td><td>10.9</td><td>42.2</td><td>14.0</td><td>46.7</td><td>48.4</td><td>91.7</td><td>31.3</td><td>60.4</td></tr></table>

Table 2: Evaluation results on mathematical reasoning benchmarks. We report avg@k and pass@k for each dataset.

where I(·) is the indicator function, and the coefficient α is linearly annealed during training to gradually decay the exploration incentive, allowing the model to refine its learned reasoning paths.

Finally, we optimize the policy directly across the generated tree topology. Given a set of contiguous blocks B generated for a prompt $q ,$ the DATPO objective is formulated as:

$$
\begin{array} { c l } { \mathcal { I } _ { \mathrm { D A T P O } } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { Q } , \mathcal { B } \sim \pi _ { \theta _ { \mathrm { o l d } } } } \Bigg [ \frac { 1 } { \sum _ { b \in \mathcal { B } } \vert b \vert } \underset { b \in \mathcal { B } } { \sum } \underset { t = 1 } { \overset { \vert b \vert } { \sum } } \underset { \theta \in \mathcal { B } } { \sum } \underset { \theta \in \mathcal { B } } { \sum } \Bigg ] } \\ { \operatorname* { m i n } \Big ( \rho _ { b , t } ( \theta ) \hat { A } ( b ) , \mathrm { c l i p } \big ( \rho _ { b , t } ( \theta ) , \quad } & { } \\ { 1 - \epsilon , 1 + \epsilon \big ) \hat { A } ( b ) \Big ) \Bigg ] , } \end{array}\tag{5}
$$

where |b| is the sequence length of block b, and $\hat { A } ( b )$ is the block-level augmented advantage. Comprehensive algorithmic details and training procedures are provided in Appendix E.

## 4 Experiments

## 4.1 Experimental Setup

Models and Datasets We use Qwen2.5-3B-Base (Yang et al., 2024) and Qwen3-4B-Base (Yang et al., 2025) as base models. For training, we use the MATH dataset (Hendrycks et al., 2021).

Evaluation We evaluate the trained models on mathematical reasoning benchmarks, specifically MATH500 (Hendrycks et al., 2021), AIME26, AIME25, AIME24, and AMC23. Using a temperature of 1.0, we report avg@k as the primary evaluation metric across all benchmarks. Additionally, we report pass@k to assess the reasoning coverage of the trained models. For robustness, the pass@k results are computed by averaging over three independent evaluation runs.

![](images/c080f56d1758be3cc646909d138ac6dd0acdbe18fa12565cc74e324fb1d2bf16.jpg)  
Figure 4: Learning curves of MATH500 accuracy over training steps for tree-based methods. (Smoothed)

Baselines We compare DATPO against the Base model and several advanced RLVR baselines. These include GRPO (Shao et al., 2024) (enhanced with clip-higher and token-level loss (Yu et al., 2025b)) and its variant, Dr.GRPO (Liu et al., 2025b). Furthermore, we evaluate two tree-based methods: TreeRL (Hou et al., 2025), which utilizes top-k high-entropy forking and combined localglobal advantages, and AttnRL (Liu et al., 2025a), which leverages attention-based forking, adaptive sampling, and a one-step off-policy for enhanced efficiency. To ensure a fair comparison, we maintain a comparable total number of generated tokens per problem across all methods while adopting the treeexpansion hyperparameters reported in the original TreeRL and AttnRL papers. Comprehensive details for experiments are provided in Appendix F.

![](images/0435ec04f9ac4985db2700f5c5397f5723091f769fe97908a995521b341b47ff.jpg)  
Figure 5: Comparison of average leaf count generated at train-time and PassRate across difficulty levels for tree-based methods.

## 4.2 Main Results

Mathematical Reasoning Performance As reported in Table 2, DATPO achieves the best aggregate avg@k and pass@k across the evaluated benchmarks. Notably, while the improvements in single-sample accuracy (avg@k) are marginal compared to the strongest baseline, AttnRL (e.g., +1.1 and +0.6 on Qwen2.5-3B-Base and Qwen3- 4B-Base, respectively), DATPO achieves substantial gains in pass@k, outperforming AttnRL by +1.9 and +3.0, respectively. This indicates that DATPO’s difficulty-adaptive rollout and siblingdiversity term successfully expand the model’s intrinsic reasoning coverage.

Analysis of Training Dynamics We first examine the MATH500 accuracy over training steps (Figure 4). While all three tree-based methods exhibit similar accuracy gains during the initial training steps, TreeRL and AttnRL fail to sustain this upward trend, eventually plateauing or even suffering performance degradation. In contrast, DATPO shows a consistent upward trend throughout the training. This stability can be attributed to the annealed sibling-diversity term, which injects semantic diversity early in training to prevent premature convergence.

Furthermore, we analyze the search behavior during training by comparing the average leaf count and PassRate across the difficulty levels of the MATH dataset (Figure 5). Note that these difficulty labels are used strictly for this analysis and were not provided to the model during training. Unlike TreeRL, which maintains a fixed leaf count across all difficulties, DATPO and AttnRL employ difficulty-adaptive rollouts. However, DATPO allocates fewer rollouts to easy problems and significantly more rollouts to hard problems compared to AttnRL. The latter’s adaptive strategy relies heavily on an attention-based filtering mechanism that simply discards problems with below-average attention scores to maximize computational efficiency. Because this approach acts as a rigid cut-off, it fails to concentrate computational resources on the most challenging problems. Consequently, DATPO’s highly adaptive resource allocation enables it to achieve a higher PassRate on harder problems (Levels 4 and 5) compared to other tree-based methods.

![](images/721823ba3ce9dea22da0827b0a86f21f1354d74fce8b372edf747ed35163430d.jpg)  
Figure 6: Test-time scaling performance using majority voting (maj@k). The green arrows indicate the performance improvement of maj@k over avg@k.

## 4.3 Test-time Scaling Performance

In this section, we investigate whether the substantial gains in pass@k directly translate into improved test-time scaling performance. To this end, we apply majority voting (maj@k) (Wang et al., 2022)—one of the simplest and most widely adopted test-time scaling strategies—to compare the methods applied to Qwen2.5-3B-Base. We evaluate the effectiveness of these methods across five mathematical reasoning benchmarks, applying k = 8 for MATH500 and k = 64 for the remaining datasets. For robustness, all maj@k results are computed as the average over three independent evaluation runs.

As illustrated by the average performance across five benchmarks in Figure 6, DATPO, which achieved the highest pass@k during training, also shows the largest maj@k gain, improving by +7.2 over its avg@k. This demonstrates that expanding a model’s reasoning coverage during training directly enhances test-time scaling performance.

## 4.4 Ablation Studies

Effect of Forking Strategies Building upon the analysis during inference presented in Section 2.3, we further investigate the impact of different forking strategies during training. We compare DATPO’s default sent-entropy strategy against four alternatives: random, fixed-seg, ATB, and tok-entropy, evaluating their performance on the MATH500 benchmark. As reported in Table 3, the sent-entropy yields the highest performance. This confirms our hypothesis: the sent-entropy effectively guides exploration using the model’s uncertainty, while avoiding localization that limits tok-entropy.

<table><tr><td rowspan="2">Method</td><td colspan="2">MATH500</td></tr><tr><td>avg@8</td><td>pass@8</td></tr><tr><td>random</td><td>61.5</td><td>81.6</td></tr><tr><td>fixed-seg</td><td>61.1</td><td>80.6</td></tr><tr><td>ATB</td><td>62.1</td><td>80.7</td></tr><tr><td>tok-entropy</td><td>61.3</td><td>80.3</td></tr><tr><td>sent-entropy</td><td>63.5</td><td>81.7</td></tr></table>

Table 3: Ablation study on different forking strategies.

Effect of Sibling-Diversity To evaluate the effect of the sibling-diversity term, we conduct an ablation study by varying the coefficient α, and evaluate the performance on the MATH500 benchmark, as summarized in Table 4.

The results show that the 0.2 → 0 schedule is the most effective, surpassing both the baseline without diversity (0 → 0) and a higher initial coefficient (0.4 → 0). Furthermore, applying the diversity bonus to all blocks (0.2 → 0, all) degrades performance compared to the baseline without diversity, highlighting the need to reward diversity exclusively on valid paths. Annealing is also critical: a constant α (0.2 → 0.2) severely degrades avg@8 despite slight pass@8 gains. Conversely, annealing α to a negative value (0.2 → −0.2) improves avg@8 but noticeably harms pass@8, confirming that penalizing diversity in the later stages of training restricts the model’s reasoning coverage.

## 5 Related Work

Reinforcement Learning for LLM Reasoning RLVR has become the standard for post-training of reasoning LLMs. A representative method is Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which replaces the costly critic model of PPO (Schulman et al., 2017) with efficient groupbased advantage estimation. Recent studies build on this framework to enhance stability and resolve optimization issues: DAPO (Yu et al., 2025b) prevents mode collapse through decoupled clipping and dynamic sampling, while Dr.GRPO (Liu et al.,

<table><tr><td rowspan="2">α</td><td colspan="2">MATH500</td></tr><tr><td>avg@8</td><td>pass@8</td></tr><tr><td> $0  0$ </td><td>62.1</td><td>81.4</td></tr><tr><td> ${ \bf 0 . 2 }  { \bf 0 }$ </td><td>63.5</td><td>81.7</td></tr><tr><td> $0 . 4  0$ </td><td>61.9</td><td>81.3</td></tr><tr><td> $0 . 2  0 , a l l$ </td><td>61.6</td><td>81.3</td></tr><tr><td> $0 . 2  0 . 2$ </td><td>61.5</td><td>82.0</td></tr><tr><td> $0 . 2  - 0 . 2$ </td><td>63.4</td><td>81.0</td></tr></table>

Table 4: Ablation study on the diversity coefficient α. The right arrow (→) indicates the linear annealing schedule during training. The all applies the diversity term to all blocks, rather than exclusively to those with positive base advantages.

2025b) corrects inherent structural biases in the advantage computation.

Exploration Strategies in RLVR RLVR often fails to expand a model’s intrinsic reasoning capacity (pass@k) beyond its base capabilities (Yue et al., 2025; Dang et al., 2025; Wu et al., 2025). To overcome this exploration bottleneck, recent studies introduce explicit mechanisms to diversify search. PKPO (Walder and Karkhanis, 2025) directly optimizes pass@k to solve harder instances, while R1-zero-Div (Yao et al., 2025) and FOR (Yu et al., 2025a) use diversity-aware objectives and divergent reasoning flows to prevent trajectory collapse.

Tree-based Search in Reinforcement Learning Proven in RL milestones like AlphaGo (Silver et al., 2016), tree-based search is now adapted to LLMs to enable systematic exploration. TreeRL (Hou et al., 2025) pioneers this with an on-policy framework using entropy-guided branching. To address computational bottlenecks, AttnRL (Liu et al., 2025a) improves efficiency by leveraging difficulty-aware adaptive sampling within a onestep off-policy pipeline. While these methods use trees primarily for credit assignment or computational efficiency, we leverage them to expand the model’s reasoning coverage.

## 6 Conclusion

In this work, we investigate the structural design of train-time rollouts in RLVR to expand the intrinsic reasoning coverage. Our analysis reveals that expanding reasoning coverage can benefit from moving beyond uniform parallel sampling toward difficulty-adaptive resource allocation and the structural advantages of sentence-entropy-guided tree search. Integrating these principles, we propose DATPO, which explicitly encourages semantic exploration through a sibling-diversity term. Experiments on mathematical benchmarks validate our approach, showing significant improvements in pass@k.

## Limitations

While DATPO significantly expands reasoning coverage, calculating the sibling-diversity term requires additional forward passes through an external embedding model, which introduces computational overhead.

Additionally, while our block-level formulation effectively assigns process-level credit across the tree topology, it relies on small sample sizes. Deriving both the empirical difficulty and Monte Carlo state values from a limited number of rollouts (e.g., $N = 4 , B = 4 )$ can inject noise into the estimations, occasionally yielding high variance during policy updates.

Finally, resource limitations restricted our empirical validation to the 3B-4B parameter regime, leaving DATPO’s scalability to larger models $\mathrm { ( \ge 7 B ) }$ unverified. Furthermore, as our evaluation primarily focuses on mathematical reasoning, the framework’s direct effectiveness in other rigorous domains, such as complex logical reasoning or code generation, requires further investigation.

## Acknowledgments

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (No. RS-2019-II191906, Artificial Intelligence Graduate School Program (POSTECH); IITP-2026-RS-2026-25616370, AI Star Fellowship Support Program) and by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00335873).

## References

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787.

Lang Cao, Hui Ruan, Yongqian Li, Peng Chao, Wu Ning, Haonan Song, Renhong Chen, and Yi-

tong Li. 2026. Treeadv: Tree-structured advantage redistribution for group-based rl. arXiv preprint arXiv:2601.03703.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. 2025. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758.

Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. 2025. Assessing diversity collapse in reasoning. In Scaling Self-Improving Foundation Models without Human Supervision.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Rui Ha, Chaozhuo Li, Rui Pu, Litian Zhang, Xi Zhang, and Sen Su. 2025. DSG-MCTS: A dynamic strategyguided Monte Carlo tree search for diversified reasoning in large language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 10530–10544, Suzhou, China. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Zhenyu Hou, Ziniu Hu, Yujiang Li, Rui Lu, Jie Tang, and Yuxiao Dong. 2025. Treerl: Llm reinforcement learning with on-policy tree search. arXiv preprint arXiv:2506.11902.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, and 1 others. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Zishang Jiang, Jinyi Han, Tingyun Li, Xinyi Wang, Sihang Jiang, Jiaqing Liang, Zhaoqian Dai, Shuguang Ma, Fei Yu, and Yanghua Xiao. 2025. Selective expert guidance for effective and diverse exploration in reinforcement learning of llms. arXiv preprint arXiv:2510.04140.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. 2024. Vineppo: Refining credit assignment in rl training of llms. arXiv preprint arXiv:2410.01679.

Yizhi Li, Qingshui Gu, Zhoufutu Wen, Ziniu Li, Tianshun Xing, Shuyue Guo, Tianyu Zheng, Xin Zhou, Xingwei Qu, Wangchunshu Zhou, and 1 others. 2025a. Treepo: Bridging the gap of policy optimization and efficacy and inference efficiency with heuristic tree-based modeling. arXiv preprint arXiv:2508.17445.

Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding, Ruoyu Sun, Ge Zhang, Wenhao Huang, and Zhi-Quan Luo. 2025b. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation. arXiv preprint arXiv:2509.25849.

Mengqi Liao, Xiangyu Xi, Chen Ruinian, Jia Leng, Yangen Hu, Ke Zeng, Shuai Liu, and Huaiyu Wan. 2025. Enhancing efficiency and exploration in reinforcement learning for LLMs. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 1451–1463, Suzhou, China. Association for Computational Linguistics.

Runze Liu, Jiakang Wang, Yuling Shi, Zhihui Xie, Chenxin An, Kaiyan Zhang, Jian Zhao, Xiaodong Gu, Lei Lin, Wenping Hu, and 1 others. 2025a. Attention as a compass: Efficient exploration for processsupervised rl in reasoning models. arXiv preprint arXiv:2509.26628.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025b. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022.

Nipun Sadvilkar and Mark Neumann. 2020. PySBD: Pragmatic sentence boundary disambiguation. In Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS), pages 110–114, Online. Association for Computational Linguistics.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, and 1 others. 2016. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008– 3021.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, and 77 others. 2025. Kimi k1.5: Scaling reinforcement learning with llms. Preprint, arXiv:2501.12599.

Hieu Tran, Zonghai Yao, and Hong Yu. 2025. Exploiting tree structure for credit assignment in rl training of llms. arXiv preprint arXiv:2509.18314.

Christian Walder and Deep Karkhanis. 2025. Pass@ k policy optimization: Solving harder reinforcement learning problems. arXiv preprint arXiv:2505.15201.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, and 1 others. 2025. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, and Yejin Choi. 2025. The invisible leash: Why rlvr may or may not escape its origin. arXiv preprint arXiv:2507.14843.

Shangyu Xing, Siyuan Wang, Chenyuan Yang, Xinyu Dai, and Xiang Ren. 2025. Lookahead tree-based

rollouts for enhanced trajectory-level exploration in reinforcement learning with verifiable rewards. arXiv preprint arXiv:2510.24302.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 22 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Jian Yao, Ran Cheng, Xingyu Wu, Jibin Wu, and Kay Chen Tan. 2025. Diversity-aware policy optimization for large language model reasoning. arXiv preprint arXiv:2505.23433.

Fangxu Yu, Lai Jiang, Haoqiang Kang, Shibo Hao, and Lianhui Qin. 2025a. Flow of reasoning: Training llms for divergent reasoning with minimal examples. In Forty-second International Conference on Machine Learning.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025b. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837.

Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. mGTE: Generalized longcontext text representation and reranking models for multilingual text retrieval. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1393–1412, Miami, Florida, US. Association for Computational Linguistics.

Yuheng Zhang, Wenlin Yao, Changlong Yu, Yao Liu, Qingyu Yin, Bing Yin, Hyokun Yun, and Lihong Li. 2025. Improving sampling efficiency in rlvr through adaptive rollout and response reuse. arXiv preprint arXiv:2509.25808.

Eric Zhao, Pranjal Awasthi, and Sreenivas Gollapudi. 2025. Sample, scrutinize and scale: Effective inference-time search by scaling verification. arXiv preprint arXiv:2502.01839.

Haizhong Zheng, Yang Zhou, Brian R Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. 2025a. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. arXiv preprint arXiv:2506.02177.

Tianyu Zheng, Tianshun Xing, Qingshui Gu, Taoran Liang, Xingwei Qu, Xin Zhou, Yizhi Li, Zhoufutu Wen, Chenghua Lin, Wenhao Huang, and 1 others. 2025b. First return, entropy-eliciting explore. arXiv preprint arXiv:2507.07017.

## A Theoretical Analysis of Difficulty-Adaptive Rollout

We provide a theoretical analysis of the empirical findings in Section 2.1. In particular, we explain why increasing the rollout budget $G$ consistently improves the expected avg@k, while the behavior of pass@k can vary across difficulty subsets.

Notation. For a prompt x and a policy $\pi _ { \theta } .$ let

$$
p _ { x } ( \theta ) : = \operatorname* { P r } _ { y \sim \pi _ { \theta } ( \cdot | x ) } [ r ( y ) = 1 ]
$$

denote the single-sample correctness probability under a binary verifiable reward $r ( y ) \in \{ 0 , 1 \}$

In empirical evaluations, metrics are computed over a finite set of k independent samples $\mathcal { V } =$ $( y ^ { ( 1 ) } , \dots , y ^ { ( k ) } ) \overset { \mathrm { i . i . d . } } { \sim } \pi _ { \theta } ( \cdot \mid x )$ . We define the empirical estimators as:

$$
\begin{array} { l l } { \displaystyle \widetilde { \arg @ { \bf k } _ { x } } ( \theta ) : = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } r \left( y ^ { ( j ) } \right) , } \\ { \displaystyle \widetilde { \sf p a s s } \ @ { \bf k } _ { x } ( \theta ) : = \mathbb { I } \left( \sum _ { j = 1 } ^ { k } r \left( y ^ { ( j ) } \right) \ge 1 \right) . } \end{array}
$$

However, to rigorously analyze the optimization dynamics, we focus on their expected values with respect to the policy. Throughout this analysis, we omit the hat notation and define avg $\boldsymbol { \cdot } \ @ \boldsymbol { \mathrm { k } } _ { x } ( \theta )$ and pass $\boldsymbol { @ } \mathbf { k } _ { x } ( \theta )$ strictly as these theoretical expectations:

$$
\begin{array} { r } { \arg @ \operatorname { k } _ { x } ( \theta ) : = \mathbb { E } _ { \mathcal { V } \sim \pi _ { \theta } } \left[ \widehat { \mathrm { a v g } @ \operatorname { k } _ { x } } ( \theta ) \right] } \\ { = p _ { x } ( \theta ) \ } \end{array}
$$

and

$$
\begin{array} { r l } & { \mathrm { p a s s } @ \mathrm { k } _ { x } ( \theta ) : = \mathbb { E } _ { \mathcal { V } \sim \pi _ { \theta } } \left[ \widehat { \mathrm { p a s s } @ \mathrm { k } _ { x } ( \theta ) } \right] } \\ & { ~ = 1 - \left( 1 - p _ { x } ( \theta ) \right) ^ { k } . } \end{array}
$$

By formulating these metrics as expectations, the theoretical avg@k rigorously simplifies to the single-sample success probability $p _ { x } ( \theta )$

For the analysis below, given the current policy parameters $\theta ,$ let

$$
\theta _ { G } ^ { + } : = \theta + \eta \hat { g } _ { G } ( x )
$$

denote one GRPO update with rollout group size $G$ , learning rate $\eta ,$ and stochastic gradient estimator ${ \hat { g } } _ { G } ( x )$ . We consider only the reward-advantage component of the GRPO update, excluding auxiliary regularization terms.

Theorem A.1 (One-step expected avg@k improvement with rollout budget). Fix a prompt x and a policy $\pi _ { \theta } .$ . Let $p : = p _ { x } ( \theta ) \in ( 0 , 1 )$ , and let $S _ { G } \sim$ Binomial $( G , p )$ denote the number of correct responses in a rollout group size $G \ge 2$

Assume $p _ { x } ( \cdot )$ is twice continuously differentiable with a bounded Hessian, and the gradient estimator satisfies $\mathbb { E } [ \| \hat { g } _ { G } ( x ) \| ^ { 2 } ] < \infty .$ . Further assume:

A1. The update vanishes if all group rewards are identical: ${ \hat { g } } _ { G } ( x ) = 0$ almost surely if $S _ { G } \in$ $\{ 0 , G \}$

A2. The expected update direction on mixedreward batches is a positive constant $c _ { x } > 0$ independent ofG:

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \nabla _ { \theta } p _ { x } ( \theta ) , \hat { g } _ { G } ( x ) \rangle \mid 1 \leq S _ { G } \leq G - 1 ] } \\ & { \quad ~ = c _ { x } . } \end{array}
$$

Then,for sufficiently small $\eta ,$ the expected correctness probability satisfies:

$$
\mathbb { E } [ p _ { x } ( \theta _ { G } ^ { + } ) ] = p _ { x } ( \theta ) + \eta c _ { x } \Psi _ { G } ( p ) + O _ { G } ( \eta ^ { 2 } ) ,
$$

where $\Psi _ { G } ( p ) : = 1 - p ^ { G } - ( 1 - p ) ^ { G } .$

Consequently, for any finite set of rollout budgets ${ \mathcal { G } } ,$ there exists $\eta _ { 0 } > 0$ such thatfor all $\eta \in ( 0 , \eta _ { 0 } )$ and any $G _ { 1 } , G _ { 2 } \in \mathcal { G }$ with $G _ { 2 } > G _ { 1 }$

$$
\mathbb { E } [ p _ { x } ( \theta _ { G _ { 2 } } ^ { + } ) ] > \mathbb { E } [ p _ { x } ( \theta _ { G _ { 1 } } ^ { + } ) ] .
$$

Furthermore, the first-order gain coefficient $\Psi _ { G } ( p )$ is strictly increasing in G with diminishing returns.

Proof. By definition, theoretical expected avg@k simplifies to the single-sample correctness probability $p _ { x } ( \theta )$ . Let $E _ { G } : = \{ 1 \leq S _ { G } \leq G - 1 \}$ be the event of a mixed-reward batch, which occurs with probability $\mathrm { P r } ( E _ { G } ) = 1 - ( 1 - p ) ^ { G } - p ^ { G } = \Psi _ { G } ( p )$

By A1, the update is zero outside $E _ { G } . \mathrm { A p p l y i n g }$ $\mathbf { A } 2 .$ , the expected directional derivative is:

$$
\begin{array} { r l } & { \mathbb { E } [ \langle \nabla _ { \theta } p _ { x } ( \theta ) , \hat { g } _ { G } ( x ) \rangle ] } \\ & { \quad = \operatorname* { P r } ( E _ { G } ) \mathbb { E } [ \langle \nabla _ { \theta } p _ { x } ( \theta ) , \hat { g } _ { G } ( x ) \rangle \mid E _ { G } ] } \\ & { \quad = c _ { x } \Psi _ { G } ( p ) . } \end{array}
$$

Since $p _ { x } ( \cdot )$ has a bounded Hessian on the update region, Taylor’s theorem gives

$$
\begin{array} { r } { p _ { x } ( \theta _ { G } ^ { + } ) = p _ { x } ( \theta ) + \eta \left. \nabla _ { \theta } p _ { x } ( \theta ) , \hat { g } _ { G } ( x ) \right. + R _ { G } ( \eta ) , } \end{array}
$$

where, for some Hessian bound $\begin{array} { r l r } { L } & { { } < } & { \infty } \end{array}$ $| R _ { G } ( \eta ) | \le \frac { L } { 2 } \eta ^ { 2 } \| \hat { g } _ { G } ( x ) \| ^ { 2 }$ . Taking expectations and using $\mathbb { E } [ \| \hat { g } _ { G } ( x ) \| ^ { 2 } ] < \infty$ gives $\mathbb { E } [ R _ { G } ( \eta ) ] =$ ${ \cal O } _ { G } ( \eta ^ { 2 } )$ , completing the first claim.

To establish monotonicity, we evaluate the marginal increase in $\Psi _ { G } ( p )$ for $p \in ( 0 , 1 )$

$\Psi _ { G + 1 } ( p ) - \Psi _ { G } ( p ) = p ( 1 - p ) ^ { G } + ( 1 - p ) p ^ { G } > 0 ,$ confirming $\Psi _ { G } ( p )$ strictly increases with $G .$ For any $G _ { 2 } \ > \ G _ { 1 }$ , the first-order difference $\eta c _ { x } ( \Psi _ { G _ { 2 } } ( p ) - \Psi _ { G _ { 1 } } ( p ) )$ is strictly positive and dominates the ${ \cal O } _ { G _ { 1 } , G _ { 2 } } ( \eta ^ { 2 } )$ remainder term for η strictly less than some threshold $\eta _ { 0 } ( G _ { 1 } , G _ { 2 } ) > 0$ . Taking the minimum $\eta _ { 0 }$ across all pairs in the finite set $\mathcal { G }$ establishes the strict ordering simultaneously.

Finally, the second difference of $\Psi _ { G } ( p )$ is $- p ^ { 2 } ( 1 - p ) ^ { G } - ( 1 - p ) ^ { 2 } p ^ { G } < 0 .$ , confirming diminishing returns. □

Remark on Assumption 2. A2 serves as an idealized first-order approximation to isolate the core mechanism: larger groups increase the mixed-reward probability $\Psi _ { G } ( p )$ , thereby providing nonzero relative reward signals more frequently. In practice, GRPO normalizes advantages relative to the group (e.g., the positive advantage scales as $\sqrt { ( G - s ) / s } )$ , meaning the exact expected magnitude $c _ { x }$ technically depends on $G .$ However, this assumption keeps the mechanism analytically transparent and highlights a plausible first-order driver of the empirical monotonicity observed during training.

Remark on advantage-estimation variance. Our analysis isolates the signal-availability mechanism, demonstrating that a larger G yields more frequent informative updates. We do not explicitly model the variance reduction in advantage estimation to avoid complex distributional assumptions. Therefore, this theorem formalizes one core driver of the avg@k improvement, acknowledging that other statistical benefits also contribute.

Extension to Dataset-Level Notation. While Theorem A.1 characterizes the optimization dynamics of a single prompt, explaining the difficultydependent behavior requires dataset-level formalization. Let $p _ { x , d } ( G )$ denote the single-sample correctness probability on a test prompt $x \in \mathcal { D }$ for a policy trained on difficulty subset d using rollout budget G. We define the expected average correctness $( \mu _ { d } ( G ) )$ and pass@k over the dataset $\mathcal { D }$ as:

$$
\mu _ { d } ( G ) : = \mathbb { E } _ { x \sim \mathcal { D } } [ p _ { x , d } ( G ) ] ,
$$

$$
\mathrm { p a s s } @ \mathbf { k } _ { \mathcal { D } } ^ { ( d ) } ( G ) : = \mathbb { E } _ { x \sim \mathcal { D } } \Big [ 1 - \big ( 1 - p _ { x , d } ( G ) \big ) ^ { k } \Big ]
$$

Theorem A.2 (Variance Penalty Decomposition of pass@k). Fix $k \geq 2 .$ . Let $f _ { k } ( p ) : = 1 - ( 1 - p ) ^ { k }$ and define the Jensen gap

$$
J _ { k , d } ( G ) : = f _ { k } { \big ( } \mu _ { d } ( G ) { \big ) } - p a s s @ k _ { \mathcal { D } } ^ { ( d ) } ( G ) .
$$

Then thefollowing hold:

1. $J _ { k , d } ( G ) ~ \ge ~ 0 ,$ , with equality if and only if $p _ { x , d } ( G )$ is constant almost surely over $x \sim \mathcal { D }$

2. For any two rollout budgets $G _ { 2 } > G _ { 1 }$

$$
\begin{array} { r l } & { p a s s @ k _ { \mathcal { D } } ^ { ( d ) } ( G _ { 2 } ) - p a s s @ k _ { \mathcal { D } } ^ { ( d ) } ( G _ { 1 } ) } \\ & { \ = \underbrace { f _ { k } \big ( \mu _ { d } ( G _ { 2 } ) \big ) - f _ { k } \big ( \mu _ { d } ( G _ { 1 } ) \big ) } _ { g a i n f r o m h i g h e r a v g @ k _ { \mathcal { D } } } } \\ & { \ - \ \underbrace { \big ( J _ { k , d } ( G _ { 2 } ) - J _ { k , d } ( G _ { 1 } ) \big ) } _ { l o s s f r o m t e s t . s e t p r o m p t h e t e r o g e n e i t y } \ . } \end{array}
$$

Moreover, $i f \sigma _ { d } ^ { 2 } ( G ) : = \operatorname { V a r } _ { x \sim { \mathcal { D } } } [ p _ { x , d } ( G ) ]$ , then

$$
\begin{array}{c} J _ { k , d } ( G ) = \frac { k ( k - 1 ) } { 2 } \big ( 1 - \mu _ { d } ( G ) \big ) ^ { k - 2 } \sigma _ { d } ^ { 2 } ( G )  \\ { + O \big ( \mathbb { E } _ { x \sim \mathcal { D } } \left[ | p _ { x , d } ( G ) - \mu _ { d } ( G ) | ^ { 3 } \right] \big ) . } \end{array}
$$

Hence, to second order, the penalty term is controlled by the cross-prompt dispersion of onesample success probabilities on the test set.

Proof. For $k \geq 2$ and $p \in ( 0 , 1 )$

$$
\begin{array} { l } { { f _ { k } ^ { \prime } ( p ) = k ( 1 - p ) ^ { k - 1 } > 0 , } } \\ { { f _ { k } ^ { \prime \prime } ( p ) = - k ( k - 1 ) ( 1 - p ) ^ { k - 2 } < 0 , } } \end{array}
$$

so $f _ { k }$ is strictly increasing and strictly concave on $( 0 , 1 )$

By definition, the expected mean is

$$
\mu _ { d } ( G ) = \mathbb { E } _ { x \sim \mathcal { D } } [ p _ { x , d } ( G ) ] ,
$$

and the expected pass rate is

$$
\operatorname { p a s s } @ \operatorname { k } _ { \mathcal { D } } ^ { ( d ) } ( G ) = \mathbb { E } _ { x \sim \mathcal { D } } [ f _ { k } ( p _ { x , d } ( G ) ) ] .
$$

Applying Jensen’s inequality to the concave function $f _ { k }$ ,

$$
\begin{array} { r l } { \mathrm { p a s s } @ \mathbf { k } _ { \mathcal { D } } ^ { ( d ) } ( G ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ f _ { k } ( p _ { \boldsymbol { x } , d } ( G ) ) ] } & { } \\ { \le f _ { k } ( \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ p _ { \boldsymbol { x } , d } ( G ) ] ) } & { } \\ { = f _ { k } ( \mu _ { d } ( G ) ) . } \end{array}
$$

Thus $J _ { k , d } ( G ) \ge 0$ . Since $f _ { k }$ is strictly concave, equality holds if and only if $p _ { x , d } ( G )$ is constant almost surely.

Now rewrite

$$
\mathrm { p a s s } @ \mathbf { k } _ { \mathcal { D } } ^ { ( d ) } ( G ) = f _ { k } \big ( \mu _ { d } ( G ) \big ) - J _ { k , d } ( G ) .
$$

Subtracting the identities for $G _ { 2 }$ and $G _ { 1 }$ yields

$$
\begin{array} { r l } & { \mathrm { p a s s } @ \mathbf { k } _ { \mathcal { D } } ^ { ( d ) } ( G _ { 2 } ) - \mathrm { p a s s } @ \mathbf { k } _ { \mathcal { D } } ^ { ( d ) } ( G _ { 1 } ) } \\ & { ~ = \Big ( f _ { k } \big ( \mu _ { d } ( G _ { 2 } ) \big ) - f _ { k } \big ( \mu _ { d } ( G _ { 1 } ) \big ) \Big ) } \\ & { ~ - \Big ( J _ { k , d } ( G _ { 2 } ) - J _ { k , d } ( G _ { 1 } ) \Big ) , } \end{array}
$$

which gives the exact decomposition.

For the second-order approximation, apply Taylor’s theorem around $\mu _ { d } ( G )$

$$
\begin{array} { r l } { f _ { k } ( p _ { x , d } ( G ) ) = f _ { k } ( \mu _ { d } ( G ) ) } \\ { \displaystyle } & { ~ + f _ { k } ^ { \prime } \big ( \mu _ { d } ( G ) \big ) \big ( p _ { x , d } ( G ) - \mu _ { d } ( G ) \big ) } \\ { \displaystyle } & { ~ + \frac { 1 } { 2 } f _ { k } ^ { \prime \prime } \big ( \mu _ { d } ( G ) \big ) \big ( p _ { x , d } ( G ) - \mu _ { d } ( G ) \big ) ^ { 2 } } \\ { \displaystyle } & { ~ + O ( | p _ { x , d } ( G ) - \mu _ { d } ( G ) | ^ { 3 } ) . } \end{array}
$$

Taking the expectation over $x \sim \mathcal { D } .$ , the linear term vanishes because $\mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ p _ { \boldsymbol { x } , d } ( G ) - \mu _ { d } ( G ) ] = 0$ Hence

$$
\begin{array} { r l r } {  { \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } [ f _ { k } ( \boldsymbol { p } _ { \boldsymbol { x } , d } ( G ) ) ] = f _ { k } ( \mu _ { d } ( G ) ) } } \\ & { } & { + \frac { 1 } { 2 } f _ { k } ^ { \prime \prime } ( \mu _ { d } ( G ) ) \sigma _ { d } ^ { 2 } ( G ) } \\ & { } & { + O \Big ( \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \big [ | \boldsymbol { p } _ { \boldsymbol { x } , d } ( G ) } \\ & { } & { - \mu _ { d } ( G ) | ^ { 3 } \big ] \Big ) . } \end{array}
$$

Rearranging gives

$$
\begin{array} { l } { { \displaystyle { J _ { k , d } ( G ) = - \frac { 1 } { 2 } f _ { k } ^ { \prime \prime } ( \mu _ { d } ( G ) ) \sigma _ { d } ^ { 2 } ( G ) } } } \\ { { \displaystyle ~ + O \Big ( \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } } \big [ | p _ { \boldsymbol { x } , d } ( G ) - \mu _ { d } ( G ) | ^ { 3 } \big ] \Big ) , } } \end{array}
$$

and substituting $f _ { k } ^ { \prime \prime } ( \mu ) = - k ( k - 1 ) ( 1 - \mu ) ^ { k - 2 }$ completes the proof. □

Connection to Empirical Regimes. Rather than positing a universal law applicable to all training environments, Theorem A.2 provides a rigorous analytical lens to interpret the specific empirical observations reported in Section 2.1. Specifically, while the theorem formally establishes the mathematical mechanism—that the dynamics of pass $@ \mathbf { k } _ { \mathcal { D } }$ are entirely governed by a trade-off between the mean improvement gain $( \Delta f _ { k } )$ and the cross-prompt variance penalty $( \Delta J _ { k , d } )$ —the actual increase of this variance $( \sigma _ { d } ^ { 2 } ( G ) )$ is an empirical characteristic dictated by the intrinsic difficulty of the dataset and specific training dynamics. Consequently, while these exact behavioral patterns may not identically manifest in every setup, bridging this theoretical mechanism with our specific findings allows us to formally characterize the three observed regimes as follows:

$$
\begin{array} { r l } & { \mathrm { E a s y : } \quad \Delta _ { G } J _ { k , \mathrm { E a s y } } ( G ) > \Delta _ { G } f _ { k , \mathrm { E a s y } } ( G ) } \\ & { \implies \quad \mathrm { p a s s } @ \mathrm { k } _ { \mathcal { D } } ^ { ( \mathrm { E a s y } ) } ( G + 1 ) < \mathrm { p a s s } @ \mathrm { k } _ { \mathcal { D } } ^ { ( \mathrm { E a s y } ) } ( G ) , } \\ & { \mathrm { H a r d : } \quad \Delta _ { G } J _ { k , \mathrm { H a r d } } ( G ) < \Delta _ { G } f _ { k , \mathrm { H a r d } } ( G ) } \\ & { \implies \quad \mathrm { p a s s } @ \mathrm { k } _ { \mathcal { D } } ^ { ( \mathrm { H a r d } ) } ( G + 1 ) > \mathrm { p a s s } @ \mathrm { k } _ { \mathcal { D } } ^ { ( \mathrm { H a r d } ) } ( G ) , } \end{array}
$$

$$
\begin{array} { r l } { \pmb { \mathrm { M e d i u m } } : } & { \Delta _ { G } J _ { k , \mathrm { M e d i u m } } ( G ) } \\ & { \mathrm { c r o s s e s } \ : \Delta _ { G } f _ { k , \mathrm { M e d i u m } } ( G ) } \\ { \Longrightarrow } & { \mathrm { p a s s } @ \mathbf { k } _ { \mathcal { D } } ^ { ( \mathrm { M e d i u m } ) } ( G ) \mathrm { i s ~ n o n - m o n o t o n i c } . } \end{array}
$$

where $\Delta _ { G } J _ { k , d } ( G ) = J _ { k , d } ( G + 1 ) - J _ { k , d } ( G )$ and $\Delta _ { G } f _ { k , d } ( G ) = f _ { k } ( \mu _ { d } ( G + 1 ) ) - f _ { k } ( \mu _ { d } ( G ) ) .$

Analytically, the second-order approximation reveals that the penalty term $J _ { k , d } ( G )$ is directly proportional to the cross-prompt variance $\sigma _ { d } ^ { 2 } ( G )$ This theoretically explains the paradoxical degradation in pass@ $\mathbf { k } _ { \mathcal { D } }$ observed when training on the Easy dataset. When the rollout budget G is large, the policy over-exploits specific, easily solvable templates rather than learning generalizable reasoning skills. Evaluated on a general test set, this causes severe polarization: the success probability $p _ { x , d } ( G )$ converges to 1 for familiar easy prompts but remains near 0 for unseen or harder problems. Consequently, the surge in the variance penalty $( \Delta _ { G } J _ { k , \mathrm { E a s y } } ( G ) )$ outweighs the marginal gains in mean performance $( \Delta _ { G } f _ { k , \mathrm { E a s y } } ( G ) )$ ), resulting in a decrease in the overall $\mathrm { p a s s } @ \mathrm { k } _ { \mathcal { D } }$

Conversely, on the Hard dataset, valid reasoning paths are scarce. An increased rollout budget promotes broader exploration, enabling the model to internalize more generalizable reasoning patterns. This uniformly improves the test-set mean $\mu _ { d } ( G )$ across prompts, maintaining low cross-prompt variance and keeping the associated penalty minimal.

## B Experimental Details of Section 2.1

We train Qwen2.5-3B-Instruct (Yang et al., 2024) on the MATH (Hendrycks et al., 2021) dataset, which we partition into Easy, Medium, and Hard subsets based on the base model’s empirical accuracy. Specifically, for each problem, we sample 12 independent solutions from the base model and categorize problems according to the number of correct samples. Problems for which all 12 samples are either entirely correct or entirely incorrect are excluded, as they provide no informative signal for distinguishing difficulty. Among the remaining problems, those with 1–4 correct samples are categorized as Hard, 5–8 as Medium, and 9–11 as Easy. This results in 2,217 Easy, 1,433 Medium, and 1,712 Hard problems. To ensure a balanced training distribution, we subsample each subset to 1,400 problems. The model was trained for 5 epochs on each difficulty-specific dataset.

<table><tr><td>Training Configuration</td><td>Wall Clock (Hours)</td></tr><tr><td>Easy  $/ G = 4$ </td><td>5.0</td></tr><tr><td>Easy  $/ G = 8$ </td><td>9.2</td></tr><tr><td>Easy  $/ G = 1 6$ </td><td>16.4</td></tr><tr><td>Medium  $/ G = 4$ </td><td>5.4</td></tr><tr><td>Medium  $I G = 8$ </td><td>9.4</td></tr><tr><td>Medium  $/ G = 1 6$ </td><td>16.6</td></tr><tr><td>Hard  $\bar { / } \bar { G } \bar { = } \bar { 4 }$ </td><td>5.5</td></tr><tr><td>Hard  $/ G = 8$ </td><td>9.9</td></tr><tr><td>Hard  $/ G = 1 6$ </td><td>17.5</td></tr></table>

Table 5: Wall Clock Time for each difficulty subset and rollout budget.

Using GRPO (Shao et al., 2024), we conduct a total of 9 training runs, applying rollout budgets of $G \in \{ 4 , 8 , 1 6 \}$ for each difficulty subset. Our implementation is based on a modified version of the GRPO-Zero repository<sup>2</sup>, which adopts a tokenlevel policy gradient without KL regularization or clipping. We use a binary verifiable reward, assigning 1 to correct final answers and 0 otherwise. During training, all samples are generated with temperature 1.0 and a maximum generation length of 1024 tokens. We fix all optimization hyperparameters, including a batch size of 256, while adjusting the number of problems per batch (64, 32, 16) to realize group sizes of 4, 8, and 16, respectively. All experiments are conducted on a single A100 GPU. The wall-clock times for each training configuration are reported in Table 5.

We evaluate the resulting 9 models on three benchmarks: MATH500, AIME25, and AIME24. For each problem, we generate 256 reasoning paths in parallel. We compute pass@k for $k \in$ {1, 2, 4, 8, 16, 32, 64, 128, 256}, and report results for each model trained under different difficulty subsets and rollout budgets (Figure 7). In addition, we aggregate results across the three benchmarks and report the averaged avg@256 and pass@256, as shown in Figure 1.

```latex
Algorithm 1 Tree Rollout
Input: Policy $\pi _ { \theta } .$ , Prompt x, Base rollouts N,
Forking points K, Branch rollouts $B ,$ Selection
method.
Output: Trajectory set $\tau$
Initialize $\mathcal { R }  \emptyset , \mathcal { T }  \emptyset$
Phase 1: Base Rollouts
for $i = 1$ to $N$ do
Sample base trajectory $\tau ^ { ( i ) } \sim \pi _ { \theta } ( \cdot \mid x )$
$\mathcal { R }  \mathcal { R } \cup \{ \tau ^ { ( i ) } \}$
${ \mathcal { T } } \gets { \mathcal { T } } \cup \{ \tau ^ { ( i ) } \}$
end for
Phase 2: Branch Rollouts
for each $\tau \in \mathcal { R }$ do
Select K forking points $\mathcal { F } \subset \{ 1 , \ldots , | \tau | \}$ via
Algorithm 2
for each $f \in { \mathcal { F } }$ do
Extract trajectory prefix $x _ { f } \gets [ x ; \tau _ { 1 : f } ]$
for $j = 1$ to B do
Sample continuation $\tau _ { f } ^ { \prime ( j ) } \sim \pi _ { \theta } ( \cdot \mid x _ { f } )$
$\tilde { \tau } _ { f } ^ { ( j ) } \gets [ \tau _ { 1 : f } ; \tau _ { f } ^ { \prime ( j ) } ]$
$\mathcal { T }  \mathcal { T } \cup \{ \tilde { \tau } _ { f } ^ { ( j ) } \}$
end for
end for
end for
```

## C Details on Tree-Structured Rollout (Section 2.2)

## C.1 Tree Algorithm Details

Algorithm 1 provides the detailed procedure of the two-phase tree rollout strategy.

## C.2 Experimental Details of Section 2.2.2

To evaluate cost-efficiency during inference, we measure the PassRate—the probability of discovering at least one correct solution—against token consumption on the AIME25 benchmark using Qwen2.5-3B-Instruct. We compare standard parallel sampling with two tree-structured variants: fixed-seg (equidistant branching) and random (uniform random branching). Detailed procedures for these forking strategies are provided in Appendix D.1.

![](images/cfe2925670f8ab4e4db216f1e92b19742249e34560ec05cb38e36c32c21fab99.jpg)  
(a) Easy / MATH500

![](images/e14c82238c825691b370e922c4498f8c73357f59080cf1f0936724280a8aeabc.jpg)  
(b) Easy / AIME25

![](images/5e6c350d6b03cc238fbafbf44167aba6a7fb7e919cb253945b38bfbd2adbd8a0.jpg)

![](images/77501a19875d2f806ca960358215783a1381fbaccbc157998bc9b00b7288c08e.jpg)  
(d) Medium / MATH500

(c) Easy / AIME24  
![](images/fe8a24887ebb24069c166c059ba370b82b2525ce157235ba97dfc6f54b786e33.jpg)

![](images/57f22c4667904df40211b42a72ce482d7cc64bd00475c5094bad1be2b13289cc.jpg)

![](images/b97c970709196457f74df2f8f0143769d747c4ddaf7bb7cec935c1ef2a942e9a.jpg)  
(g) Hard / MATH500

(f) Medium / AIME24  
(e) Medium / AIME25  
![](images/c194c90dc7287bfcf9939d8ab7bba1cc0ba355dc4bbaf4ad1738c5d54a867482.jpg)  
(h) Hard / AIME25

![](images/d1c2e96cb4c979a75ca247afa1ad2357bbe91c869f5c68bc650b8144e1ba08f9.jpg)  
(i) Hard / AIME24  
Figure 7: pass@k performance on evaluation benchmarks for models trained on different difficulty subsets. Labels such as easy\_8 denote models trained on the Easy subset with group size $G = 8 .$

Formal Definition of PassRate For parallel sampling, the model generates M independent trajectories, $\mathcal { T } _ { \mathrm { p a r a l l e l } } = \{ \tau ^ { ( 1 ) } , \dots , \tau ^ { ( M ) } \}$ . The PassRate is defined as:

$$
{ \mathrm { P a s s R a t e } } _ { \mathrm { p a r a l l e l } } = \mathbb { I } \left( \sum _ { i = 1 } ^ { M } r ( \tau ^ { ( i ) } ) \geq 1 \right)
$$

where $r ( \cdot ) \in \{ 0 , 1 \}$ is the verifiable binary reward.   
Token consumption scales linearly with M.

For tree-structured rollouts, generating N base rollouts, K forking points per base, and B branches per forking point yields a set of leaf trajectories $\tau _ { \mathrm { { t r e e } } }$ . The total number of terminal paths is $N ( 1 +$ $K B )$ . The PassRate is evaluated over these leaves:

$$
\mathrm { P a s s R a t e } _ { \mathrm { t r e e } } = \mathbb { I } \left( \sum _ { \tau \in \mathcal { T } _ { \mathrm { t r e e } } } r ( \tau ) \geq 1 \right)
$$

Unlike parallel sampling, tree rollouts consume tokens sub-linearly relative to the number of terminal paths, as context prefixes up to each forking point are computed only once and shared across branches.

Configurations To analyze PassRate against token consumption (Figure 2), we varied the computational budget for both methods. For parallel sampling, we evaluated independent rollout counts of $M \in \{ 4 , 8 , 1 6 , 3 2 \}$ . For tree rollouts, we fixed the branching parameters at $K = 2$ and $B = 2$ and scaled the base rollouts $N \in \{ 2 , 4 , 8 \}$ . These configuration tuples $( N , K , B )$ yield 10, 20, and 40 terminal paths, respectively, enabling a direct comparison of exploration efficiency relative to token expenditure. Detailed forking point selection algorithms are provided in Appendix D.1.

## D Details on Forking Point Selection Strategies (Section 2.3)

## D.1 Forking Point Selection Algorithms

We detail the five forking point selection algorithms in Algorithm 2, which is utilized in Phase 2 Branch Rollouts. Let a trajectory be denoted as a sequence of tokens $\tau = ( y _ { 1 } , y _ { 2 } , \dotsc , y _ { | \tau | } )$ . For entropy-based methods, let $H _ { t }$ denote the token-level entropy at step t, which is computed using the language model’s probability distribution over the vocabulary V:

Algorithm 2 Forking Points Selection Algorithms   
Input: Trajectory τ, Number of forking points   
K, Selection method, Token entropies $\{ H _ { t } \} _ { t = 1 } ^ { | \tau | }$   
Parsed sentences $\begin{array} { c c l } { { \mathcal S } } & { { = } } & { { \{ S _ { 1 } , . . . , S _ { M } \} } } \end{array}$ , Sen  
tence entropies $\lbrace H _ { m } ^ { S } \rbrace _ { m = 1 } ^ { M }$ , Parsed steps $\mathcal { Z }$   
$\{ Z _ { 1 } , \ldots , Z _ { L } \}$ , Step-level FCI scores $\{ y _ { l } \} _ { l = 1 } ^ { L } .$   
Output: Forking points $\mathcal { F } \subset \{ 1 , \ldots , | \tau | \}$ , where   
$| { \mathcal { F } } | = K$   
if method is random then   
Sample $\mathcal { F } \subset \{ 1 , \ldots , | \tau | \}$ uniformly at ran  
dom   
else if method is fixed-seg then   
${ \mathcal { F } } \gets \{ | \boldsymbol { k } \cdot \frac { | \boldsymbol { \tau } | } { K + 1 } | | k = 1 , \ldots , K \}$   
else if method is ATB then   
$\mathcal { C } \gets \{ l \in \{ 1 , \dots , L \} \ |$   
$y _ { l } \geq \mathrm { Q u a n t i l e } ( y _ { 1 } , \ldots , y _ { L } , 0 . 8 )  \}$   
$\mathcal { L } ^ { \ast } \gets$ the $K$ smallest indices (earliest steps)   
from C   
$\mathcal { F }  \{ \nu _ { l } \ | \ l \in \mathcal { L } ^ { * } \}$   
else if method is tok-entropy then   
$\mathcal { F } $ arg top− $. K H _ { t }$   
$t { \in } \{ 1 , { \stackrel {  } { \dots } } , | \tau | \}$   
else if method is sent-entropy then   
$\mathcal { M } ^ { \ast } \gets \arg \operatorname { t o p } { - K H _ { m } ^ { S } }$   
$\mathfrak { m } \in \{ 1 , . . . , M \}$   
$\mathcal { F } \gets \{ \mu _ { m } \ | \ m \in \mathcal { M } ^ { * } \}$   
end if   
return $\mathcal { F }$

$$
H _ { t } = - \sum _ { v \in V } P ( v \mid y _ { < t } ) \log P ( v \mid y _ { < t } )
$$

For the sentence-level method, we parse the trajectory into M sentences, $\mathcal { S } = \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { M } \}$ by aligning character-level sentence boundaries detected via PySBD (Sadvilkar and Neumann, 2020) with the tokenizer’s offset mapping. Each sentence $S _ { m }$ represents a contiguous span of token indices, and $\mu _ { m }$ denotes the start token index of $S _ { m }$ . The sentence entropy $H _ { m } ^ { S }$ is defined as the average token entropy within that sentence to mitigate length bias:

$$
H _ { m } ^ { S } = { \frac { 1 } { \left| S _ { m } \right| } } \sum _ { t \in S _ { m } } H _ { t }
$$

Attention-based Tree Branching (ATB) operates on the premise that attention scores serve as meaningful metrics to identify important reasoning behaviors. To leverage this insight, the trajectory is first segmented into L discrete steps $( \mathrm { e . g . }$ , delimited by line breaks), denoted as $\mathcal { Z } = \{ Z _ { 1 } , \ldots , Z _ { L } \}$ where $\nu _ { l }$ represents the start token index of step $Z _ { l }$ ATB then computes a Forward Context Influence (FCI) score $y _ { l }$ for each step, quantifying its influence on subsequent tokens by aggregating attention weights. When selecting forking points, ATB first isolates a candidate set consisting of the top 20% of steps with the highest FCI scores. From this set, it selects the K earliest steps as the final forking points.

<table><tr><td>Metric</td><td>random</td><td>sent-entropy</td><td>tok-entropy</td></tr><tr><td>NND↓</td><td>0.157</td><td>0.130</td><td>0.106</td></tr><tr><td>MPD↓</td><td>0.337</td><td>0.296</td><td>0.251</td></tr><tr><td>WCR@5↑</td><td>8.9</td><td>12.9</td><td>23.6</td></tr><tr><td>WCR@10↑</td><td>16.4</td><td>25.3</td><td>35.3</td></tr></table>

Table 6: Quantitative comparison of forking-point localization across different selection strategies. Arrows indicate stronger localization.

Lastly, the operator arg top−K denotes the extraction of the subset of indices that correspond to the K highest values of a given metric.

## D.2 Quantitative and Qualitative Analysis of Forking-Point Localization

As illustrated in Figure 8, high-entropy tokens tend to densely cluster within narrow, highly uncertain segments of the reasoning trajectory. Consequently, the search budget is monopolized by repeated resampling at a single localized point, which severely limits the structural reach of the search tree.

To quantitatively characterize the localization phenomenon, we measure the distribution of selected forking points using three metrics: Nearest Neighbor Distance (NND), the average distance to the closest forking point; Mean Pairwise Distance (MPD), the average pairwise distance among all selected points; and Window-based Clustering Ratio (WCR@P), the percentage of forking-point pairs whose distance falls within $P \%$ of the trajectory length. Lower NND and MPD and higher WCR indicate stronger localization.

As shown in Table $^ { 6 , }$ tok-entropy exhibits the strongest localization, with the lowest NND (0.106) and MPD (0.251) and the highest WCR@5 (23.6%) and WCR@10 (35.3%). In comparison, sent-entropy increases NND and MPD to 0.130 and 0.296, respectively, while reducing WCR@5 and WCR@10 to 12.9% and 25.3%, quantitatively demonstrating that sentence-level entropy mitigates the localization of forking points.

To further observe how the selected tokens differ across forking strategies, we conducted a case study comparing tok-entropy, sent-entropy, and ATB. Figure 9 visualizes the forking points $( K = 4 )$ selected by each method along a single, fixed reasoning path. The tok-entropy method suffers from localization, trapping the selected points within a narrow segment. In contrast, sent-entropy still targets the model’s uncertain regions but distributes the forking points much more evenly, operating at a broader semantic level. Meanwhile, ATB selects steps receiving the highest attention weights; while this can be meaningful for dividing logical reasoning steps, it operates independently of the model’s uncertainty. As shown in Figure 9, ATB often selects highly deterministic points to branch from. Performing additional sampling at these points yields low SibDiv and, consequently, a limited PassRate, as reflected in Table 1. Based on these observations, sent-entropy proves to be the most effective forking strategy by successfully leveraging uncertainty signals while resolving the localization issue.

## D.3 Formal Definitions of SibDiv

Because recent studies emphasize semantic diversity for effective exploration (Jiang et al., 2025; Yao et al., 2025), it is critical to rigorously quantify the divergence of the explored reasoning paths discussed in Section 2.3.2. To this end, we model the generated reasoning tree as a collection of text segments, referred to as blocks. Formally, a block is defined as a contiguous sequence of reasoning spanning from the root to the first forking point, between two consecutive forking points, or from a final forking point to a leaf.

Let $\mathcal { F }$ denote the set of all forking points and $\boldsymbol { B }$ denote the set of all blocks within a single generated reasoning tree. We extract a dense semantic representation $\mathbf { e } _ { b }$ for each block $b \in B$ utilizing the $\mathtt { g t e - l a r g e - e n - v 1 }$ .5 model (Zhang et al., 2024). Sibling Diversity (SibDiv) is formulated as the complement of the average pairwise cosine similarity among sibling blocks:

$$
S i b D i \nu = 1 - \frac { 1 } { | \mathcal { F } | } \sum _ { f \in \mathcal { F } } \frac { \sum _ { b _ { i } , b _ { j } \in B ( f ) , i < j } \cos ( \mathbf { e } _ { b _ { i } } , \mathbf { e } _ { b _ { j } } ) } { \binom { | B ( f ) | } { 2 } }\tag{6}
$$

where $B ( f ) \subset B$ represents the specific subset of child blocks branching directly from a given forking point $f ,$ and $\cos ( \cdot , \cdot )$ denotes the cosine similarity function.

<table><tr><td>Method</td><td>PassRate</td><td>SibDiv</td></tr><tr><td>random</td><td>10.0</td><td>0.0796</td></tr><tr><td>fixed-seg</td><td>13.3</td><td>0.0853</td></tr><tr><td>ATB</td><td>14.7</td><td>0.0874</td></tr><tr><td>tok-entropy</td><td>12.0</td><td>0.1065</td></tr><tr><td>tok-entropy w/ 5% dist.</td><td>14.0</td><td>0.1041</td></tr><tr><td>tok-entropy w/ 10% dist.</td><td>15.3</td><td>0.1007</td></tr><tr><td>sent-entropy</td><td>17.3</td><td>0.1049</td></tr></table>

Table 7: Comparison of forking-point selection strategies, including distance-forced token-entropy baselines.

A higher SibDiv score indicates that the model successfully generates semantically distinct continuations from its forking points. By aggregating this localized diversity, SibDiv effectively captures the degree of semantic exploration and reasoning diversity across the entire tree.

## D.4 Experimental Details of Section 2.3.3

We evaluated each forking strategy by generating reasoning trees under a fixed inference budget, measuring SibDiv and PassRate. Specifically, we conducted inference on the AIME26 benchmark using the Qwen2.5-3B-Instruct model. To isolate the effect of the forking point selection strategy from the inherent randomness of language model generation, we generated and fixed a single base rollout for each problem. We then applied each selection method to identify $K = 4$ distinct forking points along this fixed base trajectory. From each identified forking point, we generated $B = 4$ additional branch rollouts. This procedure ensures a strictly controlled environment, yielding exactly 17 terminal leaves (1 base + 16 new branches) per problem across all methods. Finally, we calculated the Pass-Rate and SibDiv across these 17 leaves and averaged the metrics over the entire dataset. To ensure the robustness of our findings, we repeated this entire experimental procedure five independent times and report the final averaged results in Table 1.

## D.5 Analyzing the Benefits of Sentence-Level Entropy

While sent-entropy substantially improves Pass-Rate over tok-entropy, it remains unclear whether this gain arises simply from mitigating localization or from identifying more semantically meaningful decision points. To disentangle these effects, we introduce distance-forced tok-entropy

![](images/dbd2e52050b6753afcf90a9e7a70d461a1f9d10b1591b0eacba11d7edff970c2.jpg)  
Figure 8: Visualization of token-entropy localization. Dark red boxed tokens indicate the selected top-4 high-entropy forking points. These points densely cluster within a narrow segment, monopolizing the search budget and limiting structural diversity.

![](images/62a27316360e9e3be1673dee29b1aeae816a23a413b53d2b115a27705c05d2e0.jpg)

## Forking Point Selection Methods

![](images/d2e0856d541bf64fc2b4af549b8e9db10d14d78359d47d575e98a4a153a08a94.jpg)

Figure 9: Case studies comparing forking point selections across tok-entropy, sent-entropy, and ATB. Colored boxes indicate the tokens selected by each method, with overlapping boxes denoting agreement between methods. Notably, while tok-entropy suffers from localization, sent-entropy distributes points more evenly across broader semantic levels, whereas ATB often targets deterministic points regardless of uncertainty.

baselines that preserve the original token-level entropy signal while explicitly preventing nearby forking points. Starting from candidates ranked by token entropy, we greedily select forking points and skip any candidate whose distance from an already selected point is within a predefined fraction of the total trajectory length. We consider minimum-distance thresholds of 5% and 10%.

As shown in Table 7, explicitly enforcing a minimum distance between token-level forking points improves PassRate from 12.0 for standard tok-entropy to 14.0 and 15.3 with the 5% and 10% constraints, respectively. This confirms that localization itself is a limiting factor for token-level entropy-based branching. However, both distanceforced variants still underperform sent-entropy, which achieves a PassRate of 17.3. Since these variants mitigate localization while retaining the same token-level entropy signal, the remaining gap suggests that the benefit of sent-entropy cannot be attributed solely to distributing forking points more broadly. Rather, sentence-level entropy provides a more effective signal for identifying semantically meaningful decision points for branching.

## E Algorithmic and Formal Details of DATPO

This section provides the formal definitions of the advantage components and the complete training procedure for Difficulty-Adaptive Sentenceentropy Guided Tree-Structured Policy Optimization (DATPO), expanding upon the methodology outlined in Section 3.

## E.1 Formal Definitions of Value Estimation and Sibling Diversity term

Monte Carlo State Value $( \hat { V } _ { M C } )$ For a state s corresponding to a specific forking point, let $\boldsymbol { \mathcal { T } } ( \boldsymbol { s } )$ denote the set of all terminal blocks descending from s. The Monte Carlo state value is formally defined as the expected verifiable reward over these terminal paths:

$$
\hat { V } _ { M C } ( s ) = \frac { 1 } { | \mathscr { T } ( s ) | } \sum _ { \tau \in \mathscr { T } ( s ) } r ( \tau ) .\tag{7}
$$

For a terminal state $s _ { \mathrm { e n d } }$ , the set of descending paths is empty, strictly enforcing $\hat { V } _ { M C } ( s _ { \mathrm { e n d } } ) = 0$

Block-Level Sibling Diversity $( \mathbf { D i v _ { s i b } } ( b ) )$ Let $P ( b )$ denote the forking point from which block b originates, and let $\mathcal { B } _ { \mathrm { s i b } } ( b )$ denote the set of all

sibling blocks branching from $P ( b )$ . The siblingdiversity term for a specific block b is computed as its average semantic distance to its siblings:

$$
\mathrm { D i v } _ { \mathrm { s i b } } ( b ) = 1 - \frac { \sum _ { b ^ { \prime } \in \mathcal { B } _ { \mathrm { s i b } } ( b ) \backslash \{ b \} } \cos ( \mathbf { e } _ { b } , \mathbf { e } _ { b ^ { \prime } } ) } { | \mathcal { B } _ { \mathrm { s i b } } ( b ) | - 1 } ,\tag{8}
$$

where $\mathbf { e } _ { b }$ is the embedding of block b. We define $\mathrm { D i v } _ { \mathrm { s i b } } ( b ) = 0$ when b has no valid sibling block.

While the evaluation metric SibDiv (defined in Appendix D.3) aggregates pairwise similarities across all forking points to report a single treelevel scalar, the reward term $\mathrm { D i v _ { s i b } } ( b )$ is evaluated at the individual block level to provide the advantage augmentation.

## E.2 DATPO Objective Function

DATPO employs a token-level policy gradient objective mapped across the tree topology. Given a set of generated blocks $\boldsymbol { B }$ for a prompt q, the objective is defined as:

$$
\begin{array} { c l } { \mathcal { I } _ { \mathrm { D A T P O } } ( \theta ) = \mathbb { E } _ { q \sim Q , \mathcal { B } \sim \pi _ { \theta _ { \mathrm { o l d } } } } \Bigg [ \frac { 1 } { \sum _ { b \in \mathcal { B } } \vert b \vert } \underset { b \in \mathcal { B } } { \sum } \underset { t = 1 } { \overset { \vert b \vert } { \sum } } } \\ { \operatorname* { m i n } \left( \rho _ { b , t } ( \theta ) \hat { A } ( b ) , \mathrm { c l i p } \big ( \rho _ { b , t } ( \theta ) , \quad } & { } \\ { 1 - \epsilon , 1 + \epsilon \big ) \hat { A } ( b ) \right) \Bigg ] , } \end{array}\tag{9}
$$

where |b| is the sequence length of block b, $\begin{array} { r } { \rho _ { b , t } ( \theta ) = \frac { \pi _ { \theta } \left( y _ { t } | \boldsymbol x _ { < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { t } | \boldsymbol x _ { < t } \right) } } \end{array}$ is the importance ratio, and $\hat { A } ( b )$ is the block-level augmented advantage.

## E.3 Complete DATPO Algorithm

Algorithm 3 details the end-to-end training procedure. By extending the two-phase rollout strategy into the training framework, the process is structured into four distinct phases: base rollouts, adaptive branch rollouts, block-level advantage estimation, and policy update.

## E.4 Comparison with Other Tree-based Methods

To contextualize DATPO within the landscape of recent tree-based RLVR methodologies, we highlight two fundamental distinctions between our approach and existing frameworks such as TreeRL (Hou et al., 2025) and AttnRL (Liu et al., 2025a).

Algorithm 3 DATPO mon trajectory prefixes are duplicated across mul-  
Input: Initial policy $\pi _ { \theta } .$ Prompts dataset Q, Max- tiple sequences, causing the policy to redundantly   
imum number of forking points $K _ { \mathrm { m a x } }$ , Maxi- update on the same early tokens. To mitigate the re  
mum number of branch rollouts per forking point sulting overfitting, these methods rely on a heuristic   
$B _ { \mathrm { m a x } } .$ Base rollouts $N ,$ , Diversity coefficient $\alpha .$ penalty, artificially downweighting the advantage   
Output: Optimized policy $\pi _ { \theta } ^ { * }$ of non-leaf steps by dividing it by the square root of   
Sample prompt $q \sim \mathcal { Q }$ the descending leaf count $( { \sqrt { | L ( s _ { n } ) | } } )$ (Hou et al.,   
Phase 1: Base Rollouts 2025; Liu et al., 2025a).   
Generate N independent base trajectories $\mathcal { R } =$ In contrast, DATPO is structurally immune to   
$\{ \tau ^ { ( 1 ) } , \dots , \tau ^ { ( N ) } \}$ via $\pi _ { \boldsymbol { \theta } } ( \cdot \mid \boldsymbol { q } )$ this redundancy. By strictly partitioning the gener-  
Compute $\begin{array} { r } { V ( \mathrm { r o o t } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r ( \tau ^ { ( i ) } ) } \end{array}$ ated tree into non-overlapping contiguous blocks,   
Compute adaptive budgets: $\hat { K } = \lceil K _ { \mathrm { m a x } } ( 1 -$ every token within the tree belongs to exactly one   
$V ( \mathrm { r o o t } ) ) \vert , \hat { B } = \lceil B _ { \mathrm { m a x } } ( 1 - V ( \mathrm { r o o t } ) ) \rceil$ block. Because DATPO performs explicit block-  
Phase 2: Adaptive Branch Rollouts level updates directly over the tree topology, it   
Initialize tree path set $\tau  \mathcal { R }$ naturally prevents duplicate gradient updates on   
if $\hat { K } > 0$ and $\hat { B } > 0$ then shared prefixes, providing a highly robust credit as  
for each $\tau \in \mathcal { R }$ do signment mechanism without the need for ad-hoc   
Select $\hat { K }$ forking points $\mathcal { F }$ using advantage penalization.   
sent-entropy forking strategy (Al-  
Motivation for Difficulty-Adaptive Exploration   
gorithm 2)   
While AttnRL (Liu et al., 2025a) similarly incorpo  
for each $f \in { \mathcal { F } }$ do   
rates a difficulty-adaptive exploration mechanism,   
Extract trajectory prefix $x _ { f } \gets [ q ; \tau _ { 1 : f } ]$   
Generate $\hat { B }$ branch continuations from its approach is fundamentally driven by computa  
tional efficiency. Specifically, AttnRL reduces the   
prefix $x _ { f }$   
sampling budget for easy problems to prevent in-  
Add generated branches to $\tau$   
efficient exploration and filter out responses with   
end for   
zero advantages.   
end for   
Conversely, DATPO’s difficulty-adaptive rollout   
end if   
originates from a completely different motivation:   
Phase 3: Block-level Advantage Estimation   
maximizing intrinsic reasoning coverage (pass@k).   
Partition T into a set of contiguous blocks $\boldsymbol { B }$   
for each block $b \in B$ do As demonstrated in Section 2.1, adaptive budget   
Compute $\hat { V } _ { M C } ( s _ { \mathrm { s t a r t } } ^ { ( b ) } )$ and $\hat { V } _ { M C } ( s _ { \mathrm { e n d } } ^ { ( b ) } )$ allocation is not merely a resource-saving heuris  
tic, but a key algorithmic factor for successfully   
$\hat { A } _ { \mathrm { b a s e } } ( b ) = r ( b ) + \hat { V } _ { M C } ( s _ { \mathrm { e n d } } ^ { ( b ) } ) - \hat { V } _ { M C } ( s _ { \mathrm { s t a r t } } ^ { ( b ) } )$ expanding a model’s pass@k without inducing   
Compute $\mathrm { D i v _ { s i b } } ( b )$ using block embeddings   
mode collapse. Built entirely around this objective,   
(Equation 8) DATPO further maximizes coverage through the   
$\hat { A } ( \bar { b } ) = \hat { A } _ { \sf b a s e } ( b ) + \mathbb { I } ( \hat { A } _ { \sf b a s e } ( b ) > 0 ) \cdot \alpha \mathrm { ~ . ~ }$ structural superiority of trees (Section 2.2) and op   
$\mathrm { D i v _ { s i b } } ( b )$ timal semantic forking (Section 2.3). Thus, while   
end for DATPO and AttnRL share superficial similarities   
Phase 4: Policy Update   
in their adaptive nature, they are built upon fun-  
Update θ by maximizing J<sub>DATPO</sub>(θ) (Equa- damentally distinct motivations and optimization   
tion 9) using advantages $\{ { \hat { A } } ( b ) \} _ { b \in B }$ goals.   
Anneal α according to schedule   
return Optimized policy $\pi _ { \theta } ^ { * }$ F Details on Main Experiments F Details on Main Experiments

Structural Robustness in Optimization Both TreeRL and AttnRL generate tree-structured rollouts but ultimately flatten these trees into a set of $N ( 1 + K \times B )$ independent sequences to perform sequence-level policy updates (Hou et al., 2025). A critical flaw in this unfolding process is that com-

## (Section 4)

## F.1 Experimental Setups

Models and Datasets We use Qwen2.5-3B-Base (Yang et al., 2024) and Qwen3-4B-Base (Yang et al., 2025) as base models. For training, we use the MATH dataset (Hendrycks et al., 2021). From the full set of 12,500 problems, we exclude the

500 problems used as the MATH500 and use the remaining 12,000 problems as the training set.

Evaluation We evaluate the trained models on mathematical reasoning benchmarks, specifically MATH500 (Hendrycks et al., 2021), AIME26, AIME25, AIME24, and AMC23. We report avg@k as the primary evaluation metric across all benchmarks. All evaluations were performed with a temperature of 1.0 and a maximum generation length of 2048. To assess the reasoning coverage and test-time scaling performance of the trained models, we additionally report pass@k and maj@k (majority voting), respectively. We use $k = 8$ for MATH500, and $k = 6 4$ for the remaining benchmarks. For robustness, both pass@k and maj@k results are computed by averaging over three independent evaluation runs for each dataset. The final maj@k in Figure 6 is then reported as the average of these results across all mathematical reasoning benchmarks.

Baselines We compare DATPO with several baselines: (1) Base, the base model without any additional training; (2) GRPO (Shao et al., 2024), standard GRPO enhanced with the token-level loss and clip-higher strategy from DAPO (Yu et al., 2025b). Following prior work (Liao et al., 2025) showing that dynamic sampling can be detrimental for relatively weak reasoning models, we intentionally adopt only this specific subset of DAPO components; (3) Dr.GRPO (Liu et al., 2025b), a refined variant of GRPO that corrects structural biases in advantage estimation; (4) TreeRL (Hou et al., 2025), a tree-based RL method that selects forking points from high-entropy tokens and estimates process-level advantages by combining local and global advantages; and (5) AttnRL (Liu et al., 2025a), another tree-based RL method built upon TreeRL with a focus on improving efficiency by selecting forking points based on attention scores and adopting adaptive sampling and one-step off-policy. For TreeRL and AttnRL, we adopt the same treeexpansion hyperparameters as reported in original papers. To ensure a fair comparison, we control the total number of generated tokens per problem to be comparable across all methods (detailed in Appendix F.3).

## F.2 Implementation Details

Sampling and Algorithmic Hyperparameters Across all methods, we set the sampling temperature to 1.0 and the maximum generation length to

<table><tr><td>Method</td><td>Avg. Generated Tokens</td></tr><tr><td>GRPO</td><td>8,755.7</td></tr><tr><td>Dr.GRPO</td><td>8,608.3</td></tr><tr><td>TreeRL</td><td>11,029.9</td></tr><tr><td>AttnRL</td><td>8,807.3</td></tr><tr><td>DATPO (Ours)</td><td>8,952.4</td></tr></table>

Table 8: Average number of generated tokens per question during training across different methods.

1024 tokens. For GRPO and Dr.GRPO, we sample 16 responses per problem. For the tree-based baselines, we strictly follow the hyperparameters reported in their respective original papers. Specifically, for TreeRL, we set the tree expansion parameters to $( N , K , B ) = ( 6 , 2 , 2 )$ . For AttnRL, we use maximum budgets of $( N , K _ { \mathrm { m a x } } , B _ { \mathrm { m a x } } ) =$ (6, 2, 2), select the top 20% of FCI steps as forking candidates, and apply $\Delta = 4$ and λ = 0.9 for adaptive batching. For our proposed DATPO, we set the adaptive tree parameters to $( N , K _ { \mathrm { m a x } } , B _ { \mathrm { m a x } } ) =$ (4, 3, 4). We intentionally employ a larger maximum branch count $( B _ { \mathrm { m a x } } = 4 )$ compared to the baselines, as calculating the sibling diversity term $( \mathrm { D i v } _ { \mathrm { s i b } } )$ across merely two branches $( B = 2 )$ diminishes its significance. To accommodate this wider branching while keeping the total number of generated tokens per problem comparable to the baseline methods, we adjusted the base and forking budgets accordingly. The diversity coefficient α is initialized to 0.2 and linearly annealed to 0 over the course of training. To compute the semantic diversity representations, we utilize the gte-large-en-v1.5 embedding model (Zhang et al., 2024). Furthermore, to prevent Out-Of-Memory (OOM) errors caused by the expansion of adaptive methods (AttnRL and DATPO), we apply rollout chunking during the branch rollout phase. We generate branch rollouts in chunk sizes of 384 for Qwen2.5-3B-Base and 64 for Qwen3- 4B-Base.

Training Hyperparameters We formulate the task with a binary verifiable reward, assigning a reward of 1 if the final answer is correct and 0 otherwise. Models are optimized using the AdamW optimizer with $\beta = ( 0 . 9 , 0 . 9 9 9 )$ and a constant learning rate of $5 \times 1 0 ^ { - 6 }$ . We omit the KL divergence penalty and the clipping ratio is set to 0.2. For the GRPO baseline enhanced with the clip-higher strategy, we set the upper clip ratio to

<table><tr><td>Method</td><td>Wall Clock (Hours)</td></tr><tr><td>Parallel Rollout</td></tr><tr><td>GRPO 26.3</td></tr><tr><td>Dr.GRPO 27.1</td></tr><tr><td>Tree-based Rollout</td></tr><tr><td>TreeRL 56.2</td></tr><tr><td>AttnRL 55.5</td></tr><tr><td>DATPO w/o diversity 56.0</td></tr><tr><td>DATPO 60.3</td></tr><tr><td>DATPO w/ one-step off-policy 60.0</td></tr></table>

Table 9: Wall-clock comparison across different methods.

0.28 and the lower to 0.2. During training, we update the policy using 16 prompts per step, maintaining a global mini-batch size of 64 and a microbatch size of 2 for gradient accumulation across all methods. The total effective batch size varies by method: GRPO and Dr.GRPO utilize a fixed batch size of 256; TreeRL employs a batch size of 480 (comprising 96 base rollouts and 384 branch rollouts); whereas the batch sizes for AttnRL and DATPO dynamically adjust per step due to their adaptive nature. Notably, because DATPO performs block-level policy updates, its mini-batch and micro-batch sizes are defined with respect to the number of contiguous blocks rather than full sequences. All experiments were executed on a computing cluster equipped with 8 NVIDIA A100 GPUs, where each individual training run was conducted on a single A100 GPU. Our codebase is developed based on the GRPO-Zero framework<sup>3</sup>.

## F.3 Computational Cost and Wall-Clock Time

To ensure a fair comparison, we control the total number of generated tokens per problem to be comparable across all methods. The actual average number of generated tokens per question during training for each method is detailed in Table 8. However, despite this token-level equivalence, treebased methods generally incur a higher wall-clock time compared to standard parallel rollouts, as reported in Table 9. This discrepancy arises from the inherent sequential dependency in tree generation: branch rollouts cannot commence until the base rollouts are fully generated and forking points are explicitly selected, creating a computational bottleneck. Furthermore, our proposed DATPO requires additional overhead to compute block-level embeddings for the diversity term, slightly increasing its wall-clock time relative to other tree-based baselines. We note that adopting the one-step off-policy approach proposed in AttnRL (Liu et al., 2025a) can marginally mitigate this latency. Table 10 provides a detailed breakdown of the per-step runtime of DATPO by computational component.

<table><tr><td>Component</td><td>Time (s)</td><td>Total runtime (%)</td></tr><tr><td>Policy</td><td>150.44</td><td>50.14</td></tr><tr><td>Diversity embedding</td><td>13.65</td><td>4.55</td></tr><tr><td>Update</td><td>102.17</td><td>34.05</td></tr><tr><td>Others</td><td>33.76</td><td>11.25</td></tr><tr><td>Total</td><td>300.02</td><td>100.00</td></tr></table>

Table 10: Per-step runtime breakdown of DATPO across different computational components.

![](images/d40cdd4576d39feae15d1c94716954d8ee4b382effb43ec3596801586f1779ab.jpg)  
Figure 10: Comparison of category-wise distributions between the original MMLU-Pro dataset and our sampled subset (500 instances). Stratified sampling ensures the original proportions are preserved.

## F.4 Generalization to Out-Of-Domain Tasks

To investigate whether the models trained on mathematical reasoning datasets can lead to improvements in reasoning performance across other domains, we additionally conduct out-of-domain evaluations using GPQA-Diamond (Rein et al., 2023) and MMLU-Pro (Wang et al., 2024). For MMLU-Pro, we applied stratified sampling to evaluate on a subset of 500 instances, preserving the original proportional distribution across categories such as math, physics, chemistry, engineering, law, and economics. A comparison of the category-wise distributions between the original and sampled datasets is provided in Figure 10. We evaluate the models trained with each method using two base models, Qwen2.5-3B-Base and Qwen3-4B-Base, and report the avg@8 metric for both benchmarks.

Table 11 presents the results of the out-ofdomain evaluation. Overall, DATPO demonstrates consistent performance across both the GPQA-Diamond and MMLU-Pro datasets. Specifically, compared to the strongest baselines, DATPO achieves a +2.4 percentage point improvement on

<table><tr><td>Method</td><td>GPQA-Diamond</td><td>MMLU-Pro</td></tr><tr><td colspan="3">Qwen2.5-3B-Base</td></tr><tr><td>Base</td><td>21.1</td><td>18.4</td></tr><tr><td>GRPO</td><td>25.3</td><td>32.6</td></tr><tr><td>Dr.GRPO</td><td>26.1</td><td>31.2</td></tr><tr><td>TreeRL</td><td>25.3</td><td>33.1</td></tr><tr><td>AttnRL</td><td>25.9</td><td>33.2</td></tr><tr><td>DATPO (Ours)</td><td>28.5</td><td>33.1</td></tr><tr><td colspan="3">Qwen3-4B-Base</td></tr><tr><td>Base</td><td>24.2</td><td>34.8</td></tr><tr><td>GRPO</td><td>35.9</td><td>55.9</td></tr><tr><td>Dr.GRPO</td><td>38.2</td><td>57.3</td></tr><tr><td>TreeRL</td><td>36.7</td><td>55.6</td></tr><tr><td>AttnRL</td><td>38.4</td><td>56.6</td></tr><tr><td>DATPO (Ours)</td><td>39.6</td><td>57.7</td></tr></table>

Table 11: Out-of-domain evaluation results. We report avg@8 accuracy for each dataset.

GPQA-Diamond and a marginal -0.1 point difference on MMLU-Pro using the Qwen2.5-3B-Base model. When applied to the Qwen3-4B-Base model, it yields consistent improvements of +1.2 and +0.4 percentage points on the respective benchmarks. These results demonstrate that the enhancements in reasoning capacity observed on in-domain mathematical benchmarks successfully generalize to improved reasoning performance on out-of-domain tasks.

## F.5 Further Ablation Studies

Effect of Difficulty-Adaptive Rollout While Section 4.2 indirectly demonstrates the benefits of our approach against non-adaptive baselines (e.g., TreeRL), we conduct this ablation to explicitly isolate the impact of the difficulty-adaptive mechanism by evaluating a static, non-adaptive variant of DATPO. Applying our default tree-expansion parameters of $( N , K _ { \mathrm { m a x } } , B _ { \mathrm { m a x } } ) = ( 4 , 3 , 4 )$ to a static rollout would significantly increase the overall training budget. Therefore, to maintain a token consumption comparable to our adaptive framework, we restrict the tree-expansion parameters of this non-adaptive variant to a narrower configuration of $( N , K , B ) = ( 6 , 2 , 2 )$

As reported in Table 12, the full DATPO framework outperforms the non-adaptive variant in both avg@k and pass@k. The lower pass@k of the non-adaptive model aligns with our empirical analysis in Section 2.1: uniformly expanding the rollout budget across all problems, regardless of their difficulty, can lead to a decrease in pass@k. Furthermore, the absence of difficulty adaptation also degrades avg@k. We attribute this to a compounding effect during training: the restricted reasoning coverage artificially limits the variety of valid reasoning paths discovered, which consequently undermines the effectiveness of the annealed siblingdiversity term that relies on a rich, diverse set of trajectories to properly optimize the policy.

<table><tr><td>Benchmark</td><td>DATPO w/o difficulty-adaptive</td><td>DATPO</td></tr><tr><td>MATH500</td><td>62.9 / 81.0</td><td>63.5 / 81.7</td></tr><tr><td>AIME26</td><td>2.6 / 21.1</td><td>2.4 / 33.3</td></tr><tr><td>AIME25</td><td>1.7 / 25.6</td><td>1.6 / 33.3</td></tr><tr><td>AIME24</td><td>5.4 /31.1</td><td>4.8 / 35.6</td></tr><tr><td>AMC23</td><td>37.0 / 90.0</td><td>39.8 / 90.8</td></tr><tr><td>Average</td><td>21.9 / 49.8</td><td>22.4 / 54.9</td></tr></table>

Table 12: Ablation study on difficulty-adaptive rollout. Results are reported as avg@8 / pass@8 for MATH500 and avg@64 / pass@64 for all other benchmarks.

![](images/f7df06914c25a6bc4cef9ac2bc00b745e9852d69fb6ce736e9a3988a481c5da7.jpg)  
Figure 11: pass@64 on AIME26 across training checkpoints under different diversity-coefficient schedules.

Training Dynamics of the Diversity-Augmented Advantage. Table 4 reports the effect of the diversity coefficient only at the final checkpoint. To examine how the diversity-augmented advantage affects reasoning coverage throughout training, we additionally evaluate pass@64 on AIME26 at 100- step intervals. We compare training without the diversity term (α = 0) against DATPO with the default linearly annealed schedule (α : 0.2 → 0). Each result is averaged over three independent evaluation runs and reported as the mean ± standard deviation.

As shown in Figure 11, without the diversity term, pass@64 peaks at 30.0% at step 200 and decreases to 25.6% by step 700. In contrast, DATPO exhibits an overall upward trend after the initial training stage and reaches 33.3% at step 700. These results suggest that the diversity-augmented advantage mitigates the early saturation of reasoning coverage and promotes its continued expansion during training.

<table><tr><td rowspan="2">Embedding</td><td rowspan="2">Size</td><td colspan="2">MATH500</td></tr><tr><td>avg@8</td><td>pass@8</td></tr><tr><td>all-mpnet-base-v2</td><td>110M</td><td>62.9</td><td>81.9</td></tr><tr><td>bge-m3</td><td>560M</td><td>61.6</td><td>81.7</td></tr><tr><td>gte-base-en-v1.5</td><td>137M</td><td>63.4</td><td>81.9</td></tr><tr><td>gte-large-en-v1.5</td><td>434M</td><td>63.5</td><td>81.7</td></tr></table>

Table 13: Ablation study on different embedding models.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Avg. Tokens</td><td colspan="2">MATH500</td></tr><tr><td>avg@8</td><td>pass@8</td></tr><tr><td>TreeRL (4,3,4)</td><td>17,766.0</td><td>62.5</td><td>81.3</td></tr><tr><td>AttnRL (4,3,4)</td><td>12,478.7</td><td>62.1</td><td>80.0</td></tr><tr><td>DATPO</td><td>8,952.4</td><td>63.5</td><td>81.7</td></tr></table>

Table 14: Performance of baselines under wider tree topology. Avg. Tokens denotes the average number of generated tokens per question during training. The values in parentheses specify the tree expansion hyperparameters (N, K, B).

Effect of Embedding Models As noted in Appendix F.2, DATPO defaults to using gte-large-en-v1.5 to compute the block-level sibling-diversity term $( \mathrm { D i v _ { s i b } ) }$ To assess the sensitivity of our framework to the choice of the embedding model, we conduct an additional ablation study comparing it against three alternative models of varying scales: all-mpnet-base-v2 (Reimers and Gurevych, 2019), bge-m3 (Chen et al., 2024), and gte-base-en-v1.5 (Zhang et al., 2024).

As reported in Table 13, while the 434Mparameter gte-large-en-v1.5 model achieves the highest avg@8 accuracy (63.5%), smaller architectures such as all-mpnet-base-v2 (110M) and gte-base-en-v1.5 (137M) also yield highly competitive performance, even marginally outperforming the default configuration in pass@8 (81.9%). These results demonstrate that DATPO is robust to the scale and specific choice of the underlying embedding model. Consequently, the effectiveness of our diversity-augmented advantage appears to stem primarily from capturing stable, relative semantic distance signals among sibling blocks, rather than relying strictly on the absolute capacity or size of the embedding model itself.

## F.6 Performance of Baselines under Wider Tree Topology

As detailed in Appendix F.2, the tree-based baseline methods (TreeRL and AttnRL) were evaluated using the default tree expansion hyperparameters reported in their original papers, specifically $( N , K , B ) ~ = ~ ( 6 , 2 , 2 )$ for TreeRL and $( N , K _ { \mathrm { m a x } } , B _ { \mathrm { m a x } } ) = ( 6 , 2 , 2 )$ for AttnRL. In contrast, our proposed DATPO utilized a wider branching configuration of $( N , K _ { \mathrm { m a x } } , B _ { \mathrm { m a x } } ) = ( 4 , 3 , 4 )$ to ensure a sufficient number of branch rollouts for computing the sibling-diversity term. Although we rigorously constrained the total number of generated tokens to be comparable across all methods in our main experiments (Appendix F.3), one might argue that DATPO’s performance gains simply stem from this wider tree topology (B = 4) rather than its algorithmic design.

To isolate the impact of our algorithmic design, we conduct an additional experiment where both TreeRL and AttnRL are trained using the identical wider configuration of (4, 3, 4) (or (4, 3, 4) maximum limits for AttnRL).

As reported in Table 14, forcing the baselines into this wider topology does not improve their overall performance. We observe three key findings: First, our proposed DATPO achieves the highest performance in both avg@8 and pass@8 while generating the fewest average tokens per problem during training. Second, compared to their original (6, 2, 2) configurations (as reported in Table 2), both TreeRL and AttnRL experience performance degradation in some metrics when scaled to the (4, 3, 4) topology. Taken together, these results confirm that the superiority of our framework is driven by the algorithmic improvements of the difficultyadaptive rollout and diversity-guided exploration, rather than merely relying on wider tree expansion hyperparameters.

## G Further Related Works

## G.1 Tree-based Search in Reinforcement Learning

Tree-based search, famously successful in reinforcement learning milestones like AlphaGo (Silver et al., 2016), is now being adapted to the generative landscape of LLMs to unlock more systematic exploration. TreeRL (Hou et al., 2025) pioneers this with an on-policy framework using entropyguided branching. To address computational bottlenecks, AttnRL (Liu et al., 2025a) improves efficiency by leveraging difficulty-aware adaptive sampling within a one-step off-policy pipeline. Recent research optimizes structural efficiency: TEMPO (Tran et al., 2025) leverages prefix-tree structure for branch-aware credit assignment, while TreePO (Li et al., 2025a) maximizes KV-cache reuse. To prevent paths from collapsing into homogeneous reasoning, LATR (Xing et al., 2025) explicitly maximizes trajectory-level diversity by dynamically branching and pruning semantically redundant paths via lookahead simulation. Furthermore, TreeAdv (Cao et al., 2026) refines credit assignment by combining an entropy-based branching method with a tree-structured advantage redistribution. While these methods primarily utilize tree structures for credit assignment or computational efficiency, our work leverages tree structures with the primary goal of expanding the model’s reasoning coverage.

## H Use of AI assistants

AI assistants were utilized for code implementation and for refining the linguistic presentation of the manuscript. All AI-generated outputs were carefully reviewed, modified, and validated by the authors.