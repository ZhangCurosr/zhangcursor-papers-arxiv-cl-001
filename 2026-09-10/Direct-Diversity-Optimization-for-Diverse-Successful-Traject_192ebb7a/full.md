# Direct Diversity Optimization for Diverse Successful Trajectories in Preference Post-Training

Junwon Ko Dong-Jae Lee Minchan Kwon Sunghyun Baek Junmo Kim

School of Electrical Engineering, KAIST

Daejeon, Republic of Korea

{kojunewon,jhtwosun,kmc0207,baeksh}@kaist.ac.kr junmo.kim@kaist.ac.kr

## Abstract

LLM agents for sequential decision tasks are often post-trained with trajectory-level outcome labels, but such labels provide little supervision for preserving multiple successful branches from the same decision state. We study this problem as successful strategy coverage: how broadly a model realizes distinct successful strategies under a fixed rollout budget. We present Direct Diversity Optimization (DDO), an offline post-training method that combines Divergence-Tree Collection (DTC) with the Reference-Relative Target-Odds Objective (RTO).<sup>1</sup> DTC constructs state-aligned branch sets rooted at shared decision states, and RTO trains the model to match referencerelative targets over successful alternatives. DDO achieves the strongest task success and successful strategy coverage among the compared post-training methods across BabyAI, BabaIsAI, and WebShop. It also achieves the highest recovery rate after local action replacement and higher task success and coverage than successful-only imitation and decoding-time diversification controls.

## 1 Introduction

Recent work on Large Language Models (LLMs) commonly adapts pretrained models to downstream tasks through post-training (Ouyang et al., 2022; Touvron et al., 2023; Grattafiori et al., 2024; Yang et al., 2025). The same approach is now applied to LLM agents in sequential decision tasks such as web navigation, embodied control, and tool use (Zeng et al., 2024; Song et al., 2024; Wei et al., 2025). In these tasks, dense per-step supervision is rarely available, and post-training typically relies on trajectory-level success-or-failure feedback. Under such coarse supervision, standard post-training tends to narrow the model’s output distribution (Kirk et al., 2024; Slocum et al., 2025; Lanchantin et al., 2025). In sequential decision tasks, this narrowing can cause the policy to collapse onto a single successful trajectory even when several viable alternatives remain.

![](images/f7395eca3f13491a3308dd58c5cce26988af161e7c787192a942d40365e66277.jpg)  
Figure 1: A motivating online-shopping example: Model A, a standard post-trained model, completes the purchase through a single trajectory, while Model B retains multiple successful trajectories that can be used as fallbacks when conditions change.

Figure 1 illustrates this collapse in an onlineshopping task where an LLM agent must buy a laptop charger under a price constraint. A standard post-trained model (Model A) commits to a single brand-page trajectory, whereas a model trained to retain multiple successful trajectories (Model B) can succeed through distinct search, filtering, and comparison strategies. Both models succeed under normal conditions. When the brand-page trajectory is blocked, Model A has no fallback, whereas Model B retains an alternative path to success.

The contrast between Models A and B highlights a limitation of standard preference post-training: it lacks state-aligned supervision over alternative branches. Trajectory-level success labels identify a successful trajectory, but do not show the outcomes of alternative actions at the same decision state. Consequently, DPO-style training (Rafailov et al., 2023) with success–failure trajectory preferences distinguishes successful from failed trajectories but provides no explicit target for allocating probability among multiple successful branches from the same state. The learned policy is therefore prone to placing most of its probability mass on one successful branch, leaving little for the remaining alternatives.

To address this limitation, we present Direct Diversity Optimization (DDO), an offline posttraining method that couples a data-construction procedure with a reference-relative optimization objective. Given a successful source trajectory, Divergence-Tree Collection (DTC) restores an intermediate decision state and constructs an outcome-labeled branch set containing the source and alternative branches. Reference-Relative Target-Odds (RTO) replaces the single-winner preference target with a reference-relative target distribution over successful branches and fits the model to match its induced pairwise odds. Together, DTC and RTO convert trajectory-level outcomes into preference supervision over multiple successful alternatives from the same decision state.

We evaluate DDO on BabyAI, BabaIsAI, and WebShop, which cover navigation, object interaction, multi-stage instruction following, rule manipulation, and web interaction. Across the three benchmarks, DDO achieves the strongest joint performance on task success and successful strategy coverage among the compared post-training methods. These gains also translate into more robust recovery after local action replacement. DDO also achieves higher task success and coverage than successful-only imitation and decoding-time diversification controls. Component analyses show complementary roles for DTC and RTO, whose combination yields the largest joint gains.

## 2 Related Work

## 2.1 Diversity- and tie-aware post-training.

Preference- or reward-based post-training is now a standard mechanism for adapting LLMs with feedback (Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022; Rafailov et al., 2023). However, it has also been shown to narrow the output distribution of trained models: Kirk et al. (2024) and Padmakumar and He (2024) report that RLHF and DPO reduce output diversity relative to the base model, and Slocum et al. (2025) show that KL-regularized preference learning can amplify dominant preferences. To mitigate this collapse, recent approaches preserve multiple acceptable responses in the preference signal. DivPO encourages diverse preferred responses through diversity-aware preference updates (Lanchantin et al., 2025), while tie-aware DPO variants allow similarly preferred responses to be modeled as ties rather than forcing an arbitrary winner– loser direction (Chen et al., 2025). These approaches reduce the pressure to collapse acceptable outputs into a single mode, but they operate at the prompt-response or pairwise-comparison level. Related goals have long been studied in reinforcement learning: maximum-entropy RL encourages stochastic high-return behavior, while skilldiscovery and quality-diversity methods learn distinct skills or policy repertoires (Haarnoja et al., 2018; Eysenbach et al., 2019; Pugh et al., 2016; Pierrot et al., 2022). DDO targets offline preference post-training for sequential LLM agents, using state-aligned outcome-labeled branches and reference-relative targets to retain multiple observed successful branches.

## 2.2 Post-training LLMs for sequential tasks.

Preference-based post-training has been extended to sequential decision tasks using branching rollouts or step-level feedback. Chain of Preference Optimization (Zhang et al., 2024) constructs preference data from alternative reasoning trajectories, while ETO (Song et al., 2024) contrasts successful and failed exploration trajectories at the trajectory level. Other methods provide per-step signal: IPR (Xiong et al., 2024) builds step-level contrastive pairs along expert trajectories, and Agent-PRM (Xi et al., 2026) trains a process reward model for step-wise progress. Across these methods, the supervision signal primarily distinguishes better from worse trajectories or steps. When multiple branches from the same state are successful, they are not explicitly modeled as a set of valid alternatives; they are typically paired against failures, folded into independent comparisons, or scored by a progress signal.

DDO uses the labeled branch set at a decision state as the supervision unit. It preserves multiple successful branches as valid positives while distinguishing them from failed branches, without forcing an arbitrary ranking among the successful ones. This couples diversity preservation directly with task success.

## 3 Method

We introduce Direct Diversity Optimization (DDO), an offline post-training method for retaining multiple successful strategies in LLM agents trained from trajectory-level outcome labels. DDO couples Divergence-Tree Collection (DTC; Section 3.2), which constructs state-aligned branch sets with rollout outcome labels, with the Reference-Relative Target-Odds Objective (RTO; Section 3.3), which fits the model to reference-relative targets over successful branches. This coupling transforms trajectory-level outcome supervision into samestate branch comparisons and a target distribution over successful alternatives. Figure 2 summarizes the pipeline.

## 3.1 Problem Setting and Notations

We address post-training of LLM agents on sequential decision tasks with textual observations and executable actions. At each step t, the agent observes a textual state $x _ { t }$ that includes the current observation and the recent interaction history, and produces an output $y _ { t } ;$ the environment executes the action parsed from y<sub>t</sub> to produce the next state. A trajectory $\tau = ( x _ { 0 } , y _ { 0 } , \dots , y _ { T - 1 } , x _ { T } )$ is judged only at termination, receiving a binary result $R ( \tau ) \in \{ 0 , 1 \}$ provided by the environment. This trajectory-level signal is the sole supervision; no per-step rewards or learned reward model are assumed. We write $\pi _ { \boldsymbol { \theta } } ( y \mid x )$ for the trainable model and $\pi _ { \mathrm { r e f } } ( y \mid x )$ for the frozen reference model.

## 3.2 Divergence-Tree Collection

DTC takes a successful trajectory and produces, at selected decision steps, branch sets rooted at a shared state with rollout outcome labels. For a source trajectory $\tau ,$ the candidate divergence steps are the interior decision points $1 , \ldots , | \tau | - 1$ . Under a fixed collection budget, DTC selects divergence steps to cover different depths along the source, so that branch roots are spread along the trajectory rather than concentrated near its initial prefix. At each selected divergence step, DTC restores the corresponding decision state as the shared state x (Algorithm 1). It keeps the source output as one branch and queries an expert model for alternative outputs whose parsed actions differ from the source action. Each retained output is executed to termination by the expert model and receives a binary success label. If the state cannot be restored or no distinct valid alternative is obtained, DTC skips that state.

Algorithm 1 Divergence-Tree Collection (DTC).   
Implementation details of SELECT, RESTORE,   
ALT, and REJECT are provided in Appendix B.2.   
Input: successful sources $\tau ^ { + }$ , expert model $E ,$ budgets   
$\bar { K _ { d } } , K _ { a }$   
Output: DTC records $\mathcal { D } _ { \mathrm { D T C } }$   
1: $\mathcal { D } _ { \mathrm { { D T C } } }  \emptyset$   
2: for $\tau \in \mathcal { T } ^ { + }$ do   
3: $D \gets \mathrm { S E L E C T } ( \{ 1 , \dots , | \tau | - 1 \} , K _ { d } )$   
4: for $d \in D$ do   
5: $x \gets \mathrm { R E S T O R E } ( \tau , d )$   
6: i $: x = \perp$ then   
7: continue   
8: $B  \{ j _ { \mathrm { s r c } } \} , y _ { j _ { \mathrm { s r c } } }  y _ { d } ^ { \tau } , r _ { j _ { \mathrm { s r c } } }  1$   
9: $\mathcal { V }  \mathrm { A L T } ( E , x , y _ { d } ^ { \tau } , K _ { a } )$   
10: for $y \in \mathcal { V }$ do   
11: if REJECT(y, B) then   
12: continue   
13: $r \gets R ( \mathrm { R O L L O U T } ( E , x , y ) )$   
14: add new j with $( y _ { j } , r _ { j } ) \gets ( y , r )$ to B   
15: $\mathbf { i f } \ B \neq \{ j _ { \mathrm { s r c } } \}$ then   
16: add RECORD(x, B) to D<sub>DTC</sub>   
17: return D<sub>DTC</sub>

Each collected branch set is represented as one DTC record:

$$
C = ( x , B , \{ ( y _ { j } , r _ { j } ) \} _ { j \in B } ) .\tag{1}
$$

Here, x is the shared decision state, $B$ is a finite set of branch indices, $y _ { j }$ is the LLM output of branch $j ,$ and $r _ { j } \in \{ 0 , 1 \}$ is the binary result obtained after executing that branch. For each record we partition $B$ into successful branches $S ^ { + } = \{ j : r _ { j } = 1 \}$ and non-successful branches $S ^ { 0 } = \{ j : r _ { j } = 0 \}$ RTO uses $S ^ { + }$ for target-odds estimation and $S ^ { 0 }$ for boundary comparisons between successful and non-successful observed branches.

## 3.3 Reference-Relative Target-Odds Objective

Pairwise preference objectives that contrast a single winner against a single loser with hard binary labels risk collapsing $\pi _ { \theta }$ onto a single successful strategy. The Reference-Relative Target-Odds (RTO) objective addresses this by aligning the log-odds $\pi _ { \theta }$ assigns to pairs of successful branches with the log-odds prescribed by a fixed distribution q over $S ^ { + }$ . For each record $C = ( x , B , \{ ( y _ { j } , r _ { j } ) \} _ { j \in B } )$

![](images/990180156b6d24f21160c225aec57cac0bc744b8bc230bda38c38e2bccf7c800.jpg)  
Figure 2: DDO pipeline. DTC constructs state-aligned branch sets, and RTO defines a reference-relative target distribution whose induced pairwise targets are optimized with a soft logistic objective.

RTO constructs $q$ from $\pi _ { \mathrm { r e f } }$ and matches the two log-odds in reference-relative form, aligning $\pi \varrho ^ { \cdot } \mathrm { s }$ shift over $\pi _ { \mathrm { r e f } }$ with $q ^ { * } { \bf s }$ shift over $q _ { \mathrm { r e f } }$ . Two pair types are handled asymmetrically: success–success pairs use q to set the desired log-odds within $S ^ { + }$ and success–failure pairs across $S ^ { + }$ and $S ^ { 0 }$ use task success labels to order successful branches above failed ones.

Branch distribution. On $S ^ { + }$ , define the renormalized reference distribution $q _ { \mathrm { r e f } }$ as

$$
q _ { \mathrm { r e f } } ( j ) = \frac { \pi _ { \mathrm { r e f } } ( y _ { j } \mid x ) } { \sum _ { h \in S ^ { + } } \pi _ { \mathrm { r e f } } ( y _ { h } \mid x ) } .\tag{2}
$$

The distribution $q$ is defined as

$$
q ( j ) = \frac { q _ { \mathrm { r e f } } ( j ) ^ { \alpha } } { \sum _ { h \in S ^ { + } } q _ { \mathrm { r e f } } ( h ) ^ { \alpha } } ,\tag{3}
$$

where $\alpha ~ \in ~ [ 0 , 1 ]$ controls how strongly q preserves $\pi _ { \mathrm { r e f } } \gamma _ { \mathrm { s } }$ relative weighting among successful branches: smaller $\alpha$ flattens these differences, larger α keeps them. $\operatorname { A t } \alpha = 0 , q$ is uniform on $S ^ { + }$ providing maximal within-success diversity pressure; at $\alpha = 1 , q = q _ { \mathrm { r e f } }$ and the success–success targets reduce to preserving $\pi _ { \mathrm { r e f } } \gamma _ { \mathrm { s } }$ relative odds among successful branches. Both $q$ and $q _ { \mathrm { r e f } }$ are computed once from $\pi _ { \mathrm { r e f } }$ and held fixed during training. We use $\alpha = 0 . 5$ unless otherwise stated.

Distribution margin. For an ordered pair $u , v \in$ $S ^ { + }$ of successful branches, the target–reference log-odds margin is given by

$$
m ^ { \star } = \log \frac { q ( u ) } { q ( v ) } - \log \frac { q _ { \mathrm { r e f } } ( u ) } { q _ { \mathrm { r e f } } ( v ) } .\tag{4}
$$

The margin $m ^ { \star }$ is therefore the reference-relative log-odds shift needed for $\pi _ { \theta }$ to match q at $( u , v )$

The corresponding target is $p ^ { \star } = \sigma ( \beta m ^ { \star } )$ with logistic scale $\beta > 0 ;$ for a success–failure pair with $u \in S ^ { + }$ and $v \in S ^ { 0 }$ , q is undefined on v and we set $p ^ { \star } = 1$ directly, so task success labels enter the objective only through these pairs.

Model margin. For $\pi _ { \theta }$ , the analogous margin is defined in the same form so that it can be directly compared to $m ^ { \star }$ :

$$
m _ { \theta } = \log \frac { \pi _ { \theta } ( y _ { u } \mid x ) } { \pi _ { \theta } ( y _ { v } \mid x ) } - \log \frac { \pi _ { \mathrm { r e f } } ( y _ { u } \mid x ) } { \pi _ { \mathrm { r e f } } ( y _ { v } \mid x ) } .\tag{5}
$$

Aligning m<sub>θ</sub> with $m ^ { \star }$ is equivalent to matching $\pi \varrho ^ { \cdot } \mathrm { s }$ log-odds at $( u , v )$ to $q ^ { * } { \bf s }$ , which is what the loss enforces.

Objective function. We train $\pi _ { \theta }$ by minimizing

$$
\mathcal { L } _ { \mathrm { D D O } } ( \theta ) = \mathbb { E } \big [ D _ { \mathrm { K L } } \big ( P ^ { \star } \| P _ { \theta } \big ) \big ] ,\tag{6}
$$

where $P ^ { \star } = \mathrm { B e r n } ( p ^ { \star } )$ and $P _ { \theta } = \mathrm { B e r n } ( p _ { \theta } )$ are Bernoulli distributions with parameters $p ^ { \star }$ and $p _ { \theta } = \sigma ( \beta m _ { \theta } )$ . The expectation is estimated over materialized pairs: each unordered success–success pair is included once, success–failure pairs are oriented toward the successful branch, and statenormalized weights prevent larger branch sets from dominating. Up to a θ-independent constant, this is equivalent to soft binary cross-entropy with target $p ^ { \star }$ and prediction $p _ { \theta }$ . The success–success terms match the target odds $q ( u ) / q ( v )$ between successful branches, while success–failure terms set $p ^ { \star } ~ = ~ 1$ and reduce to the DPO-style loss $- \log \sigma ( \beta m _ { \theta } )$ , with $u \in S ^ { + }$ and $v \in S ^ { 0 }$

## 4 Experimental Setup

## 4.1 Benchmarks

We evaluate DDO on three benchmarks that span complementary forms of sequential text-agent behavior. BabyAI (Chevalier-Boisvert et al., 2019) probes controlled navigation, object interaction, and multi-stage instruction following. BabaIsAI (Cloos et al., 2024) targets rule manipulation, where the agent must change or exploit environment dynamics. For BabyAI and BabaIsAI, we use the BALROG text-action implementations (Paglieri et al., 2025), which provide standardized textual observations and executable action interfaces. Web-Shop (Yao et al., 2022) examines web-style shopping behavior involving search, product inspection, option selection, and purchase decisions. Details of each benchmark are provided in Appendix C.1.

## 4.2 Baselines and Comparison Protocol

We compare DDO against DPO and diversityaware post-training baselines. DPO is the standard Direct Preference Optimization baseline (Rafailov et al., 2023). DivFreq and DivProb adapt DivPO to state-aligned branch data by treating each shared state as a prompt, its executable branch outputs as candidate responses, and terminal rollout outcomes as quality labels (Lanchantin et al., 2025). They retain DivPO’s frequency- and probabilitybased pair-selection criteria, respectively. TieDPO-RK and TieDPO-Dav are tie-aware DPO baselines based on the Rao-Kupper and Davidson variants of Chen et al. (2025).

We use Qwen3-1.7B as the target model and Qwen3.5-122B-A10B-FP8 as the expert model for DTC collection (Yang et al., 2025; Qwen Team, 2026). Base denotes the unadapted target model before task SFT or preference optimization. Reference denotes the frozen task-SFT model, fine-tuned from the base on the successful trajectories that also serve as DTC sources. It is used both as the initialization for all post-training methods and as the reference model for reference-relative objectives.

For fair comparison, all post-training methods within a benchmark use the same DTC-collected branch sets, so the comparison isolates objectiveside differences while holding the data side fixed. The DTC ablation in Section 5.1 separately varies the data side by replacing this resource with comparisons formed without DTC, allowing us to test the contribution of the collection procedure itself. Training and evaluation otherwise follow the same task budgets and decoding protocols within each benchmark. Full experimental details are provided in Appendix C.

## 4.3 Metrics

We evaluate each model along two axes: task success and successful strategy coverage under a fixed rollout budget. A rollout is valid if it terminates without system failures or missing model outputs. Success rate is the fraction of valid rollouts that solve the task. BabyAI and BabaIsAI count a rollout as successful when the benchmark progression reaches 1.0. WebShop returns a graded purchase score rather than a binary success label; we treat scores of at least 0.9 as successful, using the same fixed threshold across all methods.

To measure successful strategy coverage, we group successful rollouts into trajectory classes using a per-benchmark equivalence relation that abstracts surface variation in the action sequence (see Appendix C.1 for the per-benchmark rules). Two successful rollouts belong to the same class when they are equivalent under this relation. Within each benchmark, we apply the same equivalence relation to the rollouts from all evaluated methods. We use effective strategy diversity (ESD) and entropy effective strategy diversity (H-ESD) to summarize the resulting classes. Both metrics are success-restricted and budget-normalized: they are computed from successful trajectory classes but normalized by the total number of valid rollouts.

For each evaluation item i, let $K _ { i }$ be the number of valid rollouts and $U _ { i }$ the number of unique successful trajectory classes. Let $H _ { i }$ be the entropy of the empirical distribution over these classes:

$$
\mathrm { E S D } ( i ) = \frac { U _ { i } } { K _ { i } } , \qquad \mathrm { H \mathrm { - E S D } } ( i ) = \frac { 2 ^ { H _ { i } } } { K _ { i } } .
$$

ESD counts how many distinct successful trajectory classes are observed under the rollout budget, while H-ESD is smaller when successful rollouts concentrate on a few repeated trajectory classes. Normalizing by $K _ { i }$ , rather than by the number of successful rollouts, makes the metric reflect finitebudget successful strategy coverage: a model receives a high score only when it both solves the task and produces distinct successful trajectory classes. Reported scores are uniform means of the itemlevel ESD and H-ESD over the evaluation items in each task. Both metrics are zero when no successful rollout is observed.

![](images/a540571ad939d5e47cc841dd8ece82bf701b945cd622a3c89f2c3374057e7d1d.jpg)  
Figure 3: Effect of DTC on BabyAI and BabaIsAI. Each curve reports the task-averaged number of unique successful trajectories at a matched per-task request budget, under benchmark-specific trajectory normalization.

## 5 Results

We first evaluate DTC as a data-construction procedure by measuring collection-time successful strategy coverage and downstream performance with and without DTC. We then fix DTC and compare post-training objectives across the three benchmarks, isolating RTO under the same state-aligned branch supervision.

## 5.1 DTC Improves Branch Supervision

To isolate the data side of DDO, we run DPO and DDO with and without DTC. DTC converts trajectory-level outcome labels into same-state branch supervision. For the collection-efficiency comparison, Without DTC independently samples full trajectories with the same expert under the matched request budget. For downstream training, settings without DTC form same-task comparisons from separately sampled rollouts matched to the training comparison exposure.

Figure 3 shows that DTC discovers more unique successful trajectories under the same request budget on both BabyAI and BabaIsAI, thereby increasing collection-time successful strategy coverage before post-training begins.

DTC pairs are distributed across the trajectory, with 56.4% formed at or beyond the midpoint. Pairs without DTC are concentrated near the initial prefix: 79.3% occur within the first 20% of the source trajectory. This structural difference matters for sequential strategies, because early, middle, and late decisions can play different roles in reaching success. DTC therefore provides branch-level supervision across a broader range of decision depths.

![](images/733106b35c8c542041c8e83dfe2fa7ea5c08d39d9a2a5808a862d7b1054b165a.jpg)  
Figure 4: Distribution of normalized pair-formation positions along source trajectories with and without DTC. Positions are normalized by trajectory length.

These collection differences translate into posttraining results. Table 1 compares DPO and DDO with and without DTC. For DPO, adding DTC raises task success by 8 percentage points (pp) on BabyAI and 16 pp on BabaIsAI, although its effect on coverage is mixed on BabyAI. For DDO, adding DTC raises task success by 5 pp on BabyAI and 16 pp on BabaIsAI. Adding DTC also improves DDO’s coverage metrics: on BabyAI, H-ESD and ESD each increase by 0.06; on BabaIsAI, they increase by 0.10 and 0.12, respectively. Averaged across the two benchmarks, adding DTC to DDO improves task success by 10.5 pp, H-ESD by 0.08, and ESD by 0.09.

DTC broadens the collected set of successful trajectories and improves downstream post-training. Under matched request budgets, DTC discovers more unique successful trajectories and provides branch supervision at decision states spanning early, middle, and late portions of the source trajectories. The factorization in Table 1 shows complementary contributions from DTC and RTO: DTC expands branch supervision and predominantly raises success, whereas RTO improves both success and coverage under either collection condition. Combining DTC and RTO yields the strongest joint success and coverage result on both BabyAI and BabaIsAI. We next fix the DTC resource and compare post-training objectives under the same statealigned branch supervision.

<table><tr><td>Method DTC</td><td></td><td colspan="3">BabyAI</td><td colspan="3">BabaIsAI</td></tr><tr><td></td><td></td><td>Succ. ↑</td><td>H-ESD↑</td><td>ESD↑</td><td>Succ. ↑</td><td>H-ESD↑</td><td>ESD↑</td></tr><tr><td>DPO</td><td>×</td><td>0.76</td><td>0.38</td><td>0.44</td><td>0.78</td><td>0.26</td><td>0.30</td></tr><tr><td>DPO</td><td>√</td><td>0.84</td><td>0.36</td><td>0.43</td><td>0.94</td><td>0.31</td><td>0.38</td></tr><tr><td>DDO</td><td>×</td><td>0.87</td><td>0.39</td><td>0.46</td><td>0.83</td><td>0.28</td><td>0.35</td></tr><tr><td>DDO</td><td>√</td><td>0.92</td><td>0.45</td><td>0.52</td><td>0.99</td><td>0.38</td><td>0.47</td></tr></table>

Table 1: Effect of DTC on DPO and DDO.

<table><tr><td>Method</td><td colspan="4">Success rate ↑</td><td colspan="4">H-ESD ↑</td><td colspan="4">ESD↑</td></tr><tr><td></td><td></td><td>BabyAI BabaIsAI WebShop Avg.</td><td></td><td></td><td>BabyAI BabaIsAI WebShop</td><td></td><td></td><td>Avg.</td><td></td><td>BabyAI BabaIsAI WebShop Avg.</td><td></td><td></td></tr><tr><td>Base</td><td>0.32</td><td>0.21</td><td>0.18</td><td>0.24</td><td>0.28</td><td>0.08</td><td>0.02</td><td>0.13</td><td>0.29</td><td>0.09</td><td>0.02</td><td>0.13</td></tr><tr><td>Reference</td><td>0.82</td><td>0.90</td><td>0.26</td><td>0.66</td><td>0.41</td><td>0.37</td><td>0.15</td><td>0.31</td><td>0.49</td><td>0.43</td><td>0.17</td><td>0.36</td></tr><tr><td>DPO</td><td>0.84</td><td>0.94</td><td>0.26</td><td>0.68</td><td>0.36</td><td>0.31</td><td>0.11</td><td>0.26</td><td>0.43</td><td>0.38</td><td>0.12</td><td>0.31</td></tr><tr><td>DivFreq</td><td>0.85</td><td>0.91</td><td>0.26</td><td>0.67</td><td>0.37</td><td>0.30</td><td>0.12</td><td>0.26</td><td>0.44</td><td>0.39</td><td>0.13</td><td>0.32</td></tr><tr><td>DivProb</td><td>0.89</td><td>0.93</td><td>0.26</td><td>0.69</td><td>0.38</td><td>0.36</td><td>0.11</td><td>0.28</td><td>0.44</td><td>0.43</td><td>0.12</td><td>0.33</td></tr><tr><td>TieDPO-RK</td><td>0.88</td><td>0.80</td><td>0.28</td><td>0.65</td><td>0.35</td><td>0.28</td><td>0.16</td><td>0.26</td><td>0.40</td><td>0.35</td><td>0.17</td><td>0.31</td></tr><tr><td>TieDPO-Dav</td><td>0.86</td><td>0.77</td><td>0.16</td><td>0.60</td><td>0.38</td><td>0.30</td><td>0.13</td><td>0.27</td><td>0.43</td><td>0.38</td><td>0.13</td><td>0.31</td></tr><tr><td>DDO</td><td>0.92</td><td>0.99</td><td>0.36</td><td>0.76</td><td>0.45</td><td>0.38</td><td>0.21</td><td>0.35</td><td>0.52</td><td>0.47</td><td>0.22</td><td>0.40</td></tr></table>

Table 2: Main results across BabyAI, BabaIsAI, and WebShop. BabyAI and BabaIsAI report task averages, WebShop reports the shopping evaluation summary, and Avg. is the unweighted mean across benchmarks.

## 5.2 RTO Improves the Success–Coverage Frontier

With DTC fixed, we compare post-training objectives on the same state-aligned branch sets. Table 2 reports benchmark averages; full tables appear in Appendix A.6. The three benchmarks probe different aspects of successful strategy coverage— diverse successful action trajectories for the same instruction (BabyAI), diverse exploitable rule configurations (BabaIsAI), and diverse shopping behaviors ending in valid purchases (WebShop).

Joint success and coverage gains. DDO achieves the highest benchmark-average success (0.76), H-ESD (0.35), and ESD (0.40). Relative to the DPO row, these values correspond to gains of 8 pp, 0.09, and 0.09, respectively. The largest relative gain appears on WebShop, where H-ESD nearly doubles from 0.11 to 0.21 and ESD nearly doubles from 0.12 to 0.22, alongside a 10 pp success gain from 0.26 to 0.36. Because WebShop trajectory classes incorporate purchase realization, the gain reflects broader successful strategy coverage under the composite class definition.

Coverage relative to task-SFT initialization. DDO raises task success by 10 pp and both coverage metrics by 0.04 over the shared Reference initialization. The remaining post-training methods stay below Reference on both coverage metrics; DPO, for example, raises average success from 0.66 to 0.68 while H-ESD and ESD each fall by 0.05.

Other post-training variants. Across the compared methods, DDO leads every benchmark-level aggregate and all three overall averages. DivFreq and DivProb, which filter pairs by frequency- or probability-based diversity, change success and coverage by at most ±0.02 on average. TieDPO-RK and TieDPO-Dav reach lower average success than DPO (−3 and −8 pp), with the largest drops on BabaIsAI (−0.14 and −0.17 success) and, for TieDPO-Dav, WebShop (−0.10); their coverage matches DPO on average.

## 6 Analysis

## 6.1 Strategic Recovery

We use strategic recovery to test whether broader successful strategy coverage provides alternative routes after a local disruption. For each successful source trajectory, we first sample one interior decision point. We then replace the source action at that point with a different valid task action and roll out the same model from the edited prefix. Because every source trajectory solves the task before the edit, the score measures recovery from the local action replacement rather than ordinary task success. Appendix C details the probe construction.

DDO achieves the highest recovery rate, 75.2%, compared with 70.1% for DivFreq and 69.7% for DPO. After the initial trajectory is disrupted, DDO more often finds another viable trajectory to success, showing that its broader successful strategy coverage is accompanied by higher recovery.

## 6.2 Imitation Control

To separate successful-branch exposure from samestate outcome comparisons and reference-relative targets, we train successful-only SFT models on the same successful DTC branches used by DDO. We compare DDO with two imitation settings matched by optimizer steps and nominal epochs, respectively. At matched optimizer steps, successful-only imitation reaches H-ESD 0.39 and ESD 0.44, compared with DDO’s 0.42 and 0.50, while DDO retains a 32 pp advantage in task success. At 7.30× exposure, imitation reaches H-ESD 0.41 and ESD 0.47, while DDO retains a 24 pp advantage in task success. Successful-only SFT broadens coverage, while DDO retains a 24–32 pp advantage in task success under both matching conditions.

![](images/834548db35f856358431f86bb62b67857a499bdb1137d915c60d4901196dca1e.jpg)

Figure 5: Strategic recovery on BabyAI after local action replacement in successful source trajectories.
<table><tr><td>Method</td><td>Exposure</td><td>Success ↑</td><td>H-ESD ↑</td><td>ESD ↑</td></tr><tr><td>DDO, 5 ep.</td><td>1.00×</td><td>0.96</td><td>0.42</td><td>0.50</td></tr><tr><td>Imit. matched</td><td>1.00×</td><td>0.64</td><td>0.39</td><td>0.44</td></tr><tr><td>Imit. 5 ep.</td><td>7.30×</td><td>0.72</td><td>0.41</td><td>0.47</td></tr></table>

Table 3: Successful-only imitation control averaged over BabyAI and BabaIsAI.

## 6.3 Decoding Control

We also test whether inference-time diversification can recover successful strategy coverage without post-training changes. We compare DDO against DPO under a range of sampling temperatures.

Simply increasing DPO’s sampling temperature raises strategy coverage, but the gain comes with lower task success. At T=1.5, DPO reaches an ESD of 0.39, close to DDO’s 0.40, while DDO retains a 9 pp advantage in task success. Increasing temperature therefore recovers coverage by sacrificing success, whereas DDO improves the joint success and coverage result through post-training.

## 7 Conclusion

We present Direct Diversity Optimization (DDO), an offline post-training method for preserving multiple successful strategies in LLM agents trained from trajectory-level outcome labels. DDO consists of two components: Divergence-Tree Collection (DTC) and the Reference-Relative Target-Odds Objective (RTO). DTC builds state-aligned branch sets with per-branch outcome labels, and RTO trains the model toward reference-relative targets over successful branches.

<table><tr><td>Method</td><td>T</td><td>Success ↑</td><td>H-ESD ↑</td><td>ESD↑</td></tr><tr><td>DDO</td><td>0.6</td><td>0.76</td><td>0.35</td><td>0.40</td></tr><tr><td>DPO + temp sweep</td><td>0.0</td><td>0.60</td><td>0.22</td><td>0.25</td></tr><tr><td>DPO + temp sweep</td><td>0.1</td><td>0.62</td><td>0.20</td><td>0.24</td></tr><tr><td>DPO + temp sweep</td><td>0.6</td><td>0.68</td><td>0.26</td><td>0.31</td></tr><tr><td>DPO + temp sweep</td><td>1.0</td><td>0.66</td><td>0.29</td><td>0.35</td></tr><tr><td>DPO + temp sweep</td><td>1.5</td><td>0.67</td><td>0.33</td><td>0.39</td></tr></table>

Table 4: DPO temperature sweep averaged across BabyAI, BabaIsAI, and WebShop; DDO at T = 0.6 is shown for comparison.

Across all benchmarks, DDO achieves the strongest joint performance in task success and successful strategy coverage among the compared posttraining methods. DDO also achieves the highest recovery rate in the local action replacement evaluation. Compared with successful-only imitation and decoding-time diversification, DDO achieves broader coverage together with higher task success by training on same-state outcome comparisons with reference-relative targets. These results establish DDO as a training-time method that improves task success while broadening successful strategy coverage in sequential decision tasks.

## Limitations

For controlled benchmarking and method comparison, our experiments are confined to environments that support state reconstruction and the execution of alternative branches from a shared decision state under a common success predicate. This setting makes the outcomes of same-state alternative branches directly observable, but it excludes continuous control, partially observable or stochastic dynamics, multimodal observations, and real-world agents whose actions induce non-reversible external side effects. Extending DTC to settings without exact state reconstruction would require approximate state-aligned branch sets. Such sets could be constructed from rollouts generated by learned simulators or world models, or from logged trajectories that contain comparable decision contexts and alternative outcomes.

## Ethical Considerations

License of Existing Assets. The models used in this work, Qwen3-1.7B and Qwen3.5-122B-A10B-FP8, are released under the Apache 2.0 license. The benchmark environments are used under their respective licenses: BabyAI under the BSD 3-Clause License, BabaIsAI under the MIT License, and WebShop under the MIT License with copyright attributed to Princeton Natural Language Processing. All assets and environments were used for academic, non-commercial evaluation purposes and in accordance with the applicable license terms.

## Acknowledgments

This work was supported by the Artificial Intelligence Industrial Convergence Cluster Development Project, funded by the Ministry of Science and ICT (MSIT), Korea, and Gwangju Metropolitan City.

## References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, and 12 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. Preprint, arXiv:2204.05862.

Jinghong Chen, Guangyu Yang, Weizhe Lin, Jingbiao Mei, Chenxu Lyu, and Bill Byrne. 2025. On extending direct preference optimization to accommodate ties. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 53932– 53977. Curran Associates, Inc.

Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. 2019. BabyAI: A platform to study the sample efficiency of grounded language learning. In International Conference on Learning Representations.

Nathan Cloos, Meagan Jens, Michelangelo Naim, Yen-Ling Kuo, Ignacio Cases, Andrei Barbu, and Christopher J. Cueva. 2024. Baba is AI: Break the rules to beat the benchmark. In ICML 2024 Workshop on LLMs and Cognition.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. 2019. Diversity is all you need: Learning skills without a reward function. In International Conference on Learning Representations.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh

Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 540 others. 2024. The Llama 3 herd of models. Preprint, arXiv:2407.21783.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. 2018. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1861–1870. PMLR.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2024. Understanding the effects of RLHF on LLM generalisation and diversity. In International Conference on Learning Representations, volume 2024, pages 20620–20653.

Jack Lanchantin, Angelica Chen, Shehzaad Dhuliawala, Ping Yu, Jason Weston, Sainbayar Sukhbaatar, and Ilia Kulikov. 2025. Diverse preference optimization. Preprint, arXiv:2501.18101.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Vishakh Padmakumar and He He. 2024. Does writing with language models reduce content diversity? In International Conference on Learning Representations, volume 2024, pages 642–669.

Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wołczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kucinski, Lerrel Pinto, Rob´ Fergus, Jakob Foerster, Jack Parker-Holder, and Tim Rocktäschel. 2025. BALROG: Benchmarking agentic LLM and VLM reasoning on games. In International Conference on Learning Representations, volume 2025, pages 96666–96702.

Thomas Pierrot, Valentin Macé, Felix Chalumeau, Arthur Flajolet, Geoffrey Cideron, Karim Beguir, Antoine Cully, Olivier Sigaud, and Nicolas Perrin-Gilbert. 2022. Diversity policy gradient for sample efficient quality-diversity optimization. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 1075–1083. ACM.

Justin K. Pugh, Lisa B. Soros, and Kenneth O. Stanley. 2016. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741. Curran Associates, Inc.

Stewart Slocum, Asher Parker-Sartori, and Dylan Hadfield-Menell. 2025. Diverse preference learning for capabilities and alignment. In International Conference on Learning Representations, volume 2025, pages 24760–24790.

Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. 2024. Trial and error: Exploration-based trajectory optimization of LLM agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7584–7600, Bangkok, Thailand. Association for Computational Linguistics.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. 2020. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008–3021. Curran Associates, Inc.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 others. 2023. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.

Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, and Lihong Li. 2025. WebAgent-r1: Training web agents via end-to-end multi-turn reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7909–7928, Suzhou, China. Association for Computational Linguistics.

Zhiheng Xi, Chenyang Liao, Guanyu Li, Zhihao Zhang, Wenxiang Chen, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, and Xuanjing Huang. 2026. AgentPRM: Process reward models for LLM agents via step-wise promise and progress. In Proceedings ofthe ACM Web Conference 2026, pages 4184–4195. ACM.

Weimin Xiong, Yifan Song, Xiutian Zhao, Wenhao Wu, Xun Wang, Ke Wang, Cheng Li, Wei Peng, and Sujian Li. 2024. Watch every step! LLM agent

learning via iterative step-level process refinement. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1556–1572, Miami, Florida, USA. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. WebShop: Towards scalable realworld web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35, pages 20744–20757. Curran Associates, Inc.

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2024. AgentTuning: Enabling generalized agent abilities for LLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3053–3077, Bangkok, Thailand. Association for Computational Linguistics.

Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. 2024. Chain of preference optimization: Improving chain-of-thought reasoning in LLMs. In Advances in Neural Information Processing Systems, volume 37, pages 333–356. Curran Associates, Inc.

## A Additional Results

## A.1 Model-Scale Sensitivity

We vary the expert and target models separately on BabyAI while keeping all other collection, training, and evaluation settings fixed (Table 5).

<table><tr><td>Scale</td><td>Policy</td><td>Success ↑</td><td>H-ESD ↑</td><td>ESD ↑</td></tr><tr><td colspan="3">Expert model scale</td><td></td><td></td></tr><tr><td>122B</td><td>DDO</td><td>0.92</td><td>0.45</td><td>0.52</td></tr><tr><td>35B</td><td>DDO</td><td>0.87</td><td>0.41</td><td>0.48</td></tr><tr><td colspan="3">Target model scale</td><td></td><td></td></tr><tr><td>1.7B</td><td>Reference</td><td>0.82</td><td>0.41</td><td>0.49</td></tr><tr><td rowspan="2">4B</td><td>DDO</td><td>0.92</td><td>0.45</td><td>0.52</td></tr><tr><td>Reference</td><td>0.85</td><td>0.37</td><td>0.42</td></tr><tr><td></td><td>DDO</td><td>0.91</td><td>0.40</td><td>0.45</td></tr></table>

Table 5: Model-scale sensitivity on BabyAI. Each block varies one model while holding the other fixed.

Reducing the expert model size from 122B to 35B changes task success from 0.92 to 0.87, H-ESD from 0.45 to 0.41, and ESD from 0.52 to 0.48. Even with the 35B expert model, DDO retains more than 90% of the corresponding 122B value for each metric: 94.6% for task success, 91.1% for H-ESD, and 92.3% for ESD. The number of completed alternative branches also remains similar (3,329 versus 3,406).

DDO improves over its corresponding Reference at both target model sizes. For the 1.7B target model, DDO improves task success, H-ESD, and ESD by 0.10, 0.04, and 0.03, respectively. For the 4B target model, the corresponding gains are 0.06, 0.03, and 0.03.

## A.2 Coverage Growth

Beyond final-budget scores, Figure 6 traces the cumulative number of unique successful trajectories found within the first N rollouts on BabyAI, BabaIsAI, and WebShop. DDO stays above the other plotted methods across the sampled budget range, so its coverage gain is visible throughout sampling and at the final evaluation budget.

## A.3 Shared Cross-Task Post-Training

The main benchmark uses SFT and preference adapters for each task to provide controlled, matched comparisons. As an additional setting, we test whether DDO remains effective when both the initializer and post-training model are shared across tasks. On BabyAI, a single BabyAI-4 SFT adapter initializes all four tasks, and preference

![](images/dabf8b08299342a4997d53085f93ba3c25185221e034237f042d0b2318d93820.jpg)  
Figure 6: Coverage growth over BabyAI, BabaIsAI, and WebShop. Each curve indicates the cumulative number of unique successful trajectories found within the first N rollouts.

methods are trained on pooled BabyAI-4 preference data. The table reports BabyAI task averages under the same evaluation budgets and decoding settings as the main results.
<table><tr><td>Method</td><td>Success ↑</td><td>H-ESD↑</td><td>ESD↑</td></tr><tr><td>SFT</td><td>0.78</td><td>0.37</td><td>0.41</td></tr><tr><td>DPO</td><td>0.81</td><td>0.38</td><td>0.43</td></tr><tr><td>DDO</td><td>0.92</td><td>0.48</td><td>0.52</td></tr></table>

Table 6: Shared cross-task SFT and post-training on BabyAI using one BabyAI-4 SFT initialization and pooled preference data.

Table 6 shows that DDO consistently improves both task success and successful strategy coverage in the shared-model setting. DDO achieves 0.92 success, 0.48 H-ESD, and 0.52 ESD, extending its joint success and coverage gains from task-specific adapters to pooled cross-task post-training.

## A.4 Branch-Set Target Distribution

In Eq. (3), we interpret the RTO target as a logodds interpolation between two endpoints: a uniform distribution on $S ^ { + } \left( \alpha = 0 \right.$ , encoding withinsuccess coverage) and the reference distribution restricted to $S ^ { + } \left( \alpha = 1 \right.$ , preserving the relative ordering inherited from $\pi _ { \mathrm { r e f } } )$ . The intermediate setting $\alpha = 0 . 5$ retains both signals. Table 7 compares these three regimes on BabyAI.

The intermediate target $( \alpha ~ = ~ 0 . 5 )$ achieves the highest success (0.92), H-ESD (0.45), and ESD (0.52). At the endpoints, $\alpha = 1$ preserves the reference-relative weighting over successful branches, while $\alpha = 0$ uses a uniform target. The intermediate setting combines both target components and yields the strongest joint success and coverage result.

<table><tr><td>α</td><td>Success ↑</td><td>H-ESD↑</td><td>ESD↑</td></tr><tr><td>1.0</td><td>0.90</td><td>0.39</td><td>0.42</td></tr><tr><td>0.5</td><td>0.92</td><td>0.45</td><td>0.52</td></tr><tr><td>0.0</td><td>0.87</td><td>0.42</td><td>0.48</td></tr></table>

Table 7: Effect of the target-distribution coefficient α on BabyAI.

## A.5 Margin Sharpness Sensitivity

The inverse-temperature parameter $\beta$ controls the sharpness of the pairwise margin loss, determining how strongly deviations from the target pair odds are penalized.

![](images/2ce1d11789f1bf4338b2489c63a3966daa6fc5ca7f9936b83616c0d40f5774c6.jpg)  
Figure 7: Effect of the margin sharpness $\beta$ on DDO, averaged over BabyAI and BabaIsAI. Axes report success rate and H-ESD.

Figure 7 reports the DDO $\beta$ sweep over {0.1, 0.5, 1.0}. Across this range, success changes only mildly, while H-ESD remains nearly unchanged after rounding. DDO therefore varies little with margin sharpness across the tested values.

## A.6 Full Main Benchmark Tables

We provide the full benchmark results in Tables 8 to 10. Bold marks the strongest preference-trained result in each column, including ties.

BabyAI. DDO leads on average success rate, H-ESD, and ESD. The task columns show the same pattern: success and H-ESD increase for Goto, Pick, Open, and Comp, and ESD increases on three tasks and reaches 0.57 on Comp. The Open and Comp columns make this pattern visible in tasks involving object interaction and multi-step composition; DDO improves both success and coverage there, so broader successful strategy coverage accompanies higher task success.

BabaIsAI. DDO is again the strongest method on average success rate, H-ESD, and ESD, with gains across Basic, Room, Stop, and Flex. The Stop column is particularly informative: DDO raises success to 1.00 while also improving both coverage metrics, indicating that the gains extend to rule manipulation tasks where successful behavior depends on changing the rule structure. DivProb and the tie-aware baselines are strongest or tied in a few individual coverage columns. DDO leads the benchmark averages while improving success and coverage together.

WebShop. DDO is the strongest method across all three reported metrics. Because WebShop counts unique successful trajectories using both trajectory structure and purchase realization, the gains reflect broader successful strategy coverage under the composite trajectory-class definition.

## B Additional Method Details

## B.1 Action Parsing and Branch Output Probabilities

At each environment step, the LLM emits a model output $y _ { t }$ , and a benchmark-specific parser extracts the executable action for the environment. Some prompts structure $y _ { t }$ as reasoning text followed by an action field; DDO assigns sequence probability to the complete model output and uses the parsed action for execution. For a DTC record $C = ( x , B , \{ ( y _ { j } , r _ { j } ) \} _ { j \in B } ) , \pi _ { \theta } ( y _ { j } \ | \ x )$ is the sequence probability of the model output under standard left-to-right token factorization. The rollout determines the branch outcome label $r _ { j }$

## B.2 DTC Collection Details

The helper routines in Algorithm 1 are instantiated as follows. Across benchmarks, we set $K _ { d }$ to at most 5 and $K _ { a }$ to at most 3; shorter trajectories and branch rejection can yield fewer retained points or alternatives. SELECT allocates the divergence budget $K _ { d }$ over interior decision points $1 , \ldots , T - 1$ so that selected steps cover different trajectory depths. RESTORE returns the decision state associated with the selected source prefix, or ⊥ when the state cannot be constructed under the benchmark environment. ALT queries the expert model E for at most $K _ { a }$ alternative outputs from the shared state x, required to differ from the source output under the benchmark’s executable-action normalization. REJECT removes outputs that cannot be executed or that do not provide a distinct executable branch decision, and retained outputs are rolled out by E to obtain outcome labels.

<table><tr><td rowspan="2">Method</td><td colspan="5">Success rate ↑</td><td colspan="5">H-ESD ↑</td><td colspan="5">ESD↑</td></tr><tr><td>Goto.</td><td>Pick.</td><td>Open.</td><td>Comp.</td><td>Avg.</td><td>Goto.</td><td>Pick.</td><td>Open.</td><td>Comp.</td><td>Avg.</td><td>Goto.</td><td>Pick.</td><td>Open.</td><td>Comp.</td><td>Avg.</td></tr><tr><td>Base Reference</td><td>0.76 0.90</td><td>0.22 0.94</td><td>0.04 0.69</td><td>0.26 0.74</td><td>0.32 0.82</td><td>0.50 0.14</td><td>0.28 0.36</td><td>0.02 0.67</td><td>0.30 0.46</td><td>0.28 0.41</td><td>0.55 0.27</td><td>0.28 0.47</td><td>0.02 0.67</td><td>0.30 0.57</td><td>0.29 0.49</td></tr><tr><td>DPO</td><td>0.90</td><td>0.92</td><td>0.72</td><td>0.82</td><td>0.84</td><td>0.09</td><td>0.15</td><td>0.76</td><td>0.45</td><td>0.36</td><td>0.15</td><td>0.23</td><td>0.77</td><td>0.57</td><td>0.43</td></tr><tr><td>DivFreq</td><td>0.94</td><td>0.96</td><td>0.70</td><td>0.80</td><td>0.85</td><td>0.13</td><td>0.19</td><td>0.75</td><td>0.40</td><td>0.37</td><td>0.18</td><td>0.30</td><td>0.75</td><td>0.52</td><td>0.44</td></tr><tr><td>DivProb</td><td>0.94</td><td>0.98</td><td>0.78</td><td>0.86</td><td>0.89</td><td>0.13</td><td>0.16</td><td>0.79</td><td>0.45</td><td>0.38</td><td>0.18</td><td>0.22</td><td>0.80</td><td>0.55</td><td>0.44</td></tr><tr><td>TieDPO-RK</td><td>1.00</td><td>1.00</td><td>0.74</td><td>0.76</td><td>0.88</td><td>0.11</td><td>0.17</td><td>0.70</td><td>0.43</td><td>0.35</td><td>0.17</td><td>0.27</td><td>0.70</td><td>0.47</td><td>0.40</td></tr><tr><td>TieDPO-Dav</td><td>0.94</td><td>1.00</td><td>0.64</td><td>0.86</td><td>0.86</td><td>0.10</td><td>0.26</td><td>0.72</td><td>0.45</td><td>0.38</td><td>0.18</td><td>0.33</td><td>0.72</td><td>0.50</td><td>0.43</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DDO</td><td>0.94</td><td>1.00</td><td>0.80</td><td>0.92</td><td>0.92</td><td>0.21</td><td>0.24</td><td>0.84</td><td>0.51</td><td>0.45</td><td>0.30</td><td>0.35</td><td>0.85</td><td>0.57</td><td>0.52</td></tr></table>

Table 8: Main results on BabyAI. Avg. is the unweighted mean across the four tasks.
<table><tr><td rowspan="2">Method</td><td colspan="5">Success rate ↑</td><td colspan="5">H-ESD ↑</td><td colspan="5">ESD ↑</td></tr><tr><td>Basic.</td><td>Room.</td><td>Stop.</td><td>Flex.</td><td>Avg.</td><td>Basic.</td><td>Room.</td><td>Stop.</td><td>Flex.</td><td>Avg.</td><td>Basic.</td><td>Room.</td><td>Stop.</td><td>Flex.</td><td>Avg.</td></tr><tr><td>Base</td><td>0.22</td><td>0.26</td><td>0.08</td><td>0.26</td><td>0.21</td><td>0.02</td><td>0.05</td><td>0.06</td><td>0.19</td><td>0.08</td><td>0.02</td><td>0.05</td><td>0.07</td><td>0.22</td><td>0.09</td></tr><tr><td>Reference</td><td>0.94</td><td>0.97</td><td>0.86</td><td>0.84</td><td>0.90</td><td>0.20</td><td>0.11</td><td>0.59</td><td>0.59</td><td>0.37</td><td>0.29</td><td>0.17</td><td>0.62</td><td>0.64</td><td>0.43</td></tr><tr><td>DPO</td><td>0.96</td><td>0.92</td><td>0.92</td><td>0.96</td><td>0.94</td><td>0.10</td><td>0.06</td><td>0.58</td><td>0.50</td><td>0.31</td><td>0.19</td><td>0.10</td><td>0.65</td><td>0.59</td><td>0.38</td></tr><tr><td>DivFreq</td><td>0.94</td><td>0.92</td><td>0.78</td><td>0.98</td><td>0.91</td><td>0.13</td><td>0.07</td><td>0.58</td><td>0.42</td><td>0.30</td><td>0.25</td><td>0.12</td><td>0.71</td><td>0.47</td><td>0.39</td></tr><tr><td>DivProb</td><td>0.96</td><td>0.96</td><td>0.84</td><td>0.96</td><td>0.93</td><td>0.13</td><td>0.06</td><td>0.68</td><td>0.56</td><td>0.36</td><td>0.23</td><td>0.08</td><td>0.78</td><td>0.63</td><td>0.43</td></tr><tr><td>TieDPO-RK</td><td>0.94</td><td>0.90</td><td>0.60</td><td>0.76</td><td>0.80</td><td>0.14</td><td>0.07</td><td>0.45</td><td>0.44</td><td>0.28</td><td>0.25</td><td>0.10</td><td>0.53</td><td>0.52</td><td>0.35</td></tr><tr><td>TieDPO-Dav</td><td>0.90</td><td>0.94</td><td>0.50</td><td>0.74</td><td>0.77</td><td>0.19</td><td>0.07</td><td>0.42</td><td>0.54</td><td>0.30</td><td>0.28</td><td>0.10</td><td>0.50</td><td>0.62</td><td>0.38</td></tr><tr><td>DDO</td><td>0.98</td><td>0.98</td><td>1.00</td><td>0.98</td><td>0.99</td><td>0.18</td><td>0.07</td><td>0.75</td><td>0.52</td><td>0.38</td><td>0.30</td><td>0.13</td><td>0.81</td><td>0.63</td><td>0.47</td></tr></table>

Table 9: Main results on BabaIsAI. Avg. is the unweighted mean across the four tasks.

<table><tr><td>Method</td><td>Success rate ↑</td><td>H-ESD ↑</td><td>ESD↑</td></tr><tr><td>Base</td><td>0.18</td><td>0.02</td><td>0.02</td></tr><tr><td>Reference</td><td>0.26</td><td>0.15</td><td>0.17</td></tr><tr><td>DPO</td><td>0.26</td><td>0.11</td><td>0.12</td></tr><tr><td>DivFreq</td><td>0.26</td><td>0.12</td><td>0.13</td></tr><tr><td>DivProb</td><td>0.26</td><td>0.11</td><td>0.12</td></tr><tr><td>TieDPO-RK</td><td>0.28</td><td>0.16</td><td>0.17</td></tr><tr><td>TieDPO-Dav</td><td>0.16</td><td>0.13</td><td>0.13</td></tr><tr><td>DDO</td><td>0.36</td><td>0.21</td><td>0.22</td></tr></table>

Table 10: Main results on WebShop.

## C Experimental Details

## C.1 Benchmark Details

BabyAI provides controlled navigation and object interaction tasks with executable text actions. We use four tasks: goto, pickup, open, and pick-up sequence go-to. They are reported in Table 8 as Goto, Pick, Open, and Comp. For trajectory-class normalization, we collapse modulo-four same-direction turn repetitions and remove alternating turn blocks that cancel out, both of which leave the agent’s pose unchanged.

BabaIsAI provides tasks based on rule manipulation with executable text actions. We use four tasks: goto, two-room goto, two-room break-stop goto, and two-room optional break-stop goto. They are reported in Table 9 as Basic, Room, Stop, and Flex. For trajectory-class normalization, we drop non-terminal steps with no observation change and two-step inverse moves that return the agent to a prior state, both of which leave the environment state unchanged.

WebShop provides shopping tasks in which an agent searches, inspects products, selects options, and purchases an item for a user instruction. For trajectory-class normalization, the class combines the purchased item, selected options, and normalized trajectory structure, making coverage invariant to surface variation in the action trace. Figure 8 shows how this rule merges query variants along the same route while distinguishing different search and navigation routes to the same purchase.

Example input prompts and LLM outputs for all three benchmarks are shown in Appendix D.

## C.2 Models and Optimization

For the experiments, expert trajectories are collected with Qwen3.5-122B-A10B-FP8 under the shared thought-action format, and all optimized models use Qwen3-1.7B (Yang et al., 2025; Qwen Team, 2026). The FP8 model provides the expert trajectory source. SFT and preference post-training for the target model use bf16 mixed precision. Base denotes the unadapted target model before SFT or preference optimization for each task. For each benchmark task, the Reference model is a shared SFT adapter trained from the base model for 10 epochs. DPO (Rafailov et al., 2023), DivFreq, DivProb, TieDPO-RK, TieDPO-Dav, and DDO all start from this same Reference adapter, which is also used as the frozen reference model.

Same class (query variation). Two successful roll  
outs begin with search[blue coated steel   
end table] and search[blue-coated   
steel end table], follow the same nor  
malized trajectory structure, and purchase item   
B08MF23ZPL with the blue option.   
Different classes (route variation). Two success  
ful DDO rollouts purchase item B09P572DP9 with   
the redblack option. Their normalized routes are   
search → product inspection → refined search →   
purchase and search → pagination → refined search   
→ purchase.  
Figure 8: WebShop trajectory-class examples.

LoRA (Hu et al., 2022) is applied to the Q/K/V/O attention projections and MLP projections, with rank 32, alpha 32, and dropout 0.05. All preference runs use AdamW (Loshchilov and Hutter, 2019) with betas (0.9, 0.999), epsilon $1 0 ^ { - 8 }$ weight decay 0.0, gradient clipping at norm 1.0, and a linear learning-rate schedule. Unless otherwise stated, preference-trained methods use learning rate $5 \times 1 0 ^ { - 6 } .$ , warmup ratio 0.03, and $\beta = 0 . 1$ DDO uses $\alpha = 0 . 5$ . To construct $q _ { \mathrm { r e f } } .$ , the reference model scores the executable action field of each branch output.

## C.3 DTC Collection Cost

<table><tr><td>Collector</td><td>Requests</td><td>Tokens</td><td>Env. steps</td></tr><tr><td>DTC</td><td>28.6</td><td>59.7K</td><td>48.3</td></tr><tr><td>Without DTC</td><td>65.3</td><td>156.0K</td><td>64.7</td></tr></table>

Table 11: Collection cost per unique successful trajectory on BabyAI and BabaIsAI.

On BabyAI and BabaIsAI, DTC uses 44% of the expert requests, 38% of the expert tokens, and 75% of the environment steps used by Without DTC per unique successful trajectory. RTO requires no additional expert calls or environment interactions.

## C.4 Evaluation and Ablation Protocols

A fixed seed protocol is used throughout. Training collection and evaluation use disjoint task seeds or WebShop evaluation sessions. BabyAI and BabaIsAI use the epoch-5 preference policies, while Web-Shop uses the epoch-15 preference policies. Main BabyAI and BabaIsAI results use sampled decoding with a temperature of 0.6, top-p of 0.95, and a max token limit of 8192. WebShop coverage evaluation uses the same model and decoding/environment configuration as the corresponding successrate evaluation.

Main results. For BabyAI and BabaIsAI, success rate is computed from 50 rollouts per evaluation item. H-ESD and ESD use 20 rollouts for each of three seeds used for coverage evaluation and are averaged. For WebShop, success rate is reported over 50 evaluation sessions. WebShop coverage uses sessions 500, 501, and 502, with 20 rollouts per session and varied LLM sampling seeds. Failed purchases, invalid actions, retry exhaustion, and max-step failures count toward the fixed rollout denominator. Rollouts interrupted by system failures or missing model outputs are excluded and replaced with reruns.

Component comparison. The component comparison in Table 1 uses the BabyAI and BabaIsAI evaluation protocol described above. Settings without DTC train on same-state comparisons derived from separately sampled rollouts: successful and failed trajectories are sampled for the same task seed, converted into a preference comparison at their first shared decision state divergence, and matched to the corresponding budget for training comparisons. Settings with DTC use the statealigned branch sets. All four settings use the corresponding DPO and DDO hyperparameters in this appendix and report benchmark averages.

Strategic recovery. The strategic recovery probe in Figure 5 is run on BabyAI. For each method-task pair, we use successful evaluation rollouts, sample one interior decision point per rollout, replace the source action with a different valid task action, and roll out the same model from the edited prefix. To keep the probe balanced, the probe uses up to 30 successful rollouts per method-task pair. The reported value is success over valid edited rollouts.

Imitation control. The imitation control in Table 3 is run on BabyAI and BabaIsAI. It starts from the same SFT initialization for each task and continues SFT on all successful DTC branches. The matched condition uses the same optimizer step count as DDO. The full-epoch condition runs successful-only imitation for the same nominal epoch count as DDO. Exposure is the number of processed training examples normalized by the DDO comparison exposure. Metrics use the twobenchmark aggregate.

Decoding controls. The decoding diversification ablation in Table 4 reports an equal-weight benchmark macro average over BabyAI, BabaIsAI, and WebShop. The temperature sweep evaluates DPO at $T \in \{ 0 . 0 , 0 . 1 , 0 . 6 , 1 . 0 , 1 . 5 \}$ with one rollout per item. The DDO setting uses sampled decoding at $T = 0 . 6$ . All preference methods use epoch-5 policies for BabyAI and BabaIsAI and epoch-15 policies for WebShop.

Coverage growth. Coverage growth curves use the BabyAI, BabaIsAI, and WebShop rollouts used for coverage evaluation. For each $N _ { ☉ }$ , task averages are computed within each benchmark and then summarized with the three-benchmark aggregate.

Branch-Set Target Distribution. The target distribution comparison in Table 7 uses $\beta = 0 . 1$ . Under the parameterization in Eq. (3), $\alpha = 0 . 0$ targets a uniform distribution over the observed successful branches, $\alpha = 0 . 5$ contracts the reference logodds among successful branches halfway toward uniformity, and $\alpha = 1 . 0$ preserves the reference distribution restricted to successful branches.

Margin sharpness sensitivity. The $\beta$ sensitivity figure fixes $\alpha = 0 . 5$ and varies $\beta ,$ using the twobenchmark aggregate.

## D Example LLM Calls

This appendix shows representative input prompts and LLM outputs for each benchmark used in our evaluation. The three benchmarks share a common thought-action response format: the model produces a reasoning trace followed by an executable action, which a benchmark-specific parser extracts and forwards to the environment (Appendix B.1). Figures 9–11 illustrate this format on a single decision step in each benchmark.

BabyAI. The input prompt in Figure 9 contains the task instruction, a textual rendering of the gridworld observation around the agent, the recent interaction history, and the admissible action set. The

LLM output reasons over the visible objects and the current goal before emitting a single low-level navigation or manipulation action.

BabaIsAI. The input prompt in Figure 10 additionally exposes the active rule configuration of the puzzle, since success depends on identifying and, when necessary, manipulating the rules. The model reasons over the grid state and active rules, then outputs either a goal-directed action or an action that changes the puzzle’s rule configuration.

WebShop. The input prompt in Figure 11 provides the user’s purchase instruction, the contents of the current page, and the available interaction options such as search, product clicks, option selection, and purchase. The LLM output reasons about product attributes relative to the instruction before emitting a single page interaction.

![](images/c9555761b9f77e1050ee60cb5ee6e775958e51e13d2176092966b310f80041af.jpg)  
Figure 9: Example BabyAI input prompt and LLM output.

![](images/da722fd38acaf73ec8a7c6c8949dac47fadced441d34643641cae30992c50b98.jpg)  
Figure 10: Example BabaIsAI input prompt (part 1).

![](images/e86e91ef6ef2775c36c1b39224cedc6784c56971e835dd9f9d3695096c88c82b.jpg)  
Figure 10: Example BabaIsAI input prompt and LLM output (part 2).

![](images/14cdaef5cd35838f0324ff62964f14f984c10fa68660e8dba8c16e813eaba517.jpg)  
Figure 11: Example WebShop input prompt and LLM output.