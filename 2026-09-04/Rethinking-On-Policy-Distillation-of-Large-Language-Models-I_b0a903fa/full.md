# Rethinking On-Policy Distillation of Large Language Models II: One Training Example

Zixuan Fu<sup>∗1</sup>, Bingxiang He<sup>∗†‡1</sup>, Yuxin Zuo<sup>∗†1</sup>, Haohuan Huang<sup>∗1,2</sup>, Jinqian Zhang<sup>1</sup>, Ruhang Xiao<sup>3</sup>, Cheng Qian<sup>4</sup>, Qinyu Luo<sup>5</sup>, Huan-ang Gao<sup>1</sup>, Yudong Wang<sup>1</sup>, Zhiyuan Liu<sup>1</sup>, Ning Ding<sup>‡1</sup>, Chaojun Xiao<sup>‡1</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>Northeastern University <sup>4</sup>University of Illinois Urbana-Champaign <sup>5</sup>Johns Hopkins University

<sup>∗</sup>Equal Contribution. <sup>†</sup>Project Lead. <sup>‡</sup>Corresponding Authors.

<sup>§</sup> Code: https://github.com/Thinking-Space/One-Shot-OPD.

<sup>#</sup> hebx24@mails.tsinghua.edu.cn, {dingning,xcj}@tsinghua.edu.cn

## Why can OPD train on a single query for hundreds of steps and still im prove substantially?

![](images/901e0731348ac8dd14b29fc588959511b51c89b0d77f156790d0f7063257834d.jpg)

![](images/e798f78e1268789a9f7124165d457934c191f034e8833c3ca4e0aa7e3b57b0f0.jpg)  
Figure 1 | A single query recovers most of full-data OPD’s gain on mathematical reasoning (left), and the efect holds across domains and model families. Under multi-teacher OPD, 16 queries per domain match full-data training (right).

Abstract | On-policy distillation (OPD) combines student-generated rollouts with dense token-level supervision from a teacher. Existing work has mainly studied its algorithmic behavior, leaving the role of training data unclear. We examine this role at the data-minimal limit by training on a single query. One-shot OPD keeps improving for hundreds of steps and recovers most of full-data OPD’s gain across task domains and model families. We explain this result through the states visited during training and the rate at which the student aligns with the teacher. We measure state coverage, the fraction of the states full-data OPD visits that a query set’s rollouts reach. A single query already reaches 71.5%, most of it within the first 100 steps. Adding semantically distinct queries raises coverage and validation accuracy together, until 16 queries reach 98.9% and match full-data training. Yet alignment slows at a similar pace whether OPD trains on one query or the whole dataset, and even a fixed set of states takes hundreds of steps to absorb. OPD is therefore data-overfed but algorithm-starved. Its rollouts quickly expose broad supervision, while the student absorbs that supervision increasingly slowly. The state-coverage result extends to multi-teacher OPD, where 16 semantically diverse queries per domain match full-data MOPD. As a further stress test, content-light templates and of-domain WildChat queries also approach the real-query baseline. Task content and induced state coverage can therefore come apart. We hope these findings direct future work toward the step eficiency of OPD, and prompt a re-examination of the data and the mechanisms behind its recent successes in frontier post-training.

## 1. Introduction

On-policy distillation (OPD) is becoming increasingly common in frontier LLM post-training. Qwen3 [Yang et al., 2025], MiMo [Xiao et al., 2026], GLM-5 [Zeng et al., 2026], DeepSeek-V4 [Xu et al., 2026], and Kimi K3 [Team et al., 2026] all use OPD alongside supervised fine-tuning (SFT) and reinforcement learning (RL). What makes OPD distinctive is its combination of on-policy state visitation and dense distillation supervision [Gu et al., 2024, Agarwal et al., 2024]. The student samples its own rollouts, while the teacher provides the full next-token distribution at every visited prefix, yielding dense, token-level supervision rather than the single outcome-level reward typically used in reinforcement learning with verifiable rewards (RLVR).

A growing line of work has systematically investigated OPD’s training dynamics and mechanisms, largely from an algorithmic perspective, to explain how and why the method works [Li et al., 2026, Fu et al., 2026, Cai et al., 2026, Zhu et al., 2026]. Yet no prior work has studied how training data shapes OPD, or how the data and the algorithm interact. Closing this gap is essential to a complete understanding of OPD. In RLVR, Wang et al. [2026a] introduce one-shot RLVR as an extreme controlled experiment to isolate the role of training data. We bring the same experimental lens to OPD by training on a single query, a setting we call one-shot OPD. To our surprise, one-shot OPD produces a learning curve strikingly similar to that of one-shot RLVR as shown in Figure 1: despite repeatedly training on just one query, the model continues to improve for hundreds of steps and ultimately achieves a substantial gain. We therefore devote this work to answering a single question:

Why can OPD, when trained on just one example, keep learning for so long and improve by so much?

![](images/b5f956781522125e116c02d6d1bb215dfad33721e67572b0050c720d5a86b2a0.jpg)

![](images/304a3f053d6791069a3c2bc22b20b7e7d9f62daa5ba6a7a17d42726becd306ed.jpg)  
Figure 2 | OPD is data-overfed but algorithm-starved. Data: one query already covers most of the states full-data OPD visits. Algorithm: the student absorbs an ever smaller proportion of the remaining teacher–student gap, no matter how many queries it trains on.

This question exposes two aspects of OPD eficiency that standard training entangles: keep learning for so long concerns how fast the algorithm absorbs supervision, whereas improve by so much concerns why a small amount of data supply suficient supervision. By reducing the data supply to its minimum, one-shot OPD disentangles the two, and the phenomenon itself proves robust: the gain holds across four task domains and three model families, and persists even on queries the student never solves (Section 3). Our answer is that OPD is data-overfed but algorithm-starved. From the data perspective, a query acts on the student through the states its rollouts reach, each state being a prefix $s = ( x , y _ { < i } )$ at which the teacher supplies token-level supervision, so a query set is worth the part of that space it covers. We quantify this by state coverage: we group the states full-data OPD visits into semantic clusters and report the fraction a setting’s rollouts reach. A single query already supplies enough states on its own, covering 71.5% of what full-data OPD visits, and 16 semantically diverse queries cover 98.9% and match full-data training (Section 4). This is why so little data improves the student by so much. From the algorithm perspective, what falls as training proceeds is the absorption rate, the proportion of the remaining teacher–student gap that one update closes, and it declines in much the same way on one query as on all 17k (Section 5), so the pace of a run is a property of OPD rather than of the training set. In other words, the supervision a single query supplies remains largely unexploited, rate-limited by how quickly an on-policy student can absorb it.

The same mechanism informs data design beyond the one-shot setting. In multi-teacher OPD (MOPD) [Xiao et al., 2026, Xu et al., 2026], where a single student is trained across several domains and each query is routed to its domain teacher, we find that 16 semantically diverse queries per domain sufice to match full-data training (Section 6). Taking the data side to its extreme, we further show that even a training input that states no problem can remain efective: of-domain WildChat prompts and content-free templates drive OPD nearly as efectively as the full training set (Section 7.1). Therefore, an input is useful largely because it starts the student reasoning. Furthermore, we conduct a controlled comparison with one-shot RLVR on the same query to make the algorithmic contrast concrete: the outcome reward is exhausted once the query is solved on nearly every rollout, whereas OPD’s token-level signal persists throughout training, and OPD’s validation gain over 1000 steps is more than twice that of RLVR (Section 7.2).

In short, OPD is supplied with more supervision than its algorithm can absorb. What data curation has to settle is therefore no longer how many problems to collect, but which states an input induces. We hope this state-level view of the training data can guide future work on selecting queries by the states they induce, and on raising the rate at which a student absorbs them.

## 2. Preliminaries

## 2.1. Notation

Let $x \sim \mathcal { D }$ denote a training input and $y = ( y _ { 1 } , \ldots , y _ { L } )$ a response, with $y _ { < i } = ( y _ { 1 } , \dotsc , y _ { i - 1 } )$ the prefix up to token position �. We consider two LLMs: a student $\pi _ { \theta }$ and a teacher $\pi _ { T }$ , each defining a nexttoken distribution over a shared vocabulary. A trajectory is a pair $( x , y )$ with � sampled autoregressively from a policy given �. We use � to index token positions and � to index optimization steps, writing $\theta _ { t }$ for the student parameters at step �. A state is the autoregressive context $s _ { i } = ( x , y _ { < i } )$ , the object on which the OPD objective is defined.

## 2.2. On-Policy Distillation

On-policy distillation (OPD) samples trajectories from the student and aligns the student to the teacher on the prefixes the student actually visits [Agarwal et al., 2024]. OPD can also be viewed as a special case of dense KL-constrained reinforcement learning, where the teacher distribution induces a token-level reward and the KL regularizer has a fixed relative weight [Yang et al., 2026].

In the full-distribution form, OPD minimizes a per-token KL divergence on visited states:

$$
\mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } } \left[ \sum _ { i = 1 } ^ { L } \mathrm { K L } ( \pi _ { \theta } ( \cdot  { | } s _ { i } )  { | | } \pi _ { T } ( \cdot  { | } s _ { i } ) ) \right] .\tag{1}
$$

In practice, we estimate it with a per-token advantage:

$$
A _ { i } ^ { \mathrm { O P D } } = \log \pi _ { T } ( y _ { i } \mid s ) - \log \pi _ { \theta } ( y _ { i } \mid s ) ,\tag{2}
$$

where $s = ( x , y _ { < i } )$ . Two properties follow from this formulation: (i) the signal is local in that it depends only on the prediction problem at state $s ,$ not on the trajectory’s outcome; and (ii) it is dense in that the teacher supplies a distributional signal at every visited state instead of a terminal scalar reward.

Equation 2 evaluates the correction only at the token the student emitted. A second estimator of the same divergence keeps the student’s � most likely tokens at each visited state and weights them by the student’s probability:

$$
A _ { i } ^ { \mathrm { t o p - } k } = \sum _ { \nu \in \mathcal { V } _ { i } } \tilde { \pi } _ { \theta } ( \nu \mid s ) \left[ \log \pi _ { T } ( \nu \mid s ) - \log \pi _ { \theta } ( \nu \mid s ) \right] ,\tag{3}
$$

where $\mathcal { V } _ { i } = \mathrm { T o p K } ( \pi _ { \theta } ( \cdot \mid s ) , k )$ and $\tilde { \pi } _ { \boldsymbol { \theta } }$ is $\pi _ { \theta }$ renormalized over that set. It truncates the divergence to $k$ terms rather than estimating it from one sampled token, trading the distribution’s tail for lower variance. Section 3.1 states which form each run uses. The alignment metrics below are diagnostics computed from the two policies’ logged distributions, so they are available under either form.

## 2.3. Dynamic Metrics

Gap recovery. Because evaluation criteria and initial teacher–student performance gaps vary across domains and model pairs, raw score improvements are not directly comparable. We therefore report improvement as a ratio to the initial teacher–student gap. Let $M _ { 0 } , M _ { T }$ , and $M _ { t }$ denote the evaluation scores of the initial student, the teacher, and the student at optimization step $t ,$ respectively. The gap recovery ratio at step � is $\frac { M _ { t } { - } M _ { 0 } } { M _ { T } { - } M _ { 0 } } \times 1 0 0 \%$ . A value of 0%, 100%, or above 100% indicates no improvement toward the teacher, matching the teacher, or surpassing the teacher, respectively.

Full-data recovery. When the reference is full-data OPD rather than the teacher, we normalize by the gain of full-data OPD instead. Let $M _ { F }$ denote the full-data OPD score at the same optimization step. The full-data recovery ratio is $\frac { M _ { t } - M _ { 0 } } { M _ { F } - M _ { 0 } } \times 1 0 0 \% .$ A value approaching 100% means that a reduced query set nearly matches the gain of full-data OPD while a value above 100% means it surpasses it.

Top-� token overlap ratio. This metric measures agreement between the two policies’ high-probability token sets. Let $S _ { t } ^ { \theta } ( s ) = \mathrm { T o p K } ( \pi _ { \theta _ { t } } ( \cdot \mid s ) , k )$ and $S ^ { T } ( s ) = \mathrm { T o p K } ( \pi _ { T } ( \cdot \mid s ) , k )$ . The overlap at optimization step � is

$$
O _ { t } ^ { ( k ) } = \frac { 1 } { \vert \mathcal { T } _ { t } \vert } \sum _ { i \in \mathcal { T } _ { t } } \frac { \vert S _ { t } ^ { \theta } ( s _ { i } ) \cap S ^ { T } ( s _ { i } ) \vert } { k } ,\tag{4}
$$

where $\mathcal { T } _ { t }$ indexes the non-padding response positions of the rollouts in the batch collected at step � [Li et al., 2026].

Overlap-token advantage. Inside that agreement, we average the teacher–student log-probability diference over the shared tokens $S _ { t } ^ { \theta } ( s _ { i } ) \cap S ^ { T } ( s _ { i } )$ , weighting each shared token by the probability the student assigns it. The result reports how far apart the two policies remain on the tokens they already rank highly. Because of that weighting, and because it averages over top-� entries rather than over sampled tokens, it sits on a much smaller scale than the teacher–student distance of Section 5.1, and the levels of the two are not comparable.

## 3. The One-Shot Phenomenon

This section presents the experimental setup and results of One-Shot OPD. We investigate how much of full-data OPD’s gain a single query recovers, and how robust that gain is.

Table 1 | Student–teacher pairs used in our experiments.
<table><tr><td>Setting</td><td>Student</td><td>Teacher</td></tr><tr><td>Mathematical reasoning</td><td>DeepSeek-R1-Distill-Qwen-1.5B (R1-Distill-1.5B) Llama-3.2-3B-Instruct</td><td>JustRL-DeepSeek-1.5B (JustRL-1.5B) GT-Llama-3.2-3B-Instruct-MATH</td></tr><tr><td>Code generation</td><td>OLMo-3-7B-Instruct-DPO (OLMo-7B-It-DPO) DeepSeek-R1-Distill-Qwen-1.5B (R1-Distill-1.5B)</td><td>OLMo-3-7B-Instruct (OLMo-7B-It) Nemotron-Research-Reasoning-Qwen- 1.5B-v2-RLVE</td></tr><tr><td>Instruction following</td><td>DeepSeek-R1-Distill-Qwen-1.5B</td><td>(Nemotron-1.5B) UltraData-IF-1.5B</td></tr><tr><td>Agentic tool use</td><td>(R1-Distill-1.5B) Qwen2.5-Coder-1.5B-Instruct (Qwen-Coder-1.5B)</td><td>Hammer2.1-1.5b</td></tr></table>

## Takeaways

One query is enough to induce robust OPD gains. A single query recovers most of the teacher–student gap across task domains and model families. The gain is insensitive to query dificulty, response length, and sampling temperature, and a query the student never solves is as efective as one it always solves.

## 3.1. Experimental Setup

We establish one-shot OPD across math, code generation, instruction following, and agentic tool use for controlled analysis to test generality.

Models. For each domain, we pair a student with a post-trained teacher from the same family. Table 1 lists the full identifiers and the short names used throughout. Math, code generation, and instruction following share the student R1-Distill-1.5B [Guo et al., 2025], whose teachers are, respectively, JustRL-1.5B [He et al., 2025b], Nemotron-1.5B [Zeng et al., 2025], and an instruction-following variant we post-train from the same student (Appendix A.1). For agentic tool use, the student is Qwen-Coder-1.5B [Hui et al., 2024] and the teacher is Hammer-1.5B [Lin et al., 2024]. To test whether the phenomenon is specific to Qwen-based pairs, we add two mathematical-reasoning pairs from other families: Llama-3B-It [Grattafiori et al., 2024] with GT-Llama-3B-Math [Zhang et al., 2025], and OLMo-7B-It-DPO with OLMo-7B-It [Olmo et al., 2025].

Datasets. The domain-specific training sets comprise DAPO-Math-17K [Yu et al., 2026] for mathematical reasoning, Open-R1 Codeforces [Penedo et al., 2025] for code generation, a sampled subset of UltraData-SFT-2605 [OpenBMB, 2026] for instruction following, and xLAM-function-calling-60K [Liu et al., 2024] for agentic tool use. For One-shot OPD in mathematics, we select three queries spanning easy, medium, and hard initial dificulty, scored by the student’s pass rate over 8 rollouts before training (8/8, 4/8, and 0/8, respectively). In each of the other domains, the one-shot query is sampled randomly from the corresponding training set. Appendix A.1 details the dataset preprocessing and the selected one-shot queries.

Training. We implement OPD in veRL [Sheng et al., 2025]. The mathematical-reasoning runs of this section optimize the top-� advantage of Eq. 3 with � = 16; the code, instruction-following, and agentic runs optimize the sampled-token advantage of Eq. 2. Every update uses a batch of 64 rollouts. We use AdamW with a learning rate of 10<sup>−6</sup>, a rollout temperature of 1.0, and a gradient clip norm of 1.0; remaining hyperparameters are listed in Table 3.

![](images/eb43a9cb5eb94ee00ef2adb9d6efe8bf614e40b480d1bf2ab8e85f43974982ac.jpg)  
Figure 3 | Validation accuracy and Training dynamics of one-shot OPD on mathematical reasoning.

![](images/4b2e910833511a50cdf729c63edc6a517571f64d79b642d607f09d9e56a3198b.jpg)  
Figure 4 | Validation accuracy of one-shot OPD across 3 student–teacher families, on mathematical reasoning averaged over MATH-500 and AMC 2023. AIME 2025 is excluded because the Llama pair scores near zero on it.

Evaluation. For math, we evaluate on MATH-500 [Hendrycks et al., 2021], AMC 2023 [Li et al., 2024], and AIME 2025 [Balunovic et al., 2025], sampling 16 responses per problem and reporting avg@16 accuracy. For code generation, LiveCodeBench v6 (LCB v6) [Jain et al., 2025] uses 3 sampled solutions per problem and reports avg@3 with the oficial execution-based evaluator. For instruction following, Multi-IF [He et al., 2024] evaluates three-turn conversations and reports the final-turn score averaged over its eight languages. For agentic tool use, BFCL v3 [Patil et al., 2025] reports avg@8 over the evaluated subsets. And the response caps are 31,744 tokens for mathematics, 65,536 for LCB v6, 16,384 per turn for Multi-IF, and 4,096 for BFCL v3. Unless stated otherwise, mathematics figures report validation accuracy macro-averaged over MATH-500, AMC 2023, and AIME 2025, and a dashed line marks the teacher.

![](images/e8db9ac0a974ec0999e5abac458386ed96b455d8099a021cfdf3c12f4e4c8cec.jpg)

Figure 5 | Validation accuracy of one-shot OPD across three task domains.  
![](images/4d27e69fc83182007e65682154c3f5a1154cfd84ee0e5181d50c1521cd4b9593.jpg)  
Figure 6 | Validation Accuracy of one-shot OPD under varying query dificulty, response-length cap on the hard query, and sampling temperature on the medium query.

## 3.2. Experimental Results

One-shot OPD recovers most of full-data OPD’s gain in mathematics. Figure 3 shows that training on a single query improves accuracy on MATH-500, AIME 2025, and AMC 2023, approaching the full-data OPD on all three benchmarks. Averaged over them, one-shot OPD reaches 68.5 against 69.8 for full-data OPD, recovering 69% of the teacher–student gap and 87% of full-data OPD’s gain at step 300. Beyond step 300 both curves stay within a band of about 3 points, and the recovered fraction ranges from 62% to 89% through step 1000, where one-shot OPD reaches 68.4 against 72.1 and recovers 72% of full-data OPD’s gain (Figure 1). The training dynamics follow the same alignment process under both settings: the top-16 overlap ratio climbs to the full-data level, the overlap-token advantage approaches zero, and the absolute entropy gap nearly closes. Thus, even when all rollouts originate from a single query, OPD improves accuracy while progressively aligning the student distribution with the teacher on the visited states.

The one-shot efect is robust across families and task domains. Figure 4 shows that one-shot OPD improves mathematical reasoning across all three student–teacher families. At the final checkpoint, the averaged scores increase from 77.1, 28.2, and 70.8 for the respective student baselines to 85.5, 40.2, and 82.4 for R1-Distill-1.5B, Llama-3B-It, and OLMo-7B-It-DPO. Figure 5 further shows that the efect extends beyond mathematical reasoning: on code generation, instruction following, and agentic tool use, one-shot OPD recovers 73%, 66%, and 64% of the corresponding teacher–student gaps, respectively. Together, these results show that the one-shot efect is robust across both model families and task domains.

The one-shot efect is robust to query and rollout properties. We next examine whether one-shot OPD remains efective under varying query dificulty, response-length budgets, and rollout sampling temperature. As shown in Figure 6, one-shot OPD works across easy, medium, and hard queries, even though the fraction of correct rollouts evolves very diferently in the three settings (Appendix A.1): the easy query is solved on nearly every step, the medium query becomes largely solvable during training, and the hard query is never solved. Tightening the response-length cap and lowering the rollout temperature both preserve the one-shot gain. Together, these results show that one-shot OPD is robust to query dificulty, response-length budget, and rollout sampling temperature.

## 4. Data Perspective: Abundant States

Section 3 showed that One-Shot OPD recovers much of full-data OPD’s gain robustly. This section asks why training on a single query produces such a large gain, and whether the same explanation also covers the cases where n-shot OPD can match full-data OPD. <sup>1</sup>

## Takeaways

• One query already reaches a large part of the full-data state space. What OPD consumes is primarily states rather than queries. In practice, repeated rollouts from a single query can reach 71.5% state coverage, with most of it within the first 100 steps.

Diverse but few queries reach suficient states and rival full-data performance. Adding semantically distinct queries raises both state coverage and validation accuracy, and 16 queries already match full-data OPD. The value of an extra example lies in whether it covers states missed by earlier ones.

## 4.1. State Coverage of One-Shot OPD

Setup. OPD trains on states rather than queries. A query � and a sampled response � produce one state $s = ( x , y _ { < i } )$ at every token position �, each paired with a target distribution from the teacher. With 64 rollouts per update, even a single query yields tens of thousands of supervised states, so query count can substantially understate the amount of supervision available to OPD.

Hypothesis

A single query can provide enough supervision to recover most of full-data OPD’s gain if its rollouts cover a broad part of the state space visited by full-data OPD.

To test this hypothesis, we measure the breadth rather than the raw number of visited states, since every generated token creates a distinct prefix. We call this measure state coverage: we represent states in a shared representation space, partition that space into clusters, and report the fraction reached by a setting’s rollouts.

• Representation. We represent each state by its teacher signature ℎ<sub>�</sub>(�), the teacher’s final-layer hidden vector at the state’s last token.

• Reference space. We pool states visited by full-data OPD over its entire run on DAPO-Math-17K, sampling 8 uniformly spaced positions from each rollout. A held-out portion of these rollouts takes no part in defining the clusters and is measured as a setting of its own, full data (held-out).<sup>2</sup>

• Clusters. After PCA, �-means partitions the reference signatures into � = 200 clusters. Each state falls in the nearest one, denoted �(�).

![](images/661c1809e7093ddcb16f8a965430d96079164de5f6c13ea49f00a9426beab4b8.jpg)

![](images/03e662dd631dfd97ca355cc3344981b0f91071a8d99525944766056d4cc90f3c.jpg)  
Figure 7 | State coverage and validation accuracy of one-shot OPD on the medium query, against full-data OPD. Coverage is measured against the state space full-data OPD reaches, which is therefore the 1.0 level rather than a separate curve.

The state coverage of a set of states � is then the fraction of the � clusters it reaches:

$$
\operatorname { C o v } ( S ) = { \frac { 1 } { K } } \left| \left\{ c ( s ) \ : \ s \in S \right\} \right| .\tag{5}
$$

Coverage records which clusters a setting reaches, not how often it visits them. We fit the clusters once and measure every setting over the same 300 steps. Full data (held-out) reaches 100% on this budget, so the top of the scale is a level full-data OPD attains rather than a maximum the construction guarantees. Appendix B.1 gives the construction and schedule, and shows that the comparisons below are stable across choices of $K ,$ reference set, and state positions.

One-shot OPD reaches 71.5% of the state space. Figure 7 shows that one-shot OPD reaches 71.5% state coverage by step 300. Most of this coverage appears early: the run reaches 65.9% by step 100, then adds only 5.6 percentage points over the next 200 steps. Repeated rollouts from the same query therefore continue to discover new clusters, but at a sharply diminishing rate. The same run raises validation accuracy from 59.1 to 66.9, compared with 70.8 for full-data OPD at step 300. These results show the same pattern on the data side: a single query already covers most of the state-space clusters reached by full-data OPD and produces a substantial validation gain. One-shot OPD is therefore small in query count, yet broad in the supervision it generates.

## 4.2. Diversity Expands State Coverage

The analysis above shows that one query already reaches most of the full-data state space, but it leaves the causal question open: a run that trains well might simply visit more states along the way, with the extra states doing none of the work. We therefore ablate state coverage directly, varying how many distinct states the training data reaches while holding the rest of the setup fixed.

Setup. We raise data diversity from two directions, each with its own control. (i) Response diversity (of-policy). We sample 64 trajectories once from the initial student on the one-shot query and keep that pool fixed, which also removes a confound of on-policy training, where the states keep changing as the student does. Nested subsets retain 1, 4, 16, or all 64 trajectories, and the retained ones are repeated to fill each batch of 64. Every condition shares the query, the batch size, and the optimization budget, so the number of distinct trajectories, and hence of states, is the only quantity that changes. (ii) Query diversity (on-policy). These runs instead change the query set, and all follow the mathematical configuration. Starting from the medium query used by one-shot OPD, we build 4- and 16-shot sets by clustering DAPO-Math-17K with BGE-M3 [Chen et al., 2024a] and taking one representative per semantic cluster, so each query added to the ladder is semantically distinct from those already in it, and we compare these sets with full-data OPD. We report validation accuracy for every run, and state coverage for the on-policy ones, where the rollouts are those the student actually generates during training. Of-policy scores are averaged over the five evaluation checkpoints from step 300 to 500. On the coverage side, the full-data bar is the held-out split that sets the 100% ceiling.

![](images/8855d9a5d960a47141cf7b00a1f1750269ad2b66512a397f549c2e506e9e8f7e.jpg)

![](images/a7ecc2dc6875f8c3c2e468441ea1c6285bd2659001746d7dd52db212bd1678fd.jpg)

![](images/bc2b6edb28cdf45f4c15b58824133f970a5d5170b0b8ca37c85d59d0e7ab2dc8.jpg)  
Figure 8 | Validation accuracy and state coverage as the training data is made more diverse, on the response side by of-policy training on distinct trajectories and on the query side by on-policy training on distinct queries.

Data diversity is state diversity, and a few queries match full-data OPD. Figure 8 shows of-policy validation accuracy rising monotonically with the number of distinct trajectories. Everything else is held fixed, so the only thing the extra trajectories add is states. The on-policy runs reach the same conclusion from the query side. Semantically distinct queries raise validation accuracy until 16 of them match full-data OPD, and they match it on state coverage at the same point, which rises from 71.5% for one query to 98.9% for 16. Every query on the ladder is a new semantic cluster, so it raises diversity and count together, and two controls in Appendix B.2 separate them: holding the count at 16, drawing those queries from 16 semantic clusters rather than from one raises state coverage and validation accuracy sharply, whereas the order in which the student trains on a fixed query set does not matter. What an added query is worth is therefore set by whether it reaches new states, and a few diverse queries already reach nearly all the states full-data OPD does.

## 5. Algorithm Perspective: Slow Alignment

Section 4 analyzed the one-shot gain from the data side: what a query is worth to the student is set by the states its rollouts reach. Turning to the algorithm side, this section asks why one query keeps producing gains over so many steps.

## Takeaways

Alignment slows over time, and this trend is insensitive to query count. The student continuously narrows the teacher–student gap throughout the run, but the rate of progress diminishes steadily, which is why a run requires hundreds of steps rather than tens. Increasing the number of queries does not accelerate or decelerate this process.

Even a fixed set of states takes hundreds of steps to learn from. A run that trains on the same states from beginning to end still improves over hundreds of steps, so a steady supply of fresh states is not required to keep a run going that long.

![](images/a496eea4933966a4d949746ef00ed86e10215205fe74eb26c7ff997d6e392d26.jpg)  
Figure 9 | Teacher–student alignment for OPD trained on 1, 4, 16, and all DAPO-Math-17k queries. Distance (left) and absorption rate (right) are both on logarithmic vertical axes against linear steps.

## 5.1. Alignment Metrics

OPD aligns the student with the teacher. Two quantities describe the state of that alignment at any moment: how much teacher–student disagreement is left, and how fast it is falling. We measure both on the states the student actually visits.

• Distance. The teacher and the student disagree at each visited position, by the per-token advantage of Eq. 2. We take the distance $d _ { t }$ to be the average size of that disagreement over the positions $\mathcal { T } _ { t }$ the student visits at step �:

$$
d _ { t } = \frac { 1 } { \vert \mathcal { T } _ { t } \vert } \sum _ { i \in \mathcal { T } _ { t } } \left. \log \pi _ { T } ( y _ { i } \mid s _ { i } ) - \log \pi _ { \theta _ { t } } ( y _ { i } \mid s _ { i } ) \right. .\tag{6}
$$

Here $s _ { i } = ( x , y _ { < i } )$ is the state at token position $i ,$ as defined in Section 2. Averaging magnitudes keeps positive and negative per-token gaps from cancelling, so $d _ { t }$ falls to zero only as the two policies come to agree at the positions the student visits.

• Absorption rate. The absorption rate $\upsilon _ { t }$ is then the proportion of that distance one update absorbs,

$$
\nu _ { t } = \frac { d _ { t } - d _ { t + 1 } } { d _ { t } } .\tag{7}
$$

We track both metrics from step 30, after gradient clipping no longer binds. Since each run measures distance over the states it visits, these runs have settled at diferent levels by then. We therefore report each run against its own value at step 30, which we refer to as the distance left $d _ { t } / { d _ { 3 0 } } . ^ { 3 }$

## 5.2. Alignment Slows Similarly Across Data

We track $d _ { t }$ and $\upsilon _ { t }$ for OPD trained on 1, 4, 16, and all DAPO-Math-17k queries, over the same 300 steps and under the mathematical configuration of Section 3.1.

The student aligns ever more slowly. The distance falls for the whole run, so the student is never stuck. What slows is the absorption rate, which falls throughout the run in the right panel of Figure 9: on the logarithmic distance axis at left a constant rate would trace a straight line, and all four curves bend flatter instead. Each update thus absorbs less of what is left than the last, which is why a run takes hundreds of steps rather than tens.

![](images/3e44e6219490b32d903c9682dafd3a2a7833c56171f224784d659c00a05cb647.jpg)  
Figure 10 | Validation accuracy and alignment metrics for the on-policy and of-policy one-shot runs.

The absorption rate falls in much the same way at every training-set size. The four runs do this alike: each removes 78% to 84% of its step-30 distance by step 300, and all four slow by a similar factor during training (Appendix C.1 gives the per-run numbers and the conventions these readings are taken under). One query therefore does not make alignment slower or faster: the pace is a property of OPD rather than of the training set. The learning rate changes how many steps that takes, but not the way the distance falls (Appendix C.2).

## 5.3. Ablation on Fixed States

On-policy training keeps producing new states as the student changes, and one reading of the hundreds of steps is that the run lasts as long as that supply does. Holding the training states fixed tests that reading directly: if a fixed set of states is exhausted quickly, the of-policy run should finish early.

Setup. We compare two one-shot runs that difer only in where their training states come from. The always-on-policy run draws fresh rollouts from the current student at every step. The always-of-policy run reuses the 64 trajectories of Section 4.2, sampled once from the initial student, for every update, so its training states stay fixed while the student changes.

Holding the states fixed leaves the run just as long. Figure 10 shows the of-policy run gains accuracy steadily for about 200 steps and then stops, so its gain builds up over hundreds of steps rather than over the first few updates. The alignment metrics move just as slowly: the of-policy run’s top-16 token overlap rises and its overlap-token advantage approaches zero only over hundreds of steps. Removing that supply therefore does not shorten the run. This suggests that a fixed set of states is enough on its own to keep an OPD run going for hundreds of steps, so the length of a run is not explained by how long fresh states keep arriving.

## 6. One-Shot Extends to Multi-Teacher OPD

Section 4.2 showed that a small, semantically diverse query set can match full-data OPD within one domain. Modern post-training pipelines often use multi-teacher on-policy distillation (MOPD) [Xiao et al., 2026, Xu et al., 2026], where one student is trained on several domains in a single run and each query is routed to its domain teacher. This section asks whether the same query-diversity result carries over: can a small, diverse query set for each domain match full-data MOPD?

## Takeaways

16 semantically diverse queries within each domain are suficient to match full-data MOPD, which reaches nearly the same average validation accuracy as separate full-data OPD.

![](images/ab22a780c07ea8d573a29150e1cda9ad192006614405a7c016a53c6ec92907fe.jpg)  
Figure 11 | Validation accuracy of one-shot, 16-shot, and full-data MOPD.

Setup. We train one R1-Distill-1.5B student with MOPD on mathematical reasoning, code generation, and instruction following. The domain teachers from Section 3.1 are JustRL-1.5B for mathematics, Nemotron-1.5B for code, and UltraData-IF-1.5B for instruction following. We exclude agentic tool use because it uses a diferent student, Qwen-Coder-1.5B. The three settings are one-shot, 16-shot, and full-data MOPD. As in Section 4.2, the 16-shot setting selects one representative from each of 16 BGE-M3 [Chen et al., 2024a] semantic clusters fitted separately within each domain. All settings run for 300 steps with the hyperparameters in Table 8.

Full-data MOPD matches separate full-data OPD. At step 300, full-data MOPD raises the average validation accuracy from 43.5 to 52.8, recovering 79% of the teacher–student gap. The three corresponding full-data OPD runs reach 53.8. This 1.0-point diference shows that MOPD remains efective when all three domains are trained in a single run and establishes full-data MOPD as the reference for the query-diversity comparison below.

16 diverse queries per domain match full-data MOPD. Increasing the query count from 1 to 16 per domain raises the average validation accuracy from 50.1 to 52.9 at step 300. Full-data MOPD reaches 52.8, so 16-shot MOPD recovers 101% of full-data MOPD’s gain. The result is not driven by one domain: calculated separately, the corresponding proportions are 93% for mathematics, 136% for code generation, and 109% for instruction following. Because 16-shot and full-data MOPD use the same rollout and optimization budgets, expanding each domain’s query set beyond 16 diverse queries produces no additional average gain in this setting.

Together, these results are consistent with a state-space account for MOPD: semantically diverse queries can induce complementary states within each domain, allowing 16 queries per domain to match full-data MOPD.

## 7. Discussion

The mechanism above shifts attention from prompts to the regions of reasoning states they induce. An input is useful when the student’s rollouts reach regions where the teacher can provide useful supervision. Stating a problem is only one way to reach them. This view raises two broader questions: how much task content must the input itself contain, and why does the same one-shot setting behave diferently under OPD and RLVR? We discuss these questions through content-light and of-domain inputs, followed by a controlled comparison with one-shot RLVR.

![](images/f176e05b47de4b12671632f2c3cddc5ebb7d785046eb4c463ca5d6427a2f0bcf.jpg)

Figure 12 | The two content-light templates used as OPD training inputs.  
![](images/755ab234d4f634b51e0cdaf41e5950682bcddedd30934d5163857ea0014c4979.jpg)  
Figure 13 | Validation accuracy and actor entropy for mathematics OPD trained on the real training set, on a content-light template, or on general-domain WildChat queries.

## 7.1. Inputs as State Generators

The state-generator view suggests that task content may be unnecessary if an input still induces trajectories that the teacher can supervise. We stress-test this prediction with an empty user turn ending in <think>, the same template with a simple domain system prompt, and general-domain queries from WildChat [Zhao et al., 2024] (Figure 12). The WildChat set contains 192,824 predominantly English queries from general conversation, creative writing, role-playing, and rewriting. Conservative heuristics label only 0.17% as mathematics-related and 2.63% as code-related.

Content-light and of-domain inputs can still induce useful states. All three conditions track the real-query baseline, which lifts the three-benchmark average from 59.1 to 69.8, and all three reach that level on a third to a half of its rollout tokens (Figure 13). Code generation gives the same picture (Appendix E.1). However, the requirement is not empty: a scafold that closes the block at once, <think>\n</think>\n, collapses into short meta-level replies we cannot train on. These results do not imply that task content is generally dispensable. They show that content is not the only source of useful signal in OPD. An input also acts as a state generator, and even content-light or of-domain inputs can sometimes lead the student to states that support useful teacher supervision. Explicit domain content changes final performance by about one point in these experiments. This provides a complementary view of OPD data eficiency. Data quality depends not only on the problems being collected, but also on the state regions they induce and the teacher targets available there.

## 7.2. Relation to One-Shot RLVR

One-shot OPD and one-shot RLVR [Wang et al., 2026a] share the same surface phenomenon that a single query supports continued improvement over hundreds of updates in OPD and over a thousand updates in RLVR. The source of learning is diferent. OPD receives a teacher target at every visited state, while RLVR receives a verifiable outcome for each trajectory. We compare them on the same medium query to isolate this diference (Figure 14).

![](images/dabc2bd9e64195ea8a93e438e302bb6599905d0a3f37be6c4b8b4b72620a435e.jpg)  
Figure 14 | Validation accuracy and training dynamics for one-shot OPD and one-shot RLVR on the same query. Both take a batch of 64 queries per step. RLVR draws 8 rollouts per query and scores each by outcome under GRPO. OPD draws 1 and scores every sampled token against the teacher. Training accuracy is the fraction of rollouts that solve the training query.

Useful inputs difer across objectives. RLVR updates from trajectory outcomes. Its training input must therefore define a task whose sampled outcomes can be verified. The outcomes must also vary enough to produce a useful advantage. OPD instead updates from the teacher–student gap at each visited state. It can receive useful signal whenever a prompt leads to trajectories that expose such gaps. The value of an OPD input is therefore not tied directly to verifiability or dificulty. Our results further show that even weak domain alignment can work in the settings we study. Coverage of the induced state regions still matters (Section 4), but obtaining such coverage is less restrictive than obtaining tasks with verifiable outcomes.

OPD extracts more signal from the same query. Both algorithms obtain fresh rollouts from the training query at every step. Over 1000 steps, OPD closes 72% of its gap to the teacher. Its gain in validation accuracy is more than twice that of RLVR. This advantage remains when the two methods are matched by rollout tokens rather than training steps (Figure 14).

The RLVR signal weakens as the model learns to solve the training query. Its groups soon become nearly unanimous, which leaves no advantage under GRPO. OPD does not depend on variation in trajectory outcomes. It can continue learning from local teacher–student gaps after the query is solved, although these gaps gradually shrink. One-shot RLVR is therefore limited by the outcome variation available from its query. One-shot OPD is instead limited by how quickly it absorbs dense teacher supervision, as shown in Sections 4 and 5. The objectives also imply diferent ceilings. RLVR is not tied to a teacher’s distribution, while OPD is trained to match one.

## 8. Related Work

On-policy distillation. On-policy distillation (OPD) trains the student on its own rollouts with teacher supervision [Song and Zheng, 2026]. MiniLLM [Gu et al., 2024] optimized this under reverse KL by policy gradient, and GKD [Agarwal et al., 2024] generalized it across divergences and on- and of-policy mixtures. Yang et al. [2026] cast OPD as dense KL-constrained RL whose per-token log-ratio acts as an implicit reward. OPD now appears across frontier post-training stacks [Yang et al., 2025, Zeng et al., 2026, Xiao et al., 2026, Xu et al., 2026, Blakeman et al., 2026], several of which adopt the multi-teacher form we study in Section 6, routing each query to a domain specialist. A growing line explains why OPD is efective, and does so entirely on the algorithm side. One thread studies its objective and update geometry: the dense per-token target replaces the single sequence-level scalar of outcome-reward RL, which in parameter space steers an early-locked, low-rank update subspace [Cai et al., 2026, Shen et al., 2026] and can be recast as token-level policy gradient [Ko et al., 2026], connecting to analyses that separate the dense updates of SFT from the localized subnetworks of RL [Mukherjee et al., 2026, Zhu et al., 2025]. A second thread charts when OPD succeeds or fails: a thinking pattern shared between teacher and student and the limits of an over-strong teacher [Li et al., 2026], which tokens are actually learnable [Armandpour et al., 2026, Wang et al., 2026b], and recurring failure modes with simple fixes [Zhu et al., 2026, Fu et al., 2026]. These accounts take the training set as given. We instead ask how many training inputs OPD needs, and find that the binding resource is not the data but the rate at which the student absorbs it.

Data-eficient reasoning post-training. A parallel line studies how little data elicits strong reasoning. Under sparse outcome-reward RL, Wang et al. [2026a] show that a single example keeps improving a model for over a thousand steps, the precedent we adapt to the dense-signal setting of OPD. LIMR [Li et al., 2025] prunes the RLVR training set several-fold by ranking each example’s learning impact. With supervised fine-tuning, LIMA [Zhou et al., 2023], LIMO [Ye et al., 2025], and s1 [Muennighof et al., 2025] elicit competitive alignment or reasoning from on the order of a thousand curated examples. Within OPD, scheduling supervision over reasoning prefixes cuts training cost severalfold [Zhang et al., 2026]. Broader post-training work selects high-value subsets by influence or quality [Xia et al., 2024, Chen et al., 2024b, He et al., 2025a,c]. These works share our data-eficiency theme, but their supervision still depends on curated problems, delivered as a sparse outcome scalar in the RL case or as reference solutions in the SFT case. OPD instead supplies a dense per-token target at each visited prefix and remains efective on queries the student never solves (Section 3), where the teacher signal stays informative even on incorrect rollouts [Armandpour et al., 2026], so the limiting factor moves from whether a useful signal is available, the standing dificulty under sparse reward, to how fast an on-policy student absorbs an already-abundant one (Section 5.2).

Synthetic and unsupervised post-training data. A growing body of work reduces post-training’s dependence on curated, human-written data. One direction synthesizes the inputs: Self-Instruct [Wang et al., 2023] and Evol-Instruct [Xu et al., 2024] bootstrap instructions from a language model, and Magpie [Xu et al., 2025] elicits full instruction–response pairs from nothing but a chat-template prefix. A second direction removes ground-truth labels from RL [He et al., 2026], replacing verifiable rewards with self-generated signals such as model confidence or entropy [Zhao et al., 2026b, Agarwal et al., 2026] and majority agreement across rollouts [Zuo et al., 2026], or drawing reward from unlabeled corpora [Dong et al., 2025]. A third removes external problems altogether, letting the model propose its own tasks [Zhao et al., 2026a, Zweiger et al., 2026]. These lines relax supervision on the reward or the query source. Our template- and WildChat-OPD results (Section 7.1) push the query side further: because the teacher supplies dense supervision at each visited prefix, the query’s role reduces to placing the student in a useful reasoning state, so an of-domain chat log or even a content-free template drives OPD nearly as well as a curated problem.

## 9. Conclusion, Limitations, and Future Work

Conclusion. Motivated by one-shot RLVR [Wang et al., 2026a], we asked whether a single query is also enough for OPD, and it is: one-shot OPD keeps the student improving for hundreds of steps and recovers most of full-data OPD’s gain, across model families and task domains. The data and the algorithm sides give one answer: OPD is data-overfed but algorithm-starved. A query acts on the student through the states its rollouts reach, and one query already covers most of the state space full-data OPD visits, while 16 semantically diverse queries match full-data training in single-domain OPD and in MOPD alike. What limits a run is the absorption rate: the proportion of the remaining teacher–student gap one update absorbs keeps falling, at the same pace on one query as on all 17k, so the optimizer rather than the training set sets how long a run takes. An input need only start the student reasoning, which turns data design from collecting problems into choosing teachers.

Limitations. State coverage is a semantic-level proxy measured against a reference space built from full-data rollouts, so it reports how much of that space a query set reaches rather than what it covers on its own, and it weights every cluster equally, however often the cluster is visited and however much teacher signal it still carries. What sets the absorption rate is still open. Our MOPD runs use three domains with one teacher each, leaving open how far the diversity result extends as teachers are added.

Future work. Three directions follow.

• Selecting data by state coverage: the question is no longer how many queries to collect but which states they induce, so state coverage becomes a selection criterion once it can be estimated from the queries themselves, without a full-data run to measure it against.

• Making training more step-eficient, by reusing a batch for several epochs under a trust region on the per-token gap, or weighting tokens by how much teacher signal they still carry.

• Growing the setting, to MOPD with more teachers and domains, and to larger students, agentic tool use, and long contexts, where training data is most expensive to collect.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pages 21246– 21263, 2024.

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable efectiveness of entropy minimization in llm reasoning. Advances in Neural Information Processing Systems, 38: 107150–107180, 2026.

Mohammadreza Armandpour, Fatih Ilhan, David Harrison, Ajay Jaiswal, Duc NM Hoang, Fartash Faghri, Yizhe Zhang, Minsik Cho, and Mehrdad Farajtabar. Unmasking on-policy distillation: Where it helps, where it hurts, and why. arXiv preprint arXiv:2605.10889, 2026.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions. Advances in Neural Information Processing Systems, 38, 2025.

Aaron Blakeman, Aaron Thomas, Aastha Jhunjhunwala, Abhibha Gupta, Abhinav Khattar, Adam Rajfer, Adi Renduchintala, Adil Asif, Aditya Vavre, Adriana Flores Miranda, et al. Nemotron 3 ultra: Open, eficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2606.15007, 2026.

Yuchen Cai, Ding Cao, Liang Lin, Chunxi Luo, Xin Xu, Kai Yang, Weijie Liu, Saiyong Yang, Tianxiang Zhao, Guangzhong Sun, et al. Learning to foresee: Unveiling the unlocking eficiency of on-policy distillation. arXiv preprint arXiv:2605.11739, 2026.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 4(5), 2024a.

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al. Alpagasus: Training a better alpaca with fewer data. In International Conference on Learning Representations, volume 2024, pages 34767–34797, 2024b.

Qingxiu Dong, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, and Furu Wei. Reinforcement pre-training. arXiv preprint arXiv:2506.08007, 2025.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562, 2026.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations, volume 2024, pages 32694–32717, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Bingxiang He, Ning Ding, Cheng Qian, Jia Deng, Ganqu Cui, Lifan Yuan, Haiwen Hong, Huan-ang Gao, Longtao Huang, Hui Xue, et al. The right time matters: Data arrangement afects zero-shot generalization in instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 222–243, 2025a.

Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, et al. Justrl: Scaling a 1.5 b llm with a simple rl recipe. arXiv preprint arXiv:2512.16649, 2025b.

Bingxiang He, Wenbin Zhang, Jiaxi Song, Cheng Qian, Zixuan Fu, Bowen Sun, Ning Ding, Haiwen Hong, Longtao Huang, Hui Xue, et al. Air: A systematic analysis of annotations, instructions, and response pairs in preference dataset. arXiv preprint arXiv:2504.03612, 2025c.

Bingxiang He, Yuxin Zuo, Zeyuan Liu, Shangziqi Zhao, Zixuan Fu, Junlin Yang, Cheng Qian, Kaiyan Zhang, Yuchen Fan, Ganqu Cui, et al. How far can unsupervised rlvr scale llm training? In International Conference on Learning Representations, volume 2026, pages 14823–14865, 2026.

Yun He, Di Jin, Chaoqi Wang, Chloe Bi, Karishma Mandyam, Hejia Zhang, Chen Zhu, Ning Li, Tengyu Xu, Hongjiang Lv, et al. Multi-if: Benchmarking llms on multi-turn and multilingual instructions following. arXiv preprint arXiv:2410.15553, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation

of large language models for code. In International Conference on Learning Representations, volume 2025, pages 58791–58831, 2025.

Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning eficiently via relaxed on-policy distillation. arXiv preprint arXiv:2603.11137, 2026.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling. arXiv preprint arXiv:2502.11886, 2025.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026.

Qiqiang Lin, Muning Wen, Qiuying Peng, Guanyu Nie, Junwei Liao, Jun Wang, Xiaoyun Mo, Jiamu Zhou, Cheng Cheng, Yin Zhao, et al. Hammer: Robust function-calling for on-device language models via function masking. arXiv preprint arXiv:2410.04587, 2024.

Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, et al. Apigen: Automated pipeline for generating verifiable and diverse function-calling datasets. Advances in Neural Information Processing Systems, 37:54463–54482, 2024.

Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, et al. Deepcoder: A fully open-source 14b coder at o3-mini level. Notion Blog, 1, 2025.

Niklas Muennighof, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20286–20332, 2025.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models. Advances in Neural Information Processing Systems, 38:132119–132138, 2026.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025.

OpenBMB. Ultradata-sft-2605, 2026. URL https://huggingface.co/datasets/openbmb/Ul traData-SFT-2605.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.

Guilherme Penedo, Anton Lozhkov, Hynek Kydlíček, Loubna Ben Allal, Edward Beeching, Agustín Piqueres Lajarín, Quentin Gallouédec, Nathan Habib, Lewis Tunstall, and Leandro von Werra. Codeforces. https://huggingface.co/datasets/open-r1/codeforces, 2025.

Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hanna Hajishirzi. Generalizing verifiable instruction following. Advances in Neural Information Processing Systems, 38, 2026.

Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo Han, Sunbowen Lee, and Yi R Fung. On the geometry of on-policy distillation. arXiv preprint arXiv:2606.07082, 2026.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626, 2026.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. Advances in Neural Information Processing Systems, 38:122721–122764, 2026a.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 13484–13508, 2023.

Yuanyi Wang, Su Lu, Yanggan Gu, Pengkai Wang, Yifan Yang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, and Hongxia Yang. Not all disagreement is learnable: Token teachability in on-policy distillation. arXiv preprint arXiv:2605.26844, 2026b.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333, 2024.

Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. Mimo-v2-flash technical report. arXiv preprint arXiv:2601.02780, 2026.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly eficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In International Conference on Learning Representations, volume 2024, pages 30745– 30766, 2024.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In International Conference on Learning Representations, volume 2025, pages 76346–76382, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Zhiyuan Zeng, Hamish Ivison, Yiping Wang, Lifan Yuan, Shuyue Stella Li, Zhuorui Ye, Siting Li, Jacqueline He, Runlong Zhou, Tong Chen, et al. Rlve: Scaling up reinforcement learning for language models with adaptive verifiable environments. arXiv preprint arXiv:2511.07317, 2025.

Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler II, Qian Qian, Gregory D Lyng, Sanjit Singh Batra, and Robert E Tillman. Fast and efective on-policy distillation from reasoning prefixes. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 25553–25569, 2026.

Zizhuo Zhang, Jianing Zhu, Xinmu Ge, Zihua Zhao, Zhanke Zhou, Xuan Li, Xiao Feng, Jiangchao Yao, and Bo Han. Co-reward: Self-supervised reinforcement learning for large language model reasoning via contrastive agreement. arXiv e-prints, pages arXiv–2508, 2025.

Andrew Zhao, Yiran Wu, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. Advances in Neural Information Processing Systems, 38:105816–105879, 2026a.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatgpt interaction logs in the wild. In International Conference on Learning Representations, volume 2024, pages 34590–34605, 2024.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. In International Conference on Learning Representations, volume 2026, pages 2548–2581, 2026b.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023.

Hanqing Zhu, Zhenyu Zhang, Hanxian Huang, DiJia Su, Zechun Liu, Jiawei Zhao, Igor Fedorov, Hamed Pirsiavash, Zhizhou Sha, Jinwon Lee, et al. The path not taken: Rlvr provably learns of the principals. arXiv preprint arXiv:2511.08567, 2025.

Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, and Ge Liu. The many faces of on-policy distillation: Pitfalls, mechanisms, and fixes. arXiv preprint arXiv:2605.11182, 2026.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Xinwei Long, Ermo Hua, Biqing Qi, et al. Ttrl: Test-time reinforcement learning. Advances in Neural Information Processing Systems, 38:131459–131483, 2026.

Adam Zweiger, Jyo Pari, Han Guo, Yoon Kim, and Pulkit Agrawal. Self-adapting language models. Advances in Neural Information Processing Systems, 38:74084–74115, 2026.

<table><tr><td>Difficulty</td><td>ID</td><td>Query</td></tr><tr><td>Easy</td><td>t59</td><td>In triangle ABC, let D be the foot of the altitude from A. Suppose that  $A D = 4 ,$   $B D = 3 , C D = 2 ,$  and AB is extended past B to a point E such that  $B E = 5$  Determine the value of  $C E ^ { 2 }$ </td></tr><tr><td>Medium</td><td>t22</td><td>Let S be the set of triples  $( a , b , c )$  of non-negative integers such that a + b + c is even. Determine the value of  $\sum ( a , b , c ) \in S { \frac { 1 } { 2 ^ { a } 3 ^ { b } 5 ^ { c } } }$  . This sum can be expressed as  ${ \frac { m } { n } } ;$  where m and n are relatively prime positive integers. Compute  $m + n .$ </td></tr><tr><td>Hard</td><td>t56</td><td>Byan is playing a game called &quot;raven, raven, falcon&quot; with his three friends. His friends sit in a circle, and Byan walks clockwise around them, tapping each friend he passes on the head and saying either &quot;raven&quot; or &quot;falcon,&quot; each with probability  ${ \frac { 1 } { 2 } } .$  The game ends when Byan has said “falcon&quot; twice. The probability that one of his friends will be called a “falcon&quot; twice can be expressed as  ${ \frac { m } { n } } .$  , where m and n are relatively prime positive integers. Find  $1 0 0 m + n .$ </td></tr></table>

Table 2 | Mathematical queries used in the query-dificulty experiment, labelled by the student’s pass rate before OPD training.

## A. Details for Section 3

## A.1. Experimental Setup and Training Queries

Instruction-following setup. Unlike the other primary-domain teachers, the instruction-following teacher is trained specifically for our experiments. Starting from DeepSeek-R1-Distill-Qwen-1.5B, we post-train the model on synthetic instruction-following tasks with programmatically checkable constraints. We use GRPO with rule-based outcome rewards that indicate whether the response satisfies the target-turn constraints. Training samples four completions per prompt at temperature 0.9, uses a maximum prompt length of 4,096 tokens and a maximum response length of 16,384 tokens, and optimizes with AdamW at learning rate $1 \times 1 0 ^ { - 6 }$ , train batch size 16, and clip ratio 0.2. The step-600 checkpoint is used as the instruction-following teacher, improving IFBench [Pyatkin et al., 2026] from 14.33 to 20.65 and Multi-IF final-turn accuracy from 20.84 to 28.58.

For this domain, “full-data” denotes the subset sampled from UltraData-SFT-2605 [OpenBMB, 2026] rather than the complete dataset, and the one-shot query is drawn at random from that subset.

Mathematical queries for one-shot OPD (Table 2, Figure 15). We sample 8 rollouts from the untrained student for each of the first 64 queries of DAPO-Math-17K and assign dificulty by the resulting pass rate, which selects t59 (8/8), t22 (4/8), and t56 (0/8) as the easy, medium, and hard queries. How often each is solved during training follows that assignment: the easy query stays near 1.0 throughout, the medium one rises from 0.36 to about 0.89, and the hard one is never solved in 300 steps, yet all three produce comparable benchmark gains in Figure 6.

Default hyperparameters for OPD (Table 3). Unless otherwise noted, all experiments use these defaults. The top-� row covers both objectives: the mathematical one-shot runs of Section 3.2 use $K = 1 6 ,$ and all other experiments use the sampled-token objective.

## B. Details for Section 4

## B.1. Training-State Coverage

Construction. Rollouts are collected by optimization step rather than at random, so that every stage of training is equally represented and the budget is comparable across settings, which run for the same number of steps: at each step from 1 to 500 of the full-data run we take 5 rollouts, three building the clusters and two forming full data (held-out), for 12,000 reference and 8,000 held-out states. PCA reduces these to 50 dimensions, retaining 71.9% of the variance, before �-means. Every evaluated setting is then measured on that same schedule, drawing 2 of the 5 rollouts at each step and accumulating over the run, so the step-300 budget of 600 rollouts matches the held-out split. We repeat the draw 30 times; bands and error bars are one standard deviation over these repetitions, which reflects rollout subsampling rather than variation across trained models.

![](images/a7a1e0211f1eb6ef413a160173c029d3222cdaf8c8b17e943722a2e79ce0f842.jpg)  
Figure 15 | Fraction of the 64 rollouts per step that solve the training query itself, for the three queries of Table 2.

Table 3 | Default hyperparameters for OPD.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Rollout batch size</td><td>64</td></tr><tr><td>Mini batch size</td><td>64</td></tr><tr><td>Responses per prompt</td><td>1</td></tr><tr><td>KL coefficient</td><td>0.0</td></tr><tr><td>LogProb top-K</td><td>0(default) / 16</td></tr><tr><td>Top-K strategy</td><td>Student Top-K</td></tr><tr><td>Loss aggregation</td><td>token-mean</td></tr><tr><td>Training temperature</td><td>1.0</td></tr><tr><td>Top-p</td><td>1.0</td></tr><tr><td>Optimizer</td><td>AdamW (β₁=0.9, β2=0.999, wd = 0.01)</td></tr><tr><td>Learning rate</td><td>1e-6</td></tr><tr><td>Gradient clip norm</td><td>1.0</td></tr><tr><td>Max prompt length Max response length</td><td>1024 (math); 4096 (code, IF, agentic) 7168 (math, code, IF); 2048 (agentic)</td></tr></table>

The ordering is stable across �, reference construction, and state positions (Tables 4 and 5, Figure 16). Coverage depends on how finely the reference states are clustered and on which states define the clusters, so we vary both. The first table repeats the analysis at � ∈ {50, 100, 200, 500}: finer clusters lower absolute coverage, but coverage still increases from 1- to 4- to 16-shot with 16-shot close to full data. The second rebuilds the clusters from the union of 2,000 rollouts per setting instead of from full-data states alone, and the ordering survives; because states from all settings help define these pooled clusters, pooled coverage is not a recovery measure, and its rows are to be read only against each other. The figure varies the positions read from each rollout, comparing the final non-padding token and fixed absolute positions 2,000 and 4,000 against the 8 relative positions of the main analysis. All four preserve the ordering. The fixed-position variants land close to the main construction, whereas the final token gives markedly lower coverage when there are few queries, which is what one expects of a position governed by response length and termination; we therefore read 8 relative positions, which sample throughout a response of any length.

Table 4 | Full-data-anchored training-state coverage at step 300 under diferent cluster counts �.
<table><tr><td>K</td><td>1-shot</td><td>4-shot</td><td>16-shot</td><td>Full data (held-out)</td></tr><tr><td>50</td><td>0.755</td><td>0.829</td><td>1.000</td><td>1.000</td></tr><tr><td>100</td><td>0.727</td><td>0.803</td><td>1.000</td><td>1.000</td></tr><tr><td>200</td><td>0.715</td><td>0.798</td><td>0.989</td><td>1.000</td></tr><tr><td>500</td><td>0.564</td><td>0.695</td><td>0.904</td><td>0.956</td></tr></table>

Table 5 | Training-state coverage under alternative cluster constructions, at � = 200.
<table><tr><td>Reference construction</td><td>1-shot</td><td>4-shot</td><td>16-shot</td><td>Full data (held-out)</td></tr><tr><td>Full-data anchored</td><td>0.663</td><td>0.773</td><td>0.953</td><td>0.998</td></tr><tr><td>Equal-size pooled</td><td>0.750</td><td>0.845</td><td>0.958</td><td>0.966</td></tr></table>

One-shot has a heavier tail of distant states, but is not separable from full data (Table 6). Nearest-centroid assignment places a state in some cluster however far it lies from every full-data state, so coverage alone cannot tell a setting that fills the reference space thinly from one that leaves it. We therefore also measure, for each state, the mean cosine distance to its ten nearest reference states in the unreduced 1536-dimensional space, and call a state of-reference when that distance exceeds the 95th percentile of the held-out full-data states. One-shot OPD exceeds the threshold more often than the other settings, so its distances have a heavier upper tail. But an AUROC of 0.552 against held-out full-data states is close to chance, so one-shot states are not separable from full-data states as a whole.

## B.2. Controls on the Training Data

Diversity in 16-shot OPD (Figure 17). Every query on the ladder of Section 4.2 represents a diferent semantic cluster, so it raises diversity and count together. Holding the count at 16 and drawing the second set from a single cluster separates them: the 16 cluster representatives reach 70.9 validation accuracy at step 300 against 69.9 for the single-cluster set, which is about what 4-shot OPD reaches with a quarter as many queries, and their state coverage separates the same way and does so early, 98.9% against a single-cluster set that flattens near 76.8% within 100 steps. Queries from one cluster keep returning the student to the part of the state space it has already seen.

Query scheduling in 2-shot OPD (Figure 18). Holding a two-query set fixed and varying only when each query is shown, the sequential schedule trains on the hard query for the first 300 steps of a 600-step budget and on the medium query for the rest, while the mixed schedule puts 32 copies of each in every batch. At step 600 they reach 68.8 and 69.1, both above the 67.4 of the one-shot baseline they share, so ordering a fixed query set barely matters next to what the second query itself buys. These runs train longer than those of Section 4.2 and take the hard query as their baseline, so neither their horizon nor their baseline is interchangeable with Figure 8.

## C. Details for Section 5

![](images/6ee3a0a580deeed36844a9974e6f9b2d131956dafb5f9615a870c4f51932a9ad.jpg)  
Figure 16 | Training-state coverage under alternative state-extraction positions, at $K = 2 0 0$ . Error bars show one standard deviation over rollout subsamples.

Table 6 | Reference-distance diagnostics across query-set sizes. AUROC compares each setting with the held-out full-data states. The last row is the complete full-data pool, including the rollouts that built the clusters, so its exceedance rate need not equal the 5% the threshold is set at.
<table><tr><td>Setting</td><td>Exceedance (%)</td><td>Mean distance</td><td>AUROC</td></tr><tr><td>1-shot</td><td>17.2</td><td>0.283</td><td>0.552</td></tr><tr><td>4-shot</td><td>4.4</td><td>0.218</td><td>0.477</td></tr><tr><td>16-shot</td><td>3.8</td><td>0.231</td><td>0.512</td></tr><tr><td>Full data (all)</td><td>4.0</td><td>0.215</td><td>0.476</td></tr></table>

## C.1. Estimating the Absorption Rate and the Measurement Window

Section 5.2 measures under three conventions. Why each one:

• [�/1.4, 1.4�]. A diference between adjacent steps is mostly noise once � moves slowly, so $\upsilon _ { t }$ is read as the slope of log � across the endpoints of this window, using the median � near each. Only centers whose window fits inside the measured range are plotted, which is why the main text reads the rate at steps 50 and 200 rather than at step 300.

$d _ { t } / d _ { 3 0 }$ . Each run measures its distance on the states it visits and the four sit at diferent levels by step 30, so plotting each against its own value there lets the curves be compared by shape rather than by level.

• Start at step 30. Gradient clipping binds until then, and a clipped update is not the update the objective asked for: $\nu _ { t }$ inside that window measures what onefixed-size step buys rather than what one step buys, which is why the measured rate rises there rather than falls.

Table 7 collects the readings Section 5.2 quotes from this range: the proportion of the step-30 distance still left at step 300, and the factor by which the absorption rate falls between steps 50 and 200.

## C.2. The Learning Rate Rescales the Step Axis

We repeat on-policy one-shot OPD at half, the default, and twice the default learning rate, holding every other setting fixed, and read all three over the same 300 steps as Section 5.2.

The learning rate rescales the step axis and leaves the slowdown untouched. A larger learning rate reaches any given distance sooner, the three curves separated by roughly the ratio of the learning rates (Figure 19). Changing the horizontal axis from the step � to $u = ( \mathrm { l r } / 1 0 ^ { - 6 } )$ � brings them onto one curve: the spread between the three runs narrows from about 2.2× to about 1.1×. The learning rate therefore sets only how fast a run moves along that one curve, not the shape of the curve itself. It bends flatter as it goes, just as the four query settings of Section 5.2 do against the step count. A proportionally larger learning rate therefore shortens the run proportionally, over the fourfold range we tested, without reaching whatever makes alignment slow in the first place.

![](images/56e15ff557a6542db4c9dc9a464d7b5ca224adddc811929411fbffdc79cd81b7.jpg)  
Figure 17 | Two 16-shot query sets that difer only in diversity, against full-data OPD.

![](images/813a017f85aeb5a462d46814968cd0b58e6b7cb244bacfd72846e0f4b043f0f9.jpg)  
Figure 18 | Efect of query scheduling in 2-shot OPD. The vertical dotted line marks the query switch in the sequential schedule.

## D. Details for Section 6

Table 8 shows the default hyperparameters for MOPD.

## E. Details for Section 7

## E.1. Template and WildChat OPD in Code Generation

Section 7.1 reports the four input conditions in mathematics. Here we report the same four in code generation.

Setup. The student is R1-Distill-1.5B, the teacher DeepCoder-1.5B-Preview [Luo et al., 2025], and the baseline full-data OPD on TACO [Li et al., 2023]. The base template, the system template, and the WildChat queries are exactly those of Section 7.1; the system template’s one-line instruction is the competitive-programming instruction of Figure 12b. We evaluate on LiveCodeBench v6 under avg@3.

![](images/557fb8e95bc889820d6e5c5dfbb59451aca8bea6a2a5ad84cdef0ba36b494ed3.jpg)

Table 7 | Alignment dynamics over steps 30–300 for on-policy OPD, at the default learning rate.
<table><tr><td>Setting (queries)</td><td>1</td><td>4</td><td>16</td><td>Full data</td></tr><tr><td>d3oo/d30</td><td>0.16</td><td>0.17</td><td>0.18</td><td>0.22</td></tr><tr><td>v50/v200</td><td>5.6</td><td>5.2</td><td>4.2</td><td>6.4</td></tr></table>

Figure 19 | Validation accuracy and teacher–student distance for one-shot OPD at half, one, and twice the default learning rate. Distance is on a logarithmic vertical axis against linear time.  
Table 8 | Default hyperparameters for MOPD (multi-teacher on-policy distillation).
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Number of teachers LogProb top-k KL coefficient</td><td>3 (math / code /IF) 0 0.0</td></tr><tr><td>Loss aggregation Rollout batch size Mini batch size Responses per prompt Training temperature</td><td>seq-mean-token-mean 64 64 1</td></tr><tr><td>Top-p Optimizer</td><td>1.0 1.0</td></tr><tr><td>Learning rate Gradient clipping</td><td>AdamW (β₁=0.9, β₂=0.999, wd = 0.01) 1e-6 1.0</td></tr></table>

![](images/76abe61dd7aa9ba157d965f31770a9a9195331017d65758f15286fea1a0d71c0.jpg)  
Figure 20 | Validation accuracy and actor entropy for code-generation OPD.

Results. Figure 20 shows the four conditions ending within 1.7 points of one another, and the token accounting separating them much further than the step accounting does: over the same 500 steps the baseline spends 277M rollout tokens against 18M for the base template, with the other two in between. Actor entropy orders the conditions exactly as it does in mathematics, with WildChat highest and the system template lowest. The one diference is that no condition falls behind here, whereas WildChat costs about a point in mathematics.