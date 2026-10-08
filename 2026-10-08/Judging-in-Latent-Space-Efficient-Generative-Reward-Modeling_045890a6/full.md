# Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression

Mingqing Yuan<sup>1</sup>, Xiaobo Liang<sup>1</sup>, Junwei Yang<sup>2</sup>, Ziwei Chen<sup>3</sup>, Zeren Zhang<sup>4</sup>, Hejin Wang<sup>5</sup>, Yubin Wang<sup>6</sup>, Juntao Li<sup>1</sup>\*

<sup>1</sup> Soochow University <sup>2</sup> University of Cambridge

<sup>3</sup> Chalmers University of Technology <sup>4</sup> Peking University

<sup>5</sup> Tsinghua University <sup>6</sup> The Hong Kong University of Science and Technology

## Abstract

Reward modeling often requires jointly representing and reasoning over multiple evaluation criteria, yet verbalizing this process token by token can incur substantial inference cost. Recent work on latent reasoning suggests that continuous states may support this computation more compactly. We introduce LatentGRM, a latent evaluation framework built on semantic chunking, compression, and reconstruction. By using the structure of rubric-guided evaluations to guide compression, LatentGRM learns compact continuous trajectories that support autonomous pairwise judgments without generating textual assessments. A separate interpreter reconstructs evaluation text from these trajectories, providing an offline view of the information retained under compression. Under matched training data and backbones, LatentGRM achieves competitive aggregate preference accuracy relative to explicit Supervised Fine-Tuning (SFT) judges at both 4B and 8B scales. Across four benchmark domains, LatentGRM-8B compresses evaluation trajectories by 8.9–9.2× and reduces total judge inference time by 6.1–7.0× at vote@5. Controlled rubric interventions show that criterion-dependent preference information is carried through the latent sequence. Together, these results demonstrate that continuous latent evaluation can substantially reduce inference cost while preserving competitive judgment quality.

Code https://github.com/EADMO/LatentGRM Correspondence mqyuan@stu.suda.edu.cn • ljt@suda.edu.cn

![](images/a8c8ba9bbb78fa40755411c3b39266131a7186f449595c3e8688ac924b47332f.jpg)

![](images/ecc8e76dcc88afa349168a33c7eb9fd6f712569586ef5e7f6ffc372b6f1fae4d.jpg)

![](images/8b9ec7a17048a4866062db3a721d561e6a116db765128cb654dd184944ae34ac.jpg)

![](images/3a4f854df26befbf51ab018bedd15b59fb41f7006f6cbeaf355d9721c288377a.jpg)  
Figure 1: Accuracy–efficiency scaling of LatentGRM-8B (stars) and Rubric-RM-8B (circles) from vote@1 to vote@9. Points run left to right at 1, 3, 5, 7, and 9 votes.

## 1 Introduction

Reward modeling often requires integrating evidence across multiple aspects of a response. A response may be factually correct yet violate an explicit constraint, while a detailed answer may fall short of a request for concision. These assessments must be combined into a preference that guides decisions in policy optimization and response selection (Ouyang et al., 2022; Zhang et al., 2025a). Since reward models are used repeatedly in these settings, both judgment accuracy and inference efficiency are central to their practical value.

Among existing approaches, Generative Reward Models (GRMs) have become a dominant paradigm. Generative reward models formulate evaluation as language generation, with reasoningbased judges producing intermediate assessments before issuing a verdict (Zhang et al., 2025a; Mahan et al., 2024; Chen et al., 2026). Recently, rubric-based reward modeling has gained increasing attention by organizing these assessments around rubrics: explicit sets of task-specific evaluation criteria, such as correctness, instruction compliance, and clarity (Liu et al., 2026; Yuan et al., 2026). While sampling and aggregating multiple evaluations can improve judgment quality, verbalizing judgments and supporting evidence for each criterion can lengthen evaluation trajectories and increase autoregressive decoding costs. This raises a central question: can a generative judge retain the information neededfor multi-criterion evaluation while carrying out its intermediate computation more compactly?

To address this, we turn to continuous latent reasoning, which offers a promising way to represent intermediate computation with fewer autoregressive steps (Hao et al., 2025; Shen et al., 2025; Deng et al., 2025). Rubric-based reward modeling provides a particularly structured setting for studying such compact computation. Its criterion-level organization and linguistic boundaries can guide where to place compression boundaries, while the criteria themselves provide natural reference points for probing how latent judgments respond to changes in evaluation requirements and what assessment content remains recoverable.

We introduce LatentGRM, a framework for rubric-guided latent evaluation. Its organizing idea is to treat rubric criteria as intervenable and alignable criterion-level semantic units. Building on Latent-SFT’s vocabulary-space distillation framework (Deng et al., 2025), Semantic Chunking uses rubric-aligned blocks to allocate a fixed latent budget, then uses linguistic cues to refine boundaries within each block. The trained judge generates compact continuous trajectories and predicts pairwise preferences autonomously. A separate Latent Trace Interpreter (LTI) reconstructs evaluation text from saved trajectories outside the online judgment path. Aligning its output with teacher assessments by criterion provides a fine-grained measure of reconstruction fidelity on teacher-derived trajectories. Rubric structure therefore connects compression, intervention, and interpretation within one evaluation framework.

Against explicit Supervised Fine-Tuning (SFT) judges trained on matched data and 4B/8B backbones, LatentGRM achieves competitive aggregate preference accuracy across eight benchmark domains. On four domains, LatentGRM-8B compresses evaluation trajectories by 8.9–9.2× and reduces total judge inference time by 6.1–7.0× at vote@5. Controlled rubric edits and latent-source replacement show that criterion-dependent preference information is carried through autonomous latent trajectories. Separately, reconstruction experiments recover criterion-level states and justifications from teacher-derived trajectories. These findings support efficient latent judgment together with the recovery of useful evaluation content.

Our contributions are:

• We introduce LatentGRM, to our knowledge the first framework to bring continuous latent reasoning to reward modeling, combining rubric-guided semantic compression, autonomous latent evaluation, and a separate interpreter for retrospective reconstruction.

• Under matched training data and backbones, we demonstrate competitive aggregate preference accuracy at both 4B and 8B scales and substantial reductions in trajectory length and total judge inference time.

• We examine the evaluation information retained under compression through controlled rubric interventions and criterion-aligned reconstruction. These analyses show that latent trajectories carry criterion-dependent preference information and that the LTI can recover criterionlevel supporting justifications from teacher-derived trajectories.

## 2 Related Works

## 2.1 Latent Reasoning

Continuous latent reasoning replaces explicit intermediate tokens with recurrent continuous states, reducing autoregressive decoding while retaining multi-step computation. Existing methods recur rently feed hidden states back into the model, distill textual reasoning chains into latent trajectories, or construct continuous thoughts in vocabulary space (Hao et al., 2025; Shen et al., 2025; Zhang et al., 2025b; Deng et al., 2025). These methods have primarily been studied on mathematical and general reasoning tasks. We adapt vocabulary-space latent reasoning to reward modeling, where the latent trajectory must represent a multi-criterion comparison between candidate responses.

## 2.2 Reward Modeling

Generative reward models formulate evaluation as language generation and can produce critiques or reasoning traces before issuing a judgment (Mahan et al., 2024; Ankner et al., 2024; Chen et al., 2026; Guo et al., 2025). In recent years, rubric-based reward modeling has rapidly gained attention as a way to decompose holistic judgments into explicit, task-specific criteria for evaluation, feedback, reward construction, and policy optimization (Liu et al., 2026; Yuan et al., 2026). However, such rubric-based and reasoning-heavy judges introduce substantial inference costs, since they must generate and apply criteria explicitly at test time. Our work therefore focuses on the inference cost of reasoning-based generative judges: LatentGRM compresses the evaluations into continuous trajectories, with matched explicit SFT judges as the primary comparison. We further repurpose rubrics as anchors for semantic chunking and as an evaluation and analysis tool for latent reasoning.

## 3 Method

## 3.1 Overview

We study pairwise reward modeling with an evaluation input $\mathit { x } ~ = ~ ( q , \mathcal { R } , a , b )$ , comprising a user request, a task-specific rubric, and two candidate responses. Training examples additionally provide an explicit evaluation trace $\begin{array}{c} \begin{array} { l } { c } { \end{array} = \begin{array} { l } { \left( c _ { 1 } , \ldots , c _ { M } \right) } \end{array} } \end{array}$ and a preference label $y \in$ {Response A, Response B}. An explicit reasoning judge produces:

$$
[ x , < \mathrm { t h i n k } > , c _ { 1 } , \ldots , c _ { M } , < / \mathrm { t h i n k } > , y ] .\tag{1}
$$

LatentGRM replaces the textual evaluation with fewer continuous steps:

$$
[ x , < \mathrm { t h i n k } > , z _ { 1 } , \ldots , z _ { N } , < / \mathrm { t h i n k } > , y ] .\tag{2}
$$

We build on Latent-SFT’s vocabulary-space representation and two-stage distillation framework (Deng et al., 2025), replacing fixed-rate segmentation with Semantic Chunking. Stage 1 distills explicit evaluation traces into latent trajectories. Stage 2 trains the judge to autoregressively generate these trajectories and predict the final preference from the evaluation input at inference. A separate interpreter reconstructs explicit evaluations from saved trajectories for offline analysis. Figure 2 summarizes the framework.

![](images/4d24e1ed5531920e1d1f1df04b883ecb3c7a77257caef0ce9bbe47d655ad70ac.jpg)  
Figure 2: LatentGRM training and inference. Stage 1 encodes semantic boundaries into vocabulary-space targets and trains suffix reconstruction; Stage 2 distills autonomous latent judging. An offline interpreter reconstructs evaluations from saved trajectories.

## 3.2 Stage 1: Semantic Chunking for Latent Compression

Stage 1 follows an encoder and decoder objective. At each semantic boundary, the encoder compresses the evaluation input and teacher-trace prefix into latent states, while the decoder learns to recover the remaining evaluation and final preference from the latent prefix.

Let $c _ { 1 : M }$ denote an evaluation trace containing M tokens. Semantic Chunking divides it into N spans using ordered boundaries $0 = b _ { 0 } < b _ { 1 } < \dots < b _ { N } = M$ . We set:

$$
N = \left\lceil { \frac { M } { \rho } } \right\rceil ,\tag{3}
$$

where $\rho$ is the fixed compression rate. Rubric-guided evaluations naturally separate into compliance check, criterion-specific analysis, and final synthesis (Figure 8). We fix the latent budget of each rubric block with an exact-budget allocator, then refine its internal frontiers using sentence, clause, and dependency evidence. Semantic Chunking places compression boundaries to preserve semantic integrity as much as possible.

Figure 3 illustrates how the two boundary choices treat the same rubric and sentence-level passages.

At boundary $b _ { i }$ , the encoder reads x and the explicit prefix $c _ { 1 : b _ { i } }$ and produces a hidden state $h _ { i }$ . To make this state directly consumable by the decoder, we express it through the decoder’s vocabulary interface. Let $W _ { \mathrm { o u t } }$ be the decoder’s frozen vocabulary output projection and $\tau$ the projection temperature. The resulting full-vocabulary distribution is $\alpha _ { i } = \mathrm { s o f t m a x } ( W _ { \mathrm { o u t } } h _ { i } / \tau )$ . We define $S _ { i } = \mathrm { T o p K } ( \alpha _ { i } )$ as the indices of the K largest entries of $\alpha _ { i }$ and renormalize their probability mass, so $\bar { \alpha } _ { i , v } = { \alpha _ { i , v } } / { \sum _ { u \in S _ { i } } { \alpha _ { i , u } } }$ for $v \in S _ { i }$ , and $\bar { \alpha } _ { i , v } = 0$ otherwise. Using the decoder’s frozen input embedding table $E _ { \mathrm { i n } }$ , we form the compressed state:

$$
z _ { i } = \sum _ { v \in S _ { i } } \bar { \alpha } _ { i , v } E _ { \mathrm { i n } } [ v ] .\tag{4}
$$

![](images/6aa212cd0237ffd84e1acbb3216a39132368449c20a1c041b4ce43b4bf33927f.jpg)  
Figure 3: Illustrative frontiers in cropped evaluations. Green and red mark the semantic and fixed-rate cuts on identical text.

Thus $z _ { i }$ is a continuous mixture of token embeddings and can be inserted directly into the decoder’s input sequence. Given $[ x , < \pm \mathrm { h i n k } > , z _ { 1 : i } ]$ , the decoder predicts the remaining evaluation and final preference:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { C E } \left( \left[ c _ { b _ { i } + 1 : M } , < / \mathrm { t h i n k } > , y , \mathrm { E O S } \right] \mid [ x , < \mathrm { t h i n k } > , z _ { 1 : i } ] \right) .\tag{5}
$$

Here, CE is token-averaged autoregressive cross-entropy. We optimize the encoder, the decoder, and then both jointly. After training, the encoder exports $( S _ { i } , \bar { \alpha } _ { i } ) _ { i = 1 } ^ { N }$ as sparse targets for Stage 2.

## 3.3 Stage 2: Autonomous Latent Evaluation

Starting from the Stage 1 decoder, Stage 2 trains the judge to predict the exported distributions autoregressively and then output the final preference. Let $\mathbf { g } = ( \mathbf { g } _ { 1 } , \ldots , \mathbf { g } _ { N } )$ collect independent Gumbel noise across the trajectory. For each target $\bar { \alpha } _ { i }$ , we add g to its log-probabilities and renormalize within $S _ { i } ,$ , obtaining $\widetilde { \alpha } _ { i } ( \mathbf { g } )$ . Replacing $\bar { \alpha } _ { i }$ with these perturbed weights in Eq. 4 gives the corresponding target embedding $\widetilde { z } _ { i } ( \mathbf { g } )$ . During training, the prediction at step i is conditioned on the preceding perturbed target embeddings $\widetilde { z } _ { 1 : i - 1 } ( \mathbf { g } )$ . This stochastic perturbation discourages overfitting to deterministic teacher targets; at inference, the same mechanism produces diverse latent trajectories for voting. Let $x _ { \mathrm { l a t } } = [ x , < \mathrm { t h i n k } > ]$ and let $q _ { \theta }$ denote the judge’s predicted full-vocabulary distribution. We optimize:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a u t o } } = \mathbb { E } _ { \mathbf { g } } \left[ \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \mathrm { K L } } ( \widetilde { \alpha } _ { i } ( \mathbf { g } ) \parallel q _ { \theta } ( \cdot  { | } x _ { \mathrm { l a t } } , \widetilde { z } _ { < i } ( \mathbf { g } ) ) ) \right. } \\ { \left. + \mathrm { C E } ( [ < / \mathrm { t h i n k } > , y , \mathrm { E O S } ] \mid x _ { \mathrm { l a t } } , \widetilde { z } _ { 1 : N } ( \mathbf { g } ) ) \right] . } \end{array}\tag{6}
$$

The KL term distills each perturbed latent target, while cross-entropy predicts the end marker, preference label, and EOS from the complete trajectory. The expectation is estimated with one noise draw per example at each update.

At inference, the judge feeds back its own predictions. At step i, it produces $q _ { i } ~ = ~ q _ { \theta } ( \cdot ~ |$ $x , < \pm \mathrm { h i n k } > , z < i )$ . If the unperturbed argmax of $q _ { i }$ is $< / \mathrm { t h i n k } >$ , latent generation terminates and the preference is decoded greedily. Otherwise, the judge retains and renormalizes the top-K probabilities of $q _ { i } ,$ , applies the same Gumbel perturbation, and substitutes the resulting support and weights into Eq. 4 to form $z _ { i }$ . We extend vLLM (Kwon et al., 2023) to support this soft-embedding feedback while retaining continuous batching, KV caching, and tensor parallelism.

## 3.4 Retrospective Interpretation of Latent Trajectories

The Latent Trace Interpreter (LTI) is a separate readout that reconstructs an explicit evaluation from a completed latent trajectory. At position i, it receives saved top-K vocabulary indices $S _ { i } ^ { \mathrm { s a v e } }$ and normalized weights $\vec { q } _ { i } ^ { \mathrm { s a v e } }$ . Using its embedding table $E _ { \mathrm { i n t } }$ , LTI forms:

$$
z _ { i } ^ { \mathrm { i n t } } = \sum _ { v \in S _ { i } ^ { \mathrm { s a v e } } } \bar { q } _ { i } ^ { \mathrm { s a v e } } ( v ) { E } _ { \mathrm { i n t } } [ v ] ,\tag{7}
$$

the same vocabulary-mixture construction as Eq. 4. Given the full sequence $z _ { 1 : N } ^ { \mathrm { i n t } }$ , LTI predicts the complete teacher evaluation with the token-averaged objective:

$$
\mathcal { L } _ { \mathrm { i n t } } = \mathrm { C E } \big ( [ c _ { 1 : M } , < / \mathrm { t h i n k } > ] ~ | ~ [ x , < \mathrm { t h i n k } > , z _ { 1 : N } ^ { \mathrm { i n t } } ] \big ) .\tag{8}
$$

Unlike Stage 1’s suffix reconstruction, LTI reconstructs the evaluation from its beginning; the preference label y is excluded. Autonomous judging can save its predicted top-K indices and weights for the same offline readout, without running LTI in the online decision path.

## 4 Experiments

Our experiments address three questions: (RQ1) whether latent evaluation preserves preference accuracy across model scales and responds to controlled changes in rubric criteria; (RQ2) how compressing explicit evaluations into latent trajectories affects inference cost; and (RQ3) whether criterion-level assessments can be reconstructed from latent trajectories.

## 4.1 Experimental Setup

Training data. We use 35,612 OpenRubrics (Liu et al., 2026) training records, each containing a request, two responses, a rubric, an evaluation trace, and a binary preference. OpenRubrics keeps generated rubrics whose judgments match the preference labels.

Evaluation data. We evaluate eight benchmark domains: RewardBench Chat and Chat Hard (Lambert et al., 2025), PPE-IFEval (Frick et al., 2025), IFBench (Pyatkin et al., 2025), RM-Bench Chat (Liu et al., 2025), RewardBench 2 Precise IF and Focus (Malik et al., 2026), and Help-Steer3 (Wang et al., 2025). Avg.-8 is their unweighted mean.

Models and evaluation. Our primary baselines are explicit Rubric-RM trained by SFT on the same OpenRubrics records and Qwen3-4B or Qwen3-8B backbones (Yang et al., 2025) as LatentGRM. This matched comparison isolates the effect of replacing explicit evaluation traces with compact latent trajectories. Both methods receive identical task-specific rubrics, generated once offline by OpenRubrics generator-8B. We report a single rollout (vote@1) and majority voting over five rollouts (vote@5). JudgeLRM, RRM, and two RM-R1 variants serve as additional reference points (Chen et al., 2025; Guo et al., 2025; Chen et al., 2026). We evaluate every model in both response orders and average the resulting scores.

## 4.2 RQ1: Effectiveness and Rubric-Based Process Evaluation

## 4.2.1 Preference Accuracy

At vote@5, LatentGRM scores 67.8 and 69.0 on Avg.-8 at 4B and 8B, compared with 67.0 and 68.1 for the matched explicit Rubric-RM (Table 1). Its largest advantage is on Focus (+6.0 and +6.4 points); across the other seven domains, Chat Hard and IFBench favor LatentGRM at both scales, RewardBench Chat favors Rubric-RM, and Precise IF changes direction between 4B and 8B. PPE, RM-Bench Chat, and HelpSteer3 differ by at most 0.4 points. Thus, compact latent evaluation retains broadly similar preference accuracy across domains, with a concentrated gain on Focus and Chat Hard. Voting raises LatentGRM’s Avg.-8 from 65.6 to 67.8 at 4B (+2.2) and from 66.7 to 69.0 at 8B (+2.3), showing that multiple compact trajectories can improve the final judgment.

Table 1: Pairwise preference evaluation (%). Avg.-8 is the mean of the eight benchmark scores. Shading mark the best score among the Rubric-RM and LatentGRM configurations.
<table><tr><td rowspan="2">Model</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">RewardBench IF Evaluation RM-Bench RewardBench2 HS3 Avg.-8</td><td colspan="2"></td><td rowspan="2"></td></tr><tr><td>Chat</td><td>Hard</td><td>PPE IFBench</td><td></td><td>Chat</td><td>Prec. IF Focus</td><td></td><td></td></tr><tr><td>White-box judge/reward LLMs (for reference only)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JudgeLRM-7B</td><td>93.2</td><td>56.4</td><td>47.9</td><td>47.5</td><td>55.0</td><td>29.7</td><td>39.9</td><td>60.5</td><td>53.7</td></tr><tr><td>RRM-7B</td><td>88.0</td><td>70.6</td><td>52.5</td><td>55.2</td><td>59.8</td><td>29.4</td><td>63.9</td><td>63.7</td><td>60.4</td></tr><tr><td>RM-R1-7B (Qwen-2.5-Inst)</td><td>94.4</td><td>72.3</td><td>56.1</td><td>57.8</td><td>65.8</td><td>35.6</td><td>80.0</td><td>69.3</td><td>66.4</td></tr><tr><td>RM-R1-7B (DeepSeek-Dist)</td><td>86.2</td><td>67.0</td><td>52.5</td><td>56.8</td><td>62.1</td><td>23.8</td><td>56.9</td><td>63.3</td><td>58.6</td></tr><tr><td>Rubric-RM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rubric-RM-4B, vote@1</td><td>89.1</td><td>67.8</td><td>60.2</td><td>61.7</td><td>61.6</td><td>32.2</td><td>77.0</td><td>66.8</td><td>64.5</td></tr><tr><td>Rubric-RM-4B, vote@5</td><td>91.1</td><td>69.3</td><td>61.9</td><td>62.5</td><td>61.7</td><td>40.6</td><td>80.5</td><td>68.7</td><td>67.0</td></tr><tr><td>Rubric-RM-8B, vote@ 1</td><td>89.8</td><td>69.8</td><td>61.2</td><td>59.9</td><td>61.3</td><td>31.9</td><td>77.9</td><td>68.1</td><td>65.0</td></tr><tr><td>Rubric-RM-8B, vote@5</td><td>91.8</td><td>70.5</td><td>62.9</td><td>61.5</td><td>62.1</td><td>45.0</td><td>81.1</td><td>70.0</td><td>68.1</td></tr><tr><td>LatentGRM (ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LatentGRM-4B, vote@ 1</td><td>87.2</td><td>68.4</td><td>61.2</td><td>61.0</td><td>59.9</td><td>34.1</td><td>85.4</td><td>67.9</td><td>65.6</td></tr><tr><td>LatentGRM-4B, vote@5</td><td>88.8</td><td>70.3</td><td>61.9</td><td>63.2</td><td>62.0</td><td>41.6</td><td>86.5</td><td>68.6</td><td>67.8</td></tr><tr><td>LatentGRM-8B, vote@ 1</td><td>88.7</td><td>70.5</td><td>60.4</td><td>63.0</td><td>59.8</td><td>37.9</td><td>84.6</td><td>68.2</td><td>66.7</td></tr><tr><td>LatentGRM-8B, vote@5</td><td>89.9</td><td>73.0</td><td>63.1</td><td>63.3</td><td>61.9</td><td>43.4</td><td>87.5</td><td>69.6</td><td>69.0</td></tr></table>

Appendix E.4 provides a comparison with a direct judge that predicts only the final preference.

## 4.2.2 Effect of Semantic Chunking

To assess the contribution of Semantic Chunking, we compare it with Fixed-Rate Chunking on Qwen3-8B. Both use the same training data, target compression rate $\rho = 8$ , latent representation, initialization, and training schedule.

Semantic Chunking improves Chat Hard, Precise IF, and Focus at both voting budgets (Table 2). At vote@5, the gains over Fixed-Rate Chunking are 2.2, 3.9, and 1.7 points, respectively, while Chat decreases by 0.7 points; the four-domain mean rises by 1.8 points. With the latent budget and training conditions held fixed, this pattern indicates that placing boundaries at semantic units improves compressed judgments most clearly on instruction-sensitive tasks.

## 4.2.3 Rubric Interventions and Latent-Source Replacement

We test whether changing one evaluation criterion shifts the model’s preference and whether the shift is carried by the latent trajectory, motivated by latent-state intervention studies (Li et al., 2026). With all other criteria fixed, the test measures the direction of the preference shift.

Design and readout. For each of 398 RewardBench pairs (198 Chat and 200 Chat Hard), we construct three rubrics that differ in one criterion: the baseline rubric $\mathcal { R } _ { 0 }$ is unchanged; the wording control ${ \mathcal R } _ { p }$ paraphrases the selected criterion without changing its meaning; and the criterion intervention $\mathcal { R } _ { c }$ changes that criterion so that it favors a designated response $b _ { i } .$ . GLM-5.2 (GLM-5 Team et al., 2026) proposes the two edits, and a fresh-context call verifies that ${ \mathcal R } _ { p }$ preserves meaning and $\mathcal { R } _ { c }$ changes only the selected criterion, then identifies $b _ { i }$ . Neither call receives benchmark labels or judge predictions. We generate trajectories for all three rubrics in both response orders.

Table 2: Semantic Chunking ablation on Qwen3-8B (%).
<table><tr><td rowspan="2">Method</td><td colspan="2">RewardBench</td><td colspan="2">RewardBench2</td></tr><tr><td>Chat</td><td>Hard</td><td>Prec. IF</td><td>Focus</td></tr><tr><td>vote@1</td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed-Rate Chunking</td><td>90.5</td><td>69.0</td><td>37.2</td><td>83.3</td></tr><tr><td>Semantic Chunking</td><td>88.7</td><td>70.5</td><td>37.9</td><td>84.6</td></tr><tr><td>vote@5</td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed-Rate Chunking</td><td>90.6</td><td>70.8</td><td>39.5</td><td>85.8</td></tr><tr><td>Semantic Chunking</td><td>89.9</td><td>73.0</td><td>43.4</td><td>87.5</td></tr></table>

A Criterion edit  
![](images/624db158b8816e66a5df9d280138250467d4fa2bd150e29342c87a59ffa7d43b.jpg)

B Latent-source replacement  
![](images/5f9ef0565e755eb591e83ca7d09a51e779a5fb867e73576ff56502cf0b33e860.jpg)  
Figure 4: Rubric-conditioned preference shifts on 398 pairs. (A) Each trajectory is generated and evaluated under the same rubric. (B) The evaluation prompt is fixed to the baseline rubric while the latent-trajectory source varies. Changes are measured against the baseline condition; positive values favor the response targeted by the criterion intervention. Bands show pointwise 95% question-cluster bootstrap intervals.

At 0%, 10%, . . ., 100% of each trajectory, we probe the preference between the two response labels. For a context $C$ containing the evaluation prompt and latent prefix, we measure the log-odds of the label $\ell _ { b _ { i } }$ against the other label $\ell _ { \bar { b } _ { i } }$ :

$$
\begin{array} { r } { D _ { i } ( C ) = \log p _ { \theta } ( \ell _ { b _ { i } } \mid C ) - \log p _ { \theta } ( \ell _ { \bar { b } _ { i } } \mid C ) . } \end{array}\tag{9}
$$

For either edited condition $e \in \{ p , c \} , \Delta D _ { i } = D _ { i } ( C _ { e } ) - D _ { i } ( C _ { 0 } )$ measures its effect relative to the baseline; positive values indicate a shift toward $b _ { i }$ . This contrastive readout follows Wang et al. (2023); Heimersheim & Nanda (2024).

End-to-end rubric intervention. The criterion intervention shifts preference toward the designated response, reaching a mean log-odds change of 4.58 at the trajectory endpoint (Figure 4A). Its effect becomes substantially larger than that of the wording control in the latter half of the trajectory, indicating sensitivity to the criterion’s meaning rather than its phrasing alone. The nonzero shift at 0% shows a direct effect of the rubric prompt; the replay control below measures the contribution carried by the latent sequence.

Latent-source replacement. To remove the visible rubric difference, we fix the evaluation prompt to $\mathcal { R } _ { 0 }$ and replay the latent embeddings generated under each rubric in a fresh baseline context. The criterion-intervention source has little effect at early positions, but its influence grows with the supplied trajectory and reaches a mean log-odds change of 2.75 at the endpoint (Figure 4B); the wording-control source remains much smaller. Thus, criterion-dependent preference information transfers through the latent sequence even when the evaluation prompt remains unchanged.

## 4.3 RQ2: Practical Efficiency of Latent Evaluation

We evaluate the cost of producing preference judgments. Retrospective interpretation is performed separately and is excluded from these measurements.

## 4.3.1 Runtime under Natural Generation

We compare Rubric-RM-8B and LatentGRM-8B on four domains at vote@5. The vLLM measurements use the same eight-GPU serving configuration and include prefill, scheduling, decoding, and output collection. For reference, we also report full-domain time under ordinary Transformers/Py-Torch generation (Wolf et al., 2020; Paszke et al., 2019) on the same eight GPUs, with one model replica per GPU and five sequential rollouts per assigned sample.

Table 3: Inference cost at vote@5. Steps are mean generated positions per rollout. Compression and speedup are the Rubric-RM-8B/LatentGRM-8B ratios of generated steps and total inference time. HF denotes ordinary Transformers/PyTorch generation without vLLM. Bold marks fewer steps or less time and the corresponding compression/speedup ratios within each domain.
<table><tr><td>Domain (N)</td><td>Model</td><td>Steps</td><td>Compression</td><td>HF time (s)</td><td>HF speedup</td><td>vLLM time (s)</td><td>vLLM speedup</td></tr><tr><td rowspan="2">RB Chat (716)</td><td>Rubric-RM-8B</td><td>659.60</td><td>1.00×</td><td>8196</td><td>1.00×</td><td>539.70</td><td>1.00×</td></tr><tr><td>LatentGRM-8B</td><td>74.38</td><td>8.87×</td><td>1002</td><td>8.17×</td><td>87.76</td><td>6.15×</td></tr><tr><td rowspan="2">RB Chat Hard (912)</td><td>Rubric-RM-8B</td><td>582.09</td><td>1.00×</td><td>11088</td><td>1.00×</td><td>589.58</td><td>1.00×</td></tr><tr><td>LatentGRM-8B</td><td>64.42</td><td>9.04×</td><td>1308</td><td>8.46×</td><td>96.44</td><td>6.11×</td></tr><tr><td rowspan="2">RB2 Precise IF (960)</td><td>Rubric-RM-8B</td><td>720.89</td><td>1.00×</td><td>13416</td><td>1.00×</td><td>812.93</td><td>1.00×</td></tr><tr><td>LatentGRM-8B</td><td>78.59</td><td>9.17×</td><td>1524</td><td>8.82×</td><td>126.99</td><td>6.40×</td></tr><tr><td rowspan="2">RB2 Focus (2970)</td><td>Rubric-RM-8B</td><td>695.49</td><td>1.00×</td><td>37572</td><td>1.00×</td><td>2350.30</td><td>1.00×</td></tr><tr><td>LatentGRM-8B</td><td>75.37</td><td>9.23×</td><td>4476</td><td>8.39×</td><td>338.34</td><td>6.95×</td></tr></table>

At vote@5, LatentGRM-8B reduces generated positions by 8.87–9.23×, yielding 8.17–8.82× speedups under ordinary Transformers and 6.11–6.95× under vLLM (Table 3). All four domains exceed 6× vLLM speedup despite mixed accuracy differences.

Figure 1 extends the comparison across vote@1–9. LatentGRM remains faster and scores higher on Chat Hard and Focus at every voting budget; Chat remains below Rubric-RM, and Precise IF varies by budget. More votes generally improve latent judgment quality, with nonmonotonic intermediate results and 6.9×–7.8× speedups at vote@9.

An equal-length control that forces both judges to generate exactly 50 positions yields only a 1.017× runtime difference, showing that latent reasoning does not introduce additional per-step computa tional cost. (Appendix F.2).

## 4.4 RQ3: Recovering Rubric-Level Evaluations

Task rubrics provide predefined semantic units for evaluating reconstruction fidelity. We align decoded and teacher evaluations by response and criterion, measuring whether each assessment’s state and justification are preserved. This makes changes in individual judgments observable even when most of the reconstructed text remains similar.

Using cached teacher-derived trajectories, we compare three independently trained readouts initialized from the LatentGRM-8B Stage 1 decoder. Prompt-only receives the evaluation input without latent states; Replay-hidden additionally reads projected hidden states obtained by replaying the latent sequence; Latent Trace Interpreter (LTI) reads its vocabulary-probability representation (Section 3.4).

The test set contains 607 records with 9,992 criteria, of which 9,768 have explicit states. State Macro-F1 scores MET, NOT MET, and PARTIAL; missing or unparseable states count as errors. Vector Exact requires all states in a fully labeled record to match. Rubric Semantic measures BERTScore similarity (Zhang\* et al., 2020) between aligned justifications, assigning zero to missing criteria; whole-trace ROUGE-L (Lin, 2004) provides a lexical comparison.

Table 4: LatentGRM-8B retrospective reconstruction on the 607-record criterion-rich test set (%).
<table><tr><td>Readout input</td><td>State Macro-F1</td><td></td><td>Vector Exact Rubric Semantic</td><td>ROUGE-L</td></tr><tr><td>Prompt-only</td><td>40.20</td><td>18.61</td><td>56.42</td><td>63.36</td></tr><tr><td>Replay-hidden</td><td>69.42</td><td>33.03</td><td>61.29</td><td>68.22</td></tr><tr><td>Latent Trace Interpreter</td><td>98.93</td><td>93.98</td><td>74.65</td><td>79.24</td></tr></table>

LTI raises State Macro-F1 from 40.20 (Prompt-only) and 69.42 (Replay-hidden) to 98.93, and Vector Exact from 18.61 and 33.03 to 93.98 (Table 4). Prompt-only reaches 63.36 ROUGE-L despite its 40.20 State Macro-F1, showing that lexical overlap misses criterion-state errors. LTI also leads on Rubric Semantic (74.65) and ROUGE-L (79.24), reflecting stronger criterion-level recovery on teacher-derived trajectories.

Downstream utility through data revision. We also test whether the recovered evaluation can guide a frozen Qwen3-8B Base reviser on 240 RewardBench Rubric examples. Prompt-only gives the reviser just the question and candidate response; Rubric-RM trajectory supplies explicit textual feedback; Latent only inserts the soft top-10 embedding mixture at each latent step without training the reviser to consume it; and Interpreter-restored trajectory supplies LTI’s textual reconstruction of the same latent feedback. The reviser does not see the rubric. Appendix E.5 gives the input construction and scoring protocol.

An external evaluator scores each rubric criterion before and after revision. Repair is the fraction of originally unsatisfied criteria that become satisfied, whereas Regression is the fraction of originally satisfied criteria that become unsatisfied. Full is the percentage of revised answers satisfying every criterion; Criterion satisfaction is the overall percentage of satisfied criterion decisions.

Table 5: Frozen-model revision on 240 examples, equally split between Chat and Chat Hard (percent). Repair is computed over 897 originally unsatisfied criteria and Regression over 1,076 originally satisfied criteria. Bold marks the best result in each column.
<table><tr><td>Feedback interface</td><td>Repair ↑</td><td>Regression↓</td><td>Full ↑</td><td>Criterion sat. ↑</td></tr><tr><td>Prompt-only</td><td>54.07</td><td>5.20</td><td>34.17</td><td>76.28</td></tr><tr><td>Rubric-RM trajectory</td><td>78.71</td><td>4.65</td><td>56.67</td><td>87.78</td></tr><tr><td>Latent only (soft top-10)</td><td>63.32</td><td>5.11</td><td>41.67</td><td>80.54</td></tr><tr><td>Interpreter-restored trajectory</td><td>80.49</td><td>1.86</td><td>57.50</td><td>90.12</td></tr></table>

Prompt-only revision reaches 34.17% Full. Explicit Rubric-RM feedback raises this to 56.67% and Repair from 54.07% to 78.71% (Table 5). Even without reviser adaptation, the latent-only interface improves Repair by 9.25 points and Full by 7.50 points over Prompt-only. Interpreter-restored feedback yields the best result in every reported column: 80.49% Repair, 1.86% Regression, 57.50% Full, and 90.12% criterion satisfaction. Relative to explicit Rubric-RM feedback, its Full rate is comparable while Regression is lower (1.86% versus 4.65%).

These results show that latent reasoning preserves the data-synthesis and revision utility of explicit reasoning: after textual restoration, it matches or exceeds the explicit trajectory on every reported metric. More strikingly, the training-free Latent-only interface already improves substantially over

Prompt-only revision, demonstrating that latent feedback is directly usable before any specialized adaptation.

## 5 Conclusion

We introduced LatentGRM, a framework for latent reward modeling built on semantic chunking, compression, and reconstruction. Rubric structure guides the compression of explicit evaluations into compact continuous trajectories, which the judge uses to predict pairwise preferences without generating evaluation text. Compared with explicit SFT judges trained on matched data and backbones, LatentGRM achieves competitive aggregate accuracy while substantially reducing trajectory length and inference time. Controlled rubric interventions show that criterion-dependent preference information is carried through the latent sequence, while the Latent Trace Interpreter reconstructs criterion-level assessments for retrospective inspection. Together, these results show how structured latent trajectories can support efficient judgment and make compressed evaluations inspectable.

## References

Zachary Ankner, Mansheej Paul, Brandon Cui, Jonathan D Chang, and Prithviraj Ammanabrolu. Critique-out-loud reward models. arXiv preprint arXiv:2408.11791, 2024. URL https:// arxiv.org/abs/2408.11791.

Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. Judgelrm: Large reasoning models as a judge. arXiv preprint arXiv:2504.00050, 2025. URL https://arxiv.org/abs/2504.00050.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru WANG, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, and Heng Ji. Rm-r1: Reward modeling as reasoning. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 88313–88342, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 8e3b8de251afd887fb4589c1e3a3c793-Paper-Conference.pdf.

Jingcheng Deng, Liang Pang, Zihao Wei, Shicheng Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng. Llm latent reasoning as chain of superposition. arXiv preprint arXiv:2510.15522, 2025. URL https://arxiv.org/abs/2510.15522.

Jingcheng Deng, Zihao Wei, Liang Pang, Junhong Wu, Shicheng Xu, Zenghao Duan, and Huawei Shen. Latent-grpo: Group relative policy optimization for latent reasoning. arXiv preprint arXiv:2604.27998, 2026. URL https://arxiv.org/abs/2604.27998.

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. How to evaluate reward models for rlhf. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 18128–18163, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 2e01083b381b4865919b4915ef32e3d2-Paper-Conference.pdf.

GLM-5 Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, et al. GLM-5: From vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/ 2602.15763.

Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Reward reasoning models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz,

M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 150477–150510. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-5031. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/dd35bb9efff094897fb6688a57675212-Paper-Conference.pdf.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason E Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= Itxz7S4Ip3.

Stefan Heimersheim and Neel Nanda. How to use and interpret activation patching. arXiv preprint arXiv:2404.15255, 2024. URL https://arxiv.org/abs/2404.15255.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, pp. 611–626, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702297. doi: 10.1145/3600006.3613165. URL https: //doi.org/10.1145/3600006.3613165.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. RewardBench: Evaluating reward models for language modeling. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Findings of the Association for Computational Linguistics: NAACL 2025, pp. 1755–1797, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl.96. URL https://aclanthology.org/2025.findings-naacl.96/.

Zirui Li, Xuefeng Bai, Kehai Chen, Yizhi LI, Jian Yang, Chenghua Lin, and Min Zhang. Dynamics within latent chain-of-thought: An empirical study of causal structure. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum? id=kHB8m3ojGe.

Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. OpenRubrics: Towards scalable synthetic rubric generation for reward modeling and LLM alignment. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 17417–17437, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.791. URL https: //aclanthology.org/2026.acl-long.791/.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. Rm-bench: Benchmarking reward models of language models with subtlety and style. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 44323– 44355, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ file/6da1eec80095dc5937f7716db15aca4b-Paper-Conference.pdf.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Franken, Chelsea Finn, and Alon Albalak. Generative reward models.¨ arXiv preprint arXiv:2410.12832, 2024. URL https://arxiv.org/abs/2410.12832.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah Smith, Hanna Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 144839–144866, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ ea4fe0a56d02c93401902b5b4c6b12da-Paper-Conference.pdf.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-2011. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alche-Buc, E. Fox, and R. Garnett (eds.), ´ Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hanna Hajishirzi. Generalizing verifiable instruction following. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025. doi: 10.52202/085713-1645. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 46499a0622ecf568b72d17b61e45dbd5-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: Compressing chain-of-thought into continuous space via self-distillation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 677–693, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/ 2025.emnlp-main.36. URL https://aclanthology.org/2025.emnlp-main.36/.

Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=NpsVSN6o4ul.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025. doi: 10.52202/085713-1448.

URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 3e0271cf7df2cdb3b91565ad1f525f3a-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Qun Liu and David Schlangen (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.emnlp-demos.6. URL https://aclanthology.org/2020.emnlp-demos.6/.

Anyi Xu, B Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, CC Yu, et al. Deepseek-v4. 1-flash: Pushing the limits of kv cache compression. arXiv preprint arXiv:2609.19969, 2026. URL https://arxiv.org/abs/2609. 19969.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Mingqing Yuan, Xiaobo Liang, Qipeng Huang, Zixuan Cai, Wanfu Wang, Yang Qiao, Pu Lu, Caishuang Huang, Meng Zhou, Lijun Wu, Juntao Li, and Min Zhang. Seeing the forest and the trees: A survey of analytic rubrics for holistic reward modeling in llms. Preprints, 2026. doi: 10.20944/preprints202605.1624.v1. URL https://www.preprints.org/ manuscript/202605.1624.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Seyed Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 12476–12505, 2025a. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ file/214308a2d5e3f83ef9ad2739e1cbc46d-Paper-Conference.pdf.

Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SkeHuCVFDr.

Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, and Xin Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 168990–169012. Curran Associates, Inc., 2025b. doi: 10.52202/ 085713-5629. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/f7396d1c54d51416958d63e285377103-Paper-Conference.pdf.

## Appendix Contents

1. Discussion

2. Semantic Chunking and Exact-Budget Allocation

3. Stage 1 Latent Distillation and Stage 2 Autonomous Evaluation

4. Latent Trace Interpreter Details

5. Supplementary RQ1: Effectiveness and Rubric-Based Process Evaluation

• Data, backbones, and evaluation protocol

• Evaluation-time parameters and Semantic Chunking training dynamics

• Rubric-based intervention protocol

• Direct judgment without an intermediate evaluation trace

• Downstream utility through frozen-model data revision

6. Supplementary RQ2: Practical Efficiency of Latent Evaluation

• vLLM implementation for latent reasoning

• Equal-length runtime control

7. Supplementary RQ3: Recovering Rubric-Level Evaluations

• Readout data, controls, and evaluation protocol

• Metrics, parsing, and reconstruction examples

8. Training Setup and Compute

9. Prompt Templates

## A Discussion

An important challenge in latent reward modeling is preserving essential evaluation evidence under compression. Fixed-Rate Chunking and the current heuristic Semantic Chunking approximate evaluation structure through fixed token intervals and linguistic boundary cues, respectively. Developing compression strategies that better preserve the evidence most relevant to the final judgment remains an important direction.

A related challenge is recovering the compressed evidence through faithful retrospective interpretation. Appendix G.4.1 illustrates a coherent, nearly complete reconstruction that changes one criterion assessment. The interpreter receives renormalized top-K soft labels, which preserve only a subset of the vocabulary distribution. Richer trajectory representations and reconstruction objectives could strengthen the connection between recovered criterion states and their supporting evidence.

These challenges accompany substantial opportunities. By shortening generation and reducing inference time, latent reasoning can make repeated reward evaluations more affordable in policy learning, data selection, and related workflows. The gains from vote@1 to vote@5 also motivate exploring reinforcement learning to improve individual latent trajectories, building on approaches such as Latent-GRPO (Deng et al., 2026). Together, advances in compression, trajectory optimization, and interpretation could extend the value of latent reward modeling from faster evaluation to more capable and understandable learning systems.

## B Semantic Chunking and Exact-Budget Allocation

Semantic Chunking first divides an evaluation into outer rubric blocks and allocates a fixed global latent budget across them. It then refines each block’s internal boundaries while holding its allocated count fixed. The procedure preserves the original token sequence and produces exactly $N = \lceil M / 8 \rceil$ latent positions. Preprocessing is performed offline and cached. Figures 5 and 6 illustrate Semantic Chunking and Fixed-Rate Chunking.

## B.1 Rubric Structure and Initial Budget Allocation

We tokenize the evaluation once with character offsets. The outer blocks comprise the initial compliance and gatekeeper discussion, each response’s individual criterion assessments, and the final synthesis; each response header joins its first criterion. Structural expressions of at most 10 tokens are protected against internal cuts, including section and response headers, criterion identifiers, [Hard Rule], [Principle], Justification:, Met., and Not Met. We use boundary for a selected token position in the partition; each selected boundary serves as a compression frontier in Stage 1.

Let $m _ { s }$ be the token length of outer block s, for $s = 1 , \ldots , S$ , and let $n _ { s }$ be its latent budget. We require

$$
\sum _ { s = 1 } ^ { S } n _ { s } = N , \qquad \left\lceil \frac { m _ { s } } { 1 6 } \right\rceil \leq n _ { s } \leq \left\lfloor \frac { m _ { s } } { 4 } \right\rfloor , \qquad N = \left\lceil \frac { \sum _ { s } m _ { s } } { 8 } \right\rceil .\tag{10}
$$

For a candidate count k, let $d _ { s , 1 : k }$ be the balanced integer partition of $m _ { s }$ , with lengths differing by at most one. Define $\delta ( d ) = \operatorname* { m a x } ( 6 - d , 0 , d - 1 0 )$ . The allocation cost is

$$
A _ { s } ( k ) = \left( \sum _ { j } \mathbf { 1 } [ \delta ( d _ { s , j } ) > 0 ] , \quad \sum _ { j } \delta ( d _ { s , j } ) , \quad \sum _ { j } ( d _ { s , j } - 8 ) ^ { 2 } \right) .\tag{11}
$$

We minimize the sum of these costs lexicographically using

$$
D ( s , n ) = \operatorname* { m i n } _ { k } \{ D ( s - 1 , n - k ) + A _ { s } ( k ) \} , \qquad D ( 0 , 0 ) = ( 0 , 0 , 0 ) .\tag{12}
$$

The three priorities are the number of lengths outside 6–10 tokens, their distance from this interval, and their deviation from the target rate.

With $n _ { s }$ fixed, a local dynamic program constructs feasible reference boundaries $b _ { s , 1 : n . } ^ { 0 }$ using lengths in [4, 16] and avoiding protected expressions. It applies the same three priorities to the realized segment lengths, then uses total displacement from the evenly spaced boundaries round $( j m _ { s } / n _ { s } )$ as the final tie-breaker.

## B.2 Linguistic Boundary Costs

We combine rule-based sentence detection with a spaCy dependency parser to identify natural sentence and clause boundaries. Before parsing, structural templates, code, mathematics, and markup are replaced with equal-length whitespace. This prevents non-prose spans from interfering with the syntactic analysis while preserving character alignment, so linguistic evidence can be mapped back to the original token boundaries.

Each candidate boundary receives a base cost $\beta ( b )$ from local linguistic evidence. We prefer boundaries at the ends of sentences or complete clauses, and discourage boundaries that split structural labels, attached punctuation, compact code or mathematical expressions, or lexical units. Sentence endings are detected with deterministic rules that avoid common false positives such as decimals, ellipses, abbreviations, and protected expressions. Table 6 summarizes these preferences.

The dependency parse additionally assigns a local risk level $r ( b ) \in \{ 0 , 1 , 2 \}$ , where higher values indicate a more severe disruption of a short local dependency. We consider only relations spanning at most eight tokens so that long parser arcs do not penalize large portions of a sentence. The combined boundary cost is

$$
C ( b ) = \beta ( b ) + 3 r ( b ) .\tag{13}
$$

Cuts inside words, numbers, or identifiers receive a high base cost and are also marked separately, allowing lexical integrity to be optimized before the remaining soft preferences. We additionally use a rule-based fragment penalty $F ( a , b ) \in \{ 0 , 4 , 8 \}$ to discourage short, incomplete rubric fragments, while assigning zero cost to complete headers and status statements.

Table 6: Base costs for candidate internal boundaries. Lower costs favor a cut; protected expressions are excluded before scoring.
<table><tr><td>Boundary type</td><td>Base cost  $\beta ( b )$ </td></tr><tr><td>Observed sentence end</td><td>-3</td></tr><tr><td>Complete criterion-header end or predicate-bearing clause edge</td><td>0.5</td></tr><tr><td>Soft punctuation, such as a comma, colon, or semicolon</td><td>1</td></tr><tr><td>Lexical separator within code or mathematics</td><td>1.5</td></tr><tr><td>Ordinary word boundary</td><td>2</td></tr><tr><td>Internal position of a criterion header</td><td>6</td></tr><tr><td>Boundary that detaches prose punctuation or quotation delimiters</td><td>8</td></tr><tr><td>Internal position of a short literal, word, number, or identifier</td><td>12</td></tr></table>

## B.3 Constrained Local Refinement

Consider one outer block of length m with fixed count n and reference boundaries $b _ { 1 : n } ^ { 0 }$ . A candidate partition $b _ { 0 : n }$ has $b _ { 0 } = 0 , b _ { n } = m$ , and lengths $d _ { j } = b _ { j } - b _ { j - 1 }$ . We require

$$
\begin{array} { r l r l } & { 4 \le d _ { j } \le 1 6 , \quad } & { b _ { j } \notin \mathcal { F } } & { ( j < n ) , } \\ & { | b _ { j } - b _ { j } ^ { 0 } | \le w , } & { \displaystyle \sum _ { j < n } \mathbf { 1 } [ r ( b _ { j } ) = 2 ] \le \displaystyle \sum _ { j < n } \mathbf { 1 } [ r ( b _ { j } ^ { 0 } ) = 2 ] , } \end{array}\tag{14}
$$

where $\mathcal { F }$ contains the protected boundaries and the default search radius is $w = 4$ tokens.

Let $W ( { \bf { b } } )$ count the internal boundaries that cut a lexical unit. Among feasible partitions, we first minimize W, then minimize

$$
\begin{array} { c } { { \displaystyle J ( { \bf b } ) = \sum _ { j < n } C ( b _ { j } ) + \sum _ { j = 1 } ^ { n } F ( b _ { j - 1 } , b _ { j } ) + 0 . 5 0 \sum _ { j = 1 } ^ { n } \delta ( d _ { j } ) ^ { 2 } } } \\ { { + 0 . 0 2 \sum _ { j = 1 } ^ { n } ( d _ { j } - m / n ) ^ { 2 } + 0 . 1 0 \sum _ { j < n } | b _ { j } - b _ { j } ^ { 0 } | . } } \end{array}\tag{15}
$$

The first two terms favor linguistic boundaries and complete fragments; the remaining terms control preferred length, length variance, and displacement from the reference partition.

A dynamic-programming state records the chunk count, current boundary, and risk-2 count. Transitions add feasible chunks and minimize $( W , J )$ lexicographically, breaking ties by the number of moved boundaries and then their coordinates.

We adopt the candidate if it reduces W. When W is unchanged, we require J not to increase and the boundary-plus-fragment cost $\begin{array} { r } { H ( \mathbf { b } ) = \sum _ { j < n } C ( b _ { j } ) + \sum _ { j } F ( b _ { j - 1 } , b _ { j } ) } \end{array}$ to decrease by at least 0.5; otherwise we retain the reference.

Algorithm 1: Syntax-Aware Semantic Chunking with an Exact Latent Budget   
Require: teacher evaluation $c ,$ tokenizer T, rate $\rho = 8$   
1 (v, o) ← TOKENIZEONCE(c, T)   
2 O ← OUTERRUBRICBLOCKS(c, o)   
3 $N \gets \lceil \lvert \mathbf { v } \rvert / 8 \rceil$   
4 (n , . . . , n ) ← ALLOCATEBUDGET $( \mathcal { O } , N )$   
5 F ← PROTECTEDSTRUCTURALCUTS(c, o)   
6 (C, r, W, F) ← BOUNDARYANDFRAGMENTEVIDENCE(c, o)   
7 foreach outer block s do   
8 b<sup>0</sup> ← PROTECTEDREFERENCEPARTITION(m<sub>s</sub>, n<sub>s</sub>, F<sub>s</sub>)   
9 b<sub>s</sub> ← REFINEANDSELECT(b<sup>0</sup>, C, r, W, F, w = 4)   
10 if $W ( { \bf { b } } _ { s } ) > 0$ then   
11 $\mathbf { b } _ { s } ^ { \prime } \gets$ REFINEANDSELECT(b<sup>0</sup>, C, r, W, F, w = 5)   
12 if $\mathbf { \bar { \rho } } _ { W \left( \mathbf { b } _ { s } ^ { \prime } \right) } < W ( \mathbf { b } _ { s } )$ then   
13 ${ \bf b } _ { s } \gets { \bf b } _ { s } ^ { \prime }$   
Return : concatenated block-local boundaries, shifted to global token offsets

Refinement leaves each outer block’s span and latent count unchanged without increasing lexical or risk-2 cuts. Stage 1 uses the resulting compression frontiers with the supervision objective in Appendix C.

## B.4 Realized Allocation and a Complete Example

The training cache contains 35,612 evaluations, 27,541,067 explicit tokens, and 3,458,245 latent positions: 97.11 latents per evaluation and 7.96 explicit tokens per latent. The per-example latent counts match Fixed-Rate Chunking. As Table 7 shows, 86.18% of chunks contain 6–10 tokens. Refinement moves 1,836,211 internal boundaries, reducing lexical cuts from 142,295 to 151 and risk-2 cuts from 694,857 to 946.

Figures 5 and 6 compare the boundary locations produced by Semantic Chunking and Fixed-Rate Chunking on the same cached evaluation, containing 219 explicit tokens and 28 latent positions.

Table 7: Chunk lengths in the Qwen3-8B semantic training cache. All 35,612 evaluations retain their exact rate-8 latent budgets.
<table><tr><td>Explicit tokens per latent</td><td>Chunk count</td><td>Share (%)</td></tr><tr><td>4</td><td>37,514</td><td>1.08</td></tr><tr><td>5</td><td>223,876</td><td>6.47</td></tr><tr><td>6</td><td>399,275</td><td>11.55</td></tr><tr><td>7</td><td>572,989</td><td>16.57</td></tr><tr><td>8</td><td>1,053,395</td><td>30.46</td></tr><tr><td>9</td><td>523,161</td><td>15.13</td></tr><tr><td>10</td><td>431,511</td><td>12.48</td></tr><tr><td>11</td><td>185,989</td><td>5.38</td></tr><tr><td>12</td><td>28,541</td><td>0.83</td></tr><tr><td>13-16</td><td>1,994</td><td>0.06</td></tr><tr><td>Total</td><td>3,458,245</td><td>100.00</td></tr></table>

![](images/597bed45a09ecd5cff9847ac14fa55a9e5ab22a10d579c5fd17c95eeffb230b0.jpg)  
Figure 5: A complete evaluation from the semantic training cache, with the syntax-aware frontiers used for training. All 219 explicit tokens and all 28 latent frontiers are shown inside one frame. Contiguous background regions follow the coarse semantic hierarchy, their identities are written at the right, and each uniform dark | marks one latent frontier.

```diff
--- Compliance Check ---
Identified Gatekeeper| Criterion: Criterion 1: The response| directly
addresses the question posed by providing a| clear and explicit answer.
--- Analysis ---|
Response A:
- Criterion |1 [Hard Rule]: Met. The| response provides the explicit
answer "7".|
- Criterion 2 [Principle]:| Met. The answer "7" is| self-contained and
clear.
- Criterion |3 [Principle]: Met. The| response uses conventional
notation (a numeral)| to communicate the answer.
<sub>**</sub>Response B|:<sub>**</sub> Fixed-Rate
- Criterion 1 [Hard| Rule]: Met. The response provides the| explicit Chunking
answer "7". (ρ = 8)
- Criterion |2 [Principle]: Met. The| answer is presented in a complete
sentence,| making it self-contained and clear.
-| Criterion 3 [Principle]: Met|. The response uses appropriate and
conventional language| to communicate the answer effectively.
--- Final| Judgment ---
Justification: Both responses successfully| meet all criteria.
Response B is slightly| better as it provides a more complete and|
conversational answer, directly addressing the question| in a full
sentence while still being concise| and clear.|
Preference label (outside the chunked trace): Response B
```  
Figure 6: Fixed-Rate Chunking of exactly the same evaluation trace as Figure 5. The 219 explicit tokens are divided into 27 complete groups of eight and one final group of three, yielding the same 28 latent tokens. Only the red frontier locations change: unlike Semantic Chunking, they may split a phrase, a rubric criterion, or a transition between coarse semantic regions.

![](images/c5bd8a5212df5df205887b7ac1ec74aec9a4d3a52488fdd27823339cfe943907.jpg)  
Figure 7: Stage 2 loss for the LatentGRM-8B Semantic Chunking ablation. Semantic Chunking maintains a lower Stage 2 loss after the initial decline.

## C Stage 1 Latent Distillation and Stage 2 Autonomous Evaluation

We now give the full objectives for an evaluation input x, teacher evaluation $^ { c , }$ and preference label y. Let $\phi$ and ψ denote the trainable Stage 1 encoder and decoder parameters, and let θ denote the trainable Stage 2 judge parameters. Semantic Chunking specifies $\begin{array} { l } { \displaystyle \mathrm { S e g } ( c ) = \{ c _ { b _ { i - 1 } + 1 : b _ { i } } \} _ { i = 1 } ^ { N } , } \end{array}$ adapting the compression frontiers to rubric structure. Throughout this section, $c _ { 1 : M }$ denotes the tokenized evaluation retained under the training sequence-length limit.

## C.1 Encoder Sequence and Vocabulary-Space State

Let $0 = b _ { 0 } < b _ { 1 } < \dots < b _ { N } = M$ be the compression frontiers and let $\kappa _ { i }$ denote the compression placeholder following $c _ { b _ { i - 1 } + 1 : b _ { i } }$ . The encoder sequence is

$$
\xi ^ { \mathrm { e n c } } = [ x , < \mathrm { t h i n k } > , c _ { 1 : b _ { 1 } } , \kappa _ { 1 } , c _ { b _ { 1 } + 1 : b _ { 2 } } , \kappa _ { 2 } , \dots , c _ { b _ { N - 1 } + 1 : b _ { N } } , \kappa _ { N } , < / \mathrm { t h i n k } > ] .\tag{16}
$$

The encoder attention mask is causal and additionally prevents each query from attending to earlier compression placeholders. If u and v are non-padding query and key positions, respectively, then

$$
A _ { u v } ^ { \mathrm { e n c } } = 1 \quad \Longleftrightarrow \quad v \leq u \wedge \lnot ( v \in \{ \kappa _ { j } \} _ { j = 1 } ^ { N } \wedge v < u ) .\tag{17}
$$

Hence $h _ { i } = \mathrm { E n c } _ { \phi } ( { \xi } ^ { \mathrm { e n c } } ; A ^ { \mathrm { e n c } } ) _ { \kappa _ { i } }$ can use x and $c _ { 1 : b _ { i } }$ , but neither future evaluation tokens nor earlier compression placeholders.

Let V be the vocabulary size, d the hidden dimension, and $\tau$ the projection temperature. The decoder’s frozen output projection $W _ { \mathrm { o u t } } \in \mathbb { R } ^ { V \times d }$ maps $h _ { i }$ to vocabulary scores, while its frozen input embedding table $E _ { \mathrm { i n } } ~ \in ~ \mathbb { R } ^ { V \times d }$ maps a vocabulary distribution back to the decoder’s input space. The full-vocabulary construction is $\alpha _ { i } =$ softmax $( W _ { \mathrm { o u t } } h _ { i } / \tau )$ and $z _ { i } ^ { \mathrm { f u l l } } = E _ { \mathrm { i n } } ^ { \top } \alpha _ { i }$ . We retain the indices of the K largest entries in $S _ { i } = \mathrm { T o p K } ( \alpha _ { i } )$ and compute

$$
s _ { i , v } = W _ { \mathrm { o u t } } [ v ] ^ { \top } h _ { i } / \tau ,\tag{18}
$$

$$
\bar { \alpha } _ { i , v } = \frac { \exp { s _ { i , v } } } { \sum _ { u \in S _ { i } } \exp { s _ { i , u } } } , \qquad v \in S _ { i } ,\tag{19}
$$

$$
z _ { i } = \sum _ { v \in S _ { i } } \bar { \alpha } _ { i , v } E _ { \mathrm { i n } } [ v ] .\tag{20}
$$

Equivalently, $\bar { \alpha } _ { i , v } = { \alpha _ { i , v } } / \sum _ { u \in S _ { i } } { \alpha _ { i , u } }$ on $S _ { i }$ , with zero probability outside $S _ { i }$ . The full-vocabulary formula is recovered when $K = { \dot { V } }$ . Gradients pass through the mixture weights to the encoder state $h _ { i }$ but not into $W _ { \mathrm { o u t } }$ or $E _ { \mathrm { i n } }$

## C.2 Suffix Reconstruction Loss

The suffix-reconstruction objective trains each latent prefix to support the subsequent explicit evaluation and final preference. The remaining spans concatenate to $c _ { b _ { i } + 1 : M }$ , followed by the preference suffix. Our implementation uniformly samples one frontier and materializes its corresponding causal suffix view.

For frontier i, define the decoder prefix and target

$$
u _ { i } = [ x , < \mathrm { t h i n k } > , z _ { 1 : i } ] ,\tag{21}
$$

$$
t _ { i } = [ c _ { b _ { i } + 1 : M } , < / \mathrm { t h i n k } > , y , \mathrm { E O S } ] .\tag{22}
$$

If $L _ { i }$ is the tokenized length of $t _ { i }$ , the per-frontier loss is

$$
\ell _ { i } ( \phi , \psi ) = - \frac { 1 } { L _ { i } } \sum _ { j = 1 } ^ { L _ { i } } \log p _ { \psi } ( t _ { i , j } \mid u _ { i } , t _ { i , < j } ) ,\tag{23}
$$

and the complete Stage 1 objective is

$$
\mathcal { L } _ { \mathrm { r e c } } ( \phi , \psi ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { i } ( \phi , \psi ) .\tag{24}
$$

The implementation samples $I \sim \mathrm { U n i f o r m } \{ 1 , \dots , N \}$ for each example and evaluates only $\ell _ { I }$ This estimator is unbiased because

$$
\mathbb { E } _ { I } [ \ell _ { I } ] = \sum _ { i = 1 } ^ { N } \frac { 1 } { N } \ell _ { i } = \mathcal { L } _ { \mathrm { r e c } } .\tag{25}
$$

For the sampled frontier, $[ u _ { I } , t _ { I } ]$ uses consecutive position indices and a causal decoder mask. The compressed explicit prefix and future latent positions are omitted, while earlier target tokens remain visible through teacher forcing. Loss is applied only to $t _ { I }$ . Each sampled causal view provides one estimate of the full-suffix objective.

## C.3 The Three Stage 1 Optimization Phases

Let $\phi _ { 0 }$ and $\psi _ { 0 }$ denote the initial encoder and decoder parameters. The three phases optimize the same reconstruction objective with different parameter blocks:

$$
\widehat { \phi } = \arg \operatorname* { m i n } _ { \phi } \mathcal { L } _ { \mathrm { r e c } } ( \phi , \psi _ { 0 } ) ,
$$

encoder phase: decoder frozen,

$$
\widehat { \psi } = \arg \operatorname* { m i n } _ { \psi } \mathcal { L } _ { \mathrm { r e c } } ( \widehat { \phi } , \psi ) ,\tag{26}
$$

decoder phase: encoder frozen,

(27)

$$
( \phi ^ { * } , \psi ^ { * } ) = \arg \operatorname* { m i n } _ { \phi , \psi } \mathcal { L } _ { \mathrm { r e c } } ( \phi , \psi ) \quad \mathrm { i n i t i a l i z e d ~ a t } ( \widehat { \phi } , \widehat { \psi } ) , \quad \mathrm { j o i n t ~ p h a s e } .\tag{28}
$$

After joint training, the encoder runs once over every full teacher trace and exports $( S _ { i } , \bar { \alpha } _ { i } )$ for all frontiers. These sparse targets are stored by dataset index. Stage 2 initializes from the jointly trained decoder adapter, and the encoder’s role ends after target export.

## C.4 Stage 2 Soft-Target Training and Inference

Let $q _ { \theta }$ denote the Stage 2 judge’s predicted distribution over the full vocabulary. For every stored top-K target, training resamples independent $g _ { i , v } \sim$ Gumbel(0, 1) at each update and forms

$$
\widetilde { \alpha } _ { i , v } ( \mathbf { g } ) = \frac { \exp { ( ( \log { \bar { \alpha } _ { i , v } } + \lambda _ { g } g _ { i , v } ) / \tau _ { g } ) } } { \sum _ { u \in S _ { i } } \exp { ( ( \log { \bar { \alpha } _ { i , u } } + \lambda _ { g } g _ { i , u } ) / \tau _ { g } ) } } , \quad v \in S _ { i } .\tag{29}
$$

where $\lambda _ { g }$ controls the noise scale, $\tau _ { g }$ is the perturbation temperature, and $\widetilde { \alpha } _ { i , v } ( \mathbf { g } ) = 0$ for $v \not \in S _ { i }$ This perturbation discourages overfitting to deterministic teacher targets and provides diverse latent trajectories at inference. Teacher forcing inserts $\begin{array} { r } { \widetilde { z } _ { i } ( \mathbf { g } ) \ : = \ : \sum _ { v \in S _ { i } } \widetilde { \alpha } _ { i , v } ( \mathbf { g } ) E _ { \mathrm { i n } } [ v ] } \end{array}$ at the i-th latent input position. The prediction immediately preceding this position is matched to the same perturbed distribution, so latent inputs and KL targets share each noise draw. The student distribution remains normalized over the full vocabulary. For one draw, the sparse distillation loss is

$$
\mathcal { L } _ { \mathrm { K L } } ( \mathbf { g } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { v \in S _ { i } } \widetilde { \alpha } _ { i , v } ( \mathbf { g } ) \log \frac { \widetilde { \alpha } _ { i , v } ( \mathbf { g } ) } { q _ { \theta } ( v \mid x , < \mathrm { t h i n k } > , \widetilde { z } _ { < i } ( \mathbf { g } ) ) } .\tag{30}
$$

This is teacher-to-student KL: the teacher target is zero outside $S _ { i }$ while the student’s normalizer includes all vocabulary entries. Let $r = [ < / \mathrm { t h i n k } > , y , \mathrm { E O S } ]$ be the visible suffix. Its cross-entropy is

$$
\mathcal { L } _ { \mathrm { C E } } ( r ; \mathbf { g } ) = - \frac { 1 } { | r | } \sum _ { j = 1 } ^ { | r | } \log q _ { \theta } ( r _ { j } \mid x , < \mathrm { t h i n k } > , \widetilde { z } _ { 1 : N } ( \mathbf { g } ) , r _ { < j } ) .\tag{31}
$$

Prompt and latent positions are excluded from this CE term. The stochastic objective combines the latent and explicit losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a u t o } } = \mathbb { E } _ { \mathbf { g } } [ \mathcal { L } _ { \mathrm { K L } } ( \mathbf { g } ) + \mathcal { L } _ { \mathrm { C E } } ( r ; \mathbf { g } ) ] . } \end{array}\tag{32}
$$

Both terms depend on the sampled trajectory through teacher-forced inputs. Each training update estimates the expectation with one draw per example.

At inference, the judge generates its own latent trajectory. Given the preceding latent states, step i predicts

$$
q _ { i } = q _ { \theta } ( \cdot \mid x , < \mathtt { t h i n k } > , z _ { < i } ) .\tag{33}
$$

If arg max<sub>v</sub> $q _ { i } ( v ) = < / \mathrm { t h i n k } >$ , latent generation terminates. Otherwise, we define

$$
S _ { i } ^ { \theta } = \mathrm { T o p K } ( q _ { i } ) , \qquad \bar { q } _ { i } ( \boldsymbol { v } ) = \left\{ \begin{array} { l l } { q _ { i } ( \boldsymbol { v } ) } & { \boldsymbol { v } \in S _ { i } ^ { \theta } , } \\ { \sum _ { \boldsymbol { u } \in S _ { i } ^ { \theta } } q _ { i } ( \boldsymbol { u } ) } & { \boldsymbol { v } \in S _ { i } ^ { \theta } , } \\ { 0 , } & { \boldsymbol { v } \notin S _ { i } ^ { \theta } . } \end{array} \right.\tag{34}
$$

Replacing $( S _ { i } , \bar { \alpha } _ { i } )$ in Equation 29 with $( S _ { i } ^ { \theta } , \bar { q } _ { i } )$ yields the perturbed weights $\widetilde { q } _ { i } ( \mathbf { g } )$ . The next latent input is

$$
z _ { i } = \sum _ { v \in S _ { i } ^ { \theta } } \widetilde { q } _ { i } ( v ; \mathbf { g } ) E _ { \mathrm { i n } } [ v ] .\tag{35}
$$

The recurrence also ends when the latent-length cap is reached; the judge then feeds the end-marker embedding and greedily decodes the preference label. Distinct Gumbel draws provide the rollout diversity used for voting. Appendix E specifies generation limits, and Appendix F.1 describes the vLLM implementation.

## D Latent Trace Interpreter Details

The Latent Trace Interpreter (LTI) reconstructs an explicit evaluation from a latent trajectory. It is initialized from the LatentGRM-8B Stage 1 decoder, whose pretrained weights remain frozen, and learns its own LoRA adapter.

## D.1 Vocabulary-Probability Readout

For each position i, the readout consumes top-K vocabulary indices $S _ { i } ^ { \mathrm { s a v e } }$ and normalized weights $\vec { q } _ { i } ^ { \mathrm { s a v e } }$ , following the notation in Section 3.4. For a teacher-derived trajectory, $( S _ { i } ^ { \mathrm { s a v e } } , \bar { q } _ { i } ^ { \mathrm { s a v e } } ) \ =$ $( S _ { i } , \bar { \alpha } _ { i } )$ is exported by the Stage 1 encoder. For an autonomous trajectory, $( S _ { i } ^ { \mathrm { s a v e } } , \bar { q } _ { i } ^ { \mathrm { s a v e } } ) ~ =$ $( S _ { i } ^ { \theta } , \widetilde { q } _ { i } ( \mathbf { g } ) )$ is saved from the inference recurrence in Equations 34–35.

Equation 7 maps the saved vocabulary probabilities through the interpreter’s embedding table. The input is $[ x , < \pm \mathrm { h i n k } > , z _ { 1 : N } ^ { \mathrm { i n t } } ]$ , where $\boldsymbol { x } = ( q , \mathcal { R } , a , b )$ . Each mixture occupies one position in the original trajectory order, with consecutive position indices and causal attention. Text generation begins after $z _ { N } ^ { \mathrm { i n t } }$

## D.2 Full-Trajectory Reconstruction Objective

We pair each cached trajectory with the complete teacher evaluation that produced it. Let ω denote the trainable LTI parameters, let $t = [ c _ { 1 : M } , < / \mathrm { t h i n k } > ]$ , and let L be its token length. The training sequence is

$$
[ x , < \mathrm { t h i n k } > , z _ { 1 : N } ^ { \mathrm { i n t } } , c _ { 1 : M } , < / \mathrm { t h i n k } > ] ,\tag{36}
$$

with the token-averaged teacher-forcing objective

$$
\mathcal { L } _ { \mathrm { i n t } } ( \omega ) = - \frac { 1 } { L } \sum _ { j = 1 } ^ { L } \log p _ { \omega } ( t _ { j } \mid x , < \mathrm { t h i n k } > , z _ { 1 : N } ^ { \mathrm { i n t } } , t _ { < j } ) .\tag{37}
$$

Loss covers the complete evaluation and closing marker; prompt and latent positions are masked, and the binary preference label is excluded. Each example supplies one full trajectory. LTI thus reconstructs the evaluation from its beginning, whereas the Stage 1 decoder predicts the suffix after a sampled frontier.

## D.3 LTI Training Settings

For the reconstruction study, the LTI adapter is optimized with Equation 37; the frozen weights include the input embeddings. Table 8 summarizes its training configuration.

Table 8: LTI training configuration. Effective batch size includes data-parallel workers and gradient accumulation.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Initialization</td><td>LatentGRM-8B Stage 1 decoder; frozen backbone</td></tr><tr><td>Training data</td><td>28,430 examples; 512 validation examples; seed 42</td></tr><tr><td>Trajectory view</td><td>Full trajectory in forward order; one view per example</td></tr><tr><td>LTI representation</td><td>Probability-weighted top-K input embeddings</td></tr><tr><td>LoRA</td><td>Rank 32;  $\alpha = 6 4 ;$  dropout 0.05; all attention and MLP projections</td></tr><tr><td>Training duration</td><td>2 epochs</td></tr><tr><td>Learning rate / effective batch</td><td> $1 0 ^ { - 5 } / 3 2$ </td></tr><tr><td>Per-device batch / accumulation</td><td>1 example / 4 steps</td></tr><tr><td>Optimizer and schedule</td><td>AdamW; cosine decay; 3% warmup; zero weight decay</td></tr><tr><td>Precision / sequence length</td><td>bfloat16; gradient checkpointing; maximum 6,144 tokens</td></tr><tr><td>Hardware</td><td>8× NVIDIA H20</td></tr></table>

## E Supplementary RQ1: Effectiveness and Rubric-Based Process Evaluation

## E.1 Data, Backbones, and Evaluation Protocol

Each training record contains the pairwise judge prompt x, an explicit rubric-based evaluation $^ { c , }$ and the binary label y. The Semantic Chunking cache contains 35,612 OpenRubric SFT records. It contains 27,541,067 explicit evaluation tokens and 3,458,245 latent frontiers, corresponding to 773.36 explicit tokens and 97.11 latent states per example on average. The realized ratio is 7.96 explicit tokens per latent state. These counts are computed after applying the Qwen tokenizer and before Stage 1 frontier subsampling.

We use Qwen3-4B and Qwen3-8B backbones. Stage 1 starts from the base model and uses the dataset’s explicit evaluations as supervision.

In the main benchmark runs, a prediction is valid only if its whitespace-trimmed text equals Response A or Response B. Invalid generations cast no vote; ties are resolved by the earliest valid rollout. In the dedicated efficiency runs, an equal vote count or no valid label yields an invalid prediction.

## E.2 Evaluation-Time Parameters and Seeded Diversity

The main benchmark runs use maximum sequence length 6,144, top-K = 10, and Gumbel temperature and noise scale of one. Prompts reserve space for the latent start/end markers, 256 latent positions, and eight answer tokens.

Gumbel perturbations supply rollout diversity. For example index n and vote index $r ,$ the seed is

$$
s _ { n , r } = s _ { 0 } + n + r | \mathcal { D } | , \qquad s _ { 0 } = 4 2 .\tag{38}
$$

This schedule assigns a distinct noise stream to each example–vote pair and shares seed assignments across voting budgets.

## E.3 Protocol for Rubric-Based Process Evaluation

This appendix specifies the LatentGRM-8B interventions in Section 4.2.3 and supplementary trajectory analyses.

## E.3.1 Data and Annotation

<table><tr><td>Component</td><td>Text from the accepted example</td></tr><tr><td>Request Response A</td><td>Who created the Superman cartoon character? Superman, the iconic comic book superhero, was created by writer Jerry Siegel and</td></tr><tr><td></td><td>artist Joe Shuster. Superman first appeared in Action Comics #1, which was published by Detective Comics, Inc. (later DC Comics) in June 1938. The character&#x27;s immense popularity established him as one of the most enduring and recognizable figures in the</td></tr><tr><td>Response B</td><td>superhero genre. The Superman cartoon character was created by Jerry Siegel and Joe Shuster in 1938.</td></tr><tr><td>Original</td><td>3. The response should be concise and avoid unnecessary elaboration. [Principle]</td></tr><tr><td>Paraphrase</td><td>3. The response should be brief and avoid including extraneous information. [Princi-</td></tr><tr><td>Counterfactual</td><td>ple] 3. The response should provide thorough and enriching details that go beyond the minimum facts required to answer. [Principle]</td></tr></table>

Table 9: Example 1: brevity versus contextual detail.

<table><tr><td>Component</td><td>Text from the accepted example</td></tr><tr><td>Request</td><td>Generate an advertisement slogan that promotes healthy eating.</td></tr><tr><td>Response A</td><td>“Get Fit, Fuel Up - Find Balance with Our App!&quot;</td></tr><tr><td>Response B</td><td>Eat healthy, feel healthy!</td></tr><tr><td>Original</td><td>5. The response should employ vivid, engaging imagery or metaphors to capture at- tention. [Principle]</td></tr><tr><td>Paraphrase</td><td>5. The response should use striking, imaginative language or figurative expressions to draw attention. [Principle]</td></tr><tr><td>Counterfactual</td><td>5. The response should use simple, literal language that directly communicates its message without figurative embellishment. [Principle]</td></tr></table>

Table 10: Example 2: figurative versus literal language.

Sampling. Using seed 42, we sampled 200 distinct question and response pairs from each of RewardBench Chat and Chat Hard, balancing the reference winner’s displayed position within each domain. Semantic validation produced 398 accepted pairs, comprising 198 Chat and 200 Chat Hard pairs. Each pair is evaluated in both response orders, and intervention scores are mapped to response identity before aggregation.

Criterion generation and verification. GLM-5.2 was called separately for two distinct roles: generation and verification. In the generation call, it proposed an equivalent paraphrase and a counterfactual for one criterion, while all other text remained fixed. In the verification call, it independently checked the proposed edits against the rubric without benchmark labels or judge outputs. We retain for which the beneficiary inferred in the generation stage agrees with the beneficiary inferred in the verification stage.

Examples of GLM-5.2 criterion edits. Tables 9 and 10 show two accepted edits from the 398- pair analysis. Requests, responses, and criterion texts are reproduced verbatim; only the indicated criterion changes. Directions below are the proposer and reviewer’s agreed criterion-level predictions, not measured judge outcomes or overall winner labels.

## E.3.2 Judge Analysis

For each pair, we build three rubric conditions: the original rubric $\mathcal { R } _ { 0 }$ , a paraphrase control ${ \mathcal R } _ { p }$ that rewords one criterion without changing its meaning, and a criterion intervention $\mathcal { R } _ { c }$ that changes that criterion to favor a designated response. The paraphrase condition serves as a control: if a semantics-preserving rewording also shifted the preference, the effect could not be attributed to the criterion’s meaning. We index the trajectory source by $s \in \{ 0 , p , c \}$ , meaning the trajectory was generated under $\mathcal { R } _ { s }$

For each source $s \in \{ 0 , p , c \}$ , we probe its latent trajectory at $0 \% , 1 0 \% , \ldots , 1 0 0 \%$ of its recorded length. This matches relative progress across sources without forcing equal lengths. At each probe, we append the latent end marker and a shared Response prefix, and score the remaining label tokens. We measure the log-odds difference $D _ { i } ( C )$ between the two response labels (Equation 9).

Criterion intervention. Each rubric generates its own autonomous trajectory. For $s \in \{ p , c \}$ , we compare

$$
\begin{array} { r } { \Delta D _ { i , s } ( j ) = D _ { i } ( \mathcal { R } _ { s } , Z _ { s } ^ { \leq k _ { i , s , j } } ) - D _ { i } ( \mathcal { R } _ { 0 } , Z _ { 0 } ^ { \leq k _ { i , 0 , j } } ) . } \end{array}\tag{39}
$$

This contrast combines prompt and trajectory effects; its zero-latent value captures a difference already present in the rubric prompt.

Source replacement. We fix the recipient rubric to $\textstyle { \mathcal { R } } _ { 0 }$ and replay the source trajectory generated under $\mathcal { R } _ { s }$ in a fresh baseline context. Donor prompt tokens and donor KV caches are not transferred. For $s \in \{ p , c \}$ , we compare

$$
\begin{array} { r } { \Delta D _ { i , s } ( j ) = D _ { i } ( \mathcal { R } _ { 0 } , Z _ { s } ^ { \leq k _ { i , s , j } } ) - D _ { i } ( \mathcal { R } _ { 0 } , Z _ { 0 } ^ { \leq k _ { i , 0 , j } } ) . } \end{array}\tag{40}
$$

We compute signed effects per pair/order, average the two orders within each pair, then average pairs within each domain and weight the domains equally. The interventions establish criterion-directed preference shifts and their transfer through latent prefixes.

## E.4 Direct Judgment and Voting

We compare DirectJudge, explicit Rubric-RM, and LatentGRM using Qwen3-8B on the Reward-Bench Chat and Chat Hard domains and the RewardBench2 Precise IF and Focus domains. DirectJudge is initialized from the same backbone and trained for two epochs on the same OpenRubrics Dataset, supervising only the final Response $\mathtt { A } / \mathtt { B }$ verdict. It receives the request, rubric, and both responses, but generates no intermediate evaluation trace.

Direct judgment is a strong low-decoding-cost baseline. Table 11 shows competitive Chat performance and strong Focus scores, including an advantage over Rubric-RM. On Chat Hard, however, DirectJudge reaches only 67.43% at vote@5, compared with 70.50% for Rubric-RM and 73.00% for LatentGRM, and increasing votes from one to five improves it by only 0.22 points. This gap shows that direct judgment is limited on challenging preference comparisons.

Table 11: Direct judgment versus explicit and latent evaluation (%).
<table><tr><td></td><td></td><td colspan="2">RewardBench</td><td colspan="2">RewardBench2</td></tr><tr><td>Model</td><td>Votes</td><td>Chat</td><td>Chat Hard</td><td>Precise IF</td><td>Focus</td></tr><tr><td>DirectJudge</td><td>1</td><td>90.92</td><td>67.21</td><td>35.94</td><td>85.15</td></tr><tr><td></td><td>5</td><td>91.62</td><td>67.43</td><td>41.56</td><td>86.57</td></tr><tr><td></td><td>9</td><td>90.78</td><td>67.56</td><td>41.25</td><td>86.47</td></tr><tr><td>Rubric-RM</td><td>1</td><td>89.80</td><td>69.80</td><td>31.90</td><td>77.90</td></tr><tr><td></td><td>5</td><td>91.80</td><td>70.50</td><td>45.00</td><td>81.10</td></tr><tr><td></td><td>9</td><td>91.78</td><td>70.39</td><td>45.68</td><td>82.53</td></tr><tr><td>LatentGRM</td><td>1</td><td>88.70</td><td>70.50</td><td>37.90</td><td>84.60</td></tr><tr><td></td><td>5</td><td>89.90</td><td>73.00</td><td>43.40</td><td>87.50</td></tr><tr><td></td><td>9</td><td>90.08</td><td>73.24</td><td>43.50</td><td>87.68</td></tr></table>

DirectJudge produces a terminal assessment without an autoregressive evaluation trajectory, and thus loses the test-time scaling enabled by sampling multiple evaluation paths. It also lacks the diversity needed for reinforcement learning to optimize. LatentGRM instead retains a compact continuous trajectory that can be probed, inspected, and potentially optimized, at lower generation cost than explicit evaluation.

## E.5 Data Revision Protocol and Evaluation Details

An evaluation model uses the full judging context, including the task-specific rubric, to produce feedback F for a candidate response x. A separate reviser then generates

$$
{ \cal G } ( q , x , F )  x ^ { \prime } .\tag{41}
$$

The reviser receives only the question, the response to revise, and the feedback interface under study; the rubric is never included in its input.

We evaluate 240 RewardBench Rubric examples, with 120 from Chat and 120 from Chat Hard. The rubric is used to generate evaluation feedback and to score the resulting revision, but is not exposed to the reviser. All four conditions use the same Qwen3-8B Base reviser in a fresh context:

1. Prompt-only: the reviser receives only the question and the response to revise.

2. Explicit trajectory: the textual reasoning generated by the Rubric-RM-8B.

3. Latent only: the complete LatentGRM trajectory, represented at every recurrent step by the probability-weighted mixture of its top-K vocabulary embeddings, is inserted directly in the reviser’s context. This is a training-free interface: the reviser has never been trained to decode or consume LatentGRM trajectories. It measures whether a pretrained model can directly turn latent reasoning into a better response.

4. Interpreter-restored trajectory: LTI converts the same soft latent trajectory into textual feedback, which is then supplied to a fresh frozen reviser context.

We use DeepSeek-V4.1-Flash (Xu et al., 2026) as the external evaluation model. For each original or revised answer, it sees the question, the complete rubric. It independently marks every rubric criterion as satisfied or unsatisfied and provides brief supporting evidence. The original 240 answers contain 1,973 criterion decisions: 1,076 satisfied and 897 unsatisfied. On a preselected 20-example subset, reversing criterion order and paraphrasing the judging instruction yields 93.76% criterionlevel agreement and Cohen’s $\kappa = 0 . 8 0 3$

For original labels $s _ { i j }$ and revised labels $s _ { i j } ^ { \prime } .$ , we measure

$$
{ \mathrm { R e p a i r } } = { \frac { \sum _ { i j } \mathbf { 1 } [ \lnot s _ { i j } \wedge s _ { i j } ^ { \prime } ] } { \sum _ { i j } \mathbf { 1 } [ \lnot s _ { i j } ] } } ,\tag{42}
$$

$$
{ \mathrm { R e g r e s s i o n } } = { \frac { \sum _ { i j } { \bf 1 } [ s _ { i j } \wedge \neg s _ { i j } ^ { \prime } ] } { \sum _ { i j } { \bf 1 } [ s _ { i j } ] } } ,\tag{43}
$$

the percentage of answers satisfying every rubric criterion (Full), and overall criterion satisfaction.

## F Supplementary RQ2: Practical Efficiency of Latent Evaluation

## F.1 vLLM Implementation for Latent Reasoning

We extend vLLM to pass a continuous embedding back into the decoder at each latent step. The unperturbed argmax determines whether reasoning has ended; otherwise, the next input is a Gumbelweighted mixture of the top-K token embeddings. After </think>, decoding returns to discrete tokens for the preference label. Each request keeps its own feedback and random state, allowing independent latent rollouts under continuous batching and native parallel sampling.

For tensor-parallel inference, workers select local candidates before forming the global top-K and mix their embeddings on device. Cached input embeddings and persistent feedback buffers support the existing KV cache and CUDA-graph decode path. The explicit CoT baseline uses the same vLLM scheduler and request-submission policy.

## F.2 Equal-Length Runtime Control

The 50-position diagnostic covers all 716 Chat and 912 Chat Hard samples at vote@1, using the same prefix-cache reset after warmup. Both samplers set minimum and maximum output length to 50 and ignore EOS; the latent request uses an unreachable latent-end ID to retain soft feedback throughout the measured trajectory. Each model therefore produces exactly 81,400 positions.

Table 12: Equal-length serving throughput. The full RewardBench Chat and Chat Hard subsets are evaluated at vote@1 with exactly 50 generated positions per sample.
<table><tr><td>Model</td><td>Mean prompt positions</td><td>Output positions</td><td>Total inference time (s)</td><td>Throughput (positions/s)</td></tr><tr><td>Rubric-RM-8B</td><td>913.8</td><td>81,400</td><td>46.469</td><td>1,751.7</td></tr><tr><td>LatentGRM-8B</td><td>914.8</td><td>81,400</td><td>45.714</td><td>1,780.6</td></tr></table>

LatentGRM adds one prompt position per sample for its start marker (0.11% of mean prompt length). Total inference time is 46.469 seconds for Rubric-RM-8B and 45.714 seconds for LatentGRM-8B, a difference of only 1.017×. Throughput uses the same total-time boundary as natural generation. Forced latent generation never enters visible-answer decoding, so this control measures serving cost without producing accuracy predictions. The near-equal runtime at matched length shows that latent reasoning does not introduce additional per-step computational cost.

## G Supplementary RQ3: Recovering Rubric-Level Evaluations

## G.1 Readout Data and Splits

The readouts use LatentGRM-8B Stage 1 artifacts and teacher-derived trajectories. The seed-42 split contains 28,430 training, 512 validation, and 3,298 test records.

Of the test pool’s 57,097 parsed criteria, 9,768 have explicit states: 7,476 MET, 1,644 NOT MET, and 648 PARTIAL. We evaluate all 607 test records containing at least one such state. These records contain 9,992 criteria; 548 are fully labeled and 59 partially labeled.

## G.2 Readout Controls and Evaluation Protocol

To isolate the information contributed by the saved vocabulary-probability trajectory, we compare LTI with two independently trained controls. All three readouts are initialized from the LatentGRM-8B Stage 1 decoder, keep its pretrained weights frozen, and learn separate LoRA adapters.

Prompt-only. This control uses the same reconstruction target as LTI but receives only [x, <think>]. Removing the latent positions measures how much of the teacher evaluation can be reconstructed from the prompt alone.

Replay-hidden. We replay the latent trajectory through the frozen Stage 2 judge and extract the final-layer state $h _ { i } ^ { \mathrm { r e p } }$ at each latent position. By causal attention, $h _ { i } ^ { \mathrm { r e p } }$ summarizes x and $z _ { 1 : i }$ without access to the explicit evaluation. A learned bridge maps these states into the readout’s input space:

$$
\begin{array} { r l } & { \widehat { h } _ { i } ^ { \mathrm { r e p } } = \mathrm { R M S N o r m } ( h _ { i } ^ { \mathrm { r e p } } ) , } \\ & { z _ { i } ^ { \mathrm { h i d } } = W _ { \mathrm { d i r } } \widehat { h } _ { i } ^ { \mathrm { r e p } } + W _ { \mathrm { u p } } \mathrm { S i L U } ( W _ { \mathrm { d o w n } } \widehat { h } _ { i } ^ { \mathrm { r e p } } ) . } \end{array}\tag{44}
$$

Here $W _ { \mathrm { d i r } } \in \mathbb { R } ^ { d \times d }$ is a direct linear projection, while $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { d _ { b } \times d }$ and $W _ { \mathrm { u p } } \in \mathbb { R } ^ { d \times d _ { b } }$ form a nonlinear bottleneck correction. The outputs $z _ { 1 : N } ^ { \mathrm { h i d } }$ replace $z _ { 1 : N } ^ { \mathrm { i n t } }$ in Equation 36. The bridge and readout LoRA are trained jointly with Equation 37, without propagating gradients into the Stage 2 judge.

Prompt-only follows the LTI training configuration in Table 8. Replay-hidden uses bottleneck width $d _ { b } = 5 1 2$ , learning rate $5 \times 1 0 ^ { - 6 }$ , and effective batch size 28; its remaining settings follow LTI. All readouts use eight H20 GPUs. Reconstruction is decoded greedily until $< / \mathrm { t h i n k } > \mathrm { o r }$ min(4096, $L _ { \mathrm { r e f } } + 1 0 0 )$ generated tokens, where $L _ { \mathrm { r e f } }$ is the reference length. The three outputs are evaluated with the same parser and metrics below.

## G.3 Metrics and Parsing

State metrics use the explicit labels, Rubric Semantic uses all criterion justifications, and ROUGE-L uses all 607 records.

We parse traces by $g \ =$ (response identifier, criterion number, criterion type), retaining the last occurrence of duplicate keys. Let $\mathcal { G }$ contain the 9,768 gold keys with explicit states $y _ { g } \in$ {MET, NOT MET, PARTIAL}, and let $\hat { y } _ { g }$ be the parsed prediction. Missing or unparseable predictions count as errors; unlabeled reference criteria are excluded from state scoring.

Criterion State Macro-F1. For class c, precision and recall are $P _ { c } = T P _ { c } / ( T P _ { c } + F P _ { c } )$ and $R _ { c } = T P _ { c } / ( T P _ { c } + F N _ { c } )$ , with $F 1 _ { c } = 2 P _ { c } R _ { c } / ( P _ { c } + R _ { c } )$ . We report

$$
\mathrm { M a c r o F 1 } = { \frac { 1 } { 3 } } \sum _ { c \in \{ \mathrm { M E T , N o r ~ M E T , P A R T I A L } \} } F 1 _ { c } .\tag{45}
$$

The three states receive equal weight; missing predictions contribute false negatives.

Criterion Vector Exact. Let $\mathcal { F }$ be the 548 records for which every parsed gold criterion has an explicit state, and let $\mathcal { G } _ { i }$ be the criterion keys in record i. The strict record-level score is

$$
\mathrm { V e c t o r E x a c t } = \frac { 1 } { | \mathcal { F } | } \sum _ { i \in \mathcal { F } } \mathbf { 1 } [ \forall g \in \mathcal { G } _ { i } , \ \hat { y } _ { g } = y _ { g } ] .\tag{46}
$$

Partially labeled records contribute to criterion-level metrics only.

Rubric-Aligned Semantic Score. For each of the 9,992 parsed gold criteria, we align the predicted and reference justifications by g, compute rescaled RoBERTa-large BERTScore-F1 using bert-score 0.3.12, assign zero when the predicted key is missing, and macro-average over gold criteria:

$$
\mathrm { R u b r i c S e m a n t i c } = \frac { 1 } { | \mathcal { G } _ { \mathrm { a l l } } | } \sum _ { g \in \mathcal { G } _ { \mathrm { a l l } } } \mathbf { 1 } [ g \in \widehat { \mathcal { G } } ] \mathrm { B E R T S c o r e F } 1 ( \widehat { e } _ { g } , e _ { g } ) .\tag{47}
$$

All readouts use the same alignment policy and RoBERTa-large checkpoint. Table 13 breaks down criterion-state F1 by class.

ROUGE-L F1. We macro-average ROUGE-L F1 between complete generated and reference traces over the 607 records. This measures whole-trace lexical overlap, including shared templates.

Table 13: Class-wise criterion-state F1 on the 607-record test set (%).
<table><tr><td>Readout input</td><td>Met F1</td><td>Not-Met F1</td><td>Partial F1</td></tr><tr><td>Prompt-only</td><td>59.95</td><td>50.21</td><td>10.44</td></tr><tr><td>Replay-hidden</td><td>94.45</td><td>79.46</td><td>34.36</td></tr><tr><td>Latent Trace Interpreter</td><td>99.64</td><td>98.68</td><td>98.46</td></tr></table>

LTI exceeds 98% F1 for every state (Table 13). The largest separation from Replay-hidden is on PARTIAL (98.46 versus 34.36), showing that LTI better preserves distinctions among criterion judgments.

## G.4 Qualitative Reconstruction Examples

In test record 14270, LTI recovers all ten criterion states and the comparative rationale. The teacher trace (Figure 8) and LTI reconstruction (Figure 9) preserve the decisive facts. Green marks wording in the teacher reference, and purple marks the corresponding semantic paraphrase reconstructed by LTI.

```diff
--- Compliance Check ---
Identified Gatekeeper Criterion: Criterion 1: The response must directly address
the specific question posed in the request.
--- Analysis ---
Response A:
- Criterion 1 [Hard Rule]: Met. The response directly addresses the calculation
requested.
- Criterion 2 [Principle]: Met. The response correctly applies the order of
operations (PEMDAS) and performs the calculation accurately.
- Criterion 3 [Principle]: Met. The response shows the steps: (2-4) = -2, then 2
-2 = -4.
- Criterion 4 [Principle]: Met. The final answer, -4, is presented explicitly.
- Criterion 5 [Principle]: Met. The response uses standard mathematical notation
and the dollar sign for LaTeX formatting.
<sub>**</sub>Response B:<sub>**</sub>
- Criterion 1 [Hard Rule]: Met. The response provides an answer to the question.
- Criterion 2 [Principle]: Not Met. The response states the answer is 0, which is
incorrect. The correct answer is -4.
- Criterion 3 [Principle]: Not Met. No reasoning or steps are shown.
- Criterion 4 [Principle]: Met. The final answer, 0, is presented explicitly.
- Criterion 5 [Principle]: Met. The response uses standard numerical format.
--- Final Judgment ---
Justification: Response A correctly answers the mathematical question, shows the
steps involved, and adheres to all other criteria in the rubric. Response B
fails to provide the correct answer and does not show any reasoning.</think>
```

Figure 8: Case study, record 14270: teacher reference evaluation. Green highlights decisive wording in the original trace.

--- Compliance Check ---   
Identified Gatekeeper Criterion: Criterion 1: The response must directly address   
the specific question posed in the request.   
--- Analysis ---   
Response A:   
- Criterion 1 [Hard Rule]: Met. The response directly answers the mathematical   
question.   
- Criterion 2 [Principle]: Met. The response correctly applies the order of   
operations (PEMDAS).   
- Criterion 3 [Principle]: Met. The response shows the steps (2-4 = -2, then 2 -2 =   
-4).   
- Criterion 4 [Principle]: Met. The final answer (-4) is presented explicitly.   
- Criterion 5 [Principle]: Met. The response uses appropriate mathematical notation   
and format.   
Response B:   
- Criterion 1 [Hard Rule]: Met. The response directly answers the mathematical   
question.   
- Criterion 2 [Principle]: Not Met. The response states the answer is 0, which is   
incorrect. The correct answer is -4.   
- Criterion 3 [Principle]: Not Met. No steps or reasoning are shown.   
- Criterion 4 [Principle]: Met. The final answer (0) is presented explicitly.   
- Criterion 5 [Principle]: Met. The response uses appropriate notation (a number).   
--- Final Judgment ---   
Justification: Response A correctly answers the mathematical question, shows the   
correct steps, and adheres to all principles of the rubric. Response B provides   
an incorrect answer and fails to show any reasoning or steps.</think>  
Figure 9: Case study, record 14270: LTI reconstruction. Purple highlights semantic paraphrases of the teacher trace.

## G.4.1 A Reconstruction Failure

We additionally examine a reconstruction error from the same frozen 607-record test set. Among complete generations that recover every response–criterion key, record 17234 differs from the teacher reference on exactly one of 16 explicit states. The reconstruction preserves the other states and the final preference for Response A, but it is locally inconsistent: it marks Response B’s Criterion 3 as MET after noting the correct permutation formula, even though both its later analysis and final judgment recognize that the formula is applied incorrectly. Figures 10 and 11 show the teacher trace and reconstruction, respectively; green marks the decisive reference evidence, while red marks the erroneous reconstructed judgment.

--- Compliance Check ---   
Identified Gatekeeper Criterion: Criterion 1: The response must provide a final   
numerical answer to the question posed. [Hard Rule]   
--- Analysis ---   
<sub>\*\*</sub>Response A:<sub>\*\*</sub>   
- Criterion 1 [Hard Rule]: Met. The response provides the numerical answer "30".   
- Criterion 2 [Hard Rule]: Met. The response explicitly states "the word MADAM has   
two A’s and two M’s" and mentions "five letters".   
- Criterion 3 [Hard Rule]: Met. The response correctly applies the formula for   
permutations with repetitions: 5! / (2! <sub>\*</sub> 2!).   
- Criterion 4 [Principle]: Met. The response explains why division is needed (to   
avoid duplicates) and how the factorial terms are derived (arranging identical   
letters).   
- Criterion 5 [Principle]: Met. The mathematical reasoning for applying the formula   
and calculating the factorials is consistent.   
- Criterion 6 [Principle]: Met. The calculation 5! / (2! <sub>\*</sub> 2!) = 120 / (2 <sub>\*</sub> 2) =   
120 / 4 = 30 is accurate.   
- Criterion 7 [Principle]: Met. The response clearly states "two A’s and two M’s,   
which are indistinguishable from each other" and explains the need to divide by   
the arrangements of these identical letters.   
- Criterion 8 [Principle]: Met. The response follows a logical flow, starting with   
identifying repetitions, explaining the problem of duplicates, introducing the   
formula, and then calculating the result.   
<sub>\*\*</sub>Response B:<sub>\*\*</sub>   
- Criterion 1 [Hard Rule]: Met. The response provides the numerical answer "60".   
- Criterion 2 [Hard Rule]: Met. The response states "MADAM has five letters, but   
some of them are repeated. There are two A’s and two M’s."   
- Criterion 3 [Hard Rule]: Not Met. While the response states the correct formula   
(n! / (r1! r2! ... rk!)), it incorrectly applies   
it by stating the result is 60. The calculation is flawed.   
- Criterion 4 [Principle]: Met. The response explains the formula and its   
components.   
- Criterion 5 [Principle]: Not Met. The mathematical reasoning leads to an   
incorrect conclusion due to a calculation error.   
- Criterion 6 [Principle]: Not Met. The calculation (5 times 4 times 3 times 2   
times 1) / (2 times 1 times 2 times 1) simplifies to 120 / 4 = 30, not 60. The   
cancellation step is also described incorrectly ("cancel out some factors...   
such as 2 and 2, and 1 and 1. This leaves me with (5 times 4 times 3) / 1,   
which is 60"). This simplification is incorrect.   
- Criterion 7 [Principle]: Met. The response identifies the repeated letters.   
- Criterion 8 [Principle]: Met. The response presents the reasoning in a step-by  
step manner.   
--- Final Judgment   
Justification: Response A correctly identifies the problem, applies the appropriate   
combinatorial formula for permutations with repetitions, and performs the   
calculation accurately, arriving at the correct answer of 30. Response B   
correctly identifies the problem and the formula but makes a significant   
computational error in its simplification and calculation, leading to an   
incorrect answer of 60. Therefore, Response A is superior due to its   
mathematical accuracy.</think>  
Figure 10: Reconstruction failure, record 17234: teacher reference evaluation. Green marks the evidence that Response B’s Criterion 3 is not met.

Compliance Check   
Identified Gatekeeper Criterion: Criterion 1: The response must provide a final   
numerical answer to the question posed.   
--- Analysis ---   
<sub>\*\*</sub>Response A:<sub>\*\*</sub>   
- Criterion 1 [Hard Rule]: Met. The response provides the numerical answer "30".   
- Criterion 2 [Hard Rule]: Met. The response explicitly states "MADAM has two A’s   
and two M’s" and identifies them as indistinguishable.   
- Criterion 3 [Hard Rule]: Met. The response correctly applies the formula for   
permutations with repeated elements: 5! / (2! <sub>\*</sub> 2!).   
- Criterion 4 [Principle]: Met. The response explains why division by factorials is   
needed to account for identical elements and how the formula is derived.   
- Criterion 5 [Principle]: Met. The mathematical reasoning is accurate and   
consistent, leading to the correct calculation.   
- Criterion 6 [Principle]: Met. The calculation 5! / (2! 2!) = 120 / 4 = 30 is   
correct.   
- Criterion 7 [Principle]: Met. The response clearly distinguishes between   
identical (A’s and M’s) and distinct elements by stating "indistinguishable"   
and explaining the need to divide by the number of ways to arrange identical   
letters.   
- Criterion 8 [Principle]: Met. The reasoning is presented in a logical, step-by  
step manner, starting with identifying repetitions, explaining the need for   
division, and then performing the calculation.   
<sub>\*\*</sub>Response B:<sub>\*\*</sub>   
- Criterion 1 [Hard Rule]: Met. The response provides the numerical answer "60".   
- Criterion 2 [Hard Rule]: Met. The response states "MADAM has five letters, but   
some of them are repeated. There are two A’s and two M’s."   
- Criterion 3 [Hard Rule]: Met. The response correctly identifies the   
formula for permutations with repeated elements: n! / (r1! r2! ... rk!).   
- Criterion 4 [Principle]: Met. The response explains the formula and its   
components.   
- Criterion 5 [Principle]: Not Met. The mathematical reasoning is incorrect. The   
calculation 5! / (2! <sub>\*</sub> 2!) is 120 / 4 = 30, not 60.   
- Criterion 6 [Principle]: Not Met. The calculation (5 4 3 2 1) / (2 1   
2 <sub>\*</sub> 1) simplifies to 120 / 4 = 30, not 60. The explanation of canceling factors   
is also incorrect in this context.   
- Criterion 7 [Principle]: Met. The response correctly identifies the repeated   
elements.   
- Criterion 8 [Principle]: Met. The reasoning is presented in a step-by-step manner   
, although the final calculation is incorrect.   
--- Final Judgment ---   
Justification: Response A correctly identifies the problem, applies the appropriate   
combinatorial formula, and performs the calculation accurately, arriving at   
the correct answer of 30. Response B correctly identifies the formula but makes   
a significant computational error in the final calculation, leading to an   
incorrect answer of 60. Therefore, Response A is superior due to its accuracy   
.</think>  
Figure 11: Reconstruction failure, record 17234: LTI reconstruction. Red marks the erroneous MET state for Response B’s Criterion 3.

## H Training Setup and Compute

We train separate Qwen3-4B and Qwen3-8B LatentGRM models on the same 35,612 OpenRubrics records. Each run uses one node with eight NVIDIA RTX 6000D GPUs (85,651 MiB of visible memory per GPU). Training uses bfloat16 precision, DeepSpeed ZeRO Stage 2, gradient checkpointing, and LoRA. Stage 1 uses SDPA for its four-dimensional encoder attention mask, whereas Stage 2 uses FlashAttention-2. Table 14 summarizes the remaining settings.

Table 14: Training configuration for LatentGRM. Effective batch size includes eight data-parallel workers and gradient accumulation.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbones / training records</td><td>Qwen3-4B and Qwen3-8B; 35,612 records</td></tr><tr><td>Hardware per run</td><td>8× NVIDIA RTX 6000D (85,651 MiB per GPU)</td></tr><tr><td>Precision / distributed training</td><td>bfloat16; DeepSpeed ZeRO Stage 2; gradient checkpointing</td></tr><tr><td>Maximum sequence length Compression rate / latent support</td><td>6,144 tokens</td></tr><tr><td>Semantic Chunking constraints</td><td> $\rho = 8 ; { \mathrm { t o p } } { \cdot } K = 1 0 ;$  projection temperature  $\tau = 1$  hard  $_ { 4 - 1 6 }$  tokens; preferred 6–10 tokens; local search radius 4</td></tr><tr><td></td><td>tokens (5 for lexical rescue)</td></tr><tr><td>Stage 1 encoder/decoder/joint phases</td><td>3 epochs each; learning rate  $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Stage 1 batch configuration</td><td>1 per device; 8 accumulation steps; effective batch 64</td></tr><tr><td>Stage 2 reported checkpoint</td><td>Epoch 9; learning rate  $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Stage 2 batch configuration</td><td>2 per device; 16 accumulation steps; effective batch 256</td></tr><tr><td>Optimizer and schedule</td><td>AdamW; weight decay 0.01; 5% warmup; WarmupDecayLR</td></tr><tr><td>LoRA Objective weights</td><td>rank 64; scaling  $\alpha = 1 2 8 ;$  dropout 0.1; no bias</td></tr><tr><td>Training Gumbel perturbation</td><td> $\lambda _ { \mathrm { K L } } = 1$  and  $\lambda _ { \mathrm { p r e f } } = 1$ </td></tr><tr><td>Random seed</td><td>temperature 1.0; noise scale 1.0</td></tr><tr><td></td><td>42</td></tr><tr><td>Checkpointing / logging</td><td>one checkpoint per epoch; log every 10 optimizer steps</td></tr></table>

The encoder LoRA targets the attention projections, MLP projections, and embed tokens; the last module learns the newly introduced <|compress token|> embedding. Decoder and Stage 2 adapters target the attention and MLP projections. The vocabulary output matrix and input embedding basis are accessed separately, which is necessary for the untied Qwen3-8B backbone. Soft labels are generated in bfloat16 with batch size 16 and stored in shards of 1,000 examples. Deterministic tokenization, Semantic Chunking boundaries, and frontier indices are cached once; only the uniformly sampled Stage 1 supervision frontier remains dynamic across epochs.

## I Prompt Templates

![](images/2661424b604d86a3cdf69e880ff138fda0e3549d8a804c5d8c28e747981113d4.jpg)  
Figure 12: Prompt used to construct task-specific rubric criteria.

Explicit Evaluation-Trace Prompt   
You are a fair and impartial judge. Your task is to evaluate ’Response A’ and   
’Response B’ based on a given instruction and a rubric. You will conduct this   
evaluation in distinct phases as outlined below.   
### Phase 1: Compliance Check Instructions   
First, identify the single most important, objective ’Gatekeeper Criterion’ from the   
rubric.   
- <sub>\*\*</sub>A rule is objective (and likely a Gatekeeper) if it can be verified without   
opinion. Key examples are: word/paragraph limits, required output format (e.g.,   
JSON validity), required/forbidden sections, or forbidden content.<sub>\*\*</sub>   
- Conversely, a rule is subjective if it requires interpretation or qualitative   
judgment. Subjective rules about quality are NOT Gatekeepers. Examples include   
criteria like "be creative," "write clearly," "be engaging," or "use a   
professional tone."<sub>\*\*</sub>   
Think step-by-step to determine this single most important Gatekeeper.   
### Phase 2: Analyze Each Response   
Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate   
each response item by item. For each item, think step-by-step and cite concrete   
evidence from the response before assigning your judgment.   
### Phase 3: Final Judgment Instructions   
Based on the results from the previous phases, determine the winner using these simple   
rules. Provide a final justification explaining your decision first and then give   
your decision. Think step-by-step to aggregate the findings and make the decision;   
keep the reasoning explicit and concise.   
### REQUIRED OUTPUT FORMAT   
You must follow this exact output format below.   
--- Compliance Check ---   
Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>   
--- Analysis ---   
<sub>\*\*</sub>Response A:<sub>\*\*</sub>   
- Criterion 1 [Hard Rule]: Justification: <...>   
- Criterion 2 [Hard Rule]: Justification: <...>   
- Criterion 3 [Principle]: Justification: <...>   
- ... (and so on for all other criteria)   
<sub>\*\*</sub>Response B:<sub>\*\*</sub>   
- Criterion 1 [Hard Rule]: Justification: <...>   
- Criterion 2 [Hard Rule]: Justification: <...>   
- Criterion 3 [Principle]: Justification: <...>   
- ... (and so on for all other criteria)   
--- Final Judgment ---   
Justification: <...>   
Winner: <Response A / Response B>   
Task to Evaluate:   
Instruction:   
{instruction}   
Rubric:   
{rubric}   
Response A:   
{response\_a}   
Response B:   
{response\_b}  
Figure 13: Prompt used to elicit a structured explicit evaluation trace.

![](images/848d7bf087640022e9901605814c8506465e7ff4e42dd3e0ddbad6681b67ea64.jpg)  
Figure 14: Prompt used for LatentGRM training and inference.