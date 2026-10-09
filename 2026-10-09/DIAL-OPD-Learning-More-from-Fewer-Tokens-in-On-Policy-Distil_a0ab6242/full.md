# DIAL-OPD: Learning More from Fewer Tokens in On-Policy Distillation

Anhao Zhao1,2 Haoran Xin3 Junlong Tong1,4 Yingqi Fan⁵ Xuan Lu1,4 Ping Nie6 Wenjie Li2 Xiaoyu Shen¹†

1EIT-NLP Lab, Eastern Institute of Technology, Ningbo

2The Hong Kong Polytechnic University 3HKUST (GZ)

4Shanghai Jiaotong University 5University of Hong Kong 6University of Waterloo

†Corresponding author.

anhao.zhao@connect.polyu.hkxyshen@eitech.edu.cn

EIT-NLP/DIAL-OPD

## Abstract

On-policy distillation (OPD) supervises student-generated trajectories with token-level teacher signals, and its sampled-token variant has become the practical default by avoiding the cost of full-vocabulary probabilities. Yet we find that supervising more tokens is not better: because the student's capacity is finite, training on a carefully chosen subset can significantly outperform full-token OPD. Effective OPD therefore hinges not on how much supervision to provide, but on which tokens deserve it. Existing disagreement-based criteria, however, are scale-blind: the log-ratio reward treats a token the teacher strongly endorses and one that both models assign negligible probability as equally informative, causing supervision to concentrate on low-low tokens that hinder learning. We propose DIAL-OPD, a token-selection method that bridges log-probability and probability spaces by weighting reward magnitude with the logarithmic mean of teacher and student probabilities. The parameter β controls this weighting, and the highest-scoring tokens are retained for training. We evaluate DIAL-OPD against 9 baselines across 4 teacher-student pairs and 7 mathematical reasoning benchmarks, showing that retaining only 40% of tokens, it outperforms Vanilla OPD and its full-token variants by up to 5.25 percentage points in mean accuracy, while doubling Vanilla OPD's AIME25 Pass@16 from 13.33% to 26.67%. It also outperforms the strongest token-selection baseline at matched retention ratios, with up to an 18% relative improvement in mean accuracy. Even with a 4B teacher, DIAL-OPD surpasses the strongest full-token baseline using an 8B teacher at both student scales, showing that effective supervision allocation can outweigh teacher scaling. Further analysis shows that moderate β best balances suppressing low-low tokens against preserving useful disagreements, while token-level evidence reveals that DIAL-OPD removes high-reward tokens with limited reasoning value without sacrificing supervision critical to reasoning correctness.

## 1 Introduction

On-policy distillation (OPD) has become a standard component of LLM post-training [5, 28, 30] By supervising student-generated trajectories, OPD reduces the exposure bias of fixed-response distillation [1, 10] while providing dense token-level guidance beyond the sparse outcome rewards used in reinforcement learning [4, 44]. A straightforward formulation is to minimize the reverse KL divergence between the full teacher and student distributions at every generated token. However, obtaining and transmitting vocabulary-wide teacher probabilities is prohibitively expensive in both computation and memory [32, 47|. As a result, the prevailing practical formulation is sampled-token OPD, which uses the token sampled by the student to construct an unbiased single-sample Monte Carlo estimate of the reverse KL and optimizes the student with a token-level

![](images/c24b95f82b550218bfb9bfbf58004ea1e811483956ca7a57bae7fba4592b110a.jpg)  
Figure 1: Supervision density has a nonmonotonic effect. Random selection (1-90%) vs. full-token OPD (100%), 0.6B student; inset: final accuracy.

![](images/8c8457c95d28bb1d6a308bace8b1eaeede2d645b3a0a773a028e69ff54060727.jpg)  
Figure 2: The non-monotonic effect persists without a teacher-student size gap. Random selection vs. full-token OPD, 4B student; inset: final accuracy.

## log-ratio reward [9, 21, 37].

While sampled-token OPD retains the dense, token-level nature of OPD, its supervision is fundamentally noisier: rather than comparing the full distributions, each token provides only a single-sample estimate of the teacher-student discrepancy. This raises a basic question: if the key advantage of OPD is dense token-level supervision, should we necessarily supervise every token even under sampled-token OPD? We investigate this question with Qwen3-4B as the teacher and Qwen3-0.6B as the student. Surprisingly, randomly retaining only a subset of tokens consistently outperforms full-token Vanilla OPD across all tested retention ratios from 40% to 90%, with 80% retention improving MATH-500 accuracy by 8.2 percentage points (Figure 1). This advantage disappears when the student is enlarged to match the 4B teacher (Figure 2), indicating that the benefit is closely tied to student capacity. A limited-capacity student cannot absorb all noisy token-level signals equally well; forcing it to learn from every token can therefore dilute useful supervision with signals that are less informative or harder to learn. These results suggest that in sampled-token OPD, more supervision is not necessarily better: the key is to spend limited learning capacity on the tokens that provide the most useful signals.

This observation naturally leads to the next question: which tokens should be selected? Random selection treats all tokens equally and therefore ignores differences in their learning value. We first consider teacher-student disagreement as a natural selection criterion [31, 38, 41]. In sampled-token OPD, the log-ratio reward compares the probabilities that the teacher and student assign to the sampled token, making its magnitude a natural measure of their disagreement. We find that selecting tokens with the largest absolute rewards outperforms both selecting those with the smallest rewards and random selection across various retention ratios (Figure 3). These results support larger reward magnitudes as a proxy for greater token learning value. However, this strategy exhibits prolonged stagnation early in training, and its final performance does not consistently exceed full-token Vanilla OPD. Examining the log-ratio reward reveals a key blind spot: it measures relative disagreement but ignores absolute probability scale. For example, student-teacher probabilities of (0.08, 0.8) and $( 1 0 ^ { - 4 } , 1 0 ^ { - 3 } )$ yield the same reward, although the former reflects strong teacher support and the latter negligible probability under both models. We call tokens in the latter regime low-low tokens. Prior work identifies reliability issues in low-probability generation tails [13] and shows that suppressing rewards where both models assign low probability improves OPD training [43], motivating closer scrutiny of these tokens. We find that (i) when selecting tokens with the largest absolute rewards at 10% retention, low-low tokens account for only 20.6% of selected tokens but 49.8% of total absolute reward (Figure 4); (ii) replacing selected low-low tokens with non-low-low tokens at the same retention ratio mitigates early training stagnation and improves final accuracy across various retention ratios (Figure 5). Selecting tokens solely by reward magnitude can therefore concentrate supervision on tokens that hinder learning. To address this blind spot, token selection should consider both the relative disagreement captured by reward magnitude and the absolute probability scale at which it occurs.

Building on these findings, we propose DIAL-OPD, a token-selection method that bridges log-probability and probability spaces. We use the logarithmic mean of the probabilities assigned to each token by the teacher and student to incorporate probability scale into reward-based selection. A tunable parameter $\beta$ controls the strength of probability weighting, reducing the priority of low-low tokens for $\beta > 0$ . The resulting criterion provides a continuous dial: $\beta = 0$ recovers selection based solely on reward magnitude, $\beta = 1$ corresponds to selection by the absolute probability ${ \mathrm { g a p } } ,$ and intermediate values geometrically interpolate between the two, while $\beta > 1$ further strengthens probability weighting. DIAL-OPD selects the highest-scoring tokens within each response at a specified retention ratio and applies the original OPD loss only to the selected tokens.

Experiments show that incorporating probability scale mitigates early training stagnation and improves final accuracy, with extensive $\beta$ sweeps favoring moderate weighting. We evaluate DIAL-OPD against 9 baselines across 4 teacher-student pairs and 7 mathematical reasoning benchmarks. Retaining only 40% of tokens, DIAL-OPD surpasses Vanilla OPD and the strongest OPD variant, which train on all response tokens, in mean accuracy across all settings, by up to 5.25 and 1.24 percentage points, respectively. Notably, it doubles Vanilla OPD's AIME25 Pass@16 from 13.33% to 26.67%. At 20% retention, DIAL-OPD exceeds the strongest token-selection baseline at the same ratio by up to 5.23 points in mean accuracy and 10.33 points on GSM8K Avg@4. With a 4B teacher, it also surpasses the strongest full-token reference using an 8B teacher by up to 1.76 mean accuracy points, demonstrating that effective supervision allocation can outweigh teacher scaling. Selection analyses suggest that moderate weighting suppresses low-low tokens, whereas excessive weighting may discard useful disagreements. Token-level studies reveal that DIAL-OPD filters high-reward tokens with limited reasoning value while preserving supervision essential to reasoning correctness.

## 2 Preliminaries

On-policy distillation (OPD) aligns a student policy $\pi _ { \theta }$ with a fixed teacher $\pi _ { T }$ on studentgenerated responses [1, 10]. For $x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot \mid x )$ , and $c _ { t } = ( x , y _ { < t } )$ , OPD minimizes the reverse KL between their next-token distributions over vocabulary V:

$$
\mathcal { L } _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { x , y } \left[ \sum _ { t = 1 } ^ { | y | } D _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot \mid c _ { t } ) \parallel \pi _ { T } ( \cdot \mid c _ { t } ) ) \right] , \quad D _ { \mathrm { K L } } ( P \| Q ) = \sum _ { v \in \mathcal { V } } P ( v ) \log \frac { P ( v ) } { Q ( v ) } .\tag{1}
$$

Transferring and storing full-vocabulary teacher distributions incurs substantial communication and memory costs; sampled-token OPD reduces these costs by querying only the teacher logprobabilities of student-generated tokens [32, 47]. With the rollout context distribution held fixed during differentiation, the reverse-KL descent direction takes the policy-gradient reinforcement learning (RL) form [33, 36]:

$$
g _ { \mathrm { O P D } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot \vert x ) } \left[ \sum _ { t = 1 } ^ { \vert y \vert } \underbrace { \log \frac { \pi _ { T } ( y _ { t } \mid c _ { t } ) } { \pi _ { \theta } ( y _ { t } \mid c _ { t } ) } } _ { r _ { t } ^ { \mathrm { O P D } } } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } \mid c _ { t } ) \right] .\tag{2}
$$

Student rollouts provide an unbiased Monte Carlo estimate of $g _ { \mathrm { O P D } }$ , with reward $r _ { t } ^ { \mathrm { O P D } }$ treated as a stop-gradient weight (22, 28, 43; derivation in Appendix C). This formulation is adopted by recent open-source LLMs [9, 21, 37] and is the focus of this work.

![](images/96f714eb9dcbc779e6c2c95d5f50d42c0354f2bda19d2a26ca2fb50a22ff9517.jpg)  
Figure 3: Reward-based selection; inset: final accuracy.

![](images/791eeaef74b0818636f794bb120274c4c5dcb79b3146ed475ef5fe6adb6a195b.jpg)  
Figure 4: Low-low token and absolute reward shares (Reward-Top).

![](images/4555f0a2dee00ead29b393621e692cb2a6c7005839ce71575e874b06c33801d2.jpg)  
Figure 5: Low-low replacement improves learning.

## 3 Rethinking Supervision Allocation in On-Policy Distillation

## 3.1 More Supervision Does Not Imply Better Learning

Dense supervision is a central advantage of OPD, providing teacher guidance at every generated token [28]. To test the implicit expectation that denser supervision improves learning, we conduct a controlled experiment comparing full-token training with uniform random token selection under a fixed update budget.

Experimental setup. We train Qwen3-0.6B with a Qwen3-4B teacher [30] on DeepScaleR [29]. Under a fixed update budget, we compare uniform random selection within each response $( \rho \in \{ 1 \% , 5 \% , 1 0 \% , 2 0 \% , \dots , 9 0 \% \} )$ with full-token Vanilla OPD $( \rho = 1 0 0 \% )$ . All tokens remain in context; the OPD loss is averaged over selected tokens in each batch. We track MATH-500 accuracy throughout training [12, 26]; implementation and evaluation details appear in Appendix B.1.

Results. Surprisingly, random token selection outperforms full-token Vanilla OPD at all tested retention ratios from 40% to 90%, reaching 38.4% accuracy at 80% versus 30.2%, a gain of 8.2 percentage points (Figure 1). However, this benefit may reflect the smaller student's limited capacity to utilize dense supervision from a larger teacher. To test this explanation, we use a Qwen3-4B student with the same teacher. Figure 2 shows that (i) 10% retention nearly matches full-token Vanilla OPD (70.6% vs. 71.2%); and (ii) 30% retention outperforms it (72.6%). Together, these results show that more supervision does not necessarily improve learning, even at equal teacher-student model size. Since random selection already improves learning, how can we identify the token subset that maximizes learning performance under a fxed update budget?

## 3.2 Which Tokens Are Worth Learning From?

Random selection improves learning without distinguishing token value, motivating a more principled criterion. Teacher-student disagreement offers a natural starting point: recent work improves OPD by using disagreement or token update impact to guide selection [31, 38, 41]. Full-vocabulary OPD measures disagreement by KL divergence; sampled-token OPD uses a log-ratio reward that weights each token's policy-gradient term (Equation 2). We therefore examine whether reward magnitude identifies tokens worth prioritizing.

Experimental setup. At 10%, 15%, and 20% retention per response, we compare Reward-Top (largest absolute rewards) with (i) Reward-Bottom (smallest), testing whether larger rewards are more useful; and (ii) Random (uniform sampling), testing whether reward-based selection outperforms uninformed selection. All methods use matched retention ratios, the original OPD loss, and the training and evaluation settings of Section 3.1.

Results. Reward-Top outperforms Reward-Bottom and Random at 10%, 15%, and 20% retention, reaching 25.6%, 30.4%, and 34.4% accuracy, respectively (Figure 3). However, (i) all three runs show little improvement during the first 350 training steps, indicating that large rewards do not readily translate into learning gains; and (ii) Reward-Top does not consistently outperform full-token Vanilla OPD. Thus, reward magnitude helps distinguish token learning value but is insufficient for reliable selection. These limitations raise a question: what information does reward magnitude miss when assessing token learning value?

## 3.3 Low-Low Supervision Can Hinder Learning

To understand these limitations, we examine the log-ratio reward: it captures relative disagreement but ignores absolute probability scale. Student-teacher probabilities of (0.08, 0.8) and $( 1 0 ^ { - 4 } , 1 0 ^ { - 3 } )$ both yield log 10, despite strong teacher support in the former and negligible probabilities under both models in the latter, which we term low-low tokens. Prior work identifies unreliable lowprobability generation tails [13], while PowerOPD improves training stability and performance by suppressing low-low rewards [43]. Accordingly, we test (i) whether Reward-Top overemphasizes low-low tokens and (ii) whether training on them hinders learning.

Experimental setup. Following Section 3.2, we study Reward-Top at 10%, 15%, and 20% retention, defining low-low tokens by student and teacher probabilities both at most 1%. (i) To quantify their concentration, we track their shares of selected tokens and total absolute reward throughout training. (ii) To test whether they hinder learning, we replace selected low-low tokens with the highest-reward unselected non-low-low tokens (Low-low replacement). As a control, Random replacement removes the same number of randomly selected tokens and refills by reward rank without excluding low-low tokens. Both preserve the retention ratio and OPD loss (Appendix B.3).

Results. (i) Low-low tokens carry disproportionate reward. At 10%, 15%, and 20% retention they constitute 20.59%, 13.49%, and 10.26% of selected tokens but contribute 49.80%, 41.43%, and 38.01% of total absolute reward, respectively (Figure 4). (ii) Targeted replacement accelerates learning and improves accuracy. Low-low replacement yields earlier gains and accuracies of 38.4%, 37.6%, and 40.8%, exceeding Reward-Top by 12.8, 7.2, and 6.4 percentage points, respectively (Figure 5). It also consistently outperforms random replacement, supporting targeted low-low removal. Thus, low-low supervision can hinder learning despite large rewards, indicating that token selection should consider both relative disagreement and absolute probability scale.

## 4 Method

Reward magnitude helps identify valuable tokens, yet low-low tokens contribute disproportionate absolute reward and can hinder learning. We therefore propose DIAL-OPD, a token-selection method that preserves the relative-disagreement signal captured by reward magnitude while incorporating absolute probability scale to downweight low-low tokens.

## 4.1 Bridging Log-Probability and Probability Spaces

Let $p _ { t } = \pi _ { \theta } ( y _ { t } \mid c _ { t } )$ and $q _ { t } = \pi _ { T } ( y _ { t } \mid c _ { t } )$ denote student and teacher probabilities for a sampled token. To make log-ratio-based selection probability-aware, we use the logarithmic mean [2]

![](images/4f371f45198b8b84ed931dd52bfa9698c6de6e084e0c010c9ed07eb2f4180eea.jpg)  
Figure 6: Learning dynamics of DIAL-OPD across $\beta$ for Qwen3-0.6B and Qwen3-1.7B students, using a Qwen3-4B teacher and 40% token retention. Insets show final accuracy as a function of $\beta$

which averages probabilities along a uniform interpolation between log pt and log qt:

$$
L ( p _ { t } , q _ { t } ) = \int _ { 0 } ^ { 1 } p _ { t } ^ { 1 - u } q _ { t } ^ { u } d u ,\tag{3}
$$

where $u \in [ 0 , 1 ]$ parameterizes the interpolation and $p _ { t } ^ { 1 - u } q _ { t } ^ { u }$ is the corresponding probability. Since these probabilities lie between $p _ { t }$ and $q _ { t }$ , low probabilities under both models yield a small logarithmic mean. DIAL-OPD incorporates this probability scale by weighting the OPD reward magnitude with $L ( p _ { t } , q _ { t } ) ^ { \beta }$ , where $\beta$ controls the weighting strength:

$$
\boxed  \begin{array} { r l r } { s _ { \beta , t } = L ( p _ { t } , q _ { t } ) ^ { \beta } \vert r _ { t } ^ { \mathrm { O P D } } \vert } \end{array} \} , \qquad \beta \geq 0 .\tag{4}
$$

For $\beta > 0$ , the small logarithmic mean of low-low tokens reduces their selection scores. Substituting $L ( p _ { t } , q _ { t } ) = ( q _ { t } - p _ { t } ) / ( \log q _ { t } - \log p _ { t } )$ for $p _ { t } \neq q _ { t }$ connects relative and absolute disagreement:

$$
s _ { \beta , t } = \left( \frac { \lvert q _ { t } - p _ { t } \rvert } { \lvert r _ { t } ^ { \mathrm { O P D } } \rvert } \right) ^ { \beta } \lvert r _ { t } ^ { \mathrm { O P D } } \rvert = \lvert r _ { t } ^ { \mathrm { O P D } } \rvert ^ { 1 - \beta } \lvert q _ { t } - p _ { t } \rvert ^ { \beta } .\tag{5}
$$

At $\beta = 0$ and $\beta = 1$ , selection uses the OPD reward magnitude and absolute probability gap, respectively. Intermediate values geometrically interpolate between them, while $\beta > 1$ strengthens probability weighting. DIAL-OPD thus provides a continuous dial between relative and absolute disagreement for token selection.

## 4.2 The Role of $\beta$ in DIAL-OPD

We examine how $\beta$ affects token selection when (i) student and teacher probabilities are both low, or (ii) one is low and the other is moderate or high.

Proposition 1 (Controllable Suppression of Low-Low Disagreement) For $p , q \in ( 0 , 1 ]$ and $\beta \geq 0$ , scaling both probabilities by $\varepsilon \in ( 0 , 1 ]$ changes the selection score according to

$$
s _ { \beta } ( \varepsilon p , \varepsilon q ) = \varepsilon ^ { \beta } s _ { \beta } ( p , q ) .\tag{6}
$$

For $\varepsilon < 1$ and $p \neq q .$ the score remains unchanged at $\beta = 0 .$ while larger $\beta > 0$ increasingly suppresses low-low disagreement. We next examine when only one model assigns low probability.

Proposition 2 (Sensitivity to One-Sided Low Probabilities) For fixed $a \in ( 0 , 1 ]$ and $\beta \geq$ 0,

$$
\operatorname* { l i m } _ { b \to 0 ^ { + } } s _ { \beta } ( a , b ) = \operatorname* { l i m } _ { b \to 0 ^ { + } } s _ { \beta } ( b , a ) = { \left\{ \begin{array} { l l } { + \infty , } & { 0 \leq \beta < 1 , } \\ { a , } & { \beta = 1 , } \\ { 0 , } & { \beta > 1 . } \end{array} \right. }\tag{7}
$$

Proofs are given in Appendices E.1 and E.2. Together, these propositions reveal a trade-off: increasing $\beta$ strengthens low-low suppression, but $\beta > 1$ also drives scores to zero when only one probability vanishes. Only $\beta = 1$ preserves a finite, nonzero limit in this case.

## 4.3 DIAL-OPD

Let $\nu _ { i }$ denote the candidate token positions in student-generated response $i ,$ with $| \nu _ { i } | > 0$ . At retention ratio $\rho \in ( 0 , 1 ]$ , DIAL-OPD selects $K _ { i }$ tokens maximizing their total score:

$$
K _ { i } = \operatorname* { m a x } \{ 1 , \mathrm { r o u n d } ( \rho | \mathcal { V } _ { i } | ) \} , \quad \mathcal { S } _ { i } ^ { \star } \in \underset { S \subseteq \mathcal { V } _ { i } } { \arg \operatorname* { m a x } } \sum _ { t \in S } s _ { \beta , i , t } , \quad m _ { i , t } = \mathbb { I } \{ t \in \mathcal { S } _ { i } ^ { \star } \} .\tag{8}
$$

Using the binary selection mask $m _ { i , t }$ , DIAL-OPD applies the original signed OPD reward (Section 2) only to selected tokens. For a training batch B, it minimizes the masked objective:

$$
\mathcal { L } _ { \mathrm { D I A L } } ( \theta ) = - \frac { 1 } { M _ { B } } \sum _ { i \in B } \sum _ { t \in \mathcal { V } _ { i } } m _ { i , t } \mathrm { s g } [ r _ { i , t } ^ { \mathrm { O P D } } ] \log \pi _ { \theta } ( y _ { i , t } \mid c _ { i , t } ) , \qquad M _ { B } = \sum _ { i \in B } \sum _ { t \in \mathcal { V } _ { i } } m _ { i , t } .\tag{9}
$$

Here, sg denotes stop-gradient, with the mask fixed during differentiation. Unselected tokens remain in context but are excluded from the loss. DIAL-OPD recomputes scores and masks at each training step using fresh rollouts from the current student policy.

## 5 Experimental Setup

Models and training. We train Qwen3 students (0.6B/1.7B) with 4B/8B teachers [30] on DeepScaleR [29] for 800 steps (details in Appendix B).

Evaluation. We evaluate on GSM8K [3], MATH-500 [26], AMC23, AIME24, AIME25 [6], Minerva Math [23], and OlympiadBench [11].

Baselines. (i) Reference baselines include full-token Vanilla OPD, uniform Random selection, PowerOPD with bounded power-transformed rewards [43], and ExOPD with extrapolated teacherreference rewards [39]. (ii) Token-selection baselines include TIP, combining student entropy and teacher-student divergence [38]; SA-OPD, using input dependence and distillation divergence [18]; TA-OPD, combining disagreement and teacher support [35]; and Prefix OPD, retaining response prefixes [42]. We also include Capability-Contrast OPD (CC-OPD), adapting Direct-OPD's policy contrast [7] through absolute teacher-reference probability gaps. Token-selection methods are compared at matched retention ratios; implementation details appear in Appendix B.2.

## 6 Experiments

## 6.1 The Effect of $\beta$ on Learning

DIAL-OPD uses probability scale to downweight low-low tokens, with $\beta$ controlling the weighting strength. Increasing $\beta$ suppresses low-low disagreement more strongly, but may also reduce the priority of useful disagreements supported by one model. We therefore examine how this trade-off affects learning dynamics and final accuracy across different student model sizes.

Experimental setup. We fix the retention ratio at 40% and sweep $\beta$ over {0, 0.25, 0.5, 0.75, 1, 1.2, 1.4, 1.6, 1.8, 2}. We use Qwen3-4B as the teacher and Qwen3-0.6B and Qwen3-1.7B as students, following the training settings in Section 5. Figure 6 reports MATH-500 learning curves, with insets summarizing final accuracy.

Table 1: Results with a Qwen3-4B teacher on seven benchmarks. Bold and underlined mark the best and second-best accuracy per student and retention ratio, excluding full-token references.
<table><tr><td colspan="2">Keep Method</td><td>GSM8K</td><td>MATH500</td><td>AMC23</td><td colspan="2">AIME24</td><td>AIME25</td><td>Minerva</td><td>Olympiad</td><td>Mean</td></tr><tr><td></td><td></td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td></td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg</td></tr><tr><td colspan="9">Student: Qwen3-0.6B</td><td></td></tr><tr><td>Vanilla OPD</td><td></td><td>58.5 79.7 506</td><td>35.952.61687</td><td>17.3 50.05059</td><td>1.0</td><td>6.77266</td><td>1.2 10.0 7369</td><td>9.2 22.81436</td><td>12.824.55412</td><td>19.4</td></tr><tr><td>ExOPD</td><td>100% PowerOPD</td><td>68.5 84.9 428</td><td>43.560.01511</td><td>23.6 62.54338</td><td>1.7</td><td>3.36535</td><td>1.010.06816</td><td>10.922.81229</td><td>17.530.34531</td><td>23.8</td></tr><tr><td></td><td></td><td>54.8 78.8 533</td><td>34.4 52.61876</td><td>13.1 45.05512</td><td>1.7</td><td>6.77416</td><td>0.610.0 7704</td><td>8.2 20.21702</td><td>12.224.55565</td><td>17.9</td></tr><tr><td>TIP</td><td>Random selection</td><td>38.8 70.71880</td><td>12.0 32.03196</td><td>5.6 32.56893</td><td>0.0</td><td>0.07787</td><td>0.4 6.77555</td><td>4.015.42996</td><td>4.814.47057</td><td>9.4</td></tr><tr><td></td><td></td><td>54.7 76.4 610</td><td>29.948.02058</td><td>10.855.05663</td><td>0.8</td><td>3.37611</td><td>0.2 3.37801</td><td>6.819.11856</td><td>10.922.85856</td><td>16.3</td></tr><tr><td></td><td>SA-OPD</td><td>56.680.0 533</td><td>35.5 52.41731</td><td>15.652.55142</td><td>1.2</td><td>6.76979</td><td>1.010.07163</td><td>9.322.81429</td><td>13.826.65296</td><td>19.0</td></tr><tr><td>20%</td><td>TA-OPD</td><td>7.6 25.83160</td><td>5.0 17.43386</td><td>2.2 30.06711</td><td>0.0</td><td>0.07097</td><td>0.2 3.37209</td><td>1.7 9.93218</td><td>2.1 7.46876</td><td>2.7</td></tr><tr><td>CC-OPD</td><td>Prefix OPD</td><td>45.6 75.51087</td><td>23.8 47.82307</td><td>11.4 47.55736</td><td>0.4</td><td>3.37041</td><td>0.8 6.77290</td><td>5.819.52133</td><td>9.4 23.65773</td><td>13.9</td></tr><tr><td>DIAL-OPD</td><td></td><td>50.266.92615</td><td>21.7 35.03523</td><td>6.4 27.57934</td><td>0.0</td><td>0.08192</td><td>1.0 3.38192</td><td>4.414.03775</td><td>6.213.87883</td><td>12.8</td></tr><tr><td>DIAL-OPD</td><td>(β=1)</td><td>67.0 81.31072</td><td>34.2 47.22609</td><td>15.9 45.06554</td><td>0.4</td><td>6.77985</td><td>1.510.08016</td><td>7.6 16.92617</td><td>13.1 24.26658</td><td>19.9</td></tr><tr><td></td><td>(β=1.2)</td><td>63.482.6 553</td><td>39.754.61793</td><td>17.3 52.54910</td><td>1.9</td><td>3.36874</td><td>0.810.07239</td><td>9.323.91522</td><td>13.927.44907</td><td>20.9</td></tr><tr><td>TIP SA-OPD</td><td>Random selection</td><td>53.880.0 610</td><td>33.7 52.01997</td><td>14.1 52.55545</td><td>0.6</td><td>3.37376</td><td>0.6 6.77381</td><td>8.0 22.11730</td><td>11.8 24.55724</td><td>17.5</td></tr><tr><td></td><td></td><td>56.6 79.2 576</td><td>35.1 51.41829</td><td>16.965.05321</td><td>1.0</td><td>3.37405</td><td>0.8 6.77666</td><td>8.6 21.71622</td><td>12.324.85520</td><td>18.8</td></tr><tr><td>TA-OPD</td><td></td><td>55.8 79.8 500</td><td>37.053.01749</td><td>14.7 55.05142</td><td>1.0</td><td>3.36997</td><td>0.6 10.07447</td><td>8.5 21.01491</td><td>12.6 24.55491</td><td>18.6</td></tr><tr><td>40% Prefix OPD</td><td></td><td>11.938.12574</td><td>7.8 24.62782</td><td>3.0 27.55864</td><td>0.2</td><td>3.36250</td><td>0.2 3.36177</td><td>2.615.12744</td><td>2.7 9.26042</td><td>4.1</td></tr><tr><td>CC-OPD</td><td></td><td>50.3 78.0 754</td><td>32.2 50.22099</td><td>13.8 52.55715</td><td>0.4</td><td>3.37379</td><td>0.4 3.37596</td><td>8.7 21.31863</td><td>10.2 21.85964</td><td>16.6</td></tr><tr><td></td><td></td><td>67.5 82.71120</td><td>39.9 53.62312</td><td>22.7 55.06381</td><td></td><td>1.7 10.07916</td><td>2.313.37931</td><td>8.922.12302</td><td>16.1 27.06225</td><td>22.7</td></tr><tr><td>DIAL-OPD DIAL-OPD</td><td>(β=1)</td><td>69.184.7 578</td><td>44.961.61714</td><td>25.5 50.04774</td><td></td><td>2.513.37001</td><td>1.2 6.77179</td><td>10.7 22.11484</td><td>18.833.54705</td><td>24.7</td></tr><tr><td></td><td>(β=1.2)</td><td>67.2 83.5 439</td><td>44.0 59.01521</td><td>25.2 62.54520</td><td></td><td>2.1 10.06761</td><td>0.2 3.36884</td><td>10.925.71216</td><td>17.1 29.44604</td><td>23.8</td></tr><tr><td colspan="9">Student: Qwen3-1.7B</td><td></td></tr><tr><td>100% PowerOPD ExOPD</td><td>Vanilla OPD</td><td>74.2 92.6 81.5 92.6</td><td>348 340</td><td>58.0 73.41150 59.6 74.21208</td><td>33.8 72.53471</td><td>7.7 16.76150</td><td>7.3 26.75791</td><td>17.2 30.91018</td><td>26.6 41.23887</td><td>32.1</td></tr><tr><td></td><td></td><td>65.0 90.6 327</td><td></td><td>47.368.61265</td><td>35.575.03453 25.9 65.03653</td><td>7.920.05946</td><td>7.5 26.75668 4.8 23.35435</td><td>18.4 32.0 992 14.1 30.11047</td><td>28.5 43.33768 22.339.03843</td><td>34.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>6.2 23.35951</td><td></td><td></td><td></td><td>26.5</td></tr><tr><td></td><td>Random selection</td><td>36.4 78.71173</td><td></td><td>15.840.62126</td><td>7.3 47.55092</td><td>1.0 13.35719</td><td>2.5 13.35482</td><td>4.9 23.22285</td><td>6.319.14847</td><td>10.6</td></tr><tr><td>20%</td><td>TIP</td><td>64.3 89.8</td><td>346</td><td>47.4 70.01167</td><td>25.675.03561</td><td>6.0 20.05818</td><td>3.816.75516</td><td>13.0 28.3 987 15.7 30.5931</td><td>19.4 35.53725</td><td>25.7</td></tr><tr><td></td><td>SA-OPD</td><td>68.9 91.0</td><td>367</td><td>52.970.41155</td><td>30.275.03529</td><td>5.6 16.75592</td><td>4.826.75545</td><td></td><td>23.838.63617</td><td>28.8</td></tr><tr><td>Prefix OPD</td><td>TA-OPD</td><td>42.6 83.01472</td><td></td><td>35.167.61930</td><td>18.062.54506</td><td>5.4 20.05738</td><td>4.016.75472</td><td>10.4 27.61854</td><td>16.934.94433</td><td>18.9</td></tr><tr><td>CC-OPD</td><td></td><td>54.3 86.6</td><td>459</td><td>36.6 65.61218</td><td>21.6 62.53101</td><td>4.6 20.05331</td><td>3.316.74745</td><td>10.9 29.01170</td><td>15.934.03100</td><td>21.0</td></tr><tr><td>DIAL-OPD</td><td></td><td>79.6 92.8</td><td>662</td><td>48.462.21949</td><td>28.9 62.55556</td><td>4.4 20.07717</td><td>4.816.77665</td><td>15.731.62030</td><td>20.332.25736</td><td>28.9</td></tr><tr><td>DIAL-OPD</td><td>(β=1)</td><td>82.3 91.8</td><td>404</td><td>57.8 70.21445</td><td>37.8 75.04276</td><td>8.826.76287</td><td>4.8 16.76170</td><td>17.932.71216</td><td>29.543.24222</td><td>34.1</td></tr><tr><td>TIP</td><td>(β=1.2)</td><td>81.793.5</td><td>342</td><td>57.773.01288</td><td>35.9 72.53640</td><td>7.5 20.06040</td><td>6.9 23.3 5730</td><td>17.0 30.91045</td><td>26.5 42.63911</td><td>33.3</td></tr><tr><td></td><td>Random selection</td><td>63.8 90.8</td><td>349</td><td>50.1 72.21076</td><td>28.3 70.03229</td><td>6.226.75177</td><td>5.613.34830</td><td>15.4 30.5</td><td>947 23.040.43382</td><td>27.5</td></tr><tr><td>40%</td><td></td><td>70.8 91.4</td><td>333</td><td>50.7 70.41151</td><td>29.2 72.53264</td><td>7.9 23.35789</td><td>6.520.05571</td><td>14.2 28.3 933</td><td>23.7 38.73765</td><td>29.0</td></tr><tr><td></td><td>SA-OPD</td><td>68.9 91.6</td><td>343</td><td>54.3 72.21104</td><td>30.5 72.53605</td><td>7.7 20.05885</td><td>7.7 16.75210</td><td>16.3 32.0913</td><td>25.141.23634</td><td>30.1</td></tr><tr><td>TA-OPD</td><td></td><td>9.2 31.22890</td><td></td><td>6.7 23.63040</td><td>2.7 27.56197</td><td>0.8 10.06496</td><td>0.2 3.36174</td><td>2.8 15.82957</td><td>2.7 9.96384</td><td>3.6</td></tr><tr><td>Prefix OPD CC-OPD</td><td></td><td>57.7 88.5 81.6 92.8</td><td>435 376</td><td>40.067.61364 59.2 73.21337</td><td>23.465.03994 34.2 70.04265</td><td>5.4 20.06371 8.123.36608</td><td>4.216.75823 7.526.76483</td><td>12.4 27.21256 18.2 29.41077</td><td>19.5 36.53980 28.2 42.14339</td><td>23.2 33.9</td></table>

Results. Figure 6 shows (i) shorter initial stagnation as $\beta$ increases from 0 to 1 and (ii) non-monotonic final accuracy. For the 0.6B student, accuracy reaches 35.2%, 44.6%, and a peak of 44.8% at $\beta = 0 , 1 , 1 . 2$ , respectively. For the 1.7B student, accuracy peaks at 61.8% (β = 1), versus 57.0% (β = 0) and 60.0% (β = 1.2). Appendix D compares β = 0 and β = 1 across all seven benchmarks. Probability weighting improves learning speed and final accuracy over reward-only selection, but excessive weighting reduces performance: both students peak near $\beta = 1$

## 6.2 Main Results

Better performance with less supervision. Retaining only 40% of response tokens, DIAL-OPD outperforms Vanilla OPD and both full-token variants in seven-benchmark mean accuracy across all four model pairs. Against Vanilla OPD, gains reach 5.25 percentage points in mean accuracy (4B→0.6B, β = 1) and 13.33 points on AIME25 Pass@16 (8B→1.7B, β = 1.2), doubling it from 13.33% to 26.67%. Against the strongest full-token variant, gains reach 1.24 points in mean accuracy (4B→1.7B) and 10.00 points on AMC23 Pass@16 (8B→1.7B), both at β = 1 and over PowerOPD. These results challenge supervision density as a guiding principle for effective distillation: substantially less supervision can outperform standard and improved full-token OPD.

Substantially outperforming existing token-selection methods. DIAL-OPD achieves the highest seven-benchmark mean accuracy across all eight combinations of model pair and retention ratio. At 20% retention, it exceeds the strongest existing selection method by up to 5.23 percentage points in mean accuracy (4B→1.7B, over CC-OPD) and 10.33 points on GSM8K Avg@4 (4B→0.6B, over SA-OPD), both at β = 1. Together, these comparisons demonstrate that DIAL-OPD identifies more effective supervision than existing token-selection methods at the same token budget.

Table 2: Results with a Qwen3-8B teacher on seven benchmarks. Bold and underlined mark the best and second-best accuracy per student and retention ratio, excluding full-token references.
<table><tr><td rowspan="2" colspan="2">Keep Method</td><td>GSM8K</td><td>MATH500</td><td>AMC23</td><td>AIME24</td><td>AIME25</td><td>Minerva</td><td>Olympiad</td><td>Mean</td></tr><tr><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg Pass Len</td><td>Avg</td></tr><tr><td colspan="10">Student: Qwen3-0.6B</td></tr><tr><td>100% PowerOPD</td><td>Vanilla OPD</td><td>58.4 81.1 483</td><td>39.2 56.81573</td><td>16.2 57.54488</td><td>1.7 6.76651</td><td>1.0 6.77033</td><td>9.1 21.01327</td><td>14.1 26.94864</td><td>20.0</td></tr><tr><td>ExOPD</td><td></td><td>67.2 83.7 404</td><td>43.861.21513</td><td>23.8 57.54249</td><td>2.310.06417</td><td>1.5 10.06577 6.77153</td><td>10.6 21.71163 8.7 22.81303</td><td>17.030.94415 13.526.15076</td><td>23.7</td></tr><tr><td></td><td></td><td>57.0 79.8 442</td><td>38.1 57.01568</td><td>17.7 52.54496</td><td>2.3 13.36715</td><td>0.8</td><td></td><td>3.7 11.36611</td><td>19.7 9.0</td></tr><tr><td rowspan="9">20%</td><td>Random selection</td><td>38.2 70.91314</td><td>9.927.02879</td><td>6.245.06345</td><td>0.0 0.07430</td><td>1.0 10.0 7201</td><td>3.515.82420</td><td></td><td></td></tr><tr><td>TIP</td><td>56.881.0 515</td><td>34.4 51.81706</td><td>13.1 42.55191</td><td>1.5 3.37147</td><td>0.86.77231</td><td>8.5 21.71526</td><td>12.826.35333</td><td>18.3</td></tr><tr><td>SA-OPD</td><td>59.2 79.8 463</td><td>40.0 57.01596</td><td>19.555.04603</td><td>1.5 3.36502</td><td>1.713.37003</td><td>9.4 22.81301</td><td>13.7</td><td>25.74911 20.7</td></tr><tr><td>TA-OPD</td><td>8.6 28.13109</td><td>6.019.83286</td><td>2.8 30.06722</td><td>0.4 3.36910</td><td>0.4 3.37076</td><td>1.8 10.73148</td><td>2.9 9.96795</td><td>3.3</td></tr><tr><td>Prefix OPD</td><td>49.7 77.0722</td><td>30.6 51.21889</td><td>13.450.05274</td><td>0.8 3.36911</td><td>1.213.37144</td><td>8.2 22.11597</td><td>10.2 23.35256</td><td>16.3</td></tr><tr><td>CC-OPD</td><td>54.3 73.12035</td><td>26.3 39.43219</td><td>9.7 32.57672</td><td>0.0 0.08179</td><td>1.0 3.38188</td><td>5.115.83464</td><td>7.1 14.87654</td><td>14.8</td></tr><tr><td>DIAL-OPD (β=1)</td><td>63.8 83.2 465</td><td>42.3 59.61633</td><td>21.262.54372</td><td>1.710.06566</td><td>2.1 10.06583</td><td>9.723.21265</td><td>16.029.14494</td><td>22.4</td></tr><tr><td>DIAL-OPD (β=1.2)</td><td>63.984.1 469</td><td>40.560.01701</td><td>21.6 57.54556</td><td>0.8 3.36544</td><td>1.0 10.06786</td><td>8.723.21351</td><td>13.6 25.74567</td><td>21.4</td></tr><tr><td>Random selection</td><td>55.9 79.7 627</td><td>36.4 54.01715</td><td>15.8 47.54963</td><td>1.5 3.36809</td><td>1.5 10.0 7066</td><td>8.9 22.81449</td><td>13.726.64988</td><td>19.1</td></tr><tr><td rowspan="7">40%</td><td>TIP</td><td>59.2 81.6 468</td><td>39.6 57.81587</td><td>18.4 57.54665</td><td>1.9 10.07000</td><td>0.8 10.07065</td><td>8.6 22.81344</td><td>14.526.45132</td><td>20.4</td></tr><tr><td>SA-OPD</td><td>58.8 80.7 459</td><td>39.7 56.41529</td><td>19.260.04517</td><td>2.3 10.06681</td><td>0.2 3.37151</td><td>9.419.91261</td><td>15.6 26.74739</td><td>20.7</td></tr><tr><td>TA-OPD</td><td>17.3 48.72615</td><td>12.1 34.02881</td><td>7.5 47.56181</td><td>0.8 6.76850</td><td>0.2 3.36944</td><td>3.817.62715</td><td>5.016.26151</td><td>6.7</td></tr><tr><td>Prefix OPD</td><td>54.5 79.8 627</td><td>37.056.61868</td><td>16.4 57.55483</td><td>0.6 3.37401</td><td>0.6 6.77555</td><td>9.223.91561</td><td>12.1 24.05597</td><td>18.6</td></tr><tr><td>CC-OPD</td><td>69.285.1 942</td><td>40.853.82163</td><td>23.6 52.56103</td><td>2.9 6.77683</td><td>1.7 10.07972</td><td>9.421.32089</td><td>15.7 27.46170</td><td>23.3</td></tr><tr><td>DIAL-OPD (β=1)</td><td>66.6 84.6 436</td><td>45.661.41482</td><td>27.3 62.53872</td><td>3.113.36432</td><td>1.2 6.76589</td><td>10.6 22.11197</td><td>17.7 29.74392</td><td>24.6</td></tr><tr><td>DIAL-OPD (β=1.2)</td><td>65.6 82.7 438</td><td>43.2 60.41496</td><td>26.765.04165</td><td>1.5 3.36162</td><td>1.913.36526</td><td>10.4 22.81216</td><td>18.131.04376</td><td>23.9</td></tr><tr><td colspan="10">Student: Qwen3-1.7B</td></tr><tr><td colspan="10">Vanilla OPD</td></tr><tr><td rowspan="2"></td><td>100% PowerOPD</td><td>78.7 92.5 348 80.1 92.4 327</td><td>59.5 72.81201 58.3 71.21160</td><td>34.7 77.53457 35.6 72.53343</td><td>7.5 20.06135 9.2 30.05593</td><td>4.2 13.35501 6.7 26.75118</td><td>17.9 30.5 16.8 31.2</td><td>983 27.7 42.13718 925 27.6 40.53531</td><td>32.9 33.5</td></tr><tr><td>ExOPD</td><td>80.3 92.3 337</td><td>58.7 73.01191</td><td>35.2 70.03413</td><td>9.2 26.75640</td><td>6.2 20.05323</td><td>18.0 28.7</td><td>977 27.7 41.83721</td><td>33.6</td></tr><tr><td rowspan="9">TIP</td><td>Random selection</td><td>66.8 91.0 541</td><td>49.070.01290</td><td>26.4 65.03323</td><td>6.0 16.75586</td><td>5.223.35176</td><td>15.530.91164</td><td>22.3 37.73572</td><td>27.3</td></tr><tr><td></td><td>74.9 91.8 354</td><td>54.6 71.41167</td><td>28.475.03794</td><td>10.026.75756</td><td>6.216.75284</td><td>15.7 29.8 942</td><td>24.1 38.73785</td><td>30.6</td></tr><tr><td>SA-OPD</td><td>75.9 91.4 331</td><td>58.6 73.81093</td><td>31.975.03240</td><td>6.2 20.05838</td><td>6.723.35009</td><td>17.132.0 905</td><td>25.740.73540</td><td>31.7</td></tr><tr><td>TA-OPD</td><td>80.593.3 337</td><td>58.474.41066</td><td>35.575.02977</td><td>9.0 23.35310</td><td>6.520.04810</td><td>17.932.0895</td><td>26.140.83163</td><td>33.4</td></tr><tr><td>Prefix OPD</td><td>76.1 91.2 390</td><td>52.2 66.01181</td><td>31.467.53355</td><td>5.8 23.35872</td><td>5.623.35303</td><td>17.1 31.61017</td><td>21.8 34.33528</td><td>30.0</td></tr><tr><td>CC-OPD</td><td>76.2 91.8 567</td><td>49.867.21770</td><td>29.560.05146</td><td>5.0 20.07371</td><td>5.4 16.77150</td><td>14.430.91645</td><td>21.8 34.95313</td><td>28.9</td></tr><tr><td>DIAL-OPD (β=1)</td><td>80.6 92.9 333</td><td>60.274.41199</td><td>35.8 70.03404</td><td></td><td>5.4 16.75339</td><td>18.3 31.2 967</td><td>28.242.33570</td><td></td></tr><tr><td>DIAL-OPD (β=1.2)</td><td>77.6 91.5 336</td><td>59.1 73.6 1213</td><td>34.5 72.53478</td><td>7.9 16.75791 7.526.75793</td><td>6.9 16.75678</td><td>17.3 31.6</td><td>26.4 41.83775</td><td>33.8</td></tr><tr><td>Random selection</td><td>75.5 91.6 362</td><td>55.570.81141</td><td>30.8 67.53025</td><td>9.226.75670</td><td>4.426.75339</td><td>16.5 30.1</td><td>992 983</td><td>32.8</td></tr><tr><td rowspan="9">40%</td><td>TIP</td><td>75.7 91.1 346</td><td>58.174.21127</td><td>35.075.03422</td><td>7.926.75727</td><td>4.616.75441</td><td>17.632.7 961</td><td>27.0 41.83518 26.041.83674</td><td>31.3 32.1</td></tr><tr><td>SA-OPD</td><td>76.7 91.1 330</td><td></td><td>33.0 80.03225</td><td></td><td>5.4 20.04997</td><td>17.9 30.9 887</td><td></td><td></td></tr><tr><td>TA-OPD</td><td>323</td><td>59.0 72.61153 57.9 71.61113</td><td>34.2 72.53123</td><td>7.1 20.05637 8.816.75622</td><td>6.916.74898</td><td>18.2 32.0 867</td><td>26.541.23584 27.541.13460</td><td>32.2</td></tr><tr><td></td><td>79.1 92.3 76.9 92.2 359</td><td></td><td>32.0 75.03696</td><td>7.9 20.06420</td><td>4.823.35363</td><td>16.8 30.11011</td><td>22.1 34.93839</td><td>33.2</td></tr><tr><td>Prefix OPD</td><td>346</td><td>54.3 67.41199</td><td>35.8 72.53800</td><td></td><td>5.623.36121</td><td>17.6 30.5 982</td><td></td><td>30.7</td></tr><tr><td>CC-OPD</td><td>79.9 92.5</td><td>59.2 73.21245</td><td></td><td>7.9 23.36323</td><td></td><td></td><td>27.4 40.44047</td><td>33.3</td></tr><tr><td>DIAL-OPD (β=1)</td><td>82.493.5 323</td><td>59.7 73.81173</td><td>38.0 82.53098</td><td>8.8 16.75764</td><td>6.5 23.35425</td><td>18.0 31.6 913</td><td>28.9 43.5 3533</td><td>34.6</td></tr><tr><td>DIAL-OPD (β=1.2)</td><td>82.1 92.9 331</td><td>59.7 72.41156</td><td>36.4 77.53134</td><td>9.2 20.05823</td><td>6.026.75306</td><td>17.1 28.7 898</td><td>27.8 42.43536</td><td>34.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Effective supervision allocation can outweigh teacher scaling. DIAL-OPD with a 4B teacher outperforms full-token OPD with an 8B teacher at both student scales. With β = 1 and 40% retention, it achieves mean accuracies of 24.68% and 35.36% for the 0.6B and 1.7B students, respectively, exceeding not only Vanilla OPD but also the strongest full-token variant under the larger teacher by 0.95 and 1.76 percentage points. These findings suggest that allocating supervision effectively can yield greater gains than simply increasing teacher size.

## 6.3 Understanding DIAL-OPD's Selection Behavior

Moderate $\beta$ achieves the best balance in token selection. We examine how $\beta _ { i }$ the dial between log-probability and probability spaces, shapes token selection. We compare $\beta = 0 , 0 . 5 , 1 , 2$ on the same 128 responses to DeepScaleR prompts at 20% retention. Figure 7 reveals two stages: (i) increasing $\beta$ from 0 to 1 shifts selection away from the low-low region, reducing its share of selected tokens from 8.26% to 0.29%; (ii) increasing $\beta$ to 2 suppresses this share to 0.02%, but also reduces selection in regions where one model assigns substantial probability while the other does not. Together with the performance trends in Section 6.1, these observations suggest that moderate probability weighting improves performance by suppressing low-low disagreement, whereas excessive weighting may degrade performance by discarding useful signals supported by one model.

![](images/fe71321a1eea1076fe5dfc8147de6472702d8605bdf45b6d15ea94cdf9d4d3a3.jpg)  
Student probability p

Figure 7: Token selection under varying β at 20% retention. Hexagon colors indicate selection rates; dotted lines delimit the low-low region, and teacher probabilities $q \leq 1 0 ^ { - 6 }$ are pooled for visualization.  
![](images/e99424db1e20d04026c79acf923b1823b2013a18240106ac3909ff170ffd6f0c.jpg)  
Figure 8: Token selections and signed OPD rewards in context (β = 1, 20% retention).Blue/pinkmark retained tokens with positive/negative rewards; graymarks filtered tokens. Left: low-low zoom on log axes.

Reward magnitude alone is insufficient to assess learning value. Figure 8 shows retained and filtered tokens in context (β = 1, 20% retention). DIAL-OPD filters the unrelated identifier serialVersionUID (-28.13) and Therefore (+4.52), a redundant transition to an already-obtained answer. Retained supervision instead encourages Area in a correct incircle formula (+0.94) and penalizes an incorrect arithmetic digit (−0.81). Tokens that do not advance reasoning can receive larger reward magnitudes than those providing useful supervision. This contrast supports using both relative disagreement and absolute probability scale for selection.

## 7 Conclusion

We show that more supervision does not necessarily improve OPD, motivating selective supervision. Yet prioritizing large teacher-student disagreements can overemphasize harmful low-low tokens. DIAL-OPD incorporates absolute probability scale through the logarithmic mean to address this blind spot. Selection analyses suggest that moderate probability weighting suppresses low-low tokens while preserving useful disagreements, whereas excessive weighting can discard valuable supervision. Across four teacher-student pairs and seven benchmarks, it outperforms full-token OPD with fewer tokens and existing selection methods at matched retention ratios. Effective supervision allocation can even outweigh teacher scaling, underscoring the importance of token selection.

## References

[1] Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes, 2024. URL https://arxiv.org/abs/2306.13649

[2] Rajendra Bhatia. The logarithmic mean. Resonance, 13(6):583–594, 2008. doi: 10.1007/s1 2045-008-0063-4. URL https://repository.ias.ac.in/2562/.

[3] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

[4] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948, 2025. URL https://arxiv.org/abs/2501.12948.

[5] DeepSeek-AI. Deepseek-v4.1-flash: Pushing the limits of kv cache compression, 2026. URL https://arxiv.org/abs/2609.19969.

[6] Jasper Dekoninck, Nikola Jovanović, Tim Gehrunger, Kári Rögnvaldsson, Ivo Petrov, Chenhao Sun, and Martin Vechev. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms, 2026. URL https://arxiv.org/abs/2605.00674.

[7] Shiyuan Feng, Huan ang Gao, Haohan Chi, Hanlin Wu, Zhilong Zhang, Zheng Jiang, Bingxiang He, Wei-Ying Ma, Ya-Qin Zhang, and Hao Zhou. Weak-to-strong generalization via direct on-policy distillation, 2026. URL https://arxiv.org/abs/2607.05394.

[8] Zixuan Fu, Bingxiang He, Yuxin Zuo, Haohuan Huang, Jinqian Zhang, Ruhang Xiao, Cheng Qian, Qinyu Luo, Huan ang Gao, Yudong Wang, Zhiyuan Liu, Ning Ding, and Chaojun Xiao. Rethinking on-policy distillation of large language models ii: One training example, 2026. URL https://arxiv.org/abs/2609.04172.

[9] GLM-5 Team. GLM-5: From vibe coding to agentic engineering. arXiv:2602.15763, February 2026. URL https://arxiv.org/abs/2602.15763.

[10] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: On-policy distillation of large language models, 2026. URL https://arxiv.org/abs/2306.08543.

[11] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL https://arxiv.org/abs/2402.14008.

[12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

[13] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration, 2020. URL https://arxiv.org/abs/1904.09751.

[14] ZhiYan Hou, Xinyu Tang, Hongyan An, Jianjin Zhang, Weizhen Wang, Yunyun Han, Gengsheng Li, Xiangzhao Hao, Haiyun Guo, Wenbin Hu, Jinqiao Wang, and Yafeng Deng Dash: Divergence-adaptive supervision horizons for on-policy self-distillation of reasoning models, 2026. URL https://arxiv.org/abs/2608.06243.

[15] Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. Reinforcement learning via self-distillation, 2026. URL https://arxiv.org/abs/2601 .20802.

[16] Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggyu Lim, and Taesup Kim. Stable on-policy distillation through adaptive target reformulation, 2026. URL https://arxiv.org/abs/2601.0 7155.

[17] Nan Jia, Haojin Yang, Xing Ma, Jiesong Lian, Shuailiang Zhang, Weipeng Zhang, Ke Zeng, Xunliang Cai, and Zequn Sun. Asymmetric on-policy distillation: Bridging exploitation and imitation at the token level, 2026. URL https://arxiv.org/abs/2605.06387.

[18] Yinuo Jiang, Yongjie Ye, Zhou Tao, Xiang Zhuang, Qiang Zhang, Huajun Chen, and Tiankai Li. When teachers mislead: Spurious-signal-aware on-policy distillation, 2026. URL https://arxiv.org/abs/2608.03632.

[19] Yuxuan Jiang and Francis Ferraro. Bridging reasoning trajectories in on-policy distillation via near-future guidance, 2026. URL https://arxiv.org/abs/2606.00305

[20] Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models, 2026. URL https://arxiv.org/abs/2603.07079.

[21] Kimi Team. Kimi K3: Open frontier intelligence. arXiv:2607.24653, July 2026. URL https://arxiv.org/abs/2607.24653.

[22] Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation, 2026. URL https://arxiv.org/abs/26 03.11137.

[23] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022. URL https://arxiv.org/abs/2206.14858.

[24] Enhan Li, Junhao He, and Hongyang Du. Crop: Task relevance via counterfactuals for selective on-policy distillation, 2026. URL https://arxiv.org/abs/2608.13387.

[25] Yuying Li, Leqi Zheng, Yongzi Yu, Wenrui Zhou, Xuchang Zhong, Xing Hu, Jing Jin, Hangjie Yuan, and Tao Feng. Filter, then reweight: Rethinking optimization granularity in on-policy distillation, 2026. URL https://arxiv.org/abs/2606.02684.

[26] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step, 2023. URL https://arxiv.org/abs/2305.20050.

[27] Zhishuai Liu, Xingzi Xu, Mehmet Saygin Seyfioglu, Pan Xu, and Karim Bouyarmane. Extremely sparse supervision incentivizes reasoning ability, 2026. URL https://arxiv.org/ab s/2609.04565.

[28] Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/onpolicy-distillation.

[29] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/DeepScale R-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca3 03013a4e2, 2025. Notion Blog.

[30] Qwen Team. Qwen3 technical report. arXiv:2505.09388, 2025. URL https://arxiv.org/abs/ 2505.09388.

[31] Bing Shao, Jiazheng Zhang, Long Ma, Yujiong Shen, Senjie Jin, Xin Guo, Yuming Yang, Mingxu Chai, Zhiheng Xi, Boyang Liu, Junlin Shang, Tao Gui, Qi Zhang, and Xuanjing Huang. A token-level analysis of sampled-token reverse-kl on-policy distillation, 2026. URL https://arxiv.org/abs/2608.25643.

[32] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

[33] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. Solla, T. Leen, and K. Müller, editors, Advances in Neural Information Processing Systems, volume 12. MIT Press, 1999. URL https://proceedings.neurips.cc/paper\_files/paper/1999/file/464d828b85 b0bed98e80ade0a5c43b0f-Paper.pdf.

[34] Qitai Tan, Zefang Zong, Mo Li, Yipeng Shi, Yang Li, and Peng Chen. Atod: Annealed turn-aware on-policy distillation for multi-turn agentic tasks, 2026. URL https://arxiv.org/ abs/2606.27814.

[35] Yuanyi Wang, Su Lu, Yanggan Gu, Pengkai Wang, Yifan Yang, Zhaoyi Yan, Congkai Xie, Jianmin Wu, and Hongxia Yang. Not all disagreement is learnable: Token teachability in on-policy distillation, 2026. URL https://arxiv.org/abs/2605.26844.

[36] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992. doi: 10.1007/BF00992696. URL https://doi.org/10.1007/BF00992696.

[37] Xiaomi LLM-Core Team. MiMo-V2-Flash technical report, 2026. URL https://arxiv.org/ab s/2601.02780.

[38] Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, and Alborz Geramifard. Tip: Token importance in on-policy distillation, 2026. URL https://arxiv.org/abs/2604.140 84.

[39] Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation, 2026. URL https://arxiv.org/abs/2602.12125.

[40] Yongkang Yang, Zhezheng Hao, Hong Zhang, Yi Liu, Xiankun Lin, Wence Ji, Fanjunduo Wei, Jiarui Yu, Qiang Lin, Xiaoyun Liang, and Hande Dong. Matching supervision to the student's learning capacity: A unified framework for on-policy self-distillation, 2026. URL https://arxiv.org/abs/2608.08176.

[41] Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, and Difan Zou. Mismatch matters: On-policy distillation beyond token agreement, 2026. URL https://arxiv.org/abs/2608.09836.

[42] Dongxu Zhang, Zhichao Yang, Sepehr Janghorbani, Jun Han, Andrew Ressler II, Qian Qian, Gregory D. Lyng, Sanjit Singh Batra, and Robert E. Tillman. Fast and effective on-policy distillation from reasoning prefixes, 2026. URL https://arxiv.org/abs/2602.15260.

[43] Anhao Zhao, Junlong Tong, Yingqi Fan, Ping Nie, Wenjie Li, and Xiaoyu Shen. Poweropd: Stabilizing on-policy distillation with bounded power transformation, 2026. URL https: //arxiv.org/abs/2606.17199.

[44] Anhao Zhao, Haoran Xin, Yingqi Fan, Junlong Tong, Wenjie Li, and Xiaoyu Shen. Decoupling kl and trajectories: A unified perspective for sft, dagger, offline rl, and opd in llm distillation, 2026. URL https://arxiv.org/abs/2605.16826.

[45] Qingfei Zhao, Huan Song, Shuyu Tian, Jiawei Shao, and Xuelong Li. Prefix-guided on-policy distillation: Mining golden trajectories from rollouts, 2026. URL https://arxiv.org/abs/26 06.21994.

[46] Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026. URL https://arxiv.org/abs/2601.18734.

[47] Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contributors. slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025. GitHub repository. Corresponding author: Xin Lv.

## Appendix

## A Related Work

On-policy distillation. OPD trains a student on its own generated responses using teacher supervision, reducing the mismatch between training and inference. GKD [1] studies flexible divergence objectives on student-generated sequences, MiniLLM [10] develops reverse-KL optimization, and Lu and Lab [28] present a practical sampled-token training recipe. Subsequent developments address two complementary questions. (i) Constructing teacher supervision. Qwen3 [30] transfers capabilities from larger teachers through off-policy initialization followed by on-policy distillation; MiMo-V2-Flash [37] integrates domain-specialized teachers; GLM-5 [9] recovers capabilities from earlier training checkpoints; and Kimi K3 [21] consolidates experts specialized by domain and reasoning effort. Self-distillation instead constructs a teacher from the model itself: OPSD [46] provides it with solutions unavailable to the student, while SDPO [15] conditions it on textual feedback or successful attempts. (ii) Optimizing the distillation objective. ExOPD [39] separates reward strength from KL regularization, while Direct-OPD [7] derives rewards from the teacher's policy change during reinforcement learning. PowerOPD [43] bounds rewards through power transformations, and Veto [16] stabilizes alignment through a geometric target distribution combining teacher and student probabilities. AOPD [17] uses policy-gradient reinforcement for positive advantages and local distribution matching otherwise; EOPD [20] adds forward-KL supervision where teacher entropy is high. Recent evidence also highlights a bottleneck in supervision utilization: Fu et al. [8] show that training on a single query recovers much of full-data OPD's improvement while teacher-student alignment remains slow.

Selective supervision for OPD. Selective supervision addresses which parts of generated responses should contribute to learning by selecting, weighting, or restructuring teacher supervision. Two complementary approaches consider individual tokens and the structure of the surrounding trajectory. (i) Assessing token learning value. TIP [38] combines student uncertainty with teacher-student divergence, while REOPOLD [22] couples entropy-guided token sampling with reward clipping. SuRe [31] uses bounded surprise-based weighting to emphasize tokens assigned low probability by the student, and TA-OPD [35] assesses disagreement together with teacher support for the student's likely alternatives. Beyond distributional statistics, SA-OPD [18] filters high-impact supervision weakly dependent on the question, whereas CROP [24] measures sensitivity to changed task conditions relative to meaning-preserving rewrites. USD [40] jointly adjusts token weights and the information supplied to a self-teacher according to student learning capacity. TIDE [41] applies different corrections to tokens overestimated by the student and alternatives favored by the teacher. (ii) Using trajectory structure. Prefix OPD [42] trains on truncated reasoning prefixes, and PG-OPD [45] uses teacher-student agreement on prefixes to decide which rollouts to extend. FiRe-OPD [25] filters trajectories by teacher likelihood before weighting their tokens, while TOPD [19] uses near-future continuations to guide corrections. DASH [14] aggregates divergence across successive positions to construct token weights; ATOD [34] weights interaction turns while gradually shifting from distillation to reinforcement learning. Evidence from extremely sparse OPD further shows that supervising one or two selected tokens per trajectory can match or exceed dense training [27]. Building on these findings, we identify how relative disagreement can overemphasize tokens assigned low probability by both models and test their training value through targeted replacement. DIAL-OPD uses the logarithmic mean to connect relative disagreement with absolute probability scale, selecting a fixed fraction of tokens within each response while preserving the original signed OPD reward.

## B Experimental Details

Models. Students are initialized from Qwen3-0.6B-Base and Qwen3-1.7B-Base, while teachers use the instruction-tuned Qwen3-4B and Qwen3-8B checkpoints. The equal-size comparison in Section 3.1 additionally uses Qwen3-4B-Base as the student. We omit the “-Base" suffix in the main text for brevity.

Training details. All methods use AdamW with BF16 mixed precision, learning rate $5 \times 1 0 ^ { - 7 }$ and batch size 32. Each rollout contains up to 1,024 response tokens. Training rollouts use temperature 1.0, top-p 0.95, and top-k 20. Prompt, padding, and terminal EOS positions are excluded from the training loss for all methods.

Evaluation protocol. For the main comparisons, we sample k = 4 responses per question on GSM8K, MATH-500, and OlympiadBench; k = 8 on Minerva Math; and k = 16 on AMC23, AIME24, and AIME25. Sampling uses temperature 0.6 and top-p 0.95. Tables abbreviate average correctness, success in at least one sampled response, and mean response length as Avg, Pass, and Len, respectively. Mean denotes the unweighted average of Avg across the seven benchmarks.

## B.1 Supervision Density Experiment

Training follows the configuration in Section 5. For each response, we exclude prompt, padding, and terminal EOS positions, leaving N eligible tokens. We uniformly sample K = max(1, round(ρN)) positions without replacement, where ρ is the retention ratio. Selection masks are refreshed at every training step. Only selected positions contribute to the original signed OPD loss, averaged over all selected tokens in the batch; unselected tokens remain in the context. We evaluate on MATH-500 before training and every 50 steps, sampling one response per question with temperature 0.6, top-p 0.95, and a maximum response length of 4,096 tokens. The same-size experiment repeats the retention-ratio sweep with Qwen3-4B-Base as the student and Qwen3-4B as the teacher.

## B.2 Baseline Implementations

We adapt token-selection baselines to matched per-response retention ratios. For SA-OPD, we select tokens by input-dependence ranking instead of its original joint filtering conditions. For TA-OPD, teacher support measures the probability mass that the teacher assigns to the student's top-K candidate tokens [35]; we retain batch-level normalization and perform selection within each response. Prefix OPD selects the initial fraction of each complete rollout rather than stopping generation early. For CC-OPD, the teacher's pre-RL checkpoint is unavailable, so we use a smaller instruction-tuned model from the same family as the reference and select tokens with the largest absolute teacher-reference probability gaps.

## B.3 Low-Low Token Analysis

For the concentration analysis in Section 3.3, we pool selected-token counts and absolute reward sums across responses before computing low-low shares, rather than averaging per-response percentages. The token share divides the selected low-low count by the total selected count; the reward share divides their summed absolute rewards by the absolute reward sum over all selected tokens. Replacement candidates are ranked by absolute reward, while the OPD loss retains the original reward signs. Both replacement strategies use training and evaluation settings matched to the corresponding Reward-Top baseline at each retention ratio.

## C Derivation of the Sampled-Token OPD Update

We derive the update in Eq. (2) by differentiating the conditional reverse KL while holding the rollout context distribution fixed. The derivation also establishes the unbiasedness of its Monte Carlo estimator and explains the stop-gradient treatment of the token-level reward.

Fixed-context objective. Let $\bar { \theta }$ denote the student parameters used to collect rollouts, and let $\theta$ denote the parameters with respect to which we differentiate. For $x \sim \mathcal { D }$ and $y \sim \pi _ { \bar { \theta } } ( \cdot \mid x )$ the context preceding token $y _ { t }$ is $c _ { t } = ( x , y _ { < t } )$ . With the teacher policy fixed, define

$$
\mathcal { L } _ { \mathrm { c t x } } ( \theta ; \bar { \theta } ) = \mathbb { E } _ { { x } \sim \mathcal { D } , { y } \sim \pi _ { \bar { \theta } } ( \cdot \vert x ) } \left[ \sum _ { { t = 1 } } ^ { \vert { y } \vert } D _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot \vert c _ { t } ) \Vert \pi _ { T } ( \cdot \vert c _ { t } ) ) \right] .\tag{C.1}
$$

Here, $\theta$ is held fixed during differentiation: the contexts come from the rollout policy, while the conditional student distributions at those contexts depend on θ. We evaluate the resulting gradient at $\theta = \bar { \theta }$ We assume differentiable, strictly positive student and teacher probabilities over a finite vocabulary, bounded rollout lengths, and the regularity conditions required to interchange differentiation and expectation.

From reverse KL to a policy-gradient update. For a fixed context $^ { c , }$ write $p _ { \theta } ( v ) = \pi _ { \theta } ( v \mid c )$ and $q ( v ) = \pi _ { T } ( v \mid c )$ . Differentiating the vocabulary-wide reverse KL gives

$$
\begin{array} { l } { \displaystyle \nabla _ { \theta } D _ { \mathrm { K L } } ( p _ { \theta } \| q ) = \nabla _ { \theta } \sum _ { v \in \mathcal { V } } p _ { \theta } ( v ) \log \frac { p _ { \theta } ( v ) } { q ( v ) } } \\ { \displaystyle = \sum _ { v \in \mathcal { V } } \left( \log \frac { p _ { \theta } ( v ) } { q ( v ) } + 1 \right) \nabla _ { \theta } p _ { \theta } ( v ) } \\ { \displaystyle = \sum _ { v \in \mathcal { V } } p _ { \theta } ( v ) \log \frac { p _ { \theta } ( v ) } { q ( v ) } \nabla _ { \theta } \log p _ { \theta } ( v ) . } \end{array}\tag{C.2}
$$

The last equality uses $\nabla _ { \boldsymbol { \theta } } p _ { \boldsymbol { \theta } } ( v ) = p _ { \boldsymbol { \theta } } ( v ) \nabla _ { \boldsymbol { \theta } }$ log $p _ { \theta } ( v )$ and the normalization identity

$$
\sum _ { v \in \mathcal { V } } \nabla _ { \theta } p _ { \theta } ( v ) = \nabla _ { \theta } \sum _ { v \in \mathcal { V } } p _ { \theta } ( v ) = \nabla _ { \theta } 1 = 0 .\tag{C.3}
$$

Consequently, the negative gradient is an expectation of the teacher-student log-ratio multiplied by the student's log-probability gradient. Averaging this identity over the fixed rollout context distribution and evaluating at the rollout policy yields

$$
\begin{array} { r l } & { g _ { \mathrm { O P D } } ( \bar { \theta } ) : = - \left. \nabla _ { \theta } \mathcal { L } _ { \mathrm { c t x } } ( \theta ; \bar { \theta } ) \right| _ { \theta = \bar { \theta } } } \\ & { = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \bar { \theta } } ( \cdot | x ) } \left[ \underbrace { | y | } _ { t = 1 } \underbrace { \log \frac { \pi _ { T } ( y _ { t } \mid c _ { t } ) } { \pi _ { \bar { \theta } } ( y _ { t } \mid c _ { t } ) } } _ { r _ { t } ^ { \mathrm { O P D } } } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { t } \mid c _ { t } ) | _ { \theta = \bar { \theta } } \right] . } \end{array}\tag{C.4}
$$

This is the REINFORCE policy-gradient form, with the log-ratio serving as the token-level reward [33, 36]. Removing the distinction between the differentiation variable and the rollout parameters after evaluation recovers the notation of Eq. (2).

Monte Carlo estimation and stop-gradient implementation. Given B independent prompt-response pairs sampled from $\mathcal { D }$ and $\pi _ { \bar { \theta } }$ , let $c _ { i , t }$ denote the context of token $y _ { i , t }$ , and let $r _ { i , t } ^ { \mathrm { O P D } }$ be its log-ratio reward evaluated at ${ \bar { \theta } } .$ The sampled update is

$$
\widehat { \boldsymbol { g } } _ { B } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { t = 1 } ^ { | y _ { i } | } r _ { i , t } ^ { \mathrm { O P D } } \operatorname { V } _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( y _ { i , t } \mid c _ { i , t } ) | _ { \boldsymbol { \theta } = \boldsymbol { \bar { \theta } } } , \qquad \mathbb { E } [ \widehat { g } _ { B } ] = g _ { \mathrm { O P D } } ( \boldsymbol { \bar { \theta } } ) .\tag{C.5}
$$

Each sampled token supplies one Monte Carlo sample of the conditional expectation in Eq. (C.2). Tokens within a response need not be independent: conditioned on its prefix, each token is sampled from the student distribution, and unbiasedness follows from conditional expectation and linearity. In implementation, the same update is obtained by minimizing

$$
\widehat { \ell } _ { \mathrm { O P D } } ( \boldsymbol { \theta } ; \bar { \boldsymbol { \theta } } ) = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \sum _ { t = 1 } ^ { | y _ { i } | } \mathrm { s g } \big [ r _ { i , t } ^ { \mathrm { O P D } } \big ] \log \pi _ { \boldsymbol { \theta } } ( y _ { i , t } \mid c _ { i , t } ) ,\tag{C.6}
$$

where sg denotes the stop-gradient operator. By construction,

$$
- \left. \nabla _ { \theta } \widehat { \ell } _ { \mathrm { O P D } } ( \theta ; \bar { \theta } ) \right| _ { \theta = \bar { \theta } } = \widehat { g } _ { B } .\tag{C.7}
$$

The stop-gradient operation implements the coefficient obtained from the analytical KL gradient; it does not assume that the original log-ratio is independent of the student parameters. Its explicit parameter dependence has already been accounted for in Eq. (C.2), where the additional term cancels by normalization. Thus, Eq. (C.6) is a surrogate loss that reproduces the required gradient at the rollout policy, rather than a direct differentiation of the sampled log-ratio alone.

Relation to the full on-policy objective. At $\theta = { \bar { \theta } } ,$ the fixed-context objective in Eq. (C.1) has the same value as the objective in Eq. (1), but their derivatives need not coincide. Differentiating Eq. (1) through the entire trajectory distribution additionally accounts for how earlier token choices change the contexts encountered later in the response. Equation (2) instead uses the conditional reverse-KL gradients at the sampled contexts, following the immediate-token update used in on-policy distillation [28]. The unbiasedness statement in $\operatorname { E q }$ . (C.5) therefore concerns this fixed-context update direction. Contexts are nevertheless refreshed by new student rollouts during training; holding their distribution fixed during differentiation does not make the training procedure off-policy.

## D Probability Weighting Ablation

To isolate the contribution of probability weighting, Table 3 compares DIAL-OPD at $\beta = 0$ and $\beta = 1$ under the same retention ratio across all four teacher-student pairs. The $\beta = 0$ setting corresponds to Reward-Top in Section 3.2. With a Qwen3-4B teacher and Qwen3- 1.7B-Base student at 40% retention, $\beta = 1$ improves seven-benchmark mean accuracy by 6.96 percentage points and GSM8K Avg@4 by 15.96 points over $\beta = 0$ . These results extend the learning-dynamics analysis in Section 6.1 to the full evaluation suite.

## E Proofs of DIAL-OPD Properties

## E.1 Proof of Proposition 1

Fix $p , q \in ( 0 , 1 ] , \beta \geq 0$ , and $\varepsilon \in ( 0 , 1 ]$ . Scaling both probabilities by ε preserves their ratio while scaling their logarithmic mean by the same factor. From the integral definition in Equation 3,

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \varepsilon p , \varepsilon q ) = \int _ { 0 } ^ { 1 } ( \varepsilon p ) ^ { 1 - u } ( \varepsilon q ) ^ { u } d u } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { 1 } \varepsilon ^ { 1 - u + u } p ^ { 1 - u } q ^ { u } d u = \varepsilon L ( p , q ) . } } \end{array}\tag{E.8}
$$

Table 3: Probability-weighting ablation across four teacher-student pairs and seven benchmarks. Each retention ratio compares reward-only selection $( \beta = 0 )$ with $\beta = 1 ;$ bold marks the higher accuracy within each pair, including ties.
<table><tr><td>Keepβ</td><td colspan="3">GSM8K</td><td colspan="2">MATH500</td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">Minerva</td><td colspan="2">Olympiad</td><td>Mean</td></tr><tr><td></td><td>Avg Pass</td><td>Len</td><td>Avg</td><td>Pass Len</td><td>Avg Pass</td><td>Len</td><td>Avg Pass</td><td>Len</td><td>Avg Pass</td><td>Len</td><td>Avg Pass</td><td>Len</td><td>Avg Pass</td><td></td><td>Len</td><td>Avg</td></tr><tr><td colspan="17">Teacher: Qwen3-4B V Student: Qwen3-0.6B-Base</td></tr><tr><td>20%</td><td>0 57.3</td><td>79.2</td><td>535</td><td>36.4 52.4 1786</td><td>15.9</td><td>57.5 5254</td><td>1.7</td><td>10.0 7297</td><td>0.8</td><td>6.7</td><td>7685 8.5</td><td>20.6</td><td>1593</td><td>12.9 24.6</td><td></td><td>5419 19.1</td></tr><tr><td></td><td>1 67.0</td><td>81.3</td><td>1072</td><td>34.2 47.2</td><td>2609 15.9</td><td>45.0</td><td>6554 0.4</td><td>6.7</td><td>7985 1.5</td><td>10.0</td><td>8016 7.6</td><td>16.9</td><td>2617</td><td>13.1 24.2</td><td>6658</td><td>19.9</td></tr><tr><td>40%</td><td>0 56.1</td><td>81.3</td><td>507</td><td>34.1 52.4</td><td>1752 15.8</td><td>52.5 4996</td><td>1.9</td><td>3.3</td><td>7180 0.4</td><td>6.7</td><td>7516 8.0</td><td>21.0</td><td>1504</td><td>12.4 24.5</td><td>5495</td><td>18.4</td></tr><tr><td>1</td><td>69.1 84.7</td><td>578</td><td>44.9</td><td>61.6 1714</td><td>25.5</td><td>50.0 4774</td><td>2.5</td><td>13.3</td><td>7001 1.2</td><td>6.7</td><td>7179 10.7</td><td>22.1</td><td>1484</td><td>18.8 33.5</td><td>4705</td><td>24.7</td></tr><tr><td colspan="11">Teacher: Qwen3-4B → Student: Qwen3-1.7B-Base</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>20% 0</td><td>69.4</td><td>90.8</td><td>354 53.1</td><td>71.6</td><td>1149 30.0</td><td>75.0 3476</td><td>6.0 20.0</td><td>5589</td><td>4.2</td><td>16.7 5478</td><td>14.7 29.4</td><td>943</td><td>24.0</td><td>39.2</td><td>3568 28.8</td></tr><tr><td></td><td>1 82.3</td><td>91.8</td><td>404 57.8</td><td>70.2 1445</td><td>37.8</td><td>75.0 4276</td><td>8.8 26.7</td><td>6287</td><td>4.8 16.7</td><td>6170</td><td>17.9 32.7</td><td>1216</td><td>29.5</td><td>43.2 4222</td><td>34.1</td></tr><tr><td>40%</td><td>0 67.2</td><td>91.9</td><td>363 51.6</td><td>72.0 1124</td><td>27.7 82.5</td><td>3272</td><td>8.3 30.0</td><td>5530</td><td>5.4 13.3</td><td>5303</td><td>14.3 29.0</td><td>872</td><td>24.3</td><td>41.1 3393</td><td>28.4</td></tr><tr><td>1</td><td>83.2</td><td>92.8 344</td><td>61.5</td><td>73.0 1223</td><td>38.8 75.0</td><td>3457</td><td>9.8 26.7</td><td>5992</td><td>5.2 16.7</td><td>5951</td><td>19.4 30.9</td><td>992</td><td>29.7 44.5</td><td>3789</td><td>35.4</td></tr><tr><td colspan="10">Teacher: Qwen3-8B → Student:</td><td colspan="8">Qwen3-0.6B-Base</td></tr><tr><td>20% 0</td><td>58.2</td><td>80.4</td><td>471 39.2</td><td>57.4 1568</td><td>15.8</td><td>57.5 4527</td><td>1.5 3.3</td><td>6626</td><td>0.2</td><td>3.3 7127</td><td>8.0 20.6</td><td>1402</td><td>13.8</td><td>25.8 5025</td><td>19.5</td></tr><tr><td></td><td>1 63.8</td><td>83.2</td><td>465 42.3</td><td>59.6 1633</td><td>21.2 62.5</td><td>4372</td><td>1.7 10.0</td><td>6566</td><td>2.1 10.0</td><td>6583</td><td>9.7 23.2</td><td>1265</td><td>16.0</td><td>29.1 4494</td><td>22.4</td></tr><tr><td>40%</td><td>0 58.8</td><td>80.5</td><td>488 40.9</td><td>57.6 1562</td><td>16.9 57.5</td><td>4697</td><td>1.9 13.3</td><td>6521</td><td>1.2 10.0</td><td>7080</td><td>9.9 22.8</td><td>1276</td><td>14.4</td><td>27.2 4967</td><td>20.6</td></tr><tr><td></td><td>1 66.6</td><td>84.6</td><td>436 45.6</td><td>61.4 1482</td><td>27.3 62.5</td><td>3872</td><td>3.1 13.3</td><td>6432</td><td>1.2</td><td>6.7 6589</td><td>10.6 22.1</td><td>1197</td><td>17.7 29.7</td><td>4392</td><td>24.6</td></tr><tr><td colspan="10">Teacher: Qwen3-8B → Student: Qwen3-1.7B-Base</td><td colspan="8"></td></tr><tr><td>20% 0</td><td>74.8</td><td>91.7 345</td><td>58.2</td><td>73.8</td><td>1116 31.4 72.5</td><td>3309</td><td>7.5 16.7</td><td>6003</td><td>6.5 16.7</td><td>5179</td><td>16.3 30.1</td><td>931</td><td>27.0</td><td>42.0 3494</td><td>31.7</td></tr><tr><td>40%</td><td>1 80.6</td><td>92.9 333</td><td>60.2</td><td>74.4</td><td>1199 35.8 70.0</td><td>3404</td><td>7.9 16.7</td><td>5791</td><td>5.4 16.7</td><td>5339</td><td>18.3 31.2</td><td>967</td><td>28.2</td><td>42.3</td><td>3570 33.8</td></tr><tr><td></td><td>0 80.0</td><td>92.4</td><td>347 58.1</td><td>71.8</td><td>1183 33.1</td><td>80.0 3262</td><td>7.3 20.0</td><td>5828</td><td>5.6 20.0</td><td>5390 5425</td><td>17.5 32.7 18.0 31.6</td><td>991</td><td>26.5</td><td>40.2 3693</td><td>32.6</td></tr><tr><td></td><td>1 82.4</td><td>93.5</td><td>323 59.7</td><td>73.8 1173</td><td>38.0 82.5</td><td>3098</td><td>8.8 16.7</td><td>5764</td><td>6.5 23.3</td><td></td><td></td><td>913</td><td>28.9</td><td>43.5 3533</td><td>34.6</td></tr></table>

Meanwhile, $| \mathrm { l o g } [ ( \varepsilon q ) / ( \varepsilon p ) ] | = | \log ( q / p ) |$ . Substituting these identities into Equation 4 yields

$$
\begin{array} { c } { { s _ { \beta } ( \varepsilon p , \varepsilon q ) = L ( \varepsilon p , \varepsilon q ) ^ { \beta } \left| \log \displaystyle \frac { \varepsilon q } { \varepsilon p } \right| } } \\ { { = [ \varepsilon L ( p , q ) ] ^ { \beta } \left| \log \displaystyle \frac { q } { p } \right| = \varepsilon ^ { \beta } s _ { \beta } ( p , q ) . } } \end{array}\tag{E.9}
$$

This identity also holds when $p = q .$ since both scores are zero. For $p \neq q .$ , the score is positive; hence $\varepsilon < 1$ leaves it unchanged at $\beta = 0$ and strictly reduces it at $\beta > 0$ . Larger $\beta$ produces a smaller scaling factor $\varepsilon ^ { \beta }$ , establishing stronger suppression under the same proportional reduction in both probabilities.

## E.2 Proof of Proposition 2

Fix $a \in \left( 0 , 1 \right]$ and $\beta \geq 0$ , and let $b  0 ^ { + }$ . We prove the limit of the selection score from Equation 4 when one probability vanishes and the other remains fixed. Since $a > 0$ , we may restrict attention to $0 < b < a$ without affecting the limit. Both probabilities are strictly positive throughout the derivation; the score need not be defined at $b = 0$

Symmetry. The logarithmic mean and the absolute log-ratio are symmetric in their arguments. In particular,

$$
L ( a , b ) = { \frac { a - b } { \log a - \log b } } = { \frac { b - a } { \log b - \log a } } = L ( b , a ) , \qquad \left| \log { \frac { a } { b } } \right| = \left| \log { \frac { b } { a } } \right| .\tag{E.10}
$$

It follows that $s _ { \beta } ( a , b ) = s _ { \beta } ( b , a )$ for every positive $a , b .$ Thus, the same limit applies whether the student probability or the teacher probability approaches zero, and it suffices to analyze $s _ { \beta } ( a , b )$

Asymptotic form. For $0 < b < a$ , both $a - b$ and $\log ( a / b )$ are positive. Substituting the closed form of the logarithmic mean into the score gives the exact identity

$$
\begin{array} { l } { { \displaystyle s _ { \beta } ( a , b ) = L ( a , b ) ^ { \beta } \left| \log \frac { a } { b } \right| } } \\ { { \displaystyle ~ = \left( \frac { a - b } { \log ( a / b ) } \right) ^ { \beta } \log ( a / b ) } } \\ { { \displaystyle ~ = ( a - b ) ^ { \beta } [ \log ( a / b ) ] ^ { 1 - \beta } } . } \end{array}\tag{E.11}
$$

As $b \to 0 ^ { + }$ , the factor $( a - b ) ^ { \beta }$ converges to $a ^ { \beta } > 0$ , while $\log ( a / b ) = \log a - \log b \to + \infty$ . More precisely,

$$
{ \frac { s _ { \beta } ( a , b ) } { a ^ { \beta } [ \log ( a / b ) ] ^ { 1 - \beta } } } = \biggl ( 1 - { \frac { b } { a } } \biggr ) ^ { \beta } \longrightarrow 1 , \qquad s _ { \beta } ( a , b ) \sim a ^ { \beta } [ \log ( a / b ) ] ^ { 1 - \beta } .\tag{E.12}
$$

Thus, the limiting behavior is determined by the exponent $1 - \beta$ , since the remaining factor approaches a finite, strictly positive constant.

The three regimes. (i) If $0 \leq \beta < 1$ , then $1 - \beta > 0$ SO $[ \log ( a / b ) ] ^ { 1 - \beta } \to + \infty$ and hence $s _ { \beta } ( a , b ) \to + \infty$ . (ii) If $\beta = 1$ , the logarithmic factor cancels exactly: $s _ { 1 } ( a , b ) = a - b \to a$ . (iii) If $\beta > 1$ , then

$$
s _ { \beta } ( a , b ) = \frac { ( a - b ) ^ { \beta } } { [ \log ( a / b ) ] ^ { \beta - 1 } } \longrightarrow 0 ,\tag{E.13}
$$

because the numerator converges to $a ^ { \beta }$ while the denominator diverges. Combining these cases with symmetry establishes both limits in Proposition 2. In particular, the zero limit for $\beta > 1$ concerns arbitrarily small positive $b ;$ it does not imply that the score equals zero at any finite $b \in ( 0 , a )$