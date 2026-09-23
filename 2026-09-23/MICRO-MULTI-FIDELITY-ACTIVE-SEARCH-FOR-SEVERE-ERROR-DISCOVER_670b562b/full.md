# MICRO: MULTI-FIDELITY ACTIVE SEARCH FOR SEVERE ERROR DISCOVERY

Orlando Leone<sup>1</sup>, Niclas Pokel<sup>1,2</sup>, Pehuen Moure´ <sup>1,2</sup>, Yingqiang Gao<sup>3</sup>, Roman Boehringer<sup>1</sup>

<sup>1</sup>Institute of Neuroinformatics, University of Zurich and ETH Zurich, Switzerland

<sup>2</sup>ETH AI Center, Switzerland

<sup>3</sup>Department of Computational Linguistics, University of Zurich, Switzerland {oleone,npokel,pehuen,roman}@ini.ethz.ch, yingqiang.gao@cl.uzh.ch

## ABSTRACT

Human feedback can vary in cost and informativeness. Strong feedback can reveal severe errors but is costly, so cheaper quality ratings can help decide which items to annotate. We propose MICRO (Multi-Fidelity Impact Clustered Rollout), an active search framework that allocates a shared budget to these feedback types to maximise confirmed severe error discoveries. MICRO jointly models ratings and annotation losses conditional on item features to steer acquisition. It clusters acquisitions by their predicted impact on severity probabilities to select diverse candidates, then uses rollout to estimate their discovery value. Experiments on WMT20 English–German show that ratings improve both loss reconstruction and severity prediction. MICRO achieves the highest mean discovery count across four budget and rating cost settings, with similar performance to adapted MF-ENS in one and significant gains over all six comparison policies, including two rollout controls, in the other three (p < .001).

Index Terms— Active search, multi-fidelity, error discovery, quality estimation, sequential learning

## 1. INTRODUCTION

Human feedback can vary substantially in informativeness and effort. For example, in personalised ASR for people with complex disabilities, explicitly correcting transcription errors can be very burdensome [1]. In contrast, providing a simple quality rating is considerably easier. Machine translation deployments [2] present a similar asymmetry where direct quality assessment can be substantially easier and cheaper than one-to-one correction and post-editing. In particular, Graham et al. [3] estimated that professional post-editing can cost roughly 10–20 times as much as direct assessment.

To exploit this difference in cost and informativeness, we propose an approach that combines weak and strong feedback [4] under a shared acquisition budget. Strong feedback reveals an annotation loss, while weak feedback provides a scalar quality rating, much like bandit feedback [5].

Active search, a variant of active learning [6], aims to discover target examples under a limited acquisition budget [7].

Multi-fidelity approaches extend this problem by allowing observations with different costs and levels of reliability [8]. In our setting, the relationship between quality ratings and annotation loss is uncertain and must be inferred. We use this mapping for quality estimation (QE), enabling active search to identify severe errors under a limited acquisition budget.

Quality ratings do not necessarily exactly correspond to annotation loss: for instance, perceived ASR quality can differ from measures based on word error [9]. Therefore, we model ratings as noisy observations related to annotation loss.

We propose MICRO (Multi-fidelity Impact Clustered Rollout), a framework for discovering severe errors using strong and weak feedback under a shared budget. MICRO jointly models annotation losses and quality ratings conditional on item features, then clusters acquisitions by their predicted impact on posterior severity probabilities to select diverse candidates for rollout [10].

Targeted error discovery can reveal systematic model failures more efficiently [11, 12], providing examples which can then support fine-tuning and evaluation. In this work, we evaluate feedback reconstruction and active search, demonstrating that MICRO discovers more severe errors on average than the comparison policies under the same acquisition budget.

## 2. PROBLEM FORMULATION

Item features, such as input characteristics, embeddings or model confidence scores, can provide information about the true latent (annotation) loss [13]. We therefore model this dependence, and then treat ratings as noisy observations related to annotation loss and these features.

Let U be a large pool of unlabelled items. Each segment i has known features $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ , unknown annotation loss $Y _ { i }$ and unknown ratings $R _ { i j }$ , where $j$ indexes the segment’s ratings. We define severity as $S _ { i } = \mathbf { 1 } \{ Y _ { i } > \tau \}$ for a threshold $\tau = \hat { Q } _ { 0 . 9 0 } ( \{ y _ { i } : i \in \mathcal { C } \} )$ , where C is a separate calibration set of segments and $\hat { Q } _ { 0 . 9 0 }$ is its empirical 90th percentile. Only strong feedback which reveals a loss exceeding τ confirms a severe error discovery.

An acquisition policy [7] selects an action a at step t using the set of past observations $\mathcal { D } _ { t }$ . We assign annotations a cost of 1 and ratings a cost $\rho$ with $0 < \rho < 1$ , and limit the total acquisition cost by setting a budget B shared between both feedback modes.

Our objective is thus to choose a policy π which maximises the expected number of confirmed (annotated) severe segments:

$$
\operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \left[ U _ { B } \mid { \mathcal { D } } _ { 0 } \right] , \qquad U _ { B } = \sum _ { i \in { \mathcal { A } } } \mathbf { 1 } \{ y _ { i } > \tau \} ,\tag{1}
$$

where $\mathcal { A }$ is the set of annotated segments. We thus define the expected immediate reward of an action as

$$
\begin{array} { r } { \bar { r } _ { t } ( a ) = \left\{ { P } ( Y _ { i } > \tau \mid \mathcal { D } _ { t } ) , \quad a = \mathrm { a n n o t a t i o n } ( i ) , \right. } \\ { 0 , \quad \left. a = \mathrm { r a t i n g } ( i ) . \right. } \end{array}\tag{2}
$$

Since utility depends on labels, observations can inform later acquisitions. A policy must therefore balance annotations’ immediate discoveries against the information value of both feedback types for selecting better annotations.

## 3. MICRO

## 3.1. Feedback Reconstruction

We use a joint Gaussian model [14] for feedback reconstruction and later active search. This provides the joint predictive distribution of both annotation losses and ratings, $p ( Y , R$ | x):

$$
\begin{array} { r } { \left[ Y _ { i } \right] = \left[ \pmb { x } _ { i } ^ { \top } \pmb { w } _ { Y } \right] + \pmb { s } _ { i } + \pmb { h } _ { d ( i ) } + \left[ \begin{array} { c } { 0 } \\ { e _ { i j } } \end{array} \right] , } \end{array}\tag{3}
$$

where $_ { w _ { Y } }$ and ${ \pmb w } _ { R }$ are learned weights with a Gaussian posterior, and $e _ { i j } \sim \mathcal N ( 0 , \sigma _ { R } ^ { 2 } )$ is independent rating noise. The vectors $\mathbf { \boldsymbol { s } } _ { i }$ and $h _ { d ( i ) }$ contain loss and rating offsets for segment i and its document–system group d(i):

$$
\begin{array} { r } { \pmb { s } _ { i } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { \mathrm { s e g } } ) , \qquad \pmb { h } _ { d } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { \mathrm { g r o u p } } ) . } \end{array}
$$

Correlated offsets let ratings inform losses within and across segments in the same document–system group.

## 3.2. Multi-Fidelity Active Search

Let $O _ { a }$ be the observation returned by a (which can be either an annotation or a rating), and Z the vector of latent losses. Given $\mathcal { D } _ { t }$ , these losses have mean $\pmb { \mu } _ { t }$ and covariance $\Sigma _ { t }$ . We then define the following posterior moments:

$$
\begin{array} { r l r } { m _ { a , t } = \mathbb { E } [ O _ { a } \mid \mathcal { D } _ { t } ] , } & { v _ { a , t } = \mathrm { V a r } ( O _ { a } \mid \mathcal { D } _ { t } ) , } & \\ { \mathbf { c } _ { a , t } = \mathrm { C o v } ( \mathbf { Z } , O _ { a } \mid \mathcal { D } _ { t } ) . } & { } & \end{array}
$$

Under the joint Gaussian model, with fitted variance parameters held fixed, observing $O _ { a } = o _ { a }$ gives the standard

Gaussian posterior updates [15]

$$
\begin{array} { r l } & { \pmb { \mu } _ { t + 1 } = \pmb { \mu } _ { t } + \frac { \pmb { c } _ { a , t } } { v _ { a , t } } \big ( o _ { a } - m _ { a , t } \big ) , } \\ & { \pmb { \Sigma } _ { t + 1 } = \pmb { \Sigma } _ { t } - \frac { \pmb { c } _ { a , t } \pmb { c } _ { a , t } ^ { \top } } { v _ { a , t } } . } \end{array}\tag{4}
$$

Thus for an unannotated segment i, the posterior severity probability (with standard normal CDF Φ) is

$$
p _ { t , i } = \Phi \left( \frac { \mu _ { t , i } - \tau } { \sqrt { \Sigma _ { t , i i } } } \right) .\tag{5}
$$

Holding $\Sigma _ { t , i i }$ fixed, using a first-order Taylor approximation with respect to $\mu _ { t , i }$ and then substituting from (4):

$$
\begin{array} { r l r } {  { \Delta p _ { t , i } \approx \frac { \Phi ^ { \prime } ( ( \mu _ { t , i } - \tau ) / \sqrt { \Sigma _ { t , i i } } ) } { \sqrt { \Sigma _ { t , i i } } } \Delta \mu _ { t , i } } } \\ & { } & { \quad = \frac { \Phi ^ { \prime } ( ( \mu _ { t , i } - \tau ) / \sqrt { \Sigma _ { t , i i } } ) } { \sqrt { \Sigma _ { t , i i } } } \frac { c _ { a , t , i } } { v _ { a , t } } ( O _ { a } - m _ { a , t } ) . } \end{array}
$$

Following the expected model output change principle [16], we define an impact vector that approximates how an observation influences each segment’s severity probability:

$$
\psi _ { a , t , i } = \frac { \Phi ^ { \prime } ( ( \mu _ { t , i } - \tau ) / \sqrt { \Sigma _ { t , i i } } ) } { \sqrt { \Sigma _ { t , i i } } } \frac { c _ { a , t , i } } { \sqrt { v _ { a , t } } } .\tag{6}
$$

Under this approximation, we obtain the expected impact

$$
\begin{array} { r } { \mathbb { E } \left[ \| \Delta \pmb { p } _ { t } \| _ { 1 } \middle | \mathcal { D } _ { t } \right] \approx \mathbb { E } \left[ \| \psi _ { a , t } \| _ { 1 } \left| \frac { O _ { a } - m _ { a , t } } { \sqrt { v _ { a , t } } } \right| \middle | \mathcal { D } _ { t } \right] } \\ { = \| \psi _ { a , t } \| _ { 1 } \sqrt { \frac { 2 } { \pi } } , } \end{array}\tag{7}
$$

where $\sqrt { 2 / \pi }$ is the mean absolute value of a standard normal variable. For an annotation of i, we set $\psi _ { a , t , i } = 0$ (selfimpact coordinate), since we score immediate discovery by $\bar { r } _ { t } ( a )$ . Hence, MICRO works as follows:

1. We cluster possible annotations and ratings separately into $K / 2$ clusters each, running k-means [17] on their impact vectors $\psi _ { a , t } .$ We then score annotations by $\bar { r } _ { t } ( a )$ and ratings by $\| \psi _ { a , t } \| _ { 1 } ~ ( \sqrt { 2 / \pi }$ does not affect rankings). To reduce redundancy among rollout candidates, we construct a root pool of size K by selecting the highest scoring candidate per cluster: diversitybased selection also appears in active learning [18].

2. We evaluate each root over M simulated trajectories. For each trajectory, we sample the root’s observation to update the posterior, then simulate greedy acquisitions based on $\bar { r } _ { t } ( a )$ from the full annotation pool, conditioning on every acquisition before choosing the next. To score roots at residual budget $\delta \ < \ 1$ , we assign fractional reward $\delta p$ to the next greedy annotation with severity probability $p .$ . Greedy trajectories only use annotations because ratings have $\bar { r } _ { t } ( a ) = 0 .$ While full look-ahead is computationally intractable, our rollout evaluates K roots using M simulated trajectories of at most H sequential steps, thus requiring only $O ( K M H )$ simulated steps.

3. We acquire the root with the greatest estimated reward averaged over all the simulated trajectories. We then observe its feedback, update the posterior and then subtract its cost from B.

4. We repeat steps 1 through 3 until less than one annotation’s cost is left in the budget. A rating can be acquired only if it leaves budget for an annotation to follow.

## 4. EXPERIMENTAL SETUP

## 4.1. Feedback Reconstruction

We benchmark the model on English–German from the 2020 Workshop on Machine Translation (WMT20) dataset [19]. Translations are scored with strong Multidimensional Quality Metrics (MQM) and a weak 0–6 professional Scalar Quality Metric (pSQM): there are three pSQM ratings per segment.

We choose general purpose text features, concatenating frozen LaBSE embeddings [20] of source and translation texts, then applying PCA to 32 dimensions and standardising each component. We use document-disjoint calibration, development and test splits. PCA, model fitting and standardisation use calibration data only.

For reconstruction, we also optionally calibrate the posterior mean using two monotone functions:

$$
\widetilde { y } _ { i } = c _ { Y } ( \widehat { y } _ { i } ) , \qquad \widetilde { p } _ { i } = c _ { S } ( \widehat { y } _ { i } ) ,
$$

where $c _ { Y }$ and $c _ { S }$ map the posterior mean to MQM loss and $P ( Y _ { i } > \tau )$ , respectively. Both functions are fitted with isotonic regression on calibration data. We avoid selection bias from the acquisition policy by observing feedback at random.

## 4.2. Multi-Fidelity Active Search

We evaluate active search by confirmed severe errors discovered, over 1024 paired held-out English–German pools per setting at rating costs $\rho ~ \in ~ \{ 1 / 1 6 , 1 / 8 \}$ and budgets $B \in$ {16, 32}. These costs represent hypothetical scenarios, not measured MQM/pSQM costs. Each pool contains 128 segments from the test set, which are then shared across policies.

Besides the standard greedy baseline, we also benchmark against annotation-only ENS [21], which evaluates every annotation by its immediate reward and the expected topprobability batch value for the remaining budget. We adapt ENS to numerical annotation losses and estimate its expected utility by using 128 samples from the Gaussian predictive distribution. We adapt MF-ENS [8] from parallel queries to feedback under a shared budget (and numerical annotation losses). It scores each candidate by simulating its observation, followed by a batch of ratings and then a greedy batch of annotations. For each candidate, we simulate 16 possible outcomes and use 32 simulations each for batch size selection and independent evaluation. In line with MF-ENS’s exploration window of one strong-query duration, we limit each simulated rating batch to one annotation’s cost. Simulated batch ratings update only their own segment, while actual observations update the full posterior. “Screen-thenannotate” acquires only the highest scoring ratings, until it reaches a screening quota of $B / ( 4 \rho )$ and then annotates greedily. We then include two rollout controls: “undiversified rollout” selects candidates by highest reward/impact without clustering’s explicit diversity objective, whereas “blockbased rollout” enforces diversity by selecting a candidate per document–system block (instead of per cluster). All policies share the fitted feedback model.

While adapted MF-ENS evaluates all legal acquisitions, MICRO only evaluates K = 12 candidates with $M = 1 2 8$ trajectories, chosen for computational practicality (the calibration mentioned in Section 4.1 is not applied in active search). We report mean discoveries and paired differences with 95% confidence intervals and unadjusted two-sided paired t-test p-values.

## 5. RESULTS AND DISCUSSION

## 5.1. Feedback Reconstruction

Table 1. Held-out reconstruction of MQM loss from translation features and pSQM ratings on English–German. Metrics are averaged within source documents and across documents. Rating-only baselines use calibration-fitted isotonic regression.

<table><tr><td rowspan="8">Method</td><td>Ratings MAE↓</td><td>Brier ↓</td></tr><tr><td>Pre-rating joint model 0/3</td><td>1.762 .1248</td></tr><tr><td>Rating-only model 1/3</td><td>1.481 .1029</td></tr><tr><td>1/3</td><td>1.461 .0994</td></tr><tr><td>Joint model Calibrated joint model 1/3</td><td>1.430 .0994</td></tr><tr><td>Rating-only model 2/3 Joint model</td><td>1.360 .0929</td></tr><tr><td>2/3</td><td>1.366 .0897</td></tr><tr><td>Calibrated joint model 2/3</td><td>1.316 .0892 .0876</td></tr><tr><td>Rating-only model</td><td>3/3 1.302</td></tr><tr><td>Joint model 3/3</td><td>1.316 .0847</td></tr><tr><td>Calibrated joint model 3/3</td><td>1.259 .0844</td></tr></table>

Table 1 shows more ratings improve loss reconstruction (and in turn severity prediction). Comparing the calibrated three rating model to pre-rating predictions, MAE is reduced from 1.762 to 1.259 and Brier score from .1248 to .0844. Although the joint model’s MAE is slightly higher than the rating-only baseline with two or three ratings, it consistently improves Brier score. The calibrated joint model reaches the lowest MAE and Brier scores at every rating count.

## 5.2. Multi-Fidelity Active Search

Table 2. Severe errors discovered over 1024 paired pools per setting. Differences, confidence intervals and p-values compare each method with MICRO (method minus MICRO).
<table><tr><td>Method</td><td>Mean</td><td>Diff.</td><td>95% CI</td><td>p</td></tr><tr><td colspan="5">ρ = 1/16, B = 16</td></tr><tr><td>Greedy</td><td>3.870</td><td>-1.338</td><td>[−1.435, -1.241]</td><td>&lt; .001</td></tr><tr><td>ENS</td><td>3.871</td><td>-1.337</td><td>[-1.433, -1.241]</td><td>&lt; .001</td></tr><tr><td>Adapted MF-ENS</td><td>5.032</td><td>-0.176</td><td>[-0.237, -0.114]</td><td>&lt; .001</td></tr><tr><td>Screen-then-annotate</td><td>4.731</td><td>-0.477</td><td>[−0.550, -0.403]</td><td>&lt; .001</td></tr><tr><td>Undiversified rollout</td><td>5.078</td><td>-0.130</td><td>[-0.184, -0.076]</td><td>&lt; .001</td></tr><tr><td>Block-based rollout</td><td>5.084</td><td>-0.124</td><td>[-0.177, -0.072]</td><td>&lt; .001</td></tr><tr><td>MICRO</td><td>5.208</td><td></td><td></td><td></td></tr><tr><td colspan="5">ρ = 1/16, B = 32</td></tr><tr><td>Greedy</td><td>6.502</td><td></td><td>-3.471 [-3.597, -3.345]</td><td>&lt; .001</td></tr><tr><td>ENS</td><td>6.512</td><td>-3.461</td><td>[-3.586, -3.336]</td><td>&lt; .001</td></tr><tr><td>Adapted MF-ENS</td><td>9.500</td><td>-0.473</td><td>[−0.541, -0.405]</td><td>&lt; .001</td></tr><tr><td>Screen-then-annotate</td><td>8.874</td><td>-1.099</td><td>[-1.195, -1.003]</td><td>&lt; .001</td></tr><tr><td>Undiversified rollout</td><td>9.727</td><td>-0.246</td><td>[-0.303, -0.189]</td><td>&lt; .001</td></tr><tr><td>Block-based rollout</td><td>9.761</td><td>-0.212</td><td>[-0.267, -0.156]</td><td>&lt; .001</td></tr><tr><td>MICRO</td><td>9.973</td><td></td><td></td><td></td></tr><tr><td colspan="5"></td></tr><tr><td>Greedy</td><td>ρ = 1/8, 3.828</td><td>B = 16 -0.437</td><td>[-0.523,-0.350]</td><td>&lt; .001</td></tr><tr><td>ENS</td><td>3.830</td><td>-0.435</td><td>[-0.522, -0.347]</td><td>&lt; .001</td></tr><tr><td>Adapted MF-ENS</td><td>4.248</td><td>-0.017</td><td>[-0.070, 0.037]</td><td>.543</td></tr><tr><td>Screen-then-annotate</td><td>4.103</td><td>-0.162</td><td>[−0.212, −0.112]</td><td>&lt; .001</td></tr><tr><td>Undiversified rollout</td><td>4.220</td><td>-0.045</td><td>[-0.093, 0.003]</td><td>.069</td></tr><tr><td>Block-based rollout</td><td>4.236</td><td>-0.028</td><td>[-0.075,0.018]</td><td>.232</td></tr><tr><td>MICRO</td><td>4.265</td><td></td><td></td><td></td></tr><tr><td colspan="5"> $\rho = 1 / 8 , \quad B = 3 2$ </td></tr><tr><td>Greedy</td><td>6.695</td><td>-1.522</td><td>[-1.641,-1.404]</td><td>&lt; .001</td></tr><tr><td>ENS</td><td>6.695</td><td>-1.522</td><td>[−1.643, -1.402]</td><td>&lt; .001</td></tr><tr><td>Adapted MF-ENS</td><td>7.962</td><td>-0.256</td><td>[-0.329, -0.182]</td><td>&lt; .001</td></tr><tr><td>Screen-then-annotate</td><td>7.876</td><td>-0.342</td><td>[−0.411, -0.273]</td><td>&lt; .001</td></tr><tr><td>Undiversified rollout</td><td>8.032</td><td>-0.186</td><td>[−0.242, -0.129]</td><td>&lt; .001</td></tr><tr><td>Block-based rollout</td><td>8.029</td><td>-0.188</td><td>[−0.245, -0.132]</td><td>&lt; .001</td></tr><tr><td>MICRO</td><td>8.218</td><td></td><td></td><td></td></tr></table>

MICRO discovers the largest mean number of severe errors in all four settings (Table 2), consistently outperforming greedy, ENS and screen-then-annotate $( p < . 0 0 1 )$ . The two annotation-only policies, greedy and ENS, achieve similarly low mean yields.

Its relative advantage over these baselines increases with larger budgets. At $\rho = 1 / 1 6 .$ , the improvements rise from 34.6% to 53.4% over greedy, from 34.5% to 53.1% over ENS, and from 10.1% to 12.4% over screen-then-annotate, as B increases from 16 to 32. The same holds at $\rho = 1 / 8 ,$ , though

the gains are smaller.

Adapted MF-ENS and both rollout controls achieve higher mean yields than greedy, ENS and screen-thenannotate in all settings. ${ \mathrm { A t ~ } } \rho { \mathrm { ~ = ~ } } 1 / 8 , B { \mathrm { ~ = ~ } } 1 6 ,$ MICRO has a similar mean yield $( p = . 5 4 3 , p = . 0 6 9 , p = . 2 3 2 ) $ and significantly outperforms them at the other three settings $( p < . 0 0 1 )$ . These comparisons support impact-based clustering, and a plausible explanation for the advantage is that it can reduce overlap and redundancy among candidate roots, allowing rollout to consider a broader range of acquisitions.

## 5.3. Limitations

Although personalised ASR motivates MICRO, we evaluate only on WMT20 English–German translations. Feedback costs are also assumed and standardised: in practice, annotation and rating costs may vary due to factors like item length, error severity or user fatigue.

In additional experiments at $\rho \in \{ 1 / 4 , 1 / 2 \}$ , we did not observe any significant improvements by MICRO and the two rollout controls over the baseline acquisition policies. These results suggest that ratings’ informativeness may not justify their cost in those settings, as their value is dataset-dependent.

Since rollout trajectories only include annotations, MI-CRO is limited in evaluating the benefit of acquiring multiple ratings jointly. Our MF-ENS comparison adapts its parallel setting and binary labels to continuous feedback and a shared budget, so other adaptations could yield different results.

MICRO’s k-means step adds additional computational overhead, and we did not compare different methods under equal wall-clock time. However, in our runtime checks, clustering always took over an order of magnitude less time than the rollout phase.

## 6. CONCLUSION

Although weak feedback cannot directly confirm a discovery, it can improve quality estimation and guide the acquisition of strong feedback. MICRO jointly models annotation losses and quality ratings, performing active search by combining impact-based clustering with rollout to select acquisitions under a shared acquisition budget.

Our experiments on WMT20 English–German show improved loss reconstruction and more severe error discoveries when ratings are sufficiently cheap. MICRO achieves the highest mean discovery count across all settings, and significantly outperforms all comparison policies in three settings, including both large budget settings. Comparisons with rollout controls support the benefit of impact-based clustering.

These findings motivate further study on multi-fidelity active search and its application specifically to personalised ASR. Future work should evaluate policies under equal computational budgets and realistic user effort, as well as account for sequences of weak feedback acquisition.

## 7. REFERENCES

[1] L. P. Guldimann, “Speech recognition for Germanspeaking children with congenital disorders: Current limitations and dataset challenges,” M.S. thesis, ETH Zurich, 2024.

[2] J. De Wilde, A. Wouters, A. Tezcan, S. Van den Meersschaut, K. Maryns, and L. Macken, “MaTIAS – machine translation to inform asylum seekers: final results,” in Proc. Annu. Conf. Eur. Assoc. Mach. Transl. (EAMT). 2026, vol. 2, pp. 4–5, European Association for Machine Translation.

[3] Y. Graham, Q. Ma, T. Baldwin, Q. Liu, C. Parra, and C. Scarton, “Improving evaluation of document-level machine translation quality estimation,” in Proc. Conf. Eur. Chapter Assoc. Comput. Linguist. (EACL), Short Papers. 2017, vol. 2, pp. 356–361, Association for Computational Linguistics.

[4] C. Zhang and K. Chaudhuri, “Active learning from weak and strong labelers,” in Advances in Neural Information Processing Systems, 2015, vol. 28, pp. 703–711.

[5] J. Kreutzer, A. Sokolov, and S. Riezler, “Bandit structured prediction for neural sequence-to-sequence learning,” in Proc. Annu. Meeting Assoc. Comput. Linguist. (ACL), 2017, vol. 1, pp. 1503–1513.

[6] B. Settles, “Active learning literature survey,” Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.

[7] R. Garnett, Y. Krishnamurthy, X. Xiong, J. Schneider, and R. Mann, “Bayesian optimal active search and surveying,” in Proc. Int. Conf. Mach. Learn. (ICML), 2012, pp. 843–850.

[8] Q. Nguyen, A. Modiri, and R. Garnett, “Nonmyopic multifidelity active search,” in Proc. Int. Conf. Mach. Learn. (ICML). 2021, vol. 139 of Proceedings of Machine Learning Research, pp. 8109–8118, PMLR.

[9] S. Kim, D. Le, W. Zheng, T. Singh, A. Arora, X. Zhai, C. Fuegen, O. Kalinli, and M. L. Seltzer, “Evaluating user perception of speech recognition system quality with semantic distance metric,” in Proc. Interspeech, 2022, pp. 3978–3982.

[10] D. P. Bertsekas, J. N. Tsitsiklis, and C. Wu, “Rollout algorithms for combinatorial optimization,” J. Heuristics, vol. 3, pp. 245–262, 1997.

[11] H. Lakkaraju, E. Kamar, R. Caruana, and E. Horvitz, “Identifying unknown unknowns in the open world: Representations and policies for guided exploration,” in Proc. AAAI Conf. Artif. Intell., 2017, vol. 31, pp. 2124– 2132.

[12] M. T. Ribeiro, T. Wu, C. Guestrin, and S. Singh, “Beyond accuracy: Behavioral testing of NLP models with CheckList,” in Proc. Annu. Meeting Assoc. Comput. Linguist. (ACL), 2020, pp. 4902–4912.

[13] L. Specia and A. Farzindar, “Estimating machine translation post-editing effort with HTER,” in Proc. Second Joint EM+/CNGL Workshop: Bringing MT to the User: Research on Integrating MT in the Translation Industry, 2010, pp. 33–41.

[14] C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning, MIT Press, Cambridge, MA, 2006.

[15] P. Frazier, W. Powell, and S. Dayanik, “The knowledgegradient policy for correlated normal beliefs,” IN-FORMS J. Comput., vol. 21, no. 4, pp. 599–613, 2009.

[16] A. Freytag, E. Rodner, and J. Denzler, “Selecting influential examples: Active learning with expected model output changes,” in Computer Vision – ECCV 2014. 2014, vol. 8692 of Lecture Notes in Computer Science, pp. 562–577, Springer.

[17] S. P. Lloyd, “Least squares quantization in PCM,” IEEE Trans. Inf. Theory, vol. 28, no. 2, pp. 129–137, 1982.

[18] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal, “Deep batch active learning by diverse, uncertain gradient lower bounds,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[19] M. Freitag, G. Foster, D. Grangier, V. Ratnakar, Q. Tan, and W. Macherey, “Experts, errors, and context: A large-scale study of human evaluation for machine translation,” Trans. Assoc. Comput. Linguist., vol. 9, pp. 1460–1474, 2021.

[20] F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, “Language-agnostic BERT sentence embedding,” in Proc. Annu. Meeting Assoc. Comput. Linguist. (ACL), 2022, vol. 1, pp. 878–891.

[21] S. Jiang, G. Malkomes, G. Converse, A. Shofner, B. Moseley, and R. Garnett, “Efficient nonmyopic active search,” in Proc. Int. Conf. Mach. Learn. (ICML). 2017, vol. 70 of Proceedings of Machine Learning Research, pp. 1714–1723, PMLR.