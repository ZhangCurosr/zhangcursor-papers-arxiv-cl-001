# From Self-Confidence to Statistical Accuracy: Scaling Human-in-the-Loop Evaluation with Direct and LLM-Assisted Auditing

Harine Choi<sup>1</sup>, Chenchen Kuai<sup>2</sup>, Zihao Li<sup>3</sup>, Jiwan Jiang<sup>2</sup> Yunlong Zhang<sup>2</sup>, Zhiwen Fan<sup>4</sup>, Zhengzhong Tu<sup>5</sup>, Yang Zhou<sup>2</sup>

<sup>1</sup>Texas A&M Transportation Institute, Texas A&M University System <sup>2</sup>Zachry Department of Civil and Environmental Engineering, Texas A&M University <sup>3</sup>Department of Civil, Construction and Environmental Engineering, Marquette University <sup>4</sup>Department of Electrical and Computer Engineering, Texas A&M University <sup>5</sup>Department of Computer Science and Engineering, Texas A&M University

## Abstract

For large language models (LLMs) and multimodal large language models (MLLMs), model-reported self-confidence is often miscalibrated, yet it may still be useful for efficient evaluation. At the same time, LLM-as-a-judge offers a scalable alternative to direct human auditing, but its practical benefit depends on calibration quality and inference cost.

This paper develops a unified statistical framework for both evaluation routes under alternative sampling strategies. For direct auditing, confidence-stratified sampling is used to recover population accuracy from partially verified outputs without requiring calibrated confidence. Finite-sample uncertainty is characterized, and the analysis shows how highly uneven confidence strata affect audit efficiency and label allocation. For judge-assisted evaluation, the judge is modeled through sensitivity and false-positive rate, yielding an error-corrected estimator with calibration uncertainty. The two routes are then compared through human-label and total-cost efficiency boundaries.

Experiments on MMLU-Pro and MMMU-Pro show that miscalibrated self-confidence can still support statistically valid stratified auditing. Under highly imbalanced confidence distributions, proportional allocation substantially reduces estimation error relative to equal-perstratum allocation. For judge-assisted evaluation, only a small subset of target–verifier pairs satisfies the idealized human-label efficiency condition, and automated inference costs can further limit potential savings. These findings clarify the statistical and economic conditions under which confidence-guided auditing and automated verification are useful.

## 1 Introduction

Large language models (LLMs) and multimodal large language models (MLLMs) are increasingly evaluated at scales that make exhaustive human verification impractical (Liang et al., 2022; Lee et al.,

2024; Chiang et al., 2024; Chen et al., 2024). Since obtaining reliable reference labels can constitute a substantial fraction of evaluation cost (Kossen et al., 2021; Boyeau et al., 2025), only a subset of model outputs may be inspected in practice. Evaluation then becomes a statistical inference problem: population-level accuracy must be recovered from partial verification, together with uncertainty that reflects the limited labeling budget.

Self-confidence can guide which outputs to audit, but its relationship with correctness varies across models and tasks (Tian et al., 2023; Xiong et al., 2024). Confidence is often miscalibrated, and disproportionately auditing low-confidence outputs can underrepresent errors among confident predictions. Moreover, confidence strata can differ substantially in size, making audit allocation important for estimation precision. Thus, confidence can guide sampling, but population accuracy must be estimated using the sampling design and stratum sizes.

LLM-as-a-judge provides an alternative evaluation route in which a second LLM evaluates outputs at scale (Zheng et al., 2023). Because the judge is imperfect, human labels are still needed to estimate and correct its error rates, and calibration uncertainty propagates into the final accuracy estimate (Lee et al., 2025; Chen et al., 2026). The evaluation budget can therefore support either direct auditing or judge calibration. Their relative efficiency depends on judge discrimination, calibration requirements, and the costs of automated inference and human review.

In this work, confidence-guided auditing and judge-assisted evaluation are unified within a common statistical framework. For direct auditing, model confidence is used as a stratification variable rather than a trusted correctness probability, enabling population accuracy and finite-sample uncertainty to be estimated from partial verification. For judge-assisted evaluation, the judge is characterized by sensitivity and false-positive rate, yielding an error-corrected estimator with finite calibration uncertainty. The two routes are compared through human-label and cost-efficiency boundaries that identify when automated judging is preferable at the same target precision. Figure 1 summarizes the overall framework. We evaluate the approach on MMLU-Pro and MMMU-Pro, covering text and multimodal reasoning.

## 2 Related Work

Confidence, calibration, and selective evaluation. Model confidence has long been studied as a signal of predictive uncertainty. Calibration measures whether reported probabilities agree with empirical correctness, with methods such as temperature scaling used to correct systematic errors on labeled validation data (Guo et al., 2017). For language models, uncertainty can be elicited directly through verbalized confidence or inferred from model outputs and response consistency (Lin et al., 2022; Kadavath et al., 2022; Tian et al., 2023; Kuhn et al., 2023). Empirical studies nevertheless show substantial miscalibration across models, tasks, and modalities (Xiong et al., 2024; Jang et al., 2026; Xuan et al., 2025). Selective prediction uses uncertainty to trade coverage for error control (Geifman and El-Yaniv, 2017). Population evaluation differs because confidence-guided review determines which correctness labels become observable. Estimating overall accuracy from this selected subset therefore raises a verification-bias problem (Begg and Greenes, 1983), requiring the sampling mechanism to be accounted for explicitly.

Label-efficient statistical evaluation. Auxiliary model information can improve evaluation through both label acquisition and estimation. Active testing and Active Surrogate Estimators use adaptive sampling and learned surrogates to reduce uncertainty in estimated model performance (Kossen et al., 2021, 2022). Prediction-powered inference (PPI) uses a small labeled sample to correct estimates based on abundant model predictions (Angelopoulos et al., 2023a), a principle applied to model evaluation by AutoEval (Boyeau et al., 2025). StratPPI combines this correction with stratified sampling and label allocation to exploit differences in automatic-label quality across subpopulations (Fisch et al., 2024). These approaches make both the use of auxiliary predictions and the allocation of labels part of the evaluation design. PPI++ and Adaptive AutoEval adjust reliance on predictions according to their usefulness (Angelopoulos et al., 2023b; Park et al., 2025). With few labels, estimating prediction weights can itself introduce substantial uncertainty, as shown by studies of post-hoc regression and finite-sample PPI++ behavior (Eyre and Madras, 2025; Mani et al., 2025). Label savings therefore depend on the available budget as well as the auxiliary signal. We examine this design problem for confidencestratified direct auditing, characterizing estimation precision under fixed allocations and allocations learned from pilot audits.

LLM-as-a-judge and noisy verification. LLMbased evaluators have been studied extensively as scalable substitutes for human assessment, including G-Eval, MT-Bench, and learned judge models (Liu et al., 2023; Zheng et al., 2023; Zhu et al., 2025). However, systematic effects such as position, self-preference, knowledge, and format bias can substantially influence their decisions (Panickssery et al., 2024). For binary correctness, judge-assisted evaluation can instead be viewed as prevalence estimation under an imperfect screening test (Rogan and Gladen, 1978). A judge’s sensitivity and false-positive rate determine how its acceptance rate maps to target-model accuracy, while finite calibration data introduce additional uncertainty. Recent work has incorporated such calibration into statistical estimation and hypothesis testing (Lee et al., 2025; Feng et al., 2026; Chen et al., 2026). Building on this literature, we compare direct auditing and judge-assisted estimation at a common target precision and derive both human-label and total-cost efficiency boundaries.

## 3 Statistical Framework and Derivations

We consider two uses of limited human labels: directly auditing target outputs and calibrating an automated judge. We first establish accuracy estimation and uncertainty for each route, then compare the judge’s label and cost requirements against direct random auditing.

## 3.1 Population Accuracy and Identification

Consider a fixed population of N model outputs with reported confidence $C _ { i }$ and binary correctness $Z _ { i } \in \{ 0 , 1 \}$ . Only a subset of the correctness labels is revealed through auditing. Confidence is partitioned into K nonempty strata with sizes $N _ { j }$ population weights $\pi _ { j } = N _ { j } / N$ , and stratum accuracies $\theta _ { j }$ . The population quantity of interest is

![](images/2cc1f76ffaff406238bfcc233d3a1e09e654540f0de5e87429fedd7313094a61.jpg)  
Figure 1: Overview of the proposed evaluation framework

$$
\theta _ { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } Z _ { i } = \sum _ { j = 1 } ^ { K } \pi _ { j } \theta _ { j } .
$$

Suppose that $n _ { j } \geq 1$ outputs are sampled uniformly without replacement from stratum $j ,$ , with audited set $A _ { j }$ . The stratified estimator is

$$
\widehat { \theta } _ { \mathrm { s t r } } = \sum _ { j = 1 } ^ { K } \pi _ { j } \widehat { \theta } _ { j } , \quad \widehat { \theta } _ { j } = \frac { 1 } { n _ { j } } \sum _ { i \in \mathcal { A } _ { j } } Z _ { i } .\tag{1}
$$

Equation (1) does not require confidence to be calibrated. Confidence is used only to partition the population; statistical validity follows from random sampling within each stratum and population weighting. Miscalibrated confidence can therefore still be useful if it separates outputs with different correctness rates.

Random sampling supports statistical estimation, but the revealed labels alone do not determine every unaudited outcome. To distinguish these two points, let $p _ { j }$ denote the audited fraction of stratum j, $\mu _ { j }$ the observed audited accuracy, and $\nu _ { j }$ the unknown accuracy among unaudited outputs. Then

$$
\theta _ { j } = p _ { j } \mu _ { j } + ( 1 - p _ { j } ) \nu _ { j } .
$$

Without restrictions beyond $0 \leq \nu _ { j } \leq 1$ , the population accuracies compatible with the revealed labels satisfy

$$
\theta _ { N } \in \Big [ \sum _ { j } \pi _ { j } p _ { j } \mu _ { j } , \sum _ { j } \pi _ { j } \{ p _ { j } \mu _ { j } + 1 - p _ { j } \} \Big ] .
$$

These are identification bounds, not sampling confidence intervals. They describe what the observed labels alone establish; the sampling design provides probabilistic guarantees on estimation error, as considered next.

## 3.2 Finite-Sample Accuracy and Audit Precision

For fixed audit allocations and simple random sampling within each stratum,

$$
\mathbb { E } ( \widehat { \theta } _ { j } ) = \theta _ { j } , \quad \mathbb { E } ( \widehat { \theta } _ { \mathrm { s t r } } ) = \theta _ { N } .
$$

For independent stratum samples,

$$
\mathrm { V a r } ( \widehat { \theta } _ { \mathrm { s t r } } ) = \sum _ { j } \pi _ { j } ^ { 2 } \left( 1 - \frac { n _ { j } } { N _ { j } } \right) \frac { S _ { j } ^ { 2 } } { n _ { j } } .\tag{2}
$$

where $S _ { j } ^ { 2 }$ denotes the finite-population variance of correctness within stratum $j .$ . Thus, estimation precision depends on stratum weights, within-stratum variability, and audit allocation.

A distribution-free guarantee can also be obtained. Each audited binary observation in stratum $j$ contributes weight $\pi _ { j } / n _ { j }$ to the estimator. Hoeffding’s inequality, which also applies to sampling without replacement, gives

$$
\mathrm { P r } \Big ( | \widehat { \theta } _ { \mathrm { s t r } } - \theta _ { N } | \geq \epsilon \Big ) \leq 2 \exp \left( - \frac { 2 \epsilon ^ { 2 } } { \sum _ { j } \pi _ { j } ^ { 2 } / n _ { j } } \right)\tag{3}
$$

This bound is conservative because it does not exploit finite-population corrections.

For error tolerance ϵ and failure probability δ, it suffices that

$$
\sum _ { j } \frac { \pi _ { j } ^ { 2 } } { n _ { j } } \leq \frac { 2 \epsilon ^ { 2 } } { \log ( 2 / \delta ) } .
$$

This condition translates a precision target into an audit-budget requirement. Ignoring integer and capacity constraints, proportional allocation minimizes the Hoeffding denominator, whereas Neyman allocation minimizes estimator variance. The two allocations therefore optimize different precision criteria. Appendix C provides the allocation rules and the corresponding estimator for two-stage adaptive auditing.

## 3.3 Accuracy Recovery with an LLM Judge

Direct auditing obtains human labels for target outputs. An alternative uses human labels to calibrate a second model, which then judges a larger set of outputs. Its decisions must be treated as noisy measurements of correctness.

For a randomly drawn target output, write $\theta =$ $\theta _ { N }$ and let $W \in \{ 0 , 1 \}$ indicate whether the judge accepts its answer as correct. Define

$$
\begin{array} { r } { s = \mathrm { P r } ( W = 1 \mid Z = 1 ) , } \\ { \alpha _ { v } = \mathrm { P r } ( W = 1 \mid Z = 0 ) , } \end{array}
$$

where s is judge sensitivity and $\alpha _ { v }$ is its falsepositive rate. The acceptance probability $q \ =$ $\operatorname* { P r } ( W = 1 )$ satisfies

$$
q = s \theta + \alpha _ { v } ( 1 - \theta ) , \quad \theta = \frac { q - \alpha _ { v } } { s - \alpha _ { v } } .\tag{4}
$$

The discrimination gap is $\Delta _ { v } = s - \alpha _ { v }$ . Recovery requires $\Delta _ { v } \neq 0 ,$ , with $\Delta _ { v } > 0$ the intended operating regime. High overall judgment accuracy does not ensure a large gap: a judge that accepts most answers may appear accurate when the target is usually correct, while poorly distinguishing correct from incorrect outputs.

Suppose s is estimated from $n _ { 1 }$ human-verified correct outputs, $\alpha _ { v }$ from $n _ { 0 }$ human-verified incorrect outputs, and $q$ from M automated judgments. Assume independent samples, with calibration examples representative of the evaluation population within each correctness class. The corrected estimator is

$$
\widehat { \theta } _ { v } = \frac { \widehat { q } - \widehat { \alpha } _ { v } } { \widehat { s } - \widehat { \alpha } _ { v } } .
$$

Ignoring finite-population corrections, a first-order delta-method approximation gives

$$
\begin{array} { c } { \displaystyle \mathrm { V a r } ( \widehat \theta _ { v } ) \approx \frac 1 { \Delta _ { v } ^ { 2 } } \Bigg [ \frac { q ( 1 - q ) } M + \frac { \theta ^ { 2 } s ( 1 - s ) } { n _ { 1 } } } \\ { + \frac { ( 1 - \theta ) ^ { 2 } \alpha _ { v } ( 1 - \alpha _ { v } ) } { n _ { 0 } } \Bigg ] . } \end{array}\tag{5}
$$

The first term reflects uncertainty in the automated acceptance rate; the other two reflect uncertainty in human calibration. Increasing M reduces the first term but cannot remove calibration uncertainty. A small $\Delta _ { v }$ amplifies all three contributions.

## 3.4 Human-Label Budget and Label Efficiency

We now compare the human labels required to achieve the same target variance $\tau ^ { 2 }$

For direct random auditing, ignoring finitepopulation corrections,

$$
B _ { \mathrm { d i r e c t } } = \frac { \theta ( 1 - \theta ) } { \tau ^ { 2 } } .
$$

Confidence-stratified auditing can improve on this baseline when confidence separates groups with different accuracies. The comparison below nevertheless uses direct random auditing as its reference in order to isolate the contribution of verifier calibration relative to the standard unstratified design.

For judge-assisted evaluation, let the calibration budget be

$$
B _ { v } = n _ { 1 } + n _ { 0 } .
$$

By Eq. (5), its calibration variance is minimized when

$$
n _ { 1 } : n _ { 0 } = \theta \sqrt { s ( 1 - s ) } : ( 1 - \theta ) \sqrt { \alpha _ { v } ( 1 - \alpha _ { v } ) } .
$$

This is an oracle allocation because it depends on unknown operating rates and requires classconditional calibration samples. Since correctness is unknown before auditing, obtaining the required correct and incorrect examples may require additional human screening. These screening costs are excluded from the idealized efficiency comparison.

Under this allocation, define

$$
\begin{array} { l } { A _ { v } = \frac { \left[ \theta \sqrt { s ( 1 - s ) } + ( 1 - \theta ) \sqrt { \alpha _ { v } ( 1 - \alpha _ { v } ) } \right] ^ { 2 } } { \Delta _ { v } ^ { 2 } } , } \\ { D _ { v } = \frac { q ( 1 - q ) } { \Delta _ { v } ^ { 2 } } . } \end{array}
$$

Equation (5) then becomes

$$
\mathrm { V a r } ( \widehat { \theta } _ { v } ) \approx \frac { A _ { v } } { B _ { v } } + \frac { D _ { v } } { M } .\tag{6}
$$

Here, $A _ { v }$ determines the calibration contribution and $D _ { v }$ the automated-sampling contribution.

The label-efficiency ratio is therefore

$$
\begin{array} { l } { \displaystyle \eta = \frac { B _ { \mathrm { d i r e c t } } } { B _ { v } } = \frac { \theta ( 1 - \theta ) } { A _ { v } } } \\ { \displaystyle = \frac { \theta ( 1 - \theta ) \Delta _ { v } ^ { 2 } } { \left[ \begin{array} { c } { \theta \sqrt { s ( 1 - s ) } } \\ { + ( 1 - \theta ) \sqrt { \alpha _ { v } ( 1 - \alpha _ { v } ) } } \end{array} \right] ^ { 2 } } . } \end{array}\tag{7}
$$

The ratio η characterizes oracle human-label efficiency when automated-sampling uncertainty is negligible $( M \to \infty )$ . When $\eta > 1$ , verifier calibration requires fewer human labels than direct random auditing at equal asymptotic precision under the assumed class-conditional allocation. For finite $M ,$ , the additional variance contribution $D _ { v } / M$ reduces this advantage. Thus, η represents an idealized efficiency criterion rather than realized label savings. An extension to imperfect human verification is presented in Appendix D.

## 3.5 Cost-Efficiency Boundary

The preceding comparison neglects automated inference cost. Let $c _ { H }$ denote the cost of one human reference label and $c _ { L }$ the cost of one automated judge decision. Consider the idealized cost

$$
C _ { \mathrm { j u d g e } } = c _ { H } B _ { v } + c _ { L } M ,
$$

excluding additional screening and judge calls on calibration samples.

Under optimal calibration allocation, minimizing this cost subject to

$$
\frac { A _ { v } } { B _ { v } } + \frac { D _ { v } } { M } \leq \tau ^ { 2 }
$$

gives

$$
C _ { \mathrm { j u d g e } } ^ { * } = \frac { \left( \sqrt { c _ { H } A _ { v } } + \sqrt { c _ { L } D _ { v } } \right) ^ { 2 } } { \tau ^ { 2 } } .
$$

Direct random auditing costs

$$
C _ { \mathrm { d i r e c t } } = \frac { c _ { H } \theta ( 1 - \theta ) } { \tau ^ { 2 } } .
$$

The optimization uses continuous sample sizes and the preceding variance approximation; its derivation is given in Appendix A.

Judge-assisted evaluation is therefore less expensive under these assumptions when

$$
\left( \sqrt { A _ { v } } + \sqrt { \frac { c _ { L } } { c _ { H } } D _ { v } } \right) ^ { 2 } < \theta ( 1 - \theta ) .\tag{8}
$$

This condition accounts for both calibration uncertainty and automated-evaluation cost. When $c _ { L } / c _ { H }  0$ , it reduces to

$$
A _ { v } < \theta ( 1 - \theta ) \quad \Longleftrightarrow \quad \eta > 1 ,
$$

recovering the label-efficiency condition in Eq. (7).

## 4 Experiments and Results

## 4.1 Experimental Setup

Benchmarks and scope. We evaluate whether confidence and cross-model verification can reduce human labeling in model-accuracy audits. MMLU-Pro (Wang et al., 2024) and MMMU-Pro (Yue et al., 2025) provide exact multiple-choice ground truth for text and multimodal evaluation, respectively. We use 8,000 MMLU-Pro questions and a fixed 1,000-question MMMU-Pro subset, covering ten and eight model runs; primary MMLU-Pro comparisons require at least 80% coverage. Full model and preprocessing details are given in Appendix B.

Audit simulation. Complete benchmark ground truth is used to simulate partial human verification. Outputs are stratified by confidence, and sampling without replacement is repeated 5,000 times across five audit budgets. Only sampled labels are used for estimation; full ground truth is used to assess error and construct oracle baselines.

Verifier evaluation. Each of eight models is evaluated against the other seven, yielding 56 directed target–verifier pairs per benchmark. We report judgment accuracy, sensitivity, false-positive rate, discrimination gap, and label efficiency. The primary analysis uses binary judgments; full details are given in Appendix B.

Cost model. We estimate each verifier’s perjudgment cost from its observed input and output tokens and standard API prices on October 10, 2026. Original answer-generation costs are excluded because they are common to all audit strategies. Estimates also exclude retries, discounts, taxes, and surcharges.

![](images/d003097febd080909e0c787922317913140a35b2661926c1299c7bd3f0267397.jpg)  
Figure 2: MMLU-Pro confidence calibration. a, Reliability curves. b, Confidence–accuracy gaps. c, Expected calibration error and Brier score. d, Accuracy and mean confidence. Grey cells contain fewer than 20 observations; daggers mark incomplete runs.

## 4.2 Self-Confidence Is Informative but Miscalibrated Across Benchmarks

Self-reported confidence generally tracks accuracy but is not a calibrated probability of correctness. On MMLU-Pro (Fig. 2), higher-confidence outputs are typically more accurate, yet panels a and b show substantial, model-dependent miscalibration. Claude Sonnet 5 is the best calibrated, while several models remain strongly overconfident; panels c and d confirm persistent confidence–accuracy gaps.

The same pattern appears on MMMU-Pro (Appendix Fig. 6). All eight models are overconfident on average; even Claude Sonnet 5 retains a 10.1-point confidence–accuracy gap and an ECE of 0.108. Across models, accuracy ranges from 27.2% to 63.5%, with gaps reaching 59.6 points. Thus, confidence is informative across both benchmarks but unreliable as a direct probability of correctness.

This distinction motivates confidence-stratified auditing. Rather than inferring correctness from confidence, we use it only to partition outputs and estimate population accuracy from representative samples within each stratum.

## 4.3 Partial Audits Reliably Recover Population Accuracy

Figure 3 shows that confidence-stratified auditing reliably recovers GPT-5.5’s MMLU-Pro accuracy of 80.54%. Across budgets and allocation strategies, bias remains near zero (panel a), while estimation error decreases with additional human labels (panel c).

Allocation primarily affects precision. Uniform allocation performs worse at small budgets, whereas proportional and oracle Neyman allocations are nearly identical. With 200 labels, RMSE decreases from 4.56 percentage points under uniform allocation to 2.46 and 2.41 percentage points under proportional and oracle Neyman allocation, respectively. This reflects the long-tailed confidence distribution: equal allocation oversamples sparse strata and undersamples dominant ones, while the $\pi _ { j } ^ { 2 } / n _ { j }$ term favors larger strata.

Panels b and d show that observed failure probabilities remain below the corresponding Hoeffding bounds. The bounds are conservative but provide valid worst-case guarantees. MMMU-Pro shows the same overall pattern, with greater variability at small budgets (Appendix Fig. 7).

![](images/5646dd4c0973cb48745596542bdb220f10250be8489bfad61fa12396eda179da.jpg)  
Figure 3: Empirical validation of confidence-stratified partial auditing on MMLU-Pro. a, Bias with 95% Monte Carlo intervals. b, Five-point exceedance frequency with 95% Wilson intervals and Hoeffding bounds. c, RMSE. d, Exceedance frequencies and bounds. Zero observed exceedances are plotted at 0.0001.

## 4.4 Verifier Efficiency across Benchmarks

Figure 4 shows that verifier accuracy, discrimination, and label efficiency are distinct. Panel a reports generally high judgment accuracy, while panel b shows much larger variation in the discrimination gap $s - \alpha _ { v }$ . Panel c translates these operating characteristics into human-label efficiency: despite strong raw accuracy, only 6 of 56 MMLU-Pro target–verifier pairs achieve $B _ { \mathrm { d i r e c t } } / B _ { v } > 1$ Panel d explains this pattern: most verifiers have high sensitivity, but several exhibit substantial falsepositive rates, weakening discrimination. GPT-5.5 provides the strongest pooled operating point.

This distinction is important because similar judgment accuracy can imply very different statistical value. For example, Claude Sonnet 5 achieves 0.84 judgment accuracy when verifying GPT-5.5 but only 0.25 label efficiency, whereas GPT-5.5 verifying Llama-3.1-8B achieves 0.89 accuracy and the highest efficiency ratio, 1.45. Thus, efficient population-accuracy recovery depends more directly on separating correct from incorrect answers than on overall agreement.

The pattern is more pronounced on MMMU-Pro (Appendix Fig. 8). Across 413,562 MMLU-Pro judgments and 55,971 MMMU-Pro judgments, no MMMU-Pro pair exceeds the label-efficiency threshold. GPT-5.5, for example, reaches 0.700 judgment accuracy and 0.844 sensitivity, but its false-positive rate of 0.402 limits the discrimination gap to 0.442.

Figure 5 adds inference cost. Its six finite breakeven cells correspond exactly to the six MMLU-Pro pairs with $B _ { \mathrm { d i r e c t } } / B _ { v } > 1$ . The most favorable case is GPT-5.5 verifying Llama-3.1-8B, with $c _ { H } ^ { \star } \approx \mathfrak { F } 0 . 0 9$ , while the remaining thresholds are approximately \$0.34, \$0.75, \$0.80, \$4.33, and \$21.55. Pairs only slightly above the efficiency boundary can therefore require very high human-label costs before their statistical advantage offsets inference cost.

Overall, verifier discrimination determines the human-calibration burden, while inference cost determines how cheaply automated-audit uncertainty can be reduced. Cheaper LLM inference improves precision by enabling more verifier judgments, but it cannot remove the calibration floor.

## 5 Conclusion

We introduced a unified framework for estimating LLM and MLLM accuracy under limited human verification. Across MMLU-Pro and MMMU-Pro, self-reported confidence was miscalibrated but remained useful for stratified auditing, which recovered population accuracy with low bias. We modeled LLM-based verification through sensitivity and false-positive rate. High judgment accuracy alone does not guarantee efficient auditing; verifier discrimination, calibration uncertainty, and inference cost determine its benefits.

![](images/bc9fe258bf3edf2cc1c2fec54bcb965508747bcd4e8c23fbb47c3b55d5ea30c0.jpg)

![](images/799f81a7ea7803cd93699c93cc12e14d044536220519c5a4aa476244e803b017.jpg)

![](images/3e57c19ef2f1755405a2da5f1b6c145c02db70225754026b6a08bc4c280740dc.jpg)

![](images/c006f9c22f0f0718caea9984231d1a3771bd85242435d82112366bf6f81365c5.jpg)  
Figure 4: Observed cross-model verifier performance on MMLU-Pro. a, Binary judgment accuracy. b, Discrimination gap $s - \alpha _ { v } . \textbf { c }$ , Idealized label-efficiency ratio $B _ { \mathrm { d i r e c t } } / B _ { v } ; { \bf d }$ , Pooled verifier operating points in false-positive-rate–sensitivity space; better discrimination lies closer to the upper-left corner.

![](images/928509e8e444c73a7cb15b9a8ec4d778c71a0c58af86ddc7a6811e9ef49267a5.jpg)  
Figure 5: Break-even human-label costs on MMLU-Pro. Lower values favor verifier-assisted evaluation. Gray cells have no finite break-even point.

## Limitations

This study focuses on multiple-choice LLM/MLLM benchmarks with known references and simulated partial auditing, assuming representative sampling and stable verifier behavior. Human-label noise, distribution shift, open-ended evaluation, and continuous verifier confidence are not fully modeled. The cost analysis relies on observed token use and current API-price proxies, while actual human and inference costs may vary; thus, the derived efficiency and break-even boundaries should be interpreted as analytical guidance rather than universal deployment guarantees.

## Acknowledgments

The authors acknowledge support from an NVIDIA Academic Grant Award.

## References

Anastasios N Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I Jordan, and Tijana Zrnic. 2023a. Prediction-powered inference. Science, 382(6671):669–674.

Anastasios N. Angelopoulos, John C. Duchi, and Tijana Zrnic. 2023b. PPI++: Efficient prediction-powered inference. arXiv preprint arXiv:2311.01453.

Anthropic. 2025. Introducing Claude Haiku 4.5. Accessed: 2026-10-11.

Anthropic. 2026. Introducing Claude Sonnet 5. Accessed: 2026-10-11.

Colin B Begg and Robert A Greenes. 1983. Assessment of diagnostic tests when disease verification is subject to selection bias. Biometrics, pages 207–215.

Pierre Boyeau, Anastasios Nikolas Angelopoulos, Tianle Li, Nir Yosef, Jitendra Malik, and Michael I. Jordan. 2025. Autoeval done right: Using synthetic data for model evaluation. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net.

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. 2024. Mllm-asa-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, pages 6562–6595. PMLR / OpenReview.net.

Yiqun T Chen, Sizhu Lu, Sijia Li, Moran Guo, and Shengyi Li. 2026. Efficient inference for noisy llm-as-a-judge evaluation. arXiv preprint arXiv:2601.05420.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, pages 8359–8388. PMLR / OpenReview.net.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Benjamin Eyre and David Madras. 2025. Regression for the mean: Auto-evaluation and inference with few labels through post-hoc regression. In Fortysecond International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net.

Chen Feng, Minghe Shen, Ananth Balashankar, Carsten Gerner-Beuerle, and Miguel R. D. Rodrigues. 2026. Noisy but valid: Robust statistical evaluation of llms with imperfect judges. In The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc.

Adam Fisch, Joshua Maynez, R. Alex Hofer, Bhuwan Dhingra, Amir Globerson, and William W. Cohen. 2024. Stratified prediction-powered inference for effective hybrid evaluation of language models. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. CoRR, abs/1705.08500.

Google. 2026. Gemini api models. Accessed: 2026-10- 11.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pages 1321– 1330. PMLR.

Chaeyun Jang, Moonseok Choi, Yegon Kim, Seungyoo Lee, Juho Lee, and Hyungi Lee. 2026. Confidence is not universal: Task-dependent calibration and emergent behavior in LLMs. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 50548–50571. PMLR.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. 2021. Active testing: Sample-efficient model evaluation. In Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings ofMachine Learning Research, pages 5753–5763. PMLR.

Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. 2022. Active surrogate estimators: An active learning approach to label-efficient model evaluation. CoRR, abs/2202.06881.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Chungpa Lee, Thomas Zeng, Jongwon Jeong, Jy-yong Sohn, and Kangwook Lee. 2025. How to correctly report llm-as-a-judge evaluations. CoRR, abs/2511.21140.

Seongyun Lee, Seungone Kim, Sue Hyun Park, Geewook Kim, and Minjoon Seo. 2024. Prometheusvision: Vision-language model as a judge for finegrained evaluation. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, volume ACL 2024 of Findings ofACL, pages 11286– 11315. Association for Computational Linguistics.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching models to express their uncertainty in words. Transactions on Machine Learning Research.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 2511–2522. Association for Computational Linguistics.

Pranav Mani, Peng Xu, Zachary C. Lipton, and Michael Oberst. 2025. No free lunch: Non-asymptotic analysis of prediction-powered inference. CoRR, abs/2505.20178.

Meta AI. 2025. Llama 4 Scout 17B 16E Instruct model card. Accessed: 2026-10-11.

OpenAI. 2026a. Introducing GPT-5.4 mini and nano. Accessed: 2026-10-11.

OpenAI. 2026b. Introducing GPT-5.5. Accessed: 2026- 10-11.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Sangwoo Park, Matteo Zecchin, and Osvaldo Simeone. 2025. Adaptive prediction-powered autoeval with reliability and efficiency guarantees. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing

Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Qwen Team. 2025. Qwen3-VL-30B-A3B-Instruct model card. Accessed: 2026-10-11.

Qwen Team. 2026. Qwen3.5-35B-A3B model card. Accessed: 2026-10-11.

Walter J Rogan and Beth Gladen. 1978. Estimating prevalence from the results of a screening test. American journal ofepidemiology, 107(1):71–76.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 5433–5442. Association for Computational Linguistics.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024. Mmlu-pro: a more robust and challenging multi-task language understanding benchmark. In Proceedings of the 38th International Conference on Neural Information Processing Systems, pages 95266–95290.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Weihao Xuan, Qingcheng Zeng, Heli Qi, Junjue Wang, and Naoto Yokoya. 2025. Seeing is believing, but how much? A comprehensive analysis of verbalized calibration in vision-language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 1408–1450. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. 2025. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134– 15186.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In

Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2025. Judgelm: Fine-tuned large language models are scalable judges. In International Conference on Learning Representations, volume 2025, pages 51257–51296.

## A Derivation of the Cost-Efficiency Boundary

This appendix derives the minimum cost of verifierassisted evaluation when both human calibration labels and automated LLM judgments contribute to estimation uncertainty.

From Eq. (5), after optimally allocating the human calibration labels between correct and incorrect target outputs, the verifier-assisted variance can be written as

$$
\mathrm { V a r } ( \widehat { \theta } _ { v } ) \approx \frac { A _ { v } } { B _ { v } } + \frac { D _ { v } } { M } ,\tag{9}
$$

where

$$
A _ { v } = \frac { \left[ \theta \sqrt { s ( 1 - s ) } + ( 1 - \theta ) \sqrt { \alpha _ { v } ( 1 - \alpha _ { v } ) } \right] ^ { 2 } } { ( s - \alpha _ { v } ) ^ { 2 } } ,\tag{10}
$$

and

$$
D _ { v } = \frac { q ( 1 - q ) } { ( s - \alpha _ { v } ) ^ { 2 } } .\tag{11}
$$

Here, $B _ { v } ~ = ~ n _ { 1 } + n _ { 0 }$ is the number of humanlabeled examples used to calibrate the verifier, while M is the number of automated verifier judgments. The first term in Eq. (9) therefore represents uncertainty due to human calibration, whereas the second represents uncertainty in the estimated automated acceptance rate.

Let $c _ { H }$ denote the cost of one human reference label and $c _ { L }$ the cost of one LLM-judge evaluation. The total cost of verifier-assisted evaluation is

$$
C _ { \mathrm { j u d g e } } = c _ { H } B _ { v } + c _ { L } M .\tag{12}
$$

For a desired estimation variance $\tau ^ { 2 }$ , the minimumcost design solves

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { B _ { v } , M > 0 } } & { c _ { H } B _ { v } + c _ { L } M } \\ { \mathrm { s . t . } } & { \displaystyle \frac { A _ { v } } { B _ { v } } + \frac { D _ { v } } { M } \leq \tau ^ { 2 } . } \end{array}\tag{13}
$$

At the optimum, the variance constraint is active; otherwise either $B _ { v }$ or M could be reduced while preserving the required precision.

Introducing a Lagrange multiplier λ, the Lagrangian is

$$
\mathcal { L } = c _ { H } B _ { v } + c _ { L } M + \lambda \left( \frac { A _ { v } } { B _ { v } } + \frac { D _ { v } } { M } - \tau ^ { 2 } \right) .\tag{14}
$$

The first-order conditions are

$$
c _ { H } = \lambda \frac { A _ { v } } { B _ { v } ^ { 2 } } , \qquad c _ { L } = \lambda \frac { D _ { v } } { M ^ { 2 } } .\tag{15}
$$

Eliminating λ gives the optimal resource ratio

$$
{ \frac { B _ { v } } { M } } = \sqrt { \frac { A _ { v } c _ { L } } { D _ { v } c _ { H } } } .\tag{16}
$$

Thus, more human calibration is favored when calibration uncertainty $A _ { v }$ is large or human labeling is relatively inexpensive, whereas more automated evaluation is favored when acceptance-rate uncertainty $D _ { v }$ is large or automated judgments are inexpensive.

Substituting Eq. (16) into the active variance constraint yields

$$
M ^ { * } = \frac { D _ { v } + \sqrt { A _ { v } D _ { v } c _ { H } / c _ { L } } } { \tau ^ { 2 } } ,\tag{17}
$$

and

$$
B _ { v } ^ { * } = \frac { A _ { v } + \sqrt { A _ { v } D _ { v } c _ { L } / c _ { H } } } { \tau ^ { 2 } } .\tag{18}
$$

Substitution into Eq. (12) gives

$$
\begin{array} { l } { { C _ { \mathrm { j u d g e } } ^ { * } = c _ { H } B _ { v } ^ { * } + c _ { L } M ^ { * } } } \\ { { \ = { \frac { c _ { H } A _ { v } + c _ { L } D _ { v } + 2 \sqrt { c _ { H } c _ { L } A _ { v } D _ { v } } } { \tau ^ { 2 } } } } } \end{array}\tag{19}
$$

$$
= \frac { \left( \sqrt { c _ { H } A _ { v } } + \sqrt { c _ { L } D _ { v } } \right) ^ { 2 } } { \tau ^ { 2 } } .\tag{20}
$$

(21)

For comparison, direct random auditing of binary correctness has approximate variance

$$
\mathrm { V a r } ( \widehat { \theta } _ { \mathrm { d i r e c t } } ) \approx \frac { \theta ( 1 - \theta ) } { B _ { \mathrm { d i r e c t } } } .
$$

Achieving the same target variance $\tau ^ { 2 }$ therefore requires

$$
B _ { \mathrm { d i r e c t } } = \frac { \theta ( 1 - \theta ) } { \tau ^ { 2 } } ,\tag{22}
$$

with corresponding cost

$$
C _ { \mathrm { d i r e c t } } = \frac { c _ { H } \theta ( 1 - \theta ) } { \tau ^ { 2 } } .\tag{23}
$$

Verifier-assisted evaluation is more cost efficient whenever

$$
C _ { \mathrm { j u d g e } } ^ { \ast } < C _ { \mathrm { d i r e c t } } .
$$

Using Eqs. (21) and (23), this condition becomes

$$
\left( \sqrt { A _ { v } } + \sqrt { \frac { c _ { L } } { c _ { H } } D _ { v } } \right) ^ { 2 } < \theta ( 1 - \theta ) .\tag{24}
$$

Equation (24) is the cost-efficiency boundary reported in the main text. Notably, the target variance $\tau ^ { 2 }$ cancels from the comparison: under the asymptotic approximations used here, changing the desired precision scales both strategies’ costs proportionally but does not change which strategy is favored.

For a target–verifier pair satisfying $A _ { v } < \theta ( 1 -$ $\theta )$ , solving Eq. (24) for the human-label cost gives the break-even value

$$
c _ { H } ^ { \star } = \frac { c _ { L } D _ { v } } { \left[ \sqrt { \theta ( 1 - \theta ) } - \sqrt { A _ { v } } \right] ^ { 2 } } .\tag{25}
$$

Verifier-assisted evaluation has lower optimized total cost than direct auditing when $c _ { H } > c _ { H } ^ { \star }$ . If $A _ { v } \geq \theta ( 1 - \theta )$ , no finite positive human-label cost can make verifier-assisted evaluation preferable under this criterion.

The human-label-only comparison is recovered as a limiting case. When automated judgment cost is negligible relative to human labeling, $c _ { L } / c _ { H } $ 0, Eq. (24) reduces to

$$
A _ { v } < \theta ( 1 - \theta ) ,\tag{26}
$$

or equivalently

$$
\eta = \frac { \theta ( 1 - \theta ) } { A _ { v } } > 1 .
$$

Thus, the label-efficiency ratio in Eq. (7) is the zero-inference-cost limit of the more general costefficiency criterion.

These expressions use continuous sample sizes and asymptotic variance approximations. In implementation, $B _ { v }$ and M must be integer-valued and may be constrained by the available evaluation population. Moreover, the correctness classes required to estimate s and $\alpha _ { v }$ are not known before human review, so obtaining $n _ { 1 }$ correct and $n _ { 0 }$ incorrect calibration examples may require additional screening. Distribution shift between calibration and evaluation, finite-population corrections, and other operational costs can further affect realized savings. The derived boundary should therefore be interpreted as an analytical cost-efficiency criterion rather than a guaranteed deployment cost.

## B Evaluation Protocol and Reproducibility

## B.1 Model Outputs and Calibration

For MMLU-Pro, each model received the question and answer options and was asked for JSON containing one option letter, a justification of at most 30 words, and an integer probability of correctness from 1 to 100. Confidence was divided by

100; parsing required a valid option and numeric confidence, with values clipped to the requested range. Hosted runs used structured JSON where supported, a 320-token output limit, and no explicitly supplied temperature. The archived protocol describes vLLM inference with temperature zero, structured JSON, and a 220-token output limit.

The evaluated systems span the GPT family from OpenAI (OpenAI, 2026b,a), the Claude family from Anthropic (Anthropic, 2026, 2025), the Gemini family from Google (Google, 2026), the Llama family from Meta (Dubey et al., 2024; Meta AI, 2025), and the Qwen family (Yang et al., 2025; Qwen Team, 2026, 2025). Exact evaluated API identifiers or open-weight checkpoints are reported below.

The ten available MMLU-Pro calibration runs are Claude Haiku 4.5, Claude Sonnet 5, GPT-5.4 mini, GPT-5.4 nano, GPT-5.5, Gemini 3.1 Flash-Lite, Gemini 3.5 Flash, Llama-3.1-8B, Qwen3- 30B-A3B-Thinking-2507, and Qwen3.5-35B-A3B. The non-self verifier experiment uses the eight non-Gemini models in this list. MMLU-Pro calibration comparisons requiring at least 80% coverage exclude runs with fewer than 6,400 valid outputs from those panels, while descriptive summaries retain them with a dagger marker.

Reliability curves use ten equal-width confidence bins. Expected calibration error weights each bin’s absolute confidence–accuracy gap by its population frequency. The Brier score averages squared differences between confidence and binary correctness. Calibration-gap displays suppress bins with fewer than 20 observations on MMLU-Pro or 25 observations on MMMU-Pro.

The MMMU-Pro analysis uses the first 1,000 examples, in dataset order, from the test split of the standard (10 options) configuration of MMMU/MMMU\_Pro, fixed at revision 563f3e84bb3b90893083a1f039cfa13077f2302b. The selected questions contain 1,135 associated images. The eight usable runs are GPT-5.5, Claude Sonnet 5, GPT-5.4 mini, GPT-5.4 nano, Claude Haiku 4.5, Qwen3.5-35B, Qwen3-VL-30B, and Llama-4 Scout 17B-16E-Instruct. Claude Sonnet 5 contains 998 valid outputs, and each other run contains 1,000.

## B.2 Fixed and Adaptive Audits

The initial strata are [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], and (0.8, 1]. Adjacent strata are merged until each retained stratum contains at least 25 outputs. Integer allocations obey $1 \leq n _ { j } \leq N _ { j }$ and sum to the budget. Uniform allocation distributes counts as evenly as capacities allow. Proportional and oracle Neyman allocations assign each remaining audit to the stratum giving the greatest decrease in their respective convex objectives, $\textstyle \sum _ { j } \pi _ { j } ^ { 2 } / n _ { j }$ and $\textstyle \sum _ { j } \pi _ { j } ^ { 2 } S _ { j } ^ { 2 } / n _ { j }$

Appendix C gives the allocation formulas, pilot stabilization, and two-stage estimator. The adaptive design is compared empirically but excluded from fixed-allocation concentration validation.

The plotted MMLU-Pro GPT-5.5 audit simulation uses random seed 20260807. The MMMU-Pro multi-model audit simulation uses master seed 20260807 and NumPy’s SeedSequence to generate one child stream per model; the plotted GPT-5.5 population uses child seed 4025595370. Every budget–allocation condition uses 5,000 independent audit replications. Bias is the mean estimate minus population accuracy, and RMSE is the square root of the mean squared estimation error. Bias intervals use the Monte Carlo standard error, whereas exceedance frequencies use Wilson intervals. Error tolerances are 2.5, 5, and 7.5 percentage points. Zero observed exceedances are plotted at 0.0001 only to permit logarithmic axes; their recorded frequency is zero.

Additional MMLU-Pro populations described in the source are Claude Sonnet 5 (7,561 outputs; accuracy 0.872900), GPT-5.4 nano (8,000; 0.407750), and Qwen3.5-35B-A3B (8,000; 0.635125). The present empirical audit figures depict GPT-5.5; no unprovided audit figure for these additional populations is claimed.

## B.3 Verifier Judgments

A verifier received the question, answer options, and the target model’s selected answer and returned a correct/incorrect judgment together with an estimated probability that the selected target answer was correct. The primary analysis uses the binary judgment. On MMLU-Pro, 19 of 413,581 requested non-self judgments were unparseable, leaving 413,562 valid judgments. MMMU-Pro target– verifier pairs contain 997–1,000 usable judgments, yielding 55,971 valid judgments in total. Rates are computed separately for each target–verifier pair; pooled operating points aggregate eligible confusion-matrix counts. The full available pair population is used for the efficiency calculation rather than a selected high-confidence subset.

Sensitivity is the fraction of correct target answers accepted; the false-positive rate is the fraction of incorrect answers accepted. Judgment accuracy counts both correct acceptance and correct rejection. Complete-sample estimates are substituted into Eq. (7). This is a descriptive comparison, with neither a held-out calibration pool nor an implemented sequential class-enrichment policy.

## B.4 Artifact Availability

MMLU-Pro and MMMU-Pro are publicly available benchmark datasets. Prompts, selected model outputs, cross-model verifier judgments, the MMMU-Pro subset manifest, and derived metric and coverage summaries are available at https://github.com/harinechoi/ confidence-auditing-reproduction, subject to benchmark licenses and model-provider terms.

## C Audit Allocation Strategies

This appendix provides additional details on the audit-allocation strategies used in the experiments. The total audit budget is denoted by B, and $n _ { j }$ denotes the number of audited outputs assigned to confidence stratum j. All allocations satisfy

$$
\sum _ { j = 1 } ^ { K } n _ { j } = B , \qquad 1 \leq n _ { j } \leq N _ { j } ,
$$

so that every nonempty stratum receives at least one audit and no stratum is sampled beyond its population size.

## C.1 Fixed Audit Allocations

We consider three fixed allocation strategies: uniform, population-proportional, and oracle Neyman allocation.

Uniform allocation. Uniform allocation distributes the audit budget approximately equally across the K confidence strata,

$$
n _ { j } ^ { \mathrm { u n i } } \approx \frac { B } { K } .
$$

This allocation treats all strata equally regardless of their population size or internal variability. It therefore differs from simple random sampling over the full population. In particular, when confidence scores are highly concentrated in one or two strata, equal allocation can oversample small strata and undersample the regions containing most model outputs.

Population-proportional allocation. Proportional allocation assigns audits according to stratum prevalence,

$$
n _ { j } ^ { \mathrm { p r o p } } = B \pi _ { j } , \qquad \pi _ { j } = \frac { N _ { j } } { N } ,\tag{27}
$$

before integer and capacity adjustments. This allocation is equivalent, up to stratification constraints, to preserving the population composition in the audited sample.

Its statistical role can be seen from the Hoeffding bound in Eq. (3), whose exponent depends on

$$
\sum _ { j = 1 } ^ { K } \frac { \pi _ { j } ^ { 2 } } { n _ { j } } .
$$

For a fixed budget B, minimizing this quantity subject to $\textstyle \sum _ { j } n _ { j } = B$ gives

$$
n _ { j } \propto \pi _ { j } .
$$

Thus, proportional allocation provides the strongest bound within this family of fixed stratified designs. Importantly, this result does not depend on knowing the true stratum accuracies.

Oracle Neyman allocation. If the withinstratum correctness variances $S _ { j } ^ { 2 }$ were known, the variance in Eq. (2) would be minimized by assigning more labels to strata that are both prevalent and internally uncertain. Ignoring integer and capacity constraints, the resulting Neyman allocation is

$$
n _ { j } ^ { \mathrm { N e y } } = B \frac { \pi _ { j } S _ { j } } { \sum _ { k = 1 } ^ { K } \pi _ { k } S _ { k } } .\tag{28}
$$

For binary correctness,

$$
S _ { j } ^ { 2 } \approx \theta _ { j } ( 1 - \theta _ { j } ) ,
$$

so the largest allocations are generally assigned to large strata whose accuracies are closer to 0.5. Strata that are almost entirely correct or incorrect require fewer samples because their within-stratum variance is small.

Oracle Neyman allocation is not directly implementable because the $S _ { j }$ values depend on correctness labels that are unknown before auditing. It is therefore used only as a reference for the best precision achievable when the true within-stratum variances are available.

## C.2 Why Proportional and Neyman Allocation Differ

The proportional and Neyman designs optimize different statistical objectives. To see this, consider the generic problem

$$
\operatorname* { m i n } _ { \{ n _ { j } \} } \sum _ { j = 1 } ^ { K } \frac { a _ { j } } { n _ { j } } \qquad \mathrm { s u b j e c t ~ t o } \qquad \sum _ { j = 1 } ^ { K } n _ { j } = B .\tag{29}
$$

The Lagrangian is

$$
\mathcal { L } = \sum _ { j } \frac { a _ { j } } { n _ { j } } + \lambda \left( \sum _ { j } n _ { j } - B \right) .
$$

Setting the derivative with respect to $n _ { j }$ to zero gives

$$
- \frac { a _ { j } } { n _ { j } ^ { 2 } } + \lambda = 0 ,
$$

and therefore

$$
n _ { j } = \frac { B \sqrt { a _ { j } } } { \sum _ { k } \sqrt { a _ { k } } } .
$$

For the Hoeffding denominator, $a _ { j } = \pi _ { j } ^ { 2 }$ , yielding

$$
n _ { j } \propto \pi _ { j } ,
$$

which gives proportional allocation. For the estimator variance, $a _ { j } = \pi _ { j } ^ { 2 } S _ { j } ^ { 2 }$ , yielding

$$
n _ { j } \propto \pi _ { j } S _ { j } ,
$$

which gives Neyman allocation.

The distinction is therefore fundamental: proportional allocation minimizes a distribution-free concentration criterion, whereas Neyman allocation minimizes variance by exploiting information about within-stratum correctness variability. The latter can outperform proportional allocation when variability differs substantially across strata, but that information must first be estimated.

## C.3 Two-Stage Neyman Auditing

To approximate Neyman allocation without access to the true $S _ { j }$ , we use a two-stage design. In the first stage, a small pilot sample of size $m _ { j }$ is drawn uniformly without replacement from every stratum. In the experiments, $m _ { j } = 1 0$ whenever the available stratum size permits. Let

$$
X _ { 1 j } = \sum _ { i \in \mathcal { A } _ { 1 j } } Z _ { i }
$$

denote the number of correct outputs observed in the pilot sample.

The pilot accuracy is stabilized as

$$
\widetilde { \theta } _ { 1 j } = \frac { X _ { 1 j } + 1 / 2 } { m _ { j } + 1 } .\tag{30}
$$

The small pseudo-count prevents estimated variances from becoming exactly zero when all pilot observations are correct or all are incorrect. The estimated Bernoulli standard deviation is then

$$
\widehat { S } _ { j } = \sqrt { \widetilde { \theta } _ { 1 j } \left( 1 - \widetilde { \theta } _ { 1 j } \right) } .
$$

After the pilot stage, the remaining audit budget is

$$
B _ { 2 } = B - \sum _ { j } m _ { j } .
$$

The second-stage samples are allocated approximately according to

$$
\begin{array} { r } { n _ { 2 j } \propto ( N _ { j } - m _ { j } ) \widehat { S } _ { j } , } \end{array}
$$

subject to the remaining stratum capacities and the requirement that each nonempty remainder receive sufficient representation. Thus, the pilot is used only to determine where the remaining labels should be spent.

## C.4 Estimation after Adaptive Allocation

Because the second-stage allocation depends on the pilot outcomes, simply averaging all audited observations within a stratum can introduce bias. The final estimator therefore treats the pilot and the remaining population explicitly.

Let $\widehat { \theta _ { 2 j } }$ denote the mean correctness among the $n _ { 2 j }$ outputs sampled uniformly from the $N _ { j } - m _ { j }$ outputs not observed in the pilot. The stratum total is estimated by combining the observed pilot total with an expansion estimate of the unaudited remainder:

$$
\widehat { \theta } _ { j , \mathrm { 2 s t } } = \frac { X _ { 1 j } + ( N _ { j } - m _ { j } ) \widehat { \theta } _ { 2 j } } { N _ { j } } .\tag{31}
$$

The overall population estimator is

$$
\widehat { \theta } _ { \mathrm { 2 s t } } = \sum _ { j = 1 } ^ { K } \pi _ { j } \widehat { \theta } _ { j , \mathrm { 2 s t } } .
$$

To see why this remains unbiased, condition on the realized pilot sample. Given the pilot, the second-stage allocation $n _ { 2 j }$ is fixed, while the

second-stage sample remains uniform over the unsampled portion of the stratum. Therefore,

$$
\mathbb { E } \left[ \widehat { \theta } _ { 2 j } \mid \mathcal { A } _ { 1 j } \right] = \theta _ { j , \mathrm { r e m } } ,
$$

where $\theta _ { j , \mathrm { r e m } }$ is the true accuracy among the $N _ { j } -$ $m _ { j }$ remaining outputs. Hence,

$$
\begin{array} { r } { \mathbb { E } \left[ X _ { 1 j } + ( N _ { j } - m _ { j } ) \widehat { \theta } _ { 2 j } \mid \mathcal { A } _ { 1 j } \right] = X _ { 1 j } + ( N _ { j } - m _ { j } ) \theta _ { j , \mathrm { r e m } } , } \end{array}
$$

which equals the true total number of correct outputs in stratum j. Consequently,

$$
\mathbb { E } \left[ \widehat { \theta } _ { j , \mathrm { 2 s t } } \right] = \theta _ { j } ,
$$

and population weighting preserves unbiasedness.

This distinction is important because the pilot observations influence the second-stage sample sizes. A naive pooled mean would treat pilot and secondstage observations as though they had identical inclusion mechanisms, whereas Eq. (31) explicitly accounts for the adaptive design.

## D Extension to Imperfect Human Verification

The main analysis treats human reference labels as correct observations of the latent correctness variable Z. In practice, human verification may itself be imperfect, particularly for difficult or multimodal questions. This section shows how the framework extends when human labels are noisy.

## D.1 Noisy Human Labels

Let $H \in \{ 0 , 1 \}$ denote the human assessment of whether a model answer is correct. Define the human sensitivity and false-positive rate as

$$
\begin{array} { r } { s _ { H } = \operatorname* { P r } ( H = 1 \mid Z = 1 ) , } \\ { \alpha _ { H } = \operatorname* { P r } ( H = 1 \mid Z = 0 ) . } \end{array}
$$

and let

$$
\Delta _ { H } = s _ { H } - \alpha _ { H }
$$

denote human discrimination. Perfect human verification corresponds to $s _ { H } = 1$ and $\alpha _ { H } = 0$

For a population with true model accuracy θ, the observed fraction of answers marked correct by humans is

$$
r = s _ { H } \theta + \alpha _ { H } ( 1 - \theta ) .\tag{32}
$$

Thus, treating imperfect human labels directly as ground truth estimates r rather than θ. The resulting bias is

$$
r - \theta = \alpha _ { H } ( 1 - \theta ) - ( 1 - s _ { H } ) \theta .
$$

Even small human error rates can therefore produce systematic bias when the evaluation population is large.

If $s _ { H }$ and α<sub>H</sub> are known or estimated independently, Eq. (32) can be inverted:

$$
\theta = \frac { r - \alpha _ { H } } { s _ { H } - \alpha _ { H } } = \frac { r - \alpha _ { H } } { \Delta _ { H } } .\tag{33}
$$

As with an automated verifier, identification requires $\Delta _ { H } \neq 0$

## D.2 Confidence-Stratified Auditing with Imperfect Humans

For confidence stratum $j ,$ let

$$
r _ { j } = \operatorname* { P r } ( H = 1 \mid j ) .
$$

If human operating characteristics are stable across confidence strata,

$$
r _ { j } = s _ { H } \theta _ { j } + \alpha _ { H } ( 1 - \theta _ { j } ) ,
$$

and the corrected stratum accuracy is

$$
\theta _ { j } = \frac { r _ { j } - \alpha _ { H } } { \Delta _ { H } } .\tag{34}
$$

Let

$$
\widehat { r } _ { j } = \frac { 1 } { n _ { j } } \sum _ { i \in \mathcal { A } _ { j } } H _ { i } .
$$

A corrected population estimator is then

$$
{ \widehat { \theta } } _ { \mathrm { s t r } , H } = \sum _ { j = 1 } ^ { K } \pi _ { j } { \frac { { \widehat { r } } _ { j } - \alpha _ { H } } { \Delta _ { H } } } .\tag{35}
$$

When $s H$ and $\alpha _ { H }$ are known and human errors satisfy the assumed measurement model, this estimator remains unbiased for population accuracy. Hence, imperfect human verification does not invalidate confidence-stratified auditing, but it requires an additional measurement-error correction.

With known human operating characteristics, its sampling variance is approximately

$$
\mathrm { V a r } ( \widehat { \theta } _ { \mathrm { s t r } , H } ) = \frac { 1 } { \Delta _ { H } ^ { 2 } } \sum _ { j } \pi _ { j } ^ { 2 } \left( 1 - \frac { n _ { j } } { N _ { j } } \right) \frac { S _ { H , j } ^ { 2 } } { n _ { j } } ,\tag{36}
$$

where $S _ { H , j } ^ { 2 }$ is the finite-population variance of the human labels in stratum j. Human disagreement therefore inflates uncertainty by approximately $1 / \Delta _ { H } ^ { 2 }$

## D.3 Uncertainty in Human Reliability

In practice, $s _ { H }$ and $\alpha _ { H }$ must themselves be estimated. Suppose an adjudicated calibration set contains $m _ { 1 }$ truly correct and $m _ { 0 }$ truly incorrect outputs, from which $\widehat { s } _ { H }$ and $\widehat { \alpha } _ { H }$ are estimated. Let

$$
{ \widehat { r } } _ { \mathrm { s t r } } = \sum _ { j } \pi _ { j } { \widehat { r } } _ { j } .
$$

The corrected estimator becomes

$$
\widehat { \theta } _ { H } = \frac { \widehat { r } _ { \mathrm { s t r } } - \widehat { \alpha } _ { H } } { \widehat { s } _ { H } - \widehat { \alpha } _ { H } } .
$$

A first-order delta-method approximation gives

$$
\begin{array} { r } { \mathrm { V a r } ( \widehat { \theta } _ { H } ) \approx \displaystyle \frac { 1 } { \Delta _ { H } ^ { 2 } } \Bigg [ \mathrm { V a r } ( \widehat { r } _ { \mathrm { s t r } } ) + \frac { \theta ^ { 2 } s _ { H } ( 1 - s _ { H } ) } { m _ { 1 } } } \\ { + \frac { ( 1 - \theta ) ^ { 2 } \alpha _ { H } ( 1 - \alpha _ { H } ) } { m _ { 0 } } \Bigg ] . } \end{array}\tag{37}
$$

The first term is uncertainty from the audit sample, while the remaining terms arise from estimating human reliability. Increasing the audit budget cannot eliminate the latter. Thus, sufficiently precise population evaluation may eventually be limited by uncertainty in the human reference process rather than by the number of audited model outputs.

## D.4 Implications for LLM-as-a-Judge

Imperfect human labels also affect judge calibration. If the observed human label H is naively treated as true correctness $Z ,$ the estimated judge sensitivity and false-positive rate characterize agreement with the human annotator rather than agreement with latent correctness. The efficiency boundary in Eq. (7) can consequently be distorted.

Under the additional conditional-independence assumption

$$
W \perp H \mid Z ,
$$

known human error rates permit correction. For example,

$$
\begin{array} { r l } & { \left[ \mathrm { P r } ( W = 1 , H = 1 ) \right] } \\ & { \left[ \mathrm { P r } ( W = 1 , H = 0 ) \right] } \\ &  \quad = \left[ \begin{array} { c c } { s _ { H } } & { \alpha _ { H } } \\ { 1 - s _ { H } } & { 1 - \alpha _ { H } \right] \left[ \alpha _ { v } ( 1 - \theta ) \right] . } \end{array} \end{array}
$$

When $\Delta _ { H } ~ \neq ~ 0$ , the measurement matrix is invertible, allowing the latent quantities $s \theta$ and $\alpha _ { v } ( 1 - \theta )$ —and hence the judge operating rates—to be recovered after correcting for human error.

This extension reveals an additional cost of judge-assisted evaluation: human labels used to calibrate the judge must themselves be sufficiently reliable or independently calibrated. Human error therefore does not simply add noise to the evaluation; it introduces another measurement layer whose uncertainty propagates into both direct auditing and automated-judge calibration.

## D.5 Practical Interpretation

Three regimes are distinguished. If human error is negligible, the main-text analysis applies directly. If human sensitivity and false-positive rates can be estimated from adjudicated examples, Eqs. (33)– (37) provide an error-corrected extension. If human reliability is unknown, however, population accuracy is generally not point identified from a single noisy human label per output without additional assumptions or repeated/adjudicated annotations.

Accordingly, the perfect-human assumption in the main analysis should be interpreted as defining the statistical value of a trusted reference label. In practical deployments, disagreement resolution, repeated annotation, or expert adjudication can be used to approximate this reference process, with their additional costs incorporated into c<sub>H</sub>.

## E Additional Empirical Figures

The additional figures report confidence calibration, repeated-audit validation, and cross-model verification on MMMU-Pro.

![](images/3497fc7adb11cfd242cf6d51ac80fb45a11ed890d152c0a91bc8ad9ae834ca4d.jpg)

![](images/ebb98581dd18a6589dbc9642a016c17824b1ef821cdc0192f734a98edcd3e849.jpg)

![](images/40ba6ed88514f8738c7e32d7656a3296527b96cf5b70fc631004beb4cc1f360e.jpg)

![](images/a411ad612f271a171aa7e474a611a5f114b3270c2dbe4df0c5bcc9c0b851000f.jpg)  
Figure 6: MMMU-Pro calibration. a, Reliability curves. b, Confidence minus accuracy by bin; grey cells have fewer than 25 observations. c, ECE and Brier score. d, Accuracy and mean confidence. Seven runs contain 1,000 outputs; Sonnet 5 contains 998. The diagonal denotes perfect calibration.

![](images/35affb5ddcdc57e1d9d313a099f2228df217f4b85f317f50c774991b04d2ae51.jpg)

![](images/ec8a67ad34f215f99a538ecc44194f81d3a3b46537b3cc2a109bec4fab77ef41.jpg)

![](images/6bbc1557e2cfe60193ddb51042a8eba7c5184709181087c91723d011897e335c.jpg)

![](images/03ea7a7c7204c9f2b7b0e01324390dd78d267be1f0fe78275df1d866297f5804.jpg)  
Figure 7: MMMU-Pro audits. GPT-5.5 results over 5,000 replications with four merged strata. Panels follow Fig. 3: a, bias; b, five-point exceedances and fixed-allocation bounds; c, RMSE; d, tolerance sensitivity. Intervals are 95% Monte Carlo (a) and Wilson (b); zero failures are plotted at 0.0001.

![](images/40bf2633c5ad19076bca6f3562b49fc9f3ebbce28db251424331d4e3150077a1.jpg)

![](images/2518ce29527d619241a15ad152b94139d291c13934bfa9305e0b64d2a395b275.jpg)

![](images/c98fe538aa4469d0a9867a9e00012e3b5953490c62f1cacbb0fc8e8eeee609b8.jpg)

![](images/90c9a176b064c72a5c483d8c3894e88f6039c02c82d9148f1b6389e573eb0de2.jpg)  
Figure 8: MMMU-Pro verification. Panels follow Fig. 4. No pair has an idealized label-efficiency ratio above one. Self-verification is excluded.