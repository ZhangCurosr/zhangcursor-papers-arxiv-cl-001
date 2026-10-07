# DOES STEERING BREAK YOUR MODEL? A MULTI-DIMENSIONAL EVALUATION SUITE FOR LLM STEER-ING METHODS

Haotian Yang<sup>1,2,∗</sup> Huikang Jiang<sup>2,3,∗</sup> Yucheng Wu<sup>1,2,∗</sup> Wen-Jie Jiang<sup>2</sup> Chenpeng Wang<sup>4</sup> Yibin Lou<sup>2,5</sup> Liangming Pan<sup>1,2,†</sup>

<sup>1</sup>State Key Laboratory of Multimedia Information Processing, Peking University <sup>2</sup>School of Computer Science, Peking University <sup>3</sup>Columbia University <sup>4</sup>YiXin-AILab, YIXIN <sup>5</sup>Southern University of Science and Technology

## ABSTRACT

Activation steering provides a lightweight and flexible way to control large language model (LLM) behavior. However, effective steering requires more than inducing the intended behavior: it should also limit unintended changes and remain robust across inputs and training data. Existing evaluations cover these dimensions only in fragments. As a result, the trade-offs between efficacy and side effects have not been systematically characterized. We introduce SteerScope, a two-axis, multi-dimensional evaluation suite that jointly characterizes steering outcomes and method properties through 15 metrics. We score target efficacy and side effects on language quality, task capabilities, and safety and reliability, and further assess generalization and data dependence through steering-specific metrics for sample efficiency and sample sensitivity. Rather than comparing methods at a single operating point, we characterize their efficacy–side-effect tradeoffs. Under matched models, tasks, and evaluation protocols, we benchmark 23 methods spanning 4 families, including prompting, LoRA, and SFT as baseline methods, and release the suite as an extensible codebase. We find that current activation steering methods do not yet surpass the Prompt Steering baseline in their overall balance between steering efficacy and side effects: across both model scales, no evaluated activation steering method achieves higher efficacy without incurring greater composite side effects. We further uncover a consistent coupling between steering efficacy and side effects. Under OOD prompts, target efficacy is often preserved, whereas side effects tend to become more pronounced, particularly through declines in instruction relevance and fluency. Methods also exhibit sharply different sample-efficiency profiles.

## § htt<sub>p</sub>s://<sub>g</sub>ithub.com/<sub>y</sub>ht0511/steersco<sub>p</sub>e

## 1 INTRODUCTION

As large language models (LLMs) are deployed across diverse tasks and applications, fine-grained control over their behavior has become increasingly important (Ouyang et al., 2022). Prompting and fine-tuning are widely used for this purpose. However, prompting lacks an explicit mechanism for continuously adjusting the strength of a desired behavior, while fine-tuning requires additional training and parameter updates (Wu et al., 2025a). Recently, activation steering has emerged as a promising approach to behavioral control. It intervenes in internal representations, such as the residual stream, to induce or suppress specific concepts, styles, and safety-related behaviors (Turner et al., 2023; Zou et al., 2023; Rimsky et al., 2024). By keeping the backbone parameters frozen and allowing intervention strength to be adjusted, activation steering is more lightweight and adjustable.

<table><tr><td rowspan=13 colspan=2>SteerScope                  AxBenchEfficacy         Concept ExpressionLanguage    Fluency日QualitySteeringLanguage UnderstandingOutcomesKnowledgeTaskSide Effects      Capability    ReasoningInstruction FollowingJailbreak &amp; RefusalSafety &amp;+           BiasReliabilityTruthfulnessGeneralizationMethodProperties        Data         Sample EfficiencyDependenceSample Sensitivity</td><td rowspan=1 colspan=1>FaithSteer-BENCH</td><td rowspan=1 colspan=1>Tan et al.</td><td rowspan=1 colspan=1>SteeringSafet</td><td rowspan=1 colspan=1>y CLaS-Bench</td><td rowspan=1 colspan=1>Bhalla et al.</td><td rowspan=1 colspan=1>Sprejer et al.</td><td rowspan=1 colspan=1>SteerEval</td><td rowspan=1 colspan=1>SteerScope</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>△</td><td rowspan=1 colspan=1>△</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>△</td><td rowspan=1 colspan=1>△</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1></td></tr></table>

Figure 1: Overview of the SteerScope evaluation suite and benchmark coverage. Left: our hierarchical evaluation taxonomy. Right: coverage matrix of existing steering benchmarks, where rows denote taxonomy nodes and columns denote evaluation works (last column: SteerScope). A ✓ indicates a corresponding metric, ✗ indicates none, and △ indicates that the dimension is considered indirectly but not measured independently.

However, the effects of steering are not always confined to the intended target. Prior work has reported multi-dimensional side effects across language quality, task capabilities, safety, and reliability. These include degraded generation quality and task performance, as well as changes in bias, refusal, and jailbreak susceptibility (Stickland et al., 2024; Bhalla et al., 2024; Sprejer et al., 2026; Li et al., 2026d). Existing evidence suggests that these side effects may arise from concept entanglement, whereby concepts are not behaviorally independent and steering one can unintentionally alter others (Durmus et al., 2024; Siu et al., 2026; Li et al., 2026d). Beyond these side effects, steering outcomes may vary across input distributions. They may also depend on the amount and composition of the data used to construct the intervention (Tan et al., 2024; Braun et al., 2025; Ding et al., 2026). Together, these challenges motivate a comprehensive evaluation framework organized around two questions: what steering changes and when those changes hold.

Recent work has begun to establish benchmarks for evaluating LLM steering, such as AxBench (Wu et al., 2025a), SteeringSafety (Siu et al., 2026), FaithSteer-BENCH (Ding et al., 2026), and CLaS-Bench (Gurgurov et al., 2026). However, existing benchmarks cover the relevant dimensions only in fragments, leaving an incomplete picture of what steering changes and when its effects hold. For example, AxBench limits side-effect evaluation to instruction following and fluency. SteeringSafety centers on safety entanglement, while CLaS-Bench specializes in cross-lingual language control. FaithSteer-BENCH focuses on deployment-time reliability, but evaluates each method at a single calibrated operating point and does not cover side effects on language quality or safety. Fig. 1 (right) illustrates these gaps in coverage.

To bridge this gap, we introduce SteerScope, a comprehensive, multi-dimensional evaluation suite for LLM steering.

1. Systematic evaluation taxonomy. SteerScope organizes 15 metrics into a hierarchical taxon omy with two complementary axes (Fig. 1, left). Steering Outcomes examines what steering changes, covering target efficacy and side effects on language quality, task capabilities, and safety and reliability. Method Properties examines when steering effects hold, covering generalization under cross-lingual prompt shifts and dependence on training data.

2. Broad method coverage. We evaluate 23 methods spanning 4 families on 2 models. These include diverse activation steering methods alongside prompting, LoRA, and SFT baselines.

3. Efficacy–side-effect trade-off characterization. We characterize each adjustable method through its efficacy–side-effect trade-offs, revealing how unintended effects change alongside target efficacy.

Our central finding is that current activation steering methods do not outperform the Prompt Steering baseline in the efficacy–side-effect trade-off on either evaluated model. Although some activation steering methods outperform this baseline on existing benchmarks, none achieves higher target efficacy without greater composite side effects under our default aggregation. On Gemma-2-9B-it, for example, A-PSR improves the Concept Expression score by 1.30 points but incurs an average normalized degradation of 0.12. Consistent with this, steering efficacy and side effects generally increase together as intervention strength grows. Nevertheless, activation steering remains useful because its adjustable intervention strength provides access to high-efficacy operating points. Among activation steering approaches, methods with learned interventions tend to achieve more favorable trade-offs than optimization-free methods. We further find that target efficacy is often preserved under out-of-distribution prompts. In contrast, side effects tend to become more pronounced, particularly through declines in instruction relevance and fluency. Finally, sample efficiency varies substantially across methods, reflecting different levels of dependence on training-data size.

## 2 RELATED WORK

Representation-Based Activation Steering. Representation-based activation steering controls model behavior by intervening on internal activations at inference time. Early methods often modeled target concepts with linear directions, motivated in part by the linear representation hypothesis (Bolukbasi et al., 2016; Elhage et al., 2022; Park et al., 2024; Nanda et al., 2023). Existing methods span contrastive directions, subspaces, latent representations, geometric transformations, hypernetworks, and nonlinear interventions (Turner et al., 2023; Zou et al., 2023; Marks & Tegmark, 2024; Wu et al., 2024; Pham et al., 2026; Sun et al., 2025; Wu et al., 2025b; Jin et al., 2026; You et al., 2026; Zhao et al., 2026). We organize the methods evaluated in this work by their underlying mechanisms in Section 4.

Benchmarks for Activation Steering. Existing benchmarks have begun to standardize the evaluation of activation steering, but each covers only a subset of the relevant evaluation dimensions. AxBench (Wu et al., 2025a) evaluates 500 steering targets across activation steering, prompting, and fine-tuning methods, but its off-target evaluation is limited to instruction following and fluency. SteeringSafety (Siu et al., 2026) studies behavioral entanglement across nine predefined safety perspectives within a curated safety concept set. FaithSteer-BENCH (Ding et al., 2026) focuses on deployment-time reliability, evaluating clean controllability, capability preservation, and robustness under changing conditions at a single calibrated operating point. SteerEval (Xu et al., 2026) broadens evaluation across behavioral domains and concept granularities, while CLaS-Bench (Gurgurov et al., 2026) extends steering evaluation to 32 languages and cross-lingual transfer. Complementary studies have identified specific steering side effects, including degradation in generation coherence (Bhalla et al., 2024), capability–behavior trade-offs (Sprejer et al., 2026), and sensitivity to individual samples and prompt conditions (Tan et al., 2024). Overall, existing evaluations cover safety entanglement, deployment reliability, cross-lingual control, and other aspects, but remain fragmented across evaluation dimensions and steering methods.

## 3 STEERSCOPE: A MULTI-DIMENSIONAL EVALUATION SUITE

We propose a two-axis evaluation suite. Steering Outcomes measures target efficacy and side effects over each method’s candidate factor set $A _ { m }$ , while Method Properties measures generalization and data dependence at a fixed method-level steering factor. For comparability, all methods use the same evaluation examples for each concept and task. The $\circledcirc$ symbol marks the seven evaluation dimensions throughout this section. Definitions, datasets, and evaluation settings for all 15 metrics are detailed in Appendices A and C.4.

## 3.1 STEERING OUTCOMES

Efficacy. $\circledcirc$ Concept Expression. We measure Concept Expression on open-ended AlpacaEval instructions (Li et al., 2023b). GPT-4o-mini (OpenAI, 2024) scores each response on a three-level scale (0/1/2) indicating no, partial, or clear expression of the target concept; we denote this score by $C ( y , c )$

Side Effects. We measure the unintended changes an intervention induces along three dimensions—language quality, general task capabilities, and safety and reliability. For each sideeffect metric and factor $\alpha \in { \mathcal { A } } _ { m } ,$ , we report the absolute performance of the steered model and its change relative to the unsteered base model on the same prompts.

Let $\overline { { \widetilde { M } } } _ { m , c } ^ { ( k ) } ( \alpha )$ denote the average score of metric k for method $m ,$ , concept $c ,$ and factor $\alpha ,$ where all metrics are transformed such that larger values indicate better performance. We define the side-effect delta as $\Delta M _ { m , c } ^ { ( k ) } ( \alpha ) = \overline { { \widetilde { M } } } _ { 0 , c } ^ { ( k ) } - \overline { { \widetilde { M } } } _ { m , c } ^ { ( k ) } ( \alpha )$

A positive $\Delta M$ indicates degradation caused by steering, while a negative value indicates improvement. Risk-oriented metrics are reversed before computing the delta (e.g., 1 − ASR and 1 − FRR), and signed bias scores are converted using their negative absolute values.

Language Quality. We evaluate language quality using an LLM judge that scores model responses based on fluency-related criteria. Let $\dot { F ( y ) } \in \{ 0 , 1 , 2 \}$ denote the resulting Fluency score. Scores of 0, 1, and 2 indicate that response y is not fluent, is somewhat fluent but contains noticeable errors or awkward phrasing, and is fluent with at most minor issues, respectively.

Task Capabilities. We evaluate general task capabilities across four dimensions: language understanding using SuperGLUE (Wang et al., 2019), knowledge using MMLU (Hendrycks et al., 2021a), mathematical reasoning using MATH (Hendrycks et al., 2021b), and instruction following using two separate metrics. The first, Instruction Relevance, uses an LLM judge to assess whether response $y$ addresses its original instruction x. Let $I ( y , x ) \in \{ 0 , 1 , 2 \}$ denote this score, where 0, 1, and 2 indicate that the response is unrelated, only minimally or indirectly related, and clearly and directly related to the instruction, respectively. The second uses IFEval (Zhou et al., 2023) to measure prompt-level strict instruction-following accuracy on its own evaluation prompts. IFEval is reported as an independent side-effect metric and is not included in Overall.

Safety & Reliability. We evaluate safety and reliability across three categories: jailbreak and refusal behavior using JailbreakBench (Chao et al., 2024), social bias using the ambiguous and disambiguated bias scores from BBQ (Parrish et al., 2022), and truthfulness-related behavior using TruthfulQA (Lin et al., 2022).

Due to different computational costs, inexpensive benchmarks (MMLU, BBQ, and TruthfulQA) are evaluated on all 500 concepts, while expensive benchmarks (SuperGLUE, MATH, IFEval, and JailbreakBench) use a fixed subset of 50 concepts shared across all methods and factors.

AxBench-style Overall. Following AxBench (Wu et al., 2025a), we compute the response-level Overall score as the harmonic mean of Concept Expression, Instruction Relevance, and Fluency:

$$
O ( y , x , c ) = \frac { 3 } { C ( y , c ) ^ { - 1 } + I ( y , x ) ^ { - 1 } + F ( y ) ^ { - 1 } } ,
$$

with $O = 0$ if any constituent score is zero. We average O over prompts and concepts. Overall is used for factor selection and Method Properties, but is not included as an additional metric in the 15-metric suite.

## 3.2 METHOD PROPERTIES

We evaluate two properties: generalization, which measures retention of the Overall steering effect under prompt distribution shifts, and data dependence, which measures how the amount and composition of training data affect that effect.

Before evaluating these properties, we select a single method-level steering factor for each method.

Method-level Steering Factor Selection For each method, we select a factor $\widehat { \alpha } _ { m } \in \mathcal { A } _ { m }$ that maximizes the average response-level Overall score over its full set of evaluated concepts and their in-distribution prompts:

$$
{ \widehat { \alpha } } _ { m } = \arg \operatorname* { m a x } _ { \alpha \in { \mathcal { A } } _ { m } } { \mathrm { A v g } } ( O ( \alpha ) ) .
$$

We acknowledge that selecting the factor on the full ID evaluation set may introduce some optimism in operating-point calibration. However, only one method-level steering factor is selected per method and then held fixed across all Generalization and Data Dependence experiments, without using OOD-language or reduced-data outcomes.

Generalization. We evaluate whether the Overall steering effect transfers under cross-lingual prompt-distribution shifts.

Cross-Lingual Retention. Because absolute scores can reflect differences in the unsteered model across languages, we measure retention using base-relative improvements. We use English as the in-distribution (ID) setting and Chinese, Korean, Italian, and Spanish as out-of-distribution (OOD) settings from X-AlpacaEval (Zhang et al., 2024). For language $\ell ,$ let $\Delta O _ { m , \ell }$ denote the mean Overall improvement over the unsteered model. We define

$$
\mathrm { R e t } _ { m } ^ { \mathrm { O O D } } = \frac { \Delta O _ { m } ^ { \mathrm { O O D } } } { \Delta O _ { m } ^ { \mathrm { I D } } } , \qquad \Delta O _ { m } ^ { \mathrm { I D } } = \Delta O _ { m , \mathrm { e n } } , \qquad \Delta O _ { m } ^ { \mathrm { O O D } } = \frac { 1 } { 4 } \sum _ { \ell \in \mathcal { L } _ { \mathrm { O O D } } } \Delta O _ { m , \ell } .
$$

Higher values indicate better retention of the steering effect under prompt distribution shifts.

Data Dependence. To measure how training-data variations—in both the size of the training set and which examples it contains—affect the Overall steering effect, we evaluate data dependence through two metrics, Sample Efficiency and Sample Sensitivity, which characterize dependence on the amount of data and on which examples are sampled, respectively.

Sample Efficiency. Sample Efficiency measures how much of a method’s full-data Overall improvement can be recovered from fewer training examples. Existing studies often vary training-set size as method-specific ablations and report absolute task scores, making data-efficiency comparisons across methods difficult (Ding et al., 2026; Jiang et al., 2026). We therefore normalize each reduced-data improvement by the corresponding full-data improvement, yielding a comparable measure of how efficiently each method uses training data. The exact estimator and data settings are given in Appendix A.3.1.

Sample Sensitivity. Beyond data quantity, we measure a method’s dependence on trainingdata composition under a fixed data budget. We quantify the variation in factor-0-relative Overall improvement across same-sized training subsets; lower values indicate weaker dependence on training-data composition. The exact estimator and data settings are given in Appendix A.3.2.

## 4 STEERING METHODS

We evaluate 23 representative methods and baselines, plus a Random control. The methods span diverse steering mechanisms while avoiding redundant variants and enabling both method- and mechanism-level analysis. A broader catalogue is provided in Appendix B.

## 4.1 OPTIMIZATION-FREE STEERING METHODS

Concept Directions. These methods construct steering directions from activation patterns associated with a target concept. DiffMean adds the difference between positive and negative mean representations to model activations at inference time (Marks & Tegmark, 2024); PCA extracts the first principal component from positive-example activations (Wu et al., 2025a), while LAT applies PCA to contrastive activation differences (Zou et al., 2023). As a control, Random applies DiffMean after randomly partitioning the training examples, removing concept-specific label alignment.

Structured Interventions. These methods introduce geometric, latent-space, or unit-level interventions. Spherical Steering performs norm-preserving geodesic rotation toward a target direction, aiming to better respect the geometry of the model’s representation space (You et al., 2026). HiDRA performs steering in a higher-dimensional feature space, where discriminative structure can be captured (Pham et al., 2026). AUSteer steers in discriminative LLM components and assigns adaptive strengths (Feng et al., 2026).

SAE-Based Methods. SAE-based methods use decoder directions from a sparse-autoencoder dictionary (Huben et al., 2024). SAE uses a concept-associated feature, while SAE-A selects the feature with highest target-discriminative AUROC (Wu et al., 2025a).

## 4.2 DIRECTLY OPTIMIZED STEERING METHODS

Learned Steering Directions. Several methods learn interventions using supervised training. Linear Probe trains a linear discriminator between positive and negative representations and uses the learned weight vector as the steering direction (Alain & Bengio, 2017; Wu et al., 2025a). SSV optimizes an additive vector using a language-modeling loss on training responses (Wu et al., 2025a), and ReFT-r1 similarly learns a rank-one representation intervention, jointly optimizing concept detection and generation-time steering objectives (Wu et al., 2025a).

Learned Subspace and Adaptive Interventions. LoReFT learns a low-rank intervention subspace over hidden representations (Wu et al., 2024). PSR combines a learned steering direction with an activation-dependent gate, allowing intervention strength to vary across tokens (Heyman & Vandeputte, 2026). We evaluate its all-layer variant (A-PSR) and single-layer variant (S-PSR). RePS learns an intervention through preference optimization (Wu et al., 2025b). ODESteer constructs a vector field from the gradient of a learned nonlinear barrier function and applies multi-step integration (Zhao et al., 2026); StepODESteer uses the same construction with a single update step.

## 4.3 HYPERNETWORK-BASED STEERING METHODS

These methods use shared auxiliary networks to generate steering interventions conditioned on natural-language targets. HyperSteer trains a transformer-based hypernetwork that maps a naturallanguage steering target and the model’s internal activations to a dynamically generated steering vector (Sun et al., 2025). FLAS instead trains a transformer-style FlowBlock to parameterize a concept-conditioned velocity field over activation space (Jin et al., 2026).

## 4.4 NON-REPRESENTATION-BASED STEERING METHODS

Four baseline methods that do not operate directly on model representations are included: two prompting baselines and two parameter-update baselines. Prompt Steering uses an LLM to convert each target concept into a short steering context, whereas Simple Prompt Steering uses a fixed concept-conditioned instruction; both prepend the resulting context to the original instruction. The prompt templates are provided in Appendix C.5. We further evaluate Supervised Fine-Tuning (SFT) (Ouyang et al., 2022) and Low-Rank Adaptation (LoRA) (Hu et al., 2022) as the two parameter-update baselines.

## 5 EXPERIMENTS

Training. Following AxBench (Wu et al., 2025a), we use CONCEPT500, whose concepts are sampled from the GemmaScope SAE concept list on Neuronpedia (Lin, 2023; Lieberum et al., 2024). For each concept, we use DeepSeek-V3.2 (DeepSeek-AI, 2025) to generate 72 contrastive pairs, where the two responses share the same instruction but differ in whether they express the concept. All methods use the complete CONCEPT500 setting; the exception is SFT, discussed below.

Evaluation. Metric definitions are given in §3. All LM-judge metrics use instructions from AlpacaEval (Li et al., 2023b). We evaluate less costly metrics on all 500 concepts and costlier metrics on a 50-concept subset shared across methods and factors. Appendix Table 6 lists the concept and instance counts for each metric group. For compute reasons, SFT is trained and evaluated on 20 concepts throughout. Generalization is evaluated only on text-type concepts, since cross-lingual prompt shifts do not apply to code or math concepts.

![](images/0cbd1df391ea3d5d658a9353bc613f51f2151a3ca227ced65f52c785713738c8.jpg)  
Figure 2: Efficacy–side-effect trade-offs at layer 20 for Gemma-2-2B-it (left) and Gemma-2-9B-it (right). Curves end at peak Concept Expression; methods peaking below 0.2 are omitted for clarity.

Models. We evaluate Gemma-2-2B-it and Gemma-2-9B-it (Gemma Team et al., 2024). We use residual-stream layer 20 for all single-site interventions on both models, matching the intervention site used during both training and evaluation of HyperSteer (Sun et al., 2025), FLAS (Jin et al., 2026), and S-PSR (Heyman & Vandeputte, 2026); see Appendix C.1.

Factors. For each method with a strength parameter, we sweep its established factors (Appendix Table 5).

## 5.1 EFFICACY AND SIDE EFFECTS

Figure 2 shows efficacy–side-effect trade-offs, plotting the equal-weight mean of eleven rangenormalized side-effect degradations against Concept Expression. Complete factor sweeps appear in Appendix D.

Reducing side effects remains a key challenge for steering methods. Across both model scales, methods with an explicit steering factor generally incur larger composite side effects as target efficacy increases. At its Overall-selected method-level steering factor, A-PSR improves Concept Expression by 1.20 and 1.30 points on Gemma-2-2B-it and Gemma-2-9B-it, respectively, while incurring composite side effects of 0.11 and 0.12. Recent methods such as A-PSR outperform Prompt Steering on AxBench’s Overall and Concept Expression metrics. Figure 3 further shows that individual methods can locally surpass Prompt Steering on particular capability metrics. Yet when a broader range of side effects is considered, no activation steering method dominates the Prompt Steering baseline on the efficacy–side-effect trade-off. We examine the robustness of this aggregate comparison to alternative metric weightings in Appendix D.1.1. These results suggest that future work on steering should focus on reducing unintended effects.

Optimization-free methods generally exhibit worse efficacy–side-effect trade-offs. Compared with prompting and methods with learned interventions, optimization-free methods generally exhibit less favorable efficacy–side-effect trade-offs. One possible explanation is that prompting leverages existing instruction-following capabilities, while methods with learned interventions incorporate task-specific feedback through gradient-based training. In contrast, directions extracted from representation statistics or pretrained features do not directly constrain the resulting behavioral changes.

Low-strength steering can improve off-target metrics. Some low-strength interventions improve off-target performance relative to the unsteered model. Reversing the steering direction removes or reverses these gains, suggesting direction-dependent conceptual entanglement rather than a generic effect of weak perturbations (Appendix D.3).

![](images/2f38d408071bdd61037b5f0fe01a08efdcbd53ba03d8510495e02f5914e5f39b.jpg)  
Figure 3: Capability-wise efficacy–side-effect trade-offs on Gemma-2-9B-it at layer 20. Each point uses a method’s Overall-selected method-level steering factor; methods peaking below 0.2 in Concept Expression are omitted for clarity. Positive y-values indicate degradation relative to factor 0.

Beyond the aggregate score, method rankings vary across individual side-effect metrics, underscoring the need for multi-dimensional evaluation. Ranking agreement is generally stronger on Gemma 2-9B-it than on Gemma-2-2B-it (Appendix D.2).

## 5.2 DATA DEPENDENCE

![](images/4c2bdec66d9a199a776b413f468061f0b74bd95aa62401318132cef6262a7dbe.jpg)  
Figure 4: Sample efficiency across training-set sizes on Gemma-2-9B-it. The left panel reports Overall improvement, and the right panel reports the percentage of the full-data Overall improvement recovered at each data budget (N = 144). Only methods with full-data Overall improvement at least 0.1 are shown.

Steering methods differ substantially in sample efficiency. These differences are most pronounced under small data budgets: HyperSteer and DiffMean retain a relatively large fraction of their full-data effect with only a few examples, whereas A-PSR, FLAS, LoRA, and ReFT-r1 require more data to approach their full-data performance. As the data budget increases, most methods gradually approach their full-data effect, although the improvement is not always monotonic. This result shows that full-data performance alone does not reveal how strongly a method depends on the amount of training data. Appendix D.7 provides results for both model scales.

Sample Sensitivity. We report mean Overall gain and sample sensitivity for all methods on both model scales in Appendix D.8.

## 5.3 GENERALIZATION

Taken together across both model scales, activation steering methods tend to exhibit lower Cross-Lingual Retention than Prompt Steering, although retention varies substantially across methods and models.

Target efficacy transfers more readily than behavioral quality. For most methods, base-relative Concept Gain changes only modestly from ID to OOD prompts, and several methods achieve comparable or higher gains OOD. In contrast, steering-induced Instruction Degradation is larger under OOD prompts for 11 of the 13 methods shown in Table 1, while Fluency Degradation is larger for 9 of 13. These results suggest that cross-lingual prompt shifts often preserve the intended steering effect while amplifying certain steering-induced side effects. Complete results for all methods and both model scales are reported in Appendix D.9.
<table><tr><td rowspan="2">Method</td><td rowspan="2">α</td><td rowspan="2">|C|</td><td colspan="2">Concept Gain ↑</td><td colspan="2">Instr. Deg. ↓</td><td colspan="2">Fluency Deg. ↓</td><td rowspan="2">Retention ↑</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>Prompt Steering</td><td>1.0</td><td></td><td>500.9570</td><td>1.0288</td><td>0.1110</td><td>0.1972</td><td>0.0140</td><td>0.0162</td><td>1.0026</td></tr><tr><td>Simple Prompt Steering</td><td>1.0</td><td></td><td>500.7650</td><td>0.8675</td><td></td><td>0.30400.45100.0190</td><td></td><td>-0.0040</td><td>0.9835</td></tr><tr><td>LoRA</td><td>1.0</td><td></td><td>501.0780</td><td></td><td></td><td>1.11980.30400.47450.0280</td><td></td><td>0.0608</td><td>0.9211</td></tr><tr><td>A-PSR</td><td>2.0</td><td></td><td>501.3120</td><td></td><td></td><td>1.33850.48400.58480.0680</td><td></td><td>0.0472</td><td>0.9157</td></tr><tr><td>S-PSR</td><td>4.0</td><td></td><td>500.7840</td><td></td><td></td><td>0.66420.59200.52350.3770</td><td></td><td>0.3442</td><td>0.8156</td></tr><tr><td>LoReFT</td><td>1.0</td><td></td><td>501.2070</td><td></td><td></td><td>1.09200.42000.5715 0.0380</td><td></td><td>0.1000</td><td>0.7909</td></tr><tr><td>SFT</td><td>1.0</td><td></td><td>131.1731</td><td></td><td></td><td>0.87690.68460.68560.0962</td><td></td><td>0.0846</td><td>0.6824</td></tr><tr><td>ReFT-r1</td><td>1.0</td><td></td><td>500.6830</td><td></td><td></td><td>0.67080.31900.56200.2060</td><td></td><td>0.3952</td><td>0.6580</td></tr><tr><td>HyperSteer</td><td>1.0</td><td></td><td>50 0.7930</td><td></td><td></td><td>0.79950.34300.72120.1560</td><td></td><td>0.4458</td><td>0.5745</td></tr><tr><td>RePS</td><td>14.0</td><td></td><td>500.9190</td><td></td><td></td><td>0.8925 0.47900.82000.1690</td><td></td><td>0.4288</td><td>0.5732</td></tr><tr><td>SSV</td><td>2.0</td><td></td><td>50 0.7060</td><td></td><td></td><td>0.67850.54100.76670.3480</td><td></td><td>0.5220</td><td>0.5731</td></tr><tr><td>DiffMean</td><td>0.8</td><td></td><td>500.3140</td><td></td><td></td><td>0.3093 0.2650 0.6062 0.2600</td><td></td><td>0.5775</td><td>0.4615</td></tr><tr><td>FLAS</td><td>2.5</td><td></td><td>501.0610</td><td></td><td></td><td>0.46520.67100.57200.1460</td><td></td><td>0.2805</td><td>0.4256</td></tr></table>

Table 1: Cross-lingual generalization on Gemma-2-2B-it at layer 20. Concept Gain and Instruction/Fluency Degradation are measured relative to the language-matched unsteered model; positive degradation indicates worse performance. Retention is the OOD-to-ID ratio of base-relative Overall gains. |C| denotes the number of evaluated concepts. Methods with ID Overall gain below 0.1 are omitted.

## 6 CONCLUSION

We introduced SteerScope, a comprehensive evaluation suite for LLM steering organized around two axes: Steering Outcomes and Method Properties. Our taxonomy organizes steering evaluation into target efficacy, side effects on language quality, task capabilities, and safety and reliability, and method properties under distribution and training-data changes. We further provide a systematic taxonomy of steering methods and conduct controlled comparisons across optimizationfree, directly optimized, and hypernetwork-based approaches, alongside prompting, LoRA, and SFT baselines. By comparing strength-dependent efficacy–side-effect curves across methods and model scales, SteerScope provides a broader view than comparisons based on selected operating points.

Our evaluation reveals that performance on existing benchmarks does not fully reflect steering quality. Activation steering methods that outperform the Prompt Steering baseline on existing benchmarks do not necessarily maintain this advantage when broader side effects are considered, and stronger steering efficacy is generally accompanied by larger unintended effects. Activation steering nevertheless remains valuable for its adjustable access to high-efficacy operating points. Among activation steering approaches, methods with learned interventions tend to achieve more favorable trade-offs than optimization-free methods. Improvements at weak intervention strengths suggest that side effects can reflect conceptual entanglement rather than uniform degradation. We further find that under out-of-distribution prompts, target efficacy is often preserved, whereas side effects tend to be more pronounced. Methods also exhibit substantially different sample-efficiency profiles.

## USE OF GENERATIVE AI

We used generative AI in the experimental workflow. DeepSeek-V3.2-Instruct generated conceptconditioned positive and negative responses for the contrastive training data, as well as conceptspecific instructions for Prompt Steering, A-PSR, and S-PSR. GPT-4o-mini was used to score concept expression, instruction relevance, and fluency; Llama-3.1-8B-Instruct was used to evaluate JailbreakBench responses. The corresponding prompts, scoring criteria, and settings are reported in the experimental details and appendix. We also used generative AI to polish the manuscript. The authors take responsibility for the LLM-generated materials and LLM-based judgments used in this work.

## REPRODUCIBILITY STATEMENT

We specify the evaluation metrics and comparison protocol in Section 3, and describe the training data, models, and experimental setup in Section 5. Appendices A and C provide the metric definitions, method configurations, steering-factor grids, inference settings, and prompts used for data generation and LLM-based evaluation. Source code and instructions for running the experiments are provided at https://github.com/yht0511/steerscope.

## ACKNOWLEDGMENTS

This Project is Sponsored by CCF-Tencent Rhino-Bird Open Research Fund (No. CCF-Tencent RAGR20260112).

## REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In ICLR 2017 Workshop, 2017. URL https://openreview.net/forum?id= HJ4-rAVtl.

Usha Bhalla, Suraj Srinivas, Asma Ghandeharioun, and Himabindu Lakkaraju. Towards unifying interpretability and control: Evaluation via intervention. arXiv preprint arXiv:2411.04430, 2024. URL https://arxiv.org/abs/2411.04430.

Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. In Advances in Neural Information Processing Systems, volume 29, pp. 4349–4357. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper\_files/ paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf.

Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, and Dmitrii Krasheninnikov. Understanding (un)reliability of steering vectors in language models. In ICLR 2025 Workshop on Foundation Models in the Wild, 2025. URL https://openreview.net/forum? id=qGCp2AYosf.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, and Eric Wong. JailbreakBench: An open robustness \` benchmark for jailbreaking large language models. In Advances in Neural Information Processing Systems, volume 37, pp. 55005–55029. Curran Associates, Inc., 2024.

URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 63092d79154adebd7305dfd498cbff70-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Seyed Arshan Dalili, Ajay Narayanan Sridhar, Vijaykrishnan Narayanan, and Mehrdad Mahdavi. Conditional optimal bridge for Riemannian activation steering. In Advances in Neural Information Processing Systems, 2026. URL https://arxiv.org/abs/2607.10517.

Quy-Anh Dang and Chris Ngo. Selective steering: Norm-preserving control through discriminative layer selection. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 10887–10910. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026. findings-acl.529. URL https://aclanthology.org/2026.findings-acl.529/.

DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025. URL https://arxiv.org/abs/2512.02556.

Zikang Ding, Qiying Hu, Yi Zhang, Hongji Li, Junchi Yao, Hongbo Liu, and Lijie Hu. FaithSteer-BENCH: A deployment-aligned stress-testing benchmark for inference-time steering. arXiv preprint arXiv:2603.18329, 2026. URL https://arxiv.org/abs/2603.18329.

Esin Durmus, Alex Tamkin, Jack Clark, Jerry Wei, Jonathan Marcus, Joshua Batson, Kunal Handa, Liane Lovitt, Meg Tong, Miles McCain, Oliver Rausch, Saffron Huang, Sam Bowman, Stuart Ritchie, Tom Henighan, and Deep Ganguli. Evaluating feature steering: A case study in mitigating social biases, 2024. URL https://anthropic.com/research/ evaluating-feature-steering.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread, 2022. URL https://transformer-circuits. pub/2022/toy\_model/index.html.

Zijian Feng, Tianjiao Li, Zixiao Zhu, Hanzhang Zhou, Junlang Qian, Li Zhang, Jia Jim Deryl Chua, Lee Onn Mak, Gee Wah Ng, and Kezhi Mao. Fine-grained activation steering: Steering less, achieving more. In International Conference on Learning Representations, volume 2026, pp. 39421–39443, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/file/423d0909791493b7c10916fd328c2913-Paper-Conference.pdf.

Gemma Team et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024. URL https://arxiv.org/abs/2408.00118.

Narges Ghasemi, Amir Ziashahabi, Salman Avestimehr, and Jonathan May. ORBIT: Trainingfree multi-attribute behavioral steering via orthogonal subspace rotation. arXiv preprint arXiv:2606.22357, 2026. URL https://arxiv.org/abs/2606.22357.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv. org/abs/2407.21783.

Daniil Gurgurov, Yusser Al Ghussin, Tanja Baeumel, Cheng-Ting Chou, Patrick Schramowski, Marius Mosbach, Josef van Genabith, and Simon Ostermann. CLaS-Bench: A cross-lingual alignment and steering benchmark. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 21591–21628. Association for Computational Linguistics, 2026. URL https://aclanthology.org/2026.findings-acl.1086/.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id= d7KBjmI3GmQ.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021b. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper\_files/paper/ 2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf.

Geert Heyman and Frederik Vandeputte. Steer like the LLM: Activation steering that mimics prompting. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 42968–42997. PMLR, 2026. URL https://proceedings.mlr.press/v306/heyman26a.html.

Brandon Hsu, Daniel Beaglehole, Adityanarayanan Radhakrishnan, and Mikhail Belkin. Contextual linear activation steering of language models. arXiv preprint arXiv:2604.24693, 2026. URL https://arxiv.org/abs/2604.24693.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, volume 2024, pp. 7827–7845, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 1fa1ab11f4bd5f94b2ec20e794dbfa3b-Paper-Conference.pdf.

Xinyan Jiang, Wenjing Yu, Di Wang, and Lijie Hu. Global evolutionary steering: Refining activation steering control via cross-layer consistency. arXiv preprint arXiv:2603.12298, 2026. URL https://arxiv.org/abs/2603.12298.

Zehao Jin, Ruixuan Deng, Junran Wang, Xinjie Shen, and Chao Zhang. Beyond steering vector: Flow-based activation steering for inference-time intervention. In Advances in Neural Information Processing Systems, 2026. URL https://arxiv.org/abs/2605.05892.

Ian Li, Kapilesh Guruprasad, Raunak Sengupta, Ninad Satish, Loris D’Antoni, and Rose Yu. Manifold-guided attention steering. In ICML 2026 Workshop on Foundations of Deep Generative Models: Understanding Memorization, Generalization, and Reasoning, 2026a. URL https://arxiv.org/abs/2605.21770.

Jiaqian Li, Yanshu Li, and Kuan-Hao Huang. Steering vector fields for context-aware inference-time control in large language models. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, 2026b. URL https://arxiv.org/abs/2602.01654.

Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. Inference-´ time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, volume 36, pp. 41451–41530. Curran Associates, Inc., 2023a. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. AlpacaEval: An automatic evaluator of instruction-following models. GitHub repository, 2023b. URL https://github.com/tatsu-lab/alpaca\_ eval.

Yawei Li, Benjamin Bergner, Yinghan Zhao, Vihang Prakash Patil, Bei Chen, and Cheng Wang. Steering large reasoning models towards concise reasoning via flow matching. Transactions on Machine Learning Research, 2026c. URL https://openreview.net/forum?id= qwcJMdGerK.

Yuxiao Li, Alina Fastowski, Efstratios Zaradoukas, Bardh Prenkaj, and Gjergji Kasneci. Analysing the safety pitfalls of steering vectors. In Findings of the Association for Computational Lin guistics: ACL 2026, pp. 11182–11204. Association for Computational Linguistics, 2026d. URL https://aclanthology.org/2026.findings-acl.544/.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, Janos Kram ´ ar, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma Scope: Open´ sparse autoencoders everywhere all at once on Gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 278–300. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024. blackboxnlp-1.19/.

Johnny Lin. Neuronpedia: Interactive reference and tooling for analyzing neural networks, 2023. URL https://www.neuronpedia.org. Software available from neuronpedia.org.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3214–3252. Association for Computational Linguistics, 2022. URL https://aclanthology.org/2022.acl-long.229/.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=aajyHYjjsk.

Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 16–30. Association for Computational Linguistics, 2023. URL https://aclanthology.org/2023.blackboxnlp-1.2/.

Tuc Nguyen and Thai Le. Beyond linear activation steering: Invertible latent transformations for controlling LLM behavior. In Advances in Neural Information Processing Systems, 2026. URL https://arxiv.org/abs/2606.08454.

OpenAI. GPT-4o mini: Advancing cost-efficient intelligence. OpenAI, 2024. URL https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html.

Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 39643–39666. PMLR, 2024. URL https://proceedings.mlr.press/v235/park24c.html.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman. BBQ: A hand-built bias benchmark for question answering. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2086– 2105. Association for Computational Linguistics, 2022. URL https://aclanthology. org/2022.findings-acl.165/.

Minh-Hieu Pham, Bach Do, Laziz Abdullaev, Tan Minh Nguyen, and Khoat Than. Highdimensional random projection for activation steering in language models. arXiv preprint arXiv:2606.15092, 2026. URL https://arxiv.org/abs/2606.15092.

Joris Postmus and Steven Abreu. Steering large language models using conceptors: Improving addition-based activation engineering. In MINT: Foundation Model Interventions, 2024. URL https://openreview.net/forum?id=gyAnAq16HC.

Shivam Raval, Hae Jin Song, Linlin Wu, Abir Harrasse, Jeff M. Phillips, Fazl Barez, and Amirali Abdullah. Curveball steering: The right direction to steer isn’t always linear. arXiv preprint arXiv:2603.09313, 2026. URL https://arxiv.org/abs/2603.09313.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering Llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 15504–15522. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024. acl-long.828/.

Yingdong Shi, Ruiming Zhang, Changming Li, Zhiyu Yang, Kaixing Zhang, Jingyi Yu, and Kan Ren. UniSteer: Text-guided flow matching in activation space for versatile LLM steering. arXiv preprint arXiv:2605.30076, 2026. URL https://arxiv.org/abs/2605.30076.

Vincent Siu, Nicholas Crispino, David Park, Nathan W. Henry, Zhun Wang, Yang Liu, Dawn Song, and Chenguang Wang. SteeringSafety: Benchmarking representation steering in LLMs across safety perspectives. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 113832–113884. PMLR, 2026. URL https://proceedings.mlr.press/v306/siu26a.html.

Eitan Sprejer, Oscar Agust´ın Stanchi, Mar´ıa Victoria Carro, Denise Alejandra Mester, and Ivan´ Arcuschin. Mind the performance gap: Capability-behavior trade-offs in feature steering. arXiv preprint arXiv:2602.04903, 2026. URL https://arxiv.org/abs/2602.04903.

Asa Cooper Stickland, Alexander Lyzhov, Jacob Pfau, Salsabila Mahdi, and Samuel R. Bowman. Steering without side effects: Improving post-deployment control of language models. In NeurIPS Safe Generative AI Workshop, 2024. URL https://openreview.net/forum? id=tfXIZ8P4ZU.

Jiuding Sun, Sidharth Baskaran, Zhengxuan Wu, Michael Sklar, Christopher Potts, and Atticus Geiger. HyperSteer: Activation steering at scale with hypernetworks. arXiv preprint arXiv:2506.03292, 2025. URL https://arxiv.org/abs/2506.03292.

Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adria Garriga-Alonso, and Robert Kirk. Analysing the generalisation and reli-\` ability of steering vectors. In Advances in Neural Information Processing Systems, volume 37, pp. 139179–139212. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ fb3ad59a84799bfb8d700e56d19c231b-Paper-Conference.pdf.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023. URL https://arxiv.org/abs/2308.10248.

Minh Hieu Vu and Tan M. Nguyen. Angular steering: Behavior control via rotation in activation space. In Advances in Neural Information Processing Systems, volume 38, pp. 134812–134849, 2025. doi: 10.52202/085713-4056. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ b0223cad0e73b793f31eb6cc41cefceb-Abstract-Conference.html.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems, volume 32, pp. 3261–3275. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf.

Weixuan Wang, Jingyuan Yang, and Wei Peng. Semantics-adaptive activation intervention for LLMs via dynamic steering vectors. In International Conference on Learning Representations, volume 2025, pp. 79334–79351, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ c4d26a95fd83f8e590f81c54ae670b5d-Paper-Conference.pdf.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. ReFT: Representation finetuning for language models. In Advances in Neural Information Processing Systems, volume 37, pp. 63908–63962. Curran Associates, Inc.,

2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/75008a0fba53bf13b0bb3b7bff986e0e-Paper-Conference.pdf.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. AxBench: Steering LLMs? Even simple baselines outperform sparse autoencoders. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 67035–67080. PMLR, 2025a. URL https://proceedings.mlr.press/v267/wu25a.html.

Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, and Chris Potts. Improved representation steering for language models. In Advances in Neural Information Processing Systems, volume 38, pp. 160589–160641. Curran Associates, Inc., 2025b. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ file/eb1ef82926376d252dde00d5dd909f4b-Paper-Conference.pdf.

Ziwen Xu, Kewei Xu, Haoming Xu, Haiwen Hong, Longtao Huang, Hui Xue, Ningyu Zhang, Yongliang Shen, Guozhou Zheng, Huajun Chen, and Shumin Deng. How controllable are large language models? A unified evaluation across behavioral granularities. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 31269–31299. Association for Computational Linguistics, 2026. URL https://aclanthology.org/2026.acl-long.1443/.

Zejia You, Chunyuan Deng, and Hanjie Chen. Spherical steering: Geometry-aware activation rotation for language models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 149801–149823. PMLR, 2026. URL https://proceedings.mlr.press/v306/you26a.html.

Zhihan Zhang, Dong-Ho Lee, Yuwei Fang, Wenhao Yu, Mengzhao Jia, Meng Jiang, and Francesco Barbieri. PLUG: Leveraging pivot language in cross-lingual instruction tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7025–7046. Association for Computational Linguistics, 2024. URL https:// aclanthology.org/2024.acl-long.379/.

Hongjue Zhao, Haosen Sun, Jiangtao Kong, Xiaochang Li, Qineng Wang, Liwei Jiang, Qi Zhu, Tarek F. Abdelzaher, Yejin Choi, Manling Li, and Huajie Shao. ODESteer: A unified ODE-based steering framework for LLM alignment. In International Conference on Learning Representations, volume 2026, pp. 69961–69988, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 72136a6372df75ddc3b1cd7f27c63712-Paper-Conference.pdf.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023. URL https://arxiv.org/abs/2311.07911.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405, 2023. URL https://arxiv. org/abs/2310.01405.

## APPENDIX CONTENTS

A Metric Details 17   
A.1 Side Effects . . . . 17   
A.2 Generalization. . . .18   
A.3 Data Dependence . . . 19   
B Catalogue of Steering Methods 21   
C Experimental Details 22   
C.1 Models and Shared Setup . . . .22   
C.2 Method Configurations . . 22   
C.3 Steering Factor Grids . . . 23   
C.4 Evaluation Configurations . . . 24   
C.5 Prompt Templates . . . 25   
C.6 Prior Human Validation of AxBench-Style Judges . . 27   
D Additional Results 28   
D.1 Composite Side Effects. . . . .28   
D.2 Ranking Agreement Across Side-Effect Metrics . . . . 32   
D.3 Signed-Factor Analysis. . . . .33   
D.4 Language Quality . . . .34   
D.5 Task Capabilities . . 38   
D.6 Safety & Reliability . . . .46   
D.7 Sample Efficiency . . 56   
D.8 Sample Sensitivity . . . . . 57   
D.9 Cross-Lingual Generalization . . . 57

## A METRIC DETAILS

SteerScope comprises 15 metrics: Concept Expression for target efficacy, the eleven side-effect metrics summarized in Table 2, one generalization metric—Cross-Lingual Retention—and two datadependence metrics—Sample Efficiency and Sample Sensitivity.

## A.1 SIDE EFFECTS

The side-effect evaluation comprises eight evaluators and eleven top-level metrics. All methods use the same concept-seeded evaluation data: for a given concept, every method and factor is evaluated on exactly the same examples, while different concepts use different deterministic subsets. All responses are sampled with temperature 1.0. Table 2 summarizes the data and per-example scoring used by each metric.

<table><tr><td rowspan="2">Metric</td><td colspan="4">Concepts Samples / concept Dataset and per-example evaluation</td><td>Direction  $/ s _ { j }$ </td></tr><tr><td>Instruction Rele-</td><td>500</td><td>10</td><td>AlpacaEval instructions; GPT-4o-mini judge score in {0, 1, 2} for whether the response addresses the original instruction</td><td>↑/2</td></tr><tr><td>vance Fluency</td><td></td><td>500</td><td>10</td><td>AlpacaEval instructions; GPT-4o-mini judge score in {0, 1, 2} for linguistic fluency</td><td>↑12</td></tr><tr><td>MMLU</td><td></td><td>500</td><td>20</td><td>MMLU test split; multiple-choice accuracy</td><td>↑/1</td></tr><tr><td>TruthfulQA</td><td></td><td>500</td><td>20</td><td>Binary-choice variant; binary accuracy</td><td>↑/1</td></tr><tr><td>BBQ absolute bias</td><td>ambiguous</td><td>500</td><td>20 total</td><td>Absolute value of the official ambiguous-context bias score, in- cluding error-rate scaling</td><td>↓/1</td></tr><tr><td>BBQ biguated bias</td><td>disam- absolute</td><td>500</td><td>20 total</td><td>Absolute value of the official disambiguated-context bias score, excluding UNKNOWN predictions</td><td>↓/1</td></tr><tr><td>SuperGLUE</td><td></td><td>50</td><td>20 / task</td><td>Eight tasks evaluated with their official task metrics; the task scores are macro-averaged</td><td>↑/1</td></tr><tr><td>MATH</td><td></td><td>50</td><td>20</td><td>Level 1 problems; up to 1024 generated tokens and exact- answer accuracy</td><td>↑/1</td></tr><tr><td>IFEval</td><td></td><td>50</td><td>20</td><td>Up to 1024 generated tokens evaluated with the official instruc- tion checker; prompt-level strict accuracy</td><td>↑/1</td></tr><tr><td>JailbreakBench ASR</td><td></td><td>50</td><td>20 harmful</td><td>Up to 150 generated tokens; Llama-3.1-8B-Instruct judge; pro-</td><td>↓/1</td></tr><tr><td>JailbreakBench FRR</td><td></td><td>50</td><td>20 benign</td><td>portion classified as unsafe Up to 150 generated tokens; the same judge; proportion of be- nign requests classified as refusals</td><td>↓/1</td></tr></table>

Table 2: Side-effect evaluation details. The two BBQ metrics are computed from the ambiguous and disambiguated subsets of the same 20-example panel per concept, not from 20 examples per condition. SuperGLUE uses 20 examples from each of its eight tasks, for 160 examples per concept. JailbreakBench uses 20 harmful and 20 benign prompts per concept. SFT is trained on 20 concepts and is evaluated on those same 20 concepts for all side-effect metrics.

For metric j, we first average its per-example scores within each method–concept–factor cell to obtain $x _ { m , c , f , j } .$ . We then pair every factor with factor 0 for the same concept and convert the difference into normalized degradation:

$$
d _ { m , c , f , j } = \left\{ \begin{array} { l l } { { \left( x _ { m , c , 0 , j } - x _ { m , c , f , j } \right) } / { s _ { j } } , } & { \mathrm { h i g h e r ~ i s ~ b e t t e r , } } \\ { { \left( x _ { m , c , f , j } - x _ { m , c , 0 , j } \right) } / { s _ { j } } , } & { \mathrm { l o w e r ~ i s ~ b e t t e r . } } \end{array} \right.\tag{1}
$$

Instruction Relevance and Fluency have natural range $[ 0 , 2 ]$ , so $s _ { j } = 2 ;$ all other top-level metrics have range $[ 0 , 1 ] , \thinspace \mathrm { s o } \ s _ { j } = 1$ . Positive values denote degradation, zero denotes no change from factor 0, and negative values denote an improvement.

Each metric is first averaged equally over its concept set, after which the eleven top-level metrics receive equal weight:

$$
D _ { m , f , j } = \frac { 1 } { | { \mathcal C } _ { j } | } \sum _ { c \in { \mathcal C } _ { j } } d _ { m , c , f , j } , \qquad \mathrm { C o m p o s i t e S i d e E f f e c t } _ { m , f } = \frac { 1 } { 1 1 } \sum _ { j = 1 } ^ { 1 1 } D _ { m , f , j } .\tag{2}
$$

Thus, a composite value of 0.10 means that the eleven metrics degrade on average by ten percentage points of their respective natural ranges. The factor-sweep curves in Appendix D report all methods and factors; the main text retains methods whose maximum mean Concept Expression improvement is at least 0.2.

## A.2 GENERALIZATION

We evaluate generalization through cross-lingual prompt distribution shifts. Let $\begin{array} { r l } { \mathcal { L } _ { \mathrm { O O D } } } & { { } = } \end{array}$ {zh, ko, it, es} denote the set of out-of-distribution languages. English is treated as the indistribution (ID) language. For each text concept c, we use the aligned prompts provided by X-AlpacaEval (Zhang et al., 2024). Let $\mathcal { D } _ { c } ^ { \ell }$ denote the set of prompts corresponding to concept c in language ℓ, and let $\mathcal { C } _ { m }$ denote the concepts evaluated for method m.

For each method m, we use the fixed method-level steering factor $\widehat { \alpha } _ { m }$ selected in Section 3.2. Given a prompt $x \in \mathcal { D } _ { c } ^ { \ell } ,$ , let $y _ { m , c , \widehat { \alpha } _ { m } } ( x )$ denote the response generated by the steered model and let y<sub>0</sub>(x) denote the response generated by the unsteered base model under the same prompt.

Since the unsteered model’s behavior may differ across languages, directly comparing absolute scores would mix steering transferability with language-specific base-model behavior. We therefore measure all changes relative to the unsteered model within each language. For component score $Q \in \{ C , I , F \}$ , representing Concept Expression, Instruction Relevance, and Fluency, respectively, let $s _ { C } = 1$ and $s _ { I } = s _ { F } = - 1$ . We define

$$
E _ { m , \ell } ^ { Q } = \frac { s _ { Q } } { | { \mathcal C } _ { m } | } \sum _ { c \in { \mathcal C } _ { m } } \frac { 1 } { | { \mathcal D } _ { c } ^ { \ell } | } \sum _ { x \in { \mathcal D } _ { c } ^ { \ell } } \left[ Q \left( y _ { m , c , \widehat { \alpha } _ { m } } ( x ) , x , c \right) - Q \left( y _ { 0 } ( x ) , x , c \right) \right] .\tag{3}
$$

Thus, $E _ { m , \ell } ^ { C }$ is Concept Gain, whereas $E _ { m , \ell } ^ { I }$ and $E _ { m , \ell } ^ { F }$ are Instruction and Fluency Degradation; positive degradation indicates worse behavior after steering. ID values use English, while OOD values macro-average the corresponding effects over ${ \mathcal { L } } _ { \mathrm { O O D } }$

Specifically, for each language ℓ, we define the method-level Overall steering effect as

$$
\Delta O _ { m , \ell } = \frac { 1 } { \vert \mathcal { C } _ { m } \vert } \sum _ { c \in \mathcal { C } _ { m } } \frac { 1 } { \vert \mathcal { D } _ { c } ^ { \ell } \vert } \sum _ { x \in \mathcal { D } _ { c } ^ { \ell } } \left[ O \left( y _ { m , c , \widehat { \alpha } _ { m } } ( x ) , x , c \right) - O \left( y _ { 0 } ( x ) , x , c \right) \right] ,\tag{4}
$$

where $O ( \cdot , x , c )$ denotes Overall, the harmonic mean of Concept Expression, Instruction Relevance, and Fluency defined in Section 3.1. Overall is computed for each response before differencing and aggregation; it is not obtained by combining the three aggregated component effects. We compute retention only from Overall.

The ID Overall steering effect is defined using English:

$$
\Delta O _ { m } ^ { \mathrm { I D } } = \Delta O _ { m , \mathrm { e n } } ,\tag{5}
$$

while the OOD Overall steering effect is computed as the macro-average over the 4 OOD languages:

$$
\Delta O _ { m } ^ { \mathrm { { O O D } } } = \frac { 1 } { | \mathcal { L } _ { \mathrm { { O O D } } } | } \sum _ { \ell \in \mathcal { L } _ { \mathrm { O O D } } } \Delta O _ { m , \ell } .\tag{6}
$$

We then define the generalization retention score as

$$
\mathrm { R e t } _ { m } ^ { \mathrm { O O D } } = \frac { \Delta O _ { m } ^ { \mathrm { O O D } } } { \Delta O _ { m } ^ { \mathrm { I D } } } .\tag{7}
$$

This metric measures the proportion of the Overall steering effect obtained in the ID condition that is preserved under OOD prompt distributions. A value of 1 indicates full retention, values below 1 indicate partial loss, and values above 1 indicate a stronger Overall effect under the OOD prompts.

Because the retention score is a ratio, it can become unstable when the denominator is close to zero. Therefore, we only report $\mathrm { R e t } _ { m } ^ { \mathrm { { O O D } } }$ for methods satisfying

$$
\Delta O _ { m } ^ { \mathrm { I D } } \geq 0 . 1 .\tag{8}
$$

This filtering criterion ensures that retention is computed only when the method demonstrates a sufficiently strong Overall improvement in the ID condition.

## A.3 DATA DEPENDENCE

## A.3.1 SAMPLE EFFICIENCY

Sample Efficiency measures the fraction of the full-data Overall improvement that can be recovered using a reduced training set.

For each method, we evaluate multiple training-set sizes. Let $n \in \{ 6 , 1 2 , 3 6 , 7 2 \}$ denote the number of labeled rows per concept, and let N = 144 denote the full training-set size. To ensure a fair comparison, all reduced training sets are constructed from the same subset seed $s _ { 0 }$ as the full-data setting. All training-set sizes are evaluated on the same 10 ID prompts per concept.

Using the fixed method-level steering factor $\widehat { \alpha } _ { m }$ selected from the full-data factor sweep, we define the Overall improvement under training size n as

$$
\Delta O _ { m , n , s _ { 0 } } = \frac { 1 } { | { \mathcal C } _ { \mathrm { d a t a } } | } \sum _ { c \in { \mathcal C } _ { \mathrm { d a t a } } } \frac { 1 } { | { \mathcal D } _ { c } ^ { \mathrm { I D } } | } \sum _ { x \in { \mathcal D } _ { c } ^ { \mathrm { I D } } } \left[ O \left( y _ { m , c , \widehat { \alpha } _ { m } } ^ { ( n , s _ { 0 } ) } ( x ) , x , c \right) - O \left( y _ { 0 } ( x ) , x , c \right) \right] ,\tag{9}
$$

where $y _ { m , c , \widehat { \alpha } _ { m } } ^ { ( n , s _ { 0 } ) } ( x )$ denotes the response generated by method m trained with n labeled rows per concept using subset seed $s _ { 0 }$

We define Sample Efficiency as the percentage of the full-data Overall improvement recovered by the reduced training set:

$$
\mathrm { E f f } _ { m } ( n ) = 1 0 0 \frac { \Delta O _ { m , n , s _ { 0 } } } { \Delta O _ { m , N , s _ { 0 } } } .\tag{10}
$$

When $\mathrm { E f f } _ { m } ( n ) = 1 0 0 \%$ , the reduced training set recovers the same Overall improvement as the full training set. Since all training sizes share the same fixed method-level steering factor selected under the full-data setting, reduced training subsets may occasionally achieve a larger Overall improvement than the full training set, resulting in efficiency values above 100%.

When the full-data Overall improvement is close to zero, the ratio becomes unstable. Therefore, we only report Sample Efficiency when

$$
\Delta O _ { m , N , s _ { 0 } } \geq 0 . 1 .\tag{11}
$$

## A.3.2 SAMPLE SENSITIVITY

Sample Sensitivity measures variation in the improvement in Overall performance caused by different training-data compositions when the training-set size is fixed. Overall is the harmonic-mean aggregate defined in Section 3.1.

We fix the training-set size to $n _ { 0 } = 2 4$ labeled rows per concept and use five training subsets with subset-seed set

$$
S = \{ 4 2 , 4 3 , 4 4 , 4 5 , 4 6 \} .
$$

The training initialization seed is fixed at $4 2 ,$ , so only the sampled training subset changes across s ∈ S. For every subset, method m uses the same method-level steering factor $\widehat { \alpha } _ { m }$ , selected by Overall on the full-data main run rather than reselected for each seed. Each Overall score is computed on the same 20 evaluation prompts for concept c.

For method $m ,$ concept $c ,$ and subset seed $s ,$ we define the factor-0-relative Overall improvement as

$$
\Delta O _ { m , c , s } = \mathrm { O v e r a l l } _ { m , c , s } ( \widehat { \alpha } _ { m } ) - \mathrm { O v e r a l l } _ { m , c , s } ( 0 ) .\tag{12}
$$

Using a seed-matched factor-0 baseline isolates variation in the improvement attributable to steering from variation in the corresponding baseline evaluation. For each concept, the mean improvement across subset seeds is

$$
\overline { { \Delta O } } _ { m , c } = \frac { 1 } { | S | } \sum _ { s \in S } \Delta O _ { m , c , s } .\tag{13}
$$

We then compute the concept-level sample standard deviation across the five subset seeds:

$$
S D _ { m , c } = \sqrt { \frac { 1 } { \left|  { S } \right| - 1 } \sum _ { s \in  { S } } \left( \Delta O _ { m , c , s } - \overline { { \Delta O } } _ { m , c } \right) ^ { 2 } } .\tag{14}
$$

The denominator $| { \cal S } | - 1$ matches the sample standard deviation used by pandas.std with its default ddof=1. Finally, Sample Sensitivity is the macro-average of the concept-level standard deviations:

$$
\mathrm { S e n s i t i v i t y } _ { m } = \frac { 1 } { | { \mathcal C } _ { \mathrm { d a t a } } | } \sum _ { c \in { \mathcal C } _ { \mathrm { d a t a } } } S D _ { m , c } .\tag{15}
$$

A lower value indicates that the method’s improvement over its factor-0 baseline is more consistent across training-data subsets, whereas a higher value indicates stronger dependence on which training examples are selected. We report this absolute sensitivity for every method.

To interpret sample sensitivity alongside the achieved steering effect, we also report the mean signed Overall gain across concepts:

$$
G _ { m } = \frac { 1 } { \vert \mathcal { C } _ { \mathrm { d a t a } } \vert } \sum _ { c \in \mathcal { C } _ { \mathrm { d a t a } } } \overline { { \Delta O } } _ { m , c } , \qquad S _ { m } = \mathrm { S e n s i t i v i t y } _ { m } .\tag{16}
$$

Both $G _ { m }$ and $S _ { m }$ are reported for every method without an improvement threshold. Reporting them together distinguishes a consistently strong steering effect from one that is consistently weak or detrimental.

## B CATALOGUE OF STEERING METHODS

<table><tr><td rowspan=1 colspan=9>Contrastive          Linear  Subspace          Latent  Adaptive Multi-Step  SAE-   SharedScalingMethod                  GeodesicDataChangeChange  Space  Strength Dynamics  Based   ModuleOptimization-Free Steering Methods</td></tr><tr><td rowspan=1 colspan=2>AUSteer (Feng et al., 2026)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>XX</td></tr><tr><td rowspan=1 colspan=2>PCA (Wu et al., 2025a)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>XX</td></tr><tr><td rowspan=1 colspan=2>DiffMean (Marks &amp; Tegmark, 2024)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=2>CAA (Rimsky et al., 2024)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>XX</td></tr><tr><td rowspan=1 colspan=2>SADI (Wang et al., 2025)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=2>LAT (Zou et al., 2023)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=2>ORBIT (Ghasemi et al., 2026)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>GER-Steer (Jiang et al., 2026)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>Conceptors (Postmus &amp; Abreu, 2024)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=2>Spherical Steering (You et al., 2026)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=2>Selective Steering (Dang &amp; Ngo, 2026)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=2>Angular Steering (Vu &amp; Nguyen, 2025)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>XX</td></tr><tr><td rowspan=1 colspan=2>COBRAS (Dalili et al., 2026)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>XX</td></tr><tr><td rowspan=1 colspan=2>HiDRA (Pham et al., 2026)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Curveball Steering (Raval et al., 2026)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=2>SAE (Wu et al., 2025a)</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=2>SAE-A (Wu et al., 2025a)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>MAGS (Li et al., 2026a)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>Linear Probe (Alain &amp; Bengio, 2017;</td><td rowspan=1 colspan=2></td><td rowspan=3 colspan=5></td></tr><tr><td rowspan=1 colspan=2>Wu et al., 2025a)</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>SSV (Wu et al., 2025a)</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>ReFT-r1 (Wu et al., 2025a)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>LoReFT (Wu et al., 2024)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=2>ITI (Li et al., 2023a)</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=2 colspan=9>CLAS (Hsu et al., 2026)INNSteer (Nguyen &amp; Le, 2026)</td></tr><tr><td rowspan=1 colspan=2>A-PSR / S-PSR (Heyman & Vandeputte,</td></tr><tr><td rowspan=1 colspan=9>SVF (Li et al., 2026b)</td></tr><tr><td rowspan=1 colspan=9>ODESteer (Zhao et al., 2026)FlowSteer (Li et al., 2026c)HyperSteer (Sun et al., 2025)</td></tr><tr><td rowspan=1 colspan=9>FLAS (Jin et al., 2026)</td></tr><tr><td rowspan=1 colspan=9>UniSteer (Shi et al., 2026)</td></tr><tr><td rowspan=1 colspan=9>Legend</td></tr><tr><td rowspan=1 colspan=9>Column                 Definition</td></tr><tr><td rowspan=3 colspan=9>Contrastive Data           Uses contrastive positive and negative examples of a target concept to derive a steering vector.Scaling                  Multiplies selected activation components, directions, or subspaces rather than adding a vector.Locates a dominant linear direction that best represents a target concept and steers by linear addition.Subspace Change           Locates a dominant multi-dimensional subspace that best represents a target concept and steers in that subspace.Steers by geodesic rotation or norm-preserving.Latent Space              Steers in a transformed representation space, such as SAE, random-feature, kernel, or invertible latent spaces.Adaptive Strength           Adapts the steering strength based on the input prompt or current activations at inference time or gated.Multi-Step Dynamics        Applies steering through multiple sequential updates.SAE-Based                Uses a sparse autoencoder to select or construct steering directions.Shared Module            Creates a reusable learned steering tool across multiple concepts.</td></tr><tr><td rowspan=1 colspan=1>Linear Change</td></tr><tr><td rowspan=1 colspan=1>Subspace Change</td></tr></table>

Table 3: Catalogue of steering methods and their mechanisms.

## C EXPERIMENTAL DETAILS

## C.1 MODELS AND SHARED SETUP

We evaluate all methods on instruction-tuned Gemma-2 models at 2 scales, google/gemma-2-2b-it and google/gemma-2-9b-it. For methods defined at a single intervention site, we intervene on the residual stream at layer 20 in both models, matching the intervention site used during both training and evaluation of HyperSteer (Sun et al., 2025), FLAS (Jin et al., 2026), and S-PSR (Heyman & Vandeputte, 2026). Layer 20 corresponds to approximately 77% and 48% of the 26- and 42-layer model stacks, respectively. The training data for each concept contain 72 contrastive response pairs generated by DeepSeek-V3.2-Instruct, giving 144 labeled examples. Unless otherwise specified, training uses bfloat16 precision and a base random seed of 42, from which method- and concept-specific seeds are derived. Gradient-based methods use AdamW with a linear learning-rate schedule and no warmup, except for FLAS as specified below. All methods use the complete CONCEPT500 setting, except SFT, which is trained and evaluated on 20 concepts.

## C.2 METHOD CONFIGURATIONS

Table 4 reports the configurations used to train or construct each method. Here, B denotes the perdevice microbatch size, A the number of gradient-accumulation steps, E the number of epochs, LR the learning rate, and WD the weight decay. For methods without gradient-based training, B denotes the batch size used to extract activations or construct features.

Table 4: Training and method-specific configurations. “Same as 2B” means that the complete configuration in the preceding column is used unchanged.

<table><tr><td>Method</td><td>Gemma-2-2B-it</td><td>Gemma-2-9B-it</td></tr><tr><td>Prompt Steering</td><td>No gradient training; steering prompts are generated by DeepSeek-V3.2-Instruct with temperature 0.</td><td>Same as 2B.</td></tr><tr><td>Simple Prompt Steering</td><td>No training; uses a fixed concept-conditioned instruction.</td><td>Same as 2B.</td></tr><tr><td>DiffMean</td><td>Activation extraction with B = 6 and one pass over binarized training data.</td><td>Same as 2B.</td></tr><tr><td>PCA</td><td>Activation extraction with B = 6 and one pass over binarized training data.</td><td>Same as 2B.</td></tr><tr><td>LAT</td><td>Activation extraction with B = 6 and one pass over binarized training data.</td><td>Same as 2B.</td></tr><tr><td>Random</td><td>B = 6 and one pass; constructs a difference direction from a random partition of the labels.</td><td>Same as 2B.</td></tr><tr><td>SAE</td><td>Uses pretrained GemmaScope SAE features without concept-specific gradient training.</td><td>Same as 2B.</td></tr><tr><td>SAE-A</td><td>B = 6 and one pass over binarized data; selects the SAE feature with the highest target-discriminative AUROC.</td><td>Same as 2B.</td></tr><tr><td>Spherical Steering</td><td>B = 16, κ = 20, and β = 0.1; uses binarized positive and negative examples.</td><td>Same as 2B.</td></tr><tr><td>HiDRA</td><td>B = 16, projection dimension 8192, negative slope 0.7, and seed 42; direction normalization is disabled; uses binarized</td><td>Same as 2B.</td></tr><tr><td>AUSteer</td><td>positive and negative examples. B = 16 and top-k = 10; uses binarized</td><td>Same as 2B.</td></tr><tr><td>Linear Probe</td><td>positive and negative examples. Rank 1, binarized labels, and no L1 penalty; B = 6, A = 8, E = 24, LR = 0.005, WD = 0.001.</td><td>Rank 1, binarized labels, and no L1 penalty; B = 12, A = 4, E = 24, LR = 0.001, WD = 10−4.</td></tr><tr><td>SSV</td><td>Rank 1 additive intervention at all token positions, with the BOS token excluded; trains on positive and negative examples;  $\begin{array} { r } { B = 6 , \bar { A } = 1 , E = 3 , \bar { \mathrm { L R } } = 0 . 0 1 , } \end{array}$  WD</td><td>Same as 2B.</td></tr><tr><td>ReFT-r1</td><td> $\mathit { \Theta } = 0 .$  Rank 1, top-k = 8, and latent L1 coefficient 0.005; additive intervention at all token positions with the BOS token excluded; trains on positive and negative examples;  $B = 6 , \hat { A } = 1 , E = 3 , \bar { \mathrm { L R } }$ </td><td>Same method settings with B = 6,  $A = 1 , E = 3 , \mathbf { L R } { \stackrel { - } { = } } 0 . 0 0 5 , \mathbf { W D } = 0 .$ </td></tr><tr><td>LoRA</td><td>Rank  $4 , \alpha = 3 2 ,$  applied to o-proj at layers  $\{ 5 , 1 0 , 1 5 , \hat { 2 0 } \}$  ; trains on positive examples only;  $B = \stackrel { \cdot } { 3 } , A = 1 2 ,$   $E = 2 4 , \mathrm { L R } = 9 \times 1 0 ^ { - 4 } , \mathrm { W D } = 0 .$ </td><td>Rank  $4 , \alpha = 3 2 ,$  applied to o-pro j at layers {12, 20, 31, 39}; trains on positive examples only;  $B = \dot { 1 } 8 , A = 2 ,$   $E = \mathsf { \bar { 2 4 } } , \mathrm { L R } = 0 . 0 0 5 , \mathrm { W D } = 0 .$ </td></tr><tr><td>LoReFT</td><td>Rank 4, positions f5+15, and Loreft intervention at layers {5, 10, 15, 20}; trains on positive examples only; B = 3,  $A = 1 2 , { \dot { E } } = 2 4 , \mathrm { L R } { \dot { = } } 9 \times 1 { \dot { 0 } } ^ { - 4 } ,$  WD</td><td>Rank 4, positions f5+15, and Loreft intervention at layers {12, 20, 31, 39}; trains on positive examples only;  $B = 9 ,$   $A = 4 , \dot { E } = 2 4 , \mathrm { L R } = \mathrm { \dot { 4 } } \times 1 0 ^ { - 4 } ,$  WD</td></tr><tr><td>SFT</td><td>Trains on positive examples for 20 concepts;  $\overset { \cdot } { B } = 3 , A = \overset { \cdot } { 2 } 4 , E = 8 , \mathrm { L R }$   $= 4 \stackrel { \cdot } { \times } 1 0 ^ { - 5 } , \mathrm { W D } = 0 .$ </td><td>Trains on positive examples for 20 concepts;  $\boldsymbol { B } = 1 , \boldsymbol { A } = 3 6 , \boldsymbol { E } = 8 , \boldsymbol { \mathrm { L R } }$   $= 4 \stackrel { \cdot } { \times } 1 0 ^ { - 5 } , \mathrm { W D } = 0 .$ </td></tr><tr><td>HyperSteer</td><td>Rank 1 with a pretrained backbone-matched hypernetwork containing four hidden layers; trained jointly across concepts; B = 12, A = 1,  $\overset { \cdot } { E } = \overset { \cdot } { 1 } 0 , \mathrm { L R } = 8 \times \overset { \cdot } { 1 } 0 ^ { - 5 } , \mathrm { W D } = 0 .$ </td><td>Same as 2B, using the backbone-matched 9B hypernetwork.</td></tr><tr><td>RePS</td><td>Reference-free scaled_simpowith β = 1, scaler 1, and label smoothing 0;  $B = 6 , A = 1 , E = 1 8 , \mathbf { L R } = 0 . 0 4 ,$  WD = 0, dropout = 0.</td><td>Same objective with  $B = 6 , A = 1 ,$   $E = 1 8 , \mathrm { L R } = 0 . 0 8 , \mathrm { W D } = 0 ,$  dropout = 0.1.</td></tr><tr><td>FLAS</td><td>One FlowBlock with three integration steps, t ∈ [0.5, 2], and divergence weight with B = 16 and A = 2.  $0 . { \dot { 1 } } ; B = { \dot { 3 } } 2 , A { \dot { = } } 1 , \mathrm { L R } = { \bar { 5 } } \times 1 0 ^ { - 5 } ,$   $\mathrm { W D } = 0 . 0 1$  , with at most 80,000 updates.</td><td>Same method and optimization settings</td></tr><tr><td>A-PSR</td><td>Applied at all decoder layers; prompts generated by DeepSeek-V3.2-Instruct with temperature 0; B = 1, A = 1,  $E = 1 5 , \mathrm { { \dot { L } R } = 0 . 0 0 1 , \mathrm { { W D } = 1 0 ^ { - 6 } } } .$ </td><td>Same as 2B.</td></tr><tr><td>S-PSR</td><td>Applied only at the target layer; prompts Same as 2B. generated by DeepSeek-V3.2-Instruct with temperature 0; B = 1, A = 1,  $E = 1 5 , \mathrm { { \dot { L } R } = 0 . 0 0 1 , \mathrm { { W D } = 1 0 ^ { - 6 } } } .$ </td><td></td></tr><tr><td>ODESteer</td><td>B = 16, polynomial degree 2, 8000 components, γ = 0.1, and coefficient 1; logistic-regression classifier with at most 1000 iterations; Euler solver with 10 steps; uses binarized positive and</td><td>Same as 2B.</td></tr><tr><td>StepODESteer</td><td>negative examples. Uses the same feature and classifier configuration as ODESteer, but applies a single update.</td><td>Same as 2B.</td></tr></table>

## C.3 STEERING FACTOR GRIDS

Table 5 lists the candidate inference factors used for each method and model scale. Methods that share an identical grid are grouped together.

Table 5: Candidate inference factors for Gemma-2-2B-it and Gemma-2-9B-it
<table><tr><td>Method(s)</td><td>Gemma-2-2B-it</td><td>Gemma-2-9B-it</td></tr><tr><td>Prompt Steering, Simple Prompt Steering, LoRA, LoReFT, SFT</td><td></td><td>1</td></tr><tr><td>DiffMean, PCA, LAT, Random, Linear Probe, SSV, ReFT-r1, SAE, SAE-A, HyperSteer, A-PSR, S-PSR</td><td>0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, Same as 2B</td><td></td></tr><tr><td>RePS</td><td>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50</td><td>Same as 2B</td></tr><tr><td>FLAS</td><td>1, 1.5, 2, 2.5, 3</td><td>Same as 2B</td></tr><tr><td>Spherical Steering</td><td>0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1</td><td>Same as 2B</td></tr><tr><td>HiDRA</td><td>0.12, 0.24, 0.36, 0.5, 0.6, 0.72, 0.84, 1, 1.2, 1.5, 1.8, 2, 2.4, 3</td><td>Same as 2B</td></tr><tr><td>AUSteer</td><td>1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 75, 100, 150,200</td><td>Same as 2B</td></tr><tr><td>ODESteer</td><td>5, 10, 15, 20, 40, 60, 80, 100, 120, 140, 180, 250, 350, 500</td><td>10, 15, 18, 20, 36, 54, 72, 90, 108, 144, 180, 225, 360, 450</td></tr><tr><td>StepODESteer</td><td>10, 15, 20, 44, 88, 132, 176, 220, 308, 440, 550, 660, 880, 1100</td><td>10, 15, 20, 29, 58, 87, 116, 145, 174, 232, 261, 290, 435, 725</td></tr></table>

## C.4 EVALUATION CONFIGURATIONS

Except for Sample Sensitivity, evaluated-model generations use sampling with temperature = 1.0 and do sample=true. Sample Sensitivity instead uses greedy decoding with temperature = 0.0 and do sample=false, preventing generation noise from contributing to variation across training subsets. The GPT-4o-mini judge uses temperature = 0 throughout. Table 6 summarizes the evaluation data and inference settings. Multiple-choice benchmarks are evaluated from the logits assigned to their candidate answers.

<table><tr><td>Evaluation</td><td>Concepts</td><td>Instances / concept</td><td>Inference and scoring</td></tr><tr><td>Concept Expression, Instruction Relevance,</td><td>500</td><td>10</td><td>Up to 128 generated tokens; GPT-4o-mini judge</td></tr><tr><td>Fluency MMLU, BBQ, TruthfulQA</td><td>500</td><td>20</td><td>Candidate-answer logits</td></tr><tr><td>SuperGLUE</td><td>50</td><td>20 per task†</td><td>Candidate-answer logits; official task metrics</td></tr><tr><td>MATH</td><td>50</td><td>20</td><td>Up to 1024 generated tokens; exact-answer accuracy</td></tr><tr><td>IFEval</td><td>50</td><td>20</td><td>Up to 1024 generated tokens; official instruction checker</td></tr><tr><td>JailbreakBench</td><td>50</td><td>20 harmful and 20 benign</td><td>Up to 150 generated tokens; Llama-3.1-8B-Instruct judge</td></tr><tr><td>Generalization</td><td>50</td><td>20</td><td>Up to 128 generated tokens; GPT-4o-mini</td></tr><tr><td>Sample Efficiency</td><td>50</td><td>10</td><td>judge Up to 128 generated tokens; GPT-4o-mini</td></tr><tr><td>Sample Sensitivity</td><td>50</td><td>20</td><td>judge Up to 128 generated tokens; GPT-4o-mini judge</td></tr></table>

Table 6: Evaluation data and inference settings. The 50-concept subset is fixed and shared across all methods and factors. <sup>†</sup>SuperGLUE draws 20 instances from each of its eight tasks. MMLU uses the test split, TruthfulQA uses the binary-choice variant, and MATH is restricted to Level 1 problems. JailbreakBench responses are judged by Llama-3.1-8B-Instruct (Grattafiori et al., 2024). SFT is evaluated on its 20 training concepts; Generalization uses the available text concepts (13 on Gemma-2-2B-it and 15 on Gemma-2-9B-it).

For Generalization, we use 50 text concepts and 20 aligned X-AlpacaEval instructions per concept in English, Chinese, Korean, Italian, and Spanish. For Sample Efficiency, we train with N ∈ {6, 12, 36, 72} labeled rows per concept and compare against the full N = 144 setting using a fixed subset seed. All training-set sizes are evaluated on the same 10 prompts per concept. For Sample Sensitivity, we train with 24 labeled rows per concept using subset seed∈ {42, 43, 44, 45, 46}. The training initialization remains fixed at train.seed=42, and the same 20 evaluation prompts are used across all subset seeds for each concept.

## C.5 PROMPT TEMPLATES

We report the custom templates used to construct the contrastive training data, produce promptbased steering instructions, and obtain the three LM-judge scores. Placeholders enclosed in braces are replaced at runtime. Standard benchmark inputs follow the corresponding AxBench evaluator implementations without modification and are therefore not repeated here. All evaluated-model inputs are finally rendered with the model’s native chat template.

## C.5.1 CONTRASTIVE TRAINING-DATA GENERATION

The following templates are the same as those used in AxBench (Wu et al., 2025a). The positive and negative responses for a contrastive pair are generated from the same source instruction. The positive-response template is:

Given the following instruction:   
{INSTRUCTION}   
Your task is to:   
1. Provide a response that incorporates elements related to ’{CONCEPT}’.   
2. Try to avoid copying words from the definition of ’{CONCEPT}’ if possible.   
3. Ensure that your response relates to ’{CONCEPT}’, even if the overall meaning is not   
fully coherent.   
Formatting Guidelines:   
- Return only the response to the instruction.   
- Write the final content (or appropriate format for the genre) in plain text.   
- Do not include any additional text, explanations, or formatting.   
<sub>\*\*</sub>Final Answer:<sub>\*\*</sub> Return only the final content, following the guidelines above.

The corresponding negative-response template is:

Given the following instruction:   
{INSTRUCTION}   
Your task is to:   
1. Provide a response that continues or addresses the instruction naturally.   
2. Avoid any mention of ’{CONCEPT}’ in the continuation, regardless of coherence.   
Formatting Guidelines:   
Return only the response to the instruction.   
Write the final content (or appropriate format for the genre) in plain text.   
Do not include any additional text, explanations, or formatting.   
<sub>\*\*</sub>Final Answer:<sub>\*\*</sub> Return only the final content, following the guidelines above.

## C.5.2 PROMPT GENERATION FOR PROMPT STEERING AND PSR VARIANTS

Prompt Steering, A-PSR, and S-PSR share the following template for generating a concept-specific steering instruction. Generation uses DeepSeek-V3.2-Instruct with temperature 0. This template is the same as that used in AxBench (Wu et al., 2025a).

Generate a prompt to guide a language model in producing responses.   
Objective:   
Direct the model to include content related to {CONCEPT} (the concept) in its responses.   
Ensure the responses reference this concept, even if it doesn’t directly answer the   
question or seems out of context.   
Optionally, provide in-context examples to reinforce this behavior.   
Return only the final prompt without any additional text.

Prompt Steering directly prepends the resulting instruction to the task input, whereas A-PSR and S-PSR use the same instruction in their learned interventions. Their layer-specific configurations are given in Table 4.

## C.5.3 SIMPLE PROMPT STEERING

Simple Prompt Steering uses the fixed instruction below, where %s is replaced by the concept description. This template is the same as that used in AxBench (Wu et al., 2025a).

You must answer the question with content related to %s even if it is not related to the   
question or it does not make sense.

For both prompt-based methods, the steering instruction and original task input are combined as follows before applying the model chat template:

{steering\_instruction}   
Question: {raw\_input}

## C.5.4 LM-JUDGE TEMPLATES

The Concept Expression judge uses the following prompt:

```ini
[System]
Please act as an impartial judge and evaluate whether the specified concept is presented in
the sentence fragment provided below. Focus solely on whether the concept is present,
without regard for grammar or logical coherence or the fluency of the language.
Begin your evaluation by providing a short explanation of whether the concept is clearly
present. Point out references to the concept in the fragment, noting any minimal or
awkward incorporations. After providing your explanation, rate the concept’s relevance
on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1
indicates the concept is somewhat present but not obvious or directly demonstrated, and
2 indicates the concept is clearly and effectively incorporated. Provide your rating
using this exact format: "Rating: [[score]]".
[Concept Start]
{concept}
[Concept End]
[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
```

The Instruction Relevance judge uses:

```ini
[System]
Please act as an impartial judge and evaluate whether the sentence fragment provided below
is related to the instruction. Focus solely on the degree of relatedness in terms of
topic, regardless of grammar, coherence, or informativeness.
Begin your evaluation by providing a brief explanation of whether the sentence is related
to the instruction, and point out references related to the instruction. After
providing your explanation, rate the instruction relevance on a scale from 0 to 2,
where 0 indicates the sentence is unrelated to the instruction, 1 indicates it is
somewhat related but only minimally or indirectly relevant in terms of topic, and 2
indicates it is more clearly and directly related to the instruction. Provide your
rating using this exact format: "Rating: [[score]]".
[Instruction Start]
{instruction}
[Instruction End]
[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
```

The Fluency judge uses:

```ini
[System]
Please act as an impartial judge and evaluate the fluency of the sentence fragment provided
below. Focus solely on fluency, disregarding its completeness, relevance, coherence
with any broader context, or informativeness.
Begin your evaluation by briefly describing the fluency of the sentence, noting any
unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures
that may hinder readability. After providing your explanation, rate the sentence’s
fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly
unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but
contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent
and almost perfect. Provide your rating using this exact format: "Rating: [[score]]".
[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
```

## C.6 PRIOR HUMAN VALIDATION OF AXBENCH-STYLE JUDGES

Our Concept Expression, Instruction Relevance, and Fluency evaluators follow the AxBench prompts and ternary scoring scheme and use the same judge model, GPT-4o-mini (Wu et al., 2025a). AxBench evaluated these judges against human ratings using five participants, each of whom scored 30 steered generations spanning six methods on Gemma-2-9B at layer 20. The reported Pearson correlation was 0.57 among human raters and 0.58 between the LLM judge and human raters, indicating that, in the AxBench setting, LLM–human agreement was comparable to inter-human agreement.

The comparable LLM–human and inter-human agreement supports the use of these AxBench-style evaluators in SteerScope. We further complement the three LLM-based scores with established task-specific benchmarks and deterministic evaluators wherever available.

## D ADDITIONAL RESULTS

This section reports complete factor-sweep results for the side-effect analysis and additional results on data dependence and cross-lingual generalization. Trade-off plots show normalized metric degradation against Concept Expression across all evaluated factors. Factor grids show the corresponding absolute metric values as normalized intervention strength varies. All factor grids report pointwise 95% percentile bootstrap confidence intervals from 5,000 concept-level resamples with replacement. Resampling is paired across factors, and the composite score is recomputed within each replicate while preserving each metric’s concept coverage. Methods with multiple factors use shaded bands, whereas methods evaluated at a single nonzero factor use vertical error bars. These intervals quan tify uncertainty with respect to concept sampling conditional on the recorded evaluations; they do not capture training-seed or additional generation and judging uncertainty.

## D.1 COMPOSITE SIDE EFFECTS

![](images/4f391e5af3a970d2c0ecac115f9cb168e101a60943607ad0eb2f060569edb381.jpg)

![](images/a70ba311f3ec2d048a43ff143d8d61060ab51c13dbc376a1509073d18e0eb02a.jpg)  
Figure 5: Complete efficacy–composite-side-effect results across all methods and evaluated factors on Gemma-2-2B-it. Top: mean side effect versus Concept Expression. Bottom: mean side effect versus normalized intervention strength.

![](images/a7106203b9476a89c7a80d3b04da1f2ce98f1891056bb48cbcb7265272be559d.jpg)

![](images/3cb0574eee2d28f045656ebed89b9e36c1b93ec0e97c69d1af64ca6751103b62.jpg)  
Figure 6: Complete efficacy–composite-side-effect results across all methods and evaluated factors on Gemma-2-9B-it. Top: mean side effect versus Concept Expression. Bottom: mean side effect versus normalized intervention strength.

![](images/2931e0b299e0935d124f6b2399244902d243528adf37786ef2f3b66cfdc5f624.jpg)  
Figure 7: Concept Expression across normalized intervention strength on Gemma-2-2B-it (top) and Gemma-2-9B-it (bottom).

![](images/e979a7b7bbf9aba45d6bfee6b9bafdadf3d6cfac4e946fcb02964666517e3e98.jpg)  
Figure 8: Spearman correlations between method rankings produced by different side-effect metrics on Gemma-2-2B-it (left) and Gemma-2-9B-it (right) at layer 20. Each method is evaluated at its Overall-selected method-level steering factor.

## D.1.1 ROBUSTNESS TO METRIC WEIGHTING

The composite side-effect score assigns equal weight to the eleven normalized side-effect metrics. To test whether the resulting comparison depends on this choice, we sample 1,000,000 weight vectors from symmetric Dirichlet distributions, with the weights constrained to sum to one, and recompute the composite score for every method–factor point. For each draw, Prompt Steering is considered undominated if no other nonzero-factor point has both higher Concept Expression and lower composite side effect. Dirichlet(5) concentrates weights near uniformity, Dirichlet(1) is uniform over the weight simplex, and Dirichlet(0.2) emphasizes sparse weightings concentrated on a small number of metrics.

<table><tr><td>Model</td><td>Weight distribution</td><td>Undominated draws</td><td>95% Monte Carlo CI</td></tr><tr><td>Gemma-2-2B-it</td><td>Dirichlet(1)</td><td>99.83%</td><td>[99.83%, 99.84%]</td></tr><tr><td>Gemma-2-2B-it</td><td>Dirichlet(5)</td><td>100.00%</td><td>[100.00%, 100.00%]</td></tr><tr><td>Gemma-2-2B-it</td><td>Dirichlet(0.2)</td><td>88.61%</td><td>[88.55%, 88.68%]</td></tr><tr><td>Gemma-2-9B-it</td><td>Dirichlet(1)</td><td>99.92%</td><td>[99.91%, 99.92%]</td></tr><tr><td>Gemma-2-9B-it</td><td>Dirichlet(5)</td><td>100.00%</td><td>[100.00%, 100.00%]</td></tr><tr><td>Gemma-2-9B-it</td><td>Dirichlet(0.2)</td><td>92.07%</td><td>[92.02%, 92.13%]</td></tr></table>

Table 7: Robustness of Prompt Steering’s Pareto position to alternative weightings of the eleven normalized side-effect metrics. Confidence intervals quantify Monte Carlo sampling error only.

Competing methods therefore almost never dominate the Prompt Steering baseline under uniform or near-uniform metric preferences. This result becomes less robust under highly sparse preferences but remains consistent on both model scales, showing that the comparison against the Prompt Steering baseline is not an artifact of equal weighting. Restricting the comparison to each method’s Overallselected factor yields the same conclusion: under Dirichlet(1), the baseline remains undominated in 99.999% of draws for Gemma-2-2B-it and 99.92% for Gemma-2-9B-it.

## D.2 RANKING AGREEMENT ACROSS SIDE-EFFECT METRICS

To examine whether the aggregate score masks metric-specific differences, we rank methods by their degradation on each side-effect metric at their Overall-selected method-level steering factors. We then compute Spearman correlations between the metric-specific rankings.

The metrics do not produce fully consistent method rankings, supporting the need for multidimensional side-effect evaluation.

Ranking agreement is mixed on Gemma-2-2B-it but generally stronger on Gemma-2-9B-it. Jail breakBench false refusal is a notable exception, showing weak or negative correlations with most other metrics.

We hypothesize that this difference may reflect the range of evaluation examples each model can solve. The 9B model may solve more difficult examples whose successful completion depends on multiple overlapping capabilities; steering-induced disruption of these shared capabilities may therefore manifest across several metrics simultaneously. In contrast, floor effects on the 2B model may obscure such shared degradation.

## D.3 SIGNED-FACTOR ANALYSIS

To examine whether the improvements observed at small positive intervention strengths are direction dependent, we additionally evaluate MMLU on Gemma-2-2B-it at layer 20 using signed normalized factors. As shown in Figure 9, the MMLU gains produced by some positive interventions disappear or become degradations when their steering directions are reversed. This asymmetry is consistent with an interaction between the injected concept and the evaluated capability rather than a generic effect of weak perturbations.

![](images/64fec32e90d2e200be4785b787f8b0ad5502635c314feaa0d4dd35535c47fce1.jpg)  
Figure 9: MMLU accuracy under signed normalized intervention strengths on Gemma-2-2B-it at layer 20. The horizontal dashed line marks the unsteered accuracy at factor 0.

![](images/b819d60b029a3c102ecaf35797d82e00a5e523038cd92fb6acb02237d0aaf556.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/bdd84bf89947d2c8cf8841018f55aa297060300c883f367f83a24dc9033ab0d8.jpg)  
Figure 10: Complete Instruction Relevance results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/c12b6ae408b7cb49da808e3965af4f0b7389902d660ce39d3114138b154360b8.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/3b3dad0a40d43d199400c596416f5c79be4f5542214f59018fa7de4d75e6e704.jpg)  
Figure 11: Complete Instruction Relevance results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/1ea915302801eddc0179fe7ce431a27de34d228b63d974adac3c445e06d601bf.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/7800118b09761d6a0aedd2a1d2e18b60e1b0dd0bda02db1dcf0872081c291afe.jpg)  
Figure 12: Complete Fluency results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/789a6556c559620da8ddf72534f3ce47e72e1cfb0823f00233ac7b334a664d80.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/eec96fe4cd929f37abd435fb348d3dce5507f05449105212c4a618e067b3d49d.jpg)  
Figure 13: Complete Fluency results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

## D.5 TASK CAPABILITIES

![](images/3f92c0c5abc4ed82a6e6992edb95355124c7465b375d40dd68ae04e90bc9bfbc.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/4c8462253b52a791703271699abe9b71e6a34fee6b6f4f81dabd13ae25adacb5.jpg)  
Figure 14: Complete MMLU accuracy results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/3115adb480f74512c2cc16556d94ad57711c6d6820417233c5903fd5280058e6.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/105a794515177eeb9f1b5770cf3f6a726f9af48bc281990d068929b0724e88c6.jpg)  
Figure 15: Complete MMLU accuracy results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/50c9faf9f4dae0e1fd65b502304d9f0bec58ba556a0db6def397c9fef8dccb3f.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/023c082d672989b7f4ecbe33b1b1d75f8ab2ed73a52a0f01dc584d4ebc47cd30.jpg)  
Figure 16: Complete aggregate SuperGLUE results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/a4633c88ab9994e20f088af786854097d19118e6f02a62c92ca567864037157e.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/0264fcbe15dd3a2d7e270274c1134fd8d9bede66daea2dd3ef4259cb05cb8516.jpg)  
Figure 17: Complete aggregate SuperGLUE results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/4f2f567931eb4c290325804cbeaa90ea0870e6f12cbc0f136edeec38ce993e2f.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/ed0ee735da52483938ec4671247fb77bb9c25e896aa5daff25a24a051ab64eb9.jpg)  
Figure 18: Complete MATH accuracy results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/b73cc4165cd689c900ffee313b7389253c91e57de104e4056cb4aa11fec9614f.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/ca52546164d8cceef3858b676e3fd7a30e2b6347f63b2b005c47d457229ddad2.jpg)  
Figure 19: Complete MATH accuracy results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/f570516239479e2a2bb63800d97fa834640f46db6caed12ba6692ec35799d7f1.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/3661ceb8a5ed8230c88533994b56dc3e32904789aaca52228f2d70e484fca0ee.jpg)  
Figure 20: Complete IFEval prompt-strict accuracy results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/0d3f79f1bf03bfe97a52180c697be55452972fb21daa2d6ea03e7be030b2501d.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/d57e15ab3353128bdf7cd89b4c5b3b421a938cfee288df7c9e3ec8f7ed34e29b.jpg)  
Figure 21: Complete IFEval prompt-strict accuracy results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/8a07bd0a11f06a37fe57ef46039e1b0686b8a080cd9b713586f9a95bb54789b0.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/b8362ed41c7afa4ae7dd46541ef2300238ff259d02139b651970c5ed68f5acbc.jpg)  
Figure 22: Complete BBQ ambiguous absolute bias results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/3910c3261b1af137a61dd1010eca15d4a08eefb9c2d41a32d0126ff989cb23cc.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/799ab6ba326b7da7c46adcbfff2ec6d85b25abc3f9396688471e5be01342e1d1.jpg)  
Figure 23: Complete BBQ ambiguous absolute bias results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/278b42ba8888175dcffcf7c7c8819ccf6f97fe64e691aff7586a6afb473c3b8a.jpg)

![](images/d3489047f42a98af287f6878fb71bd379b9998917f4bb462e8daf8f60b04c7b7.jpg)  
Figure 24: Complete BBQ disambiguated absolute bias results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/0e47ff10dbe09b334ffddf89e897367fbdfc7be1fbd336c5386608b13c8d8240.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/92f2f0ec8417542b9ef91313549815954b2d79d7dab0265dcffbc741ed1412c2.jpg)  
Figure 25: Complete BBQ disambiguated absolute bias results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/77dd3a249e3c769a48931c225718fc8c22f9fdc3afc16b28a0b39cbf22986290.jpg)

![](images/4441209b767ce53be5818887232774df7d793f74895ba3261d98dc380636e88b.jpg)  
Figure 26: Complete JailbreakBench attack success rate results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer  
![](images/4a47be2636d930378ec3c9ccc1eba18d4d1a57824fa9308cd6287d04027284cd.jpg)

![](images/df3674be2bb395736c11958ba7194d83c567a9ab2b00f4d19fb9769b380d0f3e.jpg)  
Figure 27: Complete JailbreakBench attack success rate results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/bb478ded44c266baba2b9cd6abf9caa2439f5b230a232e51d8cfe919eda04af2.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/4f94a2c11bb0882d0f00b99a0913e585e7e19aefe168bf121f2472e6b53a27ee.jpg)  
Figure 28: Complete JailbreakBench false refusal rate results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer  
![](images/23ce2d659bbb2478c1e8f290eee03743053bea7397bd55946b4ea0ec3418e7c0.jpg)

![](images/c1291dc853c315604bc4621e314cb4d7848081d2005cd090b7003fb5fde82fda.jpg)  
Figure 29: Complete JailbreakBench false refusal rate results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/359757caa352654ded97f4a547725cfdd5417f126a219957fedae4a8824b7643.jpg)

![](images/8572fd637589f7e7a06366a87cce834dcddb50f2baf8ef123cfcd795099d0830.jpg)  
Figure 30: Complete TruthfulQA binary accuracy results on Gemma-2-2B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

![](images/8447df85d1944605f66fffc3fc5661f515267b3af37b63460bc627061f785632.jpg)  
A-PSR DiffMean SAE HiDRA LAT LoRA ReFT-r1 PCA Prompt Steering SFT Simple Prompt Steering SSV AUSteer FLAS SAE-A HyperSteer Linear Probe LoReFT ODESteer RePS Random S-PSR Spherical Steering StepODESteer

![](images/8d58adcaa046f11acca49f80f03cb1a9a808dfe25b15e926fb9d4055f2a97abe.jpg)  
Figure 31: Complete TruthfulQA binary accuracy results on Gemma-2-9B-it. Top: normalized degradation versus Concept Expression across all factors. Bottom: absolute metric value versus normalized intervention strength.

## D.7 SAMPLE EFFICIENCY

Figures 32 and 33 complement the main-text sample-efficiency analysis with results for both model scales. The left panels report Overall improvement for all 23 methods and the Random control. The right panels report recovery relative to the full-data improvement only for methods with $\Delta O _ { m } ( N ) \geq$ 0.1. Each method uses a fixed factor selected in the full-data setting. SFT uses 20 concepts; all other methods use 50. All sample-efficiency settings, including the full-data reference, use the same 10 evaluation prompts per concept.

![](images/d32921be70014a43e35750c7beb41408a144b60f31df04833217f5868167d708.jpg)

Figure 32: Sample efficiency on Gemma-2-2B-it. Left: factor-0-relative Overall improvement across training-set sizes for all methods. Right: percentage of the full-data improvement recovered, restricted to methods with full-data Overall improvement at least 0.1. Training sizes are 6, 12, 36, 72, and $N = 1 4 4$ labeled rows per concept.  
![](images/ab85fb4ff40c398f7b76d778b8176f0075173110e738ccad8b45a68053e963dd.jpg)  
Figure 33: Sample efficiency on Gemma-2-9B-it. Left: factor-0-relative Overall improvement across training-set sizes for all methods. Right: percentage of the full-data improvement recovered, restricted to methods with full-data Overall improvement at least 0.1. Training sizes are 6, 12, 36, 72, and $N = 1 4 4$ labeled rows per concept.

## D.8 SAMPLE SENSITIVITY

Tables 8 and 9 report mean signed Overall gain $G _ { m }$ and sample sensitivity $S _ { m }$ on Gemma-2-2Bit and Gemma-2-9B-it, respectively. Their definitions are given in Appendix A.3.2. We use each method’s factor selected in the full-data main experiment and keep it fixed across the five trainingsubset seeds. Both statistics are reported for every method without an improvement threshold.

Prompt Steering, Simple Prompt Steering, and SAE have zero sample sensitivity in both settings. All other evaluated methods exhibit nonzero variation across training subsets. We report $G _ { m }$ alongside $S _ { m }$ because low variation alone does not imply a strong steering effect.

Table 8: Sample sensitivity on Gemma-2-2B-it.
<table><tr><td>Method</td><td> $G _ { m }$ </td><td> $S _ { m }$ </td></tr><tr><td>Prompt Steering</td><td>0.7853</td><td>0.0000</td></tr><tr><td>A-PSR</td><td>0.6275</td><td>0.1191</td></tr><tr><td>Simple Prompt Steering</td><td>0.6198</td><td>0.0000</td></tr><tr><td>LoReFT</td><td>0.5342</td><td>0.1237</td></tr><tr><td>LoRA</td><td>0.3920</td><td>0.0997</td></tr><tr><td>SFT</td><td>0.3747</td><td>0.1613</td></tr><tr><td>HyperSteer</td><td>0.2514</td><td>0.0924</td></tr><tr><td>RePS</td><td>0.2022</td><td>0.0804</td></tr><tr><td>S-PSR</td><td>0.1282</td><td>0.0802</td></tr><tr><td>ReFT-r1</td><td>0.1034</td><td>0.0631</td></tr><tr><td>SSV</td><td>0.0670</td><td>0.0619</td></tr><tr><td>FLAS</td><td>0.0526</td><td>0.0485</td></tr><tr><td>SAE</td><td>0.0267</td><td>0.0000</td></tr><tr><td>DiffMean</td><td>0.0265</td><td>0.0389</td></tr><tr><td>SAE-A</td><td>0.0093</td><td>0.0899</td></tr><tr><td>PCA</td><td>0.0087</td><td>0.0449</td></tr><tr><td>Linear Probe</td><td>0.0053</td><td>0.0414</td></tr><tr><td>LAT</td><td>0.0036</td><td>0.0507</td></tr><tr><td>Spherical Steering</td><td>0.0009</td><td>0.0434</td></tr><tr><td>ODESteer</td><td>-0.0031</td><td>0.0345</td></tr><tr><td>HiDRA</td><td>-0.0036</td><td>0.0332</td></tr><tr><td>StepODESteer</td><td>-0.0057</td><td>0.0346</td></tr><tr><td>Random</td><td>-0.0171</td><td>0.0343</td></tr><tr><td>AUSteer</td><td>-0.0173</td><td>0.0267</td></tr></table>

Table 9: Sample sensitivity on Gemma-2-9B-it.
<table><tr><td>Method</td><td> $G _ { m }$ </td><td> $S _ { m }$ </td></tr><tr><td>Prompt Steering</td><td>1.0621</td><td>0.0000</td></tr><tr><td>Simple Prompt Steering</td><td>1.0605</td><td>0.0000</td></tr><tr><td>A-PSR</td><td>0.9143</td><td>0.1168</td></tr><tr><td>HyperSteer</td><td>0.7142</td><td>0.0982</td></tr><tr><td>LoReFT</td><td>0.5867</td><td>0.1221</td></tr><tr><td>RePS</td><td>0.5834</td><td>0.1148</td></tr><tr><td>S-PSR</td><td>0.4694</td><td>0.1088</td></tr><tr><td>SFT</td><td>0.3197</td><td>0.1344</td></tr><tr><td>ReFT-r1</td><td>0.2325</td><td>0.1568</td></tr><tr><td>DiffMean</td><td>0.2086</td><td>0.0821</td></tr><tr><td>FLAS</td><td>0.1709</td><td>0.0933</td></tr><tr><td>Linear Probe</td><td>0.1416</td><td>0.0821</td></tr><tr><td>LoRA</td><td>0.0948</td><td>0.0538</td></tr><tr><td>SAE</td><td>0.0713</td><td>0.0000</td></tr><tr><td>PCA</td><td>0.0589</td><td>0.0595</td></tr><tr><td>LAT</td><td>0.0401</td><td>0.0723</td></tr><tr><td>SAE-A</td><td>0.0312</td><td>0.0847</td></tr><tr><td>HiDRA</td><td>0.0274</td><td>0.0720</td></tr><tr><td>StepODESteer</td><td>0.0154</td><td>0.0561</td></tr><tr><td>ODESteer</td><td>0.0143</td><td>0.0396</td></tr><tr><td>Random</td><td>0.0066</td><td>0.0393</td></tr><tr><td>Spherical Steering</td><td>0.0020</td><td>0.0311</td></tr><tr><td>AUSteer</td><td>0.0003</td><td>0.0210</td></tr><tr><td>SSV</td><td>-0.0007</td><td>0.0383</td></tr></table>

Both models use 24 labeled rows per concept and five training-subset seeds. $G _ { m }$ is mean factor-0-relative Overall improvement, and $S _ { m }$ is the mean per-concept sample standard deviation across subset seeds (ddof=1). All 23 methods and the Random control are included and sorted by $G _ { m }$ within each table. SFT uses 20 con cepts; all other methods use 50.

## D.9 CROSS-LINGUAL GENERALIZATION

Tables 10 and 11 report complete base-relative generalization results for both model scales. Unlike the filtered main-text table, these tables retain the component and Overall effects for every method, including those whose ID Overall gain is too small to report a stable retention ratio. Such ratios are marked N/A. Each method uses one fixed method-level steering factor across all languages.

<table><tr><td rowspan="2">Method</td><td rowspan="2">α||C|</td><td rowspan="2"></td><td colspan="2">Concept Gain</td><td colspan="2">Instr. Deg.</td><td colspan="2">Fluency Deg.</td><td colspan="2">Overall Gain</td><td rowspan="2">Retention</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>Prompt Steering</td><td>1.0</td><td>50</td><td>0.9570</td><td>1.0288</td><td>0.1110</td><td>0.1972</td><td>0.0140</td><td>0.0162</td><td>0.8441</td><td>0.8463</td><td>1.0026</td></tr><tr><td>Simple Prompt Steering</td><td>1.0</td><td>50</td><td>0.7650</td><td>0.8675</td><td>0.3040</td><td>0.4510</td><td>0.0190</td><td>-0.0040</td><td>0.6007</td><td>0.5908</td><td>0.9835</td></tr><tr><td>LoRA</td><td>1.0</td><td>50</td><td>1.0780</td><td>1.1198</td><td>0.3040</td><td>0.4745</td><td>0.0280</td><td>0.0608</td><td>0.7982</td><td>0.7353</td><td>0.9211</td></tr><tr><td>A-PSR</td><td>2.0</td><td>50</td><td>1.3120</td><td>1.3385</td><td>0.4840</td><td>0.5848</td><td>0.0680</td><td>0.0472</td><td>0.9465</td><td>0.8667</td><td>0.9157</td></tr><tr><td>S-PSR</td><td>4.0</td><td>50</td><td>0.7840</td><td>0.6642</td><td>0.5920</td><td>0.5235</td><td>0.3770</td><td>0.3442</td><td>0.3205</td><td>0.2614</td><td>0.8156</td></tr><tr><td>LoReFT</td><td>1.0</td><td>50</td><td>1.2070</td><td>1.0920</td><td>0.4200</td><td>0.5715</td><td>0.0380</td><td>0.1000</td><td>0.8769</td><td>0.6935</td><td>0.7909</td></tr><tr><td>SFT</td><td>1.0</td><td>13</td><td>1.1731</td><td>0.8769</td><td>0.6846</td><td>0.6856</td><td>0.0962</td><td>0.0846</td><td>0.7250</td><td>0.4947</td><td>0.6824</td></tr><tr><td>ReFT-r1</td><td>1.0</td><td>50</td><td>0.6830</td><td>0.6708</td><td>0.3190</td><td>0.5620</td><td>0.2060</td><td>0.3952</td><td>0.4490</td><td>0.2954</td><td>0.6580</td></tr><tr><td>HyperSteer</td><td>1.0</td><td>50</td><td>0.7930</td><td>0.7995</td><td>0.3430</td><td>0.7212</td><td>0.1560</td><td>0.4458</td><td>0.5838</td><td>0.3354</td><td>0.5745</td></tr><tr><td>RePS</td><td>14.0</td><td>50</td><td>0.9190</td><td>0.8925</td><td>0.4790</td><td>0.8200</td><td>0.1690</td><td>0.4288</td><td>0.6329</td><td>0.3627</td><td>0.5732</td></tr><tr><td>SSV</td><td>2.0</td><td>50</td><td>0.7060</td><td>0.6785</td><td>0.5410</td><td>0.7667</td><td>0.3480</td><td>0.5220</td><td>0.3587</td><td>0.2056</td><td>0.5731</td></tr><tr><td>DiffMean</td><td>0.8</td><td>50</td><td>0.3140</td><td>0.3093</td><td>0.2650</td><td>0.6062</td><td>0.2600</td><td>0.5775</td><td>0.2194</td><td>0.1013</td><td>0.4615</td></tr><tr><td>FLAS</td><td>2.5</td><td>50</td><td>1.0610</td><td>0.4652</td><td>0.6710</td><td>0.5720</td><td>0.1460</td><td>0.2805</td><td>0.6152</td><td>0.2618</td><td>0.4256</td></tr><tr><td>SAE-A</td><td>1.8</td><td>50</td><td>0.1070</td><td>0.1410</td><td>0.1910</td><td>0.3002</td><td>0.1770</td><td>0.3428</td><td>0.0593</td><td>0.0500</td><td>N/A</td></tr><tr><td>SAE</td><td>1.8</td><td>50</td><td>0.1580</td><td>0.1745</td><td>0.1520</td><td>0.3143</td><td>0.1280</td><td>0.2755</td><td>0.0738</td><td>0.0386</td><td>N/A</td></tr><tr><td>Linear Probe</td><td>3.0</td><td>50</td><td>0.1080</td><td>0.1100</td><td>0.1150</td><td>0.4115</td><td>0.1170</td><td>0.3900</td><td>0.0859</td><td>0.0324</td><td>N/A</td></tr><tr><td>PCA</td><td>0.8</td><td>50</td><td>0.2440</td><td>0.2352</td><td>0.2520</td><td>0.4358</td><td>0.3580</td><td>0.4925</td><td>0.0819</td><td>0.0267</td><td>N/A</td></tr><tr><td>LAT</td><td>0.8</td><td>50</td><td>0.2410</td><td>0.2930</td><td>0.3160</td><td>0.5408</td><td>0.4710</td><td>0.5788</td><td>0.0528</td><td>0.0233</td><td>N/A</td></tr><tr><td>Spherical Steering</td><td>0.4</td><td>50</td><td>0.0830</td><td>0.0468</td><td>0.1550</td><td>0.1938</td><td>0.2260</td><td>0.2292</td><td>0.0539</td><td>0.0224</td><td>N/A</td></tr><tr><td>ODESteer</td><td>140.0</td><td>50</td><td>0.0410</td><td>0.0370</td><td>0.0830</td><td>0.2288</td><td>0.1690</td><td>0.3515</td><td>0.0309</td><td>0.0122</td><td>N/A</td></tr><tr><td>HiDRA</td><td>1.2</td><td>50</td><td>0.0740</td><td>0.0600</td><td>0.1230</td><td>0.3415</td><td>0.1900</td><td>0.4155</td><td>0.0423</td><td>0.0118</td><td>N/A</td></tr><tr><td>Random</td><td>1.0</td><td>50</td><td>0.0010</td><td>0.0085</td><td>0.0120</td><td>0.0375</td><td>0.0040</td><td>0.0378</td><td>-0.0055</td><td>0.0063</td><td>N/A</td></tr><tr><td>StepODESteer</td><td>176.0</td><td>50</td><td>0.0580</td><td>0.0522</td><td>0.1960</td><td>0.4458</td><td>0.3250</td><td>0.5520</td><td>0.0336</td><td>-0.0002</td><td>N/A</td></tr><tr><td>AUSteer</td><td>15.0</td><td>50</td><td>-0.0080</td><td>-0.0038</td><td>0.0030</td><td>0.0497</td><td>0.0220</td><td>0.0850</td><td>-0.0114</td><td>-0.0086</td><td>N/A</td></tr></table>

Table 10: Complete base-relative cross-lingual generalization results on Gemma-2-2B-it at layer 20, including all 23 methods and the Random control. Positive Concept Gain indicates stronger concept expression; positive Instruction and Fluency Degradation indicate worse behavior after steering. Retention is $\Delta O ^ { \mathrm { O O D } } / \Delta O ^ { \mathrm { I D } }$ and is marked N/A when $\Delta { O } ^ { \mathrm { I D } } < 0 . 1$ . Methods with valid retention are sorted in descending order; |C| is the number of evaluated text concepts.

<table><tr><td rowspan="2">Method</td><td rowspan="2">α |C|</td><td rowspan="2"></td><td colspan="2">Concept Gain</td><td colspan="2">Instr. Deg.</td><td colspan="2">Fluency Deg.</td><td colspan="2">Overall Gain</td><td rowspan="2">Retention</td></tr><tr><td>ID</td><td>OOD</td><td></td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>Linear Probe</td><td>5.0</td><td>50</td><td>0.2340</td><td>0.3465</td><td>0.1490</td><td>0.3528</td><td>0.1590</td><td>0.1873</td><td>0.2122</td><td>0.2604</td><td>1.2270</td></tr><tr><td>LAT</td><td>5.0</td><td>50</td><td>0.1450</td><td>0.2590</td><td>0.1460</td><td>0.4035</td><td>0.2350</td><td>0.2790</td><td>0.1060</td><td>0.1083</td><td>1.0222</td></tr><tr><td>A-PSR</td><td>2.5</td><td>50</td><td>1.4100</td><td>1.4202</td><td>0.1980</td><td>0.2702</td><td>0.1040</td><td>0.0508</td><td>1.1796</td><td>1.1335</td><td>0.9609</td></tr><tr><td>Prompt Steering</td><td>1.0</td><td>50</td><td>1.0840</td><td>1.0830</td><td>0.0380</td><td>0.0792</td><td>0.0420</td><td>0.0305</td><td>1.0280</td><td>0.9775</td><td>0.9509</td></tr><tr><td>LoRA</td><td>1.0</td><td>50</td><td>0.8190</td><td>0.9892</td><td>0.1870</td><td>0.5100</td><td>0.1050</td><td>0.1548</td><td>0.6900</td><td>0.6497</td><td>0.9417</td></tr><tr><td>S-PSR</td><td>2.5</td><td>50</td><td>1.2230</td><td>1.1522</td><td>0.3260</td><td>0.3648</td><td>0.1420</td><td>0.0945</td><td>0.9345</td><td>0.8698</td><td>0.9307</td></tr><tr><td>DiffMean</td><td>1.2</td><td>50</td><td>0.4570</td><td>0.5360</td><td>0.3790</td><td>0.6662</td><td>0.1380</td><td>0.2015</td><td>0.3463</td><td>0.3045</td><td>0.8793</td></tr><tr><td>Simple Prompt Steering</td><td>1.0</td><td>50</td><td>1.4530</td><td>1.4570</td><td>0.5140</td><td>0.6985</td><td>-0.0260</td><td>-0.1210</td><td>0.9846</td><td>0.8476</td><td>0.8609</td></tr><tr><td>SFT</td><td>1.0</td><td>15</td><td>0.8600</td><td>0.7483</td><td>0.5267</td><td>0.5217</td><td>0.2067</td><td>0.1825</td><td>0.5917</td><td>0.4938</td><td>0.8346</td></tr><tr><td>LoReFT</td><td>1.0</td><td>50</td><td>1.2250</td><td>1.1270</td><td>0.3890</td><td>0.5930</td><td>0.1300</td><td>0.1878</td><td>0.9166</td><td>0.7303</td><td>0.7967</td></tr><tr><td>HyperSteer</td><td>1.0</td><td>50</td><td>1.2460</td><td>1.3165</td><td>0.4420</td><td>0.8195</td><td>0.0920</td><td>0.1985</td><td>0.9125</td><td>0.6980</td><td>0.7649</td></tr><tr><td>RePS</td><td>12.0</td><td>50</td><td>1.3870</td><td>1.3365</td><td>0.6630</td><td>0.8885</td><td>0.1420</td><td>0.2082</td><td>0.8789</td><td>0.6722</td><td>0.7648</td></tr><tr><td>FLAS</td><td>2.0</td><td>50</td><td>1.1500</td><td>1.0110</td><td>0.4120</td><td>0.7080</td><td>0.1830</td><td>0.3113</td><td>0.8542</td><td>0.5965</td><td>0.6983</td></tr><tr><td>HiDRA</td><td>2.4</td><td>50</td><td>0.2660</td><td>0.2835</td><td>0.5470</td><td>0.7670</td><td>0.2940</td><td>0.4010</td><td>0.1207</td><td>0.0741</td><td>0.6143</td></tr><tr><td>ReFT-r1</td><td>1.6</td><td>50</td><td>1.2410</td><td>1.2362</td><td>0.9100</td><td>1.2690</td><td>0.2180</td><td>0.3240</td><td>0.5755</td><td>0.3511</td><td>0.6100</td></tr><tr><td>PCA</td><td>5.0</td><td>50</td><td>0.1190</td><td>0.2720</td><td>0.1120</td><td>0.3865</td><td>0.1740</td><td>0.2068</td><td>0.0963</td><td>0.1434</td><td>N/A</td></tr><tr><td>SAE</td><td>3.0</td><td>50</td><td>0.1990</td><td>0.2662</td><td>0.3030</td><td>0.4350</td><td>0.2320</td><td>0.2742</td><td>0.0793</td><td>0.0884</td><td>N/A</td></tr><tr><td>SAE-A</td><td>3.0</td><td>50</td><td>0.1900</td><td>0.2535</td><td>0.2460</td><td>0.3542</td><td>0.1530</td><td>0.1755</td><td>0.0683</td><td>0.0823</td><td>N/A</td></tr><tr><td>Spherical Steering</td><td>0.6</td><td>50</td><td>0.0310</td><td>0.0350</td><td>0.0270</td><td>0.0475</td><td>0.0730</td><td>0.0495</td><td>0.0216</td><td>0.0308</td><td>N/A</td></tr><tr><td>SSV</td><td>3.0</td><td>50</td><td>0.1930</td><td>0.1710</td><td>0.6980</td><td>0.9265</td><td>0.2550</td><td>0.3000</td><td>0.0571</td><td>0.0295</td><td>N/A</td></tr><tr><td>Random</td><td>1.4</td><td>50</td><td>0.0340</td><td>0.0365</td><td>0.0160</td><td>0.0450</td><td>0.0260</td><td>0.0208</td><td>0.0267</td><td>0.0278</td><td>N/A</td></tr><tr><td>ODESteer</td><td>180.0</td><td>50</td><td>0.0460</td><td>0.0510</td><td>0.0380</td><td>0.1298</td><td>0.1800</td><td>0.2998</td><td>0.0370</td><td>0.0173</td><td>N/A</td></tr><tr><td>StepODESteer</td><td>290.0</td><td>50</td><td>0.1080</td><td>0.0982</td><td>0.2170</td><td>0.3723</td><td>0.3510</td><td>0.4955</td><td>0.0528</td><td>0.0107</td><td>N/A</td></tr><tr><td>AUSteer</td><td>10.0</td><td>50</td><td>-0.0090</td><td>0.0002</td><td>-0.0130</td><td>0.0042</td><td>0.0190</td><td>0.0015</td><td>-0.0121</td><td>-0.0005</td><td>N/A</td></tr></table>

Table 11: Complete base-relative cross-lingual generalization results on Gemma-2-9B-it at layer 20, including all 23 methods and the Random control. Positive Concept Gain indicates stronger concept expression; positive Instruction and Fluency Degradation indicate worse behavior after steering. Retention is $\Delta O ^ { \mathrm { O O D } } / \Delta O ^ { \mathrm { I D } }$ and is marked N/A when $\Delta { O } ^ { \mathrm { I D } } < 0 . 1$ . Methods with valid retention are sorted in descending order; |C| is the number of evaluated text concepts.