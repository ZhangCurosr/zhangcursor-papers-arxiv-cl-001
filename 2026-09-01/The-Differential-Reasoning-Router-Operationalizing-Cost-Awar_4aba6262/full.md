# The Differential Reasoning Router: Operationalizing Cost-Aware LLM Annotation in E-commerce

Cheng Lyu, Jingyue Zhang, Vinny DeGenova, Mengwei Li, and Yuanli Pei

Wayfair

{clyu1, jzhang2, vdegenova, mli2, ypei}@wayfair.com

## Abstract

Large Language Models (LLMs) are increasingly used to annotate structured product data in e-commerce, but early deployment often begins as a cold-start problem: only limited pre-launch labels are available, the value of expensive reasoning is unknown, and human review is needed before the system can be trusted at scale. This challenge is especially common in rule-based annotation workflows, where each item must satisfy multiple business rules and both model errors and ambiguous rule boundaries affect final decisions. We introduce the Differential Reasoning Router (DRR), a cost-aware framework for cold-start LLM annotation that jointly optimizes model selection and human escalation. Rather than treating a reasoning model as a default fallback, DRR estimates separate success probabilities for a direct model and a reasoning model at both the sample and business-rule levels, enabling adaptive routing: easy cases are handled directly, reasoning is reserved for cases where it is expected to improve the decision, and likely double-failure or rule-disagreement cases are escalated to human annotators. The resulting labels provide targeted ground truth for prompt engineering, supervised fine-tuning, calibration, and rule refinement, enabling a gradual shift from human-heavy cold-start annotation toward high-confidence automated routing. In a production e-commerce workflow, DRR reaches accuracy parity with the strongest confidence-based router while achieving more than 60% reasoning-token cost savings.

## 1 Introduction

In e-commerce, high-quality structured product data are essential for search, recommendation, merchandising, and compliance systems. Maintaining these data often requires rule-based judgments over multimodal evidence, such as whether an image or product description supports a required attribute, whether the displayed product matches the listing, or whether the evidence is too ambiguous for an automated decision. Human reviewers can make these judgments reliably, but catalog review at billion-product scale creates high cost, long latency, and delayed time-to-market for new or updated listings.

Large Language Models (LLMs) and autonomous agents have shown promise in automating parts of this workflow (Mohta et al., 2023; Calderon et al., 2025), and prior work has studied when LLM labels can substitute for or complement human annotation. Existing approaches that select between direct and reasoning models rely heavily on human-labeled datasets. However, new product attributes and evolving business definitions often require immediate catalog-wide deployment before a stable and representative human-annotated set can be constructed. Moreover, human annotations are initially scarce and expensive to obtain. Random sampling may fail to capture challenging or ambiguous cases, and early annotations can quickly become outdated as preliminary rule definitions are refined over time. This creates a cold-start annotation problem: direct prompting is inexpensive but brittle, reasoning can help selectively at higher cost, and the system must allocate traffic across direct inference, additional reasoning, and human review while evidence about the task is still being accumulated. In this context, cold start therefore refers not to the absence of labels, but to a mandatory yet limited pre-launch seed set that is small relative to catalog scale and still evolving with the business rules.

Existing work on model routing improves adaptive computation and test-time compute allocation (Wang et al., 2025; Liang et al., 2025; Ding et al., 2025; Damani et al., 2025), but cold-start annotation introduces two requirements that are not addressed by conventional routing methods alone. First, the router must estimate the marginal value of reasoning: whether a more expensive reasoning model is likely to change an otherwise direct decision enough to justify its token cost. Scalar confidence and calibration methods can identify uncertain predictions (Kadavath et al., 2022; Ulmer et al., 2024; Khanmohammadi et al., 2025), but they do not directly distinguish examples where reasoning helps from examples where both automated models fail. Second, the router must treat human review as part of the policy. Reasoning models are not reliable universal fallbacks (Shojaee et al., 2025; Zhu et al., 2025), and some failures reflect ambiguous business rules or missing ground truth rather than insufficient model capacity. In these cases, human review is not only a safety mechanism for the current decision, but also a way to acquire labels that reduce future uncertainty.

We propose the Differential Reasoning Router (DRR), a cost-aware framework for cold-start LLM annotation. DRR predicts separate success probabilities for a lightweight Direct model $( M _ { d } )$ and a higher-cost Reasoning model $( M _ { r } )$ , both at the sample level and at the level of individual business rules. These differential estimates allow the router to send easy cases to $M _ { d }$ , reserve $M _ { r }$ for cases with positive expected reasoning value, and escalate likely double-failure or rule-disagreement cases to human reviewers. As reviewed labels accumulate, they can be used for prompt iteration, supervised fine-tuning, calibration, and rule refinement, enabling the system to shift gradually from human-heavy cold-start operation toward higherconfidence automated routing.

Our contributions are threefold. First, we formalize cold-start LLM annotation as a joint routing and label-acquisition problem in which the system chooses among direct inference, reasoning, and human review under limited ground truth. Second, we introduce a differentially supervised, budget-constrained routing policy that estimates the marginal value of reasoning while detecting cases that require abstention. Third, we evaluate DRR in a production e-commerce attributevalidation workflow, where it achieves accuracy parity with the strongest confidence-based router while yielding roughly one-fifth higher reasoningtoken savings and identifying rule-level disagreement patterns that inform ground-truth collection and rule refinement.

## 2 Related Work

LLMs as Data Annotators. Mohta et al. (2023)

benchmark LLMs as annotators across several tasks and show that downstream models trained on LLMgenerated labels consistently underperform those trained on human labels. Complementing this empirical view, Calderon et al. (2025) propose a statistical framework for determining when LLMs can credibly replace human annotators using a limited amount of human labeling. This motivates deployment settings where human labels are scarce but strategically valuable, particularly when a new annotation task is still being calibrated. Domainspecific studies, including financial-data annotation, further show that task expertise, prompt design, and review cost shape whether LLM labels can substitute for human judgments (Aguda et al. 2024). Nevertheless, in domain-specific industrial deployments, hallucinations and reliability concerns often remain a key barrier to adoption. Efforts to mitigate this via calibration and uncertainty estimation, such as probing whether models “know what they know" (Kadavath et al., 2022) or learning confidence signals from model generations and stability under perturbations (Ulmer et al., 2024; Khanmohammadi et al., 2025), are widespread.

Adaptive Computation and Model Routing. Wang et al. (2025) introduce a semantic router for vLLM that predicts a query's reasoning needs and routes it to a cheaper non-reasoning path when appropriate, reducing unnecessary test-time compute. Liang et al. (2025) propose ThinkSwitcher, which dynamically switches a single model between short (fast) and long (slow) chain-of-thought modes based on input complexity. Ding et al. (2025) present BEST-Route, an adaptive routing framework for test-time compute allocation that selects both the model and the best-of-n sampling budget to meet a target quality at minimal estimated cost. Related routing work also studies learning cost-quality trade-offs from data (Damani et al., 2025). While these methods primarily optimize general efficiency, they typically treat the stronger model (or more compute) as a reliable fallback rather than modeling human review as an action for both abstention and label acquisition.

Learning to Defer and Selective Prediction. Learning-to-defer and selective-prediction methods train systems to decide when to defer to an expert rather than make an automated prediction (Madras et al., 2018; Mozannar and Sontag, 2020). These approaches typically study abstention for a fixed predictive model or a fixed human expert under accuracy, fairness, or workload objectives. They do not directly address settings where abstention must be coordinated with model selection between direct and reasoning inference paths, or where deferred examples also serve as ground truth for prompt, rule, and policy refinement during cold-start deployment.

Limits of Reasoning Models. Shojaee et al. (2025) describe an “illusion of thinking", showing that inference-time reasoning effort (e.g., thinkingtoken compute) can exhibit diminishing or even non-monotonic returns as problem complexity increases. In such cases, accuracy may stagnate or collapse even as models initially spend more compute. Zhu et al. (2025) analyze behavioral divergence under “save-thinking" prompts, showing that reasoning models do not reliably suppress deliberation and can re-engage in explicit or implicit thinking depending on the instance. These findings motivate routing policies that estimate the marginal value of reasoning over direct inference instead of assuming that harder examples should automatically receive more reasoning compute.

## 3 Methodology

We formulate rule-based product annotation as costconstrained multimodal routing. Each query $q =$ $( x _ { \mathrm { { i m g } } } , x _ { \mathrm { { t x t } } } )$ contains a product image and metadata, and must be evaluated against business rules $\mathcal { R } =$ $\{ r _ { 1 } , \ldots , r _ { k } \}$ , where k is the total number of rules. A product is valid only if all rules are satisfied. DRR chooses among a low-cost Direct model $( M _ { d } )$ a higher-cost Reasoning model $( M _ { r } )$ , and human review. For each model $m \in \{ d , r \}$ and rule $r _ { j } .$ $y _ { m , j } = 1$ means $M _ { m }$ agrees with human ground truth on that rule, while $y _ { m } ^ { \mathrm { s y s } } = 1$ means it is correct on the full rule set.

The policy $\pi ( q ) ~ \in ~ \{ M _ { d } , M _ { r } , \mathrm { H u m a n } \}$ must choose among three possible actions: the direct model is sufficient, the reasoning model is worth its extra cost, or human review is needed. Let $C _ { d } ( q )$ and $C _ { r } ( q )$ denote the model costs, and define the incremental reasoning cost $\Delta C _ { q } = C _ { r } ( q ) - C _ { d } ( q ) \geq$ 0. The ideal policy minimizes expected error subject to an average reasoning budget $B _ { \mathrm { t a r g e t } }$ and a human-referral budget $H _ { \mathrm { t a r g e t } } { \mathrm { : } }$

$$
\begin{array} { r l } { \underset { \pi } { \operatorname* { m i n } } } & { \mathbb { E } _ { q } [ \mathrm { R i s k } ( \pi ( q ) , q ) ] } \\ { \mathrm { s . t . } } & { \mathbb { E } _ { q } [ \Delta C _ { q } \cdot { \bf 1 } \{ \pi ( q ) = M _ { r } \} ] \leq B _ { \mathrm { t a r g e t } } , } \\ & { \mathbb { E } _ { q } [ { \bf 1 } \{ \pi ( q ) = \mathrm { H u m a n } \} ] \leq H _ { \mathrm { t a r g e t } } . } \end{array}\tag{1}
$$

![](images/b321c75aedb8060e71c1b832935fe5a08d2ad3f30455a90d48290898f3be7eda.jpg)  
Figure 1: Overall DRR architecture for cost-aware threeway routing with human-feedback support.

We enforce the reasoning budget during training with a nonnegative budget multiplier, and handle human referrals with validation-tuned thresholds.

Figure 1 summarizes the design. Expensive multimodal features are computed once and cached; online routing uses only a small supervised model, which keeps the decision layer compatible with production latency.

## 3.1 Architecture: Lightweight Differential Predictor

Feature extraction and fusion. We use a pretrained multimodal encoder to obtain image embeddings $e _ { v }$ and text embeddings $e _ { t }$ . The router does not fine-tune this encoder; it consumes cached features, which keeps routing fast and stable. To capture both unimodal evidence and image-text alignment, we concatenate image, text, and interaction features, $z = [ e _ { v } ; e _ { t } ; e _ { v } \odot e _ { t } ]$ , then pass z through a shared MLP trunk.

Prediction heads. DRR uses three types of heads. System-level heads predict $\hat { p } _ { d } ( q ) = P ( y _ { d } ^ { \mathrm { s y s } } = 1 \mid$ $q )$ and $\hat { p } _ { r } ( q ) = P ( y _ { r } ^ { \mathrm { s y s } } = 1 \mid q )$ , the probabilities that each automated model succeeds on the full rule set. Rule-level heads predict $\hat { p } _ { d , j } ( q )$ 二 $P ( y _ { d , j } = 1 \mid q )$ and ${ \hat { p } } _ { r , j } ( q ) = P ( y _ { r , j } = 1 \mid q )$ , the probabilities that $M _ { d }$ and $M _ { r }$ agree with human ground truth on rule $r _ { j }$ . An ambiguity head predicts $\hat { p } _ { \mathrm { a m b } } ( q )$ , the probability that both automated models fail, so the router can escalate cases where more reasoning is unlikely to help.

## 3.2 Training Objective

DRR is trained to estimate annotator reliability and to spend reasoning budget only where it is expected to improve the decision. For a fixed budget multiplier λ, we combine these goals in an objective minimized over the router parameters $\theta \colon$

$$
\begin{array} { r } { \mathcal { I } ( \theta , \lambda ) = \mathcal { L } _ { \mathrm { s u p } } ( \theta ) + \mathcal { L } _ { \mathrm { r o u t e } } ( \theta ) } \\ { + \lambda ( \bar { C } _ { \mathrm { r o u t e } } - B _ { \mathrm { t a r g e t } } ) . } \end{array}\tag{2}
$$

Here $\mathcal { L } _ { \mathrm { s u p } }$ trains the system-level, rule-level, and ambiguity heads with binary cross-entropy losses; ${ \mathcal { L } } _ { \mathrm { r o u t e } }$ is the expected error of the relaxed routing policy; and $\bar { C } _ { \mathrm { r o u t e } }$ is the expected incremental reasoning cost. The multiplier λ acts as the learned price of spending reasoning budget: larger values make routing to $M _ { r }$ harder to justify.

The supervision is differential rather than distillational: DRR learns whether each model matches human ground truth, not whether $M _ { d }$ should imitate $M _ { r }$ . This distinction is important because reasoning can have negative value when $M _ { r }$ fails on a query while $M _ { d }$ gets it correct. The ambiguity target is positive when both system-level correctness labels are zero, i.e., $y _ { d } ^ { \mathrm { s y s } } = 0$ and $y _ { r } ^ { \mathrm { s y s } } = 0 ;$ we use a weighted binary cross-entropy loss for this head because double failures are less common but operationally important.

The routing loss uses the system-level success estimates to compare the two automated actions. For a query that is not sent to human review, the estimated direct and reasoning risks are $1 - \hat { p } _ { d } ( q )$ and $1 - \hat { p } _ { r } ( q )$ , respectively. The central routing quantity is therefore the Marginal Value of Reasoning (MVOR), the predicted reduction in risk from using $M _ { r }$ instead of $M _ { d } \colon$

$$
\begin{array} { r } { \mathbf { M V O R } ( q ) = \hat { p } _ { r } ( q ) - \hat { p } _ { d } ( q ) . } \end{array}
$$

A positive MVOR means reasoning is expected to improve correctness; a negative MVOR means the direct model is expected to be better. During training, the binary choice between $M _ { d }$ and $M _ { r }$ is relaxed with a sigmoid policy so that the routing loss remains differentiable:

$$
\pi _ { r } ( q ) = \sigma \left( { \frac { \mathbf { M V O R } ( q ) - \lambda \Delta C _ { q } } { T } } \right) ,
$$

where $T$ is a temperature. The term $\lambda \Delta C _ { q }$ is the opportunity cost of spending reasoning compute on query q.

The relaxed policy then determines both terms in Equation 2: ${ \mathcal { L } } _ { \mathrm { r o u t e } }$ averages the expected automated error under this mixture of $M _ { d }$ and $M _ { r }$ while $\bar { C } _ { \mathrm { r o u t e } }$ averages the corresponding incremental reasoning cost. Thus Equation 2 connects probability learning with budgeted routing: the supervised term learns which model is likely to be correct, while the routing and Lagrangian terms learn how selectively to spend reasoning tokens.

## 3.3 Optimization

We optimize Equation 2 with alternating primal and dual updates. The model parameters θ are updated by gradient descent, while the multiplier is updated by gradient ascent on the budget violation. With $\lambda = e ^ { \nu }$ , the log-space update is

$$
\nu  \nu + \eta _ { \lambda } ( \bar { C } _ { \mathrm { r o u t e } } - B _ { \mathrm { t a r g e t } } ) .
$$

If expected reasoning cost exceeds the budget, λ increases and the router becomes more selective; if the model is under budget, λ decreases and more reasoning calls are allowed. Appendix A.5 gives the budget-response and positivity properties of this log-space update, which justify its use as a stable budget price during training.

## 3.4 Inference Policy

At inference time, DRR uses the learned probabilities and the converged budget multiplier $\lambda ^ { * }$ . It first applies the human gate:

$$
\hat { p } _ { \mathrm { a m b } } ( q ) > \tau _ { \mathrm { a m b } } \quad \mathrm { o r } \quad \operatorname* { m a x } ( \hat { p } _ { d } ( q ) , \hat { p } _ { r } ( q ) ) < \tau _ { \mathrm { c o n f } } .
$$

If either condition holds, the query is routed to human review, covering both predicted double failures and broadly uncertain cases.

For the remaining automated cases, the modelselection decision is

$$
\pi ( q ) = \left\{ \begin{array} { l l } { { M _ { r } , } } & { { \mathrm { i f } \mathrm { \bf ~ M V O R } ( q ) > \lambda ^ { * } \Delta C _ { q } , } } \\ { { M _ { d } , } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right.\tag{3}
$$

Appendix A.4 derives this cost-adjusted threshold rule for the automated cases that pass the human gate. The thresholds $\tau _ { \mathrm { a m b } }$ and $\tau _ { \mathrm { c o n f } }$ are tuned on validation data under a human-referral budget. This gives operators two independent controls: $B _ { \mathrm { t a r g e t } }$ governs reasoning spend, while the escalation thresholds govern human review.

## 3.5 Cold-start Controls and Feedback Loop

The policy is designed to support cold-start deployment and continuous improvement. When task-specific labels are scarce, DRR can initially use conservative human-escalation thresholds to maintain catalog quality while routing uncertain or likely double-failure cases to reviewers. These escalations form an informative label stream rather than a purely random sample: they concentrate on ambiguous examples, cases where $M _ { d }$ and $M _ { r }$ disagree, and instances where both automated paths appear unreliable. As review labels accumulate, DRR can be retrained or recalibrated, prompts or supervised fine-tuning data can be updated, and thresholds can be relaxed to reduce human traffic while retaining an abstention pathway for unreliable cases. Rule-level heads further reveal which business rules underlie disagreement, creating a feedback loop for ground-truth collection and rule refinement.

## 4 Experiments

## 4.1 Setup

Task. We evaluate on a production lead-image eligibility task at a major e-commerce platform. Each sample contains a product image and metadata, and the system must decide whether the image can serve as the primary product image. Human ground truth evaluates each sample against k=11 conjunctive business rules spanning objective checks (e.g., no overlaid text, no visible person) and subjective visual judgments (e.g., product is the primary focus, front of product is sufficiently visible). A single rule failure makes the image invalid, so the task is sensitive to both model errors and ambiguous rule interpretation. Appendix A.1 provides the complete rule definitions, and Appendix A.2 shows borderline subjective examples. A sample is unsolvable when both $M _ { d }$ and $M _ { r }$ fail on the full rule set, which motivates DRR's human-review action.

Data split and cold-start protocol. The labeled corpus contains 9,358 examples, split into 6,550 training, 1,403 validation, and 1,405 test examples. These labels come from the mandatory pre-launch review process used for launch approval, model/vendor selection, prompt tuning, and operatingpoint selection; they were not collected as a separate production annotation campaign solely for DRR. This is the sense in which the deployment is cold-start: labels exist, but they are limited relative to catalog scale and still evolving with the business rules. DRR uses the training split to learn task-specific reliability heads, while both DRR and MaxConf use the same validation split to select human-review thresholds. We therefore interpret the comparison as an operational use of required pre-production evidence, not as a label-efficiency claim.

Models. Both automated annotators use Gemini 2.5 Flash Lite with the same structured output prompts; Appendix A.6 provides the full annotation prompts. The Direct model $( M _ { d } )$ runs with thinking disabled, while the Reasoning model $( M _ { r } )$ runs with dynamic thinking enabled, allowing the model to spend additional reasoning tokens when invoked. Note that DRR is not tied to this model version: the router only requires candidate model outputs, realized costs, and ground-truth agreement labels, so newer direct or reasoning models can be substituted by regenerating candidate predictions and recalibrating or retraining the routing layer. We use OpenCLIP ViT-L/14 for feature extraction, with a shared MLP trunk. Embeddings are precomputed and cached.

Baselines. The baselines isolate the design choices in Section 3 using the same model outputs. Direct-only always uses $M _ { d }$ and represents the cheapest automated deployment. Reasoningonly always uses $M _ { r }$ and serves as the reasoningtoken cost baseline. Random routes uniformly between $M _ { d }$ and $M _ { r }$ , testing whether learned routing adds value beyond arbitrary model selection. Oracle is a best-of-two automated selector that chooses a correct automated model whenever either $M _ { d }$ or $M _ { r }$ is correct; it shows the limit of automated selection without human review. MaxConf follows confidence-based query-aware LLM selection baselines (Maurya et al., 2025): it routes a sample to human review when the maximum modelreported Direct/Reasoning confidence is below a threshold, and otherwise selects $M _ { r }$ only when its confidence exceeds $M _ { d } { \bf \dot { s } }$ . Reasoning-token savings for MaxConf count the incremental reasoning cost only on samples where the policy selects $M _ { r }$ , using the same accounting as DRR. MVORonly uses the cost-aware $M _ { d } / M _ { \tau }$ . rule without human escalation, isolating the need for abstention.

<table><tr><td>Method</td><td>Acc (%)</td><td>Human Review (%)</td><td>Reasoning-Token Savings (%)</td></tr><tr><td>Direct-only (Md)</td><td>68.0 [65.6, 70.5]</td><td></td><td>100.0</td></tr><tr><td>Reasoning-only (Mγ)</td><td>69.3 [66.8, 71.6]</td><td></td><td>0.0</td></tr><tr><td>Random</td><td>68.6 [66.5, 70.7]</td><td></td><td>50.0</td></tr><tr><td>MaxConf</td><td>82.0 [80.0, 84.0]</td><td>20.0</td><td>55.2</td></tr><tr><td>MVOR-only</td><td> $6 9 . 5 \ _ { [ 6 7 . 1 , 7 1 . 9 ] }$ </td><td></td><td>56.4</td></tr><tr><td>Oracle</td><td> $7 9 . 1 \ [ 7 7 . 0 , 8 1 . 1 ]$ </td><td></td><td>86.6</td></tr><tr><td>DRR (ours)</td><td>82.8 [80.8, 84.8]</td><td>21.7</td><td>66.2</td></tr></table>

Table 1: Main results on the product attribute test set $_ { ( N = 1 , 4 0 5 ) }$ . Acc is end-to-end accuracy after routing. Brackets are 95% bootstrap confidence intervals. Reasoning-token savings are measured relative to always using $M _ { r }$ . Thresholded methods are selected on validation set; the table reports realized test-set review rates.

For Pareto sweeps, Ambiguity-only uses only the learned double-failure score for human escalation. Thresholded methods are tuned on the validation set under $a \leq 2 0 \%$ human-referral budget.

Metrics. End-to-end accuracy measures whether the final routed decision is correct: automated routes are correct when the selected model matches human ground truth on the full rule set, and humanreviewed samples are counted as correct. Human review rate reports the fraction of samples routed to reviewers. Reasoning token savings measures reduction in reasoning-token spend relative to always using $M _ { r } ;$ it does not include human-review labor. For diagnostic analyses, we distinguish two granularities of double failure: an unsolvable instance is a full sample where both $M _ { d }$ and $M _ { r }$ fail the conjunctive decision, while a rule-level joint failure is a specific business rule on which both models disagree with human ground truth.

## 4.2 Main Results

Table 1 addresses the practical question behind the experiment: after allocating limited human review, how much reasoning compute is still needed. The single-model and random rows show that invoking or randomly selecting the reasoning model changes accuracy only modestly. The policies that can route difficult cases to humans reach a higher accuracy range. The discussion below separates these effects into human review, cost-aware reasoning, and ambiguity supervision.

Human review must be a routing action. The methods that always choose between $M _ { d }$ and $M _ { r }$ top out at 79.1% accuracy with Oracle, below the thresholded methods that can route examples to human review. This pattern is expected when some samples are unsolvable by both candidate models: choosing the better automated output is still insufficient for those cases. Human review therefore needs to be modeled as a routing action, rather than added only after automated model selection.

<table><tr><td>Zone</td><td>N Inst.</td><td>% Inst.</td><td> $M _ { d }$  Acc</td><td>Mr Acc</td><td>% Unsolv. Inst.</td></tr><tr><td>Economy</td><td>570</td><td>40.6</td><td>76.7</td><td>78.6</td><td>12.8</td></tr><tr><td>Reasoning</td><td>530</td><td>37.7</td><td>75.1</td><td>79.6</td><td>12.5</td></tr><tr><td>Ambiguity</td><td>305</td><td>21.7</td><td>39.7</td><td>33.8</td><td>50.8</td></tr><tr><td>Population</td><td>1405</td><td>100</td><td>68.0</td><td>69.3</td><td>20.9</td></tr></table>

Table 2: Per-zone statistics on the test set. Zones correspond to DRR's three inference actions: direct routing, reasoning routing, and human escalation. Unsolv. Inst. is the percentage of instances in each zone where both automated models fail the full rule set.

Cost-aware routing must be paired with abstention. MVOR-only saves reasoning tokens but remains close to the single-model baselines in accuracy. This separates two effects: the MVOR threshold controls when to spend reasoning tokens, but it does not by itself identify cases that should leave the automated path. After adding the ambiguity gate, DRR reaches the accuracy range of the methods that route examples to humans while achieving more than 60% reasoning-token cost savings compared to MaxConf. Thus cost-aware model selection is most useful when paired with abstention for likely double failures.

Ambiguity supervision adds signal beyond confidence. MaxConf performs well because modelreported confidence is a useful first filter: many difficult examples are also low-confidence examples. However, for cold-start annotation, the review set is also training signal for the next iteration, so the useful cases are not necessarily the least confident ones. The more valuable cases are those that reveal where automation fails, especially examples likely to fail under both $M _ { d }$ and $M _ { r }$ or to expose unstable rule boundaries. DRR uses differential supervision to learn direct-model success, reasoning-model success, and joint automated failure, giving the human gate a task-specific signal beyond confidence alone.

## 4.3 Routing Zone Analysis

Table 2 checks whether the inference policy creates meaningful regions. The Ambiguity Zone has the highest unsolvable-instance rate, while the Economy and Reasoning Zones retain mostly automatable samples. Intuitively, Economy examples tend to have sufficient direct evidence, Reasoning examples require additional cross-checking of the image and metadata, and Ambiguity examples resemble the borderline subjective cases in Appendix A.2. This validates DRR's ability to preserve human capacity while choosing between cheap and expensive model calls.

<table><tr><td>Rule</td><td>Md Acc</td><td>Mr Acc</td><td>% Joint Fail</td><td>Type</td></tr><tr><td>D1: Primary Focus</td><td>68.1</td><td>79.4</td><td>17.4</td><td>Subj.</td></tr><tr><td>E1: Front Visible</td><td>76.7</td><td>78.5</td><td>16.7</td><td>Subj.</td></tr><tr><td>D5: Photorealistic</td><td>80.6</td><td>81.9</td><td>14.5</td><td>Subj.</td></tr><tr><td>D3: Standard View</td><td>79.3</td><td>80.1</td><td>14.0</td><td>Subj.</td></tr><tr><td>E3: Correct Set</td><td>90.1</td><td>84.3</td><td>6.3</td><td>Subj.</td></tr><tr><td>D2: Functionality</td><td>95.9</td><td>91.0</td><td>0.0</td><td>Obj.</td></tr><tr><td>E2: Standard State</td><td>95.2</td><td>94.2</td><td>0.0</td><td>Obj.</td></tr><tr><td>E4: Props</td><td>96.9</td><td>96.7</td><td>0.0</td><td>Obj.</td></tr><tr><td>D4: Dimensions</td><td>98.7</td><td>98.0</td><td>0.0</td><td>Obj.</td></tr><tr><td>E5: No Text</td><td>97.7</td><td>99.0</td><td>0.0</td><td>Obj.</td></tr><tr><td>D6: No Person</td><td>99.0</td><td>99.3</td><td>0.0</td><td>Obj.</td></tr></table>

Table 3: Per-rule diagnostics. Joint Fail is the percentage of samples for which both automated models fail the same rule. Subjective rules from Appendix A.2 concentrate these rule-level joint failures, while objective checks are more stable.

## 4.4 Per-Rule Diagnostics

Table 3 explains why the ambiguity region exists: rule-level joint failures concentrate in subjective rules, matching the borderline cases in Appendix A.2. Objective checks are more stable because their criteria are directly verifiable. This validates the rule-level heads in Section 3: they turn system-level routing failures into diagnoses of which business rules need clarification or human oversight. Human referrals therefore serve both as a quality safeguard and as targeted ground truth for future router training and rule refinement.

## 4.5 Pareto Analysis

Figure 2 treats the human-referral threshold as a deployment control for cold-start annotation. Moving right allocates more reviewer capacity while rules, prompts, and router calibration are still being refined; moving left preserves human capacity but requires more automated decisions. The steep accuracy gains at low review budgets show why selective human escalation is valuable during rollout, and the operating region in Table 1 shows that DRR remains competitive with simple confidencebased escalation while retaining explicit control over reasoning-token spend.

![](images/4f1c52c74670d5d0502bc890fb0d837b5b94f7c81e27c89abcfaeba5c40e4d4c.jpg)  
Figure 2: Accuracy-human budget trade-off. The xaxis varies the allowed human-referral budget, and the dashed line marks the operating point used in Table 1. Dotted horizontal lines show automated reference points. The curves show how different escalation policies convert limited human review into end-to-end accuracy during cold-start rollout.

## 5 Conclusions

DRR shows that routing can do more than select the cheapest successful inference path. By estimating where direct inference is sufficient, where reasoning is likely to help, and where both automated paths are unreliable, the framework turns cold-start LLM annotation into an explicit operating policy. Its routing zones and rule-level diagnostics identify which traffic can be automated, which cases deserve reasoning tokens, and which examples should become human-labeled ground truth. Rather than waiting for a large balanced evaluation set or committing to a single model choice, practitioners can launch with a controlled human-review budget, monitor the accuracy-review tradeoff, and improve prompts, calibration, training data, and rule definitions as the task warms up.

## Limitations

This study evaluates DRR on a specific production e-commerce annotation workflow with rule based annotations. Although the framework is intended for rule-based LLM annotations more broadly, its behavior may differ across domains, modalities, rule structures, and operational review processes.

Our experiments also reflect the model landscape at the time of deployment. Although DRR is model-agnostic by design, commercial LLMs evolve rapidly. For each new pair of models, we recommend regenerating candidate predictions and recalibrating or retraining the router to maximize performance gains.

We do not evaluate label efficiency across training-label budgets. The reported DRR result uses the full pre-launch training split, while confidence thresholding requires only validation-set operating-point selection. Thus, our comparison should be read as how to use a mandatory preproduction labeled corpus in deployment, not as evidence that DRR is preferable when all labels must be newly acquired.

Finally, we treat human review as ground truth, matching its role in the production decision process. In practice, human labels can contain disagreement or inconsistency, especially for subjective rules. Such noise may affect both reported accuracy and the supervision used to train ambiguity and rulelevel failure signals.

## References

Toyin D. Aguda, Suchetha Siddagangappa, Elena Kochkina, Simerjot Kaur, Dongsheng Wang, and Charese Smiley. 2024. Large language models as financial data annotators: A study on effectiveness and efficiency. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 10124–10145, Torino, Italia. ELRA and ICCL.

Nitay Calderon, Roi Reichart, and Rotem Dror. 2025. The alternative annotator test for LLM-as-a-judge: How to statistically justify replacing human annotators with LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16051– 16081, Vienna, Austria. Association for Computational Linguistics.

Mehul Damani, Idan Shenfeld, Andi Peng, Andreea Bobu, and Jacob Andreas. 2025. Learning how hard to think: Input-adaptive allocation of lm computation. In International Conference on Learning Representations, volume 2025, pages 102783–102802.

Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian Del Carmen Hipolito Garcia, Menglin Xia, Laks V. S. Lakshmanan, Qingyun Wu, and Victor Rühle. 2025. Best-route: adaptive llm routing with test-time optimal compute. In Proceedings of the 42nd International Conference on Machine Learning, ICML'25. JMLR.org.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language models (mostly) know what they know. Preprint, arXiv:2207.05221.

Reza Khanmohammadi, Erfan Miahi, Mehrsa Mardikoraem, Simerjot Kaur, Ivan Brugere, Charese Smiley, Kundan S Thind, and Mohammad M. Ghassemi. 2025. Calibrating LLM confidence by probing perturbed representation stability. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 10448–10514, Suzhou, China. Association for Computational Linguistics.

Guosheng Liang, Longguang Zhong, Ziyi Yang, and Xiaojun Quan. 2025. ThinkSwitcher: When to think hard, when to think fast. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 5185–5201, Suzhou, China. Association for Computational Linguistics.

David Madras, Toni Pitassi, and Richard Zemel. 2018. Predict responsibly: Improving fairness and accuracy by learning to defer. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Kaushal Kumar Maurya, Kv Aditya Srivatsa, and Ekaterina Kochmar. 2025. SelectLLM: Query-aware efficient selection algorithm for large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20847–20863, Vienna, Austria. Association for Computational Linguistics.

Jay Mohta, Kenan Ak, Yan Xu, and Mingwei Shen. 2023. Are large language models good annotators? In Proceedings on "I Can't Believe It's Not Better: Failure Modes in the Age of Foundation Models" at NeurIPS 2023 Workshops, volume 239 of Proceedings of Machine Learning Research, pages 38–48. PMLR.

Hussein Mozannar and David Sontag. 2020. Consistent estimators for learning to defer to an expert. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7076–7087. PMLR.

Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh vahid, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. 2025. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 108018–108059. Curran Associates, Inc.

Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seong Oh. 2024. Calibrating large language models using their generations only. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15440–15459, Bangkok, Thailand. Association for Computational Linguistics.

Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, and Huamin Chen. 2025. When to reason: Semantic router for vllm. Preprint, arXiv:2510.08731.

Rongzhi Zhu, Yi Liu, Zequn Sun, Yiwei Wang, and Wei Hu. 2025. When can large reasoning models save thinking? mechanistic analysis of behavioral divergence in reasoning. Preprint, arXiv:2505.15276.

## A Appendix

## A.1 Lead Image Eligibility Rules

The production task studied in this paper is leadimage eligibility: given a product image and listing metadata, decide whether the image is suitable to appear as the primary image shown to customers. The decision is rule-based rather than a single visual-classification judgment. A product image is valid only if it satisfies a conjunction of business rules covering product visibility, presentation state, image cleanliness, product-set consistency, and prohibited visual elements. If any rule fails, the full image-level decision is invalid.

Table 4 lists the 11 rules used for annotation, along with representative examples. The rules span relatively objective visual checks, such as detecting overlaid text, visible people, or dimension markings, and more contextual judgments, such as whether the target product is the primary focus, whether enough of the front is visible, or whether the displayed items match the listed set composition. This mix creates cases where direct inference is sufficient, cases where additional reasoning may help, and cases where human review remains necessary.

## A.2 Qualitative Examples of Subjective Rules

Table 5 shows borderline cases for rules with higher subjectivity. In these examples, the visual evidence is present, but the business-rule boundary is underspecified: a product may compete with surrounding room context, be shown at an angle that partially obscures the front, include small edited artifacts, or make set quantity difficult to infer from the image alone. Such cases can produce disagreement between automated annotators and human reviewers, and sometimes between human reviewers themselves.

The examples also clarify why an ambiguity pathway and rule-level diagnostics are useful. Additional reasoning does not always resolve underspecified visual judgments, while rule-level predictions identify which business rule is responsible for a system-level failure. In deployment, reviewed ambiguity cases provide both training signal for future router updates and evidence for refining the underlying rule specification.

## A.3 Implementation Details

DRR is implemented as a lightweight supervised routing layer on top of frozen multimodal features. Image and text embeddings are generated offline and cached, so online inference only requires a compact neural router rather than repeated feature extraction. The router shares a small trunk across system-level, rule-level, and ambiguity heads, which keeps the decision layer inexpensive while preserving rule-level diagnostics.

Training uses standard regularized optimization with early stopping on validation performance. The Lagrangian multiplier is updated during training to enforce the reasoning-budget constraint, while human-escalation thresholds are selected on validation data under the target review budget. No additional fine-tuning of the underlying direct or reasoning LLMs is required; all adaptation is concentrated in the routing layer.

## A.4 Optimality of Threshold Policy

This section derives the automated model-selection rule used after the human-escalation gate has been applied. It does not claim optimality for the full three-way policy; the human gate is tuned separately under the review budget. Let A denote the set of queries that pass this gate:

$$
\begin{array} { r } { \mathcal { A } = \{ q : \hat { p } _ { \mathrm { a m b } } ( q ) \leq \tau _ { \mathrm { a m b } } \mathrm { a n d } \quad } \\ { \operatorname* { m a x } ( \hat { p } _ { d } ( q ) , \hat { p } _ { r } ( q ) ) \geq \tau _ { \mathrm { c o n f } } \} . } \end{array}
$$

Conditional on $q \in { \mathcal { A } } .$ , the remaining action set is binary, $\{ M _ { d } , M _ { r } \}$

For a query q, define the estimated risks of the direct and reasoning models as

$$
R _ { d } ( q ) = 1 - \hat { p } _ { d } ( q ) , \qquad R _ { r } ( q ) = 1 - \hat { p } _ { r } ( q ) ,
$$

and let $\Delta C _ { q } = C _ { r } ( q ) - C _ { d } ( q ) \geq 0$ be the incremental cost of invoking reasoning. The Marginal Value of Reasoning is the estimated risk reduction from routing to $M _ { r } \mathbf { . }$

$$
\mathbf { M V O R } ( q ) = R _ { d } ( q ) - R _ { r } ( q ) = \hat { p } _ { r } ( q ) - \hat { p } _ { d } ( q ) .
$$

Consider the binary policy $\pi : \mathcal { A }  \{ M _ { d } , M _ { r } \}$ with the same incremental reasoning-budget constraint used in Section 3, restricted to queries that are not sent to human review:

$$
\begin{array} { r l } { \underset { \pi } { \operatorname* { m i n } } } & { \mathbb { E } [ R _ { \pi ( q ) } ( q ) \mid q \in \mathcal { A } ] } \\ { \mathrm { s . t . } } & { \mathbb { E } [ \Delta C _ { q } \cdot { \bf 1 } \{ \pi ( q ) = M _ { r } \} \mid q \in \mathcal { A } ] \leq B _ { \mathrm { t a r g e t } } . } \end{array}\tag{4}
$$

Proposition 1 (Threshold Form of Model Selection). For any fxed multiplier $\lambda \geq 0 ,$ minimizing the Lagrangian relaxation of Equation 4 over

<table><tr><td>Rule</td><td>Criterion</td><td>Example Type</td><td>Example</td></tr><tr><td>E1: Front Visible</td><td>At least 75% of the product front is visible.</td><td>Eligible</td><td><img src="images/a15ca5af839f864c42153bf231464d2d744618e1c23dd519505d7f34556b61a9.jpg"/></td></tr><tr><td>E2: Standard State</td><td>The product is shown in its normal, non-functional state.</td><td>Eligible</td><td><img src="images/cecc1f06e8217360fa06a49740c9414d8c13908b6e0eb4ac79b79322cfbe6146.jpg"/></td></tr><tr><td></td><td>The image shows the correct quantity or all pieces included in the set.</td><td>Eligible</td><td><img src="images/726b75804ef66228d410fc8423adcd99fa1ea05df8db69b5d3f492cd8746c741.jpg"/></td></tr><tr><td>E4: Props</td><td>Props are acceptable only if the target product remains the main focus.</td><td>Eligible</td><td><img src="images/e147eab9e490989f7b931d2c3b3c56c23428365ea1468232a83e19dc8d10c027.jpg"/></td></tr><tr><td>E5: No Text</td><td>The image contains no added text, labels, badges, or promotional callouts.</td><td>Eligible</td><td><img src="images/bc30aa61dbe23fb644a5c5bb68c39f2c7e550b65d7d402e5d2d5385fac87a86e.jpg"/></td></tr><tr><td>D1: Primary Focus</td><td>The target product must be the main visual focus of the image.</td><td>Violation</td><td><img src="images/966fa6f723a4b4b980af31c7184c04d420762c799815593e0edf9134c4b9fd19.jpg"/></td></tr><tr><td>D2: Functionality</td><td>The product should not be shown in use or in an altered functional state.</td><td>Violation</td><td><img src="images/4361c556c7f4fe9feda7290cc8e541cfa43f5a1c741cc94b8f470bb7e95ddb8b.jpg"/></td></tr><tr><td>D3: Standard View</td><td>The image should show a standard product view, not mainly a side, top, close-up, or detail view.</td><td>Violation</td><td><img src="images/d8d2e79cc653539bb1901f682706dc12c396a78ece51ce334498b3e519ae2684.jpg"/></td></tr><tr><td>D4: Dimensions</td><td>The image should not contain dimensions, arrows, schematics, measurement labels, or similar diagram elements.</td><td>Violation</td><td><img src="images/2ed99903ec9207141e171fc97ddb67226acf0cbaebbcea9be5dd9a1a689d8ded.jpg"/></td></tr><tr><td>D5: Photorealistic</td><td>The image should be a clean photograph or photorealistic render, without obvious compositing or unrelated inserts.</td><td>Violation</td><td><img src="images/4701598bb2ba8d491b6c149152761ed6a4e4f3eb87653d2f87d0298624b6a90a.jpg"/></td></tr><tr><td>D6: No Person</td><td>No person or visible body part should appear in the image.</td><td>Violation</td><td><img src="images/7587b0a5d8f9ab9d4ff1c172aef3543be378857fae25484011c911112c065086.jpg"/></td></tr></table>

Table 4: Lead image eligibility rules with representative examples. A product image is valid only when all rules are satisfied.

$q \in { \mathcal { A } }$ decomposes into independent per-query decisions. For each query, the cost-adjusted decision routes to $M _ { r }$ exactly when

$$
M V O R ( q ) > \lambda \Delta C _ { q } ,\tag{5}
$$

and otherwise routes to $M _ { d }$

Proof. For a fixed λ, the Lagrangian relaxation differs from the constrained objective only by the per-query penalty for using $M _ { r }$

$$
R _ { \pi ( q ) } ( q ) + \lambda \Delta C _ { q } \cdot { \bf 1 } \{ \pi ( q ) = M _ { r } \} .
$$

Because the expectation is linear, the optimal policy can be chosen independently for each query.

Routing to $M _ { r }$ is preferred when

$$
\begin{array} { r } { R _ { r } ( q ) + \lambda \Delta C _ { q } < R _ { d } ( q ) } \\ { R _ { d } ( q ) - R _ { r } ( q ) > \lambda \Delta C _ { q } } \\ { \mathbf { M V O R } ( q ) > \lambda \Delta C _ { q } . } \end{array}
$$

Ties are broken conservatively in favor of $M _ { d }$ matching the inference rule in Section 3. □

The multiplier λ can be interpreted as a learned shadow price for reasoning budget. Here, “shadow price" simply means the penalty assigned to spending one additional unit of reasoning budget in the routing rule: increasing λ makes reasoning more expensive in the router's decision rule, so fewer queries are sent to $M _ { r } . \mathrm { H } \hat { p } _ { d }$ and $\hat { p } _ { r }$ are calibrated, this threshold is optimal for true expected risk under the binary model-selection problem; otherwise it is optimal for the estimated risks used by the router. This is why DRR uses supervised probability estimation and validation-based operating-point selection rather than relying on raw model confidence alone.

Table 5: Borderline examples for rules with higher subjectivity. These cases illustrate why human review and rule-level diagnostics remain useful even when reasoning models are available.
<table><tr><td>Rule</td><td>Source of Subjectivity</td><td>Example A</td><td>Example B</td></tr><tr><td rowspan="2">D1: Primary Focus</td><td rowspan="2">Requires judging whether the target product dominates the scene. Here, the fireplace mantel blends into the room and competes with surrounding objects.</td><td></td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>E1: Front Visible</td><td>Requires estimating whether enough of the product front is visible. Example A shows the fireplace mantel at a strong angle, making the visible front area difficult to judge. Example B shows the table in a staged scene, where surrounding objects make the front view less clear.</td><td></td><td></td></tr><tr><td>D5: Photorealistic</td><td>Requires judging whether small digitally added or composited elements are severe enough to affect image eligibility. Both examples contain minor edited elements, but the overall product presentation remains photorealistic, so these artifacts may be ignored or accepted by human annotators and the model.</td><td></td><td></td></tr><tr><td>D3: Standard View</td><td>Requires judging whether an angled or elevated view still counts as a standard product view. Example A shows a three-quarter angled view that still represents the product, while Example B uses an elevated room-view angle that may be interpreted differently by annotators.</td><td></td><td></td></tr><tr><td>E3: Correct Set</td><td>Requires judging whether the visible items match the listed quantity or set composition. Example A lists a 2-piece bedding set, but appears to show three items. Example B lists 4 privacy screens/rolls, but the continuous installation makes the quantity hard to verify.</td><td><img src="images/70a39309f193d376b755d3936b10b3a7874242724f3644481170e888533da2ac.jpg"/></td><td></td></tr></table>

The same monotonicity appears in the relaxed training policy,

$$
\pi _ { r } ( q ) = \sigma \left( { \frac { \mathbf { M V O R } ( q ) - \lambda \Delta C _ { q } } { T } } \right) .
$$

For fixed temperature $T > 0$ , the probability of routing to $M _ { r }$ increases with MVOR and decreases with both incremental cost and the learned budget price λ.

## A.5 Dual Update and Constraint Stabilization

DRR uses the Lagrangian term in Equation 2 to discourage routing policies that exceed the reasoningtoken budget. Let

$$
g ( \theta ) = \bar { C } _ { \mathrm { r o u t e } } ( \theta ) - B _ { \mathrm { t a r g e t } }
$$

denote the budget violation of the relaxed routing policy. The primal update treats the current multiplier as fixed and updates the router parameters

using

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s u p } } ( \theta ) + \mathcal { L } _ { \mathrm { r o u t e } } ( \theta ) + \lambda _ { t } g ( \theta ) . } \end{array}
$$

The dual variable is then updated in log space:

$$
\nu _ { t + 1 } = \nu _ { t } + \eta _ { \lambda } g ( \theta _ { t + 1 } ) , \qquad \lambda _ { t } = e ^ { \nu _ { t } } .\tag{6}
$$

Because the router is a neural model, this section does not claim global convergence to a constrained optimum. Instead, it establishes the exact stabilization properties of the implemented dual update and explains why it is appropriate for enforcing the budget during training.

Proposition 2 (Budget Response of the Log-Space Update). For any step size $\eta _ { \lambda } > 0$ , the update in Equation 6 satisfies

$$
\lambda _ { t + 1 } = \lambda _ { t } \exp ( \eta _ { \lambda } g ( \theta _ { t + 1 } ) ) .
$$

Therefore $\lambda _ { t }$ remains positive for all t; if the expected routing cost exceeds the budget, λ increases; if the router is under budget, λ decreases; and if the budget is exactly met, the dual variable is unchanged.

Proof. Since $\lambda _ { t } = e ^ { \nu _ { t } }$ , exponentiating Equation 6

gives

$$
\begin{array} { r l } & { \lambda _ { t + 1 } = e ^ { \nu _ { t + 1 } } } \\ & { \qquad = e ^ { \nu _ { t } } \exp ( \eta _ { \lambda } g ( \theta _ { t + 1 } ) ) } \\ & { \qquad = \lambda _ { t } \exp ( \eta _ { \lambda } g ( \theta _ { t + 1 } ) ) . } \end{array}
$$

The exponential term is always positive, so positivity is preserved. If $g ( \theta _ { t + 1 } ) > 0$ , the exponential factor is greater than one; if $g ( \theta _ { t + 1 } ) < 0 .$ , it is less than one; and if $g ( \theta _ { t + 1 } ) = 0$ , it equals one. □

Proposition 3 (Mirror-Ascent Interpretation). For xed router parameters $\theta _ { t + 1 }$ , the multiplicative update in Proposition 2 is the solution to the one-step mirror-ascent problem

$$
\lambda _ { t + 1 } = \underset { u > 0 } { \arg \operatorname* { m a x } } \left\{ \eta _ { \lambda } g ( \theta _ { t + 1 } ) u - D ( u \| \lambda _ { t } ) \right\} ,
$$

where $D ( u | | \lambda _ { t } ) = u \log ( u / \lambda _ { t } ) - u + \lambda _ { t }$ is the scalar relative-entropy divergence.

Proof. Differentiate the objective with respect to u:

$$
\eta _ { \lambda } g ( \theta _ { t + 1 } ) - \log ( u / \lambda _ { t } ) .
$$

Setting this derivative to zero yields $\log ( u / \lambda _ { t } ) =$ $\eta _ { \lambda } g ( \theta _ { t + 1 } )$ , or

$$
\begin{array} { r } { u = \lambda _ { t } \exp ( \eta _ { \lambda } g ( \theta _ { t + 1 } ) ) . } \end{array}
$$

The objective is strictly concave in $u ,$ so the maximizer is unique. □

These propositions give the property needed by DRR: the multiplier is a stable, positive budget price that moves in the correct direction after every batch or epoch. When routing exceeds the target budget, the next primal update penalizes reasoning more strongly; when routing is below budget, the penalty relaxes. This feedback mechanism is sufficient for the role λ plays in the router, namely controlling the threshold $\mathbf { M V O R } ( q ) > \lambda \Delta C _ { q }$

Classical primal-dual convergence results require convexity, compactness, and exact or appropriately controlled stochastic updates. Those assumptions do not hold for the non-convex neural router used here, so we do not claim global convergence or optimal constraint satisfaction from the theory alone. In practice, the final operating point is selected on validation data, and the learned multiplier is used only together with validation-tuned human-escalation thresholds.

## A.6 Lead Image Evaluation Prompt

Each evaluation uses two prompt components. The system instruction defines the eligibility rules, confidence scoring, and required JSON output. The user message provides the product-specific inputs for each evaluation: the candidate image, the target product, and the catalog text needed to verify quantity, set composition, and visual eligibility. Placeholders such as [product name] and [image] are replaced with product-specific values before inference.

## System Instruction

You are an expert AI system performing quality assurance on e-commerce product images.   
For each item, you will receive the evaluation rules below, product information, and a candidate lead image   
Instructions:   
- Apply all rules exactly as written.   
- Do not question or reinterpret the rules or product class.   
- Evaluate every rule and provide concise rule-level reasoning   
If a rule is ambiguous or uncertain, state why and reflect that uncertainty in the confidence score.   
Return only the required JSON structure.   
Rule key:   
- E = Eligibility requirement. The image must satisfy the rule   
- D = Disqualification condition. The image must not violate the rule.   
Rules:   
E1\_Front\_Visible\_Enough: The image must show at least 75% of the target product's front, with no major truncation or cropping. The target   
product must be the primary focus. Forward-facing and angled front-facing views are acceptable when the product is visible and   
unobstructed.   
Examples: Pass - a coffee table or sofa shown from the front or slight angle, fully in frame. Fail - a product cropped at major edges, less   
than 75% front visible, or visually dominated by foreground objects.   
E2\_Standard\_State: The product must be shown in a standard, non-functional state. In-use styling and minor adjustments are acceptable, such   
as books in a bookcase, decor on a coffee table, an extended luggage handle, or swiveled wheels. Major functional changes that alter   
product form are not acceptable.   
Examples: Pass - upright suitcase with handle extended; styled bookcase; coffee table with decor; sofa with pillows. Fail - sofa bed   
unfolded; dresser drawers open; dining table extension pulled apart; storage ottoman with lid removed.   
E3\_Correct\_DSQ\_Set: The image must show the correct quantity or all pieces in the set. Use quantity/set information from the product name,   
description, and feature bullets as ground truth, even if it conflicts with the DSQ field. If product text is silent, use DSQ. If   
neither product text nor DSQ specifies quantity, assume one item. Show multiple products only when explicitly specified. Do not treat   
separate items as one unit unless inseparable by design.   
Examples: Pass - "Set of 4 Dining Chairs" shows 4 chairs; "10-piece cookware set" shows all 10 pieces; DSQ=4 and product text is silent,   
image shows 4 items. Fail - product text says 3 but image shows 1; set of 2 nightstands shows 1; DSQ=1 but image shows 5 separate sofas.   
E4\_Props\_Appropriate: Props are acceptable only if the target product remains the primary focus.   
Examples: Pass - coffee table with mug/book; sofa with pillows that do not obscure it. Fail - chair mostly covered by a prop; dresser mostly   
hidden behind other furniture.   
E5\_No\_Text\_In\_Image: The image must not contain overlaid or digitally added text, even if it does not obscure the product. Text physically   
printed on the product or visible in decorative background items is allowed when it is not the main focus.   
Examples: Pass - wall art text in the background; book titles; branding or printed text that is part of the product design. Fail   
promotional banner, watermark, URL, measurement callouts, or digitally added "SALE" text.   
D1\_Not\_Primary\_Focus: The target product must be the main visual focus. It should occupy a substantial portion of the image, be centrally or   
prominently positioned, and stand out from the surroundings. Visibility alone is not sufficient. If the image is best described as a   
room, setting, or scene with the product present, the product is not the main focus.   
Examples: Pass - product centered, large, and prominent, such as a sofa occupying most of the image. Fail - product is small, peripheral, or   
part of a wide scene, such as a light fixture in a porch photo or a chair at the edge of a room scene.   
Examples: Pass - folded towels/sheets; curtains fully closed; styled bookcase. Fail - wardrobe doors/drawers open; refrigerator doors open;   
folding chair folded; shed doors open; chandelier lights on.   
D3\_Non\_Standard\_View: The image must not primarily show the side, top, back, close-up, or detail view. Multiple views of one product in a   
single image are not acceptable. Front-facing or slightly angled front views are acceptable when they represent the overall product.   
Examples: Pass - sofa shown from front or slight angle, fully visible. Fail - sofa shown from the back only; table shown from top only;   
close-up of armrest/leg; multiple front/back/side views in one image.   
D4\_Dimensional\_Diagram: The image must not contain dimensions, schematics, measurement labels, arrows, or similar diagrams.   
Examples: Pass - clean product image with no added dimensions. Fail - product photo with measurement or diagram overlay.   
D5\_Non\_Photorealistic: The image must be a clean, high-quality photograph or photorealistic 3D render. Reject images with obvious pasted   
props, floating accessories, insets, collages, montages, overlays, composited objects, or unrelated floating items. High-quality 3D   
renders are acceptable when indistinguishable from photos.   
Examples: Pass - well-lit product photograph; high-quality photorealistic render. Fail - floating product covers/accessories; multiple   
cropped product views in one frame; computer-generated object floating above the product   
D6\_Features\_Person: The image must not contain a real person.   
Examples: Pass - no people present; printed portrait or illustration on wall art or pillow as part of the product design. Fail - person   
sitting on sofa/chair; child sitting on bed.   
Final eligibility checklist:   
- List alī failed rules.   
- If any rule fails, the image is not lead eligible.   
- If information is contradictory or insufficient, mark lead\_image\_eligible as "Unsure".   
Confidence scoring:   
- eligibility\_confidence\_score must be an integer from 1 to 5.   
5 = high confidence; all rules and product information are complete, clear, and unambiguous.   
4 = high confidence; mostly clear with minor ambiguity that does not affect the decision.   
3 = low confidence; some ambiguity or incomplete information, but a cautious decision is possible.   
2 = low confidence; significant ambiguity or gaps; decision is a limited-evidence guess.   
1 = low confidence; information is highly contradictory, mostly missing, or nearly impossible to decide.   
eligibility\_confidence\_reason must be one of:   
- High Confidence: reasoning is fully supported and information is present.   
- Low Confidence: a major part of the reasoning is unclear or ambiguous.   
Insufficient Information: not enough data to make a strong decision.   
Contradictory Information: image and product text conflict   
Rule Ambiguous: a rule is open to multiple interpretations.   
Output format: Return valid JSON only   
"structured\_reasoning": {   
"rules":   
"E1\_Front\_Visible\_Enough": { "status": "Pass" "Fail", "reasoning":   
"E2\_Standard\_State": { "status": "Pass" I "Fail", "reasoning": 3,   
"E3\_Correct\_DSQ\_Set": {"status": "Pass" "Fail" "Cannot Verify Exception", "reasoning": "..." },   
"E4\_Props\_Appropriate": { "status": "Pass" | "Fail", "reasoning":

"E5\_No\_Text\_In\_Image": { "status": "Pass" | "Fail", "reasoning": "..." },   
"D1\_Not\_Primary\_Focus": { "status": "Pass" | "Fail", "reasoning": "..." },   
"D2\_Shows\_Functionality": { "status": "Pass" | "Fail" | "Cannot Verify Exception", "reasoning": "..." },   
"D5\_Non\_Photorealistic": { "status": "Pass" | "Fail", "reasoning": "..." },   
"D6\_Features\_Person": { "status": "Pass" | "Fail" | "Cannot Verify Exception", "reasoning": "..." }   
}   
},   
"lead\_image\_eligible": "True" |"False" | "Unsure",   
"eligibility\_confidence\_score": int,   
"eligibility\_confidence\_reason": "High Confidence" |"Low Confidence"| "Insufficient Information"| "Contradictory Information" | "Rule   
Ambiguous",   
"eligibility\_correction\_suggestion": "..."  null,   
"notes": []

## Product Metadata Template

You are evaluating a lead image for the following product:

\## \*\*Product Details\*\*

\- \*\*DSQ (Display Selling Quantity):\*\* [dsq]

\- \*\*Set Information:\*\* [set information]

[image]