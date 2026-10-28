# Decision Models as Judges for Hallucination Detection

Haozheng Liu<sup>1†</sup>, Zhaoqian Xue<sup>2</sup>, Zhongyin Wang<sup>3</sup>, Junting Mu<sup>4</sup>, Zihui Ma<sup>5,6</sup>, Shuya Feng<sup>7</sup>, Lingyao Li<sup>2†</sup>

<sup>1</sup>New York University, <sup>2</sup>University of Arizona,

<sup>3</sup>University of Illinois Urbana-Champaign, <sup>4</sup>University of California, Irvine,

<sup>5</sup>Emory University, <sup>6</sup>University of Maryland, College Park,

<sup>7</sup>University of Alabama at Birmingham

## Abstract

Large language models are widely used as judges of model outputs (LLM-as-a-Judge), but they are costly and can hallucinate or be overconfident. Decision models are a potential alternative. They return a predefined verdict with a probability for each label instead of free-form text and are much cheaper and faster. We test this promise on hallucination detection. We build binary benchmarks for five hallucination types (9,852 items from six datasets) and compare five decision models with the generative judge gpt-6-sol. Performance depends strongly on the type: the best decision model matches gpt-6-sol in balanced accuracy on context inconsistency and comes within 3 percentage points on factual errors, but trails it by 4–8 points on logical and instruction inconsistency. Decision-model probabilities are overconfident, though less so than the GPT-6 models, and separate correct from incorrect decisions only moderately. We also find that a confidence cascade of decision models improves on the best single model by 1.9–3.3 points on three types and matches or exceeds gpt-6-sol at 1–3% of its cost. Overall, decision models are a practical low-cost judge for some hallucination types but need to be validated before use.

## 1 Introduction

Large language models (LLMs) are widely used as judges of model outputs, an approach often called LLM-as-a-Judge (Gu et al., 2026; Zheng et al., 2023). They rate open-ended responses (Verga et al., 2024; Badshah et al., 2025), check whether responses follow instructions (Wen et al., 2026; Zeng et al., 2024), and detect hallucinated content (Li et al., 2023; Tang et al., 2024; Xu et al., 2026; Manakul et al., 2023). However, LLM judges are not fully reliable themselves. They can hallucinate, produce incorrect or unsupported judgments, and be overconfident in their judgments (Kadavath et al., 2022; Tian et al., 2023).

Recently released decision models offer a different kind ofjudge. Models such asjev (TypeSafe AI, 2026), d1 (AI, 2026), and Perplexity Decider (Bespoke Labs and Sathiamoorthy, 2026) do not generate free-form assessments but choose among predefined verdicts and return a probability for each. This design has two potential advantages. First, providers present these probabilities as a measure of certainty, so reliable probabilities can show when a judgment should not be trusted (TypeSafe AI, 2026). Second, decision models are much cheaper and faster than generative LLMs, which matters for large-scale evaluation (Li et al., 2026; Rao and Callison-Burch, 2026; Ibrahim and Zaki, 2026).

Whether these advantages hold is still unclear. Recent studies of decision models as judges each examine a single model (Li et al., 2026; Rao and Callison-Burch, 2026), leaving several questions open. First, judge performance varies across tasks (Xu et al., 2025), but how consistently decision models perform across error types is unknown. Second, whether their probabilities are calibrated and separate correct judgments from errors has merely been tested (Xie et al., 2025). Third, multijudge studies (Verga et al., 2024; Ma et al., 2025) have not tested ensembles composed solely of decision models for hallucination detection.

We address these gaps in the context of hallucination detection. First, judge reliability matters most here, since a judge that hallucinates while checking for hallucinations defeats its purpose. Second, hallucination types require different evidence (world knowledge, sources, the response itself, or the prompt), so they test whether decision models can generalize. We ask three research questions:

• RQ1. How accurately do decision models detect different hallucination types?

• RQ2. How reliable is their confidence?

• RQ3. Can ensembles of decision models improve detection accuracy?

We evaluate five decision models against the generative LLM gpt-6-sol on 9,852 samples from six datasets based on five hallucination types (Huang et al., 2025). Our results show that decision models can match gpt-6-sol on some hallucination types at a small fraction of its cost, and that most of them are only mildly overconfident. Confidence cascades of decision models further improve the accuracy of hallucination detection. Overall, our results show that decision models are a practical low-cost judge for some hallucination types but should be validated before use.

## 2 Related Work

LLM-as-a-Judge and Decision Models. LLMas-a-Judge is widely used for automated evaluation. LLM judges can align with human evaluations (Min et al., 2023) but also show biases such as position and verbosity bias (Zheng et al., 2023; Liu et al., 2023), and because they generate text, they add cost and latency. In contrast, decision models provide a non-generative alternative by directly selecting from predefined labels and returning their associated probabilities. Recent work evaluates jev on HaluEval factuality and final-answer adjudication (Li et al., 2026), as well as on rubric criteria (Rao and Callison-Burch, 2026), and reports lower cost and latency than generative LLM judges. Ibrahim and Zaki (2026) compare several decision models on social-science annotation.

Hallucination detection. Detecting a hallucination requires evidence depending on its type (Huang et al., 2025). FAVABench and HADTest annotate factual errors (Mishra et al., 2024; Xu et al., 2026), RAGTruth annotates unsupported or contradictory content in retrieval-grounded responses (Niu et al., 2024), ContraDoc and ROSCOE cover contradictions within documents and reasoning chains (Li et al., 2024; Golovneva et al., 2023), and IF-RewardBench covers instruction following (Wen et al., 2026). AggreFact shows that factuality detector performance varies across datasets, summarizers, and error types (Tang et al., 2023).

Confidence reliability. Returning probabilities does not make a judge’s confidence reliable. Calibration asks whether probabilities match observed frequencies (Guo et al., 2017), and error ranking asks whether confidence separates correct judgments from errors. Language-model confidence can reflect correctness, but this depends on the task and on how confidence is elicited (Kadavath et al., 2022; Tian et al., 2023; Xiong et al., 2024). For decision models, calibration varies by workload (Li et al., 2026; Rao and Callison-Burch, 2026), and high confidence can coexist with near-chance accuracy (Ibrahim and Zaki, 2026).

Judge ensembles and cascades. Combining judges helps when their errors differ. PoLL aggregates judgments from diverse generative models (Verga et al., 2024), and CLEV calls a third judge only when the first two disagree (Badshah et al., 2025), the idea behind our arbitration method. Confidence-based cascades pass uncertain items to another judge (Jung et al., 2025). JEV-togenerative cascades can match the generative judge at lower cost (Li et al., 2026), although overlapping errors limit their gains on rubric tasks (Rao and Callison-Burch, 2026).

Three areas remain underexplored. First, decision models have so far been evaluated one at a time or on tasks other than hallucination detection, so it is unclear how well they detect each hallucination type. Second, few studies have examined whether the confidence of decision models is reliable across models and hallucination types. Third, it has not been tested whether ensembles built only from decision models can improve hallucination detection while keeping their cost low. Our study aims to address these research gaps.

## 3 Data and Methods

## 3.1 Benchmark Composition

Hallucination types. Hallucination takes several forms. Following the taxonomy of Huang et al. (2025), we study five types (Table 1). Factual contradiction (FC) covers claims that contradict verifiable real-world facts. Factual fabrication (FF) covers unverifiable or unsupported overclaims. Context inconsistency (CI) covers claims that conflict with the supplied source material or are not supported by it. Logical inconsistency (LI) covers contradictions within a response or between its reasoning steps. Instruction inconsistency (II) covers violations of explicit instructions. We treat each type as a separate binary task, because one response can contain several types of error.

Benchmark datasets. We choose six benchmarks whose annotations mark the target error type explicitly, so each sample maps to a binary label without new annotation (Tang et al., 2023, 2024). For

<table><tr><td>Hallucination type</td><td>Operational definition</td><td>Datasets</td><td>H</td><td>N</td><td>Total</td></tr><tr><td>Factual contradiction (FC)</td><td>Claims contradict verifiable real-world facts through incorrect entities or relations.</td><td>FAVABench, HADTest</td><td>344</td><td>1,277</td><td>1,621</td></tr><tr><td>Factual fabrication (FF)</td><td>Claims introduce fabricated, unverifiable, or unsupported facts or overclaims not established by real-world knowledge.</td><td>FAVABench, HADTest</td><td>339</td><td>1,277</td><td>1,616</td></tr><tr><td>Context inconsistency (CI)</td><td>Claims conflict with, or are unsupported by, the supplied context or retrieved evidence.</td><td>RAGTruth</td><td>943</td><td>1,732</td><td>2,675</td></tr><tr><td>Logical inconsistency (LI)</td><td>Claims or reasoning steps within the same response are mutually inconsistent.</td><td>ContraDoc, ROSCOE</td><td>538</td><td>1,108</td><td>1,646</td></tr><tr><td>Instruction inconsistency (II)</td><td>The response violates or fails to satisfy explicit instructions or constraints.</td><td>IF-RewardBench (English)</td><td>1,600</td><td>694</td><td>2,294</td></tr><tr><td colspan="6">Total</td></tr></table>

Table 1: Hallucination-detection tasks and datasets. H marks samples that contain the hallucination and N do not.

![](images/c5fb06f71d9abe7134a01be3e7102f283a9d1457f87bf7c94ff686b0a3118fcd.jpg)  
Figure 1: The single-model baseline and the three combination rules. Models 1–3: top-3 decision models by BAcc per type. Solid/dashed boxes: always/conditionally called. (c) The third model breaks ties between the two cheaper ones. (d) Each stage defers its least-confident $r _ { k }$ share; the last stage decides the rest.

FC and FF, we combine two datasets that annotate both types, given that FAVABench alone has fewer than 400 samples per type. FAVABench contains ChatGPT and Llama responses with human span-level error annotations (Mishra et al., 2024). HADTest contains responses into which GPT-4o injected a single factual error, validated by two annotators (Xu et al., 2026). For CI, RAGTruth contains responses from six LLMs to question answering, summarization, and data-to-text tasks, annotated for content that conflicts with the retrieved source or is unsupported by it (Niu et al., 2024). For II, IF-RewardBench contains responses from 16 LLMs, each checked against a list of explicit constraints (Wen et al., 2026). For LI, ContraDoc contains news, Wikipedia, and story documents with inserted self-contradictions (Li et al., 2024), and ROSCOE contains step-by-step reasoning annotated for contradictions (Golovneva et al., 2023).

An item is labeled H if its source annotation marks the target error and N if it marks the absence of that hallucination error. An N sample may therefore still contain another type of hallucination. We keep each dataset’s evaluation unit and do not rebalance the classes. Appendix A.3 details how we map each dataset’s original annotations to the binary labels used in our evaluation.

## 3.2 Evaluation Setup and Metrics

Model selection. We select five decision models from five providers, including both proprietary and open-weight models. They are jev, d1, clef, pplx, and gpt-6-luna. Details of these models are presented in Appendix B. For comparison, we also run the general-purpose generative LLM gpt-6-sol on the same items, distinguishing it from gpt-6-luna, which is included as a decision model because it directly outputs classification decisions. All six models are accessed through their hosted APIs. Latency is therefore end-to-end API response time and does not depend on local hardware.

Model input and output. For each hallucination type, all models receive the same input and the same $H / N$ criterion, based on the type’s definition. The input consists of the response to be judged and, when available, the original instruction. For CI, it also includes the source context. The models never see the gold labels or the annotated error spans. Each model answers H if the response contains the target error and N otherwise, and returns probabilities $P ( H )$ and $P ( N )$ that sum to one. The answer is the label with the higher probability, and on an exact tie we keep the model’s own answer. The decision models return these probabilities through their interfaces. For $g p t - 6 - s o l .$ we ask for a structured verdict with its own probability estimates and no explanation. Appendix A.4 provides the complete decision-model prompts.

Evaluation metrics. The classes are imbalanced (Table 1), so we compare each answer with the gold label and report balanced accuracy (BAcc), the mean recall over H and N, and macro-F1, the mean F1 over the two classes. Differences between judges are tested with paired bootstrap over items (2,000 resamples) (Dror et al., 2018). We also report the mean API latency per call and the estimated API cost. Cost is computed from recorded token usage at the providers’ list prices at the time of evaluation.<sup>1</sup>

## 3.3 Confidence Analysis

Probabilities and confidence. For each item $i ,$ a decision model returns $p _ { i } = P ( \mathrm { H } \mid x _ { i } )$ , the probability that the response contains the target hallucination type, and labels the response as hallucinated $( \hat { y } _ { i } = 1 )$ when $p _ { i } > 0 . 5$ . Providers describe $p _ { i }$ as an estimate of the probability that the predicate holds or as a calibrated score. The confidence of a decision is the probability of its predicted label,

$$
c _ { i } = \operatorname* { m a x } ( p _ { i } , 1 - p _ { i } ) \in [ 0 . 5 , 1 ] .\tag{1}
$$

Under perfect calibration, predictions assigned confidence $q$ would be correct approximately q of the time. We evaluate the extent to which this property holds in practice.

Detection and calibration. Detection performance, i.e., how well $p _ { i }$ ranks hallucinated above non-hallucinated responses, is measured by AU-ROC. ${ \bf A t } \ \tau = 0 . 5$ , we also report BAcc, MCC and the flag-rate bias $\Delta$ (defined as the share of responses flagged minus $\pi \left( \Delta \right) > $ 0: over-flagging). Calibration is measured by the expected calibration error over $M = 1 0$ equal-width confidence bins (Guo et al., 2017; Roelofs et al., 2022):

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { n } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right| .\tag{2}
$$

where $B _ { m }$ is the set of predictions in confidence bin m. We also report the slope b of the logistic model logit $\operatorname* { P r } ( y _ { i } { = } 1 ) = a + b \log \mathrm { i } { ( p _ { i } ) }$ , where $b < 1$ indicates overconfidence (Calster, 2019).

Confidence as an error signal. A useful confidence score is expected to be higher for correct decisions (C) than for errors (E) (Hendrycks and Gimpel, 2016; Geifman and El-Yaniv, 2017):

$$
\mathrm { c o n f - A U R O C } = \operatorname* { P r } ( c _ { i } > c _ { j } \mid i \in \mathcal { C } , j \in \mathcal { E } ) ,\tag{3}
$$

with ties counted as one half. A value of 0.5 means that confidence cannot distinguish correct decisions from errors. We also report the share of errors made with high confidence $( c _ { i } \geq 0 . 9 )$

We report 95% bootstrap intervals and compare models on the same items with paired bootstrap tests. A two-way ANOVA $( \eta ^ { 2 } )$ separates cell and model effects, and Kendall’s W measures ranking consistency across cells.

## 3.4 Model Ensembling Strategies

Motivated by prior studies (Verga et al., 2024; Badshah et al., 2025; Jung et al., 2025), we evaluate three ensembling methods (Figure 1) to assess whether they improve accuracy and at what additional cost. To isolate the effect of each method, we fix the model selection. For each type, all three methods combine the same three decision models, namely those with the highest BAcc on that type.

Majority voting. All three models judge the same item, and the label chosen by at least two of them is the verdict. Because every item gets all three calls, majority voting is the most expensive of the three methods. Voting gives every model the same weight, so two weaker models can outvote the strongest one.

Arbitration. When two models agree, the outcome is their shared label, and the third vote does not change it. When they disagree, the third vote decides the outcome. Arbitration therefore calls the two models first and calls the third only to resolve a disagreement (Badshah et al., 2025).

Confidence cascade. A cascade lets the strongest model decide unless it is uncertain. Cascades usually order models from cheap to expensive to reduce calls to a costly final model (Chen et al., 2024; Jung et al., 2025). We order the models by descending BAcc and ask whether consulting weaker models on the strongest model’s least confident items improves BAcc (Jitkrittum et al., 2023). At each of the first two stages, a model defers an item to the next model when its confidence $c _ { i }$ falls below a threshold, and the third model decides all items that reach it. Because confidence distributions differ across models (Section 4.2), a common threshold would defer very different shares of items at different stages. We therefore set each threshold through a target deferral rate $r _ { k } .$ , the share of items reaching stage k that are deferred, so that each stage defers its least confident items without using gold labels. We evaluate $r _ { 1 } , r _ { 2 } \in \{ 0 , 0 . 1 , \ldots , 0 . 5 \}$ $r _ { 1 } = 0$ reduces the cascade to the strongest model, and above 0.5 the strongest model would decide fewer than half of the items. As a control, we defer the same numbers of randomly chosen items.

Evaluation. For each hallucination type, we evaluate all 6 × 6 combinations of $r _ { 1 }$ and r<sub>2</sub> and report the one with the highest BAcc. Because we choose this combination on the same items we evaluate, the result is an optimistic estimate of what a cascade can achieve(Cawley and Talbot, 2010). To see how optimistic it is, we also choose the rates with five-fold cross-fitting, picking them on four folds and testing them on the fifth. The cost of an item is the total cost of all models called for it. For latency, we add up calls made one after another and take the slowest of calls made in parallel. We plot each method’s BAcc against its cost, show the results of all 36 combinations separately, and use paired bootstrap tests to compare each method with its best single model and with gpt-6-sol.

## 4 Results

## 4.1 RQ1: How Do Decision Models Perform Across Hallucination Types?

Table 2 and Figure 2 present the performance over each hallucination type. Appendix C reports their performance judge on every source dataset.

Performance depends strongly on the type. gpt-6-sol has the highest pooled BAcc on FC, LI, and II (Table 2). On FF, gpt-6-luna (82.7) edges it (82.0). On CI, pplx, gpt-6-sol, and jev are within 0.2 points of one another at about 82.6. The gap is widest on II, where gpt-6-sol reaches 78.9 against 71.1 for pplx and 61.8 for jev. Performance also varies within a model. jev spans 20.8 points between II and CI. Type and source dataset are confounded, so we read these differences as typespecific without attributing them to the type alone.

<table><tr><td colspan="2">Type (N) Model</td><td>BAcc Macro-F1</td><td></td><td></td><td>Cost Latency</td></tr><tr><td rowspan="5">FC N=1,621</td><td>jev</td><td>75.01</td><td>74.96</td><td>0.0490</td><td>264.51</td></tr><tr><td>d1</td><td>67.78</td><td>68.98</td><td>0.0295</td><td>224.15</td></tr><tr><td>clef</td><td>72.12</td><td>73.95</td><td>0.2064</td><td>347.49</td></tr><tr><td>pplx</td><td>78.93</td><td>78.48</td><td>0.0159</td><td>284.23</td></tr><tr><td>gpt-6-luna</td><td>77.75</td><td>68.36</td><td>0.0817</td><td>236.82</td></tr><tr><td rowspan="6">FF N=1,616</td><td>gpt-6-sol</td><td>81.61</td><td>77.45</td><td>3.3320</td><td>1,825.76</td></tr><tr><td>jev</td><td>80.74</td><td>74.11</td><td>0.0570</td><td>239.63</td></tr><tr><td>d1</td><td>70.51</td><td>74.40</td><td>0.0369</td><td>235.40</td></tr><tr><td>clef</td><td>71.24</td><td>75.02</td><td>0.2498</td><td>354.56</td></tr><tr><td>pplx</td><td>79.85</td><td>82.14</td><td>0.0195</td><td>290.10</td></tr><tr><td>gpt-6-luna gpt-6-sol</td><td>82.72 82.05</td><td>75.15 81.36</td><td>0.0999</td><td>236.64 3.50141,639.48</td></tr><tr><td rowspan="6">CI N=2,675</td><td>jev</td><td></td><td></td><td></td><td></td></tr><tr><td>d1</td><td>82.53</td><td>82.09</td><td>0.2214</td><td>281.30</td></tr><tr><td>clef</td><td>75.84 68.21</td><td>77.33</td><td>0.1797</td><td>319.34</td></tr><tr><td>pplx</td><td>82.66</td><td>69.38 84.55</td><td>1.0911</td><td>622.86</td></tr><tr><td>gpt-6-luna</td><td></td><td></td><td>0.0901</td><td>342.02</td></tr><tr><td>gpt-6-sol</td><td>64.04 82.65</td><td>53.01 78.94</td><td>0.4375 10.9699</td><td>258.33 2,002.29</td></tr><tr><td rowspan="6">LI N=1,646</td><td>jev</td><td>68.98</td><td>70.47</td><td></td><td></td></tr><tr><td>d1</td><td></td><td>56.95</td><td>0.0733</td><td>284.50 330.27</td></tr><tr><td>clef</td><td>58.46 61.33</td><td>61.42</td><td>0.0509 0.3398</td><td>515.09</td></tr><tr><td>pplx</td><td>73.28</td><td>75.12</td><td>0.0270</td><td>284.83</td></tr><tr><td>gpt-6-luna</td><td>66.99</td><td>68.27</td><td>0.1356</td><td>281.61</td></tr><tr><td>gpt-6-sol</td><td>77.14</td><td>78.32</td><td>4.4831</td><td>2,203.56</td></tr><tr><td rowspan="6">II N=2,294</td><td>jev</td><td>61.75</td><td>62.03</td><td>0.1403</td><td>276.94</td></tr><tr><td>d1</td><td>62.50</td><td>62.02</td><td>0.1114</td><td>478.96</td></tr><tr><td>clef</td><td>64.50</td><td>64.43</td><td>0.7021</td><td>593.57</td></tr><tr><td>pplx</td><td>71.08</td><td>65.39</td><td>0.0566</td><td>306.38</td></tr><tr><td>gpt-6-luna</td><td>54.17</td><td>50.69</td><td>0.2846</td><td>263.23</td></tr><tr><td>gpt-6-sol</td><td>78.88</td><td>78.43</td><td></td><td>8.50982,644.51</td></tr></table>

Table 2: Single-judge results pooled by hallucination type. BAcc and Macro-F1 are percentages; Cost is the estimated total API cost for that type (USD), and Latency is mean per-call latency (ms).

Similar scores hide different errors. The matched CI scores arise from different error patterns. pplx rarely raises false alarms (false-positive rate 3.2%) but misses many hallucinations (falsenegative rate 31.5%). gpt-6-sol shows the reverse pattern, with a false-negative rate of 5.9% and a false-positive rate of 28.8%. jev lies between them at 20.6% and 14.4%. The dominant error also changes with the task. jev misses most positives on LI, with false-negative rates of 49% on ContraDoc and 81% on ROSCOE, but over-flags on II, where its false-positive rate is 56%. BAcc alone therefore does not show whether a judge mostly misses hallucinations or mostly raises false alarms.

![](images/a352987ab63bba234b96d032e98f154dffd53bebc24c272f529a378f26bf019b.jpg)  
Figure 2: Judging performance by hallucination type. Bars show the pooled BAcc of each judge per type, with 95% bootstrap confidence intervals. Hatched bars mark the generative model gpt-6-sol.

Model advantages are concentrated in particular types. Relative to jev, pplx gains only 0.1 points on CI (95% paired bootstrap interval [−1.4, 1.6]) and loses 0.9 points on FF ([−3.4, 1.6]), but it gains 3.9 points on FC ([1.5, 6.4]), 4.3 points on LI ([2.2, 6.5]), and 9.3 points on II ([7.2, 11.5]). The FC, LI, and II gains remain significant after Holm correction over the 25 comparisons of each judge with jev. The ranking of decision models thus changes with the type.

Advantages do not always hold across sources. On HADTest, all six judges score 12 to 18 points higher on FF than on FC. The same comparison on FAVABench shows no consistent gap (Appendix C). For pplx, FF minus FC is −1.4 points on FAV-ABench (95% bootstrap interval [−8.5, 5.3]) but 17.9 points on HADTest ([13.2, 22.9]). An advantage on fabrication is therefore partly a property of the source. CI and II each have a single source, so their cross-source stability remains untested.

Decision models are far cheaper and faster. Per item, the decision models cost between 0.5% (pplx) and 10% (clef) of gpt-6-sol, and they answer in 0.22 to 0.62 seconds against 1.6 to 2.6 seconds (Table 2). The cheapest model, pplx, is also the most accurate decision model on four of the five types, and gpt-6-luna is the most accurate on FF. The efficiency gain is thus available on every type. Whether it comes with the accuracy of gpt-6-sol depends on the type. Additional analyses of error patterns and prediction overlap across models are provided in Appendix D.2.

## 4.2 RQ2: How Reliable is Their Confidence?

Figures 3 and 4 present the confidence reliability of the six judges across hallucination types, including calibration and the ability to distinguish correct judgments from errors.

(a) Overconfidence gap  
![](images/18885d99b93229a6efed8d53b3483b337a8e06354bed9cfbfce7184a1156534a.jpg)  
(b) conf-AUROC

![](images/dccc9822ac1b96a80f367fe0409b34d2b4ebf890e4b3ceed6ac50ba636d976b5.jpg)  
Figure 3: Confidence by model and hallucination type. (a) Overconfidence gap, i.e., mean confidence minus accuracy. (b) conf-AUROC; outlined values have a 95% CI that includes 0.5.

Decision models are overconfident overall, but less so than the GPT-6 models. Every decision model has a positive average overconfidence gap, and the calibration slope is below 1 for 21 of the 25 model–type pairs (Figure 3a). For jev, pplx, clef and d1, however, the overconfidence is mild. Mean confidence exceeds accuracy by 0.04–0.12, and ECE ranges from 0.083 to 0.136. The two GPT-6 models are clearly more overconfident. gpt-6-luna has a gap of 0.18 and an ECE of 0.187, and the comparison LLM gpt-6-sol has a gap of 0.16 and an ECE of 0.157. Their median calibration slopes (0.44 and 0.36) are also the lowest, compared with 0.54–0.58 for jev, d1 and pplx and 0.96 for clef.

Calibration depends more on the dataset than on the model. Across the decision models, the dataset accounts for 58% of the variance in the overconfidence gap, and the model for 15% (Figure 3a). Averaged over the decision models, overconfidence is largest for II (0.14) and LI (0.13), and smallest for CI and factual contradiction FC (0.07 each). The reliability diagrams show the same pattern. At a given confidence level, accuracy differs by hallucination type (Figure 4). For example, the accuracy of gpt-6-luna on CI stays below 0.50 for all confidence levels between 0.55 and 0.95.

![](images/927917ea164eb8e1c43c1d586bcd3978930191c66a51785d918d48d2ea129091.jpg)  
Figure 4: Reliability diagrams by hallucination type. Confidence is divided into 10 equal-width bins (bins with fewer than 25 items omitted); points below the diagonal indicate overconfidence. Right: ECE and conf-AUROC (cAUC) averaged over datasets, and the share of all errors (pooled over datasets) made with $c \geq 0 . 9 \ ( \mathrm { E r r } { \geq } . 9 )$ .

Confidence is less reliable for instruction inconsistency (II) and logical inconsistency (LI). Averaged over the decision models, II has the largest overconfidence gap (0.14), the highest ECE (0.145) and the lowest conf-AUROC (0.64), followed by LI (gap 0.13; ECE 0.136; conf-AUROC 0.70). FC is less overconfident (gap 0.07; ECE 0.103) but separates correct from incorrect decisions no better than LI (conf-AUROC 0.70), whereas CI (gap 0.07; ECE 0.093; conf-AUROC 0.74) and FF(gap 0.08; ECE 0.134; conf-AUROC 0.75) are the most reliable. Figure 3 shows the same pattern. Among decisions made with $c \geq 0 . 9$ accuracy is lowest for II for jev and pplx, and for LI for clef and d1 (Figure 4). Type-specific failures also occur for individual models. gpt-6-luna is most overconfident on CI (gap 0.31), where its accuracy stays below 0.50.

Confidence separates correct from incorrect decisions only moderately. conf-AUROC measures how well confidence separates correct judgments from errors. A value of 0.5 indicates performance no better than chance. Higher values indicate better separation between the two groups (Figure 3b). Averaged over the five hallucination types, conf-AUROC ranges from 0.67 (gpt-6-luna) to 0.73 (pplx and clef). For each model and type, the 95% confidence interval lies above 0.5, so confidence always carries some information about correctness, but the separation remains moderate, especially for II.

High confidence does not guarantee a correct decision. Errors can be made with high confidence $( c \ge 0 . 9 )$ (Figures 4 and 7). Averaged over the five types, gpt-6-luna assigns $c \geq 0 . 9$ to 65% of its decisions, is correct on only 77% of them, and makes 47% of its errors at this level. For the other decision models, 10–30% of errors are made at $c \ge$ 0.9, with clef lowest (10%); clef also assigns such high confidence to only 25% of its decisions. The comparison model gpt-6-sol is the most extreme case: it assigns $c \geq 0 . 9$ to 88% of its decisions and makes 69% of its errors at this level.

## 4.3 RQ3: Can Ensembling Improve the Performance?

Figure 5 presents the performance–cost trade-offs of the three ensembling methods. Deferral-rate sensitivity is reported in Appendix Figure 8.

The cascade matches or exceeds GPT-6-sol on three types. At the best tested deferral rates, the confidence cascade achieves BAcc scores of 82.2, 84.6, and 84.6 on FC, FF, and CI, respectively—0.6, 2.6, and 1.9 points higher than gpt-6-sol—at only 1.2–3.3% of its estimated cost. The advantage over gpt-6-sol is statistically significant on FF and CI, but not on FC. The cascade also improves on the strongest single member by 3.3, 1.9, and 2.0 points, respectively, with paired bootstrap 95% intervals excluding zero. It makes 1.1–1.3 calls per item and has about one fifth of the latency of gpt-6- sol. Selecting deferral rates by cross-fitting reduces BAcc by at most 0.52 points and preserves the gains over the strongest member.

![](images/fda2ed186e935d203bc73113c8e007cac2a4ebd4d753d6dfab6f705559ed79c7.jpg)  
Figure 5: BAcc against estimated cost for each hallucination type. Cost is relative to gpt-6-sol on a log scale, and the dashed line marks the BAcc of gpt-6-sol. The cascade uses the deferral rates with the highest BAcc. The circle defers the same numbers of items at random.

The cascade outperforms voting and arbitration. Majority voting gains about one BAcc point over the strongest single model on FC and FF, but falls below it on CI, LI, and II (Figure 5). Disagreement arbitration produces the same verdicts as voting while costing roughly half as much. Both rules are less accurate than the cascade on every type and, except on FF, also cost more. Unlike equal voting, the cascade consults weaker members only when the strongest member is uncertain.

The best cascade configuration differs by type. pplx leads four of the five cascades. On FF, gpt-6- luna leads (Appendix Table 12). The role of the later stages also varies. On CI, the best cascade never uses its third model (r<sub>2</sub> = 0), whereas on FF the second model defers about half of the samples it receives. On FC, FF, and CI, BAcc peaks at a firststage deferral rate of 20–30% and declines at higher rates (Appendix Figure 8). A single cascade configuration therefore does not transfer across types.

The gap persists on LI and II. On LI, the best cascade defers no items and equals pplx alone, 3.9 points below gpt-6-sol. On II, the cascade improves BAcc by only 0.7 points over the strongest single model, pplx, and remains 7.1 points below gpt-6- sol. Other members do not improve on pplx for its least confident items in balanced terms, consistent with the weaker confidence reliability on these two types (Section 4.2).

## 5 Discussion and Conclusion

We evaluate five decision models for hallucination detection. They match or approach the generative LLM judge on factual and context-related errors but lag on logical and instruction inconsistencies requiring multi-step reasoning or explicit constraint checking. We draw three takeaways.

First, judges should be selected per hallucination type. No decision model performs best across types, and similar BAcc can mask opposite error profiles: missed hallucinations or false alarms. The preferable judge thus depends on the target type and on which error is more costly, so false-negative and false-positive rates should be reported by type. Since type and source are partly confounded, judge selection should be validated on the target task.

Second, confidence is useful but does not guarantee correctness. Most decision models are less overconfident than gpt-6-luna and gpt-6-sol, but their confidence only moderately distinguishes correct from incorrect judgments, with frequent highconfidence errors on logical and instruction inconsistencies (Section 4.2). Confidence can guide requests for a second opinion but cannot establish correctness.

Third, ensembling decision models helps when they are complementary on uncertain samples. Cascades outperform their strongest member and match or exceed gpt-6-sol on factual contradiction, factual fabrication, and context inconsistency. On logical and instruction inconsistency, the tested cascades offer limited gains, consistent with the observed error overlap (Appendix D.2). Since the best configuration varies by type, deferral rates should be validated on the target type to weigh accuracy gains against the cost of additional model calls.

At 0.5–10% of the cost of gpt-6-sol, decision models could support low-cost, type-aware cascades for factual and context errors, with LLM judges reserved for logical and instruction inconsistencies. Future work should disentangle type from data source, develop decision models that target reasoning- and instruction-level errors, and learn routing policies that decide when to escalate to an LLM judge based on detection accuracy and cost.

## Limitations

This study presents several future directions.

Benchmark coverage and label consistency. Our benchmark covers five hallucination types across six datasets, but coverage is uneven: context and instruction inconsistency each rely on a single source. Differences in data construction, domain, and response length confound cross-dataset comparisons, leaving subgroup analyses observational. Mapping heterogeneous annotations to binary labels may also introduce mismatches because the source datasets define errors differently. Broader source coverage, annotation under a shared rubric, and comparisons matched on domain and response length would help address these limitations.

Confidence comparability and prompt sensitivity. Confidence values are not fully comparable because decision models return label probabilities, whereas gpt-6-sol reports its own estimates. We also use one prompt design per task, leaving prompt sensitivity untested. Comparing alternative prompts and, where supported, confidenceelicitation methods within each model would clarify their effects on accuracy and calibration.

Generalization and deployment of cascades. We evaluate fixed model versions and assess cascades through offline replay and within-source cross-fitting, with member selection and ordering based on the full sample. Selecting all cascade components using training data alone, followed by evaluation on unseen domains and in live deployments, would better assess their accuracy, latency, and cost beyond the evaluated settings.

## Statement of LLM Use

During the preparation of this paper, we used LLMbased tools (primarily Claude and ChatGPT) in two capacities. First, we used coding agents to assist with the development and refinement of the project codebase. Second, we used LLM chatbots to support language editing of manuscript drafts, including correcting grammatical errors, improving sentence structure, and enhancing the clarity of presentation. The authors take full responsibility for the content of this paper, including the code, analyses, and conclusions presented.

## References

Liquid AI. 2026. Introducing d1: The most capable decision model, now with vision. Liquid AI Blog. Www.liquid.ai/blog/d1-decision-model.

Sher Badshah, Moamen Moustafa, and Hassan Sajjad. 2025. CLEV: LLM-based evaluation through lightweight efficient voting for free-form questionanswering. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Associationfor Computational Linguistics, pages 1513–1531, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Bespoke Labs and Maheswaran Sathiamoorthy. 2026. Nimble. https://github. com/bespokelabsai/nimble. Model: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B.

B Calster. 2019. Calibration: the achilles heel of predictive analytics. BMC Med, 17(1):1.

Gavin C Cawley and Nicola LC Talbot. 2010. On overfitting in model selection and subsequent selection bias in performance evaluation. The Journal ofMachine Learning Research, 11:2079–2107.

Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research. ArXiv:2305.05176.

Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. 2018. The hitchhiker’s guide to testing statistical significance in natural language processing. In Proceedings of the 56th annual meeting of the associationfor computational linguistics (volume 1: Long papers), pages 1383–1392.

Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification for deep neural networks. Advances in neural information processing systems, 30.

Olga Golovneva, Moya Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. 2023. ROSCOE: A suite of metrics for scoring step-by-step reasoning. In The Eleventh International Conference on Learning Representations. ICLR 2023; author version arXiv:2212.07919v2.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Zhouchi Lin, Bowen Zhang, Lionel Ni, Wen Gao, Yuanzhuo Wang, and Jian Guo. 2026. A survey on LLM-as-a-judge. The Innovation, 7(6).

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Con-

ference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321– 1330. PMLR.

Dan Hendrycks and Kevin Gimpel. 2016. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55.

Hazem Ibrahim and Yasir Zaki. 2026. Evaluating decision models for text annotation in computational social science. arXiv preprint arXiv:2609.24574.

Wittawat Jitkrittum, Neha Gupta, Aditya Krishna Menon, Harikrishna Narasimhan, Ankit Singh Rawat, and Sanjiv Kumar. 2023. When does confidencebased cascade deferral suffice?, 2024. URL https://arxiv. org/abs/2307.02764.

Jaehun Jung, Faeze Brahman, and Yejin Choi. 2025. Trust or escalate: LLM judges with provable guarantees for human agreement. In International Conference on Learning Representations, volume 2025, pages 3101–3125.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221. Preprint, arXiv:2207.05221.

Jierui Li, Vipul Raheja, and Dhruv Kumar. 2024. ContraDoc: Understanding self-contradictions in documents with large language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6509–6523, Mexico City, Mexico. Association for Computational Linguistics.

Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023. HaluEval: A largescale hallucination evaluation benchmark for large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6449–6464, Singapore. Association for Computational Linguistics.

Yubo Li, Yidi Miao, Ramayya Krishnan, and Rema Padman. 2026. JEV-as-a-judge: Accept when confident, escalate when unsure. arXiv preprint arXiv:2609.26550. Preprint, arXiv:2609.26550. Version 4, 6 October 2026.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522.

Chiyu Ma, Enpei Zhang, Yilun Zhao, Wenjun Liu, Yaning Jia, Peijun Qing, Lin Shi, Arman Cohan, Yujun Yan, and Soroush Vosoughi. 2025. Judging with many minds: Do more perspectives mean less prejudice? on bias amplification and resistance in multiagent based LLM-as-judge. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 17356–17392. Association for Computational Linguistics.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 9004– 9017.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, pages 12076–12100.

Abhika Mishra, Akari Asai, Vidhisha Balachandran, Yizhong Wang, Yulia Tsvetkov, Graham Neubig, and Hannaneh Hajishirzi. 2024. Fine-grained hallucination detection and editing for language models. In First Conference on Language Modeling.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10862– 10878, Bangkok, Thailand. Association for Computational Linguistics.

Delip Rao and Chris Callison-Burch. 2026. JEV vs. LLMs as rubric judges: Cheaper, faster, and wrong in the same places. arXiv preprint arXiv:2609.29769. Preprint, arXiv:2609.29769. Version 2, 28 September 2026.

Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, and Michael C Mozer. 2022. Mitigating bias in calibration error estimation. In International Conference on Artificial Intelligence and Statistics, pages 4036– 4054. PMLR.

Liyan Tang, Tanya Goyal, Alex Fabbri, Philippe Laban, Jiacheng Xu, Semih Yavuz, Wojciech Kryscinski, Justin Rousseau, and Greg Durrett. 2023. Understanding factual errors in summarization: Errors, summarizers, datasets, error detectors. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor

Computational Linguistics (Volume 1: Long Papers), pages 11626–11644, Toronto, Canada. Association for Computational Linguistics.

Liyan Tang, Philippe Laban, and Greg Durrett. 2024. MiniCheck: Efficient fact-checking of LLMs on grounding documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8818–8847, Miami, Florida, USA. Association for Computational Linguistics.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442, Singapore. Association for Computational Linguistics.

TypeSafe AI. 2026. TypeSafe API. Version 0.2.0. Accessed September 24, 2026.

Pat Verga, Sebastian Hofstätter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. 2024. Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. arXiv preprint arXiv:2404.18796. Preprint, arXiv:2404.18796.

Bosi Wen, Yilin Niu, Cunxiang Wang, Xiaoying Ling, Ying Zhang, Pei Ke, Hongning Wang, and Minlie Huang. 2026. IF-RewardBench: Benchmarking judge models for instruction-following evaluation. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 23816–23843, San Diego, California, United States. Association for Computational Linguistics.

Qiujie Xie, Qingqiu Li, Zhuohao Yu, Yuejie Zhang, Yue Zhang, and Linyi Yang. 2025. An empirical analysis of uncertainty in large language model evaluations. In International Conference on Learning Representations.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can llms express their uncertainty. An empirical evaluation of confidence elicitation in LLMs. arXiv, 2306.

Austin Xu, Srijan Bansal, Yifei Ming, Semih Yavuz, and Shafiq Joty. 2025. Does context matter? ContextualJudgeBench for evaluating LLM-based judges in contextual settings. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9541– 9564. Association for Computational Linguistics.

Fan Xu, Xinyu Hu, Zhenghan Yu, Li Lin, Xu Zhang, Yang Zhang, Wei Zhou, Jinjie Gu, and Xiaojun Wan. 2026. HAD: HAllucination detection language models based on a comprehensive hallucination taxonomy. In Proceedings of the 64th Annual Meeting of

the Associationfor Computational Linguistics (Volume 6: Industry Track), pages 141–158, San Diego, California, USA. Association for Computational Linguistics.

Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. 2024. Evaluating large language models at evaluating instruction following. In International Conference on Learning Representations, volume 2024, pages 40193–40219.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36:46595–46623.

## Appendix Contents

A Benchmark Construction, Labels, and Prompts 12   
A.1 Hallucination Taxonomy and Examples . 12   
A.2 Dataset Versions and Construction 12   
A.3 Binary Label Mapping 13   
A.4 Decision-Model Prompts 14   
B Model Configurations, Cost, and Latency 16   
C Per-Dataset Results 16   
D Error Analysis 16   
D.1 Errors by Source Annotation 16   
D.2 Error Patterns and Overlap 16   
E Confidence Analysis Details 17   
F Cascade Deferral-Rate Sensitivity 18   
F.1 Peak and Cross-Fitted Results . 18

## A Benchmark Construction, Labels, and Prompts

## A.1 Hallucination Taxonomy and Examples

<table><tr><td>Type and definition</td><td>Example dialog</td><td>Annotation</td></tr><tr><td colspan="3">Factuality hallucination</td></tr><tr><td>Factual Contradiction States facts that can be grounded in real-world knowledge but contradict it, through a wrong entity</td><td>Instruction: How does the EU work? Response: The EU also has its own currency, the euro, which is used by 19 member states.</td><td>FAVABench, entity error; the correct number is 20.</td></tr><tr><td>or relation. Factual Fabrication States facts that cannot be verified against established knowledge, or overclaims</td><td>Instruction: Name three important inventions of the 20th century. Response: The three important inventions of the 20th century are the airplane, the automobile, and the chromoflare device, which enabled seamless digital color transitions in electronic displays.</td><td>HADTest, fabricated entity; no such device exists.</td></tr><tr><td colspan="3">without broad support. Faithfulness hallucination</td></tr><tr><td>Context Inconsistency Is unfaithful to the context supplied by the user, such as a retrieved document.</td><td>Retrieved context: Three of the alleged attackers were killed, Israeli media reported, citing IDF sources. It was not immediately known to what militant group the men belonged. Instruction: Summarize the following news within 38 words. Response: Israeli jets struck targets in the occupied Golan Heights after</td><td>RAGTruth, evident conflict; the source reports three deaths.</td></tr><tr><td>Logical Inconsistency Contains internal contradictions, between statements or reasoning</td><td>attackers were killed, but their affiliation remains unclear. Document: The text of this novel of ideas presents itself as a book that has been written as the result of a promise to a dying man. William Porphyry Benham is a man who has lived a life devoted to a complicated, the first. protean idea: “&quot;that he had to live life nobly and thoroughly.&quot; The novel has been written as the result of a bet between two friends.</td><td>ContraDoc; the last sentence contradicts</td></tr><tr><td>and the conclusion. Instruction Inconsistency Deviates from an explicit, non-malicious user instruction.</td><td>Instruction: what concentration of particles is bad for biosensors. The output should not contain any whitespace. Response: Particle concentrations exceeding 1mg/mL or 107particles/mL often cause biosensor interference depending on</td><td>IF-RewardBench; whitespace violates the instruction.</td></tr></table>

Table 3: Hallucination taxonomy with examples from our benchmark. Definitions follow Huang et al. (2025). Each example is an H item; hallucinated content is in bold.  
Table 3 lists the five hallucination types, their definitions, and benchmark examples.

## A.2 Dataset Versions and Construction

Table 4 identifies the source snapshots, evaluation subsets, and resulting type-specific views. Revisions refer to the local source snapshots used to build the benchmark. Where an upstream revision was not retained, a SHA-256 fingerprint identifies the archived source file. Hashes below use the first 12 hexadecimal characters; the accompanying provenance file records the complete hashes and revision identifiers for the archived source snapshots.

<table><tr><td>Dataset</td><td>Version / source snapshot</td><td>Split and selection</td><td>Instances</td></tr><tr><td>FAVABench</td><td>annotations.json; SHA ef3b4b289762</td><td>All 460 human-annotated passages; separate inclusive FC and FF views.</td><td>FC: 377 FF: 351</td></tr><tr><td>HADTest</td><td>hadtest.json; SHA fd13225bcc8d</td><td>Human-validated HADTest split only; HAD train and dev are not used.</td><td>FC: 1,244 FF: 1,265</td></tr><tr><td>RAGTruth</td><td>Revision c103204b9ce2</td><td>Official test split, with quality=&quot;good&quot;; join responses to source context by source_id.</td><td>2,675</td></tr><tr><td>ContraDoc</td><td>Revision d1c42d1e4335</td><td>Entire released pos/neg benchmark; these are label partitions, not train/test splits.</td><td>891</td></tr><tr><td>ROSCOE</td><td>Official annotations.zip; SHA1fe5b4ee56a6</td><td>All released CosmosQA, DROP, e-SNLI, and GSM8K annotations; exclude one empty chain.</td><td>755</td></tr><tr><td>IF-RewardBench</td><td>Revision c192fe968c15</td><td>English-filtered released benchmark, including all three instruction forms.</td><td>2,294</td></tr></table>

Table 4: Dataset versions and evaluation subsets. Counts are type-specific evaluation instances. No class balancing or additional train/test split is applied to these judge evaluations.

Factuality mapping and reuse. For FAV-ABench, entity, relation, and contradictory map to FC; invented, unverifiable, and subjective map to FF, with subjective treated as overclaim. We reconstruct the annotated error-bearing passage by removing proposed corrections inside mark spans and stripping annotation tags while retaining their content, including delete spans. A response carrying both types is positive in both views. Negatives have none of the six factual error tags; responses carrying only the other type are excluded. Thus, 115 jointly positive and 153 jointly negative records are reused across FC and FF.

For HADTest, Factual Recall Error and Factual Inference Error map to FC, while Fabricated Entity and Fictional Attribution map to FF. The 1,124 None-labeled responses are negatives in both views; other categories are excluded. The conversion removes empty responses and conflicting-label duplicate texts, retaining the first occurrence of a same-label duplicate within a view. Reuse across types is intentional because the two tasks ask different questions of the same response.

Faithfulness subsets. RAGTruth is positive when at least one hallucination span is annotated, after the test/good filtering. ContraDoc includes both inserted and replacement-based contradictions. ROSCOE uses the whole-chain 0\_full\_ newClearContradiction\_result label (yes/no);

this annotation also mentions incoherent reasoning, so the view is broader than strict contradiction. Its 756 released rows yield 755 evaluable chains after excluding one empty CosmosQA response. IF-RewardBench’s English filter retains instructions and checklists containing no characters in U+4E00–U+9FFF; it is a preprocessing heuristic, not a separate official language split. The retained responses comprise 984 single-turn, 574 multi-turn, and 736 system-prompt items. Any violated constraint maps to H.

Counting convention. The total of 9,852 counts type-specific evaluation instances, not deduplicated answers. The 268 shared FAVABench records and 1,124 shared HADTest records account for the cross-type reuse. There are 8,460 distinct (dataset, source-record ID) pairs; this is an ID count, not global text deduplication. Judges receive complete responses, documents, or reasoning chains, without gold labels, annotated spans, or corrections.

## A.3 Binary Label Mapping

Figure 6 shows how we map each dataset’s original annotations to binary labels. For each hallucination type, an item is labeled H if its annotations indicate the target error and N if they indicate its absence. FAVABench and HADTest provide annotations for both factual types, so we construct separate FC and FF views from each. In FAVABench, a response is negative only if it carries none of the six factual error tags. Responses tagged only with the other factual type are excluded. In HADTest, Nonelabeled responses serve as negatives for both FC and FF, and we exclude its other error categories. RAGTruth labels a response as H if it contains at least one annotated hallucination span. ContraDoc uses its official split, and ROSCOE uses its whole-chain ClearContradiction label. In IF-RewardBench, a response is labeled H if it violates any annotated constraint. We keep each dataset’s original evaluation unit. Because some annotations target a single error type, N means only that the target error is absent, not that the output is error-free.

<table><tr><td rowspan="2">Type Dataset</td><td rowspan="2"></td><td rowspan="2">Original label → H</td><td rowspan="2">Original label → N</td><td>H</td><td>N</td><td>H:N</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="2">FC</td><td>FAVABench response HADTest</td><td>entity / relation / contradictory</td><td>no factual tag</td><td>224</td><td>153</td><td></td></tr><tr><td>response</td><td>Factual Recall / Inference Error</td><td>None</td><td>120</td><td>1,124</td><td></td></tr><tr><td rowspan="2">FF</td><td>FAVABench response</td><td>invented / unverifiable / subjective</td><td>no factual tag</td><td>198</td><td>153</td><td></td></tr><tr><td>HADTest response</td><td>Fabricated Entity / Fictional Attribution</td><td>None</td><td>141</td><td>1,124</td><td></td></tr><tr><td>Cl</td><td>RAGTruth (test) response</td><td>≥ 1 hallucination span</td><td>no span</td><td>943</td><td>1,732</td><td></td></tr><tr><td rowspan="2">LI</td><td>ContraDoc document</td><td>pos (inserted contradiction)</td><td>neg (original)</td><td>449</td><td>442</td><td></td></tr><tr><td>ROSCOE reasoning chain</td><td>ClearContradiction = yes</td><td>ClearContradiction = no</td><td>89</td><td>666</td><td></td></tr><tr><td>II</td><td>IF-RewardBench response</td><td>any constraint violated</td><td>all constraints met</td><td>1,600</td><td>694</td><td></td></tr></table>

Figure 6: Mapping of source-dataset annotations to binary labels. H: the target error is present; N: it is absent. The bar shows each dataset’s H:N ratio.

## A.4 Decision-Model Prompts

Decision models have no separate system and user prompts. Each request carries a choice question, which is fixed for a hallucination type, and an item state, which is filled for each record. The decision models (jev, d1, clef, and pplx) receive both parts; gpt-6-luna receives the same text through the OpenAI Decisions API, with the state serialized as a JSON string and the H/N criteria given as choice descriptions. Bracketed values are placeholders for the item text. A state field is sent only when it is non-empty. The shared request structure is shown below. The type-specific instructions and criteria are reproduced verbatim below; field headings are added for display, and the state fields retain the original placeholders.

Request structure shared by all types   
{   
"questions": {   
"target\_hallucination": {   
"type": "choice",   
"instructions": "[Instructions]",   
"criteria": {   
"H": "[Criterion H]",   
"N": "[Criterion N]"}}},   
"state": {[State fields]}   
}

## Prompt for factual contradiction (FC)

Instructions: Review the entire candidate response. Determine whether it contains factual contradiction. Choose H if at least one factual claim satisfies the H criterion. Otherwise choose N. Evaluate only factual contradiction.

Criterion H: At least one factual claim conflicts with reliable, verifiable real-world information. This includes an incorrect entity, relation, date, number, location, attribute, event, or other factual statement. Reliable evidence must establish that the claim is false.

Criterion N: No factual claim conflicts with reliable, verifiable real-world information. A claim that is merely unsupported, uncertain, subjective, or difficult to verify does not count as factual contradiction.

## State:

candidate\_response: "[the model-generated answer being judged]";

instruction: "[the original user question or prompt]"

## Prompt for factual fabrication (FF)

Instructions: Review the entire candidate response. Determine whether it contains factual fabrication. Choose H if at least one factual claim satisfies the H criterion. Otherwise choose N. Evaluate only factual fabrication.

Criterion H: At least one statement is presented as established fact and satisfies at least one of the following conditions: (1) Unverifiability: it presents an invented or nonexistent entity, event, source, study, quotation, or other concrete factual detail as established fact without a reliable basis in real-world knowledge; or makes a concrete factual claim that cannot be substantiated by available reliable sources; (2) Overclaim: it presents a subjective, disputed, speculative, or exaggerated assertion as a broadly established fact despite lacking substantial supporting evidence or broad consensus. An incorrect entity, identity, attribute, relation, date, or number is not factual fabrication if the error solely contradicts an established fact.

Criterion N: The response contains no claim that is entirely nonexistent, unverifiable using available reliable sources, or an overclaim lacking broad consensus or substantial evidence. Choose N when all factual claims are verifiable, or when an error consists only of an incorrect entity substitution or relation that can be directly checked against established real-world facts.

## State:

candidate\_response: "[the model-generated answer being judged]";

instruction: "[the original user question or prompt]"

## Prompt for context inconsistency (CI)

Instructions: Review the entire candidate response against the supplied context. Determine whether the response asserts at least one substantive factual claim that is inconsistent with or unsupported by that context. Choose H only if such a claim satisfies the H criterion. Otherwise choose N. Use the supplied context as the evidence source, and evaluate only contextual faithfulness rather than formatting or general instruction compliance.

Criterion H: At least one substantive factual claim in the candidate response contradicts the supplied context or introduces information that cannot reasonably be supported or inferred from the supplied context. Judge faithfulness to the context even if the introduced information might be true in the real world.

Criterion N: Every substantive factual claim in the candidate response is supported by, entailed by, or reasonably inferable from the supplied context. Also choose N when the response merely omits contextual details, declines or is unable to answer, uses harmless introductory or meta wording, or expresses a reasonable paraphrase, summary, abstraction, or inference. Formatting, length, style, and other instruction-following violations do not by themselves count as context inconsistency.

## State:

candidate\_response: "[the model output being judged]";

context: "[the RAGTruth source material]"; instruction: "[the RAGTruth task prompt]"

## Prompt for logical inconsistency (LI)

Instructions: Review the entire candidate response, including the relationships among its statements, reasoning steps, calculations, and conclusion. Determine whether it is internally logically inconsistent. Choose H if it contains any internal contradiction defined by the H criterion. Otherwise choose N.

Criterion H: Two or more parts of the response are mutually incompatible; the response contradicts one of its earlier statements; or its conclusion conflicts with its own premises, assumptions, calculations, or stated reasoning steps.

Criterion N: The statements, reasoning steps, calculations, and conclusion are internally consistent with one another. A factual error, unverifiable claim, lack of contextual support, or failure to follow an instruction does not by itself count as internal logical inconsistency.

## State:

candidate\_response: "[ContraDoc document | ROSCOE reasoning chain]";

instruction: "Determine whether the document is self-contradictory." | "Determine whether the response is internally   
logically inconsistent."

## Prompt for instruction inconsistency (II)

Instructions: Review the entire user instruction and candidate response. Determine whether the response fails to follow the instruction. Choose H if the response violates at least one applicable explicit requirement defined by the H criterion. Otherwise choose N. Evaluate only instruction compliance.

Criterion H: The response performs the wrong task, violates an explicit requirement or prohibition, omits required content or a required deliverable, or fails to satisfy an explicitly required format, language, length, style, role, structure, or other constraint.

Criterion N: The response performs the requested task and satisfies every applicable explicit requirement and constraint. Optional preferences do not have to be satisfied. A factual error, unverifiable statement, contextual error, or internal logical problem does not by itself count unless it directly violates an explicit instruction.

## State:

candidate\_response: "[the model response being judged]";

instruction: "[the full IF-RewardBench prompt with its explicit constraints]"

B Model Configurations, Cost, and Latency
<table><tr><td>Model</td><td>Provider</td><td>Size</td><td>Version</td></tr><tr><td>Jev</td><td>TypeSafe AI</td><td>NR</td><td>1.13.0</td></tr><tr><td>d1</td><td>Liquid AI</td><td>NR</td><td>(not reported)</td></tr><tr><td>clef</td><td>Cloudflare</td><td>27B</td><td>(not reported)</td></tr><tr><td>pplx-decider- v1.1-27b</td><td>Perplexity</td><td>27B</td><td>v1.1</td></tr><tr><td>GPT-6 Luna</td><td>OpenAI</td><td>NR</td><td>(not reported)</td></tr><tr><td>GPT-6 Sol</td><td>OpenAI</td><td>NR</td><td>(not reported)</td></tr></table>

Table 5: Model names, providers, sizes, and versions. The abbreviation pplx denotes pplx-decider-v1.1-27b. The first five rows are decision models; the final row is the generative reference. B denotes billion parameters; NR means the parameter count is not reported in the consulted official documentation for the evaluated API model. Published sizes follow the official clef and pplx model cards.

Table 5 summarizes the providers, parameter counts, and recorded versions of the five decision models and the generative reference. The five decision models come from five model providers; OpenRouter is the hosting interface used for pplx, not its model provider.

Versions are extracted from the recorded model IDs; the date in the pplx row is the recorded snapshot suffix. For the other entries marked (not reported), the logs contain only an unversioned model alias and do not identify a specific checkpoint. The evaluated Liquid AI API model is d1; we do not assign it the parameter count of a separately released open-weight d1 variant.

Cost and latency. Table 6 summarizes total estimated cost and mean API latency over all 9,852 type-specific evaluation instances per model. Cost is aggregated across the five types; latency is the mean of the recorded per-call times.

<table><tr><td>Total cost Model (USD)</td><td>Mean latency (ms)</td></tr><tr><td>jev 0.5409</td><td>271.22</td></tr><tr><td>d1 0.4084</td><td>328.90</td></tr><tr><td>clef</td><td>2.5894 508.72</td></tr><tr><td>pplx 0.2092</td><td>306.14</td></tr><tr><td>gpt-6-luna 1.0393</td><td>256.26</td></tr><tr><td>gpt-6-sol</td><td>30.7962 2,096.90</td></tr></table>

Table 6: Cost and latency across all evaluated instances. Cost sums the per-type estimates in Table 2; latency averages recorded per-call times, weighted by the number of instances in each type. The cost estimator counts input tokens for decision models and both input and output tokens for gpt-6-sol.

## C Per-Dataset Results

Table 7 reports BAcc for every judge on each source dataset, and Table 8 the corresponding falsenegative and false-positive rates.

## D Error Analysis

## D.1 Errors by Source Annotation

Table 9 breaks recall of H down by attributes available in the source annotations. The attributes are the number of annotated error spans or violated constraints, the salience of the error, the task form, and the model that generated the judged response. Where both classes are present, the table reports BAcc instead of recall.

## D.2 Error Patterns and Overlap

Errors concentrate on sparse and subtle targets. To locate where decision models fail, we break recall of H down by the source annotations (Table 9). The amount of evidence matters most. On RAGTruth, pplx recovers 57% of responses with one annotated span but 96% of those with three or more. On IF-RewardBench, it recovers 46% of responses that violate one constraint but 74% of those that violate three or more. gpt-6-sol recovers 91% and 76% of the single-error cases. Salience matters next. The recall of pplx drops from 71% to 42% for RAGTruth spans annotated only as subtle, and from 91% to 32% for ContraDoc contradictions created by replacing a sentence rather than inserting one. Task form matters as well. All decision models fall to 47 to 63% BAcc on multi-turn instructions, and every judge, including gpt-6-sol, is close to chance on the non-arithmetic ROSCOE chains. The generative judge is not uniformly better either. On RAGTruth data-to-text items, pplx reaches 86% BAcc against 79% for gpt-6-sol.

Decision-model errors overlap only partly with the generative judge. pplx and gpt-6-sol are both wrong on 4% of CI items, 6 to 8% of FC and FF items, and 11 to 13% of LI and II items. The Jaccard overlap of their error sets rises from 0.13 on CI to 0.50 on LI. Overlap among decision models is higher and is again largest on LI, between 0.51 and 0.57 for pplx against each other decision model. Errors are thus least shared on the types where decision models are already strong and most shared where they are weak. This pattern anticipates RQ3.

<table><tr><td>Model</td><td>FC/FAVA</td><td>FC/HAD</td><td>FF/FAVA</td><td>FF/HAD</td><td>CI/RAGT.</td><td>LI/CDoc</td><td>LI/ROSC.</td><td>II/IFRB</td></tr><tr><td>jev</td><td>66.6</td><td>75.0</td><td>70.6</td><td>87.3</td><td>82.5</td><td>68.5</td><td>57.7</td><td>61.7</td></tr><tr><td>d1</td><td>58.4</td><td>72.8</td><td>54.9</td><td>87.6</td><td>75.8</td><td>58.9</td><td>59.7</td><td>62.5</td></tr><tr><td>clef</td><td>62.7</td><td>72.7</td><td>57.7</td><td>85.4</td><td>68.2</td><td>62.9</td><td>64.5</td><td>64.5</td></tr><tr><td>pplx</td><td>65.2</td><td>76.8</td><td>63.8</td><td>94.8</td><td>82.7</td><td>75.0</td><td>67.2</td><td>71.1</td></tr><tr><td>gpt-6-luna</td><td>65.6</td><td>73.0</td><td>68.5</td><td>86.3</td><td>64.0</td><td>68.2</td><td>61.1</td><td>54.2</td></tr><tr><td>gpt-6-sol</td><td>67.3</td><td>80.4</td><td>68.4</td><td>92.6</td><td>82.7</td><td>77.7</td><td>65.2</td><td>78.9</td></tr><tr><td>n (H)</td><td>377 (224)</td><td>1244 (120)</td><td>351 (198)</td><td>1265 (141)</td><td>2675 (943)</td><td>891 (449)</td><td>755 (89)</td><td>2294 (1600)</td></tr></table>

Table 7: BAcc (%) per source dataset. FAVA is FAVABench, HAD is HADTest, RAGT. is RAGTruth, CDoc is ContraDoc, ROSC. is ROSCOE, and IFRB is IF-RewardBench.

<table><tr><td>Model</td><td>FC/FAVA</td><td>FC/HAD</td><td>FF/FAVA</td><td>FF/HAD</td><td>CI/RAGT.</td><td>LI/CDoc</td><td>LI/ROSC.</td><td>II/IFRB</td></tr><tr><td>jev</td><td>38/29</td><td>42/8</td><td>25/34</td><td>5/20</td><td>21/14</td><td>49/14</td><td>81/4</td><td>20/56</td></tr><tr><td>d1</td><td>59/24</td><td>47/8</td><td>80/10</td><td>23/1</td><td>41/7</td><td>81/2</td><td>76/4</td><td>26/49</td></tr><tr><td>clef</td><td>48/27</td><td>50/5</td><td>74/10</td><td>28/2</td><td>60/3</td><td>73/1</td><td>62/9</td><td>22/49</td></tr><tr><td>pplx</td><td>28/42</td><td>41/6</td><td>56/16</td><td>8/3</td><td>31/3</td><td>45/5</td><td>58/7</td><td>41/17</td></tr><tr><td>gpt-6-luna</td><td>5/63</td><td>27/27</td><td>15/48</td><td>9/19</td><td>2/70</td><td>56/8</td><td>70/8</td><td>3/89</td></tr><tr><td>gpt-6-sol</td><td>17/48</td><td>28/12</td><td>41/22</td><td>8/7</td><td>6/29</td><td>31/13</td><td>63/7</td><td>14/28</td></tr></table>

Table 8: False-negative rate / false-positive rate (%) per source dataset.

![](images/d3a24731493020a92fe775f18769777253ded218f0f7d003170eca3664836e83.jpg)  
Figure 7: Confidence distributions of correct decisions and errors. For each model, the share of correct decisions (gray bars) and of errors (red outline) in each confidence bin. The results are averaged over the five hallucination types, with each dataset weighted equally within its type.

## E Confidence Analysis Details

Confidence and hallucination type. To test whether the accuracy implied by a confidence level depends on the hallucination type t, we fit, for each decision model, logit $\mathrm { P r } ( { \hat { y } } _ { i } \ = \ y _ { i } ) \ =$ $\beta _ { 0 } + \beta _ { 1 } c _ { i } + \gamma _ { t } + \delta _ { t } c _ { i }$ , with standard errors clustered by response, and test $\delta _ { t } = 0$ for all t. The interaction is significant for every model (Wald $\chi ^ { 2 } = 5 7 \mathrm { - } 1 4 2 , \mathrm { a l l } p < 1 0 ^ { - 1 0 } )$

Confidence distributions. Figure 7 shows, for each model, how its correct decisions and its errors are distributed over confidence. For a useful confidence score, errors should concentrate at low confidence and correct decisions at high confidence. This holds most clearly for clef, whose errors are spread over the lower confidence range and only 10% of which fall at $c \geq 0 . 9$ . For gpt-6-luna and especially gpt-6-sol, the error distribution peaks in the highest confidence bin, much like that of correct decisions, so high confidence alone does not reliably distinguish between the two.

Robustness check of the calibration results. The ECE ranking of the models is unchanged under 15 equal-width bins, under 10 or 15 equal-mass bins, and under a debiased ECE. Clipping probabilities at 10<sup>−2</sup> instead of $1 0 ^ { - 4 }$ changes each model’s median calibration slope by at most 0.09.

<table><tr><td>Attribute (dataset, type)</td><td>Level</td><td>n</td><td>pplx</td><td>jev</td><td>gpt-6-luna</td><td>gpt-6-sol</td></tr><tr><td>Error density</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Annotated spans (RAGTruth, CI)</td><td>1</td><td>590</td><td>57</td><td>71</td><td>97</td><td>91</td></tr><tr><td>2</td><td>206</td><td>81</td><td>90</td><td>99</td><td>99</td></tr><tr><td>≥3</td><td>147</td><td>96</td><td>98</td><td>100</td><td>99</td></tr><tr><td>1</td><td>764</td><td>46</td><td>73</td><td>95</td><td>76</td></tr><tr><td></td><td>2</td><td>465</td><td>68</td><td>83</td><td>98</td><td>91</td></tr><tr><td></td><td>≥3</td><td>371</td><td>74</td><td>89</td><td>100</td><td>98</td></tr><tr><td rowspan="3">Salience Span type (RAGTruth, CI)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>evident</td><td>866</td><td>71</td><td>82</td><td>98</td><td>95</td></tr><tr><td>subtle only</td><td>77</td><td>42</td><td>55</td><td>97</td><td>78</td></tr><tr><td rowspan="3">Contradiction plug (ContraDoc, LI) Contradiction type (ContraDoc, LI)</td><td>insert</td><td>178</td><td>91</td><td>84</td><td>73</td><td>93</td></tr><tr><td>replace</td><td>271</td><td>32</td><td>29</td><td>25</td><td>53</td></tr><tr><td>negation / numeric</td><td>63</td><td>86</td><td>86</td><td>83</td><td>90</td></tr><tr><td rowspan="3">Fabrication subtype (FAVABench, FF)</td><td>content / factual</td><td>306</td><td>55</td><td>50</td><td>41</td><td>71</td></tr><tr><td>perspective / emotion</td><td>58</td><td>31</td><td>26</td><td>28</td><td>40</td></tr><tr><td>invented</td><td>48</td><td>67</td><td>85</td><td>100</td><td>71</td></tr><tr><td></td><td>subjective / unverifiable</td><td>138</td><td>35</td><td>70</td><td>80</td><td>54</td></tr><tr><td>Task form (BAcc) Task type (RAGTruth, CI)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Instruction type (IF-RewardBench, II)</td><td>QA</td><td>875</td><td>81</td><td>82</td><td>65</td><td>81</td></tr><tr><td>Summary</td><td>900</td><td>72</td><td>74</td><td>66</td><td>82</td></tr><tr><td>Data2txt single turn</td><td>900</td><td>86</td><td>82</td><td>51</td><td>79</td></tr><tr><td>multi turn</td><td>984 574</td><td>77</td><td>66</td><td>58</td><td>79</td></tr><tr><td rowspan="4">Reasoning source (ROSCOE, LI)</td><td></td><td></td><td>63</td><td>53</td><td>51</td><td>72</td></tr><tr><td>system prompt</td><td>736</td><td>68</td><td>59</td><td>52</td><td>85</td></tr><tr><td>gsm8k</td><td>200</td><td>81</td><td>67</td><td>76</td><td>84</td></tr><tr><td>cosmos</td><td>194</td><td>57</td><td>52</td><td>53</td><td>53</td></tr><tr><td rowspan="4">Document type (ContraDoc, LI)</td><td>drop</td><td>210</td><td>63</td><td>52</td><td>51</td><td>61</td></tr><tr><td>esnli</td><td>151</td><td>68</td><td>49</td><td>49</td><td>48</td></tr><tr><td>wiki</td><td>295</td><td>83</td><td>77</td><td>74</td><td>80</td></tr><tr><td>news</td><td>312</td><td>76</td><td>68</td><td>71</td><td>82</td></tr><tr><td></td><td>story</td><td>284</td><td>65</td><td>60</td><td>59</td><td>71</td></tr><tr><td rowspan="4">Generator (BAcc) Response generator (RAGTruth, CI)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5 / GPT-4 Llama-2 (7B/13B/70B)</td><td>891</td><td>54</td><td>66</td><td>66</td><td>80</td></tr><tr><td>Mistral-7B</td><td>1348</td><td>85</td><td>82</td><td>62</td><td>78</td></tr><tr><td></td><td>436</td><td>85</td><td>79</td><td>65</td><td>81</td></tr></table>

Table 9: Where judges fail. Rows under Error density and Salience report recall of H (%) on the H items with the given attribute. Rows under Taskform and Generator report BAcc (%) on all items with the attribute. Attributes come from the source annotations and were not shown to the judges.

## F Cascade Deferral-Rate Sensitivity

Figure 8 shows sensitivity to two target deferral rates: $r _ { 1 }$ is the fraction of all items passed from the first member to the second, and $r _ { 2 }$ is the fraction of second-stage items passed to the third. Each stage defers its least-confident items. $\mathbf { A } \mathbf { t } \ r _ { 1 } \ = \ 0$ , all curves coincide with the strongest member alone.

Moderate deferral helps selected types. FC, FF, and CI peak at intermediate $r _ { 1 }$ values (20–30%), reaching 82.2%, 84.6%, and 84.6% BAcc, respectively; larger rates eventually reduce performance. The separation between curves also shows sensitivity to r<sub>2</sub>, particularly for FC and CI. LI performs best without deferral: every tested nonzero $r _ { 1 }$ lowers BAcc. II gains at most 0.7 percentage points over the strongest member and remains 7.1 points below gpt-6-sol. Thus, deferring more lowconfidence items does not consistently improve accuracy; the benefit depends on the hallucination type and both operating rates.

## F.1 Peak and Cross-Fitted Results

Peak settings maximize pooled BAcc on the 36-point grid. Cross-fitting uses five label-stratified folds and seeds 0–4: fit rates and confidence thresholds on four folds, then apply them to the held-out fold. Members stay ordered by full-sample BAcc, so this checks rate/threshold selection, not the full model-selection pipeline. Ties use ascending $( r _ { 1 } , r _ { 2 } )$ order.

Cross-fitting lowers BAcc by at most 0.52 points. FC, FF, and CI retain gains over the strongest member, with intervals excluding zero. Their cross-fitted differences from gpt-6-sol have intervals containing zero. LI always selects no deferral; II has no clear gain over its strongest member.

![](images/ed3e692e022bed9aebc2ba5531c0a4db8b68fd02cc850d72dffeb2ad02ecdacb.jpg)  
Stage-1 deferral rate r<sub>1</sub> (%)

Figure 8: Cascade performance across deferral rates. The x-axis varies $r _ { 1 } ;$ each colored line fixes $r _ { 2 } .$ . Both rates range from 0% to 50% in 10-percentage-point steps (36 configurations per type). Stars mark the highest BAcc, dashed lines mark gpt-6-sol, and dotted lines mark the strongest member alone. Members appear in stage order below each panel. Peaks are selected and evaluated on the same items; they are post-hoc maxima, not held-out results.
<table><tr><td>Type</td><td>Evaluation</td><td>BAcc</td><td>∆ single [95% CI]</td><td>∆ sol [95% CI]</td><td>∆ peak</td></tr><tr><td>FC</td><td>Peak</td><td>82.21</td><td>+3.28 [0.98, 5.57]</td><td>+0.60 [-1.88, 2.96]</td><td>+0.00</td></tr><tr><td>FC</td><td>Cross-fitted</td><td>81.69</td><td>+2.76 [0.63, 4.85]</td><td>+0.08 [-2.33, 2.51]</td><td>-0.52</td></tr><tr><td>FF</td><td>Peak</td><td>84.61</td><td>+1.89 [0.38, 3.45]</td><td>+2.57 [0.02, 5.14]</td><td>+0.00</td></tr><tr><td>FF</td><td>Cross-fitted</td><td>84.39</td><td>+1.67 [0.19, 3.18]</td><td>+2.34 [-0.22, 4.76]</td><td>-0.22</td></tr><tr><td>CI</td><td>Peak</td><td>84.61</td><td>+1.94 [0.65, 3.24]</td><td>+1.95 [0.38, 3.57]</td><td>+0.00</td></tr><tr><td>CI</td><td>Cross-fitted</td><td>84.24</td><td>+1.58 [0.36, 2.84]</td><td>+1.59 [-0.07, 3.19]</td><td>-0.37</td></tr><tr><td>LI</td><td>Peak</td><td>73.28</td><td>+0.00 [0.00, 0.00]</td><td>-3.85 [-5.89, -1.84]</td><td>+0.00</td></tr><tr><td>LI</td><td>Cross-fitted</td><td>73.28</td><td>+0.00 [0.00, 0.00]</td><td>-3.85 [-5.82, -1.77]</td><td>+0.00</td></tr><tr><td>ⅡI</td><td>Peak</td><td>71.80</td><td>+0.72 [-0.12, 1.60]</td><td>-7.07 [-9.38, -4.85]</td><td>+0.00</td></tr><tr><td>Ⅱ</td><td>Cross-fitted</td><td>71.44</td><td>+0.36 [-0.51, 1.20]</td><td>-7.43 [-9.71, -5.02]</td><td>-0.36</td></tr></table>

Table 10: BAcc and paired differences in percentage points; single is the first member alone. Cross-fitted BAcc averages five seeds. Intervals use 2,000 paired item-bootstrap resamples, sharing indices across seeds before averaging. They condition on selected members and recorded predictions; peak intervals are not selection-adjusted.

<table><tr><td>Type</td><td>Evaluation</td><td>Realized  $r _ { 1 }$ </td><td>Realized  $r _ { 2 }$ </td><td>USD/1k</td><td>Cost (% sol)</td><td>Calls</td><td>Lat. (ms)</td></tr><tr><td>FC</td><td>Peak</td><td>30.0</td><td>9.3</td><td>0.0275</td><td>1.34</td><td>1.33</td><td>362.8</td></tr><tr><td>FC</td><td>Cross-fitted</td><td>27.0</td><td>10.0</td><td>0.0257</td><td>1.25</td><td>1.30</td><td>355.5</td></tr><tr><td>FF</td><td>Peak</td><td>19.7</td><td>49.2</td><td>0.0717</td><td>3.31</td><td>1.29</td><td>313.0</td></tr><tr><td>FF</td><td>Cross-fitted</td><td>20.8</td><td>48.0</td><td>0.0722</td><td>3.33</td><td>1.31</td><td>316.2</td></tr><tr><td>CI</td><td>Peak</td><td>20.0</td><td>0.0</td><td>0.0511</td><td>1.25</td><td>1.20</td><td>398.2</td></tr><tr><td>CI</td><td>Cross-fitted</td><td>22.6</td><td>0.0</td><td>0.0535</td><td>1.30</td><td>1.23</td><td>405.7</td></tr><tr><td>LI</td><td>Peak</td><td>0.0</td><td>一</td><td>0.0164</td><td>0.60</td><td>1.00</td><td>284.8</td></tr><tr><td>LI</td><td>Cross-fitted</td><td>0.0</td><td></td><td>0.0164</td><td>0.60</td><td>1.00</td><td>284.8</td></tr><tr><td>ⅡI</td><td>Peak</td><td>10.0</td><td>10.0</td><td>0.0576</td><td>1.55</td><td>1.11</td><td>371.2</td></tr><tr><td>Ⅱ</td><td>Cross-fitted</td><td>10.1</td><td>12.9</td><td>0.0580</td><td>1.56</td><td>1.11</td><td>372.9</td></tr></table>

Table 11: Realized routing percentages and replay efficiency. Realized $r _ { 2 }$ is conditional on reaching stage 2 (– if none). Cross-fitted rates pool counts across seeds; other quantities average over items and seeds. Confidence ties can make realized rates differ from their targets.

<table><tr><td></td><td>Type Members in stage order</td><td>Peak  $( r _ { 1 } , r _ { 2 } )$ </td><td>Cross-fitted targets: setting × fold count</td></tr><tr><td>FC</td><td> ${ \mathrm { p p l x } } \to { \mathrm { g p t } } { \mathrm { - } } 6 { \mathrm { - } } { \mathrm { l u n a } } \to { \mathrm { j e v } }$ </td><td>(0.3,0.1)</td><td> $( 0 . 2 , 0 . 1 ) \times 7 ; ( 0 . 3 , 0 . 1 ) \times 1 7 ; ( 0 . 4 , 0 . 1 ) \times 1$ </td></tr><tr><td>FF</td><td> $\mathbf { g p t - 6 - l u n a  j e v  p p l x }$ </td><td>(0.2, 0.5)</td><td> $( 0 . 2 , 0 . 5 ) \times 2 3 ; ( 0 . 3 , 0 . 3 ) \times 2$ </td></tr><tr><td>CI</td><td> $\mathrm { \ p p l { x }  j e v  d l }$ </td><td>(0.2, 0.0)</td><td> $( 0 . 1 , 0 . 0 ) \times 2 ; ( 0 . 2 , 0 . 0 ) \times 1 4 ; ( 0 . 3 , 0 . 0 ) \times 9$ </td></tr><tr><td>LI</td><td> ${ \mathrm { p p l x } } \to { \mathrm { j e v } } \to { \mathrm { g p t } } - 6 { \mathrm { - l u n a } }$ </td><td>(0.0, 0.0)</td><td> $( 0 . 0 , 0 . 0 ) \times 2 5$ </td></tr><tr><td>ⅡI</td><td> $\mathrm { \ p p l { x } }  \mathrm { c l e f  d 1 }$ </td><td>(0.1,0.1)</td><td> $( 0 . 1 , 0 . 0 ) \times 5 ; ( 0 . 1 , 0 . 1 ) \times 1 0 ; ( 0 . 1 , 0 . 2 ) \times 6 ; ( 0 . 1 , 0 . 3 ) \times 4$ </td></tr></table>

Table 12: Fixed member order and selected target rates. Cross-fitted counts cover 25 fits (five folds for each of five seeds). The target $r _ { 2 }$ applies only to items reaching the second stage.