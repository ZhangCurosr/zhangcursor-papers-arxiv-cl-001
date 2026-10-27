# Understanding Jev-Style System One Models: Does Structure Reduce Selection Bias?

Mingyang Song, Mao Zheng, Zhiyuan Ma, Xinpeng Zhou Foundation Model Department, Tencent, China nickmysong@tencent.com

## Abstract

Selection bias in large language models (LLMs) can persist despite instructions to ignore presentation order or prompts that elicit reasoning. Recent efforts to reproduce Jev-style System One behavior have produced a growing family of models that predict probabilities directly over supplied candidates. Yet systematic comparisons of selection bias across Jev versions, Jev reproductions, and generative LLMs remain limited in multi-option tasks. This leaves a central question: does Jev-style decision structure reduce selection bias? We address this question with a simple test that shuffles options while keeping question and candidate content fixed. Across two Jev versions, twelve Jev reproductions, and six generative LLMs, we trace individual answers through eight presentations and compare them with fixed-order repetition. On our primary collection of 1,700 questions from two benchmarks, Jev’s accuracy varies by less than two percentage points, yet its cumulative failure set expands by 27–29% relative to the original order, with first failures continuing into the final presentation. This loss of initial successes is substantially larger than under fixed-order repetition, even for highly repeatable Jev reproductions. Nor does Jev show a general advantage over the LLM baselines in preserving correct choices. Aggregation offers limited protection: voting yields modest gains for Jev, while even unanimous answers are frequently wrong. These results expose a gap between repeatable decisions and decisions that remain correct across equivalent presentations, showing why stable accuracy alone cannot establish resistance to selection bias.

(a) Accuracy (%)  
![](images/a996944be74ce797eb145759817a2dbcfbb042ed44de9bf52cf2db6721042c58.jpg)

(b) Agreement (%)  
![](images/2627d9135adfad41ee35a165765ff2fa043e3557d8b18c133823c8b8447add52.jpg)

(c) Cumulative failures  
![](images/d8ece72567d0b7df4984d3da528dfc24efc82cb800c39752ebac63f4c0b61f86.jpg)

(d) Retained correct (%)  
![](images/5a857380d3b2854968dbc101664dafd542db5196dac668e671e7d2c06f328162.jpg)  
Figure 1: Stable Jev accuracy hides an expanding failure set. Two Jev versions and six 1generative LLMs on 1,700 questions: (a) accuracy, (b) cumulative same-candidate agreement under eight original-order requests, (c) cumulative failed questions, and (d) retention of initial successes. Panels (a,c,d) use the original order (O), reversal (R), and six random permutations (S1–S6). Each failed question counts once; all requests retain the full menu.

![](images/f97d60c279e24ef9e2f402e53ea281c2d130d5cc47edde4af5f10289713a86ee.jpg)

(a) Accuracy (%)  
![](images/9d6bcb3bba08f1e0d87d98dcfe2e36eec8cb0ba5181a6512d65ab0f406ef5cf0.jpg)

(b) Agreement (%)  
![](images/860bfb6b9ee658e090bfc1edd66aca64a69115d701aaca7a27edbc0f70f3fec1.jpg)

(c) Cumulative failures  
![](images/07e8c1e77aeb2838bb1c373d9215df586d0ad5a58c74f76bdad87dd47fd5e303.jpg)

(d) Retained correct (%)  
![](images/1b69b248a4d5f61f8c1e2671d7b847f19ca001d17fd1b0057fbbb3748a5c75e2.jpg)  
Figure 2: High repeatability does not ensure order robustness. Twelve Jev reproductions 1and both Jev versions on the same 1,700 questions. Panels follow Figure 1. The reproductions achieve 97.7–100% same-candidate agreement under original-order repetition but lose 23.2– 64.2% of initial successes under reordering.

## 1 Introduction

A correct choice should depend on candidate content, yet language models can change their answers when only the presentation changes. Reordering options can alter multiple-choice answers (Pezeshkpour & Hruschka, 2024; Zheng et al., 2024), and swapping two unchanged responses can reverse an LLM judge’s preference (Wang et al., 2023; Zheng et al., 2023; Song et al., 2025). Such selection bias lets presentation influence a decision without adding evidence about which candidate is correct. A question answered correctly in one order may therefore fail in another.

Prompting alone does not reliably remove this bias. Zheng et al. (2024) find residual bias under explicit debiasing instructions and chain-of-thought prompting in the settings tested. Calibration and targeted training offer ways to reduce it (Li et al., 2024; Xue et al., 2024; Zheng et al., 2026), while evidence from multi-document question answering suggests that reasoning models can be less position-biased (Li et al., 2026). Selection bias is therefore sensitive to how a model is trained and used. This raises the possibility that models designed expressly for candidate selection could better preserve their decisions.

Jev-style System One models make this possibility concrete. They predict a probability distribution directly over candidates supplied with each request. TypeSafe describes Jev as trained for calibrated decisions through Reinforcement Learning for Calibrated Decisions (RLCD).<sup>1</sup> Recent efforts to reproduce this behavior have produced a growing family of models (Geuter et al., 2026; Li & Wagle, 2026; Zhuang et al., 2026), which we call Jev reproductions. These implementations range from training a decision head with backbone adaptation to extracting candidate probabilities from a frozen LLM. The term denotes a shared behavioral goal, without implying recovery of Jev’s proprietary internals or a common architecture. Candidate representation, probability readout, and decision-specific learning are therefore possible sources of robustness, not guarantees of it.

Existing evidence already challenges the assumption that structured decisions resist bias. Typed decision models can repeat pairwise judgments reliably yet reverse them when the two responses are swapped (Lukassen et al., 2026). In public Jev-style implementations, option labels can override written definitions, and serialization changes can alter this behavior with model weights fixed (Azizi et al., 2026). What remains limited is a systematic comparison of Jev versions, Jev reproductions, and generative LLMs as correct choices face repeated reordering in multi-option tasks. A single swap cannot reveal how widely failures spread over further orders, and aggregate accuracy can hide losses that recoveries offset. We therefore ask: does Jev-style decision structure reduce selection bias?

We investigate this question with a simple test that shuffles options while preserving question and candidate content. The study compares two Jev versions, twelve Jev reproductions, and six generative LLMs on SuperGPQA (M-A-P et al., 2025), LEXam (Fan et al., 2026), MMLU-Pro (Wang et al., 2024), and GPQA Diamond (Rein et al., 2023). Our primary collection pools 1,700 questions from SuperGPQA and LEXam. For each question, we compare the original order, its reversal, and six random permutations with eight requests in the unchanged original order. Tracking first errors reveals how the failure set expands and how many initial successes survive. Every request retains the full candidate menu.

The test reveals no general Jev advantage in preserving correct choices. Jev’s accuracy varies by less than two percentage points, yet its cumulative failure set expands by 27–29%, with first failures continuing into the final presentation (Figure 1). Four LLMs preserve more initial successes, including on questions every model initially answers correctly. Qwen3.8- Flash-Next and GLM-5.3-Flash have lower pooled retention, preventing a universal ranking of model families. The central finding is that Jev’s structured decision formulation does not by itself protect answers the model can already get right.

Fixed-order repetition exposes a second distinction: repeatability does not ensure robustness to reordering. Jev repeats choices more consistently than several LLMs, but loses substantially more initial successes under reordering than under repetition at the same call budget. The twelve Jev reproductions make this gap especially clear: 97.7–100% fixed-order agreement coexists with losses of 23.2–64.2% of initial successes under reordering (Figure 2).

The remaining analyses examine where correct choices are most fragile and whether repeated answers can repair them. Difficulty and menu-size comparisons locate this vulnerability. MMLU-Pro and GPQA Diamond show that it persists where Jev achieves higher accuracy and with only four candidates. Voting offers only modest gains for Jev, and unanimity still admits frequent errors. Together, these results distinguish average accuracy, fixed-order repeatability, and preservation of correct choices. The supporting survey in Appendices A–I organizes differences in representation, learning, and computation that controlled comparisons could isolate to explain the observed behavior.

## 2 Evaluation Setup

## 2.1 Datasets

Our primary evaluation uses a fixed collection of 1,700 test questions: 1,200 from SuperG-PQA (M-A-P et al., 2025) and 500 from LEXam (Fan et al., 2026).<sup>2</sup> The benchmarks cover graduate-level knowledge and legal examination questions. We weight questions equally, so pooled scores reflect this sample composition rather than an equal-weight average of benchmark scores.

SuperGPQA contributes questions with 4–10 options from its official evaluation release, balanced across easy, middle, and hard difficulty. LEXam contributes 125 test questions at each of 4, 8, 16, and 32 options. This collection supports analyses by difficulty, discipline, and candidate count. Candidate count varies with benchmark and question content, so these comparisons are descriptive.

Two further datasets test the scope of the findings. MMLU-Pro (Wang et al., 2024) contributes 300 test questions, with 100 each at eight, nine, and ten options and 21–22 from each of fourteen subjects. GPQA Diamond (Rein et al., 2023) contributes all 198 questions, each with four options.<sup>3</sup> We report these datasets separately to examine whether the vulnerability persists at higher Jev accuracy and with short candidate menus. The balanced MMLU-Pro sample does not estimate accuracy over the full benchmark distribution. All question sets remain fixed across models and presentations.

## 2.2 Models

The Jev–LLM comparison includes Jev 1.13 and Jev 1.14, with DeepSeek-V4.1-Flash-1017, Claude Sonnet 5.5, GPT-6.1-sol, Qwen3.8-Flash-Next, Gemini 3.8 Flash, and GLM-5.3-Flash as generative LLM baselines.<sup>4</sup> Jev returns a choice through its native decision interface. The LLMs receive a serialized representation of the same question and candidate menu and return a selection. Their training, inference procedures, and interfaces are not matched.

The comparison with Jev reproductions includes AutoTrust-JEV-9B, AutoTrust-JEV-27B, AutoTrust-JEV-35B, AutoTrust-JEV-27B-v1.1, AutoTrust-JEV-Gemma4-26B-A4B, StartLux-Decision-v1-0.8B, StartLux-Decision-v1-2B, StartLux-Decision-v1-4B, StartLux-Decision-v1- 9B, StartLux-Decision-v1-27B, StartLux-Decision-v1-35B-A3B (Zhuang et al., 2026), and pplx-decider-v1.1-27b.<sup>5</sup> We select the candidate with the highest native probability, without generating an answer or using deliberation. Both Jev versions serve as references, giving twenty unique models across the two comparisons. Every model is evaluated on all four datasets. Unsupported menus and absent selections count as incorrect. Candidate-identity statistics require actual selections and use the fixed sets defined below.

## 2.3 Eight Presentations of Each Question

Across all four datasets, each question is presented in its original candidate order, the reversed order, and six random permutations. All models receive the same permutation for a given question. Each candidate’s identifier moves with its content, preserving their association, while the question and state remain unchanged. Every request retains the full menu. These are repeated measurements of fixed questions. In the primary collection, the first random permutation differs from the original, and each subsequent random permutation differs from all preceding permutations. MMLU-Pro has eight distinct orders per question, while GPQA Diamond has seven or eight.

The intervention changes the order of candidate entries without independently manipulating identifiers or rewriting answers. The original presentation provides a reference for trajectories; it is not assumed to be the least biased order. Requests are independent: later presentations do not include earlier model answers.

We request low reasoning effort for DeepSeek-V4.1-Flash-1017, GPT-6.1-sol, Qwen3.8-Flash-Next, Gemini 3.8 Flash, and GLM-5.3-Flash, and use adaptive thinking with low effort for Claude Sonnet 5.5. Returned keys are mapped back to the original candidates. Moving a candidate to a different position does not count as a changed decision if the model selects the same candidate.

## 2.4 Repeated Requests in the Original Order

The original-order repetition study holds candidate content, identifiers, and order fixed across eight independent requests for every model: the original presentation and seven further requests. No request receives an earlier answer. Following the distinction between repetition stability and position consistency in Shi et al. (2025), this condition measures fixed-order repeatability through same-candidate agreement across requests.

For each comparison, we contrast eight original-order presentations with eight reordered presentations: the original, reversal, and six random permutations. The two conditions share the original request and use the same questions. We compare changed-choice rates, loss of initial successes, mean accuracy, and plurality gains. Sensitivity analyses on the primary collection compare the first three original-order requests with the original and every pair of the seven non-original presentations, and remove the shared original request to compare two further repeats with reversal and the first permutation.

On the primary collection, the Jev–LLM original-order profile and Figure 1(b) use the same fixed question set across all eight models and all eight original-order requests. The condition comparison additionally requires selections in every reordered presentation. In the comparison with Jev reproductions, accuracy, loss of initial successes, and voting instead use the full 1,700-question collection. Its standalone original-order identity statistics use each model’s questions with actual selections in all eight original-order requests. Identity comparisons between repetition and reordering use that model’s questions with actual selections in all 15 distinct requests across both conditions. These fixed sets prevent the denominator from changing with the number of presentations. The MMLU-Pro and GPQA Diamond analysis sets are specified in Sections 6 and 7. Same-candidate agreement requires the same actual candidate in every included request; its complement is the changed-choice rate. The same-wrong rate uses the corresponding identity set. Loss of initial successes is the fraction of first-request successes that become incorrect at least once, including a missing selection. Plurality ties use uniform tie-breaking in expectation. Pointwise 95% intervals use 2,000 benchmark-stratified bootstrap resamples of source groups, jointly preserving models and conditions; differences use paired draws.

## 2.5 Separating Scores, Correctness, and Error Identity

Let $\widehat { y } _ { x , t }$ be the candidate selected for question x in presentation $t \in \{ 1 , \ldots , 8 \}$ , with $\widehat { y } _ { x , t } = \perp$ bwhen no candidate is selected, and let $y _ { x }$ bbe its correct candidate. Accuracy is the proportion of all questions in the evaluated dataset or subgroup answered correctly in each presentation. Let  denote the fixed question set used to trace correctness. On the primary collection, the Jev–LLM trajectories use one common paired set across their eight models and reordered presentations; their original-order comparisons use the sets defined in Section 2.4. The comparison with Jev reproductions, including its Jev references, uses the full collection for correctness and error counts. Its candidate-identity rates use each model’s fixed set of questions with actual selections in all eight reordered requests, or the repetition sets defined above. The definitions below apply separately within each dataset and comparison. For a given model, define

$$
C _ { t } = \big \{ x \in \mathcal { Q } : \widehat { y } _ { x , t } = y _ { x } \big \} , \qquad R _ { t } = \frac { \big | \bigcap _ { s = 1 } ^ { t } C _ { s } \big | } { \big | C _ { 1 } \big | } .\tag{1}
$$

We call $R _ { t }$ retention: the fraction of initially correct questions that remain correct in every presentation through t. Its complement, $\dot { 1 } - R _ { t } ,$ is the loss $o f$ initial successes. A question that becomes wrong and later recovers no longer belongs to the retained set. Retention therefore measures continued correctness, rather than overlap with the correct set in the latest presentation alone. Both quantities are undefined when there are no initial successes.

Within the Jev–LLM comparison, we additionally restrict $\mathcal { Q }$ to questions answered correctly by all eight models in the original presentation. Each model’s retention then uses this same denominator. We report paired differences with 2,000 bootstrap resamples of source groups within each benchmark, jointly preserving all models and rounds. This analysis is conditional on the observed shared initial-correct set; it removes differences in which questions enter the comparison, without matching model capability or training.

To measure how many questions have failed, define the cumulative failed set and the number of first failures in presentation t as

$$
F _ { t } = \bigcup _ { s = 1 } ^ { t } ( \mathcal { Q } \setminus C _ { s } ) , \qquad B _ { t } = | F _ { t } | , \qquad N _ { t } = | F _ { t } \setminus F _ { t - 1 } | ,\tag{2}
$$

with $F _ { 0 } = \varnothing$ . Each question enters $F _ { t }$ once, at its first incorrect outcome, including no selected candidate. Later mistakes on that question add nothing to $N _ { t } ,$ even if the selected wrong candidate changes. Figure 1(c) reports $B _ { t }$ . Excluding previously failed questions is an analysis rule, not a change to the evaluated requests. Every request retains its full menu. The cumulative count is bounded by the number of questions, and $B _ { t } - B _ { 1 } = \lvert C _ { 1 } \rvert ( 1 - R _ { t } )$ links additional failures to the loss of initial successes.

Separately, accuracy discards changes in the identity of an incorrect answer. To track wrongoption diversity, we count distinct question–wrong-option pairs through presentation t:

$$
U _ { t } = \left| \bigcup _ { s = 1 } ^ { t } \{ ( x , \widehat { y } _ { x , s } ) : x \in \mathcal { Q } , \widehat { y } _ { x , s } \notin \{ y _ { x } , \bot \} \} \right| .\tag{3}
$$

Selecting the same wrong option again on the same question adds nothing to $U _ { t } ;$ selecting another wrong option adds one. Consequently, U<sub>t</sub> may exceed the number of questions ever answered incorrectly. It is nondecreasing by construction. The empirical quantities of interest are how many new pairs each presentation adds and how often a single question receives multiple distinct wrong answers.

To separate question coverage from wrong-option diversity, let $k _ { x , t }$ be the number of distinct wrong options actually selected for question x through presentation t, and let $W _ { t } = | \{ x \in$ $\mathcal { Q } : k _ { x , t } ^ { \smile } > 0 \}$ . Then

$$
U _ { t } = W _ { t } + D _ { t } , \qquad D _ { t } = \sum _ { x \in \mathcal { Q } } \operatorname* { m a x } ( k _ { x , t } - 1 , 0 ) .\tag{4}
$$

Each question contributes its first observed wrong candidate to $W _ { t }$ and any additional distinct wrong candidates to $D _ { t }$ . Since $D _ { 1 } = 0 ,$ , growth decomposes as $U _ { t } - \mathbf { \bar { \Phi } } U _ { 1 } = ( W _ { t } -$ $W _ { 1 } ) + D _ { t } . \mathrm { \bf ~ A }$ question can enter the failed set without contributing a wrong candidate, so $W _ { t } \le B _ { t }$ . They coincide on the paired set for the Jev–LLM comparison, where every request has a selection; there, $W _ { t } - W _ { 1 } ^ { \mathbf { { \bar { \alpha } } } } = B _ { t } - B _ { 1 }$ counts initially correct questions that later fail. Neither component of wrong-pair growth counts missing selections or repeated selections of the same wrong candidate.

## 2.6 Stratification and Uncertainty

The pooled collection supports complementary views of stability. Within each subgroup, we report accuracy averaged over the eight presentations, the percentage of questions whose selected candidate changes at least once, loss of initially correct questions (100(1 $\left. R _ { 8 } \right) )$ , and extra wrong pairs per 100 questions $( 1 0 0 D _ { 8 } / | \mathcal { Q } _ { g } |$ , where $\mathcal { Q } _ { g }$ is the subgroup’s correctness-analysis set). For the Jev–LLM comparison, accuracy uses all sampled questions in the subgroup and the other quantities use its subset of $\check { \mathcal { Q } }$ . In the comparison with Jev reproductions, accuracy, retention, and per-100-question counts use the full sampled subgroup. Changed-choice rates use the model’s subgroup questions with actual selections in all eight reordered requests. Within this identity set, changed choices split into questions alternating between correct and wrong answers and questions remaining wrong throughout with different wrong candidates. These disjoint groups sum to the changed-choice rate, separating visible correctness changes from variation hidden by accuracy. For the optioncount comparison, we also report newly failed questions per 100 questions, $1 0 0 \dot { ( } B _ { 8 } -$ $B _ { 1 } ) / | \mathcal { Q } _ { g } |$ . This measures the incidence of first failures after the original presentation across the subgroup, whereas $1 0 0 ( 1 - R _ { 8 } )$ measures loss conditional on initial success.

We use the source metadata to partition SuperGPQA into easy, middle, and hard questions (400 each) and broad disciplines. We also examine difficulty within the ten-option subset: 264 easy, 258 middle, and 209 hard questions. LEXam groups use option count. Both model comparisons include every observed menu size, from four to 32 options; these groups differ in benchmark composition as well as content. Undefined conditional metrics are omitted: no initial success leaves retention undefined, and no complete selection trajectory leaves candidate-identity rates undefined. Neither case indicates robustness. We retain small categories in the analysis but pool the five SuperGPQA disciplines with fewer than 30 sampled questions for visualization. Difficulty labels are available only for SuperGPQA; we do not assign them to the other benchmarks.

Error bars show pointwise 95% percentile intervals from 2,000 bootstrap samples. We resample source groups within each benchmark, keeping all presentations and models together within each resampled group. We omit intervals for groups with fewer than 20 questions or eight independent units. These exploratory intervals summarize sampling variation within the observed groups, without simultaneous coverage or a causal interpretation of differences between groups.

## 2.7 Voting and Agreement Across Presentations

We reuse the eight recorded selections to examine whether agreement provides useful evidence of correctness. Plurality voting selects the most frequently chosen actual candidate, without requiring an absolute majority. Missing selections cast no vote, and a question with no votes counts as incorrect. For ties, we report expected correctness under a uniform draw among the tied candidates. This rule favors neither the original presentation nor the gold answer. We compare voting with mean single-presentation accuracy on the same set : the paired set for the Jev–LLM comparison and the full collection for the comparison with Jev reproductions. The difference measures the gain from combining eight answers rather than using one of the eight with equal probability.

Unanimous coverage is the fraction of  receiving the same actual candidate selection in all eight presentations; any missing selection prevents unanimity. Accuracy within this subset measures how reliably unanimous answers are correct. We report these quantities together because higher conditional accuracy can come at the cost of answering fewer questions. The pooled bootstrap resamples source groups within each benchmark, preserving all models and presentations together. These are post-hoc analyses of the observed responses, without a new evaluation sample. Voting under original-order repetition also uses eight calls for every model (Section 2.4).

## 3 Does Jev Resist Changes in Option Order?

The central question concerns the survival of correct choices, not just the stability of a score. We first trace the failures hidden by aggregate accuracy, then compare their extent with LLM baselines and with fixed-order repetition.

## 3.1 What Stable Accuracy Conceals

All eight models in the Jev–LLM comparison maintain similar accuracy across presentations despite substantial differences in their overall performance. Jev scores near 50%, spanning only 1.47 percentage points for version 1.13 and 1.53 for version 1.14. The LLMs score approximately 56–72%, with similarly narrow ranges of 1.12–2.59 points (Figure 1(a)).

Between two presentations, let $n _ { C W }$ count correct-to-wrong transitions, n<sub>WC</sub> wrong-tocorrect transitions, and $n _ { W W ^ { \prime } }$ transitions between different wrong options. On a fixed set with actual selections in every included request, the accuracy change and decision-change rate are

$$
\Delta \mathrm { A c c } _ { \mathcal { T } } = \frac { n _ { W C } - n _ { C W } } { \left| \mathcal { Z } \right| } , \qquad F = \frac { n _ { C W } + n _ { W C } + n _ { W W ^ { \prime } } } { \left| \mathcal { Z } \right| } .\tag{5}
$$

Losses and recoveries offset each other in accuracy but add in the decision-change rate. Switches between wrong options affect only the latter, exposing further variation hidden by a stable score. For the reversal comparison with Jev reproductions,  is each model’s fixed eight-presentation identity set; the displayed transitions never count a missing selection as a wrong candidate.

Jev 1.14 illustrates this cancellation under reversal (Figure 3): 103 correct answers become wrong and 106 wrong answers become correct, yielding a net gain of three. Another 143 choices move between wrong candidates, bringing the total to 352 changed decisions. GPT-6.1-sol gains nine correct answers while changing 149 selections: 48 losses, 57 recoveries, and 44 wrong-to-wrong switches. Qwen3.8-Flash-Next and GLM-5.3-Flash change 416 and 541 decisions for net gains of only 13 and 17. Similar score changes thus conceal different amounts of turnover, including failures on questions each model could previously answer.

The same cancellation occurs in Jev reproductions (Figure 4). AutoTrust-JEV-9B loses 167 correct answers and recovers 170, while another 377 choices switch between wrong candidates. Its 714 changed decisions yield a net gain of three correct answers. Across all eight presentations, the twelve Jev reproductions’ accuracy ranges are only 1.12–2.18 points (Figure 2(a)). Stable aggregate performance therefore offers little evidence that individual decisions are preserved.

![](images/93655cef68f51f0e9cd57c433be26c3dab811076795167b1d1e11e0458f89c33.jpg)

![](images/924942d3efda78544fce58c80a2ae3ee3f5f1036031f97f363253b75b7881ab9.jpg)

![](images/98f5744b47f6ec798409165180dd96dae349b09c943e0339a2c46ff7b448fa62.jpg)

![](images/b3a07e46d8bac9d54b1b9ffe8268093b030232d4a5386482b5d9d4d2065960d1.jpg)

![](images/4dbc19607530f13d757e37bdf0bc730e15994fa9c27c42e8596e345ea67944bf.jpg)

Figure 3: Accuracy hides changes in individual choices. Original-to-reversed transitions 1on paired questions: (a) correct to wrong, (b) wrong to correct, (c) wrong to a different wrong candidate, and (d) all changed choices. Opposing transitions can cancel in accuracy; changes between wrong candidates leave it unchanged.  
![](images/95c5d6d52b5021b5505fd47295639d579c4bcb9d58d3da863d8ad0ca671ec50f.jpg)

![](images/2c0a26981a83110e3255ad4b617bfd151037ea39b89cf9dfed6041d757427f3a.jpg)

![](images/a48a6681efd5b4aaf06fb46d972637dbb7df9359c839ac72f63fc3b7e0a78773.jpg)

![](images/987d2222e63c024d04342709beedaefc571591b170014da07fe8de6cc7737b89.jpg)

![](images/377249d80040024eb15e28971fb4e73541e83dda9057a40ec8b507db1da75dd1.jpg)  
Figure 4: Accuracy also hides turnover in Jev reproductions. Reversal transitions on 1each model’s fixed identity set, using the categories in Figure 3. AutoTrust-JEV-9B loses 167 correct answers and recovers 170, while another 377 choices move between wrong candidates.

## 3.2 First Failures Continue Beyond the Initial Presentations

Obs. 1. Jev’s stable accuracy hides an expanding set of failed questions. Jev 1.13 fails on 755 questions in the original presentation and on 961 across all eight, while Jev 1.14 rises from 746 to 965 (Figure 1(c)). The additional 206 and 219 questions expand the observed failure sets by 27.3% and 29.4%, despite the narrow accuracy ranges. The LLMs also expose new failures, ranging from 111 for GPT-6.1-sol to 414 for GLM-5.3-Flash, with DeepSeek-V4.1-Flash-1017, Qwen3.8-Flash-Next, and GLM-5.3-Flash exceeding Jev. Absolute additions alone do not establish a robustness ranking, however, because models begin with different numbers of correct answers available to lose.

First failures continue well beyond the initial reversal. Reversal exposes 96 and 103 for the two Jev versions; the eighth presentation still reveals seven and eight questions answered correctly in all seven preceding presentations. The LLMs add 7–32 first failures in that round, with GLM-5.3-Flash adding the most. Cumulative counts are nondecreasing by definition. The empirical finding is the scale of the expansion and the continued loss of questions that have already survived several orders.

All twelve Jev reproductions likewise expose first failures in the eighth presentation, adding 3–14 per model (Figure 2(c)). Even AutoTrust-JEV-27B-v1.1, which has comparatively favorable retention, expands from 887 failed questions in the original order to 1,076 across the sequence, with 11 first failures in the final presentation.

## 3.3 Persistence on Shared Initial Successes

Obs. 2. Jev’s retention disadvantage persists against four LLMs on shared initial successes. Retention normalizes the same first-failure expansion by the number of initial successes, since $B _ { 8 } - B _ { 1 } = | C _ { 1 } | ( 1 - R _ { 8 } )$ . Across all eight presentations, Jev retains 73.2–74.5% of its initially correct answers, compared with 77.2–90.2% for DeepSeek-V4.1-Flash-1017, Claude Sonnet 5.5, GPT-6.1-sol, and Gemini 3.8 Flash, 66.1% for Qwen3.8-Flash-Next, and 53.7% for GLM-5.3-Flash (Figure 1(d)). Equivalently, 25.5–26.8% of Jev’s initial successes fail at least once. The structured interface does not prevent this loss, while Qwen3.8-Flash-Next’s and GLM-5.3-Flash’s lower retention rule out a blanket Jev–LLM ranking.

This comparison could be affected by which questions each model answers correctly in the first place. We therefore repeat it on the shared initial-correct set, requiring every model to begin with a correct answer to every included question. Jev 1.13 and Jev 1.14 retain 85.4% and 84.4%, respectively. Retention reaches 93.1% for DeepSeek-V4.1-Flash-1017, 96.3% for Claude Sonnet 5.5, 98.3% for GPT-6.1-sol, and 97.9% for Gemini 3.8 Flash. GPT-6.1-sol’s paired advantage is 12.9 percentage points over Jev 1.13 (95% interval 9.6–16.2) and 13.9 over Jev 1.14 (10.6–17.3). The DeepSeek-V4.1-Flash-1017, Claude Sonnet 5.5, and Gemini 3.8 Flash intervals also exclude zero. Qwen3.8-Flash-Next retains 85.2%, with differences of 0.2 points from Jev 1.13 ( 4.4–3.7) and 0.8 from Jev 1.14 ( 3.5–5.0); neither interval establishes a difference. GLM-5.3-Flash retains only 70.5%, below both Jev versions, with both paired intervals excluding zero.

Matching initial successes therefore preserves Jev’s disadvantage relative to DeepSeek-V4.1-Flash-1017, Claude Sonnet 5.5, GPT-6.1-sol, and Gemini 3.8 Flash, while bringing Qwen3.8-Flash-Next close to Jev and leaving GLM-5.3-Flash below it. This comparison controls which questions enter the calculation, without matching model capability, training, or variation under repeated requests.

Retention also varies widely within the decision-model family. Under full-collection scoring, AutoTrust-JEV-27B-v1.1 loses 23.2% of initial successes, below the Jev reference losses of 27.0% and 27.8%. The other eleven reproductions lose 30.0–64.2%, including 30.0% for pplx-decider-v1.1-27b and 49.5% for AutoTrust-JEV-Gemma4-26B-A4B. Native candidate prediction thus supports implementations on both sides of the Jev results. These rates condition on each model’s own initial successes, so their spread also reflects differences in the questions each model initially answers correctly.

## 3.4 Repeatability Under a Fixed Order

Obs. 3. Jev’s repeatability under a fixed order does not imply robustness to reordering. The preceding trajectories vary both the request and the candidate order. Holding the order fixed reveals a different comparison (Figure 1(b)). Over eight original-order requests, Jev 1.13 and Jev 1.14 achieve 85.0% and 84.4% same-candidate agreement. Both exceed DeepSeek-V4.1-Flash-1017, Claude Sonnet 5.5, Qwen3.8-Flash-Next, and GLM-5.3-Flash. GPT-6.1-sol reaches 86.2%, while Gemini 3.8 Flash’s 84.7% lies between the Jev versions. GLM-5.3-Flash’s agreement falls to 44.5% even without a menu change. Thus, variability across reordered presentations includes variation that can already occur under repetition, and its extent differs substantially across models.

High repeatability also includes repeated mistakes (Figure 5). Across eight original-order requests, 36.8–38.0% of all paired questions receive the same wrong Jev answer throughout, and accuracy among its unanimous answers is only 55.4–56.4%. A stable choice therefore provides limited assurance of correctness.

Jev reproductions make the distinction between repeatability and correctness particularly clear (Figure 6). They agree on 97.7–100% of their identity-set questions across eight requests, yet 52.2–78.0% receive the same wrong answer throughout. AutoTrust-JEV-Gemma4- 26B-A4B and AutoTrust-JEV-27B-v1.1 repeat every observed choice; pplx-decider-v1.1-27b reaches 97.7%. Repeatedly selecting the argmax of native candidate probabilities can therefore produce highly stable errors.

![](images/5e2a83fde8e79c75ebd3a88f2f9390bed6a4a00a89e5c1b23b9d93660a38546b.jpg)

![](images/a4cabd7ae89bdd1e87ee4ec1ed643b22ca3f97b956d27ea33e79a313527e3583.jpg)

![](images/2d6a5a366e0deaf1b60ca3d0663d77df59b776bae9a238149df7448f20fd0621.jpg)

![](images/e092989642ae6287e80c719dec16ef5f86f79c6a3db76c12fca5eff76bb4f066.jpg)

Figure 5: Jev’s fixed-order agreement includes repeated errors. Eight original-order re-1quests: (a) mean accuracy, (b) agreement across all eight, (c) loss of initial successes, and (d) questions with the same wrong answer throughout. Panels use the same paired set as Figure 1(b). Error bars show pointwise 95% bootstrap intervals.  
![](images/cd74412b7c45e6b9dc8cc62641288a4bfaf2ea935c14cd3535c05aac260ecf8f.jpg)

![](images/52d701b02ec74853430c9728f7d77f18678780d79b30ec253e37057b126e108b.jpg)

![](images/48b215a2f368ea260fc159ad709d332a8f37876964bfb8509fbb0f58e9946abc.jpg)

![](images/66df740dd517f6ae09d50e8f36324787a7708bfc54e7aeee72d86c6928366e16.jpg)

![](images/e16e74ea4210fad945130d79a2b89f51bea7421c8ad51e80be8a9ad65e4d048b.jpg)  
Figure 6: High fixed-order agreement also repeats errors. Eight original-order requests, 1with metrics as in Figure 5. Identity statistics use each model’s fixed selection set; correctness metrics use the full collection. Error bars show pointwise 95% bootstrap intervals.

At an equal eight-call budget, Jev varies substantially more under reordering than under original-order repetition (Figure 7). On the question set shared by both conditions, its changed-choice rate is 23.6 percentage points higher for version 1.13 (95% interval 21.3–26.0) and 24.6 higher for version 1.14 (22.2–27.0), compared with differences of 2.2–7.2 across the six LLMs. This gap also extends to successful decisions. Jev loses 8.2–9.0% of its initial successes under repetition and 25.1–26.4% under reordering, with paired differences of 16.1 points (13.4–19.1) and 18.1 points (15.3–21.1), while mean accuracy differs by less than half a point. By contrast, GLM-5.3-Flash already loses 39.7% under original-order repetition, with a further 5.7-point difference under reordering. Variation under repetition and additional variation under reordering are distinct failure patterns.

The Jev reproductions make this distinction especially clear: they lose only 0–1.3% of initial successes under repetition, compared with 23.2–64.2% under reordering (Figure 8). Their changed-choice rates are 36.3–73.9 percentage points higher under reordering, versus differences of 25.8 and 25.9 for the Jev references. Identity rates here use each model’s fixed selection set across both conditions, as defined in Section 2.4.

Shorter sequences give the same qualitative result. Combining the original with each of the 21 pairs of non-original presentations yields a three-call comparison against three original-order requests. On a common question set, Jev’s excess changed-choice rate is 17.3–20.0 points for version 1.13 and 18.8–21.3 for version 1.14, versus 0.5–6.2 across the LLMs. Removing the shared original request also retains a 12.6–12.9-point excess for Jev. The large repetition–reordering gap is therefore not specific to using all eight presentations or including their shared initial response.

Eight original-order requests

![](images/1ffdccc6b5417120636f824a8d9a42500d8345514b7903a6abb8af9d6e714270.jpg)

![](images/92cce8da86bbeb6e02a52a44864ae9f9b688cd22c61a8f55aad0dc24a1cdcb93.jpg)

![](images/8dae63b98abceecf277af42b1b5289b5636dba211ec0255619d9c94864f7bc03.jpg)  
Eight reordered presentations

![](images/af40d6b3879149ff20c2d6e876076d347332288c8473896047009e23f09de1b1.jpg)

Figure 7: Reordering exposes more Jev instability at equal call counts. Eight original-order 1requests (hatched) versus eight reordered presentations (solid): (a) changed choices, (b) loss of initial successes, (c) mean accuracy, and (d) plurality gain over mean accuracy. Error bars show pointwise 95% paired bootstrap intervals.  
![](images/aa9e4e720f0a64df54909bbac1c396de4570dcedc1d1b3c0f5a11b412cf904fa.jpg)

![](images/bd3d97b57310e3fe93bc959fa0ba0fde493944d17f8b00f63d087df333c7539b.jpg)

![](images/2468f715dc199e38b69931f72ab3ed1007f856cde61fac6ddced64909d9c78e2.jpg)

![](images/7e70302b94a1837c66d4bd8afd6e89ed54b1aa5bde27040f3f23ed23604ce029.jpg)

![](images/de5359dc0f46b8513f23bd010ee3430be71eb00c715c048c3c5aedec74e357d0.jpg)  
Figure 8: Reordering reveals failures rarely seen under repetition. Eight calls per condition, 1with metrics as in Figure 7. Identity sets are fixed jointly across conditions; correctness metrics use the full collection. Error bars show pointwise 95% paired bootstrap intervals.

Qwen3.8-Flash-Next shows why the overall failure rate and the repetition–reordering gap answer different questions. Its changed-choice rate rises from 45.3% under repetition to 50.1% under reordering, an increase of 4.9 points compared with Jev’s 23.6–24.6. Its loss of initial successes rises by 3.7 points, from 29.2% to 32.9%. Qwen3.8-Flash-Next loses more successes than Jev under reordering, but much of its variability is already present with the menu fixed.

## 3.5 Errors Extend Beyond a Single Distractor

First-failure counts do not distinguish a persistent distractor from several distinct wrong answers on the same question. We therefore examine distinct question–wrong-option pairs, retaining selections after a question’s first error.

Obs. 4. Jev’s errors often extend beyond one distractor per question. If every failed question had one fixed wrong answer, the pair count would grow only when another ques tion failed. Jev 1.14 illustrates why this description is incomplete. Its original presentation produces 746 wrong pairs, one for each incorrectly answered question. After eight presentations, 965 questions have failed at least once, but the model has selected 1,449 distinct wrong pairs. The 703 additional pairs comprise 219 newly affected questions and 484 extra wrong candidates beyond the first per affected question. The latter component includes both initially failed questions and questions that first fail later and then receive another wrong answer.

Equation 4 separates these two sources for every model (Figure 9). Extra wrong candidates account for 68.8–68.9% of Jev’s pair growth, similar to Qwen3.8-Flash-Next’s 68.0% and GLM-5.3-Flash’s 70.7%, and above the other four LLMs’ 50.2–58.5%. For example, GPT-6.1-sol’s increase from 437 to 661 pairs splits almost equally between 111 newly affected questions and 113 extra wrong candidates. Most of $\mathrm { J e v } ^ { \prime } { \bar { \mathbf { s } } } ,$ Qwen3.8-Flash-Next’s, and GLM-5.3-Flash’s pair growth thus reflects additional ways to fail on affected questions, rather than the first failures counted in Obs. 1.

![](images/ed6b0ebf7d623ca6297de9e184b646aa7e931568a637c2927833414b0f2ec693.jpg)

![](images/acdc6e3e54de3743ae1d1bde862af44038f19b8ea52e02b3cb52852c407d544c.jpg)

![](images/7913bf2201596d6d0108b6f7ab02c4d2b0da19ecef5c54b089f9dce93604ceb4.jpg)

![](images/37e279c2d89be5f324bbe4646c6d02b321bbc791c0754733c6f753f6aee7472d.jpg)

Figure 9: Additional wrong candidates drive most of Jev’s wrong-pair growth. (a) Original errors $U _ { 1 } ,$ , (b) newly failed questions $B _ { 8 } - B _ { 1 } ,$ , (c) extra wrong candidates beyond the first per affected question $D _ { 8 } ,$ and (d) their sum $U _ { 8 } .$ . Counts are distinct question–wrong-option pairs and can exceed the number of questions. Every request retains the full menu.  
![](images/cc93cac5063400396064d8ee0e45f7b4604434b1d1d9e281ac043a784ea7de0a.jpg)

![](images/718290b2d27d9d2f750f0c3042c7e46660fe27993dd3db9e46cea4787bf8b75e.jpg)

![](images/5c5d6c4cdf41eaabd47a37a3b3506c6aa66d61225560774958892877ade25b44.jpg)

![](images/ab425038d9e2003eeb6f3b983f03402898a4be7cd5c8856523257199bd21e608.jpg)

![](images/8bb7c21256f4e721e5bd91e964bbbd6ed06f45e36bccc187ebc000517537e0c9.jpg)  
Figure 10: Additional distractors dominate wrong-pair growth in Jev reproductions. (a) Original wrong pairs $U _ { 1 } ,$ 1 (b) newly observed questions with an actual wrong candidate $W _ { 8 } - W _ { 1 } , \mathrm { ( c ) }$ extra wrong candidates $D _ { 8 } ,$ and (d) all distinct wrong pairs $U _ { 8 } .$ . Extra candidates account for 70.8–89.5% of growth.

Extra wrong candidates account for 70.8–89.5% of wrong-pair growth in the twelve Jev reproductions, from AutoTrust-JEV-27B-v1.1 to StartLux-Decision-v1-0.8B (Figure 10). Nearly perfect fixed-order agreement can thus coexist with several distinct wrong answers to the same question under reordering.

Jev’s distinct wrong-pair count increases by 87.8–94.2%, far more than its 27.3–29.4% increase in failed questions. This is not the largest relative pair growth: the LLM increases range from 51.3% for GPT-6.1-sol to 211.4% for GLM-5.3-Flash, with DeepSeek-V4.1-Flash-1017 and Qwen3.8-Flash-Next also exceeding Jev. The distinction is between spreading failure to more questions and exposing more ways to fail on a question.

Multiple distractors are selected on 337–354 questions for Jev, below Qwen3.8-Flash-Next’s 404 and GLM-5.3-Flash’s 569, and above the 94–222 for the other four LLMs. Their contribution persists late in the sequence. Of Jev’s 41–46 new wrong pairs in the final presentation, 34–38 select a different wrong candidate on a previously affected question, accounting for 82.6–82.9% of the additions. Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash-1017 show a similar pattern, with shares of 80.8% and 79.6%. Late error discovery therefore mainly broadens the observed ways to fail within affected questions, even as a smaller number of previously successful questions fail for the first time.

The expansion is not specific to choosing the original or final presentation as a reference. Relative to the mean wrong-pair count of a single presentation, the full union is 86.2–92.0% larger for Jev and 53.9–214.9% larger for the LLMs. Averaging over all eight ways to omit one presentation leaves 42.4–47.4 unseen pairs for Jev, versus 15.5–125.8 for the LLMs. These comparisons characterize the recorded responses without predicting saturation under further evaluation.

The recorded errors therefore extend beyond one stable distractor per question. Because every request retains the full menu, these observations do not establish how the model would respond if a wrong candidate were removed.

## 4 How Does Jev’s Order Robustness Vary Across Tasks?

The pooled results establish that Jev’s stable score conceals lost successes. We now ask how broadly this pattern holds across benchmarks, difficulty levels, topics, and menu sizes. Each profile separates changed choices from lost initial successes, since a model can appear stable by remaining wrong. These subgroup comparisons locate vulnerability without isolating its causes.

## 4.1 Comparisons Across Benchmarks

Obs. 5. Jev’s relative stability depends on the task and the LLM comparator. The ordering of models by stability can change across benchmarks (Figure 11). On LEXam, Jev changes 39.9–42.9% of choices, below DeepSeek-V4.1-Flash-1017’s 47.1%, Qwen3.8-Flash-Next’s 61.6%, and GLM-5.3-Flash’s 75.5%, but above GPT-6.1-sol, Gemini 3.8 Flash, and Claude Sonnet 5.5. DeepSeek-V4.1-Flash-1017 instead changes fewer choices than either Jev version on SuperGPQA. GPT-6.1-sol and Gemini 3.8 Flash remain more stable than Jev on both benchmarks, while Qwen3.8-Flash-Next and GLM-5.3-Flash remain less stable. Jev’s relative position therefore depends on both the task and the comparator.

![](images/56fe2c011b946b3657b6f848e518b012298b9a56521ffaace369ecd87676d876.jpg)  
Figure 11: Jev’s relative stability depends on the task and baseline. By benchmark: (a) 1eight-presentation mean accuracy, (b) changed choices, (c) loss of initial successes, and (d) extra wrong pairs beyond the first per affected question, per 100 questions. Error bars show pointwise 95% bootstrap intervals (Section 2.6).

AutoTrust-JEV-27B-v1.1 changes fewer choices than either Jev version on each benchmark, but this does not translate into uniformly better retention (Figure 12). On LEXam, its loss of 20.5% lies between Jev 1.13’s 21.7% and Jev 1.14’s 17.9%, despite changing fewer choices than both. The distinction between preserving candidate identity and preserving correct answers therefore matters even within the decision-model group.

Lower turnover also need not yield more correct answers. Over half of Jev’s LEXam questions with changed choices remain wrong in every presentation. These always-wrong switches account for 21.7% of all paired LEXam questions for each Jev version, compared with 5.5–20.3% for the LLMs. Such trajectories contain no correct answer for voting to recover, linking the stability analysis to the aggregation results in Section 5.

![](images/6977de31002f45587dba418911e05a4d6a848e477bc8dddd18b564bb393999bc.jpg)

![](images/06eeed4abd0c23b91ff94c57419c68dce2b9518a211bcf6440837330d6aa178e.jpg)

![](images/ad20d4852e065db9776ca3b58e129e62b1667d7f382c71ca9a5ced4b087b9768.jpg)

![](images/8d3c72ddde808f3db50fbc3cb6fd306215a1871a57a1ef97df00faf603752404.jpg)

![](images/4a5bc1d4a4b0d635eb665c568dda5913f27d719f986a2652d0d318d574f633b8.jpg)  
Figure 12: Lower turnover need not mean better retention. Benchmark profiles of Jev 1reproductions, with metrics as in Figure 11. AutoTrust-JEV-27B-v1.1 changes fewer choices than either Jev version on every benchmark, yet loses more initial successes than Jev 1.14 on LEXam. Error bars show pointwise 95% bootstrap intervals.

## 4.2 Difficulty and the Persistence of Correct Answers

Obs. 6. Jev loses over half its initial successes on hard SuperGPQA questions. Hard questions challenge Jev beyond lowering its accuracy. Even conditional on an initially correct answer, 51.5–54.1% later become wrong, compared with 10.2–30.4% for five LLMs and 55.5% for GLM-5.3-Flash (Figure 13). An initial Jev success on these questions is therefore more often lost than preserved across all eight presentations. Jev’s overall changed-choice rate also rises from 27.2–30.2% on easy questions to 56.8–59.0% on hard questions. The five LLMs’ rates remain similar or decrease, whereas GLM-5.3-Flash rises from 47.7% to 65.3%. For Jev, the difficulty gradient concerns both the frequency of changing an answer and the fragility of an answer already known to be correct.

Jev 1.13 DeepSeek-V4.1-Flash-1017 Jev 1.14 Qwen3.8-Flash-Next

![](images/9afbd34abc229d3d8c5e7bbc9c542d7fff7da638a2edfed8522119e984703ced.jpg)

![](images/ff7fed594869f919c9817e2b8ab174315bb1e95da293fe1d7f2db8d40c18e1dd.jpg)  
Claude Sonnet 5.5 Gemini 3.8 Flash GPT-6.1-sol GLM-5.3-Flash

![](images/b532ef12e17e70dea32216a6a8cb2d5efd8b98580c0dad9977a9fb6e30d0dc2e.jpg)

![](images/ea735cc0a548223584528590b83e90991c1782b6c3e3a12283fdf1a6020b6992.jpg)  
Figure 13: Jev changes more choices on harder questions. SuperGPQA difficulty groups: 1(a) mean accuracy, (b) changed choices, (c) loss of initial successes, and (d) changed choices restricted to ten-option questions. Error bars show pointwise 95% bootstrap intervals. Panel (d) holds menu size fixed; question content still differs.

The difficulty gradient extends across the twelve Jev reproductions (Figure 14). Every model changes more choices on hard than easy SuperGPQA questions, including within ten-option menus. On hard questions overall, loss of initial successes ranges from 36.8% for AutoTrust-JEV-27B-v1.1 to 88.5% for StartLux-Decision-v1-4B. High fixed-order agreement thus does not protect a correct answer once the model confronts harder questions in a different order.

![](images/b81008bcaa66be4222f772d3774d0c628ac7f16432a13053c2b4cbce2f3be19b.jpg)

![](images/06729847fb2941d84a0c639651c15ad91f3b0c2768d3558e43001a50bf543746.jpg)

![](images/23dce56d7bb216619b413b47d3460d741de1bca5bf69cba3865eeab43d0123a6.jpg)

![](images/53f200a6fb6e87dcead4eda930f09bfb3850d51560faa7a5714acd3318a41b60.jpg)

![](images/933d0b04506f0cab87a49daf4040f9c5876ad426000584d76ea2600d83612788.jpg)  
Figure 14: Hard questions expose greater order sensitivity in Jev reproductions. SuperG-1PQA profiles, as in Figure 13. Every reproduction changes more choices on hard than easy questions, including within ten-option menus. Error bars show pointwise 95% bootstrap intervals; content and subject composition differ.

Menu size does not fully explain this gradient. Within ten-option questions, Jev’s changedchoice rate rises from 27.0–29.0% on easy questions to 58.9–60.9% on hard questions; every LLM shows a smaller increase (Figure 13(d)). Question content and subject composition still differ, so this comparison does not isolate difficulty itself.

Correct-answer retention shows that the vulnerability is present even on easy questions (Figure 15). Within ten-option groups, Jev loses 14.9–18.8% of initial successes on easy questions, 26.0–26.1% on middle questions, and 48.6–48.7% on hard questions. Across all 731 ten-option questions, the loss is 25.2–27.4%, while each version’s accuracy varies by only 2.05 percentage points. Hard questions amplify the discrepancy between stable accuracy and fragile successes; they are not necessary for it to appear.

Jev 1.13 DeepSeek-V4.1-Flash-1017 Claude Sonnet 5.5 Gemini 3.8 Flash Jev 1.14 Qwen3.8-Flash-Next GPT-6.1-sol GLM-5.3-Flash

![](images/6e5450dc654d311dd60ad827fe006f6aed57b49edc8ada5472cc1b8d1829012a.jpg)

![](images/ca5e70539d0df62bf2c68bd43411ea549605b77d6c3821b6a6c0d6f85f875f41.jpg)

![](images/61390ceaae27060ce849331f159904beb4b714042291c361e093794572fcd8dd.jpg)

![](images/e43e7eac7cbcb2dc25a07f95a88d2e873da0ea2f116ec6b3e773919487b816f0.jpg)  
Figure 15: Harder questions amplify Jev’s loss of correct answers. Retention across eight 1presentations for ten-option SuperGPQA questions: easy, middle, hard, and all difficulties. Each curve starts at 100% and retains a question only while every observed answer remains correct. Panels share a scale; question content differs across groups.

The same ten-option analysis separates the Jev reproductions (Figure 16). AutoTrust-JEV-27B-v1.1 loses 18.2% of initial successes on easy questions and 44.9% on hard questions. The other eleven lose 66.2–93.3% on hard questions, compared with 50.0–53.4% for the Jev references in this comparison. A common candidate count therefore leaves substantial differences both across difficulty groups and within the Jev-style family.

Subject profiles help locate this difficulty gradient (Figure 17). Engineering and science contain much of Jev’s instability and many of the hard questions: 163 of 399 engineering questions and 142 of 321 science questions are hard, compared with nine of 93 in literature and arts. Their higher changed-choice rates are consistent with the difficulty result, without identifying an independent subject effect. Separating correctness changes from alwayswrong switches also reveals which variation includes an answer that aggregation could recover.

![](images/eafa9b3712d956d6aa4cbe4764dd1ce74054dd7291efa667490b6656c97640fe.jpg)

![](images/acdc1d91fa91c21b0e88353bbb24762e01688fc78e044aa5258bf16b2aa7b88c.jpg)

![](images/6fb43b22299395f4f886b8ab6a9be5c02ecf1d6692e2da635aba8920f9cf2b78.jpg)

![](images/cd1bc95fccea4f9ec178f81e3ae5fd7dd19d75012bab0ab203c6fd8742005282.jpg)

![](images/c9ffc246d39dd712ec23eaa502088a5d7f13c8f1f2baf7e4c4aaee3626f6f4ce.jpg)  
Figure 16: Correct choices by Jev reproductions are more fragile on hard questions. Ten-1option SuperGPQA retention by difficulty. AutoTrust-JEV-27B-v1.1 loses 18.2% of initial successes on easy questions and 44.9% on hard questions. Panels share a scale, whose range differs from Figure 15.

![](images/a88fbd4c458c3bc7bdd20de47c8ed471113f6b398f3d590e913a6640c0a87f34.jpg)  
Figure 17: Subject profiles separate correctness changes from wrong-answer switches. 1By SuperGPQA discipline: (a) mean accuracy, (b) changed choices, (c) both correct and wrong answers, and (d) always wrong with changing candidates. Rates in (c,d) sum to (b). Error bars show pointwise 95% bootstrap intervals. Other disciplines pools five groups with fewer than 30 questions each; subject and difficulty vary together.

Subject profiles also qualify the relatively favorable pooled result for AutoTrust-JEV-27B-v1.1 (Figure 18). It changes 43.9% of choices in engineering and 44.2% in science. Always-wrong switches account for 15.8% and 18.4% of all questions in those disciplines, respectively. Thus, its residual instability includes both lost or recovered correct answers and variation that never reaches the correct answer; the latter cannot be repaired by voting over the recorded choices.

![](images/de31b7d4cfa3afc53c88d7512f977461d3eed400ebcbf26a83a6eb29393d00a7.jpg)

![](images/ff304f0473a2b347a3b5ea487df41acd748fca734e648dff5d19eefd0a6fd627.jpg)  
Figure 18: Jev reproductions switch between wrong answers within subjects. SuperGPQA 1discipline profiles, using the decomposition in Figure 17. Correctness changes and alwayswrong switches partition each model’s changed-choice rate. Error bars show pointwise 95% bootstrap intervals; subject and difficulty vary together.

## 4.3 Locating Instability Within Legal Tasks

LEXam reveals substantial heterogeneity across menu sizes. Larger-option groups combine lower accuracy with more changed choices for every model (Figure 19). Jev’s changed-choice rates rise from 22.4–23.3% in the four-option group to 66.9–74.6% in the 32-option group. Both Jev versions exceed DeepSeek-V4.1-Flash-1017’s 65.3% in the latter group, reversing their favorable comparison on LEXam overall. Qwen3.8-Flash-Next and GLM-5.3-Flash vary even more, changing 87.3% and 97.5% of choices. The same group yields 85.6–98.3 extra wrong pairs per 100 questions for Jev, compared with 162.7 for Qwen3.8-Flash-Next and 230.5 for GLM-5.3-Flash. Larger menus thus identify a particularly unstable portion of this collection, with wrong-option diversity extending well beyond Jev. The groups contain different questions, however, and do not isolate the effect of adding distractors to a fixed question.

Every reproduction changes more choices at 16 options than at eight, and more at eight than at four (Figure 20). AutoTrust-JEV-27B-v1.1 rises from 14.4% to 33.6% between four and 16 options, then reaches 55.2% at 32. Across four to 32 options, pplx-decider-v1.1-27b rises from 11.2% to 76.0%; AutoTrust-JEV-35B reaches 92.0% in the 32-option group. The remaining nine reproductions do not support 32-option menus, leaving their conditional stability metrics undefined.

Number of options  
![](images/cd34730d4449505ad87c255c66c7dfd50d046c3def8fd131b836cec25ee12453.jpg)

![](images/6199a02aa49dc77ff815131ac2384929276b0978831152f0d70b3dd4b3ac78b1.jpg)

![](images/1c18fa37b32362d097dd052b61a941a6c475b07c6b90a790db32e56312d53a96.jpg)

![](images/74b97de0680995b59aae4222cf5544591614160a97d9f87df96cc064a26aa371.jpg)

Figure 19: Larger LEXam menus accompany more varied wrong choices. Option-count 1groups contain 125 questions each; metrics follow Figure 11. Lines connect model estimates, with pointwise 95% bootstrap intervals. Groups contain different questions, so the comparison does not isolate an effect of option count.  
![](images/7822e94c1df11a2201124cfca22ddee2eeeb114ab9614c40cd4e202beb007c1f.jpg)

![](images/1037275fd84b4130ca4fa166dffee43bf55ea3c5a836a2a501cd159912498d13.jpg)

![](images/cae12185fc72945aab7a88ac07e5d17870caec7d2d9f87c0800933ed050757da.jpg)

![](images/bfcf188642cd54ec35bafcbefb4f3ccfdd29d387b9c9a9917413576f3c0abf72.jpg)

![](images/3f73e6cdfffcb8066e22c91f1c889c934f1fa7868ca95634d0cce93e7246b5cb.jpg)  
Figure 20: LEXam exposes both varied choices and menu limitations. Profiles include 4-, 18-, 16-, and 32-option questions, using the metrics in Figure 19. Error bars show pointwise 95% bootstrap intervals. Groups contain different questions.

## 4.4 Option Count and Continued Failure Discovery

Jev’s loss of initial successes persists on both sides of an eight-option threshold (Figure 21). Across the eight presentations, Jev loses 21.7–23.6% of its initially correct answers on questions with fewer than eight options and 26.9–28.0% on questions with at least eight. The comparison with LLMs also varies across these groups. DeepSeek-V4.1-Flash-1017 loses 23.9% and 22.4%, respectively: slightly more than either Jev version in the smallermenu group, but less than both in the larger-menu group. Qwen3.8-Flash-Next loses 35.8% and 33.2%, exceeding both Jev versions in each group. These differences qualify pooled comparisons without isolating an effect of menu size.

Jev reproductions also lose initial successes in both groups (Figure 22). Losses range from 17.8–51.9% below eight options and 25.3–71.6% at eight or more. AutoTrust-JEV-27B-v1.1 has lower loss than either Jev version in both groups. The other eleven reproductions lose a larger fraction of their initial successes than either Jev version in the larger-menu group, showing that the vulnerability extends across the evaluated implementations.

The full option-count profile qualifies the LEXam trend (Figure 23). Jev’s changed-choice rate and loss of initial successes fluctuate across the four- to ten-option groups rather than increasing steadily with menu size. Composition matters: all ten-option questions are from

Jev 1.13 DeepSeek-V4.1-Flash-1017 Jev 1.14 Qwen3.8-Flash-Next

![](images/2206bcd3a6fe8b93358440a048264b25eb1499d9c9133fcf77cf15e5a95715ce.jpg)

![](images/a4fd8670777ab1e0b1027a148c14e433adb7d1d217996bece1d92b0172ed1d9f.jpg)

Figure 21: Jev loses initial successes in both option-count groups. Retention through each 1presentation for (a) 437 questions with fewer than eight options and (b) 1,263 with at least eight. Each curve is normalized by the model’s initially correct answers in its group.  
![](images/5d60009df7b4c4506a2061d16c42aad0a2043e5ca896e76b5bd44f8685b46097.jpg)

![](images/baed86be24080b9eb1f0e818c538614b10fee63ea1c993aab688c35952dc4f3b.jpg)

![](images/e199ba5b8c09c039c7a59fd8b81ee7d5b7a206967c2056d1d245f78be13e184b.jpg)  
Figure 22: Jev reproductions lose correct choices even with at least eight options. Groups and metric follow Figure 21.

SuperGPQA, while all 16- and 32-option questions are from LEXam. The profile identifies vulnerable groups without estimating what would happen if distractors were added to the same question.

![](images/a7169c0e89049a71defb52b49c3efd2384a289fe831755ff430f06dcf560e17b.jpg)

![](images/29ee3f704fcb6c3d1efff7025a04ea9410208f2acb67f41dcbb9a45b845c8811.jpg)

![](images/35f49cb02ebbf8714e0e21878e868d206002773ea60dcd0b315acc1a13b7cecc.jpg)

![](images/f97aaaac97c8d7266f0a0d4508ce287934a581fd7804991eb45c27c872418da0.jpg)  
Number of options

Figure 23: Jev’s instability has no uniform trend across menu sizes. All observed option 1counts: (a) mean accuracy, (b) changed choices, (c) loss conditional on initial success, and (d) newly failed questions after O per 100 questions, counted once. Error bars show pointwise 95% bootstrap intervals. Benchmark and difficulty composition differ across groups.

(d) New failures / 100  
![](images/86d785ed45e36e236d7575ff3b56440c7383e8db3015281342b29a2d54b5adcb.jpg)

(a) Mean accuracy (%)  
![](images/d23ff27dc89de87aa8497b34fca1b0ed05316ad7d28e829419fbfb5f73aa2ee7.jpg)

![](images/b8e0a431edb53f86c515cb921df86b34b3fe2ba31bfda3b4ae2684300b2c4733.jpg)

![](images/c4367795cabdafb4ece74593a5811176476af68ed535de1086bc9de50aec28b3.jpg)  
Number of options

![](images/4cb369b067a179c03fb57a943b5f90a8ca30e05e5a09ff808ace618008802cd8.jpg)  
Figure 24: Jev reproductions fail across the full range of menu sizes. Four to 32 options; 1metrics follow Figure 23. Error bars show pointwise 95% bootstrap intervals. Benchmark and question content differ across groups.

The pooled profile of Jev reproductions similarly prevents a simple ranking by menu size (Figure 24). AutoTrust-JEV-27B-v1.1 loses 25.6% of initial successes on ten-option questions and 22.6% on 16-option questions, although its LEXam-only turnover increases across menu sizes. These groups differ in both benchmark and question content.

Small menus do not make repeated reordering uninformative. A four-option question has 4! = 24 full permutations, so eight presentations need not repeat the same menu order. In the four-option group, 199 of 203 questions have eight distinct observed orders and the other four have seven. Jev 1.13’s cumulative failed-question count rises from 76 to 95, and Jev 1.14’s from 73 to 92. Presentations five through eight still expose four and two first failures, respectively. Counting only the first occurrence of each full permutation leaves both Jev failure trajectories unchanged. These distinct full orders need not place the correct answer in distinct positions. The curves measure robustness across the sampled menu orders, without assuming a new correct-answer position on every presentation.

The number of new failures must also be read alongside the pool of initial successes. On 32-option questions, Jev loses 43.5–70.0% of initial successes, but this amounts to only 8.5–11.9 newly failed questions per 100. The ten-option rates are 25.2–27.4% and 14.5–15.9 per 100. Low initial accuracy leaves fewer successes available to fail, so a smaller cumulative increase can accompany greater conditional fragility. Both measures are needed to avoid mistaking a small remaining pool of correct answers for robustness.

## 5 Can Aggregating Reordered Choices Correct Jev’s Errors?

The same repeated answers that expose failures might also help correct them. Plurality voting could recover a correct answer that appears across several orders, while unanimity could identify a more reliable subset. We test both possibilities, asking whether variation supplies useful correction and whether agreement warrants confidence (Figure 25).

## 5.1 Voting Recovers Only Part of the Error

Combining eight reordered answers gives Jev only a modest accuracy gain: 0.81 percentage points for version 1.13 (95% interval 0.17–1.46) and 0.93 for version 1.14 (0.27–1.60). The reference is mean single-presentation accuracy, avoiding dependence on an unusually weak original round. LLM gains range from 0.42 points for GPT-6.1-sol to 5.03 for GLM-5.3-Flash, with positive lower bounds for all six intervals. Qwen3.8-Flash-Next and GLM-5.3-Flash gain 3.23 points (2.43–4.05) and 5.03 (4.04–6.00), respectively. Their lower pooled retention thus coexists with more effective correction through voting.

![](images/cf5825de5619a4a04ff39a5952a9c97e15ccca6451029dcab995ac56964958f4.jpg)

![](images/7b4e11cd910000492821cc0b7d63da6ca1207e80f471503a6e8a7c2bfe6b0364.jpg)

![](images/201edb1fda0ee784bcb9ada77c562d6e79185f337558a7fff6c5d7ef3464ec3c.jpg)

![](images/64f3d0d60ba0485f0cce7009aa2393cadadbd22cb3c6b5f91d504ce56146c715.jpg)

Figure 25: Voting offers limited Jev correction; unanimity still includes errors. Across 1eight orders: (a) mean accuracy, (b) plurality gain in percentage points, (c) unanimous coverage, and (d) accuracy among unanimous answers. Error bars show pointwise 95% paired source-group bootstrap intervals, stratified by benchmark. Plurality ties use uniform tie-breaking in expectation.  
![](images/f4a723db55413b98a6e0e022c3559f5792417927507763c742e03ed8ed25565f.jpg)

![](images/999b16da8724e539334297ebeaa5bad4b1cd6c479c02d7a1b497c414e0182627.jpg)

![](images/5022b0f4b9adca8ec9492a1091df76a01244f29c1d4d16a279be14179ae2fe6f.jpg)

![](images/83b68679e656ba9e94e76609962482de2b03dafc342d0af612423683c572432c.jpg)

![](images/79db140d1899fdd07a28ff2b5a4244d8628442814a57ac4b9703f2ba49b75bc6.jpg)  
Figure 26: Voting and unanimity provide limited safeguards for Jev reproductions. Metrics 1follow Figure 25, with accuracy and coverage over all 1,700 questions. Unanimous accuracy ranges from 30.2% to 66.3% at differing coverage. Error bars show pointwise 95% paired bootstrap intervals; plurality ties use uniform tie-breaking in expectation.

The trajectories help explain why correction is limited. Jev 1.14 gains only 0.27 points on LEXam ( 0.75–1.28), where many changed choices remain wrong throughout. Such trajectories supply no correct vote; even when the correct answer appears, it must receive enough votes to defeat the alternatives. Qwen3.8-Flash-Next and GLM-5.3-Flash gain 2.90 points (1.19–4.42) and 4.19 (2.35–6.04) on LEXam, while the GPT-6.1-sol and Gemini 3.8 Flash intervals span zero. The usefulness of variation depends on which candidates are selected and how often, rather than on the amount of disagreement alone.

The eight-call comparison further limits the practical benefit of Jev’s variation. Voting gain, measured relative to each condition’s mean single-request accuracy, is only 0.75 points higher under reordering for Jev 1.13 (95% interval 0.06–1.49) and 0.96 higher for Jev 1.14 (0.24–1.71), alongside a roughly 24-point difference in changed-choice rates. The direct differences in plurality accuracy between conditions have intervals spanning zero for both versions. Thus, the greater variation under reordering accompanies a small additional voting gain relative to a single answer, without establishing higher final voting accuracy than original-order repetition.

Voting gains for Jev reproductions are also modest despite substantial turnover (Figure 26). They range from 0.74 points for AutoTrust-JEV-27B-v1.1 (95% interval 0.18–1.31) to 2.49 for StartLux-Decision-v1-35B-A3B (1.71–3.32), with positive lower bounds for all twelve models. Reordering provides some recoverable correct choices, but the gains remain small relative to the widespread changes it exposes.

## 5.2 Agreement and Persistent Error

Obs. 7. More than one third of unanimous Jev answers are wrong. Unanimity raises Jev’s conditional accuracy but leaves substantial error among the retained answers. The two versions select the same candidate in all eight presentations on 59.9–61.0% of questions, with accuracy of 63.2–63.9% on this subset. Thus, 36–37% of unanimous answers are wrong. These persistent mistakes account for 21.6–22.4% of all paired questions, each receiving the same wrong Jev answer eight times. They survive even the strongest agreement requirement available in the recorded trajectories. Such mistakes are not, by themselves, evidence of order bias; they show why surviving the reordering test is insufficient to certify a correct decision.

The same unanimity rule retains more accurate answers for each LLM, although at different coverage. The retained LLM answers are 17.6–21.4% wrong, compared with Jev’s 36–37%. GPT-6.1-sol has the broadest coverage at 82.6%, with 78.6% conditional accuracy, while DeepSeek-V4.1-Flash-1017, Claude Sonnet 5.5, and Gemini 3.8 Flash retain 65.5–79.0% of questions at 79.5–81.1% accuracy. Qwen3.8-Flash-Next and GLM-5.3-Flash retain fewer questions than Jev, at 49.2% and 37.3% coverage, but reach 81.8% and 82.4% accuracy. Agreement improves conditional accuracy within each model, yet the surviving Jev decisions remain substantially less reliable at these observed operating points. Differences in base accuracy and coverage prevent this comparison from isolating the predictive value of agreement or establishing a ranking at matched coverage.

The Jev reproductions reinforce this limitation of unanimity. Their unanimous subsets reach only 30.2–66.3% accuracy, with coverage ranging from 24.1% to 63.7% of the full collection. AutoTrust-JEV-27B-v1.1 has the highest coverage, yet 42.4% of its unanimous answers are wrong; these persistent errors account for 27.0% of all 1,700 questions. Across the twelve models, 12.1–27.0% of questions receive the same wrong answer in every presentation. High agreement under fixed-order repetition therefore cannot certify correctness either before or after reordering.

## 6 Does Higher Accuracy Protect Correct Choices?

Does higher accuracy protect Jev’s successful decisions? MMLU-Pro provides a setting where Jev answers most questions correctly, allowing us to examine whether those successes also survive reordering. Accuracy uses all 300 questions. The other panels use each model’s fixed set of questions with selections in all fifteen distinct requests. These sets can differ across models, so the trajectories support within-model comparisons rather than rankings on identical questions.

Obs. 8. Higher Jev accuracy does not eliminate the loss of initial successes. Jev 1.13 and Jev 1.14 average 86.4% and 86.0% accuracy over the reordered presentations, with ranges of 3.7 and 3.3 percentage points (Figure 27(a)). Yet their cumulative failure sets grow from 35 to 58 and from 36 to 63 questions, respectively. The additional 23 and 27 failures represent 8.7% and 10.2% of initially correct answers, compared with losses of only 1.1% and 2.7% under original-order repetition. Each version first fails on two more questions in the final presentation. The loss is smaller than on the primary collection, but it persists even when Jev answers most questions correctly.

All twelve Jev reproductions repeat exactly the same candidate across eight original-order requests, yet lose 8.6–45.0% of initial successes under reordering (Figure 28). Thus, the separation between repeatability and preservation of correct choices persists on a different benchmark. As in the primary collection, the comparison describes the evaluated systems without isolating an architectural cause.

Jev’s unanimous answers are more accurate here than on the primary collection, but agreement still does not certify correctness. They cover 88.3% and 86.0% of questions and reach 91.3% and 91.9% accuracy (Figure 27(d)); 8.7% and 8.1% of the retained answers are still wrong. Thus, both the loss under reordering and the reliability of agreement depend on the task collection. What transfers is the distinction among average accuracy, fixed-order repeatability, and the survival of individual correct decisions.

![](images/3c58f0ee95d0077a428e79c61748b352ab9e0a10843b38a9216ba31d63c6ce6f.jpg)

Figure 27: Jev’s higher MMLU-Pro accuracy still leaves correct answers vulnerable. (a) 1Accuracy on all 300 questions; (b) cumulative failed questions; (c) loss of initial successes after eight original-order or reordered requests; (d) coverage and accuracy of answers unanimous across reordered presentations. Panels (b–d) use each model’s fixed selection set defined in Section 6.  
![](images/c3066639ca2abfa56d6ddd705c5e88b780617707463d019a5afd37c4407437c3.jpg)

![](images/12898b19bc6245b9481c8a8d0543b6292525b048f725d5bc82f65e2f37af2675.jpg)

![](images/41c27c8e9f683217b054f71a8690a5e9c258cb4fdc7b061b4afe074be7e4d84f.jpg)

![](images/836efcd4d7f1cd5f645d1280298eb2652dcda25b6505cc3320e9389bfe0b991b.jpg)

![](images/c1ddde1c826b63b2f97045b822cd5f6d348444aa2f88cfc6ce785b558f8e2e8d.jpg)  
Figure 28: Perfect repetition does not protect Jev reproductions from reordering. MMLU-1Pro results for twelve Jev reproductions with both Jev versions as references, using the panels in Figure 27. All panels use the complete 300-question sample.

## 7 Does a Four-Option Menu Remove the Vulnerability?

Four candidates still allow many rearrangements, so a short menu need not make repeated reordering uninformative. GPQA Diamond tests whether Jev’s loss of initial successes persists when every question has only four options. Figures 29 and 30 use all 198 questions and the common eight-presentation protocol.

![](images/58b17b4df6eb227640c5846d2a639d857476d62a3fea2a1f4594ceee6e05a1a4.jpg)

Figure 29: Four candidates still leave correct Jev choices vulnerable. GPQA Diamond: 1(a) accuracy, (b) cumulative failed questions, (c) initial-success loss under repetition and reordering, and (d) distinct wrong question–candidate pairs. All panels use the full 198 questions.  
![](images/f9c17689a6308f65834d23ee8713ebc8c75ca3f5b5f6091942756d558aa4bf61.jpg)  
Figure 30: Perfect repetition coexists with lost successes on four-option questions. GPQA 1Diamond results for Jev reproductions, with both Jev versions as references. Panels and denominators follow Figure 29.

Obs. 9. Jev’s loss of correct choices is not confined to long candidate menus. Jev 1.13 and Jev 1.14 average 76.6% and 72.6% accuracy, yet their cumulative failure sets grow from 52 to 76 and from 58 to 89 questions. Reordering loses 16.4% and 22.1% of their initial successes, compared with 4.1% and 7.1% under original-order repetition. All twelve Jev reproductions repeat their original choices perfectly but lose 23.7–66.7% of initial successes after reordering. The gap between repeatability and preservation of successful decisions therefore extends to four-option questions. It does not require the larger menus in the primary collection.

The observed loss of initial successes varies across the LLMs, from 3.3% for GPT-6.1-sol to 21.7% for Qwen3.8-Flash-Next and 44.9% for GLM-5.3-Flash. These results do not establish a uniform family ranking. Repeated errors also need not select the same distractor. Jev accumulates 95 and 112 distinct wrong question–candidate pairs across its 76 and 89 failed questions. The reproductions accumulate 142–281 such pairs. Excluding the two questions with duplicate distractor text preserves the Jev repetition–reordering gap. Short menus therefore remain informative about the survival of correct choices, although comparisons across benchmarks do not isolate candidate count from content or difficulty.

## 8 Implications and Scope

What the Jev comparison establishes. Our results provide no evidence of a general Jev advantage in preserving correct choices across option orders. Four LLMs retain more shared initial successes (Obs. 2), but Qwen3.8-Flash-Next, GLM-5.3-Flash, and the taskdependent comparisons rule out a universal ranking. The contribution beyond pairwise order-sensitivity studies (Lukassen et al., 2026) is to trace the cumulative consequences: which initial successes are lost, how widely failures spread, and whether further orders still reveal first failures. The magnitude and timing of that loss carry the evidence; monotonic growth of a cumulative set does not.

The Jev reproductions show that a shared decision interface can coexist with substantially different robustness. On the primary collection, AutoTrust-JEV-27B-v1.1 retains more initial successes than the Jev references, whereas the other eleven reproductions retain fewer. Their near-perfect fixed-order agreement does not predict this ordering. Explaining the differences requires isolating the choices in representation, learning, and computation that the shared interface leaves unspecified.

Reordering diagnoses failures that agreement may not repair. A useful diagnostic need not be an effective correction. Reordering exposes lost successes and multiple wrong candidates, but voting yields only a small accuracy gain for Jev. Always-wrong trajectories contain no correct vote, and unanimous errors pass an agreement filter unchanged (Obs. 7). These persistent errors do not themselves demonstrate order bias; they limit agreement as a safeguard. Nor does observing several wrong candidates establish what would happen after one is removed. An adaptive masking study could compare removing the previously selected wrong candidate with removing a random distractor at matched menu sizes, separating targeted correction from the benefit of fewer choices.

Behavioral differences do not identify their mechanism. Original-order repetition characterizes variation without changing the menu. Its comparison with reordering describes behavioral differences without isolating a causal order effect. Candidate identifiers travel with their content while entry order changes, leaving the source of presentation sensitivity unresolved. Prior work distinguishes identifier preferences from position and context effects (Azizi et al., 2026; Zheng et al., 2024); our experiment neither separates these components nor establishes how a proprietary interface represents them.

Training, inference, and interface also differ across models, precluding a causal claim about decision heads or generation. The literature offers starting points for controlled comparisons: frozen LLM readouts keep the backbone fixed (Li & Wagle, 2026), and serialization interventions keep weights fixed (Azizi et al., 2026). Further comparisons could vary decision-specific adaptation or candidate interaction separately. The supporting survey organizes these distinctions alongside supervision, gradient paths, and execution. The present findings concern observed selections, not a recovered account of Jev’s proprietary mechanism.

The task mixture limits generalization. The results cover fixed task collections and eight presentations. Domain, language, difficulty, and candidate count vary together, while subgroup intervals are exploratory and do not adjust for multiple comparisons. MMLU-Pro and GPQA Diamond broaden the evidence, but differences in loss magnitude show its dependence on the collection. Continued first failures establish persistence within the observed sequence without predicting growth under further requests.

## 9 Related Work

Selection bias and its sources. Early work on contextual calibration shows that language models carry answer preferences that can destabilize few-shot predictions as prompts and demonstrations change (Zhao et al., 2021). Option reordering provides a more direct test when question and answer content are held fixed (Pezeshkpour & Hruschka, 2024). Position and identifier effects must nevertheless be distinguished. Zheng et al. (2024) identify preferences for option-ID tokens as a major source of multiple-choice selection bias, while FairEval and MT-Bench document response-order effects in pairwise judgments (Wang et al., 2023; Zheng et al., 2023). In their study of many-shot evaluation, Song et al. (2025) likewise find that changing response order or the symbols assigned to responses alters GPT-4o’s pairwise judgments; they combine evaluations across both transformations to mitigate these biases. Shi et al. (2025) further separate repetition stability, position consistency, and preference fairness. Our candidate identifiers move with their content, so the present experiment measures behavior across reordered menus without independently identifying these sources.

Mitigation through calibration, training, and reasoning. Contextual calibration and PriDe estimate answer priors to reduce bias (Zhao et al., 2021; Zheng et al., 2024). CalibraEval extends label-free calibration to LLM judges through an order-preserving transformation of prediction distributions (Li et al., 2024). Training offers another route. Strengthened symbol binding improves the association between an option and its identifier (Xue et al., 2024), while permutation-aware GRPO rewards consistency across rearrangements of the same instance (Zheng et al., 2026). Recent work also combines a probability-based permutation metric with LoRA fine-tuning and efficient aggregation (Guda et al., 2025). These interventions show that selection bias is responsive to both training and inference procedures. Evidence of reduced position bias in reasoning models on multi-document question answering further motivates comparison with current LLMs (Li et al., 2026). We evaluate the deployed systems without applying an additional debiasing method to any of them.

Jev-style modeling and selection bias. Methods for reproducing Jev-style decisions differ in candidate representation, readout, and learning. LLM-as-Jev can even extract candidate probabilities from frozen causal LLMs (Li & Wagle, 2026), so the decision formulation alone does not identify an architectural or training intervention. Lukassen et al. (2026) compare eight typed decision models with seven LLM judges in pairwise evaluations, finding sensitivity to swapping two responses despite strong repeatability. Azizi et al. (2026) trace option-label effects to serialization in several public Jev-style implementations through interventions with weights fixed. These studies already establish that typed decisions need not be unbiased. We examine the cumulative consequences in multiple-choice tasks with four to 32 candidates. Repeated permutations reveal how widely failures spread, how much initial success survives, and whether voting or agreement can recover reliable decisions. Fixed-order repetition and contemporary LLM baselines place those losses in context.

Consistency, correctness, and aggregation. MT-Bench cautions that a judge can be consistently wrong (Zheng et al., 2023). GRP evaluates each response pair in both orders and counts it as correct only when both judgments match the gold preference (Song et al., 2026). Asking for the worse response improves this joint criterion, without isolating gains in order agreement from gains in per-trial correctness. Ok et al. (2026) distinguish answer flipping from positional skew and show that aggregation can reduce flipping partly by stabilizing an incorrect answer. Their decomposition applies under balanced rotations, which our sampled permutations do not enforce. Likewise, the permutation metric of Guda et al. (2025) measures changes in candidate probabilities, whereas our analysis follows returned selections. We distinguish variation under repeated requests from variation under reordered menus, then test whether voting corrects errors and whether agreement identifies correct answers for Jev and the LLM baselines.

## 10 Conclusion

A simple reordering test reveals a limitation of Jev-style System One models: stable accuracy and repeatable choices can conceal the loss of successful decisions. Across eight presentations of the primary collection, the two Jev versions’ failure sets expand by 27–29% while accuracy remains nearly unchanged. These losses are substantially smaller under originalorder repetition, a gap also observed in highly repeatable Jev reproductions. Comparisons with LLMs establish no general Jev advantage. MMLU-Pro and GPQA Diamond show that the vulnerability persists at higher accuracy and with only four candidates. Voting offers limited correction for Jev, while unanimous errors show that agreement alone cannot certify correctness. Evaluating Jev-style structure therefore requires following individual correct choices across equivalent presentations; explaining their survival requires controlled comparisons of the designs behind the shared interface.

## References

Guy Amit. Permutation-Robust Decision Modeling with Candidate-Independent Block-Causal Attention, 2026. URL https://arxiv.org/abs/2610.01601. arXiv:2610.01601.

Seyedarmin Azizi, Erfan Baghaei Potraghloo, and Massoud Pedram. Labels Override Definitions in Jev-Style Typed Decision Models, 2026. URL https://arxiv.org/abs/2610.0 2586v1. arXiv:2610.02586v1.

Joas Antonio dos Santos Barbosa. Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents, 2026. URL https://arxiv.org/abs/2609.28940. arXiv:2609.28940.

Jianbo Chen. The Company a Question Keeps: Auditing Open Jev-Style Models for Retrieval, 2026. URL https://arxiv.org/abs/2610.14631. arXiv:2610.14631.

Tianyu Chen, Xinyang Liu, Ruyi An, Zhendong Wang, and Mingyuan Zhou. System1-Arena: Which Decisions Can Be Left to System 1?, 2026a. URL https://arxiv.org/abs/2610.21244 v1. arXiv:2610.21244v1.

Ziyi Chen, Yan Zhang, Jianhui Wei, Jianing Yin, and Daoan Zhang. Large Decision Models: Repurposing Large Language Models as Single-Pass Decision Makers, 2026b. URL https://arxiv.org/abs/2610.15501. arXiv:2610.15501.

Jianpeng Cheng, Guangyu Sun, Aashu Singh, Benyu Zhang, Haixing Dai, Hossein Mansour, Jiangfan Zhang, Shlok Kumar Mishra, Wei Sun, Xuanming Cui, Yanli Liu, Qi Guo, Max Xiangjun Fan, and Jun Xiao. MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface, 2026a. URL https://arxiv.org/abs/2610.11316. arXiv:2610.11316.

Zehua Cheng, Wei Dai, and Jiahao Sun. this-that-model-1.0: A typed decision model that decides in 30 ms, for a millionth of a cent, 2026b. URL https://arxiv.org/abs/2609.23886. arXiv:2609.23886.

Yuyang Dai, Yuting Tang, Futoshi Futami, and Takashi Ishida. ProbabilityBench: Model Evaluation and Calibration with Exact Ground-Truth Uncertainty, 2026. URL https:// arxiv.org/abs/2610.17016. arXiv:2610.17016.

Feiyu Duan, Jiayu Lin, Jia Wang, Jun Xiang, Jialiang Wu, Xinnong Zhang, Hanqi Yan, Siyuan Wang, and Zhongyu Wei. General Decision Models: Benchmarking and Insights Beyond Jev, 2026. URL https://arxiv.org/abs/2610.03935. arXiv:2610.03935.

Yu Fan, Jingwei Ni, Jakob Merane, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Etienne Salimbeni, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, and Joel Niklaus. LEXam: Benchmarking Legal Reasoning on 340 Law Exams. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=xNhbMyXsJn.

Ying Fang, Zhifu Gao, Qing Gu, Jiajun He, Kangqi Jing, Xiangang Li, Yu Pu, and Wupeng Wang. Qwen-Audio-Jev: A Unified Interface for Audio Decisions, 2026. URL https:// arxiv.org/abs/2610.21614v1. arXiv:2610.21614v1.

Shuyu Gan, Young-Jun Lee, and Dongyeop Kang. SanSi: A Looped Typed Decision Model for System 1.5 Thinking, 2026. URL https://arxiv.org/abs/2610.07730. arXiv:2610.07730.

Zhimin Gao and Pichao Wang. OpenJev-RLCD: A Working RLCD Implementation, 2026. URL https://arxiv.org/abs/2609.38850. arXiv:2609.38850.

Jonathan Geuter, Andrew Cheng, and David Alvarez-Melis. Releasing the Inner Jev: A Simple Recipe for Open Decision Models, 2026. URL https://arxiv.org/abs/2610.17205. arXiv:2610.17205.

Blessed Guda, Lawrence Francis, Gabrial Zencha Ashungafac, Carlee Joe-Wong, and Moise Busogi. Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA Fine-Tuning and Efficient Majority Voting Approach, 2025. URL https://arxiv.org/abs/2511.2 1709. arXiv:2511.21709.

Yixiao Huang, Hanlin Zhu, Zixuan Wang, Jiantao Jiao, Stuart Russell, Somayeh Sojoudi, and Song Mei. Transformers Provably Learn to Internalize Chain-of-Thought, 2026. URL https://arxiv.org/abs/2605.28600. arXiv:2605.28600.

Dongming Jiang, Yi Li, and Bingzhe Li. Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents, 2026. URL https://arxiv.org/abs/2609.23986. arXiv:2609.23986.

Saman Sarker Joy. Do System One Decisions Add Up? A Study of Probabilistic Coherence, 2026. URL https://arxiv.org/abs/2609.33971. arXiv:2609.33971.

Tianyou Lai, Wentao Yue, Zhiying Deng, and Qilei Li. Correct Choices, Failed Sequences: Choice2Seq Benchmarks Correctness Retention Across Output Interfaces, 2026. URL https://arxiv.org/abs/2610.16191. arXiv:2610.16191.

Yifan Lan, Zhongye Liu, Yurui Chang, Hanyu Wang, Yongkang Du, Lu Lin, and Jinghui Chen. DecisionRobust: Benchmarking Robustness for JEV Style Decision Models Across Text and Vision, 2026. URL https://arxiv.org/abs/2610.19118. arXiv:2610.19118.

Haitao Li, Junjie Chen, Qingyao Ai, Zhumin Chu, Yujia Zhou, Qian Dong, and Yiqun Liu. CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges, 2024. URL https://arxiv.org/abs/2410.15393. arXiv:2410.15393.

Qi Li and Xinchao Wang. Jeva: System One Models from First Principles, 2026. URL https://arxiv.org/abs/2610.18773. arXiv:2610.18773.

Yinheng Li and Justin Wagle. LLM-as-Jev: LLMs Are Already Jev-Style Decision Models – When and How to Fine-Tune Them, 2026. URL https://arxiv.org/abs/2610.02076. arXiv:2610.02076.

Yining Li, Huayang Li, Lemao Liu, and Taro Watanabe. On the Position Bias of Large Language Models: A Comprehensive Study, 2026. URL https://arxiv.org/abs/2610.228 07. arXiv:2610.22807.

Guoming Ling, Muen Xue, and Zijian Ye. Jev in the Wild: A Data-Driven Analysis of the Jev Model’s Functionality, Applications and Ecosystem, 2026a. URL https://arxiv.org/abs/ 2609.30216. arXiv:2609.30216.

Yuchen Ling, Shengcheng Yu, Zhenyu Chen, and Chunrong Fang. When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows, 2026b. URL https://arxiv.org/abs/2610.13092. arXiv:2610.13092.

Fabian Lukassen, Christoph Weisser, and Alexander Silbersdorff. Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges, 2026. URL https://arxiv.org/abs/ 2610.18122v1. arXiv:2610.18122v1.

Yang Luo, Tiantian Ji, Zifeng Kang, Yucheng Wang, Yong Liu, Zhiyi Yin, and Xinran Liu. Nudge the Judge: Uncovering Program Decision Hijacking in Real-World Jev Applications, 2026. URL https://arxiv.org/abs/2610.18612. arXiv:2610.18612.

M-A-P, ByteDance Seed, and 2077AI. SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc., 2025. doi: 10.52202/085713-3766. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/a3c5af1f56fc73eef1 ba0f442739f5ca-Abstract-Datasets\_and\_Benchmarks\_Track.html.

Jie Ma, Jie Gao, Yihang Liu, Zhike Qiu, Junle Li, Chongyi Zhuang, Jiayi Ji, and Xiaoshuai Sun. FastJEV: Understanding Redundancy for Compact JEV Inference, 2026a. URL https://arxiv.org/abs/2610.11379. arXiv:2610.11379.

Yi Ma, Tianpei Yang, Yaodong Yang, Weixun Wang, and Hongyao Tang. Can Jev be Your Q or Policy in Reinforcement Learning?, 2026b. URL https://arxiv.org/abs/2610.11692. arXiv:2610.11692.

Hyunjong Ok, Minjae Park, and Jaeho Lee. One Bias Score, Two Factors: How Mitigations Affect Option Bias in Multiple-Choice Evaluation, 2026. URL https://arxiv.org/abs/2610 .20993. arXiv:2610.20993.

Ujjawal K. Panchal. FunDM: Cheap & Label-Free Decision Models from Frozen Language Models, 2026. URL https://arxiv.org/abs/2610.17175. arXiv:2610.17175.

Pouya Pezeshkpour and Estevam Hruschka. Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions. In Findings of the Associationfor Computational Linguistics: NAACL 2024, pp. 2006–2017. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-naacl.130. URL https://aclanthology.org/2024 .findings-naacl.130/.

Luozheng Qin, Yuhan Zhu, Changhai Zhou, Qian Qiao, Jun Gao, Yuhua Zhou, Harry Zhang, Xiangyu Zeng, Yuxuan Liu, Kaijie Chen, Andrew Chen, and Pony Ma. MJev-Spatial: System One Spatial Intelligence via Native Attention Matching, 2026. URL https://arxiv.org/abs/2610.22276v1. arXiv:2610.22276v1.

Dushyant Rajput, Nirdesh Chauhan, and Siddharth Kosaraju. Does a Hosted Decision Model Respect Irrelevant Alternatives? A Pre-Registered Test of Option-Set Dependence in Jev, 2026. URL https://arxiv.org/abs/2610.16655. arXiv:2610.16655.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. arXiv preprint arXiv:2311.12022, 2023. URL https://arxiv.org/ abs/2311.12022.

Rahul Sharma, Andrew B. Duncan, Gaétan Marceau Caron, and Sebastian J. Vollmer. TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models, 2026. URL https://arxiv.org/abs/2610.11392. arXiv:2610.11392.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 292–314, 2025. doi: 10.18653/v1/2025.ijcnlp-long.18. URL https://aclanthology.org/2025.ijcnlp-long.18 /.

Mingyang Song, Mao Zheng, and Xuan Luo. Can Many-Shot In-Context Learning Help LLMs as Evaluators? A Preliminary Empirical Study. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 8232–8241, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025 .coling-main.548/.

Mingyang Song, Mao Zheng, and Xuan Luo. GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs, 2026. URL https://arxiv.org/abs/2503.06139v2. arXiv:2503.06139v2.

Lijuan Tang and Yuemeng Zheng. Typed Decision Models: An Early Evidence Audit and Evaluation Checklist, 2026. URL https://arxiv.org/abs/2609.32160. arXiv:2609.32160.

Yao Long Teng, Jianguo Huang, Jiayi Cai, Yanchen Deng, and Bo An. Prism-1: Position- and Rival-Independent Scoring for System One Models, 2026. URL https://arxiv.org/abs/26 10.22066v2. arXiv:2610.22066v2.

Kaixiang Wang, Zhaijiacheng Zhou, Jiong Lou, and Jie Li. Jev-LiteMAS: Separating Decisions from Generation in Multi-Agent Workflows, 2026a. URL https://arxiv.org/abs/2610.214 11v2. arXiv:2610.21411v2.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large Language Models are not Fair Evaluators, 2023. URL https://arxiv.org/abs/2305.17926v2. arXiv:2305.17926v2.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. arXiv preprint arXiv:2406.01574, 2024. URL https://arxiv.org/abs/2406.01574.

Yuxiang Wang, Shengbo Cai, Zhixian Zhao, Yingda Shen, Yuancheng Wang, Xueyao Zhang, Liqiang Zhang, Teddy Sun, Steve Yves, and Zhizheng Wu. Jarvis: Coordination Deserves a Brain of Its Own in Agentic Interactive Models, 2026b. URL https://arxiv.org/abs/26 10.23804v1. arXiv:2610.23804v1.

Zexiao Wang, Zihao Zhang, Xudong Wang, Pan Wang, Ziyi Ye, Haoyu Zhao, Zuxuan Wu, and Shuicheng Yan. Chinese-Jev: Bringing System One Model to Chinese-Language Tasks, 2026c. URL https://arxiv.org/abs/2609.36965. arXiv:2609.36965.

Yuchao Wu. SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV, 2026. URL https://arxiv.org/abs/2610.22588v1. arXiv:2610.22588v1.

Mengge Xue, Zhenyu Hu, Liqun Liu, Kuo Liao, Shuang Li, Honglin Han, Meng Zhao, and Chengguo Yin. Strengthened Symbol Binding Makes Large Language Models Reliable Multiple-Choice Selectors. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4331–4344. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.237. URL https:// aclanthology.org/2024.acl-long.237/.

Yiqu Yang, Qiaoyuan Zheng, Yang Tian, Yu Fan, and Elliott Ash. TUPLE-Bench: Evaluating and Mitigating Privacy-Utility Trade-offs in Typed Decision Models, 2026. URL https:// arxiv.org/abs/2610.23668v1. arXiv:2610.23668v1.

Furkan Yilmaz, Habibe Aleyna Tasdemir, and Muhammed Faruk Gozay. LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information, 2026. URL https://arxiv.org/abs/2609.30706. arXiv:2609.30706.

Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling System 2 into System 1, 2024. URL https://arxiv.org/abs/2407.06023. arXiv:2407.06023.

Yundaichuan Zhan, Weishi Wang, Wenbiao Liu, Daniel Dahlmeier, Chengwei Qin, Juncheng Li, Fredrik D. Johansson, and Zhongqi Yue. Dyad: Extending Large Language Models with Native Typed Decision-Making, 2026. URL https://arxiv.org/abs/2609.36116. arXiv:2609.36116.

Jiahao Zhang, Yifan Zhang, and Yu Huang. JevCoder: When Does Decision Delegation Pay Off in Code Repair?, 2026. URL https://arxiv.org/abs/2610.19596. arXiv:2610.19596.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate Before Use: Improving Few-Shot Performance of Language Models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 12697–12706. PMLR, 2021. URL https://proceedings.mlr.press/v139/zhao21c.html.

Bin Zheng and Zheng Chen. Selective Reasoning for Typed Decision Models: When to Start, Read, and Stop, 2026. URL https://arxiv.org/abs/2610.19644. arXiv:2610.19644.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large Language Models Are Not Robust Multiple Choice Selectors. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=shr9PXz7T0.

Jinquan Zheng, Jia Yuan, Jiacheng Yao, Chenyang Gu, Pujun Zheng, and Guoxiu He. Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 35125–35143. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.1621. URL https://aclanthology.org/2026.acl-long.1621/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623. Curran Associates, Inc., 2023. doi: 10.52202/075280-2020. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/91f18a1287b398d3 78ef22505bf41832-Abstract-Datasets\_and\_Benchmarks.html.

Kunlun Zhu, Tao Feng, Xu Han, Heng Wang, Ling Yang, Xiangru Tang, Jiaxuan You, and Heng Ji. JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs, 2026a. URL https://arxiv.org/abs/2610.21412v1. arXiv:2610.21412v1.

Tong Zhu, Zhe Xu, Zefeng He, Yunkai Chai, Xuyang Hu, Dehong Zheng, Yun Luo, Yafu Li, Qi Zhang, and Yu Cheng. Intern-Decision: Efficient Multimodal Decision Making with Autoregressive Masked Language Modeling, 2026b. URL https://arxiv.org/abs/2610.2 2131v1. arXiv:2610.22131v1.

Xialie Zhuang, Zunhai Su, Jia Pan, Ngai Wong, Guanhua Chen, and Chuan-Wei Kuo. StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions, 2026. URL https://arxiv.org/abs/2610.17309. arXiv:2610.17309.

## A Survey of Jev-Style System One Models

## A Survey Overview

The main study shows that Jev’s stable accuracy and high fixed-order agreement can coexist with substantial losses of correct choices under reordering. It leaves the responsible mechanisms unresolved. Jev-style methods share a candidate-distribution interface while differing in representation, learning, computation, and downstream use. This survey organizes those differences to identify what a controlled test of structural robustness would need to vary.

A candidate distribution can be extracted from a frozen language model or learned from labels, contrastive pairs, reasoning teachers, and policy rewards (Geuter et al., 2026; Li & Wagle, 2026; Duan et al., 2026; Gao & Wang, 2026). A direct response can also arise from intermediate exits, recurrent latent processing, or conditional reasoning (Panchal, 2026; Gan et al., 2026; Zhuang et al., 2026). The output format therefore does not specify how a decision is learned or computed.

We organize the evidence around model construction, downstream workflows, and evaluation, building on prior distinctions among decision-model roles (Tang & Zheng, 2026; Ling et al., 2026a). Appendices C and D examine how distributions are learned and computed; Appendix E examines what they control. Appendix F connects local prediction quality to task outcomes, where local savings need not improve overall success (Ling et al., 2026b). Appendix G develops the resulting controlled comparisons.

## B Decision Interfaces and Contribution Boundaries

## B.1 The Decision Contract

A decision interface changes the contract between a model and its caller. Given a state s, question q, and finite candidate set , the model reports

$$
p _ { \theta } ( c \mid s , q , \mathcal { C } ) , \qquad c \in \mathcal { C } .\tag{6}
$$

Because the caller supplies candidates at inference, this contract extends beyond classification into a fixed training vocabulary. Choices, binary judgments, and ordinal scores can all expose such distributions. Constraining the response to the declared output space ensures its type; it does not ensure semantic correctness, calibration, or consistency across questions.

Official Jev combines this interface with a proprietary model and a service that reuses shared state across parallel questions. TypeSafe describes a new architecture and Reinforcement Learning for Calibrated Decisions (RLCD), but the public documentation available at the evidence cutoff does not disclose a reproducible architecture, training distribution, or exact objective.<sup>6</sup> Consequently, public alternatives reproduce aspects of behavior or explore independently specified designs. They do not establish that Jev’s internal algorithm has been recovered.

Model version and service behavior are also distinct. Shared-state execution, candidate limits, scalar-score conventions, and calibration transformations can change behavior behind the same interface.<sup>7</sup>

We cover publicly documented finite-candidate models, selected multimodal extensions, and downstream decision workflows available by October 19, 2026, identified through arXiv, citation tracing, official documentation, and linked releases. Appendix I summarizes additional implementations.

## B.2 The Object of Contribution

A paper’s contribution depends on what its Jev call does (Table 1). AutoTrust uses Jev to obtain training targets for a replacement model; Jev-Mem uses it to control memory operations. An integration study may instead evaluate an existing workflow. These roles can overlap: CLM and FunDM each provide a model recipe and a coding-run selector, JEVal provides a benchmark and SII, and LAVOIR extends Laya while defining an informationacquisition policy.

<table><tr><td>Contribution</td><td>Question answered</td><td>Examples and boundaries</td></tr><tr><td>Model construction</td><td>How is a decision model obtained, and Jeff, SII, LDM, AutoTrust; what does its inference compute?</td><td>CLM/FunDM model recipes. Jev can be a target behavior, teacher, or baseline.</td></tr><tr><td>Downstream workflow</td><td>What does the model control in task execution or in learning another agent?</td><td>Jev-Mem, Ultrafast, LAVOIR interaction; run selectors and frozen-Jev RL assistance. The model need not be proprietary Jev.</td></tr><tr><td>Evaluation and analysis</td><td>Which capabilities, system outcomes, or usage patterns are supported by evidence?</td><td>JEVal benchmark, agent integration studies, coherence tests, ecosystem surveys. Evaluation is distinct from workflow invention.</td></tr></table>

Table 1: Three levels of contribution. A work may contribute at more than one level.

Existing surveys already recognize distinct uses of typed decision models. Tang & Zheng (2026) discuss roles including controllers, judges, and replacements for individual steps, while Ling et al. (2026a) distinguish application domains from decision purposes in public repositories. We build on that distinction and focus on how specific model mechanisms connect to specific workflow operations.

## C Constructing Decision Distributions

Candidate probabilities depend on the readout, the information shared among candidates, the supervision target, and the gradient estimator. These choices determine how order can enter the prediction and which interventions can change it. We examine each before turning to inference in Appendix D.

## C.1 Decision Readout Before Specialized Training

Takeaway 1. A direct decision interface can expose a frozen model’s existing predictions. A language model’s existing probabilities can be restricted and normalized over a candidate menu without updating its weights. This frozen readout provides a baseline for assessing what specialized adaptation adds, even when its predictions remain imperfect.

The simplest route reads an existing model’s candidate-token logits. The frozen baseline in Inner Jev renders alternatives with answer letters, disables thinking, and applies a softmax over the corresponding original language-model head rows (Geuter et al., 2026). Writing these logits as z<sub>k</sub>, its calibrated readout is

$$
p _ { k } = \frac { \exp ( z _ { k } / T ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { j } / T ) } .\tag{7}
$$

No model weights need change, although fitting T still requires calibration data. SemIf and several community implementations use closely related answer-logit extraction. This route is an essential control for attributing improvements to specialized training: comparison against an LLM that must generate probability strings measures a different inference procedure.

LLM-as-Jev removes the restriction to a fixed pool of single-token labels without altering the architecture or tokenizer (Li & Wagle, 2026). A prompt ending in Best answer: [ assigns candidate k the suffix k]. The closing bracket makes these suffixes prefix-free. For suffix tokens $c _ { k , 1 : m _ { k } } ^ { } ,$ , it computes

$$
s _ { k } = \sum _ { t = 1 } ^ { m _ { k } } \log p _ { \theta } ( c _ { k , t } \mid x , c _ { k , < t } ) , \qquad p _ { k } = \frac { e ^ { s _ { k } } } { \sum _ { j } e ^ { s _ { j } } } ,\tag{8}
$$

where x contains the shared context and all alternatives. Prefix caching and parallel teacherforced scoring make this inexpensive, but the paper describes generally one to three short forward steps after prefill. “No free-form answer generation” is therefore more accurate than a universal single-pass claim. The same framework can also inherit explicit thinking from its backbone.

![](images/d6573a100c807c593daf2c02522d3bf63590c0187279938a2f7245cc1047fc69.jpg)

![](images/a8ea4164e1ed202b5c0ded3a1075d0db55892bcdef19feed8f7938015c0a4c7c.jpg)  
Score every suffix token; reuse the shared prefix.

![](images/f753594955f5282fa157fe45a6c813650620cbe4c2df06f372ac87faf0f62ca5.jpg)

![](images/d71f790173c5ec35e8fcec4039087afae98a2fd36bee87a09e38eaab87a2ec33.jpg)  
Figure 31: Four readouts for candidate probabilities. (a) Vocabulary rows, (b) complete suffixes, (c) joint-context option states, and (d) separately encoded states and actions. Suffix token boundaries are illustrative. Across the mechanism figures, green denotes model computation, gold supervision or external feedback, slate operations, and gray data or memory. Labels specify training status. Sources and objectives appear in Appendix C.3.

## C.2 Candidate Representations, Interaction, and Reuse

Takeaway 2. Reusable candidate representations can support set-dependent decisions. Candidate representations can depend on the request, rival options, and serialization position. Caching constrains their computation but not how later layers combine them: reusable options can support set-dependent decisions, while request-conditioned options can remain isolated from rivals (Figure 32). Order invariance and sensitivity to meaningful set changes are therefore separate properties.

Independent scores with reusable candidate representations. CLM independently encodes state–instruction and action inputs, then compares their projected representations. If these inputs and the temperature remain fixed, adding an alternative changes the softmax denominator but not the odds between two existing alternatives. MetaEncoder also uses a shared bi-encoder and contrastive training, but supports multimodal requests and updates the encoder (Cheng et al., 2026a). Its request-to-candidate InfoNCE is unidirectional, unlike CLM’s bidirectional objective. Moreover, its closed-set request template can include the alternatives, which are randomly dropped during training. Candidate features remain request-independent, but the request representation need not be independent of the set. Thus a bi-encoder label alone does not establish invariant pairwise odds.

Reusable candidates with set-dependent decisions. Jeva independently encodes the request and each option, then refines only the request state against immutable option memory (Li & Wang, 2026). Candidate representations exclude the current request, their position, and other options, enabling reuse across requests. A compact decision stack nevertheless allows the final request state to depend on the entire option set. Shared scoring and symmetric aggregation are designed to make the resulting probability vector permutation-equivariant, while allowing the relative ranking of existing options to change when the set changes. Jeva further separates competition among options from coverage: for closed-set probabilities $q _ { i }$ and estimated coverage $\rho ,$ the open-set output is $p _ { i } = \rho q _ { i }$ with $p _ { \emptyset } = 1 - \rho .$ . A diffuse q expresses uncertainty among options; low $\rho$ expresses possible answer absence. Their empirical quality still depends on learning and evaluation.

Request-conditioned candidate blocks. Candidate-independent block-causal attention offers a third design (Amit, 2026). A causal request prefix is shared; each candidate block attends to that prefix and its own tokens, but not to other candidate blocks. Reset positions and a shared local score head aim to remove serialization-order effects. Candidate states therefore depend on the request and cannot generally be cached across different requests, even though their computation excludes other candidates. This is different from both independent bi-encoding and Jeva’s immutable option memory. The reported residual permutation flips also distinguish a structural design objective from exact numerical behavior. Its frozen contrastive baseline uses a different training setup, so that comparison does not isolate attention structure alone.

Prism-1 uses the same separation of a shared causal prefix and isolated candidate blocks, but scores each candidate with the native language-model head’s yes/no log odds at its final token (Teng et al., 2026). A softmax across these scores produces the choice distribution; LoRA and candidate cross-entropy adapt the model without a separate learned scalar head. Every candidate starts at the same position offset. For a fixed prefix and deterministic computation, isolation and position reuse make scores independent of candidate serialization order and preserve pairwise odds when unrelated candidates are added. The latter is a restriction as well as an invariance: a task requiring an existing option’s score to change with its rivals needs a different dependency structure.

## C.3 Supervision and the Target Distribution

Takeaway 3. The supervision source determines what a fitted decision distribution represents. Labels describe outcomes in the training population, contrastive pairs distinguish matched inputs from chosen negatives, and teacher probabilities reflect the teacher’s information and computation. Similar cross-entropy expressions can therefore fit different quantities.

![](images/3b61309048763def6e96ccfe301e120a446964f75853cd298bc1745719a344bb.jpg)  
Figure 32: Candidate reuse and interaction are separate choices. (a) Independent scoring, (b) set-dependent request refinement, and (c) request-conditioned candidate blocks. Arrows show dependencies; dashed outlines mark representations reusable with fixed inputs and weights. Odds statements fix request text, candidate text, and calibration.

Observed and mixed-label supervision. Jeff adds a residual decision head to the original letter readout and trains that head with backbone LoRA using cross-entropy (Geuter et al., 2026). The supervision mixes ordinary dataset labels, human uncertainty and ordinal annotations, and teacher-provided soft distributions. It is neither a pure hard-label recipe nor an RL method. Per-type temperature fitting follows training. The calibrated frozen model already provides a strong reference on the authors’ decision benchmark, while their knowledge-oriented comparisons still favor Jev. These results motivate a strong baseline; they do not establish equivalence across workloads.

The adapted LLM-as-Jev variant instead preserves the architecture and candidate suffix representation (Li & Wagle, 2026). Its training loss uses a locally normalized candidate-prefix tree plus KL anchors that preserve the frozen model’s behavior on legal/illegal token mass and outside decision positions. That locally normalized training distribution should not be confused with the globally normalized suffix likelihood in Equation (8). The recipe uses public human-annotated training data rather than teacher-synthesized labels and studies both full adaptation and LoRA.

Kev illustrates supervised adaptation with a different readout. Its pointer head projects the decision position and each option boundary into a shared space and normalizes scaled dot products over options. The released recipe uses cross-entropy; smaller models use LoRA, while the documented 27B model uses full-parameter adaptation.<sup>8</sup>

Intern-Decision instead retains the native vocabulary head and fits hard-label, fullvocabulary cross-entropy only at shifted decision positions (Zhu et al., 2026b). Its input contains the complete ordered schema and an assistant JSON skeleton with <decision> placeholders. The logits immediately before each placeholder predict that field; earlier placeholders remain visible without feeding back gold or predicted answers. Inference restricts these logits to legal options and applies temperature scaling. The language parameters are adapted while the vision encoder and projector remain frozen. Training over the full vocabulary and restricting the inference readout are distinct operations, and the shared causal pass does not imply independence or order invariance across fields.

Modality-specific readouts. MJev-Spatial reuses pretrained query/key projections from the final full-attention layer to match a decision state to option-end states, rather than adding a prediction head (Qin et al., 2026). Mean scaled inner products across query heads become logits for one candidate softmax. The retained language parameters are trained with cross-entropy while the vision encoder and merger are frozen. Because option states come from a causal sequence, later options can see earlier ones; shuffling options during training does not establish structural permutation equivariance.

Qwen-Audio-Jev uses a different design: a shared audio–language prefix state feeds three affine heads for 64 Choice indices, one binary Noul logit, and ten ordinal Score indices (Fang et al., 2026). Candidate meanings are specified in the input, and each categorical head is normalized over the valid indices. Supervised training updates the audio encoder, adaptor, backbone, and heads using the observed primitive’s loss: Choice cross-entropy, binary cross-entropy, or ordinal cross-entropy with cumulative-distribution and expected-index penalties.

Contrastive state–action alignment. Contrastive Language Models (CLM) use a frozen Qwen3-8B encoder with learned state and action projection heads.<sup>9</sup> Matched state–action pairs supervise bidirectional InfoNCE: the state retrieves its matched action and the action retrieves its matched state among batch alternatives. At inference, state plus instructions and each candidate are encoded independently, and a softmax over scaled cosine similarities gives the candidate distribution. Independent candidate representations can be cached, although different question instructions change the state encoding. This is learned contrastive alignment rather than teacher-probability distillation.

Reasoning-endpoint distributions. SII-Decision trains a direct readout against a samesize backbone’s reasoning-conditioned predictions (Duan et al., 2026). For each question, two reasoning chains yield two candidate distributions; their mean becomes the cached student target.<sup>10</sup> Cross-entropy to this target is combined with a KL constraint to the original nonthinking prediction, and ordinal tasks also use an expected-level penalty. The student updates language layers while retaining the original label head, and inference uses a single prefill with no generated rationale. This is offline reasoning-to-readout self-distillation, not student on-policy rollout distillation.

Where SII retains reasoning-endpoint probabilities, LDM constructs soft targets from sampled answers (Chen et al., 2026b). In the primary experiments, its SPRD variant draws four reasoning answers from a frozen Qwen3.8-27B teacher for all student sizes and forms a vote histogram normalized by the number of parseable answers. SPRD already requires the top vote share over all four samples to reach 0.75; SPRD-C additionally requires the majority to agree with the label intended during data construction. This checks consistency, not independent correctness. The LDM objective uses supervised NLL on one record type and weighted teacher-to-student KL on another, rather than applying both losses to every example.

Endpoint averaging and answer counting differ as estimators, but need not estimate different quantities. If both use the same rationale distribution $\rho ( r \mid x )$ and answers are sampled categorically from the same endpoint distribution $q ( \cdot \mid \dot { x } , \dot { r } )$ , then

$$
\mathbb { E } _ { r \sim \rho } [ q ( y \mid x , r ) ] = \mathbb { E } _ { r \sim \rho , a \sim q ( \cdot \mid x , r ) } [ 1 \{ a = y \} ] .\tag{9}
$$

Averaging conditional probabilities removes the final-answer sampling noise under these assumptions. Different teachers, prompts, decoding rules, or sample filters can change the target marginal; Equation (9) does not establish equivalence between the actual SII and LDM datasets.

Full-depth predictions as a teacher. Teacher computation can also differ in depth rather than in the use of reasoning traces. FunDM freezes the backbone and trains a decision head on an intermediate hidden layer to match the full-depth model’s distribution using $\mathrm { K L } ( p _ { \mathrm { t e a c h e r } } | | p _ { \mathrm { s t u d e n t } } )$ (Panchal, 2026). Inference can stop before the remaining layers, although the new head still requires training.

Black-box imitation of the deployed service. AutoTrust-JEV-27B uses output distributions obtained from TypeSafe Jev 1.13 as its main supervision stream.<sup>11</sup> Its training corpus also contains programmatic labels and downweighted uniform-placeholder targets. The documented recipe fits LoRA adapters and a decision head using target-to-student $\mathrm { K L , }$ with an ordinal ranked probability score (RPS) term. The service-target component is behavioral imitation of a black-box model. Agreement with that teacher is a different outcome from correctness against independent labels, and it does not identify the teacher’s internal training algorithm. External-teacher models such as Eikos and Torchcast provide further distribution-fitting variants, catalogued in Appendix I.

These teacher targets differ in the information and computation they retain (Figure 33).

Workflow-specific supervision. Some models are trained to support a particular workflow. LAVOIR adds slot representations and a value-of-information head to Laya, trained with decision cross-entropy and value regression (Yilmaz et al., 2026). The final recipe sets the inherited RL weight to zero. The model learns both a decision distribution and estimates used by the ask/decide/handoff policy in Appendix E.2.1.

AgentDecision-4B, introduced in JevAgentHarness, trains the decision component on agent states (Zhu et al., 2026a). Its Qwen3-4B LoRA recipe includes post-error states from mistake replay. When Jev’s top option matches the gold label, the supervised target mixes 0.3 gold one-hot mass with $0 . { \dot { 7 } }$ teacher probabilities; otherwise it uses the gold label alone. Crossentropy plus a Brier term precedes closed-loop GRPO with a fixed LLM, task-success and cost/error rewards, and a KL anchor to the supervised model. Here environment feedback updates the decision model itself. Appendix E.3 examines the different role of a fixed Jev service guiding another learner.

![](images/acd3ef9112268cb88a07bb91afc5d81510a4b397e6cdd9f4d9a8f5136c724174.jpg)  
Figure 33: Four sources of teacher targets. (a) Averaged endpoint probabilities, (b) sampledanswer votes, (c) full-depth predictions, and (d) AutoTrust service outputs, excluding its other target sources. Vectors in (a,b) are illustrative; (b) assumes four parseable samples. Dotted gold arrows carry fixed targets; dashed arrows mark updated components. Student inference omits teacher computation.

## C.4 Statistical Objectives and Gradient Credit

Takeaway 4. The probability objective and the random variable receiving credit are separate choices. A proper probability score defines an objective for truthful reporting; a gradient estimator determines how its training signal reaches the parameters. For a distribution p and outcome y, log score log $p _ { y }$ and negative Brier score $- \mathrm { \bar { Z } } _ { k } ( p _ { k } - \mathbf { 1 } [ k = y ] ) ^ { 2 }$ reward truthful probability reports in expectation. A differentiable readout can optimize either score directly. By contrast, rewarding only a sampled answer’s correctness maximizes ∑<sub>k</sub> p<sub>k</sub>q<sub>k</sub> under outcome distribution q, generally favoring a mode of q rather than reporting q. Thus the word “RL” identifies neither the statistical property being optimized nor an advantage over direct proper-score fitting.

Figure 34 contrasts direct differentiation, output-space perturbations, and rationale-policy credit. OpenJev-RLCD, the rationale-policy method depicted here, retains generated reasoning at inference; this is a property of that method rather than a requirement of every rationale-based training procedure.

![](images/6096d72423eaed31275dbac9e51185a5751cde4eb780132b7e5a2ee8d8b99140.jpg)

![](images/b1322834901ead99aee70c6ff77f08ac172e828d3744e903969c8dc023ffcd4c.jpg)

![](images/c3e48cb6a897ff5d2add5f8cbccee7749114d7573736041448dc5313c3fe250b.jpg)  
Figure 34: Three paths for learning signals. (a) Direct differentiation, (b) output-space scorefunction credit supplementing soft cross-entropy, and (c) direct answer-readout gradients with second-stage rationale-policy credit. Here b is a baseline, q a fixed target, and sg stop-gradient. Schematics omit centering, normalization, and auxiliary regularizers; they compare mechanisms, not measured estimator advantages.

Output-space policies. Laya and Chinese-Jev perturb output logits and apply policygradient updates in that space; Chinese-Jev combines soft cross-entropy with Gaussian perturbations around detached logits and a group-baseline estimator (Wang et al., 2026c). This does not require a generated chain of thought. The decider-derived this-that-model combines CE and Brier terms with a sampled-outcome Brier reward and a control-variate estimator (Cheng et al., 2026b). Another output-policy variant, decider-2B v10, uses PPO over environment actions with terminal returns and an auxiliary proper-log-score belief objective; the documented 4B and 35B releases do not use that RL stage.<sup>12</sup> These methods apply stochastic policy updates in output space rather than over generated reasoning. Their estimators, stop-gradient choices, and rewards determine what those updates contribute beyond direct fitting.

Qwen-Audio-Jev makes this distinction explicit with an importance-weighted, pairwise sampled-action estimator for negative Brier score and ordinal RPS (Fang et al., 2026). Its exploration mixture, leave-one-out baseline, and detached weights yield the direct properscore gradient in expectation under the stated construction. Thus the estimator need not define a different probability target from direct fitting. This equivalence does not establish an empirical advantage in variance, efficiency, or calibration.

Reasoning-token policies. OpenJev-RLCD first calibrates an answer distribution conditioned on sampled rationales, then adds REINFORCE credit to the rationale policy using the same proper score (Gao & Wang, 2026). If

$$
J ( \theta ) = \mathbb { E } _ { r \sim \pi _ { \theta } ( . | x ) } [ S ( p _ { \theta } ( . ~ | ~ x , r ) , y ) ] ,\tag{10}
$$

then its two relevant gradient paths are the differentiable readout term and the scorefunction term through ${ \bar { \pi } } _ { \theta } ( r \mid x )$ . Stage one uses the former; stage two adds the latter, with a baseline and KL regularization. Its main experiments score each rationale’s distribution separately and retain explicit rationales at inference. It is an independent RLCD interpretation, not a disclosed implementation of proprietary Jev.

StartLux combines instant decision learning with an AutoThink variant that allocates reasoning conditionally (Zhuang et al., 2026). Its instant model makes full-attention layers bidirectional while retaining causal linear-attention layers. $\mathrm { R L C D } ^ { 2 }$ uses a frozen reference model’s before/after probability signal to train a reasoning-token policy. Within groups containing only nonempty traces, the initial-score term cancels under reward centering; empty traces receive zero reward, so this cancellation does not extend to arbitrary mixed groups. PostHog’s Jeeves is another reasoning-policy implementation: its repository combines supervised learning, CISPO-style updates, a pointer head, and a diffusion drafter. Its documented reward uses correct-answer probability with a length penalty, which should not automatically be called a proper scoring rule.

## D Executing Decision Models

A direct choice can be produced by different computational paths and altered by serving transformations. Figure 35 contrasts computation shared across questions (a–b) with computation adapted within a decision (c–d). Both affect cost, while routing and output transformations can also change the returned prediction.

## D.1 Direct Outputs and Adaptive Computation

Takeaway 5. The absence of a returned rationale does not imply a fixed computation budget. A direct student may omit its teacher’s reasoning, an intermediate head may omit later layers, and a recurrent model may perform additional hidden steps without generating text. Conditional reasoning introduces another possibility, mixing direct and deliberative requests behind one interface. These choices change different resources, so generated-token count alone cannot compare their computational demands.

Conditional reasoning changes both the path and the prediction. LDM routes lowmargin items to the student’s untuned base for one reasoning rollout and adopts its answer; the direct branch keeps the readout’s answer (Figure 35c; Chen et al., 2026b). Its threshold is the validation margins’ 30th percentile, which does not fix the routed fraction on unseen data. StartLux AutoThink instead uses the same model weights for the initial causal readout, reasoning, and endpoint readout (Figure 35d; Zhuang et al., 2026). With a maximum initial probability below 0.9, it generates up to 4,096 reasoning tokens and returns $0 . 2 5 p _ { 0 } + 0 . 7 5 p _ { 1 } ;$ otherwise it returns $p _ { 0 }$ . These routes differ in the model called, the trigger, and the returned prediction.

![](images/2f82b1975edf9f65ab7d3b9b511ef51a5e384316a53fc729c76e7291c563a511.jpg)

![](images/ab32b7f389475054978e1897ac2daea4e6d032295eb7ca7c4b9ee91c604a9055.jpg)

![](images/cf0d6794f23ce1eab6a795a9174c268e79c9fbfda61295097b2a7b8ef49ce92e.jpg)  
Figure 35: Direct outputs permit different execution paths. (a) Kev cache reuse, (b) Clef joint schema readout, (c) LDM routing to untuned-base-model reasoning, and (d) StartLux fusion with the same weights on both paths. $p _ { 0 , ( 1 ) }$ and $p _ { 0 , ( 2 ) }$ are the two largest readout probabilities. Arrows show inference flow, without parameter updates.

Hidden computation and selective reasoning. Latent loops allocate extra computation without generating a textual rationale. SanSi repeatedly applies a recurrent backbone and reads a decision distribution after each loop (Gan et al., 2026). The base weights are frozen, while shared LoRA adapters and loop-specific readout corrections are learned. Cross-entropy and Brier losses supervise the loop outputs, with gradients passing through preceding loops. The paper trains with eight loops and evaluates different loop counts. Latent computation thus varies separately from parameter count and generated-token count, and additional loops need not improve every task.

Selective Reasoning instead exposes three control decisions: when to start generation, when to read a partial rationale, and when to accept the current answer (Zheng & Chen,

2026). A reader is trained on empty and partial reasoning prefixes using a frozen backbone. The start gate estimates correction benefit relative to regression risk, the read scheduler limits costly reader calls, and the accept rule uses the reader’s distribution. An accepted prefix answer cancels generation only if generation is still active. If no prefix answer is accepted, a terminal read returns an answer without a certainty requirement. Thus the entropy threshold governs early acceptance rather than guaranteeing the confidence of every returned answer. The threshold itself measures distribution concentration, not an independently measured probability of correctness.

FastJEV changes the execution of an existing decision model without weight training (Ma et al., 2026a). It reuses shared context and recurrent states, forks candidate-prefix caches, and selects a candidate-path layer-skipping mask using relative-score perturbations on unlabeled calibration inputs. These mechanisms preserve the context and candidate inventory while changing the work performed on candidate paths. Prefix reuse can reduce computation yet increase latency through execution overhead. Its results therefore support a serving analysis under specified workloads, rather than a universal latency improvement. The OmniJev model evaluated in that paper should not be conflated with the separately named OneJev repository.

Comparing these paths requires accounting for backbone depth, reader calls, generation, caching, and scheduling. Returned-answer length alone does not measure their cost.

A single-pass primitive can form a sequential predictor. MJev-Spatial illustrates a further distinction between the primitive and the completed prediction (Qin et al., 2026). A categorical answer uses one decision, while spatial point prediction uses three successive nine-way cell choices. The first two choices determine new crops from the original image, which are re-encoded for the next stage. Training packs teacher-forced stages into one backbone pass, but inference must follow the predicted crops sequentially. The spatial quantization bound assumes the correct cell choices; it is not a bound on model prediction error. Single-pass training, single-pass primitive scoring, and single-pass task inference are therefore separate claims.

## D.2 Serving and the Returned Prediction

Takeaway 6. Serving choices can change prediction semantics as well as execution cost. The model head is only one stage of a deployed prediction. Postprocessing can move its probability mass, scalar reduction can retain different information, and question batching can change how computation is shared. Behavior observed through a service can therefore reflect transformations applied after training as well as the learned distribution.

Sharing computation across questions. The same typed interface can expose different question-processing graphs. Kev’s documented hybrid-backbone implementation processes separate causal question rows and can reuse a state cache in serving (Figure 35a). Cloudflare Clef instead uses a joint schema head and describes one forward pass for all questions (Figure 35b). Jeff’s released server evaluates questions serially, while official Jev describes shared-state parallel question processing. These differences determine how computation is shared even when each question returns the same output type.

Documented readouts do not always identify a training recipe. Perplexity’s decider v1.1 describes a separate 255-row readout and saved temperature calibration, but the public model cards for decider and Clef do not specify their training objectives (Appendix I). JevK5 further describes a one-pass readout only up to its candidate limit and a multi-pass tournament beyond it. Candidate limits and question organization are therefore necessary conditions for interpreting an execution claim.

Transforming the returned prediction. Jeff illustrates how postprocessing can change the apparent decisiveness of a model. Its default binary “commitment” transformation moves intermediate probabilities beyond the benchmark’s abstention interval. The returned distribution therefore differs from the temperature-calibrated distribution before this transformation, and an evaluation of calibration must specify which one it measures. Reduced abstention under this rule cannot be attributed to the cross-entropy training objective alone.

SII returns the expected ordinal level, whereas Jeff’s released scalar field returns an argmax level. Identical categorical distributions can therefore yield different scalar answers.

Table 2 relates learning recipes to their readouts and public artifacts.

Table 2: Model construction, readout, and public artifacts. Rows distinguish training stages and model variants.
<table><tr><td>Model / recipe</td><td>Target and update</td><td>Readout and inference</td><td>Public evidence</td></tr><tr><td>Inner Jev / Jeff (Geuter et al., 2026)</td><td>Frozen baseline; adapted model uses mixed-label CE, LoRA and residual head; temperature fitting</td><td>Up to 16 letter labels; one forward per question; released server runs questions serially</td><td>Paper, weights, serving code and OpenDecisions</td></tr><tr><td>LLM-as-Jev (Li &amp; Wagle, 2026)</td><td>Frozen suffix scoring; optional candidate-tree NLL with KL anchors on human labels</td><td>Original architecture; short multi-token suffix scoring; optional inherited thinking</td><td>Paper describing scoring and optional adaptation</td></tr><tr><td>CLM</td><td>Bidirectional InfoNCE on matched state-action pairs; frozen backbone, learned projections</td><td>Independently encoded state and candidates; softmax of scaled cosine scores; reusable embeddings</td><td>Technical blog, heads, inference and fine-tuning code</td></tr><tr><td>SII-Decision (Duan et al., 2026)</td><td>Cached reasoning-endpoint distributions; CE, reference KL and ordinal loss</td><td>255 native single-token labels; one prefill; no rationale; T = 1</td><td>Paper, training/inference code, four models and target datasets</td></tr><tr><td>LDM (Chen et al., 2026b)</td><td>Hard-label variants; SPRD matches concentrated teacher votes; SPRD-C also checks label agreement</td><td>Direct readout; optional untuned-base reasoning below validation margin&#x27;s 30th percentile</td><td>Paper; repository states “Code coming soon&quot; at the survey cutoff</td></tr><tr><td>FunDM (Panchal, 2026)</td><td>Teacher-to-student KL; train intermediate head, freeze backbone</td><td>Early-layer decision head; remaining backbone layers can be skipped</td><td>Paper and implementation</td></tr><tr><td>27B</td><td>AutoTrust-JEV- Jev 1.13, programmatic and downweighted placeholder targets; KL + ordinal RPS; LoRA and head</td><td>Direct candidate readout; larger menus extrapolate beyond trained range</td><td>Weights, inference code and detailed model card</td></tr><tr><td>Chinese-Jev (Wang et al., 2026c)</td><td>Soft CE plus logit-perturbation policy gradient, following Laya</td><td>Direct decision readout; output-space sampling during training</td><td>Paper and public model/repository sources</td></tr><tr><td>this-that- model (Cheng et al., 2026b)</td><td>CE + Brier; sampled-outcome Direct finite-candidate Brier estimator with control</td><td>readout; follows decider</td><td>Paper and linked implementation</td></tr><tr><td>OpenJev- RLCD (Gao &amp; Wang, 2026)</td><td>variate Proper-score readout fitting, then rationale REINFORCE + reads answer-token KL</td><td>Generates a rationale, then probabilities; fitted</td><td>Paper, training code and result files</td></tr><tr><td>StartLux (Zhuang et al., 2026)</td><td>Instant learning plus RLCD² reasoning-policy</td><td>temperature Instant: bidirectional full-attention layers; AutoThink: conditional</td><td>Paper and public release links</td></tr><tr><td>Jeeves</td><td>SFT + CISPO-style reasoning updates; correct-answer probability with length</td><td>reasoning Pointer head and diffusion drafter; reasoning remains part of the design</td><td>Public training implementation</td></tr></table>

## E Decision Models in Downstream Workflows

The consequence of a decision depends on where it intervenes in a task. We distinguish control of an ongoing trajectory, selection among completed outcomes, acquisition of evidence, and assistance to another learner. These roles carry different error and recovery costs, even when they use the same decision model.

## E.1 Controlling a Trajectory or Selecting Its Outcome

Takeaway 7. Action control and completed-run selection affect different parts of a trajectory. Online selection changes which state the agent reaches next. Completed-run selection instead chooses among trajectories whose generation cost has already been incurred. Both can use a fast scorer, but only the former changes the actions within an individual run. This distinction determines which generator, candidate budget, and executor must remain fixed to attribute a downstream gain to the decision component.

Online action selection. Jev Ultrafast constructs a dynamic action space from the observed browser state.<sup>13</sup> Each observation supplies an indexed table of elements. A single Jev request asks for both an operation and speculative targets for the supported operation types. The executor uses only the target corresponding to the selected operation (Figure 36a). A small LLM supplies text when the operation requires typing. Ordinary code checks target compatibility and page freshness, executes the action, and obtains the next observation.

The division of labor depends on enumerating actionable choices from the environment. The decision model selects within that interface, while generation remains available for open-ended content. Selection of a DONE action still needs an independent outcome check. The repository reports a small set of demonstrations and repeated runs on one browser task; these establish an implemented loop, not general reliability or an isolated effect of the decision model.

JevCoder exposes a related boundary in code repair (Zhang et al., 2026). Its components recommend code snippets, guide action types, and judge whether to submit or continue. When Jev selects an action type, an LLM still generates the specific command. A judge that blocks a command does not thereby supply a better one, and some blocking interventions hurt performance. The full-workflow experiments are separate from component studies on selected success/failure panels; they should not be pooled as a single task-success estimate. Each full-workflow task receives one run per condition, which limits what can be inferred about repeated-run variability.

Decision gates within a larger controller. A decision gate can replace a local choice, advise a generator, reject an action, trigger extra work, or escalate uncertain cases. JevAgentHarness implements these roles at tool selection, bounded arguments, memory operations, reflection, stopping, and model routing (Zhu et al., 2026a). Its deployment cascade compares the maximum option probability with thresholds fitted for each decision point using held-out shadow trajectories and intervention costs. Free-form arguments still require an LLM. The gate’s confidence and the harness’s policy jointly determine behavior; a threshold alone does not establish safe execution. The same work also trains a decision model and evaluates controllers, which are distinct contributions (Appendices C.3 and F).

Jev-LiteMAS specializes this division of labor to a predefined pool of generative experts (Wang et al., 2026a). Jev selects the next expert or, when eligible, STOP, and screens artifacts for acceptance or escalation. On escalation, an LLM reviewer supplies a verdict and textual feedback; an expert revises the artifact, which is then screened again. The decision model does not generate repair instructions, and local artifact acceptance is distinct from stopping the entire workflow. Savings therefore depend on the escalation frequency and on whether the resulting artifacts still satisfy the task.

Jarvis applies typed control to an asynchronous task ledger (Wang et al., 2026b). Runtime code constructs eligible operations tied to task identifiers and versions; the decision model selects among them from the transcript, history, and ledger. Low-confidence cases delegate to a worker without speculative ledger mutation. Code enforces one live version, rejects stale results, revalidates plans, and requires a playback receipt before marking speech as delivered. These invariants belong to the executor, not to the model’s probability estimates. The paper also trains a separate audio LISTEN/DIRECT/DELEGATE gate; that learned gate and the downstream ledger policy should not be treated as one training mechanism.

A penetration-testing harness study illustrates how several decision roles can coexist in one domain (Barbosa, 2026). Typed decisions support finding adjudication, severity assessment, agent pruning, and bounded confirmation, while the harness retains evidence gates and execution. Its exploratory comparison uses one run per condition on one target, and subsequent retests change the harness. The study demonstrates this division of labor, but the separate effects of the decision model and harness remain unresolved. Establishing a benefit would require comparing the same task decomposition and execution policy with alternative decision components.

Selecting completed runs. CLM and FunDM move the intervention from actions within a run to selection among completed coding-agent runs. If the generator produces trajectories $\tau _ { 1 } , \ldots , \tau _ { N } ,$ , a scorer chooses

$$
\hat { \imath } = \arg \operatorname* { m a x } _ { i \in \{ 1 , \dots , N \} } v _ { \phi } ( \tau _ { i } ) .\tag{11}
$$

The generation cost of all N candidates remains part of the workflow. A cheap verifier may reduce selection overhead, but this alone does not make the total procedure cheaper than generating one run.

CLM’s released coding verifier averages state–action cosine scores over the last twelve trajectory steps by default.<sup>14</sup> This aggregation is a use of the contrastive scorer, not the softmax over a caller’s candidate menu described in Appendix C.3. FunDM uses a separately trained coding-agent head to judge a whole run (Panchal, 2026). Its MCQ head and coding head are distinct, and the repository distinguishes the whole-run judge from an actiontrained FunDM-CE variant.<sup>15</sup> Figure 36b therefore shows a shared selection operation with different scoring mechanisms. It does not imply shared heads, training data, or directly comparable reported performance.

## E.2 Changing the Evidence Available to a Decision

Takeaway 8. Clarification and memory control alter the information available to subsequent decisions. The preceding workflows use the available evidence to choose behavior or an outcome. Clarification obtains an external answer, while memory control changes how past evidence is organized and retrieved. These operations can improve a later decision without changing the decision model’s parameters. Their benefits and costs therefore depend on the information acquired, as well as the quality of the local probability report.

## E.2.1 Acquiring Missing Information

LAVOIR predicts the value of acquiring missing information. It extends the open Laya model so that a forward pass returns both a temperature-scaled decision distribution p and values $\hat { v } _ { k }$ for unasked information slots (Yilmaz et al., 2026). For a gold decision $y ^ { \ast }$ , the target and bounded prediction are

$$
t _ { k } = \hat { p } _ { k } ( y ^ { * } ) - \hat { p } _ { 0 } ( y ^ { * } ) , \qquad \hat { v } _ { k } = \mathrm { s t o p g r a d } \left( 1 - \sum _ { y } p _ { y } ^ { 2 } \right) \sigma ( h _ { k } ) ,\tag{12}
$$

![](images/d0ad7c955b183d87aec5bf184f39a6aeba025874289b4e7cddc6d7ae07163077.jpg)

![](images/66960a7964a0f39d17a9a56b23ee5dfac394661390ef56a95184637be552c258.jpg)

![](images/10b895a91d805cb39587c1bdb63b1774bc68ec4ba746f04b0c714f3d150bce4f.jpg)  
Figure 36: Decisions intervene at different workflow stages. (a) Online action selection, (b) completed-run selection, and (c) repeated clarification, handoff, or decision. Arrows show runtime flow. Panel (b) scores completed runs rather than controlling their individual actions; CLM includes mean score aggregation. Colors follow Figure 31; sources appear in the corresponding subsections.

where $\hat { p } _ { 0 }$ and $\hat { p } _ { k }$ come from the same frozen, temperature-scaled snapshot before and after a sampled answer to slot k. The value head produces $h _ { k } ,$ and $\sigma$ is the sigmoid function. The detached Gini factor bounds predicted values when the current decision distribution is concentrated. Training regresses $\hat { \hat { v } } _ { k }$ toward $t _ { k } ,$ , with both normalized by the target standard deviation, alongside decision cross-entropy. The final recipe sets the inherited policygradient weight to zero. Multiple complete profiles for a partial message make the regression estimate an expected gain under the sampled answers. Thus the learned value is a proxy whose validity depends on the frozen predictor and answer distribution. This construction is a model contribution, while the use of its outputs defines the interaction.

At runtime, the policy asks about $k ^ { * } = \arg \operatorname* { m a x } _ { k } \hat { v } _ { k }$ when $\hat { v } _ { k ^ { * } } > c$ and the question budget remains. The question comes from a slot template. Its answer is appended to the state, which is then re-encoded; an already asked slot is not repeated. If no question is taken, the policy hands off when $1 - \operatorname* { m a x } _ { y } p _ { y } > c _ { h }$ and otherwise returns the most probable decision (Figure 36c). Here c and $c _ { h }$ are the asking and handoff thresholds. Single-pass prediction within a turn does not make the entire interaction a single-pass procedure.

Clarification changes the source of information available to the model. LDM and StartLux conditional thinking process the existing input through an internal rationale, whereas LAVOIR obtains an external answer. It also goes beyond deferring low-confidence cases by ranking particular missing fields by predicted benefit. The evidence is strongest for synthetic multi-turn dialogues, with limited real single-exchange tests and one training seed. Comparisons against slower value-of-information computation and an LLM agent remain incomplete, leaving open whether the learned acquisition policy offers a better quality–cost trade-off across interaction settings.

## E.2.2 Controlling Memory Construction and Retrieval

Jev-Mem places a decision controller between stored observations and an LLM answerer (Jiang et al., 2026). Its contribution is the organization of memory operations around fast typed decisions, rather than a new recipe for training the controller. Shared canonical memory nodes participate in semantic, temporal, causal, and entity relation views. Figure 37 separates writing this memory from retrieving evidence, because the decision objects and feedback loops differ.

Writing preserves observations and judges relations. Incoming observations retain their original text and provenance. The controller annotates overlapping memory characteristics, while deterministic vector, lexical, entity, and time search produces a bounded set of candidate neighbors. The controller then judges inferred relations; thresholds determine which edges are added. Reliable timestamp ordering and exact identifiers can be handled directly by code. Thus the normal write path does not ask the model to irreversibly discard observations before their future relevance is known. Maintenance may mark redundancy, contradiction, or obsolescence while retaining original evidence, and textual abstraction can be escalated to an LLM.

Retrieval adapts both where to search and when to stop. Given a query, the controller predicts relevant relation views, multi-hop needs, and recency, which guide budget allocation. Hybrid vector and lexical retrieval supplies anchors. During retrieval, the controller assesses whether accumulated evidence is sufficient and whether further expansion is likely to help. If needed and within budget, bounded graph expansion proposes candidates, which are scored and added to the evidence set. The loop stops on sufficient evidence, low expected additional benefit, or a resource limit. Selected memories then go to the LLM for answer synthesis. The controller affects evidence acquisition; it does not replace the final generative answerer.

The reported evaluation uses LoCoMo with GPT-4o-mini as the answer model and LLM-asa-Judge scoring. These results concern Jev-based control on LoCoMo and do not evaluate the Laya controller or LongMemEval configuration provided by the implementation.<sup>16</sup> Separating write and read control is particularly useful for attribution: changing stored relations can affect later answers even if the query-time controller is held fixed.

Evidence expansion without a preconstructed graph. SAG2 instead starts from a local SQLite FTS5/BM25 index, without corpus-time embeddings or a neighbor graph (Wu, 2026). Jev makes a binary Noul judgment about each passage’s necessary contribution to a supported answer chain, conditioned on the retrieved evidence. This is not a single global Choice over passages. Ordinary code extracts lexical expansion terms from scored seed passages; retrieval and evidence-conditioned rescoring can repeat within a bounded budget. Local diversification then produces the final evidence set. The experiments evaluate retrieval rather than generated answers, and development includes MuSiQue and a synthetic enterprise corpus. The local index does not make the complete system local, because the evidence scorer uses the hosted Jev service. Relative to Jev-Mem, the intervention moves from constructing and traversing stored relations to building query-time evidence chains through lexical expansion.

![](images/9599a0563392ac2ac1de4e5fadc8bbec4e93b4d3251e774b1fe5660edd606319.jpg)  
Figure 37: Jev-Mem controls memory writing and retrieval (Jiang et al., 2026). (a) Writing preserves observations and judges relations. (b) Retrieval iteratively acquires evidence before answer synthesis. Four graph views share memory nodes. Arrows span multiple operations and calls; colors follow Figure 31.

## E.3 Decision Models in Agent Learning

Takeaway 9. A decision model can guide another learner without itself being trained. A fixed decision service can also change an agent’s exploration, rewards, or updates during learning. Any resulting benefit may persist in the trained agent after the service is removed. This differs from fitting a replacement decision model to service outputs, as in Appendix C.3, where training changes the model that will answer future decision requests. The relevant distinction is which model learns from the service and what it learns to do.

Ma et al. (2026b) use Jev as an action-policy reference, an exploration source, an auxiliary reward assessor, and a replay-priority signal. These interventions change an agent’s data or updates, so their success is measured in the learned agent. AutoTrust instead fits a replacement decision model to Jev’s probability outputs, making reproduction of the teacher’s predictions a training target.

The interventions have different semantics. Policy assistance supplies actions that can be imitated or compared with the learner’s actions by a critic. Exploration assistance replaces some random actions with Jev suggestions. Reward assistance maps rubric judgments to bonuses or gates. Replay assistance reweights stored experience using Jev judgments. The reported replay variant omits importance correction and therefore intentionally changes the effective training weighting. None of these roles makes an ordinal judgment a validated cardinal Q-value merely by converting its label to a number.

Information access also matters: Jev can receive structured descriptions beyond the learner’s own observation, and the trained learner can later be deployed without Jev or its auxiliary rewards. The paper’s MiniGrid and Atari experiments provide evidence for particular learning interventions, with three seeds; replay assistance does not uniformly help. The resulting gains or losses concern how the service changes learning. They cannot be attributed to the quality of its decision probabilities without separating added information, changed exploration, and changed objectives.

Joint learning changes the decision model as well as the agent. Dyad spans both levels in a different way (Zhan et al., 2026). Its language model retains token generation and switches to a typed action distribution at designated interaction positions. The model aligns state and action representations through projector training, then applies environment-reward policy optimization over token and action decisions. Its action encoder may be updated alone or jointly with the state encoder. Action features are reusable only while their environment context and descriptions remain unchanged. Thus the learned architecture belongs to model construction, while the interleaving of language and environment actions defines the downstream workflow; it is not simply a frozen service plugged into another learner.

## F From Local Probabilities to Workflow Reliability

A candidate distribution, the action selected from it, and the outcome of a complete task require different evaluations. Accurate probabilities can yield a poor action under an unsuitable decision rule; a correct local action can still fail to secure task success. Our experiment measures selections, leaving calibration and downstream utility unresolved. The studies below identify the evidence needed to connect these levels.

## F.1 Local Probability Quality and Decision Utility

Takeaway 10. Calibration, distribution fidelity, and decision utility are distinct. Ordinary calibration compares reported probabilities with outcome frequencies among cases receiving the same predicted probabilities. Subgroup calibration additionally requires this agreement within specified groups. Distribution fidelity instead asks whether the complete prediction matches the conditional target for each input. Neither calibration nor fidelity alone specifies the action taken or the cost of an error. A decision rule converts probabilities into behavior, so a utility improvement can arise from a better distribution, a better rule, or both. Comparisons must separate these changes before attributing a gain to learning.

Known distributions separate probability estimation from sampled correctness. ProbabilityBench supplies simulated problems with a known conditional distribution p∗ given the visible input (Dai et al., 2026). This permits direct comparison with p∗, rather than treating one sampled answer as the complete probability target. Accuracy evaluated in expectation under these conditional targets is bounded by the Bayes accuracy for the same visible information. Accuracy against sampled test labels can exceed that reference through label-sampling noise. For a predicted distribution q, the input-averaged KL $( p ^ { * } \| q )$ equals excess expected log loss relative to $p ^ { * }$ . A constant class prior satisfies ordinary calibration while ignoring input-specific information, and need not be calibrated within subgroups. Finite-bin estimates can also hide canceling errors or reflect sampling noise. Exact uncertainty in a synthetic generator is therefore valuable evidence about probability estimation, but does not establish access to the true uncertainty of an arbitrary real task.

Additional computation can sharpen probabilities without improving calibration. SanSi provides a concrete example of why compute allocation and probability quality must be evaluated together (Gan et al., 2026). Its later loops continue to increase confidence even after most answers have stabilized. On the 9,645 answerable test items, the 1.4B model’s expected calibration error rises from $0 . 0 8 2 \pm 0 . 0 1 0$ at loop three to $0 . 0 9 3 \pm 0 . 0 1 2$ at loop eight. These are means and standard deviations over three training seeds, using ten equal-width confidence bins. The comparison concerns different readout depths within each trained model. It shows that additional latent computation can worsen calibration even within the trained loop range, so a stopping rule needs evidence for the probability or decision objective it serves.

Utility also depends on the decision rule. TypedBench complements this view with policy-generated cases, controlled framing changes, and cost-sensitive evaluation (Sharma et al., 2026). Its calibration and stress-test protocols distinguish benign framing sensitivity from adversarial perturbation. A change from argmax to the Bayes action under a cost matrix changes the decision rule; the resulting utility improvement cannot all be credited to better calibration. Synthetic labels make some controls possible while limiting direct conclusions about unconstrained natural tasks.

Benchmark scores depend on their evaluation protocol. Published scores differ in both what they measure and how evaluation informs model development. JEVal supplies a benchmark as well as the SII model discussed in Appendix C.3, but the SII release’s separate evaluation suite does not reproduce that benchmark protocol (Duan et al., 2026). Inner Jev’s Capability metric differs from ordinary accuracy, and its OpenDecisions results inform recipe selection (Geuter et al., 2026). LLM-as-Jev likewise uses public JevBench errors to guide data-mixture development (Li & Wagle, 2026). Metric definitions and evaluation feedback must therefore be accounted for before these scores can support a common ranking.

Data overlap and execution conditions further limit comparisons. FunDM’s reproduction notes disclose overlap between a CLM post-training collection and its test collection. Localmodel and hosted-API latency also reflect different hardware and network conditions. Generalization and efficiency claims therefore require controls beyond matching the reported metric.

## F.2 Composition and Input Sensitivity

Takeaway 11. Reliable isolated decisions need not remain reliable under composition. An isolated prediction tests one input and output interface. Composing predictions introduces dependencies; changing a request can alter evidence, eligible actions, or numerical computation. Interpreting sensitivity first requires specifying what should remain invariant. Our permutations preserve question content and candidate identities, whereas adding a candidate or revealing evidence can change the decision problem.

Coherence studies test probability relationships across related questions (Joy, 2026), while option-set studies test how predictions respond to changed alternatives (Rajput et al., 2026). Neither relationship follows from calibration on isolated questions.

Request context, candidate eligibility, and numerical precision. Chen (2026) distinguishes changes to the questions bundled into a request, the evidence provided, and the eligible candidates. A model whose question compiler produces exactly the same input can be invariant for that reason, whereas a full-schema prompt exposes other questions to the decision computation. The Nimble intervention removes other questions from the prompt and eliminates the observed flips, but is not length-matched. Small Laya shifts largely disappear at higher precision or after rounding. The observed sensitivity can therefore reflect prompt construction or numerical precision as well as architectural dependencies. Conditional odds between retained alternatives separate genuine relative-score movement from the inevitable change in a softmax denominator. When a readout restricts the vocabulary, its allowed-label mass provides additional information lost by normalization.

From correct choices to correct sequences. Choice2Seq compares native candidate choice with independent-position and serial-prefix output interfaces on paired problems (Lai et al., 2026). Its serial correctness retention is P(serial correct native correct), not agreement with the native answer or fidelity to a plan. Cases improved by the serial interface and the marginal accuracies are needed alongside this conditional statistic. Gold-copy and self-plan controls address different fidelity questions. Candidate restrictions can also change the available search space and number of calls, so a composition claim requires the relevant interface and information controls.

Robustness and program consequences. Robustness studies connect input sensitivity to the consequences of a valid but incorrect choice. DecisionRobust tests label-preserving perturbations and targeted prompt injection on initially correct decisions across text and vision (Lan et al., 2026). The distinction between observation text and task instructions matters because changing task facts would change the decision problem itself. JevProbe traces the structured response through application logic and tests whether reachable lowertrust context can redirect a program decision while leaving task facts and code fixed (Luo et al., 2026). Together, these studies show why schema validity is insufficient for reliable execution. They identify failures of input handling and decision use, with modeled program consequences rather than measured real-world losses.

The released response can reveal more than the intended decision. TUPLE-Bench studies whether a response about an authorized coarse category reveals a protected finer mechanism in the model’s private evidence (Yang et al., 2026). Choice-only, confidence, full-probability, and native-response releases are derived from the same saved response, separating release policy from model reruns. Its informed decoder receives the gold coarse category and two known fine-grained hypotheses; this is a passive closed-set audit, not an estimate of unrestricted real-world leakage. A separate legitimate consumer sees neither private evidence nor those hidden labels. The proposed workflow-conditioned probability quantization selects codewords that preserve a frozen consumer’s actions, including clarification, with a native-response fallback when no match exists. It can preserve wrong actions as well as correct ones, so action preservation establishes neither improved task quality nor a confidentiality guarantee. The contribution concerns evaluation and release postprocessing, not a new decision-model training objective.

## F.3 Complete-Workflow Outcomes

Takeaway 12. Task-level gains depend on the surrounding policy and its total resource use. Ling et al. (2026b) separate changes to the action interface from changes to the decision model in ALFWorld. They compare the original baseline, an LLM using a Choice interface, and Jev using that interface. The first contrast examines the candidate-selection interface; the second compares decision components within it. Their AndroidWorld experiments compare baseline and Jev integrations without this three-way control. Across these workflows, local decision savings do not uniformly translate into better task outcomes. Action granularity also differs in one integration: FastAgent-J takes one tool action per step while its baseline can take multiple actions. The study thus provides evidence about particular integrations while illustrating the controls needed to interpret them.

JevAgentHarness distinguishes isolated multiple-choice accuracy, decisions on recorded agent states without execution, and actual closed-loop intervention (Zhu et al., 2026a). Its closed-loop tool-selection comparison lets the LLM supply tool arguments and uses unthresholded replacement. That protocol differs from the confidence cascade proposed for deployment, so its outcomes cannot be attributed directly to that cascade. Scripted acceptable actions in DecisionGym and disjoint rollout seeds also do not, by themselves, establish generalization to unseen task distributions.

System1-Arena broadens the comparison across static decisions and closed-loop environments under task-specific shared harnesses (Chen et al., 2026a). Candidate construction converts otherwise open-ended search, patches, or arguments into finite alternatives; the reported results are conditional on this interface. The static comparison uses 85,305 questions across 94 tasks, a subset of the released collection, and excludes its game and embodied questions. Reported static scores weight tasks equally, with source averaging within a task. Closed-loop scores weight environments equally. For real-time simulations, the environment advances using each model’s measured median latency while retaining the prior action, rather than measuring each call’s realized delay inside the simulation. Probability extraction also differs across native decision models, verbalized-probability APIs, and local label-logit models. These protocol choices enable useful controlled comparisons while limiting a universal ranking across interfaces, hardware, and workloads.

Adoption is a separate object of measurement. Ecosystem studies explain where decision models are used, but do not replace these capability and workflow evaluations. Ling et al. (2026a) analyze public Jev repositories by application domain and decision purpose. Purposes can overlap within a project, and repository counts or stars measure ecosystem activity rather than model quality or methodological novelty. Tang & Zheng (2026) audit early typed-decision-model evidence and connect it to label-probability readouts, reranking, calibration, and model cascades.

## G Open Questions

The unresolved question is which design changes preserve correct choices across equivalent presentations. The comparisons below separate representation, learning, computation, and workflow effects so that any improvement can be attributed to a specified intervention.

Which improvements survive equivalent presentations and related decisions? Takeaways 10 and 11 distinguish local probability quality from consistency across requests. The main study finds substantially more Jev variation under reordering than under fixed-order repetition, without identifying the mechanism behind this difference. Independent studies examine disagreement between direct and hierarchical predictions (Joy, 2026), option-set dependence (Rajput et al., 2026), serial correctness retention (Lai et al., 2026), and the limited transfer of local savings to successful agent outcomes (Ling et al., 2026b). Together, they motivate testing whether a local probability improvement persists under transformations that preserve the intended decision problem and predicts utility in a fixed workflow. The intended invariance must be specified first. Independence of irrelevant alternatives is not universally appropriate, since an added option may provide meaningful context or change the task. Prompt serialization and precision require controls (Chen, 2026), while cacheability must be separated from set-dependent refinement (Li & Wang, 2026).

When does the gradient estimator matter beyond the target and objective? Takeaway 4 separates statistical supervision from the variable receiving gradient credit. Replacing observed labels with teacher distributions changes the target; adding output noise can replace an unperturbed score with its expectation under perturbation. An estimator comparison therefore requires a fixed objective and perturbation distribution, with pathwise differentiation and score-function credit compared where both apply. Matching the backbone, prompt, candidates, examples, adaptation budget, and validation split would isolate this question, with frozen and temperature-calibrated readouts as starting controls. Whether a gain survives common calibration and probability postprocessing would help establish its source. A negative result would delimit the value of a specified stochastic update without resolving how proprietary Jev is trained.

Which aspects of reasoning transfer into a direct readout? Takeaways 3 and 5 distinguish the information supplied during learning from the computation available at inference. JEVal reports that reasoning-to-readout transfer differs between evidence-weighing and computation-heavy tasks (Duan et al., 2026); earlier System 2 distillation also documents task-dependent transfer (Yu et al., 2024). Conversely, theoretical work shows that some computations can be internalized under particular architectures and curricula (Huang et al., 2026). The unresolved empirical question concerns the interaction of training support, novel rule composition, and dependency depth. Correct-label supervision, additional data, intermediate-state hints, and explicit-reasoning controls would help distinguish imperfect targets and learning limitations from residual computational demands. Failure on a task family alone does not prove an impossibility of single-pass reasoning.

When does a better decision model improve the complete workflow? Takeaways 7–9 identify where a model intervenes; Takeaway 12 shows why that intervention matters for evaluation. A comparison within a fixed workflow can replace only the decision component while holding candidate construction, retrieval, budgets, fallback rules, and answer generation constant. A complementary comparison retains the same model while changing its role or control policy. Jev-Mem motivates separating write control from read control; run selection requires matching candidate-run generation; LAVOIR requires accounting for newly acquired information and the cost of questions. Crossing model and workflow choices can reveal whether local probability improvements transfer to task outcomes and how that transfer depends on the surrounding policy. Existing evidence does not establish that either model quality or workflow design universally dominates.

These factors can interact. Establishing a structural advantage requires identifying both the intervention and the conditions under which its benefit survives.

## H Survey Synthesis

A candidate-distribution interface accommodates different representations, objectives, gradient estimators, inference procedures, and downstream roles. The main study shows that this shared interface does not ensure preservation of correct choices under reordering. Explaining the observed differences requires controlled comparisons under common information and resource budgets, followed by tests of whether local improvements benefit the workflow the model controls.

## I Extended Public Implementation Inventory

Table 3 summarizes additional implementations and the mechanisms documented in their primary sources as of October 19, 2026. Public artifacts do not by themselves establish independent reproduction.

Table 3: Additional implementations and documented mechanisms.
<table><tr><td>Project / primary source</td><td>Documented mechanism</td><td>Scope and distinctions</td></tr><tr><td colspan="3">Frozen readout and execution recipes</td></tr><tr><td>SemIf / former OpenJev</td><td>Restricted answer-label probabilities; Runtime code; explicitly an interface</td><td>workload-specific temperature fitting pattern, not TypeSafe's training recipe</td></tr><tr><td></td><td>open-alternative-jev Multiple packed readout positions;</td><td>Runtime variants differ; some use constrained one-token generation</td></tr><tr><td>Cygnet</td><td>exact-logit and top-k logprob backends Frozen Gemma-4-12B-it; label</td><td>One output token is emitted; no</td></tr><tr><td>deck31b</td><td>probabilities, fitted temperature and İarge-menu grouping Frozen Gemma-4-31B-it for candidate Uses pretrained weights without</td><td>weight adaptation</td></tr><tr><td></td><td>selection</td><td>decision-specific adaptation</td></tr><tr><td>Table 3 continued Project / primary</td><td>Documented mechanism</td><td>Scope and distinctions</td></tr><tr><td>source reflex</td><td>Frozen Qwen3.5-4B with averaging</td><td>Tested adapters degraded</td></tr><tr><td></td><td>over two option orders Cyclic label rotations and label-prior</td><td>out-of-distribution judgments; the default model remains frozen Full rotation costs K prefills for K</td></tr><tr><td>AnyJev Decider L0</td><td>correction</td><td>candidates; no weight adaptation</td></tr><tr><td colspan="3">Adaptation and teacher-based variants Strands Decider</td></tr><tr><td></td><td>v1: Qwen3.5-2B-Base, LoRA and pointer head; supervised targets, model averaging and calibration</td><td>Training loop and inference sources available; optional KL branches do not prove their use in every released variant</td></tr><tr><td>Nimble</td><td>Qwen3.5-9B LoRA; hard-label CE over “Contrastive" describes paired data, candidate logits; paired examples stay not an InfoNCE objective; stored Jev in one split</td><td>probabilities are not the targets of this loss</td></tr><tr><td>Intern-Decision (Zhu et al., 2026b)</td><td>Native logits before decision placeholders; full-vocabulary CE at those shifted positions; frozen vision encoder and projector</td><td>Technical report and training entry points available; original training data and private calibration splits are not released</td></tr><tr><td>OneJev</td><td>Full language adaptation with frozen vision; candidate-slot soft-target CE plus Brier loss</td><td>Public data are a subset; not the OmniJev model evaluated by FastJEV</td></tr><tr><td>Kev</td><td>CE-trained pointer head; smaller models use LoRA, 27B uses full adaptation</td><td>Training/inference/evaluation sources and model links; hybrid models use separate causal question rows with serving state reuse</td></tr><tr><td>JevK5</td><td>Qwen3.5-4B and Qwen3.5-9B LoRA, teacher-derived data, letter readout and temperature</td><td>Up to 16 choices per pass; tournaments for larger menus</td></tr><tr><td>Plumb</td><td>Teacher-generated hard labels plus public-label replay; LoRA CE</td><td>Pipeline described; related to JevK5; hard labels do not imply soft-target KL</td></tr><tr><td>Hopper</td><td>LoRA and mixed synthetic, teacher-reviewed and public supervision</td><td>Model card describes multiple backbone variants</td></tr><tr><td>Tacit / AnyJev</td><td>Same-base task/reasoning-label self-distillation; optional thinking router</td><td>Distinct from training-free Decider L0</td></tr><tr><td>AutoTrust-JEV-27B</td><td>Jev 1.13 distributions mixed with programmatic and downweighted placeholder targets; KL + ordinal RPS</td><td>Detailed recipe, weights and inference; teacher agreement is not correctness</td></tr><tr><td>Eikos</td><td>External-teacher soft CE, LoRA and auxiliary rationale loss</td><td>Model-card evidence; no inference-time reasoning requirement inferred from that loss</td></tr><tr><td>Torchcast</td><td>Gold labels and larger-model probability targets; LoRA</td><td>Model card and serving release; public benchmark used in model/configuration selection</td></tr><tr><td>Winnow</td><td>Trained LoRA with synthetic, teacher and labeled data</td><td>Private training pipeline; exact loss undisclosed; inference code available</td></tr><tr><td>Quyet-1.0-Large</td><td>Merged LoRA; letter readout and per-type temperature</td><td>Description applies to the Large variant</td></tr><tr><td>decision-fast- preview</td><td>Trained adapter, structural tokens and Adapter, head and inference code pointer head</td><td>released; mixed corpus not redistributed</td></tr><tr><td>Koa-decision</td><td>Reserved atomic label tokens with randomized assignment; single-position CE</td><td>Training recipe described in the paper</td></tr><tr><td colspan="3">Table 3 continued</td></tr><tr><td>Project / primary source</td><td>Documented mechanism</td><td>Scope and distinctions</td></tr><tr><td colspan="3">Policy recipes and version-specific training</td></tr><tr><td>Laya</td><td>Bidirectional decision model; soft CE and perturbed-logit policy updates</td><td>EChinese-Jev explicitly inherits this loss; distinct from reasoning-token RL</td></tr><tr><td>decider</td><td>2B v10: PPO environment-action policy, terminal returns and auxiliary log-score belief loss</td><td>Documented 4B/35B omit this RL stage; stock decider-chat is frozen; RL loop not in release package</td></tr><tr><td>Jeeves</td><td>Pointer head; SFT, reasoning CISPO and diffusion drafting</td><td>Public trainer; correct-answer-probability reward with length penalty is not a proper score</td></tr><tr><td>RSI-Jev</td><td>Versioned SFT, averaging and early-exit variants; retained ranking RL uses NDCG@5</td><td>Task-specific ranking reward; recipes and benchmark-selection exposure vary by version</td></tr><tr><td colspan="3">Task-oriented model extensions LAVOIR (Yilmaz</td></tr><tr><td>et al., 2026)</td><td>Extends Laya with decision CE and value-of-information regression; interaction in Appendix E.2.1</td><td>Final broad release sets inherited RL weight to zero; also contributes a downstream policy</td></tr><tr><td>NanoJev</td><td>Complete-question CE on four games</td><td>Supervised game-decision model; distinct from uses of NanoJev as a simulator</td></tr></table>

Documented readout, undisclosed training objective. Cloudflare Clef attaches a small transformer joint schema head to Qwen3.8-27B. Its card describes all-question probabilities from one forward pass, an expected-value scalar score, and backbone, head and inference artifacts, but not the exact post-training objective. pplx-decider-v1.1-27b uses a separate 255 5120 readout, noncausal full-attention layers with unchanged linear-attention behavior, and saved temperature calibration. Its training loss and update method remain unspecified. These interfaces alone establish neither RL nor supervised training.

Additional documented implementations. imajev documents language LoRA with frozen vision and weight-space averaging of a hard-question adapter with its soft-target continuation. jqv offers frozen-logit backends and trainable slot or pointer heads with LoRA and CE plus Brier loss. openJev-verdict describes a marker-pointer distribution head, an out-of-fold confidence head, and symmetric KL between option-shuffled training pairs.

Other designs vary the candidate representation. jevlike learns option-conditioned context scoring with either a learned byte encoder or a frozen pretrained encoder. agent-jev combines independent candidate paths with a permutation-equivariant scoring head and documents soft CE plus Brier training for AgentJev-0.6B. GLiNER2.5-Decide publishes supervised fine-tuning formats and procedures, but the objective used for its released weights remains unspecified. jev-local offers an optional frozen-model log-probability scorer and a deterministic default stub; head training is explicitly deferred.