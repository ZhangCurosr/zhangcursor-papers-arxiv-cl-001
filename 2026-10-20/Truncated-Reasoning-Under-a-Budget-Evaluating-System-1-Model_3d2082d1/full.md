# Truncated Reasoning Under a Budget: Evaluating System-1 Models as Decision-makers

Zhongyin Wang<sup>1†</sup>, Zhaoqian Xue<sup>2</sup>, Junting Mu<sup>3</sup>, Haozheng Liu<sup>4</sup>, Ye Yuan<sup>5,6</sup>, Haolun Wu<sup>5,6</sup>, Lingyao Li<sup>2†</sup>

<sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>University of Arizona <sup>3</sup>University of California, Irvine, <sup>4</sup>New York University <sup>5</sup>Mila – Quebec Artificial Intelligence Institute, <sup>6</sup>McGill University

## Abstract

Large language models (LLMs) often generate more reasoning tokens than needed. Limiting the reasoning budget reduces cost, but forcing an answer from an incomplete trace can hurt accuracy. System-1 decision models select among candidate options without producing a reasoning response. We evaluate Jev as a decision reader of truncated reasoning across seven LLM reasoners and five multiple-choice benchmarks (35 pairs). Jev achieves higher mean accuracy than forced answering at tested budgets from 500 to 16,000 tokens, including the estimated costs of reasoning and answering. At a 1,500-token budget, it improves accuracy by 4.1 pp overall and 8.6 pp on truncated questions, with significant gains in 17 pairs under Holm-corrected bootstrap tests. Offline replay estimates 34.5% of the full-run cost, with a 1.13 pp accuracy loss. A policy selected on training folds saves 71.3% of the estimated full-run cost on held-out folds, with a 0.58 pp accuracy loss. We also find that Jev shows strong question-only accuracy, exceeding 500-token forced answering on average (68.0% vs. 65.5%), although this advantage varies by benchmark. These findings suggest that System-1 decision models can complement LLM reasoning and reduce the need for costly reasoning completion.

## 1 Introduction

Large language models (LLMs) can now reason before they answer. Reasoning models such as o1 and DeepSeek-R1 generate extended chains of thought to tackle complex problems (OpenAI, 2024; DeepSeek-AI et al., 2025). Some models also expose adjustable thinking budgets to balance performance and latency (Yang et al., 2025). However, models may generate redundant reasoning

No cut: reasoner thinks to the end Cut at 1.5k tokens, reasoner answers (forced) Cut at 1.5k tokens, Jev answers

![](images/99ba74932d0c4973c288389e4bc9ddb00b7d701042ffd114c0281fb4e2e14484.jpg)  
Figure 1: Seven LLM reasoners cut at 1,500 tokens (mean over five benchmarks). Left: full-run accuracy and the change when the reasoner answers (blue) or Jev answers (red). Right: cost as % of the uncapped run.

that adds little to answer quality while consuming additional tokens, a phenomenon known as overthinking (Chen et al., 2025). Token-budget prompting (Han et al., 2025), budget forcing (Muennighoff et al., 2025), and early-exit rules (Liu and Wang, 2025; Yang et al., 2026; Fu et al., 2026; Sun et al., 2026) seek to reduce this cost by controlling how much reasoning is generated. Yet once the reasoning stops, the reasoner must still answer from an unfinished trace, and our results show that the answering mechanism substantially affects accuracy at short budgets.

Recently, System-1 decision models such as Jev (TypeSafe AI, 2026) and the GPT-6 Luna Decision (OpenAI, 2026a) have offered a different approach. They score a fixed set of options in one forward pass without generating text, at a fraction of an LLM’s cost. Jev has been evaluated as a standalone decision model across classification, reading comprehension, and reasoning tasks (Deußer et al., 2026; Duan et al., 2026). It has also been studied as a judge (Li et al., 2026) and as a decision layer in LLM agents (Wu and Lim, 2026). These tasks involve choosing among known options using information already in the input, such as grading a response or selecting the next action, making them well suited to fast, low-cost decision models.

How decision models use partially generated reasoning and whether this improves cost–accuracy trade-offs remains less understood. An LLM reasoner produces evidence at a high cost, whereas a decision model can read that evidence at a much lower cost (Kahneman, 2011). Prior work shows that reasoning traces can be reused by other models (Cheng et al., 2026) and that decision models are most competitive when decisions can be resolved from available evidence (Duan et al., 2026). These findings motivate a division of labor in which a reasoner generates intermediate text and a decision model selects the final answer, and we ask three research questions:

• RQ1 (Reliability): How reliably can a decision model answer from truncated reasoning compared with the reasoner and other readers?

• RQ2 (Cost–accuracy trade-off): What cost– accuracy trade-offs arise under static budgets, and what benefit does adaptive stopping add?

• RQ3 (Trace contribution): When and to what extent does the reasoning trace improve or degrade the decision model’s answers?

We study these questions in multiple-choice question answering across seven LLM reasoners and five benchmarks (Rein et al., 2024; M-A-P Team et al., 2025; Jin et al., 2021; Zuo et al., 2025; Wang et al., 2024). We stop each reasoner after a fixed token budget and use a decision model to select an answer based on the truncated trace. We compare this approach with forcing the LLM reasoner to answer. For each approach, we include reasoning and answering costs. Across all seven reasoners, Jev improves accuracy over forced answering. At 1,500-token budget, it achieves 75.8% accuracy, compared with 71.7% for forced answering, at about one-third the cost of uncapped reasoning (§4.1, §4.2). We also find that Jev can answer many questions correctly without a trace, while longer reasoning remains beneficial on the hardest benchmark (§4.3). These results suggest that pairing an expensive reasoner with a fast decision model has the potential to reduce reasoning costs with little loss in performance.

## 2 Related Work

LLM Overthinking and Reasoning Efficiency Long reasoning traces can improve LLMs’ problem solving, but also cause overthinking, increasing computation and latency (Chen et al., 2025). Existing work allocates computation before reasoning through model routing (Ong et al., 2025), reasoning mode selection (Fang et al., 2025; Lou et al., 2025), and token budget control (Han et al., 2025; Muennighoff et al., 2025). However, these allocations may not match the needs that emerge during reasoning.

Dynamic stopping methods instead monitor answer the stability (Liu and Wang, 2025; Sun et al., 2026), confidence or prediction signals (Kim et al., 2026; Yang et al., 2026), or learn stopping policies (Dong et al., 2026). Methods requiring hidden states or token probabilities are difficult to apply when APIs expose only reasoning text. The black-box variant of EAT (Wang et al., 2026) uses a smaller proxy model to compute entropy-based stopping signals while retaining the original reasoner for final answering. This enables external stopping decisions, while leaving answerer choice as a separate design consideration.

System-1 Decision Models System-1 decision models directly score candidate answers without autoregressively generating responses, potentially reducing cost and latency. Deußer et al. (2026) benchmark Jev across classification, reading comprehension, and common sense reasoning. Duan et al. (2026) find that decision models are competitive when available evidence supports a decision, but remain limited on tasks requiring specialized knowledge. This motivates separating standalone answering ability from the contribution of supplied reasoning. Prior work also explores model collaboration. Jev-Mem (Jiang et al., 2026) uses System-1 to control memory retrieval and System-2 for reasoning and answer generation. Reasoning that Travels (Cheng et al., 2026) studies prefix transfer between generative models and receiver agreement for early stopping. We compare non-generative Jev with the original reasoner as final answerers at matched prefixes, evaluating accuracy, monetary cost, and latency, with question-only as a control for the contribution of external reasoning.

![](images/23bce2580b1edff8f17d61f7f7a03584e2f318e6ad894478082d4cd7fb4fafdc.jpg)  
O ine replay on retained traces • 7 reasoners × 5 benchmarks  
Costs are replay estimates, not online cancellation measurements. | Real example: MMLU-Pro • Qwen3.8 Flash • mmlu\_pro-1920  
Figure 2: Comparing four answerers by offline replay at a 1,500-token budget. Runs with a valid natural answer within the budget retain that answer for every policy. At a cut, the original reasoner receives the whole prefix, while readers receive the question, options, and the final approximately 1,000 reasoning tokens.

## 3 Methods and Experiments

## 3.1 Problem Setting

Given a question q and options O, LLM reasoner M produces a retained reasoning trace $z _ { 1 : L }$ and, when available, a final answer $a _ { \mathrm { f u l l } }$ . We compare answerers on prefixes of this trace without accessing model weights or hidden states. The reasoning budget B, checkpoint interval k, and reader window w control generation length, checking frequency, and reader context, respectively. Figure 2 summarizes the comparison.

## 3.2 Benchmarks

Our benchmark consists of five datasets: GPQA-Diamond (198 questions; Rein et al., 2024), SuperGPQA (500; M-A-P Team et al., 2025), MedQA (500; Jin et al., 2021), MedXpertQA-Text (500; Zuo et al., 2025), and MMLU-Pro (500; Wang et al., 2024). SuperGPQA contains 167 easy, 166 middle, and 167 hard questions, reweighted to the full benchmark’s difficulty distribution. Together with seven reasoners, these yield 35 reasoner– benchmark pairs.

The retained acquisition cohort contains 15,377 model–question runs, including 15,309 with a valid natural answer. The other 68 full-run outcomes, including one saved generation failure, count as incorrect. The four-way comparison at B = 1.5k uses 15,376 matched runs: one outcome lacks an exact forced answer and is excluded from this comparison, not from the acquisition cohort. Trace acquisition and missing-data accounting appear in Appendix B.

## 3.3 Model Selection

LLM as Reasoners. We use DeepSeek V4 Pro (DeepSeek-AI, 2026b), Qwen3.7 Plus (Qwen Team, 2026), MiniMax M3 (MiniMaxAI, 2026), GPT-OSS 120B with high reasoning effort (OpenAI, 2025), GLM-5.3 Flash (Z.ai, 2026), Qwen3.8 Flash (Alibaba Cloud, 2026), and DeepSeek V4 Flash (DeepSeek-AI, 2026a), all accessed through Together AI (Together AI, 2026). Forced answering supplies the whole prefix to the reasoner in a separate request, using a provider-supported strict or think-close protocol (Appendix A).

Decision and Generative Readers. We evaluate three reader configurations: Jev (TypeSafe AI, 2026), a non-generative System-1 decision model, and GPT-6 Luna accessed through either its Decisions API (OpenAI, 2026a) or generative interface (OpenAI, 2026b). Jev supports 64k input tokens per request, with a 32k limit on the state plus the longest decision question. All three receive the question, options and the final nominal w = 1,000 reasoning tokens before the cut. The decision interfaces return option choices. Luna’s generative interface returns a short answer with confidence. Reader comparisons match the saved reasoning window, where the main forced-answer comparison also changes context length and answer elicitation.

Experiment procedure   
Input: retained traces, answer records, cut settings,   
window w.   
1 for each trace and budget B or fraction f do   
2 c ← min $( B , L )$ or round $( f L ) .$   
3 if budget mode and a saved generation failure is   
reached before answering then   
4 Record failure. Do not invent an answer call.   
5 else if budget mode, $L \leq B ,$ and a valid natural   
answer exists then   
6 Retain that answer for all answerers.   
7 else   
8 Retrieve the forced answer at c. Require an exact   
cut at $B = 1 . 5 \mathrm { k } .$ , otherwise allow the earlier-position   
fallback.   
9 Omit the outcome from the four-way 1.5k com  
parison if its exact forced answer is missing.   
10 Retrieve each reader’s answer from   
$( q , \mathcal { O } , \operatorname { T a i l } _ { w } ( z _ { \leq c } ) ) .$   
11 end if   
12 Record correctness and each policy’s cost.   
13 end for  
Table 1: Offline replay using precollected answers. Policy cost includes only calls used by that policy.

## 3.4 Experiment Setup and Cost Estimation

Cost estimation and experiment procedure. A fixed budget B retains $a _ { \mathrm { f u l l } }$ only when a valid natural answer is available and $L \leq B$ . Otherwise, answerers use the cut $c = \operatorname* { m i n } ( B , L )$ , except that a saved generation failure reached before answering is scored incorrect without inventing an additional call. At $B = 1 . 5 \mathrm { k }$ , forced answers matched the exact saved cut. One outcome with an unavailable exact forced answer is omitted from the four-way comparison. Other positive budgets retain the nearest earlier forced answer when an exact record is unavailable (Appendix O). Relative-progress diagnostics use $c = \operatorname { r o u n d } ( f L )$ for $f \in \{ 0 . 1 , \ldots , 1 . 0 \}$ and invoke an answerer even at $f = 1$ . Here L is the observed saved trace length, including capped or failed traces, not necessarily a natural completion. These cuts are retrospective diagnostics. Table 1 gives the replay procedure. Answerer inputs exclude later reasoning, separately generated finalanswer content and gold labels, although the retained prefix may itself contain an explicit answer.

Adaptive stopping. Adaptive policies query Jev every $k ~ = ~ 5 0 0$ tokens through 16k, strictly before min $( B , L )$ . Stopping uses confidence and answer stability, settled/useful scores, or their logical OR (Appendix J). A trigger submits Jev’s current choice directly. Otherwise, the budget rule applies.

All checks are charged, including those preceding natural finishes.

Cost and latency. A policy specifies the reasoning budget, any intermediate stopping checks, and the final answerer. The policy cost is calculated as:

$$
C = C _ { \mathrm { r e a s o n e r } } + \sum _ { r \in \mathcal { R } } C _ { \mathrm { r e a d e r } } ( r ) + C _ { \mathrm { f o r c e d } } ,\tag{1}
$$

where R contains the policy’s reader checks and final calls. Costs cover the original prompt, retained output, and billable answerer usage, including caching and retries. $C _ { \mathrm { f o r c e d } } = 0$ when a reader answers. Offline acquisition expenses are excluded. Fixed-budget latency is estimated from retained reasoning tokens, measured decoding speed, and answer-call latency. These replay estimates do not establish online cancellation savings or asynchronous checking overhead.

## 3.5 Evaluation Protocol and Metrics

We report overall and cut-only accuracy, accuracy differences in percentage points (pp), and cost savings $1 0 0 ( 1 - \overline { { C } } _ { \pi } / \overline { { C } } _ { \mathrm { f u l l } } )$ within each pair. We also report estimated fixed-budget latency in seconds and as a percentage of the full run. Cross-pair summaries weight the 35 pairs equally.

For tuned caps and adaptive policies, we use fivefold cross-validation within each pair. In each split, four training folds select the lowest-cost policy subject to at most 1pp accuracy loss relative to the full run. If no candidate is feasible, the candidate with the highest training accuracy is selected. The selected policy is evaluated on the remaining held-out fold, and each question’s outcomes are averaged across the five repeats. The training constraint does not guarantee the same bound on held-out folds. Search grids appear in Appendix C.

In addition, paired intervals use 2,000 questionbootstrap resamples, stratified and weighted by difficulty on SuperGPQA. Positive-budget answerer comparisons use 100,000 resamples. The $B = 0$ row retains the earlier question-only analysis. For the primary Jev–forced comparison at $B = 1$ .5k, we apply Holm’s correction across the 35 pairs (Holm, 1979). Other intervals are uncorrected (Appendix S). Trace contribution is measured against each reader’s question-only answer on matched questions. Net gain equals the fix rate minus the hurt rate under a common denominator.

![](images/e3039f672917a42eef39b9c6e02d06895b0fdd251916d6505ac5e09a04b318e9.jpg)  
Figure 3: Accuracy of the four answerers, mean of the seven reasoners per benchmark. Top: fixed budget B, naturally completed runs keep the reasoner’s answer, so the curves approach one another at large budgets. Bottom: diagnostic cuts at a fraction of each saved trace’s length, known only after generation, and readers are evaluated even at 100%, rather than replaced by the natural answer. Readers see the last approximately 1,000 tokens, where forced answers generally use the whole prefix. Dashed line: full-run accuracy. Per-pair panels are in Appendix I.

## 4 Results

## 4.1 RQ1: Answer from truncated reasoning

Unless noted otherwise, results are macro-averages over the 35 pairs on the retained cohort of 15,377 model–question runs. Fixed-budget policies retain natural answers, whereas the percentage view evaluates truncated reasoning for every question. These are distinct evaluations, not interchangeable ways of setting an online budget.

Answering at a fixed cut. Jev has the highest mean accuracy among the four answerers at every tested budget from 0.5k to 16k tokens (Table 2). Its advantage over forced answering decreases from 8.4pp at 0.5k tokens to 5.1pp at 1k, 4.1pp at 1.5k and 3.2pp at 2k. At 1.5k, the respective accuracies are 75.8% and 71.7%, compared with 72.9% for Luna decision and 72.5% for Luna generative. Under the uncorrected 95% paired-bootstrap interval, Jev has higher accuracy than forced answering in 21 of 35 pairs and lower accuracy in none. Averaging over benchmarks within each reasoner, Jev leads all seven at this budget (Appendix E).

Overall versus cut-only accuracy. Natural answers are shared across policies and dilute the differences between answerers. On questions actually cut at 1.5k tokens, the mean Jev advantage is 8.6pp over forced answering, 6.2pp over Luna decision, and 6.9pp over Luna generative. At 32k tokens, the forced policy is 0.05pp higher overall. At 64k tokens, all four are within 0.1pp. Their convergence mainly reflects fewer cuts, not equivalent ability to use unfinished reasoning. Some runs never return a natural answer, so complete convergence is not required even at the largest budget.

Task and progress matter. At 1.5k, Jev exceeds forced answering by 9.2pp on GPQA-Diamond and 5.0pp on SuperGPQA, but by 0.9pp on MedQA and 2.4pp on MedXpertQA (Table 3). On both medical benchmarks it is within one point of the two Luna interfaces. The percentage curves (Figure 3, bottom) remove the natural-answer branch and show how reader accuracy changes with relative saved-trace progress. RQ3 examines these changes against each reader’s question-only baseline. The two Luna interfaces differ by less than one point in macro-accuracy. Thus, the decision interface alone does not reproduce Jev’s advantage in this control, and the comparison does not isolate the roles of model, training or backend differences.

Controlling the reasoning window. The main comparison gives forced answering the full prefix, whereas readers receive its final 1k tokens. We therefore conduct a post-hoc input-window control at B = 1.5k on 6,500 matched cut outcomes covering 1,589 unique questions. Both answerers receive either the full prefix or the identical saved 1k window, retaining their original answer-elicitation protocols. The mean Jev–forced gap is 8.43pp with full prefixes (exploratory 95% CI [7.05, 9.85]) and 7.96pp with matched 1k windows ([6.52, 9.44] in Table 4). Intervals use 20,000 paired questioncluster bootstrap resamples, sharing question multiplicities across reasoners and stratifying SuperG-PQA by difficulty. Thus, the observed gap persists under matched input windows. Model capabilities and answering interfaces remain different, so this control does not isolate an architectural effect.

<table><tr><td></td><td colspan="4">Accuracy (%)</td><td colspan="2">Cost (% of full run)</td><td colspan="3">Jev significantly better / worse (of 35 pairs)</td></tr><tr><td>B</td><td>Jev</td><td>Luna-dec.</td><td>Luna-gen.</td><td>Forced</td><td>Jev</td><td>Forced</td><td>vs. forced</td><td>vs. Luna-dec.</td><td>vs. Luna-gen.</td></tr><tr><td>0</td><td>68.0</td><td>54.2</td><td>55.2</td><td>53.4</td><td>1.7</td><td>4.7</td><td>28 /1</td><td>21 /2</td><td>21/7</td></tr><tr><td>0.5k</td><td>73.9</td><td>67.2</td><td>67.0</td><td>65.5</td><td>19.1</td><td>22.2</td><td>29/0</td><td>22/0</td><td>22/0</td></tr><tr><td>1k</td><td>75.2</td><td>71.1</td><td>70.3</td><td>70.1</td><td>28.2</td><td>31.4</td><td>25/ 0</td><td>20 /0</td><td>21 / 0</td></tr><tr><td>1.5k</td><td>75.8</td><td>72.9</td><td>72.5</td><td>71.7</td><td>34.5</td><td>38.3</td><td>21/0</td><td>14/1</td><td>20/ 0</td></tr><tr><td>2k</td><td>76.3</td><td>73.8</td><td>73.3</td><td>73.1</td><td>39.5</td><td>43.4</td><td>18/0</td><td>16 /0</td><td>18/0</td></tr><tr><td>4k</td><td>76.5</td><td>74.7</td><td>74.3</td><td>75.0</td><td>53.2</td><td>57.7</td><td>10/0</td><td>13 /0</td><td>15/0</td></tr><tr><td>8k</td><td>76.6</td><td>75.6</td><td>75.4</td><td>76.2</td><td>68.8</td><td>73.8</td><td>3/0</td><td>7/0</td><td>11/1</td></tr><tr><td>16k</td><td>77.0</td><td>76.5</td><td>76.1</td><td>76.8</td><td>84.1</td><td>88.3</td><td>4/0</td><td>6/0</td><td>6/0</td></tr><tr><td>32k</td><td>77.1</td><td>77.0</td><td>76.9</td><td>77.2</td><td>94.4</td><td>96.4</td><td>0/1</td><td>0/0</td><td>0/0</td></tr><tr><td>64k</td><td>77.2</td><td>77.1</td><td>77.1</td><td>77.1</td><td>99.8</td><td>101.3</td><td>0/0</td><td>0/0</td><td>1/0</td></tr></table>

Table 2: Fixed-budget policies, macro-average over 35 reasoner–benchmark pairs (full run 76.9%). Each policy’s estimated cost includes its retained generation and answering calls. Cost percentages are averaged within pairs, not ratios of pooled dollar totals. Significance counts refer to pairs whose uncorrected 95% paired-bootstrap interval for Jev-minus-comparator accuracy lies entirely above or below zero.
<table><tr><td colspan="3">B = 1.5k</td></tr><tr><td>Benchmark</td><td>Full Jev B=0</td><td>Jev L-d L-g Forced</td></tr><tr><td>GPQA-D</td><td>88.3 78.3</td><td>82.0 75.5 75.1 72.9</td></tr><tr><td>SuperGPQA 66.3</td><td></td></tr><tr><td>56.2 64.7 60.3 59.6</td><td>59.7</td></tr><tr><td>MedQA 95.1 48.9</td><td>88.6 95.0 94.9 94.9</td></tr><tr><td>32.2 49.9 49.3 48.9</td><td>94.1</td></tr><tr><td>MedXpertQA MMLU-Pro 86.1 84.5</td><td>47.5 87.4 84.5 84.0 84.3</td></tr></table>

Table 3: Accuracy (%), mean over seven reasoners per benchmark. $\operatorname { L - d } / \operatorname { L - g } \colon$ Luna decision/generative. $B = 0$ is question-only. B = 1.5k includes the shared naturalanswer branch. SuperGPQA is weighted by the full benchmark’s difficulty distribution.
<table><tr><td>Answerer</td><td>Reasoning window</td><td>Acc. (%)</td></tr><tr><td>Reasoner (forced)</td><td>Full prefix</td><td>56.41</td></tr><tr><td>Reasoner (forced)</td><td>Final 1k</td><td>57.01</td></tr><tr><td>Jev</td><td>Full prefix</td><td>64.84</td></tr><tr><td>Jev</td><td>Final 1k</td><td>64.97</td></tr></table>

Table 4: B = 1.5k. Accuracy is macro-averaged over 35 pairs on the same 6,500 exact-cut outcomes. Naturally completed runs are excluded. Final 1k denotes the saved window of approximately 1,000 reasoning tokens. Acquired unparseable answers count as incorrect.

Robustness and label sensitivity. Restricting the main fixed-budget comparison to the five strict-protocol reasoners gives a 2.9pp overall gap, smaller than the all-reasoner 4.1pp gap (Appendix P). For Jev versus forced answering at $B \ = \ 1 . 5 \mathrm { k } .$ , Holm-adjusted percentile-bootstrap tests with 100,000 resamples show significantly higher accuracy for Jev in 17 of 35 pairs and significantly lower accuracy in none. Paired tests with the same 35-comparison Holm correction identify 14 higher-accuracy pairs and none lower (Appendix S). The ordering also holds on the 15,309 runs with a valid natural answer (Appendix R).

Removing the 140 items flagged for model–label disagreement or manual review leaves a 4.4pp gap, with higher accuracy for Jev in 21 pairs and lower accuracy in none under the uncorrected 95% pairedbootstrap interval criterion (Appendix M). This is a sensitivity analysis, not a validated correction of benchmark labels.

## 4.2 RQ2: Static budgets and adaptive stops

Fixed-budget trade-offs. At $\textit { B } = \ 1 . 5 \mathrm { k }$ , Jev costs an estimated \$1.29 per 1,000 questions, compared with \$1.46 for forced answering and \$4.44 for the full run, averaging these costs over pairs. Its cost ratio is 34.5% of full, versus 38.3% for forced answering, when ratios are averaged by pair. The ratio of the mean costs is instead 29.2%. The corresponding accuracy changes against full are -1.13pp and -5.25pp. Estimated latency for Jev is 31.5% of the full run, averaged over pairs (Appendix G). These replay estimates do not establish online cancellation savings.

The 1.5k display point is chosen after inspecting the curve, not established as a universal optimum. Moving to 2k reduces the mean loss to 0.68pp while raising cost to 39.5% of full. At 1k, the loss is 1.72pp and cost is 28.2%. These averages indicate task differences: at 1.5k, Jev is 6.3pp below full on GPQA-Diamond and 1.6pp below on SuperGPQA, but 1.0pp above on MedXpertQA and 1.3pp above on MMLU-Pro. On answered runs alone, its mean loss is 1.4pp (Appendix R).

<table><tr><td colspan="3">vs. full run</td><td colspan="2">vs. tuned cap</td></tr><tr><td>Policy</td><td>Saving</td><td>∆Acc</td><td>∆Saving (+/-) ∆Acc (−)</td><td></td></tr><tr><td>Tuned cap</td><td>63.3</td><td>-0.13</td><td></td><td></td></tr><tr><td>conf+stable</td><td>68.8</td><td>-0.42</td><td>+5.5 (14/9)</td><td>−0.28 (3)</td></tr><tr><td>settled+useful</td><td>68.7</td><td>-0.61</td><td>+5.4 (19/1)</td><td>−0.47 (4)</td></tr><tr><td>either</td><td>70.8</td><td>-0.62</td><td>+7.5 (20/0)</td><td>−0.49 (5)</td></tr><tr><td>auto</td><td>71.3</td><td>-0.58</td><td>+8.0 (22/0)</td><td>−0.45 (5)</td></tr></table>

Table 5: Stop rules with Jev answering, out-of-fold selection, mean over 35 pairs. Saving in % of full run’s cost, and differences in points. In brackets: pairs significantly better/worse (saving) or significantly worse (accuracy). Checks every 500 tokens, where every Jev call is charged. auto selects the rule in the training folds.

On four matched subsets, standalone low-effort Luna costs less than all seven reasoner–Jev policies at B = 1.5k and is at least as accurate in 18 of 28 comparisons (Appendix H). Savings against full runs by themselves therefore do not establish an advantage over native effort control.

Tuned caps and adaptive increments. Out-offold selection chooses a separate static budget per pair and saves 63.3% at a mean change of −0.13pp (Table 5). This is not the uniform 1.5k policy. The adaptive comparisons use 500-token checkpoints up to 16k, a 1k reading window and fold-selected budgets. They do not use the doubling schedule explored elsewhere. Every answering and signalcheck call is charged. Question-only selection and cascades are additional controls in Appendix N, not substitutes for the matched cap comparison.

Relative to the tuned cap, conf+stable saves another 5.5 cost-percentage points at an additional 0.28pp accuracy loss, but is significantly more expensive in nine pairs. Settled+useful saves 5.4 points at an additional 0.47pp loss, and is significantly more expensive in one pair. Selecting the family within the training folds (auto) saves 71.3% overall: 8.0 points more than the tuned cap, at an additional 0.45pp accuracy loss. It is significantly cheaper in 22 pairs, with significant accuracy losses in five. These are jointly selected policies, so the difference includes both earlier stopping and changes in selected budgets.

What actually stops the run? For auto, 27.3% of outcomes end at a rule-triggered stop, 37.9% at the cap and 34.7% by natural completion, averaged over pairs and CV repeats. It makes 1.52 Jev calls per question, including final answering (Appendix T). Thus, a substantial share of the efficiency gain is available without an adaptive trigger. The incremental saving is concentrated on GPQA-Diamond (+21.0 points) and SuperGPQA (+12.6), versus about three points on the medical benchmarks and zero on MMLU-Pro, where the selected policies coincide.

Accuracy losses on held-out folds. Policy selection limits accuracy loss to 1pp relative to the full run on training folds, but this constraint may not hold on held-out folds. On GPQA-Diamond, auto loses 2.59pp versus 0.83pp for the tuned cap, while saving 59.6% versus 38.6% of full-run cost. Its additional saving therefore comes with a further 1.76pp accuracy loss.

Disabling early stopping while retaining the selected auto budgets reduces the loss to 1.35pp. Thus, the early-stop rule adds 1.24pp of loss at the same budgets. This GPQA-only post-hoc diagnostic does not reselect policies (Appendix J). Every auto training fold had a feasible candidate, ruling out the no-feasible-candidate fallback as the source of the held-out loss.

## 4.3 RQ3: Contribution of the reasoning trace

A paired diagnostic, not a knowledge test. We compare each reader’s percentage-cut answer with its question-only answer. This diagnostic retains 15,030 model–question outcomes covering 2,182 unique benchmark questions, requiring parseable answers from all three readers at 0, 10, 30, 50, 80 and 100%. It excludes 347 outcomes that fail this condition, unlike the main evaluation where refusals and unparseable answers count as wrong. The following pooled percentages are unweighted, including SuperGPQA, and descriptive. Several reasoners contribute traces for the same question. They should not be read as the macro-averages above or as 15,030 independent questions.

Net gains conceal both fixes and harms. Jev starts at 66.7% question-only accuracy on this cohort, versus 56.0% for Luna decision and 56.5% for Luna generative. At a 10% cut, its net gain of 2.5pp comprises 6.2% fixes and 3.7% harms. At 100%, 14.4% fixes and 5.2% harms yield a 9.2pp gain (Table 6). Luna gains more from its lower starting point, so the readers’ accuracies approach one another. Neither the higher starting accuracy nor the smaller net gain identifies how a model internally uses knowledge or reasoning.

<table><tr><td>Reader</td><td>Cut (%)</td><td>Fix (%) Hurt (%)</td><td>Net (pp)</td></tr><tr><td>Jev</td><td>10</td><td>6.2</td><td>3.7 +2.5</td></tr><tr><td rowspan="4">Luna dec.</td><td>50</td><td>12.0</td><td>4.9 +7.0</td></tr><tr><td>100</td><td>14.4</td><td>5.2 +9.2</td></tr><tr><td>10</td><td>8.9</td><td>5.3 +3.5</td></tr><tr><td>50</td><td>17.9 5.5</td><td>+12.4</td></tr><tr><td rowspan="4">Luna gen.</td><td>100</td><td>24.7</td><td>4.9 +19.8</td></tr><tr><td>10</td><td>10.2</td><td>7.4 +2.8</td></tr><tr><td>50</td><td>17.5</td><td>6.3 +11.3</td></tr><tr><td>100</td><td>24.2 5.1</td><td>+19.1</td></tr></table>

Table 6: Trace contribution on the common parseableanswer group (n = 15,030 outcomes). Fix: questiononly wrong, trace-conditioned right. Hurt: the reverse. Both divide by all included outcomes. Net gain equals fix minus hurt before rounding. A 100% cut supplies only the final 1k reasoning window. Conditional rates are presented in Appendix Q.

Different tasks benefit at different stages. Jev’s net gain at 10% is −0.4pp on GPQA-Diamond, +0.2pp on SuperGPQA and +0.3pp on MMLU-Pro, but +8.6pp on MedXpertQA (Appendix Q). GPQA-Diamond increases to +10.9pp at 100%, indicating that later trace windows can be substantially more useful than early ones. This complements the fixed-budget result: short cuts do not preserve GPQA accuracy even when Jev outperforms other answerers at those cuts. Relative progress is available only retrospectively, so these observations do not specify a deployable stopping rule.

Controlling the question-only starting point. Among 3,455 outcomes where all three readers start wrong, their accuracies at 100% are 34.2%, 34.0% and 33.7%, respectively. These similar descriptive rates do not establish equivalence. Among 6,597 outcomes where all start right, 97.8%, 96.5% and 94.6% remain correct at 10%. The observed advantage therefore includes retaining correct initial answers as well as recovering from wrong ones. Comparing conditional fix rates alone would confound different starting error sets (Appendix Q).

Completed traces as a boundary case. On the 15,309 runs with a valid natural answer, Jev reading the final approximately 1k reasoning window selects the same option as the reasoner on 99.1% on average across pairs. Its mean accuracy is 0.4pp higher, with 7 pairs having an uncorrected 95% paired-bootstrap interval entirely above zero and none below (Appendix L). High agreement does not establish copying or faithful verification.

## 5 Discussion

Our evaluation yields three main findings. First, Jev’s advantage persists under matched input windows. The fixed-cut comparisons show that answerer choice affects accuracy even without a learned stopping policy. On the exact-cut control cohort at B = 1.5k, with 6,500 matched cuts, Jev retains a 8.0pp macro-average advantage over forced answering when both receive the identical final 1k window (Section 4.1). The shorter reading window therefore does not alone explain the observed gap. These results support evaluating who answers separately from when reasoning stops. However, model capabilities and answer-elicitation interfaces remain different, so the comparison does not isolate the System-1 architectural advantage.

Second, adaptive savings come with taskdependent accuracy losses. Tuned caps provide substantial savings without intermediate checks. Joint adaptive selection adds 8.0 cost-percentage points of saving at a further 0.45pp accuracy loss on average, with larger losses on GPQA-Diamond (Section 4.2). Its benefit is therefore a taskdependent trade-off rather than a uniform improvement over static budgets. The GPQA results also show that a training-fold loss constraint is not a deployment guarantee. Budget, answerer and checking frequency should be evaluated against a strong static-budget baseline and the accuracy loss an application can tolerate.

Third, trace gains depend on the reader, task and stage. On the paired cohort, Jev has higher question-only accuracy, while Luna gains more from traces (Section 4.3). Within medicine, Jev starts high on MedQA but low on MedXpertQA, where early windows yield substantial gains. MMLU-Pro shows smaller increments, whereas GPQA-Diamond benefits from later windows despite high starting accuracy. These patterns motivate evaluating starting accuracy and trace benefit separately, rather than attributing benchmark differences to discipline alone. They describe behavioral changes, not internal mechanisms, and do not establish reasoning faithfulness. Percentage cuts are retrospective diagnostics, but not deployable stopping rules.

## 6 Conclusion

We evaluate System-1 decision models as answerers for truncated reasoning across seven LLM reasoners and five benchmarks. At a 1,500-token budget, Jev outperforms forced answering by 4.1 percentage points on average at about one-third of fullrun cost. Adaptive stopping offers further savings, with accuracy losses that vary by task. Questiononly controls show that readers’ standalone accuracy and their gains from reasoning traces both matter, while longer reasoning remains valuable on harder tasks. These findings highlight answerer choice alongside reasoning budget as a way to improve reasoning efficiency.

## Limitations

This study presents several directions for future research. First, we study multiple-choice tasks, one Jev system and one general-purpose control model through two interfaces. The findings do not establish advantages for all decision models, open-ended tasks or readers allowed additional reasoning. Same cut does not mean same input. Readers see an approximately 1k-token window, while forced answering generally receives the whole prefix. Forced-answer protocols also differ across reasoners, where an alternative protocol may perform better. The strict-only 1.5k gap is 2.9pp rather than 4.1pp. Luna’s two interfaces need not have identical backend implementations.

The reported policies are evaluated by offline replay, not by interrupting the saved run online. Costs use recorded usage and historical list prices, and latency is modelled from generation speeds and answer-call latencies. Neither measures cancellation billing, asynchronous overhead or actual deployment savings. The approach requires visible reasoning text, which need not reveal all internal computation. Token-to-text mapping is sometimes approximate. Forced answers are matched to the saved cut at 1.5k, and other budgets retain the nearest earlier collected position when exact answers are unavailable (Appendix O). Percentage cuts require the completed trace length and are diagnostic, not online policies.

The answerer comparison follows earlier analyses of these traces, and the 1.5k display point is post-hoc. Out-of-fold selection separates cap and threshold fitting from evaluation, but does not remove research-design selection effects or guarantee a one-point held-out loss. The four historical MMLU-Pro checks used an earlier protocol and are not independent confirmation (Appendix D). Per-pair intervals are mostly uncorrected. The shared questions and reasoners make comparisons dependent, and nonsignificance does not establish equivalence. RQ3 pools repeated model–question outcomes and excludes 347 outcomes lacking a parseable answer at any required point, so its descriptive rates may reflect selection effects. The GPQA diagnostic intervals condition on the recorded policy choices rather than refitting selection in each bootstrap.

Data and reproducibility. Traces and forced answers are single samples requested at temperature 1.0, not repeated estimates of generation variability.

The 408 retained 64k reruns are fresh, selectively collected samples, but other runs retain their original limits. The 67 retained no-answer runs and 1 saved failed-generation outcome count as wrong for full reasoning, as do reader refusals and unresolved parsing failures in the main evaluation. Public-benchmark exposure and label errors remain possible, but model–label disagreement is not proof that a gold label is wrong. SuperG-PQA main results are weighted estimates from a difficulty-balanced sample, unlike the unweighted RQ3 diagnostic. Hosted models and prices can change: Jev requests used the jev-latest alias, although the audited budget responses all reported jev-1.13.0. Saved outputs document the evaluated snapshot, not a guarantee of future reproducibility.

## Statement of LLM Use

During the preparation of this paper, the authors used LLM-based tools, primarily Claude and Chat-GPT, to assist in code development and debugging, improve grammar and clarity in draft text, and refine the layout of schematic figures. The authors reviewed the generated suggestions and take full responsibility for the final code, analyzes, figures, and manuscript.

## References

Alibaba Cloud. 2026. Qwen3.8-Flash released.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2025. Do NOT think that much for 2+3=? On the overthinking of long reasoning models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 9487–9499. PMLR.

Xinyuan Cheng, Beiduo Chen, Philipp Mondorf, and Barbara Plank. 2026. Reasoning that travels: Dissecting how chain-of-thought transfers across models. Preprint, arXiv:2605.28913.

DeepSeek-AI. 2026a. DeepSeek-V4-Flash-0731. Hugging Face model card. Accessed October 12, 2026.

DeepSeek-AI. 2026b. DeepSeek-V4-Pro-0813. Hugging Face model card. Accessed October 12, 2026.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Preprint, arXiv:2501.12948.

Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. 2026. Evaluating and benchmarking the system one model Jev. Preprint, arXiv:2609.37647.

Zhe Dong, Fang Qin, and Manish Shah. 2026. When does learning to stop help? a cost-aware study of early exits in reasoning models. Preprint, arXiv:2606.30852.

Feiyu Duan, Jiayu Lin, Jia Wang, Jun Xiang, Jialiang Wu, Xinnong Zhang, Hanqi Yan, Siyuan Wang, and Zhongyu Wei. 2026. General decision models: Benchmarking and insights beyond Jev. Preprint, arXiv:2610.03935.

Gongfan Fang, Xinyin Ma, and Xinchao Wang. 2025. Thinkless: Llm learns when to think. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 151268–151295. Curran Associates, Inc.

Yichao Fu, Xuewei Wang, Hao Zhang, Yuandong Tian, and Jiawei Zhao. 2026. Deep think with confidence. In International Conference on Learning Representations, volume 2026, pages 94355–94377.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile Van Krieken, and Pasquale Minervini. 2025. Are we done with MMLU? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5069–5096, Albuquerque, New Mexico. Association for Computational Linguistics.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. 2025. Tokenbudget-aware LLM reasoning. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 24842–24855, Vienna, Austria. Association for Computational Linguistics.

Sture Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70.

Dongming Jiang, Yi Li, and Bingzhe Li. 2026. Jevmem: System-one-controlled agentic memory for efficient ai agents. Preprint, arXiv:2609.23986.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Daniel Kahneman. 2011. Thinking, fast and slow. Farrar, Straus and Giroux.

Junyeob Kim, Sang-goo Lee, and Taeuk Kim. 2026. Think just enough: Leveraging self-assessed confidence for adaptive reasoning in language models.

In Findings of the Association for Computational Linguistics: EACL 2026, pages 5000–5006, Rabat, Morocco. Association for Computational Linguistics.

Yubo Li, Yidi Miao, Ramayya Krishnan, and Rema Padman. 2026. JEV-as-a-judge: Accept when confident, escalate when unsure. Preprint, arXiv:2609.26550.

Xin Liu and Lu Wang. 2025. Answer convergence as a signal for early stopping in reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17896–17907, Suzhou, China. Association for Computational Linguistics.

Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. 2025. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. Preprint, arXiv:2505.11896.

M-A-P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, Kaixin Deng, Shawn Gavin, Shian Jia, Sichao Jiang, Yiyan Liao, Rui Li, Qinrui Li, and 78 others. 2025. SuperGPQA: Scaling LLM evaluation across 285 graduate disciplines. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), Datasets and Benchmarks Track, volume 38. Curran Associates, Inc.

MiniMaxAI. 2026. MiniMax-M3. Hugging Face model card. Accessed October 12, 2026.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321, Suzhou, China. Association for Computational Linguistics.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M Kadous, and Ion Stoica. 2025. Routellm: Learning to route llms from preference data. In International Conference on Learning Representations, volume 2025, pages 34433–34448.

OpenAI. 2024. Learning to reason with LLMs. OpenAI Blog. Published September 12, 2024.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.

OpenAI. 2026a. Decisions. OpenAI API documentation. API released in beta October 6, 2026. Accessed October 11, 2026.

OpenAI. 2026b. GPT-6 Luna. OpenAI API model documentation. Model released September 22, 2026. Accessed October 11, 2026.

Qwen Team. 2026. Qwen3.7-Plus: Multimodal agent intelligence.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof Q&A benchmark. In First Conference on Language Modeling.

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, and Wei Wang. 2026. Stop when enough: Adaptive early-stopping for chain-of-thought reasoning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 27250–27268, San Diego, California, United States. Association for Computational Linguistics.

Together AI. 2026. Available models. Accessed October 12, 2026.

TypeSafe AI. 2026. Introducing system one models & Jev. Launch blog post by Diogo Almeida, published 15 September 2026. Model documentation: https: //docs.typesafe.ai/.

Xi Wang, James McInerney, Lequn Wang, and Nathan Kallus. 2026. Entropy after </think> for reasoning model early exiting. Preprint, arXiv:2509.26522.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track, volume 37, pages 95266–95290. Curran Associates, Inc.

Tiantong Wu and Wei Yang Bryan Lim. 2026. REFLEX with Jev for efficient selective control in LLM agents. Preprint, arXiv:2609.26532.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Qiaowei Li, Minghui Chen, Zheng Lin, and Weiping Wang. 2026. Dynamic early exit in reasoning models. In International Conference on Learning Representations, volume 2026, pages 88170–88210.

Z.ai. 2026. GLM-5.3-Flash. Hugging Face model card. Accessed October 12, 2026.

Yuxin Zuo, Shang Qu, Yifei Li, Zhang-Ren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. 2025. MedXpertQA: Benchmarking expert-level medical reasoning and understanding. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 80961–80990. PMLR.

## Appendix Contents

A Prompts and request settings 12   
B Data and completeness 13   
C Replay implementation and cost accounting 14   
C.1 Cut positions and reader windows 14   
C.2 Cost accounting details 14   
C.3 Trace-contribution denominators 15   
D Criteria fixed before four held-out cells 15   
E Per-reasoner results at 1.5k 15   
F Cost per reasoner 15   
G Latency analysis . 15   
H Native reasoning-effort baselines 15   
Per-pair curves 16   
Stop rules in detail 17   
K SuperGPQA by difficulty 19   
L Completed reasoning per pair 19   
M Model–label disagreement 19   
N Is reasoning needed? Question-only Jev as a policy 20   
O Exact cut positions for the forced answer 21   
P Forced-answer protocol 22   
Q Question-only baselines and trace contribution 23   
R Questions with a usable full-run answer 23   
S Multiple comparisons and loss bounds 23   
T Replay overhead of stopping rules 24

## A Prompts and request settings

Reasoners. All seven reasoners were called through the Together AI API between late September and early October 2026 with thinking enabled. The collection scripts requested temperature 1.0 and a 32,000-token output limit (64,000 for reruns):

Reasoner Model identifier In / out (\$/1M)   
DeepSeek V4 Pro deepseek-ai/DeepSeek-V4-Pro-0813 1.32 / 3.96   
Qwen3.7 Plus Qwen/Qwen3.7-Plus 0.32 / 1.28   
MiniMax M3 MiniMaxAI/MiniMax-M3 0.30 / 1.20   
GPT-OSS 120B openai/gpt-oss-120b (high) 0.15 / 0.60   
GLM-5.3 Flash zai-org/GLM-5.3-Flash 0.15 / 0.50   
Qwen3.8 Flash Qwen/Qwen3.8-Flash 0.15 / 0.47   
DeepSeek V4 Flash deepseek-ai/DeepSeek-V4-Flash-0731 0.14 / 0.28

## The full-run prompt is:

Reasoner, full run   
Answer the following multiple-choice   
question. Think step by step, then end your   
response with a single line of the form   
'Answer: X' where X is one of {option   
letters}.   
Question:   
{question}   
Options:   
A. {option A}   
B. {option B}

Forced answer, strict. A new chat request with thinking switched off (DeepSeek reasoning.enabled=false; Qwen enable\_thinking=false; MiniMax thinking\_mode=disabled), at most 48 output tokens; system and user messages:

Forced answer, strict   
System message   
You are being asked for an immediate final   
answer. Do not solve, explain or show any   
work. Reply with exactly two lines and   
nothing else:   
Answer: <letter>   
Confidence: <0-100>   
User message   
{full-run prompt}   
Below is your own reasoning so far on this   
question. It was cut off before you finished.   
<reasoning>   
{prefix}   
</reasoning>   
Based on this reasoning, give your best   
answer now and how confident you are that it   
is correct. Reply with exactly two lines:   
Answer: X   
Confidence: N   
where N is an integer from 0 to 100.

Forced answer, think-close. A raw completion request in the model’s template: the full-run prompt as the user turn, the prefix as the model’s thinking, the thinking block closed, and the answer started with the stub Answer:; 4 output tokens, the first option letter is taken. Templates were read from the model repositories on 4 October 2026.

Forced answer, think-close   
GLM-5.3 Flash   
[gMASK]<sop>   
<|system|>Reasoning Effort: Max   
<|user|>{full-run prompt}   
<|assistant|><think>{prefix}</think>Answer:   
GPT-OSS 120B (harmony format)   
<|start|>system<|message|>You are ChatGPT, a   
large language model trained by OpenAI.\n   
Knowledge cutoff: 2024-06\n\nReasoning:   
high\n\n# Valid channels: analysis,   
commentary, final. Channel must be included   
for every message.<|end|>   
<|start|>user<|message|>{full-run prompt}   
<|end|>   
<|start|>assistant<|channel|>analysis   
<|message|>{prefix}<|end|>   
<|start|>assistant<|channel|>final   
<|message|>Answer:   
Each prompt is one string. Lines start at turn boundaries for   
reading; \n marks a newline inside the prompt.

Jev. Requested as jev-latest; retained budget responses identify jev-1.13.0. Endpoint: POST /v1/systemone. State: question, lettered options, and partial\_reasoning (the last 1,000 tokens, prefixed with “. . . ” when cut; “(no reasoning yet)” when empty). Three typed questions in one call:

Jev, typed questions   
answer (choice; criteria = option texts cut to 300 characters)   
The state contains a multiple-choice question   
and a reasoner's partial chain of thought (it   
may be cut off mid-sentence, or empty). Which   
option is the correct answer?   
settled (yes/no probability)   
Has the partial reasoning already clearly   
committed to a final answer, so that   
continuing to think is unlikely to change it?   
useful (yes/no probability)   
If the reasoner keeps thinking, is it still   
likely to change or correct its current   
answer (for example: unresolved steps, open   
doubts, or a pending check that could flip   
it)?  
With B = 0 only the answer question is asked, with “The state contains a multiple-choice question”.

Luna. gpt-6-luna. Decisions API (POST /v1/decisions, public beta): the same question, options and window as text, the Jev answer instruction (“The input contains . . . ”), one choice per option letter with the option text (300 characters) as description; no fallback option. Generative (/v1/responses, reasoning effort none, at most 48 output tokens, 512 on the single retry):

the Jev answer instruction followed by the question, options and window, with the strict forced-answer system message and two-line output format.

Reader prices and request latencies. Jev costs \$0.042 per million input tokens, with median recorded latency 0.13 s over 92,763 unique budget calls. Luna Decisions costs \$0.10 per million input tokens, with median initial-call latency 0.14 s; Luna generation has median latency 0.96 s. Luna medians exclude re-asks and are acquisition statistics, not matched-window comparisons with Jev. All prices are recorded list prices, rather than independently verified per-request bills. The Luna interface comparison does not assume identical backend implementations.

## B Data and completeness

Table 7 summarizes sample counts, reruns, answer availability, and reasoning lengths for each reasoner–benchmark pair.

<table><tr><td>Reasoner</td><td>Benchmark</td><td></td><td>n Rerun 64k</td><td>No final Median answer</td><td>tokens</td><td>Finish ≤1.5k (%)</td></tr><tr><td>DeepSeek V4 Pro</td><td>GPQA-D</td><td>198</td><td>9</td><td>1</td><td>852</td><td>59</td></tr><tr><td></td><td>SuperGPQA</td><td>500</td><td>5</td><td>1</td><td>501</td><td>75</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>0</td><td>0</td><td>195</td><td>94</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>1</td><td>0</td><td>725</td><td>67</td></tr><tr><td>Qwen3.7 Plus</td><td>MMLU-Pro</td><td>500</td><td>3</td><td>0</td><td>172</td><td>86</td></tr><tr><td></td><td>GPQA-D</td><td>198</td><td>7</td><td>0</td><td>3,401</td><td>19</td></tr><tr><td></td><td>SuperGPQA</td><td>500</td><td>16</td><td>1</td><td>1,547</td><td>49</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>0</td><td>0</td><td>1,095</td><td>71</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>9</td><td>0</td><td>1,984</td><td>28</td></tr><tr><td>MiniMax M3</td><td>MMLU-Pro</td><td>500</td><td>1</td><td>0</td><td>997</td><td>69</td></tr><tr><td></td><td>GPQA-D</td><td>198</td><td>17</td><td>4</td><td>2,948</td><td>33</td></tr><tr><td></td><td>SuperGPQA</td><td>500</td><td>24</td><td>4</td><td>2,084</td><td>45</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>5</td><td>1</td><td>603</td><td>82</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>9</td><td>1</td><td>2,838</td><td>31</td></tr><tr><td>GPT-OSS 120B</td><td>MMLU-Pro</td><td>500</td><td>7</td><td>1</td><td>654</td><td>68</td></tr><tr><td></td><td>GPQA-D</td><td>194</td><td>13</td><td>0</td><td>4,005</td><td>26</td></tr><tr><td></td><td>SuperGPQA</td><td>496</td><td>13</td><td>2</td><td>2,738</td><td>32</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>0</td><td>0</td><td>683</td><td>79</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>0</td><td>0</td><td>2,114</td><td>33</td></tr><tr><td>GLM-5.3 Flash</td><td>MMLU-Pro</td><td>499</td><td>2</td><td>0</td><td>1,041</td><td>62</td></tr><tr><td></td><td>GPQA-D</td><td>198</td><td>22</td><td>8</td><td>4,538</td><td>33</td></tr><tr><td></td><td>SuperGPQA</td><td>500</td><td>31</td><td>7</td><td>2,045</td><td>43</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>2</td><td>0</td><td>636</td><td>70</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>10</td><td>0</td><td>6,719</td><td>15</td></tr><tr><td>Qwen3.8 Flash</td><td>MMLU-Pro</td><td>500</td><td>6</td><td>0</td><td>469</td><td>70</td></tr><tr><td></td><td>GPQA-D</td><td>198</td><td>34</td><td>9</td><td>6,089</td><td>27</td></tr><tr><td></td><td>SuperGPQA</td><td>500</td><td>42</td><td>7</td><td>2,981</td><td>39</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>4</td><td>0</td><td>535</td><td>72</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>50</td><td>4</td><td>7,833</td><td>15</td></tr><tr><td>DeepSeek V4 Flash GPQA-D</td><td>MMLU-Pro</td><td>500</td><td>14</td><td>0</td><td>435</td><td>69</td></tr><tr><td></td><td></td><td>198</td><td>11</td><td>3</td><td>1,053</td><td>56</td></tr><tr><td></td><td>SuperGPQA</td><td>500</td><td>23</td><td>5</td><td>471</td><td>70</td></tr><tr><td></td><td>MedQA</td><td>500</td><td>4</td><td>1</td><td>200</td><td>93</td></tr><tr><td></td><td>MedXpertQA</td><td>500</td><td>6</td><td>3</td><td>718</td><td>64</td></tr><tr><td></td><td>MMLU-Pro</td><td>500</td><td>8</td><td>5</td><td>171</td><td>85</td></tr></table>

Table 7: Per pair: retained runs, 64k reruns, outcomes without a valid natural answer (including generation failure, scored incorrect for the full run), median reasoning tokens, and share with a valid natural answer within 1,500 tokens.

The combined 64k rerun cohort contains 416 of 15,385 original runs: 413 have a length finish reason, and three lack valid answers despite other finish reasons. Each rerun generates a fresh sample rather than continuing the original trace. Failed requests were sometimes retried. Eight failed rerun requests are excluded, leaving 408 rerun traces: 341 with valid final answers and 67 without (59 with no option letter and eight with a letter only in truncated output). Ten requests that dropped the connection were repeated, with up to three attempts. The retained acquisition cohort contains 15,377 runs: 15,309 have valid natural answers, 67 are the no-validanswer rerun outcomes described above, and one additional saved full-generation stream ended with finish\_reason=error. The failed stream is retained and scored incorrect, not classified as a natural completion. At B = 1.5k, 15,376 runs enter the four-way comparison because one exact forced answer remains unavailable. The two fil tered forced-window requests for that outcome are treated as missing rather than scored as incorrect. Within each comparison, all answerers and the fullrun baseline use the same retained original or rerun trace. Selective rerun replacement is not an unconditional sample under a uniform 64k limit. For client-capped Qwen runs, the 32k and 64k limits are approximated by 115,200 and 230,400 reasoning characters, respectively. GPQA-Diamond, MedQA, MedXpertQA-Text and MMLU-Pro questions are used as released; the SuperGPQA sub set contains 167 easy, 166 middle, and 167 hard questions and is reweighted to the full dataset’s difficulty shares. Three requests of Qwen3.8 Flash on one SuperGPQA question are rejected by the provider’s input filter and count as wrong; no question is rephrased.

Luna acquisition. Deduplicated initial acquisition logs contain 191,637 calls per interface, covering question-only, budget and percentage probes before final evaluation-cohort filtering. There are 1,613 nonblocked generative parse failures (0.84%) and two blocked calls; the decision interface returns 1,107 refusals (0.58%). These are acquisition statistics, not final-cohort refusal rates. Refusals, blocked calls, and parse failures remaining after the configured re-ask count as incorrect.

Retry sensitivity. The retry increased Luna generative accuracy at the 100% cut from 74.0% to 77.0%. At all other cuts, accuracy changed by at most 0.4 percentage points.

## C Replay implementation and cost accounting

## C.1 Cut positions and reader windows

Positions use saved tokenizer offsets where available. Otherwise the nominal token position is mapped proportionally to a character offset and snapped back to a newline within 200 characters. Consequently the 1,000-token reader window is nominal, not an exact token count for every trace. At $B = 1 . 5 { \mathrm { k } }$ , 615 additional full-prefix forced answers complete exact-position coverage for 6,500 matched cut outcomes. One of the 6,501 target outcomes remains missing because both forcedwindow requests were filtered. It is excluded from matched comparisons. Other budgets retain the earlier-position fallback when an exact forced answer is unavailable. Appendix O reports the updated coverage and sensitivity analysis.

Relative-progress diagnostics. Cuts use $c _ { f } =$ round(fL) for $f \in \{ 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$ . Every point invokes an answerer, including at 100%, where the input contains completed reasoning but excludes separately generated final-answer content. Because L is known only after generation, these are retrospective diagnostics. Their answering mechanism and eligible cohort differ from the budget evaluation.

## C.2 Cost accounting details

Reasoner costs. Natural completions are charged for the original prompt and all generated output, including the final answer. Truncated runs are charged for the prompt and reasoning through the cut. Partial output usage is estimated from the saved trace, rather than a cancelled request’s billing record.

Answerer costs. Reader inputs include instructions, the question, options, and the reasoning window. Jev and Luna Decisions incur input charges only. Luna generation incurs input and output charges, including required re-asks. Forced answering incurs a separate call charged at the reasoner’s input and output prices, with cached input charged at the cached rate.

Usage, acquisition, and latency. Recorded request usage supplies token counts where available. Missing Luna Decisions input usage is estimated from serialized request length. Repeated windows are charged at every deployed check, even when acquisition reused a response. Acquiring full traces, probes, forced answers and unrelated failed retries incurs separate experimental expense, excluded from single-policy cost. Required answer-format re-asks belong to the selected answering chain and are included. Unknown usage is estimated and flagged, not treated as free. Latency estimation and its limitations are detailed in Appendix G.

## C.3 Trace-contribution denominators

A fix changes an incorrect question-only answer to a correct trace-conditioned answer, where a hurt is the reverse. Net trace gain equals the fix rate minus the hurt rate when both use a common denominator. Conditional fix and hurt rates instead divide by question-only errors and correct answers, respectively, and are not directly comparable. Pooled counts are model–question outcomes because a question may have traces from several reasoners. These outcomes are not independent questions. Trace gain measures behavioral contribution, not an internal knowledge source or reasoning faithfulness. Failure to detect an accuracy loss is not evidence of equivalence.

## D Criteria fixed before four held-out cells

Table 8 reports the six pre-specified criteria and their outcomes under the earlier frozen protocol.

The comparison of answerers was found on 31 reasoner–benchmark cells. Before the MMLU-Pro cells of DeepSeek V4 Pro, DeepSeek V4 Flash, MiniMax M3 and Qwen3.8 Flash had been seen, we fixed the analysis script (identified by its MD5 hash) and the six criteria below, and committed to report them whichever way they came out. Reader: last 12,000 characters of the prefix, forced answers: each reasoner’s main protocol, full runs with the original 32,000-token limit, paired bootstrap over questions, 2,000 resamples, 95% intervals. an interval excludes zero only if its bound is beyond 10<sup>−9</sup>. The formal check was run later: before it, the four cells had already entered aggregate figures and tables of later analyses. The frozen script was run unchanged. Only its output folder, and for the two DeepSeek models the folder it reads the MMLU-Pro data from, were set at run time. Its numbers are identical to those of the cell files produced on the day the data arrived. The later changes (64k reruns, 1,000-token window, budget-level cost model) are not part of this check.

## E Per-reasoner results at 1.5k

Table 9 reports accuracy and relative cost at B = 1.5k for each reasoner, aggregated over the five benchmarks.

## F Cost per reasoner

Figure 4 shows accuracy–cost curves for each reasoner, averaged over the five benchmarks.

## G Latency analysis

For a truncated run, latency is estimated as the retained reasoning length divided by the recorded full-run decoding speed, plus the answer-call latency. For natural finishes, we retain the full-run estimate without an additional answer call. Fullrun latency is likewise estimated from reasoning length and decoding speed.

At B = 1.5k, mean estimated latency is 16.5 s for Jev and 72.6 s for the full run across 35 reasoner– benchmark pairs. Averaging within-pair latency ratios gives 31.5%; the ratio of mean times is instead 22.7%. These summaries use unweighted question means within each pair and equal weights across pairs. The estimates do not separately account for time to first token, online cancellation, or asynchronous checking overhead. We therefore interpret them as decoding-based replay estimates rather than measured end-to-end response times.

## H Native reasoning-effort baselines

Matched-question comparison. We compare standalone GPT-6 Luna at reasoning efforts none, low, medium, and high with the current fixedbudget reasoner–Jev policies at $B = 1 . 5 { \bf k } .$ . Standalone Luna receives the question and options without external reasoning. The Jev policies retain natural answers within the budget and otherwise use the final approximately 1k-token reasoning window. All methods share the intersection of question IDs across the four Luna settings and seven retained reasoner cohorts: 194 GPQA-Diamond, 297 SuperGPQA, 500 MedQA, and 500 MedXpertQA questions. The Luna acquisition covers the earlier 300-question SuperGPQA sample and does not cover MMLU-Pro. Refusals and unparseable answers count as incorrect. SuperGPQA means use the full-dataset difficulty weights, consistent with the main evaluation.

Table 10 reports accuracy and total estimated cost at the recorded list prices. Jev policy costs include the reasoner prompt, retained output, and reader call. Luna costs use recorded input, cached input, and total output usage.

<table><tr><td>Criterion (all at B = 1,500 unless stated)</td><td>DS V4 Pro</td><td>DS V4 Flash</td><td>MiniMax M3</td><td>Qwen3.8 Flash</td><td>Result</td></tr><tr><td>1. Reader — forced on cut questions with an exact forced answer: positive in all four, CI excludes 0 in  $\geq 2$ </td><td> $+ 9 . 8 ^ { * }$ </td><td>+6.8</td><td> $+ 1 3 . 3 ^ { * }$ </td><td>+3.5</td><td>pass</td></tr><tr><td>2. Fixed budget + reader not significantly</td><td> $n = 6 1$ </td><td> $n = 5 9$ </td><td> $n = 1 3 5$ </td><td> $n = 1 1 3$ </td><td></td></tr><tr><td>below the full run in any cell</td><td> $+ 2 . 0 ^ { * }$ </td><td> $+ 2 . 0 ^ { * }$ </td><td> $+ 2 . 4 ^ { * }$ </td><td>+1.4</td><td>pass</td></tr><tr><td>3. Forced answers lose more than the reader (accuracy change vs full run, forced / reader)</td><td> $+ 0 . 8 / + 2 . 0$ </td><td> $0 . 0 / + 2 . 0$ </td><td> $- 2 . 2 / + 2 . 4$ </td><td> $- 0 . 2 / + 1 . 4$ </td><td>pass</td></tr><tr><td>4. Tuned cap (out of fold,  $t = 1 ) \colon$  reader saves more dollars than forced, CI excludes 0 in ≥ 3; cap + reader not significantly</td><td> $+ 1 0 . 5 ^ { * } / + 1 . 2$ </td><td> $+ 7 . 7 ^ { * } / + 1 . 4$ </td><td> $+ 3 7 . 1 ^ { * } \ / + 2 . 6 ^ { * }$ </td><td> $+ 7 . 2 ^ { * } \ / + 0 . 6$ </td><td>pass</td></tr><tr><td>below the full run 5. Completed reasoning, limit-hit traces removed: reader within 1 point of the rea- soner</td><td>+0.4</td><td>+0.2</td><td> $+ 1 . 2 ^ { * }$ </td><td>+0.2</td><td>fail</td></tr><tr><td>6. Stop rule + reader – cap + reader, dollars: |mean| &lt; 3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>pass†</td></tr></table>

Table 8: Pre-specified criteria on the four held-out MMLU-Pro cells. Differences in points; <sup>∗</sup> 95% paired-bootstrap interval excludes 0. Row 4 gives the dollar-saving difference / the accuracy change of cap + reader against the full run. Criterion 5 fails because the reader on the completed reasoning of MiniMax M3 is 1.2 points more accurate than the reasoner’s own answer [0.4, 2.4]; the stated tolerance was symmetric. <sup>†</sup>Trivially: on all four cells the out-of-fold selection chose a cap of 500 tokens in every fold; the stop rule checks every 500 tokens, so it has no chance to stop before the cap and the two policies coincide.

<table><tr><td rowspan="2">Reasoner</td><td rowspan="2"></td><td colspan="3">Accuracy (%)</td><td colspan="2">Cost (%)</td></tr><tr><td>Full Jev</td><td>L-d</td><td>L-g</td><td>F</td><td>Jev F</td></tr><tr><td>DeepSeek V4 Pro</td><td>79.0</td><td>77.7</td><td>76.8 76.7</td><td>76.5</td><td>35.3</td><td>38.7</td></tr><tr><td>Qwen3.7 Plus</td><td></td><td>79.1 77.8</td><td>74.574.0</td><td>74.3</td><td></td><td>35.240.6</td></tr><tr><td>MiniMax M3</td><td></td><td></td><td>75.8 74.1 70.1 70.6 69.3 27.2 32.2</td><td></td><td></td><td></td></tr><tr><td>GPT-OSS 120B</td><td></td><td></td><td>70.0 70.2 64.3 63.4 59.9 30.3 34.1</td><td></td><td></td><td></td></tr><tr><td>GLM-5.3 Flash</td><td></td><td></td><td>78.8 77.3 75.4 74.3 73.3 20.0 21.3</td><td></td><td></td><td></td></tr><tr><td>Qwen3.8 Flash</td><td></td><td></td><td>77.6 77.075.1 74.4 74.3 16.2 19.4</td><td></td><td></td><td></td></tr><tr><td>DeepSeek V4 Flash 78.2 76.4 74.0 74.2 74.3 28.9 32.0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Accuracy is averaged over five benchmarks at B = 1.5k on the matched cohort. Cost is the ratio of mean policy dollars to mean full-run dollars across those benchmarks, unlike the main table’s mean of within-pair ratios. Full-run cost per 1,000 questions, in row order: \$8.86, 5.46, 6.06, 2.79, 3.22, 3.83, 0.84.

Effort and output usage. Table 11 reports Luna’s total output tokens, including hidden reasoning. Low effort substantially reduces output usage relative to high effort on these subsets, but the accuracy differences vary by benchmark. For example, on MedXpertQA, low uses 331 output tokens per question at 53.4% accuracy, versus 924 tokens and 53.0% for high. Similar point estimates do not establish equivalence.

Interpretation and scope. At $B = 1 . 5 \mathrm { k } .$ loweffort Luna has lower estimated cost in all 28 reasoner–benchmark comparisons and at least as high accuracy in 18. Thus, the savings against full reasoner runs do not establish superiority over native effort control or alternative standalone models. This comparison changes the model as well as the answering procedure; it does not isolate effort within each of the seven reasoners or establish the best effort setting. The comparisons share questions and are not independent replications. These counts describe the 1.5k display budget, not tuned caps, adaptive policies, or the complete budget.

Luna was acquired earlier than the current reader experiments, and its historical refusal retries remain in the selected records. Usage-based costs describe selected calls and exclude earlier retry attempts; zero recorded usage on blocked calls is not verified zero billing. Jev costs are replay estimates, not observed online cancellation savings. Legacy Luna logs do not retain full question text. Their question provenance follows the collection script’s source traces. API model identifiers do not establish immutable backend revisions.

## I Per-pair curves

Figures 5 and 6 report accuracy for all 35 pairs under fixed budgets and relative-progress cuts, respectively.

![](images/4b6fd6a4f5d203ff33dbf142cc85ce19cd45a3f4c09aa8391dbd97a14af8735a.jpg)  
Cost per 1,000 questions (USD, log scale, list prices), mean over the five benchmarks. Points left to right: B = 0.5k, 1k, 1.5k, 2k, 4k, 8k, 12k, 16k, 32k, 64k.

Figure 4: Accuracy against cost per reasoner, mean over the five benchmarks, in dollars per 1,000 questions at list prices (log scale). Each line traces the budgets B = 0.5k to 64k; labels mark Jev’s points. Open circle: full run.

## J Stop rules in detail

Adaptive replay rules. At checkpoint $j ,$ Jev returns option probabilities $p _ { j } ( a )$ , with

$$
\hat { a } _ { j } = \arg \operatorname* { m a x } _ { a \in \mathcal { O } } p _ { j } ( a ) , \qquad h _ { j } = \operatorname* { m a x } _ { a \in \mathcal { O } } p _ { j } ( a ) .\tag{2}
$$

The conf+stable rule is

$$
j \geq 2 , \qquad h _ { j } \geq \tau , \qquad \hat { a } _ { j } = \hat { a } _ { j - 1 } .\tag{3}
$$

Only the current confidence is thresholded. The settled+useful rule is

$$
s _ { j } \geq t _ { s } \quad \mathrm { a n d } \quad u _ { j } \leq t _ { u } ,\tag{4}
$$

where $s _ { j }$ and $u _ { j }$ are separately elicited scores for a submittable answer and the usefulness of further reasoning. They are returned in the same request and are not derived from option probabilities. The either rule is their logical OR. The check horizon is 16k tokens, independently of the cap B. Table 12 describes the logical replay branches. Replay check overhead is summarized in Appendix T.

Policy selection. Tuned caps and stopping policies are selected separately for each reasoner– benchmark pair using five-fold cross-validation repeated five times, split by question. Selection minimizes training cost subject to at most one percentage point of accuracy loss against the full run. If no candidate is feasible, the candidate with highest training accuracy is selected. The auto setting also chooses the rule family within training folds. Each question is evaluated using a policy selected using data from the other folds. Its held-out outcomes are then averaged across the five repeats. The training constraint does not guarantee a corresponding loss bound on held-out questions. The 1.5k presentation point does not restrict the caps of other policies. Question-only Jev (B = 0) is a standalone-reader control. Expanded cap grids and cascades appear in Appendix N.

Selection grids. The budget grid is given by B ∈ {0.5, 1, 1.5, 2, 4, 8, 12, 16, 32, 64}k.

$$
\tau \in \{ 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 0 . 9 8 \} .
$$

$t _ { s } ~ \in ~ \{ 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 \}$ , and $t _ { u } \in \{ 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 1 . 0 \}$ . Either uses the same $t _ { s } , t _ { u }$ grids with $\tau \in \{ 0 . 8 , 0 . 9 , 0 . 9 5 \}$ . Adaptive checks are at 500-token multiples through 16k, strictly before min(B, L).

Table 13 summarizes benchmark-level accuracy and cost savings for the tuned cap and auto policies. The range of the per-pair saving difference against the tuned cap is −8.4 to +37.9 points for conf+stable, −6.5 to +30.5 for settled+useful, −0.4 to +38.6 for either and 0.0 to +39.1 for auto. On MMLU-Pro, all seven reasoners select a 500- token cap in every fold. No scheduled check precedes this cap, so the stopping policies coincide with the tuned cap. Table 14 lists the most frequent policy selections for each pair.

<table><tr><td rowspan="2">Method</td><td colspan="2">GPQA-D (n = 194)</td><td colspan="2">SuperGPQA (n = 297)</td><td colspan="2">MedQA (n = 500)</td><td colspan="2">MedXpertQA (n = 500)</td></tr><tr><td> $\operatorname { A c c } .$ </td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td></tr><tr><td>Luna none</td><td>64.9</td><td>0.087</td><td>53.4</td><td>0.071</td><td>91.4</td><td>0.059</td><td>42.4</td><td>0.066</td></tr><tr><td>Luna low</td><td>85.1</td><td>0.305</td><td>65.0</td><td>0.218</td><td>95.0</td><td>0.113</td><td>53.4</td><td>0.197</td></tr><tr><td>Luna medium</td><td>88.1</td><td>0.629</td><td>69.0</td><td>0.355</td><td>95.4</td><td>0.147</td><td>53.2</td><td>0.329</td></tr><tr><td>Luna high</td><td>88.1</td><td>1.028</td><td>67.1</td><td>0.562</td><td>95.6</td><td>0.181</td><td>53.0</td><td>0.494</td></tr><tr><td>Reasoner + Jev,</td><td> $B = 1 . 5 \mathrm { k }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">DS V4 Pro 84.5</td><td>4.272</td><td>68.2</td><td></td><td>3.218 95.6</td><td>1.876</td><td>53.0</td><td>3.946</td></tr><tr><td>Qwen3.7 Plus</td><td>84.5</td><td>2.100</td><td>67.6</td><td>1.871</td><td>95.2</td><td>1.885</td><td>55.4</td><td>2.094</td></tr><tr><td>MiniMax M3</td><td>83.0</td><td>1.864</td><td>64.4</td><td>1.671</td><td>93.2</td><td>1.420</td><td>43.8</td><td>1.924</td></tr><tr><td>GPT-OSS 120B</td><td>75.3</td><td>0.974</td><td>53.1</td><td>0.916</td><td>94.4</td><td>0.646</td><td>42.2</td><td>0.957</td></tr><tr><td>GLM-5.3 Flash</td><td>85.1</td><td>0.738</td><td>68.5</td><td>0.657</td><td>95.0</td><td>0.534</td><td>51.6</td><td>0.821</td></tr><tr><td>Qwen3.8 Flash</td><td>83.0</td><td>0.732</td><td>67.5</td><td>0.648</td><td>96.0</td><td>0.463</td><td>52.0</td><td>0.793</td></tr><tr><td>DS V4 Flash</td><td>84.0</td><td>0.337</td><td>65.5</td><td></td><td>0.242 95.6</td><td>0.136</td><td>51.2</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.316</td></tr></table>

Table 10: Standalone Luna effort baselines and fixed-budget Jev policies on shared questions. Accuracy in %; estimated cost in USD per 1,000 questions. All rows use the same questions within each benchmark, with difficulty weighting on SuperGPQA. Point estimates are descriptive, without significance claims.

<table><tr><td>Effort</td><td>GPQA-D</td><td>SuperGPQA</td><td>MedQA</td><td>MedX.</td></tr><tr><td>none</td><td>123</td><td>89</td><td>66</td><td>69</td></tr><tr><td>low</td><td>559</td><td>384</td><td>176</td><td>331</td></tr><tr><td>medium</td><td>1207</td><td>658</td><td>243</td><td>593</td></tr><tr><td>high</td><td>2006</td><td>1071</td><td>311</td><td>924</td></tr></table>

Table 11: Mean total output tokens for standalone Luna on the matched subsets of Table 10. Reasoning tokens are included in output usage and are not added again.

When checks occur every 1,000 rather than 500 tokens, conf+stable achieves 68.2% cost saving with an accuracy change of −0.35 points. Its saving exceeds that of the tuned cap by 4.9 points, with significantly lower costs in 10 pairs and higher costs in seven. For settled+useful, the corresponding saving is 68.6% and the accuracy change is −0.46 points. Its saving exceeds that of the tuned cap by 5.3 points, with significantly lower costs in 14 pairs and higher costs in one.

GPQA same-budget diagnostic. An offline recomputation on 2026-10-11 reproduces all seven GPQA cells in the current stop\_deploy\_a9 snapshot. Mean full accuracy is 88.32%; tuned cap gives 87.49% at 38.56% cost saving, and auto gives 85.73% at 59.55% saving. Its additional 1.76pp loss against cap combines different selected budgets and within-budget stopping. Disabling the rule while retaining each fold’s selected auto budget yields 86.97% accuracy: the same-budget rule difference is −1.24pp, and the remaining budgetselection difference is −0.52pp. This is a post-hoc diagnostic, not an independently fitted deployment policy. All 175 auto training folds have a candidate satisfying the one-point training constraint, but 95 test-fold point estimates exceed a one-point loss. The overlapping folds are not independent trials. After averaging the five CV repeats per question, a paired bootstrap jointly resampling original question IDs across the seven reasoners (10,000 draws) gives a 95% interval of [−4.49, −0.73]pp for auto versus full and [−2.33, −0.23]pp for enabling the rule at the same budgets. These intervals condition on the recorded selections, do not refit CV and are not multiplicity-adjusted. The interval for auto versus full includes a one-point loss. Although the estimated loss exceeds the selection target, the interval does not establish that the population loss

Procedure: adaptive Jev replay   
Input: retained trace, cap B, window w, interval k,   
selected stopping rule.   
1 for each $\hat { c } \in \{ k , 2 k , \ldots , 1 6 , 0 0 0 \}$ with $c \mathrm { ~ \textless ~ }$   
min $( B , L )$ do   
2 Retrieve Jev’s scores from $( q , { \mathcal { O } } ,$ Tail<sub>w</sub> $( z _ { \leq c } ) )$ and   
charge the check.   
3 if the stopping rule triggers then   
4 Submit the current top option and terminate re  
play.   
5 end if   
6 end for   
7 if generation failed and $L \leq B$ then   
8 Record an incorrect outcome. Retain generation   
and check costs, with no final answer call.   
9 else if a valid natural answer exists and $L \leq B$ then   
10 Retain the natural answer.   
11 else   
12 Retrieve Jev’s answer at min $( B , L )$ and charge the   
final call.   
13 end if  
Table 12: Logical adaptive replay. Earlier checks remain charged when a run later finishes. No additional forcedanswer call follows a rule-triggered stop.

![](images/68d3572cee5dc20c5e905c68a25da7960cc99c481058cd374c68c0b76ade7f55.jpg)  
Reasoning budget B (tokens generated before the cut)

Figure 5: Accuracy of the four answerers for every reasoner–benchmark pair against the reasoning budget B. Dashed line: full run.
<table><tr><td></td><td>Tuned cap</td><td>auto</td><td>Uncapped</td></tr><tr><td>Benchmark</td><td>Save Acc</td><td>Save Acc</td><td>Acc</td></tr><tr><td>GPQA-D</td><td>38.6 87.5</td><td>59.6 85.7</td><td>88.3</td></tr><tr><td>SuperGPQA</td><td>64.4 65.4</td><td>77.0 65.3</td><td>66.3</td></tr><tr><td>MedQA</td><td>57.4 94.8</td><td>60.8 94.8</td><td>95.1</td></tr><tr><td>MedXpertQA</td><td>76.5 49.3</td><td>79.6 48.9</td><td>48.9</td></tr><tr><td>MMLU-Pro</td><td>79.6 87.1</td><td>79.6 87.1</td><td>86.1</td></tr></table>

Table 13: Tuned cap and auto per benchmark, mean of seven reasoners, held-out folds. Saving in % of the full-run cost, accuracy in %.

exceeds that target.

## K SuperGPQA by difficulty

Table 15 reports fixed-budget accuracy and relative cost by official difficulty level, averaged over the seven reasoners.

At 1.5k tokens, Jev reaches 69.2%, 66.9% and 56.1% on easy, middle and hard questions, respectively, compared with 69.1%, 67.7% and 61.0% for the full run. The Jev advantage is concentrated on hard questions, where cuts are most frequent; 157 of the 167 hard questions are science or engineering. This analysis was added after the main results and is exploratory.

## L Completed reasoning per pair

Table 16 compares the reasoner’s natural answer with Jev reading completed reasoning, restricted to runs with valid final answers.

## M Model–label disagreement

Table 17 reports accuracy gaps before and after removing flagged questions.

Agreement of all seven runs on a non-gold option warrants label review but is not evidence of a label error: models may share knowledge gaps or misinterpret a question similarly. GPQA-Diamond items 106 and 17 meet this agreement criterion. Of items 17, 89, and 126 checked by hand, only item 17 has independent support for a wrong label; item 89 is ambiguous and item 126 repeats a distractor. Flagged counts are therefore not labelerror rates, and findings on MMLU (Gema et al., 2025) do not establish errors in our MMLU-Pro items. At B = 1.5k, removing the 140 flagged questions changes the mean Jev–forced gap from +4.1pp to +4.4pp. In both cases, uncorrected paired-bootstrap intervals identify higher accuracy in 21 pairs and lower accuracy in none. Flagging is a sensitivity screen, not a validated relabelling procedure.

![](images/2c2e6bae06b0ddb9fe12e6658021c3eebddbc2854fcb36627250ceb91648276e.jpg)  
Figure 6: As Figure 5, with every question cut at the same fraction of its own reasoning (diagnostic view).

## N Is reasoning needed? Question-only Jev as a policy

We add two kinds of policy to the out-of-fold policy selection. (i) B = 0 in the cap grid: Jev answers from the question alone and the reasoner is not called. (ii) Cascades: Jev first answers from the question alone. If its top probability is at least g, the answer is submitted. Otherwise, the question is escalated either to the full run (Jev→full) or to a cap with Jev reading the prefix (Jev→cap). The gate threshold g (0.3 to 0.98, or never) and B are selected inside the training folds; the gate call is charged to every question. Because a single tolerance hides the trade-off, Table 18 reports each policy at four training-fold tolerances.

Selecting the question-only baseline. At tolerance 1, allowing B = 0 selects question-only Jev in most folds for four pairs: GPT-OSS 120B on GPQA-Diamond, SuperGPQA and MMLU-Pro, and MiniMax M3 on MMLU-Pro. Across 35 pairs, mean cost saving increases from 63.3% to 65.2%, with a mean accuracy change of −0.03 points relative to the tuned cap. Costs are significantly lower in eight pairs and higher in none, whereas accuracy is significantly lower in four pairs (unadjusted 95% paired-bootstrap intervals).

<table><tr><td>Reasoner</td><td>Benchmark</td><td>Tuned cap</td><td>auto: family (thresholds), cap</td><td>∆Saving</td></tr><tr><td>DeepSeek V4 Pro</td><td>GPQA-D</td><td>16k (20)</td><td>either (0.95, 0.9, 0.5), 16k (12)</td><td>+13.3</td></tr><tr><td rowspan="7">Qwen3.7 Plus</td><td>SuperGPQA</td><td>1.5k (14)</td><td>s+u (0.2, 1.0), 1k (9)</td><td>+8.5</td></tr><tr><td>MedQA</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>MedXpertQA</td><td>1k (23)</td><td>s+u (0.3, 0.7), 1k (13)</td><td>+3.0</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>GPQA-D</td><td>8k (19)</td><td>either (0.9, 0.9, 0.5), 8k (3)</td><td>+26.1</td></tr><tr><td>SuperGPQA</td><td>8k (13)</td><td>c+s τ=0.5, 4k (14)</td><td>+21.7</td></tr><tr><td>MedQA</td><td>0.5k (15)</td><td>cap, 0.5k (15)</td><td>+11.2</td></tr><tr><td rowspan="5">MiniMax M3</td><td>MedXpertQA</td><td>1.5k (20)</td><td>either (0.8, 0.4, 0.8), 1.5k (5)</td><td>+16.3</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>GPQA-D</td><td>32k (14)</td><td>either (0.95, 0.2, 0.5), 12k (5)</td><td>+39.1</td></tr><tr><td>SuperGPQA</td><td>8k (24)</td><td>c+s τ=0.5, 4k (18)</td><td>+37.9</td></tr><tr><td>MedQA</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td rowspan="5">GPT-OSS 120B</td><td>MedXpertQA</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>GPQA-D</td><td>4k (14)</td><td>cap, 0.5k (10)</td><td>+7.6</td></tr><tr><td>SuperGPQA</td><td>0.5k (21)</td><td>cap, 0.5k (21)</td><td>+3.5</td></tr><tr><td>MedQA</td><td>0.5k (14)</td><td>cap, 0.5k (14)</td><td>+6.4</td></tr><tr><td rowspan="5">GLM-5.3 Flash</td><td>MedXpertQA</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>GPQA-D</td><td>16k (23)</td><td>c+s τ=0.85, 16k (8)</td><td>+26.0</td></tr><tr><td>SuperGPQA</td><td>1.5k (17)</td><td>either (0.8, 0.2, 1.0), 1.5k (14)</td><td>+4.7</td></tr><tr><td>MedQA</td><td>1k (19)</td><td>s+u (0.2, 1.0), 1k (15)</td><td>+5.9</td></tr><tr><td rowspan="5">Qwen3.8 Flash</td><td>MedXpertQA</td><td>1k (25)</td><td>s+u (0.2, 0.8), 1k (22)</td><td>+2.0</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>GPQA-D</td><td>12k (16)</td><td>c+s τ=0.7, 8k (11)</td><td>+27.9</td></tr><tr><td>SuperGPQA</td><td>1.5k (22)</td><td>either (0.8, 0.2, 1.0), 1.5k (10)</td><td>+2.9</td></tr><tr><td>MedQA</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td rowspan="7">DeepSeek V4 Flash</td><td>MedXpertQA</td><td>0.5k (14)</td><td>cap, 0.5k (14)</td><td>+0.6</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr><tr><td>GPQA-D</td><td>32k (24)</td><td>cap, 32k (14)</td><td>+6.9</td></tr><tr><td>SuperGPQA</td><td>8k (12)</td><td>cap, 2k (7)</td><td>+8.9</td></tr><tr><td>MedQA</td><td>0.5k (19)</td><td>cap, 0.5k (19)</td><td>+0.0</td></tr><tr><td>MedXpertQA</td><td>1k (21)</td><td>cap, 1k (21)</td><td>+0.1</td></tr><tr><td>MMLU-Pro</td><td>0.5k (25)</td><td>cap, 0.5k (25)</td><td>+0.0</td></tr></table>

Table 14: Most frequent selection over the 25 training folds (count in brackets). c+s: conf+stable (τ); s+u: settled+useful $( t _ { s } , t _ { u } )$ ; either: $( \tau , t _ { s } , t _ { u } )$ . ∆Saving: auto minus tuned cap, points of the full run’s cost.

<table><tr><td></td><td>Easy</td><td>Middle</td><td>Hard</td></tr><tr><td>Full run</td><td>69.1</td><td>67.7</td><td>61.0</td></tr><tr><td>Forced, 1.5k Jev, 1.5k</td><td>67.5 (52.5) 69.2 (48.5)</td><td>63.1 (36.6) 66.9 (32.7)</td><td>45.4 (19.0) 56.1 (16.3)</td></tr></table>

Table 15: SuperGPQA by official difficulty, accuracy in % (cost as % of the full run). Questions are unweighted within each level, then seven reasoners are macro-averaged. The original sample contains 167 / 166 / 167 questions. One Qwen3.8 Flash outcome is excluded from the matched 1.5k comparison because its exact forced answer is unavailable. Tuned-policy results under the main selection protocol are reported in Table 13.

Escalating to the full run is not competitive. Answering confident questions with Jev alone and running the reasoner in full on the rest saves only

33–48%, less than any cap at a similar accuracy: buying the whole trace for uncertain questions costs more than buying a short prefix for all of them.

Escalating to a cap is. Jev→cap lies on a better frontier than the cap alone: at −0.83pp it saves 73.3% against 68.7%, and at about 71.5% saving it loses 0.70pp against 0.96pp. It is comparable to the auto stop rule. The gate submits 79–93% of MedQA and MMLU-Pro questions, 53% of GPQA-Diamond and about 21% of MedXpertQA. On heldout folds, overall accuracy drops by 1.21pp relative to the full run, exceeding the 1pp tolerance used for policy selection on training folds.

## O Exact cut positions for the forced answer

Table 19 reports current exact-position coverage. The 1.5k comparison uses the completed exactforced overlay, and other budgets retain their exist-

<table><tr><td>Benchmark</td><td>Model</td><td>n</td><td>R Acc. (%)</td><td>Jev Acc. (%)</td><td>∆(pp)</td><td>95% CI (pp)</td><td>Agree (%)</td><td>Jev+/R+</td></tr><tr><td rowspan="7">GPQA-D</td><td>DeepSeek V4 Pro</td><td>197</td><td>93.4</td><td>93.4</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>Qwen3.7 Plus</td><td>198</td><td>87.9</td><td>88.4</td><td>+0.5</td><td>[0.0, 1.5]</td><td>98.5</td><td>1/0</td></tr><tr><td>MiniMax M3</td><td>194</td><td>93.3</td><td>93.3</td><td>+0.0</td><td>[-1.5, 1.5]</td><td>99.0</td><td>1/1</td></tr><tr><td>GPT-OSS 120B</td><td>194</td><td>77.8</td><td>78.9</td><td>+1.0</td><td>[0.0, 2.6]</td><td>98.5</td><td>2/0</td></tr><tr><td>GLM-5.3 Flash</td><td>190</td><td>94.2</td><td>94.2</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>Qwen3.8 Flash</td><td>189</td><td>91.0</td><td>93.1</td><td> $+ 2 . 1 ^ { * }$ </td><td>[0.5, 4.2]</td><td>97.9</td><td>4/0</td></tr><tr><td>DeepSeek V4 Flash</td><td>195</td><td>92.3</td><td>92.3</td><td>+0.0</td><td>[0.0, 0.0]</td><td>99.5</td><td>0/0</td></tr><tr><td rowspan="7">SuperGPQA</td><td>DeepSeek V4 Pro</td><td>499</td><td>69.0</td><td>69.3</td><td>+0.3</td><td>[-0.3, 1.1]</td><td>99.4</td><td>2/1</td></tr><tr><td>Qwen3.7 Plus</td><td>499</td><td>69.5</td><td>70.0</td><td>+0.5</td><td>[-0.2, 1.4]</td><td>99.0</td><td>4/1</td></tr><tr><td>MiniMax M3</td><td>496</td><td>67.0</td><td>68.3</td><td> $+ 1 . 2 ^ { * }$ </td><td>[0.3, 2.3]</td><td>98.0</td><td>6/0</td></tr><tr><td>GPT-OSS 120B</td><td>494</td><td>56.4</td><td>57.1</td><td> $+ 0 . 7 ^ { * }$ </td><td>[0.2, 1.4]</td><td>98.0</td><td>4/0</td></tr><tr><td>GLM-5.3 Flash</td><td>493</td><td>67.8</td><td>67.6</td><td>-0.2</td><td>[-0.5, 0.0]</td><td>99.6</td><td>0/1</td></tr><tr><td>Qwen3.8 Flash</td><td>493</td><td>68.2</td><td>69.2</td><td> $+ 1 . 0 ^ { * }$ </td><td>[0.3, 1.8]</td><td>97.8</td><td>6/0</td></tr><tr><td>DeepSeek V4 Flash</td><td>495</td><td>69.1</td><td>69.2</td><td>+0.1</td><td>[-0.9, 0.9]</td><td>98.8</td><td>3/2</td></tr><tr><td rowspan="7">MedQA</td><td>DeepSeek V4 Pro</td><td>500</td><td>95.4</td><td>95.4</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>Qwen3.7 Plus</td><td>500</td><td>96.0</td><td>96.0</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>MiniMax M3</td><td>499</td><td>94.0</td><td>95.4</td><td>+1.4*</td><td>[0.4, 2.4]</td><td>98.6</td><td>7/0</td></tr><tr><td>GPT-OSS 120B</td><td>500</td><td>94.0</td><td>94.0</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>GLM-5.3 Flash</td><td>500</td><td>95.6</td><td>95.6</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>Qwen3.8 Flash</td><td>500</td><td>95.2</td><td>95.4</td><td>+0.2</td><td>[0.0, 0.6]</td><td>99.8</td><td>1/0</td></tr><tr><td>DeepSeek V4 Flash</td><td>499</td><td>95.6</td><td>95.6</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td rowspan="8">MedXpertQA</td><td>DeepSeek V4 Pro</td><td>500</td><td>51.4</td><td>52.0</td><td>+0.6</td><td>[0.0, 1.4]</td><td>99.4</td><td>3/0</td></tr><tr><td>Qwên3.7 Plus</td><td>500</td><td>56.0</td><td>56.0</td><td>+0.0</td><td>[0.0, 0.0]</td><td>99.8</td><td>0/0</td></tr><tr><td>MiniMax M3</td><td>499</td><td>42.1</td><td>44.9</td><td> $+ 2 . 8 ^ { * }$ </td><td>[1.0, 4.6]</td><td>94.2</td><td>17/3</td></tr><tr><td>GPT-OSS 120B</td><td>500</td><td>39.6</td><td>39.6</td><td>+0.0</td><td>[0.0, 0.0]</td><td>99.6</td><td>0/0</td></tr><tr><td>GLM-5.3 Flash</td><td>500</td><td>51.8</td><td>51.6</td><td>-0.2</td><td>[-0.6, 0.0]</td><td>99.6</td><td>0/1</td></tr><tr><td>Qwen3.8 Flash</td><td>496</td><td>51.4</td><td>51.8</td><td>+0.4</td><td>[-0.4, 1.2]</td><td>98.4</td><td>3/1</td></tr><tr><td>DeepSeek V4 Flash</td><td>497</td><td>50.7</td><td>50.7</td><td>+0.0</td><td>[-0.8, 0.8]</td><td>99.2</td><td>2/2</td></tr><tr><td>MMLU-Pro DeepSeek V4 Pro</td><td>500</td><td>86.6</td><td>87.0</td><td>+0.4</td><td>[0.0, 1.0]</td><td>99.6</td><td>2/0</td></tr><tr><td rowspan="7"></td><td>Qwen3.7 Plus</td><td>500</td><td>86.4</td><td>86.4</td><td>+0.0</td><td>[0.0, 0.0]</td><td>100.0</td><td>0/0</td></tr><tr><td>MiniMax M3</td><td>499</td><td>85.4</td><td>86.4</td><td>+1.0*</td><td>[0.2, 2.0]</td><td>98.8</td><td>5/0</td></tr><tr><td>GPT-OSS 120B</td><td>499</td><td>82.6</td><td>82.6</td><td>+0.0</td><td>[-0.6, 0.6]</td><td>99.2</td><td>1/1</td></tr><tr><td>GLM-5.3 Flash</td><td>500</td><td>89.0</td><td>89.0</td><td>+0.0</td><td>[0.0, 0.0]</td><td>99.6</td><td>0/0</td></tr><tr><td>Qwen3.8 Flash</td><td>500</td><td>87.4</td><td>87.8</td><td>+0.4</td><td>[0.0, 1.0]</td><td>99.4</td><td>2/0</td></tr><tr><td>DeepSeek V4 Flash</td><td>495</td><td>86.7</td><td>86.9</td><td>+0.2</td><td>[0.0, 0.6]</td><td>99.8</td><td>1/0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 16: Completed reasoning: the reasoner’s own answer versus Jev reading the final approximately 1,000 reasoning tokens, on 15,309 runs with a valid natural answer. Agree is exact option-letter agreement. The last column counts outcomes where only Jev / only the reasoner is correct (77 / 14 in total). <sup>∗</sup> marks an uncorrected 95% paired-bootstrap interval excluding zero (2,000 resamples).

<table><tr><td colspan="2"></td><td colspan="2">Jev – forced</td><td colspan="2">Jev – Luna dec.</td></tr><tr><td>Benchmark</td><td>Flagged</td><td>All</td><td>Removed</td><td>All</td><td>Removed</td></tr><tr><td>GPQA-D</td><td>4</td><td>9.2</td><td>9.4</td><td>6.5</td><td>7.3</td></tr><tr><td>SuperGPQA</td><td>49</td><td>5.0</td><td>5.7</td><td>4.3</td><td>4.9</td></tr><tr><td>MedQA</td><td>1</td><td>0.9</td><td>0.9</td><td>0.1</td><td>0.1</td></tr><tr><td>MedXpertQA</td><td>66</td><td>2.4</td><td>2.9</td><td>0.6</td><td>0.7</td></tr><tr><td>MMLU-Pro</td><td>20</td><td>3.1</td><td>3.3</td><td>2.9</td><td>3.0</td></tr></table>

Table 17: Accuracy gaps (points) at B = 1.5k with all questions and with flagged questions removed, mean of seven reasoners.

ing forced records and earlier-position fallback.

## P Forced-answer protocol

GPT-OSS 120B and GLM-5.3 Flash use thinkclose; the other five reasoners use strict forced answering. At B = 1.5k, the mean Jev–forced gap is 2.9pp over the 25 strict pairs and 7.2pp over the 10 think-close pairs. Uncorrected paired-bootstrap intervals identify higher accuracy for Jev in 13 strict pairs and eight think-close pairs, with no loweraccuracy pairs. On actual cuts, the respective gaps are 6.9pp and 12.8pp. This protocol grouping is descriptive and confounded with reasoner identity.

<table><tr><td>Tol.</td><td>Cap</td><td>Cap with B=0 Jev→full</td><td></td><td>Jev→cap</td></tr><tr><td>0.25</td><td> $5 7 . 6 / + 0 . 0 5$ </td><td> $5 9 . 1 \ : / + 0 . 0 7 $ </td><td> $3 3 . 0 / - 0 . 2 9 $ </td><td> $7 1 . 4 / - 0 . 7 0$ </td></tr><tr><td>0.5</td><td> $5 9 . 7 / + 0 . 0 3$ </td><td> $6 1 . 4 / + 0 . 0 4$ </td><td> $3 5 . 1 / - 0 . 4 0$ </td><td> $7 3 . 3 / - 0 . 8 3 $ </td></tr><tr><td>1</td><td> $6 3 . 3 / - 0 . 1 3$ </td><td> $6 5 . 2 / - 0 . 1 7$ </td><td> $3 9 . 6 / - 0 . 7 9$ </td><td> $7 6 . 9 / - 1 . 2 1$ </td></tr><tr><td>2</td><td> $6 8 . 7 / - 0 . 8 3$ </td><td> $7 1 . 5 / - 0 . 9 6$ </td><td> $4 8 . 2 / - 1 . 5 0$ </td><td> $8 2 . 2 / - 2 . 1 7$ </td></tr></table>

Table 18: Cost saving (%) / accuracy change (points) against the full run on held-out folds, mean over 35 pairs. Tol.: accuracy loss allowed on the training folds. References: fixed 1.5k with Jev 65.5 / −1.13; Jev alone $9 8 . 3 / - 8 . 9 7 ;$ stop rule auto (tol. 1) 71.3 / −0.58.

<table><tr><td colspan="3">Exact Offset</td><td colspan="2">Jev — forced</td><td colspan="2">Sig. better / worse</td></tr><tr><td>B</td><td>(%)</td><td>(tok.)</td><td>All cut</td><td>Exact only</td><td>All cut</td><td>Exact only</td></tr><tr><td>0.5k</td><td>93.7</td><td>500</td><td>+11.5</td><td>+11.3</td><td>29 / 0</td><td>30 / 0</td></tr><tr><td>1k</td><td>92.7</td><td>668</td><td>+8.8</td><td>+8.0</td><td>25 / 0</td><td>21 /0</td></tr><tr><td>1.5k</td><td>100.0</td><td>一</td><td>+8.6</td><td>+8.6</td><td>21/0</td><td>21 /0</td></tr><tr><td>2k</td><td>88.4</td><td>631</td><td>+8.1</td><td>+7.6</td><td>18/0</td><td>18/0</td></tr><tr><td>4k</td><td>82.1</td><td>615</td><td>+5.3</td><td>+4.5</td><td>10 /0</td><td>8/0</td></tr><tr><td>8k</td><td>64.9</td><td>600</td><td>+3.6</td><td>+2.7</td><td>3/0</td><td>3/1</td></tr><tr><td>16k</td><td>59.4</td><td>826</td><td>+3.1</td><td>+4.2</td><td>4/0</td><td>4/0</td></tr></table>

Table 19: Exact coverage is averaged over pairs. Offset is the mean fallback distance in nominal tokens. Accuracy gaps condition on cut outcomes using the main analysis’s question weights. The exact-only subset requires at least five cut outcomes per pair (32 eligible pairs at 8k; 25 at 16k). Significance uses 100,000 pairedbootstrap resamples, uncorrected. At 1.5k, all 6,500 matched cut outcomes have exact forced answers; no fallback offset applies.

## Q Question-only baselines and trace contribution

Tables 20 and 21 report trace gains and accuracy conditioned on question-only correctness, respectively.

<table><tr><td colspan="3"></td><td colspan="2">Net change (pts)</td><td colspan="2">At 100% (%)</td></tr><tr><td>Reader</td><td>Q-only</td><td>10%</td><td>50%</td><td>100%</td><td>Fixed</td><td>Hurt</td></tr><tr><td>Jev</td><td>66.7</td><td>+2.5</td><td>+7.0</td><td>+9.2</td><td>43.1</td><td>7.7</td></tr><tr><td>Luna dec.</td><td>56.0</td><td>+3.5</td><td>+12.4</td><td>+19.8</td><td>56.1</td><td>8.8</td></tr><tr><td>Luna gen.</td><td>56.5</td><td>+2.8</td><td>+11.3</td><td>+19.1</td><td>55.6</td><td>9.1</td></tr></table>

Table 20: Question-only accuracy (Q-only) and net change at the given trace fraction, pooled over 15,030 model–question outcomes from 2,182 unique questions (unweighted, including SuperGPQA). Fixed: share of incorrect question-only answers corrected; hurt: share of correct question-only answers made incorrect.

This complete-case diagnostic requires parseable answers from all readers at 0, 10, 30, 50, 80 and 100%, excluding 347 of the 15,377 retained outcomes. It retains saved reasoning from novalid-natural-answer runs, including one failedgeneration trace. Unlike the primary evaluation, excluded refusals or parsing failures are not counted as wrong in this diagnostic. Selection and weighting differences explain why its question-only accuracy differs from the budget evaluation. Inputs use the final nominal 1k-token window, and 100% denotes the end of the observed trace. When all three readers start wrong, their overall recovery rates differ by at most about 1.7pp at the listed cuts. The similar point estimates for complete traces do not establish equivalence. When all three start right, Jev keeps the right answer more often after reading an early prefix (+1.3 points over Luna decision at 10%, +4.5 on SuperGPQA, +3.4 on MMLU-Pro), and the difference vanishes on the complete trace. These controls distinguish recovery from preserving a correct initial answer, but do not identify a reader’s internal knowledge or mechanism. Legacy pooled intervals resample model–question outcomes without accounting for shared question IDs across reasoners. We therefore report these rates descriptively and do not use the intervals to claim superiority or equivalence.

<table><tr><td>Subset</td><td>Cut at Jev</td><td></td><td></td><td>Luna dec. Luna gen.</td></tr><tr><td>All question-only answers</td><td>10%</td><td>13.4</td><td>12.8</td><td>14.5</td></tr><tr><td>incorrect (n = 3,455)</td><td>50%</td><td>27.1</td><td>26.7</td><td>25.5</td></tr><tr><td></td><td>100%</td><td>34.2</td><td>34.0</td><td>33.7</td></tr><tr><td>All question-only answers</td><td>10%</td><td>97.8</td><td>96.5</td><td>94.6</td></tr><tr><td>correct (n = 6,597)</td><td>50%</td><td>96.9</td><td>95.7</td><td>95.3</td></tr><tr><td></td><td>100%</td><td>96.3</td><td>96.2</td><td>96.3</td></tr></table>

Table 21: Accuracy (%) after reading a prefix, on outcomes where all three readers are wrong (top) or all right (bottom) from the question alone; pooled over pairs, unweighted. Counts are model–question outcomes, not independent questions.

## R Questions with a usable full-run answer

Of 15,377 retained runs, 67 no-valid-answer rerun outcomes and one saved failed-generation outcome count as incorrect for the full run. They are excluded only from this valid-natural-answer analysis. On the remaining 15,309 runs, the mean full-run accuracy is 77.4%. At $B = 1 . 5 \mathrm { k }$ , Jev reaches 76.0% (−1.4pp versus full), Luna decision 73.1%, Luna generative 72.8%, and forced answering 72.0% (−5.4pp). These are macro-averages over the same 35 pairs.

## S Multiple comparisons and loss bounds

Uncorrected per-pair intervals are exploratory and do not control a single family across all budgets and comparators. For Jev versus forced answering at B = 1.5k, we test each pair’s accuracy difference by paired question bootstrap, stratified and weighted by difficulty on SuperGPQA, using $R = 1 0 0 { , } 0 0 0$ resamples.

The percentile-tail quantity is

$$
\begin{array} { r l } & { p _ { \mathrm { t a i l } } = \operatorname* { m i n } \Bigl \{ \widehat { \mathrm { P r } } ( \hat { \Delta } ^ { * } \leq 0 ) , \widehat { \mathrm { P r } } ( \hat { \Delta } ^ { * } \geq 0 ) \Bigr \} , } \\ & { \quad p = \operatorname* { m a x } \bigl \{ R ^ { - 1 } , \operatorname* { m i n } ( 1 , 2 p _ { \mathrm { t a i l } } ) \bigr \} . } \end{array}
$$

We apply Holm’s step-down correction (Holm, 1979) across the 35 pairs at $\alpha = 0 . 0 5$ . It retains 17 higher-accuracy pairs and none lower; these counts are unchanged across seeds 0, 1 and 2. Percentiletail tests can be poorly calibrated with few discordant outcomes. We therefore also report exact paired tests with the same correction: exact Mc-Nemar tests for the four unweighted benchmarks and an exact weighted answerer-label-swap test for SuperGPQA. The latter assumes within-question answerer-label exchangeability, a stronger null than zero weighted mean difference. This sensitivity check retains 14 higher-accuracy pairs and none lower.

At B = 1.5k, the macro-average Jev–full change is −1.13pp with a one-sided 95% lower bound of −1.74pp. On the valid-natural-answer subset the corresponding values are −1.39pp and −1.98pp. Macro bounds use 20,000 paired question-cluster resamples, stratified by SuperGPQA difficulty and sharing question multiplicities across reasoners. Per-pair bounds use paired question resampling; only 12 of 35 pairs have a lower bound above −1pp. Nonsignificance does not establish equivalence or a universal one-point loss guarantee.

## T Replay overhead of stopping rules

Table 22 reports replayed calls per question, stopping outcomes, and cost savings for each policy.

The auto policy makes 1.5 Jev calls per question on average, 4.0 on GPQA-Diamond and about 0.5 on MMLU-Pro and MedQA. The conf+stable policy is more expensive than the cap in nine pairs, mostly on the medical sets and for DeepSeek V4 Flash (GPQA−Diamond: −8.4 pp), where each check is a larger share of a cheap run. In the observed held-out results, auto is no costlier than the tuned cap in any pair.

<table><tr><td rowspan="2">Policy</td><td rowspan="2">Saving (%)</td><td rowspan="2">Jev calls per question</td><td colspan="3">Questions (%)</td><td rowspan="2">Costlier</td></tr><tr><td>Rule</td><td>Cap</td><td>Finished</td></tr><tr><td>Tuned cap</td><td>63.3</td><td>0.48</td><td>一</td><td>48.2</td><td>51.8</td><td>than cap 一</td></tr><tr><td>conf+stable</td><td>68.8</td><td>1.73</td><td>20.040.9</td><td></td><td>39.1</td><td>9 (9)</td></tr><tr><td>settled+useful</td><td>68.7</td><td>1.92</td><td>28.7 39.0</td><td></td><td>32.3</td><td>4 (1)</td></tr><tr><td>either</td><td>70.8</td><td>1.69</td><td>30.1 38.0</td><td></td><td>31.9</td><td>1 (0)</td></tr><tr><td>auto</td><td>71.3</td><td>1.52</td><td>27.337.9</td><td></td><td>34.7</td><td>0 (0)</td></tr></table>

Table 22: Stop rules on held-out folds, mean over 35 pairs. Rule / Cap / Finished: share of questions stopped by the rule, cut at the cap, or finished within the cap. Costlier than cap: pairs where the rule costs more than the tuned cap (significantly in brackets).