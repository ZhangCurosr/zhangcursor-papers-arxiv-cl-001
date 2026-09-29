# RGDT-BENCH: BENCHMARKING LLM REASONING FOR RULE-GOVERNED DECISIONS AND THEIR JUSTIFICATIONS

Jianpeng Zhao<sup>1,2,†</sup>, Haihua Xu<sup>1</sup>, Haoyang Zhang<sup>1</sup>, Shuang Qian<sup>2</sup>, Yixiang Tang<sup>1</sup> Xintao Wang<sup>1</sup>, Kun Sun<sup>2</sup>, Pei Wu<sup>2</sup>, Shuhan Zhong<sup>2</sup>, Pengyang Wang<sup>1,</sup>

<sup>1</sup>State Key Laboratory of Internet of Things for Smart City and Institute of Smart City Technologies, University of Macau, Macau SAR, China <sup>2</sup>ByteDance, Beijing, China

## ABSTRACT

We study reasoning in Rule-Governed Decision Tasks (RGDTs), where models apply external rules to case facts and justify decisions, as required in policy, contract, and compliance settings. Beyond the deductive capability emphasized by standard mathematical and logical reasoning tasks, RGDTs require interpreting rules and their applicability, assessing conditions from evidence, combining judgments under rules and exceptions, and providing checkable justifications. These demands motivate a new benchmark that assesses existing LLMs’ reasoning abilities by evaluating both their decisions and their stated grounds. We introduce , providing 202.1K condition-level supervision slots across four task tracks and eight supported task–probe combinations that vary access to supporting information. Label-blind extraction and deterministic checks produce labels for warrant completeness: source-referenced coverage and consistency of stated decision grounds. The benchmark attributes failures to four process layers: rule use, condition, evidence, and aggregation, and checks the final outcome. Among evaluable correct responses, warrant incompleteness averages 40.2% across six evaluated LLMs and supported task–probe combinations. Such warrant incompleteness poses potential safety risks and remains difficult to detect: the best of seventeen existing evaluators reaches only 57.69% (random: 50%) task-averaged area under the receiver operating characteristic curve (AUROC). To address this difficulty, we train a simple reward model with warrant supervision. It achieves 69.24% task-averaged AUROC among correct answers, exceeding the matched outcome-supervised baseline by 10.37 pp (percentage points) and the best existing evaluator by 11.55 pp. Beyond completeness assessment, the model outperforms both outcome-supervised baselines across nearly all response-selection comparisons, supporting ’s warrant supervision for RGDT reasoning.

## 1 INTRODUCTION

From benefit applications and contract review to transaction approval and compliance assessment, many routine decisions depend on how written rules apply to a particular case. These are Rule-Governed Decision Tasks (RGDTs): tasks that require both a decision and an explanation of how the governing rules and case facts support it. Compared with mathematical and formal-logical benchmarks centered on deduction, this setting emphasizes interpreting external decision criteria and establishing how they apply to a case. It requires four abilities: (1) Rule interpretation and applicability. Models must identify what the supplied rules require and when they apply. (2) Evidence-grounded condition assessment. Models must link facts to conditions and judge whether this evidence supports, contradicts, or leaves them unresolved. (3) Decision aggregation. Models must combine condition judgments according to the governing decision logic, including applicable exceptions. (4) Checkable justification. Models must explicitly state the required judgments, their supporting rules and evidence, and how they justify the decision. These demands arise because policy and contract clauses must be linked to differently worded or incomplete case facts.

However, final-answer accuracy can conceal failures in these abilities. In Figure 1, both traces correctly reject a claim. One relies on a missed deadline despite a valid waiver, whereas the other recognizes the waiver and bases rejection on the lack of coverage. These omissions pose potential safety risks, motivating checks of how rules, evidence, and exceptions justify decisions. These demands motivate a benchmark that assesses LLMs’ ability to apply rules to cases, beyond general reasoning scores or final-answer accuracy. We call the expressed support linking rules and evidence to a decision a warrant (Toulmin, 2003).

![](images/9099d615d92b4ffbdae0ae0e9618c4497d5fb5b8fbc7f08602fbc868ebda4772.jpg)

We introduce , providing 202.1K condition-level supervision slots across four task tracks: Policy (Sun et al., 2022), Contract (Koreeda & Manning, 2021), Regulation (Saeidi et al., 2018), and Transaction (Zhou et al., 2025). These slots correspond to condition checks within 86,784 model responses

Figure 1: Answer correctness versus warrant completeness in RGDTs. Both traces reject correctly, but only CoT B states the waiver and coverage judgments. Red notes indicate omissions.

(Table 7). Three warrant-access probes provide evidence and rules matched to the assessed conditions, add selected hard negatives, or retain longer natural document context, yielding eight supported task–probe combinations while preserving reference decisions. Label-blind claim extraction and deterministic reference checks evaluate these task requirements in the model’s stated reasoning, producing labels for warrant completeness that we assess through human review. Process-level failure attribution covers four layers: “rule use”, “condition”, “evidence”, and “aggregation”, alongside a final outcome check (Figure 3). Rule-use checks assess the rule support for individual condition judg ments; condition and evidence checks assess the judgments and their evidential support; aggregation checks assess whether all required decision conditions are covered and correctly combined. Together, these checks measure the coverage and consistency of the stated warrant against source-derived references, including applicable exceptions. Completeness concerns the required grounds, not merely whether the reasoning suffices to reach the answer.

Across six evaluated LLMs, 24.8%–76.7% of evaluable correct responses have incomplete warrants, with each model’s results averaged equally across supported task–probe combinations. Among correct but warrant-incomplete responses, 35.9% fail at least two process layers. In distinguishing complete from incomplete warrants among correct answers, the best of seventeen existing evaluators achieves only 57.69% task-averaged AUROC under the evaluation protocol. Correct decisions thus frequently come with omissions or inconsistencies in their stated grounds, including failures spanning multiple process layers. The tested evaluators also struggle to recognize these gaps. We also train a simple reward model (Warrant-RM) on the benchmark labels, achieving 69.24% warrant AUROC among correct answers versus 58.87% for an outcome-supervised scorer with the same base model, inputs, and sample weights. In downstream selection, Warrant-RM leads both outcome-supervised baselines in mean warrant completeness at every candidate count above one and in mean accuracy at most counts across both evaluation views. Together, these results show how RGDT-Bench connects reasoning diagnosis with warrant supervision to improve completeness assessment and response selection, supporting more reliable reasoning in RGDTs.

## 2 RELATED WORK

Rule-based and evidence-grounded reasoning. Rule-governed decisions combine rule application with evidence support. Benchmarks test natural-language rule inference (Clark et al., 2020; Tafjord et al., 2021), logically annotated entailment (Han et al., 2024), and newly specified rules (Ma et al., 2025). Evidence verification tests passage support for claims (Thorne et al., 2018; Wadden et al.,

2020). Policy and contract tasks connect these abilities through conditional answers and evidencelinked entailment (Sun et al., 2022; Koreeda & Manning, 2021). RuleArena evaluates rule-use precision, recall, and application correctness (Zhou et al., 2025), while SingGuard applies policyconditioned rules to multimodal safety moderation (SingGuard Team, 2026). These perspectives motivate ’s checks of required grounds and their combination into a decision.

Warrants and reasoning coverage. Assessing these explanations requires distinguishing stated support from factors that causally influenced the prediction (Jacovi & Goldberg, 2020; Turpin et al., 2023). SemEval-2018 Task 12 examines how implicit warrants connect premises to claims (Habernal et al., 2018). For generated reasoning, Emmons et al. (2025) evaluate answer recovery without additional non-trivial reasoning and detection of synthetic omissions. To check omissions in rule-governed decisions, specifies reference requirements for rule use, conditions, evidence, and aggregation.

Process evaluation and supervision. Process supervision addresses reasoning beyond answer correctness. Studies of correct mathematical solutions reveal errors in their reasoning (Uesato et al., 2022). Human and automated step labels provide more detailed supervision (Lightman et al., 2024; Wang et al., 2024). ProcessBench and PRMBench assess error localization and finegrained step judgments (Zheng et al., 2025; Song et al., 2025), while multi-domain process reward models extend beyond mathematics (Zeng et al., 2025). Hard2Verify further examines whether mathematical steps have supported premises (Pandit et al., 2025), and the Thinking Reward Model (TRM) learns trace preferences from verified-correct solutions represented as graphs (Zhang et al., 2026). complements these approaches with response-level completeness labels to train evaluators to recognize missing decision grounds.

Reward models and downstream selection. Reward supervision must also be evaluated through assessment and selection. RewardBench, RM-Bench, and JudgeBench assess preferences, content/style sensitivity, and correctness (Lambert et al., 2025; Liu et al., 2025; Tan et al., 2025). VerifyBench extends this evaluation to reference-based rewards (Yan et al., 2026). Rubrics specify scoring criteria through prompts or reference-conditioned training (Liu et al., 2023; Kim et al., 2024), and can supervise policies through reasoning criteria or information coverage (Xie et al., 2026; Wang et al., 2026). However, persuasive instruction violations, length, and style can distort judgments (Zeng et al., 2024; Dubois et al., 2024; Feuer et al., 2025; Zheng et al., 2023). To assess stated warrants, we separate model-assisted claim extraction from deterministic reference checks and evaluate scorers without exposing those references. Beyond assessment, response selection can use verifier ranking or self-consistency (Cobbe et al., 2021; Wang et al., 2023), although gains depend on the problem and strategy (Snell et al., 2025). tests whether warrant supervision improves warrant assessment and selection over outcome supervision.

## 3 TASK AND WARRANT EVALUATION

We define task inputs and generated responses, establish reference warrants, and derive completeness checks and the correct-answer warrant gap.

## 3.1 TASK DEFINITION AND RESPONSE GENERATION

Rule-Governed Decision Tasks (RGDTs) require a justified decision given $a = ( q _ { a } , F _ { a } , R _ { a } ) \mathrm { { : } }$ : a question $q _ { a }$ , a fact set $F _ { a }$ , and a rule set $R _ { a }$ expressed in natural language. Rules specify conditions �, propositions about the case to assess, and how their judgments determine the decision. Facts used to assess whether � holds form its evidence set $E _ { a } ( c ) \subseteq \bar { F } _ { a }$ . Our unified, extensible schema makes these rule–condition and condition–evidence links explicit.

To solve instance �, a generator produces a two-part response $\tau = ( h , \hat { y } )$ : the native reasoning trace ℎ (a written chain-of-thought, CoT) and the separately reported terminal decision �ˆ. The response’s stated reasoning is its warrant, whose completeness assesses for diagnosis and supervision. The decision is correct when $\hat { y } ~ = ~ y _ { a } ,$ where $y _ { a }$ is the reference answer used for evaluation and withheld from the generator. Figure 2 introduces three probes that vary support access to diagnose reasoning while preserving the decision problem (Section 4.1).

![](images/ca170eb31c3e2869e9d0b934a2659ccd73fcb55e4e03b30ba14cd54f391ad493.jpg)  
Figure 2: Warrant-access probes in RGDT-Bench. Circles denote rules (�), conditions (�), and evidence (�). Solid links connect rules to their conditions, and dashed links connect evidence to the conditions it supports. A rule may specify several conditions, and a condition may have multiple supporting passages or remain unresolved without support. P2/P3 illustrate evidence-dense and rule-dense contexts, with hard negatives in P2 and longer natural documents in P3. Support markings are illustrative and hidden from generators.

## 3.2 REFERENCE WARRANTS

To evaluate responses against source requirements, we use a reference warrant �, which records the required condition judgments, support, and decision logic. Structured records $( a , \tau , w )$ separate input, response, and reference requirements, keeping reference judgments and verification annotations hidden from the generator. This distinguishes what the response states from what it must justify.

These requirements depend on the task: question-level judgments for Policy, inference conditions for Contract, decision conditions for Regulation, and identification of the proposed transaction plus its overall compliance for Transaction. For each condition judgment, the reference specifies the applicable rules and required evidence. Evidence may comprise jointly required passages or accepted alternatives. Because the sources for Policy and Transaction do not list all evidence passages for each condition, we check whether the response’s evidence supports its condition judgments, without requiring it to identify every supporting evidence passage (Table 10).

## 3.3 WARRANT COMPLETENESS AND CORRECT-ANSWER GAPS

Using these reference requirements, we check the response at four process layers and an outcome layer (Figure 3(b); Appendix A). (1) Rule use checks whether each condition judgment uses the required governing rules: all jointly required passages, or one accepted alternative. (2) Condition checks whether every required judgment is stated and agrees with the reference, including unresolved conditions. (3) Evidence checks whether those judgments have the required support in the supplied materials, so merely citing a passage is insufficient. (4) Aggregation checks whether all referencerequired decision conditions are combined according to the prescribed logic, such as requiring all conditions to hold or accepting alternatives. Thus, rule use concerns the grounds for individual judgments, whereas aggregation concerns their coverage and combination. The derived decision must agree with any decision stated in the trace and with the terminal decision. (5) Outcome checks the terminal decision against the reference answer. For Transaction P2/P3, an additional rule-scope diagnostic compares identified and reference rule sets without affecting completeness labels.

We define warrant completeness as passing all five checks under at least one reference warrant. For response � to instance �, the binary label $W _ { \mathrm { o p } } ( \tau , a )$ is 1 for completeness and 0 for an incomplete warrant. A warrant can be incomplete even when every stated step is correct. Formally,

$$
W _ { \mathrm { o p } } ( \tau , a ) = \bigvee _ { w \in \mathbb { W } ( a ) } \bigwedge _ { \ell \in \mathcal { L } _ { \mathrm { o p } } } R _ { \ell } ( \tau , w ) ,\tag{1}
$$

where $\mathbb { W } ( a )$ contains the reference warrants, $\mathcal { L } _ { \mathrm { o p } }$ the five checks, and $R _ { \ell } ( \tau , w )$ equals 1 if � passes check ℓ under �, and 0 otherwise.

![](images/d7db3d800ae141730d137449a0f89ca091de1605ef38200e34a3688c7f634503.jpg)  
Figure 3: Response collection and diagnosis in RGDT-Bench. (a) A generator samples a native CoT and terminal decision from the RGDT input. (b) Using the outputs of (a), LLM-driven TWC extracts claims from the sampled CoT. WCV checks these claims and the terminal decision against the reference warrant, yielding condition-level diagnostics and a response-level completeness label.

Using these completeness labels, we measure how often correct answers lack required support. We define the correct-answer warrant gap (CWG) as the proportion of evaluable correct responses with incomplete warrants. Missing required claims count as incompleteness, whereas extraction or parsing failures are not evaluable and are excluded. Writing � = 1 for evaluability, the gap is

$$
\mathrm { C W G } = \mathrm { P r } \big ( W _ { \mathrm { o p } } ( \tau , a ) = 0 \mid \hat { y } = y _ { a } , C = 1 \big ) .\tag{2}
$$

Because it is computed within correct responses, this gap is not the difference between overall answer accuracy and warrant completeness.

## 4 BENCHMARK CONSTRUCTION AND VALIDATION

Following the RGDT formulation, RGDT-Bench covers policy applicability, contract inference, administrative-rule application, and basketball-transaction compliance. Three probes vary support presentation to diagnose reasoning with matched support, hard negatives, and natural documents. We then sample and annotate responses, validate the labels, and form balanced splits.

## 4.1 WARRANT-ACCESS PROBE DESIGN

We organize case facts and rules into the three input settings shown in Figure 2: P1: Bound evidence supplies condition-matched facts as evidence and the governing rules, leaving the generator to judge conditions and derive the decision without reference judgments. P2: Hard-negative context retain this support and adds selected passages that resemble relevant evidence or rules but do not determine the conditions, testing whether the generator can distinguish useful support from misleadingly similar material. P3: Document context retains longer excerpts of natural contracts or rule documents, testing whether the generator can locate and apply support amid passages and document structure.

All four tasks support P1. We extend Contract and Transaction to P2/P3 because their contracts and rule manuals provide passages for hard negatives and document context. In contrast, Policy and Regulation organize support around question-specific evidence and rule snippets, so we retain P1 to preserve that context. This yields eight task–probe combinations and 904 prompts. Construction preserves complete source passages, their order, and the reference decision (Appendix B).

For comparison on the same case, P2 adds hard negatives to its P1 materials without changing its support or answer. P3 retains a longer support-containing contract excerpt for Contract and expands P2 with surrounding rule-manual passages for Transaction (Appendix B.3). Because this changes information location, density, and context alongside length (Liu et al., 2024), P3 tests document-context reasoning, where added context may clarify support or distract.

## 4.2 RESPONSE SAMPLING AND AUTOMATIC ANNOTATION

To obtain responses for diagnosis and supervision, we use these probe inputs to prompt six generators from the Qwen, gpt-oss, and DeepSeek families for 16 responses per prompt, yielding 86,784 trace– decision pairs. Automatic annotation separates Claim Extraction from Warrant Check (Figure 3), first extracting stated reasoning and then assessing it against reference warrants.

Claim Extraction. An LLM-driven Trace Warrant Canonicalizer (TWC) structures the response’s stated rule use, condition judgments, evidence, and aggregation without filling gaps or correcting errors. To prevent leakage, it receives task inputs, traces, and condition descriptions but not reference annotations or the separate terminal decision. Explicit paraphrases can identify passages without citation identifiers. The extracted claims then provide structured input for warrant checking.

Warrant Check. Using these extracted claims, a deterministic Warrant Consistency Verifier (WCV) checks their support and the terminal decision against the reference warrant. This extends evidencesupport evaluation (Gao et al., 2023; Li et al., 2024) to Section 3’s five checks: rule use, condition, evidence, aggregation, and outcome. Labels cover 86,435 responses and 202.1K condition judgments, providing fine-grained warrant supervision for evaluator assessment and reward-model training.

## 4.3 LABEL VALIDATION AND BALANCED SPLITS

To validate these labels, we manually review 360 responses across predefined strata, obtaining 84.2% agreement (Krippendorff’s � = 0.687), with 92.7% precision and 78.5% recall for complete warrants (Appendix C.3). Despite residual errors, labels supply decision-ground supervision beyond answer correctness. Sections 5.4 and 5.5 test its value for assessment and response selection.

To use these labels without data leakage, we group related documents, operations, and similar cases, keeping all probe variants and responses within one split. To avoid overrepresenting one family’s response style, we balance annotated responses (Section 4.2) across six generators, with a smaller and a larger model from each of three families. We also balance tasks and warrant labels, with nearly equal probe coverage and attention to response length (Table 8). Natural distributions support prevalence and attribution, while balanced subsets support evaluator comparison and training. Data organization precedes scoring, and reference annotations serve only as supervision and evaluation targets. These choices support accurate diagnosis and high-quality warrant supervision.

## 5 EMPIRICAL EVALUATION

Using RGDT-Bench, we first establish how often correct answers have incomplete warrants and attribute these failures to reasoning layers. We then use existing evaluators, comprising LLM judges and reward models (RMs), to score responses without reference annotations and test whether they distinguish complete from incomplete warrants. Their limitations motivate training RMs with warrant supervision, whose value we assess first in completeness assessment and then in downstream response selection. Together, these results show how RGDT-Bench connects the diagnosis of reasoning failures to supervision that improves both warrant assessment and response selection.

## 5.1 RQ1: HOW OFTEN DO CORRECT ANSWERS HAVE INCOMPLETE WARRANTS?

Table 1: Correct-answer warrant gaps by generator (%). Final/Warr.: answer accuracy/warrant completeness. CWG : gap without the aggregation check. Equal-weight averages over supported task–probe combinations.
<table><tr><td>Size</td><td colspan="4">Final↑ Warr. ↑ CWG↓ CWG&#x27;↓</td></tr><tr><td colspan="6">DeepSeek-R1-Distill-Llama</td></tr><tr><td>8B 70B</td><td>43.9 49.1</td><td>10.7 26.1</td><td>76.7 48.8</td><td>73.6 46.6</td></tr><tr><td colspan="5">gpt-oss</td></tr><tr><td>20B 120B</td><td>74.4 77.7</td><td>47.9 56.3</td><td>36.7 28.9</td><td>32.2 22.1</td></tr><tr><td colspan="5">Qwen3.5</td></tr><tr><td>9B 27B</td><td>83.8 87.7</td><td>63.5 66.6</td><td>25.4 24.8</td><td>22.2 17.5</td></tr><tr><td>Mean</td><td>69.4</td><td>45.2</td><td>40.2</td><td>35.7</td></tr></table>

We first examine whether correct decisions are accompanied by complete warrants. CWG quantifies the proportion of evaluable correct responses with incomplete warrants (Equation 2). Table 1 shows that CWG ranges from 24.8% to 76.7% across six generators and that the larger model in each family has higher warrant completeness. Yet family-level gaps remain substantial (Figure 4a), showing that stronger overall performance does not ensure that correct decisions are justified by complete warrants. To test whether explicit aggregation requirements explain these gaps, we ignore aggregation failures while retaining all other checks. Table 1 reports the relaxed gap as CWG<sup>′</sup>, retaining rule-use, condition, and evidence failures (Appendix E.1). Even under this relaxed check, mean CWG falls only from 40.2% to 35.7%: incomplete warrants remain widespread among correct answers, posing potential safety risks in RGDTs.

## 5.2 RQ2: HOW ARE WARRANT GAPS DISTRIBUTED ACROSS REASONING LAYERS?

To understand the incomplete warrants observed in Section 5.1, we examine four process layers: rule use, condition, evidence, and aggregation. Failures overlap: 35.9% of incomplete warrants fail at least two requirements (Appendix Table 13). To give each response equal weight, we divide its unit contribution among its failed layers. For layers $d \in \mathcal { D }$ , let $F _ { i d } = 1$ if response � fails at $d ,$ and 0 otherwise. Each failed layer receives a share $\phi _ { i d }$ equal to one divided by the number of failed layers, while other layers receive zero:

$$
\phi _ { i d } = \frac { F _ { i d } } { \sum _ { d ^ { \prime } \in \mathcal { D } } F _ { i d ^ { \prime } } } , \qquad \sum _ { d \in \mathcal { D } } \phi _ { i d } = 1 ,\tag{3}
$$

Averaging these shares yields the failure profiles in Figure 4b. Rule use remains the largest contributor under alternative task weights and P1- only evaluation, with aggregation contributing more than evidence (Appendix Table 13). This variation motivates matched-case probe comparisons to examine how support presentation af-

![](images/9094a79c8e3005150f5db2a1e9373d85b44945930f05888224867d90d8079551.jpg)

![](images/ce114a0ea37976fd39a68337c6184af6ec137f41aa32ced1bcfbf8c87b2bc975.jpg)  
Figure 4: (a) Family-equal averages of Final (answer accuracy), Warrant (warrant completeness), and CWG (correct-answer warrant gap). (b) Each correct but incomplete response contributes equally to its failed layers. DS-R1 denotes DeepSeek-R1-Distill-Llama.

fects failures (Appendix E.2.2). Since failures overlap, these profiles locate weaknesses rather than isolate causes. Together, these analyses show how RGDT-Bench supports layered diagnosis of models’ reasoning weaknesses in RGDTs across input conditions, beyond answer accuracy.

## 5.3 RQ3: HOW WELL DO EXISTING EVALUATORS ASSESS WARRANT COMPLETENESS?

Since correct answers often have incomplete warrants, we compare fourteen LLM judges and three pretrained scoring models on distinguishing complete from incomplete warrants: Skywork-Reward-V2-Llama-3.1-8B (Liu et al., 2026), dORM (Lee et al., 2025), and TRM (Zhang et al., 2026). All receive prompts, traces, and decisions without reference warrants.

To separate this ability from answer correctness, we use the full evaluation view (label-balanced data) and the correct-answer-only view (its correct responses). AUROC and average precision (AUPRC) measure discrimination, while balanced accuracy (BalAcc) measures classification. AUPRC comparisons stay within each view because prevalence differs. Table 2 averages each metric $m _ { t p }$ for task � and probe � first over supported probes $\mathcal { P } _ { t }$ , then equally across the four tasks $\mathcal { T } \backslash$

$$
\overline { { m } } _ { \mathrm { t a s k } } = \frac { 1 } { \vert \mathcal { T } \vert } \sum _ { t \in \mathcal { T } } \frac { 1 } { \vert \mathcal { P } _ { t } \vert } \sum _ { p \in \mathcal { P } _ { t } } m _ { t p } .\tag{4}
$$

Table 2 shows that although GPT-5.6 Sol leads the full view with 70.19% AUROC, it reaches only 53.44% (random: 50%) among correct answers, where the best existing evaluator reaches 57.69%. Llama-3.1-70B detects just 10.77% of incomplete warrants despite 98.01% complete-warrant recall (Appendix Table 15). These results show that existing evaluators struggle to assess warrant completeness in correct RGDT answers, motivating explicit warrant supervision.

Table 2: Existing evaluators’ warrant assessment (%). Positive class: complete warrants. Random baselines: AUROC/BalAcc, 50%; AUPRC, 50.00%/63.76% (full/correct-only). Bold/underline: best/second.
<table><tr><td rowspan="2">Evaluator</td><td colspan="3">Full evaluation view</td><td colspan="3">Correct-only view</td></tr><tr><td>AUROC ↑AUPRC ↑ BalAcc ↑ |AUROC ↑AUPRC ↑ BalAcc ↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">Other LLM Judges</td></tr><tr><td>GPT-5.6 Sol</td><td>70.19</td><td>63.84</td><td>67.09</td><td>53.44</td><td>66.14</td><td>53.51</td></tr><tr><td>Gemini-2.5-Pro G</td><td>65.37</td><td>59.10</td><td>65.62</td><td>53.83</td><td>65.61</td><td>53.82</td></tr><tr><td>Z GLM-5.2</td><td>64.07</td><td>58.20</td><td>66.05</td><td>49.00</td><td>63.02</td><td>54.43</td></tr><tr><td colspan="7">Qwen3.5 Series</td></tr><tr><td>Qwen3.5-0.8B</td><td>59.53</td><td>56.14</td><td>58.29</td><td>53.40</td><td>65.95</td><td>52.43</td></tr><tr><td>Qwen3.5-4B</td><td>68.49</td><td>62.13</td><td>63.92</td><td>57.69</td><td>67.92</td><td>53.10</td></tr><tr><td>Qwen3.5-9B</td><td>68.18</td><td>61.57</td><td>64.97</td><td>56.17</td><td>66.74</td><td>54.36</td></tr><tr><td>Qwen3.5-27B</td><td>66.86</td><td>60.73</td><td>64.97</td><td>52.64</td><td>65.08</td><td>53.40</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>66.92</td><td>60.24</td><td>65.31</td><td>54.80</td><td>66.04</td><td>54.01</td></tr><tr><td colspan="7">gpt-oss Series</td></tr><tr><td> gpt-oss-20B</td><td>64.82</td><td>59.25</td><td>62.40</td><td>53.84</td><td>65.55</td><td>53.05</td></tr><tr><td>gpt-oss-120B</td><td>68.83</td><td>63.06</td><td>64.04</td><td>57.47</td><td>68.01</td><td>53.70</td></tr><tr><td colspan="7">DeepSeek-R1 Distillation Family</td></tr><tr><td> DeepSeek-R1-Distill-Llama-8B</td><td>57.83</td><td>55.17</td><td>58.03</td><td>53.42</td><td>65.95</td><td>53.92</td></tr><tr><td> DeepSeek-R1-Distill-Llama-70B</td><td>66.40</td><td>60.23</td><td>63.91</td><td>55.48</td><td>66.55</td><td>54.14</td></tr><tr><td colspan="7">Llama Series</td></tr><tr><td>∞ Llama-3.1-8B-Instruct</td><td>62.13</td><td>57.33</td><td>61.60</td><td>49.94</td><td>63.94</td><td>53.11</td></tr><tr><td>∞ Llama-3.1-70B-Instruct</td><td>67.54</td><td>61.45</td><td>63.89</td><td>55.77</td><td>67.03</td><td>54.39</td></tr><tr><td colspan="7">Pretrained Scoring Models</td></tr><tr><td> Skywork-Reward-V2-Llama-3.1-8B</td><td>63.31</td><td>58.09</td><td>63.24</td><td>51.89</td><td>64.77</td><td>55.84</td></tr><tr><td> dORM-8B</td><td>67.48</td><td>65.56</td><td>60.58</td><td>54.37</td><td>69.36</td><td>50.66</td></tr><tr><td>T TRM-8B</td><td>61.45</td><td>59.42</td><td>56.25</td><td>54.05</td><td>66.80</td><td>51.99</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Effects of trace input and supervision on RM assessment (%). Scores report the mean ± standard deviation across three training runs. Views include all responses or only correct answers. Outcome metrics are N/A in the correct-only view because it contains one outcome class. Bold/underline: best/second.
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=2>Full evaluation view</td><td rowspan=1 colspan=1>Correct-only view</td></tr><tr><td rowspan=1 colspan=2>AUROC↑ AUPRC↑ BalAcc↑</td><td rowspan=1 colspan=1>AUROC↑ AUPRC↑ BalAcc ↑</td></tr><tr><td rowspan=1 colspan=4>Warrant Completeness</td></tr><tr><td rowspan=3 colspan=1>Final-only ORMFull-response ORMWarrant-RM</td><td rowspan=2 colspan=2> $6 7 . 3 1 _ { \pm 2 . 0 6 }$   $6 4 . 3 8 _ { \pm 2 . 5 9 }$   $6 4 . 5 0 _ { \pm 2 . 2 3 }$  $\underline { { 7 2 . 9 9 _ { \pm 2 . 5 6 } } }$   $\underline { { 6 8 . 9 6 _ { \pm 4 . 9 1 } } }$   $6 7 . 4 6 _ { \pm 0 . 4 3 }$ </td><td rowspan=3 colspan=1> $5 1 . 8 6 _ { \pm 3 . 4 0 }$   $6 6 . 9 1 _ { \pm 2 . 6 5 }$   $5 1 . 8 4 _ { \pm 1 . 3 9 }$  $\underline { { 7 1 . 2 0 _ { \pm 5 . 0 3 } } }$   $\underline { { 5 3 . 8 6 } } _ { \pm 1 . 2 1 }$  ${ \bf 6 9 . 2 4 } _ { \pm 2 . 1 5 }$   $\mathbf { 8 0 . 7 9 _ { \pm 1 . 1 1 } }$   ${ \bf 5 8 . 8 6 _ { \pm 1 . 4 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>58.87±4.61. 71.20</td></tr><tr><td rowspan=1 colspan=2> $\mathbf { 7 9 . 9 3 _ { \pm 1 . 8 5 } }$   $7 9 . 1 2 _ { \pm 1 . 5 2 }$   $\mathbf { 7 0 . 5 9 _ { \pm 1 . 8 8 } }$ </td></tr><tr><td rowspan=1 colspan=4> $O u t c o m e$  $C o r r e c t n e s s$ </td></tr><tr><td rowspan=3 colspan=1>Final-only ORMFull-response ORMWarrant-RM</td><td rowspan=3 colspan=2> $8 9 . 5 5 _ { \pm 2 . 3 1 }$   $9 6 . 2 4 _ { \pm 0 . 8 5 }$   $\underline { { 8 3 . 8 7 } } _ { \pm 2 . 6 1 }$  $9 1 . 4 8 _ { \pm 0 . 9 2 }$   $9 6 . 5 4 _ { \pm 0 . 2 3 }$   $\mathbf { 8 4 . 7 5 _ { \pm 1 . 4 6 } }$  ${ \bf 9 1 . 9 1 _ { \pm 0 . 9 1 } }$   $\mathbf { 9 7 . 0 6 _ { \pm 0 . 5 1 } }$   $8 1 . 4 8 _ { \pm 1 . 3 2 }$ </td><td rowspan=2 colspan=1>N/A      N/A      N/AN/A</td></tr><tr><td rowspan=1 colspan=1>AN/AN/A</td></tr><tr><td rowspan=1 colspan=1>N/A      N/A      N/A</td></tr></table>

## 5.4 RQ4: CAN WARRANT SUPERVISION IMPROVE COMPLETENESS ASSESSMENT?

To address these assessment difficulties, we train three Qwen3.5-9B reward models as evaluators under matched settings (Appendix D.3). Final-only ORM receives the prompt and decision, while Full-response ORM also receives the trace. Both learn correctness scores from outcome labels. Warrant-RM uses the same full-response inputs but learns completeness scores from warrant labels.

To compare these models consistently, we use Section 5.3’s views and metrics (Table 3). (1) Trace access. Full-response ORM improves warrant AUROC over Final-only ORM by 5.68 pp in the full view and 7.01 pp among correct answers. (2) Warrant supervision. With identical inputs, Warrant-RM improves AU-ROC over Full-response ORM by 6.94 pp in the full view and 10.37 pp among correct answers. Correct-only gains are positive across tasks and supported task–probe combinations (Figure 5). (3) Comparison with existing evaluators. Warrant-RM gains 11.55 pp in correctonly AUROC over Section 5.3’s best evaluator.

To test whether these AUROC gains also help

![](images/f666fc24c2489f41cf63ead61e7add0402e86314cbb168ad7348e9f7915a8fad.jpg)  
Figure 5: Warrant-supervision AUROC gains (pp). Cells show mean Warrant-RM minus Full-response ORM over three fits in both views. Gray: unsupported.

identify incomplete warrants, we apply score thresholds. Among correct answers, Warrant-RM increases incomplete-warrant detection from 20.69% to 33.56% relative to Full-response ORM, while complete-warrant recall changes from 87.02% to 84.16% (Appendix Table 15). This improves gap detection at the cost of some complete-warrant recall. Together, these results show that warrant supervision improves completeness assessment beyond trace access and outcome supervision, motivating its use for downstream response selection.

## 5.5 RQ5: DO LEARNED WARRANT SCORES IMPROVE RESPONSE SELECTION?

To test whether better assessment improves response selection, we compare two views: (1) All response groups with all sixteen candidates and resolved labels; (2) Target-specific opportunity groups containing both positive and negative labels for the evaluated target. Selectors share groups within each view and choose the highest-scoring response without reference labels (Appendix E.5). Full results for $N = 1 { - } 1 6$ appear in Appendix Table 20.

Across both views, Warrant-RM achieves the highest mean completeness at every � > 1 and the highest mean accuracy at most candidate counts (Table 20). On all response groups, completeness and accuracy improve by 4.40 and 1.69 pp over Final-only ORM, and by 1.23 and 0.46 pp over Full-response ORM (� = 16; rounded means in Table 4). Within target-specific opportunity groups, Warrant-RM exceeds Fullresponse ORM by 2.18 pp in completeness and

Table 4: Warrant-supervision effects on response selection (%). Means over three fits. Final/Full: Finalonly/Full-response ORM; WRM: Warrant-RM. Bold/underline: best/second.
<table><tr><td rowspan=2 colspan=1>N</td><td rowspan=2 colspan=1>Warrant ↑Final Full WRM</td><td rowspan=1 colspan=1>Accuracy ↑</td></tr><tr><td rowspan=1 colspan=1>Final Full WRM</td></tr><tr><td rowspan=1 colspan=3>All response groups</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>51.01 51.01 51.01</td><td rowspan=1 colspan=1>72.5872.5872.58</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>54.42 55.90 56.92</td><td rowspan=1 colspan=1>78.21 78.39 78.30</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>55.34 57.77 58.94</td><td rowspan=3 colspan=1>79.82 80.41 80.5580.49 81.43 81.7180.81 82.0482.50</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>55.85 58.79 59.94</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>56.3059.47 60.70</td></tr><tr><td rowspan=1 colspan=3>Target-specific opportunity groups</td></tr><tr><td rowspan=1 colspan=1>1481216</td><td rowspan=1 colspan=1>53.50 53.50 53.5058.33 61.81 63.7359.46 65.42 67.5760.04 67.66 69.7060.50 69.34 71.52</td><td rowspan=1 colspan=1>58.66 58.6658.6668.55 70.50 70.7970.81 74.36 75.3171.27 76.33 77.5871.12 77.57 79.23</td></tr></table>

1.66 pp in accuracy at $N = 1 6 ,$ with a 1.99 pp mean completeness gain across � = 2–16. Mean gains persist with random tie-breaking (Appendix E.5). Together, these results show that even a simple warrant-supervised reward model improves completeness assessment and response selection.

## 6 CONCLUSION

We formalize Rule-Governed Decision Tasks (RGDTs), which require models to apply external rules to case facts and justify their decisions. Correct answers can conceal incomplete warrants, leaving potential safety risks unchecked. combines a unified, extensible schema with four task families, controlled probes, and balanced evaluation. Its layered checks support joint diagnosis of reasoning failures, which existing evaluators struggle to recognize among correct answers. The resulting warrant supervision improves reward-model assessment and downstream response selection over outcome-supervised baselines. thus connects failure diagnosis with learning signals for more reliable RGDT reasoning, enabling learning from such reasoning failures.

## 7 LIMITATIONS AND FUTURE WORK

Our evaluation has three limitations. First, decisions use supplied inputs, leaving multi-turn interaction and tool use untested. Future work can track how warrants evolve as evidence accumulates. Second, text-only inputs exclude visual evidence and document layout, motivating multimodal extensions that link these sources to condition judgments. Third, we study sequence-level reward modeling rather than step-level process reward models. Evaluating the latter requires consistent reasoning-step segmentation and alignment with condition-level supervision to compare process rewards in RGDTs.

## AI USE STATEMENT

We used generative AI tools to assist with data synthesis, including candidate reasoning-trace generation, and with extracting stated claims during automatic annotation. Annotation quality was assessed through human review and agreement analysis (Appendix C.3). We also used generative AI for language polishing and typesetting assistance. The authors take responsibility for the final content of this work, including text, claims, and artifacts produced with AI assistance.

## REFERENCES

Peter Clark, Oyvind Tafjord, and Kyle Richardson. Transformers as soft reasoners over language. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, pp. 3882–3890, 2020. doi: 10.24963/ijcai.2020/537. URL https://www.ijcai.org/ proceedings/2020/537.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

Yann Dubois, Percy Liang, and Tatsunori Hashimoto. Length-controlled AlpacaEval: A simple debiasing of automatic evaluators. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=CybBmzWBX0.

Scott Emmons, Roland S. Zimmermann, David K. Elson, and Rohin Shah. A pragmatic way to measure chain-of-thought monitorability. arXiv preprint arXiv:2510.23966, 2025. URL https: //arxiv.org/abs/2510.23966.

Benjamin Feuer, Micah Goldblum, Teresa Datta, Sanjana Nambiar, Raz Besaleli, Samuel Dooley, Max Cembalest, and John P. Dickerson. Style outweighs substance: Failure modes of LLM judges in alignment benchmarking. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=MzHNftnAM1. ICLR 2025 Poster.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 6465–6488. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.398. URL https://aclanthology.org/2023. emnlp-main.398/.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1321–1330. PMLR, 2017. URL https: //proceedings.mlr.press/v70/guo17a.html.

Ivan Habernal, Henning Wachsmuth, Iryna Gurevych, and Benno Stein. SemEval-2018 task 12: The argument reasoning comprehension task. In Proceedings of the 12th International Workshop on Semantic Evaluation, pp. 763–772. Association for Computational Linguistics, 2018. doi: 10.18653/v1/S18-1121. URL https://aclanthology.org/S18-1121/.

Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James Coady, David Peng, Yujie Qiao, Luke Benson, Lucy Sun, Alexander Wardle-Solano, Hannah Szabo, Ekaterina Zubova, Matthew Burtell, Jonathan Fan, Yixin Liu, Brian Wong, Malcolm Sailor,´ Ansong Ni, Linyong Nan, Jungo Kasai, Tao Yu, Rui Zhang, Alexander Fabbri, Wojciech Maciej Kryscinski, Semih Yavuz, Ye Liu, Xi Victoria Lin, Shafiq Joty, Yingbo Zhou, Caiming Xiong, Rex Ying, Arman Cohan, and Dragomir Radev. FOLIO: Natural language reasoning with firstorder logic. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 22017–22031, 2024. doi: 10.18653/v1/2024.emnlp-main.1229. URL https: //aclanthology.org/2024.emnlp-main.1229/.

Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness? In Proceedings ofthe 58th Annual Meeting ofthe Associationfor

Computational Linguistics, pp. 4198–4205. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.386. URL https://aclanthology.org/2020.acl-main. 386/.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing finegrained evaluation capability in language models. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=8euJaTveKw.

Yuta Koreeda and Christopher Manning. ContractNLI: A dataset for document-level natural language inference for contracts. In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 1907–1919, Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.164. URL https://aclanthology. org/2021.findings-emnlp.164/.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. RewardBench: Evaluating reward models for language modeling. In Findings ofthe Association for Computational Linguistics: NAACL 2025, pp. 1755–1797, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-naacl.96. URL https://aclanthology.org/2025.findings-naacl.96/.

Dong Bok Lee, Seanie Lee, Sangwoo Park, Minki Kang, Jinheon Baek, Dongki Kim, Dominik Wagner, Jiongdao Jin, Heejun Lee, Tobias Bocklet, Jinyu Wang, Jingjing Fu, Sung Ju Hwang, Jiang Bian, and Lei Song. Rethinking reward models for multi-domain test-time scaling. arXiv preprint arXiv:2510.00492, 2025. doi: 10.48550/arXiv.2510.00492. URL https://arxiv. org/abs/2510.00492. Version 3, revised July 2026.

Yifei Li, Xiang Yue, Zeyi Liao, and Huan Sun. AttributionBench: How hard is automatic attribution evaluation? In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 14919– 14935. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.886. URL https://aclanthology.org/2024.findings-acl.886/.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=v8L0pN6EOi. ICLR 2024 Poster; first released as arXiv:2305.20050 in 2023.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl a 00638. URL https://aclanthology.org/2024.tacl-1.9/.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2511–2522, 2023. doi: 10.18653/v1/ 2023.emnlp-main.153. URL https://aclanthology.org/2023.emnlp-main.153/.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. RM-Bench: Benchmarking reward models of language models with subtlety and style. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 6da1eec80095dc5937f7716db15aca4b-Abstract-Conference.html.

Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, and Yang Liu. Skywork-Reward-V2: Scaling preference data curation via human–AI synergy. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ d8836d0294651a867c2ca51012923997-Abstract-Conference.html.

Kaijing Ma, Xinrun Du, Yunran Wang, Haoran Zhang, Zhoufutu Wen, Xingwei Qu, Jian Yang, Jiaheng Liu, Minghao Liu, Xiang Yue, Wenhao Huang, and Ge Zhang. KOR-Bench: Benchmarking language models on knowledge-orthogonal reasoning tasks. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ c6f5851a0d9cb435ed8b50e87bd6a257-Abstract-Conference.html.

Shrey Pandit, Austin Xu, Xuan-Phi Nguyen, Yifei Ming, Caiming Xiong, and Shafiq Joty. Hard2Verify: A step-level verification benchmark for open-ended frontier math. arXiv preprint arXiv:2510.13744, 2025. URL https://arxiv.org/abs/2510.13744.

Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rocktaschel, Mike Sheldon, Guil-¨ laume Bouchard, and Sebastian Riedel. Interpretation of natural language rules in conversational machine reading. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Lan-

guage Processing, pp. 2087–2097, Brussels, Belgium, 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1233. URL https://aclanthology.org/D18-1233/.

Takaya Saito and Marc Rehmsmeier. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3):e0118432, 2015. doi: 10.1371/journal.pone.0118432. URL https://journals.plos.org/plosone/ article?id=10.1371/journal.pone.0118432.

SingGuard Team. SingGuard: A policy-adaptive multimodal LLM guardrail with dynamic reasoning. arXiv preprint arXiv:2606.22873, 2026. URL https://arxiv.org/abs/2606.22873.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=4FWAwZtd2n.

Mingyang Song, Zhaochen Su, Xiaoye Qu, Jiawei Zhou, and Yu Cheng. PRMBench: A fine-grained and challenging benchmark for process-level reward models. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 25299– 25346, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025. acl-long.1230. URL https://aclanthology.org/2025.acl-long.1230/.

Haitian Sun, William Cohen, and Ruslan Salakhutdinov. ConditionalQA: A complex reading comprehension dataset with conditional answers. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 3627–3637, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.253. URL https://aclanthology.org/2022.acl-long.253/.

Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 3621–3634, 2021. doi: 10.18653/v1/2021.findings-acl.317. URL https://aclanthology.org/2021.findings-acl.317/.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. JudgeBench: A benchmark for evaluating LLM-based judges. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: A largescale dataset for fact extraction and verification. In Proceedings of NAACL-HLT, pp. 809–819, 2018. doi: 10.18653/v1/N18-1074. URL https://aclanthology.org/N18-1074/.

Stephen E. Toulmin. The Uses of Argument. Cambridge University Press, 2003. ISBN 9780511840005. doi: 10.1017/CBO9780511840005. URL https://doi.org/10.1017/ CBO9780511840005.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems, volume 36, pp. 74952–74965, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022. URL https://arxiv. org/abs/2211.14275.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 7534–7550, 2020. doi: 10.18653/v1/2020.emnlp-main.609. URL https://aclanthology.org/2020. emnlp-main.609/.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.510. URL https://aclanthology.org/ 2024.acl-long.510/.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=1PL1NIMMrw.

Yudong Wang, Zhe Yang, Wenhan Ma, Rang Li, Qibin Yang, Weimin Xiong, Jiangshan Duo, Liang Zhao, and Zhifang Sui. From refuse to richness: Rubric rewards for long-form hallucination reinforcement learning. arXiv preprint arXiv:2608.12337, 2026. URL https://arxiv.org/ abs/2608.12337.

Weichu Xie, Haozhe Zhao, Wenpu Liu, Yongfu Zhu, Liang Chen, Minghao Ye, Zirong Chen, Yuqi Xu, Shuai Dong, Ziyue Wang, Xinbo Xu, Kean Shi, Ruoyu Wu, Xiaoying Zhang, Wenqi Shao, Baobao Chang, Nan Duan, and Jiaqi Wang. Step-wise rubric rewards for LLM reasoning. arXiv preprint arXiv:2605.17291, 2026. URL https://arxiv.org/abs/2605.17291.

Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, Yang Liu, Xin Xu, Mengdi Zhang, Jian Shao, Yongliang Shen, Jun Xiao, and Yueting Zhuang. VerifyBench: Benchmarking referencebased reward systems for large language models. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=JfsjGmuFxz.

Thomas Zeng, Shuibai Zhang, Shutong Wu, Christian Classen, Daewon Chae, Ethan Ewer, Minjae Lee, Heeju Kim, Wonjun Kang, Jackson Kunde, Ying Fan, Jungtaek Kim, Hyung Il Koo, Kannan Ramchandran, Dimitris Papailiopoulos, and Kangwook Lee. VersaPRM: Multi-domain process reward model via synthetic reasoning data. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 74197–74239. PMLR, 2025. URL https://proceedings.mlr.press/v267/zeng25h.html.

Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, and Danqi Chen. Evaluating large language models at evaluating instruction following. In International Conference on Learning Rep resentations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/ 2024/hash/afc8b034823271816d14f7c1aefe1dff-Abstract-Conference. html.

Haoran Zhang, Yafu Li, Zhi Wang, Zhilin Wang, Shunkai Zhang, Xiaoye Qu, and Yu Cheng. Characterizing, evaluating, and optimizing complex reasoning. In Proceedings ofthe Forty-Third International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/ 2602.08498.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. ProcessBench: Identifying process errors in mathematical reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1009–1024, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.50. URL https: //aclanthology.org/2025.acl-long.50/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, 2023. doi: 10.52202/075280-2020. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets\_and\_ Ben chma rk s. btml. Datasets and Benchmarks Track

Ruiwen Zhou, Wenyue Hua, Liangming Pan, Sitao Cheng, Xiaobao Wu, En Yu, and William Yang Wang. RuleArena: A benchmark for rule-guided reasoning with LLMs in real-world scenarios. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 550–572, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.27. URL https://aclanthology.org/2025. acl-long.27/.

## APPENDIX GUIDE

The appendix follows the paper’s logic from reference requirements and probe construction to annotation, evaluation, and supplementary evidence for warrant supervision. Prompt examples connect these stages through one case.

A Reference Warrants and Diagnostic Checks p. 15   
B Task Coverage, Data Balance, and Warrant-Access Probes p. 16   
B.1 Task Coverage and Leakage Prevention 16   
B.2 Natural and Balanced Response Collections 16   
B.3 Construction of the Three Warrant-Access Probes 17   
C Automatic Annotation and Human Validation p. 18   
C.1 Information Boundaries 19   
C.2 Claim Extraction and Warrant Verification 19   
C.3 Human Review of Annotation Quality 20   
D Evaluation Measures and Matched Comparisons p. 21   
D.1 Metrics, Weighting, and Uncertainty 21   
D.2 Existing Evaluators 23   
D.3 Matched Reward-Model Training 24   
E Supplementary Evidence for Diagnosis and Supervision p. 24   
E.1 Robustness of Correct-Answer Warrant Gaps 25   
E.2 Layered Attribution and Probe Comparisons 25   
E.2.1 Layered Failure Attribution 25   
E.2.2 Matched Probe Comparisons 26   
E.3 Probability Quality and Incomplete-Warrant Detection 26   
E.4 Effects of Trace Access and Warrant Supervision 27   
E.4.1 Warrant Assessment Among Correct Answers 27   
E.4.2 Effect of Training Weights 29   
E.5 Response Selection and Robustness Checks 29   
F Prompt Examples for Generation, Annotation, and Evaluation p. 31

## A REFERENCE WARRANTS AND DIAGNOSTIC CHECKS

To assess a response’s warrant, the reference warrant specifies the required judgments, their support, and their roles in the decision. This appendix expands Section 3 by specifying support requirements, distinguishing rule use from aggregation, and explaining how unresolved checks are handled.

Task inputs and hidden references. The input schema separates the natural-language question, case facts, and rules. Evidence records link facts to the conditions they help assess, while reference judgments remain hidden from the generator. Reference warrants specify the required judgments, support, and decision logic. Completeness requires covering these grounds consistently, even when fewer stated grounds suffice for the decision.

Checking the support for each judgment. Rule-use and evidence checks ask whether each condition judgment has the required support. Rule use checks the governing rules, while evidence checks the supporting case information. For rules and evidence separately, the reference requires all specified passages or an accepted alternative. Evidence cannot replace required rule use, or vice versa. For example, if two evidence passages are jointly required, using only one is insufficient. If either passage is accepted as an alternative, using one meets this coverage requirement.

Some sources do not specify supporting passages for each judgment. In these cases, we do not require a particular preassigned passage, but still assess the support stated in the response against the task-specific requirements in Table 10. These checks concern the grounds for individual judgments. How those judgments combine into a decision is checked separately under aggregation.

From condition judgments to warrant completeness. Table 5 groups the checks used by WCV. Rule use concerns the grounds for each judgment, whereas aggregation concerns whether all reference required decision conditions are covered and combined correctly. The decision obtained by combining the stated condition judgments must agree with any decision stated in the trace and with the separately reported terminal decision. All five checks must pass against the same reference warrant. For Transaction, we separately check that the response identifies the transaction action posed in the question and that its compliance judgment has the required grounds (Table 10). Compliance requires an accepted governing rule and meaningful use of the supplied evidence or rules.

Table 5: Warrant checks and separate diagnostics. The five completeness checks must pass. rule scope is a separate diagnostic. Failures shown are illustrative and may co-occur. � = 1 denotes evaluability in Equation 2.
<table><tr><td>Layer</td><td>Required property</td><td>Representative failure</td></tr><tr><td>Rule scope</td><td>applied rules match a reference set where the comparison is determined, with unidentified applied rules marked NA</td><td>rule outside the reference set</td></tr><tr><td>Rule use</td><td>each condition judgment has its required rule support, using all required rule passages or an accepted alternative</td><td>missing required rule use</td></tr><tr><td>Condition</td><td>every required condition receives the reference judgment</td><td>omitted exception condition evidence does not support</td></tr><tr><td>Evidence</td><td>semantic support satisfies the source-specific requirement independently of literal citation format</td><td>the judgment</td></tr><tr><td>Aggregation</td><td>all required decision conditions enter aggregation, with their judgments combined according to the decision logic and consistent with the terminal decision</td><td>omitted aggregation input or inconsistent decision</td></tr><tr><td>Outcome</td><td>predicted label equals the final answer implied by the same reference warrant</td><td>final answer wrong</td></tr><tr><td>Evaluability C</td><td>stated claims can be extracted into the required structure, failed extraction or although missing claims still fail</td><td>non-parseable trace</td></tr></table>

Separate diagnostics and evaluability. Rule scope separately compares identified and reference rule sets for Transaction P2/P3 without affecting warrant completeness. When the applied rules cannot be identified, rule scope is marked NA (not available). Missing or incorrect claims fail the corresponding checks, whereas only extraction or parsing failures yield NOT-EVALUABLE. Several checks may fail on the same response. The results identify missing or inconsistent grounds without claiming that one failure caused another.

## B TASK COVERAGE, DATA BALANCE, AND WARRANT-ACCESS PROBES

RGDT-Bench covers four task tracks through a shared representation of decisions and their required grounds. To support both diagnosis and learning, its construction preserves task facts and rules, prevents related cases from crossing data splits, and varies access to support through three probes.

## B.1 TASK COVERAGE AND LEAKAGE PREVENTION

Cases sharing source documents, involving related operations, or having similar content are placed in the same source group. To prevent data leakage, each source group stays within one data split before probe construction or response generation. Table 6 reports source-case counts by split, with 90 test cases forming 76 such source groups.

Table 6: Source-case counts by task and split. Each case yields one prompt per supported probe (checkmarks), with all probes sharing the case split. Native labels are balanced within each supported task–split–probe combination.
<table><tr><td>Task track</td><td>Train</td><td>Valid.</td><td>Test</td><td>All</td><td>P1</td><td>P2</td><td>P3</td></tr><tr><td>Policy</td><td>56</td><td>20</td><td>18</td><td>94</td><td>√</td><td>一</td><td>一</td></tr><tr><td>Contract</td><td>66</td><td>21</td><td>21</td><td>108</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Regulation</td><td>78</td><td>27</td><td>27</td><td>132</td><td>√</td><td>一</td><td>一</td></tr><tr><td>Transaction</td><td>70</td><td>24</td><td>24</td><td>118</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Source cases</td><td>270</td><td>92</td><td>90</td><td>452</td><td></td><td>904 active prompts</td><td></td></tr></table>

To distinguish the number of responses from the condition judgments they contain, Table 7 reports generator responses and condition judgments separately. K and M denote thousands and millions, respectively. Training contains 51,832 evaluable responses, including 23,809 complete and 28,023 incomplete warrants. The 200 training responses with not-evaluable warrant labels remain outside supervision. Uncertainty estimation also uses these source groups as the resampling units, keeping all responses from cases in the same group together. This avoids treating responses from the same or related cases as independent observations.

Table 7: Data and supervision scale by probe. Tokens include the prompt, reasoning trace, and formatting text (Qwen3.5-9B tokenizer). K/M: thousands/millions. Totals are rounded after summation.
<table><tr><td>Statistic</td><td>Train</td><td>Validation</td><td>Test</td><td>All splits</td></tr><tr><td colspan="5">Response Samples</td></tr><tr><td>Probe 1</td><td>25,920</td><td>8,832</td><td>8,640</td><td>43,392</td></tr><tr><td>Probe 2</td><td>13,056</td><td>4,320</td><td>4,320</td><td>21,696</td></tr><tr><td>Probe 3</td><td>13,056</td><td>4,320</td><td>4,320</td><td>21,696</td></tr><tr><td>Total</td><td>52,032</td><td>17,472</td><td>17,280</td><td>86,784</td></tr><tr><td colspan="5"></td></tr><tr><td>Probe 1</td><td>53.0</td><td>Condition Slots (K) 17.4</td><td>18.4</td><td>88.8</td></tr><tr><td>Probe 2</td><td>33.7</td><td>11.3</td><td>11.7</td><td>56.7</td></tr><tr><td>Probe 3 Total</td><td>33.6</td><td>11.3</td><td>11.7</td><td>56.6</td></tr><tr><td></td><td>120.2</td><td>39.9</td><td>41.9</td><td>202.1</td></tr><tr><td colspan="5">Full-Response Sequence Tokens (M)</td></tr><tr><td>Probe 1</td><td></td><td>22.1</td><td>21.9</td><td></td></tr><tr><td>Probe 2</td><td>65.3 59.6</td><td>20.3</td><td>20.3</td><td>109.3 100.2</td></tr><tr><td>Probe 3</td><td>89.9</td><td>29.1</td><td>29.6</td><td>148.6</td></tr><tr><td>Total</td><td>214.8</td><td>71.5</td><td>71.7</td><td>358.1</td></tr></table>

## B.2 NATURAL AND BALANCED RESPONSE COLLECTIONS

Table 8 distinguishes balanced prompts, which control task–probe coverage, from the responses used for analysis and learning. The natural response collection contains all generated responses before balancing and supports prevalence and attribution analyses. From its responses with evaluable warrant labels, we form balanced subsets for evaluator comparison and training, balancing generators, tasks, and warrant labels (Table 8). Within each task, response counts across supported probes differ by at most one. We also account for response length and coverage across source cases. All subsets retain distinct generator responses within the existing source-group splits.

![](images/b035e0298cc88f2facd810dad0708939d24b0c7fe715c4ceaf4f355e911146e3.jpg)  
(a) Split ECDF

![](images/527cdf093d94df9e6be312affce3bd664db38a89aa847ec9b45ed352bc48b236.jpg)  
Full-response length (Qwen3.5 tokens)  
(b) Warrant-access probe distributions  
Figure 6: Prompt–trace lengths in the balanced data. (a) Cumulative distributions by split. (b) Density by probe, with dark marks denoting medians. P1 covers four tasks, whereas P2/P3 cover Contract and Transaction.

Table 8: Prompt and response counts by split. The first row counts prompts, and the remaining rows count responses. Sections 5.3–5.4 share the balanced evaluation responses.
<table><tr><td>Collection</td><td>Train</td><td>Validation</td><td>Test</td><td>Intended use</td></tr><tr><td colspan="5">Prompts</td></tr><tr><td>Balanced prompts</td><td>432</td><td>144</td><td>144</td><td>Balanced task-probe coverage, labels, and input lengths</td></tr><tr><td colspan="5">Responses</td></tr><tr><td>Natural responses</td><td>52,032</td><td>17,472</td><td>17,280 balancing</td><td>Generated responses before</td></tr><tr><td>Evaluable responses</td><td>51,832</td><td>17,400</td><td>17,203</td><td>Responses with evaluable warrant labels</td></tr><tr><td>Balanced response subset</td><td>11,856</td><td>3,720</td><td>3,564</td><td>Evaluable subset for training and evaluator comparison</td></tr></table>

Figure 6 shows prompt–trace lengths in the balanced data using the Qwen3.5-9B tokenizer. All sequences are retained without truncation. Differences across probes reflect both their context construction and supported tasks.

Response balancing precedes evaluator scoring. Sections 5.3–5.4 share evaluation responses. The reward models used in Sections 5.4–5.5 are trained on the balanced response training split and share training and validation weights. Test responses never enter fitting, calibration, or threshold selection, and reference annotations provide targets rather than model inputs. These controls make the labels useful both for comparable diagnosis and for warrant supervision.

## B.3 CONSTRUCTION OF THE THREE WARRANT-ACCESS PROBES

To diagnose support access while keeping the decision problem unchanged, P1 binds support to conditions, P2 adds confusable passages, and P3 restores longer natural document context. Contract and Transaction support P2/P3 because their contracts and rule manuals contain both related distractors and surrounding text. Policy and Regulation instead provide source-linked evidence for P1. The probes therefore test different support-access demands, rather than impose a predetermined difficulty order.

P1: Bound evidence. P1 supplies condition-matched facts as evidence, together with governing rules, leaving judgments to the generator. A condition can need several passages, and a passage can inform several conditions. Contract retains annotated support in document order, while Transaction combines case facts with relevant manual passages. Citation labels reveal neither reference roles, judgments, nor answers.

P2: Hard-negative context. P2 preserves P1 support and adds passages that resemble relevant material without changing the reference decision. For Contract, candidates are complete evidence sets for other hypotheses in the same contract. Review keeps passages that are self-contained and easy to confuse with target evidence through shared actors, actions, exceptions, or scope, but do not support, contradict, or clarify the assessed conditions. Algorithm 1 adds the eligible evidence sets intact and preserves their source order.

Algorithm 1 Contract hard-negative construction   
Require: contract, assessed conditions, P1 support �, context and selection limits   
Ensure: P2 context preserving � and the decision, or rejection   
1: � ← complete evidence sets for other hypotheses in the contract   
2: Remove duplicates, incomplete sets, and sets overlapping � from �   
3: � ← reviewed pairs (�, �): set � is self-contained and confusable with evidence for �   
through shared actors, actions, exceptions, or scope,   
but does not support, contradict, or clarify any assessed condition   
4: Sort � by condition, then fixed source order; � ← ∅, � ← ∅   
5: for each ( �, �) ∈ � do   
6: if � ∉ �, � does not overlap � ∪ �, and the complete addition fits the limits then   
7: � ← � ∪ {�}; � ← � ∪ { � }   
8: end if   
9: if the selection limit is reached then   
10: break   
11: end if   
12: end for   
13: � ← render � ∪ � in source order   
14: if � = ∅ or source, split, support, or decision checks fail then   
15: return rejection   
16: end if   
17: return �

For Transaction, candidates are rules from the same operation family, kept with the passages needed to interpret them. Review excludes rules that overlap P1 support or determine the case outcome, retaining rules that are inapplicable, already satisfied, or otherwise nondiagnostic. Among these eligible rules, construction favors additions that keep context lengths comparable and uses a fixed order to resolve ties (Algorithm 2). Thus, review establishes the distractors’ suitability, while length controls only their inclusion.

P3: Document context. P3 preserves longer natural document context instead of adding only selected distractors. Contract uses a source-ordered excerpt containing the relevant support. Transaction expands P2 with surrounding manual sections and referenced rules, prioritizing context connected to the decision (Algorithm 3). Both retain complete passages and document order within the context budget. This preserves natural context for testing how models locate and combine support in longer documents.

Keeping the correct decision unchanged. To attribute differences across probes to how models locate and use support, added context must not change the facts, required support, or correct decision. We therefore check these properties after constructing P2/P3 and reject inputs that fail, while keeping source groups in their original data splits. References guide this check but remain hidden from generators and evaluators. P3 also marks omitted passages and unavailable cross-references so readers can distinguish an excerpt from a complete document.

## C AUTOMATIC ANNOTATION AND HUMAN VALIDATION

Reliable warrant supervision requires extracting what a response states without supplying missing reasoning. The annotation pipeline therefore separates label-blind claim extraction from referencebased verification, then checks the resulting labels through human review.

Algorithm 2 Transaction hard-negative construction   
Require: transaction, P1 support �, same-family rules, context and selection limits   
Ensure: P2 context containing complete nondiagnostic rules, or rejection   
1: � ← rules from the same operation family, with all passages needed to interpret them   
2: Remove incomplete units, overlap with �, decision-determining rules, and excluded rules   
3: Retain in � only rules reviewed as inapplicable, satisfied, or otherwise nondiagnostic   
4: � ← ∅; � ← render �   
5: while � ≠ ∅ and the selection limit is not reached do   
6: if enough rules are added and � reaches the desired length range then   
7: break   
8: end if   
9: � ← rules in � that do not overlap � and fit the context budget intact   
10: if � = ∅ then   
11: break   
12: end if   
13: � ← rule in � whose addition is closest to the target length; use fixed order for ties   
14: if enough rules are added, the minimum desired length is reached, and � brings no improvement then   
15: break   
16: end if   
17: � ← � ∪ {�}; � ← � \ {�}; � ← render � ∪ � in source order   
18: end while   
19: if selection requirements or source, support, or decision checks fail then   
20: return rejection   
21: end if   
22: return �

Algorithm 3 Transaction document-context construction   
Require: P2 passages �, source manual, decision rules, rule references, context budget   
Ensure: P3 document context preserving � and the decision, or rejection   
1: � ← source blocks containing �   
2: Append blocks cited by decision rules, then blocks cited by other P2 rules   
3: Append nearby cited blocks; order � by this priority and source position   
4: � ← ∅   
5: for each block � ∈ � do   
6: if � is allowed, adds new material, and fits the budget as a complete block then   
7: � ← � ∪ {�}   
8: end if   
9: end for   
10: � ← render � ∪ � in source order, preserving rule and section boundaries   
11: Mark omitted passages and unavailable cross-references in �   
12: if � = ∅ or source, support, or decision checks fail then   
13: return rejection   
14: end if   
15: return �

## C.1 INFORMATION BOUNDARIES

To prevent reference information from filling reasoning gaps, each stage receives only the information required for its role (Table 9). Generators and evaluators never receive reference judgments. TWC additionally excludes the separately reported terminal decision, whereas WCV uses that decision and the private reference to verify the extracted claims. A decision explicitly stated within the trace remains part of the trace.

## C.2 CLAIM EXTRACTION AND WARRANT VERIFICATION

TWC uses GPT-5.6 Sol to extract explicit rule use, condition judgments, evidence, and aggregation while preserving errors and omissions (Figure 11). WCV applies the checks in Appendix A only after successful extraction, with extraction failures remaining not-evaluable rather than becoming negative labels.

The four tasks require different judgments: answering a policy question, testing contractual or regulatory conditions, or assessing a transaction. Table 10 shows how the same warrant checks apply to these task-specific judgments. For sources that do not link each judgment to specific evidence passages, we check whether the response uses the supplied evidence to justify that judgment rather than requiring it to cite a preassigned passage. Aggregation checks how the required judgments lead to the decision (Appendix A). For Transaction, we also check that the response assesses the transaction action posed in the question, separately from whether its compliance judgment is justified. Because generation prompts do not uniformly request exhaustive coverage, labels assess warrant completeness rather than instruction compliance alone. Human review next tests how closely these labels match judgments under the same requirements.

Table 9: Information available to each model or stage. Checkmarks denote visible inputs. Ref.: reference warrant; TWC: extracted claims; ℎ: reasoning trace; �ˆ: terminal decision. WCV verifies TWC claims using references withheld from generators and evaluators. Existing evaluators: Section 5.3.
<table><tr><td>Stage/model</td><td>Prompt Native h</td><td>ý</td><td>TWC</td><td>Ref.</td><td>|Output</td></tr><tr><td colspan="6">Generation and Annotation</td></tr><tr><td>Generator</td><td></td><td>X</td><td>X</td><td>X</td><td>native trace + terminal decision</td></tr><tr><td>TWC canonicalizer</td><td>√</td><td>X</td><td>X</td><td>X</td><td>explicit claims + trace locations</td></tr><tr><td>WCV verifier</td><td>X</td><td>√</td><td>√</td><td>√</td><td>layer findings + warrant label</td></tr><tr><td colspan="6">Evaluators</td></tr><tr><td>Existing evaluator</td><td></td><td></td><td>X</td><td>X</td><td>warrant score</td></tr><tr><td>Final-only ORM</td><td></td><td>√</td><td>X</td><td>X</td><td>outcome score</td></tr><tr><td>Full-response ORM</td><td></td><td></td><td>X</td><td>X</td><td>outcome score</td></tr><tr><td>Warrant-RM</td><td></td><td></td><td>X</td><td>X</td><td>warrant score</td></tr></table>

Table 10: What warrant checking requires for each task.
<table><tr><td>Task track</td><td>What must be judged?</td><td>What support is needed?</td><td>How is the decision checked?</td></tr><tr><td>Policy</td><td>The answer to the policy question</td><td>Use the supplied evidence or rules to justify the judgment without a preassigned passage.</td><td>Check that the stated grounds justify the answer.</td></tr><tr><td>Contract</td><td>Two to six condition judgments</td><td>For each condition, use the required evidence (or an accepted alternative) and its governing rule.</td><td>Combine all reference-required condition judgments into the decision.</td></tr><tr><td>Regulation</td><td>One to seven condition judgments</td><td>For each condition, use all jointly required evidence and rules, or an accepted alternative.</td><td>Apply the reference logic specifying which conditions must hold together or may serve as alternatives.</td></tr><tr><td>Transaction</td><td>Identify the transaction action and judge its overall compliance</td><td>Use supplied evidence to justify compliance and apply an accepted rule from the reference.</td><td>Check the grounds for compliance and agreement with the final decision. Check the transaction action separately.</td></tr></table>

## C.3 HUMAN REVIEW OF ANNOTATION QUALITY

To assess the labels against human judgments, reviewers examined 360 responses covering all four task tracks, with access to the prompt, trace, terminal decision, and reference warrant. Table 11 reports 84.2% agreement, Krippendorff’s $\alpha ~ = ~ 0 . 6 8 7$ , and complete-warrant precision/recall of 92.7%/78.5%. Source-group bootstrap intervals from 10,000 replicates are [79.9, 88.1]% for agreement and [0.605, 0.763] for �. In the table, � is the number of reviewed responses, � is Krippendorff’s agreement coefficient, and $P _ { \mathrm { c o m p l e t e } }$ and $R _ { \mathrm { c o m p l e t e } }$ are precision and recall for complete warrants. These statistics compare automatic labels against human judgments, rather than agreement between human reviewers.

The review also locates agreement within the reasoning process: condition judgments reach 87.1% $( \alpha = 0 . 7 2 9 )$ , aggregation 81.5% $( \alpha = 0 . 5 8 2 )$ , and terminal decisions 100%. Despite residual annotation error, the high complete-warrant precision supports warrant supervision, whose usefulness is tested directly through completeness assessment and response selection in Sections 5.4–5.5.

Table 11: Automatic annotation versus human review. Agreement is reported as a percentage (%). Precision and recall are computed with complete warrants as the positive class.
<table><tr><td>Group</td><td>n</td><td>α</td><td>Agreement</td><td>Pcomplete</td><td>Rcomplete</td></tr><tr><td>Transaction</td><td>108</td><td>0.654</td><td>83.3</td><td>一</td><td>一</td></tr><tr><td>Policy</td><td>36</td><td>0.894</td><td>97.2</td><td>一</td><td>一</td></tr><tr><td>Contract</td><td>162</td><td>0.635</td><td>83.3</td><td>一</td><td>1</td></tr><tr><td>Regulation</td><td>54</td><td>0.601</td><td>79.6</td><td>一</td><td>一</td></tr><tr><td>Overall</td><td>360</td><td>0.687</td><td>84.2</td><td>0.927</td><td>0.785</td></tr></table>

## D EVALUATION MEASURES AND MATCHED COMPARISONS

To connect warrant diagnosis with learning, the experiments distinguish response quality, evaluator performance, and downstream selection utility. The six generators listed in Table 1 each produce sixteen responses per prompt, with traces and terminal decisions recorded separately. This appendix specifies the measures and the shared conditions that make evaluator and reward-model comparisons interpretable.

## D.1 METRICS, WEIGHTING, AND UNCERTAINTY

To distinguish correct decisions from complete warrants, Section 5.1 reports answer accuracy, warrant completeness, and the correct-answer warrant gap (Final, Warr., and CWG in Table 1):

$$
\begin{array} { r l } & { \mathrm { A n s w e r ~ a c c u r a c y } = \frac { \# \mathrm { c o r r e c t ~ r e s p o n s e s } } { \# \mathrm { c o l l e c t e d ~ r e s p o n s e s } } , } \\ & { \mathrm { W a r r a n t ~ c o m p l e t e n e s s } = \frac { \# \mathrm { ~ r e s p o n s e s ~ w i t h ~ c o m p l e t e ~ w a r r a n t s } } { \# \mathrm { c o l l e c t e d ~ r e s p o n s e s } } , } \\ & { \mathrm { C W G } = \frac { \# \mathrm { c o r r e c t ~ r e s p o n s e s ~ w i t h ~ i n c o m p l e t e ~ w a r r a n t s } } { \# \mathrm { c o r r e c t ~ r e s p o n s e s ~ w i t h ~ e v a l u a b l e ~ w a r r a n t ~ l a b e l e } } . } \end{array}
$$

Here, # denotes the number of responses in the indicated category. After format retries where needed, every planned response slot has a usable response. Not-evaluable warrant labels remain in the first two denominators but are excluded from the correct-answer warrant gap and supervision. The gap is therefore a conditional proportion, not accuracy minus completeness.

Comparable averages. We choose the averaging scheme by experimental object and add modellevel averaging when summarizing multiple models. The following scopes distinguish alternative experimental summaries from further aggregation:

1. Generator results in Sections 5.1–5.2. For each generator, average the eight supported task– probe combinations equally. Only when pooling generators, first average the two generator models within each of the Qwen, gpt-oss, and DeepSeek families, then average the three family results equally. Comparisons across P1/P2/P3 use only Contract and Transaction, weighting them equally because both support all three probes.

2. Individual evaluator results in Sections 5.3–5.4. For each evaluator, first compute the metric within each task–probe combination, average supported probes within each task, then average the four task results equally (Equation 4), keeping response weights fixed. This is the evaluator-level result, before any averaging across evaluators.

3. Overall summaries across existing evaluators. Starting from the individual results in item 2, first average evaluators within each of the six categories in Table 2, then average the six category results equally. The categories are Qwen3.5, gpt-oss, DeepSeek-R1 distillations, Llama, other LLM judges, and pretrained scoring models. This additional step is used only for a summary across existing evaluators, preventing larger categories from dominating.

Thus, items 1 and 2 apply to different experiments, whereas item 3 is a further aggregation of item 2 when an overall evaluator summary is needed. Unsupported task–probe combinations are omitted. A bootstrap resample with an undefined metric in a required combination is excluded from that summary rather than averaged over fewer combinations.

What the evaluator metrics measure. We distinguish score discrimination, threshold-based identification, and probability quality. Let $w _ { i } \in \{ 0 , 1 \}$ indicate warrant completeness, $s _ { i }$ be the evaluator score, and $p _ { i }$ its calibrated completeness probability. All proportions and averages below use the fixed evaluation weights.

1. Discrimination. AUROC measures how often a complete warrant receives a higher score than an incomplete one, with half credit for ties. AUPRC is computed as average precision (AP), summarizing precision as recall increases (Saito & Rehmsmeier, 2015):

$$
\mathrm { A U R O C } = \operatorname* { P r } ( s ^ { + } > s ^ { - } ) + \frac { 1 } { 2 } \operatorname* { P r } ( s ^ { + } = s ^ { - } ) , \qquad \mathrm { A U P R C } = \sum _ { k } ( R _ { k } - R _ { k - 1 } ) P _ { k } .
$$

Here, $s ^ { + }$ and $s ^ { - }$ are scores drawn from complete and incomplete warrants in proportion to their evaluation weights. At successive distinct score thresholds, $P _ { k }$ is precision and $R _ { k }$ recall, with $R _ { 0 } = 0$ . Equal scores enter together. AUPRC is compared only within the same view because its baseline depends on the complete-warrant proportion.

2. Threshold-based identification. BalAcc gives equal importance to recognizing complete warrants and detecting incomplete ones:

$$
\mathrm { { B a l A c c } } = { \textstyle \frac { 1 } { 2 } } ( \mathrm { { T P R + T N R } } ) .
$$

TPR is the proportion of complete warrants classified as complete, and TNR the proportion of incomplete warrants classified as incomplete, at the validation-selected threshold.

3. Probability quality. Brier score measures probability error, while expected calibration error (ECE) measures agreement between predicted completeness probabilities and observed completeness rates (Guo et al., 2017):

$$
\mathrm { B r i e r } = \frac { \sum _ { i } \omega _ { i } ( p _ { i } - w _ { i } ) ^ { 2 } } { \sum _ { i } \omega _ { i } } , \qquad \mathrm { E C E } = \sum _ { b = 1 } ^ { 1 0 } \frac { W _ { b } } { W } | \bar { p } _ { b } - \bar { w } _ { b } | .
$$

Here, $\omega _ { i }$ is the evaluation weight, $\textstyle W = \sum _ { i } \omega _ { i }$ , and $W _ { b }$ is the total weight in probability bin �. The weighted means $\bar { p } _ { b }$ and $\bar { w } _ { b }$ are its predicted probability and observed completeness rate. Empty bins contribute zero. Lower Brier and ECE indicate better probability quality.

Calibration converts scores into completeness probabilities. Evaluation weights remain fixed, and full validation data determine score standardization, nonnegative-slope logistic calibration, and the threshold maximizing BalAcc, all reused in correct-only evaluation. Outcome classification uses native sigmoid probabilities at 0.5. Calibration affects probability quality without changing raw-score ranking.

Uncertainty and repeated fits. We use separate checks for source-group uncertainty, training variation, and human-review agreement:

1. Paired comparisons. To estimate uncertainty in paired comparisons without treating related responses as independent, Sections 5.1–5.2 and 5.4 use 2,000 task-stratified bootstrap resamples of linked source groups, sharing each resample across compared models. Intervals concern source-group sampling for the fitted models and are not adjusted for multiple comparisons.

2. Training variation. To describe variation across training runs, reward-model summaries report means over three fits and, where indicated, sample standard deviations (SDs).

3. Human review. To account for dependence among responses in the human review, its intervals resample responses sharing the same source identifier together.

4. Selection gains. To estimate uncertainty in selection gains for the collected candidate pools and fitted scorers, Best-of-� uses a task-stratified source-group jackknife. Each deletion removes the same source group and its responses from both scorers. Appendix E.5 reports these comparisons in both selection views.

Exact Best-of-� evaluation. To measure whether scores help choose a better response, each group consists of the sixteen responses produced by one generator for one prompt. For group �, let � be a uniformly drawn size-� subset, �<sub>�</sub> the score of response $i , w _ { i }$ its warrant-completeness label, and $o _ { i } = \pmb { \mathscr { k } } [ \hat { y } _ { i } = y _ { i } ]$ its correctness. The eligible groups for completeness and accuracy are $G _ { w }$ and $G _ { o }$ with normalized weights $\alpha _ { g } ^ { ( w ) }$ and $\alpha _ { g } ^ { ( o ) }$

$$
{ \widehat { i } } _ { g , S } ( s ) = \arg \operatorname* { m a x } _ { i \in S } s _ { i } ,\tag{5}
$$

$$
\mathrm { W a r r a n t c o m p l e t e n e s s } @ N ( s ) = \sum _ { g \in G _ { w } } \alpha _ { g } ^ { ( w ) } \mathbb { E } _ { S } [ w _ { \widehat { i } _ { g , S } ( s ) } ] ,\tag{6}
$$

$$
\mathrm { A n s w e r \ a c c u r a c y @ } N ( s ) = \sum _ { g \in { \cal G } _ { o } } \alpha _ { g } ^ { ( o ) } \mathbb { E } _ { S } [ o _ { \widehat { i } _ { g , S } ( s ) } ] .\tag{7}
$$

To avoid subset-sampling noise, expectations are computed exactly. With candidates ranked from $r = 1$ by decreasing score and original response order breaking ties, rank � is selected with probability ${ \binom { 1 6 - r } { N - 1 } } / { \binom { 1 6 } { N } }$

In the all-response-group view, $G _ { w } = G _ { o }$ contains every group with all sixteen responses present and both correctness and warrant-completeness labels determined for every response, regardless of whether these labels are positive or negative. In the target-specific opportunity view, every group contains both positive and negative labels for the evaluated target. All selectors share eligibility and weights within a view. Generators receive equal weight, as do available tasks within generator and supported probes within task: $\alpha _ { g } = 1 / ( 6 T _ { m } \bar { P _ { m t } } | { G _ { m t p } } | )$ , summing to one. Here, $T _ { m }$ counts available tasks for generator �, $P _ { m t }$ counts supported probes for task �, and $G _ { m t p }$ contains eligible groups for that generator, task, and probe. The all-response-group view uses identical groups and weights for both targets.

Random and Oracle baselines. Random chooses uniformly within each candidate subset, so its expected performance equals the group’s target-label mean for every �. Oracle uses the reference labels to choose a positive candidate whenever one is available. If group � has $c _ { g }$ positive candidates among its sixteen responses, its expected score is

$$
{ \mathrm { O r a c l e } } _ { g } ( N ) = 1 - { \frac { \binom { 1 6 - c _ { g } } { N } } { \binom { 1 6 } { N } } } .
$$

The fraction is the probability that all � candidates are negative, with impossible combinations treated as zero. Group scores are averaged using the same weights as the learned selectors. Oracle is target-specific: its two upper bounds need not be attained by the same response. In the all-responsegroup view, groups with no positive candidate keep Oracle below 100% even at $N = 1 6 .$ . All selectors coincide at $N = 1$ , whereas at $N = 1 6$ , the fixed-tie result selects directly from all responses. Oracle is nondecreasing as candidates increase, whereas a learned scorer can admit a higher-scoring negative response. The increase in Warrant-RM selection performance with � (Table 20) is therefore an empirical result, not a property imposed by the evaluation.

## D.2 EXISTING EVALUATORS

To compare existing assessment abilities under equal information access, all seventeen evaluators receive the prompt, native trace, and terminal decision, without reference warrants or diagnostic labels. Fourteen LLM judges use the rubric in Figure 12, while Skywork-Reward-V2-Llama-3.1-8B (Liu et al., 2026), dORM-8B (Lee et al., 2025), and TRM-8B (Zhang et al., 2026) supply pretrained scalar scores. Each has one complete evaluation run, with fixed score direction and validation-only calibration. A valid example and a contrasting example containing a contradiction check whether higher scalar scores indicate better responses.

This comparison tests transfer to RGDT warrant completeness. The judge rubric covers conditions, evidence, rules, and aggregation, but permits harmless omissions rather than requiring exhaustive reference coverage. Pretrained scorers instead receive no rubric. Correct-only results therefore reflect both evaluator ability and how closely its scoring criterion matches reference-warrant completeness. All scorers receive full, untruncated responses, including the decision, although TRM originally assessed reasoning alone. Native traces lack the aligned steps and labels needed for step-level process reward models (PRMs), motivating sequence-level reward modeling with one score per candidate answer here.

## D.3 MATCHED REWARD-MODEL TRAINING

To isolate trace access and supervision, all three reward models share Qwen3.5-9B, data splits, sample weights, optimizer, and a two-epoch budget, with three independent fits per configuration. Final-only ORM receives the prompt and terminal decision, while Full-response ORM and Warrant-RM also receive the same native trace (Table 9). The ORMs learn outcome correctness, whereas Warrant-RM learns warrant completeness. Thus, the full-response comparison changes only the supervision target.

Each reward model consists of a Qwen3.5-9B backbone and a scalar scoring head. The backbone represents the supplied prompt and response content, and the head maps this representation to one score for the response. The score targets outcome correctness for the ORMs and warrant completeness for Warrant-RM. The shared training configuration uses (1) rank-16 LoRA on attention and MLP projections and training of the scoring head (� = 32, dropout 0.05); (2) AdamW at $5 \times 1 0 ^ { - 5 }$ , 3% warmup, cosine decay, weight decay 0.01, and effective batch size 16; and (3) sample-weighted binary cross-entropy. Validation loss selects checkpoints, with Brier, AUROC, and the earliest step resolving ties. Inputs remain untruncated (16,384-token training/validation limit; 16,400 for formatted test inputs). Test labels never enter fitting or model selection.

Shared training weights. A response’s weight determines its contribution to the training or validation loss. We use shared weights to make the supervision comparison interpretable:

1. Shared response contributions. Each response receives the same weight for all three reward models. Full-response ORM and Warrant-RM therefore share both inputs and response contributions, allowing their comparison to isolate the supervision target.

2. Balance targets. We balance total weight across combinations of generator and answer correctness, generator and task, and task, probe, and answer correctness. Tasks receive equal weight, as do supported probes within each task. Iterative proportional fitting adjusts response weights to meet these targets, then normalizes weights to mean one within each split. Warrant-RM receives no extra class weighting or oversampling. Complete warrants account for 33.91% and 33.86% of training and validation weight.

3. Weighting sensitivity. We compare shared outcome-balanced weights with warrant-balanced weights. Table 12 first shows how differently the two schemes distribute weight: Min– max gives the weight range, and Top 1%/5% gives the share assigned to the highestweight responses. Kish effective sample size, $\begin{array} { r } { \dot { \mathrm { E S S } } = ( \sum _ { i } \omega _ { i } ) ^ { 2 } / \sum _ { i } \overline { { \omega } } _ { i } ^ { 2 } } \end{array}$ , summarizes this concentration, where $\omega _ { i }$ is a response weight and � the number of responses. Greater concentration raises top-weight shares and lowers ESS and ESS/�. Equal weights give $\mathrm { E S S } / n = 1 0 0 \%$ . Shared has a higher Top 5% share (19.84% versus 11.89%) and lower ESS/� (45.78% versus 84.32%), both indicating greater concentration. ESS does not count retained responses or independent source cases. Despite these distributional differences, Warrant-RM achieves similar mean correct-only AUROC (69.24% and 69.32%, respectively; Appendix E.4.2), supporting its warrant discrimination under both weighting choices.

Table 12: Training-weight concentration under the shared and warrant-derived schemes.
<table><tr><td>Weights</td><td>Min-max</td><td>Top 1%</td><td>Top 5%</td><td>Kish ESS</td><td>ESS/n</td></tr><tr><td>Shared</td><td>0.195-14.676</td><td>7.88%</td><td>19.84%</td><td>5,427.6</td><td>45.78%</td></tr><tr><td>Warrant-derived</td><td>0.283-3.646</td><td>3.44%</td><td>11.89%</td><td>9,996.6</td><td>84.32%</td></tr></table>

## E SUPPLEMENTARY EVIDENCE FOR DIAGNOSIS AND SUPERVISION

The main experiments connect warrant gaps to layered diagnosis and learned assessment. This appendix checks whether gap estimates depend on response collection or weighting, examines what warrant supervision improves, and tests whether those improvements help select better responses.

## E.1 ROBUSTNESS OF CORRECT-ANSWER WARRANT GAPS

To test whether format retries affect gap estimates, we exclude the 51 retried test responses and separately assign their unavailable original responses the labels that minimize or maximize the estimated gap. Gaps persist under both checks. Retries depend only on output format, without reference labels or correctness checks. In total, 273 of 86,784 responses (0.315%) required a retry beyond the four planned attempts.

Source-group bootstrap estimates support the persistence of warrant gaps across all six generators. Every draw retains the required task–probe combinations. Together with the response-collection checks, these results show that the observed gaps are not confined to one generator or a small set of formatting retries.

Removing the aggregation check. To test whether warrant gaps mainly arise from requiring responses to explain how condition judgments lead to the decision, we recompute the correctanswer warrant gap without the aggregation check. This ignores both omitted aggregation steps and incorrectly combined judgments, while retaining rule-use, condition, evidence, and outcome checks. Not-evaluable responses remain excluded from the gap calculation (0.1%–1.0% of responses, mean 0.4%). The gap decreases by only 2.3–7.3 pp per generator, and its mean falls from 40.2% to 35.7% (Table 1). Thus, even removing the entire aggregation requirement leaves substantial gaps in rule use, condition judgments, or evidence support. The observed incompleteness cannot be explained solely by the requirement to state aggregation explicitly.

## E.2 LAYERED ATTRIBUTION AND PROBE COMPARISONS

We first locate failures within the reasoning process, then compare matched probes to assess how support presentation affects warrant completeness.

## E.2.1 LAYERED FAILURE ATTRIBUTION

Failure counts and attribution shares. For correct responses with incomplete warrants, involvement counts failures and attribution divides each response’s contribution equally among its failed layers (Equation 3). In Table 13(a), Involv. abbreviates involvement: the percentage of incomplete responses that fail a given layer. Single is the attribution share contributed by responses failing only that layer, while Shared is the share allocated to that layer by responses failing multiple layers. Total is Single plus Shared. For example, a response failing both rule use and evidence counts once in each layer for involvement, but contributes half a unit to each for attribution. Involvement rates can therefore sum to more than 100%, whereas total attribution shares sum to 100% before rounding. Rule scope remains a separate diagnostic for Transaction P2/P3.

Joint failures. Joint failures explain why one error label is insufficient: 385 of 391 responses with a missing condition and 1,088 of 1,237 with an incorrect condition judgment also fail another layer. These patterns support reporting all failed requirements rather than assigning a single cause by checking order.

Robustness of attribution patterns. Using the Total column in Table 13(a) as the main-analysis reference, Table 13(b) tests whether the attribution pattern persists under different task weights and probe coverage. It uses All Probes for all supported task–probe combinations and Common P1 for the P1 probe shared by all four tasks. It compares task-equal weighting of 4,187 correct responses with incomplete warrants across 48 generator–task–probe combinations with P1-only evaluation of 1,895 such responses across 24 combinations. Although generators remain equally weighted, P1-only evaluation also changes the response population. Rule use remains the largest contribution, while aggregation overtakes evidence in both comparisons. Thus, supporting judgments with applicable rules remains a recurring difficulty, and the remaining profile helps identify which requirements deserve attention under a particular task mix.

Table 13: Failure attribution and weighting sensitivity (%). (a) Failure involvement and attribution shares. (b) Task-equal comparisons across all probes and common P1.
<table><tr><td colspan="5">(a) Primary Attribution</td></tr><tr><td>Layer</td><td>Involv.</td><td>Single</td><td>Shared</td><td>Total</td></tr><tr><td>Rule use</td><td>44.5</td><td>31.8</td><td>4.6</td><td>36.4</td></tr><tr><td>Condition</td><td>32.3</td><td>2.8</td><td>12.4</td><td>15.2</td></tr><tr><td>Evidence</td><td>33.9</td><td>20.0</td><td>5.2</td><td>25.2</td></tr><tr><td>Aggregation</td><td>41.6</td><td>9.5</td><td>13.6</td><td>23.2</td></tr></table>

<table><tr><td colspan="3">(b) Task-Equal Sensitivity</td></tr><tr><td>Layer</td><td>All Probes</td><td>Common P1</td></tr><tr><td>Rule use</td><td>34.5</td><td>33.9</td></tr><tr><td>Condition</td><td>16.9</td><td>17.8</td></tr><tr><td>Evidence</td><td>23.8</td><td>22.5</td></tr><tr><td>Aggregation</td><td>24.8</td><td>25.9</td></tr></table>

## E.2.2 MATCHED PROBE COMPARISONS

To assess how support presentation affects warrant completeness, we compare responses to the same case from the same generator and sampling position across probes (Table 14). Adding hard negatives in P2 yields a small mean change relative to P1 (+1.02 pp), whereas P3 reduces completeness in eleven of the twelve task–generator comparisons, with a mean change of −6.67 pp. For the tasks and generators evaluated here, longer natural document context therefore poses a more consistent challenge than adding hard negatives. These results illustrate how the probes distinguish reasoning performance under different presentations of supporting information, rather than form a predetermined difficulty ladder. Because P3 also changes support position, density, and structure, its effects cannot be attributed to input length alone.

Table 14: Matched changes in warrant completeness (pp). Positive values favor P2/P3 over P1. Mean gives equal weight to the twelve displayed estimates from Contract and Transaction.
<table><tr><td colspan="3">Generator Probe 2 – Probe 1 Probe 3 – Probe 1</td></tr><tr><td colspan="3">Transaction</td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>+0.78</td><td>-4.17</td></tr><tr><td> DeepSeek-R1-Distill-Llama-70B</td><td>+3.39</td><td>-10.42 -13.80</td></tr><tr><td>gpt-oss-20B S gpt-oss-120B</td><td>-0.52 0.00</td><td>-0.26</td></tr><tr><td>Qwen3.5-9B</td><td>+4.69</td><td>-10.94</td></tr><tr><td>Qwen3.5-27B</td><td>0.00</td><td>-8.85</td></tr><tr><td>Contract DeepSeek-R1-Distill-Llama-8B</td><td>+0.89</td><td>+0.30</td></tr><tr><td>DeepSeek-R1-Distill-Llama-70B</td><td>+2.68</td><td>-7.14</td></tr><tr><td>S gpt-oss-20B</td><td>0.00</td><td>-2.08</td></tr><tr><td></td><td>0.00</td><td>-10.12</td></tr><tr><td>S gpt-oss-120B</td><td></td><td>-9.52</td></tr><tr><td>Qwen3.5-9B</td><td>-1.79</td><td></td></tr><tr><td>Qwen3.5-27B</td><td>+2.08</td><td>-2.98</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Mean</td><td>+1.02</td><td>-6.67</td></tr></table>

## E.3 PROBABILITY QUALITY AND INCOMPLETE-WARRANT DETECTION

To complement discrimination metrics, Table 15 assesses two aspects of warrant evaluation: how closely calibrated probabilities match the labels, and how well thresholded predictions recognize each class. Probability maps and thresholds are fitted on full validation data and reused in the correct-only view (Appendix D.1). TPR (true positive rate) measures complete-warrant recall, while TNR (true negative rate) measures incomplete-warrant detection. TPR and TNR are reported as percentages, with subscripts f and c denoting the full and correct-only evaluation views. Brier and ECE are computed from full-view probabilities and reported on a 0–1 scale, with lower values indicating better probability quality. Reward-model rows average three fits.

The matched full-response comparison isolates the effect of warrant supervision. Relative to Fullresponse ORM, Warrant-RM reduces Brier error by 0.0231 (0.2093→0.1862), the lowest value in Table 15. Its incomplete-warrant detection rate increases by 9.11 pp in the full view (47.90%→57.01%) and 12.87 pp among correct answers (20.69%→33.56%). These gains accompany a 2.86 pp decrease in complete-warrant recall (87.02%→84.16%). Recall is identical across views because complete warrants require correct answers, so the complete-warrant responses remain the same. Thus, under

Table 15: Probability quality and warrant classification. Bold/underline: best/second.
<table><tr><td>Evaluator</td><td>Brier ↓</td><td>ECE↓</td><td>TPR↑</td><td>TNRf ↑</td><td>TNRc ↑</td></tr><tr><td colspan="6">Other LLM Judges</td></tr><tr><td>GPT-5.6 Sol</td><td>0.2194</td><td>0.0948</td><td>84.88</td><td>49.30</td><td>22.14</td></tr><tr><td>G Gemini-2.5-Pro</td><td>0.2220</td><td>0.0556</td><td>93.32</td><td>37.92</td><td>14.32</td></tr><tr><td>Z GLM-5.2</td><td>0.2190</td><td>0.0839</td><td>91.60</td><td>40.49</td><td>17.25</td></tr><tr><td colspan="6">Qwen3.5 Series</td></tr><tr><td>Qwen3.5-0.8B Qwen3.5-4B Qwen3.5-9B</td><td>0.2409 0.2147</td><td>0.0493 0.0600</td><td>76.58 93.25</td><td>40.00 34.60</td><td>28.29 12.94</td></tr><tr><td>Qwen3.5-27B Qwen3.5-35B-A3B</td><td>0.2137 0.2141</td><td>0.0737 0.0630</td><td>97.08 94.52</td><td>32.85 35.42</td><td>11.64 12.28</td></tr><tr><td></td><td>0.2143 gpt-oss Series</td><td>0.0605</td><td>97.49</td><td>33.12</td><td>10.53</td></tr><tr><td colspan="6"> gpt-oss-20B</td></tr><tr><td>S gpt-oss-120B</td><td>0.2321 0.2246</td><td>0.0945</td><td>89.56</td><td>35.25</td><td>16.54</td></tr><tr><td></td><td></td><td>0.0870</td><td>93.45</td><td>34.64</td><td>13.95</td></tr><tr><td colspan="6">DeepSeek-R1 Distillation Family</td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>0.2458</td><td>0.0672</td><td>79.03</td><td>37.04</td><td>28.81</td></tr><tr><td> DeepSeek-R1-Distill-Llama-70B</td><td>0.2153</td><td>0.0469</td><td>87.04</td><td>40.77</td><td>21.24</td></tr><tr><td colspan="6">Llama Series</td></tr><tr><td>∞ Llama-3.1-8B-Instruct</td><td>0.2265</td><td>0.0945</td><td>78.99</td><td>44.20</td><td>27.23</td></tr><tr><td>∞ Llama-3.1-70B-Instruct</td><td>0.2089</td><td>0.0409</td><td>98.01</td><td>29.77</td><td>10.77</td></tr><tr><td colspan="6">Pretrained Scoring Models</td></tr><tr><td>Skywork-Reward-V2-Llama-3.1-8B dORM-8B</td><td>0.2337 0.2335</td><td>0.1120</td><td>92.91</td><td>33.57</td><td>18.77</td></tr><tr><td>TRM TRM-8B</td><td></td><td>0.0748</td><td>76.25</td><td>44.91</td><td>25.06</td></tr><tr><td></td><td>0.2408</td><td>0.0631</td><td>57.61</td><td>54.88</td><td>46.37</td></tr><tr><td colspan="6">Matched Trained Scorers</td></tr><tr><td>Final-only ORM</td><td>0.2204</td><td>0.1045</td><td>86.13</td><td>42.87</td><td>17.55</td></tr><tr><td>Full-response ORM</td><td>0.2093</td><td>0.1073</td><td>87.02</td><td>47.90</td><td>20.69</td></tr><tr><td>Warrant-RM</td><td>0.1862</td><td>0.1003</td><td>84.16</td><td>57.01</td><td>33.56</td></tr></table>

matched inputs, warrant supervision lowers probability error and identifies more incomplete warrants in both views, with a modest reduction in complete-warrant recall.

## E.4 EFFECTS OF TRACE ACCESS AND WARRANT SUPERVISION

To distinguish gains from trace access and warrant supervision, Table 16 compares models that differ in one of these factors. In this table and Table 18, Warrant, Full-response, and Final-only denote Warrant-RM, Full-response ORM, and Final-only ORM, respectively. Each contrast subtracts the second model from the first. AUPRC differences are computed from the displayed three-fit means in Table 3. With identical full-response inputs and weights, warrant supervision improves warrant AUROC by 6.94 pp (72.99%→79.93%), AUPRC by 10.16 pp (68.96%→79.12%), and BalAcc by 3.13 pp (67.46%→70.59%). Outcome AUROC rises by 0.43 pp (91.48%→91.91%) and AUPRC by 0.52 pp (96.54%→97.06%), whereas outcome BalAcc decreases by 3.27 pp (84.75%→81.48%). The gain is strongest on the warrant target, motivating the correct-only analysis with answer correctness held constant.

## E.4.1 WARRANT ASSESSMENT AMONG CORRECT ANSWERS

To assess warrant completeness while holding answer correctness fixed, we use the correct-answeronly view defined in Section 5.3. This exploratory analysis was specified after the full-view results. It includes 2,720 responses (1,782 complete and 938 incomplete) from 90 cases in 76 source groups, with both classes in every supported task–probe combination. Scores, weights, calibration, and thresholds remain unchanged. Tables 17 and 18 report task-level comparisons.

Table 16: Paired effects of trace access and supervision in the full view (pp). Differences compare metric means across fitted runs.
<table><tr><td>Contrast</td><td>AUROC</td><td>AUPRC</td><td>BalAcc</td></tr><tr><td colspan="4">Warrant Target</td></tr><tr><td>Full-response – Final-only</td><td>+5.68</td><td>+4.58</td><td>+2.96</td></tr><tr><td>Warrant – Full-response</td><td>+6.94</td><td>+10.16</td><td>+3.13</td></tr><tr><td>Warrant – Final-only</td><td>+12.62</td><td>+14.74</td><td>+6.09</td></tr><tr><td colspan="4">Outcome Target</td></tr><tr><td>Full-response – Final-only</td><td>+1.92</td><td>+0.30</td><td>+0.88</td></tr><tr><td>Warrant – Full-response</td><td>+0.43</td><td>+0.52</td><td>-3.27</td></tr><tr><td>Warrant – Final-only</td><td>+2.35</td><td>+0.82</td><td>-2.39</td></tr></table>

Table 17: Correct-only warrant AUROC by task (%). Tasks average supported probes, and RM rows average three fits. Mean: task-equal average. Bold/underline: best/second.
<table><tr><td>Model</td><td>Transaction</td><td>Policy</td><td>Contract</td><td>Regulation</td><td>Mean</td></tr><tr><td colspan="6">Existing Evaluators</td></tr><tr><td> GPT-5.6 Sol</td><td>50.97</td><td>59.48</td><td>60.33</td><td>42.98</td><td>53.44</td></tr><tr><td>G Gemini-2.5-Pro</td><td>52.59 52.73</td><td>52.24 39.20</td><td>58.05 58.94</td><td>52.43 45.15</td><td>53.83 49.00</td></tr><tr><td>Z GLM-5.2 Qwen3.5-0.8B</td><td>52.77</td><td>46.92</td><td>60.38</td><td>53.52</td><td>53.40</td></tr><tr><td>Qwen3.5-4B</td><td>52.48</td><td>60.62</td><td>59.10</td><td>58.56</td><td>57.69</td></tr><tr><td>Qwen3.5-9B</td><td>55.08</td><td>52.17</td><td>57.89</td><td>59.54</td><td>56.17</td></tr><tr><td>Qwen3.5-27B</td><td>52.97</td><td>51.05</td><td>58.06</td><td>48.48</td><td>52.64</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>51.51</td><td>56.66</td><td>58.81</td><td>52.20</td><td>54.80</td></tr><tr><td>S gpt-oss-20B</td><td>54.60</td><td>48.04</td><td>60.17</td><td>52.53</td><td>53.84</td></tr><tr><td>S gpt-oss-120B</td><td>53.62</td><td>54.13</td><td>59.82</td><td>62.31</td><td>57.47</td></tr><tr><td> DeepSeek-R1-Distill-Llama-8B</td><td>50.42</td><td>51.03</td><td>57.30</td><td>54.92</td><td>53.42</td></tr><tr><td>DeepSeek-R1-Distill-Llama-70B</td><td>54.40</td><td>51.25</td><td>61.53</td><td>54.75</td><td>55.48</td></tr><tr><td>∞ Llama-3.1-8B-Instruct</td><td>51.46</td><td>46.77</td><td>54.60</td><td>46.94</td><td>49.94</td></tr><tr><td>∞ Llama-3.1-70B-Instruct</td><td>55.96</td><td>51.12</td><td>61.26</td><td>54.76</td><td>55.77</td></tr><tr><td> Skywork-Reward-V2-Llama-3.1-8B</td><td>54.62</td><td>45.61</td><td>56.07</td><td>51.25</td><td>51.89</td></tr><tr><td> dORM-8B</td><td>51.29</td><td>54.52</td><td>56.84</td><td>54.84</td><td>54.37</td></tr><tr><td>TAM TRM-8B</td><td>57.93</td><td>44.70</td><td>60.91</td><td>52.66</td><td>54.05</td></tr><tr><td></td><td>Matched Trained Scorers</td><td></td><td></td><td></td><td></td></tr><tr><td>Final-only ORM</td><td>48.66</td><td>37.04</td><td>57.82</td><td>63.92</td><td></td></tr><tr><td>Full-response ORM</td><td>61.97</td><td>54.15</td><td>55.11</td><td>64.26</td><td>51.86 58.87</td></tr><tr><td></td><td>67.05</td><td></td><td></td><td></td><td></td></tr><tr><td>Warrant-RM</td><td></td><td>67.99</td><td>72.48</td><td>69.44</td><td>69.24</td></tr></table>

The paired gains distinguish two effects: trace access improves AUROC by 7.01 pp (51.86%→58.87%), and warrant supervision adds 10.37 pp (58.87%→69.24%) with the same full-response inputs. Warrant-RM’s task-level mean gains over Full-response ORM are positive across all four tasks (5.08–17.37 pp): Transaction (61.97%→67.05%), Policy (54.15%→67.99%), Contract (55.11%→72.48%), Regulation (64.26%→69.44%). Warrant supervision therefore improves discrimination even when every answer is correct, rather than relying on incorrect answers to identify incomplete warrants.

Table 18: Correct-only AUROC gains from trace access and warrant supervision (pp). All models share training and validation weights.
<table><tr><td>Scope</td><td>Full-response – Final-only</td><td>Warrant – Full-response</td></tr><tr><td>Transaction</td><td>+13.32</td><td>+5.08</td></tr><tr><td>Policy</td><td>+17.11</td><td>+13.84</td></tr><tr><td>Contract</td><td>-2.71</td><td>+17.37</td></tr><tr><td>Regulation</td><td>+0.34</td><td>+5.18</td></tr></table>

To check whether supervision gains extend beyond task averages, Figure 5 compares every supported task–probe combination. Mean gains are positive throughout both views, showing that the benefit is distributed across input conditions rather than concentrated in one task average.

To compare how the models distinguish correct responses with and without complete warrants, Figure 7 contrasts their relative score ranks. Compared with Full-response ORM, Warrant-RM ranks complete warrants higher (median: 49.9→55.9) and incomplete warrants lower (46.8→38.6). This greater separation supports improved warrant assessment, while Section 5.5 tests its value for response selection.

![](images/59fe262d48dfceff32b9a20363c4c650be6cecf1068f9638bd883dbad4d04eb8.jpg)

![](images/d01d9ee1a2c5dbe146db2f98cd343baadfc32cb5dd7ec8fb63e1183084b2decf.jpg)  
Figure 7: Score-rank distributions among correct answers. Axes show weighted score percentiles averaged over three fits. Darker hexagons contain more within-panel response weight. Crosses mark median percentiles. Above the diagonal, Warrant-RM ranks responses higher than Full-response ORM, and below it, lower.

## E.4.2 EFFECT OF TRAINING WEIGHTS

As described in Appendix D.3, the three main reward models share weights balanced using outcome labels to hold response contributions fixed across supervision targets. Table 19 compares this choice with weights balanced using warrant labels using the same data and evaluation weights, with separately validated checkpoints and calibration. The selection rows use the warrant-completeness and answer-accuracy measures in Equation equation 7, with $N = 1 6 .$ . Relative to weights balanced using warrant labels, the main outcome-balanced weights improve full-view warrant AUROC by 2.12 pp $( \bar { 7 } 7 . 8 1 \% {  } 7 9 . 9 3 \% )$ and outcome AUROC by 4.79 pp $\mathsf { \bar { ( 8 7 . 1 2 \% {  } } 9 1 . 9 1 \% ) }$ , while correct-only warrant AUROC remains similar (69.32%→69.24%). Variability across fits is reported separately. The similar correct-only means support the warrant-discrimination result under both schemes, while common weights make the main supervision comparison more interpretable.

Table 19: Warrant-RM under two weighting schemes (%). Mean ± SD over three fits. Selection uses targetspecific opportunity groups. Bold/underline: larger/smaller mean.
<table><tr><td>Metric and view</td><td>Balanced using warrant labels</td><td>Balanced using outcome labels (main)</td></tr><tr><td colspan="3">AUROC ↑</td></tr><tr><td>Warrant completeness, full view</td><td> $7 7 . 8 1 _ { \pm 0 . 7 5 }$ </td><td> $\mathbf { 7 9 . 9 3 _ { \pm 1 . 8 5 } }$ </td></tr><tr><td>Warrant completeness, correct-only</td><td> ${ \bf 6 9 . 3 2 _ { \pm 0 . 6 5 } }$ </td><td> $\underline { { 6 9 . 2 4 _ { \pm 2 . 1 5 } } }$ </td></tr><tr><td>Outcome correctness, full view</td><td> $\underline { { 8 7 . 1 2 } } { \scriptstyle \pm 1 . 1 8 }$ </td><td> ${ \bf 9 1 . 9 1 _ { \pm 0 . 9 1 } }$ </td></tr><tr><td colspan="3">Best-of-N Selection ↑</td></tr><tr><td>Warrant completeness@16</td><td> $\underline { { 7 0 . 1 3 } } { \scriptstyle \pm 0 . 9 8 }$ </td><td> $7 1 . 5 2 _ { \pm 1 . 3 0 }$ </td></tr><tr><td>Answer accuracy@16</td><td> $\underline { { 7 7 . 1 4 _ { \pm 1 . 1 9 } } }$ </td><td> $7 9 . 2 3 _ { \pm 2 . 9 2 }$ </td></tr></table>

## E.5 RESPONSE SELECTION AND ROBUSTNESS CHECKS

To distinguish overall selection utility from ranking when candidates differ in quality, we compare two views of the original prompt–generator groups, then examine candidate counts, weighting, and tie-breaking. Within each view, all selectors share groups defined independently of their scores.

Evaluation views. (1) All response groups includes every group with all sixteen responses collected and both correctness and warrant-completeness labels determined for each response. This includes uniformly correct or incorrect answers and uniformly complete or incomplete warrants. Both targets use identical groups and weights. (2) Target-specific opportunity groups includes only groups containing both positive and negative labels for the evaluated target, defining groups separately for completeness and accuracy without changing any responses. The second view excludes groups where every choice has the same target result. Scoring, exact subset averaging, and group weights follow Appendix D.1. Table 20 reports both views.

Selection gains across candidate counts. To account for dependence among responses from linked source cases, we estimate uncertainty separately within each view using the paired source-group jackknife (Appendix D.1), conditional on the collected responses and fitted models. Figure 8 shows that Warrant-RM’s mean selection gains over the two outcome-supervised baselines generally widen with candidate count, with a more pronounced increase over Final-only ORM.

Across all response groups, Warrant-RM achieves the highest mean completeness throughout $N = 2 \AA$ 16 and the highest mean accuracy throughout $N = 6 – 1 6$ . Its completeness gain over Full-response ORM ranges from 0.58 to 1.23 pp, showing that the benefit extends across candidate counts without mixed-label conditioning. In target-specific opportunity groups, Warrant-RM achieves the highest mean completeness throughout $N = 2 { - } 1 6$ and accuracy throughout $N = 3 – 1 6$ , with an average completeness gain of 1.99 pp over Full-response ORM. The larger gains in this view make the ranking benefit more apparent when candidates differ in the target. Together, these results indicate that warrant supervision helps identify better-justified responses and use larger candidate sets for selection.

$$
 - \psi - \mathsf { W a r r a n t - R M - F u l l - r e s p o n s e \ O R M } \quad - \circ - \mathsf { W a r r a n t - R M - F i n a l - o n l y \ O R M }
$$

![](images/2f7a6de4cda681dd5cb13abe8b957f1bee2501efd11a3a4342d88eebb7e3252c.jpg)

## All response groups

![](images/83e9763d29882fc508121eaff8411ad65278fb02e2695827519b84d0926bd12f.jpg)  
Target-specific opportunity groups

![](images/79b3db20ff438f6afecbb8ea5159e8aeeb70faef39731eedced3f9055b5cb13c.jpg)

![](images/ad96060c1b9d6793b3bc3e20729cc255fd35c6c18b2375dad7dcc7302d206bf1.jpg)  
Figure 8: Selection gains. Bands show pointwise 95% source-group jackknife intervals. Vertical scales differ by row.

Robustness checks. To check whether these gains depend on weighting or tie-breaking choices, we examine both within the target-specific opportunity view.

1. Coverage and weighting. Following Appendix D.1, we give equal weight to tasks available for each generator. Completeness covers all 48 generator–task–probe combinations, while accuracy covers 44 because four combinations have no mixed-outcome groups: Policy for gpt-oss-120B, Qwen3.5-9B, and Qwen3.5-27B, and Regulation for Qwen3.5-27B. The resulting Transaction/Contract/Policy/Regulation weights are 31.94/31.94/12.50/23.61%.

Restricting the original weights to supported combinations changes the accuracy gain over Full-response ORM at � = 16 only from 1.66 to 1.77 pp, retaining the gain under this weighting change.

2. Tied scores. Replacing selection by original response order with uniform random choice among tied highest scores leaves Warrant-RM’s mean completeness higher than both outcome-supervised baselines and its mean accuracy higher than Full-response ORM at � = 16. Thus, changing tie-breaking in this check leaves the main response-selection conclusion unchanged.

Table 20: Complete selection results in both evaluation views (%). Means over three fits. Final/Full: Finalonly/Full-response ORM; WRM: Warrant-RM. Bold/underline: best/second among Final, Full, and WRM. Random: uniform choice; Oracle: target-specific upper bound.
<table><tr><td></td><td colspan="5">Warrant ↑</td><td colspan="5">Accuracy ↑</td></tr><tr><td></td><td>N|Random</td><td>Oracle</td><td>Final</td><td>Full</td><td>WRM</td><td>Random</td><td>Oracle</td><td>Final</td><td>Full</td><td>WRM</td></tr><tr><td colspan="10">All response groups</td></tr><tr><td>1</td><td>51.01</td><td>51.01</td><td>51.01</td><td>51.01</td><td>51.01</td><td>72.58</td><td>72.58</td><td>72.58</td><td>72.58</td><td>72.58</td></tr><tr><td>2</td><td>51.01</td><td>59.67</td><td>53.12</td><td>53.76</td><td>54.34</td><td>72.58</td><td>78.88</td><td>75.98</td><td>75.95</td><td>75.75</td></tr><tr><td>3</td><td>51.01</td><td>63.75</td><td>53.95</td><td>55.06</td><td>55.92</td><td>72.58</td><td>81.77</td><td>77.38</td><td>77.44</td><td>77.28</td></tr><tr><td>4</td><td>51.01</td><td>66.31</td><td>54.42</td><td>55.90</td><td>56.92</td><td>72.58</td><td>83.56</td><td>78.21</td><td>78.39</td><td>78.30</td></tr><tr><td>5</td><td>51.01</td><td>68.15</td><td>54.74</td><td>56.52</td><td>57.63</td><td>72.58</td><td>84.81</td><td>78.79</td><td>79.08</td><td>79.06</td></tr><tr><td>6</td><td>51.01</td><td>69.57</td><td>54.98</td><td>57.01</td><td>58.17</td><td>72.58</td><td>85.75</td><td>79.22</td><td>79.62</td><td>79.66</td></tr><tr><td>7</td><td>51.01</td><td>70.72</td><td>55.17</td><td>57.42</td><td>58.59</td><td>72.58</td><td>86.48</td><td>79.55</td><td>80.05</td><td>80.14</td></tr><tr><td>8</td><td>51.01</td><td>71.69</td><td>55.34</td><td>57.77</td><td>58.94</td><td>72.58</td><td>87.07</td><td>79.82</td><td>80.41</td><td>80.55</td></tr><tr><td>9</td><td>51.01</td><td>72.53</td><td>55.48</td><td>58.07</td><td>59.23</td><td>72.58</td><td>87.56</td><td>80.03</td><td>80.72</td><td>80.89</td></tr><tr><td>10</td><td>51.01</td><td>73.26</td><td>55.61</td><td>58.33</td><td>59.49</td><td>72.58</td><td>87.98</td><td>80.21</td><td>80.99</td><td>81.20</td></tr><tr><td>11</td><td>51.01</td><td>73.91</td><td>55.74</td><td>58.57</td><td>59.73</td><td>72.58</td><td>88.34</td><td>80.36</td><td>81.22</td><td>81.47</td></tr><tr><td>12</td><td>51.01</td><td>74.50</td><td>55.85</td><td>58.79</td><td>59.94</td><td>72.58</td><td>88.65</td><td>80.49</td><td>81.43</td><td>81.71</td></tr><tr><td>13</td><td>51.01</td><td>75.03</td><td>55.97</td><td>58.98</td><td>60.15</td><td>72.58</td><td>88.93</td><td>80.59</td><td>81.61</td><td>81.93</td></tr><tr><td>14</td><td>51.01</td><td>75.52</td><td>56.08</td><td>59.16</td><td>60.34</td><td>72.58</td><td>89.18</td><td>80.68</td><td>81.77</td><td>82.14</td></tr><tr><td>15</td><td>51.01</td><td>75.97</td><td>56.19</td><td>59.32</td><td>60.52</td><td>72.58</td><td>89.40</td><td>80.75</td><td>81.91</td><td>82.32</td></tr><tr><td>16</td><td>51.01</td><td>76.38</td><td>56.30</td><td>59.47</td><td>60.70</td><td>72.58</td><td>89.60</td><td>80.81</td><td>82.04</td><td>82.50</td></tr><tr><td colspan="9">Target-specific opportunity groups</td></tr><tr><td>1</td><td>53.50</td><td>53.50</td><td>53.50</td><td>53.50</td><td>53.50</td><td>58.66</td><td>58.66</td><td>58.66</td><td>58.66</td><td>58.66</td></tr><tr><td>2</td><td>53.50</td><td>68.70</td><td>56.60</td><td>58.07</td><td>59.14</td><td>58.66</td><td>73.86</td><td>64.82</td><td>65.76</td><td>65.54</td></tr><tr><td>3</td><td>53.50</td><td>75.77</td><td>57.74</td><td>60.32</td><td>61.92</td><td>58.66</td><td>80.82</td><td>67.17</td><td>68.66</td><td>68.67</td></tr><tr><td>4</td><td>53.50</td><td>80.29</td><td>58.33</td><td>61.81</td><td>63.73</td><td>58.66</td><td>85.22</td><td>68.55</td><td>70.50</td><td>70.79</td></tr><tr><td>5</td><td>53.50</td><td>83.60</td><td>58.73</td><td>62.95</td><td>65.05</td><td>58.66</td><td>88.32</td><td>69.45</td><td>71.82</td><td>72.35</td></tr><tr><td>6</td><td>53.50</td><td>86.22</td><td>59.03</td><td>63.90</td><td>66.06</td><td>58.66</td><td>90.66</td><td>70.07</td><td>72.84</td><td>73.55</td></tr><tr><td>7</td><td>53.50</td><td>88.41</td><td>59.27</td><td>64.71</td><td>66.88</td><td>58.66</td><td>92.47</td><td>70.50</td><td>73.66</td><td>74.51</td></tr><tr><td>8</td><td>53.50</td><td>90.28</td><td>59.46</td><td>65.42</td><td>67.57</td><td>58.66</td><td>93.93</td><td>70.81</td><td>74.36</td><td>75.31</td></tr><tr><td>9</td><td>53.50</td><td>91.93</td><td>59.63</td><td>66.06</td><td>68.17</td><td>58.66</td><td>95.12</td><td>71.02</td><td>74.95</td><td>75.98</td></tr><tr><td>10</td><td>53.50</td><td>93.40</td><td>59.78</td><td>66.64</td><td>68.71</td><td>58.66</td><td>96.12</td><td>71.15</td><td>75.47</td><td>76.57</td></tr><tr><td>11</td><td>53.50</td><td>94.74</td><td>59.91</td><td>67.17</td><td>69.22</td><td>58.66</td><td>96.97</td><td>71.23</td><td>75.92</td><td>77.10</td></tr><tr><td>12</td><td>53.50</td><td>95.96</td><td>60.04</td><td>67.66</td><td>69.70</td><td>58.66</td><td>97.71</td><td>71.27</td><td>76.33</td><td>77.58</td></tr><tr><td>13</td><td>53.50</td><td>97.08</td><td>60.15</td><td>68.12</td><td>70.17</td><td>58.66</td><td>98.37</td><td>71.27</td><td>76.69</td><td>78.03</td></tr><tr><td>14</td><td>53.50</td><td>98.12</td><td>60.27</td><td>68.56</td><td>70.62</td><td>58.66</td><td>98.97</td><td>71.24</td><td>77.02</td><td>78.45</td></tr><tr><td>15</td><td>53.50</td><td>99.09</td><td>60.38</td><td>68.96</td><td>71.07</td><td>58.66</td><td>99.51</td><td>71.19</td><td>77.31</td><td>78.85</td></tr><tr><td>16</td><td>53.50</td><td>100.00</td><td>60.50</td><td>69.34</td><td>71.52</td><td>58.66</td><td>100.00</td><td>71.12</td><td>77.57</td><td>79.23</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## F PROMPT EXAMPLES FOR GENERATION, ANNOTATION, AND EVALUATION

The examples follow one Contract case from task inputs through generation, annotation, and evaluation, distinguishing visible inputs from hidden references. The figures show input and response excerpts, instruction summaries, and selected annotation fields. Ellipses indicate omitted text, not omissions in the model inputs.

To show how the probes vary the available context while preserving the decision, Figure 9 presents one case with bound support (P1), added hard negatives (P2), and longer natural document context (P3). The same return/destruction clause appears as E001 in P1, E003 in P2, and E020 in P3 because passage identifiers are assigned within each prompt. R01–R03 identify the three conditions, not their reference judgments.

![](images/84d45f63c0cf7ebf99d4bb090669ac7386f0651d78287130c492d21ecf2ed1e6.jpg)  
Figure 9: One Contract case across three probes.

With these inputs established, Figure 10 pairs a summary of the P1 generation instructions with excerpts from a recorded gpt-oss-120B response. Its condition judgments appear in the reasoning trace, while its terminal decision is submitted separately.

![](images/44c6ec321dbc01d3aa8d44ec73fb2482240a0a2b08ce765785db1c37fc18ad9d.jpg)  
Figure 10: Generation instructions and a recorded response.

To show how this response receives a warrant label, Figure 11 follows claim extraction by TWC and reference-based verification by WCV. The extracted conditions C1, C2, and C3 correspond to input conditions R02, R03, and R01. The trace states all three judgments but uses only C3 in its final aggregation. This supports the negative decision, yet omits C1/C2 from the reference-required aggregation, yielding an incomplete warrant rather than an incorrect answer. TWC extracts any decision stated in the trace without seeing the separately submitted terminal decision (Appendix C.1).

![](images/b1c490e73bbad1265605d6506f88042e8c14312a14861b53ba7c9c84133fc42b.jpg)  
The decision follows the stated rule, but aggregation omits reference-required conditions C1/C2 . WCV marks this missing coverage without adding unstated reasoning.  
Figure 11: Claim extraction and warrant verification.

Finally, Figure 12 summarizes the LLM judge’s inputs, scoring instruction, and output format. The judge receives the full task, trace, and terminal decision, but no reference annotations or TWC/WCV outputs. Its JSON reports a warrant-completeness probability and a brief reason. Reward models instead return scalar scores, with inputs distinguished in Table 9. The rubric permits harmless omissions (Appendix D.2), so it should not be read as the reference-based WCV checking procedure.

![](images/a2909eae3d96044d56aa9fe377caca3a235ed235e0b81cf18acc00175bedbaf1.jpg)  
Figure 12: LLM-judge input and output format.