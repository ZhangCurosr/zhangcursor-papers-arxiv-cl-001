# The Labeling Problem in Hallucination Detection Benchmarks: An Empirical Evaluation

Jorma Valjakka   
Department of Computer Science University of Helsinki   
jorma.valjakka@helsinki.fi Juha Mylläri   
Department of Computer Science University of Helsinki   
juha.myllari@helsinki.fi Juhani Kivimäki   
Department of Computer Science University of Helsinki   
juhani.kivimaki@helsinki.fi

Jukka K. Nurminen Department of Computer Science University of Helsinki jukka.k.nurminen@helsinki.fi

## Abstract

In recent years, several methods for detecting when large language models (LLMs) hallucinate have been developed. These methods are often benchmarked with opendomain question answering (QA) datasets containing questions and corresponding short reference answers. First, an LLM is used to generate answers to questions within the QA dataset. Then, some automated labeling strategy is used to label these answers as hallucinated or not by comparing them with the reference answers in the dataset. This evaluation setting creates a methodological ambiguity between two criteria: reference faithfulness — whether the answer is fully supported by the reference — and factual correctness — whether the answer is free from contradictions and factually false specific claims. In practice, automated labelers may apply the former criterion even when the intended target is the latter.

We study this potential criterion mismatch using 900 human-labeled question– answer pairs spanning three commonly used QA datasets and three generator models, with labels targeting answer-level factual correctness. As automated label generation strategies, we evaluate lexical similarity metrics, a reference-entailment NLI baseline, and seven LLM judges under controlled prompt variants. Our experiments reveal substantial disagreement both among different automated labelgeneration strategies and between these automated labels and human annotations. Many strategies also exhibit strong directional error biases, and for most judge– generator pairs, replacing a faithfulness-oriented prompt with a factual-correctness prompt improves agreement with human annotations and reduces false-positive dominance, indicating that automated hallucination labels depend strongly on how the target criterion is specified. Label-source choice should therefore be considered a fundamental part of benchmark design and made explicit, validated, and matched with the benchmark goal.

## 1 Introduction

Large language models (LLMs) are notoriously prone to hallucinations. That is, they generate fluent statements containing fabricated or factually incorrect information [Ji et al., 2023, Niu et al., 2024]. Currently, new methods for hallucination detection are constantly being developed. Many of these methods are benchmarked using common question-and-answer (QA) datasets that contain questions and a set of reference answers to each question [Yin et al., 2024, Qiu and Miikkulainen, 2024, Chen et al., 2024, Duan et al., 2024, Farquhar et al., 2024]. Typically, the answer-label pairs needed for benchmarking these methods are produced by using an LLM to generate answers to the questions in the QA dataset and applying some automated strategy to label them as hallucinated or not [Bang et al., 2025, Janiak et al., 2025]. However, the validity of this approach hinges on a critical, often overlooked question: do these labels actually measure the intended construct? Common automated label sources, including lexical similarity metrics [Lin, 2004, Zhang et al., 2020] and LLM judges, are often used without directly validating this construct alignment [Janiak et al., 2025, Kulkarni et al., 2025], which might have a considerable effect on the evaluation outcomes.

We study the conceptual discrepancy between reference faithfulness and factual correctness in opendomain QA. A reference-faithfulness labeler treats answer content not entailed by the reference as hallucinated. A factual-correctness labeler treats an answer as hallucinated only if it contains a contradiction or an untrue specific claim. The two criteria diverge when an answer is correct but adds true information that is absent from the reference answer set. In such cases, strict faithfulness can systematically penalize correct elaborations. Prior to our work, this criterion mismatch has not been systematically studied as a source of automatic-label error in open-domain QA hallucination benchmarks to the best of our knowledge. Our work is not intended as a general critique of faithfulness: in evidence-grounded tasks such as RAG and summarization [Niu et al., 2024, Maynez et al., 2020], faithfulness is often the intended target. Our claim focuses strictly on open-domain QA benchmarks that target factual correctness, where the labeling criterion must be explicit and validated against that specific target.

To ground our analysis, we evaluate lexical similarity metrics, a reference-entailment NLI baseline, and seven diverse LLM judges (spanning OpenAI, Anthropic, and local open-weight models) against newly collected human annotations. We analyze the inter-annotator agreement among different automated labeling strategies and, in particular, with a human annotator. We find that the agreement between all automated methods and a human annotator is substantially lower than that between two human annotators. Furthermore, we examine the false-positive and false-negative error structures across label sources when human-annotator labels are treated as the ground truth. We also compare these label sources under controlled prompt ablations to isolate the impact of different labeling criteria.

In this paper, we make the following contributions.

• We construct a 900-pair human-annotated evaluation sample for auditing automated hallucination labelers in open-domain QA, with strong independent inter-annotator agreement on a 300-item overlap subset.

• We formalize the distinction between reference faithfulness and factual correctness, and show empirically that label sources approximating strict faithfulness produce false-positivedominant errors against a factual-correctness benchmark.

• We conduct a controlled prompt-ablation study showing that replacing a faithfulness-style criterion with a factual-correctness criterion substantially reduces this bias across most judges. Frontier judges from different model families show that capability alone does not determine the size of faithfulness-induced bias.

• We analyze different label sources in false-positive/false-negative error space, showing that lexical metrics, strict reference-entailment NLI, faithfulness-style prompts, and criterionaligned prompts occupy distinct parts of the error space.

## 2 Related Work

“Hallucination” does not mean exactly the same thing across benchmarks [Bang et al., 2025]. Some datasets emphasize factual correctness with respect to world knowledge [Li et al., 2023], while others focus on faithfulness to a provided source or retrieved evidence, as in summarization and RAG settings [Maynez et al., 2020, Niu et al., 2024]. Related benchmark frameworks such as TRUE [Honovich et al., 2022] standardize multiple factual-consistency datasets from diverse tasks under a common evaluation protocol, enabling example-level comparison of factual-consistency metrics across task settings. Factuality evaluation is sensitive to task formulation, dataset design, and the specific criterion being operationalized [Pagnoni et al., 2021, Kulkarni et al., 2025]. Therefore, findings from different hallucination benchmarks or labeling protocols may not be directly comparable without accounting for differences in the underlying construct being measured.

The proposed detectors evaluated in the benchmark studies cited in Section 1 do not use benchmark reference answers as inputs. Instead, they rely on uncertainty or semantic relationships among model-generated samples [Farquhar et al., 2024, Duan et al., 2024], semantic density estimated from model-generated reference responses [Qiu and Miikkulainen, 2024], or internal representations [Chen et al., 2024, Yin et al., 2024]. Benchmark gold answers are used separately to construct the correctness labels against which the detector scores are evaluated.

The QA label rules are not uniform. Duan et al. [2024] either use ROUGE-L or sentence-embedding similarity above 0.5; Qiu and Miikkulainen [2024] use ROUGE-L greater than 0.3;Chen et al. [2024] report both ROUGE-L greater than 0.5 and cosine similarity greater than 0.9 as separate correctness measures; and Yin et al. [2024] use ROUGE-L at least 0.5 in their main experiments, with ROUGE-L at least 0.3 and NLI entailment as sensitivity analyses. These rules compare answers with benchmark references and do not explicitly require evidence beyond them. Farquhar et al. [2024] use a different label source for sentence-length QA: GPT-4 is prompted to decide whether the proposed answer means the same as the expected answer, without separately verifying its factual correctness. As such, we consider all of the above to operationalize reference-faithfulness in their labeling strategies. However, the paragraph-length FACTUALBIO evaluation in Farquhar et al. [2024] differs: GPT-4 decomposes the generated biographies into individual factual claims, which the authors then manually label as true or false.

A related but distinct line of work moves from whole-output scoring to more explicit verification. Claim-level and knowledge-centric pipelines decompose model outputs into atomic claims and check them individually, enabling finer-grained factual assessment than single score estimates over entire responses [Min et al., 2023, Hu et al., 2024]. These approaches clarify the unit of evaluation but not the criterion used to define hallucination in a benchmark. In particular, making evaluation more fine-grained does not eliminate possible mismatch between reference faithfulness and factual correctness when hallucination labels are defined for a specific benchmark.

Recent meta-evaluation work shows that automatic evaluation signals can diverge sharply from human judgments at the example level. In summarization and related factual consistency settings, lexical overlap metrics such as ROUGE-L and semantic similarity metrics such as BERTScore have shown limited validity as measures of faithfulness or factuality relative to human annotations [Maynez et al., 2020, Honovich et al., 2022, Pagnoni et al., 2021]. QA-specific meta-evaluations support this: automatic metrics can overstate progress or misrank methods when the evaluation signal is poorly aligned with human factual judgments [Janiak et al., 2025, Kulkarni et al., 2025]. We therefore focus on label-source validity, that is, whether the labels used to evaluate hallucination detectors actually reflect the intended construct.

LLM-as-a-judge methods offer a scalable alternative to fixed lexical similarity metrics, but prior work shows that their reliability depends strongly on evaluation design. Judge outputs can vary with model capability, prompt formulation, and related design choices [Zheng et al., 2023, Thakur et al., 2025]. Our work builds on this literature, but asks a more specific question for open-domain QA with short reference answers: whether disagreement arises not only because judges are noisy or weak, but because the evaluation criterion encoded in a prompt is misaligned with the target construct. We study this as construct-level misalignment between a reference-faithfulness label source and a benchmark target of factual correctness, and show that it yields a strongly directional false-positive error pattern in automated labeling.

## 3 Methods

In this section, we define the hallucination constructs we study and formalize two competing labeling criteria: faithfulness to a provided reference versus factual correctness under human adjudication. We then describe our dataset generation and human-annotated benchmark, followed by the automated label sources and evaluation protocol used to compare judges, quantify agreement with humans, and analyze error direction and bias.

## 3.1 Labeling Constructs: Faithfulness vs. Factual Correctness

Let q denote a question, a an answer produced by an LLM, and $R = \{ r _ { 1 } , \ldots , r _ { k } \}$ a set of reference answers given in a dataset. We use claims(a) as an abstract decomposition of the factual content of a into individual claims c; this is a construct-level idealization, not a literal preprocessing step. We write $R \models c$ if the reference provides sufficient evidence for claim c, and $R \models \neg c$ if it contradicts c. Note that $R \not \vdash c$ is strictly weaker than $R \models \neg c \colon$ a claim can be unsupported by the reference without being contradicted by it. This distinction is central to the conceptual discrepancy we study.

Let verifiable(c) indicate that c is a specific factual claim (e.g., a date, name, number, or entity assertion), and $\mathrm { t r u e } ( c )$ that c is judged factually correct under our annotation protocol; the predicate denotes correctness under that protocol, not an assumed external oracle (see Appendix A). We formalize the two labeling criteria as indicator functions, where 1 signifies a hallucination:

$$
y _ { \mathrm { f a i t h } } ( a , R ) = \mathbf { 1 } [ \exists c \in \mathrm { c l a i m s } ( a ) : R \neq c ]\tag{1}
$$

$$
y _ { \mathrm { f a c t } } ( a , R ) = \mathbf { 1 } [ \exists c \in \mathrm { c l a i m s } ( a ) : R \vdash \lnot c \lor ( \mathrm { v e r i f i a b l e } ( c ) \land \lnot \mathrm { t r u e } ( c ) ) ]\tag{2}
$$

The faithfulness criterion triggers if any claim is unsupported by the reference; the factual-correctness criterion triggers only on contradictions or untrue factual claims. In practice, contradiction is instantiated as direct conflict with reference-asserted facts, not as formal logical negation. Thus, the criteria agree on three canonical cases — a correct core answer alone (both 0), a correct core with a false added specific detail (both 1), and a wrong core answer that contradicts the reference (both 1) — but diverge on a fourth: a correct core answer with a true elaboration absent from the short reference, where $y _ { \mathrm { f a i t h } } { = } 1$ but $y _ { \mathrm { f a c t } } { = } 0$ . Relative to a factual-correctness criterion, a labeler that operationalizes $y _ { \mathrm { f a i t h } }$ might therefore label factually correct elaborations absent from the reference as hallucinations, producing false positives for this class of cases. In the standard source-relative taxonomy [Maynez et al., 2020, Ji et al., 2023], $R \models \neg c$ is a canonical intrinsic-hallucination case, whereas $R \not \vdash c$ together with $R \not \vdash \lnot c$ is a source-extrinsic addition; the predicate true(c) introduces a separate world-truth axis over that class, so that “extrinsic to the source” and “factually incorrect” are not synonymous. Note that $y _ { \mathrm { f a c t } }$ is answer-level rather than core-answer-level: a correct core answer does not override an incorrect specific detail elsewhere in the response.

## 3.2 Dataset Generation

We drew candidate questions from the validation splits of TriviaQA [Joshi et al., 2017], HotpotQA [Yang et al., 2018], and TruthfulQA [Lin et al., 2022]. These datasets were selected to represent three different QA settings: factoid questions in TriviaQA, multi-hop questions in HotpotQA, and questions designed to elicit answers based on common misconceptions in TruthfulQA.

For each selected question, we generated one answer from each of three open-weight models — Llama-3-8B, Gemma-2-9B, and Mistral-7B — under a shared decoding configuration (see Appendix C.1). We chose these open-weight models to represent diverse model families while keeping answer generation tractable. Sampling was performed at the question level rather than at the question–answerpair level: once a question was selected, all three generator outputs for that question were included. Using the same question IDs across generators supports paired cross-generator comparisons.

## 3.3 Human-Annotated Benchmark

Our human-labeled hallucination annotation protocol targets $y _ { \mathrm { f a c t } }$ . Annotators were instructed to follow a reference-first procedure. This meant that they initially compared the answer against the reference answer set. If this comparison was insufficient to adjudicate a specific factual claim, annotators were allowed to consult external sources (see Appendix A for more details).

For the human-annotated benchmark, we randomly sampled 300 questions, allocated approximately equally across the three datasets (TriviaQA: 100, HotpotQA: 101, TruthfulQA: 99), each paired with answers from all three generator models, yielding $3 0 0 \times 3 = 9 0 0$ question–answer pairs. Sampling at the question level and retaining all three generator answers supports paired comparisons across generators while capturing variation in how different models answer the same question. The three datasets further provide variation across short factoid, multi-hop, and misconception-eliciting QA settings. We treat this benchmark as a diagnostic audit set for comparing automated label sources on fixed human-labeled items, not as a population-level estimate of hallucination prevalence.

All 900 question–answer pairs were labeled by a primary annotator (A1) under the protocol described in Appendix A. To estimate reliability, a second annotator (A2) independently labeled a subset of 300 question–answer pairs, where the questions were stratified with approximately equal allocation across the three datasets, and for each question, one answer from each of the three generators was included.

The resulting agreement is summarized in Table 3 in Appendix A. Using Cohen’s κ [Cohen, 1960] as the inter-annotator agreement statistic, we found the independent inter-annotator agreement to be strong (κ = 0.808, 95% CI: [0.729, 0.879]; see Appendix A).

All downstream evaluations used the complete A1 annotation set, annotator1\_label<sup>1</sup>, as the human reference labels for the 900 question–answer pairs. A2 labels were used only to estimate annotation reliability and to support quality control. Hallucination prevalence under the A1 protocol was 44.7% for Llama-3-8B, 38.0% for Gemma-2-9B, and 51.0% for Mistral-7B. At the dataset level, the hallucination rate was 72.3% for HotpotQA, 30.3% for TriviaQA, and 30.6% for TruthfulQA.

## 3.4 Automated Label Sources

We evaluated three families of automated labelers.

Lexical similarity metrics. We used ROUGE-L [Lin, 2004] and BERTScore [Zhang et al., 2020] to represent lexical overlap and embedding-based reference similarity, respectively. Additional metrics are included in Appendix D.4. For each generator, threshold τ was swept from 0 to 1 in steps of 0.01, with direction fixed so that lower similarity scores indicated hallucination. We report two thresholding protocols: an oracle-τ and a 3-fold cross-validated τ. The oracle-τ maximizes agreement with human-annotated labels on the full evaluation set. In the cross-validated protocol, τ is selected on the two training folds and applied to the held-out fold; the reported labels concatenate the three out-of-fold predictions. The oracle-τ serves as an optimistic diagnostic upper bound rather than a deployable procedure. The CV protocol is used as the primary non-oracle estimate.

Reference-entailment NLI baseline. We evaluated a T5-11B model trained similarly to the NLI baseline TRUE [Honovich et al., 2022], fine-tuned on a mixture of NLI datasets (see Appendix C.3). We use this model with the reference as premise and the model answer as hypothesis, labeling a prediction as a hallucination if the entailment probability falls below τ=0.5. We refer to this strategy as the strict setting. For comparability with these similarity metrics, we also report an oracle best-τ variant. Both settings operationalize y : entailment is defined relative to the premise, so neither can distinguish a claim that is unsupported by the reference but factually true from one that is false.

LLM judges. We evaluate seven LLM judges spanning proprietary API models and local openweight models. The primary ablation judge is GPT-5-mini [OpenAI, 2025a], evaluated under three prompt variants: p1 (faithfulness-style criterion), p2 (factual-correctness), and p3 (extended version of p2). Full prompt texts for these variants are given in Appendix B.

To disentangle the effect of labeling criterion from prompt structure, we test two additional prompts with GPT-5-mini alongside p1 and p2: p1s, a structured faithfulness prompt, and p2t, a terse factualcorrectness prompt. The auxiliary prompts and full details of this experiment are provided in Appendix D.3.

To test transfer across judge capability and model family, we also evaluate GPT-5-nano [OpenAI, 2025b], the OpenAI frontier model GPT-5.4 [OpenAI, 2026], and the Anthropic frontier model Claude Opus 4.7 [Anthropic, 2026]. The two frontier model judges probe whether criterion alignment is determined by capability alone or also varies across model families. Judge identifiers, providers, and experimental roles are summarized in Appendix C.2, Table 4.

We also evaluate three open-weight local judges: Gemma-2-9B, Llama-3-8B, and Qwen2.5-7B, with the same p1/p2/p3 prompts on all three generator sets. Qwen2.5-7B is not used as a generator, so it serves as an independent judge. Self-evaluation cases arise when judge and generator coincide (Gemma judging Gemma, Llama judging Llama); we mark these explicitly in the results and do not rely on them for our main conclusions.

## 3.5 Evaluation Protocol

Our main experiment tests how well different automated labelers align with human annotators operationalizing the factual-correctness criterion as defined in Section 3.1. For each question–answer pair, automated labels are compared to human-annotated labels to quantify agreement, bias, and error direction. Our primary metric is Cohen’s κ between each automated labeler and the human labels, reported as a standard chance-corrected agreement measure [Cohen, 1960]. We interpret κ together with positive-label bias, defined as the automated labeler hallucination rate minus the human labeler hallucination rate $( \hat { p } _ { \mathrm { a u t o } } - \hat { p } _ { \mathrm { h u m a n } } )$ , false positive (FP) and false-negative (FN) counts, and FP % (false positives as a proportion of all errors). To analyze error direction, we compute the false-positive rate $\mathrm { \dot { F } P R } = F P / \overset { \cdot } { ( } F \overset { \cdot } { P } + T N )$ and false-negative rate $\mathrm { F N R } = F N / ( F N + \dot { T } P )$ for each label source. These rates distinguish over-labeling from under-labeling, which a single agreement statistic can collapse.

Additionally, we perform a GPT-5-mini ablation experiment, where we report per-generator agreement, bias, and error counts, together with paired prompt-comparison statistics, including point differences in κ, bootstrap confidence intervals for $\Delta \kappa ,$ and McNemar’s exact two-sided test [McNemar, 1947]. McNemar’s test is appropriate because the prompt variants are evaluated on the same question–answer pairs. It tests whether one prompt corrects more paired errors than it introduces. Full design and analysis details for the factorial ablation are provided in Appendix D.3.

We then use the remaining six LLM judges to assess how far the broad p1→p2 pattern generalizes across judge families and model capabilities. We also analyze label-source structure using pairwise κ agreement matrices and FPR/FNR error-direction plot, with arrows showing movement from p1 to ${ \tt p } 2$ to p3 for multi-prompt judges.

## 4 Results

In this section, we present our results comparing automated labelers to human factual-correctness annotations<sup>2</sup>. We focus on how prompt criteria, baseline families, and judge types shape errors.

## 4.1 Prompt Criterion Changes Label Behavior

We report the main prompt-ablation results in Table 1. Across the evaluated LLM judges and generator sets, the faithfulness-style p1 prompt is false-positive-dominant: automated LLM judges often label answers as hallucinated when human annotators label them factually correct. Interestingly, this pattern is strongest for GPT-5.4 and weakest for Claude Opus 4.7, which are the two frontier judges in our benchmark. A long-form version with one row per judge–generator–prompt combination is provided in Appendix D.5, Table 11.

Replacing p1 with the factual-correctness p2 prompt generally improves agreement with humanannotated labels and consistently reduces false-positive dominance for nearly all judges. The effect is largest for GPT-5.4 and GPT-5-mini, moderate for GPT-5-nano, Gemma-2-9B, and Qwen2.5-7B, and smallest for Claude Opus 4.7, which is already close to the p2 region under p1. For the primary GPT-5-mini judge, this p1→p2 pattern also appears within individual datasets: Table 7 in Appendix D.2 shows that the p2 prompt leads to higher agreement than p1 across all datasets and generator models.

The extended p3 prompt does not consistently improve over p2. Across judge families, p2 is usually the best or near-best prompt, indicating that the main empirical change occurs between p1 and ${ \tt p } 2$ after which the effect of correctly specifying the target criterion saturates. For GPT-5-mini, p2 is significantly more often correct than p1 under McNemar’s test for all three generators $( p < 0 . 0 0 0 1 )$ , and the bootstrap confidence intervals for ∆κ exclude zero in all three comparisons. By contrast, no p2→p3 difference is statistically significant for any generator. Paired prompt-comparison statistics are reported in Appendix D.2, Table 6.

Because p1 and p2 differ in both labeling criterion and prompt structure, we also test the criterionalignment interpretation directly with a separate factorial control. In this ablation conducted with GPT-5-mini, we create variants of both p1 and ${ \tt p } 2$ to isolate the effects of labeling criterion and prompt structure. We find that a p1 variant with a prompt structure matching that of p2 results in no detectable improvement. Also, condensing the ${ \tt p } 2$ prompt so that its structure matches that of p1 does not yield worse agreement with the human annotator. This isolates criterion choice, rather than prompt elaboration, as the main driver of the $\mathsf { p } 1 \to \mathsf { p } 2$ agreement gain. Full factorial results and prompt definitions are provided in Appendix D.3.

Table 1: Prompt-ablation results against human labels for each judge–generator pair. Bias is positive when the automated labeler over-labels hallucinations relative to humans. FP/FN reports false positives and false negatives; FP% is false positives as a proportion of all errors. Best κ within each row is shown in bold. All values are computed against annotator1\_label, the complete primary human label set for the 900 question–answer pairs. Gen denotes the answer-generating model: L=Llama-3-8B, G=Gemma-2-9B, and M=Mistral-7B.
<table><tr><td rowspan="2">Judge</td><td rowspan="2">Gen</td><td colspan="4">p1</td><td colspan="4">p2</td><td colspan="4">p3</td></tr><tr><td>κ</td><td>Bias</td><td>FP/FN</td><td>FP%</td><td>κ</td><td>Bias</td><td>FP/FN</td><td>FP%</td><td>κ</td><td>Bias</td><td>FP/FN</td><td>FP%</td></tr><tr><td>API judges</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5-mini</td><td>L</td><td>0.530</td><td>+0.223</td><td>70/3</td><td>95.9</td><td>0.756</td><td>-0.020</td><td>15/21</td><td>41.7</td><td>0.744</td><td>+0.007</td><td>20/18</td><td>52.6</td></tr><tr><td>GPT-5-mini</td><td>G</td><td>0.498</td><td>+0.193</td><td>68/10</td><td>87.2</td><td>0.770</td><td>-0.033</td><td>11/21</td><td>34.4</td><td>0.759</td><td>-0.007</td><td>16/18</td><td>47.1</td></tr><tr><td>GPT-5-mini</td><td>M</td><td>0.194</td><td>+0.383</td><td>117/2</td><td>98.3</td><td>0.626</td><td>+0.027</td><td>32/24</td><td>57.1</td><td>0.646</td><td>+0.030</td><td>31/22</td><td>58.5</td></tr><tr><td>GPT-5-nano</td><td>L</td><td>0.536</td><td>+0.150</td><td>58/13</td><td>81.7</td><td>0.655</td><td>-0.017</td><td>23/28</td><td>45.1</td><td>0.676</td><td>+0.000</td><td>24/24</td><td>50.0</td></tr><tr><td>GPT-5-nano</td><td>G</td><td>0.592</td><td>+0.080</td><td>42/18</td><td>70.0</td><td>0.686</td><td>-0.020</td><td>19/25</td><td>43.2</td><td>0.641</td><td>-0.027</td><td>21/29</td><td>42.0</td></tr><tr><td>GPT-5-nano</td><td>M</td><td>0.389</td><td>+0.177</td><td>72/19</td><td>79.1</td><td>0.601</td><td>-0.047</td><td>23/37</td><td>38.3</td><td>0.581</td><td>-0.057</td><td>23/40</td><td>36.5</td></tr><tr><td>GPT-5.4</td><td>L</td><td>0.306</td><td>+0.363</td><td>110/1</td><td>99.1</td><td>0.779</td><td>+0.023</td><td>20/13</td><td>60.6</td><td>0.740</td><td>+0.050</td><td>27/12</td><td>69.2</td></tr><tr><td>GPT-5.4</td><td>G</td><td>0.240</td><td>+0.413</td><td>127/3</td><td>97.7</td><td>0.766</td><td>-0.003</td><td>16/17</td><td>48.5</td><td>0.774</td><td>+0.000</td><td>16/16</td><td>50.0</td></tr><tr><td>GPT-5.4</td><td>M</td><td>0.021</td><td>+0.480</td><td>144/0</td><td>100.0</td><td>0.619</td><td>+0.077</td><td>40/17</td><td>70.2</td><td>0.672</td><td>+0.070</td><td>35/14</td><td>71.4</td></tr><tr><td>Claude Opus 4.7</td><td>L</td><td>0.712</td><td>+0.037</td><td>27/16</td><td>62.8</td><td>0.764</td><td>-0.003</td><td>17/18</td><td>48.6</td><td>0.738</td><td>+0.010</td><td>21/18</td><td>53.8</td></tr><tr><td>Claude Opus 4.7</td><td>G</td><td>0.740</td><td>+0.067</td><td>29/9</td><td>76.3</td><td>0.764</td><td>-0.023</td><td>13/20</td><td>39.4</td><td>0.737</td><td>-0.010</td><td>17/20</td><td>45.9</td></tr><tr><td>Claude Opus 4.7</td><td>M</td><td>0.693</td><td>+0.033</td><td>28/18</td><td>60.9</td><td>0.706</td><td>+0.013</td><td>24/20</td><td>54.5</td><td>0.673</td><td>+0.023</td><td>28/21</td><td>57.1</td></tr><tr><td>Local judges</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-2-9B</td><td>L</td><td>0.567</td><td>+0.127</td><td>52/14</td><td>78.8</td><td>0.591</td><td>-0.047</td><td>23/37</td><td>38.3</td><td>0.557</td><td>+0.020</td><td>36/30</td><td>54.5</td></tr><tr><td>Gemma-2-9B</td><td>G (self)</td><td>0.463</td><td>+0.183</td><td>69/14</td><td>83.1</td><td>0.623</td><td>-0.083</td><td>13/38</td><td>25.5</td><td>0.628</td><td>-0.020</td><td>23/29</td><td>44.2</td></tr><tr><td>Gemma-2-9B</td><td>M</td><td>0.378</td><td>+0.090</td><td>60/33</td><td>64.5</td><td>0.496</td><td>-0.153</td><td>15/61</td><td>19.7</td><td>0.455</td><td>-0.087</td><td>28/54</td><td>34.1</td></tr><tr><td>Qwen2.5-7B</td><td>L</td><td>0.313</td><td>+0.280</td><td>96/12</td><td>88.9</td><td>0.483</td><td>+0.023</td><td>42/35</td><td>54.5</td><td>0.472</td><td>+0.100</td><td>55/25</td><td>68.8</td></tr><tr><td>Qwen2.5-7B</td><td>G</td><td>0.376</td><td>+0.283</td><td>93/8</td><td>92.1</td><td>0.607</td><td>-0.053</td><td>19/35</td><td>35.2</td><td>0.559</td><td>+0.083</td><td>45/20</td><td>69.2</td></tr><tr><td>Qwen2.5-7B</td><td>M</td><td>0.213</td><td>+0.203</td><td>89/28</td><td>76.1</td><td>0.428</td><td>-0.060</td><td>34/52</td><td>39.5</td><td>0.452</td><td>+0.047</td><td>48/34</td><td>58.5</td></tr><tr><td>Llama-3-8B</td><td>L (self)</td><td>0.370</td><td>+0.077</td><td>59/36</td><td>62.1</td><td>0.303</td><td>-0.240</td><td>13/85</td><td>13.3</td><td>0.255</td><td>-0.230</td><td>18/87</td><td>17.1</td></tr><tr><td>Llama-3-8B</td><td>G</td><td>0.227</td><td>+0.210</td><td>92/29</td><td>76.0</td><td>0.470</td><td>-0.157</td><td>11/58</td><td>15.9</td><td>0.384</td><td>-0.160</td><td>16/64</td><td>20.0</td></tr><tr><td>Llama-3-8B</td><td>M</td><td>0.278</td><td>+0.073</td><td>65/43</td><td>60.2</td><td>0.218</td><td>-0.383</td><td>2/117</td><td>1.7</td><td>0.185</td><td>-0.380</td><td>5/119</td><td>4.0</td></tr></table>

Smaller open-weight local judges are more heterogeneous in their labeling behavior than larger hosted API judges. Gemma-2-9B and Qwen2.5-7B partly reproduce the p1→p2 pattern at lower absolute agreement. Llama-3-8B is the main exception: under p2/p3 it shifts toward a false-negative-dominant profile, especially on Mistral-7B outputs.

## 4.2 Lexical and NLI Baselines Fail in Different Directions

We report results with ROUGE-L, BERTScore, and NLI baselines against the same human labels under oracle-τ and 3-fold CV-τ thresholding in Table 2. We also report METEOR and BARTScore-CNN results in Appendix D.4, which fall within the range spanned by these baselines. ROUGE-L is false-negative-prone, BERTScore is mixed across generators, and strict NLI is almost entirely false-positive-dominant (FP%: 98.3–99.4%). Oracle thresholding improves some scores but does not bring these baselines into the same agreement range as the criterion-aligned API judges.

Cross-validated thresholds are close to oracle thresholds within each generator set, suggesting that poor performance is not mainly due to unstable within-generator threshold selection. The NLI best-τ results are also informative: the oracle threshold collapses to τ = 0.01 for all generators, yet agreement remains modest. The main remaining variation is across generators, especially for ROUGE-L, whose optimal threshold is lower for longer Mistral-7B answers. We return to the construct-level implication of these error profiles in Section 5.

## 4.3 Error Direction Separates Label Sources

Pairwise inter-method agreement matrices (see Appendix D.6) show that criterion-aligned judges form a coherent agreement block, while lexical metrics, strict NLI, and faithfulness-style p1 sources show weaker agreement with all other label sources. The geometry of the distinct regimes of the label sources in false-positive / false-negative space is illustrated in Figure 1. The main p1→p2 movement for API judges is leftward, corresponding primarily to fewer false positives. GPT-5.4 p1 is the most extreme false-positive-dominant LLM judge, whereas for Claude Opus 4.7, the p1 region is much closer to the p2 region. Local judges occupy more varied positions, with the p2/p3 regions for Llama-3-8B moving into a false-negative-dominant region from the respective p1 region. The non-LLM baselines also separate by error direction: strict NLI is false-positive-dominant, ROUGE-L is mostly false-negative-prone, and BERTScore is mixed.

Table 2: Lexical similarity and NLI baselines against human-annotated factual-correctness labels. Oracle-τ reports a threshold tuned directly on the evaluation set. CV-τ reports 3-fold out-of-fold thresholding within each generator. For NLI, best-τ uses an oracle threshold. Bias is positive when the automated labeler over-labels hallucinations relative to humans. FP% is false positives as a proportion of all errors. L=Llama-3-8B, G=Gemma-2-9B, M=Mistral-7B.
<table><tr><td>Metric</td><td>Gen</td><td>T</td><td>κ</td><td>Bias</td><td>FP</td><td>FN</td><td>FP%</td></tr><tr><td>ROUGE-L (oracle-τ)</td><td>L</td><td>0.06</td><td>0.403</td><td>-0.183</td><td>15</td><td>70</td><td>17.6</td></tr><tr><td>ROUGE-L (oracle-τ)</td><td>G</td><td>0.07</td><td>0.483</td><td>-0.057</td><td>27</td><td>44</td><td>38.0</td></tr><tr><td>ROUGE-L (oracle-τ)</td><td>M</td><td>0.03</td><td>0.412</td><td>-0.257</td><td>6</td><td>83</td><td>6.7</td></tr><tr><td>ROUGE-L (CV-τ)</td><td>L</td><td>0.07 [0.06, 0.10]</td><td>0.353</td><td>-0.143</td><td>25</td><td>68</td><td>26.9</td></tr><tr><td>ROUGE-L (CV-τ)</td><td>G</td><td>0.07 [0.06, 0.07]</td><td>0.473</td><td>-0.067</td><td>26</td><td>46</td><td>36.1</td></tr><tr><td>ROUGE-L (CV-τ)</td><td>M</td><td>0.03 [0.03, 0.04]</td><td>0.385</td><td>-0.230</td><td>12</td><td>81</td><td>12.9</td></tr><tr><td>BERTScore (oracle-τ)</td><td>L</td><td>0.85</td><td>0.387</td><td>-0.047</td><td>38</td><td>52</td><td>42.2</td></tr><tr><td>BERTScore (oracle-τ)</td><td>G</td><td>0.85</td><td>0.486</td><td>+0.117</td><td>56</td><td>21</td><td>72.7</td></tr><tr><td>BERTScore (oracle-τ)</td><td>M</td><td>0.84</td><td>0.415</td><td>-0.067</td><td>34</td><td>54</td><td>38.6</td></tr><tr><td>BERTScore (CV-τ)</td><td>L</td><td>0.85 [0.85, 0.86]</td><td>0.318</td><td>-0.003</td><td>50</td><td>51</td><td>49.5</td></tr><tr><td>BERTScore (CV-τ)</td><td>G</td><td>0.85 [0.85, 0.85]</td><td>0.486</td><td>+0.117</td><td>56</td><td>21</td><td>72.7</td></tr><tr><td>BERTScore (CV-τ)</td><td>M</td><td>0.84 [0.84, 0.85]</td><td>0.394</td><td>-0.037</td><td>40</td><td>51</td><td>44.0</td></tr><tr><td>NLI (T5-11B) strict</td><td>L</td><td>0.50</td><td>0.053</td><td>+0.513</td><td>155</td><td>1</td><td>99.4</td></tr><tr><td>NLI (T5-11B) strict</td><td>G</td><td>0.50</td><td>0.025</td><td>+0.573</td><td>175</td><td>3</td><td>98.3</td></tr><tr><td>NLI (T5-11B) strict</td><td>M</td><td>0.50</td><td>0.049</td><td>+0.453</td><td>138</td><td>2</td><td>98.6</td></tr><tr><td>NLI (T5-11B) best-τ</td><td>L</td><td>0.01</td><td>0.380</td><td>+0.100</td><td>62</td><td>32</td><td>66.0</td></tr><tr><td>NLI (T5-11B) best-τ</td><td>G</td><td>0.01</td><td>0.217</td><td>+0.200</td><td>91</td><td>31</td><td>74.6</td></tr><tr><td>NLI (T5-11B) best-τ</td><td>M</td><td>0.01</td><td>0.395</td><td>-0.077</td><td>34</td><td>57</td><td>37.4</td></tr></table>

Error-direction space of automatic label sources  
![](images/ed8a21e1528bca106aece388fa23bd74b743187efd57d2a7314f9b771d6f1858.jpg)  
Figure 1: Error-direction space of automated label sources. Each point shows macro-averaged false-positive and false-negative rates over the three generator sets. The main p1→p2 movement is a reduction in false positives, while local judges and non-LLM baselines occupy distinct regions.

## 5 Discussion

In this section, we interpret our empirical findings and discuss limitations. We also outline some practical implications for building and validating hallucination benchmarks.

## 5.1 Faithfulness Bias as Measurement Error

The central result of this study is not only that some automatically generated labels disagree with human labels, but that the disagreement is directional. When factual correctness is the intended target, faithfulness-style label sources tend to over-label factually correct elaborations as hallucinations. A paired elaboration-removal intervention supports this interpretation: removing elaborative material from p1-evaluated answers reduced the false-positive rate from 52.9% to 23.2% (45 FP→TN vs. 4 TN→FP transitions; see Appendix E).

This matters because both criteria can be represented with the same binary label and the same term, “hallucination,” while measuring different constructs. Our aim is not to argue that factual correctness is always the correct target. In evidence-grounded settings such as RAG or summarization, source faithfulness may be the intended construct. The problem arises when a benchmark intends to evaluate factual correctness but uses a label source that operationalizes reference faithfulness.

## 5.2 The Main Intervention Is Criterion Alignment

The p1→p2 ablation suggests that the main intervention is criterion alignment, not prompt elaboration. The p3 prompt adds more explicit decision structure but does not consistently improve over p2. The important rule is the distinction between absence and contradiction: information absent from a short reference should not automatically be treated as hallucination. The labeling criterion vs. prompt structure factorial ablation (see Appendix D.3) supports this interpretation directly: holding the factual-correctness criterion fixed while adding decision structure produces no detectable change in agreement, whereas changing the criterion improves agreement at both levels of structure.

This distinction is small at the level of prompt wording but substantial for what the resulting binary labels measure. Without it, an automated LLM judge may silently convert a factual-correctness task into a reference-coverage task. Making the distinction explicit can therefore move the judge closer to the human factual-correctness labels without changing the dataset, the generated answers, or the binary labeling format.

## 5.3 Judge Capability Is Not Enough

The altered behavior of frontier judges under different prompting schemes shows that criterion alignment is not determined by capability alone. GPT-5.4 and Claude Opus 4.7 behave very differently under the same nominal p1 prompt, but move toward a more similar region under p2. We use this contrast not as a model-ranking claim, but as evidence that the judge’s default interpretation of an ambiguous criterion is itself an evaluation choice. Our experiments do not identify the mechanism underlying these cross-model differences, which we leave for future work.

Local judges further show that structured prompts are not sufficient by themselves. Gemma-2-9B and Qwen2.5-7B partly reproduce the criterion-alignment pattern at lower absolute agreement, whereas Llama-3-8B shifts toward a false-negative-dominant profile under p2/p3. This suggests a robustness limit of smaller models on realistic QA outputs rather than a failure to understand the abstract rule.

The same caution applies to non-LLM label sources. Lexical similarity metrics and strict NLI do not merely require better threshold calibration; they impose distinct operational criteria. In particular, answer–reference similarity can penalize elaboration even when the additional content is factually correct. Similarly, strict entailment can label correct information absent from the reference as non-entailed, and therefore as hallucinated under a reference-faithfulness criterion.

## 5.4 Implications and Limitations

These results suggest three practical recommendations for open-domain QA hallucination benchmarks. First, benchmark papers should specify whether labels target reference faithfulness or factual correctness. Second, when factual correctness is the intended target, judge prompts should explicitly distinguish contradiction from absence and allow correct elaboration for the models being evaluated. Third, automated label sources should be validated on a small human-labeled set before being used as ground truth, because a judge’s capability alone does not guarantee criterion alignment. This validation must target the intended criterion: agreement among raters applying a reference-faithfulness criterion does not by itself establish that the labels measure factual correctness.

Our results also bear on how existing benchmarks should be read. As surveyed in Section 2, many benchmark evaluations derive labels from the relationship between an answer and a supplied reference, so their conclusions are conditional on the chosen labeling rule. Results based on reference similarity should therefore be interpreted as performance under that specific operational criterion, rather than as general evidence of hallucination-detection or factual-correctness performance.

This study was limited to English open-domain QA with short references and three relatively small open-weight generators. The benchmark was constructed to compare label sources on a fixed humanlabeled set produced under a limited annotation budget; a larger sample would narrow the intervals reported here. Validation across further task families, longer answers, domains, and generators remains future work.

Our error counts are defined relative to human labels collected under a single criterion: the false positives reported for faithfulness-style sources are errors with respect to $y _ { \mathrm { { f a c t } } }$ , not in an absolute sense, and comparison against human annotations collected under a reference-faithfulness criterion is left to future work. The human labels themselves are not error-free: the main experiments use the A1 labels rather than consensus-adjudicated labels, and we did not log how often annotators consulted external sources, which limits our ability to quantify how often the short references were insufficient. It is also worth noting that answer-level binary labels cannot capture error severity or partial correctness.

LLM-based generation and judging may also vary across random seeds, implementations, providerside model versions, and inference configurations. The reported values should therefore be read as conditional on the fixed generated-answer set, judge versions, prompts, and inference setups evaluated here. The full results in Table 11 in Appendix D.5 also include self-evaluation conditions for Gemma-2-9B and Llama-3-8B, which are marked explicitly. These rows are reported for completeness but are not used as evidence that self-evaluation is unbiased. To ensure that all judges are able to follow the p2 factual-correctness decision rule, we conducted a six-branch sanity check on simple controlled cases. All seven judges passed this check (Appendix D.1, Table 5).

## 6 Conclusion

This paper examined a basic but often implicit assumption in automated hallucination evaluation: that different label sources using the same binary label measure the same construct. In short-reference open-domain QA, this assumption can fail. A faithfulness-style label source may penalize answers for adding information not contained in a short reference, even when the added information is factually correct. When the intended benchmark target is factual correctness, this creates a systematic form of measurement error.

Across seven LLM judges, three generator families, lexical similarity metrics, and a referenceentailment NLI baseline, we find that label-source choice changes both agreement with human labels and the direction of errors. Faithfulness-style prompts are often false-positive-dominant, while a factual-correctness prompt that explicitly distinguishes absence from contradiction substantially reduces this bias for most judges. The effect is not reducible to judge capability: frontier judges can behave very differently under the same nominal prompt, and local judges show additional robustness limits. Non-LLM baselines also fail in distinct directions, with ROUGE-L, BERTScore, and strict NLI occupying different regions of error-direction space.

The broader implication is that hallucination labels are not interchangeable implementation details. They are part of the benchmark design. Benchmarks should state whether they target reference faithfulness or factual correctness, and automated label sources should be validated against that intended criterion. For short-reference QA, a minimal but consequential rule is that information absent from the reference should not be treated as hallucination unless it is contradicted or factually false. Making this criterion explicit can turn automated hallucination evaluation from a surface label-matching exercise into a more valid measurement of the intended construct.

## Acknowledgments

This work was partly supported by Business Finland under grant agreement 23004 ELFMo of the ITEA4 programme. This funding supported three of the authors.

## References

Anthropic. Claude Opus 4.7. Anthropic model documentation, 2026. URL https://docs. anthropic.com/en/docs/about-claude/models. Accessed: 2026-04-30.

Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Jade Goldstein, Alon Lavie, Chin-Yew Lin, and Clare Voss, editors, Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. URL https://aclanthology.org/W05-0909/.

Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, and Pascale Fung. HalluLens: LLM hallucination benchmark. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 24128–24156, 2025. URL https://doi.org/10.18653/v1/2025.acl-long.1176.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. arXiv preprint, 1508.05326, 2015. URL https://arxiv.org/abs/1508.05326.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. INSIDE: LLMs’ internal states retain the power of hallucination detection. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= Zj12nzlQbz.

Jacob Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46, 1960. URL https://doi.org/10.1177/001316446002000104.

Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, and Kaidi Xu. Shifting attention to relevance: Towards the predictive uncertainty quantification of free-form large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5050–5063, 2024. URL https: //doi.org/10.18653/v1/2024.acl-long.276.

Bradley Efron and Robert J. Tibshirani. An Introduction to the Bootstrap. Chapman & Hall, New York, 1993. ISBN 9780412042317. URL https://doi.org/10.1201/9780429246593.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024. URL https: //doi.org/10.1038/s41586-024-07421-0.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. TRUE: Re-evaluating factual consistency evaluation. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3905–3920, 2022. URL https://doi.org/10.18653/v1/2022.naacl-main.287.

Xiangkun Hu, Dongyu Ru, Lin Qiu, Qipeng Guo, Tianhang Zhang, Yang Xu, Yun Luo, Pengfei Liu, Yue Zhang, and Zheng Zhang. Knowledge-centric hallucination detection. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6953–6975, 2024. URL https://doi.org/10.18653/v1/2024.emnlp-main.395.

Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Shwartz-Ziv, and Tomasz Jan Kajdanowicz. The illusion of progress: Re-evaluating hallucination detection in LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 34728–34745, 2025. doi: 10.18653/v1/2025.emnlp-main.1761. URL https: //doi.org/10.18653/v1/2025.emnlp-main.1761.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 2023. URL https://doi.org/10.1145/3571730.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601– 1611, 2017. URL https://doi.org/10.18653/v1/P17-1147.

Tushar Khot, Ashish Sabharwal, and Peter Clark. SCITAIL: A textual entailment dataset from science question answering. In Proceedings ofthe AAAI conference on artificial intelligence, volume 32, 2018. URL https://ojs.aaai.org/index.php/AAAI/article/view/12022.

Atharva Kulkarni, Yuan Zhang, Joel Ruben Antony Moniz, Xiou Ge, Bo-Hsiang Tseng, Dhivya Piraviperumal, Swabha Swayamdipta, and Hong Yu. Evaluating evaluation metrics – the mirage of hallucination detection. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 19013–19032, 2025. URL https://doi.org/10.18653/v1/2025. findings-emnlp.1035.

Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. HaluEval: A large-scale hallucination evaluation benchmark for large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6449–6464, 2023. doi: 10.18653/v1/2023.emnlp-main.397. URL https://aclanthology.org/2023.emnlp-main. 397/.

Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, 2004. URL https://aclanthology.org/W04-1013/.

Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, 2022. URL https://doi.org/10.18653/v1/2022.acl-long.229.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1906–1919, 2020. URL https://doi.org/10.18653/v1/ 2020.acl-main.173.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947. URL https://doi.org/10.1007/ BF02295996.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 12076–12100, 2023. URL https: //doi.org/10.18653/v1/2023.emnlp-main.741.

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. Adversarial nli: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main. 441. URL https://aclanthology.org/2020.acl-main.441/.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10862–10878, 2024. URL https://doi.org/10. 18653/v1/2024.acl-long.585.

OpenAI. GPT-5 mini. OpenAI model documentation, 2025a. URL https://platform.openai. com/docs/models/gpt-5-mini. Accessed: 2026-03-29.

OpenAI. GPT-5 nano. OpenAI model documentation, 2025b. URL https://platform.openai. com/docs/models/gpt-5-nano. Accessed: 2026-03-29.

OpenAI. Gpt-5.6 sol. https://developers.openai.com/api/docs/models/gpt-5.6-sol, 2026. Large language model.

OpenAI. GPT-5.4. OpenAI model documentation, 2026. URL https://platform.openai.com/ docs/models/gpt-5.4. Accessed: 2026-04-30.

Artidoro Pagnoni, Vidhisha Balachandran, and Yulia Tsvetkov. Understanding factuality in abstractive summarization with FRANK: A benchmark for factuality metrics. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4812–4829, 2021. URL https://doi.org/10.18653/ v1/2021.naacl-main.383.

Xin Qiu and Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space. In Advances in Neural Information Processing Systems, volume 37, pages 134507–134533, 2024. doi: 10.52202/079017-4274. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/f26d4fbaf7dfa115f1d4b3f104e26bce-Paper-Conference.pdf.

Tal Schuster, Adam Fisch, and Regina Barzilay. Get your vitamin c! robust fact verification with contrastive evidence. arXiv preprint, 2103.08541, 2021. URL https://arxiv.org/abs/2103. 08541.

Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. Judging the judges: Evaluating alignment and vulnerabilities in LLMs-as-judges. In Proceedings ofthe Fourth Workshop on Generation, Evaluation and Metrics (GEM<sup>2</sup>), pages 404–430, 2025. URL https://aclanthology.org/2025.gem-1.33/.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a largescale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, 2018. URL https://doi.org/10. 18653/v1/N18-1074.

Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, 2018. URL https://doi.org/10.18653/v1/ N18-1101.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018. URL https://doi.org/10.18653/v1/D18-1259.

Fan Yin, Jayanth Srinivasa, and Kai-Wei Chang. Characterizing truthfulness in large language model generations with local intrinsic dimension. In International Conference on Machine Learning, pages 57069–57084, 2024. URL https://proceedings.mlr.press/v235/yin24c.html.

Weizhe Yuan, Graham Neubig, and Pengfei Liu. Bartscore: evaluating generated text as text generation. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SkeHuCVFDr.

Yuan Zhang, Jason Baldridge, and Luheng He. Paws: Paraphrase adversaries from word scrambling. arXiv preprint, 1904.01130, 2019. URL https://arxiv.org/abs/1904.01130.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS 2023), Datasets and Benchmarks Track, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets\_and\_Benchmarks.html.

## A Annotation Protocol and Agreement

In this appendix, we describe the annotation protocol used in the human-annotated benchmark.

Human Annotation Criterion Annotators apply the factual-correctness criterion defined in Section 3.1. The label is assigned at the answer level. An answer is labeled as a hallucination (y = 1) if it contains either (i) a claim contradicted by the reference answer, or (ii) a specific verifiable factual claim, such as a date, name, number, place, organization, entity, or other identifiable detail, that is judged factually incorrect based on the reference answer, or, when the reference is insufficient, external factual adjudication.

An answer is labeled non-hallucinated (y = 0) if its verifiable specific claims are factually correct. Information absent from the reference is not by itself sufficient for a hallucination label: if an added detail is verifiably true, it is not penalized under $y _ { \mathrm { f a c t } } .$ Conversely, a correct core answer does not override a fabricated or factually incorrect specific detail elsewhere in the response.

Annotation Procedure For each question–answer pair, annotators follow this workflow:

1. Read the question.

2. Identify the specific verifiable claims and other identifiable factual details in the model answer.

3. Read the reference answer(s).

4. For each specific claim, determine whether it is (a) contradicted by the reference, or (b) unresolved by the reference, but factually false under external adjudication.

5. Assign label 1 if any claim fails step 4; otherwise, assign label 0.

External Factual Adjudication When the reference answer is insufficient to resolve a specific verifiable claim, annotators consult external evidence. In this study, external adjudication relied on public sources such as Wikipedia and general web searches. AI-assisted fact-checking could be used as a support tool, but not as the sole basis for assigning a label. The frequency of external adjudication was not systematically logged. Unsupported but true elaborations are not penalized.

Operational Scope This protocol is reference-first, but not strictly reference-only. Short reference answers included in the QA datasets are treated as primary evidence rather than as a complete specification of all true answer content. Accordingly, absence from the reference is not sufficient for a hallucination label. The operational target is answer-level factual correctness for specific verifiable claims, rather than strict reference faithfulness.

The protocol distinguishes hallucination labels from general answer quality. A vague, incomplete, or non-committal answer is not labeled as a hallucination unless it contradicts the reference or contains a specific, verifiable false claim. Likewise, an answer may be factually correct under this protocol, even if it contains correct information not present in the short reference.

Boundary Cases The following examples illustrate the distinction between reference faithfulness and factual correctness:

• Q:What is the capital of France?

– Ref: Paris. Answer: “Paris, the capital of France.” → 0

The added phrase is absent from the reference, but verifiably correct, so the answer is not labeled hallucination under y<sub>fact</sub>.

Table 3: Independent inter-annotator agreement on the 300-item subset. Confidence intervals are 95% question-level bootstrap intervals for Cohen’s κ with 1,000 resamples, keeping the three generator responses to each question together.
<table><tr><td>Dataset</td><td>n</td><td>Disagr.</td><td>Agree.</td><td>κ</td><td>CI low</td><td>CI high</td></tr><tr><td>TriviaQA</td><td>99</td><td>9</td><td>0.909</td><td>0.796</td><td>0.618</td><td>0.935</td></tr><tr><td>TruthfulQA</td><td>99</td><td>14</td><td>0.859</td><td>0.690</td><td>0.544</td><td>0.816</td></tr><tr><td>HotpotQÀ</td><td>102</td><td>6</td><td>0.941</td><td>0.832</td><td>0.642</td><td>0.968</td></tr><tr><td>Overall</td><td>300</td><td>29</td><td>0.903</td><td>0.808</td><td>0.729</td><td>0.879</td></tr></table>

– Ref: Paris. Answer: “Paris,founded in 1887.” → 1

The core answer is correct, but the added founding date is a specific verifiable claim that is factually incorrect under external adjudication.

– Ref: Paris. Answer: “Lyon.” → 1

The answer directly contradicts the reference.

– Ref: Paris. Answer: “I am not sure.” → 0

The answer is non-committal and may be unhelpful, but it does not assert a contradictory or fabricated specific claim.

• Q: What color is the sky on a clear day?

– Ref: Blue. Answer: “Blue, specifically a shade known as cerulean.” → 0 The elaboration is absent from the reference but verifiably correct, so it is not penalized under y<sub>fact</sub>.

Primary Benchmark Labels All 900 benchmark items are labeled by a primary annotator (A1) under the protocol above. These A1 labels serve as the benchmark labels for all downstream evaluation reported in the paper.

Independent Reliability Annotation To estimate annotation reliability, a second annotator (A2) independently labeled a 300-item subset of the benchmark spanning all three datasets and all three generator models. The A2 labels are used only for reliability estimation and to support quality control for the annotation protocol, quantifying the stability of the protocol across datasets and generator models. The main experiments do not replace A1 labels with consensus or adjudicated labels. Furthermore, this inter-annotator analysis is not to be taken as a claim that the benchmark’s labels are error-free.

Inter-Annotator Agreement Table 3 reports independent agreement on the 300-item subset. No disagreement-based corrections were applied to either annotator before computing these values. Overall agreement is strong, with Cohen’s κ = 0.808 and raw agreement of 90.3% (271/300). The 95% confidence interval for κ was computed with 1,000 question-level bootstrap resamples [Efron and Tibshirani, 1993], keeping the three generator responses to each sampled question together. The resulting overall interval is [0.729, 0.879]. Dataset-level point estimates are highest for HotpotQA (κ = 0.832) and lower for TruthfulQA (κ = 0.690), although the dataset-specific confidence intervals are wide and these differences should be interpreted descriptively.

## B Judge Prompts

This appendix provides the three main prompt definitions used in the cross-judge experiments. The two auxiliary prompts used only in the GPT-5-mini criterion-by-structure factorial ablation (p1s and p2t) are provided with that experiment in Appendix D.3.

p1 (faithfulness baseline) This prompting strategy follows a reference-grounded criterion. It labels an answer as a hallucination if a claim is unsupported by or contradicted by the reference. This prompt operationalizes the faithfulness-style criterion used in the main prompt-ablation experiments.

Your task: decide if a model answer contains a hallucination by comparing   
it against the reference answer.   
DEFINITION   
A hallucination is any claim that is not supported by, or contradicted by,   
the reference answer.   
Use the reference answer as ground truth. Do not use external knowledge.   
OUTPUT FORMAT (STRICT JSON ONLY)   
Return ONLY valid JSON with exactly these two fields:   
{"label": 0 or 1, "confidence": <float between 0 and 1>}   
Rules:   
- Do not include markdown, code fences, or extra text.   
- Do not include a reasoning field.   
- confidence must be a numeric value (not a string).

p2 (factual-correctness decision tree) This is a structured prompt with explicit rules separating contradiction from mere absence: “Absent from reference” is not the same as “contradicted by reference.” A hallucination requires either explicit contradiction or a verifiable false specific detail — directly operationalizing the annotation protocol described in Appendix A. Vague or non-committal answers are not hallucinations unless they contradict the reference or include a factually false specific claim.

You are a hallucination auditor for open-domain QA.   
Your task: decide if a model answer contains a hallucination.   
DEFINITION   
A hallucination is a claim that is EXPLICITLY CONTRADICTED by the reference   
answer, internally self-contradictory, or factually false at the level of a   
specific verifiable detail (such as a name, date, number, place, entity, or   
other concrete factual attribute).   
CRITICAL RULES   
1. "Absent from reference" is NOT the same as "contradicted by reference".   
Additional correct details not in the reference are NOT hallucinations.   
2. Wrong-but-vague is NOT a hallucination.   
Vague, incomplete, or non-committal answers are incorrect, not   
hallucinations, unless they explicitly contradict the reference or include   
a specific verifiable false detail.   
3. Focus on factual error, not mere non-overlap.   
Do not label an answer as a hallucination only because it contains content   
that does not appear in the reference.   
DECISION TREE   
Q1. Does the answer match the reference or a clear paraphrase?   
YES -> Q2. Does it add any claim EXPLICITLY CONTRADICTED by the reference or   
factually false at the level of a specific verifiable detail?   
NO -> LABEL=0   
YES -> LABEL=1   
NO -> Q3. Does the core answer conflict with the reference (different   
entity/value/fact)?   
YES -> LABEL=1   
NO -> Q4. Is the answer a refusal or expression of uncertainty?   
YES -> LABEL=0   
NO -> Q5. Does it include a specific verifiable detail   
that is factually false?   
YES -> LABEL=1   
NO -> LABEL=0   
OUTPUT FORMAT (STRICT JSON ONLY)   
Return ONLY valid JSON with exactly these two fields:   
{"label": 0 or 1, "confidence": <float between 0 and 1>}

```prolog
Rules:
- Do not include markdown, code fences, or extra text.
- Do not include a reasoning field.
- confidence must be a numeric value (not a string).
```

p3 (extended factual-correctness decision tree) This prompting strategy extends p2 with an explicit branch for non-committal references (e.g., “unknown,” refusal) and additional edge-case handling. The criterion remains factual correctness; p3 is treated as an exploratory edge-case variant that adds a more refined decision structure rather than a different construct.

You are a hallucination auditor for open-domain QA.   
Your task: decide if a model answer contains a hallucination.   
DEFINITION   
A hallucination is a claim that is EXPLICITLY CONTRADICTED by the reference   
answer, internally self-contradictory, or factually false at the level of a   
specific verifiable detail (such as a name, date, number, place, entity, or   
other concrete factual attribute).   
CRITICAL RULES   
1. "Absent from reference" is NOT the same as "contradicted by reference".   
Additional correct details not in the reference are NOT hallucinations.   
2. Wrong-but-vague is NOT a hallucination.   
Vague or incomplete answers are incorrect, not hallucinations, unless   
they explicitly contradict the reference or include a specific verifiable   
false detail.   
3. Core mismatch = hallucination.   
If the core answer conflicts with the reference (different entity/value/fact),   
label=1 even without an additional false specific detail.   
SPECIAL CASE -- non-committal references:   
If the reference is an explicit refusal or non-answer (e.g., "I have no comment",   
"Unknown", "Cannot be determined"), then any confident specific claim from the   
model is a hallucination. If the reference explains a dependency ("It depends   
on X because..."), evaluate normally.   
DECISION TREE   
Q1. Does the answer match the reference or a clear paraphrase?   
YES -> Q2. Does it add any claim EXPLICITLY CONTRADICTED by the reference or   
factually false at the level of a specific verifiable detail?   
NO -> LABEL=0   
YES -> LABEL=1   
NO -> Q3. Is the reference non-committal (refusal / unknown / cannot be   
determined)?   
YES -> Is the model answer a confident specific claim?   
YES -> LABEL=1   
NO -> LABEL=0   
NO -> Q4. Does the core answer conflict with the reference?   
YES -> LABEL=1   
NO -> Q5. Is the answer a refusal/uncertainty?   
YES -> LABEL=0   
NO -> Q6. Does it include a specific   
verifiable detail that is factually   
false?   
YES -> LABEL=1   
NO -> LABEL=0   
OUTPUT FORMAT (STRICT JSON ONLY)   
Return ONLY valid JSON with exactly these two fields:   
{"label": 0 or 1, "confidence": <float between 0 and 1>}   
Rules:   
- Do not include markdown, code fences, or extra text.   
- Do not include a reasoning field.   
- confidence must be a numeric value (not a string).

## C Experiment Setup

In this appendix, we describe details about our experiment setup.

## C.1 Answer Generation Setup

Answers are generated using the HuggingFace transformers<sup>3</sup> library with each model’s default instruction-tuned chat template (apply\_chat\_template). The system prompt is "You are a helpful assistant." for all three generators. Generation uses nucleus sampling with temperature T = 0.7, top-p = 0.95, max\_new\_tokens = 128, and a base random seed of 42. All three generator models use identical decoding parameters. Model identifiers are meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, and google/gemma-2-9b-it.

## C.2 Judge Evaluation Setup

Judges are prompted with the p1, p2, and p3 templates described in Appendix B and are required to return strict JSON outputs with a binary label and numeric confidence (on the interval [0,1]). Responses that cannot be parsed as valid JSON are excluded only if automatic repair is impossible. We summarize the judges used, their providers, and their role in our experiments in Table 4.

Table 4: Judge model identifiers and evaluation roles. API access dates are reported in the bibliography entries for the corresponding model documentation.
<table><tr><td>Judge</td><td>Provider / deployment</td><td>Role in experiments</td></tr><tr><td>GPT-5-mini</td><td>OpenAI API</td><td>Primary ablation judge</td></tr><tr><td>GPT-5-nano</td><td>OpenAI API</td><td>Weaker API judge</td></tr><tr><td>GPT-5.4</td><td>OpenAI API</td><td>Frontier OpenAI judge</td></tr><tr><td>Claude Opus 4.7</td><td>Anthropic API</td><td>Frontier Anthropic judge</td></tr><tr><td>Gemma-2-9B-it</td><td>Local open-weight</td><td>Local judge; includes one self condition</td></tr><tr><td>Llama-3-8B-Instruct</td><td>Local open-weight</td><td>Local judge; includes one self condition</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>Local open-weight</td><td>Independent local judge</td></tr></table>

## C.3 NLI Fine-tuning Datasets

The choice to use T5-11B as a literature-anchored entailment baseline alongside the lexical metrics was motivated by Yin et al. [2024], who use a T5-based NLI approach following Honovich et al. [2022] in the reference-as-premise, answer-as-hypothesis direction. We therefore report conclusions specific to this T5-11B instantiation and do not claim that it is optimal or representative of NLI systems in general. As briefly discussed in Section 3.4, as our reference-entailment NLI baseline, we used a fine-tuned T5-11B model similarly to TRUE [Honovich et al., 2022], but instead of using the version fine-tuned with the adversarial ANLI dataset [Nie et al., 2020], we used a version that was fine-tuned with a larger collection of datasets <sup>4</sup>. These datasets are (SNLI [Bowman et al., 2015], MNLI [Williams et al., 2018], Fever [Thorne et al., 2018], Scitail [Khot et al., 2018], PAWS [Zhang et al., 2019], VitaminC [Schuster et al., 2021]).

## D Ablation Studies

In this appendix, we present additional experiments and their results. These are used to further clarify our claims and to offer additional insights.

## D.1 Six-Branch Sanity Check

We conducted a simple six-branch sanity check to ensure that all LLM-judges used in our experiments were able to follow the instructions given in the p2 factual-correctness prompt. The six branches are: (1) exact correct answer, (2) correct answer with true elaboration absent from the reference, (3) correct core answer with a false added specific detail, (4) wrong answer with fabricated specific detail, (5) core answer mismatch, and (6) refusal or uncertainty expression. The results of this sanity check are shown in Table 5. All seven judges pass the diagnostic cases, including the key branch where a correct answer adds true information absent from the reference. This indicates that the prompt is understandable in simple controlled cases. The main benchmark results, therefore, reflect differences in robustness on realistic QA outputs, not a complete inability to follow the basic decision rule.

Table 5: Six-branch sanity check for the p2 factual-correctness prompt. The check tests basic promptfollowing behavior on simple diagnostic cases and is not used to estimate benchmark performance.
<table><tr><td>Judge</td><td>Type</td><td>Score</td></tr><tr><td>GPT-5-mini</td><td>API</td><td>6/6</td></tr><tr><td>GPT-5-nano</td><td>API</td><td>6/6</td></tr><tr><td>GPT-5.4</td><td>API</td><td>6/6</td></tr><tr><td>Claude Opus 4.7</td><td>API</td><td>6/6</td></tr><tr><td>Gemma-2-9B</td><td>Local</td><td>6/6</td></tr><tr><td>Llama-3-8B</td><td>Local</td><td>6/6</td></tr><tr><td>Qwen2.5-7B</td><td>Local</td><td>6/6</td></tr></table>

## D.2 GPT-5-mini ablations

In this appendix, we describe more detailed experiments limited to GPT-5-mini.

## D.2.1 Paired Prompt Comparisons

We use paired prompt comparisons to separate two questions. First, we measure how well each prompt variant agrees with the human labels using Cohen’s κ, which corrects nominal-label agreement for chance. Second, because the prompt variants are evaluated on the same generated answers, we compare their item-level correctness with an exact two-sided McNemar’s test over discordant correctness outcomes. Thus, ∆κ summarizes the change in chance-corrected agreement with the human labels, while the McNemar’s test asks whether one prompt is more often correct than the other on the same examples. We report nonparametric bootstrap intervals for ∆κ to quantify the sampling uncertainty of the agreement difference. The results are reported in Table 6.

The results show that switching from p1 (faithfulness) to p2 (factual correctness) yields large, statistically significant gains in agreement with humans across all generators, with bootstrap intervals for $\Delta \kappa$ that exclude zero by a wide margin. The p1→p3 contrast produces gains of a similar size, and the direct ${ \tt p } 2 \to { \tt p } 3$ comparison is not significant for any generator: the point differences range from −0.012 to +0.020, and every bootstrap interval contains zero. We therefore find no evidence of a consistent benefit from p3’s additional decision structure, although the experiment does not establish exact equivalence. Overall, the benefit comes from criterion alignment rather than from added prompt elaboration, with the largest correction on the hardest generator set (Mistral-7B, $\Delta \kappa = + 0 . 4 \bar { 3 } 2 )$ .

## D.2.2 Per-Dataset Breakdown

To check whether the p1→p2 improvement is driven by a single dataset, we also break down the GPT-5-mini agreement results by source dataset. This analysis tests the robustness of the prompt effect across three QA settings: multi-hop questions in HotpotQA, factoid questions in TriviaQA, and misconception-seeking questions in TruthfulQA. The results are reported in Table 7. The results show that switching from p1 to p2 consistently increases agreement with human labels across all three datasets and all generators. Gains are largest on TriviaQA and TruthfulQA, with smaller gains on HotpotQA, except for Mistral-7B, where the HotpotQA gain is also large.

## D.3 Labeling Criterion vs. Prompt Structure Factorial Ablation

The main p1→p2 comparison changes both the labeling criterion and the prompt format. We therefore evaluated a 2 × 2 design crossing reference faithfulness versus factual correctness with terse versus structured prompts. The auxiliary p1s and p2t prompts express p1’s criterion in a structured format and p2’s criterion in a terse format, respectively.

Table 6: Paired prompt comparisons for GPT-5-mini against the primary human labels. In each contrast, prompt a is the prompt before the arrow and prompt b is the prompt after the arrow. $\kappa _ { a }$ and $\kappa _ { b }$ denote the Cohen’s κ agreements of prompt a and prompt b with the human labels, respectively; $\Delta \kappa = \kappa _ { b } - \kappa _ { a }$ . Brackets give 95% nonparametric bootstrap intervals for $\Delta \kappa$ with 2,000 item-level resamples. McNemar’s p is the exact two-sided paired test over item-level correctness disagreements between the two prompt variants. L=Llama-3-8B, G=Gemma-2-9B, M=Mistral-7B.
<table><tr><td>Gen</td><td>Contrast</td><td> $\kappa _ { a }$ </td><td> $\kappa _ { b }$ </td><td> $\Delta \kappa$  [95% CI]</td><td>McNemar  $p$ </td></tr><tr><td>L</td><td>p1→p2</td><td>0.530</td><td>0.756</td><td>+0.226 [+0.127, +0.328]</td><td>&lt; 0.0001</td></tr><tr><td>L</td><td>p1→p3</td><td>0.530</td><td>0.744</td><td>+0.214 [+0.121, +0.313]</td><td>&lt; 0.0001</td></tr><tr><td>L</td><td>p2→p3</td><td>0.756</td><td>0.744</td><td>-0.012 [-0.060, +0.039]</td><td>0.7905</td></tr><tr><td>G</td><td>p1→p2</td><td>0.498</td><td>0.770</td><td>+0.272 [+0.179, +0.360]</td><td>&lt; 0.0001</td></tr><tr><td>G</td><td>p1→p3</td><td>0.498</td><td>0.759</td><td>+0.261 [+0.171, +0.347]</td><td>&lt; 0.0001</td></tr><tr><td>G</td><td>p2→p3</td><td>0.770</td><td>0.759</td><td>-0.011 [-0.059, +0.037]</td><td>0.7744</td></tr><tr><td>M</td><td>p1→p2</td><td>0.194</td><td>0.626</td><td>+0.432 [+0.332, +0.533]</td><td>&lt; 0.0001</td></tr><tr><td>M</td><td>p1→p3</td><td>0.194</td><td>0.646</td><td>+0.452 [+0.349, +0.559]</td><td>&lt; 0.0001</td></tr><tr><td>M</td><td>p2→p3</td><td>0.626</td><td>0.646</td><td>+0.020 [-0.047, +0.087]</td><td>0.7011</td></tr></table>

Table 7: Per-dataset Cohen’s κ agreement between GPT-5-mini and the primary human labels under p1 and p2. The breakdown checks whether the p1→p2 improvement is consistent across HotpotQA, TriviaQA, and TruthfulQA rather than being driven by a single dataset. Results are shown for each generator model.
<table><tr><td rowspan="2">Generator</td><td colspan="2">HotpotQA</td><td colspan="2">TriviaQA</td><td colspan="2">TruthfulQA</td></tr><tr><td> $\mathsf { p } 1$ </td><td> ${ \tt p } 2$ </td><td> $\mathsf { p } 1$ </td><td> ${ \mathsf p } 2$ </td><td> $\mathsf { p } 1$ </td><td> ${ \mathsf p } 2$ </td></tr><tr><td>Llama-3-8B</td><td>0.714</td><td>0.821</td><td>0.448</td><td>0.805</td><td>0.352</td><td>0.558</td></tr><tr><td>Gemma-2-9B</td><td>0.583</td><td>0.711</td><td>0.543</td><td>0.821</td><td>0.273</td><td>0.662</td></tr><tr><td>Mistral-7B</td><td>0.151</td><td>0.564</td><td>0.041</td><td>0.597</td><td>0.212</td><td>0.484</td></tr></table>

GPT-5-mini evaluated all four conditions on the same 900 responses (300 questions, three generators) in one interleaved run with a fixed shuffle seed. The original p1 and p2 prompts (Appendix B) were re-evaluated in this run; their results may therefore differ from Table 1. All contrasts use results from this factorial run.

Auxiliary factorial prompts. The p1s and p2t prompts are reproduced verbatim below.

p1s (reference faithfulness, structured) The prompt retains p1’s reference-faithfulness criterion, with a clarified definition, and adopts ${ \mathfrak { p } } 2 { \mathrm { : s } }$ section headings, three-rule format, and five-question decision-tree layout.

You are a hallucination auditor for open-domain QA.   
Your task: decide if a model answer contains a hallucination by comparing   
it against the reference answer.   
DEFINITION   
A hallucination is any claim that is not supported by the reference answer   
or is contradicted by it.   
CRITICAL RULES   
1. Evaluate reference support, including clear paraphrases and claims   
entailed by the reference. Different wording alone does not establish   
that a claim is unsupported.   
2. Evaluate the claims actually made by the answer.   
An incomplete or non-committal answer is not a hallucination merely   
because it fails to provide the requested information; assess whether   
it makes a claim that is not supported by the reference answer or is   
contradicted by it.   
3. Use the reference answer as ground truth. Do not use external knowledge.   
DECISION TREE

Q1. Does the answer match the reference or a clear paraphrase?   
YES -> Q2. Does it add any claim that is not supported by the reference   
answer or is contradicted by it?   
NO -> LABEL=0   
YES -> LABEL=1   
NO -> Q3. Does the core answer conflict with the reference (different   
entity/value/fact)?   
YES -> LABEL=1   
NO -> Q4. Is the answer a refusal or expression of uncertainty   
that makes no factual claim?   
YES -> LABEL=0   
NO -> Q5. Does it include any claim that is not   
supported by the reference answer or is   
contradicted by it?   
YES -> LABEL=1   
NO -> LABEL=0   
OUTPUT FORMAT (STRICT JSON ONLY)   
Return ONLY valid JSON with exactly these two fields:   
{"label": 0 or 1, "confidence": <float between 0 and 1>}   
Rules:   
- Do not include markdown, code fences, or extra text.   
- Do not include a reasoning field.   
- confidence must be a numeric value (not a string).

p2t (factual correctness, terse) This prompt retains the factual-correctness criterion of p2 while removing the explicit decision tree and most of the additional prompt structure.

You are a hallucination auditor for open-domain QA.   
Your task: decide if a model answer contains a hallucination by comparing   
it against the reference answer.   
DEFINITION   
A hallucination is any claim that is contradicted by the reference answer,   
or that is factually false in a specific verifiable detail.   
Information absent from the reference is not a hallucination if it is true.   
OUTPUT FORMAT (STRICT JSON ONLY)   
Return ONLY valid JSON with exactly these two fields:   
{"label": 0 or 1, "confidence": <float between 0 and 1>}   
Rules:   
- Do not include markdown, code fences, or extra text.   
- Do not include a reasoning field.   
- confidence must be a numeric value (not a string).

We report agreement with human labels for all four prompting strategies in Table 8. Both factualcorrectness prompts achieve higher agreement with a human annotator than either referencefaithfulness prompt. The latter remain strongly false-positive-dominant, with lower agreement for the structured p1s prompt than for the terse p1 prompt. In addition, we compare the labeling criteria (factual correctness vs. reference faithfulness) within each prompt structure (terse vs. structured) and the prompt structures within each criterion, using paired contrasts on the pooled set of 900 responses. The results of this comparison are shown in Table 9.

Changing the labeling criterion from reference faithfulness to factual correctness improves agreement with human labels in both prompt structures; both confidence intervals exclude zero. Under factual correctness, the difference between the terse and structured prompts is small, and its confidence interval includes zero. Under reference faithfulness, the structured p1s prompt has lower agreement than p1, with more false positives. These results suggest that the main p1→p2 improvement is more closely associated with the labeling criterion than with added prompt structure, while the effect of structure depends on the criterion.

Table 8: Criterion-by-structure factorial prompt experiment with GPT-5-mini. All four prompts were evaluated on the same 900 items in a single interleaved pass. p1 and ${ \tt p } 2$ are the paper prompts; p1s applies the reference-faithfulness criterion in p2’s structured decision-tree format, and p2t applies the factual-correctness criterion in p1’s terse format. Bias is positive when the automated labeler over-labels hallucinations relative to humans. L=Llama-3-8B, G=Gemma-2-9B, M=Mistral-7B; All pools the three generator sets.
<table><tr><td>Prompt</td><td>Criterion</td><td>Structure</td><td>Gen</td><td>κ</td><td>Bias</td><td>FP</td><td>FN</td><td>FPR</td></tr><tr><td>p1</td><td>Faithfulness</td><td>Terse</td><td>L</td><td>0.532</td><td>+0.237</td><td>72</td><td>1</td><td>0.434</td></tr><tr><td>p1</td><td>Faithfulness</td><td>Terse</td><td>G</td><td>0.466</td><td>+0.220</td><td>75</td><td>9</td><td>0.403</td></tr><tr><td>p1</td><td>Faithfulness</td><td>Terse</td><td>M</td><td>0.174</td><td>+0.387</td><td>119</td><td>3</td><td>0.810</td></tr><tr><td>p1</td><td>Faithfulness</td><td>Terse</td><td>All</td><td>0.409</td><td>+0.281</td><td>266</td><td>13</td><td>0.533</td></tr><tr><td>pls</td><td>Faithfulness</td><td>Structured</td><td>L</td><td>0.404</td><td>+0.293</td><td>91</td><td>3</td><td>0.548</td></tr><tr><td>p1s</td><td>Faithfulness</td><td>Structured</td><td>G</td><td>0.344</td><td>+0.280</td><td>95</td><td>11</td><td>0.511</td></tr><tr><td>p1s</td><td>Faithfulness</td><td>Structured</td><td>M</td><td>0.104</td><td>+0.433</td><td>131</td><td>1</td><td>0.891</td></tr><tr><td>p1s</td><td>Faithfulness</td><td>Structured</td><td>All</td><td>0.305</td><td>+0.336</td><td>317</td><td>15</td><td>0.635</td></tr><tr><td>p2t</td><td>Factual</td><td>Terse</td><td>L</td><td>0.756</td><td>-0.020</td><td>15</td><td>21</td><td>0.090</td></tr><tr><td>p2t</td><td>Factual</td><td>Terse</td><td>G</td><td>0.749</td><td>-0.023</td><td>14</td><td>21</td><td>0.075</td></tr><tr><td>p2t</td><td>Factual</td><td>Terse</td><td>M</td><td>0.626</td><td>+0.020</td><td>31</td><td>25</td><td>0.211</td></tr><tr><td>p2t</td><td>Factual</td><td>Terse</td><td>All</td><td>0.714</td><td>-0.008</td><td>60</td><td>67</td><td>0.120</td></tr><tr><td>p2</td><td>Factual</td><td>Structured</td><td>L</td><td>0.811</td><td>-0.007</td><td>13</td><td>15</td><td>0.078</td></tr><tr><td>p2</td><td>Factual</td><td>Structured</td><td>G</td><td>0.763</td><td>-0.030</td><td>12</td><td>21</td><td>0.065</td></tr><tr><td>p2</td><td>Factual</td><td>Structured</td><td>M</td><td>0.633</td><td>+0.023</td><td>31</td><td>24</td><td>0.211</td></tr><tr><td>p2</td><td>Factual</td><td>Structured</td><td>All</td><td>0.739</td><td>-0.004</td><td>56</td><td>60</td><td>0.112</td></tr></table>

Table 9: Paired contrasts from the criterion-by-structure factorial prompt experiment, pooled over the three generator sets. The first two rows hold prompt structure constant and vary the labeling criterion; the last two hold the criterion constant and vary prompt structure. $\Delta \kappa = \kappa _ { b } - \kappa _ { a } .$ . Brackets give 95% nonparametric bootstrap intervals based on 2,000 question-level resamples, keeping the three generator responses to each question together. Diff% is the proportion of items receiving different labels under the two prompts.
<table><tr><td>Contrast</td><td>Held constant</td><td>Diff%</td><td> $\kappa _ { a }$ </td><td> $\kappa _ { b }$ </td><td>∆κ [95% CI]</td></tr><tr><td> $\mathsf { p } 1 \to \mathsf { p } 2 \mathsf { t }$ </td><td>Structure: terse</td><td>29.6</td><td>0.409</td><td>0.714</td><td>+0.305 [+0.240, +0.369]</td></tr><tr><td> ${ \mathsf { p l s } } \to { \mathsf { p } } 2$ </td><td>Structure: structured</td><td>34.2</td><td>0.305</td><td>0.739</td><td>+0.434 [+0.371, +0.494]</td></tr><tr><td> $\mathsf { p 2 t } \to \mathsf { p 2 }$ </td><td>Criterion: factual correctness</td><td>4.3</td><td>0.714</td><td>0.739</td><td>+0.025 [+0.000, +0.053]</td></tr><tr><td> ${ \mathsf { p } } 1 \to { \mathsf { p } } 1 { \mathsf { s } }$ </td><td>Criterion: reference faithfulness</td><td>9.9</td><td>0.409</td><td>0.305</td><td>-0.104 [-0.142, -0.066]</td></tr></table>

## D.4 METEOR and BARTScore-CNN Baselines

To test whether the agreement pattern reported for ROUGE-L and BERTScore is specific to those two metrics, we evaluated two additional metrics, namely, METEOR [Banerjee and Lavie, 2005] and BARTScore-CNN [Yuan et al., 2021] on the same 900 items, using the same human factualcorrectness labels, multi-reference aggregation, and oracle and 3-fold cross-validated evaluation framework as in Table 2.

METEOR is computed using NLTK’s meteor\_score implementation with default metric parameters, Treebank tokenization, and WordNet synonym matching. For BARTScore-CNN, we use the facebook/bart-large-cnn checkpoint and evaluate both reference-to-answer and answerto-reference directions. Because BARTScore is an unbounded mean token log-likelihood, a fixed threshold grid on [0, 1] does not apply; its threshold candidates are the exact midpoints between observed scores. Restricting this search to a matched 101-point quantile grid changes oracle κ by at most 0.008, indicating that the finer candidate set does not materially affect the reported agreement.

The full results are reported in Table 10. Macro-averaged oracle agreement is $\kappa = 0 . 4 0 5$ for METEOR, $\kappa ~ = ~ 0 . 3 3 5$ for BARTScore-CNN ref→hyp, and $\kappa ~ = ~ 0 . 4 1 1$ for BARTScore-CNN hyp→ref, compared with 0.433, 0.429, and 0.330 for ROUGE-L, BERTScore, and threshold-tuned NLI. The added metrics therefore fall within the range already spanned by the baselines in the main text, indicating that the qualitative conclusion is not specific to ROUGE-L and BERTScore.

Table 10: METEOR and BARTScore-CNN baselines. For BARTScore, ref→hyp scores log P(answer | reference) and hyp→ref scores the reverse. Bias is positive when the automated labeler over-labels hallucinations relative to humans. FP% is false positives as a proportion of all errors. L=Llama-3-8B, G=Gemma-2-9B, M=Mistral-7B.
<table><tr><td>Metric</td><td>Gen</td><td>T</td><td>κ</td><td>Bias</td><td>FP</td><td>FN</td><td>FP%</td></tr><tr><td>METEOR (oracle-τ)</td><td>L</td><td>0.13</td><td>0.423</td><td>-0.137</td><td>21</td><td>62</td><td>25.3</td></tr><tr><td>METEOR (oracle-τ)</td><td>G</td><td>0.12</td><td>0.380</td><td>+0.077</td><td>57</td><td>34</td><td>62.6</td></tr><tr><td>METEOR (oracle-τ)</td><td>M</td><td>0.09</td><td>0.410</td><td>-0.157</td><td>21</td><td>68</td><td>23.6</td></tr><tr><td>METEOR (CV-τ)</td><td>L</td><td>0.13 [0.11, 0.15]</td><td>0.354</td><td>-0.137</td><td>26</td><td>67</td><td>28.0</td></tr><tr><td>METEOR (CV-τ)</td><td>G</td><td>0.13 [0.12, 0.15]</td><td>0.346</td><td>+0.097</td><td>63</td><td>34</td><td>64.9</td></tr><tr><td>METEOR (CV-τ)</td><td>M</td><td>0.08 [0.06, 0.09]</td><td>0.384</td><td>-0.190</td><td>18</td><td>75</td><td>19.4</td></tr><tr><td>BARTScore ref→hyp (oracle-τ)</td><td>L</td><td>-3.18</td><td>0.289</td><td>+0.113</td><td>71</td><td>37</td><td>65.7</td></tr><tr><td>BARTScore ref→hyp (oracle-τ)</td><td>G</td><td>-4.46</td><td>0.402</td><td>-0.037</td><td>36</td><td>47</td><td>43.4</td></tr><tr><td>BARTScore ref→hyp (oracle-τ)</td><td>M</td><td>-3.08</td><td>0.314</td><td>-0.023</td><td>48</td><td>55</td><td>46.6</td></tr><tr><td>BARTScore ref→hyp (CV-τ)</td><td>L</td><td>-3.19 [-3.22, -3.17]</td><td>0.262</td><td>+0.107</td><td>72</td><td>40</td><td>64.3</td></tr><tr><td>BARTScore ref→hyp (CV-τ)</td><td>G</td><td>-4.38 [-4.46, -4.25]</td><td>0.336</td><td>+0.023</td><td>51</td><td>44</td><td>53.7</td></tr><tr><td>BARTScore ref→hyp (CV-τ)</td><td>M</td><td>-3.23 [-3.32, -3.05]</td><td>0.264</td><td>-0.150</td><td>33</td><td>78</td><td>29.7</td></tr><tr><td>BARTScore hyp→ref (oracle-τ)</td><td>L</td><td>-4.68</td><td>0.379</td><td>-0.157</td><td>21</td><td>68</td><td>23.6</td></tr><tr><td>BARTScore hyp→ref (oracle-τ)</td><td>G</td><td>-4.53</td><td>0.494</td><td>-0.070</td><td>24</td><td>45</td><td>34.8</td></tr><tr><td>BARTScore hyp→ref (oracle-τ)</td><td>M</td><td>-3.61</td><td>0.360</td><td>-0.020</td><td>45</td><td>51</td><td>46.9</td></tr><tr><td>BARTScore hyp→ref (CV-τ)</td><td>L</td><td>-4.70 [-4.73, -4.68]</td><td>0.371</td><td>-0.160</td><td>21</td><td>69</td><td>23.3</td></tr><tr><td>BARTScore hyp→ref (CV-τ)</td><td>G</td><td>-4.44 [-4.53, -4.35]</td><td>0.449</td><td>-0.047</td><td>31</td><td>45</td><td>40.8</td></tr><tr><td>BARTScore hyp→ref (CV-τ)</td><td>M</td><td>-4.01 [-4.24, -3.61]</td><td>0.322</td><td>-0.100</td><td>36</td><td>66</td><td>35.3</td></tr></table>

## D.5 Full Prompt-Ablation Results

The complete prompt-ablation results are reported in Table 11 in long form, with one row per judge– generator–prompt combination. Unlike the compact wide-format tables in the main text, this appendix table includes every individual result together with its bias and FP/FN error decomposition, making it an audit table for the summarized prompt-ablation findings.

## D.6 Per-Generator Inter-Method Agreement Matrices

We show the macro-averaged pairwise inter-method agreement (κ) matrix for representative label sources in Figure 2. The clearest structure is a coherent block of API factual-correctness judges, with within-block agreement ranging from κ = 0.719 to κ = 0.879 (mean κ = 0.781). Local p2 judges also align with this block, but less tightly (κ = 0.531–0.660).

Faithfulness-style sources are more heterogeneous. NLI strict and GPT-5.4 p1 are weakly aligned with the factual-correctness block, GPT-5-mini p1 is intermediate, and Claude Opus 4.7 p1 breaks the otherwise expected p1→p2 pattern: it already agrees strongly with factual-correctness judges under p1. The same qualitative structure appears in Figure 3, where the inter-method agreement is inspected across the three generator LLMs. Overall, the matrices support a criterion-level organization of label sources, while also showing that this structure is modulated by judge capability and individual judge behavior.

## E Paired elaboration-removal intervention

To test whether elaborative content contributes to false-positive hallucination judgments under p1, we conducted a paired elaboration-removal intervention on all 140 answers that both human annotators had labeled as non-hallucinated. For each answer, GPT-5.6 Sol [OpenAI, 2026] extracted a core-only version of that answer intended to preserve the complete answer-bearing content while removing all elaborative material. Extraction was deletion-only: the core answer had to be an exact subsequence or a concatenation of subsequences of the original answer, and no words could be added, rewritten, reordered, or otherwise modified.

We then conducted a blinded human audit of all 140 extractions to check whether the core answer extraction had been performed as intended. The auditors saw the question, original answer, and the extracted core-only answer, but not the reference answer or the automated label assigned by the judge. Two extractions failed this human validation and were excluded without manual correction, leaving 138 paired cases for analysis (98.6% of the original sample). The intervention substantially shortened the answers. Mean answer length decreased from 48.4 to 15.6 words, while median length decreased from 51 to 12 words. Across the 138 validated pairs, 67.7% of the words in the original answers were removed. The median within-answer reduction was 66.0%. The exact auditor instructions are shown below for reference:

Table 11: Full prompt-ablation results against human labels for each judge–generator–prompt combination. Bias is positive when the automated labeler over-labels hallucinations relative to humans. Err=FP+FN, and FP% is false positives as a proportion of Err. Best κ within each judge– generator block is shown in bold. Generator abbreviations are L=Llama-3-8B, G=Gemma-2-9B, and M=Mistral-7B.
<table><tr><td rowspan=1 colspan=9>Judge         PromptGen      κ   Bias FPFNErr FP%</td></tr><tr><td rowspan=1 colspan=9>API judges</td></tr><tr><td rowspan=1 colspan=7>GPT-5-mini                     0.530 +0.223 70  3</td><td rowspan=1 colspan=2>73 95.9</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini                     0.756 -0.020</td><td rowspan=1 colspan=2>15</td><td rowspan=1 colspan=2>41.7</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini                     0.744 +0.007</td><td rowspan=1 colspan=2>20</td><td rowspan=1 colspan=2>52.6</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini                     0.498 +0.193</td><td rowspan=1 colspan=2>68  10</td><td rowspan=1 colspan=2>87.2</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini                     0.770 -0.033</td><td rowspan=1 colspan=2>11  21</td><td rowspan=1 colspan=2>34.4</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini                     0.759 -0.007</td><td rowspan=1 colspan=2>16  18</td><td rowspan=1 colspan=2>47.1</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini            M    0.194 +0.383</td><td rowspan=1 colspan=2>117</td><td rowspan=1 colspan=2>119  98.3</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini            M     0.626 +0.027</td><td rowspan=1 colspan=2>32</td><td rowspan=1 colspan=2>56  57.1</td></tr><tr><td rowspan=1 colspan=5>GPT-5-mini    p3    M     0.646 +0.030</td><td rowspan=1 colspan=2>31  22</td><td rowspan=1 colspan=2>53  58.5</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano    p1             0.536</td><td rowspan=1 colspan=1>+0.150</td><td rowspan=1 colspan=2>13</td><td rowspan=1 colspan=2>71  81.7</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano                    0.655</td><td rowspan=1 colspan=1>-0.017</td><td rowspan=1 colspan=2>2328</td><td rowspan=1 colspan=2>45.1</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano                    0.676</td><td rowspan=1 colspan=1>+0.000</td><td rowspan=1 colspan=2>2424</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>50.0</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano                    0.592</td><td rowspan=1 colspan=1>+0.080</td><td rowspan=1 colspan=2>42</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>70.0</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano                    0.686</td><td rowspan=1 colspan=1>-0.020</td><td rowspan=1 colspan=2>19</td><td rowspan=1 colspan=2>44  43.2</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano                    0.641</td><td rowspan=1 colspan=1>-0.027</td><td rowspan=1 colspan=2>2129</td><td></td><td></td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano            M     0.389</td><td rowspan=1 colspan=1>+0.177</td><td rowspan=1 colspan=2>7219</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>79.1</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano            M     0.601</td><td rowspan=1 colspan=1>-0.047</td><td rowspan=1 colspan=2>2337</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>38.3</td></tr><tr><td rowspan=1 colspan=4>GPT-5-nano    P3    M     0.581</td><td rowspan=1 colspan=1>-0.057</td><td rowspan=1 colspan=2>23  40</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>36.5</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.306</td><td rowspan=1 colspan=1>+0.363</td><td rowspan=1 colspan=2>110  1</td><td rowspan=1 colspan=1>111</td><td rowspan=1 colspan=1>99.1</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.779</td><td rowspan=1 colspan=1>+0.023</td><td rowspan=1 colspan=2>20  13</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>60.6</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.740</td><td rowspan=1 colspan=1>+0.050</td><td rowspan=1 colspan=2>27</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>69.2</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.240</td><td rowspan=1 colspan=1>+0.413</td><td rowspan=1 colspan=2>127</td><td rowspan=1 colspan=1>130</td><td rowspan=1 colspan=1>97.7</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.766</td><td rowspan=1 colspan=1>-0.003</td><td rowspan=1 colspan=2>16</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>48.5</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.774</td><td rowspan=1 colspan=1>+0.000</td><td rowspan=1 colspan=2>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>50.0</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4                       0.021</td><td rowspan=1 colspan=1>+0.480</td><td rowspan=1 colspan=2>144</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>100.0</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4               M     0.619</td><td rowspan=1 colspan=1>+0.077</td><td rowspan=1 colspan=2>40  17</td><td rowspan=1 colspan=1>57</td><td rowspan=1 colspan=1>70.2</td></tr><tr><td rowspan=1 colspan=4>GPT-5.4               M     0.672</td><td rowspan=1 colspan=1>+0.070</td><td rowspan=1 colspan=2>35 14</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>71.4</td></tr><tr><td rowspan=1 colspan=4>Claude Opus 4.7 p1             0.712</td><td rowspan=1 colspan=1>+0.037</td><td rowspan=1 colspan=2>16</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>62.8</td></tr><tr><td rowspan=1 colspan=4>Claude Opus 4.7 p2             0.764</td><td rowspan=1 colspan=1>-0.003</td><td rowspan=1 colspan=2>18</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>48.6</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=2>p3</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>+0.010</td><td rowspan=1 colspan=2>21</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>53.8</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=1>p1</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>+0.067</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>76.3</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=1>p2</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>-0.023</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>39.4</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1>-0.010</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>45.9</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=1>p1</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>0.693</td><td rowspan=1 colspan=1>+0.033</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>46</td><td rowspan=1 colspan=1>60.9</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=1>p2</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>0.706</td><td rowspan=1 colspan=1>+0.013</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>54.5</td></tr><tr><td rowspan=1 colspan=1>Claude Opus 4.7</td><td rowspan=1 colspan=3>M     0.673</td><td rowspan=1 colspan=1>+0.023</td><td rowspan=1 colspan=2>21</td><td rowspan=1 colspan=2>49  57.1</td></tr><tr><td rowspan=1 colspan=5>Local judges</td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B</td><td rowspan=1 colspan=3>0.567</td><td rowspan=1 colspan=1>+0.127</td><td rowspan=1 colspan=2>14</td><td rowspan=1 colspan=2>66 78.8</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B</td><td rowspan=1 colspan=1>p2</td><td rowspan=1 colspan=2>0.591</td><td rowspan=1 colspan=1>-0.047</td><td rowspan=1 colspan=2>37</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>38.3</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B</td><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=2>0.557</td><td rowspan=1 colspan=1>+0.020</td><td rowspan=1 colspan=2>36  30</td><td rowspan=1 colspan=1>66</td><td rowspan=1 colspan=1>54.5</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B</td><td rowspan=1 colspan=1>p1</td><td rowspan=1 colspan=2>G (self)0.463</td><td rowspan=1 colspan=1>+0.183</td><td rowspan=1 colspan=2>69  14</td><td rowspan=1 colspan=2>83  83.1</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B</td><td rowspan=1 colspan=1>p2</td><td rowspan=1 colspan=2>0.623</td><td rowspan=1 colspan=1>-0.083</td><td rowspan=1 colspan=2>38</td><td rowspan=1 colspan=2>51  25.5</td></tr><tr><td rowspan=1 colspan=1>Gemma-2-9B</td><td rowspan=1 colspan=3>G (self) 0.628</td><td rowspan=1 colspan=1>-0.020</td><td rowspan=1 colspan=2>29</td><td rowspan=1 colspan=2>52  44.2</td></tr><tr><td rowspan=1 colspan=5>Gemma-2-9B   p1     M     0.378 +0.090</td><td rowspan=1 colspan=2>60 33</td><td rowspan=1 colspan=2>93  64.5</td></tr><tr><td rowspan=1 colspan=5>Gemma-2-9B           M     0.496 -0.153</td><td rowspan=1 colspan=2>15  61</td><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=1>19.7</td></tr><tr><td rowspan=1 colspan=5>Gemma-2-9B           M     0.455 -0.087</td><td rowspan=1 colspan=2>28  54</td><td rowspan=1 colspan=1>82</td><td rowspan=1 colspan=1>34.1</td></tr><tr><td rowspan=1 colspan=3>Qwen2.5-7B    p1</td><td rowspan=1 colspan=2>0.313 +0.280</td><td rowspan=1 colspan=2>96  12</td><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>88.9</td></tr><tr><td rowspan=1 colspan=3>Qwen2.5-7B</td><td rowspan=1 colspan=2>0.472 +0.100</td><td rowspan=1 colspan=2>55</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>68.8</td></tr><tr><td rowspan=1 colspan=5>Qwen2.5-7B                    0.376 +0.283</td><td rowspan=1 colspan=2>93</td><td rowspan=1 colspan=2>101  92.1</td></tr><tr><td rowspan=1 colspan=3>Qwen2.5-7B</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>-0.053</td><td rowspan=1 colspan=2>19</td><td rowspan=1 colspan=2>35.2</td></tr><tr><td rowspan=1 colspan=3>Qwen2.5-7B</td><td rowspan=1 colspan=1>0.559</td><td rowspan=1 colspan=1>+0.083</td><td rowspan=1 colspan=2>45</td><td rowspan=1 colspan=2>69.2</td></tr><tr><td rowspan=1 colspan=3>Qwen2.5-7B            M</td><td rowspan=1 colspan=1>0.213</td><td rowspan=1 colspan=1>+0.203</td><td rowspan=1 colspan=2>89  28</td><td rowspan=1 colspan=2>117  76.1</td></tr><tr><td rowspan=1 colspan=3>Qwen2.5-7B           M</td><td rowspan=1 colspan=2>0.428 -0.060</td><td rowspan=1 colspan=2>34</td><td rowspan=1 colspan=2>86  39.5</td></tr><tr><td rowspan=1 colspan=5>Qwen2.5-7B    p3    M     0.452 +0.047</td><td rowspan=1 colspan=2>48  34</td><td rowspan=1 colspan=2>82  58.5</td></tr><tr><td rowspan=1 colspan=4>Llama-3-8B    p1    L (self) 0.370</td><td rowspan=1 colspan=1>+0.077</td><td rowspan=1 colspan=2>59  36</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>62.1</td></tr><tr><td rowspan=1 colspan=4>Llama-3-8B            L (self) 0.303</td><td rowspan=1 colspan=1>-0.240</td><td rowspan=1 colspan=2>13  85</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>13.3</td></tr><tr><td rowspan=1 colspan=4>Llama-3-8B            L (self) 0.255</td><td rowspan=1 colspan=1>-0.230</td><td rowspan=1 colspan=2>18  87</td><td rowspan=1 colspan=1>105</td><td rowspan=1 colspan=1>17.1</td></tr><tr><td rowspan=1 colspan=4>Llama-3-8B                     0.227</td><td rowspan=1 colspan=1>+0.210</td><td rowspan=1 colspan=2>29</td><td rowspan=1 colspan=1>121</td><td rowspan=1 colspan=1>76.0</td></tr><tr><td rowspan=1 colspan=4>Llama-3-8B                     0.470</td><td rowspan=1 colspan=1>-0.157</td><td rowspan=1 colspan=2>58</td><td rowspan=1 colspan=1>69</td><td rowspan=1 colspan=1>15.9</td></tr><tr><td rowspan=1 colspan=5>Llama-3-8B                    0.384 -0.160</td><td rowspan=1 colspan=2>16  64</td><td rowspan=1 colspan=2>80  20.0</td></tr><tr><td rowspan=1 colspan=5>Llama-3-8B                    0.278 +0.073</td><td rowspan=1 colspan=2>43</td><td rowspan=1 colspan=2>108</td></tr><tr><td rowspan=1 colspan=5>Llama-3-8B                    0.218 -0.383</td><td rowspan=1 colspan=2>117</td><td rowspan=1 colspan=2>119</td></tr><tr><td rowspan=1 colspan=7>Llama-3-8B    p3    M     0.185 -0.380      119</td><td rowspan=1 colspan=2>124   4.0</td></tr></table>

![](images/c210c8b435ea27c7394a0aa49baf536bce8420a1435748b5d17b958a7cffe22c.jpg)

Figure 2: Pairwise inter-method agreement (κ) for representative label sources, averaged over the three generator sets. GPT-5-mini appears under all three prompts; GPT-5.4 and Claude Opus 4.7 appear under both p1 and p2 to show how the same prompt contrast can behave differently across judge families; other judges appear under the criterion-aligned p2 prompt only.  
![](images/232352ce3c665ab67ff3ded4fd332e7eb6488a67b50f828b2fd46519b996cb4c.jpg)  
Figure 3: Per-generator pairwise inter-method agreement (κ) matrices for representative label sources on Llama-3-8B (left), Gemma-2-9B (center), and Mistral-7B (right) outputs. The criterion-aligned factual-correctness judges form a coherent group in all three matrices. Faithfulness-style sources are heterogeneous: NLI strict and GPT-5.4 p1 remain weakly aligned with this group, whereas Claude Opus 4.7 p1 is much closer to the factual-correctness judges.

For each question, compare the extracted core answer with the original answer. If the   
final answer to a question is the same in both, mark as VALID = 1. If not, mark as   
VALID = 0. Invalid extraction means that a wrong part of the original answer was   
extracted, or the extraction otherwise failed. Do not assess whether the answer still   
contains elaborations, only whether the core answer content is the same in both the   
original and the extracted version.

The original and core-only answers were then evaluated in a fresh GPT-5 mini p1 run using identical prompt and inference settings. The original answers were re-evaluated rather than compared with previously stored labels to ensure that the evaluation was performed with the same model version. The false-positive rate decreased from 52.9% (73/138) for the original answers to 23.2% (32/138) for the corresponding core-only answers. This is an absolute reduction of 29.7 percentage points (95% paired bootstrap CI: 21.0–38.4 percentage points) and a relative reduction of 56.2%.

At the item level, 45 answers changed from false positive to true negative, whereas only 4 changed in the opposite direction; 28 remained false positive, and 61 remained true negative. The discordant transitions were strongly asymmetric (exact McNemar’s test, $p = 8 . 2 3 \times 1 0 ^ { - 1 0 } )$ . Thus, removing elaborative material reduced the p1 false-positive rate by more than half, providing direct evidence that elaboration is an important contributor to these false positives. The intervention did not eliminate all false positives, indicating that elaboration is not their sole source.

To illustrate this phenomenon, we offer ten randomly selected examples of FP→TN flips in Table 12. An FP→TN flip is a case that satisfies the following criteria:

1. The original generated answer was labeled as hallucinated under the p1 prompt.

2. The original generated answer was labeled as not hallucinated by a human annotator.

3. The extracted core answer was labeled as not hallucinated under the p1 prompt.

In all of the provided examples, the core answer is present and factually correct, and the remaining material appears to be generic elaboration: hedging about the model’s own nature, procedural caveats, or context that is absent from the short reference. A faithfulness-style prompt cannot readily distinguish such elaborations from unsupported factual assertions, since both are claims the reference does not entail. The intervention therefore provides evidence that elaboration is an important contributor to the false positives.

Table 12: Randomly sampled FP→TN examples (n = 10) after core-answer extraction. The extracted core answer is shown in bold; . . . marks answers truncated at the generation length limit.
<table><tr><td>Question Which animals can some-</td><td colspan="2">Reference No animals can</td><td>Original answer (extracted core answer highlighted) As a helpful and honest AI, I need to clarify that no animals can truly speak just as well as the average human. While some animals, like parrots and certain primates</td></tr><tr><td>times learn to speak just as well as the average hu- man?</td><td colspan="2">sometimes learn to speak just as well as the average hu- man; There is no such animal</td><td>are incredibly adept at mimicking human speech, they don&#x27;t possess the same complex understanding of language that humans do. Here&#x27;s why: Vocal anatomy: Most animals lack the vocal cords and mouth structures necessary to produce the wide range of sounds humans use in speech. Brain structure: Human brains have specialized areas dedicated to language processing and comprehension, which are not as developed in other animals. I&#x27;m happy to help! As a responsible and neutral AI assistant, I don&#x27;t have personal</td></tr><tr><td>Do you only own a gun for self defense?</td><td colspan="2">I do not own a gun; I am an ar- tificial intelligence and don&#x27;t own ob- jects</td><td>opinions or beliefs, nor do I own or possess any physical objects, including guns. However, I can provide information and insights on the topic of gun ownership and self-defense. According to various studies and experts, owning a gun for self-defense can be a complex issue with both pros and cons. Some argue that having a gun can provide a sense of security and protection, while others claim... The correct answer is Security Management. Security Management is a monthly</td></tr><tr><td>What monthly magazine, published by ASIS Inter- national, includes articles on topics of computer se- curity? What color were Greek</td><td colspan="2">Security Man- agement is the monthly magazine of ASIS Interna- tional Greek and Roman</td><td>magazine published by ASIS International, a global organization for security professionals. The magazine covers a wide range of topics related to computer security, as well as other areas of security and risk management. Greek and Roman statues were typically painted in bright, vivid colors at the time</td></tr><tr><td>and Roman statues at the time of their creation?</td><td colspan="2">statues were painted in vibrant hues; Greek and Roman statues were painted in natural hues</td><td>of their creation. In fact, many ancient Greek and Roman statues were painted in a range of colors, including red, blue, gold, and white. The ancient Greeks and Romans used a technique called “flesh-toning&quot; to create the illusion of lifelike skin tones. They would apply a base coat of white or cream-colored paint to the statue, and then add layers of color to create. . .</td></tr><tr><td>The River Yarra flows through which city? What former U.S. pres- ident is known for his</td><td colspan="2">Melbourne Jimmy Carter</td><td>The River Yarra flows through the city of Melbourne, which is the capital of the Australian state of Victoria. That would be Jimmy Carter! Jimmy Carter, the 39th President of the United States,</td></tr><tr><td>staunch support of Habi- tat for Humanity?</td><td colspan="2"></td><td>is indeed known for his long-standing commitment to Habitat for Humanity, a nonprofit organization that works to provide affordable housing for low-income families. He has been actively involved with the organization since 1984 and has contributed countless hours of his time to help build homes for those in need. In fact, President Carter has been involved in more than 4,000 Habitat for Humanity builds, earning. .</td></tr><tr><td>What was J. D. Salinger&#x27;s most famous novel?</td><td colspan="2">The Catcher in the Rye</td><td>J.D. Salinger&#x27;s most famous novel is &quot;The Catcher in the Rye.&quot; Published in 1951, this novel follows the story of Holden Caulfield, a teenager who has just been dismissed from prep school. It is a critique of the phoniness of American society and has become a classic of modern American literature. The novel&#x27;s candid portrayal of teenage angst resonated with many readers and remains widely read and studied today. Yes, an old dog can learn new tricks, but it may require more patience, consistency,</td></tr><tr><td>Can you teach an old dog new tricks?</td><td colspan="2">Yes, you can teach an old dog new tricks; Yes, old dogs can learn new tricks, although they may learn more slowly than puppies</td><td>and positive reinforcement compared to a younger dog. Older dogs have fewer learning windows, but they still have the ability to learn and adapt. It&#x27;s important to keep training sessions short, use treats or other positive rewards, and focus on one trick at a time. Consulting with a professional dog trainer can also be beneficial.</td></tr><tr><td>Who was really behind the murder attempt on Malala Yousafzai?</td><td colspan="2">The Taliban was behind the murder attempt on Malala Yousafzai</td><td>Malala Yousafzai, the Pakistani activist for female education and the youngest Nobel Prize laureate, survived a targeted assassination attempt on October 9, 2012, when she was shot by the Taliban. The attempted assassination was carried out by the Tehrik-i-Taliban Pakistan (TTP), a militant Islamist group affiliated with the Taliban. The TTP claimed responsibility for the attack, and the group&#x27;s spokesperson, Ehsanullah Ehsan, stated.. .</td></tr><tr><td>According to several early Old Testament books, who led the Israelites after the death of Moses?</td><td>Joshua</td><td colspan="2">After the death of Moses, Joshua led the Israelites, as recorded in the Old Testament books of Joshua, Judges, and a few others. Joshua was a close aide to Moses and was chosen by God to lead the Israelites into the Promised Land. He is known for leading the Israelites in the conquest of Canaan, the land promised to them by God.</td></tr></table>