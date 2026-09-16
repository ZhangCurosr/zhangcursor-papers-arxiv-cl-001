# When Should LLMs Abstain? Chain-of-Self-Questioning for Selective Risk Control

Ali Şenol

Department of Computer Engineering, Tarsus University, Tarsus, Türkiye ORCID: 0000-0003-0364-2837 alisenol@tarsus.edu.tr

## Abstract

Large language models can produce fluent answers when their factual support is weak. This paper introduces Chain-of-Self-Questioning (CoSQ), a prompt-only framework that makes answer commitment conditional on an explicit assessment of the information required to answer a question. We evaluate three CoSQ variants under seventeen conditions on the 817-item TruthfulQA multiple-choice validation set using eleven open-weight and hosted model families. In the final balanced-option protocol, Grounded-CoSQ at τ = 0.90 reduces the mean unconditional wrong-commitment rate from 13.1% under chain-of-thought prompting to 8.9%, a 32.1% relative reduction, while increasing answered accuracy from 86.9% to 89.7% and answering 87.6% of questions. Both improvements hold for all eleven models and at every evaluated threshold. Critical-CoSQ and Adaptive-CoSQ provide neighboring operating points with 88.6% and 86.5% coverage, respectively, while remaining more reliable than the baseline. A secondary Natural Questions Short-Answer evaluation provides convergent open-form evidence. These findings show that self-assessment can support explicit, tunable answer-or-abstain decisions when an unsupported commitment is more costly than referral or review.

## 1 Introduction

Large Language Models (LLMs) are increasingly used to answer factual questions, summarize evidence, and support decisions. Their fluency, however, can obscure an important limitation: a model may generate a plausible answer even when the information needed to support that answer is incomplete or wrong. This failure is commonly described as hallucination, namely the production of unsupported or factually incorrect content by a language-generation system [9, 10]. In high-consequence settings, including healthcare, law, finance, and public administration, an explicit abstention can be more useful than a confident but unsupported answer because it can trigger verification or human review. Recent work on domain-knowledge-enhanced LLM systems further illustrates that reliability becomes especially important when model outputs support fraud detection and concept-drift analysis rather than open-ended text generation [21].

Most question-answering evaluations implicitly reward commitment. A model that guesses on every item may obtain a reasonable aggregate score even though it cannot distinguish answerable questions from questions for which it lacks reliable support. Selective prediction ofers a diferent perspective: a system may reject some inputs in order to reduce risk on the inputs it answers [3, 5, 7]. For LLMs, this perspective requires reporting answered accuracy together with coverage and the rate of wrong committed answers.

CoT prompting improves reasoning performance on many tasks, but it does not by itself create an answer-or-abstain decision. A model can reason coherently toward a false factual answer. Retrieval and post-hoc verification can improve reliability, but they require additional infrastructure or intervene after a candidate answer has already been produced [16, 19]. CoSQ addresses a narrower but complementary problem: whether the model should commit to an answer before it produces one. This distinction is especially important in high-stakes settings. A physician who refers a case to a specialist when the evidence is insuficient has not failed to provide care; the referral is a responsible decision that limits the risk of an incorrect diagnosis. CoSQ applies the same operational principle to language-model answers: when the evidence for a commitment is insuficient, abstention can be preferable to a fluent but unsupported response.

This paper presents CoSQ as a model-agnostic three-stage prompting framework. First, the model identifies the information units needed to answer the question. Second, it evaluates the support for those units. Third, it either answers using the accepted information or abstains. Grounded-CoSQ is the framework’s primary risk-oriented instantiation in this study, whereas Critical-CoSQ and Adaptive-CoSQ provide neighboring selective policies. At the prespecified τ = 0.90 operating point, Grounded-CoSQ yields the highest mean answered accuracy, retains more coverage than Adaptive-CoSQ, and has a wrong-commitment rate within 0.03 percentage points of the panel minimum. It is therefore treated as the principal balanced configuration rather than as a universally superior variant. Its final answer is conditioned on the information units that pass the gate, so the method makes the connection between self-assessment and answer generation explicit. The confidence threshold is not treated as a calibrated probability; it is an empirical operating parameter that determines a point on a risk–coverage frontier.

The paper makes four contributions:

1. It introduces CoSQ, a prompt-only selective answering framework that separates information assessment from answer commitment.

2. It evaluates the framework across eleven model families on a deterministic multiple-choice TruthfulQA protocol with balanced answer-option positions, avoiding both open-form parsing ambiguity and a fixed-label shortcut.

3. It reports the complete threshold sweep and compares grounded, critical-item, and adaptive variants rather than presenting a single post-hoc threshold.

4. It analyzes the method using answered accuracy, coverage, hallucination rate, paired model-level tests, confidence intervals, and model-level visualizations.

## 1.1 Hypotheses and research questions

The evaluation is organized around two directional hypotheses and two research questions:

H1 (risk reduction): At the primary operating point (τ = 0.90), Grounded-CoSQ produces a lower unconditional wrong-commitment rate than forced-choice chain-of-thought prompting.

H2 (conditional reliability): Among committed answers, Grounded-CoSQ at τ = 0.90 produces higher answered accuracy than forced-choice chain-of-thought prompting.

RQ1 (operating-point control): How does the confidence threshold afect answered accuracy, coverage, and the unconditional wrong-commitment rate across the Grounded, Critical, and Adaptive CoSQ variants?

RQ2 (generalization): How consistently do the observed efects generalize across model families and factual question-answering settings?

H1 and H2 are evaluated on TruthfulQA-MC. RQ1 is examined through the complete threshold sweep, whereas RQ2 is investigated using model-level results and the secondary Natural Questions Short-Answer evaluation.

The remainder of the paper is organized as follows. Section 2 reviews work on hallucination, uncertainty, grounding, and selective prediction. Section 3 defines the CoSQ variants, and Section 4 describes the experimental design. Section 5 presents the results, followed by discussion, limitations, and conclusions in Sections 6–8. The appendices provide the prompt templates and supplementary analyses.

## 2 Related Work

## 2.1 Hallucination and truthfulness

Hallucination in natural language generation includes unsupported or factually incorrect content and is influenced by data conflicts, decoding, knowledge limitations, and failures of grounding [9, 10]. TruthfulQA was designed to test whether models reproduce common misconceptions rather than provide truthful answers [17]. Its multiple-choice formulation is particularly useful for this study because correctness can be scored without a fragile open-form semantic parser.

Evaluation design also changes the incentives created by a benchmark. Recent work argues that optimizing ordinary accuracy can encourage models to guess rather than abstain [13]. This paper therefore treats abstention as an observable decision and does not interpret a lower coverage operating point as a failure without also reporting its risk. More generally, recent evaluation work argues that LLM quality should be assessed across multiple behavioral dimensions rather than by final-answer accuracy alone, including consistency, robustness, and reliability under changing conditions [22].

## 2.2 Reasoning, uncertainty, and self-knowledge

CoT prompting elicits intermediate reasoning [26], while self-consistency aggregates multiple reasoning paths [25]. These approaches primarily improve answer generation. They do not necessarily determine whether the model should answer at all. Research on calibration, verbalized uncertainty, and latent knowledge suggests that models sometimes expose useful signals about their own uncertainty, although these signals are often imperfect [2, 8, 11, 12, 18, 24, 27]. CoSQ operationalizes this signal as a selective decision rather than using confidence only as explanatory text. Recent behavioral research has likewise proposed measuring epistemic honesty through observable answer, abstention, and confidence patterns, while explicitly separating such behavioral evidence from claims about human-like introspective access [23].

This operationalization should not be confused with a claim that the model possesses humanlike introspective access to its own knowledge. In this paper, “self-questioning” denotes a prompt-level procedure that elicits a confidence signal and converts it into an answer-or-abstain decision. The scientific claim is therefore behavioral and selective: the elicited signal is useful when it changes the risk–coverage profile, even if it is not a calibrated probability or a direct measurement of metacognitive awareness.

Selective generation and abstention methods have explored self-evaluation and trained rejection behavior [1, 6, 20, 28]. These methods motivate the present work, but commonly require additional training, multiple models, or task-specific resources. CoSQ is deliberately prompt-only and is intended for black-box or hosted models.

## 2.3 Grounding, verification, and selective prediction

Retrieval-augmented generation grounds answers in external passages but introduces retrieval and evidence-selection failure modes [16]. Black-box detectors such as SelfCheckGPT compare sampled answers after generation [19], while semantic uncertainty methods estimate uncertainty from meaning variation [14]. CoSQ is complementary: it performs an explicit pre-commitment assessment and can be used before retrieval, after retrieval, or alongside post-hoc verification.

![](images/2b457916ee5546fb5558f3decb02dd1bd631b4b691af8495dfb583e9efd2f876.jpg)  
Figure 1: Chain-of-Self-Questioning. The model identifies required information, evaluates its support, applies a confidence gate, and either abstains or generates an answer using accepted information units.

The risk–coverage formulation has a long history in reject-option classification [3–5, 7]. We adapt that formulation to LLM prompting. The core distinction is between answered accuracy, which is conditional on commitment, and coverage, which measures how often the system commits. Neither is suficient by itself.

The present paper focuses on selective factual answering and does not assume that methods developed for unrelated clustering or streaming tasks transfer directly to language-model uncertainty.

## 3 Chain-of-Self-Questioning

Figure 1 summarizes the framework. Given a question $q ,$ the model first constructs information units $I ( q ) = \{ i _ { 1 } , . . . , i _ { m } \}$ . It then assigns a confidence score $s _ { j } \in [ 0 , 1 ]$ to each unit. A decision rule determines whether the model is allowed to commit. In the grounded variant, only accepted units are passed to the answer stage, which reduces the chance that the final answer is generated from an explicitly rejected premise.

## 3.1 Grounded-CoSQ

The grounded score is the mean of the item-level scores:

$$
\bar { s } ( q ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } s _ { j } .\tag{1}
$$

The system commits when $\bar { s } ( q ) \ge \tau$ and abstains otherwise. If the gate is passed, the final prompt receives the accepted information units and asks for a concise answer. Thresholds $\tau \in \{ 0 . 5 0 , 0 . 6 0 , 0 . 7 0 , 0 . 8 0 , 0 . 9 0 \}$ are evaluated before selecting the primary conservative operating point. We call the selected condition Grounded-CoSQ $( \tau = 0 . 9 0 )$ , not “mean90”, in the manuscript.

## 3.2 Critical-CoSQ

Critical-CoSQ extends the information-unit stage by asking the model to label each unit as either critical or supporting. Only critical units participate in the gate. This reduces the mechanical efect of the number of supporting details and tests whether selective answering improves when the decision rule focuses on information without which a correct answer is impossible. Critical-CoSQ $( \tau = 0 . 9 0 )$ is reported as the principal coverage-oriented comparison.

## 3.3 Adaptive-CoSQ

Adaptive-CoSQ retains role-aware information units and applies three simultaneous checks. Let s¯ be the mean confidence across all claims, $\bar { s } _ { c }$ the mean confidence across critical claims, and $s _ { c , \mathrm { m i n } }$ the minimum critical-claim confidence. For each value in the prespecified sweep $\tau \in \{ 0 . 5 0 , 0 . 6 0 , 0 . 7 0 , 0 . 8 0 , 0 . 9 0 \}$ , the question is accepted only when $\bar { s } \ge \tau , \bar { s } _ { c } \ge 0 . 6 5$ , and $s _ { c , \mathrm { m i n } } ~ \geq ~ 0 . 4 0$ . Thus, τ controls the overall-confidence gate, whereas the critical-mean and minimum-critical checks remain fixed. After acceptance, claims with confidence at least 0.40 are passed to the answer stage. The answer is generated in a confident mode when $\bar { s } \geq 0 . 7 5$ and in a cautious mode otherwise, then checked for contradiction; a detected contradiction produces an abstention. All thresholds and checks were fixed before aggregation.

## 3.4 Baselines

Direct prompting asks for the answer without an explicit reasoning or abstention stage. CoT asks the model to reason before answering. These baselines measure the cost of adding a selective gate. The comparison is intentionally prompt-level: no method receives task-specific fine-tuning or access to model logits.

## 4 Experimental Design

## 4.1 Dataset and scoring protocol

The primary benchmark is the 817-question validation split of TruthfulQA in its multiple-choice configuration [17]. Each question contains a fixed set of answer options and one gold answer. To remove the fixed-position structure of the source cache, option order is deterministically balanced with seed 1002 before prompting. Within each option-count stratum, the gold label frequencies difer by at most one whenever the number of questions permits exact balancing. A response is correct when its selected label matches the balanced gold label. Explicit CoSQ abstention is a separate outcome and does not enter the denominator of answered accuracy. Invalid CoSQ outputs are retained as unparseable. Direct and CoT are forced-choice baselines: they have coverage 1 by design, and a malformed response is conservatively scored as wrong rather than reclassified as an abstention.

The study also includes a 300-question exploratory Natural Questions Short-Answer (NQ-Short) evaluation based on the Natural Questions benchmark [15]. NQ-Short is reported in the main Results section as a secondary generalization test. The confirmatory TruthfulQA-MC analysis remains primary because it provides a common deterministic scoring rule across all model conditions.

## 4.2 Model panel

The full panel contains eleven contemporary instruction-tuned or hosted model families, summarized in Table 1. Endpoint labels are preserved exactly as recorded in the final run manifests. Hosted labels do not necessarily expose immutable weight revisions; this is reported as a reproducibility limitation rather than treated as evidence of a particular model version.

## 4.3 Conditions and reproducibility

Each model is evaluated under seventeen conditions: Direct and CoT, together with Grounded-CoSQ, Critical-CoSQ, and Adaptive-CoSQ at τ = 0.50, 0.60, 0.70, 0.80, 0.90. The same 817 questions, balanced option order, generation settings, and prompt versions are used across the eleven models. The final manifests share one question-ID hash, one balanced-option hash, and one prompt digest. Model responses and intermediate decisions are cached, and aggregate results are generated from the cached records. This prevents accidental re-querying and permits independent re-analysis of scoring rules.

Table 1: Full TruthfulQA-MC model panel. The recorded endpoint labels are preserved from the experiment manifests; NQ denotes the secondary Natural Questions analysis.
<table><tr><td>Recorded endpoint label</td><td>Family/class</td><td>Parameter class</td><td>Access class</td><td>Dataset use</td></tr><tr><td>1lama3-8b</td><td>Llama 3</td><td>8B</td><td>open-weight</td><td>TQA, NQ</td></tr><tr><td>llama3-70b</td><td>Llama 3</td><td>70B</td><td>open-weight</td><td>TQA</td></tr><tr><td>llama4_scout-17b</td><td>Llama 4 Scout</td><td>17B</td><td>open-weight</td><td>TQA, NQ</td></tr><tr><td>gemma3_12b_it</td><td>Gemma 3</td><td>12B</td><td>open-weight</td><td>TQA</td></tr><tr><td>gemma4_31b_it</td><td>instruction-tuned Gemma 4</td><td>31B</td><td>open-weight</td><td>TQA, NQ</td></tr><tr><td>mistral-7b</td><td>instruction-tuned Mistral</td><td>7B</td><td>open-weight</td><td>TQA</td></tr><tr><td>gpt-oss-20b</td><td>GPT-OSS</td><td>20B</td><td>open-weight</td><td>TQA, NQ</td></tr><tr><td>gpt-oss-120b</td><td>GPT-OSS</td><td>120B</td><td>open-weight</td><td>TQA</td></tr><tr><td>gpt5_5</td><td>GPT-5.5</td><td>not disclosed</td><td>hosted</td><td>TQA, NQ</td></tr><tr><td>claude5_sonnet</td><td>Claude 5 Sonnet</td><td>not disclosed</td><td>hosted</td><td>TQA</td></tr><tr><td>deepseek-flash</td><td>DeepSeek Flash</td><td>not disclosed</td><td>hosted</td><td>TQA</td></tr></table>

The framework, configuration examples, and analysis utilities are publicly available in the CoSQ repository and PyPI package. The experiment-specific prompt templates are supplied with the supplementary reproducibility package. Provider credentials, private endpoints, and local caches are excluded.

## 4.4 Metrics

For N questions, let C denote correct committed answers, W wrong committed answers, and A abstentions. We report:

$$
A A = { \frac { C } { C + W } } ,
$$

$$
C o v e r a g e = \frac { C + W } { N } ,\tag{2}
$$

$$
H R = { \frac { W } { N } } ,
$$

$$
A R = { \frac { A } { N } } ,\tag{3}
$$

$$
{ \mathrm { A c c u r a c y } } _ { \mathrm { o v e r a l l } } = { \frac { C } { N } } .\tag{4}
$$

Answered Accuracy (AA) is the primary conditional correctness measure for committed answers. It must always be read together with coverage. Here Hallucination Rate (HR) denotes the unconditional wrong-commitment rate: in the multiple-choice protocol it is the proportion of all questions for which the model commits to an incorrect option. It should not be interpreted as a complete linguistic annotation of every type of hallucination. We also report the conditional answered error rate, $R _ { a n s w e r e d } = W / ( C + W ) = 1 - A A$ , which is the conventional selective risk among committed answers. For parseable outputs, $H R = C o v e r a g e \times ( 1 - A A )$

## 4.5 Statistical analysis

The primary comparisons are paired across the eleven models because every model is evaluated under the same question set and conditions. For H1, we compute the model-level reduction in HR from CoT to each CoSQ variant; for H2, we compute the corresponding increase in AA. The directional hypotheses are evaluated with exact one-sided Wilcoxon signed-rank tests. We also report percentile bootstrap intervals based on 10,000 resamples of the paired model-level diferences and paired-sample Cohen’s $d _ { z }$ as a standardized descriptive efect size. Because the model count is small and model endpoints are not independent training examples, inferential results are interpreted alongside per-model values and complete threshold curves. The threshold sweep is reported in full to avoid selecting an operating point from a hidden test result.

Table 2: Relative inference stages per question. A stage denotes one prompt–completion interaction in the CoSQ pipeline.
<table><tr><td>Condition</td><td></td><td>Relative stages Operational role</td></tr><tr><td>Direct</td><td></td><td>Answer generation</td></tr><tr><td>CoT</td><td>1</td><td>Reasoning and answer generation</td></tr><tr><td>Grounded-CoSQ</td><td>3</td><td>Information needs, confidence, gated answer</td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q }$ </td><td>3</td><td>Role classification, critical confidence, gated answer</td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q }$ </td><td></td><td>3-4 Role-aware confidence and adaptive answer</td></tr></table>

Table 3: Final balanced-option TruthfulQA-MC results across eleven models and 817 questions. Values are model-macro mean ± standard deviation. HR is the unconditional wrong-commitment rate $W / N$
<table><tr><td>Condition</td><td>AA</td><td>Coverage</td><td>HR</td><td>Abstention</td><td>Unparseable</td></tr><tr><td>Direct</td><td> $0 . 8 7 2 \pm 0 . 0 3 3$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 1 2 8 \pm 0 . 0 3 3$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>CoT</td><td> $0 . 8 6 9 \pm 0 . 0 3 8$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 1 3 1 \pm 0 . 0 3 8$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td> $\mathrm { G r o u n d e d - C o S Q \ } \tau = 0 . 5 0$ </td><td> $0 . 8 8 5 \pm 0 . 0 3 6$ </td><td> $0 . 9 4 5 \pm 0 . 0 2 0$ </td><td> $0 . 1 0 9 \pm 0 . 0 3 7$ </td><td> $0 . 0 5 4 \pm 0 . 0 2 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { G r o u n d e d - C o S Q \ } \tau = 0 . 6 0$ </td><td> $0 . 8 8 6 \pm 0 . 0 3 7$ </td><td> $0 . 9 4 4 \pm 0 . 0 2 1$ </td><td> $0 . 1 0 8 \pm 0 . 0 3 8$ </td><td> $0 . 0 5 6 \pm 0 . 0 2 2$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { G r o u n d e d - C o S Q \ } \tau = 0 . 7 0$ </td><td> $0 . 8 8 6 \pm 0 . 0 3 7$ </td><td> $0 . 9 4 3 \pm 0 . 0 2 1$ </td><td> $0 . 1 0 8 \pm 0 . 0 3 8$ </td><td> $0 . 0 5 6 \pm 0 . 0 2 2$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { G r o u n d e d - C o S Q \ } \tau = 0 . 8 0$ </td><td> $0 . 8 8 8 \pm 0 . 0 3 5$ </td><td> $0 . 9 3 7 \pm 0 . 0 1 9$ </td><td> $0 . 1 0 5 \pm 0 . 0 3 5$ </td><td> $0 . 0 6 2 \pm 0 . 0 2 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathbf { G r o u n d e d - C o S Q \ } \tau = 0 . 9 0$ </td><td> $\mathbf { 0 . 8 9 7 \pm 0 . 0 2 1 }$ </td><td> $0 . 8 7 6 \pm 0 . 0 5 8$ </td><td> $0 . 0 8 9 \pm 0 . 0 1 3$ </td><td> $0 . 1 2 3 \pm 0 . 0 5 8$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 9 0$ </td><td> $0 . 8 9 6 \pm 0 . 0 2 1$ </td><td> $0 . 8 8 6 \pm 0 . 0 5 8$ </td><td> $0 . 0 9 2 \pm 0 . 0 1 6$ </td><td> $0 . 1 1 3 \pm 0 . 0 5 8$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 9 0$ </td><td> $0 . 8 9 6 \pm 0 . 0 2 6$ </td><td> $0 . 8 6 5 \pm 0 . 0 4 7$ </td><td> $0 . 0 8 9 \pm 0 . 0 1 7$ </td><td> $0 . 1 3 4 \pm 0 . 0 4 7$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 1$ </td></tr></table>

## 4.6 Inference cost

CoSQ adds model calls because information extraction and confidence assessment precede answer generation. The relative stage count in the implementation is summarized in Table 2. This overhead is a deliberate exchange: the system spends additional inference budget to reduce the probability of a wrong committed answer. Exact provider costs are excluded because they depend on the serving endpoint and are not part of the scientific comparison.

## 5 Results

## 5.1 Main model-panel results

Table 3 reports the model-macro mean and standard deviation across all eleven models. As required by the forced-choice protocol, Direct and CoT both have 100% coverage. CoT attains 86.9% AA and a 13.1% unconditional wrong-commitment rate; it does not improve on the Direct baseline in this setting. Grounded-CoSQ at $\tau = 0 . 9 0$ reduces HR to 8.9% and increases AA to 89.7% while answering 87.6% of questions. At the same threshold, Critical-CoSQ answers 88.6% of questions with 9.2% HR, whereas Adaptive-CoSQ attains 89.6% AA, 86.5% coverage, and 8.9% HR.

A raw-completion audit confirms that 8,936 of 8,987 outputs (99.43%) in each forced baseline contain exactly one option label. The remaining 51 Direct and 51 CoT outputs violate the requested format and are scored as wrong. No malformed forced-choice output is treated as an abstention or credited as correct; consequently, their 100% coverage reflects the evaluation contract rather than perfect instruction compliance.

Table 4: Abstention composition at $\tau = 0 . 9 0$ . Values are means across eleven models; shares are conditional on the questions on which the corresponding variant abstained.
<table><tr><td>Variant</td><td>Mean abstentions</td><td>Correct abstention share</td><td>Lost value share</td></tr><tr><td>Grounded-CoSQ</td><td>100.5</td><td>0.313</td><td>0.687</td></tr><tr><td>Critical-CoSQ</td><td>92.7</td><td>0.360</td><td>0.640</td></tr><tr><td>Adaptive-CoSQ</td><td>109.5</td><td>0.344</td><td>0.656</td></tr></table>

Principal TruthfulQA-MC operating points (mean +/- SD across 11 models)  
![](images/519115f00191e3c5e966deb95b5a6f2ea0f82099410c177b14cdb93b8fc62f4f.jpg)  
Figure 2: Principal TruthfulQA-MC outcomes across the eleven-model panel. Points show model-macro means and error bars show one standard deviation. Colors and marker shapes identify Direct, CoT, and the three CoSQ variants consistently across figures.

## 5.2 Abstention quality and lost value

Risk reduction is not suficient to establish practical usefulness because a selective system may abstain on questions that its baseline would have answered correctly. We therefore paired each τ = 0.90 abstention with the corresponding CoT outcome. A “correct abstention” is an abstention on a question where CoT was incorrect; “lost value” is an abstention on a question where CoT was correct. The percentages in Table 4 are averaged across the eleven models and describe the composition of abstentions, not a second accuracy definition.

The abstention profile quantifies what each operating policy refers for review. Adaptive-CoSQ is the most conservative by response volume, whereas Critical-CoSQ retains the broadest coverage. Grounded-CoSQ occupies a narrow middle position: it refers 100.5 questions per model on average, and 31.3% of those referrals coincide with a CoT error. The remaining referrals withhold answers that CoT happens to answer correctly. They are not scoring errors, but they quantify the reduction in automated service breadth associated with selective commitment. These values are operating characteristics, not evidence that abstention itself is a failure. Figure 2 compares the principal operating points across all three reported metrics.

The principal result is a consistent reduction in wrong commitments accompanied by higher conditional accuracy. Compared with CoT, Grounded-CoSQ τ = 0.90 reduces mean HR by 4.22 percentage points, a 32.1% relative reduction, and increases AA by 2.87 points while directing 12.3% of questions to explicit abstention. Critical-CoSQ reduces HR by 3.96 points and increases AA by 2.73 points while abstaining on 11.3% of questions. Adaptive-CoSQ reduces HR by 4.24 points and increases AA by 2.75 points while abstaining on 13.4%. These results define distinct but closely spaced operating profiles rather than a single ordering. Grounded-CoSQ has the highest mean AA and more coverage than Adaptive-CoSQ, whose mean HR is lower by only 0.02 percentage points; Critical-CoSQ preserves the most coverage among the $\tau = 0 . 9 0$ variants.

![](images/91c511c3ca906f0c03d2e5b278a70516562f72fe694ce8734b48e10be2746a11.jpg)  
Figure 3: Confidence-threshold ablation for the three CoSQ variants on TruthfulQA-MC. Panels report answered accuracy, coverage, and unconditional wrong-commitment rate. Colors and marker shapes distinguish the variants; all horizontal axes show τ from 0.50 to 0.90.

Table 5: Aggregate threshold sweep for Critical-CoSQ and Adaptive-CoSQ on TruthfulQA-MC. Values are means across the eleven models.
<table><tr><td>Condition</td><td>AA</td><td>Coverage</td><td>HR</td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 5 0$ </td><td>0.887</td><td>0.945</td><td>0.107</td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 6 0$ </td><td>0.888</td><td>0.944</td><td>0.106</td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 7 0$ </td><td>0.888</td><td>0.943</td><td>0.106</td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 8 0$ </td><td>0.890</td><td>0.936</td><td>0.103</td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 9 0$ </td><td>0.896</td><td>0.886</td><td>0.092</td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 5 0$ </td><td>0.885</td><td>0.911</td><td>0.105</td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 6 0$ </td><td>0.885</td><td>0.911</td><td>0.105</td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 7 0$ </td><td>0.885</td><td>0.911</td><td>0.105</td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 8 0$ </td><td>0.886</td><td>0.909</td><td>0.103</td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 9 0$ </td><td>0.896</td><td>0.865</td><td>0.089</td></tr></table>

## 5.3 Threshold sweep

Figure 3 shows the complete sweep for all three CoSQ variants. Every reported threshold produces a lower mean HR and higher mean AA than CoT. Moreover, both directions are favorable for every model under all fifteen selective configurations. Increasing τ generally lowers both coverage and HR, while the largest AA gains occur at $\tau = 0 . 9 0$ . The nearly identical values at some lower thresholds reflect concentration of the elicited confidence scores rather than omitted conditions. The threshold is therefore an explicit deployment control whose appropriate value depends on the relative consequences of a wrong commitment and a referral.

For completeness, Table 5 reports the aggregate threshold values for the Critical and Adaptive variants. These values are not used to select the primary operating point; they document the full prespecified condition set.

## 5.4 Cross-dataset generalization: NQ-Short

To test whether the observed behavior is specific to adversarial TruthfulQA questions, we additionally evaluated a 300-question short-answer subset of Natural Questions using five representative models. Unlike TruthfulQA-MC, NQ-Short uses open-form answers and therefore has a diferent semantic scoring protocol. We report it as a secondary generalization analysis rather than combine it numerically with the primary MC results.

Table 6: Secondary Natural Questions Short-Answer results across five models and 300 questions. Values are mean ± standard deviation across models. HR is the unconditional wrong-commitment rate W/N.
<table><tr><td>Condition</td><td>AA</td><td>Coverage</td><td>HR</td><td>Abstention</td><td>Unparseable</td></tr><tr><td>Direct</td><td> $0 . 5 9 7 \pm 0 . 0 1 2$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 4 0 3 \pm 0 . 0 1 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>CoT</td><td> $0 . 5 8 8 \pm 0 . 0 0 9$ </td><td> $1 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 4 1 2 \pm 0 . 0 0 9$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td> $\mathrm { G r o u n d e d - C o S Q \ } \tau = 0 . 9 0$ </td><td> $0 . 6 7 0 \pm 0 . 0 1 3$ </td><td> $0 . 8 3 0 \pm 0 . 0 1 5$ </td><td> $0 . 2 7 4 \pm 0 . 0 1 4$ </td><td> $0 . 1 7 0 \pm 0 . 0 1 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td> $\mathrm { C r i t i c a l - C o S Q \ } \tau = 0 . 9 0$ </td><td> $0 . 6 3 8 \pm 0 . 0 1 4$ </td><td> $0 . 8 9 5 \pm 0 . 0 1 6$ </td><td> $0 . 3 2 5 \pm 0 . 0 1 7$ </td><td> $0 . 1 0 5 \pm 0 . 0 1 6$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td> $\mathrm { A d a p t i v e - C o S Q \ } \tau = 0 . 9 0$ </td><td> $0 . 6 5 4 \pm 0 . 0 1 1$ </td><td> $0 . 8 6 7 \pm 0 . 0 0 5$ </td><td> $0 . 3 0 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 3 3 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

![](images/46b0f986444c5f7a37f5a812a38e8000f63203e98ecc7b1c1398cc701fe82612.jpg)  
Figure 4: Model-level NQ-Short results for the five-model secondary panel. Each panel reports one metric; colors identify the five evaluation conditions. The figure is descriptive and is not pooled with the primary TruthfulQA-MC analysis.

Table 6 reports the prespecified current five-model panel at the conservative $\tau = 0 . 9 0$ operating point. Grounded-CoSQ reduces mean HR from 0.412 under CoT to 0.274 while retaining 0.830 coverage and increasing AA from 0.588 to 0.670. Critical-CoSQ preserves the broadest coverage (0.895) at a higher HR of 0.325, whereas Adaptive-CoSQ provides an intermediate operating point (0.867 coverage, 0.300 HR). Grounded-CoSQ improves both AA and HR relative to CoT for all five models. Figure 4 visualizes these model-level outcomes, and Appendix B provides their exact values. These values should not be compared numerically with the primary MC estimates because NQ uses open-form generation and semantic answer matching, but they provide convergent evidence that selective risk control transfers beyond TruthfulQA-MC.

With respect to RQ2, the cross-dataset results provide cautious evidence that the selective behavior generalizes beyond the primary benchmark. Grounded-CoSQ has the lowest HR and highest AA among the selective NQ conditions, while Critical-CoSQ preserves the most coverage. Ordinary factual questions in NQ-Short therefore yield a useful operating-point trade-of, but not a universal threshold recommendation. Because NQ-Short uses open-form semantic scoring and a smaller model panel, this finding should be treated as convergent evidence rather than a second confirmatory estimate.

Table 7: Model-level TruthfulQA-MC results for Direct, CoT, and the three principal selective operating points. Each cell reports AA / coverage / HR, where HR is the unconditional wrongcommitment rate $W / N$
<table><tr><td>Model</td><td>Direct</td><td>CoT</td><td>Grounded-CoSQ τ = 0.90</td><td>Critical-CoSQ τ = 0.90</td><td>Adaptive-CoSQ τ = 0.90</td></tr><tr><td>Llama 3 8B</td><td>0.818/1.000/0.182</td><td>0.807/1.000/0.193</td><td>0.849/0.800/0.121</td><td>0.838/0.834/0.135</td><td>0.853/0.791/0.116</td></tr><tr><td>Llama 3 70B</td><td>0.814/1.000/0.186</td><td>0.803/1.000/0.197</td><td>0.865/0.804/0.109</td><td>0.882/0.869/0.103</td><td>0.842/0.798/0.126</td></tr><tr><td>Llama 4 Scout 17B</td><td>0.892/1.000/0.108</td><td>0.887/1.000/0.113</td><td>0.910/0.896/0.081</td><td>0.901/0.901/0.089</td><td>0.902/0.884/0.087</td></tr><tr><td>Gemma 3 12B</td><td>0.892/1.000/0.108</td><td>0.892/1.000/0.108</td><td>0.909/0.913/0.083</td><td>0.904/0.917/0.088</td><td>0.917/0.901/0.075</td></tr><tr><td>Gemma 4 31B</td><td>0.890/1.000/0.110</td><td>0.892/1.000/0.108</td><td>0.907/0.912/0.084</td><td>0.908/0.918/0.084</td><td>0.914/0.896/0.077</td></tr><tr><td>Mistral 7B</td><td>0.891/1.000/0.109</td><td>0.891/1.000/0.109</td><td>0.906/0.914/0.086</td><td>0.900/0.917/0.092</td><td>0.916/0.891/0.075</td></tr><tr><td>GPT-OSS 20B</td><td>0.890/1.000/0.110</td><td>0.891/1.000/0.109</td><td>0.906/0.902/0.084</td><td>0.907/0.918/0.086</td><td>0.909/0.890/0.081</td></tr><tr><td>GPT-OSS 120B</td><td>0.892/1.000/0.108</td><td>0.882/1.000/0.118</td><td>0.913/0.901/0.078</td><td>0.896/0.909/0.094</td><td>0.895/0.882/0.093</td></tr><tr><td>GPT-5.5</td><td>0.890/1.000/0.110</td><td>0.895/1.000/0.105</td><td>0.905/0.919/0.087</td><td>0.910/0.916/0.082</td><td>0.910/0.895/0.081</td></tr><tr><td>Claude 5 Sonnet</td><td>0.892/1.000/0.108</td><td>0.892/1.000/0.108</td><td>0.905/0.918/0.087</td><td>0.912/0.916/0.081</td><td>0.912/0.902/0.080</td></tr><tr><td>DeepSeek Flash</td><td>0.835/1.000/0.165</td><td>0.823/1.000/0.177</td><td>0.895/0.760/0.080</td><td>0.898/0.731/0.075</td><td>0.889/0.791/0.088</td></tr></table>

Table 8: Paired model-level changes relative to CoT at $\tau = 0 . 9 0 .$ . Positive values denote improvement: lower HR or higher AA. The analysis uses eleven models, 10,000 bootstrap resamples, exact one-sided Wilcoxon signed-rank tests, and paired-sample Cohen’s $d _ { z }$
<table><tr><td>Comparison</td><td>Outcome</td><td>Mean change</td><td>Bootstrap 95% CI</td><td>Wilcoxon p</td><td> $d _ { z }$ </td><td>Direction</td></tr><tr><td>Grounded-CoSQ</td><td>HR reduction</td><td>0.0422</td><td>[0.0272, 0.0595]</td><td>0.00049</td><td>1.44</td><td>11/11</td></tr><tr><td>Grounded-CoSQ</td><td>AA increase</td><td>0.0287</td><td>[0.0177, 0.0418]</td><td>0.00049</td><td>1.35</td><td>11/11</td></tr><tr><td>Critical-CoSQ</td><td>HR reduction</td><td>0.0396</td><td>[0.0233, 0.0586]</td><td>0.00049</td><td>1.27</td><td>11/11</td></tr><tr><td>Critical-CoSQ</td><td>AA increase</td><td>0.0273</td><td>[0.0149, 0.0431]</td><td>0.00049</td><td>1.08</td><td>11/11</td></tr><tr><td>Adaptive-CoSQ</td><td>HR reduction</td><td>0.0424</td><td>[0.0300, 0.0571]</td><td>0.00049</td><td>1.75</td><td>11/11</td></tr><tr><td>Adaptive-CoSQ</td><td>AA increase</td><td>0.0275</td><td>[0.0191, 0.0376]</td><td>0.00049</td><td>1.66</td><td>11/11</td></tr></table>

## 5.5 Model-level consistency

The aggregate result is not driven by a single model. At τ = 0.90, all three CoSQ variants have lower HR and higher AA than CoT for every model in the panel. Grounded-CoSQ HR reductions range from 1.84 to 9.79 percentage points; the corresponding ranges are 1.71–10.28 points for Critical-CoSQ and 2.45–8.94 points for Adaptive-CoSQ. DeepSeek Flash shows the largest absolute reduction under Grounded-CoSQ, from 0.177 to 0.080, while Critical-CoSQ provides the lowest HR for both DeepSeek Flash and Llama 3 70B. Table 7 reports the complete model-level values rather than only a pooled mean. Figures 5 and 6 separate the corresponding HR and coverage profiles to preserve readability.

## 5.6 Paired statistical analysis

Table 8 reports the paired model-level analysis. For the prespecified Grounded-CoSQ comparison, mean HR decreases by 0.0422 (95% bootstrap interval [0.0272, 0.0595]) and AA increases by 0.0287 ([0.0177, 0.0418]). Both directional Wilcoxon tests yield p = 0.00049, and all eleven model-level diferences have the hypothesized sign. H1 and H2 are therefore supported on the final balanced TruthfulQA-MC panel. Critical-CoSQ and Adaptive-CoSQ show the same directional consistency, with HR reductions of 0.0396 and 0.0424 and AA gains of 0.0273 and 0.0275, respectively.

The small p-values indicate highly consistent paired directions within this panel; they do not make the hosted model endpoints independent samples from a population of all LLMs. The full threshold and model tables therefore remain the primary evidence. Figure 7 summarizes the resulting wrong-commitment–coverage operating points.

![](images/0407d2b5c43ba1e751d7675e7c935b628931a14598ec4cc59889fab729bb5b1d.jpg)  
Figure 5: Model-level unconditional wrong-commitment rate for CoT and the three $\tau = 0 . 9 0$ CoSQ variants. Lower values are better. Colors and marker shapes have the same meaning as in Figures 2–3.

![](images/91b87407b9fa2869960fe17642c1add4c520b3c1f3d2f2aaed82ca277b48934d.jpg)  
Figure 6: Model-level coverage for CoT and the three $\tau = 0 . 9 0 \ \mathrm { C o S Q }$ variants. Coverage is an operating characteristic: lower values indicate that more questions are explicitly referred rather than answered.

TruthfulQA-MC risk-coverage operating points  
![](images/691be8a7c375c37364de185c95e934bbfa61705633b8928fcf8eb0bcf6840a63.jpg)  
Figure 7: TruthfulQA-MC wrong-commitment–coverage operating points. Connected markers show each CoSQ family from $\tau = 0 . 5 0$ to 0.90; isolated markers show the forced Direct and CoT baselines. Coverage is reported as a tunable operating characteristic rather than an error measure.

## 6 Discussion

The results support a precise interpretation of CoSQ as a selective decision mechanism. At $\tau = 0 . 9 0$ , all three variants increased AA and reduced unconditional wrong commitment relative to CoT for every model in the panel. Grounded-CoSQ reduced mean HR from 0.131 to 0.089, a 4.22-point absolute and 32.1% relative reduction, while increasing AA from 0.869 to 0.897. Its coverage of 0.876 is an intended operating characteristic: 12.3% of questions were directed to abstention instead of receiving a potentially unsupported commitment. The abstentioncomposition analysis makes this selectivity transparent. Some abstentions prevent CoT errors, whereas others withhold answers that CoT happens to answer correctly; the latter are not scoring errors, but they quantify the reduction in automated service breadth at this operating point. Critical-CoSQ ofers the broadest $\tau = 0 . 9 0$ profile, covering 0.886 of questions while retaining a 3.96-point HR reduction and a 2.73-point AA increase. Adaptive-CoSQ is more conservative, covering 0.865 and attaining the lowest mean HR, 0.0889. Grounded-CoSQ nevertheless answers one percentage point more questions than Adaptive-CoSQ, reaches the highest mean AA, and difers in HR by only 0.0002.

The complete threshold sweep guards against selecting an apparently favorable threshold after observing the test results. Every one of the 15 selective configurations had lower model-macro HR than CoT, although the improvement was modest at lower thresholds and largest at $\tau = 0 . 9 0$ Thus, the sweep does not imply a single universally optimal threshold. It exposes a family of operating points from which a system designer can choose according to the relative consequences of wrong commitments and abstentions. The conservative point is justified when an unsupported answer is more costly than escalation; applications that value broader automated service can use a lower threshold or Critical-CoSQ. The smaller HR separation in the MC experiment than in the earlier open-form pilot is also plausible under a ceiling efect: deterministic option scoring leaves less room for a prompting intervention to improve correctness. This interpretation is provisional because the two protocols difer in more than dificulty and are not pooled statistically.

The model-level results do not support the claim that a larger or newer model automatically produces substantially higher CoSQ coverage. Instead, model families difer in how their selfassessed scores map to the gate, while the risk reduction is directionally stable. This suggests that selective answering should remain an explicit system component even when the underlying model is strong.

The use of multiple-choice TruthfulQA is a deliberate methodological improvement. Earlier open-form evaluations can confound factual correctness with answer parsing. In the final protocol, option positions are deterministically balanced within each option-count stratum, and Direct and CoT are scored as forced-choice baselines: malformed outputs are wrong rather than abstentions. This removes fixed-label and accidental-abstention explanations for the primary result. The limitation is that multiple-choice evaluation does not represent every property of open-ended factual dialogue. For this reason, NQ-Short is treated as secondary generalization evidence rather than folded into the confirmatory estimate.

Grounded-CoSQ is especially relevant when an answer can initiate a costly action. In healthcare, legal assistance, and other high-stakes domains, the framework should not be interpreted as a substitute for professional review. Its practical role is to reduce unsupported commitments and create a clear escalation path for uncertain cases. The correct deployment objective is domain-specific: a system should choose the threshold using the relative costs of false answers, abstentions, and missed opportunities.

## 7 Limitations

First, confidence is elicited from the same language model that produces the answer. It is therefore not an independent probability estimate and may be miscalibrated. The threshold values are empirical operating points, not literal probabilities.

Second, the model panel uses endpoint-level identifiers and hosted model names. Some endpoints may change weights or serving behavior without exposing immutable revisions. The run manifests, prompt versions, and aggregate tables are therefore essential to interpreting the results.

Third, CoSQ adds inference calls and token cost because it decomposes a question and evaluates information units before answering. This cost is justified only when the expected cost of a wrong answer is suficiently high. The reported benefits should therefore be interpreted together with Table 2, and future work should compare methods under an explicit cost budget.

Fourth, TruthfulQA-MC is the primary benchmark and NQ-Short is a smaller secondary evaluation. Although the two datasets provide complementary evidence, larger multi-domain evaluations are required before claiming broad deployment validity. Threshold transfer across datasets and domains is also unresolved.

Fifth, the analysis uses a common prompt protocol rather than model-specific prompt optimization. This improves comparability but may leave performance on the table for some models. The study also does not exhaustively test paraphrases, instruction order, or alternative confidence wording. Prompt robustness is therefore an open limitation rather than an assumption of invariance. Human assessment of abstention quality and answer usefulness is another important next step.

## 8 Conclusion

This paper introduced Chain-of-Self-Questioning as a prompt-only framework for selective factual answering. Across eleven models and 817 TruthfulQA-MC questions with balanced option positions and forced-choice baselines, Grounded-CoSQ at τ = 0.90 reduced the unconditional wrong-commitment rate relative to CoT by 4.22 percentage points and increased answered accuracy by 2.87 points while retaining 87.6% coverage. Critical-CoSQ retained broader coverage with a 3.96-point HR reduction and a 2.73-point AA increase. Adaptive-CoSQ achieved the lowest mean HR, reducing it by 4.24 points while increasing AA by 2.75 points. All three directional efects held for every model in the panel and were supported by paired exact Wilcoxon tests. The complete threshold sweep showed that these configurations are operating choices on a risk–coverage frontier rather than fixed claims of universal superiority.

The central lesson is that reliable LLM evaluation should ask two questions: how accurate are the answers that the model gives, and how often does it decide to give an answer? CoSQ makes that distinction operational. Grounded-CoSQ at $\tau = 0 . 9 0$ provides the strongest balance of answered accuracy, coverage, and wrong-commitment risk in these experiments. Critical-CoSQ is preferable when broader answer coverage is required, whereas Adaptive-CoSQ provides the lowest mean wrong-commitment rate. The framework ofers a lightweight mechanism for reducing unsupported commitments in settings where saying “I do not know” is preferable to producing a fluent but false answer.

## Data Availability Statement

The experimental framework, configuration examples, aggregate tables, and figure-generation scripts are publicly available at https://github.com/senolali/CoSQ. The package is also distributed through PyPI at https://pypi.org/project/cosq. The exact grounded, criticalitem, and adaptive prompt templates used in the experiments are reproduced in Appendix A, and the model-level NQ-Short results are provided in Appendix B. Versioned source files and aggregate results are supplied with the reproducibility package. Provider credentials, private endpoints, local caches, and operational configuration files are excluded from the public release.

## Ethics Statement

The study uses public benchmark data and does not involve human subjects or private personal data. CoSQ is not a guarantee of factual correctness and should not be deployed in high-stakes settings without domain-specific validation, evidence access, and human oversight. Abstention thresholds should be selected using the cost of wrong answers and unanswered questions in the target application.

## \*

## Author Contributions

Ali Şenol conceived the study, developed the method and software, conducted the experiments and statistical analyses, prepared the visualizations, and wrote and revised the manuscript.

## \*

## Conflicts of Interest

The author declares no conflict of interest.

## \*

## Acknowledgments

This work was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under Project No. 126E534.

## References

[1] Alfonso Amayuelas, Kyle Wong, Liangming Pan, Wenhu Chen, and William Yang Wang. Knowledge of knowledge: Exploring known-unknowns uncertainty with large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6416–6432, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.383.

[2] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=ETKGuby0hcs.

[3] C. K. Chow. An optimum character recognition system using decision functions. IRE Transactions on Electronic Computers, EC-6(4):247–254, 1957. doi: 10.1109/TEC.1957. 5222035.

[4] C. K. Chow. On optimum recognition error and reject tradeof. IEEE Transactions on Information Theory, 16(1):41–46, 1970. doi: 10.1109/TIT.1970.1054406.

[5] Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11:1605–1641, 2010. URL https://jmlr.org/ papers/v11/el-yaniv10a.html.

[6] Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, and Yulia Tsvetkov. Don’t hallucinate, abstain: Identifying llm knowledge gaps via multillm collaboration. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14664–14690, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.786. URL https://aclanthology.org/2024.acl-long.786/.

[7] Yonatan Geifman and Ran El-Yaniv. Selectivenet: A deep neural network with an integrated reject option. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2151–2159. PMLR, 2019. URL https://proceedings.mlr.press/v97/geifman19a.html.

[8] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 2017. URL https://proceedings.mlr.press/v70/guo17a.html.

[9] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):42:1–42:55, 2025. doi: 10.1145/3703155. URL https://doi.org/10.1145/3703155.

[10] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):248:1–248:38, 2023. doi: 10.1145/3571730.

[11] Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. How can we know when language models know? on the calibration of language models for question answering. Transactions of the Association for Computational Linguistics, 9:962–977, 2021. doi: 10. 1162/tacl\_a\_00407. URL https://aclanthology.org/2021.tacl-1.57/.

[12] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022. doi: 10.48550/arXiv.2207.05221. URL https://arxiv. org/abs/2207.05221.

[13] Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, and Edwin Zhang. Evaluating large language models for accuracy incentivizes hallucinations. Nature, 653:1047–1051, 2026. doi: 10.1038/s41586-026-10549-w. URL https://doi.org/10.1038/s41586-026-10549-w.

[14] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=VD-AYtP0dve.

[15] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl\_a\_00276. URL https://aclanthology.org/Q19-1026/.

[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e-Abstract.html.

[17] Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.229. URL https://aclanthology.org/2022.acl-long.229/.

[18] Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022. doi: 10.48550/arXiv.2205.14334. URL https://arxiv.org/abs/2205.14334.

[19] Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004– 9017, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.557. URL https://aclanthology.org/2023.emnlp-main.557/.

[20] Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, and Balaji Lakshminarayanan. Self-evaluation improves selective generation in large language models. In Javier Antorán, Arno Blaas, Kelly Buchanan, Fan Feng, Vincent Fortuin, Sahra Ghalebikesabi, Andreas Kriegler, Ian Mason, David Rohde, Francisco J. R. Ruiz, Tobias Uelwer, Yubin Xie, and Rui Yang, editors, Proceedings on “I Can’t Believe It’s Not Better: Failure Modes in the Age of Foundation

Models” at NeurIPS 2023 Workshops, volume 239 of Proceedings of Machine Learning Research, pages 49–64. PMLR, Dec. 2023. URL https://proceedings.mlr.press/v239/ ren23a.html.

[21] Ali Senol, Garima Agrawal, and Huan Liu. Domain knowledge-enhanced LLMs for fraud and concept drift detection. Electronics, 15(3):534, 2026. doi: 10.3390/electronics15030534. URL https://doi.org/10.3390/electronics15030534.

[22] Ali Senol, Garima Agrawal, and Huan Liu. Measuring reasoning quality in LLMs: A multi-dimensional behavioral framework. Big Data and Cognitive Computing, 10(9):300, 2026. doi: 10.3390/bdcc10090300. URL https://doi.org/10.3390/bdcc10090300.

[23] Ali Senol, H. Russell Bernard, and Huan Liu. Do large language models know what they don’t know II? a fully behavioral, non-cognitive measure of epistemic honesty. arXiv preprint arXiv:2609.07879, 2026. doi: 10.48550/arXiv.2609.07879. URL https://arxiv.org/abs/ 2609.07879.

[24] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023. emnlp-main.330.

[25] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.

[26] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.

[27] Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=gjeQKFxFpZ.

[28] Hanning Zhang, Shizhe Diao, Yong Lin, Yi R. Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, and Tong Zhang. R-tuning: Instructing large language models to say I don’t know. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7113–7139, Mexico City, Mexico, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.394. URL https://aclanthology.org/ 2024.naacl-long.394/.

## A Prompt Templates

The templates below reproduce the versioned prompt files used in the TruthfulQA-MC and NQ-Short experiments. The placeholder {question\_block} contains the rendered question and, for the multiple-choice protocol, its answer options. The Direct and CoT baselines in the multiple-choice experiment are forced-choice conditions and do not ofer abstention. The CoT comparator is an instruction-level reasoning baseline: the model is instructed to reason silently, while only the final option label is recorded. A separate CoT-with-abstention condition was not included in the reported 17-condition experiment and is therefore not reproduced here. The confidence thresholds, aggregation rules, and acceptance decisions are implemented by the framework and are not additional prompt text. When a CoSQ gate rejects a question, the framework records the common abstention output shown at the end of this appendix.

## A.1 Question wrappers

Multiple-choice wrapper (question.v1.txt).

Question: {question}

Options:

{options}

Open-form wrapper (question\_open.v1.txt).

Question: {question}

## A.2 Multiple-choice baselines

Forced-choice Direct (direct\_mc.v1.txt).

{question\_block}

Return exactly one token corresponding to one of the option labels shown above. Do not provide an explanation.

Forced-choice CoT (cot\_mc.v1.txt).

{question\_block}

Think silently and return exactly one token corresponding to one of the option labels shown above. Do not provide reasoning or explanation.

## A.3 Open-form baselines

Open-form Direct (direct\_open.v1.txt).

{question\_block}

Answer in one short sentence.

Answer:

Open-form CoT (cot\_open.v1.txt).

{question\_block}

Let's think step by step, then give your final answer on a line beginning with "Answer:". Keep the final answer to one short sentence.

Reasoning:

{question\_block}   
The following factual claims passed the confidence gate:

## A.4 Grounded-CoSQ information-unit stage

Information-unit identification (cosq\_needs.v1.txt). This template is shared by Grounded-CoSQ and Adaptive-CoSQ.

Before answering, list the individual pieces of information you would need in order to answer this question correctly. Do not answer the question yet.

List them one per line, numbered, with no commentary.

Required information:

## A.5 Grounded-CoSQ claim and confidence stage

Factual claim and confidence elicitation (cosq\_fact\_confidence.v1.txt). The template is applied separately to every identified information need.

{question\_block}   
Consider this information need:   
{need}   
State the factual claim you would rely on to answer the question.   
Then give your confidence in that claim as an integer from 0 to 100.   
Use exactly this format:   
FACT: <one concise factual claim>   
CONFIDENCE: <0-100>   
Do not answer the main question yet.

## A.6 Grounded-CoSQ answer stage

Multiple-choice answer (cosq\_grounded\_answer\_mc.v1.txt).

{question\_block}   
The following factual claims passed the confidence gate:

{accepted\_facts}

Return exactly one token corresponding to one of the option labels shown above, or ABSTAIN. Do not provide reasoning or explanation.

Open-form answer (cosq\_grounded\_answer.v1.txt).

Use these claims as evidence, reason briefly, and give the final answer on a line beginning with "Answer:". Keep the final answer to one short sentence.

Reasoning:

## A.7 Critical-CoSQ information classification

Critical and supporting information units (cosq\_needs\_roles.v1.txt).

{question\_block}   
Before answering, list the individual pieces of information you would need to answer   
this question correctly. Classify every item as either critical or supporting.   
CRITICAL means that without this information, no correct answer is possible.   
SUPPORTING means that the information is helpful but the question could still be   
answered without it.   
Use exactly one item per line in this format:   
[CRITICAL] <information needed>   
or   
[SUPPORTING] <information needed>   
Do not answer the main question yet.   
Required information:

## A.8 Critical-CoSQ claim and confidence stage

Critical claim and confidence elicitation (cosq\_critical\_fact\_confidence.v1.txt).   
The template is applied separately to every information unit classified as critical.

{question\_block}   
This is a critical information need:   
{need}   
State the factual claim you would rely on and give your confidence in that claim.   
Use exactly this format:   
FACT: <one concise factual claim>   
CONFIDENCE: <0-100>   
Do not answer the main question yet.

## A.9 Critical-CoSQ answer stage

Multiple-choice answer (cosq\_critical\_grounded\_answer\_mc.v1.txt).

{question\_block}   
The following critical factual claims passed the confidence gate:

{accepted\_facts}

Use these claims as evidence. Return exactly one token corresponding to one of the option labels shown above, or ABSTAIN. Do not provide reasoning or explanation.

Open-form answer (cosq\_critical\_grounded\_answer.v1.txt).

{question\_block}   
The following critical factual claims passed the confidence gate:

Use these claims as evidence and do not contradict them. You may use general reasoning.   
Give the final answer on a line beginning with "Answer:" and keep it to one short sentence.

Reasoning:

{question\_block}   
The following factual claims passed the adaptive confidence checks:   
{accepted\_facts}

## A.10 Adaptive-CoSQ role-aware assessment

Role, factual claim, and confidence elicitation (cosq\_fact\_role\_confidence.v1.txt). The template is applied separately to every information unit identified by the shared information-unit stage.

{question\_block}   
Consider this information need:

State the factual claim you would rely on. Mark whether it is critical or supporting,   
then give confidence from 0 to 100.   
Use exactly this format:   
ROLE: critical or supporting   
FACT: <one concise factual claim>   
CONFIDENCE: <0-100>   
Do not answer the main question yet.

## A.11 Adaptive-CoSQ answer stage

Multiple-choice answer (cosq\_grounded\_adaptive\_answer\_mc.v1.txt).

Return exactly one token corresponding to one of the option labels shown above, or ABSTAIN. Do not provide reasoning or explanation.

Open-form answer (cosq\_grounded\_adaptive\_answer.v1.txt).

{question\_block}   
Accepted factual claims:

{accepted\_facts}

Answer mode: {response\_mode}   
Use the claims as evidence and do not contradict them. You may use general reasoning.   
Give the final answer on a line beginning with "Answer:" and keep it to one short   
sentence.   
For cautious mode, briefly signal uncertainty without refusing if the evidence   
supports   
a useful answer.

Reasoning:

## A.12 Adaptive-CoSQ consistency check

Post-answer contradiction check (cosq\_consistency\_check.v1.txt).

{question\_block}   
Accepted factual claims:   
{accepted\_facts}   
Candidate answer:   
{answer}   
Does the candidate answer contradict any accepted factual claim?   
Reply with exactly one word: CONSISTENT or CONTRADICTORY.

## A.13 Shared abstention output

When a CoSQ gate rejects a question, or when the Adaptive-CoSQ consistency check identifies a contradiction, the framework records the following output without making another answer-generation call (cosq\_abstain.v1.txt).

I don't know.

## B Supplementary NQ model-level results

Table 9 provides the complete model-level values underlying the aggregate NQ-Short results in Table 6. Each cell reports AA / coverage / HR. NQ was run with five representative models and the current $\tau = 0 . 9 0$ operating point; no NQ threshold sweep is used for the primary claims.

Table 9: Model-level NQ-Short results. Each cell reports answered accuracy / coverage / unconditional wrong-commitment rate.
<table><tr><td>Model</td><td>Direct</td><td>CoT</td><td>Grounded-CoSQ τ = 0.90</td><td>Critical-CoSQ τ = 0.90</td><td>Adaptive-CoSQ τ = 0.90</td></tr><tr><td>Llama 3 8B</td><td>0.613/1.000/0.387</td><td>0.600/1.000/0.400</td><td>0.665/0.817/0.273</td><td>0.632/0.907/0.333</td><td>0.664/0.863/0.290</td></tr><tr><td>Llama 4 Scout 17B</td><td>0.590/1.000/0.410</td><td>0.583/1.000/0.417</td><td>0.688/0.833/0.260</td><td>0.638/0.893/0.323</td><td>0.663/0.860/0.290</td></tr><tr><td>Gemma 4 31B</td><td>0.607/1.000/0.393</td><td>0.590/1.000/0.410</td><td>0.673/0.837/0.273</td><td>0.626/0.883/0.330</td><td>0.636/0.870/0.317</td></tr><tr><td>GPT-OSS 20B</td><td>0.587/1.000/0.413</td><td>0.590/1.000/0.410</td><td>0.672/0.813/0.267</td><td>0.629/0.917/0.340</td><td>0.654/0.867/0.300</td></tr><tr><td>GPT-5.5</td><td>0.590/1.000/0.410</td><td>0.577/1.000/0.423</td><td>0.651/0.850/0.297</td><td>0.662/0.877/0.297</td><td>0.653/0.873/0.303</td></tr></table>