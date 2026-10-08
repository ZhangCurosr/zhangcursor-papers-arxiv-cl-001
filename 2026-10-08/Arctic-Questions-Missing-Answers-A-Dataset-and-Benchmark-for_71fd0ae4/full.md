# Arctic Questions, Missing Answers: A Dataset and Benchmark for LLM Abstention in Arctic Science

Benjamin Wilcox<sup>1</sup>, Dawei Gao<sup>1</sup>, Pradeeban Kathiravelu<sup>2</sup>,

Douglas Causey<sup>2</sup>, Kewei Sha<sup>1</sup>,Yunhe Feng<sup>1</sup>

<sup>1</sup>University of North Texas, Denton, TX

<sup>2</sup>University of Alaska Anchorage, Anchorage, AK

benjamin.wilcox@unt.edu, dawei.gao@unt.edu, pkathiravelu@alaska.edu dcausey@alaska.edu, kewei.sha@unt.edu, yunhe.feng@unt.edu

## Abstract

Large language models (LLMs) should abstain from scientific multiple-choice questions when no option is valid, but frequent abstention alone does not demonstrate sensitivity to answer availability. We introduce ArcticQA, a dataset of 194 questions derived from primary Arctic research, with automated checks of answer support and distractor contradiction against source evidence. We further develop ArcticAbstain, a paired benchmark comparing answer-present and answer-absent conditions, with the correct answer replaced by a distractor in the latter and an explicit abstention option in both. We evaluate eight models from the Gemini, Claude, and ChatGPT families at high reasoning effort, with three trials per condition, yielding 9,312 recorded responses. Answer-present abstention rates range from 0.0% to 63.0%, whereas replacing the correct answer increases abstention by 5.05 percentage points on average. These findings highlight substantial baseline differences and the need to evaluate abstention frequency and responsiveness jointly. The dataset and benchmark are available at https: //github.com/BenWilcox8/arctic-qa.

## 1 Introduction

Large language models (LLMs) could help researchers navigate and synthesize the multidisciplinary literature of Arctic science. Their usefulness, however, depends on whether they can provide scientifically supported answers and recognize when they should abstain. Arctic research spans interconnected physical, ecological, and human systems, and its findings often depend on specific locations, seasons, populations, and measurement conditions. Moreover, field observations are unevenly distributed across the region: prior research has documented substantial spatial biases in terrestrial Arctic sampling and scientific citation (Metcalfe et al., 2018). In this setting, an answer that is scientifically plausible in general may be incorrect within the scope of a particular question. Such errors can distort the interpretation of published findings and propagate into subsequent scientific analyses, making abstention an important component of trustworthy Arctic question answering (QA).

LLM abstention has already been investigated in other scientific disciplines. For example, Wen et al. (2024) study abstention in biomedical and computer-science QA and show that behavior varies with the model, question type, and available context. Broader evaluations, including Abstain-QA and AbstentionBench, also reveal persistent difficulties in distinguishing questions that warrant an answer from those that warrant abstention (Madhusudhan et al., 2025; Kirichenko et al., 2025). However, to the best of our knowledge, no dedicated benchmark has systematically evaluated LLM abstention on scientific questions grounded in primary Arctic research. Existing results therefore leave unresolved whether LLMs can appropriately withhold answers when the available choices conflict with the specific conditions and findings of Arctic studies.

Addressing this gap requires an evaluation resource that preserves the scope of Arctic scientific evidence. Geographic variation, seasonal dependence, and differences in study populations or measurement protocols complicate the construction of both correct answers and demonstrably incorrect alternatives. A finding from one setting cannot automatically establish what is valid in another. These considerations make benchmark curation particularly demanding: questions must state the relevant conditions, answers must be traceable to source findings, and distractors must be evaluated within the same scope. A dedicated Arctic resource is therefore necessary to examine abstention against explicit scientific evidence and to make the basis of the evaluation inspectable.

To support trustworthy evaluation, we introduce ArcticQA, an evidence-grounded QA dataset constructed from primary Arctic research. For each item, we identify a clearly defined finding and its supporting evidence, then formulate a standalone question that preserves the conditions needed to interpret that finding. Separate automated checks assess whether the designated answer is supported by the source and whether each distractor is contradicted within the question’s stated scope. Checking distractors explicitly is essential because an option’s absence from a paper does not establish that it is false. Building on prior work in papergrounded and automatically generated scientific QA (Dasigi et al., 2021; Wan et al., 2024), ArcticQA provides item-level evidence and validation checks to support transparent assessment of Arctic scientific answers.

Using ArcticQA, we develop ArcticAbstain, a paired benchmark centered on the following question: How does an LLM’s abstention behavior change when a scientifically valid answer is removedfrom the available options? We operationalize abstention as selecting an explicit abstention option instead of a substantive answer. In the answerpresent condition, each question includes one designated correct answer, three distractors, and the abstention option. In the matched answer-absent condition, the correct answer is replaced with a fourth distractor, while the question and the remaining options are retained. The underlying scientific question remains answerable from its source; the manipulation changes whether a valid answer appears among the choices. Under this protocol, abstaining in the answer-present condition constitutes false abstention, whereas selecting a substantive option in the answer-absent condition constitutes false commitment. Comparing the two conditions measures sensitivity to answer availability relative to a model’s baseline tendency to abstain, which a raw abstention rate alone cannot establish.

We evaluate eight models from the Gemini, Claude, and ChatGPT families on 194 Arctic scientific questions at high reasoning effort. Each model answers both conditions three times, yielding 9,312 recorded responses. We analyze abstention alongside the accuracy of non-abstaining responses. Answer-present abstention rates range from 0.0% to 63.0% across models, while replacing the correct answer increases abstention by 5.05 percentage points on average. These results reveal substantial variation in baseline abstention and a comparatively limited average change after answer replacement under our evaluation protocol. They underscore the need to evaluate both a model’s willingness to answer Arctic scientific questions and its responsiveness to the absence of a valid option.

Our main contributions are as follows:

• ArcticQA: an evidence-grounded Arctic QA dataset. We construct questions from primary Arctic research, retain item-level supporting evidence, and apply separate automated checks to answers and distractors to support trustworthy, inspectable evaluation.

• ArcticAbstain: a paired benchmark for Arctic scientific abstention. We vary answer availability within matched question pairs to measure false abstention, false commitment, and changes in abstention relative to baseline behavior.

• An empirical evaluation of eight LLMs. We conduct repeated trials across three model families, characterizing differences in abstention behavior and sensitivity to answer replacement on Arctic scientific questions.

## 2 Related Work

Work on selective generation asks whether a model can estimate uncertainty and withhold an unreliable answer. Self-evaluation prompts can improve selective generation (Ren et al., 2023), while multimodel collaboration can expose knowledge gaps before a response is produced (Feng et al., 2024a). Abstain-QA evaluates willingness to answer across answerable and unanswerable prompts (Madhusudhan et al., 2025). AbstentionBench broadens this analysis across types of unanswerable questions and reports that reasoning-oriented training can increase forced problem solving (Kirichenko et al., 2025). These studies motivate our high-reasoningeffort setting, but their conclusions cannot be assumed to transfer directly to source-bounded Arctic scientific questions.

Scientific abstention has also been studied through controlled context changes and external verification. Context-perturbation experiments alter the evidence available to a model and observe how its answer or abstention changes (Wen et al., 2024). Abstention-aware scientific reasoning systems can decompose claims and check them against retrieved evidence or natural-language-inference models (Abdaljalil et al., 2026). Scientific claimverification datasets such as SciFact-Open likewise connect claims to evidence retrieval and verification (Wadden et al., 2022). These systems evaluate the reliability of a composite process. ArcticQA instead evaluates the response of the tested model to a self-contained multiple-choice item, without giving the model a retrieval or verification pipeline.

Scientific question-answering resources cover difficult reasoning and questions anchored in research papers. GPQA targets graduate-level questions in biology, physics, and chemistry, while MMLU-Pro expands difficult multi-task multiplechoice evaluation (Rein et al., 2023; Wang et al., 2024). Other work builds information-seeking questions around research papers or generates scientific QA datasets with fine-grained checks (Dasigi et al., 2021; Wan et al., 2024). Round-trip consistency can filter synthetic question-answer pairs by testing whether an answer can be reconstructed (Alberti et al., 2019). Decontextualization work formalizes the requirement that text remain understandable outside its original document (Choi et al., 2021). ArcticQA combines these concerns by deriving each question from bounded source evidence. The final question must also stand alone for a reader who does not have the paper.

Automatic multiple-choice construction introduces a separate risk: a plausible distractor can be partly correct, true under another scope, or equivalent to the answer. Prior work therefore treats distractor generation and distractor evaluation as distinct tasks (Yu et al., 2024; Byun and Choi, 2025; Feng et al., 2024b). Iterative critique can improve generated questions (Yao et al., 2025), but automated refinement and model judging can preserve shared errors or introduce judge bias (Zheng et al., 2023; Benedetto et al., 2025). ArcticQA responds with per-option source checks, deterministic rejection rules, bounded revision paths, and an explicitly non-gold machine-acceptance label.

## 3 ArcticQA Dataset

We introduce ArcticQA, a question-answering dataset grounded in specific findings from primary Arctic research. Reliable abstention evaluation requires both a supported correct answer and demonstrably incorrect distractors. We therefore develop a construction pipeline that links each target answer to supporting source evidence and verifies each distractor within the question’s stated scope. The pipeline comprises paper selection, finding extraction and question generation, QA validation, and distractor generation and validation.

## 3.1 Paper selection and eligibility

We identified 84,829 papers through keyword searches using common Arctic research terms in Semantic Scholar<sup>1</sup>. Metadata filtering reduced this set to 4,420 candidate papers, whose full texts were subsequently assessed for eligibility. We defined geographic eligibility using a latitude threshold of 66.56<sup>◦</sup> N for terrestrial studies and a predefined list of eligible Arctic marine regions for marine studies. This list remained fixed throughout the eligibility assessment process.

## 3.2 Finding extraction and question generation

For each eligible paper, an LLM reads the full text and identifies the primary finding together with its supporting textual evidence. Once validated, the finding is held fixed as the target answer. A separate LLM writer then formulates a standalone question that asks specifically for this answer while preserving the scope and conditions of the original finding. This answer-first approach, followed by answer reconstruction, draws on the roundtrip-consistency principle of Alberti et al. (2019). We adapt scientific-paper QA generation (Wan et al., 2024) by restricting generation to one clearly scoped finding per paper.

## 3.3 QA validation

We validate each QA pair through blinded answer reconstruction followed by a separate verification step. The reconstructor LLM receives the question, its context, and the source evidence, but not the proposed answer. It produces a reconstructed answer, supporting evidence, and alternative answers. A separate verifier compares the proposed and reconstructed answers and assesses their consistency with the source. Specifically, it checks whether the evidence entails the proposed answer, whether the question and answer preserve the finding’s scope and claim type, and whether the question’s context is sufficient without disclosing the answer. A QA pair proceeds to distractor generation only after passing all checks. This combination of reconstruction and entailment checking is informed by the complementary factual-consistency signals identified by Fabbri et al. (2022).

## 3.4 Distractor generation and validation

For each validated QA pair, we construct four distractors through separate candidate-generation and verification stages. This staged design draws on prior work on automated distractor generation and filtering (Yu et al., 2024; Byun and Choi, 2025). A writer LLM proposes four to six candidates by modifying numerical values, categories, directions of change, scope, or entities. The writer does not verify its own outputs; each candidate undergoes a separate verification step.

To be retained, a candidate must be contradicted by specific source evidence within the question’s stated scope. It is rejected if a reasonable interpretation of the paper could support it as a correct answer. Absence from the source alone does not establish falsity, and validity is assessed under the conditions specified by the question, including the relevant location and time. We retain four candidates that satisfy these criteria. These checks emphasize distractor validity, which prior work distinguishes from plausibility (Feng et al., 2024b), and make the basis for rejecting each option traceable to the source evidence.

## 4 ArcticAbstain Benchmark

ArcticAbstain evaluates LLM abstention on Arctic scientific questions using two matched multiplechoice conditions: answer-present and answerabsent. In the former, the correct answer is included among the options; in the latter, it is replaced by a validated distractor. Comparing responses across these conditions measures how abstention changes when a correct answer is no longer available for selection.

## 4.1 Paired abstention protocol

Each pair shares the same question wording and three validated distractors. Both items also include the explicit option “I abstain from answering,” consistent with prior work that allows LLMs to reject all candidate answers during self-evaluation (Ren et al., 2023). Each item contains five options, with their order randomized to reduce systematic effects of option position.

## 4.1.1 Answer-present condition

The answer-present item contains the gold answer, three distractors, and the abstention option. Under the benchmark’s scoring protocol, selecting the gold answer is correct, selecting a distractor is an incorrect answer, and selecting the abstention option constitutesfalse abstention, or over-abstention. This condition quantifies how often a model abstains despite the presence of a valid answer.

## 4.1.2 Answer-absent condition

We replace the gold answer with a fourth validated distractor, retaining the same question, three shared distractors, and abstention option. Because no candidate answer is valid within the question’s stated scope, abstention is the only response scored as correct. Selecting any distractor constitutes false commitment, indicating failure to abstain when no valid answer is available.

The underlying question remains answerable from the source paper in both conditions; only the available answer options change. Together, the paired conditions distinguish false abstention from false commitment and quantify the change in abstention relative to the model’s answer-present baseline behavior.

## 5 Experiments

## 5.1 Experimental setup

We evaluate eight models from three families: two Gemini models, three Claude models, and three ChatGPT models. All models are configured to use high reasoning effort. Each question from ArcticQA is evaluated under the paired answer-present and answer-absent conditions defined in ArcticAbstain. Each model answers each condition three times through separate calls, yielding six responses per model and 48 responses across all eight models for each base question.

The analysis includes 194 questions with all 48 planned response records, for a total of 9,312 recorded responses. Invalid responses are excluded from metric calculations; thus, the number of recorded responses is distinguished from the number of valid responses used in the metric denominators for reported results.

## 5.2 Evaluation metrics

We evaluate answer selection and abstention in ArcticAbstain using six metrics from prior abstention research (Feng et al., 2024a; Wen et al., 2025, 2024). For each model, we partition its responses across all evaluated questions and trials into five mutually exclusive categories. In the answerpresent condition, $N _ { 1 } , N _ { 2 }$ , and $N _ { 3 }$ denote the numbers of correct answers, incorrect answers, and abstentions, respectively. In the answer-absent condition, $N _ { 4 }$ and $N _ { 5 }$ denote the numbers of incorrect answers and abstentions, respectively. Because no substantive option is correct in the answer-absent condition, every non-abstaining response in that condition is counted in $N _ { 4 }$ . Under the benchmark’s answer-availability criterion, $N _ { 3 }$ represents false abstentions and $N _ { 5 }$ represents appropriate abstentions. The total number of evaluated responses is $\textstyle N = \sum _ { i = 1 } ^ { 5 } N _ { i }$

Abstention accuracy (ACC) measures overall success in providing correct answers or abstaining when necessary:

$$
\mathrm { A C C } = { \frac { N _ { 1 } + N _ { 5 } } { N } } .\tag{1}
$$

It credits correct answers when a valid option is present and abstentions when all substantive options are incorrect, thereby jointly evaluating answer correctness and abstention behavior.

Abstention precision measures the proportion of abstentions that occur when no valid answer option is available:

$$
\mathrm { P r e c i s i o n } _ { \mathrm { a b s } } = \frac { N _ { 5 } } { N _ { 3 } + N _ { 5 } } .\tag{2}
$$

A higher value indicates that a greater proportion of the model’s abstentions are appropriate under the evaluation protocol. Following the definition summarized by Wen et al. (2025), abstention recall is computed as

$$
{ \mathrm { R e c a l l } } _ { \mathrm { a b s } } = { \frac { N _ { 5 } } { N _ { 2 } + N _ { 4 } + N _ { 5 } } } .\tag{3}
$$

This metric measures the proportion of appropriate abstentions among the combined set of appropriate abstentions and incorrect substantive answers. Its denominator includes errors from both conditions: $N _ { 2 }$ captures incorrect selections despite the availability of a valid option, and $N _ { 4 }$ captures incorrect selections when no valid option is available. Thus, this definition penalizes incorrect answering in both conditions and differs from recall restricted to identifying answer-absent cases. We summarize abstention precision and recall using their harmonic mean, the abstention F1-score:

$$
\mathrm { F 1 _ { a b s } = \frac { 2 P r e c i s i o n _ { a b s } R e c a l l _ { a b s } } { P r e c i s i o n _ { a b s } + R e c a l l _ { a b s } } . }\tag{4}
$$

The abstention rate measures how frequently the model abstains across both conditions:

$$
\mathrm { A b s t e n t i o n \ : R a t e } = \frac { N _ { 3 } + N _ { 5 } } { N } .\tag{5}
$$

This rate describes the model’s overall tendency to withhold answers. Because it includes both false and appropriate abstentions, a higher abstention rate does not necessarily indicate better overall abstention performance.

Finally, reliable accuracy (R-Acc) measures answer accuracy conditional on the model selecting a substantive option:

$$
{ \mathrm { R - A c c } } = \frac { N _ { 1 } } { N _ { 1 } + N _ { 2 } + N _ { 4 } } .\tag{6}
$$

Its denominator includes all non-abstaining responses across both conditions. R-Acc therefore characterizes the correctness of committed answers and should be interpreted alongside the abstention rate to account for how frequently the model chooses to answer.

## 5.3 Aggregation and statistical analysis

For the aggregate metric comparison, we compute each metric separately for the three trials and report the median across trials in Table 2. For the paired condition analysis in Table 1, we pool responses across all three trials to estimate each model’s condition-specific abstention rates. The paired shift is the answer-absent rate minus the answer-present rate, expressed in percentage points (pp).

We estimate 95% confidence intervals for these shifts using 10,000 bootstrap resamples of questions with a random seed of 7. Each resample retains both conditions and all three trials for every selected question, preserving the dependence among responses associated with that question. For each model, a question-level sign-flip permutation test assesses whether abstention changes after the correct answer is replaced. We apply Holm correction across the eight model-specific tests to control the family-wise error rate and assess significance at an adjusted $p < 0 . 0 5$

## 5.4 Results and analysis

## 5.4.1 Variation in baseline abstention

Table 1 shows substantial variation in answerpresent abstention, ranging from 0.0% to 63.0%. ChatGPT 5.6 Sol and ChatGPT 5.6 Terra almost always select a substantive option, with abstention rates of 0.2% and 0.0%, respectively. The Gemini models abstain on 6.0%–6.4% of answer-present responses, whereas the Claude models abstain on 36.1%–48.5%. ChatGPT Astra has the highest rate, at 63.0%. The large differences among the three

![](images/c86b708ff6c65382a8c149262eb6157241e354839db2cb152cb4183f29a7874a.jpg)  
Figure 1: Abstention rates under the answer-present and answer-absent conditions for the eight models. Each segment connects a model’s two rates, pooled across three trials. Labels report the answer-absent minus answerpresent shift in percentage points.

ChatGPT models show that model-family membership alone does not characterize abstention behavior in this evaluation.

## 5.4.2 Response to correct-answer removal

Replacing the correct answer with a distractor increases the abstention point estimate for every model. The increases range from 1.5 to 11.1 percentage points, with a mean increase of 5.05 percentage points across the eight models. These shifts are modest relative to the 63.0% range in answer-present abstention. Figure 1 visualizes each model’s paired rates.

Claude Fable 5.1 and ChatGPT Astra exhibit the largest increases, at 10.8 and 11.1 percentage points, respectively. Both changes remain statistically significant after Holm correction, with adjusted p-values of 0.0008 and 0.0011. The remaining six models do not meet the adjusted significance threshold. Thus, the point estimates consistently indicate increased abstention after answer removal, while the model-specific tests provide evidence of a change for two models after correction for multiple comparisons.

## 5.4.3 Accuracy and abstention must be interpreted jointly

Table 2 reports median metrics across the three trials. ChatGPT Astra has the highest median ACC (0.541), abstention F1-score (0.618), and R-Acc (0.540), together with the highest overall abstention rate (0.686). However, its 63.0% answer-present abstention rate also indicates frequent abstention when a valid option is available. Its comparatively high conditional answer accuracy therefore coexists with substantial false abstention.

All reported R-Acc medians exceed 0.125, the expected value for uniform random selection among the four substantive options under equally represented conditions: $\textstyle { \frac { 1 } { 2 } } \ \times \ { \frac { 1 } { 4 } } \ = \ 0 . 1 2 5$ The other seven models have R-Acc medians between 0.261 and 0.379. These descriptive comparisons do not establish statistically significant differences between models. For example, ChatGPT 5.6 Sol has an R-Acc of 0.291, numerically close to the Gemini models’ values of 0.303 and 0.306, but a much lower abstention rate of 0.011. Conditional answer accuracy and the tendency to abstain therefore capture distinct aspects of performance, and both are needed to assess reliability on Arctic scientific questions.

## 6 Conclusion

We introduced ArcticQA, a dataset grounded in primary Arctic research, and ArcticAbstain, a paired benchmark for evaluating LLM abstention under answer-present and answer-absent conditions. Across eight models, answer-present abstention rates ranged from 0.0% to 63.0%, while removing the correct answer increased abstention by 5.05 percentage points on average. These findings highlight the distinction between a model’s tendency to abstain and its responsiveness to answer availability. Reliable evaluation should therefore consider false abstention, false commitment, and the accuracy of non-abstaining responses jointly. Our dataset and benchmark provide a foundation for assessing and improving LLM reliability in Arctic scientific question answering.

## 7 Limitations

ArcticAbstain is a geographically bounded case study whose coverage reflects discovery queries, full-text availability, and model-guided processing order. It is not representative of all Arctic research or other scientific domains. Dataset construction relies on model-generated content and automated validation. No sample was reviewed by domain experts; thus, machine\_accepted\_unverified denotes automated acceptance, not expert-verified ground truth. The residual error rate remains unmeasured. The evaluation covers 194 questions, eight models, a single prompt, one explicit abstention option, and high reasoning effort. The paired multiple-choice task assesses rejection of invalid option sets rather than spontaneous abstention in open-ended scientific work. The results do not establish training exposure, explain observed response policies, or demonstrate generalization beyond the evaluated setting.

<table><tr><td>Model</td><td>Present abst.</td><td>Absent abst.</td><td>Shift [95% CI] (pp)</td><td>Permutation p</td><td>Holm p</td></tr><tr><td>Gemini 3.8 Flash</td><td>6.4%</td><td>10.3%</td><td>+3.9 [0.6, 7.4]</td><td>0.037</td><td>0.187</td></tr><tr><td>Gemini 3.7 Flash</td><td>6.0%</td><td>8.6%</td><td>+2.6 [−0.1, 5.4]</td><td>0.063</td><td>0.198</td></tr><tr><td>Claude Fable 5.1</td><td>36.1%</td><td>46.9%</td><td>+10.8 [5.7, 16.1]</td><td>0.00010</td><td>0.0008</td></tr><tr><td>Claude Opus 5</td><td>41.3%</td><td>47.5%</td><td>+6.2 [0.9, 11.6]</td><td>0.031</td><td>0.187</td></tr><tr><td>Claude Sonnet 5</td><td>48.5%</td><td>51.0%</td><td>+2.5 [−1.7, 6.9]</td><td>0.331</td><td>0.331</td></tr><tr><td>ChatGPT Astra</td><td>63.0%</td><td>74.1%</td><td>+11.1 [5.7, 16.6]</td><td>0.00015</td><td>0.0011</td></tr><tr><td>ChatGPT 5.6 Sol</td><td>0.2%</td><td>2.1%</td><td>+1.9 [0.3, 4.0]</td><td>0.049</td><td>0.198</td></tr><tr><td>ChatGPT 5.6 Terra</td><td>0.0%</td><td>1.5%</td><td>+1.5 [0.3, 3.3]</td><td>0.065</td><td>0.198</td></tr></table>

Table 1: Condition-specific abstention rates pooled across three trials for 194 questions. Shifts are answer-absent minus answer-present rates, in percentage points (pp). The 95% confidence intervals use question-level bootstrap resampling. Permutation tests use question-level paired differences; Holm-adjusted p-values account for the eight model-specific tests.

<table><tr><td>Model</td><td>Abst.</td><td></td><td></td><td></td></tr><tr><td></td><td>ACC</td><td>rate</td><td> $\mathbf { F 1 _ { a b s } }$ </td><td>R-Acc</td></tr><tr><td>Gemini 3.8 Flash Gemini 3.7 Flash</td><td>0.329</td><td>0.084 0.073</td><td>0.134 0.113</td><td>0.303 0.306</td></tr><tr><td></td><td>0.326</td><td></td><td></td><td></td></tr><tr><td>Claude Fable 5.1</td><td>0.458</td><td>0.415</td><td>0.465</td><td>0.379</td></tr><tr><td>Claude Opus 5</td><td>0.438</td><td>0.444</td><td>0.459</td><td>0.359</td></tr><tr><td>Claude Sonnet 5</td><td>0.408</td><td>0.498</td><td>0.464</td><td>0.303</td></tr><tr><td>ChatGPT Astra</td><td>0.541</td><td>0.686</td><td>0.618</td><td>0.540</td></tr><tr><td>ChatGPT 5.6 Sol</td><td>0.298</td><td>0.011</td><td>0.029</td><td>0.291</td></tr><tr><td>ChatGPT 5.6 Terra</td><td>0.266</td><td>0.008</td><td>0.021</td><td>0.261</td></tr></table>

Table 2: Metrics pooled across the three trials for eight models evaluated at high reasoning effort on 194 Arctic scientific questions. Each trial’s metrics combine the answer-present and answer-absent conditions.

## References

Samir Abdaljalil, Erchin Serpedin, and Hasan Kurban. 2026. Knowing when not to answer: Abstentionaware scientific reasoning. arXiv (Cornell University).

Chris Alberti, Daniel Andor, Emily Pitler, Jacob Devlin, and Michael Collins. 2019. Synthetic QA corpora generation with roundtrip consistency. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6168–6173.

Luca Benedetto, Shiva Taslimipoor, and Paula Buttery. 2025. A survey on automated distractor evaluation in multiple-choice tasks. In Proceedings ofthe 20th Workshop on Innovative Use of NLP for Building Educational Applications, pages 55–69.

Grace Byun and Jinho D. Choi. 2025. D-GEN: Automatic distractor generation and evaluation for reliable assessment of generative models. In Findings ofACL, pages 3316–3349.

Eunsol Choi, Jennimaria Palomaki, Matthew Lamm, Tom Kwiatkowski, Dipanjan Das, and Michael Collins. 2021. Decontextualization: Making sentences stand-alone. Transactions ofthe Association for Computational Linguistics, 9:447–461.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of NAACL-HLT, pages 4599–4610.

Alexander Fabbri, Chien-Sheng Wu, Wenhao Liu, and Caiming Xiong. 2022. QAFactEval: Improved QAbased factual consistency evaluation for summarization. In Proceedings ofNAACL-HLT, pages 2587– 2601.

Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, and Yulia Tsvetkov. 2024a. Don’t hallucinate, abstain: Identifying llm knowledge gaps via multi-llm collaboration. Preprint, arXiv:2402.00367.

Wanyong Feng, Jaewook Lee, Hunter McNichols, Alexander Scarlatos, Digory Smith, Simon Woodhead, Nancy Ornelas, and Andrew Lan. 2024b. Exploring automated distractor generation for math multiple-choice questions via large language models. In Findings ofNAACL, pages 3067–3082.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. Abstentionbench: Reasoning llms fail on unanswerable questions. arXiv.org.

Nishanth Madhusudhan, Sathwik Tejaswi Madhusudhan, Vikas Yadav, and Masoud Hashemi. 2025. Do LLMs know when to NOT answer? investigating abstention abilities of large language models. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 9329–9345, Abu Dhabi, UAE. Association for Computational Linguistics.

Daniel B. Metcalfe, Thirze D. G. Hermans, Jenny Ahlstrand, Michael Becker, Martin Berggren, Robert G. Björk, Mats P. Björkman, Daan Blok, Nitin Chaudhary, Chelsea Chisholm, Aimée T. Classen, Niles J. Hasselquist, Micael Jonsson, Jeppe A. Kristensen, Bright B. Kumordzi, Hanna Lee, Jordan R. Mayor, Janet Prevéy, Karolina Pantazatou, and 10 others. 2018. Patchy field sampling biases understanding of climate change impacts across the Arctic. Nature Ecology & Evolution, 2(9):1443–1448.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. Gpqa: A graduate-level google-proof q&a benchmark.

Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, and Balaji Lakshminarayanan. 2023. Self-evaluation improves selective generation in large language models. Preprint, arXiv:2312.09300.

David Wadden, Kyle Lo, Bailey Kuehl, Arman Cohan, Iz Beltagy, Lucy Lu Wang, and Hannaneh Hajishirzi. 2022. SciFact-Open: Towards open-domain scientific claim verification. In Findings of EMNLP, pages 4719–4734.

Yuwei Wan, Yixuan Liu, Aswathy Ajith, Clara Grazian, Bram Hoex, Wenjie Zhang, Chunyu Kit, Tong Xie, and Ian Foster. 2024. SciQAG: A framework for auto-generated science question answering dataset with fine-grained evaluation. arXiv preprint arXiv:2405.09939.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Preprint, arXiv:2406.01574.

Bingbing Wen, Bill Howe, and Lucy Lu Wang. 2024. Characterizing LLM abstention behavior in science QA with context perturbations. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 3437–3450, Miami, Florida, USA. Association for Computational Linguistics.

Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. 2025. Know your limits: A survey of abstention in large language models. Preprint, arXiv:2407.18418.

Zonghai Yao, Aditya Parashar, Huixue Zhou, Won Seok Jang, Feiyun Ouyang, Zhichao Yang, and Hong Yu. 2025. MCQG-SRefine: Multiple choice question

generation and evaluation with iterative self-critique, correction, and comparison feedback. In Proceedings ofNAACL-HLT, pages 10728–10777.

Chen-Jui Yu, Wen Hung Lee, Lin Tse Ke, Shih-Wei Guo, and Yao-Chung Fan. 2024. Automating truefalse multiple-choice question generation and evaluation with retrieval-based accuracy differential. In Proceedings ofthe 17th International Natural Language Generation Conference, pages 198–212.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track.