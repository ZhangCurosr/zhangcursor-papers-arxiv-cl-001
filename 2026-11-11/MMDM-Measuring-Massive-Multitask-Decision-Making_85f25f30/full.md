# MMDM: Measuring Massive Multitask Decision Making

Junhao Shen<sup>1∗</sup> Qing Zong<sup>2∗</sup> Yue Guo<sup>3</sup>

Zhanqiu Zhang<sup>3†</sup> Yiwen Guo<sup>4†</sup> Hong Cheng<sup>1</sup>

<sup>1</sup>The Chinese University of Hong Kong <sup>2</sup>Hong Kong University of Science and Technology <sup>3</sup>LIGHTSPEED <sup>4</sup>Independent Researcher

shen.junhao@outlook.com qzong@cse.ust.hk {zqzhang27,guoyiwen89}@gmail.com hcheng@se.cuhk.edu.hk Project Page: mmdm-benchmark.github.io

Hard Single-select Accuracy

![](images/21f02473ad80792d9852d326872a872f8df1b5484e8c8d887d0d001fa79548b1.jpg)  
Soft Targets & Score

![](images/a5686a774e686f503136dd1a22fa15935e7a7d5c89592902154c8d24cf1271c9.jpg)  
Perplexity V1.2 Jev 1.14.0 Microsoft Decision-1 Intern-Decision 4B OpenJev 4B

Figure 1: MMDM capability profiles for five representative decision models. The left panel shows Hard Single-select accuracy across 16 task categories (%, higher is better). The right panel shows soft Brier error over all soft targets, soft Score targets, and other soft tasks (lower is better).

## Abstract

System-1 Decision Models directly make structured probabilistic decisions over queryspecified options given a state in a single forward pass, without autoregressively generating reasoning trajectories or textual responses. Recent evaluations cover diverse tasks but often rely on existing benchmarks, where prior exposure to test material can obscure generalization. Moreover, real-world decision making often involves long contexts, large option spaces, and complex logical dependencies, challenges that existing benchmarks do not adequately capture together. Thus, we introduce MMDM (Massive Multitask Decision Making), a benchmark of 5,448 high-quality questions across 16 task categories, combining newly constructed tasks with source-grounded recompositions and rewrites for single-select, multi-select, and softlabel decision making in fresh contexts. It covers realistic scenarios such as agent state tracking, tool and skill selection, preference judgments, and rubric scoring, with option spaces

ranging from 2 to 1,024 and inputs extending to 64,817 tokens. With MMDM, we conduct a comprehensive quantitative evaluation of 28 decision models, showing that many perform well on common knowledge yet struggle with interdependent options, constrained decisions, and multi-select tasks. For example, Perplexity Decider-V1.2 reaches 78.75% Hard Single-select Accuracy on world knowledge, but only 53.95% on option logic and 47.95% on constraint planning, while the best Hard Multi-select Accuracy across all models is 36.81%. These results highlight joint reasoning, constraint-aware selection, and complete multiselect recovery as key priorities for decisionmodel development, with MMDM providing a testbed for measuring this progress.

## 1 Introduction

System-1 Decision Models make structured probabilistic decisions over query-specified options given a textual state in a single forward pass (Type-Safe AI, 2026; Microsoft, 2026), which support diverse applications, including text and intent classification, model and tool selection, and rubric-based quality scoring (Ling et al., 2026). Therefore, a rigorous and comprehensive evaluation of these models must account for the diversity and complexities of decision making in real-world scenarios.

Recent benchmarks have assessed decision models across diverse tasks and calibration criteria (Duan et al., 2026; Solka, 2026; Standhartinger, 2026). However, many existing evaluations draw heavily on widely circulated public benchmarks, making test-set contamination a central concern for benchmark validity (Oren et al., 2024). Moreover, real-world decisions often combine long contexts, large option spaces, and complex logical dependencies. For example, a practically useful model needs to track the current agent state, determine whether an earlier approval remains valid, and select an eligible tool from a large catalog with heterogeneous prerequisites. These settings call for an evaluation that combines task breadth with new instances and explicit tests of these capabilities.

In this paper, we introduce MMDM (Massive Multitask Decision Making), a benchmark of 5,448 questions for evaluating text-based System-1 Decision Models across realistic decision scenarios. It combines broad task coverage with explicit tests of context understanding, option discrimination, and logical consistency. Table 1 compares its scope with existing decision benchmarks.

Specifically, MMDM evaluates Hard Singleselect, Hard Multi-select, and Soft-label questions, requiring a unique correct choice, the complete valid set, or a prediction matching the reference distribution over outcomes. These answer types capture practical requirements such as choosing the next actions, identifying all eligible tools, and assigning probabilities to rubric scores under uncertainty. Decisions can depend on agent states, action prerequisites, or relationships among options. In benchmark construction, source answers and contextual evidence ground rewrites of states, queries and distractors to reduce memorizationinduced score inflation (Zhu et al., 2024; Zhao et al., 2025). We combine these adaptations with recompositions and newly constructed tasks. The resulting 16 categories span knowledge and comprehension, logic and constraints, agent and workflow decisions, and preference and rubric judgments. With 2–1,024 options and complete inputs of up to 64,817 tokens, MMDM also tests these capabilities across varied context lengths and option spaces.

As shown in Figure 1, our evaluation of 28 decision models reveals substantial differences across capabilities. Models that perform well on world knowledge still struggle with interdependent options, decisions subject to constraints, and complete multi-select recovery. These capabilities matter in agent workflows, where choices interact, actions have prerequisites, and several tools or skills may be valid for the current goal. Specifically, Perplexity Decider-V1.2 (Perplexity AI, 2026) achieves 78.75% Hard Single-select accuracy on world knowledge, compared with 53.95% on option logic and 47.95% on constraint planning. WeMM-JEV 27B (WeMM-JEV Team, 2026) achieves the highest Hard Single-select accuracy of 76.00%, while the highest Hard Multi-select accuracy is 36.81%, highlighting the challenge of recovering all valid options. Model rankings also differ between hardanswer accuracy and soft-distribution error, making both necessary to characterize decision quality.

Overall, our contributions are threefold. First, we introduce a multitask decision benchmark that combines fresh decision instances with long contexts, large option spaces, and interdependent options. Second, we provide a unified evaluation protocol for single-select correctness, multi-select completeness, and soft-label distributional fidelity, together with an open dataset and toolkit. Third, our evaluation reveals capability gaps that are not captured by previous decision-model benchmarks, including failures in joint consistency and complete multi-select recovery, providing concrete directions for future model development.

## 2 Related Work

System-1 Decision Models. System-1 Decision Models produce structured probabilities over request-defined options in a single inference step. Jev introduces a typed decision interface and Reinforcement Learning for Calibrated Decisions, making probability quality a central training objective (TypeSafe AI, 2026). After this closedsource model, open implementations explore different ways to obtain these decisions from languagemodel backbones. SemIf reads logits for option tokens directly and supports shared-prefix reuse across queries (Lee, 2026). OpenJev learns a shared decision head over request-defined options, using supervised learning followed by reinforcement learning (Shen, 2026). Intern-Decision and H2O-Lightning extend typed decisions to multimodal inputs (Zhu et al., 2026; H2O.ai, 2026). Microsoft Decision-1 post-trains Qwen3.5-9B for single-pass scoring in routing, verification, and workflow control (Microsoft, 2026). Together, these approaches develop efficient readout, task adaptation, and probability calibration. Their growing use in software and agent pipelines makes it essential to evaluate whether they preserve decision quality when the relevant evidence, prerequisites, and options interact.

<table><tr><td>Benchmark</td><td>Max. options</td><td>Context length</td><td>New instances</td><td>Multi-select</td><td>Soft targets</td><td>Score tasks</td></tr><tr><td>JEVal (Duan et al., 2026)</td><td>255</td><td>82,678</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>DecisionBench 1.0 (Solka, 2026)</td><td>255</td><td>35,435</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>JevBench (Standhartinger, 2026)</td><td>N/A</td><td>N/A</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>DecideBench 1.1 (Yong, 2026)</td><td>6</td><td>229</td><td>V</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Jev Evaluation (Deußer et al., 2026)</td><td>151</td><td>N/A</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>ReflexBench 1.0 (Navarro, 2026)</td><td>255</td><td>N/A</td><td>X</td><td>X</td><td>X</td><td>」</td></tr><tr><td>MMDM (ours)</td><td>1,024</td><td>64,817</td><td>」</td><td>L</td><td>L</td><td>V</td></tr></table>

Table 1: Comparison of the main evaluation sets of existing decision benchmarks. Context length is the maximum complete input in Qwen3.5-4B tokens; N/A indicates an undisclosed exact value. New instances denotes newly authored or semantically recomposed questions beyond direct reuse or format-only adaptation of existing benchmarks. Multi-select evaluates the complete valid set from individual option rankings; soft targets use non-one-hot references. $\checkmark / \times \colon$ the defined feature is documented/not documented in the main benchmark. MMDM combines all four properties with 1,024-way decisions and long inputs (Detailed distribution is shown in Table A1).

Benchmarks for Decision Making. JevBench establishes an early comparison of typed decision quality, calibration, latency, and cost (Standhartinger, 2026). DecisionBench broadens applied coverage to 43 tasks with runtime-defined options (Solka, 2026), while ReflexBench separated raw model quality from workflow-policy effects (Navarro, 2026). DecideBench subsequently introduced 400 contrastive decisions to test sensitivity to answer-changing context (Yong, 2026). Jev Evaluation examined 37 existing datasets with frozen templates (Deußer et al., 2026), and JEVal expanded systematic evaluation across 36 datasets and ten domains (Duan et al., 2026). These tool kits improve coverage and comparability, but their main evaluations leave complete multi-select decisions underexplored. Additionally, broad source reuse also leaves a need for fresh decision instances (Oren et al., 2024). Complementary resources target individual capabilities. AgentBench evaluates interactive agents (Liu et al., 2024). ToolBench evaluates tool-use tasks (Qin et al., 2024). LongBench and LongBench v2 studies long-context understanding in challenging realistic tasks (Bai et al., 2024, 2025). MuSiQue evaluates multi-hop evidence composition (Trivedi et al., 2022). They provide valuable capability tests, while a common assessment of option selection and probability allocation across these settings remains necessary. MMDM addresses this gap by combining fresh decision scenarios, long inputs, and large option spaces with single-select, multiselect, and soft-label objectives.

## 3 The MMDM Benchmark

MMDM evaluates decision capabilities across diverse tasks through a shared state–query–options interface. First, we formalize the decision problems, answer types and evaluation objectives (Section 3.1). Then, we describe data collection and construction (Section 3.2). Finally, we present task coverage and dataset statistics (Section 3.3).

## 3.1 Task Formulation

Decision Instance. A decision instance consists of a state s, a query $q ,$ and an ordered set of options O. The state provides the evidence and current situation, the query specifies the decision to make, and each option $o _ { j } = ( a _ { j } , c _ { j } )$ contains a name $a _ { j }$ and an optional criterion $c _ { j }$ describing its meaning or applicability. A decision model $f _ { \theta }$ maps these inputs to an option distribution

$$
x = ( s , q , \mathcal { O } ) , \qquad p = f _ { \theta } ( x ) \in \Delta ^ { K - 1 } ,\tag{1}
$$

where $\mathcal { O } = \left( o _ { 1 } , \ldots , o _ { K } \right)$ and $\Delta ^ { K - 1 }$ is the probability simplex, so $p _ { j } \geq 0$ and $\begin{array} { r } { \sum _ { j = 1 } ^ { K } p _ { j } = 1 } \end{array}$ . Options are provided in each request, allowing their names, meanings, and number to vary beyond a fixed output vocabulary. Criteria may refer to other options, requiring the model to interpret the option set jointly with the state and query.

Answer Types. Each instance has a reference distribution $\pi \in \Delta ^ { K - 1 }$ with one of three answer semantics. We use $\mathbf { 1 } [ \cdot ]$ for the indicator function.

Hard Single-select specifies one correct option y, such as the only tool that meets a request’s requirements. The model should assign this option the highest probability. The reference places all probability mass on it,

$$
\pi _ { j } = { \bf 1 } [ j = y ] .\tag{2}
$$

Hard Multi-select specifies a complete set $G$ of equally valid options, such as several tools that can each fulfill the same request. Every member must rank above every invalid option. Equal reference mass represents their equal validity,

$$
\pi _ { j } = { \frac { \mathbf { 1 } [ j \in G ] } { | G | } } .\tag{3}
$$

Soft-label specifies how likely each outcome is under the uncertainty described in the state. For example, a tool call may succeed or fail because the service is intermittently available. If the stated service conditions imply a 70% success probability, the reference assigns 0.7 to success and 0.3 to $f a i l -$ ure. The model should match the probabilities of all outcomes, including less likely ones. For outcomes $\omega \in \Omega$ and a rule r mapping each outcome to an option index, the reference is

$$
\pi _ { j } = \sum _ { \omega \in \Omega } P ( \omega \mid x ) \mathbf { 1 } [ r ( \omega ) = j ] .\tag{4}
$$

Scoring tasks use ordered options with numeric values, such as levels of report completeness. A known score gives a hard single-select reference; uncertain outcomes give a soft distribution over levels. The expected score summarizes this distribution on the rubric’s numeric scale. Appendix B.3 details label derivation.

Evaluation Metrics. Per-question correctness for the two hard-answer tracks is

$$
a _ { \mathrm { s i n g l e } } ( p , y ) = { \bf 1 } \bigg [ p _ { y } > \operatorname* { m a x } _ { j \neq y } p _ { j } \bigg ] ,\tag{5}
$$

$$
a _ { \mathrm { m u l t i } } ( p , G ) = \mathbf { 1 } \left[ \operatorname* { m i n } _ { j \in G } p _ { j } > \operatorname* { m a x } _ { j \notin G } p _ { j } \right] .\tag{6}
$$

Hard Single-select and Hard Multi-select Accuracy average these correctness values over their respective tracks, counting failed requests and boundary ties as incorrect. Higher accuracy is better.

Separation–Balance F-score (SB-F) combines separation from incorrect options with balanced support for the correct set. For $N = | G |$ , let $\mu =$ $N ^ { \bar { - 1 } } \sum _ { i \in G } p _ { i }$ be the mean correct-option probability and $b = \operatorname* { m a x } _ { j \not \in G } p _ { j }$ the largest incorrect-option probability, with $b = 0$ if every option is correct. The separation component is

$$
S ( p , G ) = \frac { 1 } { N } \sum _ { i \in G } \frac { \operatorname* { m a x } ( p _ { i } - b , 0 ) } { \operatorname* { m a x } ( p _ { i } , \mu ) + b } .\tag{7}
$$

A summand is zero when its denominator is zero. Each correct option earns credit only above b, and µ prevents negligible probability from earning full credit. Correct-set balance is

$$
B ( p , G ) = \frac { \left( \sum _ { i \in G } p _ { i } \right) ^ { 2 } } { N \sum _ { i \in G } p _ { i } ^ { 2 } } ,\tag{8}
$$

with $B ( p , G ) = 0$ when the correct set receives no mass. Their product defines

$$
\mathrm { S B F } ( p , G ) = S ( p , G ) B ( p , G ) .\tag{9}
$$

SB-F lies in [0, 1], with higher values better. We average per-question scores over all valid multiselect responses to show the performance.

Soft Brier measures error across the full reference distribution. Score nMAE measures expectedscore error relative to the rubric range, using the numeric levels $v _ { j }$ . These per-question errors are

$$
\operatorname { B r i e r } ( p , \pi ) = \sum _ { j = 1 } ^ { K } ( p _ { j } - \pi _ { j } ) ^ { 2 } ,\tag{10}
$$

$$
e _ { \mathrm { s c o r e } } ( p , \pi , v ) = { \frac { \left| \sum _ { j = 1 } ^ { K } v _ { j } ( p _ { j } - \pi _ { j } ) \right| } { \operatorname* { m a x } _ { j } v _ { j } - \operatorname* { m i n } _ { j } v _ { j } } } .\tag{11}
$$

Soft Brier and Score nMAE average Equations (10) and (11) over valid native responses in their respective tracks. Lower values are better for both.

## 3.2 Data Collection and Construction

Collection. We select public datasets and agent trajectories that support a concrete decision with explicit options and a checkable answer. Sources span knowledge, reading, reasoning, preferences, code, and agent interactions. For example, QuAL-ITY (Pang et al., 2022) and LongBench v2 (Bai et al., 2025) supply document-grounded questions, BIRD (Li et al., 2023) supplies database problems, and ComplexFuncBench (Zhong et al., 2025) and ToolTalk (Farn and Shin, 2023) supply interaction histories. Source questions and annotations provide the basis for adaptation and verification.

Construction. We combine source-grounded adaptation with semantic recomposition and newly constructed tasks. For rewritten source instances, we retain the original answers and supporting evidence while reformulating the state, query, and distractors. Semantic and option-quality checks select the resulting instances. This strategy builds on evidence that benchmark rewriting can mitigate memorization-induced score inflation (Zhu et al., 2024; Zhao et al., 2025). Recompositions introduce decision questions grounded in source databases or agent histories. Executable mechanisms construct interacting options, constrained plans, state transitions, and rubric outcomes with computable answers. Across all three routes, distractors are designed around identifiable failure modes, such as applying an expired rule, confusing entities, or overlooking the prerequisite.

Validation and Selection. We verify answers using source evidence, executable checks, and model review appropriate to each construction route. Fivemodel majority screening is followed by independent human expert review of borderline cases. Both stages apply the same quality standards for distractors, answer leakage, duplicates, and source overlap. An option-only probe masks the state and query; instances with unresolved shortcuts are excluded. Appendix B details the review procedure and selection rules.

## 3.3 Task Coverage and Dataset Statistics

MMDM contains 5,448 questions across 16 task categories, comprising 4,112 Hard Single-select, 633 Hard Multi-select, and 703 Soft-label questions. The 663 Score questions include 35 with hard levels and 628 with soft targets. The collection covers 62 registered source identifiers and 80 category–subcategory combinations. Option sets range from 2 to 1,024 options, including 94 questions with more than 255 options. Complete inputs include the state, query, option names, and criteria. We measure complete inputs with the Qwen3.5- 4B tokenizer from the Qwen family (Yang et al., 2025). Their median length is 827 tokens, the 95th percentile is 13,318, and the maximum is 64,817. These ranges support evaluation from common short decisions to long-document reasoning and large-catalog selection.

The categories connect knowledge and language understanding with decisions about evidence, rules,

Task Coverage and Answer Semantics

![](images/85bf491b7883c30e14e5ee932573dee7e79230d28b53a3199e9a7797b8a6ddba.jpg)  
Figure 2: Task coverage and answer semantics. The colors of the 16 task categories are fixed across figures. Labels show the total count followed by (Hard Singleselect / Hard Multi-select / Soft-label) counts.

actions, and assessments. They cover world knowledge and reading comprehension; deductive, quantitative, and option-dependent reasoning; agent state tracking, tool and skill selection, and workflows; and preference, rubric, and emotion judgments. Figure 2 summarizes all categories and their answer types. Appendix A further elaborate each category and option-count distributions; Appendix F presents representative questions and model responses.

## 4 Experiment

## 4.1 Experimental Setup

Models. We evaluate 28 System-1 decision models and use Random as a heuristic reference. These models return option probabilities directly through a decision-scoring head or a single decoding step, without generated reasoning or adaptive thinking. They include hosted APIs and locally deployed checkpoints. Each model uses its supported context window and option limit, preserving the capabilities available in its released checkpoint or API. Appendix C.1 details model selection, deployment settings, and the Random reference.

Protocol and Metrics. The evaluator preserves the query and all option names and criteria, deterministically removing the middle of the state when the context budget is exceeded. Independent option sets exceeding a model’s option cap are divided into groups that fit the cap. Each request scores all options in a group together. From each group, the highest-probability option advances for singleselect, and up to |G| highest-probability options advance for multi-select. This reduction repeats until the retained options fit one final request. Reference cardinality and gold identities are hidden from the model. Option-dependent and distributional tasks keep the entire option set without splitting; inputs exceeding the model’s option cap count as capacity failures. Grouped outputs contribute to hard selection metrics; full-distribution metrics use native responses. We use the metrics defined in Section 3.1. Hard accuracies retain every question in the corresponding track, including failed requests. SB-F, Soft Brier, and Score nMAE use native predictions. Appendix C.2 specifies token budgets, grouping, and failure accounting.

<table><tr><td>Decision Model</td><td>Single-select Acc.↑</td><td>Multi-select Acc.↑</td><td>SB-F↑|</td><td>Soft Brier↓</td><td>Score nMAE↓</td></tr><tr><td colspan="6">Heuristic reference</td></tr><tr><td>Random (uniform)</td><td>25.35 |</td><td>3.28</td><td>0.0000 |</td><td>0.2051</td><td>0.1624</td></tr><tr><td colspan="6">Closed-source decision models</td></tr><tr><td>Jev 1.14.0</td><td>67.17</td><td>22.12</td><td>0.1187</td><td>0.1101</td><td>0.0926</td></tr><tr><td>Microsoft Decision-1</td><td>61.36</td><td>21.96</td><td>0.1186</td><td>0.2795</td><td>0.1836</td></tr><tr><td>Liquid-d1-Ultra</td><td>73.32</td><td>24.33</td><td>0.1624</td><td>0.1321</td><td>0.0883</td></tr><tr><td>Liquid-d1</td><td>62.33</td><td>21.96</td><td>0.1228</td><td>0.2827</td><td>0.1401</td></tr><tr><td>GPT-6 Luna Decisions</td><td>56.57</td><td>12.01</td><td>0.0886</td><td>0.3333</td><td>0.1588</td></tr><tr><td colspan="6">Open-source decision models</td></tr><tr><td>WeMM-JEV 27B</td><td>76.00</td><td>36.81</td><td>0.1946</td><td>0.2327</td><td>0.2203</td></tr><tr><td>KnowLine 31B Gen1</td><td>75.07</td><td>32.39</td><td>0.1777</td><td>0.1233</td><td>0.0624</td></tr><tr><td>Perplexity Decider-V1.2</td><td>74.42</td><td>33.81</td><td>0.1907</td><td>0.1694</td><td>0.1641</td></tr><tr><td>Autotrust JEV 27B-VL</td><td>67.07</td><td>20.22</td><td>0.1268</td><td>0.3405</td><td>0.2128</td></tr><tr><td>Decisio-Gemma-4-31B</td><td>64.64 64.18</td><td>15.32</td><td>0.1026</td><td>0.1653</td><td>0.1034</td></tr><tr><td>Clef Perplexity Decider-V1</td><td>62.38</td><td>16.43</td><td>0.0987</td><td>0.1422 0.1555</td><td>0.1092</td></tr><tr><td>Perplexity Decider-Flash-V1.2</td><td>62.06</td><td>21.48</td><td>0.1291</td><td>0.2836</td><td>0.0999</td></tr><tr><td></td><td>58.29</td><td>16.27</td><td>0.0965</td><td>0.3482</td><td>0.1674</td></tr><tr><td>H2O-Lightning 4B Decisio-Gemma-4-12B</td><td>57.71</td><td>20.54</td><td>0.1031</td><td>0.1627</td><td>0.2684</td></tr><tr><td></td><td>57.44</td><td>18.17</td><td>0.0962</td><td></td><td>0.1045</td></tr><tr><td>Quyet-1.0-Large</td><td></td><td>16.59</td><td>0.0890</td><td>0.1804</td><td>0.1079</td></tr><tr><td>OpenJev 4B</td><td>57.20</td><td>17.38</td><td>0.1085</td><td>0.3687</td><td>0.3106</td></tr><tr><td>Clef-Omni</td><td>56.91</td><td>14.85</td><td>0.0827</td><td>0.2761</td><td>0.1704</td></tr><tr><td>Mapika Decider 4B</td><td>56.49</td><td>14.06</td><td>0.0835</td><td>0.2131</td><td>0.1918</td></tr><tr><td>Clef-Flash</td><td>55.74</td><td>14.85</td><td>0.0843</td><td>0.3553</td><td>0.2586</td></tr><tr><td>Decisio-Qwen3.6-35B-A3B</td><td>53.33</td><td>16.11</td><td>0.1196</td><td>0.2554</td><td>0.1397</td></tr><tr><td>Intern-Decision 4B</td><td>49.03</td><td>11.06</td><td>0.0896</td><td>0.1564</td><td>0.1449</td></tr><tr><td>JevK54B</td><td>48.69</td><td>10.43</td><td>0.0800</td><td>0.2291</td><td>0.1770</td></tr><tr><td>Strands-Decider 2B</td><td>44.65</td><td>8.21</td><td>0.0580</td><td>0.2052</td><td>0.1689</td></tr><tr><td>SemIf 4B</td><td>42.39</td><td>6.32</td><td>0.0750</td><td>0.3402</td><td>0.1243</td></tr><tr><td>Intern-Decision 2B</td><td>38.76</td><td>6.00</td><td>0.0585</td><td>0.2011</td><td>0.1519</td></tr><tr><td>Intern-Decision 0.8B</td><td>32.17</td><td>3.95</td><td>0.0399</td><td>0.2051</td><td>0.1531</td></tr><tr><td>NanoJev 0.6B</td><td>26.68</td><td>7.11</td><td>0.0390</td><td>0.3061</td><td>0.2732</td></tr></table>

Table 2: Overall results. Accuracies are percentages. SB-F uses model-specific valid native multi-select outputs. Sample counts and fixed-cohort SB-F rankings appear in Table A9. Best and second-best accuracy and error values are marked across all models.

## 4.2 Overall Performance

Similar single-select accuracy can accompany very different probability quality. In Table 2, WeMM-JEV and KnowLine differ by less than one percentage point in single-select accuracy. Yet KnowLine has nearly half the Soft Brier error and less than one-third the Score nMAE of WeMM-JEV. A model that chooses the right option frequently can therefore still be a poor fit for applications that use its probabilities or expected scores.

Complete multi-select identification and probability support remain weak. The highest multiselect accuracy in Table 2 is only 36.81%, and every model has an SB-F below 0.20. This means that current models often fail to recover the complete set of valid options and to assign probability appropriately across them. Thus, improving multi-select decisions requires progress not only in finding all correct options, but also in representing their relative support reliably.

<table><tr><td rowspan=2 colspan=4>Decision Model</td><td rowspan=1 colspan=1>Overall</td><td rowspan=1 colspan=16>Hard Single-select Accuracy by task category</td></tr><tr><td rowspan=1 colspan=1>ALL|</td><td rowspan=1 colspan=16>WKNRDC DEDOPLCPL QTR WFD AST PEX EVDCDB IRTTSSAPF SCR EMR</td></tr><tr><td rowspan=1 colspan=21>Heuristic reference</td></tr><tr><td rowspan=1 colspan=4>Random (uniform)</td><td rowspan=1 colspan=1>25.3</td><td rowspan=1 colspan=16>30.730.133.525.7 4.130.7  4.410.435.646.331.9 28.512.837.729.5  9.2</td></tr><tr><td rowspan=1 colspan=21>Closed-source decision models</td></tr><tr><td rowspan=1 colspan=4>Jev 1.14.0</td><td rowspan=1 colspan=1>67.2</td><td rowspan=1 colspan=15>63.469.681.238.242.775.8 52.251.071.584.276.4 78.594.265.785.7</td><td rowspan=1 colspan=1>34.3</td></tr><tr><td rowspan=1 colspan=3>Microsoft Decision-1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=5>62.663.576.838.213.5</td><td rowspan=1 colspan=2>65.043.5</td><td rowspan=1 colspan=2>57.859.0</td><td rowspan=1 colspan=1>76.2</td><td rowspan=1 colspan=2>76.4 60.4</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=2>66.171.4</td><td rowspan=1 colspan=1>47.8</td></tr><tr><td rowspan=1 colspan=2>Liquid-d1-Ultra</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=4>75.175.282.338.8</td><td rowspan=1 colspan=1>47.4</td><td rowspan=1 colspan=1>80.5</td><td rowspan=1 colspan=1>60.2</td><td rowspan=1 colspan=1>74.2</td><td rowspan=1 colspan=1>76.4</td><td rowspan=1 colspan=1>82.6</td><td rowspan=1 colspan=1>88.2</td><td rowspan=1 colspan=1>75.2</td><td rowspan=1 colspan=1>89.6</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>77.1</td><td rowspan=1 colspan=1>39.9</td></tr><tr><td rowspan=1 colspan=2>Liquid-d1</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=4>66.756.376.432.2</td><td rowspan=1 colspan=1>30.4</td><td rowspan=1 colspan=1>67.9</td><td rowspan=1 colspan=1>28.6</td><td rowspan=1 colspan=1>41.7</td><td rowspan=1 colspan=1>72.8</td><td rowspan=1 colspan=1>84.2</td><td rowspan=1 colspan=1>74.9</td><td rowspan=1 colspan=1>79.2</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=2>66.174.3</td><td rowspan=1 colspan=1>49.4</td></tr><tr><td rowspan=1 colspan=4>GPT-6 Luna Decisions</td><td rowspan=1 colspan=1>56.6</td><td rowspan=1 colspan=4>53.566.460.527.6</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>65.3</td><td rowspan=1 colspan=1>16.83</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>69.0</td><td rowspan=1 colspan=1>81.6</td><td rowspan=1 colspan=1>74.5</td><td rowspan=1 colspan=2>64.465.7</td><td rowspan=1 colspan=1>32.0</td></tr><tr><td rowspan=1 colspan=21>Open-source decision models</td></tr><tr><td rowspan=1 colspan=4>WeMM-JEV 27B</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=2>81.075.9</td><td rowspan=1 colspan=2>86.357.9</td><td rowspan=1 colspan=1>47.4</td><td rowspan=1 colspan=1>82.7</td><td rowspan=1 colspan=1>49.1</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>67.2</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>82.8</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>72.1</td><td rowspan=1 colspan=1>77.1</td><td rowspan=1 colspan=1>55.1</td></tr><tr><td rowspan=1 colspan=4>KnowLine 31B Gen1</td><td rowspan=1 colspan=1>75.1</td><td rowspan=1 colspan=2>71.4 72.1</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>54.6</td><td rowspan=1 colspan=1>48.5</td><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>65.8</td><td rowspan=1 colspan=1>81.1</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>83.8</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>77.68</td><td rowspan=1 colspan=1>9.9</td><td rowspan=1 colspan=1>74.2</td><td rowspan=1 colspan=1>85.7</td><td rowspan=1 colspan=1>41.6</td></tr><tr><td rowspan=1 colspan=4>Perplexity Decider-V1.2</td><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>78.8</td><td rowspan=1 colspan=1>73.2</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>53.9</td><td rowspan=1 colspan=1>48.0</td><td rowspan=1 colspan=1>79.8</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>79.4</td><td rowspan=1 colspan=1>70.8</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>80.8</td><td rowspan=1 colspan=1>81.6</td><td rowspan=1 colspan=1>91.4</td><td rowspan=1 colspan=1>69.5</td><td rowspan=1 colspan=1>74.3</td><td rowspan=1 colspan=1>50.6</td></tr><tr><td rowspan=1 colspan=4>Autotrust JEV 27B-VL</td><td rowspan=1 colspan=1>67.1</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>71.8</td><td rowspan=1 colspan=1>72.3</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>74.7</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>62.2</td><td rowspan=1 colspan=1>69.8</td><td rowspan=1 colspan=1>84.5</td><td rowspan=1 colspan=1>81.877.6</td><td rowspan=1 colspan=1>77.6</td><td rowspan=1 colspan=1>92.4</td><td rowspan=1 colspan=1>69.57</td><td rowspan=1 colspan=1>7.1</td><td rowspan=1 colspan=1>35.4</td></tr><tr><td rowspan=1 colspan=4>Decisio-Gemma-4-31B</td><td rowspan=1 colspan=1>64.6</td><td rowspan=1 colspan=1>60.8</td><td rowspan=1 colspan=1>69.1</td><td rowspan=1 colspan=1>65.3</td><td rowspan=1 colspan=1>28.9</td><td rowspan=1 colspan=1>35.7</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>64.5</td><td rowspan=1 colspan=1>71.8</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>77.8</td><td rowspan=1 colspan=1>81.9</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>69.5</td><td rowspan=1 colspan=1>82.9</td><td rowspan=1 colspan=1>31.5</td></tr><tr><td rowspan=1 colspan=4>Clef</td><td rowspan=1 colspan=1>64.2</td><td rowspan=1 colspan=1>72.2</td><td rowspan=1 colspan=1>71.2</td><td rowspan=1 colspan=1>66.8</td><td rowspan=1 colspan=1>36.8</td><td rowspan=1 colspan=1>26.9</td><td rowspan=1 colspan=1>70.8</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>58.9</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>70.4</td><td rowspan=1 colspan=1>76.1</td><td rowspan=1 colspan=1>72.3</td><td rowspan=1 colspan=1>65.2</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>52.8</td></tr><tr><td rowspan=1 colspan=4>Perplexity Decider-V1</td><td rowspan=1 colspan=1>62.4</td><td rowspan=1 colspan=1>61.5</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>68.3</td><td rowspan=1 colspan=1>30.9</td><td rowspan=1 colspan=1>28.1</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>29.8</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1>59.7</td><td rowspan=1 colspan=1>81.9</td><td rowspan=1 colspan=1>78.8</td><td rowspan=1 colspan=1>77.9</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>64.8</td><td rowspan=1 colspan=1>77.1</td><td rowspan=1 colspan=1>37.1</td></tr><tr><td rowspan=1 colspan=4>Perplexity Decider-Flash-V1.2</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1>65.2</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>74.5</td><td rowspan=1 colspan=1>30.9</td><td rowspan=1 colspan=1>17.0</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>17.4</td><td rowspan=1 colspan=1>57.0</td><td rowspan=1 colspan=1>72.8</td><td rowspan=1 colspan=1>79.2</td><td rowspan=1 colspan=1>67.0</td><td rowspan=1 colspan=1>76.1</td><td rowspan=1 colspan=1>82.4</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>68.6</td><td rowspan=1 colspan=1>48.9</td></tr><tr><td rowspan=1 colspan=4>H2O-Lightning 4B</td><td rowspan=1 colspan=1>58.3</td><td rowspan=1 colspan=1>57.5</td><td rowspan=1 colspan=1>57.4</td><td rowspan=1 colspan=1>59.8</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>62.5</td><td rowspan=1 colspan=1>24.8</td><td rowspan=1 colspan=1>54.2</td><td rowspan=1 colspan=1>64.9</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>67.7</td><td rowspan=1 colspan=1>75.2</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>36.0</td></tr><tr><td rowspan=1 colspan=4>Decisio-Gemma-4-12B</td><td rowspan=1 colspan=1>57.7</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>59.5</td><td rowspan=1 colspan=1>56.8</td><td rowspan=1 colspan=1>27.6</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>42.6</td><td rowspan=1 colspan=1>64.9</td><td rowspan=1 colspan=1>78.1</td><td rowspan=1 colspan=1>72.9</td><td rowspan=1 colspan=1>77.3</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>68.6</td><td rowspan=1 colspan=1>28.7</td></tr><tr><td rowspan=1 colspan=4>Quyet-1.0-Large</td><td rowspan=1 colspan=1>57.4</td><td rowspan=1 colspan=1>71.1</td><td rowspan=1 colspan=1>64.9</td><td rowspan=1 colspan=1>68.3</td><td rowspan=1 colspan=1>28.9</td><td rowspan=1 colspan=1>28.7</td><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>28.2</td><td rowspan=1 colspan=1>77.4</td><td rowspan=1 colspan=1>89.1</td><td rowspan=1 colspan=1>82.8</td><td rowspan=1 colspan=1>71.3</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>67.8</td><td rowspan=1 colspan=1>74.3</td><td rowspan=1 colspan=1>21.9</td></tr><tr><td rowspan=1 colspan=4>OpenJev 4B</td><td rowspan=1 colspan=1>57.2</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>42.6</td><td rowspan=1 colspan=1>73.8</td><td rowspan=1 colspan=1>48.0</td><td rowspan=1 colspan=1>25.7</td><td rowspan=1 colspan=1>70.4</td><td rowspan=1 colspan=1>22.4</td><td rowspan=1 colspan=1>49.7</td><td rowspan=1 colspan=1>68.5</td><td rowspan=1 colspan=1>79.2</td><td rowspan=1 colspan=1>73.9</td><td rowspan=1 colspan=1>57.1</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>63.9</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>33.7</td></tr><tr><td rowspan=1 colspan=4>Clef-Omni</td><td rowspan=1 colspan=1>56.9</td><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1>62.4</td><td rowspan=1 colspan=1>34.2</td><td rowspan=1 colspan=1>25.1</td><td rowspan=1 colspan=1>62.1</td><td rowspan=1 colspan=1>31.7</td><td rowspan=1 colspan=1>34.8</td><td rowspan=1 colspan=1>70.8</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>64.0</td><td rowspan=1 colspan=1>58.6</td><td rowspan=1 colspan=1>71.2</td><td rowspan=1 colspan=1>59.7</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>50.0</td></tr><tr><td rowspan=1 colspan=4>Mapika Decider 4B</td><td rowspan=1 colspan=1>56.5</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>54.3</td><td rowspan=1 colspan=1>69.4</td><td rowspan=1 colspan=1>30.9</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>61.0</td><td rowspan=1 colspan=1>14.3</td><td rowspan=1 colspan=1>40.4</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>84.9</td><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>64.4</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>58.8</td><td rowspan=1 colspan=1>74.3</td><td rowspan=1 colspan=1>64.0</td></tr><tr><td rowspan=1 colspan=4>Clef-Flash</td><td rowspan=1 colspan=1>55.7</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>58.3</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>36.2</td><td rowspan=1 colspan=1>19.9</td><td rowspan=1 colspan=1>65.3</td><td rowspan=1 colspan=1>28.6</td><td rowspan=1 colspan=1>34.6</td><td rowspan=1 colspan=1>62.0</td><td rowspan=1 colspan=1>76.2</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>64.4</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>62.9</td><td rowspan=1 colspan=1>51.1</td></tr><tr><td rowspan=1 colspan=4>Decisio-Qwen3.6-35B-A3B</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>64.5</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>62.4</td><td rowspan=1 colspan=1>28.9</td><td rowspan=1 colspan=1>18.7</td><td rowspan=1 colspan=1>65.3</td><td rowspan=1 colspan=1>19.9</td><td rowspan=1 colspan=1>29.7</td><td rowspan=1 colspan=1>60.7</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1>69.5</td><td rowspan=1 colspan=1>47.4</td><td rowspan=1 colspan=1>73.0</td><td rowspan=1 colspan=1>61.4</td><td rowspan=1 colspan=1>68.6</td><td rowspan=1 colspan=1>28.7</td></tr><tr><td rowspan=1 colspan=4>Intern-Decision 4B</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1>46.5</td><td rowspan=1 colspan=1>48.6</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>25.0</td><td rowspan=1 colspan=1>14.6</td><td rowspan=1 colspan=1>65.3</td><td rowspan=1 colspan=1>12.4</td><td rowspan=1 colspan=1>31.6</td><td rowspan=1 colspan=1>41.3</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>66.5</td><td rowspan=1 colspan=1>62.5</td><td rowspan=1 colspan=1>64.0</td><td rowspan=1 colspan=1>63.9</td><td rowspan=1 colspan=1>68.6</td><td rowspan=1 colspan=1>31.5</td></tr><tr><td rowspan=1 colspan=4>JevK5 4B</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>57.9</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>70.1</td><td rowspan=1 colspan=1>30.3</td><td rowspan=1 colspan=1>19.9</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>6.8</td><td rowspan=1 colspan=1>26.9</td><td rowspan=1 colspan=1>50.5</td><td rowspan=1 colspan=1>79.6</td><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=1>43.5</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1>58.4</td><td rowspan=1 colspan=1>68.6</td><td rowspan=1 colspan=1>15.7</td></tr><tr><td rowspan=1 colspan=4>Strands-Decider 2B</td><td rowspan=1 colspan=1>44.6</td><td rowspan=1 colspan=1>44.0</td><td rowspan=1 colspan=1>45.5</td><td rowspan=1 colspan=1>56.1</td><td rowspan=1 colspan=1>30.9</td><td rowspan=1 colspan=1>18.7</td><td rowspan=1 colspan=1>49.5</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>17.2</td><td rowspan=1 colspan=1>59.7</td><td rowspan=1 colspan=1>71.3</td><td rowspan=1 colspan=1>64.0</td><td rowspan=1 colspan=1>51.1</td><td rowspan=1 colspan=1>53.6</td><td rowspan=1 colspan=1>57.9</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>34.8</td></tr><tr><td rowspan=1 colspan=4>SemIf 4B</td><td rowspan=1 colspan=1>42.4</td><td rowspan=1 colspan=1>41.8</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>23.0</td><td rowspan=1 colspan=1>8.2</td><td rowspan=1 colspan=1>53.4</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>46.2</td><td rowspan=1 colspan=1>70.6</td><td rowspan=1 colspan=1>62.1 43.5</td><td rowspan=1 colspan=1>43.5</td><td rowspan=1 colspan=1>57.6</td><td rowspan=1 colspan=1>54.1</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>16.9</td></tr><tr><td rowspan=1 colspan=4>Intern-Decision 2B</td><td rowspan=1 colspan=1>38.8</td><td rowspan=1 colspan=1>37.7</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>50.2</td><td rowspan=1 colspan=1>27.6</td><td rowspan=1 colspan=1>10.5</td><td rowspan=1 colspan=1>55.6</td><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>15.3</td><td rowspan=1 colspan=1>41.3</td><td rowspan=1 colspan=1>61.9</td><td rowspan=1 colspan=1>58.6</td><td rowspan=1 colspan=1>36.</td><td rowspan=1 colspan=1>52.29</td><td rowspan=1 colspan=1>48.9</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>30.9</td></tr><tr><td rowspan=1 colspan=4>Intern-Decision 0.8B</td><td rowspan=1 colspan=1>32.2</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>36.3</td><td rowspan=1 colspan=1>35.1</td><td rowspan=1 colspan=1>21.7</td><td rowspan=1 colspan=1>8.8</td><td rowspan=1 colspan=1>48.4</td><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>11.8</td><td rowspan=1 colspan=1>37.0</td><td rowspan=1 colspan=1>55.8</td><td rowspan=1 colspan=1>38.4</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>34.9</td><td rowspan=1 colspan=1>37.3</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>23.6</td></tr><tr><td rowspan=1 colspan=4>NanoJev 0.6B</td><td rowspan=1 colspan=1>26.7</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>30.2</td><td rowspan=1 colspan=1>40.2</td><td rowspan=1 colspan=1>19.7</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>39.7</td><td rowspan=1 colspan=1>47.9</td><td rowspan=1 colspan=1>24.6</td><td rowspan=1 colspan=1>39.9</td><td rowspan=1 colspan=1>10.1</td><td rowspan=1 colspan=1>45.5</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>13.5</td></tr></table>

Table 3: Hard Single-select Accuracy (%, ↑). ALL aggregates the 4,112 single-select questions. Colors mark the best and second-best model values as in Table 2. Task codes are WKN (World knowledge); RDC (Reading comprehension); DED (Deductive reasoning); OPL (Option logic); CPL (Constraint planning); QTR (Quantitative and temporal reasoning); WFD (Workflow decision); AST (Agent state tracking); PEX (Policies and exceptions); EVD (Evidence validation); CDB (Code and databases); IRT (Intent routing); TSS (Tool and skill selection); APF (Answer preference); SCR (Score and rubrics); EMR (Emotion recognition).

Expected-score accuracy and distribution fidelity favor different models. Jev achieves the lowest Soft Brier, while KnowLine achieves the lowest Score nMAE. GPT-6 Luna Decisions also has a lower Score nMAE than Microsoft Decision-1 despite a higher Soft Brier. This difference arises because the expected score compresses an entire probability distribution into a single value, allowing probability errors at different score levels to cancel out. Therefore, reporting both metrics shows whether a model predicts a useful final score and whether the probabilities behind that score are themselves accurate.

## 4.3 Performance by Task Category

Models struggle when several conditions must hold at once. WeMM-JEV leads Deductive Reasoning in Table 3, yet scores below 60% on Option Logic, Constraint Planning, and Workflow Decision. KnowLine shows the same contrast, with 86.0% on Deductive Reasoning but 54.6% on Option Logic and 48.5% on Constraint Planning. In these tasks, an option may look correct on its own but conflict with another option, a rule, or an earlier action. MMDM separates these requirements from deduction, exposing a weakness in joint consistency that the broad knowledge-and-reasoning score of JEVal does not isolate.

![](images/d2cdc039197c7a01b9adac90527044134c6784b54e6e0d55f0425c6e361dd505.jpg)

![](images/11502a4415381b017ec949a329d27b74db28a8689872b1d7e39315468d6b3038.jpg)  
Figure 3: Confidence calibration on 3,537 shared single-select questions with native, full-state predictions and unique maxima. The left panel groups predictions into ten equal-width confidence bins. The gray diagonal marks perfect calibration. Above it, accuracy exceeds confidence (underconfidence); below it, confidence exceeds accuracy (overconfidence). The right panel reports ECE (↓) on the same questions.

![](images/ee6f3c820d5a222b9a96a2775ef550b53ebc5873231c024a88d93f634c804a77.jpg)

![](images/5f3688b466e3e7388e9416d1e093985e1f42af60198225984841459bb22e424f.jpg)  
Figure 4: Hard Single-select Accuracy across complete-input length and option-count bins. Counts refer to benchmark items, with failures included. Each bin retains its released task composition and model-specific context and option handling. The curves describe performance under these deployment conditions.

Similar overall accuracy can hide very different strengths across tasks. Table 3 shows that WeMM-JEV performs better on world knowledge and emotion recognition, while KnowLine is stronger on workflow decisions and policy handling. Models with lower overall scores can still lead particular categories. Quyet achieves the highest accuracy on Evidence Validation, while Mapika leads Emotion Recognition. These differences show why categorylevel results matter when selecting a model for a specific workload, since a higher overall score does not guarantee better performance on the capabilities that workload actually requires.

Strong routing performance does not guarantee strong agent decision making. GPT-6 Luna Decisions outperforms Jev and Microsoft Decision-1 on Intent Routing, but performs worse on Tool and Skill Selection, Agent State Tracking, and Workflow Decision. The largest gap appears in Workflow Decision, which requires the model to account for prerequisites and action order. This shows that routing correctly does not necessarily mean the model can also choose the right action from the current agent state or coordinate a multi-step workflow.

## 4.4 Confidence Calibration

Overconfidence appears at different operating ranges. Figure 3 compares five representative models using reliability curves and expected calibration

## (a) Approval and later edits

Finch certifies the current content. Lumen and Fern modify it. Publication requires a certificate matching the final revision.

## (b) Complete set identification

Eight reports refer to one another. For example, Hazel says Maple’s report is false; Maple says Pine’s or Willow’s report is true.

<table><tr><td>Model</td><td>Selected option</td></tr><tr><td>Perplexity V1.2</td><td>Invalid (0.353)</td></tr><tr><td>Jev</td><td>Valid (0.520)</td></tr><tr><td>Microsoft Decision-1</td><td>Valid (0.894)</td></tr></table>

Later revisions invalidate an earlier approval.

Reference. all seven reports except Hazel are true under the unique consistent assignment.
<table><tr><td>Model</td><td>PHazel</td><td>min ptrue</td></tr><tr><td>Perplexity V1.2</td><td>0.083</td><td>0.075</td></tr><tr><td>Jev</td><td>0.120</td><td>0.110</td></tr><tr><td>Microsoft Decision-1</td><td>0.075</td><td>0.067</td></tr></table>

All three rank a false report above a true one.

## (c) A distribution over scores

Reference. the incomplete record is drawn with probability 3/9.
<table><tr><td>Distribution</td><td>Score 3 Score 4</td></tr><tr><td>Reference</td><td>0.333 0.667</td></tr><tr><td>Perplexity V1.2</td><td>0.649 0.331</td></tr><tr><td>Jev</td><td>0.590 0.310</td></tr><tr><td>Microsoft Decision-1</td><td>0.191 0.754</td></tr></table>

Two models reverse the dominant score probabilities.  
Figure 5: Saved model responses to three decisions involving workflow prerequisites, interdependent options, and uncertain rubric outcomes. Inputs are condensed for display; complete examples are Examples 7, 4, and 15 in Appendix F. Probabilities in (c) omit levels 0–2.

error (ECE), where lower is better. Intern-Decision is overconfident in the middle of its range. Open-Jev is closer to the diagonal there, but its highestconfidence bin averages about 0.97 confidence and 0.87 accuracy. Thus, this difference matters in deployment because the same confidence threshold may be reliable for one model but overly aggressive for another one when making decision.

Higher selection accuracy does not ensure better calibration. OpenJev has higher single-select accuracy than Intern-Decision in Table 2, yet Figure 3 gives it a larger ECE. Perplexity and Jev achieve the lowest ECE among the five representatives, while Microsoft Decision-1 has the largest error and remains below the diagonal through much of the moderate-to-high confidence range. The curves and aggregate errors together show why model selection should account for both how often a model is correct and how reliably its confidence reflects that correctness.

## 4.5 Context and Option-Space Scaling

Model rankings change between long inputs and large option sets. The two panels of Figure 4 reveal different scaling profiles. OpenJev outperforms Intern-Decision in every option-count bin, but falls below it once complete inputs exceed 8K tokens. At 8K–16K tokens, their accuracies are 15.5% and 34.5%, respectively; on 129–1,024 options, they are 26.0% and 8.3%. This reversal distinguishes performance on long-input workloads from performance on large-option workloads. MMDM measures both axes because success on one does not predict the model ordering on the other.

Large catalogs expose differences hidden by compact choices. In Figure 4, Jev and Microsoft Decision-1 differ by about seven percentage points on 2–4 options, but by more than 37 points on 129–1,024 options. Perplexity retains the highest accuracy in the largest option bin, although its advantage over Jev is much smaller there than on 9–16 options. Across the released workloads and the protocol in Section 4.1, the curves reveal both severe losses and changing relative advantages as the option space grows. A single aggregate accuracy would conceal these differences in the scale of decisions each model can handle.

## 4.6 Case Studies

Models can accept a workflow or report that conflicts with other evidence. In Figure 5(a), Perplexity V1.2 selects a workflow that approves a document before editing it. Publication requires approval of the final revision, so those later edits invalidate the certificate. Jev and Microsoft Decision-1 select a valid ordering. In Figure 5(b), Hazel’s report is false under the unique consistent interpretation, yet all three models rank it above at least one true report. The two cases expose errors in checking an option against history or other options, even when the relevant conditions are explicit.

Models struggle to translate a simple recordlevel defect into the correct score distribution. In Figure 5(c), eight of nine records satisfy every rubric criterion, while one record lacks a review stamp. A three-record sample receives the lower score only when it contains that record, which happens with probability one-third. However, Perplexity V1.2 and Jev assign more probability to the lower score, while Microsoft Decision-1 predicts the correct most likely score but still assigns the probabilities incorrectly. This example shows that even a simple scoring rule can lead to errors in both the predicted score and its probability distribution.

## 5 Conclusion

We introduced MMDM, a benchmark of 5,448 questions across 16 task categories for evaluating text-based System-1 Decision Models. With option sets ranging from 2 to 1,024 and inputs reaching approximately 64K tokens, MMDM tests decisions across both familiar requests and complex contexts. Its single-select, multi-select, and soft-label tasks assess whether models choose correctly, identify all valid options, and assign useful probabilities and scores. Our evaluation of 28 decision models reveals clear gaps between these capabilities and exposes dimensions of decision quality that are not captured by previous benchmarks. The strongest models vary across answer types and task categories. High single-select accuracy can accompany inaccurate probabilities, while deduction and routing strengths coexist with weaknesses in joint constraints and workflow decisions. Reliability curves and scaling profiles further distinguish confidence quality from accuracy and long-input performance from large-option performance. These findings direct progress toward checking interacting conditions, handling larger decision spaces, and producing probabilities that support reliable choices and scores. We release MMDM and its evaluation toolkit to make progress on these capabilities easy to measure and compare.

## Limitations

MMDM has three main limitations. First, it focuses on text-based decisions defined by a fixed state, query, and option set. Extending MMDM to multimodal inputs and interactive trajectories would broaden its coverage of real-world decision settings. Second, multi-select questions are concentrated in option logic, constraint planning, workflows, and tool or skill selection, while most soft targets come from rubric scoring. These settings provide clear labels and verifiable target distributions, but broader evaluation of subjective judgments will require references that better capture variation across human assessments. Third, MMDM combines newly constructed tasks with source-preserving adaptations, recompositions, and rewrites. Although recompositions and rewrites introduce new decision requirements and contexts, prior exposure to the underlying public source material may still occur.

## Ethics Statement

MMDM combines publicly available research data with newly constructed decision tasks. We cite the source datasets and document their contributions in Appendix B. Source materials remain subject to their original terms of use. The same appendix describes model-assisted construction, label verification, and targeted human review, including independent assessment and adjudication of disagreements. Our experiments evaluate model predictions from recorded or simulated inputs; they do not execute the tools or workflows described in the questions. We release the benchmark and evaluation toolkit, with source attribution, to support transparent and reproducible research.

## References

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367, Minneapolis, Minnesota. Association for Computational Linguistics.

AutoTrust AI Lab. 2026. JEV-27B-VL.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3119–3137, Bangkok, Thailand. Association for Computational Linguistics.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2025. LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3639–3664, Vienna, Austria. Association for Computational Linguistics.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In

Proceedings of the AAAI Conference on Artificial Intelligence.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632–642, Lisbon, Portugal. Association for Computational Linguistics.

Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,ˇ Matthew Henderson, and Ivan Vulic. 2020.´ Efficient Intent Detection with Dual Sentence Encoders. In Proceedings of the 2nd Workshop on Natural Language Processingfor Conversational AI, pages 38– 45, Online. Association for Computational Linguistics.

Tianyu Chen. 2026. NanoJev.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Ziyang Chen, Xing Wu, Junlong Jia, Chaochen Gao, Qi Fu, Debing Zhang, and Songlin Hu. 2026. Long-Bench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark. arXiv preprint arXiv:2601.02872.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv preprint arXiv:1803.05457.

Peter Clark, Oyvind Tafjord, and Kyle Richardson. 2020. Transformers as Soft Reasoners over Language. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pages 3882–3890. International Joint Conferences on Artificial Intelligence Organization.

Cloudflare. 2026. Introducing Clef-omni with Full Multimodality, plus a Faster Clef and a Cheaper Clefflash.

Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating Crosslingual Sentence Representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics.

DeepSeek-AI. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A Dataset of Fine-Grained Emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054, Online. Association for Computational Linguistics.

Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. 2026. Evaluating and Benchmarking the System One Model Jev. arXiv preprint arXiv:2609.37647.

Feiyu Duan, Jiayu Lin, Jia Wang, Jun Xiang, Jialiang Wu, Xinnong Zhang, Hanqi Yan, Siyuan Wang, and Zhongyu Wei. 2026. General Decision Models: Benchmarking and Insights Beyond Jev. arXiv preprint arXiv:2610.03935.

Nicholas Farn and Richard Shin. 2023. ToolTalk: Evaluating Tool-Usage in a Conversational Setting. arXiv preprint arXiv:2311.10775.

Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. 2023. MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4277–4302, Toronto, Canada. Association for Computational Linguistics.

Gemma Team. 2026. Gemma 4 technical report. arXiv preprint arXiv:2607.02770.

Google DeepMind. 2026. Gemini 3.8 Flash Model Card.

Alex Gu, Baptiste Roziere, Hugh James Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida Wang. 2024. CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235, pages 16568–16621. PMLR.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Pro-

ceedings ofMachine Learning Research, pages 1321– 1330, Sydney, Australia. PMLR.

H2O.ai. 2026. H2O-Lightning-4B.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations, Online.

Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natu ral Language Processing (EMNLP-IJCNLP), pages 2391–2401, Hong Kong, China. Association for Computational Linguistics.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7036–7050, Mexico City, Mexico. Association for Computational Linguistics.

Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 3441–3460, Online. Association for Computational Linguistics.

Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. 2020. QASC: A Dataset for Question Answering via Sentence Composition. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 8082–8090.

Yuta Koreeda and Christopher Manning. 2021. ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts. In Findings of the Associationfor Computational Linguistics: EMNLP 2021, pages 1907–1919, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Theodore Lee. 2026. SemIf-OpenJev.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin C. C. Chang, Fei Huang, Reynold Cheng, and Yongbin Li. 2023. Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs. In Advances in Neural Information Processing Systems, volume 36, pages 42330–42357, New Orleans, United States. Curran Associates, Inc.

Xiangyi Li, Yimin Liu, Wenbo Chen, Bingran You, Zonglin Di, Yifeng He, Shenghan Zheng, Kyoung Whan Choe, Jiankai Sun, Shuyi Wang, Chujun Tao, Binxu Li, Xuandong Zhao, Hejia Geng, Xiaojun Wu, Junwei Zhou, Xiaokun Chen, Hanwen Xing, Yubo Li, and 59 others. 2026. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv preprint arXiv:2602.12670.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, and 7 others. 2022. Competitionlevel code generation with AlphaCode. Science, 378(6624):1092–1097.

Zi Lin, Zihan Wang, Yongqi Tong, Yangkun Wang, Yuxin Guo, Yujia Wang, and Jingbo Shang. 2023. ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 4694–4702, Singapore. Association for Computational Linguistics.

Guoming Ling, Muen Xue, and Zijian Ye. 2026. Jev in the Wild: A Data-Driven Analysis of the Jev Model’s Functionality, Applications and Ecosystem. arXiv preprint arXiv:2609.30216.

Liquid AI. 2026. d1-ultra: Introducing the World’s Best Decision Model.

Dawei Liu, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, and Lichao Sun. 2026. Graph of Skills: Dependency-Aware Structural Retrieval for Massive Agent Skills. arXiv preprint arXiv:2604.05333.

Hanmeng Liu, Jian Liu, Leyang Cui, Zhiyang Teng, Nan Duan, Ming Zhou, and Yue Zhang. 2023a. LogiQA 2.0—An Improved Dataset for Logical Reasoning in Natural Language Understanding. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:2947–2962.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023b. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. In Advances in Neural Information Processing Systems.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, and 3 others. 2024. AgentBench: Evaluating LLMs as Agents. In International Conference on Learning Representations, pages 52989–53046, Vienna, Austria.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. 2025. RM-Bench: Benchmarking Reward Models of Language Models with Subtlety

and Style. In International Conference on Learning Representations.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi, and Nathan Lambert. 2026. RewardBench 2: Advancing Reward Model Evaluation. In International Conference on Learning Representations.

Mark Marosi. 2026. decider-4b: Typed Decisions with Calibrated Probabilities in One Forward Pass.

Vangelis Metsis, Ion Androutsopoulos, and Georgios Paliouras. 2006. Spam Filtering with Naive Bayes— Which Naive Bayes? In Proceedings of the Third Conference on Email and Anti-Spam, Mountain View, California, USA.

Microsoft. 2026. Introducing Microsoft-Decision-1, Our Model for Fast Decision-Making.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.

Mike Moore. 2026. Jev Benchmark: Agent Tool-Call Risk.

Carlos Navarro. 2026. ReflexBench: Benchmarking System One and Typed Decision Models.

Chinh Nguyen. 2026. Quyet-1.0-Large.

OpenAI. 2026a. Decisions.

OpenAI. 2026b. GPT-6 Astra.

Yonatan Oren, Nicole Meister, Niladri Chatterji, Faisal Ladhak, and Tatsunori B. Hashimoto. 2024. Proving Test Set Contamination in Black-Box Language Models. In International Conference on Learning Representations, Vienna, Austria.

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman. 2022. QuALITY: Question Answering with Long Input Texts, Yes! In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5336–5358, Seattle, United States. Association for Computational Linguistics.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. 2025. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 48371–48392. PMLR.

PelaAI. 2026. KnowLine-31B-Gen1: A 31B Decision Model.

Perplexity AI. 2026. Decisions API.

Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hannaneh Hajishirzi. 2025. Generalizing Verifiable Instruction Following. In Advances in Neural Information Processing Systems.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. In International Conference on Learning Representations, pages 9695–9717, Vienna, Austria.

Qwen Team. 2026a. Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All.

Qwen Team. 2026b. Qwen3.8-Max: A New Bar for Coding and Cowork.

Amin Roudaki. 2026. Decisio: Decision Inference.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. In Proceedings of the AAAI Conference on Artificial Intelligence.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense Reasoning about Social Interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463– 4473, Hong Kong, China. Association for Computational Linguistics.

Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018. CARER: Contextualized Affect Representations for Emotion Recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3687–3697, Brussels, Belgium. Association for Computational Linguistics.

Alibi Serikbay. 2026. JevK5.

Junhao Shen. 2026. OpenJev-4B.

Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, and Zhaochun Ren. 2025. Retrieval Models Aren’t Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 24497–24524, Vienna, Austria. Association for Computational Linguistics.

Stephen Solka. 2026. DecisionBench.

Florian Standhartinger. 2026. JevBench: Benchmark for AI Decision Models.

Strands Agents. 2026. Strands Decider.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, Willian Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. 2025. JudgeBench: A Benchmark for Evaluating LLM-Based Judges. In International Conference on Learning Representations.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics, 10:539–554.

TypeSafe AI. 2026. Introducing System One Models & Jev.

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024. Fresh-LLMs: Refreshing Large Language Models with Search Engine Augmentation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13697–13720, Bangkok, Thailand. Association for Computational Linguistics.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024a. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In Advances in Neural Information Processing Systems, volume 37, pages 95266–95290. Curran Associates, Inc.

Zhilin Wang, Yi Dong, Olivier Delalleau, Jiaqi Zeng, Gerald Shen, Daniel Egert, Jimmy J. Zhang, Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. 2024b. HelpSteer 2: Open-source dataset for training top-performing reward models. In Advances in Neural Information Processing Systems, volume 37, pages 1474–1501. Curran Associates, Inc.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. 2017. Crowdsourcing Multiple Choice Science Questions. In Proceedings ofthe 3rd Workshop on Noisy Usergenerated Text, pages 94–106, Copenhagen, Denmark. Association for Computational Linguistics.

WeMM-JEV Team. 2026. WeMM-JEV.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Cho Yin Yong. 2026. DecideBench: Accuracy, Cost and Latency of Decision Models.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3911–3921, Brussels, Belgium. Association for Computational Linguistics.

Z.ai. 2026. GLM-5.2: Built for Long-Horizon Tasks.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level Convolutional Networks for Text Classification. In Advances in Neural Information Processing Systems.

Zihan Zhang, Meng Fang, and Ling Chen. 2024. RetrievalQA: Assessing Adaptive Retrieval-Augmented Generation for Short-form Open-Domain Question Answering. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 6963–6975, Bangkok, Thailand. Association for Computational Linguistics.

Qihao Zhao, Yangyu Huang, Tengchao Lv, Lei Cui, Qinzheng Sun, Shaoguang Mao, Xin Zhang, Ying Xin, Qiufeng Yin, Scarlett Li, and Furu Wei. 2025. MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13371–13391, Vienna, Austria. Association for Computational Linguistics.

Lucen Zhong, Zhengxiao Du, Xiaohan Zhang, Haiyi Hu, and Jie Tang. 2025. ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario. arXiv preprint arXiv:2501.10132.

Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3277–3287, Online. Association for Computational Linguistics.

Qin Zhu, Qinyuan Cheng, Runyu Peng, Xiaonan Li, Ru Peng, Tengxiao Liu, Xipeng Qiu, and Xuanjing Huang. 2024. Inference-Time Decontamination: Reusing Leaked Benchmarks for Large Language Model Evaluation. In Findings of the Association

for Computational Linguistics: EMNLP 2024, pages 9113–9129, Miami, Florida, USA. Association for Computational Linguistics.

Tong Zhu, Zhe Xu, Zefeng He, Yunkai Chai, Xuyang Hu, Dehong Zheng, Yun Luo, Yafu Li, Qi Zhang, and Yu Cheng. 2026. Intern-Decision: Efficient multimodal decision making with autoregressive masked language modeling. arXiv preprint arXiv:2610.22131.

# Appendices

The appendices are organized as follows.

• Appendix A reports input coverage and defines the tasks.

• Appendix B describes data sources, construction, label derivation, and quality control.

• Appendix C details decision models, the Random reference, capacity handling, and supplementary metrics.

• Appendix D reports complete category results, matched cohorts, and reliability analyses.

• Appendix E describes release artifacts and reproducibility.

• Appendix F presents examples with saved model responses for all 16 categories.

## A Input Coverage and Task Definitions

Classification and Coverage. Table A1 defines the 16 capabilities and links each one to its corresponding example in Appendix F. A category identifies the capability a decision requires; the answer type specifies whether the target is one option, a complete valid set, or a distribution. We measure input length using the full text seen by the model, including the state, query, option names, and option criteria. Each example is capped at 65,535 tokens. Among the 5,448 examples, 3,155 are shorter than 1K tokens, while 356 exceed 8K tokens, including 107 longer than 32K. The number of options ranges from 2 to 1,024. At inference time, the actual token usage can be slightly higher because model providers add their own request formatting and reserved tokens.

Decision Interfaces. Jev exposes three interfaces for different kinds of requests (TypeSafe AI, 2026). They specify what the model returns. The answer types in Section 3.1 specify what makes a prediction correct and how it is evaluated.

Choice compares named options. A support request might ask whether a ticket belongs to billing, shipping, or returns, with each option’s criterion defining its responsibilities. The response contains the selected option and a probability for every option. In MMDM, this distribution supports evaluation against a single correct option, a set of equally valid options, or a soft reference, according to the task’s answer type.

Noul answers a yes/no query, such as whether a message requests a refund. It returns the probability of yes; values near one favor yes, values near zero favor no, and values near one-half express uncertainty between them. The remaining probability belongs to no, giving the two-option distribution used in evaluation. A definite reference answer is hard single-select, while a reference that quantifies uncertainty is soft-label.

Score evaluates an input against ordered, descriptive rubric levels. For report completeness, these might range from missing required evidence to meeting every requirement. The response contains a probability for each level and their probabilityweighted numeric score. MMDM evaluates both the level distribution and the expected score. Therefore, Scoring questions belong to the hard singleselect or soft-label track, depending on whether the reference identifies one level or a distribution.

## B Data Sources and Construction

## B.1 Source Collection and Adaptation

Table A2 lists every public source represented in the release, its contribution, and the decision task obtained from it. We collect labeled questions, supporting documents, response comparisons, code and database problems, agent trajectories, and tool or skill descriptions. Table A3 describes the question families that we construct from explicit rules, state transitions, and outcome mechanisms. For source-grounded rewrites, the original answer and its supporting context serve as semantic anchors. We rewrite the state and query and rephrase or replace distractors, preserving the facts and relations needed to justify the answer. The semantic and option-quality checks in Appendix B.4 verify that the revised input supports the reference answer and that incorrect options remain plausible but invalid.

Knowledge, Language, and Document. Adaptations preserve task-relevant evidence and label semantics within the state–query–options interface. Knowledge and reading tasks retain the premises needed to distinguish the answer; multi-hop questions preserve the documents and links joining the required evidence. Entailment and moderation tasks express the source labels as named options with explicit meanings. Emotion tasks preserve the source text and the relevant emotion inventory. Preference tasks retain the compared responses and the criterion governing the comparison. A preference between responses remains a selection label; numerical annotations are used according to their original meaning.

Code and Database. Program tasks preserve executable code and its input or specification.

![](images/51edabfabe3aeba53061ae46fd2d1548017f37489a749c5b5aff447f72282429.jpg)

![](images/116d4ab35a9c4f1e60b105f823bf554baf414bffc085ce38cd84bf4acde0e35a.jpg)  
Figure A1: Released input-length and option-count distributions. Complete input includes state, query, option names, and criteria under the common Qwen3.5-4B tokenizer. The state-plus-query view separates document content from catalog size.

Database tasks provide the relevant schema and database conditions. In the BIRD adaptation (Li et al., 2023), we derive a decision query from the target SQL semantics and execute every SQL option against the source database. Validation checks the required operation, the resulting records, and whether distinct options produce distinguishable outputs. An option-only similarity probe checks whether a response can be selected by superficial resemblance among SQL statements without using the query, which rules out some naive samples.

Retrieval and Freshness. Adaptive-RAG (Jeong et al., 2024) supplies retrieval-strategy decisions, while RetrievalQA (Zhang et al., 2024) supplies cases requiring a judgment about external retrieval. FreshQA (Vu et al., 2024) supplies stable, time-sensitive, and false-premise questions. Its adaptation provides an explicit evidence date and decision date, and asks whether the available evidence suffices, should be refreshed, or contains a premise that should be rejected. The target is tied to the dated evidence presented in the state.

Agent Histories and Skill Catalogs. At a decision point t, the state is the observed interaction prefix $h { \leq } t$ , and the query asks which tool or skill is appropriate next. The prefix ends before the target action. Options contain tool names and contracts or skill names and descriptions. The release includes 239 such decisions from Complex-FuncBench (Zhong et al., 2025) and ToolTalk (Farn and Shin, 2023), comprising 163 tool selections and 76 skill selections. Two separate DeepSeek-V4-Pro (DeepSeek-AI, 2026) calls review the history and every option with labels hidden and option order changed. Acceptance requires agreement on present applicability and the complete valid set.

Catalog-based questions use task and environment descriptions from SkillsBench (Li et al.,

2026), retrieval annotations from ToolRet (Shi et al., 2025), and skill descriptions from Graph of Skills (Liu et al., 2026). Generated requests introduce concrete goals and prerequisites grounded in these descriptions. An additional 31 questions are built from a collected directory of 10,798 skills with provenance preserved. For each question, we review every option for functional overlap and applicability to the current request; all valid options are included in the multi-select target. Original retrieval positives and packaged skills serve as reference candidates, while the expanded option set is independently reviewed to determine the labels.

## B.2 Original Task Construction

We construct new decision instances by specifying a mechanism, computing its reference answer, and rendering its states and conditions as natural language. The mechanism determines which details affect the decision. Distractors implement plausible mistakes, including an omitted prerequisite, an expired authorization, a stale artifact, a confused entity, or a missed exception. The final families and their validation procedures appear in Table A3.

Interdependent Options and Constrained Plans. Logic tasks define propositions whose truth values depend on other propositions. We enumerate satisfying assignments and check them with an independent truth-table implementation. Singleselect questions ask for the unique complete interpretation; multi-select questions ask for all true propositions under that interpretation. Global interpretation rules are stated in the state, and interdependent statements appear in option.criteria. We check entity references, quantifier scope, and natural-language statements against the formal propositions. Planning tasks evaluate complete proposed assignments against eligibility, scheduling, resource, and precedence constraints; all feasible plans remain valid options.

<table><tr><td>Category</td><td>Capability and example</td></tr><tr><td>WKN World knowledge</td><td>Apply physical, social, scientific, and general knowledge to a situated request. Options distinguish plausible causes, intentions, continuations, or facts. Example 1.</td></tr><tr><td>RDC Reading comprehension</td><td>Identify and connect evidence within passages, long documents, or multiple documents. Includes answerability, cross-paragraph inference, and multi-hop evidence composition.</td></tr><tr><td>DED Deductive reasoning</td><td>Example 2. Derive what follows from stated premises and rules, including entailment, quantifier scope,</td></tr><tr><td>OPL Option logic</td><td>and natural-language logical arguments. Example 3. Resolve propositions that refer to one another inside option criteria. Select either the unique</td></tr><tr><td>CPL Constraint planning</td><td>complete interpretation or every true proposition under a consistent assignment. Example 4. Identify complete plans satisfying interacting eligibility, resource, scheduling, and</td></tr><tr><td>QTR Quantitative and temporal reasoning</td><td>precedence constraints. Several plans can be feasible. Example 5. Apply practical arithmetic, table and financial evidence, temporal ordering, numeric</td></tr><tr><td>WFD Workflow decision</td><td>thresholds, and freshness requirements to a decision. Example 6. Compare complete ordered workflows under tool preconditions, artifact revisions, and certificate validity. Each option is a workflow of up to seven operations; catalogs contain at</td></tr><tr><td>AST Agent state tracking</td><td>most 128 options. Example 7. Reconstruct goal scope, artifact versions, transaction effects, evidence readiness, and</td></tr><tr><td>PEX Policies and exceptions</td><td>execution failures from events and receipts. Example 8. Apply a policy to a concrete case, resolving exceptions, scope, authority, and effective</td></tr><tr><td>EVD Evidence validation</td><td>versions; includes context-sensitive moderation. Example 9. Assess whether evidence supports, refutes, or leaves a claim unresolved, whether further</td></tr><tr><td>CDB Code and databases</td><td>retrieval is needed, and whether a response meets explicit requirements. Example 10. Determine program behavior, compare implementations, or select SQL with the required</td></tr><tr><td>IRT Intent routing</td><td>semantics using code, schema, and database context. Example 11. Identify an intent or topic and select the appropriate domain or retrieval strategy from the</td></tr><tr><td>TSS Tool and skill selection</td><td>options supplied for the request. Example 12 Select the next applicable tool or skill from its name and description, given the current request or observed agent history. Multiple independently applicable tools or skills form a</td></tr><tr><td>APF Answer preference</td><td>valid set. Example 13. Compare responses against correctness, relevance, reasoning quality, instruction following,</td></tr><tr><td>SCR Score and rubrics</td><td>or stated preference criteria. Example 14. Assign a rubric level or a distribution over levels from explicit evidence. Covers deterministic grading and uncertain outcomes under a specified sampling mechanism.</td></tr><tr><td>EMR Emotion recognition</td><td>Example 15. Identify the emotion conveyed by a text or interaction, including context-dependent and closely related emotional categories. Example 16.</td></tr></table>

Table A1: Task definitions and direct links to worked examples with saved model responses in Appendix F.

State Transitions and Workflows. State tasks reconstruct the current situation from events and tool receipts. Their mechanisms track goal scope, artifact versions, committed or rolled-back updates, evidence readiness, and execution failures. Transactional histories are checked both by folding the event sequence and by independent SQLite replay. Workflow tasks compare complete action sequences under prerequisites and revision-sensitive certificates. A simulator tracks state and content versions, while a separate checker verifies action positions and required orderings. Workflow option sets contain at most 128 sequences.

Scenario and Rule Composition. Additional original tasks combine policy conditions, evidence, tools, practical planning, and scoring in concrete scenarios. Compositional rule programs introduce thresholds, exceptions, authority levels, document versions, and joins between pieces of evidence. Natural-language renderings preserve the facts and rules that determine the computed answer. For tasks with finite uncertain outcomes, the state supplies the underlying weights, transition probabilities, or sampling process used to derive the reference distribution.

## B.3 Answer Encoding and Label Derivation

The released input follows $x = ( s , q , \mathcal { O } )$ in Equation (1). Each option has a stable identifier $\iota _ { j }$ that links its name and criterion to the reference label. The record also specifies its Choice, Noul, or Score primitive and, for Score, the numeric levels $v _ { 1 } , \ldots , v _ { K }$ . The released field question stores the query, and option.criteria stores each option’s description. Labels, categories, provenance, and verification traces are separate from the modelvisible input.

Table A2: Public sources used in the final release. Counts refer to released MMDM questions; codes follow Table 3. Catalog sources ground new requests.
<table><tr><td>Source</td><td>Items</td><td>Categories</td><td>Adaptation role</td></tr><tr><td>Adaptive-RAG (Jeong et al., 2024)</td><td>267</td><td>IRT</td><td>Route to a retrieval strategy</td></tr><tr><td>AG News (Zhang et al., 2015)</td><td>18</td><td>IRT</td><td>Topic or intent mapped to named options</td></tr><tr><td>ARC (Clark et al., 2018)</td><td>3</td><td>WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>BANKING77 (Casanueva et al., 2020)</td><td></td><td>2 IRT</td><td>Topic or intent mapped to named options</td></tr><tr><td>BFCL V4 (Patil et al., 2025)</td><td></td><td>4 TSS</td><td>Tool names and contracts as options</td></tr><tr><td>BIRD Mini-Dev (Li et al., 2023)</td><td></td><td>80 CDB</td><td>SQL decisions with database evidence</td></tr><tr><td>BoolQ (Clark et al., 2019)</td><td>52</td><td>RDC</td><td>Document evidence and comprehension</td></tr><tr><td>CodeContests (Li et al., 2022)</td><td>45</td><td>CDB</td><td>Code behavior and implementation decisions</td></tr><tr><td>ComplexFuncBench (Zhong et al., 2025)</td><td>167</td><td>TSS</td><td>Observed agent-history decision points</td></tr><tr><td>ContractNLI (Koreeda and Manning, 2021)</td><td>128</td><td>PEX</td><td>Premise-claim or rule-based judgments</td></tr><tr><td>CosmosQA (Huang et al., 2019)</td><td>75</td><td>WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>CRUXEval (Gu et al., 2024)</td><td>60</td><td>CDB</td><td>Code behavior and implementation</td></tr><tr><td>Emotion (Saravia et al., 2018)</td><td>77</td><td>EMR</td><td>decisions Emotion labels mapped to options</td></tr><tr><td>Enron-Spam (Metsis et al., 2006)</td><td>15</td><td>EVD</td><td>Safety, spam, or action-risk decisions</td></tr><tr><td>FinQA (Chen et al., 2021)</td><td>6</td><td>QTR</td><td>Practical arithmetic and table-based reasoning</td></tr><tr><td>FreshQA (Vu et al., 2024)</td><td>165</td><td>QTR</td><td>Dated evidence: refresh or reject a</td></tr><tr><td>GoEmotions (Demszky et al., 2020)</td><td>101</td><td>EMR</td><td>premise Emotion labels mapped to options</td></tr><tr><td>Graph of Skills (Liu et al., 2026) HellaSwag (Zellers et al., 2019)</td><td>47</td><td>TSS</td><td>Skill catalog grounding for new requests</td></tr><tr><td></td><td>15</td><td>WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>HelpSteer2 (Wang et al., 2024b)</td><td>70</td><td>APF</td><td>Compare responses under source criteria</td></tr><tr><td>HoVer (Jiang et al., 2020) HumanEval+ (Liu et al., 2023b)</td><td>39</td><td>RDC</td><td>Document evidence and comprehension</td></tr><tr><td></td><td></td><td>CDB</td><td>Code behavior and implementation decisions</td></tr><tr><td>IF-Multi (verifiable instructions) (Pyatkin et al., 2025)</td><td>15</td><td>EVD</td><td>Convert verifiable instruction constraints</td></tr><tr><td>JudgeBench (Tan et al., 2025)</td><td></td><td>36 APF</td><td>Compare responses under source criteria</td></tr><tr><td>LogiQA 2.0 (Liu et al., 2023a)</td><td>141</td><td>DED</td><td>Premise-claim or rule-based judgments</td></tr><tr><td>LongBench Pro (Chen et al., 2026)</td><td></td><td>RDC</td><td>Document evidence and comprehension</td></tr><tr><td>LongBench v2 (Bai et al., 2025)</td><td>164</td><td>AST, CDB, RDC</td><td>Document evidence and comprehension</td></tr><tr><td>MASSIVE (FitzGerald et al., 2023)</td><td>44</td><td>IRT</td><td>Topic or intent mapped to named options</td></tr><tr><td>MathQA (Amini et al., 2019)</td><td>50</td><td>QTR</td><td>Practical arithmetic and table-based reasoning</td></tr><tr><td>MMLU (auxiliary) (Hendrycks et al., 2021)</td><td></td><td>6 WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>MMLU-Pro (Wang et al., 2024a)</td><td></td><td>3 WKN</td><td>Knowledge and contextual answer</td></tr><tr><td>MuSiQue (Trivedi et al., 2022)</td><td>18</td><td>RDC</td><td>selection Document evidence and comprehension</td></tr><tr><td>OpenBookQA (Mihaylov et al., 2018)</td><td>10</td><td>WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>PIQA (Bisk et al., 2020)</td><td></td><td>28 WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>QASC (Khot et al., 2020)</td><td></td><td>23 WKN</td><td>Knowledge and contextual answer</td></tr><tr><td>QuALITY (Pang et al., 2022)</td><td></td><td>156 RDC</td><td>selection Document evidence and comprehension</td></tr><tr><td>RetrievalQA (Zhang et al., 2024)</td><td></td><td>154 EVD</td><td>Decide whether external retrieval is needed</td></tr><tr><td>RewardBench 2 (Malik et al., 2026)</td><td>68</td><td>APF</td><td>Compare responses under source criteria</td></tr><tr><td>RM-Bench (Liu et al., 2025)</td><td>59</td><td>APF</td><td>Compare responses under source criteria</td></tr><tr><td>RuleTaker (Clark et al., 2020)</td><td>73</td><td>DED</td><td>Premise-claim or rule-based judgments</td></tr><tr><td>SciQ (Welbl et al., 2017)</td><td>12</td><td>WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>SkillsBench (Li et al., 2026)</td><td>18</td><td>TSS</td><td>Task/environment context and skill descriptions</td></tr><tr><td>SNLI (Bowman et al., 2015)</td><td>57</td><td>DED</td><td>Premise-claim or rule-based judgments</td></tr><tr><td>SocialIQA (Sap et al., 2019)</td><td>76</td><td>WKN</td><td>Knowledge and contextual answer selection</td></tr><tr><td>Spider (Yu et al., 2018)</td><td>12</td><td>CDB</td><td>SQL decisions with database evidence</td></tr><tr><td>TAT-QA (Zhu et al., 2021)</td><td>3</td><td>QTR</td><td>Practical arithmetic and table-based reasoning</td></tr><tr><td>Tool-call risk (Moore, 2026)</td><td></td><td>3 TSS</td><td>Safety, spam, or action-risk decisions</td></tr><tr><td>ToolRet (Shi et al., 2025)</td><td></td><td>14 TSS</td><td>Tool names and contracts as options</td></tr><tr><td>ToolTalk (Farn and Shin, 2023)</td><td>72</td><td>TSS</td><td>Observed agent-history decision points</td></tr><tr><td>ToxicChat (Lin et al., 2023)</td><td>65</td><td>PEX</td><td>Safety, spam, or action-risk decisions</td></tr><tr><td>WinoGrande (Sakaguchi et al., 2020)</td><td>22</td><td>WKN</td><td>Knowledge and contextual answer</td></tr><tr><td>XNLI (Conneau et al., 2018)</td><td>38</td><td>EVD</td><td>selection Premise-claim or rule-based judgments</td></tr><tr><td>Total from public sources</td><td>2,880</td><td></td><td></td></tr></table>

Hard Single-select and Hard Multi-select. A unique correct option y is encoded as $\pi _ { j } = { \bf 1 } [ j =$ $y ]$ . A complete set of equally valid options G is encoded as $\pi _ { j } = \mathbf { 1 } [ j \in G ] / | G |$ . The latter target records known set membership, and evaluation tests whether every member ranks above every nonmember. Source labels, executable constraints, or agreement-based review determine these references according to the construction route. A deterministic rubric score uses a hard label at the corresponding numeric level.

Soft-label and Score. For finite outcomes $\omega \in$ Ω and an answer operator $r ( \omega )$ mapping each outcome to an option index, the soft reference is

$$
\pi _ { j } = \sum _ { \omega \in \Omega } P ( \omega \mid x ) \mathbf { 1 } [ r ( \omega ) = j ] .\tag{A1}
$$

For Score, option $j$ carries the numeric rubric level $v _ { j }$ . The reference follows the uncertainty explicitly specified in the task. Weighted finite worlds, state transitions, and sampling without replacement provide computable outcome probabilities.

The main Score family evaluates a common rubric over a subset of records. Records carry attributes such as signed handoffs, attachments, current templates, and review markers. For a uniform sample S of size m from a population of L records,

$$
\pi _ { j } = { \frac { | \{ S : | S | = m , \ : r ( S ) = j \} | } { \binom { L } { m } } } .\tag{A2}
$$

Every criterion is evaluated on the same sample, preserving dependencies among the checks. Exact enumeration is cross-checked by an independent combinatorial calculation. This family contributes 598 questions. Across all families, the 703 soft questions comprise 628 Score and 75 Choice or Noul questions. Every distribution records all levels, including zero-probability outcomes; Score also records the numeric scale. Example 15 illustrates this construction.

## B.4 Cleaning, Correction, and Verification

We initially collected 683,237 instances from public sources, retaining 669,135 after normalization and deduplication. We subsequently constructed 75,588 additional instances, producing a pool of 744,723 for selection.

Representation and Semantic Checks. We check unique item and option identities, complete option references, valid score scales, normalized nonnegative labels, and agreement between answer types and references. Source-parent and requestoverlap checks compare related instances with registered training, replay, and evaluation records. Answers are checked against source evidence, execution, independent formal computation, or blind review. Repairs undergo the corresponding check again; unresolved failures are excluded.

<table><tr><td>Construction family</td><td></td><td>Items Mechanism and reference derivation</td></tr><tr><td>Scenario-based decision constructions</td><td>260</td><td>Concrete policy, evidence, tool, planning, and scoring scenarios; explicit finite outcome models for soft targets.</td></tr><tr><td>Compositional rule and evidence programs</td><td>157</td><td>Thresholds, exceptions, versions, evidence joins, and output constraints; computed targets and verified natural-language renderings.</td></tr><tr><td>Single-select logical interpretations</td><td>152</td><td>Enumerate assignments to mutually referring propositions; each option gives a complete interpretation.</td></tr><tr><td>Interdependent reports</td><td>359</td><td>Enumerate the consistent assignment and select its true propositions; cross-check with an independent truth-table implementation.</td></tr><tr><td>Constraint-based plans</td><td>240</td><td>Enumerate proposed schedules and resource assignments; evaluate every feasibility condition.</td></tr><tr><td>Version-sensitive workflows</td><td>250</td><td>Simulate prerequisites, edits, certificates, and action order; independently verify positions and version requirements.</td></tr><tr><td>Context-dependent state programs</td><td>441</td><td>Track goal scope, current artifacts, tool outcomes, evidence readiness, and error</td></tr><tr><td>Transactional state histories</td><td>80</td><td>causes using event receipts. Fold updates, commits, and rollbacks; cross-check the resulting state by independent SQLite replay.</td></tr><tr><td>Sampled rubric judgments</td><td></td><td>598 Evaluate rubric conditions over sampled records; obtain the full score distribution by exact enumeration and combinatorial verification.</td></tr><tr><td>Catalog-grounded skill requests</td><td>31</td><td>Generate requests grounded in a collected skill directory; review semantically close options and preserve skill provenance.</td></tr><tr><td>Total original constructions 2,568</td><td></td><td></td></tr></table>

Table A3: Original question families, named by their construction mechanism. Catalog-grounded requests retain the provenance of their skill descriptions.

Option Quality. We further check whether the options can be distinguished from the information available in the state and query, rather than from superficial cues. Distractors should reflect realistic failure modes, such as overlooking a prerequisite or choosing a tool with similar functionality but incompatible requirements. We also mask the state and query and ask whether the answer can still be guessed from option names or descriptions alone. Options with obvious shortcuts, copied answers, or weak distinctions are removed. For SQL tasks, we additionally check whether different queries are semantically equivalent and whether surface similarity reveals the answer.

Multi-model Screening. Candidate instances are screened by DeepSeek (DeepSeek-AI, 2026), GPT-6-Astra (OpenAI, 2026b) with high reasoning effort, Gemini-3.8-Flash (Google DeepMind, 2026), Qwen3.8-Max (Qwen Team, 2026b), and GLM-5.2 (Z.ai, 2026). Each model independently judges whether an instance meets the representation, semantic, and option-quality standards above. Four or five accept votes support automatic inclusion; a 3–2 vote sends the instance to human review, and fewer than three accept votes exclude it. This stage accepted 4,715 instances automatically and referred 928 borderline cases for human review.

Independent Human Review. Two reviewers independently assessed all 928 borderline cases under a common standard operating procedure (SOP). The SOP applies the representation, semantic, and option-quality standards above, using source evidence, verification traces, and option-only probe results from the earlier checks. Reviewers verify that the state and query support the reference answer, that all valid options are represented, and that distractors reflect plausible errors rather than superficial cues. Their initial decisions agreed on 97.3% of cases; a third reviewer adjudicated the remaining disagreements. Corrections underwent the corresponding semantic or executable checks before inclusion. Combining automatically accepted and human-reviewed instances produced the final 5,448 questions.

## C Models and Evaluation Details

## C.1 Decision Models and Random Reference

Table A4 lists the selected decision models, release references, and deployment settings. Selection requires direct option-probability output through a decision-scoring head or a single decoding step, without generated reasoning or adaptive thinking. The run settings record model versions, tokenizers, context budgets, service and output reserves, option caps, code versions, and grouping seeds. Decisio entries identify both the inference implementation and base checkpoint. Limits reflect the model and serving interface.

<table><tr><td>Decision Model and release reference</td><td>Context budget</td><td>Cap</td><td>Counter</td><td>Native</td><td>Grouped</td></tr><tr><td>Autotrust JEV 27B-VL (AutoTrust AI Lab, 2026)</td><td>65536</td><td>256</td><td>provider</td><td>5380</td><td>68</td></tr><tr><td>Clef (Cloudflare, 2026)</td><td>65536</td><td>255</td><td>proxy</td><td>5354</td><td>94</td></tr><tr><td>Clef-Flash (Cloudflare, 2026)</td><td>24576</td><td>255</td><td>proxy</td><td>5422</td><td>26</td></tr><tr><td>Clef-Omni (Cloudflare, 2026)</td><td>65536</td><td>255</td><td>proxy</td><td>5354</td><td>94</td></tr><tr><td>Decisio-Gemma-4-12B (Gemma Team, 2026; Roudaki, 2026)</td><td>32768</td><td>255</td><td>provider</td><td>5396</td><td>52</td></tr><tr><td>Decisio-Gemma-4-31B (Gemma Team, 2026; Roudaki, 2026)</td><td>32768</td><td>255</td><td>provider</td><td>5396</td><td>52</td></tr><tr><td>Decisio-Qwen3.6-35B-A3B (Qwen Team, 2026a; Roudaki, 2026)</td><td>32768</td><td>255</td><td>provider</td><td>5396</td><td>52</td></tr><tr><td>GPT-6 Luna Decisions (OpenAI, 2026a)</td><td>Full input</td><td>255</td><td>provider</td><td>5354</td><td>94</td></tr><tr><td>H2O-Lightning 4B (H2O.ai, 2026)</td><td>32000</td><td>255</td><td>provider</td><td>5396</td><td>52</td></tr><tr><td>Intern-Decision 0.8B (Zhu et al., 2026)</td><td>32768</td><td>62</td><td>provider</td><td>5186</td><td>262</td></tr><tr><td>Intern-Decision 2B (Zhu et al., 2026)</td><td>32768</td><td>62</td><td>provider</td><td>5186</td><td>262</td></tr><tr><td>Intern-Decision 4B (Zhu et al., 2026)</td><td>32768</td><td>62</td><td>provider</td><td>5186</td><td>262</td></tr><tr><td>Jev 1.14.0 (TypeSafe AI, 2026)</td><td>262144</td><td>255</td><td>proxy</td><td>5354</td><td>94</td></tr><tr><td>JevK5 4B (Serikbay, 2026)</td><td>32768</td><td>16</td><td>provider</td><td>4908</td><td>540</td></tr><tr><td>KnowLine 31B Gen1 (PelaAI, 2026)</td><td>262144</td><td>255</td><td>provider</td><td>5354</td><td>94</td></tr><tr><td>Liquid-d1 (Liquid AI, 2026)</td><td>65536</td><td>255</td><td>proxy</td><td>5354</td><td>94</td></tr><tr><td>Liquid-d1-Ultra (Liquid AI, 2026)</td><td>32768</td><td>255</td><td>proxy</td><td>5396</td><td>52</td></tr><tr><td>Mapika Decider 4B (Marosi, 2026)</td><td>32768</td><td>255</td><td>provider</td><td>5396</td><td>52</td></tr><tr><td>Microsoft Decision-1 (Microsoft, 2026)</td><td>32768</td><td>255</td><td>proxy</td><td>5396</td><td>52</td></tr><tr><td>NanoJev 0.6B (Chen, 2026)</td><td>32768</td><td>100000</td><td>provider</td><td>5448</td><td>0</td></tr><tr><td>OpenJev 4B (Shen, 2026)</td><td>65536</td><td>100000</td><td>provider</td><td>5448</td><td>0</td></tr><tr><td>Perplexity Decider-Flash-V1.2 (Perplexity AI, 2026)</td><td>262144</td><td>255</td><td>proxy</td><td>5354</td><td>94</td></tr><tr><td>Perplexity Decider-V1 (Perplexity AI, 2026)</td><td>8192</td><td>255</td><td>provider</td><td>5448</td><td>0</td></tr><tr><td>Perplexity Decider-V1.2 (Perplexity AI, 2026)</td><td>262144</td><td>255</td><td>proxy</td><td>5354</td><td>94</td></tr><tr><td>Quyet-1.0-Large (Nguyen, 2026)</td><td>8000</td><td>10</td><td>provider</td><td>4552</td><td>896</td></tr><tr><td>SemIf 4B (Lee, 2026)</td><td>32768</td><td>16</td><td>provider</td><td>4908</td><td>540</td></tr><tr><td>Strands-Decider 2B (Strands Agents, 2026)</td><td>32768</td><td>100000</td><td>provider</td><td>5448</td><td>0</td></tr><tr><td>WeMM-JEV 27B (WeMM-JEV Team, 2026)</td><td>262144</td><td>255</td><td>provider</td><td>5354</td><td>94</td></tr></table>

Table A4: Evaluated decision models. Full input indicates no evaluator-side truncation. Context budgets and option caps are run settings; the full run identity pins service/model versions, tokenizers, adapters, and reserves. A cap of 100,000 denotes the configured effectively uncapped native path, not a measured architectural limit.

Random supplies a heuristic point of reference by assigning equal probability to all K options. We report analytical expected accuracies of $1 / K$ for single-select and $1 / \binom { K } { | G | }$ for multi-select under uniform random tie-breaking. Its uniform distribution is used to compute SB-F, Soft Brier, and Score nMAE. Random has SB-F zero on every released multi-select question, since each contains at least one incorrect option.

## C.2 Protocol and Metrics

Context and Option Handling. For context budget B, service allowance H, and output reserve $R ,$ the adapter preserves the query, all options, and their criteria. It keeps approximately equal token spans from the start and end of the state and recounts the serialized request against $B - H - R$ Truncation preserves Unicode boundaries and inserts an omission marker. Runs record the provider or proxy tokenizer. Fixed-field overflow and provider rejection count as failures; grouped calls share the prepared state. All requests follow the protocol in Section 4.1. A supported catalog is evaluated in one call. An independent catalog above the option cap is reduced recursively, with each call presenting all options in its group alongside the same prepared state. The highest-probability option in each group advances for hard single-select. For hard multi-select, the evaluator uses $N = | G |$ both as the evaluation cutoff and as the maximum number retained from each group; neither N nor the reference option identities enter model requests. The metric evaluates complete-set ordering at this reference cardinality. Option-dependent and distributional tasks retain the complete option set without splitting. If that set exceeds a model’s option cap, the request is recorded as a capacity failure.

Implementation and Averaging. Hard accuracies include all questions and count boundary ties, failed requests, and missing outputs as incorrect. The implementation treats probability differences of at most $1 0 ^ { - 1 2 }$ as ties. SB-F, Soft Brier, and Score nMAE use valid native outputs over the complete option set. Each Score error is normalized by that item’s rubric range.

Confidence Calibration. We also measure how well single-select confidence predicts correctness using expected calibration error (ECE) (Guo et al., 2017). For L predictions partitioned into ten equalwidth confidence bins $B _ { b }$

$$
\mathrm { E C E } = \sum _ { b = 1 } ^ { 1 0 } \frac { | B _ { b } | } { L } \left| \bar { a } _ { b } - \bar { u } _ { b } \right| ,\tag{A3}
$$

where $\bar { a } _ { b }$ is mean correctness and $\bar { u } _ { b }$ is mean maximum option probability in bin $b ;$ empty bins contribute zero. Lower ECE indicates closer agreement between confidence and accuracy. Matched calibration analyses use the same single-select questions with valid, native, full-state predictions and unique maxima across the compared models.

<table><tr><td rowspan="2">Decision Model</td><td colspan="4">Hard Multi-select Accuracy (%, ↑) </td><td colspan="6">Soft Brier (↓)</td></tr><tr><td>RDC</td><td>OPL</td><td>CPL</td><td>WFD</td><td>TSS CPL</td><td>QTR</td><td>PEX</td><td>EVD</td><td>TSS</td><td>SCR</td></tr><tr><td colspan="9">Heuristic reference</td><td></td></tr><tr><td>Random (uniform)</td><td>6.67</td><td>4.40</td><td>0.61</td><td>0.59 3.71</td><td>0.0026</td><td>0.0821</td><td>0.0334</td><td>0.1039</td><td>0.0816</td><td>0.2206</td></tr><tr><td colspan="9">Closed-source decision models</td></tr><tr><td>Jev 1.14.0</td><td>100.00</td><td>5.01</td><td>22.50</td><td>21.35</td><td>80.58 0.0425</td><td>0.0453</td><td>0.0374</td><td>0.0275</td><td>0.0937</td><td>0.1170</td></tr><tr><td>Microsoft Decision-1</td><td>50.00</td><td>10.86</td><td>8.75</td><td>16.85</td><td>74.76</td><td>0.2496 0.1475</td><td>0.1133</td><td>0.1371</td><td>0.1647</td><td>0.2959</td></tr><tr><td>Liquid-d1-Ultra</td><td>100.00</td><td>6.13</td><td>18.75</td><td>29.21</td><td>86.41</td><td>0.0073 0.1663</td><td>0.0948</td><td>0.0851</td><td>0.1584</td><td>0.1334626</td></tr><tr><td>Liquid-d1</td><td>0.00</td><td>7.52</td><td>21.25</td><td>19.10</td><td>75.73</td><td>0.1689 0.2351</td><td>0.1080</td><td>0.1886</td><td>0.2124</td><td>0.2946625</td></tr><tr><td>GPT-6 Luna Decisions</td><td>50.00</td><td>3.06</td><td>6.25</td><td>6.74</td><td>51.46</td><td>0.1994 0.2015</td><td>0.1112</td><td>0.156220</td><td>0.134821</td><td>0.3548627</td></tr><tr><td colspan="9">Open-source decision models</td></tr><tr><td>WeMM-JEV 27B</td><td>100.00</td><td>20.61</td><td>26.25</td><td>47.19</td><td>91.26</td><td>0.1950 0.1477</td><td>0.1102</td><td>0.1227</td><td>0.2204</td><td>0.2422</td></tr><tr><td>KnowLine 31B Gen1</td><td>100.00</td><td>14.48</td><td>35.00</td><td>40.45</td><td>84.47</td><td>0.1332 0.1466</td><td>0.0672</td><td>0.0682</td><td>0.1217</td><td>0.1265</td></tr><tr><td>Perplexity Decider-V1.2</td><td>100.00</td><td>19.78</td><td>25.00</td><td>37.08</td><td>85.44</td><td>0.3949 0.0737</td><td>0.0776</td><td>0.0632</td><td>0.1840</td><td>0.1767626</td></tr><tr><td>Autotrust JEV 27B-VL</td><td>100.00</td><td>9.47</td><td>8.75</td><td>13.48</td><td>70.87</td><td>0.1783 0.2352</td><td>0.0428</td><td>0.1658</td><td>0.1331</td><td>0.3647</td></tr><tr><td>Decisio-Gemma-4-31B</td><td>100.00</td><td>5.57</td><td>717.50</td><td>10.11</td><td>50.49</td><td>0.1046 0.3715</td><td>0.3445</td><td>0.2181</td><td>0.2452</td><td>0.1516</td></tr><tr><td>Clef</td><td>100.00</td><td>5.01</td><td>15.00</td><td>8.99</td><td>62.14</td><td>0.0438 0.1101</td><td>0.1418</td><td>0.1916</td><td>0.1686</td><td>0.1403</td></tr><tr><td>Perplexity Decider-V1</td><td>50.00</td><td>10.86</td><td>17.50</td><td>5.62</td><td>74.76</td><td>0.0357 0.2022</td><td>0.0715</td><td>0.0671</td><td>0.1062</td><td>0.1621</td></tr><tr><td>Perplexity Decider-Flash-V1.2</td><td>100.00</td><td>3.90</td><td>5.00</td><td>11.24</td><td>70.87</td><td>0.0744 0.1575</td><td>0.0331</td><td>0.1470</td><td>0.0937</td><td>0.3050625</td></tr><tr><td>H2O-Lightning 4B</td><td>50.00</td><td>11.42</td><td>12.50</td><td>10.11</td><td>66.99</td><td>0.0103 0.1128</td><td>0.1219</td><td>0.2102</td><td>0.1349</td><td>0.3720</td></tr><tr><td>Decisio-Gemma-4-12B</td><td>100.00</td><td>4.46</td><td>22.50</td><td>10.11</td><td>67.96</td><td>0.0169</td><td>0.21400.1612</td><td>0.1034</td><td>0.1493</td><td>0.1646</td></tr><tr><td>Quyet-1.0-Large</td><td>50.00</td><td>4.18</td><td>30.00</td><td>14.61</td><td>50.49</td><td>0.2222 0.1660</td><td>0.1563</td><td>0.100321</td><td>0.137519</td><td>0.1853</td></tr><tr><td>OpenJev 4B</td><td>0.00</td><td>5.29</td><td>18.75</td><td>12.36</td><td>63.11</td><td>0.0070 0.2168</td><td>0.0423</td><td>0.1066</td><td>0.0681</td><td>0.4012</td></tr><tr><td>Clef-Omni</td><td>100.00</td><td>4.18</td><td>316.25</td><td>7.87</td><td>55.34</td><td>0.0470 0.2112</td><td>0.0780</td><td>0.245720</td><td>0.150421</td><td>0.2886625</td></tr><tr><td>Mapika Decider 4B</td><td>50.00</td><td>5.57</td><td>8.75</td><td>8.99</td><td>51.46</td><td>0.1338 0.1100</td><td>0.1128</td><td>0.1517</td><td>0.1366</td><td>0.2229</td></tr><tr><td>Clef-Flash</td><td>100.00</td><td>3.90</td><td>12.50</td><td>10.11</td><td>57.28</td><td>0.0286 0.2906</td><td>0.0722</td><td>0.1677</td><td>0.1227</td><td>0.3799</td></tr><tr><td>Decisio-Qwen3.6-35B-A3B</td><td>100.00</td><td>6.13</td><td>8.75</td><td>11.24</td><td>59.22</td><td>0.2216 0.1718</td><td>0.1177</td><td>0.1222</td><td>0.1128</td><td>0.2706</td></tr><tr><td>Intern-Decision 4B</td><td>50.00</td><td>4.46</td><td>5.00</td><td>4.49</td><td>43.69</td><td>0.0258 0.1743</td><td>0.0690</td><td>0.0883</td><td>0.0836</td><td>0.1637</td></tr><tr><td>JevK5 4B</td><td>50.00</td><td>5.01</td><td>7.50</td><td>1.12</td><td>38.83</td><td>0.0251 0.1283</td><td>0.0916</td><td>0.0773</td><td>0.1304</td><td>0.2441</td></tr><tr><td>Strands-Decider 2B</td><td>50.00</td><td>3.34</td><td>7.50</td><td>1.12</td><td>31.07</td><td>0.0819 0.1483</td><td>0.0719</td><td>0.0822</td><td>0.1078</td><td>0.2181</td></tr><tr><td>SemIf 4B</td><td>50.00</td><td>3.06</td><td>2.50</td><td>0.00</td><td>25.24</td><td>0.0026 0.2521</td><td>0.0952</td><td>0.1107</td><td>0.1829</td><td>0.3630</td></tr><tr><td>Intern-Decision 2B</td><td>50.00</td><td>3.34</td><td>1.25</td><td>0.00</td><td>23.30</td><td>0.0001 0.1347</td><td>0.0559</td><td>0.0954</td><td>0.1117</td><td>0.2137</td></tr><tr><td>Intern-Decision 0.8B</td><td>50.00</td><td>3.06</td><td>0.00</td><td>1.12</td><td>11.65</td><td>0.0037 0.0727</td><td>0.0362</td><td>0.0987</td><td>0.0909</td><td>0.2205</td></tr><tr><td>NanoJev 0.6B</td><td>0.00</td><td>10.86</td><td>0.00</td><td>0.00</td><td>5.83</td><td>0.0616 0.1626</td><td>0.0585</td><td>0.1885</td><td>0.2123</td><td>0.3237</td></tr></table>

Table A5: Category results for multi-select and soft-label questions. RDC denotes Reading Comprehension; OPL, Option Logic; CPL, Constraint Planning; WFD, Workflow Decision; TSS, Tool and Skill Selection; QTR, Quantitative and Temporal Reasoning; PEX, Policies and Exceptions; EVD, Evidence Validation; and SCR, Score and Rubrics. Multi-select totals are 2, 359, 80, 89, and 103 in column order; soft-label totals are 1, 12, 18, 22, 22, and 628. Soft Brier uses valid native outputs; superscripts give counts below the category total. Colors mark the best and second-best model values within each column, as in Table 2.

Unified Model Output. Every decision model returns a probability distribution over the options supplied in its request. The answer-type annotation and reference number of correct options are not included in the model input. Thus, singleselect, multi-select, and soft-label questions share the same prediction interface, without an additional prompt specifying the target answer type or set size. Reference labels are only used by the evaluator

## D Additional Experimental Results

## D.1 Results by Task Category

Table A5 extends Table 3 to every category with multi-select or soft-label questions. The two blocks distinguish exact multi-select accuracy from distributional error. Jev achieves the lowest Soft Brier in three of the six categories, while the strongest multi-select results vary across models and tasks.

## D.2 Matched-Cohort Analysis

Table A6 compares the five representative models on shared questions. For each metric, we retain questions for which all five models produced valid, native predictions using the complete input. Calibration also requires a unique highest-probability option. With the questions and input handling held fixed, Perplexity V1.2 leads both hard-answer metrics, while Jev has the lowest Soft Brier and Score nMAE. The contrast between selection accuracy and probability quality therefore remains on shared inputs. On the shared single-select questions, Perplexity V1.2 exceeds Jev by 7.93 percentage points (95% CI [6.41, 9.44]), while Jev’s Soft Brier is lower by 0.0592 on the shared soft-label questions (95% CI [0.0515, 0.0661]). We estimate both confidence intervals from 2,000 paired bootstrap resamples of source-parent groups, computing itemweighted means within each resample.

<table><tr><td>Decision Model</td><td>Hard Single-select Acc.↑</td><td>Hard Multi-select Acc.↑</td><td>Soft Brier↓</td><td>Score nMAE↓</td><td>SB-F↑</td><td>ECE↓</td></tr><tr><td>Perplexity V1.2</td><td>76.66</td><td>36.90</td><td>0.1694</td><td>0.1641</td><td>0.1990</td><td>0.046</td></tr><tr><td>Jev 1.14.0</td><td>68.73</td><td>22.75</td><td>0.1101</td><td>0.0927</td><td>0.1198</td><td>0.054</td></tr><tr><td>Microsoft Decision-1</td><td>64.27</td><td>24.67</td><td>0.2794</td><td>0.1836</td><td>0.1260</td><td>0.125</td></tr><tr><td>Intern-Decision 4B</td><td>51.70</td><td>11.85</td><td>0.1565</td><td>0.1450</td><td>0.0845</td><td>0.085</td></tr><tr><td>OpenJev 4B</td><td>61.82</td><td>18.93</td><td>0.3685</td><td>0.3105</td><td>0.1183</td><td>0.092</td></tr></table>

Table A6: Results on shared native, full-state inputs. Cohorts contain 3,706 single-select, 523 multi-select, 701 soft-label, and 661 Score questions; ECE uses 3,537 single-select questions with unique maxima. Accuracies are percentages. Each metric uses the same questions across models; SB-F uses the 523-question multi-select cohort.

<table><tr><td>Decision Model</td><td>OPL</td><td>CPL</td><td>WFD</td><td>TSS</td></tr><tr><td>Perplexity V1.2</td><td>19.78</td><td>25.00</td><td>37.08</td><td>85.44</td></tr><tr><td>Jev 1.14.0</td><td>5.01</td><td>22.50</td><td>21.35</td><td>80.58</td></tr><tr><td>Microsoft Decision-1</td><td>10.86</td><td>8.75</td><td>16.85</td><td>74.76</td></tr><tr><td>Intern-Decision 4B</td><td>4.46</td><td>5.00</td><td>4.49</td><td>43.69</td></tr><tr><td>OpenJev 4B</td><td>5.29</td><td>18.75</td><td>12.36</td><td>63.11</td></tr></table>

Table A7: Hard Multi-select Accuracy (%, ↑) for representative models. OPL denotes Option Logic (359 questions); CPL, Constraint Planning (80); WFD, Workflow Decision (89); and TSS, Tool and Skill Selection (103). The two Reading Comprehension questions are reported in Table A5.
<table><tr><td colspan="2">Native</td><td colspan="2">Grouped</td></tr><tr><td>Decision Model</td><td>Items</td><td>Acc.</td><td>Items Acc.</td></tr><tr><td>Perplexity V1.2</td><td>605</td><td>35.04</td><td>28 7.14</td></tr><tr><td>Jev 1.14.0</td><td>605 22.81</td><td>28</td><td>7.14</td></tr><tr><td>Microsoft Decision-1</td><td>611 22.75</td><td>22</td><td>0.00</td></tr><tr><td>Intern-Decision 4B</td><td>541 12.75</td><td>92</td><td>1.09</td></tr><tr><td>OpenJev 4B</td><td>633</td><td>17.38</td><td>0 1</td></tr></table>

Table A8: Hard Multi-select Accuracy (%, ↑) by inference mode under Section 4.1. Failures count as incorrect. Native and grouped cohorts are determined by each model's option cap and therefore differ across models; – denotes an empty cohort.

## D.3 Multi-select Results

Table A7 focuses on the four substantial multiselect categories for the same five models. All five perform best on Tool and Skill Selection, where options describe independently applicable capabilities. Option Logic, Constraint Planning, and Workflow Decision instead require resolving dependencies or validating complete plans, and their accuracies remain much lower. This contrast helps explain why a model’s overall multi-select score depends on the task composition.

Table A8 breaks down the same protocol by whether the complete option set fits the model’s option cap. Accuracy remains low on catalogs requiring recursive reduction. These larger catalogs differ across models because their caps differ, so Table A6 complements this breakdown with 523 shared multi-select questions evaluated in one call with the complete state and option set. The strongest result on this common cohort is 36.90%, showing that the multi-select difficulty persists without option reduction. SB-F complements exact accuracy by measuring the strength and balance of support for the correct set.

## D.4 Multi-select Separation and Balance

We further examine the probability distributions behind these multi-select results. Table A9 reports SB-F, separation, and balance on 284 questions shared by all 28 decision models, with results also broken down by category.

Weak Separation. WeMM-JEV and Perplexity V1.2 have the highest SB-F (0.1905 and 0.1900, respectively), but excel on different tasks. WeMM-JEV is stronger on Option Logic and Constraint Planning, while Perplexity performs better on Workflow Decision and Tool and Skill Selection. We average each metric over questions for each model. The medians across models are 0.113 for separation and 0.862 for balance, while median correct-option coverage is only 37.8%. Models therefore often distribute probability evenly among correct options without separating them from incorrect ones. Jev illustrates this pattern, with average balance of 0.915 and separation of 0.137. OpenJev has stronger separation but lower balance. Jev’s category results make the distinction clearer. Its SB-F falls from 0.465 on Tool and Skill Selection to 0.022 on Option Logic, where balance remains high (0.973) but separation is low (0.023). The main difficulty in these questions is distinguishing interdependent true statements from false ones.

<table><tr><td rowspan="2"></td><td rowspan="2">n</td><td rowspan="2">Rank</td><td colspan="3">Overall</td><td rowspan="2"></td><td colspan="3">SB-F by category</td></tr><tr><td>SB-F↑</td><td>S</td><td>B</td><td>OPL</td><td>CPL WFD</td><td>TSS</td></tr><tr><td colspan="9">Heuristic reference</td></tr><tr><td>Random (uniform)</td><td>633</td><td>-1</td><td>0.0000</td><td>0.000</td><td>1.000 </td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td colspan="10">Closed-source decision models</td></tr><tr><td>Jev 1.14.0</td><td>605</td><td>11</td><td>0.1057</td><td>0.137</td><td>0.915</td><td>0.022</td><td>0.244</td><td>0.261</td><td>0.465</td></tr><tr><td>Microsoft Decision-1</td><td>601</td><td>8</td><td>0.1194</td><td>0.169</td><td>0.812</td><td>0.069</td><td>0.186</td><td>0.212</td><td>0.339</td></tr><tr><td>Liquid-d1-Ultra</td><td>604</td><td>4</td><td>0.1521</td><td>0.192</td><td>0.801</td><td>0.056</td><td>0.304</td><td>0.399</td><td>0.527</td></tr><tr><td>Liquid-d1</td><td>604</td><td>9</td><td>0.1183</td><td>0.156</td><td>0.848</td><td>0.049</td><td>0.226</td><td>0.262</td><td>0.411</td></tr><tr><td>GPT-6 Luna Decisions</td><td>596</td><td>21</td><td>0.0800</td><td>0.112</td><td>0.838</td><td>0.043</td><td>0.151</td><td>0.105</td><td>0.260</td></tr><tr><td colspan="10">Open-source decision models</td></tr><tr><td>WeMM-JEV 27B</td><td>605</td><td>1</td><td>0.1905</td><td>0.259</td><td>0.780</td><td>0.123</td><td>0.516</td><td>0.408</td><td>0.342</td></tr><tr><td>KnowLine 31B Gen1</td><td>605</td><td>3</td><td>0.1795</td><td>0.258</td><td>0.678</td><td>0.102</td><td>0.500</td><td>0.335</td><td>0.429</td></tr><tr><td>Perplexity Decider-V1.2</td><td>602</td><td>2</td><td>0.1900</td><td>0.244</td><td>0.808</td><td>0.110</td><td>0.386</td><td>0.436</td><td>0.450</td></tr><tr><td>Autotrust JEV 27B-VL</td><td>623</td><td>6</td><td>0.1242</td><td>0.164</td><td>0.770</td><td>0.064</td><td>0.239</td><td>0.186</td><td>0.405</td></tr><tr><td>Decisio-Gemma-4-31B</td><td>605</td><td>12</td><td>0.0959</td><td>0.126</td><td>0.838</td><td>0.046</td><td>0.168</td><td>0.115</td><td>0.356</td></tr><tr><td>Clef</td><td>601</td><td>15</td><td>0.0948</td><td>0.109</td><td>0.904</td><td>0.036</td><td>0.219</td><td>0.113</td><td>0.384</td></tr><tr><td>Perplexity Decider-V1</td><td>593</td><td>5</td><td>0.1290</td><td>0.158</td><td>0.874</td><td>0.067</td><td>0.244</td><td>0.131</td><td>0.453</td></tr><tr><td>Perplexity Decider-Flash-V1.2</td><td>600</td><td>17</td><td>0.0853</td><td>0.112</td><td>0.915</td><td>0.018</td><td>0.118</td><td>0.190</td><td>0.416</td></tr><tr><td>H2O-Lightning 4B</td><td>605</td><td>13</td><td>0.0956</td><td>0.114</td><td>0.923</td><td>0.029</td><td>0.189</td><td>0.158</td><td>0.421</td></tr><tr><td>Decisio-Gemma-4-12B</td><td>605</td><td>18</td><td>0.0851</td><td>0.113</td><td>0.891</td><td>0.027</td><td>0.216</td><td>0.106</td><td>0.371</td></tr><tr><td>Quyet-1.0-Large</td><td>461</td><td>14</td><td>0.0953</td><td>0.135</td><td>0.869</td><td>0.023</td><td>0.369</td><td>0.178</td><td>0.373</td></tr><tr><td>OpenJev 4B</td><td>620</td><td>10</td><td>0.1102</td><td>0.155</td><td>0.693</td><td>0.060</td><td>0.192</td><td>0.125</td><td>0.368</td></tr><tr><td>Clef-Omni</td><td>601</td><td>22</td><td>0.0754</td><td>0.090</td><td>0.916</td><td>0.031</td><td>0.158</td><td>0.147</td><td>0.270</td></tr><tr><td>Mapika Decider 4B</td><td>605</td><td>20</td><td>0.0800</td><td>0.112</td><td>0.845</td><td>0.030</td><td>0.187</td><td>0.137</td><td>0.303</td></tr><tr><td>Clef-Flash</td><td>601</td><td>19</td><td>0.0800</td><td>0.095</td><td>0.899</td><td>0.039</td><td>0.111</td><td>0.144</td><td>0.276</td></tr><tr><td>Decisio-Qwen3.6-35B-A3B</td><td>605</td><td>7</td><td>0.1202</td><td>0.157</td><td>0.762</td><td>0.059</td><td>0.144</td><td>0.163</td><td>0.452</td></tr><tr><td>Intern-Decision 4B</td><td>535</td><td>16</td><td>0.0880</td><td>0.110</td><td>0.854</td><td>0.038</td><td>0.135</td><td>0.122</td><td>0.350</td></tr><tr><td>JevK5 4B</td><td>501</td><td>23</td><td>0.0738</td><td>0.093</td><td>0.880</td><td>0.032</td><td>0.128</td><td>0.062</td><td>0.310</td></tr><tr><td>Strands-Decider 2B</td><td>605</td><td>26</td><td>0.0588</td><td>0.069</td><td>0.877</td><td>0.030</td><td>0.142</td><td>0.034</td><td>0.214</td></tr><tr><td>SemIf 4B</td><td>501</td><td>24</td><td>0.0716</td><td>0.105</td><td>0.718</td><td>0.052</td><td>0.069</td><td>0.040</td><td>0.208</td></tr><tr><td>Intern-Decision 2B</td><td>535</td><td>25</td><td>0.0589</td><td>0.072</td><td>0.874</td><td>0.033</td><td>0.062</td><td>0.034</td><td>0.226</td></tr><tr><td>Intern-Decision 0.8B</td><td>535</td><td>28</td><td>0.0363</td><td>0.045</td><td>0.870</td><td>0.034 0.054</td><td>0.005</td><td>0.014</td><td>0.075</td></tr><tr><td>NanoJev 0.6B</td><td>605</td><td>27</td><td>0.0429</td><td>0.047</td><td>0.915</td><td></td><td>0.009</td><td>0.005</td><td>0.013</td></tr></table>

Table A9: SB-F on a fixed common cohort of 284 native multi-select predictions per model. n gives the valid native multi-select sample count used for each model in Table 2; the statistics here use the common cohort. Ranks compare the 28 decision models using unrounded SB-F. OPL denotes Option Logic (214 questions); CPL, Constraint Planning (14); WFD, Workflow Decision (20); and TSS, Tool and Skill Selection (36). S and B are mean separation and correct-set balance; SB-F averages their per-question products. Colors mark the best and second-best SB-F within each column.

Coverage and Probability Margins. Table A10 examines these errors on the same 284 questions. A valid top choice satisfies $\operatorname* { m a x } _ { i \in G } p _ { i } > b ,$ where b is the largest incorrect-option probability. Coverage measures the fraction of correct options above b, averaged across questions. Probability differences of at most $1 0 ^ { - 1 2 }$ count as ties. GPT-6 Luna Decisions and Jev rank a valid option first on 184 and 181 questions, respectively, but place the entire correct set above all distractors on only 36 and 57. Recognizing one valid option thus frequently leaves other valid options behind.

Even when the full set is correctly ordered, the probability margin can be narrow. We define this as $b \ge { \frac { 1 } { 2 } } \operatorname* { m i n } _ { i \in G } p _ { i }$ , meaning that the strongest distractor receives at least half the probability of the weakest correct option. This occurs in 20 of Luna’s 36 successes and 43 of WeMM-JEV’s 110. Across all 28 models, the median proportion is 45.6%. SB-F captures this weakness in probability separation even when exact accuracy counts the prediction as correct.

<table><tr><td>Decision Model</td><td>Valid top choice (%)</td><td>Multi-select Acc. (%)</td><td>Coverage (%)</td><td>Narrow margins (count)</td></tr><tr><td>Random (uniform)</td><td>51.7</td><td>4.2</td><td>0.0</td><td></td></tr><tr><td>WeMM-JEV 27B</td><td>88.7</td><td>38.7</td><td>64.7</td><td>43/110</td></tr><tr><td>Perplexity Decider-V1.2</td><td>84.9</td><td>34.9</td><td>59.8</td><td>30/99</td></tr><tr><td>Jev 1.14.0</td><td>63.7</td><td>20.1</td><td>38.2</td><td>14/57</td></tr><tr><td>Microsoft Decision-1</td><td>71.5</td><td>23.2</td><td>45.8</td><td>24/66</td></tr><tr><td>GPT-6 Luna Decisions</td><td>64.8</td><td>12.7</td><td>34.8</td><td>20/36</td></tr><tr><td>OpenJev 4B</td><td>74.3</td><td>19.7</td><td>42.8</td><td>24/56</td></tr><tr><td>Intern-Decision 4B</td><td>59.5</td><td>14.1</td><td>32.9</td><td>21/40</td></tr></table>

Table A10: Diagnostics on the same 284 native multi-select questions. Valid top choice checks whether the highestranked option is correct. Coverage averages the fraction of correct options outranking the strongest distractor. Narrow margins count complete-set successes in which the strongest distractor has at least half the probability of the weakest correct option; the denominator is the number of complete-set successes. Random uses analytical uniform tie-breaking for the first two columns.
<table><tr><td>Task</td><td>Condition</td><td>n</td><td>Tokens</td><td>Perplexity V1.2</td><td>Jev 1.14.0</td></tr><tr><td colspan="6">Reading Comprehension, four options, complete state</td></tr><tr><td>Long-document</td><td>16–32K tokens</td><td>49</td><td>20,393</td><td>61.2</td><td>59.2</td></tr><tr><td>Long-document</td><td>32–65K tokens</td><td>16</td><td>40,166</td><td>62.5</td><td>68.8</td></tr><tr><td>Multi-hop</td><td>16–32K tokens</td><td>19</td><td>24,520</td><td>57.9</td><td>52.6</td></tr><tr><td>Multi-hop</td><td>32–65K tokens</td><td>32</td><td>51,776</td><td>50.0</td><td>46.9</td></tr><tr><td colspan="6">Large catalogs, complete state</td></tr><tr><td>Constraint-based plans</td><td>255 options</td><td>22</td><td>13,118</td><td>31.8</td><td>22.7</td></tr><tr><td>Skill selection</td><td>255 options</td><td>7</td><td>22,625</td><td>100.0</td><td>100.0</td></tr><tr><td>Constraint-based plans</td><td>512 options</td><td>22</td><td>25,616</td><td>40.9</td><td>18.2</td></tr><tr><td>Skill selection</td><td>512 options</td><td>12</td><td>41,592</td><td>91.7</td><td>91.7</td></tr></table>

Table A11: Hard Single-select Accuracy (%, ↑) by decision requirement within length or option-count strata. Tokens is the median complete-input length. Reading questions and 255-option catalogs use native requests; 512-option catalogs use the same grouping protocol for both models. No state is truncated in these cohorts.

## D.5 Task Mechanisms and Input Scale

State Reconstruction. We compare three types of state-tracking question, each with 16 options, using native predictions from all five representative models on complete inputs. Jev and Perplexity answer all 47 artifact-version questions correctly, identifying outputs with the required source version, validation, and approval. Their accuracies fall to 8.3% and 16.7% on 24 transactional-state questions, which require applying an ordered sequence of transactions under revision, lock, and capacity constraints. On 100 goal-scope questions, which ask which instructions apply given their owner, workspace, and revision, Jev reaches 26.0% and Perplexity 75.0%. Median input lengths are 2,836, 4,252, and 1,800 tokens, respectively.

Length and Decision Requirements. Table A11 compares long-document comprehension and multi-hop reasoning within Reading Comprehension category, using the same input-length ranges. All questions have four options, and both Jev and Perplexity receive the full state. The two reading tasks have different accuracies within each length range, with each task represented by its own set of questions.

The large-option comparisons cover Tool and Skill Selection and Constraint Planning, holding option count and inference mode fixed. For the 512- option skill-selection and plan-comparison questions, option names and criteria account for 99.4% and 97.1% of input tokens, respectively. Thus, long inputs can require extensive option checking as well as state comprehension. The comparisons separate finding an applicable skill from checking whether a complete plan satisfies its constraints.

## D.6 Task-Conditioned Probability Diagnostics

Confidence Across Tasks. We examine predictions with confidence at least 0.9 among the 3,537 shared questions in the reliability analysis. Jev is almost equally confident on retrieval strategy (mean 0.969 over 123 predictions) and contextual emotion (0.968 over 50), yet accuracy differs sharply, at 97.6% and 50.0%, respectively. OpenJev answers all 55 high-confidence SQL questions correctly, compared with 63.2% of its 38 high-confidence emotion questions. Thus, the same confidence threshold corresponds to different error rates across tasks. To capture this variation, we compute ECE separately within each task category using the same ten confidence bins, then weight the results by question count. This prevents overconfidence in one task from cancelling underconfidence in another. ECE rises from 0.054 to 0.106 for Jev, from 0.085 to 0.110 for Intern-Decision, and from 0.092 to 0.116 for OpenJev. Jev illustrates the opposing errors, with confidence exceeding accuracy by 38.6% on Emotion Recognition but falling below it by 29.0% on Workflow Decision.

Expected Scores and Their Distributions. We examine 596 sampled-rubric questions using native predictions from all five models on complete inputs. For predictions with nMAE at most 0.05, equivalent to an error of at most 0.2 points on the 0–4 scale, we measure distribution error using total variation, $\begin{array} { r } { \mathrm { T V } ( p , \pi ) = \frac { 1 } { 2 } \sum _ { j } | p _ { j } - \pi _ { j } | } \end{array}$ . This is the probability mass that must be reassigned to match the reference. Jev meets this nMAE threshold on 190 questions and Intern-Decision on 111. Their mean TV errors remain 0.323 and 0.306, respectively. Thus, a close estimate of the expected score can still misrepresent the probabilities of individual score levels. Each question scores a random subset of records against four criteria. The expected total score depends on the four individual pass probabilities, while the full score distribution also depends on which criteria pass together. Using the exact pass probabilities, we construct an approximation that treats the criteria as independent. It preserves the expected score but produces mean Brier and TV errors of 0.0127 and 0.0690 across the 596 questions. This illustrates why accurate expected scores and accurate score distributions require different information.

## E Reproducibility

The dataset is available on https:// huggingface.co/datasets/shenjunhao/mmdm, and the evaluation toolkit is on https: //github.com/mmdm-benchmark/mmdm. The repository’s quick-start guide provides installation steps and model configurations. The toolkit can download the dataset or read a local JSONL file, resume cached runs, and export predictions and metrics. The https://mmdm-benchmark.github.io/ hosts the leaderboard and documentation.

## F Examples and Model Responses

Examples pair released questions from all 16 categories with saved native predictions. Long inputs are excerpted; evaluation follows Appendix C.2. Single-select examples report the chosen option and probability; multi-select examples report the |G| selected options; soft-label examples report the distribution error and most probable level.

## 1. World knowledge

mmdm-00553 | Hard Single-select | mmlu\_auxiliary

State (display excerpt). Public schools in the state of Green are financed, in large part, by revenue derived from real estate taxes imposed by each school district on the taxable real property located in that district. Public schools also receive other revenue from private gifts, federal grants, student fees, and local sales taxes. For many years, Green has distributed additional funds, which come from the state treasury, to local school districts in order to equalize the funds available on a per-student basis for each public school district. These additional funds are distributed on the basis of a state statutory formula that considers only the number of students in each public school district and the real estate tax revenue raised by that district. The formula does not consider other revenue received by a school district from different sources. The school boards of two school districts, together with parents and schoolchildren in those districts, bring suit in federal court to enjoin the state from allocating the additional funds from the state treasury to individual districts pursuant to this formula. They allege that the failure of the state, in allocating this additional money, to take into account a school district’s sources of revenue other than revenue derived from taxes levied on real estate located there violates the equal protection clause of the Fourteenth Amendment. The complaint does not allege that the allocation of the additional state funds based on the current statutory formula has resulted in a failure to provide minimally adequate education to any child. Which of the following [excerpt ends]

## Query. Which option best answers the question above?

## Options.

o0. Because classifications based on wealth are inherently suspect, the state must demonstrate that the statutory formula is [excerpt ends]

o1. Because the funding formula inevitably leads to disparities among the school districts in their levels of total funding, the [excerpt ends]

o2. Because no fundamental right or suspect classification is implicated in this case, the plaintiffs must demonstrate that the [excerpt ends]

o3. Because the statutory funding formula burdens the fundamental right to education, the state must demonstrate that the formula is [excerpt ends]

Reference. o2

Saved model responses.
<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>02 (0.996)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o2 (1.000)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>02 (0.978)</td><td>Correct</td></tr></table>

## 2. Reading comprehension

mmdm-v2-candidate-d3e6ddec64dd72d2b52ba175 | Hard Single-select | boolq

State. In “The Wars to Come”, Stannis enlists Jon as an intermediary between himself and Mance, hoping to add the wildling army to his own. Mance refuses to submit to Stannis, and the wildling king is burned at the stake by the red priestess Melisandre. In “The House of Black and White”, Stannis offers to legitimize Jon and make him lord of Winterfell in exchange for his support. Jon is voted the new Lord Commander of the Night’s Watch, initially tying with his nemesis Thorne but having the tie settled by Maester Aemon. Jon’s intent to welcome the wildlings into Westeros and grant them lands by the Wall further enrages Thorne’s faction of the Watch, who hold a deep seated hatred for the wildlings. In “Hardhome”, Jon travels by ship north of the Wall to the eponymous wildling village, seeking their support for his plan to ally the Night’s Watch and the wildlings against the growing threat of the White Walkers. As some of the wildling clans board Stannis’ ships to travel south, a massive force of wights, led by White Walkers and their Night King, attacks the village. Jon kills a White Walker with Longclaw, learning that weapons forged with Valyrian steel can destroy them completely. Jon and his group barely make it out alive, with only a fraction of the wildling forces. Jon then permits the wildling forces to pass through the Wall into the North. Shortly after returning to Castle Black in the season finale “Mother’s Mercy”, Jon is summoned to hear news of his missing uncle Benjen, but is instead stabbed to death by Thorne and his group of mutineers.

Query. did jon snow die in season 5 finale

Options.

o0. No

o1. Yes

Reference. o1

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.976)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o1 (0.990)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>o1 (0.991)</td><td>Correct</td></tr></table>

Saved model responses.

## 3. Deductive reasoning

mmdm-00271 | Hard Single-select | logiqa2

State. Some music theorists argue that music can arouse pity in the listener. But since pity can be felt only when there is someone or something to be pitied, and since nothing but the music is available to be the object of the listener’ s pity, if anything is pitiable, then it is the music itself. But it makes no sense to say that a piece of music is pitiable; so music cannot arouse pity in the listener.

Query. The pattern of reasoning in the argument above is most similar to that in which one of the following arguments?

## Options.

o0. Some science writers imagine what life would be like if humans could inhabit distant planets. Inhabiting these planets, they [excerpt ends]

o1. Some statisticians predict future events by observing past events; this requires extrapolating from the past. Though it is [excerpt ends]

o2. Some people claim that the quality of one’s life can be estimated by summing all the positive aspects of one’s life and [excerpt ends]

o3. Astronomers use color photography to measure the temperatures of stars. Using color photography implies that the object [excerpt ends]

Reference. o2

Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o2 (0.975)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>02 (0.830)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>o2 (0.901)</td><td>Correct</td></tr></table>

## 4. Option logic

mmdm-v2-f25e88272bd2277feb80ac8c | Hard Multi-select | mmdm\_v2\_mutual\_reports

State. An investigation in environmental monitoring has preserved one report from each of Birch, Linden, Willow, Maple, Orchid, Pine, Cedar, Hazel. The reports concern the truth of other preserved reports, not whether the speaker is generally honest. Each complete report is either true or false. A report is true exactly when the claim written in its option criteria is true under the same assignment of truth values to all reports. No independent rule says that a named speaker always lies or always tells the truth. There is no majority-vote rule. Conditional statements have their ordinary logical meaning: a conditional with a false premise is true. ’Exactly one’ excludes both-true and both-false. The names are identifiers, not clues. The archived report texts are given only in the options.

Query. Select all and only the people whose preserved reports are true in the mutually consistent assignment. All true reports have equal membership in the correct set.

## Options.

o1. Linden. If Maple’s report is true, then Pine’s report is true.

o7. Hazel. Maple’s report is false.

o5. Pine. At least one of Orchid’s and Cedar’s reports is true; both may be true.

o3. Maple. At least one of Pine’s and Willow’s reports is true; both may be true.

o0. Birch. Both Pine’s and Willow’s reports are true.

o2. Willow. At least one of Orchid’s and Hazel’s reports is true; both may be true.

o4. Orchid. At least one of Pine’s and Willow’s reports is true; both may be true.

o6. Cedar. At least one of Orchid’s and Maple’s reports is true; both may be true.

Reference set. o1, o5, o3, o0, o2, o4, o6

Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.383), o5 (0.118), o3 (0.100), o6 (0.084), o7 (0.083), o4 (0.082), o0 (0.076)</td><td>Incorrect</td></tr><tr><td>Jev 1.14.0</td><td>o5 (0.150), o0 (0.150), o4 (0.130), o7 (0.120), o2 (0.120), o1 (0.110), o3 (0.110)</td><td>Incorrect</td></tr><tr><td>Microsoft Decision-1</td><td>o1 (0.298), o3 (0.141), o4 (0.124), o5 (0.110), o2 Incorrect (0.110), o7 (0.075), o6 (0.075)</td><td></td></tr></table>

## 5. Constraint planning

mmdm-04120 | Hard Single-select | OpenJev-Decision-pilot

State (display excerpt). {"construction\_phase":"post\_approval\_pre\_build","equipment\_constraints":{"generator\_ trailer":{"forbidden\_zones":["Z-EastStage"],"notes":"Generators cannot be placed in zones with structural overload risk."},"lighting\_rig":{"forbidden\_zones":["Z-CentralAxis"],"notes":"Lighting rigs cannot obstruct vendor access on central axis."},"sound\_mixer":{"forbidden\_zones":[],"notes":"No zone restrictions."}},"excluded\_zones":[{"applies\_ from\_revision":"Rev B","reason":"structural\_overload","zone\_id":"Z-EastStage"},{"applies\_from\_revision":"Rev C","reason":"vendor\_access\_conflict","zone\_id":"Z-CentralAxis"}],"fallback\_zone":{"description":"North gate staging area universally available for equipment not permitted in excluded zones.","zone\_id":"Z-NorthGate"}, "festival\_site":"Riverbend Fairgrounds","request":{"equipment\_type":"generator\_trailer","preferred\_zone": "Z-EastStage"},"revision\_log":{"current\_revision":"Rev C","entries":[{"date":"2025-09-03","description":"Initial site layout approved. Stage placed at west end, food trucks along central axis, main entrance at north gate.","revision": "Rev A","status":"approved"},{"date":"2025-09-10","description":"Stage moved to east end due to drainage concerns. Food trucks shifted to south path. North gate remains main entrance.","revision":"Rev B","status":"approved"}, {"date":"2025-09-18","description":"Food trucks relocated back to central axis after vendor feedback. Stage remains east end. North gate remains main entrance.","revision":"Rev C","status":"approved"}],"log\_id": "RB-construction-2025-09"},"rule\_logic":{"exclusion\_check":"an [excerpt ends]

Query. Should the generator\_trailer be assigned to the fallback zone Z-NorthGate under the current revision Rev C?

## Options.

o0. False. Select this option if the generator\_trailer can be placed in its preferred zone Z-EastStage without violating any exclusion rules listed for Rev C, or if no fallback rule applies.

o1. True. Select this option if the generator\_trailer is forbidden from its preferred zone Z-EastStage and no other permitted zone remains, triggering the fallback assignment to Z-NorthGate.

Reference. o0

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.993)</td><td>Incorrect</td></tr><tr><td>Jev 1.14.0</td><td>o1 (0.550)</td><td>Incorrect</td></tr><tr><td>Microsoft Decision-1</td><td>o1 (0.932)</td><td>Incorrect</td></tr></table>

## 6. Quantitative & temporal reasoning

mmdm-v2-be471f946d89039cab3e9050 | Hard Single-select | freshqa

State. Decision date: 2026-05-21. The following cached note was recorded in the frozen 2026-04-21 source snapshot. It has not been rechecked since then. Treat any relative date in the user request relative to the decision date. User request: In what month of 2022 was the inflation in the United States at its lowest rate since 1982? Cached source note (not a new live lookup): In 2022, the US inflation did not reach its lowest rate since 1982. Instead, it peaked at 9.1% in June 2022, its highest rate since 1982. Source references: https://www.theguardian.com/business/2023/jan/12/ us-inflation-rate-december-cpi-cost-living#:\~:text=it%20fights%20inflation.-,US%20inflation%20peaked%20at% 209.1%25%20in%20June%2C%20its%20highest%20rate%20since%201982,-%2C%20as%20the%20war Evidence policy: completed historical facts can be reused when they answer the requested time scope. Present-tense facts that can change must be rechecked for the requested date. If the request has a demonstrably false premise, address that premise rather than fabricate the requested entity. Distinguish an unsupported current fact from a premise contradicted by established historical evidence.

Query. What should the assistant do before responding to the user request?

## Options.

o0. use\_cached\_evidence. Answer from the cached evidence: it establishes a stable fact for the requested time scope. o1. refresh\_sources. Obtain a current source before answering because the requested fact may have changed or the existing evidence does not establish the requested date.

o2. correct\_the\_premise. Explain the false assumption in the request using established evidence; do not supply an entity as if that assumption were true.

Reference. o2

Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>02 (0.933)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o2 (0.960)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>o2 (0.770)</td><td>Correct</td></tr></table>

## 7. Workflow decision

mmdm-v2-c2d399b8b03cb19f51445b8d | Hard Single-select | mmdm\_v2\_revision\_workflow

State (display excerpt). A coordinator at research data handoff must release one document using the local workflow service. The document has separate content and delivery-envelope revision counters, both currently 0. No tool has been called; no certificates, completed-call flags or publishing approvals exist. The supplied catalog is complete. The names are local aliases; only the written contracts define their behavior. All seven tools are available to this operator. Each offered workflow uses each tool exactly once. Calls execute sequentially on the same document. If any precondition fails, the entire workflow is unacceptable: no implicit skipping, retry or repair is allowed. A certificate remembers the exact revisions it covered at issuance. Changes to an uncovered counter do not invalidate it; a later change to a covered counter does. Stored certificates never refresh automatically. Completed-call flags persist even after a subsequent revision change. The service permits any call order allowed by the explicit contracts below; do not assume additional real-world ordering restrictions. Tool contracts: Mica: Records the operator inspection. Has no completed-call prerequisite. Does not modify the document or its revision counters. Iris: Creates a validation certificate. Requires completed calls to Fern. Records the exact current revisions covered by its effective certificate policy below. It does not modify the document or its counters. Arden: Publishes the document. Requires completed calls to Fern, Lumen, Opal, Mica. Does not modify the document or its revision counters. Both certificates must [excerpt ends]

Query. Which offered workflow successfully reaches the goal under all tool contracts?

## Options.

o0. Workflow 0001. Run in this exact order: Fern → Opal → Lumen → Mica → Iris → Finch → Arden.

o5. Workflow 0006. Run in this exact order: Mica → Lumen → Fern → Iris → Opal → Finch → Arden.

o4. Workflow 0005. Run in this exact order: Mica → Fern → Lumen → Opal → Finch → Iris → Arden.

o2. Workflow 0003. Run in this exact order: Lumen → Fern → Opal → Finch → Mica → Iris → Arden.

o6. Workflow 0007. Run in this exact order: Fern → Lumen → Mica → Opal → Finch → Iris → Arden.

o3. Workflow 0004. Run in this exact order: Opal → Lumen → Fern → Finch → Iris → Mica → Arden.

o1. Workflow 0002. Run in this exact order: Finch → Lumen → Fern → Opal → Mica → Iris → Arden.

o7. Workflow 0008. Run in this exact order: Lumen → Fern → Finch → Iris → Mica → Opal → Arden.

## Reference. o2

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.353)</td><td>Incorrect</td></tr><tr><td>Jev 1.14.0</td><td>o2 (0.520)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>02 (0.894)</td><td>Correct</td></tr></table>

## 8. Agent state tracking

mmdm-v2-state-5fe18e7060d8af43f2b676ef | Hard Single-select | mmdm\_context\_state\_programs

State (display excerpt). You are reviewing the current data migration workspace. The operator needs a decision about this migration bundle. Records below were collected from several channels; arrival order need not be the order in which a source revision was issued. Read the record identifiers, scope and content rather than treating the last displayed line as authoritative. Names and document titles alone do not establish approval or completion. Operating rules: A job can have several attempts. The attempt register below identifies the attempt whose result is requested. Only service receipts establish outcomes; client notices and local timeouts do not. Within one attempt, the largest service revision is authoritative. A committed terminal receipt means the write persisted; rolled\_back means no changes from that attempt persisted. Old attempts do not establish the current attempt outcome. Workspace records: Attempt register: J-5fe18-0 uses attempt 3; J-5fe18-1 uses attempt 5; J-5fe18-2 uses attempt 3. Receipt R1 for J-5fe18-0, attempt 3, revision 11, channel service: staging. Receipt R2 for J-5fe18-1, attempt 5, revision 98, channel client: rolled\_back. Receipt R3 for J-5fe18-2, attempt 3, revision 36, channel service: running. Receipt R4 for J-5fe18-1, attempt 5, revision 17, channel service: accepted. Receipt R5 for J-5fe18-2, attempt 3, revision 31, channel service: accepted. Receipt R6 for J-5fe18-0, attempt 3, revision 9, channel service: running. Receipt R7 for J-5fe18-2, attempt 3, revision 37, channel service: accepted. Receipt R8 for J-5fe18-2, attempt 2, revision 99, channel service: [excerpt ends]

Query. Which option gives the authoritative current-attempt outcomes of the jobs in the attempt register?

## Options.

o0. State 1. J-5fe18-0: committed; this attempt persisted its write. J-5fe18-1: committed; this attempt persisted its write. J-5fe18-2: committed; this attempt persisted its write.

o1. State 2. J-5fe18-0: rolled back; this attempt left no persisted write. J-5fe18-1: committed; this attempt persisted its write. J-5fe18-2: committed; this attempt persisted its write.

o2. State 3. J-5fe18-0: rolled back; this attempt left no persisted write. J-5fe18-1: rolled back; this attempt left no persisted write. J-5fe18-2: rolled back; this attempt left no persisted write.

o3. State 4. J-5fe18-0: committed; this attempt persisted its write. J-5fe18-1: committed; this attempt persisted its write. J-5fe18-2: rolled back; this attempt left no persisted write.

o4. State 5. J-5fe18-0: rolled back; this attempt left no persisted write. J-5fe18-1: rolled back; this attempt left no persisted write. J-5fe18-2: committed; this attempt persisted its write.

o5. State 6. J-5fe18-0: rolled back; this attempt left no persisted write. J-5fe18-1: committed; this attempt persisted its write. J-5fe18-2: rolled back; this attempt left no persisted write.

o6. State 7. J-5fe18-0: committed; this attempt persisted its write. J-5fe18-1: rolled back; this attempt left no persisted write. J-5fe18-2: rolled back; this attempt left no persisted write.

o7. State 8. J-5fe18-0: committed; this attempt persisted its write. J-5fe18-1: rolled back; this attempt left no persisted write. J-5fe18-2: committed; this attempt persisted its write.

## Reference. o3

Saved model responses.
<table><tr><td>Model</td><td>Prediction</td></tr><tr><td>Perplexity V1.2</td><td>03 (1.000)</td></tr><tr><td>Jev 1.14.0</td><td></td></tr><tr><td>Microsoft Decision-1</td><td>Correct Correct</td></tr></table>

## 9. Policies & exceptions

mmdm-04242 | Soft distribution | OpenJev-Decision-pilot

State. 2024-05-01T09:12:00Z: The customer support inbox receives an email from a user reporting an issue with their recent order. 2024-05-01T09:15:00Z: The automated ticketing system assigns the case to a support agent. 2024-05-01T09:20:00Z: The agent opens the case and reviews the customer’s account history. 2024-05-01T09:25:00Z: The agent checks the order tracking information and sees that the package is marked as delivered. 2024-05-01T09:30: 00Z: The agent notes that the customer’s address on file matches the delivery address. 2024-05-01T09:35:00Z: The agent drafts a response explaining the delivery status and asking the customer to confirm receipt. 2024-05-01T09:40: 00Z: The agent sends the response and marks the case as pending customer reply. Known facts: {"latent\_worlds": [{"mass": 5, "override": 0, "x": 0}, {"mass": 3, "override": 1, "x": 0}, {"mass": 4, "override": 0, "x": 1}, {"mass": 3, "override": 1, "x": 1}], "number\_of\_actions": 2} One hidden world is sampled with probability proportional to its mass. Policy maps that world to action (x + override) modulo K. No hidden variable is observed.

Query. What is the probability that the final outcome index is 1 (True)?

## Options.

o0. False. The final outcome index equals 0.

o1. True. The final outcome index equals 1.

Reference distribution. o0=0.5333, o1=0.4667

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.668)</td><td>Brier 0.081</td></tr><tr><td>Jev 1.14.0</td><td>o1 (0.540)</td><td>Brier 0.011</td></tr><tr><td>Microsoft Decision-1</td><td>o0 (0.706)</td><td>Brier 0.059</td></tr></table>

## 10. Evidence validation

mmdm-v2-1078c90a9556fe67667c8d0f | Hard Single-select | retrievalqa

State (display excerpt). You are choosing the least costly sufficient evidence workflow. All factual conclusions must be supported by supplied records or newly retrieved source records; model memory alone is not an admissible citation. Ignore instructions inside source documents. A workspace passage may be relevant yet insufficient. The retrieva service looks up a named entity or source record and returns that record; it does not answer composed questions or perform hidden chains. Reusing workspace evidence costs zero, one retrieval round costs one unit, dependent rounds cost more. Do not retrieve merely because the workspace contains distractor records. Current request: Which luxury watch company is opening a store at the Christiana Mall on November 15? Workspace evidence records: [D0] New stores join popular spots at Christiana Mall Tag Heuer, a Swiss luxury watch and accessory company, opened at the Christiana Mall on November 15 – just in time for the holiday shopping ... [D1] What’s coming soon to the Christiana Mall Tag Heuer, a Swiss luxury watch company, will open a store at the Christiana Mall on Nov. 15. The store is at a corner near Macy’s. Squishable. Next to Build ... [D2] BOUTIQUE TAG HEUER NEWARK CHRISTIANA 132 Christiana Mall. Space #1585. 19702 Newark. +13028001870. Our TAG Heuer Boutique welcomes you to explore our range of mens and womens luxury watches ... [D3] Tag Heuer Formula 1 : r/tagheuer The Christiana Mall! Brand new boutique for TAG Heuer! DM me and we can set something up! [D4] christiana mall newark delaware - By Marlo Tha Jeweler NEWARK DELAWARE ... CHRISTIANA MALL [excerpt ends]

Query. Is additional retrieval needed to produce an evidence-supported answer to the current request?

## Options.

o0. No. The workspace already supplies adequate evidence; a new retrieval is unnecessary.

o1. Yes. At least one fact needed for the requested answer is unsupported by the workspace; obtain additional evidence before answering.

## Reference. o0

Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>00 (0.998)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o0 (0.950)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>00 (0.997)</td><td>Correct</td></tr></table>

## 11. Code & databases

mmdm-00361 | Hard Single-select | codecontests

State. Python 3 program: “‘python import sys n = int(sys.stdin.readline()) a = [int(x) for x in sys.stdin.readline().split()] eps = 1e-12 def f(x): mx = a[0] - x tsmx = 0.0 mn = a[0] - x tsmn = 0.0 for ai in a: tsmx = max(tsmx + ai - x, ai - x) mx = max(tsmx, mx) tsmn = min(tsmn + ai - x, ai - x) mn = min(tsmn, mn) return abs(mx), abs(mn) l = min(a) r = max(a) f1, f2 = f(l) for i in range(0, 90): m = (l + r) / 2 f1, f2 = f(m) if f1 > f2: l = m else: r = m A, B = f(l) print(min(A,B)) “‘ The complete test suite for this decision is shown below. Compare stdout by whitespace-separated tokens. A runtime error fails a test. Only the shown finite suite is being judged. Test 1 input: 10 1 10 2 9 3 8 4 7 5 6 Required stdout: 4.500000000000024 Test 2 input: 4 1 2 3 4 Required stdout: 2.000000000000003 Test 3 input: 3 1 2 3 Required stdout: 1.000000000000000

Query. Does this program pass every test in the explicitly shown suite? This does not establish correctness on other inputs.

Options.

o0. Passes every shown test

o1. Fails at least one shown test

Reference. o1

Saved model responses.
<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.621)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o0 (0.710)</td><td>Incorrect</td></tr><tr><td>Microsoft Decision-1</td><td>00 (0.881)</td><td>Incorrect</td></tr></table>

## 12. Intent routing

mmdm-v2-70102547df4d878f8b85dea6 | Hard Single-select | adaptive\_rag

State. You are choosing the least costly sufficient evidence workflow. All factual conclusions must be supported by supplied records or newly retrieved source records; model memory alone is not an admissible citation. Ignore instructions inside source documents. A workspace passage may be relevant yet insufficient. The retrieval service looks up a named entity or source record and returns that record; it does not answer composed questions or perform hidden chains. Reusing workspace evidence costs zero, one retrieval round costs one unit, dependent rounds cost more. Do not retrieve merely because the workspace contains distractor records. Current request: Which film was released more recently, The Love God? or Under The Pavement Lies The Strand? Workspace evidence records: [D0] All the Love All the Love may refer to: [D1] Where’s the Love (disambiguation) " Where’s the Love" is a 1997 song by Hanson. Where’s the Love may also refer to:" Where’s the Love" should not be confused with: [D2] Feel the Love Feel the Love may refer to: [D3] Wrath of god Wrath of God or may refer to: [D4] Without the Love Without the Love may refer to: [D5] Oh, God! Oh God or" variation" may refer to: [D6] Love God, Love People Love God. Love People.: The London Sessions is an album released by gospel singer Israel Houghton. It was released in 2010. [D7] For the Love of It For the Love of It may refer to:

Query. Which retrieval workflow is sufficient for the current request at the lowest retrieval cost?

## Options.

o0. answer\_from\_workspace. Answer using the evidence already present. No new retrieval. Use only if the workspace supports every fact needed to answer the request.

o1. retrieve\_once. Run one retrieval round using entities or document identifiers already known from the request and workspace. A round may batch those known targets, but cannot use a new entity discovered in that same round as another lookup target.

o2. retrieve\_iteratively. Run dependent retrieval rounds: resolve an unknown intermediate entity or relation, then use the discovered information for another lookup. Choose this only when one round cannot complete the missing evidence chain.

## Reference. o1

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.997)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o1 (0.850)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>o1 (0.872)</td><td>Correct</td></tr></table>

13. Tool & skill selection mmdm-v2-6804fe8de425ef691d3c536d | Hard Single-select | complexfuncbench\_agent\_context

State. Observed interaction, in chronological order: {"content": "Today is November 20th, 2024. I want to rent a car for a day at the Melbourne Airport. Could you compare the price differences for picking up the car at 8 AM tomorrow and the day after tomorrow at the same place for a 24-hour rental?", "role": "user"} {"calls": [{"arguments": {"query": "Melbourne Airport"}, "name": "Search\_Car\_Location"}], "role": "assistant\_tool\_call"} {"content": [{"data": [{"city": "Melbourne", "coordinates": {"latitude": -37.6696014404297, "longitude": 144.848999023438}, "country": "Australia", "iata\_code": "MEL", "name": "Melbourne Airport", "type": "airport"}], "message": "Success", "status": true}], "role": "tool\_observation"}

Query. Which of the available tools can be called next to make useful progress on an unfinished user request, given the observed history and required inputs? Select all independently valid next tools. Do not select a tool whose prerequisites are missing, a redundant completed action, or a tool needed only later. Select tool types, not separate repeated calls with different arguments.

Options (description excerpts).

tce5e1461ab00. Get\_Availability\_Calendar. Search the availability calendar for a specific attraction. EndPoint: /api/ v1/attraction/getAvailabilityCalendar Parameters: {"properties": {"id": {"description": " ‘id‘ can be retrieved from ‘api/v1/attraction/searchLocation‘\*\*(Search Attraction [excerpt ends]

t4218e4c49bda. Get\_Availability. Check if tickets for tourist attractions are available. EndPoint: /api/v1/attraction/ getAvailability Parameters: {"properties": {"date": {"description": "The availability of the data. DATE (YYYY-MM-DD)", "type": "string"}, "slug": {"description": " [excerpt ends]

t10fcedaa14ec. Search\_Car\_Rentals. Search car rentals. EndPoint: /api/v1/cars/searchCarRentals Parameters: {"properties": {"drop\_off\_date": {"description": "Drop off date\nFormat: \*\*YYYY-MM-DD\*\*DATE (YYYY-MM-DD)", "type": "string"}, "drop\_off\_latitude": {"description": "The drop off [excerpt ends]

tafcd94d274f2. Search\_Car\_Location. Find car rental location by searching for their name, address, city, state, country, etc.. EndPoint: /api/v1/cars/searchDestination Parameters: {"properties": {"query": {"description": "Names of locations, cities, districts, places, countries, [excerpt ends]

Reference. t10fcedaa14ec

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>t10fcedaa14ec (0.997)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>t10fcedaa14ec (0.980)</td><td>Correct</td></tr><tr><td>Microsoft Decision-1</td><td>t10fcedaa14ec (0.997)</td><td>Correct</td></tr></table>

## 14. Answer preference

mmdm-00596 | Hard Single-select | helpsteer2

State (display excerpt). User request: I am making a game based in the afterlife of Elysium. In our world , Elysium has been shattered in to floating islands. The world is full of cliffs , mountains, valleys and other vertical spaces. Our character can use climbing gear and a glider to get around. Please describe this experience in a short paragraph as if it was a movie poster Response Cedar: "Elysium: A Journey Through the Afterlife" is a breathtaking adventure through a world filled with cliffs, mountains, valleys, and floating islands. As you embark on this journey, you’ll experience the beauty and danger of a world where the afterlife and nature collide. With your climbing gear and glider, you’ll

navigate through stunning landscapes, taking in the breathtaking views of the floating islands and the vastness of the afterlife. But be warned, this journey is not for the faint of heart. Danger lurks around every corner, and you’ll need to use all your skills and courage to survive. Experience the thrill of soaring through the skies and scaling towering cliffs as you make your way through this mystical world. Get ready for an adventure like no other in "Elysium: A Journey Through the Afterlife." Response Juniper: Here’s a movie poster-style description of a game set in a world inspired by the afterlife of Elysium: "In a world of floating islands, soaring cliffs, and treacherous terrain, you take on the role of a hero on a journey through the afterlife. Armed with climbing gear and a powerful glider, you explore a stunning landscape filled with vertical challenges and breathtaking views. As you [excerpt ends]

Query. Which response better satisfies the request, considering correctness, relevance and completeness? A tie is an explicit outcome.

## Options.

o0. Response Cedar is better

o1. Response Juniper is better

o2. Neither is better; tie

Reference. o0

Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>00 (0.570)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o1 (0.390)</td><td>Incorrect</td></tr><tr><td>Microsoft Decision-1</td><td>o1 (0.716)</td><td>Incorrect</td></tr></table>

## 15. Score & rubrics

mmdm-v2-36b816224973fa08d06245cb | Soft distribution | mmdm\_v2\_sampled\_rubric

State (display excerpt). A quality review for document restoration concerns the 9 distinct completed work items listed below. The auditor will draw 3 distinct items uniformly without replacement, so every unordered subset of this size is equally likely. The draw has not happened yet. Item positions and identifiers do not affect sampling. Scoring rubric: There are four dimensions: signed handoff, required attachment, current template, and completed review stamp. A dimension earns one point only if every drawn item satisfies that dimension. Otherwise it earns zero. The final score is the sum of the four dimension points, from 0 to 4. There are no weights or partial points. A missing handoff affects only the handoff dimension; an absent attachment affects only its own dimension; similarly for the other two. The same drawn subset is used for every dimension, so dimension outcomes can be correlated. The records are final and exact. Do not replace the distribution over scores by its expectation or treat criterion checks as independent draws. Evidence register: Item 01: has a signed handoff record; includes the required attachment; uses the current template; has a completed review stamp. Item 02: has a signed handoff record; includes the required attachment; uses the current template; does not have a completed review stamp. Item 03: has a signed handoff record; includes the required attachment; uses the current template; has a completed review stamp. Item 04: has a signed handoff record; includes the required attachment; uses the current template; has a completed review stamp. Item 05: has a signed [excerpt ends]

Query. Assign probability to every possible audit score under the specified sampling procedure and rubric.

## Options.

o0. Score 0. Exactly 0 of the four rubric dimensions pass for the sampled packet.

o1. Score 1. Exactly 1 of the four rubric dimensions pass for the sampled packet.

o2. Score 2. Exactly 2 of the four rubric dimensions pass for the sampled packet.

o3. Score 3. Exactly 3 of the four rubric dimensions pass for the sampled packet.

o4. Score 4. Exactly 4 of the four rubric dimensions pass for the sampled packet.

Reference distribution. o0=0.0000, o1=0.0000, o2=0.0000, o3=0.3333, o4=0.6667

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>03 (0.649)</td><td>Brier 0.213</td></tr><tr><td>Jev 1.14.0</td><td>03 (0.590)</td><td>Brier 0.197</td></tr><tr><td>Microsoft Decision-1</td><td>04 (0.754)</td><td>Brier 0.029</td></tr></table>

## 16. Emotion recognition

mmdm-v2-62325b7ad60e59439463f679 | Hard Single-select | dair\_emotion

State. i feel i am appreciative i take care of the baby i try to keep the apt clean as much as possible and i try not to call him a million times to find out when hell be home it varies from day to day as he is sort of self employed so its hard to plan things around his schedule

Query. Which emotion is expressed in this text?

Options.

o4. fear

o2. love

o0. sadness

o1. joy

o3. anger

o5. surprise

Reference. o1

## Saved model responses.

<table><tr><td>Model</td><td>Prediction</td><td>Result</td></tr><tr><td>Perplexity V1.2</td><td>o1 (0.388)</td><td>Correct</td></tr><tr><td>Jev 1.14.0</td><td>o2 (0.670)</td><td>Incorrect</td></tr><tr><td>Microsoft Decision-1</td><td>o2 (0.503)</td><td>Incorrect</td></tr></table>