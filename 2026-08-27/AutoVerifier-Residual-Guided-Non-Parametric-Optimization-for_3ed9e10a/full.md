# AutoVerifier: Residual-Guided Non-Parametric Optimization for Reference-Based Answer Verification

Zebei Zhao<sup>1</sup>, Zhihao Shi<sup>1</sup>, Minqi Shi<sup>2</sup> <sup>1</sup>University of Science and Technology of China <sup>2</sup>Beihang University {zbzhao,zhihaoshi}@mail.ustc.edu.cn Sherrie@buaa.edu.cn

## Abstract

Reference-based verifiers are important for evaluating reasoning models and providing accurate outcome rewards in reinforcement learning with verifiable rewards. To improve verification accuracy, prior work has explored rulebased, model-based, and tool-augmented verifiers for checking answer equivalence across diverse answer forms. However, the equivalence of answer forms such as 1 + 3.14 and 1 + π may depend on the question and scoring criterion. We frame such implicit assumptions as verifier inductive biases. To address this challenge, we propose AUTOVERIFIER, a residual-guided non-parametric optimization method that learns these biases from recurring verifier errors. Specifically, AUTOVERIFIER records these biases in rule cards and promotes them to code modules or prompt guidance only after replay validation detects no direct regressions, keeping accepted updates auditable, editable, and reusable. Experiments on four verifier benchmarks demonstrate that AUTOVERI-FIER outperforms state-of-the-art verifiers by a large margin.

## 1 Introduction

Answer verification is widely used to check outputs of reasoning models in benchmark evaluation (Yan et al., 2025; Chen et al., 2025a; Liu et al., 2025a) and to provide outcome rewards for reinforcement learning with verifiable rewards (Shao et al., 2024; DeepSeek-AI, 2025; Yu et al., 2025). In reference-based verification, a verifier takes a question, a reference answer, and a candidate response as input to determine the correctness of the candidate response. As reasoning models produce longer and more diverse responses, exact matching alone is insufficient; the verifier must judge answer equivalence across diverse answer forms.

Recent verifier work has improved either answer matching coverage or made parts of comparison logic more explicit. Rule-based verifiers and libraries such as Math-Verify make comparison logic inspectable as code for fixed checks such as symbolic equality and numerical tolerances (Kydlícekˇ , 2025). Model-based verifiers and judge models improve coverage for challenging answer matching cases such as formulas, sequences, and multi-part answers, with recent reward models using generated rationales or critiques to support judgment (Chen et al., 2025a; Liu et al., 2025a; Chen et al., 2025b; Liu et al., 2025b; Xia et al., 2025). Toolaugmented verifiers add executable computation for scientific questions (Feng et al., 2025).

![](images/e9e4837be1219365dd4644f8bc3968601c814931ac43f1584a6cbe2aab04f4ce.jpg)  
Figure 1: Answer equivalence can depend on implicit assumptions. The examples cover rounded probabilities, symbolic forms for rounded numeric references, and scientific notation with units. Whether each pair is equivalent depends on the question and scoring criterion.

Despite these advances, a key gap remains when human annotations reflect implicit assumptions about which candidate answers should count as equivalent for a given question and scoring criterion. Figure 1 illustrates this issue with rounded probabilities, symbolic forms for rounded numeric references, and scientific notation with units. For example, 1 + 3.14 is not equivalent to 1 + π in an exact computation problem, but can be acceptable in an applied numerical problem where a rounded value of π is expected. Following Baxter (2000), we frame these implicit assumptions as verifier inductive biases to be learned from recurring verifier errors. Rule-based and tool-augmented verifiers remain limited by fixed comparison logic. Recent model-based verifier work covers complex answer matching, but treats ambiguous numerical acceptability thresholds as outside the binary verification setting (Liu et al., 2025a). Moreover, criteria learned by model-based verifiers remain embedded in model behavior rather than exposed as editable comparison logic. The unresolved problem is therefore to learn verifier inductive biases automatically from evidence while keeping accepted updates auditable, editable, and reusable.

To address this problem, we propose AUTOVER-IFIER, a non-parametric optimization method that learns verifier inductive biases from recurring verifier errors. Specifically, AUTOVERIFIER groups recurring errors, records the corresponding biases in rule cards with supporting examples and counterexamples, and promotes a card only after evidence and replay validation confirm no direct regressions. Cards with deterministic comparison logic become code modules after replay validation, while cards that require model judgment are incorporated into prompt guidance for the model-based fallback.

Experiments on four benchmarks demonstrate that AUTOVERIFIER outperforms existing reference-based verifier baselines. AUTOVERIFIER reaches 93.05% macro accuracy, exceeding the reported accuracy of rule-based, model-based, scientific, and tool-augmented verifiers. Compared with the prompt-only setting, adding code modules improves macro accuracy by +1.76 points, indicating that learned comparison logic contributes beyond prompt rewriting. The same direct decisions also reduce fallback calls by 32.13%.

Our key contributions can be summarized as follows:

• We identify implicit annotation assumptions as verifier inductive biases that can be learned from recurring verifier errors.

• We propose AUTOVERIFIER, a residualguided non-parametric optimization method that records recurring verifier errors as rule cards and promotes them through replay validation.

• We conduct experiments on four referencebased verification benchmarks, demonstrating that AUTOVERIFIER improves over a promptonly baseline, outperforms reported verifier baselines, and reduces fallback model calls.

## 2 Related Work

Answer verifiers. Answer verifiers commonly rely on predefined rules, model judgments, or tool pipelines. Rule libraries such as Math-Verify encode extractors, parsers, normalizers, numeric tolerances, symbolic equivalence checks, and abstention conditions as inspectable procedures (Kydlícekˇ , 2025); scientific verifiers add domain logic for algebraic equivalence, unit conversion, and notation handling (Zheng et al., 2025b). This auditability still requires manual coverage growth, and subtle equivalence remains failure prone (Huang et al., 2025). Model-based verifiers broaden coverage beyond fixed answer matching rules (Chen et al., 2025a; Liu et al., 2025a; Chen et al., 2025b; Liu et al., 2025b; Xia et al., 2025), while toolaugmented verifiers such as CoSineVerifier execute computation or unit conversion for scientific questions (Feng et al., 2025). However, rule and tool pipelines still rely on fixed equivalence checks, while model-based acceptance criteria remain embedded in model behavior. AUTOVERIFIER targets the middle ground by recording recurring verifier errors in rule cards that become auditable code modules, while unresolved cases remain with the model-based fallback.

Non-parametric updates from experience. The learning beyond gradients perspective studies systems that improve by editing external state, including code, tests, memory, and evaluation records, rather than model parameters (Weng, 2026). Procedural memory work such as Skill-Pro promotes reusable procedures through activation, execution, termination, and validation (Mi et al., 2026); related agents and program search systems evolve skills, tools, scaffolds, or code from experience (Wang et al., 2023; Hao et al., 2026; Zheng et al., 2025a; Romera-Paredes et al., 2024; Novikov et al., 2025). AUTOVERIFIER instantiates this nonparametric update pattern for verification, where recurring verifier errors are recorded as rule cards and accepted cards become code modules or prompt guidance after replay validation.

## 3 Method

AUTOVERIFIER learns verifier inductive biases from recurring verifier errors with a model-based fallback verifier. Construction replay exposes answer forms that the current verifier mishandles, and accumulated errors are grouped into recurring patterns before any update is proposed. A recurring pattern becomes a rule card that records the bias, its support examples, its counterexamples, and the conditions under which a code module should abstain. Here, a code module is a deterministic verifier unit that can return CORRECT, INCORRECT, or ABSTAIN. Cards with deterministic comparison logic enter code audit and replay validation before becoming code modules, while cards that require model judgment are incorporated into prompt guidance. This section defines the verification setting, the verifier update, the rule card paths, the replay validation procedure, and the frozen inference procedure.

![](images/3f495486c2899c888174e6b911ef2e12f01b463dd982db2a75cf4de03f16b481.jpg)  
Figure 2: Overview of AUTOVERIFIER. During construction, the current verifier is replayed on 5,000 examples, and recurring verifier errors are recorded as rule cards. Cards with deterministic comparison logic enter code audit and replay validation before becoming code modules, while cards that require model judgment are incorporated into prompt guidance. After freezing, code modules make direct decisions when they apply, and examples for which all code modules abstain go to the model-based fallback.

## 3.1 Problem Formulation

We focus on pointwise reference-based answer verification. Each example is

$$
\boldsymbol { x } _ { i } = ( q _ { i } , a _ { i } ^ { \star } , \hat { a } _ { i } , m _ { i } ) ,
$$

where $q _ { i }$ is a question, $a _ { i } ^ { \star }$ is the reference answer, $\hat { a } _ { i }$ is a candidate response, and $m _ { i }$ contains optional metadata such as domain or answer type. The binary label $y _ { i } \in \{ 0 , 1 \}$ indicates correctness (CORRECT or INCORRECT).

The construction problem starts from a fixed base verifier $V _ { \mathrm { b a s e } }$ and updates the verifier around it. We denote the construction pool by $D _ { \mathrm { { c o n } } }$ . A residual is a construction example where the current verifier disagrees with $y _ { i } .$ , and these residuals provide evidence for rule extraction. Support examples and counterexamples are construction examples used to validate a proposed rule card before benchmark evaluation. A recurring error pattern is treated as evidence that the current verifier lacks a verifier inductive bias for deciding whether a recurring answer transformation is equivalent or nonequivalent. Patterns that can be expressed as deterministic code become updates to a module library of code modules after code audit and replay validation, while patterns that require model judgment are handled by a model-based fallback verifier $V _ { \mathrm { f b } }$ implemented by the fixed model with fallback prompt $P _ { \mathrm { f b } }$

## 3.2 Verifier State Updates

At construction round t, the verifier state is $s _ { t } =$ $( \mathrm { R U L } _ { t } , P _ { \mathrm { f b } , t } )$ , where $\mathrm { R U L } _ { t }$ is the module library and $P _ { \mathrm { f b } , t }$ is the fallback prompt. Replaying this state on construction data yields the residual set:

$$
R _ { t } = \{ x _ { i } \in D _ { \mathrm { c o n } } : V _ { s _ { t } } ( x _ { i } ) \neq y _ { i } \} .
$$

The residual buffer accumulates recurring errors before rule card drafting. Let $V _ { s _ { t } }$ denote the verifier induced by $s _ { t }$ under the fixed routing rule. We maintain a construction residual buffer $\boldsymbol { B } _ { t } =$ $\textstyle \bigcup _ { \tau < t } R _ { \tau }$ . Here $R _ { t }$ is the residual set for the round, while $B _ { t }$ stores accumulated residuals used for later clustering. The coordinator groups recurring error patterns in $B _ { t }$ by answer object, surface form, source domain, and observed error type before drafting rule cards.

Rule cards specify how a bias should be applied. Each card records the answer comparison procedure, activation region, abstention behavior, and support examples and counterexamples that validate the proposal. A card with deterministic comparison logic proposes a code update $r _ { c } ,$ which may repair or merge an existing module path or add a new code module when the behavior is not already covered; a card that requires model judgment proposes prompt guidance $P _ { c } = \mathrm { G u i d e } ( P _ { \mathrm { f b } , t } , c )$

The promotion check serves as conservative validation because a candidate may expand direct decisions only if replay shows no direct regression on protected construction examples. Let $G _ { r } ( c ; s _ { t } )$ be the conjunction of support, counterexample, overlap, and full replay validation for a code module candidate, and let $G _ { p } ( c ; s _ { t } )$ be the validation check for prompt guidance. Let $\mathcal { C } _ { \mathrm { c o d e } }$ and $\mathcal { C } _ { \mathrm { g u i d e } }$ denote code and guidance cards. We write $s _ { t } \oplus r _ { c }$ for applying a code update that passes replay validation to $\mathrm { R U L } _ { t . }$ and $s _ { t } \oplus P _ { c }$ for revising $P _ { \mathrm { f b } , t }$ with guidance from card c. The state update is

$$
s _ { t + 1 } = \left\{ \begin{array} { l l } { s _ { t } \oplus r _ { c } , } & { c \in \mathcal { C } _ { \mathrm { c o d e } } , \ G _ { r } ( c ; s _ { t } ) , } \\ { s _ { t } \oplus P _ { c } , } & { c \in \mathcal { C } _ { \mathrm { g u i d e } } , \ G _ { p } ( c ; s _ { t } ) , } \\ { s _ { t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Thus, the construction loop uses a proposal followed by conservative validation, while updating verifier state around a fixed model rather than model parameters.

Replay validation objective. The replay validation objective formalizes conservative promotion as supported coverage under zero direct error constraints. Let $V _ { 0 } = V _ { \mathrm { b a s e } }$ . In the first construction round, residuals are collected from the fixed base verifier:

$$
R _ { 0 } = \{ x _ { i } : \ x _ { i } \in D _ { \mathrm { c o n } } , \ V _ { \mathrm { b a s e } } ( x _ { i } ) \neq y _ { i } \} .
$$

In later rounds, residuals are collected by replaying the current verifier state $( \mathrm { R U L } _ { t } , P _ { \mathrm { f b } , t } )$ on construction data. Let $K _ { \mathrm { R U L } _ { t } } ( x ) \in \{ 0 , 1 , \bot \}$ be the direct output of the current module library, where ⊥ means that all activated code modules abstain. For a candidate rule card, let $S ^ { + }$ denote positive support examples, $S ^ { - }$ <sup>−</sup> denote counterexamples, and C denote the larger construction replay pool. AUTOVERIFIER constructs code modules RUL to maximize supported direct coverage under replay constraints with no regressions:

$$
\mathrm { C o v } _ { S ^ { + } } ( \mathrm { R U L } ) = \frac { 1 } { | S ^ { + } | } \sum _ { x _ { i } \in S ^ { + } } \mathbf { 1 } [ K _ { \mathrm { R U L } } ( x _ { i } ) \neq \bot ] .
$$

$$
\begin{array} { r l } { \mathrm { E r r } _ { A } \mathrm { ( R U L ) } = \displaystyle \sum _ { x _ { i } \in A } \mathbf { 1 } \big [ K _ { \mathrm { R U L } } ( x _ { i } ) \neq \perp } & { } \\ & { \qquad \land K _ { \mathrm { R U L } } ( x _ { i } ) \neq y _ { i } \big ] . } \end{array}
$$

$$
\begin{array} { r l } { \underset { \mathrm { R U L } } { \mathrm { m a x } } } & { \mathrm { C o v } _ { S ^ { + } } ( \mathrm { R U L } ) } \\ { \mathrm { s . t . } } & { \mathrm { E r r } _ { S ^ { + } \cup S ^ { - } } ( \mathrm { R U L } ) = 0 , } \\ & { \mathrm { E r r } _ { C } ( \mathrm { R U L } ) = 0 . } \end{array}
$$

Support examples encourage intended coverage, while counterexamples and replay examples require the code module to abstain or return the construction label on nearby negatives. In this sense, the promotion check serves as conservative validation because a candidate may add supported direct decisions but cannot introduce direct regressions on protected construction replay. The fallback verifier handles abstained or novel cases after the verifier is fixed.

## 3.3 Rule Cards and Verifier Updates

AUTOVERIFIER treats residuals as evidence of recurring verifier error patterns. Repeated residuals are summarized into structured profiles containing source, domain, answer object, candidate surface form, observed label mismatch, and short error hints. The coordinator first groups profiles by answer object and observed error type, then filters out groups defined only by benchmark identity or a single example. Only after this clustering step are surviving groups written as rule cards. Each rule card is a structured proposal with activation, comparison procedure, abstention rule, support examples, counterexamples, and replay evidence.

• Rule cards with deterministic activation and comparison logic are compiled into code modules, which repair an existing module path or add a new direct decision path to the reusable library.

• Rule cards that cannot be encoded safely as deterministic code are incorporated into prompt guidance, which updates the textual prompt processed by the fixed model but never makes direct decisions.

This proposal and replay loop implements the objective above. Accepted code modules expand supported direct coverage while preserving the replay constraints.

Code module and prompt guidance paths. AU-TOVERIFIER implements two update paths. A rule card with deterministic logic first enters code audit against the current module library. A missing guard or overbroad normalizer triggers repair or merge of the existing module path, while uncovered behavior triggers a new code module. Cards that require model judgment are incorporated into prompt guidance, preserving lessons useful for judgment but unsafe for direct deterministic decisions. A deterministic coordinator applies the checks above and stores rejected candidates for later rounds.

## 3.4 Replay Validation for Promotion

This subsection defines the replay validation used to promote a candidate code module. Each candidate code module must expose activation, execution, and termination structure, specialized to binary answer verification:

• Activation condition $\alpha _ { r } \colon$ defines when the direct comparison may run.

• Execution procedure $\pi _ { r }                 \colon$ performs the deterministic answer comparison procedure.

• Termination rule $\tau _ { r } :$ returns CORRECT, IN-CORRECT, or ABSTAIN.

Candidate code modules undergo four construction data checks:

• Support: target support examples must trigger the intended activation path and return the construction label.

• Counterexamples: nearby negative examples must abstain or return the construction label.

• Overlap: the candidate must add coverage rather than duplicate existing checks.

• Replay: full construction replay must not introduce new direct errors.

The checks enforce narrow activation, nearby counterexamples, and replay protected promotion. Accepted updates are compiled into deterministic code, while failed candidates are stored to avoid repeated proposals. Support examples come from recurring verifier errors or fixtures that exercise the intended activation path. Counterexamples are nearby construction cases that share surface form but differ in answer object, polarity, unit, interval endpoint, terminal slot, or invalid output status.

Numeric tolerances, unit normalizers, terminal extraction windows, and closed vocabulary sets are written into the rule card during construction and can be changed only through construction replay. If a candidate changes a construction example that the prompt labels correctly into a direct error, duplicates an earlier check without verified gain, or depends on an unchecked semantic assumption, it is rejected.

Appendix A gives a concrete rule card example, including activation, execution, abstention, counterexamples, and promotion evidence.

## 3.5 Frozen Inference and Construction Trace

AUTOVERIFIER freezes the code modules, prompt, parser, and routing order before benchmark evaluation. This subsection defines how direct decisions and the model-based fallback are combined after construction.

Fallback verifier. The model-based fallback verifier $V _ { \mathrm { f b } }$ uses the final AUTOVERIFIER sectioned binary prompt $P _ { \mathrm { f b } }$ produced during construction. It handles examples for which no validated code module can make a direct decision. Prompt optimization uses recurring verifier errors, accepted rule cards, and rejected rule feedback to update prompt guidance while preserving the same binary interface as the official benchmark prompts. Each call receives a question, a reference answer, and a candidate answer, then emits CORRECT or INCOR-RECT. This design keeps recurring verifier errors that are not safe to encode as deterministic code available as prompt guidance while reserving deterministic promotion for rule cards with support examples.

Inference and routing. At inference, activated code modules run first in a fixed priority order set before evaluation. Let $r ^ { \star }$ be the first activated code module that returns a direct decision. If such a module outputs CORRECT or INCORRECT, $V _ { \mathrm { f b } }$ is not invoked, and later modules cannot override the decision. During construction replay, any candidate whose direct decisions conflict with an earlier promoted module on overlapping support or replay examples is rejected unless it abstains on the overlap or the coordinator updates the fixed priority order and replays the full construction pool. Only when all code modules abstain is the fallback verifier $V _ { \mathrm { f b } }$ consulted:

<table><tr><td rowspan="2">Verifier</td><td colspan="4">Accuracy (%)</td><td rowspan="2">Avg.</td></tr><tr><td>VerifyBench</td><td>VerifyBench-Hard SCI-VerifyBench</td><td></td><td>VerifierBench</td></tr><tr><td>Rule-based verifiers</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Exact Match</td><td>50.55</td><td>70.76</td><td>46.68</td><td>61.52</td><td>57.38</td></tr><tr><td>Math-Verify</td><td>66.95</td><td>76.00</td><td>60.32</td><td>59.92</td><td>65.80</td></tr><tr><td>Model-based verifiers</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>XVerify-8B-I</td><td>91.95</td><td>83.10</td><td>80.16</td><td>80.80</td><td>84.00</td></tr><tr><td>CompassVerifier-32B</td><td>92.55</td><td>87.90</td><td>86.24</td><td>90.80</td><td>89.37</td></tr><tr><td colspan="6">Scientific and tool-augmented verifiers</td></tr><tr><td>SCI-Verifier-8B</td><td></td><td>90.30</td><td>86.28</td><td>93.01</td><td>89.86*</td></tr><tr><td>CoSineVerifier-Label-32B</td><td>95.70</td><td>90.00</td><td>86.40</td><td></td><td>90.70*</td></tr><tr><td>CoSineVerifier-Tool-4B</td><td>96.60</td><td>91.90</td><td>89.70</td><td>一</td><td>92.73*</td></tr><tr><td colspan="6">Same GPT-5.4-Mini verifier interface</td></tr><tr><td>Official prompt</td><td>96.30</td><td>90.29</td><td>86.76</td><td>90.77</td><td>91.03</td></tr><tr><td>AUTOVERIFIER Prompt Only</td><td>96.20</td><td>90.49</td><td>86.80</td><td>91.66</td><td>91.29</td></tr><tr><td>AUTOVERIFIER Prompt + Code</td><td>96.75</td><td>92.43</td><td>89.92</td><td>93.11</td><td>93.05</td></tr></table>

Table 1: Results of reference-based answer verification benchmarks. The local variants use the same GPT-5.4-Mini binary verifier interface; Prompt + Code adds frozen code modules before the same AUTOVERIFIER prompt verifier. Avg. averages reported columns; starred averages use fewer than four benchmarks.

$$
V _ { \mathrm { A V } } ( x ) = \left\{ \begin{array} { l l } { \tau _ { r ^ { \star } } ( \pi _ { r ^ { \star } } ( x ) ) , } & { \mathrm { i f ~ } r ^ { \star } \mathrm { ~ e x i s t s , } } \\ { V _ { \mathrm { f b } } ( x ) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

This design keeps deterministic decisions guarded and validated by replay while maintaining flexibility for complex or ambiguous cases.

The construction process also records a round trace that is evaluated after the runtime state is frozen.

## 4 Experiments

We evaluate AUTOVERIFIER under a frozen construction protocol to test whether recurring verifier errors can be converted into reliable verifier inductive biases. Across four reference-based verification benchmarks, we compare with reported rulebased, model-based, scientific, and tool-augmented verifiers, then use prompt controls, construction replay, direct decision audits, and failure analysis to isolate the contribution of accepted code modules.

## 4.1 Experimental Setup

Construction data. We build AUTOVERIFIER from a construction pool of 5,000 examples designed to expose recurring verifier errors before benchmark evaluation. The pool combines verifier examples from VAR and the TIGER-Lab verifiersft-data dataset, pairs of questions and reference answers from SuperGPQA and WebInstruct, and constructed diagnostic examples, all normalized into the verifier schema $( \boldsymbol { q } , \boldsymbol { a } ^ { \star } , \hat { \boldsymbol { a } } , \boldsymbol { y } )$ . We select examples to cover answer objects and recurring error types, including numeric and unit normalization, symbolic or interval comparison, terminal extraction, closed choice alignment, string or sequence verification, and invalid output cases. During construction, this pool provides residuals, support examples, and counterexamples for learning verifier inductive biases, with source counts and decontamination checks reported in Appendix B.2.

Evaluation benchmarks. We evaluate the frozen verifier on four reference-based verification benchmarks: VerifyBench and VerifyBench-Hard (Yan et al., 2025), SCI-VerifyBench (Zheng et al., 2025b), and VerifierBench (Liu et al., 2025a). These benchmarks span general answer verification, hard final answer cases, scientific answer checking, and broad LLM output verification.

Baselines. We group baselines by verification style. Rule-based verifiers include Exact Match (Yan et al., 2025) and Math-Verify (Kydlícekˇ , 2025). Model-based verifiers include XVerify (Chen et al., 2025a) and CompassVerifier (Liu et al., 2025a). Scientific and tool-augmented verifiers include SCI-Verifier (Zheng et al., 2025b) and CoSineVerifier (Feng et al., 2025). For controlled attribution, we also include GPT-5.4-Mini variants under the same binary verifier interface.

Metrics. We use accuracy as the primary metric. We report macro accuracy across benchmarks and use micro accuracy weighted by dataset size for the controlled prompt comparison. We also report code module coverage, fallback verifier accuracy on examples for which all code modules abstain, fallback call reduction, and direct decision audits from evaluation records.

Construction and freeze protocol. We separate construction from benchmark evaluation by freezing every verifier component before scoring. During construction, we use GPT-5.5 to propose rule cards, code module candidates, and prompt guidance from the construction pool, while GPT-5.4- Mini runs the current fallback verifier during construction replay whenever all active code modules abstain. We admit candidate updates only after structured parsing and replay validation. Before benchmark evaluation, we freeze the module library, fallback prompt, parser, module priority order, and evaluation scripts. During benchmark evaluation, the same GPT-5.4-Mini fallback verifier is called only for examples for which all frozen code modules abstain. Appendix B reports construction data, support statistics, decontamination checks, model configuration, and LLM call accounting.

## 4.2 Main Results

Benchmark comparison. We demonstrate that AUTOVERIFIER consistently outperforms rulebased, model-based, scientific, and tool-augmented verifiers across four answer-verification benchmarks. Table 1 reports the full accuracy comparison, where AUTOVERIFIER achieves the best result on every benchmark. Relative to the best compared baseline on each benchmark, AUTOVERIFIER improves accuracy by +0.15 on VerifyBench, +0.53 on VerifyBench-Hard, +0.22 on SCI-VerifyBench, and +0.10 on VerifierBench. The final frozen verifier reaches 93.05 macro accuracy across the four benchmarks.

Controlled prompt comparison. We attribute most of the controlled gain to adding code modules, whereas prompt rewriting alone gives only a small mixed improvement. All prompt controls use GPT-5.4-Mini and the same binary verifier interface. We replay the public benchmark prompt format in the official prompt setting, and we use the sectioned binary prompt produced during construction in the AUTOVERIFIER prompt setting. For Prompt + Code, we keep the fallback model, final prompt, parser, and binary interface fixed, adding only code modules before the prompt verifier. Prompt rewriting raises macro accuracy from 91.03 to 91.29, while adding code modules raises it to 93.05 with positive gains on every benchmark. The +1.76 macro point gain also holds under micro accuracy weighted by dataset size, where the three local variants score 90.84, 91.15, and 92.95.

## 4.3 Construction Replay Trace

Round coverage. We demonstrate that accepted construction rounds expand supported direct coverage progressively as recurring residuals are converted into validated code modules. We evaluate the verifier state after each accepted round on the construction monitor and, for reference, on the four benchmarks in Figure 3. We interpret the stepwise increases as evidence that the final verifier is built through accumulated rule promotion rather than a single late update.

Accepted rule groups. We analyze the construction trace by accepted rule groups and observe that early rounds account for most covered construction examples while later rounds add smaller, more specialized updates. By Round 10, the accepted construction path covers 2,109 examples in the construction monitor, which contains 5,000 examples. Rounds 1–5 account for 1,840 of these covered examples, and Rounds 6–10 add 269 more. This split across rounds is also reflected in the accepted groups, with extraction and normalization rules entering earlier and sequence, invalid-output, and boundary rules entering later. Appendix Tables 8 and 9 report the full construction audit and one revision case.

## 4.4 Direct Decision Attribution

Direct coverage and corrections. We observe that code modules both make direct decisions and correct errors from the prompt-only verifier. Table 2 reports 2,665 direct decisions, corresponding to 32.13% fewer fallback calls, and 149 promptonly verifier errors corrected by code modules. Direct coverage is substantial across all four benchmarks (26.65%–38.04%), while corrections are largest on SCI-VerifyBench (78), where the fallback verifier is weakest on abstained examples.

Coverage versus corrections. We interpret direct coverage and accuracy contribution as related signals that are not interchangeable. Corrections are larger where the fallback prompt is weaker, while direct coverage also measures routing. We therefore treat fallback call reduction as a consequence of direct routing, not as the main evidence for inductive bias acquisition. The appendix direct decision audit reports zero benchmark label disagreements for direct decisions made by code modules.

![](images/9f02d28a05432a1106b06c34aee5afc1035ed1d36d7f3157acb7faf72449dda6.jpg)  
Figure 3: Per round replay trace. Round 0 is the baseline without code modules and Round 10 is the frozen verifier. Curves plot cumulative direct coverage on four evaluation benchmarks and the frozen construction monitor of 5,000 examples.

<table><tr><td>Bench.</td><td></td><td></td><td>Rules Direct Direct% Corr. Fallback</td><td></td><td></td></tr><tr><td>VB</td><td>29</td><td>533</td><td>26.65</td><td>11</td><td>95.57</td></tr><tr><td>Hard</td><td>31</td><td>372</td><td>38.04</td><td>19</td><td>87.79</td></tr><tr><td>SCI</td><td>41</td><td>897</td><td>35.88</td><td>78</td><td>84.28</td></tr><tr><td>VerifierBench</td><td>24</td><td>863</td><td>30.64</td><td>41</td><td>90.07</td></tr><tr><td>All</td><td>一</td><td>2,665</td><td>32.13</td><td>149</td><td>89.61</td></tr></table>

Table 2: Direct decision attribution. Direct% is the share of examples handled by code modules, equal to fallback call reduction; Corr. counts prompt errors fixed by code modules. Fallback is prompt verifier accuracy on examples for which all code modules abstain.

Negative controls. We test generic shortcut rules as an alternative explanation, but the controls do not reproduce the gains from code modules. A small generic heuristic control covers 502 examples but introduces 15 regressions, and promotion using support alone admits replay and counterexample violations. We report these controls and fallback diagnostics in Appendix C.2, and construction call accounting in Appendix D.2.

## 4.5 Bias Categories and Failure Boundaries

Bias categories. We find that learned verifier inductive biases concentrate around recurring answer forms and guard conditions tied to implicit annotation assumptions, specifying when common answer forms should be treated as equivalent or nonequivalent.

Accepted rule categories. We observe that no single rule group explains the full improvement, but each accepted group contributes a guarded portion of the gain. Table 3 reports direct decisions, corrected prompt-only verifier errors, and macro accuracy drops for each removed group. These removal results indicate that the gains come from reusable comparison logic for answer forms rather than dataset-specific patches.

<table><tr><td>Removed group</td><td>Direct</td><td>Corr.</td><td>∆Avg</td></tr><tr><td>Invalid or conflicting outputs</td><td>207</td><td>38</td><td>0.43</td></tr><tr><td>Closed choice / verdict mapping</td><td>1,016</td><td>20</td><td>0.33</td></tr><tr><td>Numeric / unit normalization</td><td>369</td><td>22</td><td>0.26</td></tr><tr><td>Boundary / format stress rules</td><td>138</td><td>24</td><td>0.21</td></tr><tr><td>Symbolic math / intervals</td><td>175</td><td>19</td><td>0.21</td></tr><tr><td>Terminal / object extraction</td><td>540</td><td>15</td><td>0.20</td></tr><tr><td>String / sequence verification</td><td>220</td><td>11</td><td>0.13</td></tr></table>

Table 3: Group removal under the same prompt verifier. ∆Avg is the macro accuracy drop when direct decisions from the group are routed back to the prompt verifier.

Failure boundaries. We characterize the remaining errors as cases that should remain with the fallback verifier. Code modules route deterministic comparison cases directly and abstain on ambiguous or underspecified judgments, keeping rule activation conservative. In a manual analysis of 80 remaining disagreement cases, 53 involve annotation or ambiguity issues and 27 are remaining verifier errors.

## 5 Conclusion

We demonstrate that AUTOVERIFIER converts recurring verifier errors into auditable comparison logic. It learns verifier inductive biases from construction residuals and promotes supported patterns into code modules or prompt guidance. Across four benchmarks, it reaches 93.05% macro accuracy, improves the prompt-only verifier by +1.76 points, and cuts fallback calls by 32.13% without changing model parameters.

## Limitations

The current scope is reference-based binary verification with a fixed construction pool and frozen inference interface. Future work should extend the same construction loop driven by recurring verifier errors to richer grading rubrics, multiple reference answers, and domain-specific evaluation standards while preserving support, counterexample, overlap, and replay validation constraints. Future work should also make construction easier to reproduce and maintain with open construction agents, versioned rule libraries, clearer record visualization, and continuous replay tests as new domains or answer formats are added.

## Ethical Considerations

AUTOVERIFIER judges candidate answers against references and may be used as a reward component in downstream filtering or training pipelines, so incorrect verifier outputs can mislead data selection when references are incomplete or ambiguous. We keep module activation conservative, retain abstention for deterministic rules, and separate benchmark scores from hard case analysis. Construction and benchmark data should be used only under their licenses and permitted use terms; scientific and medical examples are verification cases, not expert advice. If diagnostic cases are released, they should be anonymized and distributed only under the relevant dataset licenses.

## References

Jonathan Baxter. 2000. A model of inductive bias learning. Journal ofArtificial Intelligence Research, 12:149–198.

Ding Chen, Qingchen Yu, Pengyuan Wang, Mengting Hu, Wentao Zhang, Zhengren Wang, Bo Tang, Feiyu Xiong, Xinchi Li, Chao Wang, Minchuan Yang, and Zhiyu Li. 2025a. xVerify: Efficient answer verifier for reasoning model evaluations. Preprint, arXiv:2504.10481.

Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. 2025b. JudgeLRM: Large reasoning models as a judge. Preprint, arXiv:2504.00050.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Preprint, arXiv:2501.12948.

Yaxin Du, Xiyuan Yang, Zhifan Zhou, Wanxu Liu, Zixing Lei, Zimeng Chen, Fenyi Liu, Haotian Wu, Yuzhu Cai, Zexi Liu, Xinyu Zhu, WenHao Wang,

Linfeng Zhang, Chen Qian, and Siheng Chen. 2026. DataMaster: Data-centric autonomous AI research. Preprint, arXiv:2605.10906.

Ruixiang Feng, Zhenwei An, Yuntao Wen, Ran Le, Yiming Jia, Chen Yang, Zongchao Chen, Lisi Chen, Shen Gao, Shuo Shang, Yang Song, and Tao Zhang. 2025. CoSineVerifier: Tool-augmented answer verification for computation-oriented scientific questions. Preprint, arXiv:2512.01224.

Zhezheng Hao, Hong Wang, Jian Luo, Jianqing Zhang, Yuyan Zhou, Qiang Lin, Can Wang, Hande Dong, and Jiawei Chen. 2026. ReCreate: Reasoning and creating domain agents driven by experience. Preprint, arXiv:2601.11100.

Yuzhen Huang, Weihao Zeng, Xingshan Zeng, Qi Zhu, and Junxian He. 2025. From accuracy to robustness: A study of rule- and model-based verifiers in mathematical reasoning. Preprint, arXiv:2505.22203.

Hynek Kydlícek. 2025. ˇ Math-Verify: Math verification library. Software library.

Shudong Liu, Hongwei Liu, Junnan Liu, Linchen Xiao, Songyang Gao, Chengqi Lyu, Yuzhe Gu, Wenwei Zhang, Derek F. Wong, Songyang Zhang, and Kai Chen. 2025a. CompassVerifier: A unified and robust verifier for LLMs evaluation and outcome reward. Preprint, arXiv:2508.03686.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. 2025b. Inference-time scaling for generalist reward modeling. Preprint, arXiv:2504.02495.

M-A-P Team. 2025. SuperGPQA: Scaling LLM evaluation across 285 graduate disciplines. Preprint, arXiv:2502.14739.

Qirui Mi, Zhijian Ma, Mengyue Yang, Haoxuan Li, Yisen Wang, Haifeng Zhang, and Jun Wang. 2026. Skill-Pro: Learning reusable skills from experience via Non-Parametric PPO for LLM agents. Preprint, arXiv:2602.01869.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. 2025. AlphaEvolve: A coding agent for scientific and algorithmic discovery. Preprint, arXiv:2506.13131.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. 2024. Mathematical discoveries from program search with large language models. Nature, 625:468–475.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

TIGER-Lab. 2025. TIGER-Lab/verifier-sft-data. Hugging Face dataset.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An openended embodied agent with large language models. Preprint, arXiv:2305.16291.

Jiayi Weng. 2026. Learning beyond gradients. Blog post.

Yu Xia, Jingru Fan, Weize Chen, Siyu Yan, Xin Cong, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2025. AgentRM: Enhancing agent generalization with reward modeling. Preprint, arXiv:2502.18407.

Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, Yang Liu, Xin Xu, Mengdi Zhang, Jian Shao, Yongliang Shen, Jun Xiao, and Yueting Zhuang. 2025. VerifyBench: Benchmarking reference-based reward systems for large language models. Preprint, arXiv:2505.15801.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and 16 others. 2025. DAPO: An open-source LLM reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. 2024. MAmmoTH2: Scaling instructions from the web. Preprint, arXiv:2405.03548.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. 2025a. SkillWeaver: Web agents can self-improve by discovering and honing skills. Preprint, arXiv:2504.07079.

Shenghe Zheng, Chenyu Huang, Fangchen Yu, Junchi Yao, Jingqi Ye, Tao Chen, Yun Luo, Ning Ding, Lei Bai, Ganqu Cui, and Peng Ye. 2025b. SCI-Verifier: Scientific verifier with thinking. Preprint, arXiv:2509.24285.

<table><tr><td>Construction Case 1. Unit Implied Numeric Equivalence</td></tr><tr><td>Training source. TIGER construction example. Fields. The question asks for torque in Newton meters. The reference is 6.79Nm, while the candidate gives the same terminal value as the bare number 6.79. Why the table form was insufficient. A table entry saying</td></tr><tr><td>“numeric/unit normalization&quot; hides the actual scale choice. The verifier may accept the missing unit only when the task fixes the unit and the candidate supplies a unique</td></tr><tr><td>terminal scalar. Rule action. The accepted rule extracts the terminal scalar, checks that the question supplies the omitted unit, and returns CORRECT only for the supported unit preserving</td></tr></table>

## Appendix Overview

The appendix records implementation, audit, and additional evaluation details that support the main text.

<table><tr><td>Part</td><td>Contents</td></tr><tr><td>A</td><td>Section A. Illustrative construction cases and rule card anatomy. Anonymized support patterns and compact rule card contracts for terminal extrac- tion, unit normalization, verdict guards, and invalid</td></tr><tr><td>B</td><td>output guards. Section B. Construction implementation and records. Construction pipeline, model configura- tion, data pool composition, leakage audit, round trace, rule revision case, support records, and</td></tr><tr><td>C</td><td>prompt templates. Section C. Additional evaluation, controls, and diagnostics. Statistical uncertainty, direct decision audit, group removal, heuristic control, promotion check ablation, and the Qwen3-4B fallback diag-</td></tr><tr><td>D</td><td>nostic. Section D. Audit, cost, and responsible artifact notes. Manual error analysis, construction and in- ference cost accounting, artifact access policy, and screening notes.</td></tr></table>

## A Illustrative Construction Cases and Rule Card Anatomy

This section presents case evidence for construction. It starts from paraphrased construction support examples, then shows the rule card contract that records recurring verifier errors and admits code modules through replay validation. The boxed examples are deliberately compact so that the appendix remains in the ACL two column format.

## Construction Case 2. Closed Choice Polarity Conflict

Training source. VAR construction example.

Fields. The task asks for a Yes or No judgment. The reference is Yes, while the candidate’s final answer is No after a longer explanation.

Why this belongs in a case box. The error is not visible from aggregate counts. The verifier must align the final answer with the task’s closed vocabulary rather than accept explanatory text around an opposite terminal label.

Rule action. The closed choice rule extracts the unique terminal label from both sides and returns INCORRECT on the polarity conflict. If a response lacks a unique terminal label or mixes equally strong labels, the rule abstains.

## Construction Case 3. Short Answer Slot Mismatch

Training source. VAR construction example.

Fields. The task asks for a named creator. The reference is one named person, while the candidate gives a step-by-step lookup response and commits to a different person.

Why this case matters. The candidate contains plausible explanatory text, but the requested output is a compact short answer object. This is a different failure mode from numeric normalization. The verifier must reject process descriptions or wrong objects when the task asks for a compact final object.

Rule action. The short answer guard detects schema, workflow, or process outputs in short answer slots and returns INCORRECT when the required final object is absent or replaced by a different terminal object. It abstains on genuine alias or synonym questions that require semantic knowledge outside the deterministic guard.

## Rule Card Contract Encoded by the Code Module

Activation. Fire only after terminal object extraction identifies a supported answer object, such as a numeric speed ratio or a closed vocabulary verdict.

Procedure. Normalize the extracted objects, preserve tasklevel constraints such as units or verdict vocabularies, and compare only the bounded final answer region.

Abstention condition. Abstain on missing terminal answers, conflicting final values, unsupported units or vocabularies, and expressions that cannot be parsed into the expected object type.

Counterexamples. Reject or abstain on values outside tolerance, incompatible final objects, omitted final answers, and responses that mix a correct intermediate object with a different final answer.

Promotion evidence. Admit the rule only after support examples trigger the intended path, counterexamples abstain or match construction labels, overlap checks add new coverage, and full construction replay introduces no direct decision regression.

## B Construction Implementation and Records

This section records the implementation details needed to audit how recurring verifier errors become verifier behavior. It includes the construction pipeline, construction pool composition, per round growth trace, representative rule card contracts, and construction agent prompt protocols.

## B.1 Construction Implementation Details

Table 4 summarizes the construction round pipeline and the audit contract for each stage.

The implementation follows the pipeline in Table 4. It initializes an empty module library and a base prompt verifier, then replays the current verifier, clusters recurring verifier errors, drafts rule cards, audits code cards, and promotes only candidates that pass support, counterexample, overlap, and full replay validation. It repairs or merges existing code modules when a cluster exposes an existing module gap, adds a new code module only for uncovered deterministic behavior, incorporates cards that require model judgment into prompt guidance, and freezes the module order, prompt, parser, routing order, and output contract. The design adapts recurring verifier errors and procedural memory to non-parametric verifier optimization with replay validation. The empirical claim remains restricted to the frozen reference-based answer verification experiments reported in the paper.

## B.2 Construction Data Details

Construction data is converted to the canonical $( \boldsymbol { q } , \boldsymbol { a } ^ { \star } , \hat { \boldsymbol { a } } , \boldsymbol { y } )$ schema. VAR train (Chen et al., 2025a), the TIGER-Lab verifier-sft-data dataset (TIGER-Lab, 2025), and SuperGPQA-Records (M-A-P Team, 2025) are natively compatible after field parsing. WebInstruct-verified (Yue et al., 2024) is used only as a source of questions with reference answers; derived verifier examples are constructed by generating candidate responses and assigning binary labels. Construction samples are decontaminated against evaluation sources by exact and fuzzy question/reference matching before code module construction.

The construction procedure follows verifier data construction patterns used by recent verifier work, but it is specialized for deterministic rule induction rather than neural fine tuning. First, all source examples are normalized into a shared verifier schema and assigned source, answer object, and candidate surface metadata. Second, exact and fuzzy decontamination removes examples overlapping the target evaluation questions or references. Third, recurring verifier errors from the fixed construction verifier are grouped by observed error type, such as numeric/unit normalization, symbolic equivalence, terminal extraction, closed choice alignment, string/sequence verification, and invalid output detection. Fourth, candidate rule cards are promoted only when they have positive support examples, counterexamples, and replay validation with no direct decision regression. We use a single dataset size convention throughout the paper, with the construction pool defined as 5,000 source-derived and constructed examples. Internal validation records are implementation records for rule promotion, not separate data pool definitions.

The construction pool is therefore not a random sample of available records. It is a selected construction pool of 5,000 examples that combines source records with constructed diagnostic examples and balances domain coverage with answer object and error type coverage. For each target error type, the construction pipeline selects positive support examples and counterexamples from the construction pool. The module library is induced only from recurring verifier errors and support examples; every promoted code module must retain live support under promotion validation.

The final frozen construction pool contains source-derived examples and constructed diagnostic examples. The constructed examples exercise malformed outputs, invalid-answer conditions, option or verdict conflicts, ordered answer-list surfaces, terminal answer formats, and other answerformat cases that should be covered before deterministic rules are promoted. Following the leakageaudit practice used in data-centric autonomous research systems, we run a layered construction– evaluation overlap check over exact hashes, fuzzy token overlap, provenance, and 3–5-gram overlap diagnostics (Du et al., 2026). As shown in Figure 4, the final construction pool has no exact question, exact question/reference, exact verifier-triple, or fuzzy verifier-triple matches against the 8,295 evaluation examples; the only nonzero plotted signal is low verifier-triple 5-gram surface overlap. Table 7 reports the corresponding counts and diagnosticonly overlaps.

The runtime support records provide a stricter rule view. The final support set contains 2,096 live source support examples with 0 direct support errors, plus 13 constructed boundary/format validation records, giving 2,109 supported construction examples in the construction monitor over 5,000 examples. The final support records validate all 63 delivered code module behaviors with 0 missing support entries and 0 runtime errors. These support records are construction records for rule promotion, not additional data size conventions.

<table><tr><td></td><td>Step Operation</td><td>Audit contract</td></tr><tr><td></td><td>1 Normalize construction ex- Convert all sources into amples</td><td> $( \boldsymbol { q } , \boldsymbol { a } ^ { \star } , \hat { \boldsymbol { a } } , \boldsymbol { y } )$  with source, domain, answer object, and surface form metadata.</td></tr><tr><td></td><td>2 Decontaminate</td><td>Remove exact and fuzzy question/reference overlaps with all evaluation benchmarks before rule construction.</td></tr><tr><td>3</td><td>Collect residuals</td><td>Run the base verifier in the first round and the current verifier in later rounds; retain label mismatches in the residual buffer.</td></tr><tr><td></td><td>4 Cluster errors</td><td>Group residuals by answer object and observed error type, rather than by benchmark identity or evaluation labels.</td></tr><tr><td></td><td>5 Draft rule cards</td><td>For each cluster, specify activation, deterministic procedure, termination rule, support examples, and counterexamples.</td></tr><tr><td></td><td>6 Audit code cards</td><td>Compare each code card with the current module library; repair or merge an existing code module when it has the right behavior type but incomplete guards, add a new code module only when no existing path covers the behavior, and send cards that require model</td></tr><tr><td></td><td>7 Promote or reject</td><td>judgment to prompt guidance. Accept a code update only if support, counterexamples, overlap checks, and full construc-</td></tr><tr><td></td><td>8 Freeze runtime state</td><td>tion replay satisfy the direct decision promotion contract. Lock module order, prompts, parsers, routing, and output contracts after the construction rounds.</td></tr></table>

Table 4: Construction round pipeline. Each accepted update is generated from construction replay on the construction pool and promoted by replay validation before the runtime is frozen.
<table><tr><td>Component</td><td>Model</td><td>Reasoning/decoding</td><td>Role and evaluation boundary</td></tr><tr><td>Rule mining agent</td><td>GPT-5.5</td><td>xhigh reasoning</td><td>Construction only; records recurring verifier error clusters in rule cards.</td></tr><tr><td>Code generation agent</td><td>GPT-5.5</td><td>xhigh reasoning</td><td>Construction only; audits code cards against the current library, then proposes repairs,</td></tr><tr><td>Prompt optimization agent GPT-5.5</td><td></td><td>xhigh reasoning</td><td>merges, new code modules, and tests. Construction only; incorporates rule card ev- idence that requires model judgment into prompt guidance for the sectioned binary</td></tr><tr><td>Fallback verifier  $V _ { \mathrm { f b } }$ </td><td>GPT-5.4-Mini</td><td>supported</td><td>prompt. deterministic decoding when Inference/evaluation only on examples for which all code modules abstain; outputs are parsed by a strict binary parser.</td></tr></table>

Table 5: Model configuration for construction and inference. Construction agents operate only on construction replay records and cannot promote code modules without replay validation; the module library, prompt, parser, and routing order are fixed before reported inference runs.

## B.3 Round Trace Records

Figure 3 is generated from the accepted round records. The construction curve uses the single frozen construction pool of 5,000 source-derived and constructed examples, with 2,096 live support examples from source replay plus 13 constructed boundary/format validation examples, for 2,109 covered examples in the construction monitor over 5,000 examples. After the runtime is frozen, we replay the same final module library on the four benchmarks to report direct routing counts of 533 on VerifyBench, 372 on VerifyBench-Hard, 897 on SCI-VerifyBench, and 863 on VerifierBench. The benchmark counts are post-freeze routing summaries, not construction feedback.

This table is an audit record rather than a performance curve. A later round is considered more general when it increases direct coverage on the construction pool through a typed code module behavior that also retains support examples, counterexamples, and a clean construction replay. Candidates that looked useful but lacked construction support or failed counterexample replay remain in the rejected proposal or support gap records; they are not counted as accepted updates.

## B.4 Construction Side Rule Revision Case

Table 9 expands one Round 8 identifier revision from the construction records. The original support example is a SuperGPQA corn genotype verifier pair. The reference answer is AaCCRr, the candidate answer is AaCCrr, and the construction label is INCORRECT. The revision fixes an unsafe compact surface match by preserving case for mixed case scientific identifiers, then promotes a fail rule that fires only when a whole terminal identifier can be extracted from both sides.

<table><tr><td>Axis</td><td>Coverage</td><td>Role</td></tr><tr><td>Domain/source</td><td>examples 207</td><td>VAR 1630; TIGER Supplies construction questions and 1196; SuperGPQA de- references across math, science, en- rived 1014; WebIn- gineering, medicine, general reason- struct derived 953; ing, instruction following, short an- constructed diagnostic swer verification, closed choice veri- fication, and controlled comparison</td></tr><tr><td>Answer object</td><td>swer 858; open text references and candidates. 77; scientific identifier 13; list/sequence/word</td><td>symbolic 1481; nu- Covers the object types and compar- meric 1373; closed ison surfaces that deterministic ver- choice 1184; short an- ifiers must parse before comparing</td></tr><tr><td>Error type</td><td>set/equation 14 alignment; meric/unit malization; sym- bolic/interval compar- ison; terminal extrac- tion; string/sequence verification; invalid or</td><td>closed choice/verdict Guides error clustering, support ex- nu- amples, and counterexamples for nor- candidate rule cards.</td></tr></table>

Table 6: Two dimensional construction pool design. The domain and source axis supplies broad task coverage, while the answer object and error type axes ensure that construction examples expose the verifier errors that code modules are meant to solve. Counts are from the selected construction pool of 5,000 source-derived and constructed examples; promotion validation is summarized in the text.

<table><tr><td>Audit item</td><td>Construction side evidence</td></tr><tr><td>Construction example</td><td>SuperGPQA support example with reference AaCCRr, candidate AaCCrr, and label INCORRECT; internal row identifiers are omitted from the paper.</td></tr><tr><td>Observed pattern</td><td>Mixed case identifier strings may differ only by case or terminal slots; lowercase compact normalization is unsafe for genotype, alloy, spectral class, and</td></tr><tr><td>Generalized edit</td><td>formula like identifiers. Add whole terminal mixed case opaque identifier extraction and route different extracted identifiers to a direct fail. The guard abstains if the reference</td></tr><tr><td>Same group support</td><td>identifier appears anywhere in the candidate. Five SuperGPQA examples support the same group, including AaCCRr/AaCCrr, Na_2HXO_3/Na_3XO_3, M2IIIa/M2IIIb, ZCuPb30/ZPbSn10Cu5, and</td></tr><tr><td>Replay result</td><td>ACBacb/AaBcBC. Runtime replay returns INCORRECT with reason code terminal_short_slot_mismatch; clean construction replay has 1,980 direct correct decisions, 0 direct wrong decisions, and 0 runtime</td></tr></table>

![](images/fc2459c5d395f33c62df8f08adb14646d26ddb39700f8f5f637c0915bf2470df.jpg)  
Table 9. Code module construction audit case for a mixed case identifier revision. The selected source lines are copied from the frozen runtime snapshot and show the activation guard plus the direct fail routing path.

## B.5 Module Library Support

The final module library contains 63 supported code module behaviors with selected pool or fixture based support. Support is verified by replaying the accepted code modules on support tests outside evaluation. No direct errors are observed, and all 63 target behaviors have at least one live runtime support hit in the final closure pool. The largest supported groups include compact exact object extraction, closed vocabulary verdict matching, numeric and unit normalization, single choice terminal mismatch detection, interval endpoint mismatch detection, and invalid answer guards.

![](images/d45ca73387a1605aa54ba0128af2eadd9d2f1219d6eda8df8474bb2e97bd2457.jpg)  
Figure 4: Construction–evaluation leakage check for the frozen construction pool. In the contamination-audit format used by data-centric autonomous research systems, the decisive exact and fuzzy verifier-triple checks are zero; the only nonzero plotted signal is low verifiertriple 5-gram surface overlap. Diagnostic question– reference and reference-only repeats are reported in Table 7 because short answers and templates can repeat without copying full verifier examples.

Each rule card records an activation condition, deterministic execution logic, abstention conditions, support examples, and counterexamples, following the boxed contract in Section A. Code modules are promoted only when support examples trigger the intended reason, counterexamples abstain or return the construction label as expected, and full construction replay shows no direct error regression. Rejected rule cards are retained as construction feedback when they are too broad, overlap an existing module without coverage gain, or rely on semantic assumptions that cannot be checked deterministically.

## B.6 Prompt Templates

The rule mining prompt consumes recurring verifier errors, answer object metadata, support hints, and counterexamples, and must emit structured rule card candidates with activation, procedure, termination, support examples, counterexamples, and rejection risks. The code generation prompt consumes an accepted rule card plus parser utilities and replay constraints, and must emit the narrowest code module that returns CORRECT, INCORRECT, or ABSTAIN. The prompt optimization prompt consumes accepted and rejected rule contracts plus prompt verifier errors, then extracts transferable comparison principles into the same sectioned binary prompt used after module abstention.

<table><tr><td>Audit check</td><td>Field / threshold</td><td></td><td>Count Interpretation</td></tr><tr><td>Exact question hash</td><td>Normalized question</td><td></td><td>0 No construction example copies an evaluation question.</td></tr><tr><td>Exact question-reference hash</td><td>Normalized question + reference</td><td></td><td>0 No exact evaluation question-answer pair appears in con- struction.</td></tr><tr><td>Exact verifier-triple hash</td><td>Question + reference + candidate</td><td></td><td>0 No exact evaluation verifier example appears in construc- tion.</td></tr><tr><td>Fuzzy verifier-triple match</td><td>Token Jaccard &gt; 0.90</td><td></td><td>0 No near-duplicate verifier example is detected under the strict triple check.</td></tr><tr><td>Fuzzy match</td><td>question-reference Token Jaccard ≥ 0.90</td><td></td><td>84 Template-level overlap in task wording; retained as diag- nostic because the stricter verifier-triple match is zero.</td></tr><tr><td>Reference-only exact match</td><td>Normalized reference</td><td></td><td>64,091 Diagnostic only: short answers repeat across verifier tasks and are not treated as leakage without question or candi- date overlap.</td></tr><tr><td>Verifier-triple 5-gram overlap</td><td>Unique 5-grams</td><td></td><td>0.80% Low surface overlap diagnostic over full verifier triples.</td></tr></table>

Table 7: Construction–evaluation leakage audit for the final construction pool of 5,000 examples against 8,295 benchmark evaluation examples. Exact checks use normalized hashes; fuzzy checks use token-Jaccard matching at threshold 0.90.

The AUTOVERIFIER prompt verifier uses Role, Rules, Steps, and Output Contract blocks; it is not an insertion into the official benchmark prompts, which are used only as baselines. Benchmark rendering receives the question, reference answer, and candidate answer, and it is always evaluated as a binary classifier over CORRECT and INCORRECT; there is no third prompt label. Raw construction examples are not inserted into the evaluation prompt. Output parsing maps explicit correctness labels to the binary interface and treats unparseable outputs as incorrect under reported evaluation.

## C Additional Evaluation, Controls, and Diagnostics

This section keeps only the evaluation views needed for reviewer interpretation. These views are uncertainty estimates, direct decision audit, controls, promotion check ablations, and fallback diagnostics. Code module coverage is summarized in the main text; longer category breakdowns, direct decision confusion matrices, and fallback error slices are omitted because they restate the same routed examples at finer granularity.

Representative construction support cases are shown in gray boxes in Section A. The omitted per benchmark category slices, fallback error concentration, and direct decision group tables are diagnostic views of the same routed examples rather than separate evidence needed to understand the main claim.

## C.1 Statistical Uncertainty

Table 10 reports uncertainty for the AUTOVERI-FIER prompt-only and prompt-plus-code variants under the same final prompt. Accuracy intervals use Wilson intervals, and ∆ intervals use a paired normal approximation over per example correctness differences. Under this paired comparison, all four code module gains over the prompt component are positive.

We do not repeat overall F1, balanced accuracy, MCC, direct decision confusion matrices, and group level Wilson intervals because they restate the same paired evaluation records at a finer granularity; the uncertainty summary in Table 10 and the direct audit in Table 11 are the necessary checks for the appendix.

Table 11 reports the direct decision audit behind the final evaluation records. It lists direct coverage, prompt only errors corrected by code modules, direct decision label distribution, and Wilson lower bounds for observed direct accuracy. The per example records contain module predicted COR-RECT/INCORRECT counts, benchmark label counts, and per label precision/recall.

## C.2 Additional Controls and Diagnostics

The additional checks target five alternatives a reviewer may consider. The gain might come from prompt rewriting rather than code modules, be reproducible with generic rule heuristics, rely on one dominant module group, depend on weak replay validation, or be tied to the proprietary GPT fallback. The same prompt comparison addresses the first alternative. The AUTOVERIFIER component that uses only the prompt improves the four benchmark average over official GPT-5.4-Mini prompts from 91.03 to 91.29 (+0.26), although individual benchmark deltas are mixed. The deltas are -0.10 on VerifyBench, +0.20 on VerifyBench-Hard, +0.04 on SCI-VerifyBench, and +0.89 on Verifier-Bench. Adding code modules to the same prompt verifier then improves VerifyBench from 96.20 to 96.75 (+0.55), VerifyBench-Hard from 90.49 to 92.43 (+1.94), SCI-VerifyBench from 86.80 to 89.92 (+3.12), and VerifierBench from 91.66 to 93.11 (+1.45). The four benchmark average improves from 91.29 to 93.05 (+1.76) after adding code modules. This paired comparison fixes both the fallback model and the final prompt, so the gain is attributed to code modules rather than to another prompt change.

<table><tr><td>Round</td><td>Generalized construction scope</td><td>Contracts / groups</td><td>Construction coverage</td><td>Audit interpretation</td></tr><tr><td>0</td><td>No code modules</td><td>0/0</td><td>0 → 0</td><td>Baseline with all construction examples routed to the fallback verifier.</td></tr><tr><td>1</td><td>Exact object and terminal slot rules</td><td>5/5</td><td>+1,117 → 1,117</td><td>Repeated exact terminal objects are consolidated into reusable activation regions instead of exceptions tied to individual examples.</td></tr><tr><td>2</td><td>Closed option terminal rules</td><td>12 / 12</td><td>+144 → 1,261</td><td>Option rules generalize over recoverable option maps and terminal option surfaces, with nearby</td></tr><tr><td>3</td><td>Closed vocabulary verdict and polarity rules</td><td>5/5</td><td>+102 → 1,363</td><td>option conflicts retained as counterexamples. Verdict and polarity phrases are promoted as typed closed vocabulary comparators rather than free</td></tr><tr><td>4</td><td>Numeric, unit, and scientific notation rules</td><td>16 /16</td><td>+315 → 1,678</td><td>form string matches. Numeric rules add tolerance, unit, and notation boundaries only when construction replay</td></tr><tr><td>5</td><td>Symbolic, formula, and interval rules</td><td>13 / 13</td><td>+162 → 1,840</td><td>preserves label aligned direct decisions. Algebraic and interval rules broaden comparison beyond exact text while keeping endpoint, set, and</td></tr><tr><td>6</td><td>Sequence, string, and invalid output rules</td><td>11 / 11</td><td>+70 → 1,910</td><td>surface form counterexamples. Structured string and invalid answer guards add direct failures for recurring terminal object errors and abstain outside their typed region.</td></tr><tr><td>7</td><td>Terminal no answer plus angle/ratio support</td><td>8/8</td><td>+40 → 1,950</td><td>The audit record keeps narrow terminal non answer, angle, and ratio rules after support reselection and construction replay, not benchmark</td></tr><tr><td>8</td><td>Unit rate and opaque identifier rules</td><td>4/4</td><td>+45 → 1,995</td><td>only hits. Identifier and SI rate rules are admitted only with whole terminal extraction and unit context guards, turning earlier boundary fixes into reusable rules.</td></tr><tr><td>9</td><td>Terminal phrase and short phrase expansions</td><td>4/3</td><td>+54 → 2,049</td><td>Phrase rules add bounded terminal window containment; broader substring candidates remain rejected or routed to fallback.</td></tr><tr><td>10</td><td>Scientific relative, prefix, unit bare, and boundary/format rules</td><td>5/5</td><td>+60 → 2,109</td><td>The final checkpoint closes the construction monitor used by the figure; the separate support closure replay validates all 63 delivered code module behaviors.</td></tr></table>

Table 8: Construction generalization audit trace for the ten accepted optimization rounds. Coverage is measured only on the frozen construction monitor used by Figure 3; benchmark examples are not used to promote a code module. Each row records the generalized rule scope that entered the module library after support, counterexample, overlap, and construction replay validation
<table><tr><td>Benchmark</td><td>Rows</td><td></td><td>Prompt</td><td>AUTOVERIFIER</td><td> $\Delta$ </td><td>95% CI for  $\Delta$ </td></tr><tr><td>VerifyBench</td><td>2000</td><td></td><td>96.20 [95.27, 96.95]</td><td>96.75 [95.88, 97.44]</td><td>+0.55</td><td> $[ + 0 . 2 3 , + 0 . 8 7 ]$ </td></tr><tr><td>VerifyBench-Hard</td><td>978</td><td></td><td>90.49 [88.49, 92.17]</td><td>92.43 [90.60, 93.93]</td><td> $+ 1 . 9 4$ </td><td> $\left[ + 1 . 0 8 , + 2 . 8 1 \right]$ </td></tr><tr><td>SCI-VerifyBench</td><td>2500</td><td></td><td>86.80 [85.42, 88.07]</td><td>89.92 [88.68, 91.04]</td><td> $+ 3 . 1 2$ </td><td> $[ + 2 . 4 4 , + 3 . 8 0 ]$ </td></tr><tr><td>VerifierBench</td><td>2817</td><td></td><td>91.66 [90.58, 92.62]</td><td>93.11 [92.12, 93.99]</td><td> $+ 1 . 4 5$ </td><td> $[ + 1 . 0 1 , + 1 . 9 0 ]$ </td></tr></table>

Table 10: Statistical uncertainty for AUTOVERIFIER prompt and prompt plus code results under the same prompt verifier. Confidence intervals for $\Delta$ use a paired normal approximation over per example correctness differences.

Table 3 in the main text reports an offline removal ablation under the same prompt verifier. For each group, direct decisions from that group are replaced by the prompt verdict, while all other code module decisions and prompt calls are unchanged. This isolates the marginal contribution of each module group without introducing additional model calls. A group can therefore have high direct coverage but a modest accuracy drop if most of its direct decisions were already correct under the prompt verifier; the corrected count measures the stricter contribution to accuracy. High value code modules are those that remove systematic prompt failures, not necessarily those that fire most often.

The per benchmark removal and cumulative routing views are omitted from the appendix table set because they are deterministic decompositions of the same evaluation records. Invalid or conflicting output guards matter most on SCI-VerifyBench, closed choice and verdict mapping matter most on VerifyBench-Hard and VerifierBench, and the cumulative route reaches 2,665 direct decisions,

<table><tr><td>Benchmark</td><td></td><td></td><td>Direct P-corr. P-err. fixed M-err.</td><td></td><td></td><td>Eq. Not-eq. Wilson LB</td><td></td></tr><tr><td>VerifyBench</td><td>533</td><td>522</td><td>11</td><td>0</td><td>353</td><td>180</td><td>99.28</td></tr><tr><td>VerifyBench-Hard</td><td>372</td><td>353</td><td>19</td><td>0</td><td>131</td><td>241</td><td>98.98</td></tr><tr><td>SCI-VerifyBench</td><td>897</td><td>819</td><td>78</td><td>0</td><td>571</td><td>326</td><td>99.57</td></tr><tr><td>VerifierBench</td><td>863</td><td>822</td><td>41</td><td>0</td><td>621</td><td>242</td><td>99.56</td></tr><tr><td>All</td><td>2,665</td><td>2,516</td><td>149</td><td></td><td>01,676</td><td>989</td><td>99.86</td></tr></table>

Table 11: Direct decision audit under the fixed final verifier. P-err. fixed counts direct decisions for which the AUTOVERIFIER prompt verifier is wrong and the code module is correct. M-err. counts code module disagreements with the benchmark label. Wilson LB is the lower bound of a two sided 95% Wilson interval for direct accuracy.

32.13% fallback call reduction, and 149 prompt errors corrected after all promoted code modules are enabled.

Table 12 adds a small frozen rule control. It uses only generic exact final answer matching, single choice option comparison, closed vocabulary verdict comparison, and exact numeric equality checks, then falls back to the same prompt verifier on all other examples. The control covers fewer examples than code modules learned from recurring verifier errors, has nonzero direct errors, and slightly reduces final micro accuracy relative to the prompt verifier. This negative control is deliberately conservative. It tests whether the improvement can be explained by obvious deterministic shortcuts before crediting the construction process. Its regressions show why AUTOVERIFIER promotes code modules with abstention guards admitted by replay validation instead of always applying broad exact matching heuristics.

Table 13 reports an archived candidate promotion check ablation. We collect 200 generated candidate code modules from saved construction records, including support checks, counterexample checks, construction replay summaries, protected construction-slice summaries, and candidate check reports. The strict policy reproduces the full promotion contract on these archived candidates. We then weaken one acceptance condition at a time and measure the direct errors or construction violations that would have entered the module library. This experiment audits the promotion checks over generated candidates using saved construction records, rather than reconstructing alternate checks from scratch. The important pattern is that weaker checks accept more candidates but also admit errors. Removing full replay accepts 51 candidates and admits 17 replay errors, while promotion using support alone accepts 95 candidates but admits replay, protected-slice, and counterexample failures. Thus the promotion check ablation supports precision preserving promotion rather than merely showing that the strict policy is conservative.

Table 14 reports a public 4B class fallback diagnostic over the full benchmark suite. We keep the module library, routing records, parser, and direct decisions fixed, replace only the fallback verifier with local Qwen3-4B, and evaluate all 8,295 benchmark examples. The run uses one record per context, strict binary parsing, and counts unparseable fallback outputs as incorrect. This diagnostic checks whether the same code module protocol improves a fallback model outside GPT under fixed examples and parser settings.

Group removal, the frozen rule control, the archived candidate promotion check ablation, and the full Qwen3-4B fallback diagnostic are the completed checks used for the current accuracy attribution and robustness diagnosis. The promotion check ablation supports the practical need for protected construction-slice replay, precision checks, counterexamples, and full replay over archived generated candidates. The Qwen3-4B diagnostic shows that the same code modules improve a public 4B fallback from 89.31 to 91.62 macro accuracy while avoiding 32.13% of fallback calls. We treat this as a diagnostic of replacing the fallback, not as a claim that the rule library is invariant across all fallback models.

The official prompt baseline uses the public benchmark prompt format with GPT-5.4-Mini and the same binary label parser. Under the same binary evaluation protocol, the official prompt runs reach 96.30% on VerifyBench full, 90.29% on VerifyBench-Hard, 86.76% on SCI-VerifyBench, and 90.77% on VerifierBench.

## D Audit and Cost

This section keeps the audit material needed to interpret the reported numbers. It includes the manual error analysis and construction versus inference cost accounting.

<table><tr><td>Benchmark</td><td>Rows</td><td>Direct</td><td>Direct Corr.</td><td>Coverage</td><td></td><td>Prompt Acc. Heur.+Prompt Acc.</td><td>Reg.</td></tr><tr><td>VerifyBench</td><td>2,000</td><td>278</td><td>96.76</td><td>13.90</td><td>96.20</td><td>95.80</td><td>9</td></tr><tr><td>VerifyBench-Hard</td><td>978</td><td>224</td><td>95.98</td><td>22.90</td><td>90.49</td><td>90.18</td><td>6</td></tr><tr><td>SCI-VerifyBench</td><td>2,500</td><td>0</td><td>一</td><td>0.00</td><td>86.80</td><td>86.80</td><td>0</td></tr><tr><td>VerifierBench</td><td>2,817</td><td>0</td><td>一</td><td>0.00</td><td>91.66</td><td>91.66</td><td>0</td></tr><tr><td>All</td><td>8,295</td><td>502</td><td>96.41</td><td>6.05</td><td>91.15</td><td>91.02</td><td>15</td></tr></table>

Table 12: Frozen rule heuristic control. The control uses only generic exact normalized final answer match, single choice option comparison, closed vocabulary verdict comparison, and exact numeric equality, with the same prompt verifier as fallback. Reg. counts examples where the prompt verifier was correct but the rule direct decision is wrong. This control performs no model calls and does not use generated AUTOVERIFIER code modules.
<table><tr><td>Promotion policy</td><td>Eval.</td><td>Accept</td><td>Replay</td><td>Replay err.</td><td>Slice err.</td><td>Slice fail</td><td>Ctr-ex</td></tr><tr><td>Full checks</td><td>200</td><td>20</td><td>20</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>w/o protected replay</td><td>102</td><td>28</td><td>28</td><td>0</td><td>9</td><td>8</td><td>0</td></tr><tr><td>w/o precision regression check</td><td>102</td><td>23</td><td>23</td><td>3</td><td>0</td><td>0</td><td>0</td></tr><tr><td>w/o counterexample check</td><td>88</td><td>14</td><td>14</td><td>0</td><td>0</td><td>0</td><td>21</td></tr><tr><td>w/o full replay</td><td>186</td><td>51</td><td>32</td><td>17</td><td>9</td><td>3</td><td>0</td></tr><tr><td>Support only</td><td>186</td><td>95</td><td>55</td><td>36</td><td>22</td><td>5</td><td>286</td></tr></table>

Table 13: Archived candidate promotion check ablation over saved construction records. Eval. counts candidates evaluable under the weakened policy; Replay counts accepted candidates with saved full replay summaries. Error columns count replay, protected-slice, and counterexample violations admitted by each policy.

## D.1 Manual Error Analysis

We conduct the manual error analysis after removing all direct code module decisions. For each sampled disagreement, the authors inspect the question, reference answer, candidate answer, benchmark label, and model verdict, without seeing the rule reason or construction history. Each case is assigned to one of four diagnostic categories: true verifier error, ambiguous or underspecified reference, multiple valid answers, or likely label/reference issue. An additional author reviews each assignment and records a natural language justification and, for math/science cases, a normalized object comparison when possible. The analysis covers 80 remaining disagreement cases, 20 from each benchmark, and stores one final diagnostic category per case, summarized in Figure 5. We use this analysis for hard case diagnosis; benchmark scores continue to use the original benchmark labels, and diagnostic counts are excluded from rule promotion.

The diagnostic categories are true verifier error, ambiguous or underspecified reference, multiple valid answers, and likely label/reference issue. The accompanying analysis report also records the longer taxonomy of reference sensitive disagreements, cases involving implicit assumptions in annotations, sparse reasoning failures, domain external equivalence, and mixed or overlong responses. These categories are diagnostic only. Benchmark scores continue to use the original benchmark labels, and diagnostic labels are never fed back into rule promotion.

![](images/acdebe1328c953a0bab3f267effbeb2307fb16736ff916a38aa447dc8eefc82a.jpg)  
Figure 5: Manual error analysis of 80 remaining disagreement cases. Each benchmark contributes 20 cases. Counts are diagnostic only and are not used for construction round rule updates or prompt revision.

## D.2 Cost and Audit Accounting

Table 15 separates construction time accounting from inference time savings. Code modules replace fallback calls on direct decisions after the verifier is frozen; construction agent calls are counted separately from reported inference calls.

## D.3 Responsible Artifact and Checklist Notes

Table 16 summarizes the artifact access policy used for this work. We use public or derived sources only for verifier construction, benchmark evaluation, and audit records. The supplementary package contains an artifact README, requirements file, prompt templates, anonymized synthetic examples, and alignment notes; it should contain derived schemas, hashes, code modules, prompts, routing ledgers, and aggregate audit records rather than redistributing raw third-party examples unless their original licenses and terms permit redistribution.

<table><tr><td>Benchmark</td><td>Rows</td><td>Qwen3-4B prompt</td><td>Direct decisions</td><td>Calls↓</td><td>Prompt+Code</td></tr><tr><td>VerifyBench</td><td>2,000</td><td>95.40</td><td>533</td><td>26.65</td><td>96.40</td></tr><tr><td>VerifyBench-Hard</td><td>978</td><td>90.39</td><td>372</td><td>38.04</td><td>92.43</td></tr><tr><td>SCI-VerifyBench</td><td>2,500</td><td>83.64</td><td>897</td><td>35.88</td><td>86.44</td></tr><tr><td>VerifierBench</td><td>2,817</td><td>87.82</td><td>863</td><td>30.64</td><td>91.20</td></tr><tr><td>Macro avg.</td><td></td><td>89.31</td><td>一</td><td></td><td>91.62</td></tr><tr><td>All examples</td><td>8,295</td><td>88.69</td><td>2,665</td><td>32.13</td><td>91.16</td></tr></table>

Table 14: Full Qwen3-4B open fallback diagnostic. The module library, routing records, parser, and direct decisions are fixed; only the fallback verifier is swapped from GPT-5.4-Mini to local Qwen3-4B. Prompt+Code routes direct decisions through the same code modules and sends abstained examples to Qwen3-4B. Calls↓ is the percentage of examples that avoid a Qwen fallback call. Macro averages are unweighted over the four benchmarks; All examples are size weighted.
<table><tr><td>Component</td><td>Model/code</td><td>Rows/calls</td><td>Calls saved</td><td>Accounting note</td></tr><tr><td>Construction agents</td><td>GPT-5.5 xhigh</td><td>Offline calls</td><td></td><td>Construction only rule mining, code generation, and prompt optimization. These calls are counted sepa- rately from inference and are not included in fallback-</td></tr><tr><td>Submission session cost</td><td>Codex submission sessions 6.97B tokens</td><td></td><td></td><td>call reduction. The submission sessions record 6.71B cached input tokens and 18.77M output tokens. Under standard API token prices, this corresponds to approximately $765.65 if charged as GPT-5.4-Mini, $2.55K as GPT-</td></tr><tr><td>Prompt only inference</td><td>GPT-5.4-Mini</td><td>8,295 fallback calls</td><td></td><td>5.4, or $5.10K as GPT-5.5. Controlled baseline with every benchmark example sent to the binary prompt verifier.</td></tr><tr><td>Prompt + Code inference</td><td>GPT-5.4-Mini + code mod- 5,630 fallback calls ules</td><td></td><td>2,665</td><td>Final verifier: code modules route 2,665 examples directly, reducing fallback calls by 32.13%.</td></tr><tr><td>Open fallback diagnostic</td><td>Qwen3-4B + code modules 5,630 fallback calls</td><td></td><td>2,665</td><td>Same code module records and parser; only the fall- back verifier is swapped.</td></tr><tr><td>Direct code execution</td><td>Runtime code</td><td>2,665 direct deci- - sions</td><td></td><td>Direct decisions have no model token cost at inference. Runtime code, route, rule id, prompt verdict, and final verdict are recorded in the evaluation records.</td></tr><tr><td>Evaluation audit</td><td>Runtime records</td><td>All reported exam- - ples</td><td></td><td>Recomputes macro/micro accuracy, direct decision confusion, fallback calls, and paired prompt vs code deltas from the frozen evaluation records</td></tr></table>

Table 15: Cost and audit accounting. Construction calls are counted separately from inference calls; inference efficiency is reported as fallback call reduction after the verifier is frozen.
<table><tr><td>Artifact class</td><td>Examples</td><td>Use in this paper</td><td>Access and release policy</td></tr><tr><td>Construction sources</td><td>verified questions</td><td>data, SuperGPQA-Records, WebInstruct- amples, and counterexamples</td><td>VAR train, TIGER-Lab verifier-sft- Construction replay, support ex- Governed by the original source licenses and access terms. We do not assert a new license over raw source records; derived manifests and hashes can be shared when redistribution of raw examples is not</td></tr><tr><td>Constructed diagnostics</td><td>conflict, and terminal-answer cases</td><td>Boundary, format, invalid-output, option- Construction support and replay validation</td><td>permitted. Generated verifier examples are released only after removing identifying strings and only with enough metadata to reproduce support and counterexample</td></tr><tr><td>Evaluation benchmarks</td><td>VerifyBench, and VerifierBench</td><td>VerifyBench, VerifyBench-Hard, SCI- Frozen benchmark evaluation</td><td>roles. Benchmark data remains under the original bench- mark licenses and terms. Evaluation ledgers report benchmark identifiers, routes, module ids, prompt verdicts, final verdicts, parser status, and hashes</td></tr><tr><td>Verifier records</td><td>Code modules, prompts, parser settings, Reconstructing reported routing, The release package should exclude API keys, ab- priority order, and route ledgers</td><td>direct decisions, and fallback solute local paths, author-identifying metadata, and calls</td><td>needed to recompute reported tables. non-anonymous repository links. Model calls to pro- prietary services are documented by model name and role rather than by undisclosed model parameters.</td></tr><tr><td>Manual audit records</td><td>Author audit of 80 remaining disagree- Diagnostic taxonomy only ment cases</td><td></td><td>We report aggregate categories in the paper. Case- level diagnostic records should be released only when source licenses allow and after redacting iden- tifying or sensitive text fields.</td></tr></table>

Table 16: Artifact access, intended use, and release policy. The table distinguishes raw third-party sources from derived records that are sufficient for auditing the reported verifier behavior.

Before sharing derived records, we screen released text fields for personally identifying strings, contact information, author or local path metadata, and offensive content that is not necessary for verifier auditing. Flagged fields are removed, redacted, or replaced by hashes when the raw text is not needed to recompute a reported result. The construction and evaluation records are intended for reference-based verifier research and audit replication, not for high-stakes grading, medical advice, or deployment as an expert system.

The manual error analysis in Section D.1 is an internal author audit rather than a recruited humansubjects annotation study. No external annotators are recruited, no demographic attributes are collected, and the audit labels are not used for rule promotion or benchmark relabeling. If future work uses external annotators, it should report task instructions, recruitment, compensation, consent, and institutional review or exemption status in the artifact documentation.