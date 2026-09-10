# DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs

## Abstract

Bhuvan Arora1\* Devesh Saraogi1 \* Sravya Varada1 \* Dhruv Kumar 1

Large language models (LLMs) are increasingly deployed in globally used assistants, yet their default choices in culturally grounded everyday situations can systematically favour some cultures over others, affecting localisation, user trust, and equitable behaviour. Existing cultural benchmarks evaluate accuracy against a single "correct" answer, making it difficult to characterise an LLM's cultural preference prior when multiple culturally grounded responses are all valid; they also conflate default preferences with context-driven adaptation. We propose DiSCo, a distribution-first forced-choice evaluation framework that isolates default cultural priors and tests steerability via a four-level context gradient (C0— C3). Using DiSCo-Bench (304 items) derived from BLEnD spanning 12 cultures, we evaluate six diverse instruction-tuned LLMs. Default priors are heavily concentrated, with UK and US together absorbing approximately 35% of all selections despite representing only 2 of 12 cultures. Most critically, prompt-based steering consistently widens the selection gap between highand low-resource cultures, and injecting explicit cultural facts produces negligible distributional disruption, confirming that cultural preference bias cannot be resolved through prompt-based personalisation alone.

## 1. Introduction

## 1.1. Background & Motivation

Large language models (LLMs) have rapidly transitioned from research artefacts to widely deployed systems that mediate writing, translation, tutoring, customer support, and everyday decision-making for users across regions and languages (Bommasani et al., 2021; Liang et al., 2022). As this deployment becomes global, concerns shift from purely linguistic competence to how models behave across diverse cultural expectations, value systems, and norms, especially when model outputs implicitly recommend “appropriate" actions (Bender et al., 2021; Weidinger et al., 2021; Blodgett et al., 2020). This is practically consequential: culturally mismatched suggestions can reduce trust, create friction in localised products, and amplify representational harms for communities already underrepresented in training data. At the same time, alignment and post-training methods improve instruction following and safety, but do not guarantee culturally pluralistic behaviour or consistent adaptation to local contexts (Ouyang et al., 2022). Outside NLP, cross-cultural research has long documented structured variation in values and social norms across societies, supported by established frameworks and large-scale survey instruments (Hofstede, 2001; Schwartz, 2012; World Values Survey Association, 2022). These realities motivate evaluation paradigms that treat “culture" as a first-class axis, alongside language, when assessing LLM behaviour (Tao et al., 2024).

## 1.2. Problem Statement

We study cultural preference bias in forced-choice everyday scenarios: given an ordinary lifestyle prompt and multiple culturally grounded response options that are all plausible, what selection distribution does an LLM exhibit by default, and how does that distribution change as cultural context becomes more explicit? Concretely, each instance consists of a short scenario and four candidate options (A– D), and the model outputs one choice; the main object of interest is not accuracy but the distribution of selections and its sensitivity to context. This setting is non-trivial because there is no single “correct" answer (different cultures may justify different choices) and because multiple-choice formats can introduce artefacts (e.g., option/letter preferences, and as we show, primacy bias in context fact ordering) that confound measured cultural effects (Zheng et al., 2024; Pezeshkpour & Hruschka, 2024).

We define cultural preference bias as the systematic tendency of a language model to select options associated with certain cultures at rates significantly above their fair share, while consistently underselecting options associated with other cultures, independent of any instruction or contextual signal. This differs from value-alignment bias, which measures whether a model's expressed opinions match nationally representative surveys (Tao et al., 2024), and from cultural knowledge bias, which measures factual recall accuracy (Chiu et al., 2025). Cultural preference bias is a behavioural and distributional property: it manifests in which culture's option the model gravitates toward when all options are equally culturally grounded. A model with zero cultural preference bias would select each culture's option at the exposure-normalised rate of 1/|C| across a balanced benchmark. Deviation from this uniform distribution, quantified here via KL divergence and Gini coefficient, constitutes the bias we measure. This form of bias has direct equity implications: cultures systematically underselected by default are also, as we demonstrate, harder to steer toward through prompting, creating a compounding disadvantage for users from already-underrepresented cultural backgrounds.

## 1.3. Existing Work & Research Gaps

A growing body of work evaluates culture, norms, and social bias in LLMs through datasets, benchmarks, and auditing frameworks. Cultural knowledge benchmarks test whether models know culturally situated facts and conventions, including everyday knowledge across diverse societies (Myung et al., 2024; Chiu et al., 2025). Complementary efforts focus on values and normative judgements, probing what models deem acceptable, appropriate, or desirable across cultural contexts (Rao et al., 2025; Zhao et al., 2024) and comparing model outputs to human survey or polling distributions to quantify “whose opinions"models reflect (Santurkar et al., 2023; Durmus et al., 2023). In parallel stereotype and bias benchmarks measure representational harms using controlled minimal pairs and structured QA setups (Nangia et al., 2020; Nadeem et al., 2021; Parrish et al., 2022; Dhamala et al., 2021), with extensions toward broader geo-cultural coverage and multilingual analysis (Jha et al., 2023; Nie et al., 2024; Gupta et al., 2024). Recent cultural alignment work also explores whether prompting, persona conditioning, or training objectives can steer models toward culturally aligned behaviour (Tao et al., 2024; Li et al., 2024; AlKhamissi et al., 2024).

Despite this progress, two practical gaps remain for realworld deployments. First, many evaluations implicitly assume a correct answer (knowledge) or a target distribution (opinion alignment), which makes it difficult to characterise an LLM's default cultural preference prior in settings where multiple responses are valid and culturally grounded. Second, forced-choice evaluations can be unstable: LLMs may rely on superficial selection heuristics, show sensitivity to option order or labels, or, as we demonstrate in this work, exhibit primacy bias in context fact ordering when all cultural facts are injected simultaneously, potentially masquerading as “cultural preference" unless controlled (Zheng et al., 2024; Pezeshkpour & Hruschka, 2024; Liu et al., 2024). These issues motivate a distribution-first approach that isolates priors, measures steerability, and explicitly accounts for multiple-choice artefacts including context position effects.

## 1.4. Proposed Approach

We propose DiSCo, a distribution-first forced-choice evaluation framework designed to separate default priors from contextual adaptation. Our key idea is to evaluate the same culturally grounded lifestyle items under a graded context escalation: C0 provides no cultural signal and estimates the model's baseline cultural preference distribution; C1 adds a lightweight location cue; C2 adds an explicit directive to choose what is locally appropriate; and C3 injects peroption cultural facts without specifying a target culture, testing whether grounding information alone disrupts prior preferences. Across conditions, we compute interpretable distributional measures (e.g., concentration and shifts) and steerability measures (e.g., compliance and residual stickiness), while incorporating robustness controls: (i) four option-order rotations to reduce letter-position confounds, and (ii) four independent context-order rotations in C3 to detect and mitigate primacy bias.

## 1.5. Experimental Overview & Main Results

We conduct evaluation on DiSCo-Bench (Myung et al., 2024) spanning 12 countries, evaluating six diverse instruction-tuned LLMs under the CO-C3 context gradient with four option-order rotations (and four additional context-order rotations for C3). We measure (i) the baseline selection distribution under no context, (ii) the extent to which location cues and explicit “locally appropriate" instructions shift selections toward the intended cultural frame, and (iii) whether untargeted cultural fact injection changes the baseline distribution.

Key findings: all models show a concentrated UK/USdominant default prior; C1 compliance ranges from 0.43 to 0.57 (well above random but with large cross-culture variance); C2 reduces but does not eliminate stickiness (PSI = 0.34–0.50); Jensen-Shannon Divergence (JSD) remains negligible across all models (≤ 0.018). Primacy bias is present but controlled via four independent context-order rotations in C3.

## Contributions.

• We introduce DiSCo, a context-escalated, distributionfirst evaluation protocol (CO–C3) that separates default cultural preference priors from context-driven adaptation in forced-choice settings.

• We construct and release the DiSCo Dataset (150,816 rows), a 150,816-row BLEnD-derived cultural preference dataset by generalising BLEnD's 393 MCQ questions to country-neutral form, generating peroption one-line cultural fact strings, and filtering dummy-culture rows. From the DiSCo Dataset we derive DiSCo-Bench, a compact 304-item, 12- country evaluation benchmark where all four options per item are equally valid culturally-grounded answers, enabling preference-distribution measurement rather than accuracy scoring. Both are publicly available at https://huggingface.co/datasets/ DiSCo2026/DiSCo\_Dataset\_and\_Benchmark.

• We introduce two metrics (Signal Lift and Prior Stickiness Index) and assemble an evaluation suite that adapts five established measures (KL Divergence, Gini, Compliance Rate, SPD, and JSD) to the cultural preference bias setting via an exposure-normalised Cultural Selection Distribution.

• Through evaluation of six LLMs, we show that cultural preference priors persist across model families despite explicit cultural prompting and added cultural facts, highlighting a gap between surface-level personalisation and deeper cultural adaptation.

## 2. Methodology

## 2.1. Overview

We benchmark cultural preference bias using a controlled forced-choice setup where each scenario presents four culturally grounded yet equally valid lifestyle options, and we test how model choices change as cultural context increases from C0 to C3. The pipeline proceeds as follows: scenarios are cleaned and normalised, prompts are generated under four context conditions, the LLM is queried, and outputs are mapped back to cultures to compute bias and controllability metrics. Figure 1 illustrates the end-to-end pipeline. To control for letter-position and primacy bias, all conditions use four cyclic option-order rotations, and C3 additionally applies four independent context-order rotations (full details in Appendix F).

## 2.2. DiSCo Dataset & Benchmark

What it is. DiSCo-Dataset is a 150,816-row cultural preference dataset where every row presents a country-neutral lifestyle question with four answer options, each option paired with a one-line cultural fact string (Context\_L1— L4) and a hidden culture-to-option metadata mapping used only for evaluation. Crucially, all four options per item are equally valid and culturally grounded — there is no single correct answer. This design isolates cultural preference from factual recall.

Key statistics. The dataset spans 304 unique questions, 12 cultural regions (Algeria, Assam, China, Ethiopia, Greece, Indonesia, Northern Nigeria, South Korea, Spain, UK, US,

West Java), and three lifestyle categories (food, family, leisure). From it we derive DiSCo-Bench: a compact 304- item evaluation benchmark selecting one balanced row per question across 12 cultures. Culture frequency across option slots is approximately uniform (counts range 65–136 against an expected 101); Figure 4 in Appendix B shows the full distribution.

How we built it. Starting from BLEnD's (Myung et al., 2024) 393 country-specific MCQ lifestyle questions, we (i) rewrote each question to country-neutral form via an LLM pipeline, (ii) generated per-option one-line cultural fact strings, and (iii) filtered dummy-culture rows, yielding DiSCo-Dataset. Full construction details are in Appendix B.

Availability. DiSCo-Dataset and DiSCo-Bench are publicly available at https://huggingface.co/datasets/ DiSCo2026/DiSCo\_Dataset\_and\_Benchmark.

## 2.3. Core Evaluation Protocol (Context Gradient C0-C3)

Our primary methodological contribution is a four-level context gradient that separates default cultural priors from context-driven controllability and fact-driven disruption in a setting where all choices are equally valid.

C0 — No Context (default prior). The model receives only the question, instructions, and options, with no identity or factual cultural signal, so selections reflect the model's unconstrained prior.

C1 —Location Hint (minimal identity signal). We add a single location line (User\_Info) indicating a target culture; each item is run once per culture present in that item (4 runs per item).

C2 — Location + Intent (max identity signal). We add both User\_Info and a fixed intent instruction (User\_Instruction): “The User wants a locally appropriate, familiar realistic choice." This is held constant across all items and cultures, again rotating the target culture per item (4 runs per item).

C3 — Fact Injection with Independent Context Rotation (no target). We inject all four per-option cultural facts (Context\_L1-L4) simultaneously, without specifying any target culture, to test whether explicit written evidence disrupts the default distribution. Critically, we apply independent cyclic rotations to both option order and context fact order, yielding a 4 × 4 = 16 combination design per scenario. This design isolates primacy bias (Appendix F): models may favour whichever option has its cultural fact listed first, irrespective of content. By decoupling context order from option order, we can directly measure and control for this artefact. Figure 5 (Appendix B.4) illustrates the exact prompt structure for each condition.

![](images/1962bd427e4cf386392a0de424b409f3b62b0a8df593a33ea111912ff59f527c.jpg)  
Figure 1. End-to-end pipeline for evaluating cultural preference bias across a four-level context gradient (C0–C3). C3 additionally applies independent context-order rotations to measure and control for primacy bias.

Run accounting. In total, this yields 12,160 runs per model (plus 3,648 supplementary C3 context-rotation runs for primacy bias analysis). Full run accounting is in Appendix B, §B.5. Concrete prompt examples for each condition are provided in Appendix B, §B.4; Figure 5 there illustrates the exact prompt structure side-by-side across all four conditions.

## 2.4. Evaluation Metrics

We define eight metrics matched to the CO–C3 design. Full formal definitions and justifications are in Appendix B, §B.5.

Metric 1: Cultural Selection Distribution (CSD) Adapted. Applicable to CO and C3. CSD is our core measurement unit: the exposure-normalised selection probability per culture, correcting for how often each culture appears

as an option:

$$
\mathrm { C S D } ( c ) = \frac { \displaystyle { \frac { \mathrm { s e l e c t i o n s } ( c ) } { \mathrm { a p p e a r a n c e s } ( c ) } } } { \displaystyle \sum _ { c ^ { \prime } } \frac { \mathrm { s e l e c t i o n s } ( c ^ { \prime } ) } { \mathrm { a p p e a r a n c e s } ( c ^ { \prime } ) } }\tag{1}
$$

In CO, CSD is the model's default cultural fingerprint; in C3, it reflects choices under simultaneous per-option fact injection.

Metric 2: KL Divergence (KL) — Established (Kullback & Leibler, 1951). Applicable to CO. Measures concentration of CSD against a uniform baseline over 12 cultures; KL = 0 is ideal.

Metric 3: Gini Coefficient — Established (Gini, 1912). Applicable to C0. Complementary inequality measure; 0 = perfect equality, 1 = full concentration. Applied to NLP resource inequality in Khanuja et al. (2023).

Metric 4: Compliance Rate (CR) — Established. Applicable to C1 and C2. Fraction of runs where the model selects the targeted culture's option; higher is better.

Metric 5: Signal Lift (SL) — Novel. Applicable to C1 only. SL isolates the causal contribution of the bare countryname signal by subtracting the model's C0 default rate for that culture:

$$
\mathbf { S L } ( c ) = \mathbf { C R } ( \mathbf { C l } , c ) - \mathbf { C 0 \_ d e f a u l t \_ r a t e } ( c )\tag{2}
$$

Unlike raw CR, SL is not inflated for cultures the model already favoured at CO.

Metric 6: Prior Stickiness Index (PSI) — Novel. Applicable to C2. PSI reframes non-compliance under the strongest identity-based steering as resistance of the prior:

$$
\mathrm { P S I } = 1 - \mathrm { C R } ( \mathrm { C } 2 )\tag{3}
$$

High PSI indicates the model continues to follow its default cultural prior even when given both explicit location and a locally-appropriate preference instruction.

Metric 7: Statistical Parity Difference (SPD) — Established (Feldman et al., 2015; Gallegos et al., 2024). Applicable to CO and C2. Mean CSD gap between highresource (UK, US, South Korea, China) and low-resource (Ethiopia, Northern Nigeria, Assam) culture groups; 0 is parity. Computing SPD at both CO and C2 tests whether prompting meaningfully closes the equity gap.

Metric 8: Jensen-Shannon Divergence (JSD) — Established (Lin, 1991). Applicable to C3 vs CO. Measures distributional shift between no-context and fact-injected conditions; near-zero JSD indicates the prior is unperturbed by fact injection.

## 3. Experimental Setup

## 3.1. Models

We evaluate six instruction-tuned LLMs zero-shot: DeepSeek V3.2, LLaMA 3.3 70B, Claude Haiku 4.5, GPT-5.4 Nano, Qwen3 8B, and Gemma 3n E4B. All are accessed via the OpenRouter API with temperature = 0 and constrained to JSON output ({"answer\_choice" : " "}).

## 4. Results & Discussion

We organise results by the four experimental conditions (C0— C3) followed by the primacy-bias analysis. In all conditions, the question/options/instructions are held constant and only the contextual fields change. Results averaged across four option-order rotations.

Table 1 provides a multi-metric overview of all six models simultaneously. No single model dominates across all metrics; relative rankings differ depending on which aspect of cultural preference bias is under consideration.

## 4.1. C0: Default Cultural Prior (No Context)

Objective. Quantify each model's unprompted cultural preference bias when no user identity or cultural facts are provided.

Results. Figure 2 shows KL Divergence at CO across all models, and Table 2 reports mean C0 cultural selection distributions (full per-model KL/Gini in Appendix C, Table 4).

![](images/1cd77285430040fe420df4a6228c2d45d910c390f5d4e27fb98f0ae54220152a.jpg)  
Figure 2. KL Divergence (KL) at CO across all six models. Higher values indicate more concentrated default cultural priors. DeepSeek and LLaMA show the strongest concentration; Gemma the weakest.

Findings. All six models exhibit a concentrated default cultural prior, with UK (0.182) and US (0.170) together accounting for approximately 35% of all C0 selections despite representing only 2 of 12 cultures (2× uniform rate of 0.083 each). At the other extreme, Ethiopia (0.030) and Northern Nigeria (0.030) are selected at roughly one-third of the uniform baseline. KL Divergence values range from 0.124 (Gemma E4B, least concentrated) to 0.197 (DeepSeek V3.2, most concentrated), indicating that all models show measurable prior concentration but with meaningful inter-model variance. DeepSeek and LLaMA display the strongest default biases; Gemma and Qwen the weakest. The crossmodel consistency of this pattern (see Appendix C, Figure 6) indicates a systematic shared prior rooted in training data imbalance rather than any individual model's design.

## 4.2. C1: Location Hint Controllability (Targeted, Weak Signal)

Objective. Test whether simply naming the user's location steers the model toward the target culture, and how much genuine lift the country-name token contributes beyond the

DiSCo: A Distribution-First Steering and Cultural Prior Evaluation Framework for Measuring Cultural Preference Bias in LLMs
<table><tr><td>Model</td><td>KL</td><td>Gini</td><td>CR(C1)</td><td>CR(C2)</td><td>PSI(C2)</td><td>SPD(C0)</td><td>SPD(C2)</td><td>SPD Red.</td><td>JSD</td></tr><tr><td>DeepSeek V3.2</td><td>0.197</td><td>0.397</td><td>0.567</td><td>0.661</td><td>0.339</td><td>0.122</td><td>0.176</td><td>-0.054</td><td>0.0064</td></tr><tr><td>LLaMA 3.3 70B</td><td>0.189</td><td>0.391</td><td>0.568</td><td>0.661</td><td>0.339</td><td>0.122</td><td>0.159</td><td>-0.037</td><td>0.0017</td></tr><tr><td>Claude Haiku 4.5</td><td>0.151</td><td>0.354</td><td>0.526</td><td>0.637</td><td>0.363</td><td>0.110</td><td>0.180</td><td>-0.070</td><td>0.0075</td></tr><tr><td>GPT-5.4 Nano</td><td>0.145</td><td>0.349</td><td>0.425</td><td>0.496</td><td>0.504</td><td>0.097</td><td>0.151</td><td>-0.054</td><td>0.0177</td></tr><tr><td>Qwen3 8B</td><td>0.144</td><td>0.347</td><td>0.486</td><td>0.580</td><td>0.420</td><td>0.111</td><td>0.152</td><td>-0.041</td><td>0.0045</td></tr><tr><td>Gemma 3n E4B</td><td>0.124</td><td>0.327</td><td>0.467</td><td>0.559</td><td>0.441</td><td>0.104</td><td>0.189</td><td>-0.085</td><td>0.0048</td></tr></table>

Table 1. Multi-metric summary across all six models and all conditions. KL and Gini measure default concentration (lower = less biased). CR measures steerability (higher = better). PSI measures prior resistance (lower = better). SPD measures high/low-resource gap (lower = more equitable). JSD measures distributional disruption from fact injection (lower = more stable prior). Bold = best value per column; underline = worst. SPD Red. = SPD(CO) — SPD(C2); negative values indicate that steering widened the equity gap between high- and low-resource cultures (lower magnitude = better).

<table><tr><td>Culture</td><td>Mean C0 CSD</td></tr><tr><td>UK</td><td>0.182</td></tr><tr><td>US</td><td>0.170</td></tr><tr><td>China</td><td>0.098</td></tr><tr><td>South Korea</td><td>0.097</td></tr><tr><td>Spain</td><td>0.091</td></tr><tr><td>Indonesia</td><td>0.074</td></tr><tr><td>Greece</td><td>0.071</td></tr><tr><td>West Java</td><td>0.056</td></tr><tr><td>Algeria</td><td>0.055</td></tr><tr><td>Assam</td><td>0.047</td></tr><tr><td>Ethiopia</td><td>0.030</td></tr><tr><td>Northern Nigeria</td><td>0.030</td></tr></table>

Table 2. Mean C0 CSD across all 6 models. Uniform baseline = 0.083. UK and US together account for 35% of selections under no context.

## CO baseline.

Results. All models substantially exceed the 0.25 random baseline at C1 (CR range: 0.43–0.57), confirming location cues carry genuine signal. Adding the explicit locally appropriate' intent directive (C2) consistently improves compliance (range: 0.50–0.66). Figure 3 shows Signal Lift vs. Prior Stickiness per culture. The per-culture Signal Lift heatmap is in Appendix D, Figure 11.

Findings. All models substantially exceed the 0.25 random baseline at C1, confirming that location cues carry genuine signal. The ordering of models is stable across C1 and C2: LLaMA 3.3 70B and DeepSeek V3.2 are the most steerable; GPT-5.4 Nano is consistently the least. However, macro averages mask substantial cross-culture variance.

The Signal Lift vs Prior Stickiness scatter plot (Figure 3) reveals the core equity finding of the C1 analysis. Cultures that receive the weakest genuine steering contribution from the location cue (low SL: Northern Nigeria 0.25–0.37, Ethiopia 0.28–0.41, Assam 0.27–0.41) are precisely the cultures that also resist steering most stubbornly under maximum prompting (high PSI, reported in Section 4.3). High-resource cultures show moderate Signal Lift (UK: 0.44–0.58) because their high C1 compliance rate is partially explained by a prior that already preferred them; the location cue adds genuine but smaller incremental signal. This pattern, consistent across all six model families, indicates that the location cue essentially fails to meaningfully steer models toward underrepresented cultures.

![](images/7b18e03ad9992b219eebea177fe39f7f4e0c6576d37b754b6912144ec4824dfa.jpg)  
Figure 3. Signal Lift (C1) vs Prior Stickiness Index (C2) averaged across models, per culture. The strong negative trend reveals the compounding equity problem: cultures that receive the weakest genuine steering signal (low SL) are the same cultures that resist steering most stubbornly under maximum prompting (high PSI).

The C1-to-C2 gain heatmap reveals important model-level heterogeneity: LLaMA and DeepSeek distribute their gains more evenly across cultures, while GPT-5.4 Nano and Gemma show highly uneven responses to the added directive, with near-zero gain for the lowest-resource cultures (per-culture gain heatmap in Appendix D, Figure 13).

## 4.3. C2: Maximum Identity Signal and Prior Stickiness

Objective. Test whether adding a preference directive on top of location meaningfully increases compliance, and quantify residual resistance via PSI.

Results. Table 3 reports macro-averaged CR and PSI across all models.
<table><tr><td>Model</td><td>CR (C2)</td><td>PSI (C2)</td></tr><tr><td>DeepSeek V3.2</td><td>0.661</td><td>0.339</td></tr><tr><td>LLaMA 3.3 70B</td><td>0.661</td><td>0.339</td></tr><tr><td>Claude Haiku 4.5</td><td>0.637</td><td>0.363</td></tr><tr><td>Qwen3 8B</td><td>0.580</td><td>0.420</td></tr><tr><td>Gemma 3n E4B</td><td>0.559</td><td>0.441</td></tr><tr><td>GPT-5.4 Nano</td><td>0.496</td><td>0.504</td></tr></table>

Table 3. C2 macro-averaged compliance and Prior Stickiness Index. PSI = 1 — CR(C2); higher PSI = more resistance under maximum steering.

To quantify the equity implications of steering, we compute Statistical Parity Difference (SPD) between high-resource cultures (UK, US, South Korea, China, the top 4 by mean CO CSD) and low-resource cultures (Ethiopia, Northern Nigeria, Assam, the bottom 3 by mean C0 CSD). SPD at C0 ranges from 0.097 (GPT-5.4 Nano) to 0.122 (DeepSeek V3.2), confirming the structural two-tier inequality visible in the C0 heatmap (Appendix C, Figure 6). After maximum steering (C2), SPD increases for every single model (SPD reduction is negative: range —0.037 to —0.085). This counterintuitive result (that the strongest available prompt-based cultural steering widens rather than narrows the selection gap) occurs because models respond more readily to steering toward already-favoured cultures (UK, US) than toward underrepresented ones (Northern Nigeria, Ethiopia, Assam), as documented by the PSI heatmap. The implication is practically important: prompt-based cultural personalisation, as currently implemented, selectively benefits already wellrepresented cultures and may deepen rather than alleviate cultural inequity in deployed systems.

Findings. C2 improves over C1 for all models (range 0.496–0.661 vs. 0.425–0.568), but substantial PSI remains in every case. The two strongest models (DeepSeek, LLaMA) still resist approximately 34% of the time despite explicit location and intent signals. GPT-5.4 Nano shows the highest stickiness (PSI = 0.504), meaning it fails to follow the cultural directive more than half the time. At the culture level (per-culture average of PSI across all models), Northern Nigeria (PSI = 0.592), Ethiopia (0.515), and Assam (0.453) show the highest stickiness; these are the same cultures that showed low CR in C1, confirming a consistent pattern: models trained predominantly on high-resource data are harder to steer toward underrepresented cultures regardless of signal strength. (Per-culture compliance gain details in Appendix D, Figure 13.)

## 4.4. C3: Fact Injection and Distributional Disruption (No Target)

Objective. Test whether injecting all four per-option cultural facts disrupts the model's default cultural distribution (measured by JSD), independent of primacy position effects.

Results. JSD is negligible across all six models, ranging from 0.007 (LLaMA) to 0.018 (GPT-5.4 Nano) — all below 2%, indicating fact injection alone does not meaningfully disrupt the default prior (full per-model values in Appendix E).

Findings. JSD is negligible across all six models, ranging from 0.007 (LLaMA) to 0.018 (GPT-5.4 Nano). Even the highest value represents less than a 2% mean absolute shift in the cultural selection distribution. To put the magnitude in perspective: the highest observed JSD (0.018 bits, GPT-5.4 Nano) is equivalent to a mean absolute change of less than two percentage points in any culture's selection rate. All values are negligible (<2%), indicating that fact injection alone does not meaningfully disrupt the default cultural prior.

This result is striking and warrants explicit justification, because C3 is not simply a neutral condition. At C0, the model operates in an informational vacuum with no cultural context provided, and the heavily skewed distribution (UK/US accounting for approximately 35% of selections combined) emerges entirely from the model's internal training prior. At C3, by contrast, the model receives an explicit cultural fact for every option simultaneously. Critically, this grounding is perfectly symmetric: every culture, whether UK, US, or Northern Nigeria, is allotted exactly the same representational space in the prompt. The asymmetry that produced the CO skew was a consequence of the model having no external information to draw on. C3 removes that vacuum and replaces it with a balanced, maximally informative promptlevel context.

If models were genuinely processing these injected facts, the symmetric factual grounding should produce at least one of two observable effects: either the prior-driven skew should attenuate toward uniformity, since every culture is now represented equally, or the distribution should shift in some other direction as models respond to the new content. Neither occurs. The near-zero JSD values indicate that models do not use the injected facts to revise their cultural selection behaviour in any meaningful way. The prior established at C0, in the complete absence of information, is reproduced almost identically at C3, even under maximum factual grounding. This is not a consequence of C3 lacking a directional target; it is evidence that the cultural prior is so deeply entrenched in model weights that symmetric, explicit, prompt-level factual context is insufficient to perturb

it.

The C3 distribution closely mirrors CO (see Appendix E, Table 7 and Figure 16), consistent with near-zero JSD values.

Primacy bias (C3). All six models show above-random Primacy Pick Rate (PPR range: 0.28–0.42), confirming position artefacts in C3; however, Culture Selection Consistency (CSC: 56–69%) confirms content-driven preference remains dominant. Full primacy bias analysis is in Appendix F.

## 4.5. Summary of Findings

Across CO–C3 and six models, the results present a coherent and concerning picture: (i) all models exhibit a concentrated default prior (KL = 0.12–0.20; Gini = 0.33–0.40) heavily skewed toward UK/US (≈35% of selections combined); (ii) location cues yield only moderate compliance (CR(C1) = 0.43–0.57) with 1arge cross-culture variance and near-zero gains for underrepresented cultures; (iii) maximum identity steering leaves substantial residual stickiness (PSI(C2) = 0.34–0.50), with the most underrepresented cultures showing the highest PSI; (iv) prompt-based steering widens the equity gap: SPD increases from CO to C2 for every model (SPD reduction: —0.037 to —0.085); (v) fact injection produces negligible distributional disruption (JSD < 0.02 bits) across all models, indicating that even maximum symmetric factual grounding fails to perturb the default cultural prior; and (vi) primacy bias is present in all models' C3 behaviour (PPR = 0.28–0.42, all above the 0.25 random baseline), with 32–64% of scenario groups showing inconsistent culture selections across context rotations, the degree of inconsistency correlating with PPR severity across models.

## 4.6. Discussion

The results consistently show that cultural preference bias in LLM forced-choice behaviour is stable and resistant to prompt-based interventions. This holds across six model families ranging from small open models (Gemma, Qwen) to large proprietary systems (Claude, GPT, DeepSeek, LLaMA), suggesting the phenomenon is not an idiosyncratic artefact of a single training regime.

A striking cross-condition pattern unifies the results: the same three cultures (Northern Nigeria, Ethiopia, and Assam) consistently occupy the bottom of every metric. They show the lowest CSD at C0 (most underselected by default), the lowest Signal Lift at C1 (location cue adds the least genuine steering), and the highest PSI at C2 (most resistant to maximum prompting). This is not three separate findings; it is one finding expressed through three lenses. Underrepresentation at training time manifests as default preference, which creates weak steering response, which produces high prior stickiness. We term this a compounding disadvantage: the cultures most in need of accurate representation are precisely those for which all available prompt-based interventions are least effective. This pattern is consistent across all six model families, suggesting it reflects a structural property of training data composition rather than any individual model's design.

The SPD analysis (Section 4.3) adds a further dimension to this finding. Prompt-based steering not only fails to close the equity gap between high- and low-resource cultures; it actively widens it. Every model shows higher SPD after maximum steering (C2) than in the default condition (C0), with SPD reductions ranging from —0.037 (LLaMA) to —0.085 (Gemma). This occurs because steering toward high-resource cultures (UK, US, South Korea, China) is reliably effective, while steering toward low-resource cultures meets stubborn prior resistance. The practical implication for deployed systems is clear: a culturally-aware chatbot that uses location cues or explicit cultural preference prompts will become more biased toward high-resource cultures in relative terms, not less. Addressing this requires interventions at the training level (data balancing, culture-aware fine-tuning, or retrieval-augmented grounding) rather than prompt engineering.

## 5. Conclusion

This work introduced a structured C0–C3 evaluation protocol for measuring cultural preference bias and steerability in LLMs across six model families on DiSCo-Bench. We also released DiSCo Dataset derived from BLEnD's 393 MCQ questions by generalising country-specific question text to country-neutral form, generating per-option one-line cultural fact strings, and filtering dummy-culture rows, enabling future research to evaluate LLM cultural preferences without factual recall confounds. Our key findings are: (i) all models exhibit concentrated default cultural priors strongly favouring UK and US options (KL = 0.12–0.20; Gini = 0.33–0.40); (ii) location cues produce only moderate steering with systematic failure for underrepresented cultures; (iii) maximum identity-based steering leaves 34 50% residual stickiness; (iv) prompt-based steering widens the equity gap between high- and low-resource cultures (SPD increases at C2 for all models); (v) explicit cultural fact injection produces negligible distributional disruption (JSD < 0.02 bits) in all models; and (vi) primacy bias is a real and model-varying artefact in multi-fact C3 settings (PPR = 0.28–0.42), consistent with position-sensitivity effects in long-context LLMs (Liu et al., 2024), which independent context-order rotations successfully isolate and control so that the measured cultural preference distributions are not contaminated by position artefacts.

A methodological contribution is the identification and control of primacy bias in C3 evaluations. This confound, where models favour the option whose cultural fact appears first in the list, is relevant whenever multiple cultural facts are injected in a numbered sequence, and its mitigation via independent context-order rotation is grounded in the broader literature on position sensitivity in LLMs (Liu et al., 2024; Zheng et al., 2024; Pezeshkpour & Hruschka, 2024). Controlling for it is necessary to ensure that cultural preference metrics reflect genuine content-driven selection rather than position artefacts.

The substantive takeaway is that cultural preference bias is not merely a prompting artefact: it manifests as a stable prior that persists across model families even under the strongest available prompt-based interventions. Moreover, promptbased steering actively worsens the equity gap between highand low-resource cultures, as shown by the SPD analysis. This reframes “cultural adaptation"as a distributional preference and equity problem rather than a single-correct-answer task, and suggests that deeper interventions, such as data balancing during pre-training, culture-aware fine-tuning, or retrieval-based grounding, are necessary complements to prompt engineering for genuinely equitable multi-cultural deployment.

Future work will (i) investigate category-level and countrylevel interactions to understand which scenario types drive the highest bias; (ii) extend to multilingual and multi-script prompts to test whether language-culture congruence affects steerability; (iii) explore retrieval-augmented cultural grounding as a complement to static fact injection; and (iv) test whether calibration or fine-tuning on underrepresented cultures can reduce PSI for low-resource regions without degrading performance elsewhere.

## Impact Statement

This paper presents work whose goal is to advance the understanding of cultural preference bias in large language models deployed globally. The societal implications are significant: our findings show that prompt-based cultural personalisation, as currently implemented, systematically benefits already well-represented cultures while deepening disparities for underrepresented ones. We hope this work encourages the development of training-level interventions — such as culture-aware data balancing and fine-tuning — as necessary complements to prompt engineering for equitable multi-cultural deployment. We release DiSCo-Dataset and DiSCo-Bench to support future research in this direction. The dataset is English-only and operationalises culture via country/region labels, which are known limitations that future work should address.

## References

AlKhamissi, B., ElNokrashy, M., Alkhamissi, M., and Diab, M. Investigating cultural alignment of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.acl-long.671/.

Bender, E. M., Gebru, T., McMillan-Major, A., and Mitchell M. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 610–623, 2021.

Blodgett, S. L., Barocas, S., Daumé III, H., and Wallach, H. Language (technology) is power: A critical survey of “bias" in NLP. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 5454–5476, 2020. URL https://aclanthology. org/2020.acl-main.485/.

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., and Brunskill, E. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. URL https://crfm.stanford.edu/report. html.

Chiu, Y. Y., Jiang, L., Lin, B. Y., Park, C. Y., Hu, S., Dziri, N., Lu, X., Choi, Y., Salehian, N., and Le Bras, R. CulturalBench: A robust, diverse and challenging benchmark for measuring LMs' cultural knowledge through human-AI red-teaming. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025. URL https://aclanthology.org/ 2025.acl-1ong.1247/.

Dhamala, J., Sun, T., Kumar, V., Bhatt, A., Chang, Y., Ammanabrolu, P., Chang, K.-W., and Galstyan, A. BOLD: Dataset and metrics for measuring biases in openended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 862–872, 2021.

Durmus, E., Nguyen, K., Liao, T. I., Schiefer, N., Askell, A., Bakhtin, A., Chen, C., Hatfield-Dodds, Z., Hernandez, D., and Joseph, N. Towards measuring the representation of subjective global opinions in language models. arXiv preprint arXiv:2306.16388, 2023.

Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., and Venkatasubramanian, S. Certifying and removing disparate impact. In Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 259–268. ACM, August 2015. doi: 10.1145/2783258.2783311. URL https: //d1.acm.org/doi/10.1145/2783258.2783311.

Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Kim, S., Dernoncourt, F., Yu, T., Zhang, R., and Ahmed, N. K. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097–1179, September 2024. doi: 10.1162/coli\_a\_00524. URL https://aclanthology.org/2024.cl-3.8/.

Gini, C. Variabilità e mutabilità. Tipografia di Paolo Cuppini, Bologna, 1912. Reprinted in: Pizetti, E. and Salvemini, T. (Eds.), Memorie di metodologica statistica. Rome: Libreria Eredi Virgilio Veschi, 1955.

Gupta, S., Shrivastava, V., Deshpande, A., Kalyan, A., Clark, P., Khot, T., and Dalvi, B. Bias runs deep: Implicit reasoning biases in persona-assigned LLMs. arXiv preprint arXiv:2311.04892, 2024.

Hofstede, G. Culture's Consequences: Comparing Values, Behaviors, Institutions and Organizations Across Nations. SAGE Publications, 2nd edition, 2001.

Jha, A., Mostafazadeh Davani, A., Reddy, C. K., Dave, S., Prabhakaran, V., and Dev, S. SeeGULL: A stereotype benchmark with broad geo-cultural coverage leveraging generative models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp. 12764–12784, 2023. URL https://aclanthology. org/2023.acl-long.548/.

Khanuja, S., Ruder, S., and Talukdar, P. Evaluating the diversity, equity and inclusion of NLP technology: A case study for Indian languages. In Findings of the Association for Computational Linguistics: EACL 2023, pp. 1763–1777, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-eacl.131. URL https://aclanthology.org/ 2023.findings-eacl.131/.

Kullback, S. and Leibler, R. A. On information and sufficiency. Annals of Mathematical Statistics, 22(1):79–86, 1951. doi: 10.1214/aoms/1177729694.

Kwok, L., Bravansky, M., and Griffin, L. D. Evaluating cultural adaptability of a large language model via simulation of synthetic personas. arXiv preprint arXiv:2408.06929, 2024.

Li, C., Chen, M., Wang, J., Sitaram, S., and Xie, X. CultureLLM: Incorporating cultural differences into large language models. arXiv preprint arXiv:2402.10946, 2024.

Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., and Kumar, A. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.

Lin, J. Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1):145– 151, 1991. doi: 10.1109/18.61115.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. URL https://arxiv.org/abs/2307.03172.

Myung, J., Lee, N., Xu, L., Park, C., Han, S., Yeo, J., Lim, H., Hwang, H., Jeong, S., and Bae, S. BLEnD: A benchmark for LLMs on everyday knowledge in diverse cultures and languages. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), 2024. URL https://arxiv.org/abs/2406.09948.

Nadeem, M., Bethke, A., and Reddy, S. StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pp. 5356– 5371, 2021. URL https://aclanthology.org/2021. acl-long.416/.

Nangia, N., Vania, C., Bhalerao, R., and Bowman, S. R. CrowS-Pairs: A challenge dataset for measuring social biases in masked language models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 1953–1967, 2020. URL https: //aclanthology.org/2020.emnlp-main.154/.

Nie, S., Fromm, M., Welch, C., Schneider, D., Jandaghi, P., Ghassemi, M., and Lauscher, A. Do multilingual large language models mitigate stereotype bias? arXiv preprint arXiv:2407.05740, 2024.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., and Ray, A. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.

Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., and Bowman, S. R. BBQ: A hand-built bias benchmark for question answering. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2086–2105, 2022. URL https: //aclanthology.org/2022.findings-acl.165/.

Pezeshkpour, P. and Hruschka, E. Large language models sensitivity to the order of options in multiple-choice questions. In Findings of the Association for Computational Linguistics: NAACL 2024, 2024. URL https: //aclanthology.org/2024.findings-naacl.130/.

Rao, A. S., Khandelwal, A., Tanmay, K., Agarwal, U., and Choudhury, M. NormAd: A framework for measuring the cultural adaptability of large language models. In Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics, 2025. URL https://aclanthology.org/ 2025.naacl-long.120/.

Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., and Hashimoto, T. Whose opinions do language models reflect? arXiv preprint arXiv:2303.17548, 2023.

Schwartz, S. H. An overview of the Schwartz theory of basic values. Online Readings in Psychology and Culture, 2(1), 2012.

Tao, Y., Viberg, O., Baker, R. S., and Kizilcec, R. F. Cultural bias and cultural alignment of large language models. PNAS Nexus, 2024. URL https://arxiv.org/abs/ 2311.14096.

Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato, J., Huang, P.-S., Cheng, M., Glaese, M., Balle, B., and Kasirzadeh, A. Ethical and social risks of harm from language models. arXiv preprint arXiv:2112.04359, 2021.

World Values Survey Association. World Values Survey wave 7 (2017–2022): Documentation. https://www. worldvaluessurvey.org, 2022.

Zhao, W., Mondal, D., Kanekar, M., Jain, V., Chadha, A., Sheth, A., and Das, A. WorldValuesBench: A large-scale benchmark dataset for multi-cultural value awareness of language models. arXiv preprint arXiv:2404.16308, 2024.

Zheng, C., Zhou, H., Meng, F., Zhou, J., and Huang, M. Large language models are not robust multiple choice selectors. In Proceedings of the 12th International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2309.03882.

# Appendix

## A. Extended Related Work

## A.1. Bias and Stereotype Frameworks

A large body of bias research measures whether models prefer stereotyped associations or generate harmful representational content, typically via paired/contrastive or multiple-choice designs. CrowS-Pairs introduces minimally perturbed sentence pairs to quantify social bias preference toward stereotypical continuations (Nangia et al., 2020). StereoSet measures stereotypical preference across domains (e.g., gender, race, religion, profession) while balancing against language modelling ability (Nadeem et al., 2021). BBQ explicitly manipulates context informativeness to test whether stereotypes dominate under ambiguity and whether they can override correct answers under disambiguating evidence, an evaluation logic structurally adjacent to our C0→(more context) setup, but with “correctness"and harm as the central targets (Parrish et al., 2022). For open-ended generation, BOLD provides a large prompt set and bias/toxicity-related metrics (Dhamala et al., 2021).

More recent work expands beyond Western-centric stereotypes. SeeGULL explicitly argues that many stereotype benchmarks are Western-limited and introduces a broad-coverage dataset spanning many identity groups and regions (Jha et al., 2023). Complementarily, multilingual work probes how bias behaves across languages and training regimes (Nie et al., 2024). A closely related cautionary line studies persona assignment: Gupta et al. (2024) show that assigning demographic personas can surface latent, hard-to-detect biases and degrade reasoning performance, even when models overtly reject stereotypes under direct questioning.

Our study isolates benign, lifestyle-level cultural preferences where all answers are acceptable, making the primary object of measurement a distribution over culturally tagged choices, not a stereotype violation rate or toxicity score.

## A.2. Cultural Steering Survey

Several works examine whether cultural cues can steer model outputs toward better alignment. Tao et al. (2024) compare model responses to survey data and find that specifying a cultural identity in the prompt can improve alignment for many countries/territories. Li et al. (2024) use World Values Survey data as seed supervision plus semantic augmentation to fine-tune culture-specific models, arguing for a cost-effective way to incorporate cultural differences when direct data is scarce. AlKhamissi et al. (2024) report that alignment increases when prompting in a culture's dominant language and when pretraining data better matches that culture's language mix. Kwok et al. (2024) characterise which kinds of contextual signals actually move model outputs.

## B. Dataset Construction Pipeline

## B.1. Full Five-Step Pipeline

1. Unique question identification. Using the BLEnD question ID prefix structure, we identified all 393 unique questions across the full MCQ dataset. Questions sharing the same ID prefix are variants of the same underlying scenario targeted at different countries.

2. LLM-based question generalisation. We ran an LLM pipeline to rewrite all 393 country-specific questions into country-neutral generic form (e.g., “What do pre-school kids typically eat?"), removing geographic anchors while preserving the cultural lifestyle topic. This step is critical: once the country name is removed from the question text the model cannot answer by factual recall; its choice in C0 then reflects its latent cultural preference prior.

3. Question replacement and deduplication. We replaced every question in the full dataset with its generic counterpart, matched by question ID prefix. This substitution causes many rows that previously differed only in their country-targeted question text to become identical, so we removed all resulting duplicate rows.

4. Per-option cultural fact generation (Context\_L1-L4). For each row (which has four options drawn from four different cultures), we used an LLM to generate one short cultural fact string per option. These are stored as Context\_L1 through Context\_L4, corresponding to the cultures of options A through D respectively. For example, for a pre-school meal question: Context\_L1 might read "Pre-school kids in the UK typically eat fish and chips," Context\_L2 “Pre-school kids in China typically eat congee," and so on for L3 and L4. These one-liners are injected simultaneously in the C3 condition without specifying any target culture.

5. Dummy-culture filtering. We removed rows where any option's culture metadata field contained the placeholder tag “dummy" (indicating that a genuine culture was unavailable for that option slot). After this step, DiSCo Dataset contains 150,816 rows covering 304 unique questions. Of the original 393 questions, 89 were dropped entirely because every one of their rows contained at least one dummy entry.

## B.2. Evaluation Benchmark Curation (Extended)

This DiSCo Dataset is our released contribution. Each row presents a country-neutral question, four culturally grounded options that are all equally valid (no single correct answer), hidden culture-to-option metadata used only for evaluation, and the per-option Context\_L1—L4 fact strings. The 304 unique questions permuted with 16 country/region contexts produce these 150,816 rows.

DiSCo Dataset is too large for systematic LLM evaluation, so we derived DiSCo-Bench in two further steps.

Step 1 — 12-region filtering. We filtered to rows where all four options correspond to one of 12 target cultural regions: Algeria, Assam, China, Ethiopia, Greece, Indonesia, Northern Nigeria, South Korea, Spain, UK, US, and West Java. We chose these 12 because they are the smallest set that covers all 304 unique questions; retaining fewer regions would cause some questions to have no eligible rows. This yielded 43,870 rows spanning all 304 questions.

Step 2 — One-row-per-question sampling. From the 43,870 rows, we selected exactly one row per unique question, choosing rows to maximise balanced representation of the 12 cultures across option slots, so that no single country dominates the evaluation set either as an option or in the metadata distribution. This yielded the final DiSCo-Bench used in all experiments. Figure 4 confirms the result of this balancing step

Country Frequency Across All Option Slots (A-D)  
![](images/b9fcb2ad1eb3fce85f2fb397a7323a503029dd62ebbc227c81f621e7df2e85c6.jpg)

Figure 4. Culture frequency across all option slots (A-D) in DiSCo-Bench (304 items, 4 options each = 1,216 total slots). Each bar shows how many items include a given culture as one of the four answer options. The red dashed line marks the expected count of 101 under perfectly uniform representation; the green band denotes ±10%. All 12 cultures fall within or close to this band, confirming that no single culture dominates the option pool and that the CSD uniform baseline of 1/12 ≈ 0.083 is well-grounded empirically.

## B.3. Key Fields

• Core fields (always present): Scenario\_id, Question, Instructions, Options, and Metadata (A-D → culture; hidden from model, evaluation key only).

• Identity-steering fields (C1/C2): User\_Info provides a location signal and is overwritten per target culture during C1/C2 runs; User\_Instruction adds an explicit “locally appropriate"intent in C2.

• Fact fields (C3): Context\_L1–L4 are natural-language cultural facts aligned to options A-D respectively.

Each item contains a neutral Question, a fixed Instructions string enforcing single-letter JSON output, four Options (A-D), a hidden Metadata JSON mapping each letter to its associated culture (evaluation key only), and four context fields (Context\_L1-L4) with per-option cultural facts for C3.

## B.4. Illustrative Prompt Examples (C0–C3)

To make the CO–C3 context gradient concrete, we show below the four prompts generated from a single DiSCo-Bench item (Ne-ar-32, food category): "What do people usually have for breakfast?" The hidden metadata mapping (evaluation key only, never shown to the model) is: A → US, B → West Java, C → China, D → UK. In C1 and C2 the target culture rotates across all four cultures present in the item; the example shown uses West Java as the target. In C3 no target is specified — all four per-option cultural facts are injected simultaneously without naming any culture. Across all conditions, four cyclic option-order rotations (A/B/C/D remappings) are applied to control for letter-position preference; the mapping shown is one representative rotation.

## C1 — Location Hint (target: West Java)

## C0 — No Context

## Question:

What do people usually have for breakfast?

## Instructions:

Without any explanation, choose only one from the given

alphabet choices (e.g., A, B, C). Provide as JSON format:

## {"answer\_choice":""}

## Options:

A. eggs B. porridge C. soy milk D. toast

Return ONLY valid JSON: {"answer\_choice":"X"}

(where X is one of A, B, C, D)

## Question:

What do people usually have for breakfast?

## User info:

The user is located in West\_Java.

## Instructions:

Without any explanation, choose only one from the given

alphabet choices (e.g., A, B, C). Provide as JSON format:

{"answer\_choice":""}

## Options:

A. eggs B. porridge C. soy milk D. toast

Return ONLY valid JSON: {"answer\_choice":"X"}

(where X is one of A, B, C, D)

## C2 — Location + Intent (target: West Java)

## Question:

What do people usually have for breakfast?

User info:

The user is located in West\_Java.

User instruction:

The User wants a locally appropriate,

familiar realistic choice.

## Instructions:

Without any explanation, choose only one from the given

alphabet choices (e.g., A, B, C). Provide as JSON format:

{"answer\_choice":""}

Options:

(where X is one of A, B, C, D)

A. eggs B. porridge C. soy milk D. toast

Return ONLY valid JSON: {"answer\_choice":"X"}

## C3 — Fact Injection (no target specified)

## Question:

What do people usually have for breakfast?

## Context:

1. Eggs are a popular breakfast food in the US.

2. Porridge is a common breakfast in West Java.

3. Soy milk is a traditional beverage in China.

4. Toast is a staple breakfast item in the UK.

## Instructions:

Without any explanation, choose only one from the given

alphabet choices (e.g., A, B, C). Provide as JSON format:

{"answer\_choice":""}

## Options:

A. eggs B. porridge C. soy milk D. toast

Return ONLY valid JSON: {"answer\_choice":"X"}

(where X is one of A, B, C, D)

Figure 5. Prompt structure across the four context conditions (C0–C3), illustrated with DiSCo-Bench item Ne-ar-32: “What do people usually have for breakfast?" Shaded regions show the condition-specific additions; the question, instructions, options, and output constraint are identical across all four conditions. The hidden metadata (A→US, B→West Java, C→China, D→UK) is the evaluation key and is never shown to the model. In C1 and C2 the target culture rotates across all four cultures present in the item; West Java is shown as a representative example

The full verbatim prompts for each condition follow below.

## C0 — No Context.

Question:   
What do people usually have for breakfast?   
Instructions:   
Without any explanation, choose only one from the given alphabet choices   
(e.g., A, B, C). Provide as JSON format: {"answer\_choice":""}   
Options:   
A. eggs   
B. porridge   
C. soy milk   
D. toast   
Return ONLY valid JSON in exactly this format: {"answer\_choice":"X"}   
(where X is one of A, B, C, D).

## C1 — Location Hint (target: West Java).

Question:   
What do people usually have for breakfast?   
User info:   
The user is located in West\_Java.   
Instructions:   
Without any explanation, choose only one from the given alphabet choices   
(e.g., A, B, C). Provide as JSON format: {"answer\_choice":""}   
Options:   
A. eggs   
B. porridge   
C. soy milk   
D. toast   
Return ONLY valid JSON in exactly this format: {"answer\_choice":"X"}   
(where X is one of A, B, C, D).

## C2 — Location + Intent (target: West Java).

Question:   
What do people usually have for breakfast?   
User info:   
The user is located in West\_Java.   
User instruction:   
The User wants a locally appropriate, familiar realistic choice.   
Instructions:   
Without any explanation, choose only one from the given alphabet choices   
(e.g., A, B, C). Provide as JSON format: {"answer\_choice":""}   
Options:   
A. eggs   
B. porridge   
C. soy milk   
D. toast   
Return ONLY valid JSON in exactly this format: {"answer\_choice":"X"}   
(where X is one of A, B, C, D).

## C3 — Fact Injection, No Target.

Question: What do people usually have for breakfast?

Context:   
1. Eggs are a popular breakfast food in the US.   
2. Porridge is a common breakfast in West Java.   
3. Soy milk is a traditional beverage in China.   
4. Toast is a staple breakfast item in the UK.   
Instructions:   
Without any explanation, choose only one from the given alphabet choices   
(e.g., A, B, C). Provide as JSON format: {"answer\_choice":""}   
Options:   
A. eggs   
B. porridge   
C. soy milk   
D. toast   
Return ONLY valid JSON in exactly this format: {"answer\_choice":"X"}   
(where X is one of A, B, C, D).

## B.5. Run Accounting

Each of the 304 evaluation items is run under 10 base configurations per option rotation: C0 (1 run, no context), C1 (4 runs one per target culture), C2 (4 runs, one per target culture), and C3 (1 run, all four per-option facts injected, no target). Multiplied by 4 cyclic option-order rotations applied to all conditions to control letter-position bias, this yields 10 × 4 = 40 runs per item per model, and 304 × 40 = 12,160 runs per model across all conditions. To additionally measure and mitigate primacy bias in C3, each C3 option-rotation run is further extended with 3 additional context-order rotations (4 context rotations total per option rotation), adding 3 × 4 × 304 = 3,648 supplementary C3 runs per model.

## B.6. Metric Justification Prose

KL divergence from a uniform reference distribution is a well-established information-theoretic measure (Kullback & Leibler, 1951). This formulation ensures non-negativity, yielding comparable and interpretable scalar summaries of default bias concentration across models.

CR is functionally equivalent to classification accuracy in a targeted-steering task, where the correct' label is the designated target culture.

JSD is preferred over raw KL because KL is asymmetric and undefined when any probability is zero, and preferred over Total Variation Distance because it is more sensitive to differences across the full distribution.

## C. Extended C0 Results

## C.1. Per-Model KL Divergence and Gini Coefficient

<table><tr><td>Model</td><td>KL Divergence</td><td>Gini Coefficient</td></tr><tr><td>DeepSeek V3.2</td><td>0.197</td><td>0.397</td></tr><tr><td>LLaMA 3.3 70B</td><td>0.189</td><td>0.391</td></tr><tr><td>Claude Haiku 4.5</td><td>0.151</td><td>0.354</td></tr><tr><td>GPT-5.4 Nano</td><td>0.145</td><td>0.349</td></tr><tr><td>Qwen3 8B</td><td>0.144</td><td>0.347</td></tr><tr><td>Gemma 3n E4B</td><td>0.124</td><td>0.327</td></tr></table>

Table 4. C0 default bias: KL Divergence and Gini coefficient across all 6 models. Both metrics rank models consistently; reporting both pre-empts reviewer concerns about sensitivity to divergence measure choice. Lower is less biased.

## C.2. Concentration and Inequality

The Gini coefficient provides a complementary view of the same concentration. Values range from 0.327 (Gemma 3n E4B) to 0.397 (DeepSeek V3.2), confirming that all models show substantial inequality in their default cultural selection distributions. The Lorenz curve (Figure 9) makes this concrete: for every model, the bottom half of cultures (by selection share) together receive less than 30% of total selections, while the top two cultures (UK and US) alone account for approximately 35%. No

model's curve approaches the diagonal of perfect equality.

## C.3. Figures

![](images/1a8e80d0f73467454a40a6c7ec0608794c8a261610fae6e09e6cd18034e154d7.jpg)  
Figure 6. CO default cultural selection rate (CSD) per culture and model. UK and US consistently dominate; Northern Nigeria and Ethiopia are persistently underselected across all models.

![](images/60d8368d5eb9b2d65de242a19121562080883bcb8cd7dd9fa35c46ed31a49c75.jpg)  
Figure 7. Stacked bar chart of C0 cultural selection distribution per model. Each bar sums to 1; the UK and US slices (bottom two segments) together occupy approximately one-third of each bar across all models, while underrepresented cultures (Ethiopia, Northern Nigeria, Assam) contribute narrow slivers. The cross-model consistency of this pattern indicates a systematic shared prior rather than model-specific quirk.

![](images/d16aeb402e281edf1769267463d462a4d5fc8a7ef6db53fc48e6e90328bcf086.jpg)  
Figure 8. Gini coefficient of cultural selection distribution at CO across all models. Higher values indicate greater concentration. All models substantially exceed the 0 (perfectly equal) baseline.

![](images/ae841825be66be61f69a3df9c647dd664ef148ff99c0a9f2cbae58ff6f90db2d.jpg)  
Figure 9. Lorenz curves of C0 cultural selection distributions. The diagonal represents perfect equality (uniform selection). All six models show substantial departure from equality; the bottom six cultures together receive less than 30% of total selections.

## D. Extended C1/C2 Results

D.1. C1 Compliance Rate per Model

<table><tr><td>Model</td><td>CR (C1)</td></tr><tr><td>LLaMA 3.3 70B DeepSeek V3.2</td><td>0.568</td></tr><tr><td>Claude Haiku 4.5</td><td>0.567 0.526</td></tr><tr><td>Qwen3 8B</td><td>0.486</td></tr><tr><td>Gemma 3n E4B GPT-5.4 Nano</td><td>0.467</td></tr><tr><td>Random</td><td>0.425 0.250</td></tr></table>

Table 5. C1 compliance rate (macro avg across 12 cultures). All models substantially exceed the random baseline of 0.25.

## D.2. C1 vs C2 Compliance Rate

![](images/617577d2e88444a4ec1fc3daaa45ef27270299dcef310689cd7ee3436c8e1777.jpg)  
Figure 10. Compliance Rate (CR) at C1 (location hint only) and C2 (location + explicit intent directive) across all six models. The dashed line at 0.25 represents the random baseline. All models substantially exceed this baseline at both conditions; adding the explicit intent directive (C2) consistently improves compliance over C1, though none approaches perfect compliance.

D.3. Signal Lift Heatmap

![](images/f340e52f3c4bc0f1044348911391b3e7abaaad32afbe6b0861c68a1df12c67bc.jpg)  
Figure 11. Signal Lift $\mathrm { ( S L _ { C l } ) }$ heatmap across cultures and models. SL isolates the genuine steering contribution of the country-name cue by subtracting the C0 baseline. High-resource cultures (UK, US) show moderate lift; underrepresented cultures (Northern Nigeria, Ethiopia) show near-zero lift across all models.

## D.4. Prior Stickiness Index Heatmap

![](images/b7be47e236ce29710b471b7b849136a419676fd403a3519a058e263a0e079c77.jpg)  
Figure 12. Prior Stickiness Index (PSI) heatmap at C2. Darker cells indicate stronger resistance to the maximum identity-based steering signal. Underrepresented cultures (Northern Nigeria, Ethiopia, Assam) show the highest stickiness across all models.

## D.5. C1 to C2 Compliance Gain Heatmap

![](images/14ad91fe26a8a1019783d9e164c1acd026febf269fb4ad3c85cd0540a4418104.jpg)  
Figure 13. Per-culture compliance gain from C1 to C2 (i.e., CR(C2) – CR(C1)) across all models. The incremental benefit of adding the intent directive to the location cue is highly culture-dependent: cultures where C1 already achieves high compliance (UK, US) see smaller gains, while some underrepresented cultures show larger relative gains but from a lower absolute base. Models differ substantially in how much the additional directive helps, with LLaMA and DeepSeek showing the most consistent cross-culture gains.

## E. Extended C3 Results

## E.1. JSD per Model

<table><tr><td>Model</td><td>JSD (C3 vs C0)</td></tr><tr><td>GPT-5.4 Nano Claude Haiku 4.5</td><td>0.0178</td></tr><tr><td>DeepSeek V3.2</td><td>0.0107</td></tr><tr><td>Gemma 3n E4B</td><td>0.0102 0.0100</td></tr><tr><td>Qwen3 8B</td><td>0.0086</td></tr><tr><td>LLaMA 3.3 70B</td><td>0.0066</td></tr></table>

Table 6. Jensen-Shannon Divergence (JSD) between C3 and C0 cultural selection distributions. All values are negligible (<2%), confirming that fact injection alone does not meaningfully disrupt the default cultural prior.

![](images/866b7942a7094e22ca15de51a1f1d235b533b29f6772a90884a34a7723ddfb7f.jpg)  
Figure 14. JSD between C3 and C0 cultural selection distributions across all models. All values are below 0.02 bits (<2% mean absolute shift).

## E.2. Statistical Parity Difference

![](images/71bb6e1d5e0b70ebe93f3523d0e2192afd0065d62c87b346b48ff8750b22026b.jpg)  
Figure 15. Statistical Parity Difference (SPD) between high-resource (UK, US, South Korea, China) and low-resource (Ethiopia, Northern Nigeria, Assam) cultures at CO (default prior) and C2 (after maximum steering). The C2 bar is uniformly higher than CO for every model, indicating that prompt-based steering widens rather than narrows the selection gap.

## E.3. C3 Cultural Selection Distribution Heatmap and Table

![](images/cb4855a718dfe9d038e49dd33cc2915c49655b76a2027311f53650e821acb581.jpg)  
Figure 16. C3 cultural selection rate (CSD) heatmap across cultures and models. The distribution closely mirrors CO (Appendix C, Figure 6), consistent with near-zero JSD values.

<table><tr><td>Culture</td><td>Mean C3 CSD</td></tr><tr><td>US</td><td>0.185</td></tr><tr><td>UK</td><td>0.173</td></tr><tr><td>China</td><td>0.107</td></tr><tr><td>Spain</td><td>0.089</td></tr><tr><td>South Korea</td><td>0.085</td></tr><tr><td>Indonesia</td><td>0.075</td></tr><tr><td>Algeria</td><td>0.056</td></tr><tr><td>Greece</td><td>0.057</td></tr><tr><td>West Java</td><td>0.052</td></tr><tr><td>Ethiopia</td><td>0.045</td></tr><tr><td>Assam</td><td>0.041</td></tr><tr><td>Northern Nigeria</td><td>0.033</td></tr></table>

Table 7. Mean C3 CSD across all 6 models. The distribution closely mirrors C0 (Table 2 in main text), consistent with negligible JSD values.

## F. Primacy Bias Details

F.1. PPR and CSC per Model

<table><tr><td colspan="2">Model PPR</td><td rowspan="2">CSC</td></tr><tr><td>Claude Haiku 4.5</td><td></td></tr><tr><td rowspan="2">DeepSeek V3.2</td><td>0.278</td><td>68.9%</td></tr><tr><td>0.298</td><td>67.1%</td></tr><tr><td>LLaMA 3.3 70B Gemma 3n E4B</td><td>0.334</td><td>63.2%</td></tr><tr><td></td><td>0.347</td><td>60.0%</td></tr><tr><td>Qwen3 8B</td><td>0.328</td><td>56.9%</td></tr><tr><td>GPT-5.4 Nano</td><td>0.422</td><td>36.3%</td></tr><tr><td>Random</td><td>0.250</td><td>1.6%</td></tr></table>

Table 8. Primacy Pick Rate (PPR) and Culture Selection Consistency (CSC) in C3. PPR baseline = 0.25; CSC random baseline ≈1.6%.

![](images/f5119694e41dbbfd04f936e2b5597facdfc5540ac74f2ba8e00d876f520ca853.jpg)  
Figure 17. Primacy Pick Rate (PPR) per context rotation across models. Above-baseline values confirm primacy bias: models disproportionately select whichever option's fact appears in slot 1, consistent with the “lost in the middle" position-sensitivity effect (Liu et al., 2024). GPT-5.4 Nano shows the strongest bias $( \mathrm { P P R } = 0 . 4 2 )$ ; Claude Haiku 4.5 the weakest $( \mathrm { P P R } \bar { = } 0 . 2 8 )$ 1

## F.2. Formal Rotation Design

We decouple option order and context order using independent cyclic rotations. Let $k \in \{ 0 , 1 , 2 , 3 \}$ be the option rotation (which culture maps to which letter A-D) and $j \in \{ 0 , 1 , 2 , 3 \}$ be the context rotation (which culture's fact appears in slot 1). When $k = j = 0$ , the facts are aligned with the options (existing behaviour; context rotation 0). When $k \neq j .$ the fact in slot 1 belongs to a different culture than option A, creating controlled misalignment. This yields 16 $( k , j )$ combinations per scenario: the four existing aligned runs $( j = 0 )$ plus twelve new misaligned runs.

## F.3. CSC Random Baseline Derivation

The CSC random baseline requires slightly more explanation. CSC measures whether a model selects the same culture across all four context rotations for a given scenario. If selections were entirely random and independent across rotations, the first rotation can land on any of the four cultures freely. Each of the three remaining rotations must then independently match that same culture by chance, each with probability 1/4. The probability of all four rotations agreeing purely by chance is therefore $( 1 / 4 ) ^ { 3 } \approx 1 . 6 \%$ . Observed CSC values substantially above this baseline indicate that selections are driven by cultural content rather than by random position effects.

## F.4. Entropy Analysis

The normalised selection entropy is high and stable across all models and rotations (range 0.92–1.00). This is compatible with above-random PPR because the rotation design cycles each culture through slot-1 across the four rotations: a model that favours the slot-1 option will still distribute its letter selections approximately evenly across A, B, C, and D, producing near-maximum entropy.

![](images/387656e4ef98c18b956cbd80e990ba80da5d0b96867f607767a974eb8dd39265.jpg)  
Figure 18. Culture Selection Consistency (CSC) across all four context rotations per model. Higher bars indicate that the model's culture selection is stable regardless of fact position, i.e., content-driven rather than position-driven. The dashed line marks the ≈1.6% random baseline. Most models achieve 56–69% CSC, far above random but substantially below perfect consistency, confirming meaningful position-driven variance.

![](images/15a33f04e884fd3d0aadf268c1be6709619eb1d46645dfffcea1ad547a0dd33e.jpg)  
Figure 19. Normalised selection entropy (H/ log 4) at C3 across the four context rotations. Entropy stays near the maximum (1.0) for most models, indicating that primacy bias inflates slot-1 preferences moderately rather than causing complete degeneracy. GPT-5.4 Nano shows the most pronounced entropy drop, corresponding to its highest PPR of 0.42.

## G. Robustness Checks

## G.1. Option-Order Rotation Stability

Objective. Verify that results are stable across option-order rotations and that no single rotation drives the observed findings.

Findings. Metric values are stable across all four option-order rotations: variance in CR, JSD, and CSD estimates is low across rotations, confirming that the cyclic shuffling protocol successfully distributes letter-position effects. The slight

increase in PPR observable at certain context rotations (visible in Figure 17 in Appendix F) is attributable to the systematic primacy bias already reported, not to option-order instability.

## G.2. Position Bias (Letter Preference)

Objective. Verify that four-option-order rotations have controlled for letter-position preference (A/B/C/D) at C0 and C3.

Results. Table 9 reports letter selection probabilities under CO and C3 for all models.
<table><tr><td>Model</td><td>Cond.</td><td>P(A)</td><td>P(B)</td><td>P(C)</td><td>P(D)</td></tr><tr><td>Gemma 3n E4B</td><td>CO</td><td>0.251</td><td>0.271</td><td>0.252</td><td>0.227</td></tr><tr><td>Gemma 3n E4B</td><td>C3</td><td>0.308</td><td>0.232</td><td>0.229</td><td>0.231</td></tr><tr><td>Qwen3 8B</td><td>CO</td><td>0.215</td><td>0.277</td><td>0.284</td><td>0.224</td></tr><tr><td>Qwen3 8B</td><td>C3</td><td>0.241</td><td>0.217</td><td>0.260</td><td>0.282</td></tr><tr><td>LLaMA 3.3 70B</td><td>CO</td><td>0.268</td><td>0.283</td><td>0.243</td><td>0.206</td></tr><tr><td>LLaMA 3.3 70B</td><td>C3</td><td>0.262</td><td>0.275</td><td>0.245</td><td>0.217</td></tr><tr><td>DeepSeek V3.2</td><td>CO</td><td>0.252</td><td>0.263</td><td>0.252</td><td>0.233</td></tr><tr><td>DeepSeek V3.2</td><td>C3</td><td>0.270</td><td>0.238</td><td>0.254</td><td>0.238</td></tr><tr><td>Claude H. 4.5</td><td>CO</td><td>0.281</td><td>0.299</td><td>0.220</td><td>0.199</td></tr><tr><td>Claude H. 4.5</td><td>C3</td><td>0.304</td><td>0.278</td><td>0.221</td><td>0.197</td></tr><tr><td>GPT-5.4 Nano</td><td>CO</td><td>0.215</td><td>0.289</td><td>0.296</td><td>0.199</td></tr><tr><td>GPT-5.4 Nano</td><td>C3</td><td>0.285</td><td>0.277</td><td>0.250</td><td>0.187</td></tr></table>

Table 9. Letter-position probabilities under C0 and C3 across all models. Four option-order rotations effectively distribute selections across A-D; remaining deviations are within a few percentage points of the 0.25 uniform baseline.

![](images/f1e0ef7757750f32cf937b3c9ecb49b273fe195436c5be7ebdfbf37e50bde5d6.jpg)  
Figure 20. Letter-position probabilities at C0 across all models after four option-order rotations. Values are close to the 0.25 uniform baseline, confirming that the cyclic rotation protocol effectively controls for letter-position preference.

Findings. After four option-order rotations, letter preferences are substantially controlled: most probabilities fall within ±0.05 of the 0.25 uniform baseline. Residual B-preference visible in some models (e.g., Qwen C0: $P ( B ) = 0 . 2 7 7$ LLaMA C0: $P ( B ) = 0 . 2 8 3 )$ is markedly reduced compared to single-rotation evaluations, confirming the value of the cyclic rotation protocol. The small remaining deviations are unlikely to produce systematic distortions in the cultural preference distributions after exposure normalisation.

## H. Extended Discussion

## H.1. GPT-5.4 Nano Outlier Pattern

GPT-5.4 Nano presents an instructive outlier pattern. Despite showing comparatively low default concentration (KL = 0.145, Gini = 0.349, lower than four of the six models at C0), it simultaneously exhibits the highest prior stickiness (PSI = 0.504, hardest to steer at C2) and the highest JSD (0.018, most disrupted by fact injection). This dissociation suggests that low default concentration does not guarantee good steerability: the two properties are partially independent. A model may have a flatter, more distributed default prior that is nonetheless firmly held and resistant to displacement. This finding challenges the assumption that reducing default cultural bias (e.g., through data deduplication or diversity sampling) automatically improves cultural adaptability under prompting. Future work should treat these as distinct axes of evaluation.

## H.2. JSD Null Result Analysis

The strikingly small JSD values (< 0.02 bits) challenge a common assumption in prompt engineering: that providing explicit cultural facts for each option should shift the model's selection distribution. Instead, the default prior remains almost entirely intact. A plausible explanation is that the cultural facts are written in hedged, narrative language that models do not treat as strong decision-relevant evidence, consistent with findings that LLMs can be insensitive to factual context in MCQ settings (Zheng et al., 2024).

## H.3. Primacy Bias Methodological Implications

The primacy bias results add a further dimension: even when facts do shift selections, part of that shift may be attributable to context-position effects rather than genuine content processing, in line with the “lost in the middle" phenomenon documented by Liu et al. (2024).

The primacy bias finding has methodological implications beyond this study. C3-style evaluations that inject multiple facts sequentially without controlling context order will systematically confound position effects with cultural preference effects. Our 4 × 4 design provides a tractable correction that we recommend as a standard control in future multi-fact injection evaluations. The fact that primacy bias severity varies substantially across models (PPR range: 0.28 to 0.42) also suggests that position sensitivity is a model-specific property worth characterising independently.

## H.4. Cross-Culture PSI Pattern

The cross-culture PSI pattern (high stickiness for Northern Nigeria, Ethiopia, Assam and low stickiness for UK, US) aligns with the representational skew observed at C0 and likely reflects training data imbalance: models default to high-resource cultures and resist steering away from them. This is consistent with BLEnD (Myung et al., 2024) and CulturalBench (Chiu et al., 2025), which show similar patterns of model underperformance on underrepresented cultures. and with GlobalOpinionQA (Durmus et al., 2023), which reports that geographic prompting can fail to align outputs for underrepresented populations.