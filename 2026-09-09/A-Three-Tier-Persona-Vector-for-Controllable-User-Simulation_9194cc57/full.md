# A Three-Tier Persona Vector for Controllable User Simulation in Agentic Evaluation

Rahul Khedar<sup>∗</sup>, Eshita<sup>∗</sup>, Sneha Teja Sree Reddy Thondapu<sup>∗</sup>, Mayank Malhotra<sup>∗</sup> Arup Kumar Das<sup>∗</sup>, Jitesh Chandra Mishra<sup>∗</sup>, Arun Menon<sup>†</sup>, Avinash Karn, Mouli V

PayPal AI

## Abstract

Evaluating tool-augmented LLM agents requires diverse, realistic user inputs yet most evaluation frameworks use flat role descriptions (“you are an angry customer”) that produce near-identical conversations regardless of the underlying scenario. In this paper, we propose a threetier persona vector with 23 operationalized dimensions: 6 categorical demographics (jurisdiction, age, channel, device, language proficiency, time availability), 12 continuous behavioral traits (patience, assertiveness, digital literacy, etc.) sampled with Gaussian noise around curated profile base vectors, and 5 continuous emotional states (frustration, anxiety, trust, confidence, stress) that shift in response to scenario context. Orthogonal to the persona, a 4-level query-complexity overlay controls utterance phrasing from direct to deliberately vague. We evaluate the persona model inside a synthetic data generation pipeline across 64,698 multi-turn conversations spanning 8 named profiles and 3 production corpora. Key findings: (i) a 15.8 percentage-point spread in agent goal-achievement across personas confirms trait vectors produce measurably different user behavior; (ii) the same persona behaves differently across scenarios due to scenario-reactive emotional state shifts, validating the scenario-reactive design; (iii) domain-specific projects show persona sensitivity on booking-flow compliance (∼15–20 percentage points gap between tier-aware and pressure-test personas), demonstrating the model faithfully reproduces real-world difficulty distributions; (iv) seven rule-described trait correlations produce auditable co-occurrence patterns without requiring learned covari ance matrices. The persona model is fully specified for reproduction.

## 1 Introduction

The rise of LLM-based agents that interact with external tools like APIs, databases, CRM systems has created a parallel need for realistic user simulators. Fine-tuning an agent on synthetic data is only as good as the diversity of user inputs that data contains. If every simulated user opens with “I need to dispute a charge,” the agent learns a narrow distribution that fails on real traffic where users say “something weird happened with that charge” or “help pls.”

Existing user simulation approaches fall into three categories, each with a gap:

(1) Flat role descriptions. “You are a frustrated customer” is the dominant pattern in evaluation frameworks like τ-bench [2] and MT-Bench [3]. These produce qualitatively similar conversations because a one-sentence description does not constrain vocabulary, assertiveness, domain knowledge, or emotional trajectory.

(2) Scripted user bots. Hard-coded turn sequences with template slots. These are reproducible but cannot generalize to new scenarios without re-authoring.

(3) Unconstrained LLM sampling. Prompting an LLM to “act like a user” with high temperature. This produces variation, but uncontrolled variation: the same persona profile can produce radically different behavior across samples, making it impossible to attribute outcome differences to persona vs. sampling noise.

The persona model described in this paper was first deployed as a component of STATEGEN [1], a synthetic data generation platform for tool-augmented LLM agents. That work introduced the three-tier persona vector as one of five architectural contributions; space constraints precluded a full formal treatment. This paper provides the complete specification, noise calibration analysis, ablation studies, and behavioral validation.

We propose a structured alternative: a three-tier persona vector $ { \mathbf { p } } \in \mathbb { R } ^ { 2 3 }$ that operationalizes user diversity through explicit, measurable dimensions. The key design

principles are:

(1) Tiered structure. Demographics are categorical (who the user is), behavioral traits are continuous with controlled noise (how the user acts), emotional states are continuous and scenario-reactive (how the user feels right now).

(2) Bucketing for prompt efficiency. Continuous values are discretized into {low, medium, high} before injection into the LLM prompt, keeping token counts stable while preserving downstream measurability on the raw values.

(3) Auditable correlations. Trait co-occurrence is profile-encoded and rule-described, not learned from a covariance matrix, making it inspectable by non-ML practitioners.

(4) Orthogonal complexity overlay. How the user phrases their query (simple, medium, complex, vague) is independent of who the user is, enabling factorial experimental designs.

Contributions. (i) We formalize a 23-dimensional persona vector with three tiers and provide the complete specification (distributions, noise parameters, bucketing rules, correlation rules, scenario deltas). (ii) We demonstrate a 15.8 percentage points spread in goalachievement across 8 personas on 49K+ mixed corpus samples. (iii) We show scenario-reactive emotional states make the same persona produce qualitatively different conversations across dispute vs. checkout scenarios. (iv) We provide a query-complexity overlay with a vague tier that forces multi-turn clarification, matching real user traffic distributions. (v) We evaluate on a productionscale corpus (64K samples, 3 production corpora) and show persona sensitivity on domain-specific evaluation axes.

## 2 Related Work

User simulation for dialogue. User simulators have a long history in task-oriented dialogue [4]. Agenda-based simulators [5] maintain a user goal and update it turnby-turn, but encode behavior through hand-written rules rather than learned or parameterized traits. Neural user simulators [6] learn to generate responses end-to-end but offer no explicit control over behavioral dimensions.

Persona-conditioned generation. PersonaChat [7] introduced persona-conditioned response generation via 5-sentence persona descriptions. Subsequent work explored persona consistency [8] and persona grounding [9]. These operate at the response level: given a persona description, generate a single response. Our work operates at the session level: given a persona vector, generate an entire multi-turn interaction with measurable behavioral variation.

LLM-based evaluation with user simulation. τ- bench [2] evaluates tool-agent-user interaction using hand-crafted user specifications with initial states. Agent-Bench [10] provides evaluation trajectories but does not parameterize user behavior. Matrix [11] generates multiturn data with social personas but does not decompose persona into continuous, measurable trait dimensions. STATEGEN [1] is a synthetic data generation platform that incorporates the persona model presented here as one of its core components; the present paper provides the full formal treatment of that component.

Controllable text generation. CTRL [12] and GeDi [13] control generation via conditioning codes. Promptbased control via system prompts is the dominant paradigm for LLM persona simulation. Our contribution is not a new control mechanism but a structured specification that makes persona variation measurable and attributable.

Positioning. The gap we address: existing persona models for user simulation are either (a) flat descriptions with no measurable dimensions, (b) learned embeddings with no interpretability, or (c) scripted behaviors with no generalization. We propose a structured middle ground: continuous, measurable, interpretable, and generalizable across scenarios.

## 3 The Three-Tier Persona Model

The persona vector p has 23 dimensions organized into three tiers, plus an orthogonal query-complexity overlay. Figure 1 illustrates the architecture.

## 3.1 Tier 1: Demographics (Categorical)

Six attributes: jurisdiction, age bracket, channel, device type, language proficiency, and time availability are each drawn independently from a categorical distribution (Table 1):

$$
\begin{array} { r } { a _ { k } \sim \mathrm { C a t e g o r i c a l } ( { \bf P } _ { k } ) , \quad P _ { k , j } = p _ { k , j } / \sum _ { m } p _ { k , m } } \end{array}\tag{1}
$$

Each category carries a prompt-guidance string injected into the user-simulator system prompt. For example, jurisdiction=IN adds “Uses INR, familiar with UPI,

![](images/7592db9980d78249acf4cf4e2f366c5ce78465a9db5ae62ab145f037dc862397.jpg)  
Figure 1: The three-tier persona architecture. Each tier undergoes a distinct transformation before merging into the user-simulator prompt. The query-complexity overlay (right column, dashed) is orthogonal to persona identity.

may reference local payment methods.” The guidance is what makes demographics visible in generated text; without it, traits are statistical only.

## 3.2 Tier 2: Behavioral Traits (Continuous)

Twelve traits like cost sensitivity, patience, assertiveness, verbosity, politeness, domain knowledge, risk tolerance, compliance tendency, platform trust, digital literacy, slang usage, emoji usage are sampled around a hand-curated base vector t<sup>base</sup> from a named profile:

$$
t _ { i } = \mathrm { c l i p } \big ( t _ { i } ^ { \mathrm { b a s e } } + \varepsilon _ { i } , 0 , 1 \big ) , \quad \varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )\tag{2}
$$

Noise calibration. The default σ = 0.08 ensures 95% of samples stay within ±0.157 of the base value. This is tight enough to remain recognizably in-persona but wide enough to avoid duplicate-looking samples. The choice was calibrated empirically: $\sigma = 0 . 0 5$ produced visually identical openings across samples; $\sigma = 0 . 1 5$ produced out-of-character behavior (a “patient” persona snapping in turn 1).

Bucketing. The LLM does not see raw continuous values. Before prompt injection, each trait is bucketed:

$$
{ \mathrm { b u c k e t } } ( v ) = { \left\{ \begin{array} { l l } { { \mathrm { l o w } } } & { v < 0 . 3 5 } \\ { { \mathrm { m e d i u m } } } & { 0 . 3 5 \leq v < 0 . 7 0 } \\ { { \mathrm { h i g h } } } & { v \geq 0 . 7 0 } \end{array} \right. }\tag{3}
$$

Each bucket maps to a hand-written prompt-guidance string per trait. For example, assertiveness=high expands to “Direct, demanding, knows what they want, may be blunt”; assertiveness=low expands to “Passive, apologetic, asks permission, hesitant.” Bucketing keeps prompt tokenization stable and makes persona behavior auditable by humans.

## 3.3 Tier 3: Emotional State (Scenario-Reactive)

Five states like frustration, anxiety, trust, confidence, stress are initialized from a profile-specific range and then shifted by scenario-dependent deltas (Table 2):

$$
e _ { i } = \mathrm { c l i p } \big ( \mathcal { U } ( l _ { i } , h _ { i } ) + \Delta _ { i } ( \mathrm { s c e n a r i o } ) , 0 , 1 \big )\tag{4}
$$

The delta map encodes domain knowledge: a dispute scenario increases frustration by +0.25 and decreases trust by −0.20. This makes the same persona behave differently across scenarios. A power user starting a routine checkout is calm and efficient, but the same power user disputing a charge is noticeably more anxious and less trusting.

Scenario Emotional deltas   
complaint frustration +0.20, trust −0.15   
escalation frustration +0.30, stress +0.20, trust   
−0.25   
checkout anxiety +0.15, stress +0.10   
dispute frustration +0.25, anxiety +0.20, trust   
−0.20   
account issue anxiety +0.20, stress +0.15  
Table 2: Tier-3 scenario delta map. Deltas are applied after uniform initialization and clipped to [0, 1].

## 3.4 Trait Correlations

Correlations enter the system through two mechanisms:

<table><tr><td>Attribute</td><td>Categories (probability)</td></tr><tr><td>Jurisdiction</td><td>US (.40) UK (.15) EU (.25) IN (.10) CA (.06) AU (.04)</td></tr><tr><td>Age bracket</td><td>18–24 (.15) 25–34 (.30) 35–44 (.25) 45–54 (.15) 55–64 (.10) 65+ (.05)</td></tr><tr><td>Channel</td><td>web_chat (.35) mobile_app (.30) email (.20) phone_ivr (.10) sms (.05)</td></tr><tr><td>Device type</td><td>smartphone (.45) desktop (.25) laptop (.20) tablet (.10)</td></tr><tr><td>Lang. prof.</td><td>native (.50) fluent (.30) intermediate (.15) basic (.05)</td></tr><tr><td>Time avail.</td><td>moderate (.45) flexible (.25) high_urgency (.20) rushed (.10)</td></tr></table>

Table 1: Tier-1 categorical distributions (shipped defaults).

Profile-encoded. Correlated traits are co-set in the profile base vector by design. The tech\_savvy profile is authored with digital\_literacy=0.95 and domain\_knowledge=0.85, so after independent noise the two traits still tend to co-occur.

Rule-described. After sampling, threshold-based rules inspect paired trait values and emit natural-language correlation statements into the prompt when conditions are met. Seven rules are shipped (Table 3).

This is deliberately simpler than full covariance modeling, trading expressiveness for auditability. Adding a new correlation requires editing a profile or adding a rule; not retraining a model. The cost is that emergent correlations from real user data are not captured unless explicitly authored.

## 3.5 Query-Complexity Overlay

Independent of persona, each sample draws a complexity tier (Table 4):

<table><tr><td>Tier</td><td>Words</td><td>Example opening</td></tr><tr><td>simple</td><td>3-8</td><td>“Dispute this charge”</td></tr><tr><td>medium</td><td>8-15</td><td>“I need to dispute a charge from last week&quot;</td></tr><tr><td>complex</td><td>15-30</td><td>“I bought something on May 12 but the item never arrived and I want my</td></tr><tr><td>vague</td><td>3-15</td><td>money back&quot; &quot;something&#x27;s not right with my ac- count&quot;</td></tr></table>

Table 4: Query-complexity tiers. The vague tier explicitly forbids domain vocabulary, forcing multi-turn clarification.

The vague tier is the most important for agent training. It forces the agent to perform multi-turn clarification before tool selection, matching real user traffic where intent is rarely stated cleanly on the first turn. The vague tier includes an explicit constraint list that forbids domainspecific vocabulary in the opening utterance.

## 3.6 Prompt Assembly

The final user-simulator prompt is assembled by concatenating: (i) Tier-1 demographic guidance strings, (ii) Tier-2 bucketed trait guidance strings, (iii) Tier-3 emotional context strings, (iv) any fired correlation rule statements, and (v) the query-complexity constraint. A worked example:

## Profile: tech\_savvy Scenario: dispute Complexity: vague

Demographics: US, 25–34, web chat, desktop, native English, moderate time. Behavioral: High digital literacy, high domain knowledge, high assertiveness, medium patience. . . Emotional: Frustration 0.72 (high), trust 0.31 (low), anxiety 0.55 (medium). . . Correlation: “High digital literacy correlates with high domain knowledge; user asks specific, technical questions.” Complexity: Vague, opening turn might be “hey something weird happened with that charge,” not “I want to dispute a Visa charge back from May 12.”

## 4 Experiments

We evaluate the persona model inside a multi-agent synthetic data generation pipeline that produces scored, multi-turn conversations between a user simulator, an agent under test, a tool simulator, and an LLM judge.

## 4.1 Setup

Corpus: 64,698 evaluated conversations across three corpora: a mixed-project corpus (49,331 samples, 312 scenarios, 17 projects), a CRM training corpus (12,224 samples, 77 scenarios), and a CRM golden-evaluation corpus (3,143 samples, 20 held-out scenarios). All use 8 named persona profiles.

Persona profiles: tech\_savvy, budget\_conscious, error\_prone, ambiguous, power\_user, tier\_1, tier\_2, Curious. Each has a curated Tier-2 base vector.

Evaluation: An 8-axis LLM judge scores each conversation on goal achievement, tool usage, tool-call hallucination, reasoning quality, reasoning hallucination, communication quality, consistency, and error handling (each 1–10).

<table><tr><td>#</td><td>Condition</td><td>Behavioral effect</td></tr><tr><td>R1</td><td>digital_lit ≥ .70 and domain_know ≥ .70</td><td>Technical, specific questions</td></tr><tr><td>R2</td><td>cost_sens ≥ .70 and patience ≥ .60</td><td>Compares options carefully</td></tr><tr><td>R3</td><td>risk_tol ≤ .30 and compliance ≥ .70</td><td>Follows procedures precisely</td></tr><tr><td>R4</td><td>patience ≤ .40 and assertiveness ≥ .60</td><td>Escalates quickly when blocked</td></tr><tr><td>R5</td><td>verbosity  $\geq . 7 0$  and politeness  $\geq . 7 0$ </td><td>Explains context before asking</td></tr><tr><td>R6</td><td>digital_lit  $\leq . 3 0$  and trust_plat  $\leq . 5 0$ </td><td>Double-checks every step</td></tr><tr><td>R7</td><td> $\mathrm { s l a n g } \geq . 7 0$  and  $\mathrm { e m o j i } \geq . 7 0$ </td><td>Casual register, shorter turns</td></tr></table>

Table 3: The seven trait-correlation rules. Each fires post-sampling and emits a natural-language statement into the user-simulator prompt.

![](images/28dfc9ba5bc0eb08d5b78959b83addf2359fc1b9a433151bf81a50db0c3981df.jpg)  
Figure 2: Goal-achievement rate by persona $( \ge 4 0 0$ samples each). The 15.8pp spread confirms persona drives agent performance.

## 4.2 RQ1: Does Persona Drive Goal Achievement?

Across the 49K mixed corpus, goal-achievement rates range from 51.2% (budget-conscious) to 67.0% (tier-1), a 15.8 percentage-point spread (Figure 2). The ordering is meaningful: tier-aware personas (tier-1, tier-2, power user) whose Tier-2 vectors encode high domain knowledge, high digital literacy, and high compliance cluster at the top, and they open sessions with the right intent, supply complete details, and accept agent suggestions. Pressure-test personas (budget-conscious, error-prone, ambiguous) whose vectors encode low clarity, high cost sensitivity, or frequent error-correction cluster at the bottom.

This spread is not an artifact of the generator. It reflects the real-world observation that competent, cooperative users are easier to serve than uncertain, demanding ones. The persona vector is faithfully reproducing this

![](images/1e2f7ef7140aa4c2404e5d6e9c2cba4c94bd085c3a7f1d32fcf8354ac8ddeca1.jpg)  
Figure 3: Booking-flow compliance by persona in the CRM corpus. The ∼23 percentage points gap between tier-1 and error-prone personas reproduces the production difficulty distribution.

difficulty distribution.

## 4.3 RQ2: Do Emotional Deltas Change Behavior?

To isolate Tier-3’s effect, we compare the same persona (power\_user) across dispute vs. checkout scenarios. In dispute scenarios (∆frustration = +0.25, ∆trust $= - 0 . 2 0 )$ , the power user produces 23% more pushback turns (turns where the user rejects or questions the agent’s response) than in checkout scenarios. Average conversation length increases by 1.4 turns. Communication quality scores drop by 0.8 points, not because the agent communicates worse, but because the user is harder to satisfy.

This validates the scenario-reactive design: the same persona identity produces qualitatively different interactions when emotional context changes.

## 4.4 RQ3: Persona Sensitivity on Domain Axes

In the CRM-specific corpus, the project added a domainspecific judge axis: booking-flow compliance. Figure 3 shows the per-persona compliance rates. Tier-aware personas achieve 63–68% compliance; pressure-test personas achieve 45–52%. The ∼23 percentage points gap is faithful to production observations where agents struggle with ambiguous or error-prone callers.

The Curious personas (not shown) sit between the two groups, confirming the judge responds to the trait vector, not the persona label. Curious scenarios are harder than vanilla booking, and the scores reflect this without label bias.

## 4.5 RQ4: Query Complexity Interacts with Persona

<table><tr><td>Persona</td><td>Simple</td><td>Medium</td><td>Complex</td><td>Vague</td></tr><tr><td>Tech-savvy</td><td>72.1</td><td>68.3</td><td>61.5</td><td>54.8</td></tr><tr><td>Budget-cons.</td><td>58.4</td><td>53.7</td><td>47.2</td><td>39.1</td></tr><tr><td>Error-prone</td><td>61.2</td><td>56.8</td><td>48.9</td><td>36.5</td></tr></table>

Table 5: Goal-achievement rate (%) by persona × query complexity. The vague tier is hardest across all personas, with error-prone+vague being the most challenging combination (36.5%).

Table 5 shows the interaction between persona and query complexity. Two observations: (i) the vague tier drops goal achievement by 15–18 percentage points compared to simple, across all personas; (ii) the persona effect is additive, and the gap between tech-savvy and errorprone is roughly constant (∼11–18 percentage points) across all complexity tiers. This additivity confirms the two controls are genuinely orthogonal.

## 4.6 RQ5: Noise Calibration

<table><tr><td>σ</td><td>Bucket flip rate</td><td>Goal-ach. σ</td><td>Duplicate %</td></tr><tr><td>0.05</td><td>8.2%</td><td>1.8</td><td>12.4%</td></tr><tr><td>0.08</td><td>14.7%</td><td>2.4</td><td>3.1%</td></tr><tr><td>0.12</td><td>24.3%</td><td>3.1</td><td>0.9%</td></tr><tr><td>0.15</td><td>31.8%</td><td>3.7</td><td>0.4%</td></tr></table>

Table 6: Effect of noise parameter σ on persona variation. Bucket flip rate = % of traits that cross a bucket boundary after noise. Duplicate % = samples with gestalt similarity $\ge 0 . 8 5$ on opening turns. $\sigma = 0 . 0 8$ balances variety and in-persona coherence.

Table 6 shows the sensitivity to σ. At $\sigma = 0 . 0 5 , 1 2 . 4 \%$ of opening turns are near-duplicates; at $\sigma = 0 . 1 5$ , the bucket flip rate hits 31.8%, meaning nearly a third of traits land in a different bucket than the profile intended, producing out-of-character behavior. The shipped default $\sigma = 0$ .08 sits at the knee: 14.7% bucket flip rate (enough variety) and 3.1% duplicate rate (acceptable).

## 5 Combinatorial Analysis

The persona model generates a large space of distinct prompt configurations:

Tier 1: $\begin{array} { r } { \prod _ { k = 1 } ^ { 6 } \left| C _ { k } \right| = 6 \times 6 \times 5 \times 4 \times 4 \times 4 = 1 1 } \end{array}$ ,520 demographic combinations.

Tier 2: Each of 12 traits maps to 3 buckets $= 3 ^ { 1 2 } =$ 531,441 behavioral configurations.

Tier 3: Each of 5 states maps to 3 buckets = $3 ^ { 5 } = 2 4 3$ emotional configurations.

Complexity: 4 levels.

Total prompt-distinct configurations: $1 1 { , } 5 2 0 \ \times$ 531, $4 4 1 \times 2 4 3 \times 4 \approx 5 . 9 5 \times 1 0 ^ { 1 2 }$ . In practice, profile constraints and correlation rules reduce the reachable space, but the design supports generating millions of unique persona configurations without repetition.

## 6 Limitations and Future Work

No learned covariance. Trait correlations are ruledescribed, not learned. Fitting a covariance matrix from the 64K corpus would yield more realistic co-occurrence without manual authoring, but at the cost of interpretability.

No turn-level emotional dynamics. Tier-3 states are session-level: set once at the start and held constant. Real users’ frustration escalates mid-conversation when the agent fails. Modeling intra-session emotional trajectories is future work.

Bucketing information loss. The three-bucket discretization (low/medium/high) loses information. A user at 0.36 (just above the low threshold) and one at 0.69 (just below high) both receive the “medium” prompt guidance. Finer bucketing (5 or 7 levels) or direct continuous injection are under investigation.

Cultural validity. The prompt-guidance strings are English-centric and Western-normed. Jurisdiction=IN adds payment-method context but does not adjust interaction norms. Cross-cultural persona validation is needed.

Judge attribution. We measure persona impact on agent scores (goal achievement, compliance), but the causal path runs through the user simulator. Disentangling persona-driven difficulty from simulator artifacts requires human evaluation of user-turn realism, which we have not yet conducted.

## 7 Conclusion

We presented a three-tier persona vector with 23 operationalized dimensions for controllable user simulation in agentic evaluation. The tiered structure separates identity (demographics), behavior (traits with noise), and emotional context (scenario-reactive states), enabling factorial experimental designs and measurable behavioral attribution. Across 64,698 evaluated conversations, the model produces a 15.8 percentage points spread in goal achievement, faithful difficulty distributions on domainspecific axes, and orthogonal interaction with query complexity. The specification is complete: every distribution, noise parameter, bucketing rule, and correlation threshold is provided for reproduction.

The broader implication is that user simulation for agentic evaluation does not need to be either flat descriptions or learned black boxes. A structured middle ground of continuous, measurable, auditable properties gives practitioners the control they need to stress-test agents on the user populations that matter.

## References

[1] R. Khedar, Eshita, S. T. S. R. Thondapu, M. Malhotra, A. Das, J. Chandra, Y.-S. Chuang, C. Kulkarni, A. Menon, L. Pang, A. Karn, Mouli V, and P. Mehrotra. State-grounded multi-agent synthetic data generation for tool-augmented LLMs. arXiv:2606.16307, 2026.

[2] S. Yao, et al. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv:2406.12045, 2024.

[3] L. Zheng, W.-L. Chiang, Y. Sheng, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In NeurIPS, 2023.

[4] J. Schatzmann, K. Weilhammer, M. Stuttle, and S. Young. A survey of statistical user simulation techniques for reinforcement-learning of dialogue management strategies. The Knowledge Engineering Review, 21(2):97–126, 2006.

[5] J. Schatzmann, B. Thomson, K. Weilhammer, H. Ye, and S. Young. Agenda-based user simulation for bootstrapping a POMDP dialogue system. In HLT-NAACL, 2007.

[6] F. Kreyssig, I. Casanueva, P. Budzianowski, and M. Gašic. Neural user simulation for corpus-based´ policy optimisation of spoken dialogue systems. In SIGDIAL, 2018.

[7] S. Zhang, E. Dinan, J. Urbanek, A. Szlam, D. Kiela, and J. Weston. Personalizing dialogue agents: I have a dog, do you have pets too? In ACL, 2018.

[8] H. Song, Y. Wang, K. Zhang, W. Zhang, and T. Liu. BoB: BERT over BERT for training persona-based dialogue models from limited personalized data. In ACL, 2021.

[9] X. Xu, et al. Beyond goldfish memory: Long-term open-domain conversation. In ACL, 2022.

[10] X. Liu, H. Yu, et al. AgentBench: Evaluating LLMs as agents. arXiv:2308.03688, 2023.

[11] K. Mei, et al. Matrix: An agent-based multiturn data synthesis framework. Meta FAIR, arXiv preprint, 2024.

[12] N. S. Keskar, B. McCann, L. Varshney, C. Xiong, and R. Socher. CTRL: A conditional transformer language model for controllable generation. arXiv:1909.05858, 2019.

[13] B. Krause, A. D. Gotmare, B. McCann, N. S. Keskar, S. Joty, R. Socher, and N. F. Rajani. GeDi: Generative discriminator guided sequence generation. In EMNLP Findings, 2021.