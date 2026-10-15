# BANGLAPLAN: A Benchmark for Constraint-Based Planning and Decision-Making in Bangla

Md Rayhanul Masud, S M Maksudul Alam, Mohammad Jaminur Islam University of California, Riverside

## Abstract

Large language models (LLMs) increasingly help users arrange appointments, plan jour neys, and allocate limited resources. These tasks require models to satisfy constraints on time, cost, capacity, and order, or determine that no feasible solution exists. Bangla is the world’s seventh most widely spoken language, and existing benchmarks cover many language and reasoning tasks. However, benchmarks for evaluating planning and decision-making in Bangla under explicit constraints remain lacking. We introduce BANGLAPLAN, a benchmark of 1,000 problems across ten task families grounded in everyday life and insti tutions in Bangladesh. It includes feasible and infeasible problems in four language registers: everyday, formal, colloquial, and Bangla–English mixed. Answers cover eight structured types, including schedules, routes, allocations, and ordered steps, in a common JSON format. We construct the benchmark through language-model-assisted drafting and independent human validation. We evalu ate 11 open-weight and proprietary model configurations using exact match, separately reporting feasible-case accuracy, infeasibil ity detection, and format errors. We find a large performance gap: GPT-6.1 Sol achieves 98.7% accuracy and Qwen3.5-27B with thinking reaches 80.3%, while direct-answer configurations score at most 30.9%. Seven of eleven configurations perform below the always-“infeasible” baseline. Our findings also suggest that harder problems primarily reduce plan-construction accuracy, while infeasibility detection remains more robust. BANGLAPLAN provides a reproducible basis for evaluating planning and decision-making in Bangla. BANGLAPLAN is available at: https://github.com/mrayhanulmasud/banglapla

## 1 Introduction

Large language models (LLMs) can follow natural-language instructions and perform multistep reasoning (Brown et al., 2020; Wei et al., 2022). Their evaluation has therefore expanded from question answering and text generation to planning and constrained decision making (Valmeekam et al., 2023; Zheng et al., 2024; Tso et al., 2026; Zhao et al., 2026; Zhang et al., 2026), including scheduling (Zheng et al., 2024; Tso et al., 2026), travel planning (Zheng et al., 2024; Xie et al., 2024), and resource allocation (Tso et al., 2026). A valid response must satisfy every constraint (Xie et al., 2024; Tso et al., 2026) or, when that is impossible, recognize that no feasible solution exists (Qi et al., 2026).

Existing planning benchmarks span formal, natural-language, and real-world settings (Valmeekam et al., 2023; Zheng et al., 2024; Xie et al., 2024). Recent work covers practical tasks (Zhao et al., 2026), constrained optimization (Tso et al., 2026), long-horizon planning (Zhang et al., 2026), and infeasible problems (Qi et al., 2026). Bangla is the seventh most widely spoken language, yet it is rarely represented in planning benchmarks. Existing Bangla benchmarks cover subject knowledge (Joy and Shatabda, 2026), question answering (Shafayat et al., 2024), mathematical reasoning (Prama et al., 2025), and social interaction (Sijan et al., 2026). Multilingual planning and workflow benchmarks do not include Bangla (Jung et al., 2025; Ranaldi et al., 2026), while other multilingual evaluations include Bengali but target different capabilities (Pahuja et al., 2026; Jain et al., 2025). These benchmarks leave a gap in evaluating models’ ability to plan and make decisions under explicit constraints in Bangla.

Research questions. This gap motivates two research questions:

1. RQ1: Overall performance. How well do current models construct valid plans and identify infeasible requests under explicit constraints in Bangla?

![](images/1ad8f38caf4936304104a79d4717a6d711484a29d3798bea10c07eff51a3ce01.jpg)  
Figure 1: A BANGLAPLAN problem and its near-boundary infeasible variant. Left: the Bangla request with an English translation. A student needs 10 minutes to print her admit card at a shop that opens at 8:00, followed by 25 minutes of travel to the exam centre. Right: a gate-closing time of 8:45 permits the gold plan; changing it to 8:30 makes the request infeasible, with gold answer is\_feasible: false.

2. RQ2: Variation and errors. How does performance vary across benchmark dimensions, and which failure modes dominate?

Contributions. We make the following contributions to address these questions:

• A Bangla planning benchmark. We introduce BANGLAPLAN, a benchmark of 1,000 problems authored directly in Bangla across ten task families grounded in Bangladesh. It contains 744 feasible and 256 infeasible problems, with 300 easy, 300 medium, and 400 hard instances. The problems span four language registers and eight structured answer types in a common JSON format. Section 3 describes the construction process. Figure 1 presents an example.

• Validated reference answers. We combine language-model-assisted drafting with independent human validation to establish canonical answers for deterministic exact-match scoring.

• Model evaluation and error analysis. We evaluate 11 open-weight and proprietary model configurations, separating plan accuracy, infeasibility detection, and format errors, and analyzing performance across benchmark dimensions.

<sup>ক্ষা</sup> <sup>রী</sup> <sup>ক্ষা</sup> <sup>কে</sup> <sup>ন্দরে</sup> <sup>যা</sup> <sup>য়া</sup> <sup>গে</sup>Results. Our evaluation yields the following key <sub>া খো লে কা</sub>findings:

<sup>বং</sup> <sup>দো</sup> <sup>কা</sup> <sup>থে</sup> <sup>কে</sup> <sup>রী</sup> <sup>ক্ষা</sup> <sup>কে</sup> <sup>ন্দরে</sup>• RQ1. Test-time reasoning yields larger gains than scaling. Models using test-time reasoning perform best: GPT-6.1 Sol achieves 98.7% exact match, GPT-6 Luna 85.1%, and

Qwen3.5-27B with thinking 80.3%. Directanswer configurations reach at most 30.9%, and seven of eleven configurations score below the 25.6% always-“infeasible” baseline. Enabling thinking with the same 27B weights adds 49.4 percentage points, exceeding the gain from scaling Qwen3.5 from 2B to 27B.

• RQ2. Difficulty affects planning more than infeasibility detection. Harder problems reduce plan accuracy more than infeasibility detection: GPT-6 Luna’s plan accuracy falls from 96.0% to 69.0%. Constructed answers, such as schedules, are harder than label selection. Format errors further limit performance, affecting 70.3% of Gemma 3 outputs.

• Translation and fine-tuning offer limited gains in plan construction. English prompts and gold-answer fine-tuning mainly reduce format errors and change the frequency of “infeasible” responses, while improving plan construction by only a few percentage points.

## 2 Related Work

Planning benchmarks. LLM planning benchmarks, mainly in English, vary in problem formality and task coverage. PLANBENCH evaluates plan generation and reasoning through PDDL problems (Valmeekam et al., 2023). NATURAL PLAN covers natural-language trip, meeting, and calendar planning (Zheng et al., 2024), while TRAVELPLANNER evaluates travel plans against real-world constraints (Xie et al., 2024). PLAN-NINGBENCH (Zhao et al., 2026) broadens coverage to six problem families, and DEEPPLAN-

Table 1: The ten task families in BANGLAPLAN, each comprising ten problem types with ten instances per type.
<table><tr><td>Family</td><td>Representative problem types</td><td>Family</td><td>Representative problem types</td></tr><tr><td>F01 Eligibility</td><td>scholarships, exams, leave, mobile-money rules</td><td>F06 Sequencing</td><td>recipes, construction, crop calendars</td></tr><tr><td>F02 Procedures</td><td>NID correction, passports, land mutation</td><td>F07 Allocation</td><td>beds, relief rice, irrigation, Qurbani meat</td></tr><tr><td>F03 Appointments F04 Journeys</td><td>clinics, biometrics, banks, legal aid Eid travel, ferries, hospital referrals</td><td>F08 Procurement F09 Logistics</td><td>haat shopping, Iftar supplies, tenders ports, ferry berths, waste collection</td></tr><tr><td>F05 Rosters</td><td>exams, invigilation, garment shifts</td><td>F10 Emergency</td><td>evacuation, triage, ambulance dispatch</td></tr></table>

NING examines long-horizon travel and shopping tasks (Zhang et al., 2026). Translating English datasets can introduce errors and lose cultural nuance (Salazar et al., 2026; Romanou et al., 2025), motivating benchmarks rooted in local contexts. BANGLAPLAN addresses this need with problems authored directly in Bangla and grounded in everyday life in Bangladesh.

Multilingual planning. Multilingual planning evaluation remains narrow. MULTI-PLAN compares Korean and English travel planning (Jung et al., 2025), CHINATRAVEL targets Chinese (Shao et al., 2026), and TICKET-BENCH covers six European languages, with 15% of its queries infeasible (Almeida et al., 2025). WORLDBENCH builds scheduling, travel, and expense tasks natively in seven languages and eight cultures, but includes no South Asian language (Ranaldi et al., 2026). Two resources include Bangla, yet neither targets locally grounded planning under explicit constraints: GAIA-MAPS diagnoses multi-agent planning failures across eleven languages, with Bangla tasks produced by machine translation of GAIA (Pahuja et al., 2026), and VOICEAGENT-BENCH tests spoken tool calls built from translated Indian-context queries in seven languages (Jain et al., 2025).

Bangla evaluation. Existing Bangla benchmarks cover subject knowledge (Joy and Shatabda, 2026), science question answering and reasoning in Bangla and English (Shafayat et al., 2024), reading comprehension (Rony et al., 2024), school mathematics (Prama et al., 2025), and sociopragmatic and cultural alignment (Sijan et al., 2026). Prior work also studies whether script fragmentation and prompt format explain the English– Bangla performance gap (Bhowmik et al., 2025). A few Bangla datasets contain items with no valid answer: NCTB-QA labels 42.75% of its questions unanswerable (Eyasir et al., 2026), and

GSM-PLUS-BN includes math problems with missing information (Paul et al., 2026). Both, however, test reading or arithmetic rather than planning. To our knowledge, no Bangla benchmark evaluates planning and decision making under explicit constraints.

## 3 The BANGLAPLAN Benchmark

BANGLAPLAN benchmarks constraint-based planning in Bangla across tasks such as scheduling, routing, and allocation. Each instance requires a valid structured answer or an infeasibility decision. Explicit constraints, objectives, and tie-breaking rules determine a unique canonical answer for deterministic scoring without an LLM judge.

## 3.1 Task Definition

Each instance is a tuple $x = ( p , C , \operatorname { f } , \tau )$ , where p is a Bangla prompt specifying hard constraints $C ,$ ordered objectives ${ \mathrm { f } } = \left( f _ { 1 } , \ldots , f _ { K } \right)$ , and a tiebreaking rule τ . A plan y is a well-formed answer of the required type, such as a schedule, route, or allocation. Let $S _ { 0 } = S ( x )$ denote all plans satisfying C. For feasible instances, we apply the objectives sequentially:

$$
S _ { k } = \operatorname * { a r g o p t } _ { y \in S _ { k - 1 } } f _ { k } ( y ) , \qquad k = 1 , \ldots , K ,
$$

where each objective is minimized or maximized as specified in the prompt. The gold answer is

$$
y ^ { * } ( x ) = { \left\{ \begin{array} { l l } { { \mathrm { I N F E A S I B L E } } } & { { \mathrm { i f ~ } } S ( x ) = \emptyset , } \\ { \tau ( S _ { K } ) } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right. }
$$

where $\tau$ selects one plan from $S _ { K }$ . This defines a lexicographic objective order: each objective is optimized only among plans satisfying all earlier objectives. For example, a scheduling problem may prioritize the earliest completion time, then the earliest start of the first task, and finally task-ID order to resolve remaining ties. Some instances require no optimization $( K = 0 )$ , as in eligibility decisions where the constraints determine a single answer. The stated objectives and tie-breaking rules ensure a unique gold answer.

![](images/12ca0943fc091842977c78fe206e1a84d9613d715e8b3c973460b5e7b2dce4b5.jpg)  
Figure 2: Construction of BANGLAPLAN. Authors define the taxonomy. Annotators select and rewrite Claudesuggested scenarios in Bangla, creating feasible and near-boundary infeasible problems with unique canonical gold answers. An independent annotator solves each problem without seeing the gold answer, followed by a second check using Claude Opus 5.5. Humans resolve disagreements and revise flagged issues (red).

The model must return a one-element JSON array. A feasible response takes the form

$$
[ \{ { ^ { n } \mathrm { i } } \mathsf { s } _ { - } \mathsf { f e a s i b l e } ^ { n } : \mathsf { t r u e } , { ^ { n } \mathrm { v e r i f i e d } _ { - } } \mathsf { a n s w e r } ^ { n } : \{ \dots \} \} ]
$$

and an infeasible response is

$$
[ \{ { ' } ^ { n } \mathrm { i } s _ { - } \mathsf { f e a s i b l e } ^ { \prime } : \mathsf { f a l s e } \} ]
$$

The verified\_answer object follows one of eight schemas: classification, ordered steps, schedule, route, allocation, matching, selection, or ranking. A response is correct if it parses as JSON, conforms to the required schema, and matches the gold answer’s JSON representation after deterministic canonicalization. Parsing and schema violations are recorded as format errors.

Coverage. BANGLAPLAN contains 1,000 problems across ten task families, each with ten problem types and ten instances per type (Table 1). It includes 744 feasible and 256 infeasible instances, divided into 300 easy, 300 medium, and 400 hard problems. Prompts span four registers: everyday, formal, colloquial, and Bangla–English codemixed. Answers follow eight types: allocation, classification, matching, ordered steps, ranking, route, schedule, and selection. Appendix D reports counts by register and answer type.

Linguistic variation. Prompts vary in how they express numbers, dates, and times. Numbers appear as Bangla digits, English digits, or words; dates may follow the Bengali calendar; and times may be relative (“next Thursday”) or expressed conversationally (“half past ten”). Relative-date problems specify a reference date and time, while Bengali-calendar problems provide the information needed for conversion. Output values consistently use ASCII digits and ISO-8601 timestamps with the +06:00 offset. Each problem has an English version preserving its entities, quantities, constraints, and gold answer as we describe later in Section 4. The Bangla version is the original, authored directly in Bangla.

## 3.2 Benchmark Construction

We construct BANGLAPLAN in six stages: taxonomy design, scenario drafting, problem construction, gold-answer creation, human validation, and a model-assisted consistency check. Figure 2 illustrates the benchmark construction pipeline.

Annotators and pilot. Five native Bangla speakers with relevant experience volunteered: three drafted problems and two independently validated them. A 100-problem pilot refined guidelines, answer schemas, and rules for ambiguity, feasibility, and tie-breaking. Appendix C provides the annotation and validation guidelines.

Stage 1: Taxonomy design. We defined ten task families reflecting planning and decision-making needs in Bangladesh, including administrative procedures, river travel, garment shifts, irrigation, and emergency response. Each family contains ten concrete problem types. Before drafting, we established five constraint categories temporal, spatial, resource-based, logical, and quantitative and the eight answer schemas described in Section 3.1.

Stage 2: Scenario drafting. Annotators used Claude Sonnet 5.5 to suggest people, places, resources, rules, and plausible quantities for each problem type. They selected, revised, and checked these suggestions for natural wording, local relevance, and internal consistency. Each scenario was recorded in a structured form specifying entities, identifiers, quantities, time windows, and constraints. Annotators then authored the Bangla prompt directly from this record. Every prompt explicitly states that its names, dates, and rules are fictional.

Stage 3: Problem construction. Feasible problems admit at least one plan satisfying all constraints $( S ( x ) ~ \neq ~ \emptyset )$ Annotators created infeasible variants by changing one binding quantity, time window, capacity, or policy rule until no valid plan remained $( S ( x ) ~ = ~ \emptyset )$ We prioritized near-boundary conflicts, such as requiring 187 units when only 186 are available. An instance is minimally infeasible only if relaxing its designated conflicting constraint restores feasibility. This design requires precise constraint checking. Some prompts also include irrelevant facts or non-binding preferences to test robustness; these do not affect the gold answer.

Stage 4: Gold-answer creation. We establish a unique canonical answer for each feasible instance through three steps. First, we order objectives lexicographically, such as maximizing savings before educational benefit. Second, explicit tie-breaking rules resolve remaining choices. Third, we clarify potentially ambiguous conventions, including whether the start date counts as day 0 and whether N days means $N \times 2 4$ hours.

We also standardize output representation. Identifiers must match those in the prompt, and zerovalued allocations are omitted. Allocations follow prompt order, while scheduled actions follow start-time order, with ties resolved by the prompt’s task-ID order. The scorer parses and canonicalizes valid JSON before exact matching, making scores independent of whitespace and key order.

## 3.3 Validation and Difficulty

Stage 5: Human validation. Each author derived the gold answer using the stated constraints, objectives, and tie-breaking rules. An independent validator then solved the problem from the Bangla prompt alone, without seeing the gold answer, and checked its completeness and clarity. The author and validator resolved disagreements about feasibility, the plan, or the output schema. Revised problems underwent validation again.

Stage 6: Model-assisted consistency check. Claude Opus 5.5 independently solved every human-validated problem without access to its gold answer. Mismatches and prompts flagged as ambiguous or incomplete underwent human review. Claude suggested possible causes and revisions, which annotators accepted, edited, or rejected. Revised problems were solved and checked again until humans confirmed that any remaining mismatches were model errors. Claude’s outputs served as review signals; human judgment determined the final prompts and gold answers.

Difficulty. Annotators recorded the binding constraints and supporting facts required to solve each problem. We define structural difficulty by their total count k: easy instances have $k \leq 1 0$ , medium instances have $1 0 < k \le 2 0$ , and hard instances have $k > 2 0 .$ . We fixed both thresholds before model evaluation, yielding 300 easy, 300 medium, and 400 hard instances. This measure reflects structural complexity rather than prompt length.

## 4 Evaluation

We discuss the experimental setup for the evaluation of the performance of different LLMs against BANGLAPLAN.

Models. We evaluate 11 configurations across four groups, as shown in Table 2: two frontier models, GPT-6 Luna and GPT-6.1 Sol; Qwen3.5 at four sizes (2B, 4B, 9B, and 27B), with an additional thinking-enabled 27B configuration to assess test-time reasoning at fixed weights; the Bangla-adapted TigerLLM (12B) (Raihan and Zampieri, 2025) alongside Gemma 3 (12B); and two additional open models, Gemma 2 (9B) and Llama 3.1 (8B). We use greedy decoding (temperature = 0) for all configurations.

Protocol. We evaluate all models zero-shot using the released BANGLAPLAN prompts, which specify the scenario, constraints, tie-breaking rule, and required JSON-array format. We set reasoning effort to low for the two frontier models and disable thinking for all open models. We additionally evaluate Qwen3.5-27B with thinking enabled. We discuss the prompt template in Appendix A.

Scoring. An answer is correct if it parses as the required JSON array and exactly matches the gold answer after canonicalization, as we discussed in Section 3. The benchmark contains 744 feasible instances and 256 infeasible instances. Table 2 reports overall accuracy and outcome breakdowns. EM (exact match), our primary metric, measures accuracy across all instances. For feasible instances, Plan measures correct-plan accuracy, Wrong plan measures schema-compliant but incorrect plans, and Abstention measures how often a model answers “infeasible” despite a valid plan existing. For infeasible instances, Flag measures correct “infeasible” responses, while False plan measures schema-compliant responses proposing a plan when none exists. Format measures outputs that violate the required JSON-array format or answer schema, reported overall and separately for feasible and infeasible instances.

Table 2: Zero-shot results on 1,000 BANGLAPLAN instances (all values in %). Each feasible or infeasible block sums to 100, up to rounding. EM: exact-match accuracy over all instances. For feasible instances, Plan denotes a correct plan, Wrong plan an incorrect plan, and Abstention an incorrect “infeasible” answer. For infeasible instances, Flag denotes a correct “infeasible” answer and False plan a proposed plan. Format denotes failure to return the required JSON array. Bold and underline mark the best and second-best EM, Plan, and Flag scores. Gray marks EM below the always-“infeasible” baseline of 25.6.
<table><tr><td></td><td colspan="3">Al1 1,000 instances</td><td colspan="4">744 feasible instances</td><td colspan="3">256 infeasible instances</td></tr><tr><td>Model</td><td>EM↑</td><td>Incorrect</td><td>Format</td><td>Plan↑</td><td>Wrong plan</td><td>Absten- tion</td><td>Format</td><td>Flag↑</td><td>False plan</td><td>Format</td></tr><tr><td>Always “infeasible”</td><td>25.6</td><td>74.4</td><td>0.0 |</td><td>0.0</td><td>0.0</td><td>100.0</td><td>0.0 |</td><td>100.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="9">FRONTIER MODELS (REASONING EFFORT: LOW)</td><td></td><td></td></tr><tr><td>GPT-6 Luna</td><td>85.1</td><td>14.9</td><td>0.0</td><td>82.7</td><td>14.8</td><td>2.6</td><td>0.0</td><td>92.2</td><td>7.8</td><td>0.0</td></tr><tr><td>GPT-6.1 Sol</td><td>98.7</td><td>1.3</td><td>0.0</td><td>98.4</td><td>0.8</td><td>0.8</td><td>0.0</td><td>99.6</td><td>0.4</td><td>0.0</td></tr><tr><td colspan="9">QWEN3.5, FOUR SIZES (THINKING OFF UNLESS MARKED)</td><td></td></tr><tr><td>Qwen3.5-2B</td><td>3.3</td><td>71.9</td><td>24.8</td><td>4.2</td><td>70.2</td><td>0.1</td><td>25.5</td><td>0.8</td><td>76.6</td><td>22.7</td></tr><tr><td>Qwen3.5-4B</td><td>14.1</td><td>84.9</td><td>1.0</td><td>8.3</td><td>73.8</td><td>16.5</td><td>1.3</td><td>30.9</td><td>69.1</td><td>0.0</td></tr><tr><td>Qwen3.5-9B</td><td>19.6</td><td>79.6</td><td>0.8</td><td>12.8</td><td>64.8</td><td>21.8</td><td>0.7</td><td>39.5</td><td>59.4</td><td>1.2</td></tr><tr><td>Qwen3.5-27B</td><td>30.9</td><td>69.1</td><td>0.0</td><td>19.5</td><td>39.5</td><td>41.0</td><td>0.0</td><td>64.1</td><td>35.9</td><td>0.0</td></tr><tr><td>Qwen3.5-27B (think)</td><td>80.3</td><td>13.9</td><td>5.8</td><td>76.9</td><td>14.5</td><td>2.0</td><td>6.6</td><td>90.2</td><td>6.2</td><td>3.5</td></tr><tr><td colspan="9">BANGLA-ADAPTED MODEL AND ITS BASE</td><td></td><td></td></tr><tr><td>Gemma 3 12B</td><td>8.5</td><td>21.2</td><td>70.3</td><td>10.3</td><td>17.5</td><td>1.9</td><td>70.3</td><td>3.1</td><td>26.6</td><td>70.3</td></tr><tr><td>TigerLLM 12B</td><td>11.1</td><td>69.5</td><td>19.4</td><td>13.3</td><td>63.8</td><td>2.6</td><td>20.3</td><td>4.7</td><td>78.5</td><td>16.8</td></tr><tr><td colspan="9">OTHER OPEN MODELS</td><td></td><td></td></tr><tr><td>Gemma 2 9B</td><td>9.7</td><td>86.8</td><td>3.5</td><td>12.2</td><td>84.0</td><td>0.8</td><td>3.0</td><td>2.3</td><td>92.6</td><td>5.1</td></tr><tr><td>Llama 3.1 8B</td><td>6.7</td><td>84.2</td><td>9.1</td><td>6.9</td><td>80.0</td><td>4.0</td><td>9.1</td><td>6.2</td><td>84.8</td><td>9.0</td></tr></table>

EM combines plan construction and infeasibility detection:

$$
\mathrm { E M } = 0 . 7 4 4 \cdot \mathrm { P l a n } + 0 . 2 5 6 \cdot \mathrm { F l a g } .\tag{1}
$$

Reporting these components separately shows whether a model succeeds by constructing valid plans or detecting infeasible requests. As a reference baseline, a policy that always returns “infeasible” in the required format achieves 25.6% EM with no format errors. Models below this baseline perform worse than a fixed response that exploits the benchmark’s label distribution.

English translations. To assess how prompt language affects performance, we translate all prompts from Bangla into English using Claude Sonnet 5.5. We preserve identifiers, numbers, times, and the JSON answer block verbatim, allowing the same gold answers to be used for both versions. We evaluate GPT-6.1 Sol and Qwen3.5- 2B/4B/9B on both language versions.

Fine-tuning. To test whether lightweight supervision improves small-model performance, we fine-tune Qwen3.5-2B using LoRA (rank 16, α=32, dropout 0.05) for 3 epochs, with a learning rate of $2 \times 1 0 ^ { - 4 }$ , 5% warm-up, and a batch size of 4. The training target is the gold JSON answer, and loss is computed only on target tokens. We train on randomly selected 834 instances and report all fine-tuning results on the remaining 166 held-out instances, including 42 infeasible cases. We ran the fine-tuning experiment on a MacBook Pro with 64 GB of memory.

## 5 Results

We address RQ1 through the overall results in Table 2 and the score decomposition §5.1. We address RQ2 through breakdowns by difficulty, answer type, and failure mode, alongside the language and fine-tuning experiments (§5.2–5.4).

## 5.1 RQ1: Performance across models and reasoning settings

Models with reasoning lead in exact match. Table 2 shows a clear performance gap. GPT-6.1 Sol achieves the highest EM at 98.7%, followed by GPT-6 Luna at 85.1% and Qwen3.5-27B with thinking at 80.3%. These three configurations use test-time reasoning. In contrast, all other configurations score at or below 30.9%, and 7 of 11 perform below the always-“infeasible” baseline.

![](images/304bd3b44e4d68b53562462292b527e742737985e96c7ff67144ce16f39ff52f.jpg)  
Figure 3: Exact-match score decomposed into correct plans (blue) and “correct infeasible” answers (orange). The dashed line shows “always-infeasible” baseline.

Exact match masks different response behaviors. Similar EM scores can reflect different behaviors. Figure 3 separates contributions from correct plans and correctly flagged infeasible instances. The three reasoning models earn most of their scores through correct plans. In contrast, Qwen3.5-27B without thinking earns 53% of its 30.9% EM from correctly answering “infeasible”. It returns this answer on 46.9% of all instances, almost twice the benchmark’s infeasible share of 25.6%. Gemma, LLaMA, and TigerLLM show the opposite tendency: they rarely identify infeasible instances, with Flag scores no higher than 6.2%. Appendix B audits Wrong plan errors for two models, Qwen3.5-2B and GPT-6.1 Sol, and finds mostly hard-constraint violations for the former and more tie-break differences for the latter.

Thinking yields larger gains than scaling or Bangla adaptation. Within the Qwen3.5 family, increasing model size from 2B to 27B raises EM from 3.3% to 30.9%, averaging 7.4 percentage points per parameter doubling. Plan accuracy improves more slowly, by 4.1 points per doubling. Enabling thinking with the same 27B weights yields a larger gain than scaling across this range: EM rises by 49.4 points to 80.3%, Plan increases from 19.5% to 76.9%, and incorrect abstentions fall from 41.0% to 2.0%.

Bangla adaptation offers a much smaller gain. Compared with its base model, Gemma 3 12B, TigerLLM improves EM from 8.5% to 11.1%, alongside a substantial reduction in format errors from 70.3% to 19.4%. Both models remain below the always-“infeasible” baseline.

![](images/b820b9538333475ad0bb3021ad85f5fb96957b5bce22f8a7050d11f05a95b719.jpg)  
Figure 4: Planning accuracy drops with difficulty, while infeasibility detection remains stable. Left: correct plans on feasible instances. Right: correct “infeasible” answers on infeasible instances.

Observation 1. Test-time reasoning outperforms scaling: reasoning configurations reach 80.3–98.7% EM, versus at most 30.9% for direct answers. Seven of eleven configurations fall below the 25.6% always-“infeasible” baseline.

## 5.2 RQ2: Performance across difficulty levels and failure modes

Difficulty affects planning more than infeasibility detection. Figure 4 shows that plan accuracy declines sharply from easy to hard instances: GPT-6 Luna drops from 96.0% to 69.0%, and Qwen3.5- 27B with thinking from 91.6% to 59.6%. Infeasibility detection remains more stable. GPT-6.1 Sol’s Flag score falls from 100.0% to 99.0%, while Qwen3.5-27B with thinking declines from 96.0% to 90.3%. These results suggest that constructing a valid plan under more interacting constraints is harder than identifying an infeasible request. The gap is particularly clear for Qwen3.5-27B without thinking: Plan accuracy falls from 37.2% to 7.4%, while Flag rises from 54.0% to 67.0%. Better detection of infeasible cases therefore does not imply better plan construction.

Constructed answers are the hardest. Models perform better when choosing a label than when constructing a structured answer (Appendix Table 7). Qwen3.5-27B with thinking achieves 99.0% accuracy on classification but only 68.6% on schedules, its weakest answer type. Models without thinking show the same pattern at lower accuracy levels. Results by task family are consistent (Appendix Table 6): single-label eligibility decisions (F01) are easiest for every model except GPT-6.1 Sol, which performs near the ceiling. Procurement (F08), sequencing (F06), and appointments (F03) are among the hardest families, requiring models to select items within a budget, respect dependencies, or coordinate timed bookings. In contrast, language register has little effect (Appendix Table 8): GPT-6 Luna ranges from 82.5% (code-mixed) to 87.0% (formal), and Qwen3.5-27B with thinking stays between 79.0% and 81.0%. Difficulty comes from what the answer requires, not the presentation.

Table 3: Bangla prompts vs. their English translations (same instances, same scoring) English improves Qwen EM but yields limited gains in plan accuracy; GPT-6.1 Sol performs better in Bangla.
<table><tr><td></td><td colspan="2">EM↑</td><td colspan="2">Plan↑</td><td colspan="2">Flag↑</td></tr><tr><td>Model</td><td>bn</td><td>en</td><td>bn</td><td>en</td><td>bn</td><td>en</td></tr><tr><td>GPT-6.1 Sol</td><td>98.7</td><td>97.4</td><td>98.4</td><td>96.6</td><td>99.6</td><td>99.6</td></tr><tr><td>Qwen3.5-2B</td><td>3.3</td><td>5.8</td><td>4.2</td><td>6.6</td><td>0.8</td><td>3.5</td></tr><tr><td>Qwen3.5-4B</td><td>14.1</td><td>22.4</td><td>8.3</td><td>8.6</td><td>30.9</td><td>62.5</td></tr><tr><td>Qwen3.5-9B</td><td>19.6</td><td>24.3</td><td>12.8</td><td>15.2</td><td>39.5</td><td>50.8</td></tr></table>

Format errors limit performance. Gemma 3 fails to follow the required format on 70.3% of instances. Even among correctly formatted outputs, its accuracy is only 28.6%. For comparison, Qwen3.5-27B, the strongest open model without thinking, achieves 30.9% accuracy across all instances. Qwen3.5-2B produces format errors on 24.8% of instances, rising from 14.0% on easy problems to 32.0% on hard problems, where answers are longer.

Observation 2. Hard instances break plan construction, not infeasibility detection. Answer kinds that must be constructed (schedules, selections, ordered steps) are the hardest.

## 5.3 Bangla vs. English prompts

Bangla -> English translation does not consistently improve performance. Table 3 shows that English translation does not consistently improve performance. GPT-6.1 Sol performs slightly better in Bangla, achieving 98.7% EM compared with 97.4% in English. The smaller Qwen models gain 2.5, 8.3, and 4.7 EM points at 2B, 4B, and 9B, respectively, but the score decomposition shows limited improvement in plan construction.

Language gains mainly reflect formatting and infeasibility responses. Qwen3.5-4B illustrates this distinction. Plan accuracy changes little, from 8.3% in Bangla to 8.6% in English, while “infeasible” responses become much more frequent: Flag rises from 30.9% to 62.5%, and incorrect Abstention rises from 16.5% to 48.3%. English therefore improves infeasibility detection at the cost of rejecting more feasible requests. For Qwen3.5-2B, English roughly halves format errors, from 24.8% to 12.0%. Overall, language gains mainly reflect better formatting and more infeasibility responses; small models still struggle to construct plans in both languages.

Table 4: Fine-tuning Qwen3.5-2B on gold answers (%), scored on the 166 held-out instances, 42 of them infeasible, so always answering “infeasible” scores 25.3.
<table><tr><td></td><td colspan="2">Feasible</td><td colspan="2">Infeasible</td></tr><tr><td></td><td>EM↑ Plan↑ Abstention</td><td></td><td></td><td>Flag↑ Format</td></tr><tr><td>Always “infeasible&quot;</td><td>25.3</td><td>0.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Qwen3.5-2B, untrained</td><td>2.4</td><td>3.2</td><td>0.0</td><td>0.0</td><td>18.7</td></tr><tr><td>+ SFT (gold answers)</td><td>27.1</td><td>4.0</td><td>89.5</td><td>95.2</td><td>0.0</td></tr></table>

## 5.4 Fine-tuning improves format but offers little planning gain

Fine-tuning Qwen3.5-2B on gold answers substantially improves overall accuracy. Table 4 shows: on the 166 held-out instances, EM rises from 2.4% to 27.1%, format errors fall from 18.7% to zero, and Flag increases from zero to 95.2%. However, Plan accuracy improves only from 3.2% to 4.0%. The model answers “infeasible” on 90.9% of instances, although only 25.3% are infeasible, exceeding the always-“infeasible” baseline by just 1.8 points. This response pattern may reflect the training targets: infeasible instances share one short answer, whereas feasible instances require longer, distinct plans.

## 6 Conclusion

We introduced BANGLAPLAN for evaluating constraint-based planning in Bangla with validated answers and deterministic scoring. GPT-6.1 Sol achieves 98.7% accuracy, while smaller models struggle with planning and formatting. Test-time reasoning raises Qwen3.5-27B’s accuracy from 30.9% to 80.3%. English prompts, Bangla adaptation, and gold-answer fine-tuning reduce format errors but yield limited planning gains. Future work should explore reasoningfocused training, including reinforcement learning with rewards for verified plans, to improve small open models’ reliability in Bangla.

## Limitations

BANGLAPLAN is grounded in Bangladesh and written in its registers of Bangla; its institutions and conventions may not transfer to other Banglaspeaking regions. Each problem is pinned to one canonical answer by stating objectives and tiebreaking rules explicitly. This makes scoring deterministic, but it removes the open-endedness of real requests, and exact match gives no credit to nearly correct plans. Our difficulty labels count binding constraints and supporting facts; we do not validate them against human solving effort. Claude assisted scenario drafting and the final consistency check, so the problems may carry some of its stylistic tendencies, although humans wrote, validated, and adjudicated every item. All models are evaluated zero-shot with one prompt; few-shot examples, other prompt formats, or higher reasoning effort could change the numbers. Finally, our fine-tuning experiment uses one 2B model and one recipe, so the negative result on answer-only supervision is suggestive rather than general.

## References

Thales Sales Almeida, João Guilherme Alves Santos, Thiago Laitz, and Giovana Kerche Bonás. 2025. Ticket-Bench: A kickoff for multilingual and regionalized agent evaluation. arXiv preprint arXiv:2509.14477.

Shimanto Bhowmik, Tawsif Tashwar Dipto, Md Saz zad Islam, Sheryl Hsu, and Tahsin Reasat. 2025. Script fragmentation and format: What drives the English-Bengali performance gap in open LLMs? arXiv preprint arXiv:2507.23248.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901.

Abrar Eyasir, Tahsin Ahmed, and Muhammad Ibrahim. 2026. NCTB-QA: A large-scale Bangla educational question answering dataset and benchmarking performance. arXiv preprint arXiv:2603.05462.

Dhruv Jain, Harshit Shukla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, and Shubham Agarwal. 2025. VoiceAgentBench: Are Voice Assistants Ready for Agentic Tasks? arXiv preprint arXiv:2510.07978.

Saman Sarker Joy and Swakkhar Shatabda. 2026. Bn-MMLU: Measuring massive multitask language understanding in Bengali. In Findings of the Association for Computational Linguistics: ACL 2026, pages 12211–12230. Association for Computational Linguistics.

Gayeon Jung, HyeonSeok Lim, Minjun Kim, Joon-ho Lim, KyungTae Lim, and Hansaem Kim. 2025. Can LLMs truly plan? a comprehensive evaluation of planning capabilities. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 13069–13084. Association for Computational Linguistics.

Vikas Pahuja, Jonathan Brokman, Omer Hofman, Tamir Nizri, Daniel Vishna, Seraphina Goldfarb-Tarrant, Kelly Marchisio, Hisashi Kojima, and Roman Vainshtein. 2026. An actionable diagnosis of multilingual, multi-agent planning failures. arXiv preprint arXiv:2608.03735.

Bidyarthi Paul, Nahida Jannat Mayouree, Md. Asif Karim, Sagar Chandra Nath, and Swastika Kundu. 2026. GSM-Plus-BN: A perturbation-based benchmark for Bangla mathematical reasoning in large language models. arXiv preprint arXiv:2607.13248.

Tabia Tanzin Prama, Christopher M. Danforth, and Peter Dodds. 2025. BanglaMATH: A Bangla Benchmark Dataset for Testing LLM Mathematical Reasoning at Grades 6, 7, and 8. In Proceedings of the 3rd Workshop on Mathematical Natural Language Processing, pages 134–149. Association for Computational Linguistics.

Jinhu Qi, Wentao Zhang, Siu Man Ng, Feiyang Xu, Yanyu Chen, Yaoman Li, and Irwin King. 2026. TREK: A travel reasoning and evaluation kit for LLM agents in complex trip planning. arXiv preprint arXiv:2607.26977.

Nishat Raihan and Marcos Zampieri. 2025. TigerLLM - a family of Bangla large language models. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 887–896, Vienna, Austria. Association for Computational Linguistics.

Leonardo Ranaldi, Sherrie Shen, Jushi Kai, and Alexandra Birch. 2026. WorldBench: Culturally grounded benchmark for multilingual agents. arXiv preprint arXiv:2609.01056.

Angelika Romanou, Negar Foroutan, Anna Sotnikova, Sree Harsha Nelaturu, Shivalika Singh, Rishabh Maheshwary, Micol Altomare, Zeming Chen, Mohamed A. Haggag, Snegha A, Alfonso Amayuelas, Azril Hafizi Amirudin, Danylo Boiko, Michael Chang, Jenny Chim, Gal Cohen, Aditya Kumar Dalmia, Abraham Diress, Sharad Duwal, and 38 others. 2025. INCLUDE: Evaluating multilingual language understanding with regional knowledge. In The Thirteenth International Conference on Learning Representations.

Md Rashad Al Hasan Rony, Sudipto Kumar Shaha, Rakib Al Hasan, Sumon Kanti Dey, Amzad Hossain Rafi, Ashraf Hasan Sirajee, and Jens Lehmann. 2024. BanglaQuAD: A Bengali open-domain question answering dataset. arXiv preprint arXiv:2410.10229.

Israfel Salazar, Manuel Fernández Burda, Shayekh Bin Islam, Arshia Soltani Moakhar, Shivalika Singh, Fabian Farestam, Angelika Romanou, Danylo Boiko, Dipika Khullar, Mike Zhang, Dominik Krzeminski, Jekaterina Novikova, Luísa Shimabu-´ coro, Joseph Marvin Imperial, Rishabh Maheshwary, Sharad Duwal, Alfonso Amayuelas, Swati Rajwal, Jebish Purbey, and 26 others. 2026. Kaleidoscope: In-language exams for massively multilingual vision evaluation. In The Fourteenth International Conference on Learning Representations.

Sheikh Shafayat, H M Quamran Hasan, Minhajur Rahman Chowdhury Mahim, Rifki Afina Putri, James Thorne, and Alice Oh. 2024. BEnQA: A question answering benchmark for Bengali and English. In Findings of the Association for Computational Linguistics: ACL 2024, pages 1158–1177. Association for Computational Linguistics.

Jie-Jing Shao, Bo-Wen Zhang, Xiao-Wen Yang, Baizhi Chen, Siyu Han, Pang Jinghao, Wen-Da Wei, Guohao Cai, Zhenhua Dong, Lan-Zhe Guo, and Yu-Feng Li. 2026. Chinatravel: An open-ended travel planning benchmark with compositional constraint validation for language agents. In The Fourteenth International Conference on Learning Representations.

Tanvir Ahmed Sijan, S. M. Golam Rifat, Pankaj Partha, Md. Tanjeed Islam, and Md Musfique Anwar. 2026. BanglaSocialBench: A benchmark for evaluating sociopragmatic and cultural alignment of LLMs in Bangladeshi social interaction. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 4: Student Research Workshop), pages 247–280. Association for Computational Linguistics.

Joseph Tso, Preston Schmittou, Quan Huynh, and Jibran Hutchins. 2026. ConstraintBench: Benchmarking LLM constraint reasoning on direct optimization. arXiv preprint arXiv:2602.22465.

Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2023. PlanBench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In Advances in Neural Information Processing Systems, volume 36, pages 38975–38987.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and

Yu Su. 2024. TravelPlanner: A benchmark for real-world planning with language agents. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 54590–54613. PMLR.

Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, ChenXu Lv, and Junyang Lin. 2026. DeepPlanning: Benchmarking long-horizon agentic planning with verifiable constraints. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7377–7407. Association for Computational Linguistics.

Ziliang Zhao, Zenan Xu, Shuting Wang, Hongjin Qian, Yan Lei, Minda Hu, Zhao Wang, Shihan Dou, Zhicheng Dou, and Pluto Zhou. 2026. Planning-Bench: Generating scalable and verifiable planning data for evaluating and training large language models. arXiv preprint arXiv:2605.20873.

Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, and Denny Zhou. 2024. NATURAL PLAN: Benchmarking LLMs on natural language planning. arXiv preprint arXiv:2406.04520.

## A Prompt Template

Each model receives, for every instance, the following prompt as a single user turn:

![](images/a6eaeb37c151630e03f96782f5db078e1122ff5d510330988d6cb4e192cbbcf9.jpg)  
Figure 5: The evaluation prompt: the instance’s gold\_prompt, sent verbatim.

That is, the instance’s gold\_prompt is the entire input. We add no system prompt, no in-context exemplars, and no chain-of-thought instruction; API models receive it as the sole user message, and open-weight models receive it through their own chat template. The prompt itself is self-contained: it states the scenario, the facts and constraints, an objective with a deterministic tie-break rule where needed, the JSON-array output schema, and the infeasibility marker [{"is\_feasible": false}], so each instance has exactly one correct answer.

## B Investigating Wrong plan Errors

Exact match labels any parsed plan on a feasible instance that differs from the gold plan as Wrong plan. This category may contain three kinds of error: a hard-constraint violation (V), a feasible plan that loses on the stated objective (S), or an optimal plan that differs only on a tie-break or output convention (N). Because the constraints are written in Bangla prose, we manually audited errors from a low-scoring (Qwen3.5-2B) and a high-scoring model (GPT-6.1 Sol). Table 5 presents ten such audited cases.

Qwen3.5-2B. Qwen3.5-2B produced 522 wrong plans across the 1,000 instances. We randomly sampled 50, stratified in proportion to answer type: 15 schedules, 8 routes, 7 ordered-step plans, 6 classifications, 5 selections, 4 allocations, 3 matchings, and 2 rankings. An author checked each model output against the prompt’s hard constraints, objectives, and tie-breaking rules.

The 50 sampled Qwen3.5-2B plans comprise 49 V, 0 S, and 1 N. Common violations include copying 2026-11-11 from the output example instead of using the requested date, scheduling before an office opens, and ignoring stated durations or prerequisites. Some exceed limits substantially: one route costs 3,700 BDT against a 2,100 BDT budget; another plan seats 3,300 candidates from a pool of 3,000. Other errors include selecting excluded options, assigning the same slot twice, and inventing identifiers. The sole N case is a fourpatient triage ranking that satisfies the constraints but swaps the final two patients against the ledgerorder tie-break. In this sample, Wrong plan therefore almost always indicates a constraint violation rather than a near-canonical answer.

GPT-6.1 Sol. GPT-6.1 Sol misses 13 of 1,000 instances, and here the composition is different. Six are wrong plans: three meet the constraints and objective but fail a tie-break, one is feasible but suboptimal, and two violate a hard constraint. In two cases, the model schedules an SMS check on a later working day despite an earlier one being available. In the third, it starts a supply slot at 11:20 although the line is free from 10:00. In the suboptimal plan, an extra sack takes the costlier river route, raising transport cost from 6,925 to 6,960 BDT. In one case, the model starts an ultrasound 40 minutes after a blood test, although the prompt requires a 45-minute wait. In another, the team returns at 20:50, missing the 20:30 deadline. A further miss proposes a plan for an infeasible instance, exceeding its 61,799 BDT budget by one taka while reporting a total of 60,300 BDT.

## C Annotator and Validator Guidelines

## Guideline for Annotators.

• Write prompts directly in natural Bangla using the assigned task family, register, and answer type. Identify scenario rules as fictional.

• State all entities, quantities, constraints, objectives, and tie-breaking rules explicitly. Provide reference dates and calendar-conversion information where needed, and distinguish optional preferences from mandatory constraints.

• For infeasible variants, introduce a binding conflict and verify that no valid plan remains. Claim minimal infeasibility only when relaxing the designated constraint restores feasibility.

• Derive the gold answer using the stated constraints, ordered objectives, and tie-breaking rules. Follow the required JSON schema, identifiers, digits, timestamps, and ordering conventions; omit zero-valued allocations.

• Record the binding constraints and supporting facts needed to solve the problem. Assign difficulty by their total count k: easy $( k \le 1 0 )$ medium $( 1 0 < k \leq 2 0 )$ , or hard $( k > 2 0 )$

Table 5: Ten audited Wrong plan cases. V: hard-constraint violation; S: feasible but worse on the objective; N: optimal but fails a tie-break. Instance IDs omit the BP- prefix. The audit sheets include the relevant rules in Bangla.
<table><tr><td>Model</td><td>Instance</td><td>Kind</td><td>Rule in the prompt</td><td>What the model&#x27;s plan does</td><td>Verdict</td></tr><tr><td>Qwen3.5-2B</td><td>F04-T038-001</td><td>route</td><td>Fare at most 2,100 BDT; patient needs Takes ambulance and oxygen, which the microbus lacks; arrival 3,700 BDT; arrives at 02:05 by 02:00</td><td>microbus for</td><td>V</td></tr><tr><td></td><td>Qwen3.5-2B F03-T025-007 schedule</td><td></td><td>tween villages; no overlaps</td><td>Plan for 9–11 December; 60 minutes be- Dates every action 2026-11-11, copied from the format example; allows no travel time</td><td>V</td></tr><tr><td></td><td>Qwen3.5-2B F02-T015-005</td><td>ordered steps</td><td>tion at least 24 h later, on Sun/Tue</td><td>Hospital slip before application; verifica- Applies before obtaining the slip; verifies im- mediately on a Monday</td><td>V</td></tr><tr><td></td><td>Qwen3.5-2B F01-T002-008 classification</td><td></td><td>International track requires English score Selects the international track ≥ 6.5; applicant has 6.0</td><td></td><td>V</td></tr><tr><td></td><td>Qwen3.5-2B F08-T078-007 selection</td><td></td><td>copier test; bidder 3 is blacklisted; capac- copiers, and buys 75 desk sets and 24 cabi- ity is 40 pieces</td><td>One bidder per lot; bidder 1 fails the Selects all three bidders, including ineligible nets</td><td>V</td></tr><tr><td></td><td>Qwen3.5-2B F10-T097-002 matching</td><td></td><td>Exactly one unit per slot; list each slot Assigns two engines to the same slot and once</td><td>lists it twice</td><td>V</td></tr><tr><td></td><td>Qwen3.5-2B F10-T094-001 ranking</td><td></td><td>Chest pain before crash; crash before breathing; at most one wound patient in two patients relative to ledger order the first call; ledger-order tie-break</td><td>Satisfies the constraints but swaps the final</td><td>N</td></tr><tr><td>GPT-6.1 Sol</td><td>F03-T022-008</td><td>schedule</td><td>Ultrasound at least 45 minutes after blood test ends at 08:20</td><td>Starts ultrasound at 09:00, after only 40 min- utes</td><td>V</td></tr><tr><td>GPT-6.1 Sol</td><td>F07-T070-008 allocation</td><td></td><td>Minimise transport cost; road costs 20 BDT/sack, river 35; satisfy minimums a preference; cost rises from 6,925 to and stock limits</td><td>Moves one extra sack by river to honor 6,960 BDT</td><td>S</td></tr><tr><td></td><td>GPT-6.1 Sol F03-T026-009 schedule</td><td></td><td>rics, within 13–20 Oct, on a working day; sible date, 13 Oct earliest-start tie-break</td><td>SMS check at least two days after biomet- Checks on 15 Oct instead of the earliest fea-</td><td>N</td></tr></table>

## Guideline for Validators.

• Solve the Bangla prompt independently without viewing its gold answer. Check clarity, completeness, feasibility, objectives, tiebreaking, and schema compliance.

• Compare the independently derived answer with the gold answer. For infeasible items, verify that the constraints rule out every valid plan.

• Resolve disagreements with the author and validate revised items again. Treat modelgenerated mismatches and suggestions as review signals; humans determine the final prompt and gold answer.

## D Results by Task Family and Answer Kind

Tables 6–8 report exact match on Bangla instances for all eleven configurations, broken down by task family, answer kind, and language register, respectively.

## E Acknowledgements

AI Assistance Disclosure. We used generative AI tools, including ChatGPT and Claude, for language editing and paraphrasing of author-written text. Claude also suggested scenario details during benchmark drafting and flagged potential inconsistencies for human review. Human annotators selected and revised the scenarios, independently validated the problems, and determined the final gold answers. All content was reviewed and revised by the authors, who take full responsibility for the paper’s correctness and citations.

Table 6: Exact match (%) by task family (bn). Bold = best, underline = second best per row. Procurement (F08), sequencing (F06), and appointments (F03) are hardest, requiring models to select items within a budget, respect dependencies, or coordinate timed bookings.
<table><tr><td>F01 Eligibility</td><td>100</td><td>G  -Lna n 98.0</td><td>GPT 97.0</td><td>Owe-2B So</td><td>Ow-4B</td><td>Oe-9B</td><td>Oe--27B</td><td>() (-  hin)</td><td>12B eemm</td><td>TiL12B</td><td>9B 2 Gema</td><td>8 31 LIamma</td></tr><tr><td>F02 Procedures</td><td>100</td><td>88.0</td><td>100.0</td><td>21.0 0.0</td><td>34.0</td><td>46.0</td><td>64.0</td><td>99.0</td><td>43.0</td><td>45.0 40.0</td><td></td><td>34.0</td></tr><tr><td>F03 Appointments</td><td>100</td><td>78.0</td><td></td><td></td><td>7.0</td><td>10.0</td><td>24.0</td><td>77.0</td><td>0.0</td><td>0.0</td><td>2.0</td><td>2.0</td></tr><tr><td>F04 Journeys</td><td>100</td><td>94.0</td><td>97.0 100.0</td><td>2.0 1.0</td><td>15.0 19.0</td><td>19.0 27.0</td><td>28.0 30.0</td><td>68.0 88.0</td><td>0.0 0.0</td><td>5.0 9.0</td><td>6.0 2.0</td><td>4.0</td></tr><tr><td>F05 Rosters</td><td>100</td><td>80.0</td><td>100.0</td><td>0.0</td><td>17.0</td><td>19.0</td><td>30.0</td><td>77.0</td><td>4.0</td><td>9.0</td><td>7.0</td><td>4.0</td></tr><tr><td></td><td>100</td><td>81.0</td><td>99.0</td><td>0.0</td><td>10.0</td><td>13.0</td><td>23.0</td><td>68.0</td><td>0.0</td><td>2.0</td><td>4.0</td><td>4.0</td></tr><tr><td>F06 Sequencing</td><td>100</td><td>88.0</td><td>99.0</td><td>2.0</td><td>11.0</td><td>16.0</td><td>29.0</td><td>87.0</td><td>15.0</td><td>14.0</td><td>16.0</td><td>0.0</td></tr><tr><td>F07 Allocation</td><td>100</td><td>78.0</td><td>99.0</td><td>0.0</td><td>3.0</td><td>9.0</td><td>10.0</td><td>83.0</td><td>0.0</td><td>1.0</td><td>0.0</td><td>8.0</td></tr><tr><td>F08 Procurement</td><td>100</td><td>79.0</td><td>98.0</td><td>0.0</td><td>5.0</td><td>11.0</td><td>30.0</td><td>76.0</td><td>2.0</td><td>3.0</td><td>1.0</td><td>0.0</td></tr><tr><td>F09 Logistics</td><td>100</td><td>87.0</td><td>98.0</td><td>7.0</td><td>20.0</td><td>26.0</td><td>41.0</td><td>80.0</td><td>21.0</td><td>23.0</td><td></td><td>3.0</td></tr><tr><td>F10 Emergency</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>19.0</td><td>8.0</td></tr></table>

Table 7: Exact match (%) by answer kind (bn). Bold = best, underline = second best per row. GPT-6.1 Sol leads on seven of eight answer kinds, while all other models achieve their highest exact match on classification.
<table><tr><td></td><td></td><td>n</td><td>GP -Lna</td><td>GPT6-60I1 Owe-2B</td><td>Oe-4B</td><td>O-9B</td><td>Owe--7B</td><td>() (-  in)</td><td>Gmm</td><td>12B TI2B</td><td>Gema</td><td>9B 8 1a &#x27;1</td></tr><tr><td>allocation classification</td><td>160 100</td><td>88.8 98.0</td><td>99.4 97.0</td><td>1.2 21.0</td><td>10.0 34.0</td><td>13.8 46.0</td><td>28.7 64.0</td><td>84.4 99.0</td><td>8.1 43.0</td><td>10.0 45.0</td><td>10.6 40.0</td><td>1.2</td></tr><tr><td>matching</td><td>60</td><td>95.0</td><td>100.0</td><td>5.0</td><td>25.0</td><td>26.7</td><td>43.3</td><td>93.3</td><td>30.0</td><td>30.0</td><td>28.3</td><td>34.0 20.0</td></tr><tr><td></td><td>140</td><td>88.6</td><td>100.0</td><td>0.0</td><td>6.4</td><td>8.6</td><td>23.6</td><td>74.3</td><td>0.0</td><td>0.0</td><td>1.4</td><td>1.4</td></tr><tr><td>ordered steps</td><td>40</td><td>80.0</td><td>95.0</td><td>10.0</td><td>22.5</td><td>32.5</td><td>40.0</td><td>72.5</td><td>27.5</td><td>20.0</td><td>12.5</td><td></td></tr><tr><td>ranking</td><td>100</td><td>94.0</td><td>100.0</td><td>1.0</td><td>19.0</td><td>27.0</td><td>30.0</td><td>88.0</td><td>0.0</td><td>9.0</td><td>2.0</td><td>10.0 4.0</td></tr><tr><td>route</td><td>290</td><td>76.2</td><td>97.9</td><td>0.7</td><td>12.1</td><td>17.2</td><td>28.6</td><td>68.6</td><td>0.0</td><td>4.8</td><td>4.8</td><td>3.1</td></tr><tr><td>schedule</td><td>110</td><td>75.5</td><td>99.1</td><td>0.0</td><td>3.6</td><td>9.1</td><td>10.0</td><td>84.5</td><td>0.0</td><td>0.9</td><td>0.0</td><td></td></tr><tr><td>selection</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0</td></tr></table>

Table 8: Exact match (%) by language register (bn). Bold = best, underline = second best per row. GPT-6.1 So leads on every register. Among the three strongest models, GPT-6 Luna varies most, scoring lowest on code-mixed prompts (82.5%)
<table><tr><td>code-mixed</td><td>n 200</td><td>GP -Lna 82.5</td><td>GP16-I 99.0</td><td>Owe-2B</td><td>Ow-4B</td><td>Oe-9B</td><td>Owe--7B</td><td>(w) (-in)</td><td>G3 m12B</td><td>TiL12B</td><td>Gm9B</td><td>L1a 8B</td></tr><tr><td>colloquial</td><td>100</td><td>87.0</td><td>100.0</td><td>4.0</td><td>17.0</td><td>21.0</td><td>29.0</td><td>79.0</td><td>9.0</td><td>12.0</td><td>11.0</td><td>8.5</td></tr><tr><td></td><td>500</td><td>85.0</td><td></td><td>1.0</td><td>18.0</td><td>25.0</td><td>29.0</td><td>81.0</td><td>7.0</td><td>8.0</td><td>9.0</td><td>6.0</td></tr><tr><td>everyday</td><td></td><td></td><td>98.8</td><td>4.0</td><td>12.4</td><td>18.8</td><td>31.0</td><td>80.8</td><td>9.8</td><td>12.4</td><td>9.8</td><td>6.2</td></tr><tr><td>formal</td><td>200</td><td>87.0</td><td>97.5</td><td>2.0</td><td>13.5</td><td>17.5</td><td>33.5</td><td>80.0</td><td>5.5</td><td>8.5</td><td>8.5</td><td>6.5</td></tr></table>