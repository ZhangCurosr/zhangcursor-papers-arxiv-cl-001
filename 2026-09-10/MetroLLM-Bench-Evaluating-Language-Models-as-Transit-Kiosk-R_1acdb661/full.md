# MetroLLM-Bench: Evaluating Language Models as Transit Kiosk Runtimes

Remco Hendriks

Continker

remco.hendriks@continker.ai

## Abstract

We introduce MetroLLM-Bench, a 955-case benchmark for testing language models as the policy layer of a transit kiosk. It covers six real metro systems, ranging from 37 to 414 stations, and eleven categories that include routing, fare calculation, disruptions, accessibility, and adversarial input. In each case, the model must call structured tools and submit a machine-renderable terminal state containing an outcome, a per-ticket fare quote when applicable, and a kiosk action. Fourteen deterministic scoring components form Tier 1; eight semantic-quality components form Tier 2, six of which use a language-model judge. We report Tier 1 and the combined score of both tiers. A stratified 75/25 split reserves 717 cases for training-data generation and 238 for held-out evaluation.

We evaluate twenty-six models from six vendors, of which twenty-three are ranked. On the held-out partition, a 4B Qwen 3.5 student trained through parameter-eficient fine-tuning (PEFT) exceeds both GPT-5.6 tiers on Tier 1 (91.3 against 90.6 and 90.0) and matches GPT-5.4 full at maximum reasoning efort (91.4), with a 2.6 GB Q4\_K\_M footprint. Larger 9B and 27B students provide no further Tier 1 improvement over the 4B student at this training scale. Across the four Qwen sizes, the PEFT gain over the corresponding base model decreases from +7.03 points at 2B (three training seeds) to −0.91 at 27B; every seed shows the same direction at every size. A deterministic rule-based baseline reaches 84.6 on Tier 1, with the remaining language-model advantage concentrated in policy adaptation, compound scenarios, accessibility, and temporal reasoning. Muse Glimmer 30B leads the composite ranking, and serving configuration alone moves the Qwen 3.5-to-3.8 comparison by 2.7 Tier 1 points. The benchmark, harness, reproduction guide, and fine-tuned students are released at https://github.com/continker/metrollm-bench.

## 1 Introduction

Transit kiosks encode fare rules, route topology, and disruption responses as programmed state machines. When an operator wants to reflect a station closure, a new fare bracket, or a holiday schedule, the change typically passes through a code deployment cycle: a developer ticket, an integrator patch, regression tests, and a software release window. We evaluate an alternative in which the kiosk’s policy logic is replaced by a language model that reads a natural-language system description (a framebook), calls structured tools, and emits a renderable terminal state that the kiosk hardware can act on.

The transit kiosk is a useful testbed for this question. The output must be correct (a wrong fare is a billing error), renderable (the kiosk has fixed display slots), adaptable (rules change), and auditable (operators need to explain what the kiosk did). The interaction is short and goal-directed, which keeps token cost bounded. And the operational rules change often enough that the cost of code-defined logic is concrete.

Prior LLM-powered transit work has explored trip planning (Wang & Shalaby, 2024) and passenger travelchoice prediction under train delays (Chen et al., 2024). A GTFS-comprehension benchmark (Devunuri et al., 2024) evaluates whether models understand transit-data semantics, but not whether they can make operational decisions. General-purpose agent benchmarks such as τ-bench (Yao et al., 2024) and GAIA (Mialon et al., 2023) cover a much broader range of domains. They report partial completion under binary pass-or-fail scoring: GPT-4o completes 61 percent of τ-bench retail tasks and 35 percent of its airline tasks in a single attempt, and GPT-4 with plugins answers 15 percent of GAIA questions, 30 percent at its easiest level. Work on declarative chatbots (Sánchez Cuadrado et al., 2024) and prompt-compiled policy classifiers (Kholkar & Ahuja, 2025) establishes that prompt-driven runtimes can support narrow tasks. Whether that holds under disruption, multi-turn context, and adversarial input is the open question; at a passenger-facing kiosk, all three are routine.

MetroLLM-Bench makes two methodological contributions. The first is the benchmark itself: 955 cases across six metro systems, scored so that the deterministic components, clean enough to double as a fine-tuning reward, stay separate from the broader semantic-quality tier. The second is measurement discipline: a system-stratified train/held-out split fixed before any training, and a calibration of the deployed scoring stack against blind ratings from two independent human annotators: judge–author agreement (quadratic-weighted Cohen’s $\kappa _ { w } = 0 . 5 3$ , moderate under Landis & Koch (1977)) exceeds the agreement between the two raters themselves $\left( \kappa _ { w } = 0 . 2 5 \right)$ ).

The empirical contribution is a twenty-three-model leaderboard and a four-size PEFT sweep at two to three independent training seeds. The sweep reveals a monotonic decline in PEFT utility as base capability increases, ending in a negative delta at 27B. A scripted agent that only chains the tools scores within a few points of the models on routing and fare arithmetic, so the language model’s advantage lies in the categories that require a decision: temporal reasoning, policy changes, accessibility, and compound scenarios. The central deployment result: a 4B student that fits in 2.6 GB exceeds both GPT-5.6 tiers and matches GPT-5.4 full at maximum reasoning efort on held-out Tier 1, and nothing we trained above 4B measurably improves on it.

## 2 Benchmark Design

![](images/b1def59b1315e9745167be38b9583e5b564c35b48047ed54a705d91ea0b9528b.jpg)  
Figure 1: End-to-end kiosk loop. (1) The framebook (terminology, currency, operating hours) and the case events (origin, destination, passenger count, optional disruption, optional freetext) feed (2) a system-prompt assembler. (3) A ReAct loop of up to 20 rounds calls the six tools and terminates by (4) submit\_assistant\_state, which emits (5) the renderable terminal state (outcome, per-ticket fare quote, advisory banners, kiosk action). The tool cards carry display names for route\_planner, fare\_calculator, station\_info, line\_info, disruption\_feed, and knowledge\_base.

Figure 1 shows the unit of evaluation; case BART-C-006 illustrates a pass through it. A passenger requests travel from 12th St Oakland to Embarcadero during a Transbay Tube closure for seismic work. The BART framebook and the case events (1) are assembled into the prompt (2). Inside the loop (3), the model reads the closure from disruption\_feed, re-plans with route\_planner to a route ending at West Oakland, and prices it with fare\_calculator. It then submits (4) an advisory\_only terminal state (5) directing the passenger to the free AC Transit bus bridge. The scorer evaluates the tool sequence and the passenger-facing state.

## 2.1 Systems and cases

The six systems were chosen to vary along dimensions that a kiosk policy layer must absorb. They cover three fare models (flat, distance-based, and flat with exceptions), four currencies, and networks ranging from 37 to 414 stations. The systems are MARTA (Atlanta, 38 stations), Doha Metro (37), BART (San Francisco, 50), Taipei MRT (107), CTA (Chicago, 142), and Beijing Subway (414).

Case selection within those systems was manually curated to vary both task structure and regional operating context. Alongside routine routing and fare requests, the set includes locally salient disruptions and policies, such as earthquakes, severe winter weather, and system-specific cultural rules. The US systems provide comparatively well-documented contexts that may be familiar to many readers; Doha, Beijing, and Taipei add diferent fare conventions, terminology, languages, and operating patterns. Their representation in model pretraining is unknown and is not estimated here.

Each system has a framebook that specifies terminology, currency, operating hours, and cultural conventions. The framebook is inserted into the system prompt at runtime, allowing the same model to operate under six rule sets without code changes. The set includes both Latin and non-Latin station names. The prompt and tool results are designed to provide every operational fact required by a case, with one documented exception (Appendix F). The benchmark therefore tests whether a model can follow supplied rules, rather than recall a network from pretraining. The per-system results in Appendix F are consistent with that design.

The benchmark contains 955 cases in eleven categories: A Routing, B Fare, C Disruption, D Accessibility, E Cultural, F Policy, G Multi-turn, H Adversarial, I Temporal, J Tool-Hallucination, and K Compound Stress. The cases are distributed as BART 157, Beijing 162, CTA 157, Doha 156, MARTA 156, and Taipei 167.

The cases are designed benchmark scenarios rather than an attempt to exhaust real-world transit operations. Category-specific templates are combined with per-system station-pair and disruption metadata; cases/generator.py then uses the benchmark graph and fare engines to derive route and fare ground truth and emits the event stream, runtime context, expected fields, and scoring configuration. Generation-time tests check required fields, unique identifiers, station references, graph-valid paths, exact fare consistency for Category B, and category-specific invariants. The committed case files form the fixed evaluation set used for all reported runs. An independent annotator additionally validated a stratified 50-case sample of the committed answer key against the underlying network and fare data (Appendix B.7).

## 2.2 Interaction and terminal state

For each case, the framebook and scenario events are assembled into a prompt. The model then enters a ReAct-style (Yao et al., 2023) loop with native function calling and a budget of twenty tool rounds. It can call the six tools in Figure 1 before ending the case with submit\_assistant\_state. Family-specific runtime settings are reported in Section 3 and Appendix B.

The terminal tool defines the kiosk’s render contract. Every submission must include one of five outcomes: route\_and\_fare\_ready, advisory\_only, service\_unavailable, request\_declined, or policy\_answer\_only. It must also include a kiosk action with a reason code and a passenger-facing message. Route fields are required for routable outcomes; a fare quote is required when a route and fare are ready. The quote contains a passenger summary and per-ticket line items.

The mock server validates this structure with Pydantic. If the submission is inconsistent, it returns an HTTP 422 response with field-level errors. The runner gives those errors back to the model, which can correct its state within the remaining round budget.

## 2.3 Scoring and held-out evaluation

The scorer evaluates twenty-two components in two tiers:

• Tier 1 contains fourteen deterministic components: route and fare correctness, tool-call accuracy and no-hallucination, renderable-state validity, outcome and reason-code correctness, fare breakdown, passenger summary, purchase gate, disruption detection, advisory issuance, context-update detection, re-planning eficiency, and a keyword-presence check for cultural references. These components are computed without a model judge and also serve as the PEFT reward signal.

• Tier 2 contains eight semantic-quality components: framebook conformance, advisory content, policy acknowledgement, safety, accessibility, temporal accuracy, no-data-fabrication, and scope adherence. Six use Anthropic’s Claude Haiku 4.5, sometimes alongside structural checks: advisory content, policy acknowledgement, safety, temporal accuracy, no-data-fabrication, and scope adherence. Framebook conformance and accessibility accuracy are scored programmatically; temporal accuracy also includes a structural subscore. Every language-model judgment is cached to disk.

The composite score is the percentage of available points earned across both tiers. Table 2 and Table 4 report the unweighted mean of the six per-system means; the per-category figures pool cases within category, and the bootstrap comparisons average per-case scores directly, so the same diference can vary by a few hundredths of a point between tables.

The four fine-tuned models require a partition fixed before training. A system-stratified 75/25 split (seed=42) assigns 717 cases to training-data generation and 238 to held-out evaluation. Fifteen gap-audit cases, added after the training set was frozen, are pinned to the held-out partition. The other 223 held-out cases are drawn at random within systems, producing per-system held-out fractions between 24.7 and 25.2 percent. All PEFT training data comes from the 717-case training partition, and every fine-tuning comparison uses the 238-case held-out partition as its primary evaluation set. The split specification is committed to the repository.

Some held-out cases still share structural templates with training cases. As one proxy for this overlap, we count training-set neighbours with the same origin-destination pair. Among the 149 held-out cases for which such a pair is defined, 59 (40 percent) have no training-set neighbour with the same pair; the median is one neighbour and the 90th percentile is thirty-one.

The held-out partition is the primary generalisation evaluation. We also report results on the full 955-case matrix as a secondary precision and sensitivity analysis. Because that matrix includes the 717 cases used for training-data generation, it is not independent held-out evidence; we use it to test whether the observed direction persists with lower case-sampling variance.

## 2.4 Scoring-stack calibration

Six of the eight Tier 2 components use a language-model judge, so we compare the deployed scoring stack with human ratings. The calibration set holds 100 case-rubric pairs, one from each of 100 cases spanning all six systems and ten of the eleven categories; the rated responses are GPT-5-mini outputs from a single benchmark run. The sample is stratified across six Haiku-using Tier 2 rubrics and the deterministic Tier 1 cultural\_accuracy check, with fourteen or fifteen pairs per rubric, and is enriched for non-full-credit automated outputs. Deployed scores are mapped to a common ordinal $0 / 1 / 2$ scale. Two annotators rated al 100 pairs independently: the author, with each automated score revealed only after the rating was locked, and a second independent annotator who saw no judge output at any point.

Table 1 reports the three pairings; judge–author agreement $( \kappa _ { w } = 0 . 5 3$ , moderate under Landis & Koch (1977)) exceeds the agreement between the two raters themselves $\left( \kappa _ { w } = 0 . 2 5 \right)$ : on this task the judge disagrees with the author no more than the two humans disagree with each other. The near-zero $\kappa _ { w }$ for the second annotator is a prevalence artefact (Feinstein & Cicchetti, 1990): that rater awarded the top score on 89 of 100 pairs (author 78, judge 73), and chance-corrected κ degenerates under skewed marginals even when raw agreement remains high. Gwet’s AC2 (Gwet, 2008; 2014), computed with the same quadratic weights and robust to prevalence, places all three pairings between 0.81 and 0.89.

Table 1: Agreement between the deployed scoring stack and two human annotators on 100 case-rubric pairs, on the common $0 / 1 / 2$ scale. $\kappa _ { w }$ is quadratic-weighted Cohen’s kappa with its 95 percent bootstrap interval; Gwet’s AC2, computed with the same quadratic weights (Gwet, 2008; 2014), is robust to skewed marginals.
<table><tr><td>pairing</td><td>exact</td><td>within one point</td><td> $\kappa _ { w }$ </td><td>(quadratic)</td><td>Gwet AC2</td></tr><tr><td>author vs. judge</td><td>82%</td><td>97%</td><td></td><td>0.53 [0.27, 0.75]</td><td>0.89</td></tr><tr><td>second annotator vs. judge</td><td>69%</td><td>93%</td><td></td><td>0.02 [−0.12, 0.18]</td><td>0.81</td></tr><tr><td>author vs. second annotator</td><td>76%</td><td>97%</td><td></td><td>0.25 [0.02, 0.46]</td><td>0.89</td></tr></table>

For context, MT-Bench reports 81 percent human-to-human agreement and 85 percent agreement for its GPT-4 judge on non-tie pairwise votes (Zheng et al., 2023). That comparison is only indicative because MT-Bench measures pairwise preference, whereas MetroLLM-Bench uses an ordinal $0 / 1 / 2$ rubric.

Three adversarial scenic-route cases account for the clearest substantive disagreement. In the H-014 cases for TRTC, MARTA, and CTA, the judge penalises an agent that ofers a scenic route despite a system-prompt prohibition. Both annotators instead credit the agent for returning a valid route to the requested destination; the second annotator, rating blind, made the same call on all three cases, so the split is systematic rather than particular to one rater. The disagreement concerns the definition of safety\_response\_quality: strict constraint adherence versus outcome utility. Both interpretations are familiar from work on instruction hierarchy and sycophancy (Sharma et al., 2023; Wallace et al., 2024; Geng et al., 2026).

Class imbalance also reduces κ (Feinstein & Cicchetti, 1990). On four rubrics, 13 to 15 of the 14 or 15 cases receive the maximum score, so agreement can be high even when per-rubric κ approaches zero. The class-balanced rubrics reach $\kappa _ { w } \ = \ 0 . 6 9$ for policy\_acknowledged and 0.68 for scope\_adherence; safety\_response\_quality reaches 0.20. Cross-judge calibration with a non-Anthropic model remains future work. No headline claim is based on Tier 2 alone: the central deployment comparison uses deterministic Tier 1, while the composite score is a secondary measure that necessarily includes Tier 2.

## 3 Evaluation

## 3.1 Models and ranking

We evaluate twenty-six models from six vendors on all six systems. The local matrix, run on one NVIDIA RTX 5090 through llama.cpp at Q4 to Q8 GGUF quantisation, covers the Qwen 3.5 (Yang et al., 2025a; Qwen Team, 2026a) base models from 0.8B to 35B-A3B, four Qwen PEFT students (2B, 4B, 9B, and 27B), the Qwen3.6-27B and Qwen3.8-27B dense models (Qwen Team, 2026b;c), Muse Glimmer 30B (Meta Superintelligence Labs, 2026), GLM-4.7-Flash (Z.ai, 2026), Llama 3.1-8B (Llama Team, AI@Meta, 2024), and three Gemma 4 (Google DeepMind, 2026) variants (E2B, E4B, and 26B-A4B). The Mistral models (Mistral AI, 2026) use the public Mistral API. The OpenAI rows use Azure OpenAI: GPT-5.6 sol and luna (OpenAI, 2026) at xhigh and medium reasoning efort, and GPT-5.4 nano, mini, and full (OpenAI, 2025b;a), the last at medium, high, and xhigh efort.

Each PEFT student is trained from traces generated only on the 717-case training partition. We retain 600 deduplicated examples for which a base 27B or 35B teacher achieves at least 90 percent on Tier 1, then train with QLoRA rank 16 for three epochs. Every student is trained at seeds 42 and 43; the 2B student, which shows the largest seed sensitivity, is additionally trained at seed 44. Table 2 reports per-size seed means; Table 4 gives the held-out per-seed scores, Appendix D gives the bootstrap intervals, and Appendix G describes the released artefacts. Across all seeds, training requires 9.4 GPU-hours; individual runs take 27 minutes at 2B and 103 minutes at 27B.

Twenty-three models are ranked. We exclude the Gemma 4 E2B and E4B variants because 28 to 33 percent of their cases exhaust the twenty-round budget without a valid submit\_assistant\_state. Those failures create floor-efect zeros that are not comparable with models that terminate normally on at least 98 percent of cases. Gemma 4 26B-A4B terminates normally and remains in the ranking. Llama 3.1-8B is excluded on the same basis. Appendix F reports the raw Gemma and Llama scores.

Figure 2 and Table 2 show no clear break among the leading models: the top eleven rows span 3.18 composite points, and no adjacent gap exceeds 0.68 points. Muse Glimmer 30B ranks first on the composite, with Qwen3.8-27B and Qwen3.6-27B ahead of Qwen3.5-27B base; on the deterministic tier the order difers, with Qwen3.6-27B leading Tier 1 at 93.63. GPT-5.6 luna, OpenAI’s budget tier, ranks fifth, one place above GPT-5.4 full at xhigh efort; GPT-5.6 sol at xhigh efort ranks eighth. The three leading rows were served at per-model configurations, and Section 3.4 measures how much of such diferences configuration alone can account for. Among the six highest-ranked rows, only the two OpenAI rows are proprietary.

![](images/9408ce160ee88e945d55d18c4e44831b18a9a73145fef55f293f485dfe1ea398.jpg)  
Figure 2: Held-out composite for the twenty-five ranked rows (twenty-three distinct models; GPT-5.4 full at three efort levels), coloured by family. Panel (a) holds the twenty-two rows above the rule-based baseline (dotted line, 77.1); panel (b) holds the three rows below it on its own scale. Footprint labels mark the deployment-relevant rows.

The most relevant comparison for local deployment is the 4B PEFT student. Its 2.6 GB Q4\_K\_M build scores 91.32 on Tier 1, above both GPT-5.6 rows (90.63 for luna at medium efort, 90.00 for sol at xhigh) and within 0.05 points of GPT-5.4 full at xhigh efort (91.37), which at medium efort scores 89.17. The student gains 2.00 points over the 4B base model. The 0.05-point diference from GPT-5.4 xhigh is well inside the paired-bootstrap interval reported in Appendix D. On this bounded task, the result is operationally decisive: frontier-level Tier 1 performance does not require a proprietary frontier API. A 2.6 GB, open-weight Qwen 3.5 4B model adapted with PEFT can deliver it within an operator-controlled deployment stack. This is a parity claim on deterministic task performance, not a claim of superiority on the composite score or across every category.

Within the GPT-5.4 family, reasoning efort moves scores more than model size does. Moving the full mode from medium to xhigh raises its composite score by 2.73 points; moving from high to xhigh raises it by 2.25. At medium efort, full, mini, and nano span only 1.35 points, close to the single-run interval discussed in Section 6. The GPT-5.6 pair does not repeat this pattern: sol at xhigh efort trails luna at medium by 0.75 composite points. The two rows difer in both model tier and efort level, so we draw no mechanism from the gap; the return on additional reasoning efort is model-specific, as Section 3.4 also finds for the local rows. The Qwen base models cover a much wider range, from a composite score of 59.98 at 0.8B to 90.60 at 27B.

The Qwen progression contains one large step: from 2B to 4B, the base model gains 15.15 points on Tier 1 and 15.35 composite points on the held-out partition. The Gemma E2B and E4B variants show the same direction of change, from composite scores of 42.8 to 66.8, but neither appears in the main ranking. The available Mistral models do not provide a clean comparison across this size range, and the OpenAI models do not expose comparable active-parameter counts. Qwen 35B-A3B further complicates any cross-family interpretation: it has 3B active parameters yet ranks seventh by held-out composite score. The observed 2B-to-4B gap is therefore a within-Qwen result, not a general parameter threshold.

Table 2: Held-out leaderboard (n=238). PEFT rows report the mean over independent training seeds, three at 2B and two at the other sizes. GPT-5.6 rows are marked <sup>5</sup>, sol at xhigh efort and luna at medium; GPT-5.4 nano and mini are shown at medium efort and full at all three efort levels. Rows marked <sup>3</sup> were served at per-model configurations (Appendix B.4).
<table><tr><td>Rank</td><td>Model</td><td>License</td><td>Vendor</td><td>Comp.</td><td>Tier 1</td></tr><tr><td>1</td><td>Muse Glimmer 30B 3</td><td>Apache 2.0</td><td>Meta</td><td>92.03</td><td>92.75</td></tr><tr><td>2</td><td>Qwen3.8-27B 3</td><td>Apache 2.0</td><td>Alibaba</td><td>91.83</td><td>92.20</td></tr><tr><td>3</td><td>Qwen3.6-27B 3</td><td>Apache 2.0</td><td>Alibaba</td><td>91.28</td><td>93.63</td></tr><tr><td>4</td><td>Qwen3.5-27B base</td><td>Apache 2.0</td><td>Alibaba</td><td>90.60</td><td>92.32</td></tr><tr><td>5</td><td>ĠPT-5.6 luna (medium) 5</td><td>Proprietary</td><td>OpenAI</td><td>90.57</td><td>90.63</td></tr><tr><td>6</td><td>GPT-5.4 full (xhigh)</td><td>Proprietary</td><td>OpenAI</td><td>90.45</td><td>91.37</td></tr><tr><td>7</td><td>Qwen3.5-35B-A3B base</td><td>Apache 2.0</td><td>Alibaba</td><td>89.90</td><td>92.12</td></tr><tr><td>8</td><td>GPT-5.6 sol (xhigh)</td><td>Proprietary</td><td>OpenAI</td><td>89.82</td><td>90.00</td></tr><tr><td>9</td><td> $\mathrm { Q w e n 3 . 5 - 2 7 B + P E F T \ ( n { = } 2 ) }$ </td><td>Apache 2.0</td><td>Alibaba</td><td>89.72</td><td>91.41</td></tr><tr><td>10</td><td>Qwen3.5-4B + PEFT (n=2)</td><td>Apache 2.0</td><td>Alibaba</td><td>89.12</td><td>91.32</td></tr><tr><td>11</td><td>Qwen3.5-9B + PEFT (n=2)</td><td>Apache 2.0</td><td>Alibaba</td><td>88.85</td><td>91.03</td></tr><tr><td>12</td><td>GPT-5.4 full (high)</td><td>Proprietary</td><td>OpenAI</td><td>88.20</td><td>89.48</td></tr><tr><td>13</td><td>Qwen3.5-9B base</td><td>Apache 2.0</td><td>Alibaba</td><td>88.05</td><td>89.38</td></tr><tr><td>14</td><td>Mistral Small 26031</td><td>Mistral RL 2</td><td>Mistral</td><td>87.82</td><td>90.45</td></tr><tr><td>15</td><td>GPT-5.4 full (medium)</td><td>Proprietary</td><td>OpenAI</td><td>87.72</td><td>89.17</td></tr><tr><td>16</td><td>Qwen3.5-4B base</td><td>Apache 2.0</td><td>Alibaba</td><td>87.25</td><td>89.32</td></tr><tr><td>17</td><td>GLM-4.7-Flash 3 4</td><td>MIT</td><td>Z.ai</td><td>86.73</td><td>89.67</td></tr><tr><td>18</td><td>GPT-5.4-nano</td><td>Proprietary</td><td>OpenAI</td><td>86.58</td><td>87.10</td></tr><tr><td>19</td><td>GPT-5.4-mini</td><td>Proprietary</td><td>OpenAI</td><td>86.37</td><td>87.57</td></tr><tr><td>20</td><td>Ministral 8B 2512</td><td>Mistral RL 2</td><td>Mistral</td><td>85.68</td><td>87.33</td></tr><tr><td>21</td><td>Gemma 4 26B-A4B</td><td>Gemma Terms</td><td>Google</td><td>81.98</td><td>85.23</td></tr><tr><td>22</td><td> $\mathrm { Q w e n 3 . 5 - 2 B + P E F T \ ( n { = } 3 ) }$ </td><td>Apache 2.0</td><td>Alibaba</td><td>79.69</td><td>81.20</td></tr><tr><td>23</td><td>Qwen3.5-2B base</td><td>Apache 2.0</td><td>Alibaba</td><td>71.90</td><td>74.17</td></tr><tr><td>24</td><td>Qwen3.5-0.8B base</td><td>Apache 2.0</td><td>Alibaba</td><td>59.98</td><td>61.93</td></tr><tr><td>25</td><td>Mistral Nemo 12B</td><td>Apache 2.0</td><td>Mistral</td><td>56.67</td><td>57.50</td></tr></table>

<sup>1</sup> Mistral Small 2603 is a 119B-parameter MoE with 6.5B active per token.  
<sup>2</sup> Mistral Research License, non-commercial use only.  
<sup>3</sup> Served at per-model configurations on llama.cpp b10398: Qwen3.8 with vendor-recommended sampling (temperature 1.0,  
single run), Qwen3.6 with a 16384-token output budget, Muse and GLM at the 4096-token budget used for the other rows. Appendix B.4 details all rows.  
<sup>4</sup> GLM-4.7-Flash is a 30B-parameter MoE with 3B active per token.  
<sup>5</sup> Azure OpenAI via the Responses API, temperature 1.0, single run. Appendix B.4 details the configuration.

## 3.2 Results by category

Figure 3 breaks the held-out score down by category for the models that enter the deployment comparison: the two Qwen 3.5 teachers, the three students from 4B upward, and the two OpenAI rows evaluated at xhigh efort. No model dominates the category breakdown. Against GPT-5.4 full at xhigh efort, Qwen 27B base leads in Fare, Disruption, Accessibility, Cultural, and Policy. It trails in Routing, Adversarial, Temporal, and Compound Stress. The Multi-turn and Tool-Hallucination scores difer by less than one point.

The largest diferences point in opposite directions. Qwen 27B base leads Accessibility by 13.5 composite points (91.1 versus 77.6, n=20), while GPT-5.4 xhigh leads Temporal by 13.7 (87.2 versus 73.5, n=22). GPT-5.4’s other clear advantage is in adversarial handling. These categories require more case-specific judgment than routine routing or fare calculation. GPT-5.6 sol at xhigh efort keeps GPT-5.4’s profile in nine categories but not in these two. It matches Qwen 27B base on Accessibility (90.8) and falls to 68.6 on Temporal, the lowest score in the cluster and below the 4B student; luna at medium efort shows the same pattern (92.1 and 72.9). Sol’s Temporal deficit alone exceeds its 0.63-point composite gap to GPT-5.4 full at xhigh efort in Table 2, and luna finishes 0.12 points above GPT-5.4 despite the same weakness.

The 4B PEFT student posts the cluster’s best Fare score (97.7), shares the best Policy score (96.9) with GPT-5.6 sol, and trails only sol on Routing (97.2 against 97.7); it gives ground mainly on Temporal and Tool-Hallucination. By contrast, the 27B PEFT student shows no category improvement over its base model beyond the per-category single-run interval, consistent with the negative aggregate delta in Section 4.

![](images/cb8d0d206b8571e89f764912119dfafff425343fc3d08aee4568dc85ff888afc.jpg)  
Figure 3: Per-category composite for the seven models of the deployment comparison (the two Qwen 3.5 teachers, the students from 4B upward, and the two OpenAI xhigh rows) on the held-out partition (n = 238), columns in held-out rank order. Cell shading encodes the mean composite score (light = 75, dark = 100); the orange outline marks the best model in each category (the row maximum). No model wins more than three of the eleven categories.

The held-out category counts are small, from 11 cases for Compound Stress to 32 for Disruption, so each cell carries more uncertainty than the matrix-level ±1.9 points; we do not read smaller cell diferences as meaningful.

## 3.3 Rule-based baseline

We use a deterministic scripted agent to estimate how much of the benchmark can be solved through fixed tool orchestration. The agent calls route\_planner, fare\_calculator, and submit\_assistant\_state in a fixed order, while reading structured case fields for temporal and disruption checks. It reaches a composite score of 77.1 and a Tier 1 score of 84.6 on the held-out partition; its per-system Tier 1 scores range from 80.4 to 88.7.

The script performs well on Routing (93.5 composite), Fare (91.7), and Cultural cases (82.1), then drops sharply on Temporal (52.1), Compound Stress (68.9), Accessibility (69.7), and Policy (73.3). As Figure 4 shows, the language-model advantage is concentrated in cases that require a decision beyond forwarding tool output.

The rule-based score also anchors the lower end of Table 2. Mistral Nemo 12B falls below it, scoring 56.67 composite and 57.50 on Tier 1, compared with 77.1 and 84.6 for the script. Ministral 8B exceeds the baseline, scoring 85.68 composite and 87.33 on Tier 1. Qwen 2B base and 0.8B base fall below the script on both measures; the 2B PEFT student clears it by 2.59 points on composite but remains 3.40 points lower on Tier 1. All remaining ranked rows exceed both baseline scores. The scripted agent ships as harness/rule\_agent.py; the repository’s reproduction guide covers the runs.

![](images/fbf97d68a8d086c42e2fde7ff0927a73cbfeb3be5c2e75308d42e5ab844565b3.jpg)  
Figure 4: Held-out per-category composite for the rule-based baseline and Qwen3.5-27B base, with the diference in points in the right margin. Routing and fare show small gains; every category that requires a per-scenario decision gains more than ten points.

The excluded Gemma variants fail in a specific way. E2B and E4B achieve 99.9 percent scope adherence and 100 percent no-tool-hallucination, but roughly half of their submission attempts receive an HTTP 422 validation error. Their main problem is therefore structured-state synthesis, not tool selection. The Gemma 4 technical report (Google DeepMind, 2026) does not report multi-round function-calling results on BFCL (Patil et al., 2025) or MCP Atlas, and Gemma’s native tool format uses special tokens rather than OpenAI-compatible JSON schemas. Gemma 4 26B-A4B completes the same protocol normally, suggesting that the failure is size-dependent within this family.

## 3.4 Serving-configuration sensitivity

Under the uniform serving configuration used for the rest of Table 2, greedy decoding with a 4096-token output budget, Qwen3.8-27B appears to regress against Qwen3.5-27B: 89.48 versus 93.08 on held-out Tier 1, an apparent decline of 3.60 points. Most of that gap is configuration rather than capability, and decomposing it determines how Table 2 reports Muse Glimmer 30B, GLM-4.7-Flash, Qwen3.6-27B, and Qwen3.8-27B.

The gap has two sources, the first of which is the output budget. Qwen3.8 produces much longer reasoning traces, and at 4096 tokens a case can exhaust the budget inside the reasoning block, so that no terminal state is ever emitted; the harness records such cases as truncations rather than as normal completions and escalates the budget. Raising the budget to 16384 lifts Qwen3.8 by 1.72 points and leaves Qwen3.5’s output unchanged, since it completes every case within 4096 tokens and the cap never binds. The second source is decoding: the Qwen3.8 model card (Qwen Team, 2026c) recommends sampling at temperature 1.0, top-p 0.95, and top-k 20 for thinking mode, and under greedy decoding the model intermittently falls into repetition loops, whereas at the recommended sampling Qwen3.8 gains a further 1.00 point. The same change costs Qwen3.5 0.81 points and Qwen3.6 1.10 points, so the configuration optimum flips between generations.

The generational comparison therefore depends on the protocol: Qwen3.8 trails Qwen3.5 by 3.60 points under the uniform configuration, by 1.88 with the budget raised, by 0.07 with both models at the vendor sampling, and by 0.88 with each model at its own best configuration, so roughly 2.7 of the apparent 3.6-point regression is configuration. Repeated runs put the per-system spread at 0.23 Tier 1 points under greedy decoding (server-level nondeterminism, three MARTA runs) and 0.32 under sampling (three seeds, BART), so we quote pooled six-system deltas only and treat per-system deltas as noise; the sampled rows are single runs at seed 42. A larger budget is not uniformly better either: given 16384 tokens, Muse Glimmer drops from 92.75 to 92.38 and GLM-4.7-Flash from 89.67 to 89.62, and GLM falls further to 88.58 with 32768 tokens and forty tool rounds, where the extra room feeds non-terminating tool loops rather than better answers.

Table 3: Held-out Tier 1 for three Qwen 27B-class generations under three serving configurations: greedy decoding at the 4096-token budget used for the rest of Table 2, greedy decoding at 16384 tokens, and vendor-recommended sampling at 16384 tokens.
<table><tr><td>decoding, output budget</td><td>Qwen3.5-27B</td><td>Qwen3.6-27B</td><td>Qwen3.8-27B</td></tr><tr><td>greedy, 4096 (uniform configuration)</td><td>93.08</td><td>92.58</td><td>89.48</td></tr><tr><td>greedy, 16384</td><td>93.08 †</td><td>93.63</td><td>91.20</td></tr><tr><td>vendor sampling, 16384</td><td>92.27</td><td>92.53</td><td>92.20</td></tr></table>

† Zero truncations at 4096, so the larger budget cannot change the greedy output.

A single global serving configuration is not neutral across model generations, and a leaderboard that imposes one converts configuration mismatch into apparent capability diferences; swapping the model name while keeping the serving configuration would have cost 2.7 points here. Table 2 therefore ranks each of these rows at its own best configuration, with the full per-row serving record in Appendix B.4. Those configurations were selected on the same held-out cases that Table 2 ranks, so the rows marked <sup>3</sup> carry a selection optimism that this analysis does not quantify.

## 4 PEFT and the Capacity-Ceiling Curve

We distil four Qwen students, at 2B, 4B, 9B, and 27B, from 600 examples drawn only from the 717-case training partition. The teachers are Qwen 3.5 27B-dense and 35B-A3B. We retain traces that score at least 90 percent on Tier 1 and deduplicate by case, keeping the higher-scoring trace. This produces 540 examples from the 27B teacher and 60 from the 35B teacher, with a mean Tier 1 score of 99.0 percent. No held-out case enters the teacher pool.

All four students use QLoRA (Dettmers et al., 2023), the 4-bit-quantised form of low-rank adaptation (LoRA) (Hu et al., 2022). The shared recipe uses rank 16, three epochs, and batch size 2 with gradient accumulation 4. The maximum sequence length is 4096 tokens for the 2B, 4B, and 9B students and 2048 for 27B, which is limited by 32 GB of VRAM. We train each size at seeds 42 and 43, and the 2B size additionally at seed 44, varying the LoRA initialisation, optimiser state, and the split between training and validation. Per-seed training time on one RTX 5090 ranges from 27 minutes at 2B to 103 minutes at 27B. Appendix C gives the complete recipe.

The RTX 5090 is a deliberate resource constraint. We use one high-end consumer GPU to test whether the open-weight PEFT sweep and local evaluation can be completed without a datacenter accelerator, not to maximise training throughput. The GPT-5.6, GPT-5.4, Mistral, and Haiku components remain API-hosted.

Table 4: Held-out Tier 1 results by Qwen student size (n=238). Student scores are means over training seeds (n=3 at 2B, n=2 otherwise); ∆ is relative to the corresponding base model.
<table><tr><td>Size</td><td>Base T1</td><td>seed=42 T1</td><td>seed=43 T1</td><td>seed=44 T1</td><td>mean</td><td></td><td>∆ vs base seed-spread</td><td>Q4_K_M GGUF</td></tr><tr><td>2B</td><td>74.17</td><td>76.80</td><td>82.07</td><td>84.74</td><td>81.20</td><td>+7.03</td><td>±3.97</td><td>1.2 GB</td></tr><tr><td>4B</td><td>89.32</td><td>91.82</td><td>90.83</td><td></td><td>91.32</td><td>+2.00</td><td>±0.49</td><td>2.6 GB</td></tr><tr><td>9B</td><td>89.38</td><td>90.53</td><td>91.53</td><td></td><td>91.03</td><td>+1.65</td><td>±0.50</td><td>5.3 GB</td></tr><tr><td>27B</td><td>92.32</td><td>91.93</td><td>90.88</td><td></td><td>91.41</td><td>-0.91</td><td>±0.53</td><td>16 GB</td></tr></table>

Figure 5 shows that the PEFT gain decreases monotonically as base capability rises and becomes negative at 27B. Every seed agrees on the direction at every size: the 2B (three seeds), 4B, and 9B students improve over their bases at every seed, while both 27B students regress. The full 955-case matrix shows the same sequence, with deltas of +6.03, +1.72, +1.09, and −1.07. Appendix G describes the underlying artefacts.

![](images/149e91eda1633e36bb85b54db432529b5bfeed73f8b3718006a45c6296bc20d9.jpg)

(B) seed spread by partition  
![](images/a4cea2e5bfea9dc861c2703751573292c58f09ecf4126b3d2c3c7145d58060e5.jpg)  
Figure 5: PEFT gain and seed sensitivity by model size. Panel A: held-out Tier 1 delta over the corresponding base model, with error bars spanning the per-size seed spread (three seeds at 2B, two otherwise). Panel B: seed spread (half the Tier 1 range across seeds) on the full 955-case matrix with the held-out spread overlaid; the full-matrix spread collapses from 3.96 to 0.09 while the held-out spread is floored near 0.5 by case-sampling noise.

The held-out partition is not large enough to establish any individual pairwise diference. Its paired-bootstrap 95% intervals all include zero. The 4B gain is 1.97 points on Tier 1 [−0.17, +4.16], the 27B regression is −0.94 [−2.26, +0.39], and the 4B student difers from GPT-5.4 full xhigh by −0.05 [−1.80, +1.70].

The full matrix has greater power. There, the 4B PEFT gain is 1.72 points on Tier 1 [+0.72, +2.74], and the 27B regression is −1.09 [−1.82, −0.38]. Both intervals exclude zero. The 4B student and GPT-5.4 full xhigh are tied on Tier 1 $\left( + 0 . 0 1 \ [ - 0 . 9 3 , + 0 . 9 4 ] \right)$ ), although the student trails by about one composite point. The capacity-ceiling result is supported by the consistent trend across four sizes, two to three seeds per size, and both evaluation partitions, together with the significant 4B gain and 27B regression on the full matrix. It does not depend on any single held-out comparison. Appendix D reports the bootstrap procedure and paired results.

Training-seed sensitivity changes with model size. On the full matrix, the seed spread (half the Tier 1 range across seeds) is 3.96 points at 2B across three seeds, 0.33 at 4B, 0.17 at 9B, and 0.09 at 27B. The 2B result therefore depends materially on the training seed, whereas the two 27B runs are nearly identical. On the smaller held-out partition, the spread is 3.97 points at 2B and about 0.5 from 4B upward; sampling variation limits what can be resolved at that size.

One interpretation connects this decline in variance to the decline in PEFT utility. A weaker base model leaves more room for an adapter to alter task behaviour, and the resulting student is correspondingly more sensitive to training variation. As the base approaches the task ceiling, both the average benefit and the variation between runs contract.

This conclusion is limited to the recipe tested here: QLoRA rank 16, three epochs, and 600 high-quality teacher traces from the training partition. Larger or diferently composed datasets, other ranks, and other learning rates may move the point at which PEFT stops helping. The hardware constraint also makes maximum sequence length part of the tested configuration: the 27B student uses 2048 tokens rather than 4096. A datacenter accelerator could remove that diference, and we do not test whether doing so would alter the 27B delta.

## 5 Related Work

MetroLLM-Bench sits between agent evaluation and structured-output evaluation. Agent benchmarks such as τ-bench (Yao et al., 2024), GAIA (Mialon et al., 2023), and AgentBench (Liu et al., 2023) test multi-step too use across many domains, usually with pass-or-fail or score-out-of-100 outcomes. Live API-Bench (Elder et al., 2025) similarly emphasises breadth, exposing more than 2,500 executable tools. MetroLLM-Bench narrows the action space to six tools in one operational domain. That narrower setting supports diagnostic categories, including a separate test of service hours and disruption windows, and allows deterministic components to be reused during fine-tuning.

JSONSchemaBench (Geng et al., 2025) and StructEval (Yang et al., 2025b) isolate the ability to produce schema-valid output from short prompts. A kiosk state is also structured, but validity alone is insuficient: the route, fare, outcome, and purchase decision must agree with the preceding tool calls. MetroLLM-Bench therefore evaluates the terminal contract together with the decisions that produced it.

The PEFT experiments build on work showing that distilled trajectories can improve smaller agents. FireAct (Chen et al., 2023) reports a 77 percent improvement on HotpotQA after LoRA fine-tuning a 7B student on GPT-4 trajectories, although the student still trails the frontier model. Jhandi et al. (2025) fully fine-tune a 350M-parameter model to 77.55 percent on ToolBench, exceeding much larger prompted baselines. Our contribution is a four-size sweep at two to three seeds, which identifies a setting in which the benefit of adapter training changes sign.

Transit-specific studies have used language models for trip planning (Wang & Shalaby, 2024), travel-choice prediction under delays (Chen et al., 2024), and GTFS understanding (Devunuri et al., 2024). These systems treat the model as a feature extractor or conversational assistant. To our knowledge, no previous benchmark evaluates a language model as the policy layer of a kiosk that must call tools and commit a machine-renderable operational state.

## 6 Discussion and Limitations

The results above describe a bounded task, in which the model chooses among six tools, typically makes three to seven calls within a twenty-round budget, and returns a constrained terminal state. Routing and fare arithmetic have deterministic ground truth, and the setting does not strongly reward long-horizon reasoning.

That boundary shows in the category results, above all in Temporal reasoning, where GPT-5.4 full at xhigh efort scores 87.2 composite against 73.5 for Qwen 27B base and 68.6 for GPT-5.6 sol at the same efort level, the widest spread among the leading models. The advantage therefore belongs to one frontier configuration rather than to reasoning efort as such, and where it exists it is large. The aggregate parity in Section 3 should not be generalised to tasks with broader action spaces, longer horizons, or less constrained outputs.

Among the PEFT students, measured quality is flat above 4B. On the held-out partition, the 4B, 9B, and 27B students score 91.32, 91.03, and 91.41 on Tier 1, a range of 0.38 points. The 4B student thus provides the same measured task quality as the 27B student, exceeds both GPT-5.6 tiers on Tier 1, and matches GPT-5.4 full at xhigh efort. Past that plateau, what separates the students in practice is footprint: the 4B ships in 2.6 GB of Q4\_K\_M, the 27B in 16 GB. Appendix B reports exploratory Apple Silicon inference measurements; the benchmark does not establish an end-to-end latency requirement for kiosk deployment.

The evaluation has several statistical limits. Most non-PEFT models have one run. The bootstrap intervals in Appendix D have half-widths of about one Tier 1 point on the full matrix and about two on the 238-case held-out partition, so diferences below one point should not be treated as meaningful, and the held-out leaderboard resolves only larger gaps. The GPT-5.6 and GPT-5.4 rows also run at temperature 1.0, the only value these products permit, and therefore carry unmeasured run-to-run variance; the temperature-zero local evaluations show only the server-level spread of 0.23 Tier 1 points per system measured in Section 3.4, and the Qwen3.8-27B row is a single sampled run. A temperature-zero cross-check would require a diferent model configuration.

The leading rows of Table 2 are compressed: eleven models lie within 3.18 composite points, a range comparable to the ±1.9 single-run interval, so the leaderboard resolves the order of the strongest models only weakly. The compression is consistent with the design of the benchmark, which bounds the task deliberately; the scripted agent places a floor at 84.6 Tier 1, and Section 4 shows the deterministic ceiling being reached by a 4B student. The resolving power of the benchmark therefore lies below that ceiling, where the 2B to 9B students and the budget API tiers separate by several points, and within the categories that require a decision, where the leading models still difer by 13.5 points on Accessibility and 18.6 on Temporal. Extending the measured range upward would require harder cases, in particular compound and temporal scenarios.

The training seeds measure training stochasticity but do not by themselves measure case-sampling variation. Appendix D addresses the latter with paired bootstrap intervals for the four principal comparisons. Even so, the capacity-ceiling result remains specific to this scoring contract and to the recipe tested here: QLoRA rank 16, 600 traces, and three epochs. It need not hold for other tasks, datasets, or adapter settings.

The 2B-to-4B jump is similarly limited to the evidence in the matrix. It is reproducible within Qwen and points in the same direction for Gemma, but it is not a general law. Qwen 35B-A3B ranks seventh overall despite having only 3B active parameters, placing it inside the apparent threshold range by active count. Architecture and training data remain important confounders.

## 7 Conclusion

On this bounded kiosk task, a proprietary frontier API is not necessary. The 2.6 GB open-weight Qwen 3.5 4B PEFT student exceeds both GPT-5.6 tiers on Tier 1 (91.32 against 90.63 for luna and 90.00 for sol) and matches GPT-5.4 full at xhigh efort (91.37). Weights, adapter, prompts, tools, and inference can all stay inside infrastructure the operator controls, at no cost in deterministic performance.

The scaling sweep makes a separate point, about suficiency. Under the tested configuration, 4B reaches the measured Tier 1 plateau: neither 9B nor 27B PEFT improves on it, while 27B adaptation significantly reduces Tier 1 relative to its base model on the full matrix. This is not evidence that smaller models are intrinsically better—the unadapted Qwen 3.5 27B remains the strongest of the four sizes—but it shows that added capacity and adaptation need not help a competent base model. The nine runs took 9.4 GPU-hours on one high-end consumer GPU; this adaptation study did not require a datacenter accelerator.

The boundary is equally important. The student trails GPT-5.6 luna, GPT-5.6 sol, and GPT-5.4 xhigh on composite score; the frontier rows retain an advantage on Adversarial cases, and GPT-5.4 xhigh on Temporal cases as well. The 27B regression is specific to the tested recipe, including its shorter training sequence length. The result does not show that small models generally replace frontier models. It shows that when an operational task is deliberately bounded, an open-weight model adapted to that setting can match a frontier API on its core executable requirements.

## References

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. FireAct: Toward language agent fine-tuning. arXiv:2310.05915, 2023.

Chen Chen, Yuxin He, Hao Wang, Jingjing Chen, and Qin Luo. DelayPTC-LLM: Metro passenger travel choice prediction under train delays with large language models. arXiv:2410.00052, 2024.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Eficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems 36, 2023. arXiv:2305.14314.

Saipraneeth Devunuri, Shirin Qiam, and Lewis Lehe. ChatGPT for GTFS: Benchmarking LLMs on GTFS understanding and retrieval. Public Transport, 16(2):333–357, 2024. arXiv:2308.02618.

Benjamin Elder, Anupama Murthi, Jungkoo Kang, Ankita Rajaram Naik, Kiran Kate, Kinjal Basu, and Danish Contractor. Live API-Bench: 2500+ live APIs for testing multi-step tool calling. arXiv:2506.11266, 2025.

Alvan R. Feinstein and Domenic V. Cicchetti. High agreement but low kappa: I. the problems of two paradoxes. Journal of Clinical Epidemiology, 43(6):543–549, 1990.

Saibo Geng, Hudson Cooper, Michał Moskal, Samuel Jenkins, Julian Berman, Nathan Ranchin, Robert West, Eric Horvitz, and Harsha Nori. JSONSchemaBench: A rigorous benchmark of structured outputs for language models. arXiv:2501.10868, 2025.

Yilin Geng, Haonan Li, Honglin Mu, Xudong Han, Timothy Baldwin, Omri Abend, Eduard Hovy, and Lea Frermann. Control illusion: The failure of instruction hierarchies in large language models. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026), Main Track, 2026. arXiv:2502.15851.

Google DeepMind. Gemma 4 model card. https://ai.google.dev/gemma/docs/core/model\_card\_4, 2026.

Kilem L. Gwet. Computing inter-rater reliability and its variance in the presence of high agreement. British Journal of Mathematical and Statistical Psychology, 61(1):29–48, 2008.

Kilem L. Gwet. Handbook of Inter-Rater Reliability: The Definitive Guide to Measuring the Extent of Agreement Among Raters. Advanced Analytics, Gaithersburg, MD, 4th edition, 2014.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. arXiv:2106.09685.

Polaris Jhandi, Owais Kazi, Shreyas Subramanian, and Neel Sendas. Small language models for eficient agentic tool calling: Outperforming large models with targeted fine-tuning. In AAAI 2026 Workshop on Agentic AI Benchmarks and Applications for Enterprise Tasks, 2025. arXiv:2512.15943.

Gauri Kholkar and Ratinder Ahuja. Policy-as-prompt: Turning AI governance rules into guardrails for AI agents. In 3rd Regulatable ML Workshop, NeurIPS 2025, 2025. arXiv:2509.23994.

J. Richard Landis and Gary G. Koch. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174, 1977.

Xiao Liu, Hao Yu, Hanchen Zhang, et al. AgentBench: Evaluating LLMs as agents. arXiv:2308.03688, 2023.

Llama Team, AI@Meta. The Llama 3 herd of models. arXiv:2407.21783, 2024.

Meta Superintelligence Labs. Muse Glimmer-30B model card. Hugging Face, https://huggingface.co/ meta-models/Muse-Glimmer-30B, August 2026.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: A benchmark for general AI assistants. arXiv:2311.12983, 2023.

Mistral AI. Introducing Mistral Small 4. Blog post and model card, https://mistral.ai/news/ mistral-small-4, https://docs.mistral.ai/models/mistral-small-4-0-26-03, March 2026. Model ID mistral-small-2603; 119B total parameters, 6.5B active per token, MoE.

OpenAI. GPT-5 system card update (GPT-5.2). https://openai.com/index/ gpt-5-system-card-update-gpt-5-2/, 2025a.

OpenAI. OpenAI GPT-5 system card. arXiv:2601.03267, 2025b.

OpenAI. Previewing GPT-5.6 Sol, and the GPT-5.6 preview system card. https://openai.com/index/ previewing-gpt-5-6-sol/, https://deploymentsafety.openai.com/gpt-5-6-preview, June 2026.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The Berkeley function-calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), volume 267 of PMLR, pp. 48371–48392, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents. Blog post, https://qwen.ai/blog?id=qwen3.5, February 2026a.

Qwen Team. Qwen3.6-27B model card and release blog. https://huggingface.co/Qwen/Qwen3.6-27B, https://qwen.ai/blog?id=qwen3.6-27b, April 2026b.

Qwen Team. Qwen3.8-27B model card. https://huggingface.co/Qwen/Qwen3.8-27B, August 2026c.

Jesús Sánchez Cuadrado, Sara Pérez-Soler, Esther Guerra, and Juan de Lara. Automating the development of task-oriented LLM-based chatbots. In Proceedings of the 6th ACM Conference on Conversational User Interfaces (CUI 2024), 2024. URL https://miso.es/pubs/CUI24.pdf.

Mrinank Sharma, Meg Tong, Tomasz Korbak, et al. Towards understanding sycophancy in language models. arXiv:2310.13548, 2023.

Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, and Alex Beutel. The instruction hierarchy: Training LLMs to prioritize privileged instructions. arXiv:2404.13208, 2024.

J. Wang and A. Shalaby. Leveraging large language models for enhancing public transit services. arXiv:2410.14147, 2024.

An Yang et al. Qwen3 technical report. arXiv:2505.09388, 2025a.

Jialin Yang, Dongfu Jiang, Lipeng He, et al. StructEval: Benchmarking LLMs’ capabilities to generate structural outputs. arXiv:2505.20139, 2025b.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. arXiv:2210.03629.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ -bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv:2406.12045, 2024.

Z.ai. GLM-4.7-Flash model card. Hugging Face, https://huggingface.co/zai-org/GLM-4.7-Flash, 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track, 2023. arXiv:2306.05685.

## A AI-assistance disclosure

Generative AI tools were used in preparing this work: for case authoring, for harness and analysis code, and for drafting text. The Figure 1 artwork was produced with an image-generation model from an author-written specification; Figures 2 to 5 are plotted directly from the scored result files. The author reviewed and edited all generated content and takes full responsibility for the content of this paper.

## B Reproduction

Source code repository: https://github.com/continker/metrollm-bench (tag paper-v1.2).

## B.1 Harness

• Mock server: FastAPI (port 8100), Pydantic-validated responses, per-case disruption state, six tools (route\_planner, fare\_calculator, station\_info, disruption\_feed, line\_info, knowledge\_base) and the terminal tool submit\_assistant\_state.

• Runner: async httpx, OpenAI chat-completions interface, at most 20 tool rounds per case, 240 s timeout per LLM round, semaphore-gated concurrency (N=2 local, N=4 Mistral API, N=8 Azure).

• Cases: cases/{system}\_cases.json, 156–167 cases per system across six systems, 955 total.

• Scorer: deterministic Tier 1 (14 components); semantic-quality Tier 2 (8 components: 6 using Haiku and 2 scored programmatically), with language-model judgments cached per (case, rubric).

• Framebook: per-system YAML injected into the system prompt at runtime (terminology, fare rules, operating hours, cultural notes).

## B.2 Commands

The full command sequence, from partition build through teacher-trace generation, PEFT training, benchmarking, and the partition-filtered statistics, is maintained as REPRODUCING.md in the repository, where it stays in step with the code. The pipeline entry points are scripts/build\_holdout\_split.py and scripts/slice\_cases\_by\_split.py (partition), harness/mock\_server.py, harness/runner.py, and harness/scorer.py (evaluation), harness/rule\_agent.py (baseline), scripts/peft/ (training-set construction, training, merge, GGUF export), and scripts/score\_split.py, scripts/compute\_bootstrap\_heldout.py, and scripts/heldout\_percategory.py (statistics).

## B.3 Hardware

All local inference ran on a single NVIDIA RTX 5090 with 32 GB of VRAM. The host used Ubuntu 24.04, a llama.cpp build from early April 2026, two parallel slots, flash attention, and bf16 compute. PEFT training used the same GPU through Hugging Face transformers, peft, and bitsandbytes.

## B.4 Model configurations

Table 5: Evaluation-time model configurations.
<table><tr><td>Family</td><td>Temperature</td><td>Thinking</td><td>reasoning_effort</td></tr><tr><td>Qwen 3.5 local</td><td>0.0</td><td>on</td><td> $\mathrm { n / a }$ </td></tr><tr><td>Gemma 4 local</td><td>0.0</td><td> $\mathrm { n / a }$ </td><td> $\mathrm { n / a }$ </td></tr><tr><td>Mistral API</td><td>0.0</td><td> $\mathrm { n / a }$ </td><td> $\mathrm { n / a }$ </td></tr><tr><td>GPT-5.6 Azure</td><td>1.0</td><td> $\mathrm { n / a }$ </td><td>medium (luna) / xhigh (sol)</td></tr><tr><td>GPT-5.4 Azure</td><td>1.0</td><td> $\mathrm { n / a }$ </td><td>medium / high / xhigh</td></tr></table>

As Table 5 shows, the OpenAI rows run at temperature 1.0, the only value GPT-5 deployments accept at the author’s Azure subscription tier; GPT-5.4 full was run at all three efort levels and nano and mini at medium.

That run-to-run variance is not separately measured. The local and Mistral evaluations use temperature 0.0, where Section 3.4 measures a server-level spread of 0.23 Tier 1 points per system, and Qwen3.8-27B is the one local row served with sampling. Appendix D quantifies case-sampling uncertainty only; Section 6 discusses the remaining sources as limitations.

GPT-5.6 was served through Azure OpenAI’s Responses API: for this family the chat-completions endpoint rejects function tools combined with a reasoning-efort setting. The harness converts the Responses payload to the chat-completions shape before recording, so scoring is unchanged, and the run metadata records the dialect. Sol uses xhigh efort to match the GPT-5.4 full comparator row; luna uses the medium default, as nano and mini do. Both are single runs at temperature 1.0.

Rows added after v1 were served on llama.cpp build b10398 (the v1 rows on b8642) at the per-model configurations in Table 6 (all with twenty tool rounds), which Section 3.4 motivates. Re-benchmarking Qwen3.5-27B on b10398 under the otherwise unchanged v1 configuration gives 93.08 held-out Tier 1 against the published 92.32 on b8642: the serving build itself is part of the configuration, and cross-build comparisons should be read with that margin in mind.

Table 6: Per-model serving configurations for the rows marked <sup>3</sup> in Table 2 and for Llama 3.1-8B, all on llama.cpp b10398 with twenty tool rounds.
<table><tr><td>Model</td><td>Decoding</td><td>Output budget</td><td>Notes</td><td></td></tr><tr><td>Muse Glimmer 30B</td><td>greedy (0.0)</td><td>4096</td><td>reasoning_strength: chat-template kwargs</td><td>high via</td></tr><tr><td>Qwen3.6-27B</td><td>greedy (0.0)</td><td>16384</td><td></td><td></td></tr><tr><td>Qwen3.8-27B</td><td>temp 1.0, top-p 0.95, top-k 20, min-</td><td>16384</td><td colspan="2">vendor-recommended sampling; single run, seed 42</td></tr><tr><td>GLM-4.7-Flash</td><td>p 0.0 greedy (0.0)</td><td>4096</td><td colspan="2">larger budgets degrade (§3.4)</td></tr><tr><td>Llama 3.1-8B</td><td>greedy (0.0)</td><td>4096</td><td colspan="2">excluded from ranking (Appendix F)</td></tr></table>

## B.5 Case generation

Case selection was manually curated as described in Section 2.1. cases/generator.py combines categoryspecific definitions with per-system metadata in test\_pairs.json and disruption templates in events.yaml. It calls MetroGraph and FareCalculator to derive routes and fares, then writes (events, system\_context, ground\_truth) records together with scoring weights and tolerances. Generation-time tests cover required fields, unique identifiers, station references, graph-valid routes, exact Category B fare consistency, outcome constraints, and category-specific structure. The committed cases/{system}\_cases.json files are the canonical evaluation artefacts used for all reported runs; reproducing the reported scores does not require regenerating them. Version 23 added fifteen cases selected after the documented scenario gap audit, all of which are pinned to the held-out partition.

## B.6 Apple Silicon hardware envelope

Exploratory measurements characterise local inference on two Apple Silicon systems, separate from the RTX 5090 configuration in Section B.3; they do not constitute an end-to-end kiosk latency study. The fanless 16 GB MacBook Air M2 sustains 39 tokens per second of 2B Q4\_K\_M decode after thermal stabilisation, the fan-cooled 96 GB M2 Max 108 tokens per second, and a 4B student is bandwidth-projected (not measured) to roughly 18 tokens per second on the Air. Throughput depends on architecture and quantisation rather than the adapter, and the 2B student used for the probe scores 85.8 Tier 1 on the fifteen-case probe set under deterministic Q4\_K\_M decoding. The figures describe decode throughput in the most favourable realistic operating state (lid open, on AC power), not user-perceived response time. The measurement protocol and thermal-curve scripts are documented in the repository’s REPRODUCING.md and published as continker/metrollm-bench-mac on Hugging Face, so the measurements can be repeated on any Apple Silicon Mac.

## B.7 Independent ground-truth validation

A second annotator, independent of the project, audited the answer key itself, separately from the judge calibration in Appendix E. Fifty cases were sampled one per (system, category) cell (seed 42), covering all six systems and all eleven categories. A browser-based review interface presented the expected route and any closures on the network map, the system’s fare rules beside the expected fare and breakdown, and the expected terminal kiosk state; the annotator gave a verdict on each of route, fare, and outcome, with a cannot-verify option throughout. The audit confirmed 38 of 38 expected routes, 36 of 38 fares, and 49 of 50 outcomes. One flag identified a confirmed answer-key error: in the passenger flip-flop scenario (H-006, present in all six systems), the expected fare was priced for two adults while the case’s own acceptance pattern requires the one-adult total. The committed case files are corrected; no reported score was afected, because Category H does not weight fare\_correct. The two remaining flags raise design questions rather than defects and are discussed in Appendix F.

The follow-up investigation surfaced three further issues. Some disruption definitions list closure endpoints that are not adjacent in the graph, so the entry closes no edge; in all but two cases another closure in the same event still severs the route, and the two contradictory cases are retained unchanged and documented in Appendix F. The day-of-week label in 90 of 114 temporal contexts disagrees with the weekday of its timestamp; the open-or-closed determination is identical under either day in all 90 cases, so no expected outcome is afected. Finally, regeneration revealed that the generator’s tie-breaking between lines that share track is not deterministic; the committed case files remain canonical (Section B.5), and the tie-break should be made deterministic before the case files are regenerated.

## C PEFT training details

## C.1 Training-set composition

The two teacher models, Qwen 3.5 27B-dense and 35B-A3B MoE, run only on the 717-case training partition. We retain a trace when the teacher scores at least 90 percent on Tier 1, then deduplicate by case\_id, keeping the higher-scoring instance. The final set contains 540 traces from the dense 27B teacher and 60 from the 35B-A3B teacher. Its mean Tier 1 score is 99.0 percent. No held-out case enters a student’s training data.

Table 7: Per-category PEFT training coverage (training examples / training-partition cases).
<table><tr><td>Cat</td><td>In training</td><td>Train-partition cases</td><td>Coverage</td></tr><tr><td>A Routing</td><td>91</td><td>91</td><td>100%</td></tr><tr><td>B Fare</td><td>73</td><td>73</td><td>100%</td></tr><tr><td>C Disruption</td><td>66</td><td>92</td><td>72%</td></tr><tr><td>D Accessibility</td><td>67</td><td>71</td><td>94%</td></tr><tr><td>E Cultural</td><td>27</td><td>28</td><td>96%</td></tr><tr><td>F Policy</td><td>72</td><td>72</td><td>100%</td></tr><tr><td>G Multi-turn</td><td>54</td><td>67</td><td>81%</td></tr><tr><td>H Adversarial</td><td>53</td><td>65</td><td>82%</td></tr><tr><td>I Temporal</td><td>33</td><td>68</td><td>49%</td></tr><tr><td>J Hallucination</td><td>48</td><td>71</td><td>68%</td></tr><tr><td>K Compound</td><td>16</td><td>19</td><td>84%</td></tr></table>

The Tier 1 threshold is a capability filter, not a category filter, so the coverage in Table 7 reflects where the teachers succeed. Temporal reasoning (Category I) is undersampled because the base teachers score about 83 percent there and most traces fail the threshold. Compound Stress (Category K) has high proportional coverage but only nineteen cases in the training partition. It is therefore the category most vulnerable to memorisation, and the capacity-ceiling argument in Section 4 does not depend on it.

## C.2 Hyperparameters

All four students follow the same recipe. The maximum sequence length is reduced from 4096 to 2048 at 27B to fit within 32 GB of VRAM.

• Base: Qwen 3.5 2B / 4B / 9B / 27B (bf16 weights)

• Method: QLoRA with 4-bit NF4 base quantisation

• LoRA rank 16, alpha 32, dropout 0.05

• Target modules: q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj

• 3 epochs, batch 2 with gradient accumulation 4 (efective batch 8)

• Optimiser paged\_adamw\_32bit, lr 2e-4 cosine, 5 percent warmup

• Seeds 42 (published as continker/Qwen3.5-{size}-metro-v24) and 43 (held internally) at every size, with seed 44 added at 2B, where seed sensitivity is largest, and evaluated in September 2026 under the same toolchain and serving configuration as the other seeds (llama.cpp b8642, 4096- token budget, greedy decoding); Section 4 reports the per-seed comparison. The --seed flag of scripts/peft/train.py propagates to LoRA initialisation, optimiser state, and the split between training and validation.

Re-running the published seed-42 artefact twice under that configuration reproduces its full-matrix Tier 1 within 0.5 points (76.61 and 75.74 against 76.24) while moving the held-out value by 2.1 and 0.3 points, and a second run of the seed-44 student difers from the first by 1.1 held-out points. Run-to-run variation for a 2B model is therefore about one held-out point, several times the 0.23 per-system spread measured for the 27B models in Section 3.4, and the 2B seed spread in Table 4 should be read against that floor.

Table 8: PEFT training time and artefact sizes by student size.
<table><tr><td>Size</td><td>max_seq</td><td>Training time / seed</td><td>Adapter</td><td>Q4_K_M GGUF</td></tr><tr><td>2B</td><td>4096</td><td>27 min</td><td>42 MB</td><td>1.2 GB</td></tr><tr><td>4B</td><td>4096</td><td>63 min</td><td>100 MB</td><td>2.6 GB</td></tr><tr><td>9B</td><td>4096</td><td>74 min</td><td>150 MB</td><td>5.3 GB</td></tr><tr><td>27B</td><td>2048</td><td>103 min</td><td>305 MB</td><td>16 GB</td></tr></table>

Table 8 reports per-seed training time and final artefact sizes. Adapter merging, GGUF conversion, and Q4\_K\_M quantisation follow the scripted path in scripts/peft/, documented with exact library versions in the repository’s REPRODUCING.md.

## C.3 Held-out partition

scripts/build\_holdout\_split.py creates the system-stratified 75/25 partition at seed 42. It assigns 717 cases to teacher-trace generation and student training, and reserves 238 for reporting. Fifteen gap-audit cases written after the first PEFT cycle are pinned to the held-out partition: MARTA-D-016, MARTA-F-016, MARTA-F-017, BART-F-016, BART-B-016, CTA-F-016, CTA-C-018, DOHA-E-006, DOHA-E-007, DOHA-C-016, TRTC-B-016, TRTC-C-018, BJM-A-021, BJM-B-016, and BJM-C-022. The per-system held-out fractions range from 24.7 to 25.2 percent. Appendix G describes the underlying artefacts.

## D Statistical methodology

All non-PEFT scores come from a single run. Treating each case as a Bernoulli trial at 90 percent accuracy gives a standard error of $\sqrt { 0 . 9 \cdot 0 . 1 / 9 5 5 } \approx 0 . 9 7$ points for 955 cases and 1.9 points for the 238 held-out cases, or 95 percent half-widths of about 1.9 and 3.8 points. This is a conservative bound, since each case contributes a fractional component score rather than a binary outcome, so the observed intervals below are narrower; diferences below one point remain within measurement noise on either partition.

## D.1 Bootstrap confidence intervals

The PEFT rows use the per-case mean over the two seeds, as in Table 2. The command scripts/compute\_bootstrap\_heldout.py --partition {full,holdout} --n-resamples 5000 re produces the intervals at seed 42. Table 9 reports the results for the full 955-case matrix.

Table 9: Bootstrap confidence intervals for the full matrix.
<table><tr><td>Model</td><td>n</td><td>T1 mean</td><td>T1 95% CI</td><td>Comp. mean</td><td>Comp. 95% CI</td></tr><tr><td>Qwen 27B base</td><td>955</td><td>92.59</td><td>[91.60, 93.51]</td><td>90.69</td><td>[89.83, 91.57]</td></tr><tr><td>Qwen 35B-A3B base</td><td>955</td><td>92.22</td><td>[91.23, 93.13]</td><td>89.85</td><td>[88.95, 90.69]</td></tr><tr><td>Qwen  $2 7 8 + \mathrm { P E F T \ ( n { = } 2 ) }$ </td><td>955</td><td>91.50</td><td>[90.49, 92.45]</td><td>89.63</td><td>[88.74, 90.49]</td></tr><tr><td>Qwen  $9 \mathrm { B } + \mathrm { P E F T \ ( n { = } 2 ) }$ </td><td>955</td><td>91.26</td><td>[90.29, 92.24]</td><td>88.83</td><td>[87.96, 89.63]</td></tr><tr><td>Qwen  $4 \mathrm { B } + \mathrm { P E F T \ ( n { = } 2 ) }$ </td><td>955</td><td>90.94</td><td>[89.94, 91.90]</td><td>88.79</td><td>[87.91, 89.68]</td></tr><tr><td> $\mathrm { G P T - 5 . 4 ~ f u l l ~ x h i g h }$ </td><td>955</td><td>90.93</td><td>[90.05, 91.77]</td><td>89.90</td><td>[89.11, 90.65]</td></tr><tr><td>Qwen 4B base</td><td>955</td><td>89.23</td><td>[88.05, 90.35]</td><td>87.00</td><td>[85.94, 88.06]</td></tr></table>

On the full matrix, the Tier 1 intervals have half-widths of about 1.0 point, roughly half the Bernoulli bound above because component scores are fractional. The corresponding intervals on the 238-case held-out partition are about twice as wide. This loss of precision explains why the individual held-out diferences do not reach significance.

## D.2 Paired bootstrap

The paired bootstrap resamples cases while preserving the pairing between models. An interval that excludes zero is significant at $\alpha = 0 . 0 5$ . Table 10 reports the four principal comparisons for both partitions.

Table 10: Paired-bootstrap Tier 1 comparisons on the held-out and full partitions.
<table><tr><td>A</td><td>B</td><td>∆T1 held-out (n=238)</td><td></td><td>∆T1 full (n=955)</td></tr><tr><td> $4 \mathrm { B } + \mathrm { P E F T }$ </td><td>GPT-5.4 full xhigh</td><td>-0.05</td><td> $[ - 1 . 8 0 , + 1 . 7 0 ]$ </td><td>+0.01  $[ - 0 . 9 3 , + 0 . 9 4 ]$ </td></tr><tr><td> $4 \mathrm { B } + \mathrm { P E F T }$ </td><td>Qwen 4B base</td><td>+1.97</td><td> $[ - 0 . 1 7 , + 4 . 1 6 ]$ </td><td>+1.72  $[ + 0 . 7 2 , + 2 . 7 4 ]$ </td></tr><tr><td> $2 7 \mathrm { B } + \mathrm { P E F T }$ </td><td>Qwen 27B base</td><td>-0.94</td><td> $[ - 2 . 2 6 , + 0 . 3 9 ]$ </td><td>-1.09  $\left[ - 1 . 8 2 , - 0 . 3 8 \right]$ </td></tr><tr><td> $9 \mathrm { B } + \mathrm { P E F T }$ </td><td>Qwen 35B-A3B base</td><td>-1.10</td><td> $[ - 3 . 3 4 , + 1 . 0 7 ]$ </td><td>-0.95  $[ - 1 . 9 1 , + 0 . 0 1 ]$ </td></tr></table>

The full-matrix composite diferences also exclude zero: $- 1 . 1 1 \ [ - 1 . 9 5 , \ - 0 . 3 1 ]$ for the 4B student against $\mathrm { G P T  – 5 . 4 , - 1 . 0 6 \ [ - 1 . 7 2 , - 0 . 4 0 ] }$ for the 27B regression, and $+ 1 . 7 9 \ [ + 0 . 9 0 , + 2 . 7 1 ]$ for the 4B student against its base. Every held-out interval includes zero because 238 cases do not provide enough power to establish an individual pairwise diference. On the full matrix, however, the 4B PEFT gain and 27B regression are significant on both Tier 1 and composite score. The 4B student ties GPT-5.4 full xhigh on Tier 1 (+0.01 $\left[ - 0 . 9 3 , + 0 . 9 4 \right] )$ while trailing by approximately one composite point. Section 4 bases the capacity-ceiling conclusion on these intervals and on the consistency of the trend across partitions and seeds.

## D.3 What the bootstrap does not show

• The bootstrap captures case-sampling variation only. Section 4 reports training-seed variation separately from two seeds (three at 2B), ranging from a 3.96-point spread at 2B to 0.09 at 27B on the full matrix. The two analyses answer diferent questions and are not combined.

• The per-category diferences in Figure 3 are not paired-tested. Held-out category counts range from 11 to 32, so some apparent diferences are within sampling noise.

• The four principal comparisons were selected by the author. A confirmatory study should apply a correction for multiple comparisons.

## E Scoring-stack calibration and judge rubrics

Claude Haiku 4.5 (claude-haiku-4-5-20251001), Anthropic’s small, low-cost model, provides the languagemodel judgments for six of the eight Tier 2 components, sometimes alongside structural checks. It is afordable enough to score every applicable judge rubric throughout the evaluation, and its refusal behaviour is well characterised. All judgments are cached. No Anthropic model is evaluated as an agent, which also avoids comparing an Anthropic agent with an Anthropic judge; the author has no afiliation with Anthropic, which acts solely as an inference provider for the judge.

For calibration, two annotators independently rated the same 100 case-rubric pairs, drawn from 100 diferent cases across all six systems and ten of the eleven categories; the rated responses are GPT-5-mini outputs from a single benchmark run. The author rated first, with each automated score revealed only after the rating was locked; the second annotator, independent of the project, later rated the identical pairs in a session that displayed no judge output at any point. The sample is stratified across six Haiku-using Tier 2 rubrics and the deterministic Tier 1 cultural\_accuracy check used by the deployed scorer, with fourteen or fifteen pairs per rubric, and is enriched for non-full-credit automated outputs. Deployed scores are mapped to a common ordinal $0 / 1 / 2$ scale. Author versus judge: 82 percent exact agreement, 97 percent within one point, $\kappa _ { w } =$ 0.53 (95% bootstrap CI [0.27, 0.75]), moderate under Landis & Koch (1977); on the 85 Haiku-scored pairs, 84 percent exact agreement and $\kappa _ { w } = 0 . 5 6$ . Second annotator versus judge: 69 percent exact, 93 percent within one point, $\kappa _ { w } = 0 . 0 2 \ ( \mathrm { C I } \ [ - 0 . 1 2 , 0 . 1 8 ] )$ . Between the annotators: 76 percent exact, 97 percent within one point, $\kappa _ { w } = 0 . 2 5$ (CI [0.02, 0.46]). Under prevalence-robust Gwet AC2 (Gwet, 2008) the three pairings score 0.89, 0.81, and 0.89; the $\kappa _ { w }$ spread reflects the score marginals, since the second annotator awarded $2 / 2$ on 89 of 100 pairs. The same leniency produced two clear misses: on CTA-H-003 (safety\_response\_quality) and DOHA-D-006 (scope\_adherence) the second annotator awarded full credit where the author and the judge both scored zero.

Two efects explain the diference between the high exact agreement and moderate $\kappa _ { w }$ . First, the H-014 scenicroute cases for TRTC, MARTA, and CTA expose a definitional disagreement over safety\_response\_quality. The judge penalises an ofer that violates a system-prompt constraint, while both annotators credit the valid route to the destination. Second, four rubrics are highly imbalanced: 13 to 15 of the 14 or 15 cases receive $2 / 2$ . The resulting per-rubric κ approaches zero despite exact agreement between 73 and 93 percent. Across the more informative rubrics, $\kappa _ { w }$ is 0.20 for safety\_response\_quality, 0.68 for scope\_adherence, and 0.69 for policy\_acknowledged.

Three rubric changes were made during development. We moved cultural\_accuracy to a deterministic keyword check after the judge introduced unsupported elaboration requirements. We added “do not penalise” clauses and a 60-minute last-train threshold to the temporal and safety rubrics. We also supplied the rejection reason as context for adversarial cases. The author stopped tuning at that point to avoid overfitting the calibration set. Calibration against a judge from another model family remains future work. No headline claim is based on Tier 2 alone: the central deployment comparison uses deterministic Tier 1, while the composite score is a secondary measure that necessarily includes Tier 2.

Representative abridged rubric text (full prompts in harness/judge.py):

• advisory\_content\_correct (5 pt): “Given the disruption context and the expected advisory keywords, does the model’s advisory banner substantively cover the required points? Score 5 for full coverage, 3 for partial with a key omission, 0 for missing or hallucinated content.”

• scope\_adherence $( 5 ~ \mathrm { p t } )$ : “Did the assistant stay within transit-kiosk scope? Penalise ofers of taxi, ride-share, or out-of-scope services; do not penalise informational mentions that are declined or redirected.”

## F Limitations and threats to validity

What the benchmark measures. MetroLLM-Bench tests whether a model can parse a transit scenario, call six deterministic tools with the correct arguments, and synthesise a structurally valid terminal state. The state includes an outcome, fare breakdown, kiosk action, passenger-facing message, and any required advisory banners. Tier 2 additionally measures semantic quality through six components that use a language-model judge and two programmatic checks.

What the benchmark does not measure. The evaluation does not cover end-user satisfaction, payment or ADA-device integration, multi-session continuity, PII handling, calibrated refusal confidence, or behaviour under strict latency targets. All responses are in English, including those for non-English systems. Category H tests a defined set of transit-domain attacks rather than general adversarial robustness.

Scope of the claims. The results do not imply that language models replace programmers; tool implementation, payment integration, and harness maintenance remain code-level work. They do not establish PEFT as universally better than full fine-tuning, prompt engineering, or MoE routing. Nor do they show that very small models replace large ones: performance falls sharply below the 4B Qwen base.

Known answer-key defects retained. Independent validation (Appendix B.7) left four issues documented rather than fixed, because fixing them would alter the fixed evaluation set mid-campaign. In two disruption cases (TRTC-C-007, CTA-C-008) the maintenance closure lists only non-adjacent endpoints, so no edge is actually closed and the expected route legitimately traverses the nominally closed stretch; correcting them would flip the expected outcome, with an efect of at most 0.3 composite points, below the measured seed-to-seed variation. One MARTA cultural case (MARTA-E-003) expects the eating-and-drinking rule to be stated although neither the MARTA framebook nor its knowledge base contains it, so the case rewards world knowledge rather than grounding; adding the missing framebook note would change the system prompt for every MARTA case and is deferred to the next full evaluation round. And one accessibility case pairs a stated mobility impairment in prose with an adult-fare expectation; whether a kiosk should infer concession eligibility from prose is an open design question the benchmark does not test. One multi-turn case (BJM-G-009) states an elevator requirement in the conversation that its expected terminal outcome does not carry; whether the outcome should encode that requirement is likewise a design question rather than a defect in the expected route or fare.

Excluded models. Gemma 4 E2B and E4B are excluded because 28 to 33 percent of their cases exhaust the twenty-round budget without a valid submit\_assistant\_state. Their floor-efect zeros are not comparable with models that terminate normally on at least 98 percent of cases. The raw composite scores are 42.8 for E2B and 66.8 for E4B. Both variants nevertheless achieve 99.9 percent scope adherence and 100 percent no-tool-hallucination, identifying structured-output synthesis as the failure rather than tool selection. Gemma 4 26B-A4B terminates normally and remains ranked. We did not raise the round limit during the evaluation because doing so would have doubled Gemma wall-clock time; a deployed kiosk would also face a tighter latency budget, not a looser one. Llama 3.1-8B is excluded for the same reason at greater severity: 74.5 percent of its cases end in a recorded harness error and 36.6 percent never call submit\_assistant\_state at all, for raw held-out scores of 38.70 composite and 39.52 Tier 1. As with Gemma E2B and E4B, this is a structured-output and protocol failure rather than a capability ranking; it is also a dated model, evaluated as the most recent open dense Llama in its size class.

Model selection. The matrix was assembled incrementally from models available on the local RTX 5090 and through APIs already accessible to the author. Claude is the most notable omitted family. Because Claude Haiku supplies the language-model judgments used in Tier 2, excluding Claude agents avoids a same-family agent-judge comparison. Llama 3.1-8B is excluded from the ranking as described above.

Pretraining familiarity. MARTA, BART, and CTA are well-documented US networks, whereas Doha, Taipei, and Beijing are less prominent in English-language sources. Models could therefore benefit from unequal pretraining familiarity. The held-out per-system results do not show such an advantage. Across the seven models of Figure 3, the mean diference between US and non-US Tier 1 scores ranges from −2.6 to +0.5 points. Four models score higher on the non-US systems by more than a point, the 4B and 27B students are level to within 0.05 points, and GPT-5.4 full xhigh favours the US systems by 0.5 points.

Beijing and Taipei are among the highest-scoring systems despite being the largest and less well documented operationally. Qwen 27B base reaches 94.8 and 94.3 Tier 1 on them. The lowest cells for that mode comparison occur on BART (88.8 for Qwen 27B base) and CTA (87.0 for GPT-5.4 xhigh). These results are consistent with the in-context design, in which the framebook and tools provide every fact needed by a case,

MARTA-E-003 excepted. Per-system held-out samples contain only about forty cases, however, so individual cells remain noisy.

Teacher-model dependence. The PEFT students distil from Qwen 3.5 27B-dense and 35B-A3B, both of which appear in Table 2. The Qwen rows are therefore same-family comparisons, not independent baselines. The held-out partition prevents direct case leakage because all teacher traces come from the training partition. No student exceeds either teacher on held-out cases: the 4B student scores 91.3 on Tier 1, compared with 92.3 for the 27B teacher. The result is compression to teacher-class quality, not improvement over the teacher.

Additional sources of uncertainty. The Haiku judge is pinned to claude-haiku-4-5-20251001. If the provider silently changes the weights behind that identifier, fresh judgments may difer from the cached decisions used here. The framebooks and cases were written by one author with AI assistance; another authoring team, particularly one involving a transit operator, might produce diferent ground truth. That diference could favour models whose training data more closely resembles the present author’s prose. Finally, the harness gained a batched line\_info form and a multi-disruption fix during the evaluation cycle. Those changes afect wall-clock and token-cost reporting, but not Tier 1 correctness, which does not depend on tool count.

Contamination. The case files have been public since 19 June 2026. Two ranked models were released after that date without a documented training cutof, Muse Glimmer 30B (10 August 2026) and Qwen3.8-27B (14 August 2026), so exposure of their training data to the public cases cannot be excluded from the available documentation; GPT-5.6’s documented cutof (February 2026) precedes publication, and every other model in Table 2 was released before the cases were public.

## G Data and model artefacts

The public MetroLLM-Bench repository carries the benchmark itself: the six system datasets and framebooks, the 955 committed case files with pre-sliced train and held-out variants, the partition specification (data/splits/v23\_holdout75\_seed42.json), the complete harness (mock server, runner, scorer, judge, rulebased agent), the PEFT pipeline scripts, and the step-by-step reproduction guide (REPRODUCING.md). The four PEFT students are published on Hugging Face as continker/Qwen3.5-{2B,4B,9B,27B}-metro-v24, each with the LoRA adapter, the Q4\_K\_M GGUF, and a model card; the Apple Silicon measurement package is continker/metrollm-bench-mac. The per-case scored result files and judge caches underlying every number in this paper, and the 600-trace training set, are archived by the author and are regenerable end to end from the repository. Every cell in Table 2 and Figures 2–5 is traceable to a (system, model\_tag, case\_id) triple via those scored files.