# ScienceArena: Benchmarking LLMs on Latest Scientific Olympiad Competitions

Guangxiang Zhao<sup>1,ð</sup>, Qilong Shi<sup>2,ð</sup>, Xusen Xiao<sup>3,ð</sup>, Wenpu Liu<sup>4</sup>, Yaoming Li<sup>4,B</sup>, Linfeng Hao<sup>4</sup>, Shuyang Hou<sup>4</sup>, Zijian Guo<sup>4</sup>, Xinrui Zhang<sup>4</sup>, Yuntian Zhao<sup>4</sup>, Zhengyang Wang<sup>4</sup>, Wenrui Liu<sup>4</sup>, Yuhan Wu<sup>4</sup>, Tong Yang<sup>4,B</sup>, Lin Sun<sup>1,B</sup>, Xiangzheng Zhang<sup>1</sup> <sup>1</sup>Qiyuan Tech, <sup>2</sup>Tsinghua University, <sup>3</sup>The University of Hong Kong, <sup>4</sup>Peking University

<sup>B</sup> Correspondence: yibachenggong@gmail.com, yangtong@pku.edu.cn, sunlin1@360.cn

## Abstract

Benchmark saturation and data contamination increasingly obscure genuine scientific reasoning in frontier LLMs. We introduce SCI-ENCEARENA, an olympiad-style benchmark from thirteen public science competitions in physics, chemistry, and biology, including IPhO and IChO 2025–2026, IBO 2023, US-APhO 2026, and USNCO 2025. Its openended, multi-step problems use process-credit rubrics, making faithful scoring difficult. We build ScienceArena through an expert-audited digitization pipeline that converts official exams, figures, solutions, and rubrics into structured items verified by olympiad medalists. To scale evaluation beyond costly human grading, we calibrate LLM-as-judge against medalist ground truth on archived answers from five models across IPhO and IChO; two strong judges stay within one point of expert total scores. Medalist notes show that failures often stem from visual grounding, structure fidelity, and global problem control rather than missing terminology. Evaluating fourteen recent LLMs with interleaved solving, we find that top models obtain medal-equivalent rubric scores on several public international exams, while chemistry and long-horizon consistency remain key bottlenecks. We provide an interactive demo<sup>\*</sup>.

## 1 Introduction

Scientific reasoning has become a central target for frontier LLMs, yet evaluation is at an impasse. Popular benchmarks are increasingly saturated and vulnerable to contamination (Hendrycks et al., 2021; Li et al., 2023; Rein et al., 2023; White et al., 2024; Phan et al., 2025). Recent live or frontier benchmarks respond by using time-sensitive sources, contest problems, or newly authored expert questions (Jain et al., 2024; Glazer et al., 2024; Balunovic´ et al., 2025), but scientific reasoning remains difficult to measure when answers are long, visual, and partially correct. At the same time, many science evaluations remain confined to short-answer or multiple-choice formats, which cannot fully test the abilities needed in real scientific problem solving: long derivations, diagram and structure interpretation, multi-step consistency, and partial credit under explicit rubrics (Wang et al., 2023; Lu et al., 2023; Yue et al., 2023; He et al., 2024; Lu et al., 2022; Laurent et al., 2024; Mirza et al., 2025).

Science olympiad competitions are a natural high-resolution testbed. They are written by domain experts, released publicly after the contest, and accompanied by official answers, detailed solutions, and scoring schemes. They also provide interpretable human reference points, including gold-medal thresholds and winner scores. However, turning olympiad exams into a reliable LLM benchmark is not automatic. The source materials are PDFs with formulas, figures, tables, and chemical structures; more importantly, faithful scoring requires stepwise rubrics and expert judgment rather than simple exact matching.

We present SCIENCEARENA, an olympiad-style benchmark and evaluation protocol for scientific reasoning. The suite contains thirteen physics, chemistry, and biology contest-year datasets released in 2023–2026. An archived five-model subset covers IPhO 2025, IChO 2025, and IBO 2023; the full evaluation also includes international exams such as IPhO and IChO 2026 and nationallevel exams such as USAPhO and USNCO. We standardize the data construction pipeline: official public materials are collected, converted into structured JSON with OCR and multimodal parsing, and then audited by olympiad medalists for problem text, figure descriptions, answers, detailed solutions, rubrics, and point totals. This preserves the original exam structure while making the problems readable to text-based and multimodal LLM APIs.

<table><tr><td>Benchmark</td><td>Main signal</td><td>ScienceArena distinction</td></tr><tr><td>Chatbot Arena; Arena-Hard</td><td>Human or judge preference</td><td>Tests verifiable scientific correctness rather than broad helpfulness.</td></tr><tr><td>OlympicArena/OlympiadBench</td><td>Cross-discipline hard reasoning</td><td>Keeps recent science exams, release dates, official rubrics, and medalist audits.</td></tr><tr><td>MathArena</td><td>Fresh math answers/proofs</td><td>Adds physics, chemistry, biology, diagrams, structures, and science process credit.</td></tr><tr><td>HiPhO/PhyArena/PhysicsArena</td><td>Physics olympiad or skill axes</td><td>Extends olympiad-style evaluation across three sciences with one unified protocol</td></tr><tr><td>SUPERChem; BABE</td><td>Domain research or expert tasks</td><td>Uses official competitions with comparable scoring scales and medal references.</td></tr><tr><td>SciArena</td><td>Researcher preference</td><td>Scores against official answers and rubrics, then analyzes medalist grading notes.</td></tr><tr><td>SCIENCEARENA</td><td>Release-aware olympiad scoring</td><td>Combines official credit rubrics, calibrated judges, medals, and expert diagnostics.</td></tr></table>

Table 1: Positioning of SCIENCEARENA among representative Arena-style and domain-specific evaluations. The comparison highlights the niche targeted by this paper: release-aware science olympiad evaluation with official process-credit rubrics and medalist-audited diagnostics.

Table 1 positions SCIENCEARENA relative to recent Arena-style and domain-specific evaluations. General-purpose arenas such as Chatbot Arena and Arena-Hard emphasize human or judge-model preferences over open-ended assistant behavior (Chiang et al., 2024; Li et al., 2024). Olympiad and competition benchmarks provide harder reasoning tasks, including OlympicArena, OlympiadBench, and MathArena (Huang et al., 2024; He et al., 2024; Balunovic et al.´ , 2025). The closest physics-only line is HiPhO/PhyArena, which uses latest physics olympiad exams and official medal thresholds (Yu et al., 2025); other domain benchmarks decompose physics, chemistry, biology, or scientific literature reasoning with different task sources and scoring signals (Dai et al., 2025; Zhao et al., 2025b; Zhou et al., 2026; Zhao et al., 2025a). Our focus is complementary: recent public olympiad exams across physics, chemistry, and biology, official process-credit rubrics, medal-based human reference points, medalist-audited construction, and expert-calibrated automated grading.

A second challenge is evaluation cost. Human olympiad experts provide the most reliable ground truth, but repeatedly asking them to grade every new model is impractical. We therefore develop an expert-calibrated LLM-as-judge system. Using human medalist scores on IPhO/IChO 2025 as ground truth, we calibrate the full judge system, including rubric packaging, structured scoring, score validation, and aggregation, on archived answers from five models. The resulting judge system follows recent rubric-based LLM evaluation practice (Zheng et al., 2023; Rao and Callison-Burch, 2026) and matches expert total scores within ±1 point across the two exams, making automated scoring practical while retaining an explicit human validity check.

Finally, we examine how LLMs should solve long olympiad problems. One-shot prompting asks for an entire multi-part solution at once, while interleaved prompting presents subparts sequentially and preserves prior context. We do not claim interleaving as a new method; rather, we test this natural prompting choice and find that it generally improves performance and better resembles how users interact with LLMs on difficult problems. We therefore use interleaved prompting for the main evaluation. Across fourteen recent LLMs, the strongest systems now obtain medal-equivalent rubric scores on several public international exams. The gains are uneven: biology appears closest to saturation, whereas chemistry remains challenging because of molecular structures, stereochemistry, spectra, and rubric-sensitive drawings; physics still exposes failures in visual grounding and long-horizon coherence. Beyond aggregate scores, we study the written comments produced by olympiad medalist graders. These notes are unusually diagnostic: they distinguish fluent but physically misgrounded derivations from correct problem models, separate chemically plausible prose from exact structure commitments, and expose when partial credit depends on an explicit intermediate rather than a vague explanation. We use these comments to build the subfield and ability-boundary analysis in Section 3.6 and Appendix C.

Contributions. We make four contributions:

• We build SCIENCEARENA, an expert-audited benchmark of thirteen science olympiad competitions with questions, figures, official answers, detailed solutions, and scoring rubrics.

• We define a reusable pipeline for converting official olympiad PDFs into structured LLM-readable benchmark items.

• We develop an expert-calibrated LLM-as-judge system that closely matches olympiad-medalist grading on IPhO & IChO calibration experiments.

• We compare one-shot and interleaved prompting, evaluate fourteen recent LLMs under the stronger protocol, and distill comments from human experts into diagnostics of persistent domainspecific bottlenecks.

![](images/9ef550ec0b411a272e432a6ee01aa583d09b10ea6216dbbc07794d931a2a43a7.jpg)  
Figure 1: SCIENCEARENA benchmark construction pipeline.

<table><tr><td>Benchmark</td><td>Subject</td><td>Level</td><td>Rel.</td><td>Scale</td></tr><tr><td>IPhO 2025</td><td>Physics</td><td>International</td><td>25-07</td><td>30</td></tr><tr><td>IChO 2025</td><td>Chemistry</td><td>International</td><td>25-07</td><td>60</td></tr><tr><td>IBO 2023</td><td>Biology</td><td>International</td><td>25-08</td><td>453</td></tr><tr><td>IPhO 2026</td><td>Physics</td><td>International</td><td>26-07</td><td>30</td></tr><tr><td>IChO 2026</td><td>Chemistry</td><td>International</td><td>26-07</td><td>60</td></tr><tr><td>APhO 2025</td><td>Physics</td><td>Continental</td><td>25-05</td><td>30</td></tr><tr><td>EuPhO 2025</td><td>Physics</td><td>Continental</td><td>25-06</td><td>30</td></tr><tr><td>NBPhO 2026</td><td>Physics</td><td>National</td><td>26-04</td><td>72</td></tr><tr><td>USAPhO 2026</td><td>Physics</td><td>National</td><td>26-05</td><td>150</td></tr><tr><td>CPhO 2025</td><td>Physics</td><td>National</td><td>25-10</td><td>320</td></tr><tr><td>INChO 2026</td><td>Chemistry</td><td>National</td><td>26-01</td><td>106</td></tr><tr><td>USNCO 2025</td><td></td><td></td><td></td><td></td></tr><tr><td>CChO 2025</td><td>Chemistry Chemistry</td><td>National National</td><td>25-06 25-10</td><td>100 100</td></tr></table>

Table 2: SCIENCEARENA benchmark sources. Release dates use YY-MM. “Scale” is the full score; rubric status and source caveats are retained in the released metadata.

## 2 The ScienceArena Framework

SCIENCEARENA is designed around two constraints that are often absent from standard LLM benchmarks. First, the source tasks must remain fresh, public, and scientifically meaningful. Second, grading must respect olympiad rubrics: a wrong final value may still receive credit for a correct setup, while a correct-looking answer may receive little credit if the derivation or diagram is unsupported. This section describes the benchmark construction pipeline, the calibrated judge system, and the solving protocols used for the experiments.

## 2.1 Benchmark Construction

SCIENCEARENA currently contains thirteen public olympiad-style competitions across physics, chemistry, and biology (Table 2). International competitions provide medal thresholds; national-level competitions expand topic diversity and reduce dependence on one exam style. For every benchmark, we record the public release date and preserve the original hierarchy: each multi-part problem becomes linked turns with text, figure context, official answer, rubric, and maximum score. This keeps rubrics intact while supporting automated evaluation.

Digitization pipeline. Olympiad materials are PDFs with equations, plots, molecule drawings, apparatus diagrams, answer images, and marking schemes, so we convert them through a five-stage pipeline. A curator first collects official public problems, figures, answer keys, solutions, and marking schemes; pages are then rendered and parsed with OCR/multimodal models. The parsed content is normalized into fields such as question\_text, official\_answer, official\_detailed\_solution,

official\_rubric, max\_score, release\_date, and source provenance. Subject medalists audit completeness, figure alignment, answer fidelity, and point totals; for IPhO and IChO, the groundtruth graders are listed in Table 3. The final export is a unified interleaved dataset with one row per subproblem turn. Figure 1 summarizes the intended five-stage construction schematic.

The audit trail is important because the three sciences stress different failure modes. Physics often requires plot reading and symbolic carry-over; chemistry depends on reaction schemes, stereochemistry, spectra, and crystal structures; biology is more often objective-keyed but can involve long passages and diagrams. Preserving sources, parsed text, image-derived descriptions, and audit notes makes these differences inspectable rather than hiding them behind one score.

<table><tr><td>Track</td><td>Expert 1</td><td>Expert 2</td></tr><tr><td>Physics</td><td>Physics Olympiad gold medalist world top 10</td><td>Physics Olympiad gold medalist world top 10</td></tr><tr><td>Chemistry</td><td>gold medalist national top 30</td><td>Chemistry Olympiad Chemistry Olympiad gold medalist national top 30</td></tr><tr><td>Biology</td><td>Biology Olympiad gold medalist national top 30</td><td>Biology Olympiad gold medalist national top 30</td></tr></table>

Table 3: Human experts recruited for ground-truth grading. The two experts for each exam jointly graded model answers and resolved rubric-level scores.

![](images/7aa8b59739655a4c55e07cf0af50b99727f9f273178e6da79c37df74f171147c.jpg)  
Figure 2: Calibration of the LLM-as-judge system against human expert ground truth. The blue bar is the original score assigned by the human experts in Table 3; the other bars show the two judge models.

## 2.2 Expert-Calibrated LLM-as-Judge System

Human olympiad experts are the strongest graders for open-ended science problems, but they cannot regrade every model update. We therefore use medalist grading as a calibration anchor and build a reusable LLM-as-judge system for large-scale evaluation. Table 3 summarizes the experts recruited by our team for this human ground-truth calibration.

The judge is a pipeline, not a single prompt. For each non-objective turn, it receives the staticized question, official answer, detailed solution, rubric, maximum score, and candidate response; here staticized denotes a frozen textual rendering of the question-side problem content, including any visual evidence needed by a text-only route. It acts as a strict olympiad examiner, assigns partial credit, and returns JSON with total\_score, max\_score, sub\_scores, evidence, deductions, and confidence. Post-processing validates JSON, repairs common LaTeX escape errors, clamps impossible scores, checks maximum scores, and aggregates turns; objective-key items such as IBO are scored deterministically. Each score remains traceable to a turn packet, rubric item, and evidence field, so expert-identified blind spots can be fixed locally.

For the four 2026 competition columns in Table 6, we use a stricter sealed implementation with a fixed Gemini 3.5 Flash judge route. The judge receives the candidate response, provenance-labeled official answer and solution, the exact rubric criterion identifiers and maxima, and the original question/solution page PNGs directly when the frozen item supplies them. It must return bounded decimal sub-scores, evidence quotations, and an arithmetic-consistent total. Validation rejects rather than clamps an invalid response and stores one initial call plus at most six retries in a contiguous ledger; an exact solver-no-answer sentinel is deterministically scored zero without a judge call.

![](images/c57f64470bb40a7dd50a6e91c273f6853e88ac5977448fbea16753a6bb9ccd3a.jpg)

Figure 3: One-shot vs. interleaved solving schematic.
<table><tr><td>Model Full Scores</td><td>Protocol</td><td>IPhO 30.0</td><td>IChO 60.0</td><td>CChO 100.0</td><td>CPhO 320.0</td></tr><tr><td>Gemini 3.5 Flash</td><td>One-shot Interleaved</td><td>22.84 25.67</td><td>45.52 48.43</td><td>52.87 57.12</td><td>234.1 271.6</td></tr><tr><td>Gemini 3.1 Pro</td><td>One-shot Interleaved</td><td>24.19 27.30</td><td>42.02 46.45</td><td>50.94 57.44</td><td>242.4 302.1</td></tr><tr><td>GPT-5.5</td><td>One-shot Interleaved</td><td>20.80 25.63</td><td>33.52 33.98</td><td>39.48 46.59</td><td>285.6 289.5</td></tr><tr><td>Claude Opus 4.7</td><td>One-shot Interleaved</td><td>19.90 22.73</td><td>31.15 41.68</td><td>36.45 49.61</td><td>251.0 273.0</td></tr></table>

Table 4: One-shot versus interleaved solving on openended runs across four models and four contest. Bold marks the higher score for each model–exam pair. Interleaved prompting wins all 16 displayed comparisons.

We calibrate the system on archived, nonregenerated answers from five models for IPhO 2025 and IChO 2025. Gemini 3.5 Flash and Gemini 3.1 Pro both stay within one point of the medalist total score for every model–exam pair (Figure 2). Item-level replay further shows strong correlations with expert scores, indicating that the close totalscore agreement is not driven solely by cancellation across items. The final judge reached this agreement only after explicitly packaging expert rationales, requiring structured sub-scores, and preserving post-processing rules for known edge cases such as surface-similar chemistry answers and tiny subpart allocations. The resulting judge is a scalable proxy for medalist grading only when the official solution and rubric are available and correctly digitized. It shifts the expensive human work from repeatedly grading every model to validating benchmark items and calibrating the judge.

## 2.3 Solving Protocols

Physics and chemistry exams often contain dependent subparts, so we compare two prompting protocols. In one-shot solving, the model receives the full problem and writes one complete solution. In interleaved solving, it answers one subpart at a time while previous turns remain in context. Interleaving is not a new algorithm; it is a natural chat interaction pattern whose effect we test directly.

Figure 3 illustrates the two protocols, and Table 4 reports the archived comparison. Interleaving wins all 16 displayed model–exam comparisons. We use interleaved prompting for main open-ended evaluations; IBO is excluded because it is objectivekey scored. If a solver response is empty, incomplete, or non-concrete—for example, because the reasoning-token limit is reached—the request is retried up to five times, and the first valid response is retained. If all five retries fail, the turn is marked as unsuccessful and this marker remains in the context for subsequent subparts. No correction is provided, and error-forward credit is awarded only when the response contains a concrete intermediate result.

## 3 Experimental Results and Diagnostics

## 3.1 Main Evaluation Setting

We evaluate fourteen recent models from major model families on the thirteen-benchmark SCIENCEARENA suite. The model set includes the Gemini 3/3.5 family, GPT-5.4/5.5, Claude Opus 4.7, Doubao Seed 2.0 Pro, Qwen3.6/3.7, DeepSeek V4, MiniMax M2.7, GLM-5.1, and Xiaomi MiMo. Model release dates are taken from official announcements when available—e.g., Gemini 3 Flash, Gemini 3.1 Pro, Gemini 3.5 Flash, GPT-5.4/5.5, Claude Opus 4.7, and Qwen3.7-Max (Google, 2025a, 2026a,b; OpenAI, 2026a,b; Anthropic, 2026; Qwen Team, 2026)—and otherwise from public availability reports or the provider registry snapshot used by our evaluation scripts.

Nine benchmark columns use a uniform textonly interleaved protocol. For the four 2026 competition columns in Table 6, input routing follows the verified capability of each exact API route: eight vision-capable routes receive native question text together with the exact original question-page PNG bytes directly in the solver API call, never an image description generated by another model; six genuine text-only routes receive an answerblind question-only staticization and zero image blocks. The text-staticization candidates are generated from question text and question-page images by Gemini 3.1 Pro, then reviewed item by item and corrected when needed before freezing; they contain no official answer, solution, or rubric. Because input mode is necessarily coupled to route capability, the four-column block is a capability evaluation rather than a controlled modality ablation. Open-ended physics and chemistry turns are graded by the calibrated LLM-as-judge system from Section 2.2. IBO is the exception: it is objective-key scored and is evaluated with the dedicated IBO runner over the original keyed items rather than by an LLM judge.

<table><tr><td>Metric Full score</td><td>IPhO 25 30.0</td><td>IChO 25 60.0</td><td>IBO 23 453.0</td></tr><tr><td>Human References a Human No.1 Human Gold Human Silver</td><td>29.2 23.4</td><td>57.1 36.6</td><td>395.4 354.7</td></tr><tr><td>Human Bronze</td><td>17.0 11.5</td><td>28.7 26.3</td><td>293.1 249.7</td></tr><tr><td colspan="4">Archived Five-Model Performance</td></tr><tr><td>Gemini 2.5 Pro</td><td>24.1</td><td>0 36.7</td><td>P 404.0</td></tr><tr><td>GPT-5</td><td>BO 21.8</td><td>31.8</td><td>395.0</td></tr><tr><td>GLM 4.5</td><td>BO 21.2</td><td>19.8</td><td></td></tr><tr><td></td><td></td><td></td><td>329.0</td></tr><tr><td>DeepSeek V3.1</td><td>BO 19.4 15.2</td><td>26.6 25.9</td><td>BO 312.0</td></tr></table>

Table 5: Human reference scores used for medal annotations on international competitions. IPhO and IChO use theory-only scores; IBO uses the objective-key total used by our evaluation runner. All five models were released before the start of the competition.

For international competitions, we annotate model scores with human reference medals. The references are theory-score aggregates from the official rankings: IPhO 2025 winner/gold/silver/bronze are 29.2/23.4/17.0/11.5 out of 30; IChO 2025 are 57.1/36.6/28.7/26.3 out of 60; IBO 2023 are 395.4/354.7/293.1/249.7 out of 453. We do not assign medals to national competitions, because national selection pools and award rules are not directly comparable across countries.

## 3.2 Overall Trends

Table 6 shows that frontier models now reach extremely high scores on several olympiad-style exams, but the picture is not uniform. Table 5 provides the human reference points used to interpret the three international columns in the main table. On IPhO 2025, Gemini 3.1 Pro reaches 29.45/30, slightly above the best human theory score in the official ranking. Several other models, including Gemini 3.5 Flash, Gemini 3 Flash, Qwen3.7-Max, GLM-5.1, and DeepSeek V4 Pro, clear the IPhO gold reference. On IChO 2025, the best current score is Gemini 3.1 Pro at 48.95/60: well above the gold threshold but still below the human winner score of 57.1. On IBO 2023, the current best models exceed the human-winner reference, with the Gemini family occupying the top three scores in our objective-key run. Thus, the strongest current models obtain gold-medalequivalent rubric scores across physics, chemistry, and biology on these public benchmarks, but exceeding the human-winner reference is domaindependent: it occurs on IPhO and IBO in this suite, while the hardest IChO structures still leave a substantial gap to the human winner.

<table><tr><td rowspan="2">Contest Release Max. Score</td><td rowspan="2">IBO 25-08 453</td><td colspan="3">APhO EuPhO USNCO</td><td rowspan="2">IPhO 25-07</td><td rowspan="2">IChO</td><td colspan="7">CPhO CChO INChO</td><td rowspan="2">IChO Weighted Avg.</td></tr><tr><td>25-05 30</td><td>25-06 30</td><td>25-06 100</td><td>25-07 60</td><td>25-10 320</td><td>25-10 100</td><td>26-01 106</td><td>NBPhO USAPhO 26-04 72</td><td>IPhO 26-05 26-07 30</td><td>26-07 60</td></tr><tr><td>Gemini 3 Flash (25-12)</td><td>428.00</td><td>26.75</td><td>23.60</td><td>93.50</td><td>25.20</td><td>44.74</td><td>300.50</td><td>59.55</td><td>81.50</td><td>46.80</td><td>150.00</td><td>28.00 1</td><td>50.35</td><td>83.6 (#3)</td></tr><tr><td>Seed 2.0 Pro (26-02)</td><td>415.00</td><td>21.72</td><td>13.60</td><td>72.30</td><td>8 18.92</td><td></td><td>37.14 243.70</td><td>53.18</td><td>87.20</td><td>35.60</td><td>147.00</td><td>20.75</td><td>40.50</td><td>69.4 (#11)</td></tr><tr><td>Gemini 3.1 Pro (26-02)</td><td>D 431.00</td><td>28.50</td><td>22.50</td><td>97.00</td><td>29.45</td><td>48.95</td><td>303.50</td><td>72.28</td><td>95.95</td><td>48.30</td><td>146.00</td><td>29.75</td><td>山 54.18</td><td>88.7 (#1)</td></tr><tr><td>GPT-5.4 (26-03)</td><td>1 358.00</td><td>22.01</td><td>14.30</td><td>76.90</td><td>20.94</td><td>31.70</td><td>235.30</td><td>47.79</td><td>83.50</td><td>37.60</td><td>145.00</td><td>21.95</td><td>31.87</td><td>67.3 (#12)</td></tr><tr><td>GLM-5.1 (26-04)</td><td>349.00</td><td>28.40</td><td>8.20</td><td>90.00</td><td>26.11</td><td>38.82</td><td>283.00</td><td>58.61</td><td>85.75</td><td>48.60</td><td>150.00</td><td>29.75</td><td>48.54</td><td>78.2 (#6)</td></tr><tr><td>Qwen3.6-Plus (26-04)</td><td>413.00</td><td>27.56</td><td>13.30</td><td>88.20</td><td>22.87</td><td>36.22</td><td>252.80</td><td>49.33</td><td>79.25</td><td>46.80</td><td>149.25</td><td>27.25</td><td>45.12</td><td>75.8 (#8)</td></tr><tr><td>MiniMax M2.7 (26-04)</td><td>B 337.00</td><td>22.56</td><td>8.20</td><td>71.80</td><td>19.64</td><td></td><td>26.06184.70</td><td>43.78</td><td>82.75</td><td>35.70</td><td>139.50</td><td>27.00</td><td>27.51</td><td>62.7 (#14)</td></tr><tr><td>Claude Opus 4.7 (26-04)</td><td>回 407.00</td><td>24.19</td><td>16.50</td><td>75.20</td><td>22.29</td><td>44.49</td><td>262.00</td><td>56.59</td><td>82.70</td><td>48.70</td><td>132.50</td><td>26.05</td><td>1 45.99</td><td>75.8 (#9)</td></tr><tr><td>MiMo V2.5 Pro (26-04)</td><td>317.00</td><td>21.33</td><td>11.70</td><td>74.00</td><td>18.97</td><td>28.94</td><td>248.10</td><td>50.11</td><td>86.00</td><td>26.30</td><td>139.50</td><td>22.80</td><td>1 29.83</td><td>63.8 (#13)</td></tr><tr><td>GPT-5.5 (26-04)</td><td>415.00</td><td>21.50</td><td>19.00</td><td>75.40</td><td>22.08</td><td>35.77</td><td>270.30</td><td>53.30</td><td>83.00</td><td>59.20</td><td>150.00</td><td>29.75</td><td>55.12</td><td>78.8 (#5)</td></tr><tr><td>DeepSeekV4-Pro (26-04)</td><td>327.00</td><td>26.01</td><td>13.30</td><td>90.00</td><td>25.10</td><td>35.82</td><td>257.70</td><td>49.78</td><td>89.75</td><td>51.40</td><td>148.25</td><td>29.75</td><td>48.98</td><td>77.1 (#7)</td></tr><tr><td>DeepSeekV4-Flash (26-04)</td><td>à 324.00</td><td>20.44</td><td>20.40</td><td>94.50</td><td>20.53</td><td>38.33</td><td>182.90</td><td>46.46</td><td>83.80</td><td>51.50</td><td>150.00</td><td>27.75</td><td>1</td><td>49.46 74.1 (#10)</td></tr><tr><td>Gemini3.5-Flash (26-05)</td><td>6 430.00</td><td>27.91</td><td>16.80</td><td>95.70</td><td>26.41</td><td>47.98</td><td>307.00</td><td>70.60</td><td>92.25</td><td>49.30</td><td>148.50</td><td>29.75</td><td>D 54.91</td><td>86.1 (#2)</td></tr><tr><td>Qwen3.7-Max (26-05)</td><td>349.00</td><td>28.10</td><td>15.60</td><td>93.00</td><td>26.52</td><td>41.28</td><td>277.50</td><td>52.66</td><td>83.00</td><td>46.80</td><td>148.00</td><td>29.75</td><td></td><td>46.68 79.3 (#4)</td></tr></table>

Table 6: Main evaluation on thirteen olympiad-style benchmarks. Dates use YY-MM, and rows use provider family/API dates. The red stair-step marks a descriptive provider-date boundary, including the boundary between the May 2026 model rows and the July 2026 IPhO/IChO columns; it is not proof of absent training exposure because exact evaluated-route availability and training cutoffs are unavailable. The four 2026 columns are therefore reported as public-benchmark capability measurements. Medal icons use the international reference cutoffs in Table 5 and are shown only for IPhO 2025, IChO 2025&26, and IBO. For other competitions, gold, blue, and green mark first, second, and third place within the column. Weighted average normalizes each score to 0–100 and averages them.

Across the full suite, Gemini models dominate the most stable comparisons. Using equalweighted benchmark averages over models with complete coverage, Gemini 3.1 Pro is strongest in physics (89.5% of full score), chemistry (86.3%), and biology (431.00/453, 95.1%). The result is similar under point-weighted aggregation, where Gemini 3.1 Pro also remains first in physics after the 2026 results are included. The three Gemini variants take every top-three position on IChO 2025 and IBO, and two of the three top positions on IPhO 2025. OpenAI and Anthropic models are competitive on several individual exams–GPT-5.5 leads across the four 2026 columns in Table 6 and Claude Opus 4.7 remains strong on IChO 2025– but neither matches the Gemini 3.1/3.5 pair across the full suite. The strongest non-Gemini outliers are informative: Qwen3.7-Max is second on IPhO, GLM-5.1 is near the top on APhO/USAPhO, and

DeepSeek V4 Flash is surprisingly strong on US-NCO despite weaker results on physics tracks.

The red release screen is essential for interpreting these numbers, but it is deliberately descriptive. It compares documented provider family/API dates with public exam-material months; it does not establish the first availability date or backend revision of the exact evaluated 360 route, and pre-release timing alone would not prove absence from training data. Post-release and temporally unknown cells therefore measure current capability under a public benchmark, not a strict time-release holdout. The only narrow release-based holdout we identify is Gemini 2.5 Pro: its stable 17 June 2025 release predates the IChO (10 July) and IPhO (21 July) theory examinations, and expert grading yields 36.71/60 and 24.1/30, just above the respective gold references of 36.6 and 23.4 (Google, 2025b; IChO 2025 Committee, 2025b; IPhO 2025 Committee, 2025). We do not extend this observation to IBO or treat release timing as proof of absent training exposure. All 182 model–benchmark cells in Table 6 are populated; its four 2026 columns contribute 56 complete cells, with no imputation for route, generation, or judge failures.

The sealed four-contest block comprises NBPhO, USAPhO, IPhO, and IChO 2026 under the protocol described above. GPT-5.5 has the highest four-contest normalized macro score at 93.31%, followed by Gemini 3.5 Flash at 89.54% and Gemini 3.1 Pro at 88.47%. This four-contest ranking is not the same estimand as the thirteen-column average: it covers only those four sealed 2026 contests, whereas the final column of Table 6 also includes nine text-only benchmark measurements. Appendix Table 7 reports every route’s four-contest macro score, sequence-cluster interval, and input assignment. Readers can use its V/T column to identify whether an exact model route was evaluated with native images (V) or as text-only (T).

<table><tr><td>Contest metric</td><td>Gemini 3.5 Flash</td><td>GPT-5.5</td><td>Claude Opus 4.7</td></tr><tr><td>IPhO mean (/30)</td><td>28.75</td><td>29.75</td><td>26.75</td></tr><tr><td>IPhO SD</td><td>1.00</td><td>0.00</td><td>1.32</td></tr><tr><td>IPhO range</td><td>2.00</td><td>0.00</td><td>2.50</td></tr><tr><td>IChO mean (/60)</td><td>53.73</td><td>54.84</td><td>46.45</td></tr><tr><td>IChO SD</td><td>0.92</td><td>0.80</td><td>2.64</td></tr><tr><td>IChO range</td><td>1.77</td><td>1.50</td><td>4.95</td></tr></table>

Table 8: Contest-level stability over five independent end-to-end calls per exact model route. Each call regenerates all answers and re-runs the fixed Gemini 3.5 Flash judge; SD is the sample standard deviation.

## 3.3 Solver Response Stability Across Repeated API Calls

We repeat the complete solver-and-scoring call three times on IPhO and IChO 2026 while holding the exact API route, input packet, prompt, rubric, and fixed Gemini 3.5 Flash judge configuration constant. These recent, demanding open-ended exams provide a direct stress test of response stability under ordinary API use; the frozen-answer crossjudge experiment below isolates judge variation.

The three routes remain in the same highperformance regime across repeated calls. GPT-5.5 is especially stable (zero IPhO spread and a 1.50/60 IChO range); Gemini 3.5 Flash spans 2.00/30 and 1.77/60. Claude Opus 4.7 has the largest variation, yet its maximum span is 2.50/30 on IPhO and 4.95/60 on IChO. Thus the API-level contest scores are operationally repeatable, although stability is provider- and exam-dependent. In a separate frozen-answer experiment, exact three-repeat item agreement across six judges ranges from 90.11% to 96.70%; Appendix Table 12 reports the per-judge totals and isolates judge-family sensitivity.

## 3.4 Controlled Modality Ablation

To isolate input modality from route capability, we evaluate 12 verified vision-capable routes on 12 image-dependent subquestions (four each from IPhO 2025, IChO 2025, and IBO 2023) under native-image, faithful audited-description, and novisual-information conditions, with four repeats per arm. Open-ended physics and chemistry cells use the mean of three rubric judges, whereas IBO are scored deterministically against the objective key.

<table><tr><td>Condition</td><td>Mean normalized score</td></tr><tr><td>Native image</td><td>72.75%</td></tr><tr><td>Faithful textual description</td><td>71.73%</td></tr><tr><td>No visual information</td><td>49.85%</td></tr></table>

Table 9: Controlled modality ablation over contests.

Native images outperform no visual information by 22.90 percentage points (paired-bootstrap 95% CI [20.02, 25.77]), showing that visual evidence materially affects performance on this deliberately image-dependent panel. The overall native-image– description estimate is 1.02 points and statistically unresolved (95% CI [−1.20, 3.27]); chemistry nevertheless shows a 5.01-point native-image advantage (95% CI [0.57, 9.56]). These results also support a practical composition for text-only solvers: a multimodal captioner can first convert the image into a faithful, quality-controlled description, after which the text model recovers most of the nativeimage performance. This captioner–solver pipeline therefore provides strong multimodal problemsolving capability when direct vision input is unavailable, while native pixels remain preferable for structure-sensitive chemistry. Domain-specific estimates are reported in Appendix Table 11.

## 3.5 Subfield and Modality Diagnostics

Aggregate scores hide where models fail. We therefore classify each benchmark turn into a scientific subfield and skill bucket using a Gemini 3.5 Flash classification pass over the staticized question, official answer, and rubric. The classifier never sees model answers and does not change the benchmark. We then join these labels with clean turn-level grades to measure score rates by subfield, image dependence, structure dependence, and quantitative reasoning. Figure 5 summarizes five representative subfields for each domain; the full numeric table, including coverage counts, is in Appendix B. Across domains, representative failures share fragile binding between problem evidence, the scoring target, and the required answer form; Figure 4 gives one case per domain, with more examples in Appendix A. These coverage counts are model–turn grading instances rather than independent questions; slices with small counts are exploratory.

The strongest pattern is that chemistry is bottlenecked by structure fidelity. Physical chemistry is comparatively strong (78.4%), while stereochemistry is much lower (34.7%). Organic-structure questions are also substantially weaker (71.8%) than analytical/spectroscopy questions (86.6%).

![](images/012c7564363924d14d4b0dddabee0d878fcf4f1c966e6b5ec7391d9b32bb59b4.jpg)  
Figure 4: Representative cross-domain failure cases. The common theme is not absence of scientific terminology, but fragile binding between the problem evidence, the scoring target, and the final answer format. These cases expose errors that answer-only accuracy would often obscure. Additional cases appear in Appendix A.

More generally, chemistry turns that require structural diagrams score 64.5%, compared with 74.1% for chemistry turns without such requirements; visually grounded chemistry turns score 63.3%, compared with 83.1% for non-visual chemistry turns. This matches our qualitative inspection: models often know the relevant reaction logic but lose credit when they must commit to an exact connectivity, stereochemical relation, spectrum-to-structure inference, or diagram-derived intermediate. The error is rarely a simple lack of chemistry vocabulary. Instead, models often retrieve a plausible named mechanism or biosynthetic template, then fail to bind that template to the substrate, substituent positions, and stereochemical constraints in problems.

Physics shows a different and more globally coupled failure profile. Thermodynamics and astrophysics are the weakest physics slices in Figure 5, while mechanics, electromagnetism, and modern/ atomic physics are all in the 78–84% range. The visual/non-visual gap is small (77.6% vs. 79.7%). The main physics errors are usually global: a wrong early approximation or variable definition propagates through later subparts. Interleaving helps because each subpart receives focused attention, but it cannot fully repair a mistaken physical model once the conversation has anchored on it.

Biology is the highest-scoring international track. Seven models exceed the IBO human-winner reference, and the best score is Gemini 3.1 Pro at 431.00/453. This does not mean biology reasoning is solved: IBO is objectively keyed, which makes grading less fragile than process-rubric chemistry, and many questions reward careful elimination among constrained options. Within IBO, models are strongest on biochemistry (91.4%) and molecular genetics (85.8%), but weaker on experimentaldata interpretation (64.3%) and bioinformatics/statistics (76.6%). The remaining errors are less about missing factual knowledge and more about multi-statement consistency, quantitative interpretation, and careful reading of experimental setups.

## 3.6 Insights from Gold-Medalist Grading

The archived gold-medalist grading for five model submissions on IPhO/IChO 2025 separates scientifically fluent prose from contest-credit evidence. IPhO notes expose fragile global reasoning after strong local derivations, whereas IChO notes require explicit carbon skeletons, stereochemistry, and valid error-forward structures; these observations motivate our rubric gates for answer form, structure specificity, and subpart evidence. The provenance-preserving export contains 525 expertscored rows—410 with a nonempty medalist comment and 115 explicitly marked as having no comment—and the longer note archive appears in Appendix C. A deterministic replay of 1,050 stored rows from the two calibrated Gemini judges gives interjudge normalized MAE 0.093 and Pearson correlation 0.907 across 525 shared expert rows; this remains a narrow same-family diagnostic.

![](images/6da0aa0a6a5fbb045ee8c8a52d7fdf0167fd0ec6cab12145910614bb42409815.jpg)  
Figure 5: Subfield diagnostics aggregated over clean model–benchmark cells. Each radar uses five domain-specific axes, so polygon shapes should be read within a domain rather than across domains. Values are awarded points divided by available points. Together, the plots localize which kinds of scientific reasoning remain most fragile.

## 3.7 Judge Stability and Protocol Diagnostics

The human calibration in Figure 2 shows close agreement between the judge pipeline and medalist scores. Structure-heavy chemistry is the most demanding case because ambiguous candidate prose may not uniquely specify the required molecule; when resources permit, expert adjudication provides the strongest supplementary safeguard for low-confidence or structure-only responses.

Exact three-repeat item agreement ranges from 90.11% to 96.70% across the six judges. Among the four direct-PNG vision judges, the six judge pairs have mean normalized MAE 1.732 percentage points, mean Pearson correlation 0.888, and mean Spearman correlation 0.843. The two text-staticized judges are reported separately as modality-sensitivity conditions. Together with the expert calibration, these results demonstrate a repeatable, cross-family-stable judging pipeline under matched inputs. Per-judge totals and strictreplay details are reported in Appendix Table 12.

Table 4 favors interleaved prompting in all 16 displayed model–exam comparisons across four scientific examinations, providing consistent endto-end evidence for decomposing multi-part problems into focused turns while retaining relevant reasoning context throughout extended solutions. Interleaving approximates multi-turn use; tool, retrieval, and verifier-agent stacks also evaluate orchestration and therefore belong in a separate track.

## 4 Conclusion

SCIENCEARENA evaluates frontier LLMs on olympiad-style science problems with multi-step reasoning, public source provenance, and rubricfaithful scoring. The benchmark spans thirteen recent international, regional, and national competitions; preserves a common interleaved interaction structure while directly routing original images to vision-capable models in the four 2026 competition columns; calibrates LLM-as-judge against gold-medalist human grading; and supports a broad latest-model comparison with explicit release-time risk metadata. The results show that recent models can obtain medal-equivalent rubric scores on public international open-ended exams, with Gemini 3.1 Pro reaching the IPhO 2025 winner reference under the text-only interleaved protocol. On the IBO track, several recent models exceed the humanwinner score, while IChO remains below the best human contestant despite strong gold-level scores.

The diagnostic picture is as important as the headline scores. Physics errors concentrate in global regime control and graph/figure binding; chemistry errors concentrate in exact structure and stereochemistry; biology errors concentrate in multi-statement data consistency rather than factual recall. Human expert comments confirm that scientific fluency is not enough: contest credit depends on explicit, checkable commitments that match the rubric. Because many latest-model results are post-release relative to the public exams, we report them as transparent current-capability measurements rather than strict uncontaminated holdouts. Time-locked competitions, moleculeaware judging, and richer agentic protocols constitute complementary evaluation tracks; the simple interleaved API protocol provides a reproducible baseline for comparing model reasoning ability.

## Limitations

The benchmark spans thirteen competitions, but results may not generalize to all scientific curricula. We evaluate theory components only, since experimental sections require physical apparatus and proctoring. Expert grading is costly and currently covers judge calibration on IPhO/IChO. Across those contests, expert agreement and the cross-family stability study support the automated pipeline; contest-specific expert spot checks remain the strongest standard for high-stakes comparisons on additional contests. The three repeated calls jointly measure solver-generation and fixed-route judge variation rather than separating them. The frozen-answer cross-judge study isolates judge variation for one solver on two contests. Capabilityconditioned vision/text route comparisons jointly reflect route and modality; Table 9 provides the controlled input-modality estimate. Finally, our subfield taxonomy is model-assisted and may reflect classifier priors; alternative decompositions could yield different diagnostic slices.

## References

58th International Chemistry Olympiad Organizing Committee. 2026a. Icho 2026 official program. Official competition website. Accessed: 2026-08-30; event dates 10–19 July 2026; theoretical examination 15 July 2026.

58th International Chemistry Olympiad Organizing Committee. 2026b. Icho 2026 problems and solutions. Official competition website. Accessed: 2026- 08-30; official theoretical-question files report July 2026 file metadata.

American Association of Physics Teachers. 2026. 2026 u.s. physics team news. Official AAPT website. Accessed: 2026-08-29; exam and solution links posted under 2026-05-20.

Anthropic. 2026. Introducing claude opus 4.7. https: //www.anthropic.com/news/claude-opus-4-7. Accessed: 2026-05-26.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola´ Jovanovic, and Martin Vechev. 2025.´ Matharena: Evaluating llms on uncontaminated math competitions. arXiv preprint arXiv:2505.23281.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. 2024. Chatbot arena: An open platform for evaluating llms by human preference. arXiv preprint arXiv:2403.04132.

China Chemistry Olympiad (CChO). 2025. 39th china chemistry olympiad (final) 2025: Theory exam paper, reference answers, and marking guidelines (public pdf). Online (PDF). Accessed: 2026-01-05.

Chinese Chemical Society. 2025. 2025 39th china chemistry olympiad (final): Theory exam score conversion weights. Online (PDF). Accessed: 2026-01-05.

Chinese Physics Olympiad (CPhO). 2025a. 42nd national high school physics competition (final) 2025: Theory exam paper (public pdf). Online (PDF). Accessed: 2026-01-05.

Chinese Physics Olympiad (CPhO). 2025b. 42nd national high school physics competition (final) 2025: Theory solutions and marking scheme (public pdf). Online (PDF). Accessed: 2026-01-05.

CPhO Committee. 2025a. 42nd national high school physics competition (final) 2025: Practical answer key / marking (public pdf). Online (PDF). Accessed: 2026-01-05.

CPhO Committee. 2025b. 42nd national high school physics competition (final) 2025: Practical exam paper (public pdf). Online (PDF). Accessed: 2026-01- 05.

CPhO Committee. 2025c. National high school physics competition charter (revised april 2025). Online (PDF). Accessed: 2026-01-05.

Song Dai, Yibo Yan, Jiamin Su, Dongfang Zihao, Yubo Gao, Yonghua Hei, Jungang Li, Junyan Zhang, Sicheng Tao, Zhuoran Gao, and Xuming Hu. 2025. Physicsarena: The first multimodal physics reasoning benchmark exploring variable, process, and solution dimensions. arXiv preprint arXiv:2505.15472.

Elliot Glazer, Ege Erdil, Tamay Besiroglu, Diego Chicharro, Evan Chen, Alex Gunning, Caroline Falkman Olsson, Jean-Stanislas Denain, Anson Ho, Emily de Oliveira Santos, Olli Järviniemi, Matthew Barnett, and 1 others. 2024. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai. arXiv preprint arXiv:2411.04872.

Google. 2025a. Gemini 3 flash: Frontier intelligence built for speed. https: //blog.google/products-and-platforms/ products/gemini/gemini-3-flash/. Accessed: 2026-05-26.

Google. 2025b. We’re expanding our gemini 2.5 family of models. https://blog.google/ products-and-platforms/products/gemini/ gemini-2-5-model-family-expands/. Published: 2025-06-17; accessed: 2026-08-30.

Google. 2026a. Gemini 3.1 pro: A smarter model for your most complex tasks. https://blog.google/ innovation-and-ai/models-and-research/ gemini-models/gemini-3-1-pro/. Accessed: 2026-05-26.

Google. 2026b. Gemini 3.5: Frontier intelligence with action. https://blog.google/ innovation-and-ai/models-and-research/ gemini-models/gemini-3-5/. Accessed: 2026- 05-26.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. Olympiadbench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Proceedings ofthe International Conference on Learning Representations (ICLR).

Zhen Huang, Zengzhi Wang, Shijie Xia, Xuefeng Li, Haoyang Zou, Ruijie Xu, Run-Ze Fan, Lyumanshan Ye, Ethan Chern, Yixin Ye, and 1 others. 2024. Olympicarena: Benchmarking multi-discipline cognitive reasoning for superintelligent ai. arXiv preprint arXiv:2406.12753.

IBO. 2023a. 34th international biology olympiad (ibo 2023): Final results. Online (PDF). Accessed: 2026- 01-05.

IBO. 2023b. Ibo examination papers archive (includes ibo 2023 uae papers). Online. Accessed: 2026-01- 05.

IChO 2025 Committee. 2025a. Icho 2025: Problems (official theory/practical exams and solutions). Online. Accessed: 2026-01-05.

IChO 2025 Committee. 2025b. Icho 2025 program. https://www.icho2025.ae/program/. Theory examination: 2025-07-10; accessed: 2026-08-30.

IPhO 2025 Committee. 2025. Ipho 2025 important dates. https://www.ipho2025.fr/ dates-importantes. Theory examination: 2025-07-21; accessed: 2026-08-30.

IPho 2025 Committee. 2025. Ipho 2025: Official questions and solutions (exam topics). Online. Accessed: 2026-01-05.

IPhO 2026 Organizing Committee. 2026a. Ipho 2026 official program. Official competition website. Accessed: 2026-08-30; event dates 4–12 July 2026; theoretical examination 8 July 2026.

IPhO 2026 Organizing Committee. 2026b. Ipho 2026 official questions. Official competition website. Accessed: 2026-08-30; official theoretical-exam asset reports July 2026 file metadata.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974.

Jon M. Laurent, Joseph D. Janizek, Michael Ruzo, Michaela M. Hinks, Michael J. Hammerling, Siddharth Narayanan, Manvitha Ponnapati, Andrew D. White, and Samuel G. Rodriques. 2024. LAB-Bench: Measuring capabilities of language models for biology research. arXiv preprint arXiv:2407.10362.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. 2024. From crowdsourced data to highquality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939.

Yucheng Li, Frank Guerin, and Chenghua Lin. 2023. An open source data contamination report for large language models. arXiv preprint arXiv:2310.17589.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. arXiv preprint arXiv:2209.09513.

A. Mirza and 1 others. 2025. A framework for evaluating the chemical knowledge and reasoning abilities of large language models. Nature Chemistry.

Nordic-Baltic Physics Olympiad. 2026a. Nbpho 2026. Official competition website. Accessed: 2026-08-29.

Nordic-Baltic Physics Olympiad. 2026b. Official nbpho 2026 media metadata. Official WordPress API. Accessed: 2026-08-29; problem media object dated 2026-04-25.

OpenAI. 2026a. Introducing gpt-5.4. https:// openai.com/index/introducing-gpt-5-4/. Accessed: 2026-05-26.

OpenAI. 2026b. Introducing gpt-5.5. https:// openai.com/index/introducing-gpt-5-5/. Accessed: 2026-05-26.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, Michael Choi, Anish Agrawal, and 1 others. 2025. Humanity’s last exam. arXiv preprint arXiv:2501.14249.

Qwen Team. 2026. Qwen3.7: The agent frontier. https://qwen.ai/research. Accessed: 2026-05- 26.

Delip Rao and Chris Callison-Burch. 2026. Autorubric: Unifying rubric-based llm evaluation. arXiv preprint arXiv:2603.00077.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2023. GPQA: A graduate-level google-proof Q&A benchmark. arXiv preprint arXiv:2311.12022.

Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. 2023. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. arXiv preprint arXiv:2307.10635.

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddhartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. 2024. Livebench: A challenging, contamination-free llm benchmark. arXiv preprint arXiv:2406.19314.

Fangchen Yu, Haiyuan Wan, Qianjia Cheng, Yuchen Zhang, Jiacheng Chen, Fujun Han, Yulun Wu, Junchi Yao, Ruilizhen Hu, Ning Ding, Yu Cheng, Tao Chen, Lei Bai, Dongzhan Zhou, Yun Luo, Ganqu Cui, and Peng Ye. 2025. HiPhO: How far are (M)LLMs from humans in the latest high school physics olympiad benchmark? arXiv preprint arXiv:2509.07894.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2023. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. arXiv preprint arXiv:2311.16502.

Yilun Zhao, Kaiyan Zhang, Tiansheng Hu, Sihong Wu, Ronan Le Bras, Charles McGrady, Taira Anderson, Jonathan Bragg, Joseph Chee Chang, Jesse Dodge, Matt Latzke, Yixin Liu, Xiangru Tang, Zihang Wang, Chen Zhao, Hannaneh Hajishirzi, Doug Downey, and Arman Cohan. 2025a. Sciarena: An open evaluation platform for non-verifiable scientific literaturegrounded tasks. arXiv preprint arXiv:2507.01001.

Zehua Zhao, Zhixian Huang, Junren Li, Siyu Lin, Junting Zhou, Fengqi Cao, Kun Zhou, Rui Ge, Tingting Long, Yuexiang Zhu, Yan Liu, and 1 others. 2025b. SUPERChem: A multimodal reasoning benchmark in chemistry. arXiv preprint arXiv:2512.01274.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yanping Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

Junting Zhou, Jin Chen, Linfeng Hao, Denghui Cao, Zheyu Wang, Qiguang Chen, Chaoyou Fu, Jiaze Chen, Yuchen Wu, Ge Zhang, Mingxuan Wang, Wenhao Huang, and Tong Yang. 2026. BABE: Biology arena benchmark. arXiv preprint arXiv:2602.05857.

## A Additional Failure Cases

Figure 4 shows one representative failure from each domain. Here we provide one additional case per domain to make the qualitative diagnosis more complete.

## Physics: correct theory, wrong graph binding (IPhO 2025 S1 B2)

The human expert noted that GPT-5’s theoretical analysis was correct, but the model used the wrong point on the correct curve. This earned 0.3/0.5 rather than full credit. The case illustrates a recurring physics pattern: symbolic derivation can be near-perfect while the final numerical answer is damaged by one visual-grounding decision.

## Chemistry: plausible mechanism, wrong intermediates (CChO 2025 Q9.7)

The task asks for key intermediates in a Trigonoliimine C rearrangement. Representative Gemini runs described a plausible spirocyclopropane/spiroindolenium-style pathway, but did not provide the official intermediate structures and mismatched the rearrangement sequence, receiving 0/6. The failure is retrieval-adjacent: the prose resembles a real mechanism, but the scored object is an exact set of structures.

## Biology: experimental-design matching error (IBO 2023 Item 20)

The item asks models to match four guppy mate-choice hypotheses to four experimental designs. Several strong models returned 2 4 3 1 instead of the official 2 4 1 3, confusing the control for male activity/preference with the control for general side or group preference. The answer is not factually obscure; the difficulty is maintaining the causal role of each experimental manipulation across four similar alternatives.

## B Full Subfield Diagnostics

Table 10 reports the numeric values underlying Figure 5 in Section 3. The main-text radar makes the within-domain comparison visually explicit; its values and coverage counts are tabulated below.

The available diagnostic table records model– turn counts but lacks a question-level classification map; unique-question counts are therefore unavailable and are not inferred from N.

## C Human Expert Notes and Ability Boundary

This appendix expands the shortened discussion in Section 3.6. The notes come from archived expert grading of the original five model answers on IPhO 2025 and IChO 2025. They are useful because the graders comment on why a response deserves or loses partial credit, not merely on whether the final answer is correct. We therefore use the same notes to define the ability-boundary radar plots in this appendix: symbolic derivation, graph/visual grounding, global coherence, structure/stereochemistry, analytical reasoning, and data interpretation are not abstract categories, but recurring distinctions made by the human graders.

<table><tr><td>Domain</td><td>Subfield</td><td>Score</td><td>N</td></tr><tr><td>Phys.</td><td>thermodynamics</td><td>74.1%</td><td>805</td></tr><tr><td>Phys.</td><td>astrophysics</td><td>76.8%</td><td>470</td></tr><tr><td>Phys.</td><td>electromagnetism</td><td>78.2%</td><td>258</td></tr><tr><td>Phys.</td><td>mechanics</td><td>81.3%</td><td>354</td></tr><tr><td>Phys.</td><td>modern quantum atomic</td><td>83.7%</td><td>332</td></tr><tr><td>Chem.</td><td>stereochemistry</td><td>34.7%</td><td>187</td></tr><tr><td>Chem.</td><td>inorganic coordination</td><td>56.8%</td><td>383</td></tr><tr><td>Chem.</td><td>organic structure</td><td>71.8%</td><td>419</td></tr><tr><td>Chem.</td><td>physical chemistry</td><td>78.4%</td><td>719</td></tr><tr><td>Chem.</td><td>analytical spectroscopy</td><td>86.6%</td><td>60</td></tr><tr><td>Bio.</td><td>experimental data</td><td>64.3%</td><td>14</td></tr><tr><td>Bio.</td><td>bioinformatics statistics</td><td>76.6%</td><td>42</td></tr><tr><td>Bio.</td><td>animal behavior neurobiology</td><td>78.6%</td><td>56</td></tr><tr><td>Bio.</td><td>molecular genetics</td><td>85.8%</td><td>266</td></tr><tr><td>Bio.</td><td>biochemistry</td><td>91.4%</td><td>70</td></tr></table>

Table 10: Numeric subfield diagnostics corresponding to Figure 5. Score is total awarded points divided by total available points; N counts model–turn grading instances, not independent questions. Small strata, especially biology experimental data (N = 14), are exploratory.

Fluent derivation is not the same as physical control. In IPhO 2025, GPT-5 often received full credit for local symbolic derivations, and experts explicitly described several early subparts as essentially perfect. The same solution then lost credit on graph- or regime-dependent items. For example, in S1 B2 the expert noted that the theoretical analysis was correct, but the model used the wrong point on the correct curve, leading to only 0.3/0.5. In S1 C4, the formula work was again strong, but the peakfrequency reading and statistical inference were off, giving 0.3/0.6. The deeper lesson is that physics olympiad grading is not just equation checking: the model must bind a symbol to the correct plotted quantity, choose the right curve, and propagate measured values with the intended uncertainty.

Long reasoning can hide a wrong global model. The clearest IPhO failure appears in S2 C1–C2. The human grader marked GPT-5 at 0/1 and 0/1 because the answer failed to realize a conserved total-force constraint and did not provide the necessary case analysis. This is a different failure from arithmetic error. The model can write a long solution, but if the first physical invariant is wrong, the response becomes a plausible derivation of the wrong problem. This observation motivates our preference for interleaved evaluation: it gives each subpart focused context, but it also shows that interleaving alone cannot fix a mistaken regime choice once the conversation has anchored on it.

Chemistry experts grade exact structures, not chemical vibes. The IChO comments are even sharper. On IChO 2025 Q1.3, Gemini 2.5 Pro suggested that the molecular formula might contain a typo and then used the wrong carbon skeleton; the expert marked the subpart 0 and identified this as a structure-comprehension failure rather than a minor wording issue. On Q1.6, the same model described reaction logic that was partly reasonable, but repeatedly used a decalin-like skeleton instead of the required tricyclic skeleton; without a valid error-forward path, I/J/K all received 0. For GPT-5, the expert comments similarly flagged failures to recognize a pictured compound, wrong methyl positions in a carbon skeleton, and incorrect nucleophile approach geometry. These cases explain why chemistry remains harder than physics in Figure 5: a response can name the right reaction family and still receive no credit if the final drawn structure is not uniquely correct.

Partial credit depends on explicit, checkable commitments. The expert notes often reward model behavior that ordinary automatic metrics would miss. For IPhO drawing tasks, GPT-5 was sometimes credited when it could not literally draw a figure but gave an unambiguous verbal description of the correct picture. Conversely, chemistry answers with vague structural prose were often marked zero, even when the prose contained relevant terms, because the grader could not reconstruct a unique molecule. The IChO notes also distinguish valid error-forward reasoning from invalid carryover: if a later step is consistent with an earlier explicitly drawn wrong structure, partial credit may be possible; if the earlier structure is ambiguous or chemically impossible, later prose is not enough. This distinction became a strict rule in our judge prompt: score only what is explicitly present, do not infer missing structures, and apply error-forward only when the candidate’s intermediate is concrete and chemically coherent.

Human comments sharpen judge calibration. The archived IChO grading files include cases where experts commented on both the model answer and its automatic score. They identify demanding distinctions such as an incorrect carbon skeleton, a wrong nucleophile approach, or a correct net reaction expressed in the wrong requested form. Accordingly, the judge pipeline uses rubric gates for answer format, structure specificity, and subpart-by-subpart score allocation. These gates turn expert observations into an explicit, auditable checklist for structure-heavy chemistry.

## C.1 Ability Axis Glossary

The expert comments were then abstracted into reusable ability axes. The axes are intentionally concrete: each one corresponds to a type of credit or deduction that appeared repeatedly in medalist grading.

• Phys: Symbolic Deriv. — multi-step algebraic manipulation and equation-based reasoning.

• Phys: Numeric+Units — numerical accuracy with unit/scale discipline (dimensional checks, sanity bounds).

• Phys: Approx. & Limits — correct approximations and limiting-case validation.

• Phys: Global Coherence — maintain consistent assumptions/variables across coupled subparts (regime tracking).

• Phys: Graph Reading — extract trends/values from curves and use them downstream.

• Shared: Visual Grounding — bind non-text evidence to symbols/claims (diagrams/structures/- figures; “read-the-figure” reliability).

• Chem: Quant Modeling — quantitative physical chemistry (equilibria/thermo/electro) under algebraic constraints.

• Chem: Structure+Stereo — structure fidelity and stereochemical correctness (connectivity, configuration).

• Chem: Analytical — interpret analytical evidence (e.g., spectroscopy/assays) into chemically valid conclusions.

• Bio: Knowledge — biology knowledge and reading comprehension in constrained formats (mechanisms/terminology).

• Bio: Data/Exp — reasoning from experimental data (figures/tables, controls, statistical cues, causality).

• Shared: Low Halluc. — calibration/faithfulness proxy: lower rate of confidently-wrong outputs when confidence is available.

![](images/b543101ddae6ee53a7fcf91ec627e87fd17cd2453404c6b522d38cb1cadbcf8b.jpg)  
Figure 6: IPhO 2025. Physics abilities follow our rubric: symbolic derivation, numerical computation with units, approximation/limiting cases, graph reading, plotting/representation, global coherence across regimes, data inference from empirical relations, and visual grounding.

## C.2 Domain-specific ability profiles

Figure 6, Figure 7, and Figure 8 summarize withindomain ability profiles induced from the expertnote taxonomy. Each radar overlays afrontier envelope (per-axis best among the compared frontier models), GPT-5, and Gemini 2.5 Pro. Because different axes may be supplied by different systems, the frontier envelope is a diagnostic upper bound rather than the profile of a single model. Axes are domain-native (Physics/Chemistry/Biology) and are not directly comparable across figures.

IPhO 25 notes. Both models are strongest on local symbolic manipulation and routine numerics; remaining errors concentrate in global coherence (propagating assumptions consistently across long solutions) and visual grounding (extracting nontextual cues), with noticeable variance across tasks requiring regime switching.

IChO 25 notes. Gemini 2.5 Pro tends to be more uniformly strong on quantitative and analytical axes (physical chemistry and spectroscopy), while GPT-5 exhibits higher dispersion: it can solve isolated substeps but is less consistent when multiple constraints (structure + stereochemistry + quantitative checks) must be satisfied simultaneously.

IBO 23 notes. Frontier models reach near-ceiling performance on knowledge-centric multiple-choice biology (molecular biology and genetics), but are noticeably weaker on evidence-centric reasoning (statistics/design and data/figure interpretation), consistent with broader multimodal grounding limitations. Together, the three profiles show that remaining weaknesses are concentrated in evidence binding, representational precision, and longhorizon consistency rather than simple recall.

IChO 2025: Ability Profile  
![](images/1ac638572601e2921844997727499706ec022d76fd01710cfaece0e1c09dcc01.jpg)  
Figure 7: IChO 2025. Chemistry abilities use the human-coded decomposition: quantitative physical chemistry; spectroscopy/analytical reasoning; organic structure depiction; stereochemistry; inorganic/organometallic reasoning; coordination/combinatorics; biochemistry/enzyme reasoning; and biosynthesis/isotopic tracing.

## D Prompt Templates

The benchmark uses generated prompts at the turn level; below are the reusable templates and the sealed 2026 routing invariant. Problem-specific fields are filled from the released JSON rows. Textonly routes contain no image blocks or remote image URLs, whereas vision routes in the sealed 2026 runs receive the exact original question PNG bytes as API image blocks.

## Benchmark digitization prompt

Convert the provided competition problem, rendered pages, figures, official answer, detailed solution, and rubric into canonical text. Preserve all numerical values, labels, tables, figure relationships, chemical structures, diagrams, and point allocations. If a figure is needed for solving, describe it explicitly enough that a text-only solver can use it. Do not introduce outside facts, do not omit subparts, and do not include remote URLs or imagedata strings in solver-visible fields. Return structured JSON with question text, figure descriptions, official answer, detailed solution, rubric, source caveats, and completeness notes.

IBO 2023: Ability Profile  
![](images/9ee179bebbbf2bdacd8097d23246c06bdc22b99369fc2466310b565a7d5907d5.jpg)  
Figure 8: IBO 2023. Biology abilities are summarized into observed question categories derived from model outputs: molecular biology/biochemistry, genetics/probability, physiology/systems, experimental design/statistics, data/figure interpretation, and a “other” bucket.

## Sealed 2026 question-only staticization prompt for text routes

Convert only the supplied official question text and question-page images into answer-blind canonical problem text for a text-only solver. Preserve every visible number, label, table, diagram relation, chemical structure, and requested answer format needed to state the question. Do not use or infer any official answer, solution, marking rubric, or solution-page content. Return only the frozen question-side representation and provenance/completeness fields; include no URLs or image-data strings.

## Interleaved solver prompt for open-ended items

You are an expert olympiad contestant in the relevant scientific domain. Solve the current subpart using only the provided staticized problem text, prior subpart context from the same sequence, and the official answer-format instruction. Show enough reasoning for the grader to identify partial-credit steps, keep variables and assumptions consistent with earlier turns, and make the final answer explicit. When a structure, diagram, or table entry is required, describe the complete checkable object rather than only naming a reaction, concept, or qualitative trend.

## Sealed 2026 vision-route payload invariant

Send the native question text and every frozen questionpage PNG needed by the current linked sequence directly as image blocks in the solver API request. The image bytes must match the official question-page render manifest; do not substitute a caption or description generated by another multimodal model. Never attach official answers, solutions, rubrics, or solution-page images to the solver request.

## IBO objective-key prompt

Answer the biology competition question based solely on the provided information. For true/false questions output T or F; for choice questions output the requested letter or number; for numerical blanks output the requested rounded value. Place the final ordered answer sequence between <boa> and <eoa> exactly, separated by single spaces. Explanations may appear outside the tags, but the text inside the tags must contain only the final answer tokens.

## Rubric-faithful judge prompt

You are a strict olympiad grader. Given the problem, official answer, detailed solution, point rubric, and candidate response, assign partial credit exactly according to the rubric. Score only explicit, checkable claims in the candidate response. Do not infer missing structures, diagrams, numerical values, or reaction intermediates. Apply errorforward only when the candidate has made a concrete intermediate commitment that is internally coherent and the later reasoning follows from it. When original question or solution page images are attached to the judge packet, treat those images and the provenance-labeled official rubric as authoritative evidence. Return JSON with score, max\_score, subcriterion scores, concise evidence, and any uncertainty or escalation flag.

## Subfield and modality classification prompt

Classify the benchmark turn using only the staticized question, official answer, and rubric. Return JSON fields for scientific domain, subfield, skill type, whether visual evidence is required, whether chemical/biological structures are required, whether quantitative calculation is required, and a short rationale. Do not look at model answers and do not change any benchmark content.

## E 2026 Evaluation and Calibration Audit

Evaluation completeness. The four 2026 runs contain 23 IPhO, 68 IChO, 28 NBPhO, and 34 US-APhO linked turns, respectively. All 2,142 solver terminals and all 56 model–contest aggregates are present. For every vision-route attempt, the audit reconstructs the API payload and verifies each direct PNG block against the frozen official-page SHA-256; every text-route attempt has zero image blocks and a recomputed question-only prompt hash. Every non-deterministic judge request binds the fixed Gemini 3.5 Flash route, exact item rubric, criterion identifiers, and maxima.

Route-level macro scores and input assignments. The four-contest evaluation contains 153 linked turns grouped into 28 whole problem sequences, yielding 2,142 sealed model–turn terminals and complete aggregates for all 14 routes. Payload audits verify that every stored V request includes direct PNG data-image blocks whose bytes match the frozen official question pages, while every T request contains zero image blocks and exactly the frozen question-only staticization. All four contest evaluations use the same route roster, modality assignments, score maxima, fixed Gemini 3.5 Flash judge route, and frozen rubric criteria.

<table><tr><td>Model</td><td>Input</td><td>Macro</td><td>95% CI</td></tr><tr><td>Gemini 3 Flash</td><td>V</td><td>85.56</td><td>[79.24, 91.66]</td></tr><tr><td>Seed 2.0 Pro</td><td>V</td><td>71.03</td><td>[63.67, 78.85]</td></tr><tr><td>Gemini 3.1 Pro</td><td>V</td><td>88.47</td><td>[83.80, 93.18]</td></tr><tr><td>GPT-5.4</td><td>V</td><td>68.79</td><td>[61.86, 75.78]</td></tr><tr><td>GLM-5.1</td><td>T</td><td>86.89</td><td>[81.13, 92.05]</td></tr><tr><td>Qwen3.6-Plus</td><td>V</td><td>82.63</td><td>[75.55, 89.74]</td></tr><tr><td>MiniMax M2.7</td><td>T</td><td>69.61</td><td>[62.01, 77.48]</td></tr><tr><td>Claude Opus 4.7</td><td>V</td><td>79.86</td><td>[73.19, 86.37]</td></tr><tr><td>MiMo V2.5 Pro</td><td>T</td><td>63.81</td><td>[54.46, 72.66]</td></tr><tr><td>GPT-5.5</td><td>V</td><td>93.31</td><td>[90.95, 95.88]</td></tr><tr><td>DeepSeekV4-Pro</td><td>T</td><td>87.76</td><td>[82.86, 92.43]</td></tr><tr><td>DeepSeekV4-Flash</td><td>T</td><td></td><td></td></tr><tr><td>Gemini3.5-Flash</td><td>V</td><td>86.62 89.54</td><td>[80.97, 91.99]</td></tr><tr><td></td><td>T</td><td></td><td>[86.12, 93.19]</td></tr><tr><td>Qwen3.7-Max</td><td></td><td>85.16</td><td>[79.82, 90.54]</td></tr></table>

Table 7: Four-contest normalized macro scores and sequence-cluster bootstrap intervals. V routes receive original question PNGs directly; T routes receive answer-blind question-only staticizations and no images. The interval resamples whole problem sequences within each contest and captures item/sequence sampling uncertainty only, not generation or judge variance.
<table><tr><td>Domain</td><td>Native</td><td>Description</td><td>Blind</td><td>∆N-D 95% CI</td></tr><tr><td>Physics</td><td>78.82</td><td>81.57</td><td>55.47</td><td>-2.75 [-6.62, 1.15]</td></tr><tr><td>Chemistry</td><td>53.72</td><td>48.70</td><td>17.73</td><td>+5.01 [0.57, 9.56]</td></tr><tr><td>Biology</td><td>85.79</td><td>84.99</td><td>76.50</td><td>+0.80 [-2.13,3.74]</td></tr></table>

Table 11: Domain split of the controlled modality ablation. Arm entries are mean normalized scores in percent; ∆ N–D is native image minus faithful description in percentage points, with paired-bootstrap intervals.

The 10,000 paired resamples use seed 20260829 and sample whole problem sequences: 3 IPhO, 9 IChO, 10 NBPhO, and 6 USAPhO problems, with all linked subparts moving together. Each contest score is normalized by its explicit maximum before the four percentages are equally averaged, and no missing aggregate is imputed. Intervals overlap substantially for several neighboring systems, so small point-ranking differences should not be read as statistically resolved. Exact first-availability days and training cutoffs are unavailable for the evaluated routes; same-month status is therefore unknown. Official day-level problem-release evidence is available for NBPhO (25 April 2026) and USAPhO (20 May 2026), whereas the official IPhO and IChO pages display no publication date.

The transport audit confirms delivery of every scheduled native image block across all retained API requests, passes visible-image-use checks for every retained native response, and finds no image blocks or hidden URLs in either text arm. These checks confirm that the three conditions differ in visual evidence as designed rather than through unintended route leakage.

<table><tr><td>Judge</td><td>Input</td><td>IPhO (/30)</td><td>IChO (/60) MAE (pp)</td><td></td></tr><tr><td>Gemini 3.5 Flash</td><td>V</td><td> $2 9 . 7 5 0 \pm 0 . 0 0 0$ </td><td> $5 6 . 0 1 1 \pm 0 . 5 1 6$ </td><td>1.582</td></tr><tr><td>Gemini 3.1 Pro</td><td>V</td><td> $2 8 . 7 5 0 \pm 1 . 0 0 0$ </td><td> $5 6 . 4 5 6 \pm 0 . 0 9 7$ </td><td>0.866</td></tr><tr><td>Claude Opus 4.7</td><td>V</td><td> $2 9 . 0 8 3 \pm 1 . 1 5 5$ </td><td> $5 6 . 7 3 8 \pm 0 . 5 0 8$ </td><td>2.323</td></tr><tr><td>GPT-5.5</td><td>V</td><td> $2 8 . 2 5 0 \pm 0 . 8 6 6$ </td><td> $5 7 . 0 9 9 \pm 0 . 1 9 2$ </td><td>0.733</td></tr><tr><td>Qwen3.7-Max</td><td>T</td><td> $2 7 . 7 5 0 \pm 1 . 0 0 0$ </td><td> $5 4 . 4 4 2 \pm 0 . 8 1 5$ </td><td>1.691</td></tr><tr><td>DeepSeek V4 Pro</td><td>T</td><td> $2 8 . 0 8 3 \pm 0 . 5 7 7$ </td><td> $5 4 . 7 6 1 \pm 1 . 5 6 8$ </td><td>1.672</td></tr></table>

Table 12: Three-repeat judging of the same frozen Gemini 3.5 Flash answers from IPhO and IChO 2026. Contest entries are mean ± sample SD; MAE is the overall within-judge normalized repeat MAE in percentage points. V and T denote the problem representation: V uses original official question and available solution/marking PNGs, whereas T replaces problem images with the frozen question-side staticization and sends zero image blocks; both modes retain the same candidate-answer, official-solution, and rubric packet.

Frozen-answer cross-family judge study. Strict replay covers all 1,638 effective final judge terminals. Gemini 3.1 Pro uses one same-configuration recovery cell, and the DeepSeek row uses one unified thinking-enabled 65k-token configuration over all 273 cells; only these matched-configuration results enter the table. The four V rows define the primary cross-family comparison, while the two T rows are modality-sensitivity conditions and are not pooled into V–V agreement.

Archived calibration replay. The structured replay finds 1,050 stored judgment rows over 525 distinct expert rows, with no missing or errored judgment. Expert-agreement metrics use 1,048 scope-aligned pairs. One IChO expert label for each judge scores the full multipart Q2.3 group, whereas the corresponding judge row scores only Q2.3c on its six-point rubric; these two records address different units and therefore do not enter item-level agreement metrics.

Structured expert feedback. The deterministic export contains 525 provenance-addressed expertscore rows: all 325 IChO rows and 85 of 200 IPhO rows retain nonempty comments, while the remaining 115 IPhO rows explicitly record a null comment; each row also preserves the two archived calibrated-judge records when available. The archived sources contain no human-authored error tags, evidence spans, uncertainty labels, or conflict labels; the corresponding fields remain null and are not synthesized automatically. The resulting schema supports item-level analysis of judge– expert discrepancies while retaining exact provenance to the original grading records.

<table><tr><td>Judge</td><td>Exam</td><td>n</td><td>Raw MAE</td><td>Norm. MAE</td><td>r</td></tr><tr><td>Gemini 3.1 Pro</td><td>IPhO 25</td><td>200</td><td>0.066</td><td>0.092</td><td>0.833</td></tr><tr><td>Gemini 3.1 Pro</td><td>IChO 25</td><td>324</td><td>0.303</td><td>0.075</td><td>0.909</td></tr><tr><td>Gemini 3.5 Flash</td><td>IPhO 25</td><td>200</td><td>0.069</td><td>0.097</td><td>0.829</td></tr><tr><td>Gemini 3.5 Flash</td><td>IChO 25</td><td>324</td><td>0.537</td><td>0.138</td><td>0.868</td></tr></table>

Table 13: Item-level replay of the two calibrated judges against expert rows under the frozen scoring protocol. Raw and normalized metrics use the same 324 scopealigned IChO pairs; the whole-group Q2.3 expert label is outside this item-level estimand. Cross-family robustness is evaluated separately in Table 12.

## F Additional Temporal and Judge Sensitivity Checks

The following analyses test two complementary sensitivities: the temporal probe tests direct answer memorization, and the IChO subset stresstests structure-aware grading. They supplement the repeated-run and cross-family experiments by targeting distinct aspects of evaluation reliability.

Exploratory temporal probe. Across three models and four seeds, 69/72 isomorphic numeric perturbations elicited the perturbed target, with 0/72 stale original-answer substitutions; a strict noquestion control produced 0/72 recalls. This small behavioral probe finds no direct memorization signal, but it cannot establish absence of training exposure and does not change our public-benchmark interpretation of Table 6.

Structure-heavy judge stress test. On a 50-row IChO structure/stereochemistry subset scored by each of four image-aware judges, MAE against expert scores ranges from 0.66 to 1.24 points; the numbers of rows over-credited by at least 0.5 point are 5, 8, 9, and 21 out of 50 across the four judges. Across this demanding subset, all four judges remain close to expert scores; multi-judge agreement or expert adjudication provides the strongest safeguard for the lowest-confidence structural cases.

## G Competition Sources

SCIENCEARENA is built from thirteen olympiadstyle scientific competitions spanning physics, chemistry, and biology, with explicit source provenance. We record the public release time of official problem materials separately from the contest year for descriptive temporal screening. When an exact official publication date or evaluated- route availability date is unavailable, it is marked unknown rather than treated as a strict time-release holdout; no release timestamp by itself proves absence from training data. Table 2 summarizes the benchmark sources and rubric status.

International Physics Olympiad (IPhO 2025). We use the official IPhO 2025 examination page and its released questions and answer sheets (IPho 2025 Committee, 2025). IPhO exams consist of theory and experimental components; our benchmark focuses on the theory problems, since experimental tasks require physical apparatus and proctoring conditions that are not directly reproducible for LLM evaluation.

International Chemistry Olympiad (IChO 2025). We source IChO 2025 materials from the official English examination-paper release (IChO 2025 Committee, 2025a). Similar to IPhO, IChO includes theory and laboratory components; we evaluate models on the theory portion to ensure a welldefined, reproducible setting with rubric-faithful scoring across all models.

International Physics and Chemistry Olympiads (IPhO/IChO 2026). The fresh international tracks use the official IPhO questions page and the official IChO problems-and-solutions page (IPhO 2026 Organizing Committee, 2026b; 58th International Chemistry Olympiad Organizing Committee, 2026b). The 56th IPhO ran from 4–12 July 2026 and held its theoretical examination on 8 July; the 58th IChO ran from 10–19 July 2026 and held its theoretical examination on 15 July (IPhO 2026 Organizing Committee, 2026a; 58th International Chemistry Olympiad Organizing Committee, 2026a). The official hosts expose the evaluated examination files with July 2026 file metadata, so the tables record the month-level release as 26-07; this does not assert an exact first-publication day.

International Biology Olympiad (IBO 2023). IBO provides an official archive of examination papers for past competitions (IBO, 2023b); the archived 2023 materials used here were released in August 2025. In Science Arena, IBO 2023 serves as an objectively scorable biology track (e.g., multiple-choice or otherwise unambiguously keyed questions), enabling automatic evaluation without judge-model mediation. When reporting medalcalibrated comparisons, we reference the official IBO 2023 results report (IBO, 2023a).

China Chemistry Olympiad (CChO 2025). For CChO 2025, we use publicly released final-round materials (problem statements and marking guidance) distributed as a consolidated PDF (China Chemistry Olympiad (CChO), 2025). When normalizing scores and aligning evaluation to official scoring practice, we additionally cite the Chinese Chemical Society document describing the theory score conversion weights (Chinese Chemical Society, 2025). As with other laboratory-inclusive competitions, our primary benchmark setting targets the theory component for reproducibility.

China Physics Olympiad (CPhO 2025). For CPhO 2025, we use publicly released finalround theory materials and the associated marking scheme (Chinese Physics Olympiad (CPhO), 2025a,b). Where relevant to describing the competition formally (e.g., structure, governance, or official rules), we also cite the published competition charter (CPhO Committee, 2025c). We evaluate the theory component as the reproducible portion of the exam; experimental components and their marking documents can be referenced separately when needed (CPhO Committee, 2025b,a).

Additional regional and national sources. The expanded suite also includes APhO 2025, EuPhO 2025, NBPhO 2026, USAPhO 2026, INChO 2026, and USNCO 2025. NBPhO problem media are dated 25 April 2026 on the official host, while the official USAPhO news page links the exam and solution under 20 May 2026 (Nordic-Baltic Physics Olympiad, 2026a,b; American Association of Physics Teachers, 2026). These sources are normalized through the same digitization and expert-audit pipeline described in Section 2; source caveats and rubric are kept in the released metadata.

Notes on scope and reproducibility. All openended competitions are high-stakes, rubric-driven assessments with well-defined problem sets and partial-credit grading, while IBO is objectively keyed. SCIENCEARENA prioritizes publicly released materials for reproducibility and transparent sourcing, and it uses theory-only evaluation where experimental replication would otherwise confound model comparisons. Each digitized item preserves its source examination, official answer, and grading rules, making the evaluation packet auditable. Frozen item definitions and score maxima are shared across model routes, ensuring every system is evaluated against the same scientific target.