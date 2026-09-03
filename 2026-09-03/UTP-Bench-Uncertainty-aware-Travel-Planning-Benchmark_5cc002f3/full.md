# UTP-Bench: Uncertainty-aware Travel Planning Benchmark

Etcharla Revanth Rao<sup>1†</sup>, Priyanshu Karmakar<sup>1†</sup>, Shubhojit Mallick<sup>2</sup>,

Manish Gupta<sup>2</sup>, Shreya Ghosh<sup>1</sup>, Abhik Jana<sup>1</sup>

<sup>1</sup>IIT Bhubaneswar, India <sup>2</sup>Microsoft, India

a24cs08008,24CS06010,abhikjana,shreya@iitbbs.ac.in shubhojit.mallick,gmanish@microsoft.com

## Abstract

Large Language Models (LLMs) have recently demonstrated strong capabilities in automated travel itinerary generation. However, realworld travel planning is inherently uncertain: transportation delays, crowd fluctuations, and unexpected stochastic delays frequently invalidate otherwise feasible schedules. Existing benchmarks like TravelPlanner and TripCraft assume deterministic environments, evaluating only static constraint satisfaction and ignoring whether generated plans remain robust when such uncertainties arise.

To address this limitation, we introduce UTP-Bench<sup>1</sup>, a large-scale benchmark for uncertainty-aware travel planning. The dataset integrates real-world travel data spanning 504 cities of India, including attractions, restaurants, accommodations, and multi-modal transportation networks. To model realistic disruptions, UTP-Bench incorporates empirical delay distributions and crowd-density patterns collected from major cities, enabling evaluation of travel plans under stochastic conditions.

We further propose three evaluation metrics, namely Buffer Adequacy Score (BAS), Crowd-Aware Timing Score (CATS), and Transport Delay Absorption Score (TDAS), which quantify the ability of generated itineraries to maintain robustness against transit delays and crowd variability. Experiments with state-of-the-art LLMs like GPT-5, Qwen3, Mistral and Phi-4 reveal substantial gaps between model-generated and human-authored plans, particularly in temporal buffering, delay-aware transportation scheduling, and crowd-sensitive planning.

## 1 Introduction

Large Language Models (LLMs) have recently shown strong potential for automated travel itinerary generation, enabling personalized travel plans that satisfy user preferences, budget constraints, and temporal requirements. Recent works, including TravelPlanner (Xie et al., 2024), TripCraft (Chaudhuri et al., 2025), and Retail (Deng et al., 2025), demonstrate that LLMs can generate structured multi-day itineraries while reasoning over diverse travel constraints such as attraction preferences, transportation connectivity, and scheduling requirements.

Despite recent advances, existing benchmarks remain limited in modeling real-world travel planning. First, they largely assume deterministic environments with fixed transportation schedules, attraction durations, and transit conditions. In reality, travel planning is inherently uncertain due to delays, congestion, and unexpected disruptions, making otherwise feasible itineraries impractical. Consequently, evaluating plans solely through static constraint satisfaction fails to capture robustness under realistic conditions. Second, current benchmarks fail to represent the variability of real-world travel. Datasets such as TravelPlanner, TripCraft, and ChinaTravel (Shao et al., 2026) rely on simplified assumptions regarding transit reliability and scheduling, while overlooking factors such as multimodal dependencies and dynamic crowd conditions that critically influence itinerary feasibility.

To address these limitations, we introduce UTP-Bench, a benchmark for uncertainty-aware travel planning. UTP-Bench comprises 1000 real-world travel queries spanning 3-, 5-, and 7-day spatiotemporal fine-grained itineraries which is paired with a human-annotated gold-standard itinerary. Unlike prior datasets, UTP-Bench incorporates empirical delay statistics and crowd-density patterns enabling evaluation of travel itineraries under realistic uncertainty.

Evaluating travel itineraries under uncertainty requires moving beyond binary feasibility checks. Existing approaches determine whether constraints are satisfied, but fail to assess how robust an itinerary remains under uncertainties. But what truly makes a travel plan reliable? Does the itinerary allocate sufficient transition time to absorb unexpected delays and variability in attraction visits? Can transportation schedules tolerate realistic disruptions without causing cascading failures across subsequent activities? Does the plan avoid highly congested periods, or does it schedule visits during peak crowd hours that may significantly affect the overall experience? Most importantly, does the itinerary adapt its pacing according to different traveller risk preferences, balancing exploration with reliability? To systematically assess these aspects, we introduce three uncertaintyaware evaluation metrics: Buffer Adequacy Score (BAS), which measures whether activity transitions include sufficient temporal buffers; Transport Delay Absorption Score (TDAS), which evaluates the resilience of transportation schedules under empirical delay distributions; and Crowd-Aware Timing Score (CATS), which quantifies the extent to which attraction visits avoid peak crowd periods. Furthermore, UTP-Bench models diverse traveler behaviors through risk-aware travel profiles, including Risk-Tolerant, Risk-Optimized, and Risk-Averse travelers. These profiles capture different levels of tolerance to scheduling uncertainty and influence the amount of buffer time and itinerary density that are considered acceptable. Our contributions are three-fold:

![](images/29deca91f9987b0def815525947a7c40f7c9afbe48e0c14fc304795d532892f4.jpg)  
Figure 1: UTP-Bench: Bridging the gap between rigid travel plans and real-world uncertainty

1. Uncertainty-aware travel planning benchmark. We introduce UTP-Bench, a large-scale travel planning dataset integrating real-world attractions, restaurants, accommodations, and multimodal transportation data.

2. Modeling real-world travel uncertainty and traveler profiles. UTP-Bench incorporates empirical transit delay statistics and crowd-density patterns to enable systematic evaluation of travel plans with uncertainties. The benchmark also introduces risk-aware traveler profiles that model varying tolerance to scheduling uncertainty, supporting more realistic itinerary generation and evaluation.

3. Novel evaluation metrics for robust travel planning. We propose BAS, TDAS, and CATS to quantify the robustness of generated itineraries with respect to temporal buffers, crowd dynamics, and transportation delays.

## 2 Related Work

Planning with Large Language Models. Large language models (LLMs) have shown strong capabilities in planning tasks, including scheduling, commonsense reasoning, and heuristic-guided decision-making (Valmeekam et al., 2023b; Pallagani et al., 2023; Prasad et al., 2024; Lee et al., 2025). Prior work has improved planning performance through grounded few-shot planning (Song et al., 2023) and integration with classical search methods such as Monte Carlo Tree Search (Coulom, 2006; Swiechowski et al.<sup>´</sup> , 2023; Zhao et al., 2023). However, LLMs still struggle to generate reliable plans in open-ended domains due to challenges in modeling subgoal dependencies, handling cascading constraint violations, and reasoning under uncertainty (Valmeekam et al., 2023a; Kambhampati et al., 2023).

LLMs in Travel Planning. Travel planning requires satisfying spatial, temporal, budgetary, and preference-driven constraints over extended horizons (Xi et al., 2025; Jonnala et al., 2025). Prior benchmarks, including TravelPlanner (Xie et al., 2024), TravelPlanner+ (Singh et al., 2024), and TripCraft (Chaudhuri et al., 2025), advanced constraint-aware and persona-driven itinerary generation using real-world data. However, these approaches assume deterministic conditions, overlooking uncertainties such as transit delays, variable visit durations, and crowd dynamics. While TripTide (Karmakar et al., 2026) introduces disruptions, its scenarios remain synthetic. In contrast, UTP-Bench models travel planning as a stochastic problem grounded in empirical uncertainty data.

Evaluation of LLM-Generated Travel Plans. Early travel-planning benchmarks primarily relied on binary constraint satisfaction (Xie et al., 2024), later extended with personalization and itinerary-quality metrics (Chen et al., 2024; Singh et al., 2024). TripCraft (Chaudhuri et al., 2025) further introduced fine-grained spatial, temporal, and persona-based evaluation. However, existing frameworks remain deterministic and overlook real-world uncertainty such as transit delays and crowd congestion. UTP-Bench addresses this gap through three uncertainty-aware metrics (BAS, TDAS, CATS) that evaluate itinerary robustness under empirical uncertainty distributions. To our knowledge, UTP-Bench is the first benchmark to explicitly model stochastic robustness in LLMbased travel planning.

## 3 UTP-Bench Dataset Curation

UTP-Bench is a benchmark for evaluating LLMs in uncertainty-aware, constraint-rich travel itinerary planning. It assesses whether models can generate itineraries that are feasible, preferencealigned, and robust to real-world uncertainties, including transit delays, crowd fluctuations, and dynamic scheduling conditions. The benchmark comprises 1,000 travel queries spanning 3-, 5-, and 7-day itineraries across diverse planning scenarios of varying complexity. Each query is paired with a human-annotated gold-standard itinerary, alongside model-generated itineraries produced under uncertainty-aware prompting settings.

<table><tr><td>Database</td><td>Data Entries (#)</td></tr><tr><td>City Set</td><td>504</td></tr><tr><td>States &amp; Union Territories</td><td>33</td></tr><tr><td>Flights†</td><td>29,346</td></tr><tr><td>Trains† Buses†</td><td>68,068</td></tr><tr><td>Restaurants†</td><td>34,925</td></tr><tr><td>Attractions†</td><td>3,433</td></tr><tr><td>Accommodations</td><td>2,990</td></tr><tr><td>Events</td><td>3,670</td></tr><tr><td></td><td>783</td></tr><tr><td>Nearest Transit Stop Distance Matrix</td><td>10,025 253,513</td></tr></table>

Table 1: Dataset statistics in UTP-Bench. Components marked with † include uncertainty-aware signals

## 3.1 Uncertainty-Aware Data Curation

The benchmark is constructed from large-scale travel data covering 504 cities across 33 Indian states and union territories (Table 1). Data is collected from real-world sources through web scraping and open-source resources such as Open-StreetMap (Appendix A). The database includes city-level information on transportation, attractions, restaurants, accommodations, and related metadata, with incomplete records removed or normalized for consistency.

To model realistic travel uncertainty, we augment the dataset with empirical transit-delay statistics and crowd-density patterns. Specifically, we collect historical delay data for flights, trains, buses, and cabs, along with crowd-density information for attractions and restaurants across 20 major Indian cities (Appendix A). These uncertainty signals capture real-world variability that can undermine otherwise feasible itineraries, such as cascading delays or congestion during peak hours. By incorporating such factors, UTP-Bench enables evaluation of itinerary robustness beyond constraint satisfaction.

## 3.2 Constraints and Persona Data

UTP-Bench builds upon the constraint and persona framework introduced in TripCraft (Chaudhuri et al., 2025), while extending it with an explicit uncertainty-aware layer. The benchmark includes commonsense constraints, hard constraints, persona information, and newly introduced uncertainty constraints that are absent from prior benchmarks.

## Uncertainty Constraints.

These constraints arise from two major sources. First, transit delays: itineraries must allocate sufficient buffer to absorb empirically observed delays across flights, trains, buses, and cabs. Second, crowd-induced variability: visit durations at attractions and restaurants depend on the time of day and the expected crowd intensity, so plans that schedule activities during highly congested periods without appropriate adjustments are penalized. These uncertainty constraints form the basis of our uncertainty-aware evaluation metrics.

Persona Information. Each query in UTP-Bench is paired with a persona profile that influences itinerary generation along four dimensions: traveler type, purpose of travel, spending preference, and location preference. Traveler type distinguishes, for instance, between laid-back and adventure-oriented travelers; purpose of travel captures motivations such as relaxation, adventure, or cultural exploration; spending preference shapes cost-sensitive decisions; and location preference reflects favored environments such as beaches, mountains, heritage cities, wildlife destinations, or religious sites. In addition, UTP-Bench augments each query with a risk profile, which governs how much uncertainty a traveler is willing to tolerate. A Risk-Tolerant traveler accepts tighter schedules and smaller buffers, a Risk-Averse traveler prefers conservative timing and larger recovery margins, and a Risk-Optimized traveler lies between these two extremes. This allows the benchmark to evaluate not only whether a plan is feasible, but also whether it is appropriately calibrated to the traveler’s uncertainty tolerance (See Table 2).

Commonsense and Hard Constraints. These constraints ensure basic itinerary realism and coherence. For example, events cannot repeat across days, meals must maintain appropriate temporal gaps, and PoIs must follow a valid chronological order. Additionally, except for the final departure day, each itinerary day begins and ends at the designated accommodation with feasible transitions across activities. UTP-Bench further enforces hard constraints on travel feasibility and diversity (e.g., budget, room rules, and accommodation type) (Xie et al., 2024). To encourage activity diversity, attractions are mapped to 20 semantic categories spanning different travel interests.

<table><tr><td colspan="2">Constraint</td></tr><tr><td colspan="2">Uncertainty Constraints</td></tr><tr><td>Transit Delay Buffer</td><td>Buffers must account for empirical delay distribu- tions across flights, trains, buses, and cabs.</td></tr><tr><td>Crowd-Induced Delay</td><td>Visit and wait times must reflect peak-hour crowd density patterns.</td></tr><tr><td>Activity Buffer Gap</td><td>Gaps between activities must absorb real-world transit and crowd disruptions.</td></tr><tr><td>Risk Profiles</td><td></td></tr><tr><td>Risk-Tolerant</td><td>Tighter scheduling with limited additional buffer.</td></tr><tr><td>Risk-Optimized</td><td>Balanced schedule with moderate buffer allocation.</td></tr><tr><td>Risk-Averse</td><td>Conservative planning with larger safety margins.</td></tr></table>

Table 2: Constraint and persona details in UTP-Bench. All hard and commonsense constraints are retained from TripCraft, TravelPlanner (See Table 8, Appendix C).

## 3.3 Query Construction

Travel queries are created by sampling a structured set of planning inputs, including departure city, destination, travel dates, budget, and persona profile. Trip duration determines the geographic scope of the itinerary: 3-day plans focus on a single destination city, while 5-day and 7-day plans cover one state with visits to two and three cities, respectively. Each query is further associated with hard constraints, persona attributes, and a risk profile. These components are combined using GPT-5 in a few-shot setting to produce naturalistic travelplanning queries that are both diverse and structurally controlled. To capture different levels of planning difficulty, queries are categorized as Easy, Medium, or Hard based on the density of available PoIs, complexity of transit connectivity, and the degree of scheduling constraints required by the itinerary using GPT5.

## 3.4 Annotation and Refinement

Each query is paired with a gold-standard itinerary created by 14 trained human annotators through multiple rounds of refinement. Our annotation process is not purely manual: annotators are supported by purpose-built helper scripts that surface relevant contextual information during planning, including historical transit-delay statistics, crowddensity patterns, and expected attraction-visit durations, drawn directly from the database. This automated script-assisted workflow allows annotators to produce itineraries that are not only logically valid and preference-aligned, but also genuinely uncertainty-aware. In particular, annotators must reason about temporal feasibility, spatial coherence, crowd-sensitive scheduling, transit-delay absorption, and the risk profile associated with each query. Each annotated itinerary is accompanied by a written rationale that explains the key planning decisions, the constraint-satisfaction logic, and the uncertainty-buffer choices, thereby improving interpretability. See annotation guidelines in Appendix E, Table 17.

![](images/33e38c89f068a3114467a4882a249ba4318ace4ae3efdb3dbcac14d71344a86d.jpg)  
Figure 2: Hourly crowd-level distributions across attractions and restaurants. Each bar represents the number of venues within different crowd categories at a given hour.

## 3.5 Quality Control

To ensure the reliability of the benchmark, we adopt a multi-stage quality-control process. First, annotation proceeds through iterative refinement rounds, during which annotators revise itineraries based on feedback from domain experts. Second, all query-plan pairs undergo final manual review to verify feasibility, optimality, and consistency with the intended persona and risk profile. Third, we use automated evaluation scripts to check structural validity, including temporal consistency, entity correctness, and compliance with key itinerary constraints. This combination of human refinement, domain-expert review, and script-based verification ensures that the benchmark captures realistic travel-planning behavior while maintaining high annotation quality.

## 3.6 Analysis of Uncertainty Distribution

Transportation uncertainty in our framework is modeled using historical delay signals across rail, flight, and road travel. Train delays range from 40.8-46.7 mins on average, while delayed flights exhibit a mean delay of 40.59 mins and road travel shows the highest average delay (52.2 mins). Across all modes, mean delays consistently exceed median delays, indicating skewed distributions with occasional large disruptions. Figure 2 further illustrates temporal crowd dynamics, showing that attractions experience higher crowd density during daytime and afternoon hours, whereas restaurants exhibit stronger evening concentration. Together, these observations highlight the strong temporal dependence of transportation delays and crowd patterns, motivating uncertainty-aware itinerary generation.

## 4 Evaluation Metrics

Plan feasibility is assessed using the hard, commonsense, and uncertainty constraints mentioned in Section 3.2. UTP-Bench uses a comprehensive eight-metric evaluation suite that captures temporal, spatial, sequential, persona-specific, and uncertainty-aware nuances of a travel plan.

We adopt five metrics, namely Temporal Meal Score, Temporal Attraction Score, Spatial Score, Persona Score, and Ordering Score from TripCraft (Chaudhuri et al., 2025), retaining their formulations and parameter derivation methodology. Further, we introduce three novel uncertainty-aware metrics, namely Buffer Adequacy Score (BAS), Crowd-Aware Timing Score (CATS), and Transport Delay Absorption Score (TDAS) to evaluate plan robustness under the stochastic disruptions characteristic of real-world travel.

Buffer Adequacy Score (BAS): The Buffer Adequacy Score evaluates the temporal realism of an itinerary by assessing whether the allocated buffer time at each Point of Interest (POI) is consistent with its expected visit duration, local crowd conditions, and the traveler’s risk tolerance.

For a POI with arrival time $t _ { a }$ and departure time

$t _ { d } .$ , the actual buffer time $B _ { a }$ is defined as:

$$
B _ { a } = ( t _ { d } - t _ { a } ) - v\tag{1}
$$

where v denotes the base visit duration of the POI.

To account for uncertainty, we define an expected buffer range $[ a _ { c } , b _ { c } ]$ that adapts to both crowd congestion and traveler preferences:

$$
\begin{array} { c } { { a _ { c } = l o w \_ m u l t \cdot v \cdot C _ { f } } } \\ { { b _ { c } = h i g h \_ m u l t \cdot v \cdot C _ { f } } } \end{array}\tag{2}
$$

where $C _ { f }$ represents the crowd congestion factor, and low\_mult, high\_mult are scaling parameters determined by the traveler’s risk profile. We ensure that $h i g h _ { - }$ \_mul > low\_mul and $C _ { f } > 1$

We define a deviation penalty $Q _ { i }$ that quantifies the extent to which the allocated buffer deviates from the acceptable range. We use min rather than max because max would measure the distance to the farther boundary, which would over-penalize the itinerary.

$$
Q _ { i } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f } \ B _ { a } \in [ a _ { c } , b _ { c } ] } \\ { \mathrm { ~ m i n ~ } \big ( | B _ { a } - a _ { c } | , | B _ { a } - b _ { c } | \big ) } \\ { b _ { c } - a _ { c } } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{3}
$$

Finally, for an itinerary containing N POI transitions, the Buffer Adequacy Score is computed as:

$$
\mathrm { B A S } = 1 - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } m i n ( 1 , Q _ { i } )\tag{4}
$$

A BAS value closer to 1 indicates that the itinerary allocates buffer times are well-aligned with expected uncertainty, while lower values reflect either overly tight or excessively conservative scheduling. Crowd-Aware Timing Score (CATS). The Crowd-Aware Timing Score evaluates how effectively POI visits are scheduled to avoid peak congestion and align with low-crowd periods. For each POI visit, we compute a crowd intensity score based on the overlap of the visit duration with predefined crowd windows. This is obtained as a duration-weighted average of crowd levels across the visited time intervals. Details in Appendix B.2. This assigns higher scores to visits occurring during less crowded periods. To further refine this measure, we incorporate two adjustments: (i) a quiet-hour bonus, which rewards visits overlapping with low-density time windows, and (ii) a peak-hour penalty, which penalizes visits scheduled during high-density periods. These adjustments are applied proportionally to the fraction of time spent in quiet or peak intervals.

The final score for each POI is normalized to the range [0, 1], where higher values indicate better alignment with favorable crowd conditions. We convert this into a penalty term score as detailed in Appendix B.2

For an itinerary with N POI visits, the overall Crowd-Aware Timing Score is computed as:

$$
\mathrm { C A T S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s c o r e _ { i }\tag{5}
$$

A higher CATS value indicates that the itinerary consistently schedules visits during less crowded periods, reflecting better crowd-aware planning.

Transport Delay Absorption Score (TDAS). The Transport Delay Absorption Score evaluates whether the buffer allocated for each transport segment is sufficient to absorb realistic delays without propagating disruptions to subsequent activities in the itinerary.

For a transport segment j, the planner implicitly allocates a buffer defined as the difference between the planned travel duration and the corresponding historical duration:

$$
B _ { j } = p l a n n e d _ { - } d u r a t i o n _ { j } - h i s t _ { - } d u r a t i o n _ { j }\tag{6}
$$

To assess adequacy, we estimate an expected delay E[D] for each segment based on historical data, which varies across transport modes (e.g., flights, trains, and road-based travel). This expected delay captures both the likelihood and magnitude of delays observed in practice.

Given $E [ D ]$ , we define an acceptable buffer range that scales with the traveler’s risk tolerance. Specifically, lower and upper bounds are obtained by applying risk-dependent multipliers to $E [ D ]$ , reflecting how conservative or aggressive the itinerary is with respect to delay absorption. We then compute a deviation penalty $Q _ { j }$ for each segment, which is zero when the allocated buffer lies within the acceptable range, and increases proportionally as the buffer deviates from this range. This captures both under-allocation (fragile plans) and over-allocation (inefficient plans).

For an itinerary with M transport segments, the overall TDAS is defined as:

$$
\mathrm { T D A S } = 1 - \frac { 1 } { M } \sum _ { j = 1 } ^ { M } m i n ( 1 , Q _ { j } )\tag{7}
$$

A higher TDAS value indicates that the itinerary allocates buffers that are well-aligned with expected transport delays. (Details in Appendix B)

## 5 Experimental Results and Analysis

We adopt the direct sole planning strategy (Xie et al., 2024; Singh et al., 2024), modifying the prompt to include PoI lists, uncertainty parameters, and a refined one-shot example tailored to our constraints. Additionally, we incorporate natural language descriptions of the parameterised distributions modelling the metrics

including traveller risk profiles, congestion tiers, and transport delay statistics in the prompt (details in Appendix C). We evaluate GPT-5 (Singh et al., 2026), Qwen3 (14B) (Yang et al., 2025), Mistral-7B-Instruct-v0.3(Jiang et al., 2023) and Phi-4 mini-Instruct (Microsoft et al., 2025) across 3-day, 5- day, and 7-day travel plans using the UTP-Bench<sup>2</sup>. Results are summarized in Tables 3.

Query: Plan a 3 days trip from Pune to Indore from   
2025-11-08 to 2025-11-10, with a budget of INR   
24,073. Traveler Type: Culture Enthusiast; Risk Tolerant.   
Plan: {   
"days": 3,   
"current\_city": "from Pune to Indore",   
"transportation": "Flight 6E-284, Pune to Indore,   
Dep: 05:15, Arr: 07:10",   
"Day-1":   
{Indore Airport, arrival   
buffer until 07:40;   
Indore Marriott Hotel, check-in from   
08:00 to 08:30;   
...}   
"Day-2": {   
Shree Bijasan Mata Mandir, visit from   
09:30 to 12:30 (Above Avg crowd at   
16:00–17:00 avoided by morning visit,   
15 min buffer included);   
Indore Kitchen Restaurant, visit from   
13:00 to 14:00;   
Fagun Restaurant, visit from 19:30   
to 21:00 (15 min buffer for Above   
Avg crowd at 19:00–22:00);   
Indore Marriott Hotel, stay from   
21:15 to 07:30."   
}   
"Day-3":{...}   
Remark: Good CATS score: the planner consistently sched  
ules attraction visits during morning quiet windows, deliber   
ately avoiding known peak crowd hours. Shree Bijasan Mata   
Mandir is visited 09:30–12:30, well before its Above Average   
crowd window at 16:00–17:00. By front-loading visits into   
low-intensity hours, nearly all POIs incur no peak-hour penalty   
and benefit from a quiet-hour bonus, resulting in high per-POI   
crowd scores and a strong aggregate CATS.  
Figure 3: Example of Crowd-Aware Scheduling Avoiding Peak-Hour Congestion

## 5.1 Key Observations and Discussions

Observation 1. Trade-offs between qualitative coherence and constraint adherence. Table 3 shows distinct trade-offs between qualitative coherence and feasibility under uncertainty-aware planning. GPT-5 consistently achieves the strongest qualitative performance, leading in spatial reasoning $( S _ { \mathrm { s p a t i a l } } ) _ { \mathrm { : } }$ , persona preservation $( P _ { \mathrm { p e r s o n a } } )$ , and meal coherence $( T _ { \mathrm { m e a l } } )$ , while attaining the highest micro-CPR at both 3-day (88.86%) and 7-day (87.44%) settings. Mistral-7B-Instruct-v0.3 performs best on temporal attraction alignment $( T _ { \mathrm { a t t r } } )$ and ordering coherence $( O _ { \mathrm { o r d } } )$ , maintaining a perfect 100% delivery rate across all horizons. In contrast, Qwen3 exhibits stronger hard-constraint adherence, achieving the highest macro pass rates (12.78%, 9.16%, and 10.45% for $3 \mathrm { - } , 5 \mathrm { - }$ , and 7-day settings, respectively), whereas Phi-4-mini-Instruct shows the weakest qualitative performance with increasing degradation as planning horizon grows. These findings suggest that stronger qualitative realism does not necessarily imply stronger feasibility, revealing distinct trade-offs between coherent planning and constraint satisfaction under uncertainty.

Observation 2. The robustness gap exposed by uncertainty-aware metrics. Traditional passrate metrics become less informative once plans violate key constraints. Although all models maintain high delivery rates (92–100%), final pass rates rapidly deteriorate as the planning horizon increases and approach near-zero values, making it difficult to distinguish partially robust plans from fundamentally fragile ones. In contrast, the proposed uncertainty-aware metrics (BAS, CATS, and TDAS) reveal distinct model behaviours.

For BAS, GPT-5 and Qwen3 consistently achieve low scores (1–12%), indicating poor temporal buffer allocation, whereas Mistral-7B-Instructv0.3 substantially improves with longer horizons (e.g., 14.99%→40.97% for $\mathbf { R i s k } _ { t o l } )$ . Interestingly, $\mathrm { R i s k } _ { o p t }$ consistently yields the lowest BAS scores across models, suggesting that balanced scheduling with moderate buffering is more difficult than either aggressive or conservative planning.

For CATS, all models achieve substantially higher scores (47–64%) than BAS and TDAS, suggesting that avoiding crowded periods is comparatively easier than modeling temporal or transportation uncertainty. Phi-4-mini-Instruct achieves the strongest performance (63.94% for $\mathbf { R i s k } _ { t o l }$ at $7 -$ day), while Qwen3 consistently outperforms GPT-

<table><tr><td rowspan="2">Model</td><td rowspan="2">Category</td><td rowspan="2">Delivery Rate (%)</td><td colspan="2">CPR (%)</td><td colspan="2">HCPR (%)</td><td rowspan="2">Final Pass Rate (%)</td><td rowspan="2">Tmeal</td><td rowspan="2"></td><td rowspan="2">Tattr Sspatial Ppersona</td><td rowspan="2"></td><td rowspan="2">Oord</td><td rowspan="2">BAS(%) ↑</td><td colspan="2"></td><td rowspan="2"></td><td colspan="2">CATS(%) ↑</td><td rowspan="2"></td><td colspan="2">TDAS(%)↑</td></tr><tr><td>Micro Macro</td><td></td><td>Micro Macro</td><td></td><td>Risktol Riskopt</td><td>Riskav</td><td>Risktol</td><td>Riskopt</td><td>Riskav Risktol</td><td>Riskopt</td><td>Riskav</td></tr><tr><td rowspan="4">GPT-5</td><td>3-day</td><td>99.76</td><td>88.86</td><td>15.24</td><td>38.64</td><td>5.00</td><td>0.95</td><td>0.669</td><td>0.004</td><td>0.903</td><td>0.494</td><td>0.692</td><td>6.06</td><td>1.71</td><td>1.41</td><td>52.62</td><td>53.05</td><td>53.90</td><td>9.88</td><td>3.23 10.86 5.59</td></tr><tr><td>5-day</td><td>99.29</td><td>87.94</td><td>6.43</td><td>32.68</td><td>1.19</td><td>0.24</td><td>0.727</td><td>0.009</td><td>0.906</td><td>0.506</td><td>0.864</td><td>7.90</td><td>0.51</td><td>1.45</td><td>47.44</td><td>46.57</td><td>49.28</td><td>6.39</td><td>8.67</td></tr><tr><td>7-day</td><td>99.52</td><td>87.44</td><td>3.11</td><td>34.85</td><td>2.38</td><td>0.00</td><td>0.748</td><td>0.006</td><td>0.911</td><td>0.502</td><td>0.948</td><td>7.63</td><td>1.67</td><td>1.58</td><td>47.71</td><td>48.57</td><td>48.93</td><td>7.36 2.43</td><td>11.85</td></tr><tr><td>3-day</td><td>100.00</td><td>84.57</td><td>9.04</td><td>39.48</td><td>12.78</td><td>0.00</td><td>0.542 0.007</td><td></td><td>0.859</td><td>0.467</td><td>0.704</td><td>5.32</td><td>2.67</td><td>1.52</td><td>56.12</td><td>55.66</td><td>54.92 7.58</td><td>3.65</td><td>4.58</td></tr><tr><td rowspan="4">Qwen3</td><td>5-day</td><td>100.00</td><td>83.69</td><td>4.28</td><td>32.08</td><td>9.16</td><td>0.30 0.580</td><td>0.017</td><td>0.826</td><td>0.465</td><td>0.876</td><td>11.54</td><td>2.42</td><td>2.67</td><td>51.89</td><td>52.13</td><td>56.39</td><td>1.57</td><td>3.77 6.32</td></tr><tr><td>7-day</td><td>100.00</td><td>82.06</td><td>2.74</td><td>31.29 10.45</td><td></td><td>0.00 0.632</td><td>0.013</td><td>0.880</td><td>0.492</td><td>0.949</td><td>8.79</td><td>3.26</td><td>3.95</td><td>55.19</td><td>53.65</td><td>53.22</td><td>3.78</td><td>3.98 2.67</td></tr><tr><td></td><td>100.00</td><td>87.81</td><td>10.08</td><td>36.95</td><td>2.59</td><td>0.00 0.434</td><td>0.072</td><td>0.891</td><td>0.291</td><td>0.841</td><td>14.99</td><td>7.56</td><td>10.03</td><td>53.65</td><td>59.17</td><td>56.29</td><td>3.28</td><td>0.45 1.14</td></tr><tr><td>3-day 5-day</td><td>100.00</td><td>87.69</td><td>10.31</td><td>30.09</td><td>1.22</td><td>0.61 0.436</td><td>0.061</td><td>0.811</td><td>0.209</td><td>0.929</td><td>28.74</td><td>6.58</td><td>15.59</td><td>49.03</td><td>52.73</td><td>58.11</td><td>1.14</td><td>0.97 0.75</td></tr><tr><td rowspan="3">Instruct v0.3</td><td>7-day</td><td>100.00</td><td>78.34</td><td>2.67 36.00</td><td>0.00</td><td></td><td>0.00 0.365</td><td>0.062</td><td>0.805</td><td>0.279</td><td>0.975</td><td>40.97</td><td>20.15</td><td>38.53</td><td>60.49</td><td>59.93 51.83</td><td>1.67</td><td>1.03</td><td>1.94</td></tr><tr><td>3-day</td><td>99.22</td><td>87.06</td><td>10.27</td><td>35.53</td><td></td><td>0.79 0.193</td><td>0.065</td><td>0.659</td><td>0.277</td><td>0.809</td><td>8.37</td><td>6.70</td><td>9.57</td><td>58.31</td><td>58.70 57.08</td><td>2.34</td><td>1.29</td><td>2.39</td></tr><tr><td>5-day</td><td>92.94</td><td>84.97</td><td>11.89 28.66</td><td>3.16 0.75</td><td></td><td>0.38 0.153</td><td>0.059</td><td>0.677</td><td>0.224</td><td>0.845</td><td>11.67</td><td>7.71</td><td>14.56</td><td>57.16</td><td>55.58</td><td>56.11 0.74</td><td>0.37</td><td>2.41</td></tr><tr><td rowspan="2">Instruct</td><td>7-day</td><td>92.79</td><td>76.12</td><td>3.61</td><td>17.32 0.91</td><td></td><td>0.00 0.101</td><td>0.055</td><td>0.351</td><td>0.054</td><td>0.757</td><td>11.05</td><td>5.57</td><td>17.86</td><td>63.94</td><td>59.93</td><td>53.72</td><td>0.74 1.26</td><td>4.75</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Combined pass-rate, qualitative and Uncertainty evaluation metrics across 3-day, 5-day, and 7-day plans. Pass-rate metrics include Delivery Rate, Commonsense Pass Rate (CPR), Hard Constraint Pass Rate (HCPR) with Micro / Macro variants, and Final Pass Rate. Qualitative evaluation metrics include: $T _ { \mathrm { m e a l } } , T _ { \mathrm { a t t r } } , S _ { \mathrm { s p a t i a l } } , P _ { \mathrm { p e r s o n a } } , O _ { \mathrm { o r d } }$ and BAS, CATS and TDAS metrics across different traveller type(Risk tolerance, optimizers and averse). Bolded values represent the best result among the four models.

5; however, stronger crowd-aware behaviour does not translate into stronger overall robustness (see Example 3).

For TDAS, all models obtain uniformly low scores (0.37–11.85%), substantially below human performance. GPT-5 performs best, reaching 11.85% for $\mathrm { R i s k } _ { a v }$ travellers, whereas Qwen3 exhibits noticeable degradation with increasing planning horizon (7.58%→1.57% for $\mathbf { R i s k } _ { t o l } )$ . Mistral-7B and Phi-4 remain consistently weak across traveller profiles.

Overall, uncertainty-aware metrics reveal modelspecific failure patterns hidden by binary evaluation: GPT-5 and Qwen3 struggle with temporal buffering, Qwen3 degrades at longer horizons, Phi-4’s strong crowd-aware behaviour masks weak temporal robustness, and all models consistently struggle with the $\mathrm { R i s k } _ { o p t }$ traveller profile.

Observation 3. Limitations in LLM-generated itineraries. Despite uncertainty-aware prompting, LLMs exhibit several systematic failure modes. Models frequently generate temporal inconsistencies (such as, PoI visits can extend beyond planned departure times, meal schedules may not align with natural dining hours, and activity timestamps can be misordered) disrupting itinerary flow (see Appendix D). They also struggle to adapt activity density and temporal buffering across traveller risk profiles, as reflected by consistently low BAS scores relative to human performance. Moreover, although GPT-5 achieves stronger spatial reasoning scores $( S _ { \mathrm { s p a t i a l } } )$ , all models rely on optimistic transit assumptions in multi-city settings, leading to fragile itineraries under realistic delays.

These limitations become more prominent with increasing planning horizon. Performance degrades from 3-day to 7-day itineraries, while human-annotated plans also decline from 91% to 63%, indicating that long-horizon planning is inherently challenging even for humans. Nevertheless, model performance remains substantially below human levels across all settings, suggesting that current LLMs still struggle to reason effectively over stochastic travel dynamics, user risk preferences, and long-range temporal dependencies.

## 5.2 Agentic baseline.

To evaluate the effectiveness of explicit uncertainty modeling beyond agentic planning, we additionally compare against ATLAS (Choi et al., 2026), an agentic framework for real-world travel planning. Table 4 provides a complementary comparison between agentic task decomposition and our uncertainty-aware planning approach.

Analysis. ATLAS does not consistently outperform our uncertainty-aware prompting strategy despite its agentic planning framework. While it achieves competitive performance on selected crowd-awareness measures such as CATS, its performance is lower on delivery rate, constraint preservation, and several uncertainty-aware metrics, particularly BAS and TDAS for longer itineraries. These results indicate that agentic decomposition alone does not fully address the challenges posed by UTP-Bench and highlight the importance of explicitly modeling travel uncertainty.

## 5.3 Performance Without Uncertainty

We also evaluated GPT-5 on the UTP-Bench dataset without explicitly modeling uncertainty factors, such as flight delays, traffic, and weather disruptions. This setting provides a deterministic baseline for assessing the model’s ability to satisfy travel planning constraints before introducing real-world uncertainty. Results are reported in Table 5 and 6.

Table 4: Combined pass-rate and uncertainty metrics for Phi-4-mini across different traveler types (risk-tolerant, risk-optimistic, and risk-averse) for 3-day, 5-day, and 7-day plans using agentic baseline ATLAS(Choi et al., 2026).
<table><tr><td></td><td></td><td colspan="2">CPR</td><td colspan="2">HCPR</td><td></td><td></td><td colspan="2">BAS</td><td></td><td colspan="3">CATS</td><td colspan="3">TDAS</td></tr><tr><td>Type</td><td>Del</td><td></td><td></td><td></td><td></td><td>Final Micro Macro Micro Macro Pass Rate</td><td>Risktol</td><td>Riskopt</td><td>Riskav</td><td>Risktol</td><td></td><td> $R i s k _ { o p t }$ </td><td>Riskav</td><td> $R i s k _ { t o l }$ </td><td>Riskopt</td><td> $R i s k _ { a v e }$ </td></tr><tr><td>3d</td><td>100.00</td><td>87.31</td><td>8.93</td><td>40.31</td><td>40.32</td><td>5.36</td><td></td><td>15.23</td><td>6.57</td><td>5.00</td><td>51.72</td><td>50.65</td><td>51.51</td><td>1.34</td><td>2.64</td><td>3.12</td></tr><tr><td>5d</td><td>100.00</td><td>82.83</td><td>6.74</td><td>15.55</td><td>15.55</td><td>1.82</td><td>25.34</td><td>10.54</td><td>10.04</td><td>50.62</td><td>54.99</td><td>55.52</td><td></td><td>0.85</td><td>0.42</td><td>2.84</td></tr><tr><td>7d</td><td>99.74</td><td>78.88</td><td>7.02</td><td>0.51</td><td>0.53</td><td>0.00</td><td>27.57</td><td>15.40</td><td>13.03</td><td>51.98</td><td>51.08</td><td>52.33</td><td></td><td>0.00</td><td>0.00</td><td>0.87</td></tr></table>

The qualitative metrics remain relatively stable across itinerary lengths. The meal-related score $( T _ { \mathrm { m e a l } } )$ increases from 0.66 for 3-day plans to 0.74 for 7-day plans, while spatial consistency $( S _ { \mathrm { s p a t } } )$ remains within 0.89–0.91. Persona alignment $( P _ { \mathrm { p e r s } } )$ remains stable at approximately 0.47–0.48. The ordering score $( O _ { \mathrm { o r d } } )$ increases from 0.73 to 0.95 with increasing plan length, indicating stronger ordering consistency. The $T _ { \mathrm { a t t r } }$ values remain low across all settings (0.0057–0.0087), consistent with the metric definition, where lower values indicate better performance. Despite these strong qualitative scores, constraint-based metrics show clear degradation as itinerary length increases. Micro-CPR decreases from 84% for 3-day plans to 76% and 71% for 5- and 7-day plans, respectively, while macro-CPR decreases from 29% to 14% and 4.6%. Similarly, macro-HCPR decreases from 10.66% to 2.13% and 0%, while the Final Pass Rate falls from 2.5% to 0.677% and 0% across 3-, 5-, and 7-day plans.

The divergence between qualitative scores and constraint satisfaction highlights an important limitation of LLM-based travel planning: an itinerary can appear coherent and preference-aligned while violating underlying constraints. As itinerary length increases, interacting temporal, spatial, and planning constraints become harder to satisfy simultaneously. The zero Final Pass Rate for 7-day plans demonstrates this compounding effect even without explicit uncertainty, motivating uncertainty-aware planning in UTP-Bench.

## 6 Conclusion

We introduced UTP-Bench, a high-fidelity benchmark for uncertainty-aware travel itinerary planning that bridges the gap between deterministic planning assumptions and the stochastic nature of real-world travel. By incorporating fine-grained attraction categories, traveller risk profiles, and empirical transportation-delay and crowd-dynamics signals, UTP-Bench provides a realistic testbed for evaluating spatio-temporal reasoning in LLMbased itinerary generation. We further proposed three uncertainty-aware evaluation metrics namely BAS, CATS and TDAS to assess itinerary robustness beyond conventional constraint-satisfaction metrics. Experimental results demonstrate that although current LLMs can generate qualitatively coherent itineraries, they struggle to reason effectively over temporal uncertainty, transportation variability, traveller risk preferences, and long-horizon dependencies. Overall, UTP-Bench advances travelplanning evaluation from deterministic feasibility to robustness under realistic uncertainty, enabling more reliable and adaptive human-centric planning systems.

Table 5: Qualitative metrics and Delivery rates for GPT-5 (3,5,7-day plans).
<table><tr><td>Type</td><td> $T _ { \mathrm { m e a l } }$ </td><td> $T _ { \mathrm { a t t r } }$ </td><td> $S _ { \mathrm { s p a t } }$ </td><td> $P _ { \mathrm { p e r s } }$ </td><td> $O _ { \mathrm { o r d } }$ </td><td>Deliv. Rate</td></tr><tr><td>3-day</td><td>0.66</td><td>0.0087</td><td>0.89</td><td>0.48</td><td>0.73</td><td>94.1</td></tr><tr><td>5-day</td><td>0.72</td><td>0.0057</td><td>0.90</td><td>0.47</td><td>0.91</td><td>98.7</td></tr><tr><td>7-day</td><td>0.74</td><td>0.0063</td><td>0.91</td><td>0.48</td><td>0.95</td><td>96.8</td></tr></table>

Table 6: Pass-rates (CPR/HCPR) and Final Pass for GPT-5 (3,5,7-day plans).
<table><tr><td rowspan="2">Type</td><td colspan="2">CPR (%)</td><td colspan="2">HCPR (%)</td><td>Final</td></tr><tr><td>Micr.</td><td>Macr.</td><td>Micr.</td><td>Macr.</td><td>Pass</td></tr><tr><td>3-day</td><td>84</td><td>29</td><td>16.3</td><td>10.66</td><td>2.5</td></tr><tr><td>5-day</td><td>76</td><td>14</td><td>6.7</td><td>2.13</td><td>0.677</td></tr><tr><td>7-day</td><td>71</td><td>4.6</td><td>7.9</td><td>0.0</td><td>0.0</td></tr></table>

## Limitations

While UTP-Bench significantly enhances the realism and coherence of travel planning benchmarks, certain limitations remain. First, UTP-Bench is designed as a static benchmark and relies on historical statistics rather than real-time signals. Incorporating live transportation and traffic APIs represents a promising direction for future extensions. Second, our objective is not to propose a novel planning architecture, but rather to establish a benchmark and uncertainty-aware evaluation framework for robust travel planning. UTP-Bench provides a strong foundation for assessing LLM-driven travel planning, future research may explore diverse planning paradigms, including agentic and tool-augmented systems, and evaluate their ability to reason under realistic travel uncertainty. Another key constraint is the geographic scope, which is currently concentrated on Indian. If the necessary data becomes available, the construction pipeline can be extended to any regions, enabling broader geographic generalization. Our benchmark is currently designed primarily in English, which does not fully capture the multilingual nature of the Indian travel ecosystem. Expanding to regional languages would require accounting for language-specific differences in place names, cultural travel preferences, and transit terminology, which remain open challenges for future research.

## Ethical Considerations

Our study utilizes publicly available data from the Internet, which we have carefully scraped to construct our databases while ensuring compliance with relevant terms of use and ethical considerations. To safeguard privacy, we have fully anonymized sensitive personal details. However, with annotators consent, aggregate demographic statistics are provided in the Appendix E. AIassisted tools were used during the preparation of this work for language refinement and polishing. All research decisions, methodology, data curation, experiments, analyses, and conclusions were performed and verified by the authors.

## Acknowledgments

This research was partially supported by the Technology Innovation Hub (TIH), IIT Tirupati (IITTNiF/TPD/2024-25/P16). We sincerely thank the annotators for their efforts in annotating, validating, and quality-checking the dataset. We also thank the anonymous reviewers for their valuable comments and constructive feedback, which helped improve this work.

## References

Soumyabrata Chaudhuri, Pranav Purkar, Ritwik Raghav, Shubhojit Mallick, Manish Gupta, Abhik Jana, and Shreya Ghosh. 2025. TripCraft: A benchmark for spatio-temporally fine grained travel planning. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 17035–17064, Vienna, Austria. Association for Computational Linguistics.

Aili Chen, Xuyang Ge, Ziquan Fu, Yanghua Xiao, and Jiangjie Chen. 2024. Travelagent: An ai assistant for personalized travel planning. In arXiv preprint arXiv:2409.08069.

Jihye Choi, Jinsung Yoon, Jiefeng Chen, Somesh Jha, and Tomas Pfister. 2026. Atlas: Constraints-aware multi-agent collaboration for real-world travel planning. In International Conference on Learning Representations.

Rémi Coulom. 2006. Efficient selectivity and backup operators in monte-carlo tree search. In Proceedings of the 5th International Conference on Computers and Games, pages 72–83, Berlin, Heidelberg. Springer-Verlag.

Bin Deng, Yizhe Feng, Zeming Liu, Qing Wei, Xiangrong Zhu, Shuai Chen, Yuanfang Guo, and Yunhong Wang. 2025. RETAIL: Towards real-world travel planning for large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 14870–14902, Suzhou, China. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Ramya Jonnala, Gongbo Liang, Jeong Yang, and Izzat Alsmadi. 2025. Exploring the potential of large language models in public transportation: San antonio case study. In arXiv preprint arXiv:2501.03904.

Subbarao Kambhampati, Karthik Valmeekam, Matthew Marquez, and Lin Guan. 2023. On the role of large language models in planning. In Tutorial presented at the International Conference on Automated Planning and Scheduling (ICAPS), Prague.

Priyanshu Karmakar, Soumyabrata Chaudhuri, Shubhojit Mallick, Manish Gupta, Abhik Jana, and Shreya Ghosh. 2026. TripTide: A benchmark for adaptive travel planning under disruptions. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 40269–40292, San Diego, California, United States. Association for Computational Linguistics.

Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, and Xinyun Chen. 2025. Evolving deeper llm thinking. In arXiv preprint arXiv:2501.09891.

Microsoft, :, Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi ling Chen, Qi Dai, and 57 others. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. Preprint, arXiv:2503.01743.

Vishal Pallagani, Bharath Muppasani, Keerthiram Murugesan, Francesca Rossi, Biplav Srivastava, Lior Horesh, Francesco Fabiano, and Andrea Loreggia. 2023. Understanding the capabilities of large language models for automated planning. In arXiv preprint arXiv:2305.16151.

Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. 2024. ADaPT: As-needed decomposition and planning with language models. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 4226–4252, Mexico City, Mexico. Association for Computational Linguistics.

Jie-Jing Shao, Bo-Wen Zhang, Xiao-Wen Yang, Baizhi Chen, Siyu Han, Jinghao Pang, Wen-Da Wei, Guohao Cai, Zhenhua Dong, Lan-Zhe Guo, and Yu-Feng Li. 2026. Chinatravel: an open-ended travel planning benchmark with compositional constraint validation for language agents. In International Conference on Learning Representations.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 467 others. 2026. Openai gpt-5 system card. Preprint, arXiv:2601.03267.

Harmanpreet Singh, Nikhil Verma, Yixiao Wang, Manasa Bharadwaj, Homa Fashandi, Kevin Ferreira, and Chul Lee. 2024. Personal large language model agents: A case study on tailored travel planning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 486–514, Miami, Florida, US. Association for Computational Linguistics.

Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. 2023. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 2998–3009.

Maciej Swiechowski, Konrad Godlewski, Bartosz Saw-<sup>´</sup> icki, and Jacek Mandziuk. 2023. Monte carlo tree´

search: a review of recent modifications and applications: M. swiechowski et al.´ Artificial Intelligence Review, 56(3):2497–2562.

Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2023a. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. Advances in Neural Information Processing Systems, 36:38975–38987.

Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. 2023b. On the planning abilities of large language models-a critical investigation. Advances in neural information processing systems, 36:75993–76005.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, and 1 others. 2025. The rise and potential of large language model based agents: A survey. Science China information sciences, 68(2):121101.

Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. 2024. Travelplanner: a benchmark for real-world planning with language agents. In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24. JMLR.org.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Zirui Zhao, Wee Sun Lee, and David Hsu. 2023. Large language models as commonsense knowledge for large-scale task planning. Advances in neural information processing systems, 36:31967–31987.

## Overview of Appendix Sections

• Appendix A: Data Sourcing Details

• Appendix B: More about Evaluation

• Appendix C: Prompt and Annotation Details

• Appendix D: Case Studies

• Appendix E: Annotator Guidelines and Demographics

## A Data Sourcing Details

UTP-Bench is constructed entirely from realworld, India-specific data sources to ensure spatiotemporal consistency, operational accuracy, and uncertainty-awareness across all database components. Below, we detail the sourcing methodology and heuristics for each component.

## A.1 Restaurants

Restaurant details were extracted using the TripAdvisor API<sup>3</sup>, providing attributes including name, category, cuisine type, ratings, reviews, and price tier.The cost is denoted using the symbol (\$\$\$) rather than exact values. To estimate absolute prices, we leveraged city-specific restaurant price indices from Numbeo<sup>4</sup>, scaling them according to the price tier rating of each restaurant entry. Additionally, crowd density and peak visiting hours for restaurants were collected via the BestTime API<sup>5</sup>, providing hour-by-hour crowd intensity data across 20 major Indian cities. This data forms the empirical foundation for the crowd-induced delay modeling in UTP-Bench’s CATS and BAS uncertainty metrics.

## A.2 Attractions

Attraction details, including subcategories and ratings, were sourced from the TripAdvisor API.<sup>6</sup> Since a majority of attractions lacked predefined visit durations, we adopted the category-wise average durations from TripCraft, assigning each attraction’s duration as the mean of the categories it belonged to, ensuring a realistic time allocation (Table 7). Crowd density and peak-hour patterns for attractions were collected via the BestTime API, covering 20 major Indian cities and enabling crowdsensitive scheduling through the CATS metric. All

504 cities in the UTP-Bench city set were validated against attraction availability, with cities lacking sufficient PoI coverage filtered out during the data cleaning phase.

## A.3 Accommodations

Accommodation listings were collected via the TripAdvisor API,<sup>7</sup> providing attributes including name, property type, star rating, amenities, and location coordinates. Absolute nightly prices were calibrated using city-level accommodation cost baselines from Numbeo,<sup>8</sup> ensuring budget constraint evaluations are grounded in realistic Indiaspecific pricing. Accommodation listings are used as the designated start and end point for each day’s PoI sequence in UTP-Bench’s itinerary structure.

## A.4 Trains

Train schedules were sourced from Tripozo,<sup>9</sup> yielding 68,068 train records covering intercity and interstate routes across India. Each record includes train number, source and destination stations, departure and arrival times, journey duration, and availability by class. Historical delay data for trains was collected across multiple time horizons (1 week, 1 month, 3 months, 6 months, and 1 year), providing the empirical delay distributions used in UTP-Bench’s TDAS metric for train delay absorption modeling.

## A.5 Buses

Intercity bus schedules were sourced from Tripozo,<sup>10</sup>, yielding 34,925 bus records covering state and private operator routes across India. Each record includes operator name, source and destination cities, departure and arrival times, bus type, and fare. Road delay information for bus routes was obtained from the TomTom API,<sup>11</sup> covering 4,052 city-pair routes with empirical traffic delay distributions, providing the delay parameters used in TDAS bus delay modeling.

## A.6 Flights

Domestic flight schedules were sourced from Tripozo,<sup>12</sup> yielding 29,346 flight records across 67 Indian cities served by domestic carriers. Each record includes airline, flight number, source and destination airports, departure and arrival times, and fare class.Delay information is collected via FlightStats <sup>13</sup>. For each flight, on-time performance statistics including on-time probability p<sub>on-time</sub>, mean delay $\mu _ { \mathrm { l a t e } } .$ , and standard deviation $\sigma _ { \mathrm { l a t e } }$ were derived from historical records, forming the flight delay distribution parameters used in UTP-Bench’s TDAS metric.

## A.7 Events

Event data was scraped from District by Zomato, 14 covering a diverse range of concerts, sports, arts & theatre, music, and film events across India. The scraped dataset yields 783 events.

## A.8 Road Delay and Traffic Data

City-to-city road delay information was collected via the TomTom Traffic API,<sup>15</sup> covering 4,052 citypair routes across India. Each route entry includes Estimated delay, Historical delay and Free Flow Traffic.

## A.9 Crowd Density and Best-Time Data

Peak-hour crowd density patterns for attractions and restaurants were collected via the BestTime API,<sup>16</sup> which provides hour-by-hour crowd intensity classifications across a week for each venue. Data was collected for venues across 20 major Indian cities, forming the empirical backbone of UTP-Bench’s crowd-induced delay modeling. Crowd intensity is classified into five tiers (Low, Below Average, Average, Above Average, and High) with corresponding congestion factors $C _ { f }$ used in BAS and risk values R(t) used in CATS.

## A.10 Cost of Living

City-level cost-of-living data was sourced from Numbeo,<sup>17</sup> covering 20 major Indian cities. Attributes include average meal costs across price tiers, local transport fares, and accommodation price baselines. These data points are used for two purposes: (1) calibrating budget constraints in travel queries, and (2) converting relative price tier ratings from TripAdvisor into absolute cost estimates for restaurants and accommodations.

Table 7: Attraction visiting duration (hrs) for each category. Note that an attraction can belong to one or more categories.
<table><tr><td>Category</td><td>Duration (hrs)</td></tr><tr><td>Boat Tours &amp; Water Sports</td><td>3.5</td></tr><tr><td>Casinos &amp; Gambling</td><td>2.5</td></tr><tr><td>Classes &amp; Workshops</td><td>1.5</td></tr><tr><td>Concerts &amp; Shows</td><td>2.5</td></tr><tr><td>Food &amp; Drink</td><td>2.5</td></tr><tr><td>Fun &amp; Games</td><td>1.5</td></tr><tr><td>Museums</td><td>3.0</td></tr><tr><td>Nature &amp; Parks</td><td>4.5</td></tr><tr><td>Nightlife</td><td>2.5</td></tr><tr><td>Outdoor Activities</td><td>4.0</td></tr><tr><td>Shopping</td><td>1.5</td></tr><tr><td>Sights &amp; Landmarks</td><td>3.0</td></tr><tr><td>Spas &amp; Wellness</td><td>2.0</td></tr><tr><td>Water &amp; Amusement Parks</td><td>5.0</td></tr><tr><td>Zoos &amp; Aquariums</td><td>2.5</td></tr></table>

## A.11 Distance Matrix

All pairwise city-to-city distances and travel times were computed using the TomTom Routing $\mathsf { A P I } ^ { 1 8 }$ covering 253,513 city-pair entries across UTP-Bench’s 504-city set. For each Point of Interest, including accommodations, restaurants, and attractions, the nearest public transit stop was determined using geo distance computed via OpenStreetMap $\mathsf { A P I } ^ { 1 9 }$ , yielding 10,025 nearest transit stop entries. This enables realistic spatial reasoning and transitaware itinerary generation across India’s diverse urban and semi-urban geographies.

## B Evaluation Metrics (Details)

We introduce four complementary metrics to evaluate the temporal realism, crowd-awareness, transport robustness, and global schedule resilience of travel itineraries. Each metric is computed independently for human-annotated and LLM-generated plans.

## B.1 Buffer Adequacy Score (BAS)

Overview. Buffer Adequacy Score (BAS) measures the temporal realism and feasibility of a travel itinerary by evaluating whether the buffer time allocated at each Point of Interest (PoI) is consistent with the base visit duration, scaled appropriately to crowd congestion, and within the range expected for the traveller’s risk profile.

Inputs. For each PoI the following inputs are required:

• Place name and type (Attraction or Restaurant)

Table 8: Comprehensive constraint and persona description.
<table><tr><td>Constraint</td><td>Description</td></tr><tr><td>Environment Constraint</td><td></td></tr><tr><td>Unavailable Transportation</td><td>There is no available flight or driving information between the two cities.</td></tr><tr><td>Unavailable Attractions</td><td>There is no available attraction information in the queried city.</td></tr><tr><td>Commonsense Constraint</td><td></td></tr><tr><td>Within Sandbox</td><td>All information in the plan must be within the closed sandbox; otherwise, it will be considered a hallucination.</td></tr><tr><td>Complete Information</td><td>No key information should be left out of the plan, such as the lack of accommodation during travel.</td></tr><tr><td>Sufficient Meal Gaps</td><td>Meal timings must have a minimum gap of four hours between breakfast, lunch, and dinner to maintain a natural schedule.</td></tr><tr><td>Valid PoI list</td><td>The point-of-interest (PoI) list must follow strict validity rules: each day&#x27;s itinerary must begin and end at the designated accommodation, except on the final day when the traveler departs. The list should be limited to accommodations, attractions, and restaurants, ensuring adequate time gaps between flight arrivals and</td></tr><tr><td>Diverse Events</td><td>accommodation check-ins, as well as between accommodation check-outs and departures.</td></tr><tr><td>Within Current City</td><td>Event choices should not be repeated throughout the trip.</td></tr><tr><td>Reasonable City Route</td><td>All scheduled activities for the day must be located within that day&#x27;s city(ies). Changes in cities during the trip must be reasonable.</td></tr><tr><td>Diverse Restaurants</td><td>Restaurant choices should not be repeated throughout the trip.</td></tr><tr><td>Diverse Attractions</td><td>Attraction choices should not be repeated throughout the trip.</td></tr><tr><td>Non-conf. Transportation</td><td>Transportation choices within the trip must be reasonable. For example, having both “cab/taxi&quot; and &quot;flight&quot; would be considered a conflict.</td></tr><tr><td>Hard Constraint</td><td></td></tr><tr><td>Budget</td><td>The total budget of the trip.</td></tr><tr><td>House Rule</td><td>House rules include &quot;Safety Conduct&quot;, &quot;No smoking&quot;, &quot;No children&quot;, &quot;No pets&quot;, and &quot;Checkin/Checkout&quot;.</td></tr><tr><td>House Type</td><td>House types include &quot;Apartment Extended&quot;, &quot;Luxury Boutique&quot;, &quot;Resort/Beach&quot;, and &quot;Family Friendly&quot;.</td></tr><tr><td>Cuisine Group</td><td>Cuisines include &quot;African&quot;, &quot;American&quot;, &quot;Asian&quot;, &quot;Bar/Pub/Cafe&quot;, &quot;Eastern European/Russian&quot;, &quot;European&quot;, &quot;Indian Subcontinent&quot;, &quot;International Fusion&quot;, &quot;Latin/South American&quot;, &quot;Mediterranean/Middle Eastern&quot;,</td></tr><tr><td>Transportation</td><td>&quot;Pizza/Italian Specialties&quot;, and &quot;Seafood/Steak/Grill&quot;. Transportation options include &quot;No flight&quot; and &quot;No Cab&quot;.</td></tr><tr><td>Attraction Types</td><td>Each attraction belongs to one or more of 15 predefined categories, ensuring a well-distributed selection of activities.</td></tr><tr><td>Persona Components</td><td></td></tr><tr><td>Traveler Type</td><td>Defines how a traveler approaches their journey whether they seek relaxation in cozy spots or adrenaline- pumping adventures.</td></tr><tr><td>Purpose of Travel</td><td>Captures the main motivation behind the trip, whether it is to unwind, seek thrills, explore cultures, or connect with nature.</td></tr><tr><td>Spending Preference</td><td>Reflects the traveler&#x27;s budget and style, from luxurious indulgence to cost-conscious experiences.</td></tr><tr><td>Location Preference</td><td>Highlights preferred environments, such as beaches, mountains, cities, or wildlife-rich forests.</td></tr><tr><td>Uncertainty Constraints †</td><td></td></tr><tr><td>Transit Delay Buffer</td><td>Buffers must account for empirical delay distributions across flights, trains, buses, and cabs</td></tr><tr><td>Crowd-Induced Delay</td><td>Visit and wait times must reflect peak-hour crowd density patterns.</td></tr><tr><td>Activity Buffer Gap</td><td>Gaps between activities must absorb real-world transit and crowd disruptions.</td></tr><tr><td>Persona Components</td><td></td></tr><tr><td>Traveler Type</td><td>Laid-back or adventure-seeking travel style.</td></tr><tr><td>Purpose of Travel</td><td>Relaxation, adventure, cultural exploration, or nature.</td></tr><tr><td>Spending Preference</td><td>Luxury or budget-conscious, shaping all cost decisions</td></tr><tr><td>Location Preference</td><td>Beaches, mountains, heritage cities, wildlife, or religious destinations.</td></tr><tr><td>Risk Profile †</td><td>Risk-Tolerant, Risk Optimized, or Risk-Averse calibrates uncertainty tolerance.</td></tr><tr><td></td><td></td></tr></table>

• Arrival time $t _ { a }$ and departure time $t _ { d }$ (minutes from midnight)

• Base visit duration v (minutes), defined as follows:

For Attractions, v is taken from the visit\_duration column of the attractions dataset. For Restaurants, v is derived from the meal type determined by start hour, as shown in Table 9.

<table><tr><td>Meal Type</td><td>Start Hour</td><td>Duration v</td></tr><tr><td>Breakfast</td><td>06:00- 10:59</td><td>50 min</td></tr><tr><td>Lunch</td><td>11:00-15:59</td><td>60 min</td></tr><tr><td>Dinner</td><td>16:00-23:59</td><td>75 min</td></tr></table>

Table 9: Base visit duration v for restaurants by meal type.

The congestion tier factor $C _ { f }$ is parsed from the crowd description field (poi\_str) and mapped as

shown in Table 10.

<table><tr><td>Crowd Level</td><td> $C _ { f }$ </td></tr><tr><td>Low</td><td>1.00</td></tr><tr><td>Below Average</td><td>1.25</td></tr><tr><td>Average</td><td>1.50</td></tr><tr><td>Above Average</td><td>1.75</td></tr><tr><td>High</td><td>2.00</td></tr></table>

Table 10: Congestion tier factor $C _ { f }$ derived from Best-Time API crowd analysis.

The traveller risk-profile multipliers [low\_mult, high\_mult] are defined in Table 11.
<table><tr><td>Traveller Type</td><td>low_mult</td><td>high_mult</td></tr><tr><td>Risk Tolerant</td><td>0.50</td><td>1.00</td></tr><tr><td>Risk Optimized</td><td>1.00</td><td>1.25</td></tr><tr><td>Risk Averse</td><td>1.25</td><td>2.00</td></tr></table>

Table 11: Risk-profile buffer multipliers for BAS.

Step 1: Actual Buffer. The actual buffer is the free time allocated by the planner beyond the base visit duration:

$$
B _ { \mathrm { a c t u a l } } = ( t _ { d } - t _ { a } ) - v\tag{8}
$$

$B _ { \mathrm { a c t u a l } }$ captures the extra time padded by the planner on top of the standard visit duration; it is not travel time to the next place.

Step 2: Expected Buffer Range. The expected buffer range is derived from the base visit duration, the congestion factor, and the traveller risk profile with no hardcoded category thresholds:

$$
a _ { c } = l o w _ { - } m u l t \times v \times C _ { f }\tag{9}
$$

$$
b _ { c } = h i g h \_ m u l t \times v \times C _ { f }\tag{10}
$$

This ensures that a high-crowd PoI $( C _ { f } = 2 . 0 )$ naturally demands a larger buffer, and a risk-averse traveller requires a proportionally wider range. Both effects are fully data-driven.

Step 3: Quantitative Deviation Score. The distance of the actual buffer from the expected range, normalised by the range width:

$$
D = \left\{ \begin{array} { l l } { 0 , } & { B _ { \mathrm { a c t u a l } } \in [ a _ { c } , b _ { c } ] } \\ { \operatorname* { m i n } ( | B _ { \mathrm { a c t u a l } } - a _ { c } | , | B _ { \mathrm { a c t u a l } } - b _ { c } | ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{11}
$$

$$
Q = { \frac { D } { b _ { c } - a _ { c } } }\tag{12}
$$

Step 4: Final BAS. For an itinerary with N PoI visits:

$$
\boxed { \mathrm { B A S } = 1 - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } m i n ( 1 , Q _ { i } ) }\tag{13}
$$

Interpretation.

• BAS ≈ 1: Buffer is consistent with visit duration, crowd level, and traveller risk profile.

${ \mathrm { B A S } } \approx 0 { \mathrm { : } }$ Buffer is too short or too long relative to expected planning behaviour.

## B.2 Crowd-Aware Timing Score (CATS)

Overview. Crowd-Aware Timing Score (CATS) evaluates how well an itinerary aligns visit timings with crowd patterns at each PoI, using place-specific crowd-window data obtained from the BestTime API. CATS does not use PoI category or planner-assigned visit duration; scores are driven entirely by crowd data, making the metric fully independent of subjective category labels.

Inputs. For each PoI:

• Place name

• Visit start time $t _ { \mathrm { s t a r t } }$ and end time $t _ { \mathrm { e n d } }$

• avg\_besttime\_analysis containing:

$$
\begin{array} { r l } & { - \ \mathrm { c r o w : } } \\ & { \ \left[ s t a r t \_ h o u r , e n d \_ h o u r , l e v e l \right] } \\ & { - \ \mathrm { q } \mathrm { : } \ \mathrm { q u i e t } \ \mathrm { h o u r s } \ \mathrm { l i s t } } \\ & { - \ \mathrm { p } \mathrm { : } \ \mathrm { p e a k } \ \mathrm { h o u r s } \ \mathrm { l i s t } } \end{array}
$$

Crowd Intensity Mapping. Each crowd level is mapped to a numeric intensity using a uniform ordinal scale with equal intervals of 0.25 (Table 12).

<table><tr><td>Label</td><td>Meaning</td><td>Intensity</td></tr><tr><td>L</td><td>Low</td><td>0.00</td></tr><tr><td>BA</td><td>Below Average</td><td>0.25</td></tr><tr><td>A</td><td>Average</td><td>0.50</td></tr><tr><td>AA</td><td>Above Average</td><td>0.75</td></tr><tr><td>H</td><td>High</td><td>1.00</td></tr></table>

Table 12: Crowd level to numeric intensity mapping for CATS.

Step 1: Weighted Crowd Intensity. A visit may span multiple crowd windows. Intensity is computed as a weighted average proportional to actual overlap minutes:

$$
i n t e n s i t y = \frac { \displaystyle \sum _ { w } o v e r l a p _ { w } \times i n t e n s i t y _ { w } } { v i s i t \_ d u r a t i o n }\tag{14}
$$

where $o v e r l a p _ { w }$ is the number of minutes the visit overlaps crowd window w. This ensures that a visit spanning both a Below Average and an Above Average window is scored proportionally rather than on the start hour alone.

Step 2: Base Score.

$$
b a s e \_ s c o r e = 1 - i n t e n s i t y\tag{15}
$$

Step 3: Quiet-Hour Bonus. Overlap with quiet hours q is measured proportionally:

$$
\boxed { q u i e t \_ r a t i o = \frac { \displaystyle \sum _ { h \in q } o v e r l a p _ { h } } { v i s i t \_ d u r a t i o n } }\tag{16}
$$

The bonus scales with how uncrowded the place already is:

$$
q u i e t \_ b o n u s = q u i e t \_ r a t i o \times ( 1 - i n t e n s i t y )\tag{17}
$$

A higher quiet overlap combined with a lower intensity yields a larger reward. If q is absent, quiet\_bonus = 0.

Step 4: Peak-Hour Penalty. Overlap with peak hours p is measured proportionally:

$$
\sum _ { \substack { p e a k \_ r a t i o } } { \sum _ { \substack { n \in p } } { o v e r l a p } _ { h } }\tag{18}
$$

The penalty scales with how crowded the place is:

$$
p e a k \_ p e n a l t y = p e a k \_ r a t i o \times i n t e n s i t y\tag{19}
$$

A higher peak overlap combined with a higher intensity yields a larger penalty. If p is absent, peak\_penalty = 0.

Step 5: Per-PoI Score.

$$
\begin{array} { l c l } { { s c o r e _ { i } } } & { { = } } & { { \mathrm { c l i p } ( b a s e + q u i e t \_ b o n u s } } \\ { { } } & { { - } } & { { p e a k \_ p e n a l t y , 0 , 1 ) } } \end{array}\tag{20}
$$

Case handling based on data availability is summarised in Table 13.

<table><tr><td>Case</td><td>q</td><td>p</td><td>Effect</td></tr><tr><td>full</td><td>√</td><td>√</td><td>bonus and penalty applied</td></tr><tr><td>peak_only</td><td>X</td><td>√</td><td>Only penalty applied</td></tr><tr><td>quiet_only</td><td>√</td><td>X</td><td>Only bonus applied</td></tr><tr><td>cw_only</td><td>×</td><td>X</td><td>Intensity-based score only</td></tr><tr><td>no_data</td><td></td><td></td><td>PoI skipped from scoring</td></tr></table>

Table 13: CATS case handling based on available crowd data fields.

Step 6: Aggregate CATS. For an itinerary with N valid PoI visits:

$$
\boxed { \mathrm { C A T S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s c o r e _ { i } }\tag{21}
$$

Interpretation.

• CATS ≈ 1: Visits are well-timed around crowd patterns.

• CATS ≈ 0: Visits clash heavily with peak crowd hours.

## B.3 Transport Delay Absorption Score (TDAS)

Overview. Transport Delay Absorption Score (TDAS) measures whether the buffer embedded in each planned transport segment is sufficient to absorb historical delays across flights, trains, buses, and cabs without disrupting the overall itinerary.

Inputs. For each transport segment j:

$$
\begin{array} { r l } { \bullet \ \mathrm { T r a n s p o r t a t i o n } \qquad \mathrm { m o d e } \qquad m _ { j } \qquad } & { { } \in } \\ { \ \{ \mathrm { F l i g h t , ~ T r a i n , ~ C a b , ~ B u s } \} \qquad } & { { } } \end{array}
$$

• Planned duration from the itinerary (departure → arrival as written by the planner)

• Historical duration from the transport dataset (actual departure → actual arrival)

• OnTimePerformance (JSON) and DelayStats (JSON) for flights

• Multi-horizon average delay columns for trains

• Free-flow travel time and historical road delay for cabs/buses

• Traveller risk profile

Step 1: Buffer Extraction. The buffer is already embedded in the planned itinerary by the planner:

$$
B _ { j } = p l a n n e d _ { - } d u r a t i o n _ { j } - h i s t _ { - } d u r a t i o n _ { j }\tag{22}
$$

where planned\_duration is the arrival minus departure time as given in the plan (including planneradded buffer) and hist\_duration is the actual historical duration from the transport dataset.

Step 2: Expected Delay E[D] by Transport Mode. Flight — Robust Delay Formula. The expected delay for flights accounts for highvariance delays and worst-case disruptions:

$$
E [ D ] _ { \mathrm { { f l i g h t } } } = { \frac { n _ { \mathrm { { l a t e } } } } { N } } \left( \mu + \sigma \right) + ~ { \frac { n _ { \mathrm { { d i s } } } } { N } } D _ { \mathrm { { m a x } } }\tag{23}
$$

where the symbols are defined in Table 14.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $N$ </td><td>total_flights</td></tr><tr><td>Nlate</td><td>late + very_late + excessive</td></tr><tr><td>Ndis</td><td>cancelled + diverted</td></tr><tr><td> $\mu$ </td><td>avg_delay_min</td></tr><tr><td>σ</td><td>std_delay_min</td></tr><tr><td> $D _ { \mathrm { m a x } }$ </td><td>max_delay_min</td></tr></table>

Table 14: Symbols used in the flight robust delay formula.

The term $\mu + \sigma$ captures high-variance delays at approximately the 84th percentile under a normal distribution. The term $\frac { n _ { \mathrm { d i s } } } { N } \times D _ { \mathrm { m a x } }$ penalises worstcase disruptions proportionally.

Train — Weighted Historical Delay Mean. A weighted average across multiple time horizons captures structural delay patterns, with longer horizons receiving greater weight:

$$
E [ D ] _ { \mathrm { t r a i n } } = \frac { d _ { 1 w } + 2 d _ { 1 m } + 3 d _ { 3 m } + 4 d _ { 6 m } + 6 d _ { 1 y } } { 1 6 }\tag{24}
$$

where $d _ { h }$ is the average delay over horizon h (1 week, 1 month, 3 months, 6 months, 1 year). Recent short horizons are weighted less; longer horizons reflect structural delay patterns in the rail network.

## Cab / Bus — Empirical Road Delay.

$$
E [ D ] _ { \mathrm { r o a d } } = \mathrm { E s t . \ h i s t o r i c a l { \ d e l a y } f r o m { \ r o a d } { d a t a s e t } }\tag{26}
$$

Empirical road-delay distributions are sourced from the TomTom API.

Step 3: Risk-Profile Aware Expected Buffer Range. Each traveller risk profile maps to a safe buffer multiplier range (Table 15):
<table><tr><td>Traveller Type</td><td>low_mult</td><td>high_mult</td></tr><tr><td>Risk Tolerant</td><td>0.50</td><td>1.00</td></tr><tr><td>Risk Optimized</td><td>1.00</td><td>1.25</td></tr><tr><td>Risk Averse</td><td>1.25</td><td>2.00</td></tr></table>

Table 15: Risk-profile buffer multipliers for TDAS.

The expected safe buffer range for segment $j$ is:

$$
a _ { c } = l o w \_ m u l t \times E [ D ]\tag{27}
$$

$$
b _ { c } = h i g h \_ m u l t \times E [ \boldsymbol { D } ]\tag{28}
$$

$$
\begin{array} { r } { Q _ { j } = \left\{ \begin{array} { l l } { 0 , } & { B _ { j } \in [ a _ { c } , b _ { c } ] } \\ { \displaystyle \frac { \operatorname* { m i n } ( | B _ { j } - a _ { c } | , | B _ { j } - b _ { c } | ) } { b _ { c } - a _ { c } } , } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{29}
$$

Step 4: Aggregate TDAS. For an itinerary with M transport segments:

$$
\boxed { \mathrm { T D A S } = 1 - \frac { 1 } { M } \sum _ { j = 1 } ^ { M } m i n ( 1 , Q _ { j } ) }\tag{30}
$$

Interpretation.

• TDAS ≈ 1: Delays are highly absorbable; the itinerary remains sTable

• TDAS ≈ 0: High disruption risk due to insufficient buffer allocation.

Table 16: Data entries in the UTP-Bench dataset collected from real-world sources.
<table><tr><td>Database</td><td>Data Entries (#)</td></tr><tr><td>City Set</td><td>20</td></tr><tr><td>Trains</td><td>35,117</td></tr><tr><td>Buses</td><td>11,313</td></tr><tr><td>Flights</td><td>1,198</td></tr><tr><td>Restaurants</td><td>244</td></tr><tr><td>Attractions</td><td>242</td></tr><tr><td>Accommodations</td><td>325</td></tr><tr><td>Nearest Transit Stop</td><td>861</td></tr><tr><td>Road Delay Routes</td><td>380</td></tr></table>

## C Prompt and Annotation Details

## C.1 Query Generation using GPT-5

You are an expert at generating natural-language travel planning queries, Given an input Travel {JSON}, generate a concise and natural English   
query describing the trip. In the JSON, ‘org’ denotes the departure city. When ‘days’ exceeds 3, ‘visiting\_city\_number’ specifies the number of   
cities to be covered in the destination state (use the ‘target state’ in the query). The budget is in INR. Please disregard the level’ attribute. Here   
are four examples.   
\*\*\*\*\*\* Example \*\*\*\*\*\*   
–EXAMPLE 1—–   
JSON:   
{"org": "Jaipur", "dest": "Nagpur", "days": 3, "visiting\_city\_number": 1, "date": ["2025-11-02",   
"2025-11-03", "2025-11-04"], "people\_number": 1, "local\_constraint": {"house\_rules": null, "house\_type": null,   
"facilities\_amenities": null, "services": null, "cuisine": null, "transportation": null, "event": null, "attraction":   
null}, "budget": 18673, "query": null, "level": "easy"}   
{generated\_query\_english:"Plan a 3-day trip for 1 person from Jaipur to Nagpur from November 2nd to November 4th,   
2025, with a budget of INR 18673."}   
–EXAMPLE 2—   
JSON:   
{"org": "Jabalpur", "dest": ["Thiruvananthapuram", "Kochi"], "target\_state": "Kerala", "days": 5,   
"visiting\_city\_number": 2, "date": ["2025-11-14", "2025-11-15", "2025-11-16", "2025-11-17", "2025-11-18"],   
"people\_number": 4, "local\_constraint": {"house\_rules": null, "house\_type": null, "facilities\_amenities":   
"Dining\_Drinks", "services": null, "cuisine": null, "transportation": null, "event": null, "attraction": null},   
"budget": 57671, "level": "medium"}   
{generated\_query\_english:"Organize a 5-day itinerary for 4 people traveling from Jabalpur to explore 2 cities in   
Kerala, between November 14th and November 18th, 2025. The budget is INR 57671, and accommodations should include   
Dining and Drinks."}   
–EXAMPLE 3—–   
JSON:   
{"org": "Jaipur", "dest": ["Jamnagar", "Vadodara", "Ahmedabad"], "target\_state": "Gujarat", "days": 7,   
"visiting\_city\_number": 3, "date": ["2025-11-08", "2025-11-09", "2025-11-10", "2025-11-11", "2025-11-12",   
"2025-11-13", "2025-11-14"], "people\_number": 2, "local\_constraint": {"house\_rules": "No\_Pets", "house\_type":   
"Luxury\_Boutique", "facilities\_amenities": null, "services": "Entertainment", "cuisine": null, "transportation":   
null, "event": null, "attraction": null}, "budget": 65466, "level": "hard"}   
{generated\_query\_english:"Create a detailed 7-day travel plan for 2 individuals starting from Jaipur and visiting 3   
cities in Gujarat between November 8th and November 14th, 2025. The budget is INR 65466. Accommodations should be   
Luxury Boutique and strictly No Pets. The plan should also include Entertainment services."}   
\*\*\*\*\*\* Example Ends \*\*\*\*\*\*   
Input:   
Travel JSON: {JSON}   
Output Requirements:   
• Output must be valid JSON.   
• The response must start with the key "generated\_query\_english".   
Output format:   
[{"generated\_query\_english": "<generated English query>"}]

## C.2 Prompt to Generate Uncertainty Plans

You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., 6E-789), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense.

## CRITICAL INSTRUCTION: INCORPORATE UNCERTAINTY & BUFFER TIMES

You must account for uncertainty in your planning by incorporating buffer times based on the historical data provided in the context:

1. Attractions & Restaurants: Use the provided besttime\_analysis to gauge crowd levels. If visiting during ‘High’ or ‘Above average’ intensity hours, add a 15–30 minute buffer to the visit duration. Try to avoid peak\_hours.

2. Road Transportation (which include bus and Cab Transportation): In addition to Total Travel Time, add the Estimated Historical Delay to the travel duration to ensure realistic arrival times.

3. Train Transportation: Check avg\_delay\_1month (or similar available delay metric) and add this expected delay to the train’s arrival time before scheduling the next activity.

4. Flight Transportation: Review OnTimePerformance and DelayStats. If the flight has a low on-time rating or high delay probability, add a 30–45 minute buffer to the arrival time before the next scheduled item.

The symbol ‘-’ indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.

IMPORTANT: YOUR OUTPUT MUST BE IN JSONL FORMAT ONLY. Do not include any explanations, markdown formatting, or text outside the JSON structure.

## === COMPRESSED JSON DATA FORMAT GUIDE ===

You are given venue crowd and timing data in a COMPRESSED JSON format to reduce context size.

## IMPORTANT:

• This compressed format is lossless.

• Do NOT assume missing information.

• All times are in 24-hour format.

• Interpret numeric ranges inclusively (e.g., [11,16] = 11:00–16:59).

GLOBAL FIELDS (apply to the venue / average day):

• "o": venue opening hour

• "c": venue closing hour

• "s": surge behavior

– "in": hour when most visitors arrive – "out": hour when most visitors leave

• "I": intensity dictionary (code → meaning)

## AVERAGE DAY OBJECT FORMAT:

The key "avg\_day" represents aggregated crowd behavior across days. Keys inside "avg\_day":

• "cw": crowd windows, each as [start\_hour, end\_hour, intensity\_code]

• "q": quiet hours (lowest crowd)

• "p": peak window as [start, end] (if present)

## INTENSITY CODES:

• BA → Below average crowd

• A → Average crowd

• AA → Above average crowd

• H → High crowd

• L → Low crowd

## HOW TO INTERPRET:

• Use "cw" to reason about time-dependent crowd levels.

• Prefer "q" hours for low waiting time and higher feasibility.

• Treat "p" as the busiest continuous window.   
• Use surge times to anticipate entry and exit congestion.   
• Do NOT expand or rewrite the data unless explicitly asked.   
GOAL: Use this data to perform crowd-aware reasoning, scheduling, uncertainty modeling, or feasibility checks without adding new assumptions.   
=== END OF COMPRESSED JSON DATA FORMAT GUIDE ===   
\*\*\*\*\*\* Example \*\*\*\*\*\*   
Query: Plan a 7-day trip for 2 people starting from Bhopal to visit 3 cities—Kanpur, Surat, and Pune—from November 6th to November 12th,   
2025, with a budget of INR 91306. Accommodations should feature a Wellness Spa, provide Cleaning and Laundry services, and offer African   
cuisine.   
Traveler Persona:   
Traveler Type: Culture Enthusiast; Purpose of Travel: Cultural Immersion; Spending Preference: Luxury; Location Preference: Historical Sites   
Travel Plan:   
[   
{   
"days": 1,   
"current\_city": "from Bhopal to Kanpur",   
"transportation": "Train Number: 12191, from Bhopal to Kanpur,   
Departure Time: 06:30, Arrival Time: 15:15",   
"breakfast": ”\_"   
"attraction": "Cawnpore Club, Kanpur",   
"lunch": "-",   
"dinner": "Haveli Restaurant, Kanpur",   
"accommodation": "Hotel Bhagyaraj Palace, Kanpur",   
"event": "-",   
"point\_of\_interest": "Hotel Bhagyaraj Palace, check-in from 15:45   
to 16:15, nearest transit: Kanpur Central Station, 3km away;   
Cawnpore Club, visit from 16:30 to 18:30, nearest transit:   
Civil Lines, 0.5km away; Haveli Restaurant, visit from 20:45   
to 22:00, nearest transit: Mall Road Stop, 1km away; Hotel   
Bhagyaraj Palace, stay from 22:15 to 08:00, nearest transit:   
Kanpur Central Station, 3km away."   
},   
"days": 2,   
"current\_city": "Kanpur",   
"transportation": "-"   
"breakfast": "Little Chef Restaurant, Kanpur",   
"attraction": "Brahmavart Ghat, Kanpur",   
"lunch": "Haveli Restaurant, Kanpur",   
"dinner": "Dadi ki rasoi, Kanpur",   
"accommodation": "Hotel Bhagyaraj Palace, Kanpur",   
"event": "-",   
"point\_of\_interest": "Hotel Bhagyaraj Palace, stay from 08:00 to   
09:40, nearest transit: Kanpur Central Station, 3km away;   
Little Chef Restaurant, visit from 09:40 to 10:30, nearest   
transit: Mall Road Stop, 1km away; Brahmavart Ghat, visit   
from 11:00 to 13:30, nearest transit: Bithoor Ghat, 0.2km   
away; Haveli Restaurant, visit from 14:20 to 15:20, nearest   
transit: Mall Road Stop, 1km away; Dadi ki rasoi, visit from   
20:45 to 22:00, nearest transit: Bithoor Road Stop, 1.5km   
away; Hotel Bhagyaraj Palace, stay from 22:15 to 06:00,   
nearest transit: Kanpur Central Station, 3km away."   
},   
{   
"days": 3,   
"current\_city": "from Kanpur to Surat",   
"transportation": "Flight Number: 6E-2145, from Kanpur to Surat,   
Departure Time: 08:00, Arrival Time: 10:30",   
"breakfast": "-",   
"attraction": "Dutch Garden, Surat",   
"lunch": "Mysore Cafe, Surat",   
"dinner": "Levvel 5 Terrace Restro & Cafe, Surat",   
"accommodation": "Surat Marriott Hotel, Surat",   
"event": "-",   
"point\_of\_interest": "Hotel Bhagyaraj Palace, checkout from 06:00   
to 06:30, nearest transit: Kanpur Central Station, 3km away;   
Kanpur Airport, arrive from 07:00 to 07:30, nearest transit:   
Airport Road, 2km away; Surat Marriott Hotel, check-in from   
11:00 to 11:30, nearest transit: Surat Railway Station, 4km   
away; Dutch Garden, visit from 12:00 to 13:30, nearest   
transit: Athwalines, 1km away; Mysore Cafe, visit from 14:20   
to 15:20, nearest transit: Ring Road Stop, 1km away; Levvel   
5 Terrace Restro & Cafe, visit from 20:45 to 22:00, nearest   
transit: VIP Road Stop, 1km away; Surat Marriott Hotel, stay   
from 22:15 to 08:00, nearest transit: Surat Railway Station,   
4km away."   
},

"days": 4,   
"current\_city": "Surat",   
"transportation": "-"   
"breakfast": "Taste Of India Restaurant, Surat",   
"attraction": "ISKCON Temple, Surat",   
"lunch": "Levvel 5 Terrace Restro & Cafe, Surat",   
"dinner": "Mysore Cafe, Surat",   
"accommodation": "Surat Marriott Hotel, Surat",   
"event": "-",   
"point\_of\_interest": "Surat Marriott Hotel, stay from 08:00 to   
09:40, nearest transit: Surat Railway Station, 4km away;   
Taste Of India Restaurant, visit from 09:40 to 10:30, nearest   
transit: Varachha Road, 1km away; ISKCON Temple, visit from   
11:00 to 13:30, nearest transit: Athwalines, 0.5km away;   
Levvel 5 Terrace Restro & Cafe, visit from 14:20 to 15:20,   
nearest transit: VIP Road Stop, 1km away; Mysore Cafe, visit   
from 20:45 to 22:00, nearest transit: Ring Road Stop, 1km   
away; Surat Marriott Hotel, stay from 22:15 to 07:00, nearest   
transit: Surat Railway Station, 4km away."   
},   
"days": 5,   
"current\_city": "from Surat to Pune",   
"transportation": "Flight Number: 6E-3456, from Surat to Pune,   
Departure Time: 09:00, Arrival Time: 10:30",   
"breakfast": "-",   
"attraction": "Mahadaji Shinde Chatri, Pune",   
"lunch": "Spice Kitchen, Pune",   
"dinner": "Sante Spa Cuisine, Pune",   
"accommodation": "Conrad Pune, Pune",   
"event": "-",   
"point\_of\_interest": "Surat Marriott Hotel, checkout from 07:00   
to 07:30, nearest transit: Surat Railway Station, 4km away;   
Surat Airport, arrive from 08:00 to 08:30, nearest transit:   
Airport Road, 3km away; Conrad Pune, check-in from 11:00 to   
11:30, nearest transit: Pune Junction, 5km away; Mahadaji   
Shinde Chatri, visit from 12:00 to 13:30, nearest transit:   
Wanowrie, 1km away; Spice Kitchen, visit from 14:20 to 15:20,   
nearest transit: Koregaon Park, 0.5km away; Sante Spa Cuisine,   
visit from 20:45 to 22:00, nearest transit: Koregaon Park,   
0.3km away; Conrad Pune, stay from 22:15 to 08:00, nearest   
transit: Pune Junction, 5km away."   
"days": 6,   
"current\_city": "Pune",   
"transportation": "-",   
"breakfast": "Spice Kitchen, Pune",   
"attraction": "ISKCON NVCC Temple, Pune",   
"lunch": "Sante Spa Cuisine, Pune",   
"dinner": "The Flour Works, Pune",   
"accommodation": "Conrad Pune, Pune",   
"event": "-",   
"point\_of\_interest": "Conrad Pune, stay from 08:00 to 09:40,   
nearest transit: Pune Junction, 5km away; Spice Kitchen,   
visit from 09:40 to 10:30, nearest transit: Koregaon Park,   
0.5km away; ISKCON NVCC Temple, visit from 11:00 to 13:30,   
visit from 14:20 to 15:20, nearest transit: Koregaon Park,   
0.3km away; The Flour Works, visit from 20:45 to 22:00,   
nearest transit: Kalyani Nagar, 1km away; Conrad Pune, stay   
from 22:15 to 07:00, nearest transit: Pune Junction, 5km   
away."   
},   
"days": 7,   
"current\_city": "from Pune to Bhopal",   
"transportation": "Flight Number: 6E-5432, from Pune to Bhopal,   
Departure Time: 11:00. Arrival Time: 12:30"   
"breakfast": "Kumar Pacific Mall, Pune",   
"attraction": "-",   
"lunch": "-",   
"dinner": "-",   
"accommodation": "-",   
"event": "-",   
"point\_of\_interest": "Conrad Pune, checkout from 07:00 to 07:30,   
nearest transit: Pune Junction, 5km away; Kumar Pacific Mall,   
visit from 07:45 to 08:45, nearest transit: Swargate, 1km   
away; Pune Airport, arrive from 09:30 to 10:30, nearest   
transit: Airport Road, 2km away; Flight 6E-5432, travel from   
11:00 to 12:30."   
\*\*\*\*\*\* Example Ends \*\*\*\*\*\*   
Given information: {enriched\_ref\_info}

## C.3 Prompt to fill the realistic timings

You are a proficient travel plan time allocator. You do NOT design new itineraries from scratch.

Your ONLY job is to take an ALREADY GENERATED TRAVEL PLAN (with days, cities, attractions, restaurants, and accommodations already chosen) that is missing some or all timing information, and FILL IN REALISTIC, CONSISTENT TIMINGS for every activity, stay, visit, and transport segment.

You must:

• Preserve the existing structure of the plan (days, current\_city, transportation, breakfast, attraction, lunch, dinner, accommodation, event, and the text segments inside point\_of\_interest / point\_of\_interest\_list).

• Preserve the exact list and order of cities, accommodations, attractions, restaurants, events, and transit segments.

• ONLY add or adjust times and clearly stated buffers; do NOT add new places or remove existing ones.

• Ensure that the final plan is temporally consistent (no overlaps, no negative gaps, realistic travel + activity durations) and respects uncertainty and buffer rules.

## CRITICAL INPUT FORMAT

You receive each sample as a JSON object with the following structure (in JSONL format, one object per line). The field JSON contains information like: origin city, destination city (or list of cities), dates array (one date per day), number of days, number of people, traveler type/persona, local constraints, budget, and the natural language query. The field plan is an array of per-day obiects where transportation, meals, attraction, accommodation, and event are l d h d MUST NOT b h d h point\_of\_interest point\_of\_interest\_list i i l d d d li f describing stays, check-in/checkout windows, visits, and travel segments. In many inputs, some time windows are missing and represented as empty parentheses: ().

Your task is to REPLACE ALL SUCH EMPTY PARENTHESIS TIME SLOTS WITH CONCRETE 24-HOUR TIMES (HH:MM), while keeping the rest of the text unchanged.

IMPORTANT: You may slightly adjust already filled times (e.g., shift by 10–30 minutes) only if necessary to maintain chronological consistency, to account for uncertainty buffers, or to avoid overlaps. Otherwise, keep existing times as-is.

## UNCERTAINTY & BUFFER RULES (MUST FOLLOW)

1. Attractions & Restaurants (crowd-aware): When besttime\_analysis or similar crowd data is available, use its crowd intensity windows. If an attraction/restaurant is visited during a “High” or “Above average” intensity window, add a 15–30 minute buffer to the base visit duration. Prefer visits during quieter windows when possible, but do NOT change the chosen venue or day. Avoid peak\_hours where explicitly marked, or if unavoidable, add an appropriate wait-time buffer.

2. Road Transportation (buses, cabs, cars): When total travel time and Estimated Historical Delay are provided, the scheduled travel window must be: base\_travel\_duration + estimated\_delay.

3. Train Transportation: Use available delay metrics such as avg\_delay\_1month to extend the arrival time. The planned arrival time must be: scheduled\_arrival + expected\_delay. Ensure subsequent activities start AFTER this buffered arrival time.

4. Flight Transportation: Use OnTimePerformance and DelayStats where provided. If the flight has a poor on-time record or high delay probability, add a 30–45 minute buffer after the scheduled arrival before starting the next activity.

## DAY STRUCTURE & MEAL-TIME HEURISTICS

When filling times, follow these heuristics unless they clearly conflict with already-fixed times or explicit constraints:

• Each day should start and end at the accommodation (except pure travel days with final return to origin).

• Morning wake-up / stay at accommodation: around 07:00–09:40.

• Breakfast: start around 09:40, duration ∼50 minutes.

• Late-morning attractions: roughly 10:30–13:30.

• Lunch: start around 14:20, duration ∼60 minutes.

• Afternoon attractions/activities: roughly 15:00–18:30.

• Evening free time / events: roughly 18:30–20:30.

• Dinner: start around 20:45, duration ∼75 minutes.

• Night stay at accommodation: from around 22:15 until next morning 06:00–08:00.

## TRAVELER PERSONA & ATTRACTION DENSITY

You must respect the intended intensity of sightseeing implied by the traveler type:

• Risk Tolerant: Strict timings, shorter buffers, close to or below ideal buffer duration. Typically 2–3 attractions per day, sometimes more.

• Risk Optimized: Balanced schedule, moderate buffer equal to or slightly more than ideal buffer duration.

• Risk Averse: Significantly larger buffers,may go upto twice or well beyond ideal buffer duration; may prefer earlier returns to accommodation.

Do NOT change which attractions are visited. Use these rules only to set visit durations and buffers.

TEMPORAL CONSISTENCY RULES

• All segments in point of interest for each day must be chronological (each start time ≥ previous end time, with realistic small transfer gaps).

• Transport windows must be long enough to cover base travel time + uncertainty buffers.

• Check-in/check-out windows must surround realistic arrival/departure times.

• Never schedule an attraction or restaurant before the traveler physically arrives in that city.

• The final stay at accommodation must reach at least 06:00–08:00 the next morning, except for overnight transport.

## OUTPUT REQUIREMENTS (VERY IMPORTANT)

• Output MUST be in valid JSONL format ONLY each line is one JSON object with the same schema as the input.

• For each input object, output exactly one JSON object with the same id and JSON fields.

• Preserve all non-time textual content (city names, attraction names, restaurant names, accommodation names, nearest transit info, delay descriptions, etc.).

• Replace every occurrence of () with concrete 24-hour times in HH:MM format.

• Ensure every segment ends up with a well-defined start and end time, or explicit Departure Time and Arrival Time fields filled.

• Do NOT output any explanations, comments, or markdown formatting outside of the JSON objects.

## Given information: {enriched\_ref\_info}

Input (JSONL with existing plan and missing times): {input\_plan\_with\_missing\_times}, reference\_information

Output (JSONL with same structure and all times filled)

## C.4 Prompt to Generate Without-Uncertainty Plans

You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., 6E-789), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense.

The symbol ‘-’ indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying.

## \*\*\*\*\*\* Example \*\*\*\*\*\*

Query: Plan a 7-day trip for 2 people starting from Bhopal to visit 3 cities—Kanpur, Surat, and Pune—from November 6th to November 12th, 2025, with a budget of INR 91306. Accommodations should feature a Wellness Spa, provide Cleaning and Laundry services, and offer African cuisine.

## Traveler Persona:

Traveler Type: Culture Enthusiast; Purpose of Travel: Cultural Immersion; Spending Preference: Luxury; Location Preference: Historical Sites

```csv
Travel Plan:
"plan": [
{
"day": 1,
"current_city": "from Bhopal to Kanpur",
"transportation": "Train Number: 12191, from Bhopal to Kanpur,
Departure Time: 06:30, Arrival Time: 15:15",
"breakfast": "-",
"attraction": "Cawnpore Club, Kanpur",
"lunch": "_"
"dinner": "Haveli Restaurant, Kanpur",
"accommodation": "Hotel Bhagyaraj Palace, Kanpur",
"event": "-",
"point_of_interest_list": "Hotel Bhagyaraj Palace, check-in from
15:45 to 16:15, nearest transit: Kanpur Central Station, 11km
away; Cawnpore Club, visit from 16:30 to 18:30, nearest transit:
Civil Lines, 0.5km away; Haveli Restaurant, visit from 20:45 to
22:00, nearest transit: Mall Road Stop, 1km away; Hotel Bhagyaraj
Palace, stay from 22:15 to 08:00, nearest transit: Kanpur Central
Station, 3km away."
},
{
"day": 2,
"current_city": "Kanpur",
"transportation": "-",
"breakfast": "Little Chef Restaurant, Kanpur",
"attraction": "Brahmavart Ghat, Kanpur",
"lunch": "Haveli Restaurant, Kanpur",
"dinner": "Dadi ki rasoi, Kanpur",
"accommodation": "Hotel Bhagyaraj Palace, Kanpur",
"event": "-",
"point_of_interest_list": "Hotel Bhagyaraj Palace, stay from 08:00
to 09:40, nearest transit: Kanpur Central Station, 3km away;
Little Chef Restaurant, visit from 09:40 to 10:30, nearest
transit: Mall Road Stop, 1km away; Brahmavart Ghat, visit from
11:00 to 13:30, nearest transit: Bithoor Ghat, 0.2km away;
Haveli Restaurant, visit from 14:20 to 15:20, nearest transit:
Mall Road Stop, 1km away; Dadi ki rasoi, visit from 20:45 to
22:00, nearest transit: Bithoor Road Stop, 6km away; Hotel
Bhagyaraj Palace, stay from 22:15 to 06:00, nearest transit:
Kanpur Central Station, 3km away."
},
{
"day": 3,
"current_city": "from Kanpur to Surat",
"transportation": "Flight Number: 6E-2145, from Kanpur to Surat,
Departure Time: 08:00, Arrival Time: 10:30",
"breakfast": "-",
"attraction": "Dutch Garden, Surat",
"lunch": "Mysore Cafe, Surat",
"dinner": "Levvel 5 Terrace Restro & Cafe, Surat",
"accommodation": "Surat Marriott Hotel, Surat",
"event": "-",
"point_of_interest_list": "Hotel Bhagyaraj Palace, checkout from
06:00 to 06:30, nearest transit: Kanpur Central Station, 3km
away; Kanpur Airport, arrive from 07:00 to 07:30, nearest
transit: Airport Road, 2km away; Surat Marriott Hotel, check-in
from 11:00 to 11:30, nearest transit: Surat Railway Station,
4km away; Dutch Garden, visit from 12:00 to 13:30, nearest
transit: Athwalines, 1km away; Mysore Cafe, visit from 14:20
to 15:20, nearest transit: Ring Road Stop, 1km away; Levvel 5
Terrace Restro & Cafe, visit from 20:45 to 22:00, nearest
transit: VIP Road Stop, 1km away; Surat Marriott Hotel, stay
from 22:15 to 08:00, nearest transit: Surat Railway Station,
4km away."
```

```csv
},
"day": 4,
"current_city": "Surat",
"transportation": "-",
"breakfast": "Taste Of India Restaurant, Surat",
"attraction": "ISKCON Temple, Surat",
"lunch": "Levvel 5 Terrace Restro & Cafe, Surat",
"dinner": "Mysore Cafe, Surat",
"accommodation": "Surat Marriott Hotel, Surat",
"event": "-",
"point_of_interest_list": "Surat Marriott Hotel, stay from 08:00
to 09:40, nearest transit: Surat Railway Station, 4km away;
Taste 0f India Restaurant visit from 09:40 to 10:30 nearest
transit: Varachha Road, 1km away; ISKCON Temple, visit from
11:00 to 13:30, nearest transit: Athwalines, 0.5km away;
Levvel 5 Terrace Restro & Cafe, visit from 14:20 to 15:20,
nearest transit: VIP Road Stop, 1km away; Mysore Cafe, visit
from 20:45 to 22:00, nearest transit: Ring Road Stop, 1km
away; Surat Marriott Hotel, stay from 22:15 to 07:00, nearest
transit: Surat Railway Station, 4km away."
},
"day": 5,
"current city": "from Surat to Pune"
"transportation": "Flight Number: 6E-3456, from Surat to Pune,
Departure Time: 09:00, Arrival Time: 10:30",
"breakfast": "-",
"attraction": "Mahadaji Shinde Chatri, Pune",
"lunch": "Spice Kitchen, Pune",
"dinner": "Sante Spa Cuisine, Pune",
"accommodation": "Conrad Pune, Pune",
"event": "-",
"point_of_interest_list": "Surat Marriott Hotel, checkout from
07:00 to 07:30, nearest transit: Surat Railway Station, 4km
away; Surat Airport, arrive from 08:00 to 08:30, nearest
transit: Airport Road, 3km away; Conrad Pune, check-in from
11:00 to 11:30, nearest transit: Pune Junction, 5km away;
Mahadaji Shinde Chatri, visit from 12:00 to 13:30, nearest
transit: Wanowrie, 1km away; Spice Kitchen, visit from 14:20
to 15:20, nearest transit: Koregaon Park, 0.5km away; Sante
Spa Cuisine, visit from 20:45 to 22:00, nearest transit:
Koregaon Park, 0.3km away; Conrad Pune, stay from 22:15 to
08:00, nearest transit: Pune Junction, 5km away."
},
{
"day": 6,
"current_city": "Pune",
"transportation": "-",
"breakfast": "Spice Kitchen, Pune",
"attraction": "ISKCON NVCC Temple, Pune",
"lunch": "Sante Spa Cuisine, Pune",
"dinner": "The Flour Works, Pune",
"accommodation": "Conrad Pune, Pune",
"event": "-",
"point_of_interest_list": "Conrad Pune, stay from 08:00 to 09:40,
nearest transit: Pune Junction, 5km away; Spice Kitchen, visit
from 09:40 to 10:30, nearest transit: Koregaon Park, 0.5km
away; ISKCON NVCC Temple, visit from 11:00 to 13:30, nearest
transit: NIBM Road, 1km away; Sante Spa Cuisine, visit from
14:20 to 15:20, nearest transit: Koregaon Park, 0.3km away;
The Flour Works, visit from 20:45 to 22:00, nearest transit:
Kalyani Nagar, 1km away; Conrad Pune, stay from 22:15 to
07:00, nearest transit: Pune Junction, 5km away."
},
"day": 7,
"current_city": "from Pune to Bhopal",
"transportation": "Flight Number: 6E-5432, from Pune to Bhopal,
Departure Time: 11:00, Arrival Time: 12:30",
"breakfast": "Kumar Pacific Mall, Pune",
"attraction": "-",
"lunch": "-",
"dinner": "-",
"accommodation": "-",
"event": "-",
"point_of_interest_list": "Conrad Pune, checkout from 07:00 to
07:30, nearest transit: Pune Junction, 5km away; Kumar Pacific
Mall, visit from 07:45 to 08:45, nearest transit: Swargate,
1km away; Pune Airport, arrive from 09:30 to 10:30, nearest
transit: Airport Road, 2km away; Flight 6E-5432, travel from
11:00 to 12:30."
****** Example Ends ******
```

Given information: {enriched\_ref\_info}   
Query: {query}   
Traveler Persona:   
{persona}   
Output:

## D Case Studies

![](images/255a98218bbe7b88cba8fc40fe272c6c385734f8faabdf0b9010e77c1d6f9d4b.jpg)  
Figure 4: Case: Good BAS — Risk Averse profile with well-calibrated POI buffers.

![](images/826cf16a63e560361dc46fda5d043ce006c545f1e437760929ffe43eff306b9a.jpg)  
Figure 5: Case: Good TDAS — delay-aware train buffer within Risk Averse acceptable range.

![](images/c02e95b68a674c6777fd141f079e3b0df6cb56a75233c0f223dd19e2bf677a77.jpg)  
Figure 6: Case: Incosistency in filling timings.

![](images/0d6fff59f263417ba1b06d33a07c5374729449ad57e27ee77e13daf2ffcf89d9.jpg)  
Figure 7: Case: Missing buffer times and crowd information.

![](images/10de1b2cda99ae7d0703b53bfc909cdcd52926e528af4ed25a18701b72436bcb.jpg)  
Figure 8: Case: Excessive crowd buffer inflating visit duration.

![](images/0530fb50f3f84a73bf9fa65e12db6f8f44bb423a03fe2b0b7cdb805bef9585a6.jpg)  
Figure 9: Case: Temporal overlap between consecutive activities.

Geographic Distribution across India

## E Annotator Details

All annotators were students affiliated with our research lab and voluntarily participated in the annotation process as part of their regular research activities. Therefore, no separate monetary compensation was provided specifically for the annotation task.

## E.1 Guidelines for Annotators

The annotation process involves providing annotators with a structured travel plan skeleton containing pre-selected cities, accommodations, attractions, and restaurants rather than a fully detailed itinerary. Annotators are tasked with filling in realistic and optimal timings for each activity based on their own real travel experiences and practical knowledge. The annotated plan must reflect feasible scheduling while considering reference information and local preferences such as cuisine type, attraction category, and traveler personas (e.g., laidback, economical). Additionally, common sense should be applied when assigning visit durations and transit gaps, and any deviations from suggested durations or costs must be justified. A detailed breakdown of these annotation guidelines, including priority handling, public transit considerations, and documentation requirements, is provided in Table 17.

Age Distribution  
![](images/f25692d4aa3e19e024ce41ad4eacc82162036820b24d08f7e48d56df432c4b74.jpg)  
Figure 10: Age distribution of our graduate student annotators.

## E.2 Annotator Demographics

Total 14 trained human annotators participated in this study. The annotator demographics, as illustrated by the figures, show a diverse range of backgrounds. The age distribution (Figure 10) shows all annotators fall between 22–25 years old, with the majority aged 23 (35.7%), followed by age 22 (28.6%), age 24 (21.4%), and age 25 (14.3%), indicating a young and energetic graduate student cohort. The gender distribution reflects balanced participation, with 57.1% male, 35.7% female, and 7.1% other annotators. The years of English education (Figure 11) show that most annotators have between 13–18 years of formal English instruction, suggesting strong language proficiency. We specifically recruited annotators from India only as UTP-Bench is India-centric dataset. The geographic distribution (Figure 12) also highlights that annotators were recruited from diverse regions of India — North (42.9%), East (21.4%), South (21.4%), and West (14.3%) — ensuring a wide variety of real-world travel experiences and regional familiarity, contributing to reliable and contextually aware annotations.

![](images/f94bce362bc25e554b919c41b5180a17ed5a48ab8c60a9cb423d3a456d40a0c2.jpg)  
Figure 11: Years of formal English education statistics of our graduate student annotators.

![](images/2de768c4862cd46b3b30f21f5d8ddc8f864014259f83c4c84f587ba0cb1204fe.jpg)  
Figure 12: Geographic distribution of annotators from diverse regions of India.

Table 17: Guidelines for Annotation of Travel Plans and Remarks.
<table><tr><td>#</td><td>Annotation Guideline</td></tr><tr><td colspan="2">Route &amp; Structure</td></tr><tr><td>1</td><td>The trip must follow a closed-loop route: the first and last city must be the origin city (org). The city sequence must be logical with no unnecessary back-and-forth.</td></tr><tr><td>2</td><td>Do not repeat restaurants, attractions, or events across the entire trip. Each restaurant, attraction, and event may appear only once in the full itinerary. The Point of Interest (PoI) list for each day must include all activities: accommodation (check-</td></tr><tr><td>3</td><td>in/stay), breakfast, lunch, dinner, attractions, and events, each with a valid timeline and transit info.</td></tr><tr><td colspan="2">Transportation &amp; Accommodation Transportation must be consistent. Day 1 must include valid transport from the origin. Do</td></tr><tr><td>4</td><td>not mix conflicting modes (e.g., cab/taxi and flight). Mode must match local_constraint if specified. For multi-night stays in the same city, use the same accommodation. Accommodations must</td></tr><tr><td>5</td><td>be valid, verifiable places in the current city. At least 4 hours must separate the end of one meal from the start of the next (Breakfast →</td></tr><tr><td>6</td><td>Lunch, Lunch → Dinner). The PoI timeline must reflect these gaps.</td></tr><tr><td colspan="2">Hard Constraints</td></tr><tr><td>7</td><td>Hard constraints must be satisfied: total trip cost must not exceed the budget; cuisine, house rules, room type, facilities, and services specified in local_constraint must all be respected.</td></tr><tr><td colspan="2">Uncertainty &amp; Buffer Times</td></tr><tr><td>8</td><td>Use the provided Python scripts to compute mean buffer times for transportation and attrac- tions/restaurants. These are reference values only; annotators may adjust based on trip context and traveler type.</td></tr><tr><td>9</td><td>Buffer time should be scaled by traveler risk type — Risk Tolerant: (0.5–1.0) × ideal buffer; Risk Optimized: (1.0–1.25) × ideal buffer; Risk Averse: (1.25–2.0) × ideal buffer. Avoid scheduling attraction or restaurant visits during peak crowd hours where possible.</td></tr><tr><td>10</td><td>Partial overlap with peak slots is acceptable if unavoidable (e.g., a 1:00–3:00 PM visit when peak is 2:00–5:00 PM is permitted).</td></tr><tr><td colspan="2">Documentation &amp; Flagging</td></tr><tr><td>11</td><td>Adjust the timings and Document the rationale for each timing choice in a Remarks column.</td></tr><tr><td>12</td><td>If a repeated attraction is found across days, flag the row and note the repetition in the Remarks</td></tr></table>