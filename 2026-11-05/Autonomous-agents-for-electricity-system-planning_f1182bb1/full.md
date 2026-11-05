# Autonomous agents for electricity-system planning

Yoon Pyo Lee<sup>1</sup>, Samrendra Roy<sup>1</sup>, Jay Phil Yoo<sup>1</sup>, Kazuma Kobayashi<sup>1</sup>, Souvik Chakraborty<sup>1,3,4</sup>, Syed Bahauddin Alam<sup>1,2\*</sup>

<sup>1</sup>Department of Nuclear, Plasma, and Radiological Engineering, The Grainger College of Engineering, University of Illinois Urbana-Champaign, Urbana, IL, USA.

<sup>2</sup>National Center for Supercomputing Applications, University of Illinois Urbana-Champaign, Urbana, IL, USA.

<sup>3</sup>Department of Applied Mechanics, Indian Institute of Technology Delhi, New Delhi, India.

<sup>4</sup>Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi, New Delhi, India.

\*Corresponding author(s). E-mail(s): alams@illinois.edu;

## Abstract

Planning electricity systems requires repeated decisions that connect regional resources, investment, operation and network constraints. Here we present an autonomous planning framework that carries prepared regional scenarios through proposal, numerical evaluation, feedback interpretation, redesign and final plan selection. Agents propose investment bounds, storage operating conditions and feeder interventions, interpret numerical feedback, and decide how to revise designs and when to stop. The framework automatically executes annual investment-and-dispatch optimization and representative-feeder verification, applies fixed acceptance criteria and retains the lowest-cost accepted candidate. We evaluate forty carbon-constrained trajectories across ten system– year scenarios in five electricity systems and four countries, using two model deployments and single- and multi-agent architectures. All forty retain candidates satisfying the implemented physical and fixed annual operational carbon-budget checks. Twenty-eight of forty matched reference designs exceed the new scenario ceilings. A separate archive of ninety-six trajectories under a common 1,000 Mt annual ceiling provides matched references and broader regional coverage. The results demonstrate autonomous planning under explicit emissions constraints using public regional data and published institutional demand outlooks.

## 1 Introduction

Electricity-system planning is becoming more urgent as AI data centres, electric vehicles, industrial electrification and rising cooling demand accelerate electricity use. The International Energy Agency projects global electricity demand to grow by an average of 3.6% annually from 2026 to 2030, with data centres accounting for around half of the increase in the United States[1]. This growth coincides with constrained grid connections, ageing infrastructure and increasing exposure to extreme weather. Large, concentrated data-centre loads link computing operations to local grid reliability and infrastructure costs[2]. Decisions about generation, storage and network expansion therefore shape both the reliability of electricity supply and the cost of long-lived infrastructure.

Capacity-expansion research examines these decisions through coupled investment and operation, including the representation of storage[3]. Spatial studies coordinate renewable deployment with transmission and storage[4], and recent global models jointly optimize investment and hourly operation at fine spatial resolution[5]. Local network studies show that electrical constraints and spatially varying flexibility opportunities also afect deployment[6, 7]. An investment proposal changes how the regional system operates over the year. If the resulting dispatch exceeds a network limit, planners must decide whether to revise the portfolio, change storage operation or reinforce the network. The ability to repeat this process matters across countries with diferent electricity needs and analytical resources. Published work has documented the unequal geographic distribution of expertise in African energy modelling[8]; comprehensive studies such as Net-Zero America illustrate the coordination required across generation and networks[9].

US ofshore wind projections vary with assumptions about technology costs, transmission and policy[10]. The location of new interregional transmission also afects costs and reliability during extreme events[11]. In western Europe, accounting for the efects of heat on demand and grid capacity reveals transmission bottlenecks[12]. Evaluating alternatives across regional systems requires repeated investment-anddispatch optimization, network evaluation and revision of proposed designs. Full-year hourly operation and storage couple decisions across time, while finer temporal detail increases computational demands[13]. Surrogate models trained on numerical simulations can accelerate energy-system optimization[14]. Completing a planning study also requires interpreting numerical outputs, choosing design changes and deciding when to stop. We use pretrained language models to carry out these planning decisions through repeated numerical feedback.

Open models provide important components of this process. PyPSA optimizes investment and operation, PyPSA-Earth supports geographically extensive model construction, and OpenDSS resolves distribution voltages and currents[15–17]. Their integration raises a separate question: can agents autonomously carry a planning study from prepared regional inputs to a verified plan, interpreting physical feedback, revising designs and deciding when to stop?

Language-model agents can use feedback to run power-system simulations[18]. Applications to distribution planning include a workflow with human oversight that evaluates reliability interventions on utility feeders[19] and a multi-agent approach to planning photovoltaic generation and electric vehicle charging within distribution networks[20]. Table 1 compares the decisions addressed in these studies with those examined in recent electricity-system planning research.

Here we develop and evaluate an autonomous electricity-planning framework that connects regional investment and annual operation with representative-feeder verification. To our knowledge, no previous study has demonstrated a planning process across multiple regional electricity systems in which language-model agents revise investment proposals using feedback from both year-round dispatch and distribution network checks. The framework carries the planning process from prepared inputs through feedback-driven redesign and agent-directed stopping to final plan selection across heterogeneous regional and future conditions.

Once a regional scenario is supplied, the framework carries out the planning loop autonomously. Agents interpret regional inputs and numerical feedback to choose technology permissions, investment bounds, storage operating conditions and feeder interventions. PySAM converts regional weather inputs into hourly photovoltaic generation profiles. A turbine power curve supplies the wind profiles. PyPSA uses these profiles to select investments and hourly dispatch within the proposed bounds. OpenDSS then screens the dispatch on representative feeders. The framework automatically executes the required system and feeder evaluations, returns their evidence to the agents and applies fixed acceptance criteria. Agents use this feedback to choose subsequent design changes and decide whether to continue or stop. The framework preserves the lowest-cost accepted candidate found so far in a separate reporting record and returns it as the final result. This adaptation takes place at inference time, without task-specific training or fine-tuning of model weights. Single agents maintain strategy notes, whereas specialist teams additionally maintain an advisory procedure graph inspired by procedural memory[21].

The primary evaluation comprises ten system–year scenarios across NYISO, MISO, South Africa, Mexico and Brazil. These five systems cover four countries and forty model zones. Each scenario is evaluated with GPT-6.1 Sol and Claude Opus 5.5 in both single- and multi-agent configurations, giving forty trajectories. The scenarios comprise NYISO in 2023, 2030 and 2035; MISO, South Africa and Brazil in 2030 and 2035; and Mexico in 2030. Four trajectories use 2023 demand and ERA5 resources, while thirtysix use future demand and resource profiles reconstructed for the matching target year from NASA NEX-GDDP-CMIP6 daily fields and ERA5 intraday templates. Published allowance budgets, planning constraints and emissions pathways define fixed annual operational $\mathrm { C O _ { 2 } }$ ceilings under the mappings in Methods. A separate archive contains ninety-six trajectories across eight systems under the original 1,000 Mt annual ceiling. Forty of those trajectories match the primary cohort by system, year, deployment and architecture, while the remaining fifty-six provide broader regional and temporal coverage.

Table 1 Planning decisions and scope of electricity-system studies and agent-assisted distribution planning.
<table><tr><td>Study</td><td>Planning decision addressed</td><td>Scope demonstrated</td></tr><tr><td colspan="3">Electricity-system studies in Nature Energy</td></tr><tr><td>Zhu et al.[5]</td><td>How much generating capacity is needed to meet global electricity needs under net-zero pathways?</td><td>Joint capacity expansion and hourly operation in a spatially resolved global model.</td></tr><tr><td>Few et al.[6]</td><td>Where can local flexibility reduce network upgrades?</td><td>Network simulations quantify how upgrade costs and the value of flexibility vary across Great Britain.</td></tr><tr><td>Beiter et al.[10]</td><td>How much offshore wind is built under different planning assumptions?</td><td>US expansion scenarios quantify deployment across cost, transmission and policy assumptions.</td></tr><tr><td>Senga et al.[11] Liang et al.[12]</td><td>Where should interregional transmission expand?</td><td>US expansion strategies are compared on cost, emissions and reliability during extreme events. Weather-dependent demand and</td></tr><tr><td></td><td>Where does heat limit grid capacity?</td><td>equipment limits reveal transmission bottlenecks in western Europe.</td></tr><tr><td colspan="3">Agent-assisted distribution planning</td></tr><tr><td>Zhuang and Sadikovic[19]</td><td>Which feeder interventions improve reliability?</td><td rowspan="2">Agent proposals undergo numerical evaluation with human oversight on 100 utility feeders. Agents negotiate stakeholder objectives within a search over</td></tr><tr><td>Shao et al.[20]</td><td>How should a distribution network accommodate solar</td></tr><tr><td>This study</td><td>generation and electric vehicle charging? How can agents carry a regional</td><td>network capacity and topology. Agents revise proposals using</td></tr><tr><td></td><td>scenario through design, evaluation and revision to a verified plan?</td><td>annual dispatch and feeder checks. Across five systems all 40 carbon-constrained searches retain an accepted design.</td></tr></table>

Published expert plans provide a reference for interpreting the carbon-constrained portfolios in Brazil and South Africa in 2035. We compare technology-specific capacities where archived documents permit an explicit year and category mapping. The plans also supply some scenario inputs, including the adopted emissions ceilings, so the comparison describes portfolio diferences under partially shared assumptions.

Across all forty carbon-constrained trajectories the agents completed the planning cycle and the framework retained a design that passed the implemented system, feeder and emissions checks. The separate loose-cap archive shows how agents revised their proposals and when feeder checks ruled out a lower-cost alternative.

## 2 Results

## 2.1 Autonomous agents complete the planning cycle

The framework assigns distinct roles to resource, system and distribution models (Fig. 1). PySAM converts irradiance to photovoltaic output, and a reference turbine curve converts wind speeds to generation potential. PyPSA selects additions and hourly dispatch within agent-specified technology permissions, siting bounds and storage operating envelopes. OpenDSS evaluates that dispatch on representative feeders. Agents use the resulting evidence to choose the next experiment.

In the primary evaluation, the annual operational $\mathrm { C O _ { 2 } }$ budget is a fixed scenario input enforced within each system optimization. The agent cannot relax this budget. Acceptance also checks the reported emissions against the fixed ceiling, alongside the system and representative-feeder requirements. The budget constrains direct operational emissions within the model boundary and does not establish compliance with the full institutional policy or planning framework from which its numerical value was drawn.

The single-agent architecture maintains one conversation and strategy notes. The multi-agent architecture separates specialist proposals, orchestration, evaluation, criticism, procedure refinement, compression and convergence judgement. Both architectures receive system and feeder evidence for every successful committed solve. In the multi-agent architecture, hypothetical system-only verifications support the judge’s decision to continue or stop within the current cycle. A candidate is physically accepted only after its committed system and distribution results satisfy every applicable physical and carbon-budget check. Throughout the search, a separate reporting record retains the lowest all-in-cost committed candidate that has passed every applicable physical and carbon-budget check, without changing the agents’ prompts or stopping decisions. This candidate is reported when the search ends through an agent stopping decision or the configured search ceiling.

Every search is initialized with the engine’s default design, which permits all extendable technologies, sets capacity bounds to their maxima and uses four-hour batteries without feeder interventions. Agents may modify this design before the first committed simulation. In the primary cohort, every system optimization includes the scenario-specific carbon budget. Agents then revise design restrictions and feeder controls using numerical feedback. Final selection retains the lowest all-in-cost committed candidate satisfying every implemented physical check and the fixed carbon budget. This selection rule is applied independently of the last proposal evaluated and the agent’s stopping decision.

## 2.2 Planning outcomes under fixed annual carbon budgets

The primary cohort contains forty completed trajectories across ten system–year scenarios. All forty retain an accepted committed candidate under the implemented system, representative-feeder and carbon-budget checks. Each retained result is feasible within its modelled scenario and stated tolerances. Numerical optimality bounds apply to the optimizer’s domain for that candidate, rather than to every design the agent could have proposed.

![](images/e136319ab5e091917ad2fd3ac44c8cea87cf151f39e16c5adb46dd04e8535035.jpg)

![](images/76ddeb8568b015bd6337cdfa2e0d2a51eadcda593cd0a8ee43808df5477f50b4.jpg)  
Fig. 1 Reality-grounded planning as an adaptive experimental loop. The broader input library covers eight systems and 58 model zones. Regional records, inherited assets, network assumptions, costs and demand outlooks define scenarios. ERA5 supplies 2023 weather; NASA projections for 2030 and 2035 are reconstructed with ERA5 intraday templates. PySAM and a turbine curve produce power profiles. Agents propose design bounds through a single conversation with strategy notes or a ten-role specialist team. Specialists propose, an orchestrator commits one experiment, and reviewers evaluate, critique, refine a procedure graph, compress memory and judge convergence. PyPSA optimizes investment and annual dispatch; OpenDSS screens representative feeders. Evidence returns to guide subsequent experiments, while a separate record retains the lowest-cost accepted committed candidate. In the primary evaluation, PyPSA also enforces an immutable annual operational CO<sub>2</sub> budget. The ERCOT traces and schematic, regional 2035 additions and 96-trajectory count illustrate the loose-cap reference evaluation under the original 1,000 Mt ceiling (Tables C3 and C4). Regional bars select the lowest observed accepted cost across four reference trajectories per system. Black denotes inherited assets and output; colours identify added technologies. Connections have model-zone resolution and the feeder is schematic. Primary carbon-constrained outcomes are reported in Fig. 2. The basemap uses Natural Earth[22].

Table 2 Selected annual costs in all forty carbon-constrained trajectories. Values are billion USD per year, including annualized representative-feeder interventions, rounded to six decimals. Each entry is one retained accepted committed candidate. S and M denote single- and multi-agent architectures. All four trajectories in every row pass the physical and fixed carbon-budget checks. Full precision, emissions, solver bounds and source identities accompany the archived numerical data.
<table><tr><td>System</td><td>Year</td><td>Sol S</td><td>Sol M</td><td>Opus S</td><td>Opus M</td><td>Accepted</td></tr><tr><td>NYISO</td><td>2023</td><td>8.030088</td><td>8.030121</td><td>8.030136</td><td>8.030121</td><td>4/4</td></tr><tr><td>NYISO</td><td>2030</td><td>6.561827</td><td>6.564198</td><td>6.564198</td><td>6.564198</td><td>4/4</td></tr><tr><td>NYISO</td><td>2035</td><td>8.438767</td><td>8.438767</td><td>8.438767</td><td>8.438767</td><td>4/4</td></tr><tr><td>MISO</td><td>2030</td><td>47.898514</td><td>47.984071</td><td>47.953352</td><td>47.898514</td><td>4/4</td></tr><tr><td>MISO</td><td>2035</td><td>55.959875</td><td>55.970535</td><td>55.968389</td><td>55.970535</td><td>4/4</td></tr><tr><td>South Africa</td><td>2030</td><td>12.019165</td><td>12.019593</td><td>12.051896</td><td>12.020077</td><td>4/4</td></tr><tr><td>South Africa</td><td>2035</td><td>14.365241</td><td>14.384747</td><td>14.405887</td><td>14.406203</td><td>4/4</td></tr><tr><td>Mexico</td><td>2030</td><td>22.257606</td><td>22.258090</td><td>22.258090</td><td>22.258090</td><td>4/4</td></tr><tr><td>Brazil</td><td>2030</td><td>24.729291</td><td>24.729291</td><td>24.729291</td><td>24.729291</td><td>4/4</td></tr><tr><td>Brazil</td><td>2035</td><td>33.736655</td><td>33.736655</td><td>33.736655</td><td>33.736655</td><td>4/4</td></tr></table>

The budget is active to within $1 0 ^ { - 5 } ~ \mathrm { M t }$ in twenty-eight retained outcomes across seven scenarios. NYISO 2023, Mexico 2030 and South Africa 2030 retain emissions below their budgets in all four configurations. Complete outcome tables retain all four deployment–architecture trajectories for every scenario. Portfolio and geographic illustrations select the lowest observed all-in annual cost among the four accepted candidates for each displayed system and year. GPT-6.1 Sol single-agent supplies all ten displayed portfolios, attaining the unique observed minimum in six scenarios and a minimum shared with another configuration in four, using a USD 10 annual-cost tie tolerance. These illustrations describe achieved designs; they do not establish that one deployment or architecture is generally superior. Comparisons of deployment or architecture retain every matched pair and the candidate-specific numerical bounds.

Figure 2 reports costs and emissions relative to the fixed budgets for all forty outcomes. Table 2 retains each selected cost, and Fig. A2 shows candidate-specific numerical bounds.

## 2.3 Accepted carbon-constrained portfolios retain their geographic context

For each of the ten system–year scenarios, we select the lowest observed all-in annual cost among four candidates satisfying the physical and carbon-budget checks. GPT-6.1 Sol single-agent supplies each representative, with tied minima identified. Figure 3 pairs the resulting grid additions with operational emissions for the latest evaluated year in each system. NYISO, MISO, South Africa and Brazil use 2035, while Mexico uses 2030. The years are explicit because these panels do not form a common-year comparison. All ten scenarios remain in the outcome tables and the detailed atlas.

![](images/8c3ba644ce5f6412ccbbf047b99273df8e31bbdaa8eaeb69b86f1294ae049daa.jpg)  
Fig. 2 Complete outcomes under annual operational carbon budgets. The ten system– year scenarios each contain two model deployments and two agent architectures. All forty retained outcomes satisfy the implemented physical and carbon-budget checks. Selected annual cost includes representative-distribution interventions. Emissions are direct operational $\mathrm { C O _ { 2 } }$ expressed relative to the corresponding scenario budget. A budget ratio of one denotes an outcome at the ceiling within numerical tolerance. The budgets are research constraints adopted from sources with the distinctions listed in Table 6.

The paired maps connect the selected investments to the annual emissions of the retained dispatch. Each panel reports total emissions against its fixed whole-system budget; no zonal emissions ceiling is imposed. Figure 4 separately distinguishes added generation, battery power and battery energy across all ten scenarios. The detailed atlas provides the zonal capacity and emissions ledgers. These allocations have modelzone resolution.

## 2.4 Carbon-constrained portfolios difer from published planning pathways

We compare the four retained accepted carbon-constrained outcomes for each of Brazil and South Africa in 2035 with technology-specific capacities from PDE 2035 and IRP 2025 (Table 3). These eight candidates satisfy their physical checks and annual

![](images/f118ede79c7934dce4dded5e17283044430dd0261d1db36972b45f3d5d1ce56d.jpg)  
Model-zone locations and schematic links. Budgets apply to each whole system.

Fig. 3 Optimized grid additions and operational carbon emissions. Paired maps show the minimum observed accepted-cost design for each system’s latest evaluated year. GPT-6.1 Sol single-agent supplies all representatives; tied minima and years are marked. Pie colours distinguish technology additions; grey and magenta links show existing connections and selected transfer increments. Fixed-size emission circles use a shared logarithmic colour scale; hollow circles denote zero. Each row reports annual cost, direct operational CO<sub>2</sub>, the whole-system budget and its source. Budget values and source-to-model mappings are in Table 6. Emissions are modelled from retained dispatch and declared inventory factors. Locations represent model zones. The ten-case atlas retains all years. The basemap uses Natural Earth[22].

d Minimum-cost configurations  
c Added battery energy  
![](images/f1c379b0709cfa6a5ee0113d2eec695eb18e48a409c92dc771df312a031b563c.jpg)

![](images/e8f9e63c5355b2b3ba7fe9e0fe7a43fd343f82fbb24752c0803d147346c94f9c.jpg)

![](images/c85faefb07ff9bfd9efb476e19baa5d8d81e3d9251200c266672ac9342c0724e.jpg)  
Fig. 4 Minimum observed accepted portfolios under carbon budgets. Each system–year scenario shows the portfolio with the lowest observed all-in annual cost among its four accepted trajectories. Six scenarios have a unique observed minimum and four have tied minima within a USD 10 annual-cost tolerance. Panels a–c display the GPT-6.1 Sol single-agent representative, while panel d lists every deployment–architecture configuration attaining the observed minimum within this tolerance. Generation and storage values describe additions to the inherited fleet, with battery power and energy distinguished. Selection is conditional on the system, representative-feeder and carbonbudget checks. The complete comparisons retain all forty trajectories and their numerical bounds.

<table><tr><td>NYISO 2023</td><td>Sol single</td><td>8.030</td></tr><tr><td>NYISO 2030</td><td>Sol single</td><td>6.562</td></tr><tr><td>NYISO 2035</td><td>Sol single, Sol multi Opus single, Opus multi</td><td>8.439</td></tr><tr><td>MISO 2030</td><td>Sol single, Opus multi</td><td>47.899</td></tr><tr><td>MISO 2035</td><td>Sol single</td><td>55.960</td></tr><tr><td>Mexico 2030</td><td>Sol single</td><td>22.258</td></tr><tr><td>Brazil 2030</td><td>Sol single, Sol multi Opus single, Opus multi</td><td>24.729</td></tr><tr><td>Brazil 2035</td><td>Sol single, Sol multi Opus single, Opus multi</td><td>33.737</td></tr><tr><td>South Africa 2030</td><td>Sol single</td><td>12.019</td></tr><tr><td>South Africa 2035</td><td>Sol single</td><td>14.365</td></tr><tr><td></td><td></td><td></td></tr></table>

Minimum cost (billion USD yr⁻¹)

model emissions budgets of 29 and $\mathrm { 1 4 2 \ M t C O _ { 2 } }$ , respectively. The comparisons use total capacity, including inherited assets, and retain all four deployment–architecture configurations.

Brazil’s four outcomes contain 120.19 GW of wind, compared with 49.55 GW in PDE 2035. Their 96.66 GW of model solar is compared with the plan’s 29.19 GW centralized-solar category, which excludes a further 78.10 GW of micro-/minidistributed generation. The model does not represent a separate future distributedgeneration rollout, so this category-level diference does not measure a diference in total national photovoltaic capacity. The plan includes 6.63 GW of batteries, whereas all four carbon-constrained outcomes contain zero battery capacity.

Table 3 Published plans and carbon-constrained portfolios in 2035. Values are total capacities in GW, including inherited assets. Carbon ranges span all four retained accepted deployment–architecture trajectories per system. Diferences are carbon-constrained agent values minus the published plan, calculated before rounding. Brazil and South Africa use annual model budgets of 29 and 142 $\mathrm { M t C O _ { 2 } } ,$ respectively.
<table><tr><td>System</td><td>Capacity</td><td>Expert plan</td><td>Carbon range</td><td>Difference</td></tr><tr><td>Brazil</td><td>Wind</td><td>49.55</td><td>120.19</td><td>+70.64</td></tr><tr><td>Brazil</td><td> $\mathrm { S o l a r } ^ { \dagger }$ </td><td>29.19</td><td>96.66</td><td>+67.47</td></tr><tr><td>Brazil</td><td>Battery power</td><td>6.63</td><td>0.00</td><td>-6.63</td></tr><tr><td>South Africa</td><td>Wind</td><td>26.39</td><td>5.57–6.24</td><td>[-20.82,-20.14]</td></tr><tr><td>South Africa</td><td> $\mathrm { S o l a r } ^ { \dagger }$ </td><td>21.16</td><td>63.02–64.24</td><td>[+41.86, +43.08]</td></tr></table>

<sup>†</sup> Brazil’s PDE 2035 centralized-solar category excludes 78.099 GW of micro-/mini-distributed generation. South Africa’s IRP 2025 Solar PV column excludes 16.065 GW in its separate distributed-generation column, CSP and hybrid capacity. Model solar has no separate future distributed-generation rollout. The Brazilian values describe December 2035; the South African values combine the end-2025 base and scheduled additions through 2035. Demand, resources, inherited fleets, retirements, technology options and delivery schedules retain their respective study assumptions. Source locations and mappings are given in Methods and Appendix G.

South Africa’s outcomes contain 5.57–6.24 GW of wind and 63.02–64.24 GW of solar. The IRP schedule yields 26.39 GW of wind and 21.16 GW in its Solar PV column by 2035, with 16.07 GW in a separate distributed-generation column. The portfolios therefore retain a diferent technology balance while satisfying the adopted annual emissions ceiling. The numerical ceilings share institutional sources with the capacity references, while resource, retirement, technology and delivery assumptions remain distinct. These comparisons describe the resulting portfolios rather than validate them against withheld planning targets.

## 2.5 Recorded efort in the carbon-constrained searches

The forty carbon-constrained trajectories contain 202 logical system-evaluation requests, comprising 130 committed simulation requests and 72 hypothetical systemonly verification requests. These counts include requests served through exact-result reuse. The archived summaries record 764 received model responses, 90,562,224 input tokens and 1,314,432 output tokens. Input includes reported cache reads and writes, and output includes the reported reasoning subset. Provider usage conventions, recovery histories and the result-reuse revision remain attached to the trajectory evidence. The counts describe search efort rather than a controlled wall-runtime benchmark.

Figure 5 retains the counts for every deployment–architecture–scenario trajectory. Received responses, logical evaluation requests and newly executed numerical optimizations measure diferent components of efort.

![](images/4c4dd51bf481b16b23998c93ecac53adfa830ae6cbaeaad39449caec5323f82c.jpg)  
Fig. 5 Recorded efort in the carbon-constrained planning searches. Logical systemevaluation requests, received API responses and provider-reported tokens describe diferent components of search efort. The forty trajectories contain 202 logical system-evaluation requests and 764 recorded received responses, with 90.562 million input tokens and 1.314 million output tokens. Log ical requests include exact-result reuse and therefore are not counts of newly executed numerical optimizations. Recovery histories and the recorded reuse revision are retained in the evidence. These summaries do not constitute a wall-runtime or scaling benchmark.

## 2.6 Loose-cap reference scenarios distinguish imposed limits from observed emissions

The original cohort used an annual $\mathrm { C O _ { 2 } }$ ceiling of 1,000 Mt. We retain its ninetysix trajectories as a separate reference dataset and compare the forty trajectories matching the primary cohort by system, year, deployment and architecture. Twentyeight of these reference designs exceed the tighter budgets, corresponding to seven scenarios. The comparison reports achieved annual cost, operational emissions and investment portfolios for each matched pair. In NYISO 2035, the selected annual cost is USD 8.439 billion in all four carbon-constrained configurations, 23.56% above each matching loose-cap outcome. These observed contrasts are conditional on the archived trajectories and their respective optimization domains. A source-derived ceiling need not change the selected outcome when the corresponding loose-cap design already emits below it. Diferences in binding and non-binding scenarios must therefore be distinguished when interpreting the response to a tighter budget.

These are comparisons of archived agent trajectories under diferent scenario inputs. The 2023 Opus reference used a 16,000-token output ceiling and an earlier provider adapter, whereas the primary campaign used a 64,000-token ceiling and the updated adapter. Those two NYISO pairs are reported separately from the other matched pairs. Execution revisions and exact-result reuse are also recorded, and one trajectory per condition does not support estimates of stochastic variation or an isolated causal efect of the carbon ceiling on the full agent search.

Figure A1 shows the matched cost and emissions contrasts. Across the full loose-cap archive, 36 assessments in 16 searches had a lower annual objective than a previously accepted design but failed a feeder check. In one ERCOT 2035 search, removing a proposed line upgrade reduced the annual objective by 0.220% but caused three hours of undervoltage. The agent team restored the upgrade in the next round and the framework retained the earlier accepted design (Fig. C2 and Appendix C).

## 3 Discussion

The central result is autonomous execution of a coupled electricity-planning process across heterogeneous regional systems. Starting from prepared inputs, agents interpret numerical feedback, revise investment and operating decisions, and determine when to end the search. Automated system and representative-feeder evaluations, acceptance checks and final selection carry these decisions through to retained plans. All forty primary trajectories retain candidates satisfying the implemented physical and carbon-budget checks. Twenty-eight matched loose-cap reference designs exceed the new scenario ceilings. The budgets are enforced by the optimizer; agents navigate design choices and interpret system and representative-feeder evidence within these fixed constraints.

Comparisons between the carbon-constrained designs and published plans make the interpretation of numerical acceptance more concrete. Brazil’s larger modelled wind fleet and South Africa’s solar-heavy portfolio persist under the adopted emissions ceilings. Satisfying a plan-derived annual carbon budget therefore does not reproduce the plan’s technology mix. Model diagnostics document systematic diferences in how energy models respond to scenario assumptions[23]; transmission studies also expose trade-ofs among cost, emissions and reliability under policy-driven and costminimizing designs[11]. Evidence from transition modelling further identifies roles for social acceptance, investment risk and infrastructure lock-in[24]. The portfolio diferences identify retirement, resource, technology and staged-delivery assumptions for further sensitivity analysis.

Investment timing and available technologies remain important model boundaries. Apart from storage and the hydrogen chain, nuclear is the only new dispatchable technology in the simulator configuration; gas, coal and hydro plants exist only as inherited capacity. The single-period formulation makes selected additions available without a construction schedule, financing sequence or permitting constraint, while the inherited fleet remains available throughout the modelled year. Investment-pathway studies show why foresight and deployment timing matter for the transition[4, 25].

The carbon-constrained cohort adopts numerical limits from several kinds of institutional evidence. Allowance budgets, planning constraints and projected emissions are not interchangeable policy instruments. Our source-to-model mappings turn these quantities into fixed annual model constraints, while retaining diferences in geographical coverage, emissions accounting, inherited assets and resource assumptions. Mexico and Brazil additionally use published CO<sub>2</sub>-equivalent values as numerical proxies for direct operational $\mathrm { C O _ { 2 } }$ . These choices define research scenarios and do not demonstrate legal compliance, lifecycle decarbonization or reproduction of the complete institutional plans.

Local data preparation, source interpretation and the predefined simulator interfaces remain essential expert contributions. Evaluation with regional practitioners would extend the study to systems with poorer network records, constrained financing or substantial unmet demand.

Further development can address network detail, climate uncertainty and replication. Network evaluation uses reduced representative feeders and a zonal transport formulation. The fixed annual reliability tolerance permits limited unmet demand. Detailed AC network, contingency and site-level analyses would extend this representation.

Weather-aware studies have coupled planning and operation across weather years or optimized under climate uncertainty and extremes[26, 27]. Recent work further connects climate-sensitive adequacy to renewable siting and system flexibility[28, 29]. Our future weather uses one climate model, one emissions scenario and synthetic hourly chronology. The change of data product between 2023 and the future years is not separated from projected climate change. A same-GCM historical control, multiple weather realizations and a climate-model ensemble would extend the analysis to climate attribution and robustness.

Each deployment–architecture–scenario cell contains one trajectory, and campaign inference settings difer. Solver bounds apply to each candidate-specific optimization problem.

A controlled expert comparison would hold inputs, technology options, reliability criteria and computational resources fixed, then evaluate both expert-guided and agent-guided candidates through the same independent gate. A complementary ablation would compare the adaptive loop with a predefined sequence of system and feeder checks. These tests would separate the contribution of adaptive decisions from the numerical models and data preparation.

The present results demonstrate autonomous completion of electricity-planning studies under fixed operational carbon budgets, grounded in public regional data and institutional demand outlooks, with the reference archive extending regional and temporal coverage. Explicit regional assumptions, committed dispatch, rejected and accepted candidates, and recorded search histories make both the resulting plans and the decisions leading to them inspectable.

## 4 Methods

## 4.1 Regional systems and data basis

The primary evaluation covers five electricity systems in four countries, drawn from the broader eight-system, six-country input library in Table 4; the loose-cap reference evaluation covers all eight systems. Each system is represented as a coupled zonal planning problem in brownfield mode, with the inherited generation and storage fleet available for dispatch alongside optimized additions.

Table 4 Selected evaluation systems. Counts describe demand aggregation units within each coupled system. All reported trajectories use brownfield mode. The common base demand year is 2023, with diferent source types retained explicitly.
<table><tr><td>System</td><td>Units</td><td>Demand basis</td></tr><tr><td>MISO</td><td>6</td><td>EIA-930 hourly observations, grouped LRZs</td></tr><tr><td>ERCOT</td><td>8</td><td>EIA-930 hourly observations, weather zones</td></tr><tr><td>NYISO</td><td>11</td><td>EIA-930 hourly observations, zones A-K</td></tr><tr><td>South Africa</td><td>10</td><td>ESRG reference scenario, corrected supply-area hourly profiles</td></tr><tr><td>Australia (NEM)</td><td>5</td><td>AEMO five-minute regional TOTALDEMAND from price-and- demand records, averaged to hourly</td></tr><tr><td>New Zealand</td><td>5</td><td>EMI metered grid offtake by point of connection, summed to the five EMI zones</td></tr><tr><td>Mexico</td><td>9</td><td>CENACE daily real-demand settlement reports, hourly by con- trol area</td></tr><tr><td>Brazil</td><td>4</td><td>ONS hourly load curve by subsystem</td></tr></table>

The eight systems contain 58 coupled demand units in total.

National data sources include EIA for the United States[30], ESRG for South Africa[31], AEMO for Australia[32], EMI for New Zealand[33], SENER for Mexico[34], and ONS for Brazil[35]. Specific source editions for demand, fleet, network and forecast inputs and expert-plan comparisons are listed with direct links in Appendix G.

The US demand inputs contain 2023 hourly observations for six MISO reporting groups, eight ERCOT weather zones and eleven NYISO zones. MISO’s groups are LRZ 1, LRZ 4, LRZ 6, LRZs 2+7, LRZs 3+5 and LRZs 8+9+10. Existing generation is drawn from EIA-860 and mapped using geographic and identifier records, including NYISO’s generator-zone assignments.

South Africa uses the ESRG demand model and its spatialized reference-scenario outputs. The ten electricity supply areas are Eastern Cape, Limpopo, Mpumalanga, Gauteng, Western Cape, Northern Cape, Hydra Central, KwaZulu-Natal, North West and Free State. The supply-area partition assigns Hydra Central separately. The published series contains 8,760 hourly model outputs per year for 2023–2050, informed by Eskom regional transformer data and national demand projections. The archived spatialization notebook uses 2017 provincial load profiles as calibration inputs.

The common base demand year is 2023. South African demand uses the spatialized model outputs, and Eskom ESK19622 supplies a national demand and load-reduction reference. Existing South African capacity follows Eskom’s 2023 nominal fleet and independent-power-producer totals, with documented spatial allocation and operating assumptions.

Cost inputs retain the NREL Annual Technology Baseline basis for US electricity technologies and PyPSA technology-data for South Africa and the hydrogen chain[36, 37]; the Australian, New Zealand, Mexican and Brazilian packs use the ATB basis in US dollars as a declared assumption, and published project capex is converted at 2023 average exchange rates and annualized at a 7.5% capital recovery factor. The South African cost inputs use a 10% discount rate.

Input manifests record prepared data, raw extracts and source documents with SHA-256 digests. Baseline resources use one ERA5 point per demand node; NASAweather input packs use the reconstruction in Section 4.10. Each result records its input, code and dependency identities.

Existing assets are assembled from dated inventory records and mapped to the model’s demand zones. The US packs use the 2023 EIA-860 inventory, including battery-energy schedules and the documented geographic and generator-identifier mappings. South African capacity follows Eskom’s 31 March 2023 nominal-fleet and independent-power-producer totals, with WRI plant coordinates and historical capacity weights supporting spatial allocation[38]. Australian registered capacities and maximum storage energies come from AEMO’s archived Registration and Exemption List; the December 2023 MMSDM dispatch-unit records identify units registered on 31 December 2023, with identifier-stem matching for batteries whose registration format changed. New Zealand uses the EMI generation inventory with commissioning and decommissioning dates to select units operating during 2023. Mexico uses the state-level technology capacities in PRODESEN 2024–2038, Annex 1, mapped to the model’s control regions. Brazil uses ONS unit-level efective capacities and entry/exit dates; the ANEEL SIGA register is retained as a coordinate reference in the source archive. Measured 2023 hydro generation supplies the annual hydro-energy budgets for New Zealand and Brazil. Availability factors, storage-duration fallbacks and regional allocation rules are declared modelling assumptions.

Inter-area transmission is a zonal abstraction constrained by published topology and limits where available. Each region carries a zonal network assembled from public documents (Table 5). Only zone pairs that are electrically adjacent in the source topology are corridors. Existing capacities are taken from published values where they exist, and published limits that apply to several paths at once are kept as interface constraints on the summed flow rather than converted to single-line ratings. MISO publishes zonal import and export limits rather than pairwise ratings, and these enter as zone constraints. Oficial planned projects are separate build decisions that raise the limits they cross by published increments where available and by declared assumptions otherwise. A generic additional circuit along each existing route is also available as a labelled assumption. Where an operator publishes no pairwise capacity, the corridor carries a non-binding proxy so that the published limits are the binding constraints; the published zonal and interface constraints remain explicit in the model.

The ERCOT fixed limits retain specific study assumptions. The 12,240 MW limit is the RTP economic study’s 2027 West Texas Export value without outages, and 5,595 MW is the North-to-Houston base value with planned facilities. The source applies an hourly profile to the latter; our model holds the base value fixed. The 3,500 MW improvement reported for the full 765 kV STEP Core Plan, from 12,700 to 16,200 MW, is assigned to the Permian import-path candidate as an aggregate model approximation. Applied to the model’s 12,240 MW baseline, that assigned increment gives 15,740 MW before any additional generic-circuit increments.

Table 5 Zonal transmission networks in the evaluation package. Public records supply topology, interface limits and project descriptions. Corridor aggregation, estimated ratings, non-binding proxies and assigned project increments are modelling choices recorded in the input manifests.
<table><tr><td>System</td><td>Corridors</td><td>Network basis</td><td>Candidate projects</td></tr><tr><td>NYISO</td><td>12</td><td>Directional transfer limits; five model cut-sets assembled from operating and reliability</td><td>Propel NY (1,300 MW into Zone K)</td></tr><tr><td>MISO</td><td>7</td><td>studies Zonal import/export limits; Midwest-South physical transfer limits</td><td>LRTP Tranche 1 and 2.1 lines, segments of one project bundled</td></tr><tr><td>ERCOT</td><td>17</td><td>3,000/2,500 MW Fixed study limits of 12,240 MW for West Texas Export and 5,595 MW for</td><td>Permian 765 kV import paths and STEP eastern backbone; aggregate STEP increment assigned as described below</td></tr><tr><td>South Africa</td><td>21</td><td>North-to-Houston NTCSA line GIS aggregated to supply areas; ratings estimated using the</td><td>31 inter-area candidates extracted from the TDP</td></tr><tr><td>Australia</td><td>5</td><td>PyPSA-RSA method[39, 40] Static directional capabilities from AEMO&#x27;s archived September 2025 edition (v5)</td><td>VNI West, QNI Connect, Marinus Link stages 1/2 and EnergyConnect; ISP</td></tr><tr><td>New Zealand</td><td>4</td><td>HVDC inter-island capability from the Transmission Planning Report 2023; island backbones as non-binding</td><td>Appendix 5 HVDC cable replacement and possible fourth-cable upgrade to 1,400 MW northward</td></tr><tr><td>Mexico</td><td>10</td><td>proxies PRODESEN Annex 1 circuit/voltage inventory aggregated to nine model control areas; MW ratings estimated by a declared</td><td>none in the base pack</td></tr><tr><td>Brazil</td><td>4</td><td>ONS pairwise interchange limits and Northeast/Southeast receiving</td><td>Graça Aranha-Silvânia HVDC candidate, represented by an assumed 4 GW bipole</td></tr></table>

NYISO’s published Propel NY gain is 1,300 MW of additional import capability into Long Island. The model assigns 1,300 MW to both directions of the I–K corridor and to its selected ConEd–LIPA and Dunwoodie/Y49/Y50 interface constraints; reverse interface limits receive the increment where present. These mappings are proxies for the aggregate project benefit. New Zealand’s HVDC candidate increases northward capacity from 1,200 to 1,400 MW and assumes a southward increase from

850 to 1,000 MW. The possible fourth-cable enhancement in the TPR supports the northward target; the 150 MW southward increment is a study assumption.

MISO’s 3,000/2,500 MW constraints represent the physical regional directional transfer limits, separately from the auction’s subregional import/export allowances. South African corridor counts and the 31 TDP candidates are study aggregations of GIS records and planned lines; expected in-service years determine eligibility under the study’s stated rule. Australia’s archived nominal interconnector capabilities are held fixed, although realized capability depends on operating conditions. New Zealand’s 1,400 MW candidate represents the possible fourth-cable enhancement discussed alongside cable replacement, including the need to assess reactive support.

The representative feeders are calibrated so that bare peak load gives approximately 0.96 per unit on the 12.47 kV US feeder and 0.98 per unit on the 22 kV South African feeder. Demand, photovoltaic output, and existing and new battery operation are mapped to those circuits. Line-current limits are 400 A for the normal circuit and 600 A for the reinforced circuit.

## 4.2 Geographic representation of committed outcomes

The carbon-constrained geographic figures read retained committed candidates without new inference, optimization or power-flow simulation. Each of the ten system–year scenarios displays its minimum observed accepted-cost portfolio across the four deployment–architecture trajectories. GPT-6.1 Sol single-agent supplies the representative in all ten scenarios, with tied minima identified explicitly. The carbon maps show model-zone additions and schematic connections. The main figure displays the latest evaluated year in each of the five systems, while the atlas retains all ten system– year scenarios. Capacity-pie area increases with added generation and battery power, subject to a minimum visible area. For each model zone z, annual emissions are $\begin{array} { r } { E _ { z } = 1 0 ^ { - 6 } \sum _ { g : \mathrm { b u s } ( g ) = z } \epsilon _ { g } \sum _ { t } p _ { g , t } \Delta t } \end{array}$ , calculated from the retained accepted dispatch and checked by summing the zonal values back to the reported system total. Fixed-size emissions circles share a logarithmic scale from 0.01 to 200 Mt per year; smaller positive values receive the minimum colour and zero values are hollow. This is generation-based direct operational accounting, not consumption-based allocation, lifecycle accounting or measured plant-level emissions. The budget applies to the system total rather than individual zones.

The reference figures C1 and C2 read committed candidates without new inference, optimization or power-flow simulation. Figure C1 and Appendix E select the minimum observed accepted 2035 cost across the four deployment–architecture trajectories per system; tied minima retain the displayed representative. Figure C2 separately follows the Opus multi-agent ERCOT 2035 repair trajectory. New and inherited capacity totals are aggregated from the retained asset records by model node. Added transfer capacity sums selected generic-circuit and oficial-project increments on each model corridor.

Natural Earth version 5.1.1, at 1:50 million scale, supplies public-domain country and US state boundaries[22]; each panel uses a local Lambert azimuthal equal-area projection. Model-node coordinates come from the prepared regional input packs. They represent grouped-zone centroids, administrative proxies or reference cities, as recorded for each node. In the reference figures, US operating-plant points join the archived 2023 inventory (status OP) to plant coordinates and are deduplicated by plant code. The other five systems use model-zone coordinates. New capacity is allocated by zone, and model connections are drawn as straight schematic corridors.

The reference ERCOT verification panels use annual extrema, currents, ratings and violation counts already committed in the initial and selected Far West representativefeeder outputs. Endpoint colours show annual feeder voltage extrema; line colours show annual per-line peak current divided by the corresponding candidate’s rating. Extrema may occur at diferent hours. The stylized three-bus circuit depicts the numerical feeder topology. A frozen geographic evidence file and SHA-256 source manifest record all extracted values, coordinate files and basemap archives.

## 4.3 Brownfield investment and dispatch

Existing generation and storage retain their locations and capacities while dispatch and additional investment are optimized. Historical construction costs are treated as sunk; specified existing fixed costs and operating costs remain included. The model represents one planning year with exogenous demand. Each demand area has one representative bus with hourly load, photovoltaic and wind profiles.

Inter-area transmission uses a zonal transport formulation. Each corridor is represented by directed links with published or declared capacity and a wheeling cost of 0.1 USD/MWh. Published multi-path limits enter as cut-set constraints, while zonal import and export limits constrain net nodal interchange. Candidate circuits and oficial projects are binary build decisions. An investment raises the constraints it crosses by a published increment or a declared capacity assumption. Equivalent project choices are ordered to remove symmetric branches without changing the set of feasible networks.

## 4.4 Task formulation and acceptance criteria

Let w denote scenario assumptions, d an agent-selected design and x the capacities and dispatch optimized by the system model. For each candidate, the optimization problem computes

$$
x ^ { \star } ( d ; w ) \in \arg \operatorname* { m i n } _ { x \in \mathcal { X } ( d ; w ) } \left[ \sum _ { k } a _ { k } K _ { k } + \sum _ { t , k } m _ { k } p _ { k , t } \Delta t + v \sum _ { t } s _ { t } \Delta t \right] ,\tag{1}
$$

where $a _ { k }$ is the annualized capacity coeficient, $K _ { k }$ the associated capacity, $m _ { k }$ the marginal production cost, $s _ { t }$ unserved power and v the value of lost load. Conversion and storage capacities enter the first term with their corresponding power or energy units. Each of the 8,760 snapshots represents one hour. In the regional package the problem is a mixed-integer programme. Binary variables $y _ { c }$ for candidate circuits and $z _ { p }$ for oficial projects fix the corresponding link capacities, and each interface I with member corridors $c \in I .$ , directions $\sigma _ { c } ,$ base limit $L _ { I }$ and increments $\delta _ { p , I }$ satisfies

$$
\sum _ { c \in I } \sigma _ { c } f _ { c , t } \leq L _ { I } + \sum _ { p } \delta _ { p , I } z _ { p } + \sum _ { c \in I } r _ { c } y _ { c } \quad \forall t ,\tag{2}
$$

with $f _ { c , t }$ the signed corridor flow and $r _ { c }$ the rating of an assumed expansion. Zonal import and export limits take the same form on net nodal import. The feasible set further includes nodal power balance, technology availability, capacity bounds, cyclic storage dynamics, conversion eficiencies, annual energy budgets for hydro and concentrating solar, and a fixed annual $\mathrm { C O _ { 2 } }$ budget.

For an annual budget $B _ { r , y }$ in $\mathrm { M t C O _ { 2 } }$ , direct operational emissions satisfy

$$
E ( d ; w ) = \frac { 1 } { 1 0 ^ { 6 } } \sum _ { g \in \mathcal { G } _ { \mathrm { e m i t } } } \sum _ { t = 1 } ^ { 8 7 6 0 } \epsilon _ { g } p _ { g , t } \Delta t \leq B _ { r , y } ,\tag{3}
$$

where ${ p } _ { g , t }$ is generator dispatch in ${ \mathrm { M W } } , \epsilon _ { g }$ is the supplied direct emissions factor in tonnes of $\mathrm { C O _ { 2 } }$ per MWh, and $\Delta t = 1$ h. The sum covers generators with nonzero inventory emissions factors. Lifecycle emissions and emissions associated with external electricity imports are outside this accounting boundary. The annual budget is immutable through the agent action interface. Short diagnostic horizons receive a proportional budget, but all reported planning outcomes use 8,760 hours. The inventory emissions factors are declared technology-level modelling defaults per MWh of electrical output rather than plant-level operating measurements. For the emitting technologies represented here, they are 0.95, 0.42, 0.75, 0.40, 0.04 and 0.50 $\mathrm { t C O _ { 2 } / M W h }$ for coal, gas, oil, waste, geothermal and other generation, respectively; biomass is assigned zero direct emissions under the inventory convention.

The agent evaluates complete designs using

$$
J ( d ; w ) = J _ { \mathrm { s y s t e m } } ( d ; w ) + C _ { \mathrm { d i s t } } ( d ; w ) .\tag{4}
$$

Acceptance requires a valid solver solution, unmet energy below $1 0 ^ { - 4 }$ of demand, passing voltage, convergence and thermal checks at every representative feeder, and nodal, storage and interface residuals within the declared tolerances. For the primary cohort, a separate carbon-budget audit checks finite reported emissions, the fixed scenario and result ceilings, source identity and the 8,760-hour horizon. The emissions reporting tolerance is max $( 1 0 ^ { - 9 } , 1 0 ^ { - 8 } B _ { r , y } )$ Mt, and does not relax the optimizer’s budget. Gurobi uses eighteen threads, a barrier root relaxation, a requested relative MIP gap of 0.1% and a 24-hour ceiling per solve[41]. Each numerical cost interval spans the solver bound and incumbent objective for the candidate’s fixed optimization domain, with the selected distribution cost added to both endpoints. Unserved energy is penalized at $v = 5 0 , 0 0 0 \ \mathrm { U S D / M W }$ h and checked against the acceptance threshold after optimization.

Let $\mathcal { A }$ denote committed candidates satisfying all applicable physical and carbonbudget acceptance checks. The selected result is

$$
\hat { d } \in \arg \operatorname* { m i n } _ { d \in \mathcal { A } } J ( d ; w ) .\tag{5}
$$

Equal-cost candidates retain the first recorded accepted result. Hypothetical systemonly evaluations are outside A. This rule is applied to the complete committed trajectory for every reported deployment, independently of its final attempted design or stopping decision.

## 4.5 Carbon-budget scenarios and source mappings

Ten researcher-selected system–year scenarios adopt annual emissions ceilings from archived oficial policy and planning sources (Table 6). We restricted the carbonconstrained evaluation to system–year pairs with a documented mapping from an oficial electricity-sector emissions budget or pathway to an annual model ceiling, excluding cases such as MISO 2023 and Mexico 2035. These ceilings are implemented through the existing operational co2 budget constraint. The original input packs remain unchanged on disk, and a fixed scenario overlay records the source value, unit conversion, mapping assumptions and content hashes. The baseline policy mode leaves this independent constraint active. Agents can revise design variables but cannot alter the budget, its source identity or the acceptance requirements. The oficial sources determine the scenario ceilings, whereas achieved emissions are calculated from optimized dispatch and the declared inventory coeficients.

NYISO uses New York’s RGGI base allowance budgets as model-wide annual ceilings. US short tons are converted to metric tonnes using the exact multiplier 0.90718474. Regulated-unit coverage, interstate allowance trading, banking, reserve supply and multiyear compliance are not recreated. MISO combines the 2025 Future 2 aggregate planning emissions constraint with the 2026 LTLF Current Trajectory demand inputs. South Africa combines the IRP 2025 balanced-plan emissions pathway with ESRG spatialized reference demand excluding exports. These combinations preserve the published source values without claiming to reproduce each oficial scenario in full.

Mexico adopts the PROSENER 2030 electricity-emissions projection, combining it with the existing PRODESEN demand inputs. The model excludes SIMUL/Mulege and does not infer a proportional emissions allocation to that excluded subsystem. Brazil uses the PDE 2035 SIN projection, corresponding to the model’s four interconnected subsystems, rather than the larger whole-sector projection. The Brazilian plan reports these quantities as projected outcomes rather than imposed emissions limits, and its average-hydrology assumption difers from the model’s inherited hydroenergy assumptions. Mexico and Brazil report $\mathrm { C O _ { 2 } } \mathrm { - e q u i v a l e n t }$ emissions. We adopt their numerical values as direct $\mathrm { C O _ { 2 } }$ ceilings as an explicit proxy, without converting between greenhouse gases. Exact document locations and source-to-model assumptions are preserved in the carbon-scenario archive.

Table 6 Annual operational CO<sub>2</sub> budgets in the primary evaluation. Each scenario contains both deployments and both architectures. Displayed values are rounded; the archive retains exact source values and conversions. The source categories are retained because an allowance budget, a planning-model constraint and a projected emissions outcome have diferent institutional meanings. Mexico and Brazil use published $\mathrm { C O } _ { 2 }$ -equivalent values as numerical proxies for direct operational CO<sub>2</sub>.
<table><tr><td>System</td><td>Year</td><td>Budget  $( \mathrm { M t \ y r ^ { - 1 } } )$ </td><td>Source basis</td></tr><tr><td>NYISO</td><td>2023</td><td>24.761865</td><td>New York RGGI base allowance budget</td></tr><tr><td>NYISO</td><td>2030</td><td>12.425308</td><td>New York RGGI base allowance budget</td></tr><tr><td>NYISO</td><td>2035</td><td>3.550088</td><td>New York RGGI base allowance budget</td></tr><tr><td>MISO</td><td>2030</td><td>200.033343</td><td>Future 2 aggregate planning constraint</td></tr><tr><td>MISO</td><td>2035</td><td>164.641838</td><td>Future 2 aggregate planning constraint</td></tr><tr><td>South Africa</td><td>2030</td><td>168</td><td>IRP 2025 balanced-plan emissions pathway</td></tr><tr><td>South Africa</td><td>2035</td><td>142</td><td>IRP 2025 balanced-plan emissions pathway</td></tr><tr><td>Mexico</td><td>2030</td><td>119</td><td>PROSENER electricity-emissions projection</td></tr><tr><td>Brazil</td><td>2030</td><td>27</td><td>PDE 2035 SIN emissions projection</td></tr><tr><td>Brazil</td><td>2035</td><td>29</td><td>PDE 2035 SIN emissions projection</td></tr></table>

## 4.6 Resource and demand construction

Photovoltaic capacity factors use the PVWatts v8 model through PySAM[42, 43]. Baseline profiles use ERA5 from the Open-Meteo historical archive, including global, direct-normal and difuse irradiance, temperature and 100 m wind speed[44, 45]. Arrays are fixed racks facing the equator, with tilt equal to absolute latitude, a DCto-AC ratio of one and 14% losses. Radiation averages are aligned to the load-interval start. The same conversion settings are applied to reconstructed future weather.

Wind capacity factors use the archived 2020 ATB NREL reference turbine curve, rated at 4 MW with a 150 m rotor[46]. For the ERA5-weather evaluation, each node’s ERA5 100 m wind-speed series feeds this curve directly. The same reference curve is used across regions, while the node-specific weather produces distinct hourly resource profiles.

US demand uses EIA-930 hourly subregion records aligned to UTC. EIA-930 bulksystem delivery and the South African sent-out demand include distribution losses in their accounting basis. South African demand retains the published regional values and 8,760-hour model-year convention. Each prepared pack aligns load and resource profiles on its declared hourly index.

## 4.7 Cost and storage representation

US electricity-technology costs are derived from the locally archived 2024 ATB version 4, using the Market financing case, Moderate scenario and 2030 technology values with a 30-year cost-recovery period[36]. The battery power and energy coeficients are fitted from the available duration-specific products using the derivation script. South African technology costs and the hydrogen chain use archived PyPSA technology-data inputs[37]; Australia, New Zealand, Mexico and Brazil use the declared ATB electricity-cost basis in US dollars. Annualization assumptions and per-value provenance are retained in the derived cost JSON files.

Battery power $P _ { b }$ and energy $E _ { b }$ are related by the agent-selected duration $h _ { b } .$ , so $\boldsymbol { E _ { b } } = h _ { b } \boldsymbol { P _ { b } }$ and the annualized investment contribution is

$$
\begin{array} { r } { C _ { b } = ( a _ { P } + h _ { b } a _ { E } ) P _ { b } . } \end{array}\tag{6}
$$

New batteries use charging and discharging eficiencies of 0.95 each; existing pumped storage and other storage use declared round-trip eficiencies. Storage is cyclic over the simulated horizon. Hydrogen is represented by an electrolyzer link, an energy store on a hydrogen bus and a fuel-cell link with independently extendable capacities. New nuclear is an extendable generator with a minimum output of 70% of installed capacity; existing nuclear is dispatched flexibly. Existing thermal, hydro and biomass fleets are fixed generators with declared availability, marginal cost and emission rate, and hydro and concentrating solar carry annual energy budgets. Historical construction costs of existing assets are sunk; declared annual fixed costs remain in the objective. The primary cohort uses the scenario-specific annual $\mathrm { C O _ { 2 } }$ budgets in Table 6; the reference archive retains the original 1,000 Mt annual ceiling.

## 4.8 Representative-feeder model

The representative feeder has a source bus, two serial line segments and two loads allocated 40% and 60% of feeder demand, with photovoltaic generation at the downstream bus and battery charging and discharging split between the two buses by a design variable. Both loads use a nominal power factor of 0.95. US feeders are 12.47 kV with 2 km sections and South African feeders 22 kV with 3 km sections, at 0.25 and 0.33 ohm/km; these lengths were calibrated so that the bare peak load sits inside the band. A node’s demand is mapped to a 5 MW feeder at the node’s peak, with photovoltaic and battery dispatch mapped by the same share. Hourly photovoltaic injection and battery power follow the optimized system dispatch.

At each hour, OpenDSS solves a snapshot and returns bus-voltage magnitudes and line currents. The acceptance band is 0.95–1.05 per unit, non-convergent hours count as failures, and line currents are checked against 400 A normal and 600 A reinforced ratings. If reactive control is enabled and the initial maximum exceeds 1.03 per unit, the photovoltaic generator absorbs reactive power at the configured 0.95 power factor and the hour is solved again. The line-upgrade intervention halves line impedance and raises the rating. Its annualized cost is the configured per-feeder cost multiplied by

$$
N _ { \mathrm { f e e d e r s } } = \left\lceil \frac { D _ { \mathrm { p e a k } } 1 0 0 0 } { P _ { \mathrm { f e e d e r , p e a k } } } \right\rceil ,\tag{7}
$$

with system demand in MW and feeder peak power in kW. This representativecircuit replication scales intervention costs with each zone’s peak demand. Distribution controls are evaluated on the retained system dispatch using OpenDSS steady-state snapshots[17].

Table 7 Annual electricity demand in the year-matched inputs. The 2023 demand is paired with ERA5 2023 weather; 2030 and 2035 demand are paired with the corresponding NASAderived weather. Energy is in TWh; changes are relative to 2023. Both architectures share each input. South African values are spatialized model outputs, Australian growth is mapped between consumption definitions, and New Zealand uses an island winter-growth annualization.
<table><tr><td>System</td><td>Demand 2023</td><td>Demand 2030</td><td>Change (%)</td><td>Demand 2035</td><td>Change (%)</td></tr><tr><td>MISO</td><td>640.968</td><td>868.579</td><td>35.5</td><td>961.187</td><td>50.0</td></tr><tr><td>ERCOT</td><td>446.802</td><td>983.792</td><td>120.2</td><td>1136.972</td><td>154.5</td></tr><tr><td>NYISO</td><td>146.997</td><td>160.160</td><td>9.0</td><td>170.170</td><td>15.8</td></tr><tr><td>South Africa</td><td>227.005</td><td>257.571</td><td>13.5</td><td>294.539</td><td>29.7</td></tr><tr><td>Australia</td><td>180.155</td><td>203.655</td><td>13.0</td><td>252.242</td><td>40.0</td></tr><tr><td>New Zealand</td><td>37.630</td><td>44.919</td><td>19.4</td><td>51.399</td><td>36.6</td></tr><tr><td>Mexico</td><td>345.292</td><td>410.615</td><td>18.9</td><td>460.419</td><td>33.3</td></tr><tr><td>Brazil</td><td>645.656</td><td>854.450</td><td>32.3</td><td>1003.957</td><td>55.5</td></tr></table>

## 4.9 Future-demand sources and scenario construction

Demand inputs comprise eight regional packs for each of 2023, 2030 and 2035. Forecast editions published in 2024–2026 determine the target-year scale, and each pack supplies the same demand to both architectures. Table 7 reports annual energy and growth relative to 2023. Demand year, resource-weather source and hourly-template year are recorded separately. Reported cases pair 2023 demand with ERA5 2023 resources and future demand with the matching NASA projection year. The ERA5 2023 hourly template supplies intraday shape for future resources, not their projection year.

## Hourly demand transformation.

Let $D _ { z , t } ^ { 2 0 2 3 }$ be the prepared demand in MW for zone z and hourly snapshot t, and let $E _ { z , y }$ be that zone’s target annual energy in GWh. For $y ~ \in ~ \{ 2 0 3 0 , 2 0 3 5 \}$ , the transformation is

$$
E _ { z , 2 0 2 3 } = \frac { \Delta t } { 1 0 0 0 } \sum _ { t = 1 } ^ { 8 7 6 0 } D _ { z , t } ^ { 2 0 2 3 } , \qquad g _ { z , y } = \frac { E _ { z , y } } { E _ { z , 2 0 2 3 } } , \qquad D _ { z , t } ^ { y } = g _ { z , y } D _ { z , t } ^ { 2 0 2 3 } ,\tag{8}
$$

with $\Delta t = 1$ hour. Thus $\begin{array} { r } { \sum _ { t } D _ { z , t } ^ { y } \Delta t / 1 0 0 0 = E _ { z , y } } \end{array}$ . The zonal multiplier preserves the 2023 load chronology and seasonal shape. Source-specific mappings determine $E _ { z , y } \ \mathrm { o r } .$ equivalently, $g _ { z , y } .$

## Published regional annual targets.

MISO uses the Current Trajectory annual net-load forecast in the 2026 Long-Term Load Forecast regional supplementary workbook. Its ten local resource zones are aggregated into the six model groups. ERCOT uses the 2025 ERCOT-adjusted hourly Long-Term Load Forecast; the eight weather-zone net-load columns are summed over each target year to obtain annual energy. NYISO uses the eleven zonal annual-energy values in Table I-2 of the 2026 Gold Book Baseline Forecast Tables. Mexico uses the planning-scenario net-consumption values in Table 3.5 of SENER’s PRODESEN 2024– 2038 for the nine modelled control areas. Brazil uses the reference-scenario subsystem loads in Table 2-3 of the 2026 final report of PDE 2035. Each value in average MW is multiplied by 8,760 hours and divided by 1,000 to obtain GWh for the four model subsystems.

## Australian consumption-growth mapping.

Australia uses the Step Change scenario in AEMO’s 2026 National Electricity Market Electricity Statement of Opportunities (ESOO) results workbook. Workbook values in TWh are converted to GWh. Let $C _ { z , F } ^ { \mathrm { E S O O } }$ be regional delivered consumption in GWh for financial year F. Its growth relative to 2022–23 sets

$$
g _ { z , y } = \frac { C _ { z , F _ { y } } ^ { \mathrm { E S O O } } } { C _ { z , 2 0 2 2 - 2 3 } ^ { \mathrm { E S O O } } } , \qquad E _ { z , y } = g _ { z , y } E _ { z , 2 0 2 3 } ,\tag{9}
$$

where $F _ { 2 0 3 0 } ~ = ~ 2 0 2 9 - 3 0$ and $F _ { 2 0 3 5 } ~ = ~ 2 0 3 4  – 3 5$ . We apply the ESOO deliveredconsumption growth factors to the prepared calendar-2023 TOTALDEMAND series as a modelling assumption. This transfers the published annual growth to the baseline series while retaining its chronology and dispatch-demand definition at the regional reference node.

## New Zealand winter-growth annualization.

New Zealand uses Medium Demand in Transpower’s 2026 Security of Supply Assessment (SOSA) supplementary data. Let $\bar { W } _ { i , y } ^ { \mathrm { S O S A } }$ be target-year April–September grid-exit-point oftake in GWh for island i. The corresponding model baseline and multiplier are

$$
W _ { i , 2 0 2 3 } = \frac { \Delta t } { 1 0 0 0 } \sum _ { z \in i } \sum _ { t \in \mathrm { A p r - S e p } } D _ { z , t } ^ { 2 0 2 3 } , \qquad g _ { i , y } = \frac { W _ { i , y } ^ { \mathrm { S O S A } } } { W _ { i , 2 0 2 3 } } .\tag{10}
$$

All zones on an island receive $D _ { z , t } ^ { y } = g _ { i , y } D _ { z , t } ^ { 2 0 2 3 }$ for all 8,760 hours. North Island comprises UNI, CNI and LNI; South Island comprises USI and LSI. The island wintergrowth factors are thereby annualized while retaining the within-island spatial shares and each zone’s hourly pattern.

## South African spatialized reference demand.

South Africa uses the University of Cape Town ESRG Hourly Demand Model v2025 by Province and its spatialized IRP REF series, with exports excluded. The source’s hourly model outputs are summed over each target year separately for the ten supply areas to obtain $E _ { z , y } .$ Equation 8 maps those annual targets onto the corresponding prepared 2023 templates. Both the baseline and future inputs retain the source’s status as spatially allocated model outputs informed by national demand projections and regional calibration data.

## Fixed assumptions and input checks.

Demand totals retain the source assumptions about electrification, industrial activity and large loads, including NYISO’s large-load tables, AEMO’s data-centre outlook and Transpower’s DataGrid step-load assumptions. The 2023 evaluation uses ERA5 2023 resource profiles; the future packs use the corresponding target-year NASA profiles described below. All cases retain the inherited fleet, baseline policy and $\mathrm { A T B / P y P S A }$ technology-cost basis[36, 37]. The primary evaluation applies its fixed carbon-budget overlay independently of this policy mode. Each target year is a singleperiod brownfield problem, and representative-feeder replication follows the scenario peak.

South African 2030 cases apply commissioning-year eligibility to the oficial transmission projects in the NTCSA Transmission Development Plan. A project is eligible when its documented in-service year is at or before 2030; an indivisible project with a date range uses the final year. Other region–year routes retain their inherited project permissions. Generic candidate-circuit expansion follows the declared circuit bounds.

Preparation records source editions, scenarios, extraction coordinates, geographic mappings, unit conversions and file hashes. Demand checks confirm 8,760 finite nonnegative values per zone, agreement with annual-energy targets and identical inputs across architectures.

## 4.10 Future-weather inputs and hourly reconstruction

For the combined input library, we prepared 16 future-weather input packs covering eight systems in 2030 and 2035, comprising 116 annual node profiles. The primary cohort uses nine of these future system–year packs. NASA NEX-GDDP-CMIP6 v2.0 supplies daily, bias-corrected projections at $0 . 2 5 ^ { \circ }$ resolution[47, 48]. The selected realization is MPI-ESM1-2-HR, member r1i1p1f1, under SSP2-4.5. At each of the 58 representative locations, we selected the nearest grid cell with complete records for all four variables, comprising minimum and maximum temperature (tasmin, tasmax; K), daily-mean shortwave irradiance (rsds; $\mathrm { W ~ m ^ { - 2 } } )$ and near-surface wind speed (sfcWind; $\mathrm { ~ m ~ s ~ } ^ { - 1 } )$

Target-year daily values are combined with the corresponding month–day ERA5 2023 hourly patterns[44, 45]. Each retrieval includes the preceding 31 December and following 1 January; their templates are 2022-12-31 and 2024-01-01. Both projection years have 365 days. Complete UTC days are reconstructed before selecting 8,760 hours using each node’s fixed model-clock ofset. Solar geometry uses target-year dates and interval midpoints. The profiles retain the 2023 hourly index and inherited targetyear demand values (Table 9).

Let N denote NASA daily values, E the ERA5 template, d a UTC day and h an hour. After converting temperature to $^ \circ \mathrm { C }$ , hourly temperature and wind are

$$
T _ { d , h } ^ { y } = T _ { d } ^ { N , \mathrm { m i n } } + \left( T _ { d } ^ { N , \mathrm { m a x } } - T _ { d } ^ { N , \mathrm { m i n } } \right) \frac { T _ { d , h } ^ { E } - \mathrm { m i n } _ { j } T _ { d , j } ^ { E } } { \mathrm { m a x } _ { j } T _ { d , j } ^ { E } - \mathrm { m i n } _ { j } T _ { d , j } ^ { E } } ,\tag{11}
$$

$$
u _ { \ell , d , h } ^ { y } = u _ { \ell , d , h } ^ { E } \frac { \overline { { u } } _ { d } ^ { N } } { \overline { { u } } _ { 1 0 , d } ^ { E } } , \qquad \ell \in \{ 1 0 , 1 0 0 \} \mathrm { ~ m } ,\tag{12}
$$

Table 8 Mean resource capacity factors in the reported input packs. Values are unweighted means of annual node capacity factors in each system. The 2023 packs use ERA5; the 2030 and 2035 packs use NASA NEX-GDDP-CMIP6 fields reconstructed as described above.
<table><tr><td></td><td colspan="3">Wind</td><td colspan="3">Solar PV</td></tr><tr><td>System</td><td>2023</td><td>2030</td><td>2035</td><td>2023</td><td>2030</td><td>2035</td></tr><tr><td>MISO</td><td>0.321</td><td>0.255</td><td>0.271</td><td>0.168</td><td>0.202</td><td>0.204</td></tr><tr><td>ERCOT</td><td>0.339</td><td>0.256</td><td>0.245</td><td>0.179</td><td>0.211</td><td>0.210</td></tr><tr><td>NYISO</td><td>0.230</td><td>0.422</td><td>0.441</td><td>0.149</td><td>0.184</td><td>0.189</td></tr><tr><td>South Africa</td><td>0.252</td><td>0.228</td><td>0.218</td><td>0.201</td><td>0.205</td><td>0.206</td></tr><tr><td>Australia (NEM)</td><td>0.249</td><td>0.439</td><td>0.449</td><td>0.170</td><td>0.180</td><td>0.180</td></tr><tr><td>New Zealand</td><td>0.398</td><td>0.596</td><td>0.594</td><td>0.159</td><td>0.176</td><td>0.178</td></tr><tr><td>Mexico</td><td>0.075</td><td>0.198</td><td>0.195</td><td>0.203</td><td>0.212</td><td>0.211</td></tr><tr><td>Brazil</td><td>0.140</td><td>0.168</td><td>0.195</td><td>0.163</td><td>0.170</td><td>0.167</td></tr></table>

where overbars denote daily means. The common wind multiplier retains the template’s hourly variation and vertical wind-speed ratio.

Daily solar energy is allocated proportionally to the ERA5 shortwave pattern, aligned from preceding-hour averages to interval starts. Allocation is bounded by extraterrestrial horizontal irradiance, with remaining energy redistributed across unsaturated daylight hours to satisfy

$$
\sum _ { h = 1 } ^ { 2 4 } G _ { d , h } ^ { y } \Delta t = 2 4 \mathrm { h } \overline { { R } } _ { d } ^ { N } , \qquad 0 \le G _ { d , h } ^ { y } \le I _ { 0 , d , h } \operatorname* { m a x } ( \cos \theta _ { d , h } , 0 ) ,\tag{13}
$$

where $\Delta t = 1$ h, G is global-horizontal irradiance, $\overline { { R } } ^ { N }$ is rsds, $I _ { 0 }$ is extraterrestrial normal irradiance and θ is midpoint solar zenith. The pvlib Erbs decomposition supplies direct-normal irradiance, capped at $I _ { 0 } ;$ difuse-horizontal irradiance is recomputed as G − DNI max(cos θ, 0).

We apply PVWatts v8 through PySAM (NREL-PySAM 7.1.1.post1 in the archived generation environment)[42, 43] with a 1 MW DC fixed array, DC/AC ratio one, equator-facing orientation, tilt equal to absolute latitude and 14% losses as study settings. Synthetic 100 m wind supplies its wind-speed input. Photovoltaic capacity factors are AC output divided by 1 MW DC. Wind capacity factors are the interpolated NREL 4 MW, 150 m-rotor reference-curve output at synthetic 100 m wind, divided by 4 MW[46]. Checks verify daily temperature extrema, solar energy and near-surface mean wind, complete hourly profiles and preserved demand. ERA5 and NEX-GDDP-CMIP6 difer in their wind and irradiance climatology at the study locations, so mean capacity factors difer between the 2023 and future-year packs (Table 8). Diferences between 2023 and future-year outcomes therefore combine demand growth, projected climate and the change of data product. Source, processing-code and output SHA-256 hashes accompany each pack.

Table 9 Weather inputs and time coordinates. Future resource profiles combine the indicated NASA daily projection with ERA5 2023 intraday patterns. Each row covers eight systems and 58 nodes; every profile contains 8,760 hours. The two future rows comprise 16 prepared input packs and 116 annual profiles.
<table><tr><td>Demand year</td><td>Weather source</td><td>Hourly template</td><td>Hourly index</td></tr><tr><td>2023</td><td>ERA5 2023 reanalysis</td><td>2023</td><td>2023</td></tr><tr><td>2030</td><td>NASA MPI-ESM1-2-HR</td><td>ERA5 2023</td><td>2023</td></tr><tr><td>2035</td><td>SSP2-4.5, 2030 projection NASA MPI-ESM1-2-HR SSP2-4.5, 2035 projection</td><td>ERA5 2023</td><td>2023</td></tr></table>

## 4.11 Agent architectures, models and evidence retention

The shared action interface includes technology permissions, global and node-level capacity caps, battery duration, charging and discharging fractions, distribution reinforcement, reactive control, battery placement, candidate-circuit bounds and project permissions. These variables constrain the optimizer. Demand, weather, costs, existing assets, network assumptions, policy and the annual carbon budget are immutable through the agent interface. Code reconstructs fixed roles, tools and acceptance requirements independently of editable guidance.

The single agent maintains a continuous conversation and free-form strategy notes. At each turn, it chooses among state inspection, design modification, system simulation, feeder rechecking, note revision and termination. The agent determines the order of these actions, subject to fixed tool preconditions. The current notes are reinserted into the system prompt at every turn, while the preceding conversation remains available.

At the start of each multi-agent cycle, four specialists in system architecture, renewable generation, storage and firm generation receive the same shared state and advisory procedure-graph guidance. They independently propose changes without access to one another’s proposals for that cycle. The orchestrator receives all four proposals and selects one experiment, which may combine proposed changes or retain the current design. A valid selection is evaluated by PyPSA, followed by OpenDSS if the system solve succeeds. The evaluator interprets the cycle record, and the critic reviews that record together with the evaluation. The procedure refiner then updates the advisory graph, the compressor records a structured summary, and the convergence judge decides whether to continue or stop. Roles use separate model-call contexts and exchange information through explicitly supplied records. Across cycles, the framework retains the current design, latest system and feeder evidence, accumulated cycle summaries and advisory graph. The judge additionally receives a code-computed outcome trajectory.

The procedure graph contains planning steps with directed edges, conditions, guidance and pitfalls, inspired by procedural memory[21]. It begins with a six-node expert skeleton and is limited to thirty nodes and sixty edges with bounded text fields.

Structurally valid edits are applied and logged. Their contents provide advice without altering tools or numerical requirements.

Every successful committed system evaluation triggers an OpenDSS check. A distribution-only edit invalidates feeder evidence while preserving a matching system dispatch, whereas a system-design edit invalidates both. The convergence judge can request up to two system-only verifications of hypothetical alternatives within a judge session. Their outputs are logged and used for that session’s decision to continue or stop. They are not included in subsequent-cycle prompts and do not establish physical acceptance. The single agent requests termination through finish design, and the multi-agent judge issues its verdict through judge convergence. These stopping decisions are subject to current-evidence requirements, and the configured search ceilings limit further execution. Stopping does not itself establish physical acceptance. The selected accepted candidate and the last attempt are recorded separately.

Provider adapters translate a common action vocabulary to OpenAI Responses and Anthropic Messages. The reported deployments are GPT-6.1 Sol (gpt-6.1-sol) and Claude Opus 5.5 (claude-opus-5-5)[49]. Within each multi-agent run, all roles use the same model deployment, with specialization provided by role-specific instructions and inputs. Hosted models use their provider’s high efort setting, with the recorded inference configurations in Appendix B. We use the pretrained deployments without updating model weights or constructing a task-specific training set. Adaptation is confined to the recorded conversation, strategy notes, advisory procedure graph and proposed actions.

The primary and completed hosted-weather campaigns permit 120 single-agent turns, forty committed simulation requests and 120 multi-agent cycles for both deployments. The default response ceiling is 64,000 tokens, with smaller role-specific ceilings retained. Each multi-agent cycle contains several role-specific responses. The earlier base-year trajectories retain their recorded execution settings in Appendix B.

The runtimes record exact system-result reuse separately from numerical optimization. Eligible reuse requires matching system design, scenario, solver identity and solver settings, and preserves the archived optimal dispatch and objective evidence. Distribution-only controls can be changed while reusing a matching system dispatch, after which the requested feeder evaluation is performed. During the primary campaign, a separately recorded execution revision extended this matching to systemequivalent designs and retained the latest committed dispatch independently of the recent trial cache. The source and deployment manifests identify the afected trajectories. Logical evaluation requests, actual numerical solves and cache hits are distinct measures of efort. Reuse does not change the numerical acceptance requirements, but runtime and trajectory comparisons retain the execution-revision distinction. This revision applies from the interrupted NYISO 2035 GPT-6.1 Sol multi-agent verification onward, covering that trajectory and the seven subsequent GPT-6.1 Sol multi-agent trajectories.

## 4.12 Outcome extraction and comparison

The primary reporting manifest selects forty completed carbon-constrained trajectories. A separate reference manifest retains the ninety-six trajectories from the original loose-cap evaluation. For each carbon trajectory, extraction reads its retained committed design, system result, distribution evidence, carbon-budget audit and numerical bound. The reported outcome is the lowest-cost accepted committed candidate selected by the retention rule in Eq. 5. Physical and emissions checks are distinguished from the agent’s stopping decision. Source manifests preserve the result paths, scenario and implementation identities, and links to numerical and model-response records.

Search histories, received-response tokens and process durations are reconstructed separately from the annual power-system objective (Section 4.13).

Reference comparisons match system, demand year, resource profiles, deployment and architecture. Input-file hashes match for all forty pairs, with the source-derived carbon ceiling applied through the recorded scenario overlay. The reported contrasts describe observed trajectories, and any diferences in provider configuration, recovery history and result-reuse revision remain attached to the pair. Numerical bounds are conditional on the optimization domain of each selected design. The comparison contains one trajectory per condition and does not establish a distribution of performance across repeated searches.

## 4.13 Reconstruction of search efort and token usage

We analyse the forty primary trajectories and the ninety-six loose-cap reference trajectories separately. A committed assessment is the chronological evaluation of a candidate after a system-simulation request or an additional feeder-only check; repeated designs and rechecks remain separate assessments. For the reference cohort, candidate reconstruction reproduces the hashes of all 382 assessed designs and links each archived selected design and objective to its first recorded assessment. The primary counts use the completed response ledgers and recorded committed and hypothetical system-evaluation requests. Native single-agent turns and multi-agent cycles are retained in the evidence; common comparisons use assessment order. A rejected improvement is an assessment whose recorded all-in objective is lower than an already accepted incumbent beyond the declared numerical comparison tolerance $( 1 0 ^ { - 6 }$ in the recorded billion-per-year objective units), but which fails the acceptance gate. Such an objective does not represent a feasible saving. Uncommitted system-only probes are tabulated separately because they lack feeder validation.

We deduplicate received responses by provider and response identifier and reconcile 764 received responses in the primary cohort and 2,257 in the reference cohort with their archived ledgers and journals. Input tokens are normalized as uncached input plus cache reads plus cache writes; repeated context is counted each time it is supplied. Output tokens include the reported reasoning subset, so reasoning is not added again to the total. Tokens describe the complete selected search histories, including inherited responses exactly once within each retained trajectory. Responses from separate discarded attempts are excluded from these trajectory totals. Provider tokenizers and usage-reporting conventions difer.

Matched comparisons pair single- and multi-agent trajectories within deployment, system and year. Aggregate ratios divide sums across a cohort, whereas median paired ratios summarize within-scenario quotients. Reference-cohort process-duration summaries use 92 trajectories and 44 pairs; four resumed reference histories lack full uninterrupted duration records. The primary campaign is not treated as a controlled wall-runtime benchmark because pauses, recovery and the result-reuse revision afect execution histories. Process duration covers the recorded workflow, and logged solver time is reported separately. Evaluation requests count logical calls, including exact-result reuse. All summaries are descriptive, with one trajectory per deployment–architecture–scenario cell.

## 4.14 Comparison with published expert plans

We compare the retained carbon-constrained designs for Brazil and South Africa in 2035 with archived institutional planning outputs, without new inference or simulation. These two systems have explicit archived 2035 capacity mappings for the categories in Table 3. All four accepted deployment–architecture trajectories per system are included. Capacity values are read from each retained candidate’s totalcapacity record and checked against the final reported output. Minima and maxima span the four observed outcomes rather than statistical uncertainty. For capacity category k, the descriptive diference is $\Delta C _ { k } = C _ { k } ^ { \mathrm { a g e n t } } - C _ { k } ^ { \mathrm { p l a n } }$ in GW.

Brazil uses the PDE 2035 chapter-three workbook, Figura 3-25, December 2035 row 8, with wind from E8, centralized solar from F8 and batteries from I8. MMGD capacity G8 is kept separate. South Africa uses the IRP 2025 user-friendly edition, Table 1, printed page 31 (PDF page 41). The wind and Solar PV columns are reconstructed as the end-2025 current base plus scheduled 2026–2035 additions. Separate distributed-generation, CSP and hybrid columns are excluded. The model inherits its 2023 fleet, whereas the South African plan’s capacity base includes assets available by end-2025.

Each comparison retains the published capacity category. Model solar is a zonal allocation without a separate future distributed-generation rollout, so the solar diferences cannot be interpreted as national photovoltaic surpluses. Demand, retirements, resources, costs, permitted technologies and delivery schedules retain the respective study assumptions. PDE 2035 supplies Brazilian demand and the numerical emissions ceiling adopted here; IRP 2025 supplies the South African emissions pathway, while South African demand uses ESRG. The comparison thus shares planning inputs with its institutional references and is not independent validation against withheld targets. The carbon-budget mappings, including Brazil’s $\mathrm { C O _ { 2 } } \mathrm { - e q u i v a l e n t }$ proxy, are specified in Section 4.5. The comparison archive retains all eight candidate identities, capacities, acceptance records, source hashes, workbook cells or page locators and calculations.

Data availability. Demand and fleet sources include EIA, ESRG, Eskom, AEMO, the New Zealand Electricity Authority, Transpower, CENACE, SENER, ONS and ANEEL. ERA5 provides the baseline weather, and NASA NEX-GDDP-CMIP6 v2.0 provides the future daily climate fields[44, 45, 47, 48]. The country-indexed source catalogue in Appendix G provides direct document links. The study archive retains source snapshots, geographic mappings, forecast targets, hourly resource profiles and input hashes. The manuscript evidence links each table and figure to its selected planning outcomes and comparison records. The carbon-constrained expert-plan comparison retains the eight Brazilian and South African 2035 candidates, document editions, workbook cells, category mappings and source hashes. The search-and-token archive contains the 96 reference assessment histories and the separately identified carboncohort response and evaluation records, with normalized response counts, token totals, role attribution, matched comparisons and source hashes. Carbon scenario records preserve the source-derived budgets, conversion assumptions and acceptance audits.

Code availability. The planning and analysis code is publicly available at https://github.com/sixticket/grid-agent. The repository contains the system simulator, regional scenario preparation, NASA-to-hourly reconstruction, model adapters, launchers and dependency records. The manuscript package includes the 96-trajectory reporting manifest drawn from the frozen Opus base-year and hosted-weather evidence, read-only committed-history extractors, table and figure generators, the frozen-evidence search-and-token table builder, and processing provenance for the 16 future-weather packs. A separate forty-trajectory manifest, fixed carbon-budget overlay, source catalogue, execution-revision manifest and carbon figure generators accompany the primary evaluation.

Acknowledgements. This work was supported by the U.S. National Science Foundation (NSF) through CAREER Award No. ECCS-2543177, by the U.S. Department of Energy through Distinguished Early Career Award No. DE-NE0009635, and by the U.S. Nuclear Regulatory Commission under Grant No. 31310024M0041.

Computing resources on Delta and DeltaAI at the National Center for Supercomputing Applications at the University of Illinois Urbana-Champaign were provided through two ACCESS allocations: CIS262451 (AI Agents for Safety-Critical Energy Systems) and CIS261154 (Deployable Energy-Eficient Agentic Foundation Model with Edge AI for Energy Systems). The State of Illinois and NSF support Delta (OAC-2005572) and DeltaAI (OAC-2320345). NSF supports the ACCESS program[50] through grants 2138259, 2138286, 2138307, 2137603 and 2138296.

We thank the organisations whose public data made the regional package possible. The U.S. Energy Information Administration provided hourly demand (EIA-930) and the generator inventory (EIA-860). The New York Independent System Operator, the Midcontinent Independent System Operator and the Electric Reliability Council of Texas published the reliability, operating-study, resource-adequacy and transmission-planning documents from which zonal constraints and planned projects were taken, and the Public Utility Commission of Texas the 765 kV import-path approvals. The National Transmission Company South Africa published the Transmission Development Plan 2024 and Eskom the Integrated Report 2023 and station data. The Energy Systems Research Group at the University of Cape Town (G. Ireland and colleagues) published the spatialized South African demand model, and Meridian Economics and the PyPSA-ZA authors (J. H¨orsch and J. Calitz) the open PyPSA-RSA model whose topology method we replicated. The Australian Energy Market Operator published the NEMWEB demand data, the registration lists, the interconnector capabilities and the Integrated System Plan; the New Zealand Electricity Authority the EMI datasets and Transpower the Transmission Planning Report; Mexico’s Centro Nacional de Control de Energ´ıa the real-demand reports and the Secretar´ıa de Energ´ıa the PRODESEN; Brazil’s Operador Nacional do Sistema El´etrico the open data portal and the PEN, and ANEEL the SIGA plant register. The World Resources

Institute provided the Global Power Plant Database and geoBoundaries the administrative boundaries. ERA5 reanalysis was produced by ECMWF under the Copernicus Climate Change Service and served through the Open-Meteo archive. NASA provided the NEX-GDDP-CMIP6 projections, including the MPI-ESM1-2-HR simulation used here. The National Laboratory of the Rockies (formerly NREL) provided the Annual Technology Baseline, the reference turbine power curves and the System Advisor Model through PySAM, and the PyPSA project the technology-data repository. We used PyPSA, linopy, OpenDSS through OpenDSSDirect.py, HiGHS and Gurobi under an academic licence.

Author contributions. Y.P.L. conceived the study, developed the framework, prepared the regional data and performed the experiments. S.C. and S.B.A. supervised the work. All authors wrote and revised the manuscript.

Competing interests. The authors declare no competing interests.

## Appendix A Carbon-constrained outcomes and matched reference evidence

The following figures retain the forty matched cost and emissions comparisons and candidate-specific numerical bounds. Main-text Fig. 3 combines grid additions and emissions for the latest evaluated year in each system. The full-precision forty-row ledger and ten-case grid-and-emissions atlas are provided in the evidence archive. The original 1,000 Mt reference ceiling is nonbinding in every matched reference outcome. Configuration diferences are retained with the pairs, so these figures describe matched retrospective comparisons rather than a controlled carbon-only ablation.

![](images/b15cac607ea1adc4ecfe219bdb0b1a5eab820f1c316a08586885b922be138e38.jpg)  
Open triangles identify the two Opus 2023 pairs with diferent output-token ceilings.

Fig. A1 Carbon-constrained outcomes and matched loose-cap references. Comparisons pair each primary trajectory with the archived trajectory having the same system, year, deployment and architecture under the original 1,000 Mt annual ceiling. Twenty-eight reference designs exceed the source-derived ceilings. Costs include annualized representative-feeder interventions, and emissions are direct operational CO<sub>2</sub> within the model boundary. Open triangles identify the two 2023 Opus pairs with distinct output ceilings and provider configurations. The points describe observed trajectories and are not repeated samples.

![](images/8d906f6aba553a554bd005a5064fe1099244189648cb104c8ad107f8850dfaa5.jpg)  
Fig. A2 Candidate-specific numerical bounds for the selected carbon-constrained outcomes. Each interval spans the numerical solver lower bound and achieved system objective for the selected candidate’s fixed optimization domain, adding the selected distribution cost at both end points. These intervals describe numerical uncertainty within each candidate-specific problem. They are not confidence intervals or bounds on the full agent search space. All forty selected outcomes pass the physical and carbon-budget checks.

## Appendix B Execution configurations

The primary evaluation contains forty carbon-constrained trajectories. Both deployments use a 64,000-token default output ceiling, high efort, 120 single-agent turns, forty committed single-agent simulation requests and 120 multi-agent cycles, with the recorded role-specific response ceilings. The scenario manifests retain the fixed budgets and the completed trajectory identities. Pause, recovery and system-result reuse records remain attached to the afected trajectories. A separate reference archive contains the ninety-six GPT-6.1 Sol and Claude Opus 5.5 trajectories described below. The sixteen Opus trajectories with 2023 demand and ERA5 2023 weather come from the earlier base-year campaign (frozen on 29 September 2026 at 23:35:46 UTC). The remaining eighty trajectories retain the settings of the later hosted campaign. Table B1 records the earlier Opus configuration.

Table B1 Execution settings for the sixteen reported Opus baseline trajectories. Evaluation counts include requests served by an exact matching cached result.
<table><tr><td>Setting</td><td>Claude Opus 5.5</td></tr><tr><td>Inference</td><td>Anthropic Messages API</td></tr><tr><td>Completed trajectories</td><td>16 for demand year 2023</td></tr><tr><td>Single-agent ceiling</td><td>120 turns; 40 committed evaluations</td></tr><tr><td>Multi-agent ceiling</td><td>120 cycles</td></tr><tr><td>Default output ceiling</td><td>16,000 tokens</td></tr><tr><td>Generation</td><td>Adaptive thinking; high effort; standard-only tier</td></tr><tr><td>Numerical solver</td><td>Gurobi 13.0.3; 18 threads; 0.1% requested gap</td></tr><tr><td>Planning horizon</td><td>8,760 hourly snapshots</td></tr><tr><td>Selection</td><td>Lowest-cost physically accepted committed candidate</td></tr></table>

GPT-6.1 Sol links successive Responses API requests through previous response id. Opus uses adaptive thinking, high efort and a five-minute prompt cache. Its revised adapter uses the thinking-binding-controls-2026-08-01 header and prefix mismatch behavior=drop block so that strategy-note edits can update the system prompt under the preserved-thinking protocol[51]. Original response blocks remain in saved history; the server records exclusions of prefixincompatible thinking blocks in input transformations. Trajectory records identify the applied adapter, retained response history and provider-reported thinking transformations.

## Completed hosted NASA-weather campaign.

A separate completed campaign reports eighty hosted trajectories, with its final case completed on 3 October 2026 at 06:08:08 UTC (Table B2). It includes sixty-four cases with the prepared MPI-ESM1-2-HR SSP2-4.5 weather inputs for 2030 and 2035, and sixteen GPT-6.1 Sol controls with 2023 demand and ERA5 2023 weather. Both deployments use a 64,000-token default output ceiling, with 12,000-token critic requests; the single-agent ceiling is 120 turns and forty committed evaluations, and the multi-agent ceiling is 120 cycles. They share the agent workflow, action interface and physical acceptance rules, with the 8,760-hour horizon and numerical solver settings in Table B1. GPT-6.1 Sol uses high efort and the standard tier; Opus uses the revised thinking protocol above with high efort and the standard-only tier. Original execution identities and complete response ledgers remain attached to each trajectory. Three resumed trajectories have no uninterrupted wall-runtime estimate; their retained numerical evidence and unique received responses are counted once.

Table B2 Completed cases in the new hosted weather campaign. Each listed demand year contains eight systems and two agent architectures. The sixteen earlier Opus base-year trajectories are reported separately.
<table><tr><td>Model</td><td>Demand and weather inputs</td><td>Completed cases</td></tr><tr><td>GPT-6.1 Sol</td><td>2023 demand with ERA5 2023 weather; 2030 and 2035 demand with the corresponding NASA-derived</td><td>48</td></tr><tr><td>Claude Opus 5.5</td><td>weather 2030 and 2035 demand with the corresponding NASA-derived weather</td><td>32</td></tr></table>

## Appendix C Planning outcomes under the loose-cap reference

This section preserves the original ninety-six trajectories under a common annual CO<sub>2</sub> ceiling of 1,000 Mt. Forty trajectories match the primary cohort by system, year, deployment and architecture; the remaining fifty-six extend regional or temporal coverage. All numerical results, maps and search histories in this section belong to the reference evaluation.

## C.1 Initial designs and retained reference candidates

In the loose-cap reference evaluation, the engine’s default design permits all extendable technologies, sets capacity bounds to their maxima and uses four-hour batteries without feeder interventions. This default is the least-cost plan that the optimizer returns without agent restrictions under the original carbon ceiling. It passes the gate in 17 of the 24 system–year problems and fails only the distribution component in seven. The 28 trajectories for those seven problems comprise twelve South African, eight ERCOT, four Mexican and four NYISO trajectories; all subsequently retain an accepted candidate. In 46 of the 56 trajectories that retain the default design, agents test 99 alternatives before stopping, comprising 60 hypothetical system-only checks and 39 committed designs; none is cheaper. The remaining 12 trajectories in the passing problems select modified designs that are up to 0.483% cheaper. Twelve initial rejections occur in the 2023 cohort and eight in each future-year cohort. The records show transitions from rejected to accepted planning candidates in multiple systems.

## C.2 Reference experiment coverage and retained planning outcomes

The three-year GPT-6.1 Sol and Claude Opus 5.5 coverage contains 96 trajectories across eight systems, three matching demand/weather years, two deployments and single- and multi-agent architectures. All 96 trajectories are reported in Table C3. The Opus 2023 results come from the earlier base-year campaign; deployment versions and recorded inference settings remain distinct.

Every reported trajectory retains a physically accepted committed candidate. Table C4 reports its selected total annual cost for every system, deployment, year and architecture. These outcomes demonstrate operation of the framework across the full evaluated scope. Detailed costs, candidate-specific solver bounds and efort records remain in Appendix D.

The geographic portfolios in Fig. C1 show the lowest-cost accepted 2035 design observed for each system across both deployments and architectures, with tied costs explicitly identified. The Opus ERCOT repair in Fig. C2 is a separate process example. These spatial outputs complement the complete results in Tables C3 and C4.

Table C3 Loose-cap reference coverage and physical acceptance. Each cohort contains eight systems and both architectures. The complete GPT-6.1 Sol–Opus grid contains 96 trajectories. Accepted counts refer to trajectories with a retained committed candidate passing the implemented system and representative-feeder gate. Baseline resources are ERA5 2023; future resources are reconstructed for the indicated NASA projection year.
<table><tr><td>Deployment</td><td>Demand / weather</td><td>Single</td><td>Multi</td><td>Accepted</td><td>Total</td></tr><tr><td>GPT-6.1 Sol</td><td>2023 / ERA5</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>GPT-6.1 Sol</td><td>2030 / NASA</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>GPT-6.1 Sol</td><td>2035 / NASA</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>Opus 5.5</td><td>2023 / ERA5</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>Opus 5.5</td><td>2030 / NASA</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>Opus 5.5</td><td>2035 / NASA</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>Total</td><td>Year matched</td><td>48</td><td>48</td><td>96</td><td>96</td></tr></table>

Table C4 Selected planning outcomes for all 96 loose-cap reference trajectories. Entries are total annual power-system costs in billion USD per year, including annualized representative-feeder interventions, rounded to three decimals. S and M denote single- and multi-agent architectures. The Accepted column counts retained physically accepted trajectories within each system–deployment row. Full numerical precision, candidate-specific solver bounds and evaluation efort are retained in Appendix D.
<table><tr><td rowspan="2">System</td><td rowspan="2">Deployment</td><td colspan="2">2023</td><td colspan="2">2030</td><td colspan="2">2035</td><td rowspan="2">Accepted</td></tr><tr><td>S</td><td>M</td><td>S</td><td>M</td><td>S</td><td>M</td></tr><tr><td rowspan="2">MISO</td><td>GPT-6.1 Sol</td><td>28.293</td><td>28.293</td><td>46.001</td><td>46.146</td><td>52.960</td><td>53.217</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>28.293</td><td>28.293</td><td>46.004</td><td>46.004</td><td>52.961</td><td>53.217</td><td>6/6</td></tr><tr><td rowspan="2">ERCOT</td><td>GPT-6.1 Sol</td><td>18.225</td><td>18.225</td><td>68.058</td><td>68.076</td><td>82.980</td><td>82.958</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>18.225</td><td>18.225</td><td>68.407</td><td>68.408</td><td>83.763</td><td>83.151</td><td>6/6</td></tr><tr><td rowspan="2">NYISO</td><td>GPT-6.1 Sol</td><td>8.030</td><td>8.030</td><td>6.478</td><td>6.478</td><td>6.830</td><td>6.830</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>8.030</td><td>8.030</td><td>6.478</td><td>6.478</td><td>6.830</td><td>6.830</td><td>6/6</td></tr><tr><td rowspan="2">South Africa</td><td>GPT-6.1 Sol</td><td>10.647</td><td>10.648</td><td>12.018</td><td>12.019</td><td>14.098</td><td>14.098</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>10.652</td><td>10.656</td><td>12.052</td><td>12.020</td><td>14.202</td><td>14.101</td><td>6/6</td></tr><tr><td rowspan="2">Australia (NEM)</td><td>GPT-6.1 Sol</td><td>7.053</td><td>7.053</td><td>6.861</td><td>6.861</td><td>8.853</td><td>8.854</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>7.053</td><td>7.053</td><td>6.861</td><td>6.861</td><td>8.854</td><td>8.854</td><td>6/6</td></tr><tr><td rowspan="2">New Zealand</td><td>GPT-6.1 Sol</td><td>0.569</td><td>0.569</td><td>0.647</td><td>0.647</td><td>0.800</td><td>0.800</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>0.569</td><td>0.569</td><td>0.647</td><td>0.647</td><td>0.800</td><td>0.800</td><td>6/6</td></tr><tr><td rowspan="2">Mexico</td><td>GPT-6.1 Sol</td><td>19.552</td><td>19.552</td><td>22.258</td><td>22.258</td><td>26.409</td><td>26.412</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>19.553</td><td>19.555</td><td>22.258</td><td>22.258</td><td>26.409</td><td>26.412</td><td>6/6</td></tr><tr><td rowspan="2">Brazil</td><td>GPT-6.1 Sol</td><td>15.870</td><td>15.870</td><td>24.655</td><td>24.655</td><td>33.736</td><td>33.736</td><td>6/6</td></tr><tr><td>Opus 5.5</td><td>15.870</td><td>15.870</td><td>24.655</td><td>24.655</td><td>33.736</td><td>33.736</td><td>6/6</td></tr></table>

## C.3 Accepted planning portfolios retain their geographic context

The same planning loop produces accepted investment-and-dispatch candidates in all eight regional systems. Tables C3 and C4 retain all 96 trajectories. For the geographic views, we compare the four completed 2035 trajectories in each system—GPT-6.1 Sol and Opus 5.5, each with single- and multi-agent architectures—using identical regional demand, resource, inventory and network inputs. Figure C1 maps the lowest observed selected total annual cost among their physically accepted candidates. GPT-6.1 Sol single-agent supplies the displayed MISO, South Africa, Australia and Mexico designs, and GPT-6.1 Sol multi-agent supplies ERCOT. In NYISO, New Zealand and Brazil, all four trajectories have the same reported minimum cost; an Opus multi-agent portfolio is retained as a stable representative of each tie. Additional inter-area transfer capacity is selected in MISO (1.00 GW), NYISO (4.00 GW), South Africa (1.20 GW) and Australia (4.61 GW). These values sum selected corridor increments rather than net import capabilities. Appendix E provides enlarged maps and a complete zonal installation ledger for each displayed design.

Across these eight selected designs, new generation ranges from 1.43 GW in New Zealand to 179.07 GW in Brazil. The geographic pattern reflects the declared zonal demand, resource realization, inherited fleet and investment bounds. It demonstrates that the framework returns explicit regional planning outputs across heterogeneous input packs. New assets are represented by zonal allocations.

Opus multi-agent designs represent four-way cost ties in NYISO, New Zealand and Brazil.

## C.4 Feeder evidence changes the accepted planning output

A separate Opus 5.5 multi-agent ERCOT 2035 trajectory illustrates how the framework links regional capacity choices to a physical check outside the annual system objective (Fig. C2). Its initial Far West representative feeder has two voltage-violation hours and thirteen thermal-overload hours. The annual minimum voltage is 0.948 p.u., below the 0.95 p.u. lower limit, and the second line reaches 105.8% of its 400 A rating. This candidate fails the distribution gate.

The selected accepted candidate combines a changed portfolio and battery operation with the Far West line-upgrade decision. The upgrade increases both representative-line ratings from 400 to 600 A and halves their impedances under the declared intervention model. The final Far West feeder has no voltage violations, thermal overloads or nonconverged hours; its annual voltage range is 0.974–1.015 p.u., and the second line’s annual peak current is 65.2% of its upgraded rating. All eight ERCOT representative feeders pass.

The committed regional result contains 89.662 GW of new solar, 79.519 GW of new nuclear and 35.128 GW of new battery power with 210.770 GWh of battery energy. It selects no additional inter-area transmission capacity. The annualized distribution intervention is USD 183.04 million. These outcomes connect zonal additions, a named feeder intervention and the resulting acceptance record.

<table><tr><td>New power (GW)</td></tr><tr><td>o O</td></tr><tr><td>O 1 10 50</td></tr></table>

a MISO  
![](images/4bc08a4ad0223dc40c9779c8e0213b3c73e572db3ad43cec33a3e9839027e474.jpg)  
\$52.960 bn/yr

<table><tr><td>Added power (GW)</td></tr><tr><td>Wind 40.26</td></tr><tr><td>Solar 70.68</td></tr><tr><td>Nuclear 15.49</td></tr><tr><td>Battery 19.78</td></tr></table>

b ERCOT  
![](images/6fa35178061af68f35851dd650d0479a757db30f4faca3e183ad88c02d15d84f.jpg)

\$82.958 bn/yr
<table><tr><td>Added power (GW)</td></tr><tr><td>Wind 0.00 Solar 93.65</td></tr><tr><td></td></tr><tr><td>Nuclear 79.52</td></tr><tr><td>Battery 34.86</td></tr></table>

c NYISO  
![](images/8f3b7f28a68f9031429f48c8871fe762447bfe083f5238ddb0b9a1a9fe8d763d.jpg)  
\$6.830 bn/yr

\$14.098 bn/yr  
d South Africa
<table><tr><td>Added power (GW)</td></tr><tr><td>Wind 18.08</td></tr><tr><td>Solar 0.38 0.00</td></tr><tr><td>Nuclear Battery 0.00</td></tr></table>

![](images/d5b71edc21e1c40c5aa931e8a30eac3fb3fc071c8d8b9c1ab1a356e317d69cb6.jpg)  
e Australia (NEM)

<table><tr><td>Added power (GW)</td><td>0.00</td></tr><tr><td>Wind Solar</td><td>53.29</td></tr><tr><td>Nuclear</td><td>0.00</td></tr><tr><td></td><td>5.14</td></tr><tr><td>Battery</td><td></td></tr></table>

![](images/46b8061c274d47b653f3f5cb575452ffb4d375337aa0d2a98cf708f9caaef1d3.jpg)  
\$8.853 bn/yr

<table><tr><td>Added power (GW)</td></tr><tr><td>Wind 24.11</td></tr><tr><td>Solar 0.00</td></tr><tr><td>Nuclear 0.00</td></tr><tr><td>Battery 0.54</td></tr></table>

f New Zealand  
![](images/fab06e5a655fe079afeec857e5b26fd02d214fc41e163f2cd180369b16b48078.jpg)

\$0.800 bn/yr
<table><tr><td>Added power (GW) Wind 1.43</td></tr><tr><td>Solar 0.00</td></tr><tr><td>Nuclear 0.00</td></tr><tr><td>Battery 0.00</td></tr><tr><td></td></tr></table>

g Mexico  
![](images/0f8be70e6d6d4d3f1360026225a549a5cde0e6261459cbdfcb6b97d1083fd79b.jpg)  
\$26.409 bn/yr

<table><tr><td>Added power (GW)</td></tr><tr><td>Wind 30.29</td></tr><tr><td>Solar 50.53</td></tr><tr><td>Nuclear 1.59</td></tr><tr><td>Battery 0.39</td></tr></table>

h Brazil  
![](images/1c227209948f2b58a573ae3f21c686582ec4b0f37f527663d4757f46e0651696.jpg)

\$33.736 bn/yr
<table><tr><td>Added power (GW)</td></tr><tr><td>Wind 91.86 84.88</td></tr><tr><td>Solar</td></tr><tr><td>Nuclear 2.33</td></tr><tr><td>Battery 0.00</td></tr></table>

Fig. C1 Lowest-cost accepted regional designs for 2035. a–h, Minimum accepted annual costs among four trajectories per system (billion USD). Power bars share a 0–100 GW scale; sector area encodes added generation and battery power. Colours identify technologies; hollow points mark additions below 0.005 GW. Black points locate US plants; black corridors are existing, dashed for assumed ratings. Magenta widths encode added transfer capacity. New allocations and straight corri dors have model-zone resolution. All 96 outcomes remain tabulated; Appendix E gives zonal amounts. The basemap uses Natural Earth[22].

b Far West feeder  
![](images/626d312ba3f7c420c9bb44c4c2bfadbe699f951b574895e2049eef9179e64b09.jpg)  
Fig. C2 ERCOT allocation and representative-feeder repair in an Opus multi-agent trajectory. This process example is distinct from the lower-cost Sol ERCOT design in Fig. C1 and Appendix E. a, New generation and battery power at numbered model zones, with the same technology colour key. b, Initial and selected Far West three-bus numerical circuits; line ratings increase from 400 to 600 A and impedances are halved under the intervention model. Line colours and percentages use annual peak current divided by each line’s own rating. Annual voltage ranges are shown separately; extrema can occur at diferent hours. c, Selected additions and battery energy; no new inter-area transfer is selected. d, Recorded violation hours, accepted feeder count and annualized intervention cost (million USD per year). All eight selected representative feeders pass. The selected design combines feeder reinforcement with a revised portfolio and dispatch.

## C.5 Search histories distinguish selection from continued exploration

The 96 trajectories contain 382 committed candidate assessments, including 369 system-simulation requests and thirteen additional feeder-only assessments. Of these,

307 pass the recorded acceptance gate and 75 fail its distribution component. The median trajectory contains three assessments (range, one to eighteen). The selected candidate first appears at assessment one in 56 trajectories and later in 40; 35 trajectories improve the objective after their first accepted candidate. In 28 trajectories, the selected candidate’s source assessment precedes the last assessment. Fifteen of these end by rechecking the same design, whereas thirteen end with a diferent design. Thus, the source of the selected outcome and the final evaluated proposal convey diferent information about the search.

Sixteen trajectories produce 36 subsequent assessments with a lower recorded objective than an already accepted incumbent but fail distribution validation. In the Opus ERCOT 2035 multi-agent example (Fig. C2), round four supplies the selected accepted design. Removing feeder reinforcement in round six lowers the objective by 0.220% but introduces three undervoltage hours. Round seven restores the accepted design, leaving round four as its selected source. A further 228 uncommitted systemonly probes inform exploration across the cohort; their results cover the system model and are reported separately from committed feeder assessments. These records show the predefined validation-and-retention rule operating throughout agent-directed exploration.

## C.6 Coverage and reproducible numerical evidence

The year-matched evaluation contains 96 completed annual trajectories. Thirty-two pair 2023 demand with ERA5 2023 resources; sixty-four pair 2030 or 2035 demand with resource profiles reconstructed for the same target year. Every trajectory yields a committed candidate satisfying the implemented system and representative-feeder acceptance gate. The eighty newly completed hosted trajectories also undergo independent reconstruction of committed candidate selection and recomputation of the physical gate from their retained numerical outputs. The evidence package preserves input and implementation identities, accepted candidates, costs, stopping decisions and recorded search efort. Appendix D reports the complete descriptive outcome tables and deployment comparisons.

## C.7 Token usage characterizes the efort of the planning loop

The complete reported search histories contain 2,257 distinct received model responses, totalling 276.386 million input tokens and 3.834 million output tokens (Table C5). Input includes cache reads and writes; output includes the reported reasoning subset. Across 48 trajectories per architecture, multi-agent input and output totals are respectively 2.08 and 23.34 times their single-agent counterparts. Every one of the 48 pairs matched by deployment, system and year uses more output tokens in the multi-agent configuration. The median paired output ratio is 5.94 for Sol and 25.10 for Opus. Input increases in 28 pairs; for Sol, the median paired input ratio is 0.74 despite an aggregate ratio of 1.55, reflecting a small number of large search histories. Totals, means and medians therefore describe distinct aspects of the observed efort. These comparisons describe search efort in the configured deployments.

Table C5 Input and output tokens across all 96 loose-cap reference trajectories. Input columns use millions of tokens; output columns use thousands. Means and medians are per trajectory. Counts cover complete search histories, including repeated context and inherited responses counted once. Input includes cache reads and writes; output includes reasoning. The archived evidence retains integer token counts for every response.
<table><tr><td>Deployment</td><td>Architecture</td><td>n</td><td colspan="3">Input tokens (million)</td><td colspan="3">Output tokens (thousand)</td></tr><tr><td></td><td></td><td></td><td>Total</td><td>Mean</td><td>Median</td><td>Total</td><td>Mean</td><td>Median</td></tr><tr><td>GPT-6.1 Sol</td><td>Single</td><td>24</td><td>74.854</td><td>3.119</td><td>1.756</td><td>91.667</td><td>3.819</td><td>3.437</td></tr><tr><td>GPT-6.1 Sol</td><td>Multi</td><td>24</td><td>116.006</td><td>4.834</td><td>0.495</td><td>1,364.790</td><td>56.866</td><td>11.160</td></tr><tr><td>Opus 5.5</td><td>Single</td><td>24</td><td>14.914</td><td>0.621</td><td>0.428</td><td>65.864</td><td>2.744</td><td>2.165</td></tr><tr><td>Opus 5.5</td><td>Multi</td><td>24</td><td>70.611</td><td>2.942</td><td>0.724</td><td>2,311.868</td><td>96.328</td><td>33.580</td></tr></table>

Appendix F gives assessment counts, selected assessment indices, rejected assessments, probes and token usage for every trajectory. The histories also record 142 applied multi-agent advisory-graph edits and 129 single-agent strategy updates. These records capture changes to the search procedure. Among the 44 matched pairs with complete duration records, the median multi/single runtime ratio is 1.84 for Sol and 4.01 for Opus; the four resumed trajectories are excluded from duration summaries but retain their full reconstructed token histories.

## C.8 Reconstruction of the reference outcome evidence

The reported evidence comprises sixteen Claude Opus 5.5 trajectories with 2023 demand and ERA5 2023 weather, plus forty-eight GPT-6.1 Sol and thirty-two Opus trajectories from the completed hosted-weather campaign. The latter includes sixteen Sol 2023 ERA5 controls and sixty-four future-weather cases. A reporting manifest selects these 96 trajectories, each with matching demand and resource-weather years. Extraction reconstructs committed candidates, applies the common acceptance gate and selects the minimum-cost accepted candidate in Eq. 5. Source manifests link the selected design to its system result, dispatch, feeder checks and model-response records. For all eighty new cases, reconstruction of the full committed history yields the same selected design and cost as the archived final; physical acceptance is recomputed from the recorded numerical results without new inference or optimization.

The thirty-two Sol–Opus pairs in the future-weather cohort share input and physical-code hashes and numerical settings, and all sixteen single-agent initial exposed states match. The 2023 controls retain campaign-specific inference settings. Comparisons pair deployment outcomes by system, demand year, weather source and architecture, with candidate-specific numerical bounds retained. All comparisons contain one trajectory per cell. Solver bounds apply to the optimization problem within each selected design domain.

Table D6 Claude Opus 5.5 planning with 2023 demand and ERA5 2023 weather. Entries are single/multi. Selected costs are billion USD per year, including representative-distribution interventions. System evaluations count committed and uncommitted verification requests, including exact-result reuse; model requests count unique received responses in the selected trajectory ledger.
<table><tr><td>System</td><td>Selected cost</td><td>System evaluations</td><td>Model requests</td></tr><tr><td>MISO</td><td>28.293256/28.293256</td><td>1/3</td><td>3/11</td></tr><tr><td>ERCOT</td><td>18.225239/18.225239</td><td>3/3</td><td>6/11</td></tr><tr><td>NYISO</td><td>8.030121/8.030121</td><td>2/3</td><td>5/21</td></tr><tr><td>South Africa</td><td>10.652093/10.656181</td><td>8/13</td><td>11/55</td></tr><tr><td>Australia</td><td>7.052820/7.052820</td><td>3/3</td><td>7/11</td></tr><tr><td>New Zealand</td><td>0.569150/0.569150</td><td>1/3</td><td>4/11</td></tr><tr><td>Mexico</td><td>19.552919/19.554932</td><td>3/11</td><td>6/44</td></tr><tr><td>Brazil</td><td>15.869564/15.869564</td><td>1/3</td><td>3/11</td></tr></table>

## Appendix D Loose-cap reference outcomes and evaluation efort

These records preserve the achieved outcomes and execution efort of the ninetysix loose-cap reference trajectories. They document the verified workflow under the recorded deployment and inference settings.

## D.1 Opus planning with 2023 demand and ERA5 2023 weather

Claude Opus 5.5 runs sixteen trajectories with 2023 demand and ERA5 2023 weather, with both architectures in each of the eight systems (Table D6). Every selected candidate passes the same code-computed system and representative-feeder conditions. Selection retains the lowest all-in annual cost among physically accepted committed candidates. Together with the sixteen GPT-6.1 Sol 2023 controls, these results give thirty-two baseline trajectories (Fig. D1).

The 2023-demand cohort records 38 committed evaluations, 26 verification requests and 220 received model responses. Solver cost intervals condition on each selected design. Each cell contains one trajectory. Model protocols, search ceilings and implementations are listed in Appendix B.

## D.2 Planning with projected weather and matched demand

The completed hosted campaign adds eighty annual trajectories, comprising sixteen GPT-6.1 Sol controls with 2023 demand and ERA5 2023 weather, and sixty-four future-weather trajectories with GPT-6.1 Sol and Claude Opus 5.5. Each future deployment covers eight systems, 2030 and 2035 demand, and both architectures. NASA MPI-ESM1-2-HR SSP2-4.5 daily fields and ERA5 2023 intraday templates supply the target-year resource profiles (Methods, Section 4.10). All eighty selected candidates pass the recomputed system and representative-feeder acceptance gate.

a System-evaluation requests  
![](images/5eece167e912685a89ca5ef02a1a001baa78eb61394bd23d260de2d1adea2761.jpg)

b NYISO  
![](images/b97f8c621c716ae41b0e5d7d2ff78213a2aa455a781c87f7e54a56a2b96cca6e.jpg)

c South Africa  
![](images/108a15fcc40682917d4c3697d618365c2c3172273949f6c0c76570615e913637.jpg)

d Mexico  
![](images/aa9ba7dcac0b7a290c52727086684fcb303f4831d3d37cdb1c10fbc783fd1626.jpg)  
Annual cost (USD billion)  
Fig. D1 Two deployments under 2023 demand and ERA5 2023 weather. a, System-evaluation requests across 32 trajectories, comprising committed simulations and hypothetical verifications, including exact-result reuse. Cell values are exact and colour uses one linear scale. b–d, Selected all-in annual costs in NYISO, South Africa and Mexico on panel-specific scales. Filled circles denote singleagent and open squares multi-agent trajectories. Blue denotes GPT-6.1 Sol and orange Claude Opus 5.5. Horizontal intervals extend from the candidate’s solver lower bound plus its distribution cost to its selected total cost; the magnified axes condition on each selected design.

Full committed-history reconstruction confirms that each selected result is the lowestcost physically accepted committed candidate. All trajectories end through their single-agent finish decision or multi-agent judge.

Across the 32 matched future-weather deployment pairs, Opus selects a lower annual cost in 1, an equal cost to the nearest cent in 18, and a higher cost in 13. The candidate-specific solver intervals overlap in 18 pairs. Shared demand, weather, inventory and network hashes, numerical settings and physical-code hashes are verified for all 32 pairs; all sixteen single-agent initial exposed states also match. Figure D2 shows achieved costs and their candidate-conditional numerical bounds.

Table D7 Selected annual cost under 2030 demand and NASA-derived 2030 weather. Entries are single/multi, in billion USD per year including annualized representative-distribution interventions. Each cell is one trajectory; candidate-specific solver intervals are retained in the comparison evidence.
<table><tr><td>System</td><td>GPT-6.1 Sol</td><td>Claude Opus 5.5</td></tr><tr><td>MISO</td><td>46.000616/46.145542</td><td>46.003694/46.003694</td></tr><tr><td>ERCOT</td><td>68.057741/68.076200</td><td>68.407019/68.407550</td></tr><tr><td>NYISO</td><td>6.477653/6.477653</td><td>6.477653/6.477653</td></tr><tr><td>South Africa</td><td>12.017688/12.019324</td><td>12.052166/12.020040</td></tr><tr><td>Australia</td><td>6.861099/6.861099</td><td>6.861099/6.861099</td></tr><tr><td>New Zealand</td><td>0.646734/0.646734</td><td>0.646734/0.646734</td></tr><tr><td>Mexico</td><td>22.257606/22.258090</td><td>22.258090/22.258090</td></tr><tr><td>Brazil</td><td>24.655438/24.655438</td><td>24.655438/24.655438</td></tr></table>

Table D8 Selected annual cost under 2035 demand and NASA-derived 2035 weather. Entries are single/multi, in billion USD per year including annualized representative-distribution interventions. Each cell is one trajectory; candidate-specific solver intervals are retained in the comparison evidence.
<table><tr><td>System</td><td>GPT-6.1 Sol</td><td>Claude Opus 5.5</td></tr><tr><td>MISO</td><td>52.959760/53.216539</td><td>52.961390/53.216539</td></tr><tr><td>ERCOT</td><td>82.980096/82.957622</td><td>83.762708/83.150660</td></tr><tr><td>NYISO</td><td>6.829947/6.829947</td><td>6.829947/6.829947</td></tr><tr><td>South Africa</td><td>14.097805/14.098005</td><td>14.202359/14.101251</td></tr><tr><td>Australia</td><td>8.853329/8.854495</td><td>8.854495/8.854495</td></tr><tr><td>New Zealand</td><td>0.799669/0.799669</td><td>0.799669/0.799669</td></tr><tr><td>Mexico</td><td>26.408564/26.411673</td><td>26.408751/26.411673</td></tr><tr><td>Brazil</td><td>33.735917/33.735917</td><td>33.735917/33.735917</td></tr></table>

Selected-cost diferences vary across architecture pairs. Across the sixteen future region–year architecture pairs per deployment, GPT-6.1 Sol’s multi-agent candidate is cheaper in one, equal to the nearest cent in seven and more expensive in eight. Opus’s corresponding counts are three, ten and three. In Opus ERCOT 2035, multiagent selection reduces annual cost from USD 83.762708 billion to USD 83.150660 billion (USD 612.048 million; 0.731%). In South Africa 2035, it reduces cost from USD 14.202359 billion to USD 14.101251 billion (USD 101.108 million; 0.712%). Conversely, the MISO 2035 single-agent selections cost approximately 0.48% less than their multi-agent counterparts in both deployments. These are achieved outcomes of individual searches, with feasibility, economic improvement and stopping efort reported separately.

Across the new hosted campaign, GPT-6.1 Sol records 244 committed and 132 verification requests, with 1,493 unique received responses. Opus records 87 committed and 70 verification requests, with 544 responses. The median multi-agent search length is one cycle in each deployment/year cohort; the longest new searches occur in Sol ERCOT 2030 (eighteen cycles) and South Africa 2030 (fifteen). Ledger attribution counts each inherited or newly received response once; three resumed trajectories have no uninterrupted wall-runtime estimate.

Table D9 Evaluation efort in the completed eighty-trajectory hosted campaign. Each row contains sixteen trajectories. Evaluations are logical requests, including exact-result reuse, rather than physical optimizer invocations. The 2023 control row uses ERA5 2023 weather; the future rows use the corresponding NASA-derived weather.
<table><tr><td>Deployment/year Committed</td><td></td><td>Verify</td><td>Responses</td><td>Median cycles</td></tr><tr><td>Sol 6.1 2023</td><td>62</td><td>29</td><td>398</td><td>1</td></tr><tr><td>Sol 6.1 2030</td><td>91</td><td>50</td><td>578</td><td>1</td></tr><tr><td>Sol 6.1 2035</td><td>91</td><td>53</td><td>517</td><td>1</td></tr><tr><td>Opus 5.5 2030</td><td>41</td><td>29</td><td>254</td><td>1</td></tr><tr><td>Opus 5.5 2035</td><td>46</td><td>41</td><td>290</td><td>1</td></tr></table>

Table D10 GPT-6.1 Sol controls with 2023 demand and ERA5 2023 weather. Entries are single/multi. Selected costs are billion USD per year including distribution interventions; evaluations include committed and hypothetical requests.
<table><tr><td>System</td><td>Selected cost</td><td>Evaluations</td><td>Responses</td></tr><tr><td>MISO</td><td>28.293256/28.293256</td><td>3/3</td><td>9/12</td></tr><tr><td>ERCOT</td><td>18.225239/18.225239</td><td>3/3</td><td>9/11</td></tr><tr><td>NYISO</td><td>8.030084/8.030121</td><td>4/4</td><td>14/21</td></tr><tr><td>South Africa</td><td>10.647249/10.647689</td><td>7/25</td><td>23/149</td></tr><tr><td>Australia</td><td>7.052820/7.052820</td><td>4/3</td><td>12/11</td></tr><tr><td>New Zealand</td><td>0.569150/0.569150</td><td>3/3</td><td>9/11</td></tr><tr><td>Mexico</td><td>19.551675/19.551810</td><td>7/12</td><td>21/64</td></tr><tr><td>Brazil</td><td>15.869564/15.869564</td><td>4/3</td><td>11/11</td></tr></table>

The sixteen GPT-6.1 Sol base-year controls also yield accepted candidates (Table D10). Five systems retain identical selected costs across architectures; NYISO, Mexico and South Africa have slightly higher multi-agent selected costs. The sixteen Opus 2023 trajectories complete the 32-case baseline comparison. Their recorded default response ceiling is 16,000 tokens, compared with 64,000 tokens in the GPT-6.1 Sol campaign.

![](images/baf63e802c63c89bd2473f0fbd4e7b270f996870734941674412fae829903cee.jpg)

![](images/b67d1dfead984911e975fb5efdea1494c48270cd5d2a0edc65c7c0e7d8599a59.jpg)

![](images/47f68cb2ec69b9773475e3e4cc0a9c2986b75463eac16b1cb6507bc1b477c5b4.jpg)

![](images/355fac09a83bf5a02b6233a6a873a0c5b91474821c4ef0e44e95d7ec940ff4fe.jpg)  
Fig. D2 Selected-cost comparisons with NASA-derived future weather. a,b, Multi minus single selected annual cost, as a percentage of the single-agent cost, for GPT-6.1 Sol and Claude Opus 5.5 under 2030 and 2035 demand and weather. c,d, Opus minus Sol selected annual cost as a percentage of the Sol cost, for the two architectures. Intervals extend from the first candidate’s solver lower bound minus the second candidate’s achieved cost to the first candidate’s achieved cost minus the second candidate’s lower bound. Each candidate’s distribution cost is included at both endpoints. The intervals describe numerical bounds conditional on the two selected design domains. Negative values favour multi in a,b and Opus in c,d. Each point represents one matched pair.

## Appendix E Loose-cap reference regional installation atlas

These eight 2035 portfolios retain the original 1,000 Mt annual ceiling. Carbonconstrained geography is reported separately in Fig. 3 and the accompanying carbon atlas.

![](images/f1bc16ead1e4498038d059b8f3ff52f6b998831a5f84e459f20db60b9f91e119.jpg)

![](images/d4d3bfacdb155a08f6ca26e79fad4bfb1e6722f4c98e9696e4a3fe52dac5019b.jpg)

<table><tr><td rowspan="2">#</td><td rowspan="2">Zone</td><td rowspan="2">Existing GW</td><td rowspan="2">Wind GW</td><td rowspan="2">Solar GW</td><td rowspan="2">Nuclear GW</td><td rowspan="2">Battery GW</td><td rowspan="2">Energy GWh</td></tr><tr><td></td></tr><tr><td>1</td><td>LRZ 1</td><td>27.88</td><td>9.66</td><td>12.25</td><td></td><td>9.18</td><td>30.21</td></tr><tr><td>2</td><td>LRZ 4</td><td>15.37</td><td></td><td></td><td>3.24</td><td></td><td></td></tr><tr><td>3</td><td>LRZ 6</td><td>21.54</td><td></td><td></td><td>10.31</td><td></td><td></td></tr><tr><td>4</td><td>LRZ 2+7</td><td>45.09</td><td>30.59</td><td>8.40</td><td>一</td><td>3.79</td><td>12.48</td></tr><tr><td>5</td><td>LRZ 3+5</td><td>31.78</td><td></td><td>27.89</td><td>0.42</td><td>6.81</td><td>22.41</td></tr><tr><td>6</td><td>LRZ 8+9+10</td><td>54.80</td><td></td><td>22.15</td><td>1.52</td><td></td><td></td></tr><tr><td></td><td>Total</td><td>196.46</td><td>40.26</td><td>70.68</td><td>15.49</td><td>19.78</td><td>65.09</td></tr></table>

New power (GW)

Fig. E1 Lowest-cost accepted 2035 design for MISO. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is GPT-6.1 Sol, single-agent, at USD 52.960 billion per year. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. Additional transfer capacity is 1.00 GW on 0035–8910. All represented feeder checks pass. Locations and corridors have model-zone resolution. The basemap uses Natural Earth.

![](images/1c6797a208930b711357c2286429f73b19fc1877636f3a60d99694e446104e7d.jpg)

<table><tr><td>#</td><td>Zone</td><td>Existing GW</td><td>Wind GW</td><td>Solar GW</td><td>Nuclear GW</td><td>Battery GW</td><td>Energy GWh</td></tr><tr><td>1</td><td>COAS</td><td>32.53</td><td></td><td>一</td><td>15.29</td><td></td><td></td></tr><tr><td>2</td><td>EAST</td><td>11.20</td><td></td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>3</td><td>FWES</td><td>13.90</td><td></td><td>69.53</td><td>一</td><td>26.77</td><td>157.26</td></tr><tr><td>4</td><td>NCEN</td><td>26.18</td><td></td><td></td><td>24.21</td><td>0.02</td><td>0.13</td></tr><tr><td>5</td><td>NRTH</td><td>15.08</td><td></td><td>3.00</td><td>5.11</td><td>2.44</td><td>14.33</td></tr><tr><td>6</td><td>SCEN</td><td>13.72</td><td></td><td></td><td>24.86</td><td></td><td></td></tr><tr><td>7</td><td>SOUT</td><td>15.88</td><td></td><td>21.12</td><td>10.05</td><td>5.63</td><td>33.07</td></tr><tr><td>8</td><td>WEST</td><td>11.57</td><td></td><td>一</td><td></td><td></td><td></td></tr><tr><td></td><td>Total</td><td>140.06</td><td></td><td>93.65</td><td>79.52</td><td>34.86</td><td>204.79</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>New power (GW)</td><td></td></tr><tr><td></td><td></td><td></td><td>Added transfer</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>_</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>一 Existing link</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Existing plant</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>01 10</td><td>50</td></tr></table>

Fig. E2 Lowest-cost accepted 2035 design for ERCOT. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is GPT-6.1 Sol, multi-agent, at USD 82.958 billion per year. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. No additional transfer capacity is selected. All represented feeder checks pass. Locations and corridors have model-zone resolution. The basemap uses Natural Earth.

![](images/e3b1df907ce9390b4afc2fb78dc48f8fe4399114c65ea3b5e6c61f12333e7212.jpg)

<table><tr><td>#</td><td>Zone</td><td>Existing GW</td><td>Wind GW</td><td>Solar GW</td><td>Nuclear GW</td><td>Battery GW</td><td>Energy GWh</td></tr><tr><td>1</td><td>ZONA</td><td>3.94</td><td></td><td>0.38</td><td></td><td></td><td></td></tr><tr><td>2</td><td>ZONB</td><td>1.06</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td>ZONC</td><td>7.30</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>ZOND</td><td>2.08</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>ZONE</td><td>1.92</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td>ZONF</td><td>3.21</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7</td><td>ZONG</td><td>6.86</td><td>0.28</td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td>ZONH</td><td>0.01</td><td>13.10</td><td></td><td></td><td></td><td></td></tr><tr><td>9</td><td>ZONI</td><td>0.06</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>ZONJ</td><td>10.76</td><td>1.29</td><td></td><td></td><td></td><td></td></tr><tr><td>11</td><td>ZONK</td><td>5.82</td><td>3.40</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Total</td><td>43.00</td><td>18.08</td><td>0.38</td><td>一</td><td>New power (GW)</td><td></td></tr><tr><td></td><td>Existing plant</td><td>— Existing link</td><td>Added transfer</td><td></td><td></td><td>01 10</td><td>50</td></tr></table>

Fig. E3 Lowest-cost accepted 2035 design for NYISO. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is Claude Opus 5.5, multi-agent, at USD 6.830 billion per year. All four trajectories have the same reported minimum cost; this portfolio is a stable representative of the tie. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. Additional transfer capacity is 1.00 GW on ZONB–ZONC, 1.00 GW on ZONE–ZONF, 1.00 GW on ZONI–ZONJ and 1.00 GW on ZONI–ZONK. All represented feeder checks pass. Locations and corridors have modelzone resolution. The basemap uses Natural Earth.

![](images/44c437c2c3f1dfa53300cec4da9e2e93ceba7ac26ae2929cbeb2d3c35685df93.jpg)

<table><tr><td>#</td><td>Zone</td><td>Existing GW</td><td>Wind GW</td><td>Solar GW</td><td>Nuclear GW</td><td>Battery GW</td><td>Energy GWh</td></tr><tr><td>1</td><td>EC</td><td>1.82</td><td></td><td>0.39</td><td></td><td>0.26</td><td>0.73</td></tr><tr><td>2</td><td>FS</td><td>0.33</td><td></td><td>4.60</td><td></td><td>0.23</td><td>0.65</td></tr><tr><td>3</td><td>GP</td><td>16.95</td><td></td><td>18.15</td><td></td><td></td><td></td></tr><tr><td>4</td><td>KZN</td><td>1.01</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>LP</td><td>3.84</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>6</td><td>MP</td><td>14.89</td><td></td><td>一</td><td></td><td>一</td><td></td></tr><tr><td>7</td><td>NW</td><td></td><td></td><td>17.80</td><td></td><td>3.51</td><td>10.02</td></tr><tr><td>8</td><td>NC</td><td>1.81</td><td></td><td>4.20</td><td></td><td>1.14</td><td>3.25</td></tr><tr><td>9</td><td>WC</td><td>8.61</td><td></td><td>7.70</td><td></td><td>一</td><td></td></tr><tr><td>10</td><td>HYDRA</td><td>1.92</td><td>一</td><td>0.45</td><td></td><td>一</td><td></td></tr><tr><td></td><td>Total</td><td>51.17</td><td>一</td><td>53.29</td><td>一</td><td>5.14</td><td>14.64</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>New power (GW)</td><td></td></tr><tr><td></td><td>Existing link</td><td>_ Added transfer</td><td></td><td></td><td></td><td>01 10</td><td>50</td></tr></table>

Fig. E4 Lowest-cost accepted 2035 design for South Africa. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is GPT-6.1 Sol, single-agent, at USD 14.098 billion per year. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. Additional transfer capacity is 1.20 GW on HYDRA–WC. All represented feeder checks pass. Locations and corridors have modelzone resolution. The basemap uses Natural Earth.

Existing link Added transfer

![](images/74adc584b117c33c0327920019f332fe0d5b41c091ce121c328db141552fb8ed.jpg)

<table><tr><td colspan="2">Added power (GW)</td></tr><tr><td>Wind</td><td>24.11</td></tr><tr><td>Solar</td><td>0.00</td></tr><tr><td></td><td>0.00</td></tr><tr><td>Nuclear</td><td></td></tr><tr><td>Battery</td><td>0.54</td></tr><tr><td></td><td></td></tr></table>

Battery energy

1.13 GWh

Added transfer

4.61 GW

<table><tr><td>#</td><td>Zone</td><td>Existing GW</td><td>Wind GW</td><td>Solar GW</td><td>Nuclear GW</td><td>Battery GW</td><td>Energy GWh</td></tr><tr><td>1</td><td>NSW1</td><td>20.44</td><td>8.90</td><td></td><td></td><td></td><td></td></tr><tr><td>2</td><td>QLD1</td><td>16.91</td><td>10.12</td><td></td><td></td><td>0.09</td><td>0.18</td></tr><tr><td>3</td><td>VIC1</td><td>14.89</td><td>1.34</td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>SA1</td><td>6.37</td><td></td><td></td><td></td><td>0.45</td><td>0.95</td></tr><tr><td>5</td><td>TAS1</td><td>3.21</td><td>3.76</td><td>一</td><td></td><td></td><td></td></tr><tr><td></td><td>Total</td><td>61.82</td><td>24.11</td><td>一</td><td>一</td><td>0.54</td><td>1.13</td></tr></table>

New power (GW)

![](images/006957cd3169d079653903e3b45395dab3bbf6b5fe09900d64fff8d4f7d775ce.jpg)

Fig. E5 Lowest-cost accepted 2035 design for Australia (NEM). Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is GPT-6.1 Sol, single-agent, at USD 8.853 billion per year. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. Additional transfer capacity is 1.26 GW on NSW1–QLD1, 2.55 GW on VIC1–TAS1 and 0.80 GW on NSW1–SA1. All represented feeder checks pass. Locations and corridors have model-zone resolution. The basemap uses Natural Earth.

<table><tr><td>Added power (GW) Wind</td><td>1.43</td></tr><tr><td></td><td></td></tr><tr><td>Solar</td><td>0.00</td></tr><tr><td>Nuclear</td><td>0.00</td></tr><tr><td></td><td></td></tr><tr><td>Battery</td><td>0.00</td></tr></table>

![](images/69ce46be13f6edd8484af44b52823ff6b193c713e617144e30f28fcbf3ec874e.jpg)

Battery energy

0.00 GWh

Added transfer

0.00 GW

<table><tr><td>#</td><td>Zone</td><td>Existing GW</td><td>Wind GW</td><td>Solar GW</td><td>Nuclear GW</td><td>Battery GW</td><td>Energy GWh</td></tr><tr><td>1</td><td>UNI</td><td>0.12</td><td></td><td>一</td><td></td><td></td><td></td></tr><tr><td>2</td><td>CNI</td><td>5.17</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td>LNI</td><td>3.00</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4</td><td>USI</td><td>0.28</td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>LSI</td><td>4.45</td><td>1.43</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Total</td><td>13.01</td><td>1.43</td><td>一</td><td></td><td></td><td></td></tr></table>

New power (GW)

![](images/a6da632f3fa10bc9a5522e4909494ece5f8dda5a75162990b761c77ea234fd33.jpg)

Fig. E6 Lowest-cost accepted 2035 design for New Zealand. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is Claude Opus 5.5, multi-agent, at USD 0.800 billion per year. All four trajectories have the same reported minimum cost; this portfolio is a stable representative of the tie. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. No additional transfer capacity is selected. All represented feeder checks pass. Locations and corridors have model-zone resolution. The basemap uses Natural Earth.

![](images/a0ef317d4473384bb91aa9fbf489bb95f60729113659d3f9e2cde51be799f7b5.jpg)

<table><tr><td>#</td><td>Zone</td><td>Existing GW</td><td>Wind GW</td><td>Solar GW</td><td>Nuclear GW</td><td>Battery GW</td><td>Energy GWh</td></tr><tr><td>1</td><td>BCA</td><td>3.26</td><td></td><td>1.47</td><td>1.50</td><td>0.39</td><td>1.91</td></tr><tr><td>2</td><td>BCS</td><td>1.21</td><td>0.38</td><td>0.35</td><td>0.09</td><td></td><td></td></tr><tr><td>3</td><td>NOR</td><td>9.12</td><td>4.52</td><td>6.88</td><td></td><td></td><td></td></tr><tr><td>4</td><td>NTE</td><td>6.66</td><td></td><td>11.78</td><td></td><td></td><td></td></tr><tr><td>5</td><td>NES</td><td>19.36</td><td>17.65</td><td>5.36</td><td></td><td></td><td></td></tr><tr><td>6</td><td>OCC</td><td>17.21</td><td></td><td>14.45</td><td></td><td></td><td></td></tr><tr><td>7</td><td>CEN</td><td>6.43</td><td></td><td>10.24</td><td></td><td></td><td></td></tr><tr><td>8</td><td>ORI</td><td>20.05</td><td></td><td>一</td><td></td><td></td><td></td></tr><tr><td>9</td><td>PEN</td><td>2.80</td><td>7.74</td><td>一</td><td></td><td></td><td></td></tr><tr><td></td><td>Total</td><td>86.12</td><td>30.29</td><td>50.53</td><td>1.59</td><td>0.39</td><td>1.91</td></tr></table>

Fig. E7 Lowest-cost accepted 2035 design for Mexico. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is GPT-6.1 Sol, single-agent, at USD 26.409 billion per year. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. No additional transfer capacity is selected. All represented feeder checks pass. Locations and corridors have model-zone resolution. The basemap uses Natural Earth.

![](images/4d777d7979766086143ef09f28d1d456575c98729ea8d97d324ed1cef1e2f57f.jpg)

<table><tr><td>Added power (GW) Wind</td><td>91.86</td></tr><tr><td></td><td></td></tr><tr><td>Solar</td><td>84.88</td></tr><tr><td>Nuclear</td><td>2.33</td></tr><tr><td></td><td></td></tr><tr><td>Battery</td><td>0.00</td></tr></table>

![](images/ba082092cb528bd18e4c22c193c5c05db5dcc9e0bd10b5a0899fffecd29282bc.jpg)

Battery energy

0.00 GWh

Added transfer

0.00 GW

<table><tr><td rowspan="2">#</td><td rowspan="2">Zone</td><td rowspan="2">Existing GW</td><td rowspan="2">Wind GW</td><td rowspan="2">Solar GW</td><td rowspan="2">Nuclear GW</td><td rowspan="2">Battery GW</td><td rowspan="2">Energy GWh</td></tr><tr><td></td></tr><tr><td>1</td><td>N</td><td>27.24</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2</td><td>NE</td><td>49.64</td><td>23.52</td><td></td><td></td><td></td><td></td></tr><tr><td>3</td><td>SE</td><td>82.46</td><td>68.33</td><td>64.87</td><td></td><td></td><td></td></tr><tr><td>4</td><td>S</td><td>21.11</td><td></td><td>20.01</td><td>2.33</td><td></td><td></td></tr><tr><td></td><td>Total</td><td>180.45</td><td>91.86</td><td>84.88</td><td>2.33</td><td>一</td><td></td></tr></table>

New power (GW)

Fig. E8 Lowest-cost accepted 2035 design for Brazil. Lowest observed selected total annual cost among four physically accepted trajectories using identical 2035 demand and NASA resource inputs. The displayed portfolio is Claude Opus 5.5, multi-agent, at USD 33.736 billion per year. All four trajectories have the same reported minimum cost; this portfolio is a stable representative of the tie. Map symbols follow Fig. C1; numbers link to the zonal ledger. Existing is inherited generation; Energy is added battery energy; dashes are below 0.005 in the displayed units. No additional transfer capacity is selected. All represented feeder checks pass. Locations and corridors have model-zone resolution. The basemap uses Natural Earth.

## Appendix F Search histories and token usage for the loose-cap reference

This section reports the ninety-six trajectories under the original 1,000 Mt annual ceiling. Tables F11 and F12 retain every trajectory. Each paired entry is single/multi for the same system and year. Assessments include feeder-only rechecks; their selected index identifies the first occurrence of the retained candidate. Failed assessments fail the recorded distribution check. Probes are uncommitted system-only verifications without feeder acceptance. Repeated designs remain separate assessments. Input tokens include cache reads and writes, and output includes the reasoning subset. Fullprecision counts, native round/turn indices, role attribution and response provenance are supplied in the frozen evidence archive.

Table F11 Assessment histories and tokens for all 48 GPT-6.1 Sol trajectories. Entries are single/multi. Input is in millions of tokens and output in thousands; displayed values are rounded. Selected is the source assessment index, not a native turn or cycle.
<table><tr><td>System</td><td>Year</td><td>Assessments</td><td>Selected</td><td>Failed</td><td>Probes</td><td>Input (M)</td><td>Output (k)</td></tr><tr><td rowspan="3">MISO</td><td>2023</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.582/0.509</td><td>2.133/11.211</td></tr><tr><td>2030</td><td>8/1</td><td>8/1</td><td>0/0</td><td>0/2</td><td>3.131/0.453</td><td>4.114/10.040</td></tr><tr><td>2035</td><td>11/1</td><td>11/1</td><td>0/0</td><td>0/2</td><td>6.279/0.454</td><td>4.805/10.811</td></tr><tr><td rowspan="3">ERCOT</td><td>2023</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.691/0.504</td><td>2.802/10.468</td></tr><tr><td>2030</td><td>13/18</td><td>13/17</td><td>6/6</td><td>0/20</td><td>9.287/23.873</td><td>8.033/276.833</td></tr><tr><td>2035</td><td>12/13</td><td>10/13</td><td>5/3</td><td>0/21</td><td>7.897/15.469</td><td>5.235/188.850</td></tr><tr><td rowspan="3">NYISO</td><td>2023</td><td>5/2</td><td>4/2</td><td>2/1</td><td>0/2</td><td>1.963/1.661</td><td>4.347/27.141</td></tr><tr><td>2030</td><td>4/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>1.549/0.731</td><td>3.147/12.452</td></tr><tr><td>2035</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.865/0.729</td><td>2.188/12.638</td></tr><tr><td rowspan="3">South Africa</td><td>2023</td><td>8/14</td><td>8/14</td><td>4/5</td><td>0/11</td><td>5.379/19.473</td><td>6.205/189.135</td></tr><tr><td>2030</td><td>10/15</td><td>10/15</td><td>3/3</td><td>0/18</td><td>8.130/24.199</td><td>8.328/221.683</td></tr><tr><td>2035</td><td>12/12</td><td>12/12</td><td>4/3</td><td>0/18</td><td>9.852/18.279</td><td>8.143/188.018</td></tr><tr><td rowspan="3">Australia</td><td>2023</td><td>4/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.746/0.317</td><td>2.300/10.856</td></tr><tr><td>2030</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.443/0.400</td><td>1.955/11.656</td></tr><tr><td>2035</td><td>8/2</td><td>6/2</td><td>0/0</td><td>0/4</td><td>2.455/1.045</td><td>3.996/24.353</td></tr><tr><td rowspan="3">New Zealand</td><td>2023</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.370/0.276</td><td>2.235/10.645</td></tr><tr><td>2030</td><td>5/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.965/0.346</td><td>2.729/11.050</td></tr><tr><td>2035</td><td>4/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.577/0.296</td><td>1.744/10.541</td></tr><tr><td rowspan="3">Mexico</td><td>2023</td><td>8/6</td><td>5/6</td><td>4/4</td><td>0/6</td><td>3.622/5.220</td><td>4.641/73.423</td></tr><tr><td>2030</td><td>8/1</td><td>8/1</td><td>0/0</td><td>0/2</td><td>3.890/0.486</td><td>3.727/10.554</td></tr><tr><td>2035</td><td>9/1</td><td>9/1</td><td>0/0</td><td>0/2</td><td>4.921/0.485</td><td>4.045/10.476</td></tr><tr><td rowspan="3">Brazil</td><td>2023</td><td>4/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.521/0.256</td><td>1.563/10.022</td></tr><tr><td>2030</td><td>4/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.539/0.273</td><td>1.829/10.824</td></tr><tr><td>2035</td><td>2/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.203/0.273</td><td>1.423/11.110</td></tr></table>

Table F12 Assessment histories and tokens for all 48 Opus 5.5 trajectories. Entries are single/multi. Input is in millions of tokens and output in thousands; displayed values are rounded. Selected is the source assessment index, not a native turn or cycle.
<table><tr><td>System</td><td>Year</td><td>Assessments</td><td>Selected</td><td>Failed</td><td>Probes</td><td>Input (M)</td><td>Output (k)</td></tr><tr><td rowspan="3">MISO</td><td>2023</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.087/0.621</td><td>1.213/26.456</td></tr><tr><td>2030</td><td>7/3</td><td>4/3</td><td>0/0</td><td>0/6</td><td>1.575/3.093</td><td>3.093/113.202</td></tr><tr><td>2035</td><td>7/1</td><td>5/1</td><td>0/0</td><td>0/2</td><td>1.577/0.656</td><td>3.177/31.438</td></tr><tr><td rowspan="3">ERCOT</td><td>2023</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.520/0.745</td><td>2.055/30.383</td></tr><tr><td>2030</td><td>4/6</td><td>4/6</td><td>2/1</td><td>0/9</td><td>0.609/8.323</td><td>5.636/290.225</td></tr><tr><td>2035</td><td>3/7</td><td>3/4</td><td>2/2</td><td>0/14</td><td>0.379/9.984</td><td>3.018/293.209</td></tr><tr><td rowspan="3">NYISO</td><td>2023</td><td>2/2</td><td>2/2</td><td>1/1</td><td>0/1</td><td>0.458/2.303</td><td>2.211/66.623</td></tr><tr><td>2030</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.230/0.898</td><td>2.282/31.254</td></tr><tr><td>2035</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.229/0.898</td><td>1.543/36.060</td></tr><tr><td rowspan="3">South Africa</td><td>2023</td><td>8/5</td><td>6/5</td><td>3/2</td><td>0/8</td><td>3.120/8.695</td><td>9.034/219.023</td></tr><tr><td>2030</td><td>3/5</td><td>3/4</td><td>1/1</td><td>0/6</td><td>0.813/9.487</td><td>6.051/245.520</td></tr><tr><td>2035</td><td>2/7</td><td>2/7</td><td>1/1</td><td>0/11</td><td>0.479/13.025</td><td>3.970/349.532</td></tr><tr><td rowspan="3">Australia</td><td>2023</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.398/0.471</td><td>1.781/25.197</td></tr><tr><td>2030</td><td>3/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.332/0.512</td><td>2.120/33.855</td></tr><tr><td>2035</td><td>5/3</td><td>2/3</td><td>0/0</td><td>0/6</td><td>0.872/2.405</td><td>3.195/122.218</td></tr><tr><td rowspan="3">New Zealand</td><td>2023</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.099/0.417</td><td>1.247/33.305</td></tr><tr><td>2030</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/0</td><td>0.107/0.366</td><td>1.110/29.497</td></tr><tr><td>2035</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.062/0.448</td><td>1.309/33.979</td></tr><tr><td rowspan="3">Mexico</td><td>2023</td><td>4/4</td><td>4/4</td><td>2/2</td><td>0/7</td><td>0.591/4.643</td><td>2.874/157.947</td></tr><tr><td>2030</td><td>4/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.740/0.704</td><td>2.061/26.903</td></tr><tr><td>2035</td><td>6/1</td><td>4/1</td><td>0/0</td><td>0/2</td><td>1.390/0.699</td><td>3.143/27.783</td></tr><tr><td rowspan="3">Brazil</td><td>2023</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.052/0.387</td><td>1.208/26.849</td></tr><tr><td>2030</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.098/0.418</td><td>1.386/29.177</td></tr><tr><td>2035</td><td>1/1</td><td>1/1</td><td>0/0</td><td>0/2</td><td>0.098/0.415</td><td>1.147/32.233</td></tr></table>

## Appendix G Regional data source catalogue

This catalogue groups the regional source documents by country. The Methods specify data definitions, model mappings and comparison boundaries. Preparation and comparison manifests retain extraction coordinates, workbook cells, source hashes and archived copies. Links identify the editions used in the experiments.

## United States.

Demand and fleet records use EIA-930 and EIA-860, with ERCOT weather-zone maps and 2023 NYCA generator-zone assignments. Demand outlooks are MISO 2026 LTLF regional supplementary data, ERCOT-adjusted 2025 LTLF and the 2026 NYISO Gold Book forecast tables and report. NYISO network sources are the 2024 RNA appendices, Summer 2025 Operating Study and final 2025–2034 CRP. MISO limits use the 2025–2026 LOLE study, corrected PRA results and SRIC/SREC posting; projects use LRTP Tranche 1, Tranche 2.1 and MTEP25 chapter 2. ERCOT network assumptions refer to the 2025 RTP economic study, Permian Basin reliability plan, PUCT 765 kV approval and STEP Eastern Backbone recommendation.

## South Africa.

Demand uses the ESRG Hourly Demand Model v2025 by Province and Demand Spatialization South Africa corrected regional outputs. The spatialization notebook retains 2017 calibration inputs; the experimental base demand year is 2023. Fleet totals use the Eskom Integrated Report 2023 snapshot of 31 March 2023; global plant coordinates and allocation weights are described in Methods. Network records use NTCSA GCCA line shapefiles and TDP 2025–2034 (TDP 2024), Public Report Rev. 2. The expert comparison uses User-Friendly IRP 2025, Table 1, printed page 31 (PDF page 41), combining the end-2025 base with scheduled 2026–2035 additions while retaining separate distributed-generation, CSP and hybrid categories.

## Australia.

The baseline uses AEMO 2023 price-and-demand records and the EMMS demandterm definitions; five-minute TOTALDEMAND is averaged hourly. Fleet selection uses the archived Registration and Exemption List and December 2023 MMSDM DUDETAIL-SUMMARY records. Network assumptions use AEMO’s Interconnector Capabilities (version 5 of 25 September 2025) and 2024 ISP Appendix 5. Demand growth uses 2026 NEM ESOO results, Step Change with the data-centre outlook; financial-year growth is mapped to the calendar-2023 demand template as stated in Methods.

## New Zealand.

Demand, fleet, hydro generation and connection records use the Electricity Authority EMI wholesale datasets. Network assumptions use the Transmission Planning Report 2023; the 1,400 MW northward HVDC target and assumed southward increment retain their distinct status. Future demand uses Security of Supply Assessment 2026 and its supplementary data, Medium Demand April–September grid-exit-point oftake.

## Mexico.

Baseline hourly demand uses CENACE daily real-demand settlement reports for 2023. Fleet and network records use PRODESEN 2024–2038, Annex 1; future demand uses chapter 3, Table 3.5, retaining the nine control-area mapping and the stated voltageto-rating rule.

## Brazil.

Baseline load, fleet, hydro generation and interchange records use ONS open datasets; ANEEL SIGA supplies archived coordinates. Network and project descriptions use PEN 2023 and PDE 2032 GET Norte, Volume I; the represented 4 GW HVDC bipole remains a model assumption. Future demand uses PDE 2035 final report, Table 2-3. Capacity benchmarks use its chapter-three workbook, Figura 3-25, December 2035 row 8, retaining wind E8, centralized solar F8, separate MMGD G8 and batteries I8.

## Operational carbon-budget sources.

NYISO 2023 uses RGGI Distribution of 2023 $\mathrm { C O _ { 2 } }$ Allowances. sheet1, New York row 14; base budget B14, adjustment C14, adjusted budget D14. The archived source value is 27295284 US short tons $\mathrm { C O _ { 2 } / y e a r } ;$ the adopted model ceiling is 24.761865118766 $\mathrm { M t C O _ { 2 } }$ per year.

NYISO 2030 uses New York adopted 6 NYCRR Part 242 Express Terms (2026 amendments). §242-5.1(d); PDF p. 47 of 84. Short-ton definition: §242-1.2 definition 71, PDF pp. 20–21. The archived source value is 13696558 US short tons $\mathrm { C O } _ { 2 } ,$ /year; the adopted model ceiling is 12.425308408125 $\mathrm { M t C O _ { 2 } }$ per year.

NYISO 2035 uses New York adopted 6 NYCRR Part 242 Express Terms (2026 amendments). §242-5.1(i); PDF p. 47 of 84. Short-ton definition: §242-1.2 definition 71, PDF pp. 20–21. The archived source value is 3913302 US short tons $\mathrm { C O _ { 2 } / y e a r ; }$ the adopted model ceiling is 3.550087857411 $\mathrm { M t C O _ { 2 } }$ per year.

MISO 2030 uses MISO Futures 1–4 Preliminary Results, September 24, 2025 workshop (updated September 23). Slide 66, Future 2 MISO aggregate constraint, year 2030; use exact table value rather than rounded slide 65 chart. The archived source value is 220499016 US short tons $\mathrm { C O _ { 2 } / y e a r ; }$ the adopted model ceiling is 200.033342500216 $\mathrm { M t C O _ { 2 } }$ per year.

MISO 2035 uses MISO Futures 1–4 Preliminary Results, September 24, 2025 workshop (updated September 23). Slide 66, Future 2 MISO aggregate constraint, year 2035; use exact table value rather than rounded slide 65 chart. The archived source value is 181486560 US short tons $\mathrm { C O _ { 2 } / y e a r ; }$ the adopted model ceiling is 164.641837747094 $\mathrm { M t C O _ { 2 } }$ per year.

South Africa 2030 uses South Africa Integrated Resource Plan 2025, Government Gazette 53596, Notice 6767. Figure 21, printed p. 35 $/ \mathrm { \ P D F { \mathrm { ~ p ~ } } }$ . 44, balanced-plan electricity $\mathrm { C O _ { 2 } }$ trajectory, 2030. The archived source value is 168 $\mathrm { M t C O _ { 2 } / y e a r } ;$ the adopted model ceiling is 168.000000000000 $\mathrm { M t C O _ { 2 } }$ per year.

South Africa 2035 uses South Africa Integrated Resource Plan 2025, Government Gazette 53596, Notice 6767. Figure 21, printed p. 35 / PDF p. 44, balanced-plan electricity $\mathrm { C O _ { 2 } }$ trajectory, 2035. The archived source value is 142 $\mathrm { M t C O _ { 2 } / y e a r ; }$ the adopted model ceiling is 142.000000000000 $\mathrm { M t C O _ { 2 } }$ per year.

Mexico 2030 uses Mexico Programa Sectorial de Energ´ıa 2025–2030 (PROSENER). Electricity-generation emissions paragraph: net-injected-electricity emissions peak in 2027, then decline 23.9% from 156.42 to 119 $\mathrm { M t C O _ { 2 } e }$ by the 2030 planning horizon. The archived source value is 119 $\mathrm { M t C O _ { 2 } e / y e a r } ;$ ; the adopted model ceiling is 119.000000000000 $\mathrm { M t C O _ { 2 } }$ per year.

Brazil 2030 uses Brazil EPE/MME Plano Decenal de Expans˜ao de Energia 2035, approved final report. Table 10-1, p. 393, SIN row, 2030; scope/hydrology discussion pp. 394–395 and explicit absence of model GHG limits p. 396. The archived source value is 27 $\mathrm { M t C O _ { 2 } e / } $ year; the adopted model ceiling is 27.000000000000 $\mathrm { M t C O _ { 2 } }$ per year.

Brazil 2035 uses Brazil EPE/MME Plano Decenal de Expans˜ao de Energia 2035, approved final report. Table 10-1, p. 393, SIN row, 2035; scope/hydrology discussion pp. 394–395 and explicit absence of model GHG limits $\mathrm { p . }$ 396. The archived source value is 29 $\mathrm { M t C O _ { 2 } e / y e a r } ;$ ; the adopted model ceiling is 29.000000000000 $\mathrm { M t C O _ { 2 } }$ per year.

## References

[1] International Energy Agency. Electricity 2026. Tech. Rep., International Energy Agency, Paris (2026). URL https://www.iea.org/reports/electricity-2026. Published 6 February 2026.

[2] Colangelo, P. et al. AI data centres as grid-interactive assets. Nature Energy 11, 254–261 (2026). URL https://www.nature.com/articles/s41560-025-01927-1.

[3] Levin, T. et al. Energy storage solutions to decarbonize electricity through enhanced capacity expansion modelling. Nature Energy 8, 1199–1208 (2023). URL https://www.nature.com/articles/s41560-023-01340-6.

[4] Wang, Y. et al. Accelerating the energy transition towards photovoltaic and wind in China. Nature 619, 761–767 (2023). URL https://www.nature.com/articles/ s41586-023-06180-8.

[5] Zhu, Z. et al. Integrated planning of net-zero power systems for all. Nature Energy 11, 1111–1131 (2026). URL https://www.nature.com/articles/s41560- 026-02054-1.

[6] Few, S., Djapic, P., Strbac, G., Nelson, J. & Candelise, C. A geographically disaggregated approach to integrate low-carbon technologies across local electricity networks. Nature Energy 9, 871–882 (2024). URL https://www.nature.com/ articles/s41560-024-01542-6.

[7] Brockway, A. M., Conde, J. & Callaway, D. Inequitable access to distributed energy resources due to grid infrastructure limits in California. Nature Energy 6, 892–903 (2021). URL https://www.nature.com/articles/s41560-021-00887-6.

[8] Dioha, M. O. & Mutiso, R. M. Generating meaningful energy systems models for Africa. Issues in Science and Technology 39 (2023).

[9] Larson, E. et al. Net-zero america: Potential pathways, infrastructure, and impacts. final report. Tech. Rep., Princeton University (2021). URL https: //netzeroamerica.princeton.edu/the-report.

[10] Beiter, P., Mai, T., Mowers, M. & Bistline, J. Expanded modelling scenarios to understand the role of ofshore wind in decarbonizing the United States. Nature Energy 8, 1240–1249 (2023). URL https://www.nature.com/articles/s41560-023- 01364-y.

[11] Senga, J. R. L., Botterud, A., Parsons, J. E., Story, S. D. & Knittel, C. R. Impli cations of policy-driven transmission expansion for costs, emissions and reliability in the USA. Nature Energy 11, 230–243 (2026). URL https://www.nature.com/ articles/s41560-025-01921-7.

[12] Liang, E., Chen, M. & Keshav, S. Heatwave-induced capacity bottlenecks in European electricity grids. Nature Energy (2026).

[13] Hofmann, M., Kotzur, L. & Stolten, D. The pareto-optimal temporal aggregation of energy system models. Applied Energy 315, 119029 (2022).

[14] Prina, M. G., Dallapiccola, M., Moser, D. & Sparber, W. Machine learning as a surrogate model for EnergyPLAN: Speeding up energy system optimization at the country level. Energy 307, 132735 (2024).

[15] Brown, T., H¨orsch, J. & Schlachtberger, D. PyPSA: Python for power system analysis. Journal of Open Research Software 6, 4 (2018).

[16] Parzen, M. et al. PyPSA-Earth. a new global open energy system optimization model demonstrated in Africa. Applied Energy 341, 121096 (2023).

[17] Electric Power Research Institute. Introduction to OpenDSS. https://opendss. epri.com/. Accessed 5 September 2026.

[18] Jia, M., Cui, Z. & Hug, G. Enhancing LLMs for power system simulations: A feedback-driven multi-agent framework. IEEE Transactions on Smart Grid 16, 5556–5572 (2025).

[19] Zhuang, M. & Sadikovic, A. A Human-in-the-Loop LLM-Assisted Multi-Agent Framework for Distribution Grid Reliability Planning. In 2026 IEEE Power & Energy Society General Meeting (PESGM), 1–5 (2026).

[20] Shao, K., Xiang, Y., Li, L. & Liu, J. LLM-empowered multi-agent intelligent planning for PV-EV penetrated distribution networks. Applied Energy 421, 128280 (2026).

[21] Lu, Y., Chen, Y., Wu, S. & Arık, S. O. Procedural graphs: Self-evolving execution<sup>¨</sup> structures for LLM agents. arXiv preprint (2026). ArXiv:2609.09153.

[22] Natural Earth. 1:50m cultural vectors: Admin 0 countries and admin 1 states and provinces, version 5.1.1. https://www.naturalearthdata.com/downloads/ 50m-cultural-vectors/ (2022). Public-domain geographic boundaries; accessed 3 October 2026.

[23] Dekker, M. M. et al. Identifying energy model fingerprints in mitigation scenarios. Nature Energy 8, 1395–1404 (2023). URL https://www.nature.com/articles/ s41560-023-01399-1.

[24] Fisch-Romito, V., Jaxa-Rozen, M., Wen, X. & Trutnevyte, E. Multi-country evidence on societal factors to include in energy transition modelling. Nature Energy 10, 460–469 (2025). URL https://www.nature.com/articles/s41560-025- 01719-7.

[25] Heuberger, C. F., Stafell, I., Shah, N. & Mac Dowell, N. Impact of myopic decision-making and disruptive events in power systems planning. Nature Energy 3, 634–640 (2018). URL https://www.nature.com/articles/s41560-018-0159-3.

[26] Zeyringer, M., Price, J., Fais, B., Li, P.-H. & Sharp, E. Designing low-carbon power systems for Great Britain in 2050 that are robust to the spatiotemporal and inter-annual variability of weather. Nature Energy 3, 395–403 (2018). URL https://www.nature.com/articles/s41560-018-0128-x.

[27] Perera, A. T. D., Nik, V. M., Chen, D., Scartezzini, J.-L. & Hong, T. Quantifying the impacts of climate change and extreme climate events on energy systems. Nature Energy 5, 150–159 (2020). URL https://www.nature.com/ articles/s41560-020-0558-0.

[28] Qiu, L., Khorramfar, R., Wang, S., Amin, S. & Howland, M. F. Climate change reshapes resource adequacy risks and optimal renewable energy siting in wind and solar energy systems. Nature Energy 11, 1449–1458 (2026). URL https: //www.nature.com/articles/s41560-026-02109-3.

[29] Zheng, D. et al. Strategies for climate-resilient global wind and solar power systems. Nature 643, 1263–1270 (2025). URL https://www.nature.com/articles/ s41586-025-09266-7.

[30] U.S. Energy Information Administration. Hourly electric grid monitor. https: //www.eia.gov/electricity/gridmonitor/about. Source of EIA-930 demand data; accessed 5 September 2026.

[31] Merven, B. ESRG hourly demand model v2025 by province. University of Cape Town, ZivaHub (2025). URL https://doi.org/10.25375/uct.26942134. Accessed 8 October 2026.

[32] Australian Energy Market Operator. Aggregated price and demand data, NEM regions (5-minute), 2023 (2023). URL https://aemo.com.au/energy-systems/ electricity/national-electricity-market-nem/data-nem/aggregated-data.

[33] Electricity Authority Te Mana Hiko. Electricity market information (EMI) wholesale datasets: Grid export, generation by plant, existing generation fleet, network supply points (2026). URL https://www.emi.ea.govt.nz/Wholesale/Datasets.

[34] Secretar´ıa de Energ´ıa. Programa de desarrollo del sistema el´ectrico nacional 2024– 2038, anexo 1. Tech. Rep., SENER, Mexico (2024). URL http://base.energia. gob.mx/PRODESEN2024/prodesen24-38A1.PDF.

[35] Operador Nacional do Sistema El´etrico. ONS dados abertos: Curva de carga hor´aria, capacidade de gera¸c˜ao, gera¸c˜ao por usina, intercˆambios entre subsistemas (2026). URL https://dados.ons.org.br/.

[36] Mirletz, B. et al. 2024 Annual Technology Baseline (ATB) Cost and Performance Data for Electricity Generation Technologies. National Renewable Energy Laboratory, Open Energy Data Initiative (2024). URL https://doi.org/10.25984/ 2377191. Version 4 workbook released 22 July 2026.

[37] PyPSA Contributors. Energy System Technology Data. https://github.com/ PyPSA/technology-data. Study uses locally archived 2030 cost inputs; repository accessed 5 September 2026.

[38] Global Energy Observatory, Google, KTH Royal Institute of Technology in Stockholm, Enipedia & World Resources Institute. Global Power Plant Database (2021). URL https://datasets.wri.org/datasets/global-power-plant-database. Version 1.3.0 released 2 June 2021.

[39] Meridian Economics. PyPSA-RSA: An open optimisation model of the south african power system. Software repository. URL https://github.com/MeridianEconomics/pypsarsa/tree/89872c1ea703af3d8a3f198706d1ab7958f50a5f. Commit 89872c1ea703af3d8a3f198706d1ab7958f50a5f; accessed 8 October 2026.

[40] H¨orsch, J. & Calitz, J. PyPSA-ZA: Investment and operation co-optimization of integrating wind and solar in south africa at high spatial and temporal detail (2017). URL https://arxiv.org/abs/1710.11199. Preprint, arXiv:1710.11199.

[41] Gurobi Optimization, LLC. Gurobi optimizer reference manual, version 13. https: //www.gurobi.com (2026).

[42] National Laboratory of the Rockies. PVWatts V8 API. URL https://developer. nlr.gov/docs/solar/pvwatts/v8/. PVWatts version 8 model documentation; accessed 8 October 2026.

[43] National Renewable Energy Laboratory. Pvwattsv8: PySAM module reference. URL https://nrel-pysam.readthedocs.io/en/v7.1.0/modules/Pvwattsv8. html. PySAM 7.1.0 documentation; accessed 8 October 2026.

[44] Hersbach, H., Bell, B., Berrisford, P. et al. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146, 1999–2049 (2020).

[45] Open-Meteo. Historical weather api (era5). https://open-meteo.com/en/docs/ historical-weather-api (2026).

[46] National Renewable Energy Laboratory. 2020ATB NREL Reference 4MW 150. https://natlabrockies.github.io/turbine-models/2020ATB NREL Reference 4MW 150.html. 2020 ATB reference model, 4 MW and 150 m rotor; locally archived tabular curve. Documentation accessed 28 September 2026.

[47] Thrasher, B. et al. NASA Global Daily Downscaled Projections, CMIP6. Scientific Data 9, 262 (2022).

[48] NASA Earth Exchange. Global Daily Downscaled Projections version 2 (NEX-GDDP-CMIP6 v2.0): Technical Note. Tech. Rep., NASA Center for Climate Simulation (2025). URL https://www.nccs.nasa.gov/wp-content/uploads/2025/ 06/NEX-GDDP-CMIP6-v2-Tech Note.pdf. Updated 31 May 2025.

[49] Anthropic. Claude Opus 5.5 (2026). URL https://platform.claude.com/docs/ en/models/opus-5-5/overview. Model identifier claude-opus-5-5; accessed 24 September 2026.

[50] Boerner, T. J., Deems, S., Furlani, T. R., Knuth, S. L. & Towns, J. ACCESS: Advancing Innovation: NSF’s Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support. In Practice and Experience in Advanced Research Computing, PEARC ’23, 173–176 (Association for Computing Machinery, 2023). URL https://doi.org/10.1145/3569951.3597559.

[51] Anthropic. Preserved thinking (2026). URL https://platform.claude.com/ docs/en/build-with-claude/preserved-thinking. Claude Platform documentation; accessed 28 September 2026.