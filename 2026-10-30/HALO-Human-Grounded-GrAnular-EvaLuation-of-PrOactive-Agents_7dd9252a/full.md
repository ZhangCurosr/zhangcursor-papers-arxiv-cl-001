# HALO: Human-Grounded GrAnular EvaLuation of PrOactive Agents

Wenqi Pei<sup>1</sup>, Xinyu Wang<sup>2</sup>, Jinrui Liu<sup>2</sup>, Cynthia Zhuge<sup>2</sup>, Jie Sheng<sup>2</sup> Zihan Wang<sup>2</sup>, Tenghao Huang<sup>3</sup>, Muhao Chen<sup>1</sup>

<sup>1</sup>University of California, Davis <sup>2</sup>2077AI <sup>3</sup>University of Southern California

## Abstract

Proactive agents must decide whether, when, and how far to assist without an explicit request. Existing evaluations often rely on automated scores or binary verdicts, offering limited human diagnosis of evidence use and intervention quality. To address these gaps, we introduce HALO<sup>1</sup>, Human-Grounded GrAnular EvaLuation of PrOactive Agents. HALO-Arena covers 4 proactive need types, Dependency, Blockage, Risk, and Opportunity, across 10 everyday domains and 50 personal environments. Scenarios require agents to retrieve evidence from structured databases or unstructured documents before judging whether assistance is warranted and choosing its scope. On HALO-Arena, we construct HALO-8K, a dataset of 8,000 responses from 10 models across 400 scenarios, annotated by 20 experts for action level, acceptability verdicts, and human diagnosis. For rejected responses, human diagnosis covers timing, intensity, and other issues. Recorded reasoning tokens and retrieval rounds measure the effort behind these responses. The best proprietary and opensource models achieve 72.88% and 56.63% acceptability, respectively. Our central finding is that over-proactivity is the dominant observed failure mode, with Premature timing or Excessive intensity accounting for 61.20% of rejected responses. Further analyses yield complementary findings on the balance between intervention and restraint, evidence acquisition, and the calibration of timing and intensity.

## 1 Introduction

Proactive agents infer user needs from observed activity and initiate assistance without an explicit request (Lu et al., 2024). They have attracted growing attention for their potential to anticipate problems and reduce user effort (Tang et al., 2026; Oh et al., 2026). They can identify missing prerequisites, anticipate risks, and prepare useful next steps within the user’s authority. For example, Lu et al. (2024) train agents to propose tasks from observed user activities and report a 66.47% F1 score for proactive assistance. Useful initiative requires deciding whether, when, and how far to help, while recognizing when restraint is warranted.

A calendar entry for an upcoming trip may prompt checking a database for needed supplies and a file for purchase authorization. These facts can justify a reminder without permitting a purchase. Retrieved evidence must guide the intervention’s timing, inform its content, and establish the authorized scope.

Benchmarks differ in how agents obtain evidence for proactive decisions. Some provide user activity directly (Lu et al., 2024; Tang et al., 2026). Such observations may signal a need without establishing a remedy or permission to act. PROBE supports SQL and semantic document retrieval (Pasternak et al., 2025), while interactive benchmarks support additional tool use (Oh et al., 2026; Zhang et al., 2026). Evaluating these capabilities requires linking retrieved evidence to human verdicts on the resulting interventions.

Detecting a relevant need does not by itself make an intervention acceptable. An intervention may still fail because it is mistimed, insufficient, excessive, irrelevant, or unsupported by evidence. Existing benchmarks primarily rely on scalable automated metrics: ProAgentBench evaluates intervention prediction and response similarity (Tang et al., 2026), while ProactiveBench uses a reward model trained on human acceptance labels (Lu et al., 2024). Human studies have examined how users perceive proactive assistance (Oh et al., 2026), but existing benchmarks provide limited human diagnosis of why individual model responses are accepted or rejected.

To address these gaps, we introduce HALO, Human-Grounded GrAnular EvaLuation of PrOactive Agents, with 2 components. HALO combines structured database querying and unstructured file retrieval with fine-grained human evaluation. HALO-Arena models 4 proactive need types: Dependency, Blockage, Risk, and Opportunity. It spans 10 everyday domains and 50 personal environments. Operational applications expose evolving events, while separate databases and text files provide structured and unstructured evidence. Addressing a need requires the agent to retrieve relevant evidence and then judge whether assistance is warranted, when to act, and how far to intervene. At each evaluation checkpoint, agents receive an observed event prefix and may retrieve evidence over multiple rounds before returning 1 executable response or no action.

![](images/3a3b0ac9b37a81f7326c6dcf749aa98a4e581606b005c2616434da4b00b1e040.jpg)  
Figure 1: Overview of HALO. HALO-Arena supplies personal apps and local evidence. Agents retrieve evidence and decide whether, when, and how far to intervene. HALO-8K records human verdicts, timing and intensity diagnosis, and resource consumption. The circled choices illustrate a premature and excessive L3 intervention.

On HALO-Arena, we construct HALO-8K, comprising 8,000 responses from 10 models across 400 scenarios, annotated by 20 experts. Annotators assign action levels and acceptability verdicts, with human diagnosis of rejections covering timing, intensity, and other issues such as unsupported evidence, irrelevance, and factual contradiction. Verdicts on no-action responses distinguish erroneous Inaction from warranted restraint. Reasoning tokens and retrieval rounds quantify agent effort.

Our central finding is that over-proactivity is the dominant observed failure mode: Premature timing or Excessive intensity accounts for 61.20% of rejected responses. Proprietary and open-source models peak at 72.88% and 56.63% acceptability. Further analysis yields 3 complementary findings.

(1) Restraint contributes to acceptability: gains in accepted action can be outweighed by losses in warranted restraint. (2) Evidence acquisition accompanies better help: among reminders, local content access is associated with higher acceptability and lower Insufficient diagnosis rates. (3) Timing alone is not enough: among well-timed rejections, reminders tend to be insufficient and executions excessive. These findings support granular evaluation of proactive assistance.

Our contributions are as follows:

• We introduce HALO-Arena, an executable benchmark spanning 4 proactive need types and 10 everyday domains, with evidence retrieval informing proactive decisions.

• We construct HALO-8K: 8,000 responses across 400 scenarios, annotated by experts for action level, verdicts, and human diagnosis.

• We evaluate 10 models, revealing tradeoffs among intervention, restraint, timing, intensity, and retrieval effort.

## 2 Related Work

## 2.1 Proactive Agents

Early agents predict useful tasks from observed activity and refine proposals using human acceptance verdicts (Lu et al., 2024). ContextAgent extends perception to wearable video and audio, combining sensory and persona context for tool-assisted services (Yang et al., 2025).

Recent work also examines timing and scope. ProActor models actionable windows for timingaware scheduling (Ding et al., 2026). Proactivity-Gym connects task capability with temporal allocation and user trust (Oh et al., 2026).

## 2.2 Benchmarks for Proactive Agents

Evaluation has expanded from executing explicit requests to identifying unstated needs. WebArena and τ-bench test request-driven task execution (Zhou et al., 2023; Yao et al., 2024). ProactiveBench evaluates proposals from activity traces (Lu et al., 2024), while ContextAgentBench adds multimodal context and tool calling (Yang et al., 2025). PROBE evaluates bottleneck discovery and action selection from document stores, while its agent baselines use SQL and semantic retrieval (Pasternak et al., 2025).

Recent benchmarks examine proactive assistance in interactive settings. ProAgentBench evaluates intervention decisions and generated assistance in real-world computer activity (Tang et al., 2026). FingerTip 20K covers proactive intent suggestions and personalized Android execution (Yang et al., 2026). PARE-Bench introduces stateful apps and active user simulation (Nathani et al., 2026), while π-Bench evaluates hidden-intent resolution and task completion across persistent sessions (Zhang et al., 2026).

Complementary evaluations focus on intervention calibration. AgentAbstain tests action and restraint under explicit requests (Liu et al., 2026). ProEvent evaluates response timing and timetable updates in ongoing chats (Li et al., 2026). Proactivity-Gym measures temporal allocation, intervention depth, and trust in multi-day simulations and includes a human study (Oh et al., 2026).

HALO combines executable apps, structured database queries, and unstructured file retrieval with human verdicts, granular human diagnosis, and recorded resource consumption. Appendix A compares benchmark environments and evaluation.

## 3 HALO

## 3.1 HALO-Arena

## 3.1.1 Overview

Applications. HALO-Arena combines email, calendar, chat, contacts, and wallet apps with SQLite databases and text files. Agents use 29 interaction commands for evidence retrieval and 19 action commands for attention, preparation, and execution. Appendix B lists the apps and commands by category.

![](images/5082167a2cd44d05c5d7100403dfb68ca2b046e018ca6a44ed4b9f5568c0041b.jpg)  
Figure 2: HALO-Arena construction. Validator and Reviewer return revision feedback before background records and human audit finalize the dataset.

Domains. The 10 domains are education and learning, work and professional life, home and household management, family and caregiving, health and wellbeing, travel and mobility, personal finance and consumer life, community and civic life, social and cultural life, and recreation (sports and hobbies). Each domain has 5 instances, each representing a different person, yielding 50 distinct personal environments.

Proactive needs. We consider 4 need types. Dependency is a missing prerequisite for a future activity. Blockage occurs when an ongoing activity cannot continue. Risk is a possible adverse outcome that has not yet occurred. Opportunity is an optional benefit whose omission causes no failure or loss. Each environment contains 2 scenarios per need type, yielding 8 scenarios.

As shown in Figure 1, HALO-Arena provides executable environments for agent actions, with 400 designed scenarios.

## 3.1.2 Construction

Creators. We use GPT5.6-Sol to construct HALO-Arena through the pipeline shown in Figure 2. Instance Creator defines personal responsibilities, authority, and application and storage structures. Scenario Creator populates these structures to form an initial state $S _ { 0 }$ and generates $T$ timestamped events $( e _ { 1 } , \dots , e _ { T } )$ for each scenario, where $e _ { t }$ denotes event t. Trajectories span at least 3 operational apps and include prefixes where intervention would be premature or belated, with local evidence for that decision but no explicit help requests or agent interventions. After all 8 scenarios pass review, Noise Creator adds settled background records to $S _ { 0 }$ to broaden retrieval.

Validator and Reviewer. Validator deterministically checks structure, command signatures, state consistency, and isolated execution. Reviewer assesses causal support, need identity, timing, retrieval necessity, and proportionate assistance. Scenario Creator revises failed scenarios.

Table 1: Model performance and resource consumption. Action shares and error rates are percentages of all responses. Inact., Prem., Bel., Insuff., and Excess. denote Inaction, Premature, Belated, Insufficient, and Excessive. Shading marks best , second , and third
<table><tr><td rowspan="2">Model</td><td colspan="4">Action Levels</td><td rowspan="2">Inact.↓</td><td colspan="2">Timing Errors↓</td><td colspan="2">Intensity Errors↓</td><td colspan="2">Consumption↓</td><td rowspan="2">Acc (%)↑</td></tr><tr><td>L0</td><td>L1</td><td>L2</td><td>L3</td><td>Prem.</td><td>Bel.</td><td>Insuff.</td><td></td><td>Excess. Reas. (k) Rounds</td><td></td></tr><tr><td colspan="10">Proprietary Models</td><td colspan="3"></td></tr><tr><td>Gemini3.8-Flash</td><td>30.13</td><td>57.50</td><td>2.13</td><td>10.25</td><td>7.38</td><td>10.50</td><td>3.75</td><td>2.63</td><td>7.00</td><td>14.99</td><td>4.19</td><td>72.88</td></tr><tr><td>Claude-Opus5.5</td><td>31.75</td><td>48.00</td><td>5.13</td><td>15.13</td><td>6.75</td><td>10.88</td><td>3.88</td><td>2.00</td><td>8.38</td><td>3.48</td><td>2.27</td><td>69.75</td></tr><tr><td>Grok4.7</td><td>3.63</td><td>76.50</td><td>2.00</td><td>17.88</td><td>0.88</td><td>21.38</td><td>6.88</td><td>3.25</td><td>14.63</td><td>5.04</td><td>2.70</td><td>65.25</td></tr><tr><td>Gemini3.1-Pro</td><td>3.88</td><td>58.63</td><td>7.63</td><td>29.88</td><td>0.13</td><td>27.63</td><td>6.13</td><td>4.63</td><td>19.25</td><td>3.80</td><td>1.85</td><td>55.75</td></tr><tr><td>GPT5.6-Luna</td><td>2.50</td><td>87.00</td><td>3.25</td><td>7.25</td><td>0.13</td><td>26.25</td><td>9.38</td><td>10.63</td><td>14.75</td><td>0.36</td><td>1.77</td><td>51.75</td></tr><tr><td>Avg.</td><td>14.38</td><td>65.53</td><td>4.03</td><td>16.08</td><td>3.05</td><td>19.33</td><td>6.00</td><td>4.63</td><td>12.80</td><td>5.53</td><td>2.55</td><td>63.08</td></tr><tr><td colspan="10">Open-source Models</td><td colspan="3"></td></tr><tr><td>DeepSeekV4-Pro</td><td>2.63</td><td>79.50</td><td>2.25</td><td>15.63</td><td>0.13</td><td>24.25</td><td>8.25</td><td>7.50</td><td>16.13</td><td>5.42</td><td>1.39</td><td>56.63</td></tr><tr><td>Qwen3.8-27B</td><td>0.38</td><td>68.50</td><td>3.75</td><td>27.38</td><td>0.00</td><td>25.38</td><td>7.88</td><td>5.50</td><td>19.75</td><td>9.86</td><td>1.54</td><td>55.13</td></tr><tr><td>GLM5.3-Flash</td><td>4.88</td><td>73.63</td><td>2.88</td><td>18.63</td><td>0.75</td><td>23.88</td><td>7.75</td><td>5.75</td><td>18.25</td><td>1.48</td><td>1.82</td><td>54.13</td></tr><tr><td>Qwen3.5-9B</td><td>20.63</td><td>69.00</td><td>4.38</td><td>6.00</td><td>9.63</td><td>22.50</td><td>7.75</td><td>13.00</td><td>11.75</td><td>6.37</td><td>0.96</td><td>41.50</td></tr><tr><td>GPT-OSS-20B</td><td>5.00</td><td>73.75</td><td>8.38</td><td>12.88</td><td>1.50</td><td>27.75</td><td>13.75</td><td>17.38</td><td>17.25</td><td>0.75</td><td>0.37</td><td>30.38</td></tr><tr><td>Avg.</td><td>6.70</td><td>72.88</td><td>4.33</td><td>16.10</td><td>2.40</td><td>24.75</td><td>9.08</td><td>9.83</td><td>16.63</td><td>4.77</td><td>1.21</td><td>47.55</td></tr></table>

Human Audit. 2 reviewers audit each scenario for causal support, timing, evidence, and feasible assistance within authority. A third reviewer adjudicates cases with exactly 1 initial approval. Scenarios that do not pass review are revised before model-based acceptance checks. Detailed procedures and results appear in Appendix C.

## 3.2 HALO-8K

Built on HALO-Arena, HALO-8K contains 8,000 human-annotated responses from 10 models across 2,160 checkpoints.

## 3.2.1 Overview

The 8 scenarios within a personal environment share its initial state $S _ { 0 }$ and have separate event sequences. For checkpoint $t = 1 , \dots , T$ , the runner independently resets the environment to $S _ { 0 }$ and replays only that scenario’s observed prefix $( e _ { 1 } , \ldots , e _ { t } )$ . Earlier agent interventions do not affect other checkpoints or scenarios. The agent receives protagonist context and the event prefix, using the newest timestamp as simulation time.

For human annotation, we sample 2 distinct checkpoints per model in each scenario, yielding 800 responses per model, with 200 per need and 80 per domain. Sampling balances event positions without using human verdicts. Details appear in Appendix D.

## 3.2.2 Annotation Protocol

20 annotators provide 1 verdict per response, keeping each scenario’s annotator fixed across models. Human verdicts and diagnosis use the event prefix, retrieved evidence, response, and execution result. Appendix E describes annotator preparation.

Action levels. Annotators confirm or correct the command-based action level pre-filled by the interface. L0 Observation observes or retrieves information without initiating an intervention. L1 Attention draws the user’s attention to relevant information or needs without carrying out the underlying task. L2 Preparation prepares an intervention or organizes internal information for later use without creating an external commitment. L3 Execution carries out an intervention that changes the user’s active commitments or creates external effects. Higher levels do not imply better assistance.

Human Verdict and Diagnosis. Annotators first give an acceptability verdict on whether a response provides grounded, useful help with appropriate timing and scope, accepting L0 when restraint is warranted. Acceptability is the proportion of accepted responses. These automatically receive Appropriate timing and intensity. Rejected L0 responses receive the human diagnosis Inaction for missing warranted help, without further diagnostic labels. For rejected L1–L3 responses, separate questions assess timing and intensity. Premature and Belated indicate intervention before and after its useful window, respectively. Insufficient indicates inadequate help, while Excessive indicates intervention beyond supported scope or authority. Appropriate denotes suitable timing or proportionate intensity. Annotators may also flag Other Issues, detailed in Appendix G.

Table 2: Results by proactive need type and domain. All values are percentages of category responses. Error abbreviations follow Table 1. Shading marks best , second , and third values within each group.
<table><tr><td rowspan="2">Category</td><td colspan="4">Action Levels</td><td rowspan="2">Inact.↓</td><td colspan="2">Timing Errors↓</td><td colspan="2">Intensity Errors↓</td><td rowspan="2">Acc (%)↑</td></tr><tr><td>L0</td><td>L1</td><td>L2</td><td>L3</td><td>Prem.</td><td>Bel.</td><td>Insuff.</td><td>Excess.</td></tr><tr><td colspan="10">Proactive Need Types</td></tr><tr><td>Blockage</td><td>12.05</td><td>66.35</td><td>4.35</td><td>17.25</td><td>3.15</td><td>21.50</td><td>7.95</td><td>8.70</td><td>13.85</td><td>54.35</td></tr><tr><td>Dependency</td><td>7.20</td><td>71.45</td><td>4.95</td><td>16.40</td><td>1.95</td><td>21.05</td><td>7.80</td><td>7.10</td><td>16.05</td><td>55.05</td></tr><tr><td>Opportunity</td><td>13.60</td><td>65.80</td><td>4.75</td><td>15.85</td><td>2.95</td><td>24.00</td><td>6.60</td><td>5.80</td><td>15.30</td><td>55.85</td></tr><tr><td>Risk</td><td>9.30</td><td>73.20</td><td>2.65</td><td>14.85</td><td>2.85</td><td>21.60</td><td>7.80</td><td>7.30</td><td>13.65</td><td>56.00</td></tr><tr><td colspan="10">Domains</td></tr><tr><td>Community</td><td>10.13</td><td>71.00</td><td>2.63</td><td>16.25</td><td>3.13</td><td>20.88</td><td>6.63</td><td>7.50</td><td>15.00</td><td>53.88</td></tr><tr><td>Education</td><td>15.00</td><td>65.50</td><td>2.50</td><td>17.00</td><td>2.38</td><td>22.13</td><td>9.00</td><td>5.88</td><td>15.38</td><td>58.50</td></tr><tr><td>Family</td><td>14.38</td><td>60.25</td><td>4.25</td><td>21.13</td><td>2.50</td><td>23.50</td><td>7.25</td><td>6.88</td><td>14.63</td><td>55.00</td></tr><tr><td>Health</td><td>10.00</td><td>77.38</td><td>7.00</td><td>5.63</td><td>3.13</td><td>21.38</td><td>9.75</td><td>9.13</td><td>11.88</td><td>53.25</td></tr><tr><td>Household</td><td>8.75</td><td>68.25</td><td>3.25</td><td>19.75</td><td>2.25</td><td>18.75</td><td>6.00</td><td>7.88</td><td>13.38</td><td>58.88</td></tr><tr><td>Finance</td><td>9.63</td><td>71.88</td><td>8.50</td><td>10.00</td><td>3.38</td><td>19.38</td><td>8.25</td><td>8.38</td><td>12.50</td><td>55.25</td></tr><tr><td>Recreation</td><td>7.63</td><td>80.25</td><td>1.50</td><td>10.63</td><td>2.38</td><td>22.00</td><td>6.50</td><td>8.75</td><td>14.25</td><td>58.25</td></tr><tr><td>Social</td><td>10.88</td><td>63.00</td><td>3.75</td><td>22.38</td><td>3.13</td><td>26.75</td><td>7.38</td><td>6.38</td><td>18.63</td><td>50.88</td></tr><tr><td>Travel</td><td>10.75</td><td>71.13</td><td>1.38</td><td>16.75</td><td>2.75</td><td>26.25</td><td>7.25</td><td>5.38</td><td>15.75</td><td>53.25</td></tr><tr><td>Work</td><td>8.25</td><td>63.38</td><td>7.00</td><td>21.38</td><td>2.25</td><td>19.38</td><td>7.38</td><td>6.13</td><td>15.75</td><td>56.00</td></tr></table>

Resource Consumption. For each annotated response, we sum reasoning tokens across retained calls and count completed retrieval rounds.

## 4 Experiments

## 4.1 Settings

## 4.1.1 Evaluated Models

We evaluate 5 proprietary models: Gemini3.8- Flash and Gemini3.1-Pro (Google DeepMind, 2026b,a), Claude-Opus5.5 (Anthropic, 2026), Grok4.7 (xAI, 2026), and GPT5.6-Luna (OpenAI, 2026). The 5 open-source models are DeepSeekV4- Pro (DeepSeek, 2026), GLM5.3-Flash (Z.ai, 2026), Qwen3.5-9B (Qwen Team, 2026a), Qwen3.8-27B (Qwen Team, 2026b), and GPT-OSS-20B (OpenAI, 2025).

## 4.1.2 Annotation Platform

We deploy an online annotation platform on Streamlit Community Cloud, with a Streamlit frontend, Python backend, and Supabase PostgreSQL storage. Its 3 workspaces support scenario audit, response evaluation, and administration. The 8,000 responses are annotated in 2 sessions of 4,000 each. Annotators inspect assigned items and save or revise annotations centrally. Interface screenshots appear in Appendix J.

## 4.2 Main Results

## 4.2.1 Model Comparison

Table 1 compares acceptability, action levels, recorded consumption, and timing and intensity errors on the same 800 responses per model.

Acceptability. Scores range from 30.38% for GPT-OSS-20B to 72.88% for Gemini3.8-Flash, a 42.50-point gap. DeepSeekV4-Pro leads opensource models with 56.63% acceptability. The score range reflects substantial variation in acceptability under the same annotation protocol, while even the strongest model leaves over 25% of responses unacceptable. Proprietary models average 63.08%, versus 47.55% for open-source models, a 15.53-point gap. These means also conceal uneven performance within each group. The 3 strongest open-source models lie within 2.50 points of each other, whereas the full group spans 26.25 points. Their lower mean thus reflects a weak tail as well as a gap at the top. The groups overlap: DeepSeekV4- Pro exceeds both Gemini3.1-Pro and GPT5.6-Luna, so group membership provides an incomplete account of individual quality. These scores cover completed, human-annotated responses under the recorded inference settings, excluding transport and parsing failures.

Action choices. Attention (L1) is the most frequent level for every model, accounting for 69.20% of all responses. L1 exceeds 50% of responses in 9 models and ranges from 48.00% for Claude-Opus5.5 to 87.00% for GPT5.6-Luna. Preparation (L2) remains rare at 4.18%, while execution (L3) accounts for 16.09%. This concentration makes the timing and usefulness of L1 responses central to the aggregate result. At the group level, L3 shares are almost identical at about 16.1%, despite the 15.53-point acceptability gap. The main observed policy difference lies in observation and attention: proprietary models choose L0 about 2 times as often, while open-source models choose L1 more often. Execution frequency alone cannot explain the observed acceptability gap.

Resource Consumption. Open-source models use fewer retrieval rounds at similar mean reasoning usage. They average 4.77k reasoning tokens and 1.21 retrieval rounds per response, compared with 5.53k and 2.55 for proprietary models. The reasoning-token means differ by about 14%, whereas open-source models use about 53% fewer retrieval rounds. Lower recorded retrieval use accompanies lower acceptability, so resource savings should be assessed alongside the quality of the resulting responses. Within the proprietary group, second-ranked Claude-Opus5.5 uses about 23% of the recorded reasoning tokens of the leading Gemini3.8-Flash. Strong performance thus occurs at different levels of recorded consumption. These profiles support comparing quality and resource use jointly under the recorded inference settings and provider accounting conventions.

## 4.2.2 Need and Domain Profiles

Table 2 summarizes action levels, errors, and acceptability by proactive need and domain. Acceptability ranges from 54.35% to 56.00% across needs and from 50.88% to 58.88% across domains. Finance has more L2 preparation than Travel, and Social more L3 execution than Health, with both contrasts holding in every model. These profiles describe sampled scenarios and available action opportunities. Model-specific and joint profiles appear in Appendix F.

![](images/24d9a5045891f8397b3d240ccd22ccf1359e66b1febaed40b115a42aa979797c.jpg)  
Figure 3: Over-proactivity among rejected responses.

## 4.2.3 Error Analysis

We identify over-proactivity as Premature timing or Excessive intensity, counting each response once when either diagnosis applies. As shown in Figure 3, it occurs in 2,188 of 3,575 rejected responses (61.20%). Model-wise shares range from 44.23% to 79.66%, exceeding 50% of rejections in 6 of 10 models. Across all responses, Premature timing is almost 3 times as common as Belated timing, while Excessive intensity is about 2 times as common as Insufficient. These 2 asymmetries indicate more frequent early or excessive intervention. This pattern holds across all 4 proactive need types and all 10 domains, with over-proactivity accounting for more than 50% of rejections in each category.

The 2 highest-acceptability models also have the lowest Premature and Excessive rates, associating stronger overall performance with fewer premature or excessive interventions. Inaction alone gives a different picture: Qwen3.8-27B records 0% Inaction but reaches only 55.13% acceptability and chooses L0 on just 0.38% of responses. Because Inaction applies only to rejected L0 responses and is reported over all responses, a policy that rarely chooses observation has few opportunities to incur this error. Action-level shares therefore provide essential context for interpreting individual error rates. Over-proactivity is the dominant failure mode of proactive agents.

## 5 Further Pattern Analysis

We examine action and restraint, need-dependent strategies, evidence acquisition, and residual failures despite appropriate timing.

![](images/dbbfa57b8319ba1df526a676062534fccbfd09908cf8d59300fa4e2ced179a67.jpg)

![](images/b0539ef6c2309d2405c260d5a91043cfcda381066bf521aa0c88167be25b1465.jpg)  
Figure 4: Paired intervention rates and acceptability.

## 5.1 Action and Restraint

We compare intervention choices and acceptability at shared checkpoints.

More frequent intervention can accompany lower acceptability when warranted restraint losses outweigh accepted action gains.

Figure 4 compares Qwen3.5-9B to Qwen3.8- 27B at 201 shared checkpoints and Gemini3.8- Flash to Gemini3.1-Pro at 623. Pairs match instance, scenario, event, and annotator without filtering verdicts. Intervention frequency, the L1–L3 share, rises by 27.36 and 28.89 points, respectively, yet acceptability rises by 14.43 points for Qwen and falls by 14.77 for Gemini. The Gemini loss combines a 5.94-point gain from accepted interventions with a 20.71-point decline from accepted L0. Both changes use all matched responses as their denominator and sum to the acceptability change, so the intervention gain reflects a frequency change, not conditional success.

The error changes clarify the different outcomes: Qwen reduces Inaction-or-Insufficient by 17.91 points while increasing Premature-or-Excessive by 5.97, whereas Gemini increases the latter by 21.51. Matching fixes the observed prefix and annotator, but configurations and retrieved evidence can differ, with Qwen’s version and parameter count changing together. The comparisons therefore do not isolate the causal effect of intervening more. Restraint can also follow substantial inspection: Flash’s 182 accepted L0 responses average 16.74 commands, versus 14.65 for 401 accepted interventions. These groups cover different checkpoints, but show that L0 records final commitment rather than inspection effort.

## 5.2 Needs and Strategies

Similar acceptability across need types can mask different balances of accepted intervention and warranted restraint. Figure 5 contrasts Opportunity and

![](images/1c53572a619f8835ba6c66c5c14c21f665ea1b4d7ff986a26787b9fde345a16a.jpg)  
Figure 5: L0 choices by model on the left and accepted action and restraint contributions on the right.

Dependency. Despite similar acceptability (55.85% vs. 55.05%), models are more likely to refrain from intervention for Opportunity: L0 accounts for 13.60% of Opportunity responses, compared with 7.20% for Dependency. Accordingly, accepted L0 contributes 10.65 points to Opportunity acceptability and 5.25 points to Dependency acceptability, while accepted interventions contribute 45.20 and 49.80 points, respectively.

All 10 models choose L0 more often for Opportunity than Dependency, although the size of this difference varies substantially. Gemini3.8-Flash chooses L0 for 39.00% of Opportunity responses and 17.00% of Dependency responses, whereas Qwen3.8-27B chooses L0 for 0.50% and 0.00%, respectively. These differences do not imply fixed action requirements for each need type: need labels characterize entire scenarios, while the appropriate action depends on the checkpoint and its available evidence. Appendix F reports model-specific acceptability profiles.

## 5.3 Evidence Acquisition

We relate retrieval effort and local content access to human verdicts and diagnosis.

Evidence acquisition is associated with more   
adequate proactive assistance and higher   
acceptability in human verdicts.

Figure 6 shows a positive association between mean retrieval rounds and acceptability across 10 models $( r ~ = ~ 0 . 8 8 5 , ~ \rho ~ = ~ 0 . 8 3 0 )$ . To examine help at fixed commitment, we compare content access among 5,536 L1 responses. Access means obtaining non-empty database records or file text, while the no-content group includes no inspection, filenames or schemas alone, and empty results. Acceptability is 59.0% with content versus 40.8% without it, with Insufficient rates of 5.2% and 15.5%, respectively. Content access is thus associated with more adequate help within L1.

![](images/fb1140ed2ded7763797623c78c62b1530600ff1915e415bd2012a9f166aaee1f.jpg)

![](images/c4d0614275b36ac88aaa17fc4ef0c24843ed5273f384cffcd44f5b4499cfcbd8.jpg)  
Figure 6: Mean retrieval rounds and acceptability across models (left), and L1 response proportions with and without local content access (right).

After excluding singleton checkpoints, linear probability models with checkpoint fixed effects and model indicators use 5,232 L1 responses at 1,730 checkpoints. Content access corresponds to +9.25 points in acceptability and −5.86 in Insufficient. These associations persist when restricting the comparison to responses that inspect the environment, so the contrast is not confined to responses with versus without inspection.

## 5.4 Timing and Scope

We examine intensity failures within rejected interventions with Appropriate timing.

Among well-timed rejections, reminders tend to provide insufficient help, while executions tend to exceed acceptable scope.

Figure 7 examines 991 rejected interventions with Appropriate timing. Among them, 329 of 623 L1 responses (52.81%) are Insufficient and 238 of 312 L3 responses (76.28%) are Excessive. The opposite intensity patterns hold within every model, although intensity does not explain all failures: 282 timely L1 rejections retain Appropriate intensity. At the same checkpoints, 229 of the Insufficient L1 responses (69.60%) have an accepted L1 counterpart from another model, and 178 Excessive L3 responses (74.79%) have an accepted L1 or L2 counterpart. Both counterpart patterns cover most corresponding rejected responses within each need. These response-level shares depend on sampled candidates and do not measure repair success.

Paired examples clarify the role of content and permission. At a shared household checkpoint, Gemini3.8-Flash’s accepted L1 supplies the no-pry instruction and a concrete dispatch route. GPT-OSS-20B’s Insufficient reminder instead suggests freeing the panel and supplies no usable service contact. At a family checkpoint, the stored authorization permits a message containing only “kitchen sneakers.” Qwen3.8-27B sends that phrase and is accepted, while Claude-Opus5.5 adds a name, location, and Saturday-morning request and receives an Excessive diagnosis. Both outputs are L3, and both models retrieve the wording restriction. The same action level can thus receive different verdicts: useful detail can improve L1, while respecting narrow authorization can make L3 acceptable.

![](images/58c9ae2519f518b440a323f2ac32e290d95bb568ed402c8d044398eea1fe5bfb.jpg)  
Figure 7: Intensity errors among appropriately timed but rejected interventions.

## 6 Conclusion

HALO evaluates proactive assistance through executable environments, human verdicts, and human diagnosis. HALO-8K contains 8,000 humanannotated responses from 10 models across 400 scenarios and 2,160 checkpoints. Over-proactivity dominates the observed rejections. Further analyses distinguish useful action from warranted restraint, relate evidence acquisition to acceptability, and identify contrasting intensity errors among well-timed rejections. These patterns expose calibration problems obscured by aggregate acceptability. HALO-Arena provides operational applications and separate structured and unstructured evidence stores. Repeated retrieval lets an agent inspect relevant evidence before deciding whether to notify, prepare, execute, or remain silent.

Acceptability peaks at 72.88% for proprietary and 56.63% for open-source models. HALO records retrieved evidence and executed commands, enabling comparison of human verdicts and diagnosis for responses produced at identical checkpoints. The practical targets differ: timely reminders need enough grounded detail to be useful, while execution must respect the scope of available permission. These results motivate better evidence gathering and response calibration. More frequent action or higher commitment alone does not establish better assistance. Systems should be compared jointly on acceptable action, warranted restraint, residual intensity errors, and inspection effort.

## Limitations

Real-world validity. HALO uses constructed scenarios and simulated apps to study proactive decisions under controlled conditions. Its designed need distribution does not estimate how often those needs or failure modes occur in everyday use. Checkpoint evaluation also omits users’ responses and the cumulative effects of repeated interventions, so performance may not transfer directly to sustained deployment.

Human personalization. Human verdicts depend on preferences for initiative, useful detail, and commitment. Shared criteria ground judgments in personal context and authority, but borderline responses can be useful to some users and intrusive to others. The annotators’ broadly similar backgrounds further limit the preferences represented, so aggregate acceptability should not be treated as a universal user preference.

Evaluation granularity. Action levels group interventions by commitment, but actions at the same level can differ in reversibility and user burden. Human verdicts and diagnostic labels identify acceptable help and common failures without ranking acceptable alternatives by usefulness. Consequently, aggregate acceptability does not capture the full tradeoff between benefit, burden, and intervention risk.

## References

Anthropic. 2026. Claude Opus 5.5. https://www.anthropic.com/claude-opus-5-5.

DeepSeek. 2026. DeepSeek-V4 preview. https://deepseek.com/en/news/v4-preview/.

Lei Ding, Bin He, Chenguang Wang, and Yang Liu. 2026. ProActor: Timing-aware reinforcement learning for proactive task scheduling agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18257–18303. Association for Computational Linguistics. https://aclanthology.org/2026.acl-long.832/.

Google DeepMind. 2026a. Gemini 3.1 Pro model card. https://deepmind.google/models/model-cards/ gemini-3-1-pro/.

Google DeepMind. 2026b. Gemini 3.8 Flash model card. https://deepmind.google/models/model-cards/ gemini-3-8-flash/.

Guanzhen Li, Liangming Pan, and Leye Wang. 2026. ProEvent: An event-centric benchmark for proactive agents. ArXiv preprint 2607.17701. https://arxiv.org/abs/2607.17701.

Xun Liu, Yi Evie Zhang, Vira Kasprova, Parisa Rabbani, Pardis Sadat Zahraei, Tianyu Zhang, Ali Ebrahimpour-Boroojeny, and Varun Chandrasekaran. 2026. AgentAbstain: Do LLM agents know when not to act? ArXiv preprint 2607.10059. https://arxiv.org/abs/2607.10059.

Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, Qinyu Luo, Yesai Wu, Huadong Wang, Xin Cong, Zhong Zhang, Yankai Lin, Weiwen Liu, Yasheng Wang, Zhiyuan Liu, Fangming Liu, and Maosong Sun. 2024. Proactive agent: Shifting LLM agents from reactive responses to active assistance. ArXiv preprint 2410.12361. https://arxiv.org/abs/2410.12361.

Deepak Nathani, Cheng Zhang, Chang Huan, Jiaming Shan, Yinfei Yang, Alkesh Patel, Zhe Gan, William Yang Wang, Michael Saxon, and Xin Eric Wang. 2026. Proactive agent research environment: Simulating active users to evaluate proactive assistants. ArXiv preprint 2604.00842. https://arxiv.org/abs/2604.00842.

Jio Oh, Seunghyun Do, Young-Jun Lee, Steven Euijong Whang, and Dongyeop Kang. 2026. Foundations of proactive agents: Principles, technical layers, and Proactivity-Gym. ArXiv preprint 2609.37267. https://arxiv.org/abs/2609.37267.

OpenAI. 2025. gpt-oss-20b: Model documentation. https://developers.openai.com/api/docs/models/gptoss-20b.

OpenAI. 2026. GPT-5.6 Luna: Model documentation. https://developers.openai.com/api/docs/models/gpt-5.6-luna.

Gil Pasternak, Dheeraj Rajagopal, Julia White, Dhruv Atreja, Matthew Thomas, George Hurn-Maloney, and Ash Lewis. 2025. Beyond reactivity: Measuring proactive problem solving in LLM agents. ArXiv preprint 2510.19771. https://arxiv.org/abs/2510.19771.

Qwen Team. 2026a. Qwen3.5-9B model card. https://huggingface.co/Qwen/Qwen3.5-9B.

Qwen Team. 2026b. Qwen3.8-27B model card. https://huggingface.co/Qwen/Qwen3.8-27B.

Yuanbo Tang, Huaze Tang, Tingyu Cao, Lam Nguyen, Anping Zhang, Xinwen Cao, Chunkang Liu, Wenbo Ding, and Yang Li. 2026. ProAgentBench: Evaluating LLM agents for proactive assistance with realworld data. ArXiv preprint 2602.04482. https://arxiv.org/abs/2602.04482.

xAI. 2026. Grok 4.7: Model documentation. https://docs.x.ai/developers/models/grok-4.7.

Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, and Zhenyu Yan. 2025. ContextAgent: Context-aware proactive LLM agents with open-world sensory perceptions. In Advances in Neural Information Processing Systems, volume 38. https://doi.org/10.52202/085713-5582.

Qinglong Yang, Haoming Li, Haotian Zhao, Xiaokai Yan, Jingtao Ding, Fengli Xu, and Yong Li. 2026. FingerTip 20K: A benchmark for proactive and personalized mobile LLM agents. In The Fourteenth International Conference on Learning Representations. https://openreview.net/forum?id=n3iFV0gLMc.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. ArXiv preprint 2406.12045. https://arxiv.org/abs/2406.12045.

Z.ai. 2026. GLM-5.3-Flash model card. https://huggingface.co/zai-org/GLM-5.3-Flash.

Haoran Zhang, Luxin Xu, Zhilin Wang, Runquan Gui, Shunkai Zhang, Haodi Lei, Zihao He, Bingsu He, Chicheng Qin, Tong Zhu, Xiaoye Qu, Yang Yang, Yu Cheng, and Yafu Li. 2026. π-Bench: Evaluating proactive personal assistant agents in long-horizon workflows. ArXiv preprint 2605.14678. https://arxiv.org/abs/2605.14678.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2023. WebArena: A realistic web environment for building autonomous agents. ArXiv preprint 2307.13854. https://arxiv.org/abs/2307.13854.

Table 3: Selected benchmark capabilities. Checks denote explicit coverage. Dashes denote features not reported in the evaluated protocol.
<table><tr><td>Benchmark</td><td colspan="3">Environment</td><td colspan="3">Evaluation</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Func. Apps DB Docs Human Verdicts Human Diagnosis Consumption</td><td></td></tr><tr><td>PROBE (Pasternak et al., 2025)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ProAgentBench (Tang et al., 2026)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PARE-Bench (Nathani et al., 2026)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>π-Bench (Zhang et al., 2026)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Proactivity-Gym (Oh et al., 2026)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HALO (Ours)</td><td></td><td></td><td></td><td></td><td>L</td><td></td></tr></table>

## A Benchmark Comparison

Table 3 contrasts selected capabilities. Func. Apps denotes state-changing applications, DB denotes SQL queries, and Docs denotes document retrieval. PROBE’s retrieval marks cover agent baselines, while its action score measures selection and parameters. Human Verdicts include acceptance annotation, judge-validation decisions, and user preferences. Human Diagnosis covers per-response timing, intensity, and other issue labels for rejections. Consumption includes reported token usage, tool actions, interaction rounds, latency, or cost, including PARE-Bench’s read actions and π-Bench’s interaction turns.

## B Applications and Commands

Table 4 lists app functions and their automatic action levels, with Assistant denoting the usernotification surface. No action and interaction-only responses map to L0. Of the 19 action commands, 6 map to L1, 6 to L2, and 7 to L3, providing comparable coverage across intervention levels.

## C Construction Details

## C.1 Automated Checks

Candidates pass a deterministic Validator followed by an LLM Reviewer. Each scenario permits up to 5 candidates, and only accepted candidates commit initialization facts and events. Table 5 covers 568 persisted, parseable candidates, excluding malformed outputs and transport retries. Reviewer rates condition on 509 Validator passes, while overall rates use all 568 candidates. Revised candidates are counted separately. Counts are historical project records because the original candidate archive is no longer retained.

After all 8 scenarios of an instance pass, the Noise Creator adds settled, causally irrelevant records to shared initialization, expanding the inspection space without introducing unresolved activities.

## C.2 Human Audit

Table 6 reports individual votes, review pairs, and adjudication outcomes. Vote rates use 800 verdicts, while review-pair and adjudication rates use 400 scenarios. Scenarios approved by both reviewers were retained. Cases with 2 Needs Revision votes or 1 Needs Revision and 1 Reject vote entered revision directly. Revisions followed reviewer feedback, prioritizing scenario corrections. Shared environments were adjusted only when these corrections were insufficient, to limit effects on unrelated scenarios.

## D Checkpoint Selection

Table 7 reports checkpoint counts by scenario length. Within each 5-model batch, randomized maximum-flow sampling selects 2 distinct completed checkpoints per model and scenario, without using verdicts. The resulting 10 responses cover all event positions with counts differing by at most 1. Table 8 reports event positions and diagnostics. Wasserstein distances use normalized position $( t - 1 ) / ( T - 1 )$ for event index t and trajectory length T.

A linear probability model with checkpoint fixed effects and model indicators preserves the ranking, with a maximum mean-centered adjustment of 1.69 points. This suggests limited selection sensitivity, without establishing equal difficulty or resolving small score gaps.

## E Annotation Consistency

2077AI oversaw annotator recruitment, screening, management, and ongoing follow-up to support consistent application of the annotation guidelines.

Table 4: Agent functions by app and category. Superscripts indicate automatically inferred intensity levels, provided to guide human verification and reduce verification time.
<table><tr><td>App</td><td>Interaction commands (29)</td><td>Action commands (19)</td></tr><tr><td>Email</td><td>get_email, list_emails, search_emails, list_drafts</td><td>flag_emailL1, create_draftL2, send_emailL3</td></tr><tr><td>Calendar</td><td>get_event, list_events, find_conflicts, get_event_draft, list_event_drafts, get_reminder,list_reminders</td><td>create_reminderL1, create_event_draftL2, create_eventL3 , update_eventL3, delete_eventL3</td></tr><tr><td>Chat</td><td>get_conversation, list_conversations, get_message, list_messages, search_messages, list_drafts</td><td>pin_conversationL1, create_draftL2, send_message L3</td></tr><tr><td>Contacts</td><td>get_contact, list_contacts, search_contacts</td><td>flag_contactL1,create_contactL2 update_contactL2</td></tr><tr><td>Wallet</td><td>get_wallet, get_transaction, list_transactions,list_drafts</td><td>flag_transactionLl, ,create_draftL2, payL3, refund_transaction L3</td></tr><tr><td>Database</td><td>get_schema, execute</td><td></td></tr><tr><td>Files</td><td>get_file,list_files, search_files</td><td></td></tr><tr><td>Assistant</td><td></td><td>notify_userL1</td></tr></table>

Table 5: Initial pipeline pass and fail rates. Reviewer rates are conditional on Validator acceptance.
<table><tr><td>Stage</td><td>Pass Total Pass (%) Fail (%)</td></tr><tr><td>Validator</td><td>509 568 89.61 10.39</td></tr><tr><td>Reviewer 400 509</td><td>78.59 21.41</td></tr><tr><td>Overall 400</td><td>568 70.42 29.58</td></tr></table>

Table 7: Checkpoint distribution by scenario length.
<table><tr><td>Events</td><td>Scenarios</td><td>Checkpoints</td></tr><tr><td>4</td><td>14</td><td>56</td></tr><tr><td>5</td><td>212</td><td>1,060</td></tr><tr><td>6</td><td>174</td><td>1,044</td></tr><tr><td>Total</td><td>400</td><td>2,160</td></tr></table>

Table 6: Human audit and adjudication. Review pairs are unordered.
<table><tr><td>Record Count Rate (%)</td></tr><tr><td>Audit Votes</td></tr><tr><td>Approve 593 74.13 21.88</td></tr><tr><td>Needs Revision 175</td></tr><tr><td>Reject 32 4.00</td></tr><tr><td>Review Pairs</td></tr><tr><td>Approve + Approve 214 53.50 34.50</td></tr><tr><td>Approve + Needs Revision 138</td></tr><tr><td>Approve + Reject 27 6.75</td></tr><tr><td>Needs Revision + Needs Revision 16 4.00</td></tr><tr><td>Needs Revision + Reject 5 1.25</td></tr><tr><td>Reject + Reject 0 0.00</td></tr><tr><td>Adjudication</td></tr><tr><td>Approved by third reviewer 75 18.75</td></tr><tr><td>Retained after adjudication 289 72.25</td></tr><tr><td>Selected for revision 111 27.75</td></tr></table>

A shared briefing and dry run familiarized 20 annotators with the rubric and interface. Annotators had broadly similar demographic backgrounds.

## F Pairwise Comparison

Table 9 and Figures 8 and 9 report model–need, model–domain, and need–domain acceptability, respectively. Model–domain outlines mark the best unrounded scores, including ties. Need orderings vary by domain: Opportunity exceeds Risk in Fi-

Table 8: Selected event positions and selection diagnostics. pp denotes percentage points.
<table><tr><td rowspan="2">Model</td><td colspan="5">Selected positions (%)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>Gemini3.8-Flash</td><td>15.75</td><td>19.25</td><td></td><td>19.75 21.63</td><td>17.25</td><td>6.38</td></tr><tr><td>Claude-Opus5.5</td><td></td><td>17.6319.5018.8819.5017.25 7.25</td><td></td><td></td><td></td><td></td></tr><tr><td>Grok4.7</td><td></td><td></td><td></td><td>17.7516.6319.25 20.1319.00 7.25</td><td></td><td></td></tr><tr><td>Gemini3.1-Pro</td><td></td><td></td><td></td><td>15.75 20.00 21.3819.0017.63 6.25</td><td></td><td></td></tr><tr><td>GPT5.6-Luna</td><td></td><td></td><td></td><td></td><td>19.3819.3816.8818.7517.25 8.38</td><td></td></tr><tr><td>DeepSeekV4-Pro 18.00 18.50 18.25 19.88 17.88 7.50</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.8-27B</td><td></td><td></td><td></td><td></td><td>18.2516.75 20.8819.7517.25 7.13</td><td></td></tr><tr><td>GLM5.3-Flash Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td>17.25 20.3819.8817.8818.38 6.25</td><td></td></tr><tr><td>GPT-OSS-20B</td><td>17.8818.8817.88</td><td></td><td></td><td>18.63</td><td>17.75 20.1316.75 20.3816.888.13</td><td>19.88 6.88</td></tr><tr><td>All models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>17.5418.9418.98</td><td></td><td>19.55</td><td></td><td>17.86 7.14</td></tr><tr><td>Selection diagnostic</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cramér&#x27;s V</td><td></td><td></td><td></td><td></td><td>Value</td><td></td></tr><tr><td>Max. pairwise Wasserstein distance</td><td></td><td></td><td></td><td></td><td>0.029</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.0288</td><td></td></tr><tr><td>Max. centered score adjustment</td><td></td><td></td><td></td><td></td><td>1.69 pp</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Max. pairwise score-gap change</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>3.27 pp</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Adjusted model ranking</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Unchanged</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

nance, with the reverse in Travel.

Table 9: Acceptability (%) by model and proactive need type.
<table><tr><td colspan="4">Model Blockage Dependency Opportunity Risk</td></tr><tr><td>Gemini3.8-Flash</td><td>73.0</td><td>70.0 73.5</td><td>75.0</td></tr><tr><td>Claude-Opus5.5</td><td>69.5</td><td>68.5 70.0</td><td>71.0</td></tr><tr><td>Grok4.7</td><td>63.5</td><td>65.5 64.0</td><td>68.0</td></tr><tr><td>DeepSeekV4-Pro</td><td>53.5</td><td>54.0 57.5</td><td>61.5</td></tr><tr><td>Gemini3.1-Pro</td><td>54.0</td><td>57.0 54.0</td><td>58.0</td></tr><tr><td>Qwen3.8-27B</td><td>54.0</td><td>59.5 52.5</td><td>54.5</td></tr><tr><td>GLM5.3-Flash</td><td>54.5</td><td>51.0 53.5</td><td>57.5</td></tr><tr><td>GPT5.6-Luna</td><td>49.0</td><td>53.0 50.5</td><td>54.5</td></tr><tr><td>Qwen3.5-9B</td><td>43.0</td><td>40.5 49.0</td><td>33.5</td></tr><tr><td>GPT-OSS-20B</td><td>29.5</td><td>31.5 34.0</td><td>26.5</td></tr></table>

![](images/9c4687677a6faa21a6d4ac4f7c2cbfc04794ccaf67560e38cfc0cb533f00934e.jpg)  
Figure 8: Acceptability by model and domain. Outlines mark domain leaders.

## G Other Issues

Other Issues are recorded for 1,000 of 3,575 rejected responses (27.97%), corresponding to 12.50% of all 8,000 responses. Table 10 reports the category distribution. Each flagged response has 1 recorded category, which can accompany timing or intensity errors. Others requires a written explanation. We retain the exported categories without reclassifying free-text comments. Unsupported Evidence and Irrelevant Action together account for 79.40% of Other Issues flags, linking human diagnosis to grounding and relevance. These categories supplement timing and intensity by distinguishing unsupported content from actions that do not address the current need.

![](images/a0c168881b64769e76b3c8ff5a5ec4a71f035f9ec658026dd9cd6e7f51c65e86.jpg)  
Figure 9: Acceptability by need and domain.

Table 10: Other Issues distribution. Within and Rejected are percentages of the 1,000 responses with Other Issues and all 3,575 rejected responses, respectively.
<table><tr><td colspan="4">Category Count Within (%) Rejected (%)</td></tr><tr><td>Unsupported Evidence</td><td>503</td><td>50.30</td><td>14.07</td></tr><tr><td>Irrelevant Action</td><td>291</td><td>29.10</td><td>8.14</td></tr><tr><td>Factual Contradiction</td><td>128</td><td>12.80</td><td>3.58</td></tr><tr><td>Others</td><td>78</td><td>7.80</td><td>2.18</td></tr></table>

## H Potential of HALO

We adapt Qwen3.5-9B and Qwen3.5-4B with nonthinking LoRA to predict human verdicts and diagnostic labels from 7,600 training responses. The 400 validation items, 1 per scenario, share scenarios with training and select the lowest-loss checkpoint. Both models train for 1 epoch with effective batch size 16, using microbatch size and gradient accumulation of 8 and 2 for 9B, and 2 and 8 for 4B. Original and adapted models use identical prompts and decoding. Intensity excludes nulls, while Joint requires all 3 labels, including nulls, to match. Changes use unrounded scores.

Table 11 shows verdict accuracy gains of 7.50 and 3.50 points for 9B and 4B, alongside joint accuracy declines of 11.00 and 28.75 points. Verdict gains can therefore mask diagnostic regressions. These are exploratory validation comparisons, not independent test results, and do not isolate effects of class weighting or model size.

## I Prompts

We provide prompt templates.

Table 11: Qwen3.5 judge accuracy (%).
<table><tr><td>Measure</td><td>N</td><td>Original LoRA</td><td></td><td>Change</td></tr><tr><td colspan="5">Qwen3.5-9B</td></tr><tr><td>Human verdict</td><td>t 400</td><td>48.50</td><td>56.00</td><td>+7.50</td></tr><tr><td>Timing</td><td>400</td><td>61.25</td><td>57.25</td><td>-4.00</td></tr><tr><td>Intensity</td><td>389</td><td>57.07</td><td>41.90</td><td>-15.17</td></tr><tr><td>Joint</td><td>400</td><td>33.75</td><td>22.75</td><td>-11.00</td></tr><tr><td colspan="5">Qwen3.5-4B</td></tr><tr><td>Human verdict</td><td>400</td><td>51.25</td><td>54.75</td><td>+3.50</td></tr><tr><td>Timing</td><td>400</td><td>66.50</td><td>60.00</td><td>-6.50</td></tr><tr><td>Intensity</td><td>389</td><td>67.10</td><td>27.76</td><td>-39.33</td></tr><tr><td>Joint</td><td>400</td><td>44.75</td><td>16.00</td><td>-28.75</td></tr></table>

## I.1 Agent Prompt

You are a proactive personal Agent in a   
simulated environment.   
At this checkpoint, decide whether to keep observing, inspect local information,   
or proactively help. Judge only the observed Event prefix; checkpoints are   
independent, so no earlier Agent behavior is part of the state. Use only observed   
Events and inspection results. Do not infer future Events or invent records.   
Intervene only when it is useful now. Choose an intervention that is effective   
for the current need, supported by the   
available evidence, and safe. This   
judgment is internal and must not be described in the JSON.

Events may reveal the need without containing   
every fact required for a good   
response. When a possible intervention depends   
on a qualification, prior   
record, identifier, procedure, constraint,   
preference, inventory fact, or safe   
next step, inspect the local SQLite database   
and/or TXT Files before responding.   
Do not assume that communications and calendar   
entries are the complete source   
of truth, and do not invent a missing detail   
instead of retrieving it.

Use assistant.notify\_user when a grounded   
message to the Instance user would   
provide useful help now. Explain the relevant   
finding or concrete next step;   
merely restating Events or giving generic   
advice is insufficient. The message   
does not obtain a user reply or approval, or   
carry out the operations it   
suggests. Never claim to have completed an   
operation that was not performed.

The time field of the newest observed Event is   
the current simulation time. The   
runner supplies operation timestamps for   
Action commands whose signatures omit

them. App record timestamps and Calendar   
business times do not replace the   
Event time. Ignore the real wall-clock date   
and any model training date; an   
Event is not "in the past" merely because its   
absolute date precedes the real   
date outside this episode.   
Return one compact JSON object in exactly one   
of these forms:   
"decision": "inspect",   
"content": [{"app":"...","function":"...","   
parameter":...}]   
or   
"decision": "final",   
"content": null | {"app":"...","function":" <sub>⌋</sub>   
...","parameter":...}

Rules:   
- inspect: content must be one non-empty list   
of private information-gathering   
commands from the Inspect commands below.   
The runner executes these commands   
and returns their results at the next round.   
You may repeat retrieval as often   
as the evidence requires; there is no fixed   
retrieval limit.   
final: content must be null for no action or   
exactly one intended command   
from the Action commands below. The command   
is executed once at the isolated   
checkpoint to verify its result. Never   
return multiple final   
commands or a standalone text response.   
- Null final content applies only to this   
checkpoint.   
Instance context:   
<<INSTANCE>>   
Observed Events through the current   
checkpoint:   
<<EVENTS>>   
Inspection results from this checkpoint:   
<<TOOL\_RESULTS>>   
Agent-facing App surface:   
<<TOOLS>>

## I.2 Construction Prompts

## I.2.1 Instance Creator

You create base Instances for a proactive-agent benchmark.

Given a broad Domain, define one coherent   
person and a concrete part of that   
person's life. The Instance should be narrower than the Domain while supporting   
eight meaningfully different Scenarios across several recurring activities.   
Create stable background state, not a   
proactive need.

Keep the setting understandable to a general reader and valid throughout calendar year <<BENCHMARK\_YEAR>>. Make it materially different from existing Instances in role, responsibilities, workflows, and record design.

Return raw JSON only, without Markdown or code fences. Return one JSON object with exactly these fields: 1. instance\_id: copy the assigned ID. 2. instance\_description: roughly 100 words

total, written as four concise, newline-separated items using exactly these labels: Setting, Scope, Resources, and Exclusions. Cover the setting, recurring activity areas, available records, and Instance boundaries needed for later construction. Format every item as \`Label: text\` without bullets or Markdown.

total, written as four concise, newline-separated items using exactly these labels: Identity, Responsibilities, Authority, and Restrictions. Define one protagonist whose full name is unique in the benchmark and the stable personal context relevant across its Scenarios. Later Scenario data will use that name verbatim as user\_id. Format every item as \`Label: text\` without bullets or Markdown.

Define 2-3 empty SQLite tables with at most 5 columns each, including one table   
that can hold protagonist-related records; define 2-3 reusable empty TXT Files;   
and initialize exactly one zero-balance Wallet. The Instance establishes only   
reusable structure: use CREATE statements only, add no database rows, and put   
no content in the Files. Scenario-specific facts and File content belong in   
later Scenario initialization. Put all table and File definitions before the   
Wallet command. Give every table and File   
definition a description, and keep   
native App state out of the database.

Use the exact lowercase App functions and argument shapes in the App surface. Each command has app, function, parameter, and description only when required for a table or File definition.

<<RETRY\_FEEDBACK>> App surface: <<APP\_SURFACE>>

## I.2.2 Scenario Creator

You create Scenarios for a proactive-agent benchmark.

Create one <<TRIGGER\_TYPE>> Scenario. Definition: <<TRIGGER\_DEFINITION>>

Choose the causal design by the state of the need across the evaluated prefixes, not by its surface stakes. If another trigger definition fits that state more directly, redesign the Scenario rather than relabel it.

Decision Context, and Consequence. Cover the causal situation and intended

action, or exposing the hidden design checks. Format every item as

cannot be later than the Event time. Calendar start, end, reminder, and due

times remain business times rather than Event times. Every initialization

Design requirements:

\- Build one causal trajectory around exactly

one need of the assigned trigger type. Every Event must develop that same need by changing its evidence, feasibility, options, or appropriate response; do not introduce a second proactive need or a competing trigger.

\- As silent design checks, ensure that at

least one Event prefix would make the contemplated intervention premature and therefore unacceptable, and that at least one later prefix would make intervention unacceptable because acting only then is already too late to achieve its intended benefit or prevention. Both judgments must follow from evidence in their respective prefixes. Do not state timing labels, prefix verdicts, or this design reasoning anywhere in the output JSON; present only the Scenario's world state and observable Events.

\- Events are observable changes in the world.

They may supply evidence, urgency, options, or confirmation, but must not contain the evaluated Agent's action, resolve the need, or turn the intervention into a direct response to an explicit request. Information may move between actors or Apps only through prior shared state or an observable communication in the prefix. - The best intervention must require at least

one decision-relevant fact from the Instance SQLite database or a TXT File. That fact must distinguish plausible recognition, timing, or action choices rather than merely support a conclusion already forced by Events. Events must neither state that fact nor independently imply the same decision. Use ini\_data\_commands when the shared state needs additional pre-Event facts or prerequisites.

\- The contemplated intervention must

materially improve the protagonist's situation when supported by the Event state, match the trigger definition, and have a commitment level proportionate to the available authority, consent, and reversibility. Every fact and parameter needed for that intervention must be available in the relevant prefix; when authority or preference is missing, use a lower-commitment step that preserves the protagonist's options. Trace each decisive condition for timing and action to a source available in that prefix; if one is missing or appears only later, revise the action or timing design. For blockage, the same activity must be shown as started and then actually unable to continue.

\- Use at least three distinct non-retrieval

Apps in the Event sequence. Do not count database, files, initialization, or information-retrieval steps toward this minimum. Every counted App must supply necessary causal evidence or a

state transition for the single need; if three Apps do not fit naturally, redesign the causal progression rather than adding a decorative channel. Compare the accepted summaries before

choosing the causal design. Make the Scenario materially different in its goal, cause, key information, Event progression, and likely intervention; changing surface details within the same workflow is not a material difference. A new Trigger label or failure mechanism alone does not make reuse of the same underlying activity and protagonist goal distinct. Compare the causal spine as well: if the same state-transition pattern leads to the same intervention, changing its subject matter is still a surface variation. Across the full set, distribute activity areas, retrieval sources, Event channels, and useful Agent actions.

State and consistency requirements:

\- ini\_data\_commands describe only facts that

exist before event1. Preserve the protagonist, Wallet, accepted shared state, existing SQL schemas, and natural key uniqueness. Keep every dated initialization fact on or before the assigned reference date. Add no unrelated noise.   
- Use native personal Apps for their own data. - Use nat ve persona Apps or t e r own ata.

Information-retrieval calls are not Events. send\_email and send\_message are evaluated-Agent actions, not Scenario Events or initialization commands. - Match every command's parameter keys to its

exact App signature. An Event's time does not add a timestamp argument to a function that does not define one.

\- The eight Scenarios share accumulated

initialization data, while their Event sequences are independent branches and are never each other's history.

\- Correct every retry issue with a genuinely

revised candidate rather than a renamed version of the same causal design.

Before returning, inspect the assembled JSON

keys, collection shapes, counts, and member

schemas, then check the causal

requirements above. Contrast the assigned

trigger with its closest alternative

across the Event sequence; if the distinction

depends on wording rather than

Assigned Scenario ID: <<SCENARIO\_ID>>

Assigned reference date: <<REFERENCE\_DATE>>

Protagonist: <<PERSONAL\_DESCRIPTION>>

Current shared Instance state: <<INSTANCE\_CONTEXT>>

Accepted Scenarios with the same trigger type:

<<SAME\_TRIGGER\_SCENARIOS>>

Accepted Scenarios with other trigger types:

<<OTHER\_SCENARIOS>>   
Retry feedback:   
<<RETRY\_FEEDBACK>>   
App surface:   
<<APP\_SURFACE>>

## I.2.3 Reviewer

You perform semantic review for a proactive-agent benchmark.

The deterministic Validator has already   
checked JSON structure and isolated App   
execution. The candidate is not committed: its initialization commands would   
join the shared initial state, while every Scenario's Events remain an   
independent branch. The candidate   
intentionally does not reveal an expected   
Agent answer or timing labels. Infer the   
concrete, most useful intervention   
that the observable state and available tools support; it must not already   
occur in Events or initialization. Reject when no such intervention is   
reasonably inferable or feasible, not merely because the JSON does not state   
the answer explicitly.   
Return raw JSON only, without Markdown or code   
fences. Return one JSON object   
with exactly:   
{"valid": boolean, "issues": [concise,   
concrete strings]}

Judge only material benchmark quality: 1. Trigger fidelity: the evidence matches the expected trigger definition.

supports at least one prefix where action would be unacceptably premature and at least one later prefix where acting only then would be unacceptably late. Treat each Event's time as the current time at that prefix, distinct from App record and business times. Judge each boundary only from evidence available in that prefix.

3. Retrieval dependency: a necessary database

or TXT fact is available, not duplicated by Events, and materially changes recognition, choice, or action.

4. Intervention value: the contemplated help is useful, feasible when supported by the Event state, proportionate to the available authority, consent, and reversibility, and not already supplied by the normal process or by an explicit request containing the answer.

5. Causal and shared-state integrity:

prerequisites exist, initialization is genuinely pre-Event, external developments have causes, and all shared facts can coexist. Do not treat Events from independent Scenarios as shared history.

6. Set quality: the candidate adds meaningful diversity in activity, cause,

information source, Event progression, and likely intervention without using irrelevant Apps or merely renaming an accepted design. A different Trigger label or failure mechanism does not by itself justify repeating the same underlying activity and protagonist goal; the causal spine and likely intervention must also be materially different from the nearest example.

Trigger definitions: dependency=future prerequisite missing; blockage=an active attempt has actually stopped; risk=plausible future loss has not happened yet; opportunity=optional benefit with no failure or loss from inaction.

Do not reject harmless additions to sparse base state, stylistic preferences, or set-level quotas that cannot yet be satisfied during an intermediate round. A candidate is valid only when no material issue remains.

Expected candidate trigger: <<EXPECTED\_TRIGGER>> Domain: <<DOMAIN\_SPEC>> Instance: <<INSTANCE\_PROFILE>>

Accepted Scenarios: <<ACCEPTED\_SCENARIOS>>

Candidate Scenario, including initialization: <<CANDIDATE\_SCENARIO>>

Shared state before the candidate: <<BASE\_STATE>>

## I.2.4 Noise Creator

You create background Noise for a proactive-agent benchmark.

All eight Scenarios are complete. Return raw   
JSON only, without Markdown or   
code fences. Return one JSON object with   
exactly:   
{"noise\_commands": [...]}

Create 12-20 valid app/function/parameter commands that add ordinary, Domain-coherent background state across several Apps. Use existing database schemas, exact App signatures, unique natural keys, affordable completed Wallet activity, and valid Messaging conversations.

Noise must remain causally irrelevant to every accepted Scenario. It must be   
settled or inert at the initial state: it   
cannot introduce evidence, urgency,   
an unresolved need, a future obligation, a pending outcome, or an evaluated   
Agent action. Preserve all existing objects, identity, Wallet configuration,   
and Scenario facts.

Protagonist: <<PERSONAL\_DESCRIPTION>>

Final shared state before Noise:   
<<INSTANCE\_CONTEXT>>   
Accepted Scenarios:   
<<ACCEPTED\_SCENARIOS>>   
Retry feedback:   
<<RETRY\_FEEDBACK>>   
App surface:   
<<APP\_SURFACE>>

## J Annotation Interface

The platform separates scenario audit from response evaluation. Figure 10 shows the audit workspace above the response evaluation workspace. The upper panel presents the full scenario and supporting environment beside the audit criteria, while the lower panel presents the event prefix, intermediate retrieval rounds, and final response for evaluation. For rejected interventions, separate timing and intensity questions reveal the diagnostic form shown in Figure 11. These screenshots use the original application with a local preview database and a recorded benchmark example. The displayed human diagnosis reproduces the exported annotations.

![](images/df5b5a1e0897bc79f00b80079ff76cce637c587f809339251955200969d698aa.jpg)  
Figure 10: Annotation workspaces for scenario audit (upper) and response evaluation (lower).

![](images/c3cfd78f5cae0f63f3143037cd9b9428747a412a26148fdeea15e8486cf29cd2.jpg)  
Figure 11: Human diagnosis form for a rejected intervention, with independent timing and intensity questions and optional other issues.