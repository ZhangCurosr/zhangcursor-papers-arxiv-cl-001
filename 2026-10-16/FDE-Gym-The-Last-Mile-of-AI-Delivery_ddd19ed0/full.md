# FDE-Gym: The Last Mile of AI Delivery

Website | Online Gym | Leaderboard | GitHub | Hugging Face | Submission | Email us

Core Authors

Dongxu Yang<sup>1,</sup>∗, Hongda Yan<sup>1</sup>, Renxin Wang<sup>1</sup>, Baoli Zhang<sup>8</sup>, Haojia Zuo<sup>2</sup>, Ying Zhao<sup>2</sup>, Youmin Chen<sup>3</sup> ∗Corresponding author.

## Contributors

Xiaojian Liao<sup>4</sup>, Congming Gao<sup>5</sup>, Tianyang Jiang<sup>15</sup>, Yixiao Chen<sup>11</sup>, Qing Wang<sup>6</sup>, Zhiguo Yu<sup>16</sup>, Xiaqing Li<sup>7</sup>, Yuhao Zhang<sup>9</sup>, Jiawen Sun<sup>17</sup>, Renzhe Xu<sup>10</sup>, De Mi<sup>19</sup>, Tiancheng Wu<sup>18</sup>, Xianya Fu<sup>12</sup>, Xuanlu Zhao<sup>13,20</sup>, Bo Feng<sup>21</sup>, Meixuan Ren<sup>22</sup>, Guanming Lin<sup>14</sup>, Ziheng Zhou<sup>23</sup>, Xinhua Wang<sup>23</sup>

<sup>1</sup>DeepLethe <sup>2</sup>Tsinghua University <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Beihang University <sup>5</sup>Xiamen University <sup>6</sup>Nanjing University <sup>7</sup>Beijing Jiaotong University <sup>8</sup>Beijing University of Posts and Telecommunications <sup>9</sup>Tianjin University <sup>10</sup>Shanghai University of Finance and Economics <sup>11</sup>Harvard University <sup>12</sup>Stanford University <sup>13</sup>University College London <sup>14</sup>Cornell University <sup>15</sup>Huawei <sup>16</sup>Atlassian <sup>17</sup>Meta <sup>18</sup>Microsoft Research Asia <sup>19</sup>StellarCube.ai <sup>20</sup>UWits Technology (Beijing) Co., Ltd. <sup>21</sup>Bopai Management Consulting (Shenzhen) Co., Ltd. <sup>22</sup>Exploration and Development Research Institute, PetroChina Southwest Oil & Gasfield Company <sup>23</sup>Kunlun Digital Technology Co., Ltd.

![](images/b332dd774790f1547e6abc7b5e312ee2be1980701483c7afc7609b7ee7135d27.jpg)  
Figure 1: Construction of FDE-Gym. Experts adapt 20 real-world delivery projects from 18 sectors into operational customer environments, alter key facts to create variant tasks, and retain the 95 tasks whose graders have been validated against the predefined scoring anchors.

![](images/a58307e035efd5711752cda614dee5c93538ee56cc5e4376c5ec284168e15734.jpg)

![](images/350fa9953e4a838ffd7d46a368285e1e1ef7f64ee90c28b5dc0776c222af6d02.jpg)  
Figure 2: Model performance across task instruction levels. (a) Mean L3 score for each model on all 95 tasks, on a normalized scale where 0 represents the do-nothing baseline and 1 represents the reference solution. Solid bars show scores after deducting stakeholder contact cost; lighter bars show normalized business uplift before this deduction. (b) Scores as progressively less information is provided in the task instruction. L0-L2 are evaluated on the 20 original tasks and L3 on all 95 tasks.

## Abstract

Existing agent benchmarks typically evaluate agents on well-specified tasks defined by issue reports, specifications, or task tickets. Real-world customer delivery, however, poses fundamentally diferent challenges: forward-deployed engineers (FDEs) must clarify ambiguous requirements, work in live systems, and uncover unstated constraints. To address this gap, we introduce FDE-Gym, a benchmark of 95 end-to-end delivery tasks from 20 customer scenarios across 18 sectors, each adapted by industry experts from real-world delivery projects. In simulated customer environments featuring scripted stakeholders and modifiable customer systems, agents interact with stakeholders and operate on production-like systems to deliver working solutions. Beyond binary task completion, FDE-Gym scores agents on how much their delivered system improves the customer’s own success metric, normalized between 0 (unchanged system) and 1 (reference solution); scores are penalized for stakeholder interactions (messages, trial runs, and waiting time) to discourage excessive or careless engagement. FDE-Gym also provides an online platform where humans can develop and practice FDE skills under the same grading criteria.

We evaluate 13 model configurations from seven providers. In the realistic setting (unstated constraints need to be discovered), the best configuration achieves a mean FDE-Gym score of 0.56, and resolves only 14 of the 95 tasks (a score of at least 0.8 with no incident). Discovering unstated constraints is the primary bottleneck: performance drops sharply when models must uncover them without explicit guidance. However, information discovery alone does not account for the gap: even on the 23 tasks where the best configuration obtains every key fact from stakeholders, it achieves a normalized business uplift of 0.8 or above on only 10.

These results reveal a substantial gap between solving well-specified tasks and delivering efective AI systems for real customers. Closing this gap requires agents that can surface unstated constraints, operate reliably in live systems, and achieve measurable improvements on customer metrics.

## 1 Introduction

Deploying AI in enterprise settings requires more than meeting stated customer requirements. The challenge often lies in identifying which changes will improve customer operations in practice. One scenario in our benchmark illustrates this problem. A vice president at a drilling operator seeks a generative-AI success story for the November board meeting. The vice president requests a daily summary email and an “Ask your wells” chat box, noting that the downtime-coding system is already in place and therefore outside the project scope.

Yet a solution built strictly to the task instruction can be delivered on time, pass every functional check, and still fail to move the customer’s own business metric by a single point. The bottleneck lies in monthly downtime credit claims. The existing keyword-based script incorrectly attributes some downtime to the contractor, resulting in at least one disputed line each month and suspending the credit for that month. Addressing this problem requires revisiting the downtime-coding system that the sponsor had explicitly excluded from the project scope. The engineer must identify the applicable contractual rule, confirm the interpretation with the cost engineer, and obtain approval for the change from an approver who is unavailable that week.

This is the role of a forward-deployed engineer (FDE), an engineer assigned by a vendor to work within a customer organization. The role involves more than building what the sponsor initially requests. An FDE must identify what will actually help and work with customer systems and stakeholders to deliver the improvement.

Existing benchmarks rarely address this last mile of AI delivery, typically assuming that the problem has already been well defined. Software-engineering benchmarks provide issues and hidden tests [1, 2, 3], deployment bench marks provide specifications [4], enterprise benchmarks provide predefined tasks [5, 6], and clarification benchmarks withhold information from an otherwise correctly specified task [7, 8] — in each case, the underlying prob lem remains fixed. Recent benchmarks come closer to real-world delivery. τ<sup>τ</sup>-bench [9] evaluates agents that build customer-service systems from client requirements and deploy the resulting systems against held-out users. A posttraining benchmark [10] assigns agents the role of forward-deployed engineers and evaluates whether the delivered model improves performance. XYEval [11] tests agents on misleading user requests. However, these benchmarks still address individual aspects of delivery in isolation.

FDE-Gym addresses these challenges in a single benchmark of 95 tasks from 20 customer scenarios across 18 sectors, each adapted by industry experts from real-world delivery projects. The sponsor’s request may target the wrong problem, critical constraints must be discovered through stakeholder interaction, and performance is measured using the customer’s own business metric. A solution that addresses the wrong problem therefore earns no credit.

Delivered systems are deployed without modification and evaluated on held-out production data. The FDE-Gym score measures improvement on the customer’s own business metric after deducting stakeholder contact cost (Section 2.3 details what is charged). The do-nothing baseline (the unchanged system) scores 0, the expert-authored reference solution scores 1, and literal compliance with the sponsor’s request scores at most 0.2. Experimental results show that current models perform well below the reference solution on this scale.

To identify where agents fail, we present each scenario at four task instruction levels with progressively less information (from a fully specified problem (L0) down to the sponsor’s original request, which may not address the underlying problem (L3)). The environment, stakeholders, and grader remain unchanged across all four levels. Be cause only the task instruction varies, diferences in performance isolate the efect of the information withheld. We evaluate 13 model configurations from seven providers using this setup. Five main findings emerge.

• Even the best model configuration falls well short of the reference solution. At L3, Claude Opus 5.5 achieves the highest mean score of 0.56 and resolves 14 of 95 tasks, while five of the thirteen configurations resolve none. Selecting the best-performing configuration for each task still yields a score of only 0.70, with 30 tasks resolved.

• Discovering unstated constraints is the primary bottleneck for leading models. When the relevant rules and constraints are provided, the three leading models score 0.73–0.82. Requiring these models to discover the constraints reduces scores sharply to 0.39–0.62. On the original tasks, L3 scores are at most 0.08 points lower than L1 scores.

• Obtaining the necessary information does not ensure efective delivery. At L3, leading models obtain 72–85% of the key facts available from stakeholders, yet the delivered systems achieve only 0.60–0.70 of the reference improvement. Stakeholder contact cost reduces the final scores by a further 0.14–0.22.

• Weak models can leave customers worse of than if no changes had been made. Claude Haiku 4.5 scores below the do-nothing baseline on 79 of 95 tasks, while systems delivered by Mistral Small 4 fall below this baseline on 51 tasks even before deducting stakeholder contact cost.

• No model performs well across all scenarios. In five of the 20 scenarios, no model achieves a mean score of at least 0.5. Across tasks within a scenario, the median score range for a leading model is 0.34–0.44.

We make three contributions. First, we introduce FDE-Gym, a benchmark that evaluates whether agents can identify the underlying customer problem and deliver a working solution that measurably improves business outcomes. Second, we develop a methodology for constructing scenarios that combines task variants, scripted stakeholders, and predefined scoring anchors to support reproducible evaluation and limit rewards for literal compliance and shortcut solutions. Third, we evaluate 13 model configurations and provide an online platform where humans and agents can practice delivery under the same grading criteria.

## 2 Anatomy of a Task

Each task represents a single customer engagement involving one customer (the organization that hires the agent, whose people are the stakeholders the agent can question), one sponsor (the customer-side person who assigns the task), and one production system to improve. Figure 3 shows the four components of a task.

![](images/b4c7e1a601098a95a689d7b493694c761405b4c1c44b3465a9649e6e69470806.jpg)  
Figure 3: Components of an FDE-Gym task, showing the inputs, agent actions and deliverable, deployment and evaluation of the delivered system, and interpretation of the resulting score. The agent reads the task inputs and exchanges messages with customer stakeholders, who answer questions and may also initiate contact. A task is included only if the scoring anchors shown on the axis are satisfied.

## 2.1 Environment and Available Information

The agent starts in a workspace containing the task instruction (the description of the work assigned by the sponsor), a copy of the customer’s production system, relevant documents and historical data, and a staf directory with contact details. The runtime environment provides standard Python and SQLite. Installing additional packages and accessing the general network are prohibited. The delivered system may send network requests only to the company’s LLM gateway, which charges according to usage. Once delivery is complete, the contents of system/ are deployed to production without modification. Production continues to call the original entry points, so new functionality must be integrated with these entry points to take efect.

Table 1: The four task instruction levels. The workspace, stakeholders, and grading criteria remain identical across levels; only the task instruction varies.
<table><tr><td>Level</td><td>Information provided in the task instruction</td><td>What the agent must determine</td></tr><tr><td>L0</td><td>The underlying problem, applicable rules and constraints, How to design and implement an effective solution and how the result is scored</td><td></td></tr><tr><td>L1</td><td>A clear statement of the underlying problem</td><td>The applicable rules and constraints, and how to design and implement a solution</td></tr><tr><td>L2</td><td>A vague but directionally correct request</td><td>The problem to be solved, the applicable constraints, and how to design and implement a solution</td></tr><tr><td>L3</td><td>The sponsor&#x27;s original request, which may target the wrong problem</td><td>Whether the request addresses the underlying problem, which constraints apply, and how to design and implement a solution</td></tr></table>

The agent is not explicitly informed that stakeholder contact incurs costs, that some people should not be contacted or are irrelevant to the problem, or that the sponsor’s request may be misdirected. Identifying these issues is a central objective of the evaluation. Relevant clues appear in the staf directory, customer documents, and data, where an FDE would typically look for such information. As in real-world projects, documents may be outdated, legacy code may reflect incorrect assumptions, and important information may not be explicitly stated in the available materials.

## 2.2 Task Instruction Levels

The four task instruction levels (Table 1) distinguish the challenges of implementation, constraint discovery, and problem clarification by varying the task instruction while keeping the environment and evaluation criteria fixed. The L0 instruction explicitly states the underlying problem, relevant rules and constraints, and scoring criteria, leaving the agent to design and implement a solution. The L1 instruction withholds the rules and constraints, while the L2 instruction also provides only a vague description of the problem. The L3 instruction presents the sponsor’s original request, requiring the agent to assess whether the request addresses the underlying problem. We use L3 as the primary evaluation setting because this level most closely reflects the starting point of real-world customer delivery projects. L0 does not eliminate the need for engineering decisions. By specifying the rules while leaving solution design open, L0 measures what an agent can build when all underlying facts are provided.

## 2.3 FDE-Gym Score

FDE-Gym evaluates a delivered system by the improvement it achieves on the customer’s business metric, measured on held-out production data. For example, an after-sales system is assessed by its net financial benefit, while a drilling system is assessed by the downtime hours correctly attributed to a contractor for compensation claims.

We define the following metrics to assess agent performance on these delivery tasks.

Normalized Business Uplift (U). We measure the delivery’s business improvement relative to that achieved by an expert-authored reference solution:

$$
U = \frac { \mathrm { K P I } _ { \mathrm { d e l i v e r e d } } - \mathrm { K P I } _ { \mathrm { b a s e l i n e } } } { \mathrm { K P I } _ { \mathrm { r e f e r e n c e } } - \mathrm { K P I } _ { \mathrm { b a s e l i n e } } }\tag{1}
$$

Here, $\mathrm { K P I } _ { \mathrm { b a s e l i n e } } , \mathrm { K P I } _ { \mathrm { d e l i v e r e d } }$ , and $\mathrm { K P I } _ { \mathrm { r e f e r e n c e } }$ denote the customer’s business metric for the unchanged system, the delivered system, and the expert-authored reference solution, respectively, evaluated on the same held-out data. The unchanged system is the do-nothing baseline. Thus, U = 0 represents no improvement, U = 1 matches the reference improvement, and $U < 0$ indicates an outcome worse than the baseline.

Incident Adjustment $( U _ { \mathbf { a d j } } )$ . If a predefined operational incident occurs, such as an unauthorized action, positive uplift is set to zero while negative uplift is retained:

$$
U _ { \mathrm { a d j } } = { \left\{ \begin{array} { l l } { U , } & { { \mathrm { i f ~ n o ~ i n c i d e n t ~ o c c u r s , } } } \\ { \operatorname* { m i n } ( U , 0 ) , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }\tag{2}
$$

Stakeholder Contact Cost (C). Stakeholder contact cost is expressed in the same normalized units as business uplift. It is the sum of the charges incurred by the agent’s contacts with the customer:

$$
C = \sum _ { k \in K } c _ { k } ,\tag{3}
$$

![](images/04dbba747b8b39ba0e98cc8a738eb496c7600797f5f64a5a777ccb6a2010db0c.jpg)  
Figure 4: A worked example: the drilling scenario (oil\_npt), original task.

where K is the set of chargeable contacts in the run and $c _ { \mathrm { k } }$ is the charge for contact k under the task’s predefined pricing and contact rules. Each stakeholder has a hidden per-message rate, with higher charges when the agent exceeds their patience or violates a contact restriction. Unanswered messages are not charged. In scenarios with a trial environment or a calendar, each trial run and each day of waiting is also a chargeable contact.

FDE-Gym Score (S). The FDE-Gym Score measures delivery performance in a single evaluation run. It is the incident-adjusted normalized business uplift minus stakeholder contact cost:

$$
S = U _ { \mathrm { a d j } } - C .\tag{4}
$$

Without an incident, S = U  C. A task is considered resolved if its score is at least 0.8 and no incident occurs.

Average FDE-Gym Score (S). The Average FDE-Gym Score summarizes a model’s delivery performance across tasks and repeated runs. To prevent extreme business losses from dominating the reported average, we floor the incident-adjusted uplift at 1 before deducting the full stakeholder contact cost. We then average across tasks and runs:

$$
\overline { { S } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \left[ \operatorname* { m a x } \left( U _ { \mathrm { a d j } , i , r } , - 1 \right) - C _ { i , r } \right] .\tag{5}
$$

Here, N is the number of evaluated tasks, R is the number of runs per task, and i and r index tasks and runs, respectively. Each task receives equal weight. The floor applies only to aggregation and does not change the singlerun score S used to determine whether a task is resolved.

Figure 4 presents the drilling scenario, its customer metric, and delivery results.

## 3 Building the Benchmark

## 3.1 Scenario Construction from Real-World Delivery Projects

Figure 1 traces how a real delivery becomes a task in the benchmark. Industry experts adapted real-world customer delivery projects carried out in China, Singapore, and the United States since 2024 into 20 scenarios spanning 18 sectors (Table 2). Each scenario yields an original task and several variants of it, which share a common customer environment but difer in key facts. Each original task and each variant is a separate benchmark task, for a total of 95 tasks. Tasks require agents to deliver an agent, workflow, or service that operates in the customer’s environment. The scenarios are grounded in industry practice, with the relevant regulations, standards, and public sources documented. Companies and individuals are either de-identified or fictional, and some data are adapted from real-world industry sources.

## 3.2 Scenario Variants and Underlying Facts

A single fixed task per scenario may allow models to rely on memorized solutions or recurring patterns. To reduce this risk, experts designed two to seven variants in addition to each scenario’s original task. Variants change key facts, including which rule is outdated, who holds critical information, and whether the sponsor’s request addresses the underlying problem. These changes may require diferent delivery solutions, so a solution that succeeds on one task of a scenario may fail on another. Across scenarios, no single problem pattern appears in more than one-third of the set. Together, these design choices encourage agents to evaluate the evidence in each task rather than apply a fixed solution.

Table 2: The 20 scenarios of FDE-Gym v1.0. A scenario’s tasks (the original and its variants) share the same customer environment and difer in the key facts behind the problem; there are 95 tasks in all. The last column gives the held-out data on which the original task is evaluated.
<table><tr><td>Scenario</td><td>What is delivered</td><td>Sector</td><td>Tasks</td><td>Evaluated on</td></tr><tr><td>bank_scam_hold</td><td>Anti-scam hold queue agent</td><td>Banking</td><td>3</td><td>523 queue calls</td></tr><tr><td>build_pay_cutover</td><td>Subcontract payment-ledger cut-over</td><td>Construction</td><td>4</td><td>one cut-over and its claims</td></tr><tr><td>campus_booking</td><td>Space-booking assistant</td><td>Education</td><td>5</td><td>82 conversations</td></tr><tr><td>charge_care_config</td><td>No-code driver-support agent</td><td>EV charging</td><td>4</td><td>the customer&#x27;s test suite</td></tr><tr><td>estate_repair_cases</td><td>Agent for fund-paid major repairs</td><td>Property mgmt.</td><td>4</td><td>145 cases over 63 days</td></tr><tr><td>fund_recon</td><td>Reconciliation break clearing</td><td>Asset management</td><td>8</td><td>188 breaks</td></tr><tr><td>grid_interval_pipeline</td><td>Nightly interval-settlement job</td><td>Power grid</td><td>4</td><td>nightly settlement runs</td></tr><tr><td>hiring_screen</td><td>Applicant screening tool</td><td>Logistics</td><td>3</td><td>two site configurations</td></tr><tr><td>hospital_triage</td><td>Triage chat with a trial run</td><td>Healthcare</td><td>4</td><td>190 sessions</td></tr><tr><td>listing_handover</td><td>Listing compliance checker</td><td>E-commerce</td><td>4</td><td>345 listings</td></tr><tr><td>logistics_claims</td><td>After-sales agent that can pay out</td><td>Logistics</td><td>7</td><td>212 tickets</td></tr><tr><td>mfg_chatbi</td><td>&quot;Ask the data&quot; assistant</td><td>Manufacturing</td><td>7</td><td>94 live-database questions</td></tr><tr><td>news_incident</td><td>AI flash desk for a newsroom</td><td>Media</td><td>6</td><td>two weeks of stories</td></tr><tr><td>oil_npt</td><td>Drilling downtime coder</td><td>Energy</td><td>5</td><td>90 drilling reports</td></tr><tr><td>oncall_incident</td><td>AI night responder</td><td>Software</td><td>3</td><td>184 incidents</td></tr><tr><td>prior_auth</td><td>Prior-authorization agent</td><td>Health insurance</td><td>3</td><td>165 requests</td></tr><tr><td>revenue_dispute</td><td>Monthly revenue report</td><td>Manufacturing</td><td>3</td><td>100 disputed items</td></tr><tr><td>review_screening</td><td>Systematic-review screening</td><td>Research</td><td>6</td><td>1,400 records</td></tr><tr><td>robot_cell_rollout</td><td>Mobile-manipulator rollout plan</td><td>Warehouse robotics</td><td>5</td><td>12 simulated months</td></tr><tr><td>water_twin</td><td>Reservoir release rule</td><td>Water</td><td>7</td><td>six scenario years</td></tr></table>

## 3.3 Stakeholder Simulation

FDE-Gym simulates more than 130 customer stakeholders across its 20 scenarios. Stakeholder responses are scripted for reproducibility, as switching the user-simulation model can shift measured agent success rates by sev eral percentage points [14]. A small model maps questions to topics, while each stakeholder’s knowledge, misconceptions, and conditions for disclosing information are fixed in advance. The scripts incorporate limited patience, requirements for supporting evidence, dependencies on prior conversations, temporary unavailability, and contact restrictions. Agents must infer stakeholder roles and communication constraints from customer documents.

## 3.4 Admission Anchors and Anti-Gaming Validation

Each task included in FDE-Gym must satisfy four predefined scoring anchors (Figure 3). (1) The reference solution must score at least 0.95. (2) The do-nothing baseline must score 0. (3) The literal solution, which implements exactly what the task instruction asks for without seeking any further constraint, must score at most 0.2. This condition holds even when the sponsor’s request identifies the right problem, because the constraints omitted from the task instruction still determine the outcome. The only exception is a task in which the sponsor’s request is both correct and complete; there the literal solution coincides with the reference solution and must score at least 0.95, so that distrusting the sponsor is never a universally rewarded strategy. (4) Shortcut strategies must score at most 0.8. These strategies include overfitting to historical data, guessing round-number thresholds, and deferring all uncertain decisions to humans. Scores are capped near zero if an agent attempts to bypass customer controls or manipulate the grader. The graders were tested against more than 270 shortcut and bypass strategies. Agents that solve tasks using only customer documents and data are scored on the same basis as those that interact with stakeholders.

## 3.5 Expert Review and the Online Platform

Industry experts cross-reviewed and iteratively refined each task. Their total contribution to the development of FDE-Gym exceeded 10,000 hours. FDE-Gym also provides an online platform (fde-gym.com) where human participants use the same workspaces, interact with the same stakeholders, and are evaluated by the same graders as agents. A human baseline is currently being collected there under recorded conditions from industry experts working on the L3 tasks. The results will be reported in a future version of this paper.

## 4 Experimental Setup

Models. We evaluate 13 model configurations from seven providers (12 models, one evaluated both with and without extended thinking): Claude Opus 5.5, Claude Sonnet 5.5, GPT-6 Astra, GPT-6.1 Sol, GLM-5.3 Flash, DeepSeek V4.1 Flash, Gemini 3.8 Flash, Qwen 3.8 Flash, GPT-5.5, GPT-6 Luna, Mistral Small 4, Claude Haiku 4.5, and Claude Haiku 4.5 (thinking). All models use the same minimal agent framework with five tools (execute commands, read files, write files, edit files, and list files), without model-specific customization. They contact stakeholders by invoking a command in the workspace.

Protocol. We evaluate models at four task instruction levels (L0–L3), from a fully specified problem to the sponsor’s original request. L3 is the primary setting, reflecting real-world delivery where the request may not address the underlying problem. It covers all 95 tasks: 20 original tasks and 75 variants with altered key facts. To measure the efect of information provided in the task instruction on performance, we also evaluate L0–L2 on the original tasks. Three scenarios have identical L2 and L3 task instructions; for these, the L3 run on the original task also serves as the L2 result, so all four levels are reported on the same 20 original tasks. Each model is therefore run on 20 tasks at L0, 20 at L1, 17 at L2, and 95 at L3, totaling 152 task–instruction-level combinations per model and 1,976 across all 13 model configurations. Each combination is run three times, and the scores are averaged. Each run ends when the agent submits its delivery or reaches a limit of 200 turns, 150 minutes, or USD 8 in model usage cost, whichever comes first. The system is then deployed and evaluated in the state reached at that point.

Reporting. We report the mean FDE-Gym score across evaluated tasks at each task instruction level, with 95% confidence intervals computed by bootstrapping over tasks for the full L3 evaluation. We also report the normalized business uplift before stakeholder contact cost, to distinguish system performance from stakeholder contact cost, along with the model usage cost per run, as recommended in work on cost-aware evaluation [15, 16]. Scores, confidence intervals, normalized business uplift, and stakeholder contact cost are shown in Table 3; mean model usage cost per run is shown in Figure 5b.

Comparison with other benchmarks. We compare FDE-Gym scores with task success rates on Terminal-Bench 2.1 [2], Vibe Code Bench [12], and Finance Agent [13], using results from Vals AI (vals.ai/benchmarks, accessed 6 October 2026). The published results are task success rates, whereas the FDE-Gym result is a normalized uplift; the figure therefore places the two side by side without equating them. We use a single independent source so that each benchmark is run in the same way for every model. The other benchmarks cited in this paper are not run by this source, and SWE-bench Verified is no longer run on new models there, so comparable results are not available for the models shown. For each provider, we show its best-performing model on FDE-Gym among those evaluated in our study with results available on all three external benchmarks. To compare with success rates reported as percentages, we multiply FDE-Gym scores by 100.

Availability. Both the harness that creates task workspaces, simulates stakeholders, and grades deliveries, and the experiment runner used in this study are open source (github.com/deeplethe/fde-gym, version 1.1.0). The 20 original tasks and their graders are available on request (huggingface.co/datasets/DeepLethe/FDE-Gym), with three tasks public as examples. The 75 variants and all reference solutions are held out.

## 5 Results

## 5.1 Even the Best Model Is Far from the Reference

Finding 1 The best configuration achieves an average FDE-Gym score of 0.56 and resolves 14 of 95 tasks.   
Even selecting the best configuration for each task does not close the gap.

Figure 2 and Table 3 present the main results. At L3, Claude Opus 5.5 leads with 0.56 (95% interval 0.50 to 0.62), followed by Claude Sonnet 5.5 at 0.49, GPT-6 Astra at 0.38, and GPT-6.1 Sol at 0.37. We refer to the first three as the three leading models throughout, with GPT-6.1 Sol just below them. Three further configurations score near 0.2, two are only marginally above zero, and the remaining four score below zero. No model resolves more than 15 of 95 tasks, and five of the thirteen configurations resolve none.

The shortfall is not driven by a few unusually dificult scenarios. On only one scenario, bank\_scam\_hold, does any model average at least 0.8 (Figure 7). Taking, for each task, the best score achieved by any of the thirteen configurations yields 0.70 with 30 tasks resolved; on 16 tasks no configuration reaches 0.5.

The upper end of the scale is attainable: every reference solution scores at least 0.95 under the same grader, and when the rules are provided (L0) the leading models reach 0.73–0.82. How far human experts can proceed when given only the sponsor’s request will be established by the human baseline currently being collected.

## 5.2 Withholding Constraints Produces the Largest Observed Score Drop

Finding 2 Withholding the constraints reduces the scores of the three leading models by 0.28 on the same tasks (95% interval 0.17 to 0.39). Making the request vague, and then misdirected, reduces them by only a further 0.04 ( 0.03 to 0.14).

Figure 2b traces the progression across task instruction levels, and the left half of Table 3 reports every level on the same 20 original tasks; the values quoted in this section refer to these tasks. With the rules given (L0), the three leading models score 0.73–0.82 and resolve 11–13 of 20 tasks. At L1, where the problem is still stated correctly but its constraints must be found, they score 0.39–0.62 and resolve 1–7. The loss has two components. The delivered systems deteriorate, from 0.81–0.90 to 0.63–0.80 of the reference uplift, and discovering the constraints incurs a cost: questions per task rise from 3.9–7.0 to 11.3–13.0, and stakeholder contact cost from 0.07–0.10 to 0.18–0.24.

Beyond this step, the scores of these models are nearly flat. Claude Opus 5.5 scores 0.62, 0.54, and 0.55 at L1, L2, and L3; Claude Sonnet 5.5 scores 0.45, 0.47, and 0.46; and GPT-6 Astra scores 0.39, 0.34, and 0.33. Using paired results for the same 20 tasks and averaging across the three models, the decline from L0 to L1 is 0.28 (95% interval 0.17 to 0.39) and the decline from L1 to L3 is 0.04 ( 0.03 to 0.14). The first step therefore costs more than the two that follow, although a further loss of up to 0.14 cannot be excluded, and for an individual model the upper bound is as large as 0.25. Our initial hypothesis was that problem definition would be the bottleneck: an agent given a symptom or a wrong solution would build the wrong system. For the leading models, the data do not support this. One interpretation is that once these models must search for constraints at all, they also discover what the sponsor actually needs. For weaker models, the steps beyond L1 remain costly: across L0–L3, GPT-5.5 scores 0.51, 0.15, 0.04, and 0.06, and Claude Haiku 4.5 scores 0.11, 0.11, 0.30, and 0.25.

Table 3: Main results. Left: score by task instruction level on the 20 original tasks. Right: all 95 tasks at L3, with the 95% interval, the normalized business uplift before stakeholder contact cost, the stakeholder contact cost, and the number of tasks resolved (score 0.8, no incident) and the number of tasks finished below the do-nothing baseline (score < 0).
<table><tr><td></td><td colspan="4">Original tasks</td><td colspan="6">All 95 tasks, L3</td></tr><tr><td>Configuration</td><td>L0</td><td>L1</td><td>L2</td><td>L3</td><td>Score</td><td>95% CI</td><td>U</td><td>C</td><td>Resolved</td><td>&lt; 0</td></tr><tr><td>Claude Opus 5.5</td><td>0.82</td><td>0.62</td><td>0.54</td><td>0.55</td><td>0.56</td><td>0.50, 0.62</td><td>0.70</td><td>0.14</td><td>14</td><td>5</td></tr><tr><td>Claude Sonnet 5.5</td><td>0.74</td><td>0.45</td><td>0.47</td><td>0.46</td><td>0.49</td><td>0.43, 0.55</td><td>0.67</td><td>0.18</td><td></td><td>7</td></tr><tr><td>GPT-6 Astra</td><td>0.73</td><td>0.39</td><td>0.34</td><td>0.33</td><td>0.38</td><td>0.30, 0.46</td><td>0.60</td><td>0.22</td><td></td><td>18</td></tr><tr><td>GPT-6.1 Sol</td><td>0.76</td><td>0.33</td><td>0.28</td><td>0.43</td><td>0.37</td><td>0.28, 0.44</td><td>0.57</td><td>0.20</td><td></td><td>17</td></tr><tr><td>GLM-5.3 Flash</td><td>0.68</td><td>0.29</td><td>0.38</td><td>0.24</td><td>0.22</td><td>0.11,0.31</td><td>0.40</td><td>0.18</td><td></td><td>28</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>0.61</td><td>0.20</td><td>0.27</td><td>0.28</td><td>0.21</td><td>0.10, 0.30</td><td>0.38</td><td>0.18</td><td></td><td>26</td></tr><tr><td>Gemini 3.8 Flash</td><td>0.58</td><td>0.25</td><td>0.15</td><td>0.21</td><td>0.19</td><td>0.09, 0.28</td><td>0.45</td><td>0.26</td><td></td><td>28</td></tr><tr><td>Qwen 3.8 Flash</td><td>0.36</td><td>0.06</td><td>0.09</td><td>0.07</td><td>0.07</td><td>-0.03, 0.17</td><td>0.27</td><td>0.21</td><td></td><td>39</td></tr><tr><td>GPT-5.5</td><td>0.51</td><td>0.15</td><td>0.04</td><td>0.06</td><td>0.03</td><td>-0.05,0.11</td><td>0.11</td><td>0.08</td><td></td><td>44</td></tr><tr><td>GPT-6 Luna</td><td>0.12</td><td>-0.02</td><td>-0.07</td><td>-0.12</td><td>-0.12</td><td>-0.20, -0.05</td><td>0.00</td><td>0.12</td><td></td><td>63</td></tr><tr><td>Mistral Small 4</td><td>-0.07</td><td>-0.25</td><td>-0.31</td><td>-0.31</td><td>-0.32</td><td>-0.41, -0.23</td><td>-0.31</td><td>0.01</td><td></td><td>57</td></tr><tr><td>Claude Haiku 4.5</td><td>0.11</td><td>-0.11</td><td>-0.30</td><td>-0.25</td><td>-0.33</td><td>-0.42, −0.24</td><td>-0.23</td><td>0.09</td><td>0</td><td>79</td></tr><tr><td>Claude Haiku 4.5 (thinking)</td><td>0.14</td><td>-0.23</td><td>-0.29</td><td>-0.24</td><td>-0.35</td><td>-0.44, -0.26</td><td>-0.27</td><td>0.08</td><td>0</td><td>74</td></tr></table>

U: normalized business uplift; C: stakeholder contact cost. For three scenarios whose L2 and L3 task instructions are identical, the L3 result is also reported as the L2 result.

![](images/cdb5fe471eb164bf9bb0d4a38850479a4255aaef9f606988bae8e07bda97c683.jpg)

![](images/08a51abb918488c4db0541d6bf610b60c08ffdeb5b237b9da2e2da1658cc104b.jpg)  
Figure 5: (a) Share of key stakeholder facts obtained against normalized business uplift, at L3. Leading models learn most of the facts and deliver 0.60–0.70 of the reference uplift. (b) Mean model usage cost per run against score at L3.

The fully specified level yields a result of its own: even with every rule provided, no model exceeds 0.82 on the same 20 tasks. Knowing the facts does not remove the need for engineering.

## 5.3 Recovering Key Facts Does Not Ensure Efective Delivery

Finding 3 Leading models obtain most of the facts held by stakeholders yet deliver only 0.60–0.70 of the reference uplift; stakeholder contact costs further reduce their scores.

The grader records which key facts an agent obtained from stakeholders, which allows us to separate information discovery from system building. At L3 the three leading models ask 10.0–12.3 questions per task and obtain 72– 85% of the key facts. Their normalized business uplift is 0.60–0.70. Figure 5a shows the pattern across all thirteen configurations: more facts are associated with better systems, but the leading group sits at the far right of the plot and still well below the reference solution. In the 73 task–model cases where one of the three obtained every key fact, the normalized business uplift averages 0.66 and reaches at least 0.8 in 27 of them (37%). For Claude Opus 5.5 alone, every key fact was obtained on 23 tasks, and the uplift reached at least 0.8 on 10 of them.

![](images/a10b9ee8f5749de32f2dd6f133d29b63dd473b7a1a5e7f1ece9cb2b891131b50.jpg)  
Figure 6: Distribution of each model’s 95 tasks at L3 by score band. Scores of 0.2 or below correspond to the cap on literal compliance with the sponsor’s task instruction.

The second component of the finding is stakeholder contact cost. GPT-6 Astra’s delivered systems reach 0.8 on 36 of 95 tasks, but only 9 remain resolved once stakeholder contact cost is deducted; for Claude Opus 5.5 the figures are 40 and 14. Agents are not informed that questions incur a cost, and they ask as if questions were free. Stakeholder contact cost consumes 0.14 of the 0.70 uplift delivered by Claude Opus 5.5 and 0.22 of the 0.60 delivered by GPT-6 Astra. At the other extreme are models that rarely ask: Mistral Small 4 asks 0.7 questions per task and obtains 6% of the key facts.

## 5.4 Weak Models Frequently Score Below the Do-Nothing Baseline

Finding 4 The weakest models score below the do-nothing baseline on 60–83% of tasks; on these tasks, delivering nothing would have been the better outcome.

A score of zero is not the floor: a system that pays out claims it should have refused, charges the wrong party, or wakes the wrong engineer performs worse than the system it replaced. Figure 6 sorts each model’s 95 tasks at L3 into four score bands. Claude Opus 5.5 scores below zero on 5 tasks and Claude Sonnet 5.5 on 7, whereas Claude Haiku 4.5 does so on 79 and GPT-6 Luna on 63.

Part of this reflects stakeholder contact cost, but much of it reflects the delivered system itself. Before any stakeholder contact cost is deducted, the systems delivered by Mistral Small 4 score below the do-nothing baseline on 51 tasks and those delivered by Claude Haiku 4.5 (thinking) on 55, compared with 1–4 for the three leading models. On 23 tasks, the system delivered by Mistral Small 4 caused more harm than the benefit produced by the reference solution. Extended thinking does not help: Claude Haiku 4.5 scores 0.33 without it and 0.35 with it. For a buyer, the practical implication is that a weak agent is not a cheaper version of a strong one; its expected contribution on these tasks is negative.

## 5.5 Model Performance Varies Substantially Across Scenarios and Task Variants

Finding 5 On five of the 20 scenarios no model averages at least 0.5, and the same scenario with diferent key facts yields a diferent score.

Figure 7 breaks the headline result down by scenario. Claude Opus 5.5 is the best-performing model on 11 scenarios, Claude Sonnet 5.5 on four, GPT-6 Astra on two, and GLM-5.3 Flash, Gemini 3.8 Flash, and GPT-6.1 Sol on one each. The spread within a row can be wide even among the leading models: on revenue\_dispute the three score 0.12, 0.27, and 0.44. On five scenarios no model averages at least 0.5, and on the two hardest, news\_incident and revenue\_dispute, the best scores are only 0.18 and 0.27.

The variants show how strongly a score depends on the particular facts of a task. Across the tasks within a scenario, the median range of a leading model’s score is 0.34–0.44, and this range exceeds 0.5 in 6–8 of the 20 scenarios.

## 5.6 Higher Spending and Efort Do Not Consistently Correspond to Higher Scores

Cost and efort explain little of the ranking (Figure 5b). GPT-6.1 Sol spends USD 0.24 and 25 turns per run and scores 0.37, whereas Gemini 3.8 Flash spends USD 1.54 and 121 turns and scores 0.19. Qwen 3.8 Flash works longest, at 169 turns and 110 minutes per run, submits within the limits on only 11 of its 152 task–instructionlevel combinations, and scores 0.07; GLM-5.3 Flash submits on 95. The three leading models submit on every combination, in 26–60 turns. Recognizing when the work is complete appears to be part of the required skill.

<table><tr><td>-1</td><td>米</td><td>米</td><td>S</td><td>S</td><td>Z</td><td>Q</td><td></td><td>女</td><td>S</td><td>S</td><td>H</td><td>米</td><td>米</td></tr><tr><td>bank_scam_hold</td><td>0.91</td><td>0.87</td><td>0.83</td><td>0.81</td><td>0.91</td><td>0.91</td><td>0.80</td><td>0.89</td><td>0.53</td><td>0.47</td><td>0.10</td><td>0.40</td><td>0.10</td></tr><tr><td>review_screening</td><td>0.79</td><td>0.79</td><td>0.51</td><td>0.53</td><td>0.01</td><td>0.37</td><td>0.70</td><td>0.18</td><td>-0.26</td><td>-0.20</td><td>-0.24</td><td>-0.49</td><td>-0.60</td></tr><tr><td>logistics_claims</td><td>0.75</td><td>0.45</td><td>0.77</td><td>0.52</td><td>0.67</td><td>0.45</td><td>0.36</td><td>0.25</td><td>0.17</td><td>0.08</td><td>-0.64</td><td>-0.51</td><td>-0.11</td></tr><tr><td>campus_booking</td><td>0.76</td><td>0.55</td><td>0.74</td><td>0.68</td><td>0.38</td><td>0.33</td><td>0.51</td><td>0.19</td><td>-0.09</td><td>-0.11</td><td>-0.01</td><td>-0.39</td><td>-0.14</td></tr><tr><td>oncall_incident</td><td>0.75</td><td>0.72</td><td>0.67</td><td>0.75</td><td>0.50</td><td>0.56</td><td>0.50</td><td>0.43</td><td>0.37</td><td>0.11</td><td>-0.05</td><td>-0.03</td><td>-0.16</td></tr><tr><td>fund_recon</td><td>0.73</td><td>0.74</td><td>0.61</td><td>0.56</td><td>0.65</td><td>0.58</td><td>0.47</td><td>0.50</td><td>0.36</td><td>0.06</td><td>-0.88</td><td>-0.89</td><td>-0.94</td></tr><tr><td>hospital_triage</td><td>0.72</td><td>0.63</td><td>0.52</td><td>0.57</td><td>0.35</td><td>0.27</td><td>0.25</td><td>0.27</td><td>-0.02</td><td>-0.06</td><td>0.03</td><td>-0.08</td><td>-0.02</td></tr><tr><td>charge_care_config</td><td>0.50</td><td>0.57</td><td>0.72</td><td>0.60</td><td>0.47</td><td>0.37</td><td>0.65</td><td>0.37</td><td>0.17</td><td>0.24</td><td>0.02</td><td>0.18</td><td>0.20</td></tr><tr><td>listing_handover</td><td>0.58</td><td>0.62</td><td>0.64</td><td>0.67</td><td>0.09</td><td>0.48</td><td>-0.01</td><td>-0.05</td><td>0.28</td><td>0.10</td><td>0.02</td><td>-0.05</td><td>0.01</td></tr><tr><td>prior_auth</td><td>0.63</td><td>0.52</td><td>0.36</td><td>0.44</td><td>0.59</td><td>0.52</td><td>0.05</td><td>0.31</td><td>0.17</td><td>-0.11</td><td>-0.90</td><td>-0.50</td><td>-0.43</td></tr><tr><td>oil_npt</td><td>0.61</td><td>0.52</td><td>0.04</td><td>0.18</td><td>0.01</td><td>-0.01</td><td>-0.20</td><td>0.00</td><td>-0.02</td><td>-0.38</td><td>-0.01</td><td>-0.03</td><td>-0.02</td></tr><tr><td>grid_interval_pipeline</td><td>0.58</td><td>0.45</td><td>0.15</td><td>0.18</td><td>-0.09</td><td>0.24</td><td>-0.12</td><td>-0.11</td><td>0.21</td><td>-0.08</td><td>-0.01</td><td>-0.16</td><td>-0.32</td></tr><tr><td>water_twin</td><td>0.57</td><td>0.35</td><td>0.09</td><td>0.07</td><td>-0.20</td><td>-0.42</td><td>-0.60</td><td>-0.45</td><td>-0.46</td><td>-0.74</td><td>-0.63</td><td>-0.28</td><td>-0.51</td></tr><tr><td>mfg_chatbi</td><td>0.57</td><td>0.34</td><td>0.20</td><td>0.10</td><td>-0.09</td><td>-0.39</td><td>-0.21</td><td>-0.38</td><td>-0.30</td><td>-0.83</td><td>-0.65</td><td>-0.87</td><td>-0.98</td></tr><tr><td>build_pay_cutover</td><td>0.46</td><td>0.50</td><td>0.33</td><td>0.47</td><td>0.30</td><td>0.49</td><td>0.57</td><td>0.44</td><td>0.53</td><td>0.34</td><td>0.06</td><td>0.06</td><td>0.17</td></tr><tr><td>robot_cell_rollout</td><td>0.39</td><td>0.49</td><td>0.48</td><td>0.49</td><td>0.35</td><td>0.23</td><td>0.29</td><td>-0.24</td><td>0.10</td><td>-0.06</td><td>-0.96</td><td>-0.74</td><td>-0.72</td></tr><tr><td>hiring_screen</td><td>0.00</td><td>0.29</td><td>-0.13</td><td>-0.15</td><td>0.19</td><td>0.04</td><td>0.29</td><td>-0.11</td><td>0.29</td><td>-0.05</td><td>0.00</td><td>-0.08</td><td>-0.10</td></tr><tr><td>estate_repair_cases</td><td>0.27</td><td>0.14</td><td>0.07</td><td>0.14</td><td>-0.06</td><td>0.01</td><td>0.04</td><td>-0.07</td><td>-0.21</td><td>-0.05</td><td>-0.03</td><td>-0.09</td><td>-0.12</td></tr><tr><td>revenue_dispute</td><td>0.12</td><td>0.27</td><td>-0.44</td><td>-0.50</td><td>-0.77</td><td>-0.40</td><td>0.07</td><td>-0.73</td><td>-0.22</td><td>-0.32</td><td>-0.06</td><td>-0.13</td><td>-0.25</td></tr><tr><td>news_incident</td><td>0.18</td><td>0.03</td><td>0.14</td><td>0.16</td><td>0.09</td><td>0.05</td><td>-0.12</td><td>-0.02</td><td>-0.22</td><td>-0.01</td><td>-0.25</td><td>-0.34</td><td>-0.54</td></tr><tr><td>all 95 tasks</td><td>0.56</td><td>0.49</td><td>0.38</td><td>0.37</td><td>0.22</td><td>0.21</td><td>0.19</td><td>0.07</td><td>0.03</td><td>-0.12</td><td>-0.32</td><td>-0.33</td><td>-0.35</td></tr></table>

Figure 7: Mean score at L3 on each scenario, for each model; scenarios are ordered by the best model’s score, and the highestscoring cell in each row is outlined. Color runs from red (below the do-nothing baseline) to blue (at the reference solution).

![](images/4aa2a7026ce956108d675fb8e3bf96c585f79100876e058d1fa5143af0f6ee92.jpg)  
Figure 8: Model performance on Terminal-Bench 2.1, Vibe Code Bench, Finance Agent, and FDE-Gym.

## 5.7 Comparison with other benchmarks

Figure 8 compares FDE-Gym with three public benchmarks covering work that precedes delivery. These benchmarks evaluate engineering tasks in a terminal (Terminal-Bench 2.1 [2]), complete web application development from a specification (Vibe Code Bench [12]), and research tasks performed by financial analysts (Finance Agent [13]). Dashed outlines show normalized business uplift before deducting stakeholder contact cost. Models that solve up to 88%, 90%, and 61% of tasks on these three benchmarks, respectively, score 0.19–0.56 on FDE-Gym. Because task success rates and normalized business uplift are diferent quantities, the figure places them side by side on their respective scales without treating the diference as a directly comparable gap. The closest counterpart to a success rate in FDE-Gym is the proportion of resolved tasks (a score of at least 0.8 with no incident). Among the models shown in Figure 8, the leading model Claude Opus 5.5 resolves 14 of 95 tasks (15%). For a customer, the practical implication is that a model’s performance on component benchmarks is a weak guide to what it will deliver: the same models that solve most terminal, coding, and research tasks improve the customer’s business metric in at most one of seven engagements.

Table 4: Prior benchmarks by the delivery capability they measure, and what FDE-Gym adds to each.
<table><tr><td>Capability</td><td>Representative benchmarks</td><td>What they measure</td><td>What FDE-Gym adds</td></tr><tr><td>Framing</td><td>Ambig-SWE [7], LHAW [8], SWE-RPG [17], UserBench [18], ReqElicitGym [19], XYEval [11]</td><td>Whether agents ask when information is missing, elicit what a user has not said, or resist a misleading suggestion</td><td>A misdirected request inside a full engagement; a link from the framing error to a business outcome</td></tr><tr><td>Discovery</td><td>TheAgentCompany [5], Workspace-Bench [20], RUBICON [21], DRBench [22] Implicit Intelligence [23]</td><td>Finding information and unstated constraints across systems, files, and simulated colleagues</td><td>Sources that contradict one another, become outdated, or represent a particular position; a stakeholder contact cost for asking</td></tr><tr><td>Solution design</td><td>WorkArena++ [24], CRMArena-Pro [6], EnterpriseBench [25],</td><td>Completing knowledge work and operations in enterprise systems</td><td>Whether the solution fits the customer&#x27;s organization and constraints</td></tr><tr><td>Implementation</td><td>SWE-bench [1], SWE-Bench Pro [27], Terminal-Bench [2], SWE-Lancer [3]</td><td>Writing code that passes hidden tests</td><td>Whether the code was the right thing to write</td></tr><tr><td>Deployment and operations</td><td>FDE-Bench [4], ITBench [28], AIOpsLab [29]</td><td>Building deployable artifacts that pass gated checks; diagnosing and mitigating incidents</td><td>Customer interaction, requirements discovery, and business outcomes</td></tr><tr><td>Delivery</td><td>ττ-bench [9], Trains but Doesn&#x27;t Learn [10]</td><td>Building an agent, or a fine-tuned model, for a client and judging what was delivered after deployment</td><td>Task instructions varying from a full statement of the rules to a request that may be misdirected; uplift on a business metric minus stakeholder contact cost; many</td></tr><tr><td>Outcome</td><td>Vending-Bench [30], E-Commerce Bench [31], GDPval [32], Remote Labor Index [33]</td><td>Profit from running a simulated business; expert judgment of a deliverable</td><td>sectors and forms of deliverable An engineering deliverable that produces the measured outcome; reproducibility without human judges</td></tr></table>

## 6 Related Work

Existing benchmarks each measure one stage of a delivery. Table 4 arranges them by the capability they isolate. To our knowledge, none joins these stages into a single causal chain in which a misreading of the problem renders good code worthless and the loss appears on a business metric.

The most closely related work was published within weeks of ours. τ<sup>τ</sup>-bench [9] makes agent construction the task: a developer agent receives a business’s records, a client who holds the requirements, an existing codebase, and limits on serving cost, and the customer-service agent it delivers is then deployed against held-out simulated users. Its strongest configuration passes 23.9% of simulations, compared with 82.2% for an expert-authored reference; we likewise find the best model well short of a reference written by experts. FDE-Gym difers in what it varies and in what it measures. The task instruction varies from a full statement of the rules to a request that may be misdirected, stakeholder contact incurs a charge, the result is measured as uplift on the customer’s business metric relative to a do-nothing baseline of zero, which a delivery can fall below, and the scenarios span 18 sectors and deliverables beyond customer-service agents. Trains but Doesn’t Learn [10] places an agent in the role of a forward-deployed engineer for model post-training and identifies the failure mode in which every check continues to pass while the delivered model is no better than the base model; our literal-compliance anchor is designed to expose the same kind of delivery.

Two lines of work bear directly on our task instruction levels. XYEval [11] turns existing benchmarks, τ<sup>2</sup>-bench [34] among them, into XY problems, in which the user asks about an attempted solution rather than the real problem, and reports relative drops of up to 46.7%. Our leading models lose little when the sponsor’s request is misdirected, perhaps because, within a full engagement, the documents, data, and stakeholders around the sponsor are all available to contradict it. UnderSpecBench [35] holds the environment and the correct action fixed while degrading the instruction along three axes, an isolation strategy that our task instruction levels also use, and finds that underspecification leads agents to guess rather than fail; Implicit Intelligence [23] reports a best pass rate of 48.3% on simple requests whose constraints must be discovered. Both are consistent with where our levels locate the largest drop: at the constraints that nobody stated.

Three of the benchmarks in Table 4 shaped FDE-Gym directly. TheAgentCompany [5] is the closest in form, a selfcontained simulated company with colleagues to talk to; we keep the self-contained world and replace “a colleague assigned this task” with “a customer hired you”, and task completion with a business metric. LHAW [8] is the most systematic treatment of missing information and models the cost of asking; our L0 to L1 step reproduces its finding that agents rarely recover full-specification performance by asking, and our L3 adds a situation it does not cover: a request that is misdirected. Vending-Bench [30] showed that an executable business simulation can replace human judges; we attach one to an engineering deliverable.

Two methodological warnings also shaped the design. Results obtained against LLM-played users depend on which model plays the user [14], which is why our stakeholders are scripted, and single-run success overstates reliability [36, 37], which is why every task is run three times. The admission anchors follow the practice of auditing an agentic benchmark with do-nothing and shortcut solutions before trusting its scores [38]. Beyond single benchmarks, long horizon evaluations track the length of software tasks agents can complete [39], and Collaborative Gym [40] studies agents working alongside people; FDE-Gym instead sets people and agents the same tasks separately, so that the two can be compared.

## 7 Limitations

A simulated customer. Stakeholders are scripted so that results are reproducible. Real people are less predictable: they change their minds, answer a diferent question from the one asked, and may respond diferently on the basis of previous interactions. The contact costs are set by the experts who wrote each scenario and are a model of the cost of stakeholder time, not a measurement of it.

Sample size across levels. Levels L0 to L2 are reported on the 20 original tasks only, and for three of these the L2 result is taken from the identical L3 task instruction. The drop from L0 to L1 is large relative to this sample; the diferences among L1, L2, and L3 are not large, and we claim only that no large further drop is visible.

No human baseline yet. This version reports models only. The scale is anchored by the do-nothing baseline and the reference solution, which shows that each task is solvable with full knowledge of the underlying facts; how far human experts can proceed when given only the sponsor’s request has not yet been measured.

A single agent framework. All models run in the same minimal agent framework. The results describe these models in that framework, not the products built around them, and a stronger framework may change both the scores and the ranking.

Isolation and cost. Runs were executed on several hosts with process-level isolation of varying strength; each run’s record is audited after grading, but one container per run would provide a stronger guarantee. Model usage cost is as metered on the route through which each model was accessed and is not a comparison of list prices.

Coverage and contamination. Twenty scenarios across 18 sectors are a sample of delivery work, weighted toward agents and workflows in regulated operations. The scenarios are adapted from real deliveries, and although companies are fictional and variants change the facts, we cannot rule out that related material is in the training data.

## 8 Conclusion

FDE-Gym measures an agent’s delivery the way a customer does: not by whether a system was delivered, but by whether the customer’s business improved. Measured in this way, the last mile of AI delivery remains long. The best of the thirteen configurations has an average FDE-Gym score of 0.56 and resolves about one task in seven. The gap is not where we expected it to be. Leading models show little performance loss when the sponsor’s request is misdirected, and they obtain most of the facts held by stakeholders. Their principal performance losses arise when translating those facts into an efective system and from the stakeholder contact costs incurred during delivery. Weaker models lose more: on most tasks, the customer would have been better served by no delivery at all.

These are the capabilities an experienced engineer brings to a customer. Closing the gap is therefore a question of judgment in delivery, and FDE-Gym makes that judgment measurable: one number per task, on the customer’s own metric, for people and agents under the same rules.

## Acknowledgments and Disclosure

Competing interests. DeepLethe develops and operates the FDE-Gym online platform and maintains its leaderboard. None of the evaluated models is developed by an organization with which any author is afiliated.

Anonymous contributors. Several industry experts who contributed to the scenarios are not listed as authors because of confidentiality obligations, the sensitivity of the projects involved, their employers’ policies, or their own preferences. We are grateful for their expertise and their contributions to this benchmark.

## References

[1] Carlos E. Jimenez, John Yang, Alexander Wettig, et al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In International Conference on Learning Representations (ICLR), 2024.

[2] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, et al. Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces. arXiv preprint arXiv:2601.11868, 2026.

[3] Samuel Miserendino, Michele Wang, Tejal Patwardhan, and Johannes Heidecke. SWE-Lancer: Can Frontier LLMs Earn \$1 Million from Real-World Freelance Software Engineering? In International Conference on Machine Learning (ICML), 2025

[4] Weihang Ding, Junfei Zhan, Yueting Li, and Qirong Guo. FDE-Bench: Evaluating LLM Agents for Deployment Environment Configuration. arXiv preprint arXiv:2609.27571, 2026.

[5] Frank F. Xu, Yufan Song, Boxuan Li, et al. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025.

[6] Kung-Hsiang Huang, Akshara Prabhakar, Onkar Thorat, et al. CRMArena-Pro: Holistic Assessment of LLM Agents Across Diverse Business Scenarios and Interactions. Transactions on Machine Learning Research (TMLR), 2025.

[7] Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, and Graham Neubig. Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering. In International Conference on Learning Representations (ICLR), 2026.

[8] George Pu, Michael S. Lee, Udari Madhushani Sehwag, et al. LHAW: Controllable Underspecification for Long-Horizon Tasks. arXiv preprint arXiv:2602.10525, 2026.

[9] Quan Shi, Keshav Dhandhania, Karthik Narasimhan, and Victor Barres. τ<sup>τ</sup>-Bench: An Environment for End-To-End, Realistic Agent Construction. arXiv preprint arXiv:2609.04611, 2026.

[10] Weihang Ding and Junfei Zhan. Trains but Doesn’t Learn: A Post-Training Delivery Benchmark for LLM Agents as Forward-Deployed Engineers. In Conference on Empirical Methods in Natural Language Processing (EMNLP), Industry Track, 2026.

[11] Zhengxuan Wu, Yuxuan Li, Oyvind Tafjord, and Been Kim. XYEval: Agents say yes to bad advice. arXiv preprint arXiv:2609.23939, 2026.

[12] Hung Tran, Langston Nashold, Rayan Krishnan, Antoine Bigeard, and Alex Gu. Vibe Code Bench: Evaluating AI Models on End-to-End Web Application Development. In ACM CAIS, 2026.

[13] Antoine Bigeard, Langston Nashold, Rayan Krishnan, and Shirley Wu. Finance Agent Benchmark: Benchmarking LLMs on Real-world Financial Research Tasks. arXiv preprint arXiv:2508.00828, 2025.

[14] Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, and Seraphina Goldfarb-Tarrant. Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations. In Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2026.

[15] Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI Agents That Matter. Transactions on Machine Learning Research (TMLR), 2025

[16] Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, et al. Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation. In International Conference on Learning Representations (ICLR), 2026.

[17] Xin Zhou, Chun Yong Chong, Kisub Kim, et al. A Unified Issue Resolution Benchmark for Requirement Clarification, Planning, and Code Generation for Coding Agents. arXiv preprint arXiv:2608.09072, 2026.

[18] Cheng Qian, Zuxin Liu, Akshara Prabhakar, et al. UserBench: An Interactive Gym Environment for User-Centric Agents. arXiv preprint arXiv:2507.22034, 2025.

[19] Dongming Jin, Zhi Jin, Zheng Fang, et al. ReqElicitGym: An Evaluation Environment for Interview Competence in Conversational Requirements Elicitation. arXiv preprint arXiv:2602.18306, 2026.

[20] Zirui Tang, Xuanhe Zhou, Yumou Liu, et al. Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies. arXiv preprint arXiv:2605.03596, 2026.

[21] Fabian Wenz, Felix Treutwein, Çagatay Demiralp, and Michael Stonebraker. RUBICON: Agentic AI for Messy Enterprise Data. arXiv preprint arXiv:2604.21413, 2026.

[22] Amirhossein Abaskohi, Tianyi Chen, Miguel Muñoz-Mármol, et al. DRBench: A Realistic Benchmark for Enterprise Deep Research. arXiv preprint arXiv:2510.00172, 2025.

[23] Ved Sirdeshmukh and Marc Wetter. Implicit Intelligence: Evaluating Agents on What Users Don’t Say. arXiv preprint arXiv:2602.20424, 2026

[24] Léo Boisvert, Megh Thakkar, Maxime Gasse, et al. WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[25] Min Yang, Yichen Pan, Jinghua Piao, Dandan Song, Yongshun Gong, and Yong Li. EnterpriseBench: Benchmarking LLM Agents on Enterprise-Level Strategic Reasoning and Decision-Making. arXiv preprint arXiv:2609.37658, 2026.

[26] Bertie Vidgen, Austin Mann, Abby Fennelly, et al. APEX-Agents. arXiv preprint arXiv:2601.14242, 2026.

[27] Xiang Deng, Jef Da, Edwin Pan, et al. SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? arXiv preprint arXiv:2509.16941, 2025.

[28] Saurabh Jha, Rohan Arora, Yuji Watanabe, et al. ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks. In International Conference on Machine Learning (ICML), 2025.

[29] Yinfang Chen, Manish Shetty, Gagan Somashekar, et al. AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds. In Conference on Machine Learning and Systems (MLSys), 2025.

[30] Axel Backlund and Lukas Petersson. Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous Agents. arXiv preprint arXiv:2502.15840, 2025.

[31] Wei Fan, Xinjie Shen, Xudong Guo, et al. E-Commerce Bench: Evaluating LLM Agents on Long-Horizon Autonomous Business Operation. arXiv preprint arXiv:2608.30730, 2026.

[32] Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, et al. GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks. arXiv preprint arXiv:2510.04374, 2025.

[33] Mantas Mazeika, Alice Gatti, Cristina Menghini, et al. Remote Labor Index: Measuring AI Automation of Remote Work. arXiv preprint arXiv:2510.26787, 2025.

[34] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv preprint arXiv:2506.07982, 2025.

[35] Zimo Ji, Zekai Zhang, Congying Xu, et al. Coding Agents Are Guessing: Measuring Action-Boundary Violations in Underspecified DevOps Instructions. arXiv preprint arXiv:2607.02294, 2026.

[36] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv preprint arXiv:2406.12045, 2024.

[37] Stephan Rabanser, Sayash Kapoor, Peter Kirgis, Kangheng Liu, Saiteja Utpala, and Arvind Narayanan. Towards a Science of AI Agent Reliability. In International Conference on Machine Learning (ICML), 2026.

[38] Yuxuan Zhu, Tengjun Jin, Yada Pruksachatkun, et al. Establishing Best Practices for Building Rigorous Agentic Benchmarks. arXiv preprint arXiv:2507.02825, 2025.

[39] Thomas Kwa, Ben West, Joel Becker, et al. Measuring AI Ability to Complete Long Software Tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[40] Yijia Shao, Vinay Samuel, Yucheng Jiang, John Yang, and Diyi Yang. Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration. In International Conference on Learning Representations (ICLR), 2026.

Image credits (Figure 1). Photographs from Wikimedia Commons, cropped. Energy: “North Sea Oil Rig”, Gary Bembridge, CC BY 2.0; Healthcare: “Medical Care and Compassion in a Hospital Bed provided by Nova Ukraine”, NUSF, CC0; Robotics: “BRS2 Amazon fulfilment centre in Swindon”, Auledas, CC BY 4.0; EV charging: “Flo EV fast charging station at Hingham Depot”, 4300streetcar, CC BY 4.0; Water: “Three Gorges Dam”, Gaynor, CC BY 2.0; Logistics: “UPS worker unloads boxes from semi trailer truck onto conveyor belt”, U.S. Department of Agriculture, Public domain; Construction: “Johor Bahru–Singapore Rapid Transit System construction site”, Renek78, CC0; Power grid: “Switchyard of electric substation 750kV”, Novoklimov, CC BY 4.0; Banking: “Safe Deposit Boxes”, Fixedsun, CC0; Media: “The newsroom at DR”, James Cridland, CC BY 2.0; Manufacturing: “Tesla Motors Assembly Line”, Steve Jurvetson, CC BY 2.0; Property: “Xujiahui Shanghai September 2023”, 4300streetcar, CC BY 4.0.