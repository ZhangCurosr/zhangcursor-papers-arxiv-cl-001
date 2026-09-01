# GPAGENTBENCH-2K: Benchmarking Large Language Model Agents in Complex Clinical Action Space

Boqi Chen<sup>1,3</sup>\*, Xudong Liu<sup>2</sup>\*, Yunke Ao<sup>1</sup>, Heejin Do<sup>1</sup>, Jianing Qiu<sup>3</sup> <sup>1</sup>ETH Zurich, <sup>2</sup>University of Toronto, <sup>3</sup>MBZUAI Correspondence: jianing.qiu@mbzuai.ac.ae

## Abstract

Large Language Models (LLMs) show great potential as clinical agents, yet existing benchmarks reduce clinical workflows to static predictions or unconstrained Markov Decision Processes (MDPs) with coarse action sets. To address this, we introduce GPAGENTBENCH-2K, the first Constrained MDP (CMDP) LLM-agent benchmark for primary-care clinical decisionmaking, constructed from expert-validated records of real-world GP encounters. Our environment models a full spectrum of six foundational clinical actions, imposes a topological workflow prior over the action space, and operationalizes safety-informed abstention as a first-class outcome. Evaluating 16 state-ofthe-art LLMs reveals a significant performance degradation as the action space scales. Crucially, we uncover a clinical quality-safety gap: even frontier models with the highest diagnosis accuracy violate safety constraints in over half of high-risk cases. Finally, we establish a reference point using Constrained Group Relative Policy Optimization (C-GRPO), and show that while explicitly modeling constraints improves performance over unconstrained RL methods, it remains far from clinically acceptable safety.

0 https://github.com/jianing -lab/GPAgentBench

## 1 Introduction

Large Language Models (LLMs) have shown great potential as clinical agents capable of autonomously eliciting patient information, retrieving evidence, and executing multi-step reasoning through tool use (Qiu et al., 2024). Yet existing clinical agent benchmarks fail to capture the complexity of real-world clinical workflows, largely due to unrealistic and oversimplified problem formulation. Most existing frameworks (Kim et al., 2024; Yan et al., 2026; Zhou et al., 2025; Chen et al.,

![](images/3647afe49b9ef53ce4a081bac1ba81d8edebb4ecff54ee4969c69f1c19580089.jpg)

(a)  
![](images/da533ce22ad5acd7f6ea105670cd25a0ef646e93d584a8bfc6d0009891659572.jpg)  
(b)  
Figure 1: (a) Comparison: Our CMDP models a full clinical action space, including recursive evidence gathering and safety-abstention. (b) Findings: Diagnosis accuracy drops substantially as action space scales.

2026) cast diagnosis as a static prediction problem, a non-Markov Decision Process (MDP) formulation that assumes comprehensive, structured patient data (e.g., Electronic Health Records) is available upfront. This reduces clinical decision-making to a single-step prediction task, entirely bypassing the sequential process of active evidence-gathering.

To better reflect this process, recent works formulate the problem as an MDP, where agents must interact with the environment to reach a diagnosis. This interaction can occur either between a simulated patient and a doctor agent (Feng et al., 2025) or among multiple specialized agents in a collaborative pipeline (Fan et al., 2025; Bani-Harouni et al., 2025). Nevertheless, these environments cannot fully capture real-world complexity due to two major limitations. First, they typically collapse the action space into a coarse action sets. For instance,

DoctorAgent (Feng et al., 2025) defines a minimal two-action set (ask and diagnose). In reality, even a General Practitioner (GP), often the first point of contact and a foundational role in healthcare systems, navigates a complex action space where they must gather evidence from incomplete presentations, judiciously order tests, and prescribe treatment after diagnosis. Second, these environments operate as an unconstrained MDP with a single, absolute goal: finding the correct disease. Even when secondary metrics such as cost are included (Bani-Harouni et al., 2025), they are treated as soft additive penalties. Real-world clinical workflow, however, is inherently a Constrained MDP (CMDP) (Achiam et al., 2017), where the agent must optimize diagnostic performance within strict operational boundaries. For example, prescribing treatment before sufficient evidence, ordering excessive diagnostic tests, or failing to refer a highrisk patient all constitute clinically unsafe behavior that cannot be retroactively compensated by downstream task performance. The objective is therefore not to maximize diagnostic accuracy in isolation, but to maximize it subject to safety constraints.

To operationalize this setting, we introduce GPAGENTBENCH-2K, a clinical CMDP environment constructed from expert-validated records of real-world GP encounters. Unlike conventional benchmarks that assume a fixed diagnostic trajectory, our benchmark spans the full GP decision spectrum via six foundational actions: ask, body\_exam, test, diagnose, treat, and refer. In addition, it imposes a topological workflow prior through action masking, and explicitly models multiple valid terminal outcomes, including diagnoseand-manage for safely manageable conditions and abstain-and-refer for safety-informed escalation. This design enables evaluation of diagnosis (Dx), treatment (Tx) accuracy and safety and efficiency.

Our CMDP environment also presents unique challenges for reinforcement learning (RL)-based fine-tuning. Specifically, unconstrained RL methods such as Group Relative Policy Optimization (GRPO) (Guo et al., 2025) optimize a scalar reward over an unconstrained action space. Encoding strict safety requirements as additive penalties induces a dangerous compensatory tradeoff where the policy simply absorbs safety violations if the accuracy reward is sufficiently large, effectively trading safety for task performance. To contextualize this challenge, we establish a reference baseline using Constrained GRPO (C-GRPO) (Girgis et al., 2026), a tailored adaptation that augments GRPO with a Lagrangian relaxation to explicitly penalize safety violations. Our results suggest that while C-GRPO improves over unconstrained RL methods, absolute safety-failure rates remain high, demonstrating that achieving clinically acceptable safety in a complex action space remains an unsolved problem.

Our contributions are summarized as follows:

1 We introduce GPAGENTBENCH-2K, the first CMDP LLM-agent benchmark for primarycare clinical decision-making, which models the full spectrum of primary clinical actions and operationalizes safety-informed abstention as a first-class outcome using clinicianvalidated real-world GP encounter data;

2 We evaluate 16 state-of-the-art LLMs, finding that clinical quality degrades significantly in a complex action space. Crucially, we reveal a quality-safety gap, when even frontier models with the highest Dx accuracy violate safety constraints in over half of high-risk cases;

3 We establish a reference point by benchmarking C-GRPO and show that despite improvement over unconstrained RL baselines, absolute safety-failure rates remain high, suggesting that achieving clinically acceptable safety in complex a action space remains unsolved.

## 2 Related Work

Non-MDP Clinical Agent Environment. Many existing works cast clinical decision-making as a static prediction problem (a non-MDP formulation), assuming comprehensive patient data is available upfront. Some frameworks use LLMs to assume specialized personas (e.g., generalists vs. specialists) for collaborative problem-solving (Lu et al., 2024; Zhou et al., 2025; Zhao et al., 2025; Zhu et al., 2025). While these multi-agent designs enhance capability beyond a single model (Wang et al., 2024), partitioning the workflow into narrow roles confines individual agents to an oversimplified action space, entirely bypassing the core challenges of real-world clinical decision-making: the sequential, active evidence-gathering and balancing diagnosis against safety-informed abstention under uncertainty. A detailed comparison of per-agent action space is provided in Appendix B.

MDP Clinical Agent Environment. A complementary line of work models clinical workflows as MDPs. These environments simulate active evidence gathering, either through multi-turn patientdoctor dialogue (Feng et al., 2025; Fan et al., 2025) or iterative diagnostic test selection (Bani-Harouni et al., 2025). Despite advancing beyond static prediction, these environments share two critical limitations that prevent them from reflecting real-world complexity. First, individual agents’ action space remains oversimplified. For example, DoctorAgent (Feng et al., 2025) collapse the action space to a minimal two-action set (ask and diagnose). Second, these environments operate as unconstrained MDPs with a single determined terminal outcome: outputting a diagnosis. This unconstrained formulation allow agents to freely trade safety for diagnostic accuracy. In contrast, GPAGENTBENCH-2K is the first CMDP benchmark for clinical agent systems, requiring agents to navigate a full-spectrum clinical action space while optimizing diagnostic performance within strict safety boundaries.

![](images/990d50ecfda7a0c54d3a6ca3e842a37e872f07f18a5e03b13f95d533f8da67d3.jpg)  
Figure 2: Overview of GPAGENTBENCH-2K. (1) Each case pairs an expert-validated GP record with a sampled patient persona; the ground-truth diagnosis and disposition is hidden from the agent. (2) Agent produces trajectories scored by diagnosis and treatment rewards $( R _ { D x } , R _ { T x } )$ under a safety cost $C _ { \mathrm { s a f e } }$ and a diagnostic test cost $C _ { \mathrm { d i a g } } .$

## 3 Benchmark Construction

Data Collection. GPAGENTBENCH-2K was constructed using real-world GP visit records from a primary-care network. These records span 9 primary departments and 62 secondary departments. Each record encompasses patient demographics, symptoms, medical history, physical exam findings, diagnostic test results, and subsequent patient management plans. Unlike previous diagnosis benchmarks that always assume a single correct terminal prediction (e.g., forced diagnosis), our dataset explicitly incorporates multiple valid clinical outcomes, including both successful diagnoseand-manage and safety-informed abstain-and-refer decisions. Specifically, the diagnose-and-manage trajectory captures cases where the GP provides a definitive diagnosis and treatment plan for safely manageable conditions, while the abstain-and-refer trajectory captures complex cases requiring escalation to specialists. This branching nature enables the evaluation of the agent’s ability to dynamically select actions based on evolving evidence, rather than being constrained to a single predefined trajectory.

To ensure data quality, collected cases undergo rigorous validation by two senior physicians. For the diagnose-and-manage subset, cases are evaluated on evidence completeness (i.e., if the diagnosis can be confirmed given the clinical evidence), disease prevalence in primary care, diagnosis correctness, and treatment suitability. For the abstain-andrefer subset, validation is based on the clinical necessity of the referral and the correctness of the chosen specialty (details in Appendix D.7). Only cases satisfying all respective criteria are retained. After this filtering, we obtain 2428 high-quality cases, featuring an average of 4.9 symptoms per case, with patients’ chief complaints including an average of 2.2 symptoms and a total of 1,050 unique symptoms represented. Additionally, each case includes an average of 4.1 physical exams per patient, drawn

Size of Action Space

from $^ 6$ core assessments (general appearance, pulmonary, abdominal, extremities, neurological, and cardiovascular), and an average of 2.9 diagnostic tests, with a total of 584 unique test items across 4 different types (blood, imaging, functional and pathological test). The comprehensive statistics of the dataset is in Appendix A.4. Finally, all cases undergo de-identification to ensure patient privacy.

Data Preprocessing. We employ Gemini-3- Flash (Google, 2025) to transform clinical records into standardized JSON files, establishing a direct mapping from free-text descriptions to structured clinical facets that encompasses two primary domains: patient simulation (e.g., demographics, patient symptoms and medical history) and clinical observations $( e . g .$ , vital signs, physical exam findings and diagnostic test results). We strictly constrain the model to extract only explicitly stated findings and assign the null value to absent fields to avoid hallucinations. To prevent label leakage, we remove diagnostic statements and treatment recommendations from the clinical notes. Finally, we manually cross-reference all processed JSON files with original clinical records to ensure completeness and mitigate hallucinations, and a randomly selected 20% subset is further inspected by clinical experts. The template prompts and examples of processed data are in Appendix A.2 and A.3.

## 4 Environment Design

## 4.1 Persona-Driven Patient Simulator.

Real consultations are shaped by both underlying conditions and how patients communicate. Their temperament, language skills (Wilson et al., 2005), history-recall ability, and cognitive clarity (Curcio et al., 2019) dictate what evidence a GP can reliably elicit (Hampton et al., 1975; Peterson et al., 1992). To simulate this complexity, we augment structured patient data with diverse personas (Kyung et al., 2026) spanning four dimensions: (1) personality: impatient, overanxious, verbose, distrustful, pleasing (minimizing symptoms), or neutral; (2) language proficiency: basic, intermediate, or advanced; (3) anamnestic fidelity: high or low; and (4) cognitive disorientation: normal, moderate, or high. This persona-driven simulator forces agents to navigate realistic communication hurdles; e.g., a disoriented patient with low anamnestic fidelity may omit critical history, requiring the agent to ask targeted clarifying questions.

![](images/74107c86c4a77dc1b62f783439163a0e558fe2e8bb97615587552c4cc60a1aab.jpg)

![](images/2599363a9118fae883d3a8da937360b90acda2ba1f8a02af76bc1c2c88196f66.jpg)  
Figure 3: Impact of action space complexity on performance. Clinical quality degrades as parsed action set grows from a minimal 2-action set (diagnose, treat) to the full 6 actions (+ ask, test, body\_exam, refer).

To validate patient simulator fidelity, we conduct an independent physician audit of randomly sampled complete interaction trajectories, assessing factual consistency, clinical realism, and persona adherence. The audit shows high performance across all three dimensions, with strong inter-rater agreement between the two senior physicians (see Appendix C.4 for details).

## 4.2 Constrained Markov Decision Process.

Our environment is formalized as a CMDP defined by a tuple $( \mathcal { S } , \mathcal { A } , d , P , R , \{ C _ { k } \} _ { k = 1 } ^ { K } )$ . Here, $\boldsymbol { \mathcal { S } }$ is the state space, A is the action space, d is the initial state distribution, and $P ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$ denotes the transition dynamics. At each step $t ,$ the agent selects an action $a _ { t }$ conditioned on the current state $s _ { t } .$ , according to a policy $\pi ( \boldsymbol { a } _ { t } \mid \boldsymbol { s } _ { t } )$ The environment yields a reward $R ( s _ { t } , a _ { t } )$ and $K$ distinct cost functions $C _ { k } ( s _ { t } , a _ { t } )$ . Episodes are capped at a maximum of T action steps. Let $\tau = ( s _ { 0 } , a _ { 0 } , s _ { 1 } , . . . , a _ { T - 1 } , s _ { T } )$ denote a trajectory sampled from the distribution $p _ { \pi } ( \tau ) \ =$ $\begin{array} { r } { d ( s _ { 0 } ) \prod _ { t = 0 } ^ { T - 1 } \pi ( a _ { t } ~ \vert ~ s _ { t } ) P ( s _ { t + 1 } ~ \vert ~ s _ { t } , a _ { t } ) } \end{array}$ . The expected return and costs for policy π are defined respectively as $\begin{array} { r } { J _ { R } ( \pi ) = \mathbb { E } _ { \tau \sim p _ { \pi } } \big [ \sum _ { t = 0 } ^ { T - 1 } R ( s _ { t } , a _ { t } ) \big ] } \end{array}$ and $J _ { C _ { k } } ( \pi ) \ = \ \mathbb { E } _ { \tau \sim p _ { \pi } } \big [ \sum _ { t = 0 } ^ { T - 1 } C _ { k } ( s _ { t } , a _ { t } ) \big ]$ . The CMDP objective is to maximize expected return subject to feasibility thresholds $\epsilon _ { k }$ for each cost:

$$
\operatorname* { m a x } _ { \pi } J _ { R } ( \pi ) \quad \mathrm { s . t . } \quad J _ { C _ { k } } ( \pi ) \leq \epsilon _ { k } , \forall k \in \mathcal { K } .\tag{1}
$$

where K = {safety, diagnostic\_cost} is the set of cost terms. This objective defines the benchmarklevel constrained decision-making problem; in zero-shot evaluation, we report the realized reward and cost metrics, while in RL fine-tuning, we use relaxed constrained optimization as described in Section 5.2. Specifically, the components of the CMDP are instantiated as follows:

State (S): The state represents the accumulated dialogue history, encompassing the initial patient presentation (subject to d, i.e., the joint distribution over clinical records and persona configurations) alongside subsequent interactions, physical observations, and diagnostic test results revealed up to the current turn.

Action (A): Formally, the action space contains six distinct clinical actions: ask, body\_exam, test, diagnose, treat, and refer. The agent must generate XML-style tags (e.g., <action $\mathrm { t y p e } { = } ^ { \prime \prime } \mathrm { t e s t } ^ { \prime \prime } { > } . . . < / \mathsf { a c t i o n } { > } )$ containing required fields $( e . g .$ , ordered test). A topological workflow prior designed by clinical experts defines a state-dependent admissible action set $\mathcal { M } ( s _ { t } ) \subseteq \mathcal { A }$ to model the iterative and branching structure of primary-care decision-making. Each episode begins in the evidence-collection phase, where the agent may recursively execute information-gathering actions (ask, body\_exam, and test). After collecting sufficient evidence, the agent can either take a refer action to abstain from diagnosis and immediately terminate the episode, or take a diagnose action, advancing to the treatment phase where treat becomes the terminal action. In both zero-shot evaluation and RL finetuning, actions are restricted to $\mathcal { M } ( s _ { t } )$ as a benchmark control, rather than as a claim of autonomous workflow compliance. This avoids confounding clinical decision-making with invalid workflow transitions or parser failures. For body\_exam and test, available targets are record-derived; unavailable requests return unavailable results rather than fabricated evidence.

Transition (P): The transition dynamics updates the accumulated state by appending environment observations, consisting of either natural language responses from the virtual patient (for ask) or structured medical results (for body\_exam and test).

Reward (R): The reward signal measures clinical quality, including diagnosis (Dx) accuracy, treatment (Tx) accuracy, and management accuracy.

![](images/c8de0714e74ae6b73a2934b18cac11b7a9a5272cb8548d21dd67f83928a1b064.jpg)  
Figure 4: The referral and local management preference across evaluated models.

Dx and Tx are assigned by an LLM judge against the ground-truth diagnosis and treatment labels. Management accuracy is a binary episode-level metric indicating whether the agent selects the correct terminal management pathway, $i . e . ,$ treat for diagnose-and-manage cases and refer for abstainand-refer cases. Detailed definitions for reward functions are provided in Appendix C.2.

To mitigate circular evaluation risk, we validate the LLM judge on randomly sampled outputs across all evaluated models with human experts. The audit shows strong agreement with two senior physicians (Cohen’s κ = 0.92 and κ = 0.89 for Dx accuracy and Tx score, respectively). Re-scoring the frontier models with a different LLM judge under the same rubric yields changes within the reported standard errors while preserving model rankings (see Appendix C.3 for details). Together, these results support the robustness of our LLMbased evaluation.

Costs $( C _ { k } ) \colon$ The cost signals measure safety failures and diagnostic efficiency. Safety costs penalize clinical risks, including missed referral (a harmful miss where the agent manage a patient requiring a referral locally) and missed local management (an unnecessary referral). Efficiency costs penalize resource overutilization by tracking the total dialogue turns and the monetary cost of ordered diagnostic tests per episode. Detailed definitions for these cost functions are provided in Appendix C.2.

For zero-shot benchmarking of off-the-shelf LLMs, we simply sample rollouts of the policy within the environment, collecting and aggregating the reward and cost terms as evaluation metrics.

## 5 Experiments

Setup. In all experiments, the patient agent is instantiated with GPT-4o-mini (OpenAI, 2024), and each episode is capped at a maximum of 10 dialogue turns (see Appendix C.1 for details). The turn cap standardizes interaction length across models and prevents degenerate endless-questioning policies; we therefore report average turns as an efficiency metric. The Dx accuracy and Tx score are computed using GPT-5.4 as the LLM judge, whereas the management and safety metrics are computed deterministically against the groundtruth management label, with referral destinations evaluated by string match.

## 5.1 Zero-Shot Evaluation Results

We evaluate off-the-shelf LLMs on a 20% random test split stratified by management label (284 diagnose-and-manage and 201 abstain-and-refer cases). For proprietary and large-scale open-source models (DeepSeek, Kimi and MiniMax), we use the official API. For the remaining, we conduct evaluation on 4 A6000 GPUs with 48GB memory.

Clinical Quality Drops as Action Space Scales. Figure 3 shows Dx accuracy and Tx score as the parsed action set grows from a minimal two-action set (diagnose, treat) to six actions. Only the admissible action set varies across settings; test cases and patient personas are fixed. The clinical evidence normally obtainable through omitted action type is provided upfront in the initial presentation, so that degradation reflects action-space complexity rather than missing information. We can observe a monotonic decrease in both Dx accuracy and Tx score consistent across all models, with Dx accuracy drops up to 21 percentage point (pp). Specifically, the two largest single-step Dx accuracy drops occur when the action ask is introduced (an average of 4.1 pp) to force the agent to actively elicit patient information, effectively transforming the task from a static prediction to an interactive decision-making problem, and when test is added (an average of 7.0 pp across all models), which greatly expands the action space by introducing a massive combination of diagnostic exams.

Conversely, the largest single-step Tx score drop (an average of 2.9 pp) occurs when an alternative terminal action refer is introduced, indicating the models’ inability to calibrate safety-abstention, recognizing when not to act. Notably, even topperforming models such as GPT-5.4 are no exception to this performance drop, suggesting this branching, sequential complexity of real-world clinical workflows cannot be overcome by mere prompt engineering.

LLM Agents Exhibit a Clinical Quality-Safety Gap. Table 1 reveals a critical gap between clinical quality and safety. Crucially, no model is Pareto-dominant across all dimensions, and models largely fall into two uncalibrated extremes: the over-confident and the over-cautious. While proprietary models demonstrate high clinical quality, with Claude Opus 4.7 leading Dx accuracy at 70.0% and GPT-5.4 leading Tx score at 53.2%, they exhibit a dangerous overconfidence bias. As shown in Figure 4, models like Claude Opus 4.7 and Sonnet 4.6 (bottom-right quadrant) aggressively attempt to manage patients locally, resulting in low Miss GP rates (e.g., 13.4% for Sonnet) but missing 67.2% and 77.1% of required referrals, respectively. Conversely, models in the top-left quadrant, such as Kimi K2.6 and Huatuo-o1 (72B), fall into the overcautious cluster. Kimi K2.6 has mediocre Dx accuracy (56.7%) but achieves the second-lowest Miss Refer rate (40.3%) by defaulting to specialist referrals, consequently failing at local management (71.8% Miss GP). These results highlight that diagnostic capabilities do not equate to safe triage calibration. Models either confidently hallucinate local management or over-refer out of uncertainty.

![](images/2c1bd33a8acc8156e45aa8502d72600525bc6870a96566afe30bf7ecbc24bb44.jpg)

Scaling model parameters alone does not close the clinical quality-safety gap.

While scaling up model parameters generally improves model’s instruction following capability and clinical knowledge, we observe that it does not necessarily close the clinical quality-safety gap. For example, while a monotonic increase in both Dx Accuracy (from 40.4% to 54.1%) and Tx Score (from 18.4% to 25.4%) can be observed when scaling up model parameters of Qwen2.5 from 7B to 72B, the management accuracy fluctuate unpredictably. For example, the larger Qwen2.5 models (32B and & 72B) miss almost twice as many patients compared to the 7B model, and 72B model still fails to refer 54.7% of high-risk patients.

Furthermore, scaling does not necessarily improve clinical cost-efficiency. We observe a "shotgun medicine" effect where models like Qwen2.5 (72B) incur the highest diagnostic costs (\$473) by over-ordering tests, yet yield worse safety profiles than Kimi K2.6, which acts as a rapid triage agent by safely referring patients while spending the least on diagnostics (\$34). In addition, among opensource models, neither of the three largest evaluated model demonstrates consistent lower safety failures or cost compared to those of more than 10 times fewer parameters. These results suggest parameter scaling alone does not inherently improve model’s capability to safely navigate the complex clinical action space.

Domain-specific training requires carefully  
designed environment.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="3">Clinical Quality ↑</td><td colspan="2">Safety Failures ↓</td><td colspan="2">Efficiency ↓</td></tr><tr><td>Dx Acc. (%)</td><td>Tx Score (%)</td><td>Mgmt Acc. (%)</td><td>Miss Refer (%)</td><td>Miss GP (%)</td><td>Avg. Turns</td><td>Cost (USD)</td></tr><tr><td rowspan="4">Proprietary</td><td>GPT-5.4 (OpenAI, 2026)</td><td>64.1 (2.8)</td><td>53.2 (2.4)</td><td>56.3 (2.3)</td><td>51.2 (3.5)</td><td>38.4 (2.9)</td><td>6.91 (0.07)</td><td>$147 (10)</td></tr><tr><td>GEMINI 3.1 PRO (Google, 2026)</td><td>66.8 (3.1)</td><td>41.6 (2.0)</td><td>55.9 (2.3)</td><td>47.8 (3.5)</td><td>41.5 (2.9)</td><td>6.07 (0.07)</td><td>$231 (12)</td></tr><tr><td>CLAUDE OPUS 4.7 (Anthropic, 2025)</td><td>70.0 (2.5)</td><td>46.0 (2.0)</td><td>55.9 (2.3)</td><td>67.2 (3.3)</td><td>27.8 (2.7)</td><td>5.57 (0.06)</td><td>$253 (13)</td></tr><tr><td>CLAUDE SONNET 4.6 (Anthropic, 2025)</td><td>61.7 (1.6)</td><td>36.0 (1.8)</td><td>60.2 (2.2)</td><td>77.1 (3.0)</td><td>13.4 (2.0)</td><td>6.57 (0.05)</td><td>$218 (11)</td></tr><tr><td rowspan="10">Open-Source</td><td>General Purpose</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DEEPSEEK-V4-PRO (DeepSeek, 2026)</td><td>59.5 (2.8)</td><td>41.2 (2.0)</td><td>56.7 (2.3)</td><td>57.2 (3.5)</td><td>33.5 (2.8)</td><td>6.98 (0.07)</td><td>$146 (10)</td></tr><tr><td>KIMI K2.6 (AI, 2026)</td><td>56.7 (4.1)</td><td>49.2 (3.5)</td><td>41.2 (2.2)</td><td>40.3 (3.5)</td><td>71.8 (2.7)</td><td>8.74 (0.08)</td><td>$34 (5)</td></tr><tr><td>MINIMAX M2.7 (MiniMax, 2026)</td><td>60.0 (2.5)</td><td>31.3 (1.9)</td><td>53.0 (2.3)</td><td>78.1 (2.9)</td><td>25.0 (2.6)</td><td>6.95 (0.07)</td><td>$278 (14)</td></tr><tr><td>QwEN2.5 (72B) (Yang et al., 2025)</td><td>54.1 (3.2)</td><td>25.4 (1.8)</td><td>46.4 (2.3)</td><td>54.7 (3.5)</td><td>52.8 (3.0)</td><td>6.86 (0.09)</td><td>$473 (20)</td></tr><tr><td>QWEN2.5 (32B) (Yang et al., 2025)</td><td>49.1 (3.0)</td><td>23.3 (2.0)</td><td>39.6 (2.2)</td><td>68.2 (3.3)</td><td>54.9 (3.0)</td><td>8.85 (0.06)</td><td>$364 (17)</td></tr><tr><td>QwEN2.5 (7B) (Yang et al., 2025)</td><td>40.4 (2.5)</td><td>18.4 (1.3)</td><td>47.4 (2.3)</td><td>89.6 (2.2)</td><td>21.1 (2.4)</td><td>6.27 (0.09)</td><td>$445 (20)</td></tr><tr><td>Medical</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HUATUO-01 (72B) (Chen et al., 2024)</td><td>62.3 (3.5)</td><td>33.2 (2.4)</td><td>48.5 (2.3)</td><td>39.3 (3.5)</td><td>60.2 (2.9)</td><td>6.79 (0.08)</td><td>$325 (13)</td></tr><tr><td>LINGSHU (32B) (Xu et al.)</td><td>50.9 (2.9)</td><td>25.2 (1.9)</td><td>53.6 (2.3)</td><td>62.2 (3.4)</td><td>35.2 (2.8)</td><td>7.01 (0.08)</td><td>$387 (19)</td></tr><tr><td>MEDGEMMA (27B) (Sellergren et al., 2025)</td><td>49.5 (3.4)</td><td>29.7 (2.2)</td><td>52.0 (2.7)</td><td>65.7 (4.0)</td><td>35.2 (3.4)</td><td>8.51 (0.08)</td><td>$145 (12)</td></tr><tr><td>HuATUO-01 (7B) (Chen et al., 2024)</td><td>48.9 (5.2)</td><td>23.4 (2.6)</td><td>20.6 (2.4)</td><td>77.6 (3.0)</td><td>80.6 (2.6)</td><td>8.58 (0.08)</td><td>$124 (10)</td></tr><tr><td>LINGSHU (7B) (Xu et al.)</td><td>47.1 (2.7)</td><td>23.0 (1.7)</td><td>44.5 (2.3)</td><td>90.0 (2.1)</td><td>31.0 (2.7)</td><td>7.10 (0.10)</td><td>$484 (23)</td></tr><tr><td>DOCTORAGENT-RL (7B) (Feng et al., 2025)</td><td>41.0 (3.5)</td><td>22.4 (2.0)</td><td>15.7 (1.7)</td><td>89.1 (2.2)</td><td>81.0 (2.3)</td><td>9.44 (0.06)</td><td>$89 (9)</td></tr></table>

Table 1: Zero-shot evaluation of clinical quality, safety failures, and efficiency across off-the-shelf LLMs ( best second-best , and third-best ). Open-source models are grouped into general-purpose and medical categories. Parentheses report the standard error of the mean.

![](images/32aa48de2848de7c0de4eb63c4bed3b2eb29be362fab3ee4dbdb66991289e416.jpg)

Domain-specific fine-tuning on clinical data, including supervised fine-tuning on static medical question-answering (QA) data and RL in environments with oversimplified action space, are empirically shown to be insufficient to close the safetyaccuracy gap. For instance, while Lingshu (7B) and Huatuo-o1 (7B) effectively boost Dx Accuracy and Tx Score over their base Qwen2.5 (7B) model, both their overall management accuracies drop. Notably, the management accuracy of Huatuo-o1 (7B) drops by more than half to 20.6%, with both Miss Refer and Miss GP rate exceeding 77%. For larger models, Huatuo-o1 (72B) outperforms its base Qwen2.5 (72B) model by a small margin (2.1pp), whereas the Miss GP rate increases by 7.4 pp. Together, these results suggest that Medical QA pretraining does not directly transfer to sequential, safetyconstrained clinical decision-making, where the bottleneck lies in policy optimization under partial observability and strict safety/cost constraints, rather than a lack of factual medical knowledge.

Conversely, sequential policies learned through RL may transfer poorly when trained in an oversimplified action space. We therefore view DoctorAgent-RL (7B) as a transfer baseline trained in a restricted two-action environment, rather than as a direct comparison of RL algorithms. Specifically, since it was fine-tuned on a strict dialogue dataset limited exclusively to ask and diagnose, entirely omitting foundational actions like physical exams or ordering tests, its resulting policy shows minimal improvement over the base Qwen2.5 (7B) in Dx accuracy (41.0% vs. 40.4%), likely due to the inability to diagnose cases requiring evidence beyond patient presentation (e.g., diagnostic tests). Consequently, it exhibits low costs (\$89) by avoiding diagnostic tests, yet records the highest average turns (9.44) as it gets trapped in repeated questioning loops to compensate for missing evidence. This "rambling effect" contrasts sharply with models like Claude Opus 4.7, which takes the fewest turns (5.57) but makes unsafe triage decisions prematurely. Both extremes prove that without a properly structured clinical action space, models cannot productively utilize conversational turns to converge on safe decisions.

Overall, these results highlight the value of our CMDP-based environment. Because policies trained in a constrained action space inevitably learn degenerate behaviors, modeling the full spectrum of clinical actions is critical for training clinically viable agents. Our benchmark provides this essential foundation, allowing agents to be trained and evaluated on their ability to optimize diagnostic performance within real-world safety boundaries.

![](images/e74c08b2924f6b61af268f138fc3def48c0b5a80acbdf58913ca195f9ba947dc.jpg)  
Figure 5: Case study of a high-risk patient. Left: GPAgent safely terminates with an urgent refer after detecting a rectal mass. Right: Qwen2.5 base model unsafely diagnose and treat the cancer locally instead of escalating.

<table><tr><td rowspan="2">Method</td><td colspan="3">Clinical Quality ↑</td><td colspan="2">Safety ↓</td><td>Efficiency ↓</td></tr><tr><td>Dx Acc. (%)</td><td>Tx Score (%)</td><td>Mgmt Acc. (%)</td><td>Miss Refer (%)</td><td>Miss GP (%)</td><td>Cost (USD)</td></tr><tr><td>QWEN2.5 (Base)</td><td>40.4</td><td>18.4</td><td>47.4</td><td>89.6</td><td>21.1</td><td>$445</td></tr><tr><td>GPAGENT (GRPO)</td><td>48.5 (+8.1)</td><td>20.0 (+1.6)</td><td>48.7 (+1.3)</td><td>83.6 (-6.0)</td><td>28.5 (+7.4)</td><td>438 (-7)</td></tr><tr><td>GPAGENT (C-GRPO)</td><td>52.2 (+11.8)</td><td>20.5 (+2.1)</td><td>51.0 (+3.6)</td><td>82.1 (-7.5)</td><td>26.1 (+5.0)</td><td>435 (-10)</td></tr></table>

Table 2: Comparison of different metrics of Qwen2.5 (7B) optimized using GRPO and C-GRPO.

## 5.2 Reinforcement Learning Results

We use Constrained Group Relative Policy Optimization (C-GRPO) (Girgis et al., 2026), a Lagrangian relaxation-based extension of GRPO (Guo et al., 2025), as a constrained-RL reference baseline for fine-tuning Qwen2.5 (7B). Both GRPO and C-GRPO use the same environmentdefined admissible action set (Section 4). C-GRPO optimizes Eq. 1 with primal-dual updates over standardized reward and cost advantages, incorporating safety and diagnostic-cost signals through Lagrangian multipliers. Since this relaxation optimizes a penalized surrogate rather than enforcing constraints exactly, C-GRPO does not provide formal safety guarantees. We therefore cap the safety multiplier at 6 for training stability and report posttraining safety outcomes directly, including Miss-Refer and Miss-GP (Details in Appendix C.5).

Table 2 compares the results of RL fine-tuning on Qwen2.5 (7B) (standard errors in Table 10; Appendix C.7). While both methods demonstrate increase in total return (Figure 13; Appendix C.7) and improved clinical quality over the base model, C-GRPO outperforms GRPO across all metrics, increasing Dx accuracy, Tx score and management accuracy by 11.8 pp, 2.1 pp and 3.6 pp, respectively. Furthermore, it yields the most efficient policy, requiring the fewest average turns (6.24; Appendix Table 9) and incurring the lowest diagnostic cost (\$435). Crucially, the safety metrics show both why RL helps on this task and why a constrained objective is preferable to standard scalarized RL. The base policy almost never escalates high-risk cases, missing 89.6% of required referrals; its deceptively low over-referral rate (21.1% Miss-GP) reflects near-constant local management rather than calibrated triage. When baseline GRPO is applied using a scalarized reward, it learns to refer more (reducing Miss-Refer to 83.6%) but overcompensates in the opposite direction, inflating the over-referral rate to 28.5%. Compared to baseline GRPO, C-GRPO improves both safety dimensions (Miss-Refer 82.1%, Miss-GP 26.1%). The gain is nonetheless relative, not a solution: C-GRPO still misses 82.1% of required referrals, narrowing the absolute safety gap rather than closing it. Reaching clinically acceptable safety thus remains an open challenge that our benchmark is built to measure.

<table><tr><td rowspan="2">Model</td><td colspan="3">Duplicated actions per trajectory ↓</td></tr><tr><td>ask</td><td>body_exam</td><td>test</td></tr><tr><td>GPT-5.4</td><td>0.08 (1.2%)</td><td>0.00 (0.0%)</td><td>0.00 (0.0%)</td></tr><tr><td>GEMINI 3.1 PRO</td><td>0.04 (0.7%)</td><td>0.00 (0.0%)</td><td>0.00 (0.0%)</td></tr><tr><td>CLAUDE OPUS 4.7</td><td>0.02 (0.4%)</td><td>0.00 (0.0%)</td><td>0.02 (0.4%)</td></tr><tr><td>CLAUDE SONNET 4.6</td><td>0.04 (0.6%)</td><td>0.04 (0.6%)</td><td>0.04 (0.6%)</td></tr><tr><td>QWEN2.5 (72B)</td><td>0.32 (4.7%)</td><td>0.36 (5.2%)</td><td>0.16 (2.3%)</td></tr><tr><td>QWEN2.5 (32B)</td><td>1.80 (20.3%)</td><td>0.68 (7.7%)</td><td>0.12 (1.4%)</td></tr><tr><td>QWEN2.5 (7B)</td><td>0.00 (0.0%)</td><td>0.52 (8.3%)</td><td>0.50 (8.0%)</td></tr><tr><td>HUATUO-01 (72B)</td><td>0.34 (5.0%)</td><td>0.26 (3.8%)</td><td>0.18 (2.7%)</td></tr><tr><td>DOCTORAGENT-RL (7B)</td><td>3.46 (36.7%)</td><td>0.82 (8.7%)</td><td>0.34 (3.6%)</td></tr><tr><td>GPAGENT (C-GRPO, 7B)</td><td>0.06 (1.0%)</td><td>0.48 (7.7%)</td><td>0.40 (6.4%)</td></tr></table>

Table 3: Step-level decision quality. Mean duplicated actions per trajectory, with the share of the model’s average turns in parentheses. Lower is better.

Figure 5 compares the trajectories of GPAgent-7B and the base model (Qwen2.5-7B) for a patient presenting with bloody stool. While both agents uncover the same critical evidence (an irregular rectal mass), the base model unsafely proceeds to further test, diagnose rectal cancer, and treat the patient locally. In contrast, GPAgent identifies the operational boundary of primary care and immediately refers the patient to a specialist, successfully exercising safety-informed abstention.

## 5.3 Step-Level Decision Quality

Table 3 shows that step-level quality tracks the outcome-level ranking of Table 1. Frontier models are nearly duplication-free (≤ 1.2% of average turns), whereas DoctorAgent-RL (7B) spends 36.7% of its turns re-asking known information, rising to 48.9% with duplicated exams and tests, a per-step confirmation of the rambling effect of Section 5.1. Notably, our 7B GPAgent (C-GRPO) holds duplicated questioning at the frontier level (1.0% of turns, on par with GPT-5.4) while duplicating fewer exams and tests than its Qwen2.5 (7B) base. Outcome-level metrics on GPAGENTBENCH-2K therefore correlate with intermediate decision quality rather than concealing it.

We do not evaluate exact trajectory matching because multiple evidence-acquisition paths may be clinically reasonable for the same case, and the appropriate sequence may also depend on the patient persona. Requiring agreement with a single expert trajectory would therefore penalize valid alternative clinical strategies.

## 6 Conclusion

We introduce GPAGENTBENCH-2K, a CMDPbased benchmark for clinical agents that models a full spectrum of six foundational actions and operationalizes safety-abstention as a first-class clinical outcome. Our extensive evaluation of 16 LLMs reveals a significant performance drop in a complex action space, alongside a critical quality-safety gap where frontier models with high diagnostic accuracy violate safety constraints in over half of high-risk cases.

To probe whether constrained optimization can close this gap, we benchmark C-GRPO, an existing constrained-RL method; while it improves diagnostic accuracy and reduces the safety trade-off incurred by scalarized RL, the residual safety gap remains substantial, indicating that building safe clinical agents is an open challenge that our benchmark is designed to drive progress.

## Limitations

English-only cases. Our benchmark is constructed entirely from English-language clinical records, so it does not capture the linguistic and cultural variation of multilingual primary care. Whether the observed trends transfer to other languages and healthcare systems remains an open question, and extending GPAGENTBENCH-2K to non-English settings is left to future work.

Benchmark Scope. This benchmark should not be interpreted as clinical guidelines. Its conclusions are limited by the use of a simulated patient agent and healthcare-institution-specific referral norms. In addition, to improve reproducibility and prevent hallucinated observation, we adopt action masks designed by clinical experts in an academic setting. However, the validity relative to real-world clinical practice needs further verification.

## Acknowledgement

This research was primarily supported by the ETH AI Center through an ETH AI Center doctoral fellowship to Boqi Chen and Yunke Ao, and an ETH AI Center postdoctoral fellowship to Heejin Do.

## References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. 2017. Constrained policy optimization. In International conference on machine learning, pages 22–31. Pmlr.

AFC Urgent Care. 2024. Self-pay pricing for urgent care patients without insurance. https://www.afcurgentcare.com/mentor/ resources/no-insurance-self-pay-pricing/. Accessed: 2026-05-24.

Moonshot AI. 2026. Kimi k2.6: Advancing opensource coding. Kimi Blog.

Anthropic. 2025. Introducing claude 4. Anthropic News.

David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Nassir Navab, and Matthias Keicher. 2025. Language agents for hypothesis-driven clinical decision making with reinforcement learning. arXiv preprint arXiv:2506.13474.

Centers for Medicare & Medicaid Services. 2025. Clinical laboratory fee schedule (2025 annual update). https://www.cms. gov/medicare/payment/fee-schedules/ clinical-laboratory-fee-schedule-clfs. Accessed: 2026-05-24.

Boqi Chen, Xudong Liu, Jiachuan Peng, Marianne Frey-Marti, Kyle Lam, Bang Zheng, Lin Li, and Jianing Qiu. 2026. MEDSYN: Benchmarking multi-EviDence SYNthesis in complex clinical cases for multimodal large language models. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 23631–23655, San Diego, California, United States. Association for Computational Linguistics.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. 2024. Huatuogpt-o1, towards medical complex reasoning with llms. arXiv preprint arXiv:2412.18925.

Nicholas Curcio, Kristin Wilmoth, Christian LoBue, and C Munro Cullum. 2019. Reliability of medical history reporting in older adults with and without cognitive impairment. Journal of Central Nervous System Disease, 11:1179573519843874.

DeepSeek. 2026. Deepseek news 260424. DeepSeek API Docs.

Zhihao Fan, Lai Wei, Jialong Tang, Wei Chen, Wang Siyuan, Zhongyu Wei, and Fei Huang. 2025. Ai hospital: Benchmarking large language models in a multi-agent medical interaction simulator. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10183–10213.

Yichun Feng, Jiawei Wang, Lu Zhou, Zhen Lei, and Yixue Li. 2025. Doctoragent-rl: A multiagent collaborative reinforcement learning system

for multi-turn clinical dialogue. arXiv preprint arXiv:2505.19630.

Roger Girgis, Rodrigue de Schaetzen, Luke Rowe, Azalée Robitaille, Christopher Pal, and Liam Paull. 2026. Constrained group relative policy optimization. arXiv preprint arXiv:2602.05863.

Google. 2025. Gemini 3 flash: frontier intelligence built for speed. Google The Keyword.

Google. 2026. Gemini 3.1 pro: A smarter model for your most complex tasks. Google The Keyword.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 180 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

John R Hampton, MJ Harrison, John R Mitchell, Jane S Prichard, and Carol Seymour. 1975. Relative contributions of history-taking, physical examination, and laboratory investigation to diagnosis and management of medical outpatients. Br Med J, 2(5969):486– 489.

Yixing Jiang, Kameron C Black, Gloria Geng, Danny Park, James Zou, Andrew Y Ng, and Jonathan H Chen. 2025. Medagentbench: a virtual ehr environment to benchmark medical llm agents. Nejm Ai, 2(9):AIdbp2500144.

Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik S Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, and Hae W Park. 2024. Mdagents: An adaptive collaboration of llms for medical decision-making. Advances in Neural Information Processing Systems, 37:79410–79452.

Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim, Jae Ho Sohn, Taerim Kim, Soo Kyung Kim, and Edward Choi. 2026. Patientsim: A persona-driven simulator for realistic doctor-patient interactions. Advances in Neural Information Processing Systems, 38.

Ruoqi Liu, Imran Q Mohiuddin, Austin J Schoeffler, Kavita Renduchintala, Ashwin Nayak, Prasantha L Vemu, Shivam C Vedak, Kameron C Black, John L Havlik, Isaac Ogunmola, Stephen P. Ma, Roopa Dhatt, and Jonathan H. Chen. 2026. Physicianbench: Evaluating llm agents in real-world ehr environments. arXiv preprint arXiv:2605.02240.

Meng Lu, Brandon Ho, Dennis Ren, and Xuan Wang. 2024. Triageagent: Towards better multi-agents collaborations for large language model-based clinical triage. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5747–5764.

Ananya Mantravadi, Shivali Dalmia, and Abhishek Mukherji. 2026. Art: Action-based reasoning task benchmarking for medical ai agents. arXiv preprint arXiv:2601.08988.

MiniMax. 2026. Minimax m2.7: Early echoes of selfevolution. MiniMax News.

OpenAI. 2024. Hello gpt-4o. OpenAI Index.

OpenAI. 2026. Introducing gpt-5.4. OpenAI Index.

MICHAEL C Peterson, JOHN H Holbrook, DE Von Hales, NL Smith, and LV Staker. 1992. Contributions of the history, physical examination, and laboratory investigation in making medical diagnoses. Western Journal ofMedicine, 156(2):163.

Yunhang Qian, Xiaobin Hu, Jiaquan Yu, Siyang Xin, Xiaokun Chen, Jiangning Zhang, Peng-Tao Jiang, Jiawei Liu, and Hongwei Bran Li. 2026. Medmaslab: A unified orchestration framework for benchmarking multimodal medical multi-agent systems. arXiv preprint arXiv:2603.09909.

Jianing Qiu, Kyle Lam, Guohao Li, Amish Acharya, Tien Yin Wong, Ara Darzi, Wu Yuan, and Eric J Topol. 2024. Llm-based agentic systems in medicine and healthcare. Nature Machine Intelligence, 6(12):1418–1420.

Samuel Schmidgall, Rojin Ziaei, Carl Harris, Eduardo Reis, Jeffrey Jopling, and Michael Moor. 2024. Agentclinic: a multimodal agent benchmark to evaluate ai in simulated clinical environments. arXiv preprint arXiv:2405.07960.

Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, Justin Chen, Fereshteh Mahvar, Liron Yatziv, Tiffany Chen, Bram Sterling, Stefanie Anna Baby, Susanna Maria Baby, Jeremy Lai, Samuel Schmidgall, and 62 others. 2025. Medgemma technical report. arXiv preprint arXiv:2507.05201.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. 2024. A survey on large language model based autonomous agents. Frontiers ofComputer Science, 18(6):186345.

Elisabeth Wilson, Alice Hm Chen, Kevin Grumbach, Frances Wang, and Alicia Fernandez. 2005. Effects of limited english proficiency and physician language on health care comprehension. Journal of general internal medicine, 20(9):800–806.

Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng, Zihan Dong, Xian Wu, Xiangru Tang, Hongtu Zhu, Yun Li, Linjun Zhang, Shujie Liu, Yan Lu, and Huaxiu Yao. 2026. Mmedagent-rl: Optimizing multi-agent collaboration for multimodal medical reasoning. volume 2026, pages 81513–81546.

Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, and Yu Rong. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning.

Weixiang Yan, Haitian Liu, Tengxiao Wu, Qian Chen, Wen Wang, Haoyuan Chai, and Jiayi Wang. 2026. Clinicallab: Aligning agents for multi-departmental clinical diagnostics in the real world. Advances in Neural Information Processing Systems, 38.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yutian Zhao, Huimin Wang, Yefeng Zheng, and Xian Wu. 2025. A layered debating multi-agent system for similar disease diagnosis. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 539–549.

Yucheng Zhou, Lingran Song, and Jianbing Shen. 2025. Mam: Modular multi-agent framework for multimodal medical diagnosis via role-specialized collaboration. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 25319–25333.

Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, and Lequan Yu. 2025. Medagentboard: Benchmarking multi-agent collaboration with conventional methods for diverse medical tasks. arXiv preprint arXiv:2505.12371.

## A Data Collection and Processing

## A.1 Examples of Raw Clinical Records

Figure 6 illustrates a typical clinical note from a routine outpatient encounter. The record details a patient presenting with a chronic tension-type headache, where the physical examination is normal and the assessment leads to conservative management. Crucially, this encounter concludes with the patient being discharged with routine follow-up, requiring no specialist referral.

In contrast, Figure 7 demonstrates a clinical record for a more severe, urgent case. This record documents a patient presenting with gross hematuria and worsening urinary symptoms. Unlike the routine discharge seen in the previous example, the severity of the clinical findings here necessitates an urgent same-day referral to the Emergency Department and a Urology specialist, culminating in the patient’s hospital admission.

## A.2 Prompt for Data Processing

As described in the main text, we employ Gemini-3-Flash to systematically extract and structure clinical facets from raw clinical texts. To ensure high fidelity and mitigate hallucinations, we apply a concise prompt template that maps specific sections of our clinical record format (e.g., Most Recent Vitals, Management / Disposition) directly to a predefined JSON schema.

The data processing pipeline utilizes a two-part prompt: a System Prompt that defines the persona and general constraints, and a User Prompt that provides the target schema, extraction rules, and the raw clinical text. The placeholders [INSERT CASE ID] and [INSERT RAW CLINICAL TEXT] are dynamically populated during execution.

## System Prompt

You are a clinical data structuring   
assistant.   
Your task is to transform a raw clinical   
record into a standardized JSON object.   
Key Constraints:   
1. Extract information ONLY from the   
provided source clinical texts.   
2. Do not invent or infer unsupported facts,   
findings, or results.   
3. If a field is missing, uncertain, or not   
stated, use null (or empty arrays for   
lists).   
4. Use clear, concise, professional medical   
English.   
5. Output valid JSON only, with no markdown   
formatting or comments.

## User Prompt

Transform the following raw clinical record   
into the target JSON schema.   
SOURCE CASE:   
"case\_id": "[INSERT CASE ID]",   
"clinical\_record": "[INSERT RAW CLINICAL   
TEXT]"   
}   
TARGET SCHEMA:   
"id": string,   
"age": integer | null,   
"sex": string | null,

```csv
"diagnosis": null,
"disposition": string ,
"vital_signs": {
"temperature": string ,
"blood_pressure": string ,
"heart_rate": string ,
"respiratory_rate": string ,
"oxygen_saturation": string
},
"physical_exam": {
"general_appearance": string ,
"cardiovascular": string ,
"pulmonary": string ,
"abdominal": string ,
"neurological": string ,
"extremities": string
},
"test_results": {
"ECG": string ,
"troponin": string ,
"CBC": string ,
"BMP": string ,
"CT": string ,
"X-ray": string ,
"MRI": string ,
"Ultrasound": string ,
"others": string
},
"referral_reference": {
"referral_needed": boolean,
"urgency": string ,
"destination": string
},
"treatment_reference": {
"medications": [string],
"non_pharmacologic": [string],
"follow_up": string
}
}
EXTRACTION RULES:
1. id, age, sex, & diagnosis:
- "id" must be exactly "[INSERT CASE ID]".
"age": Extract as an integer from the
header (e.g., 28). If N/A or missing,
use null.
"sex": Extract from the header (e.g., "
Male", "Female"). If N/A or missing, use
null.
"diagnosis" will be populated externally;
always leave as null in the output.
2. disposition:
- Extract directly from the "Disposition"
header or "Management / Disposition"
section.
3. vital_signs, physical_exam, test_results:
- Map the "Most Recent Vitals" table to
vital_signs.
Map the "Physical Examination" text to the
corresponding physical_exam body
systems.
Map the "Test Results" table to
test_results.
Do not infer normal findings if not
explicitly mentioned. Use null.
```

![](images/0a1c3f7e7dc8cc437952e8eaefd8e957c34417d804ff193d249ba65a4fa4ff18.jpg)  
Figure 6: An example of a de-identified clinical record from a routine outpatient encounter.

<table><tr><td>x</td><td>□</td><td>Summary Clinical Note</td><td></td><td>Test Results</td><td>Medications</td><td colspan="3">Disposition</td><td>↓ Export</td><td>Read-only</td></tr><tr><td colspan="7">De-identified Record Encounter ID: ENC-XXXX MRN: PHI-REDACTED</td><td>Allergies: N/A Disposition: Admitted</td><td colspan="3">Encounter Diagnosis: Urinary Tract Infection with Hematuria</td></tr><tr><td>Summary</td><td>Clinical Note</td><td></td><td>Test Results</td><td>Medications</td><td colspan="2">Disposition</td><td colspan="3"></td><td></td></tr><tr><td colspan="3">Visit Summary Clinical Note Chief Complaint: Worsening urinary symptoms</td><td colspan="7"></td></tr><tr><td colspan="3" rowspan="2">with lower abdominal pain and gross hematuria. Diagnosis: Urinary Tract Infection with</td><td colspan="7">Chief Complaint</td></tr><tr><td colspan="3">Worsening urinary symptoms with lower abdominal pain and gross hematuria. Medical History</td></tr><tr><td colspan="3">Hematuria Encounter Type: Urgent assessment Service: General practice</td><td colspan="7">Female patient presents with 3 days of worsening urinary symptoms, including dysuria, lower abdominal pain, and</td></tr><tr><td colspan="3">Referral: Emergency Department / Urology assessment, urgent</td><td colspan="7">gross hematuria. Symptoms have worsened over the illness course. Current use of an oral anti-inflammatory medication is reported, but details are unavailable in the source record. Additional past medical history, allergies, code status, and advance directive information are not available in the source</td></tr><tr><td colspan="3">Disposition: Admitted record.</td><td colspan="7"></td></tr><tr><td colspan="3">Active Problems</td><td></td><td colspan="7">Physical Examination Normal female appearance, cooperative. Abdomen with equivocal lower abdominal tenderness and dullness to</td></tr><tr><td colspan="3">Problem</td><td>Status</td><td colspan="7">percussion. No bladder distension. No tenderness over renal or ureteral areas.</td></tr><tr><td colspan="3">UTI with Hematuria Gross hematuria</td><td>Active Active</td><td colspan="7">Assessment</td></tr><tr><td colspan="3">Lower abdominal pain</td><td>Active</td><td colspan="7">Urinary tract infection with hematuria. Findings are supported by worsening dysuria, lower abdominal pain, gross hematuria, leukocytes 2+ on urinalysis, microscopic urine WBC 136/uL, and peripheral WBC 10.67 × 10^9/L. Vitals are stable. Worsening symptoms with gross hematuria and admitted disposition support urgent hospital assessment</td></tr><tr><td colspan="10">Current Medications rather than outpatient-only management.</td></tr><tr><td>Medication</td><td colspan="3">Details</td><td colspan="3">Management / Disposition</td><td colspan="3">Arrange urgent same-day referral to the Emergency Department for hospital assessment, with urology review as</td></tr><tr><td>Oral anti- inflammatory</td><td colspan="3">Patient-reported; details unavailable</td><td colspan="3"></td><td colspan="2">indicated given gross hematuria and worsening urinary symptoms. Provide supportive care while awaiting transfer, including oral hydration if tolerated, rest, and avoidance of bladder irritants. Review the reported oral anti-</td></tr><tr><td colspan="3">Most Recent Vitals</td><td colspan="3"></td><td colspan="3">inflammatory medication before adding analgesics or anti-inflammatory treatment. Definitive antibiotic selection is deferred to hospital assessment according to local protocol. Monitor for fever, rigors, flank pain, inability to pass urine, increasing gross hematuria or clots, vomiting, worsening pain, or systemic</td></tr><tr><td colspan="3">Measurement</td><td colspan="3">Value</td><td colspan="3">deterioration. Patient admitted for further assessment and management.</td></tr><tr><td colspan="3">Temperature</td><td colspan="3">36.2℃ 110/80 mmHg</td><td colspan="4"></td></tr><tr><td colspan="3">Blood Pressure</td><td colspan="3"></td><td colspan="4"></td></tr><tr><td>Heart Rate</td><td colspan="3"></td><td colspan="3">76 bpm 19 bpm</td><td colspan="4"></td></tr><tr><td colspan="11">Respiratory Rate</td></tr><tr><td colspan="11">Test Results</td></tr><tr><td colspan="2"></td><td>Result</td><td colspan="3"></td><td colspan="4"></td></tr><tr><td>Test</td><td></td><td colspan="3"></td><td colspan="4">Status</td></tr><tr><td>CBC</td><td></td><td colspan="3">WBC 10.67 × 10^9/L</td><td colspan="3">Final Final</td></tr><tr><td>Urinalysis</td><td>WBC 136/uL</td><td>Leukocytes 2+; microscopic</td><td colspan="3"></td><td colspan="3"></td></tr></table>

Chart view: Summary | De-identified record | PHI removed | Read-only clinical display  
Figure 7: An example of a de-identified clinical record for an urgent encounter requiring hospital admission and referral to specialists.

4. referral\_reference:   
Check the "Referral" field in the Visit   
Summary or text in "Management /   
Disposition".   
"referral\_needed": true if a referral/   
transfer is requested, otherwise false.   
"urgency": e.g., "none", "routine",   
urgent", "emergent".   
"destination": e.g., "Emergency Department   
", "Urology", "GP", or null.   
5. treatment\_reference:   
"medications": List items from "Current   
Medications" or drugs prescribed in the   
Management section.   
"non\_pharmacologic": List supportive care,   
hydration, rest, or lifestyle advice   
from Management.   
"follow\_up": Specific follow\_up   
instructions or monitoring parameters   
from Management.   
Now output the JSON only.

## A.3 Examples of Processed Data

Following the extraction process described in Appendix A.2, the raw clinical texts shown above are transformed into a highly structured format. Below are the final JSON outputs generated by the model for the two respective encounters, demonstrating how unstructured clinical narratives are mapped to our predefined schema while filtering out explicit diagnostic labels.

Processed JSON Output corresponding to   
Figure 6   
{   
"id": "ENC-XXXX",   
"age": 28,   
"sex": "Male",   
"diagnosis": null,   
"disposition": "Discharged",   
"vital\_signs": {   
"temperature": "36.4°C",   
"blood\_pressure": "130/80 mmHg",   
"heart\_rate": "78 bpm",   
"respiratory\_rate": "20 bpm",   
"oxygen\_saturation": null   
},   
"physical\_exam": {   
"general\_appearance": "Alert, fair   
spirit, fluent speech, cooperative.",   
"cardiovascular": "Heart rate 83 bpm,   
regular rhythm, no pathological murmurs   
"pulmonary": "Clear breath sounds   
bilaterally, no dry or moist rales.",   
"abdominal": "Soft, no tenderness or   
rebound tenderness.",   
"neurological": "Pupils equal and   
reactive (3mm), EOM intact, symmetric

facial features, tongue midline, no   
dysphagia or coughing when drinking,   
hearing normal, neck turning and   
shrugging strong, muscle strength Grade   
V, normal muscle tone, physiological   
reflexes present, no pathological   
reflexes, meningeal signs negative,   
sensation normal.",   
"extremities": "No edema in lower limbs   
},   
"test\_results": {   
"ECG": null,   
"troponin": null,   
"CBC": null,   
"BMP": null,   
"CT": "Head CTA: No abnormal lesions in   
brain parenchyma.",   
"X-ray": null,   
"MRI": "Brain MRI performed.",   
"Ultrasound": null,   
"others": null   
},   
"referral\_reference": {   
"referral\_needed": false,   
"urgency": "none",   
"destination": "GP"   
},   
"treatment\_reference": {   
"medications": [],   
"non\_pharmacologic": [   
"Rest during attacks.",   
"Stress management and sleep hygiene.",   
"Reduce fatigue triggers."   
],   
"follow\_up": "Routine GP follow-up if   
symptoms persist or worsen over the next   
few weeks."   
}   
}

Processed JSON Output corresponding to   
Figure 7   
"id": "ENC-XXXX",   
"age": null,   
"sex": "Female",   
"diagnosis": null,   
"disposition": "Admitted",   
"vital\_signs": {   
"temperature": "36.2°C",   
"blood\_pressure": "110/80 mmHg",   
"heart\_rate": "76 bpm",   
"respiratory\_rate": "19 bpm",   
"oxygen\_saturation": null   
},   
"physical\_exam": {   
"general\_appearance": "Normal female   
appearance, cooperative.",   
"cardiovascular": null,   
"pulmonary": null,   
"abdominal": "Lower abdominal tenderness   
(equivocal), dullness to percussion, no   
bladder distension, no tenderness over

```csv
renal or ureteral areas.",
"neurological": null,
"extremities": null
},
"test_results": {
"ECG": null,
"troponin": null,
"CBC": "White blood cell count 10.67 x
10^9/L",
"BMP": null,
"CT": null,
"X-ray": null,
"MRI": null,
"Ultrasound": null,
"others": "Urinalysis: Leukocytes 2+,
microscopic white blood cell count 136/
uL"
},
"referral_reference": {
"referral_needed": true,
"urgency": "urgent",
"destination": "Emergency Department"
},
"treatment_reference": {
"medications": [],
"non_pharmacologic": [
"Encourage oral fluids if able.",
"Advise rest and avoid bladder
irritants until reviewed."
],
"follow_up": "Same-day specialist or
hospital assessment is appropriate
rather than routine GP follow-up."
}
}
```

## A.4 Statistics of the Dataset

Patient Demographics. Figure 8 details the demographic characteristics of our dataset. As shown in Figure 8a, male patients account for 57.6% and females for 42.4% of the 2,256 cases with recorded gender data. Figure 8b presents the age distribution (excluding 161 cases with unspecified ages). The mean and median age of the patient cohort is 50.9 and 52.0 years, respectively.

Medical Specialties. Figure 9 categorizes the dataset based on the required level of care. As shown in Figure 9, approximately half of the dataset (45.1%; 1,097 cases) falls under general (internal) medicine, representing cases manageable by general practitioners. The remaining half (54.9%; 1,337 cases) comprises complex conditions that require specialist referral. Figure 10 provides a detailed breakdown of cases requiring specialist referral, demonstrating a long-tail characteristic that spans across 29 distinct clinical specialties. Among these, Respiratory medicine (173 cases), Trauma and orthopaedic surgery (159 cases), and Oncology (130 cases) represent the most frequent referral targets. Conversely, the tail end of the distribution includes highly specialized fields such as Clinical genetics, Plastic surgery, and Anaesthetics, which contain only a single case each. Beyond this wide array of specialties, our analysis identifies 756 unique disease categories across the dataset after standardizing semantic variations in the raw clinical text.

![](images/f68816e998f6b4456979a89999cf14f9efa4cbd6ac6393552f23030574887d80.jpg)

![](images/48ea49750fe3cca5125c90c5305febd42c7b869bc9112743e3079972e0930286.jpg)  
(b)  
Figure 8: Patient demographics of the dataset. (a) Gender distribution of the patient cohort. (b) Age distribution of the patients, grouped into 20-year intervals.

Patient Presentations and Symptoms. Patient presentations in our dataset are highly detailed, featuring an average of 4.9 symptoms per case, whereas their initial chief complaints contain an average of only 2.2 symptoms. This discrepancy suggests that patients rarely articulate their complete clinical picture upfront. Rather, physicians must iteratively elicit additional symptoms as the consultation unfolds. Overall, these symptoms feature 1,050 unique descriptions and span a wide array of body systems, including respiratory, gastrointestinal, neurological, cardiovascular, and musculoskeletal systems. Figure 11 illustrates the frequency distribution of the top 20 most prevalent symptoms, among which cough is the most frequently reported symptom (170 occurrences), followed by dizziness (135), nausea (115), and fever (110). Notably, the distribution is heavily dominated by gastrointestinal issues (e.g., nausea, poor appetite, vomiting, diarrhea) and general systemic complaints (e.g., fever, poor sleep, fatigue).

![](images/9a347545ffa1521e8b6cec932dfad0392d339c9d4aa7db2c0a9258a560841ba4.jpg)  
Figure 9: Categorization of cases by required level of care. The dataset is split between cases manageable by general practitioners (General, 45.1%) and complex ones requiring referral to specialists (Specialized, 54.9%). This distribution supports the evaluation of both diagnosis-and-management and safety-informed abstain-and-refer decisions.

Physical Examinations and Diagnostic Tests. Following the initial presentation, cases involve rigorous clinical evaluations. Patients undergo an average of 4.1 physical examinations, which are systematically distributed across 6 core assessment areas: general appearance, pulmonary, abdominal, extremities, neurological, and cardiovascular. Furthermore, an average of 2.9 diagnostic tests are ordered per patient, drawing from a vast pool of 584 unique diagnostic items. Figure 12 illustrates the distribution and frequency of these diagnostic evaluations. As shown in Figure 12a, the tests are broadly distributed across four categories: blood tests account for the majority at 43.3%, followed by imaging tests at 36.2%, pathological tests at 10.3%, and functional tests at 10.2%. To provide a more granular view, Figure 12b details the top 20 most frequently ordered individual tests, highlighting the important role of common imaging modalities (e.g., CT, Ultrasound, X-ray) and routine blood work (e.g., Complete Blood Count, Basic Metabolic Panel) in clinical workflow.

## B Extended Related Work

As discussed in Section 4, our proposed environment necessitates individual agents to autonomously navigate a complex action space that models the recursive and branching nature of realworld clinical workflows. To substantiate our claim that existing clinical agent frameworks often circumvent this complexity by partitioning the workflow into highly specialized roles, we provide a detailed comparative analysis of individual agent action space across recent clinical agent frameworks.

In several frameworks, the action space of individual agents is reduced to a single-step classification or static text generation task, completely omitting the recursive evidence-gathering phase. For instance, TriageAgent (Lu et al., 2024) utilizes a multi-agent group chat to determine the Emergency Severity Index based on static clinical notes. The action space for each agent is strictly limited to outputting a severity level and a confidence score, effectively reducing the agent to a simple classifier. Similarly, Layered Debating (Zhao et al., 2025) is designed for similar disease diagnosis, where agents participate in inner- and inter-group debates. Because it operates in a case-first scenario where comprehensive patient data is provided upfront, the agent’s action space is limited to proposing textual arguments for a pre-assigned disease. This setup entirely bypasses the need for the agent to autonomously select information-gathering actions to actively elicit clinical evidence.

Several frameworks attempt to model the clinical workflow by partitioning the decision-making process across multiple agents, effectively dividing the fundamental trade-off of a constrained Markov decision process (CMDP) across separate policies. This prevents individual agents from autonomously balancing diagnostic accuracy against safety and resource constraints within a full-spectrum clinical action space. For instance, MAM (Zhou et al., 2025) decomposes multimodal diagnosis into five distinct roles. Operating on static, comprehensive patient data, the general practitioner (GP) agent is restricted solely to initial disease classification and routing, while specialist agents merely generate diagnostic opinions. By acting strictly as a router, the GP avoids the critical branching decision between local management and safetyinformed abstention required in real-world practice. Likewise, MMedAgent-RL (Xia et al., 2026) employs a triage-and-referral pipeline where a triage agent assigns a specialty and an attending physician integrates the resulting opinions. The triage agent’s action space is reduced to simply outputting a department name, completely bypassing both the iterative evidence-gathering loop and the highstakes management decisions inherent to primary care. Furthermore, LA-CDM (Bani-Harouni et al., 2025) attempts to replicate hypothesis-driven clinical decision-making by splitting the cognitive load between two distinct agents. While its decision agent can choose between recursively requesting a test or outputting a final diagnosis, the framework does not actively model interactions between the agent and the patient (e.g., symptom elicitation via dialogue). Moreover, despite attempting to minimize the diagnostic cost as a training objective, LA-CDM incorporates it as a soft, additive penalty rather than formalizing a rigorous CMDP. This scalarized approach fails to enforce strict resource constraint, allowing the policy to freely trade operational costs for diagnostic accuracy, or vice versa. Finally, in their evaluation of lay summary generation and workflow automation, MedAgentBoard (Zhu et al., 2025) highlights frameworks that deploy up to nine specialized agents in a sequential pipeline, reducing each agent’s action space to a microscopic textual transformation or coding task. More recently, MedMASLab (Qian et al., 2026) provides a unified orchestration framework for benchmarking multimodal medical multiagent systems, standardizing agent communication, multimodal inputs, and cross-specialty evaluation. This line of work is complementary to ours: it addresses how to orchestrate and compare multi-agent medical systems, whereas our focus is whether a single clinical agent can autonomously operate over a unified action space that includes evidence gathering, diagnosis, treatment, and referral. Thus, MedMASLab (Qian et al., 2026) further supports our motivation that existing clinical-agent research often emphasizes system-level orchestration, while leaving the single-agent CMDP-style decision problem underexplored.

![](images/bbe14b569d4773ab2a95b93146fc5ace7b0c4eaf04355b966fa5fdc4aa234f3d.jpg)

Figure 10: Distribution of medical specialties across the dataset, excluding General (internal) medicine. The chart highlights the destinations for abstain-and-refer decisions, demonstrating a distinct long-tail distribution. Respiratory medicine, Trauma and orthopaedic surgery, and Oncology are the most frequent specialized fields, followed by a steep drop-off into highly niche specialties.  
![](images/6923a1ea30c252ff7fecb0d13efcb97167f3d52d7a255c998050f0212bfca299.jpg)  
Figure 11: Frequency distribution of the top 20 most common symptoms across the dataset. Symptoms are colorcoded by the corresponding clinical categories.

![](images/d8abe09027ed976616952c01afa7d0570358bd177e21f219c327f647ec0fbe6f.jpg)

(a)  
![](images/e36b1cc2256a15632d28ff3a4a22a8ddb3ef9e9cefaf7127338ff73fa9806c4c.jpg)  
(b)  
Figure 12: Overview of diagnostic test distributions. (a) A pie chart categorizing diagnostic test items into four groups; (b) A bar chart detailing the top 20 most frequent diagnostic tests ordered across the dataset.

A few frameworks model an interactive environment but still oversimplify the action space of individual agents. Specifically, Agent-Clinic (Schmidgall et al., 2024) is one of the closest prior efforts because it introduces a multimodal simulated clinical environment with patient interaction, incomplete information, tool use, multiple specialties, and multilingual scenarios. However, its main focus is evaluating diagnostic behavior and tool use in simulated clinical encounters. It does not explicitly formulate the primary-care decision process as a constrained action-space problem in which the agent must jointly decide when to ask, examine, test, diagnose, treat, or refer under safety and cost constraints. Therefore, while AgentClinic moves beyond static QA, it does not fully capture the CMDP-style trade-off that our environment is designed to evaluate. DoctorAgent-RL (Feng et al., 2025) simulates multi-turn clinical dialogue between a doctor agent and a patient agent. However, the action space of the doctor agent is collapsed into only a small set of actions: ask and diagnose, omitting physical exams, diagnostic tests, and management (i.e., treat or refer). AI Hospital (Fan et al., 2025) simulates a slightly more complex interactive dialogue, allowing the doctor agent to also order tests. However, it fails to explicitly model the high-stakes branching decision of safety-informed abstention (i.e., the refer action). Furthermore, it functions purely as an evaluation sandbox, simply sampling rollouts of the policies of different off-the-shelf large language models and collecting rewards as evaluation metrics. Another closely related direction is Electronic Heath Record (EHR)-based medical-agent benchmarking. MedAgentBench (Jiang et al., 2025) introduces a realistic virtual EHR environment with physician-written tasks, patient-specific records, and FHIR-compliant interactions, providing an important testbed for evaluating medical LLM agents in record-centered workflows. ART (Mantravadi et al., 2026) similarly targets action-based clinical reasoning over EHRs, emphasizing retrieval, temporal aggregation, threshold evaluation, and conditional logic. PhysicianBench (Liu et al., 2026) further evaluates long-horizon physician tasks in realistic EHR environments, highlighting the need to assess complex workflows rather than isolated medical question-answering. These benchmarks are highly relevant because they move toward interactive and execution-grounded clinical tasks. However, they primarily focus on EHR-centered actions such as information retrieval, record navigation, order placement, or task completion within an existing health-record system. In contrast, our environment focuses on primary-care decision-making under uncertainty, where the agent must recursively gather evidence from the patient, choose physical exams and tests, and make terminal management decisions between treatment and referral. This distinction makes the CMDP structure explicit and allows us to evaluate how agents balance diagnostic accuracy, safety, and resource use within a single unified action space.

## C Experiment

## C.1 Experiment Setting

For zero-shot benchmarking, every off-the-shelf model receives the same system prompt at each turn, specifying the GP role, the six-action schema, the clinical workflow guidelines, and the statedependent admissible action set. The prompt is reconstructed each turn with the current round counter, the case-specific physical examination and test targets, the consultation transcript so far, and the current workflow state. Case-specific fields are populated dynamically and shown below as bracketed placeholders (e.g., [chief complaint], [consultation transcript]); the example below shows the prompt in the initial collecting\_evidence state.

## Zero-Shot Six-Action Evaluation Prompt

Current workflow state: collecting\_evidence

Allowed actions for this turn:   
- ask   
- body\_exam   
- test   
- diagnosis   
- refer   
Masked actions for this turn:   
- treatment   
You MUST choose exactly one allowed action.   
Do not choose a masked action. Use the   
exact lowercase action name in <action   
type="...">.   
Stopping rule: do not keep collecting   
evidence indefinitely. Once you have   
enough evidence, choose diagnosis before   
treatment, or choose refer for unsafe/   
out-of-scope cases.   
Now produce exactly one valid response.   
Return only the <think> block and one <   
action type="..."> block; do not add   
explanations outside those tags.

## C.2 Reward, Cost, and Evaluation Metrics

In this section, we provide detailed definitions of the reward and cost terms used in our environment. For each case, the agent interacts with the environment and produces a terminal action: either treat with a predicted diagnosis and a treatment plan, or refer. Each case includes a ground-truth (GT) label as either safely manageable in primary care or requiring specialist referral, corresponding to the aforementioned terminal actions, respectively. For zero-shot benchmarking of off-the-shelf LLMs, we sample policy rollouts in the environment and collect the corresponding reward and cost terms as our evaluation metrics. We report the metrics averaged over all test cases.

Diagnosis Accuracy Reward. The diagnosis (Dx) accuracy reward measures whether the agent’s predicted diagnosis is clinically equivalent to the GT diagnosis. We use GPT-5.4 as an automated judge to assign a binary score, where 1 indicates that the predicted and GT diagnoses are clinically equivalent and 0 otherwise. Clinical equivalence allows synonymous wording and minor specificity differences, but penalizes overly broad or vague diagnoses. For reinforcement learning (RL)-based optimization, we scale the score by a weight $w _ { \mathrm { d i a g } }$ to form the diagnosis reward. For zero-shot benchmarking, we report the raw score.

The detailed prompt is shown below. The placeholders [INSERT PREDICTED DIAGNOSIS] and [INSERT GROUND-TRUTH DIAGNOSIS] are dynamically populated during execution.

Diagnostic Accuracy Evaluation Prompt   
You are a clinical expert. Determine whether   
the predicted diagnosis is clinically   
equivalent to the ground-truth diagnosis.   
Predicted diagnosis: "[INSERT PREDICTED   
DIAGNOSIS]"   
Ground-truth diagnosis: "[INSERT GROUND-  
TRUTH DIAGNOSIS]"   
Rules:   
1. Clinical equivalence: Count as correct if   
both diagnoses refer to the same   
underlying condition, even with   
different wording, e.g., "URI" and "   
upper respiratory infection".   
2. Partial match: Count as correct only if   
the core condition is the same, e.g., 11   
pneumonia" and "bacterial pneumonia".   
3. Overly broad diagnosis: Count as   
incorrect if the prediction is vague or   
nonspecific, e.g., "infection" for "   
urinary tract infection" or "abdominal   
pain" for "appendicitis".   
Answer with exactly one value:   
1 if clinically equivalent;   
0 otherwise.

Treatment Score Reward. The treatment (Tx) score reward measures whether the agent’s predicted treatment plan is clinically aligned with the GT treatment plan. We use GPT-5.4 as an automated judge to compare the predicted and GT treatments and assign an episode-level score in {0, 0.5, 1}, where 1 indicates a clinically equivalent treatment plan, 0.5 indicates a partially aligned treatment plan, and 0 indicates an incorrect or unrelated treatment plan. For RL-based optimization, we scale the score by a weight $w _ { \mathrm { t r e a t } }$ to form the treatment reward. For zero-shot benchmarking, we report the raw score.

The detailed prompt is shown below. The placeholders [INSERT PREDICTED TREATMENT] and [INSERT GROUND-TRUTH TREATMENT] are dynamically populated during execution.

Treatment Score Evaluation Prompt   
You are a clinical expert. Determine whether   
the predicted treatment plan is   
clinically aligned with the ground-truth   
treatment plan.

Predicted treatment: "[INSERT PREDICTED   
TREATMENT]"   
Ground-truth treatment: "[INSERT GROUND-  
TRUTH TREATMENT]"   
Rules:   
1. Correct treatment: Assign 1 if the   
predicted treatment covers the key   
therapeutic components of the ground  
truth treatment, even if worded   
differently.   
2. Partially correct treatment: Assign 0.5   
if the predicted treatment includes some   
clinically relevant components but   
misses important elements, is too   
nonspecific, or only partially matches   
the ground-truth treatment.   
3. Incorrect treatment: Assign 0 if the   
predicted treatment is unrelated,   
clinically inappropriate, unsafe, or   
fails to match the main treatment   
strategy in the ground truth.   
Answer with exactly one value:   
1 for correct;   
0.5 for partially correct;   
0 for incorrect.

Management Accuracy. Management accuracy measures whether the agent chooses the correct terminal action: it is 1 if the agent terminates with diagnose for a case that can be safely managed in primary care, or with refer for a case requiring specialist escalation, and 0 otherwise. We report it as an evaluation metric; during RL, terminal-action correctness is driven by the safety cost $C _ { \mathrm { s a f e t y } }$ rather than a separate reward term.

Safety Cost. The safety cost is defined as the binary indicator of an incorrect terminal action, $C _ { \mathrm { s a f e t y } } = \mathbb { 1 } _ { \mathrm { m g m t \mathrm { - } w r o n g } } \in \{ 0 , 1 \}$ , where 1<sub>mgmt\_wrong</sub> = 1 if the terminal action is inconsistent with the GT label and 0 otherwise. Safety is enforced as a CMDP constraint through Lagrangian dual ascent (Appendix C.5). Because missed escalation is high-stakes and asymmetric, its dual variable rises quickly under violation; to prevent it from dominating optimization and destabilizing training, we cap the safety multiplier at $\lambda _ { \mathrm { s a f e t y } } \le p _ { \mathrm { m g m t } } = 6$ . This does not constitute a hard guarantee that the constraint is satisfied, so we report the resulting safety levels directly. The safety cost captures two types of management errors: (1) Missed Referral (Miss Refer): Failing to refer a case that requires specialist escalation (GT requires referral, but agent terminates with treat). This corresponds to the false negative rate when referral is treated as the positive class; and (2) Missed Local Management (Miss GP): Unnecessary escalation (GT is safely manageable locally, but agent terminates with refer), corresponding to the false positive rate.

Diagnostic Test Cost. Unlike safety, diagnostic efficiency is modeled as a thresholded constraint $( \epsilon _ { \mathrm { c o s t } } = \tau _ { \mathrm { c o s t } } )$ , which measures the monetary cost of diagnostic tests ordered by the agent per episode. Physical exams and vital sign checks incur no cost, while laboratory and imaging tests incur realistic monetary penalties based on a USD-grounded lookup table (Table 4) compiled from public 2024– 2025 US Medicare and self-pay fee schedules (Centers for Medicare & Medicaid Services, 2025; AFC Urgent Care, 2024), used as a standardized reporting proxy rather than exact billing claims. The raw monetary cost is normalized such that a value of 0.2 corresponds to \$100 USD per consultation. During RL training, this cost is penalized adaptively via the dual variable $\lambda _ { \mathrm { c o s t } }$ (initialized as 0) to satisfy the constraint that the average test cost remains below the predefined threshold $\tau _ { \mathrm { c o s t } } .$ For evaluation, we report the raw diagnostic costs incurred in each episode.

## C.3 Automatic Judge Validation

The GPT-5.4 judge is used only for the semantic Dx and Tx scores defined above. Management accuracy, Miss-Refer, and Miss-GP are computed deterministically from the agent’s terminal action against the clinician-validated ground-truth management label, and the referral destination is checked by string match.

Physician audit. We randomly sampled 200 outputs from diagnose-and-manage cases across the 16 evaluated models. Two senior physicians (Appendix D.7) independently scored the predicted diagnosis and treatment plan against the groundtruth labels using the same rubric given to the judge. The averaged Cohen’s κ between GPT-5.4 judge and physicians is 0.92 for Dx accuracy and 0.89 for Tx score.

Cross-judge robustness. As GPT-5.4 is both a judge and an evaluated model, to mitigate circular evaluation risk, we re-scored three frontier models using Claude Opus 4.7 with identical prompts and rubric. Table 5 demonstrates that all scores change within the standard errors in Table 1, and model rankings remain the same. Notably, neither judge favors its own model family. For instance, GPT-5.4 has lowest Dx accuracy among three frontier models when used as the automatic judge.

<table><tr><td>Diagnostic Category</td><td>Aliases</td><td>Cost (USD)</td></tr><tr><td>Rapid Tests</td><td>strep test, flu test, pregnancy test, rapid test, h. pylori</td><td>$25</td></tr><tr><td>Complete Blood Count</td><td>cbc, complete blood count, blood count, blood test</td><td>$30</td></tr><tr><td>Blood Glucose</td><td>fingerstick glucose, blood glucose, hba1c</td><td>$30</td></tr><tr><td>Basic Metabolic Panel</td><td>bmp, basic metabolic panel, metabolic panel</td><td>$35</td></tr><tr><td>Urinalysis</td><td>urinalysis, urine, urine dipstick</td><td>$35</td></tr><tr><td>Cardiac Biomarker</td><td>troponin, cardiac biomarker</td><td>$40</td></tr><tr><td>Electrocardiogram</td><td>ecg, ekg, electrocardiogram</td><td>$50</td></tr><tr><td>Thyroid Function</td><td>tsh, thyroid, thyroid function</td><td>$50</td></tr><tr><td>X-ray</td><td>x-ray, xray, chest x-ray, radiograph</td><td>$80</td></tr><tr><td>Spirometry</td><td>spirometry, pulmonary function</td><td>$100</td></tr><tr><td>Ultrasound</td><td>ultrasound, us, sonograph</td><td>$250</td></tr><tr><td>Computed Tomography</td><td>ct, ct scan, computed tomography</td><td>$500</td></tr><tr><td>Magnetic Resonance</td><td>mri, magnetic resonance</td><td>$700</td></tr></table>

Table 4: Monetary costs of diagnostic tests used in the environment to calculate efficiency penalties.

<table><tr><td rowspan="2">Model</td><td colspan="2">Dx Acc. (%) ↑</td><td colspan="2">Tx Score (%) ↑</td></tr><tr><td>GPT-5.4 judge</td><td>Claude judge</td><td>GPT-5.4 judge</td><td>Claude judge</td></tr><tr><td>GPT-5.4</td><td>64.1</td><td>63.5</td><td>53.2</td><td>52.5</td></tr><tr><td>GEMINI 3.1 PRO</td><td>66.8</td><td>67.2</td><td>41.6</td><td>42.1</td></tr><tr><td>CLAUDE OPUS 4.7</td><td>70.0</td><td>69.4</td><td>46.0</td><td>45.6</td></tr></table>

Table 5: Cross-judge robustness. Dx accuracy and Tx score for the frontier models under the default GPT-5.4 judge and under an independent Claude Opus 4.7 judge using an identical rubric. All differences are below 1 pp and rankings are preserved.

## C.4 Patient Simulator Fidelity Audit

While the clinical records are real GP encounters, the multi-turn interaction is generated using the persona-driven patient simulator (details in Section 4). To validate the patient simulator fidelity, we ask two senior physicians to audit the simulated interactions.

Protocol. We sampled 32 complete interaction trajectories from the main evaluation runs, two from each of the 16 evaluated models. Each model’s pair is sampled to contrast persona settings, i.e., one standard persona and one challenging persona (low anamnestic fidelity and/or disorientation), so that all four persona dimensions are represented. Two senior physicians (Appendix D.7) reviewed each trajectory alongside its source GP record and persona configuration, and were asked to make independent binary judgments (i.e., Yes/No) on the following questions:

• Factual consistency (per patient turn): “Is all clinical information in this patient turn consistent with the source GP record? Please always answer Yes for turns containing no recordrelated clinical information, e.g., greetings or conversational filler. Answer No only when unsupported or contradictory content is presented.”

• Clinical realism (per patient turn): “Could this turn plausibly have been said by a real patient in a primary-care consultation?”

• Persona adherence (per trajectory): “Does the patient’s behavior across the dialogue reflect the assigned persona configuration (e.g., vague or incomplete answers under low anamnestic fidelity)?”

Results. Table 6 reports the proportion of Yes ratings per dimension together with inter-rater agreement (i.e., Cohen’s κ) between two senior physicians. The simulator produces patient turns that are highly consistent with the source records and clinically plausible, with strong agreement across all dimensions. The persona-adherence score further indicates that the simulator reliably reflects the assigned challenging personas, including low anamnestic fidelity and disorientation, rather than defaulting to a cooperative patient.

<table><tr><td>Rating dimension (unit)</td><td>% Yes</td><td>Cohen&#x27;s κ</td></tr><tr><td>Factual consistency with source record (per turn)</td><td>96.9</td><td>0.98</td></tr><tr><td>Clinical realism (per turn)</td><td>93.8</td><td>0.92</td></tr><tr><td>Persona adherence (per trajectory)</td><td>87.5</td><td>0.81</td></tr></table>

Table 6: Physician audit of 32 interaction trajectories produced by the patient simulator (two per evaluated model, contrasting standard and challenging personas).

## C.5 Constrained Group Relative Policy Optimization Details

We benchmark Constrained Group Relative Policy Optimization (C-GRPO) (Girgis et al., 2026), which augments GRPO with a component-wise Lagrangian relaxation and operates over the environment’s admissible action set $\boldsymbol { \mathcal { M } } ( \boldsymbol { s } _ { t } )$ (Section 4). We detail its optimization below.

GRPO. For each patient prompt $s _ { 0 } ,$ GRPO (Guo et al., 2025) samples a group of G trajectories $\{ \tau _ { i , 1 } , . . . , \tau _ { i , G } \} \sim \pi _ { \theta } ( \cdot \mid x _ { i } )$ . Each trajectory receives a scalar reward, and a normalized advantage is computed by standardizing these rewards within the sampled group using the group mean and standard deviation.

Policy Masking. At each step, C-GRPO masks the policy onto the environment’s admissible action set $\boldsymbol { \mathcal { M } } ( \boldsymbol { s } _ { t } )$ (Section 4), with mask $m ( s _ { t } , a ) =$ $\mathbb { 1 } [ a \in \mathcal { M } ( s _ { t } ) ]$ , yielding the masked, renormalized policy

$$
\pi _ { \theta } ^ { m } ( a \mid s _ { t } ) = { \frac { m ( s _ { t } , a ) \pi _ { \theta } ( a \mid s _ { t } ) } { \sum _ { a ^ { \prime } \in { \mathcal { A } } } m ( s _ { t } , a ^ { \prime } ) \pi _ { \theta } ( a ^ { \prime } \mid s _ { t } ) } } .\tag{2}
$$

This standard action-masking step steers the sampled group toward valid clinical trajectories and concentrates the relative-advantage signal on meaningful clinical variations.

Lagrangian Relaxation. To optimize the constrained objective $( \mathrm { E q . 1 } )$ , we employ a Lagrangian relaxation by introducing dual variables $\lambda _ { k } \ge$ 0 to form the unconstrained min-max objective max $_ { \tau \pi } \operatorname* { m i n } _ { \lambda \geq 0 } { \mathcal { L } } ( \pi , \lambda )$ , where ${ \mathcal L } ( \pi , \lambda ) = J _ { R } ( \pi ) -$ $\begin{array} { r } { \sum _ { k \in \mathcal { K } } \lambda _ { k } ( \bar { J _ { C _ { k } } } ( \pi ) - \epsilon _ { k } ) } \end{array}$

To integrate this into GRPO without variance distortion, we standardize the reward and costs independently. For each trajectory $\tau _ { i , g }$ , we compute component-wise standardizations $Z _ { R } ( \tau _ { i , g } )$ and $Z _ { C _ { k } } ( \tau _ { i , g } )$ using their respective within-group means and standard deviations. The constrained group-relative advantage $\hat { A } _ { i , g } ^ { c }$ is then constructed by linearly combining these standardized components using the current Lagrangian weights:

$$
\hat { A } _ { i , g } ^ { c } = Z _ { R } ( \tau _ { i , g } ) - \sum _ { k \in \mathcal { K } } \lambda _ { k } Z _ { C _ { k } } ( \tau _ { i , g } ) .\tag{3}
$$

C-GRPO executes an iterative primal-dual optimization scheme. The primal step updates the

projected policy $\pi _ { \theta } ^ { m }$ using a PPO-style clipped surrogate objective with the constrained advantage:

$$
\begin{array} { l } { \displaystyle \mathcal { T } _ { \mathrm { C . G R P O } } ( \theta ) = \frac { 1 } { B G } \sum _ { i = 1 } ^ { B } \sum _ { g = 1 } ^ { G } [ \frac { 1 } { T _ { i , g } } \sum _ { t = 1 } ^ { T _ { i , g } } \Bigg \{  }  \\ { \displaystyle \operatorname* { m i n } ( \rho _ { i , g , t } ^ { m } ( \theta ) \hat { A } _ { i , g } ^ { c } ,  } \\ {   \mathrm { c l i p } ( \rho _ { i , g , t } ^ { m } ( \theta ) , 1 - \epsilon _ { c } , 1 + \epsilon _ { c } ) \hat { A } _ { i , g } ^ { c } )  } \\ { \displaystyle  - \beta \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } ^ { m } \parallel \pi _ { \mathrm { r e f } } ^ { m } ) ( s _ { i , g , t } ) \} ] , } \end{array}\tag{4}
$$

where B is the batch size, $T _ { i , g }$ is the trajectory length, $\rho _ { i , g , t } ^ { m } ( \theta )$ is the probability ratio between the current and old projected policies, $\pi _ { \mathrm { r e f } } ^ { m }$ is the reference policy (i.e., the initial supervised finetuned model) projected via the same mask $m ,$ and $\epsilon _ { c } , \beta$ are hyperparameters.

Following the policy update, the dual step adjusts the multipliers via projected gradient descent based on the empirical constraint violation $\hat { C } _ { k } \mathbf { . }$ $\begin{array} { r c l } { \lambda _ { k } } & {  } & { [ \lambda _ { k } + \eta _ { \lambda } \overline { { ( \hat { C } _ { k } - \epsilon _ { k } ) } } ] _ { + } } \end{array}$ , where $\eta _ { \lambda }$ is the dual learning rate, so that when a constraint is exceeded its multiplier increases until the policy returns below the threshold. We apply this update to both cost terms in $\mathcal { K } = \{ \mathrm { s a f e t y }$ , diagnostic\_cost} (Appendix C.2). Because missed escalations are high-stakes, the safety multiplier would otherwise grow large and destabilize training, so we cap it at $\lambda _ { \mathrm { s a f e t y } } \le p _ { \mathrm { m g m t } } = 6$ . This does not constitute a hard guarantee that the safety constraint is satisfied; we therefore report the resulting safety levels directly.

## C.6 Hyperparameter

The specific hyperparameter values used to compute the reward and cost terms are summarized in Table 7 below.

Reward-coefficient selection. For a fair comparison, we do not fix the reward coefficients a priori but tune the baseline GRPO and C-GRPO under the same protocol. For each method, we sweep multiple combinations of the diagnosis and treatment reward weights $( w _ { \mathrm { d i a g } } , w _ { \mathrm { t r e a t } } )$ and the safety multiplier cap p<sub>mgmt</sub>, and report each method under its best-performing combination. The selected values are listed in Table 7.

The environment also issues two fixed protocol penalties. A workflow violation penalty c<sub>workflow</sub> is applied when the agent attempts an inadmissible action forbidden by the mask rule (e.g., treat before diagnose), and a timeout penalty c<sub>timeout</sub> is applied when an episode does not reach a terminal action within the 10-turn limit.

<table><tr><td>Hyperparameter</td><td>Symbol</td><td>Value</td></tr><tr><td colspan="3">Rewards</td></tr><tr><td>Diagnosis Accuracy Weight</td><td>Wdiag</td><td>+5</td></tr><tr><td>Treatment Score Weight</td><td>Wtreat</td><td>+3</td></tr><tr><td colspan="3">Costs</td></tr><tr><td>Safety Threshold</td><td>Tsafety</td><td>0</td></tr><tr><td>Safety Multiplier Cap</td><td>pmgmt</td><td>6</td></tr><tr><td>Diagnostic Cost Threshold</td><td>Tcost</td><td>0.2</td></tr><tr><td>Workflow Violation Penalty</td><td>Cworkflow</td><td>5.0</td></tr><tr><td>Timeout Penalty</td><td>Ctimeout</td><td>5.0</td></tr></table>

Table 7: Hyperparameters used for computing the reward and cost terms.

![](images/58cab99f20dcddfbe3564b391d2ba733a223c88c5a9efedddefea09222382537.jpg)  
Figure 13: Total return vs. training steps for fine-tuning agents initialized from Qwen2.5 (7B), using baseline GRPO and C-GRPO. Faint lines show raw returns while bold lines show the exponential moving average.

## C.7 Additional Results

We report additional results here: Table 8 provides extended zero-shot evaluation for the remaining off-the-shelf models; Table 9 lists the average dialogue turns per episode for the GRPO and C-GRPO agents; Table 10 restates the RL comparison with standard errors of the mean; and Figure 13 shows the total return (∆) trajectories of GRPO and C-GRPO over training.

## D Ethical Consideration and Applications

## D.1 Potential Risks

Our benchmark is constructed using real EHRs. Releasing data derived from real clinical cases introduces a potential re-identification risk, particularly for patients with rare conditions or distinctive clinical findings. Although we apply stringent deidentification pipelines to mitigate this, a marginal residual risk may remain. Additionally, there is a risk that the medical management plans included in the dataset could be misinterpreted as universally applicable medical advice. While these cases and their corresponding management plans have been rigorously audited by senior physicians to ensure clinical soundness, they reflect specific, localized medical decisions. Therefore, this benchmark dataset is provided solely for academic evaluation. It remains a research artifact and must not be used to guide direct patient care or real-world clinical decision-making.

## D.2 Intended Use Specifications

Intended Use The benchmark and environment are intended strictly for academic research in natural language processing, RL, and the development of clinical agents. It supports the systematic evaluation and optimization of models’ abilities to navigate clinical workflows and make safe decisions.

Restrictions This benchmark dataset and the accompanying environment do not constitute a medical device. They are designed exclusively as a research benchmark and should not be used as a standalone diagnostic system, nor deployed directly in real-world clinical or healthcare settings. Highstakes medical or commercial use is strictly prohibited without further extensive clinical validation, regulatory approval, and ethical review.

## D.3 Terms of Use

This section outlines the terms and conditions for the use of our benchmark and environment. By using the code and datasets in this project, users agree to the following terms:

Prohibited Use The code and datasets shall not be used for commercial purposes without prior written consent from the authors. Furthermore, users are strictly prohibited from using the dataset for real-world clinical decision-making, patient care, or attempting to re-identify any patients or healthcare providers from the anonymized records.

Attribution When using or referencing the code, environment, or datasets, users must provide proper attribution to the original authors.

No Warranty This project is provided "as is" without warranties of any kind, either expressed or implied, including but not limited to fitness for a particular purpose or clinical accuracy. The authors are not responsible for any damage, loss, or medical errors resulting from the use of this project.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="3">Clinical Quality ↑</td><td colspan="2">Safety Failures ↓</td><td colspan="2">Efficiency↓</td></tr><tr><td>(%)</td><td>Dx Acc. Tx Score Mgmt Acc. (%)</td><td>(%)</td><td>Miss Refer (%)</td><td>Miss GP (%)</td><td>Avg. Turns</td><td>Cost (USD)</td></tr><tr><td rowspan="6">Open-Source</td><td>General Purpose</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DEEPSEEK-V4-FLASH</td><td>63.9</td><td>40.6</td><td>53.2</td><td>49.3</td><td>45.1</td><td>7.58</td><td>$210</td></tr><tr><td>HUNYUAN-3 PREVIEW</td><td>66.4</td><td>40.7</td><td>53.0</td><td>46.8</td><td>47.2</td><td>7.09</td><td>$241</td></tr><tr><td>MIMO-V2.5 PRO</td><td>66.0</td><td>38.3</td><td>55.3</td><td>59.2</td><td>34.5</td><td>7.36</td><td>$189</td></tr><tr><td>GLM-5.1</td><td>54.2</td><td>34.3</td><td>54.4</td><td>51.2</td><td>41.5</td><td>8.58</td><td>$77</td></tr><tr><td>QWEN3.7-MAX</td><td>63.1</td><td>41.1</td><td>56.1</td><td>64.7</td><td>29.2</td><td>6.39</td><td>$181</td></tr></table>

Table 8: Extended results (Appendix) for the remaining off-the-shelf models.

<table><tr><td rowspan=1 colspan=2>Avg.MethodTurns ↓</td></tr><tr><td rowspan=1 colspan=1>QWEN2.5 (Base)</td><td rowspan=1 colspan=1>6.72</td></tr><tr><td rowspan=1 colspan=1>GPAGENT (GRPO)</td><td rowspan=1 colspan=1>6.51 (-0.21)</td></tr><tr><td rowspan=1 colspan=1>GPAGENT (C-GRPO)</td><td rowspan=1 colspan=1>6.24 (-0.48)</td></tr></table>

Table 9: Average dialogue turns per episode for Qwen2.5 (7B) optimized with GRPO and C-GRPO (lower is better); green denotes a reduction relative to the base model. Safety failure rates are reported in the main results (Table 2).
<table><tr><td rowspan="3">Method</td><td colspan="3">Clinical Quality ↑</td><td colspan="2">Safety ↓</td><td>Eff. ↓</td></tr><tr><td>Dx Acc. (%)</td><td>Tx Score (%)</td><td>Mgmt Acc. (%)</td><td>Miss Refer (%)</td><td>Miss GP (%)</td><td>Cost (USD)</td></tr><tr><td>QWEN2.5 (Base)</td><td>40.4 ± 3.2</td><td>18.4 ± 1.7</td><td>47.4 ± 2.3</td><td>89.6 ± 2.2</td><td>21.1 ± 2.4</td><td>$445 ± 24</td></tr><tr><td>GPAGENT (GRPO)</td><td>48.5 ± 3.4</td><td>20.0 ± 1.7</td><td>48.7 ± 2.3</td><td>83.6 ± 2.6</td><td>28.5 ± 2.7</td><td>$438 ± 25</td></tr><tr><td>GPAGENT (C-GRPO)</td><td>52.2 ± 3.4</td><td>20.5 ± 1.8</td><td>51.0 ± 2.3</td><td>82.1 ± 2.8</td><td>26.1 ± 2.7</td><td>$435 ± 26</td></tr></table>

Table 10: RL fine-tuning results with standard errors. This table reports the same evaluation protocol as Table 2, with mean ± standard error across test cases.

Liability The authors shall not be held liable for any direct, indirect, incidental, special, exemplary, or consequential damages arising in any way out of the use of this benchmark or environment.

Updates and Changes The authors reserve the right to make changes to the terms of this license or the benchmark itself at any time, including the removal of specific cases if unforeseen privacy risks are identified.

## D.4 Compliance with Artifact Usage and Intended Use Specifications

## D.4.1 Compliance with Existing Artifact Usage

In this study, we utilized several existing artifacts, including de-identified real-world general-practice encounter records, publicly available model APIs, open-source language models, and external clinical cost references. All clinical records were processed under the applicable institutional data-governance requirements and were de-identified before use to protect patient privacy. The records were further transformed into structured case representations and manually checked against the original clinical notes to reduce extraction errors and hallucinations.

We also used external computational artifacts, including proprietary model APIs, open-source model checkpoints, and clinical cost references, solely for benchmarking, simulation, and evaluation purposes. Their use was conducted in accordance with their respective licenses, access policies, and terms of service. No proprietary model outputs or third-party artifacts are redistributed as part of the benchmark unless explicitly permitted by their corresponding licenses.

## D.4.2 Specification of Intended Use for Created Artifacts

Our research produces two primary artifacts: (1) GPAGENTBENCH-2K, a benchmark dataset and CMDP-based clinical-agent environment for primary-care decision-making, and (2) the accompanying evaluation code, simulator, and RL scripts.

Dataset License GPAGENTBENCH-2K is under a non-commercial research-use license. This license permits use, redistribution, and adaptation for academic and research purposes with proper attribution, but prohibits commercial use, clinical deployment, or incorporation into commercial products without additional permission. The accompanying evaluation code and scripts will be released under the Apache License 2.0.

## GPAGENTBENCH-2K: Benchmark Dataset and CMDP Environment

Intended Use: GPAGENTBENCH-2K is intended for academic research on clinical language agents, constrained decision-making, RL, and safety-aware medical AI. The benchmark supports systematic evaluation of agents’ ability to navigate a complex primary-care action space, including evidence gathering, diagnosis, treatment planning, and safety-informed referral. It is designed to facilitate research on clinical decision-making under uncertainty, especially the trade-off between diagnostic quality, safety constraints, and diagnostic efficiency.

Restrictions: GPAGENTBENCH-2K is a research benchmark and must not be used as a standalone clinical decision-support system. It should not be used for real-world diagnosis, treatment recommendation, triage, referral decisions, or any other patient-facing clinical application. Commercial use of the dataset is not permitted. Any use in high-stakes healthcare settings requires additional clinical validation, institutional review, regulatory assessment, and expert oversight.

Ethical Considerations: The benchmark is constructed from de-identified clinical records and validated by senior physicians to improve data quality and clinical plausibility. Nevertheless, the dataset may still reflect biases from the source records, healthcare system, case selection, and annotation process. Users should therefore interpret benchmark results as research findings rather than evidence of clinical readiness. Researchers extending or applying the benchmark are encouraged to preserve its intended focus on safety-aware primary-care decision-making and to report limitations transparently.

## Evaluation Code, Simulator, and RL Scripts

Intended Use: The released codebase is intended to support reproducible research on clinicalagent benchmarking and constrained policy optimization. It includes tools for running agentenvironment interactions, parsing structured actions, computing diagnostic and management metrics, and evaluating safety and efficiency outcomes.

Restrictions: The code is provided for research and benchmarking purposes. Although the code may be released under a permissive software license, the accompanying dataset remains restricted to non-commercial research use. Users must not combine the code and dataset to build or deploy clinical products without additional validation, authorization, and compliance review.

Responsible Use: Users should carefully document model settings, prompts, evaluation protocols, and any modifications to the environment. Because the benchmark concerns medical decision-making, reported results should avoid overstating clinical capability and should clearly distinguish benchmark performance from real-world clinical safety.

## D.5 Data Collection and Anonymization Procedures

The original clinical records were rigorously anonymized prior to any processing. Our verification includes: (i) automated and manual screening to remove all personally identifiable information (PII) and protected health information (PHI); (ii) redacting or generalizing high-risk quasi-identifiers (e.g., uncommon procedures or extreme age values); and (iii) replacing specific dates, locations, and medical record numbers with synthetic placeholders (e.g., ENC-XXXX). The released data explicitly excludes physician names, facility names, and other extraneous identifying details. Where quasiidentifiers remain that could lead to deductive disclosure, we apply further redaction prior to release to strictly minimize re-identification risk.

## D.6 Artifact Documentation

Domain Coverage The benchmark targets primary care and general practice medicine, focusing on the diagnosis, management and referral of patients based on clinical presentations.

Language and Linguistic Phenomena All clinical records, patient interaction observations, and agent prompts are in English. The dataset heavily features domain-specific linguistic phenomena, including medical jargon, standard clinical abbreviations, and EHR shorthand. Furthermore, the environment requires a specific structural linguistic format, with agent actions output as structured text blocks enclosed by predefined XML-style tags.

Demographic Representation The dataset includes basic demographic attributes, specifically age and gender, where available in the source clinical records. Ethnicity is not recorded in this dataset. To ensure strict patient privacy and adhere to de-identification standards, exact dates of birth and highly specific demographic intersections have been generalized or redacted.

## D.7 Human Evaluation Details

To validate the quality and safety of the collected clinical records, we conducted a rigorous human evaluation study. This study obtained ethical approval, and written consent was collected from each volunteer prior to the assessment.

## D.7.1 Physician Demographics

We recruited two volunteer senior physicians (one male, one female). Both are board-certified in their respective countries of practice and hold prestigious distinctions with 35 and 41 years of clinical experience, respectively. Participation was voluntary and uncompensated. Both participants use English as their primary language for clinical documentation and instruction.

## D.7.2 Study Procedure

The evaluation was conducted in an online environment. Both participants received a standardized briefing describing the study objective, task format, and evaluation interface, followed by a brief training phase (two example cases) to familiarize themselves with the interface and the clinical rubric.

The physicians were asked to make a binary choice ("Pass" or "Fail") for each questions. Following the individual questions, the physicians were required to make a final overall decision to either retain or discard the case. Ultimately, only cases that received unanimous "Pass" grades across all individual criteria and an overall "Retain Case" decision from both senior physicians were included in the final dataset. This rigorous quality control process ensures the safety, reliability and usability of the dataset for future research, guaranteeing that the benchmark is based on expert-validated, high-quality data.

## D.7.3 Evaluation Interface

Figure 14 illustrates the custom interface we developed to facilitate the quality inspection of the clinical records by the physicians. The interface utilizes a three-column layout to minimize context switching, displaying the raw clinical record, the structured case data, and a dynamic clinical validation rubric that updates based on the selected management category.

## D.7.4 Disclaimer for Annotators

Thank you for participating in our clinical evaluation. Please read the following before you begin.

Your participation in this study is completely voluntary, and you may stop at any time without penalty. During this task, you will review anonymized clinical records that explicitly exclude all personally identifiable information and protected health information. Your ratings and responses will also be kept strictly confidential. Although the task is low risk, some cases may include clinical descriptions of severe illness that could be uncomfortable; you are free to skip any item or stop the evaluation at any time. If you have any questions or concerns during the task, please contact the study organizers.

## D.7.5 Instructions for Clinical Validation

Thank you for participating in our study. Please read the instructions below carefully before you begin evaluating the cases.

You will first complete a short training phase with two example cases to familiarize yourself with the three-column interface and the dynamic grading rubric. These examples are for practice only and are not included in our final evaluation.

For each actual case, review the clinical notes and the proposed management plan. Next, select the appropriate management category in the rubric to reveal the specific evaluation criteria. You will be asked to provide a "Pass" or "Fail" grade for several criteria depending on the case type. For "Diagnose & Manage" cases, please evaluate whether the clinical evidence is sufficient for a general practitioner to confidently confirm the diagnosis, if the condition is commonly managed in primary care, if the stated ground-truth diagnosis is correct, and whether the proposed treatment plan is safe and aligned with primary care guidelines. Alternatively, for "Abstain & Refer" cases, please evaluate whether the severity, acuity, or complexity of the findings strictly necessitates escalating care, and whether the chosen referral destination is clinically appropriate.

After grading the individual criteria, you must provide an overall decision to either retain the case (if all criteria are fundamentally correct) or discard the case (if any criterion is obviously incorrect, unsafe, or highly controversial). If you choose to discard a case, please provide a brief explanation in the text box below.

## D.7.6 Data Consent

The information you provide in this study will be used only for academic research. Your responses will be stored securely and handled confidentially; we will remove direct identifiers and, where possible, report results only in aggregate to protect your privacy. Participation is voluntary. You may withdraw from the study at any time without penalty, and you may request that your data not be used in our analyses to the extent permitted after withdrawal. If you have any questions about data handling or use, please feel free to contact us.

![](images/fac12c9c0e64c9ef628d78acdb21885a2a19a8934c00c358fa264e11c9319340.jpg)  
(b) Abstain & Refer case.  
Figure 14: Examples of the interface used for clinical expert validation.

## D.8 Use of AI Assistants in Research

In our study, generative AI assistants are used sparingly and in accordance with the guidelines on ARR’s Policy on AI Writing Assistance. We utilize ChatGPT for basic polishing and grammar checks. These tools are applied minimally to ensure the authenticity of our work and to adhere strictly to the regulatory standards set by ARR. Our use of these AI tools is focused, responsible, and aimed at supplementing rather than replacing human input and expertise in our research.