# SKILLSANDBOX: SKILL VERIFICATION VIA DYNAMIC SCENARIO SYNTHESIS

Serin Kim Kwangwook Seo Dokyung Song Jinyoung Yeo Dongha Lee<sup>†</sup> Yonsei University

{kimserin,tommy2130,dokyungs,jinyeo,donalee}@yonsei.ac.kr

## ABSTRACT

Self-evolving agents distill task-solving experience into skills for future reuse, but these skills can encode incorrect procedures or non-transferable knowledge. It is therefore critical to verify each skill’s reusability: whether its guidance remains useful beyond the experience from which it was distilled. Such verification requires observing how a skill affects execution in new tasks, yet existing tasks may not expose the situations where the target skill can actually be exercised. To construct such situations, we propose SKILLSANDBOX, a framework that dynamically synthesizes a task and its environment for each skill that are skill-relevant yet novel. A PROPOSER specifies the conditions to preserve and the source-specific details to vary, a BUILDER constructs an executable scenario, and a VERIFIER compares executions with and without the skill. The VERIFIER assesses executability, utility, and efficiency to assign a Keep or Reject verdict, determining whether the skill enters the library. Across ALFWorld and WebShop with three models, SKILLSANDBOX consistently yields the strongest downstream performance and improved execution efficiency. Further analyses examine whether these gains reflect accurate assessment of skill reusability and identify which components of SKILLSANDBOX contribute to them. [CODE]

## 1 INTRODUCTION

Self-evolving agents seek to improve their capabilities by learning from their own experiences (ang Gao et al., 2026; Yang et al., 2026). They distill task-solving experience into skills (Anthropic, 2025) that provide procedural guidance and store them in a library for future reuse (Wang et al., 2023; Ouyang et al., 2026b). However, distilled skills do not always guarantee the performance improvement (Li et al., 2026; Han et al., 2026; Gautam et al., 2026). Because task-solving experience contains redundant exploration, mistakes, and task-specific details, distillation may yield incorrect procedures or non-transferable knowledge (Ni et al., 2026; Ma et al., 2026; Xiong et al., 2026). Once stored and retrieved, such skills can repeatedly interfere with the agent’s behavior (Dong et al., 2026; Liang et al., 2026). Self-evolving agents therefore need to verify each skill’s reusability: whether the guidance remains useful when applied beyond the source experience.

Such verification requires execution evidence from a scenario that is both skill-relevant and novel. That is, the agent should actually encounter the situation the skill addresses in a new environment, so that the evidence reveals whether the guidance improves its behavior beyond the source experience. Because different skills address different situations, the scenario that provides this evidence is inherently skill-specific. However, existing approaches (Wang et al., 2025a; Zhang et al., 2026; Ouyang et al., 2026a) verify skills against a static pool of tasks that already exists, and merely select from it. Replaying the source task likely exposes the targeted situation, but only in the context the skill was distilled from, so the evidence is relevant but not novel (Chen et al., 2026a; Liu et al., 2026). Evaluating the skill on other existing tasks introduces new contexts, but since these tasks are defined independently of the skill, they need not expose the targeted situation.

Verification should therefore shift from selecting existing tasks, to constructing the task and environment around the target skill, on demand. To this end, we propose SKILLSANDBOX, a framework that dynamically synthesizes a task and environment where the target situation arises in a context different from the source experience to verify skills. Each synthesized task and its configured environment form a verification scenario, which serves as a controlled test of that skill. SKILLSANDBOX consists of a PROPOSER, a BUILDER, and a VERIFIER, which design, construct, and evaluate this test, respectively. (i) The PROPOSER specifies the conditions that reproduce this situation and the modifications that distinguish the scenario from the source experience. (ii) The BUILDER realizes this specification as an executable task and its configured environment. (iii) The VERIFIER compares executions in the scenario with and without the skill and measures the effect of the skill in terms of executability, task success, and efficiency. Consequently, a reusable skill yields a positive effect across its scenarios and is admitted to the library, whereas a skill that is incorrect, overly specific, or not actionable yields no benefit or degrades execution and is rejected.

We evaluate SKILLSANDBOX on ALFWorld and WebShop by comparing the library of verified skills with libraries constructed by baseline methods, which either accumulate every generated skill or continually update the library as experience accumulates. Across three models, the verified library achieves the highest success rate with the fewest steps on both benchmarks. Further analyses show that the performance gains originate from the synthesized scenarios, as verification on source tasks or randomly selected tasks yields smaller gains and the gap widens on task types unseen in the source experience. In addition, the verification predicts the helpful and harmful effects of individual skills on held-out tasks. Each criterion also proves necessary, as the full VERIFIER outperforms variants that omit executability or efficiency.

Our contributions are as follows:

• We identify that skill verification should shift from selecting existing tasks to constructing the task and environment around the target skill on demand.

• We propose SKILLSANDBOX, which synthesizes a task and environment that expose the situation needed to verify the skill. Its PROPOSER and BUILDER generate scenarios, and VERIFIER measures the reusability of skill and determines whether to keep or reject it.

• Our experiment confirms that the skill library verified by SKILLSANDBOX significantly improves downstream performance. Extensive analyses validate the effect of SKILLSANDBOX and show that the synthesized scenarios provide a reliable verification environment.

## 2 PRELIMINARY ANALYSES: SKILL-SPECIFIC EVIDENCE IS SPARSE IN EXISTING EXPERIENCE

We first ask whether the scenario synthesis is actually necessary to verify skills. Can existing tasks provide the evidence needed for verification? To answer this question, we examine alternatives to scenario synthesis and evaluate whether either one reliably provides the skill-specific evidence.

• Question I: Do semantically similar tasks expose the skill-relevant situation?

• Question II: How sparse are skill-relevant situations in downstream experience?

## 2.1 QUESTION I: DO SEMANTICALLY SIMILAR TASKS EXPOSE THE SKILL-RELEVANT SITUATION?

We examine whether semantic similarity can identify existing tasks that expose the skill-relevant situation. We use 50 skills per benchmark, 140 heldout tasks in ALFWorld (Shridhar et al., 2020) and nested pools of 200, 300, and 500 tasks in Web-Shop (Yao et al., 2022a), keeping the same 50 skills across pool sizes. For each skill, we select the 10 held-out tasks whose instructions are most similar to the skill text. The agent executes each selected task with the target skill provided, and an LLM judge determines from the trajectory whether execution encounters the skill-relevant situation.

Table 1: Mean skill-relevant tasks out of 10 retrieved tasks. N: candidate task count.
<table><tr><td>Executor</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>N</td><td>140</td><td>200 300 500</td></tr><tr><td>Gemini 3.1 Flash</td><td>5.96</td><td>5.96 5.90 6.10</td></tr><tr><td>Qwen3.5-27B</td><td>6.00</td><td>6.60 6.58 6.66</td></tr></table>

Instruction similarity does not reliably identify tasks that exercise the target skill. As shown in Table 1, the skill-relevant situation is observed in only up to 6.66 out of 10 selected tasks per skill across the settings. Even expanding the candidate pool from 200 to 500 tasks yields only marginal gains in skill-relevant coverage among the top-10 retrieved tasks. The main reason to this is that semantic similarity is measured at the task level, whereas skill relevance depends on what occurs during execution. Tasks with similar instructions can still induce different states, interactions, and intermediate decisions, and therefore need not expose the same skill-relevant situation. Consequently, task-level similarity does not reliably provide the execution evidence required for skill verification.

## 2.2 QUESTION II: HOW SPARSE ARE SKILL-RELEVANT SITUATIONS IN DOWNSTREAM EXPERIENCE?

A second alternative is to accumulate verification evidence from subsequent downstream experience (Zhang et al., 2026; Xiong et al., 2026; Ouyang et al., 2026a). We construct a library of 50 skills and stream 500 held-out tasks in a fixed random order, repeating each setting three times. For each task, referenced skill IDs are recorded in rationale and counted. We vary the number of skills accessible, with Top-3 and Top-5 Retrieval providing the top-3 or 5 skills retrieved per task, respectively, and Full Library all 50 skills on every task. Figure 1 (a) reports the percentage of skills used in at least one task, and (b) reports the percentage used in at least 5 distinct tasks.

![](images/24340f527f2dccbc7295b8b61e18c1a05e820461490751b57aff238e48adebcc.jpg)

Skill-relevant situations remain sparse as downstream experience accumulates. (a) shows that coverage grows slowly after 300 tasks. Executing 200 additional tasks raises the percentage of skills used at least once by only 4.7–6.7 percentage points. After 500 tasks, only 68% of skills are used under Top-3 Retrieval, 83% under Top-5, and 70% under Full Library. Thus, depending on the access setting, 17–32% of skills remain unexercised, which means that their effects on the agent’s behavior are never observed, leaving their reusability unverifiable. More reliable verification benefits from repeated evidence across distinct tasks, yet such opportunities are even more sparse. As shown in (b), only 44–55% of skills are used in at least 5 tasks. Even after 500 tasks, 45–56% of the skills fail to accumulate five execution opportunities. Overall, downstream experience leaves a substantial number of skills with insufficient evidence for verification.

![](images/6ef34300f2841807e267b637464edce13336b5bc64299611d4748bb8723cd3a1.jpg)  
Figure 1: Percentage of skills used in (a) at least one task and (b) at least 5 tasks as downstream tasks are streamed and solved sequentially (Gemini 3.1 Flash-Lite, WebShop). Mean of three runs of the same task stream; bands show one standard deviation.

Together, Questions I and II reveal complementary limitations of relying on existing tasks for verification. Neither reliably provides the skill-specific evidence that verification needs. We therefore construct verification scenarios that induce the target situation beyond its source experience.

## 3 METHODOLOGY

## 3.1 PROBLEM SETUP

We model an environment as $E = ( S , { \mathcal { A } } , { \mathcal { O } } , T , \kappa )$ , where $s , A ,$ and $\mathcal { O }$ are the state, action, and observation spaces, $T$ is the state-transition rule, and the configuration κ specifies the environment’s objects, arrangements, task-related data, and presentation of information. A task x specifies the goal and requirements. At each step, an agent π receives a partial observation (Kaelbling et al., 1998) $o _ { t } \in \mathcal { O }$ of state $s _ { t }$ and takes an action $a _ { t } \in { \mathcal { A } }$ , yielding $s _ { t + 1 } \sim T ( \cdot \mid s _ { t } , a _ { t } )$ , and the trajectory is $\tau _ { x } = ( o _ { 0 } , a _ { 0 } , o _ { 1 } , a _ { 1 } , \dots )$ . We call the environment and information encountered during execution the task context.

A synthesis specification $\delta$ programmatically specifies the requirements and changes that construct a new task $x _ { \delta }$ and configuration $\kappa _ { \delta }$ , which with the success criterion $g _ { \delta }$ define

$$
E _ { \delta } = ( { \mathcal { S } } , A , { \mathcal { O } } , T , \kappa _ { \delta } ) , \qquad \sigma _ { \delta } = ( x _ { \delta } , E _ { \delta } , g _ { \delta } ) .
$$

![](images/f43342de6ffbe9c78ac5d06d2d194a401d7b881bbaf9cd17f44bda5fe0034edf.jpg)  
Figure 2: Overview of skill verification via dynamic scenario synthesis.

## 3.2 SKILL VERIFICATION

A skill k = (condition, principle) is distilled from a source task $x _ { \mathrm { s r c } }$ and trajectory $\tau _ { \mathrm { s r c } } .$ The condition describes when the skill applies, and the principle provides knowledge or guidance for that situation. Verification assesses the skill’s reusability, defined as its ability to guide executable behavior that improves task success or execution efficiency in new situations satisfying the condition. We verify the skill’s reusability along three dimensions:

1. Executability: The principle can be executed through admissible actions in the environment when the condition is satisfied.

2. Utility: Applying the principle helps the agent achieve the task goal, improving task success relative to execution without the skill.

3. Efficiency: The principle enables the task completion with fewer actions than execution without the skill.

Evaluating these dimensions only on the source task $x _ { \mathrm { s r c } }$ does not establish reusability in new task contexts, while arbitrary tasks may not reproduce the situation described by the condition. Verification scenarios must therefore be both skill-relevant and novel. A scenario is skill-relevant when completing the task requires addressing the situation described by condition, in which the guidance in principle can be evaluated. It is novel when the agent encounters a task context that differs from the source experience.

## 3.3 SCENARIO SYNTHESIS

To construct scenarios that are both skill-relevant and novel, we reproduce the situation described by the condition while reconstructing the new task context. We jointly construct a task $x _ { \delta }$ and configuration $\kappa _ { \delta }$ so that the condition holds during execution, while varying the details. Execution in the resulting scenario tests whether the principle remains useful when the situation described by the condition occurs in a context different from the source experience. Figure 2 illustrates this synthesis process. In SKILLSANDBOX, the PROPOSER specifies how to reproduce the condition in a new context, and the BUILDER constructs the scenario according to this specification.

PROPOSER. The PROPOSER first uses the condition, $x _ { \mathrm { s r c } } ,$ and $\tau _ { \mathrm { s r c } }$ to jointly plan reproduction and reconstruction. It encodes its natural language plan as a programmatic specification $\delta ,$ using shared construction operations exposed by the construction interface $\mathcal { T } _ { E }$ . This interface specifies which task and configuration fields can be modified (e.g., list order, product option) and the operations permitted for each field (e.g., add distractor, remove, alter). The specification includes variables for objects and values that the BUILDER later instantiates. For reproduction, it specifies the requirements that the task query and configuration must satisfy for the situation described by the condition to occur during execution. For reconstruction, it specifies the modifications that produce a context different from the source experience while satisfying the reproduction requirements. The resulting specification δ is expressed by the operations permitted by $\dot { \mathcal { T } } _ { E } ,$ providing the BUILDER with a plan for scenario construction.

BUILDER. To construct an executable scenario according to given $\delta ,$ the BUILDER uses the task and configuration data $\mathcal { D } _ { E }$ (e.g., product database for WebShop, object list for Alf-World) available in $E .$ The BUILDER first binds the variables in δ to concrete objects and values from $\mathcal { D } _ { E }$ . It then translates the specified operations into an executable construction program for E and runs it. As specified in δ, this program constructs new task instruction, environment, and corresponding ground truth, using $\mathcal { D } _ { E }$

To faithfully reproduce the skill’s condition across diverse contexts, this construction proceeds within a validation loop. Using the same LLM model as the executor, the loop checks whether each constructed scenario satisfies the requirements in the PROPOSER’s specification δ and whether it is identical to any previously generated scenario for the same skill. Scenarios that fail validation are discarded, and new candidates are generated until five valid scenarios are obtained. Together, $x _ { \delta }$ and $E _ { \delta }$ reproduce the situation described by the condition in a reconstructed task context, allowing the agent to reuse the principle beyond the source experience. The BUILDER returns the resulting verification scenario $\sigma _ { \delta } \colon$

$$
\delta = \mathrm { P R O P O S E R } \big ( S k i l l , x _ { \mathrm { s r c } } , \tau _ { \mathrm { s r c } } , \mathcal { T } _ { E } \big ) , \qquad \sigma _ { \delta } = \mathrm { B U I L D E R } _ { E } \big ( \delta ; \mathcal { D } _ { E } \big ) .
$$

To illustrate what kind of scenarios are generated and how they are constructed, we provide modification statistics (Appendix $\mathrm { C } ) .$ and case studies (Appendix D).

VERIFIER. For each synthesized scenario, the VERIFIER compares trajectories $\tau ^ { + }$ and $\tau ^ { - }$ produced by the same executor with and without skill k (Figure 2, right). At aligned steps with identical pre-action observations, ∆u measures the difference in terminal success rewards discounted by the remaining action counts, capturing utility and efficiency. The executability indicator e is 1 when the action in $\tau ^ { + }$ implements the skill’s instruction and 0 otherwise. We average $e \Delta u$ over retained step pairs within each trajectory pair and then across scenarios to score reusability and assign a verdict (score calculation are detailed in Appendix B):

$$
R ( k ) = \widehat { \mathbb { E } } [ e \Delta u ] , \qquad \mathbf { V } \scriptscriptstyle \mathrm { E R I F I E R } ( k ) = \left\{ \begin{array} { l l } { \mathbf { K E E P } , } & { R ( k ) > 0 , } \\ { \mathbf { R E J E C T } , } & { R ( k ) \le 0 } \end{array} \right. .\tag{1}
$$

## 4 EXPERIMENTS

We first present the experimental setup (Section 4.1) and main results (Section 4.2). We compare SKILLSANDBOX with baselines that construct skill libraries with or without verification. To examine how effectively each method retains reusable skills, we build a library using each method and evaluate it on the same held-out tasks.

## 4.1 EXPERIMENTAL SETUP

Benchmarks. We evaluate on ALFWorld (Shridhar et al., 2020) and WebShop (Yao et al., 2022a). ALFWorld provides six household task types, and WebShop requires agents to find and purchase products that satisfy user instructions.

Metrics. We report success rate (SR, %) on both benchmarks, average task score (Score, 0–100) on WebShop, and average steps across all episodes. To characterize how a skill library changes individual task outcomes beyond SR, we additionally report Recovery (Rec.) and Regression (Reg.) relative to the run without skill library of the same model. Recovery is the percentage of all held-out tasks that fail without skills but succeed with the library, whereas Regression is the percentage that succeed without skills but fail with the library.

Baselines. No Skill executes tasks without a skill library. Vanilla Skill retains all skills generated with the distillation prompt from SkillRL (Xia et al., 2026). We additionally compare with ReasoningBank (Ouyang et al., 2026b) and MemP (Fang et al., 2026b), as well as execution-based library curation methods including ExpeL (Zhao et al., 2024), ACE (Zhang et al., 2026), and SkillOS (Ouyang et al., 2026a). ExpeL, ACE, and SkillOS admit skills using execution feedback: ExpeL extracts insights from repeated attempts at the source task, whereas ACE and SkillOS update the library from subsequent tasks that are selected independently of any skill. Detailed descriptions of each baseline are provided in Appendix A. SKILLSANDBOX retains only the skills from Vanilla Skill that are verified as Keep.

Table 2: Main results on held-out ALFWorld and WebShop tasks. Vanilla Skill, ReasoningBank, and MemP retain skills without verification; ExpeL uses source-task replay, ACE and SkillOS use downstream feedback, and SKILLSANDBOX uses synthesized scenarios for verification. Bold marks the best value per model; colored percentages are relative changes from No Skill.
<table><tr><td rowspan="2">Method</td><td colspan="4">ALFWorld</td><td colspan="5">WebShop</td></tr><tr><td>SR↑</td><td>Steps↓</td><td>Rec.↑</td><td>Reg.↓</td><td>SR↑</td><td>Score↑</td><td>Steps↓</td><td>Rec.↑</td><td>Reg.↓</td></tr><tr><td colspan="10">Qwen3.5-9B</td></tr><tr><td>No Skill</td><td>32.1</td><td>23.8</td><td></td><td></td><td>18.8</td><td>47.1</td><td>8.0</td><td></td><td></td></tr><tr><td>Vanilla Skill</td><td>57.9+80.4%</td><td>19.3</td><td>30.0</td><td>4.3</td><td>15.6-17.0%</td><td>38.0-19.3%</td><td>9.6</td><td>5.0</td><td>8.2</td></tr><tr><td>ReasoningBank</td><td>53.6+67.0%</td><td>20.0</td><td>27.1</td><td>5.7</td><td>17.2-8.5%</td><td>38.4-18.6%</td><td>10.0</td><td>5.8</td><td>7.4</td></tr><tr><td>MemP</td><td>42.9+33.6%</td><td>22.1</td><td>14.3</td><td>3.6</td><td>17.2-8.5%</td><td>41.2-12.6%</td><td>9.4</td><td>6.6</td><td>8.2</td></tr><tr><td>ExpeL</td><td>42.1+31.2%</td><td>22.1</td><td>13.6</td><td>3.6</td><td>21.8+16.0%</td><td>52.9+12.4%</td><td>7.0</td><td>8.2</td><td>5.2</td></tr><tr><td>ACE</td><td>45.0+40.2%</td><td>21.7</td><td>18.6</td><td>5.7</td><td>15.6-17.0%</td><td>35.4-24.9%</td><td>10.4</td><td>5.2</td><td>8.4</td></tr><tr><td>SkillOS</td><td>40.0+24.6%</td><td>22.7</td><td>14.3</td><td>6.4</td><td>11.4-39.4%</td><td>25.2-46.6%</td><td>12.2</td><td>3.8</td><td>11.2</td></tr><tr><td>SKILLSANDBOX</td><td>66.4+106.9%</td><td>17.5</td><td>36.4</td><td>2.1</td><td>22.2+18.1%</td><td>59.1+25.5%</td><td>7.0</td><td>8.0</td><td>4.6</td></tr><tr><td colspan="10">Gemini 3.1 Flash-Lite</td></tr><tr><td>No Skill</td><td>56.4</td><td>19.9</td><td></td><td></td><td>25.6</td><td>58.9</td><td>5.6</td><td></td><td></td></tr><tr><td>Vanilla Skill</td><td>49.3-12.6%</td><td>20.6</td><td>6.4</td><td>13.6</td><td>27.6+7.8%</td><td>54.9-6.8%</td><td>8.0</td><td>7.2</td><td>5.2</td></tr><tr><td>ReasoningBank</td><td>52.9-6.2%</td><td>20.3</td><td>9.3</td><td>12.9</td><td>23.4-8.6%</td><td>53.7-8.9%</td><td>6.9</td><td>5.4</td><td>7.6</td></tr><tr><td>MemP</td><td>55.0-2.5%</td><td>19.5</td><td>12.1</td><td>13.6</td><td>25.0-2.3%</td><td>55.3-6.1%</td><td>6.3</td><td>5.0</td><td>5.6</td></tr><tr><td>ExpeL</td><td>50.0-11.3%</td><td>21.3</td><td>9.3</td><td>15.7</td><td>27.4+7.0%</td><td>59.4+0.8%</td><td>5.8</td><td>6.0</td><td>4.2</td></tr><tr><td>ACE</td><td>42.9-23.9%</td><td>22.0</td><td>5.7</td><td>19.3</td><td>20.8-18.8%</td><td>41.5-29.5%</td><td>10.4</td><td>6.6</td><td>11.4</td></tr><tr><td>SkillOS</td><td>51.4-8.9%</td><td>20.8</td><td>10.7</td><td>15.7</td><td>28.4+10.9%</td><td>56.4-4.3%</td><td>7.2</td><td>7.4</td><td>4.6</td></tr><tr><td>SKILLSANDBOX</td><td>60.0+6.4%</td><td>19.1</td><td>12.1</td><td>8.6</td><td>36.2+41.4%</td><td>64.8+10.0%</td><td>5.3</td><td>12.4</td><td>1.8</td></tr><tr><td colspan="10">Qwen3.5-27B</td></tr><tr><td>No Skill</td><td>75.7</td><td>15.4</td><td></td><td></td><td>32.2</td><td>55.4</td><td>8.0</td><td></td><td></td></tr><tr><td>Vanilla Skill</td><td>75.0-0.9%</td><td>15.0</td><td>7.9</td><td>8.6</td><td>31.8-1.2%</td><td>54.5-1.7%</td><td>8.5</td><td>8.8</td><td>9.2</td></tr><tr><td>ReasoningBank</td><td>67.1-11.4%</td><td>17.1</td><td>10.7</td><td>19.3</td><td>29.2-9.3%</td><td>53.0-4.4%</td><td>8.4</td><td>5.4</td><td>8.4</td></tr><tr><td>MemP</td><td>52.9-30.1%</td><td>19.8</td><td>2.9</td><td>25.7</td><td>32.4+0.6%</td><td>57.5+3.8%</td><td>8.6</td><td>7.2</td><td>7.0</td></tr><tr><td>ExpeL</td><td>70.0-7.5%</td><td>16.2</td><td>10.0</td><td>15.7</td><td>34.6+7.5%</td><td>54.7-1.2%</td><td>8.3</td><td>9.6</td><td>7.2</td></tr><tr><td>ACE</td><td>74.3-1.8%</td><td>15.5</td><td>12.1</td><td>13.6</td><td>37.4+16.1%</td><td>59.7+7.9%</td><td>7.4</td><td>11.4</td><td>6.2</td></tr><tr><td>SkillOS</td><td>63.6-16.0%</td><td>18.1</td><td>10.7</td><td>22.9</td><td>30.4-5.6%</td><td>49.7-10.3%</td><td>9.0</td><td>8.0</td><td>9.8</td></tr><tr><td>SKILLSANDBOX</td><td>80.0+5.7%</td><td>14.3</td><td>12.9</td><td>8.6</td><td>45.0+39.8%</td><td>62.2+12.2%</td><td>7.2</td><td>18.0</td><td>5.2</td></tr></table>

Implementation details. We use Qwen3.5-9B (Qwen Team, 2026), Gemini 3.1 Flash-Lite (Google DeepMind, 2026), and Qwen3.5-27B (Qwen Team, 2026) as executors within the ReAct framework (Yao et al., 2022b). The same model is used for execution and skill generation, and our scenario synthesis and verification also use the same model. All baselines use the same 100 source tasks to execute and generate skills, and each library is frozen before held-out evaluation. We evaluate on 140 tasks from the official ALFWorld test sets and 500 tasks from the official WebShop test sets, retrieving the top-3 skills for each task (Appendix E).

## 4.2 MAIN RESULTS

Unverified skills can degrade agent performance. As shown in Table 2, libraries admitted without our verification achieve lower SR than No Skill in most settings. This degradation is often accompanied by an increase in average steps, which means that the agent inefficiently spends more steps on redundant exploration yet solves fewer tasks. These results suggest that distilled skills can act as noise, impairing both task success and execution efficiency. Verifying their reusability before admission is therefore important for filtering out non-reusable skills.

Effective skill verification depends on the quality of execution evidence. SKILLSANDBOX consistently outperforms execution-based curation methods (ExpeL, ACE, and SkillOS) across all models and benchmarks, improving SR over the strongest of these baselines by up to 21.4%p. These results suggest that using execution feedback alone does not guarantee effective skill verification. ExpeL obtains evidence from repeated executions of the source task, whereas ACE and SkillOS use subsequent task stream. SKILLSANDBOX instead acquires the verification evidence from execution on the skill-specific scenarios. Section 5.1 validates the contribution of scenario synthesis by comparing with other execution-based evidence under the same rollout budget.

Reliable verification must account for both Recovery and Regression. On ALFWorld with Gemini, MemP and SKILLSANDBOX achieve the same Recovery, but SKILLSANDBOX has approximately 37% lower Regression. Consequently, MemP reduces SR relative to No Skill, whereas SKILLSANDBOX improves it. Reliable verification should therefore both help the agent solve previously failed tasks and preserve success on tasks it already solves.

Verification selects an effective subset of the original skill library. Compared with the unverified Vanilla Skill library, verification increases Recovery by up to 9.2%p and reduces or maintains Regression across all settings. The resulting library achieves the lowest Regression across all settings and the highest Recovery in all but one setting. It is the only library with higher Recovery than Regression in every setting, consistently improving SR over No Skill. Together, these results show that verification selects an effective subset of the original library.

Gains from verification are bounded by the reusable skills in the original library. On ALF-World, Gemini and Qwen(27B) already solve most held-out tasks without skills, which leaves little room for verification to improve SR, yet SKILLSANDBOX is the only library that does. On Web-Shop, where the executors solve far fewer tasks on their own, the verified library adds more than 10%p with lower Regression. Verification selects from the original library rather than improving the skills themselves, so it can add only what that library contains. Indeed, far fewer of the ALFWorld skills are helpful on held-out tasks (Section 5.2).

## 5 ANALYSIS

The main experiments show that the verified library improves downstream performance. We now examine which components of our verification framework contribute to this improvement and whether it reflects accurate assessment of skill reusability through the following research questions:

• RQ1: Does scenario synthesis improve the effectiveness of verification?

• RQ2: Does verification predict skill reusability?

• RQ3: Does reliable skill verification require a stronger model?

• RQ4: What do the synthesized scenarios reveal about skill reusability?

## 5.1 RQ1: DOES SCENARIO SYNTHESIS IMPROVE THE EFFECTIVENESS OF VERIFICATION?

To evaluate the contribution of synthesized scenarios, we compare verification evidence generated from synthesized scenarios, source tasks, and randomly selected tasks. We use Gemini 3.1 Flash-Lite on both benchmarks and keep the VERIFIER and rollout budget fixed, varying only the tasks on which evidence is generated. Under this budget, each skill is verified on 5 synthesized scenarios, 5 executions of its source task, or 5 random tasks. We evaluate verification accuracy against each skill’s effectiveness on held-out tasks, measured through paired executions with and without the skill (Appendix B). A skill is labeled helpful if it increases the success rate and harmful if it decreases.

Synthesized scenarios yield the highest verdict F1 and the best-performing skill library on both benchmarks (Figure 3), with held-out SR up to 15.7%p above the alternatives. On ALFWorld, the verdicts from source and random tasks barely separate helpful from harmful skills, and the libraries they retain fall below No Skill, with random-task verification even below the unverified library. Source tasks expose the skill-relevant situation only in its original context and random tasks rarely expose it at all, so neither provides the evidence that verification requires.

No Skill Unverified Source task  
![](images/7b16a964e488b9b46de87c704fb973b70f3f8894fcce1fa3fbdd94e959098c90.jpg)

![](images/ac8da02808e4ff0f9a24f947aae773af94aa95f95a3583a532eacd271ef524c0.jpg)  
Figure 3: Verification with evidence from different types of tasks: held-out SR (bars, left axis) and F1 of the verdicts against held-out skill effects (line, right axis).

## 5.2 RQ2: DOES VERIFICATION PREDICT SKILL REUSABILITY?

Table 3 reports precision, recall, F1, and specificity of the verdicts on both benchmarks. On WebShop, verification closely predicts the held-out utility of retained skills. High precision and specificity indicate that verification reliably retains helpful skills on held-out tasks while filtering out harmful ones. The 82.8% recall shows that verification preserves most helpful skills, although some are incorrectly rejected. Thus, on Web-Shop, a KEEP verdict provides a strong indication that the skill remians useful beyond its source experience.

Table 3: Verdict agreement with held-out skill effects.
<table><tr><td rowspan="2">PROPOSER</td><td colspan="4">Verdict agreement (%)</td></tr><tr><td>Prec. Rec.</td><td></td><td>F1</td><td>Spec.</td></tr><tr><td>WebShop</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Flash 96.0 82.8 88.9 GPT-5.6 Luna</td><td>95.2</td><td>69.0</td><td>80.0</td><td>94.4 94.4</td></tr><tr><td>ALFWorld</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Flash 39.1 81.8 52.9</td><td></td><td></td><td></td><td>60.0</td></tr><tr><td>GPT-5.6 Luna</td><td></td><td>34.8 72.747.1</td><td></td><td>57.1</td></tr></table>

Overall agreement is lower on ALFWorld, but verification still preserves the useful skills and improves the

composition of the library. Only 23.9% of the ALFWorld skills are helpful on held-out tasks, which lowers precision. Verification nevertheless retains 81.8% of the helpful skills, nearly the same recall as on WebShop, and helpful skills make up 39.1% of the verified library, which is higher than the unverified library. Verification thus selects a more useful subset from execution evidence even where helpful skills are scarce.

## 5.3 RQ3: DOES RELIABLE SKILL VERIFICATION REQUIRE A STRONGER MODEL?

Our framework targets self-evolving agents, which the quality of verification must therefore not depend on the capability of the model. To test this property, we replace only the PROPOSER with a stronger model (GPT-5.6 Luna (OpenAI, 2026)). The two PROPOSERs in Table 3 share the skills, the executor, the BUILDER, and the VERIFIER. Comparing their verdicts against the same held-out effects evaluates whether a stronger model is essential for reliable verification.

On both benchmarks, the executor’s own model matches or exceeds GPT-5.6 Luna in every metric and retains more of the helpful skills. The stronger PROPOSER is not more accurate, since it rejects more helpful skills but no more harmful ones. Reliable verdicts therefore do not depend on the intuitive judgment that only a stronger model could provide. This result follows from the division of scenario synthesis into planning and construction, since the LLM is used only for planning and rollouts, and construction and scoring are fixed procedures. Structured execution evidence thus lowers the dependence of verification on model capability, and a self-evolving agent can select useful skills from its own execution evidence without an external model stronger than itself.

## 5.4 RQ4: WHAT DO THE SYNTHESIZED SCENARIOS REVEAL ABOUT SKILL REUSABILITY?

Figure 4 analyze the skills’ reusability according to three verification criterion executability, utility, and efficiency (Section 3.2) across all models. Executability is the number of scenarios with at least one action assigned e = 1, indicating that the action implements the skill’s principle. Utility is the gain of the success rate between paired runs. Efficiency compares remaining action counts from the skill-used step to termination. In case of efficiency, we include only success pairs in both runs.

![](images/4a203bb36fb77917a03f512cac31937825e60e3a89a259003c633cb904c8db7f.jpg)  
Figure 4: Executability, utility, and efficiency across all executors and benchmarks.

On WebShop, agents take more actions on average with skills, even when both with and without-skill runs succeed. For Qwen3.5-27B and Gemini, skills improve task success but also add inefficiency on tasks that succeed without them. This indicates that since success gains alone are insufficient to assess skill reusability, verification must also account for the additional actions introduced by skill guidance. On ALFWorld, both Qwen models instead gain in success and save actions, showing that skill guidance can also yield successful execution. Gemini exhibits a different failure pattern. Despite high executability, skill lowers success and increases action counts, exposing a limitation in the benefit of the guidance beyond whether it can be followed.

## 6 RELATED WORK

Skill acquisition and curation. Agents acquire reusable skills by abstracting past executions into programs or workflows that can be retrieved and applied in subsequent tasks (Wang et al., 2023; 2025b;a). A complementary approach distills successful and failed experiences into textual insights, reasoning strategies, or procedural instructions that guide future decisions (Zhao et al., 2024; Ouyang et al., 2026b; Fang et al., 2026b). Feedback-driven curation provides an implicit form of verification, using subsequent execution outcomes to revise stored guidance or learn policies for managing the skill repository (Fang et al., 2026b; Zhang et al., 2026; Ouyang et al., 2026a). More explicit verification evaluates the effects of individual skills through source-task tests or paired runs with and without the skill, informing refinement and retention decisions (Wang et al., 2025a; Ma et al., 2026). However, these methods can leave skills insufficiently verified, as the verification evidence they provide are neither skill-relevant nor new. We identify the lack of reliable opportunities to evaluate each skill as a central verification problem.

Task and environment synthesis. Recent approaches synthesize tasks and environments to train LLM agents. Task generation expands training opportunities within available environments (Zhou et al., 2025; Sun et al., 2025; Qi et al., 2025; Zhou et al., 2026). Environment synthesis constructs the interaction setting itself through executable environments or LLM-based simulators (Chae et al., 2026; Fang et al., 2026a; Song et al., 2026; Chen et al., 2026b). Adaptive approaches use the learner’s performance to guide generation toward suitable difficulty levels or weaknesses in the current policy (Dennis et al., 2020; Zala et al., 2024; Guo et al., 2025). However, these objectives are insufficient for skill verification on their own, as they do not ensure that execution encounters the situation that the skill targets. To address the challenges in skill verification, we propose a framework that dynamically synthesizes tasks and environment for each target skill at verification time.

## 7 CONCLUSION

We introduce SKILLSANDBOX, a framework that verifies skill reusability by dynamically synthesizing tasks and environments that reproduce each skill’s applicable conditions in contexts beyond its source experience. The framework verifies each skill by having the agent solve the synthesized tasks in the synthesized environments, testing whether its guidance remains useful in these new sit uations. Across ALFWorld and WebShop, the verified libraries achieve the highest success rate and efficiency compared to other methods. Further analyses show that synthesized scenarios improve the quality of verification, and that effective verification is possible using the executor’s own model.

## AI USE STATEMENT

We used LLMs to assist with manuscript writing and polishing for clarity and readability, as well as with implementing code for our experiments. The authors reviewed all AI-assisted text and manually verified all AI-generated code before using it in the experiments. The authors take full responsibility for the final content of this work, including all AI-assisted text and code.

## REFERENCES

Huan ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Qihan Ren, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence, 2026. URL https://arxiv.org/abs/2507.21046.

Anthropic. Agent Skills. GitHub repository, 2025. URL https://github.com/ anthropics/skills. Accessed: 2026-09-26.

Hyungjoo Chae, Jungsoo Park, and Alan Ritter. Safe and scalable web agent learning via recreated websites. arXiv preprint arXiv:2603.10505, 2026.

Kunfeng Chen, Qihuang Zhong, Juhua Liu, and Bo Du. Skillcat: Contrastive assessment and topology-aware skill self-evolution for llm agents. arXiv preprint arXiv:2606.13317, 2026a.

Zhaorun Chen, Zhuokai Zhao, Kai Zhang, Bo Liu, Qi Qi, Yifan Wu, Tarun Kalluri, Xuefei Cao, Yuanhao Xiong, Haibo Tong, et al. Scaling agent learning via experience synthesis. In International Conference on Learning Representations, volume 2026, pp. 121394–121420, 2026b.

Michael Dennis, Natasha Jaques, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, and Sergey Levine. Emergent complexity and zero-shot transfer via unsupervised environment design. Advances in neural information processing systems, 33:13049–13061, 2020.

Gen Dong, Yanjie Gao, Liqun Li, Tianyin Xu, Yu Hua, and Fan Yang. Agent skills can be harmful: An empirical study of skill-induced failures in llm agents. arXiv preprint arXiv:2608.11888, 2026.

Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, Xinyu Wang, Xiaobin Wang, Liangcai Su, Zhen Zhang, et al. Towards general agentic intelligence via environment scaling. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 17610–17621, 2026a.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. In Findings of the Associationfor Computational Linguistics: ACL 2026, pp. 17490–17502, 2026b.

Srishti Gautam, Arjun Radhakrishna, and Sumit Gulwani. Skillaxe: Sharpening llm-authored agent skills through evaluation-guided self-refinement. arXiv preprint arXiv:2606.10546, 2026.

Google DeepMind. Gemini 3.1 Flash-Lite. Model card, March 2026. URL https: //deepmind.google/models/model-cards/gemini-3-1-flash-lite/. Accessed: 2026-09-26.

Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, and Mengdi Wang. Genenv: Difficulty-aligned co-evolution between llm agents and environment simulators. arXiv preprint arXiv:2512.19682, 2025.

Tingxu Han, Yi Zhang, Wei Song, Chunrong Fang, Zhenyu Chen, Youcheng Sun, and Lijie Hu. Sweskills-bench: Do agent skills actually help in real-world software engineering? arXiv preprint arXiv:2603.15401, 2026.

Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artif. Intell., 101:99–134, 1998. URL https://api. semanticscholar.org/CorpusID:5613003.

Xiangyi Li, Yimin Liu, Wenbo Chen, Bingran You, Zonglin Di, Yifeng He, Shenghan Zheng, Kyoung Whan Choe, Jiankai Sun, Shuyi Wang, et al. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026.

Sirui Liang, Pengfei Cao, Jian Zhao, Wenhao Teng, Xiangwen Liao, Jun Zhao, and Kang Liu. Learning how to remember: A meta-cognitive management method for structured and transferable agent memory. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Association for Computational Linguistics: ACL 2026, pp. 30733– 30753, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.1535. URL https: //aclanthology.org/2026.findings-acl.1535/.

Yuxuan Liu, Zhaochen Su, Lingyun Xie, Yuhao Zhang, Qing Zong, Jiahe Guo, Zhongwei Xie, Yiyan Ji, Yauwai Yim, Hongyu Luo, et al. Skillrevise: Improving llm-authored agent skills via trace-conditioned skill revision. arXiv preprint arXiv:2606.01139, 2026.

Yuchen Ma, Yue Huang, Han Bao, Haomin Zhuang, Swadheen Shukla, Michel Galley, Xiangliang Zhang, and Stefan Feuerriegel. Skillgen: Verified inference-time agent skill synthesis. arXiv preprint arXiv:2605.10999, 2026.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. Trace2skill: Distill trajectory-local lessons into transferable agent skills. arXiv preprint arXiv:2603.25158, 2026.

OpenAI. GPT-5.6, 2026. URL https://deploymentsafety.openai.com/gpt-5-6/ gpt-5-6.pdf.

Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, et al. Skillos: Learning skill curation for self-evolving agents. arXiv preprint arXiv:2605.06614, 2026a.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. In International Conference on Learning Representations, volume 2026, pp. 94327–94354, 2026b.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. In International Conference on Learning Representations, volume 2025, pp. 79791–79821, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and Matthew´ Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 8326–8357, 2026.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 5555–5579, 2025.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. Inducing programmatic skills for agentic tasks. arXiv preprint arXiv:2504.06821, 2025a.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 63897–63911. PMLR, 2025b. URL https://proceedings.mlr.press/v267/wang25bx.html.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.

Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Zirui Liu, Jiliang Tang, Himabindu Lakkaraju, and Zhen Xiang. How memory management impacts LLM agents: An empirical study of experiencefollowing behavior. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 623–645, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.27. URL https://aclanthology.org/2026.acl-long.27/.

Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Jia Xu, Botian Shi, Yu Qiao, and Haifeng Li. Towards self-evolving agents: Enabling autonomy through interactive experience refinement. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Association for Computational Linguistics: ACL 2026, pp. 30424–30451, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395- 1. doi: 10.18653/v1/2026.findings-acl.1522. URL https://aclanthology.org/2026. findings-acl.1522/.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022a.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022b.

Abhay Zala, Jaemin Cho, Han Lin, Jaehong Yoon, and Mohit Bansal. Envgen: Generating and adapting environments via llms for training embodied agents. arXiv preprint arXiv:2403.12014, 2024.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations, volume 2026, pp. 86069–86100, 2026.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024.

Yifei Zhou, Qianlan Yang, Kaixiang Lin, Min Bai, Xiong Zhou, Yu-Xiong Wang, Sergey Levine, and Li Erran Li. Proposer-agent-evaluator (pae): Autonomous skill discovery for foundation model internet agents. In Forty-second International Conference on Machine Learning, 2025.

Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, and Sainbayar Sukhbaatar. Self-challenging language model agents. Advances in Neural Information Processing Systems, 38:113959–113991, 2026.

A Baseline Details 14   
B Implementation Details 14   
C Scenario Synthesis 15   
D Case Study 16   
E Additional Results 17   
F Prompts 19   
F.1 Executor . 19   
F.2 Skill distillation 20   
F.3 PROPOSER 20

## A BASELINE DETAILS

We distinguish no verification (unconditionally retaining generated skills) from execution-based verification, which includes feedback-driven skill admission, revision, removal, or reweighting. The latter uses either source replay (repeated attempts at the original task) or downstream tasks (subsequent source tasks selected independently of a skill). All libraries are built from the same 100 source tasks and frozen before held-out evaluation.

Vanilla Skill. No verification. Each source trajectory is distilled into an applicability condition and an actionable principle using the skill distillation prompts in Appendix F.2. All generated skills are retained; SKILLSANDBOX verifies skills from this same initial pool.

ReasoningBank. No verification. ReasoningBank sequentially solves source tasks using prior memories, judges each outcome, and extracts up to three structured memories from each trajectory. Our basic variant, without MaTTS, appends all extracted items without revising or deleting existing ones: the judge assesses task success, not memory utility.

MemP. No verification. We use MemP’s Script representation with Vanilla updates. The executor sequentially solves source tasks using retrieved scripts; after each task, a builder appends one workflow summary regardless of success, without utility-based filtering, revision, or deletion.

ExpeL. Execution-based verification: source replay. ExpeL collects source trajectories before extracting insights. After failure, our executor reflects and retries the same task at most once. Using the same model, the extractor compares successful and failed attempts and groups of successes, curating rules through ADD, EDIT, AGREE, and REMOVE with importance counters.

ACE. Execution-based verification: downstream tasks. ACE solves tasks with an evolving playbook, reflects on entry helpfulness, and incrementally curates the playbook. Our implementation makes three source passes, adding entries and updating feedback counters, with separate duplicate merging and budget-triggered pruning of harmful entries. Held-out evaluation uses the entire frozen playbook rather than top-three retrieval.

SkillOS. Execution-based verification: downstream tasks. We use training-free SkillOSbase (Ouyang et al., 2026a), with the paper’s Appendix A prompts. The executor makes one ordered source pass, retrieving skills from the evolving repository with BM25. After each task, the curator uses its trajectory, outcome, and retrieved skills to insert, update, or delete skills before the next task.

## B IMPLEMENTATION DETAILS

Executor. Table 4 summarizes the executor settings for the two benchmarks. At each step, the executor receives the current observation and admissible actions together with the recent interaction history. The executor prompts are provided in Appendix F.1.

Table 4: Executor settings. History length counts preceding observation–action pairs; the current observation is supplied separately.
<table><tr><td>Setting</td><td>ALFWorld</td><td>WebShop</td></tr><tr><td>Action history length</td><td>10</td><td>5</td></tr><tr><td>History content</td><td>Observation-action pairs</td><td>Observation-action pairs</td></tr><tr><td>Maximum steps per episode</td><td>30</td><td>15</td></tr></table>

Skill distillation and retrieval. We distill each source trajectory into an applicability condition and an actionable principle (Appendix F.2). On both benchmarks, held-out retrieval ranks full skill texts against the task query by cosine similarity using text-embedding-3-small, supplying the top three skills in descending order.

For Question I (Section 2), GPT-5.6 Luna extracts fixed conditions from When $_ { \mathrm { t o } } ~ \mathsf { a p p l y }$ , consulting ${ \tt P r i n c i p l e }$ for context, then checks them against the task objective and with-skill trajectory. OBSERVED requires evidence for all conditions and their logical and temporal relationships; state and history conditions require trajectory evidence, not task outcomes or claimed skill use. Otherwise, the label is NOT OBSERVED. We report the mean number of OBSERVED tasks per skill across 50 skills, each evaluated on 10 retrieved tasks.

For Question II, Gemini 3.1 Flash-Lite executes a fixed random stream of 500 held-out WebShop tasks using the same 50-skill library, with three repeats per setting: Full Library, Top-3, and Top-5 Retrieval. Retrieval uses the embedding method above. The ReAct executor cites skill identifiers when using them; at each stream prefix, we report the percentage of skills cited in at least one or five distinct tasks.

VERIFIER. The VERIFIER compares trajectories $\tau ^ { + }$ and $\tau ^ { - }$ produced by the same executor in the same synthesized scenario, with and without skill $k ,$ respectively. We align the full trajectories by maximizing the number of pairs with identical pre-action observations, preserving temporal order and using each step at most once. Ties are resolved by choosing the lexicographically earliest sequence of paired step indices, with the $\tau ^ { + }$ index first. For each distinct observation, we retain only the earliest aligned pair whose step in $\tau ^ { + }$ is marked as using skill k. From the same observation, we examine how using the skill changes the agent’s subsequent behavior compared with execution without the skill.

For each retained pair, let $t ^ { \pm }$ denote the matched step indices and $T ^ { \pm }$ the termination indices of the corresponding trajectories. The remaining action counts $T ^ { \pm } - t ^ { \pm }$ include all subsequent actions in both trajectories, including those taken when the same observation recurs. To assess utility, we compare terminal rewards $r ^ { \pm } \in \{ 0 , 1 \}$ , which are 1 for task success and 0 for failure under the scenario’s success criterion. To account for efficiency, we discount these rewards back to the matched steps so that successful continuations requiring fewer actions receive higher returns:

$$
\Delta u = r ^ { + } \gamma ^ { T ^ { + } - t ^ { + } } - r ^ { - } \gamma ^ { T ^ { - } - t ^ { - } } , \qquad 0 < \gamma < 1 .\tag{2}
$$

Executability concerns whether the principle can be executed as admissible actions supported by the environment. The VERIFIER uses agreement between the principle and an action in $\tau ^ { + }$ as evidence of this feasibility at each matched point. It sets $e = 1$ when the action implements the corresponding instruction and $e = 0$ when this evidence is absent. Only points with observed evidence of executability contribute to $R ( k )$ . We first average $e \Delta u$ over the retained pairs within each trajectory pair, then across synthesized scenarios. This empirical averaging gives the reusability score and KEEP/REJECT verdict in Equation 1.

Evaluation protocol. For RQ1 (Section 5.2), we execute the 500 held-out WebShop tasks with and without each of the 50 evaluated skills. Let ${ \mathrm { H e l p } } _ { k }$ denote the number of held-out tasks that skill k turns from failure to success and Harm the number it turns from success to failure; the effectiveness of the skill is calculated $E _ { k } = ( \mathrm { H e l p } _ { k } - \mathrm { H a r m } _ { k } ) / 5 0 0$ . Spearman’s $\rho ,$ with average ranks assigned to ties, and Pearson’s r are computed between $E _ { k }$ and the reusability score $R ( \bar { k } )$ assigned by the verification pipeline over all 50 skills. For the agreement metrics, a skill is labeled helpful when $\mathrm { H e l p } _ { k } > \mathrm { H a r m } _ { k }$ and harmful when $\mathrm { H e l p } _ { k } < \mathrm { H a r } \mathbf { \bar { m } } _ { k }$

## C SCENARIO SYNTHESIS

Synthesis Types The synthesis typology of Section $5 . 4$ covers scenarios generated for WebShop and ALFWorld skills in the Qwen3.5-27B setting. For each environment, GPT-5.6 Luna first sum marizes each scenario’s operations and implementation plan in one or two sentences and induces types from these summaries. To categorize how skills cause harm, we select the scenarios, in which the executor fails with the skill, succeeds without it, and receives a negative scenario score. Given the skill, both trajectories, and their outcomes, the same model explains how the skill could have contributed to the failure, and the explanations are grouped into types in the same way. Figure 5 reports the share of scenarios in each synthesis type; types are not exclusive, so shares sum to more than 100%.

![](images/0ae877e80e02e2bb723170816500522d60405709c79fe15ef4add62a1a6eb5a6.jpg)

![](images/c5f186124cbb877bdb33b642f9695429bc90cd204cd4f6ccc07acebf1c238ccb.jpg)  
Figure 5: Planned synthesis mechanisms for Qwen3.5-27B on (a) WebShop and (b) ALFWorld. Each bar shows the percentage of scenarios using that transformation. A scenario may involve multiple transformations.

## D CASE STUDY

Table 5: Case study of rejected skills. The shaded block is the skill under verification, and the red bold text marks the part of its principle that causes the rejection. $\tau ^ { + }$ and $\tau ^ { - }$ denote the executions of the same synthesized scenario with and without the skill, and the highlighted action is the one guided by the skill.

<table><tr><td colspan="3">ALFWorld Verdict: REJECT</td></tr><tr><td colspan="3">Task Examine the tissuebox with the desklamp</td></tr><tr><td colspan="3">condition: The task requires using a stationary appliance on a movable object located in a different area. principle: Locate and activate the stationary tool first, retrieve the target and transport it to the tool&#x27;s location</td></tr><tr><td>Synthesized Scenario Task</td><td>Execution Result</td><td>τ − : Success, 6 steps</td></tr><tr><td>put a cool bread in countertop Environment • bread on the diningtable, not cooled • fridge in a different area, closed</td><td>go to fridge 2 open fridge 3 close fridge 4 go to cabinet</td><td>1 take bread from diningtable 2 go to fridge 3 open fridge 4 cool bread with fridge√ 5 go to countertop</td></tr><tr><td colspan="3">picked up immediately never returns for the bread WebShop Verdict: REJECT</td></tr><tr><td colspan="3">Task I need some hair drying towels that are easy to clean, and price lower than 50.00 dollars</td></tr><tr><td colspan="3">condition: Evaluate a product for an easy-to-clean requirement after confirming its price and category criteria. principle: Verify the claim in both Features and Description before purchase.</td></tr><tr><td>Synthesized Scenario</td><td>Execution Result</td><td></td></tr><tr><td>Task I want to purchase a hairdressing backpack with an</td><td>τ+: Fail, 15 steps 4</td><td>τ−: Success, 4 steps</td></tr><tr><td>easy clean feature for under 50. Environment</td><td>click[Description] 5 click[Back to Search]</td><td>4 click[Buy Now]√</td></tr><tr><td>• required attribute easy clean, price at most $50, product kind hairdressing backpack • search query hairdressing backpacks easy to</td><td>6 search[same query] 7–15: reopens the product, alternates the two sections;</td><td></td></tr></table>

Table 5 illustrates how synthesized scenarios expose the skill-relevant situation and use it as verification evidence.

ALFWorld. In the source task, the agent successfully examines a tissue box with a desk lamp by activating the lamp before retrieving the tissue box from another location. This ordering work because the lamp can be activated independently and remains on while the agent retrieves the ob ject. The distilled skill generalizes this source-specific ordering into a broader principle: activate a stationary appliance before retrieving the target object. To reproduce the skill’s condition, the synthesized scenario places a stationary appliance (fridge) and a movable object (bread) at different locations. The Builder places the uncool bread at the agent’s starting location and keeps the fridge closed elsewhere, requiring the agent to cool the bread before placing it on a countertop. Unlike the lamp, cooling requires the agent to hold the bread at the fridge. Nevertheless, $\tau ^ { + }$ follows the skill and goes to the fridge first, then fails to return for the bread and reaches the step limit. In contrast, $\tau ^ { - }$ takes the bread first, carries it to the fridge, cools it, and completes the task in six steps. The scenario thus shows that the principle distilled from the source does not transfer to this new context, leading the verifier to reject the skill.

WebShop. The source task asks for an easy-to-clean hair-drying towel under \$50. During the source experience, the agent finds a candidate whose cleaning information appears in both the Features and Description sections, motivating a skill that requires checking both before purchase. For verification, the synthesized task instead asks for an easy-to-clean hairdressing backpack under \$50 and places a qualifying product at rank 3 of the search results. This creates a new instance of the same product-verification situation, but one in which sufficient evidence is already available from a single section. After inspecting the Features, $\tau ^ { - }$ treats the requirement as satisfied and purchases the product in four steps. $\tau ^ { + }$ , however, follows the principle and continues to seek confirmation from the Description. Because the same claim is absent there, it repeatedly revisits the product sections and exhausts the action budget without purchasing. The synthesized scenario therefore exposes that requiring evidence from both sections is not generally necessary, but reflects the information layout encountered in the source task, leading the verifier to reject the skill.

## E ADDITIONAL RESULTS

WebShop score correlation. Table 6 supplements RQ1 with the correlations between the verification score R(k) and the net held-out effect $E _ { k }$ on WebShop, using the evaluation protocol in Appendix B. Both PROPOSERs have positive Spearman correlations, indicating that skills with higher verification scores tend to have larger held-out effects. Pearson’s r provides a supplementary check of a linear relation; it does not establish that the score is calibrated to the magnitude of the improvement.

Table 6: WebShop score correlation with held-out skill effects. $\rho$ and r denote Spearman and Pearson correlations.
<table><tr><td>PROPOSER</td><td>Spearman  $\rho$ </td><td>Pearson r</td></tr><tr><td>Gemini 3.1 Flash</td><td>0.665</td><td>0.552</td></tr><tr><td>GPT-5.6 Luna</td><td>0.662</td><td>0.645</td></tr></table>

Retrieval size. Table 7 examines how the number of retrieved skills affects downstream task success. We vary the retrieval size $k \in \{ 1 , 3 , 5 \}$ with Gemini 3.1 Flash-Lite as the executor, while keeping the KEEP/REJECT verdicts, the retained skill pool, the held-out tasks, and the executor settings fixed. For each task, we rank the retained skills by cosine similarity between embeddings of the task instruction and the full skill text, using the same embedding model as in the main experiments. The top-k skills are injected into the executor’s prompt. We set $k = 3$ in all other experiments because it achieves the best SR on both benchmarks, tying with $k = 5$ on ALFWorld and outperforming both $k = 1$ and $k = 5$ on WebShop.

Causes of rejection. Figure 6 shows the primary failure causes of the rejected skills according to the verification criterion for each executor. A rejected skill can fail several criteria; within each

Eficiency

Table 7: Held-out SR (%) of Gemini 3.1 Flash-Lite under different retrieval sizes k, with the number of solved tasks in parentheses. $k = 3$ is the setting of all other experiments. Bold marks the best value in each row.
<table><tr><td>Benchmark</td><td> $k = 1$ </td><td> $k = 3$ </td><td> $k = 5$ </td></tr><tr><td>ALFWorld (140 tasks)</td><td> $5 8 . 6 ( 8 2 ) $ </td><td>60.0 (84)</td><td>60.0 (84)</td></tr><tr><td>WebShop (500 tasks)</td><td> $3 1 . 4 ( 1 5 7 )$ </td><td>36.2 (181)</td><td>35.0 (175)</td></tr></table>

criterion, one primary cause is assigned per skill, and the three largest cause types are shown with the rest pooled as Other.

![](images/9dfd48075312a15b03fabec9e80837076bcc9f4864218ba980030461fedf110e.jpg)

![](images/19f80b068337257108d874e4d2aa470511c3a421a6c84178db5c558f2b0051f2.jpg)

![](images/8164f18e33841d60c4533899aa4cf34c78b802e7f16d3cae193b0e9d546342b0.jpg)

![](images/9ca3a4c168b7d4bd0951ccf5a3c4a10c4fe2df4cfb5df1dcd098d516af0f13a7.jpg)

![](images/3c52f6b67aefb3f0b9e2b5f2c7eddaed04b83aa1f2eb8ea341f080a3f75e6ba1.jpg)

![](images/c87dc90fd654240022225179a1c69539ec6c93d7ea79ec3d6ab6b80479c3e23a.jpg)

![](images/e9bb0440b75ce35b4e1cd19986f3d4ee45086725a5438aeab0f0574d7416e3e2.jpg)  
Figure 6: Primary failure cause of rejected skills according to the verification criterion. Each rejected skill and its scenario trajectories are analyzed and classified by GPT-5.6 Luna. The three largest types of each criterion are shown and the rest are pooled as Other.

## F PROMPTS

We present prompt templates for execution and skill distillation, and structured summaries of the scenario-proposal prompts. Braced placeholders denote runtime inputs, such as a task, trajectory, skill, or environment interface. The proposal summaries describe the inputs and output schemas for each stage; retry and format-repair messages are omitted.

## F.1 EXECUTOR

The following prompts execute tasks without skills, using the current task, observation, admissible actions, and recent action history.

Executor: ALFWorld   
Follow the user’s ALFRED task instructions.   
Return only JSON with fields in this order: {"rationale": "step-by-step   
reasoning", "action": "one exact admissible action"}.   
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to:   
{task}   
Below are the most recent {history length} observations and the corresponding actions   
you took: {action history}   
Your current observation is: {current observation}   
Your admissible actions of the current situation are: {admissible actions}.   
Now it’s your turn to take an action. You should first reason step-by-step about the current   
situation. Once you’ve finished your reasoning, choose one admissible action for the current   
step.   
Return only JSON: {"rationale": "step-by-step reasoning",   
"action": "one exact admissible action"}.

## Executor: WebShop

You are an expert autonomous agent operating in the WebShop e-commerce environment.   
Your task is to: {task description}.   
Below are your most recent {history length} interaction steps (observations when   
provided, and actions):   
{action history}   
You are now at step {current step} out of {total steps} steps, and your current   
observation is: {current observation}   
Your admissible actions of the current situation are: [   
{available actions} ].   
Now it’s your turn to take one action for the current step. You should first reason step-by-step   
about the current situation, then think carefully which admissible action best advances the   
shopping goal.   
- Thought: Reasoning process   
- Action: Return exactly one executable admissible command for current step   
Return only a valid JSON object with exactly these two fields, using rationale for   
Thought and action for Action: {"rationale": "the complete Thought   
described above", "action": "the single executable Action   
described above"}

## F.2 SKILL DISTILLATION

These prompts generate one skill from the source trajectory, before verification or revision. The trajectory input includes the source task, observed transitions, recorded actions, and execution outcome. The output consists of a When to apply condition and a Principle in one or two sentences. We distinguish prompts for successful and failed source trajectories.

## Skill distillation: ALFWorld failure

You are an expert at extracting reusable skills from agent failures. Analyze the failed trajectory from an AI agent operating in the ALFRED embodied household environment (ALFWorld).

Trajectory: {trajectory}

Generate one concise skill that captures the reusable lesson from this failure.

\- When to apply: Specific trigger condition.

\- Principle: The core actionable insight in 1-2 sentences.

## Skill distillation: ALFWorld success

You are an expert at extracting reusable skills from agent successes. Analyze the successful trajectory from an AI agent operating in the ALFRED embodied household environment (ALFWorld).

Trajectory: {trajectory}

Generate one concise skill that captures the reusable lesson from this success.

\- When to apply: Specific trigger condition.

\- Principle: The core actionable insight in 1-2 sentences.

## Skill distillation: WebShop failure

You are an expert at extracting reusable skills from agent failures. Analyze the failed trajectory from an AI agent operating in an online shopping environment (WebShop).

Trajectory: {Trajectory}

Generate one concise skill that captures the reusable lesson from this failure.

\- When to apply: Specific trigger condition.

\- Principle: The core actionable insight in 1-2 sentences.

## Skill distillation: WebShop success

You are an expert at extracting reusable skills from agent successes. Analyze the successful trajectory from an AI agent operating in an online shopping environment (WebShop).

Trajectory: {Trajectory}

Generate one concise skill that captures the reusable lesson from this success.

\- When to apply: Specific trigger condition.

\- Principle: The core actionable insight in 1-2 sentences.

## F.3 PROPOSER

For brevity, we summarize the proposal prompts by their inputs, core instructions, and required JSON outputs. Each summary combines condition interpretation and scenario specification for one environment.

## Proposer: ALFWorld

## Input

<skill condition>   
<source task>   
<source trajectory summary>   
<environment interface>   
<skill>   
<interpreted requirements>   
<compatible or fixed goals>   
<earlier scenario summaries>   
<selected test specification> (optional)

## Core instructions

1. Preserve the full condition, including alternatives, negation, quantities, and prior events. Separate task objectives from the current state and execution history; do not derive applicability requirements from the principle.

2. Determine whether the condition is established by the initial state, real setup history, or the executor’s own rollout. Leave exploration and manipulation to the executor when the condition arises during execution.

3. Specify symbolic roles and supported construction operations. Vary objects, placements, and compatible task context while preserving required capabilities and any fixed goal; concrete entities are bound later.

4. Keep the goal and the skill-guided decision unfinished. Do not invent observations or capabilities, pre-execute the principle, or design the scenario to guarantee a favorable verdict.

## Output

Condition interpretation: A JSON object with status, encounter mode, encounter reason, history required, and task checks.

Scenario specification: One symbolic program with roles, operations, prepare start, setup events, condition checks, test checks, and novelty; include goals only when no fixed goal is supplied. Condition checks concern the encounter, while test checks concern the initial configuration.

## Proposer: WebShop

## Input

<skill condition>   
<environment interface>   
<skill>   
<interpreted requirements>   
<fixed target product>   
<linked human task>   
<scoring conditions>   
<earlier scenario summaries>   
<response schema>   
<frozen design>   
<operator definitions>

## Core instructions

1. Restate every condition clause without adding restrictions from the principle. Preserve alternatives, history, and local absence claims; report a missing native capability rather than weaken the condition.

## Proposer: WebShop (continued)

2. Keep the fixed target’s identity and product type. Preserve the linked task and scoring conditions except for changes needed to reproduce the condition; use existing product facts whenever sufficient.

3. Specify minimal supported edits to product records, options, or result lists. Start at the native search screen with no preselected options, and use one search query derived from the final task. Leave inspections and selections to the executor.

4. Translate the frozen design into symbolic operations and identify which requirements they establish. Do not fabricate observed evidence, pre-execute the principle, or guarantee that the skill helps; an empty operation list is valid when no edits are needed.

## Output

Condition interpretation: A JSON object containing status, scoped requirements (task, catalog, or trajectory), and rationale; if unsupported, return status and reason.

Scenario design: A JSON object with mode, intervention, goal, wording, and novelty. Goal fields express requirements for subsequent binding.

Operation specification: A JSON object with value declarations, operations, and coverage, linking requirements to edits and the intended evidence locations.