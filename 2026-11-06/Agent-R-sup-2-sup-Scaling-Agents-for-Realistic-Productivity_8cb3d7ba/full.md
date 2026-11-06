# Agent-R<sup>2</sup>: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis

Teng Pan<sup>1,2</sup>, Zixuan Wang<sup>1,2</sup>, Yuchen Yan<sup>1</sup>, Qifei Wu<sup>2</sup>, Yao Chen<sup>2</sup>, Yingqi Qu<sup>2</sup>, Ruiqing Zhang<sup>2</sup>, Yongliang Shen<sup>1,†</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>Baidu Inc.

## Abstract

Large language models have made rapid progress in agentic reasoning and tool use, raising expectations for agents that can take over realistic productivity work from humans. However, current agents still struggle to finish such tasks, where success requires following long-horizon professional workflows, understanding heterogeneous file contexts, and producing usable deliverables under fine-grained requirements. A key bottleneck is the lack of scalable post-training data that reflects real work intent, authentic file environments, and deliverable-level evaluation. To close this gap, we introduce Agent-R<sup>2</sup>, a real-source data synthesis framework that scales training data for realistic productivity work. Instead of relying on manually specified workflows or LLM-generated files, Agent-R<sup>2</sup> grounds synthesis throughout the pipeline in real occupational seeds, corpus-derived workflow evidence, and authentic web-sourced files. From 10k web-corpus candidates, Agent-R<sup>2</sup> distills 4k task instances that pair real multi-format attachments with fine-grained, deliverable-level rubrics. On this data, we train Agent-R<sup>2</sup>-4B via supervised fine-tuning and rubric-based reinforcement learning, improving both endto-end productivity performance and file-grounded reasoning. Trained on only 4k instances, Agent-R<sup>2</sup>-4B surpasses Qwen3-235B-A22B-Thinking-2507 on both GDPval (51.0% vs. 48.0%) and OficeQA (14.8% vs. 13.8%). Further analysis shows that Agent-R<sup>2</sup> mitigates shortcut behavior, where agents produce superficially plausible deliverables before sufficiently grounding, constructing, or verifying them, while promoting sustained exploration.

## 1 Introduction

Large language models have made remarkable progress in agentic reasoning and tool use, enabling agents to plan, interact with environments, and solve increasingly open-ended tasks (Yao et al. 2023; Schick et al. 2023; Zhou et al. 2024). These advances raise the expectation that agents can take over realistic productivity work from humans. Such work is economically valuable but demanding: it involves longhorizon professional workflows, where agents must understand professional contexts, gather and integrate evidence from heterogeneous digital artifacts, invoke the necessary tools, and ultimately deliver usable work products such as analytical reports, spreadsheets, presentations, and structured documents.

![](images/294571ca340de0b7e89df8f46f5c160de576940e61eff803d07e275093eab79f.jpg)  
Figure 1: GDPval Pass@1 comparison. Agent-R<sup>2</sup>-4B improves the Qwen3-4B-Thinking base model by 18.6 points and surpasses several larger open-source baselines.

However, current agents still fall short on such work. First, real professional workflows involve interdependent steps, implicit domain conventions, and evolving intermediate states, so models miss critical operations or fail to adapt based on feedback. Second, real file contexts are heterogeneous and messy: models must integrate evidence across artifacts with diferent formats, layouts, and levels of completeness, and often misread content or overlook cross-file dependencies. Third, real deliverables must satisfy fine-grained requirements over factual consistency, evidence coverage, structure, formatting, and practical usability; producing an output file is not the same as meeting deliverable standards. Mastering all three demands at once is precisely what current training data fails to teach.

Existing training data targets these capabilities only in isolation: instruction following (Xu et al. 2025a; Guo et al. 2025), multidisciplinary reasoning (Liu et al. 2025), document-grounded question answering (Tanaka et al. 2023; Zhang et al. 2026a), or generation of a single deliverable type such as presentations and spreadsheets (Zheng et al. 2025, 2026; Chi et al. 2026). While efective in their respective settings, these data sources do not provide end-to-end supervision that jointly covers real work objectives, cross-file evidence, professional workflow execution, and deliverable constraints. What is missing, we argue, is not more data but data that reflects real work: real intents, authentic files, and deliverable-level standards.

Code is available at https://github.com/ZJU-REAL/Agent-R2

Such data is hard to fabricate: manually authored workflows do not scale, and LLM-generated files lack the layouts, gaps, and cross-file references of real documents. To address this data bottleneck, we introduce $\mathrm { \bf A g e n t { - } R ^ { 2 } }$ , a real-source data synthesis framework that grounds every stage of synthesis in real sources through a three-stage pipeline. First, rather than relying on manually authored workflows as in recent benchmark construction (Patwardhan et al. 2025; Chen et al. 2026), Agent- $\cdot \mathbf { R } ^ { 2 }$ mines file-oriented workflow blueprints from web corpora using occupational activity seeds, improving both realism and scalability. Second, it searches for and verifies public files that support each workflow, retaining the heterogeneous formats, layouts, incomplete structures, and cross-file dependencies of realistic digital work. Third, it synthesizes user instructions and fine-grained rubrics from the verified workflow-file contexts: the instructions require complex deliverables involving evidence gathering, reasoning, file operations, and artifact construction, while the rubrics specify detailed criteria for evidence grounding, analytical correctness, and deliverable quality. Starting from 10k web-corpus candidates, Agent- $\cdot \mathbf { R } ^ { 2 ^ { \frac { i } { } } }$ produces 4k highquality training instances.

We train $\mathrm { A g e n t - R ^ { 2 } - 4 B }$ on the synthesized data using supervised fine-tuning followed by rubric-based reinforcement learning. The reinforcement learning stage optimizes the model in the persistent sandbox with outcome rewards: each generated deliverable is evaluated against its fine-grained rubric, which encourages the model to satisfy evidence, reasoning, and artifact-quality requirements rather than merely produce an output file. Trained on only 4k synthesized instances, Agent- $\cdot \mathbf { R } ^ { 2 }$ -4B reaches 51.0% Pass@1 on the full GDPval set (Figure 1), improving the base model by 18.6 points and surpassing Qwen3-235B-A22B-Thinking-2507 (48.0%) and Qwen3-VL-235B-A22B-Thinking (49.7%).

The gains transfer to document-grounded reasoning on OficeQA: Agent-R<sup>2</sup>-4B improves the base model from 2.2% to 14.8% Pass@1 and again surpasses Qwen3-235B-A22B-Thinking-2507 (13.8%). Behavioral analysis further shows that training on rubric-scored real-source data mitigates shortcut behavior, a general failure mode where agents emit superficially plausible deliverables before suficiently grounding, constructing, or verifying them, while promoting sustained exploration.

Our contributions are summarized as follows:

• We introduce $\mathrm { \ A g e n t { - } R ^ { 2 } }$ , a data synthesis framework that grounds workflow extraction, file collection, and instruction-rubric synthesis in real sources, scaling training data for realistic productivity work.

• We synthesize 4k high-quality productivity-work instances with authentic multi-format attachments and finegrained rubrics, and train Agent-R<sup>2</sup>-4B with SFT and rubric-based RL.

• Agent-R2-4B surpasses Qwen3-235B-A22B-Thinking-2507 on GDPval and OficeQA; further analysis indicates that the training mitigates shortcut behavior while promoting sustained exploration.

## 2 Problem Formulation

We study realistic productivity tasks, in which an agent must carry out long-horizon professional work in a sandbox environment with general tools, heterogeneous input files, and deliverable-level requirements.

Task. A task instance is denoted as

$$
x = ( I , { \mathcal { F } } _ { a } , { \mathcal { F } } _ { r } , { \mathcal { R } } ) ,
$$

where I is the user instruction, ${ \mathcal { F } } _ { a }$ is the attachment set available during solving, $\mathcal { F } _ { r }$ is an optional reference set used only for evaluation, and R is the task-specific rubric. When no reference files are provided, $\mathcal { F } _ { r } ~ = ~ \emptyset$ . The instruction I specifies the required deliverables, which the agent produces as a set of generated files $\mathcal { F } _ { d }$

Environment. Both solving and evaluation use a persistent sandbox environment

$$
\mathcal { E } = ( \mathcal { S } , \mathcal { T } , \delta , \Omega ) ,
$$

where $s$ is the state space, $\tau$ is the tool set, δ is the transition function, and Ω is the observation function. The tool set is

$$
\mathcal { T } = \{ b a s h , w e b \_ s e a r c h , w e b \_ f e t c h , \nu i e w \_ i m a g e \} .
$$

The bash tool executes shell commands and file operations inside the sandbox. The web\_search and web\_fetch tools are used to locate relevant webpages and retrieve the content of a specified webpage, respectively. The view\_image tool provides visual access to images or rendered file pages, and its return depends on whether the solver has multimodal capability. If so, the visual observation is returned directly; otherwise, the tool requires a textual visual query q and invokes an auxiliary vision policy $\pi _ { v }$ to return the result:

$$
o ^ { v } = \left\{ \begin{array} { l l } { { m , } } & { { \mathrm { w i t h ~ m u l t i m o d a l ~ c a p a b i l i t y , } \nonumber } } \\ { { \pi _ { v } ( m , q ) , } } & { { \mathrm { w i t h o u t ~ m u l t i m o d a l ~ c a p a b i l i t y , } } } \end{array} \right.
$$

where $o ^ { v }$ denotes the tool’s return, m is an image, and $q$ is the textual visual query used only when an auxiliary vision policy is needed.

Solving and Evaluation. During solving, the sandbox is initialized with the attachment files ${ \mathcal { F } } _ { a }$ . The solver’s goal is to generate the deliverable files $\mathcal { F } _ { d }$ requested by I. At turn t, the solver receives an observation, produces an action conditioned on the instruction and the interaction history, and the environment transitions accordingly:

$$
o _ { t } = \Omega ( s _ { t } ) , \quad a _ { t } \sim \pi _ { s } ( \cdot \mid I , o _ { \leq t } , a _ { < t } ) , \quad s _ { t + 1 } = \delta ( s _ { t } , a _ { t } ) .
$$

Evaluation uses a separate sandbox initialized with the generated deliverable files $\mathcal { F } _ { d }$ and, when available, the reference files ${ \mathcal { F } } _ { r }$ . The evaluator policy $\pi _ { e }$ interacts with E to judge the deliverables under the rubric R and outputs the final score.

## 3 Data Synthesis

## 3.1 Pipeline

In this section, we detail the data synthesis pipeline of $\mathrm { \bf A g e n t { - } R ^ { 2 } }$ . The overall pipeline is illustrated in Figure 2. The prompts used in each stage are provided in the supplementary material.

![](images/652d5de234727b310dc3c038a8a006f18949a2d7fd91ee4b455f687f243f6b50.jpg)  
Figure 2: Agent-R<sup>2</sup> data synthesis pipeline. The framework extracts file-oriented workflow blueprints from real web-corpus evidence, retrieves and verifies public attachments, and synthesizes instructions, rubrics, and references in sandbox.

Workflow Extraction. Prior work observes that raw web text can encode implicit tool-use workflows (Xu et al. 2026b). Inspired by this observation, we further treat web corpora as a natural source of workflow evidence for realistic productivity work and focus on file-oriented workflows, where completing the work requires locating, reading, editing, or producing file artifacts. By leveraging broad web corpora, Agent-R<sup>2</sup> enables scalable workflow construction and avoids the limited scalability of manually curated workflows (Patwardhan et al. 2025; Chen et al. 2026). We begin with a web corpus C and occupational activity seeds G. These seeds can be drawn from diferent occupational resources; in this work, we use seeds from O\*NET. For the web corpus, we use samples from the DLR-Web dataset (Attention1115 2025) used by DESIGNER (Liu et al. 2025), which are filtered from Fine-FineWeb (m-a-p 2025). Each segment has corpus-provided discipline metadata, which we map to an occupational category to select the corresponding seeds from G. For each segment, we perform workflow extraction under two filtering criteria: workflow filtering and file-oriented filtering. A segment that satisfies both criteria is converted into a workflow blueprint, information dependencies, required file types, and expected outputs, where the extraction is performed by Qwen3.6-Plus as the synthesis model $\pi _ { \mathrm { s y n } }$

$$
\mathcal { W } = \{ w \mid c \in \mathrm { F i l t e r e d } ( \mathcal { C } ) , w \sim \pi _ { \mathrm { s y n } } ( \cdot \mid c , \mathcal { G } ) \} .
$$

Attachment Searching. Given the workflow blueprints W, we retrieve public file artifacts and construct candidate file sets. Unlike prior settings built on large amounts of LLMgenerated synthetic files (Ge et al. 2026), we ground each task in authentic files retrieved from the web, guided by cues that the extracted workflows provide; such files preserve the natural complexity of file structures, layouts, and cross-file evidence. For each workflow w, we use $\pi _ { \mathrm { s y n } }$ to run a multiturn search-and-verification loop. At each turn, it generates a search query conditioned on the workflow, required file types, previous queries, and prior search results, and executes web\_search to obtain results that include titles and snippets. For each result, it first checks its title and snippet for workflow relevance; promising candidates are then downloaded and inspected with bash and view\_image to check key parts such as cover pages and tables ofcontents. Files verified as relevant to the workflow are retained in $\mathcal { F } _ { c } ;$ to control download latency and construction overhead, we discard files longer than 50 pages or larger than 2 MB. The loop stops when the retained files jointly satisfy the information and operation needs of w, or the maximum number of search turns is reached. Only satisfied pairs are passed to the next stage, while unsatisfied instances are discarded.

$$
\begin{array} { r l } & { \mathcal { F } _ { c } = \{ f \mid \mathrm { r e l } ( f , w ) = 1 \} , } \\ & { \mathcal { P } = \{ ( w , \mathcal { F } _ { c } ) \mid w \in \mathcal { W } , \mathrm { s a t } ( w , \mathcal { F } _ { c } ) = 1 \} . } \end{array}
$$

Task Synthesis. We instantiate each retained workflow and candidate-file pair $( w , \mathcal { F } _ { c } ) \in \mathcal { P }$ inside the sandbox environment E. To expose candidate-file content to the model, we render paginated files into page images and parse spreadsheets into sheet-level tables while preserving the original files for inspection. Conditioned on the workflow blueprint and the workflow-relevant candidate files, $\pi _ { \mathrm { s y n } }$ performs a multi-turn synthesis process. Across turns, it inspects file contents, searches and fetches web evidence when the workflow requires online information, determines which candidate files should be included as the attachment set $\mathcal { F } _ { a } \subseteq \mathcal { F } _ { c }$ and jointly designs the instruction I and rubric R. The instruction specifies the expected deliverable, while the rubric ties success criteria to file-level evidence and constraints.

![](images/9de3a0acfba47dba69994cc67c620d5a92d7e250b0130883013cfbcb7d05ade9.jpg)  
Figure 3: Rubric-based RL training. Agent-R<sup>2</sup> samples multi-turn rollouts in sandbox, renders generated deliverables for MLLM-as-a-judge scoring against rubrics and references, and updates the policy with group advantages and KL regularization.

We then use Claude-Opus-4.5 as the expert policy π<sub>exp</sub> to solve the synthesized task in the sandbox. The resulting files are kept as reference files $\mathcal { F } _ { r }$ for evaluation and judging. Along with the deliverable, we also record the expert trajectory τ , which captures the tool interactions used to inspect files, execute operations, and construct the final output. Let $\boldsymbol { u } = ( I , \mathcal { F } _ { a } , \mathcal { R } )$ denote the candidate solving context:

$$
( \tau , \mathcal { F } _ { r } ) \sim \pi _ { \mathrm { e x p } } ( \cdot \mid u ; \mathcal { E } ) .
$$

We further verify each synthesized candidate by evaluating the generated reference files against the rubric. The verifier identifies rubric items that are not satisfied by $\mathcal { F } _ { r }$ . For each unmet item, if the item is unsupported by the attachments, mismatched with the instruction, or otherwise unreasonable, we revise the rubric; if the item is reasonable, we keep the rubric and ask the expert policy to revise the deliverables once in the sandbox. The retained instances form the final task set $\mathcal { D } = \{ ( I , \mathcal { F } _ { a } , \mathcal { F } _ { r } , \mathcal { R } ) \}$ } and the corresponding verified expert trajectories Z.

## 3.2 Statistics

Starting from 10k corpus samples, Agent-R<sup>2</sup> yields 4k realistic productivity task instances. Table 7 summarizes the real web-sourced ofice attachments. The type distribution follows real-source availability, with more PDFs and fewer presentations, and thereby preserves the authenticity of file environments across occupations. The average sizes and page counts further show that these files retain practical content, layout, and format complexity. Figure 9 further summarizes task-level characteristics. Each task contains 2.51 attachments on average, requiring agents to inspect and integrate evidence from multiple artifacts (Figure 9(a)). The rubrics are fine-grained and comprehensive, covering both content and formatting requirements, with 15.3 items on average (Figure 9(b)), indicating that tasks require multiple localized reasoning and computation steps as well as both explicit and latent instruction following. Finally, Figure 9(c) reflects expert-trajectory complexity: trajectories contain 18.0 steps and 41.8k tokens on average, indicating that solving the tasks requires long-horizon evidence gathering, file manipulation, and deliverable construction.

## 4 Experiment

## 4.1 Training Framework

Using the synthesized task set D and filtered expert trajectories Z, we train Qwen3-4B-Thinking-2507 in two stages: supervised fine-tuning and rubric-based reinforcement learning (Huang et al. 2025; Gunjal et al. 2025). SFT teaches the model to follow expert trajectories, inspect files, and perform tool-based operations for deliverable construction. RL further optimizes the policy for rubric-scored outcome quality, encouraging error recovery and stronger satisfaction of task requirements. We split the data into 3k instances for SFT and 1k tasks for RL, with no task overlap.

Rubric-Based Reinforcement Learning. Starting from the SFT model, we optimize the policy with GRPO (Shao et al. 2024) in the persistent sandbox environment, as illustrated in Figure 3. Since agent-as-a-judge evaluation requires additional interactive tool use and is ineficient during RL training, we use Qwen3.6-Plus as an MLLM-as-ajudge (Chen et al. 2024) for outcome scoring. After task completion, the generated deliverable, relevant file evidence, and reference files are converted into model-consumable representations: .pdf, .pptx, and .docx files are rendered into images, while .xlsx files are represented by recalculated values and formulas in Markdown together with rendered page images, preserving both cell-level content and visual layout. The judge then scores the result against the synthesized rubric and reference files. To avoid inducing tool-specific calling preferences, we use only outcome reward and do not add explicit penalties for intermediate tool-call errors. The reward is computed as

$$
r = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f } \hat { \mathcal { F } } _ { d } = \emptyset , } \\ { \frac { \mathrm { S c o r e } ( \hat { \mathcal { F } } _ { d } , \mathcal { F } _ { r } , \mathcal { R } ) } { \mathrm { m a x ~ s c o r e } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

where ${ \hat { \mathcal { F } } } _ { d } , { \mathcal { F } } _ { r } ,$ , and R denote the generated deliverable files, synthesized reference files, and task-specific rubric used by the judge, respectively.

## 4.2 Experiment Setup

Baselines. We compare our proposed models against two primary categories of baselines: closed-source models, including GPT-4.1, Gemini-2.5-Pro, and Claude-Opus-4.5, and open-source models, including DeepSeek-v3.1,

<table><tr><td rowspan="2">Model</td><td colspan="2">Office</td><td colspan="2">Non-Office</td><td colspan="2">Full</td></tr><tr><td>Pass@1</td><td>Pass@3</td><td>Pass@1</td><td>Pass@3</td><td>Pass@1</td><td>Pass@3</td></tr><tr><td colspan="7">Closed-Source Models</td></tr><tr><td>GPT-4.1 (OpenAI 2025b)</td><td>38.0</td><td>51.5</td><td>44.9</td><td>55.5</td><td>39.4</td><td>52.3</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al. 2025)</td><td>57.0</td><td>59.4</td><td>61.4</td><td>63.3</td><td>57.9</td><td>60.2</td></tr><tr><td>Claude-Opus-4.5 (Anthropic 2025)</td><td>73.8</td><td>77.4</td><td>68.2</td><td>72.3</td><td>72.6</td><td>76.3</td></tr><tr><td colspan="7">Open-Source Models</td></tr><tr><td>DeepSeek-v3.1 (DeepSeek-AI et al. 2025)</td><td>48.3</td><td>57.5</td><td>50.9</td><td>61.7</td><td>48.8</td><td>58.4</td></tr><tr><td>GL-4.7 (Team et ai. 2025)</td><td>60.8</td><td>67.1</td><td>61.9</td><td>68.3</td><td>61.1</td><td>67.4</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507 (Yang et al. 2025)</td><td>33.7</td><td>38.5</td><td>40.4</td><td>44.9</td><td>35.0</td><td>39.8</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507 (Yang et al. 2025)</td><td>46.9</td><td>52.9</td><td>52.0</td><td>58.6</td><td>48.0</td><td>54.1</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking (Bai et al. 2025)</td><td>48.4</td><td>50.9</td><td>54.7</td><td>57.9</td><td>49.7</td><td>52.4</td></tr><tr><td colspan="7"> $A g e n t  – R ^ { 2 } \ : ( O u r s )$ </td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td>31.1</td><td>35.7</td><td>37.8</td><td>42.3</td><td>32.4</td><td>37.0</td></tr><tr><td> $\mathbf { A g e n t - R ^ { 2 } - 4 B - S F T }$ </td><td>45.9</td><td>51.5</td><td>42.8</td><td>47.6</td><td>45.3</td><td>50.7</td></tr><tr><td> $\mathbf { A g e n t - R ^ { 2 } { \cdot } } 4 \mathbf { B }$ </td><td>51.5</td><td>54.7</td><td>48.9</td><td>55.1</td><td>51.0</td><td>54.8</td></tr></table>

Table 1: Main results on GDPval. Pass@k allows up to k attempts and scores the first attempt that produces a deliverable; results are reported for ofice, non-ofice, and full subsets. Agent-R<sup>2</sup>-4B consistently improves over its base and SFT variants and outperforms several larger open-source baselines on the full set.

GLM-4.7, and Qwen3-series 30B and 235B text-only and multimodal variants. For Qwen-series text-only models, view\_image calls are handled by the same-series multimodal model with a matching parameter scale. For DeepSeek-v3.1, we use DeepSeek-v4-Flash for view\_image calls.

Benchmarks. We evaluate model performance on both indomain and out-of-domain benchmarks. GDPval (Patwardhan et al. 2025) serves as the in-domain benchmark, and we further categorize its tasks by the required output deliverable type into ofice and non-ofice tasks. The latter covers outputs beyond ofice formats, including audio (.wav, .mp3), video (.mp4), and other media or file types, which are outside our primary focus in this work. OficeQA (Opsahl-Ong et al. 2026) serves as the out-of-domain benchmark. We use its oficial Markdown version, where ofice source files are provided as plain-text Markdown files, to evaluate whether training improves document perception and reasoning rather than only the surface quality of generated deliverables.

Evaluation Protocol. For GDPval, we evaluate deliverables against GDPval rubrics with an agent-as-a-judge protocol (OpenAI 2025a), using Gemini-3.1-Pro as the judge model (Google 2026). For OficeQA, we combine rule-based matching with model-based assessment. Both benchmarks set a maximum of 100 steps. We report aggregate benchmark performance to quantify the incremental contributions of the SFT and RL stages.

## 4.3 Main Results

Agent-R<sup>2</sup> Delivers Strong Performance. Table 1 shows GDPval results. On the full set, Agent-R<sup>2</sup>-4B surpasses Qwen3-235B-A22B-Thinking-2507 (51.0% vs. 48.0%) despite using a much smaller backbone, and also outperforms DeepSeek-v3.1 and GPT-4.1. The advantage is especially meaningful on ofice tasks, our primary focus in this work, where Agent-R<sup>2</sup>-4B comes within 4.7 points of Gemini-2.5- Pro on Pass@3. Both SFT and RL consistently improve the base model on ofice and non-ofice tasks. More specifically, SFT brings larger gains on ofice tasks by imitating expert trajectories, while RL contributes larger gains on the less directly targeted non-ofice tasks, highlighting its importance for generalization beyond specific formats.

<table><tr><td>Model</td><td>Pass@1 Pass@3</td><td></td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td>2.2</td><td>4.5</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>2.9</td><td>5.7</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>13.8</td><td>25.6</td></tr><tr><td>Agent-R²-4B-SFT</td><td>11.4</td><td>21.7</td></tr><tr><td>Agent-R²-4B</td><td>14.8</td><td>26.8</td></tr></table>

Table 2: OOD results on OficeQA. Pass@k measures whether at least one of k attempts yields a correct answer (k = 1, 3). Agent-R<sup>2</sup>-4B outperforms Qwen3-235B-A22B-Thinking-2507.

Agent-R<sup>2</sup> Generalizes Beyond Productivity Work. Table 2 shows that the improvement transfers to OficeQA. Both SFT and RL improve over the base model, and the final Agent-R<sup>2</sup>-4B further surpasses Qwen3-235B-A22B-Thinking-2507 on both Pass@1 and Pass@3. These results suggest that training on synthesized productivity work improves not only deliverable construction, but also the underlying ability to inspect files, integrate evidence, and extract numerical answers across documents.

## 4.4 Analysis

We further analyze how SFT and RL reshape the model’s behavior, and show that our synthesized data help mitigate shortcut behavior through grounded exploration and deliverable verification.

SFT Encourages Sustained Exploration. The left panel of Figure 4 shows that SFT substantially increases the model’s exploratory behavior. The base model makes only 6.7 tool-call steps and 2.88 tool-call errors per sample on average. Such limited interaction suggests that the base mode often stops active exploration after encountering early parsing, execution, or file-handling errors, and instead rushes to construct deliverables with insuficient grounding. Agent-$\mathtt { R } ^ { 2 } .$ -4B-SFT increases the average number of tool-call steps to 39.0, with tool-call errors also increasing to 16.3. This shift indicates that SFT teaches the model to inspect files more actively, execute multi-step workflows, and continue attempting corrections after errors. Although limited model capacity and realistic environmental complexity lead to more errors, the longer exploration process ultimately guides the model toward more complete deliverables. Due to limited generalization, not every exploration attempt produces a successful deliverable, as reflected by the completion-rate drop from 86.8% to 78.2%, which limits the potential gain. However, this shapes a more proactive behavior pattern and lays the foundation for subsequent RL optimization.

RL Improves Reliable and High-Quality Completion. RL preserves the exploratory behavior introduced by SFT and turns it into more reliable task completion. Figure 4(left) shows that Agent- $\cdot \mathbf { R } ^ { 2 } .$ -4B raises the completion rate to 90.9%, above both the base and SFT variants. On the one hand, it averages 33.1 tool-call steps, far above the base model’s 6.7, indicating that it does not fall back to sparse inspection and premature output. On the other hand, tool-call errors decrease from 16.3 under SFT to 12.9, suggesting that rubric-based RL helps the model recover from errors, explore more eficiently, and generalize better in complex environments. Furthermore, the completion quality improves. Figure 4(right) compares $\mathrm { A g e n t - R ^ { \bar { 2 } } - 4 B }$ with the SFT-only variant on GDPval tasks that both models complete successfully. The comparison shows that RL wins more often than SFT on both ofice and nonofice tasks, indicating that rubric-based RL improves not only completion rate but also the quality of deliverables.

Mitigating Shortcut Behavior Drives Improvement. We define efective steps as tool-call steps that do not result in errors, which to some extent reflect exploration and completion depth. In Figure 5, we observe that efective steps positively correlate with GDPval Pass@1. Across model scales, agents with few efective steps tend to stop before suficient grounding or verification, producing plausible artifacts without adequate evidence extraction or computation. We refer to this failure mode as shortcut behavior. $\mathrm { \ A g e n t { - } R ^ { 2 } }$ mitigates this failure mode through two-stage SFT and rubric-based RL on the synthesized 4k data; in contrast, the 4B base model shows shallow interaction and low performance.

We further examine the trajectories and deliverables of the base, SFT, and RL models on all 220 GDPval tasks. Metric definitions are provided in Appendix C. Table 3 shows that both training stages reduce shortcut behavior. The base model rarely re-checks its deliverable and, in the build phase, writes the deliverable right after an unfixed error in 55% of cases. After SFT and RL, the agent re-checks almost every deliverable, and such bypasses fall to 16% and 11%. Numerical errors are fixed more often at each stage, and placeholder text falls to 5.7%, the same rate as in the expert-written GDPval deliverables. From SFT to RL, writer code that masks errors with except→default/pass falls from 4.1% to 0.9%, and the share of traceable numbers in deliverables rises from 34.6% to 38.8%.

![](images/a269981ea149ba6ce17010e807a8d8ece2457e8be7d6e26765c250fa8d496a5b.jpg)

![](images/79ccb1df8099a5f8d94c7dc8ecd5454aa18be7eacff3a1b51fd9508b749c4121.jpg)  
Figure 4: Model behavior and completion quality. Left: completion rate, average tool-call steps, and average toolcall errors across the base, SFT, and RL variants. Right: win/tie/loss outcomes of the RL-trained model against the SFT-only model after valid completion on GDPval, where green, orange, and gray segments indicate RL wins, ties, and SFT wins, respectively.

![](images/8038b5730fc36fa91e9ac81aad0efb891938497657fc3aa3cf7bae2baa66e215.jpg)  
Figure 5: Efective Steps vs. GDPval Performance. Effective steps are computed as average tool-call steps minus tool-call errors. Models with more efective steps generally achieve higher GDPval Pass@1.

Case Study on Mitigating Shortcut Behavior. Figure 6 illustrates how shortcut behavior appears in a materials-test report task. The base Qwen3-4B model fails to locate the target spreadsheet values, but still writes a report with dummy evidence, producing a superficially complete deliverable. Gemini-2.5-Pro performs more inspection, yet silently converts key measurements to NaN and generates empty charts. In contrast, Agent- $\mathbf { \nabla } \cdot \mathbf { R } ^ { 2 } \mathbf { \partial } \cdot 4 \mathbf { B }$ recovers from parsing, computation, and drawing errors, computes the required hardness values, and verifies the final plots, achieving a score far above Gemini-2.5-Pro.

![](images/6683cdf22b788d3535c50fc108511630e8a858ef338e70057bf40a86653350ab.jpg)  
Figure 6: Case study of shortcut behavior in materials-test report generation. The base model and Gemini-2.5-Pro produce flawed reports from incomplete or invalid spreadsheet evidence, whereas Agent-R<sup>2</sup>-4B recovers from parsing and execution errors, computes the required values, and verifies the generated plots in the final PDF deliverable.

<table><tr><td>Metric (%)</td><td>Base</td><td>SFT</td><td>RL</td></tr><tr><td>Re-checking the deliverable ↑</td><td>0.5</td><td>91.7</td><td>95.8</td></tr><tr><td>Writing despite an unfixed error ↓</td><td>55.0</td><td>16.0</td><td>11.0</td></tr><tr><td>Numerical-error fix rate ↑</td><td>60.7</td><td>79.6</td><td>85.7</td></tr><tr><td>Deliverables with placeholder text ↓</td><td>12.5</td><td>9.6</td><td>5.7</td></tr><tr><td>Traceable numbers in deliverables ↑</td><td>31.2</td><td>34.6</td><td>38.8</td></tr></table>

Table 3: Direct evidence on shortcut behavior on GDPval. All values are percentages: re-checking and placeholder rates are measured over eligible tasks, bypass and fix rates over the corresponding error episodes, and numeric traceability as the mean per-task share of traceable numbers.

## 4.5 Ablation Studies

To isolate the contribution of real files, we replace the attachment-searching stage with prompting the model to generate files for a 1k subset of our SFT tasks, then resynthesize each task using the same pipeline. This yields 1k matched pairs built from the same source instances, with a similar number of attachments on both sides. Training one SFT model on each side under the same recipe yields GDPval Pass@1 of 43.3 for real files versus 41.4 for synthetic files (Table 4). The expert trajectories also use 2.54× more tool calls with real files (20.4 vs. 8.0), reflecting the layouts, structures, and cross-file dependencies that require more inspection and processing. To isolate the contribution of finegrained rubrics, we train an RL variant from the same SFT model without rubric rewards. Under the evaluation protocol of Section 4.2, removing rubric rewards lowers GDPval Pass@1 from 51.0 to 49.5 (Table 4), demonstrating the benefit of rubric-based rewards.

<table><tr><td>Ablation</td><td>Method</td><td>Pass@1</td></tr><tr><td>Data grounding</td><td>w/ synthetic files</td><td>41.4</td></tr><tr><td></td><td>w/ real files</td><td>43.3</td></tr><tr><td>Reward design</td><td>w/o rubrics</td><td>49.5</td></tr><tr><td></td><td>w/ rubrics</td><td>51.0</td></tr></table>

Table 4: Ablation of real files and rubric rewards on GDPval. SFT models use matched real-file or synthetic-file tasks. RL variants use or omit rubric rewards, starting from the same SFT model.

## 5 Related Work

Benchmarks for Real-World Productivity Tasks. Recent benchmarks increasingly move beyond static question answering toward realistic, long-horizon work completion. GDPval (Patwardhan et al. 2025), \$1M-Bench (Yang et al. 2026), and Agents’ Last Exam (Sun et al. 2026) evaluate economically valuable work through expert-designed occupational and high-stakes tasks. Finance Agent Benchmark (Bigeard et al. 2025) focuses on finance research tasks grounded in SEC filings and expert rubrics. APEX– Agents (Vidgen et al. 2026) and AgentSkillOS (Li et al. 2026) study evaluations centered on tool use and skill orchestration, where agents must select and coordinate external capabilities to complete tasks. ClawBench (Zhang et al. 2026b) and EdgeBench (Zhu et al. 2026) study agents in real-world interactive environments, emphasizing live operations, sustained exploration, and feedback-driven progress. Workspace-Bench (Tang et al. 2026) and EnterpriseClaw-Bench (Zhong et al. 2026) focus on workspace and enterprise settings, where agents must reason over heterogeneous files, business artifacts, and workplace-specific context.

Data Synthesis for Agent Post-Training. Synthesizing post-training data has become an important direction for improving agent capabilities. Existing work scales tool-use trajectories and executable environments from corpora, programs, or workflow templates (Xu et al. 2026b; Song et al. 2026; Xu et al. 2026a; Dong et al. 2026), or replay web tutorials as verified GUI trajectories (Xu et al. 2025b). These methods are efective for scaling tool-use data, but they mainly target specific tools with predefined interfaces rather than more general tools such as terminals, which ofer broader applicability for realistic productivity work. Recent data synthesis for terminal agents improves verifiability by generating command-line environments, tasks, or skills (Gandhi et al. 2026; Cheng et al. 2026; Hua et al. 2026). However, these settings mainly focus on structured text files and rarely involve structurally complex ofice files commonly used in professional productivity work. Studies based on ofice files mainly focus on document-grounded question answering and analysis (Zhang et al. 2026a; Qiao et al. 2026; Zhang et al. 2025; Chi et al. 2026) or single-type file generation (Zheng et al. 2025, 2026). In contrast, $\dot { \mathrm { A g e n t - R ^ { 2 } } }$ grounds task synthesis in real occupational workflows and authentic multi-format attachments, producing deliverable-oriented tasks together with fine-grained rubrics.

## 6 Conclusion

We introduced Agent- ${ \bf - R } ^ { 2 }$ , a real-source data synthesis framework that scales agents for realistic productivity work. Agent-$\mathtt { R } ^ { 2 }$ constructs training instances grounded in real workflows and authentic files with fine-grained rubrics. Using 4k synthesized instances, we train $\bar { \bf A } \mathrm { g e n t - } { \bf R } ^ { 2 } { \bf - } 4 { \bf B }$ with SFT and rubric-based RL. Experiments on GDPval and OfficeQA show that $\mathrm { A g e n t - } \bar { \mathrm { R } ^ { 2 } } \mathrm { - } 4 \mathrm { B }$ improves task completion and file-grounded reasoning, outperforming several larger open-source baselines. Behavioral analysis further shows that $\bar { \mathrm { A g e n t - R ^ { 2 } - 4 B } }$ reduces shortcut behavior and improves completion reliability in complex productivity tasks.

## References

Anthropic. 2025. Introducing Claude Opus 4.5. https://www. anthropic.com/news/claude-opus-4-5. Accessed: 2026-07- 11.

Attention1115. 2025. DLR-Web. https://huggingface.co/ datasets/Attention1115/DLR-Web. Hugging Face dataset. Accessed: 2026-07-11.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; et al. 2025. Qwen3-VL Technical Report. arXiv:2511.21631.

Bigeard, A.; Nashold, L.; Krishnan, R.; and Wu, S. 2025. Finance Agent Benchmark: Benchmarking LLMs on Real-World Financial Research Tasks. arXiv:2508.00828.

Chen, D.; Chen, R.; Zhang, S.; Liu, Y.; Wang, Y.; et al. 2024. MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark. arXiv:2402.04788.

Chen, K.; Wu, Q.; Hou, T.; Tang, T.; Hu, X.; et al. 2026. AgentIF-OneDay: A Task-Level Instruction-Following Benchmark for General AI Agents in Daily Scenarios. arXiv:2601.20613.

Cheng, Z.; Wang, H.; Liu, Z.; Wang, X.; Zhu, X.; et al. 2026. Terminal-World: Scaling Terminal-Agent Environments via Agent Skills. arXiv:2605.20876.

Chi, B.; Xie, Y.; Wu, M.; Yang, J.; Jiang, J.; et al. 2026. Spreadsheet-RL: Advancing Large Language Model Agents on Realistic Spreadsheet Tasks via Reinforcement Learning. arXiv:2605.22642.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; et al. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv:2507.06261.

DeepSeek-AI; Liu, A.; Feng, B.; Xue, B.; Wang, B.; et al. 2025. DeepSeek-V3 Technical Report. arXiv:2412.19437.

Dong, G.; Lu, J.; Huang, J.; Zhong, W.; Liu, L.; et al. 2026. Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence. arXiv:2604.18292.

Gandhi, K.; Garg, S.; Goodman, N. D.; and Papailiopoulos, D. 2026. Endless Terminals: Scaling RL Environments for Terminal Agents. arXiv:2601.16443.

Ge, T.; Peng, B.; Cheng, H.; and Gao, J. 2026. Synthetic Computers at Scale for Long-Horizon Productivity Simulation. arXiv:2604.28181.

Google. 2026. Gemini 3.1 Pro Preview. https://ai.google. dev/gemini-api/docs/models/gemini-3.1-pro-preview. Accessed: 2026-07-11.

Gunjal, A.; Wang, A.; Lau, E.; Nath, V.; He, Y.; Liu, B.; and Hendryx, S. 2025. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. arXiv:2507.17746.

Guo, Z.; Liu, W.; Xie, M.; Xu, J.; Huang, Z.; et al. 2025. RECAST: Expanding the Boundaries of LLMs’ Complex Instruction Following with Multi-Constraint Data. arXiv:2505.19030.

Hua, Z.; Yao, Y.; Xie, W.; Zhao, Y.; Liu, M.; et al. 2026. CLI-Universe: Towards Verifiable Task Synthesis Engine for Terminal Agents. arXiv:2606.22883.

Huang, Z.; Zhuang, Y.; Lu, G.; Qin, Z.; Xu, H.; et al. 2025. Reinforcement Learning with Rubric Anchors. arXiv:2508.12790.

Li, H.; Mu, C.; Chen, J.; Ren, S.; Cui, Z.; et al. 2026. Organizing, Orchestrating, and Benchmarking Agent Skills at Ecosystem Scale. arXiv:2603.02176.

Liu, W.; Zhao, Y.; Luo, Y.; Xu, M.; Liu, J.; et al. 2025. DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning. arXiv:2508.12726.

m-a-p. 2025. FineFineWeb. https://huggingface.co/datasets/ m-a-p/FineFineWeb. Hugging Face dataset. Accessed: 2026- 07-15.

OpenAI. 2025a. GDPval. https://huggingface.co/datasets/ openai/gdpval. Hugging Face dataset. Accessed: 2026-07- 11.

OpenAI. 2025b. Introducing GPT-4.1 in the API. https: //openai.com/index/gpt-4-1/. Accessed: 2026-07-11.

Opsahl-Ong, K.; Singhvi, A.; Collins, J.; Zhou, I.; Wang, C.; et al. 2026. OficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning. arXiv:2603.08655.

Patwardhan, T.; Dias, R.; Proehl, E.; Kim, G.; Wang, M.; et al. 2025. GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks. arXiv:2510.04374.

Qiao, S.; Zhao, Y.; Qiu, Z.; Wang, X.; Zhang, J.; et al. 2026. Scaling Generalist Data-Analytic Agents. arXiv:2509.25084.

Schick, T.; Dwivedi-Yu, J.; Dessì, R.; Raileanu, R.; Lomeli, M.; et al. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Song, X.; Chang, H.; Dong, G.; Zhu, Y.; Wen, J.-R.; and Dou, Z. 2026. EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis. arXiv:2601.05808.

Sun, Y.; Han, X.; Zhang, W.; Pang, Y.; Wang, T.; et al. 2026. Agents’ Last Exam. arXiv:2606.05405.

Tanaka, R.; Nishida, K.; Nishida, K.; Hasegawa, T.; Saito, I.; and Saito, K. 2023. SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images. arXiv:2301.04883.

Tang, Z.; Zhou, X.; Liu, Y.; Li, L.; Wu, Y.; et al. 2026. Workspace-Bench 1.0: Benchmarking AI Agents on Workspace Tasks with Large-Scale File Dependencies. arXiv:2605.03596.

Team, G.-. .; Zeng, A.; Lv, X.; Zheng, Q.; Hou, Z.; et al. 2025. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models. arXiv:2508.06471.

Vidgen, B.; Mann, A.; Fennelly, A.; Stanly, J. W.; Rothman, L.; et al. 2026. APEX-Agents. arXiv:2601.14242.

Wang, J.; Liu, X.; Li, Y.; Zhang, S.; Wang, Y.; Shan, Z.; et al. 2026. GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows. arXiv:2604.15715.

Xu, C.; Sun, Q.; Zheng, K.; Geng, X.; Zhao, P.; et al. 2025a. WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions. arXiv:2304.12244.

Xu, M.; Wang, Z.; DENG, M.; Li, Z.; Yang, Z.; et al. 2026a. EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL. arXiv:2605.18703.

Xu, Y.; Lu, D.; Shen, Z.; Wang, J.; Wang, Z.; Mao, Y.; Xiong, C.; and Yu, T. 2025b. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials. International Conference on Learning Representations (ICLR).

Xu, Z.; Li, R.; Li, J.; Weng, R.; Wang, J.; Cai, X.; and Wang, X. 2026b. Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text. arXiv:2601.10355.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; et al. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Yang, Q.; Liu, Y.; Li, J.; Bai, J.; Chen, H.; et al. 2026. \$OneMillion-Bench: How Far Are Language Agents from Human Experts? arXiv:2603.07980.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.

Zhang, Q.; Lv, X.; Wu, J.; Li, B.; Tao, Z.; et al. 2026a. Doc-Dancer: Towards Agentic Document-Grounded Information Seeking. arXiv:2601.05163.

Zhang, S.; Fan, J.; Fan, M.; Li, G.; and Du, X. 2025. Deep-Analyze: Agentic Large Language Models for Autonomous Data Science. arXiv:2510.16872.

Zhang, Y.; Wang, Y.; Zhu, Y.; Du, P.; Miao, J.; et al. 2026b.ClawBench: Can AI Agents Complete Everyday OnlineTasks? arXiv:2604.08523.

Zheng, H.; Guan, X.; Kong, H.; Zheng, J.; Zhou, W.; et al. 2025. PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides. arXiv:2501.03936.

Zheng, H.; Mo, G.; Yan, X.; Yuan, Q.; Zhang, W.; et al. 2026. DeepPresenter: Environment-Grounded Reflection for Agentic Presentation Generation. arXiv:2602.22839.

Zhong, J.; Wang, W.; Jiang, C.; Tian, K.; Yuan, Z.; et al. 2026. EnterpriseClawBench: Benchmarking Agents from Real Workplace Sessions. arXiv:2606.23654.

Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; et al. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. arXiv:2307.13854.

Zhu, D.; Zhou, X.; Qin, S.; Zhu, X.; Ding, H.; et al. 2026. EdgeBench: Unveiling Scaling Laws of Learning from Real-World Environments. arXiv:2607.05155.

## A Model Configuration and Training Details A.1 Model Configuration

Table 5 specifies the auxiliary vision model used by each text-only solver evaluated on GDPval. For these solvers, view\_image invokes the model listed in the second column and returns its textual visual observation. To promote a fair comparison, we select, where possible, an auxiliary vision model from the same model family and with a parameter count comparable to that of the solver. All Agent-R<sup>2</sup> variants use the same visual backend as their Qwen3-4B-Thinking-2507 base model. Solvers with native visual capability are omitted from the table.

<table><tr><td>Solver model</td><td>Auxiliary vision model</td></tr><tr><td>DeepSeek-v3.1</td><td>DeepSeek-v4-Flash</td></tr><tr><td>Qwen3-30B-A3B-Thinking-2507</td><td>Qwen3-VL-30B-A3B-Thinking</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>Qwen3-VL-235B-A22B-Thinking</td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td>Qwen3-VL-4B-Thinking</td></tr><tr><td>Agent-R²-4B-SFT</td><td>Qwen3-VL-4B-Thinking</td></tr><tr><td> $\mathbf { A g e n t - R ^ { 2 } - 4 B }$ </td><td>Qwen3-VL-4B-Thinking</td></tr></table>

Table 5: Auxiliary vision models used for text-only solvers in GDPval evaluation.

## A.2 Training and Evaluation Details

Agent Runtime and Tools The agent loop is built on verl v0.6.x and runs in the open-source AIO Sandbox. Tools are exposed through the OpenAI function-calling interface. The web\_search and web\_fetch tools are backed by Serper; during attachment searching, queries use file-type filters such as filetype:pptx to target the required formats.

Training Details Each SFT trajectory is split into perround samples so that every sample fits the Qwen3 chat template. SFT runs for a fixed two epochs. For both SFT and RL, we use the final checkpoint and perform no validationbased checkpoint selection; neither GDPval nor OficeQA is used for model selection. Table 6 summarizes the hyperparameters used for reinforcement learning. Figure 7 shows that the training reward follows an overall upward trend over the course of optimization.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>batch_size</td><td>16</td></tr><tr><td>max_response_length</td><td>65,536</td></tr><tr><td>group_size</td><td>4</td></tr><tr><td>max_assistant_turns</td><td>40</td></tr><tr><td>max_tool_response_length</td><td>4,096</td></tr><tr><td>training_temperature</td><td>1.0</td></tr><tr><td>learning_rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr></table>

Table 6: Hyperparameters used for reinforcement learning.

Evaluation Protocol All results in Tables 1 and 2 are freshly evaluated under a common runtime setup with the same sandbox and tool framework. Each run allows at most

![](images/a315f59e3feacad63cfc9845f10a7f43050916c91631c726ae89b3c66bf17666.jpg)  
Figure 7: Training reward during reinforcement learning.

![](images/15331c2d0a100603323a634d865065570adbae19d9f28ba3a2981b9f390f9834.jpg)  
Figure 8: Cosine similarity across synthesis stages.

100 steps and an unlimited number of parallel tool calls per step. API baselines use their default sampling settings, and the auxiliary vision models used by text-only solvers are listed in Table 5. The benchmark-specific Pass@k definitions are given in the corresponding table captions. API-error retries are handled separately from the k task attempts.

## B Additional Results

## B.1 Judge Stability across Repeats and Judge Models

GDPval scores rely on an LLM judge, so we test whether our conclusions depend on judge stochasticity or on the choice of judge. We use two judge models, Gemini-3.1-Pro and DeepSeek-v4-Flash, and each judge independently evaluates the deliverables of $\mathbf { A g e n t - R ^ { 2 } - 4 B }$ and Qwen3-235B-A22B-Thinking-2507 on all 220 GDPval tasks three times.

Table 8 reports the range of Pass@1 across the three repeats. Per-task scores do vary between repeats, with a median absolute diference of 3.2 and 3.1 points (mean 5.7 and 5.4) under the two judges, but these diferences largely cancel in the aggregate score. Agent-R<sup>2</sup>-4B stays within 50.8–51.1 under Gemini-3.1-Pro, consistent with the 51.0 reported in Table 1, and within 51.7–52.3 under DeepSeek-v4-Flash. The two judges difer slightly in absolute scores but agree on relative quality, with a per-task Pearson correlation of 0.93. Under both judges, the lowest score of Agent-R<sup>2</sup>-4B is above the highest score of Qwen3-235B-A22B-Thinking-2507, so $\mathrm { A g e n t { - } R ^ { 2 } { - } } 4 \mathrm { B }$ leads in all six evaluations.

<table><tr><td>Model</td><td>Gemini-3.1-Pro</td><td>DeepSeek-v4-Flash</td></tr><tr><td>Qwen3-235B-A22B</td><td>47.7-48.2</td><td>49.2-49.9</td></tr><tr><td>Agent-R²-4B</td><td>50.8–51.1</td><td>51.7–52.3</td></tr></table>

Table 8: GDPval Pass@1 range over three independent evaluations by each judge model.

## B.2 Data Scaling

To examine how performance scales with the amount of synthesized data, we train SFT models on 1k, 2k, and 4k trajectories and evaluate them on 100 held-out synthesized tasks, scored from 0 to 1 against their rubrics, as well as on GDPval. Table 9 shows that scores on the held-out synthesized tasks keep rising with more data, and the win rate over the base model grows accordingly, while GDPval Pass@1 saturates. The continued gains on held-out synthesized tasks indicate that the pipeline keeps producing useful supervision as it scales, and the saturating GDPval curve shows that we do not overfit the benchmark.

Table 7: Attachment statistics. Counts and complexity statistics for the four major ofice-oriented attachment types.
<table><tr><td>Type</td><td>Count Avg Sz. (MB)</td><td></td><td>Mean Pg.</td></tr><tr><td>·pdf</td><td>5,010</td><td>1.00</td><td>15.0</td></tr><tr><td>.docx</td><td>3,149</td><td>0.25</td><td>13.7</td></tr><tr><td>.xlsx</td><td>1,258</td><td>0.20</td><td></td></tr><tr><td>·pptx</td><td>390</td><td>0.78</td><td>23.6</td></tr></table>

![](images/5bee61c66686a7a8a29a817ba1d6f66920c3b8ad0bfe3356a16eeff229192128.jpg)  
(a) Attachment Count

![](images/02d6c48b9f2d36c4930c73b02d355a1328a2ffae8b776ead34c2db8c8caa0514.jpg)

![](images/25d5da25b1b3bb130324172fa9794b445e27ce813b0fd38621d5f9feb14f6b26.jpg)  
Figure 9: Synthesized task statistics: attachments per task, rubric items per task, and expert-trajectory length.

<table><tr><td>SFT data</td><td>Held-out score</td><td>Win rate</td><td>GDPval Pass@1</td></tr><tr><td>0 (base)</td><td>0.254</td><td></td><td>32.4</td></tr><tr><td>1k</td><td>0.486</td><td>64%</td><td>43.3</td></tr><tr><td>2k</td><td>0.507</td><td>67%</td><td>44.9</td></tr><tr><td>4k</td><td>0.617</td><td>73%</td><td>45.5</td></tr></table>

Table 9: Data scaling of SFT on 100 held-out synthesized tasks and on GDPval.

## B.3 Additional Data versus the RL Stage

The final model uses about 3k instances for SFT and about 1k additional tasks for RL, so its gain over the SFT model could in principle come from exposure to more data. To control for this, we train an SFT model on all 4k expert trajectories. Table 10 shows that this 4k-SFT control reaches 45.5 Pass@1, only slightly above the 3k-SFT model (45.3) and well below SFT+RL (51.0). The gain therefore comes from RL rather than from more data, consistent with the behavioral analysis in Section 4.4.

This control also addresses whether the gains come mainly from distilling a strong teacher. Imitating all 4k teacher trajectories stops at 45.5 Pass@1, whereas RL on the same data adds 5.7 Pass@1 over the 3k-SFT model (45.3→51.0).

<table><tr><td>Model</td><td>Training data</td><td>GDPval Pass@1</td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td></td><td>32.4</td></tr><tr><td>Agent-R²-4B-SFT</td><td>3k SFT</td><td>45.3</td></tr><tr><td>4k-SFT control</td><td>4k SFT</td><td>45.5</td></tr><tr><td>Agent-R²-4B</td><td>3k SFT + 1k RL</td><td>51.0</td></tr></table>

Table 10: Separating the efect of additional data from the RL stage.

## B.4 Additional Benchmark

We additionally evaluate on GTA-Workflow (Wang et al. 2026), which contains 132 open-ended workflows. Agent-R<sup>2</sup>-4B achieves a Root Score of 4.52/10 under the same

<table><tr><td>Effective steps per task</td><td>SFT</td><td>RL</td></tr><tr><td>Computation</td><td>11.9</td><td>7.3</td></tr><tr><td>Attachment reading</td><td>3.4</td><td>3.4</td></tr><tr><td>Deliverable writing</td><td>2.3</td><td>2.7</td></tr><tr><td>Build phase (all step types)</td><td>4.9</td><td>5.5</td></tr><tr><td>Share of reading, writing, and checking</td><td>40%</td><td>48%</td></tr></table>

Table 11: Decomposition of efective steps on GDPval from SFT to RL.

agent-as-a-judge protocol, compared with 1.53/10 for its base model and 5.15/10 for DeepSeek-v4-Flash.

## B.5 Why Efective Steps Decrease after RL

We decompose efective steps by step type under the error definition in Appendix C. Note that this definition counts only genuine error signatures, so absolute counts difer slightly from Figure 5. Table 11 shows that from SFT to RL, almost the entire decrease comes from computation steps such as data processing, which drop from 11.9 to 7.3 per task. Attachment reading is unchanged, while deliverable writing and the build phase grow. As a result, RL spends a larger share of its efective steps on reading, writing, and checking the deliverable (40%→48%).

More steps do not mean better work. Within each model, longer trajectories score lower (per-task Spearman correlation −0.15 for SFT and −0.41 for RL), and the share of runs that hit the 100-step limit drops from 8.2% to 3.6%. RL thus cuts redundant computation while preserving the grounding and verification steps that matter for the deliverable. The gain comes from how the steps are used rather than how many steps are taken.

## C Metric Definitions for Shortcut Analysis

The following definitions are used for the trajectory- and deliverable-level analysis in Section 4.4.

All statistics are computed automatically with rule-based matching on trajectories and deliverables.

• Error. A tool call whose output contains a genuine error signature, such as a traceback, an error message, or a missing file. Output on stderr without an error signature is not counted.

• Phases. The exploration phase precedes the first successful deliverable write; the build phase spans the first to the last successful write; the verification phase follows the last successful write.

• Numerical-error fix rate. The percentage of numerical/data-logic error episodes that are subsequently fixed under the criterion above. This category includes indexing, value, type, arithmetic, and related computation exceptions.

• Writing despite an unfixed error. A deliverable is written before an error is fixed, with at most one intervening successful step. The reported rate is the percentage of build-phase error episodes that satisfy this condition.

• Re-checking the deliverable. The agent reopens or inspects the deliverable after a successful write. The rate is measured over tasks with a successful deliverable-writing operation.

• Placeholder text. The deliverable contains placeholder strings such as “TBD”, “[insert . . . ]”, “lorem”, or “placeholder”. The rate is measured over tasks with reference attachments and an output deliverable.

• Error masking. The code that writes the deliverable catches exceptions and falls back to a default value or pass. We report the percentage of tasks with a successful deliverable-writing operation whose writer code matches this pattern.

• Traceable numbers. The share of distinct numbers in the output deliverables that can be found in the prompt or the attachments. We compute this share for each task with reference attachments and at least five distinct output numbers, then report the mean across eligible tasks.

## D Decontamination Audit

Our training materials are collected from web corpora and online search, so we audit the synthesized tasks and their attachments for overlap with the evaluation benchmarks at four levels.

File hashes and source URLs. Neither the SHA-256 hashes of our attachment files nor the source URLs we recovered overlap with any file in GDPval or OficeQA.

Source domains. Only 0.6% of our source domains appear in GDPval instructions, mostly broad public sources such as irs.gov.

Near-duplicate documents. For each attachment, we compute the nearest-neighbor 5-gram MinHash Jaccard similarity to the benchmark documents. The average is 0.005 for GDPval (maximum 0.125) and 0.009 for OficeQA (maximum 0.023). For reference, two diferent issues of the same U.S. Treasury Bulletin have a similarity of 0.36.

Instruction-level similarity. For each benchmark prompt, we compute the embedding cosine similarity to its nearest training instruction. The average is 0.43 for GDPval and 0.36 for OficeQA. For GDPval, this is below the similarity between a GDPval prompt and its nearest other GDPval prompt (0.46), and far below the similarity of a paraphrase (0.93).

## E Data Details

Retrieving real files does not by itself guarantee realistic tasks: topic-relevant attachments may difer in entities, dates, or assumptions, and synthesized rubrics may not be grounded in the files. This section brings together filtering decisions, synthesis statistics, manual quality audits, and data compliance and release details.

## E.1 Filtering Decisions and Data Attrition

Each filtering step is implemented as a structured decision that the prompts in Appendix F require the model to output in JSON. For example, during workflow extraction only documents labeled "workflow": true are kept, and during attachment searching only files listed under "highly\_relevant\_files" are retained. Starting from 10k seed documents, about 2.2k are discarded at workflow extraction and about 3.8k during attachment searching and verification, leaving about 4k task instances.

## E.2 Semantic Similarity Across Synthesis Stages

Figure 8 reports text similarity measured by Qwen3- Embedding-4B across synthesis stages. The final instructions have low similarity to the seed tasks, with an average similarity of 0.33.

## E.3 Manual Quality Inspection

We conduct a manual audit of100 examples from the retained synthesis set across three quality dimensions: workflow– question alignment (86%), rubric-item validity (99.8%), and visual presentation (97%).

Workflow and question construction are fully aligned in 86 examples, while the remaining 14 show partial alignment. In these cases, some questions use descriptive calculations in place of causal or simulation analysis, some omit key crossvalidation, and others omit or modify a requested artifact. Despite these diferences, all 14 examples remain topically consistent with their workflows, although topical consistency alone does not meet the criterion for full alignment. For rubric validity, two rubric items are inaccurate because repeated calculations produce incorrect numerical values. For visual presentation, the examples with room for improvement mainly contain sparse table or continuation pages and otherwise blank trailing pages with only a footer. These presentation issues do not afect the overall readability of the deliverables.

## E.4 Cross-File Consistency and Task Feasibility

We further manually evaluate the same 100 tasks along two additional dimensions. Cross-file consistency asks whether the attachments of a task describe mutually compatible entities, dates, and assumptions. Feasibility asks whether the task can be completed from its instruction and attachments. Crossfile consistency reaches 95%, and task feasibility reaches 85%. Among the inconsistent tasks, one example mixes GPU specifications with data on ASIC Bitcoin miners.

## E.5 Rubric Grounding

The manual audit finds that 99.8% of rubric items are valid. To measure how strongly rubrics depend on the attachments, we trace every specific number that appears in the training rubrics. Of these numbers, 41.9% appear in the instruction, 39.0% appear only in the attachments, and 19.1% appear in neither and must be computed. Most rubric numbers therefore cannot be checked from the instruction alone, so satisfying the rubric requires reading and processing the attached files.

## F Prompts

Listing 1: Workflow extraction prompt.

```ini
[System]
You are a professional productivity task scenario designer. Your responsibility is to
analyze web corpus content, determine whether it is suitable for designing workflow
scenarios, and when suitable, mine potential workflow scenarios from it.
[Your Core Tasks]
1. Deeply analyze the input web corpus, understanding its core themes, domain background,
and key information.
2. Determine whether the corpus can support one or more real, complex, file-based workflow
scenarios as described in the task description.
3. If so, mine potential workflows and required files based on the corpus content and task
description, conducting multi-step workflows based on one or more files.
4. You may freely combine, decompose, or derive one or more task scenarios without being
limited to the preset task descriptions.
[Workflow Design Principles]
Authenticity: Scenarios should reflect real-world work requirements that exist in practice
Generalizability: Workflow steps should use generic sub-goal descriptions, avoiding overly
specific entity information that would be hard to find relevant files for (e.g.,
specific chapter numbers, codes, etc.).
Complexity: Workflows should have sufficient complexity, requiring multi-step reasoning
and operations to complete.
Reasonableness: Required files should conform to actual work habits in the relevant domain
[Output Format]
Please strictly follow the JSON format below.
When the corpus is not suitable for question design (workflow is false):
{"summary": "Brief summary of the web corpus content, describing its core theme", "workflow
": false}
When the corpus is suitable for question design (workflow is true):
"summary": "Brief summary of the web corpus content, describing its core theme",
"workflow": true,
"task": ["Reference task content verbatim", "Reference task content verbatim"],
"scenario": {
"title": "Scenario title",
"workflow": [
{"step": 1, "description": "Step 1 description"},
{"step": 2, "description": "Step 2 description"}
]
},
"required_files": [
"Description of file type and purpose",
"Description of file type and purpose"
],
"reasoning": "Rationale for designing this scenario, explaining how the web corpus
supports this workflow"
[User]
Based on the following information, first determine whether the web corpus is suitable for
question design, then decide whether to design a workflow scenario.
[Job Title]
<JOB_TITLE>
[Web Corpus Content]
```

<WEB\_CORPUS\_CONTENT>   
[Reference Tasks] (Fill the task field with the verbatim content of the tasks actually used)   
<CANDIDATE\_OCCUPATIONAL\_TASKS>   
Please deeply analyze the web corpus content:   
1. First determine whether the corpus is suitable for question design.   
2. If suitable, mine potential workflow scenarios and fill the task field with the verbatim   
reference task content actually used.   
3. Each item in required\_files should be a single string describing the file type and   
purpose.   
Output the result in JSON format.

Listing 2: Attachment search and verification prompt.  
```ini
[System]
You are a professional resource search assistant. Your responsibility is to search for and
retrieve the required file resources over multiple rounds based on the workflow
requirements.
[Your Core Tasks]
Analyze the types and purposes of files required by the workflow scenario, design
effective search keywords, call search tools to retrieve files, evaluate whether the
file content meets the scenario requirements, and rename files if their names do not
conform to realistic naming conventions (e.g., Proposals%20Evaluation%20Report%20
Template%20.docx).
[Search Rules]
1. Call only one search tool (web_search_ ) per round. Choose the file type to search based
on the workflow needs.
2. After a search tool successfully downloads a file, you must use the terminal tool or
view_image tool to review the file content and evaluate whether it meets the scenario
requirements before proceeding to the next round.
3. If the scenario is too specific and no results are found, try broader keywords.
4. Execute at most <MAX_STEPS> rounds; return the result once suitable files are found or
the maximum number of rounds is reached.
[File Evaluation Criteria]
Evaluate file relevance across three dimensions:
1. Topic Match - Does the file’s main subject align with the scenario’s professional
domain and work content?
2. <sub>**</sub>Utility Match<sub>**</sub> - Can the file’s content type and data directly support the workflow
tasks? (e.g., a numerical dataset supports calculations; a technical manual supports
equipment assessment)
Highly Relevant : Satisfies all three dimensions above. The file directly provides the
specific data, standards, technical specifications, or documentation that the scenario’s
workflow requires.
Partially Relevant : Satisfies 1 dimensions but with limitations, OR serves as useful
background context. Common cases: the file covers a related subfield but needs
significant adaptation; the file contains background knowledge but lacks task-specific
details; the file was caught by a keyword hit in metadata or a cited reference rather
than its main body content.
<sub>**</sub>Irrelevant<sub>**</sub>: Fails on most dimensions - the file comes from a unrelated industry, or
serves a different purpose (e.g., management/process documents for a scientific research
scenario, or generic intro materials for an expert-level scenario).
[Return Format]
{
"highly_relevant_files": [
```

{   
"sandbox\_path": "Path of the file in the sandbox",   
"relevance": "Explanation of relevance to the scenario"   
}   
],   
"partially\_relevant\_files": [   
{   
"sandbox\_path": "Path of the file in the sandbox",   
"relevance": "Explanation of relevance to the scenario"   
}   
],   
"irrelevant\_files": [   
{   
"sandbox\_path": "Path of the file in the sandbox",   
"reason": "Reason for not meeting requirements"   
}   
]   
}   
[User]   
Please search for the required files based on the following workflow scenario requirements.   
[Scenario]   
Scenario: <SCENARIO\_TITLE>   
[Workflow]   
<NUMBERED\_WORKFLOW\_STEPS>   
[Required Files Description]   
<REQUIRED\_FILE\_TYPES\_AND\_PURPOSES>   
[File Save Path]   
Save all files to: /home/workspace/search\_files   
[Finalization Message]   
Based on the files already searched and inspected, return the result in JSON format.

## Listing 3: Task question and rubric generation prompt.

```ini
[System]
You are a professional productivity task question designer. Based on workflow scenarios and
retrieved files, design high-quality productivity questions and generate rubric scoring
criteria.
<sub>**</sub>File System<sub>**</sub>
‘/mnt/user-data/uploads‘ -- Relevant files have been uploaded here. Read directly from
this path.
- Each file may have a companion <sub>**</sub>preprocessed folder<sub>**</sub> (same name as the file, without
extension) containing ‘content.md‘ and extracted images. These are provided <sub>**</sub>only to
help you understand the file content<sub>**</sub> -- do NOT treat them as attachments for the
question.
‘/home/workspace‘ -- Scratch space for temp files and intermediate work. Not visible
externally.
‘/mnt/user-data/outputs‘ -- Copy selected <sub>**</sub>original files<sub>**</sub> (not preprocessed folders)
from ‘/mnt/user-data/uploads‘ here as question attachments. Don’t include other files.
[Your Core Tasks]
1. Thoroughly understand the content of the provided files.
2. Based on the workflow and file content, design a question that requires producing a real
deliverable.
4. Generate verifiable rubric scoring criteria.
[Question Design Principles]
```

You must analyze the files in detail before designing questions. Do not skip file analysis   
and jump directly to question design.   
Questions should reflect real-world work requirements, require multi-step reasoning and   
operations, and be closely related to the attachment content.   
Deliverable types: pdf, pptx, xlsx. Choose the deliverable type that best fits the task   
nature.   
Not all provided files need to be included, you can choose the most suitable files for the   
question.   
You can adjust and enrich the workflow appropriately based on the file content.   
<sub>\*\*</sub>Keep the question description concise<sub>\*\*</sub>: do NOT embed all details from the attachments   
into the question text. The question should set the task goal and context; solvers are   
expected to open the attachments and infer the specifics themselves.   
Questions should ideally test the model’s ability to extract information from complex file   
structures (e.g., nested tables, multi-sheet workbooks, structured spreadsheets, or   
hierarchically organized documents).   
[Rubric Rules]   
1. <sub>\*\*</sub>Source-backed and verifiable<sub>\*\*</sub>: Each criterion must clearly determine "satisfied/not   
satisfied"; the basis must come from the question description or attachment content. Do   
not fabricate requirements or values without a source.   
2. Content category (category=content): Assesses information retrieval, multi-step   
reasoning, and specific operations required to complete the task, e.g., extracting key   
data from attachments, calculating or summarizing according to specified logic,   
executing specific analysis steps.   
3. Form category (category=form): Assesses the structural and formatting requirements of   
the deliverable, e.g., file format, section structure, number of charts, naming   
conventions.   
[Return Format]   
After completing file analysis, first copy the selected attachments from /mnt/user-data/   
uploads to /mnt/user-data/outputs, then return the following JSON:   
{   
"question": {   
"title": "Question title",   
"description": "Detailed question description (background context, task requirements,   
delivery requirements)",   
"deliverable": {   
"type": "pdf|pptx|xlsx|docx",   
"filename": "Suggested filename"   
},   
"attachments": [   
{   
"filename": "Attachment filename (already copied to outputs directory)",   
"reason": "Reason for selecting this attachment"   
}   
]   
},   
"rubric": [   
{   
"criterion": "Specific verifiable scoring description",   
"category": "content|form",   
"points": 2,   
"source": "Source basis for the rubric and the approach needed to fulfill it"   
}   
You have at most <MAX\_STEPS> steps. Return the result when your task is complete.   
[User]   
Please design a productivity task question and generate rubric scoring criteria based on the   
following workflow scenario and retrieved files.   
[Job Title] <JOB\_TITLE>

[Scenario Description]   
Scenario title: <SCENARIO\_TITLE>   
[Workflow Description]   
<NUMBERED\_WORKFLOW\_STEPS>   
[Relevant Files] (Uploaded to /mnt/user-data/uploads, annotated with relevance type)   
Note: Preprocessed folders are for reference only to help you understand file content. Only   
original files may be used as question attachments.   
<FILE\_PATHS\_RELEVANCE\_AND\_PREPROCESSED\_FOLDERS>   
Please start analyzing the files and designing the question, then return the result in JSON   
format.

Listing 4: Expert solving prompt.  
```ini
[System]
You are a capable, resourceful AI Agent. You cut through ambiguity, plan before acting, and
deliver high-quality results.
<sub>**</sub>File System<sub>**</sub>
‘/mnt/user-data/uploads‘ - User-uploaded files. Read directly from this path.
‘/home/workspace‘ - Scratch space for temp files and intermediate work. Not visible to the
user.
‘/mnt/user-data/outputs‘ - Save final deliverables here before replying. Files not copied
here are invisible to the user.
You have at most <MAX_STEPS> steps to complete the task.
[User]
Please complete the following work task.
[Task Title]
<TASK_TITLE>
[Task Description]
<TASK_INSTRUCTION>
[Deliverable]
File type: <DELIVERABLE_TYPE>
Filename: <DELIVERABLE_FILENAME>
[Input Attachments]
<ATTACHMENT_PATHS_AND_DESCRIPTIONS>
Input attachments have been uploaded to the /mnt/user-data/uploads/ directory.
Please first review the content of the input attachments, then complete the task as required
. Save the output file to the /mnt/user-data/outputs/ directory.
```

## G Seed Task Example: Lawyers

As an example of the seed task source, we list the task statements for the Lawyers occupation from the O\*NET seed task file.   
The occupation code is 23-1011.00.

1. 3774, Core. Interpret laws, rulings and regulations for individuals and businesses.

2. 3775, Core. Analyze the probable outcomes of cases, using knowledge of legal precedents.

3. 3778, Core. Gather evidence to formulate defense or to initiate legal actions by such means as interviewing clients and witnesses to ascertain the facts of a case.

4. 3779, Core. Represent clients in court or before government agencies.

5. 3777, Core. Evaluate findings and develop strategies and arguments in preparation for presentation of cases.

6. 3773, Core. Advise clients concerning business transactions, claim liability, advisability of prosecuting or defending lawsuits, or legal rights and obligations.

7. 3780, Core. Examine legal data to determine advisability of defending or prosecuting lawsuit.

8. 20877, Core. Prepare, draft, and review legal documents, such as wills, deeds, patent applications, mortgages, leases, and contracts.

9. 3783, Core. Study Constitution, statutes, decisions, regulations, and ordinances of quasi-judicial bodies to determine ramifications for cases.

10. 3786, Core. Negotiate settlements of civil disputes.

11. 3789, Core. Supervise legal assistants.

12. 23992, Core. Negotiate contractual agreements.

13. 3787, Core. Confer with colleagues with specialties in appropriate areas of legal issue to establish and verify bases for legal proceedings.

14. 3788, Core. Search for and examine public and other legal records to write opinions or establish ownership.

15. 3790, Core. Perform administrative and management functions related to the practice of law.

16. 3776, Supplemental. Present and summarize cases to judges and juries.

17. 3781, Supplemental. Select jurors, argue motions, meet with judges, and question witnesses during the course of a trial.

18. 3782, Supplemental. Present evidence to defend clients or prosecute defendants in criminal or civil litigation.

19. 3792, Supplemental. Probate wills and represent and advise executors and administrators of estates.

20. 3785, Supplemental. Prepare legal briefs and opinions, and file appeals in state and federal courts of appeal.

21. 3791, Supplemental. Act as agent, trustee, guardian, or executor for businesses or individuals.

22. 3793, Supplemental. Help develop federal and state programs, draft and interpret laws and legislation, and establish enforcement procedures.

## H Example

We include one generated example from the synthesis pipeline. The generated task is titled “Election Contest Legal Analysis and Litigation Strategy Memorandum.” The original corpus for this example comes from the first sample in the DESIGNER sample set (Liu et al. 2025), following the corpus source described in Section 3.

## Listing 5: Example workflow.

[Workflow]   
1. Research and interpret relevant election contest statutes and regulations.   
2. Examine election records and certification documents to identify potential grounds for   
contesting the results.   
3. Analyze legal precedents from similar contested elections.   
4. Evaluate legal arguments, including equal protection claims and Voting Rights Act   
challenges.   
5. Develop a litigation strategy and prepare legal briefs for potential state and federal   
court proceedings.

## Listing 6: Example question description.

Background Context:   
You are a senior attorney at a law firm specializing in election law. Your client, a losing   
candidate in a recent Virginia General Assembly election, is considering contesting the   
election results. The certified results show your client lost by 0.8% of the total votes   
cast. Additionally, there are concerns about mail-in ballots that were received after   
election day but postmarked before election day being counted, similar to the issue in   
Bost v. Illinois State Bd. of Elections.   
Task Requirements:   
Prepare a comprehensive legal memorandum analyzing the viability of an election contest and   
developing a litigation strategy. Your memorandum must:   
1. Analyze the Virginia recount and contest statutes to determine:   
- Whether your client is eligible to request a recount based on the margin of victory   
- The specific filing deadlines and procedural requirements for initiating a recount   
petition   
- The appropriate court venue for filing the petition   
- The bond requirements for contesting a General Assembly election   
2. Apply the precedent from Bost v. Illinois State Bd. of Elections (2026) to assess:   
- Whether your client has standing to challenge the counting of mail-in ballots received   
after election day

The legal arguments supporting standing based on the Supreme Court’s majority opinion   
- Potential counterarguments based on the dissenting opinion   
- How the concurrence’s pocketbook injury theory might strengthen or weaken your position   
3. Develop a litigation strategy that addresses:   
- The sequence of legal actions (recount petition followed by contest, or direct contest)   
- Specific grounds for contest under Virginia law that would have probable impact on the   
election outcome   
- Timeline for all required filings and proceedings   
- Evidence preservation and discovery considerations   
Delivery Requirements:   
Prepare a formal legal memorandum in PDF or DOCX format with the following sections:   
Executive Summary (maximum 1 page)   
Legal Framework Analysis (Virginia recount and contest statutes)   
Standing Analysis (application of Bost precedent)   
Litigation Strategy and Recommendations   
Timeline and Procedural Checklist   
Conclusion   
The memorandum should be professionally formatted with proper legal citations, clear   
headings, and actionable recommendations.

Listing 7: Example rubric.
<table><tr><td>[Rubric]</td><td></td><td></td><td></td><td>1. criterion: Correctly identifies that a recount petition is permissible because the 0.8%</td></tr><tr><td></td><td>category: content</td><td>margin is within the 1% threshold specified in Virginia law</td><td></td><td></td></tr><tr><td></td><td></td><td>source: Virginia Recounts and Contests document states: &quot;A petition for a recount may be filed only if there is a difference of not more than one percent (1%) of the total vote</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>2. criterion: Accurately states the 10-day filing deadline for recount petitions from the</td><td></td><td></td><td></td></tr><tr><td></td><td>date of certification</td><td></td><td></td><td></td></tr><tr><td></td><td>category: content</td><td>source: Virginia document states: &quot;a petition for a recount of an election, other than an</td><td></td><td></td></tr><tr><td></td><td>State Board or the electoral board certifies the result&quot;</td><td>election for presidential electors, shall be filed within 10 days from the day the</td><td></td><td></td></tr><tr><td></td><td>3. criterion: Identifies the correct venue for filing a General Assembly election contest</td><td></td><td></td><td></td></tr><tr><td></td><td>category: content</td><td>petition (circuit court where the challenged candidate resides)</td><td></td><td></td></tr><tr><td></td><td>in the circuit court of the county or city in which the candidate being challenged</td><td>source: Virginia document states: &quot;For non-statewide offices, the petition must be filed</td><td></td><td></td></tr><tr><td></td><td>resides&quot;</td><td></td><td></td><td></td></tr><tr><td></td><td>election as $100 per precinct</td><td>4. criterion: Correctly identifies the bond requirement for contesting a General Assembly</td><td></td><td></td></tr><tr><td></td><td>category: content source: Virginia document states: &quot;the contestant must post a surety bond in the amount</td><td></td><td></td><td></td></tr><tr><td></td><td>of $100 for each precinct contained in whole or in part of the district contested&quot;</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>5. criterion: Accurately explains the standing holding from Bost v. Illinois State Bd. of Elections that candidates have standing to challenge vote-counting rules based on their</td><td></td><td></td><td></td></tr><tr><td></td><td>interest in a fair electoral process</td><td></td><td></td><td></td></tr><tr><td></td><td>category: content</td><td></td><td></td><td></td></tr><tr><td></td><td>source: Bost case syllabus states: &quot;As a candidate for office, Congressman Bost has</td><td></td><td></td><td></td></tr><tr><td></td><td>standing to challenge the rules that govern the counting of votes in his election&quot;</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>substantial risk of election loss to establish standing</td><td>6. criterion: Correctly identifies that under Bost, candidates do not need to show a</td><td></td><td></td><td></td></tr></table>

```csv
source: Bost majority opinion states: "Candidates do not need to show a substantial risk
that a rule will cause them to lose the election or prevent them from achieving a
legally significant vote threshold in order to have standing"
7. criterion: Identifies and explains the reputational harm theory from Bost as a basis for
standing (erosion of public confidence undermines political legitimacy)
category: content
source: Bost opinion states: "Rules that undermine the integrity of the electoral process
also undermine the winner’s political legitimacy" and "reputational harms are classic
Article III injuries"
8. criterion: Addresses the dissenting opinion’s counterargument that the interest in fair
elections is a generalized grievance common to all voters
category: content
source: Bost dissent states: "the interest in a fair electoral process is common to all
members of the voting public" and candidates cannot sue based only on "generalized
grievances about the conduct of government"
9. criterion: Explains Justice Barrett’s concurrence regarding pocketbook injury and
mitigation costs as an alternative standing theory
category: content
source: Bost concurrence states: "Congressman Bost has standing because he has suffered a
traditional pocketbook injury" from costs incurred "to mitigate or avoid the
substantial risk of harm"
10. criterion: Correctly identifies the notice deadline for contesting a General Assembly
election (30 days after election or 3 days after recount conclusion, whichever is later)
category: content
source: Virginia document states: "Notice must be submitted within thirty days following
the date of the election or three days after the conclusion of a recount, whichever is
later"
11. criterion: Identifies the required grounds for contest (eligibility objections or
conduct/results objections with probable impact on outcome)
category: content
source: Virginia document states grounds must include "Objections to the contestee’s
eligibility based on specific allegations" or "Objections to the conduct or results of
the election, accompanied by specific allegations that, if proven true, would have a
probable impact on the outcome"
12. criterion: Provides a logically sequenced litigation strategy addressing whether to
pursue recount first or proceed directly to contest
category: content
source: Requires multi-step reasoning based on Virginia document provisions about
recount procedures (Section 15.2) and contest procedures (Section 15.3), including the
provision that a candidate who loses after recount may contest
13. criterion: Includes a timeline or procedural checklist covering all key deadlines (
recount petition, preliminary hearing, contest notice, contestee answer, depositions)
category: content
source: Requires synthesis of multiple deadlines from Virginia document: 10 days for
recount petition, 5-7 days for preliminary hearing, 30 days/3 days for contest notice,
10 days for contestee answer, 20-30 days for depositions
14. criterion: Memorandum includes all six required sections (Executive Summary, Legal
Framework Analysis, Standing Analysis, Litigation Strategy, Timeline/Checklist,
Conclusion)
category: form
source: Question delivery requirements specify six required sections
15. criterion: Document is properly formatted as a legal memorandum with clear headings,
professional formatting, and proper legal citations
category: form
source: Question delivery requirements specify "formally formatted with proper legal
citations, clear headings, and actionable recommendations"
```

16. criterion: Executive Summary is present and does not exceed 1 page   
category: form   
source: Question delivery requirements specify "Executive Summary (maximum 1 page)"   
17. criterion: Legal citations reference both the Virginia Code sections and the Bost case   
with appropriate pinpoint citations   
category: form   
source: Question requires "proper legal citations" and attachments contain specific Code   
sections (Sections 24.2-800 through 24.2-814) and Bost case citations   
18. criterion: Provides actionable recommendations rather than merely summarizing legal   
principles   
category: content   
source: Question delivery requirements specify "actionable recommendations"

The source workflow starts from a legal scenario involving a contested Virginia Attorney General election and asks the model to move from statutory interpretation and record review to precedent analysis, argument evaluation, and litigation planning. The generated question instantiates this workflow as a solver-facing task: it asks for a comprehensive legal memorandum with six required sections, including an executive summary, legal framework analysis, standing analysis, litigation strategy and recommendations, a timeline and procedural checklist, and a conclusion. The substantive task requires the solver to analyze Virginia recount and contest statutes, apply Bost v. Illinois State Board of Elections to candidate standing for challenging latecounted mail-in ballots, and recommend a litigation sequence covering recount, contest, evidence preservation, and discovery considerations.

The rubric evaluates whether the response identifies the 1% recount threshold, the 10-day recount petition deadline, the correct venue for a General Assembly contest, the bond requirement, and the contest notice deadline. It also checks whethe the response explains the majority, dissenting, and concurring standing theories in Bost, provides a coherent litigation strategy, includes the requested memorandum sections, uses proper legal citations, and gives actionable recommendations rather than only summarizing legal principles.