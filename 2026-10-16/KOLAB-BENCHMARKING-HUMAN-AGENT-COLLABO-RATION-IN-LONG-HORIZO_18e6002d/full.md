# KOLAB: BENCHMARKING HUMAN-AGENT COLLABO-RATION IN LONG-HORIZON WORKFLOWS

Shirley Anugrah Hayati<sup>1</sup>, Patricia Anugrah Setiani<sup>1</sup>, Ruizi Wang<sup>1</sup>, Zhouyi Li<sup>2</sup>, Khanh Chi Le<sup>1</sup>, Young-Jun Lee<sup>1</sup> & Dongyeop Kang<sup>1</sup>

<sup>1</sup>Department of Computer Science, <sup>2</sup>Department of Economics

University of Minnesota

{hayat023, dongyeop}@umn.edu

## ABSTRACT

When humans collaborate with AI agents on long-horizon tasks, what happens to the agents’ contributions? For solving long-horizon tasks, humans place greater importance on ongoing interaction with AI agents (e.g., revising plans) than on an agent’s ability to complete the task in a single attempt. However, most agentic benchmarks have focused more on automation. Strong performance on such benchmarks does not imply effective real-world collaboration. Professional workflows offer a useful setting, as they often involve interdependent steps. To address this gap, we introduce KOLAB, a new dataset of 115+ hours of screen recordings of 50 professionals from 40 occupations collaborating with agents on realistic, long-horizon problems. KOLAB also contains participants’ collaboration and output quality judgments. Our analysis reveals diverse collaboration patterns, highlighting the importance of when and how professionals coordinate with agents. A collaborative agent must do two things that automation benchmarks never test: anticipate whether experts will adopt, modify, or reject its contribution once it leaves the chat window to guide its next steps, and assess collaboration quality from the unfolding workflow without constant expert feedback. Our benchmark operationalizes both: (1) forecasting how professionals use agent contributions at diverse workflows, and (2) verifying their judgments of collaboration satisfaction and output quality. Findings show that frontier models, both agentic and non-agentic, overestimate collaboration with experts and struggle to distinguish between high- and low-quality collaboration. Longer workflow context does not consistently improve these predictions. As frontier agents cannot yet understand long-horizon collaboration with experts, we position KOLAB as a valuable dataset for studying human–agent collaboration grounded in real professional workflows.<sup>1</sup>

## 1 INTRODUCTION

As AI agents become more capable of complex reasoning and knowledge work, recent research has examined whether they can complete such tasks autonomously in a single attempt, given a single instruction (Patwardhan et al., 2026; Vidgen et al., 2026; Zhu et al., 2026; Sun et al., 2026b; Merrill et al., 2026; Xu et al., 2026; Xie et al., 2024). However, real-world problem solving may not follow this autonomous setting: human goals can evolve, earlier decisions shape subsequent work, and errors may require intervention before they propagate (Tack et al., 2026). In professional work, for example, workers formulate requirements, produce intermediate artifacts, revise decisions, and move between agents and domain-specific tools. Thus, strong autonomous task performance may not translate into effective human-agent collaboration (Zou et al., 2025; Mozannar et al., 2025; Shen et al., 2026). Despite the need for studying effective human-agent collaboration, existing datasets provide limited grounding in real collaborative workflows. They are often not publicly released, rely on predefined tasks constructed or sourced by researchers, or focus on a single professional domain such as software development (Wang et al., 2025; Chen et al., 2026; Shao et al., 2026b). As a result, benchmarking long-horizon human-agent collaboration workflows remains underexplored.

![](images/f78f73a9e3802d2cf26936d6ebb713f94851d8f5650f4bee2db8e981fe50b695.jpg)  
Figure 1: An example human-agent collaborative workflow for an architect. Each step is a subgraph of human and agent actions; annotations link a source node, the target AI output it triggers, and the human response to it (here, MODIFY and ADOPT).

To address this gap, we introduce KOLAB, a dataset of screen recordings capturing human-agent collaboration on long-horizon professional problems. We represent each workflow as a graph of semantically meaningful human and agent actions and their dependencies (Figure 1), allowing us to understand when and how collaboration occurs in the workflow. In this example, an architect completes a long-horizon task in five steps, and agent outputs can influence later steps. A task delegated in Step 2 (source node) produces information (target AI node) that is later modified and used in Step 3 (MODIFY). Human responses to agent outputs may also occur outside the chat interface. The architect ultimately adopts and executes the resulting plan in Revit, an action not directly observable by the agent.

Our analysis reveals that human-agent collaboration workflow varies in when and how professionals engage with AI agents. Burstier workflows are associated with longer periods of independent work before interacting with AI, suggesting that professionals may organize their collaboration around periods of independent work rather than continuous AI interaction. Workflows with more concurrent human-agent activities are associated with lower collaboration satisfaction, suggesting that parallel activities may introduce additional coordination costs.

Beyond characterizing collaboration, we ask whether frontier multimodal large language models (MLLMs) and agents can understand their own place in a professional’s workflow. We frame this as two capabilities that a deployed collaborative agent needs but that autonomous benchmarks never measure.

Anticipating the fate of a contribution (RQ1). In long-horizon work, an agent’s output is rarely the end of the story: it is adopted, modified, or rejected, often outside the chat interface (e.g., an architect incorporating an agent’s plan in Revit). An agent that can recognize what happened to its output after the fact (1a) could turn naturalistic screen recordings into feedback signals for training and for auditing deployed agents, without asking the expert to label anything. An agent that canforecast this response at delivery time (1b)—before the human reacts—could act proactively: asking a clarifying question when rejection is likely, flagging the parts it expects to be edited, or escalating to the human when its contribution carries risk that would propagate to later steps (Oh et al., 2026). Because such forecasts depend on how much of the collaboration the agent remembers, we vary the available history from the preceding steps (1c) to the entire workflow so far (1d), which directly informs how much context a long-running or computer-use agent must retain to model its collaborator.

Judging the collaboration itself (RQ2). Expert ratings of collaboration satisfaction (2a) and output quality (2b) are the ground truth for whether human-agent collaboration works, yet they are expensive to collect and unavailable at deployment. If an MLLM could recover these ratings from the workflow alone, it could serve as a process-level reward model for training collaborative agents, as an automatic monitor that detects friction or low-quality outcomes in deployed systems, and as a scalable evaluator for professional tasks where no reference answer exists. Crucially, such a verifier must separate genuinely successful interactions from dissatisfying ones, rather than echoing the predominantly positive ratings.

Table 1: Comparison of KOLAB with long-horizon, multi-domain workflow benchmarks. “H-AI Collab.” indicates the presence of human-agent collaborative workflows; “Software Diversity” refers to tasks spanning multiple software applications; “Modalities” lists types of publicly released data.
<table><tr><td>Dataset</td><td>Human</td><td>H-AI</td><td>Software</td><td></td><td>Workflow Collab. Diversity #Domains Modalities</td></tr><tr><td>Agents&#x27; Last Exam (Sun et al., 2026b)</td><td>x</td><td>x</td><td></td><td>55</td><td>Image, text, files</td></tr><tr><td>Workflow-GYM (Zhu et al., 2026)</td><td>√</td><td>x</td><td>√</td><td>6</td><td>Image</td></tr><tr><td>WeaveBench (Li et al., 2026)</td><td>x</td><td>x</td><td>√</td><td>8</td><td>Image, text</td></tr><tr><td>TheAgentCompany (Xu et al., 2026)</td><td>x</td><td>x</td><td>x</td><td>6</td><td>Text</td></tr><tr><td>GDPVal (Patwardhan et al., 2026)</td><td>x</td><td>x</td><td>√</td><td>44</td><td>Text, files</td></tr><tr><td>DeskCraft (Wang et al., 2026)</td><td>x</td><td>√</td><td>√</td><td>5</td><td>Image</td></tr><tr><td>KOLAB (Ours)</td><td></td><td></td><td></td><td>40</td><td>Video, text, files</td></tr></table>

We observe that frontier multimodal agents still struggle to anticipate how professionals use agent outputs and to recover their judgments of collaboration quality. For example, Gemini 3.1 Pro and GPT-6 achieve only 39.1% and 51.3% F1 in forecasting adoption behavior, while their predictions of collaboration satisfaction correlate weakly with professional ratings (Spearman’s $\rho = . 0 8$ and .38). Providing longer workflow context or additional modalities does not consistently improve these predictions. Further analysis reveals systematic biases: models often favor certain adoption labels and overpredict high ratings, limiting their ability to distinguish successful from poor collaboration. These findings suggest that current models may rely on their own prediction tendencies rather than reliably understanding professionals’ behaviors and judgments from the workflow. These results highlight an important challenge for developing collaborative agents that can anticipate human decisions and adapt to how their contributions are actually used.

Our main contributions are as follows.

• We present KOLAB, a new dataset of 115 hours of screen-recorded human-agent workflows from 50 professionals solving their own real-world problems, comprising 508 steps across 90+ softwares. Each workflow includes a final deliverable with step-level human ratings of collaboration experience and output quality.

• We analyze human-agent collaboration dynamics using three metrics: collaboration burstiness, workflow topology, and AI retention rate.

• We benchmark frontier MLLMs on their ability to understand human-agent collaborative workflows when solving professional problems.

• We uncover systematic MLLM failure modes, revealing prediction biases that vary by model and input modality.

## 2 RELATED WORK

Long-horizon agent and workflow benchmarks. Autonomous agent benchmarks cover desktop applications (Xie et al., 2024; Yuan et al., 2026), enterprise software (Dai et al., 2026), hybrid GUI/- CLI and scientific computer use (Li et al., 2026; Jang et al., 2026; Wang et al., 2026; Zhu et al., 2026; Sun et al., 2026a), professional deliverables (Patwardhan et al., 2026; Sun et al., 2026b; Xu et al., 2026), and long-running agent behavior (Maldaner et al., 2026). Some include human baselines but do not release human action trajectories. Workflow-GYM (Zhu et al., 2026) collects expert workflow trajectories but uses them as demonstrations to evaluate autonomous agents. Human-only process datasets such as ScholaWrite (Le et al., 2026) and LawFlow (Das et al., 2025) capture expert work in a single domain (scholarly writing and legal reasoning) without agent collaboration. Among the benchmarks in Table 1, KOLAB uniquely combines multimodal trajectories of professionals col laborating with agents on self-defined problems with step-level human ratings.

Human-agent collaborative workflows. Prior work argues that autonomous task performance does not ensure effective collaboration (Shen et al., 2026), and that the role humans play relative to an AI system shapes the interaction (Wan et al., 2022); indeed, varying LLM access measurably changes how people work (Christenson et al., 2026). Existing evaluations often rely on simulated users (Wu et al., 2026; Yao et al., 2024; Vijayvargiya et al., 2026; Qian et al., 2025) or human intervention in short shopping and web browsing tasks (Huq et al., 2025; Mozannar et al., 2025). Collaborative Gym (Shao et al., 2026a) supports real or simulated users but covers only three predefined tasks. CollabSkill (Shao et al., 2026b) assigns professionals problems from existing datasets; Chen et al. (2026) study interaction-level satisfaction with software-engineering agents; and Wang et al. (2025) compare separate human and agent workflows. Our study setup is most similar to that of Hayati et al. (2026) who investigate human-agent collaboration in interdisciplinary research using only chat logs. Meanwhile, our work also captures the complete workflow, including how professionals act on agent contributions in their work that may happen outside the chat interface.

## 3 KOLAB: A MULTIMODAL, MULTIDOMAIN DATASET OF HUMAN-AGENT COLLABORATIVE WORKFLOWS ON LONG-HORIZON PROBLEMS

KOLAB is built to understand human-agent collaborative workflows when humans solve longhorizon problems.

Problem criteria. To ensure that the collected workflows represent substantive professional problem solving rather than generic computer-use activities, we required each problem to satisfy four criteria:

• Real-world. The problem must originate from the professional’s actual work. Professionals were not given tasks from existing workflow datasets or scenarios.

• Domain-specific. The problem must require the application of professional knowledge, skills, or judgment from the participant’s field. Tasks that could be completed without domain expertise, such as routine personal activities or generic information seeking, were excluded.

• Complex and long-horizon. The problem must require a sequence of interdependent steps to achieve its final goal. Participants were asked to plan a workflow requiring at least one hour of work and consisting of at least five steps, with each step involving multiple computer actions or decisions.

• Deliverable-oriented. The problem must result in a concrete final deliverable. Each step must also produce an intermediate output that can be evaluated.

Data collection. We recruited 50 industry professionals via word of mouth, Prolific, and Upwork. Each participant received a compensation of \$100 with a bonus for extra time. Figure 2 shows our participants’ domains with details in Appendix A.1. Professionals interacted with an agentic AI, Claude (Anthropic, 2026), for two 1-hour recording sessions.<sup>2</sup> Each participant planned their workflow beforehand to ensure that their problem satisfied our criteria. Then they updated the workflow document, rated their collaboration experience and output quality, and filled out the post-interaction survey in the Appendix A.12.

50 professionals, 40 occupations  
![](images/acb3121a4ca57916475d6307a1b56608e8eaf33b85eb80ddc540200d8feca247.jpg)  
Figure 2: KOLAB domain distributions.

Dataset statistics. KOLAB covers 100 problem sets and human-agent collaborative workflows conducted by professionals from 10 sectors of 40 occupations (Figure 2). The post-interaction survey also measured collaboration satisfaction and output quality on a 1 (low) to 5 (high) scale.<sup>3</sup> Collaboration satisfaction was rated 5 in 54.9% of sessions and 4 in 29.7%, with 15.3% rated 3 or below. Similarly, output quality was rated 5 in 53.5% of sessions and 4 in 31.0%.

Table 2: Real examples of ADOPT, MODIFY, and REJECT responses from KOLAB, drawn from a music producer workflow, architecture workflow, and a video-production workflow respectively.
<table><tr><td>Response</td><td>Source Node Action</td><td>Target AI Node Action</td><td>Current Node Action</td><td>Software</td></tr><tr><td>ADOPT</td><td>Request Afrobeats/techno back- ground music in ElevenLabs (67:33–68:30)</td><td>ElevenLabs generates Afrobeats instrumental candidates (69:50– 70:04)</td><td>Capture the ElevenLabs track and import it into the DaVinci Resolve project (72:05–74:36)</td><td>ElevenLabs, DaVinci Resolve</td></tr><tr><td>MODIFY</td><td>Upload the existing shell plan and request a full room layout (24:10–25:28)</td><td>Claude delivers a dimensioned floor-plan layout for the lobby/s- torage circulation (51:21–56:20)</td><td>Rebuild walls/partitions in Revit, keeping the liked zoning but fixing lobby/storage circulation (58:40– 72:59)</td><td>Revit</td></tr><tr><td>REJECT</td><td>Ask Kling AI to generate a Zanzibar schoolyard video clip (36:30-39:04)</td><td>Kling AI generates the requested video clip (39:45–41:17)</td><td>Judge the clip&#x27;s audio unusable and abandon it, out of credits to it- erate further (41:17–42:06)</td><td>Kling AI</td></tr></table>

Table 3: Summary statistics of KOLAB for videos, workflow graphs, and human action annotations.
<table><tr><td colspan="2">Video</td><td colspan="2">Graph</td><td colspan="2">Annotation (692 labels)</td></tr><tr><td>Total duration</td><td>115 hrs</td><td>Workflow steps</td><td>508</td><td>ADOPT</td><td>65.9% (456)</td></tr><tr><td>Avg. duration/session</td><td>69 mins</td><td>Avg. steps/session</td><td>5.24</td><td>MODIFY</td><td>19.1% (132)</td></tr><tr><td></td><td></td><td>Avg. nodes/step</td><td>8.68</td><td>REJECT</td><td>15.0% (104)</td></tr></table>

Data release. We publicly release KOLAB with seven components: (1) screen-recording videos for all sessions; (2) the workflow documents written by the professionals, including goal, context, and step names/actions; (3) the human–agent chat logs, including agent tool calls; (4) optional input and output files of each workflow; (5) human ratings for collaboration experience and output quality, along with step-level output rubrics written by professionals; (6) the workflow graph per session, with step alignment, timestamps, and adoption labels; and (7) post-session survey responses.

## 4 UNDERSTANDING HUMAN-AGENT COLLABORATION DYNAMICS

## 4.1 DATA PROCESSING AND ANNOTATION

To make the workflow structured, each step in the workflow is represented as a temporal graph, where each node $v ~ \in ~ V$ corresponds to a human or agent action at a time interval $\tau ( v ) \ =$ $[ t _ { s } ( v ) , t _ { e } ( v ) ] \subset \mathbb { R } _ { \geq 0 }$ . The research team and professionals refined these graphs together. An example graph for one step is shown in Figure 12 in the Appendix. A node can branch when activities occur at overlapping timestamps and loops happen when the same action is repeated. Formally, each workflow is represented as follows:

$$
G = ( V , E , a , w , \tau ) ,\tag{1}
$$

where V is a set of human (H) or AI (A) action nodes; $E \subseteq V \times V$ is the set of directed edges, with $( u , v ) \in E$ indicating that action v directly follows from action u.

After graph construction, two human annotators labeled how humans incorporated agent outputs. A human node is identified as a source node if its action triggered the agent’s generation (e.g., “design a floor plan”). An agent node is the target AI node if it generates an output that is subsequently acted on by the participant. The human node that acts on this AI output is labeled as follows.

• ADOPT: The agent output is observably incorporated into the workflow as-is, such as directly using an agent-generated design or adding a suggested paper to a literature review.

• MODIFY: The agent output is incorporated with modifications, e.g., the human revises agentgenerated text before using it.

• REJECT: The agent output is not incorporated after being reviewed, such as when the human discards it or clearly expresses a negative sentiment and proceeds with a different approach.

As seen from real examples in Table 2, professionals evaluate agent contributions and act on softwares outside the AI interface. A summary of KOLAB is provided in Table 3. In total, KOLAB contains 100 workflow videos. Each video has corresponding human-curated temporal graphs annotated with AI adoption labels, yielding a total of 692 instances for benchmarking MLLMs.

![](images/40739060752a7bc8c589ca9147b1bd37a06767a58215801f77f000db916a53b7.jpg)  
(a) Collaboration burstiness

![](images/29cf4b9dc041937d30fde5a96b53fc294dfe56d1ea25f9846876c3f5dd625735.jpg)  
(b) Workflow topology

![](images/554d1277ec529ecb0271506bdc622e9ce2134ab84d4418909485abb8614a1277.jpg)  
(c) AI retention rate  
Figure 3: Collaboration dynamics in KOLAB. (a) Burstiness vs. solo lead-in; (b) workflow branching vs. collaboration rating; (c) timing of ADOPT, MODIFY, and REJECT responses across sectors. \*\*\* refers to $p < 0 . 0 0 1$ and \* is $p < 0 . 0 5$

## 4.2 COLLABORATION DYNAMICS METRICS

Collaboration burstiness. Human-agent interactions are not uniformly distributed over time. Humans may work with an agent steadily or in short, intensive episodes separated by longer periods of independent work. To capture this temporal rhythm, following Goh & Barabási (2008), we estimate collaboration burstiness as $B = \frac { \sigma _ { \tau } - \mu _ { \tau } } { \sigma _ { \tau } + \mu _ { \tau } } \in [ - 1 , 1 ]$ , where $\mu _ { \tau }$ and $\sigma _ { \tau }$ are the mean and standard deviation of the inter-event intervals. $B  1$ indicates bursty interactions, $B  - 1$ indicates regular interactions, and $B = 0$ corresponds to a Poisson-like process.

Workflow topology. Motivated by graph-theoretic representations of complex systems (Bondy & Murty, 2008; Boccaletti et al., 2006; Newman et al., 2011), we characterize workflow topology to capture the extent to which humans work concurrently while the agent generates an output. We operationalize this as the proportion of branch nodes with out-degree $> \mathrm { ~ 1 ~ } ( B r )$ relative to the number of edges $( | E | )$ as a workflow topology $\begin{array} { r } { W = \frac { B r } { | E | } } \end{array}$

AI retention rate. To measure how agent outputs are retained in subsequent work (Solé et al., 2026), we define AI retention through adoption and modification rates:

$$
R _ { A } = \frac { N _ { \mathrm { A D o p r } } } { N _ { \mathrm { A D o p r } } + N _ { \mathrm { M o D I F Y } } + N _ { \mathrm { R E B E C } } } , R _ { M } = \frac { N _ { \mathrm { M o D I F Y } } } { N _ { \mathrm { A D o p r } } + N _ { \mathrm { M o D I F Y } } + N _ { \mathrm { R E B E C } } } .
$$

These capture the proportion of agent outputs incorporated directly or with modification.

## 4.3 ANALYSIS

Using these three metrics, we analyze how human-agent collaboration varies in its temporal rhythm (B), topology of concurrent work (W), and integration of agent contributions (R). Our analyses are intended to describe the collaborative patterns observed in KOLAB, rather than as population-level comparisons across professional domains. Figure 3 shows the results.

#1: Burstier workflows reflect more planning before interacting with the agent. Collaboration burstiness was relatively consistent within individuals across two sessions (Spearman $\rho = 0 . 5 9 )$ . We define solo lead-in as the proportion of a step completed before the first AI interaction, averaged over all steps in a session. This reflects the proportion of time a person spends working independently before first interacting with AI. For example, if the first AI interaction occurs 4 minutes into a 10- minute step, the solo lead-in is 40%. As shown in Figure 3a, burstier workflows had longer solo lead-ins $( \rho = 0 . 3 5 , p < 0 . 0 0 1 )$ . Consistent with this pattern, participants who reported having a clearer plan before using AI in the post-interaction survey also had longer solo lead-ins $( \rho = 0 . 5 3$ $p < 0 . 0 0 1 )$ .

#2: Workflows with branches are associated with lower perceived collaboration experience. Workflow branching (W) is negatively associated with collaboration ratings (Figure 3b) $( \rho = - 0 . 2 4 $ $p < 0 . 0 5 )$ . Although modest, this association suggests that concurrent human and agent activity may introduce coordination costs.

Table 4: Benchmark tasks. $T _ { x } ^ { \mathrm { s t a r t } }$ and $T _ { x } ^ { \mathrm { e n d } }$ denote the start and end timestamps of event or segment x, respectively; x can refer to the source node, target AI node, human adoption node, or the full workflow.
<table><tr><td>Experiment Task</td><td>Observed Workflow</td><td>Response Visible?</td><td>Prediction</td></tr><tr><td>RQ1: AI adoption</td><td></td><td></td><td></td></tr><tr><td>1a: Recognition</td><td> $T _ { \mathrm { s o u r c e } } ^ { \mathrm { s t a r t } }  T _ { \mathrm { a d o p t i o n } } ^ { \mathrm { e n d } }$ </td><td>L</td><td></td></tr><tr><td>1b: Forecasting</td><td> $T _ { \mathrm { s o u r c e } } ^ { \mathrm { s t a r t } }  T _ { \mathrm { A I } } ^ { \mathrm { e n d } }$ </td><td>x</td><td>ADOPT MODIFY</td></tr><tr><td>1c: Step-N forecasting</td><td> $T _ { \mathrm { s t e p } } ^ { \mathrm { s t a r t } }  T _ { \mathrm { A I } } ^ { \mathrm { e n d } }$ </td><td>x</td><td>REJECT</td></tr><tr><td>1d: Workflow-level forecasting</td><td> $T _ { \mathrm { w o r k f l o w } } ^ { \mathrm { s t a r t } } \to T _ { \mathrm { A I } } ^ { \mathrm { e n d } }$ </td><td>X</td><td></td></tr><tr><td>RQ2: Human rating</td><td></td><td></td><td></td></tr><tr><td>2a: Collaboration satisfaction</td><td> $T _ { \mathrm { s t e p } } ^ { \mathrm { s t a r t } }  T _ { \mathrm { s t e p } } ^ { \mathrm { e n d } }$ </td><td></td><td>N/A</td></tr><tr><td>2b: Output quality</td><td> $T _ { \mathrm { s t e p } } ^ { \mathrm { s t a r t } }  T _ { \mathrm { s t e p } } ^ { \mathrm { e n d } }$ </td><td>N/A</td><td>Rating 1–5</td></tr></table>

#3: AI retention patterns vary across professional domains. As shown in Figure 3c, most domains predominantly adopt agent outputs as-is, with adoption rates ranging from 57% in Legal to 79% in Sciences and Computer & Mathematical work. Legal workflows stand out for modification: 41% of agent outputs were modified, about twice the rate of any other domain (Arts & Sports, the next highest, is 22%), consistent with lawyers’ reluctance to adopt AI outputs as-is (Schwarcz et al., 2026). This shows work style variation in how professional domains integrate agent contributions.

## 5 EXPERIMENTS AND RESULTS

## 5.1 BENCHMARK TASKS

Our benchmark tasks aim to answer the following research questions as summarized in Table 4.

• RQ1: (AI Adoption) How well can frontier MLLMs forecast how humans use agent contributions in long-horizon professional workflows?

• RQ2: (Human Rating) How well can frontier MLLMs predict human ratings of collaboration satisfaction and output quality given long-horizon human-agent collaboration workflows?

RQ1 (AI Adoption). We evaluate whether MLLMs can predict whether humans ADOPT, MODIFY, or REJECT agent contributions under two settings—recognition (1a) and forecasting (1b) —which differ in whether the human response is observable or not (Table 4). Each input begins at the source node, where the human delegates to the agent $( T _ { \mathrm { s o u r c e } } ^ { \mathrm { s t a r t } } )$ . Forecasting clips average 14.0 minutes $( S D = 1 3 . 7 ; \mathrm { { m a x } = 7 3 . 0 ) }$ , while prediction clips average 8.8 minutes $( \bar { S } D \bar { = } 1 1 . 7 ; \mathrm { { m a x } = 7 2 . 0 ) }$ with more details in Table 10.

We are also curious whether longer context improves or degrades model performance. Thus, for Experiment 1c (step-N forecasting), the model must forecast the AI adoption label for each target AI node in the last step, while providing the preceding 1, 2, or 3 steps as additional context. For the longest context setting, motivated by how human annotators often need to watch the entire video context to determine the adoption label, we introduce Experiment 1d (workflow-level forecasting). In this setting, we provide the entire workflow up to the current target AI node.

RQ2 (Human Rating). An evaluator should distinguish genuinely successful interactions from those in which professionals were dissatisfied or produced lower-quality outcomes, rather than simply reflecting the predominantly positive ratings. Thus, we investigate whether MLLMs can similarly recover professionals’ ratings from multimodal, long-horizon human-agent collaboration workflows. Given the full workflow video for each step, we evaluate whether frontier MLLMs can predict the professional’s 1-5 rating of collaboration satisfaction (Experiment 2a) and output quality (Experiment 2b).

Table 5: Benchmarking performance for AI adoption recognition (Experiment 1a) and for forecasting (Experiment 1b), for video only, mean±std over 3 runs per model. The best result in each column is shaded and the second best result is bolded.
<table><tr><td rowspan="2">Model</td><td>1a: Recognition</td><td colspan="3">1b: Forecasting</td><td colspan="3">1b: Per-label F1</td></tr><tr><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>ADOPT</td><td>MODIFY</td><td>REJECT</td></tr><tr><td>Majority-class</td><td>0.265</td><td>0.220</td><td>0.333</td><td>0.265</td><td>0.794</td><td>0.000</td><td>0.000</td></tr><tr><td> Gemini 3.1 Pro</td><td>0.617±0.007</td><td>0.461±0.026</td><td>0.402±0.011</td><td>0.391±0.013 0.710±0.012</td><td></td><td>0.333±0.018</td><td>0.129±0.031</td></tr><tr><td>• Gemini 3.8 Flash</td><td>0.587±0.010</td><td>0.437±0.016</td><td>0.383±0.008</td><td>0.383±0.009</td><td>0.758±0.009</td><td>0.218±0.027</td><td>0.172±0.013</td></tr><tr><td>• GPT-6 Astra+</td><td>0.713</td><td>0.539±0.003</td><td>0.524±0.013</td><td>0.513±0.003</td><td>0.721±0.023</td><td>0.582±0.098</td><td>0.235±0.070</td></tr><tr><td>• Claude Opus 5.5</td><td>0.676</td><td>0.610</td><td>0.483</td><td>0.492</td><td>0.790</td><td>0.434</td><td>0.250</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>0.511±0.019</td><td>0.389±0.010</td><td>0.404±0.011</td><td>0.364±0.007 0.501±0.004</td><td></td><td>0.305±0.019</td><td>0.285±0.012</td></tr><tr><td>Qwen3-VL-32B</td><td>0.337±0.007</td><td>0.436±0.012</td><td>0.400±0.012</td><td>0.235±0.010</td><td>0.092±0.006</td><td>0.337±0.007</td><td>0.274±0.022</td></tr></table>

Experimental Setup We evaluate four leading multimodal agentic models, Gemini 3.8 Flash, Gemini 3.1 Pro, GPT-6 Astra<sup>4</sup>, and Claude Code Opus 5.5, and two open-weight video-capable MLLMs: Qwen3-VL-30B-A3B-Instruct (MoE) and Qwen3-VL-32B-Instruct (dense) (Bai et al., 2025). We compare against majority-class (ADOPT) and mode-rating baselines for RQ1 and RQ2, respectively. Our primary input is workflow video, which captures actions inside and outside the AI interface. We additionally evaluate all seven combinations of video, graph, and chat to assess the contribution of each modality.

## 5.2 RESULTS AND DISCUSSION

Performance on forecasting AI adoption. Table 5 shows each MLLM’s performance with video only as input. Claude performs the best for both recognizing and forecasting human adoption behavior, yet even the upper-bound only reaches an F1-score of 0.676 aside from GPT-6, which was evaluated on fewer instances than other models. Full details of model comparison with the common instance sets are shown in the Table 15 and Table 16 in the Appendix. Since the default AI agent used in this study is Claude, we hypothesize that it has a strong advantage in understanding interactions within its own interface. Even so, Claude’s F1 for forecasting human adoption is only 0.492, dominated by ADOPT. It has substantially lower performance on MODIFY and REJECT. Among the remaining models, Gemini 3.1 Pro performs the best when the human’s actual response is observable, with an F1 score of 0.617. Gemini models performs particularly poorly at forecasting REJECT, even worse than the open-weight Qwen models. Gemini 3.1 Pro is very biased toward generating ADOPT. 52% of true MODIFY and 71% of true REJECT are predicted as ADOPT by Gemini 3.1 Pro (Figure 7).

Why should an AI model care to understand human collaboration with other AI models? Humans orchestrate various software tools in our dataset, and their workflows are not necessarily limited to a single AI model. Even within our study, professionals used other AI systems, such as Gemini, Chat-GPT, and Grok, in addition to Claude. This makes it increasingly important for agents to understand human collaboration with other models, rather than only interactions involving themselves.

Does longer context help models anticipate human adoption behavior? We investigate whether providing more workflow context improves MLLMs’ performance on forecasting human’s behavior for AI adoption. Using 114 instances in which both the source and target AI nodes appear in the last step of each session, we extend the context to include up to three preceding steps (Experiment 1c) or the entire workflow up to the target AI node (Experiment 1d). Longer context seems to provide limited improvements, with macro-F1 changes below 0.07 for most video and graph settings even though none of the improvements with video are statistically significant (Figure 4). Chat logs benefit the most from additional context, with Gemini 3.1 Pro improving by 0.12 F1 when given three preceding steps $( p < 0 . 0 1 )$ . However, this improvement does not persist when the entire workflow is provided. Longer context can also hurt performance, as Qwen-32B drops from 0.43 to 0.32 F1 with the full graph $( p < 0 . 0 1 )$ . These findings suggest that limited context alone may not explain models’ difficulty in anticipating human adoption behavior as providing the entire workflow does not consistently improve adoption forecasting.

![](images/caaffb3cd2b3fa705b871d221abf87e72cb7940b0b843e4fd52f0d97448c35cd.jpg)  
Figure 4: Change in macro-F1 as forecasting context grows. X-axis shows how many N previous steps are included as a context. Full refers to the start of the whole workflow.

Table 6: Rating prediction for Experiments 2a and 2b, mean±std over 3 runs. The best result in each column is shaded ; the second best result is bolded.
<table><tr><td></td><td colspan="3">Exp. 2a: Collaboration</td><td colspan="3">Exp. 2b: Output Quality</td></tr><tr><td>Model</td><td>MAE↓</td><td></td><td>Within-1↑ Spearman ρ ↑</td><td>MAE↓</td><td></td><td>Within-1↑ Spearman ρ ↑</td></tr><tr><td>Mode baseline</td><td>0.682</td><td>0.854</td><td></td><td>0.735</td><td>0.831</td><td></td></tr><tr><td>• Gemini 3.1 Pro</td><td>0.685±0.006</td><td>0.850±0.003</td><td>0.078±0.057</td><td>0.764±0.016</td><td>0.821±0.006</td><td>0.046±0.039</td></tr><tr><td> Gemini 3.8 Flash</td><td>0.800±0.029</td><td>0.819±0.014</td><td>0.011±0.003</td><td>0.792±0.006</td><td>0.811±0.002</td><td>0.042±0.012</td></tr><tr><td>• GPT-6  $\mathrm { { A s t r a } ^ { + } }$ </td><td>0.945±0.070</td><td>0.841±0.032</td><td>0.379±0.130</td><td>0.912±0.084</td><td>0.805±0.030</td><td>0.352±0.101</td></tr><tr><td>• Claude Opus 5.5†</td><td>0.858</td><td>0.865</td><td>0.136</td><td>0.817</td><td>0.849</td><td>0.208</td></tr><tr><td>Qwen3-VL-30B-A3B 0.898±0.066 0.814±0.033</td><td></td><td></td><td>0.102±0.075</td><td></td><td>1.281±0.011 0.641±0.010</td><td>0.048±0.034</td></tr><tr><td> $\mathrm { \mathrm { ~ \circ ~ } } _ { \mathrm { Q w e n 3 - V L - 3 2 B } }$ </td><td> $1 . 1 9 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 6 3 4 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 1 3 0 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $1 . 2 2 3 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 6 3 3 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 1 5 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr></table>

Performance on predicting collaboration satisfaction and output quality. We evaluate alignment with human ratings using mean absolute error (MAE), within-1 accuracy, and Spearman’s ρ. The mode baseline outperforms every MLLM in MAE of 0.682 for collaboration satisfaction and 0.735 for output quality, reflecting the strong skew of human ratings toward 5. Gemini 3.1 Pro achieves the lowest MAE of 0.685 for collaboration satisfaction and 0.764 for output quality, but even the best-performing model fails to outperform the baseline. Moreover, rank agreement with human judgments remains weak. GPT-6 achieves the highest Spearman correlation for both collaboration satisfaction $( \rho = 0 . 3 7 9 )$ and output quality $( \rho = 0 . 3 5 2 )$ , while several models exhibit near-zero correlations. Thus, when disregarding differences in absolute rating scale, current frontier multimodal agents and LLMs still struggle to recover how professionals comparatively judge their collaboration and resulting outputs.

## 5.3 MODALITY ANALYSIS

We examine how video, graph, and chat, individually and in combination, affect models’ understanding of human-agent collaboration (Figures 5 and 6). No modality consistently performs best across models or experiment tasks. Video alone yields the highest recognition performance with GPT-6 (F1= 0.71, Experiment 1a). Graphs, either alone or combined with video or chat, tend to yield the best forecasting performance, with GPT-6 achieving an F1 score of 0.52 in Experiment 1b. However, chat alone particularly hurts recognition of REJECT cases: relative to video, F1 changes by +0.07 for ADOPT, +0.01 for MODIFY, and −0.07 for REJECT (Figure 6b), likely because chat does not capture how agent outputs are subsequently used outside the AI interface. Yet video is not consistently optimal. Human-curated workflow graphs, which capture high-level professional activities, particularly benefit the non-agentic Qwen models; for Qwen-32B, adding graphs to video improves forecasting F1 from 0.23 to 0.37.

Qwen-30B Qwen-32B  
![](images/3eda4cb8d629db84a1b61dac5dce171463822ea0fd5bc0652f6e9335fa050504.jpg)  
(a) 1a: Recognition (F1)

![](images/71105ef4342d167f4836090dcf841267c99dfaac5c21a570c81e7c7e7722fea8.jpg)  
(b) 1b: Forecasting (F1)

![](images/042e7ecfa03c57be85ddb4eb4e93b89d750a03190a48d94a09b596cec44330bc.jpg)  
(c) 2a: Collaboration satisfaction (ρ)

![](images/ebabd02d1a6cada3aa360f74d35dc341d3f2c950752e7d62ea6eba5081661519.jpg)  
(d) 2b: Output quality (ρ)  
Figure 5: Full results for seven input-modality combinations across the four benchmark tasks. V stands for video, G: graph, and C: chat log; “+” is a combination of modalities, “All” refers to using video, graph, and chat as the input. The best score in each row is in bold.

![](images/d2e4172d8de3e98641124dc9906b853b54599a53f8e605dfac8af83d8b8d1041.jpg)  
(a) Video misses low ratings

![](images/5bf375cf20abdbc07baf08b6f0c41a3a45faf8588b4553c09e54d35de9566d66.jpg)  
(b) Chat misses rejections

![](images/b87841cee52712f54e6dcf6dbac5f8a19ad8048cdffbbda74e398ca48d338677.jpg)  
(c) Graph fills video gaps  
Figure 6: How each modality helps.

For human-rating prediction, performance remains limited regardless of modality (Figure 5c–d). Correlations peak at only $\rho = 0 . 3 8$ for collaboration satisfaction (GPT-6, video only) and $\rho = 0 . 4 2$ for output quality (GPT-6, video+graph). Graphs are particularly informative for output quality: video alone yields near-zero correlations for Gemini 3.8 Flash and Qwen-30B $( \rho = 0 . 0 4 \mathrm { - } 0 . 0 5 )$ while adding graphs raises them to $\rho = 0 . 2 3 – 0 . 3 0$ . Overall, providing additional modalities does not resolve MLLMs’ difficulty in approximating professionals’ judgments of collaboration experience and output quality.

![](images/45e414a3f364939dba94c1a04004106c9ef3178a4b001fd43bdd74e713017aef.jpg)

![](images/bd1105982e1f3c8261bc92a5521086bfa218e52b56016b9019d6f97427c41a5a.jpg)  
Figure 7: Distributions of model-generated labels for MODIFY and REJECT cases in AI adoption forecasting.

![](images/2d9d6284de404e762602b01b9d199e3fa83612bbc228883507426eef2e5392c3.jpg)  
Figure 8: Mean predicted rating vs. human rating for collaboration (Exp 2a) and output quality (Exp 2b).

## 5.4 FAILURE CASES

For forecasting whether humans adopt agent contributions, models often strongly prefer certain labels although the true labels are MODIFY or REJECT (Figure 7). Gemini 3.1 Pro and Flash overpredict ADOPT. With video inputs, Gemini 3.8 Flash forecasts ADOPT for 83.8% of examples compared with a 65.9% ground-truth rate, and Qwen3-32B instead favors MODIFY, predicting it in 87.3% of cases compared with 19.1% in the ground truth. These biases largely persist with graph and chat inputs as well (e.g., Flash generates ADOPT for 82.3% and 74.0%, and Qwen3-32B outputs MODIFY for 67.7% and 62.0%). Models therefore exhibit distinct forecasting biases, which vary with both the model and input modality.

For human-rating prediction (Figure 8), some models predict similar ratings all the time. With video, Gemini 3.1 Pro predicts 4.83 for a true rating of 1 and 4.95 for a 5 (Experiment 2a); in Experiment 2b, it predicts 5.00 and 4.94. Graph input improves this separation, with gaps between true ratings of 1 and 5 reaching 1.95 and 2.45 for Experiments 2a and 2b. Claude and Qwen variants also show a similar pattern. Claude’s predictions range only from 3.3 to 3.9 across the entire 1–5 rating scale with video input in both experiments, while the ratings of Qwen variants range from 3.04 to 4.15. However, GPT-6 already separates low and high ratings with video and remains the most stable across modalities.

These systematic biases raise a more fundamental question: to what extent are MLLMs actually grounding their predictions in the observed workflow? If models understand the workflow context, their predictions should vary with the human behaviors and outcomes they observe. Some models repeatedly produce similar labels or ratings despite different inputs and ground-truths. It suggests that their predictions may be driven by model-specific priors instead of the workflow itself.

## 6 CONCLUSION AND FUTURE WORK

We have introduced KOLAB as a new dataset to benchmark frontier multimodal agentic models on their capabilities to support human-agent collaboration better on long-horizon workflows. Our results show that even frontier MLLMs, including GPT-6, struggle to predict and understand human–AI collaboration. As a future work, we will explore ways to improve these capabilities by constructing training datasets or developing sandbox environments where models can practice collaborating with simulated users before real-world deployment.

## ETHICS STATEMENT

Understanding how humans collaborate with AI agents is important for developing AI systems that better support human needs and decisions. However, evaluating human-agent collaboration also raises ethical concerns. First, models that inaccurately predict whether humans will adopt, modify, or reject AI contributions may misrepresent human intentions and preferences. Our findings highlight the limitations of current MLLMs in understanding professional human-agent collaborative workflows. Relying on these models as proxies for human judgments could lead to misleading assessments, particularly in professional domains where decisions require specialized expertise and contextual understanding. Second, collecting and releasing real-world professional workflows raises privacy concerns, as screen recordings and interaction logs may contain sensitive information. To mitigate these risks, we remove all personally identifiable information (PII) in our data. Our work aims to support the development of human-centered collaborative agents while emphasizing that model predictions should not replace human judgments or undermine human agency. This study was approved by our institution’s IRB. All participant data are anonymized and securely stored.

## REPRODUCIBILITY STATEMENT

Code and data are released in our project webpage https://minnesotanlp.github.io/ kolab/. Details about data collection details are explained in A.2 in the Appendix. Data processing and annotation details are explained in A.4. All model prompts and model settings are elaborated in the Appendix.

## ACKNOWLEDGMENTS

We are grateful to Jingcheng Liang, Risako Owan, and Su Yeon Han for helping with data cleaning and annotation. This research was supported in part by the Interdisciplinary Doctoral Fellowship from the University of Minnesota for the first author.

## REFERENCES

Anthropic. Claude. https://claude.ai, 2026. Accessed: 2026-09-25.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Stefano Boccaletti, Vito Latora, Yamir Moreno, Martin Chavez, and D-U Hwang. Complex networks: Structure and dynamics. Physics reports, 424(4-5):175–308, 2006.

John Adrian Bondy and Uppaluri Siva Ramachandra Murty. Graph theory. Springer Publishing Company, Incorporated, 2008.

Valerie Chen, Rohit Malhotra, Xingyao Wang, Juan Michelini, Xuhui Zhou, Aditya Bharat Soni, Hoang H. Tran, Calvin Smith, Ameet Talwalkar, and Graham Neubig. How can we assess human-agent interactions? case studies in software agent design. In International Conference on Machine Learning (ICML), Seoul, South Korea, 2026. URL https://arxiv.org/abs/ 2510.09801.

Julia Christenson, Karin Johanna Denton de Langis, Shirley Anugrah Hayati, and Dongyeop Kang. Effects of varying LLM access on essay writing behavior. In Proceedings of the Workshop on

Innovative Use of NLP for Building Educational Applications (BEA), 2026. URL https:// arxiv.org/abs/2606.00250.

Yutong Dai, Krithika Ramakrishnan, Jing Gu, Matthew Fernandez, Yanqi Luo, Viraj Prabhu, Zhenyu Hu, Caiming Xiong, Zeyuan Chen, Ran Xu, et al. Scuba: Salesforce computer use benchmark. In International Conference on Learning Representations, volume 2026, pp. 24363–24386, 2026.

Debarati Das, Khanh Chi Le, Ritik Sachin Parkar, Karin De Langis, Brendan Madson, Chad M. Berryman, Robin M. Willis, Daniel H. Moses, Brett McDonnell, Daniel Schwarcz, and Dongyeop Kang. Lawflow: Collecting and simulating lawyers’ thought processes on business formation case studies. In Conference on Language Modeling (COLM), 2025. URL https://openreview. net/forum?id=MsgdEkcLRz.

K-I Goh and A-L Barabási. Burstiness and memory in complex systems. Europhysics Letters, 81 (4):48002, 2008.

Shirley Anugrah Hayati, Moyan Zhou, Patricia Anugrah Setiani, Ruizi Wang, Joseph Chee Chang, and Dongyeop Kang. Orchestrating genai for interdisciplinary research, 2026. URL https: //arxiv.org/abs/2609.30588.

Faria Huq, Zora Zhiruo Wang, Frank F Xu, Tianyue Ou, Shuyan Zhou, Jeffrey P Bigham, and Graham Neubig. Cowpilot: A framework for autonomous and human-agent collaborative web navigation. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (System Demonstrations), pp. 163–172, 2025.

Lawrence Keunho Jang, Andrew Keunwoo Jang, Jing Yu Koh, and Ruslan Salakhutdinov. Mypcbench: A benchmark for personally intelligent computer-use agents. arXiv preprint arXiv:2606.16748, 2026.

Khanh Chi Le, Linghe Wang, Minhwa Lee, Ross Volkov, Luan Tuyen Chau, and Dongyeop Kang. ScholaWrite: A dataset of end-to-end scholarly writing. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 34755–34788, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8- 89176-390-6. doi: 10.18653/v1/2026.acl-long.1606. URL https://aclanthology.org/ 2026.acl-long.1606/.

Wanli Li, Bowen Zhou, Yunyao Yu, Zhou Xu, Yifan Yang, Dongsheng Li, and Caihua Shan. Weavebench: A long-horizon, real-world benchmark for computer-use agents with hybrid interfaces. arXiv preprint arXiv:2606.09426, 2026.

Matheus Kunzler Maldaner, Adam Fourney, Amanda Swearngin, Hussein Mozannar, Gagan Bansal, Maya Murad, Rafah Hosn, and Saleema Amershi. Sentinelbench: A benchmark for long-running monitoring agents. arXiv preprint arXiv:2606.05342, 2026.

Mike Merrill, Alexander Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Shin, Thomas Walshe, E Kelly Buchanan, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations, volume 2026, pp. 40903–40986, 2026.

Samuel Miserendino, Michele Wang, Tejal Patwardhan, and Johannes Heidecke. SWE-lancer: Can frontier LLMs earn \$ 1 million from real-world freelance software engineering? In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 44412–44450. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/miserendino25a.html.

Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, et al. Magentic-ui: Towards human-in-the-loop agentic systems. arXiv preprint arXiv:2507.22358, 2025.

Mark Newman, Albert-László Barabási, and Duncan J Watts. The structure and dynamics of networks. Princeton university press, 2011.

Jio Oh, Seunghyun Do, Young-Jun Lee, Steven Euijong Whang, and Dongyeop Kang. Foundations of proactive agents: Principles, technical layers, and proactivity-gym. arXiv preprint arXiv:2609.37267, 2026.

Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simon Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, et al. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. In International Conference on Learning Representations, volume 2026, pp. 24005–24040, 2026.

Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, et al. Userbench: An interactive gym environment for user-centric agents. arXiv preprint arXiv:2507.22034, 2025.

Daniel Schwarcz, Debarati Das, Dongyeop Kang, and Brett McDonnell. Thinking like a lawyer in the age of generative AI: Cognitive limits on AI adoption among lawyers. Journal of Institutional and Theoretical Economics, 2026. URL https://papers.ssrn.com/sol3/papers. cfm?abstract\_id=5260645.

Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, and Diyi Yang. Future of work with ai agents: Auditing automation and augmentation potential across the us workforce. arXiv preprint arXiv:2506.06576, 2025.

Yijia Shao, Vinay Samuel, Yucheng Jiang, John Yang, and Diyi Yang. Collaborative gym: A framework for enabling and evaluating human-agent collaboration. In International Conference on Learning Representations, volume 2026, pp. 99616–99649, 2026a.

Yijia Shao, Zora Zhiruo Wang, Neel Ahuja, Yicheng Wang, Bowen Liu, and Diyi Yang. Collabskill: Evaluating human-agent collaboration on real-world tasks. arXiv preprint arXiv:2606.09833, 2026b.

Shannon Zejiang Shen, Valerie Chen, Ken Gu, Alexis Ross, Zixian Ma, Jillian Ross, Alex Gu, Chenglei Si, Wayne Chi, Andi Peng, Jocelyn J Shen, Ameet Talwalkar, Tongshuang Wu, and David Sontag. Scaling collaborative effort with agents. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 12254–12271, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.595. URL https://aclanthology.org/2026.findings-acl.595/.

Ricard Solé, Giulio Ruffini, Francesca Castaldo, Marco Tuccio, Luis F Seoane, Manlio de Domenico, Santiago F Elena, David C Krakauer, and Michael Levin. Large-language models as a cognitive virus. arXiv preprint arXiv:2609.03344, 2026.

Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, et al. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. In International Conference on Learning Representations, volume 2026, pp. 75694–75731, 2026a.

Yiyou Sun, Xinyang Han, Weichen Zhang, Yuanbo Pang, Tianyu Wang, Yuhan Cao, Yixiao Huang, Chris Duroiu, Haoyun Zhang, Jeffrey Lin, et al. Agents’ last exam. arXiv preprint arXiv:2606.05405, 2026b.

Jihoon Tack, Philippe Laban, and Jennifer Neville. Llms get lost in evolving user intent. arXiv preprint arXiv:2607.20734, 2026.

Bertie Vidgen, Austin Mann, Abby Fennelly, John Wright Stanly, Lucas Rothman, Marco Burstein, Julien Benchek, David Ostrofsky, Anirudh Ravichandran, Debnil Sur, et al. Apex-agents. arXiv preprint arXiv:2601.14242, 2026.

Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, and Graham Neubig. Ambig swe: Interactive agents to overcome underspecificity in software engineering. In International Conference on Learning Representations, volume 2026, pp. 146716–146737, 2026.

Ruyuan Wan, Naome Etori, Karla Badillo-Urquiola, and Dongyeop Kang. User or labor: An interaction framework for human-machine relationships in NLP. In Proceedings of the Workshop on Data Science with Human-in-the-Loop (DaSH), 2022. URL https://arxiv.org/abs/ 2211.01553.

Wenkai Wang, Tao Xiong, Jingchen Ni, Yunpeng Bao, Xiyun Li, Tianqi Liu, Hongcan Guo, Zilong Huang, and Shengyu Zhang. Deskcraft: Benchmarking desktop agents on professional workflows and human-in-the-loop collaboration. arXiv preprint arXiv:2606.03103, 2026.

Zora Zhiruo Wang, Yijia Shao, Omar Shaikh, Daniel Fried, Graham Neubig, and Diyi Yang. How do ai agents do human work? comparing ai and human workflows across diverse occupations. arXiv preprint arXiv:2510.22780, 2025.

Yaozu Wu, Wei-Chieh Huang, Jizhou Guo, Dongyuan Li, Renhe Jiang, Henry Peng Zou, Chunyu Miao, Shanghao Li, Weizhi Zhang, WeiWei Ye, et al. Has-bench: Evaluating llm-based human agent systems under configurable human participation. arXiv preprint arXiv:2607.04329, 2026.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Frank Fangzheng Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, et al. Theagentcompany: benchmarking llm agents on consequential real world tasks. Advances in Neural Information Processing Systems, 38, 2026.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

Mengqi Yuan, Zilong Zhou, Xinzhuang Xiong, Weiming Wu, Jiayang Sun, Jiamin Song, Kaiqian Cui, Bowen Wang, Haoyuan Wu, Yitong Li, Dunjie Lu, Haikong Lu, Qi Zhen, Xinyuan Wang, Jiaqi Deng, Yuhao Yang, Cheng Chen, Boyuan Zheng, Alex Su, Xiao Yu, Hao Zou, Saaket Agashe, Xing Han Lu, Manpreet Kaur, Zhengyang Qi, Vincent Sunn Chen, Frederic Sala, Dayiheng Liu, Junyang Lin, Zhou Yu, Yu Su, Siva Reddy, Xin Eric Wang, Peng Qi, Tianbao Xie, and Tao Yu. Osworld2.0: Benchmarking computer use agents on long-horizon real-world tasks, 2026. URL https://arxiv.org/abs/2606.29537.

Liya Zhu, Jingzhe Ding, Jian Zhang, Jianbo Xue, Shihao Liang, Ge Zhang, Yi Zhu, Duju Zeng, Xiang Gao, Qingshui Gu, et al. Workflow-gym: Towards long-horizon evaluation of computeruse agentic tasks in real-world professional fields. arXiv preprint arXiv:2606.11042, 2026.

Henry Peng Zou et al. Llm-based human-agent collaboration and interaction systems: A survey. arXiv preprint arXiv:2505.00753, 2025. Findings of ACL 2026. 20 total co-authors, abbreviated here.

## A APPENDIX

## A.1 PROFESSIONALS BACKGROUND AND SOFTWARE USAGE

The 50 study participants span 10 O\*NET sectors and 40 distinct occupations, ranging from film editors, graphic designers, and audio engineers to aerospace engineers, GIS specialists, attorneys, and bioinformaticians. We prioritize non-computer science professions, which remain comparatively under-resourced with real-world task data relative to software engineering (Patwardhan et al., 2026; Shao et al., 2025; Sun et al., 2026b; Miserendino et al., 2025). Across these roles, participants reported using 90 distinct profession-specific non-AI softwares (excluding Google or Microsoft office products) and 9 AI tools in their day-to-day work, as shown in Figure 9. This mix of shared productivity tools and domain-specific software illustrates the breadth of professional practice that the benchmark’s workflows are grounded in.

Scientific / Biomedical  
![](images/a0b36a39ed6d47ba4cdd2538eb208537b10f84dfd85e111e91a2adfc145684ee.jpg)

![](images/20a1e1f85519867a68172c05700ab866460b5b892348d98c9095ac5edba104c9.jpg)

![](images/f515d99baf17420e2af742202d53a17e167f74f3876e71626b340efc7e5aea07.jpg)

![](images/48d2414481ee1f667fe591cc5b5efd29a22e90bd74ca8aaa9a9f69f1d49e1a57.jpg)

![](images/903d43a02be5025320aee3ab0826a3587e845345161eb3369aed08b5d22e2a5a.jpg)

![](images/94d9c1e03a208e5164016cd57f53811574c2a89e0ab667429d220872eaa66f69.jpg)

![](images/73dbf425ba65d23b09cb54f960df941d6aa2e34972aa8b19d1a7c7f777816208.jpg)  
Figure 9: Software used by participants in their day-to-day work, grouped by category.

Table 7: Summary statistics of the 50 professionals.
<table><tr><td>Category</td><td>Item</td><td>Count</td></tr><tr><td rowspan="3">Coverage</td><td>Total participants</td><td>50</td></tr><tr><td>O*NET sectors</td><td>10</td></tr><tr><td>Occupations</td><td>40</td></tr><tr><td rowspan="4">Education</td><td>PhD</td><td>10</td></tr><tr><td>Master&#x27;s</td><td>20</td></tr><tr><td>Bachelor&#x27;s</td><td>15</td></tr><tr><td>Other</td><td>5</td></tr><tr><td rowspan="5">Age range</td><td>18-25</td><td>8</td></tr><tr><td>26-35</td><td>20</td></tr><tr><td>36-45</td><td>13</td></tr><tr><td>46-55</td><td>8</td></tr><tr><td>56-65</td><td>1</td></tr><tr><td rowspan="5">Work experience</td><td>&lt; 1 year</td><td>7</td></tr><tr><td>1–3 years</td><td>19</td></tr><tr><td>4–6 years</td><td>7</td></tr><tr><td>6–10 years</td><td>5</td></tr><tr><td>&gt; 10 years</td><td>12</td></tr><tr><td rowspan="4">Self-rated expertise</td><td>Basic (2/5)</td><td>2</td></tr><tr><td>Intermediate (3/5)</td><td>11</td></tr><tr><td>Advanced (4/5)</td><td>18</td></tr><tr><td>Expert (5/5)</td><td>19</td></tr><tr><td rowspan="3">GenAI familiarity</td><td>Low (1–4/10)</td><td>1</td></tr><tr><td>Medium (5–7/10)</td><td>12</td></tr><tr><td>High (8–10/10)</td><td>37</td></tr></table>

<table><tr><td>Category</td><td>Country</td><td>Count</td></tr><tr><td></td><td>USA</td><td>20</td></tr><tr><td></td><td>Colombia</td><td>3</td></tr><tr><td></td><td>India</td><td>3</td></tr><tr><td></td><td>Mexico</td><td>3</td></tr><tr><td></td><td>Germany</td><td>3</td></tr><tr><td></td><td>Nigeria</td><td>2</td></tr><tr><td></td><td>UK</td><td>2</td></tr><tr><td></td><td>Argentina</td><td>1</td></tr><tr><td></td><td>Ecuador</td><td>1</td></tr><tr><td>Geography</td><td>Egypt</td><td>1</td></tr><tr><td>(21 countries)</td><td>El Salvador</td><td>1</td></tr><tr><td></td><td>Finland</td><td>1</td></tr><tr><td></td><td>Kenya</td><td>1</td></tr><tr><td></td><td>Montenegro</td><td>1</td></tr><tr><td></td><td>Morocco</td><td>1</td></tr><tr><td></td><td>Netherlands</td><td>1</td></tr><tr><td></td><td>Pakistan</td><td>1</td></tr><tr><td></td><td>South Africa</td><td>1</td></tr><tr><td></td><td>Sri Lanka</td><td>1</td></tr><tr><td></td><td>Turkey</td><td>1</td></tr><tr><td></td><td>Yemen</td><td>1</td></tr></table>

![](images/777c759b25ed5f781254d35ae45c1736cef9df4ee8c6eb11b1b610430d8c81b7.jpg)  
Figure 10: Occupation distributions of KOLAB. “Other” refers to three O\*NET sectors: Life & Social Science, Sales & Related, and Healthcare.

![](images/ee32df57728fa1b94135a7a21e001933daabbe3facbffe3c0e643ed9c6bb2b76.jpg)  
Figure 11: Time spent in each recording session working with agentic AI versus the participant’s estimated time to solve the same problem without agentic AI.

## A.2 DATA COLLECTION DETAILS

We collected KOLAB through two sessions per participant, in which we recorded the participant’s computer screen. Each session lasted approximately one hour or longer and centered on a problem chosen by the participant. After data processing and due to time constraints, some problems contained fewer than five steps; however, each interaction session still lasted approximately one hour. To capture different patterns of human-agent collaboration, we used two protocols. In Session 1, participants were encouraged to interact with the agent at least five times while completing their workflow. In Session 2, participants were encouraged to interact with the agent at each step of the workflow. The two sessions did not have to address the same problem. However, participants could use the second session to continue or extend the work from the first session. For example, a participant might use the first session to develop a branding logo and use the second session to create a marketing advertisement based on the resulting brand identity. Participants were required to interact with agentic AI, Claude, using our credentials during each session and were free to use other AI agents when appropriate for their work. For example, an architect might use Claude for planning and reasoning while using ChatGPT to generate or render architectural images. At the time of data collection, most participants used the default Claude model (Sonnet 5.5), while a few used Opus 5. Each participant planned their workflow before the recording sessions to ensure that their problem satisfied our task criteria. After the recording sessions, they updated the workflow document and completed surveys about their collaboration experience and output quality. During each recording session, participants were encouraged to think aloud; we used the transcriptions to construct the workflow graphs and to guide the annotation of response labels.

## A.3 GRAPH AS A COLLABORATION WORKFLOW REPRESENTATION

Actions and timestamps from workflow video are aligned with steps documented by each professional. We then represented each workflow as a directed graph, where each node corresponds to a human or agent action. The research team and professionals jointly reviewed and refined these graphs. Graphs can capture non-linear workflows; for instance, a node branches when activities occur in parallel and loops happen when the same action is repeated.

![](images/5a728cc047da277e77afd05d97a820250c9f38231f73acc2090e978321944e65.jpg)  
Figure 12: Example workflow graph for one step of a participant’s session.

Formally, each session is represented as a directed temporal graph

$$
G _ { t e m p o r a l } = ( V , E , a , w , \tau ) ,\tag{2}
$$

where V is the set of nodes, each corresponding to a discrete human or AI action, and $E \subseteq V \times V$ is the set of directed edges, with $( u , v ) \in E$ indicating that action v directly follows from action u within the workflow. Each node $v \in V$ is annotated by three functions: an actor label

$$
a : V \to \{ \mathbf { H } , \mathbf { A } \} ,\tag{3}
$$

indicating whether the action was performed by the human or the agent; a workflow-step label

$$
w : V \to \{ 1 , \dots , K \} ,\tag{4}
$$

indicating which of the K steps of the task the action belongs to; and a time interval

$$
\tau ( v ) = [ t _ { s } ( v ) , t _ { e } ( v ) ] \subset \mathbb { R } _ { \geq 0 } .\tag{5}
$$

recording when the action began and ended.

After graph construction, two human annotators annotated how each AI output was incorporated into the workflow. For every observed user response to an AI output, annotators identified the prompt node that triggered generation, the AI-content node being acted on, and a response label: ADOPT (incorporated without modification), MODIFY (incorporated with changes), or REJECT (explicitly rejected or not incorporated). This is an annotation layer on top of $G _ { t e m p o r a l } \mathrm { : }$ each node $\bar { n } \in \bar { V }$ representing a user response carries a set of response triples as follows.

$$
U ( n ) = \{ ( p _ { i } , c _ { i } , \ell _ { i } ) \} _ { i = 1 } ^ { m _ { n } } ,\tag{6}
$$

where $p _ { i } \in V \cup \{ \varnothing \}$ is the prompt node (absent if the user acts on existing AI content without a new prompt), $c _ { i } \in V$ is the AI-content node $( a ( c _ { i } ) = \mathbf { A } )$ , and $\ell _ { i } \in \{ \mathrm { a d o p t , m o d i f y , r e j e c t } \}$ is the response label.

The following is an example workflow document that an architect filled out to plan their workflow before each recording session. After the session, they revised the content and filled out the assessment questions about collaboration experience and output quality ratings. Other step details are omitted due to space.

Goal: Developing an architectural plan for Residential House

Context: I have chosen this project because it requires a set of expertise to layout an architectural plan. Conditions including design brief, site and weather analysis, building codes, area and zoning standards to evaluate and design according to that as an expert.

Did you achieve your goal after \~1-hour interaction with Claude? Yes

Why or why not? Claude was able to produce layouts

Step 1 name: Summary of design brief, a set of requirements to start the project

Step action: Prepare the design brief and define scope of project.

Input: client brief shared with me

Output: Revise details when design brief in comprehensive format is generated.

Softwares: Claude

## Rubric:

1. Each design improvement includes a clear architectural reason

2. Each revision idea includes a rationale.

3. The revised layout remains consistent with the client brief.

4. Suggestions are consistent with the requirements.

## Output Quality and Collaboration Assessment

1. Rate the output quality from 1–5 (1 not satisfy, 5 satisfy). 4

2. Reason for your rating? “It created a set of pointers in design brief from the narrative, and curated a long brief in detail. Added extra things also that were not questioned.”

3. Which ones of the rubrics above are satisfied in this output? 2, 3, and 4

## Human-Agent Collaboration

1. Did the AI Agent successfully complete the task? Yes

2. Why? It created a set of pointers in design brief from the narrative, and curated a long brief in detail. Added extra things also that were not questioned.

3. Collaboration satisfaction — write a number (1–5) to the left of each statement, where 1 = not satisfied and 5 = satisfied:

• 5 — The agent understood the task, context, and instructions provided.

• 4 — The agent responded appropriately to requests and adapted when given feedback or corrections.

• 4 — The agent reduced effort, time, or cognitive load during the task.

• 4 — The agent proactively suggested useful ideas, alternatives, or next steps when appropriate.

• 5 — The agent produced accurate, relevant, and trustworthy outputs that required minimal correction.

• 2 — The interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work.

• 3 — I was able to guide, redirect, and override the agent’s actions when desired.

4. Rate the overall collaboration experience from 1–5 (1 not satisfy, 5 satisfy). 4

Step 2: area codes to use in project

Step 3: plan zoning and divisions of spaces

Step 4: Arrangement of spaces and Finalizing layout

Step 5: Architectural floor Plan finalization

Another example of workflow document by a different participant (sound designer and composer) is shown below. In this case, the agentic AI failed to help and actually slowed down the process.

## Goal: Sound Design a promotional video for a real estate sales pitch

Context: This goal will allow me to evaluate how AI can enhance my work as a sound designer. The plan is to work on a promotional sales pitch for apartments in cozumel, which was recorded by a production company, but they lack sound for many of the camera shots they use (I.E: Shots of the beach, of the town and its localities, etc.) This lack of sound makes the project seem empty, therefore, a sound designer is needed to enhance the immersion of what is shown by adding the ambience noise, sound of actions (Walking, placing, etc) while ensuring audio files don’t overlap and work in symphony.

## Did you achieve your goal after \~1-hour interaction with Claude? No

Why or why not? “Although we did advance on the video, It was notfinished in the 1 hour mark. Ifelt like I had to keep up with Claude at the start, rather than being benefited by it while working. This gradually improved when I used it for smaller activities like finding specific SFX or a good usage of effect plugins, but when I initially tried to use it as an analysis assistant, it took almost 20 minutes just analyzing the video,followed by 10 more minutes ofanalyzing just afragment ofit and providing ideas, which had slowed me down significantly, as I was already done with most ofthe other things I had not delegated to Claude.”

## Step 1 name: Analyze the video and understand key moments from it

Step action: Understand key moments where sound design can exponentially increase quality and immersion for the video, uploading the video to Claude for it to create this in depth analysis. Input: MP4 video with dialogue and visual insights to the required SFX Output: A pdf or DOCX document with key moments, time stamps and descriptions of the required sound effects. Softwares: FL Studio, Artlist, Claude Max

## Rubric:

1. A score of 5 means that Claude was able to analyze the video and describe with detail what it is, what its selling, what key visuals and moments it has and how to interpret it as a client.

2. A score of 5 means the analysis included accurate, verifiable timestamps for each key moment, ready to use directly without needing to re-watch the video.

## Output Quality and Collaboration Assessment

1. Rate the output quality from 1–5 (1 not satisfy, 5 satisfy). 1

2. Reason for your rating? “Claude was not able tofully analyze the video, due to a runout of time.”

3. Which ones of the rubrics above are satisfied in this output? N/A

## Human-Agent Collaboration

1. Did the AI Agent successfully complete the task? No.

2. Why? “Claude mentioned a timeout when trying to analyze the video.”

3. Collaboration satisfaction — write a number (1–5) to the left of each statement, where 1 = not satisfied and 5 = satisfied:

• 3 — The agent understood the task, context, and instructions provided.

• 1 — The agent responded appropriately to requests and adapted when given feedback or corrections.

• 1 — The agent reduced effort, time, or cognitive load during the task.

• 1 — The agent proactively suggested useful ideas, alternatives, or next steps when appropriate.

• 1 — The agent produced accurate, relevant, and trustworthy outputs that required minimal correction.

• 1 — The interaction felt smooth, with the agent contributing at the right moments without creating unnecessary work.

• 2 — I was able to guide, redirect, and override the agent’s actions when desired.

4. Rate the overall collaboration experience from 1–5 (1 not satisfy, 5 satisfy). 1

Step 2: Develop the sound design concept   
Step 3: Find and select appropriate sound effects   
Step 4: Generate a sound effect from scratch   
Step 5: Create a step by step guide to improve the realism of a sound effect using effects (EQ, Reverb,   
etc.)

Table 8: Distribution of collaboration satisfaction and output quality ratings, by session.
<table><tr><td>Rating Session 1</td><td>Session 2</td><td>Total</td></tr><tr><td colspan="3">Collaboration Satisfaction</td></tr><tr><td>5 117 (53.7%) 4 61 (28.0%) 3 28 (12.8%)</td><td>122 (56.2%) 68 (31.3%) 16 (7.4%)</td><td>239 (54.9%) 129 (29.7%) 44 (10.1%)</td></tr><tr><td>2 8 (3.7%) 1 4 (1.8%)</td><td>7 (3.2%) 4 (1.8%)</td><td>15 (3.4%) 8 (1.8%)</td></tr><tr><td colspan="3">Output Quality</td></tr><tr><td>5 119 (49.8%)</td><td>133 (57.3%)</td><td>252 (53.5%)</td></tr><tr><td>4 81 (33.9%)</td><td>65 (28.0%)</td><td>146 (31.0%)</td></tr><tr><td>3 27 (11.3%)</td><td>20 (8.6%)</td><td>47 (10.0%)</td></tr><tr><td>2 6 (2.5%)</td><td>9 (3.9%)</td><td>15 (3.2%)</td></tr><tr><td>1 6 (2.5%)</td><td>5 (2.2%)</td><td>11 (2.3%)</td></tr></table>

Table 8 shows the rating distribution by participants for collaboration satisfaction (1 low, 5 high) and output quality (1 low, 5 high) for each step.

## A.4 DATA PROCESSING AND ANNOTATION DETAILS

The actions from workflow video are aligned with steps documented by each professional. Claude was used to map video timestamps to the corresponding steps, after which professionals reviewed and corrected the alignments. We then represented each workflow as a directed graph, where each node corresponds to a human or AI action and is associated with a workflow step. Nodes with outdegree greater than one denote a branch into concurrent sub-paths, and nodes with in-degree greater than one denote a merge of previously concurrent branches back into a single line of work. The authors and professionals jointly reviewed and refined these graphs. Graphs can include branches when activities occur in parallel and loops when the same action is repeated, allowing the representation to capture non-linear workflow structures as shown in Figure 12. After graph construction, two human annotators annotated how agent outputs were incorporated into the workflow. For each observed response to an agent output, annotators identified the source node, corresponding to the human action that initiated the request, and the agent node, corresponding to the resulting agent response. They then identified the subsequent human node in which the response was acted upon and labeled the user’s response as ADOPT, MODIFY, or REJECT. ADOPT indicates that the AI output was incorporated without modification; MODIFY indicates that it was incorporated with changes; and REJECT indicates that the user explicitly rejected the output or did not incorporate it into the workflow. Because an AI output may be acted upon later in the workflow rather than immediately after it is produced, these annotations explicitly link each user response to the AI output it concerns.

## A.4.1 ANNOTATION GUIDELINES

Each workflow graph records not only the sequence of user prompts and Claude responses, but also how each Claude response was incorporated into the user’s workflow. This is captured by the user\_response field, which is added only to the user node that represents the adopt, modify, or reject decision.

user\_response takes the following form:

![](images/e452ff5788484640ba7e081d0812e25adec379ffe3687a02c8da949c822534df.jpg)

![](images/f68a243b82f28f910f57eee8a89d0c00ad2b379f50c876dbbdc36e973a6a7357.jpg)

![](images/5950d5abc0f55ee283620bee5b4e81d880fd37e9f60167c5c8842e9222182861.jpg)  
Source (user) nodeAl node☆Adoption☆Modification☆Rejection  
Figure 13: How three professionals (musician, marketing specialist, and attorney) adopt, modify, and reject agent outputs.This is based on the real data point in KOLAB.

Table 9: Real examples of ADOPT, MODIFY, and REJECT responses from KOLAB, drawn from a music producer workflow, architecture workflow, and a video-production workflow respectively.
<table><tr><td>Response</td><td>Source Node Action</td><td>Target AI Node Action</td><td>Current Node Action</td><td colspan="2">Software</td></tr><tr><td>ADOPT</td><td>Request Afrobeats/techno back- ground music in ElevenLabs (67:33–68:30)</td><td>ElevenLabs generates Afrobeats instrumental candidates (69:50– 70:04)</td><td>Capture the ElevenLabs track and import it into the DaVinci Resolve project (72:05–74:36)</td><td>ElevenLabs, DaVinci Resolve</td><td></td></tr><tr><td></td><td>Ask Claude to add a faith-based closing line to the voiceover script (13:07-13:41)</td><td>Claude adds the requested faith- based reference to the script (34:17-34:22)</td><td>Generate the final voiceover in ElevenLabs and import it into DaVinci Resolve (34:22–36:30)</td><td>Labs, Resolve</td><td>Claude, Eleven- DaVinci</td></tr><tr><td rowspan="2">MODIFY</td><td>Upload the existing shell plan and request a full room layout (24:10– 25:28)</td><td>Claude delivers a dimensioned floor-plan layout for the lobby/s- torage circulation (51:21–56:20)</td><td>Rebuild walls/partitions in Revit, keeping the liked zoning but fix- ing lobby/storage circulation (58:40– 72:59)</td><td>Revit</td><td></td></tr><tr><td>Ask whether CapCut&#x27;s AI Re- light tool requires the desktop app (63:49–63:55)</td><td>Claude explains AI Relight also works in the web version (64:01– 64:04)</td><td>Retry AI Relight in the browser; still blocked by the desktop-app require- ment (64:04–65:17)</td><td>CapCut</td><td></td></tr><tr><td rowspan="2">REJECT</td><td>Ask Kling AI to generate a Zanz- ibar schoolyard video clip (36:30— 39:04)</td><td>Kling AI generates the requested video clip (39:45–41:17)</td><td>Judge the clip&#x27;s audio unusable and abandon it, out of credits to iterate fur- ther (41:17–42:06)</td><td>Kling AI</td><td></td></tr><tr><td>Ask ChatGPT to generate the same video clip (40:09–40:28)</td><td>ChatGPT explains it cannot gen- erate video and offers a still image instead (40:28–41:03)</td><td>Review ChatGPT&#x27;s limitation and move to another tool (41:03–41:17)</td><td>ChatGPT</td><td></td></tr></table>

"source\_node\_time", "claude\_node\_time", "user\_response\_time"]]

This ensures that each adopt, modify, or reject annotation is linked to the correct occurrence of the repeated node, rather than being ambiguous across its several instances.

## Field definitions.

• source\_node\_id — The user node ID that initiated the Claude request for this step of the workflow. This node may originate from a previous step: all subsequent user responses within the same workflow continue to reference it as the source, even as the user issues additional requests. source\_node\_id is left empty when the user reuses a prior Claude response within a new step whose narrative is not coherently linked back to the node from that earlier step.

• claude\_node\_id — The Claude node ID containing the response to the request.

• label — One of adopt, modify, or reject, indicating how the user incorporated the Claude response into the workflow. On rare occasions a single node carries more than one label; in that case, each label is recorded as a separate tuple with its own source\_node\_id and claude\_node\_id.

## Label definitions.

• ADOPT: The user incorporates Claude’s response into the workflow without changes.

• MODIFY: The user incorporates Claude’s response into the workflow after modifying it.

• REJECT: The user does not incorporate Claude’s response into the workflow. This label is applied only when the user explicitly states that they are rejecting the response.

Examples of how professionals adopt, modify, or reject agent outputs are shown in Figure 13. A request from a previous step can undergo multiple refinements before being adopted in a later step.

## A.5 DETAILS FOR CONSTRUCTING AND ANNOTATING GRAPHS FROM VIDEOS

Figure 14 shows our workflow annotation interface. Professionals first align their documented workflow steps with the corresponding video timestamps (Figure 14a). Step names are taken from their workflow documents and initially mapped to the video by Claude Code Sonnet 5.5. After professionals correct the step boundaries, we extract a workflow graph for each step and present it for verification (Figure 14b), where professionals confirm whether the high-level workflow structure is consistent with their actual process. The prompt for generating the initial graph by Claude is shown below.

Four members of the research team then reconstruct and annotate the full workflow graphs, as the automatically extracted graphs frequently contain errors in action labels, edges, timestamps, and actor assignments. Annotators use the screen recordings, think-aloud transcripts, agent chat logs, and input/output files to ground these corrections. When an action or dependency remains ambiguous, the research team consults the corresponding professional for clarification.

## Workflow Graph Extraction

## You are given:

1. A workflow video.

2. A list of workflow steps with timestamps.

3. The workflow goal.

4. A transcript of the video (.srt).

5. A workflow planning document (.docx).

6. A chat log between human user and Claude

Your task is to extract a behavioral workflow graph, capturing goal-directed activities performed by the human user and AI throughout the workflow.

## Behavioral nodes

A behavioral node represents what the actor is trying to accomplish, not the computer use actions.

You must focus on the underlying work activity, intention, or objective and include relevant context, such as the object being worked on or the purpose of the activity.

## Examples:

\* Define objective

\* Plan approach

\* Seek information

\* Gather evidence

\* Analyze information

\* Compare alternatives

\* Refine requirements

\* Verify findings

\* Make decision

\* Implement solution

\* Validate outcome

\* Revise solution

These are examples only, not a fixed taxonomy. Use a more specific label when the context supports it.

For example, prefer:

> “Evaluate candidate methods for the analysis”

## over:

> “Read information”

Do not create nodes for low-level actions such as clicking, typing, scrolling, switching tabs, opening files, copying text, or reading an AI response. Similarly, do not create nodes simply because the actor is using a particular software application.

Only include a software interaction when it represents a meaningful work activity.

A behavioral graph corresponds to the workflow step at one time-stamp range. For example, if the workflow is “Step 2” → “Step 3” → “Step 2”, you must have 3 different behavioral graphs.

Use the video and workflow context to infer the actor’s behavior. Do not invent intentions that are unsupported by the evidence.

## Actors

Include meaningful behavioral activities performed by both:

\* \`user\`   
\* \`ai brand\`   
For AI, indicate whether it is \`claude\`, \`grok\`, or other types of AI. Most of AI here are Claude, but   
some users may use other AI too, so please check carefully.   
For AI behavior, describe the meaningful contribution to the user’s workflow rather than individual tool   
calls.   
Softwares   
Infer the softwares being used in the screen.   
Graph structure   
Construct a directed graph:   
\* Nodes represent behavioral activities.   
\* Edges represent transitions between activities.   
\* Include loops when the workflow revisits an earlier behavioral activity or repeatedly follows a   
behavioral cycle.   
\* Use edge types such as \`sequential\` or \`loop\` when appropriate.   
Output   
Return only valid JSON following the format below:   
{   
"session\_id": "...",   
"participant\_id": "...",   
"title": "...",   
"goal": "...",   
"steps": [   
{   
"key": "seg0a",   
"badge": "Step ID (e.g., Step-1, Custom-1, and so on",   
"step\_name": "Step name from steps.json",   
"step\_action": "Step action from steps.json",   
"time\_range": "Step time range from steps.json",   
"start\_seconds": 0,   
"end\_seconds": 600,   
"nodes": [   
{   
"id": "n1",   
"actor": "user",   
"label": "The behavioral node label",   
"detail": "A brief description of the behavioral activity.",   
"time": ["HH:MM:SS1-HH:MM:SS2", ...],   
"actor\_type": "user",   
"softwares": "...",   
"source\_steps": "..."   
},   
{   
"id": "n2",   
"actor": "claude or grok or gemini or chatgpt",   
"label": "The behavioral node label for an AI-authored   
action",   
"detail": "A brief description of the behavioral activity.",   
"time": ["HH:MM:SS1-HH:MM:SS2", ...],   
"actor\_type": "ai",   
"softwares": "...",

"source\_steps": "..."   
}   
],   
"edges": [   
{   
"from": "node\_id",   
"to": "node\_id",   
"type": "sequential"   
}   
]   
}   
]   
}   
Note on \`actor\` vs. \`actor\_type\`: \`actor\` is the specific actor performing the node — \`user\`, or the   
AI’s actual name (\`claude\`, \`grok\`, \`chatgpt\`, \`gemini\`, etc.). \`actor\_type\` is always just the coarse   
category — \`user\` or \`ai\` — never an AI brand name. Do not swap these: an AI node must have   
\`actor\_type: "ai"\` (with the brand name in \`actor\`), not \`actor\_type: "claude"\`.   
For each node:   
\* \`label\`: the behavioral node label   
\* \`detail\`: brief contextual description of the activity. Only include 2–3 short sentences!   
\* \`time\`: [the start and ending timestamps of that behavior]   
\* \`source\_steps\`: workflow step(s) supporting the node   
Before returning the result, verify that every node represents a meaningful goal-directed activity rather   
than a low-level interface action, and that the graph accurately reflects the workflow’s behavioral   
sequence and iterations.   
You must watch the video to correctly create the graph! Don’t rely on the .srt transcript since it   
can be wrong! Save the video frames for later use.   
Double-check if there are some loops in the graph. If there are loops, make sure that it is presented in the   
JSON file correctly. For example, if the loop is node A → node B → node A, there should be two edges   
(from node A to node B and node B back to node A). A loop can also be A → B → C → A → D, so   
please check carefully.

## A.5.1 HOW HUMANS CONSTRUCT THE GRAPH

While an LLM was used to generate initial workflow graphs from the video, think-aloud transcript, workflow document, step mappings, and AI chat logs collected from professionals, the research team manually reviewed and substantially reconstructed the graphs, often effectively from scratch.

## A.6 PROMPTS FOR EVALUATING MLLMS

We run each experiment under seven input conditions, one for every non-empty combination of video, workflow graph, and chat. The combined conditions reuse the parts of the single-source prompts, so we show only the video-only, graph-only, and chat-only prompts.

## A.6.1 EXPERIMENT 1A: ADOPTION BEHAVIOR PREDICTION PROMPT (WITH TARGET TIMESTAMP)

Experiment 1a: video-only   
You are given a video snippet of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal:   
Goal: {GOAL}   
The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an

![](images/52b3eed7c5379510df8e777fe39bd908ae92503a25f3a2592763fe67162ffda2.jpg)

![](images/316ba13934952f0056749246afe27ba9afb3303ef53890b74694afe5df5be827.jpg)

(a) The interface for professionals to align steps and timestamps.  
![](images/3f2f8c74a34d42531d123e7e7d06f1224d132b021c259d346eb89b5d94028163.jpg)  
(b) The interface for professionals to verify and revise workflow.  
Figure 14: A webpage for professionals to align timestamps with steps and verify the generated graphs. Professionals first verify workflow steps and their corresponding video timestamps (top), and subsequently inspect the workflow graph for each step (bottom).

## Experiment 1a: graph-only

You are given the workflow graph of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal (no video is provided for this task – rely only on the workflow graph below):   
Goal: {GOAL}   
The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an   
output in one step, and the user may act on that output either within the same step or in a later step.   
AI Action Step: {AI\_ACTION\_STEP}   
User Action Step: {USER\_ACTION\_STEP}   
Workflow graph for this portion of the workflow is defined as follows (each node is one action taken by   
the human or the agent; edges show the order they happened in):   
[n0] ({actor}) {label}: {details}   
[n1] ({actor}) {label}: {details}   
A directed edge X → Y means that action Y follows from or depends on action X. The graph is directed   
and may contain branches.   
Edges (order of actions):   
n0 -> n1   
n1 -> n2   
You must predict how the user will respond to the output of the target AI action (the node above   
matching the “AI Action Step” described earlier):   
\* ADOPT: Use the AI output as provided, without any modification.   
\* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or   
other adjustments.   
\* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.   
Use the workflow context available to make your prediction.   
Output exactly one label:   
\`ADOPT\`, \`MODIFY\`, or \`REJECT/BYPASS

## Experiment 1a: chat-only

You are given the chat of {PROFESSION\_PHRASE} working with an AI agent toward the following   
goal (no video is provided for this task – rely only on the chat below):   
Goal: {GOAL}   
The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an   
output in one step, and the user may act on that output either within the same step or in a later step.   
AI Action Step: {AI\_ACTION\_STEP}   
User Action Step: {USER\_ACTION\_STEP}   
Chat between the human and the AI agent during this portion of the workflow, in chronological order   
(each message from the human is wrapped in \`<human>\` tags, each message from the AI agent in   
\`<ai>\` tags):   
<human>   
{PROMPT}   
</human>   
<ai>

{AI\_RESPONSE}   
</ai>   
{...ADDITIONAL TURNS...}   
You must predict how the user will respond to the output of the target AI action:   
\* ADOPT: Use the AI output as provided, without any modification.   
\* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or   
other adjustments.   
\* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.   
Use the workflow context available to make your prediction.   
Output exactly one label:   
\`ADOPT\`, \`MODIFY\`, or \`REJECT/BYPASS

## A.6.2 EXPERIMENT 1B & 1C & 1D: ADOPTION BEHAVIOR FORECASTING PROMPT (CLIP TRIMMED TO AI ACTION)

## Experiment 1b & 1c & 1d: video-only

You are given a video snippet of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal:   
Goal: {GOAL}   
The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an   
output in one step, and the user may act on that output either within the same step or in a later step.   
AI Action Step: {AI\_ACTION\_STEP}   
You must predict how the user will respond to the output of this AI action:   
\* ADOPT: Use the AI output as provided, without any modification.   
\* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or   
other adjustments.   
\* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.   
Use the workflow context available to make your prediction.   
Output exactly one label:   
\`ADOPT\`, \`MODIFY\`, or \`REJECT/BYPASS

## Experiment 1b & 1c & 1d: graph-only

You are given the workflow graph of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal (no video is provided for this task – rely only on the workflow graph below):   
Goal: {GOAL}   
The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an   
output in one step, and the user may act on that output either within the same step or in a later step.   
AI Action Step: {AI\_ACTION\_STEP}   
Workflow graph for this portion of the workflow is defined as follows (each node is one action taken by   
the human or the agent; edges show the order they happened in):   
[n0] ({actor}) {label}: {details}   
[n1] ({actor}) {label}: {details}

A directed edge X → Y means that action Y follows from or depends on action X. The graph is directed   
and may contain branches.

Edges (order of actions):   
n0 -> n1   
n1 -> n2

You must predict how the user will respond to the output of the target AI action (the node above matching the “AI Action Step” described earlier):

\* ADOPT: Use the AI output as provided, without any modification.   
\* MODIFY: Use the AI output but modify it in any way, including minor edits, formatting changes, or   
other adjustments.   
\* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.

Use the workflow context available to make your prediction.

Output exactly one label:   
\`ADOPT\`, \`MODIFY\`, or \`REJECT/BYPASS\`

## Experiment 1b & 1c & 1d: chat-only

You are given the chat of {PROFESSION\_PHRASE} working with an AI agent toward the following goal (no video is provided for this task – rely only on the chat below):

Goal: {GOAL}

The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an output in one step, and the user may act on that output either within the same step or in a later step.

AI Action Step: {AI\_ACTION\_STEP}

Chat between the human and the AI agent during this portion of the workflow, in chronological order (each message from the human is wrapped in \`<human>\` tags, each message from the AI agent in \`<ai>\` tags):

<human> {PROMPT} </human>

<ai>   
{AI\_RESPONSE}   
</ai>

{...ADDITIONAL TURNS...}

You must predict how the user will respond to the output of the target AI action:

\* REJECT/BYPASS: Do not use the AI output; reject, ignore, replace, or proceed without it.

Use the workflow context available to make your prediction.

Output exactly one label:   
\`ADOPT\`, \`MODIFY\`, or \`REJECT/BYPASS

A.6.3 EXPERIMENT 2A: COLLABORATION RATING PREDICTION PROMPT  
![](images/be0d8aa9bfe0dcd448ddb56b8ab9b55fac8b27eea5f9ca2d1e1e7bc0a4f2dc1c.jpg)

Experiment 2a: graph-only   
You are given a video snippet of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal:   
Goal: {GOAL}   
The video shows a step of a multi-step workflow for accomplishing the goal:   
Step: {STEP\_NAME}   
The workflow consists of multiple steps that contribute to achieving this goal. The AI may produce an   
output in one step, and the user may act on that output either within the same step or in a later step.   
Workflow graph for this portion of the workflow is defined as follows (each node is one action taken by   
the human or the agent; edges show the order they happened in):   
[n0] ({actor}) {label}: {details}   
[n1] ({actor}) {label}: {details}   
...   
A directed edge X → Y means that action Y follows from or depends on action X. The graph is directed   
and may contain branches.   
Edges (order of actions):   
n0 -> n1   
n1 -> n2   
Read the graph above. Examine whether the agent understood the task, context, and instructions;   
whether it adapted appropriately when given feedback or corrections; whether it reduced the human’s   
effort, time, or cognitive load; whether it proactively suggested useful ideas or next steps; whether its   
output was accurate, relevant, and trustworthy with minimal correction needed; whether the interaction   
felt smooth, with the agent contributing at the right moments without creating unnecessary work; and   
whether the human was able to guide, redirect, or override the agent’s actions when desired.

Based on this, rate the overall collaboration experience for this step on a scale from 1 to 5, where 1   
means that the user is not satisfied with the collaboration experience with the AI agent and 5 means that   
the user is satisfied with the collaboration experience with the AI agent.   
Output exactly one integer, with no other text:   
1, 2, 3, 4, or 5

![](images/a51bb62e20bb18df8e88bf38fba18d132388eb707d9d9060c30ecca10418cd59.jpg)

## A.6.4 EXPERIMENT 2B: OUTPUT QUALITY RATING PREDICTION PROMPT

![](images/175687dd6e9a893bb003fb5777cd2abf17c9357fb90e0ede72087033f090fd3a.jpg)

rework, or was discarded; and whether it met the specific requirements of this step.   
Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is   
not satisfied with the output quality and 5 means that the user is satisfied with the output quality.   
Output exactly one integer, with no other text:   
1, 2, 3, 4, or 5

## Experiment 2b: graph-only

You are given a video snippet of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal:   
Goal: {GOAL}   
The video shows a step of a multi-step workflow for accomplishing the goal:   
Step: {STEP\_NAME}   
Workflow graph for this portion of the workflow is defined as follows (each node is one action taken by   
the human or the agent; edges show the order they happened in):   
[n0] ({actor}) {label}: {details}   
[n1] ({actor}) {label}: {details}   
A directed edge X → Y means that action Y follows from or depends on action X. The graph is directed   
and may contain branches.   
Edges (order of actions):   
n0 -> n1   
n1 -> n2   
...   
Read the graph above, focusing on the output the AI agent produced and how it was used during this   
step. Examine whether the output was accurate, relevant, and directly usable for the task; whether it   
required significant correction, rework, or was discarded; and whether it met the specific requirements of   
this step.   
Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is   
not satisfied with the output quality and 5 means that the user is satisfied with the output quality.   
Output exactly one integer, with no other text:   
1, 2, 3, 4, or 5

## Experiment 2b: chat-only

You are given a video snippet of {PROFESSION\_PHRASE} working with an AI agent toward the   
following goal:   
Goal: {GOAL}   
The video shows a step of a multi-step workflow for accomplishing the goal:   
Step: {STEP\_NAME}   
Chat between the human and the AI agent during this portion of the workflow, in chronological order   
(each message from the human is wrapped in \`<human>\` tags, each message from the AI agent in   
\`<ai>\` tags):   
<human>   
{PROMPT}

</human>   
<ai>   
{AI\_RESPONSE}   
</ai>   
{...ADDITIONAL TURNS...}   
Read the chat above, focusing on the output the AI agent produced and how it was used during this step.   
Examine whether the output was accurate, relevant, and directly usable for the task; whether it required   
significant correction, rework, or was discarded; and whether it met the specific requirements of this step.   
Based on this, rate the output quality for this step on a scale from 1 to 5, where 1 means that the user is   
not satisfied with the output quality and 5 means that the user is satisfied with the output quality.   
Output exactly one integer, with no other text:   
1, 2, 3, 4, or 5

## A.7 MODEL SELECTION

For proprietary models, we evaluate leading video-capable multimodal models from major providers: Gemini 3.1 Pro and Gemini 3.8 Flash (Google), GPT-6 Astra (OpenAI), and Claude Opus 5.5. Gemini natively accepts video as an input. GPT-6 Astra has no native video modality, so we sample evenly-spaced frames from each clip as its input; this also lets GPT-6 Astra serve as a frame-sampled point of comparison against Gemini’s genuinely video-native processing. We additionally evaluate Claude despite it likewise lacking native video input (also requiring frame sampling), because Claude is the specific agent our study participants interacted with during their recorded sessions, so evaluating it tests whether the very model embedded in the human-AI collaboration can itself judge or predict the outcome of that collaboration.

For open-weight models, we evaluate two representative video-capable Qwen3-VL variants (Bai et al., 2025): Qwen3-VL-30B-A3B-Instruct, a mixture-of-experts model (30B total parameters, 3B active per token) included to test whether MoE-style efficiency scaling preserves video-temporal understanding, and Qwen3-VL-32B-Instruct, a dense model of comparable total scale that instead invests its full parameter budget in deeper visual-language fusion at every token. Comparing the two isolates the effect of the MoE-vs-dense architectural choice at a fixed parameter scale.

## A.8 SAMPLE SIZE AND SELECTION

Claude evaluation subsets Claude was queried manually through its web interface, so for Experiments 1a, 2a and 2b we evaluated it on fixed subsets; Experiment 1b uses all instances. For 1a, we sampled 25 instances per label (N = 75), because the labels are imbalanced (13% reject), and a random sample would leave too few rejects for per-class F1. For 2a, we kept every step rated 1–4 and sampled 30 of the steps rated 5 (N = 78), since 5s make up 60% of the data. 2b uses the same steps, except six with no output-quality rating (N = 72). All subsets were drawn with a fixed seed, spread across participants

## A.9 EXPERIMENT RESULTS

Results for Experiment 1a (recognizing the user’s response to the AI’s action) are shown in Table 11, and results for Experiment 1b (forecasting the user’s response) are shown in Table 12. Results for Experiments 2a and 2b are given in Table 13 and Table 14, respectively.

## A.10 OUTPUT QUALITY × COLLABORATION RATING

As shown in Figure 15, output quality and the collaboration experience perceived by professionals are correlated.

<table><tr><td>Experiment</td><td># Clips</td><td>Mean ± std</td><td>Min</td><td>Max</td></tr><tr><td>Experiment 1a</td><td>692</td><td> $1 4 . 0 { \pm } 1 3 . 7 $ </td><td>0.8</td><td>73.0</td></tr><tr><td>Experiment 1b</td><td>691</td><td> $8 . 8 { \pm } 1 1 . 7$ </td><td>0.2</td><td>72.0</td></tr><tr><td>Experiment 1c</td><td>638</td><td> $3 7 . 9 { \scriptstyle \pm 2 2 . 7 }$ </td><td>1.5</td><td>117.1</td></tr><tr><td>Experiment 1d: last1</td><td>114</td><td> $1 0 . 1 { \pm } 9 . 6 $ </td><td>0.4</td><td>53.6</td></tr><tr><td>Experiment 1d: last2</td><td>114</td><td> $2 0 . 7 { \scriptstyle \pm 1 1 . 7 }$ </td><td>3.6</td><td>72.7</td></tr><tr><td>Experiment 1d: last3</td><td>114</td><td> $3 2 . 3 { \pm } 1 3 . 8 $ </td><td>7.0</td><td>87.8</td></tr><tr><td>Experiment 1d: full</td><td>114</td><td> $5 8 . 6 { \pm } 1 7 . 3 $ </td><td>26.1</td><td>117.1</td></tr><tr><td>Experiment 2a</td><td>350</td><td> $1 4 . 9 { \pm } 1 1 . 0 $ </td><td>0.8</td><td>66.5</td></tr><tr><td>Experiment 2b</td><td>373</td><td> $1 4 . 9 { \pm } 1 1 . 0 $ </td><td>0.8</td><td>66.5</td></tr></table>

Table 10: Length (in minutes) of the video clips given to the models in each experiment.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="3">Macro</td><td colspan="3">Per-class F1</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>Adopt</td><td>Modify</td><td>Reject</td></tr><tr><td rowspan="3">GPT-6-Astra</td><td>video-only</td><td>0.742</td><td>0.725</td><td>0.713</td><td>0.804</td><td>0.602</td><td>0.732</td></tr><tr><td>chat-only</td><td>0.648</td><td>0.673</td><td>0.658</td><td>0.777</td><td>0.644</td><td>0.552</td></tr><tr><td>video-graph-chat</td><td>0.699</td><td>0.742</td><td>0.673</td><td>0.812</td><td>0.480</td><td>0.727</td></tr><tr><td rowspan="3">Claude</td><td>video-only</td><td>0.736</td><td>0.648</td><td>0.676</td><td>0.843</td><td>0.556</td><td>0.630</td></tr><tr><td>chat-only</td><td>0.685</td><td>0.584</td><td>0.618</td><td>0.831</td><td>0.449</td><td>0.574</td></tr><tr><td>video-graph-chat</td><td>0.688</td><td>0.641</td><td>0.659</td><td>0.835</td><td>0.505</td><td>0.638</td></tr><tr><td rowspan="7">Gemini-3.1-Pro-Preview</td><td>video-only</td><td>0.616±0.005</td><td>0.622±0.009</td><td>0.617±0.007</td><td>0.795±0.005</td><td>0.479±0.014</td><td>0.577±0.008</td></tr><tr><td>graph-only</td><td>0.683±0.003</td><td>0.702±0.005</td><td>0.689±0.004</td><td>0.823±0.002</td><td>0.533±0.004</td><td>0.711±0.007</td></tr><tr><td>chat-only</td><td>0.589±0.004</td><td>0.617±0.006</td><td>0.600±0.005</td><td>0.764±0.005</td><td>0.494±0.018</td><td>0.543±0.007</td></tr><tr><td>video-graph</td><td>0.690±0.010</td><td>0.712±0.008</td><td>0.697±0.009</td><td>0.830±0.007</td><td>0.557±0.011</td><td>0.703±0.017</td></tr><tr><td>video-chat</td><td>0.618±0.009</td><td>0.633±0.003</td><td>0.621±0.006</td><td>0.784±0.008</td><td>0.493±0.007</td><td>0.587±0.014</td></tr><tr><td>graph-chat</td><td>0.691±0.009</td><td>0.693±0.009</td><td>0.682±0.007</td><td>0.804±0.005</td><td>0.525±0.016</td><td>0.716±0.006</td></tr><tr><td>video-graph-chat</td><td>0.675±0.007</td><td>0.673±0.006</td><td>0.665±0.005</td><td>0.804±0.006</td><td>0.518±0.006</td><td>0.671±0.008</td></tr><tr><td rowspan="7">Gemini-3.8-Flash</td><td>video-only</td><td>0.588±0.009</td><td>0.588±0.009</td><td>0.587±0.010</td><td>0.809±0.002</td><td>0.425±0.020</td><td>0.527±0.008</td></tr><tr><td>graph-only</td><td>0.699±0.011</td><td>0.700±0.007</td><td>0.699±0.009</td><td>0.849±0.003</td><td>0.566±0.006</td><td>0.684±0.018</td></tr><tr><td>chat-only</td><td>0.625±0.012</td><td>0.632±0.010</td><td>0.628±0.011</td><td>0.803±0.007</td><td>0.509±0.025</td><td>0.571±0.010</td></tr><tr><td>video-graph</td><td>0.690±0.006</td><td>0.700±0.006</td><td>0.694±0.006</td><td>0.849±0.004</td><td>0.545±0.013</td><td>0.689±0.002</td></tr><tr><td>video-chat</td><td>0.612±0.004</td><td>0.629±0.007</td><td>0.620±0.005</td><td>0.799±0.006</td><td>0.479±0.016</td><td>0.580±0.006</td></tr><tr><td>graph-chat</td><td>0.700±0.010</td><td>0.706±0.012</td><td>0.702±0.012</td><td>0.842±0.005</td><td>0.560±0.020</td><td>0.702±0.013</td></tr><tr><td>video-graph-chat</td><td>0.689±0.014</td><td>0.696±0.015</td><td>0.691±0.015</td><td>0.834±0.008</td><td>0.565±0.024</td><td>0.675±0.020</td></tr><tr><td rowspan="7">Qwen3-VL-30B-A3B-Instruct</td><td>video-only</td><td>0.500±0.017</td><td>0.557±0.023</td><td>0.511±0.019</td><td>0.680±0.009</td><td>0.371±0.035</td><td>0.481±0.025</td></tr><tr><td>graph-only</td><td>0.549±0.001</td><td>0.605±0.001</td><td>0.542±0.005</td><td>0.723±0.012</td><td>0.409±0.013</td><td>0.494±0.013</td></tr><tr><td>chat-only</td><td>0.540±0.009</td><td>0.518±0.002</td><td>0.502±0.002</td><td>0.773±0.001</td><td>0.306±0.004</td><td>0.427±0.002</td></tr><tr><td>video-graph</td><td>0.593±0.011</td><td>0.611±0.010</td><td>0.589±0.010</td><td>0.791±0.005</td><td>0.426±0.018</td><td>0.550±0.013</td></tr><tr><td>video-chat</td><td>0.583±0.026</td><td>0.541±0.018</td><td>0.536±0.019</td><td>0.788±0.007</td><td>0.327±0.033</td><td>0.494±0.018</td></tr><tr><td>graph-chat</td><td>0.586±0.009</td><td>0.607±0.011</td><td>0.570±0.012</td><td>0.785±0.006</td><td>0.382±0.009</td><td>0.544±0.024</td></tr><tr><td>video-graph-chat</td><td>0.611±0.007</td><td>0.611±0.001</td><td>0.591±0.005</td><td>0.806±0.004</td><td>0.401±0.012</td><td>0.568±0.002</td></tr><tr><td rowspan="7">Qwen3-VL-32B-Instruct</td><td>video-only</td><td>0.524±0.009</td><td>0.529±0.010</td><td>0.337±0.007</td><td>0.123±0.007</td><td>0.361±0.004</td><td>0.526±0.014</td></tr><tr><td>graph-only</td><td>0.589±0.003</td><td>0.558±0.001</td><td>0.476±0.002</td><td>0.430±0.004</td><td>0.365±0.001</td><td>0.634±0.005</td></tr><tr><td>chat-only</td><td>0.492±0.005</td><td>0.476±0.006</td><td>0.393±0.001</td><td>0.424±0.009</td><td>0.351±0.007</td><td>0.403±0.003</td></tr><tr><td>video-graph</td><td>0.597±0.004</td><td>0.615±0.002</td><td>0.559±0.004</td><td>0.621±0.007</td><td>0.414±0.002</td><td>0.643±0.004</td></tr><tr><td>video-chat</td><td>0.481±0.005</td><td>0.492±0.006</td><td>0.416±0.004</td><td>0.462±0.005</td><td>0.364±0.006</td><td>0.423±0.012</td></tr><tr><td>graph-chat</td><td>0.564±0.001</td><td>0.585±0.001</td><td>0.528±0.002</td><td>0.609±0.003</td><td>0.405±0.005</td><td>0.570±0.001</td></tr><tr><td>video-graph-chat</td><td>0.533±0.011</td><td>0.570±0.009</td><td>0.529±0.010</td><td>0.664±0.007</td><td>0.388±0.007</td><td>0.537±0.017</td></tr><tr><td rowspan="3">GPT-6-Astra</td><td>video-only</td><td>0.539±0.003</td><td>0.524±0.013</td><td>0.513±0.003</td><td>0.721±0.023</td><td>0.582±0.098</td><td>0.235±0.070</td></tr><tr><td>chat-only</td><td>0.365</td><td>0.371</td><td>0.368</td><td>0.695</td><td>0.408</td><td>0.000</td></tr><tr><td>video-graph-chat</td><td>0.546</td><td>0.512</td><td>0.512</td><td>0.696</td><td>0.592</td><td>0.250</td></tr><tr><td rowspan="3">Claude</td><td>video-only</td><td>0.610</td><td>0.483</td><td>0.492</td><td>0.790</td><td>0.434</td><td>0.250</td></tr><tr><td>chat-only</td><td>0.428</td><td>0.414</td><td>0.408</td><td>0.770</td><td>0.378</td><td>0.076</td></tr><tr><td>video-graph-chat</td><td>0.623</td><td>0.494</td><td>0.514</td><td>0.809</td><td>0.461</td><td>0.272</td></tr><tr><td rowspan="7">Gemini-3.1-Pro-Preview</td><td>video-only</td><td>0.461±0.026</td><td>0.402±0.011</td><td>0.391±0.013</td><td>0.710±0.012</td><td>0.333±0.018</td><td>0.129±0.031</td></tr><tr><td>graph-only</td><td>0.523±0.017</td><td>0.478±0.016</td><td>0.492±0.017</td><td>0.775±0.006</td><td>0.390±0.037</td><td>0.311±0.015</td></tr><tr><td>chat-only</td><td>0.399±0.011</td><td>0.387±0.016</td><td>0.390±0.014</td><td>0.694±0.007</td><td>0.313±0.010</td><td>0.162±0.027</td></tr><tr><td>video-graph</td><td>0.522±0.008</td><td>0.469±0.003</td><td>0.482±0.004</td><td>0.781±0.005</td><td>0.404±0.013</td><td>0.261±0.022</td></tr><tr><td>video-chat</td><td>0.431±0.007</td><td>0.415±0.004</td><td>0.420±0.005</td><td>0.743±0.003</td><td>0.320±0.028</td><td>0.195±0.018</td></tr><tr><td>graph-chat</td><td>0.488±0.031</td><td>0.446±0.023</td><td>0.459±0.026</td><td>0.760±0.012</td><td>0.357±0.037</td><td>0.259±0.033</td></tr><tr><td>video-graph-chat</td><td>0.509±0.025</td><td>0.458±0.014</td><td>0.472±0.017</td><td>0.773±0.009</td><td>0.374±0.025</td><td>0.269±0.018</td></tr><tr><td rowspan="7">Gemini-3.8-Flash</td><td>video-only</td><td>0.437±0.016</td><td>0.383±0.008</td><td>0.383±0.009</td><td>0.758±0.009</td><td>0.218±0.027</td><td>0.172±0.013</td></tr><tr><td>graph-only</td><td>0.557±0.013</td><td>0.458±0.011</td><td>0.475±0.014</td><td>0.797±0.005</td><td>0.333±0.031</td><td>0.293±0.008</td></tr><tr><td>chat-only</td><td>0.387±0.027</td><td>0.357±0.020</td><td>0.351±0.023</td><td>0.725±0.008</td><td>0.163±0.040</td><td>0.164±0.027</td></tr><tr><td>video-graph</td><td>0.526±0.020</td><td>0.454±0.007</td><td>0.467±0.010</td><td>0.779±0.006</td><td>0.319±0.013</td><td>0.302±0.020</td></tr><tr><td>video-chat</td><td>0.465±0.030</td><td>0.408±0.018</td><td>0.414±0.021</td><td>0.767±0.009</td><td>0.263±0.021</td><td>0.212±0.033</td></tr><tr><td>graph-chat</td><td>0.532±0.022</td><td>0.450±0.016</td><td>0.465±0.018</td><td>0.786±0.009</td><td>0.337±0.020</td><td>0.272±0.029</td></tr><tr><td>video-graph-chat</td><td>0.522±0.012</td><td>0.445±0.001</td><td>0.458±0.001</td><td>0.783±0.005</td><td>0.319±0.007</td><td>0.273±0.002</td></tr><tr><td rowspan="7">Qwen3-VL-30B-A3B-Instruct</td><td>video-only</td><td>0.389±0.010</td><td>0.404±0.011</td><td>0.364±0.007</td><td>0.501±0.004</td><td>0.305±0.019</td><td>0.285±0.012</td></tr><tr><td>graph-only</td><td>0.478±0.014</td><td>0.448±0.014</td><td>0.458±0.014</td><td>0.748±0.006</td><td>0.246±0.022</td><td>0.381±0.019</td></tr><tr><td>chat-only</td><td>0.474±0.016</td><td>0.387±0.007</td><td>0.385±0.008</td><td>0.746±0.002</td><td>0.203±0.010</td><td>0.205±0.016</td></tr><tr><td>video-graph</td><td>0.520±0.017</td><td>0.426±0.008</td><td>0.439±0.011</td><td>0.772±0.006</td><td>0.252±0.026</td><td>0.295±0.006</td></tr><tr><td>video-chat</td><td>0.396±0.038</td><td>0.358±0.013</td><td>0.343±0.018</td><td>0.769±0.002</td><td>0.105±0.016</td><td>0.155±0.036</td></tr><tr><td>graph-chat</td><td>0.507±0.019</td><td>0.439±0.008</td><td>0.444±0.011</td><td>0.781±0.006</td><td>0.193±0.019</td><td>0.358±0.013</td></tr><tr><td>video-graph-chat</td><td>0.447±0.007</td><td>0.389±0.003</td><td>0.383±0.004</td><td>0.782±0.003</td><td>0.142±0.008</td><td>0.227±0.006</td></tr><tr><td rowspan="7">Qwen3-VL-32B-Instruct</td><td>video-only</td><td>0.436±0.012</td><td>0.400±0.012</td><td>0.235±0.010</td><td>0.092±0.006</td><td>0.337±0.007</td><td>0.274±0.022</td></tr><tr><td>graph-only</td><td>0.494±0.012</td><td>0.395±0.002</td><td>0.316±0.007</td><td>0.454±0.002</td><td>0.335±0.004</td><td>0.158±0.023</td></tr><tr><td>chat-only</td><td>0.339±0.003</td><td>0.344±0.005</td><td>0.266±0.004</td><td>0.371±0.003</td><td>0.314±0.007</td><td>0.115±0.012</td></tr><tr><td>video-graph</td><td>0.524±0.038</td><td>0.428±0.015</td><td>0.370±0.014</td><td>0.523±0.012</td><td>0.358±0.010</td><td>0.229±0.039</td></tr><tr><td>video-chat</td><td>0.358±0.002</td><td>0.375±0.005</td><td>0.315±0.002</td><td>0.480±0.014</td><td>0.336±0.012</td><td>0.129±0.021</td></tr><tr><td>graph-chat</td><td>0.445±0.009</td><td>0.440±0.004</td><td>0.395±0.006</td><td>0.616±0.008</td><td>0.380±0.003</td><td>0.188±0.010</td></tr><tr><td>video-graph-chat</td><td>0.428±0.009</td><td>0.425±0.004</td><td>0.396±0.005</td><td>0.629±0.005</td><td>0.356±0.005</td><td>0.204±0.013</td></tr></table>

Table 11: Result for Experiment 1a. Mean±std over runs where a model has several; bold marks the best score within each model.

Table 12: Result for Experiment 1b. Mean±std over runs where a model has several; bold marks the best score within each model.

![](images/41fdd6b65e632977259d680e67208356846ac0cae811daf6d335a9324e88dc77.jpg)  
Figure 15: Professionals’ ratings of collaboration satisfaction and output quality across all steps.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="4">Overall</td><td colspan="4">MAE by true rating</td></tr><tr><td>MAE↓</td><td>Exact acc.</td><td>Within-1</td><td>Spearman ρ</td><td>Low 1–2</td><td>Moderate 3</td><td>Mod-high 4</td><td>High 5</td></tr><tr><td rowspan="3">GPT-6-Astra</td><td>video-only</td><td> $0 . 9 4 5 { \scriptstyle \pm 0 . 0 7 0 }$ </td><td> $0 . 2 4 4 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 8 4 1 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $\mathbf { 0 . 3 7 9 { \scriptstyle \pm 0 . 1 3 0 } }$ </td><td> $1 . 7 7 5 { \scriptstyle \pm 0 . 3 9 1 }$ </td><td> $0 . 6 8 4 \pm 0 . 0 3 1$ </td><td> $0 . 5 7 7 { \scriptstyle \pm 0 . 1 5 0 }$ </td><td>0.988±0.062</td></tr><tr><td>chat-only</td><td>1.246</td><td>0.183</td><td>0.611</td><td>0.107</td><td>1.714</td><td>0.636</td><td>0.759</td><td>1.480</td></tr><tr><td>video-graph-chat</td><td>0.740</td><td>0.365</td><td>0.887</td><td>0.263</td><td>1.429</td><td>0.917</td><td>0.373</td><td>0.802</td></tr><tr><td rowspan="3">Claude</td><td>video-only</td><td>0.858</td><td>0.266</td><td>0.865</td><td>0.136</td><td>1.675</td><td>0.742</td><td>0.275</td><td>1.131</td></tr><tr><td>chat-only</td><td>0.857</td><td>0.276</td><td>0.867</td><td>0.244</td><td>1.778</td><td>0.633</td><td>0.346</td><td>1.048</td></tr><tr><td>video-graph-chat</td><td>0.692</td><td>0.407</td><td>0.895</td><td>0.299</td><td>1.353</td><td>0.933</td><td>0.371</td><td>0.735</td></tr><tr><td rowspan="8">Gemini-3.1-Pro-Preview</td><td>video-only</td><td>0.685±0.006</td><td> $0 . 5 2 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 8 5 0 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 0 7 8 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td>3.175±0.050</td><td> $1 . 9 6 8 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td>0.961±0.009</td><td>0.048±0.006</td></tr><tr><td>graph-only</td><td>0.632±0.012</td><td> $0 . 5 0 9 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>0.872±0.003</td><td> $0 . 2 7 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td>1.563±0.027</td><td> $1 . 6 9 9 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td>0.221±0.020</td></tr><tr><td>chat-only</td><td> $0 . 9 9 2 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 4 7 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 7 5 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 0 7 8 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $2 . 6 1 1 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $1 . 8 5 6 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $1 . 1 3 1 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 5 7 6 { \scriptstyle \pm 0 . 0 3 0 }$ </td></tr><tr><td>video-graph</td><td> $0 . 6 3 5 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 5 2 5 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>0.862±0.010</td><td> $0 . 2 6 4 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $1 . 6 4 2 \substack { \pm 0 . 0 7 6 }$ </td><td> $1 . 7 3 1 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $0 . 9 5 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>0.161±0.021</td></tr><tr><td>video-chat</td><td></td><td>0.768±0.029 0.530±0.011</td><td>0.813±0.005</td><td>0.215±0.024</td><td>2.627±0.034</td><td>1.811±0.038</td><td>1.051±0.048</td><td>0.232±0.057</td></tr><tr><td>graph-chat</td><td> $0 . 6 3 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 5 5 9 } { \scriptstyle \pm 0 . 0 0 3 }$ </td><td>0.851±0.002</td><td> $\mathbf { 0 . 2 9 5 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $1 . 8 5 2 { \scriptstyle \pm 0 . 1 1 6 }$ </td><td> $1 . 7 7 8 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 9 8 4 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td>0.100±0.007</td></tr><tr><td>video-graph-chat</td><td> $\mathbf { 0 . 6 0 9 { \scriptstyle \pm 0 . 0 1 9 } }$ </td><td> $0 . 5 5 0 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 8 6 4 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 6 1 { \scriptstyle \pm 0 . 0 4 8 }$ </td><td> $1 . 7 8 4 { \scriptstyle \pm 0 . 2 3 8 }$ </td><td> $1 . 8 2 2 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 9 6 2 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 0 7 9 { \scriptstyle \pm 0 . 0 2 0 }$ </td></tr><tr><td></td><td>0.800±0.029</td><td> $0 . 5 0 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.819±0.014</td><td> $0 . 0 1 1 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $3 . 1 0 8 { \scriptstyle \pm 0 . 2 8 9 }$ </td><td> $1 . 9 1 4 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $1 . 0 1 5 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td>0.245±0.026</td></tr><tr><td rowspan="7">Gemini-3.8-Flash</td><td>video-only</td><td> $0 . 6 0 7 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td></td><td>0.882±0.002</td><td> $0 . 2 7 6 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $1 . 7 3 8 { \scriptstyle \pm 0 . 0 4 8 }$ </td><td> $1 . 6 8 8 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 8 7 0 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>0.149±0.008</td></tr><tr><td>graph-only chat-only</td><td> $0 . 8 9 2 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 5 2 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td></td><td> $0 . 0 8 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $2 . 5 3 7 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $1 . 6 4 4 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $1 . 0 7 6 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td>0.471±0.030</td></tr><tr><td>video-graph</td><td> $0 . 6 1 1 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 4 8 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 7 7 7 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td></td><td> $1 . 8 2 5 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $1 . 8 4 9 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td>0.925±0.018</td><td>0.094±0.008</td></tr><tr><td>video-chat</td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td> $0 . 5 2 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.876±0.003 0.826±0.007</td><td> $0 . 2 5 5 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td></td><td></td><td></td><td>0.180±0.053</td></tr><tr><td>graph-chat</td><td> $0 . 6 0 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 5 4 3 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td></td><td> $0 . 1 5 9 { \scriptstyle \pm 0 . 0 4 0 }$   $0 . 2 7 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $2 . 8 4 3 { \scriptstyle \pm 0 . 1 8 0 }$   $1 . 9 2 6 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $1 . 8 1 1 { \scriptstyle \pm 0 . 0 1 9 }$   $1 . 7 0 0 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td></td></tr><tr><td>video-graph-chat 0.589±0.011 0.559±0.008</td><td></td><td> $0 . 5 3 7 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 8 7 8 { \scriptstyle \pm 0 . 0 0 3 }$  0.868±0.002</td><td>0.299±0.027</td><td>1.980±0.068</td><td>1.789±0.051</td><td> $0 . 9 2 4 { \scriptstyle \pm 0 . 0 3 2 }$  0.924±0.019</td><td>0.094±0.004 0.048±0.007</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Qwen3-VL-30B-A3B-Instruct video-graph</td><td>video-only</td><td> $0 . 8 9 8 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $0 . 3 3 0 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 8 1 4 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 1 0 2 { \scriptstyle \pm 0 . 0 7 5 }$ </td><td> $2 . 2 2 5 { \scriptstyle \pm 0 . 1 0 0 }$ </td><td> $1 . 0 4 3 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $0 . 6 3 5 { \scriptstyle \pm 0 . 0 8 6 }$ </td><td>0.887±0.083</td></tr><tr><td>graph-only</td><td> $0 . 7 7 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 9 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 9 3 0 } 2 0 . 0 0 2$ </td><td> $0 . 1 9 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $1 . 7 7 0 { \scriptstyle \pm 0 . 0 5 5 }$ </td><td> $0 . 9 8 9 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $0 . 2 7 8 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td>0.910±0.003</td></tr><tr><td>chat-only</td><td></td><td></td><td>0.827±0.031 0.438±0.030 0.844±0.004</td><td>0.145±0.047</td><td>2.500±0.056</td><td>1.467±0.033</td><td>0.764±0.074</td><td>0.549±0.032</td></tr><tr><td></td><td> $\mathbf { 0 . 6 2 2 { \scriptstyle \pm 0 . 0 1 7 } }$ </td><td> $0 . 4 8 2 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 9 0 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 2 7 5 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $1 . 9 9 2 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $1 . 3 8 7 { \scriptstyle \pm 0 . 1 1 2 }$ </td><td> $0 . 6 0 9 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 3 5 8 { \scriptstyle \pm 0 . 0 0 9 }$ </td></tr><tr><td>video-chat</td><td> $0 . 7 2 1 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 5 1 9 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 8 3 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 1 7 0 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $2 . 7 6 5 { \scriptstyle \pm 0 . 1 5 6 }$ </td><td> $1 . 7 7 8 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 9 0 8 { \scriptstyle \pm 0 . 0 4 6 }$ </td><td> $0 . 2 1 1 { \scriptstyle \pm 0 . 0 3 1 }$ </td></tr><tr><td>graph-chat</td><td> $0 . 6 8 4 \pm 0 . 0 1 1$ </td><td> $0 . 4 3 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 0 2 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 2 7 7 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $1 . 9 0 7 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $1 . 1 6 7 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 5 6 2 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 5 1 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td></tr><tr><td>video-graph-chat</td><td> $0 . 6 3 4 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 5 2 0 { \scriptstyle \pm 0 . 0 1 1 } }$ </td><td> $0 . 8 7 5 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 2 8 5 { \scriptstyle \pm 0 . 0 3 3 } }$ </td><td> $2 . 0 5 9 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $1 . 5 4 4 { \scriptstyle \pm 0 . 0 8 4 }$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td> $0 . 2 2 2 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td rowspan="7">Qwen3-VL-32B-Instruct</td><td>video-only</td><td> $1 . 1 9 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 1 9 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 6 3 4 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 1 3 0 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $1 . 4 2 5 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $0 . 3 3 3 { \scriptstyle \pm 0 . 1 4 5 }$ </td><td> $0 . 7 3 9 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $1 . 5 7 8 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>graph-only</td><td> $0 . 6 9 2 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 4 1 5 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 8 9 1 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 2 3 6 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $1 . 7 5 4 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $1 . 0 4 3 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 4 7 9 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 6 4 0 { \scriptstyle \pm 0 . 0 0 9 }$ </td></tr><tr><td>chat-only</td><td> $0 . 8 0 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 4 7 6 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 8 3 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 8 7 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $2 . 5 1 9 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $1 . 5 4 4 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 8 9 9 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 4 2 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>video-graph</td><td> $0 . 6 5 7 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 4 2 6 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 1 4 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $0 . 2 4 3 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $1 . 8 2 5 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $1 . 2 2 6 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td> $0 . 4 9 3 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 5 3 5 { \scriptstyle \pm 0 . 0 2 5 }$ </td></tr><tr><td>video-chat</td><td></td><td>0.719±0.004 0.496±0.007 0.840±0.004</td><td></td><td>0.172±0.007</td><td> $2 . 5 8 8 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>1.567±0.088</td><td>0.861±0.037</td><td> $0 . 2 8 9 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td>graph-chat</td><td> $0 . 6 1 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 5 2 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.878±0.003</td><td> $\mathbf { 0 . 2 9 7 { \scriptstyle \pm 0 . 0 1 2 } }$ </td><td>2.259±0.085</td><td> $1 . 4 8 9 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 7 7 7 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.184±0.013</td></tr><tr><td>video-graph-chat</td><td> $\mathbf { 0 . 6 0 4 } \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 5 3 2 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td>0.876±0.002</td><td> $0 . 2 8 2 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td>2.196±0.034</td><td> $1 . 5 8 9 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 7 8 1 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 1 5 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr></table>

Table 13: Result for Experiment 2a. Mean±std over runs where a model has several; bold marks the best score within each model. Band columns give the MAE within each true-rating band.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="4">Overall</td><td colspan="4">MAE by true rating</td></tr><tr><td>MAE↓</td><td>Exact acc.</td><td>Within-1</td><td>Spearman ρ</td><td>Low 1-2</td><td>Moderate 3</td><td>Mod-high 4</td><td>High 5</td></tr><tr><td rowspan="3">GPT-6-Astra</td><td>video-only</td><td>0.912±0.084</td><td>0.332±0.067</td><td>0.805±0.030</td><td>0.352±0.101</td><td>1.853±0.255</td><td>0.711±0.077</td><td>0.710±0.157</td><td>0.855±0.105</td></tr><tr><td>chat-only</td><td>1.134</td><td>0.221</td><td>0.713</td><td>0.163</td><td>1.500</td><td>0.727</td><td>0.788</td><td>1.320</td></tr><tr><td>video-graph-chat</td><td>0.765</td><td>0.438</td><td>0.854</td><td>0.311</td><td>1.500</td><td>0.750</td><td>0.500</td><td>0.741</td></tr><tr><td rowspan="3">Claude</td><td>video-only</td><td>0.817</td><td>0.344</td><td>0.849</td><td>0.208</td><td>1.568</td><td>0.625</td><td>0.405</td><td>1.018</td></tr><tr><td>chat-only</td><td>0.973</td><td>0.267</td><td>0.809</td><td>0.137</td><td>1.789</td><td>1.059</td><td>0.610</td><td>1.052</td></tr><tr><td>video-graph-chat</td><td>0.741</td><td>0.362</td><td>0.917</td><td>0.322</td><td>1.556</td><td>0.765</td><td>0.280</td><td>0.897</td></tr><tr><td rowspan="7">Gemini-3.1-Pro-Preview</td><td>video-only</td><td>0.764±0.016</td><td>0.508±0.002</td><td>0.821±0.006</td><td>0.046±0.039</td><td>3.225±0.050</td><td>2.000±0.000</td><td>1.023±0.018</td><td>0.062±0.017</td></tr><tr><td>graph-only</td><td>0.750±0.022</td><td>0.471±0.008</td><td>0.836±0.005</td><td>0.299±0.021</td><td>1.428±0.050</td><td>1.458±0.014</td><td>0.891±0.030</td><td>0.440±0.030</td></tr><tr><td>chat-only</td><td> $1 . 0 6 9 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 4 3 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 7 2 9 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 0 9 3 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td>2.561±0.110</td><td> $1 . 6 7 6 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td>1.248±0.063</td><td>0.621±0.066</td></tr><tr><td>video-graph</td><td>0.763±0.005</td><td> $0 . 4 7 8 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>0.825±0.004</td><td>0.280±0.013</td><td> $1 . 5 1 4 { \scriptstyle \pm 0 . 1 0 9 }$ </td><td> $1 . 5 5 8 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 9 7 4 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td>0.384±0.033</td></tr><tr><td>video-chat</td><td>0.895±0.032</td><td>0.485±0.004</td><td> $0 . 7 7 6 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 1 3 9 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $3 . 1 2 3 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $1 . 7 6 5 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $1 . 1 4 2 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.246±0.056</td></tr><tr><td>graph-chat</td><td> $\mathbf { 0 . 7 3 4 { \scriptstyle \pm 0 . 0 3 1 } }$ </td><td> $0 . 4 8 5 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 8 4 5 { \scriptstyle \pm 0 . 0 1 4 } }$ </td><td> $0 . 2 8 7 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $1 . 7 8 9 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td> $1 . 5 9 8 { \scriptstyle \pm 0 . 0 9 0 }$ </td><td> $0 . 9 5 1 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td>0.261±0.035</td></tr><tr><td>video-graph-chat</td><td>0.759±0.010</td><td>0.479±0.009</td><td>0.834±0.004</td><td>0.253±0.010</td><td>1.842±0.053</td><td> $1 . 7 0 6 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 9 7 6 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td>0.262±0.018</td></tr><tr><td rowspan="7">Gemini-3.8-Flash</td><td>video-only</td><td> $0 . 7 9 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 5 0 3 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 4 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $3 . 1 6 7 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $1 . 9 7 5 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $1 . 0 4 9 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td>0.114±0.036</td></tr><tr><td>graph-only</td><td>0.652±0.016 0.513±0.009</td><td></td><td>0.870±0.007</td><td> $0 . 2 8 6 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $1 . 6 5 9 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $1 . 5 1 7 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td>0.810±0.023</td><td>0.260±0.005</td></tr><tr><td>chat-only</td><td>1.113±0.004</td><td>0.431±0.004</td><td>0.697±0.004</td><td>0.094±0.008</td><td> $2 . 5 9 6 { \scriptstyle \pm 0 . 1 3 2 }$ </td><td>1.804±0.045</td><td>1.313±0.019</td><td>0.633±0.025</td></tr><tr><td>video-graph</td><td> $\mathbf { 0 . 6 4 3 { \scriptstyle \pm 0 . 0 0 3 } }$ </td><td>0.524±0.004</td><td>0.869±0.003</td><td> $0 . 3 0 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td>1.877±0.025</td><td> $1 . 6 6 7 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 8 9 1 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td>0.138±0.009</td></tr><tr><td>video-chat</td><td> $0 . 7 7 7 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 5 0 9 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.811±0.009</td><td> $0 . 1 8 7 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $2 . 8 4 2 { \scriptstyle \pm 0 . 0 9 1 }$ </td><td> $1 . 8 5 3 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $1 . 0 1 2 { \scriptstyle \pm 0 . 0 5 3 }$ </td><td>0.108±0.008</td></tr><tr><td>graph-chat</td><td> $0 . 6 6 9 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 5 1 3 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td>0.864±0.002</td><td> $0 . 3 0 6 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $1 . 8 0 7 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $1 . 6 1 8 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 9 1 9 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.146±0.020</td></tr><tr><td>video-graph-chat</td><td> $0 . 6 5 8 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 5 2 5 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td>0.862±0.006</td><td> $\mathbf { 0 . 3 2 0 { \scriptstyle \pm 0 . 0 2 2 } }$ </td><td> $1 . 7 7 2 { \scriptstyle \pm 0 . 1 3 2 }$ </td><td> $1 . 6 4 7 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 9 0 7 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td>0.129±0.022</td></tr><tr><td rowspan="7">Qwen3-VL-30B-A3B-Instruct</td><td>video-only</td><td> $1 . 2 8 1 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 2 7 8 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 6 4 1 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 0 4 8 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $1 . 9 7 8 { \scriptstyle \pm 0 . 1 9 0 }$ </td><td> $1 . 1 2 5 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 9 7 1 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td>1.415±0.019</td></tr><tr><td>graph-only</td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 3 8 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.876±0.010</td><td> $0 . 2 0 6 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $1 . 5 5 8 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 4 3 7 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 8 2 9 { \scriptstyle \pm 0 . 0 0 8 }$ </td></tr><tr><td>chat-only</td><td>0.926±0.019</td><td> $0 . 4 3 9 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td>0.783±0.013</td><td> $0 . 1 3 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $2 . 7 0 2 { \scriptstyle \pm 0 . 0 6 1 }$ </td><td> $1 . 5 2 9 { \scriptstyle \pm 0 . 0 7 8 }$ </td><td> $0 . 9 2 3 { \scriptstyle \pm 0 . 0 6 0 }$ </td><td>0.546±0.026</td></tr><tr><td>video-graph</td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 4 4 5 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td>0.868±0.004</td><td> $0 . 2 2 5 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $1 . 5 2 9 { \scriptstyle \pm 0 . 1 0 9 }$ </td><td> $1 . 3 0 0 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $0 . 6 5 2 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td>0.566±0.021</td></tr><tr><td>video-chat</td><td>0.884±0.040</td><td> $0 . 4 7 3 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $0 . 7 8 1 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 1 1 3 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $2 . 8 7 7 { \scriptstyle \pm 0 . 1 5 2 }$ </td><td>1.706±0.078</td><td> $1 . 0 0 8 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td>0.348±0.072</td></tr><tr><td>graph-chat</td><td> $0 . 7 1 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>0.469±0.006</td><td>0.866±0.004</td><td>0.268±0.011</td><td>1.789±0.139</td><td> $1 . 3 6 3 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td>0.772±0.014</td><td>0.376±0.007</td></tr><tr><td>video-graph-chat</td><td> $\mathbf { 0 . 6 9 9 } 2 0 . 0 2 3$ </td><td> $\mathbf { 0 . 4 9 9 } 2 \mathrm { { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $0 . 8 5 8 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 3 2 3 { \scriptstyle \pm 0 . 0 2 7 } }$ </td><td> $1 . 6 6 7 { \scriptstyle \pm 0 . 1 3 2 }$ </td><td> $1 . 5 8 8 { \scriptstyle \pm 0 . 1 3 5 }$ </td><td> $0 . 9 1 5 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td>0.232±0.021</td></tr><tr><td rowspan="7">Qwen3-VL-32B-Instruct</td><td>video-only</td><td>1.223±0.013</td><td>0.202±0.007</td><td>0.633±0.019</td><td>0.153±0.005</td><td> $1 . 4 4 2 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td>0.408±0.123</td><td>0.828±0.046</td><td>1.601±0.026</td></tr><tr><td>graph-only</td><td> $0 . 7 7 5 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 3 8 5 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td>0.849±0.003</td><td> $0 . 2 5 4 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>1.558±0.025</td><td> $0 . 8 5 8 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td>0.509±0.015</td><td>0.825±0.016</td></tr><tr><td>chat-only</td><td> $0 . 9 2 5 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $0 . 4 1 3 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td>0.794±0.000</td><td> $0 . 1 6 5 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $2 . 4 5 6 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $1 . 3 7 3 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $0 . 9 3 5 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td>0.607±0.011</td></tr><tr><td>video-graph</td><td> $0 . 7 3 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 4 2 5 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td>0.858±0.007</td><td> $0 . 2 5 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $1 . 6 7 4 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $1 . 0 0 8 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td>0.520±0.025</td><td>0.689±0.040</td></tr><tr><td>video-chat</td><td> $0 . 8 0 4 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $0 . 4 3 6 { \scriptstyle \pm 0 . 0 2 3 }$ </td><td> $0 . 8 1 7 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 1 9 9 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $2 . 4 0 4 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td> $1 . 2 8 4 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td>0.813±0.007</td><td>0.470±0.048</td></tr><tr><td>graph-chat</td><td>0.686±0.013</td><td> $0 . 4 5 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>0.898±0.009</td><td> $0 . 3 1 7 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $1 . 9 8 2 { \scriptstyle \pm 0 . 0 3 0 }$ </td><td>1.118±0.088</td><td>0.703±0.014</td><td>0.399±0.008</td></tr><tr><td>video-graph-chat 0.661±0.010</td><td></td><td>0.478±0.008</td><td>0.893±0.008</td><td> $\mathbf { 0 . 3 1 9 { \scriptstyle \pm 0 . 0 2 2 } }$ </td><td> $2 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 2 1 6 { \scriptstyle \pm 0 . 0 4 5 }$ </td><td>0.703±0.007</td><td>0.324±0.014</td></tr></table>

Table 14: Result for Experiment 2b. Mean±std over runs where a model has several; bold marks the best score within each model. Band columns give the MAE within each true-rating band.

## A.11 MODEL COMPARISON ON COMMON INSTANCE SETS

Due to resource limitations, GPT-6-Astra was evaluated on fewer instances than the other models.   
For a fair comparison, we also evaluate all models on the same instances used for GPT-6-Astra.   
Scores therefore differ from those in the full tables (Tables 11 and 12).

<table><tr><td>Model</td><td>F1</td><td>Runs</td></tr><tr><td>GPT-6-Astra</td><td>0.713</td><td>1</td></tr><tr><td>Claude</td><td>0.699</td><td>1</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>0.667</td><td>3</td></tr><tr><td>Gemini-3.8-Flash</td><td>0.620</td><td>3</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>0.560</td><td>3</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>0.374</td><td>3</td></tr></table>

Table 15: F1 for Experiment 1a on the common set of 279 instances from 32 participants. The common set is limited by GPT-6-Astra, which has 279 valid outputs.

<table><tr><td>Model</td><td>F1</td><td>Runs</td></tr><tr><td>Claude</td><td>0.533</td><td>1</td></tr><tr><td>GPT-6-Astra</td><td>0.511</td><td>1</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>0.399</td><td>3</td></tr><tr><td>Gemini-3.8-Flash</td><td>0.392</td><td>3</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>0.375</td><td>3</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>0.221</td><td>3</td></tr></table>

Table 16: F1 for Experiment 1b on the common set of 349 instances from 37 participants. The common set is limited by GPT-6-Astra, which has 349 valid outputs. GPT-6-Astra’s score is based on its single run covering all 349 instances.

## A.12 POST-INTERACTION QUESTIONNAIRE

## OVERVIEW: ALL BEHAVIORAL LIKERT ITEMS (Q13–Q38)

![](images/a113e067c5582dde54a9842fde4c02df16f045c8d61b53f495fd6fa9f63c5b5c.jpg)  
Figure 16: Mean rating (1–10) for each of the 26 AI-collaboration behavior items, sorted low to high (error bars: 1 SD). The dashed line marks the scale midpoint.

![](images/d418487357a29fe2facabea6918a47c0b74ed8ab13cb3af06ae5fd82fa3997ff.jpg)  
(a) Q1. After watching the session 1 recording again and looking at the extracted workflow, how natural do you think your workflow there compared to your general workflow?

Q2. If you answered not natural at all in the previous question, why do you think so? Openended response  
![](images/851f95feaac59a0f689af933d081418205a48df8d8cda625fcf998e25586d6d6.jpg)  
(a) Q3. How difficult was the problem you worked on in Session 1?

![](images/714a37d3ccc2b44845ec0b4aae67438c9d115d3ad77cfd23f3e43695908433c1.jpg)  
(b) Q4. Have you ever done problem 1 manually without agentic AI’s help?

![](images/19679164b0fafe1426318ad798a7ad24668567f7fb1faa27663fb6ac665c5422.jpg)  
(a) Q5. How many hours on average did it take or do you think it takes for you to finish the problem for session 1 without agentic AI’s help?

![](images/87075dff18db723f29f0873bce07a86ed9c45abe0079de049b8b0d7c0143bab5.jpg)  
(b) Q6. After watching the session 2 recording again and looking at the extracted workflow, how natural do you think your workflow there compared to your general workflow?

Q7. If you answered not natural at all in the previous question, why do you think so? Openended response  
![](images/c4fdc87054b3e6cc089b139f29954b73e05d49cec8c5ab536df5549e3fc01a90.jpg)  
(a) Q8. How difficult was the problem you worked on in Session 2?

![](images/5b7d2d9ffa19b6233bd0f1d57aeed4e36cc9e7ccc4ad995b0a2a7ddb9d4a1df5.jpg)  
(b) Q9. Have you ever done problem 2 manually without agentic AI’s help?

![](images/44f0c21ec352c3cebe5f4e2dadc803296906c148a16b6c571375fd0401799c21.jpg)  
(a) Q10. How many hours on average did it take or do you think it takes for you to finish the problem for session 2 without agentic AI’s help?

Q11. For either or both session 1 and 2, were there any points where you changed your plan or workflow? What prompted that? Open-ended response

Q12. If you did a similar task again, would you approach it differently? If so, what would you change? Open-ended response.

![](images/7df36bd92bcc3ad33632ef568202ebd521ee2b67f6151027afd429baca41d975.jpg)  
(a) Q13. I prefer letting AI complete substantial portions of my work before reviewing it.

![](images/9027e2d9e1c632f71f79ba9128a1a06bf9e3daa338db9e39d84b5c26ab176dc8.jpg)  
(b) Q14. I frequently ask AI to generate complete drafts instead of working incrementally.

![](images/a77ed8c34bc2b16a215488b7d049d3adbae02b3864d05b33a7bc1a53171dd424.jpg)  
(a) Q15. When possible, I let AI handle routine subtasks.

![](images/6ac4ac7de5bacef378e2876bee5e9e742cac37801a3c8a413d19c138908b0f52.jpg)  
(b) Q16. I prefer completing most work myself before involving AI.

![](images/fe4798de3ac69c3893a7d6d616e1ee8ed3e576ce4fd966b553283732442ae29e.jpg)  
(a) Q17. I usually verify AI-generated information using external sources.

![](images/85b52e1fab23baa6ecee95512851fa4ed18847e0e6123b0193dcc74dbd12eadd.jpg)  
(b) Q18. I rarely use AI output without checking it.

![](images/6509a1475b6851e40aa79b148b6fa0c09111346e783575198a04b07415764ac1.jpg)  
(a) Q19. I like understanding how AI reached its conclusions.

![](images/0f1c52730692474df77e87b1fe2e3b46d243288eeb75c2e17b58f77994e0de22.jpg)  
(b) Q20. If the answer looks reasonable, I usually move on without further verification.

![](images/ef7141951d35ef9b2e39a73301f047402cf9be02457a2c65ca72f2a74eb42663.jpg)  
(a) Q21. I often redirect AI when it takes an unhelpful direction.

![](images/c0a0df04a3d1f70a71e5b0e49759d0bf407db91a697f618cd2f792b7eab7f54c.jpg)  
(b) Q22. I refine my requests multiple times until the AI matches my intent.

![](images/9944ed25ed21bcd1cbdd28b1d8493b9956413293c12d22d5f915d5f6bdcf0af6.jpg)  
(a) Q23. I actively shape how the collaboration unfolds.

![](images/1a54988004d018e322e745be3dd6d65f3e58a89bc4d1d89e27498eed85d8b68e.jpg)  
(b) Q24. I usually follow the direction suggested by the AI.

## EXPLORATION OF ALTERNATIVES

![](images/e4dbf6133ccd7a553e0d4dfd6740e136434746ce517d59930cb1e4e5a6a46a19.jpg)  
(a) Q25. I use AI to explore multiple perspectives before deciding.

![](images/21d1412b8b4c0968729ed3e2972d695f29938c86c7bdf2b04dbdc71918d46602.jpg)  
(b) Q26. I intentionally ask AI for alternative viewpoints.

![](images/45dcc2bd45bf4b80f145f4b05a18d53b661de382695e6f27c090794736c3c0bd.jpg)  
(a) Q27. I enjoy exploring ideas with AI before committing to one.

![](images/74f84ea03d91a69dfd22c596a386e347ca81b01c25ce63eb63cbd88966cde3a2.jpg)  
(b) Q28. I usually stop searching once I find one satisfactory answer.

![](images/4831c49494c21d58f793456219abbf9b00e7a55824efb9a16d1741b47d43bff6.jpg)  
(a) Q29. I frequently combine information from AI with external documents.

![](images/9f330d36be613acf043632203953c6acc1afa56bd786aa29a5de8960168e2b31.jpg)  
(b) Q30. I compare AI responses with academic papers or other trusted sources.

TOOL-SWITCHING BEHAVIOR  
![](images/bf7206d14d3e673ea415a42725b6155f8062056e9c50a5800075725a17933c82.jpg)  
(a) Q31. My workflow often involves moving between AI and other tools.

![](images/26137d920a788978b151a6847194d83ee27f0b39351c212e7ff260f0ba9b5bf7.jpg)  
(b) Q32. Most of my work happens within a single AI chat.

## PLANNING BEHAVIOR

![](images/117636c02b969ae9f48e676a86d62d555702ada9ca6ffb5cf40a0ba50c6fb0e1.jpg)  
(a) Q33. Before using AI, I usually have a clear plan.

![](images/a1b8b9c4f8dade09afc3c80e9be940e4ec7acb68eddc572674579e3aabf0bb24.jpg)  
(b) Q34. I break complex tasks into smaller steps before asking AI.

![](images/7fc3883cb75e50c1abb4376937f4ee141aa270804f59c83f34a7b085dbb42a27.jpg)  
(a) Q35. I use AI to execute a plan rather than determine one.

## REFLECTION & REFRAMING

![](images/09bb0c46e26601f5b327743150bfa185542935339fe66f9759f1eef47e2a0c11.jpg)  
(a) Q36. I often reconsider my approach after interacting with AI.

![](images/01dee481863d8f6180e91f9d90e87dd596775d3ccde348b2d970a6fcee919e77.jpg)  
(b) Q37. Working with AI changes how I think about the problem.

![](images/4e2dcb6dd6a2db90711f09f29048c6a35536f28d8ba9cd2db064afb4f03f5c0a.jpg)  
(a) Q38. AI helps me discover assumptions I hadn’t considered.

![](images/226724b7890ad8f65ead5633a3a0463b31eda950b671a49b7119bf15c0749c2b.jpg)  
(a) Q39. What is the average hourly wage (in US Dollar) for a job similar to yours?

![](images/35106046420aed249dc6784cecfb59b282e14f20cf96de704fb9e004d2c5a4a8.jpg)  
Figure 37: Q40. How serious would an undetected material error be in the tasks you did for this study?

![](images/de85591d65d3e604497350152ecb32ff030db3808c4fe12408e279a598996d5e.jpg)  
Figure 38: Q41. Estimated financial/resource cost of an undetected error:

![](images/753226c5055bec00b88126a407f4e2971283c7d5b73d4662f7a2669343334707.jpg)  
Figure 39: Q42. Did you notice any issues in the AI content during the tasks you did for this study?

![](images/c625f712e1622e8fe89dcfa7b07d9c87d4fadf523dee19a846c1fe4a3d7aa509.jpg)  
(a) Q43. Estimated chance your final output still contains a material error: