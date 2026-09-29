# JUST-IN-TIME AGENT MEMORY WITH RUNTIME AGENTIC RESEARCH

Bingyu Yan<sup>1</sup>, Chaofan Li<sup>1</sup>, Hongjin Qian<sup>1,2</sup>, Shuqi Lu<sup>1</sup>, Chaozhuo Li<sup>1</sup>, Zheng Liu<sup>1,3∗</sup> <sup>1</sup> Beijing Academy of Artificial Intelligence

<sup>2</sup> Peking University

<sup>3</sup> Hong Kong Polytechnic University

{zhengliu1026}@gmail.com

## ABSTRACT

Memory is critical for AI agents. Many existing agent-memory systems follow an Ahead-of-Time (AOT) design, constructing memory before a specific request arrives. While this reduces online serving cost, such request-agnostic memory construction can discard fine-grained information that later becomes important. To address this limitation, we propose Just-In-Time Agent Memory (JAM), a trainable framework for query-conditioned context construction at runtime. A Memorizer preserves complete raw histories in a hierarchical page-store with compact navigational summaries, while a Researcher iteratively retrieves, inspects, and integrates evidence for each request. To train these memory-use behaviors, we introduce Memory-Gym, an evidence-grounded data synthesis pipeline covering nine task types across six domains, and optimize the Researcher through verifiedtrajectory supervised fine-tuning followed by Hint-guided Group Relative Policy Optimization. We demonstrate the effectiveness of JAM across a variety of benchmarks on agent memory and long-context processing, where it achieves stronger task performance than AOT-style memory systems while remaining substantially more efficient than prior trained agentic memory approaches. To support reproducibility and future research, we release our anonymized source code at https: //github.com/VectorSpaceLab/general-agentic-memory.

## 1 INTRODUCTION

Recent progress in large language models (LLMs) has accelerated the development of AI agents capable of carrying out increasingly complex workflows, such as deep research, software engineering, and other multi-stage applications Huang et al. (2025); He et al. (2025). Such workflows continuously produce long and evolving histories of observations, reasoning traces, and interactions, making agent memory essential for constructing the context needed by downstream tasks Hu et al. (2025).

Existing agent memory systems often follow an Ahead-of-Time (AOT) design philosophy. They compress or organize raw histories before any concrete request arrives, producing pre-constructed memories such as summaries or memory notes Xu et al. (2025); Kang et al. (2025); Fang et al. (2025). Although this request-agnostic preprocessing reduces online serving cost, it can discard fine-grained details and cross-session dependencies that may later become crucial. Such information loss is particularly problematic when task-relevant evidence is dispersed across large, evolving, and temporally interdependent histories, rather than contained in a single retrievable passage Maharana et al. (2024); Wu et al. (2024). Consequently, fixed retrieval or summarization pipelines struggle to assemble the query-specific context needed for downstream reasoning.

These limitations suggest that agent memory should not be viewed merely as storing past information, but as constructing query-relevant context at runtime. We therefore propose Just-In-Time Agent Memory (JAM), a dual-component framework that follows a Just-In-Time (JIT) design philosophy. Instead of relying on fully pre-compiled memory, JAM separates offline memory organization from runtime memory construction. During the offline stage, a Memorizer preserves complete historical records in a page-store while maintaining lightweight memory as navigational guidance. At runtime, a Researcher conducts runtime agentic research guided by the lightweight memory, retrieving, inspecting, and integrating relevant information to assemble an optimized context for each request. By shifting memory from request-agnostic compression to query-conditioned context construction, JAM reduces information loss and improves adaptability to dynamic information needs.

![](images/2a706cd519275e146ca6b1f60da0f9fee0a0dcb409a4c0c2444ef986c3883d8d.jpg)  
Figure 1: Overview of JAM. (a) The dual-agent JAM framework, where the Memorizer builds a hierarchical workspace and the Researcher performs iterative memory exploration. (b) The two-stage Researcher optimization pipeline, consisting of verified-trajectory SFT and Hint-guided GRPO.

JAM exposes runtime memory construction as an explicit agentic process, making the Researcher’s thinking, tool-use, and sufficiency-assessment behaviors traceable and trainable. Although recent training-based memory agents have explored supervision or reinforcement learning for memory use Yu et al. (2025); Yan et al. (2025); Wang et al. (2025); Shi et al. (2025); Yue et al. (2026), their training data are often derived from conventional multi-hop QA datasets or benchmark-specific splits, offering limited coverage of realistic long-history memory scenarios. To address this data scarcity, we introduce Memory-Gym, a scalable data synthesis pipeline that provides evidencegrounded training scenarios for memory agents. Memory-Gym covers single-hop localization, multi-hop evidence connection, and multi-session summarization across heterogeneous domains, producing queries, answers, and supporting evidence over accumulated histories.

Building on Memory-Gym, we optimize JAM through a cascaded SFT-RL workflow. Supervised fine-tuning (SFT) on verified teacher-generated trajectories first initializes the Researcher’s thinking, tool-use, and sufficiency-assessment behaviors. Hint-guided Group Relative Policy Optimization (GRPO) then further improves multi-step evidence exploration under sparse feedback, enabling JAM to construct more accurate query-relevant contexts at runtime.

Our contributions are summarized as follows: (1) We introduce a Just-In-Time paradigm, which shifts agent memory from static Ahead-of-Time compression to runtime query-conditioned context construction. (2) We propose JAM, a dual-component framework that instantiates this paradigm with a Memorizer for lightweight offline organization and a Researcher for runtime agentic research over complete historical records. (3) We introduce Memory-Gym, a scalable data synthesis pipeline that provides various training scenarios for memory agents, and develop a cascaded optimization workflow that combines verified-trajectory SFT with Hint-guided GRPO to improve JAM’s run time agentic research. (4) Extensive experiments demonstrate that JAM consistently improves task performance over AOT-style memory systems while providing a favorable quality–cost trade-off relative to trained agentic memory approaches.

## 2 JUST-IN-TIME AGENT MEMORY

JAM instantiates JIT memory with two components: a Memorizer that preserves and organizes accumulated histories offline, and a Researcher that constructs query-conditioned context at runtime.

## 2.1 PROBLEM FORMULATION

AI agents often accumulate long histories while performing complex tasks such as software engineering and deep research. We represent such a history as a sequence of temporally ordered sessions $H = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { T } \}$ , where each session may contain observations, reasoning traces, or interactions. As H grows, directly feeding the full history to the agent becomes inefficient and error-prone, motivating a memory system that produces a compact yet sufficient context for each online request.

Given an online request $q$ and an accumulated history $H$ , the memory system produces a context for the client agent: $c ^ { * }  \mathrm { M e m o r y } ( q , H )$ . Ideally, this context should be both useful and compact: it should maximize the client agent’s downstream performance while minimizing context size or serving cost. We formalize this cost-effectiveness objective as:

$$
\mathcal { C } ^ { * } ( q , H ) = \underset { c \in \mathcal { C } ( H ) } { \arg \operatorname* { m a x } } \mathrm { P e r f } _ { \mathcal { A } } ( q , c ) , \qquad c ^ { * } = \underset { c \in \mathcal { C } ^ { * } ( q , H ) } { \arg \operatorname* { m i n } } \ | c | .\tag{1}
$$

Here, $\mathcal { C } ( H )$ denotes the space of contexts constructible from history $H , { \mathcal { C } } ^ { * } ( q , H )$ denotes the set of performance-optimal contexts for request $q , \mathrm { P e r f } _ { \mathcal { A } } ( q , c )$ measures the downstream performance of the client agent A given context c, and $| c |$ denotes context size or serving cost. Under this formulation, AOT-style memory approximates $c ^ { * }$ through request-agnostic pre-compression, whereas JIT memory constructs $c ^ { * }$ at runtime conditioned on the specific request $q .$

## 2.2 MEMORIZER: HIERARCHICAL WORKSPACE CONSTRUCTION

The Memorizer operates during the offline stage, converting accumulated agent histories into a hierarchical workspace that serves as JAM’s persistent page-store. Given a history $H = \{ s _ { 1 } , . . . , s _ { T } \}$ , it preserves raw sessions while constructing compact navigational summaries through two lightweight operations: memorizing and workspace writing. The same procedure supports both batch construction from existing histories and incremental updates when new sessions are appended.

First, memorizing produces a concise memo $\mu _ { i }$ for each session $s _ { i }$ , summarizing its key information and serving as a lightweight descriptor for later organization and navigation:

$$
\operatorname { M e m o r i z e r . m e m o r i z e } ( s _ { i } , { \mathcal { W } } _ { < i } ) \to \mu _ { i } ,\tag{2}
$$

where $\mathcal { W } _ { < i }$ denotes the workspace context before incorporating $s _ { i }$

Second, workspace writing stores the raw content of $s _ { i }$ as a source file and uses its memo $\mu _ { i }$ to organize the file within the hierarchical workspace:

$$
\operatorname { M e m o r i z e r . w r i t e } ( s _ { i } , \mu _ { i } , \mathcal { W } _ { < i } ) \to \mathcal { W } _ { \leq i } .\tag{3}
$$

During this process, semantically related sessions are grouped into coherent directories according to their memos. Each directory maintains a README constructed from the memos of its child files and subfolders, providing a compact navigational summary of the corresponding workspace region.

This design keeps raw files as complete, path-traceable evidence, while memos and READMEs guide the Researcher toward relevant sessions for fine-grained inspection.

## 2.3 RESEARCHER: RUNTIME AGENTIC RESEARCH

During online serving, given a request q and the hierarchical workspace W, the Researcher iteratively explores the workspace, collects useful information, and assesses sufficiency to construct an optimized context for the request.

The Researcher is equipped with three workspace-access tools: open $( p )$ reads the README and directory listing under path p; search(κ) performs hybrid retrieval on candidate paths using BM25 and dense embeddings; and browse $( p , \kappa )$ reads a specific path and extracts query-relevant information. These tools define the action space for runtime exploration.

We formulate the Researcher as an iterative thinking–exploration–reflection process. At step $i ,$ the Researcher observes the request $q$ and the previous interaction trajectory $h _ { i - 1 }$ . It first performs thinking to identify the current information needs and generate a set of memory-access actions:

$$
A _ { i } = \{ a _ { i } ^ { 1 } , \ldots , a _ { i } ^ { K _ { i } } \} \sim \pi _ { \theta } ( \cdot \mid q , h _ { i - 1 } ) , \qquad a _ { i } ^ { k } = ( t _ { i } ^ { k } , \rho _ { i } ^ { k } ) , \quad t _ { i } ^ { k } \in \mathcal { T } ,\tag{4}
$$

![](images/a904eadd377dbda4078a8b15140d89ecb7d907d32ca1711a78e63fd88ff45852.jpg)  
Figure 2: Overview of Memory-Gym. (a) Memory-Gym organizes realistic memory use into three task families covering nine task types across six heterogeneous domains. (b) Memory-Gym constructs evidence-grounded training instances through a four-stage pipeline.

where $\mathcal { T } = \{ \mathsf { o p e n } , \mathsf { s e a r c h } , \mathsf { b r o w s e } \}$ $K _ { i }$ is the number of tool calls generated at step $i ,$ and $\rho _ { i } ^ { k }$ denotes the parameters of the selected tool.

The exploration step executes all tool calls in $A _ { i }$ over the workspace $\mathcal { W }$ and returns a set of observations: $O _ { i } = \{ o _ { i } ^ { 1 } , \ldots , o _ { i } ^ { K _ { i } } \} , \quad o _ { i } ^ { k } = \operatorname { E x e c } ( a _ { i } ^ { k } , \mathcal { W } )$ . The interaction trajectory is then updated by appending the tool calls and their observations: $h _ { i } = h _ { i - 1 } \oplus ( A _ { i } , O _ { i } )$ .

After exploration, the Researcher implicitly assesses information sufficiency over the accumulated trajectory: $y _ { i } = \operatorname { R e f l e c t } _ { \theta } ( q , h _ { i } )$ . It stops when $y _ { i }$ is true or the configured action budget is reached, and continues otherwise. For either stopping condition, the Researcher constructs the final context from the accumulated trajectory: $c ^ { * } = \mathrm { F i n a l i z e } _ { \theta } ( q , h _ { T } )$ , retaining request-relevant information and source provenance for the client agent. If the budget is exhausted without a positive sufficiency assessment, the same finalization step summarizes the evidence collected so far for best-effort answering, without further exploration.

## 3 TRAINING JAM WITH MEMORY-GYM

To train the Researcher, we introduce Memory-Gym for evidence-grounded supervision and a cascaded optimization procedure combining verified-trajectory SFT with Hint-guided GRPO.

## 3.1 MEMORY-GYM: DATA SYNTHESIS FOR MEMORY AGENTS

Effectively training the Researcher requires suitable data that captures diverse memory-use scenarios over accumulated histories, which existing QA-style or benchmark-specific training sources often fail to provide. To address this data scarcity, we introduce Memory-Gym, a scalable data synthesis pipeline designed for JAM’s optimization, as shown in Figure 2.

Given a memory corpus $\mathcal { M } = \{ s _ { 1 } , . . . , s _ { T } \}$ consisting of accumulated sessions and a target task type z, Memory-Gym constructs an instance $\boldsymbol { x } = ( q , a , \mathcal { E } , z )$ . Here, q is a synthesized request, a is the reference answer, and $\mathcal { E } \subseteq \mathcal { M }$ records the supporting sessions used for construction and validation. Below, we introduce the task taxonomy, source domains, and construction pipeline.

## 3.1.1 TASK TAXONOMY AND SOURCE DOMAINS

Memory-Gym organizes realistic memory use over accumulated histories into three task families: single-hop, multi-hop, and multi-session. These families focus on the memory operations required by different requests: locating a specific session-level fact, connecting information across sessions, and aggregating observations over broader histories.

Single-hop localization. Single-hop tasks require the agent to identify a memory session that directly supports the answer. Although the required evidence is local once found, the challenge lies in locating it from a large memory corpus containing irrelevant or semantically similar sessions. This family includes attribute extraction, concept explanation, and condition localization.

Multi-hop evidence connection. Multi-hop tasks require the agent to combine partial information from multiple sessions. The answer cannot be obtained from any single session alone, but must be derived by linking entities, constraints, events, or temporal relations across memory. This family includes comparison, provenance tracing, and intersection selection.

Multi-session summarization. Summarization tasks require the agent to synthesize a structured answer from a broader collection of related sessions. The target answer is often an abstraction over multiple observations, such as temporal changes, repeated mentions, or partially overlapping information. This family includes state evolution, set enumeration, and aggregation summarization.

To improve domain coverage, we instantiate these task families over six heterogeneous domains, including web, books, dialogue, news, academic papers, and long-form documents. Together, these domains expose memory agents to diverse accumulated histories and provide broad training scenarios for learning realistic memory use. Detailed data sources are provided in Appendix A.1.

## 3.1.2 TASK CONSTRUCTION PIPELINE

Memory-Gym constructs each training instance through a four-stage pipeline: anchor discovery, task-dependent session expansion, instance instantiation, and validation. The pipeline first builds the session set required by the target task type, and then generates a query-answer pair.

Stage 1: Anchor discovery. Memory-Gym first selects a task-specific anchor, such as an entity, event, concept, condition, or evolving state, as the semantic starting point of an instance. For each sampled source session, it extracts multiple candidate anchors and filters out those that are underspecified, weakly grounded, or unsuitable for the target task type.

Stage 2: Task-dependent session expansion. Starting from the anchor, Memory-Gym expands a supporting session set E according to the memory-use pattern required by the target task family. For single-hop tasks, it identifies the local session that directly supports the instance and performs corpus-level ambiguity checking. For multi-hop tasks, it retrieves and links related sessions through shared entities, events, constraints, or temporal relations. For summarization tasks, it collects a broader set of sessions associated with the same theme, state, or evolving process, so that the instance requires aggregation rather than isolated fact retrieval.

Stage 3: Instance instantiation. Given the expanded session set E, Memory-Gym generates a query-answer pair (q, a) consistent with the target task type z. The query is constructed to require the target memory-use pattern, while the reference answer is derived only from the selected sessions. The provenance of supporting sessions is also recorded for validation and downstream training.

Stage 4: Validation and filtering. Memory-Gym filters candidates using task-specific quality checks. For single- and multi-hop tasks, we verify answer uniqueness, evidence support, and the absence of answer leakage; for summarization, we additionally assess coverage, completeness, and faithfulness to the source sessions. Of 28,276 candidates, 16,794 passed validation (59.4%). A manual audit of 200 retained instances found that 95.0% met all applicable quality criteria. See Appendices A.3 and A.4 for construction and human-validation details, respectively.

## 3.2 CASCADED OPTIMIZATION OF JAM

JAM exposes runtime agentic research as a sequential and trainable process. We therefore focus on optimizing the Researcher, while keeping the Memorizer fixed to provide a stable hierarchical workspace. As shown in Figure 1, Memory-Gym supports a cascaded optimization workflow: verified-trajectory SFT initializes the Researcher’s tool-use behavior, and Hint-guided GRPO further improves context construction under sparse task-level feedback.

Verified-trajectory SFT. To initialize the Researcher with effective thinking, tool-use, and sufficiency-assessment behaviors, we first construct a supervised corpus of verified Researcher trajectories. Given a Memory-Gym instance and the workspace constructed by the Memorizer, a strong teacher model solves the task under the JAM workflow and produces an interaction trajectory τ.

We retain trajectories whose final answers are correct and whose exploration processes are judged to be high-quality, using an LLM-as-a-judge to filter unsupported or shortcut behaviors such as shallow search without sufficient inspection. The remaining trajectories form a supervised dataset $\mathcal { D } _ { \mathrm { S F T } }$ . We fine-tune the Researcher policy by maximizing the likelihood of demonstrated action sets:

$$
{ \mathcal L } _ { \mathrm { S F T } } ( \theta ) = - \sum _ { \tau \in { \mathcal D } _ { \mathrm { S F T } } } \sum _ { i = 1 } ^ { | \tau | } \log \pi _ { \theta } ( A _ { i } \mid q , h _ { i - 1 } ) .\tag{5}
$$

This stage provides a stable behavioral initialization before policy optimization.

Hint-guided GRPO. SFT provides a useful initialization, but multi-step exploration remains unstable under sparse task-level feedback. We therefore apply Hint-guided GRPO from the SFT checkpoint, injecting lightweight hints during training rollouts to guide search and tool-use decisions.

Specifically, every $K _ { h }$ steps, a hint generator produces a task-specific hint $\eta _ { k }$ conditioned on the current interaction trajectory and the reference answer: $\tilde { s } _ { k } = s _ { k } \oplus \eta _ { k } , \mathrm { i f } k$ mod $K _ { h } = 0$ . The hint provides high-level guidance about missing information, unexplored search directions, or potentially useful tool choices. It does not directly reveal the final answer or prescribe a complete search path. Hints are used only during training rollouts and are removed at inference time.

We use source recall as the primary reward, since this stage aims to improve the Researcher’s ability to locate task-relevant information rather than merely imitate teacher trajectories. For a sampled trajectory $\tau ,$ let ${ \mathcal { E } } _ { \tau }$ denote the set of source sessions collected or cited by the Researcher, and let $\mathcal { E } _ { q } ^ { * }$ denote the gold supporting session set provided by Memory-Gym. The reward is defined as $\begin{array} { r } { \dot { R ( \tau ) } = \mathrm { R e c a l l } ( \mathcal { E } _ { \tau } , \mathcal { E } _ { q } ^ { * } ) = \frac { | \mathcal { E } _ { \tau } \cap \mathcal { E } _ { q } ^ { * } | } { | \mathcal { E } _ { q } ^ { * } | } } \end{array}$ . For each query, we sample a group of G trajectories $\{ \tau _ { j } \} _ { j = 1 } ^ { G }$ and compute their rewards $\{ R ( \tau _ { j } ) \} _ { j = 1 } ^ { G } .$ The advantage $A _ { j }$ is obtained by normalizing rewards within the group. We then optimize the Researcher with the GRPO objective:

$$
J _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \mathrm { m i n } ( \rho _ { j } A _ { j } , \mathrm { c l i p } ( \rho _ { j } , 1 - \epsilon , 1 + \epsilon ) A _ { j } ) - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \right] ,\tag{6}
$$

where $\rho _ { j }$ denotes the policy ratio for $\tau _ { j } , \pi _ { \mathrm { r e f } }$ is the reference policy, and $\beta$ controls the KL penalty.

## 4 EXPERIMENTS

We conduct extensive experiments to evaluate JAM and validate the effectiveness of Memory-Gym as a training source. Specifically, our evaluation aims to answer the following research questions: RQ1: How does JAM perform compared with existing AOT-style memory systems and trained memory agents? RQ2: Does Memory-Gym provide effective and transferable supervision? RQ3: How do the key components of JAM contribute to the overall performance? RQ4: How does JAM benefit from increased test-time computation, and what is its effectiveness–efficiency trade-off?

## 4.1 EXPERIMENT SETTING

Datasets & Metrics. We evaluate JAM on four long-context and memory-intensive benchmarks: LoCoMo Maharana et al. (2024), LongMemEval (LME) Wu et al. (2024), NarrativeQA (NAQA) Kociskˇ y et al. (2018), and HotpotQA Yu et al. (2025). These benchmarks cover long-term\` conversational memory, interactive memory, long-document reasoning, and multi-hop reasoning over dispersed evidence. We use the official evaluation metric for each benchmark: F1 for LoCoMo, NarrativeQA, and HotpotQA, and accuracy for LongMemEval.

Table 1: Results of JAM and baselines on benchmarks. Best results are in bold; second-best results are underlined. Single-hop(SH); Multi-hop(MH); Temporal(TE); Open Domain(OD); Overall(OA). <sup>†</sup>: As Memory-R1 is not publicly released, we faithfully reported the results from the original paper.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Base LLM</td><td colspan="4">LoCoMo</td><td rowspan="2"></td><td rowspan="2">LME NAQA</td><td colspan="3">HotpotQA</td><td rowspan="2">224k</td></tr><tr><td>SH</td><td>MH</td><td>TE</td><td>OD</td><td>OA</td><td>56k</td><td>112k</td></tr><tr><td>Training-free</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VANILLA</td><td>Qwen3.5-4B</td><td>48.15</td><td>32.98</td><td>42.66</td><td>19.63</td><td>42.45</td><td>54.80</td><td>31.23</td><td>63.56</td><td>53.04</td><td>36.97</td></tr><tr><td>RAG</td><td>Qwen3.5-4B</td><td>49.04</td><td>34.39</td><td>44.11</td><td>17.68</td><td>43.37</td><td>56.20</td><td>32.86</td><td>49.97</td><td>44.83</td><td>49.87</td></tr><tr><td>A-MEM</td><td>Qwen3.5-4B</td><td>46.99</td><td>29.88</td><td>42.56</td><td>18.65</td><td>41.17</td><td>56.80</td><td>31.30</td><td>27.08</td><td>25.39</td><td>28.32</td></tr><tr><td>Mem0</td><td>Qwen3.5-4B</td><td>34.84</td><td>26.45</td><td>34.47</td><td>16.83</td><td>32.10</td><td>34.40</td><td>26.72</td><td>30.19</td><td>27.57</td><td>24.53</td></tr><tr><td>MemoryOS</td><td>Qwen3.5-4B</td><td>32.75</td><td>28.95</td><td>36.36</td><td>14.50</td><td>31.67</td><td>36.00</td><td>27.21</td><td>20.27</td><td>22.49</td><td>19.61</td></tr><tr><td>LightMem</td><td>Qwen3.5-4B</td><td>35.35</td><td>27.88</td><td>35.47</td><td>18.68</td><td>32.97</td><td>52.40</td><td>28.24</td><td>30.73</td><td>27.12</td><td>26.51</td></tr><tr><td>Trained</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MEM1</td><td>MEM1-7B</td><td>28.56</td><td>20.48</td><td>32.46</td><td>14.73</td><td>27.11</td><td>32.40</td><td>23.86</td><td>33.94</td><td>32.47</td><td>27.32</td></tr><tr><td>MemAgent</td><td>MemAgent-14B</td><td>49.19</td><td>36.33</td><td>53.77</td><td>22.50</td><td>46.13</td><td>55.20</td><td>28.24</td><td>56.26</td><td>50.70</td><td>46.88</td></tr><tr><td>Memory-R1-PPO†</td><td>Mem-R1-8B†</td><td>32.52</td><td>26.86</td><td>41.57</td><td>45.30</td><td>41.05</td><td></td><td></td><td>一</td><td></td><td></td></tr><tr><td>Memory-R1-GRPO†</td><td>Mem-R1-8B†</td><td>35.73</td><td>35.65</td><td>49.86</td><td>47.42</td><td>45.02</td><td></td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>Our Method JAM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>with training</td><td>Qwen3.5-4B</td><td></td><td>55.14 40.21 62.59 25.12 52.09 65.20</td><td></td><td></td><td></td><td></td><td>45.72</td><td></td><td>64.56 59.28 58.69</td><td></td></tr></table>

Baselines. We compare JAM against both training-free and training-based baselines. Trainingfree baselines include Vanilla LLM and RAG Jiang et al. (2023), as well as AOT-style memory systems A-MEM Xu et al. (2025), Mem0 Chhikara et al. (2025), MemoryOS Kang et al. (2025), and LightMem Fang et al. (2025). Training-based baselines include MEM1 Zhou et al. (2026), MemAgent Yu et al. (2025), and Memory-R1 Yan et al. (2025).

Implementation Details. We use Qwen3.5-4B Qwen Team (2026) as the backbone model for JAM and all AOT-style baselines. For training-based baselines, we use their released checkpoints or official reported settings. JAM uses BM25 and BGE-M3 Chen et al. (2024) as its default lexical and dense retrievers. Further implementation and training details are provided in Appendix B.1.

## 4.2 MAIN RESULTS

Table 1 reports the performance of JAM and ten competitive baselines across four benchmarks, demonstrating the overall effectiveness of our framework for memory-intensive tasks.

First, JAM exhibits strong generalization across heterogeneous settings. While several AOT-style memory systems remain competitive on dialogue-centric benchmarks such as LoCoMo, their performance often degrades on long-document and multi-hop reasoning benchmarks. In contrast, JAM remains consistently strong across different task types, suggesting that its hierarchical workspace and runtime agentic research enable more robust memory use across domains.

Second, JAM performs strongly even against substantially larger trained memory agents. Despite using a Qwen3.5-4B backbone, JAM outperforms 7B–14B trained baselines across most settings. The results highlight the importance of effective training supervision: Memory-Gym exposes the Researcher to diverse memory-use scenarios, enabling it to learn exploration behaviors that generalize across heterogeneous benchmarks. We examine the contribution of Memory-Gym supervision more directly in Section 4.3.

Third, JAM remains stable as context length increases. On HotpotQA with increasing context sizes, prior methods often degrade when relevant information is buried among longer distractor contexts, while JAM sustains comparatively strong performance. This suggests that JAM can better handle dispersed information because its offline workspace organization and trained Researcher support progressive, query-conditioned exploration at runtime.

Complementary analyses further show that these gains are stable across inference seeds and persist under an LLM-judge evaluation (Appendices B.5 and B.6).

## 4.3 EFFECTIVENESS AND TRANSFERABILITY OF MEMORY-GYM

To evaluate Memory-Gym, we first compare different supervision sources and task families under a fixed JAM framework and backbone, with all trained variants using SFT only. We then examine whether the learned memory-use behaviors transfer beyond Memory-Gym’s source domains on LongCodeQA Rando et al. (2025).

Supervision sources and task diversity. Table 2 shows that alternative supervision sources yield uneven gains across benchmarks. MemAgent-derived supervision performs best on HotpotQA, whereas both MemAgent- and Memory-R1-derived super-

Table 2: Memory-Gym supervision analysis. All trained variants use SFT only.
<table><tr><td>Training data</td><td>LoCoMo LME NAQA HotpotQA</td></tr><tr><td>w/o SFT</td><td>48.59 48.80 34.98 45.44</td></tr><tr><td>MemAgent data</td><td>48.60 55.20 31.98 54.53</td></tr><tr><td>Memory-R1 data</td><td>48.9057.20 32.42 49.81</td></tr><tr><td>SH only</td><td>50.59 58.80 34.40 50.45 51.96</td></tr><tr><td>MH only SU only</td><td>48.9757.60 37.42 49.43 59.80 40.94</td></tr><tr><td>Memory-Gym</td><td>50.86 50.70 60.20 41.59 52.08</td></tr></table>

vision underperform the untrained Researcher on NarrativeQA. In contrast, Memory-Gym improves over the untrained Researcher on all four benchmarks and achieves the strongest source-comparison results on three. The task-family analysis shows a complementary pattern: among single-family variants, SH performs best on LoCoMo, MH on HotpotQA, and SU on LongMemEval and NarrativeQA, while the full mixture outperforms every single-family variant across all four benchmarks. These results are consistent with complementary benefits from evidence localization, cross-session connection, and broader aggregation, motivating the use of diverse memory-use tasks for supervision.

Cross-domain transfer. We further evaluate the Researcher on LongCodeQA, a code-comprehension benchmark from a domain not represented in Memory-Gym, without using any LongCodeQA examples for training or adaptation. As shown in Table 3, overall accuracy increases from 54.08% for the untrained Researcher to 65.67% after SFT and 75.54% after the full SFT+RL pipeline. Improvements are consistent across the 64K, 128K, and 256K subsets. These results indicate that the memory-use behaviors learned from Memory-Gym transfer beyond its source domains, with policy optimization providing further gains.

Table 3: Cross-domain transfer on LongCodeQA.
<table><tr><td>Researcher</td><td>64K</td><td>128K</td><td>256K Overall</td></tr><tr><td>Untrained</td><td>56.58</td><td>55.43 49.23</td><td>54.08</td></tr><tr><td>SFT only</td><td>68.42</td><td>64.13 64.62</td><td>65.67</td></tr><tr><td>SFT+RL</td><td>76.32</td><td>76.09 73.85</td><td>75.54</td></tr></table>

## 4.4 ABLATION STUDY

To evaluate the contribution of each core component, we conduct systematic ablation studies by removing individual modules. The results are summarized in Table 4. First, Researcher optimization consistently improves performance. The untrained variant already benefits from the JAM framework, but remains limited on more challenging memoryintensive tasks. SFT improves the Researcher by providing high-quality trajectory supervision for thinking, tool use, and workspace exploration. GRPO without hints brings further gains over SFT-only, showing that reinforce-

Table 4: Ablation results on different benchmarks. LoCoMo denotes the overall score. HotpotQA denotes the average over 56k, 112k, and 224k settings.
<table><tr><td>Method</td><td>LoCoMo LME NAQA</td><td>HotpotQA</td></tr><tr><td>w/o Training</td><td>48.59 48.80</td><td>34.98 45.44</td></tr><tr><td>SFT Only</td><td>50.70 60.20</td><td>41.59 52.08</td></tr><tr><td>GRPO w/o Hint</td><td>51.42 62.40</td><td>43.62 55.64</td></tr><tr><td>w/o Memorizer</td><td>49.06 64.00</td><td>38.87 57.72</td></tr><tr><td>w/o Researcher</td><td>43.30 58.40</td><td>32.51 39.10</td></tr><tr><td>w/o BM25</td><td>45.90 62.40</td><td>38.88 52.14</td></tr><tr><td>w/o Embedding</td><td>50.86 61.80</td><td>39.45 58.13</td></tr><tr><td>Full</td><td>52.09 65.20</td><td>45.72 60.84</td></tr></table>

ment learning helps the Researcher improve source discovery beyond imitation. Adding hint guidance achieves the best performance, especially on NAQA and HotpotQA, indicating that lightweight hints stabilize multi-step exploration under sparse rewards. Second, both the Memorizer and the

![](images/5d3f8586ab63b99002cc569673e078017514480734a360d0e7d3b184d35d7f62.jpg)  
(a) Impact of maximum action rounds

![](images/0f8365bd6f1bb6c77f0f6cc623cc6660f90498e41536435251576743d08b7e80.jpg)  
(b) Impact of the amount of retrieved items.

![](images/9bff069dc17c78a71b9c687b9f5afb47921c5722f2242f5b87acfdf726f4f0a1.jpg)  
Figure 3: Test-time scaling with increasing action rounds Figure 4: Wall-clock latency and (left) and retrieved items (right). overall F1 on LoCoMo.

Researcher contribute substantially to JAM. Removing the Memorizer weakens performance by eliminating the hierarchical organization and navigational summaries of the workspace. Removing the Researcher causes an even larger degradation, highlighting the importance of iterative, queryconditioned exploration at runtime. A controlled comparison against a flat raw-history store further shows that hierarchical workspace organization improves both effectiveness and online efficiency: it improves LoCoMo F1 from 49.06 to 52.09 while reducing online latency from 17.11 to 13.81 seconds per query and average research rounds from 3.57 to 3.16 (Appendix B.3). We further examine the sensitivity to the Memorizer backbone in Appendix B.7, where scaling to larger backbone models yields no consistent downstream improvement. Finally, removing either BM25 or dense retrieval also degrades performance, indicating that lexical and semantic retrieval are complementary and jointly provide more informative candidates for downstream exploration.

## 4.5 TEST-TIME SCALABILITY AND EFFICIENCY

Scaling test-time computation. Figure 3 examines two forms of inference-time budget: the maximum number of action rounds and the number of retrieved items per search. Increasing the action budget from 5 to 20 consistently improves performance, providing the Researcher with more opportunities to refine its search directions and inspect candidate workspace regions, with diminishing gains at larger budgets. Increasing the number of retrieved items from 1 to 5 shows a similar trend by broadening the candidate set available at each search step.

Importantly, the maximum action budget provides exploration headroom rather than imposing a fixed inference cost. Under the default 20-round limit, the median number of rounds is only 3 on LoCoMo, LongMemEval, and NarrativeQA and 6 on HotpotQA, while budget exhaustion remains below 5% across all benchmarks. Thus, JAM varies its computation across queries and terminates well before the maximum budget in most cases. Detailed runtime statistics are in Appendix B.4.

Runtime efficiency. Figure 4 reports wall-clock latency on LoCoMo, separating one-time offline workspace construction from online query serving. JAM requires 79.53 s to construct a workspace, lower than A-MEM (200.42 s) and Mem0 (90.67 s), but higher than MemoryOS (41.36 s) and Light-Mem (6.82 s). This cost is incurred once, after which the resulting workspace is reused across queries. The iterative Researcher introduces additional query-time latency relative to one-shot memory systems. However, this additional computation is accompanied by substantially higher answer quality: JAM achieves an overall LoCoMo F1 of 52.09, outperforming all AOT-style baselines despite their lower serving latency. Compared with the trained agentic baseline MemAgent, JAM reduces query latency from 58.08 s to 13.81 s (76.2%) while improving F1 from 46.13 to 52.09. Together, these results indicate a favorable quality–latency trade-off: JAM spends more computation than lightweight one-shot memory systems to obtain markedly higher answer quality, while remaining substantially faster than MemAgent. Complementary token-consumption results show a similar quality–cost pattern, with JAM achieving the highest F1 while using substantially fewer tokens per task than MemAgent (Appendix B.3).

## 5 RELATED WORK

## 5.1 AOT-STYLE AGENT MEMORY SYSTEMS

Recent studies on agent memory systems mainly focus on preserving and organizing accumulated contexts for AI agents Packer et al. (2023). A large body of work follows an AOT design philosophy, where historical observations are compressed or structured into pre-constructed memories before specific online requests arrive. Representative methods organize observations into linked memory notes, hierarchical memory stores, or lightweight summarization and consolidation pipelines Xu et al. (2025); Kang et al. (2025); Fang et al. (2025). Nevertheless, their request-agnostic preprocessing can discard fine-grained details or cross-session dependencies that later become important. Moreover, stored memories are often utilized through relatively simple retrieval or summarization procedures, which are less effective for complex memory-intensive tasks that require iterative in spection and query-conditioned context construction.

## 5.2 TRAINING AND DATA FOR MEMORY AGENTS

Recent work has trained memory agents with SFT or RL to improve memory use in long-context scenarios Wang et al. (2025); Shi et al. (2025). These studies demonstrate the promise of optimizationbased memory agents, but their supervision is often tied to constrained multi-hop QA datasets such as HotpotQA Yu et al. (2025) or benchmark-specific splits Yu et al. (2025); Yan et al. (2025); Yue et al. (2026). This limits the diversity of memory-use scenarios and makes it difficult to train agents for realistic settings that require localization, cross-session connection, temporal tracking, and aggregation over accumulated long contexts.

## 6 CONCLUSION

In this paper, we propose JAM, a Just-In-Time agent memory framework that shifts memory from request-agnostic pre-construction to runtime query-conditioned context construction. JAM combines a Memorizer that preserves and organizes accumulated histories in a hierarchical workspace with a Researcher that iteratively explores the workspace to construct task-relevant context. To optimize this process, we introduce Memory-Gym and a cascaded training workflow combining verifiedtrajectory SFT with Hint-guided GRPO. Experiments across diverse long-context and memoryintensive benchmarks show that JAM consistently improves task performance over AOT-style memory systems, transfers across domains, and offers a favorable quality–cost trade-off for runtime memory construction.

## REFERENCES

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 4(5), 2024.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4599–4610, 2021.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3029–3051, 2023.

Alexander Richard Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Pro-

ceedings of the 57th annual meeting of the association for computational linguistics, pp. 1074– 1084, 2019.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, et al. Lightmem: Lightweight and efficient memoryaugmented generation. arXiv preprint arXiv:2510.18866, 2025.

Wikimedia Foundation. Wikimedia downloads. https://dumps.wikimedia.org, 2023. English Wikipedia dump, version 20231101.en, accessed via the Hugging Face wikimedia/wikipedia dataset.

Google DeepMind. Gemini 3 Flash: Model card. Model card, December 2025. URL https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-Flash-Model-Card.pdf.

Junda He, Christoph Treude, and David Lo. Llm-based multi-agent systems for software engineering: Literature review, vision, and the road ahead. ACM Transactions on Software Engineering and Methodology, 34(5):1–30, 2025.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, et al. Memory in the age of ai agents. arXiv preprint arXiv:2512.13564, 2025.

Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. In Proceedings ofthe 2021 conference ofthe north American chapter of the association for computational linguistics: Human language technologies, pp. 1419–1436, 2021.

Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, et al. Deep research agents: A systematic examination and roadmap. arXiv preprint arXiv:2506.18096, 2025.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, pp. 7969–7992, 2023.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 25972–25981, 2025.

Toma´s Koˇ ciskˇ y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G\` abor Melis,´ and Edward Grefenstette. The narrativeqa reading comprehension challenge. Transactions ofthe Associationfor Computational Linguistics, 6:317–328, 2018.

Wojciech Krysci´ nski, Nazneen Rajani, Divyansh Agarwal, Caiming Xiong, and Dragomir Radev.´ Booksum: A collection of datasets for long-form narrative summarization. In Findings of the associationfor computational linguistics: EMNLP 2022, pp. 6536–6558, 2022.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, 2024.

MiniMax. Minimax m2.5: Built for real-world productivity. https://www.minimax.io/ news/minimax-m25, 2026. Accessed: 2026-04-28.

Charles Packer, Vivian Fang, Shishir G Patil, Kevin Lin, Sarah Wooders, and Joseph E Gonzalez. Memgpt: towards llms as operating systems. 2023.

Guilherme Penedo, Hynek Kydl´ıcek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandroˇ Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Stefano Rando, Luca Romani, Alessio Sampieri, Luca Franco, John Yang, Yuta Kyuragi, Fabio Galasso, and Tatsunori Hashimoto. Longcodebench: Evaluating coding llms at 1m context windows. arXiv preprint arXiv:2505.07897, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, and An Zhang. Look back to reason forward: Revisitable memory for long-context llm agents. arXiv preprint arXiv:2509.23040, 2025.

Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng, Guangsheng Bao, Xiangkun Hu, Zheng Zhang, Qian Wang, and Yue Zhang. Novelqa: Benchmarking question answering on documents exceeding 200k tokens, 2024. URL https://arxiv.org/abs/ 2403.12766.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-{\alpha}: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911, 2025.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2024.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z Pan, et al. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025.

Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.

Yanwei Yue, Boci Peng, Xuanbo Fan, Jiaxin Guo, Qiankun Li, and Yan Zhang. Mem-t: Densifying rewards for long-horizon memory agents. arXiv preprint arXiv:2601.23014, 2026.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low, and Paul Liang. Mem1: Learning to synergize memory and reasoning for efficient longhorizon agents. In International Conference on Learning Representations, volume 2026, pp. 58413–58438, 2026.

## A DETAILS OF MEMORY-GYM

## A.1 DATA SOURCES AND CORPUS CONSTRUCTION

Memory-Gym is built from six heterogeneous domains: web, books, dialogue, news, academic papers, and long-form documents. These domains are selected to cover diverse forms of accumulated long-context memory, including factual lookup, narrative reasoning, conversational memory, event evolution, scholarly evidence tracing, and structured document understanding. We normalize all raw inputs into memory corpora $\mathcal { M } = \{ s _ { 1 } , . . . , s _ { T } \}$ , where each $s _ { i }$ denotes a memory session. The following paragraphs describe the domain-specific data sources and session construction procedures.

Web. The web domain is constructed from Wikipedia Foundation (2023) and FineWeb Penedo et al. (2024), covering both structured encyclopedic knowledge and diverse open-domain web content. FineWeb provides cleaned general web documents with broad topics and heterogeneous writing styles. We retain FineWeb documents whose token count falls within [30,000, 150,000] and process each retained document with the JAM Memorizer to obtain memory sessions. Wikipedia provides relatively clean, structured, and entity-centric encyclopedic articles. Since individual articles are usually short, we construct long-context memory corpora through random composition: each session is formed by concatenating 5–8 randomly sampled articles, and each memory corpus contains 30–50 such sessions. This creates web memory corpora with many factual units and distractor articles for evidence localization across sessions.

Books. The books domain is constructed from BookSum Krysci´ nski et al. (2022) and Nov-´ elQA Wang et al. (2024), both of which provide long-form narrative texts involving extended plots, characters, events, and state changes. BookSumBooks contains book-length literary texts, while NovelQA consists of full-length novels designed for long-context narrative understanding. For both sources, we filter candidate books by length and retain full texts whose token count falls within [20,000, 500,000]. Each retained book is treated as a long-form raw input and processed by the JAM Memorizer, which segments it into memory sessions and organizes them into the hierarchical workspace. This preserves the original narrative order while converting each book into a sessionbased memory corpus.

Dialogue. The dialogue domain is constructed from UltraChat Ding et al. (2023) and LoCoMo Ma harana et al. (2024), which provide multi-turn conversational data for simulating accumulated conversational contexts. UltraChat is a large-scale open-domain dialogue dataset covering diverse user intents, topics, and conversational styles. For UltraChat, we retain conversations with at least 5 turns and treat each retained conversation as one memory session. For each memory corpus, we randomly sample 25–50 such sessions, resulting in an accumulated dialogue memory corpus composed of multiple conversational episodes. LoCoMo is a long-term conversational memory benchmark designed to evaluate reasoning over information accumulated across extended interactions. We use the official scripts to generate dialogue data. Each generated session contains more than 15 dialogue turns, and each memory corpus contains more than 30 sessions. Compared with UltraChat, LoCoMo provides longer and more memory-intensive conversational contexts for cross-session evidence localization and temporal tracking.

News. The news domain is constructed from MultiNews Fabbri et al. (2019), which contains articles about diverse real-world events, topics, and social issues. Since related news articles often describe evolving events from different perspectives, this domain supports tasks involving event tracking, provenance tracing, comparison, and aggregation. We retain articles with at least 1,000 tokens, encode them using BGE-M3, and cluster them based on cosine similarity. For each memory corpus, we randomly sample 30–35 articles from a target cluster, with each article treated as one memory session. This yields long-context news memory corpora with distributed evidence and realistic distractor reports.

Academic. The academic domain is constructed from Qasper Dasigi et al. (2021), a questionanswering dataset grounded in academic papers. Since academic papers contain structured information about research problems, methods, experiments, and results, this domain supports tasks such as concept explanation, method comparison, and evidence tracing. We use the original section structure of each paper for corpus construction, treating each section as one memory session. Papers with fewer than 10 sections are filtered out to ensure sufficient cross-session structure for long-context reasoning.

Document. The document domain is constructed from Gov-report Huang et al. (2021), which contains long-form government reports with hierarchical section structures. These reports describe policies, programs, institutional plans, outcomes, and conditions, making them suitable for condition localization, state tracking, set enumeration, and aggregation tasks. For corpus construction, we preserve the original document hierarchy by recursively traversing the report tree. Each node is treated as one memory session, and its child nodes are further expanded into additional sessions. We retain reports with at least 28 recursively expanded sessions to ensure sufficient depth and breadth for evidence localization and reasoning over structured long documents.

Evaluation contamination control. To avoid evaluation contamination, we ensure that all memory corpora used for Memory-Gym construction are disjoint from the evaluation data used in our experiments. We exclude all released evaluation instances, including their questions, answers, supporting evidence, and source sessions, from Memory-Gym construction. In particular, although LoCoMo is included as a dialogue-domain source, we do not use any released LoCoMo evaluation conversations, questions, answers, or supporting evidence. Instead, we generate new LoCoMo-style dialogue corpora using the official data-generation pipeline, with independently sampled scenarios and sessions. Therefore, the Memory-Gym training corpora have no overlap with the evaluation benchmarks in terms of users, conversations, sessions, queries, answers, or supporting evidence.

## A.2 IMPLEMENTATION DETAILS OF THE CONSTRUCTION PIPELINE

We use MiniMax-2.5 MiniMax (2026) as the LLM-based generator for anchor extraction and evidence-grounded question-answer instantiation. Given a memory corpus and a target task type, the generator identifies candidate anchors from the available sessions and constructs query-answer pairs grounded in the selected evidence. We also use MiniMax-2.5 as the judge model to validate generated instances according to task-specific criteria. We use separate prompts for generation and validation to reduce the chance that the same generation procedure accepts invalid instances. For retrieval and clustering, we use BGE-M3 Chen et al. (2024) as the embedding model. Candidate documents or sessions are encoded into dense vectors, and cosine similarity is used to measure semantic relatedness. The resulting similarity scores are used for clustering related documents during corpus construction and for retrieving candidate evidence sessions during task construction.

## A.3 TASK-SPECIFIC CONSTRUCTION AND VALIDATION

Although Memory-Gym follows a unified construction pipeline, different task types require different evidence structures and validation criteria. We therefore define task-specific rules to ensure that each generated query reflects the intended memory-use pattern, including single-session localization, cross-session reasoning, and multi-session aggregation. Across all tasks, valid instances must be evidence-supported, leakage-free, and consistent with the target task type.

Single-hop tasks. Single-hop tasks require the answer to be supported by a single memory session, while the challenge is to locate the correct session from a large memory corpus. We construct three single-hop task types: attribute extraction, concept explanation, and condition localization.

Attribute Extraction. For attribute extraction, we randomly sample a memory session and ask the generator to identify salient attributes of entities, objects, events, or concepts in the session. We then generate a query about one selected attribute, with the attribute value used as the reference answer. For validation, we retrieve passages using the generated query and discard instances if alternative plausible answers are found. The judge model further checks whether the query-answer pair is reasonable, evidence-supported, and consistent with the attribute extraction task.

Concept Explanation. For concept explanation, we randomly sample a session and ask the generator to identify key concepts or objects that have explicit definitions or property descriptions. The query asks for the definition or properties of the selected concept, and the answer is derived from the corresponding description. During validation, we retrieve relevant passages to check whether other sessions contain competing definitions or descriptions, and further verify that the concept is uniquely grounded in the target session. The judge model then checks evidence support and task consistency.

Condition Localization. For condition localization, we randomly sample a session and ask the generator to identify causal, conditional, or prerequisite relations. We extract the resulting event or phenomenon as the query target and generate a question asking what condition or cause leads to it. The answer is the corresponding condition or cause in the evidence session. For validation, the judge model checks whether the answer has a valid causal or prerequisite relation with the queried outcome, and we discard instances if other sessions provide conflicting causes or conditions.

Multi-hop tasks. Multi-hop tasks require the answer to be derived by connecting evidence from multiple memory sessions. We construct three types of multi-hop tasks: comparison, provenance tracing, and intersection selection.

Comparison. For comparison, we randomly sample several sessions and ask the generator to identify a feature dimension suitable for entity comparison, such as location, scale, membership, or outcome. We then use this feature as a query to retrieve relevant sessions and ask the generator to identify two to three entities with corresponding evidence. A comparison question is generated based on the selected entities and feature dimension. For validation, the judge checks whether the answer is supported by the retrieved evidence, correctly addresses the comparison, and remains unique after searching each compared entity in the memory corpus.

Provenance Tracing. For provenance tracing, we construct cross-session entity-linking tasks. We first select a target entity that appears in at least two different sessions. One session provides a unique abstract description of the entity without mentioning its name, while another session provides an event or fact associated with the entity. The question is generated using the abstract description and event details, but the entity name is strictly removed. For validation, we apply name-leakage checks, verify that the abstract description uniquely identifies the target entity, and ask the judge to confirm that the full reasoning chain from identity resolution to event grounding is valid.

Intersection Selection. For intersection selection, we construct questions whose target can only be identified by combining multiple weak constraints. Starting from a target entity, we retrieve several sessions containing it and ask the generator to extract broad descriptions from each session. Each individual description should be ambiguous, while their intersection should uniquely identify the target. The generated question combines these constraints and asks about a specific fact or event without revealing the entity name. The judge validates individual ambiguity, intersection-level uniqueness, and answer correctness.

Summarization tasks. Summarization tasks require the agent to aggregate information over multiple memory sessions rather than locating a single evidence unit. We construct three types of summarization tasks: state evolution, set enumeration, and aggregation summarization.

State Evolution. For state evolution, we first sample five consecutive sessions and ask the generator to identify entities whose states change over time. After selecting a target entity, we scan the full memory corpus in chronological order by batches of sessions and ask the generator to identify all sessions that describe the entity’s state changes. The collected sessions are then used to generate a question about the evolution of the target entity, with the answer summarizing the complete state transition process. For validation, the judge model checks whether the question is reasonable and whether the answer faithfully reflects the state changes supported by the selected sessions.

Set Enumeration. For set enumeration, we first cluster sessions using embeddings and FAISS to identify semantically related contexts. From a sampled cluster, the generator identifies an extraction criterion defined by a category and a constraint, such as entities of a certain type mentioned under a specific condition. We then scan the full memory corpus to incrementally collect all unique entities satisfying the criterion, together with their source sessions. The final query is generated from the extraction criterion, and the answer consists of the collected entity set. For validation, the judge model checks both correctness and completeness, ensuring that all listed items are supported by evidence and that no supported items are omitted.

Aggregation Summarization. For aggregation summarization, we cluster sessions based on embedding similarity and ask the generator to infer a coherent topic from the top sessions in a cluster. We then use this topic to scan the full memory corpus in batches and collect all sessions relevant to the topic. Given the collected evidence, the generator constructs a question asking for an aggregated summary and produces a reference answer. During validation, the judge model checks whether the instance matches the aggregation summarization task, whether the question and answer are aligned, and whether the answer is faithful to the relevant sessions.

Table 5: Human–LLM agreement on 200 pre-filter candidates. The adjudicated human labels serve as the reference. Acceptance is the positive class for precision, recall, and F1. All values except Cohen’s κ are percentages.
<table><tr><td colspan="7"></td></tr><tr><td>Evaluator</td><td>Accept rate</td><td>Agreement</td><td>κ</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Human reference</td><td>62.0</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MiniMax-2.5</td><td>58.5</td><td>91.5</td><td>0.823</td><td>95.7</td><td>90.3</td><td>92.9</td></tr><tr><td>GPT-5.5</td><td>60.0</td><td>92.0</td><td>0.832</td><td>95.0</td><td>91.9</td><td>93.4</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>57.5</td><td>90.5</td><td>0.803</td><td>95.7</td><td>88.7</td><td>92.1</td></tr></table>

## A.4 HUMAN VALIDATION OF MEMORY-GYM

We assess the quality of Memory-Gym instances and the reliability of automatic validation through three complementary audits: human–LLM agreement on candidates before filtering, a human quality audit of retained instances, and an error analysis of rejected candidates.

Annotation protocol. Across all three audits, two primary annotators independently apply predefined task-specific criteria while blinded to the automatic decisions and each other’s labels. A third annotator adjudicates disagreements. Inter-annotator agreement and Cohen’s κ are computed from the independent labels before adjudication, whereas quality estimates and human–LLM comparisons use the adjudicated labels. The quality criteria follow the task-specific validation requirements described in Appendix A.3. All instances are assessed for evidence support, absence of answer leakage in the query, and task consistency. Single-hop and multi-hop instances are additionally assessed for answer uniqueness, while summarization instances are assessed for coverage, completeness, and faithfulness to the collected sessions. An instance passes the overall quality assessment only if it satisfies all criteria applicable to its task type.

Human–LLM agreement before filtering. We sample 200 candidates from the pre-filter pool, stratified across all nine task types and six source domains. The two primary annotators achieve 96.0% agreement and Cohen’s κ = 0.916. After adjudication, 124 candidates are accepted, corresponding to a human acceptance rate of 62.0%. We evaluate the same candidates with MiniMax-2.5, GPT-5.5, and Qwen3.5-122B-A10B using identical evidence, validation criteria, prompts, and output formats across the three models. MiniMax-2.5 is the judge used in the construction pipeline; the other two models provide additional comparisons against the human reference. Precision, recall, and F1 treat acceptance as the positive class. As shown in Table 5, all three judges achieve over 90% agreement and κ > 0.80 with the adjudicated human labels. Their acceptance rates are slightly below the human reference, suggesting a moderately conservative tendency on this sample.

Human audit of retained instances. We sample 200 instances from the 16,794 retained Memory-Gym instances, including 134 single-hop or multi-hop instances and 66 summarization instances. Table 6 reports criterion-level quality and inter-annotator agreement. Overall, 190 of the 200 instances satisfy all applicable criteria, yielding a 95.0% pass rate with a 95% Wilson confidence interval of 91.0%–97.3%. Criterion-level pass rates range from 95.5% to 99.5%. These results support the quality of the retained instances while identifying a small proportion with residual quality issues.

Error analysis of rejected candidates. We sample 100 rejected candidates across task types and domains. Annotators independently assess candidate validity and assign one primary error category to invalid instances. Agreement on the binary validity decision is 98.0%, with Cohen’s κ = 0.789. Among the 94 candidates judged invalid by both primary annotators before adjudication, agreement on the primary error category is 90.4%, with multiclass κ = 0.875. After adjudication, 96 candidates are confirmed as correctly rejected, while four are judged valid. Table 7 summarizes the adjudicated outcomes. Evidence-grounding failures and ambiguity are the most frequent reasons for justified rejection. The 4.0% figure refers specifically to valid instances within the sampled rejected pool, rather than to the proportion of all valid candidates rejected by the pipeline.

Table 6: Human audit of 200 retained Memory-Gym instances. Applicable denotes the number of instances assessed under each criterion. Passed counts and pass rates use adjudicated labels; agreement and Cohen’s κ use the two primary annotators’ labels before adjudication.
<table><tr><td colspan="3"></td><td rowspan="2">Pass rate (%)</td><td rowspan="2">Agreement (%)</td></tr><tr><td>Quality criterion</td><td>Applicable</td><td>Passed</td></tr><tr><td>Evidence support</td><td>200</td><td>196</td><td>98.0</td><td>99.5 0.886</td></tr><tr><td>Answer uniqueness</td><td>134</td><td>131</td><td>97.8</td><td>99.3 0.853</td></tr><tr><td>Leakage-free</td><td>200</td><td>199</td><td>99.5</td><td>100.0 1.000</td></tr><tr><td>Task consistency</td><td>200</td><td>197</td><td>98.5</td><td>99.5 0.855</td></tr><tr><td>Coverage</td><td>66</td><td>64</td><td>97.0</td><td>98.5 0.792</td></tr><tr><td>Completeness</td><td>66</td><td>63</td><td>95.5</td><td>98.5 0.849</td></tr><tr><td>Faithfulness</td><td>66</td><td>65</td><td>98.5</td><td>100.0 1.000</td></tr><tr><td>Pass all applicable criteria</td><td>200</td><td>190</td><td>95.0</td><td>97.5 0.787</td></tr></table>

Table 7: Human error analysis of 100 rejected Memory-Gym candidates. Each invalid candidate is assigned one primary error category. Counts use adjudicated labels, and percentages are calculated over all 100 sampled rejected candidates.
<table><tr><td>Outcome / primary error type</td><td>Count</td><td>Percentage (%)</td></tr><tr><td>Evidence-grounding failure</td><td>33</td><td>33.0</td></tr><tr><td>Ambiguity/non-uniqueness</td><td>22</td><td>22.0</td></tr><tr><td>Query leakage or construction issue</td><td>16</td><td>16.0</td></tr><tr><td>Task-consistency failure</td><td>14</td><td>14.0</td></tr><tr><td>Coverage/completeness failure</td><td>11</td><td>11.0</td></tr><tr><td>Correctly rejected (subtotal)</td><td>96</td><td>96.0</td></tr><tr><td>Incorrectly rejected (valid candidates)</td><td>4</td><td>4.0</td></tr><tr><td>Total</td><td>100</td><td>100.0</td></tr></table>

Together, these audits provide human evidence for the quality of retained Memory-Gym instances and the reliability of automatic filtering on the evaluated samples.

## B EXPERIMENT DETAILS

## B.1 IMPLEMENTATION DETAILS

All experiments were conducted on computational nodes equipped with 8 NVIDIA H100 80GB GPUs. For SFT, we optimize the Researcher on verified trajectories with the standard next-token prediction objective. We use Qwen3.5-122B-A10B Qwen Team (2026) to generate interaction trajectories and Gemini 3 Flash Google DeepMind (2025) to filter low-quality or unsupported trajectories. A typical SFT run takes approximately 5 hours under this hardware configuration.

Unless otherwise specified, the Researcher performs at most 20 action rounds for each query. For each search action, we retrieve the top-5 candidates from BM25 and the top-5 candidates from BGE M3, and then merge and deduplicate them. The browse model and the final answer generation model are both instantiated with the untrained Qwen3.5-4B backbone, and are not updated during SFT or RL.

For RL, we optimize the Researcher policy using Hint-guided GRPO, initialized from the SFT checkpoint and guided by the source-recall reward. To improve training efficiency, we use the Verl library Sheng et al. (2025) with vLLM-based rollouts, gradient checkpointing, and FSDP offloading. We sample $G = 8$ trajectories per query for group-relative advantage estimation and train the policy for 200 optimization steps. The policy learning rate is set to 1e-6, the KL loss coefficient β is fixed at 0.001, and the clipping ratio ϵ is set to 0.2. Hints are used only during training rollouts and are removed at test time. A typical GRPO run takes approximately 6 hours under the same hardware configuration.

![](images/4be0532954a279c6cf68fe71d6bc16a08d3625fb334b4ee8540f2988ce3db519.jpg)

Figure 5: Efficiency comparison on LoCoMo in terms of average token consumption per task and overall F1 score.
<table><tr><td>Setting</td><td>Offline Tokens (/workspace)</td><td>Offline Time (s/workspace)</td><td>Online Tokens (/query)</td><td>Online Time (s/query)</td><td>Avg. Rounds</td><td>F1</td></tr><tr><td>Flat Raw Store</td><td>0</td><td>0</td><td>8174.29</td><td>17.11</td><td>3.57</td><td>49.06</td></tr><tr><td>JAM Workspace</td><td>36924.51</td><td>79.53</td><td>6720.61</td><td>13.81</td><td>3.16</td><td>52.09</td></tr></table>

Table 8: Effect of hierarchical workspace organization on LoCoMo. The Flat Raw Store is the w/o Memorizer variant in Table 4. Both settings preserve the same raw histories and use the same trained Researcher, retrievers, answer model, and action budget; the Flat Raw Store removes memos, hierarchical organization, and README navigation.

## B.2 LONGCODEQA: CROSS-DOMAIN EVALUATION

We evaluate cross-domain transfer on 233 LongCodeQA queries, comprising 76, 92, and 65 examples in the 64K, 128K, and 256K subsets, respectively. Memory-Gym contains no code repositories or code-specific tasks, and no LongCodeQA examples are used for training or adaptation. We compare the untrained Qwen3.5-4B Researcher, the checkpoint after verified-trajectory SFT, and the full checkpoint after SFT followed by Hint-guided GRPO; hints are used only during training. Accuracy is reported for each context-length subset, while the overall score is computed over all 233 querie rather than as the unweighted mean of the three subset scores.

## B.3 EFFICIENCY EVALUATION DETAILS

We evaluate efficiency on LoCoMo under the same system configurations as the main experiments, separately accounting for one-time offline memory construction and per-query online serving.

Wall-clock latency. We report offline construction time in seconds per workspace and online serving time in seconds per query, averaged over the LoCoMo evaluation instances. For JAM, offline time covers hierarchical workspace construction by the Memorizer, while online time includes the Researcher’s iterative exploration and final query processing.

Token consumption. As a hardware-agnostic measure of model-processing cost, we also report token consumption for offline construction and online serving. For the per-task comparison in Figure 5, the one-time offline cost is amortized over the queries that reuse the corresponding workspace and added to the average online token cost per query. Figure 5 shows a similar quality–cost trade-off to the wall-clock results: JAM achieves the highest LoCoMo F1 while using substantially fewer tokens per task than MemAgent, whereas the one-shot memory systems incur lower token costs but also substantially lower answer quality.

Effect of workspace organization. We further analyze the w/o Memorizer variant from Table 4, corresponding to a Flat Raw Store that preserves the same raw histories but removes hierarchical organization and navigational summaries.

<table><tr><td colspan="3">Stopped within</td><td colspan="3"></td></tr><tr><td>Benchmark</td><td>5 rounds (%)</td><td>Mean</td><td>Median</td><td>P90</td><td>Budget exhausted (%)</td></tr><tr><td>LoCoMo</td><td>95.58</td><td>3.16</td><td>3</td><td>4</td><td>0.00</td></tr><tr><td>LongMemEval</td><td>82.80</td><td>3.82</td><td>3</td><td>6</td><td>0.40</td></tr><tr><td>NarrativeQA</td><td>84.00</td><td>3.92</td><td>3</td><td>8</td><td>0.00</td></tr><tr><td>HotpotQA</td><td>43.83</td><td>7.11</td><td>6</td><td>14</td><td>4.70</td></tr></table>

Table 9: Observed Researcher action rounds under a maximum budget of 20. Mean, median, and P90 are computed over all queries, with budget-exhausted queries counted as 20 rounds.

As shown in Table 8, the JAM Workspace incurs a one-time construction cost of 36,924.51 tokens and 79.53 seconds per workspace. Once constructed, however, it reduces online token consumption from 8,174.29 to 6,720.61 tokens per query, latency from 17.11 to 13.81 seconds per query, and average research rounds from 3.57 to 3.16, while improving LoCoMo F1 from 49.06 to 52.09.

This controlled comparison shows that hierarchical organization does more than improve answer quality: by providing compact navigational structure over the same underlying histories, it reduces the amount of online exploration required by the Researcher.

## B.4 RESEARCHER RUNTIME AND TERMINATION

We characterize the runtime behavior of the fully trained JAM Researcher across the four main benchmarks under the default maximum budget of 20 action rounds. We report the distribution of exploration length and the frequency of budget exhaustion.

Round counting and stopping criteria. An action round corresponds to one iteration of the Researcher’s thinking–exploration–reflection loop and may contain multiple tool calls; the subsequent finalization step is not counted as an exploration round. The Researcher terminates when it produces a positive sufficiency assessment or reaches the 20-round budget. We report the mean, median, 90th percentile (P90), and proportion of queries stopping within five rounds. Queries that reach Round 20 without a positive sufficiency assessment are considered budget-exhausted and are counted as 20 rounds in the reported statistics.

Observed runtime behavior. Table 9 shows that at least 82.80% of queries terminate within five rounds on LoCoMo, LongMemEval, and NarrativeQA. HotpotQA requires longer exploration, with only 43.83% terminating within five rounds and a P90 of 14 rounds, compared with 4–8 on the other benchmarks. Budget exhaustion remains rare, ranging from 0% to 4.70% across benchmarks. These results show that the 20-round setting acts primarily as exploration headroom rather than a fixed inference cost.

Budget-exhaustion handling. When the exploration budget is exhausted, JAM applies the same finalization step used after sufficiency-based termination to construct a context from the evidence collected so far. The 20-round cap therefore bounds exploration cost but does not imply that an exhausted query is incorrect; likewise, a positive sufficiency assessment is an internal stopping decision rather than a guarantee of complete evidence recovery.

## B.5 INFERENCE-TIME VARIABILITY

To assess the sensitivity of JAM to stochasticity during runtime exploration, we repeat inference on all four main benchmarks using three independent random seeds. The trained checkpoint, workspace, prompts, retrieval configuration, action budget, and all other evaluation settings are held fixed, so only inference-time randomness varies across runs.

Across the four benchmarks, the sample standard deviation ranges from 0.31 to 1.34 points. Variability is lowest on LongMemEval and LoCoMo and somewhat higher on NarrativeQA and HotpotQA. Overall, the results remain reasonably stable across inference seeds, with standard deviation below 1.5 points on all four benchmarks.

<table><tr><td>Benchmark</td><td>Run 1</td><td>Run 2</td><td>Run 3</td><td>Mean</td><td>Std.</td></tr><tr><td>LoCoMo F1</td><td>52.09</td><td>52.68</td><td>51.10</td><td>51.96</td><td>0.80</td></tr><tr><td>LongMemEval Acc.</td><td>65.20</td><td>64.60</td><td>65.00</td><td>64.93</td><td>0.31</td></tr><tr><td>NarrativeQA F1</td><td>45.72</td><td>43.37</td><td>45.21</td><td>44.77</td><td>1.24</td></tr><tr><td>HotpotQA F1</td><td>60.84</td><td>58.63</td><td>58.42</td><td>59.30</td><td>1.34</td></tr></table>

Table 10: Performance across three independent inference runs using the same trained JAM checkpoint. Mean and sample standard deviation are reported across inference seeds.
<table><tr><td>Method</td><td>LoCoMo</td><td>NarrativeQA</td><td>HotpotQA</td></tr><tr><td colspan="2">Evaluated with our judge</td><td></td><td></td></tr><tr><td>RAG</td><td>58.25</td><td>48.00</td><td>53.13</td></tr><tr><td>A-MEM</td><td>54.74</td><td>45.00</td><td>35.68</td></tr><tr><td>Mem0</td><td>46.23</td><td>43.00</td><td>38.02</td></tr><tr><td>MemoryOS</td><td>47.66</td><td>40.00</td><td>30.21</td></tr><tr><td>LightMem</td><td>48.57</td><td>40.00</td><td>41.15</td></tr><tr><td>MEM1</td><td>39.22</td><td>37.00</td><td>40.36</td></tr><tr><td>MemAgent</td><td>66.62</td><td>38.00</td><td>58.59</td></tr><tr><td>JAM</td><td>79.16</td><td>68.00</td><td>69.79</td></tr><tr><td colspan="2">Reported in prior work</td><td></td><td></td></tr><tr><td>Memory-R1-PPO</td><td>57.54</td><td></td><td>1</td></tr><tr><td>Memory-R1-GRPO</td><td>62.74</td><td>一</td><td>一</td></tr></table>

Table 11: LLM-judge accuracy (%) on three benchmarks. The upper block is evaluated using GPT-4o-mini. Memory-R1 results are reproduced from the original paper and are included only for reference because they are not re-evaluated under our judging pipeline. Dashes denote unreported results.

## B.6 LLM-AS-A-JUDGE EVALUATION

To assess whether JAM’s answer-quality gains persist beyond token-level F1, we additionally evaluate LoCoMo, NarrativeQA, and HotpotQA using an LLM judge. For each question, GPT-4o-mini with temperature 0 receives the reference answer and system prediction and returns CORRECT or WRONG; accuracy is the proportion of predictions judged CORRECT. We follow the Memory-R1 evaluation prompt for LoCoMo and adapt only the task description for NarrativeQA and HotpotQA. LongMemEval is not re-evaluated because its main metric is already accuracy. Memory-R1-PPO and Memory-R1-GRPO results are reproduced from the original paper and are shown separately for reference, rather than as directly comparable results under our evaluation pipeline.

As shown in Table 11, JAM achieves the highest accuracy among methods evaluated under the same judging protocol on all three benchmarks. Compared with the strongest baseline in this group, JAM improves accuracy by 12.54, 20.00, and 11.20 percentage points on LoCoMo, NarrativeQA, and HotpotQA, respectively. These results show that JAM’s gains persist under a complementary answer-level evaluation, providing additional evidence that the improvements are not limited to lexical-overlap-based scoring.

## B.7 MEMORIZER BACKBONE SENSITIVITY

We examine the sensitivity of JAM to the Memorizer backbone on LoCoMo by replacing the default Qwen3.5-4B Memorizer with Qwen3.5-122B-A10B and GPT-5.5. Each backbone constructs its own workspace from the same raw histories, while the trained Qwen3.5-4B Researcher, retrievers, browse and final-answer models, action budget, and other online inference settings are held fixed. The Researcher is not retrained for the alternative workspaces.

As shown in Table 12, changing the Memorizer backbone does not yield a consistent improvement in downstream performance. Qwen3.5-122B-A10B decreases F1 from 52.09 to 51.30, whereas GPT-5.5 increases it modestly to 52.62. Both alternatives incur higher workspace construction cost, while online latency and average exploration rounds change only slightly. The default Qwen3.5- 4B Memorizer therefore provides a favorable balance between workspace-construction cost and downstream performance in the evaluated setting.

<table><tr><td>Memorizer backbone</td><td>Build time (s/workspace)</td><td>Online time (s/query)</td><td>Avg. rounds</td><td>F1</td></tr><tr><td>Qwen3.5-4B (default)</td><td>79.53</td><td>13.81</td><td>3.16</td><td>52.09</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>132.43</td><td>14.36</td><td>3.25</td><td>51.30</td></tr><tr><td>GPT-5.5</td><td>117.65</td><td>13.19</td><td>3.03</td><td>52.62</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 12: Effect of the Memorizer backbone on LoCoMo. Each backbone constructs its own workspace from the same raw histories, while the trained Researcher and online inference configuration are held fixed.

We keep the Memorizer fixed during Researcher training because its query-agnostic workspace is reused across requests, providing a stable environment for learning query-conditioned exploration. We do not study Memorizer fine-tuning or joint Memorizer–Researcher optimization; this experiment isolates sensitivity to the Memorizer model choice.

## C CASE STUDY

To clearly illustrate the execution process of the Researcher, we present a representative NarrativeQA case in Figure 6 and Figure 7. As shown in Figure 6, the Memorizer converts the original narrative materials into a structured workspace with two major branches, film script/ and source materials/. This structure preserves raw evidence while exposing navigable paths, allowing the Researcher to reason over the workspace at both coarse and fine granularities.

Figure 7 shows the corresponding research trajectory. After receiving the query, the Researcher first inspects the overall workspace and issues multiple complementary tool calls, including search, open, and browse, to explore potentially relevant regions. After observing the initial results, it identifies a more promising evidence path and performs focused browsing over the screenplay file. This coarse-to-fine trajectory allows the Researcher to progressively narrow the search space, locate the supporting evidence, and return the correct answer with source provenance. The case illustrates that JAM enables the Researcher to start from global workspace understanding, combine multiple memory-access tools for efficient exploration, and incrementally construct query-relevant evidence beyond one-shot retrieval.

## D PROMPT

Table 13 shows the prompt template used by the Researcher for memory exploration. The prompt specifies the Researcher’s role, the available tools, the think–tool-use loop, and the required sourcegrounded answer format.

![](images/3965c3e02d8b43d02c65c70a01d5489f46fe22bb62604f23c5906c3b3e3a4d95.jpg)  
Figure 6: Case-study setup and Memorizer-constructed workspace for a NarrativeQA example.

![](images/e1c2c2f277e1b16b01b5bbe9afce65267972fb592af0c16b339f3fe84542ffee.jpg)  
Figure 7: Researcher trajectory for locating the supporting evidence in the NarrativeQA case.

Table 13: Researcher prompt for deep-research-style exploration.  
Researcher Prompt Template   
Your Role.   
You are a Research Agent exploring a hierarchical knowledge base to answer a question.   
Knowledge Base Structure.   
A Knowledge Base Overview is provided at the end of this system prompt. It includes a summary of the knowledge base   
content and the full directory structure. Each folder has a README that summarizes the files and subfolders inside it. You can   
use these README files to quickly understand the workspace structure and decide which files to inspect.   
Available Tools.   
1. search. Search for files matching a query in the knowledge base. It returns relevant file names, paths, and summaries. Use   
this tool to find potentially relevant files. Parameter: query, a keyword or phrase to search for in file content.   
2. browse. Open a file and extract information relevant to a query. An AI assistant reads the file and returns a summary of   
query-relevant content. Parameters: path, the file path to browse; query, a specific query asking for the exact information   
needed from the file.   
3. open. Open a folder to view its README summary and directory listing. Use this tool to understand the contents and   
structure of a directory. Parameter: path, the folder path to open.   
Process.   
Follow the think → tool-use loop. (1) Think about what is known, what is missing, and which tool should be used. (2) Use one   
or more tools to gather information, with at most five tool calls per round. (3) Receive observations from the tools. (4) Think   
again based on the observations and decide the next step. (5) Repeat until enough information has been collected. (6) Output   
the final answer in <answer> tags.   
Output Format During Exploration.   
Each exploration round should contain one <think> block followed by one or more <tool use> blocks. Each tool call   
must be wrapped in its own <tool use> block and must contain valid JSON.   
<think>   
[Reason about what you know, what information is missing, which tool to use, and why.]   
</think>   
<tool use>   
{"tool": "search", "query": "your search query"}   
</tool use>   
<tool use>   
{"tool": "browse", "path": "your browse path", "query": "your browse query"}   
</tool use>   
Final Answer Format.   
When enough information has been collected, output a final answer in the following format and stop the exploration loop.   
<think>   
[Summarize the collected evidence and reasoning.]   
</think>   
<answer>   
{"answer": "Your comprehensive answer to the question", "sources":   
["/path/to/source1.md", "/path/to/source2.md"], "notes": "Additional notes or   
caveats"}   
</answer>   
Guidelines.   
First check the Knowledge Base Overview, since it already provides the directory tree. Use multiple tools in one round when   
this can speed up exploration, but use at most five tool calls per round. Use multiple rounds of thinking and tool use until the   
evidence is sufficient. When outputting <answer>, make sure the answer is grounded in collected evidence and includes   
source paths. Each <tool use> block must contain valid JSON with a "tool" field.   
Begin.   
Review the Knowledge Base Overview, identify the most relevant areas for the question, and start exploring.   
Knowledge Base Overview: {knowledge base overview}