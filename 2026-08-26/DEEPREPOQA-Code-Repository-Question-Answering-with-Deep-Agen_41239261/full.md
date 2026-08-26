# DEEPREPOQA: Code Repository Question Answering with Deep Agent Exploration

Weihan Peng<sup>1</sup> Yuling Shi<sup>1</sup> Yingwei Ma<sup>2</sup> Longfei Yun<sup>3</sup>

Beijun Shen<sup>1</sup> Xiaodong Gu<sup>1,B</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>The Hong Kong University of Science and Technology <sup>3</sup>University of California San Diego

## Abstract

Answering developer questions about a software repository is a critical yet under-explored problem in software engineering. While existing repository understanding methods have advanced the field, they predominantly rely on surface-level code retrieval and lack the ability for deep reasoning over multiple files, complex software architectures, and grounding answers in long-range code dependencies. To address these limitations, we propose DEEPREPOQA, a novel question answering (QA) framework for repository-level code understanding. DEEPRE-POQA builds on the agentic framework where LLM agents find answers through a systematic tree search over the repository structure. A Monte-Carlo Tree Search (MCTS) mechanism is employed to empower agents to dynamically search, navigate, and inspect code, enabling effective multi-hop reasoning over long-range code dependencies. Comprehensive experiments on the SWE-QA benchmark demonstrate substantial performance gains over strong baselines, validating the effectiveness of systematic MCTS-guided exploration for multihop repository reasoning.

## 1 Introduction

Answering questions about a software repository is an underexplored yet critical problem (Liu and Wan, 2021; Peng et al., 2026; Chen et al., 2025a). To accomplish a function or resolve an issue, developers frequently need to navigate large, interconnected codebases, trace dependencies across multiple files, and synthesize architectural knowledge in order to answer complex questions (Jimenez et al., 2024; Zhang et al., 2023; Ouyang et al., 2024). With the rapid advancement of LLMs (Yao et al., 2023; Yang et al., 2024), there is growing interest in repository-aware assistants that can provide end-to-end support for such tasks.

Despite rapid progress, existing repository understanding methods often operate at the function level (Huang et al., 2021; Liu and Wan, 2021; Lee et al., 2022; Li et al., 2024; Sahu et al., 2024) or rely on basic retrieval-augmented generation (RAG) that surfaces local fragments of code (Zhang et al., 2023). These approaches are ill-suited to deep repository reasoning, where answers require following call chains across files, combining dispersed evidence, and articulating architectural intent (Ouyang et al., 2024). ReAct-style agents (Yao et al., 2022), such as OpenHands (Wang et al., 2025b), address some of these limitations by introducing single-path, reasoning-augmented interaction, allowing models to interleave action execution and reasoning for multi-step question answering. While they enable basic planning and verification, their single-path design still constrains exploration: important evidence may be missed if it lies outside the chosen trajectory, and reasoning depth is limited by sequential execution. Moreover, balancing exploration and exploitation remains challenging, which can result in retrieval bias or incomplete synthesis of cross-file dependencies.

To address these limitations, we propose DEEP-REPOQA<sup>1</sup>, a novel agentic question answering (QA) framework for repository-level deep understanding. Unlike flat retrieval pipelines (Zhang et al., 2023), DEEPREPOQA casts QA as a deep search-and-verify process over the repository structure, enabling LLM agents to systematically explore repositories and synthesize evidence before producing answers. The framework defines a compact, semantically meaningful action space spanning semantic search, structural navigation, and targeted code inspection. Given a question, the agent initializes a search tree and iteratively (i) perceives the current state and reasoning trajectory, (ii) plans the next step, (iii) executes the action over an index-backed representation to gather verifiable evidence, and (iv) evaluates utility to steer future exploration. Collected evidence is curated and appended to a memory path from the root to the current node; this path structures prompts, stabilizes reasoning, and, once sufficient evidence is collected, enables synthesis of a final answer grounded in cited code spans. Compared to singlepath ReAct-style agents (Peng et al., 2026), our approach introduces two key innovations: (1) an iterative search-and-verify process via MCTS for multi-hop repository reasoning, and (2) learned priors/values from LLM feedback that reduce search depth and drift. Together, these elements transform repository QA from ad-hoc retrieval to systematic search with consistent, evidence-based answers.

![](images/1d618ce57d7a04725e598408ea2aa2036991112e796e628d6f6580ba55c6de1b.jpg)  
Figure 1: Overview of the DEEPREPOQA framework.

We evaluate DEEPREPOQA on the SWE-QA benchmark (Peng et al., 2026) across multiple LLM backbones, including Kimi K2, GLM-4.6, Qwen3- Coder-480B-A35B-Instruct and GPT-5.1. DEEP-REPOQA achieves consistent improvements over strong baselines. On all models, DEEPREPOQA outperforms existing state-of-the-art ReAct-style agents, with the largest gain of 7.08 points over SWE-agent on Qwen3-Coder-480B-A35B-Instruct. DEEPREPOQA also surpasses or matches leading commercial tools such as Cursor and Tongyi Lingma when using GPT-5.1 and Kimi K2.

Ablations highlight the critical importance of MCTS, while semantic search, Evaluation Agent, and Perception Agent further enhance stability and efficiency. Efficiency analysis demonstrates that

DEEPREPOQA achieves strong performance in both computational cost and answer quality. Furthermore, a case study shows that our MCTSguided, verification-centric exploration mitigates retrieval bias by grounding answers in cross-file evidence.

Our contributions are as follows:

• We propose an agentic QA framework for repository understanding, enabling structured, verifiable multi-hop reasoning over long-range code dependencies.

• We introduce MCTS-guided exploration with LLM feedback and memory for efficient, repository-grounded search.

• We conduct a comprehensive evaluation on SWE-QA, demonstrating consistent improvements over state-of-the-art baselines.

## 2 Related Work

Repository-level code understanding has become a central challenge for modern software development, such as automated code generation (Shrivastava et al., 2023; Zhang et al., 2024; Shi et al., 2026a), code translation (Wang et al., 2024a, 2025a), and issue resolution (Jimenez et al., 2024; Chen et al., 2025b). These approaches include retrieval-augmented context gathering (Zhang et al., 2023), agent-based repository navigation (Ma et al., 2024; Yang et al., 2024; Li et al., 2026), graphbased cross-file relation modeling (Ouyang et al.,

2024), and context-aware code completion (Shrivastava et al., 2023). LongCodeZip compresses long code contexts by retaining task-relevant functions and code blocks (Shi et al., 2025). While highly effective for their intended purposes, they primarily focus on code synthesis and modification, rather than on the comprehensive, multi-hop reasoning required for deep question answering.

In code question answering, most existing work targets snippet or function-level understanding. Studies include neural QA systems for subroutine behaviors (Bansal et al., 2021), taskadaptive pre-training for code QA (Yu et al., 2022), and transformer-based models for regulatory codes (Xue et al., 2024). Recently, attention has shifted to repository-level QA, which addresses architectural questions requiring multi-file, multihop reasoning across entire codebases. While benchmarks and improvements have been proposed (Chen et al., 2025a; Strich et al., 2024; Peng et al., 2026), the field remains nascent. SWE-Bench ProMax benchmarks agents on expert-curated, multilingual, large-scale code-refactoring tasks (Shi et al., 2026b). Analyses indicate that directly applying LLMs and RAG to repository-level QA faces inherent limitations (Andryushchenko et al., 2024), highlighting the need for more structured reasoning approaches.

Comparatively, our DEEPREPOQA overcomes these limitations by formulating repository QA as a planning problem. Using a multi-module Monte-Carlo Tree Search (MCTS), our framework enables a principled and systematic exploration of the vast and complex reasoning space within a codebase. This supports deliberate, multi-hop inference across entire repositories, surpassing simple retrieval or localized analysis.

## 3 Methodology

We propose DEEPREPOQA, a repository QA method that leverages deep agent exploration. DEEPREPOQA adopts an agentic framework grounded in Monte-Carlo Tree Search (MCTS), where four specialized agents, responsible for perception, planning, execution, and evaluation, collaborate to enable deep exploration of the codebase. As illustrated in Figure 1, DEEPREPOQA answers repository questions through an iterative search-and-verify loop. Upon receiving a question, the system constructs a compact search tree representing potential next steps (e.g., search, navigate, inspect, or stop). Each step in this loop consists of four stages governed by the corresponding agents: a Perception Agent analyzes the current state of exploration. A Planning Agent then proposes promising subsequent actions based on the perceived context. An Execution Agent performs these actions to gather concrete evidence from the code. Finally, an Evaluation Agent assesses the results, assigns value scores, and propagates feedback to guide subsequent exploration. This cycle repeats until sufficient evidence has been accumulated, after which the system synthesizes a final answer grounded in explicit code citations. Through this structured, feedback-driven process, DEEPRE-POQA transforms repository QA from a linear retrieval task into a multi-path reasoning process, thereby enabling effective handling of complex, multi-hop questions. Detailed descriptions of each agent are provided in the following sections.

## 3.1 Repository Parsing and Representation

DeepRepoQA’s ability to comprehend and reason over large codebases relies on a systematic repository parsing and representation pipeline. The goal of this pipeline is to transform the raw repository into structured information that can be efficiently queried and analyzed by the model.

## 3.1.1 Repository Parsing

The parsing process involves traversing the files and directory structure of the repository to identify and extract relevant source code elements. Specifically, DeepRepoQA analyzes files to detect classes, functions, and code snippets, capturing their essential syntactic and semantic information. This process leverages language-specific parsers and heuristics based on common project layouts (e.g., src, tests), and employs Tree-sitter<sup>2</sup> to obtain lightweight, language-agnostic Abstract Syntax Trees that support precise structural extraction. Parsing also includes resolving cross-file relationships, such as function calls, class inheritance, and module dependencies, enabling the agent to build a comprehensive view of the repository’s overall architecture.

## 3.1.2 Repository Representation

Once parsing is complete, the extracted information is transformed into structured representations suitable for downstream reasoning. DeepRepoQA employs a multi-level representation strategy:

![](images/96884241f9d1f98a7eaf871171f15bf71e413f6184c5c896bdcb78ed9b9dc174.jpg)  
Figure 2: Core elements and their relations.

• Index-based Lookup: The repository directory structure and AST trees are modeled to help the agent understand the positions and contexts of classes and functions within modules. By directly parsing ASTs, the agent can precisely locate classes, functions, and code snippets without relying on fuzzy or semantic search, enabling structure-aware retrieval via FindClass, FindFunction, and FindCodeSnippet.

• Semantic Retriever: Code elements requiring conceptual-level retrieval (e.g., arbitrary code snippets) are vectorized to support semantic search and reasoning. Indexed nodes are embedded and queried using a RAGstyle Semantic Retriever, corresponding to the SemanticSearch operation in the system.

By combining these representations, DeepRepoQA achieves both high-level structural understanding andfine-grained semantic comprehension, enabling effective repository analysis and intelligent question-answering.

## 3.2 Action Space

The DeepRepoQA agent operates with a discrete action space consisting of six core capabilities. These actions equip the agent with essential tools for repository comprehension, targeted code retrieval, and answer synthesis.

The action space is designed to be both comprehensive and concise, providing a powerful set of operations for repository exploration. Actions can be categorized into name-based retrieval (FindClass, FindFunction, and FindCodeSnippet), which enables precise lookups when specific identifiers are known, and semantic retrieval (SemanticSearch), which supports conceptual queries based on natural language. The ViewCode action allows the agent to inspect specific code regions returned by previous searches, while the Finish action signals the termination of the reasoning process. This structured action space guides the agent’s reasoning and ensures that its interactions with the codebase are grounded, efficient, and interpretable. These actions and their corresponding arguments are summarized in Table 1.

Table 1: Action Options in DEEPREPOQA
<table><tr><td>Action</td><td>Arguments</td><td>Embedding?</td></tr><tr><td>FindClass</td><td>class-name, file-pattern</td><td>No</td></tr><tr><td>FindFunction</td><td>function-name, class-name, file-pattern</td><td>No</td></tr><tr><td>FindCodeSnippet</td><td>code-snippet, file-pattern</td><td>No</td></tr><tr><td>ViewCode</td><td>code-span-list, file-pattern</td><td>No</td></tr><tr><td>SemanticSearch</td><td>query, category, file-pattern</td><td>Yes</td></tr><tr><td>Finish</td><td>answer, finish-reason</td><td>No</td></tr></table>

We now describe each action in detail as follows:

• FindClass: enables the agent to locate class definitions within the repository. This action supports high-level structural reasoning by identifying key abstractions, inheritance relationships, and encapsulated functionalities.

• FindFunction: allows the agent to identify specific function or method definitions. This facilitates fine-grained understanding of code behavior, enabling the agent to trace logic, inputs, and outputs within targeted modules.

• FindCodeSnippet: provides the agent with the ability to extract relevant code fragments based on contextual needs. This action is essential for gathering actionable examples, verifying implementation details, or focusing on critical sections within large files.

• SemanticSearch: empowers the agent to perform advanced search over repository content using semantic embeddings rather than mere keyword matching. Leveraging a code embedding model, this action allows the agent to retrieve conceptually related code or documentation.

• ViewCode: grants the agent the ability to inspect the full contents of specific files. This low-level inspection is useful for understanding implementation details, configuration, and documentation. Standard command-line utilities such as cat or grep may be executed to facilitate precise content access.

• Finish: marks the completion of the answer task, indicating that the agent has synthesized a definitive answer from the collected information or reaches a predefined maximum number of iterations.

## 3.3 Agents

## 3.3.1 Perception Agent

The Perception Agent analyzes the current search state before any new action is proposed, serving as the sensory input for the MCTS framework. It synthesizes the trajectory from the root to the current node, along with information from sibling branches, to produce a concise situational report. This report serves multiple purposes: it detects redundancy to avoid re-exploring dead ends, highlights unvisited but promising code regions, surfaces conflicts or gaps in the collected evidence, and extracts structured cues (e.g., candidate symbols, files, or API calls) for further investigation. For a question like, “How does Sphinx implement the autodoc extension?”, if one branch has explored sphinx/ext/autodoc/\_\_init\_\_.py and found Documenter classes, the Perception Agent may notice that the setup() function, the standard entry point for Sphinx extensions, has not been analyzed. It will therefore flag the setup() function as a target to guide the next planning phase toward discovering how the extension integrates with Sphinx’s core. By delivering a compact, noisereduced state assessment to downstream agents, it ensures that subsequent decisions are based on a holistic view of the search progress and blind spots. The prompt is provided in Appendix F.

## 3.3.2 Planning Agent

The Planning Agent is the core decision-maker, responsible for turning the situational report from the Perception Agent into a concrete action plan. When a leaf node is selected for expansion, this agent generates a set of candidate actions to create new child nodes, effectively growing the search frontier. It inspects the curated state summary to propose diverse and strategic next steps, while explicitly avoiding redundant trials already attempted in sibling branches. Building on the same example in which perception report flagged the setup() function as a key target, the Planning Agent might generate three candidate actions for expansion: 1) a FindFunction call for setup within the sphinx/ext/autodoc/ directory, 2) a SemanticSearch for “autodoc directive registration” to find where the automodule directive is defined, and 3) a ViewCode action on sphinx/ext/autodoc/\_\_init\_\_.py to manually locate the setup function. Each proposed action becomes a new child node, representing a distinct reasoning path conditioned on both long-term goals and expected immediate information gain. The prompt is provided in Appendix F.

## 3.3.3 Execution Agent

The Execution Agent leverages the repository representations and action space introduced in Sections 3.1.1 and 3.2 to carry out the planner-selected actions over the unified structural–semantic repository index. It produces raw outputs that include results from structure-aware lookup, semantic retrieval, and file-level inspection. Fuzzy name matching is applied to mitigate minor discrepancies in model-generated queries. Immediately after execution, the evidence-curation stage filters and normalizes the raw outputs: it ranks evidence by task relevance, de-duplicates overlapping spans, collapses boilerplate, and extracts focused quotations with exact file and line ranges. Low-value or noisy snippets are discarded, and salient context is stitched into a compact, citation-ready bundle. This curated evidence set becomes the observation for the current node and the context for subsequent reasoning. The detailed argument specifications and descriptions for each action are presented in Section 3.2; the evidence-identification prompt is provided in Appendix F.

## 3.3.4 Evaluation Agent

The Evaluation Agent assesses the outcome of an executed action, providing critical feedback for the MCTS learning loop. After the Execution Agent returns a curated piece of evidence, this agent scores the action-observation pair by estimating a scalar value representing its utility in answering the original question. It also generates targeted, qualitative suggestions for subsequent steps, such as “verify a dependency path” or “inspect call sites.” For instance, if the FindFunction action for setup returns a code snippet containing the call app.add\_directive(’automodule’, AutomoduleDirective), the Evaluation Agent would assign a high value (e.g., 95) to this node, as it directly reveals the link between the directive name and its implementing class. Its feedback might suggest, “The AutomoduleDirective class is the entry point for the directive. Next, inspect its implementation to understand how it processes the directive’s content.” This value is then backpropagated up the search tree, updating the visit counts and value estimates of all nodes on the path from the current node to the root. This feedback loop allows the agent to continuously refine its search policy, prioritizing more promising reasoning paths. The prompt is provided in Appendix F.

Algorithm 1: DEEPREPOQA Algorithm   
Input: User question Q, Repository context R, Max   
iterations N   
Output: Final answer A   
1 search\_tree ← MCTSTree(Q, R);   
2 iteration\_count ← 0;   
3 max\_iterations ← N;   
4 while iteration\_count < max\_iterations do   
5 iteration\_count ← iteration\_count + 1;   
6 node ← Select(search\_tree.root); // Select   
promising node with highest UCT score,   
except those that have 3+ children   
7 new\_node ← Expand(node);   
8 situational\_report ← Perceive(new\_node,   
new\_node.parent, new\_node.siblings);   
9 new\_node.action ← PlanNextAction(Q,   
situational\_report);   
10 raw\_result ← ExecuteAction(new\_node.action);   
11 new\_node.observation ← FilterEvidence(Q,   
raw\_result);   
12 value ← Evaluate(new\_node);   
13 Backpropagate(new\_node, value);   
14 if new\_node.action.type = "Finish" then   
15 break;   
16 end   
17 end   
18 finished\_nodes ← GetFinishedNodes(search\_tree);   
19 T ← GetBestTrajectory(finished\_nodes);   
20 A ← ExtractAnswer(T);   
21 return A

## 3.4 Monte-Carlo Tree Search Process

The four agents collaborate within the MCTS framework to answer a user’s question. Starting from the question, DEEPREPOQA builds a small search tree and repeatedly runs the four-agent simulation: the Perception Agent forms a situational report, the Planning Agent proposes an action, the Execution Agent performs it with internal Filtering to curate evidence, and the Evaluation Agent scores utility and guidance. MCTS selects and expands promising branches until Finish is chosen, after which we synthesize a concise, citation-backed answer. The full pseudocode is provided in Algorithm 1. The MCTS engine guides repository exploration through four phases—selection, expansion, simulation, and back-propagation—designed to systematically navigate the complex search space of repository-level reasoning. Appendix A provides a step-by-step illustration (Figure A1).

## 3.4.1 Selection

Starting from the root node, child nodes are selected recursively according to the Upper Confi-

dence Bound (UCT) policy until reaching a leaf node. The selection strategy balances exploration and exploitation using the UCT formula:

$$
\mathrm { U C T } ( s , a ) = Q ( s , a ) + c \cdot { \sqrt { \frac { \ln N ( s ) } { N ( s , a ) } } }\tag{1}
$$

where $Q ( s , a )$ is the average value of taking action a at state s, $N ( s )$ is the visit count of s, $N ( s , a )$ is the visit count of $( s , a )$ , and c is the exploration coefficient. In practice, nodes with higher UCT scores are prioritized for expansion. For example, as illustrated in Figure A1, node5, having the highest UCT score and being eligible for expansion, is selected.

## 3.4.2 Expansion

When selection reaches a node that is not fully expanded, a new child node is added. Before proposing candidates, the Perception Agent produces a concise situational report from the direct trajectory and sibling attempts, highlighting redundancy, gaps, and promising regions. This lateral awareness lets planning avoid repeated trials without storing full sibling transcripts, since the perception report summarizes the relevant horizontal context. In practice, this expansion introduces a new branch for further exploration; for example, as illustrated in Figure A1, node5 is expanded to generate node8.

## 3.4.3 Simulation

Unlike traditional MCTS rollouts, our simulation is a single-step agent loop: Perception summarizes state and siblings, Planning proposes the next action, Execution runs it and performs internal Filtering to curate evidence, and Evaluation assigns a scalar value. The evaluation output $V ( s , a )$ directly sets $Q ( s , a )  V ( s , a )$ , allowing effective assessment without multi-step rollouts and reducing compute while preserving decision quality. In practice, as illustrated in Figure A1, this single-step simulation is executed by the four agents sequentially, and the resulting evaluation value is used to update the corresponding state-action pair. For example, after the simulation, node8 receives a value of 80, which directly updates its Q value.

## 3.4.4 Back-propagation

The simulation value is back-propagated to update Q and visit counts along the path to the root. High-quality nodes are revisited more often, while low-value branches are implicitly pruned as their selection probability drops, focusing exploration on promising regions over time. The value obtained from the simulation is propagated back up the tree from the new node to the root, updating the visit counts and Q-values of all nodes along the path. This ensures that more promising branches are explored more frequently in subsequent iterations. For example, as illustrated in Figure A1, the value of node5 increases from 60 to 70, and the value of node2 increases from 30 to 40 during backpropagation. The process continues iteratively until termination.

## 3.4.5 Answer Synthesis and Termination

When the Finish action is selected, the system composes the final answer by summarizing the reasoning trace from the best trajectory and citing decisive evidence with file and line spans for traceability. We require at least one supporting code span to ensure grounding. The process concludes by validating that the answer addresses the question type (“what,” “where,” “how,” or “why”).

This iterative process allows DeepRepoQA to dynamically focus its search on the most promising reasoning paths, leading to more effective and efficient question answering.

## 4 Experimental Setup

## 4.1 Research Questions (RQs)

We evaluate the effectiveness of DEEPREPOQA across multiple dimensions, aiming to answer the following research questions:

• RQ1: How effectively does DEEPREPOQA answer code questions at repository scale?

• RQ2: To what extent do individual components influence the overall performance of our method?

• RQ3: How does the number of exploration iterations in DEEPREPOQA affect the answering performance?

• RQ4: What are the efficiency implications of DEEPREPOQA while improving accuracy?

• RQ5: How does DEEPREPOQA perform across different types of repository-level code questions?

## 4.2 Baselines and Models

We evaluate DEEPREPOQA against four categories of baselines: Direct Prompting, RAG-based Methods, Agent-based Methods, and Commercial Tools.

Direct Prompting. This baseline queries LLMs directly without providing any repository context, aiming to assess the intrinsic knowledge of the models.

RAG-based Methods. We evaluate two retrieval-augmented generation strategies:

• Function Chunking RAG (Wang et al., 2024b), which parses the repository into function- or class-level chunks as retrieval units. Each function or class is treated as an individual code entity, and a semantic index is constructed over these entities using code embeddings. At query time, the retriever uses voyage-code-3 to encode both the query and code entities, and retrieves the top-10 most relevant chunks as context for answer generation.

• Sliding Window RAG (Zhang et al., 2023), which segments code files into overlapping chunks using a sliding window mechanism. Specifically, each file is divided into chunks of 500 lines with an overlap of 50 lines between adjacent windows. A semantic index is constructed over these chunks using voyage-code-3 embeddings. At query time, the top-10 most relevant chunks are retrieved to provide context for answer generation.

Agent-based Methods. We adopt two representative ReAct-style code agents:

• SWE-AGENT (Yang et al., 2024). We use SWE-agent v1.0, retaining all original tools and core agent logic, with only minor modifications to the entry and exit interfaces to support QA tasks.

• OPENHANDS (Wang et al., 2025b). We use version 1.1.0 of both openhands-tools and openhands-sdk, with the default agent “get\_default\_agent”, which includes tools such as terminal, file editor, task tracker, finish, and think. The agent is adapted to the QA setting without modifying its core reasoning or tool-use mechanism.

For all agent-based methods, we set the maximum number of interaction iterations to 15. When the iteration limit is reached, the final response is generated based on the accumulated message history. For DEEPREPOQA, only the adopted reasoning trajectory is used rather than all node messages.

Commercial Tools. We further compare against two state-of-the-art proprietary systems:

Table 2: Overall Results on SWE-QA
<table><tr><td rowspan="2">Methods</td><td colspan="5">Evaluation Metrics</td><td rowspan="2">Overall</td></tr><tr><td>Correctness</td><td>Completeness</td><td>Relevance</td><td>Clarity</td><td>Reasoning</td></tr><tr><td colspan="7">Commercial Tools</td></tr><tr><td>Tongyi Lingma</td><td>11.91</td><td>10.88</td><td>16.66</td><td>15.89</td><td>13.78</td><td>69.12</td></tr><tr><td>Cursor</td><td>12.31</td><td>11.34</td><td>17.32</td><td>16.07</td><td>13.67</td><td>70.71</td></tr><tr><td colspan="7">GLM-4.6</td></tr><tr><td>Direct Prompting</td><td>7.84</td><td>5.30</td><td>15.34</td><td>15.94</td><td>7.23</td><td>51.65</td></tr><tr><td>Sliding Window RAG</td><td>8.87 (+1.03)</td><td>7.11 (+1.81)</td><td>14.06 (-1.28)</td><td>15.95 (+0.01)</td><td>10.01 (+2.78)</td><td>56.00 (+ 4.35)</td></tr><tr><td>Function Chunking RAG</td><td>9.24 (+1.40)</td><td>7.45 (+2.15)</td><td>14.02 (-1.32)</td><td>15.97 (+0.03)</td><td>10.22 (+2.99)</td><td>56.90 (+ 5.25)</td></tr><tr><td>SWE-agent</td><td>10.98 (+3.14)</td><td>10.93 (+5.63)</td><td>12.53 (-2.81)</td><td>14.15 (-1.79)</td><td>12.47 (+5.24)</td><td>61.06 (+ 9.41)</td></tr><tr><td>OpenHands</td><td>10.99 (+3.15)</td><td>10.51 (+5.21)</td><td>13.92 (-1.42)</td><td>16.19 (+0.25)</td><td>13.41 (+6.18)</td><td>65.02 (+13.37)</td></tr><tr><td>DEEPREPOQA</td><td>11.11 (+3.27)</td><td>11.30 (+6.00)</td><td>13.43 (-1.91)</td><td>16.65 (+0.71)</td><td>13.41 (+6.18)</td><td>65.90 (+14.25)</td></tr><tr><td colspan="7">Kimi K2</td></tr><tr><td>Direct Prompting</td><td>6.88</td><td>4.53</td><td>15.66</td><td>15.44</td><td>7.94</td><td>50.45</td></tr><tr><td>Sliding Window RAG</td><td>8.89 (+2.01)</td><td>7.16 (+2.63)</td><td>15.07 (-0.59)</td><td>16.61 (+1.17)</td><td>10.22 (+2.28)</td><td>57.95 (+ 7.50)</td></tr><tr><td>Function Chunking RAG</td><td>9.04 (+2.16)</td><td>7.50 (+2.97)</td><td>14.49 (-1.17)</td><td>16.51 (+1.07)</td><td>10.51 (+2.57)</td><td>58.05 (+ 7.60)</td></tr><tr><td>SWE-agent</td><td>10.06 (+3.18)</td><td>9.86 (+5.33)</td><td>14.56 (-1.10)</td><td>16.03 (+0.59)</td><td>11.85 (+3.91)</td><td>62.36 (+11.91)</td></tr><tr><td>OpenHands</td><td>10.99 (+4.11)</td><td>10.35 (+5.82)</td><td>14.71 (-0.95)</td><td>15.68 (+0.24)</td><td>13.57 (+5.63)</td><td>65.30 (+14.85)</td></tr><tr><td>DEEPREPOQA</td><td>11.92 (+5.04)</td><td>11.08 (+6.55)</td><td>15.12 (-0.54)</td><td>16.31 (+0.87)</td><td>14.28 (+6.34)</td><td>68.71 (+18.26)</td></tr><tr><td colspan="7">Qwen3-Coder-480B-A35B-Instruct</td></tr><tr><td>Direct Prompting</td><td>7.71</td><td>5.85</td><td>14.85</td><td>15.00</td><td>7.92</td><td>51.33</td></tr><tr><td>Sliding Window RAG</td><td>10.04 (+2.33)</td><td>8.22 (+2.37)</td><td>15.06 (+0.21)</td><td>15.83 (+0.83)</td><td>10.98 (+3.06)</td><td>60.13 (+ 8.80)</td></tr><tr><td>Function Chunking RAG</td><td>10.01 (+2.30)</td><td>8.17 (+2.32)</td><td>15.54 (+0.69)</td><td>15.91 (+0.91)</td><td>11.50 (+3.58)</td><td>61.13 (+ 9.80)</td></tr><tr><td>SWE-agent</td><td>8.98 (+1.27)</td><td>8.49 (+2.64)</td><td>13.26 (-1.59)</td><td>15.60 (+0.60)</td><td>11.02 (+3.10)</td><td>57.35 (+ 6.02)</td></tr><tr><td>OpenHands</td><td>10.30 (+2.59)</td><td>9.57 (+3.72)</td><td>13.86 (-0.68)</td><td>15.64 (+0.64)</td><td>12.96 (+5.04)</td><td>62.33 (+11.00)</td></tr><tr><td>DEEPREPOQA</td><td>10.97 (+3.26)</td><td>10.17 (+4.32)</td><td>13.82 (-1.72)</td><td>16.19 (+1.19)</td><td>13.28 (+5.36)</td><td>64.43 (+13.10)</td></tr><tr><td colspan="7">GPT-5.1</td></tr><tr><td>Direct Prompting</td><td>7.78</td><td>5.83</td><td>14.90</td><td>16.33</td><td>7.15</td><td>51.99</td></tr><tr><td>Sliding Window RAG</td><td>8.88 (+1.10)</td><td>6.24 (+0.41)</td><td>16.89 (+1.99)</td><td>16.03 (-0.30)</td><td>10.15 (+3.00)</td><td>58.19 (+ 6.20)</td></tr><tr><td>Function Chunking RAG</td><td>10.15 (+2.37)</td><td>8.55 (+2.72)</td><td>15.57 (+0.67)</td><td>16.82 (+0.49)</td><td>11.72 (+4.57)</td><td>62.81 (+10.82)</td></tr><tr><td>SWE-agent</td><td>12.12 (+4.34)</td><td>10.61 (+4.78)</td><td>15.06 (+0.16)</td><td>14.55 (-1.78)</td><td>13.05 (+5.90)</td><td>65.39 (+13.40)</td></tr><tr><td>OpenHands</td><td>10.38 (+2.60)</td><td>9.72 (+3.89)</td><td>15.27 (+0.37)</td><td>15.85 (-0.48)</td><td>13.00 (+5.85)</td><td>64.22 (+12.23)</td></tr><tr><td>DEEPREPOQA</td><td>12.20 (+4.42)</td><td>11.63 (+5.80)</td><td>15.33 (+0.43)</td><td>16.15 (-0.18)</td><td>14.75 (+7.60)</td><td>70.06 (+18.07)</td></tr></table>

• Tongyi Lingma (VSCode plugin v2.5.16, auto mode)

• Cursor-agent (v2026.01.09-231024f, auto mode)

Tongyi Lingma relies on its proprietary model, while Cursor operates in its default “auto” mode, which dynamically selects the most suitable model based on the user query and integrates built-in retrieval and orchestration mechanisms. For both systems, we do not impose any restrictions on inference cost or the number of interaction turns. Instead, we directly adopt the final answer produced by the agent after its complete internal reasoning and tool usage process.

All open-source methods are evaluated with the same set of underlying LLMs to ensure a fair comparison, including four powerful and representative models: GLM-4.6 (Zhipu AI (Z.ai), 2025), Kimi K2 (Moonshot-AI, 2025), Qwen3-Coder-480B-

A35B-Instruct (Qwen Team, 2025), and GPT-5.1 (OpenAI, 2025).

## 4.3 Dataset

We evaluate all methods on the SWE-QA benchmark (Peng et al., 2026). SWE-QA is a repositorylevel code question answering benchmark designed to evaluate automated QA systems in realistic software development environments. The benchmark originally consists of 576 high-quality questionanswer pairs drawn from a set of open-source Python repositories. To further improve its coverage and mitigate potential data leakage issues, three additional repositories—conan, streamlink, and reflex—were incorporated, selected from SWE-Bench Live. After this expansion, the benchmark comprises 15 repositories in total, containing 720 question-answer pairs.

Table 3: Ablations on Key Components
<table><tr><td rowspan="2">Variant</td><td colspan="5">Evaluation Metrics</td><td rowspan="2">Overall</td></tr><tr><td>Correctness</td><td>Completeness</td><td>Relevance</td><td>Clarity</td><td>Reasoning</td></tr><tr><td>DEEPREPOQA</td><td>10.97</td><td>10.17</td><td>13.82</td><td>16.19</td><td>13.28</td><td>64.43</td></tr><tr><td>w/o MCTS</td><td>9.18 (-1.79)</td><td>9.62 (-0.55)</td><td>13.27 (-0.55)</td><td>16.88 (+0.69)</td><td>13.21 (-0.07)</td><td>62.16 (-2.27)</td></tr><tr><td>w/o Perception Agent</td><td>9.38 (-1.59)</td><td>9.77 (-0.40)</td><td>13.56 (-0.26)</td><td>16.77 (+0.58)</td><td>13.22 (-0.06)</td><td>62.70 (-1.73)</td></tr><tr><td>w/o Evaluation Agent</td><td>8.55 (-2.42)</td><td>9.41 (-0.76)</td><td>13.11 (-0.71)</td><td>16.45 (+0.26)</td><td>13.41 (+0.13)</td><td>60.93 (-3.50)</td></tr><tr><td>w/o Semantic Search</td><td>10.48 (-0.49)</td><td>10.08 (-0.09)</td><td>13.49 (-0.33)</td><td>16.00 (-0.19)</td><td>13.31 (+0.03)</td><td>63.36 (-1.07)</td></tr></table>

## 4.4 Evaluation Metrics

Following the evaluation protocol of SWE-QA (Peng et al., 2026), we adopt the LLM-asa-Judge paradigm to assess answer quality. To mitigate potential evaluation biases introduced by individual LLM judges, we further enhance the protocol by employing three distinct judge models, namely GPT-5.4 (OpenAI, 2025), Claude-Sonnet-4-6 (Anthropic, 2026), and Gemini-3.1-Pro (Deep-Mind, 2026). The final score is obtained by averaging the outputs of these judges. This multi-judge setup improves robustness and reduces the variance associated with any single evaluator. The LLM-asa-judge prompt is provided in Appendix F.

The evaluation is conducted across five key dimensions:

• Correctness: assesses the factual accuracy of the answer.

• Completeness: evaluates how thoroughly the answer addresses all aspects of the question.

• Relevance: measures how well the answer matches the query.

• Clarity: reflects readability and ease of understanding.

• Reasoning: evaluates the logical soundness, depth, and validity of the reasoning process underlying the answer.

Each dimension is scored independently on a continuous scale from 0 to 20, following a detailed scoring rubric with predefined criteria for different score ranges. Specifically, each score interval (e.g., 16–20, 12–16, etc.) is associated with explicit qualitative guidelines that describe the expected level of performance for that dimension, enabling finegrained and consistent evaluation across responses.

## 4.5 Implementation Details

Our SemanticSearch component employs the voyage-code-3 code embedding model. For

MCTS, we cap the number of children expanded per node at three (max\_expand = 3) and set the maximum exploration iterations to 15 for the main results reported in Table 2. During the exploration phase of MCTS, we use a temperature of 0.7 to balance exploration and exploitation, while for answer generation we set the temperature to 0. For answer generation, we fix the temperature to 0 while keeping other decoding parameters (e.g., top\_p) at their default values as provided by each API, without additional tuning.

## 5 Results

## 5.1 RQ1: Main Results

Table 2 compares the overall results by various methods. Across models, DEEPREPOQA is the strongest open-source system and closes the gap to commercial tools. With GPT-5.1 it achieves an overall score of 70.06, surpassing Tongyi Lingma at 69.12 and remaining close to Cursor (70.71). With GLM-4.6, Kimi K2 and Qwen3-Coder-480B-A35B-Instruct it remains the top open-source method, and on Kimi K2, further narrows the gap to commercial tools.

The most significant gains manifest in Correctness, Completeness, and Reasoning—dimensions critically important for multi-hop retrieval and evidence synthesis, while maintaining strong Relevance and fluent Clarity. This pattern reveals that MCTS-guided exploration and index-grounded actions excel at generating answers that are not only more accurate and comprehensive, but also more rigorously reasoned.

RAG variants outperform direct prompting but still trail agentic approaches, particularly on Correctness and Completeness, which indicates that RAG’s retrieval may introduce substantial irrelevant code while missing critical code segments. DEEPREPOQA strengthens the substantive dimensions across backbones and, with stronger backbones, reaches or exceeds commercial performance. Overall, systematic, feedback-guided search over repositories proves more effective than flat retrieval or single-trajectory agents. However, while SWEagent and OpenHands execute more sophisticated actions, they may commit to incorrect search trajectories without timely correction mechanisms, causing them to persist in unproductive explorations, whereas our MCTS-guided approach enables continuous feedback-based course correction.

## Finding 1

DEEPREPOQA achieves state-of-the-art performance among open-source systems and approaches leading commercial tools, especially in correctness, completeness, and reasoning.

## 5.2 RQ2: Ablation Study

To understand the contribution of individual components, we conduct ablation studies removing: (i) MCTS framework (replaced with greedy singletrajectory search), (ii) Perception Agent (without state summarization), (iii) Evaluation Agent (reverted to rollout-based evaluation), and (iv) Semantic Search (exact name-based retrieval only). The results with Qwen3-Coder-480B-A35B-Instruct as the base model in Table 3 demonstrate that each component meaningfully contributes to overall performance. Removing MCTS causes a 2.27-point performance drop, effectively reducing our method to OpenHands-level performance and underscoring the critical value of guided exploration over greedy single-trajectory search. Notably, the Evaluation Agent emerges as the most critical component—its removal results in the largest performance degradation of 3.50 points, reflecting its essential role in guiding path selection within the MCTS framework. The Perception Agent and Semantic Search provide additional improvements of 1.73 and 1.07 points respectively, demonstrating that comprehensive state summarization and semantic-aware retrieval collectively enhance answer quality across all dimensions.

The MCTS framework’s substantial impact across all evaluation dimensions confirms that systematic exploration-exploitation balance is fundamental for repository-level reasoning.

The component hierarchy reveals that systematic exploration mechanisms outweigh individual optimization techniques in importance. The notable impact of removing the Evaluation Agent validates our design choice to replace expensive rollouts with learned value estimation, while semantic search contributes balanced improvements across diverse question types. These findings demonstrate that principled algorithmic innovations (MCTS, value estimation, semantic actions) drive primary performance gains, providing clear guidance for future development priorities in repository-level QA systems.

## Finding 2

All components in DEEPREPOQA prove essential with no redundancy: each removal incurs performance loss (−1.07 to −3.50), validating the necessity of our integrated design.

## 5.3 RQ3: Impact of Exploration Iterations

To address RQ3, we analyze the impact of varying the maximum number of exploration iterations on the performance of DEEPREPOQA. The results with Qwen3-Coder-480B-A35B-Instruct as the base model are summarized in Table 4. We can observe that increasing the maximum number of explored nodes steadily improves performance: the overall score rises from 55.06 (5 nodes) to 62.97 (10 nodes, +7.91) and to 65.33 (20 nodes, +10.27). Correctness increases from 8.27 to 11.57 (+3.30), Completeness from 7.97 to 10.76 (+2.79), and Reasoning from 8.81 to 12.88 (+0.07). Notably, the largest performance gain occurs between 5 and 10 nodes, indicating that the MCTS framework requires a minimum number of iterations to fully realize its advantage. Within the experimental range, more iterations generally lead to higher scores, reflecting more thorough exploration; however, the incremental gains diminish at higher iteration counts, and performance between 20 and 30 nodes is observed to be almost identical. This suggests that users should balance the trade-off between computational cost and performance improvement when choosing the number of exploration iterations.

We further analyze the agent’s exploration behavior by examining its action usage. The results are summarized in Figure 3.

The distribution of actions used during trajectories (Figure 3) clarifies the underlying mechanism. The agent’s behavior is dominated by a combination of broad, meaning-based exploration (SemanticSearch) and targeted, structure-aware navigation (FindClass,

FindCodeSnippet), supplemented by frequent use of ViewCode for verification. Actions like FindFunction and the terminal Finish action are less common, indicating that the agent spends most of its budget actively gathering and confirming evidence. This pattern of guided exploration followed by verification directly contributes to the observed improvements in Correctness and Completeness as the exploration budget increases, as it allows the agent to build a more comprehensive and wellgrounded understanding of the repository before synthesizing an answer.

Table 4: Effect of exploration budget (higher is better).
<table><tr><td rowspan="2">Max Nodes</td><td colspan="5">Evaluation Metrics</td><td rowspan="2">Overall</td></tr><tr><td>Correctness</td><td>Completeness</td><td>Relevance</td><td>Clarity</td><td>Reasoning</td></tr><tr><td>5</td><td>8.27</td><td>7.97</td><td>13.71</td><td>16.30</td><td>8.81</td><td>55.06</td></tr><tr><td>10</td><td>10.43</td><td>9.99</td><td>13.64</td><td>16.21</td><td>12.70</td><td>62.97</td></tr><tr><td>15</td><td>10.97</td><td>10.17</td><td>13.82</td><td>16.19</td><td>13.28</td><td>64.43</td></tr><tr><td>20</td><td>11.57</td><td>10.76</td><td>13.83</td><td>16.29</td><td>12.88</td><td>65.33</td></tr><tr><td>30</td><td>11.43</td><td>10.89</td><td>13.77</td><td>16.16</td><td>12.90</td><td>65.15</td></tr></table>

![](images/b150f6a6ea4ec5fe8348efe7ad83d72cb5845177c3e2185b577c309f196d5652.jpg)  
Figure 3: Distribution of action usage during trajectories.

## Finding 3

Increasing the number of exploration iterations improves performance, but the gains diminish at higher iteration counts, with results plateauing at large budgets.

## 5.4 RQ4: Efficiency Analysis

Appendix B (Table B1) summarizes the computational efficiency of all methods, including the number of input and output tokens. Under the same number of iterations (15), DEEPREPOQA uses fewer input tokens compared to SWE-agent, OpenHands, and even commercial tools without iteration limits, while slightly more output tokens, mainly due to the design of the Perception Agent and Evaluation Agent. Overall, considering both input and output tokens, DEEPREPOQA’s computational cost is comparable to state-of-the-art agents and commercial tools, while still achieving better benchmark performance. These results highlight that DEEPREPOQA achieves efficient token usage while delivering superior answer quality across diverse repositories.

## Finding 4

DEEPREPOQA achieves strong computational efficiency while delivering superior repositorylevel QA performance.

## 5.5 RQ5: Performance across Question Types

Table 5 shows the performance of DEEPREPOQA across different question types. The results indicate that DEEPREPOQA performs strongly across all major question categories, with “Why” and “How” questions achieving the highest average scores, suggesting the model is particularly effective at understanding design rationale and reasoning about system design or algorithmic implementation. “What” and “Where” questions achieve slightly lower scores but still demonstrate solid performance, typically involving fetching definitions, basic concepts, or precise locations of features and identifiers. This detailed breakdown highlights the versatility of our approach while pointing to potential areas for improvement in handling these types of queries.

## Finding 5

MCTS-guided, feedback-driven exploration substantially improves correctness, completeness, and reasoning compared to both RAG and agentic baselines.

Table 5: Results by question type (higher is better).
<table><tr><td rowspan=1 colspan=11>Qwen3-CoderQuestion Type                GLM-4.6    Kimi K2                                GPT-5.1480B-A35B-Instruct    Average</td></tr><tr><td rowspan=1 colspan=1>What</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>63.67</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>67.12</td><td rowspan=1 colspan=1>63.30</td><td rowspan=1 colspan=2>67.89</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.50</td></tr><tr><td rowspan=1 colspan=1>Architecture exploration</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>59.80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62.64</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>60.49</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>63.41</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>61.59</td></tr><tr><td rowspan=1 colspan=1>Concept / Definition</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>71.00</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.66</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70.62</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.52</td></tr><tr><td rowspan=1 colspan=1>Dependency tracing</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.41</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>67.72</td><td rowspan=1 colspan=1>63.75</td><td rowspan=1 colspan=2>69.64</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.38</td></tr><tr><td rowspan=1 colspan=1>Why</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.01</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>71.49</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.35</td></tr><tr><td rowspan=1 colspan=1>Design rationale</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70.02</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70.26</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.28</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>73.95</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70.13</td></tr><tr><td rowspan=1 colspan=1>Purpose Exploration</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.84</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>69.46</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.83</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>73.11</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>69.31</td></tr><tr><td rowspan=1 colspan=1>Performance</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.17</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>66.65</td><td rowspan=1 colspan=1>63.19</td><td rowspan=1 colspan=2>67.41</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.61</td></tr><tr><td rowspan=1 colspan=1>Where</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.03</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>66.21</td><td rowspan=1 colspan=1>65.05</td><td rowspan=1 colspan=2>69.58</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.47</td></tr><tr><td rowspan=1 colspan=1>Data / Control-flow</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>63.25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>64.57</td><td rowspan=1 colspan=1>63.82</td><td rowspan=1 colspan=2>66.48</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.53</td></tr><tr><td rowspan=1 colspan=1>Feature Location</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.85</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.09</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.79</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67.92</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65.91</td></tr><tr><td rowspan=1 colspan=1>Identifier Location</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.99</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67.97</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.54</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>74.34</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.96</td></tr><tr><td rowspan=1 colspan=1>How</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>66.89</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>72.72</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.27</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>71.28</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.79</td></tr><tr><td rowspan=1 colspan=1>System Design</td><td rowspan=1 colspan=5>66.72        72.19</td><td rowspan=1 colspan=5>63.80              70.95       68.42</td></tr><tr><td rowspan=1 colspan=1>Algorithm Implementation</td><td rowspan=1 colspan=2>67.11</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>73.21</td><td rowspan=1 colspan=1>64.64</td><td rowspan=1 colspan=2>72.41</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>69.34</td></tr><tr><td rowspan=1 colspan=1>API / Framework Support</td><td rowspan=1 colspan=3>66.84</td><td rowspan=1 colspan=2>72.76</td><td rowspan=1 colspan=1>64.37</td><td rowspan=1 colspan=2>70.48</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.61</td></tr></table>

## 6 Discussion

## 6.1 Judge Reliability

To assess the reliability of our evaluation, we examine it from two complementary angles: agreement among the three LLM judges (Section 4.4), and validation of the LLM panel against human experts.

Among the three judges, we measure ranking consistency with a Spearman-style score whose definition and full derivation are given in Appendix C. We obtain an overall consistency of $\rho _ { \mathrm { o v e r a l l } } = 1$ , indicating perfect agreement among the LLM judges on system rankings.

We further validate the LLM panel against a panel of three human experts. The LLM-consensus and human-consensus scores correlate strongly (Pearson r = 0.972), and the two panels agree on 88.2% of pairwise system comparisons (Cohen’s $\kappa = 0 . 7 2 5$ , substantial). The complete study—perdimension Krippendorff’s α, per-answer human– LLM correlations, panel-consensus agreement, and a length-bias analysis—is reported in Appendix C.

## 6.2 Case Study

To illustrate the differences in answer quality, we examine a scikit-learn question: “What does sklearn.preprocessing.StandardScaler do to target variables?” (Figure A2). The baseline OpenHands retrieves only snippets about how StandardScaler processes feature vectors, and therefore wrongly treats the target vector the same way— missing that StandardScaler ignores the y argument of fit entirely. In contrast, DEEP-REPOQA grounds its answer in the actual API:

it locates the fit documentation (“y: None, Ignored”), notes that transform accepts only X, and surfaces TransformedTargetRegressor as sklearn’s dedicated mechanism for target-variable transformation.

Besides, for a systematic account of where DEEPREPOQA fails, Appendix E reports an error analysis over the lowest-scoring trajectories, identifying four recurring failure modes—wrong location (F1), adjacent-concept retrieval (F2), keyword-only exploration (F3), and open-ended design (F4)— together with the symbol-anchoring and earlyretrieval patterns that characterize high-scoring answers.

## 6.3 Threats to Validity

## 6.3.1 Internal Validity

A primary threat to internal validity is data contamination: LLMs may have seen benchmark repositories or similar code during pre-training, achieving high performance through memorization rather than genuine reasoning. We mitigate this in three ways. First, we evaluate both open-weight and proprietary models, reducing the chance that all models memorized the same repositories (Section 4.2). Second, we include a retrieval-free baseline that reflects each model’s unaided knowledge, isolating gains from our exploration and reasoning mechanisms rather than latent knowledge (Section 4.2). Third, our dataset draws on diverse, relatively less popular repositories, lowering the risk of overlap with pre-training corpora (Section 4.3).

## 6.3.2 External Validity

First, in terms of model generalization, we evaluate four representative large models, covering both open-source and closed-source models, indicating that our findings are not limited to a single model type (Section 4.2). Second to mitigate potential bias in evaluation, we employ three different models as judges, compute the average score, and verify the correlation among them, ensuring that the scoring is robust and reliable (Section 6.1). Third, to test language generalization beyond Python, we evaluate DEEPREPOQA on a Java subset (30 QA pairs over Strata, Fineract, and Shiro). It ranks first among open-source methods and trails only the commercial Cursor system, mirroring the Python ranking; full results are reported in Appendix D.

## 7 Conclusion

In this paper, we introduced DEEPREPOQA, a novel agent-based framework that reformulates repository-level code question answering as a planning problem solved with Monte-Carlo Tree Search. Comprehensive evaluations demonstrate that DEEPREPOQA achieves consistent 4-7% improvements over SOTA agent baselines across multiple LLMs, with gains concentrated on Correctness, Completeness and Reasoning dimensions critical for multi-hop repository understanding. Ablation studies identify MCTS as the primary performance driver, while the design of Perception, Evaluation, and other agents also contributes to stability and efficiency. Our work shows the effectiveness of systematic tree search for repositorylevel reasoning with solid efficiency and provides a foundation for building more capable software engineering assistants.

## Data-Availability Statement

Our code and data are available at https://doi.   
org/10.5281/zenodo.21063159.

## References

Georgy Andryushchenko, Vladimir Ivanov, Vladimir Makharev, Elizaveta Tukhtina, and Aidar Valeev. 2024. Leveraging large language models in code question answering: Baselines and issues. In NLP (CCIS).

Anthropic. 2026. Claude sonnet 4.6. https://www. anthropic.com/news/claude-sonnet-4-6. Accessed: 2026-03-26.

Aakash Bansal, Zachary Eberhart, Lingfei Wu, and Collin McMillan. 2021. A neural question answering system for basic questions about subroutines. In 28th IEEE International Conference on Software Analysis, Evolution and Reengineering, SANER 2021, Honolulu, HI, USA, March 9-12, 2021, pages 60–71. IEEE.

Jialiang Chen, Kaifa Zhao, Jie Liu, Chao Peng, Jierui Liu, Hang Zhu, Pengfei Gao, Ping Yang, and Shuiguang Deng. 2025a. Coreqa: uncovering potentials of language models in code repository question answering. arXiv preprint arXiv:2501.03447.

Silin Chen, Shaoxin Lin, Xiaodong Gu, Yuling Shi, Heng Lian, Longfei Yun, Dong Chen, Weiguo Sun, Lin Cao, and Qianxiang Wang. 2025b. Swe-exp: Experience-driven software issue resolution. arXiv preprint arXiv:2507.23361.

Google DeepMind. 2026. Gemini 3.1 pro model card.

Junjie Huang, Duyu Tang, Linjun Shou, Ming Gong, Ke Xu, Daxin Jiang, Ming Zhou, and Nan Duan. 2021. Cosqa: 20,000+ web queries for code search and question answering. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5690–5700.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157.

Changyoon Lee, Yeon Seonwoo, and Alice Oh. 2022. CS1QA: A dataset for assisting code-based question answering in an introductory programming course. CoRR, abs/2210.14494.

Han Li, Yuling Shi, Shaoxin Lin, Xiaodong Gu, Heng Lian, Xin Wang, Yantao Jia, Tao Huang, and Qianxiang Wang. 2026. SWE-Debate: Competitive multiagent debate for software issue resolution. In ICSE.

Zehan Li, Jianfei Zhang, Chuantao Yin, Yuanxin Ouyang, and Wenge Rong. 2024. ProCQA: A largescale community-based programming question answering dataset for code search. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 13057–13067.

Chenxiao Liu and Xiaojun Wan. 2021. Codeqa: A question answering dataset for source code comprehension. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021, pages 2618–2632. Association for Computational Linguistics.

Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. 2024. How to understand whole software repository. arXiv preprint arXiv:2406.01422.

Moonshot-AI. 2025. kimi-k2-0905-preview model. https://platform.moonshot.ai.

OpenAI. 2025. gpt-5.1-2025-11-13 model. https:// openai.com/index/gpt-5-1-for-developers/.

OpenAI. 2025. Gpt-5.4 technical report / model card.

Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, Jiawei Han, Hongming Zhang, and Dong Yu. 2024. Repograph: Enhancing ai software engineering with repository-level code graph. In The Thirteenth International Conference on Learning Representations.

Weihan Peng, Yuling Shi, Yuhang Wang, Xinyun Zhang, Beijun Shen, and Xiaodong Gu. 2026. Swe-qa: Can language models answer repository-level code questions? In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 8230–8245.

Qwen Team. 2025. qwen3-coder-480b-a35b-instruct model. https://qwenlm.github.io/blog/ qwen3-coder/.

Surya Prakash Sahu, Madhurima Mandal, Shikhar Bharadwaj, Aditya Kanade, Petros Maniatis, and Shirish K. Shevade. 2024. Codequeries: A dataset of semantic queries over code. In Proceedings of the 17th Innovations in Software Engineering Conference, ISEC 2024, Bangalore, India, February 22-24, 2024, pages 7:1–7:11. ACM.

Yuling Shi, Yichun Qian, Hongyu Zhang, Beijun Shen, and Xiaodong Gu. 2025. Longcodezip: Compress long context for code language models. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 141–153. IEEE.

Yuling Shi, Songsong Wang, Chengcheng Wan, and Xiaodong Gu. 2026a. From code to correctness: Closing the last mile of code generation with hierarchical debugging. In ICSE.

Yuling Shi, Jinghan Xu, Kelin Fu, Wenhao Zeng, Shilin He, Lei Zhang, Yue Liu, Zelin Zhao, Terry Yue Zhuo, Jialun Cao, et al. 2026b. Swe-bench promax: Benchmarking agents on large-scale multilingual code refactoring. arXiv preprint arXiv:2608.09802.

Disha Shrivastava, Denis Kocetkov, Harm De Vries, Dzmitry Bahdanau, and Torsten Scholak. 2023. Repofusion: Training code models to understand your repository. arXiv preprint arXiv:2306.10998.

Jan Strich, Florian Schneider, Irina Nikishina, and Chris Biemann. 2024. On improving repository-level code qa for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 209–244.

Chaofan Wang, Tingrui Yu, Beijun Shen, Jie Wang, Dong Chen, Wenrui Zhang, Yuling Shi, Chen Xie, and Xiaodong Gu. 2025a. Evoc2rust: A skeletonguided framework for project-level c-to-rust translation. arXiv preprint arXiv:2508.04295.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. 2025b. Openhands: An open platform for ai software developers as generalist agents. In International Conference on Learning Representations, volume 2025, pages 65882–65919.

Yanli Wang, Yanlin Wang, Suiquan Wang, Daya Guo, Jiachi Chen, John Grundy, Xilin Liu, Yuchi Ma, Mingzhi Mao, Hongyu Zhang, et al. 2024a. Repotransbench: A real-world benchmark for repositorylevel code translation. arXiv e-prints, pages arXiv– 2412.

Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and Zibin Zheng. 2024b. RLCoder: Reinforcement learning for repositorylevel code completion. In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE), pages 165–177. IEEE Computer Society.

Xiaorui Xue, Jiansong Zhang, and Yunfeng Chen. 2024. Question-answering framework for building codes using fine-tuned and distilled pre-trained transformer models. Automation in Construction, 168:105730.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528– 50652.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Tingrui Yu, Xiaodong Gu, and Beijun Shen. 2022. Code question answering via task-adaptive sequence-tosequence pre-training. In 2022 29th Asia-Pacific Software Engineering Conference (APSEC), pages 229–238. IEEE.

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code completion through iterative retrieval and generation. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repolevel coding challenges. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational

Linguistics (Volume 1: Long Papers), pages 13643– 13658.

Zhipu AI (Z.ai). 2025. glm-4.6 model. https://docs. bigmodel.cn/cn/guide/models/text/glm-4.6.

## A MCTS Workflow Illustration

![](images/6e0bf4c93ee406d8bd0f64b1954099f4576a2426469e958ef6e00a43ceb4d28d.jpg)  
Figure A1: Illustration of our MCTS workflow.

![](images/fa4de62ed128058764bcc7b831a21728a92ba8f27ca6a36aa192aa2d764aa9ec.jpg)  
(b) DEEPREPOQA: retrieved context and answer.  
Figure A2: Case study on scikit-learn StandardScaler, comparing the retrieved evidence and generated answers of OpenHands and DEEPREPOQA.

## B Token Usage

Table B1: Token usage per question, reported as input/output tokens (lower is better).
<table><tr><td>Model</td><td>Qwen3-Coder- 480B-A35B-Instruct</td><td>Kimi K2</td><td>GLM-4.6</td><td>GPT-5.1</td><td>Average</td></tr><tr><td>Direct</td><td>94/65</td><td>94/34</td><td>94/76</td><td>94/87</td><td>94/63</td></tr><tr><td>Function Chunking RAG</td><td>5,084/106</td><td>5,084/92</td><td>5,084/240</td><td>5,084/196</td><td>5,084/130</td></tr><tr><td>Sliding Window RAG</td><td>6,302/112</td><td>6,302/110</td><td>6,302/233</td><td>6,302/201</td><td>6,302/138</td></tr><tr><td>SWE-agent</td><td>165,206/4,546</td><td>154,295/1,398</td><td>70,870/1,902</td><td>122,833/6,100</td><td>126,026/4,627</td></tr><tr><td>OpenHands</td><td>108,703/2,128</td><td>101,594/1,787</td><td>42,433/2,158</td><td>69,192/2,127</td><td>87,045/1,930</td></tr><tr><td>DEEPREPOQA</td><td>119,633/3,812</td><td>68,679/4,059</td><td>42,888/9,867</td><td>83,526/7,251</td><td>78,681/6,247</td></tr><tr><td>Tongyi Lingma</td><td colspan="3">116,372/1,589</td><td></td><td>116,372/1,589</td></tr><tr><td>Cursor-agent</td><td colspan="3">129,458/1,635</td><td></td><td>129,458/1,635</td></tr></table>

## C LLM-as-a-Judge Reliability

All human and LLM evaluations in this section are conducted on answers generated by the GPT-5.1 backbone (60 questions × 6 systems = 360 answers).

The 60 questions are sampled from the 15 evaluation repositories with stratification over the four top-level intent categories (why, where, how, what): for each repository we randomly draw one question from each category, yielding $1 5 \times 4 = 6 0$ questions. This design guarantees that every repository and every intent category is equally represented, so the reliability statistics below are not dominated by any single repository or question type.

## C.1 Intra-panel consistency — Krippendorff’s α (interval level)

Table C1: Intra-panel inter-rater reliability (Krippendorff’s $\alpha ,$ interval level) for the human panel, the LLM panel, and the combined 6-rater panel, broken down by evaluation dimension.

<table><tr><td>Dimension</td><td>Human α</td><td>LLM α</td><td>All-6 α</td></tr><tr><td>Correctness</td><td>0.837</td><td>0.908</td><td>0.878</td></tr><tr><td>Completeness</td><td>0.822</td><td>0.883</td><td>0.855</td></tr><tr><td>Relevance</td><td>0.828</td><td>0.832</td><td>0.838</td></tr><tr><td>Clarity</td><td>0.324</td><td>0.406</td><td>0.420</td></tr><tr><td>Reasoning</td><td>0.821</td><td>0.825</td><td>0.834</td></tr><tr><td>Overall</td><td>0.840</td><td>0.871</td><td>0.861</td></tr></table>

4/5 dimensions exceed the 0.80 substantial-reliability threshold for both panels. Clarity is consistently the hardest dimension for both human and LLM panels, indicating a domain-inherent difficulty.

## C.2 Individual cross-group Spearman $\rho$ matrix (per-answer, all $\mathbf { p < 0 . 0 0 1 ) }$

Table C2: Per-answer Spearman $\rho$ between each LLM judge and each human expert (all $\mathrm { p } < 0 . 0 0 1 )$ .
<table><tr><td></td><td>Expert 1</td><td>Expert 2</td><td>Expert 3</td><td>LLM row mean</td></tr><tr><td>GPT-5.4</td><td>0.871</td><td>0.793</td><td>0.791</td><td>0.818</td></tr><tr><td>Claude-Sonnet-4-6</td><td>0.661</td><td>0.915</td><td>0.918</td><td>0.831</td></tr><tr><td>Gemini-3.1-Pro</td><td>0.789</td><td>0.848</td><td>0.847</td><td>0.828</td></tr><tr><td>Human col mean</td><td>0.774</td><td>0.852</td><td>0.852</td><td>Cross-group mean: 0.826</td></tr></table>

Within-human pairwise $\rho ; \operatorname { E l - E } 2 = 0 . 6 8 0 , \operatorname { E l - E } 3 = 0 . 6 6 8 , \operatorname { E } 2 \mathrm { - } \operatorname { E } 3 = 0 . 9 1 7 $ mean 0.755.

The lower Claude–Expert1 entry (0.661) mirrors Expert 1’s own low within-human correlations (0.668– 0.680): Expert 1 also correlates weakly with the other two human experts, pointing to an idiosyncratic rating style specific to this expert.

## C.3 Panel-consensus agreement (mean-of-3 vs mean-of-3)

Beyond individual rater consistency, we compare the two panels at the consensus level by averaging the three human ratings into a human-consensus score and the three LLM ratings into an LLM-consensus score. Table C3 reports the agreement between these two consensus signals on the 0–100 scale.

Table C3: Panel-level agreement between the human-consensus score (mean of 3 experts) and the LLM-consensus score (mean of 3 LLM judges).
<table><tr><td>Measure</td><td>Value</td></tr><tr><td>Pearson r</td><td>0.972 [95% CI 0.964–0.979, bootstrap 2,000 resamples]</td></tr><tr><td>Spearman ρ</td><td>0.942</td></tr><tr><td>MAE (0–100 scale)</td><td>5.22 pts (5.2% of scale; inter-method gap ~25 pts)</td></tr></table>

MAE (5.22 pts) is roughly one-fifth of the typical gap between competing systems (\~25 pts), so the panel-level signal is precise enough to distinguish methods reliably. Combined with the very high Pearson r (0.972) and Spearman ρ (0.942), the LLM-consensus closely tracks the human-consensus both in absolute level and in ordering.

## C.4 Pairwise preference agreement

The question of primary practical interest is “which system is better?” Table C4 therefore reports agreement at the decision level, converting each pair of system scores into a preference (A > B, tie, or B > A) and measuring how often the two panels reach the same verdict.

Table C4: Pairwise preference agreement between human and LLM panels on system-vs-system comparisons.
<table><tr><td>Measure</td><td>Value</td></tr><tr><td>Preference agreement rate</td><td>88.2%</td></tr><tr><td>Cohen&#x27;s κ (3-class: A &gt; B / tie / B &gt; A)</td><td>0.725 (substantial)</td></tr><tr><td>System-level ranking Spearman ρ</td><td>0.900 (p = 0.037)</td></tr></table>

## C.5 Length-bias analysis

We also examine whether answer length influences LLM scores independently of quality, using two analyses: an OLS regression that partials out system and dimension effects, and a stratified comparison across short / medium / long answers (Table C5).

OLS model: score \~ length + system + dim.

Correlation of (LLM consensus − Human consensus) with answer length: $\mathbf { r } = \mathbf { 0 . 1 7 9 , } \mathrm { p } = 0 . 0 0 6 , \mathrm { r } ^ { 2 } =$ 0.032.

Stratified human-LLM agreement and score difference:

Table C5: Length-stratified human-LLM agreement, showing the share of answers in each bin, the mean LLM − Human score gap, and the per-answer Pearson r between the two panels.
<table><tr><td>Length bin</td><td>percentage</td><td>Mean (LLM — Human)</td><td>Pearson r (human vs LLM)</td></tr><tr><td>Short (&lt; 100 words)</td><td>53.9%</td><td>-4.85</td><td>0.983</td></tr><tr><td>Medium (100–300 words)</td><td>14.6%</td><td>-1.16</td><td>0.985</td></tr><tr><td>Long (&gt; 300 words)</td><td>31.4%</td><td>-3.06</td><td>0.919</td></tr></table>

LLM scores are uniformly lower than human scores across all length bins (LLMs are more stringent). Human-LLM agreement is uniformly excellent $( \mathbf { r } \geq 0 . 9 2 )$ .

## C.6 Inter-judge ranking consistency

To assess the reliability of our evaluation, we also measure the ranking consistency among the three LLM judges. We observe a high level of consistency which, combined with the human verification protocol of SWE-QA, indicates that the judges are reliable. The computation is as follows.

Let $R _ { i , j } ( m )$ denote the rank assigned by judge j to baseline m for target model i, and let $S _ { j , k } =$ $\begin{array} { r } { \sum _ { m = 1 } ^ { n } \big ( \tilde { R _ { i , j } } ( m ) - R _ { i , k } ( m ) \big ) ^ { 2 } } \end{array}$ . The per-model consistency score is

$$
\rho _ { i } = \frac { 1 } { 3 } \sum _ { ( j , k ) \in \{ ( 1 , 2 ) , ( 1 , 3 ) , ( 2 , 3 ) \} } \left( 1 - \frac { 6 S _ { j , k } } { n ( n ^ { 2 } - 1 ) } \right) ,\tag{C1}
$$

and the overall consistency is $\begin{array} { r } { \rho _ { \mathrm { o v e r a l l } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \rho _ { i } } \end{array}$ . In our experiments, we find $\rho _ { \mathrm { o v e r a l l } } = 1$ , indicating perfect agreement among the LLM judges; this judge-level agreement is complemented by the humanvalidation study reported in the preceding subsections.

## D Cross-language Generalization — Java Results

Table D1: Cross-language generalization on the Java subset (30 QA pairs across Strata, Fineract, and Shiro). DeepRepoQA ranks first among open-source methods and trails only the commercial Cursor system, mirroring the Python ranking.
<table><tr><td>Method</td><td>Correctness</td><td>Completeness</td><td>Relevance</td><td>Clarity</td><td>Reasoning</td><td>Overall</td></tr><tr><td>Direct</td><td>5.42</td><td>4.18</td><td>10.85</td><td>11.36</td><td>5.71</td><td>37.52</td></tr><tr><td>Function-Chunking RAG</td><td>7.83</td><td>6.97</td><td>12.14</td><td>12.05</td><td>7.62</td><td>46.61</td></tr><tr><td>Sliding-Window RAG</td><td>8.46</td><td>7.51</td><td>12.48</td><td>12.23</td><td>8.05</td><td>48.73</td></tr><tr><td>SWE-agent</td><td>11.27</td><td>9.94</td><td>13.78</td><td>12.58</td><td>11.32</td><td>58.89</td></tr><tr><td>OpenHands</td><td>11.41</td><td>10.08</td><td>13.91</td><td>12.65</td><td>11.46</td><td>59.51</td></tr><tr><td>DeepRepoQA</td><td>12.29</td><td>11.30</td><td>14.37</td><td>13.13</td><td>12.21</td><td>63.29</td></tr><tr><td>Tongyi Lingma (commercial)</td><td>11.84</td><td>10.51</td><td>14.08</td><td>12.79</td><td>11.79</td><td>61.01</td></tr><tr><td>Cursor (commercial)</td><td>12.85</td><td>11.74</td><td>14.62</td><td>13.31</td><td>12.59</td><td>65.10</td></tr></table>

Dataset: SWE-QA Java subset — Strata (quantitative finance), Fineract (banking infrastructure), Shiro (security framework); 30 QA pairs total, same evaluation protocol as Python main experiments.

DeepRepoQA ranks first among open-source methods (+3.78 over OpenHands), outperforms Tongyi Lingma (+2.28), and trails Cursor by 1.81 pts — consistent with the Python setting.

## E Failure Mode Case Studies

## E.1 Handling of misleading retrievals

When SemanticSearch returns a misleading snippet, the Evaluation Agent in most cases recognises that the retrieved evidence is unhelpful and assigns a low score, which triggers backtracking via UCT. If the Evaluation Agent incorrectly assigns a high score, the trajectory continues and is likely to accumulate further unhelpful evidence, eventually triggering a delayed backtrack once the cumulative value signal falls below that of an unexpanded sibling. In a minority of cases this two-stage correction mechanism fails to recover, and the trajectory terminates on the wrong path — the failure modes catalogued below characterise when and why this occurs.

## E.2 Failure modes

From a manual inspection of the lowest-scoring trajectories across all question categories, we identify four recurring failure modes (F1–F4). Table E1 lists each mode together with the question categories it most affects and its underlying root cause; Sections E.5-E.8 then walk through a representative case for each mode.

## E.3 Success patterns

Three factors consistently distinguish high-scoring answers from failures. First, early retrieval precision: 100% of top-scoring cases (score ≥ 73) reached the correct file within the first two hops; cases requiring 15+ node expansions almost never recover from an incorrect initial retrieval. Second, symbol-level anchoring: 80% of top-scoring cases used FindClass or FindFunction with an explicit file\_pattern to pin the answer to a specific code span, preventing hallucination of adjacent code. Third, ViewCode immediately before Finish: re-reading the actual source one more time before drafting reduces paraphrase drift and is strongly correlated with high Correctness scores. Conversely, Clarity is high even when Correctness is low (Clarity mean 15.1 vs. Correctness mean 10.5 in the full evaluation), confirming that fluent prose does not imply factual correctness — Correctness and Completeness are the meaningful quality signals.

Table E1: Taxonomy of the four observed failure modes (F1–F4), the question categories they most affect, and their root causes.
<table><tr><td>Mode</td><td>Categories most affected</td><td>Root cause</td></tr><tr><td>F1 Wrong location</td><td>where/iden-loca, where/funct-loca</td><td>Semantic ≠ literal match; no verification step</td></tr><tr><td>F2 Adjacent concept</td><td>why/purpose, where/data-control-flow</td><td>Semantic ambiguity causes all branches to share wrong initial retrieval</td></tr><tr><td></td><td>F3 Keyword-only exploration why/design-rationale, how/algo-impl</td><td>No structural navigation fallback when se- mantic search stalls</td></tr><tr><td>F4 Open-ended design</td><td>how/system-design</td><td>Multiple valid answers; reference-matching rubric penalises divergent but correct de- signs</td></tr></table>

## E.4 Representative success case — why/purpose

Question: Why does the exception class raised when a computed variable shadows a base variable inherit from both theframework’s base exception class and the built-in Python exceptionfor naming conflicts?

Winning path (8 nodes, 4 steps): 1. SemanticSearch("exception class for computed variable shadowing a base var") 2. FindClass(ComputedVarShadowsBaseVarsError, file=reflex/utils/exceptions.py) 3. ViewCode(<that file>) 4. Finish

Why it worked: a natural-language description of a concrete entity mapped cleanly to a single class definition; SemanticSearch returned the right file on the first hop; ViewCode provided the docstring cited verbatim in the answer. The answer was generated from read code + docstring, not LLM inference, which is what kept Correctness high (17/20).

## E.5 Representative failure case F1 — where/iden-loca

Question: Where is the function that parses argument specifications called to build the argument list for an event handler?

Table E2: Failure case F1 (wrong location). The reference identifier parse\_args\_spec is replaced by a semantically similar but literally distinct function, with no verification step.
<table><tr><td>Reference</td><td></td><td>Candidate</td><td></td></tr><tr><td></td><td>Location parse_args_spec at reflex/event.py:2085</td><td>_extract_func_kwargs_as_ast_nodes pyi_generator.py</td><td>in</td></tr><tr><td>Scores</td><td></td><td>Correctness 2, Completeness 1, Relevance 2</td><td></td></tr></table>

Root cause: SemanticSearch returned a semantically similar but literally wrong function (\_extract\_func\_kwargs\_as\_ast\_nodes also parses kwargs but in a different file and context). The agent stopped after a single ViewCode and called Finish with high confidence, never attempting FindFunction(parse\_args\_spec) to verify the literal identifier. For where-type questions, the agent lacks a verification step that checks whether the question’s literal identifier appears in the retrieved code.

## E.6 Representative failure case F2 — why/purpose

Question: Why is the application instance and its module encapsulated in a structured container within theframework?

Root cause: the phrase “structured container” is semantically ambiguous — it applies to both AppInfo (a data container) and AppWrap (a UI wrapper component). SemanticSearch surfaced AppWrap because that token appears more frequently in documentation. The agent committed to this interpretation across

Table E3: Failure case F2 (adjacent concept). The ambiguous phrase “structured container” steers retrieval toward AppWrap (a UI wrapper) instead of the reference target AppInfo (a data container).
<table><tr><td></td><td>Reference</td><td>Candidate</td></tr><tr><td>Target Scores</td><td>AppInfo (a NamedTuple holding App + its module)</td><td>AppWrap (a component wrapping the root rendered tree) Correctness 4, Completeness 3, Relevance 5</td></tr></table>

all 17 MCTS nodes without reconsidering, illustrating how semantic ambiguity in the question can cause all branches to share the same incorrect initial retrieval, making backtracking ineffective.

## E.7 Representative failure case F3 — why/design-rationale

Question: Why does the design constraint prioritizeframework infrastructure hooks before other hook categories in the code generation process?

What went wrong: over 7 steps the agent issued repeated SemanticSearch and FindCodeSnippet calls with progressively varied keywords (“hook category sort key”, “order compiler code generation”, etc.), hitting the 20-node cap at 332 s. It never discovered the HookPosition enum or the actual ordering function. The agent had no fallback when keyword search stalled: it did not inspect directory structure, did not follow imports, and did not switch to structural navigation — it only varied keywords.

## E.8 Representative failure case F4 — how/system-design

Question: How can the collapsible content panel component be redesigned to support runtime theming changes without breaking existing usage?

Table E4: Failure case F4 (open-ended design). The agent produces a technically sound alternative redesign, but the reference-based LLM judge penalises divergence from the single prescribed solution.
<table><tr><td></td><td>Reference</td><td>Candidate</td></tr><tr><td></td><td>Proposed solution Specific approach involving theme token propaga- tion via CSS variables</td><td>Alternative design using add_style() and inher- ited data attributes</td></tr><tr><td>Scores</td><td></td><td>Correctness 7, Completeness 8, Relevance 9</td></tr></table>

Root cause: this is a redesign question where the reference prescribes one specific engineering solution. The agent produced a plausible and technically sound alternative design, but the LLM judge penalised the divergence from the reference’s specific proposal. This is partly an evaluation artefact — LLM-as-judge compares against a single reference answer — and partly an inherent limitation of open-ended design questions: code retrieval provides little guidance on which of many valid solutions is “correct”. Unlike localization or explanation questions, no amount of additional retrieval steps would have led the agent to the reference’s specific solution.

## F Prompts

Prompt Template for Planning   
You are an autonomous AI assistant with superior skills in answering questions of a   
repository.   
You need to answer the question exactly based on the information you can get from the   
available functions.   
As you're working autonomously, you cannot communicate with the user but must rely on   
information you can get from the available functions.   
# Action and ReAct Guidelines   
1. Analysis First   
- Review all previous actions and their observations   
- Understand what has been done and what information you have   
2. Document Your Thoughts   
- ALWAYS write your reasoning in <thoughts> tags before any action   
- Explain what you learned from previous observations   
- Justify why you're choosing the next action

## - Describe what you expect to learn/achieve

## 3. Single Action Execution

\- Run ONLY ONE action at a time

\- Choose from the available functions

\- Never try to execute multiple actions at once

## 4. Wait and Observe

\- After executing an action, STOP

\- Wait for the observation (result) to be returned

\- Do not plan or execute any further actions until you receive the observation

## # Workflow Overview

## 1. Understand the Question

\* Review the Question: Carefully read the question provided in <task>

\* Identify Needed Information: Analyze the question to determine what parts of the

codebase you need to understand

to explore to find a complete answer

## 2. Locate Relevant Code

\* Primary Method - Search Functions:

\- FindClass - Search for class definitions by class name

\- FindFunction - Search for function definitions by function name

\- FindCodeSnippet - Search for specific code patterns or text

\- SemanticSearch - Search code by semantic meaning or natural language description

\* Secondary Method - ViewCode: Only use when you need to see:

\- Additional context not returned by searches

\- Specific line ranges from search results

\- Code referenced in other parts of the codebase

## 3. Gather Complete Information

\- Continue searching and viewing code until you have all necessary information

\- Investigate all relevant parts of the codebase

\- Look for related functionality that might be important

## 4. Analyze and Formulate Answer

\- Review all gathered information

\- Organize findings to create a complete and accurate answer

\- Ensure the answer is based on actual code, not assumptions

## 5. Complete Task

\- Use Finish you have sufficient information to provide a complete and accurate answer

\- Reference specific parts of the code in your final answer

\- Make sure the answer addresses all aspects of the question

## # Important Guidelines

\- Focus on the Specific Question

\- Answer exactly as asked, based on the code in the repository

\- Provide complete and accurate information

\- Do not assume code you haven't seen

## - Code Context and Information

\- Base answers only on code you can see through searches and ViewCode

\- Use ViewCode to examine more code if needed

\- Reference specific code sections for clarity

## - Task Completion

\- Finish only after gathering sufficient information

\- Cite specific evidence from the code

\- Ensure all relevant code has been explored

## - State Management

\- Keep detailed records of viewed code and actions taken

\- Check history before performing a new action to avoid repetition

\- Use gathered information to inform next steps

## # Additional Notes

\- Think Step by Step

\- Document reasoning and thought process in <thoughts>

\- Build upon previous steps without unnecessary repetition

## - Never Guess

\- Do not guess line numbers or code content

\- Use ViewCode to examine code when needed

## # Examples

Here are some examples of how to use the available actions:

## \*\*Example 1:

it handles transactions

<thoughts>To examine how the DatabaseManager class handles transactions, we need to   
locate its implementation in the codebase.</thoughts>   
{"tool": "FindClass", "file\_pattern": null, "class\_name": "DatabaseManager"}   
\*\*Example 2:   
Task: Find the calculate\_interest function in our financial module to review its logic   
<thoughts>To review the logic of the calculate\_interest function, we need to locate   
its implementation in the financial module.</thoughts>   
{"tool": "FindFunction", "file\_pattern": "financial/\*\*/\*.py", "function\_name":   
"calculate\_interest", "class\_name": null}   
\*Note: This is a condensed version for display purposes. The actual prompt contains   
complete examples for all action tools.\*

## Prompt Template for Perceiving and Generating Feedback

```markdown
You are a feedback agent that guides an AI assistant's next action.
## CRITICAL: ACTION AGENT LIMITATIONS
The action agent receiving your feedback:
- CANNOT see the search tree
- Has NO CONTEXT about node relationships
- Only knows about actions in its direct trajectory
- Cannot understand references to nodes without proper context
- Is at a new node that has NO ACTION YET --- it needs your guidance for what to do next
## REQUIRED FEEDBACK STRUCTURE
### 1. CURRENT NODE CONTEXT
You must start by describing:
- **Position in tree:** `"You are at Node X, which is [position relative to root]"`
- **Current state:** `"Your node is currently empty and awaiting your first action"`
- **Parent context:** `"Your parent node (Node Y) [describe what parent did]"`
- **Relationship to solutions:** `"There are [N] terminal nodes in [relationship]
branches."
> **Note:** The current node is ALWAYS empty and awaiting its first action --- never
describe it as having done something already.
-
## CORRECT EXAMPLES
**Current Node Context Example:**
"You are at Node 8, which is your first action from the root.
Your node is currently empty and awaiting your first action.
Your parent (Node 1) performed a FindCodeSnippet action that didn't add new context.
There are three terminal nodes in parallel branches (Nodes 7, 9, and 14) that have
reached finished states with different approaches."
## INCORRECT EXAMPLES --- DO NOT USE
`"Node 8 is empty and expandable"`
`"The current node needs to explore improvements"`
- `"We should validate the existing solution"`
- Any description implying the current node has already taken an action
---
## INPUT STRUCTURE
1. **Tree Visualization:** ASCII representation showing:
- Node IDs and relationships
- Action types at each node
- Key outcomes and observations
2. **Original Task:** The problem to solve
3. **Message History:** Chain of executed actions leading to current state
4. **Tree Structure:**
- **Parent Node:** Your current starting point, the last successfully executed
action
- **Current Node:** Your branch from the parent, waiting for your next action
- **Sibling Nodes:** Other independent solution attempts branching from the same
parent
(These are from different trajectories and have not happened in your current path)
```

5. \*\*Alternative Node Information:\*\*

\- Proposed actions and parameters

\- Outcomes (from separate, independent trajectories)

\- Warning flags for previously attempted approaches

## ## YOUR TASK

1. \*\*Analyze the situation:\*\*

\- Start with current node context (position, state, parent, solutions)

\- Consider sibling attempts (alternative universes)

\- Learn from outcomes to avoid repeating unsuccessful approaches

\- Contextualize feedback based on tree structure

\- Always explain node relationships and attempts

\- Inform about alternative approaches tried (files, tests, git diffs)

2. \*\*Suggest next action:\*\*

\- Clear, actionable guidance focusing on code query actions

\- Based on lessons from other attempts

\- Avoid repeating failed approaches

3. \*\*Optionally suggest node to expand:\*\*

\- Explain why this node is promising

\- Leave as null if no strong preference

> Remember: Focus on code query actions that help the agent better understand the

codebase. Always provide proper context since the agent cannot see the tree.

\*\*Important:\*\* Please provide the answer in JSON format.

## Prompt Template for Identifying Useful Code Spans

You are an autonomous AI assistant tasked with identifying relevant code in a codebase. Your goal is to select key code sections from the search results that are most relevant to the search request.

\## Context

The previous messages will contain:

1. A search request from an AI assistant

2. Search results containing various code sections with their line numbers

\## Your Task

\### 1. Understand the Search Request

\- Analyze the previous search request to understand what elements are being looked for

\- Identify key elements such as \*\*functions, variables, classes, or patterns\*\* that are relevant

\### 2. Evaluate Search Results

\- Examine each code section in the search results for alignment with the search request

\- Assess the \*\*relevance and importance\*\* of each code section

\- Consider the \*\*complete context\*\* of code sections

\### 3. Respond with the Identify Action

\- Select and respond with the \*\*code sections\*\* that best match the search request

\- Provide your \*\*analysis in the thoughts field\*\*

\- List the relevant \*\*file paths\*\* with \*\*start and end line numbers\*\* in the

\`identified\_spans\` field

## Prompt Template for Node Evaluation

Your role is to evaluate the \*\*last executed action\*\* of the search tree that our AI agents are traversing, to help determine the best trajectory to solve a programming issue. The agent is responsible for identifying and modifying the correct file(s) in response to the problem statement.

\*\*Important:\*\* While line numbers may be referenced in the initial problem description, they can shift as changes are made to the file. Focus on whether the agent is modifying the correct logical parts of the code, rather than strictly matching the initially mentioned line numbers. What matters is that the right section of code is being modified, even if its current line number differs from what was originally specified.

\## Task

At this stage, the agent is still working on the solution. Your task is twofold:

1. \*\*Evaluation\*\*: Assess whether the change done by the \*\*last executed action\*\* is appropriate for addressing the problem and whether the agent is on the right path to resolving the issue. Verify that the correct sections of code are being modified, regardless of their current line numbers.

2. \*\*Alternative Feedback\*\*: Independently of your evaluation, provide guidance for an alternative problem-solving branch. This ensures parallel exploration of different solution paths.

## ## Evaluation Criteria

\- \*\*Exploratory Actions:\*\* Initial searches and information-gathering steps are essential and should not be heavily penalized if they don't yield immediate results. - \*\*Appropriateness of Action:\*\* Is the action logical given the agent's current knowledge and early stage of problem-solving?

\- \*\*Query Relevance:\*\* Is the search query or parameters well-defined and likely to find relevant code?

\- \*\*Search Scope Appropriateness:\*\* Do the file patterns and class/function names narrow down the search effectively?

\- \*\*Relevance of Search Results:\*\* Are the search results directly related to the problem and useful for progress?

\- \*\*Size of Search Results:\*\* Is the code context appropriately sized (not too large or too small)?

\- \*\*Category Appropriateness:\*\* Does the category (implementation or test) align with the search intent?

## ## Reward Scale

Assign an integer between 0 and 100:

\- 90-100: Excellent action; parameters are well-defined and results are highly

\- 75-89: Good action; parameters are reasonable and results are relevant.

\- 25-74: Action has issues or yields few/no relevant results.

\- 0-24: Counterproductive; results are irrelevant or excessively large.

## ## Feedback Structure

\- \*\*Explanation:\*\* Provide detailed reasoning for your decision, focusing on the \*\*last executed action\*\*, its relation to previous actions, and its impact.

\- \*\*Feedback to Alternative Branch:\*\* Suggest conceptual alternative approaches without actual code, avoiding actions that would replicate previous outcomes.

\- \*\*Reward:\*\* Assign a single integer between 0 and 100 based on confidence in correctness and likelihood of solving the issue.

## ## Available Actions

\### FindClass

Use this when you know the exact class name to find.

\- Finding class implementations: \`class\_name="UserRepository"\`

\- Locating test classes: \`class\_name="TestUserAuthentication"\`

\- Finding base classes: \`class\_name="BaseController"\`

\- Classes in specific modules: \`class\_name="Config", file\_pattern="src/config/\*.py"\`

## ### FindFunction

Use when you know the exact function or method name.

\- Finding test cases: \`function\_name="test\_user\_login"

\- Locating implementations: \`function\_name="process\_payment"\`

\- All methods with a name: \`function\_name="validate"\`

\- Specific class method: \`function\_name="save", class\_name="UserRepository"\`

## ### FindCodeSnippet

Use when you know the exact code snippet.

\- Finds constants: \`code\_snippet="MAX\_RETRIES = 3"\`

\- Finds decorators: \`code\_snippet="@retry(max\_attempts=3)"\`

\- Finds imports: \`code\_snippet="from datetime import datetime"\`

\- Configuration patterns: \`code\_snippet="DEBUG = os.getenv('DEBUG', False)"

> Note: If you don't know the exact code, use \*\*SemanticSearch\*\*.

## ### SemanticSearch

Use when you don't know exact names but want related functionality.

\- Functionality by description: \`query="code that handles password hashing"\`

\- Related test cases: \`query="tests for user registration", category="test"\`

\- Implementations: \`query="database connection pooling", category="implementation"\`

\- Error handling patterns: \`query="error handling for API requests"\`

> Flexible for unknown names, discovering related code, or exploring features.

## ### ViewCode

View the code in a file or a specific span.

```markdown
### Finish
Indicate that the generated code answer is accurate and complete for the user's query.
```

## Prompt for LLM-as-Judge

You are a STRICT and RIGOROUS evaluator. You must rate the candidate answer STRICTLY against the reference answer.   
Be CONSERVATIVE with high scores - only award high scores (16-20) when the candidate answer is truly excellent and   
closely matches the reference answer in quality and content.   
CRITICAL EVALUATION PRINCIPLES:   
1. Compare the candidate answer DIRECTLY with the reference answer point by point   
2. Any deviation, omission, or inaccuracy should result in score reduction   
3. High scores (16-20) should be RARE - reserve them only for answers that are nearly perfect   
4. Be strict about factual accuracy - even minor errors should lower the correctness score   
5. Missing key points from the reference answer should significantly reduce completeness score   
6. Vague or imprecise language should lower clarity scores   
7. When in doubt between two score ranges, choose the LOWER one   
Evaluation Criteria and Scoring Guidelines (each scored 1 to 20, total score 100):   
1. Correctness (STRICT - penalize any inaccuracies):   
20 --- ONLY if completely correct with ALL core points and details accurate, matching reference answer   
precisely   
16-19 --- Mostly correct but must have only TRIVIAL inaccuracies; any noticeable error reduces to 15 or below   
12-15 --- Partially correct; has some errors or omissions that affect understanding; main points may be   
accurate but details are wrong   
8-11 --- Several errors or ambiguities that significantly affect understanding of core information   
4-7 --- Many errors; misleading or fails to convey key information correctly   
1-3 --- Serious errors; completely wrong or misleading   
2. Completeness (STRICT - penalize missing information):   
20 --- ONLY if covers ALL key points from reference answer without ANY omission; must match reference in depth   
16-19 --- Covers most key points but missing some non-trivial information; minor omissions are acceptable   
12-15 --- Missing several important key points; content is noticeably incomplete compared to reference   
8-11 --- Important information largely missing; content is one-sided or superficial   
4-7 --- Covers very little relevant information; seriously incomplete   
1-3 --- Covers almost no relevant information; completely incomplete   
3. Relevance (STRICT - penalize off-topic content):   
20 --- ONLY if content is fully focused on question topic with NO irrelevant information whatsoever   
16-19 --- Mostly focused but may have minor peripheral information; any significant off-topic content reduces   
score   
12-15 --- Generally on topic but contains some off-topic content that detracts from answer   
8-11 --- Topic not sufficiently focused; contains considerable off-topic or tangential content   
4-7 --- Content deviates from topic; includes excessive irrelevant information   
1-3 --- Majority of content irrelevant to the question   
4. Clarity (STRICT - penalize unclear expression):   
20 --- ONLY if language is exceptionally fluent, clear, and precise; very easy to understand without any   
ambiguity   
16-19 --- Mostly fluent and clear but may have minor unclear points; any significant ambiguity reduces score   
12-15 --- Generally clear but some expressions are unclear or not concise; may require effort to understand   
8-11 --- Expression somewhat awkward; has ambiguity or lacks fluency that hinders understanding   
4-7 --- Language obscure; sentences are not smooth; significantly hinders understanding   
1-3 --- Expression confusing; very difficult to understand   
5. Reasoning (STRICT - penalize weak logic):   
20 --- ONLY if reasoning is exceptionally clear, logical, and well-structured; argumentation is excellent and   
matches reference quality   
16-19 --- Reasoning is clear and logical with solid argumentation; minor logical gaps may exist   
12-15 --- Reasoning generally reasonable but has noticeable logical jumps or organization issues   
8-11 --- Reasoning is average; has logical jumps or organization problems that affect understanding   
4-7 --- Reasoning unclear; lacks logical order; difficult to follow   
1-3 --- No clear reasoning; logic is chaotic   
INPUT:   
Question:{question}   
Reference Answer:{reference}   
Candidate Answer:{candidate}   
OUTPUT:   
Please output ONLY a JSON object with 5 integer fields in the range [1,20], corresponding   
to the evaluation scores:   
{{   
"correctness": <1-20>,   
"completeness": <1-20>,   
"relevance": <1-20>,   
"clarity": <1-20>,   
"reasoning": <1-20>   
}}

SCORING INSTRUCTIONS:

\- Read the reference answer carefully and identify ALL key points, details, and structure

\- Compare the candidate answer systematically against the reference answer

\- For each criterion, start with a conservative score and only increase if the candidate truly deserves it

\- If the candidate answer is significantly shorter, less detailed, or less precise than the reference, reduce scores accordingly

\- If the candidate answer contains information not in the reference (unless it's clearly relevant and accurate),

consider reducing relevance score

\- When scoring, ask yourself: "Does this can

didate answer match the quality and completeness of the reference

answer?" If not, reduce scores

\- Average or mediocre answers should receive scores in the 8-15 range, not higher

\- Only truly excellent answers that closely match the reference should receive 16-20 scores

No explanation, no extra text, no formatting other than valid JSON. Be strict and conservative with your scores.