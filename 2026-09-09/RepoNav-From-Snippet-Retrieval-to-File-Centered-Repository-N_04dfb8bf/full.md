# RepoNav: From Snippet Retrieval to File-Centered Repository Navigation for Code Agents

Hongzheng Chai<sup>1</sup> Jiakun Li<sup>1</sup> Hongyue Yu<sup>2</sup> Yuan Yuan<sup>1,3,4</sup>\* Hongzheng Chai1 Jiakun Li1 Hongyue Yu2 Yuan Yuan1,3,4\* <sup>1</sup>School of Computer Science and Engineering, Beihang University <sup>2</sup>National College for Excellent Engineers, Beihang University <sup>3</sup>Hangzhou Innovation Institute, Beihang University <sup>4</sup>Qingdao Research Institute, Beihang University chaihongzheng@buaa.edu.cn, yuan21@buaa.edu.cn

## Abstract

Solving repository-level code tasks requires LLM-based agents to use code search tools to navigate large codebases and identify a small set of relevant files and functions. However, current retrieval tools typically return flat lists of isolated code snippets: such lists can surface relevant files, but provide insufficient structure for agents to distinguish the target function from semantically similar alternatives in the same file. We introduce RepoNav, a lightweight post-retrieval interface that reorganizes retrieved snippets into a file-centered navigation scaffold. By presenting compact structural cues and candidate targets, this scaffold guides on-demand file-structure browsing, helping agents compare sibling symbols before selecting a target function. Across diverse models on LocBench, RepoNav improves function-level localization and narrows the fileto-function gap. Controlled ablations demonstrate that these gains come from structured evidence organization rather than simply exposing additional file structure, and the approach also improves performance on a repository-level question-answering benchmark. <sup>1</sup>

## 1 Introduction

Large language models (LLMs) have driven rapid progress in autonomous software engineering. In repository-scale tasks such as issue resolution and bug localization, agents must operate across two levels of granularity: identifying the relevant files and pinpointing the specific functions within them. Modern code agents equipped with retrieval or agentic search tools (Zhang et al., 2023; Wang et al., 2025b; Yang et al., 2024) can often surface relevant files, yet function-level localization still lags significantly behind. We analyze this behavior using a common mini-SWE-agent scaffold (Yang et al., 2024), keeping the underlying agent loop fixed across exploration interfaces.

![](images/8aa79cec7aa8c4f3cba706d8049294247829293afe2e8cc463b2679042c73a3f.jpg)  
Figure 1: Failure taxonomy for function-level misses from interactive agent executions. Percentages are computed over misses within each LLM backbone and setting. A substantial fraction of misses occurs after the agent has reached the correct file, indicating within-file navigation failures. Percentages may not sum to 100 due to rounding.

Why do agents that reach the correct file still miss the target function? Our analysis reveals premature anchoring: agents often commit to a salient nearby symbol before inspecting sibling definitions or file-structure views. Flat snippet retrieval reinforces this behavior by presenting relevant chunks and distractors as isolated evidence, with limited cues about what else the file contains. As Figure 1 shows, Correct File, Wrong Function accounts for 44–58% of function-level misses across agents instantiated with different LLM backbones, confirming within-file navigation as a central bottleneck in function-level localization.

To mitigate this bottleneck under a fixed retrieval substrate, we introduce RepoNav, a lightweight post-retrieval interface for repository navigation.

Here, lightweight refers to low incremental infrastructure and deployment overhead: RepoNav reuses the existing dense retrieval substrate and requires no additional persistent structural index or repository-wide graph. RepoNav reorganizes the retrieved evidence into a file-centered navigation scaffold. The scaffold exposes compact structural cues, candidate targets, and continuation hints, and supports on-demand file-structure browsing through list\_symbols. This design separates file discovery from within-file verification: retrieval proposes candidate files, while RepoNav helps the agent decide which internal symbols to inspect next. Throughout this paper, we use symbol to denote a named structural code unit, such as a function, method, or class. Rather than asking agents to read more code upfront, RepoNav makes candidate files structurally expandable and encourages verification of sibling symbols before commitment. Unlike heavyweight graph-based systems (Ouyang et al., 2025; Liu et al., 2025; Chen et al., 2025) that rely on repository-wide graph construction or static-analysis infrastructure, RepoNav is an agent-facing interface layer built on top of existing snippet retrieval outputs.

Across seven models on LocBench (Chen et al., 2025), RepoNav improves function-level localization and reduces the gap between file discovery and function discovery. A tool-matched Snippet+ListSym baseline shows that file-structure browsing is useful, while RepoNav further improves over this baseline by making such browsing more actionable through file-centered organization. Results on SWE-QA-Bench (Peng et al., 2026) show that RepoNav also improves repository-level question answering beyond explicit localization. Our contributions are:

1. An empirical diagnosis of the file-to-function gap. We show that a large fraction of functionlevel failures occur after the agent has already reached the correct file, identifying within-file navigation as a major bottleneck in repositoryscale code localization.

2. A structurally expandable, file-centered navigation interface. RepoNav reorganizes flat snippet-style outputs into file-centered scaffolds while keeping the underlying dense index and raw retrieval scores fixed. It integrates list\_symbols as an on-demand filestructure browsing tool, enabling agents to inspect lightweight file skeletons and compare sibling symbols without reading full file bodies.

3. Behavioral and tool-matched evidence for scaffolded navigation. A Snippet+ListSym baseline controls for access to file-structure browsing, and controlled ablations show that RepoNav provides additional gains by making structural evidence more actionable and efficient for agent exploration.

## 2 Background and Related Work

## 2.1 Code Agents and Exploration Interfaces

SWE-bench (Jimenez et al., 2024) has become a standard benchmark for autonomous code agents (Zhang et al., 2024; Yang et al., 2024; Wang et al., 2025a; Antoniades et al., 2025; Xia et al., 2025), which navigate repositories through interleaved reasoning and tool use (Yao et al., 2023). These agents typically interact with repositories through bash-style exploration or snippetstyle search interfaces. Retrieval-augmented generation (Lewis et al., 2020) has been widely adopted for injecting external knowledge into language models; in the code domain, retrieval-based methods such as RepoCoder (Zhang et al., 2023) and CodeRAG-Bench (Wang et al., 2025b) extend this paradigm to repository-level tasks. However, snippet-search interfaces usually present retrieved chunks as flat, chunk-centered lists, while bashstyle interfaces expose the raw repository but require the agent to infer structure from command outputs. In both cases, the interface provides limited support for turning retrieved evidence into navigable action choices. This limitation is especially problematic for function-level localization, where the agent must compare multiple candidate symbols after reaching a relevant file rather than merely identify the file itself.

## 2.2 The File-to-Function Gap

A natural assumption is that function-level localization failures mainly arise because the correct file was never surfaced. Our failure analysis shows otherwise. We categorize each function-level miss into three types: Wrong File (the predicted target lies outside the gold file), Correct File, Wrong Function (the agent reaches the correct file but selects the wrong function), and Same Name, Wrong Qualifier (the prediction matches the local symbol name but not the fully qualified target). As Figure 1 shows, a substantial share of misses occurs after the agent has already reached the correct file: Correct File, Wrong Function accounts for 44–58% of all misses. This reveals a file-to-function localization gap: reaching the correct file does not reliably translate into identifying the correct function.

![](images/93efa9ddb3f1f238b75c1ecc6d7798b1b4fa16255f71938a3137f1be7b9f765d.jpg)  
Figure 2: A real GitPython case illustrating function-level divergence under different evidence organization. Top: snippet-style retrieval returns helper-heavy flat evidence and misses Git.execute. Bottom: RepoNav organizes the candidate file into a file-centered scaffold and prompts on-demand file-structure browsing with list\_symbols, making Git.execute visible as a sibling candidate for targeted verification.

Premature anchoring. This asymmetry gives rise to a recurring failure mode: the agent commits to the most salient nearby symbol before inspecting sibling or structurally adjacent candidates. Typical anchors include public wrappers, request handlers, or entry methods, as illustrated in Figure 2 (see Appendix F for further case studies). This behavior is reminiscent of the “lost in the middle” phenomenon in long-context models (Liu et al., 2024a), where information position affects utilization; here, the agent fixates on the most prominent symbol rather than systematically comparing alternatives. The interface does not encourage continued exploration, motivating a navigation-oriented view: postretrieval interfaces should organize evidence into actionable choices that make continued exploration easier than premature commitment.

Positioning. Recent efforts have scaled structural representations to the repository level. Repo-Graph (Ouyang et al., 2025) and CodexGraph (Liu et al., 2025) map entire codebases into graph databases; LocAgent (Chen et al., 2025) constructs directed code graphs for multi-hop traversal; LingmaAgent (Ma et al., 2024) builds repository-level knowledge representations with Monte Carlo tree search for exploration; and GraphCoder (Liu et al., 2024b) uses code context graphs for retrievalaugmented completion. These methods rely on repository-wide graph construction and specialized traversal or query mechanisms, which introduce additional infrastructure and interaction complexity for agent systems. In the broader NLP setting, StructRAG (Li et al., 2025) converts retrieved documents into task-appropriate structured formats to aid reasoning; RepoNav shares this motivation of post-retrieval restructuring but targets code agents and operates as a lightweight interface layer rather than a general document transformation framework. Rather than precomputing a global graph, RepoNav dynamically extracts local structural cues from only the top-ranked retrieved files and presents them as an immediately consumable text-based scaffold.

Research questions. The preceding analysis motivates three questions: (RQ1) Does the file-tofunction gap exist consistently across models, and does a navigation-oriented interface reduce it? (RQ2) Is the behavioral shift driven by the volume of structural information or by how that information is organized? (RQ3) Does the benefit transfer beyond localization to repository-level question answering?

## 3 RepoNav: A Navigation Interface for Repository Exploration

RepoNav is a post-retrieval interface layer rather than a new retriever. It does not modify the raw dense chunk retrieval results or similarity scores used by the snippet-search baseline. Instead, it deterministically aggregates retrieved chunks into file-level candidates and changes how the retrieved evidence is presented to the agent. RepoNav turns raw retrieval hits into cues for comparing plausible symbols before commitment.

## 3.1 Interface Design

RepoNav follows three design principles. First, it aggregates chunk-level evidence into candidate files, since agents ultimately inspect, reason over, or modify files rather than isolated chunks. Second, it exposes compact file-internal structure, giving the agent enough context to compare nearby symbols without dumping the full file. Third, it adds explicit continuation cues, so that retrieved evidence is treated as a starting point for targeted inspection rather than as a terminal answer.

Concretely, RepoNav serializes the top-ranked files as a plain-text, indentation-based tree. Each file entry contains three blocks.

[ANCHORS]: Entry points. After dense retrieval results are aggregated into candidate files, RepoNav identifies anchor symbols within those files using a lightweight lexical match between symbol names and the retrieval query issued by the agent. These anchors provide query-relevant entry points into each file. When no symbol name matches the query, RepoNav falls back on the structural sketch described below. Each anchor is annotated with its line span and compact same-file call context, giving the agent a grounded starting point for inspection. In our implementation, anchors are capped at two per file.

[GLIMPSE]: Structural sketch. To prevent fixation on anchors, RepoNav exposes up to three non-anchor symbols from the same file, or up to five when the top-ranked file has no anchor match. These symbols provide a compact sketch of what else the file contains, such as additional classes, functions, or methods that may not directly match the query but are structurally relevant. Rather than dumping full function bodies or long signatures, [GLIMPSE] presents schematic entries, allowing the agent to notice alternative candidate symbols with minimal context cost.

[CANDIDATE\_TARGETS]: Actionable target shortlist. This block turns the structural sketch into a compact set of candidate inspection targets. It summarizes anchors, selected glimpse symbols, and lightweight same-file call context without adding new retrieval evidence or full function bodies. The shortlist is capped at four entries and paired with fixed continuation hints, encouraging the agent to inspect sibling symbols before committing. The call context serves only as a navigation cue rather than sound static analysis.

list\_symbols: File-structure browsing. RepoNav exposes list\_symbols as an on-demand file-structure browsing tool. Given a file path, it returns a lightweight file skeleton, including imports, classes, functions, methods, line ranges, and optional signatures, but not full function bodies. It does not perform semantic search or rank target functions. Instead, it lets the agent inspect the editable objects inside a candidate file and compare sibling symbols before selecting a target function.

The scaffold itself remains plain text and requires no graph query language. It can therefore be consumed by standard text-based code agents while still exposing enough structure to support targeted navigation. The full serialization budgets, ordering rules, and a worked output example are provided in Appendix G (Figure 8).

## 3.2 Implementation

RepoNav is implemented as a two-stage postretrieval pipeline. Both stages require minimal computation and no offline preprocessing beyond the standard dense chunk index used by the snippet baseline. The specific retrieval model and indexing configuration are detailed in Section 4.1.

Stage 1: Evidence aggregation. Given an agent retrieval query, the dense retrieval backend returns top-ranked code chunks and their associated retrieval scores. RepoNav aggregates these chunklevel scores into a file-level ranking using a hybrid scoring rule. For each candidate file $f _ { : }$ , let $C _ { f }$ denote the retrieved chunks from $f ,$ with their similarity scores sorted as $s _ { f , ( 1 ) } \geq s _ { f , ( 2 ) } \geq . . .$ . We aggregate the top $m _ { f } =$ min $( m , | C _ { f } | )$ scores as:

$$
\mathrm { S c o r e } ( f ) = \alpha s _ { f , ( 1 ) } + ( 1 - \alpha ) \frac { 1 } { m _ { f } } \sum _ { i = 1 } ^ { m _ { f } } s _ { f , ( i ) } .
$$

We set $m { = } 3$ in all experiments. Files with fewer than m retrieved chunks are averaged over their available chunks. The first term captures peak evidence strength, while the second rewards files supported by multiple high-scoring regions rather than a single spurious match. A parameter sensitivity analysis in Appendix A shows that performance is stable across a wide range of α values, with ranking-oriented metrics peaking near $\alpha { = } 0 . 5$ and recall metrics plateauing for $\alpha \ge 0 . 7$ . Since α only affects the file ranking before agent interaction, we select its value on held-out development splits using offline retrieval metrics and then keep it fixed for all downstream experiments.

Stage 2: On-the-fly structural extraction. RepoNav dynamically parses only the top-k files from Stage 1, with $k { = } 5$ fixed across experiments by default. The initial scaffold auto-expands full threeblock entries for the top three candidate files, while the remaining parsed files remain available for ondemand structure browsing. For Python files, the built-in ast module extracts function and class definitions, import statements, and a lightweight same-file call neighborhood. The call neighborhood connects functions only when a call expression can be matched to another function defined in the same file; it does not attempt sound static analysis, cross-file dependency recovery, or interprocedural call-graph construction. When call information is unavailable, RepoNav falls back to file-structure skeletons without caller and callee cues. These extracted cues are assembled into the three-block interface described in Section 3.1.

Figure 3 summarizes the two-stage post-retrieval pipeline. The process is deterministic and local: RepoNav aggregates retrieved chunks into candidate files and extracts structure only from the topranked files, avoiding repository-wide graph construction. This design allows evidence from multiple retrieved regions to jointly support a file while retaining multiple candidate files for subsequent within-file inspection and cross-file pivoting.

## 4 Experimental Evaluation

We now evaluate the three research questions introduced in Section 2: whether the file-to-function localization gap holds consistently across models and can be reduced by a navigation-oriented interface (RQ1), whether the observed behavioral shift is driven by the volume of structural information or by how that information is organized (RQ2), and whether the benefit transfers beyond localization to repository-level question answering (RQ3).

![](images/55208f09fca00aabe45322b1694153cf88208cbefc7b7c9ec2a166ffce0a17b6.jpg)  
Figure 3: Implementation overview of RepoNav. The retrieval backend is shared with the snippet-search baseline; RepoNav changes the post-retrieval organization of evidence by aggregating chunks into candidate files and extracting lightweight file structure.

## 4.1 Experimental Setup

Benchmark. We evaluate on LocBench (Chen et al., 2025), a repository-level code localization benchmark with 560 Python instances, each pairing a natural language issue with gold files and functions from the corresponding patch.

Agent settings. All experiments use a common mini-SWE-agent scaffold (Yang et al., 2024) with a fixed agent loop, environment, and interaction budget. We compare four interfaces: Bash (shell only), Snippet Search (adding a dense search tool returning flat chunk lists), Snippet+ListSym (adding the same list\_symbols tool used by RepoNav to the flat snippet interface, controlling for file-structure browsing access), and RepoNav (reorganizing the same dense retrieval substrate into a file-centered scaffold with list\_symbols access).

Retrieval settings. All three retrieval-based settings share the same chunk index, embedding model, retrieval backend, and raw chunk-level similarity scores. Following Chen et al. (2025), each function is embedded as a single chunk using CodeRankEmbed (Suresh et al., 2025). The main experiments retrieve 80 raw chunks per query before file-level aggregation. CodeRankEmbed is the primary retriever used in the full experiments. We additionally conduct a fixed 100-instance pilot with UniXcoder (Guo et al., 2022) to examine compatibility with a second embedding model. Retrievaldepth sensitivity and alternative-embedder results are reported in Appendix B.

<table><tr><td></td><td colspan="4">File Acc@5</td><td colspan="4">Module Acc@5</td><td colspan="4">Function Acc@5</td><td colspan="4">Function Rec@10</td></tr><tr><td>Model</td><td>B</td><td>S</td><td>S+L</td><td>R</td><td>B</td><td>S</td><td>S+L</td><td>R</td><td>B</td><td>S</td><td>S+L</td><td>R</td><td>B</td><td>S</td><td>S+L</td><td>R</td></tr><tr><td>Qwen2.5-72B</td><td>0.589</td><td>0.680</td><td>0.673</td><td>0.698</td><td>0.396</td><td>0.461</td><td>0.562</td><td>0.570</td><td>0.254</td><td>0.300</td><td>0.384</td><td>0.434</td><td>0.307</td><td>0.377</td><td>0.490</td><td>0.549</td></tr><tr><td>GPT-OSS-120B</td><td>0.705</td><td>0.759</td><td>0.759</td><td>0.770</td><td>0.500</td><td>0.613</td><td>0.630</td><td>0.650</td><td>0.339</td><td>0.464</td><td>0.488</td><td>0.527</td><td>0.417</td><td>0.571</td><td>0.586</td><td>0.618</td></tr><tr><td>Qwen3-Next-80B</td><td>0.734</td><td>0.732</td><td>0.733</td><td>0.739</td><td>0.523</td><td>0.568</td><td>0.589</td><td>0.632</td><td>0.339</td><td>0.411</td><td>0.454</td><td>0.498</td><td>0.410</td><td>0.517</td><td>0.564</td><td>0.622</td></tr><tr><td>Qwen3-Coder-30B</td><td>0.700</td><td>0.693</td><td>0.707</td><td>0.727</td><td>0.564</td><td>0.584</td><td>0.598</td><td>0.629</td><td>0.420</td><td>0.459</td><td>0.476</td><td>0.516</td><td>0.498</td><td>0.551</td><td>0.572</td><td>0.618</td></tr><tr><td>MiniMax-M2.5</td><td>0.760</td><td>0.764</td><td>0.780</td><td>0.803</td><td>0.660</td><td>0.671</td><td>0.671</td><td>0.691</td><td>0.525</td><td>0.532</td><td>0.534</td><td>0.601</td><td>0.633</td><td>0.641</td><td>0.652</td><td>0.708</td></tr><tr><td>GLM-4.7</td><td>0.808</td><td>0.821</td><td>0.812</td><td>0.839</td><td>0.685</td><td>0.698</td><td>0.709</td><td>0.734</td><td>0.585</td><td>0.600</td><td>0.577</td><td>0.632</td><td>0.714</td><td>0.722</td><td>0.695</td><td>0.783</td></tr><tr><td>Gemini-3-Flash</td><td>0.836</td><td>0.823</td><td>0.822</td><td>0.840</td><td>0.709</td><td>0.702</td><td>0.717</td><td>0.752</td><td>0.625</td><td>0.623</td><td>0.648</td><td>0.689</td><td>0.737</td><td>0.722</td><td>0.739</td><td>0.803</td></tr><tr><td>Average</td><td>0.733</td><td>0.753</td><td>0.755</td><td>0.774</td><td>0.577</td><td>0.614</td><td>0.639</td><td>0.665</td><td>0.441</td><td>0.484</td><td>0.509</td><td>0.557</td><td>0.531</td><td>0.586</td><td>0.614</td><td>0.672</td></tr></table>

Table 1: Main LocBench localization results across seven models. We report File Acc@5, Module Acc@5, Function Acc@5, and Function Rec@10. Methods are abbreviated as B = Bash, S = Snippet Search, S+L = Snippet+ListSym, and R = RepoNav; S+L is a tool-matched snippet baseline with the same list\_symbols access as RepoNav. Bold marks the best setting within each model block and metric group. Full Accuracy@k and Recall@k results are provided in Appendix C, Table 8.

Models and metrics. We evaluate seven proprietary and open-weight models.<sup>2</sup> We report Accuracy@k and Recall@k at file, module, and function levels. Accuracy@k requires all gold targets (up to k) to appear in the top-k predictions; Recall@k measures the fraction recovered. Module-level matching maps predictions to their enclosing module. Transfer is additionally evaluated on SWE-QA-Bench (Peng et al., 2026); its setup is described with the RQ3 results.

## 4.2 RQ1: File-to-Function Gap on LocBench

Table 1 reports the main LocBench results. To control for tool access, Snippet+ListSym (S+L) adds list\_symbols to the flat snippet interface. This lets us separate file-structure browsing gains (S→S+L) from RepoNav’s additional file-centered organization gains (S+L→R), as visualized in Figure 4.

Finding 1: File-structure browsing helps, and RepoNav adds further function-level gains. Adding list\_symbols to the flat snippet interface (S+L) already improves function-level localization over Snippet Search: average Function Acc@5 rises from 0.484 to 0.509 (+2.5 points). However, RepoNav yields a further gain to 0.557 (+4.8 points over S+L), indicating that the filecentered scaffold provides substantial additional benefit beyond file-structure browsing access alone. As Figure 4a shows, the S+L→R increment is largest at the module and function levels, precisely where within-file navigation matters most. This two-stage decomposition indicates that RepoNav’s gains cannot be fully explained by the availability of list\_symbols; the organization of retrieved evidence into actionable, file-centered cues is a key contributing factor.

Finding 2: The file-to-function gap persists under snippet retrieval; RepoNav narrows it substantially. We define the file-to-function gap as the difference between File Acc@5 and Function Acc@5. Under Bash, this gap averages 29.2 points. Snippet Search slightly narrows it to 26.9 points by improving both file- and function-level accuracy, but the gap remains large because function-level gains do not keep pace with file-level gains, even after relevant files are retrieved. RepoNav reduces the gap to 21.7 points, a 5.2-point reduction over Snippet Search. This indicates that structured navigation closes a qualitatively different bottleneck than flat retrieval.

Finding 3: RepoNav remains effective even when file-level performance is already high. For stronger models such as GLM-4.7 and Gemini-

![](images/0aac80c842bc25897beb525959aeb52592a29a6755214d3d6ca090a459662efa.jpg)

(a) Three-stage gain decomposition.  
![](images/35ffb023804773872f225ab6fdbbd60e354470c352af2986b4fcbaabb0588798.jpg)  
(b) File-to-function gap (pp; lower is better)  
Figure 4: LocBench gain decomposition and file-tofunction gap. (a) Average gains over Bash are decomposed into retrieval gains (S−B), file-structure browsing gains ((S+L)−S), and RepoNav gains under matched list\_symbols access (R−(S+L)). (b) RepoNav reduces the average file-to-function gap from 26.9 to 21.7 percentage points compared with Snippet Search.

3-Flash, File Acc@5 already exceeds 0.80 under Bash or Snippet Search, yet RepoNav still improves Function Acc@5 by 3.2 and 6.6 points (5.3% and 10.6% relative), respectively. Notably, Snippet Search yields negligible or even slightly negative function-level changes relative to Bash for these models (e.g., −0.2 points on Gemini-3-Flash), consistent with the retrieval-only gap reported in Table 6 (Appendix B). RepoNav continues to yield gains, indicating that once relevant files are reachable, the remaining bottleneck increasingly lies in navigating and verifying evidence within those files. A paired instance-level analysis over all 560 LocBench instances with GPT-OSS-120B further confirms statistically significant improvements over Snippet Search on Function Acc@5 and Function Rec@10 $( p = 0 . 0 1 7$ and $p = 0 . 0 0 7$ , respectively; see Appendix C.1). Together, these findings provide evidence that file discovery and function discovery are qualitatively different challenges, and that snippet-style retrieval provides insufficient structural context for the latter.

## 4.3 RQ2: Does Structure Help by Volume or by Actionability?

RQ1 shows that RepoNav improves function-level localization, but the mechanism remains unclear. Does the improvement come from exposing more structural information to the agent, or from organizing retrieved evidence into an action-oriented interface that encourages targeted exploration?

To answer this question, we conduct a controlled ablation on the full 560-instance LocBench benchmark using the same model, dense retrieval backend, search-first workflow, tool access, and verification constraints. Only the presentation of retrieved evidence differs. File-Only exposes aggregated candidate files without file-internal structure. Inline Scaffold exposes richer file-internal structure directly in the retrieval output. Tree Scaffold (RepoNav) organizes structural evidence into a compact, navigable scaffold with selective cues and explicit next-step hints. This design separates filelevel aggregation, in-context structural volume, and action-oriented organization, allowing us to test whether structure helps by being larger or by being easier to act on.

<table><tr><td></td><td colspan="2">Endpoint Localization</td><td colspan="4">Behavior &amp; Efficiency</td></tr><tr><td>Method</td><td>Acc@5 (%)</td><td>Rec@10 (%)</td><td>ListSym%</td><td>Files Insp.</td><td>Steps</td><td>Tokens</td></tr><tr><td>File-Only</td><td>32.61</td><td>44.29</td><td>39.11</td><td>2.587</td><td>10.587</td><td>50.2k</td></tr><tr><td>Inline Scaffold</td><td>46.96</td><td>53.84</td><td>36.96</td><td>2.457</td><td>9.630</td><td>56.2k</td></tr><tr><td>Tree Scaffold</td><td>52.13</td><td>60.98</td><td>56.61</td><td>2.304</td><td>9.391</td><td>47.2k</td></tr></table>

Table 2: RQ2 controlled ablation on the full 560- instance LocBench benchmark using GPT-OSS-120B. All settings share the same retrieval backend, tool access, workflow, and verification constraints; only evidence presentation differs. Acc@5/Rec@10 are percentages measuring endpoint localization, while the remaining columns summarize tool use and exploration efficiency.

Table 2 shows that endpoint localization improves from File-Only to Inline Scaffold and further to Tree Scaffold under matched retrieval, workflow, tool access, and verification constraints. Function Acc@5 increases from 32.61% to 46.96% and 52.13%, while Function Rec@10 increases from 44.29% to 53.84% and 60.98%. Thus, exposing file-internal structure is important, but the compact tree scaffold provides additional gains beyond simply placing more structure in the retrieval output.

The behavioral metrics show a similar pattern. Compared with Inline Scaffold, Tree Scaffold inspects fewer files, takes fewer steps, and uses fewer tokens, while invoking list\_symbols more frequently. Figure 5 makes this contrast explicit: Tree Scaffold increases list\_symbols use by 19.65 percentage points (56.61% vs. 36.96%, a 53% relative increase) while using 16% fewer tokens. Together with the evidence-funnel analysis in Appendix D.1, this demonstrates that Tree Scaffold improves post-inspection actionability rather than simply broadening inspection. A complementary leave-one-out ablation further shows that all three scaffold blocks contribute to function-level localization. Removing [ANCHORS], [GLIMPSE], or [CANDIDATE\_TARGETS] reduces Function Acc@5, with the largest degradation observed when removing [CANDIDATE\_TARGETS] (−7.6 points; see Appendix D.2). The benefit is also concentrated in structurally harder files: under a median-split analysis, RepoNav achieves larger Function Acc@5 gains over Snippet Search for files with more functions (+6.4 points), longer files (+8.2 points), and more sibling symbols (+6.0 points; Appendix D.3).

![](images/64d7e5f03630e6dd322aaffde83ca680e70ececbd4f1c275e9be9ba45bbf93e0.jpg)  
Figure 5: Tree Scaffold vs. Inline Scaffold in the RQ2 controlled ablation. Values report Tree Scaffold minus Inline Scaffold. Endpoint localization and ListSym use are absolute percentage-point changes; efficiency metrics are relative percentage changes, where negative values indicate lower cost.

We additionally profile the main interaction costs on GPT-OSS-120B over all 560 LocBench instances. Compared with Snippet Search, RepoNav uses 62.4k versus 31.7k average trace tokens and 25.3 s versus 15.5 s average wall-clock time, while making fewer retrieval-tool calls (1.19 vs. 1.39). RepoNav’s scaffold construction itself has a median tool-side latency of 0.34 s (mean 0.74 s) and requires no additional repository-wide index or graph. Accordingly, we use lightweight to refer to low incremental infrastructure and deployment overhead, rather than lower interaction-token or latency cost.

## 4.4 RQ3: Does the Benefit Transfer Beyond Localization?

The preceding analyses show that RepoNav improves function-level localization on LocBench. We further ask whether RepoNav also benefits repository-level understanding beyond explicit localization. To this end, we evaluate on SWE-QA-Bench (Peng et al., 2026), a repository-level question-answering benchmark that requires agents to locate, inspect, and synthesize evidence from code repositories. Unlike RQ1, this experiment is intended as a transfer probe of the integrated RepoNav interface rather than a causal ablation of tool access. We compare three methods, Bash, Snippet Search, and RepoNav, using the same agent setup, interaction budget, and evaluation protocol. We use the per-model intersection of questions successfully completed and scored by all three methods.

![](images/03cb7dfc79f332244c2bdea9602d8b8265f1b80676060cbc53c3b127b4923fa0.jpg)

(a) Overall answer quality.  
![](images/f68f2b4413fd6c2fb1211b5ad7df3a85362279052930f1b29ee68469a6ffac6a.jpg)  
(b) Per-dimension ∆ (RepoNav − Snippet Search).

Figure 6: SWE-QA-Bench overall results. (a) RepoNav achieves the highest total score across all four models; exact values in Table 3. (b) Gains concentrate in Correctness and Completeness, consistent with the localization findings.

<table><tr><td>Model</td><td>Bash</td><td>Snippet</td><td>RepoNav</td><td> $\Delta _ { \mathrm { S } }$ </td><td> $\Delta _ { \mathrm { B } }$ </td></tr><tr><td>GPT-OSS-120B</td><td>76.12</td><td>78.97</td><td>80.33</td><td>+1.36</td><td>+4.21</td></tr><tr><td>Qwen3-Next-80B</td><td>71.72</td><td>74.31</td><td>76.29</td><td>+1.98</td><td>+4.57</td></tr><tr><td>GLM-4.7</td><td>84.96</td><td>84.75</td><td>86.84</td><td>+2.09</td><td>+1.88</td></tr><tr><td>Gemini-3-Flash</td><td>86.53</td><td>86.55</td><td>88.14</td><td> $+ 1 . 5 9$ </td><td>+1.61</td></tr><tr><td>Average</td><td>79.83</td><td>81.15</td><td>82.90</td><td> $+ 1 . 7 6$ </td><td>+3.07</td></tr></table>

Table 3: SWE-QA-Bench total scores. Scores are reported on the per-model common subset successfully completed and judged for all three methods. $\Delta _ { \mathrm { S } }$ denotes RepoNav minus Snippet Search, and $\Delta _ { \mathrm { B } }$ denotes RepoNav minus Bash. Full dimension-level scores are reported in Appendix E, Table 13.

Answer quality. Table 3 shows that RepoNav obtains the highest total score across all four models, with an average gain of +1.76 over Snippet Search and +3.07 over Bash. The gains concentrate in Correctness and Completeness (Figure 6b), showing that structured navigation mainly helps agents find and synthesize the right repository evidence rather than merely improving surface-level answer style. A similar pattern appears for the two strongest models: Snippet Search yields negligible changes relative to Bash, including a slight drop for GLM-4.7 (84.75 vs. 84.96) and a near-tie for Gemini-3-Flash (86.55 vs. 86.53), whereas RepoNav yields clear improvements (86.84 and 88.14). A question-type breakdown (Appendix E.2) shows that gains span all categories, with the largest improvements on Why questions requiring cross-file evidence synthesis. This pattern shows that the scaffold supports repository-level question answering by helping agents assemble evidence chains across functions and files rather than only localizing isolated targets.

Summary. The SWE-QA results show that RepoNav also improves repository-level question answering beyond explicit localization. Across four models, RepoNav consistently outperforms both Bash and Snippet Search.

## 5 Conclusion

This paper investigates how the organization of retrieval outputs shapes the exploration behavior of LLM-based code agents at the repository scale. Across seven models on LocBench, we identify a persistent file-to-function gap: agents can often reach relevant files, yet still fail to localize the correct function. RepoNav narrows this gap by reorganizing retrieved snippets into a lightweight, file-centered navigation scaffold without changing the underlying retriever. Controlled ablations further show that simply exposing more file-structure information is insufficient; RepoNav’s gains come from organizing retrieved evidence into a structured, navigable form. Overall, these findings show that retrieval-augmented code agents depend not only on what evidence is retrieved, but also on how that evidence is organized for exploration.

## Limitations

Our current evaluation focuses on Python repositories and two benchmarks, LocBench and SWE-QA-Bench. Although RepoNav is designed as a lightweight interface layer rather than a Pythonspecific method, our implementation currently uses Python’s ast module for file-structure extraction. We therefore leave validation on repositories written in other programming languages, such as Java, C, C++, Go, and Rust, to future work. These languages may require language-specific structure extractors, scaffold serialization rules, and evaluation settings. Extending RepoNav to these settings is an important step toward building a more general and practical repository navigation interface for code agents. Our experiments focus on localization and repository-level question answering. Evaluating RepoNav in downstream patch generation, long-horizon maintenance tasks, and interactive developer workflows remains future work.

## References

Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, and William Wang. 2025. SWEsearch: Enhancing software agents with Monte Carlo tree search and iterative refinement. In The Thirteenth International Conference on Learning Representations.

Zhaoling Chen, Robert Tang, Gangda Deng, Fang Wu, Jialong Wu, Zhiwei Jiang, Viktor Prasanna, Arman Cohan, and Xingyao Wang. 2025. LocAgent: Graphguided LLM agents for code localization. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8697–8727, Vienna, Austria. Association for Computational Linguistics.

Tulsee Doshi and Gemini Team. 2025. Gemini 3 Flash: Frontier intelligence built for speed. https: //blog.google/products-and-platforms/ products/gemini/gemini-3-flash/. Accessed: 2026-05-23.

Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. UniXcoder: Unified crossmodal pre-training for code representation. In Proceedings ofthe 60th Annual Meeting ofthe Associa-

tion for Computational Linguistics (Volume 1: Long Papers), pages 7212–7225, Dublin, Ireland. Association for Computational Linguistics.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can language models resolve real-world GitHub issues? In The Twelfth International Conference on Learning Representations.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei Han, Le Sun, and Yongbin Li. 2025. StructRAG: Boosting knowledge intensive reasoning of LLMs via inference-time hybrid information structurization. In The Thirteenth International Conference on Learning Representations.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024a. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173.

Wei Liu, Ailun Yu, Daoguang Zan, Bo Shen, Wei Zhang, Haiyan Zhao, Zhi Jin, and Qianxiang Wang. 2024b. GraphCoder: Enhancing repository-level code completion via coarse-to-fine retrieval based on code context graph. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, pages 570–581. Association for Computing Machinery.

Xiangyan Liu, Bo Lan, Zhiyuan Hu, Yang Liu, Zhicheng Zhang, Fei Wang, Michael Qizhe Shieh, and Wenmeng Zhou. 2025. CodexGraph: Bridging large language models and code repositories via code graph databases. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 142–160, Albuquerque, New Mexico. Association for Computational Linguistics.

Yingwei Ma, Qingping Yang, Rongyu Cao, Binhua Li, Fei Huang, and Yongbin Li. 2024. Alibaba LingmaAgent: Improving automated issue resolution via comprehensive repository exploration. Preprint, arXiv:2406.01422.

MiniMax. 2026. MiniMax-M2.5. https: //huggingface.co/MiniMaxAI/MiniMax-M2.5. Model card. Accessed: 2026-05-25.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

Siru Ouyang, Wenhao Yu, Kaixin Ma, Zilin Xiao, Zhihan Zhang, Mengzhao Jia, Jiawei Han, Hongming Zhang, and Dong Yu. 2025. RepoGraph: Enhancing AI software engineering with repository-level code graph. In The Thirteenth International Conference on Learning Representations.

Weihan Peng, Yuling Shi, Yuhang Wang, Xinyun Zhang, Beijun Shen, and Xiaodong Gu. 2026. SWE-QA: Can language models answer repository-level code questions? In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 8230–8245, San Diego, California, United States. Association for Computational Linguistics.

Qwen Team. 2024. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Qwen Team. 2025a. Qwen3-Coder-30B-A3B-Instruct. https://huggingface.co/Qwen/ Qwen3-Coder-30B-A3B-Instruct. Model card. Accessed: 2026-05-25.

Qwen Team. 2025b. Qwen3-Next-80B-A3B-Instruct. https://huggingface.co/Qwen/ Qwen3-Next-80B-A3B-Instruct. Model card. Accessed: 2026-05-25.

Tarun Suresh, Revanth Gangi Reddy, Yifei Xu, Zach Nussbaum, Andriy Mulyar, Brandon Duderstadt, and Heng Ji. 2025. CoRNStack: High-quality contrastive data for better code retrieval and reranking. In The Thirteenth International Conference on Learning Representations.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, and 5 others. 2025a. OpenHands: An open platform for AI software developers as generalist agents. In The Thirteenth International Conference on Learning Representations.

Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F. Xu, Yiqing Xie, Graham Neubig, and Daniel Fried. 2025b. CodeRAG-Bench: Can retrieval augment code generation? In Findings ofthe Association for Computational Linguistics: NAACL 2025, pages 3199–3214, Albuquerque, New Mexico. Association for Computational Linguistics.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2025. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE):801–824.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Z.AI. 2025. GLM-4.7. https://huggingface.co/ zai-org/GLM-4.7. Model card. Accessed: 2026- 05-25.

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2471–2484, Singapore. Association for Computational Linguistics.

Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. 2024. AutoCodeRover: Autonomous program improvement. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, ISSTA 2024, pages 1592– 1604. Association for Computing Machinery.

## A Parameter Sensitivity Analysis

The hybrid scoring rule uses a weight α to balance peak evidence from the highest-scoring chunk $( s _ { f , ( 1 ) } )$ against consistency across multiple retrieved chunks. To verify that our results are not sensitive to this choice, we perform a lightweight offline sensitivity analysis: for each of 3 repositorylevel splits, we reuse the cached dense index and chunk-level scores, recompute file-level rankings for $\alpha \in \{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ , and re-evaluate file retrieval metrics without rerunning the downstream agent.

<table><tr><td>α</td><td>Acc@1</td><td>Hit@10</td><td>Recall@10</td><td>MRR</td></tr><tr><td>0.0 (Avg)</td><td>0.4965</td><td>0.8056</td><td>0.7648</td><td>0.6077</td></tr><tr><td>0.1</td><td>0.5035</td><td>0.8056</td><td>0.7648</td><td>0.6125</td></tr><tr><td>0.2</td><td>0.5069</td><td>0.8056</td><td>0.7648</td><td>0.6136</td></tr><tr><td>0.3</td><td>0.5208</td><td>0.8056</td><td>0.7648</td><td>0.6231</td></tr><tr><td>0.4</td><td>0.5174</td><td>0.8056</td><td>0.7648</td><td>0.6222</td></tr><tr><td>0.5</td><td>0.5243</td><td>0.8056</td><td>0.7648</td><td>0.6251</td></tr><tr><td>0.6</td><td>0.5174</td><td>0.8090</td><td>0.7671</td><td>0.6229</td></tr><tr><td>0.7</td><td>0.5139</td><td>0.8125</td><td>0.7679</td><td>0.6212</td></tr><tr><td>0.8</td><td>0.5069</td><td>0.8125</td><td>0.7679</td><td>0.6155</td></tr><tr><td>0.9</td><td>0.5104</td><td>0.8125</td><td>0.7679</td><td>0.6168</td></tr><tr><td>1.0 (Max)</td><td>0.5139</td><td>0.8125</td><td>0.7679</td><td>0.6168</td></tr></table>

Table 4: File-level retrieval metrics as a function of α, aggregated over 3 repository-level development splits. Performance is stable across the full range: Acc@1 and MRR peak near $\alpha { = } 0 . 5$ , while Hit@10 and Recall@10 plateau for $\alpha \ge 0 . 7$ . We use $\alpha { = } 0 . 5$ in all reported experiments.

As shown in Table 4, all metrics vary within a narrow band across the full α range. Rankingoriented metrics (Acc@1, MRR) peak near α=0.5, while recall metrics reach a broad plateau for $\alpha \ge 0 . 7 .$ The two extremes, pure averaging (α=0) and pure max pooling (α=1), are both competitive, demonstrating that the hybrid formulation is robust rather than requiring careful tuning. We fix $\alpha { = } 0 . 5$ throughout all experiments reported in this paper.

## B Retrieval Configuration and Baseline Performance

Our dense index follows the function-level chunking protocol of Chen et al. (2025): each toplevel function or method definition is treated as a single chunk and embedded using CodeRankEmbed (Suresh et al., 2025). During the agent loop, the agent issues free-form natural language search queries; the retrieval backend returns the topranked chunks by cosine similarity. RepoNav’s file-level aggregation rule and the snippet baseline operate on the same retrieved chunks; only the postretrieval presentation differs.

Raw-chunk retrieval depth. The main experiments retrieve 80 raw chunks per query. To assess sensitivity to this choice, we conduct a retrieval-only analysis over all 560 LocBench instances, varying the raw retrieval depth over {20, 50, 80, 100}. We keep the post-aggregation file budget fixed at 15 and use the same CodeRankEmbed index and file-aggregation rule throughout. Table 5 reports the resulting retrieval performance.

<table><tr><td colspan="3">Snippet</td><td colspan="2">RepoNav</td></tr><tr><td>Depth</td><td>Hit@15</td><td>Rec@15</td><td>Hit@15</td><td>Rec@15</td></tr><tr><td>20</td><td>79.64</td><td>75.28</td><td>79.46</td><td>75.01</td></tr><tr><td>50</td><td>81.61</td><td>78.32</td><td>81.07</td><td>77.67</td></tr><tr><td>80</td><td>82.50</td><td>79.59</td><td>82.14</td><td>79.11</td></tr><tr><td>100</td><td>82.86</td><td>80.04</td><td>82.32</td><td>79.34</td></tr></table>

Table 5: Retrieval-depth sensitivity on all 560 LocBench instances. The post-aggregation file budget is fixed at 15; all values are percentages.

Both retrieval settings exhibit a similar saturation pattern. Increasing the depth from 20 to 80 yields clear gains, whereas increasing it from 80 to 100 provides only marginal additional improvement. The default depth of 80 therefore lies near the observed saturation region while avoiding an unnecessarily deeper retrieval pool.

Table 6 reports the retrieval-only accuracy of the shared dense index, evaluated without any downstream agent. These numbers provide a retrievalonly reference point for the fixed dense index before downstream agent interaction. The file-tofunction localization gap is already visible at the retrieval-only stage: File Acc@5 reaches 0.721, whereas Function Acc@5 is only 0.348, which is less than half. Since Snippet Search and RepoNav use the same fixed dense index and raw chunk-level scores, this result supports our interpretation that RepoNav’s downstream gains arise from post-retrieval evidence organization and navigation rather than from changes to the underlying retriever.

<table><tr><td rowspan="3"></td><td colspan="3">File</td><td colspan="3">Module</td><td colspan="2">Function</td></tr><tr><td>@1</td><td>@3</td><td>@5</td><td>@5</td><td>@10</td><td>@15</td><td>@5</td><td>@10</td></tr><tr><td>Acc</td><td>0.505</td><td>0.661</td><td>0.721</td><td>0.521</td><td>0.604</td><td>0.666</td><td>0.348</td><td>0.430</td></tr></table>

Table 6: Retrieval-only performance of the dense index used by all agent experiments. No downstream agent is involved; scores reflect pure embedding-based retrieval.

Alternative embedder. To examine whether RepoNav’s benefit transfers beyond the primary CodeRankEmbed retriever, we conduct a fixed 100-instance pilot using UniXcoder (microsoft/unixcoder-base) while retaining the same function-level chunking scheme. Snippet Search and RepoNav use the same UniXcoder index, GPT-OSS-120B backbone, agent loop, and interaction budget. Table 7 reports the results.

<table><tr><td>Method</td><td>File Hit@5</td><td>Func Hit@5</td><td>Func Acc@5</td><td>Func Rec@10</td></tr><tr><td>Snippet Search</td><td>76.0</td><td>60.0</td><td>19.0</td><td>36.2</td></tr><tr><td>RepoNav</td><td>79.0</td><td>63.0</td><td>23.0</td><td>41.8</td></tr><tr><td>∆</td><td>+3.0</td><td>+3.0</td><td>+4.0</td><td>+5.6</td></tr></table>

Table 7: Alternative-embedder pilot on a fixed 100- instance LocBench subset using UniXcoder. All values are percentages.

RepoNav retains positive gains under UniXcoder, improving Function Acc@5 by 4.0 points and Function Rec@10 by 5.6 points over Snippet Search. This pilot provides preliminary evidence that RepoNav is compatible with a second embedding model, while CodeRankEmbed remains the primary retriever evaluated in the full experiments.

## C Full LocBench Results

Table 8 provides the complete LocBench localization results across all seven models, four exploration settings, and the reported Accuracy@k and

Recall@k variants. The main text (Table 1) reports File Acc@5, Module Acc@5, Function Acc@5, and Function Rec@10; this table additionally includes lower- and higher-rank Accuracy@k and Recall@k variants where applicable, enabling finegrained comparison across the reported k values.

Several patterns emerge from the complete results that are not visible in the summary table. First, RepoNav’s gains are consistent across k values: improvements at Acc@1 tend to be slightly smaller than those at larger k, suggesting that RepoNav helps agents recover more of the gold target set within the candidate budget rather than always ranking all required targets first. Second, the recall-level gains are generally larger than the accuracy-level gains at comparable k, reflecting that RepoNav helps agents localize a greater fraction of the total gold targets per instance. Third, the file-level recall columns show that RepoNav largely preserves file-level coverage while improving module- and function-level localization, although small drops appear for some models.

## C.1 Statistical Significance

All configurations use greedy decoding (temperature = 0), and each configuration is evaluated once on every instance. We conduct a paired instance-level analysis over all 560 LocBench instances using GPT-OSS-120B to quantify uncertainty in the primary function-level results.

Both confidence intervals exclude zero, providing statistical evidence for the improvements on the two primary function-level localization metrics.

## D Additional RQ2 Analyses

## D.1 Evidence Funnel

To complement the behavioral analysis in Section 4.3, we report three diagnostic trajectory statistics: whether the gold file becomes available to the agent either through the retrieved candidate set or through subsequent shell exploration (Coverage), whether the agent inspects it given coverage (Inspection), and whether inspection leads to correct localization (Resolution). These statistics are coarse trajectory diagnostics rather than mutually exhaustive paths through the agent workflow, and should not be interpreted as a multiplicative decomposition of the endpoint localization metrics in Table 2.

As Table 10 shows, Inline Scaffold and Tree

<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td rowspan="2">@1</td><td colspan="2">File Acc</td><td rowspan="2">@1</td><td colspan="2">Module Acc @3</td><td rowspan="2">@10</td><td rowspan="2">@1</td><td rowspan="2">Function Acc @3</td><td rowspan="2">@5</td><td rowspan="2">@10</td><td rowspan="2">@3</td><td rowspan="2">File Recall @5</td><td rowspan="2">@10</td><td rowspan="2">Module Recall @3 @5</td><td rowspan="2">@10</td><td rowspan="2">@3</td><td rowspan="2">Function Recall</td><td rowspan="2">@5 @10</td></tr><tr><td></td><td>@3</td></tr><tr><td rowspan="5">Qwen2.5-72B</td><td>Bash</td><td>0.4714</td><td>0.5679</td><td>0.5893</td><td>0.3429</td><td>0.3661</td><td>0.3964</td><td>0.4214</td><td>0.2571</td><td>0.2339</td><td>0.2536</td><td>0.2643</td><td>0.5929 0.6206</td><td>0.6295</td><td>0.4048</td><td>0.4345</td><td>0.4622</td><td>0.2756</td><td>0.2946</td><td>0.3067</td></tr><tr><td>Snippet Search</td><td>0.6304</td><td>0.6607</td><td>0.6804</td><td>0.4357</td><td>0.4446</td><td>0.4607</td><td>0.4696</td><td>0.3196</td><td>0.2821</td><td>0.3000</td><td>0.3089 0.6985</td><td>0.7179</td><td>0.7223</td><td>0.4943</td><td>0.5176</td><td>0.5272</td><td>0.3482</td><td>0.3663</td><td>0.3771</td></tr><tr><td>Snippet+ListSym</td><td>0.6464</td><td>0.6696</td><td>0.6732</td><td>0.5229</td><td>0.5139</td><td>0.5621</td><td>0.5386</td><td>0.3886</td><td>0.3629</td><td>0.3843</td><td>0.4064</td><td>0.7063 0.7106</td><td>0.7112</td><td>0.5405</td><td>0.5785</td><td>0.5886</td><td>0.4193</td><td>0.4436</td><td>0.4901</td></tr><tr><td>RepoNav</td><td>0.6661</td><td>0.6911</td><td>0.6982</td><td>0.5411</td><td>0.5375</td><td>0.5696</td><td>0.5929</td><td>0.4071</td><td>0.3946</td><td>0.4339</td><td>0.4786</td><td>0.7301 0.7382</td><td>0.7382</td><td>0.5914</td><td>0.6297</td><td>0.6488</td><td>0.4634</td><td>0.5113</td><td>0.5492</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.4007</td><td>0.4171</td></tr><tr><td rowspan="4">GPT-OSS-120B</td><td>Bash</td><td>0.6179</td><td>0.6875</td><td>0.7054</td><td>0.4679</td><td>0.4643 0.5786</td><td>0.5000 0.6125</td><td>0.5196 0.6357</td><td>0.3643 0.4929</td><td>0.3214</td><td>0.3393</td><td>0.3536 |0.7223</td><td>0.7397</td><td>0.7457</td><td>|0.5167 0.6372</td><td>0.5549 0.6748</td><td>0.5742 0.6936</td><td>|0.3798</td><td></td></tr><tr><td>Snippet Search</td><td>0.7125 0.7268</td><td>0.7518</td><td>0.7589</td><td>0.5964</td><td></td><td></td><td>0.4911</td><td>0.4393 0.4521</td><td>0.4643</td><td>0.4893</td><td>0.7887</td><td>0.7967</td><td>0.7976</td><td></td><td></td><td>0.5226</td><td>0.5462</td><td>0.5707</td></tr><tr><td>Snippet+ListSym</td><td>0.7250</td><td>0.7500 0.7589</td><td>0.7589 0.7696</td><td>0.5929</td><td>0.6007</td><td>0.6304 0.6500</td><td>0.6429 0.6625 0.5064</td><td>0.4868</td><td>0.4875 0.5271</td><td>0.5089 0.5486</td><td>0.7863 0.7964</td><td>0.7967</td><td>0.7976</td><td>0.6567</td><td>0.6815 0.7020</td><td>0.5409 0.5636</td><td>0.5663</td><td>0.5858</td></tr><tr><td>RepoNav</td><td></td><td></td><td></td><td>0.5982</td><td>0.6071</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.8056</td><td>0.8074</td><td>0.6810</td><td>0.7163</td><td>0.7269</td><td>0.5990</td><td>0.6184</td></tr><tr><td rowspan="7">Qwen3-Next-80B</td><td>Bash</td><td>0.6482 0.7179</td><td>0.7107 0.7286</td><td>0.7339 0.7321</td><td>0.4804 0.5536</td><td>0.4929 0.5589</td><td>0.5232 0.5679</td><td>0.5464 0.3696 0.5804 0.4482</td><td>0.3321 0.4054</td><td>0.3393 0.4107</td><td>0.3500 0.4268</td><td>0.7430</td><td>0.7668</td><td>0.7760</td><td>0.5352</td><td>0.5701</td><td>0.5936 0.3751</td><td>0.3944 0.4957</td><td>0.4098 0.5170</td></tr><tr><td>Snippet Search</td><td>0.7103</td><td>0.7304</td><td>0.7329</td><td>0.5807</td><td>0.5818</td><td>0.5886</td><td>0.6464 0.4986</td><td>0.4557</td><td>0.4539</td><td>0.4718</td><td>0.7698 0.7380</td><td>0.7764 0.7535</td><td>0.7764 0.7557</td><td>0.6062 0.6200</td><td>0.6346 0.6520 0.6542</td><td></td><td>0.4634 0.5010</td><td>0.5309 0.5636</td></tr><tr><td>Snippet+ListSym</td><td>0.7339</td><td>0.7357</td><td>0.7393</td><td>0.6018</td><td>0.6018</td><td>0.6321</td><td>0.6482 0.5146</td><td>0.4893</td><td>0.4982</td><td>0.5268</td><td>0.7770</td><td>0.7814</td><td>0.7819</td><td>0.6562</td><td>0.6768 0.7000</td><td>0.5355</td><td></td><td>0.6215</td></tr><tr><td>RepoNav</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7212</td><td></td><td>0.5774</td><td></td></tr><tr><td>Bash</td><td>0.6429</td><td>0.6857</td><td>0.7000</td><td>0.5393</td><td>0.5464</td><td>0.5643</td><td>0.5732 0.4571</td><td>0.4089</td><td>0.4196</td><td>0.4232</td><td>0.7215</td><td>0.7375</td><td>0.7396</td><td>0.5834 0.6186</td><td>0.6322</td><td>|0.4523</td><td>0.4858</td><td>0.4982</td></tr><tr><td>Snippet Search</td><td>0.6786</td><td>0.6893</td><td>0.6929</td><td>0.5679</td><td>0.5571</td><td>0.5839</td><td>0.5875 0.4804</td><td>0.4464</td><td>0.4589</td><td>0.4679</td><td>0.7225</td><td>0.7282</td><td>0.7309</td><td>0.6035</td><td>0.6429 0.6537</td><td>0.4972</td><td>0.5321</td><td>0.5507</td></tr><tr><td rowspan="7">Qwen3-Coder-30B</td><td>Snippet+ListSym</td><td>0.6804 0.6946</td><td>0.6964 0.7196</td><td>0.7071 0.7268</td><td>0.5857 0.6036</td><td>0.5768 0.5982 0.6036 0.6286</td><td>0.6036 0.6357</td><td>0.4964 0.5054</td><td>0.4643 0.4893</td><td>0.4757 0.5161</td><td>0.4982 0.5393</td><td>0.7326</td><td>0.7436</td><td>0.7454</td><td>0.6235 0.6577</td><td>0.6663</td><td>0.5231 0.5303</td><td>0.5607 0.5887</td><td>0.5717 0.6180</td></tr><tr><td>RepoNav</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7555</td><td>0.7623</td><td>0.7623</td><td>0.6424 0.6887</td><td>0.6997</td><td></td><td></td><td></td></tr><tr><td>Bash</td><td>0.7354</td><td>0.7425</td><td>0.7604</td><td> 0.6193</td><td>0.6264</td><td>0.6604</td><td>0.6729 0.5443</td><td>0.5086</td><td>0.5246</td><td>0.5407</td><td>0.7771</td><td>0.7948</td><td>0.8098</td><td>|0.6669 0.7141</td><td>0.7347</td><td>|0.5525 0.5722</td><td>0.6038</td><td>0.6328 0.6157 0.6409</td></tr><tr><td>Snippet Search</td><td>0.7393 0.7589</td><td>0.7518 0.7732</td><td>0.7643 0.7804</td><td>0.6321 0.6482</td><td>0.6429 0.6411</td><td>0.6714 0.6714</td><td>0.6857 0.5500 0.6893 0.5696</td><td>0.5268 0.5339</td><td>0.5321 0.5339</td><td>0.5482 0.5589</td><td>0.7822 0.8093 0.8199</td><td>0.7963 0.8235</td><td>0.7996</td><td>0.6847 0.7298</td><td>0.7485</td></table>

Table 8: Complete LocBench localization results with all @k variants. This table supplements Table 1 with additional Acc@k and Recall@k values not shown in the main text. Bold indicates the best setting within each model block.

<table><tr><td>Metric</td><td>Δ</td><td>95% CI</td><td>p-value</td></tr><tr><td>Function Acc@5</td><td>+6.28 pp</td><td>[+0.7, +6.4]</td><td>0.017</td></tr><tr><td>Function Rec@10</td><td>+4.77 pp</td><td> $[ + 1 . 0 , + 6 . 2 ]$ </td><td>0.007</td></tr></table>

Table 9: Paired instance-level analysis of RepoNav versus Snippet Search over all 560 LocBench instances with GPT-OSS-120B. Confidence intervals are 95% paired-bootstrap CIs.

<table><tr><td>Setting</td><td>Coverage</td><td>Inspection</td><td>Resolution</td></tr><tr><td>File-Only</td><td>78.26%</td><td>97.22%</td><td>92.50%</td></tr><tr><td>Inline Scaffold</td><td>80.43%</td><td>97.30%</td><td>92.68%</td></tr><tr><td>Tree Scaffold</td><td>82.61%</td><td>92.11%</td><td>97.50%</td></tr></table>

Table 10: Evidence funnel across the three RQ2 interface variants. Each stage conditions on the previous one and should be read as a diagnostic decomposition, not as a multiplicative estimate of endpoint localization performance. All three settings share the same dense retrieval backend; Coverage can still vary because agents may discover files through shell exploration after the initial retrieval output.

Scaffold exhibit different diagnostic profiles. Inline Scaffold attains the highest inspection rate (97.30%), consistent with an information-heavy interface that encourages the agent to read broadly. Tree Scaffold leads to more selective inspection (92.11%) but achieves the highest resolution rate once the gold file is inspected (97.50%), consistent with the more frequent list\_symbols usage and fewer wasted steps reported in Table 2.

The key difference between these two behavioral profiles can be summarized as follows. Inline Scaffold maximizes the probability of looking at the gold file: by embedding rich structural detail directly in the retrieval output, it lowers the cost of passive browsing. Tree Scaffold instead maximizes the probability of correctly acting on the gold file once inspected: by presenting compact cues with explicit next-step prompts, it encourages the agent to actively verify candidates through tool use rather than relying on in-context information alone. This decomposition reinforces the conclusion that Tree Scaffold’s advantage lies in the quality of postinspection navigation rather than the breadth of initial coverage.

## D.2 Block-level Scaffold Ablation

To complement the representation-level comparison in Section 4.3, we conduct a matched leave-oneout ablation of the three RepoNav scaffold blocks on all 560 LocBench instances using GPT-OSS-120B. All variants use identical prompts, tools, interaction budgets, metrics, and evaluation data, with one scaffold block removed at a time.

<table><tr><td>Variant</td><td>Acc@5</td><td>Rec@10</td><td>ListSym%</td><td>Tokens</td></tr><tr><td>Full Scaffold</td><td>52.13</td><td>60.98</td><td>56.61</td><td>52.6k</td></tr><tr><td>w/o Anchors</td><td>48.90</td><td>57.10</td><td>68.00</td><td>62.8k</td></tr><tr><td>w/o Glimpse</td><td>49.60</td><td>58.10</td><td>66.20</td><td>58.7k</td></tr><tr><td>w/o Targets</td><td>44.50</td><td>53.50</td><td>36.60</td><td>54.2k</td></tr></table>

Table 11: Block-level leave-one-out ablation of the RepoNav scaffold. Acc@5 and Rec@10 denote functionlevel localization performance.

Removing any scaffold block reduces functionlevel localization performance. Removing [ANCHORS] decreases Function Acc@5 by 3.2 points and Function Rec@10 by 3.9 points, while removing [GLIMPSE] decreases them by 2.5 and 2.9 points, respectively. The largest degradation occurs without [CANDIDATE\_TARGETS], where Function Acc@5 drops by 7.6 points and Function Rec@10 by 7.5 points. Moreover, removing [ANCHORS] or [GLIMPSE] increases both list\_symbols use and token consumption, suggesting that these blocks reduce additional manual browsing. Overall, the full scaffold achieves the strongest accuracy–token trade-off among the evaluated variants.

## D.3 Complexity-Stratified Analysis

To examine whether RepoNav’s benefit is associated with within-file difficulty, we perform a median-split analysis over all 560 LocBench instances using GPT-OSS-120B. We stratify instances according to three properties of the gold file: number of functions, file length, and number of sibling symbols.

<table><tr><td>Higher-complexity subset</td><td>∆ Acc@5</td><td>∆ Rec@10</td></tr><tr><td>More functions</td><td>+6.4</td><td>+6.1</td></tr><tr><td>Longer files</td><td>+8.2</td><td>+7.3</td></tr><tr><td>More sibling symbols</td><td>+6.0</td><td>+5.9</td></tr></table>

Table 12: RepoNav gains over Snippet Search on the higher-complexity half of LocBench under three median-split criteria. Values are absolute percentagepoint improvements.

RepoNav’s gains are consistently larger on files with greater within-file complexity, while the corresponding gains on simpler files are small. This pattern supports the interpretation that RepoNav primarily helps agents discriminate among plausible targets after reaching a relevant file, rather than merely improving initial file discovery.

## E SWE-QA Full Results

## E.1 Dimension-Level Scores

Table 13 presents the full dimension-level SWE-QA-Bench results, complementing the aggregate total scores reported in Table 3. Each answer is independently scored on a 20-point scale across five dimensions: Correctness, Completeness, Relevance, Clarity, and Reasoning.

Across all four models, RepoNav achieves the highest score on every individual dimension. The gains are most pronounced in Correctness and Completeness, consistent with the hypothesis that structured navigation helps agents find and synthesize the right evidence rather than merely improving surface-level answer quality. Relevance, Clarity, and Reasoning show smaller but consistently positive improvements, indicating that better evidence navigation has a downstream effect on overall answer coherence.

<table><tr><td>Model</td><td>Method</td><td>Corr.</td><td>Comp.</td><td>Rel.</td><td>Clar.</td><td>Reas.</td><td>Total</td></tr><tr><td rowspan="3">GPT-OSS-120B</td><td>Bash</td><td>14.06</td><td>11.68</td><td>17.55</td><td>16.54</td><td>16.29</td><td>76.12</td></tr><tr><td>Snippet Search</td><td>14.46</td><td>13.20</td><td>18.08</td><td>16.72</td><td>16.51</td><td>78.97</td></tr><tr><td>RepoNav</td><td>14.84</td><td>13.66</td><td>18.35</td><td>16.82</td><td>16.66</td><td>80.33</td></tr><tr><td rowspan="3">Qwen3-Next-80B</td><td>Bash</td><td>12.44</td><td>10.52</td><td>17.91</td><td>15.71</td><td>15.14</td><td>71.72</td></tr><tr><td>Snippet Search</td><td>13.22</td><td>11.78</td><td>17.96</td><td>15.88</td><td>15.47</td><td>74.31</td></tr><tr><td>RepoNav</td><td>13.63</td><td>12.32</td><td>18.07</td><td>16.31</td><td>15.96</td><td>76.29</td></tr><tr><td rowspan="3">GLM-4.7</td><td>Bash</td><td>16.11</td><td>15.96</td><td>17.93</td><td>17.50</td><td>17.46</td><td>84.96</td></tr><tr><td>Snippet Search</td><td>16.05</td><td>15.96</td><td>17.81</td><td>17.50</td><td>17.43</td><td>84.75</td></tr><tr><td>RepoNav</td><td>16.69</td><td>16.97</td><td>18.08</td><td>17.63</td><td>17.47</td><td>86.84</td></tr><tr><td rowspan="3">Gemini-3-Flash</td><td>Bash</td><td>16.39</td><td>16.16</td><td>18.17</td><td>17.91</td><td>17.86</td><td>86.53</td></tr><tr><td>Snippet Search</td><td>16.35</td><td>16.02</td><td>18.11</td><td>17.90</td><td>17.87</td><td>86.55</td></tr><tr><td>RepoNav</td><td>17.04</td><td>17.13</td><td>18.21</td><td>17.93</td><td>17.93</td><td>88.14</td></tr></table>

Table 13: SWE-QA-Bench dimension-level results. Results are reported on the per-model common subset successfully scored for all three methods. Each dimension is scored on a 20-point scale; Total is their sum (max 100). Dimensions: Correctness (Corr.), Completeness (Comp.), Relevance (Rel.), Clarity (Clar.), and Reasoning (Reas.).

## E.2 Question-Type Breakdown

Figure 7 breaks down SWE-QA-Bench performance by question type (How, What, Where, Why) across all four evaluated models. The analysis complements the aggregate results in Table 3 by showing that RepoNav’s gains are not confined to a single question category.

Score improvements are observed across all types, with the largest and most consistent gains on Why questions. This is consistent with the localization findings: Why questions typically require synthesizing evidence across multiple files and tracing causal chains through the codebase—precisely the scenario where structured navigation helps agents avoid premature anchoring on a single entry point.

Interaction savings (measured in steps) are substantial for three of four models. GPT-OSS-120B shows smaller and mixed savings, suggesting that efficiency gains may depend on model-specific exploration behavior rather than model strength alone. The explicit continuation cues in RepoNav therefore appear to interact with each model’s default exploration policy.

![](images/b43f81ac133dbc8c064c256903a36eefae0fa141d5dbdaaeec3962c64cf712ff.jpg)  
(a) Score by question type.

![](images/f9a36dba0a31272136db2e8cddb6dc58a4e7d7374a93175d4e58fab63ef08c76.jpg)  
(b) Interaction savings by question type.  
Figure 7: SWE-QA-Bench question-type breakdown across four models. (a) Score improvements span all question types, with the largest gains on Why questions. (b) Interaction savings are consistent for three of four models; GPT-OSS-120B shows mixed results. Model names are abbreviated for space; see Section 4.1 for full names.

## F Additional Case Studies

We provide additional analyses illustrating navigation regimes and residual failure modes beyond those highlighted in the main text. Figure 2 illustrates the core same-file disambiguation mechanism; below we describe a cross-file pivoting case, summarize residual file-to-function failures, and present a representative remaining failure boundary.

## Cross-file pivoting (LocBench).

In DS4SD\_\_docling-314, the visible symptom appears near Markdown output code, while the gold target lies in an upstream implementation file, msword\_backend.py. Snippet-style search keeps the agent near the output-side sink, since the highest-scoring chunks come from Markdownrelated code that shares surface keywords with the issue. RepoNav retains the upstream implementation file in its candidate structure because its hybrid file-level scoring rule rewards files supported by multiple moderately scored chunks, rather than only by a single dominant match. The tree scaffold then exposes both symptom-side and source-side files in a navigable list, enabling the agent to pivot from output-related code to the gold implementation target.

## Residual failure diagnostic (LocBench).

We further analyze the 69 GPT-OSS-120B cases in which RepoNav reaches the gold file but misses the target function. An automatic diagnostic based on rule-based file-complexity and trajectory signals attributes 71.0% of these failures to ambiguous or similar sibling functions, 17.4% to long or crowded gold files, and 11.6% to within-file ranking or incomplete verification, indicating that most residual errors arise from fine-grained discrimination within the correct file.

## Remaining failure boundary (LocBench).

In yt-dlp\_\_yt-dlp-11615, RepoNav successfully brings the agent to the correct file, but the agent still stops at higher-level wrapper methods rather than drilling down to the gold targets, which are lower-level helper functions called by the wrapper. The [GLIMPSE] block lists these helpers, but the agent does not inspect them further, treating the wrapper as a sufficient answer. This suggests that RepoNav can reduce wrong-file fixation, but deep same-file evidence harvesting remains an open problem, particularly when the gold target is multiple call-hops away from the most salient entry point.

## G RepoNav Scaffold Specification and Output Format

## G.1 Serialization Budgets and Ordering

RepoNav uses a single pre-specified set of serialization budgets and deterministic ordering rules across all experiments to keep the scaffold compact and reproducible. For each candidate file, [ANCHORS] includes at most two symbols. Anchors are selected by case-insensitive substring matching between query tokens and symbol names, and ranked by token overlap, symbol-kind priority, and span length.

<-). The glimpse block exposes non-anchor symbols as a structural sketch, preventing fixation on anchors alone. The candidate targets block aggregates actionable options and ends with an explicit continuation cue (» Next: run \`list\_symbols\` on this file to inspect sibling symbols), lowering the cost of continued exploration.

[GLIMPSE] includes up to three non-anchor symbols per file, selected by query-aware ranking while preserving symbol-kind diversity. If the top-ranked file has no anchor match, this budget is increased to five to expose a broader file sketch.

[CANDIDATE\_TARGETS] is capped at four entries. Candidates are ordered first by anchors, then by same-file call-neighborhood symbols of anchors, and finally by selected glimpse symbols. A fixed continuation cue is appended whenever a candidate block is present.

Call context is name-only: it records same-file caller and callee symbol names without function bodies, arguments, or interprocedural analysis. By default, RepoNav parses the top five candidate files and auto-expands full three-block scaffolds for the top three.

## G.2 Serialized Output Example

[ DIR ] src / backend /   
[ FILE ] backend / server .py ( evidence : 3)   
|-- imports -by <- backend /app .py   
|-- [ ANCHORS ]   
| \`-- oauth\_callback ( function )[L120 - L156 ]   
| invokes -> validate\_token , init\_app   
|-- [ GLIMPSE ]   
| \`-- class HTTPServer :   
| \`-- def handle\_request ():   
[ CANDIDATE\_TARGETS ]   
- backend / server . py : oauth\_callback   
- backend / server . py : validate\_token   
>> Next : run \`list\_symbols \` on this file to   
inspect sibling symbols .  
Figure 8: A truncated example of RepoNav’s serialized output. The indentation-based tree provides filecentered organization, a compact structural sketch, and explicit continuation cues. The plain-text format requires no specialized query language from the agent.

Figure 8 shows a truncated example of RepoNav’s serialized output as presented to the agent. The indentation-based tree provides file-centered organization without requiring any specialized query language or structured API from the agent.

The three blocks—[ANCHORS], [GLIMPSE], and [CANDIDATE\_TARGETS]—correspond to the design principles described in Section 3.1. The anchor block provides grounded entry points with callcontext annotations (invokes -> and invoked-by