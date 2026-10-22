# Reasoning–Search–Augmented LLMs: An Operational Taxonomy, Benchmark Analysis and Controlled Evidence Synthesis

Biswas Poudel<sup>1,\*</sup>, Nilson Chapagain<sup>1</sup>, Amit Kumar<sup>2</sup>, and Xianshun Jiang<sup>1</sup>

<sup>1</sup>Texas A&M University, College Station, TX, USA

{bpoudel,nchapagain,xianshun}@tamu.edu

<sup>2</sup>Texas A&M University–Corpus Christi, Corpus Christi, TX, USA

akumar3@islander.tamucc.edu

Abstract—Large language models (LLMs) demonstrate impres sive emergent reasoning capabilities, yet they remain constrained by fixed model parameters and finite context windows. Even when augmented with static retrieval, they can produce statements that are unsupported by, or even contradictory to, retrieved evidence. This leads to hallucinations, difficulty incorporating up-to-date information, and failures to correctly interpret or reconcile sources. Recent work addresses these limitations by tightly coupling stepby-step reasoning with interactive search, treating retrieval as an action within an iterative agentic loop rather than a one-shot preprocessing step. This study reviews frameworks that implement such reasoning-search loops by interleaving planning, querying, retrieval, integration, verification, and stopping. We analyze these systems through four architectural axes: retrieval coupling, controller design, knowledge source, and grounding strictness. We separately treat the learning signal as a cross-cutting dimension because prompting, supervision, preference optimization, and reinforcement learning can instantiate the same architecture while producing materially different search, stopping, and grounding behavior. We further distinguish benchmark difficulty from RSAloop demand and synthesize controlled within-study ablations and quality–cost deltas, avoiding invalid pooling across heterogeneous model backbones, retrievers, corpora, and evaluation protocols. Our goal is to clarify the landscape of reasoning-search-augmented LLMs and outline a unified perspective on the design choices shaping future research.

Index Terms—large language models, retrieval-augmented generation, reasoning, search agents, taxonomy

## I. INTRODUCTION

Large language models (LLMs) have demonstrated strong emergent reasoning capabilities across mathematics, programming, and open-domain question answering [1], [2]. However, they remain constrained by (i) static parametric knowledge fixed at training time [3], [4] and (ii) finite context windows [5]. As a result, LLMs may hallucinate or omit critical facts on complex, long-horizon, or time-sensitive queries [5], [6]. Chain-of-thought and self-consistency prompting improve intermediate reasoning, yet still operate primarily over the model’s parametric memory [1], [7], [8].

Two lines of work have partially addressed these limitations. Early tool-use methods such as ReAct and Self-Ask interleave natural-language reasoning with calls to search APIs, but often rely on hand-crafted prompts and heuristics to decide when to search and what to retrieve [9], [10]. In parallel, retrievalaugmented generation (RAG) grounds generation in retrieved documents and reduces hallucinations compared to ungrounded LLMs [11]. Nevertheless, standard RAG is typically a oneshot pipeline and is vulnerable to retrieval irrelevance, poor evidence use, and latency bottlenecks [12]. These limitations have motivated systems that treat search as a first-class, iterative action within the reasoning process.

In many reasoning-intensive settings, retrieval is not only about finding relevant text: evidence may be structured, attributed, and distributed across heterogeneous sources such as documents, knowledge bases, databases, and tool outputs. This makes reasoning-search systems more than simple retrievethen-generate pipelines, since query formulation, evidence integration, and verification must often operate over constrained interfaces and partially structured observations. Several systems reason over graph-structured state spaces or traverse knowledgegraph paths, making intermediate evidence more inspectable and multi-hop reasoning more explicit (e.g., Think-on-Graph, ReKG-MCTS, GraphReader) [13]–[15]. Likewise, tool- and knowledge-base-mediated retrieval turns querying into an explicit action over external environments, where results come from typed or otherwise constrained interfaces such as APIs, databases, and knowledge graphs (e.g., MRKL, Toolformer) [16], [17]. These developments reinforce a broader view of retrieval as controlled evidence acquisition within the reasoning process rather than a fixed preprocessing step.

Building on this motivation, a new generation of systems learns search behavior as part of the reasoning process. For instance, Search-R1 uses reinforcement learning (RL) to train LLMs to generate multiple queries during step-by-step reasoning and learn a stopping policy, improving over static RAG on open-domain and multi-hop QA [18].

These developments motivate the present study of reasoning– search-augmented large language models (RSA-LLMs). We formalize an RSA-LLM as a system in which external evidence acquisition is embedded within the model’s internal reasoning and control loop, rather than being treated as an offline or fixed preprocessing stage. A system is classified as an RSA-LLM if and only if it satisfies the following three necessary conditions:

1) State-dependent evidence acquisition. At least one decision governing evidence acquisition is conditioned on the current internal reasoning state and/or accumulated evidence. Such decisions include whether or when to retrieve, how to formulate or reformulate a query, which external source or tool to invoke, and when to terminate retrieval.

2) Evidence-bearing external interaction. The system must obtain task-relevant information through interaction with an external information environment, including but not limited to web search engines, document corpora, structured knowledge bases, databases, or API-based evidence retrieval services.

3) Closed-loop evidence integration. Retrieved evidence is not merely appended to the context but is integrated into the system’s evolving internal state, thereby influencing subsequent reasoning steps, verification processes, answer construction, and/or future evidence-acquisition decisions.

From an operational perspective, RSA-LLMs instantiate a recurrent plan–query–retrieve–integrate–verify–stop computation cycle. Planning identifies missing or uncertain information required for task completion; querying operationalizes these information needs into executable retrieval requests; retrieval obtains external evidence; integration updates the model’s working memory or latent reasoning state; verification evaluates evidential adequacy and claim support; and stopping determines whether the system should terminate with a response or continue further search. These components should be interpreted as functional abstractions rather than strictly modular architectural components, and may be implemented implicitly within a unified controller. Importantly, a system may complete a task after a single retrieval step; however, it must support state-conditioned control decisions that allow continuation, query revision, verification, or termination based on observed evidence. Recent adaptive information-seeking frameworks such as AISO and interleaved reasoning–action paradigms such as ReAct exemplify this closed-loop structure, while more recent approaches including Search-R1 and ReSearch further demonstrate learned multi-turn retrieval and stopping policies [9], [18]–[20].

This definition delineates RSA-LLMs from closely related but structurally distinct system classes. In particular, classical retrieval-augmented generation (RAG) and Fusion-in-Decoder (FiD) architectures incorporate external evidence but operate under a fixed or predetermined retrieval schedule, and therefore function as static conditioning mechanisms rather than adaptive reasoning–search systems; accordingly, they are treated as reference baselines rather than RSA-LLMs [4], [21]. Similarly, repeated or multi-step retrieval pipelines remain outside the RSA-LLM class when retrieval actions are pre-specified and not conditioned on intermediate reasoning outcomes or observed evidence. Generic tool-use agents are included only insofar as tool invocations retrieve or access external, task-relevant evidence and participate in the closed-loop reasoning process; purely computational or procedural tool calls do not satisfy the evidence-acquisition criterion. Finally, internal structured reasoning mechanisms such as tree- or graph-based search over an already provided context do not constitute reasoning–search augmentation unless they explicitly trigger additional external evidence acquisition. Consequently, RSA-LLM membership is determined strictly by the three conditions above, whereas design dimensions such as controller architecture, reasoning topology, knowledge source, grounding constraints, and learning paradigm characterize implementation variants within the RSA-LLM class. To make the scope of the survey auditable, we additionally report the literature-search and selection procedure, distinguish core RSA systems from adjacent baselines and evaluation studies, and apply the operational inclusion criteria consistently across the reviewed literature.

Our contributions are: (i) an operational RSA-LLM definition and boundary; (ii) a four-axis system taxonomy with edgecase coding rules; (iii) a cross-cutting analysis of learning signals; and (iv) a structured review and evaluation framework connecting architectures with benchmarks, budgets, grounding mechanisms, and controlled within-study evidence.

## A. Review Scope and Selection Protocol

We followed a structured and auditable literature-selection protocol designed to make query construction and screening reproducible while providing representative coverage of reasoning-search-augmented LLMs. The search covered work published or publicly released between January 2020 and 31 May 2026. Candidate studies were identified through ACL Anthology, IEEE Xplore, ACM Digital Library, Open-Review, arXiv (cs.CL, cs.IR, cs.AI, and cs.LG), Semantic Scholar, and Google Scholar. For each source, we applied the Boolean template (“large language model” OR “LLM”) AND q, instantiating q separately with each of the following exact phrases: “reasoning with search,” “search-augmented reasoning,” “reasoning-search,” “adaptive retrieval,” “iterative retrieval,” “agentic RAG,” “deep research agent,” “multi-step retrieval,” “web-search agent,” and “reinforcement learning for search.” Database-specific syntax was adapted where necessary without changing the underlying search concepts. We supplemented these searches through backward and forward citation tracing from foundational and recent systems, including RAG, AISO, ReAct, Search-R1, and relevant RAG-reasoning surveys [4], [9], [18], [19], [22]. During manuscript preparation, bibliographic metadata were updated only for studies already eligible by that date; no work first released after the cutoff was introduced.

Candidate studies were first screened by title and abstract and then assessed using the full text. The texts were eval uated against the operational conditions defined above. We excluded retrieval-only methods, fixed retrieve-then-generate pipelines, predetermined iterative retrieval, internal-only reasoning methods, and general tool-use systems whose tools did not acquire external evidence. Static RAG, retrieval components, and internal reasoning methods were retained only when they provided informative adjacent baselines. Feedback-space methods that search over internally generated artifacts using symbolic validator outputs are retained as adjacent control methods but excluded from the core RSA set when they do not acquire external task evidence; model-space search for planning-domain generation is one such case [23]. Benchmark and evaluation studies were recorded separately when they measured retrieval, grounding, tool execution, or long-horizon search behavior relevant to RSA systems.

![](images/155b9e4f867b3984e455eeacc2c4c660ec4d893e736ad3cb436cacf5ae46796e.jpg)  
Fig. 1. Unified taxonomy of Reasoning–Search-Augmented LLMs. The top panel illustrates the iterative reasoning–search loop, while the bottom pane characterizes the design space along four orthogonal axes: retrieval coupling, controller design, knowledge source, and grounding strictness. Scheduled/Static retrieval is shown only as a reference baseline; learning signal is treated as a cross-cutting dimension.

For each retained system, we extracted its publication status, learning signal, retrieval-coupling mechanism, controller architecture, reasoning topology, knowledge source, grounding mechanism, evaluation benchmarks, reported retrieval budget, and principal results. When multiple versions existed, we used the latest version available before the cutoff date and preferred peer-reviewed versions when available. Because of the rapid development of this area, relevant preprints were included but distinguished from peer-reviewed publications. The structured comparison set reported in Tables I and II comprises 13 core RSA systems, four adjacent systems, and 15 benchmark or evaluation studies. Additional studies were retained as component-level or contextual references but were not included in the system-level coding counts. The objective is transparent and reproducible coverage of representative design configurations rather than an exhaustive bibliometric census.

## B. Positioning Relative to Prior Surveys

Existing surveys address several neighboring but distinct levels of analysis. General RAG surveys organize systems around retrieval, augmentation, generation, training, and application pipelines [11]. RAG-reasoning surveys primarily distinguish how reasoning improves retrieval-augmented generation, how retrieved knowledge supports reasoning, and how the two processes are synergistically combined [22]. Agentic-RAG surveys emphasize agent autonomy, orchestration patterns, reflection, planning, multi-agent collaboration, and practical workflow construction [24]. Tool-learning surveys adopt the broader problem of teaching foundation models to select, invoke, and learn from external tools, regardless of whether those tools acquire evidence [25].

Our scope is complementary but more operationally constrained. The unit of analysis is the closed-loop evidenceacquisition system: how the evolving reasoning state controls external search, how retrieved observations modify subsequent control, where evidence originates, and what evidential obligations govern the final output. The proposed taxonomy therefore contributes: (i) an explicit membership boundary separating RSA-LLMs from static RAG, internal-only reasoning, and nonevidential tool use; (ii) four jointly applicable system-level axes that separate acquisition control, controller organization, source constraints, and grounding obligations; (iii) reproducible coding rules for hybrid and boundary cases; and (iv) structured comparisons that connect architectural configurations with learning signals, benchmarks, retrieval budgets, grounding mechanisms, and reported outcomes. Unlike pipeline-centered or direction-of-influence taxonomies, this framing is designed to support system-level auditing and comparison of how evidence is acquired, controlled, and enforced during reasoning. Viewed as a data-systems problem, RSA-LLMs must search, integrate, and verify high-volume, heterogeneous, and rapidly changing evidence under token, latency, and API budgets. This connects retrieval control to scalable search, data integration, provenance, and cost-aware processing.

## II. TAXONOMY OF REASONING–SEARCH–AUGMENTED LLMS

Using the operational boundary established in Section I, we characterize qualifying systems along four architectural axes: retrieval coupling, controller design, knowledge source, and grounding strictness. The four axes are operational descriptors rather than statistically selected predictors of benchmark performance. They were chosen to separate independently variable system properties and to support reproducible coding; their empirical consequences are assessed only where matched comparisons are available. For comparative purposes, the retrieval-coupling axis also includes scheduled/static retrieval as an explicitly marked adjacent baseline that lies outside the RSA-LLM boundary. The remaining axes describe architectural design choices but do not independently fulfill RSA-LLM criteria. Learning signal is analyzed separately as the crosscutting dimension introduced in Section II-E [22].

Core or adjacent status is determined exclusively by the three membership conditions in Section I. Judge-, tree-, graph-, memory-, or tool-based mechanisms receive a core RSA label only when their evolving state changes subsequent external evidence acquisition or stopping. Hybrid systems may receive multiple architectural labels, while learning signal is coded separately from RSA membership.

## A. Axis 1: Retrieval Coupling

This axis describes how retrieval is invoked and controlled during inference: when the system searches, what it queries, and when it stops searching.

a) Adjacent reference point: scheduled/static retrieval: Scheduled/static retrieval occurs at predetermined points, with a fixed number of retrieval rounds or a predetermined top-k evidence set. Classic RAG retrieves passages before generation, while FiD strengthens evidence integration by jointly attending over multiple retrieved passages without introducing a statedependent retrieval controller [4], [21]. Even a multi-round pipeline remains scheduled when earlier observations do not alter later retrieval actions. Prior work on adaptive information seeking similarly distinguishes learned, evidence-conditioned retrieval policies from iterative strategies that repeatedly apply predefined retrieval operations [19]. Scheduled systems are retained on the coupling axis to provide a common reference point, but they do not satisfy the RSA-LLM membership criterion.

b) Adaptive coupling: the RSA boundary: Adaptive coupling begins at the RSA-LLM boundary. At least one retrieval trigger, query, source-selection, or stopping decision is conditioned on the evolving reasoning or evidence state. AISO, ReAct, Search-R1, and ReSearch instantiate this principle through learned or prompted policies that use previous reasoning and observations to determine subsequent actions [9], [18]– [20]. ZeroSearch reduces training cost by simulating retrieval during training while transferring to real search at test time [26] and StepSearch makes control more fine-grained through step-wise PPO with information-gain and redundancy-aware rewards [27]. Other systems in this regime include R1-Searcher and Search-o1, which likewise vary when to retrieve, how to use evidence, and when to stop [28], [29]. Recent search-andrefine systems introduce an explicit evidence-distillation state between retrieval rounds. AutoRefine and OThink-SRR1 filter or compress retrieved documents before continued reasoning, allowing the refined evidence state to influence subsequent querying, integration, and stopping decisions [30], [31].

c) Stateful adaptive coupling: Stateful coupling is a stronger form of adaptive coupling, rather than a mutually exclusive membership category, in which a persistent evidence, memory, or trajectory state conditions later retrieval and reasoning decisions. This changes coupling in a practical way: retrieval decisions become conditioned not only on the raw question and the latest snippets, but also on what has already been discovered and recorded [32], [33]. ReSum [32] addresses long-horizon search under limited context by periodically summarizing the interaction history into a compact state that can support further exploration without overflowing the context window. MemSearcher [33] similarly maintains and updates a compact memory after each search step, and trains the full loop end-to-end with RL. These designs improve stability in long search sessions, but they introduce new problems in evidence selection, compression, and drift: what gets written into the workspace can dominate later retrieval and reasoning, so pruning and update rules become first-class design choices [32], [33]. Related iterative RAG systems such as ReSP likewise make retrieval state explicit by recording and compressing the retrieval trajectory, reducing context overload while supporting continued multi-step evidence acquisition [34].

B. Axis 2: Controller Design (Where control lives, and how inference is organized)

Controller design specifies who runs the loop and how inference is structured. We decompose this axis into two complementary sub-axes.

a) Agency architecture (who runs the loop): Singleagent designs place planning, retrieval, verification, and answering inside one loop; ReAct is a canonical prompt-based instance, while Search-R1 and Search-o1 are representative agentic search variants [9], [18], [29]. Modular architectures separate roles such as planner, retriever, and verifier, improving controllability but requiring coordination across modules and interfaces [35]. VerlTool exemplifies this setting by standardizing tool interaction and supporting modular training and evaluation across multiple tools [36]. BrowseMaster provides a recent open-web example of modular control in which a planner adaptively delegates retrieval subtasks to a programmatic executor and replans from the executor’s distilled observations [37]. Multi-agent orchestration distributes reasoning across specialized agents, often improving robustness at the cost of latency and aggregation complexity [38], [39]. PRISM is a retrieval-centric example that splits question analysis, evidence filtering, and evidence expansion across agents [40].

b) Reasoning topology (how inference is structured): Linear controllers follow one chain of reasoning, optionally inserting retrieval and verification steps; they are efficient but vulnerable to early errors that propagate through the remaining trace [1], [41]. Branching controllers explore multiple candidate paths to hedge against such errors, as in Tree-of-Thoughts and MCTS-style search [41], [42]. Structured-state controllers maintain an explicit state representation such as a graph or belief state, improving traceability at the cost of additional state construction and pruning [43]. Think-on-Graph, Graph of-Thoughts-style methods, and GraphReader illustrate how graph-structured intermediate states can support controlled multi-hop reasoning and provenance tracking [13], [15], [44].

CoT, ToT, MCTS, and graph-based reasoning are included here as topology archetypes; they constitute RSA controllers only when their evolving states trigger external evidence acquisition.

## C. Axis 3: Knowledge Source

This axis specifies where external evidence resides and the constraints that the source imposes on acquisition, integration, and verification. Knowledge source alone does not determine RSA eligibility: a system qualifies only when observations obtained from the source influence subsequent reasoning or retrieval actions.

Open-world unstructured sources: The open web and large public corpora provide broad and potentially current coverage, but expose the controller to source volatility, redundancy, con flicting evidence, and variable authority. WebGPT exemplifies this regime because its browser actions are conditioned on the evolving interaction history [45]. Evaluation in this setting must therefore report the search service, access date, and retrieval budget, since changes in indexing or page availability can materially affect results.

Bounded or private collections: Enterprise repositories, domain specific collections, and fixed corpus snapshots permit stable indexing, access control, and more reproducible evaluation, but their coverage and freshness are bounded by corpus maintenance. Search-R1 illustrates how state-dependent queries can be executed against a fixed local retrieval stack [18]. Such environments enable controlled comparisons, although performance may remain sensitive to corpus construction, retriever configuration, and document version.

Structured environments: Databases, knowledge graphs, and evidence-bearing APIs expose typed operations rather than unconstrained text search. Controllers must select an appropriate interface, generate valid calls, interpret structured responses, and recover from execution failures. KnowTrace illustrates this regime by converting retrieved information into an explicit knowledge trace that supports subsequent retrieval and reasoning [46]. Purely computational tool calls remain outside the RSA boundary unless their outputs constitute external evidence used by later control decisions.

## D. Axis 4: Grounding Strictness

Grounding strictness characterizes how strongly a system constrains its output to retrieved evidence. Operationally, evidence may be merely available to the generator, encouraged through learning or revision signals, or required through an explicit support contract.

Best-effort grounding: Retrieved evidence informs generation, but individual claims need not be explicitly supported. Static RAG and FiD provide adjacent reference points for this regime: retrieved passages can improve answer quality, yet the generator may ignore evidence or introduce unsupported details [4], [21]. Best-effort grounding therefore provides neither claim-level attribution nor a prescribed response when sufficient evidence is unavailable.

Soft grounding: Unsupported outputs are discouraged through rewards, reflection, reranking, critique, or revision, but remain possible. Search-R1 and StepSearch optimize outcome- or retrieval-aware signals without imposing claim-level support constraints [18], [27]. Self-RAG uses reflection tokens to regulate retrieval and critique, whereas RARR retrieves evidence to revise potentially unsupported statements [47], [48]. These mechanisms improve evidence use while preserving generation flexibility.

Hard grounding: A hard-grounded system requires claims to be supported, cited, revised, or withheld when adequate evidence cannot be established. GopherCite approximates this regime through attributable quotations and abstention under insufficient support [49]. ALCE operationalizes complementary citation-coverage and citation-correctness criteria for evaluating attributable generation [50]. Hard grounding improves auditability, but may reduce answer coverage when retrieval fails or automated support judgments are unreliable.

## E. Cross-Cutting Dimension: Learning Signal

The four primary axes describe system architecture and inference-time behavior, whereas the learning signal describes how that behavior is elicited or optimized. These dimensions are architecturally separable: similar reasoning–search loops may be implemented through prompting, supervised trajectories, preference optimization, or reinforcement learning. Nevertheless, the learning signal is behaviorally consequential because it affects query formulation, source selection, evidence integration, stopping, and grounding. We therefore treat it as a cross-cutting analytic dimension rather than a fifth architectural axis.

We assign non-exclusive learning-signal tags to each reviewed system. Prompted or in-context control (P) elicits search behavior through instructions and demonstrations without parameter updates, as exemplified by ReAct [9]. Self-supervised tool induction (SS) derives training targets from modelgenerated or automatically filtered interactions, as in Toolformer [17]. Supervised fine-tuning or imitation (SFT) learns from curated reasoning, search, or tool-use trajectories. Preferencebased optimization (Pref) uses human or model preferences to favor helpful, attributable, or evidence-consistent behavior, as illustrated by WebGPT [45]. Outcome-level reinforcement learning (RL-O) optimizes final-task reward and can induce multi-turn search without supervised reasoning traces, as in Search-R1 and ReSearch [18], [20]. Process- or retrieval-aware reinforcement learning (RL-P) supplies intermediate signals such as information gain, evidence utility, redundancy penalties, or retrieval cost, as in StepSearch and R-Search [27], [51].

These signals introduce different control trade-offs. Prompting is inexpensive and portable but sensitive to model capability and prompt formulation. SFT and imitation stabilize action syntax and search trajectories but may reproduce demonstration biases and generalize poorly beyond observed workflows. Outcome-level RL permits strategy discovery but creates sparse credit-assignment problems because final correctness does not identify which retrieval actions were useful. Process-level rewards provide denser supervision, but their proxy metrics may reward excessive search, superficial evidence overlap, or other behavior that does not improve final grounding. Preference optimization can strengthen attribution and evidence adherence, although its behavior remains dependent on the coverage and reliability of the preference model.

Learning signal must therefore be reported when comparing RSA architectures. A performance difference attributed to retrieval coupling or controller design may instead arise from supervision quality, reward granularity, training data, or policy optimization. Accordingly, Table II records learning signal alongside the four architectural axes, thereby preventing differences in supervision or optimization from being conflated with differences in system architecture.

## III. BENCHMARKS AND EVALUATION

Evaluation of RSA-LLMs requires separating benchmark difficulty from RSA-loop necessity. A dataset may require substantial factual knowledge or multi-step reasoning without testing state-dependent evidence acquisition: standard RAG, pre retrieved documents, or a sufficiently long context may still be adequate. We therefore characterize benchmarks by the extent to which successful completion depends on adaptive querying, navigation, evidence integration, and stopping, in addition to summarizing the capability each benchmark measures.

## A. Benchmark Coverage and RSA-Loop Demand

Table I groups representative benchmarks into three levels of RSA-loop demand. Low-demand benchmarks primarily evaluate retrieval, answer generation, or verification and can generally be addressed using static retrieval. Moderate-demand benchmarks require distributed evidence or multi-hop reasoning, but their fixed corpora or provided contexts may permit predetermined retrieval or long-context inference. High-demand benchmarks make evidence discovery itself part of the task: subsequent queries, navigation actions, or source choices must typically be revised in response to previously observed evidence. Variable denotes heterogeneous suites in which RSA-loop necessity depends on the selected subset.

This distinction prevents benchmark performance from being interpreted as direct evidence for the value of adaptive coupling. Natural Questions, TriviaQA, and FEVER primarily test retrieval or verification quality rather than search control. HotpotQA, 2WikiMultiHopQA, MuSiQue, and FRAMES require evidence composition, but do not necessarily force statedependent acquisition when relevant documents can be retrieved in advance. By contrast, SEALQA, SGR-Bench, MERRIN, BrowseComp, and WebWalkerQA expose systems to noisy results, hidden retrieval states, multimodal sources, or evolving navigation paths for which later actions depend on earlier observations. Long-form suites such as DeepResearch Bench II additionally test whether retrieved evidence can support coherent, attributable synthesis.

## B. Metrics, Budgets, and Reproducibility

Final-answer correctness is necessary but insufficient because it does not identify whether failure arose from evidence acquisi tion, integration, grounding, or control. We therefore organize evaluation metrics into five complementary dimensions: (i)

answer quality, measured by exact match, token-level F1, task-specific accuracy, or rubric-based judgments; (ii) retrieval quality, measured by evidence recall, precision, Recall@k, or supporting-fact F1; (iii) grounding quality, measured by citation precision and recall, attribution coverage, and entailment between claims and cited evidence; (iv) control and efficiency, measured by search calls, pages visited, retrieved and generated tokens, latency, monetary cost, redundancy, and stopping behavior; and (v) robustness, measured under distractor evidence, conflicting sources, tool failures, source drift, and repeated stochastic runs.

For reproducibility, each reported result should specify the base model and learning signal; corpus snapshot or search API and access date; retriever, reranker, and retrieval depth; maximum search calls or pages; context and token budgets; available tools; prompting protocol; evaluator or judge model and version; number of trials; and variance or confidence intervals where available. Architectural comparisons should use matched search and token budgets or report quality–cost Pareto curves. Results obtained using fixed corpora and live-web environments should not be pooled, and EM, F1, citation scores, and judge-based scores should not be treated as interchangeable effect sizes. Generic function-calling benchmarks may be retained as adjacent diagnostics of interface reliability, but they validate RSA behavior only when the invoked tools return task-relevant external evidence [67], [68].

For the quantitative synthesis in Section IV-C, we retained only contrasts reported within the same primary study. A comparison was treated as a component ablation when the study held the model backbone, retrieval environment, evaluation set, and principal inference budget fixed while removing or modifying a specified controller, state, or learning component. Comparisons that matched the backbone and evaluation environment but jointly changed multiple components were recorded separately as multi-factor contrasts. For EM, F1, and accuracy, effects are expressed as absolute percentage-point differences; relative changes are retained only when the primary study did not report the corresponding baseline score. When a study evaluated the same metric over several datasets, we computed an unweighted macro-average over exactly those datasets. Efficiency effects are reported as relative reductions in retrieval calls, processed tokens, or latency. Because most studies omit repeated-run variance or confidence intervals, we do not pool effects across papers, infer statistical significance, or treat heterogeneous metrics as interchangeable effect sizes.

## IV. COMPARATIVE PERFORMANCE AND TRENDS

## A. System-Level Comparison

Table II maps representative systems onto the RSA boundary, the four architectural axes, and the cross-cutting learning signal. Adjacent references are shown separately and excluded from the descriptive counts below. The set is representative rather than exhaustive and supports taxonomy-level comparison, not performance ranking.

TABLE I  
REPRESENTATIVE BENCHMARKS AND THEIR DEMAND FOR AN ADAPTIVE REASONING–SEARCH LOOP. L, M, AND H DENOTE LOW, MODERATE, AND HIGH RSA-LOOP DEMAND; VAR. DENOTES ITEM-DEPENDENT DEMAND.
<table><tr><td>Setting</td><td>Benchmark</td><td>Primary capability</td><td>Demand</td><td>Main limitation</td></tr><tr><td>Fact lookup</td><td>NQ; TriviaQA [52], [53]</td><td>Open-domain retrieval and short-answer generation</td><td></td><td>Often solvable from one passage; static corpora and con- tamination limit evaluation of search control.</td></tr><tr><td>Multi-hop QA</td><td>HotpotQA; MuSiQue; 2Wiki- MultiHopQA [54]–[56]</td><td>Cross-document evidence composition and M supporting-fact reasoning</td><td></td><td>Fixed or provided contexts permit pre-retrieval or long- context inference; EM/F1 obscures retrieval control.</td></tr><tr><td>End-to-end RAG</td><td>FRAMES [57]</td><td>Factuality, retrieval, and multi-constraint M reasoning</td><td></td><td>Small, Wikipedia-centered set that does not isolate controller behavior.</td></tr><tr><td>Verification</td><td>FEVER [58]</td><td>Evidence retrieval, claim support, and ve- L racity classification</td><td></td><td>Static Wikipedia and synthetic claims allow fixed retrieval without adaptive search.</td></tr><tr><td>Noisy search</td><td>SEALQA [59]</td><td>Ambiguous, conflicting, time-sensitive, and H distractor-heavy evidence</td><td></td><td>Small, version-sensitive set; LongSeal tests selection from supplied documents rather than active acquisition.</td></tr><tr><td>State-gated re- trieval</td><td>SGR-Bench [60]</td><td>Configuring filters, hierarchies, and scopes H to expose answer-bearing evidence</td><td></td><td>Only 100 tasks; preprint status and changing interfaces limit stability.</td></tr><tr><td>Multimodal web</td><td>MERRIN [61]</td><td>Multimodal evidence retrieval and reason- H ing under noisy or conflicting sources</td><td></td><td>High tool and media costs plus changing web content limit reproducibility.</td></tr><tr><td>General assis- tants</td><td>GAIA [62]</td><td>Planning, browsing, multimodal reasoning, Var. and tool use</td><td></td><td>Mixes search with non-evidential tools; subset choice affects</td></tr><tr><td>Deep browsing</td><td>ZH [63], [64]</td><td>BrowseComp; BrowseComp- Persistent search over entangled English H and Chinese web clues</td><td></td><td>comparisons. Accuracy omits evidence and citation quality; web drift and leakage remain concerns.</td></tr><tr><td>Web traversal</td><td>WebWalkerQA [65]</td><td>Hierarchical website navigation, evidence H accumulation, and stopping</td><td></td><td>Changing sites reduce reproducibility; scores conflate navi- gation, retrieval, and generation.</td></tr><tr><td>thesis</td><td>Research syn- DeepResearch Bench II [66]</td><td>Long-form evidence recall, analysis, and H report presentation</td><td></td><td>Only 132 costly tasks; rubric scores conflate search, analysis, and writing.</td></tr></table>

TABLE II

REPRESENTATIVE SYSTEMS ACROSS FIVE DESIGN DIMENSIONS. ADJACENT REFERENCE SYSTEMS ARE EXCLUDED FROM CROSS-SYSTEM COUNTS.
<table><tr><td>Method</td><td>Coupling</td><td>Controller</td><td>Knowledge source / retriever</td><td>Ground.</td><td>Learn.</td></tr><tr><td colspan="6">Panel A: Adjacent reference systems</td></tr><tr><td>RAG / FiD [4], [21]</td><td>Sch.</td><td>Single / linear or fusion</td><td>Wikipedia; DPR</td><td>BE</td><td>SFT</td></tr><tr><td>Feedback-space planning [23]</td><td>Ad.</td><td>Search over candidate</td><td>VAL and landmark feedback</td><td>N/A</td><td>P</td></tr><tr><td>GopherCite [49]</td><td>Sch.</td><td>models Single / rerank</td><td>Open web; Google Search top-K</td><td>Hard</td><td>SFT+Pref</td></tr><tr><td colspan="6">Panel B: Core RSA-LLMs</td></tr><tr><td>ReAct [9]</td><td>Ad.</td><td>Single / linear</td><td>Wikipedia API</td><td>BE</td><td>P</td></tr><tr><td>IRCoT [69]</td><td>Ad.</td><td>Single / linear</td><td>Task-specific Wikipedia corpora; BM25</td><td>BE</td><td>P</td></tr><tr><td>Self-RAG [47]</td><td>Ad.</td><td>Single / linear</td><td>Contriever and web search; top-5</td><td>Soft</td><td>SFT</td></tr><tr><td>WebGPT [45]</td><td>Ad.</td><td>Single / linear</td><td>Open-web browser</td><td>Soft</td><td>SFT+Pref</td></tr><tr><td>ReKG-MČTS [14]</td><td>St.</td><td>Single / tree</td><td>Freebase knowledge graph</td><td>Soft</td><td>P</td></tr><tr><td>Search-R1 [18]</td><td>Ad.</td><td>Single / linear</td><td>Wikipedia-2018; E5</td><td>Soft</td><td>RL-O</td></tr><tr><td>ReSearch [20]</td><td>Ad.</td><td>Single / linear</td><td>Wikipedia-2018; E5</td><td>Soft</td><td>RL-O</td></tr><tr><td>StepSearch [27]</td><td>Ad.</td><td>Single / linear</td><td>Wikipedia-2018; E5</td><td>Soft</td><td>RL-P</td></tr><tr><td>AutoRefine [30]</td><td>St.</td><td>Single / linear-refine</td><td>Wikipedia-2018; E5</td><td>Soft</td><td>RL-P</td></tr><tr><td>S2G-RAG [70]</td><td>St.</td><td>Modular / linear</td><td>Dataset Wikipedia; BM25 or E5</td><td>Soft</td><td>SFT</td></tr><tr><td>OThink-SRR1 [31]</td><td>St.</td><td>Single / linear-refine</td><td>Wikipedia-2023; Elasticsearch</td><td>Soft</td><td>RL-P</td></tr><tr><td>BrowseMaster [37]</td><td>St.</td><td>Modular / hierarchical</td><td>Open web; programmatic tools</td><td>BE</td><td>P</td></tr><tr><td>PiCA [71]</td><td>Ad.</td><td>Single / linear</td><td>Wikipedia-2018; E5</td><td>Soft</td><td>RL-P</td></tr></table>

Notation: Sch. = scheduled/static; Ad. = adaptive; St. = stateful adaptive; BE = best-effort grounding; P = prompted or training-free control; Pref = preference optimization; RL-O = outcome-level reinforcement learning; RL-P = process- or retrieval-aware reinforcement learning;

The mapping confirms that retrieval coupling and learning signal are analytically distinct: adaptive or stateful acquisition appears under prompting, supervised control, preference feedback, and both outcome- and process-aware reinforcement learning [9], [18], [27], [45], [70]. Controller comparisons must therefore account for differences in supervision, reward design, and optimization. Feedback-space planning is retained only as an adjacent boundary case because it adaptively optimizes a symbolic artifact from validator feedback rather than acquiring external task evidence [23].

## B. Cross-System Trends

Among the 13 core systems, ten use single-controller linear or linear-refinement topologies, ten operate over fixed corpora or knowledge graphs, five maintain an explicit stateful retrieval or evidence representation, and six use outcome- or processaware reinforcement learning. All 13 employ best-effort or soft grounding; GopherCite provides the hard-grounded adjacent reference. These descriptive counts should not be interpreted as bibliometric prevalence estimates.

Trends supported across several systems include the separation of adaptive retrieval from reinforcement learning, the movement from transient retrieval observations toward persistent evidence states, and increasing use of process-aware rewards for search control. By contrast, conclusions about branching controllers or planner–executor architectures remain setting-specific: ReKG-MCTS demonstrates their utility over knowledge-graph paths, while BrowseMaster does so in longhorizon open-web search [14], [37]. Such observations motivate controlled comparisons but do not, by themselves, establish that one controller family is generally superior.

## C. Controlled Quantitative Synthesis

Direct pooling of published RSA-LLM scores would confound model scale, training data, retriever quality, corpus snapshot, benchmark subset, and search budget. Table III therefore reports only controlled component ablations and explicitly identified matched comparisons extracted from the original studies. This design converts the comparative discussion from a cross-paper leaderboard into a set of localized effect estimates: each delta answers whether a particular component mattered under the study’s own fixed evaluation conditions.

Figure 2 makes the direction, magnitude, and heterogeneity of these reported changes visible without converting them into standardized effect sizes. Three conclusions emerge. First, acquisition control and evidence-state management are consequential when isolated: the S2G-RAG judge and the interacting BrowseMaster planner–primitive modules produce some of the largest reported gains [37], [70]. Second, retrievedtoken masking, step-wise rewards, retrieval-aware rewards, and pivot-based credit assignment generally improve otherwise matched RSA loops [18], [27], [30], [71]. Third, stateful evidence compression can improve the quality–cost trade-off, as shown by S2G-RAG’s lower latency and OThink-SRR1’s reductions in retrieval calls and input tokens [31], [70].

The evidence remains uneven across the taxonomy. The available ablations primarily test retrieval coupling, controller design, evidence-state management, and learning signal; we found no eligible matched comparison that independently isolates knowledge source or grounding strictness. Accordingly, the results provide construct-level evidence that several taxonomy dimensions identify consequential design decisions, but they do not establish that the four axes are sufficient or maximally predictive of RSA-LLM performance.

## V. CHALLENGES AND FUTURE DIRECTIONS

The proposed taxonomy exposes four complementary research gaps that are obscured when RSA-LLMs are evaluated as a single architectural family.

Axis 1: Retrieval coupling. Learned search policies remain computationally expensive, and additional retrieval does not necessarily produce useful evidence. Recent work explores simulated retrieval and information-foraging objectives [26], [72], but reliable action-level credit assignment remains unresolved. Retrieval should be optimized as budgeted evidence acquisition, using calibrated stopping criteria and marginal evidence utility relative to search calls, tokens, latency, and monetary cost. Evaluation should consequently report quality– cost Pareto frontiers under matched budgets.

Axis 2: Controller design. Controller state and coordination logic are frequently embedded in prompts or heuristics, limiting reproducibility and diagnosis. Future systems should expose explicit representations of uncertainty, accumulated evidence, and remaining budget, together with typed interfaces among planner, retriever, integrator, and verifier components. Treeand graph-based controllers additionally require reproducible pruning, backtracking, and stopping rules. Decision traces

Search-R1: mask (7B) Search-R1: mask (3B) StepSearch: reward AutoRefine: reward (Base) AutoRefine: reward (Instr.) AutoRefine: refine (Base) AutoRefine: refine (Instr.) S2G-RAG: trained judge S2G-RAG: judge present S2G-RAG: extractor S2G-RAG: gap query   
BrowseMaster: planner (-prim.)   
BrowseMaster: planner (+prim.) BrowseMaster: prim. (-plan) BrowseMaster: prim. (+plan) BrowseMaster: full OThink vs ReSearch (7B) OThink vs ReSearch (3B)

A EM / accuracy / success rate  
![](images/f33d29ff7c635b5bc82e85c85e3abf46afd7da040d47c7b7501a65e36245ff8e.jpg)

B F1  
![](images/3fd88897ac182cf1589cc1eabf31589b056278a2618f22a6e1f8ebd9b1f72d8b.jpg)

![](images/6551a092d44c0e35e18e719d4f93f96dd4a1be4a7d77296e9127a97d5ad421de.jpg)  
Fig. 2. Reported within-study deltas from Table III: (A) EM, accuracy, and success-rate; (B) F1 changes; and (C) resource reductions. Points denote individual results or medians across benchmarks; while whiskers show benchmark ranges, not confidence intervals. Marker style distinguishes single component (S), conditional (C), and multifactor (M) contrasts. Results are not pooled across panels.

and failure logs are particularly important in high-stakes deployments.

Axis 3: Knowledge source. Comparisons are confounded when systems use different corpora, search APIs, snapshots, or access policies. Evaluations should record these conditions and compare matched controllers across bounded and open-world sources. Unified interfaces are also needed for heterogeneous evidence spanning text, tables, databases, knowledge graphs, and APIs, together with robustness tests for source drift, missing fields, and retrieval-service failure.

TABLE III  
CONTROLLED QUANTITATIVE EVIDENCE. QUALITY ENTRIES GIVE MEDIAN ABSOLUTE CHANGE [MIN, MAX], POSITIVE-EFFECT COUNTS, AND k<sub>Q</sub> BENCHMARK–METRIC OBSERVATIONS. S, C, M, AND A DENOTE SINGLE-COMPONENT, CONDITIONAL, MULTIFACTOR, AND AGGREGATE-ONLY EVIDENCE; STUDIES ARE NOT POOLED.
<table><tr><td>Study</td><td>Controlled contrast</td><td>Type</td><td> $k _ { Q }$ </td><td>Reported quality change</td><td>Efficiency change</td><td>Source</td></tr><tr><td colspan="7">Panel A: Single-component and module-level evidence</td></tr><tr><td>Search-R1 [18]</td><td>Retrieved-token masking</td><td>S</td><td>14</td><td> $^ { 7 \mathrm { B } \mathrm { E M : } + 8 . 8 \left[ + 6 . 1 , + 1 2 . 8 \right] , 7 / 7 } \uparrow ; 3 \mathrm { B } \mathrm { E M : } + 4 . 3 \left[ - 1 . 6 , + 1 0 . 3 \right] , 5 / 7 \uparrow$ </td><td>Not reported</td><td>T6</td></tr><tr><td>StepSearch [27] AutoRefine</td><td>Step-wise process reward Retrieval-specific reward</td><td>S S</td><td>8</td><td> $\mathrm { E M : } + 2 . 4 5 [ ( + 1 . 9 , + 8 . 3 ] , 4 / 4 \uparrow ; \mathrm { F l } ; + 2 . 9 5 [ + 1 . 7 , + 8 . 8 ] , 4 / 4 \uparrow$   $\mathrm { B a s e E M : + 3 . 7 \ [ 0 . 0 , + 4 . 4 ] , 6 / 7 \ \uparrow ; I n s t r u c t { E M : + 4 . 5 \ [ + 1 . 0 , + 6 . 4 ] , 7 / 7 \uparrow } }$ </td><td>Not reported Not reported</td><td>T4</td></tr><tr><td>[30]</td><td></td><td></td><td>14</td><td></td><td></td><td>T2</td></tr><tr><td>AutoRefine [30]</td><td>Refinement step under no re- trieval reward</td><td>S</td><td>14</td><td> $\mathrm { B a s e ~ E M : } + 7 . 4 \left[ - 0 . 2 , + 2 0 . 0 \right] , 6 / 7 \uparrow ; \mathrm { I n s t r u c t ~ E M : } + 2 . 3 \left[ + 0 . 7 , + 6 . 2 \right] , 7 / 7 \nonumber$  ↑</td><td>Not reported</td><td>T2</td></tr><tr><td>S2G-RAG [70]</td><td>Fine-tuned versus untrained judge</td><td>S</td><td></td><td> $\mathrm { \dot { H o t p o t Q A : } + 4 . 1 E M ; + 5 . 7 F 1 }$ </td><td>Not reported</td><td>T2</td></tr><tr><td>S2G-RAG [70]</td><td>S2Ġ-Judge versus no judge</td><td>S</td><td></td><td> $_ { \mathrm { H o t p o t Q A : + 1 5 . 8 E M : + 1 8 . 9 F 1 } }$ </td><td>Not reported</td><td>T2</td></tr><tr><td>S2G-RAG [70]</td><td>Evidence extractor versus raw accumulation</td><td>S</td><td></td><td> $\mathrm { H o t p o t \bar { Q } A : + 3 . 8 \ E M ; + 4 . 0 \ F 1 }$ </td><td>Latency ↓ 17.7%</td><td>T2-3</td></tr><tr><td>S2G-RAG [70]</td><td>Structured versus free-text gap query</td><td>S</td><td></td><td> $\mathrm { H o t p o t Q A : } + 1 . 8 \mathrm { E M : } + 2 . 0 \mathrm { F 1 ; + 2 . 2 }$  correct retrieval</td><td>Not reported</td><td>T6</td></tr><tr><td colspan="7">Panel B: Conditional and multifactor architectural evidence</td></tr><tr><td>BrowseMaster [37]</td><td>Planner contribution</td><td>C</td><td>2</td><td>BrowseComp: +1.5 without primitives; +15.0 with primitives</td><td>Not reported</td><td>T2</td></tr><tr><td>BrowseMaster [37]</td><td>Programmatic-primitives contribution</td><td>C</td><td>2</td><td>BrowseComp: +5.5 without planner; +19.0 with planner</td><td>Not reported</td><td>T2</td></tr><tr><td>BrowseMaster [37]</td><td>Full planner-primitive sys- tem</td><td>M</td><td></td><td>BrowseComp: +20.5 accuracy</td><td>Not reported</td><td>T2</td></tr><tr><td>OThink-SRR1 [31]</td><td>Search-refine-reason versus ReSearch</td><td>M</td><td>16</td><td> $7 { \mathrm { B } } \colon { \mathrm { E M } } + 1 . 2 6 \ [ - 0 . 5 2 , + 1 . 6 6 ] , 3 / 4 \ \uparrow ; { \mathrm { F } } 1 + 0 . 3 6 \ [ - 0 . 8 0 , + 1 . 1 9 ] , 3 / 4 \ \uparrow .$   $3 \mathrm { B } ; \mathrm { E M } + 3 . 8 4 [ + 2 . 5 3 , + 8 . 4 5 ] , 4 / 4 \uparrow ; \mathrm { F } 1 + 5 . 8 8 [ + 4 . 3 0 , + 1 1 . 3 2 ] , 4 / 4 \uparrow$ </td><td>7B calls ↓ 31.2% [28.0,35.5]; tokens ↓</td><td>T1/F4</td></tr><tr><td colspan="7">Panel C: Aggregate-only evidence</td></tr><tr><td>PiCA [71]</td><td>Pivot-based versus outcome- only credit</td><td>A</td><td></td><td>Across four model families: relative EM +6.6% to +34.0%; relative F1 Not reported +4.1% to +29.1%</td><td></td><td>T3</td></tr></table>

Notes: Brackets denote the range across benchmarks, not confidence intervals. For BrowseMaster, component effects are conditional on whether the complementary component is present and therefore should not be treated as additive. OThink-SRR1 changes refinement architecture and reward design jointly. PiCA reports aggregate relative improvements; its benchmark-level outcome-only baselines are unavailable and it is excluded from benchmark-level effect plots. T and F identify the source table and figure, respectively.

Axis 4: Grounding strictness. Retrieval does not guarantee that generated claims are supported. Future systems should enforce claim-level grounding contracts that require attribution, revision, or abstention when support is insufficient. Evaluation should jointly report citation precision and recall, attribution coverage, unsupported-claim rate, and selective risk [50]. Because automated verification is itself fallible, adversarial testing and targeted human auditing remain necessary.

## VI. CONCLUSION

In this paper, we defined RSA-LLMs as systems that embed state-dependent external evidence acquisition within a closed reasoning–control loop, while treating scheduled retrieval as an adjacent reference point. We introduced four architectural axes: retrieval coupling, controller design, knowledge source, and grounding strictness. We also separated these properties from the learning signals used to optimize system behavior. Beyond descriptive system coding, we synthesized matched within-study ablations and quality–cost contrasts. The available evidence shows that explicit acquisition control, persistent evi dence management, and process-aware learning can materially affect answer quality and retrieval efficiency under controlled settings. However, empirical coverage remains uneven for knowledge-source and grounding interventions; the taxonomy should therefore be interpreted as an operational comparison framework rather than a statistically complete predictor of performance. This distinction provides a more reproducible basis for evaluating future RSA systems and identifying where genuinely controlled evidence is still required.

## REFERENCES

[1] J. Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” in NeurIPS, 2022.

[2] T. Kojima et al., “Large language models are zero-shot reasoners,” in NeurIPS, 2022.

[3] F. Petroni, T. Rocktaschel, S. Riedel, P. Lewis, A. Bakhtin, Y. Wu, ¨ and A. Miller, “Language models as knowledge bases?” 2019, pp. 2463–2473. [Online]. Available: https://aclanthology.org/D19-1250/

[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel, S. Riedel, and D. Kiela,¨ “Retrieval-augmented generation for knowledge-intensive nlp tasks,” in NeurIPS, 2020.

[5] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the middle: How language models use long contexts,” TACL, vol. 12, pp. 157–173, 2024.

[6] S. Lin, J. Hilton, and O. Evans, “Truthfulqa: Measuring how models mimic human falsehoods,” in ACL, 2022, pp. 3214–3252.

[7] X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou, “Self-consistency improves chain-of-thought reasoning in language models,” in ICLR, 2023.

[8] M. Besta et al., “Demystifying chains, trees, and graphs of thoughts,” IEEE TPAMI, 2025.

[9] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in ICLR, 2023.

[10] O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis, “Measuring and narrowing the compositionality gap in language models,” in Findings of EMNLP, 2023, pp. 5687–5711.

[11] W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li, “A survey on rag meeting llms: Towards retrieval-augmented large language models,” 2024.

[12] S. Barnett et al., “Seven failure points when engineering a retrievalaugmented generation system,” in IEEE/ACM CAIN, 2024, pp. 194–199.

[13] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y. Gong, L. Ni, H.-Y. Shum, and J. Guo, “Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph,” in ICLR, 2024.

[14] X. Song, S. Zhang, and T. Yu, “ReKG-MCTS: Reinforcing LLM reasoning on knowledge graphs via training-free Monte Carlo tree search,” in Findings of ACL, 2025, pp. 9288–9306.

[15] S. Li, Y. He, H. Guo et al., “Graphreader: Building graph-based agent to enhance long-context abilities of large language models,” 2024.

[16] E. Karpas et al., “MRKL systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning,” arXiv:2205.00445, 2022.

[17] T. Schick, J. Dwivedi-Yu, R. Dess\`ı et al., “Toolformer: Language models can teach themselves to use tools,” 2023.

[18] B. Jin, H. Zeng, Z. Yue, J. Yoon, S. O. Arik, D. Wang, H. Zamani, and J. Han, “Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning,” in COLM, 2025.

[19] Y. Zhu, L. Pang, Y. Lan, H. Shen, and X. Cheng, “Adaptive information seeking for open-domain question answering,” in EMNLP, 2021.

[20] M. Chen, L. Sun, T. Li, H. Sun, Y. Zhou, C. Zhu, H. Wang, J. Z. Pan, W. Zhang, H. Chen, F. Yang, Z. Zhou, and W. Chen, “Research: Learning to reason with search for LLMs via reinforcement learning,” in NeurIPS, vol. 38, 2025.

[21] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open-domain question answering,” in EACL, 2021.

[22] Y. Li, W. Zhang, Y. Yang et al., “A survey of RAG-reasoning systems in large language models,” in Findings of EMNLP, 2025.

[23] J. Oswald, D. Obolensky, V. Varha, V. Dragovic, K. Srinivas, H. Kokel, M. Katz, and S. Sohrabi, “Model space reasoning as search in feedback space for planning domain generation,” in ICLR Workshop on World Models, 2026.

[24] A. Singh, A. Ehtesham, S. Kumar, T. T. Khoei, and A. V. Vasilakos, “Agentic retrieval-augmented generation: A survey on agentic RAG,” arXiv preprint arXiv:2501.09136, 2025.

[25] Y. Qin, S. Hu, Y. Lin et al., “Tool learning with foundation models,” ACM Computing Surveys, vol. 57, no. 4, pp. 1–40, 2024.

[26] H. Sun et al., “ZeroSearch: Incentivize the search capability of LLMs without searching,” arXiv preprint arXiv:2505.04588, 2025.

[27] X. Zheng, K. An, Z. Wang, Y. Wang, and Y. Wu, “Stepsearch: Igniting LLMs search ability via step-wise proximal policy optimization,” in EMNLP, 2025, pp. 21 805–21 830.

[28] H. Song et al., “R1-searcher: Incentivizing the search capability in LLMs via reinforcement learning,” 2025.

[29] X. Li, G. Dong, J. Jin, Y. Zhang, Y. Zhou, Y. Zhu, P. Zhang, and Z. Dou, “Search-o1: Agentic search-enhanced large reasoning models,” 2025.

[30] Y. Shi, S. Li, C. Wu, Z. Liu, J. Fang, H. Cai, A. Zhang, and X. Wang, “Search and refine during think: Facilitating knowledge refinement for improved retrieval-augmented reasoning,” in NeurIPS, vol. 38, 2025.

[31] H. Liang, Z. Niu, J. Wu, C. Zhang, W. Zhou, and J. Wang, “OThink-SRR1: Search, refine and reasoning with reinforced learning for large language models,” arXiv preprint arXiv:2604.19766, 2026.

[32] X. Wu et al., “Resum: Unlocking long-horizon search intelligence via context summarization,” 2025.

[33] Q. Yuan et al., “Memsearcher: Training LLMs to reason, search and manage memory via end-to-end reinforcement learning,” 2025.

[34] Z. Jiang, M. Sun, L. Liang, and Z. Zhang, “Retrieve, summarize, plan: Advancing multi-hop question answering with an iterative approach,” in Companion Proc. WWW. ACM, 2025, p. 1677–1686. [Online]. Available: https://doi.org/10.1145/3701716.3716889

[35] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “HuggingGPT: Solving AI tasks with ChatGPT and its friends in hugging face,” in NeurIPS, 2023.

[36] D. Jiang et al., “Verltool: Towards holistic agentic reinforcement learning with tool use,” 2025.

[37] X. Pang, S. Tang, R. Ye, Y. Du, Y. Du, and S. Chen, “BrowseMaster: Towards scalable web browsing via tool-augmented programmatic agent pair,” arXiv preprint arXiv:2508.09129, 2025.

[38] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, “Improving factuality and reasoning in language models through multiagent debate,” arXiv preprint arXiv:2305.14325, 2023.

[39] T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, S. Shi, and Z. Tu, “Encouraging divergent thinking in large language models through multi-agent debate,” in EMNLP, 2024, pp. 17 889–17 904. [Online]. Available: https://aclanthology.org/2024.emnlp-main.992/

[40] M. M. H. Nahid and D. Rafiei, “Prism: Agentic retrieval with llms for multi-hop question answering,” arXiv preprint arXiv:2510.14278, 2025.

[41] S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan, “Tree of thoughts: Deliberate problem solving with large language models,” 2023.

[42] C. B. Browne et al., “A survey of monte carlo tree search methods,” IEEE Trans. Comput. Intell. AI Games, vol. 4, no. 1, pp. 1–43, 2012.

[43] M. Besta et al., “Graph of thoughts: Solving elaborate problems with large language models,” vol. 38, no. 16, pp. 17 682–17 690, 2024.

[44] B. Jin, C. Xie, J. Zhang, K. K. Roy, Y. Zhang, Z. Li, R. Li, X. Tang, S. Wang, Y. Meng, and J. Han, “Graph chain-of-thought: Augmenting large language models by reasoning on graphs,” in Findings of ACL, 2024, pp. 163–184.

[45] R. Nakano, J. Hilton, S. Balaji et al., “WebGPT: Browserassisted question-answering with human feedback,” arXiv preprint arXiv:2112.09332, 2021.

[46] R. Li, Q. Dai, Z. Zhang, X. Chen, Z. Dong, and J.-R. Wen, “Knowtrace: Bootstrapping iterative retrieval-augmented generation with structured knowledge tracing,” arXiv preprint arXiv:2505.20245, 2025.

[47] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning to retrieve, generate, and critique through self-reflection,” in ICLR, 2024. [Online]. Available: https://openreview.net/forum?id=hSyW5go0v8

[48] L. Gao, Z. Dai, P. Pasupat, A. Chen, A. T. Chaganty, Y. Fan, V. Y. Zhao, N. Lao, H. Lee, D.-C. Juan, and K. Guu, “Rarr: Researching and revising what language models say, using language models,” in ACL, 2023.

[49] J. Menick et al., “Teaching language models to support answers with verified quotes,” arXiv preprint arXiv:2203.11147, 2022.

[50] T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to generate text with citations,” in EMNLP, 2023.

[51] Q. Zhao, R. Wang, D. Xu, D. Zha, and L. Liu, “R-search: Empowering LLM reasoning with search via multi-reward reinforcement learning,” 2025.

[52] T. Kwiatkowski, J. Palomaki, O. Redfield et al., “Natural questions: A benchmark for question answering research,” TACL, vol. 7, pp. 453–466, 2019.

[53] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension,” in ACL, 2017, pp. 1601–1611.

[54] Z. Yang et al., “Hotpotqa: A dataset for diverse, explainable multi-hop question answering,” in EMNLP, 2018, pp. 2369–2380.

[55] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “MuSiQue: Multihop questions via single-hop question composition,” TACL, vol. 10, pp. 539–554, 2022.

[56] X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps,” in COLING, 2020, pp. 6609–6625.

[57] S. Krishna, K. Krishna, A. Mohananey, S. Schwarcz, A. Stambler, S. Upadhyay, and M. Faruqui, “Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation,” in NAACL-HLT, 2025, pp. 4745–4759.

[58] J. Thorne et al., “FEVER: a large-scale dataset for fact extraction and VERification,” in NAACL, 2018, pp. 809–819.

[59] T. Pham, N. Nguyen, P. Zunjare, W. Chen, Y.-M. Tseng, and T. Vu, “SEALQA: Raising the bar for reasoning in search-augmented language models,” in ICLR, 2026.

[60] N. Li, H. Shen, M. Liu, Y. Han, Z. Shi, S. Xie, and Y. Ma, “SGR-bench: Benchmarking search agents on state-gated retrieval,” arXiv preprint arXiv:2605.22219, 2026.

[61] H. Wang, D. Wan, H. Lee, T. Pham, M. Cankosyan, W. Chen, E. Stengel-Eskin, T. Vu, and M. Bansal, “MERRIN: A benchmark for multimodal evidence retrieval and reasoning in noisy web environments,” arXiv preprint arXiv:2604.13418, 2026.

[62] G. Mialon, C. Fourrier, T. Wolf, Y. LeCun, and T. Scialom, “GAIA: a benchmark for general AI assistants,” in ICLR, 2024.

[63] J. Wei, Z. Sun, S. Papay et al., “Browsecomp: A simple yet challenging benchmark for browsing agents,” arXiv:2504.12516, 2025.

[64] P. Zhou, B. Leon, X. Ying, C. Zhang, Y. Shao, Q. Ye, D. Chong, Z. Jin, C. Xie, M. Cao, Y. Gu, S. Hong, J. Ren, J. Chen, C. Liu, and Y. Hua, “BrowseComp-ZH: Benchmarking web browsing ability of large language models in chinese,” arXiv preprint arXiv:2504.19314, 2025.

[65] J. Wu, W. Yin, Y. Jiang, Z. Wang, Z. Xi, R. Fang, L. Zhang, Y. He, D. Zhou, P. Xie, and F. Huang, “Webwalker: Benchmarking LLMs in web traversal,” in ACL, 2025, pp. 10 290–10 305.

[66] R. Li, M. Du, B. Xu, C. Zhu, X. Wang, and Z. Mao, “Deepresearch bench ii: Diagnosing deep research agents via rubrics from expert report,” arXiv preprint arXiv:2601.08536, 2026.

[67] M. Li, Y. Zhao, B. Yu, F. Song, H. Li, H. Yu, Z. Li, F. Huang, and Y. Li, “API-bank: A comprehensive benchmark for tool-augmented LLMs,” in EMNLP, 2023, pp. 3102–3116.

[68] S. G. Patil, H. Mao, F. Yan, C. C.-J. Ji, V. Suresh, I. Stoica, and J. E. Gonzalez, “The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models,” in ICML, 2025.

[69] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multistep questions,” in ACL, 2023, pp. 10 014–10 037.

[70] M. Li, J. Zou, X. Lv, C. Zhang, and G. Zhou, “S2G-RAG: Structured sufficiency and gap judging for iterative retrieval-augmented QA,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 25 846– 25 862. [Online]. Available: https://aclanthology.org/2026.acl-long.1185/

[71] D. Liu, Y. Niu, Q. Wang, H. Xiao, and J. Li, “PiCA: Pivot-based credit assignment for search agentic reinforcement learning,” arXiv preprint arXiv:2605.09287, 2026.

[72] H. Qian and Z. Liu, “Scent of knowledge: Optimizing search-enhanced reasoning with information foraging,” in NeurIPS, 2025.