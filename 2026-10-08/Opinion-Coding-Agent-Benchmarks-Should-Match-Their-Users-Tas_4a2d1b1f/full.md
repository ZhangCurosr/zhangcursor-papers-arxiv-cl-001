SWE-Bench Pro

# Opinion: Coding-Agent Benchmarks Should Match Their Users’ Task Flows

Igor Slinko<sup>1</sup> Yaroslav Golubev<sup>1</sup> Sergey Titov<sup>1</sup> <sup>1</sup>JetBrains Research

## Abstract

The evaluation of coding agents generally strives to be as realistic as possible. In our study, we collect 4,782 agent sessions of real software engineers in JetBrains IDEs, which we call Production Sessions. Since our subject is interactive agents, we study the sessions with at least three user messages (33% of the sample). These long sessions differ from issue-derived benchmark tasks in two ways: (i) user requests span a far wider mix of task types—questions about the project’s code, planning, review, refactoring, execution—and (ii) users switch between types throughout a session. Long-session samples from three public interaction corpora exhibit markedly different Task Flows (the distributions of session lengths, task types, and type-to-type transitions), so no single interaction distribution is universally realistic: benchmarks should name a target use case and calibrate to measurements from it. We present SWE-TaskFlow, an approach for transforming any issue-derived benchmark: it preserves the verified tasks and tests while steering the interaction toward a target Task Flow through prompt splitting and verifiable repository QA, with a TaskFlow Alignment Score (TFAS) for selecting among generated trajectories. In a pilot on 700 SWE-Bench Pro tasks, solving the task sequentially in several steps approximately doubles agent cost without a stable change in resolve rate: the interaction protocol itself is an important dimension of evaluation.

## Production Sessions

![](images/54255cb2e2b17fc646c9c0ab5c490f5f3e1d362bd79e16146022aea30a711acb.jpg)  
Figure 1: Left: an issue-derived benchmark (SWE-Bench Pro) presents each task as a single user turn, almost always a new feature or a bug fix. Right: Task Flow of Production Sessions that contain at least three messages; nodes are primary intents and link widths are transition frequencies.

## 1 Introduction

The field of agent evaluation aims to assess agents in realistic scenarios. To see what such a scenario might look like, we start our study by getting access to 4,782 agent sessions of real software engineers working in JetBrains IDEs; we call them Production Sessions. In this data, we focus on the longinteraction slice: sessions with at least three user messages (accounting for as much as 33% of the sample). We describe such a corpus by its Task Flow: session lengths, the task types of user messages, and the transitions between types from one message to the next (formal definition in Section 2). Two observations about the Task Flow of these sessions motivate this work (Figure 1).

First, user prompts arefar more varied than benchmark tasks. In SWE-Bench Pro [Deng et al., 2026] (left panel of Figure 1), new features (54.7%) and bug fixes (39.1%) account for 93.8% of tasks (labels from Section 2); in Production Sessions the same two intents cover well under half of user messages, and the remainder asks the agent to explain the project’s code, answer questions, plan, review, refactor, and execute. Second, users switch task types throughout a session: one that opens with a feature request often continues with a question about existing behavior, a planning step, and a review request. Issue-derived benchmarks ignore this: SWE-bench [Jimenez et al., 2024] and successors such as SWE-Bench Pro, SWE-rebench, and DeepSWE [Deng et al., 2026, Badertdinov et al., 2025, Huang et al., 2026] deliver one detailed issue and score one final patch. Their infrastructure and tests are valuable, but their interaction layer—the user messages the agent receives and the order in which it receives them—measures implementation from an upfront specification rather than adaptation to requirements unfolding across a session.

Which types of interaction steps to add depends on what users you target: we also applied our classifier to three public interaction corpora [Baumann et al., 2026, Tang et al., 2026, DataClaw Contributors, 2026] and obtained visibly different Task Flows for each (Appendix B)—a distribution measured on one product does not transfer to another. Our position is therefore: there is no universal realistic coding-agent benchmark; benchmark builders should state the target use case, measure its Task Flow, and calibrate the benchmark’s interaction layer to it.

Contributions. We contribute (i) a Task-Flow analysis of long Production Sessions that makes a measured interaction distribution the design target; (ii) SWE-TaskFlow, an approach that turns verified SWE tasks into replayable multi-turn trajectory candidates while preserving the original requirements and tests; and (iii) a per-turn evaluation design in which one task yields several verifiable signals instead of a single terminal score, with a pilot on 700 SWE-Bench Pro tasks showing that the interaction protocol is a real evaluation variable. All generated tasks are released as one public dataset (details in Appendix E).

## 2 Task Flows of Production Sessions

Data and classification. A Production Session is one conversation between a developer and the coding agent inside the IDE. We analyze only the user’s side of it: roughly $1 4 { , } 0 0 0$ user messages across the 4,782 sessions, collected from users of JetBrains IDEs in the course of their everyday work. Each message is independently classified by an open-weight LLM into one of 16 task types, or intents (bug fix, new feature, explain, planning, code review, execution, and so on); the definitions were iterated with manual review, and three classifier models agree on 76.1% of labels (taxonomy, prompt, and robustness check in Appendix A). The same pipeline is applied to Production Sessions, public corpora, and benchmark prompts, so every distribution shares the same footing.

Task Flow. Formally, a corpus-level Task Flow is a triple of distributions $( P _ { L } , P _ { T } , P _ { R } ) \colon ( \mathrm { i } )$ session lengths, (ii) task types over messages, and (iii) directed transitions between adjacent messages. The transition distribution preserves ordering information that a label histogram alone would lose.

Production Sessions. Of the 4,782 sessions (analyzed only as anonymized aggregate statistics), 50.9% contain a single user message and 33% at least three; the 75th and 90th percentiles are three and eight messages (Appendix A, Figure 3), and first messages average 30.7 words against 16–19 at later positions. Bug fixes (25.9%) and new features (18.9%) lead but do not dominate as they do in issue-derived datasets. All subsequent Task-Flow analysis conditions on this long-session slice.

QA on project code is a frequent intent. In long Production Sessions, EXPLAIN requests (questions about how the project’s code works) account for 6.1%, while general programming QUESTIONs for another 9.7%. Explanation is the intent that issue-derived benchmarks lack; it remains common at every position (Figure 1) and motivates the QA augmentation of Section 3.

Public corpora have different Task Flows. We applied the same pipeline to long-session samples from SWE-chat, SpecStory, and DataClaw [Baumann et al., 2026, Tang et al., 2026, DataClaw Contributors, 2026]. Their Task Flows visibly differ from Production Sessions and from each other in both intents and transitions (Appendix B): there is no ground-truth interaction distribution, and the target must be measured in its own setting.

## 3 SWE-TaskFlow: Transforming Verified Tasks

Building a new interactive benchmark for every target is expensive and would discard the investment behind existing verified SWE tasks (repository states, execution environments, test-based verifiers). SWE-TaskFlow is therefore a transformation of existing issue-derived datasets, instantiated here on SWE-Bench Pro: the repository, requested change, environment, and final tests remain fixed, while the interaction layer is regenerated. Related transformations (Appendix D) also fix trajectories before inference for exact replayability; SWE-TaskFlow differs in calibrating their shape to a measured Task Flow rather than choosing it ad hoc.

Dataset augmentation. We propose two augmentations to the existing tasks. The first, prompt splitting, targets overspecification: a SWE-bench-style issue states the goal, context, constraints, and often the approach in one polished message, whereas real users reveal requirements gradually (Section 2). Splitting turns one issue into K ordered, self-contained user turns (K=3 here; adjacent parts can be merged). The LLM-based splitter must preserve every original requirement, introduce none, keep technical artifacts verbatim, and make each turn actionable in the repository state left by earlier turns. Issues that do not pass these checks stay single-turn; 700 of the 731 public SWE-Bench Pro tasks are retained after splitting and evaluation-context checks. The second, repository QA, generates zero to three verifiable questions about repository behavior, injecting the explain intent that benchmarks lack. Question placement—before, between, or after issue turns—determines which transitions appear; together the augmentations yield one to six user turns per task, of which candidates with at least three form the long-session slice.

Where the questions come from. Questions are mined from an agent trajectory that solved the original issue, read together with the live repository state. Every candidate carries a hidden reference answer and must be answerable from the repository, stay valid before and after the issue patch, not leak the solution, and pass an executable proof script run in a clean repository copy at creation time. At evaluation time the agent answers in text and is judged against the reference (full generation methodology and pipeline in Appendix E).

Measuring alignment. To make “closer to the target Task Flow” explicit and optimizable, we propose the TaskFlow Alignment Score (TFAS). Let $P _ { k }$ and $Q _ { k }$ be target and benchmark distributions for $k \in \{ L , T , R \}$ : session-length bins, task types, and directed transitions between adjacent messages. TFAS uses 13 of the 16 classes—GREETING, META-QUESTION, and OTHER are not instantiable as repository tasks; transitions are computed over retained messages. With base-2 Jensen–Shannon divergence, we set $S _ { k } = 1 - \mathrm { J S D } _ { 2 } ( P _ { k } , Q _ { k } )$ and $\mathrm { T F A S } = 1 0 0 \cdot \left( S _ { L } \cdot S _ { T } \cdot S _ { R } \right) ^ { 1 / 3 }$ , where each $S _ { k }$ lies in [0, 1], so TFAS lies in [0, 100], with 100 an exact match of all three distributions. Candidate selection maximizes TFAS over the generated family subject to task-count and diversity constraints, and each component must accompany the aggregate (computation details, the optimized selection, and ablations in Appendix F).

## 4 Pilot: The Interaction Layer Changes the Measurement

Does progressive disclosure change difficulty? With mini-swe-agent [Yang et al., 2024] as a common scaffold, we evaluated Single and Split prompts on the same 700 tasks with four models. Split approximately doubled agent cost (Appendix H, Figure 11). Resolve-rate shifts were mixed across models and largely comparable to run-to-run variation, so these runs establish no success ordering but show that progressive disclosure reliably changes the effort an agent expends.

Calibrated QA turns. Tuning the number and positions of QA turns to maximize TFAS against the long-session Production target (Split+QA in Figure 2) raises TFAS from 59.6 for pure Split to

![](images/ba2f1f0c2d4257319116c27d47c7407de3c9f50b3db96ebe38396b8ab9c1f029.jpg)  
Figure 2: Resolve rate versus mean agent cost per issue (log scale) on the 700 tasks; markers are three-run means, whiskers min–max (not confidence intervals).

77.8 (Appendix F, Table 2; Task Flow in Figure 7). The calibrated QA turns add moderate cost on top of Split; resolve rate again shows no stable ranking across models (Figure 2), and QA order has no stable effect on resolve rate or answer accuracy (Appendix H). Answer accuracy on the selected benchmark spans 69.8–83.0% across the four models, a second verified signal alongside resolve rate (Appendix G).

Per-turn verified signals. A Single trajectory produces one binary signal, while a Split+QA trajectory yields a test-suite verdict after each issue turn (scored only against requirements revealed so far) and a judged answer per QA turn (Appendix H, Figure 14). Denser verified signals support finer-grained diagnosis and ranking-based training [Choi et al., 2026, Yari and Koto, 2026, Yu, 2026].

## 5 Discussion: Toward Richer Task Flows

Most user intents still lack augmentations. SWE-TaskFlow currently covers prompt splitting for implementation intents and repository QA for EXPLAIN (cf. SWE-QA, SWE Atlas, RepoProbe [Peng et al., 2026, Raghavendra et al., 2026a, Yang et al., 2026]). Everything else in Figure 1 (general questions, planning, code review, refactoring, execution, debugging without a fix, etc.) still needs its own augmentation: a way to generate turns of that intent on a verified task and to check the response. Most are open problems: it is unclear how to verify a planning or code-review turn automatically. A measured Task Flow tells us which augmentations are missing and in what proportion.

Static and reactive interaction are complementary regimes. Fixing all user messages before inference enables exact replay and paired comparisons: SWE-TaskFlow measures progressive disclosure, memory, and adaptation, but not feedback reacting to what the agent did. The reactive benchmarks in Appendix D (withheld requirements, simulated users, injected edits, help-seeking) measure exactly that, at the price of simulator assumptions and trajectory variance. A full evaluation suite can combine a calibrated static core with targeted reactive slices.

Limitations. Long Production Sessions define only one use case, though the approach applies to any measured target (Appendix C). The intent classifier lacks a human-labeled accuracy set (Appendix A: inter-model agreement 76.1%, manually checked samples per intent). Some issues resist splitting, and QA difficulty is generator-dependent. Finally, each pilot condition ran three times with min–max ranges, not confidence intervals; no comparison carries a significance test, so resolve-rate readings are directional only.

## 6 Conclusion

An interactive benchmark must state whose interaction distribution it represents: real users mix and switch task types in ways issue-derived benchmarks miss, and public corpora do not represent each other. SWE-TaskFlow addresses this gap while preserving verified tasks: splitting, verifiable QA, and TFAS-selected placement turn issue-derived benchmarks into calibrated, replayable multi-turn evaluations.

## References

Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, Maria Trofimova, Daria Litvintseva, and Boris Yangel. SWE-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, Datasets and Benchmarks Track, 2025. URL https://openreview.net/forum?id=nMpJoVmRy1.

Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, and Sanmi Koyejo. SWE-chat: Coding agent interactions from real users in the wild. In The Third Conference on Language Modeling, 2026. URL https://arxiv.org/abs/2604.20779.

Kyuseong Choi, Dwaipayan Saha, Woojeong Kim, Anish Agarwal, and Raaz Dwivedi. GOPO: Policy optimization using ranked rewards, 2026. URL https://arxiv.org/abs/2602.03876.

DataClaw Contributors. Dataclaw: Agent harness for publishing agent chat history as hugging face datasets. GitHub repository, 2026. URL https://github.com/peteromallet/dataclaw. Accessed August 27, 2026.

Xiang Deng, Jeff Da, Edwin Pan, Yannis Yiming He, Charles Ide, Kanak Garg, Niklas Lauffer, Andrew Park, Chetan Rane, Karmini Sampath, Maya Krishnan, Srivatsa R. Kundurthy, Sean M. Hendryx, Zifan Wang, Chen Bo Calvin Zhang, Noah Jacobson, Bing Liu, and Brad Kenstler. SWE-bench pro: Can AI agents solve long-horizon software engineering tasks? In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 23917–23934. PMLR, 2026. URL https://proceedings.mlr.pr ess/v306/deng26f.html.

Spandan Garg, Benjamin Steenhoek, and Yufan Huang. Saving SWE-bench: A benchmark mutation approach for realistic agent evaluation. In Proceedings of the IEEE/ACM 5th International Conference on AI Engineering – Software Engineeringfor AI (CAIN), 2026. doi: 10.1145/379365 3.3793762. URL https://doi.org/10.1145/3793653.3793762.

Wenqi Huang, Charley Lee, Leonard Tng, and Serena Ge. DeepSWE: Measuring frontier coding agents on original, long-horizon engineering tasks, 2026. URL https://arxiv.org/abs/2607 .07946.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net /forum?id=VTF8yNQM66.

Brendan King and Jeffrey Flanigan. Dialogue SWE-bench: A benchmark for dialogue-driven coding agents, 2026. URL https://arxiv.org/abs/2606.13995.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multi-turn conversation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=VKGTGGcwl6.

Weihan Peng, Yuling Shi, Yuhang Wang, Xinyun Zhang, Beijun Shen, and Xiaodong Gu. SWE-QA: Can language models answer repository-level code questions? In Findings of the Association for Computational Linguistics: ACL 2026, 2026. doi: 10.18653/v1/2026.findings-acl.402. URL https://aclanthology.org/2026.findings-acl.402.

Mohit Raghavendra, Soham Dan, Miguel Romero Calvo, Yannis Yiming He, Johannes Baptist Mols, Gautam Anand, Cole McCollum, Edgar Arakelyan, Vijay Bharadwaj, Andrew Park, Jeff Da, MohammadHossein Rezaei, Bing Liu, Brad Kenstler, and Yunzhong He. SWE atlas: Benchmarking coding agents beyond issue resolution, 2026a. URL https://arxiv.org/abs/2605.08366.

Mohit Raghavendra, Anisha Gunjal, Aakash Sabharwal, and Yunzhong He. SWE-interact: Reimagining SWE benchmarks as user-driven long-horizon coding sessions, 2026b. URL https: //arxiv.org/abs/2606.30573.

Yuqiao Tan, Jinxiang Meng, Fangyu Lei, Minzheng Wang, Shizhu He, Jun Zhao, and Kang Liu. SWE-Touch: Benchmarking coding agents when users touch the code, 2026. URL https: //arxiv.org/abs/2608.02499.

Ningzhi Tang, Chaoran Chen, Zihan Fang, Gelei Xu, Maria Dhakal, Yiyu Shi, Collin McMillan, Yu Huang, and Toby Jia-Jun Li. Programming by chat: A large-scale behavioral analysis of 11,579 real-world ai-assisted ide sessions. In Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering, 2026. doi: 10.1145/3832783.3834377. URL https://arxiv.org/abs/2604.00436.

Tu Trinh, Mohamed Elfeki, Guangze Luo, Kelvin Luu, Nathan Hunt, Ernesto Hernández, Nandan Marwaha, Yannis Yiming He, Charles Wang, Fernando Carabedo, Alessa Castillo, and Bing Liu. HiL-Bench (human-in-loop benchmark): Do agents know when to ask for help?, 2026. URL https://arxiv.org/abs/2604.09408.

Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, and Graham Neubig. Ambig-SWE: Interactive agents to overcome underspecificity in software engineering. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net /forum?id=X2yzXtH4wp.

Yifan Wu, Zhuokai Zhao, Songlin Li, Ho Hin Lee, Jiacheng Zhu, Shirley Wu, Tianhe Yu, Serena Li, Lizhu Zhang, Xiangjun Fan, and Shengzhi Li. SWE-together: Evaluating coding agents in interactive user sessions, 2026. URL https://arxiv.org/abs/2606.29957.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R. Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2405.15793.

Yuexi Yang, Alyssa Wu, Ji Luo, Richeng Xuan, Zhichao Hu, Yuhong Liu, and Zhen Qin. RepoProbe: Benchmarking architecture-aware repository comprehension with checklists. In Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering, 2026. doi: 10.1145/3832783.3837452. URL https://arxiv.org/abs/2608.04783.

Amir Hossein Yari and Fajri Koto. AMIR-GRPO: Inducing implicit preference signals into GRPO, 2026. URL https://arxiv.org/abs/2601.03661.

Hao Yu. A unified pair-GRPO family: From implicit to explicit preference constraints for stable and general RL alignment, 2026. URL https://arxiv.org/abs/2605.06375.

## A Task Taxonomy and Classification

This appendix documents how every Task Flow in the paper is produced: the classification pipeline, the full 16-label taxonomy with a prompt excerpt, an inter-model robustness check, and summary statistics of the classified sessions.

Classification pipeline. Each user message is classified independently by an open-weight LLM (Gemma 4-31B) that receives the message text together with the label definitions below and returns exactly one task\_type as structured output; when a message contains several requests, the classifier assigns the dominant actionable intent. The same pipeline, prompt, and definitions are applied to Production Sessions, the three public interaction corpora, and benchmark prompts, so every Task Flow in the paper is computed on the same footing. The taxonomy itself is grounded in industrial practice: it originates from an intent categorization used at JetBrains to analyze production assistant usage, which we consolidated into the 16 labels below. We chose it over published taxonomies because it was iterated directly against the kind of IDE sessions we study, and because it keeps IDE-specific intents—execution, meta-questions about the assistant itself—separable from repository tasks. During prompt development, the authors manually reviewed samples of classifications, revised ambiguous label instructions, and re-ran the pipeline; a quantitative inter-model robustness check is reported below.

tests Write or extend test coverage. Fixing an existing failing test belongs to bug-fix or debug.

bug-fix Fix an existing error, bug, or exception. A bare stack trace defaults to a fix.

new-feature Add new functionality.

generate-docs Write documentation, comments, or README files.

refactor Improve same-language code structure without changing behavior.

transformation Convert code between formats, languages, library versions, or paradigms.

debug Investigate why something fails without fixing it.

planning Plan a change, design an approach, or evaluate a plan or specification.

explain Explain how specific code in the user’s project works.

question Ask a general programming, framework, library, or tool question.

meta-question Ask about the assistant, IDE, quota, billing, or environment.

code-review Review specific changes for style, correctness, or performance.

execute Run a command, build, test, query, deployment, or external action.

data-analysis Analyze data or build and interpret queries, charts, or metrics.

greeting Provide no programming content beyond a greeting, ping, or meaningless input.

other Provide residual non-programming content such as thanks or feedback.

Prompt excerpt. The classifier receives one user message at a time. The task-type portion of the prompt is:

Classify the primary intent of a developer message from a coding-assistant session. Assign exactly one task\_type. Focus on what the user asks the assistant to do, not on the topic, language, or solution. If the message contains several requests, choose the dominant actionable intent. Use only one label from the supplied taxonomy and return the requested structured output.

Inter-model robustness. To estimate how much the labels depend on the choice of classifier, we classified the same 100 sessions with three open-weight models: Qwen 3.6-35B, Gemma 4-31B, and GPT-OSS 20B. Pairwise exact agreement on task\_type was 79.2%, 73.7%, and 75.5%, respectively, for a mean of 76.1% (Table 1). This measures inter-model robustness rather than accuracy: no formal human-labeled test set is available.

Session statistics. Figure 3 summarizes the classified sample: the distribution of Production Sessions by number of user messages.

Table 1: Exact agreement on task\_type over the same 100-session sample.
<table><tr><td>Model pair Agreement</td></tr><tr><td>Qwen 3.6-35B / Gemma 4-31B 79.2% Qwen 3.6-35B / GPT-OSS 20B 73.7%</td></tr><tr><td>Gemma 4-31B / GPT-OSS 20B 75.5% Mean pairwise agreement 76.1%</td></tr></table>

![](images/13c4ed4ce33a855567fcb4387fa180715e6b7d2498d4e50acc680935f53b3500.jpg)  
Figure 3: Production Sessions by number of user messages. The long-session slice studied throughout the paper aggregates all bins with at least three messages.

## B Task Flows in Public Interaction Corpora

We applied the classification pipeline of Appendix A to long-session samples (at least three user messages) from three public interaction corpora, 1,000 classified sessions each. SpecStory (Figure 4) consists of public IDE-native session exports analyzed by Programming by Chat [Tang et al., 2026]: of the three corpora its Task Flow sits closest to Production Sessions, but with a larger planning share. DataClaw (Figure 5) collects community-shared coding-agent session exports [DataClaw Contributors, 2026]: it is dominated by execution and short operational turns. SWE-chat (Figure 6) is an interaction corpus from a chat-based coding setting [Baumann et al., 2026]: execution and planning dominate its flows. These corpora are comparison datasets; the paper’s calibration target remains Production Sessions. Their collection mechanisms, user populations, and products all differ, so the figures support a qualitative claim: the interaction distribution depends on the source, and none of them is ground truth. Large-scale behavioral analysis of coding sessions has precedent— Programming by Chat characterizes 11,579 real-world AI-assisted IDE sessions [Tang et al., 2026]; what our Task-Flow analysis adds is the transition view and its use as a calibration target for benchmarks.

![](images/cd64153c2bfdc2ffcb294b6be79577cebc06acb7eb265275335cc5ff2788a117.jpg)  
Figure 4: SpecStory long-session Task Flow: first five user messages of the 521 of 1,000 classified sessions with at least three; full 16-label taxonomy.

![](images/36bcf55296623de0485ffd268de5e2e29b55f2bf18e9efe5232bdfe300ba2955.jpg)  
Figure 5: DataClaw long-session Task Flow: first five user messages of the 585 of 1,000 classified sessions with at least three; full 16-label taxonomy.

![](images/0fe372ee626b54d37856863e8425529d086b7854c944790386460d68608c63a4.jpg)  
Figure 6: SWE-chat long-session Task Flow: first five user messages of the 698 of 1,000 classified sessions with at least three; full 16-label taxonomy.

## C Calibrating to a New Target

This appendix turns the calibration approach of Sections 2–3 into a step-by-step procedure for a new target. For each step we state what goes in and what comes out, how we instantiated it in this paper, and what a benchmark builder working in a different domain has to decide anew. The procedure assumes a source benchmark with verified tasks (repository state, environment, tests) whose interaction layer is to be re-shaped; it does not assume that the source benchmark and the target share a domain.

1. Name the target. A target is a user population, a product surface, a session filter, and a measurement window. Ours: users of JetBrains IDEs during their everyday work, sessions with at least three user messages, one sample of 4,782 sessions. The filter is part of the definition: the same sample with a filter of at least one message is dominated by singlemessage sessions (50.9%) and yields a different Task Flow. Task Flows also drift as products and models change, so a target should carry its measurement date, and a benchmark calibrated to it should state that date. New domain: the population and filter should match the question the benchmark is meant to answer (for example, terminal-agent users with at least one tool call, or data-science notebook sessions longer than ten cells). Borrowing a target from a different product is not a shortcut: the three public corpora in Appendix B differ from each other and from Production Sessions.

2. Sample user-side messages. Only the user’s messages are needed—no agent output, no code, no tool logs—which keeps collection cheap and limits privacy exposure. Ours: roughly 14,000 user messages across 4,782 sessions, analyzed in aggregate only. New domain: the sample must be large enough for the transition distribution to be stable; with 13 retained labels plus START/END, the transition support has about 200 cells, most of them rare. Resample complete sessions with the bootstrap and check that the component scores $S _ { k }$ of Appendix F are stable before fixing the target; if they are not, enlarge the sample or coarsen the taxonomy.

3. Classify with a fixed taxonomy. Every message receives exactly one label from a taxonomy whose definitions are frozen before measurement. Ours: the 16-label taxonomy, prompt, and open-weight classifier of Appendix A, with a 76.1% mean pairwise inter-model agreement on 100 sessions. New domain: reuse the taxonomy where possible and extend rather than replace it, so that Task Flows measured for different targets remain comparable; every new label must be marked as instantiable or not as a repository task, because only instantiable labels enter TFAS. Run the inter-model check on at least 100 sessions before measuring; low agreement signals ambiguous label definitions, which should be revised and re-run before the target is fixed, not corrected afterwards.

4. Measure the target Task Flow. Compute $( P _ { L } , P _ { T } , P _ { R } )$ exactly as in Appendix F: length bins, task types over the retained labels, and directed transitions with explicit START/END edges after filtering non-instantiable messages. Ours: the long-session Production target (Figure 1, right). New domain: publish the three vectors together with the benchmark. They are the calibration target, they let others recompute TFAS and its components, and they are aggregate statistics that reveal nothing about individual sessions.

5. Find the gaps. Classify the source benchmark’s prompts with the same pipeline to obtain its own Task Flow $( Q _ { L } , Q _ { T } , Q _ { R } )$ and compare it with the target component by component. Every task type that is frequent in the target but rare or absent in the benchmark, and every session length that occurs in the target but that the benchmark cannot produce, needs an augmentation that stays verifiable. Ours: SWE-Bench Pro is single-turn and 93.8% newfeature or bug-fix, so the length gap is closed by prompt splitting and the EXPLAIN gap by repository QA; the remaining gaps—QUESTION, PLANNING, CODE-REVIEW, REFACTOR, EXECUTE, DEBUG—are listed as uncovered in Section 5. New domain: the gap table is the to-do list of augmentations, ordered by how frequent each missing intent is in the target. An intent without a verifier should be reported as uncovered rather than approximated by an unverifiable turn.

6. Generate candidate families and select. For each verified task, enumerate the admissible combinations of augmentations (here: merge level of the split turns, number of QA turns, their placement), then choose one candidate per task to maximize TFAS against the target, subject to task-count and diversity constraints. Ours: the randomized local search of Appendix F, with the component scores reported alongside the aggregate (Table 2). New domain: before granting the optimizer freedom along some dimension, check that moving along it does not change task difficulty; our QA-placement ablation (Appendix H) is this check for placement. Without it, a higher TFAS could be bought by changing what the benchmark measures.

7. Separate stress tests. Any selection that maximizes alignment with observed frequencies pushes rare intents toward zero. Intents that are rare in the target but consequential for the product—destructive EXECUTE commands, security-relevant CODE-REVIEW—should be evaluated as explicit stress-test slices with their own task counts and reported separately from the calibrated core. Ours: the released dataset keeps every experimental condition as a separate split, and the SWE-chat ablation (Table 3) is reported apart from the Productioncalibrated selection. New domain: decide which rare intents matter before optimizing, so that their slices are built deliberately rather than recovered from what the optimizer discarded.

If no production data is available, the closest public interaction corpus (Appendix B) can serve as a provisional target, provided the substitution is stated explicitly; TFAS still yields a measurable distance, and the target can be replaced once production measurements exist.

## D Related Transformations of Verified Tasks

SWE-TaskFlow builds on a growing family of works that add an interaction layer on top of existing verified tasks rather than collecting new ones. We group them into three families.

Static transformations fix the user messages before inference, as we do. Sharded decomposition of a task into a fixed sequence of prompts has been studied for HumanEval and LiveCodeBench, where progressive disclosure consistently changed measured performance [Laban et al., 2026], and Saving SWE-Bench lowers the specificity of the issue text while keeping the interaction single-turn [Garg et al., 2026]. These transformations choose their trajectory shapes ad hoc; none calibrates the resulting interaction distribution to a measured user population, which is the gap SWE-TaskFlow targets.

Reactive designs make the environment respond to the agent instead of fixing the messages upfront. Ambig-SWE and Dialogue SWE-Bench withhold requirements and reveal them when the agent asks [Vijayvargiya et al., 2026, King and Flanigan, 2026]; SWE-Interact and SWE-Together employ LLM-simulated users that answer clarification questions [Raghavendra et al., 2026b, Wu et al., 2026]; SWE-Touch injects user edits into the repository mid-trajectory [Tan et al., 2026]; and HiL-Bench studies when agents seek human help [Trinh et al., 2026]. Reactive designs measure adaptation to feedback but give up exact replayability: two runs of the same task can see different user messages (Section 5 contrasts the two regimes).

Repository QA benchmarks evaluate question answering about a codebase as a task in its own right: SWE-QA, SWE Atlas, and RepoProbe offer complementary designs [Peng et al., 2026, Raghavendra et al., 2026a, Yang et al., 2026]. SWE-TaskFlow uses repository QA differently—as inserted turns inside a longer trajectory, with creation-time proof scripts guaranteeing that each question is answerable from the repository.

## E Generation and Verification Pipeline

Figure 7 contrasts the interaction layer of the source benchmark with the result of the transformation at the benchmark level, Figure 8 shows the intermediate Task Flow of pure Split trajectories before QA insertion, and Figure 9 shows the complete generation, assembly, inference, and verification pipeline.

Prompt-splitting contract. The splitter receives an issue description and first decides whether three meaningful, ordered user turns are possible. It then extracts specificity units, subtask units, and verbatim technical artifacts before writing the sequence and performing a final model-based completeness check. An eligible decomposition must preserve every original requirement, introduce no new requirement, keep technical artifacts unchanged, avoid references to the decomposition process, and make each turn actionable in the repository state produced by earlier turns. Ineligible atomic issues remain single-turn. Figure 8 shows the Task Flow of the resulting Split trajectories.

Question-generation contract. The generator reads a prior solving trajectory and the corresponding live repository state, extracts stable cross-file behavior, and proposes one to three questions with reference answers and evidence chains. Questions must be answerable from the repository, remain valid before and after the issue patch, and avoid revealing the issue solution. In the current pipeline, generation runs in Codex using Sol or Terra: the TFAS-optimized selection and the Split+QA results use the Sol-generated pool—454 questions across 255 of the 700 tasks (Appendix F)—while the 64-task placement ablation uses Terra-generated questions (Appendix H). A separate model pass reviews clarity and repository support. Executable proof scripts serve as mandatory creation-time instrumentation: for each candidate question, the pipeline generates a reference script, executes it in a clean environment, checks that it produces evidence without modifying the repository, and retains only candidates whose proof passes. This requirement bounds the admissible question space—questions without observable evidence, such as the rationale for an architectural decision, are excluded by construction.

Inference and issue scoring. All reported inference runs use mini-swe-agent inside isolated Harbor task environments. The agent receives the source repository checkout and the complete ordered sequence of typed prompts, but no reference answers or validation artifacts. User turns are delivered sequentially within one conversation. The agent may edit and test the repository during issue turns, and the harness records patch checkpoints. After the final turn, the source benchmark’s tests/test.sh evaluates the cumulative patch in the same patched sandbox without resetting the repository. QA answers are stored separately and never enter the issue patch.

QA validation. Answer scoring uses GPT-Sol in Codex to compare the submitted answer with the hidden reference answer and its evidence chain. Original issue tests and creation-time proof execution provide deterministic evidence, but final QA correctness remains model-judged.

Released dataset. All generated tasks used in this paper are released as one public dataset on Hugging Face, https://huggingface.co/datasets/JetBrains-Research/SWE-TaskF low-SWE-Bench-Pro, over the same ordered 700-task SWE-Bench Pro subset; each experimental condition is a separate split. The single\_prompt, split\_prompt, and split\_prompt\_concat splits contain the prompt-only Single, Split, and Concat conditions (see Appendix H). The remaining splits contain the QA-first/-last/-random placements and QA-only variants for the Terra and Sol question pools, plus the TFAS-optimized selection of Appendix $\operatorname { F } ;$ every question ships with its reference answer (hidden from the evaluated agent), golden proof script, and creation-time proofexecution report. Content hashes and provenance manifests make every experimental condition exactly replayable. The dataset contains ordered user prompts and validation artifacts, not agent runs. Source repositories, gold patches, and verifier tests are not redistributed and remain with the original SWE-Bench Pro release [Deng et al., 2026].

## F TFAS Details and Dataset Tuning

TFAS compares three pairs of distributions: for each component $k \in \{ L , T , R \}$ —session-length bins, task types, and directed transitions— $P _ { k }$ is the target distribution measured on long Production Sessions and $Q _ { k }$ is the corresponding distribution over the benchmark’s candidate trajectories. The score defined in Section 3 is

$$
S _ { k } = 1 - \mathrm { J S D } _ { 2 } ( P _ { k } , Q _ { k } ) , \qquad \mathrm { T F A S } = 1 0 0 \cdot ( S _ { L } \cdot S _ { T } \cdot S _ { R } ) ^ { 1 / 3 } ,\tag{1}
$$

where, for two distributions P and Q on a shared support with $M = ( P + Q ) / 2$

$$
\begin{array} { r } { \mathrm { J S D } _ { 2 } ( P , Q ) = \frac 1 2 \mathrm { K L } _ { 2 } ( P \| M ) + \frac 1 2 \mathrm { K L } _ { 2 } ( Q \| M ) . } \end{array}\tag{2}
$$

We use length bins 1, 2, 3, 4, 5, 6–10, and 11+. For the current target, bins below three have zero target mass and candidate trajectories below three turns serve as ablation baselines. The task distribution is defined over the 13 labels retained after excluding GREETING, META-QUESTION, and OTHER, which are not instantiable as repository tasks. We remove those messages before constructing transitions, so an excluded turn between two retained messages is skipped and the retained messages become adjacent. The transition distribution contains the resulting ordered pairs and explicit START/END edges around each filtered sequence. This filtering applies only to TFAS; the Sankey diagrams in the paper are unfiltered and retain all 16 labels to show the source data transparently. Each of the three distributions is normalized independently. Jensen–Shannon divergence remains finite when one distribution assigns zero mass to an event, but rare transitions can still be unstable; bootstrap intervals should therefore resample complete sessions rather than individual edges.

![](images/a114fb37246ad81e2f4916ad89f80a3dc17fccb0463e5a120cec83403736218e.jpg)  
Figure 7: Task Flow of SWE-Bench Pro (left) and of the TFAS-optimized selection (right)—the configuration reported in Table 2 with TFAS 77.8: one user prompt per issue becomes multiple sequential prompts spanning different task types. Repositories, requested changes, and final test-based verifiers are unchanged.

![](images/a32c582c5d9abac0946a99ce2ba11bb8f12f28084705aee0eb9531ea1b547060.jpg)  
Figure 8: Task Flow after three-way prompt splitting. Splitting increases trajectory length and exposes several intents that were implicit in the original issue.

![](images/e765c63027dce3ad0b778e8daf7237dd8c01f5762cea2c3ad3a672f63085c002.jpg)  
Figure 9: Complete generation, assembly, inference, and verification pipeline. Reference answers and proof artifacts remain hidden from the evaluated agent.

For each verified source task, the generated candidate set includes the original single prompt, one to three issue turns formed by merging adjacent split prompts, zero to three available QA turns, and valid placements of the selected QA turns. Long-slice tuning filters this set to trajectories with at least three turns, then selects one candidate per source task to maximize TFAS against the long-session Production target. The component scores and full candidate-selection configuration should accompany every aggregate TFAS.

TFAS-optimized selection. The question pool contains 467 validated Sol-generated questions covering 255 of the 700 splittable SWE-Bench Pro tasks. A randomized local search (64 restarts, up to 20 sweeps, fixed seed) decides, per task, how many of its questions to keep and where to place them, maximizing Equation 1 against the long-session Production target; the 445 tasks without questions keep their three issue turns. The optimizer retained 454 of the 467 questions, augmenting 113 sessions with one question, 85 with two, and 57 with three. The selected placements mix all patterns: 67 sessions carry their questions only before the issue turns, 86 only after, and 102 interleaved; by slot, 151 questions precede the first issue turn, 133 sit between issue turns, and 170 follow the last

one. Table 2 reports the resulting alignment: adding QA lifts every component over pure Split, fixed placements already reach TFAS 75.6–77.1, and optimizing count and placement adds mainly transition alignment, for a final TFAS of 77.8 versus 59.6 for Split alone. The original single-turn prompts score $S _ { L } { = } 0$ against the three-message target (their length bin has zero target mass), which collapses the geometric mean to zero.  
Table 2: TFAS and its components against the long-session (≥3 messages) Production target, computed on the same 700 splittable SWE-Bench Pro tasks. Split keeps three issue turns; the QA rows place the Sol-generated questions at fixed positions; TFAS-optimized selects question count and placement per task.
<table><tr><td>Trajectory variant</td><td> $S _ { L }$ </td><td> $S _ { T }$ </td><td> $S _ { R }$ </td><td>TFAS</td></tr><tr><td>Split</td><td>49.1</td><td>74.0</td><td>58.2</td><td>59.6</td></tr><tr><td> $S p l i t { + } Q A { - } \ f r s t$ </td><td>83.9</td><td>81.5</td><td>63.7</td><td>75.8</td></tr><tr><td> $S p l i t { + } Q A { - } l a s t$ </td><td>83.9</td><td>81.5</td><td>63.3</td><td>75.6</td></tr><tr><td> $S p l i t { + } Q A { - } r a n d o m$ </td><td>83.9</td><td>81.5</td><td>67.0</td><td>77.1</td></tr><tr><td> $T F A S - o p t i m i z e d$ </td><td>84.0</td><td>81.5</td><td>68.8</td><td>77.8</td></tr></table>

Table 3 additionally preserves an earlier fixed-variant ablation against SWE-chat (64 tasks, Terragenerated questions): calibrating to a different target favors different variants, which is precisely why the target must be stated.

Table 3: TFAS ablation against SWE-chat, reported separately because it is not the paper’s calibration target. All values are percentages on the same 64 source tasks.
<table><tr><td>Condition</td><td> $S _ { L }$ </td><td> $S _ { T }$ </td><td> $S _ { R }$ </td><td>TFAS</td></tr><tr><td>Single</td><td>37.0</td><td>52.9</td><td>17.3</td><td>32.4</td></tr><tr><td> $S p l i t$ </td><td>25.4</td><td>52.6</td><td>37.7</td><td>36.9</td></tr><tr><td> $S p l i t { + } Q A { - } \ f r s t$ </td><td>50.1</td><td>53.9</td><td>31.3</td><td>43.9</td></tr><tr><td> $S p l i t { + } Q A { - } l a s t$ </td><td>50.1</td><td>53.9</td><td>32.5</td><td>44.5</td></tr><tr><td> $S p l i t { + } Q A$  -random</td><td>50.1</td><td>53.9</td><td>35.1</td><td>45.6</td></tr></table>

## G QA Answer Accuracy

At evaluation time the agent answers each repository question in text, and the answer is judged against the hidden reference (Appendix E); answer accuracy is the fraction of questions judged correct. Figure 10 measures it on the target benchmark itself: the TFAS-optimized Split+QA trajectories carrying the Sol-generated question pool (454 questions across 255 of the 700 tasks; Appendix F), evaluated with the same four models and runs as Figure 2. Each model contributes two points from the same trajectories—answer accuracy over its questions and resolve rate over the same QA-scored issues—so every trajectory yields both a patch outcome and a QA outcome. Everything in the figure is computed on these 255 issues that carry questions, not on the full 700-task set.

Answer accuracy spans 69.8% (GPT-5.4 Mini) to 83.0% (GPT-5.6 Terra) across the four models. We report the two metrics side by side because they come from the same trajectories, not to claim any relationship between them.

![](images/902984829e64ae4548c6da266783748bb1130271f08a79101ba1d1eb298d6610.jpg)  
Figure 10: Answer accuracy (solid) and resolve rate (hollow) versus agent cost for the TFASoptimized Split+QA trajectories with Sol-generated questions; each vertical pair comes from the same runs of one model. Both metrics are computed on the 255 issues that carry QA questions, not on all 700 tasks, so resolve rate is not directly comparable to Figure 2.

## H Additional Experimental Results

Protocol. The models are GPT-5.6 Luna, GPT-5.6 Terra, and GPT-5.4 Mini, all run with mini-sweagent in Harbor and with reasoning effort set to medium; Figure 2 additionally includes Gemini 3.7 Flash. Its Single/Split/Split+QA comparison uses all 700 tasks and three runs per model and mode. Concat delivers the three Split segments in one message and has one available 700-task run per GPT model. Figure 11 therefore combines three-run summaries for Single and Split with single-run Concat points; these comparisons are descriptive, not significance tests. Intermediate checkpoint results are reported below.

Concat ablation (information preservation). Automatic decomposition could lose or distort information from the original issue; if it did, any Split effect would confound the interaction protocol with rewriting artifacts. The Concat control removes this confound: it delivers the same three rewritten segments as Split, but concatenated into a single message. The gap between Single and Concat therefore measures information change introduced by rewriting, while the gap between Concat and Split isolates the effect of sequential disclosure itself. Figure 11 shows all three conditions.

QA-placement ablation. Placement shapes the transition distribution, and TFAS optimization moves questions to whatever positions the target Task Flow demands; before permitting that, we check whether position alone changes task difficulty. The placement policy of Section 3 admits three strategies for a session that combines issue turns and QA turns: QA-first places all questions before the issue turns, QA-last places them after, and QA-random interleaves them at random positions. The ablation uses the same 64 instances and 114 Terra-generated questions in every condition; only phase order differs, so the comparison isolates ordering rather than the effect of adding QA (no matched no-QA run exists for these tasks). Resolve-rate differences between the strategies show no consistent ordering across models and are comparable to run-to-run variation (Figure 12). This is the license TFAS optimization relies on: within the tested range, moving questions around changes the measured Task Flow without changing how hard the underlying tasks are. Answer accuracy under the same placements is 78.1–86.8% for GPT-5.4 Mini, 86.6–93.8% for GPT-5.6 Luna, and 93.8–98.2% for GPT-5.6 Terra (Figure 13). Placement shifts accuracy by a few points with no consistent direction across models, so we read question placement as having no real effect on answer accuracy either. A QA-only condition, in which the trajectory contains only the question turns and no issue turns at all, costs a fraction of any mixed trajectory and still reaches 86.8% with Luna, matching its QA-random accuracy: the questions are answerable without any issue work, as the generation contract requires (every question must stay valid before and after the issue patch).

Checkpoint-level signals. Split trajectories admit intermediate verification: after each issue turn, the harness snapshots the repository state and re-runs the original test suite, yielding a resolve rate and a test-pass rate at every checkpoint rather than a single terminal score. Figure 14 plots both metrics for the placement-ablation runs, measured on the clean repository and after the first, second, and third issue turns, aggregated and per model. Since QA turns never modify the repository, the QA-last arm doubles as a Split-only baseline: all of its questions follow the issue turns, so its checkpoints measure pure splitting with no QA influence at all. At every checkpoint, for every model and in aggregate, the three placement curves lie nearly on top of each other: in this scenario, neither inserting repository questions nor their position measurably changes intermediate resolve or test-pass rates, while both metrics grow steadily as issue turns accumulate—the per-turn signal the interaction layer is designed to expose.

![](images/98ad81fca4ef85de905256bd2f1d7116a3f24781a5286dc82beea471729c0e53.jpg)  
Figure 11: Resolve rate versus average agent cost per issue (log scale). Single and Split markers are arithmetic means of three runs on the 700 tasks; whiskers show observed min–max cost and resolve rate separately, not confidence intervals. Each Concat marker is one 700-task run and therefore has no whiskers.

![](images/5b07f63f033a09f55e724f35fcd79d1bbd86326afba9a05ae45189a4e8e086e6.jpg)  
Figure 12: Resolve rate versus agent cost for QA-first (circle), QA-last (square), and QA-random (diamond) on the same 64 tasks and 114 questions; color encodes the model. Only the order of phases differs between the conditions.

![](images/5dec47d3fae90cf4652b141d8559ddf1d3208a99e69badf2d6a0140cb27b7675.jpg)  
Figure 13: Answer accuracy versus agent cost per issue on the same 64 tasks and 114 Terra-generated questions. Color encodes the model, shape the placement of the QA turns; QA-only delivers only the question turns, with no issue-resolve turns at all.

![](images/9469ffd11e5b4094920cf208337544bf1e32d6ae8e6781901747aa5f40fbb191.jpg)

![](images/921b65c8b848f97e2953574189504a9aedc9f8502b38fcce4a6610b83e6113d3.jpg)

![](images/41bfdc71533881a80e5910fc87a48b9051584dc20d946030b2ef45b3dc79a1fa.jpg)

![](images/2c91a046fa42a05edfb4561079d28409cddb552025fc1560fef6a74019569053.jpg)

![](images/359e0e8382c76954960e7e26c86713316783fe3f0e59d14151ef4c12bbb1c410.jpg)

![](images/1517a458964050cab08e0c72398f1c8aa55b67d132a83c3652ada3bb7dc32515.jpg)

![](images/a6e1918cd287d790af4ca848b9add47ed5f4b351086ade205f57af3716bea41e.jpg)

![](images/3471bc42b9325d63855bcc3c2b079652fb3daecfd6f9a5cde133e5db245ae760.jpg)  
Figure 14: Checkpoint-level resolve rate (left) and test-pass rate (right) on the clean repository and after each of the three issue turns, under the three QA placements, aggregated and per model. QA turns do not modify the repository, so QA-last doubles as a Split-only baseline. The placements are nearly indistinguishable at every checkpoint, while both metrics grow steadily with each issue turn.

## I Ethical Considerations and Broader Impacts

Only aggregate statistics from Production Sessions are reported; the paper includes no session text or user identifiers, and the raw sessions are not released. The Task-Flow analysis is restricted to the longsession slice, while the full sample is used only for prevalence statistics. Public interaction corpora are cited and represented only through aggregate diagrams. The released task datasets (Appendix E) contain only content derived from the public SWE-Bench Pro tasks and model-generated turns; no user data enters them. Better interaction-level evaluation could improve agent reliability and reveal the user effort hidden by terminal scores. Negative effects include additional inference cost, privacy risks from operational data, and the possibility that matching frequent behavior overweights noisy or low-value workflows. We mitigate these risks by reporting aggregates, preserving existing verified tasks, separating representative mixtures from targeted stress tests, and treating model-generated classifications as provisional.

LLM use. LLMs are core methodological components: they classify user-message intent, determine split eligibility, generate ordered prompt segments, extract repository facts, generate questions and reference answers, review candidates, and judge answers in the QA protocol. Executable task tests and creation-time proof execution provide non-textual evidence, but they do not remove model dependence from classification and generation.