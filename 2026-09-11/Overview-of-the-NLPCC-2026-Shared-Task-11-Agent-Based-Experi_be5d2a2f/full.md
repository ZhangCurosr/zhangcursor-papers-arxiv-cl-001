# Overview of the NLPCC 2026 Shared Task 11: Agent-Based Experiment Reproduction from Scientific Papers

Hanhua Hong1, Yizhi Li2, Luu Gia Huy3, Jian Yang4, Ming Zhou⁵, and Chenghua Lin1

1The University of Manchester, 2IQuest Research 3Vietnam National University Beihang University, Langboat

Abstract. Reproducibility is essential to scientific progress, yet the growing volume and complexity of scientific publications make exhaustive manual verification increasingly impractical. Although recent advances in large language model (LLM) agents enable automated experiment reproduction, existing evaluations largely focus on final repositories and are typically limited to machine learning (ML). We introduce AGENTACTIONBENCH, a process-oriented benchmark for evaluating agent-based experiment reproduction across ML and AI4Science domains. Our framework uses an MCP-based Action Recorder to capture agents' behaviour throughout the reproduction process and evaluates the resulting traces with paper-specific rubrics. AGENTACTION-BENCH contains 150 papers, including 120 ML papers and 30 AI4Science papers. A human-annotated subset covering 10% of the benchmark provides validation data, while model-assisted augmentation expands the full benchmark to more than 10,000 rubric items. Experimental results show that current systems remain limited, with execution as the primary bottleneck. Meanwhile, the strong Pearson and Spearman correlations between model-generated and human-annotated rubrics validate the reliability of our scalable rubric-generation approach.

Keywords: LLM Agents. Rubric-Based Evaluation· Data Augmentation

## 1 Introduction

Reproducibility is essential to trustworthy scientific progress, yet it remains difficult to achieve in practice. A Nature survey found that more than 70% of researchers had failed to reproduce another scientist's experiments, while recent studies indicate that only around 20% of papers at top computer science conferences release code sufficient for reproduction [9,19,23]. As scientific publications grow in both volume and complexity, manual verification at scale is becoming increasingly impractical.

Recent advances in large language models (LLMs) have substantially improved longcontext understanding, code generation, reasoning, and tool use. Consequently, LLM-based agents are increasingly applied to complex scientific workflows, from machine learning experimentation to AI for Science (AI4Science) tasks in domains such as biology and chemistry [4,10,16,22,36,38]. These capabilities make them promising candidates for automated experiment reproduction, which requires paper comprehension, implementation planning, code generation, experiment execution, and result verification.

Recent benchmarks, including Paper2Code [26] and PaperBench [29], evaluate whether AI systems can reproduce research papers. However, most of them focus on machine learning and assess final artefacts rather than the reproduction process. Such output-oriented evaluation can be unreliable since an agent may generate plausible files or descriptions while overlooking important paper content, failing to execute its code, or fabricating execution records. As illustrated in Fig. 1, this ambiguity makes it difficult to determine whether a reproduction is

![](images/7df36fe9d946a78375911455f9fc6c40e778bdc87ad0638a909ec5612d06ad42.jpg)  
Fig. 1: Illustration of output-oriented evaluation in prior benchmarks. When descriptions and implementations are evaluated together, LLM-based evaluators may incorrectly infer implementations that were never produced or executed.

![](images/04924fbfa6f1ad342462cd60f0ba26d8f15c7333cc01e0ede824d69de10174dd.jpg)  
Fig. 2: Overview of our framework. A standardised execution environment records agent actions, which are matched against paper-specific rubrics for auditable, process-oriented evaluation.

To address these limitations, we introduce a process-oriented benchmark for agent-based experiment reproduction. Rather than evaluating only the final repository, our framework decomposes reproduction into fine-grained stages and verifies whether agents perform the required operations. An MCP-based Action Recorder captures reading, writing, and commandexecution actions within a standardised environment. The resulting action logs provide auditable evidence for rubric-based evaluation, allowing each stage of the reproduction workflow to be assessed directly. Fig. 2 presents an overview of the framework.

We further extend evaluation beyond conventional machine learning papers to AI4Science research in astronomy, biology, chemistry, environmental science, materials science, and medicine. We first construct AGENTACTIONBENCH-HUMAN, a manually annotated subset containing 12 ML and 3 AI4Science papers. To reduce the cost of expert annotation, we develop an agent-assisted rubric-generation pipeline, validate it against the human annotations in AGENTACTIONBENCH-HUMAN, and scale the benchmark to the full AGENTACTION-BENCH, comprising 120 ML and 30 AI4Science papers with more than 10,000 rubric items. In summary, our contributions are threefold:

\- We introduce AGENTACTIONBENCH, a process-oriented, multi-domain benchmark that evaluates the complete experiment-reproduction workflow across 150 ML and AI4Science papers.

– We develop an MCP-based Action Recorder and a rubric-grounded evaluation protocol that use auditable action traces to assess paper understanding, implementation, execution, and result verification.

– We evaluate systems submitted to the NLPCC 2026 shared task, revealing key limitations of current agents in experiment reproduction and validating the reliability of our scalable rubric-generation pipeline.

## 2 Related Work

## 2.1 Agent-Based Experiment Reproduction

Autonomous scientific agents are increasingly studied for their ability to reproduce published experiments, an important capability for improving scientific efficiency and reproducibility. Existing systems address different stages of this process. The AI Scientist [21] supports an endto-end workflow from idea generation to paper writing; DOLPHIN [36] adopts a closed-loop process of ideation, implementation, and feedback; PaperCoder [26] converts research papers into executable repositories through a three-stage pipeline; and HiRAS [14] coordinates specialised sub-agents across fine-grained reproduction stages. A growing body of benchmarks evaluates these capabilities, including end-to-end ML reproduction [18,?,27,29], NLP-focused implementation from algorithmic descriptions [32,35], and general ML-engineering tasks based on large-scale experiments and public competitions [11,17] Their results consistently show that current agents remain far behind human experts, particularly when tasks require robust implementation and execution. However, most existing evaluations focus on coarse outcome-level metrics, provide limited visibility into the intermediate reproduction process, and primarily target a single domain. In contrast, our benchmark evaluates the complete workflow from paper understanding to result verification through auditable action traces, enabling fine-grained assessment across both ML and AI4Science papers.

## 2.2 Rubric-Based Evaluation

Rubrics provide structured supervision by decomposing complex tasks into fine-grained evaluation criteria, addressing the limitations of coarse, outcome-based metrics [13]. Benchmarks such as PaperBench [29] and HealthBench [8] use human-authored rubrics for stepwise evaluation, but their reliance on domain experts makes them costly to construct and difficult to scale. This limitation has motivated model-generated rubrics [3,20,37]. For example, RubricHub [3] combines principle-guided synthesis, multi-model aggregation, and difficulty evolution to produce discriminative criteria, while prior work shows that question-specific rubrics improve the accuracy and consistency of logical evaluation [1]. Building on these studies, we generate rubrics specifically for research-paper reproduction. Unlike methods designed for general reward modelling or open-ended question answering, our approach targets the multi-stage, repository-grounded nature of experiment reproduction.

Table 1: Tools provided by the Action Recorder. Optional arguments are enclosed in brackets.
<table><tr><td>Tool Name</td><td>Arguments</td><td>Results</td></tr><tr><td>Read</td><td>path: str[, start_offset: int, end_offset: int]</td><td>content: str</td></tr><tr><td>Write</td><td>path: str, content: str</td><td>success: bool</td></tr><tr><td>Execute</td><td>cmd: str</td><td>success: bool, stdout: str, stderr: str</td></tr></table>

## 2.3 LLM-Based Data Augmentation

Large language models have increasingly been used to generate training and benchmark data themselves, driven by the high cost of manual annotation and the need to scale supervision alongside larger models [2]. [2] survey this shift, organizing LLM-based augmentation methods into simple, prompt-based, and retrieval-based categories. One prominent line of work focuses on scaling instruction diversity to align models with human intent, from bootstrapping new instruction-response pairs from a small seed set [31] to progressively rewriting instructions into more complex variants [34]. [12] apply the same idea to a specific domain, prompting an LLM to expand existing geometry problem sets (GeoQA+, Geometry3K) into the larger Geo170K dataset. Together, these works show that LLM-driven generation can substitute for costly manual curation when scaling supervision or data diversity. We apply this same principle to automatically synthesise fine-grained, paper-specific rubrics for experiment reproduction, using an agent framework in place of the single-pass prompting adopted in prior augmentation work.

## 3 Methodology

## 3.1 Action Recorder

To faithfully capture agent behaviour during experiment reproduction, we implement an Action Recorder based on the Model Context Protocol (MCP). Rather than granting agents direct access to the underlying operating system, the recorder mediates all interactions with the execution environment through a small set of MCP tools. Each operation is therefore captured automatically as a structured action log, providing a complete and reproducible trace of the reproduction process.

As summarised in Table 1, the Action Recorder provides three core tools: Read, Write, and Execute. The Read tool retrieves file contents and optionally supports line-range selection for efficient inspection of large files. The Write tool creates or modifies source code, configuration files, and other artefacts required for reproduction. The Execute tool runs shell commands and returns the execution status, standard output, and standard error, enabling agents to conduct experiments and inspect their results. For safety and fairness, all tool interactions are restricted to a designated workspace.

The Action Recorder treats each tool invocation as an atomic, standardised action. For each invocation, it records the input arguments, returned results, and timestamp in a chronological JSON log, as illustrated in Appendix A. The log is automatically exported when the agent terminates. Because all interactions are mediated through the MCP interface, the recorder captures agent behaviour without modifying the agent and remains compatible with any MCP-enabled system. These structured logs provide the primary evidence for rubric-based evaluation, enabling fine-grained assessment of the reproduction process rather than relying solely on the final repository.

Table 2: Mapping between rubric types and the Action Recorder tools whose logs provide the corresponding evaluation evidence.
<table><tr><td>Rubric Type</td><td>Relevant Tools</td></tr><tr><td>Paper Observation</td><td>Read</td></tr><tr><td>Plan Writing</td><td>Write</td></tr><tr><td>Code Implementation</td><td>Write</td></tr><tr><td>Command Execution</td><td>Execute</td></tr><tr><td>Result Matching</td><td>Read, Execute</td></tr></table>

## 3.2 Benchmark Construction

We collect 150 papers to construct the full AGENTACTIONBENCH benchmark. Of these, 120 are drawn from top-tier machine learning conferences, including ICML, ICLR, ACL, and NeurIPS, while the remaining 30 are AI4Science papers published in leading scientific journals, such as Nature, Bioinformatics, and npj Computational Materials.

We adopt a rubric-based evaluation protocol in which each paper is paired with a paperspecific checklist for assessing reproduction quality. Formally, each rubric item is represented as $\boldsymbol { r } _ { i } = \left( c _ { i } , s _ { i } , t _ { i } \right)$ , where $c _ { i }$ denotes the evaluation criterion, $s _ { i }$ its importance score, and $t _ { i }$ the corresponding stage of the reproduction process. We define five rubric types: Paper Observation, Plan Writing, Code Implementation, Command Execution, and Result Matching. Together, these categories cover the full reproduction workflow and enable a more comprehensive assessment than approaches that consider only the consistency between a generated repository and the source paper.

We construct the human-annotated subset, AGENTACTIONBENCH-HUMAN, by randomly sampling 12 ML papers and 3 AI4Science papers, representing 10% of the full benchmark. Three research students with extensive experience in publishing at and reviewing for top-tier conferences annotate the rubrics. After the initial annotation, each rubric set is independently reviewed by the other two annotators in rotation and revised as needed. This process continues until all annotators reach consensus. The resulting subset contains an average of 67.75 rubric items per paper and more than 1,000 manually annotated items in total.

Building on recent advances in LLM-based data augmentation, we scale the benchmark by automatically generating rubrics for all papers. We first convert the papers to Markdown using MinerU [30], providing LLMs with cleaner and more structured inputs. We then apply the rubric-generation framework introduced in prior work [15] to produce paper-specific rubrics, expanding AGENTACTIONBENCH-HUMAN into the full AGENTACTIONBENCH benchmark. The resulting benchmark contains more than 10,000 rubric items, with an average of 69.92 items per paper. Finally, we assess the alignment between model-generated and humanannotated rubrics on AGENTACTIONBENCH-HUMAN using correlation analysis. Detailed results are presented in §5.2.

Table 3: Leaderboard results for the shared task. Our reproduction baseline is underscored.
<table><tr><td rowspan="2">Team</td><td colspan="3">Reproduction Score (%)</td></tr><tr><td>Machine Learning AI4Science Overall</td><td></td><td></td></tr><tr><td>YNU-HPCC-Task11-AgentRep</td><td>46.69</td><td>61.25</td><td>49.64</td></tr><tr><td>zzunlp_wu</td><td>22.97</td><td>31.50</td><td>24.70</td></tr><tr><td>Codex-GPT-5.4</td><td>22.89</td><td>29.26</td><td>24.19</td></tr><tr><td>QueenAgent</td><td>6.80</td><td>7.89</td><td>7.03</td></tr></table>

## 3.3 Evaluation

Our evaluation framework primarily relies on the action log L exported by the Action Recorder described in §3.1. For each paper ${ \mathcal P } ,$ the evaluator processes every rubric item ${ r _ { i } } = ( { { c } _ { i } } , { { s } _ { i } } , { { t } _ { i } } )$ To reduce context length and exclude irrelevant evidence, the evaluator model M first extracts a subset of logs $L _ { i } ^ { \prime } \subseteq L$ associated with the tools relevant to rubric type $t _ { i } .$ Table 2 specifies the mapping between each rubric type and its corresponding tools.

Given the extracted logs $L _ { i } ^ { \prime } ,$ rubric criterion $c _ { i } .$ , and source paper $\mathcal { P } _ { : }$ the evaluator assigns a binary judgment $J _ { i } \in \{ \mathrm { P a s s , F a i l } \}$ as $J _ { i } { = } \mathcal { M } ( L _ { i } ^ { \prime } , c _ { i } , \mathcal { P } )$ . The overall reproduction score is then computed as the importance-weighted proportion of passed rubric items:

$$
{ \mathrm { s c o r e } } { = } \frac { \bar { \sum _ { i } } [ \bar { J _ { i } } { = } \mathrm { P a s s } ] s _ { i } } { \sum _ { i } s _ { i } } ,\tag{1}
$$

where [-] equals 1 when the enclosed condition is satisfied and 0 otherwise.

## 4 Experimental Setup

Models. For rubric generation, we follow the configuration adopted in prior work [15]. Specifically, we use Claude Code [5] as the agentic scaffold, Claude-Sonnet [7] as the backbone model, and a skill document distilled from Claude-Opus [6]. For evaluation, we use ChatGPT-4o-mini [24] as the evaluator to reduce computational cost.

Benchmarks. Because AGENTACTIONBENCH-HUMAN is released as a training set for participants to develop and refine their models and systems, the main leaderboard is computed on the remaining papers in AGENTACTIONBENCH. However, since both human-annotated and model-generated rubrics are available for AGENTACTIONBENCH-HUMAN, we additionally collect participants' reproduction results on this subset to analyse the alignment between model-generated and human-authored rubrics.

Repositories. We evaluate submissions from three participating teams: YNU-HPCC-Task11- AgentRep, zzunlp\_wu, and QueenAgent. To provide an additional reference point, we also reproduce all 150 papers in AGENTACTIONBENCH using GPT-5.4 [28] with the Codex scaffold [25], which serves as our reproduction baseline. Correlation analyses on AGENTACTIONBENCH-HUMAN are conducted using the results from all participating teams and the baseline.

Environment. All experiments are conducted on two NVIDIA L40S GPUs.

Table 4: System performance across rubric types. Each value denotes the percentage of the maximum available score obtained for the corresponding rubric type. PO, PW, CI, CE, and RM denote Paper Observation, Plan Writing, Code Implementation, Command Execution, and Result Matching, respectively.
<table><tr><td rowspan="2">Team</td><td colspan="4">Obtained Score (%)</td></tr><tr><td>PO</td><td>PW</td><td>CI CE</td><td>RM</td></tr><tr><td>YNU-HPCC-Task11-AgentRep 43.73 82.45 61.8918.0417.24</td><td></td><td></td><td></td><td></td></tr><tr><td>zzunlp_wu</td><td></td><td>46.09 35.9129.1510.715.30</td><td></td><td></td></tr><tr><td>Codex-GPT-5.4</td><td></td><td>24.07 35.12 32.372.84 7.67</td><td></td><td></td></tr><tr><td>QueenAgent</td><td></td><td>52.34 3.07 0.95</td><td>0.00</td><td>0.39</td></tr></table>

Table 5: Correlation between evaluation scores produced by human-annotated and model-generated rubrics on AGENTACTIONBENCH-HUMAN, with 95% confidence intervals and summary statistics. Avg. cnt. denotes the average number of rubric items per paper, and Avg. len. denotes the average criterion length calculated by characters.
<table><tr><td rowspan="2">Rubric Source</td><td colspan="2">Correlation Coefficients</td><td colspan="2">Statistics</td></tr><tr><td>Pearson</td><td>Spearman</td><td>Avg. Cnt. Avg. Len.</td><td></td></tr><tr><td>Human</td><td></td><td></td><td>67.75</td><td>153.55</td></tr><tr><td>Model</td><td>0.93 ± 0.04 0.88 ± 0.07</td><td></td><td>69.92</td><td>180.82</td></tr></table>

## 5 Result Analysis

## 5.1 Leaderboard

Table 3 reports the performance of the participating systems and our reproduction baseline. YNU-HPCC-Task11-AgentRep performs best, achieving an overall score of 49.64%. However, no system exceeds 50%, and the Codex-GPT-5.4 baseline reaches only 24.19%, underscoring the difficulty of reproducing scientific experiments with current LLM-based agents. All systems also perform better on AI4Science papers than on ML papers, possibly because the selected AI4Science papers use more standard ML methods or simpler experimental configurations.

A breakdown by rubric type is provided in Table 4, where each percentage represents the score obtained relative to the maximum available for that category. Systems perform relatively well on Paper Observation, Plan Writing, and Code Implementation, but substantially worse on Command Execution and Result Matching. This gap indicates that execution remains a major bottleneck: current agents can often generate plausible plans and code, yet struggle with runtime failures, environment and dependency issues, and reproducing the reported results. These findings are consistent with prior studies identifying implementation and execution as key limitations of scientific agents [39,33,14].

## 5.2 Rubric Generation Analysis

Table 5 reports the correlations between the evaluation scores produced by human-annotated and model-generated rubrics on AGENTACTIONBENCH-HUMAN, together with their 95%

![](images/e5a417005eb2c169b6363769cd37862c4fdeb4acdafc8fc8ae52c7a57dcffc6f.jpg)

Fig. 3: Comparison of score distributions across rubric types for human-annotated and model-generated rubrics.

confidence intervals and summary statistics. The model-generated rubrics achieve Pearson and Spearman correlation coefficients of 0.93 and 0.88, respectively, both with narrow confidence intervals. These results demonstrate strong agreement with the human annotations and support the reliability of the proposed rubric-generation pipeline for scaling the benchmark. The two rubric sources also have similar average numbers of rubric items per paper, suggesting that the generated rubrics broadly capture the structure of the human annotations.

Figure 3 further shows that their score distributions are comparable across most rubric types. However, the model-generated criteria are more verbose on average and assign a larger share of the total score to Code Implementation, while assigning smaller shares to Plan Writing and Command Execution. These differences suggest that the generated rubrics could benefit from further calibration of their granularity and category emphasis, consistent with prior findings [15].

## 6 Conclusion

We introduced AGENTACTIONBENCH, a process-oriented benchmark for evaluating agentbased experiment reproduction across ML and AI4Science papers. Our framework combines an MCP-based Action Recorder with fine-grained, paper-specific rubrics, enabling auditable evaluation of the full reproduction process rather than relying solely on final repositories. To support evaluation at scale, we constructed a human-annotated subset and expanded it to 150 papers using an LLM-assisted rubric-generation pipeline across multiple domains. Experimental results show that current agents remain far from reliably reproducing scientific experiments, with execution and result verification emerging as the primary bottlenecks. Meanwhile, the strong agreement between model-generated and human-annotated rubrics supports the scalability of our benchmark construction approach.

## Limitations

Although rubric-based evaluation substantially reduces hallucinations, it cannot eliminate them entirely. Due to budget constraints, we use only GPT-4o-mini as the evaluator; future work should assess whether stronger models provide more reliable judgments. In addition, our coverage of AI4Science remains preliminary, and extending the benchmark to broader scientific domains is an important direction for future research.

## References

1. Rubric is all you need: Improving llm-based code evaluation with question-specific rubrics. In: Proceedings of the 2025 ACM Conference on International Computing Education Research V.1 (2025). https://doi.org/10.1145/3702652.3744220

2. Text data augmentation for large language models: A comprehensive survey of methods, challenges, and opportunities. Artificial Intelligence Review (2025)

3. Rubrichub: A comprehensive and highly discriminative rubric dataset via automated coarse-to-fine generation. arXiv preprint arXiv:2601.08430 (2026)

4. Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A.J., Bambrick, J., Bodenstein, S.W., Evans, D.A., Hung, C.C., O'Neill, M., Reiman, D., Tunyasuvunakool, K., Wu, Z., Žemgulytė, A., Arvaniti, E., Beattie, C., Bertolli, O., Bridgland, A., Cherepanov, A., Congreve, M., Cowen-Rivers, A.I., Cowie, A., Figurnov, M., Fuchs, F.B., Gladman, H., Jain, R., Khan, Y.A., Low, C.M.R., Perlin, K., Potapenko, A., Savy, P., Singh, S., Stecula, A., Thillaisundaram, A., Tong, C., Yakneen, S., Zhong, E.D., Zielinski, M., Žídek, A., Bapst, V., Kohli, P., Jaderberg, M., Hassabis, D., Jumper, J.M.: Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 630(8016), 493–500 (Jun 2024). https://doi.org/10.1038/s41586-024-07487-w, https://doi.org/10.1038/s41586-024-07487-w

5. Anthropic: Claude code overview (2026), https://code.claude.com/docs

6. Anthropic: Claude-opus system card (2026), https://www-cdn.anthropic.com/ 6a5fa276ac68b9aeb0c8b6af5fa36326e0e166dd.pdf

7. Anthropic: Claude-sonnet system card (2026), http://anthropic.com/claude-sonnet-4-6-system-card

8. Arora, R.K., Wei, J., Hicks, R.S., Bowman, P., Quiñonero-Candela, J., Tsimpourlas, F., Sharman, M., Shah, M., Vallone, A., Beutel, A., Heidecke, J., Singhal, K.: Healthbench: Evaluating large language models towards improved human health (2025), https://arxiv.org/abs/2505.08775

9. Baker, M.: 1,500 scientists lift the lid on reproducibility (2016)

10. Bran, A.M., Cox, S., Schilter, O., Baldassari, C., White, A.D., Schwaller, P.: Chemcrow: Augmenting large-language models with chemistry tools (2023), https://arxiv.org/abs/2304.05376

11. Chan, J.S., Chowdhury, N., Jaffe, O., Aung, J., Sherburn, D., Mays, E., Starace, G., Liu, K., Maksin, L., Patwardhan, T., Weng, L., Mądry, A.: MLE-bench: Evaluating machine learning agents on machine learning engineering. In: Proceedings of the International Conference on Learning Representations (ICLR) (2025)

12. Gao, J., Pi, R., Zhang, J., Ye, J., Zhong, W., Wang, Y., Hong, L., Han, J., Xu, H., Li, Z., Kong, L.: G-LLaVA: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370 (2023)

13. Gunjal, A., Wang, A., Lau, E., Nath, V., He, Y., Liu, B., Hendryx, S.M.: Rubrics as rewards: Reinforcement learning beyond verifiable domains. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=c1bTcrDmt4

14. Hong, H., LI, Y., Chen, J., Ananiadou, S., Li, X., jae Kim, J., Lin, C.: Hiras: A hierarchical multi-agent framework for paper-to-code generation and execution (2026), https://arxiv.org/abs/2604.17745

15. Hong, H., Li, Y., Chen, J., Huy, L.G., Ananiadou, S., jae Kim, J., Lin, C.: Can Illms write reliable rubrics? a meta-evaluation for experiment reproduction (2026), https://arxiv.org/abs/2607.12835

16. Hong, H., Xiao, C., Wang, Y., Liu, Y., Rong, W., Lin, C.: Beyond one-size-fits-all: Inversion learning for highly effective nlg evaluation prompts (2025), https://arxiv.org/abs/2504.21117

17. Huang, Q., Vora, J., Liang, P., Leskovec, J.: MLAgentBench: Evaluating language agents on machine learning experimentation. In: Proceedings of the 41st International Conference on Machine Learning (ICML). pp. 20271–20309 (2024)

18. Kim, G.J., Wilf, A., Morency, L.P., Fried, D.: From reproduction to replication: Evaluating research agents with progressive code masking. In: Proceedings of the International Conference on Learning Representations (ICLR). Carnegie Mellon University, Pittsburgh, PA, USA (2026)

19. Lin, J., Wang, Y., Yu, Y., Zhou, Y., Chen, Y., Shi, X.: Automatic analysis of available source code of top artificial intelligence conference papers. International Journal of Software Engineering and Knowledge Engineering 32(07), 947–970 (2022). https://doi.org/10.1142/S0218194022500358, https://doi.org/10.1142/S0218194022500358

20. Liu, T., Xu, R., Yu, T., Hong, I., Yang, C., Zhao, T., Wang, H.: Openrubrics: Towards scalable synthetic rubric generation for reward modeling and llm alignment (2026), https://arxiv.org/abs/2510.07743

21. Lu, C., Lu, C., Lange, R.T., Foerster, J.N., Clune, J., Ha, D.: The ai scientist: Towards fully automated open-ended scientific discovery. CoRR abs/2408.06292 (2024), https://doi.org/10.48550/arXiv.2408.06292

22. Lyu, Y., Zhang, X., Yi, X., Zhao, Y., Guo, S., Hu, W., Piotrowski, J., Kaliski, J., Urbani, J., Meng, Z., Zhou, L., Yan, X.: Evoscientist: Towards multi-agent evolving ai scientists for end-to-end scientific discovery (2026), https://arxiv.org/abs/2603.08127

23. Magnusson, I., Smith, N.A., Dodge, J.: Reproducibility in NLP: What have we learned from the checklist? In: Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Findings of the Association for Computational Linguistics: ACL 2023. pp. 12789–12811. Association for Computational Linguistics, Toronto, Canada (Jul 2023). https://doi.org/10.18653/v1/2023.findings-acl.809, https://aclanthology.org/2023.findings-acl.809/

24. OpenAI: Gpt-4o system card (2024), https://arxiv.org/abs/2410.21276

25. OpenAI: Introducing codex (2025), https://openai.com/index/introducing-codex/

26. Seo, M., Baek, J., Lee, S., Hwang, S.J.: Paper2code: Automating code generation from scientific papers in machine learning (2025), https://arxiv.org/abs/2504.17192

27. Siegel, Z.S., Kapoor, S., Nadgir, N., Stroebl, B., Narayanan, A.: CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark. Transactions on Machine Learning Research (01 2025), https://openreview.net/forum?id=BsMMc4MEGS

28. Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., et al.: Openai gpt-5 system card (2026), https://arxiv.org/abs/2601.03267

29. Starace, G., Jaffe, O., Sherburn, D., Aung, J., Chan, J.S., Maksin, L., Dias, R., Mays, E., Kinsella, B., Thompson, W., Heidecke, J., Glaese, A., Patwardhan, T.: Paperbench: Evaluating ai's ability to replicate ai research (2025), https://arxiv.org/abs/2504.01848

30. Wang, B., He, T., Ouyang, L., Wu, F., Zhao, Z., Chu, T., Qu, Y., Jin, Z., Zeng, W., Miao, Z., et al.: Mineru2. 5-pro: Pushing the limits of data-centric document parsing at scale. arXiv preprint arXiv:2604.04771 (2026)

31. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N.A., Khashabi, D., Hajishirzi, H.: Self-instruct: Aligning language models with self-generated instructions. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (2023)

32. Xiang, Y., Yan, H., Ouyang, S., Gui, L., He, Y.: SciReplicate-Bench: Benchmarking llms in agent-driven algorithmic reproduction from research papers. In: Proceedings of the Conference on Language Modeling (COLM). King's College London and The Alan Turing Institute (2025)

33. Xie, Q., Weng, Y., Zhu, M., Shen, F., Huang, S., Lin, Z., Zhou, J., Mao, Z., Yang, Z., Yang, L., Wu, J., Zhang, Y.: How far are ai scientists from changing the world? (2025), https://arxiv.org/abs/2507.23276

34. Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Jiang, D.: WizardLM: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244 (2023)

35. Yan, S., Li, R., Luo, Z., Wang, Z., Li, D., Jing, L., He, K., Wu, P., Ni, J., Michalopoulos, G., Zhang, Y., Zhang, Z., Zhang, M., Chen, Z., Du, X.: LMR-BENCH: Evaluating LLM agent's ability on reproducing language modeling research. In: Christodoulopoulos, C., Chakraborty, T., Rose, C., Peng, V. (eds.) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 6164–6186. Association for Computational Linguistics, Suzhou, China (Nov 2025). https://doi.org/10.18653/v1/2025.emnlp-main.314, https://aclanthology.org/2025.emnlp-main.314/

36. Yuan, J., Yan, X., Zhang, B., Chen, T., Shi, B., Ouyang, W., Qiao, Y., Bai, L., Zhou, B.: Dolphin: Moving towards closed-loop auto-research through thinking, practice, and feedback. In: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.) Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 21768–21789. Association for Computational Linguistics, Vienna, Austria (Jul 2025). https://doi.org/10.18653/v1/2025.acl-1ong.1056, https://aclanthology.org/2025.acl-long.1056/

37. Zhang, Q., Zhou, J., Wang, Y., Lyu, F., Ming, Y., Xu, C., Sun, Q., Zheng, K., Kang, P., Liu, X., Ma, C.: Rubricbench: Aligning model-generated rubrics with human standards (2026), https://arxiv.org/abs/2603.01562

38. Zhao, X., Sang, Z., Li, Y., Shi, Q., Zhao, W., Wang, S., Zhang, D., Han, X., Liu, Z., Sun, M.: Autoreproduce: Automatic ai experiment reproduction with paper lineage (2025), https://arxiv.org/abs/2505.20662

39. Zhu, M., Xie, Q., Weng, Y., Wu, J., Lin, Z., Yang, L., Zhang, Y.: Ai scientists fail without strong implementation capability (2025), https://arxiv.org/abs/2506.01372

## A Log Example

[   
{   
"id": 1,   
"timestamp": .,   
"tool": "Read",   
"arguments": {   
"path": "paper.md",   
"start\_offset": null,   
"end\_offset": null   
},   
"result": {   
"content": "# My Paper\n## Abstract \n...",   
},   
{   
"id": 2,   
"timestamp": "...",   
"tool": "Write",   
"arguments": {   
"path": "model.py",   
"content": "import torch\nimport torch.nn as nn\n...'   
},   
"result": {   
"success": true   
},   
1   
"id": 3,   
"timestamp": "...",   
"tool": "Execute",   
"arguments": {   
"cmd": "python train.py —epochs 10"   
}, "result": {   
"stdout": "Epoch 1/10 loss=0.45\nEpoch 2/10 loss=0.32\n...",   
"stderr": "",   
"success": true   
},   
}   
1  
Fig. 4: An example of Action Recorder logs.