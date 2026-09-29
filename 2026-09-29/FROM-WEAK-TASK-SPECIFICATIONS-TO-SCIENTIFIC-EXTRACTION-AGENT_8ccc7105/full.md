# FROM WEAK TASK SPECIFICATIONS TO SCIENTIFIC EXTRACTION AGENTS: OPTIMIZING TASK CONSTRUCTION

Zixiao Dong<sup>1,3</sup>, Wei Yang<sup>2,3</sup>, Zihao Liu<sup>1,3</sup>, Chenshu Li<sup>1,3</sup>, Longzhang Liu<sup>1,3</sup>, Tao Tan<sup>4</sup>, Hong Xie<sup>1,3∗</sup>

<sup>1</sup>School of Computer Science and Technology, University of Science and Technology of China <sup>2</sup>University of Science and Technology of China, <sup>3</sup>State Key Laboratory of Cognitive Intelligence <sup>4</sup>CCCC Second Highway Consultants Co., Ltd.

## ABSTRACT

Most methods that optimize LLM prompts and agent workflows assume that task-specific output schemas, extraction instructions, and evaluation criteria are predefined. For scientific extraction agents, however, a short task goal may not fully determine these components, while specifying them manually is costly. We study the upstream problem of constructing the task-specific configuration from a weak specification containing only a short goal and unannotated reference documents. Rather than treating automatic construction as a fixed preprocessing step, our framework constructs a task-specific schema, extraction instructions, and base training rubrics, then keeps schema construction and extraction instructions editable during optimization. Failure-focused updates concentrate textual-gradient feedback on lower-scoring documents, while training-time evaluation criteria adapt to recurring failures. On a heterogeneous-catalysis literature corpus, automatic construction remains improvable, and optimizing both schema construction and extraction instructions performs best across all four judge–rubric settings, with ablations and blinded human evaluation supporting the proposed formulation.

Index Terms— weak task specifications, large language model agents, automatic task construction, scientific information extraction, prompt optimization

## 1. INTRODUCTION

Scientific extraction agents convert full papers into structured, taskspecific records. This is difficult because relevant evidence can be distributed across body text, tables, captions, and distant sections [1, 2], while extracted information often depends on relations and contextual qualifiers needed for correct interpretation [3, 4]. In practice, such an agent also depends on task-specific choices about the output structure, extraction instructions, and evaluation criteria. Users may know the information they seek without being able to fully specify these components in advance, while manually constructing them often requires substantial effort and domain knowledge. We therefore define a weak task specification as a short goal paired with unannotated reference documents, without a predefined task configuration.

Prompt optimization methods revise task-specific instructions using model feedback or search [5–7], while program optimizers extend this idea to modular and multi-stage LLM pipelines [8,9]. More recent work optimizes agent-level functions and prompts [10–12]. Existing prompt and program optimizers typically improve components within an already instantiated task-specific system. We instead study the upstream problem of constructing that system from a weak task specification alone. Because such a specification does not fully determine the task-specific configuration, we treat automatic construction as an initialization rather than a fixed preprocessing step.

We define automatic task construction as generating a taskspecific schema, extraction instructions, and base training rubrics from a weak task specification. Failure-focused optimization then updates the constructed components using lower-scoring documents and adapts the training rubric to recurring failures. Our contributions are:

• We formulate task construction itself as an optimization problem under weak task specifications, with editable schema construction and extraction instructions and adaptive training rubrics.

• We introduce failure-focused optimization that concentrates updates on poorly performing documents and adapts trainingtime evaluation criteria to recurring failures, while decoupling adaptive training feedback from frozen-rubric selection.

• Experiments on heterogeneous-catalysis literature show that both schema construction and extraction instructions remain improvable after initialization, and that sequentially optimizing them performs best across all four judge–rubric settings, supported by ablations and blinded human evaluation.

## 2. RELATED WORK

Earlier scientific information extraction systems relied on schemas, ontologies, rules, and task-specific pipelines [3, 4, 13]. Recent LLM-based methods support entity–relation, scientific-contribution, and material-property extraction [14–16]. Systems using task instructions or annotation guidelines extend IE to user-specified or unseen tasks [17,18]. Full-document extraction remains challenging because task-relevant evidence is often distributed across a paper. Collage [1] addresses this through document-aware pipeline design, while AgentCAT [2] combines progressive schema evolution with an evidence-grounded candidate–resolve–review agent workflow. In contrast, we make task construction itself an optimization target under weak specifications, optimizing both schema construction and extraction instructions.

Automatic prompt optimization revises instructions or languagemodel programs using model feedback, search, or other optimization signals. ProTeGi [5] uses natural-language gradients, and OPRO [6] treats an LLM as an optimizer. DSPy and MIPRO extend optimization to modular and multi-stage programs [8, 9], while TextGrad [7] propagates natural-language feedback to editable components. Recent work further extends optimization to agent-level components: AgentOptimizer treats agent functions as learnable parameters [10], AutoPDL optimizes agent prompts [11], GEPA uses reflective prompt evolution [12], and multi-judge optimization combines feedback from several evaluators [19]. These methods, like instruction-based IE [17, 18], assume an instantiated task-specific program. We instead make task construction itself an optimization target.

Evaluation criteria may also need to evolve during optimization. DynamicRubric [20] co-evolves an evaluator and policy using rubrics conditioned on current-policy responses. In our setting, stage-specific base training rubrics are generated from the task goal and extended with criteria for recurring failures, while checkpoints are selected using a frozen final-round rubric on held-out documents.

## 3. METHOD

Given a weak task specification consisting of a short goal g and unannotated reference documents $R s$ , our goal is to construct and optimize a scientific extraction agent by generating an initial runtime schema $S _ { 0 } ,$ extraction prompts $P _ { 0 } .$ , and stage-specific base training rubrics. Figure 1 summarizes the overall framework. We denote the deployable extraction policy by $\theta = ( S , P )$ , where S and P denote the runtime schema and extraction prompts, respectively; the initialized policy is therefore $\theta _ { 0 } = ( S _ { 0 } , P _ { 0 } )$ . Optimization operates on two editable components: the schema-construction policy ϕ and $P ,$ with $\phi$ producing S. We index optimization rounds by $t \in \{ 0 , \ldots , T \}$ where $t = 0$ denotes initialization, and use $k \in \{ \mathrm { s c h } , \mathrm { e x t } \}$ for the schema-construction and extraction stages.

## 3.1. Schema construction from weak specifications

At round t, schema construction is controlled by

$$
\phi _ { t } = ( P _ { \mathrm { p l a n } } ^ { t } , P _ { \mathrm { e v o } } ^ { t } )\tag{1}
$$

Here, $P _ { \mathrm { p l a n } } ^ { t }$ and $P _ { \mathrm { e v o } } ^ { t }$ denote the planning and schema-evolution prompts. The fixed, domain-independent meta-prompt $M _ { \phi }$ generates the initial prompts $\phi _ { 0 }$ and the schema-stage base rubric $J _ { 0 } ^ { \mathrm { s c h } }$ from $g .$ The initial planning prompt maps g to a provisional task framework $F _ { 0 }$ . A deterministic compiler converts $F _ { 0 }$ into the provisional schema $\widetilde { S } _ { 0 }$ . The initial evolution prompt then refines this schema against $R _ { S } { \mathrm { : } }$

$$
g ~ { \xrightarrow { P _ { \mathrm { p l a n } } ^ { 0 } } } ~ F _ { 0 } ~ { \xrightarrow { \mathrm { c o m p i l e } } } ~ { \widetilde { S } } _ { 0 } ~ { \xrightarrow { R _ { S } , P _ { \mathrm { e v o } } ^ { 0 } } } ~ S _ { 0 }\tag{2}
$$

At later rounds, $S _ { t } ~ = ~ S ( \phi _ { t } ; g , R _ { S } )$ denotes the schema regenerated by the current construction prompts. The goal specifies the task intent, while $R _ { S }$ supplies document-grounded detail for schema refinement.

## 3.2. Extraction-prompt generation and execution

From the short goal $^ { g , }$ the fixed meta-prompt $M _ { P }$ generates the stage-specific extraction prompts

$$
P _ { 0 } = ( P _ { \mathrm { c a n d } } ^ { 0 } , P _ { \mathrm { r e s } } ^ { 0 } , P _ { \mathrm { r e v } } ^ { 0 } )\tag{3}
$$

Here, $P _ { \mathrm { c a n d } } ^ { 0 } , P _ { \mathrm { r e s } } ^ { 0 }$ , and $P _ { \mathrm { r e v } } ^ { 0 }$ are the candidate-generation, evidenceresolution, and review prompts. The same initialization step produces the extraction-stage base rubric $J _ { 0 } ^ { \mathrm { e x t } }$ . These extraction prompts are generated from g independently of S<sub>0</sub> and paired with the runtime schema during execution.

During execution, each schema section follows the AgentCAT [2] workflow of candidate generation, evidence-only resolution, and PDF-grounded review. These stages collect supporting evidence, map it to structured values, and verify and correct the values against the full PDF before assembling the final record.

## 3.3. Failure-focused textual-gradient optimization

We use TextGrad [7] as the underlying prompt optimizer, while modifying the feedback process in two ways: updates focus on lower-scoring documents, and the training-time evaluation criteria can adapt to recurring failures. The set $D _ { \mathrm { u p d } } ^ { k }$ contains the documents used for optimization feedback at stage k. Schema-TG regenerates $S _ { t }$ from the updated construction prompts, whereas Extract-TG jointly updates the editable prompt components in $P _ { t }$ under a fixed schema. For update document $d _ { i } ,$ the stage evaluator returns score $q _ { i , t } ^ { k }$ and natural-language diagnosis $F _ { i , t } ^ { k }$ . At the schema stage, it assesses whether the constructed schema can adequately represent the task-relevant information supported by $d _ { i } .$ , without executing the full extraction workflow. We denote the lower-scoring half of the update set by $\mathcal { H } _ { t } ^ { k }$

$$
\mathcal { H } _ { t } ^ { k } = \mathrm { B o t t o m H a l f } \left( D _ { \mathrm { u p d } } ^ { k } ; \{ q _ { i , t } ^ { k } \} \right)\tag{4}
$$

Only diagnoses from $\mathcal { H } _ { t } ^ { k }$ are passed to TextGrad, so each update focuses on the currently lower-scoring documents.

Each stage starts from a fixed base rubric $J _ { 0 } ^ { k }$ and maintains an adaptive supplement $\Delta _ { t } ^ { k }$

$$
J _ { t } ^ { k } = J _ { 0 } ^ { k } \oplus \Delta _ { t } ^ { k }\tag{5}
$$

Here, ⊕ augments the base rubric with the active adaptive criteria in $\Delta _ { t } ^ { k }$ , whose capacity is bounded by

$$
| \Delta _ { t } ^ { k } | \leq C\tag{6}
$$

with $C$ the maximum number of active adaptive criteria. After each round, the evaluator summarizes the observed errors and may update $\Delta _ { t } ^ { k }$ with criteria for recurring or consequential failure modes. Redundant criteria refresh existing entries, while inactive criteria may be retired. The adaptive rubric supplies optimization feedback; checkpoint scores are produced separately as described next.

## 3.4. Frozen-rubric checkpoint selection

Since the training rubric evolves across rounds, scores produced under different round-specific rubrics are not directly comparable. This separation prevents checkpoint ranking from being confounded by rubric drift across optimization rounds. After optimization, the finalround training rubric $J _ { T } ^ { k }$ is frozen and used as the selection rubric, $J _ { \mathrm { s e l } } ^ { k } : = J _ { T } ^ { k }$ . Every saved checkpoint is then rescored on the held-out set $D _ { \mathrm { s e l } } ^ { k }$ under this common rubric. For a document d and stage artifact $\bar { A } , J _ { \mathrm { s e l } } ^ { k } ( d , A )$ returns a scalar score. The function $E ( d ; S , P _ { t } )$ denotes the record extracted from d with schema S and prompts $P _ { t } .$ We save the stage state after each round as checkpoint t. Schema and extraction checkpoints are selected separately:

$$
\phi _ { \mathrm { b e s t } } = \underset { \phi _ { t } } { \arg \operatorname* { m a x } } \frac { 1 } { | D _ { \mathrm { s e l } } ^ { \mathrm { s c h } } | } \sum _ { d \in D _ { \mathrm { s e l } } ^ { \mathrm { s c h } } } J _ { \mathrm { s e l } } ^ { \mathrm { s c h } } \big ( d , S ( \phi _ { t } ; g , R _ { S } ) \big )\tag{7}
$$

$$
P _ { \mathrm { b e s t } } ( S ) = \underset { P _ { t } } { \arg \operatorname* { m a x } } \frac { 1 } { | D _ { \mathrm { s e l } } ^ { \mathrm { e x t } } | } \sum _ { d \in D _ { \mathrm { s e l } } ^ { \mathrm { e x t } } } J _ { \mathrm { s e l } } ^ { \mathrm { e x t } } \left( d , E ( d ; S , P _ { t } ) \right)\tag{8}
$$

Here, $\phi _ { \mathrm { b e s t } }$ is the selected schema-construction checkpoint. The term $P _ { \mathrm { b e s t } } ( S )$ is the extraction-prompt checkpoint selected for a fixed schema S. Let $S _ { \mathrm { b e s t } } ~ = ~ S ( \phi _ { \mathrm { b e s t } } ; g , R _ { S } )$ and $\begin{array} { r l } { P _ { \mathrm { b e s t } } } & { { } = } \end{array}$ $P _ { \mathrm { b e s t } } ( S _ { \mathrm { b e s t } } )$ . The final deployable policy is $\theta _ { \mathrm { b e s t } } = ( S _ { \mathrm { b e s t } } , P _ { \mathrm { b e s t } } )$

![](images/23226cfc734c5e4889ed7bc44c44c42ae1bf19d564bbe08e4b5aa36a48a88e1b.jpg)  
Fig. 1. Overview of the proposed weak-specification task-construction framework. This framework constructs a task-specific policy from weak specifications, refines it through failure-focused optimization, and selects the final policy using a frozen rubric.

## 4. EXPERIMENTS

## 4.1. Experimental Setup

The corpus consists of unannotated heterogeneous-catalysis papers spanning experimental catalysis, spectroscopy, density-functional theory, mechanistic analysis, and kinetic modeling. Twelve PDFs are used for schema refinement. Each optimization stage additionally uses four update documents and two selection documents, with final evaluation on a separate held-out 50-paper target corpus.

GPT-5.6 Terra (temperature 0; high reasoning effort) is used for extraction and optimization-time evaluation, with native PDF inputs and five textual-gradient updates per stage. Auto-init deploys the generated schema and extraction prompts without optimization. Schema-TG updates schema construction only, and Extract-TG updates extraction instructions only. Joint-TG sequentially optimizes both stages. It first selects an optimized schema-construction checkpoint and then optimizes extraction instructions under the resulting schema. To isolate stage effects, Extract-TG reuses the Auto-init schema, while Joint-TG reuses the selected Schema-TG schema before extraction-instruction optimization, keeping the schema fixed within each paired comparison. We compare with the original AgentCAT [2] and an adapted AgentCAT-short variant. AgentCAT uses a manually designed task specification, extraction prompts, and task-specific evaluation criteria; AgentCAT-short keeps all other AgentCAT components unchanged and replaces only its detailed task specification with our six-word goal.

All methods use the same base model and target corpus. Within each stage, optimized variants share the same five-update budget, document sets, and evaluation protocol, differing only in whether schema construction, extraction instructions, or both are optimized.

The feedback-selection ablation keeps these settings fixed and changes only whether all document diagnoses or the lower-scoring half are passed to the optimizer. For adaptive training rubrics, we set $C = 3 .$ . At each round, the evaluator may propose zero to two new criteria. When the capacity is exceeded, criteria triggered most frequently and recently are retained; a criterion is retired after three inactive rounds. Code is available at https://github.com/ yyhlm/weak-spec-extraction-agents.

## 4.2. Evaluation

To evaluate the proposed method, we assess full-paper extraction using two expert-designed, fixed multidimensional rubrics. R1 assesses record-level extraction quality through coverage, accuracy, specificity, and usability, weighted 35%, 35%, 20%, and 10%, respectively, whereas R2 assesses scientific-context fidelity through causal linking, parameter alignment, mechanistic fidelity, and micro–macro distinction, weighted 30%, 30%, 25%, and 15%, respectively. Each dimension is scored on a 0–100 scale, and each rubric score is computed as the weighted average of its dimensions.

Following the LLM-as-a-judge paradigm [21], two PDF-aware judges evaluate identical extraction files while blinded to method and checkpoint identity. Judge-T uses GPT-5.6 Terra and Judge-5.5 uses GPT-5.5, both with temperature 0 and high reasoning effort, yielding four judge–rubric evaluation settings.

Blinded human spot-checks are additionally conducted on ten papers spanning five study types. Joint-TG is compared separately with Auto-init and AgentCAT in balanced A/B order, with assessors consulting the source PDFs and independently judging record-level quality and scientific-context fidelity.

## 4.3. Main results

Table 1 shows the main results. Auto-init averages 78.37, close to AgentCAT-short (79.18) and AgentCAT (79.81), showing that weak-specification task construction is already competitive. Joint-TG reaches 83.40 and leads all four judge–rubric settings. Without a manually specified task configuration, Joint-TG surpasses both AgentCAT baselines with manually developed prompts and evaluation rubrics. Relative to Auto-init, sequential optimization improves the four-score mean by 5.03 points. The stage-wise variants show that both components can be further improved after automatic construction, with extraction-instruction optimization producing the larger single-stage gain.

Table 1. Evaluation scores on the complete 50-PDF target corpus.
<table><tr><td>Method</td><td>T-R1</td><td>T-R2</td><td>5.5-R1</td><td>5.5-R2</td></tr><tr><td>Auto-init</td><td>79.63</td><td>77.42</td><td>79.02</td><td>77.40</td></tr><tr><td>AgentCAT-short [2]</td><td>81.55</td><td>78.93</td><td>79.05</td><td>77.20</td></tr><tr><td>AgentCAT [2]</td><td>81.04</td><td>80.49</td><td>79.52</td><td>78.17</td></tr><tr><td>Schema-TG (best)</td><td>81.68</td><td>81.63</td><td>80.95</td><td>79.02</td></tr><tr><td>Extract-TG (best)</td><td>83.63</td><td>82.90</td><td>81.30</td><td>80.75</td></tr><tr><td>Joint-TG (best/best)</td><td>84.50</td><td>84.38</td><td>83.67</td><td>81.05</td></tr></table>

For Joint-TG versus Auto-init, 95% CIs are estimated using 100,000 paired percentile bootstrap resamples over the 50 target papers. Under Judge-T, the R1 and R2 improvements are +4.88 [1.35, 8.98] and +6.95 [3.40, 10.95], respectively. Under Judge-5.5, they are +4.65 [1.73, 8.23] and +3.65 [0.25, 7.48]. All four 95% bootstrap intervals lie above zero, indicating consistent gains across both judges and both rubrics. For Joint-TG versus AgentCAT, the paired 95% CIs are [1.65, 5.42] and [1.94, 5.91] for Judge-T R1 and R2, respectively. For Judge-5.5, the corresponding intervals are [1.55, 7.23] and [0.63, 6.68].

In both blinded human comparisons, Joint-TG is preferred on all 10 papers for record-level quality and on 9 of 10 for scientificcontext fidelity. For each comparison, two-sided sign tests give p = 0.0020 and $p = 0 . 0 2 1 5$ , respectively.

## 4.4. Optimization-protocol ablations

Table 2 separates feedback selection, rubric adaptation, and checkpoint selection. Here, best/best denotes held-out selection at both stages, whereas final/final uses the last checkpoint from each stage.

Table 2. Judge-T ablations of feedback selection, rubric adaptation, and checkpoint selection.
<table><tr><td>Rubric</td><td>Feedback + checkpoint</td><td>R1</td><td>R2</td></tr><tr><td>Adaptive</td><td>All + best/best</td><td>81.91</td><td>80.29</td></tr><tr><td>Fixed</td><td>Bottom + best/best</td><td>82.15</td><td>81.98</td></tr><tr><td>Adaptive</td><td>Bottom + final/final</td><td>82.43</td><td>81.10</td></tr><tr><td>Adaptive</td><td>Bottom + best/best</td><td>84.50</td><td>84.38</td></tr></table>

Under adaptive training and frozen-rubric selection, restricting updates to the lower-scoring half improves Judge-T R1 from 81.91 to 84.50 and R2 from 80.29 to 84.38 relative to using all document diagnoses. The paired differences are +2.59 [−0.43, 5.88] and +4.09 [1.53, 6.71], respectively, suggesting a benefit from focusing feedback on lower-scoring documents, particularly for R2.

Holding bottom-half feedback and checkpoint selection fixed, adaptive training criteria raise R1 from 82.15 to 84.50 and R2 from 81.98 to 84.38. The paired gains are +2.35 [0.13, 4.85] and +2.40 [0.45, 4.40]. This comparison shows that the automatically constructed base rubric is already usable, while criteria induced by recurring failures provide additional optimization signal. Within the adaptive bottom-half run, held-out checkpoint selection improves R1 by 2.07 points and R2 by 3.28 points over using the final checkpoints, suggesting that the learned selection criterion is broadly aligned with the expert-designed evaluation criteria.

Across five update rounds, the adaptive rubric introduced four supplemental criteria, three of which remained active in the finalround rubric: cross-record reference integrity, experimental scope and field semantics, and output integrity. These criteria respectively target resolvable links across records, correct assignment of results to experimental scope and schema fields, and malformed, truncated, or schema-invalid JSON. A fourth criterion, evidence-and-entity consistency, was introduced in the first round and retired after three inactive rounds. This pattern suggests that optimization primarily exposed failures in cross-record consistency, experimental-context assignment, and output validity, while the retirement mechanism prevented inactive criteria from accumulating.

## 4.5. Gain and error analysis

Across the eight rubric dimensions, Joint-TG matches or exceeds Auto-init under both evaluators. Judge-T reports its largest gains in mechanistic fidelity (+8.00), causal linking (+7.50), and parameter alignment (+6.50); Judge-5.5 reports its largest gains in coverage (+10.00), causal linking (+6.00), and specificity (+4.00). Both judges therefore identify improvements in information recovery and in preserving relations among values, conditions, and scientific claims. The smaller gains in usability (+1.00/0.00) indicate that Auto-init already produced structurally usable records.

Document-level diagnostics reveal residual errors masked by the averages. In one dehydrogenation–cracking paper, Joint-TG misidentifies a reaction step and misattributes a literature-derived barrier. In a hexadiene-cyclization study, it omits part of a key energy map. These cases indicate remaining challenges in mechanistic grounding and evidence coverage.

## 5. CONCLUSION

We study how task-specific configurations for scientific extraction agents can be automatically constructed and optimized from weak specifications. Starting from only a short task goal and unannotated reference documents, the framework constructs a task-specific schema, extraction instructions, and base training rubrics, then refines the schema-construction policy and extraction instructions through failure-focused optimization while adapting training-time evaluation criteria to recurring failures.

On a 50-paper catalysis corpus, automatic construction is competitive with manually specified baselines, and both components remain improvable. Extraction-instruction optimization yields the larger single-stage gain, while schema-construction optimization provides additional improvement when the two stages are optimized sequentially. Ablations further show gains from adaptive training criteria and held-out checkpoint selection. Joint-TG achieves the highest mean across all four judge–rubric settings, and blinded human evaluation also favors Joint-TG over Auto-init and AgentCAT. Together, these results support treating task construction itself as an optimization target for scientific extraction from weak specifications.

Compliance with Ethical Standards. This study uses publicly available scientific literature for which no ethical approval was required.

Acknowledgments. This work was supported by the Strategic Priority ResearchProgram of the Chinese Academy of Sciences (Grant No. XDA0490000). The authors declare no conflicts of interest.

## 6. REFERENCES

[1] Sireesh Gururaja, Yueheng Zhang, Guannan Tang, Tianhao Zhang, Kevin Murphy, Yu-Tsen Yi, Junwon Seo, Anthony Rollett, and Emma Strubell, “Collage: Decomposable rapid prototyping for co-designed information extraction on scientific PDFs,” in Proceedings of the Fifth Workshop on Scholarly Document Processing, 2025, pp. 72–82.

[2] Wei Yang, Zihao Liu, Tao Tan, Xiao Hu, Hong Xie, Lulu Li, Xin Li, Jianyu Han, Defu Lian, and Mao Ye, “AgentCAT: An LLM agent for extracting and analyzing catalytic reaction data from chemical engineering literature,” arXiv preprint arXiv:2602.18479, 2026.

[3] Juraj Mavraciˇ c, Callum J. Court, Taketomo Isazawa,´ Stephen R. Elliott, and Jacqueline M. Cole, “ChemDataExtractor 2.0: Autopopulated ontologies for materials science,” Journal of Chemical Information and Modeling, vol. 61, no. 9, pp. 4280–4289, 2021.

[4] Sarthak Jain, Madeleine van Zuylen, Hannaneh Hajishirzi, and Iz Beltagy, “SciREX: A challenge dataset for documentlevel information extraction,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 7506–7516.

[5] Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng, “Automatic prompt optimization with “gradient descent” and beam search,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 7957–7968.

[6] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen, “Large language models as optimizers,” in International Conference on Learning Representations, 2024.

[7] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou, “Optimizing generative AI by backpropagating language model feedback,” Nature, vol. 639, pp. 609–616, 2025.

[8] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts, “DSPy: Compiling declarative language model calls into self-improving pipelines,” International Conference on Learning Representations, 2024.

[9] Krista Opsahl-Ong, Michael J. Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab, “Optimizing instructions and demonstrations for multi-stage language model programs,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 9340–9366.

[10] Shaokun Zhang, Jieyu Zhang, Jiale Liu, Linxin Song, Chi Wang, Ranjay Krishna, and Qingyun Wu, “Offline training of language model agents with functions as learnable weights,” in

Proceedings of the 41st International Conference on Machine Learning, 2024, vol. 235 of Proceedings of Machine Learning Research, pp. 60315–60335.

[11] Claudio Spiess, Mandana Vaziri, Louis Mandel, and Martin Hirzel, “AutoPDL: Automatic prompt optimization for LLM agents,” in Proceedings ofthe Fourth International Conference on Automated Machine Learning, 2025, vol. 293 of Proceedings ofMachine Learning Research, pp. 13/1–20.

[12] Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alex Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab, “GEPA: Reflective prompt evolution can outperform reinforcement learning,” in International Conference on Learning Representations, 2026.

[13] Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi, “Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, pp. 3219–3232.

[14] John Dagdelen, Alexander Dunn, Sanghoon Lee, Nicholas Walker, Andrew S. Rosen, Gerbrand Ceder, Kristin A. Persson, and Anubhav Jain, “Structured information extraction from scientific text with large language models,” Nature Communications, vol. 15, 2024, Art. no. 1418.

[15] Mahsa Shamsabadi, Jennifer D’Souza, and Soren Auer, “Large¨ language models for scientific information extraction: An empirical study for virology,” in Findings of the Association for Computational Linguistics: EACL 2024, 2024, pp. 374–392.

[16] Maciej P. Polak and Dane Morgan, “Extracting accurate materials data from research papers with conversational language models and prompt engineering,” Nature Communications, vol. 15, 2024, Art. no. 1569.

[17] Yizhu Jiao, Ming Zhong, Sha Li, Ruining Zhao, Siru Ouyang, Heng Ji, and Jiawei Han, “Instruct and extract: Instruction tuning for on-demand information extraction,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 10030–10051.

[18] Oscar Sainz, Iker Garc´ıa-Ferrero, Rodrigo Agerri, Oier Lopez de Lacalle, German Rigau, and Eneko Agirre, “GoLLIE: Annotation guidelines improve zero-shot information-extraction,” in International Conference on Learning Representations, 2024.

[19] ChenZhuo Zhao, Xinda Wang, Pu Zhao, Yue Huang, Junting Lu, Ziqian Liu, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang, “Gradient-guided multi-judge prompt optimization,” in Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics, 2026, pp. 23744–23773.

[20] Beining Wang, Weihang Su, Hongtao Tian, Hao Kong, Tao Yang, Ting Yao, Qingyi Pan, Yueyue Wu, Qingyao Ai, Min Zhang, and Yiqun Liu, “Co-evolving LLM evaluators and policies via DynamicRubric,” arXiv preprint arXiv:2607.20083, 2026.

[21] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica, “Judging LLM-as-a-judge with MT-bench and chatbot arena,” in Advances in Neural Information Processing Systems, 2023, vol. 36, pp. 46595–46623.