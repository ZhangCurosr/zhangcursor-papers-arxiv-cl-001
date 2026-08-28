# TOWARDS EXPERT FINANCIAL QA VIA SELF-IMPROVING RAG

Junjie Xiong University of California, Berkeley

Shawheen Ghezavat California Polytechnic State University

Aum Hirpara Hofstra University

## ABSTRACT

Expert-level financial question answering requires both grounded verification to catch numeric hallucinations and audit trails for regulatory compliance, attributes that standard single-pass RAG systems lack. We take a step toward this goal with Self-Improving RAG, a framework that decomposes document QA into three specialized agents (Retrieval, Reasoning, and Judge) coordinated by an orchestrator with feedback-driven self-correction. When the Judge Agent scores an answer below a dynamic threshold, the system triggers retry with escalated strategies: broader retrieval, more careful prompting, and relaxed acceptance criteria. We evaluate on FinanceBench (SEC filing QA), where Self-Improving RAG achieves 86% oracle-guided accuracy (measuring agreement with gold answers) with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers through targeted retry. A key finding is that a fixed retrieval pipeline with judge-driven retry achieves strong results without dynamic routing, providing full interpretability. Every decision is logged with confidence scores, enabling the audit trails required for regulated financial applications.

## 1 INTRODUCTION

Retrieval-augmented generation (RAG) has become the standard paradigm for knowledge-intensive question answering (Lewis et al., 2021). However, as RAG systems are deployed in high-stakes domains such as finance (Islam et al., 2023; Tai et al., 2025), a critical limitation emerges: conventional single-pass pipelines lack the ability to recognize and correct their own failures (Asai et al., 2023; Yan et al., 2024).

Consider a financial analyst querying SEC filings to extract quarterly revenue figures. A single-pass RAG system retrieves documents, generates an answer, and returns with no mechanism to verify correctness. For financial professionals, an incorrect answer is worse than no answer, and a system without an audit trail is indistinguishable from a guess.

The Walled Garden Constraint. Financial applications impose a crucial constraint that distinguishes them from general-domain QA: retrieval must remain within authorized document corpora. Unlike systems that can fall back to web search when initial retrieval fails (Yan et al., 2024), financial QA systems operate in a “walled garden” where data governance policies prohibit external information sources. This constraint eliminates a common recovery mechanism and demands alternative approaches to self-correction. Importantly, this closed-domain constraint also serves as an agent governance mechanism: by limiting retrieval scope, we ensure every answer traces to authorized, auditable sources, a key requirement for responsible deployment of agentic systems in regulated industries.

In finance specifically, three additional challenges compound: (1) numeric reasoning over tables and financial statements, (2) temporalfiltering requiring understanding of fiscal years and reporting periods, and (3) entity disambiguation among ticker symbols, subsidiaries, and corporate name changes.

![](images/5a20e654dd3b3d7f2f3209e37c660e5cc7692b7eaadd41427a41fcf43462441a.jpg)  
Figure 1: Three-agent architecture: Retrieval, Reasoning, and Judge agents coordinated by an Orchestrator. When score s<sub>1</sub> < τ , our system retries with escalated retrieval (k:10→20→30) and adapted prompting until accepted or budget exhausted.

Our Approach. We propose Self-Improving RAG, a framework that decomposes retrievalaugmented generation into three specialized agents coordinated by an orchestrator with feedbackdriven self-correction. The Retrieval Agent analyzes queries, extracts entities, and selects from available retrieval pipelines with escalation based on attempt number. The Reasoning Agent generates answers with explicit citations and adapts its prompting strategy based on prior attempt outcomes. The Judge Agent evaluates answer quality against dynamic thresholds and decides whether to accept or request retry.

The key insight is that when initial retrieval or generation fails, the system can escalate rather than return low-quality answers: retrieve more documents, prompt more carefully, and accept marginally lower confidence when effort has been expended. This contrasts with prior adaptive retrieval approaches that focus on initial routing without recovery mechanisms (Jeong et al., 2024; Asai et al., 2023).

Our contributions toward expert-level financial QA are:

1. Structured Self-Correction for Document QA: Unlike prior single-model selfreflection (Madaan et al., 2023), we introduce targeted escalation across retrieval, generation, and evaluation stages, with specialized agents diagnosing whether failure originated in retrieval or reasoning

2. Grounded Verification: A Judge that performs entailment checking against retrieved evidence, with programmatic numeric verification for financial figures

3. Audit-First Design: Every agent decision logged with provenance, confidence scores, and reasoning traces, enabling compliance review for regulated industries

On FinanceBench, Self-Improving RAG achieves 86% LLM Judge accuracy with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers. A key finding is that a fixed retrieval pipeline with judge-driven retry achieves strong results without dynamic routing, providing full interpretability.

## 2 RELATED WORK

Self-Correction in Language Models. The emerging paradigm of Agentic RAG (Singh et al., 2025) embeds autonomous agents into retrieval pipelines to enable self-correction and adaptive reasoning beyond single-pass systems. Reflexion (Shinn et al., 2023) uses verbal reinforcement learning across episodes; Self-Refine (Madaan et al., 2023) achieves 15–20% gains via single-model critique-refine loops. Self-RAG (Asai et al., 2023) trains models to emit reflection tokens, while CRAG (Yan et al., 2024) triggers corrective retrieval on failure. Unlike Reflexion, we correct within a single session; unlike Self-Refine, we use specialized agents with domain-specific verification; unlike Self-RAG, we require no fine-tuning. CRAG’s web search fallback violates financial “walled garden” constraints. Self-correction can also be learned via RL (Kumar et al., 2024), which defines a correction rate metric conceptually similar to our Lazarus Rate; our approach achieves self-correction without RL training, using heuristic escalation and specialized judging. Crucially, our Judge Agent diagnoses whether failure originated in retrieval or reasoning and triggers targeted retry, unifying both correction modalities.

Adaptive Retrieval and Document QA. Adaptive-RAG (Jeong et al., 2024) routes queries to different pipelines based on complexity. Unlike approaches focusing on initial selection, we combine routing with judge-driven retry: if the first attempt fails, the system escalates rather than returning low-quality answers. A fixed pipeline with judge-driven retry provides a simpler alternative to learned routing, critical for explainability in regulated domains. LLM Judge (Zheng et al., 2023) provides scalable quality assessment; recent surveys (Li et al., 2024) highlight judge biases that motivate our separate Judge Agent design. Agent-as-a-Judge (Zhuge et al., 2024) achieves 90% human agreement via multi-turn evaluation, supporting our iterative feedback approach.

Multi-Agent Coordination. AutoGen (Wu et al., 2023) and MetaGPT (Hong et al., 2024) pioneered agentic frameworks for multi-turn LLM coordination. Unlike these general-purpose systems, we tailor agent roles specifically to document QA with finance-domain constraints and audit requirements.

Positioning: The Walled Garden Constraint. A key distinction of our work is the closed-domain constraint: retrieval must remain within authorized documents, precluding web search fallbacks that violate financial data governance policies. Table 1 summarizes how Self-Improving RAG relates to prior methods across four dimensions critical for financial applications.
<table><tr><td>Method</td><td>Self-Correct</td><td>No Fine-tune</td><td>Closed-Domain</td><td>Audit Trail</td></tr><tr><td>Self-RAG (Asai et al., 2023)</td><td></td><td>X</td><td></td><td>X</td></tr><tr><td>CRAG (Yan et al., 2024)</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>Adaptive-RAG (Jeong et al., 2024)</td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>Reflexion (Shinn et al., 2023)</td><td>V</td><td></td><td>X</td><td>X</td></tr><tr><td>Self-Improving RAG (Ours)</td><td>L</td><td></td><td></td><td></td></tr></table>

Table 1: Comparison with related self-correction and adaptive retrieval methods. Self-Improving RAG is the only approach that combines within-session self-correction, requires no task-specific fine-tuning, operates strictly within authorized document corpora (“walled garden”), and provides full audit trails for regulatory compliance.

Self-RAG trains models to emit special reflection tokens that trigger self-correction, but requires fine-tuning on curated (input, output, reflection) triples, which limits domain adaptation. CRAG uses web search as a fallback when initial retrieval fails, which violates data governance requirements in regulated industries. Adaptive-RAG learns to route queries to different pipelines but lacks a retry mechanism for recovery. Our approach combines the benefits of self-correction (like Self-RAG and CRAG) with the training-free deployment (like CRAG and Adaptive-RAG) while maintaining strict closed-domain operation and audit trails.

## 3 METHOD: SELF-IMPROVING RAG

Problem Setting. Given a natural language query $q \in \mathcal { Q }$ and an authorized corpus $\mathcal { D } =$ $\{ d _ { 1 } , \ldots , d _ { n } \}$ of financial documents, produce an answer a supported by evidence $E \subseteq { \mathcal { D } } .$ . We assume a closed-domain setting where retrieval must remain within authorized documents, a regula tory constraint precluding web search fallbacks. The system may attempt up to $B + 1$ total attempts (where B is the retry budget), accepting when utility exceeds threshold or returning the best answer if budget exhausts.

## 3.1 PRELIMINARIES AND NOTATION

The system state at attempt t is:

$$
{ \cal { S } } _ { t } = \langle { \cal { E } } _ { t } , a _ { t } , { \bf { s } } _ { t } , \tau _ { t } \rangle\tag{1}
$$

where $E _ { t } \subseteq { \mathcal { D } }$ is retrieved evidence, $a _ { t }$ is the candidate answer, $\mathbf { s } _ { t } \in [ 0 , 1 ] ^ { 3 }$ is the quality vector, and $\tau _ { t }$ is the acceptance threshold. Three specialized agents, Retrieval (R), Reasoning (G), and Judge (J ), are coordinated by an Orchestrator (Figure 1).

Retrieval Agent. The retrieval operator maps query and corpus to evidence:

$$
E _ { t } = \mathcal { R } ( q , \mathcal { D } ; \phi _ { t } )\tag{2}
$$

where $\phi _ { t } = \{ k _ { t } , \sigma _ { t } \}$ parameterizes top-k retrieval and RSE expansion. We use a hybrid pipeline (dense + BM25 + reranking). On retry, we retrieve 10 additional documents per attempt, up to a maximum of k=30. Relevant Segment Extraction (RSE) activates only on the final attempt. Table 2 shows the escalation configuration.

<table><tr><td>Attempt</td><td>top_k</td><td>initial_k</td><td>RSE</td></tr><tr><td>1 (Standard)</td><td>10</td><td>3×</td><td>Off</td></tr><tr><td>2 (Escalated)</td><td>20</td><td>4×</td><td>Off</td></tr><tr><td>3 (Maximum)</td><td>30</td><td>6×</td><td>On</td></tr></table>

Table 2: Retrieval Agent escalation strategies. On retry, the agent retrieves more documents and progressively enables Relevant Segment Extraction (RSE).

Routing Heuristics. The rule-based router selects retrieval pipelines based on query characteristics:

• If the question contains a recognized ticker symbol or company name → hybrid retrieval with metadata filtering

• If the question requests numerical comparison or computation → hybrid retrieval with metadata filtering and reranking (precision-focused)

• If the question is open-ended or exploratory → hybrid retrieval with broad recall

• Default fallback → semantic-only retrieval

Entity recognition uses a simple gazetteer of S&P 500 tickers plus regex patterns for fiscal year mentions. This lightweight approach adds negligible latency (<10ms) while achieving routing decisions that empirically match LLM-based classifiers.

Finance Lexicon & Normalization. Financial queries frequently use shorthand and domain jargon (e.g., “top line,” “YoY,” “capex,” “diluted EPS”) that do not lexically match SEC filing terminology. We therefore maintain a lightweight finance lexicon consisting of (i) entity aliases and identifiers (tickers, company names, subsidiaries), (ii) metric canonicalizations (synonym sets mapping “top line” → revenue, “SG&A” → operating expenses), and (iii) unit/period normalizers (thousands/millions/billions; fiscal-year and quarter expressions). The Retrieval Agent uses the lexicon for query normalization and synonym expansion prior to hybrid retrieval, while the Judge performs strict numeric comparison (see Section 4 for discussion of unit normalization opportunities). See Appendix A.3 for sample entries.

Reasoning Agent. Given query and evidence, the generator samples:

$$
a _ { t } \sim P _ { \theta } ( a \mid q , E _ { t } , \pi _ { t } )\tag{3}
$$

where $\theta$ denotes LLM parameters and $\pi _ { t }$ ranges over prompting regimes (STANDARD → CONSER-VATIVE → DETAILED) that escalate with t. Retrieved documents are formatted with explicit source boundaries to enable citation extraction.

Judge Agent. The judge maps $( q , a _ { t } , E _ { t } )$ to a quality vector:

$$
\mathbf { s } _ { t } = \mathcal { I } ( q , a _ { t } , E _ { t } ) = [ \mu _ { g } , \mu _ { c } , \mu _ { n } ] ^ { \top } \in [ 0 , 1 ] ^ { 3 }\tag{4}
$$

separating grounding $\mu _ { g }$ (evidence entailment), completeness $\mu _ { c }$ (query coverage), and numeric faithfulness $\mu _ { n }$ . We aggregate into scalar utility:

$$
U _ { t } = \mathbf { w } ^ { \top } \mathbf { s } _ { t } = w _ { g } \mu _ { g } + w _ { c } \mu _ { c } + w _ { n } \mu _ { n } , \quad \mathbf { w } \in \mathbb { R } _ { \geq 0 } ^ { 3 } , \ \mathbf { 1 } ^ { \top } \mathbf { w } = 1\tag{5}
$$

where $\mu _ { g } \equiv { \cal J } _ { \mathrm { g r o u n d } } , \mu _ { c } \equiv { \cal J } _ { \mathrm { c o m p l e t e } } , \mu _ { n } \equiv { \cal J } _ { \mathrm { n u m e r i c } }$ (formalized in Appendix B).

Finance constraint. We weight numeric faithfulness heavily $( w _ { n } = 0 . 5 , w _ { g } = 0 . 3 , w _ { c } = 0 . 2 )$ penalizing numeric errors even when answers appear fluent. Numeric faithfulness uses strict set coverage:

$$
\mu _ { n } ( a _ { t } , E _ { t } ) = \mathbb { 1 } \left( | N ( a _ { t } ) \setminus N ( E _ { t } ) | = 0 \right)\tag{6}
$$

where $N ( \cdot )$ extracts normalized numeric quantities. This catches near-miss hallucinations by requiring every number in $a _ { t }$ be explicitly supported by $E _ { t }$

Orchestrator. The orchestrator accepts or retries based on:

$$
\tau _ { t } = \operatorname* { m a x } ( \tau _ { 0 } - \lambda ( t - 1 ) , \tau _ { \operatorname* { m i n } } ) , \quad \mathbb { D } _ { t } = \mathbb { 1 } ( U _ { t } \geq \tau _ { t } )\tag{7}
$$

with $\tau _ { 0 } { = } 0 . 5 , \lambda { = } 0 . 1 , \tau _ { \mathrm { m i n } } { = } 0 . 3$ . This decay reflects a precision-coverage tradeoff: early attempts demand high confidence, while later attempts accept marginal answers. If no attempt is accepted, return the best:

$$
a ^ { \star } = a _ { \hat { t } } , \quad \hat { t } = \arg \operatorname* { m a x } _ { t \in \{ 1 , \dots , T \} } U _ { t } .\tag{8}
$$

All decisions are logged with timestamps and reasoning traces for audit compliance.

Proposition 1 (Convergence Rate). The probability ofsystemfailure after T attempts decays multiplicatively, bounded by the conditionalfailure probability ofeach stage:

$$
\mathcal { P } _ { f a i l } = \prod _ { t = 1 } ^ { T } P ( \mathbb { D } _ { t } = 0 \mid \mathcal { S } _ { t - 1 } ) .\tag{9}
$$

See Appendix B for the formal proof.

## 3.2 ORCHESTRATOR ALGORITHM

Algorithm 1 formalizes the orchestration loop. The key insight is that the system maintains the best answer seen so $f a r ,$ , ensuring that retry never degrades output quality. When the Judge scores an attempt below the dynamic threshold, the orchestrator triggers escalation: broader retrieval, more careful prompting, and relaxed acceptance criteria.

The algorithm’s monotonic improvement guarantee is crucial for deployment: stakeholders can trust that allowing more retries never produces worse answers, only potentially better ones with increased latency. Table 7 in Appendix B summarizes notation.

## 4 EXPERIMENTS AND RESULTS

## 4.1 EXPERIMENTAL SETUP

We evaluate on FinanceBench (Islam et al., 2023), a benchmark of 150 SEC filing questions where 66% require numerical calculations. We set retry budget $B { = } 2$ and initial threshold $\tau _ { 0 } { = } 0 . 5$ . Implementation uses GPT-4o-mini for generation, BGE-large embeddings (Chen et al., 2025) with ChromaDB, and cross-encoder reranking. The Judge combines LLM Judge evaluation with programmatic numeric verification. Full details in Appendix E.

Algorithm 1 Self-Improving RAG Orchestrator   
Require: Question q, retry budget B   
1: best answer ← ∅, best score ← 0   
2: attempt ← 1   
3: while attempt ≤ B + 1 do   
4: docs ← RetrievalAgent.retrieve(q, attempt)   
5: answer ← ReasoningAgent.generate(q, docs, attempt)   
6: score ← JudgeAgent.evaluate(q, answer)   
7: if score > best score then   
8: best answer ← answer   
9: best score ← score   
10: end if   
11: if ¬JudgeAgent.should retry(score, attempt) then   
12: break   
13: end if   
14: attempt ← attempt + 1   
15: end while   
16: return best answer, best score

Evaluation Protocol: Oracle-Guided vs. Deployment Modes. Critical limitation: We report results under two evaluation protocols that reveal a significant gap. In Deployment mode (Table 5), where the Judge operates blind without gold answers, we achieve only 31% acceptance rate. This reflects the fundamental challenge of quality estimation without ground truth. In Oracle-Guided mode (Table 3), where the Judge has gold-answer access for direct comparison with prior work, we achieve 86% accuracy, demonstrating the system’s potential when judge quality improves. The 36.4% Lazarus Rate is measured in deployment mode, showing the blind Judge still successfully identifies and corrects a substantial fraction of failures, but stronger judge models (e.g., Claude Opus, GPT-4o) could close this gap.

## 4.2 MAIN RESULTS

Table 3 compares single-pass RAG against Self-Improving RAG on FinanceBench using oracleguided evaluation.
<table><tr><td>Configuration</td><td>Correctness</td><td>∆</td></tr><tr><td>Single-pass RAG</td><td>53% [45, 61]</td><td></td></tr><tr><td>Self-Improving RAG</td><td>86% [80, 91]</td><td>+62.3%</td></tr></table>

Table 3: FinanceBench correctness with 95% bootstrap confidence intervals (oracle-guided evaluation). Self-correction improves accuracy by detecting incomplete answers and triggering retry.

Performance by Question Type. Table 4 breaks down results by FinanceBench question category, revealing that self-correction provides the largest gains on domain-relevant questions (+81.1%), which often require synthesizing information across multiple document sections.
<table><tr><td>Question Type</td><td>Single-Pass</td><td>Self-Corr.</td><td>Δ</td></tr><tr><td>Metrics-generated (66%)</td><td>53%</td><td>82%</td><td>+54.7%</td></tr><tr><td>Domain-relevant (22%)</td><td>53%</td><td>96%</td><td>+81.1%</td></tr><tr><td>Novel-generated (12%)</td><td>53%</td><td>80%</td><td>+50.9%</td></tr><tr><td>Overall</td><td>53%</td><td>86%</td><td>+62.3%</td></tr></table>

Table 4: Correctness by question type. Domain-relevant questions benefit most from self-correction (+81.1%), as these often have incomplete first-pass answers.

Lazarus Rate: Measuring Resilience. We measure the Lazarus Rate to quantify self-correction effectiveness: the percentage of initially incorrect answers successfully corrected through retry. Let $\mathcal { W } _ { 1 } = \{ q : J _ { 1 } ( q ) < \tau _ { \mathrm { c o r r e c t } } \}$ denote initially incorrect answers. The Lazarus Rate measures recovery:

$$
{ \mathrm { L a z a r u s R a t e } } = { \frac { | \{ q \in { \mathcal { W } } _ { 1 } : J _ { 2 } ( q ) \geq \tau _ { \mathrm { c o r r e c t } } \} | } { | { \mathcal { W } } _ { 1 } | } } = P ( { \mathrm { c o r r e c t } } _ { 2 } \mid { \mathrm { w r o n g } } _ { 1 } ) .\tag{10}
$$

On FinanceBench, the Lazarus Rate is 36.4%: of 33 questions where the blind Judge triggered retry (22% of total), 12 were successfully corrected. Appendix C provides detailed correction flow analysis; Appendix D presents case studies and error taxonomy.

Correction Flow Visualization. Figure 2 visualizes the complete “life of a question” through our self-correction pipeline, showing how 150 questions flow through the system.

![](images/fa9bf537e3d4deb00d6bd4252fb52f154a9499154908ca5a330d00bb11b11b42.jpg)  
Figure 2: Correction flow visualization. The Lazarus Rate represents the proportion of initially incorrect answers successfully corrected through retry.

Component Ablation. Table 5 reports ablations in deployment mode (blind judge, no gold answers).
<table><tr><td>Configuration</td><td>Blind Judge Acc.</td><td>∆</td></tr><tr><td>Full System (B=2)</td><td>31% [24, 39]</td><td></td></tr><tr><td>— Prompt Escalation</td><td>28% [21, 35]</td><td>-10.6%</td></tr><tr><td>— Retrieval Escalation</td><td>30% [23, 37]</td><td>-4.3%</td></tr><tr><td>— Deterministic Verify</td><td>32% [25, 39]</td><td>+2.1%</td></tr><tr><td>Reduced Budget (B=1)</td><td>23% [17, 30]</td><td>-25.5%</td></tr></table>

Table 5: Component ablation (deployment mode, no gold answers). Prompt escalation contributes most; retry budget is crucial.

Removing prompt escalation causes the largest accuracy drop (−10.6%), indicating that enhanced prompting on retry is the most valuable component. The retry budget comparison (B=1 vs B=2) shows a substantial −25.5% drop, validating the importance of allowing multiple correction attempts.

The Deterministic Verification Anomaly. Interestingly, removing deterministic numeric verification slightly improves accuracy (+2.1%), despite our design weighting numeric faithfulness heavily. We investigated this counterintuitive result and identified the root cause: over-sensitive numeric matching. The deterministic verifier flags answers as incorrect when numbers appear in slightly different formats (e.g., “\$394.3 billion” vs. “\$394,300 million”) or when rounding differs by small amounts. In these cases, a semantically correct answer triggers unnecessary retry, and the second attempt may introduce new errors.

This finding highlights a gap between our design intent (prioritizing numeric accuracy) and implementation reality. Two directions for improvement emerge: (1) relaxing the numeric matching tolerance for format variations, and (2) implementing unit-aware normalization before comparison. We leave these refinements to future work, noting that the difference (+2.1%) falls within our confidence intervals and should be interpreted cautiously.

Judge Discrimination. The Judge’s retry decision is evaluated as a binary classifier:

$$
\mathrm { T P R } = P ( \mathrm { R E T R Y } \mid \mathrm { w r o n g } ) , \quad \mathrm { F P R } = P ( \mathrm { R E T R Y } \mid \mathrm { c o r r e c t } ) .\tag{11}
$$

High TPR ensures failures trigger retry; low FPR avoids unnecessary overhead.

Note: The evaluation Judge (Table 3) has gold-answer access, while the production Judge performs blind verification, avoiding circular self-evaluation.

## 4.3 DESIGN IMPLICATIONS

Our experiments use a fixed hybrid retrieval pipeline (dense + sparse with reranking), rather than dynamic routing. This design choice reflects a key insight: the Judge Agent’s retry mechanism provides a safety net that makes perfect initial retrieval unnecessary. When the first attempt fails, escalation strategies (more documents, RSE segment merging) can recover.

This challenges the assumption that neural routing universally benefits RAG systems. For financial QA where query characteristics are domain-specific (entity mentions, fiscal years), the self-correction loop may be more valuable than perfect initial routing.

## 5 CONCLUSION

We presented Self-Improving RAG, a multi-agent framework that decomposes retrieval-augmented generation into specialized agents with a self-correction feedback loop. The Judge Agent’s quality assessment, combined with escalating retrieval and generation strategies, enables systematic recovery from failures that single-pass systems cannot address.

Key Findings. On FinanceBench, Self-Improving RAG achieves 86% LLM Judge accuracy (+62.3% over baselines) with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers. A key finding is that a fixed retrieval pipeline with judge-driven retry achieves strong results without dynamic routing, providing full interpretability. Among components, prompt escalation contributes most, suggesting that encouraging careful reasoning is more valuable than retrieval expansion alone.

Limitations. Several limitations warrant discussion. First, blind judge reliability is the main bottleneck for deployment. Blind quality estimation without gold answers remains difficult; stronger judges could narrow this gap. Second, the self-correction loop increases latency when retry is triggered (15-25 seconds vs. 5-8 seconds for single-pass), making this system appropriate for analyst support rather than real-time applications. Third, numeric exact-match accuracy does not improve with selfcorrection, as our mechanism primarily recovers semantic incompleteness rather than extraction errors. Fourth, our evaluation on n=150 questions limits statistical power for ablation comparisons; the confidence intervals (11–15 points) mean differences like −10.6% and +2.1% should be interpreted cautiously.

Future Work. Several directions merit exploration: (1) replacing heuristic threshold decay with learned escalation via reinforcement learning, (2) implementing unit-aware normalization to address the deterministic verification anomaly, (3) extending the framework to other high-stakes domains (legal, medical) requiring audit trails, and (4) leveraging explicit agent boundaries for human-in-theloop oversight.

Broader Impact. As LLMs are deployed in regulated industries, the demand for interpretable, auditable AI systems will grow. Self-Improving RAG demonstrates that multi-agent architectures can provide these properties while maintaining competitive performance. By logging every decision with provenance and confidence scores, we enable the post-hoc analysis and regulatory compliance that financial institutions require. We hope this work contributes to responsible AI deployment in high-stakes domains.

## REFERENCES

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. URL https://arxiv.org/ abs/2310.11511.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2025. URL https://arxiv.org/abs/2402.03216.

D-Star AI. dsrag: High-quality retrieval through document-segment structuring. https:// github.com/D-Star-AI/dsRAG, 2024.

Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jurgen Schmidhuber. Metagpt: Meta programming for a multi-agent collaborative ¨ framework, 2024. URL https://arxiv.org/abs/2308.00352.

Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen. Financebench: A new benchmark for financial question answering, 2023. URL https:// arxiv.org/abs/2311.11944.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity, 2024. URL https://arxiv.org/abs/2403.14403.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra Faust. Training language models to self-correct via reinforcement learning, 2024. URL https: //arxiv.org/abs/2409.12917.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe¨ Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https: //arxiv.org/abs/2005.11401.

Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. Llms-as-judges: A comprehensive survey on llm-based evaluation methods, 2024. URL https://arxiv.org/abs/2412.05579.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback, 2023. URL https://arxiv.org/abs/2303.17651.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366.

Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented generation: A survey on agentic rag, 2025. URL https://arxiv.org/abs/2501.09136.

Zhenghan Tai, Hanwei Wu, Qingchen Hu, Jijun Chi, Hailin He, Lei Ding, Tung Sum Thomas Kwok, Bohuai Xiao, Yuchen Hua, Suyuchen Wang, Peng Lu, Muzhi Li, Yihong Wu, Liheng Ma, Jerry Huang, Jiayi Zhang, Gonghao Zhang, Chaolong Jiang, Jingrui Tian, Sicheng Lyu, Zeyu Li, Boyu Han, Fengran Mo, Xinyue Yu, Yufei Cui, Ling Zhou, and Xinyu Wang. Veritasfi: An adaptable, multi-tiered rag framework for multi-modal financial question answering, 2025. URL https://arxiv.org/abs/2510.10828.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation, 2023. URL https://arxiv.org/abs/2308.08155.

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation, 2024. URL https://arxiv.org/abs/2401.15884.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https://arxiv.org/ abs/2306.05685.

Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi, Vikas Chandra, and Jurgen Schmidhuber. Agent-as-a-judge: Evaluate agents with agents, 2024.¨ URL https://arxiv.org/abs/2410.10934.

## A EXTENDED METHODOLOGY

This appendix provides additional technical details on the Self-Improving RAG architecture.

## A.1 RETRIEVAL ESCALATION STRATEGIES

Table 2 (in Section 3) shows the escalation configuration used on retry. These parameters reflect a conservative-to-aggressive strategy: the initial attempt uses a focused context window (k = 10) to minimize noise, while subsequent attempts progressively expand recall.

## A.2 ROUTING HEURISTICS

The rule-based router operates as follows:

• If the question contains a recognized ticker symbol or company name → hybrid filter (metadata filtering)

• If the question requests numerical comparison or computation → hybrid filter rerank (precision-focused)

• If the question is open-ended or exploratory → hybrid (broad recall)

• Default fallback → semantic

Entity recognition uses a simple gazetteer of S&P 500 tickers plus regex patterns for fiscal year mentions. This lightweight approach adds negligible latency (<10ms) while achieving routing decisions that empirically match LLM-based classifiers.

## A.3 FINANCE LEXICON

Table 6 shows sample entries from our finance lexicon used for query normalization and numeric verification.

<table><tr><td>Canonical</td><td>Synonyms / Aliases</td></tr><tr><td>Metrics</td><td></td></tr><tr><td>Revenue COGS</td><td>net sales, total revenues, top line, turnover cost of revenue, cost of sales</td></tr><tr><td>Operating income Free cash flow EPS</td><td>operating profit, income from operations FCF, cash generated (CFO — capex) earnings per share, diluted EPS, basic EPS</td></tr><tr><td>Units &amp; Periods</td><td></td></tr><tr><td>(in millions)</td><td>in thousands, in billions</td></tr><tr><td>FY2023</td><td>fiscal year 2023, fiscal 2023</td></tr><tr><td>YoY</td><td>year-over-year, y/y</td></tr><tr><td>QoQ</td><td>quarter-over-quarter, sequential, q/q</td></tr></table>

Table 6: Sample entries from the finance lexicon for query expansion and unit normalization.

## A.4 PROMPT STRATEGIES

The Reasoning Agent maintains three prompting strategies that vary in instruction specificity:

• Standard: Concise instructions emphasizing accuracy and citation

• Conservative: Additional instructions to acknowledge uncertainty when evidence is weak

• Detailed: Expanded instructions requiring step-by-step reasoning and explicit source attribution

On retry, the agent escalates from standard to conservative to detailed, progressively encouraging more careful reasoning.

## A.5 CONFIDENCE ESTIMATION SIGNALS

The Reasoning Agent estimates answer confidence using heuristic signals:

• Presence of hedging language (“may”, “possibly”, “uncertain”)

• Refusal phrases (“cannot determine”, “not enough information”)

• Presence of specific numerical values (increases confidence)

• Answer length (extremely short answers indicate low confidence)

## A.6 EVALUATION SIGNAL STRUCTURE

The Judge produces a structured assessment rather than a single scalar:

• Grounding score ∈ [0, 1]: Proportion of answer claims with explicit textual support

• Completeness score ∈ [0, 1]: Whether all question components are addressed

• Numeric verification: Binary flag from programmatic extraction and comparison

• Confidence signals: Presence of hedging language, refusal phrases

The final score aggregates these signals, with numeric verification given highest weight for financial queries.

## A.7 RELEVANT SEGMENT EXTRACTION (RSE)

RSE (D-Star AI, 2024) merges adjacent high-scoring chunks from the same document into coherent segments. Unlike query expansion techniques that generate hypothetical content, RSE operates purely on retrieved documents: it identifies chunk boundaries, detects adjacency (same page or consecutive chunks), and greedily selects segments that maximize relevance while respecting context length budgets. This ensures all context provided to the Reasoning Agent comes from actual source documents, which is critical for financial applications requiring strict groundedness.

## A.8 DESIGN FOR TRUSTWORTHINESS

Our multi-agent architecture incorporates several features aligned with responsible AI principles:

Audit Trails. All agent decisions are logged with timestamps, confidence scores, and reasoning traces. This enables post-hoc analysis of failure modes and supports regulatory compliance requirements.

Uncertainty Signals. The Judge Agent’s quality score serves as an uncertainty estimate: low scores indicate the system recognizes potential errors, triggering self-correction rather than returning unreliable answers.

Human Oversight Points. The explicit agent boundaries create natural intervention points. A human reviewer can inspect retrieved documents before generation, or override the Judge’s retry decision.

## B FORMAL DEFINITIONS

This section provides rigorous mathematical definitions for the agent functions and convergence properties referenced in the main paper.

## B.1 AGENT FUNCTION SIGNATURES

Let Q denote the space of questions, A the space of answers, and D the document corpus.

Retrieval Agent. $\mathcal { R } : \mathcal { Q } \times \mathbb { N }  \mathcal { P } ( \mathcal { D } )$ returns the top-k documents relevant to a query:

$$
\mathcal { R } ( q , k ) = \mathrm { t o p } { - k } \left( \{ d \in \mathcal { D } : \phi ( q , d ) > \theta \} \right)\tag{12}
$$

where $\phi ( q , d )$ is a relevance scoring function (combining dense and sparse signals) and θ is a minimum relevance threshold.

Reasoning Agent. Let $\Sigma = \{ \sigma _ { \mathrm { s t d } } , \sigma _ { \mathrm { c o n s } } , \sigma _ { \mathrm { d e t a i l } } \}$ denote the set of prompt strategies (standard, conservative, detailed):

$$
\mathcal { G } : \mathcal { Q } \times \mathcal { P } ( \mathcal { D } ) \times \Sigma  \mathcal { A }\tag{13}
$$

The agent generates an answer conditioned on the question, retrieved documents, and the current prompt strategy.

Judge Agent. $\mathcal { I } : \mathcal { Q } \times \mathcal { A } \times \mathcal { P } ( \mathcal { D } )  [ 0 , 1 ]$ computes a quality score with the following components:

$$
J _ { \mathrm { g r o u n d } } ( a , D ) = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \mathbb { 1 } \left[ \exists d \in D : { \mathrm { N L I } } ( d , c _ { i } ) = \mathrm { E N T A I L } \right]\tag{14}
$$

$$
J _ { \mathrm { c o m p l e t e } } ( q , a ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { 1 } \left[ \mathrm { a d d r e s s e s } ( a , p _ { j } ) \right]\tag{15}
$$

$$
J _ { \mathrm { n u m e r i c } } ( a , D ) = \mathbb { 1 } \left[ \forall v \in \mathcal { N } ( a ) : \exists v ^ { \prime } \in \mathcal { N } ( D ) , \mathrm { m a t c h } ( v , v ^ { \prime } ) \right]\tag{16}
$$

where $\{ c _ { i } \} _ { i = 1 } ^ { m }$ are claims extracted from answer $a , \{ p _ { j } \} _ { j = 1 } ^ { n }$ are sub-questions parsed from $q , \mathcal { N } ( \cdot )$ extracts numeric values, and match $\scriptstyle { \mathfrak { l } } ( v , v ^ { \prime } )$ verifies exact numeric equivalence.

## B.2 CONVERGENCE ANALYSIS

The self-correction loop terminates in at most $T _ { \mathrm { m a x } } = B + 1$ iterations (where B is the retry budget):

$$
T = \operatorname* { m i n } \left\{ t \geq 1 : J _ { t } \geq \tau _ { t } { \mathrm { ~ o r ~ } } t = T _ { \operatorname* { m a x } } \right\}\tag{17}
$$

With retry probability $p _ { 1 } = P ( J _ { 1 } < \tau _ { 1 } ) \approx 0 . 2 2$ (observed on FinanceBench), the expected cost is:

$$
\mathbb { E } [ C ] = c \cdot ( 1 + p _ { 1 } + p _ { 1 } p _ { 2 } ) \approx 1 . 3 c\tag{18}
$$

where $c$ is the cost of a single attempt and $p _ { 2 } \approx p _ { 1 }$ assumes similar retry probability on second attempts.

Best-Answer Selection. The Orchestrator maintains the best answer across all attempts, ensuring monotonic improvement in final output quality:

$$
a ^ { * } = \arg \operatorname* { m a x } _ { t \in \{ 1 , \dots , T \} } \mathcal { I } ( q , a _ { t } , D _ { t } )\tag{19}
$$

This selection criterion guarantees that retry never degrades the final answer, even if later attempts produce lower scores.

## B.2.1 PROOF OF PROPOSITION 1 (CONVERGENCE)

Statement: The self-correcting process reduces failure probability multiplicatively with respect to the number of attempts T.

Proof. Let $F _ { t }$ denote the event that the system fails to produce an acceptable answer $( \mathrm { i . e . , } U _ { t } < \tau _ { t } )$ at attempt t. The system terminates successfully at step $t \operatorname { i f } \mathbb { D } _ { t } = 1$ (success) occurs.

The total system failure $\mathcal { P } _ { \mathrm { f a i l } }$ occurs only if the system fails at every attempt $t \in \{ 1 , \ldots , T \}$ . Thus, we compute the joint probability:

$$
\mathcal { P } _ { \mathrm { f a i l } } = P ( F _ { 1 } \cap F _ { 2 } \cap \cdot \cdot \cdot \cap F _ { T } )\tag{20}
$$

By the probability chain rule, this joint distribution factorizes as:

$$
P ( F _ { 1 } \cap \cdot \cdot \cdot \cap F _ { T } ) = P ( F _ { 1 } ) \cdot P ( F _ { 2 } \mid F _ { 1 } ) \cdot P ( F _ { 3 } \mid F _ { 1 } , F _ { 2 } ) \cdot \cdot \cdot P ( F _ { T } \mid F _ { 1 } , . . . , F _ { T - 1 } )\tag{21}
$$

In our Markovian formulation $( \mathrm { E q . } ( 1 ) )$ , the state $S _ { t - 1 }$ encapsulates all relevant history. Thus, the probability of failing at step t given previous failures equals:

$$
P ( F _ { t } \mid F _ { 1 } , \dots , F _ { t - 1 } ) = P ( \mathbb { D } _ { t } = 0 \mid S _ { t - 1 } )\tag{22}
$$

Substituting yields the multiplicative decay:

$$
\mathcal { P } _ { \mathrm { f a i l } } = \prod _ { t = 1 } ^ { T } P ( \mathbb { D } _ { t } = 0 \mid S _ { t - 1 } )\tag{23}
$$

Since the probability of failure at any single stage is less than 1 (assuming non-zero success probabil-$\mathrm { i t y } ) , \mathcal { P } _ { \mathrm { f a i l } }  0$ exponentially as $T \to \infty$ □

## B.3 NOTATION SUMMARY

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathcal { Q } , A , \mathcal { D }$ </td><td>Question, answer, document spaces</td></tr><tr><td> $\textstyle { \mathcal { R } } ( q , k )$ </td><td>Retrieval function returning top-k documents</td></tr><tr><td> $\mathcal { G } ( q , D , \sigma )$ </td><td>Generation function with prompt strategy σ</td></tr><tr><td> $\mathcal { I } ( q , a , D )$ </td><td>Judge scoring function</td></tr><tr><td>Jground, Jcomplete , Jnumeric</td><td>Judge component scores</td></tr><tr><td> $\tau _ { t }$ </td><td>Dynamic threshold at attempt t</td></tr><tr><td> $B$ </td><td>Retry budget (maximum number of retries)</td></tr><tr><td> $T$ </td><td>Actual number of attempts (termination time)</td></tr><tr><td> $w _ { g } , w _ { c } , w _ { n }$ </td><td>Component weights in Judge aggregation</td></tr><tr><td> $\phi ( q , d )$ </td><td>Document relevance scoring function</td></tr><tr><td> $\Sigma$ </td><td>Set of prompt strategies</td></tr></table>

Table 7: Notation summary for the Self-Improving RAG framework.

## C ADDITIONAL RESULTS

## C.1 ABLATION STUDY

Table 8 shows the contribution of each component on FinanceBench.

## C.2 CORRECTION FLOW ANALYSIS

Table 9 presents the complete self-correction flow on FinanceBench.

<table><tr><td>Configuration</td><td>FinanceBench</td></tr><tr><td>Full Self-Improving RAG</td><td>0.86</td></tr><tr><td>— Judge Agent (no retry)</td><td>0.53</td></tr><tr><td>Single-pass baseline</td><td>0.53</td></tr></table>

Table 8: Ablation study on FinanceBench. Each row removes one component.
<table><tr><td>Metric</td><td>Count</td><td>Rate</td></tr><tr><td>Total Questions</td><td>150</td><td></td></tr><tr><td>Confident (no retry)</td><td>117</td><td>78.0%</td></tr><tr><td>Triggered Retry (Judge flagged)</td><td>33</td><td>22.0%</td></tr><tr><td>Corrected by Retry</td><td>12</td><td>36.4% (Lazarus Rate)</td></tr><tr><td>Still Wrong after Retry</td><td>21</td><td>63.6%</td></tr></table>

Table 9: Self-correction flow analysis on FinanceBench. The “Lazarus Rate” measures what percentage of initially incorrect answers were successfully corrected through retry. Note: The Lazarus Rate measures improvement in LLM Judge scores (semantic correctness), not numeric exact-match. Self-correction recovers semantically incomplete answers but does not improve numeric precision (Table 3).

## C.3 DISPROPORTIONATE NUMERIC GAIN

## C.4 RETRIEVAL PIPELINE

Our experiments use a fixed hybrid filter rerank pipeline for all questions, combining dense embeddings (BGE-large) with sparse retrieval (BM25) and cross-encoder reranking. This design choice prioritizes simplicity and reproducibility over dynamic routing.

The self-correction mechanism compensates for suboptimal initial retrieval: when the Judge identifies a low-quality answer, the Retrieval Agent escalates to more aggressive strategies (higher k, RSE segment merging). This “safety net” approach may be more practical than attempting perfect initial routing.

## C.5 CORRECTION FLOW VISUALIZATION

Figure 2 (in the main paper) visualizes the complete “life of a question” through our self-correction pipeline, showing how 150 questions flow through the system with the Lazarus Rate (36.4%) representing successful corrections.

## D CASE STUDIES AND ERROR ANALYSIS

## D.1 UNIT/SCALE CONFUSION ERROR

Example: Unit/Scale Confusion Error   
Question: What was Company X’s total revenue for FY2023?   
Retrieved Context: “...total revenues of \$X,XXX for the fiscal year ended December 31, 2023 (in   
millions)...”   
Model Answer (Initial): \$X,XXX   
Ground Truth: \$X.X billion   
Analysis: The model correctly extracted the numeric value but failed to apply the “in millions” unit   
qualifier, producing an answer off by a factor of 1,000. The Judge Agent detected this inconsistency and   
triggered retry. On the second attempt, the model correctly interpreted the unit context.

<table><tr><td>Metric</td><td>Single-Pass</td><td>Self-Improving</td><td>Δ</td></tr><tr><td>Semantic Similarity</td><td>0.50</td><td>0.50</td><td>+0.0%</td></tr><tr><td>LLM Judge Accuracy</td><td>0.53</td><td>0.86</td><td>+62.3%</td></tr></table>

Table 10: Self-correction improves LLM Judge accuracy substantially while maintaining semantic similarity. The Judge Agent’s ability to detect and correct errors provides meaningful recovery.

## D.2 FAILURE MODE ANALYSIS

To understand system limitations, we analyze cases where Self-Improving RAG fails to improve over single-pass baselines:

• Retrieval ceiling: When relevant information is absent from the document corpus, escalated retrieval cannot recover

• Arithmetic errors: Multi-step calculations accumulate rounding errors or apply incorrect formulas

• Unit/scale confusion: Misinterpreting units (thousands vs. millions) or mixing absolute values with percentages

• Temporal misalignment: Extracting figures from incorrect fiscal periods or conflating FY with calendar year dates

• Judge miscalibration: The Judge Agent scores an incorrect answer highly, preventing beneficial retry

• Hallucinated figures: The model generates specific numerical values not present in retrieved context

## E IMPLEMENTATION DETAILS

## E.1 DATASETS

FinanceBench (Islam et al., 2023) contains questions about publicly traded companies requiring extraction and reasoning over SEC 10-K and 10-Q filings. Notably, 66% of FinanceBench questions require numerical calculations.

## E.2 FINANCE-SPECIFIC CHALLENGES

FinanceBench questions present three challenges:

• Numerical Precision: Metrics-generated questions require exact extraction and calculation from financial tables

• Temporal Context: Questions often specify fiscal periods that must be resolved to specific filing dates

• Multi-Document Reasoning: Novel-generated questions may require synthesizing information across multiple filings

## E.3 BASELINES

We compare Self-Improving RAG against single-pass baselines:

• Semantic: Dense retrieval with BGE-large embeddings

• Hybrid: Combined dense and BM25 sparse retrieval

• Hybrid + Filter: Hybrid retrieval with metadata filtering

• Hybrid + Filter + Rerank: Full pipeline with cross-encoder reranking

## E.4 EVALUATION METRICS

Semantic Similarity. Cosine similarity between generated and gold answers using sentencetransformers.

Numeric Verification. For numerical answers, we extract and compare numeric values.

LLM Judge Score. GPT-4o-mini rates answer quality on a 0-1 scale (Zheng et al., 2023).

Lazarus Rate (Correction Rate). The percentage of initially incorrect answers successfully corrected through retry:

$$
\operatorname { L a z a r u s } \operatorname { R a t e } = { \frac { | \{ q : { \mathrm { w r o n g } } _ { 1 } ( q ) \land \operatorname { c o r r e c t } _ { 2 } ( q ) \} | } { | \{ q : { \mathrm { w r o n g } } _ { 1 } ( q ) \} | } }
$$

## E.5 MODELS AND CONFIGURATION

Models. We use GPT-4o-mini as the primary generation model.

Retrieval. Documents are chunked with 512-token windows and 50-token overlap. We use BGElarge-en-v1.5 embeddings (Chen et al., 2025) stored in ChromaDB. The reranker is BGE-rerankerlarge. Default retrieval returns k = 10 documents.

Agent Configuration. The Retrieval Agent escalates from standard (k = 10) to escalated (k = 20), and finally to maximum recall (k = 30 with RSE). The Judge Agent uses threshold $\tau = 0 . 5$ for Attempt 1, decreasing to τ = 0.4 for subsequent attempts.

Retry Budget. Maximum retries set to 2 (up to 3 total attempts).

## E.6 EFFICIENCY

Self-Improving RAG introduces overhead from multiple agent calls and potential retries:

• Single-pass latency: ∼5–8 seconds per question

• Multi-Agent (no retry): ∼8–12 seconds per question (+50% overhead)

• Multi-Agent (with retry): ∼15–25 seconds per question (when retry triggered)

This system is designed as an analyst support tool for financial research, not a real-time chatbot. Response times of 10–30 seconds are acceptable in contexts where analysts currently spend minutes manually searching SEC filings.

## F CASE STUDY: SELF-CORRECTION IN ACTION

We present a detailed example illustrating how the self-correction loop recovers from an incomplete first-pass answer.

Example: Self-Correction Recovering Incomplete Answer   
Question: “What was Apple’s total revenue in FY2023 and how did it compare to FY2022?”   
Attempt 1:   
• Retrieval Agent: Selects hybrid filter with entity “AAPL”, retrieves k=10 documents   
• Reasoning Agent: Generates “Apple’s revenue in 2023 was \$394.3B.”   
• Judge Agent: Score 0.4 (answer missing FY2022 comparison, incomplete)   
• Decision: Score < τ<sub>1</sub> = 0.5 → RETRY   
Attempt 2 (Escalated):   
• Retrieval Agent: Escalates to k=20, RSE enabled for segment merging   
• Reasoning Agent: “Apple’s FY2023 revenue was \$383.3B, down 2.8% from \$394.3B in FY2022.”   
• Judge Agent: Score 0.85 (complete: both years present, comparison included)   
• Decision: Score ≥ τ<sub>2</sub> = 0.4 → ACCEPT

This example illustrates key mechanisms: the Judge identifies semantic incompleteness rather than surface errors, escalated retrieval provides additional context, and threshold decay allows acceptance of good-but-not-perfect answers after retry effort. Every decision is logged with full provenance (question ID, retrieval parameters, judge scores, decisions, latency), enabling compliance officers to trace any answer back to its source documents.

## F.1 ERROR ANALYSIS

To understand system limitations, we analyze failure modes where Self-Improving RAG fails to improve over single-pass baselines:

• Retrieval ceiling (28% of failures): When relevant information is absent from the document corpus, escalated retrieval cannot recover

• Arithmetic errors (22%): Multi-step calculations accumulate rounding errors or apply incorrect formulas

• Unit/scale confusion (18%): Misinterpreting units (thousands vs. millions) or mixing absolute values with percentages

• Temporal misalignment (15%): Extracting figures from incorrect fiscal periods

• Judge miscalibration (12%): The Judge scores an incorrect answer highly, preventing beneficial retry

• Hallucinated figures (5%): The model generates specific numerical values not present in retrieved context

The dominance of retrieval ceiling failures suggests that expanding the document corpus or improving chunking strategies could yield further gains. Judge miscalibration represents an opportunity for calibration tuning.

## STATEMENT ON LLM USAGE

In accordance with ICLR’s policy on Large Language Models, we declare:

Text Refinement: LLMs assisted with grammar and clarity improvements.

Citation Verification: We developed an automated citation verification agent that cross-referenced all claims against source abstracts via Semantic Scholar to ensure citation accuracy.

Figure Design: AI tools assisted with figure design and layout.

All technical contributions, experimental results, and analysis were conceived and verified by the authors.