# Legal-R1: Agentic Retrieval for Evidence-Based Legal Reasoning

Huaiwen Pang<sup>1,\*</sup>, Jiale Han<sup>2,\*,†</sup>, Junying Chen<sup>1</sup>, Wenjun Xiao<sup>3</sup>, Benyou Wang<sup>1,2,†</sup>

<sup>1</sup>The Chinese University of Hong Kong, Shenzhen,

<sup>2</sup>Shenzhen Loop Area Institute, <sup>3</sup>Shenzhen Shenten Digital Technology Co., Ltd. huaiwenpang@link.cuhk.edu.cn, jialehan@slai.edu.cn, junying.chen.cs@gmail.com, xiaowenjun@fadaifu.net, wangbenyou@cuhk.edu.cn

## Abstract

Legal question answering requires answers that are not only plausible, but also grounded in legally valid and citable evidence. Standard retrieval-augmented generation often relies on one-shot semantic matching, which is insufficient when legal evidence must be selected according to authority, applicability, and diversity across heterogeneous sources. We introduce Legal-R1, an agentic retrieval framework for evidence-based legal reasoning. Legal-R1 decomposes legal reasoning into a multi-agent workflow, where a Lawyer agent learns when to retrieve, which legal source to query, what evi dence to seek, and when the collected evidence is sufficient. A Legal Assistant agent retrieves from normative legal sources and precedent cases with authority-aware and diversity-aware selection, while a Judge agent generates the final answer from the complete evidence trajectory. We post-train only the Lawyer agent with masked supervised fine-tuning and trajectoryaware reinforcement learning. Experiments on multiple Chinese legal benchmarks show that Legal-R1 improves over general-purpose, legalspecific, and retrieval-augmented baselines.<sup>1</sup>

## 1 Introduction

Legal question answering requires more than fluent generation. A trustworthy legal answer must be grounded in legal evidence that is valid, applicable, and citable (Wiratunga et al., 2024; Louis et al., 2024; Xiao et al., 2026). This evidence is often distributed across heterogeneous sources, including laws, judicial interpretations, and precedent cases. It must also satisfy domain-specific constraints such as authority hierarchy, temporal validity, jurisdictional fit, and applicability to the target facts (Van Opijnen and Santos, 2017). These requirements make legal question answering a problem of evidence construction rather than plain information lookup.

![](images/9282c56cfbe52b5a13e3f73f40a0e6d837f185985d8e59efdee3cd779117b335.jpg)  
Figure 1: Motivation of Legal-R1. Legal-R1 replaces one-turn semantic matching with multi-turn evidence seeking over norms and cases.

Existing large language models still struggle with this setting. Parametric legal knowledge can be incomplete, outdated, or unsupported by explicit citations (Dahl et al., 2024; Van Opijnen and Santos, 2017). Retrieval-augmented generation improves grounding by retrieving external documents (Lewis et al., 2020), but many conventional pipelines rely on a fixed set of retrieved passages without adaptively deciding whether further retrieval is needed (Asai et al., 2024). This is not sufficient for legal reasoning. A semantically similar case may still fail to match the relevant legal issue (Li et al., 2024b). Other retrieved cases may come from lower-authority courts or duplicate the same sentencing outcome. Legal retrieval therefore cannot be reduced to finding similar text. It must identify legally valid evidence that can justify the answer. Figure 1 illustrates this distinction. General agentic retrieval asks what information is needed to answer a question. Legal agentic retrieval asks what legal norms and precedents are needed to support a verifiable answer. The agent must decide whether retrieval is needed, which legal source to query, what evidence to retrieve, and whether the collected evidence is sufficient.

We propose Legal-R1, an agentic retrieval framework for evidence-based legal reasoning. Legal-R1 decomposes legal reasoning into three roles. The Lawyer agent plans multi-turn evidence seeking. It identifies missing legal evidence, chooses a retrieval action, formulates sourcespecific queries, and decides when to stop. The Legal Assistant agent executes retrieval over heterogeneous legal knowledge bases. It searches legal norms, including laws and judicial interpretations, and retrieves precedent cases with authority-aware and diversity-aware selection. The Judge agent generates the final answer from the legal question and the complete retrieval trajectory. This workflow makes evidence collection explicit and encourages answers to be grounded in comprehensive, authoritative, and diverse legal evidence.

We train only the Lawyer agent. This design focuses learning on the core difficulty of legal agentic retrieval, namely evidence-seeking decisions. It also decouples retrieval planning from answer generation, so the trained Lawyer agent can be paired with different Judge agents in downstream applications. To train the Lawyer agent, we synthesize multi-turn retrieval trajectories from a structured legal QA dataset. We first use masked supervised fine-tuning to teach the retrieval protocol while preventing the model from memorizing retrieved evidence. We then apply trajectory-aware reinforcement learning to improve source selection, query construction, and stopping behavior.

Experiments on Chinese legal benchmarks validate the effectiveness of Legal-R1. Compared with existing RAG-based legal systems, Legal-R1 achieves stronger performance on LexEval and LawBench, showing that actively planning retrieval is more effective than attaching a fixed retriever to an LLM. Ablation and retrieval-quality analyses further show that post-training improves evidenceseeking behavior and that authority-aware and diversity-aware precedent reranking improves the reliability of retrieved evidence. These results support our central claim that legal reasoning benefits from learning an explicit evidence-seeking policy.

Our contributions are summarized as follows:

• We introduce a multi-agent workflow for evidence-based legal reasoning, where the Lawyer, Legal Assistant, and Judge agents separately handle evidence planning, legal retrieval, and answer generation. This workflow supports comprehensive, authoritative, and diverse evidence grounding.

• We propose Legal-R1, an agentic legal retrieval framework that post-trains the Lawyer agent to learn an autonomous evidenceseeking policy, including whether to retrieve, which source to query, what evidence to retrieve, and when to stop.

• We conduct experiments and ablations on multiple Chinese legal benchmarks, demonstrating that Legal-R1 improves legal reasoning over retrieval-augmented baselines and that its gains come from both learned retrieval policies and legal-evidence-aware reranking.

## 2 Related Work

Retrieval-Augmented Generation. Retrieval-Augmented Generation (RAG) mitigates factual hallucinations by dynamically fetching external knowledge, evolving from basic embedding lookups to hybrid frameworks that merge lexical and semantic search (Lewis et al., 2020; Karpukhin et al., 2020). To handle massive document pools, recent paradigms employ large language models as re-rankers to prioritize relevant passages (Sun et al., 2023; Nogueira et al., 2020). However, conventional RAG systems treat documents as independent text blocks, relying on surface-level semantic overlaps while completely ignoring domainspecific structured meta-attributes like judicial hierarchies or sentencing variations (Gao et al., 2023). In contrast, we propose a specialized judicial reranking mechanism integrating an authority hierarchy score and a punishment severity index (PSI). By incorporating these domain priors into a modified maximal marginal relevance formulation (Carbonell and Goldstein, 1998), our approach simultaneously optimizes both statutory authority and retrieval diversity.

Tool-Augmented Agents. Tool-augmented agents empower language models to autonomously execute external APIs, reason through complex environments, and solve multi-step tasks (Yao et al., 2022; Schick et al., 2023). Recent breakthroughs leverage post-training reinforcement learning—such as Group Relative Policy Optimization (GRPO)—to scale test-time compute and cultivate long-term thinking behaviors for autonomous error correction (Shao et al., 2024; Guo et al., 2025). Despite their success, existing frameworks heavily rely on deterministic, mathematically verifiable environments where outcomes are definitively checked by code interpreters or symbolic solvers (Cobbe et al., 2021). When applied to non-deterministic, knowledge-dense fields like law, they suffer from severe supervision bottlenecks due to the absence of absolute verification functions (Hendrycks et al., 2021). Our approach bridges this gap through a hierarchical multi-agent workflow embedded in a continuous deliberation loop. We adapt GRPO to this non-deterministic tool-use paradigm via a multi-tier dense reward function anchored by a fine-tuned critique model (Ouyang et al., 2022).

![](images/fd1c6ec65fcc2959cc3e1bc42e4931a0c4cce42d5e7342a1d2dfd90ed7f7e9de.jpg)  
Figure 2: Overview of the proposed agentic legal retrieval workflow.

LLMs and Agents in the Legal Domain. The deployment of large language models in computational law has transformed legal natural language processing (Xiao et al., 2018). Existing efforts either prompt general models with legal instructions or fine-tune them on domain-specific corpora (Fei et al., 2024). Recently, multi-agent legal pipelines have emerged to simulate role-differentiated behaviors, alongside synthetic data pipelines to compile large-scale training trajectories (Yue et al., 2023; Wang et al., 2023). Nevertheless, standard legal models function within rigid, single-turn setups that struggle with fluid reasoning, while automated data generation remains highly susceptible to training-test contamination and option overlaps (Xu et al., 2024). Our work addresses these gaps by introducing an end-to-end automated data synthesis and purification pipeline. This framework implements strict algorithmic safeguards, including fingerprint-based contamination filtering and heuristic negative sampling for distractors, ensuring data purity and legal logical uniqueness.

## 3 Method

## 3.1 Problem Formulation

Given a legal question q, the task is to generate an evidence-grounded answer a based on a heterogeneous legal knowledge base K. In our setting, K consists of legal norms and judicial precedents.

$$
\mathcal { K } = \mathcal { K } _ { n o r m } \cup \mathcal { K } _ { c a s e } ,
$$

where $\boldsymbol { \kappa } _ { n o r m }$ contains laws, regulations, and judicial interpretations, and ${ \kappa } _ { c a s e }$ contains criminal judgment cases.

Legal evidence is not judged by relevance alone. A high-quality answer should be grounded in evidence that is comprehensive, authoritative, and diverse. It should cover the key legal issues in q, rely on valid and authoritative legal sources, and include analogous precedents that reflect meaningful variation in fact patterns and judicial outcomes. The central challenge is therefore to decide what evidence should be retrieved, from which legal source, and when the collected evidence is sufficient for answering.

## 3.2 Agentic Legal Retrieval

As shown in Figure 2, Legal-R1 addresses evidence-based legal reasoning through three rolespecialized components: a Lawyer Agent, a Legal

Assistant Agent, and a Judge Agent. The Lawyer Agent is the central planner. It analyzes the user question, identifies missing legal evidence, selects a source-specific retrieval action, formulates or revises the query, and decides whether the accumulated evidence is sufficient to stop. The Legal Assistant Agent executes these requests over heterogeneous legal knowledge bases: it retrieves and reranks statutes and judicial interpretations. The Judge Agent receives the legal question and the complete evidence trajectory, and synthesizes the final evidence-grounded answer from this collected evidence. This role separation makes the learnable object explicit: Legal-R1 trains the Lawyer Agent to improve the evidence-seeking policy, while the Legal Assistant and Judge Agents provide retrieval and answer synthesis services.

## 3.2.1 Lawyer Agent

Given q, the Lawyer agent identifies the legal issues, determines what type of evidence is missing, and constructs a multi-turn retrieval trajectory.

$$
\tau = ( r _ { 1 } , u _ { 1 } , s _ { 1 } , e _ { 1 } , \dots , r _ { T } , u _ { T } , s _ { T } , e _ { T } )
$$

where $r _ { j }$ is the reasoning state, $u _ { j }$ is the retrieval action, $s _ { j }$ is the search query, and $e _ { j }$ is the retrieved evidence at turn $j$ . The action space is defined as

$$
u _ { j } \in \{ \mathrm { S T A T U T E } , \mathrm { P R E C E D E N T } , \mathrm { S T O P } \}
$$

Here, STATUTE requests legal provisions or judicial interpretations, PRECEDENT requests analogous criminal cases, and STOP indicates that the current evidence is sufficient for answering. The Lawyer agent therefore decides whether the next step should clarify the legal basis, compare precedents, refine a failed query, or terminate retrieval. After each returned evidence unit, it performs a rethink step to assess whether the evidence resolves the current uncertainty and whether the trajectory should continue with the same source, switch to another source, or stop.

## 3.2.2 Legal Assistant Agent

The Legal Assistant agent executes the Lawyer agent’s retrieval requests over K and returns compact evidence units.

Intent-Aware Statute Retrieval. For STATUTE, the Lawyer agent provides a search query and a short retrieval intent or context, such as locating a sentencing range, identifying the elements of an offense, or finding an exception clause. The Legal

Assistant agent first retrieves candidate provisions from ${ { \cal { K } } _ { n o r m } }$ using the query. It then reranks the candidates with an LLM-based normative reranker $\mathcal { L } _ { n o r m }$ conditioned on the retrieval intent, so that the returned provisions match the specific legal role requested by the Lawyer agent.

Authority-Aware Diverse Precedent Retrieval. For PRECEDENT, the Legal Assistant agent retrieves a candidate set $C$ from $\displaystyle { \mathcal { K } } _ { c a s e }$ using the target fact description and optional circumstances. Since semantic similarity alone may return lowauthority or outcome-redundant cases, we adapt Maximal Marginal Relevance (MMR) (Carbonell and Goldstein, 1998) to select precedents with both court authority and sentencing-outcome diversity. For each candidate $d _ { i } \in C$ , the base score is

$$
\mathbf { B a s e } ( d _ { i } ) = { \boldsymbol { \alpha } } \cdot \mathbf { R e l } ( d _ { i } ) + ( 1 - { \boldsymbol { \alpha } } ) \cdot \mathbf { A u t h } ( d _ { i } )
$$

where $\mathrm { R e l } ( d _ { i } )$ is the retrieval relevance score, $\operatorname { A u t h } ( d _ { i } )$ is the normalized court-authority score, and α controls their trade-off. Given selected cases S, the final MMR-style score is

$$
M ( d _ { i } \mid S ) = \operatorname { \mathrm { \bf B a s e } } ( d _ { i } ) - \beta \cdot \operatorname * { m a x } _ { d _ { j } \in S } \operatorname { S i m } _ { \operatorname { \bf P S I } } ( d _ { i } , d _ { j } )
$$

where $\beta$ controls the diversity penalty and $\begin{array} { r } { \mathrm { S i m } _ { \mathrm { P S I } } ( d _ { i } , d _ { j } ) = 1 - \frac { | \mathrm { P S I } _ { i } - \mathrm { P S I } _ { j } | } { d _ { \mathrm { m a x } } + \epsilon } } \end{array}$ measures similarity in sentencing severity. Here, $d _ { \mathrm { m a x } }$ normalizes PSI distance and ϵ ensures numerical stability. Section 4.1 details the authority annotation and PSI construction. After selection, an LLM-based case summarizer $\mathcal { L } _ { c a s e }$ condenses each precedent into a compact evidence unit with key facts, court level, judgment outcome, and sentencing information.

## 3.2.3 Judge Agent

The Judge agent directly generates the final evidence-grounded answer. We denote the answergeneration LLM as $\mathcal { L } _ { J u d g e }$ . Given the legal question $q$ and the evidence collected along the retrieval trajectory τ, the Judge agent produces

$$
a = \mathcal { L } _ { J u d g e } ( q , \tau )
$$

The prompt instructs $\mathcal { L } _ { J u d g e }$ to answer based on the retrieved legal norms and precedents, and the full prompt is provided in the appendix A.

## 3.3 Lawyer Agent Training

We train the Lawyer agent to learn an active evidence-seeking policy for legal retrieval. It learns when to retrieve, what evidence to seek, how to revise source-specific queries, and when to stop. Training only the Lawyer agent decouples evidence collection from answer generation, allowing the learned policy to be paired with different Judge agents in downstream applications.

Trajectory Synthesis. We synthesize tool-use trajectories by simulating the inference-time interaction between the Lawyer agent and the Legal Assistant agent. Given a legal question $q ,$ a teacher model $\mathcal { L } _ { T e a c h e r }$ plays the Lawyer agent and generates a reasoning state $r _ { j }$ , a retrieval action $u _ { j } .$ and a source-specific query $s _ { j }$ . When $u _ { j } \neq \mathbf { S } \mathrm { T O P }$ the Legal Assistant agent uses $s _ { j }$ to invoke the corresponding retrieval module and returns evidence $e _ { j }$ . The teacher conditions on the updated context to decide whether to refine the query, switch the evidence source, or stop retrieval. The process terminates when $\mathcal { L } _ { T e a c h e r }$ emits STOP and produces the final answer a. Each training instance records the complete evidence-seeking process.

$$
\begin{array} { r } { D = \{ ( q ^ { ( i ) } , r _ { 1 } ^ { ( i ) } , u _ { 1 } ^ { ( i ) } , s _ { 1 } ^ { ( i ) } , e _ { 1 } ^ { ( i ) } , \ldots , } \\ { r _ { T _ { i } } ^ { ( i ) } , u _ { T _ { i } } ^ { ( i ) } , s _ { T _ { i } } ^ { ( i ) } , e _ { T _ { i } } ^ { ( i ) } , a ^ { ( i ) } ) \} _ { i = 1 } ^ { N } } \end{array}
$$

where $D$ is the synthesized trajectory dataset, $N$ is the number of training instances, and $T _ { i }$ is the number of retrieval turns in instance i.

Evidence-Masked Supervised Fine-Tuning. We use supervised fine-tuning to teach the Lawyer agent the protocol of agentic legal retrieval. Each sample in $D$ is serialized into a token sequence $x = ( x _ { 1 } , \ldots , x _ { L } ) $ . We optimize the causal language modeling loss with an evidence mask:

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t = 1 } ^ { L } m _ { t } \cdot \log p _ { \theta } ( x _ { t } \mid x _ { < t } ) ,
$$

where $m _ { t } = 0$ for tokens belonging to retrieved evidence $e _ { j }$ and $m _ { t } = 1$ otherwise. This masking prevents the model from learning to reproduce retrieved statutes or cases, and instead focuses training on reasoning states, retrieval actions, and query revision.

Trajectory-Aware Reinforcement Learning. Starting from the SFT model, we use Group Relative Policy Optimization (GRPO) (Shao et al., 2024) to further train the Lawyer agent. For each question $q ,$ the current model samples a retrieval trajectory $\hat { \tau }$ and a final answer ${ \hat { a } } .$ The reward compares them with the synthesized trajectory $\tau$ and reference answer a in D. It encourages the Lawyer agent to collect useful evidence, use valid queries and formats, and avoid unnecessary retrieval turns. The total reward is

![](images/ee9c800ae3c274d5f32f1d602b84bde47627b16ee0adb7a92d49222b99f0a9d9.jpg)  
Figure 3: Distribution of the training data across major legal task categories and fine-grained subtasks. Detailed statistics are provided in Appendix G.

$$
\begin{array} { r l r } & { } & { R ( \hat { \tau } , \hat { a } \mid q , \tau , a ) = w _ { 1 } R _ { \mathrm { a c c } } ( \hat { a } , a ) + w _ { 2 } R _ { \mathrm { a l i g n } } ( \hat { \tau } , \tau ) } \\ & { } & { + w _ { 3 } R _ { \mathrm { g a i n } } ( \hat { e } , a ) + w _ { 4 } R _ { \mathrm { q u e r y } } ( \hat { s } ) } \\ & { } & { + R _ { \mathrm { f m t } } ( \hat { \tau } , \hat { a } ) - \rho ( \hat { \tau } , \hat { a } ) - \eta \mathrm { K L } } \end{array}
$$

where $\hat { e }$ and sˆ denote the evidence and search queries in $\hat { \tau } , R _ { \mathrm { a c c } }$ scores the final answer, $R _ { \mathrm { a l i g n } }$ compares the sampled trajectory with the synthesized trajectory, $R _ { \mathrm { g a i n } }$ checks whether the retrieved evidence supports the answer, $R _ { \mathrm { q u e r y } }$ evaluates query quality, $R _ { \mathrm { f m t } }$ checks the tool-use format, and $\rho$ penalizes verbosity and repeated retrieval loops. The weights $w _ { 1 } , \ldots , w _ { 4 }$ balance these terms. KL is the GRPO regularization term and η controls its weight.

## 4 Data Construction

The dataset utilized in this project consists of two components: a legal knowledge base K and training data for legal agent D.

## 4.1 Legal Knowledge Base

Normative Legal Corpus. We construct ${ { \cal { K } } _ { n o r m } }$ from the National Database of Laws and Regulations<sup>2</sup>. Each statutory article or judicial interpretation is treated as an individual retrieval unit and is annotated with its law title, chapter, section, article number, and effective date. The processed corpus contains 296 active laws with 43,117 articles and 555 judicial interpretations with 6,370 articles.

Criminal Precedent Corpus. We construct $\displaystyle { \mathcal { K } } _ { c a s e }$ from $\mathrm { L e C a R D v } 2 ^ { 3 }$ , which contains 53,277 criminal judgment records. Each case is retained as a precedent-level retrieval unit and is associated with structured attributes including the court level, case facts, judgment outcome, and sentencing result. This corpus supports precedent retrieval for analogous facts and sentencing comparison.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Retrieval Scale</td><td rowspan="2"></td><td colspan="2">Lawbench</td><td colspan="2">LexEval</td><td rowspan="2">Avg</td></tr><tr><td>Trad. Score</td><td>LLM Judge</td><td>Trad. Score</td><td>LLM Judge</td></tr><tr><td>LawGPT</td><td>x</td><td>7B</td><td>11.33</td><td>13.76</td><td>9.49</td><td>9.18</td><td>10.94</td></tr><tr><td>Luwen</td><td>x</td><td>7B</td><td>30.06</td><td>32.77</td><td>14.03</td><td>16.97</td><td>23.45</td></tr><tr><td>LegalOne</td><td>x</td><td>8B</td><td>23.25</td><td>41.02</td><td>35.30</td><td>43.79</td><td>35.84</td></tr><tr><td>Qwen3-8B</td><td>x</td><td>8B</td><td>24.70</td><td>40.20</td><td>18.04</td><td>28.47</td><td>27.85</td></tr><tr><td>A2search</td><td>√</td><td>7B</td><td>46.82</td><td>49.14</td><td>35.23</td><td>37.94</td><td>42.28</td></tr><tr><td>R-Search</td><td>√</td><td>7B</td><td>43.01</td><td>47.44</td><td>45.05</td><td>46.85</td><td>45.59</td></tr><tr><td>Qwen3-8B-Embedding</td><td>√</td><td>8B</td><td>32.92</td><td>43.25</td><td>35.79</td><td>41.40</td><td>38.34</td></tr><tr><td>Llama-Embed-Nemotron</td><td>√</td><td>8B</td><td>32.23</td><td>42.28</td><td>36.45</td><td>41.77</td><td>38.19</td></tr><tr><td>DeepSeek-V4-Flash-Preview</td><td>x</td><td>284B</td><td>62.40</td><td>68.77</td><td>61.81</td><td>72.90</td><td>66.47</td></tr><tr><td>GPT-5.4-Mini</td><td>x</td><td></td><td>54.83</td><td>62.83</td><td>50.50</td><td>59.32</td><td>56.87</td></tr><tr><td>Legal-R1</td><td>√</td><td>8B</td><td>57.36</td><td>63.97</td><td>56.54</td><td>63.28</td><td>60.28</td></tr></table>

Table 1: Model Performance Comparison on LexEval,and Lawbench Benchmarks. Judge by DeepSeek-V4-Flash Preview.

Authority and Sentencing Annotations. To support authority-aware and diversity-aware precedent retrieval, we add two annotations to ${ \cal { K } } _ { c a s e } .$ . First, cases are grouped by the level of the adjudicating court, including high people’s courts, intermediate people’s courts, and primary people’s courts or other local tribunals, and each group is mapped to an authority score. Second, the Punishment Severity Index (PSI) maps heterogeneous sentencing outcomes into a numerical severity value. Main penalties are converted into day-equivalent values and adjusted by penalty type and suspension factors, while supplementary penalties are added as extra severity terms:

$$
\mathrm { P S I } = D \cdot \gamma \cdot \phi + \sum S _ { \mathrm { s u p } }
$$

where D is the day-equivalent main penalty, $\gamma$ is the penalty-type coefficient, ϕ is the suspension factor, and $S _ { \mathrm { s u p } }$ denotes supplementary penalties such as fines or deprivation of political rights. Detailed authority scores and PSI examples are provided in Appendix F.

## 4.2 Training Data Construction

We construct the post-training dataset through an automated data synthesis and purification pipeline. The data are collected from established legal benchmarks and authoritative examination resources, including CAIL (Xiao et al., 2018), DISC-LawLLM (Yue et al., 2023), CrimeKgAssitant<sup>4</sup> , and the National Unified Legal Professional Qualification Examination<sup>5</sup>, and are further complemented with domain-specific synthetic problem sets. We then converts heterogeneous legal materials into taskspecific QA pairs while applying multiple qualitycontrol mechanisms. Specifically, the pipeline first performs task-driven applicability gating to filter source texts that are unsuitable for a given task, then removes potentially contaminated samples through N-gram fingerprint matching against heldout benchmarks, and finally applies automatic leakage detection and rule-based distractor construction to improve the reliability of open-ended and multiple-choice questions. The resulting dataset contains 18,940 high-quality structured QA instances, covering five major legal domains and 29 fine-grained subcategories. Figure 3 shows the distribution of the training data across these categories. The detailed distribution and construction details are provided in Appendix G.

## 5 Experiments

## 5.1 Setup

Benchmark. We evaluate our method on LexEval (Li et al., 2024a) and LawBench (Fei et al., 2024), two widely used Chinese legal benchmarks. LexEval consists of 23 tasks and 14,150 questions covering six legal cognitive abilities, while Law-Bench includes 20 tasks across three judicial cognitive dimensions, ranging from multiple-choice questions to practical applications such as reading comprehension, criminal amount calculation, and legal consultation.

Evaluation Metrics. We evaluate the proposed method from two perspectives: performance and efficiency. Performance is measured by benchmark default scores and an LLM-as-a-judge score, while efficiency is assessed by retrieval turns, execution time, tool-invocation time, and token consumption. Implementation Details. We train Legal-R1 from

![](images/d5e0809379814b27211e36d81d36f4b4e3aa0bfe12b15b886c6eac7eb5e75864.jpg)  
Figure 4: Model Performance across Five Categories of LexEval.

Qwen3-8B using SFT followed by GRPO. The SFT stage runs for 3 epochs on $D _ { \mathrm { S F T } }$ with a learning rate of $5 \times 1 0 ^ { - 6 }$ and a batch size of 32. The GRPO stage uses a learning rate of 1 $. \times 1 0 ^ { - 6 }$ , a batch size of $^ { 3 2 }$ 8 rollouts, and $\eta = 0 . 0 8$ . Training is conducted on four NVIDIA H20 GPUs. At inference time, Legal-R1 serves as the lawyer model that performs multiturn reasoning and retrieval. A hybrid retriever and an auxiliary Qwen3-8B assistant model refine the retrieved evidence, while another Qwen3-8B judge model generates the final evidence-grounded answer. Full deployment details are provided in Appendix B.

Baselines. We compare our model against three categories of baseline LLMs: (1) General LLMs, represented by DeepSeek-V4-Flash-Preview (DeepSeek-AI, 2026),and GPT-5.4-Mini (OpenAI, 2026); (2) Legal-Specific LLMs, including LawGPT (Zhou et al., 2024), LegalOne (Li et al., 2026), and Luwen (Wu et al., 2026); and (3) RAG-based LLMs, which include A2search(Zhang et al., 2025) and R-Search (Zhao et al., 2025).

During evaluation, we configure each baseline LLM with its native prompt template (if available). For RAG-enabled baselines, we integrate their respective RAG components and load the same retrieval knowledge base used by our agentic retrieval framework. We then execute a unified evaluation pipeline: the primary models and the judge model (Qwen3-8B) are deployed via vLLM to generate responses across all benchmarks and compute the final scores. Concurrently, we track key inference metrics, including the number of retrieval turns, execution time (latency), and token consumption.

## 5.2 Results and Discussion

Main Results We evaluate various baselines on legal tasks across Lawbench and LexEval, as shown in Table 1. The empirical results indicate that while traditional legal-specific models and

![](images/8040a86a91a6e3d995bfeab9f7484500e28ef55e32b5a41e7712acf3baede77b.jpg)  
Figure 5: Average Retrieval Turn for Different Complexity Tasks on Lawbench and Lexeval Benchmark.

RAG-augmented methods show competitive performance, they are consistently outperformed by our approach. Our method, Legal-R1, performs exceptionally well across all datasets and evaluation metrics, achieving a top-tier overall average score of 60.28%. At the 8B parameter scale, Legal-R1 outperforms the strongest RAG baseline, R-Search, by 14.69 points in overall average and exceeds GPT-5.4-Mini by 3.41 points. Legal-R1 ranks second to DeepSeek-V4-Flash-Preview (284B). Figure 4 further presents the category-level results on the five LexEval categories, providing a more fine-grained view of Legal-R1’s performance across different dimensions of legal reasoning. Overall, these results demonstrate the strong and broadly effective legal-reasoning capability of Legal-R1.

Efficiency As illustrated in Figure 5, retrieval frequency varies with task complexity: the system uses an average of 1.2–1.7 turns for Knowledge and Understanding tasks, and 3.1–4.3 turns for Reasoning, Generation, and Consultation tasks. This pattern is observed on both LawBench and LexEval. The additional retrieval has a measurable cost. Relative to Qwen3-8B w/ hybrid, Legal-R1 raises the overall average score from 55.91 to 60.28, while the average LawBench retrieval/reasoning turns increase from 0.35 to 2.33 and LawBench token consumption increases from 5,196.62 to 10,861.85. The score is averaged across benchmarks, whereas the cost measures are from LawBench; the complete comparison is reported in Appendix C.

Ablation Study To verify the effectiveness of each component in our framework, we conduct an ablation study on Lawbench and LexEval, as reported in Table 2.

First, the results underscore that incorporating external knowledge is paramount for legal reasoning; equipping the vanilla Qwen3-8B with hybrid retrieval (‘w/ hybrid’) drastically boosts the overall average score from 27.85% to 55.91%. Second, domain-specific adaptation without retrieval (‘w/ post-train’) also yields a substantial improvement (+18.03 points on average), validating the necessity of legal domain fine-tuning. Finally, our complete method, Legal-R1 , integrates domain post-training, the multi-stage RAG pipeline, and GRPO reinforcement learning to achieve the highest average score of 60.28%. This final leap demonstrates that these components provide orthogonal benefits, culminating in optimal synergistic performance in complex legal scenarios.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Retrieval Scale</td><td rowspan="2"></td><td colspan="2">Lawbench</td><td colspan="2">LexEval</td><td rowspan="2">Avg</td></tr><tr><td>Trad. Score</td><td>LLM Judge</td><td>Trad. Score</td><td>LLM Judge</td></tr><tr><td>Qwen3-8B</td><td>x</td><td>8B</td><td>24.70</td><td>40.20</td><td>18.04</td><td>28.47</td><td>27.85</td></tr><tr><td>Qwen3-8B w/ post-train</td><td>x</td><td>8B</td><td>26.24</td><td>38.98</td><td>55.51</td><td>62.78</td><td>45.88</td></tr><tr><td>Qwen3-8B w/ hybrid</td><td>√</td><td>8B</td><td>52.02</td><td>59.85</td><td>50.52</td><td>61.49</td><td>55.91</td></tr><tr><td>Qwen3-8B w/ rerank</td><td>√</td><td>8B</td><td>53.28</td><td>57.97</td><td>50.12</td><td>57.31</td><td>54.67</td></tr><tr><td>Legal-R1 w/ hybrid</td><td></td><td>8B</td><td>57.08</td><td>63.69</td><td>55.47</td><td>61.96</td><td>59.55</td></tr><tr><td>Legal-R1 w/ rerank</td><td>√</td><td>8B</td><td>57.36</td><td>63.97</td><td>56.54</td><td>63.28</td><td>60.28</td></tr></table>

Table 2: Ablation study: Model Performance Comparison on LexEval, Lawbench and average Benchmarks. Judge by DeepSeek-V4-Flash-Preview.
<table><tr><td colspan="2">Case Study: Autonomous Multi-Turn Retrieval</td></tr><tr><td>Query Trajectory</td><td>Will a debtor automatically be blacklisted if they fail to repay a debt within the court&#x27;s 25-day deadline? Thinking Step 1: I need to locate the foundational statutory criteria for blacklisting. →Q Retrieval 1: Article</td></tr><tr><td></td><td>1.→ Thinking Step 2: Article 1 hinges on &quot;fulfillment capability.&quot; I must retrieve its specific definition. → Q Retrieval 2: Article  $2 .  \pmb { \mathscr { 0 } }$  Thinking Step 3: I need to search for procedural exemptions regarding the blacklisting process. → Q Retrieval 3: Article  ${ \dot { 3 } } . \hookrightarrow { \pmb { \ 9 } }$  Thinking Step 4: Let me invoke a precedent search to</td></tr><tr><td>Answer</td><td>validate how courts apply these articles in practice.→ Q Retrieval 4: Precedent Case. No, blacklisting is not automatic; according to Articles 1, 2, and 3, it strictly depends on whether the debtor willfully refuses to comply despite having the financial capability, provided they do not meet any legal exemptions.</td></tr></table>

Table 3: A case study of Legal-R1’s autonomous multi-turn retrieval, with the full trajectory provided in Table 6 in the appendix.

<table><tr><td>Legal-R1 Pipeline</td><td>LLM Judge</td><td>Trad. Score</td></tr><tr><td>Fixed 1-turn retrieval</td><td>61.70</td><td>53.95</td></tr><tr><td>Fixed 3-turn retrieval</td><td>62.75</td><td>54.65</td></tr><tr><td>W/ hybrid</td><td>63.69</td><td>57.08</td></tr><tr><td>W/o LLM rerank</td><td>62.99</td><td>57.35</td></tr><tr><td>W/o summarization</td><td>61.44</td><td>52.95</td></tr><tr><td>Full</td><td>63.97</td><td>57.36</td></tr></table>

Table 4: LawBench ablations of retrieval rounds and evidence-processing components.

Adaptive retrieval and evidence processing. Table 4 reports LawBench-only ablations. The fixedround variants modify the prompt and retrieval procedure to require one or three retrieval rounds, whereas the complete system plans retrieval actions and decides when to stop.

The complete system scores 57.36 on the traditional metric, compared with 53.95 and 54.65 for fixed one-turn and three-turn retrieval, respectively. These comparisons support adaptive retrieval planning, although the fixed-round variants also change the prompt and retrieval procedure. Removing evidence summarization reduces the traditional score to 52.95 and the LLM-judge score to 61.44. Removing LLM reranking has little effect on the traditional score (57.35 versus 57.36) but reduces the LLM-judge score from 63.97 to 62.99.

<table><tr><td>System</td><td>nDCG</td><td>Recall</td><td>MRR</td><td>Hit Rate</td></tr><tr><td>text2vec</td><td>0.3187</td><td>0.4109</td><td>0.2884</td><td>0.4109</td></tr><tr><td>BM25</td><td>0.6993</td><td>0.7789</td><td>0.6725</td><td>0.7789</td></tr><tr><td>Legal-R1</td><td>0.7659</td><td>0.8088</td><td>0.7516</td><td>0.8088</td></tr></table>

Table 5: Statute retrieval quality on 769 provisionoriented queries.

Statute retrieval quality. We evaluate retrieval on 769 provision-oriented queries; DeepSeek-V3 synthesizes verifiable QA pairs from legal benchmark materials and provisions. The corresponding answers are associated with provisions in the retrieval knowledge base. Table 5 compares the resulting retrieval rankings with text2vec and BM25.

Relative to BM25, Legal-R1 improves nDCG from 0.6993 to 0.7659 and MRR from 0.6725 to 0.7516; recall and hit rate also rise from 0.7789 to 0.8088. These results provide direct evidence of improved retrieval ranking for provision-oriented requests.

Case Study To intuitively illustrate the autonomous multi-turn reasoning and retrieval capabilities of Legal-R1, we conduct a qualitative case study on a representative real-world legal consultation. As analyzing in Table 3, given a user query regarding enforcement penalties, the model does not rely on static memories but independently initiates a sequence of precise retrieval loops to search for statutory criteria and validation procedures for dishonest judgment debtors. Notably, it strategically executes cross-references between general rules (Article 1) and exception clauses (Article 3), while invoking a legal precedent to ground its contextual understanding in judicial practice. Ultimately, through this systematic deliberation, Legal-R1 synthesizes an accurate and legally sound judgment, demonstrating its robust reliability in professional legal applications.

<table><tr><td>Model</td><td></td><td>Retrieval Avg. Score</td></tr><tr><td>Qwen3-8B</td><td>√</td><td>54.67</td></tr><tr><td>Qwen3-8B-SFT</td><td>√</td><td>59.98</td></tr><tr><td>Legal-R1</td><td>√</td><td>60.28</td></tr></table>

Table 6: Ablation study of training progression based on overall average scores.

SFT vs RL We isolate the distinct contributions of supervised fine-tuning (SFT) and reinforcement learning (RL) within our framework, as presented in Table 6. Transitioning from the raw RAG pipeline (‘w/ rerank’) to the SFT stage (‘Qwen3- 8B-SFT w/ rerank’) yields a prominent performance leap of 5.31 percentage points, demonstrating that SFT is vital for injecting foundational legal expertise and task-specific alignment. Building upon this optimized baseline, Legal-R1 leverages GRPO reinforcement learning to further elevate the overall average score to a peak of 60.28%. This continuous gain validates that RL successfully refines the model’s underlying reasoning policy, capitalizing on the specialized knowledge acquired during SFT to unlock maximum precision.

## 6 Conclusion

In this paper, we introduced Legal-R1, an agentic legal large language model for evidence-based question answering. Our framework couples a multi-turn Lawyer Agent with a multi-stage RAG pipeline via a Legal Assistant Agent. This pipeline uses hybrid retrieval followed by authority- and diversity-aware reranking, and leverages LLM filtering and summarization to distill compact legal evidence units. To optimize the central retrieval policy, Legal-R1 undergoes a streamlined two-stage training paradigm of Evidence-Masked Supervised Fine-Tuning (SFT) and Trajectory-Aware Reinforcement Learning via GRPO. Empirical results on the LexEval and LawBench benchmarks demonstrate the exceptional efficacy and efficiency of our approach. At the 8B parameter scale, Legal-R1 achieves a state-of-the-art overall average score of 60.28%, surpassing the proprietary GPT-5.4-Mini. Furthermore, our model maintains high efficiency by dynamically scaling its retrieval turns according to task complexity, achieving a favorable balance between computational cost and performance.

## Limitations

While Legal-R1 demonstrates robust and highly accurate performance in autonomous legal reasoning, it is not without minor limitations. Primarily, the multi-turn agentic retrieval process—although dynamically adjusted for task complexity—inherently introduces a slight inference latency compared to standard, single-pass generation models. While this computational overhead is justified by the significant gains in accuracy and legal grounding, it may require further pipeline optimization (e.g., parallelized evidence fetching or KV-cache optimization) before deployment in extremely timesensitive, real-time legal consultation environments. Additionally, the current model and knowledge base are deeply tailored to the Chinese legal system; extending this framework to support multijurisdictional or multilingual legal reasoning remains a promising avenue for future work.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pages 9112–9141.

Jaime Carbonell and Jade Goldstein. 1998. The use of mmr, diversity-based reranking for reordering documents and producing summaries. In Proceedings ofthe 21st annual international ACM SIGIR conference on Research and development in information retrieval, pages 335–336.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E Ho. 2024. Large legal fictions: Profiling legal hallucinations in large language models. Journal ofLegal Analysis, 16(1):64–93.

DeepSeek-AI. 2026. DeepSeek-V4-Pro. https://huggingface.co/deepseek-ai/ DeepSeek-V4-Pro. Accessed: 2026-05-25.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, and 1 others. 2024. Lawbench: Benchmarking legal knowledge of large language models. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 7933–7962.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, Haofen Wang, and 1 others. 2023. Retrievalaugmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1):32.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459– 9474.

Haitao Li, Yifan Chen, Shuo Miao, Qian Dong, Jia Chen, Yiran Hu, Junjie Chen, Minghao Qin, Yueyue Wu, Yujia Zhou, and 1 others. 2026. Legalone: a family of foundation models for reliable legal reasoning. arXiv preprint arXiv:2602.00642.

Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, and Yiqun Liu. 2024a. Lexeval: A comprehensive chinese legal benchmark for evaluating large language models. Advances in Neural Information Processing Systems, 37:25061–25094.

Haitao Li, Yunqiu Shao, Yueyue Wu, Qingyao Ai, Yixiao Ma, and Yiqun Liu. 2024b. Lecardv2: A largescale chinese legal case retrieval dataset. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2251–2260.

Antoine Louis, Gijs Van Dijck, and Gerasimos Spanakis. 2024. Interpretable long-form legal question answering with retrieval-augmented large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 22266–22275.

Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. In Findings ofthe associationfor computational linguistics: EMNLP 2020, pages 708–718.

OpenAI. 2026. Introducing GPT-5.4 mini and nano. https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/. Accessed: 2026-05-25.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is chatgpt good at search? investigating large language models as re-ranking agents. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 14918–14937.

Marc Van Opijnen and Cristiana Santos. 2017. On the concept of relevance in legal information retrieval. Artificial Intelligence and Law, 25(1):65–87.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings ofthe 61st annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 13484–13508.

Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stewart Massie, Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, and Bruno Fleisch. 2024. Cbr-rag: case-based reasoning for retrieval augmented generation in llms for legal question answering. In International Conference on Case-Based Reasoning, pages 445–460. Springer.

Yiquan Wu, Yuhang Liu, Yifei Liu, Ang Li, Siying Zhou, and Kun Kuang. 2026. Luwen technical report. arXiv preprint arXiv:2604.06737.

Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, and 1 others. 2018. Cail2018: A large-scale legal dataset for judgment prediction. arXiv preprint arXiv:1807.02478.

Hongru Xiao, Jie Li, Zhirui Li, Bin Yang, Xiang Li, Sunzhu Li, and Jiale Han. 2026. Structuring human objectives: A survey of rubrics for evaluation, alignment, and agentic ai. SSRN.

Cheng Xu, Shuhao Guan, Derek Greene, M Kechadi, and 1 others. 2024. Benchmark data contamination of large language models: A survey. arXiv preprint arXiv:2406.04244.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, and 1 others. 2023. Disc-lawllm: Fine-tuning large language models for intelligent legal services. arXiv preprint arXiv:2309.11325.

Fengji Zhang, Xinyao Niu, Chengyang Ying, Guancheng Lin, Zhongkai Hao, Zhou Fan, Chengen Huang, Jacky Keung, Bei Chen, and Junyang Lin. 2025. $\check { A ^ { 2 } }$ search: Ambiguity-aware question answering with reinforcement learning. arXiv preprint arXiv:2510.07958.

Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, and Limin Liu. 2025. R-search: Empowering llm reasoning with search via multi-reward reinforcement learning. arXiv preprint arXiv:2506.04185.

Zhi Zhou, Jiang-Xin Shi, Peng-Xiao Song, Xiao-Wen Yang, Yi-Xuan Jin, Lan-Zhe Guo, and Yu-Feng Li. 2024. Lawgpt: A chinese legal knowledgeenhanced large language model. arXiv preprint arXiv:2406.04614.

## A Prompts Used in the Agentic Retrieval Workflow

This appendix provides the prompts used by the LLM-based components in the proposed workflow.

## A.1 Normative Evidence Reranking Prompt

This prompt is used by the LLM-based normative reranker ${ \mathcal { L } } _ { n o r m }$ . Given a retrieval intent and candidate legal provisions from $\boldsymbol { \kappa } _ { n o r m }$ , the model ranks the candidates according to their relevance to the requested legal role, such as offense elements, sentencing ranges, or exception clauses. The detailed template implementation is illustrated in Figure 7.

## A.2 Precedent Evidence Summarization Prompt

This prompt is used by the case summarizer $\mathcal { L } _ { c a s e }$ Given a selected precedent, the model compresses it into an evidence unit that preserves the key facts, court level, judgment outcome, and sentencing information. The complete template configuration is detailed in Figure 8.

## A.3 Judge Answer Generation Prompt

This prompt is used by the Judge agent $\mathcal { L } _ { J u d g e }$ Given the legal question and the collected retrieval trajectory, the model generates the final answer based only on the retrieved legal norms and precedents, while avoiding unsupported claims. The structured implementation of this template is presented in Figure 9.

## B Reproducibility and Implementation Details

## B.1 Evaluation Configuration

For all reported evaluations, we set the decoding temperature to 0.0 and set max\_retries to 5. For each benchmark setting, the reported score is computed by summing the sample-level scores over the benchmark test set and dividing by the number of evaluated samples.

## B.2 Training Configuration and Reward Design

Training Configuration. Legal-R1 is initialized from Qwen3-8B and trained in two stages. We first perform supervised fine-tuning (SFT) on $D _ { \mathrm { S F T } }$ for 3 epochs with a learning rate of $5 \times 1 0 ^ { - 6 }$ and a batch size of 32. We then apply Group Relative Policy Optimization (GRPO) to further optimize the evidence-seeking policy, using a learning rate of $1 \times 1 0 ^ { - 6 }$ , a batch size of 32, and 8 rollouts per prompt.

Training Deployment. Training is conducted on a single node equipped with four NVIDIA H20 GPUs, each with 141 GB of VRAM. GPU 0–2 are used for Fully Sharded Data Parallel (FSDP) training and rollout generation. GPU 3 hosts the auxiliary components used during training, including the hybrid retrieval module, which combines BM25 and text2vec and consumes approximately 8 GB of VRAM. Two additional Qwen3-8B instances are deployed on GPU 3: a Legal Assistant model used for filtering, summarization, and reranking in the retrieval pipeline, and a reward judge model used for reward computation during GRPO. These two models are allocated approximately 40% and 50% of GPU 3 memory, respectively.

Reward Components and Weights. The trajectory-aware reward described in the Method section combines answer accuracy, trajectory alignment, retrieved-information gain, and query quality with weights $w _ { 1 } ~ = ~ 1 . 0 0 , ~ w _ { 2 } ~ = ~ 0 . 0 8$ $w _ { 3 } ~ = ~ 0 . 3 2$ , and $w _ { 4 } = 0 . 2 0$ , respectively. The rule-based format term checks valid answer, search, and information tags. The reward further includes a small bonus when the number of searches is close to the reference trajectory, a bounded penalty for overly long trajectories, and the GRPO KL regularization term with $\eta = 0 . 0 8$

## B.3 System Deployment and Inference Pipeline

At inference time, we first initialize the hybrid retriever and launch three model instances with vLLM: (i) the Legal Assistant Agent based on Qwen3-8B, which performs filtering, summarization, and reranking over retrieved candidates; (ii) the Judge Agent based on Qwen3-8B, which synthesizes the final evidence-grounded answer; and (iii) the Lawyer Agent, instantiated by the trained Legal-R1 model, which performs evidence planning and tool invocation.

Given a user question, the Lawyer Agent first analyzes the problem and issues a retrieval request. The hybrid retriever performs coarse-grained retrieval to construct a candidate document pool, after which the Legal Assistant Agent performs fine-grained filtering and reranking. The resulting evidence is returned to the Lawyer Agent, which continues the reasoning–retrieval loop until it determines that sufficient evidence has been collected. The complete retrieval trajectory—including the original question, retrieval actions, intermediate queries, and retrieved evidence—is then passed to the Judge Agent, which synthesizes the collected evidence into the final answer.

## B.4 Serving Cost and Throughput

For the serving stress test, the complete Legal-R1 inference system is deployed with vLLM on four NVIDIA A100-SXM4-80GB GPUs. Under concurrent inference, the measured GPU memory usage is approximately 72.2 GB, 70.1 GB, 70.1 GB, and 50.7 GB across the four GPUs, respectively. The end-to-end generation throughput is 239.98 output tokens/s, including internal reasoning and tool-call tokens. These measurements characterize the deployment cost of the complete Legal-R1 pipeline rather than the resource consumption of a singlepass 8B model.

## C LawBench Efficiency Analysis

LawBench Turns is the average number of retrieval/reasoning rounds during evaluation. Law-Bench Tokens is the average total token consumption, including input tokens, output tokens, internal reasoning tokens, and retrieval-related tokens. The score column reports the overall average score used in the main comparison, while the two cost metrics are measured on LawBench.

Table 7 reports Legal-R1 uses more retrieval/reasoning rounds and tokens than lightweight or shallow-retrieval baselines. For example, relative to Qwen3-8B w/ hybrid, it improves the average score from 55.91 to 60.28 while increasing Law-Bench Turns from 0.35 to 2.33 and LawBench Tokens from 5196.62 to 10861.85. Compared with the multi-turn AutoRefine-Qwen2.5-7B-Instruct baseline, Legal-R1 attains a higher average score with fewer turns and lower token consumption. Thus, the results characterize a favorable accuracy– cost trade-off among the evaluated multi-turn retrieval systems, rather than eliminating the additional runtime cost of Legal-R1.

## D LawBench 1-1 Statute Retrieval Case Study

We further analyze LawBench 1-1, a subset of 500 questions requiring the original text of legal provisions. After introducing the RAG and reranking pipeline, 364 of the 500 questions receive higher scores. This aggregate result indicates that retrieval augmentation is especially useful when the task requires faithful reproduction of statutory text.

For question ID 101, which asks for the content of Article 14 of the Consumer Rights Protection Law, Legal-R1 retrieves and reproduces the current version of the provision, including the clause that consumers’ personal information shall be protected according to law. In contrast, the post-trained model without RAG produces an older version that omits this clause. For question ID 12, which asks for Article 36 of the Law on Sole Proprietorship Enterprises, Legal-R1 answers correctly with retrieval support, whereas the post-trained model without

<table><tr><td>Model</td><td>Avg. Score</td><td>LawBench Turns</td><td>LawBench Tokens</td></tr><tr><td>GPT-5.4-Mini</td><td>56.87</td><td></td><td>399.23</td></tr><tr><td>AutoRefine-Qwen2.5-7B-Instruct</td><td>56.03</td><td>4.35</td><td>37226.08</td></tr><tr><td>Qwen3-8B w/ hybrid</td><td>55.91</td><td>0.35</td><td>5196.62</td></tr><tr><td>Legal-R1</td><td>60.28</td><td>2.33</td><td>10861.85</td></tr></table>

Table 7: Accuracy–cost comparison with LawBench efficiency metrics.

RAG generates the content of Article 32. These examples illustrate that parametric legal knowledge may be outdated or may cite an incorrect provision, while retrieval grounds the answer in current and verifiable legal text.

## E Full Trajectory on Case Study

## E.1 Case Study

To intuitively illustrate the autonomous multi-turn reasoning and retrieval capabilities of Legal-R1, we conduct a qualitative case study on a representative real-world legal consultation. As analyzing in Figure 10, given a user query regarding enforcement penalties, the model does not rely on static memories but independently initiates a sequence of precise retrieval loops to search for statutory criteria and validation procedures for dishonest judgment debtors. Notably, it strategically executes cross-references between general rules (Article 1) and exception clauses (Article 3), while invoking a legal precedent to ground its contextual understanding in judicial practice. Ultimately, through this systematic deliberation, Legal-R1 synthesizes an accurate and legally sound judgment, demonstrating its robust reliability in professional legal applications.

## F Legal Knowledge Base Annotation Details

## F.1 Authority Tier

We use the level of the adjudicating court as a proxy for the authority of a precedent case. Since judgments from higher-level courts generally carry stronger reference value, each case is mapped to one of three authority tiers and receives a tierspecific score for precedent reranking:

• Level 1 consists of cases adjudicated by high people’s courts. These cases receive an authority score of 1.1 and include 195 cases in $\mathrm { L e C a R D v } 2$

• Level 2 consists of cases adjudicated by intermediate people’s courts. These cases receive an authority score of 1.05 and include 3,731 cases in LeCaRDv2.

• Level 3 consists of cases adjudicated by primary people’s courts or other local tribunals. These cases receive an authority score of 1.0 and include 49,351 cases in LeCaRDv2.

These scores provide a lightweight prior rather than a hard filter. Cases from lower courts can still be selected when they are more relevant or contribute distinct sentencing evidence.

## F.2 Punishment Severity Index

To systematically evaluate judicial outcomes, we develop the Punishment Severity Index (PSI), a computational framework that converts heterogeneous sentencing descriptions into a continuous numerical scale.This ensures that linguistic variations in judicial texts (e.g., “three years” vs. “36 months”) are standardized into uniform computational inputs.

The core of the PSI(Punishment Severity Index) is based on a weighted temporal-equivalent model. Main penalties are first converted into a dayequivalent baseline (D), which is then adjusted by a Severity Coefficient (α) and a Probation Adjustment Factor (ϕ). Specifically, fixed-term imprisonment serves as the baseline $( \alpha = 1 . 0 )$ , while criminal detention and public surveillance are assigned lower weights $( \alpha = 0 . 8$ and 0.3, respectively) to reflect their lower degree of liberty deprivation. Furthermore, if a suspended sentence is detected, a multiplier of $\phi = 0 . 2$ is applied. For extreme non-temporal penalties, such as life imprisonment or the death penalty, the index assigns discrete, high-order ordinal values (e.g., 15, 000 to 100, 000) to maintain the hierarchical integrity of the penal system.

Finally, the index incorporates Supplementary Penalties through additive weighting. To account for the diminishing marginal impact of monetary sanctions, fines and property forfeitures are integrated using a logarithmic scaling function:

$S _ { f i n e } = 2 0 \times \log _ { 1 0 } ( \mathrm { a m o u n t + 1 ) }$ . Other sanctions, including the deprivation of political rights and deportation, are assigned fixed weights based on their statutory duration or severity. The total PSI is the cumulative sum of these components:

$$
P S I = ( D \cdot \alpha \cdot \phi ) + \sum S _ { s u p p l e m e n t a r y }
$$

This integrated metric provides a robust basis for subsequent statistical analysis and cross-case comparisons of sentencing severity.

## G Details of Training Data Construction

This appendix provides additional details on the construction of the post-training dataset used in our legal agent model. The goal of data construction is to obtain high-quality, evidence-grounded, and tooluse-oriented legal QA instances that can improve the model’s ability to understand legal facts, invoke external tools, retrieve relevant legal knowledge, and generate reliable answers.

## G.1 Data Sources and Task Coverage

Our post-training dataset consists of 18,940 structured QA instances, covering 5 major legal domains and 29 fine-grained subcategories, detailed in Table 9. The raw materials are collected from a combination of public legal NLP benchmarks, authoritative legal examination resources, and domainspecific synthetic problem sets. Specifically, the primary data sources include the Chinese AI and Law Challenge (CAIL), DISC-LawLLM, CrimeKgAssitant, and the National Unified Legal Professional Qualification Examination. These resources provide diverse legal materials, including statutory provisions, case facts, legal reasoning questions, criminal law problems, civil law problems, procedural law questions, and examination-style multiple-choice questions.

To improve task diversity, we organize the dataset according to a task-oriented configuration profile. Each task type is associated with a dedicated task description, input format, output format, and generation prompt. This design allows the pipeline to convert heterogeneous source materials into structured training instances under a unified schema. The resulting QA pairs cover both openended verifiable questions and legal selection tasks, enabling the model to learn different forms of legal reasoning, including fact understanding, legal norm matching, punishment prediction, legal element identification, and multi-step evidence-grounded answering.

## G.2 Overview of the Automated Construction Pipeline

To construct the dataset at scale while maintaining data quality, we design an end-to-end automated synthesis and purification pipeline. We use DeepSeek-V3 with temperature 0.1 to generate verifiable QA pairs from legal materials selected by task templates. Given raw legal materials, the pipeline first determines whether each source text is suitable for a target task, then synthesizes candidate QA pairs, and finally applies automatic qualitycontrol procedures. It rejects instances with invalid JSON, missing answers, task–material mismatches, or answer leakage. The overall pipeline consists of task-driven applicability gating, fingerprint-based contamination filtering, automated answer-leakage suppression, and rule-driven distractor design.

Formally, let $\mathcal { D } _ { r a w }$ denote the collection of raw legal materials, and let $\tau$ denote the set of target task types. For each source text $x \in \mathcal { D } _ { r a w }$ and each task type $\tau \in \mathcal { T }$ , the pipeline first evaluates whether x contains sufficient information to support task $\tau .$ . If the source text passes this applicability check, the model generates one or more candidate QA instances. Each candidate instance is then validated by contamination detection and taskspecific quality filters. Only instances that pass all filtering steps are retained in the final post-training dataset $\mathcal { D } _ { t r a i n }$

## G.3 Task-Driven Two-Stage Applicability Gating

A major challenge in legal data synthesis is that not every source text is suitable for every downstream task. For example, a statutory article may be appropriate for legal norm identification but may not contain sufficient factual information for punishment prediction. Similarly, a case fact description may support legal issue analysis but may not provide enough explicit evidence for a multiple-choice question about procedural requirements. Directly generating QA pairs from mismatched source materials can therefore introduce unanswerable questions, hallucinated answers, or weakly grounded supervision signals.

To address this issue, we implement a taskdriven two-stage applicability gating mechanism before QA synthesis. In the first stage, the pipeline performs a coarse task-material matching check based on metadata, source type, and task description. This step filters out clearly incompatible combinations, such as applying a sentencing-oriented task to a source text that contains only abstract legal definitions.

<table><tr><td>Penalty Case</td><td>Calculation Process</td><td>PSI Result</td></tr><tr><td>死刑 (Death penalty)</td><td>100,000</td><td>100,000.0</td></tr><tr><td>无期徒刑，剥夺政治权利终身 (Life imprisonment with lifelong deprivation of political rights)</td><td>15,000 + 2, 000</td><td>17,000.0</td></tr><tr><td>有期徒刑十五年 (Fixed-term imprisonment of fifteen years)</td><td> $( 1 5 \times 1 2 \times 3 0 ) \times 1 . 0$ </td><td>5,400.0</td></tr><tr><td>有期徒刑三年，缓刑四年 (Fixed-term imprisonment of three years with a</td><td> $( 3 \times 1 2 \times 3 0 ) \times 1 . 0 \times 0 . 2$ </td><td>216.0</td></tr><tr><td>four-year suspension of sentence) 拘役六个月，并处罚金一万元 (Criminal detention of six months, with a concur-</td><td> $( 6 \times 3 0 ) \times 0 . 8 + 2 0 \times \log _ { 1 0 } ( 1 0 , 0 0 0 + 1 )$ </td><td>224.0</td></tr><tr><td>rent fine of 10,000 RMB) 管制一年 (Public surveillance for one year)</td><td> $( 1 \times 1 2 \times 3 0 ) \times 0 . 3$ </td><td>108.0</td></tr><tr><td>驱逐出境</td><td>1,000</td><td>1,000.0</td></tr><tr><td>(Deportation) 免予刑事处罚 (Exemption from criminal punishment)</td><td>0</td><td>0.0</td></tr></table>

Table 8: Examples of PSI Calculation for Different Penalties.

In the second stage, the pipeline performs a model-based feasibility pre-vetting. Given a source text and the target task description, the model is instructed to explicitly judge whether the material contains sufficient factual predicates, legal entities, or normative conditions required by the task. The model must output a binary decision together with a briefjustification. If the material lacks key factual disclosures, contains insufficient legal context, or cannot support a verifiable answer, the corresponding task is skipped and recorded as a discarded instance.

This two-stage design filters low-quality candidates before generation rather than after generation. It reduces unnecessary synthesis costs and prevents the pipeline from producing superficially fluent but legally unsupported QA pairs.

## G.4 Fingerprint-Based Data Contamination Filtering

To reduce the risk of benchmark leakage, we incorporate a fingerprint-based contamination filtering module. Before running the data synthesis pipeline, all designated evaluation benchmarks are cleaned and normalized to construct a global fingerprint library. The normalization process removes redundant whitespace, standardizes punctuation, and unifies common formatting variants. Each benchmark sample is decomposed into exact 20-gram fingerprints.

During data construction, every candidate source text is compared against the fingerprint library before it is used for QA synthesis. If the source text contains an 20-gram segment that overlaps with the held-out benchmark fingerprints above a predefined threshold, the sample is treated as potentially contaminated and is removed from the generation process. This step is applied at the source-text level rather than only at the final QA-pair level, because a generated question may paraphrase a contaminated source and become difficult to detect after synthesis.

The contamination filter is designed to be conservative. Its purpose is not to remove all topically similar legal materials, since different legal cases may naturally involve similar statutes or legal concepts. Instead, it aims to eliminate near-duplicate or directly overlapping textual evidence from heldout benchmarks, thereby reducing the possibility that evaluation performance is inflated by memorization.

## G.5 Automated Syntax Parsing and Answer Leakage Suppression

For open-ended verifiable QA tasks, it is important that the question does not reveal the ground-truth answer. This is especially critical in agentic retrieval settings, where the model is expected to decide whether to invoke tools, retrieve legal evidence, and reason from the retrieved information. If the answer is already exposed in the question, the instance becomes trivial and no longer provides useful supervision for retrieval or legal reasoning.

<table><tr><td>Main Category</td><td>Sub-category</td><td>Count</td></tr><tr><td rowspan="3"></td><td>Clause Recall</td><td>500</td></tr><tr><td>Knowledge Retrieval Clause Retrieval</td><td>800</td></tr><tr><td>Subtotal</td><td>1,300</td></tr><tr><td rowspan="15">Understanding</td><td>Case Similarity</td><td>101</td></tr><tr><td>Dispute Identification</td><td>300</td></tr><tr><td>Concept Comprehension</td><td>1,000</td></tr><tr><td>Marriage Assessment</td><td>454</td></tr><tr><td>Subject Identification</td><td>1,000</td></tr><tr><td>Reading Comprehension</td><td>1,000</td></tr><tr><td>Event Detection</td><td>1,000</td></tr><tr><td>Trigger Extraction</td><td>1,000</td></tr><tr><td>Relation Understanding</td><td>68</td></tr><tr><td>Plaintiff Opinion</td><td>1,000</td></tr><tr><td>Document Correction</td><td>500</td></tr><tr><td>Subtotal</td><td>7,423</td></tr><tr><td>Charge Prediction</td><td>460</td></tr><tr><td>Charge with law Prediction</td><td>426</td></tr><tr><td>Amount Calculation</td><td>401</td></tr><tr><td>Sentence Prediction Reasoning</td><td>420</td></tr><tr><td>Scenario Analysis</td><td>1,000</td></tr><tr><td>Legal Calculation</td><td>1,000</td></tr><tr><td>Multi-hop Q&amp;A</td><td>1,000</td></tr><tr><td>Subtotal</td><td>4,707</td></tr><tr><td></td><td>Document Summarization</td></tr><tr><td rowspan="6">Generation</td><td>Indictment Generation</td><td>1,000 420</td></tr><tr><td>Contract Generation</td><td></td></tr><tr><td></td><td>8</td></tr><tr><td>Opinion Summarization Term Translation</td><td>1,000 500</td></tr><tr><td></td><td></td></tr><tr><td>Subtotal</td><td>2,928</td></tr><tr><td rowspan="4">Consultation</td><td>User Consultation</td><td>2,000</td></tr><tr><td>Evidence Opinion</td><td>158</td></tr><tr><td>Case Judgment</td><td>424</td></tr><tr><td>Subtotal</td><td>2,582</td></tr><tr><td>Total</td><td></td><td>18,940</td></tr></table>

Table 9: Distribution of Dataset Categories and Subcategories.

We therefore design an automated answerleakage suppression module. Given a generated QA pair (q, a), the module performs two layers of validation. First, it checks whether the answer string a appears directly in the question q. This direct substring matching is effective for detecting explicit leakage, such as questions that already contain the target charge, statute number, sentencing result, or legal conclusion.

Second, the module applies regular-expressionbased pattern matching to identify sensitive answerbearing values. For example, in punishment-related questions, the parser detects sentencing expressions, penalty types, durations, fines, suspension conditions, and other legally meaningful values. If such values appear in the question and match the expected answer format, the instance is flagged as answer-leaking and discarded. This rule is particularly useful when the answer is not copied verbatim but appears in a slightly different surface form.

After leakage filtering, the remaining questions are more suitable for evaluating and training the model’s ability to infer answers from legal evidence rather than relying on superficial cues in the question itself.

## G.6 Rule-Driven Heuristic Distractor Design for Legal Selection Tasks

Multiple-choice legal questions require highquality distractors. If distractor options are too obviously wrong, the task becomes overly simple; if they are semantically overlapping with the correct answer, the question may become ambiguous or legally invalid. To avoid these issues, we do not rely solely on unconstrained model generation for option construction. Instead, we design a ruledriven heuristic distractor generation module.

Given a source text and the corresponding correct answer, the module constructs negative options through two main corruption strategies.

Value Swapping. The first strategy creates distractors by swapping authentic values across legal entities or factual roles. For example, if a case description contains multiple actors, amounts, dates, or sentencing results, the algorithm reassigns a correct value from one entity to another incorrect entity. Since the values are drawn from the original text, the generated distractors remain plausible at the surface level. However, because the entityvalue alignment is corrupted, the distractors are legally incorrect.

Wrong Label Assignment. The second strategy constructs distractors by assigning correct textual entities to incorrect legal labels. For example, a factual element that corresponds to one legal role, charge, or statutory condition may be deliberately mapped to another label. This creates options that are lexically grounded in the source material but invalid under legal logic.

After generating candidate distractors, the module applies validity checks to ensure that each distractor is distinct from the correct answer and from other options. The correct option and the distractors are then randomly shuffled and mapped to an answer index, such as A, B, C, or D. This randomization prevents the model from exploiting positional bias during training.

## G.7 Quality Control and Final Dataset Assembly

After the above filtering and validation steps, all retained QA instances are converted into a unified training format. Each instance contains the task instruction, question, answer, optional choices, supporting evidence, and task metadata. For tooluse-oriented tasks, the instance may additionally include tool invocation annotations, intermediate evidence, or retrieval-oriented supervision signals.

The final dataset assembly process follows three principles. First, each instance should be answerable from its associated evidence or legal knowledge source. Second, the question should not contain direct answer leakage or shortcut cues. Third, the answer should be legally valid, concise, and consistent with the intended task type. Instances that violate any of these principles are removed.

Through this automated construction pipeline, we obtain a post-training dataset that is diverse in legal domain coverage, structured in task format, and reliable in supervision quality. The resulting dataset supports the model’s post-training for legal question answering, legal retrieval, tool invocation, and evidence-grounded reasoning.

## H The Use of Large Language Models

Using an LLM to help with paper writing During the preparation of this work, the authors utilized Large Language Models for language polishing, improving the structural clarity of the manuscript, and refining the formal expression of individual sentences. The use of Large Language Models did not influence the substantive content of the study and served solely as a writing aid.

## I Potential Risks

While Legal-R1 significantly enhances the accuracy of legal question answering through its active evidence-seeking framework, we acknowledge several potential risks associated with its deployment:

• Risk of Hallucination and Misguidance: As an LLM-based system, Legal-R1 may still suffer from subtle hallucinations or generate plausible-sounding but incorrect legal reasoning in highly complex or edge cases. If users, especially non-professionals, rely entirely on the model’s outputs for real-world litigation or legal decisions without professional verification, it could lead to severe legal or financial consequences.

• Over-Reliance on the System: The structured nature and highly confident tone of the multi-turn trajectories generated by Legal-R1 might inadvertently lead users to treat it as an absolute legal authority, reducing human legal practitioners’ critical oversight.

• Jurisdictional and Temporal Limitations: The current model and its underlying heterogeneous knowledge base are strictly tailored to the contemporary Chinese legal framework. Applying this model to foreign jurisdictions, alternative legal systems (e.g., common law), or outdated legal contexts could produce entirely invalid results.

To mitigate these risks, we explicitly emphasize that Legal-R1 is designed strictly as a supportive tool to assist legal professionals in research and evidence collection, rather than a replacement for qualified legal counsel. We recommend that any downstream deployment of this system incorporate clear user disclaimers and human-in-the-loop validation mechanisms to ensure safety and compliance.

![](images/3f11fedefead8e8122cde8888582e33bd730f3ec8f4fde91680a42cd19403af1.jpg)  
Figure 6: An idealized snapshot of Legal-R1’s autonomous thought-retrieval loop and final response synthesis.

![](images/26e7da21a6633b75a76b68c522410879bfef397ff4fd4f811cd88cca71241721.jpg)  
Figure 7: The prompt template utilized for normative evidence reranking. Within this template, {topk} denotes the predefined upper bound of retrieval turns, {query} and {context} represent the primary retrieval informational query, and {results} contains the candidate document pool obtained from the initial coarse-grained retrieval stage.

![](images/7cfd9dee93d331cfb78482f3beb74986170e68c41079d7178713872eae57f6fa.jpg)  
Figure 8: The prompt template utilized for precedent evidence summarization. Within this template, {fact\_query} denotes the target case query to be investigated, while {doc\_fact}, {doc\_reason}, and {doc\_result} represent the factual narrative, the judicial rationale, and the final adjudication outcome of the retrieved candidate precedent, respectively.

![](images/7a618d53bc67cf64cf83d7cebcdc3bee9645255fc603bde27cb65ddcfa4cbee3.jpg)  
Figure 9: The prompt template utilized for judge answer generation. Within this template, {query} denotes the initial legal question inputted by the user, and {history} represents the comprehensive reasoning chain and multi-turn retrieval trajectory generated by the lawyer agent.

![](images/b252e546887679af35818b7e100768e22307a55491a1e12a308f7ed787898979.jpg)  
Figure 10: An idealized snapshot of Legal-R1’s autonomous thought-retrieval loop and final response synthesis.