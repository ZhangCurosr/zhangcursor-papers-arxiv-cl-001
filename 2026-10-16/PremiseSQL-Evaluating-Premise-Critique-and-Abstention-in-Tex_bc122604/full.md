# PremiseSQL: Evaluating Premise Critique and Abstention in Text-to-SQL Systems

Shuo Lin<sup>1</sup> Jinzhe Li<sup>1,3</sup> Chenlu Guo<sup>4</sup> Yanhan Li<sup>5</sup> Yuan Wu<sup>1</sup>\* Yi Chang<sup>1,2,3</sup>\* <sup>1</sup>School of Artificial Intelligence, Jilin University

<sup>2</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China <sup>3</sup>International Center of Future Science, Jilin University <sup>4</sup>Bilibili Inc.

<sup>5</sup>Xingchen AGI Lab, China Telecom Artificial Intelligence Technology (Beijing) Co., Ltd {linshuo26,jinzhe25}@mails.jlu.edu.cn guochenlu01@bilibili.com, liyh181@chinatelecom.cn {yuanwu,yichang}@jlu.edu.cn

## Abstract

Recent advances in large language models (LLMs) have substantially improved Text-to-SQL performance, yet existing NL2SQL systems largely assume that user queries are valid and answerable. As a result, models often generate executable SQL even for queries with flawed premises. We introduce premise critique ability in NL2SQL, namely the ability to recognize flawed queries and abstain from generating misleading SQL. To evaluate this capability, we propose PremiseSQL, a benchmark containing 1,800 flawed NL2SQL queries across four categories of premise failures, along with their 1,800 original queries. We further design an evaluation framework to measure intrinsic and prompt-induced premise-aware behavior across three prompting settings. Experiments on frontier LLMs show that current models are strongly premise-trusting by default and rarely critique flawed premises. Stronger critiqueoriented prompting improves detection and abstention on flawed queries, but also increases false invalid judgments and suppresses SQL generation on originals. The code is available at https://github.com/vegeet/PremiseSQL.

## 1 Introduction

Text-to-SQL (or NL2SQL) translates natural language (NL) questions into executable SQL given a database schema (Popescu et al., 2003). Recent advances in large language models (LLMs) have substantially improved schema understanding, compositional reasoning, and cross-domain generalization in modern NL2SQL systems (OpenAI et al., 2024; Touvron et al., 2023; Yang et al., 2025; DeepSeek

![](images/99e35b00768bbfcb7850635e80e86370b0c5b65301f5e14f61521e87a7a53a95.jpg)  
Figure 1: Illustration of how LLMs handle an NL2SQL query with an ambiguous premise. The example contrasts two model behaviors: one generates executable SQL from an unsupported interpretation directly; the other identifies the ambiguity, requests clarification, and abstains from SQL generation. This contrast demonstrates the necessity of premise critique ability, namely the ability to recognize flawed queries and abstain from generating misleading SQL.

AI, 2025; Chang et al., 2024; Scholak et al., 2021;   
Yu et al., 2018; Pourreza and Rafiei, 2023).

Despite this progress, existing Text-to-SQL research largely assumes that user questions are valid, well-formed, and answerable under the given schema (Yu et al., 2018; Li et al., 2023). Consequently, most prior work focuses on improving

SQL generation accuracy, schema linking, and complex reasoning capabilities (Wang et al., 2020; Scholak et al., 2021; Pourreza and Rafiei, 2023). However, this assumption is frequently violated in realistic scenarios. User questions may contain logically inconsistent conditions, false schema assumptions, missing premises, or requests beyond the capability of SQL and the underlying database.

Nevertheless, state-of-the-art NL2SQL models often generate executable SQL even for flawed queries, producing SQL that is syntactically valid but semantically invalid (Ji et al., 2023; Whitehead et al., 2022; Ren et al., 2023; Huang et al., 2025). In many cases, models implicitly repair problematic questions by injecting unsupported assumptions (Lin et al., 2022). Figure 1 illustrates this failure: although "person with id 5" is underdetermined by the schema, a model may silently map person to actor and emit executable SQL. The resulting SQL is syntactically valid but rests on an unsupported entity choice.

We introduce premise critique ability in NL2SQL, namely the ability to recognize flawed queries and abstain from generating misleading SQL. Premise critique ability is orthogonal to conventional SQL generation accuracy: a model may produce executable SQL, or even generate SQL that coincidentally matches the user’s intent, while still failing to recognize that the provided question itself is ill-posed. Concretely, it consists of two components: (1) premise validation, namely assessing query validity in the corresponding database context and identifying actual flaws; and (2) abstention, namely refraining from generating misleading SQL for flawed queries while maintaining SQL generation for valid queries.

To systematically evaluate this capability, we introduce PremiseSQL, to our knowledge the first benchmark specifically designed to evaluate premise critique ability in NL2SQL systems. Built upon two widely used Text-to-SQL datasets, Spider 1.0 (Yu et al., 2018) and BIRD / BIRD-filtered (Li et al., 2023), PremiseSQL pairs 1,800 constructed flawed queries with their 1,800 unmodified source queries. It evaluates whether models can recognize flawed premises and abstain from generating misleading SQL, while maintaining SQL generation for original queries. We further evaluate frontier LLMs under three prompting settings, including Base, Implicit, and Explicit, to measure both intrinsic and prompt-induced premise-aware behavior.

In summary, our contributions are as follows:

• We formalize premise critique ability as a distinct requirement for reliable NL2SQL.

• We introduce PremiseSQL, a taxonomyguided benchmark comprising flawed queries as positive samples and their originals as negative samples.

• We develop an evaluation framework that separates intrinsic critique from prompt-induced behavior. Experiments on 16 frontier LLMs reveal a shift from trusting flawed premises under standard prompting to frequently rejecting valid queries under explicit critique instructions.

## 2 Related Work

## 2.1 Unanswerable and Ambiguous Queries in NL2SQL

Prior work has explored unanswerable and ambiguous queries in NL2SQL. TriageSQL (Zhang et al., 2020) formulates the problem as question intention classification, while Know What I Don’t Know (Wang et al., 2023) further categorizes ambiguity and unanswerability cases. More recent work extends beyond classification-oriented formulations. TrustSQL (Lee et al., 2024) introduces infeasible queries and proposes a reliability-oriented evaluation metric for trustworthy NL2SQL systems. Complementarily, NL2SQL-BUGs (Liu et al., 2025) focuses on errors in NL2SQL translation and categorizes semantic errors into fine-grained types. However, NL2SQL still lacks a fine-grained and comprehensive examination of diverse premise errors and the varying challenges they pose to models. PremiseSQL addresses this gap through a unified premise-error taxonomy.

## 2.2 Robustness and False Premise Handling in LLMs

Beyond NL2SQL, prior work studies LLM robustness under adversarial, noisy, or logically flawed inputs. For example, SQuADRUN (Rajpurkar et al., 2018) introduces unanswerable questions, highlighting models’ tendency to produce confident but incorrect responses when no valid answer exists. AbstentionBench (Kirichenko et al., 2025) provides a large-scale benchmark across 20 diverse datasets to assess abstention behavior. Building on these, we cover more types of flawed queries in NL2SQL and examine premise critique under different levels of premise-aware prompting, revealing both the limitations of this capability in LLMs and how it can be elicited. The paired originals further allow us to distinguish evidence-grounded critique from instruction-driven rejection.

<table><tr><td>Code</td><td>Category</td><td>Subcategory</td><td>Description</td></tr><tr><td rowspan="2">E1</td><td rowspan="2">Expression Error</td><td>E1.1 Ambiguity</td><td rowspan="2">Query contains ambiguous references, scopes, or targets that admit multiple plausible SQL interpretations. Query intent or ôperation type is underspecified, making</td></tr><tr><td>E1.2 Unclear Intent</td></tr><tr><td rowspan="2">E2</td><td rowspan="2">Premise Contradiction</td><td>E2.1 Internal Text Contradic-</td><td rowspan="2">Multiple query conditions cannot logically hold simulta- neously.</td></tr><tr><td>tion E2.2 Question-Schema Con- tradiction</td></tr><tr><td rowspan="2">E3</td><td rowspan="2">Missing Premise</td><td>E3.1 Missing Constraint</td><td rowspan="2">Essential filtering conditions or target specifications are omitted. Required tables, columns, or relations are not provided</td></tr><tr><td>E3.2 Missing Schema Premise</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>E3.3 Semantic Gap</td><td rowspan="2">in the schema. Query concepts cannot be semantically mapped to schema elements.</td></tr><tr><td>E4.1 Inexpressible in SQL</td></tr><tr><td rowspan="2">E4</td><td rowspan="2">Beyond Ability</td><td></td><td rowspan="2">Query intent cannot be precisely represented using stan- dard SQL. Correct answering requires knowledge absent from the</td></tr><tr><td>E4.2 External Knowledge Re- quired</td></tr></table>

Table 1: Taxonomy of flawed premises in Text-to-SQL.

<table><tr><td>Method</td><td>FP</td><td>Abst.</td><td>SQL</td><td>Fine</td><td>Prompt</td></tr><tr><td>TriageSQL</td><td>√</td><td>×</td><td>√</td><td>△</td><td>×</td></tr><tr><td>KWIDK</td><td>△</td><td>√</td><td>√</td><td>△</td><td>×</td></tr><tr><td>TrustSQL</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td></tr><tr><td>NL2SQL-BUGs</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td></tr><tr><td>SQuADRUN</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>AbstentionBench</td><td>√</td><td>√</td><td>×</td><td>△</td><td>△</td></tr><tr><td>PremiseSQL</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 2: Comparison with related work on flawedpremise and abstention evaluation. FP denotes explicit flawed-premise reasoning before SQL generation, Abst. denotes abstention evaluation, SQL indicates whether the benchmark is formulated around SQL generation tasks, Fine denotes fine-grained error taxonomy, and Prompt denotes prompt-aware evaluation settings. Symbols: ✓ Fully supported, △ Partially supported, × Not supported.

Summary Table 2 compares PremiseSQL with representative benchmarks. PremiseSQL jointly supports fine-grained premise-error analysis, abstention evaluation, SQL generation, paired original-query controls, and prompt-aware evaluation, enabling a more complete assessment of premise critique in NL2SQL.

## 3 Method

## 3.1 Task Formulation

Let x denote a natural language query and c the available database context, comprising the schema and any supplied auxiliary information. We associate each query–context combination $( x , c )$ with a reference label $y \in \{ 0 , 1 \}$ , where $y = 1$ indicates that the query has a premise flaw that warrants critique and abstention under the assumed SQL execution conditions, and $y = 0$ indicates the absence of such a flaw. These flaws may concern interpretation, logical consistency, required information, or the capabilities of SQL and the database, as specified in Table 1. PremiseSQL assigns $y = 1$ to constructed flawed queries and $y = 0$ to their originals, forming the positive and negative samples, respectively.

Given a prompting setting π, the complete request comprises the query x, the database context c, and the instructions specified by π. We represent the SQL and explanation components of a model’s response as

$$
\begin{array} { r } { ( \hat { q } , \hat { r } ) = f _ { \theta } ( x , c ; \pi ) , ~ } \\ { \hat { q } \in \mathcal { Q } _ { \mathrm { S Q L } } \cup \{ \perp \} , ~ } \end{array}\tag{1}
$$

where $\mathcal { Q } _ { \mathrm { S Q L } }$ denotes the space of candidate SQL outputs, ⊥ denotes abstention, and rˆ is a natural language explanation.

Premise critique requires both an evidencegrounded explanation and an appropriate generation decision. The desired abstention behavior is characterized by

$$
\begin{array} { r } { \hat { q } = \perp \quad \Longleftrightarrow \quad y = 1 . } \end{array}\tag{2}
$$

For $y = 1$ , the explanation rˆ must identify an actual premise flaw supported by $( x , c )$ ; abstention alone is insufficient to demonstrate premise critique. For $y = 0$ , the model should generate SQL without attributing unsupported flaws to the original query. The reference label y is determined by whether the query x is flawed in its database context c under the benchmark’s assumptions and is independent of π. Consequently, instructions that encourage scrutiny or assert the presence of a flaw should not alter the expected response to a given query.

![](images/a7ab0774ac527349e78b54c438800cafa8d3db5deccb55f61af508e47203c2ce.jpg)  
Figure 2: Overview of PremiseSQL. Starting from standard Text-to-SQL queries, we construct flawed premise variants under a unified taxonomy, followed by multi-stage verification and premise-aware evaluation under Base, Implicit, and Explicit settings.

## 3.2 Premise Taxonomy

We further organize flawed premises into a unified taxonomy. (1) Expression Error refers to errors caused by ambiguous or unclear natural language expressions that prevent the query intent from being uniquely mapped to SQL; (2) Premise Contradiction captures queries whose assumptions are logically inconsistent or incompatible with the database schema; (3) Missing Premise refers to queries that omit essential assumptions, constraints, or schema information required for reliable SQL generation and execution; (4) Beyond Ability includes queries whose requirements exceed the expressive or reasoning capabilities of SQL and the given database, making them unanswerable under the available execution assumptions. A complete taxonomy with fine-grained subcategories and definitions is shown in Table 1.

## 3.3 Benchmark Construction

Figure 2 shows that PremiseSQL pairs 1,800 flawed queries with their unmodified originals from Spider 1.0 (Yu et al., 2018) and BIRD / BIRD-filtered (Li et al., 2023). Each pair combines an original query with a taxonomy-guided perturbation of the query or supplied schema. The BIRD portion of E3.2 (Missing Schema Premise) uses BIRD rather than BIRD-filtered because it requires access to gold schema annotations.

To improve diversity and linguistic naturalness, perturbations are generated using GPT-5.1 under predefined error constraints (see Table 1 and Appendix A.3.1). All instances are assigned corresponding error categories and subcategories, and further verified using a two-stage process consisting of automatic evaluation by Gemini-2.5-Pro-Thinking and subsequent human review. Detailed statistics are provided in Appendix A.2.

## 3.4 Prompting Protocol

To analyze how prompting affects premise critique behavior, we evaluate models under three prompting settings: Base, Implicit, and Explicit.

Base Setting The Base setting follows the standard Text-to-SQL generation setup. Models are simply instructed to generate SQL for the given question and return a structured JSON output containing sql and reason, where sql denotes the generated SQL (or null when abstaining) and reason records the model’s explanation for its generation behavior. No additional warning is provided regarding potential flaws in the query.

Implicit Setting The Implicit setting introduces implicit premise-awareness guidance. Models are informed that the input question may contain errors and are asked to assess its validity before answering. The output JSON includes sql, is\_valid, and reason, where is\_valid indicates whether the model considers the question logically valid and answerable. This setting measures whether soft prompting can improve premise validation and abstention behavior.

Explicit Setting The Explicit setting provides direct critique-oriented instructions. Models are explicitly told that the question contains logically flawed or incorrect premises, and are required to critically examine its assumptions before responding. The output format remains sql, is\_valid, and reason. This setting examines premise critique behavior under an explicit assertion of a query flaw, which is supported for positive samples but false for negative samples.

## 3.5 Evaluation Framework

For positive samples, three metrics are considered: SQL Behavior (SQL), Detection (DET), and Type Identification (TYPE). SQL Behavior measures the degree of abstention from generating misleading SQL for a flawed query, using a three-level ordinal score from 0 to 2. Detection evaluates whether the model recognizes that the query contains a premise flaw, and Type Identification measures whether it correctly identifies the corresponding failure type.

For negative samples, two complementary metrics, namely SQL Generation (GEN) and Validity Judgment (VAL), are introduced. SQL Generation measures the proportion of negative samples for which the model generates SQL, reflecting whether it maintains generation for valid queries. GEN measures generation coverage; Appendix B.7 provides an execution-based validation showing that GEN closely tracks end-to-end execution success. Validity Judgment measures the proportion of negative samples explicitly judged valid through is\_valid=true, indicating whether the model correctly recognizes query validity.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate premise critique ability on a diverse set of state-of-the-art LLMs. The evaluated models include ChatGLM-5.1 (GLM-5 Team, 2026), Claude Opus 4.6 (Anthropic, 2026b), Claude Sonnet 4.6 (Anthropic, 2026a), Claude Opus 5.5 (Anthropic, 2026c), DeepSeek V3.2 (DeepSeek AI, 2025), DeepSeek V4.1 Flash (DeepSeek-AI, 2026), Doubao Seed 2.0 Pro (ByteDance, 2026), Gemini 3.1 Pro Preview (Google, 2026), LLaMA 4 Maverick (Meta AI, 2026), GPT-5.4 (OpenAI, 2026b), GPT-5.6-Luna (OpenAI, 2026a), GPT-6-Luna (OpenAI, 2026c), GPT-6.1-Sol (OpenAI, 2026d), and the 35B, 122B, and 397B variants of Qwen3.5 (Qwen Team, 2026).

Each of the 1,800 taxonomy-guided premise perturbations is paired with its unmodified source query. We treat the perturbed query as a positive sample, for which a model should detect the flaw and abstain, and the original query as a negative sample, for which it should recognize the query as valid and generate SQL. Each model therefore processes 3,600 requests under each of three prompting settings, or 10,800 requests in total. Across 16 models, our evaluation comprises 172,800 model responses. We use temperature 0 and the default API configuration for each model.

## 4.2 Main Results

Table 3 presents model performance on flawed queries and their paired originals. Across all three prompting settings, models exhibit limitations in premise critique ability, although these limitations take different forms. Base prompting reveals weak spontaneous critique, Implicit prompting produces partial and uneven improvements, and Explicit prompting exposes models’ susceptibility to the prompt’s assertion of a flaw. Together, these results show that current models do not reliably assess query premises across prompting settings. Figure 3 further illustrates how model behavior shifts from Implicit to Explicit prompting. Appendix E provides multi-model case studies that instantiate the main flawed-query outcomes and the corresponding failures in validity judgment and SQL generation on original queries.

Base Prompting Under Base prompting, aggregate positive-sample scores are only 0.177 SQL, 0.319 DET, and 0.311 TYPE, whereas negativesample GEN is 0.988. Even the strongest Base detector, Claude Opus 5.5 at 0.492, fails to identify more than half of the flaws. LLaMA 4 Maverick detects only 0.147 of flawed cases, though it generates SQL as expected for 0.998 of originals. Standard Text-to-SQL instructions therefore elicit limited spontaneous premise critique: models show weak ability to recognize and interpret flawed premises, while rarely exhibiting necessary abstention on flawed queries.

<table><tr><td rowspan="3">Model</td><td colspan="4">Base</td><td colspan="5">Implicit</td><td colspan="5">Explicit</td></tr><tr><td colspan="2">Flawed (+)</td><td colspan="2">Original (−) TYPE</td><td colspan="2">Flawed (+)</td><td colspan="2"></td><td colspan="2">Original (−)</td><td colspan="2">Flawed (+)</td><td colspan="2">Original (−)</td></tr><tr><td>SQL</td><td>DET</td><td></td><td>GEN</td><td>SQL</td><td>DET</td><td>TYPE</td><td>GEN</td><td>VAL</td><td>SQL</td><td>DET</td><td>TYPE</td><td>GEN</td><td>VAL</td></tr><tr><td>ChatGLM-5.1</td><td>0.118</td><td>0.228</td><td>0.222</td><td>1.000</td><td>0.585</td><td>0.599</td><td>0.565</td><td>0.981</td><td>0.963</td><td>0.975</td><td>0.957</td><td>0.785</td><td>0.516</td><td>0.396</td></tr><tr><td>Claude Opus 4.6</td><td>0.204</td><td>0.281</td><td>0.277</td><td>0.988</td><td>0.510</td><td>0.579</td><td>0.536</td><td>0.962</td><td>0.849</td><td>0.880</td><td>0.899</td><td>0.760</td><td>0.778</td><td>0.441</td></tr><tr><td>Claude Opus 5.5</td><td>0.402</td><td>0.492</td><td>0.480</td><td>0.984</td><td>0.509</td><td>0.673</td><td>0.646</td><td>0.969</td><td>0.870</td><td>0.726</td><td>0.853</td><td>0.746</td><td>0.788</td><td>0.578</td></tr><tr><td>Claude Sonnet 4.6</td><td>0.182</td><td>0.263</td><td>0.256</td><td>0.984</td><td>0.389</td><td>0.481</td><td>0.448</td><td>0.965</td><td>0.872</td><td>0.877</td><td>0.892</td><td>0.736</td><td>0.901</td><td>0.501</td></tr><tr><td>DeepSeek V3.2</td><td>0.240</td><td>0.422</td><td>0.394</td><td>0.938</td><td>0.906</td><td>0.898</td><td>0.764</td><td>0.455</td><td>0.437</td><td>0.983</td><td>0.978</td><td>0.771</td><td>0.149</td><td>0.034</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>0.110</td><td>0.247</td><td>0.242</td><td>0.989</td><td>0.566</td><td>0.625</td><td>0.569</td><td>0.926</td><td>0.844</td><td>0.966</td><td>0.932</td><td>0.771</td><td>0.311</td><td>0.264</td></tr><tr><td>Doubao Seed 2.0 Pro</td><td>0.163</td><td>0.305</td><td>0.298</td><td>0.998</td><td>0.684</td><td>0.683</td><td>0.620</td><td>0.962</td><td>0.772</td><td>0.958</td><td>0.938</td><td>0.796</td><td>0.891</td><td>0.278</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>0.142</td><td>0.272</td><td>0.272</td><td>1.000</td><td>0.706</td><td>0.706</td><td>0.688</td><td>0.995</td><td>0.990</td><td>0.934</td><td>0.923</td><td>0.829</td><td>0.941</td><td>0.704</td></tr><tr><td>LLaMA 4 Maverick</td><td>0.098</td><td>0.147</td><td>0.141</td><td>0.998</td><td>0.505</td><td>0.586</td><td>0.507</td><td>0.987</td><td>0.776</td><td>0.938</td><td>0.960</td><td>0.765</td><td>0.974</td><td>0.363</td></tr><tr><td>GPT-5.4</td><td>0.117</td><td>0.261</td><td>0.258</td><td>0.999</td><td>0.682</td><td>0.702</td><td>0.647</td><td>0.892</td><td>0.878</td><td>0.960</td><td>0.981</td><td>0.855</td><td>0.219</td><td>0.207</td></tr><tr><td>GPT-5.6-Luna</td><td>0.186</td><td>0.312</td><td>0.308</td><td>0.999</td><td>0.624</td><td>0.629</td><td>0.593</td><td>0.959</td><td>0.904</td><td>0.975</td><td>0.898</td><td>0.796</td><td>0.288</td><td>0.231</td></tr><tr><td>GPT-6-Luna</td><td>0.197</td><td>0.386</td><td>0.384</td><td>0.999</td><td>0.623</td><td>0.652</td><td>0.634</td><td>0.954</td><td>0.931</td><td>0.866</td><td>0.852</td><td>0.785</td><td>0.563</td><td>0.507</td></tr><tr><td>GPT-6.1-Sol</td><td>0.276</td><td>0.417</td><td>0.413</td><td>1.000</td><td>0.692</td><td>0.717</td><td>0.690</td><td>0.955</td><td>0.906</td><td>0.965</td><td>0.932</td><td>0.845</td><td>0.352</td><td>0.322</td></tr><tr><td>Qwen3.5-35B</td><td>0.186</td><td>0.399</td><td>0.384 0.338</td><td>0.999</td><td>0.777</td><td>0.780</td><td>0.678</td><td>0.942</td><td>0.759</td><td>0.972 0.996</td><td>0.964 0.962</td><td>0.786 0.797</td><td>0.253</td><td>0.130</td></tr><tr><td>Qwen3.5-122B</td><td>0.153</td><td>0.351</td><td></td><td>0.931</td><td>0.708</td><td>0.711</td><td>0.644</td><td>0.807</td><td>0.795</td><td></td><td></td><td></td><td>0.446</td><td>0.179</td></tr><tr><td>Qwen3.5-397B</td><td>0.054</td><td>0.315</td><td>0.312</td><td>0.992</td><td>0.690</td><td>0.723</td><td>0.676</td><td>0.737</td><td>0.665</td><td>0.998</td><td>0.968</td><td>0.817</td><td>0.004</td><td>0.005</td></tr></table>

Table 3: Premise critique performance on positive and negative samples. SQL is normalized SQL Behavior; DET and TYPE denote Detection and Type Identification. GEN is the SQL-generation rate on original queries, and VAL is the rate at which the model explicitly recognizes those queries as valid. Darker teal indicates better performance.

Implicit Prompting Implicit validity guidance increases aggregate SQL, DET, and TYPE to 0.635, 0.672, and 0.619, respectively, while original-query GEN and VAL remain 0.903 and 0.826. The gains are substantial but uneven. DeepSeek V3.2 reaches the highest DET (0.898), yet its GEN and VAL on originals are only 0.455 and 0.437. Gemini 3.1 Pro Preview attains lower DET and TYPE (0.706 and 0.688) while preserving nearly all originals. Thus, a soft warning exposes some latent critique ability, but it neither closes the recognition–explanation gap nor produces uniformly reliable premise judgments.

Explicit Prompting Explicit prompting raises aggregate SQL, DET, and TYPE to 0.936, 0.931, and 0.790. Because this prompt asserts that a flaw exists, however, high scores can reflect compliance with that assertion rather than independent critique. The evaluation therefore tests whether critique follows the evidence: models should identify a supported flaw in perturbed queries, but should also reject the prompt’s unsupported claim of a flaw when processing originals. The latter requirement distinguishes genuine premise assessment from instruction-driven rejection. Nevertheless, aggregate GEN and VAL on originals fall to 0.523 and 0.321. Qwen3.5-397B is the clearest rejection case, scoring 0.998 SQL and 0.968 DET on flawed queries but only 0.004 GEN and 0.005 VAL on originals. Gemini 3.1 Pro Preview is less susceptible, retaining 0.941 GEN and 0.704 VAL. Overall, stronger instructions improve apparent critique performance without producing consistently evidencegrounded premise judgments.

## 4.2.1 Model Comparisons

Aggregate trends conceal substantial variation within and across model families. Model rankings also shift across prompting settings.

Within model families, Base DET rises from 0.261 for GPT-5.4 to 0.313 for GPT-5.6-Luna, 0.386 for GPT-6-Luna, and 0.417 for GPT-6.1-Sol; TYPE follows the same ordering. Claude Opus 5.5 likewise improves Base DET from 0.281 to 0.492 relative to Opus 4.6 and raises Implicit TYPE from 0.536 to 0.646. Yet even the best Base result remains below 0.5, and the GPT ordering is not stable under stronger prompts.

Other comparisons reinforce the absence of a general scaling trend. Under Implicit prompting, DeepSeek V4.1 Flash has much higher originalquery GEN than V3.2 (0.926 versus 0.455), but lower DET on flawed queries (0.625 versus 0.898). The 35B Qwen variant also outperforms its larger counterparts on Implicit DET. Newer or larger models therefore show partial, setting-dependent gains rather than a consistent improvement in spontaneous premise critique.

![](images/f3090772c0d5ea64919bcc29ee07f9dcc864a2a86a2ce54054bf8fddd14b7be7.jpg)  
Figure 3: DET–GEN shifts from Implicit to Explicit prompting across 16 models. Teal and sage bars denote flaw detection (DET) on flawed queries and SQL generation (GEN) on paired originals, respectively. Black caps mark Implicit scores, while bar endpoints and arrowheads mark Explicit scores; outward/inward arrows indicate increases/decreases. Explicit prompting increases DET but decreases GEN for every model, showing that stronger flaw cues improve detection at the cost of more false rejection of valid queries.

## 4.2.2 Validity Judgments and Behavior

We pool 115,200 outputs from Implicit and Explicit prompting and exclude 145 malformed flawedquery responses, retaining 57,455 flawed-query and 57,600 original-query outputs. Among the 47,801 flawed-query outputs with is\_valid=false, 96.06% receive DET = 1 and 94.12% completely abstain; among the 9,654 with is\_valid=true, 98.66% receive DET = 0 and 99.92% generate SQL. Thus, on flawed queries, the reported validity judgment closely tracks both flaw recognition and generation behavior, even when that judgment is incorrect.

The originals show why such consistency does not imply correctness. Of the 24,568 originalquery outputs with is\_valid=false, 16,502 (67.17%) contain no SQL. By contrast, SQL is generated in 33,010 of the 33,032 original-query outputs (99.93%) with is\_valid=true. Models therefore tend to act on their own validity judgment: when the judgment is wrong, the generation decision often follows the same error.

Explicit prompting amplifies this problem.

Among outputs declaring the query invalid, the proportion corresponding to flawed queries falls from 20,100 of 25,119 (80.02%) under Implicit prompting to 27,701 of 47,250 (58.63%) under Explicit prompting. Equivalently, original-query outputs account for 41.37% of all invalid judgments under Explicit prompting. The resulting increase in false invalid judgments suggests that models rely too heavily on the prompt’s premise that the query is flawed, even when their judgments and generation behavior are internally consistent.

## 4.2.3 Automatic Evaluation and Reliability

All model outputs are independently scored by GPT-5.4 and Claude Opus 4.6 under identical instructions and scoring definitions. Figure 7 presents a representative evaluation prompt, while Appendix B.2 provides the complete rubrics and validation details. Inter-evaluator Cohen’s κ reaches 0.9737 for SQL Behavior, 0.9100 for Detection, and 0.8586 for Type Identification.

We further verify the protocol on 2,160 randomly sampled outputs from one open-weight model (DeepSeek V3.2) and one proprietary model (GPT-5.4). Against two human annotators, the mean Cohen’s κ scores are 0.9701, 0.9545, and 0.9072 for Claude Opus 4.6, and 0.9567, 0.9355, and 0.8188 for GPT-5.4, on SQL Behavior, Detection, and Type Identification, respectively. These results support the reliability of the automatic evaluation used below.

## 4.3 Error-Type Analysis

Type identification remains more demanding than general flaw detection. TYPE is below DET in every setting, and the aggregate gap widens to 0.141 under Explicit prompting. Table 4 therefore breaks down the Implicit results by premise-error type.

<table><tr><td>Code</td><td>Error Type</td><td>SQL</td><td>DET</td><td>TYPE</td></tr><tr><td>E1.1</td><td>Ambiguity</td><td>0.210</td><td>0.242</td><td>0.138</td></tr><tr><td>E1.2</td><td>Unclear Intent</td><td>0.202</td><td>0.193</td><td>0.131</td></tr><tr><td>E2.1</td><td>Internal Text Contrad.</td><td>0.920</td><td>0.972</td><td>0.966</td></tr><tr><td>E2.2</td><td>Question-Schema Contrad.</td><td>0.881</td><td>0.923</td><td>0.915</td></tr><tr><td>E3.1</td><td>Missing Constraint</td><td>0.537</td><td>0.611</td><td>0.543</td></tr><tr><td>E3.2</td><td>Missing Schema Premise</td><td>0.568</td><td>0.586</td><td>0.474</td></tr><tr><td>E3.3</td><td>Semantic Gap</td><td>0.710</td><td>0.757</td><td>0.726</td></tr><tr><td>E4.1</td><td>Inexpressible in SQL</td><td>0.853</td><td>0.900</td><td>0.838</td></tr><tr><td>E4.2</td><td>External Knowledge Req.</td><td>0.830</td><td>0.861</td><td>0.841</td></tr></table>

Table 4: Performance by taxonomy error type under Implicit prompting, aggregated across all 16 models and both datasets with labels averaged over two judges.

The results reveal a pronounced dependence on how premise flaws are expressed. Contradictions are easiest(DET and TYPE above 0.91), while ambiguity and unclear intent are hardest (DET below 0.25; TYPE below 0.14). Missing-premise cases fall between these extremes; models also perform relatively well on queries that exceed SQL expressibility or require external knowledge. Taken together, the pattern suggests that Implicit prompting elicits critique mainly when a flaw appears as an overt incompatibility or recognizable capability boundary. It is much less effective when detecting the flaw requires reasoning about underspecification or absent information, indicating that premise critique remains cue-dependent rather than robust across error forms.

## 4.4 From Error Types to Failure Behaviors

The preceding error-type analysis establishes which premise flaws are difficult to recognize, but it does not explain what models do with those flaws when critique fails. We therefore move from an input-level taxonomy to a response-level account. Whereas an error category characterizes a query flaw in its database context, a failure mechanism identifies the observable operation through which the model makes the query appear answerable.

We annotate 8,000 eligible failures with four non-exclusive mechanisms: Assumption Injection (M01) introduces an unsupported interpretation, value, or proxy; Contradiction Reinterpretation (M02) modifies incompatible requirements; Context/Evidence Misuse (M03) treats partial schema or example evidence as conclusive; and Explicit Constraint Omission (M04) drops a complete requirement. Attribution requires strict agreement between two judges, with strong automatic and human reliability (macro $\kappa = 0 . 8 1 8 3$ and 0.9434, respectively; see Appendix C for details).

Among the eligible failures analyzed across prompting settings, M01 is the dominant failure mechanism (72.60%), followed by M03 (10.75%), while M02 (1.38%) and M04 (0.86%) remain rare. More importantly, Figure 4 shows a marked redistribution across prompting settings. From Base to Explicit prompting, M01 decreases from 81.03% to 31.12%, whereas M03 increases from 1.46% to 58.06%; the Implicit setting lies between these endpoints. Stronger instructions suppress silent completion of missing query premises, yet the remaining failures increasingly arise because models overinterpret limited schema or example evidence when deciding whether, and how, the premise is defective.

![](images/41ffb53d3dc14e5d75de01dba722f78f867304c033e0a37a85bcbb38b7795ff5.jpg)  
Figure 4: Failure-mechanism incidence across prompting settings. Lines show the percentage of eligible failures assigned each mechanism by both judges. The M01–M03 reversal shows that stronger critique instructions shift residual failures from silent assumption injection to overinterpreting limited evidence rather than eliminating them, while M02 and M04 remain rare.

The association with prompting setting is substantial for M01 (Cramér’s $V = 0 . 3 5 6 )$ and stronger for M03 $( V = 0 . 5 8 1 )$ , with Holm-adjusted $p < 0 . 0 0 1$ in both cases.

Mechanism-level analysis therefore complements the error taxonomy by showing how an observed premise flaw is converted into an apparently answerable query. Despite their different surface forms, these failures reveal a shared limitation in that models tend to produce apparently answerable interpretations without recognizing the underlying premise flaws.

## 5 Conclusion

In this work, we introduce PremiseSQL, to our knowledge the first benchmark for evaluating premise critique ability in NL2SQL systems. PremiseSQL contains 1,800 flawed queries covering 9 fine-grained error types and their 1,800 originals as positive and negative samples, respectively. Experiments under Base, Implicit, and Explicit prompting settings reveal a clear reliability gap in current NL2SQL systems. Models often trust user assumptions and generate misleading SQL under standard settings. Even when explicitly prompted to scrutinize premises, they struggle to assess them reliably. Models further struggle with subtle and implicit premise flaws, showing that strong SQL generation does not necessarily imply trustworthy premise-aware reasoning. We hope PremiseSQL can facilitate future research on more reliable and premise-aware NL2SQL systems.

## Limitations

Our work has several limitations. First, PremiseSQL focuses on English NL2SQL benchmarks and may not generalize to multilingual settings or real-world interactive database systems. Second, although the benchmark covers multiple premise failure categories, the constructed errors are still taxonomy-driven and may not fully capture naturally occurring user mistakes. Finally, our evaluation relies partially on LLM-based automatic judges, which may introduce evaluation bias despite strong agreement with human annotations.

## References

Anthropic. 2026a. Introducing claude 4.6 sonnet. https://www.anthropic.com/news/ claude-sonnet-4-6. Accessed: 2026-02.

Anthropic. 2026b. Introducing claude opus 4.6. https: //www.anthropic.com/news/claude-opus-4-6. Accessed: 2026-02.

Anthropic. 2026c. Introducing claude opus 5.5. https: //www.anthropic.com/claude-opus-5-5. Accessed: 2026-10-05.

ByteDance. 2026. Seed2.0 technical introduction. https://research.doubao.com/zh/seed2. Accessed: 2026-02.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. ACM Trans. Intell. Syst. Technol., 15(3).

DeepSeek AI. 2025. Deepseek-v3.2: Pushing the frontier of open large language models. Preprint, arXiv:2512.02556.

DeepSeek-AI. 2026. DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression. Preprint, arXiv:2609.19969.

GLM-5 Team. 2026. Glm-5: from vibe coding to agentic engineering. Preprint, arXiv:2602.15763.

Google. 2026. Gemini 3.1 pro preview. https://docs.cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/3-1-pro. Accessed: 2026-02.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Inf. Syst., 43(2).

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12).

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. Abstentionbench: Reasoning llms fail on unanswerable questions. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Gyubok Lee, Woosog Chay, Seonhee Cho, and Edward Choi. 2024. TrustSQL: Benchmarking text-to-SQL reliability with penalty-based scoring.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin C.C. Chang, Fei Huang, Reynold Cheng, and Yongbin Li. 2023. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Xinyu Liu, Shuyu Shen, Boyan Li, Nan Tang, and Yuyu Luo. 2025. Nl2sql-bugs: A benchmark for detecting semantic errors in nl2sql translation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 5662–5673, New York, NY, USA. Association for Computing Machinery.

Meta AI. 2026. The llama 4 herd: Architecture, training, evaluation, and deployment notes. Preprint, arXiv:2601.11659.

OpenAI. 2026a. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/ index/gpt-5-6/. Accessed: 2026-10-05.

OpenAI. 2026b. Introducing gpt-5.4. https:// openai.com/index/introducing-gpt-5-4/. Accessed: 2026-03.

OpenAI. 2026c. Introducing GPT-6 Sol and Luna. https://openai.com/index/ introducing-gpt-6-sol-and-luna/. Accessed: 2026-10-05.

OpenAI. 2026d. Introducing GPT-6.1 Sol. https:// openai.com/index/introducing-gpt-6-1-sol/. Accessed: 2026-10-05.

OpenAI, Josh Achiam, Steven Adler, and et al. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Ana-Maria Popescu, Oren Etzioni, and Henry Kautz. 2003. Towards a theory of natural language interfaces to databases. In Proceedings ofthe 8th International Conference on Intelligent User Interfaces, IUI ’03, page 149–157, New York, NY, USA. Association for Computing Machinery.

Mohammadreza Pourreza and Davood Rafiei. 2023. Din-sql: decomposed in-context learning of text-tosql with self-correction. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for SQuAD. In Proceedings ofthe 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia. Association for Computational Linguistics.

Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J. Liu. 2023. Out-of-distribution detection and selective generation for conditional language models. Preprint, arXiv:2209.15558.

Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. 2021. Picard: Parsing incrementally for constrained auto-regressive decoding from language models. Preprint, arXiv:2109.05093.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Bailin Wang, Richard Shin, Xiaodong Liu, Oleksandr Polozov, and Matthew Richardson. 2020. RAT-SQL: Relation-aware schema encoding and linking for textto-SQL parsers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7567–7578, Online. Association for Computational Linguistics.

Bing Wang, Yan Gao, Zhoujun Li, and Jian-Guang Lou. 2023. Know what I don’t know: Handling ambiguous and unknown questions for text-to-SQL. In Findings of the Association for Computational Linguistics: ACL 2023, pages 5701–5714, Toronto, Canada. Association for Computational Linguistics.

Spencer Whitehead, Suzanne Petryk, Vedaad Shakib, Joseph Gonzalez, Trevor Darrell, Anna Rohrbach, and Marcus Rohrbach. 2022. Reliable visual question answering: Abstain rather than answer incorrectly. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVI, page 148–166, Berlin, Heidelberg. Springer-Verlag.

An Yang, Anfeng Li, Baosong Yang, and et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3911–3921, Brussels, Belgium. Association for Computational Linguistics.

Yusen Zhang, Xiangyu Dong, Shuaichen Chang, Tao Yu, Peng Shi, and Rui Zhang. 2020. Did you ask a good question? a cross-domain question intention classification benchmark for text-to-sql. arXiv preprint arXiv:2010.12634.

## A Benchmark Construction Details

## A.1 Dataset Composition

PremiseSQL consists of 1,800 flawed NL2SQL queries across four major premise error categories and nine fine-grained subcategories, together with their 1,800 originals as positive and negative samples, respectively. Each subcategory contains 200 positive samples. To preserve natural query diversity and realistic schema distributions, we do not strictly enforce identical source proportions across Spider 1.0 and BIRD / BIRD-filtered databases. The detailed statistics of positive samples across error categories and source benchmarks are summarized in Table 5.

<table><tr><td>Error Category / Subcategory</td><td>Spider</td><td>BIRD</td></tr><tr><td>Expression Error (400)</td><td></td><td></td></tr><tr><td>– Ambiguity</td><td>90</td><td>110</td></tr><tr><td>– Unclear Intent</td><td>107</td><td>93</td></tr><tr><td>Premise Contradiction (400)</td><td></td><td></td></tr><tr><td>– Internal Text Contradiction</td><td>103</td><td>97</td></tr><tr><td>– Question-Schema Contradiction</td><td>156</td><td>44</td></tr><tr><td>Missing Premise (600)</td><td></td><td></td></tr><tr><td>– Missing Constraint</td><td>29</td><td>171</td></tr><tr><td>– Missing Schema Premise</td><td>64</td><td>136</td></tr><tr><td>– Semantic Gap</td><td>155</td><td>45</td></tr><tr><td>Beyond Ability (400)</td><td></td><td></td></tr><tr><td>– Inexpressible in SQL</td><td>149</td><td>51</td></tr><tr><td>– External Knowledge Required</td><td>140</td><td>60</td></tr><tr><td>Total</td><td>993</td><td>807</td></tr></table>

Table 5: Statistics of PremiseSQL across fine-grained premise error categories and source benchmarks.

## A.2 Construction and Filtering

For the eight subcategories requiring perturbationbased construction, we generate 15,192 candidate instances and apply two-stage quality control, consisting of LLM verification followed by human review. The LLM verification uses the review prompts described in Section A.3.2. A total of 2,248 instances entered the human-approved pool; these are pre-selection counts and therefore may exceed the 200 examples ultimately retained per subcategory. The average rejection rates were 57.8% during LLM verification and 64.9% during human review. Table 6 distinguishes the approved pool from the final allocation of positive samples. E3.2 (Missing Schema Premise) is constructed separately through schema-sensitive perturbations and therefore does not undergo this candidate filtering process.

<table><tr><td>Subcat.</td><td>Cand. LLM Pass</td><td></td><td>Human Appr.</td><td>Final</td></tr><tr><td>E1.1</td><td>4,280</td><td>708</td><td>200</td><td>200</td></tr><tr><td>E1.2</td><td>1,108</td><td>673</td><td>382</td><td>200</td></tr><tr><td>E2.1</td><td>1,108</td><td>886</td><td>457</td><td>200</td></tr><tr><td>E2.2</td><td>1,108</td><td>933</td><td>350</td><td>200</td></tr><tr><td>E3.1</td><td>2,758</td><td>692</td><td>252</td><td>200</td></tr><tr><td>E3.2</td><td></td><td></td><td></td><td>200</td></tr><tr><td>E3.3</td><td>1,108</td><td>737</td><td>204</td><td>200</td></tr><tr><td>E4.1</td><td>1,107</td><td>627</td><td>203</td><td>200</td></tr><tr><td>E4.2</td><td>2,615</td><td>1,152</td><td>200</td><td>200</td></tr><tr><td>Total</td><td>15,192</td><td>6,408</td><td>2,248</td><td>1,800</td></tr></table>

Table 6: Candidate generation and filtering statistics. Cand. denotes generated candidates, LLM Pass denotes candidates passing LLM review, Human Appr. denotes those retained after manual review, and Final denotes the benchmark allocation. E3.2 is constructed separately.

## A.3 Generation and Review Prompts

Benchmark construction uses two structured prompt stages. The generation stage introduces one controlled premise flaw into a source Text-to-SQL instance, and the review stage filters candidates against category-specific validity criteria. The following prompts show the shared structure through the Ambiguity category; other categories substitute their own definitions and checks.

## A.3.1 Generation Prompt

We design a unified prompt structure for error synthesis. We show one representative example for the Ambiguity category in Figure 5. Other categories follow the same structure with category-specific definitions.

## A.3.2 Review Prompt

We design a unified prompt template for automatic quality control and validation. We show one representative example for the Ambiguity category in Figure 6. Other categories follow the same review structure with category-specific evaluation criteria.

## B Experimental and Evaluation Protocol

## B.1 Base, Implicit, and Explicit Prompts

We provide the prompt templates used for the three prompting settings in our experiments: Base, Implicit, and Explicit. All prompts follow a unified input format consisting of the database schema and the natural language question.

## Base Prompt.

The database schema is as follows:   
{schema}

Question: {question}

Generate SQL to answer the question.

```json
{
"sql": string or null,
"reason": string
}
```

Return structured JSON only.

## Implicit Prompt.

The database schema is as follows: {schema}

Question: {question}

The question may contain errors. You must assess its validity before answering.

Generate SQL to answer the question.

```json
{
"sql": string or null,
"is_valid": boolean,
"reason": string
}
```

Return structured JSON only.

## Explicit Prompt.

The database schema is as follows: {schema}

Question: {question}

The question is logically flawed and contains incorrect premises. You must critically examine its assumptions. Then explain the problem.

```json
{
"sql": string or null,
"is_valid": boolean,
"reason": string
}
```

Return structured JSON only.

## B.2 Evaluation Rubrics

Automatic Evaluation Protocol. For each positive-sample output, an evaluator receives the user question, database schema, model response, and ground-truth premise-error type. It then applies a unified instruction and the category-specific definition to assign SQL Behavior, Detection, and Type Identification labels according to the rubrics below. For each negative-sample output, the evaluator assigns SQL Generation and Validity Judgment labels based on whether the model generates SQL and recognizes the original query as valid. GPT-5.4 and Claude Opus 4.6 perform the evaluation independently under identical instructions and scoring definitions, and we average their judgments to obtain the reported scores. Figure 7 provides the full positive-sample evaluation prompt for a representative subcategory; the remaining subcategories use the same structure with their corresponding definitions.

Positive-Sample Metrics. SQL Behavior. Measures the degree of abstention from generating SQL for flawed queries.

• 0: generates complete SQL;

• 1: partial abstention or invalid SQL output;

• 2: complete abstention or explicit refusal.

Detection. Measures whether the model recognizes the query as flawed.

• 0: no detection;

• 1: detects flawed premises.

Type Identification. Measures whether the model correctly identifies the underlying flaw type.

• 0: incorrect type identification;

• 1: correct type identification.

Type Identification is assigned 0 when Detection = 0; otherwise, it evaluates whether the corresponding failure type is correctly identified.

Negative-Sample Metrics. SQL Generation. Measures whether the model generates SQL for a negative sample.

• 0: does not generate SQL;

• 1: generates SQL.

GEN is reported as the proportion of negative samples for which SQL is generated and is available in all three prompting settings.

Validity Judgment. Measures whether the model explicitly recognizes a negative sample as valid.

• 0: does not explicitly judge the query valid;

• 1: explicitly judges the query valid through is\_valid=true.

VAL is reported as the proportion of negative samples explicitly judged valid. It is available only in the Implicit and Explicit prompting settings because the Base output format does not include is\_valid.

## B.3 Automatic Evaluation Prompt

We use LLM-based evaluation prompts to assess model behavior on flawed queries. The evaluation focuses on three metrics: SQL Behavior, Detection, and Type Identification.

Given a user question, model response, and the ground-truth flaw category, the automatic evaluator determines whether the model: (1) refrains from generating misleading SQL, (2) recognizes that the question is flawed, and (3) correctly identifies the underlying flaw type.

We adopt a unified evaluation prompt structure across all error categories, with category-specific definitions and scoring criteria. Figure 7 shows a representative example for the Ambiguity subcategory, used only for positive-sample evaluation.

## B.4 Human Verification

We conduct human verification on 2,160 outputs from one open-weight model (DeepSeek V3.2) and one proprietary model (GPT-5.4). The sample is evenly stratified: each model and each source benchmark contributes 1,080 outputs, each prompting setting contributes 720, and each of the nine fine-grained error types contributes 240, yielding 80 outputs per prompting-setting–error-type combination. Human annotators follow the same evaluation protocol as the automatic evaluators. Two graduate-level annotators familiar with SQL and database systems independently evaluate the same 2,160 sampled outputs using the rubrics applied by the automatic evaluators. Agreement between the two human annotators is high, with Cohen’s κ of 0.9912, 0.9632, and 0.8759 for SQL Behavior, Detection, and Type Identification, respectively.

## B.5 Robustness to Judge–Target Overlap

GPT-5.4 and Claude Opus 4.6 serve as both evaluated models and automatic evaluators, creating potential judge–target overlap. Although averaging their judgments reduces the influence of either evaluator, it does not by itself eliminate possible self-evaluation or model-family bias. We therefore conduct a leave-self-out robustness check. For GPT-5.4 outputs, we retain only the judgments of Claude Opus 4.6; for Claude Opus 4.6 outputs, we retain only those of GPT-5.4.

Table 7 shows no systematic self-favoring pattern. Self-evaluation slightly raises some Base scores but lowers others, including GPT-5.4’s Explicit TYPE score. After removing self-evaluation, the aggregate SQL, DET, and TYPE scores change by less than 0.001 in every prompting setting. The leading models also remain unchanged: Claude Opus 5.5 under Base prompting, DeepSeek V3.2 under Implicit prompting, and GPT-5.4 for Explicit DET and TYPE.

We further apply a stricter family-disjoint evaluation in which all OpenAI models are scored only by Claude Opus 4.6 and all Claude models are scored only by GPT-5.4; models from other families retain the two-judge average. Across the nine setting– metric combinations, aggregate scores change by at most 0.0016, the top-ranked model remains unchanged, and the original and family-disjoint model rankings have Spearman correlations ranging from 0.944 to 1.000. These results indicate that judge– target family overlap does not materially affect the aggregate results, leading models, or conclusions about prompting behavior. GEN and VAL are unaffected because they are derived directly from the models’ structured outputs rather than from automatic evaluator judgments.

## B.6 Paired Evaluation Design

The paired inputs expose a progression from undercritique to instruction-driven rejection. Each pair contrasts an original query–context combination with its perturbed counterpart. Reliable critique should change the response to the perturbation while maintaining correct validity judgments and SQL generation for originals. Base prompting usually answers both; stronger prompts increase abstention, but Explicit prompting often suppresses originals as well. The separation among DET, TYPE, and SQL on flawed queries and VAL and GEN on originals therefore distinguishes evidencebased critique from broad rejection.

The five measures provide complementary evidence about premise critique behavior. DET records whether any flaw is noticed, TYPE whether the intended defect is explained, and SQL whether that assessment changes generation. On originals, VAL records the explicit judgment and GEN the action. Their divergence is informative: under Explicit prompting, LLaMA 4 Maverick retains 0.974 GEN but only 0.363 VAL, so it often generates despite incorrectly judging an original query as flawed; Qwen3.5-397B is low on both and instead follows a broad rejection policy. Evaluation on originals prevents either pattern from being mistaken for robust critique.

## B.7 Execution-Based Validation of SQL Generation

GEN is designed to measure generation coverage, namely whether a model preserves SQL generation on valid original queries. GEN and VAL are evaluated on all 1,800 original queries. For the supplementary execution-based analysis, we retain 1,726 originals whose gold SQL produces a reliable reference result in our SQLite evaluation environment; the remaining 74 are excluded only from this analysis because their execution correctness cannot be determined reliably. This yields 27,616 model– query outputs per prompting setting across the 16 evaluated models.

Let $g _ { i }$ indicate whether SQL is generated for the i-th original query, and let $e _ { i }$ indicate whether the generated SQL is execution-equivalent to the gold

<table><tr><td rowspan="2">Target Model</td><td rowspan="2">Setting</td><td colspan="3">Reported Two-Judge Average</td><td colspan="3">Cross-Family Judge Only</td></tr><tr><td>SQL</td><td>DET</td><td>TYPE</td><td>SQL</td><td>DET</td><td>TYPE</td></tr><tr><td rowspan="3">GPT-5.4</td><td>Base</td><td>0.117</td><td>0.261</td><td>0.258</td><td>0.109</td><td>0.240</td><td>0.239</td></tr><tr><td>Implicit</td><td>0.682</td><td>0.702</td><td>0.648</td><td>0.681</td><td>0.697</td><td>0.671</td></tr><tr><td>Explicit</td><td>0.960</td><td>0.981</td><td>0.855</td><td>0.959</td><td>0.982</td><td>0.907</td></tr><tr><td rowspan="3">Claude Opus 4.6</td><td>Base</td><td>0.204</td><td>0.281</td><td>0.277</td><td>0.205</td><td>0.296</td><td>0.288</td></tr><tr><td>Implicit</td><td>0.510</td><td>0.579</td><td>0.536</td><td>0.513</td><td>0.579</td><td>0.523</td></tr><tr><td>Explicit</td><td>0.880</td><td>0.899</td><td>0.760</td><td>0.882</td><td>0.896</td><td>0.714</td></tr></table>

Table 7: Leave-self-out evaluation for models that also serve as automatic evaluators. The reported scores average judgments from GPT-5.4 and Claude Opus 4.6. The cross-family scores retain only Claude Opus 4.6 judgments for GPT-5.4 outputs and only GPT-5.4 judgments for Claude Opus 4.6 outputs.

<table><tr><td>Setting</td><td>GEN</td><td>EX</td><td>EX | GEN</td></tr><tr><td>Base</td><td>0.987</td><td>0.713</td><td>0.722</td></tr><tr><td>Implicit</td><td>0.902</td><td>0.652</td><td>0.722</td></tr><tr><td>Explicit</td><td>0.524</td><td>0.399</td><td>0.762</td></tr></table>

Table 8: Execution-based validation on the 1,726 original queries with reliably executable gold SQL. EX measures execution accuracy over these queries, treating abstention as incorrect, while EX | GEN measures execution accuracy among outputs for which SQL is generated.

SQL, with $e _ { i } = 0$ when no SQL is generated. We compute overall execution accuracy as

$$
\mathrm { E X } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e _ { i } ,\tag{3}
$$

and execution accuracy conditional on generation as

$$
\operatorname { E X } \mid \operatorname { G E N } = { \frac { \sum _ { i = 1 } ^ { N } e _ { i } } { \sum _ { i = 1 } ^ { N } g _ { i } } } .\tag{4}
$$

As shown in Table 8, GEN closely tracks overall execution accuracy, whereas execution accuracy conditional on generation remains comparatively stable across prompting settings. Across the 48 model–setting combinations, GEN is strongly correlated with EX (Pearson’s $r = 0 . 9 9 0 , p < . 0 0 1 ;$ Spearman’s $\rho = 0 . 8 8 8$ $p \ < \ . 0 0 1 )$ . The association remains strong when models are compared within the Implicit $( r = 0 . 9 5 1 , p < . 0 0 1 )$ and Explicit $( r = 0 . 9 9 7 , p < . 0 0 1 )$ settings. Under Base prompting, the correlation is weaker $( r = 0 . 4 3 4$ $p = . 0 9 3 )$ because GEN is near its upper bound for almost all models, leaving little variation across models.

More directly, the model-level change in GEN from Implicit to Explicit prompting is strongly correlated with the corresponding change in EX (Pearson’s $r = 0 . 9 9 2 , p < . 0 0 1$ ; Spearman’s $\rho = 0 . 9 8 5$ $p < . 0 0 1 )$ . Meanwhile, EX | GEN changes only from 0.722 to 0.762. The decline in overall execution accuracy under Explicit prompting is therefore primarily associated with suppressed SQL generation rather than reduced correctness among the SQL outputs that remain. These results do not make GEN a substitute for execution accuracy; instead, they support its intended use as a measure of whether models preserve SQL generation on valid queries.

## B.8 Bootstrap Uncertainty Analysis

We assess the uncertainty of the principal modeland error-level comparisons using 10,000 bootstrap replicates. For comparisons across prompting settings or models, each replicate resamples the 1,800 aligned query instances with replacement, preserving the paired structure of the observations. For error-type results, we resample the 200 source queries within each error type while retaining all outputs from the 16 evaluated models and both judges for each sampled query. We report percentile 95% confidence intervals and two-sided centered-bootstrap tests. These intervals quantify uncertainty over benchmark queries; the evaluated models and judges are treated as fixed.

Table 9 confirms that Explicit prompting increases DET and decreases GEN for every evaluated model. None of the 32 confidence intervals crosses zero, and all changes remain significant after Holm correction $( p _ { \mathrm { a d j } } < . 0 0 4 )$ . Even the smallest GEN decrease, observed for LLaMA 4 Maverick, remains distinguishable from zero. The central DET–GEN trade-off is therefore not driven by sampling variation in a small subset of queries.

Uncertainty in model comparisons. Paired comparisons support the claim that model rankings depend on the prompting setting. GPT-6.1-Sol exceeds GPT-5.4 in Base DET by 0.156 (95% CI [0.135, 0.176], p < .001), whereas their difference under Implicit prompting is not significant (0.014, 95% CI [−0.004, 0.033], p = .129). Under Explicit prompting, the ordering reverses: GPT-5.4 exceeds GPT-6.1-Sol by 0.050 (95% CI [0.039, 0.061], p < .001).

<table><tr><td>Model</td><td>∆DET</td><td></td><td>∆GEN</td></tr><tr><td>ChatGLM-5.1</td><td>0.358 [0.336, 0.380]</td><td></td><td>-0.465 [-0.488, -0.442]</td></tr><tr><td>Claude Opus 4.6</td><td>0.319 [0.297, 0.341]</td><td></td><td>-0.183[-0.203, -0.164]</td></tr><tr><td>Claude Opus 5.5</td><td>0.180 [0.162, 0.199]</td><td></td><td>-0.181[-0.199, -0.163]</td></tr><tr><td>Claude Sonnet 4.6</td><td>0.411 [0.389, 0.434]</td><td></td><td>-0.064[-0.078, -0.050]</td></tr><tr><td>DeepSeek V3.2</td><td>0.079 [0.066, 0.092]</td><td></td><td>-0.306[-0.327, -0.284]</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>0.307 [0.286, 0.329]</td><td></td><td>-0.615[-0.638, -0.592]</td></tr><tr><td>Doubao Seed 2.0 Pro</td><td>0.254 [0.235, 0.275]</td><td></td><td>-0.072[-0.086, -0.058]</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>0.217 [0.198, 0.236]</td><td></td><td>-0.054[-0.066, -0.044]</td></tr><tr><td>LLaMA 4 Maverick</td><td>0.374 [0.351, 0.396]</td><td></td><td> $- 0 . 0 1 3 \left[ - 0 . 0 2 1 , - 0 . 0 0 4 \right]$ </td></tr><tr><td>GPT-5.4</td><td>0.279 [0.259, 0.299]</td><td></td><td> $- 0 . 6 7 2 \left[ - 0 . 6 9 4 , - 0 . 6 5 1 \right]$ </td></tr><tr><td>GPT-5.6-Luna</td><td>0.269 [0.249, 0.289]</td><td></td><td> $- 0 . 6 7 1 \left[ - 0 . 6 9 3 , - 0 . 6 4 9 \right]$ </td></tr><tr><td>GPT-6-Luna</td><td>0.200 [0.182, 0.219]</td><td></td><td> $- 0 . 3 9 1 \left[ - 0 . 4 1 4 , - 0 . 3 6 8 \right]$ </td></tr><tr><td>GPT-6.1-Sol</td><td>0.215 [0.197, 0.234]</td><td></td><td>-0.603[-0.625, -0.580]</td></tr><tr><td>Qwen3.5-35B</td><td>0.184 [0.166, 0.203]</td><td></td><td> $- 0 . 6 8 9 \left[ - 0 . 7 1 0 , - 0 . 6 6 7 \right]$ </td></tr><tr><td>Qwen3.5-122B</td><td>0.251 [0.232, 0.271]</td><td></td><td> $- 0 . 3 6 1 \left[ - 0 . 3 8 7 , - 0 . 3 3 3 \right]$ </td></tr><tr><td>Qwen3.5-397B</td><td>0.245 [0.226, 0.265]</td><td></td><td> $- 0 . 7 3 3 \left[ - 0 . 7 5 4 , - 0 . 7 1 3 \right]$ </td></tr></table>

Table 9: Paired bootstrap estimates for the change from Implicit to Explicit prompting. Each entry reports the point estimate and 95% confidence interval, with ∆ = Explicit − Implicit.

The non-monotonic Qwen comparison is also statistically supported. Under Implicit prompting, Qwen3.5-35B exceeds Qwen3.5-122B in DET by 0.069 (95% CI [0.054, 0.086], $p ~ < ~ . 0 0 1 )$ and Qwen3.5-397B by 0.057 (95% CI [0.040, 0.074], $p < . 0 0 1 )$ . The difference between the 122B and 397B variants is not significant (−0.013, 95% CI [−0.028, 0.002], p = .097). Thus, the 35B result is robust, but it should be interpreted as a settingspecific comparison rather than a general advantage of smaller models.

The DeepSeek comparison further illustrates why performance cannot be summarized by a single ordering. Relative to DeepSeek V3.2 under Implicit prompting, DeepSeek V4.1 Flash has substantially higher GEN on originals (+0.471, 95% CI [0.448, 0.494], p < .001) but lower DET on flawed queries (−0.274, 95% CI [−0.296, −0.252], p < .001).

Uncertainty across error types. Table 10 confirms that the large differences across error types are not explained by query sampling variation. Relative to the two expression-error subcategories, the two contradiction subcategories are higher by 0.695 in SQL Behavior (95% CI [0.670, 0.718]), 0.730 in DET (95% CI [0.705, 0.754]), and 0.807 in TYPE (95% CI [0.785, 0.828]); all three differences have $p ~ < ~ . 0 0 1$ . The conclusion that overt contradictions are substantially easier than ambiguity and unclear intent is therefore robust, while the wider intervals for several missing-premise categories indicate greater query-level heterogeneity.

## C Failure-Mechanism Attribution

## C.1 Mechanism Definitions and Annotation Protocol

This analysis builds on the behavioral labels previously assigned by GPT-5.4 and Claude Opus 4.6. We use their existing SQL Behavior, Detection, and Type Identification judgments to identify eligible failures. For mechanism attribution, a separate evaluator pair, GPT-6.1-Sol and Claude Opus 5.5, assigns M01–M04 to these selected failures without re-evaluating the three behavioral dimensions.

Each sampled response is annotated independently by the two mechanism-attribution judges using the same final attribution prompt. The task is multi-label rather than four-way classification: each judge must assess M01–M04 exactly once, and a response may receive multiple mechanisms or none. Table 16 gives the positive tests and the principal boundaries used in the prompt.

Information shown to the attribution judges. For each item, the judges receive (i) the flawed question and its source error type and flaw description; (ii) the tested model’s response, separated into its reason and SQL; (iii) the schema and auxiliary evidence actually exposed to the tested model, together with provenance and exposure-status fields; (iv) the validated SQL Behavior, Detection, and Type Identification labels; (v) the complete reference schema and its provenance, when available; (vi) the paired valid question, SQL, and evidence used during benchmark construction; and (vii) datacoverage and other recovered context metadata.

<table><tr><td>Code</td><td>Error Type</td><td>SQL</td><td>DET</td><td>TYPE</td></tr><tr><td>E1.1</td><td>Ambiguity</td><td>0.210 [0.181, 0.241]</td><td>0.242 [0.209, 0.276]</td><td>0.138 [0.115, 0.163]</td></tr><tr><td>E1.2</td><td>Unclear Intent</td><td>0.202 [0.177, 0.228]</td><td>0.193 [0.170, 0.218]</td><td>0.131 [0.111, 0.151]</td></tr><tr><td>E2.1</td><td>Internal Text Contradiction</td><td>0.920 [0.908, 0.932]</td><td>0.972 [0.963, 0.980]</td><td>0.966 [0.956, 0.976]</td></tr><tr><td>E2.2</td><td>Question-Schema Contradiction</td><td>0.881 [0.855, 0.906]</td><td>0.923 [0.896, 0.947]</td><td>0.915 [0.887, 0.941]</td></tr><tr><td>E3.1</td><td>Missing Constraint</td><td>0.537 [0.494, 0.580]</td><td>0.611 [0.564, 0.657]</td><td>0.543 [0.494, 0.593]</td></tr><tr><td>E3.2</td><td>Missing Schema Premise</td><td>0.568 [0.542, 0.594]</td><td>0.586 [0.561, 0.612]</td><td>0.474 [0.449, 0.498]</td></tr><tr><td>E3.3</td><td>Semantic Gap</td><td>0.710 [0.674, 0.746]</td><td>0.757 [0.720, 0.792]</td><td>0.726 [0.688, 0.765]</td></tr><tr><td>E4.1</td><td>Inexpressible in SQL</td><td>0.853 [0.829, 0.875]</td><td>0.900 [0.877, 0.920]</td><td>0.838 [0.811, 0.863]</td></tr><tr><td>E4.2</td><td>External Knowledge Required</td><td>0.830 [0.800, 0.859]</td><td>0.861 [0.834, 0.888]</td><td>0.841 [0.812, 0.868]</td></tr></table>

Table 10: Implicit-prompting performance by error type with query-level bootstrap 95% confidence intervals. Models and judges are retained as fixed panels within each resampled query.

Verdicts and evidence requirements. For every mechanism, a judge returns one of supported, not\_supported, or uncertain. supported requires the mechanism’s positive test and the required evidence; not\_supported means the available record is sufficient to determine that the test is not met; and uncertain is reserved for cases in which missing context prevents a defensible decision. Evidence for M01, M02, and M04 must include an exact quote of the affected requirement from the current flawed question and an exact quote from the response’s reason or SQL showing the transformation. M03 instead requires an erroneous availability or answerability inference in the response plus counterevidence from verified model-visible context. The output also records the target requirement, transformation type, context basis, a concise explanation, and three control fields: whether the target is complete and explicit, whether any partial implementation remains, and whether the response supplies a concrete replacement. Deterministic post-processing validates the required quotes, context basis, control fields, and same-target exclusivity, and then derives the positive-label list only from validated supported verdicts.

M01, M02, and M04 are mutually exclusive when they describe the same atomic requirement: a concrete substitute is M01, a changed satisfaction relation for an explicit contradiction is M02, and complete disappearance without replacement is M04. They may coexist only when they concern distinct requirements. M03 is independent, but the same inference from nonappearance in bounded examples is not counted again as M01. Complete abstention alone is not M04, and ordinary SQL implementation defects are outside this mechanism taxonomy.

Binary coding and strict consensus. The same binary coding is used for automatic and human reliability analyses: supported is coded as 1, whereas both not\_supported and uncertain are coded as 0. A mechanism enters the reported label set only when both attribution judges mark it as supported. This rule is applied separately to M01–M04, so the final label set is simply the intersection of the judges’ supported labels; their complete four-label combinations need not match.

Final attribution prompt. Figure 8 reproduces the task and output contract from the final prompt. The prompt evaluates observable response operations rather than inferring a mechanism from the validated outcome scores; its label-specific decision tests and exclusions are stated in Table 16 and the preceding paragraphs.

## C.2 Eligibility and Sampling

The mechanism analysis begins from outputs that both primary evaluators, Claude Opus 4.6 and GPT-5.4, independently regard as failures. All three evaluation fields must be valid: Detection and Type Identification must be binary, and SQL Behavior must be in {0, 1, 2}. An output is a candidate failure when Type Identification= 0 or SQL Behavior̸= 2. Candidate records are paired only when their sample identity and their database ID, question, reason, and SQL match exactly. The two evaluators must then agree on all three fields. Records with non-string reasons, schema-placeholder outputs, or no usable reason/SQL evidence are excluded because they cannot support mechanism attribution.

This strict wash retains 36,684 responses: 18,701 from BIRD and 17,983 from Spider. The exclusion log contains 4,258 otherwise matched candidates: 4,035 evaluator disagreements, 180 non-string reasons, and 43 SQL schema placeholders. We then restore the original generation context and paired valid reference. Before sampling, we remove 4,279 outputs that correctly identify the flaw but still answer it, 4,801 E3.2 cases for which the historical reduced-schema exposure cannot be recovered, and 31 cases without a recoverable paired-valid reference. This leaves 27,573 eligible failures (13,709 BIRD and 13,864 Spider).

<table><tr><td>Stage</td><td>BIRD</td><td>Spider</td><td>Total</td></tr><tr><td>Strict wash retained</td><td>18,701</td><td>17,983</td><td>36,684</td></tr><tr><td>Eligible for attribution</td><td>13,709</td><td>13,864</td><td>27,573</td></tr><tr><td>Final sampled responses</td><td>4,000</td><td>4,000</td><td>8,000</td></tr></table>

Table 11: Failure-mechanism attribution pool after the strict two-evaluator wash, context-based eligibility filtering, and sampling.

For each dataset–model combination, we sample exactly 250 responses, yielding $1 6 \times 2 \times 2 5 0 =$ 8,000 observations. Sampling strata are prompting setting × source error category. Every nonempty stratum receives at least one observation, and the remaining quota is allocated approximately in proportion to stratum size by the largest-remainder method. Within each stratum, records are ranked deterministically by a SHA-256 hash with seed 42. The final sample contains 4,587 Base, 2,433 Implicit, and 980 Explicit responses. Because of the unrecoverable historical schema exposure noted above, E3.2 is not represented in this mechanismattribution sample. All reported mechanism incidences are calculated directly from these 8,000 sampled responses without reweighting.

Aggregate incidence. Across the 8,000 sampled failures, strict two-judge consensus identifies M01 in 5,808 responses, M02 in 110, M03 in 860, and M04 in 69. Because the mechanisms are nonexclusive, these counts need not sum to the sample size.

## C.3 Reliability and Human Validation

Agreement between the attribution judges, GPT-6.1-Sol and Claude Opus 5.5, yields Cohen’s κ values of 0.7917, 0.8403, 0.8458, and 0.7955 for M01–M04, respectively, with a macro average of 0.8183.

The human-validation sample contains 300 responses drawn from three tested-model families that are distinct from the attribution-judge families: Gemini 3.1 Pro Preview, DeepSeek V4.1 Flash, and Qwen3.5-397B. We sampled 100 responses from each family, with an exact 50/50 split between BIRD and Spider. Within each model–benchmark pool, the quota was allocated proportionally over prompting-setting × source-error-category strata by the largest-remainder method, followed by simple random sampling within each stratum using seed 20261011. This yielded 182 Base, 88 Implicit, and 30 Explicit responses across the eight represented error categories; E3.2 remained excluded for the context-recovery reason given above.

<table><tr><td>Mechanism</td><td>Cohen&#x27;s κ</td><td>Raw agreement</td></tr><tr><td>M01</td><td>0.8862</td><td>96.33%</td></tr><tr><td>M02</td><td>1.0000</td><td>100.00%</td></tr><tr><td>M03</td><td>0.9800</td><td>99.67%</td></tr><tr><td>M04</td><td>0.9074</td><td>99.67%</td></tr><tr><td>Macro average</td><td>0.9434</td><td>98.92%</td></tr></table>

Table 12: Agreement between the two independent human annotators on the 300-example mechanismattribution sample.

The two human annotators label the same 300 responses independently and cannot see one another’s decisions. The interface hides the attribution judges’ rationales, but it preselects a verdict when GPT-6.1-Sol and Claude Opus 5.5 agree; judge-disagreement cells remain blank and require a human choice. The human results are therefore an assisted validation rather than a fully AI-blinded annotation study. Both annotators completed every item, and their sample IDs match exactly. Following the binary rule used in the AI-judge analysis, we map supported to 1 and both not\_supported and uncertain to 0.

The agreement statistics address three nested reproducibility questions. First, the per-mechanism κ values in Table 12 test whether each binary label is stable across humans; their macro average is .9434, accompanied by 98.92% macro-average raw agreement. Second, pooling the four binary decisions over all items summarizes overall decision-level stability $( \kappa = 0 . 9 6 9 5 )$ . Third, treating each response’s complete four-label vector as one nominal outcome tests whether the entire multi-label configuration is reproduced $( \kappa = 0 . 9 1 0 1 )$ .

Human–AI agreement addresses a different question: whether the automatic attributions align with human judgments. Tables 13 and 14 report this comparison. Their final rows first average the two human-specific κ values within each mechanism and then take an unweighted macro average over M01–M04, yielding .9212 for GPT-6.1-Sol and .8748 for Claude Opus 5.5.

<table><tr><td>Comparison</td><td>M01</td><td>M02</td><td>M03 M04</td><td>Macro</td></tr><tr><td>Human 1-GPT</td><td>0.8656</td><td>1.0000 0.9579</td><td>0.8299</td><td>0.9133</td></tr><tr><td>Human 2–GPT</td><td>0.8705</td><td>1.0000 0.9379</td><td>0.9074</td><td>0.9290</td></tr><tr><td>Mean over humans</td><td>0.8681</td><td>1.0000 0.9479</td><td>0.8687</td><td>0.9212</td></tr></table>

Table 13: Cohen’s κ between each human annotator and GPT 6.1 Sol.

<table><tr><td>Comparison</td><td>M01</td><td>M02</td><td>M03</td><td>M04</td><td>Macro</td></tr><tr><td>Human 1–Claude</td><td>0.8984</td><td>0.8872</td><td>0.8890</td><td>0.7967</td><td>0.8678</td></tr><tr><td>Human 2–Claude</td><td>0.8832</td><td>0.8872</td><td>0.8693</td><td>0.8872</td><td>0.8817</td></tr><tr><td>Mean over humans</td><td>0.8908</td><td>0.8872</td><td>0.8792</td><td>0.8420</td><td>0.8748</td></tr></table>

Table 14: Cohen’s κ between each human annotator and Claude Opus 5.5.

## C.4 Consensus Coverage

The four mechanisms are high-precision responselevel attributions rather than an exhaustive partition of failure behavior. Of the 8,000 sampled failures, 1,305 (16.31%) receive no strict-consensus M01–M04 label. This total should not be read as 1,305 cases with no cause. In 591 cases, at least one attribution judge supports a mechanism but the two judges share no positive label: Claude assigns no label while GPT assigns at least one in 222 cases, GPT assigns no label while Claude assigns at least one in 318 cases, and both assign positive but disjoint label sets in 51 cases. The remaining 714 cases (8.93%) are jointly assigned no covered mechanism by either judge.

Among these 714 jointly unlabeled cases, 687 (96.22%) have the validated behavioral profile SQL Behavior= 0, Detection= 0, and Type Identification= 0: the model produces complete SQL without explicitly recognizing the premise flaw or identifying its intended type. This profile establishes a recognition failure, but it does not by itself identify how the response transformed or misused the flawed premise. For example, a response may leave an unspecified condition broad, preserve contradictory predicates literally, or follow a context-grounded reading without injecting a concrete assumption, reinterpreting the contradiction, misusing bounded context, or deleting a complete explicit constraint. We therefore treat Detection= 0 as a behavioral outcome rather than relabeling it as M01–M04. Accordingly, an empty strict-consensus mechanism set means no observable covered mechanism, not an error-free response and not an absence of a behavioral failure.

## C.5 Prompt-Type, Benchmark-Type, and Error-Type Analyses

Prompt and benchmark breakdown. The prompt-level reversal between M01 and M03 occurs in both source benchmarks. From Base to Explicit prompting, M01 declines from 72.10% to 25.30% on BIRD and from 88.99% to 39.45% on Spider; over the same contrast, M03 rises from 1.94% to 64.99% and from 1.03% to 48.14%, respectively. The Implicit setting lies between these endpoints in both datasets. M02 and M04 remain rare across prompting settings relative to assumption injection and context/evidence misuse. Because this analysis is conditioned on eligible failures, these percentages describe the composition of observed failures rather than the unconditional probability that a prompt produces an error.

For the prompt-association tests, each of the 8,000 sampled responses is one observation. We construct a separate 3 × 2 contingency table for each mechanism, crossing prompting setting (Base, Implicit, Explicit) with the mechanism’s strictconsensus presence or absence, and apply Pearson’s $\chi ^ { 2 }$ test with Cramér’s V as the effect size. Because mechanisms are non-exclusive, a response may be positive in more than one table. We adjust the four mechanism-specific p-values as one family using Holm’s method; the reported p < .001 results for M01 and M03 remain below .001 after adjustment.

Mechanisms by source error category. Table 15 reports strict-consensus incidence within each represented source error category in the analyzed sample. M02 is sharply concentrated in internal textual contradictions, where it appears in 50.69% of sampled failures. M04 is rare overall and is most frequent for question–schema contradictions (4.51%). M01 dominates semantic gaps and both beyond-ability categories, whereas M03 is most common for ambiguity, unclear intent, and missing constraints. Thus, the source taxonomy and response mechanisms are related but not interchangeable: their mapping is many-to-many, because one source category can yield multiple mechanisms and the same mechanism can recur across categories.

## D Representative Premise Error Examples

Table 17 presents representative instances across the PremiseSQL taxonomy, illustrating how different premise failures manifest in natural-language queries.

<table><tr><td>Code Source error category</td><td>N M01 M02 M03 M04</td></tr><tr><td>E1.1 Ambiguity</td><td>1,99970.14 0.0014.460.35</td></tr><tr><td>E1.2 Unclear Intent</td><td>1,926 83.54 0.00 12.51 0.47</td></tr><tr><td>E2.1 Internal Text</td><td>217 1.84 50.69 6.45 3.23</td></tr><tr><td>Contradiction E2.2 Question-Schema</td><td>35576.34 0.00 6.484.51</td></tr><tr><td>Contradiction</td><td></td></tr><tr><td>E3.1 Missing Constraint</td><td>1,228 38.60 0.0012.460.57</td></tr><tr><td>E3.3 Semantic Gap</td><td>926 89.20 0.00 2.70 0.00 732 90.03 0.00 9.97 2.32</td></tr><tr><td>E4.1 Inexpressible in SQL E4.2 External Knowledge</td><td>617 91.25 0.00 6.81 0.97</td></tr><tr><td>Required</td><td></td></tr></table>

Table 15: Observed strict-consensus mechanism incidence (%) by source error category in the analyzed sample. E3.2 is excluded because its original reducedschema exposure is unavailable.

Distinguishing Neighboring Categories Several neighboring categories are distinguished by the source of indeterminacy. E1.1 applies when a schema-level condition admits multiple supported mappings or aggregations, whereas E1.2 leaves the overall SQL operation unspecified; E3.1 instead omits a task-defining constraint while pre serving that operation. Thus, “sales” in the E1.1 example admits several computations over existing columns, whereas “impactful reviews” in E3.3 lacks a unique rule for mapping the concept to the available indicators. E2.2 and E3.2 may both leave required schema support unavailable to the model, but they arise from different perturbations: E2.2 modifies the query against an unchanged schema, whereas E3.2 preserves the original valid query while withholding part of its required schema context. E3.3 differs from both because related fields remain available but an abstract concept lacks a unique semantic mapping. Finally, E4.1 concerns operations such as forecasting that cannot be performed by standard SQL over the stored data, while E4.2 concerns external facts that are absent from the database.

## E Behavioral Case Studies

Figures 9–20 present representative multi-model case studies on flawed queries, while Figures 21– 29 present original-query controls under Base, Implicit, and Explicit prompting. Each page reports one response together with its schema evidence, SQL behavior, evaluation labels, and analysis.

![](images/38df6a6836059b94baf1ce7a089e16946281955faf8d820a6d7314395ca2ce8e.jpg)  
Figure 5: Representative prompt for ambiguity-type instance generation.

![](images/cd32c8640c3036c7c85418c46927155f8e62ed6742fd452608ec941bb31eba3f.jpg)  
Figure 6: Representative prompt for ambiguity-type instance review.

![](images/31672b88b160ca866771565d3db588d8b9e3c01d17476b4244106924d8c55f96.jpg)  
Figure 7: Representative prompt for ambiguity-type automatic evaluation.

<table><tr><td>ID</td><td>Mechanism</td><td>supported when</td><td>Boundary and exclusion rules</td></tr><tr><td>M01</td><td>Assumption Injection</td><td>The response makes a material, answer- Ordinary literal grounding, declared foreign- changing commitment not established by the key use, and surface value normalization are</td><td>current question or model-visible context, not M01. Nonappearance in bounded rows such as an invented value, threshold, met- alone is assessed as M03. Merely execut- ric, proxy, entity, schema role, relationship, ing SQL, leaving an underspecified request or semantic mapping. A concrete response broad, or making an arithmetic/grouping er- commitment and a material alternative or di- ror does not establish M01.</td></tr><tr><td>M02</td><td>terpretation</td><td>rect counterevidence must be identifiable. AND to OR or application to separate rows). der M01 or left unlabeled. Both original conditions and the transformed</td><td>Contradiction Rein- The question contains explicit incompatible Preserving the contradiction literally is not requirements on the same target and the re- M02. Removing one contradictory branch sponse makes them executable by chang- without replacement is M04. A change of ing their logical relation, entity, time, scope, field or scope in a question without an ex- quantifier, population, or granularity (e.g., plicit internal contradiction is considered un-</td></tr><tr><td>M03</td><td>Context/Evidence Misuse</td><td>satisfaction relation must remain observable. The response makes an incorrect availabil- Unknown data coverage is uncertain, not ity or answerability claim by either treat- supported. A complete reference schema ing bounded example rows as exhaustive is insufficient unless exposure to the tested database contents or denying a table, col- model is verified. Unsupported mappings umn, or declared relation that verified model- used to construct SQL are M01; M03 is re- visible schema context contains. The claim served for misuse of supplied context to infer</td><td>and contextual counterevidence are both re- absence, impossibility, or unanswerability.</td></tr><tr><td>M04</td><td>Omission</td><td>quired. Explicit Constraint A semantically complete, explicit atomic re- Incomplete referents, missing values or quirement in the current question disappears thresholds, and vague evaluative terms are from a response that still implements part of not eligible targets. Complete abstention is the request, and no value, predicate, proxy, not omission from a remaining implementa- field, or other operation replaces that require- tion. An attempted but incorrect implemen- ment.</td><td>tation blocks M04; a concrete substitute is M01, and a changed contradiction relation is M02. Requirements appearing only in the paired valid item cannot be imported into the</td></tr></table>

Table 16: Final decision rubric for failure-mechanism attribution. Each mechanism is evaluated independently, subject to the same-target exclusivity rules described in the text.

![](images/f36af73c6c7dc533cdf8a3625f362612a2fca95ec79b4e79fa4e892b53220e28.jpg)  
Figure 8: Final prompt used for failure-mechanism attribution. The instance-specific fields listed in the final paragraph are appended as structured input.

<table><tr><td>Error Type</td><td>Question</td><td>Error Reason</td></tr><tr><td>Ambiguity</td><td>in 2018?</td><td>Which product had the highest sales The phrase &quot;highest sales&quot; admits multiple schema- supported interpretations, including the number of orders, total quantity sold, gross revenue, and discounted revenue. Because these alternatives can produce different product rankings, the query does not determine a unique SQL formu- lation.</td></tr><tr><td>Unclear Intent</td><td>at 2:06:34 AM, what is the situation operation is therefore indeterminate. with their followers?</td><td>For the lists created by the user The phrase “what is the situation with their followers&quot; does who rated the movie “Pavee Lack- not specify whether to retrieve follower counts, aggregate een: The Traveller Girl&quot; on 3/27/2011 them, compare them, or rank the lists. The intended SQL</td></tr><tr><td>tion</td><td>Internal Text Contradic- List the movies that were released in The question contains mutually contradictory constraints 2000.</td><td>1945 and also released after the year that cannot be simultaneously satisfied under the schema. A single movie record has only one release date, making it impossible for a movie to be released in both 1945 and after 2000.</td></tr><tr><td>tradiction</td><td>Question-Schema Con- Who played the character named The schema supports identifying the actor through the that actor&#x27;s email address?</td><td>&quot;Chanice Kobolowski,&quot; and what is characters and actor tables, but the actor table contains no email-address column and no related table provides this attribute. The requested email address therefore cannot be retrieved from the database.</td></tr><tr><td>Missing Constraint</td><td>coins.</td><td>What&#x27;s the percentage of coins that The expression “that day&quot; has no specified antecedent and is higher than the price 1 hour ago leaves the required date unspecified. Because the historical in that day? List the names of these table contains observations for multiple dates, the relevant time point cannot be determined.</td></tr><tr><td></td><td>Missing Schema Premise List the powers of Hunter Zolomon.</td><td>We intentionally removed 30% of the gold schema before- hand. Therefore, the schema does not provide a way to identify Hunter Zolomon or to link a character entity to their associated powers. There is no table or relationship connect- ing characters to superpowers, making the query unanswer- able under the given schema.</td></tr><tr><td>Semantic Gap</td><td></td><td>For user ID 3, what is their yearly av- The concept of “impactful reviews&quot; has no explicit or im- erage number of impactful reviews? plicit semantic mapping in the schema, and no business defi- nition is provided. As a result, the metric cannot be grounded or computed from available tables or fields.</td></tr><tr><td>Inexpressible in SQL</td><td>ings?</td><td>Among the worst directors, which The database can identify the relevant directors and their genres are they most likely to im- historical genres, but determining which genres are most prove in over the next five years, likely to improve over the next five years requires predictive based on their past movies and rat- modeling beyond the filtering, aggregation, sorting, and join- ing operations supported by standard SQL.</td></tr><tr><td>quired</td><td>their IDs.</td><td>External Knowledge Re- Which actor has acted in at least The correctness of the query depends on external knowledge 2 films that were nominated for an not represented in the schema, as Oscar nomination informa- Academy Award (Oscars)? Please list tion is not stored in any table or field, making the condition uncomputable within the database.</td></tr></table>

Table 17: Examples of Expression Error, Premise Contradiction, Missing Premise, and Beyond Ability types.

## PREMISESQL CASE STUDY

## E1.1 Ambiguity: Flawed query

## USER INPUT

“What are all the titles?”

## SCHEMA EVIDENCE

Both Songs and Albums contain a Title column, so the requested entity is unresolved.

## MODEL OUTPUT

## Llama 4 Maverick

Response excerpt

“The question asks for all song titles, which can be retrieved from the Songs table.”

SQL behavior

SELECT Title FROM Songs;

## EVALUATION

Behavior: MISSED FLAW + SQL

Metrics: DET=0, TYPE=0, SQL=0

## ANALYSIS

The response silently resolves the ambiguous Title reference as Songs.Title. Because Albums.Title is equally compatible with the query, the generated SQL lacks grounding and the flaw remains undetected.

Figure 9: E1.1 Ambiguity, LLaMA 4 Maverick. The model silently selects Songs.Title, misses the unresolved reference, and generates SQL (DET=0, TYPE=0, SQL=0).

## PREMISESQL CASE STUDY

## E1.1 Ambiguity: Flawed query

## USER INPUT

“What are all the titles?”

## SCHEMA EVIDENCE

Both Songs and Albums contain a Title column, so the requested entity is unresolved.

## MODEL OUTPUT

## Claude Opus 5.5

Response excerpt

“The question is ambiguous because both Songs and Albums contain a Title column.”

SQL behavior

SELECT Title FROM Songs UNION SELECT Title FROM Albums;

## EVALUATION

Behavior: DETECTED FLAW + SQL

Metrics: DET=1, TYPE=1, SQL=0

## ANALYSIS

The model identifies that Title could refer to either table but combines both interpretations without user support. It detects the flaw, yet its UNION SQL invents a new resolution and still generates SQL.

Figure 10: E1.1 Ambiguity, Claude Opus 5.5. The model recognizes the ambiguity but combines both interpretations without user support (DET=1, TYPE=1, SQL=0).

![](images/5c26655764e3378c4de42daa858aba9f1f4e2f5ff4397bcdcd2886c36d5f2b61.jpg)  
Figure 11: E1.1 Ambiguity, DeepSeek V3.2. The model identifies the unresolved reference and abstains from generating unsupported SQL (DET=1, TYPE=1, SQL=2).

![](images/56eee94f51bfa8ae0e0c7e3c0089c56babb9f7b0405f7b62f34a41035fc4d3a2.jpg)  
Figure 12: E2.1 Internal Text Contradiction, GPT 6.1 Sol. The model treats mutually exclusive winner and loser conditions as ordinary predicates and misses the flaw (DET=0, TYPE=0, SQL=0).

![](images/ed543318c2797ffab7c9050ab0027c780c2079231aa89f93810820393bffa182.jpg)  
Figure 13: E2.1 Internal Text Contradiction, Claude Sonnet 4.6. The model states the contradiction but still encodes the incompatible requirements in SQL (DET=1, TYPE=1, SQL=0).

![](images/22ec8fef57f85b8c49b3c1c4d151287ed7592ab6b2d75b36652261c4dabbe16d.jpg)  
Figure 14: E2.1 Internal Text Contradiction, Claude Opus 5.5. The model recognizes the incompatible requirements and abstains (DET=1, TYPE=1, SQL=2).

![](images/9d3912ba7b8c8594bdf8e6c3c612c0bae6aa986619e71b600d559e5bb0a2f649.jpg)  
Figure 15: E3.3 Semantic Gap, Claude Sonnet 4.6. The model invents a keyword-based proxy for clinical seriousness and misses the flaw (DET=0, TYPE=0, SQL=0).

![](images/ab6f045f3b610d74db1e42844b33ac2cb85cf47f1cca970fad3700dd2f4b8e64.jpg)  
Figure 16: E3.3 Semantic Gap, GPT-5.4. The model acknowledges the missing severity attribute but substitutes a count of all allergies (DET=1, TYPE=1, SQL=0).

![](images/a9863ccbf0402ff9c8514aadf7f5aae4a629501ee9cb9a783821d8debce2de12.jpg)  
Figure 17: E3.3 Semantic Gap, ChatGLM-5.1. The model preserves the unsupported distinction and abstains (DET=1, TYPE=1, SQL=2).

![](images/ab47f6700d2c554ad347a876bd5975f60d4abf98c34ce5d5e6cb014d3e4879c3.jpg)  
Figure 18: E4.1 Inexpressible in SQL, LLaMA 4 Maverick. The model turns a forecasting request into a historical count and misses the flaw (DET=0, TYPE=0, SQL=0).

![](images/f9d233c2823a7d636f30af2bbfeab4b110a34c26cb4969a22d99e437c782e85a.jpg)  
Figure 19: E4.1 Inexpressible in SQL, DeepSeek V4.1 Flash. The model recognizes the forecasting limitation but returns a historical count as a proxy (DET=1, TYPE=1, SQL=0).

## PREMISESQL CASE STUDY

# E4.1 Inexpressible in SQL: Flawed query

## USER INPUT

“Based on the existing data, how many tweets are likely to be posted next Wednesday?”

## SCHEMA EVIDENCE

The table stores historical weekday labels but provides neither a forecasting model nor sufficient temporal information for predicting next Wednesday.

## MODEL OUTPUT

## ChatGLM 5.1

Response excerpt

“Historical weekday labels cannot determine how many tweets will appear next Wednesday.”

SQL behavior

No SQL generated.

## EVALUATION

Behavior: DETECTED FLAW + ABSTAIN

Metrics: DET=1, TYPE=1, SQL=2

## ANALYSIS

The model distinguishes a future prediction from a historical weekday lookup. Since the schema contains no forecasting basis, it detects the inexpressibility and correctly abstains from SQL generation.

Figure 20: E4.1 Inexpressible in SQL, ChatGLM-5.1. The model detects the unsupported prediction request and abstains (DET=1, TYPE=1, SQL=2).

![](images/53661af7bda94efa09f544695e6622490c34a7cf2a52f86e5dfd55692582e124.jpg)  
Figure 21: Original-query control under Base prompting, ChatGLM-5.1. The model constructs the valid schemasupported join and generates SQL (GEN=1).

![](images/022971f613f55e87e6821e589493dbf4c742f97a65056e03555a24bdf557e34e.jpg)  
Figure 22: Original-query control under Base prompting, Doubao Seed 2.0 Pro. The model notes that the supplied rows contain no matching city but correctly separates answer existence from query validity and generates SQL (GEN=1).

![](images/26c5b4a68f0c420cda55397bbc6bf4e404396facae68d5baa007b2ce50923cff.jpg)  
Figure 23: Original-query control under Base prompting, Claude Opus 5.5. The model treats the absence of a matching city in the supplied rows as grounds for withholding otherwise valid SQL (GEN=0).

![](images/d5623bb18e5b35fe39c3e29b2c2c22818856414924ea24799bdabf7fb889e026.jpg)  
Figure 24: Original-query control under Implicit prompting, Claude Opus 5.5. The model accepts the valid request and generates a workable query (VAL=1, GEN=1).

![](images/b25e71cfac31c79d1148b1817c27929799385348c8108a979c6eaea870326762.jpg)  
Figure 25: Original-query control under Implicit prompting, LLaMA 4 Maverick. The model incorrectly calls the request ambiguous but still generates SQL (VAL=0, GEN=1).

![](images/8a0d93cca4bc3c0bb7d4cdb53c8cdba03ae1cf693642e61470906febd3a728c5.jpg)  
Figure 26: Original-query control under Implicit prompting, DeepSeek V3.2. The model mistakes the absence of a dedicated first-name column for invalidity and abstains (VAL=0, GEN=0).

![](images/bd71eac45086ad94416cc30eca206c71c4c9d4f43706527c2fa11ba9f0850f5e.jpg)  
Figure 27: Original-query control under Explicit prompting, DeepSeek V3.2. The model rejects the prompt’s unsupported claim of a flaw and generates SQL (VAL=1, GEN=1).

![](images/2d01c1527f690f9f88a3b22ad54baf8f61eddc1e0274f6bcd4377e92c262643f.jpg)  
Figure 28: Original-query control under Explicit prompting, Doubao Seed 2.0 Pro. The model incorrectly judges the request invalid but nevertheless generates the correct filter (VAL=0, GEN=1).

![](images/7b33975f431586d582ce0d2a567a4d89756e4349645faa1fe2f35edf9a108dad.jpg)  
Figure 29: Original-query control under Explicit prompting, Qwen3.5-397B. The model accepts the false-invalid framing and abstains (VAL=0, GEN=0).