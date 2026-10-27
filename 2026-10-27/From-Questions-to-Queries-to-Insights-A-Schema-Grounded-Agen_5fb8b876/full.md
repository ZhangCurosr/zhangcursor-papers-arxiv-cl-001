# From Questions to Queries to Insights: A Schema-Grounded Agentic Workflow for Natural-Language Crash Data Analysis

Jingwen Zhu<sup>a</sup>, Keshu Wu<sup>a,∗</sup>, Steven T. Parker<sup>a</sup>, Bin Ran<sup>a</sup> and David A. Noyce<sup>a</sup>

<sup>a</sup>Department ofCivil and Environmental Engineering, University ofWisconsin–Madison, Madison, WI, United States

A R T I C L E I N F O

Keywords:   
trafic safety   
crash database   
large language models   
agentic workflows   
retrieval-augmented generation

## A BS T R AC T

Transportation agencies maintain extensive crash databases to identify safety priorities, evaluate countermeasures, and guide resource allocation. Making these data accessible to practitioners requires flexible ways to translate analytical questions into reliable results. Existing dashboards support predefined analyses, while additional requests often depend on specialists who understand database queries, coded fields, and agency conventions. This study presents CrashDB-QA, a fully local, schema-grounded agentic workflow that enables natural-language analysis of crash databases. Specialized components interpret questions, generate and validate readonly SQL, execute queries, and explain the results. A shared knowledge base grounds query construction in the agency’s data dictionary and safety-analysis conventions, while recorded intermediate outputs support inspection and auditing. An optional extension, CrashDB-QA+, checks the alignment among the question, query, and execution result and supplies evidence for iterative repair. We evaluate the workflow on a frozen snapshot of 4.4 million Wisconsin crash records using a new benchmark of 300 questions across ten analytical categories and nine local open-weight models. Domain grounding substantially improves query accuracy, and a code-specialized 7B model provides the best accuracy-latency balance among the tested models. Compositional analyses remain more dificult than routine retrieval, particularly comparisons between groups or periods. Verification improves performance when the Judge uses authoritative field definitions and the Refiner receives execution evidence, whereas additional checking alone ofers no consistent benefit. These findings demonstrate the potential of locally deployed, domain-grounded workflows to broaden access to crash analysis and identify the reasoning and verification capabilities needed for reliable use.

## 1. Introduction

Transportation agencies collect millions of crash records to support data-driven safety management. These records provide a foundation for comparing crash patterns across locations and periods, identifying high-risk locations, and examining the safety of vulnerable road users. Translating this evidence into actionable insights helps agencies prioritize engineering improvements, evaluate countermeasures, and allocate enforcement resources. Their importance is underscored by the persistent burden of road trafic deaths and serious injuries worldwide [1]. Crash databases provide the evidence needed to identify high-risk locations [2], evaluate countermeasures, and inform safety policies, but this evidence becomes useful only when practitioners can retrieve and interpret it. Dashboards and visualizationv platforms have improved access through maps, summaries, and predefined filters [3], while predictive models have extended analysis toward anticipating crash hotspots [4]. However, questions beyond these predefined functions often require someone to identify the relevant fields, interpret coded values, and write Structured Query Language (SQL). The resulting dependence on technical specialists limits how readily practitioners can explore the data and pursue follow-up questions.

Large language models (LLMs) ofer a promising way to bridge this gap by allowing practitioners to express analytical requests in natural language. Their emerging transportation applications include signal control [5], spatiotemporal forecasting [6], tool-augmented trafic analysis [7, 8], and connected-vehicle data interpretation [9], as reviewed by Nie et al. [10]. In trafic safety, LLMs have been explored for research assistance [11], crash-narrative analysis [12, 13], contributing-factor identification [14], severity inference [15, 16], crash-record modeling [17], risk assessment [18], and personal-information screening [19]. Studies with state crash data have also used LLMs to correct the coding of secondary crashes [20] and to link crash narratives with pavement-condition records [21]. A University Transportation

Center prototype has enabled natural-language queries and chart generation over a statewide crash repository [22]. Its report presents demonstrations rather than a systematic evaluation of answer accuracy. These applications demonstrate the potential of language-based interaction with safety information. Answering an analyst’s question directly from a structured crash database, however, requires a further capability: translating the request into a computation that follows the agency’s definitions and returns the intended result.

Text-to-SQL research provides a foundation for this capability [23]. Neural schema-aware models [24, 25, 26] and more recent LLM-based approaches [27, 28] have advanced natural-language querying, supported by benchmarks such as Spider and BIRD [29, 30] and methods for decomposition, schema linking, collaboration, and correction [31, 32, 33]. In crash analysis, however, matching words to columns is only part of the problem. Consider a request to identify counties with the greatest severity-weighted crash burden. A query that counts crashes may execute successfully and return a convincing ranking, yet omit the weighting that defines the requested measure [34, 35]. Likewise, a seasonal analysis may use the correct records but group them by month, and a two-period comparison may inadvertently apply the same condition to both groups. These executable-but-wrong queries expose the central challenge: successful execution does not establish that an answer is analytically correct. The system must connect the analyst’s intent to field meanings, coded values, and the rules used to construct the requested quantity.

This challenge links domain grounding, verification, and deployment. A governed crash database has an authoritative data dictionary, but abbreviated column names alone may not reveal its definitions or coding conventions. Query generation must therefore use documented knowledge rather than rely on the model’s prior assumptions. Verification must then examine whether the generated query and its returned result satisfy the request. Reasoning and tool use [36, 37], consistency across candidates [38], iterative refinement [39, 40], and model-based evaluation [41] ofer useful mechanisms, but their presence does not guarantee correctness. Self-correction without external feedback can be unreliable [42], and evaluation in trafic safety has also revealed diferences between predictive accuracy and expert agreement [43]. A database workflow consequently needs checks grounded in both authoritative definitions and execution evidence. At the same time, sensitive or restricted crash records may preclude externally hosted models, making local inference, read-only access, and an inspectable question-to-answer process practical requirements. Together, these needs motivate a workflow in which language-based reasoning is supported by domain knowledge and constrained by deterministic database controls.

We present CrashDB-QA, a fully local, schema-grounded agentic workflow designed around these requirements. Specialized components interpret the question, link it to the schema, generate SQL, validate and execute the query, and explain the result. A shared knowledge base combines the agency’s data dictionary, stored value formats, common synonyms, and safety-analysis conventions to support retrieval-grounded generation [44]. Deterministic controls enforce read-only access and schema validity, while recorded intermediate outputs make each answer traceable. The optional CrashDB-QA+ extension adds a Judge that checks the alignment among the question, query, and execution result, together with a Refiner that supplies feedback for revision. We evaluate the workflow on a frozen snapshot of 4.4 million Wisconsin crash records maintained through the Wisconsin Transportation Portal, developed by the University of Wisconsin–Madison Trafic Operations and Safety Laboratory in collaboration with the Wisconsin Department of Transportation. A new 300-question benchmark and nine local open-weight models allow us to examine accuracy, latency, compositional dificulty, and the efects of verification. The results show that domain grounding makes compact local models substantially more efective, while comparisons between groups or periods remain a persistent weakness. They also show that verification improves performance only under suitable designs: grounding the Judge and supplying execution evidence to the Refiner produces gains that additional checking alone does not.

This paper makes two main contributions. First, we introduce CrashDB-QA, an auditable workflow that integrates domain-grounded natural-language querying, deterministic database controls, and result explanation, with an optional execution-informed verification and repair loop. Second, we introduce an execution-based benchmark and evaluation comprising 300 questions across ten analytical categories in retrieval and reasoning tiers. The evaluation identifies accuracy–latency tradeofs among local models, distinguishes routine retrieval from dificult analytical composition, and establishes the benefits and limitations of grounding and iterative repair.

## 2. The CrashDB-QA Framework

## 2.1. Problem formulation

CrashDB-QA enables transportation safety analysts to obtain answers from an agency’s crash database through natural-language questions. Let  denote the database and  its column schema. An authoritative data dictionary

specifies the meaning and coding of each field. Given a question �, the system produces a SQL query �, its execution result �, and a plain-language explanation �:

$$
F : \ q \longmapsto ( Q , R , a ) , \qquad R = \operatorname { e x e c } ( Q , D ) .\tag{1}
$$

The workflow operates entirely on agency-controlled hardware and admits only queries that satisfy the database access constraints. We denote this admissible query set by : each query must be read-only, reference existing schema elements, and satisfy the implemented SQL validation checks.

We evaluate answer correctness by comparing the returned result with the result of an analyst-written reference query $Q ^ { \star }$ . This execution-based criterion accommodates diferent SQL formulations that produce equivalent answers. Let rows(�) denote the normalized multiset of result rows, with column names, column order, and specified formatting diferences removed before comparison. For $R ^ { \star } = \operatorname { e x e c } ( Q ^ { \star } , D )$ , execution match is

$$
\operatorname { E M } ( R , R ^ { \star } ) = \mathbf { 1 } \Bigl [ \operatorname { r o w s } ( R ) = \operatorname { r o w s } ( R ^ { \star } ) \Bigr ] .\tag{2}
$$

The design objective is therefore to maximize execution accuracy while maintaining admissible database access:

$$
\operatorname* { m a x } _ { F } \mathbb { E } _ { q } \Big [ \mathrm { E M } ( R _ { q } , R _ { q } ^ { \star } ) \Big ] \qquad \mathrm { s u b j e c t ~ t o } \qquad Q _ { q } \in \mathcal { V } \quad \mathrm { f o r ~ e v e r y ~ e x e c u t e d ~ q u e r y ~ } Q _ { q } .\tag{3}
$$

Local deployment is an additional architectural requirement: model inference, query execution, and result processing remain within the agency’s environment.

The central dificulty is that an admissible query can still compute the wrong quantity. Consider a request to identify the five counties with the greatest severity-weighted crash burden in a specified year. A query that ranks counties by COUNT(\*) may use valid columns and execute successfully, while omitting the severity weights that define the requested measure. We refer to such outputs as executable-but-wrong queries. Addressing them requires both accurate grounding before execution and an assessment of question–query–result alignment afterward. Reference results support evaluation, but are unavailable during deployment; operational verification must therefore estimate correctness from the available evidence.

## 2.2. Framework overview

CrashDB-QA organizes natural-language analysis as a sequence of specialized components with explicit inputs and outputs. Its six-component core comprises the Interpreter, Schema Linker, Generator, Validator, Executor, and Explainer. The Interpreter and Schema Linker establish the analytical intent and relevant fields; the Generator and Validator construct an admissible query; and the Executor and Explainer produce the result and its explanation. A shared knowledge base supplies authoritative field definitions and analysis conventions at the stages where they are needed. The optional extension, CrashDB-QA+, inserts a Judge and a Refiner between execution and explanation to assess the result and guide revision.

The database execution boundary provides the organizing structure in Figure 1. Steps 1–4 operate on the question, schema knowledge, and generated SQL before a query accesses crash records. Step 5 executes an admitted query, making its returned values and result structure available for subsequent assessment. In the core, this result proceeds directly to explanation. In CrashDB-QA+, Steps 6–7 examine the question, query, and execution evidence and return feedback to query construction when a check fails. Step 8 explains the selected result. This arrangement allows errors to be traced to interpretation, schema selection, query construction, execution, or explanation rather than remaining hidden within a single model response.

Language models perform the tasks that require language interpretation: intent extraction, SQL generation, result assessment, and explanation. Deterministic components perform schema retrieval, query validation, database execution, and feedback assembly. The execution order and retry limits of all components are controlled by code, with the Judge’s verdict determining whether semantic refinement continues. CrashDB-QA is thus an agentic workflow with predefined orchestration. This structure supports predictable database access and a recorded trace of intermediate outputs, while retaining model flexibility within individual tasks.

## 2.3. Governed-schema knowledge base

The knowledge base  connects the language used by analysts to the database’s documented meanings and stored representations. This connection is important for the crash-level table, whose 169 columns include abbreviated identifiers such as INJSVR and CONSZONE. Column names provide limited guidance about their interpretation, permissible values, or use in an analytical measure. Built before deployment and retrieved during analysis,  supplies this information through schema-focused retrieval-augmented generation [44]. It contains column-level knowledge and analytical conventions:

![](images/8ad6de17c25993b6fdf007b7dead0a4204de7bf7b2fd671aaed7171c5c835e08.jpg)  
Figure 1: CrashDB-QA architecture. The six-component core connects question interpretation, grounded query construction, controlled execution, and result explanation. The optional CrashDB-QA+ extension, shown in orange, adds verification and iterative repair.

$$
\begin{array} { r l } & { \qquad K = ( \boldsymbol { \kappa } _ { \mathrm { c o l } } , \boldsymbol { \kappa } _ { \mathrm { c o n v } } \big ) , } \\ & { \qquad \mathcal { K } _ { \mathrm { c o l } } = \big \{ ( c , D _ { c } , V _ { c } , W _ { c } ) : c \in S \big \} , } \\ & { \qquad \mathcal { K } _ { \mathrm { c o n v } } = \big \{ ( p , \sigma _ { p } ) : p \in \mathcal { P } \big \} . } \end{array}\tag{4}
$$

For each column �, $D _ { c }$ is its dictionary description, $V _ { c }$ records its stored values or value format, and $W _ { c }$ contains common words associated with it. Each analytical phrase $p \in { \mathcal { P } }$ is paired with a corresponding SQL pattern $\sigma _ { p } .$ For example, the severity entry includes

� = “Most severe injury to any person involved in the crash”,

$$
\begin{array} { r l } & { V _ { \mathrm { T N J S V R } } = \{ K , A , B , C , O \} , } \\ & { W _ { \mathrm { T N J S V R } } = \{ s e \nu e r i t y , s e \nu e r e , w o r s t , i n j u r y , f a t a l , \ldots \} . } \end{array}
$$

Table 1 lists further examples of both parts. The entries combine three sources. Dictionary descriptions cover all 169 columns. Stored codes and formats describe how those fields appear in the database; for low-cardinality categorical fields, read-only queries retrieve the observed values directly. This supplies exact representations such as 'MILWAUKEE' rather than an alternative spelling such as Milwaukee County. Manually curated word sets cover the 28 frequently queried columns, allowing terms such as drunk and OWI to retrieve ALCFLAG. The convention entries add operational definitions that field descriptions alone do not provide. They specify, for example, how to express a factor that is not marked present, how to distinguish fatal crashes from fatality counts, and how to calculate a crash share. For this study, severity-based group rankings use the following documented convention:

$$
\mathrm { S e v } ( g ) = \sum _ { i \in g } w ( s _ { i } ) , \qquad w ( K ) = 5 , \quad w ( A ) = 4 , \quad w ( B ) = 3 , \quad w ( C ) = 2 , \quad w ( O ) = 1 ,\tag{5}
$$

Examples of column knowledge and analytical conventions in .
<table><tr><td colspan="4">(a) Column entries  $\kappa _ { \mathrm { c o l } }$ </td></tr><tr><td>Column c</td><td>Description  $D _ { c }$ </td><td>Stored values  $V _ { c }$ </td><td>Associated words  $W _ { c }$ </td></tr><tr><td>CNTYNAME</td><td>County of the crash</td><td>72 names, e.g. &#x27;MILWAUKEE&#x27;</td><td>county, counties</td></tr><tr><td>ALCFLAG</td><td>Police suspected alcohol use</td><td>&#x27;Y&#x27;, &#x27;UNKN&#x27;, or NULL</td><td>alcohol, drunk, OWI</td></tr><tr><td>CYCLFLAG</td><td>Motorcycle involvement</td><td> $" \Upsilon "$  or NULL</td><td>motorcycle, motorbike</td></tr><tr><td>TOTFATL</td><td>Number of fatalities</td><td>Integer</td><td>fatalities, deaths, killed</td></tr><tr><td colspan="4">(b) Conventions  $\kappa _ { \mathrm { c o n v } }$ </td></tr><tr><td>Phrase  $p$ </td><td>Operational interpretation</td><td>SQL pattern  $\sigma _ { p }$ </td><td></td></tr><tr><td>without a factor</td><td>Factor not marked present</td><td>flag IS NULL OR flag &lt;&gt; &#x27;Y&#x27;</td><td></td></tr><tr><td>fatal</td><td>Crash involving at least one death</td><td>TOTFATL &gt; 0</td><td></td></tr><tr><td>share of crashes</td><td>Percentage meeting a condition</td><td>100 * AVG(CASE WHEN condition THEN 1 ELSE O END) SUM(CASE INJSVR WHEN &#x27;K&#x27; THEN 5 ... END)</td><td></td></tr></table>

where $s _ { i }$ is the KABCO severity code of crash � in group �. This score combines crash frequency and severity; it is an explicit analysis convention used by the workflow rather than a meaning inferred from an abbreviated column name.

Knowledge is retrieved according to each component’s task. The Schema Linker uses descriptions and associated words to identify relevant columns. The Generator receives the selected column entries together with the analysis conventions, limiting the schema context included in its prompt. A deterministic normalization step subsequently aligns county literals with suficiently similar stored spellings and leaves unmatched names unchanged. In the grounded CrashDB-QA+ design, the Judge and evidence-carrying Refiner receive entries for both the columns used in a query and related alternatives. For example, a motorcycle query that incorrectly uses the bicycle flag BIKEFLAG can be assessed alongside the definition of CYCLFLAG. The experiments separately examine the efects of grounding generation and verification.

## 2.4. Question understanding

Question understanding converts the analyst’s request into an explicit analytical specification and retrieves the schema elements needed to express it. Keeping these outputs separate makes interpretation and field selection inspectable before SQL generation. Table 2 summarizes the roles, inputs, and outputs of the four language-model agents used in Steps 1, 3, 6, and 8.

Step 1, the Interpreter. The Interpreter extracts five elements from the question:

$$
\mathrm { I n t } : q \longmapsto z = \bigl ( \mathrm { m e t r i c , ~ f l t e r s , ~ g r o u p , t i m e , r a n k } \bigr ) .\tag{6}
$$

These fields specify what to measure, which records to include, how to group them, the relevant period, and any requested ordering or limit. The output is a compact JSON object containing no SQL. For the severity-ranking example, it identifies a severity-weighted metric, county-level grouping, the specified year, and a top-five ranking. The structured intent supports schema retrieval and subsequent assessment, while the original question remains authoritative throughout the workflow.

Step 2, the Schema Linker. The Schema Linker retrieves columns using lexical matches against their documented descriptions, associated words, and names. Let � contain the terms extracted from the question and intent, and let �(⋅) tokenize text. Each column receives the score

$$
\operatorname { s c o r e } ( c ) = \left| T \cap \left( \tau ( D _ { c } ) \cup W _ { c } \right) \right| + \lambda \left| T \cap \tau ( \operatorname { n a m e } ( c ) ) \right| .\tag{7}
$$

The first term measures semantic cues available in the knowledge base, while the second weights direct column-name matches by �. The top � columns form the linked set $S _ { L }$ . A supplementary rule includes INJSVR whenever severityrelated language is detected, ensuring that the severity score in Eq. 5 has its required input. Thus, the running example retrieves the county and severity fields together with the temporal fields needed to restrict the analysis period. Retrieval uses deterministic scoring and rules.

Table 2  
Roles and prompts of the four language-model agents. Role instructions are quoted in abridged form.  
![](images/70c9858410f10dda3669a4e9d13c7f68dea475b0b5bdb623ae38dca55850a27b.jpg)

## 2.5. Query generation and execution

The construction phase translates the question into SQL and establishes whether the query may access the database. Generation uses the retrieved domain knowledge; validation and execution apply deterministic controls independently of the model’s instructions.

Step 3, the Generator. The Generator receives the question, structured intent, linked column entries, analysis conventions, and any feedback from a previous attempt:

$$
\mathrm { G e n } : ( q , z , S _ { L } , K , \varphi ) \longmapsto \{ Q _ { 1 } , \dots , Q _ { k } \} .\tag{8}
$$

Here,  denotes the relevant knowledge supplied to the prompt, and � is empty on the initial attempt. Prompt blocks follow a fixed order: table information, linked column entries, conventions and value formats, Oracle SQL rules, ten worked examples, the question, the intent, and retry feedback. The question is explicitly designated as authoritative, with the intent serving as supporting guidance. The first candidate uses greedy decoding; additional candidates, when enabled, are sampled and deduplicated. County literals are then normalized using the stored values described in Section 2.3.

In a multi-candidate configuration, candidates that pass validation and execute successfully are grouped by their normalized results. The selected query belongs to the largest agreement group [38]:

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i } \sum _ { j } \mathbf { 1 } \bigl [ \operatorname { r o w s } ( R _ { i } ) = \operatorname { r o w s } ( R _ { j } ) \bigr ] , \qquad \hat { Q } = Q _ { i ^ { * } } .\tag{9}
$$

Result-based voting allows equivalent SQL formulations to support the same answer. Agreement is a selection criterion rather than a correctness guarantee, because multiple candidates can reproduce the same analytical error. With one candidate, the workflow uses that candidate if it passes validation and execution.

Step 4, the Validator. Every candidate is checked before database execution:

$$
\begin{array} { r } { \mathrm { V a l } ( Q ) = \mathbf { 1 } \big [ Q \mathrm { ~ i s ~ a ~ s i n g l e ~ r e a d \mathrm { - o n l y ~ s t a t e m e n t } } \big ] } \\ { \wedge \mathbf { 1 } \big [ \mathrm { c o l s } ( Q ) \subseteq S \big ] \qquad } \\ { \wedge \mathbf { 1 } \big [ Q \mathrm { ~ p a r s e s ~ i n ~ t h e ~ O r a c l e ~ d i a l e c t } \big ] . } \end{array}\tag{10}
$$

The first check admits a single statement beginning with SELECT or WITH and rejects prohibited write operations. The second checks referenced identifiers against the schema while accounting for SQL syntax and quoted values. The third

checks dialect compatibility. These controls reject malformed or inadmissible queries; they do not determine whether the query expresses the intended analysis. If no candidate passes, validation errors become feedback � for another generation attempt, subject to the configured attempt limit �.

Step 5, the Executor. The Executor runs each admitted query within the agency’s database environment:

$$
R = \operatorname { e x e c } ( { \hat { Q } } , D ) .\tag{11}
$$

It repeats the read-only check immediately before database access and applies limits on execution time and returned rows. Execution errors are recorded and returned as feedback for revision. Connection and query times are logged separately to support latency analysis. Model inference and result processing remain local, so crash records are not transmitted to an external model service. In the core workflow, the selected result proceeds directly to Step 8; CrashDB-QA+ first applies the verification and refinement stages below.

## 2.6. Result verification and explanation

CrashDB-QA+ extends admissibility checks with an assessment of analytical alignment. Its Judge examines the question, generated query, and execution evidence, while its Refiner converts identified problems into feedback for another attempt. This loop estimates correctness without access to the reference answers used in evaluation.

Step 6, the Judge. The Judge returns a verdict � and a set of identified issues $\chi \colon$

$$
\mathrm { J u d g e } : \left( q , z , \hat { \cal Q } , { \cal R } ; { \mathcal { K } } _ { J } \right) \longmapsto ( v , \chi ) ,\tag{12}
$$

where $\kappa _ { J }$ denotes the retrieved schema knowledge supplied in the grounded design and is omitted in the ungrounded design. The LLM rubric evaluates four dimensions: relevance, aggregation, filters, and completeness. It receives the question, intent, SQL, and a preview of the returned rows, and provides a suggested correction when it detects a mismatch. The question remains the basis for assessment: the Judge is instructed to avoid introducing requirements that the analyst did not request. Grounding adds definitions for used columns and related alternatives, helping the Judge distinguish fields whose abbreviated names may otherwise be confused.

Optional deterministic checks flag potentially suspicious result structures: an empty table, an entirely NULL column, an all-zero numeric column, a single-row breakdown or trend, or a seasonal breakdown containing more than four groups. When enabled, these checks act as vetoes:

$$
v = \mathbf { 1 } [ \mathrm { L L M \ r u b r i c = P A S S } ] \wedge \mathbf { 1 } [ \mathrm { n o \ e n a b l e d \ d e t e r m i n i s t i c \ c h e c k \ f i r e s } ] .\tag{13}
$$

These rules are heuristics rather than proofs of error, because valid analyses can also return empty, zero-valued, or single-row results. Their efects are therefore evaluated separately from those of the LLM Judge. In the running example, a raw-count ranking can pass the structural checks, while the rubric can reject it for omitting the requested severity weighting.

Step 7, the Refiner. When verification fails, the Refiner assembles feedback for the Generator:

$$
\mathrm { R e f } : \left( q , \hat { Q } , R , \chi ; { \mathcal { K } } \right) \longmapsto \varphi .\tag{14}
$$

The Refiner is deterministic and makes no additional model call. We compare two feedback designs:

$$
\varphi _ { \mathrm { c r i t } } = \bigl ( \hat { Q } , \chi \bigr ) ,
$$

$$
\varphi _ { \mathrm { e v i d } } = \big ( \hat { \varrho } , \chi , \mathrm { r e s u l t \ e v i d e n c e , c o l u m n \ f a c t s , r e p a i r \ p a t t e r n , t r i e d \ q u e r i e s } \big ) .\tag{15}
$$

The critique-only design forwards the failed query and the Judge’s issues and suggested correction. The evidencecarrying design additionally provides the result shape, initial rows, and warnings about repeated output values; relevant column definitions; a targeted SQL repair pattern; and previously attempted queries. Repair patterns address three recurring structures: seasonal bucketing, two-group comparisons, and breakdowns that collapse into an aggregate. The attempted-query history is included to discourage repetition. These additions allow the Generator to revise a query using both the critique and the evidence that motivated it.

Feedback returns the workflow to Step 3. Generation, validation, execution, and verification continue until a result passes or the iteration limit � is reached. If no result passes, the critique-only configuration selects the final attempt, while the evidence-carrying configuration selects the attempt with the fewest unresolved issues. The trace records that verification was not passed. The iteration limit bounds computation and latency; the experiments assess whether the resulting revisions repair errors or degrade previously correct answers.

![](images/a3f45ca891bcf488a746a610e4076723af622961f6d23f2e341b24d112fe6655.jpg)  
Figure 2: Example of execution-informed refinement. An admissible query initially ranks counties by crash count. The Judge identifies the mismatch with the requested severity-weighted measure, and refinement produces a revised query that passes verification on the second iteration. The dashed red path marks the refinement loop.

Step 8, the Explainer. The Explainer converts the selected result into a concise answer:

$$
\operatorname { E x p l } : ( q , R ) \longmapsto a .\tag{16}
$$

It receives the question and a preview of the result and produces two to four sentences stating the answer, reporting key values, and identifying a relevant limitation. Its instructions restrict numerical claims to values supplied in the result. Explanation is the final presentation stage: it communicates the selected output but does not independently establish query correctness.

Figure 2 illustrates the complete workflow using the severity-ranking example. The initial query ranks counties by crash count and passes validation because it is admissible SQL. The Judge identifies the missing severity weighting, and the Refiner supplies feedback for revision. The Generator then constructs the weighted aggregation in Eq. 5, after which the revised result passes verification and proceeds to explanation. The trace demonstrates the distinct roles of validation, semantic assessment, and evidence-informed repair.

## 3. The CrashDB-QA Benchmark

The CrashDB-QA benchmark evaluates whether a natural-language analysis system can translate an analyst’s request into the intended result under a documented crash-data schema. Cross-domain benchmarks such as Spider and BIRD assess performance across diverse databases [29, 30]. Our benchmark complements this perspective by examining analytical depth within one governed database: whether a system correctly interprets coded fields, applies domain conventions, and combines operations required for safety analysis. Each question is paired with an analystreviewed reference query and its execution result on a frozen database snapshot. The benchmark therefore evaluates the complete path from a natural-language request to a computed answer, with particular attention to executable-but-wrong queries.

## 3.1. Analytical taxonomy

The benchmark organizes questions into ten categories according to their primary analytical operation, as summarized in Table 3 and Figure 3. Categories C1–C5 form the retrieval tier, covering counts, grouped summaries, rankings, temporal trends, and location-based summaries. These tasks primarily apply standard filtering and aggregation to existing fields. Categories C6–C10 form the reasoning tier, covering seasonal grouping, interacting conditions, severity-weighted measures, explicit comparisons, and normalized quantities. These tasks place additional demands on the composition of predicates, transformations, or derived measures. The tier distinction describes the analytical structure of the questions; retrieval tasks can still require multiple SQL operations, and both tiers depend on correct domain interpretation. This organization follows the broader distinction between recognition and reasoning used in the CrashSight video benchmark [45], adapted here to structured database analysis.

Table 3 Representative questions across the ten benchmark categories.
<table><tr><td>Code</td><td>Category</td><td>Representative question</td></tr><tr><td colspan="3">Tier 1: retrieval through standard filtering and aggregation</td></tr><tr><td>C1</td><td>Descriptive count</td><td>How many alcohol-related crashes occurred in Dane County in 2023?</td></tr><tr><td>C2</td><td>Aggregation/breakdown</td><td>How many crashes occurred in each county in 2023?</td></tr><tr><td>C3</td><td>Ranking / top-N</td><td>Which five counties had the most crashes in 2023?</td></tr><tr><td>C4</td><td>Temporal trend</td><td>How has the number of fatal crashes changed year over year?</td></tr><tr><td>C5</td><td>Spatial / location</td><td>How many crashes occurred at each intersection type in Dane County?</td></tr><tr><td colspan="3">Tier 2: reasoning through additional analytical composition</td></tr><tr><td>C6</td><td>Seasonality / cyclic</td><td>How are pedestrian crashes distributed across the four seasons?</td></tr><tr><td>C7</td><td>Multi-condition filter</td><td>How many crashes involved speeding but not alcohol in a work zone?</td></tr><tr><td>C8</td><td>Severity-weighted</td><td>Which ten counties had the worst crashes by injury severity in 2023?</td></tr><tr><td>C9</td><td>Comparison / contrast</td><td>Did fatal crashes in Dane County rise or fall from 2022 to 2023?</td></tr><tr><td>C10</td><td>Rate /normalization</td><td>What share of crashes in each county involved a distracted driver?</td></tr></table>

![](images/04b7e4b134edd61176a74345460805680c84d9929152a145805fb8d417b356f3.jpg)  
Figure 3: Benchmark taxonomy and representative executable-but-wrong errors. Categories C1–C5 emphasize standard retrieval and aggregation, while C6–C10 require additional composition of conditions, groupings, or derived quantities.

The categories support diagnosis beyond an overall accuracy score. A seasonal question, for example, requires months to be assigned to the intended seasonal groups; a comparison requires distinct conditions for the quantities being contrasted; and a crash share requires an appropriate numerator and denominator. Errors in these operations can leave the query executable while changing the meaning of its result. Category-level performance therefore helps identify recurring analytical weaknesses, although each score reflects the combined efects of interpretation, schema linking, and query construction rather than an isolated model capability. Figure 3 associates each category with a representative failure pattern.

## 3.2. Question construction and development separation

Questions combine operational requests with expert-authored cases. The operational requests originate from Community Maps inquiries routed to TOPS and WisDOT specialists and were anonymized and reformulated as benchmark questions. A domain expert supplied additional questions to cover analytical categories that occurred less frequently in those requests. Most questions specify a county and year, reflecting common safety-analysis needs while keeping reference results manageable for manual review. The final benchmark contains 300 questions, with 30 in each category. This balanced design gives every category equal weight in overall accuracy and supports direct comparison across analytical tasks; it does not estimate the frequency of those tasks in routine agency use.

Initial prompt wording, knowledge-base rules, and synonym mappings were developed using a separate set of questions excluded from benchmark scoring. The 300 benchmark questions and their reference results were then fixed for evaluation. Section 4 distinguishes the initial model comparisons and pre-registered verification configurations from subsequent diagnostic improvements motivated by observed failures. This distinction preserves the separation between development and initial evaluation while making clear where later analyses reuse the benchmark to investigate and address specific weaknesses.

## 3.3. Reference construction and quality control

Reference queries were constructed through a draft–review–audit process. A language model produced an initial SQL draft for each question, which two analysts independently reviewed and corrected. When their corrected queries returned diferent results, a third reviewer adjudicated the discrepancy. All references were executed against the frozen database snapshot before finalization. A random 15% were additionally checked against raw records, and a subset was compared with published WisDOT crash statistics. These checks assessed both the intended analytical operation and its consistency with the underlying data.

Execution testing was particularly important for detecting errors in stored-value assumptions. Nine draft references tested contributing-factor flags against ’N’, although that value was not present in the relevant fields. These queries were corrected using the documented coding conventions described in Section 2.3. The resulting references therefore combine expert interpretation with checks against actual database representations. Using the same frozen snapshot for every reference and candidate query also prevents changes in the underlying records from afecting comparisons.

## 3.4. Execution-based grading

Candidate answers are graded using execution match, as defined in Eq. 2. A candidate is correct when its normalized result matches the reference result, regardless of diferences in SQL formulation. This criterion accommodates equivalent implementations and directly tests the values returned to the analyst. Ranking references include the ranking measure alongside the selected locations or groups, allowing grading to compare both the selected entities and their associated values. Because the metric compares row multisets, it does not independently assess presentation order. The score also evaluates the computed result rather than the factual accuracy or clarity of the Explainer’s prose.

Two considerations bound this evaluation. First, diferent interpretations of a question can produce diferent defensible answers. The benchmark fixes the intended interpretation through documented conventions: for example, worst refers to the severity-weighted measure in Eq. 5, and last year refers to the last full calendar year. Scores therefore measure agreement with these operational definitions. Second, an incorrect query can coincidentally return the reference result on a single snapshot. Expert review and reference audits strengthen the ground truth, but cannot rule out such coincidences in candidate queries. Evaluation across multiple database instances or targeted test suites would provide stronger evidence of semantic equivalence [46]. The benchmark consequently measures execution agreement on a fixed real-world dataset, with category-level analysis used to explain the observed successes and failures.

## 4. Results and Discussion

The evaluation examines whether CrashDB-QA can provide accurate answers within practical local-computing constraints, where its remaining errors arise, and whether execution-informed verification can repair them. We first compare the grounded core across nine open-weight models, then analyze latency and failure patterns. Case studies illustrate how valid SQL can produce an incorrect analytical answer. Finally, controlled verification experiments distinguish the Judge’s ability to detect errors from the Refiner’s ability to correct them. The model comparison uses one greedy run per model; the verification study uses the strongest model identified in that comparison.

## 4.1. Experimental setup

All experiments run locally through Ollama against a frozen DT4000 snapshot containing 4.4 million crash records. Nine open-weight models span approximately 1B–8B parameters and include general-purpose, code-specialized, and reasoning-oriented variants. The general-purpose models are Gemma-3 1B and 4B [47], Llama-3.2 3B and Llama-3.1 8B [48], and Qwen-2.5 7B [49]. The code-specialized models are DeepSeek-Coder 6.7B [50] and Qwen-2.5-Coder 7B [51]. The reasoning-oriented models are Qwen-3.5 4B and DeepSeek-R1 8B [52]. Each model is evaluated on all 300 benchmark questions using the same prompt templates, grounding resources, reference answers, and database snapshot. Accuracy is measured by execution match under Eq. 2. Full traces record intermediate outputs, agent inference times, and database connection and execution times. Repeated runs of two models reproduced their category-level scores exactly under the same configuration; broader variation across seeds and runtime settings is outside the present evaluation.

Table 4  
Execution accuracy of the grounded CrashDB-QA core by model and analytical category (%). Each category contains 30 questions, for 300 questions in total. Inv. denotes the number of questions producing non-executable queries.
<table><tr><td rowspan=1 colspan=14>Tier 1: Retrieval                  Tier 2: ReasoningModel               Regime    C1  C2  C3  C4  C5  C6  C7  C8  C9 C10 Overall Inv.</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5-Coder 7B code</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>43.3</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>80.7</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5 7B         general</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>63.3</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>43.3</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>76.3</td><td rowspan=1 colspan=1>13</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-R1 8B     reasoning</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>33.3</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>37</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-Coder 6.7B code</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>56.7</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>26.7</td><td rowspan=1 colspan=1>63.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>68.3</td><td rowspan=1 colspan=1>24</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1 8B        general</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>43.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67.7</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>Gemma-3 4B         general</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=1>56.7</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>63.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>64.0</td><td rowspan=1 colspan=1>26</td></tr><tr><td rowspan=1 colspan=1>Llama-3.2 3B        general</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>43.0</td><td rowspan=1 colspan=1>37</td></tr><tr><td rowspan=1 colspan=1>Qwen-3.5 4B         reasoning</td><td rowspan=1 colspan=1>63.3</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>26.7</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>33.3</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>33.3</td><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>6.7</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>24.7</td><td rowspan=1 colspan=1>222</td></tr><tr><td rowspan=1 colspan=1>Gemma-3 1B         general</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.3</td><td rowspan=1 colspan=1>139</td></tr></table>

The Schema Linker uses � = 0.25 and retrieves the top � = 16 columns, supplemented by the rules described in Section 2.4. The Executor limits query time to 30 s and returned results to 5,000 rows. The Judge receives the first 10 result rows, and the Explainer receives the first 20. Single-candidate generation uses greedy decoding; additional candidates, when enabled, are sampled at temperature 0.4. Verification runs use at most � = 3 iterations. The initial model comparison evaluates the grounded core without semantic verification. Subsequent experiments first test the preregistered verification configurations and then examine diagnostic improvements motivated by their observed failures.

## 4.2. Accuracy across models

Table 4 reports overall and category-level execution accuracy, while Figures 4 and 5 summarize model and tier diferences. Qwen-2.5-Coder 7B achieves the highest overall accuracy, 80.7%, followed by Qwen-2.5 7B at 76.3% and DeepSeek-R1 8B at 73.3%. Performance varies substantially within similar parameter ranges, indicating that model size alone does not determine suitability for the workflow.

Model size and executable output. The results indicate a substantial capability gap between the smallest model and the stronger local models. Gemma-3 improves from 5.3% at 1B parameters to 64.0% at 4B, while the evaluated models in the approximately 7B–8B range achieve 67.7–80.7%, as Figure 7a shows. However, Qwen-3.5 4B reaches only 24.7%, illustrating that size does not ensure compatibility with the task and output contract. Non-executable output is a major contributor to poor performance: Gemma-3 1B produces 139 non-executable queries, and Qwen-3.5 4B produces 222. Grounding supplies domain information, but cannot compensate for a model that frequently fails to produce usable SQL under the evaluated configuration.

Retrieval and analytical composition. Across the seven models scoring at least 43% overall, reasoning-tier accuracy is approximately 13–25 percentage points below retrieval-tier accuracy. Qwen-2.5-Coder 7B, for example, achieves 87.3% on retrieval and 74.0% on reasoning. This pattern is consistent with the broader recognition–reasoning distinction observed in crash-video understanding [45], although the tasks and evaluation settings difer. The database results show that identifying relevant records and applying standard aggregations are more reliable than composing additional analytical operations. The gap also varies by category: seasonality and severity weighting can be handled accurately by the stronger models, while explicit comparisons remain dificult.

Comparisons as a persistent weakness. No model exceeds 43.3% on C9. The strongest code-specialized model and its general-purpose counterpart both reach this value, despite their diferent overall scores. The 30 comparison questions include two-period pivots, population contrasts, comparisons of shares, and severity-score contrasts. These tasks require separate quantities to be computed under distinct conditions and then combined into the requested comparison. Their consistently low scores identify this form of composition as a shared weakness across the evaluated configurations. As shown in Section 4.6, the verification loop also fails to repair these questions in the present study. Specialization and output discipline. The same-family comparison between Qwen-2.5-Coder 7B and Qwen-2.5 7B provides the clearest evidence of an advantage associated with code specialization. The code model improves overall accuracy by 4.4 percentage points and reduces non-executable queries from 13 to 3. Its largest category-level gain is in rates and normalization, although it does not improve every category. Across families, DeepSeek-Coder 6.7B achieves performance similar to Llama-3.1 8B, at 68.3% and 67.7%, respectively. The reasoning-oriented models do not improve the overall accuracy frontier: DeepSeek-R1 8B reaches 73.3%, and both reasoning-oriented models exhibit outputformat failures involving deliberative text. These comparisons support the practical value of code-oriented models in this workflow, while diferences in training data and model design prevent attributing all performance diferences to training regime alone.

![](images/4803f724d2da9f1303aa2581c110c7a2a20e5b63c2dd044a873f4c510866f6de.jpg)

![](images/7f0e4f0d6abc5f8fbfc378d1edf75e8fbf41098c9b0f55c8787ea1649f704273.jpg)

Figure 4: Performance of nine local models on the 300-question benchmark. (a) Question outcomes. (b) Accuracy on the retrieval and reasoning tiers, with overall accuracy shown at right.  
![](images/a06f1ad08cb8de56c56169a1a200d3a27bfa835dad65d888d205467eba42c82c.jpg)  
Figure 5: Category-level accuracy by model, ordered by overall performance. Each circle shows retrieval categories on the right and reasoning categories on the left; concentric rings indicate 25-percentage-point increments.

![](images/fb18dca8cb076963d00e723361e8bcd5b9df0e6b0e6611a3812f47194d2a727f.jpg)  
Figure 6: Stage-level latency across local models. Ribbon widths represent time attributed to pipeline stages before and after database execution.

## 4.3. Latency and cost

Accuracy determines whether an answer is useful, while latency determines whether the workflow supports interactive analysis. Figure 6 decomposes time by pipeline stage, and Figure 7 relates end-to-end latency to accuracy. Database execution time varies relatively little among models that reliably reach execution: median query time is 4.1– 4.8 s, with negligible connection overhead. Most diferences in end-to-end latency therefore arise from model inference and workflow behavior. The two lowest-performing models reach execution less consistently, so their database-time summaries should be interpreted alongside their high failure rates

Inference time and latency variability. Median model time is approximately 3–4 s for the stronger general-purpose and code-specialized models, compared with 11.3 s and 56.2 s for the two reasoning-oriented models. The latter also exhibit long latency tails, with 95th-percentile end-to-end times of approximately 475 s and 110 s, respectively, compared with roughly 36–40 s for the other models. The additional inference time does not produce a corresponding improvement in execution accuracy under this configuration.

Accuracy–latency tradeof. Qwen-2.5-Coder 7B provides the highest accuracy at a median end-to-end latency of 9.2 s. No evaluated model is both faster and more accurate. DeepSeek-Coder 6.7B is approximately 1.4 s faster but scores 12.4 percentage points lower, while the reasoning-oriented models require more time for lower accuracy. Thus, Qwen-2.5-Coder 7B provides a favorable operating point for the subsequent verification study. These timings characterize the evaluated local configuration; they are not hardware-independent performance estimates.

## 4.4. Error analysis

We inspected incorrect outputs to distinguish failures in analytical construction from failures in domain interpretation and executable output. Among the stronger models, compositional errors were the predominant remaining pattern. Queries often selected appropriate fields and records but combined them incorrectly: a missing GROUP BY collapsed a breakdown into a single value, an unnecessary row limit truncated a trend, incorrectly grouped AND/OR conditions changed the selected population, or a seasonal request was answered with monthly counts. Comparison queries also reused the same predicate for quantities that should have been computed separately. These failures explain why high schema familiarity does not translate directly into reliable analytical composition.

![](images/37165e84d1a3e9a32f5b7d9e15de5fb266ee838539a2cef4357649867dfab33f.jpg)  
(a)

![](images/e391af973ccb5f47bf33b4ea51ed80df4c17dca1573587838204548a755f4dd0.jpg)  
(b)  
Figure 7: Execution accuracy versus (a) parameter count and (b) median end-to-end latency on a logarithmic scale. Colors identify models, and marker shapes identify training regimes.

Two additional error classes concern domain grounding. Value-grounding errors use an incorrect stored representation, such as an unmatched county name or a flag comparison against ’N’. Such queries can execute and return an empty or zero-valued result, making the error dificult to distinguish from a legitimate absence of crashes. The knowledge base and county normalization address these failures. Column-selection errors use a valid but semantically inappropriate field, such as a commercial-vehicle or bicycle flag in a motorcycle analysis. These errors motivate supplying the Judge with definitions of related alternative columns, as examined in Section 4.6.

Some requests also depend on information unavailable in the crash table, such as an external population denominator for a population-normalized rate. Such cases require recognizing that the available data do not support the requested calculation. This difers from computing a share within the crash records, where both numerator and denominator can be obtained from the table. Finally, non-executable output remains a separate failure mode. It afects 139, 37, and 222 questions for Gemma-3 1B, DeepSeek-R1 8B, and Qwen-3.5 4B, respectively, compared with only 3 for Qwen-2.5-Coder 7B. For the latter model, most incorrect answers therefore arise after the requirement to produce executable SQL has already been met.

## 4.5. Case studies

Two outputs from Qwen-2.5-Coder 7B illustrate the distinction between correct domain-grounded composition and executable analytical error. The first correctly applies the benchmark’s severity convention. The second retrieves the appropriate records but uses the wrong temporal grouping, after which the explanation introduces additional inaccuracies. The examples include the executed SQL, result excerpts, and recorded Explainer outputs.

These cases show why admissibility, analytical correctness, and explanation quality require separate attention. The seasonal query passes validation because its syntax, fields, and access mode are acceptable; detecting the error requires relating its grouping operation and returned rows to the question. The explanation can then compound the problem by presenting unsupported or incorrect conclusions. Verification should therefore inspect the query together with execution evidence, while recognizing that a correct result alone does not guarantee a faithful explanation. The following experiments assess the query-repair component of this problem.

Correct: C8 severity-weighted county ranking   
Question. Which 10 counties had the worst crashes by injury severity in 2023?   
Model query (identical to the reference query):   
SELECT CNTYNAME,   
SUM(CASE INJSVR WHEN 'K' THEN 5 WHEN 'A' THEN 4   
WHEN 'B' THEN 3 WHEN 'C' THEN 2   
ELSE 1 END) AS severity\_score   
FROM CRASH\_TABLE   
WHERE CRSHYEAR = 2023   
GROUP BY CNTYNAME   
ORDER BY severity\_score DESC   
FETCH FIRST 10 ROWS ONLY   
Result excerpt. MILWAUKEE 33,577; DANE 12,116; WAUKESHA 9,996; . .. (10 rows).   
Recorded Explainer output. “The top 10 counties with the worst crashes by injury severity in 2023 [. . . ] are: Milwaukee (33,577), Dane   
(12,116), Waukesha (9,996), . . . Caveat: the data [. . . ] may not include all types ofinjuries or other relevantfactors afecting crash severity.”   
Interpretation. The query applies the specified KABCO weights, aggregates by county, restricts the year, and returns the ten highest-scoring   
counties. Its execution result matches the reference. The explanation reports the leading values, although its caveat is generic rather than   
specific to the frequency–severity interpretation of the score.

## Incorrect: C6 monthly counts in place of seasonal totals

Question. How many pedestrian crashes occurred in each season in Milwaukee County in 2023?

```sql
SELECT CASE WHEN EXTRACT(MONTH FROM CRSHDATE) IN (3,4,5) THEN 'spring'
WHEN EXTRACT(MONTH FROM CRSHDATE) IN (6,7,8) THEN 'summer
WHEN EXTRACT(MONTH FROM CRSHDATE) IN (9,10,11) THEN 'fall'
ELSE 'winter' END AS season,
COUNT(*) AS ped_crashes
FROM CRASH_TABLE
WHERE PEDFLAG = 'Y' AND CNTYNAME = 'MILWAUKEE' AND CRSHYEAR = 2023
GROUP BY CASE ... END the same season expression
ORDER BY ped_crashes DESC
```

Reference result. Summer 141; fall 141; spring 130; winter 123.

Model query (groups by calendar month):

```sql
SELECT EXTRACT(MONTH FROM CRSHDATE) AS crash_month,
COUNT(*) AS crash_count
FROM CRASH_TABLE
WHERE CNTYNAME = 'MILWAUKEE' AND PEDFLAG = 'Y' AND CRSHYEAR = 2023
GROUP BY EXTRACT(MONTH FROM CRSHDATE)
```

Interpretation. The query selects the correct population but returns monthly counts rather than the four requested seasonal totals. The explanation does not resolve this mismatch and adds errors: the stated total of 544 difers from the reference total of 535, and February is described as the minimum despite January having fewer crashes.

## 4.6. Verification loop

The verification study separates two requirements for useful refinement. RQ1 asks whether the Judge distinguishes correct from incorrect results. RQ2 asks whether the resulting feedback improves final execution accuracy without introducing excessive regressions. Both experiments use Qwen-2.5-Coder 7B. RQ1 replays alternative Judge designs over recorded core outputs with known execution-match labels. RQ2 runs the complete workflow on all 300 questions, first with the pre-registered configurations and then with diagnostic improvements. Paired accuracy diferences are assessed using exact McNemar tests. The analysis also uses 95% percentile bootstrap intervals and applies Holm– Bonferroni correction separately within the pre-registered comparison family and the subsequent improvement family. RQ1: Detecting incorrect answers. Table 5 and Figure 8 evaluate the Judge on the 297 executable core outputs: 242 correct and 55 incorrect. The original Judge accepts 225 correct outputs, giving a recall of 0.930, but also accepts 33 incorrect outputs, giving a false-accept rate of 0.600. Consequently, most executable errors do not trigger revision. The deterministic heuristics do not resolve this problem: their vetoes occur on outputs that can legitimately be empty or scalar, illustrating the limitations of using result shape as a proxy for correctness.

Table 5  
RQ1: Judge verdicts against execution-match labels for the same 297 executable core outputs. TP and FP are pass verdicts on correct and incorrect outputs; FN and TN are fail verdicts on those outputs. False-accept rate is $\mathsf { F P } / ( \mathsf { F P } + \mathsf { T N } )$ .
<table><tr><td>Judge design</td><td>TP</td><td>FP</td><td>FN</td><td>TN</td><td>Precision</td><td>Recall</td><td>False-accept</td></tr><tr><td>Original Judge</td><td>225</td><td>33</td><td>17</td><td>22</td><td>0.872</td><td>0.930</td><td>0.600</td></tr><tr><td>Dictionary-grounded Judge</td><td>228</td><td>31</td><td>14</td><td>24</td><td>0.880</td><td>0.942</td><td>0.564</td></tr><tr><td>Evidence-checklist variant</td><td>239</td><td>51</td><td>3</td><td>4</td><td>0.824</td><td>0.988</td><td>0.927</td></tr></table>

![](images/56b585460403f72fe4909f21d77dd8fcdb18636e270c6fb3fe3e2eba3f940d8b.jpg)

![](images/23c9c6069699eafa4f5017b4764853dfd9d721de19ef2ef334792199fd0d1b6d.jpg)  
Figure 8: RQ1: Comparison of Judge designs on 297 executable outputs. (a) Verdicts on the 55 incorrect and 242 correct outputs. (b) False-accept rate versus recall; lower false acceptance and higher recall are preferable.

We then ground the Judge in the same knowledge base used for generation, supplying definitions for columns used in the query and related alternatives. This change increases accepted correct outputs from 225 to 228 and reduces accepted incorrect outputs from 33 to 31, with negligible added latency in the recorded runs. One detected error involves a motorcycle question answered using BIKEFLAG, whose definition identifies bicycle involvement. The improvement is modest: the grounded Judge still accepts 56.4% of incorrect outputs. An alternative evidence-linked checklist performs worse, raising the false-accept rate to 0.927. This result shows that a more elaborate assessment prompt does not necessarily produce better discrimination; the experiment does not establish why the checklist encourages acceptance. RQ2: Pre-registered verification configurations. The original verification configurations do not significantly improve the core, as shown in Table 6 and Figure 9. Rule checks increase accuracy from 80.7% to 81.0%, repairing one answer without degrading another. The ungrounded LLM Judge produces two repairs and two regressions, leaving accuracy unchanged. Both comparisons have exact McNemar � = 1. Combining the ungrounded Judge, rule checks, and five-candidate voting reduces accuracy to 75.0%: six answers are repaired, but 23 previously correct answers become incorrect. This decline is significant, with exact $p = 0 . 0 0 2$ and Holm-adjusted $p = 0 . 0 0 7$ , and median latency increases to nearly four times that of the core.

The replay analysis and execution traces identify two limitations of the original loop. First, the Judge accepts many incorrect answers, preventing them from receiving corrective feedback. Second, the critique-only Refiner supplies the failed SQL and critique without the returned evidence, limiting the Generator’s ability to diagnose the previous result. Together, these observations motivate changes to error detection and feedback construction. The decline in the voting configuration also shows that additional candidates and verification steps can increase regressions rather than provide a monotonic improvement.

RQ2: Verification configurations compared with the CrashDB-QA core on 300 questions. All loop configurations use $N = 3$ and one greedy candidate unless stated otherwise. Repairs and regressions are counted relative to the core; $p$ denotes the unadjusted exact McNemar test.
<table><tr><td>Configuration</td><td>Judge</td><td>Feedback</td><td>Acc. (%) Repairs Regressions</td><td></td><td></td><td> $p$ </td></tr><tr><td>Core only</td><td></td><td></td><td>80.7</td><td></td><td></td><td></td></tr><tr><td colspan="7">Pre-registered configurations</td></tr><tr><td>Rule checks</td><td>rules only</td><td>critique</td><td>81.0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>LLM Judge</td><td>LLM</td><td>critique</td><td>80.7</td><td>2</td><td>2</td><td>1</td></tr><tr><td>Voting,  $k { = } 5 , + \ r { \mathsf { u l e s } }$ </td><td> $\mathsf { L L M } + \mathsf { r u l e s }$ </td><td>critique</td><td>75.0</td><td>6</td><td>23</td><td>0.002</td></tr><tr><td colspan="7">Diagnostic improvements: deterministic heuristics disabled</td></tr><tr><td>Grounded Judge</td><td> $\mathsf { L } \mathsf { L } \mathsf { M } + \kappa$ </td><td>critique</td><td>83.0</td><td>8</td><td>1</td><td>0.039</td></tr><tr><td>+ evidence Refiner</td><td> $\mathsf { L } \mathsf { L } \mathsf { M } + \kappa$ </td><td> $\mathsf { c r i t i q u e } + \mathsf { e v i d e n c e }$ </td><td>83.7</td><td>10</td><td>1</td><td>0.012</td></tr></table>

![](images/a8d4f9fba9da20d629fca08d2707e1eea005f48862e58b1ec4c49b327abaf5f6.jpg)

![](images/10ac12ae0d5fde452918a2ad1b0344710d34af952a9d5c6517d5c1fe05802eba.jpg)  
Figure 9: RQ2: Efects of verification and refinement. (a) Final execution accuracy, with the core shown as a dashed reference line. (b) Repairs and regressions relative to the core.

RQ2: Grounded verification and evidence-carrying refinement. The diagnostic study introduces the two changes sequentially while holding the remaining settings fixed: one greedy candidate, deterministic heuristics disabled, and $N = 3$ . Replacing the ungrounded Judge with the grounded design raises final accuracy to 83.0%, with eight repairs and one regression relative to the core. The exact McNemar test gives $p = 0 . 0 3 9$ . The repairs include two motorcycle queries redirected to CYCLFLAG from bicycle and commercial-vehicle flags. Adding the evidence-carrying Refiner raises accuracy further to 83.7%, with ten repairs and one regression relative to the core, giving $p = 0 . 0 1 2$ . Additional repairs include breakdown queries whose returned structure exposes the aggregation error. Both configurations remain significant against the core after Holm correction within this improvement family, with adjusted $p$ values of 0.039 and 0.023, respectively. These tests establish improvements over the core; they do not establish that the additional gain from 83.0% to 83.7% is itself significant.

Scope and implications. In the evidence-carrying configuration, 43 of the 300 questions undergo additional iterations, and median end-to-end latency increases from 9.2 s to 13.1 s. The ten repairs span breakdowns, rankings, trends, location summaries, seasonality, and severity weighting. No C9 comparison question is repaired, and the single regression occurs in that category. The gains therefore concern errors that the current Judge can identify and the Generator can revise from the supplied feedback; they do not resolve the benchmark’s most persistent compositional weakness. Moreover, the improved designs were developed after inspecting failures and evaluated on the same benchmark. Their gains provide diagnostic evidence for grounded, execution-informed repair, with independent confirmation across additional models, runs, and held-out questions remaining necessary.

Taken together, the experiments identify two distinct requirements for reliable local analysis. The core benefits from a model that combines usable SQL output with accurate analytical composition. Verification adds value when its decisions use documented field meanings and its feedback exposes the execution evidence needed for repair. Neither additional reasoning time nor a more elaborate loop consistently improves performance on its own.

## 5. Conclusion

This study presented CrashDB-QA, a fully local, schema-grounded agentic workflow that connects naturallanguage safety questions to database queries, execution results, and plain-language explanations. The workflow combines specialized language-model tasks with deterministic controls for schema validation and read-only execution, while retaining intermediate outputs for inspection. Its optional extension, CrashDB-QA+, uses documented field definitions and execution evidence to support verification and repair. We also introduced a 300-question benchmark spanning ten analytical categories, providing an execution-based assessment of both routine retrieval and more demanding analytical composition.

Evaluation across nine local open-weight models shows that practical performance depends on executable output, domain interpretation, and analytical composition. Qwen-2.5-Coder 7B achieved the highest core accuracy of 80.7% at a median latency of 9.2 s, providing a favorable accuracy–latency balance in the evaluated environment. Retrieval tasks were generally more reliable than reasoning tasks, and comparisons between groups or periods remained dificult across all models. Verification produced mixed results: the original configurations ofered no significant improvement, and the combination of voting and verification reduced accuracy. Grounding the Judge and supplying execution evidence to the Refiner increased accuracy to 83.7%, with ten repairs and one regression relative to the core, at a median latency of 13.1 s. These findings indicate that efective refinement depends on the quality of error detection and corrective evidence rather than the number of verification steps.

The broader contribution is an auditable approach to accessing structured safety information. By linking an analyst’s request to documented definitions, generated SQL, and returned values, CrashDB-QA makes the basis of an answer available for review. The benchmark further shows why reliability should be assessed by analytical task: strong performance on counts and summaries does not establish competence in comparisons or derived measures. This distinction can help agencies identify suitable uses and determine where expert review remains necessary. The same design principles may also support other transportation databases with documented schemas and local deployment requirements.

Several limitations define the scope of these findings. Evaluation used one statewide schema and a single frozen snapshot, and execution agreement cannot establish semantic equivalence across all possible database instances. The verification improvements were developed through failure analysis and tested on the same benchmark using one model, so independent confirmation is needed. The Judge still accepted many incorrect outputs, and the case studies showed that explanation errors can arise even after query execution. Future work will extend evaluation across heldout questions, database schemas, and runtime configurations; develop stronger methods for comparative analysis and unsupported-request detection; and assess explanation faithfulness. Multi-table analysis and field studies of analyst accuracy, review efort, and time savings will further establish the workflow’s practical value.

## Acknowledgments

The Community Maps Predictive Analytics Tool and WisTransPortal MV4000 Crash Database were developed through sponsorship and collaboration with the Wisconsin Department of Transportation.

## References

[1] World Health Organization, Global Status Report on Road Safety 2023, Technical Report, World Health Organization, Geneva, 2023.

[2] E. Hauer, D. W. Harwood, F. M. Council, M. S. Grifith, Estimating safety by the empirical Bayes method: A tutorial, Transportation Research Record 1784 (2002) 126–131.

[3] Wisconsin Trafic Operations and Safety Laboratory and Wisconsin Department of Transportation Bureau of Transportation Safety, Commu nity Maps: Statewide Crash Mapping and Predictive Analytics, https://transportal.cee.wisc.edu/partners/community-maps/, 2024. University of Wisconsin–Madison; police-reported crashes from the WisDOT DT4000 report, updated nightly. Accessed 2026.

[4] J. Zhu, K. Wu, P. Li, S. T. Parker, B. Ran, D. A. Noyce, Forecasting the emergence and evolution of crash hotspots: A unified deep learning framework for proactive trafic safety, arXiv preprint arXiv:2607.24168 (2026).

[5] S. Lai, Z. Xu, W. Zhang, H. Liu, H. Xiong, LLMLight: Large language models as trafic signal control agents, in: Proceedings of the 31s ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2025, pp. 2335–2346.

[6] Z. Li, L. Xia, J. Tang, Y. Xu, L. Shi, L. Xia, D. Yin, C. Huang, UrbanGPT: Spatio-temporal large language models, in: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2024, pp. 5351–5362.

[7] S. Zhang, D. Fu, W. Liang, Z. Zhang, B. Yu, P. Cai, B. Yao, TraficGPT: Viewing, processing and interacting with trafic foundation models, Transport Policy 150 (2024) 95–105.

[8] L. Da, K. Liou, T. Chen, X. S. Zhou, X. Luo, Y. Yang, H. Wei, Open-TI: Open trafic intelligence with augmented language model, Internationa Journal of Machine Learning and Cybernetics 15 (2024) 4761–4786.

[9] K. Wu, P. Li, Y. Zhou, R. Gan, J. You, Y. Cheng, J. Zhu, S. T. Parker, B. Ran, D. A. Noyce, Z. Tu, V2X-LLM: Enhancing V2X integration and understanding in connected vehicle corridors, arXiv preprint arXiv:2503.02239 (2025).

[10] T. Nie, J. Sun, W. Ma, Exploring the roles of large language models in reshaping transportation systems: A survey, framework, and roadmap, Artificial Intelligence for Transportation 1 (2025) 100003.

[11] O. Zheng, M. Abdel-Aty, D. Wang, Z. Wang, S. Ding, ChatGPT is on the horizon: Could a large language model be suitable for intelligent trafic safety research and applications?, arXiv preprint arXiv:2303.05382 (2023).

[12] M. Mumtarin, M. S. Chowdhury, J. Wood, Large language models in analyzing crash narratives – a comparative study of ChatGPT, BARD and GPT-4, arXiv preprint arXiv:2308.13563 (2023).

[13] S. Bharati, R. K. C. Khatri, S. Bharati, Benchmarking frontier large language models against oficial crash database coding using police crash narratives, arXiv preprint arXiv:2607.29064 (2026).

[14] A. S. Abdelrahman, M. Abdel-Aty, S. Yang, A. Faden, Advanced crash causation analysis for freeway safety: A large language model approach to identifying key contributing factors, arXiv preprint arXiv:2505.09949 (2025).

[15] H. Zhen, Y. Shi, Y. Huang, J. J. Yang, N. Liu, Leveraging large language models with chain-of-thought and prompt engineering for trafic crash severity analysis and inference, Computers 13 (2024) 232.

[16] Z. Yang, J. He, J. Fan, C. Cirillo, TransportAgents: A multi-agents LLM framework for trafic accident severity prediction, arXiv preprint arXiv:2601.15519 (2026).

[17] Z. Fan, P. Wang, Y. Zhao, Y. Zhao, B. Ivanovic, Z. Wang, M. Pavone, H. F. Yang, Learning trafic crashes as language: Datasets, benchmarks, and what-if causal analyses, arXiv preprint arXiv:2406.10789 (2024).

[18] Y. Zhao, P. Wang, Y. Zhao, H. Du, H. F. Yang, SafeTrafic Copilot: adapting large language models for trustworthy trafic safety assessments and decision interventions, Nature Communications 16 (2025) 8846.

[19] J. Ma, P. Li, R. Gan, K. Cheng, S. T. Parker, B. Ran, An agentic workflow for detecting personally identifiable information in crash narratives, arXiv preprint arXiv:2604.15369 (2026).

[20] X. Zhang, M. Chen, Improving crash data quality with large language models: Evidence from secondary crash narratives in Kentucky, Journa of Safety Research 98 (2026) 588–598.

[21] S. V. Gottimukkala, L. Gao, N. Gharaibeh, Y. Tarnini, M. Talebzadeh, B. Kutela, T. S. Katale, Integrating pavement condition records with LLM-based crash narrative analysis for pavement safety assessment, Accident Analysis & Prevention 234 (2026) 108609.

[22] J. Tang, Y. Luo, T. Li, K. Wang, C. Ding, Incorporating Large Language Models (LLMs) into Transportation Safety Analytics and Equity, Final Research Report Project 161159, New England University Transportation Center, University of Massachusetts Amherst, Amherst, MA, 2024. U.S. Department of Transportation, OST-R, Grant 69A3552348301.

[23] G. Katsogiannis-Meimarakis, G. Koutrika, A survey of deep learning approaches for Text-to-SQL, The VLDB Journal 32 (2023) 905–936.

[24] V. Zhong, C. Xiong, R. Socher, Seq2SQL: Generating structured queries from natural language using reinforcement learning, arXiv preprint arXiv:1709.00103 (2017).

[25] B. Wang, R. Shin, X. Liu, O. Polozov, M. Richardson, RAT-SQL: Relation-aware schema encoding and linking for Text-to-SQL parsers, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020, pp. 7567–7578.

[26] J. Guo, Z. Zhan, Y. Gao, Y. Xiao, J.-G. Lou, T. Liu, D. Zhang, Towards complex Text-to-SQL in cross-domain database with intermediate representation, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019, pp. 4524–4535.

[27] N. Rajkumar, R. Li, D. Bahdanau, Evaluating the Text-to-SQL capabilities of large language models, arXiv preprint arXiv:2204.00498 (2022).

[28] D. Gao, H. Wang, Y. Li, X. Sun, Y. Qian, B. Ding, J. Zhou, Text-to-SQL empowered by large language models: A benchmark evaluation, Proceedings of the VLDB Endowment 17 (2024) 1132–1145.

[29] T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang, D. Radev, Spider: A large-scale humanlabeled dataset for complex and cross-domain semantic parsing and Text-to-SQL task, in: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018, pp. 3911–3921.

[30] J. Li, B. Hui, G. Qu, J. Yang, B. Li, B. Li, B. Wang, B. Qin, R. Cao, R. Geng, N. Huo, X. Zhou, C. Ma, G. Li, K. C.-C. Chang, F. Huang R. Cheng, Y. Li, Can LLM already serve as a database interface? A big bench for large-scale database grounded Text-to-SQLs, in: Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 36, 2023, pp. 42330–42357.

[31] M. Pourreza, D. Rafiei, DIN-SQL: Decomposed in-context learning of Text-to-SQL with self-correction, in: Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023, pp. 36339–36348.

[32] S. Talaei, M. Pourreza, Y.-C. Chang, A. Mirhoseini, A. Saberi, CHESS: Contextual harnessing for eficient SQL synthesis, arXiv preprint arXiv:2405.16755 (2024).

[33] B. Wang, C. Ren, J. Yang, X. Liang, J. Bai, L. Chai, Z. Yan, Q.-W. Zhang, D. Yin, X. Sun, Z. Li, MAC-SQL: A multi-agent collaborative framework for text-to-SQL, in: Proceedings of the 31st International Conference on Computational Linguistics (COLING), 2025, pp. 540–557.

[34] American Association of State Highway and Transportation Oficials, Highway Safety Manual, Technical Report, AASHTO, Washington, D.C., 2010.

[35] R. Elvik, A. Høye, T. Vaa, M. Sørensen, The Handbook of Road Safety Measures, 2nd ed., Emerald Group Publishing, 2009.

[36] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, Y. Cao, ReAct: Synergizing reasoning and acting in language models, in: Internationa Conference on Learning Representations (ICLR), 2023.

[37] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, D. Zhou, Chain-of-thought prompting elicits reasoning in large language models, in: Advances in Neural Information Processing Systems (NeurIPS), volume 35, 2022, pp. 24824–24837.

[38] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, D. Zhou, Self-consistency improves chain of thought reasoning in language models, in: International Conference on Learning Representations (ICLR), 2023.

[39] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegrefe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al., Self-refine: Iterative refinement with self-feedback, in: Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023, pp. 46534–46594.

[40] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, S. Yao, Reflexion: Language agents with verbal reinforcement learning, in: Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023, pp. 8634–8652.

[41] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, I. Stoica, Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, in: Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, volume 36, 2023, pp. 46595–46623.

[42] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, D. Zhou, Large language models cannot self-correct reasoning yet, in: International Conference on Learning Representations (ICLR), 2024.

[43] S. R. Bhagat, I. F. Shihab, A. Sharma, Accuracy is not agreement: Expert-aligned evaluation of crash narrative classification models, in: Proceedings of the IEEE 28th International Conference on Intelligent Transportation Systems (ITSC), 2025, pp. 3236–3243.

[44] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, D. Kiela, Retrieval-augmented generation for knowledge-intensive NLP tasks, in: Advances in Neural Information Processing Systems (NeurIPS), volume 33, 2020, pp. 9459–9474.

[45] R. Gan, J. Ma, P. Li, X. Yang, K. Chen, S. Chen, B. Ran, CrashSight: A phase-aware, infrastructure-centric video benchmark for trafic crash scene understanding and reasoning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2026.

[46] R. Zhong, T. Yu, D. Klein, Semantic evaluation for text-to-SQL with distilled test suites, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 396–411.

[47] Gemma Team, Gemma 3 technical report, arXiv preprint arXiv:2503.19786 (2025).

[48] A. Grattafiori, et al., The Llama 3 herd of models, arXiv preprint arXiv:2407.21783 (2024).

[49] Qwen Team, Qwen2.5 technical report, arXiv preprint arXiv:2412.15115 (2024).

[50] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. K. Li, et al., DeepSeek-Coder: When the large language model meets programming – the rise of code intelligence, arXiv preprint arXiv:2401.14196 (2024).

[51] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, et al., Qwen2.5-coder technical report, arXiv preprint arXiv:2409.12186 (2024).

[52] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, et al., DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning, Nature 645 (2025) 633–638.