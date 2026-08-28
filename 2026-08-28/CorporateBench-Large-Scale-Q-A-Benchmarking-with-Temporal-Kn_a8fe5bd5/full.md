# CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases

Sil Hamilton1, 2\*, Albert Yu Sun1, Oscar J. Romero1, Carl-Leander Henneking1, David Mimno2, Bishan Yang1, Igor Labutov1 1 Epiq AI Labs ² Cornell University

## Abstract

LLMs are increasingly able to answer complex questions about enterprise-scale document collections. But evaluation is hard: companies don't want to share internal communications, and synthetic datasets have been overly simple. We present CORPORATEBENCH (CB), a human-validated multi-task Q&A benchmark whose scale approaches the conditions LLMs encounter in corporate communication networks, with evaluation corpora surpassing 230,000 documents. CB evaluates LLMs across two dimensions (information extraction and knowledge base querying) through four synthetically generated firms ranging from 12 to 10,000 employees. Each corpus is sampled from a temporally evolving knowledge base describing a consistent world, guaranteeing crossdocument logical consistency even across hundreds of thousands of documents. We evaluate five LLMs on CB, revealing increasingly poor performance as input size approaches realistic scales. CB provides LLM developers a metric for corporate communication reasoning, filling a crucial gap in the benchmarking ecosystem.

## 1 Introduction

Complex organizations like corporations produce massive quantities of internal documents through employees communicating and working together. LLMs are increasingly helpful for knowledge work, document analysis, and other tasks requiring general reasoning about organizations and their social networks (Raza et al., 2025; Brachman et al., 2025). But despite significant advances in extending large language model (LLM) context windows (Huang et al., 2024; Beltagy et al., 2020), state-of-the-art LLMs continue to struggle with complex reasoning over many documents as regularly required in enterprise scenarios (Liu et al., 2024; Hsieh et al., 2024). This deficiency hinders wide-scale LLM adoption in industry, but current evaluation metrics fail to capture this failure mode.

![](images/2b420eb4afd02157e17b75252202e6ac1c44ddf92dd1b2445534e07e2cef86da.jpg)  
Figure 1: CORPORATEBENCH advances the state of the art in multi-document QA benchmarking when compared against nine previous benchmarks in evidence and question counts. CB achieves a high ratio of 87.6 documents per question, with the closest benchmark (EKRAG) achieving a ratio of 1.5 (Yu et al., 2025). A high ratio indicates that questions require synthesizing information across many documents, which better reflects the complexity of real enterprise knowledge work.

Common long context understanding tests do not reproduce the complexity and interconnectedness of real corporate environments. Synthetic benchmarks (Kamradt, 2023) are excessively simplified, while more realistic benchmarks based on real-world enterprise data are scarce due to nondisclosure agreements (Zhang et al., 2024). The few existing enterprise benchmarks moreover fail to adequately reproduce corporate conditions due to limitations in scale and realism. Enterprise corpora can scale to millions of documents, making the cost of generating a whole corpus from scratch a major concern when designing benchmarks.

To address these limitations, we introduce CoR-PORATEBENCH, an enterprise benchmark containing large and internally consistent corporate communication networks with verifiable ground truth as shown in Figure 2.1 CORPORATEBENCH bridges the synthetic-real gap by first sampling structures, and then documents, from procedurally generated knowledge bases (KBs) that capture complex relationships between employees, teams, projects, and tasks over time. These KBs serve as our foundation for generating diverse corporate corpora at realistic scales. Our primary contributions are as follows:

• Comprehensive company simulation. We adapt KB-to-corpus generation to produce corporate document collections closely mimicking the real world, preserving temporal sequencing and role-respecting communication directions across teams and departments.

• Deeply interconnected evidence sets. We construct a large-scale benchmark in which answering typical queries requires integrating evidence across many documents, stressing multi-document retrieval and reasoning over enterprise-scale communication sets whose total size can exceed context windows.

• Enterprise-scale tasks. We provide a task suite with deterministic labels directly computed from the KB, enabling reproducible evaluation. This KB-grounded design targets the gap between models’ extended context windows and genuine long context understanding in enterprise settings.

## 2 Related Work

Benchmarking LLMs in long contexts. Enterprise datasets are naturally long context. Contemporary LLMs can now attend to millions of tokens (OpenAI et al., 2024; Grattafiori et al., 2024; Team Gemini et al., 2024), but these models suffer from poor long context understanding (Kaplan et al., 2020; Liu et al., 2024). One method for identifying these failure modes is the “needle-in-a-haystack" test and its descendants (Kamradt, 2023; Vodrahalli et al., 2024; OpenAI, 2025), but these synthetic benchmarks trade realism for generation speed, leading to newer benchmarks testing models over documents like books and movie scripts (Kočiský et al., 2018; An et al., 2023; Karpinska et al., 2024; Hamilton et al., 2025) whose limited number fail to qualify as enterprise scale. Researchers seeking solutions have introduced enterprise benchmarks based on financial records (Deußer et al., 2022; Xie et al., 2024; Xu et al., 2024), legal (Manor and Li, 2019; Guha et al., 2023; Fei et al., 2023; Ryan et al., 2025), knowledge work (Boisvert et al., 2024; Drouin et al., 2024; Styles et al., 2024; Yao et al., 2024; Xu et al., 2025), and other general corporate tasks (Jiang et al., 2024; Zhang et al., 2024; Wang et al., 2025). These benchmarks suffer from low quality data sources, typically transforming non-corporate data to avoid violating nondisclosure agreements. An alternative approach involves generating diverse synthetic enterprise corpora with LLMs emulating knowledge work (Choubey et al., 2025; Huang et al., 2025; Vishwakarma et al., 2025), but existing implementations suffer from small scale (ca. ≤30,000 documents) and problematic ground truth because datasets are created via prompting LLMs and there is no single method for guaranteeing LLM output will match all desired characteristics. Matching real corporate scale therefore demands new generation methods.

Synthetic knowledge bases. Because enterprise environments are structurally self-similar (e.g. employees exist at all levels of an organization), they can be conveniently described in a knowledge base. KBs have long been used to represent human knowledge (Bollacker et al., 2008; Talmor and Berant, 2018), and more recently for producing grounded synthetic documents (Agarwal et al., 2021), but complex systems like enterprise environments were rarely profiled in early studies due to difficulties with parsing relations from unstructured text (Kwiatkowski et al., 2013; Yang et al., 2015; Yang and Mitchell, 2019). LLMs have since proven adept at this task (Labutov et al., 2019b; Sun et al., 2023; Machado et al., 2024; Sahay et al., 2025; Gong et al., 2025) and likewise converting text to structured queries (Labutov et al., 2019a; Trivedi et al., 2022; Wang et al., 2023; Sun et al., 2025; Chen et al., 2025). We look to KBs to provide a single, arbitrarily detailed world state from which we can (i) synthesize document collections at realistic, million-scale sizes and (ii) derive exact ground truth by querying the underlying graph.

## 3 Dataset Construction

We begin constructing our synthetic enterprise benchmark by establishing ground truth. In this section we describe a corpus construction pipeline whose output enjoys two properties:

![](images/9a765831e7a315f2034e21b8e32d633839d4291b8adb54ff438154f2bb4bb9ed.jpg)  
Figure 2: CORPORATEBENCH contains four synthetic companies spanning different industries, headcounts, and business models. The figure shows a vertical slice of a hypothetical company and its social network at a given time.

• Arbitrary scale. For any desired corpus size n, the pipeline can generate $\geq n$ documents while maintaining consistent quality across all generated documents, maintaining future relevance as LLM context windows grow.

• Logical consistency. $\forall d \in C .$ no statement in document d contradicts any fact established in our ground truth knowledge base.

We first generate a knowledge base representing the world of a company C, encoding all entities together with their properties and relationships. We then generate emails that substantiate organizational relationships, tasks, and project progress from the KB, enforcing consistency by inserting “evidence" strings revealing relationships. This pipeline satisfies logical consistency because under ideal circumstances the graph can be reconstructed from all documents sampled from it, and arbitrary scale because documents are templated.

## 3.1 Defining the Knowledge Base

We begin with a formal ontology capturing the desired semantics of our companies, written in Terse RDF Triple Language (Turtle; Beckett et al. 2014). We formally define a company $C = ( D , T , E )$ as the union of all employees E organized by teams T and departments D, where each team strictly belongs to one department and all employees strictly belong to one team, barring department leads and executives. Serializing our companies with an OWL parser guarantees consistency.

Our ontology defines two base classes, Entity and Relationship, serving as nodes and edges in the knowledge base. We extend these with organizational entities (companies, departments, teams, employees) and work matter (projects, meetings, tasks), linked by seven relationship predicates: MemberOf, ReportsTo, WorksOn, WorksAt, Attends, Organize, and BelongsTo. These tie together employees along three axes: organizational (membership and reporting hierarchies), work-wise (task assignments), and communication (meeting attendance and organization).

## 3.2 Generating Companies

To generate company knowledge bases, we run the following procedure conditioned on the type of company and desired number of employees:

1. Produce company hierarchy. We logarithmically scale department and team counts with the desired employee count, targeting an average team headcount of 5–10 and department headcount of 8–68 when scaling from $1 0 ^ { 1 }$ to $1 0 ^ { 4 }$ employees.2 C-suite scales from 1 to 13 executives; departments and teams are equipped with one or more managers. We link all entities with ReportsTo and MemberOf relationships and verify the hierarchy forms a valid tree with no orphan nodes.

2. Simulate work over time. We simulate the company over the course of one quarter, or 90 days. For each day, we randomly assign each employee a task with $P = 1 / 4 5$ such that each employee completes between 1 to 3 tasks by the end of the quarter.³ We track task assignment and completion dates. We verify that all task assignments fall within the simulated quarter and that no employee exceeds the target of 1 to 3 tasks.

3. Simulate meetings over time. We simulate meetings for employees across all six meeting types: direct reports, team meetings, task collaboration sessions, executive meetings, project reviews, and department meetings. The meeting generation process considers employee roles, reporting structures, team memberships, and project assignments to determine appropriate meeting schedules. Appendix D includes more details on meetings. We confirm all generated meetings respect role constraints, e.g. that direct report meetings only pair employees with their managers.

4. Update company hierarchy. At the end of the quarter we update the company hierarchy by hiring an additional ≈10% employees into a random assortment of teams. This process necessarily forces the splitting of those teams to maintain the desired average headcount across all teams, producing additional MemberOf relationships. This process ensures that not all MemberOf relationships are valid at the end of the quarter, thus requiring at least two documents to fully “reveal" employment status when sampling evidence. We verify that new hires are correctly linked with fresh MemberOf and ReportsTo relationships and that prior relationships are marked with appropriate end dates.

5. Annotate all entities. The company hierarchy at this point contains many entities, but no semantic properties such as names or titles. We prepare a one-paragraph “story" for each company in Figure 2 containing details such as company origin, motto, focus, industry, etc.4 We then recursively pass each entity beginning at the top of the hierarchy down to the bottom to Claude Haiku 4.5 prompting it to produce a set of appropriate properties for each entity given its type and size. We finally produce aliases for employees and projects for diversity. We manually inspect a sample of annotated entities to confirm properties are contextually appropriate.

6. Produce triples. We finally render all entities, properties, and relationships in a sequence of triples as described in subsection 3.2. We store the graph in a N-Quads file (Tomaszuk and Hyland-Wood, 2020), compressing it with the ontology. We validate the serialized graph against the OWL ontology to ensure no constraint violations.

From: Emilia A. <eanglada@hotmail.com>   
To: Jared C. <jaredchartier@gmail.com>   
Date: 2024-01-10   
Subject: Addressing code quality concerns   
in our operations   
Hi Jared, I've been reflecting on how our   
current business operations could benefit   
from a more intentional approach to   
managing technical   
debt. As our codebase continues to grow,   
I'm noticing that we're accumulating   
layers of complexity that might slow down   
our team's ability to iterate   
efficiently. I think it would be valuable   
for us to establish a clearer strategy   
around this.   
Building on that concern, Jared, I wanted   
to get your direction here on how we   
should prioritize which areas to tackle   
first. I'm thinking we could start by   
identifying the components that have the   
highest maintenance burden and then work   
through a systematic refactoring plan.   
Would you be open to discussing this   
further so we can align on the best path   
forward for our technical infrastructure?   
Best,   
Emilia Anglada  
Figure 3: An email generated for CORPORATEBENCH. Employees, topic, and evidence strings are in bold.

## 3.3 Generating Documents

We generate emails that surface a set of one or more relationships r ∈ R in our KB, where an email contains a sender, recipient, subject, body, and timestamp as retrieved from the KB. We guarantee we substantiate relationship triples by adding evidence string(s) that explicitly reveal the relationship r from a predefined set of 3,217 strings across all seven Relationship types, split into 360 strings per temporal bound and 40 strings per level of difficulty.⁵ Each string differs in how explicit the string is in revealing the relationship, and the time frame it reveals (either beginning or end of a relationship). Meetings use a separate set of evidence strings, illustrating task and project progress by the meeting date, with strings binned by progress up to 25%, 50%, 75%, and 100%. We provide examples in Appendix B. We finish the email by passing the evidence string(s) to Claude Haiku 4.5 to in-fill the remainder of the email. An example is given in Figure 3. We use the following strategies to enhance lexical diversity in the output:

Format. Our email formatting conventions (headers, signatures, reply structure) are modeled on the Enron corpus (Klimt and Yang, 2004), a publicly available corporate communication dataset, to ensure structural realism.

Topic. To ensure diversity of synthetically generated data at scale, we sample a random topic T and use it to generate the majority of each email. Topics are sampled from a Dirichlet distribution with α = 0.75 to mimic real-world distributions where topic frequency is non-uniform. There are two sets: 3,000 business topics (e.g. clinical trials), and 3,000 non-business topics (e.g. food, travel, and transportation). Meeting topics are sampled from a separate set of 60 topics.6

Personality and level of formality. We assign each employee one of sixteen personalities sourced from the Myers-Briggs Type Indicator typology (Briggs, 1976). At the document level, we instruct the model to write emails in either a formal or a casual manner. Qualitative observations indicate these contribute positively to lexical diversity.

## 4 Dataset Validation

We use the pipeline described in section 3 to produce four company knowledge bases with 10¹ to 104 employees. We present these companies in Table 1. We validate the dataset in four ways:

Network properties. We observe consistent scaling patterns across each company size. For example, the smallest company (Zenith Labs) contains 12 employees, while the largest company (Pound) contains 10,210 individuals. Zenith Labs contains 455 total relationships at an average ≈38 degrees per employee; while Pound contains ≈37 degrees. This stable scaling behaviour helps downstream tasks control for issues arising from poor scaling.

Examining sampled documents. We generate a total 263,466 documents across all four company KBs to produce four corpora containing, respectively: 354 documents, 3,926 documents, 26,493 documents, and 232,693 documents. Each document explicitly reveals one or more relationships in each company KB following the sampling methodology described in subsection 3.3.7

Validating document topics. We provide topic distributions for each company corpus in Table 8. Total topic counts scale linearly from 6 in Zenith (S), 60 for Streamvibe (M), 600 for Biocure (L), and finally 6,000 topics in Pound (XL).

Human validation. We sample 100 random emails from the full CORPORATEBENCH corpus for human validation. Because each document positively asserts a relationship (e.g. that one person manages another), we prompt Claude Sonnet 4 to invert each document by modifying the evidence string to negate the relationship. This yields a balanced test set of positive and negative examples. We pair ten groups of five non-expert reviewers with batches of 20 documents each, asking them to judge whether each document positively asserts the relationship on a binary scale. Across all 1,000 judgements, reviewers correctly identify the intended relationship 76.2% of the time.

## 5 Tasks

CORPORATEBENCH tests models on five tasks organized in two categories: extraction and QA. The two assess distinct capabilities. Extraction assesses whether a model can construct a structured knowledge base from raw documents while QA evaluates reasoning over an already-constructed representation. A model may answer KB QA questions correctly yet fail to build the KB in the first place.

## 5.1 Extraction Tasks

Our two extraction tasks evaluate models on their ability to recover (i) relations and (ii) topics from each company corpus. The tasks are:

KB Evaluation. This task involves three stages: ingestion, mapping, and evaluation. During ingestion, LLMs extract entities and relationships from documents using the prompt template in Appendix P. We then apply deduplication techniques like exact name match and email-based clustering to reduce redundancy. This process yields triples in the form (subject, predicate, object). We evaluate these ingested triples against ground truth triples from CORPORATEBENCH and compute $F _ { 1 }$ for all matched entities/relationships.

<table><tr><td>Company</td><td>Zenith</td><td>Streamvibe</td><td>Biocure</td><td>Pound</td></tr><tr><td>Size</td><td>Small (S)</td><td>Medium (M)</td><td>Large (L)</td><td>Extra Large (XL)</td></tr><tr><td>Industry</td><td>Tech</td><td>Media</td><td>Pharmacàre</td><td>Finance</td></tr><tr><td colspan="5">Entity Types</td></tr><tr><td>Task</td><td>103</td><td>1,134</td><td>6,479</td><td>52,504</td></tr><tr><td>Meeting</td><td>76</td><td>890</td><td>6,445</td><td>57,265</td></tr><tr><td>Employee</td><td>12</td><td>122</td><td>1,110</td><td>10,210</td></tr><tr><td>Project</td><td>7</td><td>77</td><td>317</td><td>587</td></tr><tr><td>Team</td><td>1</td><td>11</td><td>90</td><td>207</td></tr><tr><td>Department</td><td>1</td><td>7 1</td><td>11 1</td><td>15</td></tr><tr><td>Company</td><td>1</td><td></td><td></td><td>1</td></tr><tr><td colspan="5">Relationship Types</td></tr><tr><td>WorksOn</td><td>124</td><td>1,356</td><td>7,708</td><td>63,431</td></tr><tr><td>ReportsTo</td><td>11</td><td>133</td><td>1,255</td><td>11,455</td></tr><tr><td>MemberOf</td><td>11</td><td>126</td><td>1,576</td><td>15,954</td></tr><tr><td>WorksAt</td><td>12</td><td>122</td><td>1,110</td><td>10,210</td></tr><tr><td>Attends</td><td>221</td><td>2,709</td><td>22,772</td><td>213,590</td></tr><tr><td>Organize</td><td>76</td><td>890</td><td>6,445</td><td>57,265</td></tr><tr><td>Total Entities</td><td>201</td><td>2,242</td><td>14,453</td><td>120,789</td></tr><tr><td>Total Relationships</td><td>455</td><td>5,336</td><td>40,866</td><td>371,905</td></tr><tr><td>Total Documents</td><td>354</td><td>3,926</td><td>26,493</td><td>232,693</td></tr></table>

Table 1: Summary statistics for each of the four synthetic companies generated for CORPORATEBENCH. Note that each company represents an additional order of magnitude over the previous in all respects.
<table><tr><td>KB QA</td><td>How many people began working at Pound Financial Group LLC after 7th March 2024?</td></tr><tr><td>Topic QA</td><td>Did Marion Friend discuss both trans- portation and value creation in com- munications after 30 Jan 2024?</td></tr><tr><td>Integrated QA</td><td>Who among client financial docu- mentation preparation workers sent external emails about due diligence?</td></tr></table>

Table 2: Example questions for Pound (XL) across KB, Topic, and Integrated QA. More extensive question breakdowns are available in Appendix F.

Topic Classification. This task tests the model's ability to classify email topics via multi-class classification on a subset of topics from the dataset. The model must select one of either 31 classes (for the large company Biocure) or 17 classes (for the extra large company Pound) on a stratified test set of 1000 documents per corpus as provided in Appendix R. We subset these topics from the original full set of topics by navigating up the topic hierarchy as described in Appendix C to arrive at a lower count tenable for classification with current LLMs.8 We finally evaluate performance with a macro $F _ { 1 }$ averaged across the classes.

## 5.2 QA Tasks

Our three QA tasks evaluate models on factual questions about the company and its documents. QA questions are generated from 250 manually written question templates instantiated across companies and tasks with placeholder variables (e.g., employee or task) that are filled dynamically with values from the KB via verified SPARQL queries (and thus not by LLM generation). By construction, more complex questions require gathering information across more nodes in the knowledge graph and thus across more documents. QA questions come in five answer types: string, date, integer, boolean, and sets of strings. All types, except for sets, are judged on exact match (1 if match, 0 if not), but we calculate $F _ { 1 }$ for sets to allow models partial credit. We compare models in two QA settings. The first is RAG (Lewis et al., 2021). We equip models with a RAG-based search tool containing all the documents.9 This setting establishes performance in realistic corporate settings. The next setting is KB, where we equip models with a SQL tool for querying data directly imported with our ETL pipeline from our ground truth KB file.10 This setting establishes an upper bound on model performance relative to the RAG setting, since models receive direct access to structured ground truth rather than retrieving from noisy documents. We note that textto-SQL remains a challenging task in its own right, and that these scores do not represent an absolute performance ceiling. The tasks are as follows:

KB QA. This task involves answering questions about entities, relationships, and the temporality of these relationships. The example question in Table 2 requires identifying employees from emails, and temporal reasoning to identify when employees joined based on onboarding documents.

![](images/fe55a6469c490a819fb71ce5b2e9a326d8cc11d477d8207b7be840a9ee81a53f.jpg)

Figure 4: Topic classification $F _ { 1 }$ scores across number of few-shot examples provided in the prompt (for LLMs) or labeled training examples (for TF-IDF+LR) for Biocure (L) and Pound (XL) datasets.
<table><tr><td></td><td></td><td colspan="3">Entities</td><td colspan="3">Relationships</td><td colspan="3">Temporal Relationships†</td></tr><tr><td>Size</td><td>Model</td><td>Precision</td><td>Recall</td><td> $\mathbf { F } _ { 1 }$ </td><td>Precision</td><td>Recall</td><td> $\mathbf { F } _ { 1 }$ </td><td>Precision</td><td>Recall</td><td> $\mathbf { F } _ { 1 }$ </td></tr><tr><td rowspan="3">S</td><td>Haiku 4.5</td><td>.866</td><td>.786</td><td>.824</td><td>.751</td><td>.966</td><td>.845</td><td>.453</td><td>.488</td><td>.470</td></tr><tr><td>GPT-5 Nano</td><td>.926</td><td>.624</td><td>.746</td><td>.795</td><td>.285</td><td>.419</td><td>.317</td><td>.092</td><td>.142</td></tr><tr><td>Gemini 2.5 F-L</td><td>.846</td><td>.792</td><td>.818</td><td>.677</td><td>.969</td><td>.797</td><td>.441</td><td>.444</td><td>.442</td></tr><tr><td rowspan="3">M</td><td>Haiku 4.5</td><td>.911</td><td>.721</td><td>.805</td><td>.782</td><td>.595</td><td>.676</td><td>.427</td><td>.209</td><td>.281</td></tr><tr><td>GPT-5 Nano</td><td>.902</td><td>.645</td><td>.752</td><td>.569</td><td>.165</td><td>.256</td><td>.273</td><td>.046</td><td>.078</td></tr><tr><td>Gemini 2.5 F-L</td><td>.849</td><td>.749</td><td>.796</td><td>.613</td><td>.708</td><td>.657</td><td>.391</td><td>.279</td><td>.326</td></tr><tr><td rowspan="3">L</td><td>Haiku 4.5</td><td>.902</td><td>.704</td><td>.790</td><td>.646</td><td>.556</td><td>.597</td><td>.390</td><td>.207</td><td>.271</td></tr><tr><td>GPT-5 Nano</td><td>.928</td><td>.673</td><td>.780</td><td>.479</td><td>.167</td><td>.247</td><td>.254</td><td>.055</td><td>.090</td></tr><tr><td>Gemini 2.5 F-L</td><td>.825</td><td>.707</td><td>.762</td><td>.417</td><td>.539</td><td>.470</td><td>.010</td><td>.012</td><td>.011</td></tr><tr><td rowspan="3">XL</td><td>Haiku 4.5</td><td>.937</td><td>.648</td><td>.766</td><td>.616</td><td>.412</td><td>.494</td><td>.310</td><td>.120</td><td>.173</td></tr><tr><td>GPT-5 Nano</td><td>.962</td><td>.652</td><td>.777</td><td>.456</td><td>.133</td><td>.206</td><td>.191</td><td>.037</td><td>.062</td></tr><tr><td>Gemini 2.5 F-L</td><td>.899</td><td>.594</td><td>.715</td><td>.227</td><td>.186</td><td>.205</td><td>.289</td><td>.082</td><td>.128</td></tr></table>

Table 3: KB Evaluation Results. Bold denotes best score out of the three models.

Topic QA. This task involves answering questions about the topics of the documents that are covered in emails. The Topic QA example in Table 2 requires models to read the employee's emails during a specific interval of time and answer whether the employee discussed that topic.

Integrated QA. The final QA task combines KB QA and Topic QA. The example question in Table 2 requires models to determine which employees have worked on a task and then analyze the content of specific emails they sent.

## 6 Results

We evaluate five contemporary models on CoR-PORATEBENCH to provide a sense of the current state of the art. These models are: Claude Haiku 4.5, Claude Sonnet 4.5, GPT-5 Nano, GPT-5.1, and Gemini 2.5 Flash Lite. We specifically select “lighter"models to strike a balance between performance and cost given evaluating a model on the full 263,466 document set in CB can incur significant API costs. We provide a full run-down of our evaluation metrics in Appendix K.

## 6.1 Extraction Tasks

For KB evaluation, Table 3 shows entity extraction substantially outperforms relationship extraction across all models and dataset sizes. Entity extraction remains stable across scales $( F _ { 1 } \colon 0 . 7 1 5  – 0 . 8 2 4 )$ while relationship extraction degrades notably from smaller (S: 0.419–0.845, M: 0.256–0.676) to larger datasets (L: 0.247–0.597, XL: 0.205–0.494). Temporal relationships prove most challenging, deteriorating from $F _ { 1 } 0 . 1 4 2 – 0 . 4 7 0$ (S) to 0.062–0.173 (XL). These findings indicate models struggle to maintain relational consistency at scale, even as factual consistency (entity extraction) is preserved. Error analysis revealed two primary causes: (1) entity extraction errors cascade into relationship extraction, and (2) contradictory relationship descriptions accumulate across documents, with each model struggling with different relationship types.11

<table><tr><td rowspan="2">Method</td><td rowspan="2">Model</td><td colspan="4">KBQA</td><td colspan="4">Topic QA</td><td colspan="4">Integrated QA</td></tr><tr><td>S</td><td>M</td><td>L</td><td>XL</td><td>S</td><td>M</td><td>L</td><td>XL</td><td>S</td><td>M</td><td>L</td><td>XL</td></tr><tr><td rowspan="5">RAG</td><td>Haiku 4.5</td><td>.265</td><td>.273</td><td>.189</td><td>.164</td><td>.650</td><td>.563</td><td>.418</td><td>.464</td><td>.332</td><td>.277</td><td>.327</td><td>.424</td></tr><tr><td>Sonnet 4.5</td><td>.322</td><td>.311</td><td>.198</td><td>.215</td><td>.656</td><td>.535</td><td>.423</td><td>.492</td><td>.439</td><td>.250</td><td>.272</td><td>.401</td></tr><tr><td>GPT-5 Nano</td><td>.527</td><td>.501</td><td>.312</td><td>.244</td><td>.672</td><td>.479</td><td>.385</td><td>.456</td><td>.511</td><td>.475</td><td>.468</td><td>.510</td></tr><tr><td>GPT-5.1</td><td>.470</td><td>.440</td><td>.276</td><td>.269</td><td>.634</td><td>.485</td><td>.406</td><td>.456</td><td>.469</td><td>.564</td><td>.560</td><td>.566</td></tr><tr><td>Gemini 2.5 F-L</td><td>.230</td><td>.200</td><td>.199</td><td>.147</td><td>.572</td><td>.401</td><td>.374</td><td>.376</td><td>.338</td><td>.218</td><td>.313</td><td>.295</td></tr><tr><td rowspan="5">KB</td><td>Haiku 4.5</td><td>.702</td><td>.724</td><td>.614</td><td>.597</td><td>.811</td><td>.619</td><td>.517</td><td>.488</td><td>.488</td><td>.540</td><td>.538</td><td>.540</td></tr><tr><td>Sonnet 4.5</td><td>.767</td><td>.742</td><td>.628</td><td>.642</td><td>.848</td><td>.687</td><td>.576</td><td>.571</td><td>.611</td><td>.624</td><td>.644</td><td>.647</td></tr><tr><td>GPT-5 Nano</td><td>.577</td><td>.484</td><td>.519</td><td>.522</td><td>.528</td><td>.431</td><td>.384</td><td>.409</td><td>.434</td><td>.611</td><td>.588</td><td>.567</td></tr><tr><td>GPT-5.1</td><td>.555</td><td>.547</td><td>.481</td><td>.525</td><td>.468</td><td>.403</td><td>.387</td><td>.397</td><td>.431</td><td>.640</td><td>.653</td><td>.607</td></tr><tr><td>Gemini 2.5 F-L</td><td>.258</td><td>.207</td><td>.248</td><td>.192</td><td>.384</td><td>.384</td><td>.304</td><td>.352</td><td>.272</td><td>.483</td><td>.528</td><td>.412</td></tr></table>

Table 4: KB QA, Topic QA, and Integrated QA Results. Bold denotes best model performance for the particular QA task using the respective methods (RAG or KB).

For topic classification, we compare against a TF-IDF+LR baseline. Figure 4 and Table 13 reveal distinct patterns across datasets. On Biocure (L), LLMs achieve strong few-shot performance (F1: 0.872–0.950 zero-shot, 0.877–0.982 with 50 examples), matching TF-IDF+LR's 10K-example performance (F1: 0.993) with 200× less data. On Pound (XL), LLMs show steeper learning curves (F1: 0.357–0.531 zero-shot to 0.598–0.751 with 50 examples) but underperform the data-rich baseline (F1: 0.983 at 100K examples). LLMs performed better on Biocure (31 categories) than Pound (17 categories) despite more classes, likely because Pound's hierarchical topic merging created ambiguous boundaries and semantic overlap.

## 6.2 QA Tasks

We report all QA performance in Table 4. Models consistently perform better with direct KB access than with RAG. Our design holds the number of questions constant at 250 per company per task (Appendix K) while increasing corpus size, so larger companies present proportionally more documents per question (≈1.4 to ≈931). Across all three QA types, the best KB scores exceed the corresponding best RAG scores, and this gap widens with scale: on KB QA, the KB–RAG difference increases from 0.24 (S) to 0.37 (XL). This suggests RAG may suffice for smaller corpora, but scale causes problems when surfacing knowledge.

Performance degrades with corpus size for KB QA and Topic QA, reflecting the difficulty of locating precise information in larger collections. Integrated QA exhibits a different pattern: performance slightly improves with scale for both methods, likely because larger corpora provide richer cross-source context for synthesis.

We finally note model-level differences: both GPT-5 models outperform others with RAG, while Claude Sonnet 4.5 excels at KB querying. However, GPT-5.1 exhibits an early stopping problem that disproportionately affects the KB setting (191 occurrences vs. 56 with RAG), partially explaining the performance gap between methods.12

## 7 Conclusion

We release CORPORATEBENCH, a benchmark for testing LLMs on enterprise-specific retrieval and reasoning tasks. CB features 263,466 total synthetic documents sampled from four procedurally generated knowledge bases representing companies of varying scale. This novel generation process allows CB to produce realistic tasks over corpora of arbitrary scale while maintaining inter-document logical consistency, a quality predecessor enterprise benchmarks lacked. Testing five recent LLMs from Anthropic, OpenAI, and Google indicates contemporary LLMs are marginally competent at completing enterprise tasks, with performance decreasing inversely with corpora scale up to and beyond 10⁵ documents. Our methods, benchmark, and results are of value to researchers and industry practitioners interested in assessing LLM performance when deployed in corporate environments.

Next steps. We encourage future researchers to continue closing the gap between synthetic and realistic benchmarks. Researchers working in other domains (e.g. deep research) will want to evaluate whether replicating our use of procedurally generated KBs will be similarly fruitful.

## Limitations

We acknowledge the following limitations:

Scope limited to communication networks. CORPORATEBENCH focuses on email, which remains the dominant document type in corporate communication networks: the Enron corpus, the largest publicly available corporate dataset, is composed almost entirely of email (Klimt and Yang, 2004). Real corporate settings additionally involve communication channels such as Slack and Microsoft Teams, and document types such as technical specifications, financial reports, and presentations, each with distinct structural characteristics. Extending CORPORATEBENCH to additional communication channels is a direction for future work.

Simplified document types. Our benchmark focuses primarily on email communications. Real corporate settings involve newer communication channels such as Slack and Microsoft Teams, and documents themselves may be diverse across document types such as technical specifications, financial reports, spreadsheets, presentations. Each document type has distinct structural characteristics that may present different challenges for LLMs beyond just emails. We note that emails still remain a large majority document type in recent corporate discovery contexts (Gallivan, 2024).

Temporal dynamics and evolution. In simulating one quarter of company activity, we explicitly do not model longer-term organizational evolution such as strategic pivots, mergers and acquisitions, cultural shifts, or the gradual accumulation of institutional knowledge over years. We leave it for future work to expand beyond the 90-day simulation window to surface longer term phenomena.

LLM-generated artifacts. Despite efforts to ensure diversity through topics, personalities, and formality levels, the documents are generated by Claude Haiku 4.5, which may introduce systematic biases or artifacts in writing style, vocabulary, and document structure. Models evaluated on this benchmark may perform differently on humanwritten corporate documents. We see our tasks as a “lower bound" in terms of difficulty, where LLM agents need to be able to solve our tasks first before solving likely messier, more difficult, and more complex human-written corpora.

QA datasets. There are a few assumptions and/or design decisions that affect the QA datasets. These are offered to the reader in Appendix J.

## Ethical Considerations

All synthetically generated materials in CORPO-RATEBENCH were generated via OpenAI and Anthropic-provided API endpoints with content filtering enabled. We note LLM developers often use benchmarks as a reference point during training, and that this practice can lead to benchmarkspecific idiosyncrasies and can disproportionately emphasize certain semantic and/or syntactic qualities in model output. We therefore promoted ethnic and racial diversity in our generated companies by pre-computing names with Faker, a Python library for procedurally generating personal information like names and emails. We enable all Latin alphabet-based Faker lists, ensuring a fair and diverse sample of humans across all four companies.

## References

Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami Al-Rfou. 2021. Knowledge graph based synthetic corpus generation for knowledge-enhanced language model pre-training. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3554–3565.

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2023. L-Eval: Instituting Standardized Evaluation for Long Context Language Models. Preprint, arXiv:2307.11088.

David Beckett, Tim Berners-Lee, Eric Prud'hommeaux, and Gavin Carothers. 2014. Rdf 1.1 turtle. World Wide Web Consortium, 25:18–31.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Léo Boisvert, Megh Thakkar, Maxime Gasse, and Massimo Caccia. 2024. WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks. In 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks.

Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: A collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD International Conference on Management of Data, SIGMOD '08, pages 1247–1250, New York, NY, USA. Association for Computing Machinery.

Michelle Brachman, Amina El-Ashry, Casey Dugan and Werner Geyer. 2025. Current and future use of

large language models for knowledge work. Preprint, arXiv:2503.16774.

Katharine C Briggs. 1976. Myers-Briggs type indicator. Consulting Psychologists Press Palo Alto, CA.

Yongrui Chen, Zhiqiang Liu, Jing Yu, Lin Ren, Nan Hu, Xinbang Dai, Jiajun Liu, Jiazhen Kang, Shenyu Zhang, Xinda Wang, Keyan Ding, Pengfei Shen, Haolei Zhu, Hongjie Deng, Yisong Wang, Tongtong Wu, Sheng Bi, Wen Zhang, Tianxing Wu, and 5 others. 2025. OneEval: Benchmarking LLM Knowledge-intensive Reasoning over Diverse Knowledge Bases. Preprint, arXiv:2506.12577.

Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Kung-Hsiang Huang, Caiming Xiong, and Chien-Sheng Wu. 2025. Benchmarking Deep Search over Heterogeneous Enterprise Data. Preprint, arXiv:2506.23139.

Tobias Deußer, Syed Musharraf Ali, Lars Hillebrand, Desiana Nurchalifah, Basil Jacob, Christian Bauckhage, and Rafet Sifa. 2022. KPI-EDGAR: A Novel Dataset and Accompanying Metric for Relation Extraction from Financial Documents. In 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA), pages 1654–1659.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. 2024. WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks? Preprint, arXiv:2403.07718.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, and Jidong Ge. 2023. LawBench: Benchmarking Legal Knowledge of Large Language Models. Preprint, arXiv:2309.16289.

Bill Gallivan. 2024. ediscovery - 80% of all documents are email. https://web.archive.org/web/20 250622120805/https://www.digitalwarroom .com/blog/ediscovery-80-of-documents-a re-email-and-attachments. Blog post, Digital WarRoom. Accessed via Internet Archive.

Albert Gong, Kamilė Stankevičiūtė, Chao Wan, Anmol Kabra, Raphael Thesmar, Johann Lee, Julius Klenke, Carla P. Gomes, and Kilian Q. Weinberger. 2025. PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation. Preprint, arXiv:2502.20377.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 11 others. 2024. The Llama 3 Herd of Models. Preprint, arXiv:2407.21783.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, and 21 others. 2023. Legalbench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models. SSRN Electronic Journal.

Sil Hamilton, Rebecca MM Hicke, Matthew Wilkens, and David Mimno. 2025. Too long, didn't model: Decomposing llm long-context understanding with novels. arXiv preprint arXiv:2505.14925.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What's the Real Context Size of Your Long-Context Language Models?Preprint, arXiv:2404.06654.

Kung-Hsiang Huang, Akshara Prabhakar, Sidharth Dhawan, Yixin Mao, Huan Wang, Silvio Savarese, Caiming Xiong, Philippe Laban, and Chien-Sheng Wu. 2025. CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3830–3850, Albuquerque, New Mexico. Association for Computational Linguistics.

Yunpeng Huang, Jingwei Xu, Junyu Lai, Zixu Jiang, Taolue Chen, Zenan Li, Yuan Yao, Xiaoxing Ma, Lijuan Yang, Hao Chen, Shupeng Li, and Penghao Zhao. 2024. Advancing transformer architecture in long-context large language models: A comprehensive survey. arXiv preprint arXiv:2311.12351.

Feihu Jiang, Chuan Qin, Kaichun Yao, Chuyu Fang, Fuzhen Zhuang, Hengshu Zhu, and Hui Xiong. 2024. Enhancing Question Answering for Enterprise Knowledge Bases using Large Language Models. Preprint, arXiv:2404.08695.

Greg Kamradt. 2023. Needle In A Haystack - Pressure Testing LLMs.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. Preprint, arXiv:2001.08361.

Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyyer. 2024. One Thousand and One Pairs: A "novel" challenge for long-context language models. Preprint, arXiv:2406.16264.

Bryan Klimt and Yiming Yang. 2004. The enron corpus: A new dataset for email classification research. In European conference on machine learning, pages 217–226. Springer.

Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The NarrativeQA Reading Comprehension Challenge. Transactions of the Association for Computational Linguistics, 6:317–328.

Tom Kwiatkowski, Eunsol Choi, Yoav Artzi, and Luke Zettlemoyer. 2013. Scaling Semantic Parsers with On-the-Fly Ontology Matching. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1545–1556, Seattle, Washington, USA. Association for Computational Linguistics.

Igor Labutov, Bishan Yang, and Tom Mitchell. 2019a. Learning to Learn Semantic Parsers from Natural Language Supervision. Preprint, arXiv:1902.08373.

Igor Labutov, Bishan Yang, Anusha Prakash, and Amos Azaria. 2019b. Multi-Relational Question Answering from Narratives: Machine Reading and Reasoning in Simulated Worlds. Preprint, arXiv:1902.09093.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks. Preprint, arXiv:2005.11401.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12:157–173.

Marcelo Machado, João M B Rodrigues, Guilherme Lima, and Sandro Rama Fiorini. 2024. {LLM Store}: Leveraging Large Language Models as Sources of {Wikidata}-Structured Knowledge. In KBC-LM'24: Knowledge Base Construction from Pre-trained Language Models Workshop at ISWC 2024.

Laura Manor and Junyi Jessy Li. 2019. Plain English Summarization of Contracts. In Proceedings of the Natural Legal Language Processing Workshop 2019, pages 1–11, Minneapolis, Minnesota. Association for Computational Linguistics.

OpenAI. 2025. OpenAI MRCR: Long Context Multiple Needle in a Haystack Benchmark. https://huggingface.co/datasets/openai/mrcr.

OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, A. J. Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Mądry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, and 45 others. 2024. GPT-4o System Card. Preprint, arXiv:2410.21276.

Mubashar Raza, Zarmina Jahangir, Muhammad Bilal Riaz, Muhammad Jasim Saeed, and Muhammad Awais Sattar. 2025. Industrial applications of large language models. Scientific Reports, 15(1):13755.

Michael J. Ryan, Danmei Xu, Chris Nivera, and Daniel Campos. 2025. EnronQA: Towards Personalized RAG over Private Documents. Preprint, arXiv:2505.00263.

Rishav Sahay, Arihant Jain, Purav Aggarwal, and Anoop Saladi. 2025. AutoKB: Automated Creation of Structured Knowledge Bases for Domain-Specific Support. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: Industry Track), pages 708–723, Albuquerque, New Mexico. Association for Computational Linguistics.

Olly Styles, Sam Miller, Patricio Cerda-Mardini, Tanaya Guha, Victor Sanchez, and Bertie Vidgen. 2024. WorkBench: A Benchmark Dataset for Agents in a Realistic Workplace Setting. Preprint, arXiv:2405.00823.

Albert Sun, Varun Nair, Elliot Schumacher, and Anitha Kannan. 2024. CONSCENDI: A contrastive and scenario-guided distillation approach to guardrail models for virtual assistants. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4009–4030, Mexico City, Mexico. Association for Computational Linguistics.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2023. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. In The Twelfth International Conference on Learning Representations.

Qiang Sun, Yuanyi Luo, Wenxiao Zhang, Sirui Li, Jichunyang Li, Kai Niu, Xiangrui Kong, and Wei Liu. 2025. Docs2KG: A Human-LLM Collaborative Approach to Unified Knowledge Graph Construction from Heterogeneous Documents. In Companion Proceedings of the ACM on Web Conference 2025, pages 801–804, Sydney NSW Australia. ACM.

Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for Answering Complex Questions. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 641–651, New Orleans, Louisiana. Association for Computational Linguistics.

Team Gemini, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh Mariooryad, Yifan Ding, Xinyang Geng, Fred Alcober, Roy Frostig, Mark Omernick, Lexi Walker, Cosmin Paduraru, Christina Sorokin, and 4 others. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Preprint, arXiv:2403.05530.

Dominik Tomaszuk and David Hyland-Wood. 2020. Rdf 1.1: Knowledge representation and data integration language for the web. Symmetry, 12(1):84.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics, 10:539–554.

Harsh Vishwakarma, Ankush Agarwal, Ojas Patil, Chaitanya Devaguptapu, and Mahesh Chandran. 2025. Can LLMs help you at work? a sandbox for evaluating LLM agents in enterprise environments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9178– 9212, Suzhou, China. Association for Computational Linguistics.

Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, Rohan Anil, Ethan Dyer, Siamak Shakeri, Roopali Vij, Harsh Mehta, Vinay Ramasesh, Quoc Le, Ed Chi, Yifeng Lu, and 5 others. 2024. Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries. Preprint, arXiv:2409.12640.

Liya Wang, David Yi, Damien Jose, John Passarelli, James Gao, Jordan Leventis, and Kang Li. 2025. Enterprise Large Language Model Evaluation Benchmark. Preprint, arXiv:2506.20274.

Xintao Wang, Qianwen Yang, Yongting Qiu, Jiaqing Liang, Qianyu He, Zhouhong Gu, Yanghua Xiao, and Wei Wang. 2023. KnowledGPT: Enhancing Large Language Models with Retrieval and Storage Access on Knowledge Bases. Preprint, arXiv:2308.11761.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, and 15 others. 2024. FinBen: A Holistic Financial Benchmark for Large Language Models. In 38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks.

Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Z. Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Maben, Raj Mehta, Wayne Chi, Lawrence Jang, Yiqing Xie, and 2 others. 2025. TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks. Preprint, arXiv:2412.14161.

Liang Xu, Lei Zhu, Yaotong Wu, and Hang Xue. 2024. SuperCLUE-Fin: Graded Fine-Grained Analysis of Chinese LLMs on Diverse Financial Tasks and Applications. Preprint, arXiv:2404.19063.

Bishan Yang and Tom Mitchell. 2019. Leveraging Knowledge Bases in LSTMs for Improving Machine Reading. Preprint, arXiv:1902.09091.

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. Preprint, arXiv:1412.6575.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. \$τ\$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. Preprint, arXiv:2406.12045.

Tan Yu, Wenfei Zhou, Leiyang Leiyang, Aaditya Shukla, Mmadugula Mmadugula, Pritam Gundecha, Nicholas Burnett, Anbang Xu, Viseth Viseth, Tbar Tbar, Rama Akkiraju, and Vivienne Zhang. 2025. EKRAG: Benchmark RAG for Enterprise Knowledge Question Answering. In Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing, pages 152–159, Albuquerque, New Mexico, USA. Association for Computational Linguistics.

Bing Zhang, Mikio Takeuchi, Ryo Kawahara, Shubhi Asthana, Md Maruf Hossain, Guang-Jie Ren, Kate Soule, and Yada Zhu. 2024. Enterprise Benchmarks for Large Language Model Evaluation. Preprint, arXiv:2410.12857.

## A Company Information Overview

Table 5 contains information about the four companies that we have generated. We have generated four companies with different attributes: name, industry, size, model, and story.

## B Evidence Strings

Non-meeting evidence strings. These strings substantiate relationships between email senders and entities (teams, managers, tasks, or companies) through linguistic patterns that vary in explicitness and temporal stage.

Each relationship type contains evidence strings organized by temporal stage (start, ongoing, end) and evidence strength (explicit, implicit). Explicit strings directly state the relationship using clear relationship verbs, while implicit strings demonstrate active engagement through actions, coordination, or possessive language.

To ensure lexical diversity within implicit strings, we organize them into thematic subcategories that represent different aspects of workplace behavior.

For example, the memberOf relationship includes implicit subcategories such as "onboarding\_to\_team" for new members learning processes, "syncing\_with\_team\_members" for ongoing coordination, and "knowledge\_transfer\_to\_team" for members departing the team. The memberOf relationship contains 20 explicit strings per temporal stage and 140 implicit strings (7 subcategories with 20 strings each) per temporal stage, totaling 480 strings. Similar distributions apply to other relationship types, resulting in 3,217 total evidence string variations across all six relationship types.

Table 6 shows example evidence strings for the memberOf relationship.

Meeting evidence strings. These evidence strings differ from non-meeting strings by focusing on task and project progress rather than relationship formation or dissolution.

These strings are organized by progress level rather than temporal stage, with four progress bins: 0-25%, 25-50%, 50-75%, and 75-100%. Each progress bin contains both explicit strings that directly state completion percentage and implicit strings organized into thematic subcategories representing typical activities at that stage of work.

For example, the 0-25% progress bin includes implicit subcategories such as "initial\_research" for gathering requirements, "project\_kickoff" for launching initiatives, and "resource\_gathering" for assembling team and materials. As progress advances, the subcategories shift to reflect later-stage activities: the 50-75% bin includes "beta\_testing" and "integration\_work", while the 75-100% bin includes "final\_review", "deployment\_prep", and "closing\_activities".

Each progress bin contains 20 explicit strings and 120 implicit strings (6 subcategories with 20 strings each), totaling 560 strings for the taskProgress evidence type.

Table 7 shows example evidence strings for different progress levels.

## C Topics

Non-meeting topics. For non-meeting documents, we assign topics related to either business or non-business operations. To maintain lexical diversity as the corpus grows, we scale the number of topics proportionally by creating hierarchical topic structures of varying depth, similar to the scenarioseeded approach defined in (Sun et al., 2024).

We begin with broad top-level categories. Nonbusiness topics consistently use three main categories: "Food", "Travel", and "Transportation". Business topics vary by company domain: for example, Streamvibe (media) uses "Content Creation", "Production and Distribution", and "Audience Engagement", while Biocure (pharmaceuticals) uses "Drug Development", "Drug Approval", and "Drug Sales", and Pound (finance) uses "Deal Sourcing", "Due Diligence", and "Value Creation".

The hierarchy depth depends on corpus size. Zenith, our smallest company, requires no subcategories—its business topics are specific from the start (e.g., "API Security Standards", "Microservice Architecture Patterns"), as are its non-business topics (e.g., "Home Cooking", "Vacation Plans"). For larger companies like Streamvibe, Biocure, and Pound, we recursively expand top-level categories into deeper hierarchies using Claude Sonnet 4.5, creating subcategories at two, three, or four levels as needed. We manually review each generated topic to ensure it is appropriate for corporate email content before inclusion.

Table 9 shows the hierarchical example of topics for Biocure, our large company.

Finally, to select non-meeting topics, we use a Dirichlet topic distribution with a sparse alpha concentration of 0.75 to determine the probability at which we'd use a topic for a particular document.

<table><tr><td>Name</td><td>Industry</td><td>Size</td><td>Model</td><td>Story</td></tr><tr><td>Zenith</td><td>Tech</td><td>Small</td><td>B2C</td><td>Nestled in the heart of California&#x27;s tech scene, Zenith Labs is a nimble, people-first B2C company crafting sleek laptops, phones, and desktops for everyday users who crave simplicity without sacrificing style. With few em- ployees and a laid-back, startup vibe, Zenith Labs punches above its weight in the consumer electronics space, offering intuitive, affordable devices that blend minimalist design with reliable performance. Their audience? Young professionals, students, and creatives who want tech that feels personal—not corporate. Zenith Labs&#x27; motto &quot;Keep it crisp&quot; reflects their mission to strip away the noise and deliver clean, user-friendly experiences. Their goal is to become the go-to brand for people who want smart tech that doesn&#x27;t try too</td></tr><tr><td>Streamvibe</td><td>Media</td><td>Medium</td><td>B2C</td><td>hard. A creative powerhouse in the entertainment industry, StreamVibe Studios produces original streaming content, podcasts, and digital campaigns for the cord-cutting generation. With talented creators, producers, and mar- keters spread across Los Angeles and Austin, StreamVibe has carved out a niche creating viral web series, documentary films, and branded content for Gen Z and millennial audiences. Known for their &quot;content first, suits second&quot; mentality, StreamVibe attracts top creative talent with flexible work arrangements, profit-sharing, and a culture that celebrates bold storytelling and authentic voices.</td></tr><tr><td>Biocure</td><td>Pharma</td><td>Large</td><td>B2B</td><td>Operating from a sprawling research campus with state-of-the-art laborato- ries, BioCure Solutions is a science-driven B2B pharmaceutical company specializing in contract research and development services for drug discov- ery and clinical trials. This established industry leader employs scientists, researchers, and regulatory specialists who maintain rigorous FDA com- pliance standards and operate under strict Good Manufacturing Practice protocols that have secured partnerships with major pharmaceutical corpo- rations worldwide. Their mission is to become the premier CRO partner for pharmaceutical companies developing tomorrow&#x27;s therapies, providing the scientific rigor and regulatory expertise that transforms promising com-</td></tr><tr><td>Pound</td><td>Finance</td><td>X-Large</td><td>B2B</td><td>pounds into approved medicines that improve human health globally. Pound Financial is a mid-market investment firm specializing in private equity and corporate restructuring for manufacturing and technology com- panies. The firm maintains a conservative, relationship-driven approach, focusing on long-term value creation rather than quick exits. Their clientele includes family-owned businesses seeking growth capital and Fortune 500 companies navigating complex financial transitions. Pound Financial&#x27;s rep- utation for discretion, thorough due diligence, and strategic guidance has made them a trusted partner for executives facing critical financial decisions.</td></tr></table>

Table 5: Company/KB Information Overview.

<table><tr><td>Temporal Stage</td><td>Strength</td><td>Examples</td></tr><tr><td rowspan="2">Start</td><td>Explicit</td><td>I just joined {{object} } and wanted to introduce myself I&#x27;m excited to announce that I&#x27;m now part of {{object} } I recently became a member of {{object} } ... 17 more</td></tr><tr><td>Implicit</td><td>onboarding_to_team: Just going through the {{object} } onboarding materials today learning_team_processes: Still getting familiar with how {{object} } runs things getting_added_to_distribution_lists: Just got added to the {{object} } email lists ... 4 more subcategories, 20 strings each</td></tr><tr><td rowspan="2">Ongoing</td><td>Explicit</td><td>As a member of {{object} }, I wanted to share our latest findings I&#x27;m on {{object} }, and we&#x27;ve been tracking this issue My group, {{object} }, has identified a solution ... 17 more</td></tr><tr><td>Implicit</td><td>discussing_team_priorities: This aligns with {{object} }&#x27;s Q1 priorities syncing_with_team_members: Let me check with my {{object} } teammates first contributing_to_team_decisions: {{object} } voted on this approach last week ... 4 more subcategories, 20 strings each</td></tr><tr><td rowspan="2">End</td><td>Explicit</td><td>I&#x27;m leaving {{object} }, effective today My time with {{object} } is coming to an end I&#x27;m departing from {{object} } today ... 17 more</td></tr><tr><td>Implicit</td><td>knowledge_transfer_to_team: Documenting everything for {{object} } before I transition off handing_off_team_responsibilities: Transitioning my { {object } } responsibilities to Sarah wrapping_up_team_projects: Finishing up my last {{object} } project this week ... 4 more subcategories, 20 strings each</td></tr></table>

Table 6: Example evidence strings for the member0f relationship showing temporal stages and evidence strength levels. Explicit strings directly state team membership, while implicit strings demonstrate engagement through workplace actions. The placeholder {{object}} represents the team name.

These distributions are illustrated in Figure 5. The actual topic distributions after generation are illustrated in Figure 6.

Meeting topics. Meeting documents require a different approach to topic assignment, as their content is shaped by meeting dynamics rather than general business operations. We select meeting topics from a separate pool, conditioned on meeting type to reflect typical discussion patterns. For example, direct report meetings commonly cover "Employee Performance Review", while executive meetings focus on "Company Strategy and Vision Alignment".

Not all meeting emails receive topic assignments. Blank calendar invites and similar minimal-content emails lack substantive discussion and are therefore excluded from topic assignment and from our Topic Classification, Topic QA, and Integrated QA evaluation tasks.

Table 10 lists the meeting topics that we use.

## D Meetings

We simulate meetings for employees based on their organizational roles and relationships. Table 11 defines the six meeting types supported in our simulation framework. Each meeting type is characterized by several parameters: a description of the participants, the probability of generating meeting minutes, the probability of generating an agenda, the probability of leaving the calendar event empty (without minutes or agenda), the meeting frequency, the designated organizer, and the knowledge graph relationships that the meeting substantiates.

For each meeting type, we determine participants from relevant organizational relationships. For example, direct report meetings involve an employee and their manager, while team meetings include all employees who belong to or lead a particular team. Meetings are scheduled according to their specified frequencies (weekly, biweekly, or monthly), and associated artifacts (meeting minutes and agendas) are generated based on the probabilities shown in Table 11. The "Empty Cal." column indicates the probability that a meeting appears on calendars without any accompanying documentation.

The "Substantiates" column indicates which knowledge graph relationships are evidenced or reinforced by each meeting type. For instance, direct report meetings substantiate task progress, reporting relationships, and work assignments, while executive meetings provide evidence of project progress and employee titles. This ensures that meetings generate realistic traces of organizational activity that align with the underlying knowledge graph structure.

![](images/aee23b8fde6c491420393929212347fab361677892538087215478c02d3c5539.jpg)

![](images/135ec48dbf47dc985e6e3dc6f1fa7d37ee9dda3967800f3cf5b3237a172fc318.jpg)

![](images/318c25d95111d766549d708fb03502c17c779c923b22d7b5ec8fb6fd3d18446d.jpg)

![](images/16ff6df31555a72a872d836ef2239eb7a278a0ba1ce3042981f9335967719487.jpg)

![](images/c8173f321119b0e45629c4d5bc1da6ac491b23f8a5b9e97063fe514cfff4bd0b.jpg)

![](images/4764e07cc79e035edd0a63ca2786517f035146d410be30a9fc7d49e53a1eb480.jpg)

![](images/ee16c8c4f223084037e290ea22d4c1ccd02b011bba73805ab44b63c1ac490b16.jpg)

![](images/ed29dd883e30aacf55e8a8ca154f576c32114885e71497093022fd27cc8192bd.jpg)

![](images/97013a6384edfbb93caf28cc4b162f11cdaec474e8a1370b1e3eb02b2641ae99.jpg)

![](images/15b8d0f73cd03b4573ac4e20144b2aa01ab15c4616fd4ce7c52d91afaaf8da61.jpg)

![](images/ff53ad18f4f6b43969a1539ff80791ba6584bf952ee71ddc7688cb8d6fb08c5b.jpg)

![](images/00d88a2228e49c577ca1babe297366f9a1298ad3b36f20ee7f5844f7f930eb60.jpg)  
Figure 5: Summary of theoretical Dirichlet distributions used for topic sampling when generating non-meeting documents.

![](images/785fead1fed437622e40d1cc0a93fe65b4d1e2af542399c6dd19e91d7618397e.jpg)

![](images/dec0b6c15368115f23af9829a0bdd71cc837c9081fc8e8ba1b5d2f6abc0a18fa.jpg)

![](images/a99be2dbe76271fe74c6ee339a3cfe0da86056384f2088ec9be66d67ab4bc4eb.jpg)

![](images/3d1001d49acdd81cefbd77bb57399de85b40da7901c18909c4438ae52e52569f.jpg)

![](images/6d2594745bf40159c7429e122f3bf42faf3a3d073847c03ca7c9411a840a6115.jpg)

![](images/84a9424fa5b70569fd1e4291d028c3711a4366a5d75a2fd4f00f6ed68412120a.jpg)

![](images/996ae5c805fd310df172703c2ebfd8d00e2f49c530aca910ed1e30e5620ccf90.jpg)

![](images/8d6a9ad5d8b85c342c12c3446bc52a890780a1c6d5e68699dc88a82a66e1ea52.jpg)

![](images/afb06a77619954f109da82b030b56ced7335ea76aadd2afbf2caef3d07e19adc.jpg)

![](images/9b5d7583e2e7a1bc43976a294ea537e6640162e5bf2f3bf8936a23bcae25915d.jpg)

![](images/3ca63e12b04918cf3a65124221260172b9d6331f1773ab0aebffe5ddb055b78b.jpg)

![](images/cee4b774a58446352c33912c356c59a79f64ac617eada5a0a43b9fe3d81efee2.jpg)  
Figure 6: Summary of empirical topic distributions of generated documents across all four KBs. Includes meeting and non-meeting topics. See Appendix C for more details on topics

<table><tr><td>Progress Level</td><td>Strength</td><td>Examples</td></tr><tr><td rowspan="2">0-25%</td><td>Explicit</td><td>{{employee}} {{have_verb}} {{task}} about 20% done {{employee}} {{be_verb}} roughly a quarter of the way through {{task} } {{task} } is still in early stages with {{employee} }, around 15-25% complete ... 17 more</td></tr><tr><td>Implicit</td><td>initial_research: {{employee} } {{be_verb} } still gathering requirements for {{task} } project_kickoff: {{employee} } sent out the project charter for {{task} } today early_planning: {{employee} } {{be_verb} } sketching out the roadmap for {{task} } ... 3 more subcategories, 20 strings each</td></tr><tr><td rowspan="2">25-50%</td><td>Explicit</td><td>{{employee}} {{have_verb}} {{task}} about 40% complete {{employee}} {{be_verb} } roughly a third done with {{task}} {{task} } is coming along with {{employee} }, around 35% finished ... 17 more</td></tr><tr><td>Implicit</td><td>prototype_development: {{employee} } just got the first prototype working for {{task} } foundation_complete: {{employee} } { {have_verb} } the core framework built for {{task} } initial_deliverables: {{employee} } submitted the first milestone deliverable for {{task}} ... 3 more subcategories, 20 strings each</td></tr><tr><td rowspan="2">50-75%</td><td>Explicit</td><td>{{employee}} { {have_verb}} {{task}} about 60% done now {{employee}} {{be_verb} } past halfway on {{task}}, around 65% complete {{task} } is roughly two-thirds finished by {{employee}} ... 17 more</td></tr><tr><td>Implicit</td><td>beta_testing: {{employee} } {{be_verb} } running trial programs for {{task}} debugging_phase: { {employee} } {{be_verb} } fixing the remaining problems in {{task} } integration_work: {{employee} } {{be_verb} } bringing together all the pieces of {{task}} ... 3 more subcategories, 20 strings each</td></tr><tr><td rowspan="2">75-100%</td><td>Explicit</td><td>{{employee}} {{have_verb}} {{task} } nearly done, about 85% complete {{employee}} {{be_verb}} almost finished with {{task} }, around 80-90% there {{task}} is in the final stretch with {{employee} }, roughly 80% done ... 17 more</td></tr><tr><td>Implicit</td><td>final_review: {{employee} } {{be_verb} } doing the final review of { {task}} polishing_details: { {employee} } {{be_verb} } polishing the documentation for {{task} } deployment_prep: {{employee} } {{be_verb}} preparing {{task} } for implementation ... 3 more subcategories, 20 strings each</td></tr></table>

Table 7: Example evidence strings for the taskProgress relationship showing progress levels and evidence strength. Explicit strings directly state completion percentages, while implicit strings demonstrate typical activities at each stage. The placeholders {{employee}} and {{task}} represent the employee name and task name respectively.
<table><tr><td>Size</td><td>Documents</td><td>Non-Business</td><td>Business</td></tr><tr><td>S</td><td> $\approx 4 \cdot 1 0 ^ { 2 }$ </td><td> $3 \cdot 1 0 ^ { 0 }$ </td><td> $3 \cdot 1 0 ^ { 0 }$ </td></tr><tr><td>M</td><td> $\approx 4 \cdot 1 0 ^ { 3 }$ </td><td> $3 \cdot 1 0 ^ { 1 }$ </td><td> $3 \cdot 1 0 ^ { 1 }$ </td></tr><tr><td>L</td><td> $\approx 3 \cdot 1 0 ^ { 4 }$ </td><td> $3 \cdot 1 0 ^ { 2 }$ </td><td> $3 \cdot 1 0 ^ { 2 }$ </td></tr><tr><td> $\mathbf { X L }$ </td><td> $\approx 2 \cdot 1 0 ^ { 5 }$ </td><td> $3 \cdot 1 0 ^ { 3 }$ </td><td> $3 \cdot 1 0 ^ { 3 }$ </td></tr></table>

Table 8: Breakdown of the number of topics used to condition the generation of documents in CoRPO-RATEBENCH based on different sizes of the company. We provide more detail in Appendix C.

This simulation process results in each employee having an average of 6 meetings per quarter, with individual counts varying based on their position, team membership, project involvement, and report-

ing relationships.

## E ETL Pipeline

ETL for KB-Gold. To extract ground truth from knowledge base files into postgres databases accessible to our agent, we use the StoryBeat KB API to access entities, relationships, and documents.

Extract: We use the StoryBeat KB API's entity, relationship, and document interfaces to extract data from .kb files. The extraction uses generator functions to process data in configurable batches (default: 1000 items) rather than loading entire datasets into memory.

Transform: Entities are mapped from KB properties to database columns based on entity type (Company, Employee, Department, etc.). Each entity type has specific property mappings (e.g., Employee uses hasName for name, hasAlias for aliases). Relationships are extracted with their temporal information (beginning, end) and classified as "ongoing" or "bounded". Documents are processed separately using the KB documents API, with properties including sender/recipient information, dates, topics, and meeting references. Array fields (aliases, recipient lists) are converted from string representations to PostgreSQL array types using ast.literal\_eval.

<table><tr><td>Level 0</td><td>Level 1</td><td>Level 2</td><td>Level 3 (Examples)</td></tr><tr><td rowspan="3">Non-Business</td><td>Food</td><td>Cooking Baking Dining Out Food Culture Nutrition ... 5 more subcategories</td><td>Grilling techniques shared (... 9 more) Bread making adventures (... 9 more) Restaurant service reviews (... 9 more) Cultural food traditions (... 9 more) Vitamin supplement discussions (... 9 more)</td></tr><tr><td>Travel</td><td>Destinations Accommodations Activities Transportation Methods Travel Planning ... 5 more subcategories Public Transit</td><td>Mountain hiking trails (... 9 more) Hotel loyalty programs (.. 9 more) Adventure sports planning (... 9 more) Train journey experiences (... 9 more) Itinerary creation process (... 9 more) Bus schedule reliability (.. 9 more)</td></tr><tr><td>Transportation</td><td>Personal Vehicles Alternative Transport Commuting Ride Sharing ... 5 more subcategories Preclinical Research Clinical Trial Planning</td><td>Car model comparisons (... 9 more) Electric scooter trials (... 9 more) Route optimization strategies (... 9 more) App comparison reviews (... 9 more) Animal Model Selection (... 9 more) Protocol Design Elements (... 9 more)</td></tr><tr><td rowspan="2">Business</td><td>Drug Development</td><td>Regulatory Strategy Partnership Collaborations Biomarker Identification ... 5 more subcategories Regulatory Submission Preparation FDA Communication</td><td>Submission Pathway Selection (... 9 more) Due Process (... 9 more) Biomarker Discovery Methods (... 9 more) Module Compilation Process (... 9 more) Meeting Request Preparation (... 9 more)</td></tr><tr><td>Drug Approval</td><td>Clinical Data Analysis Safety Reporting Manufacturing Compliance ... 5 more subcategories Market Access Strategy Pricing Negotiations</td><td>Statistical Trial Evaluation (... 9 more) Adverse Event Classification (... 9 more) GMP Inspection Preparation (... 9 more) Payer Landscape Analysis (... 9 more) Price Justification Strategy (... 9 more)</td></tr><tr><td></td><td>Drug Sales</td><td>Sales Force Training Key Opinion Leader Engagement Competitive Intelligence ... 5 more subcategories</td><td>Product Knowledge Development (... 9 more) KOL Identification Strategy (... 9 more) Market Landscape Analysis (... 9 more)</td></tr></table>

Table 9: Example non-meeting topic hierarchy for Biocure showing the four-level structure from top-level categories down to specific discussion topics.

Load: Data is loaded into PostgreSQL using batch INSERT operations within transactions for consistency. Entity tables use the KB entity ID as primary key (e.g., employee\_EMP-D44FE8C5). The pipeline supports multiple schemas (one per company/knowledge base), with schema names inferred from KB filenames. Duplicate detection uses ON CONFLICT DO NOTHING to handle idempotent loading.

ETL for RAG. The RAG pipeline builds vector embeddings for document retrieval.

Extract: Documents are exported from KB files to a temporary directory using the KB documents API.

Transform: Document content is truncated to 22,000 characters (≈8,000 tokens) for embedding generation to stay within model limits. Embeddings are generated using OpenAI's textembedding-3-small model (1536 dimensions) with parallel processing (4 workers by default).

Load: Embeddings are stored in PostgreSQL using the pgvector extension with HNSW indexing for efficient similarity search. Batch inserts (500 records per batch) optimize database write performance. Duplicate documents are detected and skipped to avoid redundant API calls. Full document content (not truncated) is stored alongside

<table><tr><td>Meeting Type</td><td>Topics</td></tr><tr><td>Direct Report</td><td>Employee Performance Review, Goal Setting and Progress Check-In, Career Development Discussion, Feedback and Coaching Session, Workload and Prioritization Review, Upcoming Deadlines and Deliverables, Team Dynamics and Collaboration Feedback, Skill Growth and Training Opportunities, Manager-Report Relationship Check-In, Quarterly Performance Sum-</td></tr><tr><td>Team</td><td>Weekly Team Standup, Team Goals and KPIs Review, Process Improvement Discussion, Cross- Functional Collaboration Update, Retrospective and Lessons Learned, Upcoming Projects and Assignments, Team Communication and Workflow Alignment, Resource Planning and Allocation, Team Morale and Culture Check, Problem-Solving and Roadblock Discussion</td></tr><tr><td>Task</td><td>Task Progress and Status Update, Dependency and Blocker Resolution, Timeline and Deliverable Planning, Quality Review and Standards Alignment, Task Ownership and Accountability, Cross- Task Integration Discussion, Testing and Validation Checkpoint, Task Documentation and Handover, Risk and Issue Review, Next Steps and Action Items</td></tr><tr><td>Executive</td><td>Company Strategy and Vision Alignment, Financial Performance and Budget Review, Key Metrics and KPIs Discussion, Organizational Priorities and Focus Areas, Market Trends and Competitive Analysis, Resource Allocation and Headcount Planning, Operational Efficiency Review, Major Risks and Mitigation Planning, Executive Decisions and Approvals, Quarterly Business Review</td></tr><tr><td>Project</td><td>Project Kickoff and Scope Definition, Milestone Progress Review, Timeline and Deliverable Alignment, Resource and Budget Status, Stakeholder Communication Plan, Risk Management and Mitigation Discussion, Project Dependencies and Coordination, Quality Assurance and Testing Review, Project Retrospective and Lessons Learned, Next Phase Planning and Execution</td></tr><tr><td>Department</td><td>Department Strategy and Goals Review, Cross-Team Coordination Update, Department Resource Planning, Organizational Changes and Updates, Department Performance Metrics Review, Inter- departmental Collaboration Discussion, Budget and Resource Allocation Review, Department Culture and Team Building, Major Initiative Planning, Department Communication and Updates</td></tr></table>

Table 10: Meeting topics.
<table><tr><td>Meeting Type</td><td>Description</td><td>Minutes</td><td>Agenda</td><td>Empty Cal.</td><td>Frequency</td><td>Organizer</td><td>Substantiates</td></tr><tr><td>direct_report</td><td>employee and their manager</td><td>30%</td><td>30%</td><td>40%</td><td>weekly</td><td>manager</td><td>task progress, re- ports to, works on</td></tr><tr><td>team</td><td>employees that belong and lead à team</td><td>25%</td><td>25%</td><td>50%</td><td>monthly</td><td>team lead</td><td>task progress, works on, member of</td></tr><tr><td>task</td><td>employees collaborating on a task</td><td>30%</td><td>30%</td><td>40%</td><td>biweekly</td><td>any</td><td>task progress, works on</td></tr><tr><td>executive</td><td>employees that are execu- tives</td><td>50%</td><td>50%</td><td>0%</td><td>monthly</td><td>any</td><td>project progress, title</td></tr><tr><td>project</td><td>employees that work on tasks part of a project</td><td>50%</td><td>50%</td><td>0%</td><td>monthly</td><td>team lead of the project</td><td>project progress, works on, has task</td></tr><tr><td>department</td><td>employees that lead the department and the teams part of it</td><td>50%</td><td>50%</td><td>0%</td><td>monthly</td><td>department lead</td><td>has task, team be- longs to depart- ment, department belongs to com- pany</td></tr></table>

Table 11: Supported Meeting Types.

embeddings for retrieval.

## F QA Benchmarks

In the following section, we outline sample questions from three QA dataset types across the different companies they are generated from. In each example, we include the minimal set of documents required to review/analyze to be able to fully answer the presented question. There may be multiple minimal sets of documents to answer said questions, of which we select one. The selected questions are a small subset of all questions in the dataset and aim to show the diversity of questions, answers, and documents. Many questions require analyzing many documents (e.g. "Who are the employees of pound financial group llc?"), but these questions are omitted for the sake of conciseness.

For each question, we measure the number of constraints, which represents the number of filters on the documents or KB that an agent must apply to answer a question. However, this doesn't necessarily correlate with question difficulty, because low constraint count questions may be more difficult to answer due to their expansive breadth (see example in paragraph above).

## G KB QA

<table><tr><td>Question</td><td>Who participated in the meeting titled cross-team protocol alignment - work session?</td></tr><tr><td>Answer</td><td>James Moreira, James Vera</td></tr><tr><td>Company/KB</td><td>Streamvibe</td></tr><tr><td>Answer Type</td><td>List[str]</td></tr><tr><td>Constraints</td><td>1</td></tr></table>

calendar\_invite\_james\_vera\_f685.txt   
Subject: Cross-team protocol alignment -   
Work Session   
From: James Vera   
<vera.Jamie@streamvibestudios.com>   
To: James Vera   
<vera.Jamie@streamvibestudios.com>, James   
Moreira   
<Jim.moreira@streamvibestudios.com>   
Date: 2024-02-22   
CALENDAR INVITE   
You are invited to: Cross-team protocol   
alignment - Work Session   
Scheduled for: 2024-02-22   
Organizer: James Vera   
(vera.Jamie@streamvibestudios.com)   
Attendees:   
- James   
Vera (vera.Jamie@streamvibestudios.com)   
- James Moreira   
(Jim.moreira@streamvibestudios.com)   
This meeting has been scheduled by James   
Vera.   
Please accept or decline this invitation   
at your earliest convenience.  
The calendar invite clearly lists out all attendees:

James Vera, James Moriera. We judge these responses as sets, so ordering doesn't matter.

<table><tr><td>Question</td><td>Who organized the target company due diligence discussion meeting?</td></tr><tr><td>Answer</td><td>Birgitta Lagarde</td></tr><tr><td>Company/KB</td><td>Pound</td></tr><tr><td>Answer Type</td><td>str</td></tr><tr><td>Constraints</td><td>1</td></tr></table>

## H Topic QA

meeting\_minute\_Quality   
Review\_and\_Standards\_Alignment\_07a7.txt   
Message-ID:   
61f61ba7-8cd2-4a17-a311-6cdfd059f7bc   
From: Birgitta Lagarde   
<blagarde@poundfinancialgroupllc.com>   
To: Piermaria Ferguson   
<pferguson@poundfinancialgroupllc.com>   
Date: 2024-01-03   
Subject: Target company due diligence   
Discussion   
MIME-Version: 1.0   
Content-Type: text/plain; charset=UTF-8   
Content-Transfer-Encoding: 7bit   
Piermaria,   
Thanks for making time to sync up on the   
target company due   
diligence work we've got going. I wanted   
to recap what we covered in our last   
meeting and make sure we're both tracking   
the same priorities as we move forward   
with this.   
We spent some good time going through our   
quality review process and making sure   
our standards are aligned across the   
board. We talked through the key   
documentation we need to pull together,   
identified some gaps in our initial   
assessment, and went over what the next   
phase of verification looks like. It was   
helpful to lock in on our approach and   
get clear on who's handling what pieces.   
Here's where things stand with our   
progress:   
Target company due   
diligence is advancing steadily with you   
and I, around 30-40% finished.   
I think we're in a solid spot to keep   
pushing forward on this. Let's keep the   
momentum going and touch base again soon   
on any blockers or adjustments we need to   
make.   
Thank you,   
Birgitta   
Birgitta Lagarde   
Analyst   
Pound Financial Group LLC   
blagarde@poundfinancialgroupllc.com   
Desk: +1 (650) 925-3434   
Mobile: +1 (650) 205-2167   
[INSERT\_BOX\_LOGO]

The meeting minutes document implicitly indicates that Birgitta Lagarde, as the organizer, is sending meeting minutes to the sole other attendee Piermaria Ferguson.

<table><tr><td>Question</td><td>When was drug approval first men- tioned?</td></tr><tr><td>Answer</td><td>2024-01-01</td></tr><tr><td>Company/KB</td><td>Biocure</td></tr><tr><td>Answer Type</td><td>date</td></tr><tr><td>Constraints</td><td>1</td></tr></table>

email\_Label\_Negotiation\_Process\_546e.txt   
Message-ID:   
3ce33e9f-3d3c-4100-bf30-b2fc26cbbd05   
From: Jeronimo Zobel   
<jeronimo.zobel@yahoo.com>   
To: Manuella Barrios <manuella@gmail.com>   
Date: 2024-01-01   
Subject: Update on drug   
approval labeling coordination   
MIME-Version: 1.0   
Content-Type: text/plain; charset=UTF-8   
Content-Transfer-Encoding: 7bit   
Hi Manuella, I wanted to touch base on a   
couple of items related to our business   
operations and the drug   
approval process. As we continue managing   
our labeling requirements across various   
products, I've been thinking through how   
we can streamline our approach to the   
label negotiation process with regulatory   
bodies. The negotiations have been   
progressing, but I believe we could   
benefit from a more structured timeline   
and clearer documentation of our   
discussions with the FDA.   
Recently, manuella, checking in with you   
as my direct supervisor, and I wanted to   
ensure we're aligned on priorities for   
the labeling requirements phase. The   
label negotiation process is becoming   
more complex as we handle multiple   
products simultaneously, and having   
direct oversight will help us maintain   
consistency across submissions. I think   
it would be valuable to establish   
checkpoints where we review our   
negotiation strategy and any feedback   
we've received from regulatory reviewers.   
I'd like to schedule a brief meeting this   
week to discuss our current bottlenecks   
in the drug   
approval process, particularly around the   
labeling timeline. Once we optimize the   
label negotiation process, I believe we   
can accelerate our overall business   
operations in this area. Let me know your   
availability and if there are specific   
concerns you'd like to address.   
Thank you,   
Jeronimo Zobel   
Jeronimo Zobel   
Systems Administrator   
+1 (650) 682-8707 I   
jeronimo.z@biocuresolutions.com

The email is the first mention of Drug Approval. The topic hierarchy for this email is as follows: (0) Label Negotiation Process, (1) Labeling Requirements, and (2) Drug Approval. See Appendix C for a more detailed explanation on topic levels. The higher-level topics are interwoven into the main content of the email, which is centered around the more granular topic of Label Negotiation Process.

## I Integrated QA

<table><tr><td>Question</td><td>Did anyone who works at Zenith Labs send emails about Daily commute issues during month 1 of 2024?</td></tr><tr><td>Answer</td><td>true</td></tr><tr><td>Company/KB</td><td>Zenith</td></tr><tr><td>Answer Type</td><td>bool</td></tr><tr><td>Constraints</td><td>4</td></tr></table>

Message-ID:   
f57b61b0-4543-423d-9172-e8f1753abb7f   
From: Albert Garcia   
<Bert.garcia@zenithlabs.com>   
To: Emilia Anglada <eanglada@hotmail.com>   
Date: 2024-01-10   
Subject: Morning commute struggles   
MIME-Version: 1.0   
Content-Type: text/plain; charset=UTF-8   
Content-Transfer-Encoding: 7bit

Message-ID:   
54b67748-00c1-4d5c-b996-62c9e0241545   
From: Emilia   
Anglada <eanglada@hotmail.com>   
To: Jared   
Chartier <jaredchartier@gmail.com>   
Date: 2024-01-10   
Subject: Addressing code quality concerns   
in our operations   
MIME-Version: 1.0   
Content-Type: text/plain; charset=UTF-8   
Content-Transfer-Encoding: 7bit   
Hi Jared, I've been reflecting on how our   
current business operations could benefit   
from a more intentional approach to   
managing technical   
debt. As our codebase continues to grow,   
I'm noticing that we're accumulating   
layers of complexity that might slow down   
our team's ability to iterate   
efficiently. I think it would be valuable   
for us to establish a clearer strategy   
around this.   
Building on that concern, Jared, I wanted   
to get your direction here on how we   
should prioritize which areas to tackle   
first. I'm thinking we could start by   
identifying the components that have the   
highest maintenance burden and then work   
through a systematic refactoring plan.   
Would you be open to discussing this   
further so we can align on the best path   
forward for our technical infrastructure?   
Best,   
Emilia Anglada   
Marketing Specialist   
+1 (650) 524-2805

## email\_Daily\_commute\_issues\_5201.txt

I wanted to check in since we've both been navigating some rough commute situations lately. The traffic on my usual route has been absolutely brutal the past couple of weeks, and I've been leaving earlier just to make it to the office on time. I know you mentioned something similar last week when we were chatting by the desks.

I've been thinking about how these daily commute issues really affect our whole day and energy levels at work. When you're stuck in traffic or dealing with transit delays, it's hard to come in feeling fresh and ready to tackle complex projects. Building on that, accepted the firmware performance benchmarking role this morning, so I've been reflecting on how important it is to stay focused even when the morning doesn't go as planned.

I've actually started trying out a different route a couple times this week, and it's made a noticeable difference on days when the main highway is congested. Leaving just fifteen minutes earlier has been a game-changer for me, and it gives me time to grab coffee and settle in before diving into emails. Maybe we could compare notes on what's been working for each of us?

Anyway, just wanted to reach out and see if you've found any good solutions to the commute chaos. It seems like a lot of us in the office have been dealing with similar frustrations, so I figured it might be worth discussing what strategies have helped. Hope things smooth out for you soon!

In this Email, the sender Albert Garcia can be identified as a Zenith Labs employee by his signature or his email domain. The topic of the email is Daily commute issues and it was sent on 2024-01-10, which falls in the date range of the question.

<table><tr><td>Question</td><td>How many of jared chartier&#x27;s direct re- ports sent emails about technical debt management after 29 Mar 2024?</td></tr><tr><td>Answer</td><td>0</td></tr><tr><td>Company/KB</td><td>Zenith</td></tr><tr><td>Answer Type</td><td>int</td></tr><tr><td>Constraints</td><td>3</td></tr></table>

## email\_Technical\_Debt\_Management\_72dc.txt

This is the only email about Technical Debt Management that a direct report of Jared Chartier has sent. Due to the size of the Zenith Labs company, Emilia Anglada is also the only direct report. The email was sent on January 10, 2024, so it doesn't fall within the date range of the question. Also, the mention of "I want to get your direction here" serves as an implicit mention of Emilia reporting to Jared.

## J QA Dataset Assumptions

We explicitly state the following assumptions and details we were using to create our QA ensure clarity and proper use:

• Ground truth answers are obtained through manually-verified SPARQL queries executed on the underlying knowledge base. This verification process ensures that the correct answers are returned regardless of which template variables are instantiated.

• All questions requesting lists (e.g., "Who are all the employees of X?") return results without duplicate entries. This deduplication is applied consistently across the dataset.

• Questions exclusively target regular email topics and intentionally exclude meeting-related topics, as these constitute distinct semantic categories with different structural properties.

• Due to the scale of the Pound knowledge base, minimally-constrained questions often yield extremely large answer sets. For example, the query "Who is an employee of Pound?" returns several thousand names, potentially exceeding typical LLM context window limits. In contrast, questions with additional constraints produce more manageable, specific answer sets that are better suited for LLM agent processing.

• The answer key generation process applies SELECT DISTINCT operations to remove duplicate names from result sets. Consequently, a minor discrepancy may exist between countbased answers and the actual length of returned name lists, as overlapping names are consolidated into single entries.

## K Experimental Setup

## KB Evaluation

• Datasets: All documents from the four company datasets from Table 1.

• Models: Claude Haiku 4.5, GPT-5 Nano, Gemini 2.5 Flash Lite.

• Prompting: Few-shot with 8 examples (prompt template in Appendix P). We selected samples to properly cover diverse entity/relationship types when providing examples.

• Entity Evaluation: Per-type precision, recall, and $F _ { 1 }$ , macro-averaged across types. TP: matched entities; FP: spurious extractions; FN: missed ground truth entities.

• Relationship Evaluation: Dual evaluation measuring (1) structural correctness (triple exists) and (2) temporal accuracy (triple with correct dates). Per-type metrics macro-averaged across relationship types. Temporal evaluation requires both correct triple structure and matching start/end dates.

• Implementation: Temperature=1.0, max tokens=8,192, exponential backoff retry (max 3 attempts, 60s timeout).

## Topic Classification

• Datasets: Biocure (L, 31 topics) and Pound (XL, 17 topics) from Table 1. Topic definitions in Table 15.

• Models: Claude Haiku 4.5, Claude Sonnet 4.5, GPT-5 Nano, GPT-5.1, and Gemini 2.5 Flash Lite.

• Baseline: TF-IDF features with Logistic Regression classifier

• Train/Test Split: Stratified sampling with k ∈ {0, 10, 50} training examples for LLMs and k ∈ {10, 50, 100, 1000, 10000, 100000} for baseline. Test set: 1,000 examples per configuration (or maximum available). 5 repetitions with different random splits.

• Prompting: Zero-shot (k = 0) or few-shot (k > 0) examples per topic (template in Appendix Q).

• Evaluation: Per-topic precision, recall, and F1, macro-averaged across topics.

• Implementation: Temperature=1.0, max tokens=500, exponential backoff retry (max 3 attempts, 60s timeout).

## QA

• Datasets: KB QA, Topic QA, and Integrated QA. 250 Questions \* 4 KB sizes \* 3 QA Datasets = 3k questions. Single full dataset run.

• Models: Claude Haiku 4.5, Claude Sonnet 4.5, GPT-5 Nano, GPT-5.1, and Gemini 2.5 Flash Lite. Used through Anthropic API, OpenAI API, and OpenRouter respectively.

• Methods: RAG (k = 10): Agent with access to a tool to query a vector db. KB: Agent with access to a tool to query a Postgres DB with SQL. DBs contain ground truth information extracted from the KB and follow a standard relational schema (not relevant for RAG Agent). Database access is restricted for each dataset. Restriction applies based on KB, method, and dataset to ensure no data leakage happens: e.g. when answering questions in Topic QA for Biocure, the KB agent will exclusively have access to the table that stores information on the documents from Biocure.

• Evaluation: Case-insensitive exact match (string, date, integer, boolean) scoring 0/1 and $F _ { 1 }$ score for set[string] questions to allow models to obtain partial credits (as these questions are typically the hardest to answer). For boolean we accepted alternative wording such as "Yes" and "No".

• Implementation: Pydantic AI agent with custom tools to retrieve from Vector DB (RAG) or Postgres DB (KB). Tool call limit of 5. Few shot examples (specific to each QA dataset) were provided for each prompt.

## L QA Results

In this section, we visualize the results we have for QA in graphs. Figure 7 shows the QA results by model and Figure 8 shows the QA results by method.

## M KB Evaluation Results

Table 12 reveals distinct error patterns across relationship types and models. For non-temporal relationships, Gemini 2.5 Flash Lite exhibits the highest over-prediction rate for worksOn (69%), while all models show substantial under-prediction of attend relationships (63–71% FN rates). Temporal relationship extraction proves more challenging: Gemini 2.5 Flash Lite over-predicts worksOn at 56%, while Haiku 4.5 shows notable overprediction of attend (45%) and organize (25%). Under-prediction of attend remains problematic across temporal contexts (50–54% FN rates for all models). These patterns suggest that models struggle with event-based relationships (attend, organize) regardless of temporal context, while structural relationships (worksOn, memberOf) are prone to over-prediction, particularly for Gemini 2.5 Flash Lite. The relatively balanced error rates for core relationships like reportsTo and worksAt (mostly < 10%) indicate better model performance for hierarchical organizational structures.

## N Topic Classification Results

Table 13 presents topic classification performance across models and training sizes. On Biocure (L), all models achieve strong zero-shot performance (F1: 0.872–0.950), with minimal improvement in few-shot settings (50-shot F1: 0.877–0.982). The TF-IDF+LR baseline demonstrates clear data scaling, reaching F1 of 0.993 with 10K examples – comparable to LLM zero-shot performance but requiring substantially more training data. On Pound (XL), LLMs show moderate zero-shot performance (F1: 0.357–0.531) with notable few-shot improvements (50-shot F1: 0.598–0.751). The data-rich baseline achieves superior performance at 100K examples (F1: 0.983), outperforming all LLMs.

## O QA Error Analysis

In total, we ran 250 questions across 5 LLM models for 4 companies in 3 QA settings with 2 methods (RAG and KB). Thus, we had 30,000 total model runs. Table 14 shows the programmatic errors that we ran into for each model in each setting (Topic QA, KB QA, and Integrated QA).

Our analysis reveals substantial variation in error rates across models, with GPT-5.1 experiencing the highest failure rate (247 errors, 4.1% of its runs) and Sonnet 4.5 showing the most robust performance (91 errors, 3.0% of its runs). The total error count across all models was 697 out of 30,000 runs, representing a 2.3% overall failure rate.

KB-based queries produced 4.7× more errors than RAG-based queries (547 vs. 150 errors). This disparity suggests that knowledge base integration introduces additional failure modes, particularly related to tool calling and output validation. The complexity of KB queries—which require models to navigate structured data and execute multiple tool calls— contributes to this elevated error rate.

Error rates exhibit a clear positive correlation with query complexity (S→M→L→XL). For KB queries, XL-scale questions generated 175 errors compared to just 107 errors for S-scale questions. This pattern is particularly pronounced for context length issues: "Prompt too long" errors occur almost exclusively at L and XL scales, accounting for 72 such errors. This suggests that as queries grow in complexity, models struggle with context

![](images/899bf27b965d841ca38ed0b04bc3ff85ccd9f71fced62c783313d1e17588a9fc.jpg)

Figure 7: Visualization of results presented in Table 4.  
![](images/ab698f602ed6534c43a0c40df7357c7de1282ae587b50b6d0d5caf13e2acd650.jpg)

Figure 8: Visualization of results presented in Table 4 aggregated by method.
<table><tr><td>Error Type</td><td>Relationship</td><td>Gemini 2.5 FL</td><td>Haiku 4.5</td><td>GPT-5 Nano</td></tr><tr><td colspan="5">Non-Temporal Relationships</td></tr><tr><td rowspan="5">Over-predicts (FP)</td><td>worksOn</td><td>69%</td><td>64%</td><td>43%</td></tr><tr><td>worksAt</td><td>7%</td><td>一</td><td>3%</td></tr><tr><td>reportsTo</td><td>8%</td><td></td><td>4%</td></tr><tr><td>memberOf</td><td>16%</td><td>12%</td><td>50%</td></tr><tr><td>attend organize</td><td>一 一</td><td>16% 8%</td><td></td></tr><tr><td rowspan="5">Under-predicts (FN)</td><td>worksOn</td><td>8%</td><td></td><td></td></tr><tr><td>worksAt</td><td>5%</td><td>15%</td><td>16%</td></tr><tr><td>reportsTo</td><td>1%</td><td>6%</td><td>4%</td></tr><tr><td>memberOf</td><td>3%</td><td>3%</td><td>3%</td></tr><tr><td>attend</td><td>68%</td><td>71%</td><td>63%</td></tr><tr><td></td><td>organize</td><td>15%</td><td>5%</td><td>14%</td></tr><tr><td colspan="5">Temporal Relationships</td></tr><tr><td rowspan="6">Over-predicts (FP)</td><td>worksOn</td><td>56%</td><td>23%</td><td>43%</td></tr><tr><td>worksAt</td><td>4%</td><td>1%</td><td>16%</td></tr><tr><td>reportsTo</td><td>21%</td><td></td><td>4%</td></tr><tr><td>memberOf</td><td>19%</td><td>6%</td><td>37%</td></tr><tr><td>attend</td><td>一</td><td>45%</td><td></td></tr><tr><td>organize</td><td>一</td><td>25%</td><td>一 一</td></tr><tr><td rowspan="4">Under-predicts (FN)</td><td>worksOn</td><td>31%</td><td>34%</td><td>36%</td></tr><tr><td>memberOf</td><td>3%</td><td>3%</td><td>3%</td></tr><tr><td>attend</td><td>54%</td><td>52%</td><td>50%</td></tr><tr><td>organize</td><td>12%</td><td>11%</td><td>11%</td></tr></table>

Table 12: Error Distribution by Relationship Type and Model

<table><tr><td rowspan="2">Model</td><td rowspan="2">Training Size</td><td colspan="3">Biocure (L)</td><td colspan="3">Pound (XL)</td></tr><tr><td>Precision</td><td>Recall</td><td> $\mathbf { F } _ { 1 }$ </td><td>Precision</td><td>Recall</td><td> $\mathbf { F } _ { 1 }$ </td></tr><tr><td rowspan="3">Haiku 4.5</td><td>0</td><td>.917±.00</td><td> $. 8 8 7 \pm . 0 1$ </td><td> $. 8 8 5 { \pm . 0 1 }$ </td><td>.550±.01</td><td>.410±.02</td><td> $. 3 5 7 \pm . 0 1$ </td></tr><tr><td>10</td><td>.950±.01</td><td>.959±.01</td><td> $. 9 5 1 { \pm } . 0 1 $ </td><td>.649±.01</td><td>.650±.01</td><td> $. 5 9 4 \pm . 0 1$ </td></tr><tr><td>50</td><td>.963±.01</td><td>.967±.00</td><td>.963±.01</td><td>.697±.01</td><td>.733±.01</td><td> ${ \bf . 6 8 5 \pm . 0 1 }$ </td></tr><tr><td rowspan="3">Sonnet 4.5</td><td>0</td><td>.875±.04</td><td>.873±.04</td><td>.872±.04</td><td>.583±.01</td><td>.544±.02</td><td>.517±.02</td></tr><tr><td>10</td><td>.963±.00</td><td>.966±.01</td><td>.962±.01</td><td>.681±.03</td><td>.692±.03</td><td>.661±.03</td></tr><tr><td>50</td><td>.979±.00</td><td>.982±.00</td><td>.980±.00</td><td>.748±.02</td><td>.777±.02</td><td>.751±.02</td></tr><tr><td rowspan="3">GPT-5 Nano</td><td>0</td><td>.905±.00</td><td>.901±.01</td><td>.898±.01</td><td>.626±.00</td><td>.563±.01</td><td>.531±.01</td></tr><tr><td>10</td><td>.897±.01</td><td>.898±.02</td><td>.893±.02</td><td>.661±.01</td><td>.583±.01</td><td>.585±.00</td></tr><tr><td>50</td><td>.878±.03</td><td>.884±.03</td><td>.877±.03</td><td>.642±.01</td><td>.615±.01</td><td>.598±.01</td></tr><tr><td rowspan="3">GPT-5.1</td><td>0</td><td>.954±.02</td><td>.952±.03</td><td>.950±.03</td><td>.620±.01</td><td>.524±.01</td><td>.483±.00</td></tr><tr><td>10</td><td>.975±.00</td><td>.979±.00</td><td>.977±.00</td><td>.709±.02</td><td>.721±.02</td><td>.684±.02</td></tr><tr><td>50</td><td>.981±.00</td><td>.985±.00</td><td>.982±.00</td><td>.762±.01</td><td>.778±.01</td><td>.748±.02</td></tr><tr><td rowspan="3">Gemini 2.5 F-L</td><td>0</td><td>.938±.01</td><td>.931±.01</td><td>.929±.01</td><td>.563±.02</td><td>.456±.02</td><td>.412±.02</td></tr><tr><td>10</td><td>.826±.01</td><td>.841±.00</td><td>.831±.01</td><td>.609±.01</td><td>.656±.02</td><td>.620±.02</td></tr><tr><td>50</td><td>.873±.01</td><td>.889±.01</td><td>.879±.01</td><td>.669±.01</td><td>.713±.01</td><td>.682±.01</td></tr><tr><td rowspan="6">TF-IDF</td><td>10</td><td>.036±.05</td><td>.122±.11</td><td>.052±.06</td><td>.050±.07</td><td>.101±.09</td><td>.046±.08</td></tr><tr><td>50</td><td>.169±.09</td><td>.114±.08</td><td>.090±.07</td><td>.264±.02</td><td>.279±.03</td><td>.226±.03</td></tr><tr><td>100</td><td>.379±.02</td><td>.194±.02</td><td>.206±.02</td><td>.295±.01</td><td>.388±.01</td><td>.332±.01</td></tr><tr><td>1000</td><td>.969±.02</td><td>.934±.02</td><td>.942±.02</td><td>.900±.01</td><td>.677±.02</td><td>.709±.03</td></tr><tr><td>10000</td><td>.994±.00</td><td>.993±.00</td><td>.993±.00</td><td>.970±.01</td><td>.949±.01</td><td>.958±.01</td></tr><tr><td>100000</td><td></td><td></td><td></td><td>.986±.00</td><td>.981±.00</td><td>.983±.00</td></tr></table>

‡ Gemini 2.5 Flash Lite.  
management and multi-step reasoning.

Table 13: Topic Classification Results.

<table><tr><td></td><td></td><td colspan="3">RAG</td><td colspan="4">KB</td><td></td></tr><tr><td>Model</td><td>Error Type</td><td>S</td><td>M</td><td>L</td><td>XL</td><td>S</td><td>M</td><td>L</td><td>XL</td><td>Total</td></tr><tr><td rowspan="4">Haiku 4.5</td><td>Prompt too long</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td><td>8</td><td>22</td><td>32</td></tr><tr><td>Tool calls limit exceeded</td><td>0</td><td>0</td><td>0</td><td>0</td><td>19</td><td>15</td><td>26</td><td>24</td><td>84</td></tr><tr><td>Tool max retries exceeded</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>3</td></tr><tr><td>Total Errors</td><td>0</td><td>0</td><td>0</td><td>2</td><td>19</td><td>15</td><td>37</td><td>46</td><td>119</td></tr><tr><td rowspan="6">Sonnet 4.5</td><td>Output validation failed</td><td>0</td><td>6</td><td>9</td><td>1</td><td>0</td><td>11</td><td>10</td><td>1</td><td>38</td></tr><tr><td>Prompt too long</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td><td>2</td><td>36</td><td>40</td></tr><tr><td>Request timeout</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>2</td><td>5</td></tr><tr><td>Tool calls limit exceeded</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>3</td><td>3</td><td>7</td></tr><tr><td>Tool max retries exceeded</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>Total Errors</td><td>0</td><td>6</td><td>9</td><td>3</td><td>0</td><td>12</td><td>19</td><td>42</td><td>91</td></tr><tr><td rowspan="6">GPT-5 Nano</td><td>Empty model response</td><td>2</td><td>2</td><td>1</td><td>4</td><td>3</td><td>13</td><td>7</td><td>0</td><td>32</td></tr><tr><td>Other API errors</td><td>0</td><td>0</td><td>0</td><td>6</td><td>0</td><td>1</td><td>0</td><td>2</td><td>9</td></tr><tr><td>Request timeout</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Server error (5xx)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Tool calls limit exceeded</td><td>0</td><td>0</td><td>2</td><td>4</td><td>18</td><td>15</td><td>26</td><td>20</td><td>85</td></tr><tr><td>Total Errors</td><td>2</td><td>2</td><td>3</td><td>14</td><td>21</td><td>29</td><td>33</td><td>24</td><td>128</td></tr><tr><td rowspan="7">GPT-5.1</td><td>Empty model response</td><td>0</td><td>0</td><td>19</td><td>26</td><td>39</td><td>36</td><td>69</td><td>23</td><td>212</td></tr><tr><td>Other API errors</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>3</td></tr><tr><td>Rate limit exceeded</td><td>0</td><td>6</td><td>4</td><td>0</td><td>4</td><td>11</td><td>0</td><td>0</td><td>25</td></tr><tr><td>Server error (5xx)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Tool calls limit exceeded</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>1</td><td>0</td><td>2</td><td>5</td></tr><tr><td>Unknown error</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Total Errors</td><td>0</td><td>6</td><td>24</td><td>26</td><td>45</td><td>48</td><td>69</td><td>29</td><td>247</td></tr><tr><td rowspan="5">Gemini 2.5 Flash Lite</td><td>Invalid API response</td><td></td><td></td><td></td><td></td><td></td><td></td><td>3</td><td>4</td><td>12</td></tr><tr><td>Output validation failed</td><td>0 2</td><td>1 1</td><td>1 2</td><td>0 0</td><td>1 19</td><td>2 23</td><td>10</td><td>25</td><td>82</td></tr><tr><td>Tool calls limit exceeded</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>5</td><td>5</td><td>5</td><td>17</td></tr><tr><td>Unknown error</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Total Errors</td><td>2</td><td>2</td><td>3</td><td>0</td><td>22</td><td>31</td><td>18</td><td>34</td><td>112</td></tr></table>

Table 14: Error analysis across all models.

# Business Document Knowledge Extraction System

## P KB Evaluation Prompt

The following is the extraction prompt used for the KB Evaluation task.

You are an expert knowledge extraction system.   
Your task is to analyze business documents   
and extract structured information about   
entities (people, organizations, projects,   
tasks, meetings) and the relationships   
between them.

\*\*CRITICAL\*\*: Return ONLY valid JSON with no   
explanations, preambles, or markdown   
formatting. The response must be parseable   
JSON that exactly matches the schema provided.   
→

## ## Step 1: Document Classification

Before extracting any information, carefully   
read the entire document and classify it as   
ONE of the following types:

- \*\*\`agenda\`\*\*: An email outlining topics and   
agenda for an upcoming meeting   
- \*\*meeting\_minute\`\*\*: An email summarizing the   
discussions and outcomes of a past meeting   
- \*\*\`regular\_email\`\*\*: An email message between   
employees NOT discussing topics related to   
upcoming or past meetings

Store this classification as \`document\_type in   
your response.

## ## Step 2: Entity Extraction

## ### Entity Extraction Rules

1. \*\*Extract ONLY entities that participate in   
relationships\*\* - no orphaned entities. For   
instance, if the relationships are "Employee   
worksOn Task" and "Employee worksAt Company"   
then ONLY extract entities Employee, Task,   
and Company, nothing else. Do the same for   
each relationship.   
2. Extract entities exactly as they appear in   
the document (preserve exact wording for   
aliases)   
3. For each entity, capture ALL variations/   
aliases found in the document (e.g., ["John   
Smith", "J. Smith", "John", "John S.", "Jonny   
↔ "] etc.)   
4. Entity aliases MUST be extracted exactly as   
they appear in the document (do not make up   
aliases)   
5. Email addresses are CRITICAL for entity   
linking - extract when available   
6. Entity types must be EXACTLY as specified   
below (case-sensitive)

7. If entity type is Employee, extract the job   
title,department, and\`teamif mentioned   
in the email body or in the email signature

## ### Allowed Entity Types by Document Type

```markdown
**If document_type is `regular_email` then
allowed entity types are:**
- Employee
- Company
- Department
- Team
- Task
- Project
**If document_type is agenda or
meeting_minute then allowed entity types are
all of the above plus:**
- Meeting
### Entity Type Definitions & Schemas
#### Employee
Individual people working at the company.
```json
{
"type": "Employee",
"name": "Full Name",
"email": "email@company.com",
"aliases": ["Full Name", "First Last", "
Nickname", "F. Last"],
"title": "Job Title",
"department": "Department Name",
"team": "Team Name"
7
#### Company
The organization itself or partner organizations.
```json
{
"type": "Company",
"name": "Company Name",
"aliases": ["Company Name", "CompanyName Inc",
↔"CN"]
}
#### Department
Organizational divisions (e.g., Flight
Operations, Ground Services, Maintenance,
Customer Service).
```json
{
"type": "Department",
"name": "Department Name",
"aliases": ["Department Name", "Dept Name", "
↔ DEPT"]
}
#### Team
Working groups within departments.
```json
{
```

- "Sarah is working on the fleet maintenance   
audit"

"type": "Team",   
"name": "Team Name",   
"aliases": ["Team Name", "Team", "Group Name"],   
"department": "Parent Department Name"

```jsonl
``json
{
"type": "Task",
"name": "Task Name",
"aliases": ["Task Name", "Task shorthand"],
"description": "Task description",
"status": "active OR completed",
"assigned_to": ["Employee Name 1", "Employee
Name 2″],
"project": "Associated Project Name"
}
```

```snap
```json
{
"type": "Project",
"name": "Project Name",
"aliases": ["Project Name", "Project Nickname",
"Initiative Name"],
"description": "Project description"
3
```

\`\`json   
{   
"type": "Meeting",   
"title": "Title of Meeting",   
"aliases": ["Meeting Title", "Alternative   
→ Title"],   
"date": "YYYY-MM-DD",   
"document\_type": "agenda OR meeting\_minute",   
"meeting\_type": "direct\_report OR team OR task   
OR executive OR project OR department"   
}

\*\*Meeting Type Classification:\*\*   
If the entity is of type Meeting, classify its   
meeting\_typeas ONE of the following based   
on the document content:   
- \`direct\_report: Meeting between an employee   
and their direct manager   
- \`team: Meeting among members of the same team   
- \`task\`: Meeting focused on a specific task   
- \`executive: Meeting involving executives or   
high-level management   
- project\`: Meeting focused on a specific   
project   
- \`department: Meeting involving members of the   
same department

1. Extract ONE principal relationship (the most   
important/central relationship in the   
document)   
{% block instructions\_relationship %}   
2. Extract multiple secondary relationships   
according to the criteria below   
{% endblock %}   
3. Relationship types must be EXACTLY as   
specified (case-sensitive)   
4. \*\*Temporality is REQUIRED\*\* - must be one of:   
start, end, or ongoing (never null)   
5. \*\*Date Rules:\*\*   
- If temporality is start: beginning must   
contain a valid date (YYYY-MM-DD), end is   
empty string   
- If temporality is end: end must contain   
a valid date (YYYY-MM-DD), beginning is   
empty string   
- If temporality is \`ongoing: both   
beginning and end are empty strings   
6. Dates must be derived from the document - DO   
NOT fabricate dates   
7. Extract mentions: direct quotes or   
paraphrases from the document that evidence   
this relationship

## ### Allowed Relationship Types by Document Type

\*\*If the document\_type is \`regular\_emailthen   
the allowed relationship types are:\*\*   
- worksOn   
- worksAt   
- reportsTo   
- memberOf   
- departmentBelongsToCompany (secondary only)   
- teamBelongsToDepartment (secondary only)

\*\*If the document\_type is agendaor   
meeting\_minute then the allowed relationship   
types are all of the above plus:\*\*   
- organize   
- attend   
- mentionsTask   
- mentionsProject   
- mentionsEmployee   
- mentionsTeam   
- mentionsDepartment

## ### Relationship Type Definitions & Extraction Criteria

\- "John has been assigned the gate scheduling optimization task"

## ---

## #### worksAt

\*\*Pattern\*\*: \`Employee worksAt Company

## \*\*Extract when:\*\*

\- The document mentions an employee works at a → company

\- The email sender uses a work email -> extract sender worksAt company

\- The email recipient uses a work email ->

extract recipient worksAt company

## \*\*Example mentions:\*\*

\- "As a flight coordinator at SkyWings Airlines

\- Email from: john.smith@skywings.com -> John

Smith worksAt SkyWings Airlines

## ---

## #### reportsTo

\*\*Pattern\*\*: \`Employee reportsTo Employee ( → manager)

## \*\*Extract when:\*\*

\- The document explicitly mentions reporting structure structure

##

\- Manager-employee language is used (e.g., "your

progress on...", "I'd like you to...", "

report back to me")

## \*\*Example mentions:\*\*

\- "As your manager, I wanted to check in on the crew scheduling..."

\- "Please report your findings to Sarah"

## ---

## #### memberOf

\*\*Pattern\*\*:\`Employee memberOf TeamOR Employee memberOf Department

## \*\*Extract when:\*\*

\- The document signature contains the employee's team or department

\- Collective language indicates team membership

(e.g., "our team's progress", "we in the

Ground Operations team")

\- Explicit membership statements

## \*\*Example mentions:\*\*

\- Email signature: "John Smith | Ground

Operations Team"

\- "Our team has been coordinating baggage

handling improvements..."

\#### departmentBelongsToCompany

\*\*Pattern\*\*:\`Department

departmentBelongsToCompany Company

## \*\*SECONDARY RELATIONSHIP ONLY\*\* - never the

principal relationship

\*\*Extract when:\*\*

\- You have a principal relationship \`Employee

memberOf Department AND the sender uses a

work email

\- Document signature contains department AND

company information

## \*\*Example mentions:\*\*

\- Email signature: "Flight Operations Department

| SkyWings Airlines"

##

## #### teamBelongsToDepartment

\*\*Pattern\*\*:\`Team teamBelongsToDepartment Department

## \*\*SECONDARY RELATIONSHIP ONLY\*\* - never the

principal relationship

## \*\*Extract when:\*\*

\- The document explicitly states a team is

within/part of a department

## \*\*Example mentions:\*\*

\- "The Crew Scheduling team within our Flight

Operations Department..."

##

## #### organize

\*\*Pattern\*\*: \`Employee organize Meeting

## \*\*Extract when:\*\*

\- Document type is agenda or \`meeting\_minute

\- The document identifies a meeting organizer/ ↔ host

## \*\*Example mentions:\*\*

\- "Organized by: Sarah Johnson"

\- "Meeting host: John Smith"

\- "From: Sarah Johnson"

##

## #### attend

\*\*Pattern\*\*: \`Employee attend Meeting

## \*\*Extract when:\*\*

\- Document type is agenda or \`meeting\_minute

\- Extract ONE relationship for EACH attendee listed

## \*\*Example mentions:\*\*

\- "Attendees: Sarah Johnson, Mike Chen, Emily

Rodriguez"

\- "To: Mike Chen, Emily Rodriguez"

##

## #### mentionsTask

\*\*Pattern\*\*:\`Meeting mentionsTask Task

## \*\*Extract when:\*\*

\- Document type is agenda or \`meeting\_minute

\- A task is discussed or referenced in the

→ meeting

\*\*Example mentions:\*\*

\- "Agenda item 2: fleet maintenance audit task"

"beginning": "YYYY-MM-DD or empty string",   
"end": "YYYY-MM-DD or empty string"   
#### mentionsProject }   
\*\*Pattern\*\*:Meeting mentionsProject Project   
\*\*Extract when:\*\*   
- Document type is \`agenda\` or meeting\_minute   
- A project is discussed or referenced in the ## Complete Output Schema   
meeting   
\`json   
\*\*Example mentions:\*\* {   
- "Discussion of Fleet Modernization project "document\_type": "regular\_email OR agenda OR   
timeline" meeting\_minute",   
"entities": [   
{   
"type": "Entity Type",   
#### mentionsEmployee "name": "Entity Name",   
\*\*Pattern\*\*:Meeting mentionsEmployee Employee "aliases": ["alias1", "alias2"],   
...additional fields based on entity type   
\*\*Extract when:\*\*   
- Document type is agenda or \`meeting\_minute }   
- An employee is mentioned in the meeting (who ],   
may not be an attendee) "relationships": [   
{   
\*\*Example mentions:\*\* "type": "Relationship Type",   
"Sarah's progress on the safety audit will be "source": "Source Entity Name",   
reviewed" "target": "Target Entity Name",   
"context": "Context from document",   
"mentions": ["mention1", "mention2"],   
"temporality": "start OR end OR ongoing",   
#### mentionsTeam "beginning": "YYYY-MM-DD or empty string",   
\*\*Pattern\*\*: \`Meeting mentionsTeam Team "end": "YYYY-MM-DD or empty string"   
}   
\*\*Extract when:\*\* 1   
- Document type is agenda or meeting\_minute   
- A team is discussed or referenced in the   
↔ meeting   
\*\*Example mentions:\*\*   
- "Ground Operations Team's efficiency metrics" ## Extraction Checklist   
Before finalizing your response, verify   
#### mentionsDepartment - [ ] Document type is correctly identified   
\*\*Pattern\*\*: \`Meeting mentionsDepartment - [ ] Exactly ONE principal relationship is   
Department extracted   
- [ ] One or many secondary relationships are   
\*\*Extract when:\*\* extracted as applicable   
- Document type is agenda or \`meeting\_minute - [ ] Extract a Meeting entity for agendaor   
- A department is discussed or referenced in the meeting\_minute documents   
meeting - [ ] All extracted entities participate in at   
least one relationship   
\*\*Example mentions:\*\* - [ ] All extracted relationships have a source   
"Flight Operations Department's Q4 performance and target entity present in the entities   
review" ↔ list   
- [ ] All relationship types are spelled EXACTLY   
as specified   
- [ ] All entity types are spelled EXACTLY as   
## Relationship Schema specified   
- [ ] Every relationship has a valid temporality   
json value (never null)   
{ - [ ] Date fields follow rules (populated for   
"type": "EXACT\_TYPE\_FROM\_ABOVE", start/end, empty for ongoing)   
"source": "Source Entity Name", - [ ] All dates are in YYYY-MM-DD format   
"target": "Target Entity Name", - [ ] All dates are extracted from the document   
"context": "1-2 sentence context explaining (not fabricated)   
this relationship from the document", - [ ] Mentions contain actual quotes/paraphrases   
"mentions": ["Direct quote 1 from document", " from the document   
Direct quote 2 from document"], - [ ] Aliases capture all variations found in   
"temporality": "start OR end OR ongoing", the document

blocks - just the raw JSON.

---

\- [ ] Email addresses are extracted when

\- [ ] Title must be always extracted for

or markdown

## ## EXAMPLES:

{{ few\_shot\_examples }}

## ## Your Task

Analyze the document below and return ONLY the

JSON output following the schema. No

explanations, no preambles, no markdown code

## DOCUMENT TO ANALYZE:   
Filename: {{ document\_filename }}   
Content:   
{{ document\_content }}   
RESPONSE:

## Q Topic Classification Prompt

The following is the extraction prompt used for the Topic Classification task.

You are an expert topic classification system.

## INSTRUCTIONS:

1. Return ONLY one of the provided labels, do

2. Do NOT generate an empty text as an answer,

```markdown
# Few-shot examples:
```

```markdown
## DOCUMENT TO ANALYZE:
```

```eml
Message-ID:d46bb20a-ef29-4936-bdc1-52cf20c9316a
From: Robert Lederer <
Rlederer@poundfinancialgroupllc.com>
To: Nath Miller <miller.nath@gmail.com>
Date: 2024-03-01
Subject: Strengthening our checks and balances
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 7bit
```

Hey Nath, I wanted to touch base on something

that's been on my radar lately. As we

continue to focus on our broader business

operations and how we're creating value

across our portfolio, I think there's a real

opportunity for us to tighten up our

governance oversight. On that note, I'm newly

working on subordination lien release

verification, and I'm seeing firsthand how

verification processes in place.

You know, when you zoom out and look at

portfolio management, a lot of it comes down

to having strong accountability mechanisms

that actually work. Right now, I'm thinking

through some ways we could improve how we

handle these verification workflows to make

sure nothing slips through the cracks. It's

not just about crossing boxes---it's about

building confidence in our processes.

I think the key is making sure we've got clear

ownership and tracking at every step. The

current approach is solid, but there's

definitely room to strengthen our controls

and make the handoffs smoother between teams.

Have you noticed any pain points on your end

that we could address?

this week to brainstorm some ideas. I'm

pretty fired up about leveling up our

accountability system, and I think your

perspective would be really valuable here.

Let me know what works for your schedule.

## Thank you,

Analyst, Pound Financial Group LLC

E: Rlederer@poundfinancialgroupllc.com

## ## LABELS:

\- Competitive Analysis

\- Exit Planning

\- Financial Analysis

\- Legal Compliance

\- Market Intelligence

\- Market Research

\- Network Development

\- Operational Assessment

\- Operational Improvement

\- Opportunity Screening

\- Portfolio Management

\- Relationship Building

\- Stakeholder Management

\- Strategic Planning

\- Technology Assessment

\- Transaction Structuring

## ## RESPONSE:

Portfolio Management

Classify the following document using one of the

labels provided (DO NOT REASON):

<table><tr><td>Biocure (L) Non-business</td><td>Pound (XL) Non-business</td></tr><tr><td>Approval Timeline Management Biomarker Identification Clinical Data Analysis Clinical Trial Planning Competitive Intelligence Distribution Channel Management Efficacy Evaluation FDA Communication Formulation Optimization Healthcare Provider Education Intellectual Property Strategy International Regulatory Coordination Key Opinion Leader Engagement Labeling Requirements Manufacturing Compliance Manufacturing Process Development Market Access Strategy Advisory Committee Preparation Partnership Collaborations Patient Assistance Programs Post Marketing Commitments Preclinical Research Pricing Negotiations Promotional Campaign Development Regulatory Strategy</td><td>Exit Planning Financial Analysis Legal Compliance Market Intelligence Market Research Network Development Competitive Analysis Operational Assessment Operational Improvement Opportunity Screening Portfolio Management Relationship Building Stakeholder Management Strategic Planning Technology Assessment Transaction Structuring</td></tr></table>

Table 15: Topic Categories for Biocure (L) and Pound (XL).

\## DOCUMENT TO ANALYZE:

{{ document\_content }}

\## LABELS:

{{ topics }}

\## RESPONSE:

## R Topic Categories

Table 15 shows the topic categories used for the topic classification task for Biocure (L) and Pound (XL) datasets.