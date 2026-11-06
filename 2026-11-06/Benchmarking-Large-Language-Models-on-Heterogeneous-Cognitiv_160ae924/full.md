# Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation

Ali Sarabadani <sup>a</sup> <sup>[0000-0002-3521-1401]</sup>, Kheirolah Rahsepar Fard <sup>a</sup> <sup>[0000-0001-5452-</sup> <sup>4596]</sup> and Tahereh Jeirani <sup>b</sup>

<sup>a</sup> Department of Computer Engineering and Information Technology, University of Qom, Qom, Iran.

<sup>b</sup> Master of Science in Positive psychology, Qom, Iran

a.sarabadani@stu.qom.ac.ir, Corresponding author: rahsepar@qom.ac.ir, jeirani1386@gmail.com

## KEYWORD

Large Language Models (LLMs); Cognitive Neuroscience; Knowledge Graphs (KGs); Multi Hop Reasoning; Benchmark; Hallucination; Biomedical Informatics; Graph Grounding.

## ABSTRACT

The advancement of large language models (LLMs) in biomedical and cognitive neuroscience domains is rapidly happened. However, their capacity to reliably reason over heterogeneous and multi-relational knowledge graphs (KGs) remains to be understand.

This research presents a benchmark of five state of the art LLMs-GPT 5.1, Gemini 2.0 Pro, Llama 3, Qwen, and BioGPT-across five diverse KGs: Drug Bank, NeuroSynth, Cognitive Atlas, Brain Map, and GDPKG LLM (Integrating Gene, Disease, and Pharmacogenomics Knowledge Graphs for Cognitive Neuroscience Using Large Language Models). A new multitier query dataset is introduced, designed to test both how deeply these models can reason and how factually accurate they are under two different conditions: KG ‑ aware (with graph grounding) and KG ‑ agnostic.

Inter annotator agreement was assessed using pairwise Cohen's Kappa and Fleiss' Kappa. For Cohen’s Kappa the coefficient κ = 0.781, range 0.753-0.807 and for Fleiss’ Kappa the coefficient is $\kappa = 0 . 7 7 .$ These values indicate substantial to almost perfect agreement and confirming the robustness of our annotation framework. Our results conclude that the following three conclusions: First, general purpose frontier models (GPT 5.1 and Gemini 2.0 Pro) consistently outperform domain specific models (BioGPT and Llama 3) on multi hop and highlevel reasoning tasks, yet they exhibit a pronounced performance degradation from single hop to multi hop queries, underscoring the inherent difficulty LLMs face when traversing relational chains across cross domain entities. Second, KG-aware execution reduced hallucination rates across all models. Also, general ‑ purpose models were less likely to hallucinate than do-main ‑ specific ones when they had access to structured knowledge. Third, contrary to our initial hypothesis, domain specific models did not surpass general purpose models even in specialized tasks

without KG grounding, suggesting that the breadth of pretraining data in frontier models confers an unexpected advantage.

In this research practical guidelines for selecting LLM-KG integration strategies based on task complexity and computational constraints are presented. Hybrid neuro ‑ symbolic architectures that combine domain ‑ general reasoning with domain ‑ specific knowledge are also recommended. Our benchmark lays the groundwork for future research and supports applications in personalized medicine, cognitive disorder diagnosis, and explainable AI.

## 1. Introduction

Large language models (LLMs) have changed the way we access and reason about scientific knowledge. Transformer based architectures (Vaswani et al., 2017) and large scale pretrained models such as GPT-style systems (Brown et al., 2020; Raffel et al., 2020) have demonstrated remarkable capabilities in natural language understanding, few shot learning, and compositional reasoning. Researchers are moving beyond open‑ domain QA to see how LLMs handle structured knowledge, like knowledge graphs. Knowledge graphs are essential in neuroscience and biomedical branch. Large‑ scale biomedical KGs are used for drug–target analysis (Wishart et al., 2018), disease–gene association modeling (Himmelstein et al., 2017), and multi‑ modal data integration (Zhang et al., 2025). In cognitive neuroscience, resources such as NeuroSynth (Yarkoni et al., 2011), Brain Map (Laird et al., 2005), and Cognitive Atlas (Poldrack et al., 2011) represent distinct but complementary knowledge modalities, ranging from activation co-ordinates to structured cognitive ontologies. However, these graphs differ in schema design, entity granularity, and relation semantics, making cross-resource reasoning non trivial. The integration of LLMs with KGs has emerged as a promising paradigm for neuro-symbolic systems (Garcez et al., 2015). Recent research has focused on using LLMs to translate natural language directly into structured queries like SPARQL or Cypher (D'Abramo et al., 2025; Ma et al., 2025). Other approaches combine retrieval-augmented generation with graph-based grounding mechanisms to reduce factual hallucination (Lewis et al., 2020; Baek et al., 2020). These methods work well in controlled settings, but their performance across different domain‑ specific knowledge graphs has not yet been explored. Most prior work focuses on single graph benchmarks or general-domain datasets rather than scientifically complex domains such as cognitive neuroscience. Cognitive neuroscience is a challenging domain for evaluating how LLMs interact with knowledge graphs. Queries often require multi-hop reasoning across hierarchical cognitive con-structs, brain regions, receptor pathways, and pharmacological mechanisms. For instance, tracking a cognitive process to its neural bases and then to drug effects needs inference across different types of relations. Although multi-hop reasoning has been studied in open-domain datasets such as HotpotQA (Yang et al., 2018) and Complex Web Questions (Talmor and Berant, 2018). These benchmarks do not match validation KGs. Consequently, models may produce plausible but unsupported reasoning paths (Creswell and Shanahan, 2022; Ji et al., 2023). Another central limitation of LLMs is hallucination, de-fined as generating factually unsupported or fabricated information (Ji et al., 2023). In biomedicine and neuroscience, when models hallucinate, they can misrepresent what's already known or even make up non‑ existent associations. Grounding strategies reduce hallucination risk. However few studies compare grounded (KG-aware) and ungrounded (KG-agnostic) regimes under an integrated evaluation. Such comparative analysis is essential to understand whether performance gains stem from internal parametric knowledge or from structured graph constraints. To address these gaps, we introduce a multi-KG benchmarking frame-work for evaluating LLMs on cognitive neuroscience and related biomedical knowledge graphs. The benchmark spans five heterogeneous resources: Drug Bank (Wishart et al., 2018), NeuroSynth (Yarkoni et al., 2011), Cognitive Atlas (Poldrack et al., 2011), Brain Map (Laird et al., 2005), and GDPKG-LLM (RahseparFard et al., 2025). We construct a stratified query set consisting of single-hop factual queries, multi-hop relational queries, and higher-level reasoning tasks requiring integration of multiple graph com-ponents. Crucially, we evaluate models under two complementary regimes: a KGaware mode in which LLMs translate natural-language questions into executable structured queries, and a KG-agnostic mode in which models answer directly from their internal representations. This dual‑ mode setup allows us assess translation, semantic matching, reasoning, and hallucination. By systematically comparing multiple frontier and domain-specific LLMs across heterogeneous KGs, we provide insights into the suit-ability of current models for neuro-symbolic cognitive neuroscience applications. The hypothesis is that frontier general purpose models would excel in surface level factual retrieval, domain specific models would demonstrate superior stability in complex reasoning tasks when provided with explicit graph grounding. Three experts independently coded hallucination and reasoning failures based on a qualitative error scheme. These results help build more LLM–KG pipelines for discovering knowledge, generating hypotheses, and reasoning structured scientific data.

## 2. Related Work

Research on large language models (LLMs) and structured knowledge graphs (KGs) spans several partially overlapping domains. It is included neural semantic parsing, graph-augmented generation, biomedical knowledge integration, and question answering benchmarks.

We propose a framework of knowledge-enabled LLM to solve the issue of few-shot retrieval, which adopts knowledge base to fully utilize the potential of LLMs in retrieval. We integrate KG-neuro-symbolic into the LLM approach via two knowledge enhancement modalities: i) leveraging KG's knowledge rendering ability, we generate a knowledge-enhanced prompt (KEP) to guide LLMs match queries and passages; ii) through KG's knowledge base, we expand the support set and patch the missing KG facts by performing cross-modal chain of reasoning (CoR). By incorporating knowledge and reasoning skills, our framework can improve the situation in few-shot passage and document retrieval, query passage similarity and text entailment, and text summarization with questions as passages. The extensive experimental results indicate that the proposed LLM-based solution with multi-modal knowledge augmentation is an efficient and effective strategy.

## 2.1. LLM-Based Structured Query Generation

As in prior studies, the LLM-based KG semantic parsing for natural language question answering (D'Abramo et al., 2025; Yin et al., 2016), and its application in deep learning systems (Srivastava et al., 2021) are summarized in this section. In traditional approaches, neural SPARQL parsing models that translate natural language questions into executable logical forms (Yin et al., 2016) were first introduced for KG semantic parsing. The deep neural models are further enhanced in the reinforcement learning-based neural SEQ-to-SEQ model and its hybrid COARSE-to-FINE architecture (Srivastava et al., 2021) to enable multi-hop inference and traverse graph paths. The early works mainly rely on multiple components in the pipeline to train neural models. However, the neural models highly rely on the selection of the KG schema and are usually unstable under different environments. Recent methods adopt the structure and features of KG to improve neural models for semantic parsing, e.g., the type/role-encoded word embedding, graph features, graph neural networks, graph-representation encoder, and multi-modality attention (Sarabadani et al., 2026). Moreover, a graph attention network is further introduced to understand the QA semantics ... The above methods heavily rely on the KG schema, resulting in instability and difficulty in performance optimization. The large number of parameters in the neural architecture and the computational complexity of learning are also challenges for these methods. Different incorporating external knowledge sources enables LLMs to reduce hallucination and improve factual grounding. Retrieval Augmented Generation (RAG) (Lewis et al., 2020) first retrieves relevant knowledge from external sources and subsequently generates an answer with a fitted LLM. However, all works only retrieve documents, which do not take full advantage of structured information such as gene-disease association. Recent works bring structures into the generation, such as retrieving triples, which requires the LLM to traverse a knowledge graph (Baek et al., 2020). The research on prompt-guided LLM based knowledge graph interaction is very promising, but to our knowledge, all works only retrieve general-domain knowledge or structures. Cross-KG reasoning and maintaining consistency across multiple knee structures are nontrivial.

## 2.2. Biomedical and Neuroscience Knowledge Graph

Expanding knowledge bases for LLM reasoning, such as integrating biomedical entities in heterogeneous graphs (Himmelstein et al., 2017), has been investigated. Still, academic resources such as Drug Bank (Wishart et al., 2018) are curated for scientific researchers and not designed for LLM-based natural language reasoning.

In the medical domain, LLM-assisted knowledge graph construction frameworks have recently emerged. CNKG (Sarabadani et al., 2026) designed a mathematical modeling framework for stroke-specific knowledge graph construction via LLM-based knowledge extraction. CNKG (Sarabadani et al., 2026) also proposed a dynamic knowledge graph framework that incorporates LLM reasoning for medical diagnosis and personalized treatment recommendation. GDPKG-LLM (RahseparFard et al., 2025) also extended this direction by integrating genes, disease, pharmacogenomics knowledge graphs for cognitive neuroscience. These approaches illustrate the feasibility of employing LLM with biomedical knowledge graph construction and reasoning pipelines.

On the other hand, in the area of cognitive neuroscience, there are many knowledge graphs with structured resources such as NeuroSynth (Yarkoni et al., 2011), Cognitive Atlas (Poldrack et al., 2011), and Brain-Map. These resources represent an extensive collection on brain activation patterns, cognitive ontologies, and experimental data, but there is no controlled study in the literature investigating how the LLM framework of a given architecture interacts with heterogeneous cognitive neuroscience KGs with structured queries.

## 2.3. QA Benchmarks and Evaluation Gaps

Open-domain multi-hop benchmarks such as Hot-potQA (Yang et al., 2018) and Complex Web Questions (Talmor and Berant, 2018) require compositional reasoning but are built primarily over unstructured documents or single knowledge sources. Graph-specific datasets such as LC-QuAD (Dubey et al., 2019) and MetaQA (Zhang et al., 2018) assess structured query generation but are restricted to single-graph environments. None of these benchmarks jointly evaluate multi-KG reasoning, structured query equivalence, hallucination behavior, and performance differences between grounded and ungrounded inference modes.

## 2.4. Discussion and Positioning.

Existing behavior benchmarks mainly focus on either structured query generation (Tsatsaronis et al., 2015), or knowledge graph construction (Sarabadani et al., 2026; RahseparFard et al., 2025) or application specific tasks such as medical inference and systems biology models. We are not aware of any current benchmark which jointly captures hierarchical subgraph structure generation, structured query equivalence, hallucination behavior, and performance differences between grounded and ungrounded inference modes. In addition, the well-known query benchmarks are restricted to single graph environments. Our significance lies in providing a system-level benchmark of the behavior of LLMs and KG construction in an automated multi-KG setting. We nearly exhaustively evaluate the current state-of-the-art frontier and domain specific LLMs on a variety ranks heterogeneous cognitive neuroscience and biomedical knowledge graphs under both KGaware and KG-agnostic regimes. By sparsely identifying the LLMs that are best suited for the downstream multi-hop subgraph construction while being minimal in hallucination propensity, we propose a principled and statistically robust method for evaluation and selection of LLMs for cross-domain use cases in the "do not match" validation KGs.

## 3. Material and Methods

To avoid bias, we implemented a multi-annotator framework with annotators blind to the model identities during the evaluation. We report Cohen's Kappa coefficient calculations as a measure of inter-rater reliability. We performed pairwise comparisons between each annotator. The Kappas were calculated to be 0.753 between Annotator 1 and Annotator 2, 0.807 between Annotator 1 and Annotator 3 and 0.784 between Annotator 2 and Annotator 3. This suggests a high level of agreement (Kappa > 0.73) between each pair of annotators; the mean Kappa across all pairwise discourses were 0.781 (SD = 0.027).

Subsequently, to assess overall agreement among all three annotators simultaneously beyond chance, we calculate Fleiss' Kappa for the multi-rater setting. The overall Fleiss Kappa coefficient was calculated as 0.77, further indicating strong concordance between the three experts over the entire set of annotations. A concise summary of these findings is presented at the beginning of the Results section. Three experts independently coded hallucination and reasoning failures based on a qualitative error scheme. Cohen's Kappa coefficient was evaluated on each possible pair of annotators; the overall mean pairwise Kappa is 0.77 (mean +/- SD: 0.77 +/- 0.10). We evaluate whether contemporary large language models (LLMs) can reliably function as structured interfaces to heterogeneous cognitive neuroscience knowledge graphs (KGs). The hypothesis is that frontier general purpose models would excel in surface level factual retrieval, domain specific models would demonstrate superior stability in complex reasoning tasks when provided with explicit graph g-rounding. Our experimental pipeline consists of five stages: (1) benchmark construction, (2) ground-truth extraction from authoritative KGs,

The hypothesis is that frontier general purpose models would excel in surface level factual retrieval, domain specific models would demonstrate superior stability in complex reasoning tasks. (3) dual-mode model evaluation (KG-aware and KG-agnostic), (4) quantitative and qualitative metric computation, and (5) statistical, error, and human analyses.

## 3.1. Knowledge Graph Environment

We use five KGs covering (NLP-based) concepts, relationships, and ontologies, specifically: Pharmacokinetic Drug Bank, NeuroSynth (term-region associations), Cognitive Atlas (cognitive ontology), Brain Map (neuroimaging metadata), and GDPKG-LLM (LLM-derived neuroscience KG). KGs are represented in RDF and hosted in a SPARQLenabled triple store (e.g., Blaze-graph/Virtuoso), with property-graph versions for compatible ontologies in Neo4j. A mapping layer links entity types, predicates, and other schema components across graphs.

## 3.2. Knowledge Graph Environment

We employ five complementary KGs: Drug Bank (pharmacology), NeuroSynth (term–region asso-ciations), Cognitive Atlas (cognitive ontology), Brain Map (neuroimaging metadata), and GDPKG-LLM (LLM-derived neuroscience KG). RDF-based KGs are deployed in a SPARQL-enabled triple-store (e.g., Blaze-graph/Virtuoso), while property-graph representations are hosted in Neo4j. A mapping layer documents entity types, predicates, and across graphs.

## 3.3. Benchmark Construction

Let the benchmark query set be defined as

$$
Q ~ = ~ \{ q _ { 1 } , q _ { 2 } , \dots , q _ { N } \} , ~ N = 1 5 0 . ~ ( 1 )
$$

Queries are partitioned into: single-hop (50), multi-hop (70), and high-level reasoning (30).

Each query qi is associated with: a gold structured query gi and canonical answer set Ai obtained by executing gi over the authoritative KG(s). Multi-KG queries use predefined union/intersection opera-tors. All ground-truth outputs undergo double expert validation (Cohen’s and Fleis’s).

## 3.4. Dual Evaluation Modes

Each model M is evaluated under two regimes.

KG-aware:

$$
\hat { A } ^ { ( M ) } = e x e c u t e ( \hat { g } ^ { ( M ) } ) ,\tag{2}
$$

neuro-symbolic environments. where $\mathbf { g } { \widehat { \mathbf { \Gamma } } } ( \mathbf { M } )$ is the generated SPARQL/Cypher query. KG-agnostic:

$$
\widehat { \mathrm { A } } _ { \iota } ^ { \mathrm { \Gamma } } ^ { ( \mathrm { M } ) } = \mathrm { M } ( \mathsf { q } _ { i } ) ,\tag{3}
$$

This separation isolates translation errors from inter-nal reasoning and hallucination behavior. Formal metric definitions

Exact Match:

$$
\mathrm { E M } ^ { ( M ) } = \frac { 1 } { \mathrm { N } } \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } \mathsf { \Omega } _ { \parallel } \mid \mathrm { \Omega } \big [ \widehat { \mathrm { A } } _ { \mathrm { \Omega } } ^ { ( \mathrm { M } ) } = \mathrm { A } _ { \mathrm { i } } \big ] .\tag{4}
$$

Precision and Recall (set-valued):

$$
\mathrm { P r e c i s i o n } _ { i } ^ { ( M ) } = \frac { \left| \widehat { \mathrm { A } } _ { \mathrm { 1 } } ^ { \mathrm { \scriptsize ~ ( M ) } } \cap \mathsf { A } _ { \mathrm { i } } \right| } { \left| \widehat { \mathrm { A } } _ { \mathrm { 1 } } ^ { \mathrm { \scriptsize ~ ( M ) } } \right| } ,
$$

$$
\mathrm { R e c a l l } _ { i } ^ { ( M ) } = \frac { \left| \widehat { \mathrm { A } } _ { \mathrm { 1 } } ^ { ( \mathrm { M ) } } \cap \mathrm { A } _ { \mathrm { i } } \right| } { \left| \mathrm { A } _ { \mathrm { i } } \right| } .\tag{5}
$$

Macro-F1:

$$
\mathrm { F } _ { 1 } ^ { ( M ) } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 2 P _ { i } R _ { i } } { P _ { i } + R _ { i } } .\tag{6}
$$

Parse ability:

$$
\mathrm { P a r s e R a t e } ^ { ( M ) } = \frac { 1 } { \mathrm { N } } \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } \mathsf { \Pi } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } \mathsf { I } ^ { ( \mathrm { M } ) } ) \mathrm { I } .\tag{7}
$$

Result equivalence:

$$
\mathrm { E q u i v R a t e } ^ { ( M ) } = \frac { 1 } { \mathrm { N } } \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } \mathbb { H } \ \Bigl [ \mathrm { E x e c u t e } \widehat { \big ( g _ { \mathrm { i } } ^ { \mathrm { ( M ) } } \big ) } = \big [ \mathrm { E x e c u t e } ( ( \mathrm { g _ { i } } ) \big ] .\tag{8}
$$

Hallucination rate:

$$
\mathrm { H a l l u c R a t e } ^ { ( M ) } = \frac { 1 } { \mathrm { N } } \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } \ | \Psi \ [ \exists c \notin \mathrm { K G } ] .\tag{9}
$$

Latency:

$$
\mathrm { L a t e n s y } ^ { ( M ) } = \frac { 1 } { \mathrm { N } } \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } t _ { i } ^ { ( M ) } .\tag{10}
$$

Statistical Analysis

We compute 95% confidence intervals using boot-strap resampling (10k samples):

$$
\mathrm { C I } _ { 9 5 \% } = \big [ \Theta ^ { \ast ( 0 . 0 2 5 ) } , \Theta ^ { \ast ( 0 . 9 7 5 ) } \big ] .\tag{11}
$$

Paired comparisons use the Wilcoxon signed-rank test:

$$
H _ { 0 } \colon \mathbb { F } _ { 1 i } ^ { ( M _ { 1 } ) } = \mathbb { F } _ { 1 i } ^ { ( M _ { 2 } ) } .\tag{12}
$$

Binary correctness uses McNemar’s test. Multiple comparisons are corrected via Benjamini–Hochberg:

$$
\begin{array} { r } { p ( k ) \leq \frac { k } { m } \alpha . } \end{array}\tag{13}
$$

## 3.5. Evaluation Pipeline

The algorithm of the pipeline is as follows:

Algorithm 1 Dual-Mode LLM Evaluation   
1: for each model M in models do   
2: for each question qi in benchmark do   
3: Ai ← get\_ground\_truth(qi)   
4: if mode = KG-aware then   
5: prompt ← build\_sparql\_prompt(qi)   
6: model\_output ← call\_model(prompt)   
7: query ←extract\_query(model\_output)   
8: if query = NO\_QUERY or invalid then   
9: predicted\_answer ← ∅   
10: kg\_result ← None   
11: else   
12: kg\_result ← execute(query)   
13: predicted\_answer ←synthesize(kg\_result)   
14: end if   
15: else   
16: prompt ← build\_direct\_prompt(qi)   
17: model\_output ← call\_model(prompt)   
18: predicted\_answer ← model\_output   
19: query ← None   
20: kg\_result ← None   
21: end if   
22: metrics ← evaluate (predicted\_answer, Ai)   
23: log artifacts   
24: end for   
25: end for

## 3.6. Error and Human Evaluation

Errors are categorized into: (1) translation errors, (2) reasoning failures, (3) hallucinations, (4) schema mis match, (5) ambiguity.

For each model, 30 failure cases are manually annotated.

Human evaluation (n=150 responses) assesses correctness (3-point scale), usefulness, and hallucination presence. Inter-annotator agreement is measured via Cohen’s κ.

## Reproducibility and Ethics

We release benchmark queries, prompts, evaluation scripts, Docker environment, random seeds, and KG access instructions. All prompts, model outputs, and configurations are logged. Licensing constraints (e.g., Drug Bank) are documented. Model outputs are not in-tended as clinical advice

## 4. Results

Before reporting the primary performance, the reliability of qualitative annotations are revis-ited.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Parseable(%)</td><td rowspan=1 colspan=1>Equivalence(%)</td><td rowspan=1 colspan=1>EM Simple(%)</td><td rowspan=1 colspan=1>EM Multi-hop(%)</td><td rowspan=1 colspan=1>EM Reasoning(%)</td><td rowspan=1 colspan=1>Avg F1</td></tr><tr><td rowspan=1 colspan=1>GPT-5.1</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>91</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=1 colspan=1>Gemini 2.0Pro</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>Llama 3</td><td rowspan=1 colspan=1>87</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>86</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1>BioGPT</td><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>63</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>0.65</td></tr></table>

Table 1. Performance in KG-aware mode

In Section 3, inter-rater agreement was assessed using both pairwise and multi-rater indices. The pairwise Cohen's Kappa coefficients ranged from 0.753 to 0.807 (mean κ = 0.781), and the overall Fleiss' Kappa was 0.77. These values are more than the conventional 0.60 threshold for substantial agreement (Landis & Koch, 1977). Therefore, they confirm that our error classifications and hallucination annotations are sufficiently consistent to support the subsequent analyses.

Table 1 reports performance in KG-aware mode. GPT-5.1 achieves the highest parse ability (96%) and semantic result-equivalence (91%), indicating strong syntactic query formation and relational correctness. Gemini 2.0 Pro follows with moderate degradation under multi-hop tasks, while Llama 3, Qwen, and BioGPT exhibit substantial drops in semantic equivalence despite relatively high parse ability rates.

The gap between parse ability and equivalence reflects semantic join failures in multihop reason-ing. While several models produce syntactically valid SPARQL queries, they fail to preserve correct relational bindings across heterogeneous schemas.

Figure 1 illustrates degradation across query tiers. All models exhibit monotonic performance decline as relational complexity increases, but the magnitude differs substantially. GPT-5.1 maintains relatively stable multi-hop reasoning, whereas BioGPT and smaller open models show sharp compositional brittleness.

![](images/fd6d7ef62060bb4a1cfd47cd4d00d47b329e9b5c8de56e1ff615cbbfd9ab07ca.jpg)  
Figure 1: Structural reliability comparison in KG-aware mode. Although several models produce syntactically valid queries, semantic equivalence degrades substantially for smaller models under multi-hop reasoning.

## 4.1. Hallucination and KG-Agnostic Behavior

Table 2 presents performance in KG-agnostic mode. Removing structured grounding leads to systematic hallucination increases. GPT-5.1 maintains the lowest hallucination rate (7%) and highest semantic similarity (0.91), whereas BioGPT reaches 30% hallucination rate, particularly on cross-domain queries.

Table 2. Performance in KG-agnostic mode
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Hallucination(%)</td><td rowspan=1 colspan=1>EM Simple(%)</td><td rowspan=1 colspan=1>EM Multi-hop(%)</td><td rowspan=1 colspan=1>EM Reasoning(%)</td><td rowspan=1 colspan=1>CosineSimilarity</td></tr><tr><td rowspan=1 colspan=1>GPT-5.1</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>92</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>0.91</td></tr><tr><td rowspan=1 colspan=1>Gemini 2.0Pro</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>Llama 3</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>52</td><td rowspan=1 colspan=1>0.72</td></tr><tr><td rowspan=1 colspan=1>Qwen</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>83</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1>BioGPT</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>0.68</td></tr></table>

Figure 2 demonstrates a negative correlation be-tween hallucination rate and semantic similarity. Models with higher hallucination rates systematically exhibit lower embedding-level agreement with ground truth answers.

![](images/526473e0c16ba65843244e6d8def42ce005910ceffe05a860c62f20e9a75c9ad.jpg)  
Figure 2: Hallucination frontier in KG-agnostic mode. Models are better when they move toward the upper-left: fewer hallucinations and higher semantic similarity.

## 4.2. Latency–Accuracy Trade-off

Latency results are summarized in Table 3. Fron-tier API-based models incur moderate latency over-head but deliver substantially higher reasoning reliability. Local models are computationally efficient but exhibit reduced compositional robustness.

<table><tr><td>Model</td><td>KG-aware</td><td>KG-agnostic</td></tr><tr><td>GPT-5.1</td><td>1.5</td><td>1.2</td></tr><tr><td>Gemini 2.0 Pro</td><td>1.8</td><td>1.4</td></tr><tr><td>Llama 3</td><td>0.4</td><td>0.3</td></tr><tr><td>Qwen</td><td>0.5</td><td>0.4</td></tr><tr><td>BioGPT</td><td>0.45</td><td>0.35</td></tr></table>

Table 3. Average latency per query (seconds)

Overall, three consistent patterns emerge:

Structured grounding substantially reduces hallucination.

Multi-hop reasoning remains the dominant performance bottleneck.

Syntactic query validity does not guarantee semantic equivalence.

Frontier general-purpose models demonstrate stronger cross-KG relational abstraction, whereas domain-specialized models retain narrow factual strength but limited compositional generalization.

## 5. Discussion

The high inter-rater agreement in the annotation phase lends strong empirical sup-port to our er-ror taxonomy. These levels of consistency satisfactory with standards in the field of (NLP). This annotator concordance is important for our hallucination analysis. It ensures that the differences observed across models genuinely reflect architectural variations in hallucination vulnerability rather than artifacts of subjective annotation bias. The experts in our setup agreed far more than we expected. It suggests its potential utility for standardized evaluation protocols in future LLM-KG benchmarking efforts. Examined how different large language models perform when answering cognitive neuro-science questions using heterogeneous knowledge graphs.

The results show that performance depends not only on model size, but also on training back-ground and access to structured knowledge.

GPT-5.1 consistently achieved the best overall performance. It showed strong query translation, stable multi-hop reasoning, and low hallucination rates. Its performance stayed stable even as questions grew more complex suggesting that large general-purpose models, combined with structured knowledge graphs, can handle complex cross-domain reasoning effectively.

Gemini 2.0 Pro also performed well in KG-aware mode. However, its performance dropped more notice-ably in KG-agnostic mode. Without a knowledge graph, reasoning becomes less accurate and less reliable. Domain-specific models showed a different pattern.

BioGPT performed well on drug-related questions but struggled when tasks required integration across cognitive and neuroimaging knowledge. This indicates that domainfocused training im-proves precision within a narrow area but may limit broader reasoning.

Llama 3 and Qwen support simple factual questions well. But struggled with multihop and reasoning tasks. Multi-hop reasoning remains the most challenging setting. As relational depth increased, all models showed some degree of performance decline. GPT-5.1 showed the smallest decline, suggesting stronger in-ternal representation of relational dependencies. Other models often failed to correctly connect entities across multiple knowledge graphs, leading to incomplete or partially incorrect answers. Hallucination analysis further emphasizes the im-portance of knowledge graph grounding.

In KG-agnostic mode, hallucination rates increased for all models. When responses were con-strained by structured graph execution, hallucination decreased significantly. This confirms that structured external knowledge helps stabilize model outputs and reduces unsupported claims. For real-world neuroscience applications, grounding LLMs in validated knowledge graphs is therefore essential.

These findings have practical implications. LLM–KG systems can support complex neuro-scientific queries, cross-dataset integration, and exploratory analysis. However, model choice and ground-ing strategy must be carefully considered. A hybrid approach that combines general-purpose reasoning capacity with domain-specific knowledge may provide a balanced solution. We should note a few limitations. Our benchmark includes five main knowledge graphs.

Performance could change with future updates or fine-tuning. More study should explore reasoning transparency, uncertainty estimation and integration with experimental neuroimaging pipelines.LLMs show real promise for heterogeneous neuroscience knowledge graphs, but to be reliable, they need structured grounding and stronger multihop reasoning.

## 6. Conclusion

This study presented a systematic evaluation of five large language models interacting with five heterogeneous cognitive neuroscience and biomedical knowledge graphs. By jointly analyzing query translation, multi-hop reasoning, hallucination behavior, and latency, we provide a detailed view of how current LLMs perform in neuro-symbolic settings. The results show a good pattern. General-purpose models with broad pretraining, particularly GPT-5.1, demonstrate more stable performance across structured and unstructured evaluation modes. They maintain higher translation accuracy, stronger multihop reasoning, and lower hallucination rates. In contrast, domain-specific models such as BioGPT perform well within narrow factual do-mains but struggle when reasoning requires integration across multiple graphs or conceptual hierarchies. Good cross-domain reasoning needs more than domain expertise and it needs structural flexibility and general semantic abstraction.

Each model when a query included multi-hop reasoning in our tests broke down. Otherwise, we saw the same pattern for all models. The more complex the relations, it is the worse the performance. This concludes that structured inference over heterogeneous graphs is still a real struggle for today's LLMs.

The experiments further demonstrate that grounding models in validated knowledge graphs significantly reduces hallucination and im-proves reliability. Therefore, structured access to external knowledge should be considered a core component of neuro-symbolic systems rather than an optional enhancement.

Beyond model comparison, this work contributes a reproducible benchmark framework for evaluating LLM–KG interaction in cognitive neuroscience. The dual evaluation setup, combining KG-aware and KG-agnostic modes, helps separate reasoning ability from parametric memorization and offers a clearer under-standing of model behavior.

Future research can extend this work in several directions. More knowledge graphs, fine-tuned neuroscience models and adaptive grounding could deepen our understanding of cross-domain reasoning.

Research on explainability, uncertainty estimation and structured confidence reporting is necessary for real-world application. Integrating LLM–KG systems with experimental neuroimaging or electrophysiological data could enhance practical applicability. Finally, controlled LLM pipelines for incrementally enriching knowledge graphs could help neuroscience generate hypotheses at scale and keep knowledge continuously updated.

Overall, LLMs show real promise for working with heterogeneous neuroscience knowledge graphs, but to be consistently reliable, they need structured grounding and better multi-hop reasoning.

## 7. References

Baek, J., Aji, A. F., & Saffari, A. (2023). Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. In Proceedings of the 1st Workshop on Natural Language Reasoning and Structured Explanations (NLRSE), 78-106.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877–1901.

Creswell, A., & Shanahan, M. (2022). Faithful reasoning using large language models. arXiv preprint arXiv:2208.14271. https://doi.org/10.48550/arXiv.2208.14271

D'Abramo, J., Zugarini, A., & Torroni, P. (2025). Investigating large language models for text-to-SPARQL generation. In Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing (pp. 66–80). Association for Computational Linguistics.

Dubey, M., Banerjee, D., Abdelkawi, A., & Lehmann, J. (2019). LC-QuAD 2.0: A large dataset for complex question answering over Wikidata and DBpedia. In The Semantic Web – ISWC 2019: 18th International Semantic Web Conference (pp. 69– 78). Springer. https://doi.org/10.1007/978-3-030-30796-7\_5

Garcez, A. d., Besold, T. R., De Raedt, L., Földiak, P., Hitzler, P., Icard, T., Kühnberger, K.-U., Lamb, L. C., Miikkulainen, R., & Silver, D. L. (2015). Neural-symbolic learning and reasoning: Contributions and challenges. In 2015 AAAI Spring Symposium Series (pp. 18–21). AAAI Press.

Himmelstein, D. S., Lizee, A., Hessler, C., Brueggeman, L., Chen, S. L., Hadley, D., Green, A., Khankhanian, P., & Baranzini, S. E. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. eLife, 6, e26726. https://doi.org/10.7554/eLife.26726

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), Article 248. https://doi.org/10.1145/3571730

Laird, A. R., Lancaster, J. L., & Fox, P. T. (2005). BrainMap: The social evolution of a human brain mapping database. Neuroinformatics, 3(1), 65–78. https://doi.org/10.1385/ni:3:1:065 (Laird et al., 2005)

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems (pp. 9459–9474). Curran Associates Inc.

Ma, C., Chen, Y., Wu, T., Khan, A., & Wang, H. (2025). Large language models meet knowledge graphs for question answering: Synthesis and opportunities. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (pp. 24589–24608). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1249

Poldrack, R. A., Kittur, A., Kalar, D., Miller, E., Seppa, C., Gil, Y., Parker, D. S., Sabb, F. W., & Bilder, R. M. (2011). The cognitive atlas: Toward a knowledge foundation for cognitive neuroscience. Frontiers in Neuroinformatics, 5, 17. https://doi.org/10.3389/fninf.2011.00017

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-totext transformer. Journal of Machine Learning Research, 21(140), 1–67. http://jmlr.org/papers/v21/20-074.html

Rahsepar Fard, K., Sarabadani, A., & Dalvand, H. (2025). GDPKG-LLM: Integrating gene, disease, and pharmacogenomics knowledge graphs for cognitive neuroscience using large language models. Computer Science, 26(3). https://doi.org/10.7494/csci.2025.26.3.6673

Sarabadani, A., Rahsepar Fard, K., & Dalvand, H. (2026). CNKG: Harnessing large language models for cognitive neuroscience knowledge graph construction. Turkish Journal of Electrical Engineering and Computer Sciences, 34(4), 542–560. https://doi.org/10.55730/1300-0632.4190

Srivastava, S., Patidar, M., Chowdhury, S., Agarwal, P., Bhattacharya, I., & Shroff, G. (2021). Complex question answering on knowledge graphs using machine translation and multi-task learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume (pp. 3428– 3439). Association for Computational Linguistics.

Talmor, A., & Berant, J. (2018). The web as a knowledge-base for answering complex questions. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (pp. 641–651). Association for Computational Linguistics.

Tsatsaronis, G., Balikas, G., Malakasiotis, P., Partalas, I., Zschunke, M., Alvers, M. R., Weissenborn, D., Krithara, A., Petridis, S., Polychronopoulos, D., Almirantis, Y., Pavlopoulos, J., Baskiotis, N., Gallinari, P., Artières, T., Ngomo, A. C., Heino, N., Gaussier, E., Barrio-Alvers, L., Schroeder, M., … Paliouras, G. (2015). An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition. BMC Bioinformatics, 16, 138. https://doi.org/10.1186/s12859-015-0564-6

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 6000–6010). Curran Associates Inc.

Wishart, D. S., Feunang, Y. D., Guo, A. C., Lo, E. J., Marcu, A., Grant, J. R., Sajed, T., Johnson, D., Li, C., Sayeeda, Z., Assempour, N., Iynkkaran, I., Liu, Y., Maciejewski, A., Gale, N., Wilson, A., Chin, L., Cummings, R., Le, D., Pon, A., … Wilson, M. (2018). DrugBank 5.0: A major update to the DrugBank database for 2018. Nucleic Acids Research, 46(D1), D1074–D1082. https://doi.org/10.1093/nar/gkx1037

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 2369–2380). Association for Computational Linguistics.

Yarkoni, T., Poldrack, R. A., Nichols, T. E., Van Essen, D. C., & Wager, T. D. (2011). Large-scale automated synthesis of human functional neuroimaging data. Nature Methods, 8(8), 665–670. https://doi.org/10.1038/nmeth.1635 (Yarkoni et al., 2011)

Yin, J., Jiang, X., Lu, Z., Shang, L., Li, H., & Li, X. (2016). Neural generative question answering. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (pp. 2972–2978). AAAI Press.

Zhang, Y., Dai, H., Kozareva, Z., Smola, A., & Song, L. (2018). Variational reasoning for question answering with knowledge graph. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). https://doi.org/10.1609/aaai.v32i1.12057

Zhang, Y., Sui, X., Pan, F., Yu, K., Li, K., Tian, S., Erdengasileng, A., Han, Q., Wang, W., Wang, J., Wang, J., Sun, D., Chung, H., Zhou, J., Zhou, E., Lee, B., Zhang, P., Qiu, X., Zhao, T., & Zhang, J. (2025). A comprehensive large scale biomedical

knowledge graph for AI powered data driven biomedical research. bioRxiv. https://doi.org/10.1101/2023.10.13.562216