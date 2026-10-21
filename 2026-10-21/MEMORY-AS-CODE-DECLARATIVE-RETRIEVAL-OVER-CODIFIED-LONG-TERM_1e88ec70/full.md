# MEMORY AS CODE: DECLARATIVE RETRIEVAL OVER CODIFIED LONG-TERM MEMORY

Aoting Zeng<sup>1</sup>, Yuxin Du<sup>1</sup>, Jianwei Wang<sup>2</sup>, Yizhang He<sup>1</sup> Shan Jiang<sup>1</sup>, Xuemin Lin<sup>3</sup>, Kai Wang<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>University of New South Wales

<sup>3</sup>The Chinese University of Hong Kong, Shenzhen

{aotingzeng,dyx20031122,yizhang.he,greatjs,w.kai}@sjtu.edu.cn jianwei.wang1@unsw.edu.au, xueminlin@cuhk.edu.cn

## ABSTRACT

In long-term memory systems, memories are constructed from interaction histories and retrieved in response to queries, enhancing the reasoning ability of LLM agents. In complex scenarios, facts are scattered across sessions and their relationships evolve over time, making it difficult for existing memory systems to compose them to adapt to diverse queries. Motivated by the modularity and composability of code, we propose Memory as Code (MaC), a unified memory framework that jointly designs memory construction and retrieval through a shared code abstraction. MaC couples codified construction, which establishes executable memory abstractions, with declarative retrieval, which translates queries into structured retrieval programs. Specifically, codified construction builds a three-layer executable memory structure that progresses from an atomic function layer, through a relational layer, to a modular composition layer. Declarative retrieval recasts retrieval from relevance-based fact selection into declarative query processing over memory. Experiments on LOCOMO and LONGMEMEVAL show relative gains of up to 15.91% in overall LLM-judge accuracy over the strongest baseline in each setting, while substantially reducing token consumption, highlighting the effectiveness of memory as code for long-horizon memory tasks.

## 1 INTRODUCTION

Long-term memory enables Large Language Model (LLM) agents to retain and use information across interactions, supporting personalized assistance and long-horizon tasks (Zhong et al., 2024; Park et al., 2023). A memory system Xu et al. (2026a); Packer et al. (2023) comprises memory construction, which organizes interaction histories into memory data, and memory retrieval, which processes memory queries over memory data. For memory construction, MemoryOS (Kang et al., 2025) organizes dialogue into topic segments and summaries, Mem0 (Chhikara et al., 2025) consolidates facts through update operations, and A-Mem (Xu et al., 2026b) constructs dynamically linked memory notes. These methods support memory organization and maintenance, but provide limited support for encapsulating computations over stored facts for reuse across queries. For memory retrieval, existing approaches combine search with contextual or relational expansion to gather relevant evidence (Borgeaud et al., 2022; Liu et al., 2023), using either predefined pipelines (Han et al., 2024; Rasmussen et al., 2025) or agentic workflows (Yao et al., 2022; Li et al., 2025; Ji et al., 2026). However, predefined pipelines offer limited flexibility for diverse queries, whereas stepwise agentic methods achieve this flexibility through repeated LLM planning, incurring additional inference overhead(Kim et al., 2023; Tan et al., 2025).

The above observations motivate a joint design in which memory construction encapsulates reusable computations over stored facts, and retrieval composes them for each query while reducing the need for repeated planning. Code provides a suitable foundation: shared references preserve consistent fact identities, functions expose reusable computations through explicit interfaces, and programs specify their composition and conditional branches before execution begins. This leads to a codecentric formulation in which memory is a library of callable computations and retrieval is a queryspecific program over that library.

![](images/23958549510502503d47144235eb984a7680572d9547591a7547fb87cbad9c66.jpg)  
Figure 1: (a) An example illustrating the limitations of relevant methods. (b) MaC achieves the highest LLM-judge accuracy with the lowest token consumption among the compared methods on both LOCOMO and LONGMEMEVAL.

However, achieving these goals through a shared code abstraction poses two challenges: (1) How can interaction histories be transformed into executable memory code? Facts expressed across sessions may overlap, supplement one another, or supersede earlier states, distinctions that are not automatically resolved by encoding facts as functions. Preserving consistent fact identities and distinguishing current from historical states is a key challenge. (2) How can retrieval compose memory code to accommodate diverse queries? Relevant functions alone do not specify how their outputs should be combined, and subsequent operations may depend on intermediate results. Capturing these dependencies and conditional continuations in a complete retrieval program while reducing repeated LLM planning is another key challenge.

To address these challenges, we propose Memory as Code (MaC), a framework that represents longterm memory as executable code and retrieves information through query-specific programs. The framework comprises two components: (1) codified construction transforms facts extracted from interaction histories into a three-level executable memory structure. The atomic function layer encodes facts as functions with canonical identifiers and source references; the relational layer connects fact functions through explicit supersession relations while retaining historical evidence; and the modular composition layer composes base functions into reusable current-state and history computations. A compact memory index exposes function capabilities, arguments, and available operators for retrieval. (2) Declarative retrieval interprets each query as a high-level specification and compiles it into a complete retrieval program in one LLM planning call. Using the memory index, the program selects and composes the required memory computations, specifying operator arguments, dependencies, and conditional continuations. An executor checks and runs the program, following predefined branches based on intermediate results and assembling supporting evidence without stepwise LLM planning during successful execution. The two components work together through the memory index and a common evidence representation. The index connects constructed capabilities to query requirements, while the evidence representation supports result composition and source tracing. As illustrated in Figure 1, this design connects facts distributed across sessions through reusable functions to answer the query.

Our main contributions are as follows:

• We study long-term memory for LLM agents to support reasoning over evolving information across interactions. We propose Memory as Code (MaC), a framework that uses a shared code abstraction to connect memory capabilities with query-specific composition.

• We develop two complementary techniques: codified construction organizes facts, their evolving relations, and reusable computations in a three-level executable memory structure with consistent state semantics and source provenance; declarative retrieval compiles each query into a complete retrieval program that selects and composes these capabilities in one LLM planning call.

• Extensive experiments on widely used memory benchmarks show that MaC achieves the highest overall LLM-judge accuracy in all four evaluation settings, improving over the strongest baseline by up to 15.91% on LOCOMO and 9.06% on LONGMEMEVAL. Ablation studies further support the benefits of both techniques.

![](images/4503553aaa6de75a1760261e9f7c4f37854970aaa3309da503b5d00b476ddba3.jpg)  
Figure 2: Overview of MaC. (a) Codified construction extracts atomic facts from dialogue history and builds a code library of fact functions, relations, derived functions, and concept and session views, along with an interface index that maps views to function identifiers (view index), identifiers to callable entry points (function index), and lists available memory operations (operators index). (b) Declarative retrieval passes the question and indexed interfaces to an LLM planner, which generates a retrieval program; after checking, an executor runs the program over the library to assemble supporting evidence for the answer LLM.

## 2 MEMORY AS CODE

In this section, we present MaC for long-term agent memory. We first introduce the problem definition and solution framework (§2.1), then describe codified construction (§2.2) and declarative retrieval (§2.3), and finally analyze the time and space complexity (§2.4).

## 2.1 PROBLEM DEFINITION AND SOLUTION FRAMEWORK

Problem Definition. Given a dialogue history $\mathcal { H } = ( s _ { 1 } , . . . , s _ { n _ { \mathrm { s e s s } } } )$ of $n _ { \mathrm { s e s s } }$ sessions, memory construction organizes its information into an external memory M. For a query $q ,$ memory retrieval accesses and combines information in M to produce supporting evidence $E _ { q } ,$ , which is used to generate an answer $\boldsymbol { a } _ { \boldsymbol { q } } .$ . Appendix A.1 summarizes the notation used throughout the paper.

Solution Framework Overview. MaC organizes memory construction and retrieval around a shared code abstraction: memory provides reusable functions, and retrieval composes these functions into query-specific computations. As illustrated in Figure 2, the framework comprises two stages. Codified construction transforms dialogue history H into a code library of fact functions and reusable computations over current and historical states. A memory index exposes the callable interfaces and the available memory operators. Declarative retrieval uses these interfaces to translate a query q into a complete retrieval program in one LLM planning call. After checking, the program executes over the library to assemble supporting evidence $E _ { q }$ for answer generation. The two stages are connected by indexed interfaces and a common evidence representation, allowing retrieval to discove and compose the capabilities established during construction.

## 2.2 CODIFIED CONSTRUCTION: FACT-TO-FUNCTION

Given dialogue history ${ \mathcal { H } } ,$ memory construction begins by extracting atomic facts to distill salient information and filter out irrelevant conversational details. We then encode these facts as executable memory to support reusable computation. However, directly encoding these facts into code is challenging because they are expressed in natural language, and the relations among them are complex and evolve over time. To address these challenges, we design codified construction to organize facts into callable functions, explicitly represent their relations, and compose them into reusable operators. We first introduce the codified memory representation (§2.2.1), then describe memory organization and indexing (§2.2.2).

## 2.2.1 CODIFIED MEMORY REPRESENTATION

We define codified memory M that includes a code library $\mathcal { L }$ and a memory index I. The code library comprises four sets: base fact functions ${ \mathcal { F } } _ { : }$ , memory views V, fact relations $\mathcal { R }$ , and derived functions D:

$$
\begin{array} { r } { \mathcal { M } = ( \mathcal { L } , I ) , \qquad \mathcal { L } = ( \mathcal { F } , \mathcal { V } , \mathcal { R } , \mathcal { D } ) . } \end{array}
$$

The library organizes memory into three layers. The atomic function layer represents individual facts as callable functions in ${ \mathcal F } .$ The relational layer specifies fact connections and state replacements through $\mathcal { R } .$ The modular composition layer provides derived functions in D for reusable computations over facts and their relations, including current-state resolution and history retrieval. Base and derived functions return evidence with fact identifiers and source references, supporting composition and provenance.

The set of memory views V contains concept views, which group function references by person or topic, and session views, which group them by dialogue session. The memory index I specifies view membership, callable function interfaces, and available memory operators, enabling retrieval to select and compose memory computations.

## 2.2.2 MEMORY ORGANIZATION AND INDEXING

We construct the three layers of the code library L from extracted facts, then build the memory index I to expose the resulting functions and memory operators for retrieval.

Code Library Organization. To translate natural-language facts into executable memory, we use an LLM to produce structured function specifications. A structured prompt specifies the required fact fields and function interface, and a local generator converts each specification into an executable base fact function.

At the atomic function layer, we canonicalize and deduplicate facts and assign each a canonical identifier i. For each canonical fact, the LLM produces a specification containing its structured value $v _ { i } ,$ including the subject, predicate, and available time information, together with a statefulness annotation. The local generator uses this specification and the supporting dialogue references $S _ { i }$ to create a base fact function $f _ { i } \in F$ that returns an evidence record $e _ { i } = ( v _ { i } , \{ i \} , S _ { i } )$ . Concept views and session views import shared references to these functions.

At the relational layer, we group stateful facts by subject and predicate and ask the LLM to identify explicit replacements. Each replacement becomes a supersession relation (j, supersedes, $i ) \in$ R, indicating that fact j replaces fact i while both functions remain available. We also connect facts by temporal order, shared subject, and shared topic.

At the modular composition layer, we instantiate derived functions in D from reusable currentstate and history templates over linked facts and register them in the relevant views. The resulting functions invoke base fact functions in $\mathcal { F }$ and use their relations in $\mathcal { R }$ to compute results. For a state group G sharing a subject and predicate, let $G ( \tau )$ contain the identifiers of facts whose recorded times do not exceed the time boundary τ. The current-state template computes the active facts

$A _ { G } ( \tau )$ and resolves the current value:

$$
\begin{array} { r l } & { ~ A _ { G } ( \tau ) = \{ i \in G ( \tau ) \mid \mathcal { B } j \in G ( \tau ) : ( j , \mathsf { s u p e r s e d e s } , i ) \in \mathcal { R } \} , } \\ & { ~ \mathrm { c u r r e n t } _ { G } ( \tau ) = \left\{ \begin{array} { l l } { v _ { i } , } & { A _ { G } ( \tau ) = \{ i \} , } \\ { \bot , } & { | A _ { G } ( \tau ) | \neq 1 . } \end{array} \right. } \end{array}\tag{1}
$$

Here ⊥ denotes an unresolved state when there is no unique active fact. The history template retains all eligible facts, including superseded ones; in Figure 2, D1 combines the banking fact F1 and teaching fact F2 into Joanna’s career history. Both templates preserve supporting fact identifiers, dialogue references, and applied computation rules as provenance.

Memory Index Construction. The code library provides executable memory, but downstream retrieval also needs to discover its contents and determine how to use them. We therefore construct the memory index I with two parts: indices over library contents and interfaces to memory operators.

For the library contents, we build indices at two levels. The function index $I _ { \mathrm { f u n c } }$ maps each base or derived function identifier to its entry point, description, and accepted arguments, extracted from function metadata. The view index $I _ { \mathrm { v i e w } }$ maps each concept or session view in V to the identifiers of the functions it references. Together, these indices support navigation from views to callable functions without inspecting their implementations.

We define four categories of memory operators for downstream computation. The operator index $I _ { \mathrm { o p s } }$ records their interfaces. Memory access retrieves facts and invokes functions; context expansion follows views and relations to gather additional facts and source context; evidence composition combines, filters, ranks, and aggregates evidence; and verification provides heuristic checks of evidence completeness and unresolved states. The content indices and operator interfaces together form $I _ { \mathit { \Phi } } = \mathit { \bar { \Phi } } _ { \mathrm { { ( { I _ { v i e w } } , { I _ { f u n c } } , { I _ { o p s } } ) } } }$ , exposing both available memory resources and the operations for using them. Table 6 in Appendix A.2 provides detailed operator descriptions.

## 2.3 DECLARATIVE RETRIEVAL: QUERY-TO-PROGRAM

To adapt retrieval to diverse queries, declarative retrieval interprets each query as a high-level specification and compiles it into a structured retrieval program that selects and composes the required memory computations. We first describe declarative query planning (§2.3.1), then present retrieval program execution (§2.3.2).

Retrieval Program. Given a query q and codified memory $\mathcal { M } = ( \mathcal { L } , I )$ , we define a retrieval program $P _ { q }$ as an executable query plan over the memory operators exposed by I. The program specifies operator arguments and dependencies, together with any conditional continuations. These operators encapsulate memory access and computation, allowing the program to compose existing memory functions and process their results through predefined operations. Executing $P _ { q }$ over $\mathcal { M }$ produces the supporting evidence $E _ { q }$ specified in the problem definition.

## 2.3.1 DECLARATIVE QUERY PLANNING

Diverse queries require retrieval plans tailored to their information needs. A question about a current attribute may need a single function call. Questions combining facts across sessions may require context expansion and evidence composition. We use an LLM planner to interpret each query and reason about the required memory operations.

Initial retrieval selects a set of candidate function identifiers $K _ { q }$ using the memory index I built during construction. Their entries in $I _ { \mathrm { f u n c } }$ and the operator interfaces in $I _ { \mathrm { o p s } }$ form the planning context $I _ { q } ( \mathcal { M } )$ . Function entries describe what each callable computes and which arguments it accepts. Operator interfaces specify the available operations and their input and output constraints. This compact context supports planning without loading the full code library.

Our structured prompt presents the query, function information, operator constraints, and planning guidance (Appendix B). The guidance illustrates how query requirements translate into operations, such as temporal filtering or expansion across sessions. The prompt also specifies evidence compatibility constraints and requires the program to return supporting evidence. The LLM uses the query’s constraints to select relevant functions and operators, then links their outputs to subsequent inputs.

Table 1: Performance across different question types on LOCOMO. F1 is reported in percent, and LLM-Judge accuracy (J) is reported as mean ± standard deviation in percent over five independent judge evaluations. Improv. denotes the relative improvement of MaC over each corresponding baseline in Overall J. Bold and underline mark the best and second-best results.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Open-domain</td><td colspan="2">Single-hop</td><td colspan="2">Overall</td><td rowspan="2">Improv.</td></tr><tr><td>F1 ↑</td><td>J↑</td><td>F1 ↑</td><td>J↑</td><td>F1 ↑</td><td>J↑</td><td>F1 ↑</td><td>J↑</td><td>F1 ↑</td><td>J↑</td></tr><tr><td rowspan="5">Qwen 3.6-27B</td><td>A-Mem</td><td>39.44</td><td> $5 7 . 7 3 { \scriptstyle \pm 2 . 9 2 }$ </td><td>37.16</td><td> $4 5 . 2 3 _ { \pm 3 . 3 9 }$ </td><td>20.13</td><td> $4 5 . 0 0 { \scriptstyle \pm 0 . 8 7 }$ </td><td>54.40</td><td> $6 8 . 8 0 { \scriptstyle \pm 1 . 7 2 }$ </td><td>45.93</td><td> $6 0 . 3 8 { \scriptstyle \pm 0 . 8 0 }$ </td><td>+43.82%</td></tr><tr><td>MemoryOS</td><td>29.42</td><td> $4 2 . 0 6 _ { \pm 2 . 1 7 }$ </td><td>35.96</td><td> $2 2 . 1 2 { \scriptstyle \pm 0 . 5 8 }$ </td><td>23.43</td><td> $4 3 . 9 6 _ { \pm 1 . 7 1 }$ </td><td>38.93</td><td> $5 0 . 2 0 { \scriptstyle \pm 0 . 9 6 }$ </td><td>35.60</td><td> $4 2 . 4 7 _ { \pm 1 . 0 3 }$ </td><td>+104.47%</td></tr><tr><td>Mem0</td><td>43.67</td><td> $7 1 . 0 6 _ { \pm 2 . 1 9 }$ </td><td>60.27</td><td> $\underline { { 8 1 . 6 8 } } _ { \pm 0 . 1 4 }$ </td><td>30.53</td><td> $5 6 . 8 8 _ { \pm 0 . 9 3 }$ </td><td>53.93</td><td> $\underline { { 7 5 . 7 0 } } \pm 0 . 3 5$ </td><td>51.91</td><td> $\underline { { 7 4 . 9 2 } } \pm 0 . 2 8$ </td><td>+15.91%</td></tr><tr><td>MRAgent</td><td>48.88</td><td> $7 5 . 0 4 { \scriptstyle \pm 2 . 4 9 }$ </td><td>58.36</td><td> $7 1 . 0 3 { \scriptstyle \pm 0 . 3 8 }$ </td><td>28.19</td><td> $5 7 . 9 2 { \scriptstyle \pm 1 . 5 8 }$ </td><td>55.01</td><td> $7 0 . 1 1 { \scriptstyle \pm 0 . 6 9 }$ </td><td>52.92</td><td> $7 0 . 4 4 _ { \pm 0 . 8 1 }$ </td><td>+23.28%</td></tr><tr><td>MaC</td><td>56.20</td><td> $\mathbf { 8 7 . 0 9 } _ { \pm 0 . 7 4 }$ </td><td>65.78</td><td> ${ \bf 8 3 . 9 3 _ { \pm 0 . 8 7 } }$ </td><td>37.36</td><td> ${ \bf 6 3 . 7 5 _ { \pm 0 . 4 7 } }$ </td><td>73.49</td><td> $\mathbf { 9 0 . 5 1 { \scriptstyle \pm 0 . 4 1 } }$ </td><td>66.46</td><td> $\mathbf { 8 6 . 8 4 _ { \pm 0 . 3 5 } }$ </td><td></td></tr><tr><td rowspan="5">ChatGPT 5.6-Luna</td><td>A-Mem</td><td>41.59</td><td> $6 0 . 4 3 _ { \pm 1 . 2 4 }$ </td><td>54.26</td><td> $6 2 . 5 5 { \scriptstyle \pm 0 . 8 1 }$ </td><td>25.32</td><td> $5 1 . 4 6 _ { \pm 2 . 0 3 }$ </td><td>53.23</td><td> $7 0 . 0 6 { \scriptstyle \pm 0 . 9 5 }$ </td><td>49.58</td><td> $6 5 . 5 7 { \scriptstyle \pm 0 . 9 4 }$ </td><td>+33.69%</td></tr><tr><td>MemoryOS</td><td>32.45</td><td> $5 8 . 6 5 _ { \pm 1 . 1 1 }$ </td><td>42.22</td><td> $3 8 . 6 9 _ { \pm 0 . 1 4 }$ </td><td>22.01</td><td> $5 0 . 0 0 _ { \pm 1 . 4 7 }$ </td><td>41.64</td><td> $6 8 . 8 0 _ { \pm 1 . 0 5 }$ </td><td>38.85</td><td> $5 9 . 4 9 _ { \pm 0 . 7 5 }$ </td><td>+47.35%</td></tr><tr><td>Mem0</td><td>40.96</td><td> $7 0 . 7 8 { \scriptstyle \pm 3 . 0 9 }$ </td><td>55.25</td><td> $7 3 . 9 6 _ { \pm 0 . 2 8 }$ </td><td>28.83</td><td> $5 7 . 5 0 { \scriptstyle \pm 1 . 3 6 }$ </td><td>52.54</td><td> $\underline { { 8 8 . 0 9 } } \pm 0 . 2 6 $ </td><td>49.51</td><td> $8 0 . 0 6 _ { \pm 0 . 6 9 }$ </td><td>+9.49%</td></tr><tr><td>MRAgent</td><td>51.86</td><td> $\underline { { 7 1 . 5 6 } } \pm 1 . 5 1$ </td><td>67.64</td><td> ${ \underline { { 7 6 . 8 2 } } } \pm 1 . 2 8 $ </td><td>38.60</td><td> $\underline { { 6 0 . 2 1 } } \pm 1 . 7 1$ </td><td>69.15</td><td> $8 6 . 6 6 { \scriptstyle \pm 0 . 5 0 }$ </td><td>63.76</td><td> $\underline { { 8 0 . 1 9 } } \pm 0 . 7 7$ </td><td>+9.32%</td></tr><tr><td>MaC</td><td>52.41</td><td> $\mathbf { 8 2 . 2 0 { \scriptstyle \pm 0 . 6 8 } }$ </td><td>69.81</td><td> $\mathbf { 8 6 . 2 9 _ { \pm 0 . 4 4 } }$ </td><td>39.78</td><td> ${ \bf 6 4 . 5 8 } _ { \pm 1 . 2 8 }$ </td><td>68.41</td><td> $\mathbf { 9 } 2 . 6 5 _ { \pm 0 . 2 1 }$ </td><td>63.99</td><td> $\mathbf { 8 7 . 6 6 _ { \pm 0 . 1 5 } }$ </td><td></td></tr><tr><td rowspan="5">DeepSeek V4-Flash</td><td>A-Mem</td><td>41.63</td><td> $6 3 . 6 9 _ { \pm 1 . 8 9 }$ </td><td>37.58</td><td> $4 2 . 7 4 _ { \pm 2 . 0 7 }$ </td><td>25.15</td><td> $5 2 . 9 2 _ { \pm 1 . 7 1 }$ </td><td>55.88</td><td> $7 7 . 5 3 _ { \pm 1 . 3 3 }$ </td><td>47.54</td><td> $6 6 . 2 1 _ { \pm 1 . 4 0 }$ </td><td>+31.99%</td></tr><tr><td>MemoryOS</td><td>27.26</td><td> $5 2 . 0 6 _ { \pm 2 . 7 2 }$ </td><td>32.80</td><td> $3 2 . 0 9 _ { \pm 4 . 7 0 }$ </td><td>22.96</td><td> $5 0 . 8 3 { \scriptstyle \pm 2 . 0 0 }$ </td><td>40.84</td><td> $6 6 . 0 6 { \scriptstyle \pm 0 . 9 6 }$ </td><td>35.57</td><td> $5 5 . 4 7 _ { \pm 2 . 1 1 }$ </td><td>+57.54%</td></tr><tr><td>Mem0</td><td>39.44</td><td> $\underline { { 7 6 . 4 5 } } \pm 0 . 8 9$ </td><td>58.09</td><td> $\underline { { 8 3 . 4 9 } } \pm 0 . 0 0$ </td><td>34.16</td><td> ${ \bf 6 5 . 2 1 { \scriptstyle \pm 0 . 9 3 } }$ </td><td>50.54</td><td> $8 2 . 2 8 _ { \pm 1 . 0 7 }$ </td><td>49.06</td><td> $8 0 . 4 0 { \scriptstyle \pm 0 . 7 9 }$ </td><td>+8.69%</td></tr><tr><td>MRAgent</td><td>48.02</td><td> $7 6 . 1 0 { \scriptstyle \pm 1 . 4 7 }$ </td><td>69.67</td><td> $8 1 . 7 4 _ { \pm 0 . 2 8 }$ </td><td>27.92</td><td> $5 5 . 4 2 _ { \pm 0 . 8 7 }$ </td><td>66.45</td><td> $9 0 . 1 1 { \scriptstyle \pm 0 . 3 5 }$ </td><td>61.35</td><td> $8 3 . 6 4 { \scriptstyle \pm 0 . 4 9 }$ </td><td>+4.48%</td></tr><tr><td>MaC</td><td>52.99</td><td> $\mathbf { 8 6 . 8 1 _ { \pm 0 . 3 0 } }$ </td><td>70.48</td><td> $\mathbf { 8 4 . 7 4 _ { \pm 0 . 0 0 } }$ </td><td>40.01</td><td> $6 4 . 5 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td>74.07</td><td> ${ \bf 9 1 . 2 0 _ { \pm 0 . 1 2 } }$ </td><td>67.34</td><td> $\mathbf { 8 7 . 3 9 } _ { \pm 0 . 0 3 }$ </td><td></td></tr></table>

The planner combines these inputs to produce a complete retrieval program $P _ { q } = F _ { \mathrm { p l a n } } ( q , I _ { q } ( \mathcal { M } ) )$ in one planning call. In restricted Python, it specifies operator calls, arguments, dependencies, and conditional retrieval actions based on intermediate evidence.

## 2.3.2 RETRIEVAL PROGRAM EXECUTION

Before execution, MaC checks the retrieval program’s structural compatibility with the available memory operations, including permitted language constructs, operator calls, variable references, and evidence dependencies. These dependencies connect function invocation, any selected view expansion, and subsequent evidence processing, preserving the information required by each operation. The query determines which computations and conditional continuations the program combines.

Once accepted, $P _ { q }$ executes over the code library L, with function calls resolved through the memory index I. Calls to base and derived functions reuse the fact definitions and state computations established during construction, while evidence composition preserves their fact identities and sources. Intermediate results determine which conditional continuations already specified in $P _ { q }$ are followed. Execution can therefore perform additional memory access when indicated by these conditions with out a new LLM planning decision. This planning–execution separation combines operator composition tailored to each query with reusable memory computation.

The final evidence assembly combines selected facts, supporting dialogue, and computation provenance into $E _ { q } ,$ loading source context as specified by the program; execution also produces a trace $\mathcal { T } _ { q } .$ . A separate answer model generates the response as $a _ { q } = F _ { \mathrm { a n s w e r } } ( q , E _ { q } )$ . A valid first attempt uses one planning call and a separate answer-generation call; checking or execution failures may trigger bounded repair attempts, followed by a fixed retrieval fallback if needed. Algorithm 1 in Appendix C summarizes construction, retrieval, and answer generation using the same notation.

## 2.4 COMPLEXITY ANALYSIS

One-time construction costs $T _ { \mathrm { b u i l d } } = O ( h \log h + n ^ { 2 } \log n + \mathcal { C } _ { \mathrm { b u i l d } } )$ , where n and h are the number of extracted facts and dialogue length in tokens and C<sub>build</sub> aggregates construction-stage model inference. Each query costs $\bar { T } _ { q } = { \cal O } \bar { ( } \mathcal { C } _ { q } + p ( n ^ { 2 } + h + b ^ { 2 } ) + p ^ { 2 } \bar { ) }$ time for a retrieval program with p syntax-tree nodes and evidence expansions of size up to b, with $\mathcal { C } _ { q }$ capturing planning and answering. Persistent codified memory occupies $O ( n + h )$ space because stored references satisfy $r = O ( n )$ and query processing adds $O ( p b )$ for program intermediates and execution traces. Appendix A.3 states the assumptions and gives the full time and space analysis.

Table 2: Performance across different question types on LONGMEMEVAL using the Qwen backbone. F1 and LLM-Judge accuracy (J) are reported in percent. Sgl.: single-session-user; Tmp.: temporal-reasoning; Pref.: single-session-preference; Upd.: knowledge-update. Bold indicates the best result, and underline indicates the second best in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2">Sgl.</td><td colspan="2">Tmp.</td><td colspan="2">Pref.</td><td colspan="2">Upd.</td><td colspan="2">Overall</td><td rowspan="2">Improv.</td></tr><tr><td>F1↑</td><td>J↑</td><td>F1↑</td><td>J↑</td><td>F1↑</td><td>J↑</td><td>F1↑</td><td>J↑</td><td>F1↑</td><td>J↑</td></tr><tr><td>A-Mem</td><td>71.74</td><td>84.29</td><td>32.41</td><td>36.84</td><td>7.05</td><td>50.00</td><td>61.81</td><td>73.08</td><td>47.16</td><td>59.40</td><td>+25.59%</td></tr><tr><td>MemoryOS</td><td>70.69</td><td>80.00</td><td>30.63</td><td>32.33</td><td>6.43</td><td>63.33</td><td>54.99</td><td>69.23</td><td>43.33</td><td>54.20</td><td>+37.64%</td></tr><tr><td>Mem0</td><td>82.63</td><td>92.86</td><td>34.62</td><td>54.89</td><td>19.15</td><td>70.00</td><td>62.71</td><td>76.92</td><td>50.95</td><td>68.40</td><td>+9.06%</td></tr><tr><td>MRAgent</td><td>73.87</td><td>88.57</td><td>34.03</td><td>56.39</td><td>6.94</td><td>53.33</td><td>43.21</td><td>53.85</td><td>43.67</td><td>61.20</td><td>+21.90%</td></tr><tr><td>MaC</td><td>84.28</td><td>94.29</td><td>53.28</td><td>68.42</td><td>14.49</td><td>73.33</td><td>69.69</td><td>87.18</td><td>58.58</td><td>74.60</td><td></td></tr></table>

## 3 EXPERIMENTS

## 3.1 EXPERIMENTAL SETUPS

In this section, we describe the experimental setup and present comprehensive results to systematically evaluate MaC and address the following research questions (RQs):

• RQ1 (Overall performance) How does MaC perform against representative long-term memory systems across benchmarks, question types, and LLM backbones?

• RQ2 (Cost, runtime, and storage) What are the token consumption, wall-clock runtime, and storage footprint of MaC during memory construction and retrieval?

• RQ3 (Ablation study) How do executable memory and programmatic search individually and jointly contribute to MaC’s performance?

Datasets. We evaluate MaC on two benchmarks for long-term conversational memory: LO-COMO (Maharana et al., 2024), which evaluates question answering over long conversations, and LONGMEMEVAL (Wu et al., 2024), which evaluates memory across multiple sessions with longer interaction histories per query.

Baselines. We compare MaC with representative baselines: A-Mem (Xu et al., 2026b), MemoryOS (Kang et al., 2025), Mem0 (Chhikara et al., 2025), and MRAgent Ji et al. (2026).

Implementation Details. We conduct our experiments on NVIDIA A100 GPUs. On LOCOMO, we evaluate all methods using three LLM backbones: Qwen3.6-27B, ChatGPT-5.6-Luna, and DeepSeek-V4-Flash. On LONGMEMEVAL, we use Qwen3.6-27B as the backbone for all methods. Following prior work, we report F1 and LLM-Judge (J) scores. LLM-Judge (J) scores are obtained using DeepSeek-V4-Flash. Our code is available at:https://github.com/zengaoting/ MemoryasCode.

## 3.2 MAIN RESULTS (RQ1)

To evaluate the effectiveness of MaC, we compare it with four representative memory systems on LOCOMO and LONGMEMEVAL, with three generation backbones evaluated on LOCOMO. Tables 1 and 2 summarize the results, from which we draw two main observations.

Overall performance. Tables 1 and 2 show that MaC achieves the highest overall J in all four evaluation settings. On LOCOMO, it outperforms the strongest baseline for each backbone by 3.75– 11.92 percentage points in overall J. On LONGMEMEVAL, it improves overall F1 and J by 7.63 and 6.20 points, respectively.

Temporal reasoning and knowledge updates. MaC achieves the best temporal F1 and J across all three LOCOMO backbones. On LONGMEMEVAL, it surpasses the strongest respective baselines by 18.66 points in temporal F1, 12.03 points in temporal J, and 10.26 points in knowledge-update J, highlighting its effectiveness on queries involving temporal constraints and evolving facts.

![](images/9486fbf8ef7a7624a8bd8baf7ae6c5c04969b31abab10480f0cf6c30e9b131f5.jpg)  
(a) Token Consumption

![](images/44d9957f35714aa5087780804b1749ce1de6db2b73da5bedcb417c63ebbec589.jpg)  
(b) Runtime  
Figure 3: Normalized token consumption (a) and runtime (b) on LOCOMO across different generation models. Construction is normalized by 10 conversations and retrieval by 1,540 questions. Both panels use logarithmic vertical axes.

Table 3: Per-sample token consumption and runtime on LONGMEMEVAL across different memory systems using the Qwen backbone. Results include both memory construction and retrieval. Bold indicates the best result, and underline indicates the second best.
<table><tr><td>Method</td><td>Token Consumption ↓</td><td>Runtime (s) ↓</td></tr><tr><td>A-Mem</td><td>1,723k</td><td>16,417.98</td></tr><tr><td>MemoryOS</td><td>636k</td><td>11,256.63</td></tr><tr><td>Mem0</td><td>4,812k</td><td>11,952.47</td></tr><tr><td>MRAgent</td><td>249k</td><td>2,963.90</td></tr><tr><td>MAC</td><td>173k</td><td>4,663.66</td></tr></table>

## 3.3 COST AND EFFICIENCY ANALYSIS (RQ2)

We report construction and query-processing costs separately on LOCOMO in Figure 3, and their combined per-sample costs on LONGMEMEVAL in Table 3.

Token consumption. Figure 3(a) shows that MaC requires the fewest tokens for memory construction across all three backbones on LOCOMO. This demonstrates that building an executable memory library and its interface index can be achieved with a lower construction token budget than the evaluated baselines. The advantage is most consistent during construction, while query-stage token consumption varies across backbones and methods. On LONGMEMEVAL, MaC also achieves the lowest combined token consumption, using approximately 173k tokens per sample compared with 249k for MRAgent, a reduction of approximately 30.5%.

Runtime. Figure 3(b) shows that MaC achieves competitive runtime across all three backbones on LOCOMO. It ranks first or second in memory construction time and consistently achieves lower query latency than MRAgent. On LONGMEMEVAL, Table 3 shows that MaC achieves the secondlowest end-to-end runtime among the evaluated methods, alongside the lowest token consumption. Together with the highest overall LLM-judge accuracy reported in Table 2, these results show that MaC maintains competitive computational efficiency while providing effective memory-based reasoning.

Storage space. As shown in Table 4, MaC maintains a competitive storage footprint among the methods evaluated on LOCOMO. MaC requires 2,859.18 KB of total storage, a 33.5% reduction compared with MRAgent, a strong baseline. Although A-Mem has a smaller storage footprint (1,338.88 KB), it achieves substantially lower accuracy than MaC under the same setting.

## 3.4 ABLATION STUDY (RQ3)

Table 5 reports sequential ablations on LOCOMO with DeepSeek-V4-Flash. MaC w/o DR retains the codified library but uses a fixed retrieval program; MaC w/o DR+CC applies that program to extracted atomic facts only.

Effect of declarative retrieval. MaC w/o DR keeps the codified library but uses a fixed retrieval program that combines lexical recall over facts with execution of candidate memory functions through predefined memory operators. Enabling DR improves overall J by 6.62 percentage points, demonstrating the benefit of adapting retrieval operations and their composition to each query.

Table 4: Storage space usage on LOCOMO (KB). Construction storage is averaged per conversation, and retrieval storage is averaged per question. Bold indicates the best result, and underline indicates the second best in each column.
<table><tr><td>Method</td><td>Construction ↓</td><td>Retrieval ↓</td><td>Total ↓</td></tr><tr><td>A-Mem</td><td>1,161.82</td><td>177.06</td><td>1,338.88</td></tr><tr><td>MemoryOS</td><td>11,452.02</td><td>415.59</td><td>11,867.61</td></tr><tr><td>Mem0</td><td>7,105.06</td><td>13.20</td><td>7,118.25</td></tr><tr><td>MRAgent</td><td>3,870.32</td><td>428.13</td><td>4,298.46</td></tr><tr><td>MaC</td><td>2,565.79</td><td>293.39</td><td>2,859.18</td></tr></table>

Table 5: Sequential ablation of declarative retrieval (DR) and codified construction (CC) on LO-COMO with DeepSeek-V4-Flash. F1 and binary LLM-judge accuracy (J) are percentages. Bold indicates the best result in each column. We use ↑ to indicate an improvement in the performance of one module, and ↑↑ to indicate improvements in two modules.
<table><tr><td rowspan="3">Method</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Open-domain</td><td colspan="2">Single-hop</td><td colspan="2">Overall</td><td rowspan="3">Improv.</td></tr><tr><td>F1 ↑</td><td>J↑</td><td>F1 ↑</td><td>J↑</td><td>F1↑</td><td>J↑</td><td>F1↑</td><td>J↑</td><td>F1↑</td><td>J↑</td></tr><tr><td>MaC w/o DR+CC</td><td>44.01</td><td>70.92</td><td>50.30</td><td>64.80</td><td>36.97</td><td>58.33</td><td>61.09</td><td>78.00</td><td>54.21</td><td>72.73</td><td>+20.17%</td></tr><tr><td>MaC w/o DR</td><td>50.54↑</td><td>85.46↑</td><td>59.63↑</td><td>69.78↑</td><td>39.00↑</td><td>62.50↑</td><td>67.15↑</td><td>85.49↑</td><td>60.79↑</td><td>80.78↑</td><td>+8.20%</td></tr><tr><td>MaC</td><td>52.99个</td><td>86.52↑</td><td>70.48↑↑</td><td>84.74</td><td>40.01↑↑</td><td>64.58</td><td>74.07</td><td>91.32↑</td><td>67.34↑</td><td>87.40个</td><td></td></tr></table>

Effect of codified construction. Under fixed retrieval, CC improves overall J by 8.05 percentage points, supporting the value of reusable memory computations and indexed access. Removing both DR and CC applies the same fixed program to extracted atomic facts only and reduces overall J by 14.67 percentage points relative to full MaC.

## 4 RELATED WORK

Memory Construction. Memory construction concerns retaining useful information from growing interaction histories and revising it as new interactions arrive. A related foundation is retrievalaugmented generation (RAG) (Lewis et al., 2020), which indexes external passages as dense vectors for generation; GraphRAG (Han et al., 2024) augments this setting with graph structure, community summaries, and neighborhood-based access. For persistent conversational memory, Mem0 (Chhikara et al., 2025) extracts salient facts and reconciles them with existing memory, whereas MemoryOS (Kang et al., 2025) organizes information across short-, mid-, and long-term stores. Other methods emphasize relational and temporal structure: A-Mem (Xu et al., 2026b) dynamically links and updates structured notes, and Zep (Rasmussen et al., 2025) records fact validity and historical changes in a bi-temporal graph. More recently, MemSkill (Zhang et al., 2026) learns to select and evolve memory-construction skills. Collectively, these methods improve information selection, consolidation, and structural organization through different memory representations and construction strategies.

Memory Retrieval. Memory retrieval addresses both locating relevant records and deciding how further evidence should be accessed and composed. SeCom (Pan et al., 2025) improves retrieved context through topic-based segmentation and compression-based denoising. Structured retrieval extends access beyond flat text matching: Zep (Rasmussen et al., 2025) combines semantic and fulltext search with graph traversal and reranking. In external-knowledge search, Search-o1 (Li et al., 2025) integrates on-demand search into document-based reasoning, and Search-R1 (Jin et al., 2025) uses reinforcement learning to learn multi-turn search behavior. At the execution-interface level, Text2Mem (Wang et al., 2026) translates memory instructions into typed, validated operations and supports workflows with dependencies across backends. MRAgent (Ji et al., 2026) uses intermediate evidence to guide exploration and pruning over a Cue–Tag–Content graph. These approaches improve context quality, structured access, or adaptive exploration; nevertheless, candidate-based retrieval remains bounded by the records selected for access, while iterative reasoning can require repeated context processing and model calls. Thus, retrieving sufficient evidence under a limited budget remains a central challenge.

## 5 CONCLUSION

We presented MaC, a unified framework that jointly designs memory construction and retrieval. We develop two complementary techniques: codified construction organizes facts, their evolving relations, and reusable computations into a three-layer executable memory structure. declarative retrieval compiles queries into complete programs that select and compose these computations, enabling query-specific conditional execution without stepwise LLM planning. Experiments on LOCOMO and LONGMEMEVAL show that MaC achieves the highest overall F1 and LLM-judge scores across the evaluated settings while substantially reducing token consumption.

## REFERENCES

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pp. 2206–2240. PMLR, 2022.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. In European Conference on Artificial Intelligence, 2025.

Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-augmented generation with graphs (graphrag). arXiv preprint arXiv:2501.00309, 2024.

Shuo Ji, Yibo Li, and Bryan Hooi. Memory is reconstructed, not retrieved: Graph memory for llm agents. arXiv preprint arXiv:2606.06036, 2026.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 25972–25981, 2025.

Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. An llm compiler for parallel function calling. arXiv preprint arXiv:2312.04511, 2023.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5420–5438, 2025.

Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, and Guannan Zhang. Think-in-memory: Recalling and post-thinking enable llms with long-term memory. arXiv preprint arXiv:2311.08719, 2023.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, 2024.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, Hao Cheng, Dongsheng Li, Yuqing Yang, Chin-Yew Lin, H Vicky Zhao, Lili Qiu, et al. On memory construction and retrieval for personalized conversational agents. arXiv preprint arXiv:2502.05589, 2025.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology, pp. 1–22, 2023.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.

Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, et al. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 8416–8439, 2025.

Leo Wang, Lihai Yang, Boyu Chen, Kerun Xu, Gongyi Zou, Bo Tang, Feiyu Xiong, Siheng Chen, and Zhiyu Li. Text2mem: A unified memory operation language for memory operating system. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 2105–2119, 2026.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2024.

Buqiang Xu, Yijun Chen, Jizhan Fang, Ruobin Zhong, Yunzhi Yao, Yuqi Zhu, Lun Du, and Shumin Deng. Structmem: Structured memory for long-horizon behavior in llms. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 122–146, 2026a.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2026b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. Memskill: Learning and evolving memory skills for self-evolving agents. arXiv preprint arXiv:2602.02474, 2026.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 19724–19731, 2024.

## A APPENDIX

## A.1 NOTATION

The table below follows the problem definition and summarizes the notation for memory construction, retrieval, and complexity analysis. MaC constructs M from dialogue history ${ \dot { \mathcal { H } } } ,$ executes a program $P _ { q }$ generated for query q to assemble evidence $E _ { q } ,$ and generates an answer $a _ { q }$ while recording an execution trace $\bar { \mathcal { T } } _ { q }$ . Local variables used only in Algorithm 1 are introduced alongside the algorithm.

Notation used throughout the paper
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td colspan="2">Task notation</td></tr><tr><td> ${ \mathcal { H } } , s _ { j } , n _ { \mathrm { s e s s } }$ </td><td>Dialogue history, its jth session, and the number of sessions.</td></tr><tr><td> $\mathcal { M }$ </td><td>External memory constructed from H.</td></tr><tr><td> $q , E _ { q } , a _ { q }$ </td><td>Query, supporting evidence assembled for it, and the generated answer.</td></tr><tr><td colspan="2">Codified memory</td></tr><tr><td> $\mathcal { L } , I$ </td><td>Code library and memory index, with  ${ \mathcal { M } } = ( { \mathcal { L } } , I ) .$ </td></tr><tr><td> $\mathcal { F } , \mathcal { V }$ </td><td>Base fact functions and views referencing base and derived functions.</td></tr><tr><td> $\mathcal { R } , \mathcal { D }$ </td><td>Explicit memory relations and derived memory functions.</td></tr><tr><td> $I _ { \mathrm { v i e w } } , I _ { \mathrm { f u n c } } , I _ { \mathrm { o p s } }$ </td><td>View-to-identifier mapping, function descriptors, and operator interfaces.</td></tr><tr><td> $i , f _ { i }$ </td><td>Canonical fact identifier and its base fact function.</td></tr><tr><td> $e _ { i } , v _ { i } , S _ { i }$ </td><td>Evidence record, structured value, and supporting source references for fact i.</td></tr><tr><td> $G , G ( \tau ) , A _ { G } ( \tau )$ </td><td>Linked state group, its temporally eligible fact identifiers, and its active fact identifiers.</td></tr><tr><td> $\tau , \bot$ </td><td>Requested time boundary and the marker for an unresolved state.</td></tr><tr><td colspan="2">Declarative retrieval</td></tr><tr><td> $K _ { q }$ </td><td>Candidate function identifiers supplied to the planner for  $q .$ </td></tr><tr><td> $I _ { q } ( \mathcal { M } )$ </td><td>Compact index context supplied to the planner for q.</td></tr><tr><td> $P _ { q } , \mathcal { T } _ { q }$ </td><td>Retrieval program for q and its execution trace.</td></tr><tr><td> $F _ { \mathrm { p l a n } } , F _ { \mathrm { a n s w e r } }$ </td><td>Model stages for program generation and evidence-grounded answering.</td></tr><tr><td colspan="2">Complexity analysis</td></tr><tr><td> $n , h$ </td><td>Number of facts before deduplication and source-dialogue length in to-</td></tr><tr><td> $p , b$ </td><td>kens. Syntax-tree size of  $P _ { q }$  and maximum expanded evidence size, counting</td></tr><tr><td>r</td><td>scalar entries and text tokens. Stored references across views, relations, derived functions, and source</td></tr><tr><td></td><td>annotations.</td></tr><tr><td> $T _ { \mathrm { b u i l d } } , T _ { q }$   $\mathcal { C } _ { \mathrm { b u i l d } } , \mathcal { C } _ { q }$ </td><td>Construction time and total processing time for query q. Model inference costs during construction and query processing.</td></tr></table>

## A.2 MEMORY OPERATORS

Table 6 summarizes the operators exposed to MaC search programs.

Table 6: Memory operators exposed to MaC search programs.
<table><tr><td>Category</td><td>Core Operation</td><td>Description</td></tr><tr><td rowspan="5">Memory Access</td><td>resolve</td><td>Executes selected memory functions.</td></tr><tr><td>match</td><td>Matches candidate facts to query keywords without code retrieval.</td></tr><tr><td>search</td><td>Ranks and retrieves canonical facts using a natural-language query.</td></tr><tr><td>select</td><td>Filters facts by structured constraints.</td></tr><tr><td>search_functions</td><td>Finds memory functions matching the query constraints.</td></tr><tr><td rowspan="3">Context Expansion</td><td>execute</td><td>Runs a memory function and returns evidence. Adds related memories from the same concept or session.</td></tr><tr><td>expand_scope expand_relation</td><td>Follows graph links to add temporally or semantically related facts.</td></tr><tr><td>hydrate</td><td>Loads canonical facts and nearby dialogue for answer context.</td></tr><tr><td rowspan="4">Evidence Composition</td><td>union</td><td>Merges evidence with its facts and dialogue sources.</td></tr><tr><td>rank</td><td>Orders evidence by relevance to the query.</td></tr><tr><td>where</td><td>Filters evidence by a specified time range.</td></tr><tr><td>aggregate</td><td>Counts evidence while retaining its facts and sources.</td></tr><tr><td>Verification</td><td>validate</td><td>Checks evidence completeness and unresolved states.</td></tr></table>

## A.3 COMPLEXITY ANALYSIS

This subsection expands Section 2.4. To characterize the cost of reusable memory computation, we distinguish one-time construction from per-query retrieval and account for model inference separately. Let n be the number of extracted facts before deduplication, so $| { \mathcal { F } } | \leq n ,$ and let h be the total length of the source dialogue H in tokens. For a query $q ,$ let $p$ be the number of syntax-tree nodes in $P _ { q }$ and b the maximum fully expanded size of an evidence collection materialized during retrieval, counting scalar entries and text tokens in its values and provenance. We assume bounded fact text, structured values, and query length, bounded annotations per extracted fact, and fixed retrieval and repair limits.

Time Complexity. Construction compares facts for deduplication and scans and sorts candidate neighbors for each canonical fact. The latter costs $O ( n ^ { 2 } \log { n } )$ in the worst case, even though only a bounded number of neighbors is retained per structural relation. Including source preprocessing and model inference, the construction bound is $T _ { \mathrm { b u i l d } } = O ( h \log h + n ^ { 2 } \log n + \mathcal { C } _ { \mathrm { b u i l d } } ) \stackrel { } { = }$ . During retrieval, candidate matching scans and sorts the memory index, while view expansion and provenance composition can incur quadratic work through list-based deduplication. Structural program checking takes $O ( p )$ time, and the loop-free language permits at most $O ( p )$ operator calls. Accounting for memory access, source loading, evidence composition, and trace generation gives the conservative bound $\dot { T } _ { q } = O \big ( \mathcal { C } _ { q } + p ( n ^ { 2 } + \check { h } + b ^ { 2 } ) + p ^ { 2 } \big )$ . Here $ { \mathcal { C } } _ { \mathrm { b u i l d } }$ and $\mathcal { C } _ { q }$ sum the model inference costs in the corresponding stage, including extraction, semantic compilation, retrieval preprocessing, planning, and answer generation as applicable. This accounting includes both operator execution and the model calls used across the query pipeline.

Space Complexity. Let r count stored references across memory views, fact relations, derived functions, and source annotations. The code library, memory index, construction records, and source dialogue require $O ( n + r + h )$ ) space. A query additionally retains intermediate evidence and its execution trace $\mathcal { T } _ { q } ,$ giving $O ( n + r + h + p b )$ space, excluding model weights and inference workspace. Under the stated annotation bounds and the implemented relation limits, $r = O ( n )$ , so memory storage is linear in the input size. The pb term accounts for the generated program’s intermediate results, including matching pairs materialized by joins and their provenance.

## A.4 CASE STUDY (RQ4)

Figure 4 presents a temporal-reasoning case in which the system must connect the user’s keyboardplaying activity with a later bluegrass-band discovery across separate dialogue sessions. MaC generates a question-specific search program that retrieves the relevant activity memories, expands their temporal relations to recover the event needed by the question, and restores the associated source dialogue for temporal grounding. From the recovered evidence, the answer model correctly infers an interval of 6 days; its cited evidence exactly matches the annotated evidence. This example demonstrates how programmatic memory search combines structured retrieval, relation-based expansion, and source-context recovery to answer temporally grounded conversational questions.

## B PROMPT DETAILS

## Box 1: Fact to Code Prompt

Task: Convert canonical facts into structured specifications that a local generator turns into executable Python memory functions.

Input: Canonical facts with identifiers, factual content, subjects, temporal information, and source references.

System prompt:

You are a semantic compiler. Produce one function specification for each fact while preserving its identifier and meaning. Specify a concise Python function name, a query predicate, the factual value, optional attributes, and whether the fact describes a time-varying state.

Use only the supplied facts. Do not invent information or infer state-update links in this stage. The specifications will be converted locally into Python functions.

User prompt template:

![](images/be539e2d95caa260ec9f56482907732f67f734a501a855ed1cf2eb64c9dec6f9.jpg)

![](images/2bd8fb746a86d701ece2cb0e849b858479b22b6e7a59ab7f65ee23c3bc8fedd7.jpg)  
Figure 4: Temporal-reasoning case study on LOCOMO. MaC generates a question-specific retrieval program, expands temporal relations across sessions, and hydrates source dialogue; the answer model infers a 6-day interval matching the annotated evidence.

## Box 2: Declarative Query Planning Prompt

![](images/4a2654ec97da77606c3813fda0cbd67453867e1eb8a56b45ac7c124550e9ab7f.jpg)

## C FRAMEWORK ALGORITHM

Algorithm 1 uses the memory and query notation defined in Section 2. The local collections ${ \mathcal { X } } _ { \mathrm { r a w } }$ and X contain extracted and canonical fact records, respectively; a canonical record $x _ { i }$ supplies the value $v _ { i }$ and source references $S _ { i }$ encoded by $f _ { i } ~ \in ~ \mathcal { F }$ . For a query $q , K _ { q } ^ { \mathrm { v i e w } }$ contains function identifiers obtained through view expansion, and $E _ { q } ^ { \mathrm { c o d e } } , E _ { q } ^ { \mathrm { v i e w } } ;$ , and $E _ { q } ^ { \mathrm { l e x } }$ denote evidence from code recall, view expansion, and lexical recall. Their merged evidence is cached in $B _ { q } ;$ the program returns intermediate evidence $E ,$ which is assembled into the final evidence $E _ { q } .$ Here G denotes the LLM used for extraction, annotation, and repair, while $F _ { \mathrm { p l a n } }$ and $F _ { \mathrm { a n s w e r } }$ denote the planning and answering stages defined in the main text. Only values used by subsequent steps are shown.

Algorithm 1: Memory as Code (MaC) Framework   
Input: Dialogue history $\mathcal { H } = ( s _ { 1 } , \dots , s _ { n _ { \mathrm { s e s s } } } ) ,$ query set Q, LLM G   
Output: Codified memory ${ \mathcal { M } } .$ , answers $\{ a _ { q } \ | \ q \in Q \}$ , and traces $\check { \{ T _ { q } | q \in Q \} }$   
1: ▷ Stage 1: Codified Construction   
2: $\mathcal { X } _ { \mathrm { r a w } }  \emptyset$   
3: foreach non-empty dialogue session $s _ { j } \in \mathcal { H }$ do   
4: Extract source-annotated records with G.ExtractAtomicFacts(s<sub>j</sub>) and add them to ${ \mathcal { X } } _ { \mathrm { r a w } }$   
5: $\mathcal { X } \gets$ CanonicaliseAndSemanticDeduplicate $\mathbf { \nabla } ( \mathcal { X } _ { \mathrm { r a w } } ) ;$ assign each record $x _ { i }$ a canonical ID i   
6: Store X in $\pounds \mathrm { c t } s \_ \mathrm { b y \_ i d } .$ .json; initialize R with temporal, same-subject, and shared-topic relations   
7: Annotate records in $\mathcal { X }$ using $\mathcal { G } ;$ group state facts by subject and predicate   
8: Use $\mathcal { G }$ to identify explicit state replacements and add (j, supersedes, i) to $\mathcal { R }$   
9: $\mathcal { F }  \emptyset$   
10: foreach canonical record $x _ { i } \in \mathcal X$ do   
11: Add to $\mathcal { F }$ a function $f _ { i }$ returning $e _ { i } = ( v _ { i } , \{ i \} , S _ { i } )$ ; emit its applicable supersedes references   
12: Construct views V, derived current/history functions D, and code for R   
13: Construct $\begin{array} { r } { \mathcal { L } = ( \mathcal { F } , \mathcal { V } , \mathcal { R } , \mathcal { D } ) , I = ( I _ { \mathrm { v i e w } } \mathbf { \bar { , } } I _ { \mathrm { f u n c } } , I _ { \mathrm { o p s } } ) , } \end{array}$ and $\mathcal { M } = ( \mathcal { L } , I )$   
14: Persist memory\_code, function\_index.json, and view\_index.json   
15: ▷ Stage 2: Declarative Retrieval   
16: foreach query $\boldsymbol { l } \in Q$ do   
17: $K _ { q } \gets \dot { \mathrm { C o d e R e c a l l } } ( q , I _ { \mathrm { f u n c } } , 2 4 )$ in parallel with term $\mathbf { s } _ { q }  { \mathcal { G } } .$ .ExtractRetrievalTerms(q)   
18: E<sup>code</sup><sub>q</sub> ← ExecuteAndRehydrate $( K _ { q } , \mathcal { M } ) ;$   
$\dot { E } _ { q } ^ { \mathrm { l e x } }$ ← KeywordRecall(terms<sub>q</sub>, $\mathtt { f a c t s \_ b y \_ i d , 3 2 } )$   
19: $K _ { q } ^ { \mathrm { v i e w } }$ ← ViewExpandAndRank $( K _ { q } , I _ { \mathrm { v i e w } } , I _ { \mathrm { f u n c } } , q , 3 2 ) ;$   
$\dot { E } _ { q } ^ { \mathrm { v i e w } }$ ← ExecuteAndRehydrate $K _ { q } ^ { \mathrm { v i e w } } , \mathcal { M } )$   
20: $\mathcal { B } _ { q } \gets \mathrm { D e d u p l i c a t e } ( E _ { q } ^ { \mathrm { c o d e } } \cup E _ { q } ^ { \mathrm { v i e w } } \cup \dot { E } _ { q } ^ { \mathrm { l e x } } )$   
21: Form $I _ { q } ( \mathcal { M } ) ^ { \overline { { } } }$ from $\{ I _ { \mathrm { f u n c } } ( \kappa ) \mid \stackrel { \cdot } { \kappa } \in K _ { q } \}$ and $I _ { \mathrm { o p s } }$   
22: $P _ { q }  \dot { F } _ { \mathrm { p l a n } } ( q , I _ { q } ( \dot { \mathcal { M } } ) )$   
23: for $\ell \gets 0$ to 2 do   
24: if ValidatePlan( $\left( P _ { q } \right)$ and RunInNetworkIsolatedSandbox $( P _ { q } , { \mathcal { M } } , B _ { q } )$ succeed then   
25: E ← ExecutionResult $( P _ { q } )$ ; break   
26: $P _ { q }  \mathcal { G }$ .RepairProgram $( P _ { q } )$   
27: if no generated program succeeds then   
28: $\breve { P } _ { q } \gets$ DefaultFallbackProgram $( I _ { q } ( \mathcal { M } ) )$   
E ← RunInNetworkIsolatedSandbox $\left( P _ { q } , \mathcal { M } , B _ { q } \right)$   
29: $E _ { q } \gets$ RenderEvidenceContext(E, $B _ { q } , { \mathcal { M } } ) ; a _ { q } \gets F _ { \mathrm { a n s w e r } } ( q , E _ { q } )$   
30: if $a _ { q }$ is malformed, empty, or a refusal then   
31: $\dot { \boldsymbol { a } } _ { q } \gets \boldsymbol { \mathcal { G } } .$ .RepairAnswer(q, $\dot { E } _ { q } , a _ { q } )$   
32: Record $P _ { q } , E _ { q } , a _ { q } ,$ , and line-level provenance in $\mathcal { T } _ { q }$   
33: return M, $\{ a _ { q } \ | \ q \in Q \}$ , and $\{ \mathcal { T } _ { q } \ | \ q \in Q \}$