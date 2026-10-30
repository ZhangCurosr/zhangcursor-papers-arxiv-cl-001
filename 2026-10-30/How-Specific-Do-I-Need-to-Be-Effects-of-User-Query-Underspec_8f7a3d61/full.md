# How Specific Do I Need to Be? Effects of User Query Underspecification on Deep Research Agents

Yi Wen<sup>1,</sup>\*, Xiangjue Dong<sup>2,\*</sup>, Bingxuan Li<sup>3</sup>, Hangxiao Zhu<sup>2</sup> Leo Zhicheng Liu<sup>1</sup>, Yu Zhang<sup>2,†</sup>, James Caverlee<sup>2,†</sup>

<sup>1</sup> University of Maryland, College Park <sup>2</sup> Texas A&M University <sup>3</sup> University of Illinois, Urbana-Champaign

## Abstract

Deep research agents (DRAs) conduct longhorizon reasoning, search, and literature synthesis to provide long-form answers to complex, domain-specific questions. These agents are typically trained and evaluated on fully specified user queries. However, real-world user queries can often be underspecified. We introduce UNDERSPEC-DR, a framework for systematically evaluating how underspecified queries affect agents’ final answer quality and intermediate search behavior. Across three DRAs we evaluated, underspecification substantially reduces how well DRA’s answers fulfill users’ information needs, even though they remain grounded in cited evidence. Our analysis of intermediate search behavior suggests a possible explanation: DRAs conduct broader searches that draw less on the original user query but rely more on their learned prior during training, which can be misaligned with users’ intent. Our findings reveal the vulnerability of current DRAs to underspecified user queries and inform future efforts to build DRAs that proactively recognize when additional user input is needed to reliably complete deep research tasks.

## 1 Introduction

Deep research agents (DRAs), which autonomously plan multi-step searches, retrieve documents, read and synthesize them into longform reports with citations, have been deployed in both commercial (OpenAI, 2025; Haas and Basu Mallick, 2025) and open-source systems (Shao et al., 2026; Tongyi DeepResearch Team et al., 2025). Existing benchmarks (Mialon et al., 2024; Wei et al., 2025; Asai et al., 2026; Yifei et al., 2026; Du et al., 2026) for evaluating DRAs carefully construct fully-specified user queries to enable consistent assessment of the agent’s deep research capability. Deep research requires precise questions, but in practice, users often cannot formulate their query as required. They may not yet have a clear understanding of their information needs, or they may lack the domain knowledge needed to express those needs precisely (Taylor, 1968; Belkin et al., 1982). This creates a fundamental tension: DRAs are designed to answer questions that users cannot easily answer themselves, yet they are developed and evaluated on the assumption that users can state those questions perfectly. This tension motivates the central question of our work: Can DRAs handle cases when users cannot fully articulate those questions in thefirst place?

Prior work has examined how underspecified user queries affect performance in both standalone LLMs (Yang et al., 2026) and agentic settings (Cao et al., 2026; Vijayvargiya et al., 2026), and has developed methods to evaluate and improve their ability to resolve ambiguity (Sun et al., 2026). There are verifiable outcomes under these settings, such as whether a stated requirement is met or whether the code generated passed the test cases. However, DRAs produce long, open-ended reports that are difficult to verify, which makes it hard to measure how much underspecification harms the final answer DRAs generated and whether the they resolved ambiguity. Moreover, unlike standalone LLMs, DRAs interact with external tools over longhorizon task execution. Inspecting this intermediate process might provide an additional lens for understanding the effect of user query underspecification on DRA.

To this end, we introduce UNDERSPEC-DR, a framework for investigating DRA’s answer quality and intermediate search behavior under user queries with controlled levels of underspecification, as illustrated in Figure 1. To ensure that the underspecified queries reflect realistic forms of underspecification, we analyze initial–final query pairs collected from real-world researchers’ conversation logs with DRAs. We identified three types of information that are frequently omitted: scope constraints (e.g., temporal, spatial, or population constraints), terminology (e.g., technical or domainspecific terms), and user preferences (e.g., evaluation criteria or the desired answer format). We then construct underspecified versions of three existing deep research benchmarks that features openended questions requiring long-form answers: ResearchQA (Yifei et al., 2026), ScholarQA-Bench (Asai et al., 2026), and DeepResearch Bench (Du et al., 2026).

![](images/1146dcf932963825f0b9eeefb9b9fb273b2833b7ac4403028c8e5d807a61828d.jpg)  
Figure 1: Overview of UNDERSPEC-DR. (a) From 176 initial–final query pairs in researchers’ conversations with DRAs, we identify three omitted information types: scope (q ), terminology (q ), and preference $( q _ { p } )$ . We then rewrite benchmark queries into matched variants with an LLM followed by human verification. (b) Each agent answers the complete query q or a variant q<sub>r</sub> by alternating search queries s<sub>k</sub> and retrieved documents $D _ { k }$ . Highlighted search-query tokens come from the user query (blue), earlier search queries (yellow), retrieved documents (green), or parametric knowledge (red). (c) We analyze answer quality (RQ1), whether agents recognize underspecification (RQ2), and search behavior (RQ3).

We evaluate three widely used DRAs, DR Tulu (Shao et al., 2026), Tongyi DeepResearch (Tongyi DeepResearch Team et al., 2025), and GPT 5.4 with the same tool set as DR Tulu, on both the complete queries and their underspecified variants. We assess how well their final answers satisfy the query-specific rubrics and how well their claims are supported by the cited evidence. Additionally, we examine whether DRAs recognize underspecified user query, what actions they take upon recognizing it, and whether these actions mitigate the impact of underspecification on final answer quality. To characterize changes in intermediate search behavior, we examine how search queries evolve across steps, whether successive searches introduce new documents, and whether retrieved documents inform subsequent searches.

Our study contributes three main findings:

(1) Query underspecification does not lead to more hallucination, but likely leads to answers that are misaligned with user’s information needs. Omitting or obscuring even one key detail like scope constraints or technical terms significantly reduces how well answers satisfy the query-specific rubrics. In contrast, even the most underspecified queries do not lead to more hallucinations.

(2) DRAsfill in missing information based on their learned priors during training rather than explicitly eliciting the missing informationfrom user. They struggle to recognize when user queries lack critical information and, even when they do, they often proceed without seeking clarification from the user. This tendency is also reflected in their intermediate search behavior: their search queries rely less on the user’s original query and more on the DRA’s parametric knowledge.

(3) Query underspecification leads to more diffuse search without greater use of retrieved evidence. DRAs conduct broader searches with greater drift in their search query and retrieve more previously unseen documents. However, they do not make greater use of retrieved content in subsequent searches.

We hope our findings pave the way for developing DRAs that recognize unresolved information gaps and make them explicit before proceeding with the research task.

## 2 Related Work

Deep research agents. Deep research agents extend tool-augmented reasoning (Yao et al., 2023) with iterative search and long-form synthesis, and are trained with reinforcement learning for search (Jin et al., 2025; Li et al., 2025a,b), agentic midand post-training (Tongyi DeepResearch Team et al., 2025), or evolving rubrics (Shao et al., 2026). Their benchmarks consist of fully specified queries (Mialon et al., 2024; Wei et al., 2025; Asai et al., 2026; Yifei et al., 2026; Du et al., 2026); we construct underspecified counterparts and analyze the research process in addition to the output.

Query underspecification and ambiguity resolution. Prior work examines how omitted requirements affect LLM performance (Yang et al., 2026) and evaluates or improves agents’ ability to handle underspecified queries in software engineering (Vijayvargiya et al., 2026), question answering (Cao et al., 2026; Su and Cardie, 2026; Xie et al., 2026; Du and Hu, 2026), and other tasks with easy-toverify answers (Sun et al., 2026; Liu et al., 2026). In these settings, unresolved ambiguity can lead to observable task failures or deviations from user’s expected outcomes, providing relatively clear signals of when clarification is needed. For openended deep research, however, report-style answers are difficult to verify, and the effects of query underspecification remain unclear.

Given that human experts can select appropriate strategies to resolve ambiguity in such situations (Freeman et al., 2025), we seek to understand how to enable agents to develop similar capabilities. Prior work categorizes LLM’s disambiguation policies into query rewriting, long-form answer generation, and asking clarifying questions (Tanjim et al., 2025), and highlights the need for agentic orchestration of different disambiguation strategies based on their trade-offs. Prior work explores uncertainty quantification to inform clarification decisions (Zhang and Choi, 2025; Kuhn et al., 2023), while other work uses preference optimization to align LLM responses with human experts’ communication practices, potentially improving questionasking behavior in specific domains (Seegmiller et al., 2026). However, extending these approaches to long-horizon agentic settings remains underexplored, where agents need to decide whether uncertainty can be resolved through further interaction with the environment or requires additional information from the user.

## 3 The UNDERSPEC-DR Framework

## 3.1 Problem Formulation

Let q denote a fully specified query (complete query) that expresses a user’s information need, and let $S _ { d } ( a ; q )$ denote an evaluation function that scores answer a on one dimension d of answer quality $( \mathrm { e . g . }$ ., rubric coverage or citation precision). An underspecification operator $u _ { r }$ , with $r \in \{ \mathrm { s } , \mathrm { T } , \mathrm { P } \}$ removes scope constraints $( r \ = \ s )$ or obscures technical terminology $( r ~ { \bf \Omega } = { \bf \Omega } \mathrm { T } )$ or user preferences $( r ~ = ~ { \bf P } )$ in $q ,$ yielding the query variant $q _ { r } = u _ { r } ( q )$ . Each operator is applied only when the corresponding information is present in q and the fully underspecified variant $q _ { \mathrm { U } }$ is obtained by applying all applicable operators jointly. Given a query, an agent π produces a search trajectory $\tau = ( s _ { 1 } , D _ { 1 } , \ldots , s _ { m } , D _ { m } )$ , where $s _ { k }$ is the k-th search query and $D _ { k }$ is the corresponding set of retrieved documents retrieved for it, follwed by a final answer $a _ { \pi } ( \cdot )$ , respectively. We quantify the effect of underspecification on answer quality as

$$
\Delta _ { r , d } ^ { \mathrm { a n s } } = \mathbb { E } _ { q } \big [ S _ { d } ( a _ { \pi } ( q _ { r } ) ; q ) - S _ { d } ( a _ { \pi } ( q ) ; q ) \big ] ,\tag{1}
$$

where the expectation is taken over the queries to which $u _ { r }$ applies. By evaluating both answers against the complete query $q , \Delta _ { r , d } ^ { \mathrm { a n s } }$ measures how underspecification affects the agent’s ability to satisfy the original information need.

Similarly, we quantify changes in intermediate search behavior as

$$
\Delta _ { r , l } ^ { \mathrm { s e a r c h } } = \mathbb { E } _ { q } \bigl [ M _ { l } ( \tau _ { \pi } ( q _ { r } ) ) - M _ { l } ( \tau _ { \pi } ( q ) ) \bigr ] ,\tag{2}
$$

where $M _ { l }$ measures an aspect of the search behavior (Section 4).

## 3.2 Deriving Dimensions of Underspecification

To ground the framework in real user behavior, we collect the conversation logs of three junior researchers in electrical engineering, computer science, and construction science fields with DR Tulu (Shao et al., 2026) over a two-week period. See details in Appendix C. We focused on junior researchers because they constitute a substantial segment of deep research agent users and may struggle to fully articulate their information needs due to limited domain expertise. From these logs, we select conversations in which they reformulated their initial queries after receiving an answer and we pair the initial query with the final query of each selected conversation and compare them to identify changes in wording and explicitly stated information. Of the 203 initial–final query pairs, 176 contained information in the final query that was absent from the initial query.

<table><tr><td rowspan="2">Rubric</td><td rowspan="2">Removed / Obscured Information</td><td rowspan="2">Example</td><td colspan="3"># Query Variants</td></tr><tr><td>RQA</td><td>SQA</td><td>DRB</td></tr><tr><td>R1 Scope (qs)</td><td>Time, place, population, domain</td><td>Which regions and countries saw widespread acceptance and im- plementation of home delivery services for antiretroviral therapy (ART) during the COVID-19 pandemie?</td><td>274</td><td>129</td><td>74</td></tr><tr><td>R2 Term (qτ)</td><td>Precise technical terms (replaced by generic words)</td><td>How do non-aspirin NSAIDspain relievers and selective COX-2 enzyme inhibitors differ in their associations with aneurysmal sub- arachnoid hemorrhage risk, and what is known about the hetero- geneity across studies for these drug classes?</td><td>238</td><td>126</td><td>45</td></tr><tr><td>R3 Preference (qP)</td><td>Stated criteria, comparisons, or re- quirements on the answer</td><td>Which Obsidian plugins can effectively replicate Notion&#x27;s multi- view database functionality (including Table, Kanban, Calendar, and List views)? Please provide a detailed comparison of the strengths and weaknesses of these plugins.</td><td>27</td><td>65</td><td>58</td></tr><tr><td>Fully-Underspecified (qu)</td><td>R1-R3 jointly</td><td>How does the incidence of imminent fractures differ depending on the site of the (index) initial fracture according to pooled meta-analyses?</td><td>499</td><td>287</td><td>100</td></tr></table>

Table 1: Underspecification rubrics, examples, and number of constructed variants in ResearchQA (RQA), ScholarQA-Bench (SQA), and DeepResearch Bench (DRB). Colors distinguish scope, terminology, and preference edits; struck text is removed and bold text denotes replacements.

We characterize the types of information omitted from users’ initial queries through inductive coding analysis (Thomas, 2006). The first and second authors independently examine each initial–final query pair, identify information that is newly specified in the final query, and assign descriptive codes to characterize the type of information added. They then compare their codes and categories, discuss discrepancies, and consolidate conceptually similar categories through consensus. This iterative process result in three higher-level categories of information that users commonly omitted from their initial queries: scope, terminology, and preferences. Scope specifies the boundaries of the research question, such as the time period, place, population, or domain. Terminology identifies precise technical concepts or domain-specific entities. Preferences specify the user’s expectations for the answer, such as evaluation criteria, comparison requirements, or presentation format. Together, these three categories cover 94% of the 176 pairs: missing scope (45%), missing terminology (37%), and missing preferences (33%). 21% of the pairs lack more than one type of information.

## 3.3 Constructing Matched Variants

We formalize each category as a rewriting rubric (Table 1) and prompt GPT-5.4 mini to generate underspecified query variants (see Appendix A.1 for the prompts). To verify whether these variants resemble naturally occurring underspecified queries, we conduct a Turing-style discrimination test (Turing, 1950). We first apply the rewriting procedure to the clarified queries in our collected query pairs from real users. We then pair each generated variant with its corresponding human-written initial query and ask a separate model to identify which query is AI-generated and which is human-written.

We repeat the test twice, shuffling the order of the human-written and AI-generated queries in each run. Across 486 generated variants derived from 176 clarified queries, GPT-4o achieves identification accuracies of 58% and 54%, respectively, slightly higher than random guessing. This result suggests that the generated variants are reasonably similar to naturally occurring underspecified queries, and that the rewriting procedure can be used to construct matched query variants for existing deep research benchmarks.

We apply this rewriting procedure to three deep research benchmarks: 500 randomly sampled queries from ResearchQA, the ScholarQA-CS, ScholarQA-Multi, and ScholarQA-CS2 subsets of ScholarQA-Bench, and all queries of Deep-Research Bench. Table 1 reports the number of resulting variants for each rubric.

## 3.4 Human Validation of Synthetic Query Variants

The first and second authors each validated half of the generated query variants, checking whether the targeted information was removed or obscured, non-target information was preserved, and the resulting query remained natural and coherent. Overall, 6% of the variants were revised to improve naturalness, and 8% were revised to ensure that technical terminology was appropriately obscured.

![](images/56bb472d28aa8534625d65efa60899a46053545fc6db6b5e9f1f16881b286c9a.jpg)  
Figure 2: Effects of query underspecification on answer quality across three benchmarks and DRAs. Top: Mean paired score changes (underspecified variant minus complete query). Filled and outlined bars indicate significant and nonsignificant differences, respectively, based on paired Wilcoxon signed-rank tests $( ^ { * } p < . 0 5 , ^ { * * } p < . 0 1$ $^ { * * * } p < . 0 0 1 )$ . Bottom: Proportions of matched queries with lower scores (below zero) or higher scores (above zero), with colors indicating the magnitude of change. Segment labels are percentages; score changes within ±0.02 are omitted.

## 4 Experimental Setup

Models and harnesses. We evaluate three common deep research agents: DR Tulu-8B (Shao et al., 2026), which operates within an MCP-based agent framework with three tools: web search, scientific paper search, and webpage browsing. Its interaction protocol supports reasoning, tool calling, answer generation, and citation. The second one is GPT-5.4 with tools, which uses the same agent framework and tool set as DR Tulu. Tongyi-DR (Tongyi DeepResearch Team et al., 2025) is equipped with tools for web search, webpage browsing, Google Scholar search, Python execution, and file parsing. Unlike DR Tulu-8B, which submits a single query per search tool call, Tongyi-DR supports multiple queries within a single search tool call.

Benchmarks and answer quality metrics. We use three benchmarks of open-ended research questions requiring long-form answers: 500 randomly sampled queries from ResearchQA (Yifei et al., 2026); the ScholarQA-CS, ScholarQA-Multi, and ScholarQA-CS2 subsets of ScholarQA-Bench (Asai et al., 2026); and DeepResearch Bench (Du et al., 2026). We use Gemini 3.1 Pro as the judge to score each answer for rubric-based coverage, following each benchmark’s official evaluation protocol: query-specific rubrics for ResearchQA and ScholarQA-Bench, and general rubrics with a reference answer for DeepResearch Bench. These scores measure how well an answer fulfills the original query’s information needs. We also assess the extent to which answers are grounded in retrieved evidence, using Document utilization to measure how much of the retrieved material is cited and Citation precision to measure whether the cited evidence supports the answer’s claims. For each answer, we compute Document Utilization $= \ \# _ { \mathrm { c i t e d } } / \# _ { \mathrm { r e t r i e v e d } }$ and Citation Precision $= \ \# \mathrm { s u p p o r t i n g } / \# \mathrm { c i t e d }$ where $\# _ { \mathrm { r e t r i e v e d } }$ is the total number of documents retrieved during the whole search process, $\# _ { \mathrm { c i t e d } }$ is the number of documents cited in the final answer, and $\# _ { \mathrm { s u p p o r t i n g } }$ is the number of cited documents that support the claims for which they are cited. To compute $\# _ { \mathrm { s u p p o r t i n g } } ,$ we use Gemini 3.1 pro as a judge to examine the content of each cited document and assess whether it supports the associated claims in the answer. Only ScholarQA-Bench and DeepResearch Bench support separate evaluation of citation quality, so we evaluate citation-related metrics only for DR Tulu-8B and GPT-5.4 with tools on these two benchmarks, excluding Tongyi-

<table><tr><td>Model</td><td>Metric</td><td>Fully underspec.</td><td>w/o preference</td><td>w/o scope</td><td>w/o term</td></tr><tr><td rowspan="2">DR Tulu</td><td>Document Utilization</td><td> $+ 2 . 8 5 \% ^ { * * * }$ </td><td> $+ 0 . 0 1 \%$ </td><td> $+ 1 . 5 1 \% ^ { * }$ </td><td>+0.07%</td></tr><tr><td>Citation Precision</td><td> $+ 0 . 4 5 \%$ </td><td>+1.40%</td><td> $- 2 . 8 8 \%$ </td><td>+2.17%</td></tr><tr><td rowspan="2">GPT-5.4 w/ Tools</td><td>Document Utilization</td><td>-0.44%</td><td>+1.98%</td><td> $+ 2 . 1 1 \% ^ { \ast * }$ </td><td>-0.45%</td></tr><tr><td>Citation Precision</td><td>-2.14%</td><td>+4.68%</td><td>-0.48%</td><td>+0.83%</td></tr></table>

Table 2: Mean paired changes in document utilization in the final answer and citation precision (underspecified variant minus complete query), pooled across datasets within each model. Values are displayed as percentages and represent percentage-point differences. Significance is assessed using unadjusted paired Wilcoxon signed-rank tests. Stars denote $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1$ , and $^ { * * * } p < 0 . 0 0 1$

DR because it does not include explicit citations in its final answers. For rubric-based coverage, we use Wilcoxon signed-rank tests with Bonferroni correction for 12 comparisons per agent (three benchmarks × four variants) to assess evidence of systematic paired score changes. For document utilization and citation precision, we report unadjusted Wilcoxon signed-rank tests, which is conservative for our conclusion that these metrics do not decline.

Search behavior metrics. We adopt measures from prior work on agentic search behavior analysis. Query drift (Ning et al., 2026) measures the semantic shift between the first and last search queries in the whole deep research process. For a trajectory containing m search queries, we compute drift as $1 - \cos ( \mathbf { v } _ { 1 } , \mathbf { v } _ { m } )$ , where $\mathbf { v } _ { 1 }$ and ${ \bf v } _ { m }$ are the embeddings of the first and final search queries, respectively, obtained using OpenAI’s text-embedding-3-small model. Higher values indicate greater semantic drift. Query trajectory type (Ning et al., 2026) characterizes how consecutive search queries are reformulated. We use GPT-5.4 mini to classify each transition $s _ { k } \ \to \ s _ { k + 1 }$ into one of four types: Specialization (adding constraints to narrow the search), Generalization (relaxing constraints to broaden the search), Exploration (shifting to a different facet of the same topic), or Repetition (issuing an identical or nearduplicate query). We report the proportion of transitions of each type in a complete trajectory. Query provenance (Zhang et al., 2026) attributes each token of $s _ { k }$ to the first of four sources in which it occurs: the user query, the previous search queries $s _ { < k }$ , the retrieved documents $D _ { < k }$ , or none of these (untraced; attributed to parametric knowledge or memory). Provenance shares are computed per search tool call and then averaged. Document novelty (Zhang et al., 2026) measures the fraction of documents retrieved at each search step that have not appeared in earlier steps. Higher values in dicate more newly retrieved documents, whereas lower values indicate greater retrieval redundancy. Context-driven term adoption rate (Ning et al., 2026) measures the fraction of newly introduced search-query terms that appear in previously retrieved documents. Higher values indicate greater lexical adoption of retrieved evidence in subsequent searches. For search behavior measures, we use paired t-tests on per-trajectory values; Wilcoxon signed-rank tests yield the same conclusions. We report Cohen’s $d _ { z }$ as the effect size (Cohen, 1988).

## 5 RQ1: How Does Underspecification Affect Answer Quality?

DRAs cannot reliably infer users’ information needs. Figure 2 shows that rubric-based coverage scores decline for a large proportion of underspecified queries even when only one key detail is omitted or obscured. Among different information types, removing scope constraints or obscuring technical terminology produces substantial mean score reductions that are statistically significant across most agents and benchmarks. In contrast, removing user preferences generally produces smaller reductions that are often not statistically significant. As user queries become more underspecified, scores decline further.

Query underspecification does not lead to more hallucination. Table 2 shows no statistically significant change in citation precision even under full underspecification condition. Document utilization in the final answer also shows no statistically significant decline: it even increases in three of the eight settings. Together with the decline in rubric-based coverage, these results suggest that underspecification primarily reduces alignment with the user’s original information needs, with no evidence of increased hallucination among cited claims.

Stronger specifically-trained deep research capability come with greater sensitivity to query underspecification. Using performance on complete queries as a measure of deep research capability, Figure 2 shows that higher-scoring agents generally incur larger absolute score losses under full underspecification. Both DR Tulu-8B (8B) and Tongyi-DR (30B) are relatively compact models trained specifically for deep research. They outperform GPT-5.4 with tools on complete queries but appear to depend more heavily on users fully specifying their information needs.

![](images/8cc6b8db680ab8987deefee647df3e7cedb30effc774634af9a2144171420f93.jpg)  
Figure 3: Clarification behavior under un- Figure 4: Scaled performance drop by underspecification type, derspecification across three benchmarks. pooled across all three benchmarks and models, comparing runs Agents either ask for clarification (Ask), state that state an assumption and proceed with runs that do not recognize assumptions (Assume), or proceed without the ambiguity. recognizing omissions (Unaware).

## 6 RQ2: Can DRAs Recognize Underspecification in User Queries?

We observed that DRAs sometimes abstain from answering underspecified queries, even though we did not indicate that the user query may be underspecified in system prompt. We therefore analyze these behaviors and identified three mutually exclusive categories: Unaware, in which the DRA proceeds to answer without explicitly recognizing the underspecification; Assume, in which it recognizes the underspecification but proceeds by making one or more explicit assumptions; and Ask, in which it abstains from answering and instead asks the user for clarification. See detailed analysis protocol in Appendix D. We then use GPT-5.4 mini to classify all the responses into these categories based on observable behavior in the DRA’s reasoning trajectories and final answers.

DRAs show limited awareness of missing information. Although answer quality declines for a large proportion of underspecified queries (Figure 2), DRAs generally cannot identify that key information is missing: Unaware is the most frequent category across all different DRAs (Figure 3; Table 4). Even when they acknowledge ambiguity, they usually answer under one or more stated assumptions rather than ask the user for clarification. Across agents, Tongyi-DR shows the least explicit awareness, whereas DR Tulu-8B and GPT-5.4 with tools more often state assumptions.

Stated assumptions do not guarantee better answers, and even worse. Although stating assumptions is DRAs’ predominant response strategy upon recognizing underspecification, Assume runs exhibit larger answer-quality losses than Unaware runs in three of the four conditions, as shown in Figure 4. One possible explanation is that unrecognized underspecification may be mild enough for DRAs to infer a plausible interpretation, whereas explicitly recognized ambiguity may reflect greater uncertainty about the user’s intent. In such cases, proceeding under assumptions may be less appropriate than seeking clarification.

## 7 RQ3: How Does Underspecification Affect Intermediate Search Behavior?

DRAs change search direction more often and investigate less deeply. This shift is most evident under the most underspecified variant, with more Exploration and fewer Specialization transitions across all three DRAs (Figure 5). Obscuring technical terminology also has a pronounced effect on search strategy. These changes indicate more frequent switches between different facets of a topic and fewer refinements that add constraints to narrow the search. Changes in the proportions of Generalization and Repetition are generally smaller and are not statistically significant in most comparisons.

Searches become less grounded in user queries and draw more on parametric knowledge. According to Figure 6, the share of searchquery tokens traceable to the user query decreases significantly, while the share of untraced tokens increases significantly in most cases. These untraced tokens occur in neither the user query, earlier search queries, nor retrieved documents, suggesting potential greater reliance on parametric knowledge to formulate searches. Scope removal produces a more consistent increase in the untraced token share than obscuring terminology. Removing preferences does not significantly change token provenance of search queries.

![](images/3cfef8ce39dc796fb0b4492366727da3fa62e33f7923c5231b461e84666c78d1.jpg)  
Figure 5: Changes in search-query transition types under underspecification. Values show mean paired differences (underspecified minus complete query, in percentage points) with 95% confidence intervals. Positive values indicate more frequent transitions. Stars denote significance $( ^ { * } p < . 0 5 , ^ { * * } p < . 0 1 , ^ { * * * } p < . 0 0 1 )$ .

![](images/7d7594e9c54a2bf3f7bb0c427c61f302c6880d75a12c54c086094fc2062ee53a.jpg)  
Figure 7: Changes in document novelty and contextdriven term adoption under underspecification. Values show mean paired differences with 95% confidence intervals. Positive novelty changes indicate more newly retrieved documents; negative adoption changes indicate less use of retrieved content. Significance markers follow Fig. 5.

Broader retrieval does not translate into greater utilization of retrieved content in subsequent searches. Figure 7 has shown that document novelty increases significantly under fully underspecified conditions for all three DRAs and without term, without scope conditions for DR Tulu and Tongyi. However, the context-driven term adoption rate does not increase correspondingly, and even dropped significantly for DR Tulu and Tongyi under the most underspecified conditions.

![](images/283fab99f0dd8abe6663d12acb29be1d3824bf992ef46d51eee50e392aac6947.jpg)  
Figure 6: Changes in search-query token provenance under underspecification. Values show mean paired differences in token shares (underspecified minus complete query, in percentage points) with 95% confidence intervals. Positive values indicate larger shares. Significance markers follow Fig. 5.

## 8 Discussion and Conclusion

Users should prioritize clearly defining the research question when formulating search queries. In particular, users should state key scope constraints and use precise technical terminology, as omitting or obscuring this information can substantially affect the search process and degrade the final answer quality. By comparison, information like user preferences has a smaller effect on answer quality.

DRAs need to be explicitly trained to identify missing information during long-horizon task execution. Our experiments revealed the usability challenge of current DRAs: users may need sufficient domain knowledge to identify relevant constraints and terminology before agents can effectively address their needs. For lay person who discovers missing specifications only after reviewing a lengthy report, they must spend additional time diagnosing the mismatch and reformulating their requests. Repeating the research process also incurs potentially avoidable tool calls and token consumption. We argue that DRAs should distinguish uncertainty that can be resolved through further search from missing information that must be elicited from the user. Training this capability through imitation learning requires training data that capture both deep research trajectories and appropriate clarification actions, but such data can be difficult to obtain. Our analysis of intermediate search behavior suggests that metrics such as the proportion of untraced search-query tokens, and context-driven term adoption rate may serve as useful indicators of missing information in user queries.

We introduce UNDERSPEC-DR, a framework for evaluating Deep Research Agents under query underspecification. Using this testbed, we examine changes in final answer quality and intermediate search behavior casued by query underspecification. We further analyze DRAs’ awareness of query underspecification, their response strategies upon recognizing it, and whether these strategies mitigate answer-quality losses. We find that evidence-grounded reports may still fail to meet users’ needs.These results provide practical advice on how users should prompt DRAs to get desired answer and offer implications for building DRAs that recognize when to ask users for additional input, balancing reliable task completion with minimal disruption.

## Limitations

Language. Our experiments primarily use English queries; DeepResearch Bench includes tasks in Chinese. Underspecification may take different forms in other languages and extending UNDERSPEC-DR to these settings is a natural direction for future work.

Coverage of agents. Our evaluation covers three agents whose intermediate search and tool-use trajectories are accessible for analysis. We do not evaluate closed-source deep research systems that do not expose these trajectories, as this would prevent us from analyzing their search behavior. These systems may employ different search and clarification strategies.

## Ethics Statement

Our study does not involve the generation of harmful content. Its main implication is a risk for users: when a query is underspecified, current DRAs may return a well-cited answer to a broader or different question than the one the user intended, usually without signaling that information was missing. Users who rely on such answers, for example in medical or policy research, may draw conclusions that do not apply to their situation. We hope that our findings encourage the development of agents that recognize missing information and make their assumptions visible to users.

## References

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’Arcy, David Wadden, Matt Latzke, Jenna Sparks, Jena D. Hwang,

Varsha Kishore, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, and 9 others. 2026. Synthesizing scientific literature with retrieval-augmented language models. Nature, 650(8103):857–863.

Nicholas J. Belkin, Robert N. Oddy, and Helen M. Brooks. 1982. ASK for information retrieval: Part I. background and theory. Journal ofDocumentation, 38(2):61–71.

Zongwan Cao, Bingbing Wen, and Lucy Lu Wang. 2026. Clarify or answer: Reinforcement learning for agentic VQA with context under-specification. In Proceedings ofthe Third Conference on Language Modeling.

Jacob Cohen. 1988. Statistical Power Analysisfor the Behavioral Sciences, 2nd edition. Lawrence Erlbaum Associates.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Licheng Zhang, Xiaorui Wang, and Zhendong Mao. 2026. DeepResearch Bench: A comprehensive benchmark for deep research agents. In International Conference on Learning Representations, volume 2026, pages 42414–42448.

Yucheng Du and Xiyang Hu. 2026. Recognition–refusal misalignment in LLMs: Why models answer structurally unanswerable questions. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing. Main Conference.

Beverly Freeman, Roma Ruparel, and Laura M. Vardoulakis. 2025. Zoom in, zoom out, reframe: Domain experts’ strategies for addressing non-experts complex questions. In Extended Abstracts ofthe CHI Conference on Human Factors in Computing Systems, CHI EA ’25. Association for Computing Machinery.

Lukas Haas and Shrestha Basu Mallick. 2025. Build with Gemini Deep Research. Google Blog.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Ö. Arık, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. In Proceedings ofthe Second Conference on Language Modeling.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In International Conference on Learning Representations.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025a. Search-o1: Agentic search-enhanced large reasoning models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 5420–5438. Association for Computational Linguistics.

Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yongkang Wu, Ji-Rong Wen, Yutao Zhu, and Zhicheng Dou. 2025b. WebThinker: Empowering large reasoning models with deep research capability. In Advances in Neural Information Processing Systems, volume 38, pages 120091–120131. Curran Associates, Inc.

Xun Liu, Yi Evie Zhang, Vira Kasprova, Parisa Rabbani, Pardis Sadat Zahraei, Tianyu Zhang, Ali Ebrahimpour-Boroojeny, and Varun Chandrasekaran. 2026. AgentAbstain: Do LLM agents know when not to act? Preprint, arXiv:2607.10059.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2024. GAIA: A benchmark for general AI assistants. In International Conference on Learning Representations, volume 2024, pages 9025–9049.

Jingjie Ning, João Coelho, Yibo Kong, Yunfan Long, Bruno Martins, João Magalhães, Jamie Callan, and Chenyan Xiong. 2026. Agentic search in the wild: Intents and trajectory dynamics from 14M+ real search requests. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1404–1415. Association for Computing Machinery.

OpenAI. 2025. Introducing deep research. OpenAI Blog.

Parker Seegmiller, Joseph Gatto, Sarah E. Greer, Ganza Belise Isingizwe, Rohan Ray, Timothy E. Burdick, and Sarah Masud Preum. 2026. How much would a clinician edit this draft? evaluating LLM alignment for patient message response drafting. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 32620–32646, San Diego, California, United States. Association for Computational Linguistics.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen-Tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, and 2 others. 2026. DR Tulu: Reinforcement learning with evolving rubrics for deep research. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 110046–110101. PMLR.

Jinyan Su and Claire Cardie. 2026. Knowing but not showing: LLMs recognize ambiguity but rarely ask clarifying questions. arXiv preprint arXiv:2605.25284. Preprint, arXiv:2605.25284.

Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, and Yiming Yang. 2026. Training proactive and personalized LLM agents. In Proceedings of the Third Conference on Language Modeling.

Mehrab Tanjim, Yeonjun In, Xiang Chen, Victor Bursztyn, Ryan A. Rossi, Sungchul Kim, Guang-Jie Ren, Vaishnavi Muppala, Shun Jiang, Yongsung Kim, and Chanyoung Park. 2025. Disambiguation in conversational question answering in the era of LLMs and agents: A survey. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 9537–9550, Suzhou, China. Association for Computational Linguistics.

Robert S. Taylor. 1968. Question-negotiation and information seeking in libraries. College & Research Libraries, 29(3):178–194.

David R. Thomas. 2006. A general inductive approach for analyzing qualitative evaluation data. American Journal ofEvaluation, 27(2):237–246.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, and 38 others. 2025. Tongyi DeepResearch technical report. arXiv preprint arXiv:2510.24701. Preprint, arXiv:2510.24701.

A. M. Turing. 1950. Computing machinery and intelligence. Mind, 59(236):433–460.

Sanidhya Vijayvargiya, Xuhui Zhou, Akhila Yerukola, Maarten Sap, and Graham Neubig. 2026. Ambig-SWE: Interactive agents to overcome underspecificity in software engineering. In International Conference on Learning Representations, volume 2026, pages 146716–146737.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. BrowseComp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516. Preprint, arXiv:2504.12516.

Roy Xie, Deepak Gopinath, David Qiu, Dong Lin, Haitian Sun, Saloni Potdar, and Bhuwan Dhingra. 2026. Over-searching in search-augmented large language models. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7714–7739, Rabat, Morocco. Association for Computational Linguistics.

Chenyang Yang, Yike Shi, Qianou Ma, Michael Xieyang Liu, Christian Kaestner, and Tongshuang Wu. 2026. What prompts don’t say: Understanding and managing underspecification in LLM prompts. In Findings of the Association for Computational Linguistics: ACL 2026, pages 9072–9101. Association for Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Analyze the query and answer these three   
questions:

2. "has\_term": Does the query contain any   
proper nouns, such as named systems,   
algorithms, frameworks, organizations, or   
specific named entities?

Li S. Yifei, Allen Chang, Chaitanya Malaviya, and Mark Yatskar. 2026. ResearchQA: Evaluating scholarly question answering at scale across 75 fields with survey-mined questions and rubrics. Transactions of the Association for Computational Linguistics, 14:1365–1389.

Crystina Zhang, Zigeng Xu, Xinyu Shi, Raphael Tang, Nandan Thakur, Xueguang Ma, Yao Lu, and Jimmy Lin. 2026. Hawkeye: Inspecting agentic search trajectories. Manuscript.

Michael J. Q. Zhang and Eunsol Choi. 2025. Clarify when necessary: Resolving ambiguity through interaction with LMs. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 5541–5558. Association for Computational Linguistics.

## A Prompt Templates

## A.1 Constructing Underspecified Query Variants

The rewriting procedure first checks whether a query contains preference information, terminology, or scope constraints, then generates the applicable underspecified variants. We reproduce the default prompts from the script below. The placeholder [QUERY] denotes the original query.

## Information detection: system prompt.

You analyze research queries.   
Return exactly one JSON object and nothing   
else.   
The JSON object must have these keys:   
- "has\_preference"   
- "has\_term"   
- "has\_scope\_limitation"   
For each key, the value must be exactly "yes"   
or "no".

## Information detection: user prompt.

1. "has\_preference": Does the query explicitly   
contain user preference information, such   
as desired format, depth, target   
audience, or evaluation criteria?

## Query rewriting: shared system prompt.

You need to rewrite user queries.   
You will be given:   
1. The original query   
2. One category of information to simplify   
Return only the rewritten query. Do not   
explain the changes.

## Preference removal: user prompt.

Remove any user preference information in the   
original query(e.g. format, depth,   
audience, criteria).   
Query:   
[QUERY]

## Terminology simplification: user prompt.

Simplify any proper nouns (e.g. named   
technology categories, algorithms,   
frameworks, organizations) to a brief   
generic mention.   
Query:   
[QUERY]

## Scope removal: user prompt.

Remove any explicit scope limitations (e.g.   
scenario, domain, time period, location).   
Query:   
[QUERY]

## Full underspecification: user prompt.

Remove all the important details in the user   
query, such as user preference   
information (e.g. format, depth,   
audience, criteria), proper nouns (e.g.   
named technology categories, algorithms,   
frameworks, organizations), and scope   
limitations (e.g. scenario, domain, time   
period, location). Replace named entities   
with generic descriptions where needed.   
Query:   
[QUERY]

## A.2 Underspecification Awareness and Response Classification

The following prompt classifies whether an agent recognizes query underspecification and how it responds, assigning one of three categories: Ask, Assume, or Unaware. The placeholders [QUERY] and [AGENT\_RESPONSE] represent the query received by the agent and its full response, respectively.

You are evaluating how a deep research AI   
agent handled a user's query.   
Below is the ORIGINAL USER QUERY the agent was   
given, followed by the AGENT'S FULL   
RESPONSE.

```erb
Your task: determine whether the agent
recognized that the user's query was
ambiguous, underspecified, or missing
information needed to answer it well, AND
how it reacted to that.
Classify the agent's behavior into exactly ONE
of these three categories:
"asked_clarifying_question": the agent
explicitly asked the user one or more
questions to clarify the request, before
or instead of giving a full answer.
"stated_assumption_and_proceeded": the agent
recognized the ambiguity, proposed one or
more explicit assumptions/interpretations
to resolve it, and then went ahead and
produced a full (typically long) answer
on that basis.
"no_recognition": the agent showed no sign
of recognizing any ambiguity in the query
and just answered as if the query were
fully specified.
Also provide:
"evidence": a short (at most 2 sentences)
quote or close paraphrase from the
response that supports your
classification, in the original language
of the response. If the category is
"no_recognition", instead give a
one-sentence explanation of why there is
no sign of recognition.
Respond with ONLY a single JSON object, no
other text, no markdown code fences:
{"category": "asked_clarifying_question" |
"stated_assumption_and_proceeded" |
"no_recognition", "evidence": "..."}
<original_user_query>
[QUERY]
</original_user_query>
<agent_response>
[AGENT_RESPONSE]
</agent_response>
```

## A.3 Search-Query Transition Classification

The following prompts classify the transition between two consecutive search queries. The placeholders {q1} and {q2} are replaced with the previous and current search queries, respectively.

System prompt.

You are an expert search behavior analyst.

User prompt.

Query 1 (Previous): {q1}   
Query 2 (Current): {q2}   
Analyze the search behavior evolution from   
Query 1 to Query 2 for an autonomous   
agent.   
Classify the transition into exactly ONE of   
these four categories:

```prolog
1. Specialization (Vertical Deepening): Query
2 is MORE specific than Query 1 by adding
constraints/details (q2 \subset q1).
(e.g., 'apple' -> 'green apple
nutritional value').
2. Generalization (Vertical Broadening): Query
2 is MORE general than Query 1 by
removing constraints/abstracting (q2
\supset q1). (e.g., 'green apple
nutritional value' -> 'benefits of
fruits').
3. Exploration (Horizontal Expansion within
the same domain/task): Query 2 is NOT
simply more specific or more general. It
shifts to a different aspect / subtopic /
related entity but still within the same
overall topic/domain. (e.g., 'green apple
nutritional value' -> 'green apple
recipes', or 'MRI Scans' -> 'CT Scans').
4. Repetition (Stationary): Query 2 is
semantically equivalent to Query 1. It is
a paraphrase, reformatting, or synonym
replacement with NO significant change in
intent (e.g., 'green apple value' ->
'nutritional value of green apple').
Output ONLY the category name (Specialization,
Generalization, Exploration, or
Repetition).
```

## B Deep Research Agent Configuration

We run all three deep research agents with the same budget of at most 15 tool calls per query. All inference is run on four NVIDIA H100 GPUs. The agent-specific configurations are as follows:

• DR Tulu: uses DR-Tulu-8B as the backbone model. Web search is performed through the Serper API and scholarly search through the Semantic Scholar API, and web pages are browsed through the Jina API.

• GPT-5.4 with tools: uses the same agent framework and tool set as DR Tulu-8B, with GPT-5.4 as the backbone model. Search, scholarly search, and browsing therefore use the Serper, Semantic Scholar, and Jina APIs, respectively.

• Tongyi-DR: uses Tongyi-DR-30B-A3B as the backbone model. Web search is performed through the Serper API, and web pages are browsed through the Jina API.

## C Real-World Query Collection Protocol

This section describes the protocol used to collect and analyze real user queries in Section 3.2.

Participants used the public web demo of DR Tulu released by AI2 (https://www.dr-tulu. org/chat). We asked them to ask agent questions arising from their own research. Then they were asked to carefully check whether each response matched their expectations and, if it did not, revise the query and ask again, repeating until they obtained a satisfactory response. Because the deep research agent does not support multi-turn interaction, each reformulation was submitted as a new, self-contained query rather than as a follow-up message. For every question that was not answered satisfactorily on the first attempt, participants shared the corresponding chat history via a link. All participants gave their consent to the use of their shared chat histories in this study.

## D Answer Analysis Protocol

We first manually inspected approximately 10% of the 1,922 agent responses to underspecified queries (about 192 responses). We sampled proportionally within each query-variant type, selecting approximately 10% of responses from each of the fully underspecified, without scope, without terminology, and without preferences conditions. This inspection revealed that some agents acknowledged ambiguity but continued answering under explicit assumptions. We therefore adapt the behavioral analysis approach of Su and Cardie (2026) to distinguish three mutually exclusive categories, based on each agent’s input query, recorded reasoning trajectory, and final response: Ask, Assume, Unaware.

We then prompted GPT-5.4 mini to annotate the same 192 responses using these categories. Cohen’s κ between the human and model annotations is 0.92, indicating almost perfect agreement. Following this validation step, we used the same model and prompt to annotate all responses. The classification prompt is provided in Appendix A.2, and each label is accompanied by a brief justification.

## E Examples of Underspecified Queries

We contrast four examples from ResearchQA in which removing scope information or obscuring technical terminology has different effects on the answers generated by DR Tulu-8B. Both the complete-query answer and the underspecifiedquery answer are evaluated against the original query’s rubrics by Gemini 3.1 Pro. Struckthrough text marks removed scope information in the without-scope variants $( q _ { \mathrm { s } } )$ . For the withoutterminology variants $( q _ { \mathrm { T } } )$ , struck-through text marks the original technical terms and bold text

gives their replacements.

## E.1 Lower Rubric Score After Removing Scope from the User Query

What are the main clinical advantages and safety considerations of using nebulized tranexamic acid to manage post-tonsillectomy bleeding?

Complete-query answer (score: 0.8333). The answer focuses on post-tonsillectomy hemorrhage. It discusses local bleeding control, reduced need for reoperation and blood products, pediatric care pathways, and escalation to surgery for persistent bleeding. Its safety discussion also addresses monitoring and precautions in this clinical setting.

Without-scope answer (score: 0.4167). After the surgical context is removed, the answer shifts toward hemoptysis and pulmonary hemorrhage, with additional discussion of epistaxis and sinonasal procedures. For example, it emphasizes “higher 30-minute cessation rates than IV TXA” in emergency hemoptysis. Although it still discusses bleeding control and safety, its focus moves away from the original post-tonsillectomy setting. This shift accompanies a decrease in rubric coverage from 0.8333 to 0.4167.

## E.2 No Decrease in Rubric Score After Removing Scope from the User Query

How does methotrexate work to modulate immune and inflammatory pathways in the treatment of moderate-to-severe atopic dermatitis?

Complete-query answer (score: 1.0000). The answer describes methotrexate’s effects on folate metabolism, immune-cell proliferation, inflammatory cytokines, and epidermal biology, explicitly discussing moderate-to-severe atopic dermatitis.

Without-scope answer (score: 1.0000). The answer continues to discuss these mechanisms and retains the omitted severity context. For example, its clinical-evidence discussion states that treatment is associated with “clinical improvement in moderate– severe AD.” It also discusses severe or refractory disease. Thus, removing the severity qualifier does not prevent the answer from covering the original query’s rubric requirements, and the score remains 1.0000. This example illustrates that omission of scope information does not necessarily cause a loss in rubric coverage when the answer still addresses the intended context.

## E.3 Lower Rubric Score After Obscuring Terminology in the User Query

How do different temperatures affect the formation and longevity of trapping nets in Duddingtonia flagrans a fungus?

Complete-query answer (score: 0.7500). The answer focuses on Duddingtonia flagrans, discussing temperature-dependent trap formation, signaling, structural persistence, and the use of spores in biological control. It organizes its discussion around the formation and longevity of trapping nets in the named species.

Without-terminology answer (score: 0.2500). Replacing the species name with “a fungus” leads the answer to consider multiple interpretations. It discusses nematode-trapping fungi such as Arthrobotrys and also introduces a discussion of structures in basidiomycete fruiting bodies. The answer explicitly frames its response as interpreting trapping nets in “two closely related, evidencerich systems.” Its focus thus broadens beyond the species specified in the original query, and rubric coverage falls from 0.7500 to 0.2500.

## E.4 No Decrease in Rubric Score After Obscuring Terminology in the User Query

How does chemical activation with sodium hydroxide a base improve the adsorption capacity of zeolite a mineral adsorbent for biogas upgrading?

Complete-query answer (score: 0.59375). The answer discusses sodium-hydroxide activation of zeolites, emphasizing surface chemistry, ion exchange, and the adsorption of hydrogen sulfide and carbon dioxide. It also covers activation conditions, regeneration, and structural trade-offs.

Without-terminology answer (score: 0.84375). Despite the generalized terms, the answer still discusses sodium hydroxide, zeolites, and clinoptilolite, while also considering other bases and mineral adsorbents. It addresses surface basicity, pore accessibility, gas selectivity, humidity, and operating trade-offs in biogas upgrading. In this pair, rubric coverage increases from 0.59375 to 0.84375. Thus, generalizing technical terms does not necessarily lower the score when the answer continues to address the original materials and process; this single example does not establish that generalization systematically improves answer quality.

## F Additional Results

## F.1 Answer Quality

Table 3 reports the mean rubric-based answer quality behind Figure 2.

## F.2 Clarification Behavior

Table 4 reports the share of Ask, Assume, and Unaware responses for each agent, benchmark, and variant behind Figure 3.

## F.3 Query Drift

Search-query drift increases consistently across agents under severe underspecification. Figure 8 shows that significant increases in first-to-final search-query drift occur across all three agents only under $q _ { \mathrm { U } } ,$ , the most underspecified variant. The increase is largest for DR Tulu-8B. By contrast, removing scope or obscuring terminology significantly increases drift only for DR Tulu-8B, while removing preferences has no significant effect for any agent. This metric captures semantic displacement between the first and last search queries, rather than distance from the user’s original information need.

## F.4 Document Utilization

Table 5 reports the mean document utilization behind the paired changes in Table 2. Not every query has every type of variant, since a variant exists only if the query contains the targeted information. Each comparison therefore uses only the questions that have the given variant and were answered under both the complete query and that variant, so the complete-query mean q differs slightly across variants.

## F.5 Citation Precision

Table 6 reports the mean citation precision behind the paired changes in Table 2. As for document utilization, not every query has every type of variant, since a variant exists only if the query contains the targeted information. Each comparison therefore uses only the questions that have the given variant and were answered under both the complete query and that variant, so the complete-query mean q differs slightly across variants.

<table><tr><td>Benchmark</td><td>Agent</td><td>q</td><td> $q _ { \mathrm { P } }$ </td><td> $q _ { \mathrm { s } }$ </td><td>qT</td><td>qu</td><td>Rel. change qu</td></tr><tr><td rowspan="3">ResearchQA</td><td>DR Tulu-8B</td><td>.791</td><td> $. 6 9 4 ^ { \mathrm { n s } }$ </td><td> $. 6 1 5 ^ { ^ { * * * } }$ </td><td> $. 5 6 5 ^ { * * * }$ </td><td> $. 2 9 4 ^ { * * * }$ </td><td>-62.8%</td></tr><tr><td>GPT-5.4 with tools</td><td>.610</td><td> $. 4 9 6 ^ { \ast }$ </td><td> $. 4 6 6 ^ { ^ { * * * } }$ </td><td> $. 4 4 9 ^ { ^ { * * * } }$ </td><td> $. 2 1 5 ^ { ^ { * * * } }$ </td><td>-64.8%</td></tr><tr><td>Tongyi-DR</td><td>.840</td><td> $. 7 6 3 ^ { * * }$ </td><td> $. 6 8 1 ^ { ^ { * * * } }$ </td><td> $. 6 2 4 ^ { * * }$ </td><td> $. 3 3 4 ^ { * * * }$ </td><td>-60.2%</td></tr><tr><td rowspan="3">ScholarQA-Bench</td><td>DR Tulu-8B</td><td>.477</td><td> $. 5 0 4 ^ { \mathrm { n s } }$ </td><td> $. 4 3 1 ^ { \mathrm { n s } }$ </td><td> $. 4 3 8 ^ { * * }$ </td><td> $. 3 0 8 ^ { ^ { * * * } }$ </td><td>-35.5%</td></tr><tr><td>GPT-5.4 with tools</td><td>.524</td><td> $. 4 9 5 ^ { \mathrm { n s } }$ </td><td> $. 4 8 9 ^ { \mathrm { n s } }$ </td><td> $. 4 9 6 ^ { \mathrm { n s } }$ </td><td> $. 3 7 7 ^ { * * * }$ </td><td>-25.8%</td></tr><tr><td>Tongyi-DR</td><td>.438</td><td> $. 4 1 4 ^ { \mathrm { n s } }$ </td><td> $. 3 5 7 ^ { * * }$ </td><td> $. 3 9 5 ^ { \mathrm { n s } }$ </td><td> $. 2 0 7 ^ { * * * }$ </td><td>-52.8%</td></tr><tr><td rowspan="3">DeepResearch Bench</td><td>DR Tulu-8B</td><td>.457</td><td> $. 4 4 1 ^ { ^ { * * * } }$ </td><td> $. 4 0 7 ^ { * * * }$ </td><td> $. 3 5 8 ^ { * * * }$ </td><td> $. 2 5 8 ^ { * * * }$ </td><td>-43.6%</td></tr><tr><td>GPT-5.4 with tools</td><td>.391</td><td> $. 3 8 7 ^ { \mathrm { n s } }$ </td><td> $. 3 5 9 ^ { * * }$ </td><td> $. 3 2 5 ^ { * * }$ </td><td> $. 2 1 4 ^ { * * * }$ </td><td>-45.4%</td></tr><tr><td>Tongyi-DR</td><td>.431</td><td> $. 4 2 3 ^ { * }$ </td><td> $. 3 9 0 ^ { * * * }$ </td><td> $. 3 5 0 ^ { * * }$ </td><td> $. 2 5 9 ^ { * * * }$ </td><td>-40.0%</td></tr></table>

Table 3: Average answer quality for different types of query variants. Significance markers indicate differences from the fully specified condition (q). The last column reports the relative change from q to the fully underspecified condition $( q _ { \mathrm { U } } ) .$ , calculated as $( q _ { \mathrm { U } } - q ) / q$
<table><tr><td rowspan="2">Agent</td><td rowspan="2">Variant</td><td colspan="3">ResearchQA</td><td colspan="3">ScholarQA-Bench</td><td colspan="3">DeepResearch Bench</td></tr><tr><td>Ask</td><td>Assume</td><td>Unaware</td><td>Ask</td><td>Assume</td><td>Unaware</td><td>Ask</td><td>Assume</td><td>Unaware</td></tr><tr><td rowspan="4">DR Tulu-8B</td><td>qp</td><td>0.0</td><td>11.1</td><td>88.9</td><td>0.0</td><td>18.4</td><td>81.6</td><td>0.0</td><td>37.9</td><td>62.1</td></tr><tr><td>qs</td><td>0.0</td><td>6.6</td><td>93.4</td><td>0.0</td><td>10.1</td><td>89.9</td><td>0.0</td><td>41.7</td><td>58.3</td></tr><tr><td>qT</td><td>0.4</td><td>15.5</td><td>84.0</td><td>0.0</td><td>15.1</td><td>84.9</td><td>0.0</td><td>53.3</td><td>46.7</td></tr><tr><td>qu</td><td>0.2</td><td>12.6</td><td>87.2</td><td>1.0</td><td>12.9</td><td>86.1</td><td>1.0</td><td>51.0</td><td>48.0</td></tr><tr><td rowspan="4">GPT-5.4 with tools</td><td>qp</td><td>0.0</td><td>8.7</td><td>91.3</td><td>2.1</td><td>12.8</td><td>85.1</td><td>1.9</td><td>36.5</td><td>61.5</td></tr><tr><td>qs</td><td>1.7</td><td>5.2</td><td>93.1</td><td>0.0</td><td>13.3</td><td>86.7</td><td>0.0</td><td>42.9</td><td>57.1</td></tr><tr><td>qT</td><td>9.5</td><td>10.1</td><td>80.4</td><td>0.0</td><td>18.1</td><td>81.9</td><td>9.5</td><td>42.9</td><td>47.6</td></tr><tr><td>qu</td><td>18.1</td><td>7.6</td><td>74.3</td><td>12.9</td><td>13.9</td><td>73.1</td><td>12.6</td><td>32.6</td><td>54.7</td></tr><tr><td rowspan="4">Tongyi-DR</td><td>qp</td><td>0.0</td><td>3.7</td><td>96.3</td><td>0.0</td><td>8.2</td><td>91.8</td><td>0.0</td><td>13.8</td><td>86.2</td></tr><tr><td>qs</td><td>0.0</td><td>2.6</td><td>97.4</td><td>0.0</td><td>3.0</td><td>97.0</td><td>0.0</td><td>15.3</td><td>84.7</td></tr><tr><td>qT</td><td>0.4</td><td>6.7</td><td>92.9</td><td>0.0</td><td>3.5</td><td>96.5</td><td>4.4</td><td>17.8</td><td>77.8</td></tr><tr><td>qu</td><td>1.2</td><td>6.0</td><td>92.8</td><td>0.5</td><td>15.5</td><td>84.0</td><td>2.0</td><td>18.0</td><td>80.0</td></tr></table>

Table 4: Clarification behavior (% of runs): asks for clarification (Ask), recognizes the ambiguity and states an assumption (Assume), or proceeds without recognizing the ambiguity (Unaware).

![](images/37e555867b819ccc15bfcf7358076f0c91d3cdc4979c99c77f626b49adb2cf52.jpg)  
Figure 8: Paired changes in query drift between the first and final search queries (underspecified minus complete query), with 95% confidence intervals. Positive values indicate greater drift under underspecification. One, two, and three stars denote $p < . 0 5 , p < . 0 1$ , and $p < . 0 0 1$ , respectively; open circles indicate nonsignificant differences.

<table><tr><td></td><td colspan="2">qp</td><td colspan="2"> $q _ { \mathrm { s } }$ </td><td colspan="2">qT</td><td colspan="2">qu</td></tr><tr><td>Agent</td><td>q</td><td>qp</td><td>q</td><td> $q _ { \mathrm { s } }$ </td><td>q</td><td>qT</td><td>q</td><td>qu</td></tr><tr><td>DR Tulu-8B</td><td>56.39</td><td> $5 6 . 4 0 ^ { \mathrm { n s } }$ </td><td>55.97</td><td> $5 7 . 4 8 ^ { * }$ </td><td>56.43</td><td> $5 6 . 5 0 ^ { \mathrm { n s } }$ </td><td>56.59</td><td> $5 9 . 4 4 ^ { ^ { \ast \ast \ast } }$ </td></tr><tr><td>GPT-5.4 with tools</td><td>21.66</td><td> $2 3 . 6 4 ^ { \mathrm { n s } }$ </td><td>20.82</td><td> $2 2 . 9 3 ^ { ^ { * * } }$ </td><td>21.29</td><td> $2 0 . 8 5 ^ { \mathrm { n s } }$ </td><td>21.38</td><td> $2 0 . 9 4 ^ { \mathrm { n s } }$ </td></tr><tr><td>Both agents</td><td>42.36</td><td> $4 3 . 1 6 ^ { \mathrm { n s } }$ </td><td>41.12</td><td> $4 2 . 8 8 ^ { * * * }$ </td><td>42.27</td><td> $4 2 . 1 3 ^ { \mathrm { n s } }$ </td><td>44.23</td><td> $4 5 . 9 2 ^ { ^ { \ast \ast } }$ </td></tr></table>

Table 5: Mean document utilization (%, $\# _ { \mathrm { c i t e d } } / \# _ { \mathrm { r e t r i e v e d } } )$ for each underspecified variant and its paired complete query q, pooled across ResearchQA, ScholarQA-Bench, and DeepResearch Bench. Both agents pools the paired questions of DR Tulu-8B and GPT-5.4 with tools. Significance markers come from paired Wilcoxon signed-rank tests against q.
<table><tr><td></td><td colspan="2"> $q _ { \mathrm { P } }$ </td><td colspan="2"> $q _ { \mathrm { s } }$ </td><td colspan="2"> $q _ { \mathrm { T } }$ </td><td colspan="2">qu</td></tr><tr><td>Agent</td><td>q</td><td>qp</td><td>q</td><td>qs</td><td>q</td><td> $q _ { \mathrm { T } }$ </td><td>q</td><td>qu</td></tr><tr><td>DR Tulu-8B</td><td>43.95</td><td> $4 5 . 3 5 ^ { \mathrm { n s } }$ </td><td>45.43</td><td> $4 2 . 5 4 ^ { \mathrm { n s } }$ </td><td>50.87</td><td> $5 3 . 0 4 ^ { \mathrm { n s } }$ </td><td>50.45</td><td> $5 0 . 9 0 ^ { \mathrm { n s } }$ </td></tr><tr><td>GPT-5.4 with tools</td><td>70.52</td><td> $7 5 . 2 0 ^ { \mathrm { n s } }$ </td><td>70.59</td><td> $7 0 . 1 1 ^ { \mathrm { n s } }$ </td><td>75.78</td><td> $7 6 . 6 1 ^ { \mathrm { n s } }$ </td><td>70.93</td><td> $6 8 . 7 9 ^ { \mathrm { n s } }$ </td></tr></table>

Table 6: Mean citation precision (%, $\# _ { \mathrm { s u p p o r t i n g } } / \# _ { \mathrm { c i t e d } } )$ for each underspecified variant and its paired complete query q, pooled across ScholarQA-Bench and DeepResearch Bench. Only questions with a final answer and at least one cited document under both conditions are included. Significance markers come from paired Wilcoxon signed-rank tests against q.