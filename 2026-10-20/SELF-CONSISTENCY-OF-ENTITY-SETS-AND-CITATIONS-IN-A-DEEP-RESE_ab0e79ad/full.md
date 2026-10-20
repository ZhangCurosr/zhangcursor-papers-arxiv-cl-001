# SELF-CONSISTENCY OF ENTITY SETS AND CITATIONS IN A DEEP RESEARCH SYSTEM UNDER REPEATED PROMPTING

A PREPRINT

Lauri Pykälä and Kimmo K. Kaski

Department of Computer Science Aalto University School of Science, P.O. Box 11000, 00076 Aalto, Finland Correspondence: lauri.pykala@aalto.fi

## ABSTRACT

Deep research agents (DRA) are commonly used for open-ended synthesis tasks where closed-world QA benchmarks are poorly suited to characterize relevant aspects of performance. We study the selfconsistency of one commercial deep research system under repeated identical prompting, representing each report with the set of named entities it mentions, categorised by a custom schema, and the set of citations it retrieves. Across repeated runs of a single open-ended cybersecurity related prompt, we quantify pairwise overlap to characterize how stable the retrieved and synthesized output structure is under repeated identical prompts. In ten runs, the pairwise entity Jaccard results in a median of 0.12 and the pairwise citation Jaccard is below 0.02, with discovery curves that do not visibly saturate. The magnitude and structure of (dis)agreement between entity sets varies by entity class, which suggests that self-consistency in a DRA is better understood as a function of the underlying entity space than as a stable property of the system.

## 1 Introduction and Related Work

Retrieval-Augmented Generation (RAG) is an umbrella term for methods designed to inject external knowledge into the context of a large language model at runtime, with the goal of providing more accurate, grounded, or updated information to improve the reliability and transparency of the answers it generates. As suggested by the acronym, it has three main components: Retrieval of data from external sources, Augmenting the current query context with the most relevant information, and Generating an answer that combines the Large Language Model’s (LLM’s) pre-trained knowledge with the retrieved information.

Singh et al. (2025) suggest a taxonomy of RAG ranging from a simple keyword-based Naïve RAG through to more advanced Agentic RAG systems, known as deep research agents (DRAs) that can autonomously examine hundreds of online sources per query and produce long-form reports with citations in response to arbitrary requests. Recent surveys characterise four core stages in the deep research pipeline, i.e. planning, question development, web exploration, and report generation, each of which admits substantial design variation across commercial implementations (Zhang et al., 2025).

Existing evaluations of deep research agents often emphasize claim-level factual accuracy, leaving aspects such as their handling of multi-source synthesis and the structural properties of the knowledge these systems produce less examined (Huang et al., 2025). Because the outputs are long-form and open-ended, being based on a web-scale retrieval step rather than a fixed corpus, their real-world usage is poorly characterized by the closed-world QA style benchmarks. More recent evaluations have developed approaches such as decomposing reports into atomic facts (Min et al., 2023) and using metrics such as recall and relevance in addition to binary correctness (Gupta et al., 2026).

Large language models are inherently stochastic, and due to their autoregressive nature, the effects of small perturbations in their outputs compound over the sequences generated. Zhai et al. (2026) model the stochasticity of answers, findings, and citations in DRA using Markov information acquisition decision processes. Most prior treatments of DRA stochasticity focus on variability in answers or final claims, but the structural question of how stable the set of retrieved and synthesized referents is across repeated runs has received comparatively little attention. Another line of work evaluates long-form generation by decomposing outputs into discrete units (atomic facts, claims, or named entities) and reasoning about coverage and overlap at that level rather than at the level of full text (Min et al., 2023).

![](images/c35666c2d2aa719f8c029cfa6813fb84b6b9955d854327d373a60bca675a768e.jpg)  
Figure 1: Analysis pipeline. We query Google Deep Research ten times with the same cybersecurity-related prompt and treat each generated report as one run. For each run, we extract entities and citation URLs using langextract, canonicalize entity strings by class using a three-way voting procedure with gabriel, and compute pairwise Jaccard similarities and discovery curves over the resulting canonical entity and citation sets.

Next, the stability of DRA under repeated prompting and decomposition of its output into discrete units is presented as the following research questions:

• How consistent is a deep research system’s entity coverage across repeated runs of the same prompt?

• How consistent is its set of cited sources across repeated runs?

• How consistent is the joint structure of which entities are grounded in which sources across repeated runs?

Our approach contributes to addressing a gap in long-form LLM evaluation by examining how stable a single system’s retrieved and synthesized outputs, operationalized through entity extraction, are under repeated identical prompting, rather than whether individual statements are verifiably factually correct. The contributions presented here are threefold. First, we propose a three-level representation of DRA outputs, i.e. canonicalized named entities, citations, and entity– citation pairs, as an evaluation lens for self-consistency under repeated identical prompting. Second, applying this lens to a single commercial DRA on an open-ended topical cybersecurity related prompt, we show that entity-level agreement is substantially higher than citation-level agreement. Third, we show that pairwise agreement varies sharply across entity classes in a way that tracks the presumed size of the underlying entity space.

## 2 Method

To study the self-consistency of a deep research system under fixed user inputs, we issue an identical open-ended cybersecurity prompt to a single commercial deep research system (Google deep-research-preview-04-2026, accessed through the API on 2026-04-27) in 10 repeated iterations of the same prompt, shown in Figure 1. Each of the ten runs produced a main report with links to citations. Inline citations in the raw response were emitted by the agent a a redirect URL. These URLs are followed by an HTTP GET request with a Chrome-like User-Agent and the resolved URL is recorded. These raw reports and cited sources are cleaned with a deterministic preprocessing pipeline before extraction. Resolved citation URLs were retained as citation identifiers, but the content of cited web pages was not included in the entity-overlap analysis.

Entities were extracted from all preprocessed documents using the langextract library <sup>1</sup> with Google Gemini 2.5-flash as the extraction model (Google, 2025). Raw entity surface forms were canonicalized using the openai-gabriel library, a general-purpose information extraction library that has been described and validated in Asirvatham et al. (2026). Deduplication was run separately for each of the eleven entity classes.

The method operates in two stages: first, embedding-based grouping clusters similar surface forms into groups of up to 500 that are then assigned labels in-context. The model used was gpt-5.4-mini. The process was repeated three times (n\_runs=3) with voting to reduce the effect of random noise. The system prompt used was the default one provided with the package.

To quantify the stochasticity of the DRA reports, we compute pairwise Jaccard similarity over all ${ \binom { n } { 2 } } = { \binom { 1 0 } { 2 } } = 4 5$ unordered pairs of runs. For two runs A and B with item sets $S _ { A }$ and $S _ { B }$

$$
J ( A , B ) = \frac { | S _ { A } \cap S _ { B } | } { | S _ { A } \cup S _ { B } | } ,
$$

with $J = 0$ when both sets are empty. The pairwise Jaccard similarity was calculated for the following:

• Overall entities. $S = \{ ( c , e ) \}$ , where c is an extraction class and e is a deduplicated canonical entity mentioned in the report, across all classes jointly.

• Per-class entities. For each extraction class $c , S _ { c } = \{ e \}$ restricted to that class. This is done to disentangle the degree of variance across different entity classes.

• Citations. $S = \{ u \}$ over distinct URLs.

• Entity–citation edges. $S = \{ ( c , e , u ) \}$ , where e is a canonical entity of class $c ,$ and u is the resolved citation URL assigned to that entity mention by the nearest following inline citation marker in the generated report. Edge overlap is therefore a strict URL-level proxy for whether the same entity is grounded in the same cited source across runs.

Pairwise Jaccard describes the degree of agreement between runs but obfuscates the directional trajectory of the separate runs. To investigate the latter, discovery curves are used. Given a fixed reading order $\pi = ( d _ { 1 } , \ldots , d _ { n } )$ of the n runs and a per-document item set $X _ { d } ,$ the discovery curve is as follows:

$$
C _ { \pi } ( k ) = \left| \bigcup _ { i = 1 } ^ { k } X _ { d _ { i } } \right| , \quad k = 1 , \ldots , n ,
$$

i.e. the cumulative count of distinct items (entities or citations) seen after reading the first k documents. To reduce dependence on an arbitrary run ordering, we compute this curve over 1000 random permutations of the ten runs and report the mean accumulation curve in Figure 3. Flattening curves indicate that repeated runs are converging on a stable set; curves that remain steep indicate that each additional run keeps retrieving unseen entities.

## 3 Results

Across the 10 runs, outputs contained a median of 5,292 words (range: 4,837–6,152), 199 extracted entity mentions (162–271), 162 distinct canonical entities after deduplication (139–193), 2.96 entities per 100 words (2.42–3.64), and 79 distinct citation URLs (66–101).

Figure 2 shows the distribution of pairwise Jaccard similarities across all 45 run pairs, broken down by entity class as well as aggregated for the overall entity set and for cited sources. Agreement between any two runs is low in absolute terms: the overall entity Jaccard has a median of 0.12, meaning that any two reports share only about one canonical entity in eight, despite being responses to an identical prompt. Overlap of citations is even lower, with the median Jaccard similarity score between sets of citations in two reports being just over 0.01, indicating that repeated runs of the same query rarely cite the same URLs. We additionally computed pairwise Jaccard similarity over entity–citation edges, where an edge is treated as shared only when both the canonical entity and the resolved citation URL most closely following it match across two runs. Edge-level agreement was near zero: across the 45 run pairs, the median edge Jaccard was 0.0073, with an interquartile range of 0.0019–0.0124 and a maximum of 0.0256. The median pair of runs shared only four entity–citation edges out of a union of 678. We therefore do not plot edge overlap separately in Figure 2, as it would be visually indistinguishable from the citation row. Substantively, the result indicates that repeated runs do not stably reproduce the same URL-level grounding structure.

Per-class Jaccard ranges over more than an order of magnitude, with Geopolitical actors and threat actors showing the highest median agreement, consistent with their being drawn from small, well-known closed vocabularies. Security products, software targets, and cybersecurity techniques cluster near zero, which is consistent with their being drawn from large open vocabularies with long tails. The overall entity row sits between these extremes and is dominated, in aggregate, by the long-tailed classes that contribute the most distinct items per run.

As an alternative way to visualize the diverging pattern of entity coverage, Figure 3 shows the discovery curves for the entity classes. Most curves appear approximately linear over the range we observe, with no visible inflection by the tenth run, with the exception of the categories Threat actor and Geopolitical actor.

![](images/3d5b503d5c30d392e90cdb3d0a5209c95d28365499bd6605fdf1d56e1db9f74c.jpg)  
Figure 2: Pairwise Jaccard similarity across ten repeated runs. Each point represents one unordered pair of reports, for 45 run pairs in total. Rows show Jaccard similarity over canonical entities within each class, resolved citation URLs, and the overall canonical entity set. Boxplots summarize the distribution across run pairs. Overall entity overlap is low, with median Jaccard approximately 0.12, while citation overlap is much lower, with median citation Jaccard just over 0.01. Entity–citation edge overlap is omitted from the plot because its median Jaccard is similarly near zero and visually indistinguishable from the citation row.

## 4 Concluding remarks

Our experiment shows that a single commercial deep research agent, queried ten times with an identical prompt, produces reports with low but structured overlap. The median overall entity Jaccard between runs is approximately 0.12, the median citation Jaccard is just over 0.01 and the median entity–citation edge overlap lower still, with a median Jaccard of 0.0073. The discovery curves do not visibly saturate within ten runs, indicating that additional runs continue to surface previously unseen entities.

The central finding is an asymmetry between referents and sources. Repeated runs are more likely to mention the same organizations, geopolitical actors, threat actors, or AI models than they are to cite the same URLs in support of those mentions. This suggests that the system has some stability in what it treats as topically relevant, but much less stability in the source trail through which those topics are retrieved and presented.

The research question related to entity–citation edges resolves negatively. Exact edge agreement requires two runs to reproduce both the same canonical entity and the same resolved citation URL in support of it. Because citation overlap is already extremely low, the resulting edge overlap is also near zero.

Finally, consistency is strongly class-dependent. Geopolitical actors and threat actors show the highest pairwise agreement, while security products, software targets, and cybersecurity techniques cluster near zero. This is consistent with the hypothesis that classes drawn from relatively bounded vocabularies are more stable than classes drawn from large, open-ended technical vocabularies. Self-consistency is therefore not a single number that characterizes a DRA in isolation, but the result of an interaction among the system, the prompt, the domain, and the entity class being measured.

## Limitations

Several limitations impact the claims of this work.

Single system, prompt and domain. We study one commercial DRA on one prompt in one domain. We make no claim that the magnitudes reported here generalize to other DRAs, other prompts, or other domains. Cybersecurity was chosen due to a hypothesis that it would produce a mix of relatively stable named entities and rapidly evolving long-tail entities, but other domains might show different patterns.

![](images/b8edb08829ca131d922f74a6ed04da24d70db278d8fbc2fac72e48f6dd3e1317.jpg)  
Figure 3: Discovery curves for canonical entity coverage across repeated runs. For each entity class, we randomly order the ten reports and compute the cumulative proportion of distinct canonical entities observed after k runs. Curves that flatten quickly would indicate convergence towards a stable observed entity set; the mostly near-linear curves indicate that additional runs continue to surface previously unseen entities.

Conflation of retrieval-side and generation-side stochasticity. Our experimental design does not separate variance arising from different web pages being retrieved across runs from variance arising from the same retrieved evidence being summarized differently due to limitations of the commercial API.

Web drift and URL stability. All ten runs were conducted within a short time window, but we cannot rule out that some inter-run variance reflects genuine changes in the indexed web rather than stochasticity of the agent. Because entity–citation edge equality depends on exact resolved URL matches, the edge results should be interpreted as URLlevel grounding stability rather than semantic evidence stability. Two different URLs may support similar claims, and the same underlying source may sometimes be reachable through different URL forms.

Extraction and canonicalization noise. Our pipeline relies on LLM-based entity extraction and deduplication stages. We did not conduct a full annotation-quality audit of either stage. Spot checks suggest minor inconsistencies in granularity, for example whether versioned variants of the same model are merged into a single canonical entity or retained separately. This was likely a result of underspecified prompts to the deduplication software library. Such errors likely affect the exact numeric Jaccard values, especially for entity classes with many near-duplicate technical names but we do not have evidence that they would explain the overall patterns.

## References

Hemanth Asirvatham, Elliott Mokski, and Andrei Shleifer. GPT as a Measurement Tool, February 2026. URL https://www.nber.org/papers/w34834.

Google. LangExtract, November 2025. URL https://github.com/google/langextract.

Nikita Gupta, Riju Chatterjee, Lukas Haas, Connie Tao, Andrew Wang, Chang Liu, Hidekazu Oiwa, Elena Gribovskaya, Jan Ackermann, John Blitzer, Sasha Goldshtein, and Dipanjan Das. DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents, January 2026. URL http://arxiv.org/abs/2601.20975. arXiv:2601.20975 [cs.CL].

Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Huichi Zhou, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, and Jun Wang. Deep Research Agents: A Systematic Examination And Roadmap, September 2025. URL http://arxiv.org/abs/2506.18096. arXiv:2506.18096 [cs.AI].

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation, October 2023. URL http://arxiv.org/abs/2305.14251. arXiv:2305.14251 [cs].

Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG, February 2025. URL http://arxiv.org/abs/2501.09136. arXiv:2501.09136 [cs].

Haotian Zhai, Elias Stengel-Eskin, Pratik Patil, and Liu Leqi. Evaluating Stochasticity in Deep Research Agents, February 2026. URL http://arxiv.org/abs/2602.23271. arXiv:2602.23271 [cs].

Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Huifeng Guo, Yong Liu, and Xiangyu Zhao. Deep Research: A Survey of Autonomous Research Agents, August 2025. URL http://arxiv.org/abs/2508. 12752. arXiv:2508.12752 [cs].