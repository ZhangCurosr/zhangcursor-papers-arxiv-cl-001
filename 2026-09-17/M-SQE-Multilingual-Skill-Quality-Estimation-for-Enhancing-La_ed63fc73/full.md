# M-SQE: Multilingual Skill Quality Estimation for Enhancing Language Equality in Agentic Skill Use

Yilun Liu, Shimin Tao, Minggui He, Chenxin Liu, Li Zhang, Chen Liu, Miao Zhang, Jiaxin Guo, Min Zhang, Liqun Deng, Xiaojun Meng, Daimeng Wei

Huawei, China

liuyilun3@huawei.com

## Abstract

Agent skills, reusable procedural documents that extend LLM agents beyond their parametric memory, have become an important interface for deploying agents on real-world tasks. Community-maintained skill libraries built around this interface are growing rapidly. However, this ecosystem remains deeply English-centric: our audit finds that low-resource languages such as Swahili and Hindi have no in-language skill content, so retrieval often returns a skill written in a diferent language than the query, degrading accuracy and recall. A practical solution is to synthesize in-language skills for retrieval but the quality can be unreliable, so relevance in this setting alone often surfaces a related but unusable candidate. To address this, we propose M-SQE, a post-retrieval Multilingual Skill Quality Estimation framework that scores candidates via a Theory view for intrinsic quality and an Action view for task-grounded utility, unified into a domain-conditioned final score. We evaluate M-SQE across three skill-use domains: general, tool-use, and cultural tasks. Empirically, we build three-layer candidate skill pools mirroring today’s ecosystem, where M-SQE’s task success exceeds existing baseline’s average by at least +3.5 points across three diferent retrievers. Particularly, M-SQE lifts the lowest-resource languages most (+12.9pp on Hindi and +5.6pp on Swahili) and achieves strong performance across all six culture regions, thereby moving agentic skill use toward linguistic and cultural equality.

## 1 Introduction

Agent skills are reusable procedural documents that package step-by-step instructions, function schemas, API conventions, and domain knowledge. They have become an important interface for extending large language model (LLM) agents beyond their parametric memory, letting them act reliably on multistep tasks (Wang et al. 2024; Xu and Yan 2026; Zhou et al. 2026). Community-maintained skill libraries built around this interface have grown quickly, but almost entirely in English. Table 1 reports our audit spanning over a dozen public community skill indexes (∼84,700 entries in total, including a popular 38.2k-star skill repository) (LittleDinoC 2026; wshobson 2026; huzey 2026): genuine target-language skill content is concentrated in a handful of high-resource languages, while Swahili and Hindi, spoken natively by hundreds of millions of people combined, have zero target-language skill content of their own (full audit protocol in Appendix A). In practice, an agent serving these languages has no in-language skill to

<table><tr><td>Language</td><td colspan="2">Est. in-language skill bodies</td></tr><tr><td>English (en)</td><td>~79,700</td><td rowspan="3">~4,000</td></tr><tr><td>Chinese (zh)</td><td></td></tr><tr><td>French (fr)</td><td>~700</td></tr><tr><td>Korean (ko)</td><td>~200</td></tr><tr><td>Japanese (ja)</td><td>~80</td></tr><tr><td>Swahili (sw)</td><td>0</td></tr><tr><td>Hindi (hi)</td><td>0</td></tr></table>

Table 1: Estimated genuine in-language skill bodies among ∼84,700 audited community skill entries: the ecosystem is mostly English, while Swahili and Hindi have zero.

![](images/139e75bdea01595b4ab67d4e7e80920ce61e71557614aaef90c7379d2e5e9f67.jpg)  
Figure 1: A representative failure for a Swahili speaker: retrieve-only selection surfaces a topically relevant smarthome skill that omits a slot required for execution. M-SQE’s Theory view screens out this gap and its Action view ranks the rest by fit, selecting a skill the agent can execute.

retrieve.

However, an English-only skill library is not merely an inconvenience: it actively degrades agent performance for non-English users. Lu et al. (2026) show that when a query and its supporting evidence are written in diferent languages, both answer accuracy and evidence recall drop sharply, possibly due to the distribution mismatch introduced by cross-lingual retrieval. Skills play exactly this evidentiary role for an agent. An agent that retrieves an English skill for a Swahili or Hindi query therefore inherits the same mismatch failure mode, owing to the absence of native-language skills. This is, in efect, an agent-era language equality problem: languages already underserved by the web are now underserved by the very tools meant to make agents useful to their speakers (Joshi et al. 2020; Lynch 2025). Fig. 1 illustrates the resulting failure mode and how our proposed M-SQE avoids it.

A practical way to close this gap is to synthesize more multilingual skills (Long et al. 2024; Ma et al. 2026; Wang et al. 2026), but doing so exposes two further challenges in an actual agentic skill-use scenario, where an agent has to choose the most suitable skill from a large pool. The first is quality: multilingual skills are typically synthesized by machine translation (MT) or model self-generation, and neither is stable in quality. Li et al. (2026) find that humancurated skills help agents on average while self-generated skills can hurt task performance due to unstable quality. Liu et al. (2026c) document that MT content can introduce content errors, wrong-language artifacts, and insuficient cultural localization. The second challenge is that relevance does not imply usability. Even in English-only settings, Li et al. (2025) report that agents adopt a relevance-retrieved skill 70.1% of the time with no accompanying performance gain. Realistic multilingual pools an agent might face, which mix ecological-style, MT, and self-generated material, widen this gap further: with lower multilingual quality, even more of these skills look relevant but provide no benefit.

To address this, we introduce M-SQE, a post-retrieval Multilingual Skill Quality Estimation framework that scores every retrieved candidate from two complementary views. A Theory view estimates a skill’s intrinsic, task-independent quality, and an Action view estimates its grounded utility for the query at hand. Because what makes a skill usable varies by task (e.g., tool use hinges on a strict execution contract, while cultural queries hinge on corroborated evidence), a router conditions how the two views combine into the final score, closing the gap between what retrieval finds and what an agent can actually execute. Our contributions are:

• We propose M-SQE, to our knowledge the first postretrieval quality estimation framework purpose-built for multilingual agent skills, filling the usability gap of retrieve-only approaches with an average +6.3pp tasksuccess gain across all nine evaluation settings.

• We demonstrate that M-SQE narrows the language inequality of agentic skill use with gains on lowest-resource languages (+12.9pp/+5.6pp task success on Hindi/Swahili).

• We open-source our evaluation set, skill pools, and M-SQE implementation, facilitating future research on multilingual and cross-cultural agent skill use<sup>1</sup>.

## 2 Social Impact of M-SQE

M-SQE targets a concrete instance of linguistic and cultural inequality in agentic AI: the skills that let language agents act reliably are overwhelmingly written in English, leaving speakers of low-resource languages and members of non-Western cultural communities with agents that are, in practice, less capable on their behalf. By aiding agentic skill retrieval in a noisy, realistic multilingual pool through two-view quality estimation, M-SQE advances agent-era equality in two ways:

(1) Bridging the Skill Divide for Underserved Language Communities. Non-English speakers already face a documented digital divide at the model layer (Lynch 2025); as Section 1’s audit finds, that divide recurs one layer further down the agent stack, where Swahili and Hindi have zero genuine in-language skill content. Speakers of these languages are therefore excluded from the agent-era productivity dividend not by the model they talk to, but by the skill library the agent draws on to act on their behalf. M-SQE addresses this gap efectively. On agentic tool-use tasks, M-SQE lifts Hindi and Swahili most (Fig. 3): task success rises by +12.9pp on Hindi and +5.6pp on Swahili. A skill pool that was only usable by chance becomes one an agent serving an underservedlanguage community can actually rely on.

(2) Advancing Cultural Equality in Agentic AI. Multilingual capability is not the same as cultural competence (Rystrøm, Kirk, and Hale 2025): an MT or self-generated skill can preserve fluent language while omitting or misattributing the cultural detail that determines whether an answer is correct. The Lunar New Year red-envelope case in Section 4 illustrates the failure: a topically matched skill can still specify the wrong customary amount. M-SQE’s cultural gains hold with stable deltas across six main culture regions, matching or exceeding both baselines in every region, with strong gains in South Asia (+6.6pp) and Oceania (+8.4pp). This region-by-region consistency, not a single number, is the evidence that M-SQE’s quality signal captures cultural correctness rather than surface fluency.

## 3 Related Work

Skill-Based Agents and Skill Retrieval. Reusable procedural knowledge was popularized for embodied LLM agents by Wang et al. (2024), whose skill library lets an agent accumulate and reuse verified action sequences during exploration. This idea has since grown into a community-scale ecosystem of packaged, on-demand agent skills (Xu and Yan 2026), and Li et al. (2026) show that curated skills raise agent task success while self-generated ones ofer little or even negative benefit, underscoring that not every skill in a growing pool is worth using. Li et al. (2025) focus on scaling skill retrieval itself, with a four-stage, skill-specific retrieval pipeline that ranks candidates for the query at hand. Generic rerankers sharpen the retrieved list along the same axis: ToolRerank (Zheng et al. 2024) adapts hierarchy-aware reranking to tool retrieval, while mMARCO (Bonifacio et al. 2021) extends relevance reranking across languages.

Multilingual Data Quality. A parallel line of work has documented that scaling multilingual instruction data by MT or model self-generation, the two pipelines most multilingual synthesis eforts rely on, degrades quality rather than merely diluting fluency. Lai, Mesgar, and Fraser (2024) report substantial per-language MT error rates, and Liu et al. (2026c)

![](images/95e5ec640bd37d2031759e5fa3b82f52653e0932c148cf6095784a21afd2c7f3.jpg)  
Figure 2: Overview of M-SQE: retrieved skill candidates are scored by Theory and Action views, fused by the domainrouted unified score, and delivered to the solver for tasks.

and Zhao et al. (2026) build on this observation with expertrevision and quality-scoring pipelines that clean multilingual instruction corpora before instruction tuning; quality scorers such as DEITA (Liu et al. 2024) likewise select instruction data on intrinsic quality alone. These pipelines judge only the data’s own language quality, leaving out skill-use dimensions such as executability and context eficiency.

Positioning M-SQE. The two lines above leave complementary gaps. The skill-retrieval line ranks candidates by how well they match the query, and relevance is where its judgment stops: it asks neither whether a matched skill is intrinsically sound, nor whether the skill will actually work for the task at hand. The data-quality line, in turn, stops at the data’s own language quality. M-SQE fills both gaps: the Theory view brings intrinsic quality estimation to skill-use time, and the Action view adds what neither line measures — how much a specific skill will help the query at hand, together with an explicit misleading-risk estimate.

## 4 The M-SQE Framework

Overview. Fig. 2 gives an overview of M-SQE. Given a query, a retriever first returns a candidate set of skills. M-SQE then scores every candidate from two complementary views: a Theory view that certifies intrinsic skill quality and an Action view that estimates task-grounded utility. The two views are routed into a single score based on the task domain at hand, and the resulting ranking determines which skills, within a fixed Top-N budget, are handed to the downstream solver (the agent that executes the user’s task).

## 4.1 Problem Formulation

Given a task query q, a skill pool S, a retriever R, and a candidate depth K, the retriever returns a candidate set $C _ { K } = R ( q , S , K )$ . M-SQE then selects a fixed-size subset $Z _ { N } \subset C _ { K }$ , with budget $N \leq K$ , and the downstream solver receives only the query and the anonymized skill bodies in $Z _ { N }$ (provenance labels and source identifiers stripped), without any gold answer or checker metadata. The objective is to maximize deterministic task success under the same skill-body budget. Retrieval decides which candidates are visible; M-SQE decides which visible candidates are usable.

## 4.2 Theory View

The Theory view certifies intrinsic skill quality independent of any particular query, scoring six dimensions:

(1) Correctness (red-line): the skill’s procedural facts and logic are right; (2) Completeness: the necessary steps and boundary cases of the task class are covered; (3) Executability: steps are concrete enough for an agent to act on, not merely de scriptive; (4) Cross-lingual faithfulness: no meaning-altering mistranslation or untranslated fragments; (5) Localization: the prose is genuinely written for the target language; (6) Context eficiency: concise enough not to dilute agent’s attention.

These dimensions are inspired by existing multilingual instruction-data quality taxonomy (Liu et al. 2026c), and are further extended to the demands of multilingual agent skills. A language-service expert team averaging over six years of professional experience constructs the dimensions across multiple rounds of review, organizing them into two blocks: general quality dimensions — correctness, completeness, cross-lingual faithfulness, and localization — and agentspecific dimensions — executability and context eficiency in Theory view (as well as the six dimensions in Action view)— that instruction data does not require but an executing agent does. The overall Theory score is the six-dimension mean on a 0–100 scale, capped at 40 if the red-line dimension is violated and at 80 if any basic dimension is, so a single severe defect cannot be averaged away. Full expert profiles, the expert construction process behind both Thoery and Action views’ dimensions, per-dimension cases, and the full scoring rubric are in Appendix D.

## 4.3 Action View

The Action view estimates task-grounded utility for the current query and a candidate skill, scoring five grounded-fit dimensions plus an explicit risk estimate:

(1) Task applicability: whether the skill targets this task at all; (2) Procedure match: the steps cover what the query actually requires; (3) Constraint match: slots, parameters, and constraints line up with the request; (4) Output-format match: the skill produces the output shape the task expects; (5) Language fit: the prose is usable in the query’s language; (6) Misleading risk: the chance that a related-looking skill steers the solver to a wrong API, value, or convention.

These dimensions targets the adoption gap documented by Li et al. (2025), where skills look relevant enough to be adopted yet fail to help. Each dimension is scored 0–100. Consequently, the rubric permits a high Action score only when the procedure is executable, the constraints match, and the misleading risk is low, avoiding trapped by surface relevance.

## 4.4 Unified Scoring

Based on the expert-examined dimensions above, we prompt a scorer LLM per view with its full rubric, obtaining the Theory and Action scores used throughout this section. M-SQE evaluates every candidate skill s under the same Top-N budget with a unified score that routes three domain-specific scoring rules, $C _ { \mathrm { g e n } } , C _ { \mathrm { f u n c } }$ , and $C _ { \mathrm { c u l } }$ , through task domain τ :

$$
S _ { \mathrm { M - S Q E } } ( q , s ; \tau ) = a _ { \tau } C _ { \mathrm { g e n } } ( q , s ) + b _ { \tau } C _ { \mathrm { f u n c } } ( q , s ) + c _ { \tau } C _ { \mathrm { c u l } } ( q , s ) ,\tag{1}
$$

where $( a _ { \tau } , b _ { \tau } , c _ { \tau } ) \in \{ 0 , 1 \} ^ { 3 }$ with $a _ { \tau } + b _ { \tau } + c _ { \tau } = 1$ . The indicator tuple is fixed directly by the task domain τ: the general domain sets $( 1 , 0 , 0 )$ , the tool-use domain sets $( 0 , 1 , 0 )$ and the cultural domain sets $( 0 , 0 , 1 )$ . Each of the three components combines Theory and Action (and, for cultural tasks, Retrieval) in a way matched to the nature of the task.

Domain Router. Multilingual Agentic skill use can be roughly divided into three classes: tool-heavy tasks and culture-heavy tasks, each has its own focus, and a general class covering the rest. Based on this taxonomy, the task domain τ is predicted from the query by a 5-shot prompted LLM domain router (five examples per domain written from the public task definitions alone), reaching 98.5% accuracy on our three-domain query set. A noise analysis shows the router’s labels can be randomly flipped on up to 90% ofqueries while M-SQE stays ahead of Retrieve-only throughout (Fig. 5). A new specialized domain can be added the same way: a new adapter paired with five added examples.

General-task convex fusion. $\begin{array} { l l l } { { C _ { \mathrm { g e n } } } } & { { = } } & { { 0 . 6 \mathrm { A c t i o n } \ + } } \end{array}$ 0.4 Theory. General skill-use tasks cover heterogeneous reusable procedures (spreadsheet formulas, media-processing recipes, code patterns) whose usefulness is graded rather than binary: a skill can be comprehensive yet miss the one operator a query needs, or read as directly relevant while a small execution detail is wrong. The single mixing coeficient gives task applicability a modest majority while retaining intrinsic quality as a regularizer. For example, when a query asks for an XLOOKUP formula, a comprehensive spreadsheet overview that never states lookup semantics should not outrank a short, correct skill; conversely, a relevant-looking snippet with a malformed argument order is discounted by its Theory score.

Tool-use quality guarding. Tool use carries a discrete execution contract: the selected skill must expose the correct function schema, required slots, and admissible value formats, so usefulness here is gated, not graded. $C _ { \mathrm { f u n c } }$ ranks candidates by the lexicographic order (Theory $\geq 6 5$ , Action), equivalently $C _ { \mathrm { f u n c } } = \mathbf { \dot { \delta } } M \cdot \mathbf { 1 } [ \mathrm { T h e o r y } \geq \dot { 6 } 5 ] + .$ Action for any M larger than the Action range as a constant ofset lifting every guard-passing candidate above all others. The threshold of 65 marks the boundary below which a skill is structurally unusable on the 0–100 Theory scale — for example, missing a required slot or carrying a critical localization failure — and candidates whose Theory scoring flags a language red-line are guarded out the same way. Consider a smart-home skill retrieved for “turn of the bathroom $l i g h t ^ { * } \dot { }$ it may read as topically on point yet omit the device-location slot, leaving it unusable regardless of how relevant it looks. The guard removes such a candidate from contention first; Action then ranks the remaining executable skills by how well their slots match the request.

Cultural evidence fusion. $\begin{array} { r l r } { C _ { \mathrm { c u l } } } & { { } = } & { z ( \mathrm { R e t r i e v a l } ) \ + } \end{array}$ $z ( \mathrm { T h e o r y } ) + z ( \mathrm { A c t i o n } )$ , where z(·) denotes per-query z-score standardization within the retriever’s candidate set. Culture questions fail in three independent ways — the retrieved skill can name the wrong event, state an unreliable fact, or fail to resolve the question actually asked — so no single view is suficient and none should dominate the others’ scale. A query about Lunar New Year red envelopes illustrates two of the three: a skill about wedding-gift etiquette can be factually sound yet grounded in the wrong occasion, while a self-generated Lunar New Year skill can name the right occasion and still misstate the customary amount. With equal standardized fusing, Retrieval anchors event grounding, Theory checks factual reliability, and Action checks whether the skill answers the question at hand.

Each component’s own constants — the General fusion’s 0.6/0.4 weight, the Tool-Use guard’s Theory≥65 threshold, and the Cultural fusion’s equal standardized weights — follow from the design rationale above and stay stable under parameter perturbation (Appendix C).

## 5 Experiments

## 5.1 Construction of Evaluation Tasks

Our evaluation spans three high-frequency scenarios of agentic skill use: general procedural knowledge work, tool invocation, and culturally grounded interaction, all common surfaces for multilingual agents. The statistics of evaluation tasks as well as the paired skill pool are shown in Table 2.

All three domains are built by one construction process. Source tasks are drawn from real, published benchmarks and culture resources (detailed below). To ensure every task tests skill use rather than the solver’s parametric memory, we follow the checker-verified construction of SkillsBench (Li et al. 2026) and pass each source task through a three-stage filter: (1) verifiability, keeping only tasks whose success a deterministic checker can decide; (2) skill necessity, screening out tasks that saturate without any skill, which leaves the Prompt-only anchor well below ceiling in all three domains (General 53.2%, Tool-Use 20.4%, and Cultural 46.2%; Table 3); and (3) leakage control, rewriting task prompts into natural user queries that copy no text from any pool skill and share no content-bearing terms with the answer key (Appendix E). Each domain then stratifies the admitted tasks across its languages, regions, and originating benchmarks. All task query rewriting and composition were carried out by the same multilingual expert team introduced in Section 4, through the same multiple rounds of review.

General Skill Use. General Skill Use covers 94 curated multilingual skill-use tasks across six languages (fr, hi, ja, ko, sw, zh), built in the SkillsBench paradigm (Li et al. 2026). Task types span spreadsheet formulas, code snippets, and document workflows, so the needed skill is reusable how-to procedural documentation, not a single fixed API call.

Tool Use. Tool Use draws on real utterances from Kulkarni et al. (2025)’s 52-language function-calling benchmark, restricted to seven languages $( e n , f r , h i , j a , k o , s w , z h )$ spanning a 55-function inventory across smart-home and IoT control, calendars and alarms, media playback, and everyday information queries. In tool-use tasks the agent must invoke device and service functions with exact schemas and slot values; the skills used in this evaluation are the documents that teach those invocations. Success on each utterance is graded by a deterministic function/slot checker, reflecting whether the agent executed the correct call.

<table><tr><td>Domain</td><td>#Tasks</td><td>Query language/culture</td><td>Three-layer skill pool</td><td>Pool size</td><td>Candidate depth</td></tr><tr><td>General Skill Use</td><td>94</td><td>fr, hi, ja, ko, sw, zh</td><td>ecological-style 400 + MT 550 + self-generated 800</td><td>1,750</td><td>10</td></tr><tr><td>Tool Use</td><td>265</td><td>en, fr, hi, ja, ko, sw, zh</td><td>ecological-style  $1 , 2 5 4 + \mathrm { M T } 6 6 0 +$  self-generated 385</td><td>2,299</td><td>20</td></tr><tr><td>Cultural Skill Use</td><td>52</td><td>Six main cultural regions</td><td>ecological-style  $\mathbf { 3 } , 3 8 4 + \mathbf { M T } \mathbf { 1 } , 2 8 3 +$  self-generated 618</td><td>5,285</td><td>50</td></tr></table>

Table 2: Evaluation setup by domain. Each test task draws skills from a source-included three-layer skill pool (ecological-style, MT, and model self-generated); candidate depth is the retriever output size available to selectors.

Cultural Skill Use. Cultural Skill Use comprises 52 shortanswer tasks constructed from published culture resources, including NormAd (Rao et al. 2025), CultureBank (Shi et al. 2024), CulturALL (Lin et al. 2026), CultureScope (Zhang et al. 2025), CultureAtlas (Fung et al. 2024), SAGE (Guo et al. 2025). The tasks span six cultural regions worldwide (East and Southeast Asia, South Asia, Europe, Africa and the Middle East, the Americas, and Oceania). Task types span etiquette, gift-giving, and dining customs alongside grounded regional facts, so the needed skill is culture-point documentation, not a generic overview. Each task is an independent, natural short-answer prompt derived from a culture point.

## 5.2 Skill Pool Construction

We build each domain’s paired skill pool the way a multilingual pool is assembled in practice. The audit in Section 1 shows that genuine in-language skill content is scarce outside a handful of high-resource languages, so multilingual coverage typically comes from layering the two scalable production paths, MT and model self-generation, on top of the available ecological material. So our pools in Table 2 are three-fold:

(1) The ecological-style layer approximates the naturally available material an agent would encounter today; it contains document-derived skills, rendered from existing community skill files, oficial product skills, and API documentation, and background-derived skills distilled from domain background documentation (Xu and Yan 2026), preserving the coverage gaps and stylistic variation of real skill authorship. (2) MT is the first scalable path once target-language originals run out, and it might come with defects such as translationese (Lai, Mesgar, and Fraser 2024; Liu et al. 2026c). Our MT layer preserves these naturally occurring cross-lingual transfer artifacts; concretely, MT skills are translated from each domain’s English source material into its non-English target languages, so a domain’s MT volume follows its language roster and available source material. (3) Self-generation is the second scalable path, and its limits are similarly documented — unstable downstream gain due to occasional hallucinations (Li et al. 2026; Zhang et al. 2026). Our self-generation layer preserves the naturally occurring incompleteness and execution errors this production path is known to produce; concretely, self-generated skills are written by an LLM prompted with the domain’s public task specifications (e.g., task descriptions, function schemas, or culture-point summaries), revealing no answer and checker.

The exact composition difers by domain (Table 2), reflecting the materials naturally available in each. The pool is also source-included: a skill relevant to a given query may already sit inside it, but every candidate is judged on its own content alone. Source inclusion is the standard convention in retrieval evaluation. The relevant document stays inside the searchable corpus rather than being held out, and skill-retrieval work follows the same convention: Li et al. (2025) inject their own oracle skills into the retrieval index in evaluation. Detailed skill construction and leakage control are in Appendix E.

## 5.3 Retrievers and Baselines

Because a quality-estimation layer must work regardless of which retriever supplies candidates, we evaluate M-SQE across three retrievers: BM25, the classic lexical retriever (Robertson and Zaragoza 2009); Neural, a hybrid dense retriever (Luan et al. 2021); and SkillFlow, the state-of-the-art skill-specific retrieval pipeline (Li et al. 2025), faithfully reimplemented from its four-stage design. Together the three retrievers span lexical, neural, and skill-specific retrieval. Retriever candidate depth follows $K = \mathrm { \ i } 0 \cdot \lfloor \vert S \vert / 1 0 0 0 \rfloor$ where |S| is the domain’s pool size. This rule trades of two failure modes: too shallow confounds selection quality with retrieval recall, since the relevant skill may never surface, while too deep inflates scoring cost and floods the candidate set with skills no ranking could rescue. All post-retrieval baselines (selectors) score this same fixed candidate set of depth K and output a final selection of Top N skills.

For baselines compared with M-SQE, we reports two selection anchors, Random (a uniform selection from the candidate set) and Retrieve-only (the retriever’s own ranking), together with six strong external baselines split into two families that both address important aspects of skill use. Relevance rerankers rank candidates by query-skill match: ToolRerank (Zheng et al. 2024), an adaptive, hierarchy-aware algorithmic reranker for tool retrieval; SkillFlow (Li et al. 2025)’s own ranking stage, reused here as a post-retrieval selector regardless of which retriever supplied its candidates; and mMARCO (Bonifacio et al. 2021), a multilingual crossencoder reranker. Quality scorers assess intrinsic data quality of skills: multilingual data-quality scorers DEITA (Liu et al. 2024), M-DaQ (Zhao et al. 2026), and JQL (Ali et al. 2025).

![](images/5c4285d228ebd785ccac9e17f0393079af762d6866af85985f66941920e2a039.jpg)  
Figure 3: Success rate across culture regions and languages: M-SQE matches or exceeds both baselines in every breakdown, with the largest gains on Hindi and Swahili, the two lowest-resource languages in our evaluation.

<table><tr><td></td><td colspan="3">General</td><td colspan="3">Tool-Use</td><td colspan="3">Cultural</td><td rowspan="2">Avg</td></tr><tr><td>Method</td><td></td><td>BM Ne</td><td>SF</td><td>BM Ne</td><td></td><td>SF</td><td>BM Ne</td><td></td><td>SF</td></tr><tr><td colspan="10">Selection anchors</td></tr><tr><td>Prompt-only</td><td></td><td>53.2 53.2 53.2 20.4 20.4 20.4 46.2 46.2 46.2 39.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td></td><td></td><td></td><td></td><td></td><td>66.2 69.1 70.6 26.1 26.6 33.1 60.0 65.4 79.2 55.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Retrieve-only 71.3 70.2 74.5 28.7 25.7 37.4 78.8 88.5 90.4 62.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Relevance rerankers</td></tr><tr><td>ToolRerank</td><td>64.9 73.4 70.2 30.6 26.0 34.7 76.9 86.5 92.3 61.7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SkillFlow</td><td>71.3 74.5 74.5 33.2 36.2 37.4 86.586.5 90.4 65.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>mMARCO</td><td>67.071.3</td><td></td><td>70.2 29.4 27.5 32.5</td><td></td><td></td><td></td><td></td><td></td><td>86.5 90.4 92.3 63.0</td><td></td></tr><tr><td colspan="10">Quality scorers</td></tr><tr><td>DEITA</td><td>68.1 61.7 69.1 30.2 29.1 35.1 51.9 51.9 75.0 52.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M-DaQ</td><td>66.0 59.6 61.7 30.6 31.7 34.3 63.5 57.7 76.9 53.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JQL</td><td>59.6 69.1 70.2 25.3 20.4 33.2 53.8 57.7 75.0 51.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">M-SQE (ours)</td></tr><tr><td>M-SQE</td><td></td><td>74.5 79.8 76.6 39.6 36.2 40.4 90.4 90.4 94.2 69.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Task success rate (%) at Top3, comparing M-SQE with existing skill-selection methods across three domains and three retrievers (BM=BM25, Ne=Neural, SF=SkillFlow). Bold = best in each column, underline = second-best.

## 5.4 Metrics and Backbones

All metrics are deterministic checkers: task-specific checkers for General Skill Use and Cultural Skill Use, and an exact function/slot execution match for Tool Use. A response counts as correct only when it matches the checker’s target exactly — a gold string or accepted paraphrase for General Skill Use and Cultural Skill Use, or the correct function name with matching argument slots for Tool Use. Checker acceptance criteria for each domain are detailed in Appendix G; all other implementation details are in Appendix F.

Two tiers of models produce and score the main results in this paper. Tier 1 (skill generation) uses Qwen3.5-9B for both the self-generated skill layer and MT translation, representative of the lightweight, practical LLMs multilingual data-synthesis pipelines typically build on today (Liu et al. 2026c; Zhao et al. 2026) (e.g., MIDB adopts an 8B LLM as its main data synthesizer). Tier 2 uses Gemini-3-Flash as the scorer, router and solver, chosen for its balance of multilingual quality and speed, a trade-of widely adopted in recent multilingual agent research (Liu et al. 2026a,b). See a backbone robustness analysis in Fig. 6.

## 5.5 Experimental Results

Main Results. Table 3 shows M-SQE first or tied-first against the single strongest baseline in all 9 domain-byretriever settings, reaching 69.1% average success, +3.5pp over the strongest baseline, SkillFlow (65.6%); significance analyses are in Appendix B. M-SQE is also strictly ahead of both selection anchors in all 9 settings (+6.3pp over Retrieveonly and +14.0pp over Random on average), confirming the premise of Section 1: in realistic multilingual pools, what retrieval surfaces is often not what an agent can use, and scoring both views recovers the diference.

Robustness across Regions and Languages. Fig. 3 shows M-SQE matches or beats both baselines in all six culture regions and 7 languages, with the largest gains on the bench mark’s lowest-resource languages: Tool Use success rises +12.9pp on Hindi and +5.6pp on Swahili. These are the two languages whose in-language skill supply Table 1 measures at zero, so the improvement of M-SQE relieves exactly where the ecosystem leaves speakers with the least.

Ablation Study. Table 4 isolates the Theory view and the Action view: M-SQE beats both single views on all three domains, though the two single-view gaps are visibly uneven. The unevenness follows from how the dimensions divide: among the quality dimensions we design, the skill-specific ones concentrate in the Action view (e.g., procedure match and constraint match), so a candidate’s fit to the task weighs heavily on downstream success. The Theory view complements this as an intrinsic-quality guard, protecting the agent from the occasional catastrophically flawed skill that nonetheless reads

<table><tr><td>Method</td><td>General</td><td>Tool-Use</td><td>Cultural</td></tr><tr><td>Theory-only</td><td>67.0</td><td>28.7</td><td>59.6</td></tr><tr><td>Action-only</td><td>71.3</td><td>38.5</td><td>88.5</td></tr><tr><td>M-SQE</td><td>74.5</td><td>39.6</td><td>90.4</td></tr></table>

Table 4: Mechanism ablation (BM25 retriever, Top3): task success rate (%) for single view alone and full M-SQE.

Three-domain average success rate (%), BM25@Top3; the main backbone setting

![](images/73382c851610cff798cd268f9f2940620021601f13e29ea0aa239d34bfd98732.jpg)  
Figure 4: Three-domain average task success vs. skill budget N under BM25: M-SQE beats Retrieve-only at every budget, with the margin widening as the budget tightens.

![](images/50d3129cdf8d25b6423a5d1503846ad736da521a20bc753baee0222a7e9e027c.jpg)  
Figure 5: Router corruption sweep (BM25, Top-3, threedomain average): M-SQE stays above both baselines at every corruption rate up to 90% (1,000 seeds per rate; shaded band = empirical 95% interval); the star marks the 5-shot router used in the main experiments with 1.5% error rate.

well-fitting (as revealed by the empty-guidance Case 2 in Appendix I). Together, the two views cover complementary failure modes of multilingual agentic skills.

Skill Budget Sensitivity. Fig. 4 shows M-SQE beats Retrieve-only at every skill budget N ∈ {1, 3, 5, 10}, with the margin widening from +3.4pp at Top10 to +9.7pp at Top1. As the budget tightens, a bad skill does more damage to the solver (e.g., N = 1 leaves no chance for a single bad pick), exactly where careful selection pays of. Small budgets are also the regime real agent deployments occupy: carrying ten skills can consume ten times the context of carrying one.

Routing Robustness. The domain router supplies the only input M-SQE requires beyond the query and its candidate skills; randomly flipping its predicted domains at rates from 10% to 90% (Fig. 5), M-SQE stays above Retrieve-only throughout, from +8.1pp at 10% corruption to +4.5pp at 90%. Even with the routing signal nearly destroyed, quality estimation alone keeps M-SQE ahead: routing sharpens the margin, and the two views hold the floor.

LLM Backbone Robustness. To test whether M-SQE depends on a particular pipeline backbone, we vary the poolsynthesis, scorer, and downstream-solver LLMs. Across all six combinations, M-SQE remains +3.5–9.0pp above Retrieveonly (Fig. 6). Changing either the scorer or solver preserves this ordering, and rebuilding the MT and self-generated pool layers with a stronger Qwen3.6-Plus retains positive gains for both solvers. Together, these controls demonstrate that M-SQE’s two-view quality estimation captures transferable skill utility across the pipeline, while ruling out pool-synthesis artifacts and scorer or solver bias as explanations for the gains.

![](images/937f8f0949dd5bc1437b0b9b8f0fcded3d2f53f5e905b8479365b58f8f62d583.jpg)

Figure 6: M-SQE remains ahead of Retrieve-only across all six skill-pool synthesis, scorer, and downstream-solver backbone configurations.
<table><tr><td>Method</td><td>General</td><td>Tool-Use</td><td>Cultural</td></tr><tr><td>Random</td><td>36.17</td><td>12.08</td><td>42.31</td></tr><tr><td>Retrieve-only</td><td>36.17</td><td>12.83</td><td>50.00</td></tr><tr><td>Base (untrained)</td><td>35.11</td><td>10.19</td><td>38.46</td></tr><tr><td>M-SQE</td><td>38.30</td><td>13.21</td><td>50.00</td></tr></table>

Table 5: Downstream trajectory generalization: task success rate (%) of a Qwen3.5-9B model fine-tuned on trajectories built from each method’s selected skills.

Downstream Trajectory Generalization. Despite strong performance of M-SQE in aiding skill use, we further veirify its usefulness as training material for a downstream agent, and test whether skill quality still matters for a small open-source model. Fine-tuning Qwen3.5-9B as an agentic solver with trajectories built from each method’s selected skills, M-SQE yields the best fine-tuned model averagely in three domains (Table 5). The signal that picks better skills to read also picks better skills to learn from, extending M-SQE from an inference-time filter to a beneficial training-data constructor.

## 6 Conclusion

In this paper, we introduced M-SQE, a post-retrieval quality estimation layer that judges a retrieved skill on its own merits and on its fit to the query, then routes the two judgments by the task’s domain, turning the multilingual skill pool into one an agent can trust across domain and languages. As agentic AI reaches every language community, we hope M-SQE marks a step from agents that merely sound fluent toward agents that are equally capable in any language or culture. Future work includes extending to more languages and cultural regions, scaling to larger skill ecosystems, and broadening from skills to more agentic assets. Limitations and responsible use are discussed in Appendix H.

## References

Ali, M.; Brack, M.; Lübbering, M.; Wendt, E.; Khan, A. G.; Rutmann, R.; Jude, A.; Kraus, M.; Weber, A. A.; Stollenwerk, F.; Kaczér, D.; Mai, F.; Flek, L.; Sifa, R.; Flores-Herr, N.; Koehler, J.; Schramowski, P.; Fromm, M.; and Kersting, K. 2025. Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, 8859–8898.

Bonifacio, L.; Jeronymo, V.; Abonizio, H. Q.; Campiotti, I.; Fadaee, M.; Lotufo, R.; and Nogueira, R. 2021. mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset. arXiv preprint arXiv:2108.13897.

Fung, Y. R.; Zhao, R.; Doo, J.; Sun, C.; and Ji, H. 2024. No Culture Left Behind: Massively Multi-Cultural Knowledge Acquisition & LM Benchmarking on 1000+ Sub-Country Regions and 2000+ Ethnolinguistic Groups. arXiv preprint arXiv:2402.09369.

Guo, S.; Jiang, S.; He, Q.; Xiao, Y.; Liang, J.; Bi, Y.; He, M.; Tao, S.; and Zhang, L. 2025. Do Large Language Models Truly Understand Cross-Cultural Diferences? arXiv preprint arXiv:2512.07075.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

huzey. 2026. claude-skills. Hugging Face Datasets. Accessed 2026-07-24.

Joshi, P.; Santy, S.; Budhiraja, A.; Bali, K.; and Choudhury, M. 2020. The State and Fate of Linguistic Diversity and Inclusion in the NLP World. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, 6282–6293.

Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6769–6781.

Kulkarni, M.; Mazzia, V.; Gaspers, J.; Hench, C.; and FitzGerald, J. 2025. MASSIVE-Agents: A Benchmark for Multilingual Function-Calling in 52 Languages. In Findings of the Association for Computational Linguistics: EMNLP 2025, 20193–20215. Association for Computational Linguistics.

Lai, W.; Mesgar, M.; and Fraser, A. 2024. LLMs Beyond English: Scaling the Multilingual Capability of LLMs with Cross-Lingual Feedback. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, 8186–8213. Bangkok, Thailand: Association for Computational Linguistics.

Li, F.; Tagkopoulos, P.; and Tagkopoulos, I. 2025. SkillFlow: Scalable and Eficient Agent Skill Retrieval System. arXiv preprint arXiv:2504.06188.

Li, X.; Liu, Y.; Chen, W.; You, B.; Di, Z.; He, Y.; Zheng, S.; Choe, K. W.; Sun, J.; Wang, S.; et al. 2026. SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv preprint arXiv:2602.12670.

Lin, P.; Lyu, C.; Luo, W.; Ye, H.; Hossain, M. M.; Ma, C.; Ji, S.; Samih, Y.; Zeng, B.; Jiang, F.; Cao, Y.; Duisenbek, D.;

Xun, A. N. S.; Pozdniakova, D.; Misevich, L.; Marinković, N.; Nguyen, N. G. L.; Do, T. K. L.; Sophy, S.; Hu, B.; Chen, G.; Tang, G.; Aji, A. F.; Wang, L.; and Luo, W. 2026. CulturALL: Benchmarking Multilingual and Multicultural Competence of LLMs on Grounded Tasks. arXiv preprint arXiv:2604.19262. LittleDinoC. 2026. agent-skills. Hugging Face Datasets. Accessed 2026-07-24.

Liu, W.; Zeng, W.; He, K.; Jiang, Y.; and He, J. 2024. What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR).

Liu, Y.; Zhang, M.; Tao, S.; He, M.; Zhao, C.; Liu, C.; Zhang, L.; Liu, C.; Qian, C.; Deng, L.; Meng, X.; and Wei, D. 2026a. MADE: Beyond Scoring via a Multilingual Agentic Diagnosing Engine for Fine-Grained Evaluation Insights. arXiv preprint arXiv:2606.07020.

Liu, Y.; Zhao, C.; Piao, M.; Miao, L.; Tao, S.; He, M.; Liu, C.; Zhang, L.; Ma, H.; Guo, J.; Liu, C.; Deng, L.; Wei, J.; Meng, X.; Du, F.; Wei, D.; and Xiao, Y. 2026b. The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 21364–21384. San Diego, California, United States: Association for Computational Linguistics.

Liu, Y.; Zhao, C.; Yang, X.; Zeng, H.; Tao, S.; Meng, W.; He, M.; Yu, Y.; Ma, H.; Zhang, L.; Wei, D.; and Chen, B. 2026c. MIDB: Multilingual Instruction Data Booster for Enhancing Cultural Equality in Multilingual Instruction Synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 38952–38961.

Long, L.; Wang, R.; Xiao, R.; Zhao, J.; Ding, X.; Chen, G.; and Wang, H. 2024. On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey. In Findings of the Associationfor Computational Linguistics: ACL 2024, 11065– 11082. Bangkok, Thailand: Association for Computational Linguistics.

Lu, Y.; Zeng, Q.; Qi, H.; Yu, P.; Zhao, F.; Yang, R.; Yanaka, H.; Yokoya, N.; and Xuan, W. 2026. Beyond Monolingual Deep Research: Evaluating Agents and Retrievers with Cross-Lingual BrowseComp-Plus. arXiv preprint arXiv:2606.15345.

Luan, Y.; Eisenstein, J.; Toutanova, K.; and Collins, M. 2021. Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics, 9: 329–345.

Lynch, S. 2025. How AI is leaving non-English speakers behind. Stanford News.

Ma, Y.; Huang, Y.; Bao, H.; Zhuang, H.; Shukla, S.; Galley, M.; Zhang, X.; and Feuerriegel, S. 2026. SkillGen: Verified Inference-Time Agent Skill Synthesis. arXiv preprint arXiv:2605.10999.

Rao, A.; Yerukola, A.; Shah, V.; Reinecke, K.; and Sap, M. 2025. NormAd: A Framework for Measuring the Cultural Adaptability of Large Language Models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter

of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2373–2403. Albuquerque, New Mexico: Association for Computational Linguistics.

Robertson, S.; and Zaragoza, H. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4): 333–389.

Rystrøm, J. H.; Kirk, H. R.; and Hale, S. 2025. Multilingua != Multicultural: Evaluating Gaps Between Multilingual Capabilities and Cultural Alignment in LLMs. In Proceedings ofthe Interdisciplinary Workshop on Observations ofMisunderstood, Misguided and Malicious Use ofLanguage Models, 74–85. Varna, Bulgaria: INCOMA Ltd., Shoumen, Bulgaria.

Shi, W.; Li, R.; Zhang, Y.; Ziems, C.; Yu, S.; Horesh, R.; Paula, R. A. D.; and Yang, D. 2024. CultureBank: An Online Community-Driven Knowledge Base Towards Culturally Aware Language Technologies. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, 4996–5025.

Wang, C.; Yu, Z.; Xie, X.; Yao, W.; Fang, R.; Qiao, S.; Cao, K.; Zheng, G.; Qi, X.; Zhang, P.; and Deng, S. 2026. SkillX: Automatically Constructing Skill Knowledge Bases for Agents. arXiv preprint arXiv:2604.04804.

Wang, G.; Xie, Y.; Jiang, Y.; Mandlekar, A.; Xiao, C.; Zhu, Y.; Fan, L.; and Anandkumar, A. 2024. Voyager: An Open-Ended Embodied Agent with Large Language Models. Transactions on Machine Learning Research.

wshobson. 2026. agents: A Multi-Harness Agentic Plugin Marketplace. GitHub repository. Accessed 2026-07-24.

Xu, R.; and Yan, Y. 2026. Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward. arXiv preprint arXiv:2602.12430.

Zhang, H.; Fan, S.; Zou, H. P.; Chen, Y.; Wang, Z.; Zhou, J.; Li, C.; Huang, W.-C.; Yao, Y.; Zheng, K.; Liu, X.; Li, X.; and Yu, P. S. 2026. CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification. arXiv preprint arXiv:2604.01687.

Zhang, J.; Jiang, S.; Guo, S.; Chen, S.; Xiao, Y.; Feng, H.; Liang, J.; He, M.; Tao, S.; and Ma, H. 2025. CultureScope: A Dimensional Lens for Probing Cultural Understanding in LLMs. arXiv preprint arXiv:2509.16188.

Zhao, C.; Liu, Y.; Zeng, P.; Luo, Y.; Tao, S.; He, M.; Meng, W.; Xu, S.; Liu, C.; Ma, H.; Zhang, L.; Chen, B.; and Wei, D. 2026. M-DaQ: Retrieving Samples with Multilingual Diversity and Quality for Instruction Fine-Tuning Datasets. In Proceedings ofthe 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, 4361–4366.

Zheng, Y.; Li, P.; Liu, W.; Liu, Y.; Luan, J.; and Wang, B. 2024. ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 16263–16273. ELRA and ICCL.

Zhou, Y.; Wang, S.; Su, Y.; Du, W.; Fang, Y.; and Lin, X. 2026. A Comprehensive Survey on Agent Skills: Taxonomy, Techniques, and Applications. arXiv preprint arXiv:2605.07358.

## A Community Skill Ecosystem Audit

Section 1 reports that non-English agent skills are scarce and, for several widely spoken languages, efectively absent from the public ecosystem (Table 1). This section describes the audit protocol behind that claim and reports the full per-language methodological detail.

Audit scope. We audited over a dozen public community skill indexes and repositories on code- and dataset-sharing platforms such as Hugging Face and GitHub, contributing ∼84,700 entries in total (un-deduplicated across sources), the figure Table 1’s counts are computed against. The largest sources are LittleDinoC’s agent-skills collection on Hugging Face Datasets (∼61,000 entries) (LittleDinoC 2026), wshobson’s agents multi-harness agentic plugin marketplace on GitHub (175 skills, a 38.2k-star repository) (wshobson 2026), and huzey’s claude-skills collection on Hugging Face Datasets (22,862 curated SKILL.md files crawled from skills.sh across 522 source repositories) (huzey 2026); the remaining repositories were surfaced by these indexes and our own targeted per-language searches. All were active, community-maintained collections at the time of the audit (2026-07) – the two Hugging Face collections each aggregate skills crawled from a large number of independent GitHub repositories, and the GitHub marketplace is itself a widely-starred skill repository in its own right – so together they approximate the pool an agent’s skill retriever would plausibly draw from in practice. The goal was to estimate, for each of the paper’s evaluated languages, how many entries in this ecosystem carry a skill body that is genuinely authored in the target language, as opposed to an English skill with translated metadata, an MT stub, or a title that merely contains a target-language keyword. The English count in Table 1 is the audited total minus all identified in-language content; entries whose bodies are MT stubs wrapped around English instructions count toward the English side.

Method. We used Unicode-script-based in-language detection: the presence of kana characters flags Japanese, hangul flags Korean, and CJK ideographs without kana or hangul flag Chinese. Script matches below a 25-character threshold were discarded, since short matches are dominated by false positives — English-language skill bodies whose title, tags, or embedded example strings happen to contain a target-language trigger word. Every repository that survived this script filter was then fetched directly and its skill body checked, programmatically and by hand, to confirm genuine in-language prose content rather than a templated header, a boilerplate translation notice, or a partial MT wrapped around otherwise-English instructions.

Table 1 reports the resulting counts. The script screen and length threshold apply to every audited source; the precision of each count follows its flagged volume. Chinese surfaces by far the most flagged entries, so its genuine in-language count is estimated from expert spot-checks of the flagged set; because skill authorship clusters by author (a few prolific contributors account for a disproportionate share of any one language), we read this estimate at order-of-magnitude precision. French, Korean, and Japanese each surface only a few hundred flagged candidates, so every flagged repository was checked directly and their counts are rounded tallies. The zero counts for Swahili and Hindi come from direct enumeration of every candidate repository the script filter and targeted keyword search surfaced, each verified individually.

<table><tr><td>Slice</td><td>∆ vs Retrieve (pp)</td><td>95% CI</td><td>task-level p</td></tr><tr><td>Domain-level</td><td></td><td></td><td></td></tr><tr><td>General Skill Use</td><td>+4.3</td><td>[+1.0, +7.6]</td><td>0.0158*</td></tr><tr><td>Tool Use</td><td>+6.0</td><td>[+4.1, +8.1]</td><td>&lt;0.0001***</td></tr><tr><td>Cultural Skill Use</td><td>+5.0</td><td>[+1.1, +9.3]</td><td>0.0213*</td></tr><tr><td>Retriever-level</td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>+6.6</td><td>[+4.2, +9.0]</td><td>&lt;0.0001***</td></tr><tr><td>Neural</td><td>+7.7</td><td>[+5.3, +10.3]</td><td>&lt;0.0001***</td></tr><tr><td>SkillFlow</td><td>+2.1</td><td>[+0.6, +3.7]</td><td>0.0079**</td></tr><tr><td>Budget-level</td><td></td><td></td><td></td></tr><tr><td>Top1</td><td>+7.7</td><td>[+5.2, +10.3]</td><td>&lt;0.0001***</td></tr><tr><td>Top3</td><td>+7.1</td><td>[+4.7, +9.6]</td><td>&lt;0.0001***</td></tr><tr><td>Top5</td><td>+4.1</td><td>[+2.1, +6.2]</td><td>0.0002***</td></tr><tr><td>Top10</td><td>+2.9</td><td>[+1.3, +4.6]</td><td>0.0009***</td></tr></table>

Table 6: Paired M-SQE vs. Retrieve-only tests. Domain and retriever rows summarize all four skill budgets; budget rows hold TopN fixed. To avoid treating repeated evaluations of the same task as independent, each row first averages repeated outcomes within task, then reports a paired sign-flip p-value and task-level bootstrap 95% CI from 10,000 resamples; \* $\scriptstyle { p < 0 . 0 5 }$ , \*\* p<0.01, \*\*\* p<0.001.

Even the best-represented non-English language, Chinese, accounts for only a small fraction of the ∼84,700-entry combined index, and the count falls of sharply for French, Korean, and Japanese. For Swahili and Hindi — languages spoken natively by hundreds of millions of people combined — we found no in-language skill content at all. This scarcity is the reason a realistic multilingual skill pool cannot be built from ecological material alone. Section E describes how each evaluation domain’s skill pool accounts for this by combining ecological-style skills with MT and self-generated material, mirroring the de facto composition of the ecosystem this audit documents.

## B Statistical Significance

Table 6 compares M-SQE with Retrieve-only by skill-use domain, retriever, and downstream skill budget. To avoid treating repeated evaluations of the same task as independent, we first average outcomes within each task and then compute paired bootstrap intervals and sign-flip p-values over tasks.

Domain-level. M-SQE improves over Retrieve-only in General Skill Use (+4.3pp), Tool Use (+6.0pp), and Cultural Skill Use (+5.0pp). All three task-level tests are significant $( p = 0 . 0 1 5 8 , p < 0 . 0 0 0 1$ , and $p = 0 . 0 2 1 3$ , respectively), with bootstrap intervals that exclude zero.

Retriever-level. The gain holds under BM25 (+6.6pp), Neural (+7.7pp), and SkillFlow (+2.1pp). The task-level tests remain significant for all three retrievers $( p < 0 . 0 0 0 1$ $p < 0 . 0 0 0 1$ , and $p = 0 . 0 0 7 9 )$ , showing that the efect is not tied to one candidate-generation mechanism.

Budget-level. M-SQE improves over Retrieve-only by +7.7pp at Top1, +7.1pp at Top3, +4.1pp at Top5, and +2.9pp at Top10. All four task-level tests are significant $( p < 0 . 0 0 0 1$ $p < 0 . 0 0 0 1 , p = 0 . 0 0 0 2$ , and $p = 0 . 0 0 0 9 )$ , confirming that the gain persists across the full budget range while narrowing as more retrieved skills are passed to the solver.

## C Hyperparameter Stability Analysis

This section asks whether M-SQE’s combination constants (Section 4) are knife-edged, that is, whether a small nudge to any one of them would already change which candidates the paper’s selector hands to the downstream solver. Two of the three fusion rules carry exactly one tunable scalar, and both were fixed by the design rationale given in Section 4 rather than by fitting any evaluation outcome: the General fusion’s mixing weight $\lambda ~ = ~ 0 . 6$ gives Action a modest majority over Theory $\mathsf { \bar { ( } C _ { g e n } = 0 . 6 A c t i o n + 0 . 4 T h e o r y ) }$ , and the Tool-Use guard’s threshold $\tau = 6 5$ marks the point on the 0–100 Theory scale below which a candidate is judged structurally unusable regardless of how relevant it looks. The Cultural fusion has no comparable free scalar to begin with: $C _ { \mathsf { c u l } } = z ( \mathrm { R e t r i e v a l } ) + z ( \mathrm { T h e o r y } ) + z ( \mathrm { A c t i o n } )$ sums the three per-query standardized views with equal (1:1:1) weight by construction, not a tuned coeficient. We nonetheless subject this equal weighting to the same stability treatment below, asking whether nudging any one view’s weight away from parity would already change the selection. What follows verifies, empirically and per task, that none of these three commitments sits on a knife edge.

Concretely, for every task we define its stability width as the length of the interval of parameter values over which that task’s selected Top-3 set stays completely unchanged compared with Table 3. A stability width under 0.02 for λ, under 2 points for τ, or under 0.04 for a Cultural fusion weight (on the reference scale defined below) means the deployed value sits close enough to a boundary that a small nudge — respectively ±0.01, ±1 point, or ±0.02 — already flips which three skills are selected for that task; this is the operational meaning of “knife-edge” throughout this section.

General Skill Use. Across the 94 tasks (BM25 retriever), the per-task stability width for λ has a median of 0.635 and a mean of 0.656 over the full [0, 1] range λ can take; one task (1/94, 1.1%) is stable for every $\bar { \lambda } \in [ 0 , 1 ] .$ , its Theory/Action candidate ranking never crossing regardless of the weight. No task (0/94, 0.0%) has a stability width under 0.02, and no task sits at an exact tie with a non-selected candidate at $\lambda = 0 . 6$ itself (0/94). The selected Top-3 is identical across a ±0.05 sweep of the mixing weight (0.55–0.65) in 83/94 (88.3%) of tasks, and across a ±0.10 sweep (0.50–0.70) in 79/94 (84.0%).

Tool Use. Across the 265 tasks (BM25 retriever), the pertask stability width for τ has a median of 95 and a mean of 96 points over the full [0, 100] Theory scale; 80/265 (30.2%) of tasks are stable for every $\dot { \tau } \in [ 0 , \dot { 1 0 0 } ]$ . No task (0/265, 0.0%) has a stability width under 2 points. The Theory≥65 guard’s selection is identical across a ±5-point sweep of the guard threshold (60–70) in 265/265 (100.0%) of tasks, and across a ±10-point sweep (55–75) in 265/265 (100.0%).

Cultural Skill Use. Since $C _ { \mathrm { c u l } }$ carries no single scalar to sweep, we instead perturb each of its three equal weights one at a time, holding the other two fixed at 1, across the 52 tasks (BM25 retriever); a reference domain of [0, 2] (0 dropping a view entirely, 2 double-counting it) stands in for the [0, 1] and [0, 100] ranges λ and τ naturally live in. No task has a stability width under 0.04 for any of the three views — zero knife-edge tasks for Retrieval, Theory, or Action individually. A ±0.05 nudge to a single view’s weight leaves the Top-3 unchanged in 94.2% (49/52) of tasks for Retrieval, 92.3% (48/52) for Theory, and 96.2% (50/52) for Action; at ±0.10 these fall to 82.7% (43/52), 86.5% (45/52), and 92.3% (48/52); at ±0.25 to 57.7% (30/52), 69.2% (36/52), and 71.2% (37/52). Requiring simultaneous stability under all three single-weight nudges — the strictest reading, in which a task counts only if none of the three views flips its selection — still leaves 92.3% (48/52) of tasks unchanged at ±0.05, 80.8% (42/52) at ±0.10, and 48.1% (25/52) at ±0.25. The equal standardized weighting therefore sits inside as broad a stability plateau as the two tuned constants above, despite carrying no free scalar to tune in the first place.

Across all three domains, the deployed constants — two fixed by design rationale, one fixed by construction — lie within broad selection-stability plateaus for the large majority of tasks rather than on a knife edge.

## D Dimension Construction and Scoring Prompt Templates

Expert team. Both views’ dimensions were built by a team of professional language-service experts averaging over six years of experience in translation, localization, and multilingual content editing, drawn from an international languageservice organization and covering the paper’s evaluation languages (French, Hindi, Japanese, Korean, Swahili, and Chinese) alongside English. Experts were allocated to languages by native proficiency, and every dimension decision was cross-checked by a second expert before being finalized, mirroring the task-allocation and review discipline MIDB (Liu et al. 2026c) used to build its own instruction-data quality taxonomy.

Construction process. The dimensions were derived empirically, following the same audit-driven process MIDB (Liu et al. 2026c) used to build its own quality criteria. Starting from the ∼84,700-entry community skill audit (Section A), the expert team sampled per-language sub-pools of skills for manual review. These development subsets were strictly disjoint from both the skill pools and evaluation tasks used in all reported experiments, preventing leakage. A first round of blind scoring rated each sampled skill against a provisional checklist adapted from MIDB’s instruction-data criteria. The team then compared the lowest- and highest-scored items and traced the gap to failure modes the provisional checklist missed: skills whose steps looked complete on paper but were not concrete enough for an agent to execute, skills padded with irrelevant detail that diluted an agent’s attention, and relevant-looking skills that nonetheless steered a solver toward a wrong API, value, or convention. These gaps were distilled into the Theory view’s executability and contexteficiency dimensions and the Action view’s dimensions such as misleading-risk. Two further audit rounds refined dimension wording and cut-line examples until inter-expert agreement on a shared calibration subset stabilized, at which point both rubrics were frozen for use throughout this paper. Table 7 and Table 8 give a detailed explanation and one concrete skill-side judgment case for each Theory and Action dimension.

<table><tr><td>Dimension</td><td>Explanation</td><td>Skill-side judgment case</td></tr><tr><td>Correctness line]</td><td>The skill&#x27;s procedural facts and logic are right; missing external files or tools is not itself a correctness issue.</td><td>An XLOOKUP skill instructs the reader to leave the fourth argument blank for an exact-match lookup, when the target spreadsheet application requires it set to 0 – a wrong instruc- tion that silently produces a broken formula.</td></tr><tr><td>Completeness [ba- sic]</td><td>The necessary steps and boundary cases for the task class are covered, with no critical omission.</td><td>A PDF-form-filling skill documents populating interactive fields but never covers a form with no interactive fields at all, leaving the agent with no fallback for a boundary case it will routinely meet.</td></tr><tr><td>Executability [ba- sic]</td><td>Steps are concrete enough for an agent to act on, not merely descriptive.</td><td>A smart-home skill tells the agent to “adjust the appropriate device&quot; without naming the device-location slot or its expected value format, so no executable function call can be built from it as written.</td></tr><tr><td>Cross-lingual faith- fulness [basic]</td><td>No meaning-altering mistranslation or un- translated fragments that would impede use in the target locale.</td><td>A MT Hindi skill inverts the logic of a conditional “if&quot; clause during translation, an error that propagates into every response the skill produces.</td></tr><tr><td>Localization [ba- sic]</td><td>The prose is genuinely written for the ex- pected target locale, not merely translated into it.</td><td>A skill tagged as a Hindi smart-home guide has a translated title but a body written almost entirely in fluent English – usable to a bilingual reader, but not a Hindi-facing skill.</td></tr><tr><td>Context efficiency [advanced]</td><td>Concise enough not to dilute the agent&#x27;s at- tention; an excessively long or padded skill can itself cause execution failure.</td><td>A red-envelope custom skill opens with three paragraphs on the history of Lunar New Year before ever stating the customary gift amount the query actually needs.</td></tr></table>

Table 7: The six Theory-view dimensions (Section 4): detailed explanation and one concrete skill-side judgment case per dimension.

Both the Theory view and the Action view score every retrieved candidate through a single LLM call that returns a structured JSON object. The Theory rubric’s dimensions instantiate the expert-validated multilingual quality taxonomy of Liu et al. (2026c) (content, translation, and localization criteria) extended with the two agent-specific dimensions discussed in Section 4. Fig. 7 and Fig. 8 display the canonical rubric for each view, taken from the General Skill Use domain. For each domain, e.g., the Tool Use and Cultural Skill Use, the rubric reuse the same evaluator framing and 0–100 JSON-object output contract, while substituting the serialized candidate content for the domain at hand — a function schema and required argument slots for Tool Use, a culture entity and target region for Cultural Skill Use. Placeholders ({task}, {skill\_body}, {skill\_title}, {locale}) stand in for the per-instance fields substituted at call time. The 5-shotper-domain router prompt template is shown in Fig. 9.

## E Pool Construction and Leakage Protocol

Each evaluation domain draws its candidates from a threelayer skill pool. The three layers mirror the three ways a multilingual skill comes to exist given the ecosystem documented in Section A, and each preserves the artifacts its own production path naturally produces.

Ecological-style layer. Material in the style of what an agent natively encounters: document-derived skills rendered from public skill repositories and oficial skill or API documentation, and background-derived skills distilled from domain background libraries covering the evaluation’s languages and cultural regions. Section A shows the genuine material of this kind is scarce for several of our target languages and absent for others; where it is thin, the remaining two layers necessarily carry more of the pool’s coverage for that language, approximating what a practitioner assembling a multilingual skill pool today would actually find available.

MT layer. Produced by translating English source material into multilingual versions, this is the most scalable route to coverage when ecological material is scarce. The layer preserves the cross-lingual transfer artifacts of its production path: Lai, Mesgar, and Fraser (2024) report that machine-translation quality for low-resource languages lags markedly behind highresource languages, and Liu et al. (2026c) document that MT content routinely carries content errors, translation defects, and insuficient localization. General Skill Use translates the pool’s English sources; Tool Use translates both a schemaderived document and a self-generated document per function into each non-English target; Cultural Skill Use aggregates translated and cross-lingually adapted culture cards according to the contributing resources. Translation uses a single LLM pass (backbone named in Appendix F).

Self-generated layer. Produced by model self-generation grounded in each domain’s task specifications, mirroring the second scalable route commonly used to fill gaps in a scarce multilingual skill ecosystem. Li et al. (2026) find that self-generated skills of this kind provide no average benefit over curated skills and can hurt task success, and Zhang et al. (2026) note that self-generated material needs verification at construction time; this layer preserves that naturally occurring incompleteness. The generator (backbone named in Appendix F) receives task-type material only: General Skill Use uses source-benchmark task instructions with an explicit instruction to write reusable skill documents; Tool Use uses the function schema plus one example utterance from a held-out split; Cultural Skill Use uses answer-free summaries of its source culture resources. No generator sees an answer key, checker, evaluation utterance, or evaluation outcome, and the Tool Use build additionally scans every pool skill for verbatim evaluation text before accepting the pool.

<table><tr><td>Dimension</td><td>Explanation</td><td>Skill-side judgment case</td></tr><tr><td>Task applicability</td><td>Whether the skill targets this task at all.</td><td>For a query asking to build an XLOOKUP formula, a general “Excel functions overview&quot; skill is topically related but never demonstrates XLOOKUP syntax, so it scores low despite the</td></tr><tr><td>Procedure match</td><td>Whether the skill&#x27;s steps cover what the query actually requires.</td><td>topical overlap. A calendar skill&#x27;s steps assume a single one-time event, but the query asks to schedule a weekly recurring meeting; the skill never covers the recurrence step the query needs.</td></tr><tr><td>Constraint match</td><td>Whether slots, parameters, and constraints line up with the request.</td><td>For the query “turn off the bathroom light,&quot; a smart-home skill documents a device-control function but never states a room/location slot, so its documented constraints do not line up with what the request needs specified.</td></tr><tr><td>Output-format match</td><td>Whether the skill produces the output shape the task expects.</td><td>A tool-use query needs a single well-formed JSON function call; a candidate skill&#x27;s worked examples all end in a natural- language confirmation sentence instead of the structured call the checker expects.</td></tr><tr><td>Language fit</td><td>Whether the prose is usable in the query&#x27;s language.</td><td>A Swahili query is paired with a candidate skill whose instruc- tions are written entirely in French; procedurally sound, but not usable to a reader who only reads the query&#x27;s language.</td></tr><tr><td>Misleading risk</td><td>The chance a related-looking skill steers the solver to a wrong API, value, or convention.</td><td>A wedding-gift-etiquette skill retrieved for a Lunar New Year red-envelope question reads as fluent and on-topic but grounds its answer in the wrong occasion, risking a confident, wrong response.</td></tr></table>

Table 8: The six Action-view dimensions (Section 4): detailed explanation and one concrete skill-side judgment case per dimension.

Every layer passes only a lightweight structural check before inclusion: a well-formed skill body with its target field populated and no empty or truncated content. It is not screened against any test outcome. Table 2 summarizes the three layer totals for each domain; the detailed composition follows.

General Skill Use composition. The 1,750-skill pool contains 400 ecological-style, 550 MT, and 800 self-generated skills. French, Japanese, Korean, and Chinese each receive 100 ecological-style, 75 MT, and 75 self-generated skills, for 250 per language. The ecological-style layer contains no Swahili or Hindi skills, reflecting the community ecosystem audited in Section A; each of these languages instead receives 125 MT and 125 self-generated skills. A further 250 English skills serve as the sources for the translated layer. The six multilingual allocations contribute 1,500 skills, and the English source set brings the pool to 1,750.

Tool Use composition. The 2,299-skill pool contains 1,254 ecological-style documents, comprising 869 background documents and 385 schema-derived skill documents, together with 660 MT and 385 self-generated skills. The pool draws on the source benchmark’s 55-function inventory (Kulkarni et al. 2025): schema-derived and self-generated material each contributes one document per inventory function in seven languages (385 each), while translating both document types for each function into six non-English targets contributes 660 MT skills. The background-document component supplies the remaining 869 ecological-style entries.

Cultural Skill Use composition. Cultural Skill Use is organized along culture regions rather than a fixed language roster. Its 5,285-skill pool contains 3,384 ecological-style, 1,283 MT, and 618 self-generated skills. The ecological-style layer aggregates public culture resources and background libraries, the MT layer combines translated and cross-lingually adapted culture cards, and the self-generated layer draws on answer-free summaries; each layer’s volume follows the distribution of its contributing resources.

Representative pool entries. The three excerpts below, one per layer, are drawn from the actual General Skill Use pool. Each is shown as a compact English gloss of its original targetlanguage prose (code and package identifiers are reproduced verbatim; the source language is bracket-tagged).

Ecological-style layer, Chinese ([zh]), derived from a public community skill repository:

SYSTEM: You are an expert evaluator of AGENT SKILLS (procedural documents an agent   
retrieves, loads into context, and follows to perform a class of tasks). Judge the   
skill’s intrinsic QUALITY. Do NOT judge relevance to any query, and do NOT reward   
verbosity. Score EACH dimension INDEPENDENTLY -- a flaw in one dimension must not   
lower another. You are blind to who authored or translated the skill.   
Expected target locale: {locale}. Judge whether the PROSE of the skill is written   
for this expected target locale. Ignore code blocks, package/API names, file paths,   
commands, and variable names when judging prose language -- keeping those in   
English is normal and must NOT be penalized. Substantial English prose while the   
expected target locale is non-English is a localization failure even if the   
English itself is fluent.   
Dimensions (each scored 0-100; "violated":true means it fails this dimension’s bar)   
- correctness [red-line]: procedural facts/logic are correct; missing external   
files/tools is NOT a correctness issue.   
executability [basic]: steps are concrete and executable by an agent; needed   
tools/files are defined or obtainable.   
completeness [basic]: covers the necessary steps and boundary cases for this   
task class; no critical omission.   
cross-lingual faithfulness [basic]: no meaning-altering mistranslation, no   
untranslated fragments that impede use in the expected target locale.   
localization [basic]: usable as a skill for the expected target locale; penalize   
wrong-language or mixed-language prose, awkward literal translation, and wrong   
locale conventions (code/API names/paths/commands are exempt).   
context-efficiency [advanced]: concise, not bloated; an excessively long or   
padded skill dilutes the agent’s attention and can itself cause execution   
failure.   
Every "score" MUST be an integer 0-100; "confidence" likewise 0-100. Return ONLY   
a JSON object:   
{"dimensions": {"<dim>": {"score": <0-100 int>, "violated": <true|false>,   
"reason": "<=12 words"}}, "root\_cause\_tags": ["..."], "confidence": <0-100 int>}   
SKILL: {skill\_body}

Figure 7: Theory scoring prompt template (General Skill Use domain). The overall Theory score is a deterministic function of the returned dimension scores, computed outside the model call: the mean of the six dimension scores, capped at 40 if any red-line dimension is violated, at 80 if any basic dimension is violated (and no red-line violation occurred), or left uncapped at 100 otherwise — a single severe defect can therefore not be averaged away by otherwise-strong dimensions.

```markdown
# [zh] Prompt Classification
MT layer, Hindi ([hi]), English source skill translated into
Hindi:
# [hi] Filling PDF forms
## [hi] Description
[hi] Reads/writes PDF forms
with interactive fields
using the ‘pypdf‘ library.
pip install pypdf
Self-generated layer, Japanese ([ja]), model self-generation:
# SKILL.md: [ja] Induction
## [ja] Overview
[ja] Proves propositions
about natural numbers in
Lean 4 via induction
(‘Nat.rec‘ / ‘induction‘).
```

Leakage control. We separate test-set isolation from legitimate source inclusion in the searchable pool: (1) Test-set isolation. For General Skill Use, each query is a new input instance, and we reject any task–skill pair whose skill body contains the exact query, exact answer, or serialized gold object. For Tool Use, evaluation utterances and their gold calls, instance slot values, and checker records are held out from pool construction; skills are built from the function inventory, API and background documents, and source examples outside the evaluation split, followed by a verbatim scan against every evaluation utterance. For Cultural Skill Use, each natural user query is written separately from its source card and audited for source-body 8-gram and answer-key-term overlap; canonical answers, accepted aliases, explicit reject examples, and checker rules remain only in the evaluation manifest. (2) Skill-pool isolation. The pool is source-included by design: a reusable skill covering the required procedure, function, or cultural fact may be present because finding and using such material is the object of skill retrieval. The boundary is instance specificity. General and Tool skills may contain reusable procedures, schemas, function names, and slot definitions, but not the evaluation query, serialized gold call, or instance slot values. A Cultural skill may state the fact the task asks the agent to retrieve, but it contains neither the independently written query nor its answer-key and checker metadata. MT and self-generation receive only reusable source material or task-type specifications, never evaluation manifests or downstream outcomes. At inference time, the Theory scorer sees only the anonymous skill body and target locale; the Action scorer sees the user query and anonymous skill body; and the solver sees the task-visible input and selected anonymous skill bodies. Source identifiers, expected-source links, provenance and family labels, gold fields, and checker state remain hidden throughout retrieval, scoring, and solving, and the deterministic checker is applied only after the solver returns its answer.

SYSTEM: You are an action-oriented evaluator for retrieved agent skills. Return   
ONLY a JSON object. Do not use markdown.   
USER: Evaluate whether the CANDIDATE SKILL is expected to help an agent solve   
THIS TASK. You see only the task and the candidate skill document; you are not   
given the source skill id, provenance, gold answer, checker, previous answer, or   
execution result.   
Rules:   
- Judge task-specific expected utility, not intrinsic writing quality.   
- Relevance is not enough. Penalize related skills that can lead to a wrong API,   
field name, formula, step order, locale convention, output shape, or language   
output.   
- High expected\_utility requires strong applicability plus low misleading\_risk.   
- Do not use or mention whether the skill "looks like the original skill"; you   
cannot know that.   
- Code/API/file names may stay in English in non-English tasks. Penalize   
language only when prose mismatch blocks task use.   
Return exactly this JSON schema:   
{"task\_applicability": 0-100, "procedure\_match": 0-100, "constraint\_match": 0-100,   
"output\_format\_match": 0-100, "language\_fit": 0-100, "misleading\_risk": 0-100,   
"expected\_utility": 0-100, "brief\_reason": "<=25 words"}   
TASK: {task}   
CANDIDATE SKILL TITLE: {skill\_title}   
CANDIDATE SKILL BODY: {skill\_body}  
Figure 8: Action scoring prompt template (General Skill Use domain). One scoring call returns the overall Action score (expected\_utility) together with the six dimension scores: the dimension scores walk the scorer through each fit check, and the overall score weighs them for the query at hand, permitted to be high only when the checks pass and the misleading risk is low. The aggregation difers from the Theory view’s fixed mean-with-caps rule (Fig. 7) because the two views judge diferent objects: Theory judges the skill in isolation, where its dimensions are fixed document properties and one rule fits every query, while Action judges the skill against the current query, where the dimension that matters most changes from query to query and no single fixed weighting exists.

## F Implementation Details

Retrieval. BM25 applies Unicode word tokenization to each skill’s concatenated searchable fields and ranks the raw task query with $k _ { 1 } ~ = ~ 1 . 5$ and $b = 0 . 7 5$ (Robertson and Zaragoza 2009). Neural follows the standard denseretrieval design (Karpukhin et al. 2020), computing dense embeddings over cached text-embedding-3-small vectors and fusing them with lexical and language-match signals in the spirit of sparse-dense hybrid retrieval (Luan et al. 2021). SkillFlow follows its released four-stage pipeline (Li, Tagkopoulos, and Tagkopoulos 2025): five generated search queries retrieve a union with bge-base-en-v1.5, ms-marco-MiniLM-L-6-v2 performs shallow reranking, bge-reranker-v2-m3 performs deep reranking, and an LLM selector returns final candidates.

Selection budgets and baselines. The main comparison reports results at every skill budget N ∈ {1, 3, 5, 10}, with Top3 as the primary operating point. Random is averaged over five seeds.

Inference settings. Theory and Action scoring and the main downstream solver run at temperature at most 0.1 with thinking disabled. For downstream trajectory generalization in Table 5, we fine-tune Qwen3.5-9B for one epoch with LoRA (Hu et al. 2022) (rank 16, α = 32, dropout 0.05, targeting all language-model linear layers). All fine-tuning runs were conducted on a single NVIDIA GeForce RTX 4090 GPU.

## G Deterministic Checkers

All three evaluation domains score task success with a fully deterministic, rule-based checker; no held-out task uses an LLM judge for the headline success/failure outcome. This avoids judge variance and scorer-output self-preference in

SYSTEM: You are a lightweight task-type classifier for an agent system.   
Classify the user’s query into exactly one task type:   
- general: stand-alone knowledge, reasoning, coding, document/data manipulation,   
or procedural tasks answered directly by the agent.   
- function: select or invoke an API, tool, app action, or service function and   
fill its arguments.   
- culture: answer centrally depends on culture-specific customs, values,   
etiquette, social norms, history, or regional practices.   
Tie rules: choose culture when culture-specific knowledge is essential. Choose   
function only for an app/tool/service action or intent invocation; an ordinary   
programming question that mentions a function is general. Otherwise choose general.   
Return exactly one JSON object: {"tau":"general|function|culture"}.   
FEW-SHOT EXAMPLES:   
USER: [5 General Skill Use examples]   
ASSISTANT: {"tau":"general"}   
USER: [5 Tool Use examples]   
ASSISTANT: {"tau":"function"}   
USER: [5 Cultural Skill Use examples]   
ASSISTANT: {"tau":"culture"}   
USER: {query}

Figure 9: Domain-router prompt template. Each bracketed placeholder denotes the five fixed demonstrations supplied for that domain.

task success.

General Skill Use. Each of the 94 tasks carries a short, hand-authored set of gold answer strings: 73 tasks have exactly one gold string, 17 have two, and 4 have three. A response is scored correct if either the full output or any single line of it (splitting on newlines, to tolerate a short preamble or postamble around the actual answer) matches a gold string exactly after normalization: curly quotation marks are converted to straight quotes, internal whitespace runs are collapsed to a single space, and the string is trimmed of surrounding whitespace and one layer of surrounding quote characters. Comparison is otherwise an exact string match, with no case-folding, stemming, or numeric tolerance. Extra gold strings almost always encode benign formatting variance of a single correct answer rather than a semantically diferent one – for a spreadsheet lookup-formula task, for instance, the gold set accepts the same XLOOKUP formula with and without a space after each comma, while a formula using the wrong function, cell range, or argument order does not pass.

Tool Use. The solver must return exactly one well-formed JSON object naming one function and its argument slots, with no markdown fence and no surrounding prose; returning zero or multiple functions, or wrapping the JSON in explanatory text, is rejected outright before any name or slot comparison. The predicted function name must then match the gold name exactly (case-sensitive). Every slot the gold call requires must be present with a matching value, with one exception: a small class of “defaults to the current moment” functions may legitimately leave a “now”-valued slot unstated, mirroring how a real utterance would naturally omit an implicit “now.” An extra, unrequested slot is scored as a failure. Slot values (not function names) are normalized before comparison – Unicode width normalization, lowercasing, whitespace collapsing, and stripping of surrounding punctuation including common CJK punctuation marks – and a gold slot’s accepted value may itself be a short list, so that any one entry counts as correct: a color slot with gold value list ["red","crimson"] accepts either name, and a decoratively stylized or full-width prediction normalizes to the same token as a plainly typed one before comparison.

Cultural Skill Use. Each of the 52 tasks carries a compact, individually authored answer specification: one canonical answer, a short curated list of accepted paraphrases, a short curated list of explicit near-miss wrong answers, and a small backup set of required keywords. A response is checked in three ordered stages after lowercasing, punctuation stripping, and whitespace collapsing. First, if the normalized answer matches (as a substring, in either direction) any explicit wrong-answer example, the task fails immediately, regardless of anything else – this catches answers that share surface vocabulary with the question but land on the wrong specific fact. Second, if not, the answer is matched against the canonical answer and its paraphrases (again a substring match in either direction); a hit passes. Third, if neither set matches, the checker falls back to counting required keywords, passing only if a minimum count is reached (one keyword sufices when a task defines only one or two; otherwise at least two must appear). For example, a tipping-etiquette question whose canonical answer is a specific percentage range accepts paraphrases restating the same range in diferent words, while an explicit reject list catches plausible-sounding wrong answers for the same question (e.g. “no tip,” “50 percent”) so they cannot pass merely for sharing the word “tip” or being a percentage; if the free-form answer matches no listed paraphrase, the keyword backstop requires the numeric range or the word “tip” to appear before it can pass.

## H Limitations and Responsible Use

Despite its consistent gains across three evaluation domains and three retrievers, M-SQE has several limitations:

Domain Scope. Our evaluation instantiates three highfrequency skill-use domains, while deployed agents will meet others. The framework is built for this: both scoring views are domain-agnostic — intrinsic quality and task-grounded fit are properties any skill and any query possess — and the router’s taxonomy reserves the General fusion as the default route for every task that is not tool- or culture-heavy, so an unseen domain is scored by the General path rather than falling outside the method. Specializing further is a five-example change: as Section 4 notes, a new domain enters the router with five added examples and a fusion adapter, with no retraining of any component.

Candidate Coverage. M-SQE selects within what the retriever surfaces; it does not author new skills, so the pool’s supply sets the absolute ceiling. Selection, however, is exactly the lever an agent controls at inference time, and our results show it matters most where supply is thinnest: the languages whose native skill supply the audit measures at zero gain the most from M-SQE, because the usable MT and selfgenerated candidates that do exist are found rather than lost among relevant-looking alternatives. The downstream trajectory experiment further shows that M-SQE-selected skills make better training material, so quality estimation also strengthens the loop that produces new skills; growing native supply itself remains a task for the broader community, to which we contribute the audit, the pools, and the rubrics.

Scorer Generality. Each reported run computes both views with a single scoring model. The backbone robustness analysis (Fig. 6) shows the gains survive exchanging the scorer across model families and re-synthesizing the pool, so the finding is not tied to any one scorer; what a single-scorer setup leaves open is aggregation and distillation — combining scorers, or compressing the released rubrics into a lightweight dedicated model to cut serving cost — both direct extensions on top of the prompts we release.

Responsible use. M-SQE targets the language and cultural gaps documented throughout this paper. Our evaluation artifacts (code, skill pools, and tasks) are publicly released at https://github.com/lunyiliu/M-SQE under an MIT (code) and CC-BY 4.0 (data) license. For culturally sensitive domains we recommend a human or community review pass before deployment; M-SQE’s Cultural Skill Use scoring already routes Theory as a factual-reliability check on the retrieved skill (Section 4), giving such a review a first-pass filter to build on rather than a blank slate. We further encourage the community to contribute genuinely native-authored skills for the languages our audit finds most scarce.

## I Case Studies

The mechanism ablation in Section 5 (Table 4) assigns the two views complementary roles: the Action view carries most of the task fit, while the Theory view guards against the occasional catastrophically flawed candidate. The two cases below further instantiate this role division of two views, with per-task selections and dimension scores taken directly from the evaluation logs; both come from Tool Use under BM25 at Top3.

Case 1: the Action view supplies the missing fit signal. A task asks the agent to turn on the kitchen lights (gold function iot.hue\_lighton). Theory-only’s Top-3 never contains an on-function skill: its rank-1 candidate is a skill for turning of the lights — which Theory rates a perfect overall quality of 100 (“clear, uses appropriate terminology. . . maps perfectly to $i o t . h u e \_ l i g h t o f f ^ { \prime \prime } )$ . The correct on-function skill, sitting at retrieval rank 7, earns the same 100: Theory judges each skill on its own quality, with no access to the query by design, so sixteen of the twenty candidates tie at 100 and the selection degenerates to retrieval order, whose top ranks are filled by wrong-function lighting skills that lexically overlap the query almost verbatim. With no on-function skill in its Top-3, the solver calls a wrong function and fails. The Action view separates the pair sharply: the of-function skill scores procedure match 0, constraint match 0, and misleading risk 100 (“the exact opposite of the user’s request”), while the on-function skill scores procedure match 100 and constraint match 100. Action-only and full M-SQE both rank it first and succeed.

Case 2: the Theory view blocks catastrophic content. A second task asks the agent to post a tweet (gold function social.post). Action-only’s Top-3 admits a candidate that names the right function and a slot inventory (“Function: social.post. Slots: business\_name, media\_ $t y p e ^ { , \prime \prime } )$ but is otherwise an annotation-style note about the function’s applicability, with no executable guidance. The Action view reads it as a near-perfect fit — procedure match 100, constraint match 100, misleading risk 0, the scorer noting it “correctly identifies the social.post function and relevant slots” — and with the note in context the solver fills the spurious business\_name slot and fails the slot check. Theory reads the same candidate at an overall quality of 10 (“not a functional tool but a metacommentary note. . . uselessfor execution”), far below Tool Use’s Theory≥65 guard (Section 4); full M-SQE therefore excludes it, admits a genuine social.post skill in its place, and succeeds. Fit dimensions read applicability and cannot establish that a document’s content will execute; catching exactly that gap is the guard role the Theory view plays.