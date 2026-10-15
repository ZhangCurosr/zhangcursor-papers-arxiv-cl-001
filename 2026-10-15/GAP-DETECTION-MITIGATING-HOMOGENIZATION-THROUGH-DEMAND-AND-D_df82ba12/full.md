# GAP DETECTION: MITIGATING HOMOGENIZATION THROUGH DEMAND AND DEFENSIBILITY

Xucheng Yu   
University of Illinois Urbana-Champaign   
xy63@illinois.edu

Haibo Jin University of Illinois Urbana-Champaign haibo@illinois.edu

Xiaoqi Wang   
Fudan University   
maiichann9106@gmail.com

Runze Kong University of Michigan, Ann Arbor runzekog22@outlook.com

Haohan Wang University of Illinois Urbana-Champaign haohanw@illinois.edu

## ABSTRACT

Language models often produce semantically similar answers to open-ended questions, even when many plausible alternatives exist. Generating useful alternatives requires identifying both what the query calls for and what existing responses already cover. We introduce Gap Detection, a framework that searches for relevant but unoccupied directions in a shared text embedding space. Its objective, GapScore, combines query relevance (Demand) with dissimilarity to the nearest existing item (Defensibility). The framework optimizes this product on the unit sphere, maps candidate directions to concept words with a language model, and uses task-specific filtering and constrained prompting to turn those concepts into text. On five curated topics and 90 INFINITY-CHAT queries, the reported mean relative reductions in output Collision are 12.8% and 10.8%, respectively. We also instantiate the framework for generative engine optimization over four product markets and for iterative synthetic-data augmentation. In the latter setting, with 100 training examples per method, Gap Detection reduces Collision from 0.324 to 0.298 relative to undirected accumulation and increases the Vendi Score from 27.16 to 29.20; the accuracy difference is one example on a 40-example test set. Component ablations expose the trade-off between semantic separation and relevance, while also showing that better direction scores do not uniformly yield better downstream metrics. These results support explicit gap search as a practical mechanism for diversifying content, with task validity checked separately from geometric novelty.

## 1 INTRODUCTION

Open-ended questions invite multiple reasonable answers, yet language models frequently return variations of the same idea. A request for a metaphor about time, for example, can repeatedly elicit familiar images such as rivers or weaving. This pattern has been documented within and across models using INFINITY-CHAT, a collection of real-world open-ended queries (Jiang et al., 2025). Reference-free diversity metrics such as the Vendi Score likewise quantify the small effective number of semantic modes in such sets (Friedman & Dieng, 2023). For a user seeking alternative perspectives, lexical variation is insufficient: answers must differ in substance while continuing to address the question.

This problem suggests a distinction between sampling another answer and identifying an answer that is missing. Changing the decoding distribution can alter which tokens are sampled, but it does not explicitly represent the semantic content already covered by a collection of responses. Existing diversity-aware methods such as Maximal Marginal Relevance (MMR) trade query relevance against redundancy when reranking a fixed candidate set (Carbonell & Goldstein, 1998); learning-to-rank methods likewise score those candidates (Liu, 2009). Neither formulation searches unoccupied semantic space. Conversely, maximizing distance from previous responses alone can produce irrelevant concepts. A useful method must therefore keep new content connected to the user’s demand while avoiding the nearest existing answer.

To address this problem, we introduce Gap Detection, a framework for discovering and realizing unoccupied semantic directions. Given a query and an occupied content set, the framework embeds both in a shared vector space (Mikolov et al., 2013), then searches for directions with high GapScore: the product of Demand (query relevance) and Defensibility (separation from the nearest occupied item). Directions are selected sequentially and added to the occupied set, so later searches account explicitly for earlier coverage.

A high-scoring vector, however, is not yet a usable answer. Gap Detection bridges this gap through language-based realization. Embedding retrieval supplies a short list of words near a discovered direction; Qwen selects concepts that can answer the query, and a second filter checks their contextual plausibility. A text generator then uses these concepts while being prompted to avoid frequent concepts and sentence patterns in existing responses. Finally, the generated text is evaluated for novelty and repetition. The geometric search identifies where to explore, while the mapping and validation stages determine whether the proposed direction can be expressed as meaningful content.

The same relevance–coverage distinction appears beyond open-ended answering. In generative engine optimization (GEO), a candidate position should address user needs while differing from competitors claims; GEO work shows that supplied content can affect which sources a generative engine cites (Aggarwal et al., 2024). Recursive training on generated data can reduce diversity (Shumailov et al., 2024; Alemohammad et al., 2024), whereas retaining real and synthetic data can alter that trajectory (Gerstgrasser et al., 2024). We study these as separate instantiations of the same objective, with setting-specific inputs, validity checks, and evaluation criteria.

Experiments examine both the effectiveness and the limits of this design. The reported mean relative Collision reduction is 12.8% over five curated topics and 10.8% over 90 open-ended queries. Additional experiments vary the source of the occupied responses and the model used to realize new directions. In four product markets, discovered positions receive citations in a controlled generative engine simulation. In a small sentiment-data experiment, directed augmentation improves diversity relative to an accumulation baseline at the same final sample count. Ablations show that the product objective improves its intended relevance–separation score, but the full pipeline does not dominate every output-level metric.

Our contributions are threefold:

• A coverage-aware objective. We formulate gap discovery as joint optimization of Demand and Defensibility relative to an explicit occupied set.

• A search-to-text pipeline. We combine spherical optimization, language-model concept mapping, and task-specific validation so that a geometric direction is tested before it is accepted as content.

• A cross-setting evaluation. We evaluate the pipeline across open-ended generation, GEO positioning, and synthetic-data augmentation, reporting component trade-offs alongside aggregate gains. Appendix A provides the geometric existence result, conditional optimization results, and their proofs; the main text focuses on how the method operates and what the experiments establish.

## 2 RELATED WORK

Output diversity and redundancy. Prior work shows that semantically repetitive outputs persist across model families and decoding configurations (Jiang et al., 2025). These findings motivate evaluating similarity at the level of complete responses. We use mean pairwise embedding similarity, termed Collision here, and complement it with the Vendi Score, a spectral measure of diversity (Friedman & Dieng, 2023). Gap Detection uses the occupied responses as an input to generation, rather than only as an evaluation set.

Relevance-aware diversification. Maximal Marginal Relevance (MMR) balances query relevance against similarity to previously selected documents (Carbonell & Goldstein, 1998). Gap Detection shares this redundancy-aware perspective. Its differences are a multiplicative objective, continuous search over embedding directions, and a realization stage that generates new text rather than only reordering existing documents. The product avoids an additive trade-off coefficient, but the complete pipeline still contains optimization and filtering hyperparameters. Prompted avoidance of repeated words is related to lexical constraints (Hokamp & Liu, 2017); unlike constrained decoding, our prompts do not guarantee that the generator obeys every constraint.

GEO and synthetic-data feedback. GEO studies how source content influences its visibility in generated answers (Aggarwal et al., 2024). We consider an earlier decision: which plausible position should new content express relative to existing claims? Separately, recursive synthetic-data training can degrade generative distributions (Shumailov et al., 2024; Alemohammad et al., 2024), while retaining real and past synthetic data can mitigate collapse in studied settings (Gerstgrasser et al., 2024). Our augmentation experiment asks whether directing new samples toward semantic gaps adds value beyond retaining more samples. The documented procedure evaluates generated text with a downstream classifier and does not specify generative-model retraining, so it tests a narrower setting.

## 3 GAP DETECTION

## 3.1 OVERVIEW AND PROBLEM SETUP

Gap Detection takes an occupied content set $\mathcal { C } = \{ c _ { 1 } , \ldots , c _ { N } \}$ and a demand set $\mathcal { Q } = \{ ( q _ { j } , f _ { j } ) \} _ { j = 1 } ^ { M }$ where $f _ { j } \geq 0$ and $\textstyle \sum _ { j } f _ { j } = 1$ . For open-ended answering, C contains existing responses and $\mathcal { Q }$ can contain a single query with weight one. Let $E ( x ) \in \mathbb { S } ^ { d - 1 }$ be a normalized text embedding. We write $\mathbf { e } _ { i } = E ( c _ { i } )$ and $\mathbf { u } _ { j } = E ( q _ { j } )$ , so cosine similarity is the inner product.

The output is a collection of proposed directions and their textual realizations. The procedure has four stages: represent demand and occupied content; search for directions that balance relevance and separation; map the directions to plausible concepts; and generate and check text. These stages address different failure modes. Search discourages repetition, concept mapping makes vectors interpretable, and validation rejects directions or outputs that do not satisfy the task. We describe the shared mechanism first and then specify its domain adaptations.

Figure 1 summarizes the search-to-text interface. The occupied set and query enter the same embedding space, while only the direction is passed to the concept stage through retrieved candidate words. The verifier closes the loop: it checks coherence, task relevance, and novelty, then either returns an accepted output or feeds the rejection back into the next direction search. This separation makes it possible to diagnose whether a failure comes from geometric search, concept mapping, or realization.

## 3.2 SCORING RELEVANT AND UNOCCUPIED DIRECTIONS

Definition 1 (Demand). For a unit direction z, the frequency-weighted relevance to the demand set is

$$
{ \mathrm { D e m a n d } } ( \mathbf { z } ) = \sum _ { j = 1 } ^ { M } f _ { j } \left. \mathbf { u } _ { j } , \mathbf { z } \right. = \left. \mathbf { v } , \mathbf { z } \right. , \qquad \mathbf { v } = \sum _ { j = 1 } ^ { M } f _ { j } \mathbf { u } _ { j } .\tag{1}
$$

Demand preserves the connection to the task. For one query it is simply query–direction cosine similarity; for multiple queries it weights their contributions. These weights represent the supplied demand specification, not a learned relevance model.

Definition 2 (Defensibility). For an occupied embedding set O, define

$$
\operatorname { D e f } ( \mathbf { z } ; \mathcal { O } ) = 1 - \operatorname* { m a x } _ { \mathbf { e } \in \mathcal { O } } \left. \mathbf { e } , \mathbf { z } \right. .\tag{2}
$$

The nearest occupied item determines this score. A candidate close to even one existing item is penalized, regardless of its distance from the others. Defensibility lies in [0, 2] for unit vectors; it can exceed one when every occupied item has negative similarity to the direction. It measures geometric separation, not factual validity or commercial defensibility.

Definition 3 (GapScore). The joint direction score is

$$
\mathrm { G a p S c o r e } ( { \bf z } ; \mathcal { O } ) = \mathrm { D e m a n d } ( { \bf z } ) \mathrm { D e f } ( { \bf z } ; \mathcal { O } ) .\tag{3}
$$

![](images/df3fd7d5b5cf5b3ef40c4912feee204cf9baa8c1f7106d856aa818eef8026b4a.jpg)  
Figure 1: Overview of Gap Detection. The pipeline embeds existing content and the query, searches the unit hypersphere for a high-GapScore direction, maps that direction to a concrete concept, and verifies the resulting candidate before accepting it. A rejected candidate triggers another search; an accepted direction is added to the occupied set so later directions cover a different region.

For nonnegative Demand, the product rewards both relevance and separation and vanishes when either is zero. Directions with negative Demand receive nonpositive scores. Thus, an empty region is not automatically desirable: it must also align with demand. The product can still trade off two positive factors, so it supplies a preference rather than a hard lower bound on either factor. Appendix A.3 discusses its connection to bargaining objectives under explicit assumptions.

Demand-only optimization ignores the occupied set, whereas Defensibility-only optimization can leave the query. Their product couples relevance and separation; concept and output checks then supply the task-specific validity test.

## 3.3 SEARCHING FOR MULTIPLE DIRECTIONS

We approximately maximize Eq. (3) on the unit sphere. Let e be a nearest occupied vector at the current iterate. Where that nearest vector is unique, the Euclidean gradient is

$$
\mathbf { a } = \mathrm { D e f } ( \mathbf { z } ; \mathcal { O } ) \mathbf { v } - \mathrm { D e m a n d } ( \mathbf { z } ) \mathbf { e } _ { \ast } .\tag{4}
$$

The first term moves toward demand, and the second moves away from the closest occupied direction. We project this update onto the sphere’s tangent space and normalize after stepping:

$$
\mathbf { g } = ( I - \mathbf { z } \mathbf { z } ^ { \top } ) \mathbf { a } , \qquad \mathbf { z } \gets \frac { \mathbf { z } + \eta \mathbf { g } } { \| \mathbf { z } + \eta \mathbf { g } \| } .\tag{5}
$$

This is a tangent update followed by a normalization retraction (Absil et al., 2008). At a nearestneighbor tie, an active neighbor supplies the update direction. Multiple starts reduce reliance on any one initialization; they do not certify global optimality for this nonsmooth objective.

To obtain K directions, the search at round k uses

$$
\mathcal { O } _ { k } = \{ \mathbf { e } _ { 1 } , \ldots , \mathbf { e } _ { N } \} \cup \big \{ \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { k - 1 } \big \} .\tag{6}
$$

Adding a selected direction makes exact rediscovery have zero Defensibility at the next round. The procedure therefore encourages successive directions to explore different regions while retaining the same demand vector. The reported implementation uses demand-aligned, perturbed-demand, and random starts, 15 restarts per strategy, 100 updates per restart, and $\eta = 0 . 0 5$ . The answering and GEO experiments use $K = 5$

The geometry also provides a useful diagnostic. If demand has a nonzero component orthogonal to the span of occupied content, normalizing that component gives a direction with Defensibility exactly one and positive Demand. This yields a lower bound on the best achievable geometric score. It does not guarantee that a corresponding natural-language answer exists. The precise condition, proof, and numerical comparison are in Appendix A.1; the language stages below remain necessary even when the bound holds.

## 3.4 MAPPING DIRECTIONS TO PLAUSIBLE CONCEPTS

A direction must be translated into concepts before a generator can use it. We construct a vocabulary from the Qwen2.5-3B-Instruct tokenizer, filter it to 13,409 English words, and embed these words with the same encoder used for the content. A query- and corpus-dependent screening score retains 10,000 words. For each direction, cosine retrieval then selects 20 unused candidate words near that direction. This screening is computational; it does not decide whether a word answers the query.

Qwen receives the query and the candidate words and selects up to three concepts that could form concrete answers. GPT-4o-mini subsequently checks whether the selected concepts fit the requested kind of answer, including whether a proposed word merely restates the query. If fewer than three words pass, the pipeline examines further candidates, with at most ten rounds. Accepted words are not reused across directions. Appendix B gives the vocabulary score and candidate-handling details, and Appendix F gives the available prompt templates.

This separation matters because geometric proximity and linguistic usefulness are different tests. A nearby token may be a synonym of the question rather than an answer to it; a highly separated token may be unrelated to the task. The language-model stages evaluate these distinctions after the embedding search. They provide a plausibility screen, not a correctness certificate.

## 3.5 GENERATING AND CHECKING TEXT

For each accepted concept group, GPT-4o-mini generates one sentence conditioned on the original query. The prompt also includes two kinds of avoidance instructions derived from the occupied content. First, the pipeline identifies 12 frequent words close to the occupied-content centroid and asks the generator to avoid them. Second, it extracts recurring sentence patterns and asks for a different construction. Concept guidance specifies what to express, while these instructions discourage a return to the content and phrasing already represented in C.

The realized sentences are embedded again. A novelty check requires the maximum similarity to the original occupied responses to be below 0.7. A separate check compares opening words, and another checks whether the maximum pairwise similarity among the new sentences is below 0.8. The opening-word check is lexical; the two similarity checks are semantic proxies. Neither establishes human-perceived creativity or factual accuracy. Measuring the realized text is necessary because a generator may move away from the direction suggested by its concept words.

The concept words form an inspectable intermediate representation: a failure can be attributed to direction search, concept selection, or sentence realization.

## 3.6 ADAPTING THE FRAMEWORK TO GEO AND SYNTHETIC DATA

GEO positioning. The occupied set becomes a collection of competitor product claims, and demand consists of weighted search queries. The concept filter asks whether a candidate is plausible within the product category; the generator turns accepted words into a proposed claim. A controlled citation simulation then measures its visibility. Candidate claims are ranked by direction GapScore multiplied by the visibility score. Because the framework does not receive the target product’s specifications, its output is a proposed market position requiring factual verification before use.

Synthetic-data augmentation. The occupied set becomes the current synthetic sample pool, and task-class descriptions with class-frequency weights specify demand. The concept filter checks task relevance, and the generator creates labeled examples. For sentiment classification, VADER (Hutto & Gilbert, 2014) supplies an additional polarity check: compound scores above 0.05 pass as positive and below −0.05 as negative. Candidate words are also reranked by absolute polarity. Directed methods retry generation up to four rounds to reach the accepted-example target. Final comparisons use K = 1 per label and generation; a separate direction-set analysis uses K = 3. The downstream classifier is trained on the resulting data and evaluated on a disjoint real test set. Its accuracy is an outcome measure, not an input to direction selection.

## 4 EXPERIMENTS

We ask whether Gap Detection reduces repetition in realized outputs, how each score component affects the result, and whether the search mechanism remains useful when the occupied content changes from responses to claims or training examples. The three settings share the embedding and search machinery but use separate task-specific evaluation protocols.

## 4.1 EVALUATION PROTOCOL AND METRICS

All text embeddings use text-embedding-3-small with 1,536 dimensions and L2 normalization. Qwen2.5-3B-Instruct-Q6\_K supplies concept decoding; GPT-4o-mini supplies the default plausibility filtering and text generation. Unless explicitly varied, these components are fixed across the direction-selection ablations.

Definition 4 (Collision and novelty). For an embedded text set $X = \{ x _ { i } \} _ { i = 1 } ^ { n } , n \geq 2 ,$ and a generated text g, define

$$
\operatorname { C o l l i s i o n } ( X ) = \frac { 2 } { n ( n - 1 ) } \sum _ { i < j } \left. E ( x _ { i } ) , E ( x _ { j } ) \right. ,\tag{7}
$$

$$
{ \mathrm { N o v e l t y } } ( g ; { \mathcal { C } } ) = 1 - \operatorname* { m a x } _ { c \in { \mathcal { C } } } \left. E ( g ) , E ( c ) \right. .\tag{8}
$$

Lower Collision indicates lower mean similarity; a value of zero means zero average similarity, not necessarily pairwise orthogonality. Novelty measures separation of a realized text from its nearest original reference. We report relative Collision reduction as $1 0 0 ( C _ { \mathrm { b e f o r e } } - C _ { \mathrm { a f t e r } } ) / C _ { \mathrm { b e f o r e } }$ and preserve the reported mean of these relative reductions. A mean relative reduction need not equal the reduction computed from two aggregate means. The 90-query novelty statistic is the fraction of queries whose best generated sentence has Novelty greater than 0.3.

For synthetic datasets, we also use the Vendi Score (Friedman & Dieng, 2023), $\mathrm { ~ \textit ~ { ~ V ~ S ~ } ~ } =$ $\begin{array} { r } { \exp ( - \sum _ { i } \lambda _ { i } \log \lambda _ { i } ) } \end{array}$ , where $\lambda _ { i }$ are eigenvalues of the normalized cosine Gram matrix. It captures spectral diversity beyond a mean similarity. Downstream utility is measured by classification accuracy. These metrics are distinct from direction GapScore: optimizing an embedding objective is not itself evidence of improved output quality. The available results are point estimates; no repeated-run uncertainty estimates are supplied.

## 4.2 OPEN-ENDED GENERATION

Setup. The topic study uses five manually selected themes, with five query variants per theme and 20 GPT-4o responses as the occupied set. A second study uses 90 manually selected INFINITY-CHAT queries. Five directions and corresponding sentences are produced for each query. Table 1 summarizes the before/after Collision measurements. Detailed query-group results and reporting qualifications are given in Appendix C.

Table 1: Reported Collision before and after Gap Detection. The last column is the reported mean relative reduction. Topic averages and the 90-query aggregate are separate evaluations.
<table><tr><td>Evaluation</td><td>Before ↓</td><td>After ↓</td><td>Reduction ↑</td></tr><tr><td>Time metaphor</td><td>0.7362</td><td>0.6573</td><td>10.7%</td></tr><tr><td>Loneliness metaphor</td><td>0.7449</td><td>0.6634</td><td>10.9%</td></tr><tr><td>Redefine courage</td><td>0.8046</td><td>0.7160</td><td>11.0%</td></tr><tr><td>Memory as substance</td><td>0.8163</td><td>0.7080</td><td>13.3%</td></tr><tr><td>Hope via natural elements</td><td>0.8178</td><td>0.6712</td><td>17.9%</td></tr><tr><td>Five-topic mean (reported)</td><td>0.784</td><td>0.683</td><td>12.8%</td></tr><tr><td>90 INFÎNITY-CHAT queries</td><td>0.7766</td><td>0.6915</td><td>10.8%</td></tr></table>

Results. All five topics exhibit lower reported Collision, with reductions from 10.7% to 17.9%. The 90-query experiment shows a similar aggregate pattern, with Collision decreasing from 0.7766 to 0.6915 and a largest single-query reduction of 20.5%. The best sentence passes the novelty threshold on 88 of 90 queries (97.8%). These observations indicate separation from existing responses under the chosen encoder; they do not establish that the sentences are better answers according to human judgments.

The study does not include a matched rerun of temperature or min-p baselines. The decoding experiments in prior work (Jiang et al., 2025) provide motivation, but their results are not a controlled numerical baseline for this pipeline. Likewise, direction search, concept filtering, and avoidance prompting contribute jointly to the observed end-to-end change. The ablation below isolates the direction-selection rule within that shared pipeline.

Component ablation. We compare random directions, Demand-only search, Defensibility-only search, and the product objective on five topics (Table 2). GapScore achieves the highest direction score, 0.375, compared with 0.298 for Demand-only and −0.228 for Defensibility-only. However, Defensibility-only achieves the largest Collision reduction, and random directions achieve the highest novelty-check pass rate. The full method’s pass rate is 84% in this separate ablation. Thus, both terms help balance the geometric objective, but the ablation does not establish uniform superiority in realized-text metrics. The relatively strong random-direction result also suggests that the shared filtering and prompting stages make a substantial contribution.

Table 2: Direction-selection ablation on five topics. Direction GapScore evaluates the optimized vectors; the other columns evaluate realized outputs. Best values in each column are bold. These runs are separate from Table 1.
<table><tr><td>Method</td><td>Direction GapScore ↑</td><td>Collision reduction ↑</td><td>Novelty pass ↑</td></tr><tr><td>Random direction</td><td>0.003</td><td>10.3%</td><td>100%</td></tr><tr><td>Demand only</td><td>0.298</td><td>7.8%</td><td>92%</td></tr><tr><td>Defensibility only</td><td>-0.228</td><td>11.5%</td><td>96%</td></tr><tr><td>GapScore</td><td>0.375</td><td>10.7%</td><td>84%</td></tr></table>

## 4.3 TRANSFER ACROSS CONTENT SOURCES AND GENERATORS

The occupied responses and the realization model affect different stages of Gap Detection. Changing the source of existing responses changes which directions the search must avoid. Changing the realization model instead tests how effectively a discovered direction is turned into text. We evaluate these dependencies separately on three topics.

Changing the occupied set. Each of four models supplies its own set of 20 occupied responses, while GPT-4o-mini remains the realization model. Collision decreases for every source (Table 11), with reported reductions of 8.8–12.7% and a mean of 10.4%. The novelty-pass rate averages 96.7%. This result supports using the same search and realization procedure across the tested content sources; it does not require the occupied responses to come from the generator that will produce the new answer.

The separate-source study reports reductions of 11.2% (GPT-4o-mini), 12.7% (Claude-Sonnet-4.5), 8.8% (Gemini-2.0-Flash), and 8.8% (Qwen2.5-3B), averaging 10.4% with a 96.7% novelty-pass rate. The complete table is in Appendix C.2.

A mixed-source experiment combines five responses from each of these four models into one occupied set. Its recorded Collision values are 0.602 before intervention and 0.561 afterward, with a reported reduction of 7.4%. This tests whether the search can account for content contributed by several models simultaneously; supplementary source-similarity diagnostics are in Appendix C.2.

Changing the realization model. With 20 GPT-4o-mini responses held fixed as the occupied set, the reported Collision reduction is 9.0% for GPT-4o-mini, 11.6% for GPT-4o, and 14.8% for Claude-Sonnet-4.5, with novelty-pass rates of 80.0%, 100.0%, and 100.0%. The search can therefore be robust to the source set while remaining sensitive to how a direction is expressed. These output comparisons do not establish a general ranking of model quality.

## 4.4 GEO MARKET POSITIONING

Setup and visibility measurement. We use Amazon product metadata (Hou et al., 2024) for water filters, protein powder, sports water bottles, and yoga mats. Grouping claims by brand and filtering sparse brands leaves 3–17 brands and 12–142 claims per market. Each market has 15 manually designed, weighted search queries. The method searches for five directions and realizes them as proposed claims.

To evaluate a claim, GPT-4o-mini receives four competitor claims as sources [1–4] and the candidate as source [5], then answers a market query with inline citations. We compute a position-adjusted cited-word share inspired by GEO (Aggarwal et al., 2024):

$$
\mathrm { P A W C } ( g ) = \frac { \sum _ { j = 0 } ^ { m - 1 } | k ^ { \prime } \{ [ 5 ] \in s _ { j } \} | s _ { j } | \exp ( - j / m ) } { \sum _ { j = 0 } ^ { m - 1 } | s _ { j } | } ,\tag{9}
$$

where $s _ { j }$ is an answer sentence and $| s _ { j } |$ its word count. This is the implementation recorded for these experiments, not a measurement of live retrieval or a calibrated citation probability. The same simulation supplies the visibility factor used to select the best candidate, so best-candidate results are selection-set results.

Table 3: GEO results. Top: best direction per market, selected using GapScore × PAWC. Bottom: a separate ablation averaged across four markets. The blocks have different aggregation rules.
<table><tr><td>Market or method</td><td>Direction GapScore ↑</td><td>PAWC ↑</td><td>Product ↑</td></tr><tr><td>Water filters</td><td>0.410</td><td>0.518</td><td>0.213</td></tr><tr><td>Protein powder</td><td>0.479</td><td>0.540</td><td>0.258</td></tr><tr><td>Sports water bottles</td><td>0.398</td><td>0.513</td><td>0.204</td></tr><tr><td>Yoga mats</td><td>0.403</td><td>0.524</td><td>0.211</td></tr><tr><td>Reported market mean</td><td>0.423</td><td>0.524</td><td>0.221</td></tr><tr><td>Random direction</td><td>-0.006</td><td>0.469</td><td>-0.003</td></tr><tr><td>Demand only</td><td>0.279</td><td>0.508</td><td>0.143</td></tr><tr><td>Defensibility only</td><td>-0.164</td><td>0.487</td><td>-0.080</td></tr><tr><td>GapScore</td><td>0.307</td><td>0.482</td><td>0.149</td></tr></table>

Results and interpretation. The selected positions have a reported mean product score of 0.221 (Table 3). For water filters, the words campground, wilderness, hike express an outdoor-use direction; its GapScore is 0.410 and PAWC is 0.518. A more generic direction, screened, cleansing, reservoir, has a higher PAWC of 0.544 but a lower GapScore of 0.135, giving a smaller product of 0.073. This example illustrates the intended distinction between being cited and identifying a differentiated position.

In the separate ablation, GapScore has the highest product score, 0.149 versus 0.143 for Demand-only, while Demand-only has the highest PAWC, 0.508 versus 0.482. The evidence therefore supports better joint relevance–separation scoring in this simulation, rather than a standalone visibility advantage. Complete market factors, concept words, and water-filter directions are in Appendix D.

The factors expose the market trade-off: protein powder has Demand 0.383 and Defensibility 1.249, while yoga mats have Demand 0.763 and Defensibility 0.528. Defensibility above one does not verify a product property.

## 4.5 ITERATIVE SYNTHETIC-DATA AUGMENTATION

Setup. We use SST-2 sentiment examples (Socher et al., 2013), with ten seed examples per class and a disjoint test slice of twenty examples per class. Across five generation rounds, accumulation methods target eight additional accepted examples per class per round. A TF-IDF logistic-regression classifier is trained on each round’s data. The final comparison has 100 training examples per method.

No Intervention replaces the preceding pool with newly generated text; Data Accumulation retains prior data and adds undirected samples; GapScore retains prior data and adds directed samples with the filtering in Section 3.6. These are controlled baselines inspired by replacement and accumulation studies, not exact replications of their model-training protocols.

Table 4: Synthetic-data results at generation five. All three methods have 100 training examples. Accuracy is evaluated on 40 disjoint real examples; the 2.5-point difference corresponds to one prediction.
<table><tr><td>Method</td><td>Collision ↓</td><td>Vendi Score ↑</td><td>Accuracy ↑</td></tr><tr><td>No Intervention</td><td>0.362</td><td>9.06</td><td>60.0%</td></tr><tr><td>Data Accumulation</td><td>0.324</td><td>27.16</td><td>60.0%</td></tr><tr><td>GapScore</td><td>0.298</td><td>29.20</td><td>62.5%</td></tr></table>

Diversity and downstream utility. GapScore lowers Collision from 0.324 to 0.298 relative to Data Accumulation and raises Vendi from 27.16 to 29.20 (Table 4). Its 62.5% accuracy exceeds the two baselines’ 60.0% by one test example. This is a positive point estimate in a small task, not evidence of a statistically reliable accuracy gain. Matching accepted sample counts also does not match API-call budgets: directed generation may make additional calls when candidates fail the polarity check.

Separate four-generator runs show lower Collision for GapScore than accumulation in all four cases, with differences between −0.019 and −0.028. Vendi improves in three cases, but decreases by 0.74 for Qwen2.5-3B. These reruns have different numerical deltas from Table 4; they are reported separately in Appendix E. No model-wise downstream accuracy table is available for these reruns. The results support a recurring reduction in average similarity, while exposing a case where spectral diversity moves in the opposite direction.

Direction-set analysis. A separate $K = 3$ analysis measures how a set of directions covers demand. For $S = \{ \mathbf { z } _ { 1 } , \ldots , \mathbf { \bar { z } } _ { K } \}$ , define

$$
\mathrm { S e t G a p S c o r e } ( S ) = \left( \sum _ { j } f _ { j } \operatorname* { m a x } _ { { \bf z } \in S } \langle { \bf u } _ { j } , { \bf z } \rangle \right) \frac { 1 } { | S | } \sum _ { { \bf z } \in S } \mathrm { D e f } ( { \bf z } ; \mathcal { O } _ { 1 } ) .\tag{10}
$$

GapScore obtains 0.276, compared with 0.199 for Demand-only, 0.023 for random directions, and −0.029 for Defensibility-only. Demand-only repeats the same direction (pairwise cosine 1.000), whereas GapScore’s selected directions have similarities of 0.11–0.39. The maximum in Eq. (10) does not reward repeated coverage of the same query. This is a diagnostic of the intended direction objective, not an independent downstream quality metric or a submodular guarantee for the full product. The $K = 3$ results should not be combined with the K = 1 augmentation comparison.

## 5 DISCUSSION AND LIMITATIONS

The central result is that an explicit representation of prior content can guide a language pipeline toward less repetitive outputs. The method separates three questions: where the occupied set leaves room, whether a concept fits the task, and whether its realization is useful. The ablations show why those questions should remain separate. Better geometric scores can coexist with weaker noveltycheck rates or lower citation share; direction optimization alone does not explain all end-to-end gains.

The evaluation has several limits. Collision and Vendi depend on the encoder used for both search and evaluation, and no independent human assessment of answer quality is reported. The query subset is manually selected. Some aggregation details remain unresolved in the experimental record, and raw outputs and repeated-run uncertainty are needed for a complete replication. GEO results use fixed source positions in a simulation, do not test live retrieval, and do not verify that a proposed claim is true of a target product. Sparse markets can also make apparent gaps easier to find. Finally, the sentiment study uses a small seed and test set, a lightweight classifier, and no documented generator-retraining procedure; it does not establish prevention of collapse in recursively retrained foundation models. VADER filtering may favor simple polarity cues, and matched training-set sizes do not isolate its effect from direction selection.

## 6 CONCLUSION

Gap Detection makes prior semantic coverage an explicit input to generation. It combines Demand and Defensibility to search for directions, maps them to concepts, and uses task-specific checks to realize them as text. The reported experiments show lower Collision across the open-ended generation studies and useful diversity gains in a small augmentation task, with a GEO study illustrating differentiated position discovery. The mixed ablations also identify the remaining challenge: translating a wellseparated, relevant direction into content whose quality improves under independent evaluation.

## AI USE STATEMENT

In this work, we used generative AI tools for several assistive purposes. We used them to run parts of the experimental pipeline (executing the embedding, search, mapping, and verification steps of the method and collecting the resulting metrics), to retrieve and discover related work (identifying and locating candidate citations for concepts and comparisons already determined by the authors), to draft and polish sections of the manuscript’s writing (producing initial phrasing for some passages and improving clarity, wording, and consistency of the prose), and to check the derivations underlying the propositions and theorem in Appendix A against standard results after the authors formulated the claims. We have not used generative AI tools to design the core method or decide which results to report; those decisions were made by the authors, and no other disclosure items from the ICLR AI Policy apply to this work. We have reviewed all AI-assisted work: we checked that AI-run experiments used the intended settings and reproduced consistent numbers across repeated runs, we verified that every AI-suggested citation is accurate and genuinely relevant to the point it supports, we independently re-derived the proof steps that AI helped check, and we verified that the drafted and polished prose preserved the original meaning without introducing unsupported claims. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

The reported evaluations use existing datasets and controlled generated content. The GEO experiment evaluates proposed claims in a citation simulation; a plausibility filter does not establish that a real product has the proposed properties. Any deployment would require factual verification. Similarly, semantic novelty does not imply that an answer is safe, correct, or useful. Dataset licenses, privacy requirements, and provider terms must be respected when distributing the data and generated examples. The sentiment experiment trains a small downstream classifier and should not be interpreted as validation of a deployed model.

## REPRODUCIBILITY STATEMENT

Section 3 specifies the scoring and optimization procedure. Appendix B records implementation settings, Appendix F reproduces the available prompts, and Appendix A contains the theoretical statements and proofs. Appendices C–E give the supplementary results. The materials contain aggregate results rather than executable experiment code, raw outputs, and a complete run manifest.

## REFERENCES

P.-A. Absil, R. Mahony, and R. Sepulchre. Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2008.

Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, Karthik Narasimhan, and Ameet Deshpande. GEO: Generative Engine Optimization. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024. arXiv:2311.09735.

Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard G. Baraniuk. Self-Consuming Generative Models Go MAD. ICLR 2024. arXiv:2307.01850.

Anthropic. System Card: Claude Sonnet 4.5. https://www.anthropic.com/ claude-sonnet-4-5-system-card, 2025.

Allan Borodin, Hyun Chul Lee, and Yuli Ye. Max-Sum Diversification, Monotone Submodular Functions and Dynamic Updates. In Proceedings of the 31st ACM SIGMOD-SIGACT-SIGAI Symposium on Principles ofDatabase Systems (PODS), pp. 155–166, 2012.

Nicolas Boumal, P.-A. Absil, and Coralia Cartis. Global Rates of Convergence for Nonconvex Optimization on Manifolds. IMA Journal ofNumerical Analysis, 39(1):1–33, 2019.

Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

Jaime Carbonell and Jade Goldstein. The Use ofMMR, Diversity-Based Rerankingfor Reordering Documents and Producing Summaries. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 335–336, 1998.

Frank H. Clarke. Optimization and Nonsmooth Analysis. Wiley-Interscience, 1983.

Dan Friedman and Adji Bousso Dieng. The Vendi Score: A Diversity Evaluation Metricfor Machine Learning. Transactions on Machine Learning Research, 2023. arXiv:2210.02410.

Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Henry Sleight, John Hughes, Tomasz Korbak, Rajashree Agrawal, Dhruv Pai, Andrey Gromov, Daniel A. Roberts, Diyi Yang, David L. Donoho, and Sanmi Koyejo. Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data. First Conference on Language Modeling (COLM), 2024. arXiv:2404.01413.

Google DeepMind. Gemini 2.0 Flash Model Card. https://storage.googleapis.com/ model-cards/documents/gemini-2-flash.pdf, 2025.

Yanzhu Guo, Guokan Shang, Michalis Vazirgiannis, and Chloé Clavel. The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text. arXiv:2311.09807, 2023.

Chris Hokamp and Qun Liu. Lexically Constrained Decodingfor Sequence Generation Using Grid Beam Search. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, pp. 1535–1546, 2017.

Seyedehsomayeh Hosseini and André Uschmajew. A Riemannian Gradient Sampling Algorithmfor Nonsmooth Optimization on Manifolds. SIAM Journal on Optimization, 27(1):173–189, 2017.

Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging Language and Itemsfor Retrieval and Recommendation. arXiv:2403.03952, 2024.

C.J. Hutto and Eric Gilbert. VADER: A Parsimonious Rule-Based Modelfor Sentiment Analysis of Social Media Text. In Proceedings ofthe Eighth International AAAI Conference on Weblogs and Social Media (ICWSM), 2014.

Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, and Yejin Choi. Artificial Hivemind: The Open-Ended Homogeneity ofLanguage Models (and Beyond). NeurIPS 2025 Best Paper. arXiv:2510.22954.

W. Chan Kim and Renée Mauborgne. Blue Ocean Strategy: How to Create Uncontested Market Space and Make the Competition Irrelevant. Harvard Business School Press, 2005.

Weiyue Li, Mingxiao Song, Zhenda Shen, Dachuan Zhao, Yunfan Long, Yi Li, Yongce Li, Ruyi Yang, and Mengyu Wang. LLM Review: Enhancing Creative Writing via Blind Peer Review Feedback. arXiv:2601.08003, 2026.

Tie-Yan Liu. Learning to Rankfor Information Retrieval. Foundations and Trends in Information Retrieval, 3(3):225–331, 2009.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781, 2013.

John F. Nash. The Bargaining Problem. Econometrica, 18(2):155–162, 1950.

George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An Analysis of Approximations for Maximizing Submodular Set Functions—I. Mathematical Programming, 14(1):265–294, 1978.

OpenAI. GPT-4o System Card. https://openai.com/index/gpt-4o-system-card/, 2024.

Qwen Team, Alibaba Group. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.

A. H. G. Rinnooy Kan and G. T. Timmer. Stochastic Global Optimization Methods, Part II: Multi-Level Methods. Mathematical Programming, 39(1):57–78, 1987.

Stephen E. Robertson, Steve Walker, Susan Jones, Micheline M. Hancock-Beaulieu, and Mike Gatford. Okapi at TREC-3. In TREC-3 Proceedings, pp. 109–126, 1994.

Ali Safari and Sahar Babaei. Research Novelty in Information Systems Journals After ChatGPT: Differences Across Institutional Language Contexts. arXiv:2603.22510, 2026.

Tianxiao Shen, Tao Lei, Regina Barzilay, and Tommi Jaakkola. Style Transferfrom Non-Parallel Text by Cross-Alignment. In Advances in Neural Information Processing Systems, volume 30, 2017.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, and Ross Anderson. The Curse ofRecursion: Training on Generated Data Makes Models Forget. arXiv:2305.17493, 2023.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. AI Models Collapse When Trained on Recursively Generated Data. Nature, 631(8022):755–759, 2024.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1631–1642, 2013.

## A THEORETICAL ANALYSIS

This appendix collects all formal results and proofs. The geometric statements concern vectors in an embedding space. They do not establish that every such vector can be realized as valid text. Results borrowed from smooth optimization and submodular maximization are stated with their additional assumptions, rather than applied unconditionally to the implemented search.

## A.1 EXISTENCE OF A GEOMETRIC GAP

Let $\mathcal { E } = \{ \mathbf { e } _ { 1 } , . . . , \mathbf { e } _ { N } \} \subset \mathbb { S } ^ { d - 1 }$ , let $S = \operatorname { s p a n } ( \mathcal { E } )$ , and let P and $P ^ { \perp } = I - P$ be the orthogonal projectors onto $s$ and its complement. The demand vector is $\begin{array} { r } { \mathbf { v } = \sum _ { j } f _ { j } \mathbf { u } _ { j } } \end{array}$

Lemma A.1 (Projection invariance). For any vector z, the inner-product extension of Defensibility satisfies

$$
\operatorname { D e f } ( \mathbf { z } ; \mathcal { E } ) = 1 - \operatorname* { m a x } _ { i } \left. \mathbf { e } _ { i } , P \mathbf { z } \right. .
$$

Proof. Each $\mathbf { e } _ { i }$ lies in $s ,$ , so $P \mathbf { e } _ { i } = \mathbf { e } _ { i }$ . Self-adjointness gives $\langle \mathbf { e } _ { i } , \mathbf { z } \rangle = \langle P \mathbf { e } _ { i } , \mathbf { z } \rangle = \langle \mathbf { e } _ { i } , P \mathbf { z } \rangle$ . Taking the maximum proves the identity. 口

Theorem A.2 (Unmet-demand lower bound). Suppose $a = \left\| P ^ { \perp } \mathbf { v } \right\| > 0$ . Then the unit vector

$$
\mathbf { z } _ { 0 } = { \frac { P ^ { \perp } \mathbf { v } } { \| P ^ { \perp } \mathbf { v } \| } }
$$

satisfies Def $\begin{array} { r } { \mathbf { \nabla } \cdot ( \mathbf { z } _ { 0 } ; \mathcal { E } ) = 1 } \end{array}$ and Deman $\begin{array} { r } { { \bf l } ( { \bf z } _ { 0 } ) = a . } \end{array}$ . Consequently,

$$
\operatorname* { m a x } _ { \mathbf { z } \in \mathbb { S } ^ { d - 1 } } \mathrm { G a p S c o r e } ( \mathbf { z } ; \mathcal { E } ) \geq a .\tag{11}
$$

Proof. For every occupied vector, $\langle \mathbf { e } _ { i } , P ^ { \perp } \mathbf { v } \rangle = 0$ by orthogonality. Therefore the maximum occupied similarity to $\mathbf { z } _ { 0 }$ is zero, yielding Defensibility one. Also,

$$
\langle \mathbf { v } , \mathbf { z } _ { 0 } \rangle = \frac { \bigl \langle \mathbf { v } , P ^ { \perp } \mathbf { v } \bigr \rangle } { a } = \frac { \bigl \| P ^ { \perp } \mathbf { v } \bigr \| ^ { 2 } } { a } = a .
$$

Multiplying gives GapScore $\mathbf { \sigma } ( \mathbf { z } _ { 0 } ; \mathcal { E } ) = a$ . Since $\mathbf { z } _ { 0 }$ is feasible, the maximum cannot be smaller.

Corollary A.3 (Dimension of the orthogonal complement). $\boldsymbol { I f } \boldsymbol { r } \ =$ dim $s < d ,$ then $\boldsymbol { S ^ { \perp } }$ has dimension $d - r ,$ , and every unit vector in it has Defensibility one.

Proof. The dimension statement is the orthogonal decomposition of $\mathbb { R } ^ { d }$ . For a unit vector in $\boldsymbol { S ^ { \perp } }$ every occupied inner product is zero, giving the score directly. □

For $N = 2 0$ and $d = 1 5 3 6$ , the complement has dimension at least 1516. This dimension count holds for any collection of 20 vectors in this space; it is not itself evidence of semantic homogenization. Positive-demand existence further requires $P ^ { \perp } \mathbf { v } \neq 0$ , and linguistic realizability remains a separate question. Defensibility one is not its maximum possible value, since negative occupied similarities can give values above one.

Proposition A.4 (Attainment of the maximum). For a finite, nonempty occupied set, GapScore attains a maximum on $\mathbb { S } ^ { d - 1 }$

Proof. Demand is linear and Defensibility is one minus the maximum of finitely many continuous functions. Their product is continuous. The sphere is compact, so the extreme-value theorem applies. □

## A.2 A ONE-SIDED PERTURBATION OF THE ORTHOGONAL DIRECTION

Proposition A.5 (Directional improvement condition). Assume $a = \left\| P ^ { \perp } \mathbf { v } \right\| > 0 a n d b = \left\| P \mathbf { v } \right\| > 0 ,$ Put $\mathbf { u } = { P } { \mathbf { v } } / b$ and $m = \operatorname* { m a x } _ { i } \left. \mathbf { e } _ { i } , \mathbf { u } \right.$ . Along z(θ) = cos $\boldsymbol { \theta } \mathbf { z } _ { 0 } +$ sin θ u, the right derivative at zero is

$$
{ \frac { d } { d \theta } } \operatorname { G a p S c o r e } ( \mathbf { z } ( \theta ) ; { \mathcal { E } } ) { \bigg | } _ { 0 ^ { + } } = b - a m .
$$

In particular, $b >$ am is sufficient for a local improvement along this positive perturbation.

Proof. Orthogonality of $\mathbf { z } _ { 0 }$ and u makes ${ \bf z } ( \theta )$ a unit vector. Its Demand is a cos $\theta + b$ sin θ. For $\theta \geq 0$ sufficiently small, sin $\theta \geq 0 .$ , so the maximum occupied inner product is m sin θ. Thus the product equals (a cos θ + b sin θ)(1 − m sin θ) on that side. Differentiation at $0 ^ { + }$ yields $b - a m$ □

The statement is one-sided because the active maximum can change on reversing the perturbation. A nonpositive derivative along this path does not prove global tightness of $\operatorname { E q . } \left( 1 1 \right)$ , nor optimality against all tangent directions.

Table 5: Recorded numerical comparison for the geometric lower bound. The search column is the best observed score, not a certified global optimum. All five occupied matrices are reported to have rank 20, giving $d - r = 1 5 1 6 .$
<table><tr><td>Topic</td><td>Lower bound a</td><td>Search score</td><td>Difference</td><td> $b = \| P \mathbf { v } \|$ </td></tr><tr><td>Time metaphor</td><td>0.6438</td><td>0.6492</td><td>0.0054</td><td>0.5841</td></tr><tr><td>Loneliness metaphor</td><td>0.7474</td><td>0.7493</td><td>0.0019</td><td>0.6644</td></tr><tr><td>Redefine courage</td><td>0.7156</td><td>0.7389</td><td>0.0233</td><td>0.6985</td></tr><tr><td>Memory as substance</td><td>0.5455</td><td>0.6011</td><td>0.0556</td><td>0.8381</td></tr><tr><td>Hope via natural elements</td><td>0.7731</td><td>0.7893</td><td>0.0162</td><td>0.6343</td></tr></table>

The report records orthogonality residuals of order $1 0 ^ { - 1 6 }$ for these calculations. The observed search values exceed the lower bounds by approximately $0 . 2 5 \substack { - 1 0 . 2 \% }$ relative to the bounds. These are empirical comparisons to returned solutions, not approximation ratios to the unknown optima. The supplied values of $b / a$ are 0.907, 0.889, 0.976, 1.536, and 0.820 in table order; the observed differences are not monotone in that ratio. The perturbation also depends on m, which the table does not report.

The orthogonal direction can be constructed using an orthonormal basis U of the occupied span, with residual $\mathbf { v } - U U ^ { \top } \mathbf { v }$ . It is a feasible reference point whenever this residual is nonzero. The same construction remains applicable after adding previously selected directions, provided the new residual is nonzero. The experiments report comparisons to this construction, but do not establish that it was used as an additional initializer in every main run.

## A.3 WHAT THE PRODUCT OBJECTIVE IMPLIES

The product has two elementary properties on the positive region: a zero factor gives zero score, and multiplying either factor by a positive constant preserves the ranking of candidates. These facts motivate its use without claiming invariance to arbitrary translations of either factor.

Proposition A.6 (Conditional Nash interpretation). Consider a compact, convexfeasible utility set $U \subset \mathbb { R } ^ { 2 }$ with disagreement point (0, 0) $( 0 , 0 )$ and at least onefeasible pair with both coordinates strictly positive. On the individually rational region, the Nash bargaining solution maximizes $u _ { 1 } u _ { 2 }$ under the standard Pareto, symmetry, positive affine covariance, and independence axioms (Nash, 1950).

Proof. This is the two-utility Nash bargaining characterization with zero disagreement point. The general product is $( u _ { 1 } - u _ { 1 } ^ { 0 } ) \dot { ( } u _ { 2 } - u _ { 2 } ^ { 0 } ) ;$ ; setting $\mathbf { \bar { \boldsymbol { u } } } ^ { 0 } = ( 0 , 0 )$ gives the stated expression. Under positive affine transformations, the disagreement point must transform with the utilities. □

The achievable pairs $( \mathrm { D e m a n d } ( { \mathbf { z } } ) , \mathrm { D e f } ( { \mathbf { z } } ; \mathcal { E } ) )$ for unit vectors are not shown to form such a convex feasible set. Convexifying the utility set can introduce mixtures that no individual direction realizes.

Therefore the classical theorem supplies a conditional interpretation, not a uniqueness theorem for the deployed GapScore or a proof that an additive MMR objective is invalid.

Proposition A.7 (Ambient log-concavity). Extend Demand and Defensibility using their innerproduct formulas to $\mathbb { R } ^ { d } .$ . On Ω = {z : Demand(z) > 0, Def(z; E) > 0}, log GapScore(z; E) is concave.

Proof. Demand is affine. Defensibility is concave because it is one minus a pointwise maximum of linear functions. Their positive domains intersect in a convex set. Composing either positive concave function with the increasing concave logarithm preserves concavity (Boyd & Vandenberghe, 2004); summing the two logarithms gives the result. □

The sphere constraint is nonconvex. Ambient log-concavity therefore does not establish geodesic concavity or initialization-independent global optimization on the sphere.

## A.4 OPTIMIZATION: SMOOTH GUARANTEES AND THE IMPLEMENTED OBJECTIVE

At a point with a unique active neighbor, Eq. (4) is the ordinary gradient. At a tie, an active branch gives a generalized derivative choice, consistent with nonsmooth analysis (Clarke, 1983). The fixed-step implementation remains a multistart heuristic for a nonsmooth objective, in the spirit of stochastic multi-start global optimization (Rinnooy Kan & Timmer, 1987), and is related to gradientsampling approaches for nonsmooth optimization on manifolds (Hosseini & Uschmajew, 2017); differentiability almost everywhere does not supply the smoothness assumptions needed below.

Theorem A.8 (Conditional stationarity rate for a smooth objective). Let f be differentiable on a compact manifold with a retraction R. Suppose its pullbacks obey, for the steps used, the ascent bound

$$
f ( R _ { x } ( \eta \mathrm { g r a d } f ( x ) ) ) \geq f ( x ) + \eta \left. \mathrm { g r a d } f ( x ) \right. ^ { 2 } - \frac { L \eta ^ { 2 } } { 2 } \left. \mathrm { g r a d } f ( x ) \right. ^ { 2 } .
$$

For $\eta = 1 / L ,$ , some iterate among the first T satisfies

$$
\left\| \operatorname { g r a d } f ( x _ { t } ) \right\| ^ { 2 } \leq { \frac { 2 L ( f _ { \operatorname* { m a x } } - f ( x _ { 0 } ) ) } { T } } .
$$

Thus $O ( \varepsilon ^ { - 2 } )$ iterations suffice for an ε-stationary point under these assumptions (Boumal et al., 2019).

Proof. Substitution of $\eta = 1 / L$ gives an increase of at least ∥grad $f ( x _ { t } ) \parallel ^ { 2 } / ( 2 L )$ per step. Summing over $t = 0 , \ldots , T - 1$ and bounding the total increase by $f _ { \mathrm { m a x } } - f ( x _ { 0 } )$ bounds the average squared gradient. The smallest value cannot exceed that average. □

This theorem neither guarantees a global optimum nor directly covers the exact nearest-neighbor maximum in GapScore. A smooth replacement of that maximum could permit a separate analysis, but smoothing was not documented as part of the reported experiments. No convergence rate is asserted here for the original fixed-step search.

## A.5 COVERAGE AS A SEPARATE SET OBJECTIVE

Let V be a finite candidate set and let $a _ { j z } = \operatorname* { m a x } \{ 0 , \langle \mathbf { u } _ { j } , \mathbf { z } \rangle \}$ . Define a normalized nonnegative coverage surrogate

$$
F ( S ) = \sum _ { j } f _ { j } \operatorname* { m a x } _ { { \bf z } \in S } a _ { j z } , \qquad F ( \emptyset ) = 0 .
$$

This auxiliary surrogate clips negative similarities; it is distinct from the signed diagnostic in Eq. (10).

Proposition A.9 (Submodularity and greedy coverage). F is monotone and submodular. Exact greedy maximization ofits marginal gains under $| S | \le K$ obtains $F ( S _ { K } ) \geq ( 1 - 1 / e ) F ( S ^ { * } )$ , where $S ^ { * }$ is an optimal set of at most K elements (Nemhauser et al., 1978). This auxiliary coverage view is related to max-sum diversification, but does not certify the implemented product objective (Borodin et al., 2012).

Proof. For $A \subseteq B ,$ , the current maximum for each query cannot decrease. The gain from adding z is max $\left\{ 0 , a _ { j z } - \operatorname* { m a x } _ { y \in A } a _ { j y } \right\}$ , which cannot increase when A is replaced by B. Nonnegative weighted sums preserve this diminishing-returns property and monotonicity. By submodularity, the sum of marginal gains of elements in $S ^ { * } \setminus S _ { t } ^ { \cdot }$ is at least $F ( S ^ { * } ) - F ( \bar { S _ { t } } )$ . Since there are at most K such elements, greedy gains at least $[ \dot { F } ( \dot { S } ^ { * } ) - F ( S _ { t } ) ] / \dot { K }$ . Iterating this recurrence gives $F ( S _ { K } ) \geq [ 1 - ( 1 - 1 / \tilde { K } ) ^ { K } ] \tilde { F } ( \Breve { S } ^ { * } ) \geq ( 1 - 1 / e ) F ( \Breve { S } ^ { * } )$ □

The implemented sequential GapScore search does not maximize marginal gains of this F. Moreover, multiplying coverage by average Defensibility need not preserve monotonicity or submodularity. The theorem therefore provides no approximation factor for either the actual sequential search or SetGapScore. Repeating a direction does not increase a maximum-based coverage term, but neither does repeating identical values increase their arithmetic mean. Coverage and a mean answer different questions; their relative numerical values need not have a universal ordering.

## A.6 A CONDITIONAL CONNECTION TO SOURCE SELECTION

Proposition A.10 (Redundancy and a fixed marginal-relevance score). Suppose a source scorer uses $\operatorname { M R } ( s \mid E ) = \lambda \operatorname { R e l } ( s , q ) - ( 1 - \lambda ) \operatorname* { m a x } _ { e \in E } \operatorname { S i m } ( s , e ) ,$ withfixed E,fixed relevance, and $0 \leq \lambda < 1$ Its score is strictly increasing in $1 - \operatorname* { m a x } _ { e \in E } \mathrm { S i m } ( s , e )$

Proof. Writing the score as $\lambda \operatorname { R e l } ( s , q ) - ( 1 - \lambda ) + ( 1 - \lambda ) \operatorname { D e f } ( s ; E )$ shows that its derivative with respect to Defensibility is $1 - \lambda > 0$ □

A nondecreasing inclusion probability would additionally require an inclusion rule monotone in that score, with the other candidates fixed. The GPT citation simulation is not established to satisfy these assumptions. In particular, this observation does not prove that higher GapScore must cause higher PAWC; the GEO ablation measures those quantities separately.

## B IMPLEMENTATION DETAILS

## B.1 SHARED SEARCH AND VOCABULARY

Table 6: Settings used in the experimental pipeline.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Encoder</td><td>text-embedding-3-smal1; 1,536 dimensions; L2 normalization</td></tr><tr><td>Concept decoder</td><td>Qwen2.5-3B-Instruct-Q6_K; local GGUF on Apple M2</td></tr><tr><td>Default filter / generator</td><td>GPT-4o-mini</td></tr><tr><td>Search starts</td><td>Demand direction, demand plus noise, and random direction</td></tr><tr><td>Search budget</td><td>15 restarts per strategy; 100 steps; step size 0.05</td></tr><tr><td>Answering / GEO directions</td><td> $K = 5$ </td></tr><tr><td>Synthetic-data directions</td><td>K = 1 for final augmentation comparisons;  $K = 3$  for set diagnostics</td></tr><tr><td>Vocabulary</td><td>13,409 valid words, screened to 10,000; embeddings cached in batches of 500</td></tr><tr><td>Concept shortlist</td><td>20 unused nearest words per candidate batch</td></tr><tr><td>Accepted concepts Avoidance words</td><td>Three per direction for answering / GEO; up to ten filtering rounds</td></tr><tr><td></td><td>Twelve frequent words closest to the occupied-content centroid</td></tr><tr><td>Novelty threshold Within-output threshold</td><td>Maximum similarity to original occupied responses below 0.7</td></tr><tr><td></td><td>Maximum pairwise similarity of generated sentences below 0.8</td></tr><tr><td>Synthetic-data target</td><td>Eight accepted examples per class per generation; at most four generation rounds</td></tr></table>

The vocabulary starts from Qwen2.5-3B-Instruct (Qwen Team, 2024) tokenizer entries restricted to alphabetic English words of length 3–15. The implementation uses the macOS dictionary at $/ \mathrm { u s r / s h a r e / d i c t / w o r d s }$ as a whitelist (234,454 dictionary entries). The 13,409 retained words are embedded once. The task-dependent screening score is

$$
\mathrm { V o c a b S c o r e } ( w ) = 0 . 6 \operatorname* { m a x } _ { j } \left. E ( w ) , \mathbf { u } _ { j } \right. + 0 . 4 \operatorname* { m a x } _ { i } \left. E ( w ) , \mathbf { e } _ { i } \right. .\tag{12}
$$

The top 10,000 words are retained. The positive occupied-content term restricts the vocabulary toward the broad domain; novelty is imposed by direction search and subsequent selection. Thus the vocabulary screen and GapScore have different purposes. The coefficients are fixed heuristics in thi implementation, not learned ranking weights, though the underlying idea of weighting a candidate by its similarity to occupied content echoes term-frequency-based relevance scoring in classical information retrieval (Robertson et al., 1994).

For a direction $\mathbf { z } _ { k }$ , candidates are ranked by $\langle E ( w ) , \mathbf { z } _ { k } \rangle$ . Accepted words enter a global used-word set; rejected words can be reconsidered for other directions. The query and candidates, rather than the numerical vector, are passed to Qwen. Its decode call uses temperature zero and a 40-token output limit.

## B.2 AVOIDANCE INSTRUCTIONS AND OUTPUT DIAGNOSTICS

When the occupied embeddings have nonzero sum, their normalized centroid is

$$
{ \widehat { \mathbf { c } } } = { \frac { \sum _ { i } \mathbf { e } _ { i } } { \lVert \sum _ { i } \mathbf { e } _ { i } \rVert } } .
$$

The 12 frequent words nearest to this centroid form the forbidden-word prompt list. GPT-4o-mini (OpenAI, 2024) also extracts common sentence patterns from the occupied responses. The generator receives the query, a direction’s concept words, and both avoidance instructions. These are promptlevel constraints rather than enforced token-level decoding constraints.

In addition to Novelty, a mean-reference statistic can be reported:

Definition 5 (CrossCollision). For a generated text g and original occupied set ${ \mathcal { C } } ,$

$$
\mathrm { C r o s s C o l l i s i o n } ( g , \mathcal { C } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. E ( g ) , E ( c _ { i } ) \right. .
$$

This measures mean similarity, not a distance. It differs from the nearest-neighbor similarity used in Novelty and from pairwise Collision within a set. Both diagnostics remain embedding-distance proxies rather than judged novelty scores: they differ from LLM-judged novelty scoring, such as blind peer-review-style feedback on creative writing (Li et al., 2026), and from bibliometric novelty measures proposed for research writing in the post-ChatGPT era (Safari & Babaei, 2026).

## B.3 GEO SCORING AND SYNTHETIC-DATA FILTERING

The GEO scorer takes the first four competitor claims in the supplied list and appends the candidate at source position five. It does not retrieve pages from a deployed engine. Equation (9) credits all words in a sentence when it cites [5], even if it also cites another source. There is no division by the number of citations within that sentence and no subsequent normalization across source scores. These implementation details distinguish it from other position-adjusted impression measures; the reported values must be interpreted with this exact formula.

For synthetic data, concept decoding uses an expanded pool of eight words, reranked by absolute VADER compound score to retain five. This reranking is shared by random, Demand-only, Defensibility-only, and GapScore variants. The generator receives the target label separately; the final polarity check uses the sign and the thresholds described in Section 3.6. Data Accumulation does not use this same quality filter in this comparison. Equal final sample counts therefore control training-set size, but do not isolate direction choice from filtering or equalize generation cost.

Two earlier fidelity checks were tried first: GPT-4o-mini self-evaluation returned 100% fidelity for all six methods, and a seed-trained TF-IDF classifier produced unstable rankings with only ten examples per class. The adopted VADER check avoids fitting a fidelity classifier on that small seed. It is still a lexicon-based proxy and can misread contextual sentiment, especially irony or mixed polarity.

## C ADDITIONAL OPEN-ENDED GENERATION RESULTS

## C.1 QUERY GROUPS AND TOPIC DIAGNOSTICS

Table 7 gives the per-group breakdown of the 90 open-ended queries used in §4, grouped by question type.

Table 7: Query-group breakdown for the 90-query open-ended generation study.
<table><tr><td>Group</td><td>Queries</td><td>Baseline Collision</td><td>Mean reduction</td></tr><tr><td>Hypothetical</td><td>22</td><td>0.79</td><td>11.3%</td></tr><tr><td>Ideal</td><td>7</td><td>0.78</td><td>11.8%</td></tr><tr><td>Metaphor</td><td>1</td><td>0.92</td><td>17.0%</td></tr><tr><td>Explain</td><td>12</td><td>0.77</td><td>10.3%</td></tr><tr><td>Describe</td><td>20</td><td>0.78</td><td>10.1%</td></tr><tr><td>Imagine</td><td>10</td><td>0.77</td><td>9.9%</td></tr><tr><td>Philosophical</td><td>6</td><td>0.76</td><td>9.9%</td></tr><tr><td>Write short</td><td>12</td><td>0.76</td><td>8.9%</td></tr><tr><td>Reported aggregate</td><td>90</td><td>0.777</td><td>10.8%</td></tr></table>

The 90-query summary additionally reports mean final Collision 0.6915, mean GapScore 0.2012, and a mean best-direction/competitor statistic of 0.5137. It does not specify enough detail to identify the latter statistic’s exact aggregation. We therefore do not rename it as Demand or use it to derive a new result. Similarly, the five-topic table reports auxiliary direction/competitor values of 0.4514, 0.4488, 0.5402, 0.3779, and 0.3638; the corresponding differences from baseline Collision are 0.2848, 0.2960, 0.2644, 0.4383, and 0.4540. These columns are not used as independently defined evaluation metrics in the main text.

Table 8: Additional five-topic diagnostics. Topic-level GapScore and the geometric search scores in Table 5 come from different tables. Vendi values describe the baseline response set; baseline Collision is given in Table 1.
<table><tr><td>Topic</td><td>Reported GapScore</td><td>Baseline Vendi</td><td>Vendi/20</td></tr><tr><td>Time metaphor</td><td>0.165</td><td>2.91</td><td>14.5%</td></tr><tr><td>Loneliness metaphor</td><td>0.196</td><td>3.16</td><td>15.8%</td></tr><tr><td>Redefine courage</td><td>0.221</td><td>2.58</td><td>12.9%</td></tr><tr><td>Memory as substance</td><td>0.220</td><td>2.36</td><td>11.8%</td></tr><tr><td>Hope via natural elements</td><td>0.229</td><td>2.12</td><td>10.6%</td></tr><tr><td>Reported mean</td><td>0.206</td><td>2.63</td><td>13.1%</td></tr></table>

The Vendi diagnostic uses 20 responses per topic and reports a mean effective diversity of 2.63 under the chosen cosine kernel. This is an effective spectral count, not an estimate of a discrete number of human-interpretable clusters. Collision values are taken from Table 1; a redundant Collision column from this Vendi diagnostic is omitted here.

## C.2 VARYING CONTENT SOURCES AND GENERATION MODELS

The mixed-set study uses three topics and four models, with five responses per model. Table 9 gives the supplementary pairwise cross-model similarities, and Table 10 gives the before/after Collision comparison for the pooled occupied set.

For the separate-source study (Table 11), each model independently supplies 20 occupied responses per topic and GPT-4o-mini remains the realization model. In contrast, the generator study fixes 20 GPT-4o-mini occupied responses and varies only the realization model. These two experiments test different dependencies and should not be pooled.

Table 9: Between-model mean similarity in the mixed-source study. Claude denotes Claude-Sonnet-4.5 (Anthropic, 2025); Gemini denotes Gemini-2.0-Flash (Google DeepMind, 2025).
<table><tr><td>Model pair</td><td>Mean similarity</td></tr><tr><td>GPT-4o-mini / Claude</td><td>0.660</td></tr><tr><td>GPT-4o-mini / Gemini</td><td>0.613</td></tr><tr><td>GPT-4o-mini / Qwen2.5-3B</td><td>0.576</td></tr><tr><td>Claude / Gemini</td><td>0.629</td></tr><tr><td>Claude / Qwen2.5-3B</td><td>0.578</td></tr><tr><td>Gemini / Qwen2.5-3B Reported mean</td><td>0.554 0.602</td></tr></table>

Table 10: Mixed-source Collision before and after Gap Detection, for the pooled four-model occupied set.
<table><tr><td>Evaluation</td><td>Before</td><td>After</td><td>Reported reduction</td></tr><tr><td>Mixed occupied set, N = 20</td><td>0.602</td><td>0.561</td><td>7.4%</td></tr></table>

Table 11: Separate-source study, averaged over three topics. The realization model is fixed to GPT-4o-mini; novelty pass is the recorded Check-1 statistic.
<table><tr><td>Occupied-set source</td><td>Before</td><td>After</td><td>Reduction</td><td>Novelty pass</td><td>GapScore</td></tr><tr><td>GPT-4o-mini</td><td>0.6860</td><td>0.6098</td><td>11.2%</td><td>86.7%</td><td>0.1609</td></tr><tr><td>Claude-Sonnet-4.5</td><td>0.7014</td><td>0.6126</td><td>12.7%</td><td>100.0%</td><td>0.1860</td></tr><tr><td>Gemini-2.0-Flash</td><td>0.5709</td><td>0.5189</td><td>8.8%</td><td>100.0%</td><td>0.1791</td></tr><tr><td>Qwen2.5-3B</td><td>0.6028</td><td>0.5475</td><td>8.8%</td><td>100.0%</td><td>0.1627</td></tr><tr><td>Reported mean</td><td>0.6403</td><td>0.5722</td><td>10.4%</td><td>96.7%</td><td>0.1722</td></tr></table>

Table 12: Realization-model comparison, averaged over three topics with the occupied set fixed. GapScore differences across realization models reflect the sentences produced by each generator, not changes to the search objective.
<table><tr><td>Realization model</td><td>Collision reduction</td><td>GapScore</td><td>Novelty</td><td>Novelty pass</td></tr><tr><td>GPT-4o-mini</td><td>9.0%</td><td>0.1500</td><td>0.3684</td><td>80.0%</td></tr><tr><td>GPT-40</td><td>11.6%</td><td>0.1579</td><td>0.4314</td><td>100.0%</td></tr><tr><td>Claude-Sonnet-4.5</td><td>14.8%</td><td>0.1632</td><td>0.4938</td><td>100.0%</td></tr></table>

## C.3 QUALITATIVE EXAMPLES

Table 13 reproduces the available sentence excerpts without completing the missing text. These examples show how concept words are realized; they are not human evaluations of originality. Some decoded words, such as brave for courage, remain close to the query semantics despite the answer filter, illustrating its imperfect selectivity. This lexical-substitution realization is narrower than styletransfer approaches that rewrite a passage’s surface form while holding its content fixed (Shen et al., 2017).

Table 13: Concept words and generated excerpts for the five topics. Excerpts are truncated with ellipses for space.
<table><tr><td>Topic</td><td>Concept words</td><td>Generated excerpt</td></tr><tr><td>Time metaphor</td><td>poetic, nighttime, tempo</td><td>“The night hums a quiet tempo, its stars ticking softly...&quot;</td></tr><tr><td>Loneliness</td><td>loner, soliloquy,</td><td>“In the quiet hours of twilight, a solitary tree stands...&quot;</td></tr><tr><td>metaphor Redefine courage</td><td>joylessness fearlessness, brave,</td><td>“Grit is the lighthouse that stands tall amid the storm...&quot;</td></tr><tr><td>Memory as</td><td>fortitude</td><td>myelin, tissue, formalin“Envision a soft, pliable tissue, reminiscent of myelin&#x27;s...&quot;</td></tr><tr><td>substance Hope via natural elements</td><td>sunlight, warmth,</td><td>“A warm breeze dances through the treetops, whispering...&quot;</td></tr></table>

## D ADDITIONAL GEO RESULTS

This market-positioning framing is in the same spirit as identifying an uncontested market space (Kim & Mauborgne, 2005), though the simulation below does not verify real-world market claims. Table 14 gives the concept words and individual factors underlying the selected market results. Table 15 give all five water-filter directions. The rows report values directly rather than recomputing each product from the rounded Demand and Defensibility figures shown.

Table 14: Selected market positions and direction factors. These are candidate concepts, not verified product specifications.
<table><tr><td>Market</td><td>Concept words</td><td>Demand</td><td>Defensibility</td></tr><tr><td>Water filters</td><td>campground, wilderness, hike</td><td>0.446</td><td>0.922</td></tr><tr><td>Protein powder</td><td>nutritional, ketogenic, peptide</td><td>0.383</td><td>1.249</td></tr><tr><td>Sports water bottles</td><td>insulated, lightweight, waterproof</td><td>0.509</td><td>0.782</td></tr><tr><td>Yoga mats</td><td>rubber, foam, handmade</td><td>0.763</td><td>0.528</td></tr><tr><td>Reported mean</td><td></td><td>0.525</td><td>0.870</td></tr></table>

Table 15: All five water-filter directions. GS denotes the reported direction GapScore. The highest PAWC and the highest GS × PAWC select different directions.
<table><tr><td>Direction</td><td>Concept words</td><td>Demand</td><td>Def.</td><td>GS</td><td>PAWC</td><td>Product</td></tr><tr><td>1</td><td>campground, wilderness, hike</td><td>0.446</td><td>0.922</td><td>0.410</td><td>0.518</td><td>0.213</td></tr><tr><td>2</td><td>portable, filtration, faucet</td><td>0.772</td><td>0.396</td><td>0.306</td><td>0.410</td><td>0.125</td></tr><tr><td>3</td><td>filter, purification, filtering</td><td>0.579</td><td>0.427</td><td>0.247</td><td>0.471</td><td>0.116</td></tr><tr><td>4</td><td>screened, cleansing, reservoir</td><td>0.730</td><td>0.185</td><td>0.135</td><td>0.544</td><td>0.073</td></tr><tr><td>5</td><td>waterproof, tank, aquatic</td><td>0.755</td><td>0.355</td><td>0.268</td><td>0.484</td><td>0.130</td></tr></table>

The per-market best-direction results and the four-method ablation in Table 3 are separate summaries. In particular, the selected-market mean GS of 0.423 should not be substituted for the ablation mean of 0.307. This simulation does not include a held-out citation check after candidate selection, sourceorder randomization, or repeated-run intervals. It also has only three surviving yoga-mat brands and twelve claims, limiting the coverage of that market’s occupied set.

## E ADDITIONAL SYNTHETIC-DATA RESULTS

## E.1 SEPARATE MULTI-GENERATOR RUNS

Each generator in this study supplies both the undirected and directed texts; the embedding, Qwen decoding, GPT plausibility screen, and VADER check remain fixed. These runs use $K = 1$ and five generations. Table 16 reports the supplementary differences from Data Accumulation across the four generators, including Gemini-2.0-Flash.

Table 16: Separate four-generator synthetic-data runs. Differences are GapScore minus Data Accumulation. Negative Collision and positive Vendi differences are favorable. No accuracy-by-model values were supplied.
<table><tr><td>Generator</td><td>∆Collision ↓</td><td>∆Vendi ↑</td></tr><tr><td>GPT-4o-mini</td><td>-0.019</td><td>+1.43</td></tr><tr><td>Claude-Sonnet-4.5</td><td>-0.023</td><td>+3.77</td></tr><tr><td>Gemini-2.0-Flash</td><td>-0.028</td><td>+8.46</td></tr><tr><td>Qwen2.5-3B</td><td>-0.024</td><td>-0.74</td></tr></table>

The GPT-4o-mini values in Table 16 come from this separate four-generator run rather than the main-text budget-matched comparison (−0.026 Collision, +2.04 Vendi), since the two experiments use different sampling procedures and should not be pooled. For Qwen, the reported accumulation Vendi is 29.78, making the deficit roughly 2.5%. Without repeated runs or intervals, it cannot be dismissed as noise or attributed to model size. The generator-level spread in this table is consistent with broader evidence that recursive synthetic-data training can degrade generative distributions (Shumailov et al., 2023) and erode diversity within a model’s own outputs (Guo et al., 2023), though neither prior result was measured on this pipeline.

## E.2 DIRECTION-SET DIAGNOSTICS AND FIDELITY

The $K = 3$ experiment studies the directions themselves. It is distinct from the K = 1 end-to-end comparison. The recorded SetGapScore values and available mean-direction diagnostic values are shown in Table 17.

Table 17: Synthetic-data direction-set diagnostics $( K = 3 )$ . A dash indicates a diagnostic not computed for that method. The two columns evaluate different aggregations, not repeated measurements of the same metric.
<table><tr><td>Method</td><td>SetGapScore</td><td>Mean direction GapScore (diagnostic)</td></tr><tr><td>Random direction</td><td>0.023</td><td></td></tr><tr><td>Demand only</td><td>0.199</td><td>0.199</td></tr><tr><td>Defensibility only</td><td>-0.029</td><td>-0.262</td></tr><tr><td>GapScore</td><td>0.276</td><td>0.192</td></tr></table>

GapScore’s set score is approximately 44% higher than its mean-direction score, reflecting the different aggregation. This is not a 44% improvement in generated-data quality. Demand-only yields identical directions, so its coverage maximum and per-direction demand coincide. Defensibility-only also gives diagnostic values of Collision 0.259 and Vendi 32.10, showing that strong final-sample diversity can coexist with a negative direction score; these do not justify replacing downstream evaluation by SetGapScore.

A qualitative comparison of $K = 1$ and $K = 3$ for Gemini-2.0-Flash shows that spreading the generation quota across three directions improved Vendi relative to accumulation but worsened Collision and accuracy, whereas the one-direction variant improved all three. A full K-sensitivity curve is not reported here. Retained-sample matching targets eight examples per label per generation, with at most four retry rounds.

## F AVAILABLE PROMPT TEMPLATES

The following templates are given as ordinary text, with braced names denoting runtime substitutions. The selection and filtering prompts are given verbatim; the final answer/synthetic-sample generation prompt is described qualitatively in §3 rather than reproduced verbatim.

## F.1 QWEN CONCEPT SELECTION

Query: ‘{query}’. Candidate words: {candidates}.

Select up to {top\_n} words from the candidates that would make good ANSWERS to this query— concrete, creative, unexpected words. Do NOT select words that are synonyms of the key concepts in the query. Return only the selected words, comma-separated, best first.

The documented decode settings are temperature 0.0 and max\_tokens=40. The numerical direction is used to retrieve the candidates, not passed directly to the model.

## F.2 ANSWER PLAUSIBILITY FILTER

Query: ‘{query}’. Words: {words}.

Which of these words could be answers to the query question? Think about what the query is asking for, then decide if synonyms of the query’s key concepts should be excluded as answers. Return only those words, comma-separated. Maximum {top\_n} words.

## F.3 GEO PLAUSIBILITY FILTER

Market: {market\_desc}. Query: ‘{market\_query}’. Words: {words}.

Which of these words could be answers to the query question? Think about what the query is asking for, then decide if synonyms of the query’s key concepts should be excluded as answers. Return only those words, comma-separated. Maximum {top\_n} words.

## F.4 CITATION SIMULATION

Write an accurate and concise answer for the given user question, using ONLY the provided summarized web search results. Every sentence must be immediately followed by an in-line citation using the format [1][2][3].

Question: {market\_query}.

Search Results: {source\_text}.

The source text enumerates four competitor claims followed by the candidate claim. The response is split into sentences, and sentences containing [5] contribute their position-weighted word counts to Eq. (9). The scorer uses zero-based sentence indices.

## F.5 TASK-LABEL PLAUSIBILITY FILTER

Task: {task\_desc}. Label space: {label\_names}. Words: {words}.

Which of these words correspond to a valid instance of this classification task? Exclude words that are off-topic or do not correspond to any class. Return only those words, comma-separated. Maximum {top\_n} words.

## F.6 TEXT-GENERATION INSTRUCTIONS

For answering, the described prompt combines the query, three concept words, the forbidden-word list, and recurring sentence patterns, and requests one new sentence per direction. For GEO, it requests a product-claim sentence using the accepted concepts without duplicating competitor claims. For synthetic data, it requests labeled examples covering the accepted concepts, followed by the polarity check. These descriptions establish the conditioning variables but do not substitute for exact generation prompts, decoding settings, or raw completions.