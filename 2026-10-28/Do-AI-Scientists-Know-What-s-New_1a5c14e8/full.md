# Do AI Scientists Know What’s New?

Miri Liu Xiayu Zhao Encheng Xie ChengXiang Zhai

University of Illinois, Urbana-Champaign {miri3, xiayuz3, encheng2, czhai}@illinois.edu

## Abstract

A major challenge facing AI scientific discov ery pipelines is how to determine when a generated idea is meaningfully novel. Ideally, nov elty checks in these pipelines must be able to reject non-novel ideas while not rejecting too many novel ideas, and they should also be able to distinguish which parts of a proposal are non-novel, as opposed to opaquely discarding entire proposals. We design an analytical framework for evaluation of novelty checks with temporal ground truth at both the proposal- and claim-level, using four datasets of 1,060 pro posals built from published papers, reviewer disputes, and generated ideas matching ones the field went on to publish. Auditing existing novelty checks with this framework, we find that checks with high rejection rates on non-novel work also tend to have high false alarms on ideas that are later realized by hu man researchers, and observe that checks built to evaluate proposals as a whole can point to claim-level prior art only by rejecting the whole proposal. We therefore introduce CLAIMNOV, a fine-grained, interpretable claim-level novelty module. CLAIMNOV catches 97% of disguised papers at a 1.5% false-alarm rate, and its claim-level novelty profile cuts the LLM judge’s false alarms from 11.8% to 1.5%. Because its verdict rests on claim-level evidence, CLAIMNOV also keeps proposals that build on established contributions, rejecting only 3% of ICLR-accepted proposals with a reviewerdisputed contribution and keeping 91 of 100 novel proposals with one spliced prior contribution, against 76 for the strongest existing check. Code and data are available at https://github.com/emmatliu/claimnov.

## 1 Introduction

The development of AI scientific discovery systems is an active, fast-moving research field (Lu et al., 2026a; Yamada et al., 2025; Gottweis et al., 2026; Schmidgall et al., 2025), and such systems have begun to produce results that look like genuine, highly exciting discoveries (Wang et al., 2025; Sothanaphan, 2026; Yoon et al., 2026). This is strong evidence that LLMs can be creative, motivating their use in hypothesis generation (Wang et al., 2024; Si et al., 2025; Liu et al., 2026a; O’Neill et al., 2025) and in end-to-end discovery pipelines that generate, filter, and even execute ideas autonomously or semi-autonomously (Li et al., 2025; Baek et al., 2025; Weng et al., 2025b; Ghareeb et al., 2025; Tang et al., 2025).

![](images/b0901f35f150d1ec5efcfa30c382860f8bbd7a38612b2492689163b59452285e.jpg)  
Figure 1: A useful novelty module must catch a nonnovel copy while keeping a novel idea. We reword n = 95 published papers into 380 new-looking proposals (four disguises each) and plant the original papers in the pool. Of these disguises (Table 1), CLAIMNOV correctly rejects 367 and names the source in 366, and it rejects only 1.5% of the 400 later-realized ideas. We show ScholarEval at its default rule (best threshold in Appendix K).

One important facet of creativity, however, is novelty (Ismayilzada et al., 2025; Lu et al., 2026b). Most AI scientist pipelines employ some kind of novelty check (e.g., searching against Semantic Scholar and using an LLM judge to determine if any results are too close) which in turn decides which generated ideas will go on to receive compute and human attention. Using a novelty check only at the level of a whole proposal has three limitations. (1) Judging the novelty of a whole proposal is challenging even for human experts; the annotation itself is expensive and annotators often disagree about what counts as novelty (Si et al., 2025). (2) Deployed checks (Lu et al., 2026a; Li et al., 2025) that reject or pass a research proposal as a whole without considering its parts subtly de viate from the tradition of human science, which is incremental (Uzzi et al., 2013). (3) Whole-proposal checks also make it harder, if not impossible, to understand whether they are actually identifying the right prior work when they reject proposals. We therefore argue that the right unit to consider is the claim, not the proposal, since most papers, including ones that humans consider highly novel, are not novel in literally every contribution they make (Wang et al., 2017). With this motivation, we address the limitations of previous work and make the following contributions to the understanding and improvement of novelty checks in AI scientist pipelines:

• We introduce CLAIMNOV, a fine-grained, interpretable novelty module that treats claims as the basic unit of novelty.

• We propose an evaluation framework for novelty checks in end-to-end AI scientist pipelines which draws upon temporal ground truth from the published record and from reviewer comments at a whole-proposal and claim-level, with four datasets totaling 1,060 proposals (§4.4).

• We audit seven existing novelty checks alongside CLAIMNOV. The checks that reject the most non-novel work also reject the most work later realized by human researchers, and checks that judge proposals as a whole can only identify claim-level established work by rejecting an entire proposal.

• CLAIMNOV catches 97% of disguised papers while rejecting only 1.5% of later realized ideas. It correctly preserves partially novel work, keeping 91 of 100 proposals with a spliced prior contribution against 76 for the strongest existing check, and given the same attribution ranking, it keeps more correctly attributed proposals.

## 2 Related Work

Traditional measures of scientific novelty. Zhao and Zhang (2025) categorize scientific novelty metrics by whether they draw on citation data, textual data, or both. Bibliometric measures use unusual or first-time combinations of cited work (Uzzi et al., 2013; Wang et al., 2017; Foster et al., 2013) or related ideas such as disruptiveness (Wu et al., 2019); text-based measures consider semantic distance (Shibayama et al., 2021; Jeon et al., 2023) or new concepts and new question–method combinations (Mishra and Torvik, 2016; Luo et al., 2022); some measures also combine the two (Amplayo et al., 2018). Traditional measures are often validated against external signals such as expert judgments and later impact (Bornmann et al., 2019; Fontana et al., 2020; Arts et al., 2021). Our evaluation focuses on text-based measures, as does our introduced module CLAIMNOV, which does not consider bibliometric data.

Literature-grounded novelty assessment. Early on, IR and NLP researchers studied novelty detection at the sentence and document level, as information not covered by previously seen documents (Soboroff and Harman, 2005; Ghosal et al., 2018, 2022), and patent prior-art searches focused on naming anticipating documents (Piroi et al., 2011). LLMs have since enabled approaches that reason about the claims of existing work: retrieving prior work to revise an idea (Wang et al., 2024), judging a paper against retrieved neighbors (Lin et al., 2025), reranking retrieved papers by idea-specific facets (Shahid et al., 2025), scoring ideas against a literature set (Moussa et al., 2026; Qiao et al., 2026), and comparing atomic content units or contribution claims against full-text evidence (Ai et al., 2024; Zhang et al., 2026; Afzal et al., 2026). The typed contribution claims used in CLAIMNOV draw on scientific information extraction (Luan et al., 2018; D’Souza et al., 2021), facet-level views of ideas (Radensky et al., 2026) and claim verification (Wadden et al., 2020). Benchmarks for these LLM-based systems score them against expert annotations or review text (Schopf and Färber, 2026; Wu et al., 2026), against future papers (Amplayo et al., 2019), or under controlled, counterfactual changes (Liu and Zhai, 2026). Critiques of LLM novelty judges find that they prefer generated research questions that experts rate lower (Sinhahajari et al., 2026) and that their ratings disagree with hindsight against later papers (Jiang, 2026). We specifically evaluate novelty checks as deployed inside AI-scientist pipelines, and also focus on counterfactual changes rather than alignment with explicit human novelty labels, akin to how paraphrase attacks test text detectors (Krishna et al., 2023).

AI-scientist systems. Pipelines with a researchideation stage almost always include a novelty check for generated ideas, although these checks can vary widely in mechanism (Appendix A). Some are hard filters grounded in retrieval, including AI Scientist’s agentic search loop (Lu et al., 2026a), CoI-Agent’s reject-and-regenerate loop (Li et al., 2025), the retrieval filter of Si et al. (2025), and AutoResearchClaw’s lexical check (Liu et al., 2026b). Others are advisory or score-based, with or without explicit retrieval (Yamada et al., 2025; Baek et al., 2025; Gottweis et al., 2026; Schmidgall et al., 2025; Weng et al., 2025b,a); still others directly rank generated ideas against one another (Ghareeb et al., 2025; Su et al., 2025; Tang et al., 2025; Keya et al., 2025). Work on automatic peer review usually judges full papers, most recently with LLM judges (Kang et al., 2018; Wang et al., 2020; Taechoyotin and Acuna, 2025; Chen et al., 2025); we instead focus specifically on novelty.

## 3 Novelty Module

We introduce CLAIMNOV, an interpretable, claimlevel novelty module for research proposals that can be dropped directly into any AI for scientific discovery pipeline with a novelty check. As shown in Figure 2, CLAIMNOV extracts typed claims from research proposals, retrieves relevant samedimension claims from a pool of claims extracted from prior art, scores the entailment per extracted claim, aggregates into a novelty score, and passes the full claim profile to a verdict judge for a binary novel or non-novel decision. For validation experiments, see Appendix O, and for exact prompts, see Appendix C.

## 3.1 Claim Extraction

During the claim extraction stage, CLAIMNOV takes the title and description from the research proposal P and produces, with an LLM (gpt-4omini set to temperature 0), a list of contribution claims $C _ { P } = \{ c _ { 1 } , \ldots , c _ { m } \}$

Each claim $c _ { i }$ is typed with a dimension $d _ { c }$ and an attribute-value pair $\left( c _ { a } , c _ { v } \right)$ Dimensions are drawn from {Problem, Methodology, Resource, Result}, while an attribute represents the axis along which two works can be compared. For instance, for a Result claim, the attribute may be classification accuracy on CIFAR-10, and the value may be an improvement from 30.1% to 76.9%. We aim to extract a small number of claims per proposal (2.4– 2.7 on average, the prompt allowing 1–4), based on conventions in AI research. We process each paper in the reference pool Π through this stage once, using its abstract as the description text, to produce pool claims $C _ { \Pi }$

## 3.2 Claim Retrieval

During the claim retrieval stage, CLAIMNOV takes a claim c from the previous stage’s list of claims, embeds its text together with its paper’s title with text-embedding-3-large, and produces a set of relevant pool claims $R ( c )$ for comparison to that claim.

To obtain such a set, we retrieve the k-nearest (by cosine similarity) pool claims of the same dimension, keeping at most one claim per pool paper. We determine k by pool size, taking $k { = } 1 0$ as the minimum based on an independent retrieval validation (see Appendix O), and for larger pools we use the heuristic $k = 0 . 2 n _ { d }$ , where $n _ { d }$ is the number of pool papers with a claim in that dimension, capped at 40. This gives k=10 for the disguise pools and $k \approx 3 5$ for the anchor pools (§4.4).

## 3.3 Claim Entailment

During the claim entailment stage, CLAIMNOV judges the entailment of every pair $( c , c ^ { \prime } )$ for $c ^ { \prime } \in$ $R ( c )$ with an LLM (gpt-4o-mini at temperature 0), giving titles and abstracts of pool papers as context.

The judge rates the extent to which the pool claim entails or covers the proposal claim on an ordinal scale $\ell ( c , c ^ { \prime } ) \in \{ 1 , \ldots , 5 \}$ : 1 unrelated, 2 shared topic only, 3 partial coverage with important details differing, 4 strong coverage with minor details missing, and 5 full coverage of the claim’s substantive content. We map these levels onto the unit interval by taking $e ( c , c ^ { \prime } ) = ( \ell ( c , c ^ { \prime } ) - 1 ) / 4 \in$ [0, 1].

## 3.4 Novelty Aggregation

During the novelty aggregation stage, we first take the sorted $e _ { i } \mathbf { s }$ from the previous stage and combine

them into one coverage score $a ( c ) \in [ 0 , 1 ]$ for each claim as a rank-discounted mean:

$$
a ( c ) = \frac { \sum _ { i = 1 } ^ { k _ { c } } \alpha ^ { i - 1 } e _ { i } } { \sum _ { i = 1 } ^ { k _ { c } } \alpha ^ { i - 1 } } , ~ \nu ( c ) = 1 - a ( c ) ,\tag{1}
$$

Here, $\alpha { = } 0 . 5$ , which balances the importance of the strongest match against the weaker ones, and $k _ { c } = \operatorname* { m i n } ( k , | R ( c ) | )$ ), which prevents a claim with very few relevant pool claims from having an inflated score.

We then combine claim scores into a proposallevel score using dimension weights $w _ { d }$ (Problem 0.4, Result 0.3, Methodology 0.2, Resource 0.1) and per-claim importance weights imp(c), which are uniform for proposals and set to each claim’s share of support passages when scoring a full paper. The dimension weights encode a prior about contribution types in AI research; they can be tuned to adapt CLAIMNOV to other fields.

$$
\begin{array} { r l } & { ~ A ( P ) = \sum _ { c \in C _ { P } } w _ { d _ { c } } \operatorname* { i m p } ( c ) a ( c ) , } \\ & { A _ { \operatorname* { m a x } } ( P ) = \sum _ { c \in C _ { P } } w _ { d _ { c } } \operatorname* { i m p } ( c ) , } \\ & { ~ N ( P ) = 1 - A ( P ) / A _ { \operatorname* { m a x } } ( P ) . } \end{array}\tag{2}
$$

$A _ { \mathrm { m a x } } ( P )$ is the coverage if every claim were fully covered. $N ( P )$ summarizes the novelty profile in one number; the full profile is provided as context to the novelty verdict stage below.

## 3.5 Novelty Verdict

Finally, the novelty verdict stage of CLAIMNOV assembles previous evidence produced by the module into a profile $\Phi ( P )$ which is evaluated by a stronger judge (claude-sonnet-4-6). Φ(P) lists the proposal’s extracted claims with matched prior claims, their sources (and their abstracts), and the entailment levels.

From the judge, we receive a verdict of pass or fail. If the proposal fails, CLAIMNOV also outputs the prior art paper $p ^ { * } \in \Pi$ , its subset of claims $S \subseteq C _ { P }$ , and a one- or two-sentence reason. For the exact prompt, please see Appendix C.

## 4 Experimental Setup

We explain the shared inputs and outputs for each check, which existing checks we evaluate, the judge models and their temporal cutoffs, the test data we collect, and the metrics we define.

![](images/efb3428f32aed4e3a6b1ad3845fbb6e2dbecf3d064ec56e11b72110b7345300e.jpg)  
Figure 2: CLAIMNOV extracts typed contribution claims from a proposal P, retrieves same-dimension claims from the pool, rates the extent to which each covers $P ^ { * } { \bf s }$ claims, aggregates these into one novelty score, and gives a full claim-level novelty profile to a verdict judge.

## 4.1 Evaluation Contract

Because each of the AI scientist pipelines and standalone scientific novelty evaluations we survey (Appendix A) uses slightly different schema, literature search, and verdict formats, we standardize the evaluation contract to allow for better comparison.

We pass as input to each check a research proposal consisting of a title plus a 150–400 word description as well as a reference pool, which is a list of prior art papers given as {id, title, abstract}. Checks using search are sandboxed. For more details on how we adapt proposals to native input formats, see Table 8. For output, we consider three layers: L1, the verdict verdict $\in \{ p a s s , f a i l \}$ with reasons; L2, a novelty score $N ( P ) \in [ 0 , 1 ]$ ; and L3, a novelty profile, every claim with its matched prior claims, their entailment levels, and the papers they come from. For CLAIMNOV, L3 determines both L2 and L1 (§3.4, §3.5). The existing checks we evaluate only emit L1.

## 4.2 Evaluated Systems

We consider four kinds of novelty check. (1) Novelty checks that are a clear-cut component in some end-to-end AI-scientist pipeline: AI Scientist v1 (check\_idea\_novelty), CoI-Agent (check\_novel), Si et al. (retrieve\_novelty\_ score), and ARC’s lexical filter. (2) Novelty checks we have “extracted” from pipelines that make some kind of novelty judgment but which are not clearly plug-and-play in the same way: ResearchAgent’s Originality rating (a proposal passes at a rating of 4 or 5 out of 5), ScholarEval’s pairwise contribution scoring (a proposal passes when its mean score against the top-8 pool papers is positive), and AI-Scientist-v2’s ideation reflection. The v2 ideation reflection has no final verdict, so we append a final verdict to the prompt. (3) A claim-level reference baseline after the style of OpenNovelty (Zhang et al., 2026), which judges 1–3 extracted contributions against their BM25 top-10 and rejects when every contribution is refuted. (4) CLAIMNOV as described in §3 and its ablations. Appendix A gives our selection criteria and the full survey of 17 pipelines. Searches embedded in novelty checks are replaced by dense retrieval over the pool; see Appendix B for details.

## 4.3 Judge Models and Temporal Validity

Because six of the seven novelty checks we evaluate use an LLM as a judge at some point (and CLAIMNOV also does so), we fix a cutoff T = March 2026 and exclusively evaluate models with earlier cutoffs. This mitigates the risk of models recognizing post-T papers from training data. We run each check with Sonnet 4.6 (training cutoff January 2026), which is the same model CLAIMNOV uses as the final verdict judge (for earlier, lightweight stages, CLAIMNOV uses gpt-4omini). As we explain in more detail in §4.4, we use Sonnet 4.6 to generate ideas, disguise existing papers, and rewrite disputed papers as research proposals, and Opus 4.6 (training cutoff August 2025) as the labeler.

## 4.4 Data Collection

We collect data and establish ground truth for two levels of non-novelty. In both cases, we aim to collect data that falls after our temporal cutoff of T = March 2026, to avoid LLM judge awareness.

Already-done ideas (n=400, whole-proposal level). For this data, novelty checks should correctly reject a disguised proposal if the source paper is in the pool, but should not reject when the source is removed (making the proposal novel relative to the pool). We collect n = 100 ICML 2026 spotlight papers, filtering for papers where the v1 arXiv appeared after T, and disguise each paper as a new research proposal at four increasingly challenging to spot levels of disguise. Rung 1 swaps the voice, rung 2 paraphrases, rung 3 reframes the paper with a different motivation, reordered contributions, and an unnamed method, and rung 4 restructures it adversarially from a different theoretical starting point with generalized numbers. Because rungs 3 and 4 may depart significantly from the original paper, we accept them only when a verifier (Sonnet 4.6) confirms that the core contributions are the same. To ensure the disguise is challenging, we require that at most 2% of the disguise’s 6-grams appear in the source, regenerating up to three times; for more details and prompts, see Appendix D. We then form a reference pool from the source plus up to 19 pre-T papers closest to the proposal by embedding, which we take from the source’s reference list and Semantic Scholar recommendations. We drop five sources whose pools contain only the source, leaving 380 disguises (all 400 in Appendix K).

Claim-disputed ideas $( n { = } 1 6 0 ,$ , claim level). For this data, novelty checks should correctly find a prior work B that establishes a specific claim. We use a regex prefilter and gpt-5-nano classification to mine 8.3k ICLR 2022–23 reviews in which a reviewer identified a paper A’s contribution as already established by some prior paper B. We then rewrite each A in proposal voice with Sonnet 4.6 and the rung-1 prompt provided in Appendix D and plant B among $1 9 \ \mathrm { p r e } { \cdot } T$ papers to serve as distractors. We also record the final ICLR decision (62 accepted, 96 rejected, 2 unknown) as an additional axis of analysis of novelty check performance. Of $n = 2 0 0$ pairs of papers (A, B), we keep $n = 1 6 0$ with usable pools, and we annotate a sample for genuine disputes (see Appendix F). Unlike the other sets, these papers predate T and may appear in a judge’s training data. In this case the ground truth is reviewer attribution (and not some post-T event), so we consider this acceptable.

Later-realized ideas $( n { = } 4 0 0 ,$ , whole-proposal level). For this data, novelty checks should correctly pass the proposal, because the pool does not contain the idea and researchers went on to realize some of its contributions. We collect $n { = } 3 0$ anchor papers, where each anchor paper was published in NeurIPS 2025 proceedings and has at least 50 citing papers that fall after T. We exclude benchmarks, datasets, and surveys (which may be cited primarily for use), then generate new proposals based on the remaining anchor abstracts. We construct a relevant works pool (median 174 papers, capped at 400 papers) for each anchor from its reference list plus any citing papers that come before T, as well as a follow-up pool F of the citing papers that come after T (median 124 papers). We use each anchor paper to generate n=100 ideas from Sonnet 4.6, producing n=3000 proposals total. For each of these proposals, Opus 4.6 writes a contribution list, then labels each contribution as established or not by the 8 papers in the pool nearest to the idea (by text-embedding-3-small over title and abstract) and, separately, by the 8 nearest papers in F. We identify later-realized ideas as proposals with at least one contribution established in F and none established in the pool. From the 407 proposals of this kind generated from 26 of the 30 anchors, we draw 400 at random to constitute this dataset. We provide labeling prompts in Appendix E.

Hybrid proposals (n=100, claim level). For this data, novelty checks should correctly pass the proposal and correctly name the paper whose contribution it contains. We take n = 100 later-realized (and thus assumed novel) ideas from the previous dataset and, for each, a pool paper B which was moderately similar (ranked 5–15 by cosine similarity), then ask Sonnet 4.6 to rewrite these ideas so that B’s core contribution is also included. Sonnet 4.6 verifies that the splice preserved the original contributions while adding the new one.

## 4.5 Metrics

We evaluate based on two levels of granularity – the whole-proposal level and the claim level – motivated by the idea that we want to reject proposals that are wholly non-novel, but not proposals that simply contain some existing claims.

For the whole-proposal level, we evaluate checks’ ability to catch non-novel ideas as well as their false alarm rate, how often they reject ideas that are later realized and validated by the human researcher community. We measure false alarms both on the later-realized ideas (FA) and, as a second set of assumed novel ideas, on the disguises with their source removed from the pool (Src. out). From these, we compute informedness J = catch − FA and the pool-flip ∆ = catch − Src. out, or the drop in rejection rate when the disguised proposal’s source paper is removed from the pool.

For both whole-proposal and claim level, we evaluate checks’ ability to attribute claims to prior art. For whole-proposal, we observe whether checks name the source paper for disguises, and for claim level, we report the share of the n=160 claim-disputed proposals for which a check’s topranked paper is the prior art B. We also report rejection rates on this set by ICLR decision, with the motivation that one disputed claim does not a non-novel proposal make.

## 5 Results

We find that CLAIMNOV is both sensitive and specific, catching disguised copies of published work while keeping ideas that researchers went on to publish (§5.1). Because its verdict rests on claim-level evidence, we also find that CLAIMNOV preserves proposals in which a single contribution is established, so more correctly attributed proposals can reach the full AI-scientist pipeline (§5.2, §5.3). We report full results in Tables 1–3, with 95% intervals and paired exact McNemar tests in Appendix J.

## 5.1 Whole-Proposal Evaluation

On whole proposals (Table 1), all checks except ARC and ScholarEval catch at least 338 of 380 disguises, but most checks continue to reject after the source is removed. For instance, Si et al. catches all 380 but also rejects 359 when the source is no longer present in the pool. All checks except CLAIMNOV and CoI-Agent also reject a large share of ideas the field went on to publish. Si et al. rejects 96%, ResearchAgent 98.5%, AI Scientist v1 52%, and AI-Scientist-v2 39%. ARC and ScholarEval reject 0.0%, but do so by rejecting almost nothing.

CLAIMNOV and CoI-Agent are clear outliers in their ability to balance the two. Both catch almost every disguise (367 and 368 of 380, respectively) while rejecting few later-realized ideas (1.5% and 3.8%). CLAIMNOV has the highest J (+0.95) and ∆ (+0.92), whereas the next best pipeline checks after CoI-Agent are at +0.56 and +0.76. The per-contribution OpenNovelty-style baseline does catch more disguises (379, p<.001) but false-alarms on 11.5% of later-realized ideas and rejects 38 disguises with the source removed (p=.001 against CLAIMNOV). While clustering tests by source and anchor shows CLAIMNOV and CoI-Agent are statistically level, CLAIMNOV’s verdict call costs a quarter of CoI-Agent’s check call with the same judge (\$0.010 against \$0.039), with entailment cost growing with pool size (Table 9, Appendix M).

<table><tr><td rowspan="2">Check</td><td colspan="4">RATES</td><td colspan="2">NET SCORES</td></tr><tr><td>Caught↑</td><td>Named↑</td><td>Src. out↓</td><td>FA↓</td><td>J↑</td><td>Δ↑</td></tr><tr><td>AI Scientist v1</td><td>380</td><td>292</td><td>91</td><td>51.8%</td><td>+0.48</td><td>+0.76</td></tr><tr><td>CoI-Agent</td><td>368</td><td>362</td><td>31</td><td>3.8%</td><td>+0.93</td><td>+0.89</td></tr><tr><td>Deppoyəed Si et al.</td><td>380</td><td>379</td><td>359</td><td>96.0%</td><td>+0.04</td><td>+0.06</td></tr><tr><td>ARC</td><td>85</td><td>85</td><td>0</td><td>0.0%</td><td>+0.22</td><td>+0.22</td></tr><tr><td>ResearchAgent† Extact.</td><td>338</td><td></td><td>260</td><td>98.5%</td><td>-0.10</td><td>+0.21</td></tr><tr><td>ScholarEval†</td><td>0</td><td></td><td>0</td><td>0.0%</td><td>+0.00</td><td>+0.00</td></tr><tr><td> ${ \bf A I - S c i e n t i s t - v } 2 ^ { \dagger }$ </td><td>362</td><td></td><td>117</td><td>39.2%</td><td>+0.56</td><td>+0.64</td></tr><tr><td>OpenNovelty-style</td><td>379</td><td>379</td><td>38</td><td>11.5%</td><td>+0.88</td><td>+0.90</td></tr><tr><td>CLAIMNOV (ours)</td><td>367</td><td>366</td><td>18</td><td>1.5%</td><td>+0.95</td><td>+0.92</td></tr><tr><td>– w/o claim profile (judge over top-8 abstracts)</td><td>373</td><td>367</td><td>57</td><td>11.8%</td><td>+0.86</td><td>+0.83</td></tr><tr><td>– w/o verdict judge (graded check)</td><td>349</td><td>345</td><td>20</td><td>1.8%</td><td>+0.90</td><td>+0.87</td></tr><tr><td>– per-claim verdict rule</td><td>380</td><td>360</td><td>197</td><td>56.8%</td><td>+0.43</td><td>+0.48</td></tr><tr><td>−fixed retrieval depth (k=10)</td><td>367</td><td>366</td><td>18</td><td>2.0%</td><td>+0.95</td><td>+0.92</td></tr></table>

Table 1: Evaluation results on whole-proposal data for all checks, where every check is run with Sonnet 4.6 as its judge (excluding ARC, which is lexical). Caught and Src. out count rejections of the n = 380 disguised proposals with the source in and out of the pool; Named counts rejections that name the source; FA is on the 400 later-realized ideas. J and ∆ are defined in §4.5; see Table 17 for 95% cluster-bootstrap intervals. <sup>†</sup>Decision rule set by us (Appendix B); best in-sample thresholds are in Appendix K. For AI Scientist v1, “Named” counts any of up to three papers. Note here the fixed-depth row is equivalent to the main row because the scaled depth is also 10 on 20-paper disguise pools. Bold: best, underline: second best, ablation rows excluded.

The CLAIMNOV ablations (Appendix H) show the value of each stage of the module. Without the claim-level novelty profile, the verdict judge reads the nearest abstracts instead, and its false-alarm rate rises from 1.5% to 11.8% as a result. When we instead use a per-claim verdict rule, which fails a proposal if any central claim appears entailed, we catch all 380 disguised proposals, but the falsealarm rate rises to 56.8% and we wrongly reject 197 of the 380 proposals whose source has been removed. This reinforces the idea that (1) claims are the appropriate unit for understanding a research proposal and (2) a research proposal is not one large claim; its claims should be considered separately. On generated ideas that a whole-idea labeler matched to one pool paper, CLAIMNOV rejects 18%, against 55% for the judge without the claimlevel novelty profile (Appendix L). These ideas are noisier than disguises, and we audit a sample of them by hand, together with later-realized ideas, in Appendix F.1. Results with other verdict judges are given in Appendix K.

## 5.2 Claim-Level Prior Art

We evaluate CLAIMNOV and the seven existing checks on claim-level data (Table 2) and find that CLAIMNOV correctly treats a disputed claim as a claim-level finding, and not a reason to reject an entire research proposal. It rejects only 3% of accepted ICLR proposals, the lowest rate of any check that catches disguises (10% for CoI-Agent and 47–81% for Si et al., AI Scientist v1, and ResearchAgent). All of the whole-proposal checks we evaluate name prior art only alongside a rejection, so Si et al. names B for 47% by rejecting about 80% of proposals, and CoI-Agent passes 89%, meaning it can name B for only the remaining 11%.

Considering this design choice in existing checks, we separate attribution from the verdict by pairing each verdict with the same attribution ranking. CLAIMNOV’s top-rated profile paper is B for 21% of proposals, while the pool paper nearest to the whole proposal by dense embedding is B for 37%. Intuitively, this is because the B named by a reviewer is often closest to the proposal as a whole, which embedding similarity captures well, whereas CLAIMNOV ranks papers by how fully they cover a single extracted claim. Extraction failures further exacerbate this issue (Appendix G). We therefore pass the dense ranking to every verdict, and CLAIMNOV then passes the proposal and names B for 36% of proposals, against 31% for CoI-Agent (p=.016). Given the same attribution, CLAIMNOV’s verdict keeps more proposals building on established work.

<table><tr><td>Check</td><td>Rej. acc.↓ Rej. rej. Names B↑ on pass↑</td><td></td><td></td></tr><tr><td>AI Scientist v1</td><td>53%</td><td>40% 20%</td><td>0%</td></tr><tr><td>CoI-Agent</td><td>10%</td><td>11%</td><td>6% 0%</td></tr><tr><td>Si et al.</td><td>81%</td><td>78% 47%</td><td>0%</td></tr><tr><td>ARC</td><td>3%</td><td>0% 0%</td><td>0%</td></tr><tr><td>ResearchAgent</td><td>47%</td><td>66%</td><td></td></tr><tr><td>ScholarEval</td><td>0%</td><td>0%</td><td>一</td></tr><tr><td>AI-Sci.-v2</td><td>16%</td><td>14%</td><td></td></tr><tr><td>OpenNov.-style</td><td>8%</td><td>7%</td><td>17%</td></tr><tr><td>CLAIMNOV (ours)</td><td>3%</td><td>3%</td><td>21% 20%</td></tr><tr><td>– w/o claim profile</td><td>13%</td><td>9%</td><td>15% 7%</td></tr><tr><td>– w/o verdict judge</td><td>6%</td><td>9%</td><td>21% 20%</td></tr><tr><td>– per-claim rule</td><td>21%</td><td>20%</td><td>21% 15%</td></tr><tr><td>Verdict + shared ranking</td><td></td><td></td><td></td></tr><tr><td>CoI + dense top-1</td><td>10%</td><td>11%</td><td>37%</td></tr><tr><td>CoI + BM25 top-1</td><td>10%</td><td>11%</td><td>31% 26% 23%</td></tr><tr><td>CLAIMNoV + dense (ours)</td><td>3%</td><td>3%</td><td>37% 36%</td></tr><tr><td>CLAIMNOV + BM25 (ours)</td><td>3%</td><td>3%</td><td>26% 24%</td></tr></table>

Table 2: Claim-level prior art $( n { = } 1 6 0 ,$ , Sonnet 4.6 judge). Rej. acc./Rej. rej.: rejection rate on the 62 ICLR-accepted and 96 ICLR-rejected proposals (lower is better only for accepted proposals). Names B: share whose top-ranked paper is the reviewer’s B. On pass: same, counting only passed proposals. Note most existing checks name a paper only when rejecting. The bottom rows pair a verdict with the pool paper nearest to the whole proposal by dense embedding or BM25 to separate final verdict and attribution. — means no paper returned. Bold: best, underline: second.

## 5.3 Hybrid Proposals

Evaluating CLAIMNOV alongside the existing checks on the hybrid proposals (Table 3) allows us to more clearly distinguish CLAIMNOV’s advantages. We fix the correct behavior by construction, as each proposal is a later-realized novel idea which should not be rejected, and we control proper attribution by splicing in a contribution from a known paper B. The other checks continue to behave roughly according to established traits. ARC and ScholarEval pass all 100 and name nothing, Si et al. names B for 63 but rejects all 100, and ResearchAgent rejects 98. CoI-Agent, the closest check on whole proposals, passes 76 and names B for only 11 of the 100, all among the 24 it rejects.

CLAIMNOV, on the other hand, passes 91 of 100 (p=.004 against CoI-Agent). Because a hybrid states B’s contribution explicitly, BM25 over the whole proposal ranks B first for 74. Given this shared ranking (similarly to the isolation we perform in §5.2), CLAIMNOV keeps 67 hybrids with B named, against 53 for CoI-Agent (p=.004). The OpenNovelty-style baseline keeps 64, but this is at the cost of 11.5% false alarms on later-realized ideas (Table 1). When CLAIMNOV successfully extracts the spliced claim as its own claim (as opposed to missing it or absorbing it into a different claim), the entailment judge rates B at level 4 or 5, and the verdict stage properly reads one established claim among several novel ones as a proposal that builds on prior work. We consider these ratings reliable, as 13 of the 15 papers rated at level 4 or above are B, and this entailment level is reached by only 9 of 400 later-realized ideas (Appendix J).

<table><tr><td>Check</td><td></td><td>Passes↑ Names B↑</td><td>on pass↑</td></tr><tr><td>AI Scientist v1</td><td>18</td><td>23</td><td>0</td></tr><tr><td>CoI-Agent</td><td>76</td><td>11</td><td>0</td></tr><tr><td>Si et al.</td><td>0</td><td>63</td><td>0</td></tr><tr><td>ARC</td><td>100</td><td>0</td><td>0</td></tr><tr><td>ResearchAgent</td><td>2</td><td>一</td><td></td></tr><tr><td>ScholarEval</td><td>100</td><td></td><td></td></tr><tr><td>AI-Scientist-v2</td><td>29</td><td></td><td></td></tr><tr><td>OpenNov.-style</td><td>83</td><td>68</td><td>64</td></tr><tr><td>CLAIMNOV (ours)</td><td>91</td><td>25</td><td>18</td></tr><tr><td>– w/o claim profile</td><td>68</td><td>42</td><td>14</td></tr><tr><td>Verdict + shared ranking</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { C o I + d e n s e  t o p { - } 1 }$ </td><td>76</td><td>7</td><td>6</td></tr><tr><td> $\mathbf { C o I + B M } 2 5 \mathrm { t o p - 1 }$ </td><td>76</td><td>74</td><td>53</td></tr><tr><td>CLAIMNOV + dense (ours)</td><td>91</td><td>7</td><td>7</td></tr><tr><td> $\mathbf { C L A I M N O V } + \mathbf { B M } 2 5 \mathbf { \mathrm { ( o u r s ) } }$ </td><td>91</td><td>74</td><td>67</td></tr></table>

Table 3: Hybrid proposals (n=100, Sonnet 4.6 judge). Names B: the top-ranked paper is the spliced paper B (as in Table 2). On pass: B is named and the proposal passes. —: no paper returned. Bold: best, underline: second.

## 6 Conclusion

We establish an evaluation framework with temporal ground truth at both the proposal and the claim level, and use it to audit seven existing novelty checks. We find that the checks which reject the most non-novel work also tend to reject the most work the field went on to publish, and observe that checks built to judge proposals as a whole name claim-level prior art only when they reject. We therefore introduce CLAIMNOV, a drop-in claimlevel novelty module, and show it rejects non-novel ideas, keeps novel and partially novel ones, and, given the same attribution ranking, keeps more correctly attributed proposals. Beyond filtering, a specific claim-level check like CLAIMNOV could serve as a reward when training AI-scientist agents, since it does not penalize the incremental followups most science consists of, and its profile tells the agent which of its contributions are new. Do AI scientists know what’s new? Based on current novelty checks, they largely do not, rejecting much of what the field goes on to publish or practically nothing at all. However, CLAIMNOV enables scientific novelty to be judged claim by claim against specific prior papers, thus potentially making AI scientists much better judges of novelty.

## Limitations

Our work employs several heuristics in how we consider sets of proposals or papers to be novel or non-novel, and this is not as clean as if we had a “true” dataset. We also focus exclusively on AI research. In practice, this is where most AI-scientist work also concentrates, so this aligns well, but if we want AI-scientist pipelines to be more widely applicable, then assumptions like the ones we make about dimension weights (problem contributions being more valuable than other contributions, for instance) might not generalize and would require further tuning for different fields, thus impeding adoption. We also consider only English-language research, which can further skew generalization.

Because our pools are fixed lists of prior work rather than open search, consisting of 20 papers for disguises and disputed pairs and up to 400 for generated ideas, we have limited ability to evaluate the retrieval of novelty checks. Additionally, we identify later-realized ideas with an LLM labeler, although we do cross-check this with a different model family labeler and with a small-scale human audit in Appendix F.1. Our disguised proposals and generated ideas are generated by the same model (Sonnet 4.6) that ultimately judges, which is a known source of possible bias for LLMsas-judges (Zheng et al., 2023; Panickssery et al., 2024).

The temporal construction of our datasets also decays, since running the same analysis with more recent LLM judges (with training cutoffs post-T) will require all materials to be rebuilt at a later T. Although this is simple mechanically and our scripts support it, this still takes time and money. Finally, one important aspect of novelty modules in AI-scientist pipelines is how they affect in-the-loop generation: when the module is actively rejecting and causing regeneration of proposals, how does that affect the final committed set of ideas, and what are the compute and token costs involved? We leave this question to future work because (1) it is highly computationally expensive compared to the static question we examine and (2) establishing ground truth for ideas produced in response to a rejection is harder still and may require extensive human annotation, for which we have limited resources. We do run a small pilot (Appendix M) and report the rejection and cost dynamics of the loop.

## Ethical Considerations

Our aim with this work is to encourage the development of scientific novelty modules in AI scientist pipelines which are better aligned with human understandings of novelty, which we believe is important to the science produced by these pipelines. We recognize that development of AI for scientific discovery capabilities has several concerning implications for current review processes and for the human experience of science in general. CLAIM-NOV, like any automated judge, should be used carefully.

We do not release any intellectual property directly; we release no direct paper text, only pointers and code for reproduction of our results, and we do not release ICLR review text, but rather the code that re-mines the claim disputes from Open-Review. We follow the terms of use of both arXiv and OpenReview licenses. Additionally, while we use human annotation to validate some LLM labels, the annotation consists only of labels on published papers and generated ideas, and we collect no survey responses or personal information. It is not human-subjects research and falls outside IRB review.

## Acknowledgments

We thank Lucy Zhou for her work on an earlier version of this project. Miri Liu was supported by the Amazon AI PhD Fellowship and by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship under Award Number DE-SC0026617. This work is supported in part by the National Science Foundation under Grants 2229612 and 2433308. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the United States Government or any agency thereof.

## References

Osama Mohammed Afzal, Preslav Nakov, Tom Hope, and Iryna Gurevych. 2026. Beyond "not novel enough": Enriching scholarly critique with llmassisted feedback. Preprint, arXiv:2508.10795.

Lin Ai, Ziwei Gong, Harshsaiprasad Deshpande, Alexander Johnson, Emmy Phung, Ahmad Emami, and Julia Hirschberg. 2024. Novascore: A new automated metric for evaluating document level novelty. Preprint, arXiv:2409.09249.

Reinald Kim Amplayo, SuLyn Hong, and Min Song. 2018. Network-based approach to detect novelty of scholarly literature. Information Sciences, 422:542– 557.

Reinald Kim Amplayo, Seung-won Hwang, and Min Song. 2019. Evaluating research novelty detection: Counterfactual approaches. In Proceedings of the Thirteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-13), pages 124–133, Hong Kong. Association for Computational Linguistics.

Sam Arts, Jianan Hou, and Juan Carlos Gomez. 2021. Natural language processing to identify the creation and impact of new technologies in patent text: Code, data, and new measures. Research Policy, 50(2):104144.

Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. 2025. ResearchAgent: Iterative research idea generation over scientific literature with large language models. In Proceedings ofthe 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6709–6738.

Lutz Bornmann, Alexander Tekles, Helena H. Zhang, and Fred Y. Ye. 2019. Do we measure novelty when we analyze unusual combinations of cited references? a validation study of bibliometric novelty indicators based on F1000Prime data. Journal ofInformetrics, 13(4):100979.

Shiping Chen, Duncan Brumby, and Anna Cox. 2025. Envisioning the future of peer review: Investigating llm-assisted reviewing using chatgpt as a case study. In Proceedings of the 4th Annual Symposium on Human-Computer Interactionfor Work, CHIWORK ’25, New York, NY, USA. Association for Computing Machinery.

Jennifer D’Souza, Sören Auer, and Ted Pedersen. 2021. Semeval-2021 task 11: Nlpcontributiongraph - structuring scholarly nlp contributions for a research knowledge graph. In Proceedings ofthe 15th International Workshop on Semantic Evaluation (SemEval 2021), page 364–376. Association for Computational Linguistics.

Magda Fontana, Martina Iori, Fabio Montobbio, and Roberta Sinatra. 2020. New and atypical combinations: An assessment of novelty and interdisciplinarity. Research Policy, 49(7):104063.

Jacob G. Foster, Andrey Rzhetsky, and James A. Evans. 2013. Tradition and innovation in scientists’ research strategies. Preprint, arXiv:1302.6906.

Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Jon M. Laurent, Muhammed T. Razzak, Andrew D. White, Michaela M. Hinks, and Samuel G. Rodriques. 2025. Robin: A multi-agent system for automating scientific discovery. Preprint, arXiv:2505.13400.

Tirthankar Ghosal, Vignesh Edithal, Asif Ekbal, Pushpak Bhattacharyya, George Tsatsaronis, and Srinivasa Satya Sameer Kumar Chivukula. 2018. Novelty goes deep. a deep neural solution to document level novelty detection. In Proceedings ofthe 27th International Conference on Computational Linguistics, pages 2802–2813. Association for Computational Linguistics.

Tirthankar Ghosal, Tanik Saikh, Tameesh Biswas, Asif Ekbal, and Pushpak Bhattacharyya. 2022. Novelty detection: A perspective from natural language processing. Computational Linguistics, 48(1):77–117.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, Anil Palepu, Keran Rong, Ryutaro Tanno, Khaled Saab, Fan Zhang, Jacob Blum, Andrew Carroll, Kavita Kulkarni, Nenad Tomašev, and 32 others. 2026. Accelerating scientific discovery with co-scientist. Nature, 655(8122):487–496.

Mete Ismayilzada, Antonio Laverghetta Jr., Simone A. Luchini, Reet Patel, Antoine Bosselut, Lonneke van der Plas, and Roger Beaty. 2025. Creative preference optimization. Preprint, arXiv:2505.14442.

Daeseong Jeon, Junyoup Lee, Joon Mo Ahn, and Changyong Lee. 2023. Measuring the novelty of scientific publications: A fastText and local outlier factor approach. Journal ofInformetrics, 17(4):101450.

Bo Jiang. 2026. Hindsight: Evaluating llm-generated research ideas via future impact. Preprint, arXiv:2603.15164.

Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy, and Roy Schwartz. 2018. A dataset of peer reviews (peerread): Collection, insights and nlp applications. Preprint, arXiv:1804.09635.

Farhana Keya, Gollam Rabby, Prasenjit Mitra, Sahar Vahdati, Sören Auer, and Yaser Jaradeh. 2025. Sci-idea: Context-aware scientific ideation using token and sentence embeddings. Preprint, arXiv:2503.19257.

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. 2023. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Preprint, arXiv:2303.13408.

Long Li, Weiwen Xu, Jiayan Guo, Ruochen Zhao, Xingxuan Li, Yuqian Yuan, Boqiang Zhang, Yuming Jiang, Yifei Xin, Ronghao Dang, Yu Rong, Deli Zhao, Tian Feng, and Lidong Bing. 2025. Chain of ideas: Revolutionizing research via novel idea development

with LLM agents. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 8971–9004.

Ethan Lin, Zhiyuan Peng, and Yi Fang. 2025. Evaluating and enhancing large language models for novelty assessment in scholarly publications. In Proceedings ofthe 1st Workshop on AI and Scientific Discovery: Directions and Opportunities, pages 46–57.

Haokun Liu, Sicong Huang, Jingyu Hu, Yangqiaoyu Zhou, and Chenhao Tan. 2026a. Hypobench: Towards systematic and principled benchmarking for hypothesis generation. Preprint, arXiv:2504.11524.

Jiaqi Liu, Shi Qiu, Mairui Li, Bingzhou Li, Haonian Ji, Siwei Han, Xinyu Ye, Peng Xia, Zihan Dong, Meng Chen, Congyu Zhang, Letian Zhang, Guiming Chen, Haoqin Tu, Xinyu Yang, Lu Feng, Xujiang Zhao, Haifeng Chen, Jiawei Zhou, and 17 others. 2026b. Autoresearchclaw: Self-reinforcing autonomous research with human-ai collaboration. Preprint, arXiv:2605.20025.

Miri Liu and ChengXiang Zhai. 2026. An axiomatic benchmark for evaluation of scientific novelty metrics. Preprint, arXiv:2604.15145.

Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jeff Clune. 2026a. Towards end-to-end automation of AI research. Nature, 651:914–919.

Li-Chun Lu, Miri Liu, Pin Chun Lu, Yufei Tian, Shao-Hua Sun, and Nanyun Peng. 2026b. Rethinking creativity evaluation: A critical analysis of existing creativity evaluations. In Proceedings of the 19th Conference ofthe European Chapter ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 6329–6352.

Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi. 2018. Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction. Preprint, arXiv:1808.09602.

Zhuoran Luo, Wei Lu, Jiangen He, and Yuqi Wang. 2022. Combination of research questions and methods: A new measurement of scientific novelty. Journal ofInformetrics, 16(2):101282.

Shubhanshu Mishra and Vetle I. Torvik. 2016. Quantifying conceptual novelty in the biomedical literature. D-Lib magazine : the magazine ofthe Digital Library Forum, 22.

Hanane Nour Moussa, Patrick Queiroz Da Silva, Daniel Adu-Ampratwum, Alyson East, Zitong Lu, Nikki Puccetti, Mingyi Xue, Huan Sun, Bodhisattwa Prasad Majumder, and Sachin Kumar. 2026. Scholareval: Research idea evaluation grounded in literature. Preprint, arXiv:2510.16234.

Charles O’Neill, Tirthankar Ghosal, Roberta Raileanu,˘ Mike Walmsley, Thang Bui, Kevin Schawinski, and

Ioana Ciuca. 2025.˘ Sparks of science: Hypothesis generation using structured paper data. Preprint, arXiv:2504.12976.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. Llm evaluators recognize and favor their own generations. Preprint, arXiv:2404.13076.

Florina Piroi, Mihai Lupu, Allan Hanbury, and Veronika Zenz. 2011. Clef-ip 2011: Retrieval in the intellectual property domain. In Conference and Labs ofthe Evaluation Forum.

Shuofei Qiao, Yunxiang Wei, Xuehai Wang, Bin Wu, Boyang Xue, Ningyu Zhang, Hossein A. Rahmani, Yanshan Wang, Qiang Zhang, Keyan Ding, Jeff Z. Pan, Huajun Chen, and Emine Yilmaz. 2026. Innoeval: On research idea evaluation as a knowledge-grounded, multi-perspective reasoning problem. Preprint, arXiv:2602.14367.

Marissa Radensky, Simra Shahid, Raymond Fok, Pao Siangliulue, Tom Hope, and Daniel S. Weld. 2026. Human-llm compound system for scientific ideation through facet recombination and novelty evaluation. Preprint, arXiv:2409.14634.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. 2025. Agent laboratory: Using llm agents as research assistants. Preprint, arXiv:2501.04227.

Tim Schopf and Michael Färber. 2026. Is this idea novel? an automated benchmark for judgment of research ideas. In Proceedings ofthe Fifteenth Language Resources and Evaluation Conference, pages 4716–4727.

Simra Shahid, Marissa Radensky, Raymond Fok, Pao Siangliulue, Daniel S. Weld, and Tom Hope. 2025. Literature-grounded novelty assessment of scientific ideas. In Proceedings ofthe Fifth Workshop on Scholarly Document Processing, pages 96–113.

Sotaro Shibayama, Deyun Yin, and Kuniko Matsumoto. 2021. Measuring novelty in science with word embedding. PLOS ONE, 16(7):e0254034.

Chenglei Si, Diyi Yang, and Tatsunori Hashimoto. 2025. Can LLMs generate novel research ideas? a largescale human study with 100+ NLP researchers. In International Conference on Learning Representations.

Soumitra Sinhahajari, Navonil Majumder, and Soujanya Poria. 2026. On the limits of llm-asjudge for scientific novelty assessment. Preprint, arXiv:2606.12071.

Ian Soboroff and Donna Harman. 2005. Novelty detection: The TREC experience. In Proceedings of Human Language Technology Conference and Conference on Empirical Methods in Natural Language Processing, pages 105–112, Vancouver, British Columbia, Canada. Association for Computational Linguistics.

Nat Sothanaphan. 2026. Resolution of erdos problem˝ #728: a writeup of aristotle’s lean proof. Preprint, arXiv:2601.07421.

Haoyang Su, Renqi Chen, Shixiang Tang, Zhenfei Yin, Xinzhe Zheng, Jinzhe Li, Biqing Qi, Qi Wu, Hui Li, Wanli Ouyang, Philip Torr, Bowen Zhou, and Nanqing Dong. 2025. Many heads are better than one: Improved scientific idea generation by a llm-based multi-agent system. Preprint, arXiv:2410.09403.

Pawin Taechoyotin and Daniel Acuna. 2025. Remor: Automated peer review generation with llm reasoning and multi-objective reinforcement learning. Preprint, arXiv:2505.11718.

Jiabin Tang, Lianghao Xia, Zhonghang Li, and Chao Huang. 2025. Ai-researcher: Autonomous scientific innovation. Preprint, arXiv:2505.18705.

Brian Uzzi, Satyam Mukherjee, Michael Stringer, and Ben Jones. 2013. Atypical combinations and scientific impact. Science, 342(6157):468–472.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. Preprint, arXiv:2004.14974.

Jian Wang, Reinhilde Veugelers, and Paula Stephan. 2017. Bias against novelty in science: A cautionary tale for users of bibliometric indicators. Research Policy, 46(8):1416–1436.

Qingyun Wang, Doug Downey, Heng Ji, and Tom Hope. 2024. SciMON: Scientific inspiration machines optimized for novelty. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 279–299.

Qingyun Wang, Qi Zeng, Lifu Huang, Kevin Knight, Heng Ji, and Nazneen Fatema Rajani. 2020. ReviewRobot: Explainable paper review generation based on knowledge synthesis. In Proceedings of the 13th International Conference on Natural Language Generation, pages 384–397, Dublin, Ireland. Association for Computational Linguistics.

Yongji Wang, Mehdi Bennani, James Martens, Sébastien Racanière, Sam Blackwell, Alex Matthews, Stanislav Nikolov, Gonzalo Cao-Labora, Daniel S. Park, Martin Arjovsky, Daniel Worrall, Chongli Qin, Ferran Alet, Borislav Kozlovskii, Nenad Tomašev, Alex Davies, Pushmeet Kohli, Tristan Buckmaster, Bogdan Georgiev, and 3 others. 2025. Discovery of unstable singularities. Preprint, arXiv:2509.14185.

Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, and Linyi Yang. 2025a. Cycleresearcher: Improving automated research via automated review. Preprint, arXiv:2411.00816.

Yixuan Weng, Minjun Zhu, Qiujie Xie, Qiyao Sun, Zhen Lin, Sifan Liu, and Yue Zhang. 2025b. Deepscientist: Advancing frontier-pushing scientific findings progressively. Preprint, arXiv:2509.26603.

Lingfei Wu, Dashun Wang, and James A Evans. 2019. Large teams develop and small teams disrupt science and technology. Nature, 566:378–382.

Wenqing Wu, Yi Zhao, Yuzhuo Wang, Siyou Li, Juexi Shao, Yunfei Long, and Chengzhi Zhang. 2026. NovBench: Evaluating large language models on academic paper novelty assessment. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 32103–32133.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. 2025. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. Preprint, arXiv:2504.08066.

Peter H Yoon, Januka S Athukoralage, Emmanuel Ameisen, Eric Kauderer-Abrams, Nicholas T Perry, and Matthew G Durrant. 2026. Autonomous ai agents discover reverse transcriptases with tandem repeat arrays. bioRxiv.

Ming Zhang, Kexin Tan, Yueyuan Huang, Yujiong Shen, Chunchun Ma, Li Ju, Xinran Zhang, Yuhui Wang, Wenqing Jing, Jingyi Deng, Huayu Sha, Binze Hu, Jingqi Tong, Changhao Jiang, Yage Geng, Yuankai Ying, Yue Zhang, Zhangyue Yin, Zhiheng Xi, and 4 others. 2026. OpenNovelty: An LLM-powered agentic system for verifiable scholarly novelty assessment. Preprint, arXiv:2601.01576.

Yi Zhao and Chengzhi Zhang. 2025. A review on the novelty measurements of academic papers. Scientometrics, 130:727–753.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36.

## A Pipeline Survey

Table 4 surveys the novelty judgments embedded in AI-scientist pipelines, and Table 5 lists standalone novelty judges. Table 6 shows what each pipeline passes downstream after its check, which shapes the evaluation contract (§4.1). Table 7 gives the systems we include and exclude. We include a system when (1) its code is available, (2) it has a separable judgment call, (3) that judgment takes a research proposal as input, (4) we can control the reference pool or sandbox its retrieval, and (5) its judge can be replaced with one whose training cutoff predates T. Table 8 shows how the shared proposal is passed to each included check, and Table 9 gives the measured cost of the final judge calls.

<table><tr><td rowspan=1 colspan=11>System        Where judgment sits  Input artifact I          Pool at judgment time    Output O and enforcement  Pool controllable?</td></tr><tr><td rowspan=1 colspan=9>AI Scientist v1  Hard post-ideation filterIdea dict: Name, Title,     Agentic loop, &lt;10 rounds(Lu et al., 2026a)(check_idea_novelty)Experiment (2–4 sentences),× 10 Semantic Scholar hits</td><td rowspan=35 colspan=2>novel: bool parsed from a    Yes: oneliteral “Decision made”;      search_for_papers();replace check_idea_novelty wholesaleuse_tool(query) isthe single chokepointYes: check_novel hasa clean signaturefed back to regeneration;Yes: retrieve_novelty_score, or“same problem and same     pre-populate the paperlistevidence attachedYes (fully): static pool;feedback; steers refinement andswap the originalitypromptYes: ArxivSearchswappable; insert aphasepre-experimentNo code releasedNo: monolithic</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1></td><td rowspan=1 colspan=3>1–10 self-ratings; task</td><td rowspan=1 colspan=4>(title, abstract, venue, year,</td><td rowspan=1 colspan=1>non-novel ideas discarded; no</td></tr><tr><td rowspan=1 colspan=3>description and template</td><td rowspan=1 colspan=4>citations)</td><td rowspan=1 colspan=1>decision after 10 rounds ⇒ not</td></tr><tr><td rowspan=1 colspan=3>code</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>AI Scientist v2</td><td rowspan=1 colspan=1>No separable check;</td><td rowspan=1 colspan=3>Search-query strings; idea</td><td rowspan=1 colspan=4>Same endpoint, top-10,</td><td rowspan=1 colspan=1>Nothing machine-readable;</td><td rowspan=1 colspan=1>Yes:</td></tr><tr><td rowspan=1 colspan=1>(Yamada et al.,</td><td rowspan=1 colspan=1>advisory retrieval inside</td><td rowspan=1 colspan=3>JSON (Name, Title, Short</td><td rowspan=1 colspan=4>citation-sorted</td><td rowspan=1 colspan=1>lands as Related Work prose; a</td></tr><tr><td rowspan=1 colspan=1>2025)</td><td rowspan=1 colspan=1>ideation reflections</td><td rowspan=1 colspan=3>Hypothesis, Related Work,</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>human picks the idea to run</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Abstract ≈250w,Experiments)</td><td rowspan=1 colspan=4></td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=2 colspan=1>CoI-Agent (Li</td><td rowspan=2 colspan=1>Hard filter plus</td><td rowspan=2 colspan=3>One free-text blob</td><td rowspan=3 colspan=4>&lt;3 LLM queries × 5 hits,title+abstract; date cutoff</td></tr><tr><td rowspan=2 colspan=1>offending paper and summa</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>et al., 2025)</td><td rowspan=1 colspan=1>uncapped regeneration</td><td rowspan=1 colspan=3>(motivation, novelty,</td><td rowspan=1 colspan=1>title+abstract; date</td><td rowspan=1 colspan=1>ary.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>loop</td><td rowspan=1 colspan=3>method merged; structure</td><td rowspan=1 colspan=2>and ban list built in</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>fed back to regeneration;</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=4 colspan=1>Si et al. (Si et al.,</td><td rowspan=3 colspan=1></td><td rowspan=1 colspan=3>dissolved before the check)</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>criterion near</td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>plagiarism-strength identity;</td></tr><tr><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>empty retrieval ⇒ novel</td></tr><tr><td rowspan=1 colspan=1>(a) intra-batch</td><td rowspan=1 colspan=3>(a) title + 5 fields; (c) full</td><td rowspan=1 colspan=4>(a) own ideas only,</td><td rowspan=1 colspan=1>(a) drop duplicates (4000 →</td></tr><tr><td rowspan=1 colspan=1>2025)</td><td rowspan=1 colspan=1>embedding dedup; (b)</td><td rowspan=1 colspan=3>proposal JSON (Problem,</td><td rowspan=1 colspan=4>MiniLM cosine &gt; 0.8; (c)</td><td rowspan=1 colspan=1>≈200); (c) per-paper binary</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>Swiss tournament,</td><td rowspan=1 colspan=3>Existing Methods,</td><td rowspan=1 colspan=4>top-10 abstracts via LLM</td><td rowspan=1 colspan=1>“same problem and same</td></tr><tr><td rowspan=1 colspan=1>novelty excluded; (c)</td><td rowspan=1 colspan=3>Motivation, Proposed</td><td rowspan=1 colspan=4>keyword queries</td><td rowspan=1 colspan=1>approach?"; any yes rejects,</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>retrieval filter, the real</td><td rowspan=1 colspan=3>Method, Experiment Plan;</td><td rowspan=1 colspan=4></td><td rowspan=3 colspan=1>Origialty 1- Liker plus</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>check</td><td rowspan=1 colspan=3>1–2 pages)</td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=1>ResearchAgent</td><td rowspan=1 colspan=1>Soft critic in generate</td><td rowspan=1 colspan=3>Artifact plus rationale;</td><td rowspan=1 colspan=4>No retrieval at review time;</td></tr><tr><td rowspan=1 colspan=1>(Baek et al.,</td><td rowspan=1 colspan=1>→ validate → refine</td><td rowspan=1 colspan=3>grounding context (core</td><td rowspan=1 colspan=4>frozen jsonl pool from</td><td rowspan=1 colspan=1>feedback; steers refinement and</td></tr><tr><td rowspan=1 colspan=1>2025)</td><td rowspan=1 colspan=1>×3, per artifact</td><td rowspan=1 colspan=3>paper, references without</td><td rowspan=1 colspan=4>generation</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>(problem, method,</td><td rowspan=1 colspan=3>abstracts, entity store)</td><td rowspan=2 colspan=4></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>experiment)</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>No check; lit review</td><td rowspan=1 colspan=3>Topic string (review); full</td><td rowspan=1 colspan=4>arXiv top-5 (review);</td><td rowspan=1 colspan=1>Originality 1–4 inside a</td></tr><tr><td rowspan=1 colspan=1>Laboratory</td><td rowspan=1 colspan=1>judges relevance;</td><td rowspan=2 colspan=3>LaTeX (reviewer)</td><td rowspan=1 colspan=4>parametric (reviewer)</td><td rowspan=1 colspan=1>NeurIPS-style review JSON;</td></tr><tr><td rowspan=1 colspan=1>(Schmidgall</td><td rowspan=1 colspan=1>post-hoc reviewer scores</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>revision feedback and solver</td></tr><tr><td rowspan=1 colspan=1>et al., 2025)</td><td rowspan=1 colspan=1>the finished paper</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>reward; never checks</td></tr><tr><td rowspan=2 colspan=1>SCI-IDEA</td><td rowspan=2 colspan=1>Mid-pipeline “Aha”</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3>1–2 sentence idea,</td><td rowspan=1 colspan=4>None; novelty = 1 − max</td><td rowspan=1 colspan=1>Continuous scores → binary</td></tr><tr><td rowspan=1 colspan=1>(Keya et al.,</td><td rowspan=1 colspan=1>filter</td><td rowspan=1 colspan=3>SciBERT-embedded</td><td rowspan=1 colspan=4>cosine to sibling candidates,</td><td rowspan=1 colspan=1>flag; only flagged ideas refined</td></tr><tr><td rowspan=1 colspan=1>2025)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>surprise = − log p</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CycleResearcher</td><td rowspan=1 colspan=1>Post-hoc whole-paper</td><td rowspan=1 colspan=3>Full paper (≈14k tokens)</td><td rowspan=1 colspan=4>Parametric (fine-tuned on</td><td rowspan=1 colspan=1>Review dict (ratings,</td></tr><tr><td rowspan=1 colspan=1>(Weng et al.,</td><td rowspan=1 colspan=1>review; reward for RL</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>ICLR reviews)</td><td rowspan=1 colspan=1>contribution, decision); no</td><td rowspan=1 colspan=1>fine-tune: a module</td></tr><tr><td rowspan=1 colspan=1>2025a)</td><td rowspan=1 colspan=1>loop</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>novelty field</td><td rowspan=1 colspan=1>must sit upstream</td></tr><tr><td rowspan=1 colspan=1>DeepScientist</td><td rowspan=1 colspan=1>Valuation against own</td><td rowspan=1 colspan=3>Hypothesis record; repo:</td><td rowspan=1 colspan=4>Own findings archive:</td><td rowspan=1 colspan=1>(utility, quality, exploration)</td><td rowspan=1 colspan=1>Yes (repo): editable</td></tr><tr><td rowspan=1 colspan=1>(Weng et al.,</td><td rowspan=1 colspan=1>findings memory</td><td rowspan=1 colspan=3>draft template (claim,</td><td rowspan=1 colspan=4>literature via mandated</td><td rowspan=1 colspan=1>0–100 drives compute (paper);</td><td rowspan=2 colspan=1>markdown check;paper&#x27;s scoring code</td></tr><tr><td rowspan=1 colspan=1>2025b)</td><td rowspan=1 colspan=1>(paper); prompt-file</td><td rowspan=1 colspan=3>closest prior work, novelty</td><td rowspan=1 colspan=4>sweep (≥5 closest papers)</td><td rowspan=1 colspan=1>0/1/2 novelty rubric, &lt; 7</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>check (repo)</td><td rowspan=1 colspan=3>type)</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>blocks promotion (repo)</td><td rowspan=1 colspan=1>not public</td></tr><tr><td rowspan=1 colspan=1>Robin (Ghareeb</td><td rowspan=1 colspan=1>Novelty is one of five</td><td rowspan=1 colspan=3>Hypothesis name plus</td><td rowspan=1 colspan=4>Literature baked into the</td><td rowspan=1 colspan=1>Pairwise winner →</td><td rowspan=1 colspan=1>Indirect: pool fixed</td></tr><tr><td rowspan=1 colspan=1>et al., 2025)</td><td rowspan=1 colspan=1>criteria in pairwise</td><td rowspan=1 colspan=3>800–1500w literature report;</td><td rowspan=1 colspan=4>report upstream; parametric</td><td rowspan=1 colspan=1>Bradley-Terry strength; nothing</td><td rowspan=4 colspan=1>before the judge; swapthe pair comparisonYes: per-task reference</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ranking, after</td><td rowspan=1 colspan=3>judged in pairs</td><td rowspan=1 colspan=4>at judge time</td><td rowspan=1 colspan=1>discarded; orders for human</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>per-candidate lit reports</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>wet-lab pick</td></tr><tr><td rowspan=1 colspan=1>AI-Researcher</td><td rowspan=1 colspan=1>“Pick the most novel of</td><td rowspan=1 colspan=3>Five markdown ideas (1–3k</td><td rowspan=1 colspan=4>Seeded reference papers</td><td rowspan=1 colspan=1>Free text: one selected idea</td></tr><tr><td rowspan=1 colspan=1>(Tang et al.,</td><td rowspan=1 colspan=1>5”; no similarity</td><td rowspan=1 colspan=3>words) in one prompt</td><td rowspan=1 colspan=4>and repos on disk: search</td><td rowspan=1 colspan=1>report; the other four silently</td><td rowspan=2 colspan=1>list; replace theselection call</td></tr><tr><td rowspan=1 colspan=1>2025)</td><td rowspan=1 colspan=1>computed</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>tools disabled</td><td rowspan=1 colspan=1>dropped; no score persisted</td></tr><tr><td rowspan=1 colspan=1>ARC (Liu et al.,</td><td rowspan=1 colspan=1>Stage-08 lexical check,</td><td rowspan=1 colspan=3>Topic plus hypotheses. md</td><td rowspan=1 colspan=4>Live search (≤5 queries ×</td><td rowspan=1 colspan=1>Keyword-Jaccard score 0–1,</td><td rowspan=1 colspan=1>Yes: pass</td></tr><tr><td rowspan=1 colspan=1>2026b)</td><td rowspan=1 colspan=1>explicitly non-blocking</td><td rowspan=1 colspan=3>(H1–H4, 300–1200w) plus</td><td rowspan=1 colspan=4>15; OpenAlex, S2, arXiv)</td><td rowspan=1 colspan=1>band, recommendation in</td><td rowspan=3 colspan=1>papers_already_seenor shadowcheck_novelty()</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=3>seen-paper rows</td><td rowspan=1 colspan=4> Stage-04 candidates;</td><td rowspan=1 colspan=1>{proceed, differentiate, abort};</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>titles+abstracts</td><td rowspan=1 colspan=1>advisory to a human</td></tr><tr><td rowspan=1 colspan=1>SciMON (Wang</td><td rowspan=1 colspan=1>Retrieve → compare →</td><td rowspan=1 colspan=3>Idea plus background and</td><td rowspan=1 colspan=4>Top-20 idea sentences from</td><td rowspan=1 colspan=1>LLM rewrites the idea until</td><td rowspan=16 colspan=1>Yes: all modulessimilarity falls below a       explicitthresholdYes: separate stages;new papers injectablemay override               before rerankingYes: pool replaceablePartly: provided poolreplaceable, web searchis not fixedwork as the comparison</td></tr><tr><td rowspan=1 colspan=1>et al., 2024)</td><td rowspan=1 colspan=1>rewrite loop after initial</td><td rowspan=1 colspan=3>problem context</td><td rowspan=1 colspan=4>the training set</td><td rowspan=1 colspan=1>similarity falls below a</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>generation</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>(SentenceBERT)</td><td rowspan=1 colspan=1>threshold</td></tr><tr><td rowspan=1 colspan=1>Idea Novelty</td><td rowspan=1 colspan=1>Generate → check →</td><td rowspan=1 colspan=3>Idea decomposed into</td><td rowspan=1 colspan=4>Top-100</td><td rowspan=1 colspan=1>Binary label plus explanation</td></tr><tr><td rowspan=1 colspan=1>Checker (Shahid</td><td rowspan=1 colspan=1>replace facet →</td><td rowspan=1 colspan=3>domain, purpose,</td><td rowspan=1 colspan=4>embedding-filtered papers</td><td rowspan=1 colspan=1>and alternative facet; human</td></tr><tr><td rowspan=1 colspan=1>et al., 2025)</td><td rowspan=1 colspan=1>regenerate</td><td rowspan=1 colspan=3>mechanism, evaluation</td><td rowspan=1 colspan=4>(ideation, semantic search,</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=3 colspan=1>VirSci (Su et al.,</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=3>facets</td><td rowspan=1 colspan=4>related work), RankGPT</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>top-10</td></tr><tr><td rowspan=1 colspan=1>Relative-novelty voting</td><td rowspan=1 colspan=3>Idea description,</td><td rowspan=1 colspan=4>Pre-cutoff papers (title,</td><td rowspan=2 colspan=1>Prefere cadidate lus selects</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>2025)</td><td rowspan=1 colspan=1>plus self-review of the</td><td rowspan=1 colspan=3>experiment, self-rating</td><td rowspan=1 colspan=4>abstract, citations); per-idea</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>finished abstract</td><td rowspan=2 colspan=3></td><td rowspan=2 colspan=4>retrieval</td><td></td></tr><tr><td rowspan=1 colspan=1>the one idea written up elects</td></tr><tr><td rowspan=1 colspan=1>AI Co-Scientist</td><td rowspan=1 colspan=1>Fast initial review, then</td><td rowspan=1 colspan=3>Hypothesis or proposal plus</td><td rowspan=1 colspan=4>Parametric (initial); web</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>(Gottweis et al.,</td><td rowspan=1 colspan=1>literature-grounded full</td><td rowspan=1 colspan=3>parsed goal, constraints, and</td><td rowspan=1 colspan=4>search or scientist-provided</td></tr><tr><td rowspan=1 colspan=1>2026)</td><td rowspan=1 colspan=1>review</td><td rowspan=1 colspan=3>whether novelty is required</td><td rowspan=1 colspan=4>repository (full)</td><td rowspan=1 colspan=1>review summarizes known</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4></td><td></td></tr></table>

Table 4: Novelty judgments embedded in surveyed AI-scientist pipelines. Pool controllable: where an external module can replace the native judgment.

<table><tr><td>System</td><td>Judges what</td><td>Input I</td><td>Pool</td><td>Output O</td></tr><tr><td>RAG-Novelty et al., 2025)</td><td>(Lin Published papers</td><td>Title and abstract plus two retrieval scalars (mean neighbor date, mean cosine); neighbors never reach the pable</td><td>Own vector store (≈12k per field), Integer 1–10 plus justification k=10, pre-date filter; fully swap-</td><td></td></tr><tr><td>OpenNovelty (Zhang Full papers et al., 2026)</td><td></td><td>tion claims</td><td>PDF → core task and 1–3 contribu- Proprietary search API; top-50 per Per contribution: can refute / cannot task, top-10 per contribution; citation refute / unclear, with evidence and index is the injection point</td><td>summary; no numeric score</td></tr><tr><td>2024)</td><td>NovaScore (Ai et al., News documents</td><td>Document → atomic content units</td><td>Unit bank (SBERT, top-5, &gt;0.6); Per-unit binary novelty × salience experimenter-defined</td><td>→ [0, 1] aggregate</td></tr><tr><td>InnoEval (Qiao et al., Research ideas 2026)</td><td></td><td>tions, methods, settings, results), gory; swappable searchers auto-extracted</td><td>Six-tuple (TLDR, motivations, ques- Six-source search, top-10 per cate- 0–10 per dimension incl. novelty,</td><td>five personas, mean plus accept-tier label</td></tr><tr><td>et al., 2026)</td><td>ScholarEval (Moussa Research ideas</td><td>mensions extracted from it</td><td>Research plan, with comparison di- Retrieved literature; swappable</td><td>Contribution arm: pairwise score against each retrieved paper per di- mension, averaged to —10 to +10; no accept threshold</td></tr></table>

Table 5: Standalone novelty judges.
<table><tr><td>System</td><td>Committed idea artifact (post-check, pre-execution)</td><td>Where captured</td><td>What runs on it</td></tr><tr><td>AI Scientist v1</td><td>Idea dict (Name, Title, Experiment), novelty-filtered</td><td>ideas. json per template</td><td>experiments → full paper</td></tr><tr><td>AI Scientist v2</td><td>Idea JSON incl. Short Hypothesis, Related Work, Abstract, Ex- ideas JSON; human picks an index periments</td><td></td><td>tree-search experiments → paper</td></tr><tr><td>CoI-Agent</td><td>final_idea free-text blob</td><td>return value of the regeneration loop</td><td>experiment design</td></tr><tr><td>Si et al.</td><td>Full proposal JSON (7 fields incl. test cases, fallback plan)</td><td>top_ideas. json, rank-ordered and filter-passed</td><td>human study / execution study</td></tr><tr><td>ResearchAgent</td><td>Problem, method, experiment design, each with rationale; best- pipeline history, max-score pick of-N</td><td></td><td>(idea is the end product)</td></tr><tr><td>Agent Laboratory</td><td>Research plan from plan formulation</td><td>agent state</td><td>experiments → paper</td></tr><tr><td>SCI-IDEA</td><td>Refined “Aha” idea text</td><td>module-2 output</td><td>(idea is the end product)</td></tr><tr><td>CycleResearcher</td><td>None exposed; output is a full paper</td><td>LaTeX</td><td>review → RL reward</td></tr><tr><td>DeepScientist</td><td>Promoted hypothesis record (claim plus valuation vector)</td><td>findings memory</td><td>expensive validation runs</td></tr><tr><td>Robin</td><td>Ranked candidates (name, lit report, strength score)</td><td>ranked CSV</td><td>human wet-lab selection</td></tr><tr><td>AI-Researcher</td><td>One selected idea report (1–3k words)</td><td>selection-call output</td><td>code survey → implementation</td></tr><tr><td>ARC</td><td>hypotheses. md (H1–H4 with rationale, prediction, failure con- stage-08 artifact dition)</td><td></td><td>later stages → report</td></tr></table>

Table 6: What each pipeline passes downstream after its novelty check.

<table><tr><td>System</td><td>Reason</td></tr><tr><td>Included</td><td></td></tr><tr><td>AI Scientist v1, CoI-Agent, Si et al., ARC</td><td>deployed check</td></tr><tr><td>ResearchAgent, AI-Scientist-v2</td><td>judgment extracted from the pipeline</td></tr><tr><td>ScholarEval</td><td>standalone, but scores an idea against a supplied</td></tr><tr><td></td><td>paper set</td></tr><tr><td>Excluded SCI-IDEA</td><td></td></tr><tr><td>CycleResearcher</td><td>no code released</td></tr><tr><td>DeepScientist</td><td>monolithic fine-tune, no separable judgment</td></tr><tr><td>Agent Laboratory</td><td>paper-era judge prompt unpublished</td></tr><tr><td>Robin, AI-Researcher.</td><td>no check; its reviewer scores the finished paper relative selection among candidates, no per-idea</td></tr><tr><td>VirSci</td><td>verdict</td></tr><tr><td>AI Co-Scientist</td><td>closed source, web search not fixable</td></tr><tr><td>SciMON</td><td>pool is training-set idea sentences; output is a rewrite</td></tr><tr><td>Other standalone judges</td><td>expect a published paper or PDF; non-binary outputs</td></tr></table>

Table 7: Systems included in and excluded from the audit.

<table><tr><td>Check</td><td>Native input built from (title, description)</td></tr><tr><td>AI Scientist v1</td><td>idea dict: Name = slugified title, Title = title,</td></tr><tr><td>CoI-Agent</td><td>Experiment = description one idea blob, &quot;title: description&quot;</td></tr><tr><td>Si et al.</td><td>one idea blob, &quot;title: description&quot;; title also</td></tr><tr><td>ARC</td><td>used as a search query title and description together as the hypothesis</td></tr><tr><td>ResearchAgent</td><td>text Problem = title, Rationale = description; target</td></tr><tr><td>ScholarEval</td><td>paper = title and description research plan = &quot;title: description&quot;</td></tr><tr><td>AI-Scientist-v2</td><td>Title = title, Proposal = description, in the re- flection prompt</td></tr></table>

Table 8: How the shared proposal is passed to each check’s native input.

<table><tr><td>Call</td><td>$/ proposal Latency</td><td></td></tr><tr><td>CLAIMNoV verdict</td><td>0.010</td><td>2.6s</td></tr><tr><td>CoI-Agent check</td><td>0.039</td><td>19s</td></tr><tr><td>ARC (lexical)</td><td>0</td><td></td></tr></table>

Table 9: Cost per proposal at list price and median latency of the final judge call (Sonnet 4.6), measured over 948 CLAIMNOV and 446 CoI-Agent calls. ARC makes no model call.

## B Adapter Details

We pass the same (proposal, pool) pair to each check. For all checks that do some kind of search, we sandbox the search, ranking the pool by cosine similarity (text-embedding-3-large) to the query (checks can still use their native query style and specified depth k). In all cases, we set the final judge to Sonnet 4.6 at temperature 0 (Table 10). In the OpenNovelty-style baseline, we reimplement the per-contribution design of Zhang et al. (2026) with BM25 retrieval to avoid sharing a retriever with CLAIMNOV. BM25 throughout is rank\_bm25’s BM25Okapi with default parameters (k<sub>1</sub>=1.5, b=0.75) over lowercased alphanumeric tokens with stopwords removed, and the scientific embeddings are SciNCL (malteos/scincl) and SPECTER2 (allenai/specter2\_base with the proximity adapter).

<table><tr><td>Check</td><td>Search into pool</td><td>Decision rule</td></tr><tr><td>AI Sci. v1</td><td>up to 10 rounds, each one LLM query →</td><td>&quot;Decision made: not novel&quot; rejects; no decision after 10 rounds rejects (shipped default)</td></tr><tr><td>CoI-Agent</td><td>top-10 up to 3 LLM queries → top-5 each, deduplicated</td><td>&lt;similar&gt;1 rejects and names a paper; empty retrieval passes (shipped)</td></tr><tr><td>Si et al.</td><td>1–3 keyword queries plus the title → top-10 each</td><td>papers scored 1–10, the top 5 checked one by one, any &quot;same problem and approach&quot; rejects</td></tr><tr><td>ARC</td><td>whole pool, no LLM</td><td>keyword Jaccard and title similarity ≥ 0.25 for some</td></tr><tr><td></td><td>ResearchAgent proposal → top-10 titles</td><td>paper rejects Originality rating ≤ 3 of 5 rejects (no native threshold; shipped temperature 1.25 not</td></tr><tr><td>ScholarEval</td><td>proposal → top-8</td><td>kept) mean pairwise score over the 8 papers ≤ 0 rejects (no native</td></tr><tr><td>AI-Sci.-v2</td><td>proposal → top-10</td><td>threshold) reflection prompt plus an appended NOVELTY: pass/fail</td></tr><tr><td>OpenNov.- style</td><td>1–3 extracted contributions → BM25 top-10 each</td><td>slot one judge call over all contributions; rejects when every contribution is can_refute (rejecting on any raises false alarms to 55.8%)</td></tr></table>

Table 10: Sandboxed search and decision rule of each adapter.

## C CLAIMNOV Prompts

The exact CLAIMNOV prompts.

Claim Extraction (system prompt)   
You extract the contribution/novelty claims   
from the abstract of a scientific paper. A   
’claim’ is one atomic, headline contribution   
the authors make — what they would put in a   
numbered ’Our contributions:’ list.   
BE CONSERVATIVE. Most papers have 1-4 atomic   
contributions, not 5-8. Prefer one

consolidated claim over fine-grained sub-claims.

Dimension definitions:

\- methodology: a new algorithm, architecture, training procedure, loss function, selection mechanism, or technique that is the paper’s primary \*solution\* to a problem. The contribution IS the method — how something is done. Includes objective functions, encoders, and model claims when they are the central proposed technique.

\- result: a new empirical or theoretical finding, analysis insight, performance observation, or theoretical characterization that is discovered/proven rather than proposed as a solution. This includes: showing that an existing design choice is effective, demonstrating generalization ability, reporting accuracy/fairness/efficiency gains, and deriving theoretical distributions or bounds when the distribution/bound is the finding, not a technique being proposed.

\- resource: a new dataset, benchmark, tool, model release, or codebase released by THIS paper.

\- problem: a new problem formulation, task, setting, scenario, or research question this paper introduces — the question or framing, not the answer. Includes formalizations of novel learning settings, new task definitions, and new evaluation scenarios.

Dimension boundary rules (apply these when confused):

\- An OBJECTIVE FUNCTION or LOSS is ‘methodology‘ when it is the paper’s proposed technique for solving a problem; it is ‘result‘ only when it is a derived theoretical characterization that the paper proves as a finding.

\- A DISTRIBUTION (e.g., Laplace on SO(3)) that is the paper’s main proposed modeling tool is ‘methodology‘; if the paper’s claim is "we prove X has property Y" with no proposed use, it may be ‘result‘.

\- "We show / demonstrate / find that technique T is effective" → ‘result‘, not ‘methodology‘. Effectiveness findings about a claim are results even if the claim is also described elsewhere.

\- A FRAMEWORK or FORMALIZATION that characterizes existing assumptions, explains prior work, or provides theoretical grounding (rather than solving a new problem) is ‘methodology‘ when it is a new analytical tool, NOT ‘problem‘.

\- A DATASET or BENCHMARK constructed as part of the paper → ‘resource‘, not ‘methodology‘, even if construction required significant effort.

\- A SETTING, TASK, or SCENARIO introduced for the first time → ‘problem‘, even when phrased in proposal voice ("we propose the setting of. . . ").

\- When a paper both introduces a new setting/task AND a new method for it, extract BOTH as separate claims (‘problem‘ + ‘methodology‘).

Decomposition rules:

\- DO split contributions that are independently verifiable AND of different dimensions.

\- DO NOT split a single method into sub-claims, or a single result into sub-numbers.

\- DO NOT extract motivation/background or claims about prior work.

\- DO extract a ‘problem‘ claim whenever the paper introduces a novel task, setting, or formalization — even if it also proposes a method for it. These are separate contributions of different dimensions.

\- DO extract a ‘result‘ claim whenever the abstract reports empirical outcomes, performance gains, or evaluative findings even if the paper’s main contribution is a method.

\- DO extract a ‘resource‘ claim whenever the paper releases a dataset, benchmark, or tool — even as a secondary contribution.

Recall guidance — scan for these contribution signals before finalising:

\- Is there a novel problem/task/setting/formalization introduced? → ‘problem‘

- Is there a new algorithm, architecture,   
training procedure, or loss function? →   
‘methodology‘   
- Are benchmarks, datasets, or tools   
released? → ‘resource‘

\- Are empirical or theoretical results reported (accuracy gains, effectiveness demonstrations, theoretical bounds, generalization experiments)? → ‘result‘

\- Does the abstract describe a claim or design choice shown to be effective (e.g., encoder choice, selection mechanism, initialization)? → likely a ‘result‘ claim unless it is the central proposed technique.

\- Does the abstract describe a method with multiple named sub-claims (e.g., a selection mechanism, a refinement stage)? → consider whether each is an independently verifiable methodological contribution or whether they together form one consolidated ‘methodology‘ claim.

\- Does the paper introduce a novel scenario or learning setting that prior work has not formalized? → ‘problem‘ (check even when the abstract’s primary voice is about a proposed method).

Proposal-voice caveats (important):

\- "We propose/introduce X" where X is a SETTING, TASK, SCENARIO, or problem formulation is dimension ‘problem‘, NOT methodology — even when phrased in proposal voice. ‘methodology‘ is reserved for the technique/algorithm that solves a problem.

\- "We propose/introduce X" where X is a METHOD, ALGORITHM, OBJECTIVE, or TECHNIQUE is dimension ‘methodology‘.

\- Future-tense proposal wording ("we will develop", "we plan to") does not change the dimension: classify by what the contribution IS, not how it is voiced.

For each claim also extract its facet:

- attribute: the comparable axis of the claim,   
as a short noun phrase (the retrieval key).   
- value: what the paper asserts on that axis.   
Return JSON:   
{"claims": [{"claim": "<one self-contained   
sentence>", "dimension":   
"methodology|result|resource|problem",   
"attribute": "...", "value": "...   
"abstract\_span": "<verbatim quote>"}]}   
If the abstract contains no contribution   
claims, return {"claims": []}.

## Support Counting (Stage B)

You find ALL the supporting evidence in a scientific paper’s body for a specific   
contribution claim from its abstract. Select EVERY paragraph (by index) containing direct supporting evidence — descriptions, numbers, theorem statements, implementation details, appendix analysis tied to the claim. There is NO cap. Skip only paragraphs not actually about this claim (background, related work, other contributions).   
Order the selected paragraphs by relevance to the claim, most relevant first.   
Return JSON: {"evidence":   
[{"paragraph\_index": <int>}]}

## Graded Entailment Judge

When the claims assert numeric values: a prior claim anticipates a target only if it establishes the same kind of finding under comparable conditions. For performance/achievement claims on a shared benchmark and metric, a prior result at least as strong may fully entail the target. For measurements, effect sizes, estimates, or findings in different settings or

populations, a different (even larger) prior number does NOT by itself entail the target the direction, setting, or refined estimate may be exactly what is new.   
Claim 1 (target):   
<target claim>   
Additional context for Claim 1:   
Facet — attribute: <attribute>; value:   
<value>   
Claim 2 (prior work: <prior title>):   
<prior claim>   
Additional context for Claim 2:   
Facet — attribute: <attribute>; value:   
<value>   
Output ONLY one integer: 1, 2, 3, 4, or 5.

## Verdict (profile judge)

You are the final novelty judge for a proposed research IDEA. A claim-level analysis has already been run: the idea’s contribution claims were extracted, and for each claim the most similar existing claims were retrieved from a pool of prior papers and rated for how far they cover the idea’s claim (1 = unrelated, 2 = weak topical overlap, 3 = partial coverage with important details differing, 4 = strong coverage with minor differences, 5 = full coverage of the claim’s substantive content).

## D Disguise Prompts

We provide the exact prompts used to disguise proposals for each rung, as well as the prompt used for the verifier for rungs 3 and 4. Note the rung-1 prompt is also used to rewrite claim-disputed papers (§4.4). These are run with Sonnet 4.6.

## Rung 1: voice swap

You are drafting a research idea proposal.   
Below is the abstract of a completed research   
paper. Recast it as a NEW research idea   
someone is proposing to work on: write a   
proposal title and a ∼200-word description   
in proposal voice (future tense, "we propose   
/ we will"). Keep the technical content as   
close to the original as possible — same   
contribution, same method, same expected   
findings; only the framing changes from "we   
did" to "we propose".   
Abstract:   
{abstract}   
Return JSON only: {"title": "...",   
"description": "..."}

## Rung 2: deep paraphrase

You are drafting a research idea proposal. Below is the abstract of a completed research paper. Recast it as a NEW research idea   
someone is proposing to work on, as a deep paraphrase: preserve every technical claim, the method, and the expected findings, but rewrite from scratch in entirely different vocabulary and sentence structure — no phrase over 4 words may match the original.   
Technical proper nouns and dataset names may remain. Write a proposal title (different wording from the original title) and a   
∼200-word description in proposal voice.   
Abstract:   
{abstract}   
Return JSON only: {"title": "...",   
"description": "..."}

## Rung 3: reframe

You are drafting a research idea proposal. Below is the abstract of a completed research paper. Recast it as a NEW research idea proposal that makes the SAME core technical contribution but is presented completely differently:

\- motivate it from a different angle (different application domain or research concern than the original opens with)

\- reorder the contributions; do not mirror the original’s sentence structure or presentation order

\- do not reuse the original’s method name; either leave the method unnamed or give it a neutral descriptive name

\- do not reuse the original’s examples or illustrative scenarios

\- keep every substantive technical claim: the same method, the same key mechanism, the same expected findings (numbers may be stated approximately)

∼200 words, proposal voice (future tense).   
Title must not echo the original title.

Abstract: {abstract}

Return JSON only: {"title": "... 11   
"description": "..."}

## Rung 4: adversarial restructuring

```jsonl
You are drafting a research idea proposal.
Below is the abstract of a completed research
paper. Write a proposal that a hurried reader
would take for an unrelated new idea, but
that an expert doing a careful side-by-side
comparison would judge to be substantially
the same work:
- start from a different theoretical or
practical starting point and let the core
mechanism "emerge" as the proposed approach
- rename or leave unnamed every method/system;
drop or generalize all specific numbers
("substantial improvement", not percentages)
- restructure completely: different opening
problem, different order, different emphasis
(e.g., foreground what the original treats as
secondary)
- you may add plausible framing (motivation,
evaluation plans) but must NOT add or change
any substantive technical contribution
- the same core problem must be solved by the
same core mechanism with the same expected
outcome
∼200 words, proposal voice. Title must
appear unrelated to the original.
Abstract:
{abstract}
Return JSON only: {"title": "...",
"description": "..."}
```

## Verifier (rungs 3 and 4)

Do these two texts describe substantially the   
same core research contribution — same core   
problem solved by the same core mechanism?   
Answer "yes" or "no", then one sentence.   
TEXT A (paper abstract):   
{abstract}   
TEXT B (proposal):   
{description}

## E Later-Realized Labeling Prompts

We provide exact prompts used to label the laterrealized set, run with Opus 4.6.

## Contribution List (independent extraction)

Write the "Our contributions are:" list that   
this proposal’s authors would put in their   
introduction, one contribution per line.   
Output nothing else.   
PROPOSAL:   
{title: description}

```jsonl
Per-Contribution Label (run once against
the pool, once against F)
You are labeling a proposed research IDEA at
the level of its individual contribution
claims.
IDEA:
{title: description}
IDEA’S CONTRIBUTION CLAIMS:
{the numbered contribution list from the
prompt above}
CANDIDATE PAPERS ({published before the
cutoff | published after the cutoff}):
{title and abstract of the 8 nearest papers
in the pool, or in F}
For EACH claim, decide whether some candidate
paper already establishes it: the same thing
is done or shown for that claim (same core
problem and same core approach for that
claim), not merely related work or work the
claim builds on. If yes, name the paper.
Return JSON only:
{"claims": [{"i": <claim index>,
"established": "yes"|"no", "by": "<exact
paper title or null>", "why": "<one
sentence>"}]}
```

## F Label Validation

For n=100 pairs of papers A and B where an LLM identified a prior-art dispute, two annotators independently mark (i) whether the dispute is genuine, (ii) whether the extracted B is the correct prior art, and (iii) the dispute type. They achieve (i) 81/100 raw agreement with Cohen’s κ = 0.62, (ii) 73/100 and κ=0.47, and (iii) 52/100 and κ=0.40, and we drop the dispute-type labels due to the low kappa value. A third annotator then marked the 63 rows with any disagreements, blinded to the labels of the first two annotators. By 2-of-3 majority we find that we have 59/100 genuine disputes with 67/100 having correctly-identified B. We use the 40 pairs that have a genuine dispute and correct B as the label-clean subset, 36 of which have usable pools (Table 12). All three annotators have graduate-level education in English in the relevant research field.

<table><tr><td>Question</td><td>Agreement</td><td>κ</td></tr><tr><td>Dispute is genuine</td><td>81/100</td><td>.62</td></tr><tr><td>B is the correct prior art</td><td>73/100</td><td>.47</td></tr><tr><td>Dispute type (dropped)</td><td>52/100</td><td>.40</td></tr></table>

Table 11: Raw agreement and Cohen’s κ between the first two annotators.

We give the annotation instructions below. The audit of later-realized and sorter-matched ideas uses the second set.

<table><tr><td>Check</td><td>Rej. acc.</td><td>Rej. rej.</td><td>Names B</td></tr><tr><td>AI Scientist v1</td><td>7/16</td><td>7/19</td><td></td></tr><tr><td>CoI-Agent</td><td>2/16</td><td>3/19</td><td></td></tr><tr><td>Si et al.</td><td>14/16</td><td>14/19</td><td>19, rejecting</td></tr><tr><td>OpenNov.-style</td><td>2/16</td><td>3/19</td><td></td></tr><tr><td>CLAIMNOV</td><td>1/16</td><td>1/19</td><td>15, 14 passing</td></tr></table>

Table 12: Rejections on the 36 label-clean pairs. Names B: for CLAIMNOV, B in the profile at level 3 or above.

Fill in your file completely and independently, with no discussion of any row until both files are done. Judge only from what is in the row (plus the linked PDFs). Do not search the literature yourself. Mark any row you are unsure about as low confidence and give the reason in the notes.

For each row, read A’s title and abstract, the reviewer’s snippet, and B’s title and abstract, then answer the following.

Q1. Is the reviewer genuinely disputing A’s novelty by pointing at prior work? Yes if the reviewer asserts or clearly implies that A’s idea, method or setting already exists in prior work (“this was already done”, “very similar to [B]”, “unclear how this differs from [B]”, “the claim of being first is not true”). No for a neutral citation, a request to add a citation, background discussion, praise, or a comparison request that does not question novelty.

Q2. Is the listed B the work the reviewer is referring to? Yes if B’s title and identity match the work invoked in the snippet. No if the wrong paper was linked (e.g., same author, different paper). Cannot assess if B is unresolvable shorthand that the snippet does not identify.

Q3. What kind of dispute is it? (Answer even if Q1 is no, using your best reading.) duplicate: essentially the same thing was already done. substantial overlap: the core contribution heavily overlaps B, though not identically. incremental: A is a minor delta over B. framing: the reviewer contests a novelty statement or overclaim rather than the contribution itself. other: explain in the notes.

A reviewer who concedes the contribution while noting that the problem area is old isframing, not duplicate. A dispute about the problem setting being identical is substantial overlap.

## Claim-dispute annotation (annotator 3, blind)

Same task and instructions as above, for Q1 and Q2. You are the third independent annotator. These are the pairs where the first two annotators disagreed, but their answers are not shown and the row order is shuffled. The majority vote is computed after you return the file.

## Later-realized audit

Each idea comes with the 20 most similar papers from the prior-work pool, all published before the cutoff, most similar first (P1–P20). Some items also list one or more later papers published after the cutoff, marked “F”. These are for reference only. Do not count an F paper as prior work. Work independently, do not search the web, and judge only from the idea text and the listed abstracts.

Q1. Does any listed paper (P1–P20) already do substantially what the idea proposes, with the same core problem and the same core approach? Answer yes or no, and if yes, give the number of the paper that does it best. Answer no for a paper that only shares the topic, that the idea builds on, or that does one side component of the idea. Answer yes only if the idea’s main contribution is already there. A missing peripheral detail does not make the idea new.

Q2 (only if Q1 is no). Is any single contribution of the idea clearly established by a listed paper, even though the idea as a whole is not? Answer yes or no, and if yes, give the paper number. A few words naming the contribution are optional.

Confidence. Rate your Q1 answer 1 (guess), 2 (fairly sure) or 3 (certain).

## F.1 Audit of generated ideas

We audit 60 generated ideas by hand with the second set of instructions above, one annotator per item. These are 45 later-realized ideas (25 at random, 10 that the whole-idea Opus pass called already done, and 10 that a low-false-alarm check rejected) and 15 sorter-matched ideas. This labeling is extremely strenuous, since each item requires reading an idea against 20 abstracts and deciding whether any one paper does substantially the same thing, a judgment on which experts often disagree (Si et al., 2025). Annotators confirm 28 of the 45 later-realized ideas as novel relative to the pool (a stratum-weighted 60% of all 400) and judge 9 of the 15 sorter-matched ideas already done. Table 13 gives each check’s rejections on both groups of later-realized ideas. On the 9 sorter-matched ideas judged already done, CLAIMNOV with fixed retrieval depth rejects 1 and the w/o-profile judge 7, which also rejects 3 of the 6 judged novel.

<table><tr><td>Check</td><td>Novel (28)↓ Done (17)</td></tr><tr><td>AI Scientist v1</td><td>11 11</td></tr><tr><td>CoI-Agent</td><td>0 2</td></tr><tr><td>Si et al.</td><td>24 17</td></tr><tr><td>ARC</td><td>0 0</td></tr><tr><td>ResearchAgent</td><td>27 17</td></tr><tr><td>ScholarEval</td><td>0 0</td></tr><tr><td>AI-Sci.-v2</td><td>14 10</td></tr><tr><td>OpenNov.-style</td><td>8 2</td></tr><tr><td>W/o profile</td><td>6 2</td></tr><tr><td>CLAIMNOV</td><td>2 0</td></tr></table>

Table 13: Rejections of the later-realized ideas annotators confirm as novel and of those they judge already done.

## G Attribution by Dimension and Relation Type

Table 14 breaks down the claim-level attribution of §5.2 by the disputed claim’s dimension and by the relation type in the review.

<table><tr><td>Disputed claim</td><td>n extracted↑ B in profile↑ lvl≥3↑</td></tr><tr><td>Problem 37</td><td>19 30</td></tr><tr><td>42</td><td>17 42 29 17</td></tr><tr><td>Methodology Resource 39</td><td>12 20</td></tr><tr><td>42</td><td>6 35 25 18</td></tr><tr><td>Result</td><td>23</td></tr><tr><td>Prior art 53 Comparison 86</td><td>37</td></tr><tr><td>Overlap 21</td><td>49 24 18 11</td></tr></table>

Table 14: Attribution on the claim-disputed set by dimension of the disputed claim (top) and by relation type (bottom). Extracted: a claim was extracted in that dimension.

## H CLAIMNOV Ablations

We explain the ablations we evaluate for CLAIM-NOV. Each aims to ablate one stage, leaving the rest unchanged (Table 15).

<table><tr><td>Ablation</td><td>Change from full CLAIMNoV</td></tr><tr><td>W/o claim profile</td><td>the verdict stage (same model, prompt) reads eight pool abstracts nearest to the proposal by embedding</td></tr><tr><td>W/o verdict judge (graded check)</td><td>instead of the novelty profile fail when aggregate score N(P) &lt; 0.5 or any claim is covered at level 4 or higher by one prior claim</td></tr><tr><td>Per-claim verdict rule</td><td>one judge call per candidate paper on whether it establishes the central claim; any yes fails</td></tr><tr><td>Fixed depth (k=10)</td><td>the 10 nearest same-dimension prior claims for every claim, disregarding pool size</td></tr></table>

Table 15: CLAIMNOV ablations.

## I Disguises Caught by Rung

Table 16 breaks down Table 1 by disguise rung. Aside from ARC (which we expect to have such behavior since it’s a fully lexical check), we don’t observe significant fluctuation in catch for different checks across the rungs.

<table><tr><td>Check</td><td>r1↑</td><td>r2↑</td><td>r3↑</td><td>r4↑</td></tr><tr><td>AI Scientist v1</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>CoI-Agent</td><td>100</td><td>99</td><td>96</td><td>93</td></tr><tr><td>Si et al.</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>ARC</td><td>90</td><td>0</td><td>0</td><td>0</td></tr><tr><td>ResearchAgent</td><td>77</td><td>91</td><td>94</td><td>92</td></tr><tr><td>ScholarEval</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>AI-Scientist-v2</td><td>94</td><td>96</td><td>97</td><td>92</td></tr><tr><td>CLAIMNOV</td><td>99</td><td>97</td><td>98</td><td>92</td></tr></table>

Table 16: Disguises caught per rung (100 each, all 400 disguises). Bold: best, underline: second.

## J Intervals and Paired Tests

In Table 17 we give 95% intervals for Table 1. In Table 18 we give exact McNemar tests. Finally, in Table 19 we give repeated runs and clustered tests. Main-table rows come from one run per check on the first-party Anthropic API. Later Claude runs (repeats, profile-plus-abstracts, anchor-in, larger pools) use Sonnet 4.6 on Amazon Bedrock. Tables 20 and 21 score attribution as retrieval.

<table><tr><td>Check</td><td>FA (%)</td><td>J</td><td>Δ</td></tr><tr><td>AI Scientist v1</td><td>51.8 [46.9, 56.6]</td><td>.48 [.37, .59]</td><td>.76 [.69, .83]</td></tr><tr><td>CoI-Agent</td><td>3.8 [2.3, 6.1]</td><td>.93 [.85, .98]</td><td>.89 [.85, .93]</td></tr><tr><td>Si et al.</td><td>96.0 [93.6, 97.5]</td><td>.04 [.02, .07]</td><td>.06 [.02, .10]</td></tr><tr><td>ARC</td><td>0.0 [0.0, 1.0]</td><td>.22 [.21, .24]</td><td>.22 [.21, .24]</td></tr><tr><td>ResearchAgent</td><td></td><td>98.5 [96.8, 99.3] −.10 [−.14, −.05] .21 [.15, .26]</td><td></td></tr><tr><td>ScholarEval</td><td>0.0 [0.0, 1.0]</td><td>.00</td><td>.00</td></tr><tr><td>AI-Sci.-v2</td><td>39.2 [34.6, 44.1]</td><td>.56 [.35, .77]</td><td>.64 [.57, .72]</td></tr><tr><td>OpenNov.-style</td><td>11.5 [8.7, 15.0]</td><td>.88 [.80, .96]</td><td>.90 [.85, .94]</td></tr><tr><td>CLAIMNOV</td><td>1.5 [0.7, 3.2]</td><td>.95 [.93, .98]</td><td>.92 [.88, .96]</td></tr></table>

Table 17: 95% intervals for Table 1. FA: Wilson. J, ∆: cluster bootstrap by source and anchor.

<table><tr><td>Event</td><td>CoI-Agent</td><td>W/o profile</td><td>OpenNov.</td></tr><tr><td>Disguise caught</td><td>10/11, p=1.0</td><td> $5 / 1 1 , p { = } . 2 1$ </td><td>0/12, p&lt;.001</td></tr><tr><td>Src. out rejected</td><td>10/23, p=.0353/42, p&lt;10−9</td><td></td><td> $8 / 2 8 , p = . 0 0 1$ </td></tr><tr><td>Later-realized FA</td><td></td><td>6/15, p=.078 4/45, p&lt;10−9</td><td> $3 / 4 3 , p < 1 0 ^ { - 9 }$ </td></tr><tr><td>Claim set rejected</td><td>0/12, p&lt;.001</td><td> $0 / 1 2 , p < . 0 0 1$ </td><td>0/8, p=.008</td></tr><tr><td>Claim, B on pass</td><td>7/0, p=.016</td><td>-5</td><td></td></tr><tr><td>Hybrid passed Hybrid, B on pass 18/4, p=.004</td><td>20/5, p=.004</td><td> $2 4 / 1 , p { < } 1 0 ^ { - }$ </td><td> $1 5 / 7 , p { = } . 1 3$ </td></tr></table>

Table 18: Exact McNemar tests of CLAIMNOV against three checks, as discordant counts (CLAIMNOV only / other only).

<table><tr><td>Test</td><td>Result</td></tr><tr><td>Repeated verdict runs</td><td>4 and 3 flips against the reported run, 1 between repeats; J +.941, +.941, +.938; ∆ +.956 in all three</td></tr><tr><td>Repeated CoI-Agent</td><td>7 flips on each of 283 and 163 items</td></tr><tr><td>runs Paired ∆J</td><td> $+ . 0 2 5 \left[ - . 0 5 , + . 1 2 \right]$  on 474 items; +.083</td></tr><tr><td>(CLAIMNOV —</td><td> $[ - . 0 4 , + . 2 5 ]$  on the 163 items with three</td></tr><tr><td>CoI-Agent)</td><td>CoI-Agent runs</td></tr><tr><td>Clustered, vs</td><td> $\Delta J + . 0 2 0 [ - . 0 4 , + . 1 1 ]$  ; source-out sign-flip</td></tr><tr><td>CoI-Agent</td><td>permutation p=.07-.09</td></tr><tr><td>Clustered, vs w/o</td><td>false-alarm rate –  $\cdot . 1 0 [ - . 1 8 , - . 0 3 ] , p { = } . 0 0 2 ;$ </td></tr><tr><td>profile</td><td>source-out p&lt;10−5</td></tr><tr><td>Clustered, vs</td><td>false alarms and source-out p≤.008</td></tr><tr><td>OpenNov.-style</td><td>CLAIMNOV 3 (4 of 6 on one), CoI-Agent 2 (14 of</td></tr><tr><td>Anchors with false alarms</td><td>15 on one), w/o profile 11, OpenNov.-style 10</td></tr><tr><td>Endpoint</td><td>re-issuing the w/o-profile prompt on Bedrock raises</td></tr></table>

Table 19: Run-to-run stability on 474 byte-identical items, and clustered tests.

<table><tr><td>Set</td><td>Ranking</td><td></td><td></td><td>R@1 R@3 R@5 MRR</td><td></td></tr><tr><td>Claim (160)</td><td>CLAIMNOV dense BM25</td><td>.21 .37 .26</td><td>.44 .59 .54 .26</td><td>.53 .72 .72</td><td>.35 .52 .45 .25</td></tr><tr><td>Hybrid (100)</td><td>OpenNov. CLAIMNOV dense BM25</td><td>.24 .25 .07 .74</td><td>.39 .32 .92</td><td>.26 .42 .66 .96</td><td>.32 .29 .84</td></tr><tr><td>Disg. (347)</td><td>OpenNov. CLAIMNOV dense</td><td>.68 .89 .87</td><td>.98</td><td>.98 .91</td><td>.82 .90 .91</td></tr></table>

Table 20: Rank of the relevant paper (B, or the source) in each ranking.

<table><tr><td>Set</td><td>Level Papers</td><td></td><td>Relevant</td><td>Prec.</td></tr><tr><td>Claim (160)</td><td>≥3</td><td>291</td><td>54</td><td>.19</td></tr><tr><td>Claim (160)</td><td>≥4</td><td>14</td><td>2</td><td>.14</td></tr><tr><td>Hybrid (100)</td><td>≥4</td><td>15</td><td>13</td><td>.87</td></tr><tr><td>Disg. (347)</td><td>≥4</td><td>326</td><td>303</td><td>.93</td></tr></table>

Table 21: Profile papers at the top coverage levels, and how many are the relevant paper. Level 4 or above fires for 9 of 400 later-realized ideas.

## K Robustness

In this section, we test some assumptions to better understand if our main results depend on our specific labels, judge model, pool sizes, or the particular data we collected.

Because our later-realized labels come from one Opus labeler, we recompute false alarms under three alternative labelers in Table 23, specifically (1) a labeler from a different model family (gpt-5), (2) the ideas both Opus and gpt-5 agree on, and (3) the full set without the $n = 1 0 1$ ideas a whole-idea Opus labeling pass considered previously done. For all of these labeling paradigms, CLAIMNOV’s falsealarm rate stays between 1.3% and 1.5%.

We check whether the claim-level novelty profile we provide for our final verdict is only helpful to our particular judge (Sonnet 4.6) by rerunning this stage with two non-Anthropic judges, specifically gpt-oss-120b (117B parameters) and Llama 4 Maverick (17B active, 400B total) in Table 24. These are weaker models, so catch does drop, but we still observe as with Sonnet that providing the profile cuts false alarms and source-out rejections relative to the base judge reading nearest abstracts.

We check in Table 25 whether baselines can match the judged checks. These are thresholds on lexical or embedding similarity, a judge that sees only the single nearest paper, and the extracted score-based checks at their best threshold. The baselines are not competitive with CLAIMNOV’s J.

Because our disguise pools hold only 20 papers, in Table 26 we artificially increase the pool size to ≈ 200 to investigate if our catch can survive more realistic pool sizes. While catch does fall on the hardest rung, source-out rejections stay near zero.

We also perform a handful of smaller checks that further support these findings (Table 22). Asked with no pool, Sonnet 4.6 names B first for at most 1 of the 160 claim-disputed proposals, so recall of pre-T papers does not drive that set. Appendix F.1 reports a human audit of 60 generated ideas.

Reproducibility. We release all code and prompts at https://github.com/emmatliu/ claimnov, including the scripts that build each dataset and run each check, together with the per-item records behind our tables. We release the generated ideas themselves. For source papers, pool papers and ICLR reviews, we release identifiers and titles rather than paper or review text, and the released code re-mines the claim disputes from public OpenReview reviews. We do not release the disguised or hybrid proposals because they restate published papers, but we do release code to rebuild them.

<table><tr><td>Check</td><td>CLAIMNOV</td><td>CoI</td></tr><tr><td>All 400 disguises</td><td></td><td></td></tr><tr><td>caught</td><td>386</td><td>388</td></tr><tr><td>src. out rejected</td><td>18</td><td>31</td></tr><tr><td>Pilot items excluded</td><td></td><td></td></tr><tr><td>catch (340)</td><td>96.5%</td><td></td></tr><tr><td>FA (282)</td><td>0.7%</td><td>5.3%</td></tr><tr><td>accepted rejected Other</td><td>2/51</td><td></td></tr><tr><td>reworded question, FA</td><td>6,8</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>anchor in pool, FA</td><td>7</td><td>3</td></tr><tr><td>all 3,000 ideas, rejected</td><td>2.9%</td><td></td></tr><tr><td>hybrids, base passes, kept</td><td>91/99</td><td>74/98</td></tr></table>

Table 22: Further robustness checks. Pilot items excluded: items used to choose the verdict question, judge tier and thresholds. CoI: CoI-Agent.

<table><tr><td>Check</td><td>All gpt-5 (400) (305)</td><td>Agree -101 (292) (299)</td></tr><tr><td>AI Scientist v1</td><td>51.8 42.6</td><td>38.8</td></tr><tr><td>CoI-Agent</td><td>3.8 0.0</td><td>0.0 1.3</td></tr><tr><td>Si et al.</td><td>96.0 95.1</td><td>94.6</td></tr><tr><td>ResearchAgent</td><td>98.5 98.7</td><td></td></tr><tr><td>AI-Sci.-v2</td><td>39.2 29.5</td><td></td></tr><tr><td>OpenNov.-style</td><td>11.5 10.5</td><td>11.7</td></tr><tr><td>W/o profile</td><td>11.8 7.2</td><td>7.4</td></tr><tr><td>CLAIMNOV</td><td>1.5 1.3</td><td>1.4 1.3</td></tr></table>

Table 23: False alarms (%) on later-realized ideas under alternative labels. gpt-5: ideas a gpt-5 labeler confirms as novel. Agree: Opus and gpt-5 agree. −101: excluding 101 ideas a whole-idea pass called already done.

<table><tr><td></td><td>gpt-oss-120b</td><td colspan="2">Llama 4 Maverick</td></tr><tr><td>Measure</td><td>Profile Abstr. Profile</td><td></td><td>Abstr.</td></tr><tr><td>Subset FA (341)</td><td>0</td><td>15 25</td><td>91</td></tr><tr><td>Subset src. out (65)</td><td>0</td><td>7 6</td><td>38</td></tr><tr><td>Subset catch (68)</td><td>56</td><td>60 一</td><td></td></tr><tr><td>Full FA (400)</td><td>1</td><td>25</td><td>26 104</td></tr><tr><td>Full src. out (347)</td><td>7</td><td>31 30</td><td>175</td></tr><tr><td>Full catch (347)</td><td>287</td><td>319 304</td><td>343</td></tr><tr><td>Full J</td><td>+.83</td><td>+.86 +.81</td><td>+.73</td></tr></table>

Table 24: Non-Claude verdict judges. Profile: the judge reads the claim profile. Abstr.: it reads the top-8 abstracts. Subset: prompts byte-identical to the Sonnet 4.6 runs.

<table><tr><td>Baseline</td><td>Catch Src. out</td><td>FA</td><td></td><td>J Hyb.</td></tr><tr><td>BM25, max score Embedding, max cosine</td><td></td><td></td><td>+.57 +.21</td><td></td></tr><tr><td>SciNCL, max cosine SPECTER2, max cosine</td><td></td><td></td><td>+.45</td><td></td></tr><tr><td>Retrieve-and-judge</td><td>378</td><td>65</td><td>+.33  $+ . 8 8$ </td><td>39</td></tr><tr><td>ScholarEval, best thr.</td><td>359</td><td>13429.5%</td><td>47</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $+ . 6 5$ </td></tr><tr><td>CLAIMNOV</td><td>367</td><td>18</td><td> $6 \ + . 9 5$ </td><td></td></tr></table>

Table 25: Simpler baselines on the full sets. Similarity thresholds are cross-validated, and ScholarEval’s is the best threshold on the evaluation data.
<table><tr><td rowspan="2">Rung</td><td colspan="2">CLAIMNOV</td><td colspan="2">CoI-Agent</td></tr><tr><td></td><td>Catch Src. out</td><td></td><td>Catch Src. out</td></tr><tr><td>r1</td><td> $4 0  4 0$ </td><td>0</td><td>一</td><td>一</td></tr><tr><td>r2</td><td> $4 0  3 8$ </td><td>0</td><td></td><td></td></tr><tr><td>r3</td><td> $3 9  3 5$ </td><td>1</td><td>35</td><td>5</td></tr><tr><td>r4</td><td> $3 5  2 8$ </td><td>1</td><td>35</td><td>8</td></tr></table>

Table 26: Catch with pools enlarged from 20 to about 200 papers (40 sources). CoI-Agent is run on rungs 3 and 4 only.

## L Natural Already-Done Ideas

A whole-idea Opus sorter, reading the eight nearest pool papers, matched 837 of the 3,000 generated ideas to a paper that already does the idea. The claim-level labeler overruled it on 101 (these are in the later-realized set), and 407 of the remaining 736 have the shipped profile (Table 27). The set is noisier than the disguises because the sorter read exactly the abstracts the w/o-profile ablation reads. Appendix F.1 reports a human audit of 15 of these ideas.

<table><tr><td>Measure</td><td>W/o profile</td><td>+ abstr.</td><td>CLAIMNOV</td></tr><tr><td>Rejects (407)</td><td>224 (55.0%)</td><td>145 (35.6%)</td><td>72 (17.7%)</td></tr><tr><td>Catch (347)</td><td></td><td>345</td><td>334</td></tr><tr><td>FA (400)</td><td></td><td>25</td><td>6</td></tr><tr><td>Hybrids kept</td><td></td><td>77</td><td>91</td></tr><tr><td>Claim rej. (158)</td><td></td><td>12</td><td>5</td></tr></table>

Table 27: Sorter-matched ideas and the main sets. + abstr.: the verdict reads the profile and the eight nearest abstracts.

## M In-the-Loop Pilot

We plug each check into a generate–check– regenerate loop on two anchors (In-Context Edit and MoBA, 178–185-paper pools, 20 ideas each). Every arm starts from the same first idea, and a rejection returns its reason and cited paper to the generator, with up to three regenerations (Table 28).

A pairwise review-score proxy (Si et al., 2025) separates committed from unchecked ideas on only 3 seeds for CLAIMNOV and 1 for CoI-Agent, too few to compare the arms, so we leave in-the-loop evaluation to future work.
<table><tr><td>Arm</td><td>1st rej. Checks</td><td></td><td>$</td><td>Seconds</td></tr><tr><td>CLAIMNOV</td><td>3/40</td><td>1.08</td><td>0.087</td><td>13.8</td></tr><tr><td>CoI-Agent</td><td>1/40</td><td>1.03</td><td>0.047</td><td>22.2</td></tr><tr><td>No check</td><td></td><td>0</td><td>0</td><td>0</td></tr></table>

Table 28: In-the-loop pilot on 40 ideas, per committed idea.

## N Case Study

In Table 29, we trace two proposals through CLAIMNOV. The first rewrites the ICLR 2022 Spotlight Amortized Tree Generation for Bottomup Synthesis Planning and Synthesizable Molecular Design, whose reviewer attributed its MDP formulation to Barking up the right tree, with that paper in the pool. The second is a rung-4 disguise of Ratio-Variance Regularized Policy Optimization with the source in the pool.

## O Component Validations

We explain the validations we performed on each component in §3.

## O.1 Claim extraction

We extract typed claims from the abstract with gpt-4o-mini (temperature 0, JSON mode) using a GEPA-tuned prompt, scored against a 20-paper, three-annotator ground truth (Table 30). When we extract from introductions rather than abstracts, F1 drops from 0.577 to 0.490, so in CLAIMNOV we use abstracts. gpt-5-mini has equivalent performance on extraction but performs poorly as the entailment judge (Table 35).

## O.2 Same-dimension retrieval

To validate same-dimension retrieval, we embed each claim’s title plus its attribute–value text with text-embedding-3-large, then rank its cosine similarity with same-dimension pool claims. In Table 31 we score recall of the reviewer-named work on 150 ICLR 2022–23 reviewer-tell pairs with A’s own references as distractors, and in Table 32 we look at the value of staying within the same dimension. Because we find that only 3 of the 159 disputes are not in same dimension, we keep same dimension retrieval rather than retrieving over all types of claims.

Claim-level: ICLR Spotlight, one disputed claim. $N ( P ) =$ 0.71 ⇒ pass
<table><tr><td>Dim.</td><td>Claim</td><td>Top matches (level)</td><td> $\nu _ { c }$ </td></tr><tr><td>problem</td><td>Molecular design and synthesis planning as one task of conditional pathway generation</td><td>2 Synthesizability of generated molecules</td><td>0.75</td></tr><tr><td rowspan="2">method</td><td>Amortized generation of synthetic pathways</td><td>3 Barking up the right tree (RetroDog); 2 A</td><td rowspan="2">0.62</td></tr><tr><td>as an MDP conditioned on a target embedding</td><td>model to search for synthesizable molecules; 2 DOGS</td></tr></table>

Judge: “No single paper in the profile proposes the same unified framework . . . The closest works (Barking up the right tree, A Model to Search for Synthesizable Molecules) share partial methodological overlap but do not combine the full set ofcontributions.”

Whole-proposal: rung-4 disguise, source planted. $N ( P ) =$ 0.52 ⇒ reject
<table><tr><td>Dim.</td><td>Claim</td><td>Top matches (level)</td><td> $\nu _ { c }$ </td></tr><tr><td>method</td><td>Distributional- control approach to stability and efficiency in policy</td><td>4 Ratio-Variance Reg. PO; 3 Differentiable trust-region layers; 3</td><td>0.39</td></tr><tr><td>method</td><td>gradients Soft constraint on the second moment of the policy ratio as a trust-region proxy</td><td>Off-policy TRPO 4 Ratio-Variance Reg. PO; 3 Stochastic trust-region framework; 3 Two-phase KL</td><td>0.39</td></tr><tr><td>result</td><td>Substantial gains in sample efficiency and final performance over PPO</td><td>penalization 4 Ratio-Variance Reg. PO; 3 Quasi-Newton TRPO; 3 Tapered off-policy REINFORCE</td><td>0.38</td></tr><tr><td>problem</td><td>Evaluate across LM reasoning and robotic control</td><td>2 Metric-aware trust regions (1 retrieved)</td><td>0.75</td></tr></table>

Judge: not novel; establishing paper Ratio-Variance Reg. PO; established claims 1–3. “Already proposes the same core idea: constraining policy ratio variance as a principled soft alternative to hard clipping, implemented via a primal-dual optimization framework . . . at 4/5 coverage across all core claims.” Source removed: pass .  
Table 29: CLAIMNOV on a claim-level dispute (left), which it passes, and a rung-4 disguise (right), which it rejects. The reviewer’s paper and the source are in blue.

<table><tr><td>Prompt</td><td>Model</td><td>F1</td><td>dim. κ</td><td>cOV.</td></tr><tr><td>Human pair</td><td></td><td>.543</td><td> $. 5 1 5 ^ { \dagger }$ </td><td></td></tr><tr><td>pre-GEPA</td><td>gpt-4o-mini .646 / .667 .596 / .662 41–42</td><td></td><td></td><td></td></tr><tr><td>GEPA</td><td>gpt-4o-mini .674 / .695.793 / .744 38–39</td><td></td><td></td><td></td></tr><tr><td>GEPA</td><td>gpt-5-nano</td><td>.642</td><td>.696</td><td>41</td></tr><tr><td>GEPA</td><td>gpt-5-mini</td><td>.673</td><td>.742</td><td>43</td></tr></table>

Table 30: Extraction against a 20-paper human ground truth (two runs per configuration). Bold: shipped. † Fleiss’ κ. cov.: reviewer-named contributions covered, of 75.

<table><tr><td>Bucket</td><td>n</td><td>@5</td><td>@10</td><td>@13</td></tr><tr><td>Overall</td><td>150</td><td>0.55</td><td>0.79</td><td>0.90</td></tr><tr><td>Problem</td><td>39</td><td>0.62</td><td>0.87</td><td>0.95</td></tr><tr><td>Methodology</td><td>40</td><td>0.62</td><td>0.80</td><td>0.93</td></tr><tr><td>Resource</td><td>28</td><td>0.57</td><td>0.82</td><td>0.93</td></tr><tr><td>Result</td><td>43</td><td>0.42</td><td>0.67</td><td>0.81</td></tr><tr><td>B cited by A</td><td>86</td><td>0.50</td><td>0.79</td><td>0.90</td></tr><tr><td>B uncited</td><td>47</td><td>0.66</td><td>0.77</td><td>0.87</td></tr></table>

Table 31: Recall of reviewer-named prior work, with the target paper’s references as distractors.

<table><tr><td>Configuration</td><td>pool @1</td><td>@5</td><td>@10</td><td></td><td>@13</td></tr><tr><td>D1 claim only, same-dim D2 D1 + title on query D3 D1 + title on candidates D4 title on both (shipped)</td><td>19.7 19.7 19.7 19.7</td><td>.21 .21 .21 .26</td><td>.56 .59 .54 .64</td><td>.75 .82 .78 .84</td><td>.87 .89 .85 .90</td></tr><tr><td>D5 D3, any-dimension pool Oracle: reviewer sentence</td><td>75.4</td><td>.21</td><td>.41</td><td>.52</td><td>.60</td></tr><tr><td>Oracle + facet, titled cands.</td><td>19.7 19.7</td><td>.28 .32</td><td>.57 .69</td><td>.80 .86</td><td>.89 .89</td></tr></table>

Table 32: Recall of reviewer-named prior work by query and candidate representation.

Retrieval depth. From above validation we set depth as $k = \operatorname* { m a x } ( 1 0 , 0 . 2 n _ { d } )$ capped at 40. Table 33 checks this heuristic on the $n = 8 3 7$ sortermatched ideas (pools of up to 400 papers, median 174). In this set, using this depth rule means that only 5 of the 837 matched papers are never retrieved.

## O.3 Graded entailment judge

In Tables 34–37, we evaluate and calibrate the graded entailment stage of CLAIMNOV. Table 34 compares the value of added context (specifically, titles and abstracts) to the value of higher model tiers; we find that context is comparatively more valuable. Table 35 investigates if the graded score alone can separate reviewer-attested novel claims (n = 66) from reviewer-attested established claims (n = 159) across six judge tiers and three retrieval settings, but we find that embedding-only similarity is a strong baseline here. Table 36 compares the judge’s use of the 1–5 entailment scale with that of three human annotators on $n = 5 0$ pairs; we find every tier agrees with the human majority more than the annotators agree with each other. Table 37 tests whether our graded score can successfully rank a claim a reviewer specifically said was novel as higher than the sibling claims from the same paper (n = 86 papers and n = 233 claims); we find that it does so above chance, 59% of the time. All in all, we observe that the graded score can order claims well, but that there isn’t a good way to set a pass/fail novelty threshold based off the graded score.

<table><tr><td>Fixed k</td><td>5</td><td>10</td><td>20</td><td>30</td><td>40</td><td>80</td></tr><tr><td>In profile</td><td>44%</td><td>57%</td><td>69%</td><td>77%</td><td>82%</td><td>91%</td></tr><tr><td>Share of cands.</td><td>5% </td><td>10%</td><td>15%</td><td>20%</td><td>30%</td><td></td></tr><tr><td>In profile</td><td>46%</td><td>60%</td><td>69%</td><td>75%</td><td>83%</td><td></td></tr></table>

Table 33: Share of sorter-matched ideas whose matched paper enters the profile, by retrieval depth.

<table><tr><td>Prompt</td><td>Judge</td><td>gold mean</td><td>separation</td></tr><tr><td>facets only</td><td>gpt-4o-mini</td><td>2.43</td><td>0.57</td></tr><tr><td>+ title, abstract</td><td>gpt-4o-mini</td><td>2.70</td><td>0.93</td></tr><tr><td>+ title, abstract</td><td>Haiku 4.5</td><td>2.70</td><td>0.97</td></tr><tr><td>+ title, abstract</td><td>Sonnet 4.6</td><td>2.87</td><td>1.03</td></tr><tr><td>+ title, abstract</td><td>Opus 4.6</td><td>3.07</td><td>1.07</td></tr><tr><td>+ title, abstract</td><td>gpt-5-nano</td><td>2.97</td><td>0.87</td></tr><tr><td>+ title, abstract</td><td>gpt-5-mini</td><td>2.80</td><td>0.67</td></tr></table>

Table 34: Prompt context versus judge tier (30 gold and 30 distractor pairs).

<table><tr><td>Judge</td><td>base k=5</td><td>titled k=5</td><td>titled k=10</td></tr><tr><td>gpt-5-nano</td><td>.460</td><td>.451</td><td>.440</td></tr><tr><td>gpt-5-mini</td><td>.438</td><td>.503</td><td>.443</td></tr><tr><td>gpt-4o-mini</td><td>.516</td><td>.551</td><td>.546</td></tr><tr><td>Haiku 4.5</td><td>.539</td><td>.564</td><td>.527</td></tr><tr><td>Sonnet 4.6</td><td>.589</td><td>.606</td><td>.611</td></tr><tr><td>Opus 4.6</td><td>.576</td><td>.601</td><td>.594</td></tr><tr><td>embedding-only</td><td>.607</td><td>一</td><td></td></tr></table>

Table 35: AUC of the graded score separating novel from established claims. Titled: the shipped retrieval representation.

<table><tr><td>Judge</td><td>prompt</td><td>α</td><td>MAD</td><td>exact ±1</td><td></td></tr><tr><td>Human pair</td><td></td><td>.42</td><td>.95</td><td>.41</td><td>.74</td></tr><tr><td>gpt-4o-mini</td><td>claim</td><td>.73</td><td>.62</td><td>.43</td><td>.95</td></tr><tr><td>gpt-4o-mini</td><td>+ titles</td><td>.73</td><td>.57</td><td>.49</td><td>.95</td></tr><tr><td>Haiku 4.5</td><td>claim</td><td>.86</td><td>.35</td><td>.68</td><td>.97</td></tr><tr><td>Haiku 4.5</td><td>+ titles</td><td>.84</td><td>.38</td><td>.65</td><td>.97</td></tr><tr><td>Sonnet 4.6</td><td>claim</td><td>.78</td><td>.49</td><td>.57</td><td>.95</td></tr><tr><td>Sonnet 4.6</td><td>+ titles</td><td>.75</td><td>.54</td><td>.51</td><td>.95</td></tr><tr><td>Opus 4.6</td><td>claim</td><td>.74</td><td>.54</td><td>.51</td><td>.95</td></tr><tr><td>Opus 4.6</td><td>+ titles</td><td>.57</td><td>.73</td><td>.35</td><td>.92</td></tr></table>

Table 36: Graded entailment against the human majority label (37 pairs). The human row is agreement between annotators.

<table><tr><td>Scorer</td><td>win rate</td><td>95% CI</td><td>claim FA</td></tr><tr><td>Graded, gpt-4o-mini</td><td>0.591</td><td>[0.525, 0.655]</td><td></td></tr><tr><td>Binary veto, Sonnet 4.6</td><td>0.500</td><td>floor</td><td>0/233</td></tr></table>

Table 37: Win rate of a reviewer-praised claim over its sibling claims.

## O.4 Verdict

Calibration. We use n = 20 disputed pairs, n = 32 later-realized decoys, and n = 106 alreadydone ideas to try and calibrate a particular novel or non-novel threshold for N(P), as we show in Table 38. We consider a knockout entailment level, relation labels, and a per-candidate question; the verdict stage we ultimately adopted considers novelty once over the whole profile (§3.5), while the per-candidate question is the per-claim ablation of Table 1.

<table><tr><td>Rule</td><td>Judge</td><td>N2 (a)</td><td>(b) FA</td></tr><tr><td>t=.5, ko=4*</td><td>4o-mini</td><td>0% 0%</td><td>1%</td></tr><tr><td>t=.6, no ko</td><td>4o-mini</td><td>15% 46%</td><td>30%</td></tr><tr><td>ko=3, any t</td><td>4o-mini</td><td>75% 82%</td><td>78% 3%</td></tr><tr><td>t=.5, ko=4</td><td>Sonnet 4.6 10%</td><td></td><td>72%</td></tr><tr><td>ko=3, any t</td><td>Sonnet 4.690%</td><td></td><td></td></tr><tr><td>rel. same/subsumes</td><td>4o-mini</td><td>5%</td><td>100%</td></tr><tr><td>rel. + partial rel. same/subsumes</td><td>4o-mini Sonnet 4.6</td><td>0%</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>veto</td><td>4o-mini</td><td>45% 60%</td><td>56%</td></tr><tr><td>veto</td><td>Haiku 4.5</td><td>20%</td><td>0%</td></tr><tr><td>veto</td><td>Sonnet 4.635%</td><td></td><td>0%</td></tr><tr><td>veto</td><td>Opus 4.6</td><td>15%</td><td>0%</td></tr><tr><td>veto, outcome-reworded Sonnet 4.6 10%</td><td></td><td></td><td>0%</td></tr></table>

Table 38: Pilot calibration of decision rules (t: threshold on N(P), ko: knockout level, ∗: shipped). N2: catch on 20 disputed pairs. (a): catch on 106 already-done ideas. (b) FA: false alarms on 32 decoys.

## P AI Disclosure Statement

LLM agents (e.g., Claude Code) were used to assist with experiment and analysis code, to advise on paper structure and clarity, to generate results tables and draft appendix text, and to prepare released data and code files. All scientific contributions, including the CLAIMNOV methodology, experimental design, and data collection approaches, were devised by the authors.