# WARP: Wasserstein-Aligned RAG for Population Opinions

Aman Singh Thakur\* Aditya Agrawal Alwarappan Nakkiran Alex Karlsson

Amazon.com

## Abstract

RAG systems are increasingly used to summarize what large collections of documents say. A user asks “What do people think about X?” and receives an answer that reads as consensus. But standard top-k retrieval ranks documents by query similarity, not by how faithfully they represent the population, so minority views quietly disappear. Existing fixes fall short. Diversity re-rankers like MMR and DPP spread retrieved documents apart, but with no target distribution to aim for. Calibration methods based on KL or JS divergence do target one, yet treat opinion bins as unordered: confusing strong positive with strong negative costs no more than an adjacent-bin miss.

We introduce WARP, a family of post-retrieval algorithms that calibrate retrieved evidence to the population’s opinion distribution. WARP first recovers underrepresented opinions that cosine ranking may bury, then uses Wasserstein-1 distance to select documents whose sentimentintensity distribution matches the population target, capturing the ordinal structure ignored by KL and JS divergence. We develop three variants for dense, sparse, and variable candidate pools, trading off calibration quality and speed. Across three review domains spanning 35K documents, 156 queries, and 26 entities, WARP’s domain-matched variants reduce distributional error by at least 43% with subsecond latency. These gains carry through to generation: a five-judge LLM panel prefers WARP-generated answers in 86% of decided comparisons at k ≤ 5.

## 1 Introduction

Users used to browse reviews themselves - clicking through results, weighing contradictions, building a mental picture. Noisy, but transparent: you could see disagreement. RAG systems (Lewis et al.,

2020) replace this with a single synthesized answer, but a new failure mode emerges. Standard top-k retrieval selects evidence by query similarity, without factoring spread of opinions in the corpus. Take a hotel where 60% of reviewers are satisfied and 40% complain about noise. Retrieval over-represents the majority because similarity scoring favors the dominant cluster. Every claim in the resulting summary traces to a real review, yet the user never learns that two in five guests had a bad experience. This leads to distributional distortion. A synthesized response from a skewed sample would read factual but would not present the whole picture.

For opinion queries, uncertainty is aleatoric: it reflects genuine disagreement. A faithful system must preserve the distribution without collapsing it. Agrawal et al. (2026) formalize this distinction; Nayeem and Rafiei (2025) build a scalable opinion summarization pipeline but do not optimize the retriever for distributional fidelity. What’s missing is the engineering: in a production stack with sub-second latency budgets, how do you select a small evidence set whose opinion distribution actually matches the population?

A standard cosine retriever maximizes query similarity, producing relevant but homogeneous evidence - four glowing reviews when two would suffice. Diversity re-rankers (MMR (Carbonell and Goldstein, 1998), DPP (Kulesza and Taskar, 2012)) push back by maximizing pairwise spread, which helps until each sentiment bin has one representative; past that point they quietly revert to relevance ordering. KL and JS calibration (Steck, 2018; Dang and Croft, 2012) come closer by explicitly optimizing against a target distribution. But they remain categorical. Confusing “positive” with “negative” costs the same as confusing ”positive” and ”neutral”. Opinions are ordinal and we need a metric which supports this.

Wasserstein-1 respects this ordering. Its closedform CDF computation (O(m) for m sentiment bins) makes per-candidate evaluation cheap enough for runtime re-ranking without offline precomputation. We built WARP (Figure 1) around this in two stages. First, a deficit-aware pool expansion pass compares the retrieved distribution against the population target, identifies under-represented poles, and issues targeted re-retrievals only where gaps exist. Then a greedy re-ranker selects the final evidence set by minimizing $W _ { 1 }$ distance to $P _ { \mathrm { p o p } }$ . The pipeline instantiates in three variants (Section 3.4): the Minimizer greedily selects from entity-matched candidates - effective for dense pools; for sparser conditions, W<sub>1</sub>-MMR blends relevance with calibration, and WassRank OT solves the full slotassignment as rectangular optimal transport requiring no per-domain tuning.

![](images/7a93438640dabd397f5067c85d76e5871f90ee13609cc639d21b23285eb13556.jpg)  
Figure 1: WARP pipeline. Stage 1: semantic retrieval returns a Top-N candidate pool. Stage 1.5: deficit-aware pool expansion recovers under-represented sentiment poles via re-retrieval (Entity-Gated or Adaptive Expansion; §3.3; self-bypasses when the pool is already balanced). Stage 2: $W _ { 1 }$ re-ranking selects the final k documents whose empirical opinion distribution matches the population target $P _ { \mathrm { p o p } }$ (W<sub>1</sub> Minimizer for dense pools, W<sub>1</sub>-MMR for sparse pools, or WassRank OT as a tuning-free fallback; §3.4). Pool density drives the variant choice (decision matrix in Table 3).

Our contributions:

1. Pool expansion that recovers what cosine retrieval buries. Entity-gated and adaptive re-retrieval recover under-represented opinions with near-complete entity coverage. Selfbypasses at zero cost when already balanced.

2. W<sub>1</sub> re-rankers ensure faithful population representation. A greedy minimizer that matches opinion distribution for entity dense domains and the hybrid variants trade opinion diversity for relevance on sparse domains.

3. Deployment characterization across 3 domains. Pool density determines algorithm choice. All variants add under 330 ms reranking latency (p99) and generalize across labeling methods and index types.

## 2 Related Work

Retrieval diversity. MMR (Carbonell and Goldstein, 1998) penalizes redundancy via pairwise dissimilarity; DPP (Kulesza and Taskar, 2012) maximizes determinantal spread; xQuAD (Santos et al., 2010) diversifies over query intents. More recent variants-BQP (Lu and Sidiropoulos, 2026), Ada-GReS (Peng et al., 2025), MUSS (Nguyen and Kan, 2026)-refine these ideas in various ways (see Wu et al. (2024) for a survey). However, diversity is the wrong objective here as the signal runs out, once every sentiment bin has one representative (5–7 selections).

Optimal transport (OT) in IR. Wasserstein distances (Peyré and Cuturi, 2019) have appeared in IR before-Word Mover’s Distance (Kusner et al., 2015) for document similarity, WassRank (Yu et al., 2019) as a listwise training loss, OTExtSum (Tang et al., 2022) for extractive summarization, Wasserstein coresets (Claici et al., 2018) for data summarization. However, all these function at training time, over generic content. None use Wasserstein as a runtime selection objective conditioned on an opinion distribution.

![](images/b33f3d5f502668353698373114acbcf723a80e112c356e53744788130d64f39d.jpg)  
Figure 2: $W _ { 1 }$ re-rankers (no re-retrieval) reduce distributional error at least 43% vs. Top-k (gray dashed) across three domains — 43% Seller Forums $( W _ { 1 }$ -MMR), 79% Yelp (W Minimizer), 69% OpinRank $( W _ { 1 }$ Minimizer); see Table 1. Curves converge at $k ^ { * } { = } 3 { - } 5$ documents (black dot; Appendix C.3) as most entities’ opinion mass concentrates in 2–3 of 7 bins. $N { = } 2 0 0$

Opinion-aware retrieval. Agrawal et al. (2026) audit 30+ RAG benchmarks and find a striking gap: only one addresses opinion synthesis. They formalize the epistemic/aleatoric distinction - the posterior converges to the population distribution $P _ { \mathrm { p o p } } ( \theta )$ for opinion queries - and propose a three-term objective combining coverage $( W _ { 2 } )$ , fidelity, and demographic fairness. However, their work is limited to 2 domains without practical retrieval strategies. Nayeem and Rafiei (2025) solve a different problem entirely: generating opinion highlights from thousands of reviews via retrieve-then-synthesize with AOS-triplet verification. Their retriever is standard semantic search without targeting distributional fidelity. We instantiate the Agrawal et al. (2026) coverage objective using $W _ { 1 }$ instead of $W _ { 2 }$ - its closedform CDF difference enables $O ( m )$ per-candidate evaluation at runtime - and test across three domains.

Calibrated selection and diversity methods. Calibrated recommendation via KL minimization (Steck, 2018) and proportional slot allocation (PM-2; Dang and Croft 2012) do address distributional fidelity across facets - but assume categorical labels. Fairness-aware ranking (Singh and Joachims, 2018; Zehlike et al., 2017; Oesterling et al., 2024) enforces demographic constraints without modeling ordinal opinion at all. On the sentiment side,

Aktolga and Allan (2013) applies bias modes to retrieval, aspect-level summarization (Angelidis and Lapata, 2018; Jiang et al., 2023) diversifies across topical facets. All of these methods use categorical penalty structures that do not exploit ordinal distance between opinion bins. WARP addresses this gap: an ordinal $W _ { 1 }$ ground cost applied at query time to opinion-distribution matching in RAG, drawing on the calibrated-selection tradition (Steck, 2018) and OT-in-ranking (Yu et al., 2019) while adapting both to a training-free runtime re-ranker across three review domains.

## 3 Methodology

Below we describe the experimental setup (Sections 3.1–3.2), pool expansion (Section 3.3), and $W _ { 1 }$ re-ranking (Section 3.4) for stress-testing WARP.

## 3.1 Datasets

3 review corpora give us the spread we need across different domains:

• Amazon Seller Forums<sup>1</sup> (∼8K posts). Retrieved via Amazon Bedrock Knowledge Bases (Amazon Web Services, 2024b) with hybrid search.

• Yelp Open Dataset (Yelp, 2024) (∼14K hotel reviews). Retrieved via local FAISS (Johnson

et al., 2019) + MiniLM-L6 (Wang et al., 2020;   
Reimers and Gurevych, 2019).

• OpinRank (Ganesan and Zhai, 2012) (∼13K automotive reviews). Retrieved via local FAISS + MiniLM-L6.

Entity selection follows Agrawal et al. (2026): LLM-based entity extraction, diversity filtering (Shannon entropy $H \ge 0 . 6 .$ , minority sentiment ≥10%), then ranking by entropy.

We use six questions per entity (2 breadth, 2 polar, 2 segment; templates in Appendix A.2) which gives us 156 total queries, across all domains. Entity and sentiment-intensity labels come from a single LLM pass per document (prompt in Appendix A.3.1), mapped to a 7-bin ordinal scale $\mathcal { S } ~ = ~ \{ - 3 0 , - 2 0 , - 1 0 , 0 , + 1 0 , + 2 0 , + 3 0 \}$ (Table 4, Appendix A.1).

Full dataset statistics appear in Table 5 (Appendix A.1). We also ablate with VADER (Hutto and Gilbert, 2014) lexicon scores on OpinRank to confirm labeling-method independence $( \mathsf { A p } \cdot$ pendix F.1).

Defaults: N=200 candidate pool, k=20 output.

## 3.2 Baselines

Five baselines. Top-k is plain cosine-similarity retrieval. MMR (Carbonell and Goldstein, 1998) penalizes semantic redundancy (λ=0.5). DPP (Kulesza and Taskar, 2012) maximizes determinantal spread over 1D sentiment-intensity features. OpinionMMR penalizes opinion-bin distance rather than embedding distance. KL (JS) Minimizer implements Steck (2018)’s calibratedrecommendation objective with Jensen–Shannon divergence - identical to our $W _ { 1 }$ Minimizer but with an order-agnostic metric (Appendix C.1).

Metrics. Distributional fidelity: Wasserstein-1 distance $W _ { 1 }$ (lower is better). Entity relevance: Entity Match rate (EM%, hereafter EM; fraction of selected documents matching the queried entity, higher is better). We also report re-ranking latency in milliseconds. For generation evaluation, Cohen’s κ (Cohen, 1960) measures inter-judge alignment.

## 3.3 Pool Expansion via Re-Retrieval

For entity sparse domains, the relevance-based retrieval step is expected to include a % of documents which do not discuss the queried entity - due to near-misses leaked in from other entities sharing the same index. We tackle this with two expansion strategies (full algorithms in Appendix B):

Algorithm 1 $\overline { { W _ { 1 } } }$ Minimizer   
Require:Pool C,entity e,target $P _ { \mathrm { p o p } } { \mathrm { : } }$ output size k   
1: $C _ { e } \gets \{ d \in C : \mathrm { e n t i t y } ( d ) = \bar { e } \}$   
2: $S \gets \emptyset$   
3: while $| S | <$ k do   
4: $d ^ { * } \gets$ arg mi $1 _ { d \in C _ { e } \backslash S } W _ { 1 }$ (EmpDist(S ∪   
$\{ d \} ) , P _ { \mathrm { p o p } } )$   
5: ties broken by relevance score ↓   
6: $S \gets S \cup \{ d ^ { * } \}$   
7: end while   
8: return S

Entity-Gated Re-retrieval. Pass 1 pulls the standard top-N by relevance. Pass 2 then digs into the tail of the ranked list, fishing out documents that match the queried entity but fell below the cutoff, ranking them by opinion extremity (absolute sentiment-intensity score - strongly negative reviews surface before mild ones). The passes are merged and deduplicated.

Adaptive Expansion. We compare the pool’s opinion breakdown against $P _ { \mathrm { p o p } } .$ . If any bin is under-represented (pool fraction below threshold of the population fraction), a targeted retrieval fires for entity-matched documents in the deficit bin using pole-biased queries (Prompt A.3.4). If the pool already mirrors the target, nothing happens - zero overhead (Algorithm 3).

## 3.4 Wasserstein Re-Ranking Algorithms

Problem Formulation. For opinion queries, a faithful answer must reflect how viewpoints distribute along an ordinal axis - one where positions have a natural ordering and distances between them reflect severity of disagreement. The re-ranking objective follows directly: select k documents whose empirical distribution along this axis minimizes transport distance to the target distribution $P _ { \mathrm { p o p } } .$ We instantiate the axis as sentiment-intensity (SI), a 7-bin ordinal scale $\mathcal { S } = \{ - 3 0 , - 2 0 , \dots , + 3 0 \}$ At indexing time each document d receives an SI label; per entity e we derive $P _ { \mathrm { p o p } } ( s )$ as the corpusobserved fraction of documents for e in each bin, or as an externally-specified target (star ratings, survey-calibrated priors) when one is available — $P _ { \mathrm { p o p } }$ is an input to the re-ranker, not a claim about the true underlying population (§Limitations; tolerance to partial estimates in Section 4.4). Given a candidate pool $C$ of N documents, pick $k \ll N$ into result set $S$ minimizing:

Table 1: Main results: Diversity baselines and calibrated methods vs WARP re-rankers. $W _ { 1 \downarrow : }$ distributional distance to the population opinion target. EM%↑: fraction of selected documents matching the queried entity. EG/AE = pool expansion strategies (Entity-Gated / Adaptive Expansion). $N { = } 2 0 0$ candidates, k=20 output. Paired Wilcoxon $\mathrm { v s . \ T o p { - } } k \mathrm { : ~ } ^ { \ast \ast \ast } p { < } 0 . 0 0 1 , \mathrm { ~ } ^ { \ast \ast } p { < } 0 . 0 1 , \mathrm { ~ } ^ { \ast } p { < } 0 . 0 5 .$ . Bold: best per column. <sup>†</sup>Shared FAISS index across all OpinRank entities (EM=82.9%); entity-specific indices give $W _ { 1 } { = } 0 . 9 5$ with 100% EM (Appendix F.3).
<table><tr><td></td><td colspan="2">Seller Forums</td><td colspan="2">Yelp Hotels</td><td colspan="2">OpinRank Cars</td><td rowspan="2">Latency (ms)</td></tr><tr><td>Method</td><td> $W _ { 1 \downarrow }$ </td><td>EM%</td><td> $W _ { 1 \downarrow }$ </td><td>EM%</td><td> $W _ { 1 \downarrow }$ </td><td>EM%</td></tr><tr><td>Top-k (baseline)</td><td>10.33</td><td>42.9</td><td>13.03</td><td>40.1</td><td>13.33</td><td>27.0</td><td>0</td></tr><tr><td>MMR</td><td>10.31</td><td>43.5</td><td>8.67</td><td>14.2</td><td>9.64</td><td>11.3</td><td>12</td></tr><tr><td> $\mathrm { D P P } ^ { * * * }$ </td><td>11.69</td><td>41.2</td><td>5.63</td><td>78.8</td><td>7.38</td><td>69.2</td><td>45</td></tr><tr><td> ${ \mathrm { O p i n i o n M M R } } ^ { * * }$ </td><td>10.44</td><td>43.5</td><td>8.59</td><td>12.4</td><td>10.02</td><td>10.6</td><td>12</td></tr><tr><td>KL (JS) Minimizer* 2</td><td>8.87</td><td>51.4</td><td>3.00</td><td>93.8</td><td>4.54</td><td>85.8</td><td>9</td></tr><tr><td> $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R } ^ { \ast \ast \ast }$ </td><td>5.86</td><td>44.6</td><td>4.02</td><td>19.4</td><td>3.41</td><td>21.8</td><td>154</td></tr><tr><td> $W _ { 1 } ~ \mathrm { M i n i m i z e r } ^ { * * * }$ </td><td>8.84</td><td>51.4</td><td>2.76</td><td>93.8</td><td>4.14</td><td>85.8</td><td>9</td></tr><tr><td> $\mathbf { W a s s R a n k { O T } ^ { * * * } }$ </td><td>5.71</td><td>44.4</td><td>2.52</td><td>92.2</td><td>3.87†</td><td>82.9</td><td>89</td></tr><tr><td> $\boldsymbol { \mathrm { E G } } + \boldsymbol { W _ { 1 } } \ : \boldsymbol { \mathrm { M i n } } ^ { * * * }$ </td><td>8.59</td><td>63.5</td><td>1.52</td><td>99.2</td><td>4.14</td><td>85.8</td><td>13</td></tr><tr><td> $\mathbf { A } \mathbf { E } + W _ { 1 } \mathbf { M i n } ^ { * * * }$ </td><td>8.59</td><td>63.5</td><td>1.66</td><td>95.9</td><td>4.14</td><td>85.8</td><td>8</td></tr></table>

$$
W _ { 1 } ( P , Q ) = \sum _ { i = 1 } ^ { m - 1 } \left| \mathrm { C D F } _ { P } ( s _ { i } ) - \mathrm { C D F } _ { Q } ( s _ { i } ) \right| \cdot \Delta s\tag{1}
$$

where CDF is the cumulative distribution function, $m { = } 7$ bins, and $\Delta s { = } 1 0$ is the bin spacing. The ordinal structure matters: confusing +30 with $- 3 0$ costs $W _ { 1 } ^ { \mathrm { m a x } } { = } 6 0$ , while adjacent-bin errors cost only 10. Evaluation runs in $O ( m )$ per candidate; rank ordering is preserved under $W _ { 2 } ( { \mathsf { A p - } }$ pendix C.4).

We implemented three variants that minimize this objective. They differ in how they cope with entity pool density and the relevance-calibration tradeoff:

$W _ { 1 }$ Minimizer. Filter the pool to entity-matched candidates, then greedily add whichever document pulls the empirical distribution closest to $P _ { \mathrm { p o p } } ,$ breaking ties by relevance score (Algorithm 1). Needs a dense pool where most candidates match the queried entity.

$W _ { \mathrm { 1 } } { \cdot } \mathbf { M } \mathbf { M } \mathbf { R } .$ Built for sparse pools where entitymatched candidates alone cannot fill k slots. This variant works on the full candidate set - no entity filtering - and scores each candidate by blending retrieval relevance with the calibration gain it would contribute:

$$
\mathsf { s c o r e } ( d ) = \lambda \cdot \mathsf { r e l } ( d ) + ( 1 - \lambda ) \cdot \frac { \Delta W _ { 1 } } { W _ { 1 } ^ { \mathrm { c u r } } }\tag{2}
$$

rel(d) is the original retrieval score, λ controls the relevance-calibration blend, and dividing by $W _ { 1 } ^ { \mathrm { c u r } }$ normalizes to percentage improvement so the calibration term does not vanish as the distribution tightens. Full algorithm in Appendix B.3.

WassRank OT. Instead of greedy selection, WassRank (Yu et al., 2019) solves the assignment globally: allocate k slots proportionally to $P _ { \mathrm { p o p } } ,$ then find the minimum-cost candidate-to-slot bijection where cost blends normalized ordinal distance with a relevance penalty. This minimizes blended transport cost to a proportional discretization of $P _ { \mathrm { p o p } }$ , which empirically tracks $W _ { 1 } ~ ( { \mathsf { A p } } \cdot$ pendix F.3). We solve the rectangular assignment (Crouse, 2016) in $O ( k ^ { 2 } \cdot N )$ . Originally a listwise training loss; we repurpose it here as an inference-time re-ranker with no learned parameters and call it WassRank OT.

## 4 Results

Four questions drove our experiments: does it work, does the metric choice matter, do retrieval gains actually reach the user, and which variant belongs where? Sections 4.1–4.4 take them in order.

## 4.1 Main Results

In Table 1, the gains are large and consistent, concentrating on entities with the most skewed baseline distributions (per-entity breakdowns in Appendix C.5). Our three $W _ { 1 }$ re-ranking algorithms (Section 3.4) cut distributional error (Equation 1) by at least 43% relative to Top-k across all three domains — 43% on sparse Seller Forums $( W _ { 1 } { \cdot } \mathbf { M } \mathbf { M } \mathbf { R } )$ , 79% on dense Yelp $( W _ { 1 }$ Minimizer, rising to 88% with pool expansion), and 69% on OpinRank $( W _ { 1 }$ Minimizer). Entity relevance tracks along: pairing Entity-Gated re-retrieval (Section 3.3) with $W _ { 1 }$ Minimizer pushes Yelp EM% from 40.1% (Top-k baseline) to 99.2% (postreranking). As a sanity check, 5% Gaussian perturbation of retrieval scores yields only 2–3% $W _ { 1 }$ improvement, non-significant. This confirms that the score structure drives these results and not the arbitrary reshuffling. (Appendix F.2)

## 4.2 The Ordinal Metric Makes a Measurable Difference

The metric matters. Swapping JS divergence for $W _ { 1 }$ while keeping the same greedy loop and entity filtering cuts error by 8.7% on Yelp and 9.7% on OpinRank (Appendix C.1). JS penalizes a $+ 3 0  - 3 0$ confusion no more than an adjacentbin slip; $W _ { 1 }$ charges six times more. Entity filtering alone accounts for 43.4% $W _ { 1 }$ reduction on Yelp; $W _ { 1 }$ Minimizer adds 62.6% beyond that, and on entity sparse domains, like Seller Forums, $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ adds 37.7% beyond the entity-filter baseline $( \mathsf { A p - }$ pendix C.2). We also stress-tested under noise: corrupting both SI labels and $P _ { \mathrm { p o p } }$ simultaneously, $W _ { 1 }$ Minimizer never falls below Top-k even at 50% label error on dense pools, and still outperforms Top-k by 72% at 20% Dirichlet perturbation $( \mathsf { A p - }$ pendix F.1).

## 4.3 Retrieval Gains Propagate to Generation

Do retrieval gains actually reach users? We generated answers from each method’s retrieved evidence (prompt in Appendix A.3.2) and ran blind pairwise comparisons against Top-k using a 5- judge LLM panel (Claude Sonnet 4 (Anthropic, 2025), Llama 3.3 70B (Meta, 2024), Mistral Large 3 (Mistral AI, 2025), Amazon Nova Pro (Amazon AGI, 2025), DeepSeek v3.2 (DeepSeek-AI, 2025)) with position control (Zheng et al., 2023) and 3/5 majority vote (Appendix G.1). All five judges from independent model families agree directionally at $\kappa _ { d } { = } 0 . 6 1 { - } 1 . 0 0 ;$ the random baseline correctly shows no significance.

$\mathrm { A t } \ k { = } 5$ , calibration methods win 70–89% of decided comparisons (Table 2). At k=10 this drops to 73–88% (Table 20) as larger context lets the LLM average out distributional imbalance $( \mathsf { A p - }$ pendix G.1.3). A prompt ablation (Appendix G.1.4) disentangles two evaluation criteria: when judges assess proportional accuracy (does the answer reflect how common each view is?), $W _ { 1 }$ Minimizer leads DPP by $8 \mathrm { p p }$ and matches an oracle stratifiedsampler within 1 pp (Table 21); under pure viewpoint coverage (does it mention all perspectives?), the gap narrows to parity. $W _ { 1 } \ ' _ { \mathrm { s } }$ proportional advantage persists through generation, not just retrieval.

Table 2: Generation evaluation (k=5): 5-judge majority vote, position-controlled blind pairwise vs. Top-k $( N { = } 1 5 6$ queries). $\mathrm { F a i r } \% = \mathrm { W } / ( \mathrm { W } { + } \mathrm { L } )$ $\mathrm { W i n } \% = \mathrm { W } / N$ All approaches significant at p<0.001 (sign test) except Random. $k { = } 1 0$ in Appendix G.1.3.
<table><tr><td>Approach</td><td>W/L/T</td><td>Fair%</td><td>Win%</td></tr><tr><td>Random</td><td>16/36/104</td><td>31%</td><td>10%</td></tr><tr><td>MMR*</td><td>39/8/109</td><td>83%</td><td>25%</td></tr><tr><td> $\mathrm { D P P ^ { * } }$ </td><td>54/7/95</td><td>89%</td><td>35%</td></tr><tr><td>WassRank  $\mathrm { O T ^ { * } }$ </td><td>66/28/62</td><td>70%</td><td>42%</td></tr><tr><td> $\mathbf { W _ { 1 } { - } M M R ^ { * } }$ </td><td>44/10/102</td><td>81%</td><td>28%</td></tr><tr><td> $\mathbf { W } _ { 1 } ~ \mathbf { M i n i m i z e r } ^ { * }$ </td><td>51/8/97</td><td>86%</td><td>33%</td></tr></table>

Metric isolation. On sparse pools, $W _ { 1 }$ -MMR appears to lose to KL(JS)-MMR (Appendix C.1), but the comparison is confounded: KL(JS)-MMR selects $1 . 7 \mathrm { - } 2 \times$ more entity-matched documents than $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ on Yelp and OpinRank. Once we control for EM%, the Minimizer comparison flips: $W _ { 1 }$ wins on both domains (Appendix C.1), and a singlejudge generation evaluation confirms the same direction, with $W _ { 1 }$ Minimizer leading KL(JS) Minimizer by 7 pp Fair% cross-domain (Appendix G.2).

Two other patterns are worth noting. First, semantic-diversity baselines fade as k grows: MMR and OpinionMMR Fair% collapse to 29– 33% at $k { = } 2 0$ , while distributional methods keep gaining (WassRank OT: 70%→73%→92% across $k \in \{ 5 , 1 0 , 2 0 \}$ ; Appendix G.1.3). Second, DPP and $W _ { 1 }$ optimize different targets — bin diversity vs. bin ratios — which happen to align at k=5 with 2–3 active bins but diverge at k=20. On OpinRank, near-uniform $P _ { \mathrm { p o p } }$ compresses ordinal distances, so the $\mathrm { K L } / W _ { 1 }$ gap narrows; the ordinal advantage is largest where distributions are skewed (Yelp, Seller Forums).

## 4.4 Deployment Characterization

Which algorithm to pick comes down to one number: entity match percentage (Table 3). Dense pools, where all retrieved documents discuss the query entity, are easy. When EM≥85%, $W _ { 1 }$ Minimizer delivers 69–81% reduction, statistically indistinguishable from oracle stratified sampling.

Table 3: Deployment decision matrix. Pool density determines algorithm choice; all methods SLA-compliant $( N { = } 2 0 0 , k { = } 2 0 , m { = } 7 { \mathrm { b i n s } } )$ $| C _ { e } | = \mathrm { e n t i t y }$ -matched candidates. Noise tolerance details in Appendix F.1.
<table><tr><td>Pool Condition</td><td>Method</td><td>ms</td><td>Complexity</td></tr><tr><td>Dense (EM≥85%)</td><td>W1 Minimizer</td><td>9</td><td> $O ( | C _ { e } | \cdot k \cdot m )$ </td></tr><tr><td>Sparse (baseline EM~40%)</td><td> $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ </td><td>154</td><td> $O \ddot { ( \cal N \cdot k \cdot m ) }$ </td></tr><tr><td>Variable / unknown</td><td> $\mathrm { E G } + W _ { 1 } \mathrm { M i n }$ </td><td>13</td><td> $O \dot { ( } | C _ { e } | \cdot k \cdot \stackrel { \cdot } { m } )$ </td></tr></table>

Entity-Gated re-retrieval pushes Yelp further $( W _ { 1 } ;$ $2 . 7 6 {  } 1 . 5 2$ , EM%: 93.8→99.2); on OpinRank both expansion strategies detect no deficit and selfbypass. Sparse entity pools are a different story. On Seller Forums (Top-k baseline EM∼40%; $W _ { 1 }$ Minimizer reaches ∼51% post-reranking), $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ scores all 200 candidates regardless of entity match, reaching 43% reduction while maintaining 81% generation propagation (Table 2). This outperforms WassRank OT (70%) despite comparable retrievalside $W _ { 1 }$ , making $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ the preferred hybrid for sparse and variable pools. Re-ranking latency stays under 330 ms across the board; end-to-end depends on retrieval infrastructure (Appendix B.5). For dense pools $N { = } 1 0 0$ suffices; $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ benefits from larger N on sparse pools (Appendix F.3).

An exact $P _ { \mathrm { p o p } }$ turns out to be unnecessary: fifty entity-matched documents preserve 74–98% of oracle improvement; smoothing with a domain prior (τ=50) matches the oracle’s harm rate. Cold-start entities fall back to this prior while still beating Top-k (Appendix F.4). Label source is flexible - both LLM-extracted and VADER work, as do managed cloud vector stores and local FAISS indices.

## 4.5 Independent-Target Validation

A natural concern is circularity: our LLM-derived sentiment-intensity (SI) labels define both the population target $P _ { \mathrm { p o p } }$ used at inference and the $W _ { 1 }$ metric applied to the retrieved evidence. If the labeler had systematic biases, calibrating and measuring against it could inflate the numbers. To rule this out, we cross-validate against Yelp’s native 1–5 star ratings on all 13,533 reviews. Stars are ordinal, human-authored at review time, and participate in no part of the WARP pipeline — neither retrieval, nor re-ranking, nor the $P _ { \mathrm { p o p } }$ target. Mapped to the same 7-bin SI scale, star-derived and LLM-derived labels correlate at Spearman $\rho { = } 0 . 8 8 1 ( p { < } 1 0 ^ { - 3 0 0 } )$ and Pearson $r { = } 0 . 9 0 8$ , with 81.3% same-valence agreement and MAE 7.2 SI units — less than one bin (Appendix E).

Correlation on individual labels is necessary but not sufficient; a labeler could still consistently mis-estimate the entity-level $P _ { \mathrm { p o p } } .$ . We therefore recompute $P _ { \mathrm { p o p } }$ from star ratings alone for each of the 10 Yelp entities and evaluate ${ \mathrm { W A R P } } { \mathrm { s } }$ retrievals against this fully independent target. Mean $W _ { 1 }$ gap between star-derived and LLM-derived $P _ { \mathrm { p o p } } \colon ~ 2 . 8 8$ (Appendix Table 16). Triangle inequality then gives W (WARP, star- $P _ { \mathrm { p o p } } ) \leq 5 . 6 4$ and $W _ { 1 } ( \mathrm { T o p } { - } k , \mathrm { s t a r } { - } P _ { \mathrm { p o p } } ) \geq 1 0 . 1 5$ , i.e., a worstcase reduction of at least 44.4% against a target our labels never touched. This complements two other circularity defenses — the VADER labeling ablation and the mislabel sensitivity study (Appendix F), where $W _ { 1 }$ Minimizer never crosses the Top-k baseline on Yelp even at 50% label corruption — and the FDR-corrected significance testing and entity-clustered bootstrap in Appendix D. The gains are not an artifact of the labeler or of any single evaluation protocol.

## 5 Conclusion

As enterprise document corpora grow, RAG systems have become the default interface between users and organizational knowledge, returning relevant evidence in sub-second re-ranking latencies. Yet when the underlying corpus is diverse and highly opinionated, relevance-ranked retrieval selects only for topical similarity; the retrieved set may not reflect population opinions proportionally, producing summaries that appear grounded but misrepresent the balance of views.

WARP closes this gap by re-ranking retrieved evidence against the target population distribution, requiring no model fine-tuning, no retriever changes, and no additional inference calls. The result is at least a 43% reduction in distributional distance to the population target (W ), all within a 330 ms latency envelope suitable for production deployment.

## Limitations

Offline evaluation under production constraints. All reported results come from offline experiments, engineered to satisfy the sub-330 ms re-ranking budget typical of a large-scale e-commerce RAG setting (end-to-end depends on retrieval infrastructure; Appendix B.5). We therefore characterize retrieval and generation-side fidelity, not live user outcomes: whether proportionally faithful summaries measurably shift user trust, engagement, or decision quality under live traffic is untested, and offline $W _ { 1 }$ reductions need not map one-toone onto user-perceived faithfulness. A controlled online A/B test is the natural next step.

Corpus bias vs. retrieval bias. $P _ { \mathrm { p o p } }$ is corpusobserved, not a true population target: review corpora are self-selected, motivated writers with strong positive or negative experiences are overrepresented, and some segments never write at all. WARP removes the retrieval-induced bias — Topk layers cosine-similarity skew on top of the underlying corpus bias — but does not remove the corpus bias itself. Faithful matching over a biased sample can lend false authority: the summary reflects the writers’ distribution, not the underlying population’s. Because $P _ { \mathrm { p o p } }$ is an input to the re-ranker rather than something it learns, external signals (native star ratings, demographic priors, surveycalibrated distributions) can be plugged in directly to reweight the target — a natural extension requiring no architectural change.

Pre-computed labels required. Per-document sentiment-intensity annotations are needed at indexing time. A single LLM pass or VADER suffices, and 50–100 labels per entity are enough - but the cost–quality tradeoff of labeling strategies is not evaluated.

Scale and Temporal Drift. 156 queries across 26 entities; paired Wilcoxon provides adequate power, but scaling beyond 14K documents is untested. $P _ { \mathrm { p o p } }$ degrades gracefully under perturbation (72% better than Top-k at ε=0.2) but temporal drift is not addressed.

Single ordinal axis; multi-issue opinions. WARP operates on a single ordinal axis (sentiment intensity). Multi-issue opinion spaces — e.g., a review that praises price but criticizes support, or patient-experience, employee-engagement, and multi-issue polling settings where several stance dimensions matter jointly — would require multimarginal optimal transport, a natural extension we do not evaluate here.

Domain scope and generation. All domains are product/service reviews requiring entity-anchored opinions and an ordinal scale. Generation significance is driven by Yelp (97% decided) and Opin-Rank (94%); Seller Forums differentiates weakly (54%) due to sparse pools. A prompt ablation (Appendix G.1.4) shows reported gaps are criteriondependent.

## Ethics Statement

Distributional fidelity proportionally surfaces minority views, including potentially harmful ones. A content-safety filter should gate re-ranker output: excluded opinions are removed from both $P _ { \mathrm { p o p } }$ and the candidate pool before calibration. Because reranking is post-retrieval, it is compatible with any upstream safety filter without modification.

## Use of AI Writing Assistance

The research (design, experiments, analysis, and writing) is the authors’ own work. We used Claude (Anthropic) as a proofreader and sounding board: it flagged unclear phrasing, suggested structural edits, and helped scaffold parts of the experimental code, all of which we reviewed and revised ourselves. Figure 1 was generated with ChatGPT (OpenAI) to give readers a quick visual overview of the pipeline; we verified its accuracy. No AI system produced research claims or final prose.

## References

Aditya Agrawal, Alwarappan Nakkiran, Darshan Fofadiya, Alex Karlsson, Harsha Aduri, and Aman Singh Thakur. 2026. Retrieval-augmented generation must move beyond factual grounding to represent diverse opinions. Preprint, arXiv:2604.12138.

Elif Aktolga and James Allan. 2013. Sentiment diversification with different biases. In Proceedings ofACM SIGIR, pages 593–602.

Amazon AGI. 2025. The Amazon Nova family of models: Technical report and model card. Preprint, arXiv:2506.12103.

Amazon Web Services. 2024a. Amazon Titan text embeddings v2. https://docs.aws.amazo n.com/ai/responsible-ai/titan-text -embeddings/overview.html. Accessed: 2026-05.

Amazon Web Services. 2024b. Knowledge bases for Amazon Bedrock. https: //docs.aws.amazon.com/bedrock/la test/userguide/knowledge-base.html. Accessed: 2026-05.

Stefanos Angelidis and Mirella Lapata. 2018. Summarizing opinions: Aspect extraction meets sentiment prediction and they are both weakly supervised. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3675–3686, Brussels, Belgium. Association for Computational Linguistics.

Anthropic. 2025. The Claude model family. https://www-cdn.anthropic.com/6b e99a52cb68eb70eb9572b4cafad13df32e d995.pdf. Accessed: 2026-05.

Jaime Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based reranking for reordering documents and producing summaries. In Proceedings of ACM SIGIR, pages 335–336.

Sebastian Claici, Aude Genevay, and Justin Solomon. 2018. Wasserstein measure coresets. arXiv preprint arXiv:1805.07412.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

David F. Crouse. 2016. On implementing 2D rectangular assignment algorithms. IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679–1696.

Van Dang and W. Bruce Croft. 2012. Diversity by proportionality: An election-based approach to search result diversification. In Proceedings ofACM SIGIR, pages 65–74.

DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the frontier of open large language models. Preprint, arXiv:2512.02556.

Kavita Ganesan and ChengXiang Zhai. 2012. Opinionbased entity ranking. Information Retrieval, 15(2):116–150.

C.J. Hutto and Eric Gilbert. 2014. VADER: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings ofAAAI ICWSM.

Han Jiang, Rui Wang, Zhihua Wei, Yu Li, and Xinpeng Wang. 2023. Large-scale and multi-perspective opinion summarization with diverse review subsets. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 5641–5656, Singapore. Association for Computational Linguistics.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547.

Alex Kulesza and Ben Taskar. 2012. Determinantal point processes for machine learning. Foundations and Trends in Machine Learning, 5(2–3):123–286.

Matt Kusner, Yu Sun, Nicholas Kolkin, and Kilian Weinberger. 2015. From word embeddings to document distances. In Proceedings ofICML, pages 957–966.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Qiheng Lu and Nicholas D. Sidiropoulos. 2026. Principled and scalable diversity-aware retrieval via cardinality-constrained binary quadratic programming. Preprint, arXiv:2604.02554. Concurrent work.

Meta. 2024. Llama 3.3 model card. https: //www.llama.com/docs/model-cards -and-prompt-formats/llama3\_3/. Accessed: 2026-05.

Mistral AI. 2025. Mistral large 3. https: //docs.mistral.ai/models/model-c ards/mistral-large-3-25-12. Accessed: 2026-05.

Mir Tafseer Nayeem and Davood Rafiei. 2025. OpinioRAG: Towards generating user-centric opinion highlights from large-scale online reviews. In Second Conference on Language Modeling.

Vu Nguyen and Andrey Kan. 2026. MUSS: Multilevel subset selection for relevance and diversity. Preprint, arXiv:2503.11126.

Alex Oesterling, Claudio Mayrink Verdun, Carol Xuan Long, Alexander Glynn, Lucas Monteiro Paes, Sajani Vithana, Martina Cardone, and Flavio P. Calmon. 2024. Multi-group proportional representation in retrieval. In Advances in Neural Information Processing Systems, volume 37, pages 114601–114655. Curran Associates, Inc.

Chao Peng, Bin Wang, Zhilei Long, and Jinfang Sheng. 2025. AdaGReS: Adaptive greedy context selection via redundancy-aware scoring for token-budgeted RAG. Preprint, arXiv:2512.25052.

Gabriel Peyré and Marco Cuturi. 2019. Computational optimal transport. Foundations and Trends in Machine Learning, 11(5–6):1–257.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using siamese BERTnetworks. In Proceedings ofEMNLP-IJCNLP, pages 3982–3992.

Rodrygo L.T. Santos, Craig Macdonald, and Iadh Ounis. 2010. Exploiting query reformulations for web search result diversification. In Proceedings of WWW, pages 881–890.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. 2025. Judging the judges: A systematic study of position bias in LLMas-a-judge. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Associationfor Computational Linguistics, pages 292–314, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Ashudeep Singh and Thorsten Joachims. 2018. Fairness of exposure in rankings. In Proceedings of ACM SIGKDD, pages 2219–2228.

Harald Steck. 2018. Calibrated recommendations. In Proceedings ofACM RecSys, pages 154–162.

Peggy Tang, Kun Hu, Rui Yan, Lei Zhang, Junbin Gao, and Zhiyong Wang. 2022. OTExtSum: Extractive text summarisation with optimal transport. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1128–1141, Seattle, United States. Association for Computational Linguistics.

Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. 2025. Judging the judges: Evaluating alignment and vulnerabilities in LLMs-as-judges. In Proceedings ofthe Fourth Workshop on Generation, Evaluation and Metrics (GEM<sup>2</sup>), pages 404–430, Vienna, Austria and virtual meeting. Association for Computational Linguistics.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand. Association for Computational Linguistics.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. In Proceedings of NeurIPS.

Haolun Wu, Yansen Zhang, Chen Ma, Fuyuan Lyu, Bowei He, Bhaskar Mitra, and Xue Liu. 2024. Result diversification in search and recommendation: A survey. Preprint, arXiv:2212.14464.

Yelp. 2024. Yelp open dataset. https://www.ye lp.com/dataset. Accessed: 2026-05.

Hai-Tao Yu, Adam Jatowt, Hideo Joho, Joemon M. Jose, Xiao Yang, and Long Chen. 2019. WassRank: Listwise document ranking using optimal transport theory. In Proceedings ofthe Twelfth ACM International Conference on Web Search and Data Mining, pages 24–32. Association for Computing Machinery.

Meike Zehlike, Francesco Bonchi, Carlos Castillo, Sara Hajian, Mohamed Megahed, and Ricardo Baeza-Yates. 2017. FA\*IR: A fair top-k ranking algorithm. In Proceedings ofthe 2017ACM on Conference on Information and Knowledge Management, pages 1569– 1578. ACM.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36.

## A Experimental Setup

## A.1 Dataset Details

Opinion Scale. Sentiment-Intensity (SI) is mapped to a discrete ordinal scale:

Table 4: Sentiment-Intensity (SI) mapping to the 7-bin ordinal scale. Mixed labels (praise-and-criticism cooccurring in one review) collapse to the neutral bin; they account for 4–8% of extractions across domains and W<sub>1</sub> Minimizer never crosses the Top-k baseline on Yelp even at 50% label corruption (§F.1), so this pooling is bounded in impact. Multi-issue axes require multi-marginal transport — outside our current scope (§Limitations).

<table><tr><td>Sentiment</td><td>Intensity</td><td>SI Score</td></tr><tr><td>Positive</td><td>High / Med / Low</td><td>+30/+20 /+10</td></tr><tr><td>Neutral</td><td>Any</td><td>0</td></tr><tr><td>Mixed</td><td></td><td>0</td></tr><tr><td>Negative</td><td>High / Med / Low</td><td>-30/-20/-10</td></tr></table>

Entity Selection. Three stages. First, LLMbased entity extraction with domain-specific seed lists. Then diversity filtering: Shannon entropy $H ~ \geq ~ 0 . 6$ , minority sentiment ≥ 10%, mention count ≥ 100. Finally, top-n selection ranked by entropy descending.

Table 5: Dataset summary across three evaluation domains.
<table><tr><td>Dataset</td><td>Domain</td><td>Source</td><td>Size</td><td>Entities</td><td>Queries</td><td>Search</td><td>Embedding</td><td>Chunking</td><td>Labeling</td></tr><tr><td>Seller Forums</td><td>E-comm. seller</td><td>Public</td><td>~8K</td><td>6</td><td>36</td><td>Hybrid</td><td>Titan V2 (Amazon Web Services, 2024a) (1024d)</td><td>300 tok / 20% overlap</td><td>LLM-enriched metadata</td></tr><tr><td>Yelp Hotels</td><td>Hospitality</td><td>Yelp Open</td><td>14K</td><td>10</td><td>60</td><td>Semantic</td><td>MiniLM-L6 (384d)</td><td>Whole review</td><td>LLM-extracted (Claude)</td></tr><tr><td>OpinRank Cars</td><td>Automotive</td><td>OpinRank</td><td>13K</td><td>10</td><td>60</td><td>Semantic</td><td>MiniLM-L6 (384d)</td><td>Whole review</td><td>LLM-extracted; VADER</td></tr></table>

## A.2 Query Templates

Table 6 lists the exact query templates used to generate evaluation questions. Each entity is instantiated into all 6 templates, yielding 156 total queries (36 Seller Forums + 60 Yelp + 60 OpinRank).

Entities. Seller Forums (6): A+ Content, Account Health Rating, Brand Stores, Community Management, Coupons, Seller Education. Yelp Hotels (10): Booking Experience, Breakfast, Breakfast Quality, Business Center, Casino, Loyalty Program, Nightly Rate, Pool, Room Quality, Shuttle Service. OpinRank Cars (10): Oil Change Interval, Dealer Experience, Vehicle Size, Sales Experience, Fuel Tank Size, Resale Value, City MPG, Remote Start, Climate Control, Ground Clearance.

## A.3 Prompt Library

All prompts used in the WARP pipeline. Template variables shown in {braces}.

## A.3.1 Opinion Extraction

Prompt A.3.1: Registry-Guided Extraction (Open-Domain)

A single LLM call per document extracts entities and sentiment-intensity labels using a domainspecific tiered seed list while remaining open to discovering new entities.

```yaml
System: You extract structured opinion data from reviews.
Output ONLY valid JSON arrays. No markdown, no
explanation.
User: Extract entity-level opinions from this review.
[{registry_block} -- domain-specific
tiered entity list injected here]
Instructions:
1. Identify ALL entities/aspects the reviewer discusses.
2. If an entity matches the registry above, use the EXACT
registry name. Prefer the most specific tier (Tier 3 > Tier
2 > Tier 1).
3. If the reviewer discusses something NOT in the registry,
create a short descriptive entity name (open discovery).
4. For EACH entity mentioned, output:
- entity: the entity name (registry canonical or new)
- sentiment: “positive”, “negative”, “neutral”, or “mixed”
- intensity: “high”, “medium”, or “low”
- evidence: exact quote from the review supporting this
opinion (max 120 chars)
Only include entities explicitly discussed. Return [] if
none found.
Output ONLY a JSON array.
Review:
{review_text}
```

For closed-domain settings (e.g., Yelp Hotels), the extraction prompt uses a fixed aspect list with keyword mappings instead of the tiered registry, but the output schema is identical.

## A.3.2 Answer Generation

## Prompt A.3.2: Opinion Summary Generator

Instructs the LLM to synthesize a proportional opinion summary from retrieved documents.

System: You are summarizing community opinions from   
review data. Use ONLY the provided documents. Repre  
sent opinions proportionally — include minority views,   
not just the majority. Be specific and cite reviewer experi  
ences where possible.   
User: Question: {question}   
Reviews:   
{docs}   
Summarize what the community thinks. Be thorough —   
cover the full range of opinions proportionally.   
Do not add opinions beyond what’s in the documents.

## A.3.3 Pairwise Generation Evaluation (Judge)

Each answer pair is evaluated via positioncontrolled blind pairwise comparison (positions swapped in the second call for debiasing). The judge scores three dimensions; the OPINION FAIR-NESS dimension is swapped between variants in our ablation (§G.1.4).

## Prompt A.3.3: Pairwise Judge Template

Full evaluation template: blind comparison on opinion fairness, informativeness, and groundedness.

Table 6: Query templates per domain. Each template is instantiated with every selected entity (6 for Seller Forums, 10 for Yelp/OpinRank), producing 6 questions per entity (2 breadth, 2 polar, 2 segment).
<table><tr><td>Type</td><td>Idx</td><td>Seller Forums</td><td>Yelp Hotels</td><td>OpinRank Cars</td></tr><tr><td>breadth</td><td>0</td><td>What do sellers think about {en- tity}?</td><td>What do guests think about {en- tity}?</td><td>What do owners think about {en- tity}?</td></tr><tr><td>breadth</td><td>1 tity}.</td><td>Summarize seller opinions on {en-</td><td>Summarize guest opinions on {en- tity}.</td><td>Summarize owner opinions on {entity}.</td></tr><tr><td>polar</td><td>0</td><td>What are the biggest complaints about {entity}?</td><td>What are the biggest complaints about {entity}?</td><td>What are the biggest complaints about {entity}?</td></tr><tr><td>polar</td><td>1</td><td>What positive experiences have sellers had with {entity}?</td><td>What positive experiences have guests had with {entity}?</td><td>What do owners praise most about {entity}?</td></tr><tr><td>segment</td><td>0</td><td>How do small sellers vs large sell- ers feel about {entity}?</td><td>How do business travelers vs leisure travelers feel about {en- tity}?</td><td>How do commuters vs enthusiasts feel about {entity}?</td></tr><tr><td>segment</td><td>tity}?</td><td>Do new sellers and experienced sellers differ in their views of {en-</td><td>Do solo guests and families differ in their views of {entity}?</td><td>Do new buyers and long-term owners differ in their views of {entity}?</td></tr><tr><td colspan="3">System: You are an expert evaluator comparing two an- swers that summarize community opinions. You do NOT know which retrieval method produced which answer. Judge strictly on the quality of the answers themselves. User: You are evaluating two answers to the same ques- tion about community opinions. Both answers were gener- ated from different sets of reviews retrieved for the same question.</td><td colspan="2">OPINION FAIRNESS (Proportional Accuracy): The ac- tual community opinion distribution for this topic is: {p_pop_description}. Which answer more accu- rately reflects these real proportions? The ideal answer should make the reader walk away with a correct sense of how common each view is: a dominant opinion should dominate the summary, a rare opinion should be men- tioned but not over-emphasized. An answer that gives</td></tr><tr><td colspan="3">Question: {question} – Answer A — {answer_a} - Answer B —</td><td colspan="2">equal weight to a 10% minority and a 70% majority is MISLEADING, even if well-intentioned. Pick the answer whose emphasis better matches the actual distribution above. Where {p_pop_description} is dynamically</td></tr><tr><td colspan="3">{answer_b} Compare the answers on these dimensions. For each, pick “A”, “B”, or “tie”: 1. OPINION FAIRNESS: [fairness variant</td><td colspan="2">filled per entity from the ground-truth Ppop, e.g.: “approximately 65% positive (mostly satisfied), 25% negative (dissatisfied), 10% neutral.&quot;</td></tr><tr><td colspan="3">inserted -- see Prompts A.3.3.1 and A.3.3.2 below] 2. INFORMATIVENESS: Which answer is more helpful, specific, and actionable for someone trying to understand</td><td colspan="2">Prompt A.3.5: Fairness Variant B — Viewpoint Cov- erage Rewards distinct viewpoint coverage; favors sur-</td></tr><tr><td colspan="3">what the community thinks? Consider breadth of points covered, specificity, and usefulness. 3. GROUNDEDNESS: Which answer appears more</td><td colspan="2">facing rare perspectives even if disproportionate. OPINION FAIRNESS (Viewpoint Coverage): Which an- swer covers MORE DISTINCT VIEWPOINTS from the</td></tr><tr><td colspan="3">grounded in actual community member experiences? Look for attribution to specific perspectives vs vague gen- eralizations that could be hallucinated. Respond ONLY with JSON:</td><td colspan="2">community, giving voice to all perspectives including rare and minority ones? An answer that only represents the majority view — even if that majority is large — is LESS fair than one that surfaces unique perspectives readers might not otherwise encounter. The ideal answer ensures</td></tr></table>

The baseline evaluation uses a generic fairness instruction (“which answer more proportionally represents the full range of community opinions, including minority views?”). Our ablation replaces it with the two variants below.

## Prompt A.3.4: Fairness Variant A — Proportional Accuracy

Provides ground-truth P<sub>pop</sub> to the judge; penalizes over-representing minority opinions.

## A.3.4 Retrieval Augmentation

<table><tr><td>Prompt A.3.6: Pole-Biased Multi-Query Templates During pool expansion (Stage 1.5), three sentiment- pole queries recover documents that cosine retrieval buries.</td></tr><tr><td>Positive: What positive experiences have sellers had with {entity}?</td></tr><tr><td>Negative: What are the biggest complaints about {entity}?</td></tr><tr><td>Neutral: What factual information do sellers share about {entity}?</td></tr></table>

Evaluation queries (6 per entity) are generated from three templates—breadth (“What do {persona} think about {entity}?”), polar (“What are the biggest complaints...”), and segment (“How do {segment\_A} vs {segment\_B} feel...”)— with domain-adapted persona and segment terms (e.g., “sellers” / “small sellers vs large sellers”).

## B Algorithm Details

Table 7 summarizes the key symbols used throughout.

Table 7: Notation reference.  
```powershell
Symbol Meaning
$d$ A candidate document
e Target entity
$k$ Output size (documents returned)
$N$ Candidate pool size
$S I ( d )$ Sentiment-intensity label of d
$P _ { \mathrm { p o p } }$ Ground-truth population distribution
$P _ { \mathrm { p o o l } }$ Pool-level empirical distribution
$\dot { W } _ { 1 } ( P , Q )$ Wasserstein-1 distance between P and Q
$\lambda$ Relevance–calibration trade-off weight
$\delta$ Deficit threshold for pool expansion
$n _ { \mathrm { e x p } }$ Budget (docs per deficit pole)
$\operatorname { r e l } ( d )$ Relevance score of document d
EM Entity-matched subset of pool
```

## B.1 Entity-Gated Re-retrieval

The two-pass entity-gated pool construction recov  
ers entity-matched documents that pure relevance   
ranking buries.   
Algorithm 2 Entity-Gated Re-retrieval   
Require: Results R (by score), entity e, sizes   
$n _ { 1 } , n _ { 2 }$ , threshold τ   
1: $P _ { 1 }  R [ 1 : n _ { 1 } ]$ (top-n<sub>1</sub> by relevance)   
2: ${ \cal C } _ { 2 } \  \ \{ d \ \in \ R \ \backslash \ P _ { 1 } \ :$ entity(d) = e ∧   
score $( d ) \geq \tau \}$   
3: Sort $C _ { 2 }$ by $| S I ( d ) |$ descending   
4: $P _ { 2 }  C _ { 2 } [ 1 : n _ { 2 } ]$   
5: Pool ← dedup $\left( P _ { 1 } \cup P _ { 2 } \right)$ , sorted by score   
6: return Pool

## B.2 Adaptive Distribution-Aware Pool Expansion

When entity-matched documents under-represent a sentiment pole relative to $P _ { \mathrm { p o p } } ,$ pole-biased queries (§A.3.4) retrieve additional candidates before reranking. The cold-start path (EM = ∅) treats every pole with non-negligible mass as a deficit, ensuring new entities still receive balanced pools. In practice, one to three deficit poles trigger per entity; $n _ { \mathrm { e x p } }$ caps total expansion cost at a fixed multiple of k.

Algorithm 3 Adaptive Pool Expansion   
Require: Pool $C _ { 0 } .$ , entity e, $P _ { \mathrm { p o p } }$ , threshold δ, bud  
get $n _ { \mathrm { e x p } }$   
1: EM $ \{ d \in C _ { 0 }$ : entity $\langle d ) = e \}$   
2: if EM = ∅ then   
3: DeficitPoles $ \{ s : P _ { \mathrm { p o p } } ( s ) > 0 . 0 1 \}$   
4: else   
5: $P _ { \mathrm { p o o l } } ( s ) \gets | \{ d \in \mathrm { E M } : S I ( d ) = s \} | / | \mathrm { E M } |$   
6: DeficitPoles $ \{ s : { P _ { \mathrm { p o o l } } ( s ) < \delta \cdot { P _ { \mathrm { p o p } } ( s ) } \} }$   
7: end if   
8: if DeficitPoles = ∅ then   
9: return $C _ { 0 }$   
10: end if   
11: Retrieve $n _ { \mathrm { e x p } }$ entity-matched docs per deficit   
pole   
12: Pool ← dedup $( C _ { 0 } \cup C _ { \exp } )$   
13: return Pool   
B.3 W<sub>1</sub>-MMR   
Algorithm 4 W<sub>1</sub>-MMR   
Require: Pool C, scores rel(·), target $P _ { \mathrm { p o p } } , \lambda ,$ , size   
$k$   
1: $S \gets \emptyset$   
2: while $| S | < k$ do   
3: $W _ { 1 } ^ { \mathrm { c u r } }$ ←   
60 if $S = \emptyset$   
$\big \lfloor W _ { 1 } ( \mathrm { E m p D i s t } ( S ) , P _ { \mathrm { p o p } } )$ else   
4: for each d $\in C \setminus S$ do   
5: ∆W<sub>1</sub> ← W<sup>cur</sup> − W<sub>1</sub>(EmpDist(S ∪   
$\{ d \} ) , P _ { \mathrm { p o p } } )$   
6: score(d) ← λ · rel(d) + (1−λ) ·   
$\Delta W _ { 1 } / W _ { 1 } ^ { \mathrm { c u r } }$   
7: end for   
8: $d ^ { * } \gets \arg \operatorname* { m a x } _ { { d \in C } \backslash S }$ score(d)   
9: $S \gets S \cup \{ d ^ { * } \}$   
10: end while   
11: return S

## B.4 Label Acquisition

Per-document SI labels come from a single LLM pass at indexing time (<200 ms/doc). Labeling is cheap. The Seller Forums corpus (∼8K posts) labels in under 4 minutes with 150 concurrent workers. VADER (Hutto and Gilbert, 2014) works as a zero-cost alternative for well-structured review text. $P _ { \mathrm { p o p } }$ is just a count aggregation per entity $( O ( n )$ milliseconds); incremental updates on ingestion avoid recomputation entirely.

## B.5 End-to-End Latency Breakdown

Hardware. Intel Xeon Platinum 8175M @ 2.50 GHz, 8 cores (single-threaded execution), no GPU.

WARP’s re-ranking step runs entirely in NumPy on the retrieved candidate distribution — no additional model calls. End-to-end latency therefore decomposes into three components (Table 8):

Re-ranking latency is dominated by CDF differencing over $m { = } 7$ bins and is backend-independent. Retrieval and re-retrieval dominate managedendpoint deployments; for local FAISS the whole pipeline fits inside a 500 ms sub-second SLA.

The $W _ { 1 }$ Minimizer’s near-zero p50 on Seller Forums (0.1 ms) reflects the sparse entity-filtered pool: few candidates match the queried entity, so the greedy loop terminates almost immediately. p99 stays under 310 ms across all WARP variants on both domains, and end-to-end with local FAISS retrieval fits in <365 ms — comfortably inside a sub-second SLA. Managed vector stores add ∼1.5 s of retrieval latency that WARP does not remove.

## C Retrieval Ablations

We ablate key components of the retrieval pipeline: distance metric choice, entity filtering, pool expansion, convergence, per-entity variation, and sensitivity to hyperparameters.

## C.1 JS vs. $W _ { 1 }$ : Metric Comparison and Entity-Match Confound

This section unpacks the $\scriptstyle \mathbf { J } \mathbf { S } - \mathbf { V } \mathbf { S } - W _ { 1 }$ comparison summarized in Section 4.2. We rerun the four calibrated methods (KL(JS) Minimizer, $W _ { 1 }$ Minimizer, KL(JS)-MMR, $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R } )$ on Yelp and OpinRank at $k { = } 2 0$ , holding the greedy structure and entity filtering fixed so only the distance metric varies within each row-pair.

Table 10 shows KL(JS)-MMR beating $W _ { 1 ^ { - } }$ MMR on Yelp and OpinRank $W _ { 1 }$ . On its face this reverses the ordinal-metric advantage the Minimizer comparison establishes. It doesn’t: the two hybrids retrieve very different candidate pools, and the metric is being scored on top of that difference.

In the top block (hybrids, uncontrolled EM), KL(JS)-MMR retrieves nearly twice as many entity-matched documents as $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ (38.2% vs. 19.4% on Yelp; 36.1% vs. 21.8% on OpinRank), and its lower $W _ { 1 }$ tracks that EM gap rather than the choice of ordinal-vs-categorical ground cost. In the bottom block (Minimizers, controlled EM), both algorithms filter to the same entity-matched pool (EM=93.8% Yelp, 85.8% OpinRank), and $W _ { 1 }$ wins on both domains — 2.76 vs. 3.00 on Yelp, 4.14 vs. 4.54 on OpinRank. The controlled comparison is where the ordinal metric earns its billing; the uncontrolled hybrid comparison is a retrieval-pool artifact.

## C.2 Entity-Filtered Control (Yelp)

On sparse Seller Forums (Top-k baseline EM∼40%; entity-filter ceiling ∼51%), entity filtering provides only −11.6% $W _ { 1 }$ reduction (vs. −43.4% on Yelp, where the entity filter reaches 100% EM), confirming that $W _ { \mathrm { 1 ^ { - } } } \mathrm { M M R ^ { \circ } s }$ full-pool scoring drives gains in entity-sparse domains (−37.7% beyond the entity-filter baseline).

## C.3 Adaptive-k Convergence

$W _ { 1 }$ Minimizer hits diminishing returns at $k ^ { * } { = } 3$ (Seller Forums), $k ^ { * } { = } 4$ (Yelp), $k ^ { * } { = } 5$ (OpinRank) (Figure 3). An adaptive-k strategy could therefore feed far fewer documents to the LLM without losing distributional fidelity. This also implies robustness to imprecise $P _ { \mathrm { p o p } }$ estimates.

## $W _ { 2 }$ vs. $W _ { 1 }$ Comparison

A natural concern is whether optimizing $W _ { 1 }$ (linear ground cost) sacrifices performance under higher-order transport metrics. We compute $W _ { 2 }$ (quadratic ground cost) for all methods on Yelp Hotels (k=20). The rank ordering across methods is perfectly preserved: every method that achieves lower $W _ { 1 }$ also achieves lower $W _ { 2 }$ , with a linear relationship (slope = 1.19, $R ^ { 2 } > 0 . 9 9 )$ . This occurs because our 7-bin ordinal scale has uniform spacing $( \Delta s = 1 0 )$ , which makes $W _ { 1 }$ and $W _ { 2 }$ monotonically related for any pair of distributions on this support. The practical implication is that our choice of $W _ { 1 }$ , motivated by its $O ( m )$ closed-form computation, does not trade off optimality under alternative Wasserstein orders.

## C.5 Per-Entity Breakdowns

Aggregate results in Table 1 mask substantial perentity variation. The gains concentrate on entities whose baseline distributions are most skewed, precisely where minority-opinion misrepresentation does the most damage. We break out per-entity $W _ { 1 }$ for all three domains below.

Table 8: Per-component latency breakdown. Re-ranking is hardware-only (no model inference); retrieval and re-retrieval depend on the vector-store backend. p99 for re-ranking; typical/median for the rest.
<table><tr><td>Component</td><td>Local FAISS</td><td>Managed Bedrock Knowledge Base</td></tr><tr><td>Retrieval</td><td>&lt;50ms</td><td>~1.5s</td></tr><tr><td>Re-retrieval†</td><td>&lt;5 ms</td><td>~100 ms</td></tr><tr><td>Re-ranking (p99)</td><td>&lt;310ms</td><td>&lt;310ms</td></tr><tr><td>End-to-end</td><td>&lt;365 ms</td><td>~1.9s</td></tr></table>

<sup>†</sup> Only fires when pool expansion (Entity-Gated / Adaptive) detects a deficit.

Table 9: Per-query re-ranking latency percentiles (N=96 queries: 36 Seller Forums + 60 Yelp Hotels, single threaded on the hardware above). p99 drives production SLAs; max reported for tail auditing. All WARP variants stay under 310 ms at $\mathsf { p } 9 9 .$
<table><tr><td>Domain</td><td>Method</td><td>Queries</td><td>p50 (ms)</td><td> $\mathbf { p 9 5 } \left( \mathbf { m s } \right)$ </td><td> $\mathbf { p 9 9 } \left( \mathbf { m s } \right)$ </td><td>Max (ms)</td></tr><tr><td>Seller Forums</td><td>W1 Minimizer</td><td>36</td><td>0.1</td><td>38.7</td><td>45.5</td><td>48.0</td></tr><tr><td>Seller Forums</td><td>W1-MMR</td><td>36</td><td>156.8</td><td>163.1</td><td>166.2</td><td>167.4</td></tr><tr><td>Seller Forums</td><td>EG + W1Min</td><td>36</td><td>0.6</td><td>211.0</td><td>216.2</td><td>218.8</td></tr><tr><td>Seller Forums</td><td>DPP</td><td>36</td><td>47.8</td><td>49.1</td><td>51.9</td><td>53.0</td></tr><tr><td>Yelp Hotels</td><td>W1 Minimizer</td><td>60</td><td>65.0</td><td>270.8</td><td>289.9</td><td>295.3</td></tr><tr><td>Yelp Hotels</td><td> $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ </td><td>60</td><td>81.6</td><td>284.7</td><td>301.6</td><td>303.5</td></tr><tr><td>Yelp Hotels</td><td>EG + W1Min</td><td>60</td><td>145.7</td><td>302.0</td><td>307.6</td><td>312.5</td></tr><tr><td>Yelp Hotels</td><td>DPP</td><td>60</td><td>103.5</td><td>127.4</td><td>129.3</td><td>131.3</td></tr></table>

![](images/c4f06c821454008031ea52bd94e061cde2f31ca313e147edbfdc10955d1ec763.jpg)

![](images/4fd1891123da299c720cbcb03eab5510ce2adf2f46bb7d426c1e6a1330ac69e8.jpg)  
Figure 3: $W _ { 1 }$ Minimizer convergence vs. output size k. Median convergence point $k ^ { * }$ (where $\Delta W _ { 1 } < 0 . 5$ per additional document): Seller Forums $k ^ { * } { = } 3 ,$ , Yelp $k ^ { * } { = } 4 ,$ OpinRank $k ^ { * } { = } 5$

Table 10: Entity-match confound in hybrid variants. KL(JS)-MMR retrieves ∼1.7–2× more entity-matched documents than $W _ { 1 }$ -MMR on both Yelp and OpinRank. In the controlled Minimizer comparison (same entity filter, EM equalized) $W _ { 1 }$ wins on both domains — the KL(JS)-MMR $W _ { 1 }$ advantage in the uncontrolled comparison traces to the EM gap, not to metric superiority.
<table><tr><td></td><td colspan="2">Yelp</td><td colspan="2">OpinRank</td></tr><tr><td>Method</td><td> $W _ { 1 }$ </td><td>EM%</td><td>W1</td><td>EM%</td></tr><tr><td>KL (JS)-MMR</td><td>2.51</td><td>38.2%</td><td>3.32</td><td>36.1%</td></tr><tr><td>W1-MMR</td><td>4.02</td><td>19.4%</td><td>3.41</td><td>21.8%</td></tr><tr><td>KL (JS) Minimizer</td><td>3.00</td><td>93.8%</td><td>4.54</td><td>85.8%</td></tr><tr><td>W1 Minimizer</td><td>2.76</td><td>93.8%</td><td>4.14</td><td>85.8%</td></tr></table>

## C.5.1 Yelp Hotels (k=20)

$W _ { 1 }$ Minimizer lands below 2.0 on 7 of 10 entities. That is near-perfect calibration. Two outliers

Table 11: Entity-filter control experiment (Yelp Hotels, k=20). Entity filtering alone reduces W by 43.4%; W Minimizer achieves 62.6% additional reduction beyond the entity-filter control.
<table><tr><td>Approach</td><td>EM%</td><td>W1</td><td>∆ vs Top-K</td><td>∆ vs EF+Top-K</td></tr><tr><td>Top K (no filter)</td><td>40.1</td><td>13.03</td><td></td><td></td></tr><tr><td>EntityFilter + Top-K</td><td>100.0</td><td>7.38</td><td>-43.4%</td><td></td></tr><tr><td>EntityFilter + MMR</td><td>100.0</td><td>6.99</td><td>-46.4%</td><td>-5.3%</td></tr><tr><td>EntityFilter + DPP</td><td>100.0</td><td>5.64</td><td>-56.7%</td><td>-23.6%</td></tr><tr><td>W1 Minimizer</td><td>93.8</td><td>2.76</td><td>-78.8%</td><td>-62.6%</td></tr><tr><td>W1-MMR</td><td>19.4</td><td>4.02</td><td>-69.2%</td><td>-45.5%</td></tr><tr><td>EG + W1Min</td><td>99.2</td><td>1.52</td><td>-88.3%</td><td>-79.4%</td></tr><tr><td>AE + W1 Min</td><td>95.9</td><td>1.66</td><td>-87.3%</td><td>-77.5%</td></tr></table>

remain: Business Center (9.76) and Shuttle Service (5.78), both corresponding to entities with extreme distributional skew and thin pool diversity. Even greedy optimization cannot fully close the gap within k=20 selections under those conditions.

Table 12: Per-entity breakdown (Yelp Hotels, 10 aspects). $W _ { 1 }$ Minimizer achieves $< 2 . 0$ on 7/10 entities.
<table><tr><td>Entity</td><td>Top-k W1</td><td>W1Min</td><td>W1-MMR</td><td>WassRank OT</td></tr><tr><td>Booking Experience</td><td>18.30</td><td>1.95</td><td>3.78</td><td>1.58</td></tr><tr><td>Breakfast</td><td>7.55</td><td>1.48</td><td>2.87</td><td>1.42</td></tr><tr><td>Breakfast Quality</td><td>5.78</td><td>0.87</td><td>3.79</td><td>0.65</td></tr><tr><td>Business Center</td><td>24.51</td><td>9.76</td><td>6.72</td><td>9.26</td></tr><tr><td>Casino</td><td>14.79</td><td>0.86</td><td>4.07</td><td>0.62</td></tr><tr><td>Loyalty Program</td><td>16.31</td><td>4.07</td><td>4.49</td><td>2.95</td></tr><tr><td>Nightly Rate</td><td>7.49</td><td>0.82</td><td>3.45</td><td>0.66</td></tr><tr><td>Pool</td><td>10.79</td><td>0.73</td><td>2.68</td><td>0.46</td></tr><tr><td>Room Quality</td><td>8.55</td><td>1.24</td><td>2.67</td><td>2.18</td></tr><tr><td>Shuttle Service</td><td>21.95</td><td>5.78</td><td>5.70</td><td>5.38</td></tr></table>

## C.5.2 OpinRank Cars (k=20, top 10 by entropy)

OpinRank entities vary widely in mention count (39–1,465) and minority fraction (19–45%). Highcount entities with moderate minority shares calibrate well: City MPG, for instance, has 1,465 mentions at 25% minority and reaches $W _ { 1 } { = } 0 . 3 8$ Hard cases persist. Vehicle Size has only 42 mentions; Dealer Experience starts from a $W _ { 1 } { = } 2 3 . 0 7$ baseline. Both resist full correction.

## D Statistical Validity

Two orthogonal concerns for the significance testing: multiple comparisons across many method × domain combinations, and non-independent samples within a domain (queries drawn from the same entity share an underlying opinion distribution). We address both.

## D.1 Benjamini–Hochberg Correction

We ran 57 paired Wilcoxon tests across three domains and eight re-ranking variants (baselines and $W _ { 1 }$ family). Benjamini–Hochberg FDR correction at α=0.05 leaves 39/57 tests surviving. Table 14 shows the 14 W<sub>1</sub>-family tests: every one survives on every domain. The 18 non-surviving tests come from methods outside our contribution set (Stancebased re-ranking, generic MMR, DPP-Evidence variants).

## D.2 Entity-Clustered Bootstrap

Paired-Wilcoxon assumes independent paired observations. Queries drawn from the same entity are not independent, since they share the same underlying opinion distribution. We therefore resample at the entity level: 10,000 bootstrap iterations, drawing entities with replacement and recomputing the mean per-query $W _ { 1 }$ improvement. All 95% CIs exclude zero for every $W _ { 1 }$ method on every domain (Table 15).

Seller Forums’ wider CIs $( \mathrm { e . g . , } W _ { 1 }$ Minimizer [0.67, 10.38]) reflect its small entity count $( N { = } 4 )$ ， not a qualitatively different effect — the lower bound still exceeds zero. The aggregate signal is cross-domain consistency across 10 Yelp, 9 Opin-Rank, and 4 Seller Forums entities.

## E Independent-Target Validation

Our LLM-derived sentiment-intensity labels define both the population target $P _ { \mathrm { p o p } }$ used at inference and the $W _ { 1 }$ evaluation metric applied to the retrieved evidence. If the labeler had systematic biases, calibrating and measuring against it could inflate the numbers. We cross-validate against Yelp’s native 1–5 star ratings, which are ordinal, humanauthored at review time, and participate in no part of the WARP pipeline — neither retrieval, nor reranking, nor the $P _ { \mathrm { p o p } }$ used at inference. Section 4.5 in the main body summarizes; this appendix reports the underlying numbers.

A high correlation on individual labels is a necessary but not sufficient condition: the entity-level $P _ { \mathrm { p o p } }$ is an aggregate, and a labeler could still consistently mis-estimate it. We therefore recompute $P _ { \mathrm { p o p } }$ from star ratings alone for each of the 10 Yelp entities and quantify the divergence from the LLM-derived $P _ { \mathrm { p o p } }$ actually used at inference (Table 16). Mean $W _ { 1 }$ divergence: 2.88. Under a triangle-inequality bound, WARP’s retrievals stay within $W _ { 1 } { = } 5 . 6 4$ of the star-derived target while Top-k is at least $W _ { 1 } { = } 1 0 . 1 5$ away — a worst-case reduction of at least 44.4% against a target the labeler never touched.

## F Robustness & Sensitivity

## F.1 Noise Tolerance

Two studies stress-test the system under realistic noise. First, Dirichlet perturbation of $P _ { \mathrm { p o p } }$ simulates inaccurate population estimates. Second, mislabel sensitivity corrupts both document SI labels and $P _ { \mathrm { p o p } }$ at once. These jointly reveal how much noise each algorithm can absorb, and where deployment breaks down.

## F.1.1 Dirichlet Perturbation of $P _ { \mathbf { p o p } }$

We perturb $P _ { \mathrm { p o p } }$ with Dirichlet noise at $\varepsilon \_ { \mathbf { \Sigma } } \in$ $\{ 0 , 0 . 1 , 0 . 2 , 0 . 5 \}$ (N=30 samples per entity) to simulate estimation error in the population distribution (Figure 4):

On Yelp, which has a dense candidate pool, the $W _ { 1 }$ Minimizer at $\varepsilon { = } 0 . 2$ degrades by 31%, but it still comes in 72% ahead of Top-k. $W _ { 1 }$ -MMR barely moves under the same perturbation $( < 1 \%$

![](images/5e459e44900cde4f451a6d865b263c52cb8a7fa91e31bbebdf03adb58d59b793.jpg)

Table 13: Per-entity breakdown (OpinRank, LLM-extracted entities). Count = opinion mentions, H = Shannon entropy, Min% = minority sentiment fraction.
<table><tr><td>Entity</td><td>Count</td><td>H</td><td>Min%</td><td>Top-k</td><td>W1 Min</td><td>W1-MMR</td><td>WassRank OT</td></tr><tr><td>Oil Change Interval</td><td>144</td><td>1.49</td><td>21%</td><td>3.96</td><td>1.08</td><td>2.95</td><td>0.78</td></tr><tr><td>Dealer Experience</td><td>425</td><td>1.47</td><td>30%</td><td>23.07</td><td>11.71</td><td>3.40</td><td>10.28</td></tr><tr><td>Vehicle Size</td><td>42</td><td>1.37</td><td>19%</td><td></td><td>8.31</td><td>6.19</td><td>8.31</td></tr><tr><td>Sales Experience</td><td>776</td><td>1.33</td><td>28%</td><td>15.66</td><td>3.01</td><td>3.19</td><td>2.57</td></tr><tr><td>Fuel Tank Size</td><td>269</td><td>1.32</td><td>36%</td><td>13.67</td><td>3.06</td><td>3.25</td><td>3.05</td></tr><tr><td>Resale Value</td><td>427</td><td>1.30</td><td>39%</td><td>12.19</td><td>2.24</td><td>2.40</td><td>2.33</td></tr><tr><td>City MPG</td><td>1465</td><td>1.29</td><td>25%</td><td>9.12</td><td>0.38</td><td>2.78</td><td>0.38</td></tr><tr><td>Remote Start</td><td>39</td><td>1.25</td><td>41%</td><td>8.80</td><td>4.52</td><td>3.54</td><td>4.52</td></tr><tr><td>Climate Control</td><td>727</td><td>1.24</td><td>45%</td><td>9.95</td><td>0.95</td><td>2.43</td><td>0.94</td></tr><tr><td>Ground Clearance</td><td>201</td><td>1.22</td><td>33%</td><td>23.55</td><td>6.12</td><td>4.00</td><td>5.50</td></tr></table>

ε (Dirichlet noise)  
Figure 4: Dirichlet perturbation robustness across all three domains. Dashed red line = Top-k baseline. $\mathrm { { A t } } \varepsilon = 0 . 2 ,$ $W _ { 1 }$ Minimizer remains 72% better than Top-k on Yelp.  
Table 14: Benjamini–Hochberg FDR correction (α=0.05) on paired-Wilcoxon p-values for the $W _ { 1 }$ family. Every W<sub>1</sub>-family algorithm survives on every domain (14/14). Reduction is per-query paired reduction relative to Top-k; aggregate reductions in Table 1 weight queries differently. $^ { * * \bar { * } } p _ { \mathrm { a d j } } { < } 1 0 ^ { - 3 } , ^ { * * } p _ { \mathrm { a d j } } { < } 1 0 ^ { - 2 }$ ${ } ^ { * } p _ { \mathrm { a d j } } { < } 0 . 0 5 .$
<table><tr><td>Domain</td><td>Approach</td><td>N</td><td>Reduction</td><td>padj</td></tr><tr><td>Yelp</td><td>W1 Minimizer ***</td><td>57</td><td>+81.4%</td><td>&lt;0.000001</td></tr><tr><td>Yelp</td><td> $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R } ^ { \ast \ast \ast }$ </td><td>57</td><td>+70.2%</td><td>&lt;0.000001</td></tr><tr><td>Yelp</td><td> $\mathrm { E G } + W _ { 1 } \ \mathrm { M i n ^ { * * * } }$ </td><td>57</td><td>+89.8%</td><td>&lt;0.000001</td></tr><tr><td>Yelp</td><td> $\mathrm { A E } + W _ { 1 } \ \mathrm { M i n ^ { * * * } }$ </td><td>57</td><td>+89.1%</td><td>&lt;0.000001</td></tr><tr><td>Yelp</td><td> $\mathrm { W a s s R a n k } \mathrm { O T } ^ { * * * }$ </td><td>57</td><td>+83.3%</td><td>&lt;0.000001</td></tr><tr><td>OpinRank</td><td> $W _ { 1 } ~ \mathrm { { M i n i m i z e r } ^ { * * * } }$ </td><td>54</td><td>+72.4%</td><td>&lt;0.000001</td></tr><tr><td>OpinRank</td><td> $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R } ^ { \ast \ast \ast }$ </td><td>54</td><td>+76.7%</td><td>&lt;0.000001</td></tr><tr><td>OpinRank</td><td> $\mathrm { E G } + W _ { 1 } \ \mathrm { M i n } ^ { * * * }$ </td><td>54</td><td>+72.4%</td><td>&lt;0.000001</td></tr><tr><td>OpinRank</td><td> $\mathrm { A E } + W _ { 1 } \ \mathrm { M i n } ^ { * * * }$ </td><td>54</td><td>+72.4%</td><td>&lt;0.000001</td></tr><tr><td>OpinRank</td><td> $\mathrm { W a s s R a n k } \mathrm { O T } ^ { * * * }$ </td><td>54</td><td>+74.7%</td><td>&lt;0.000001</td></tr><tr><td>Seller Forums</td><td> $W _ { 1 } \mathrm { M i n i m i z e r } ^ { * * * }$ </td><td>24</td><td>+59.1%</td><td>0.000427</td></tr><tr><td>Seller Forums</td><td> $W _ { \mathrm { 1 ^ { - } } } \mathrm { M M R ^ { \ast \ast } }$ </td><td>24</td><td>+20.4%</td><td>0.004552</td></tr><tr><td>Seller Forums</td><td> $\mathrm { E G } + W _ { 1 } \mathrm { M i n } ^ { * }$ </td><td>24</td><td>+53.7%</td><td>0.049</td></tr><tr><td>Seller Forums</td><td> $\mathrm { A E } + W _ { 1 } \mathrm { M i n } ^ { * }$ </td><td>24</td><td>+53.7%</td><td>0.049</td></tr></table>

degradation). On Amazon Seller Forums the picture is similar in shape: all methods degrade gracefully, and $W _ { 1 }$ Minimizer at ε=0.2 stays 56% above Top-k. OpinRank is the most sensitive of the three because the unperturbed baseline is already tight $( W _ { 1 }$ Min sits at 0.87 at ε=0), so relative degrada-

Table 15: Entity-clustered bootstrap (10,000 resamples). Mean improvement in $W _ { 1 }$ relative to Top-k; 95% CIs computed by resampling entities with replacement. All CIs exclude zero.
<table><tr><td>Dataset</td><td>Method</td><td>#Entities</td><td>Mean ∆W1</td><td>95% CI</td></tr><tr><td>Yelp</td><td>W1 Minimizer</td><td>10</td><td>10.85</td><td>[8.26, 13.37]</td></tr><tr><td>Yelp</td><td>EG + W1Min</td><td>10</td><td>12.09</td><td>[8.72, 15.37]</td></tr><tr><td>Yelp</td><td> $W _ { \mathrm { 1 - M M R } }$ </td><td>10</td><td>9.58</td><td>[6.40, 12.80]</td></tr><tr><td>OpinRank</td><td>W1 Minimizer</td><td>9</td><td>9.65</td><td>[6.92, 12.34]</td></tr><tr><td>OpinRank</td><td>W1-MMR</td><td>9</td><td>10.22</td><td>[6.48, 14.17]</td></tr><tr><td>Seller Forums</td><td>W1 Minimizer</td><td>4</td><td>5.32</td><td>[0.67, 10.38]</td></tr><tr><td>Seller Forums</td><td>W1-MMR</td><td>4</td><td>1.84</td><td>[0.49, 3.25]</td></tr></table>

tion looks larger, but even there ε=0.2 holds 45% better than Top-k. Across all three domains, $W _ { 1 } \cdot$ MMR degrades the least, which we attribute to its hybrid objective partially shielding it from target noise.

## F.1.2 Mislabel Sensitivity (SI Label Error)

A harder question: at what SI label error rate does the $W _ { 1 }$ advantage drop below significance? We corrupt candidate document labels AND $P _ { \mathrm { p o p } }$ simultaneously, the realistic scenario where labeling errors propagate into the population estimate. Error rates $\in \ \{ 0 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 5 0 \}$ , N=20 repetitions per setting (Figure 5). “Crossover” marks the error rate at which a method’s $W _ { 1 }$ exceeds the Top-k baseline:

Table 16: Star-derived vs. LLM-derived $P _ { \mathrm { p o p } }$ per Yelp entity. Mean $W _ { 1 }$ gap: 2.88. Bounds: $W _ { 1 } ( { \mathrm { W A R P } } , { \mathrm { s t a r } } { - } P _ { \mathrm { p o p } } )$ ≤ 5.64, $W _ { 1 }$ (Top-k, star- $\begin{array} { r l r } { P _ { \mathrm { p o p } } ) } & { { } \ge } & { 1 0 . 1 5 ; } \end{array}$ worst-case reduction $\geq 4 4 . 4 \%$
<table><tr><td>Entity</td><td>N reviews</td><td> $W _ { 1 } ( \mathrm { s t a r } { - } P _ { \mathrm { p o p } } , \mathrm { L L M } { - } P _ { \mathrm { p o p } } )$ </td></tr><tr><td>Booking Experience</td><td>1,372</td><td>3.66</td></tr><tr><td>Breakfast</td><td>788</td><td>1.44</td></tr><tr><td>Breakfast Quality</td><td>637</td><td>4.15</td></tr><tr><td>Business Center</td><td>142</td><td>3.23</td></tr><tr><td>Casino</td><td>208</td><td>1.36</td></tr><tr><td>Loyalty Program</td><td>607</td><td>1.57</td></tr><tr><td>Nightly Rate</td><td>2,481</td><td>4.60</td></tr><tr><td>Pool</td><td>1,318</td><td>2.10</td></tr><tr><td>Room Quality</td><td>2,557</td><td>1.78</td></tr><tr><td>Shuttle Service</td><td>280</td><td>4.90</td></tr><tr><td>Mean</td><td></td><td>2.88</td></tr></table>

Dense pools are forgiving. $W _ { 1 }$ Minimizer never crosses the Top-k baseline on Yelp, even at 50% label error. Sparse pools need more care; use $W _ { 1 ^ { - } }$ MMR regardless of label quality, as its hybrid objective guarantees a floor.

## F.2 Noise Control

We need to rule out a trivial explanation: maybe any score perturbation induces diversity that looks like opinion-aware signal. So we add 5% Gaussian noise to retrieval scores and re-rank:

Table 17: Noise control: 5% Gaussian perturbation of retrieval scores yields negligible $W _ { 1 }$ improvement $( 2 - 3 \% , p > 0 . 0 5 )$ , confirming that our 43–88% distributional improvements $( W _ { 1 }$ -MMR on sparse pools; $W _ { 1 }$ Minimizer $/ \mathrm { E G } { + } W _ { 1 }$ Min on dense) represent genuine distributional optimization.
<table><tr><td>Method</td><td>Seller  $W _ { 1 }$ </td><td>Yelp  $W _ { 1 }$ </td><td>OpinRank  $W _ { 1 }$ </td></tr><tr><td>Top-k (baseline)</td><td>10.33</td><td>13.03</td><td>13.33</td></tr><tr><td>Top-k + 5% Noise</td><td>10.07</td><td>12.63</td><td>12.97</td></tr><tr><td>Reduction</td><td>2.5%</td><td>3.1%</td><td>2.7%</td></tr><tr><td>Significance</td><td>ns</td><td> $( p > 0 . 0 5 ,$ </td><td>all domains)</td></tr><tr><td> $W _ { 1 }$  Minimizer</td><td>8.84</td><td>2.76</td><td>4.14</td></tr><tr><td>Reduction</td><td>14.4%</td><td>78.8%</td><td>69.0%</td></tr></table>

Random score perturbation yields 2–3% $W _ { 1 }$ change (non-significant). Our methods hit 14–79% on the same pools.

## F.3 Hyperparameter Sensitivity

λ (relevance–calibration tradeoff). Hybrid methods $( W _ { \mathrm { 1 ^ { - M M R } } }$ and WassRank OT) each introduce λ. We sweep $\lambda \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ across all three domains (Figure 6). W<sub>1</sub>-MMR holds steady at $\lambda \leq 0 . 5$ on Seller Forums and Yelp, then degrades once relevance dominates. Opin-Rank behaves differently: $\lambda { = } 0 . 7$ is optimal there (denser pools tolerate more calibration weight). WassRank OT is nearly λ-invariant on dense pools: Yelp gives $W _ { 1 } \approx 2 . 5 2$ for $\lambda \le 0 . 5$ (identical to 3 decimal places) and OpinRank yields $W _ { 1 } { = } 0 . 9 5$ with 100% EM across that range $\cdot ^ { 2 }$ No per-domain tuning needed.

Pool size N. We vary $N \in \{ 2 5 , 5 0 , 1 0 0 , 2 0 0 \}$ with fixed k=20 (Figure 7). $N { = } 1 0 0$ suffices for near-optimal performance on dense pools. $W _ { 1 } \cdot$ MMR benefits more from larger pools because it draws from the full (unfiltered) candidate set. On OpinRank, even $N { = } 5 0$ approaches the $N { = } 2 0 0$ result because entity-specific FAISS indices guarantee most candidates are already entity-matched. On sparse Seller Forums, larger pools become essential for $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ to locate cross-entity documents that fill distributional gaps.

## F.4 Population Estimation and Cold-Start Robustness

We evaluate whether $P _ { \mathrm { p o p } }$ can be reliably estimated from indexed data and whether the system degrades safely at cold-start. For each entity e, the full-corpus distribution $P _ { \mathrm { p o p } , e } ^ { * }$ serves as our oracle evaluation target. We sample subsets of $n ~ \in ~ \{ 1 0 , 2 5 , 5 0 , 1 0 0 , 2 5 0 , 5 0 0 \}$ entitymatched documents, compute $\hat { P } _ { \mathrm { p o p } , e } ^ { ( n ) }$ , and measure $W _ { 1 } ( \hat { P } _ { \mathrm { p o p } , e } ^ { ( n ) } , P _ { \mathrm { p o p } , e } ^ { * } )$ . Crucially, the re-ranker uses $\hat { P } _ { \mathrm { p o p } }$ but selected evidence is evaluated against $P _ { \mathrm { p o p } } ^ { * }$ ; this avoids circularity.

Estimation convergence. Figure 8 shows estimation error vs. sample size across all three domains. Convergence is rapid. $\mathrm { A t } n { = } 5 0$ , mean $W _ { 1 }$ between estimated and oracle distributions drops below $3 . 0 ;$ at $n { = } 1 0 0$ , below 2.0. Bootstrap CIs (95%) narrow monotonically, which indicates stable estimation with modest data.

Cold-start behavior. When an entity has few labeled documents, the entity-level $P _ { \mathrm { p o p } }$ estimate gets noisy. We simulate this by capping observation count at $n \in \{ 1 0 , 2 5 , 5 0 , 1 0 0 \}$ . Even at $n { = } 1 0 .$ , all estimation strategies beat uncalibrated Top-k. Smoothed interpolation with a domain prior $( \tilde { P } = \alpha \hat { P } _ { e } + ( 1 - \alpha ) P _ { \mathrm { d o m a i n } } , \alpha = n / ( n + 5 0 ) )$ works best at every count, achieving 61% $W _ { 1 }$ reduction at $n { = } 1 0 .$ . Our recommended production rule: use entity-level $P _ { \mathrm { p o p } }$ when $n \geq 2 5$ and at least 2 sentiment bins have mass $> 0 . 0 5 ;$ otherwise fall back to the smoothed domain prior.

![](images/301ebc414ab212ddb845e609b2d4e5d3f8ce8d397dbad48cd98455cb167a0e28.jpg)

![](images/11a8c907dbbeb4470e24e385a9abca738567bd1c18eee85cac00ea05ac20b154.jpg)

![](images/94233f7ab78f20f14c8ec051eabace1bf8ef29767be8bc545a5110c450efdfeb.jpg)

Label error rate (%)  
Figure 5: Mislabel sensitivity: $W _ { 1 }$ vs. label error rate. Dashed red line = Top-k baseline. $W _ { 1 }$ Minimizer never crosses Top-k even at 50% error on Yelp.  
![](images/d1ceecf2e266ab0b0a54f5256bc342623769c3c3510a42b455ae5bb449f18145.jpg)  
λ(relevance weight)  
Figure 6: λ-sensitivity for $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ (left) and OpinionMMR (right). $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ is stable at $\lambda \leq 0 . 5$ and degrades at $\lambda { = } 0 . 9$ . OpinionMMR shows higher $W _ { 1 }$ across all domains with limited sensitivity to λ.

![](images/626f5c7b8c5b067f456e0bcddb6c63c75f91a0708ed60a0dfebd4b0257d3826f.jpg)

![](images/4b26b789b99c8f70834f528052728cc3a0f3003a2702d2c9f9ad64344b25b5ae.jpg)

![](images/abe5289ecbbf04444628f2ee58b67dba30d9d9814f5ad4493b7e3b367857f4fb.jpg)  
N (candidate pool size)  
Figure 7: $W _ { 1 }$ vs. pool size $N \left( k { = } 2 0 \right)$ . $N { = } 1 0 0$ suffices for dense pools; $W _ { \mathrm { 1 } } { \mathrm { - } } \mathbf { M } \mathbf { M } \mathbf { R }$ benefits from larger pools on sparse data.

## G Generation Evaluation

## G.1 Judge Validation & Positional Bias

Generation evaluation should not hinge on a single model’s quirks; leniency bias and prompt sensitivity are well-documented failure modes (Thakur et al., 2025). We validate with five independent model families from different providers: Claude Sonnet 4 (Anthropic), Llama 3.3 70B (Meta), Mistral Large 3 (Mistral AI), Amazon Nova Pro (Amazon), and DeepSeek v3.2 (DeepSeek). Appendix A.3.3 contains the full judge prompt with its three evaluation dimensions. Below we confirm positional bias, report directional inter-judge agreement, and present majority-vote consensus.

$P _ { p o p }$ Estimation Convergence  
![](images/a0f8f1357c36bc070f2a53242813ce0c30f2cbce999de6709800855c616104cf.jpg)  
Figure 8: $W _ { 1 } ( \hat { P } _ { \mathrm { p o p } } , P _ { \mathrm { p o p } } ^ { * } )$ vs. entity-matched documents sampled. With 50–100 documents per entity, estimation error is comparable to or below the re-ranking improvement magnitude itself.

## G.1.1 Positional Bias

LLM judges exhibit positional bias, a preference for whichever answer appears first (Wang et al., 2024; Shi et al., 2025). We confirm this across all five judges in our panel. Our position-swap protocol converts inconsistent position-dependent preferences into conservative ties, motivating the directional $\kappa _ { d }$ analysis in §G.1.2. Bias rates differ across models, producing asymmetric tie distributions:

Table 18: Consensus tie rates across the 5-judge panel. Higher tie rates indicate stronger positional bias (position-swap converts inconsistent preferences to ties).
<table><tr><td>Judge</td><td>Consensus Tie Rate  $\left( k = 5 \right)$ </td><td>Consensus Tie Rate  $\scriptstyle ( k = 1 0 )$ </td></tr><tr><td>Claude Sonnet 4</td><td>46%</td><td>46%</td></tr><tr><td>Mistral Large 3</td><td>48%</td><td>61%</td></tr><tr><td>DeepSeek v3.2</td><td>63%</td><td>71%</td></tr><tr><td>Llama 3.3 70B</td><td>73%</td><td>86%</td></tr><tr><td>Amazon Nova Pro</td><td>79%</td><td>86%</td></tr></table>

When judges do commit to a preference, they agree directionally 85–100% of the time.

## G.1.2 Directional Inter-Judge Agreement $( \kappa _ { d } )$

We want to separate genuine directional disagreement from artifacts of unequal tie rates. So we compute $\kappa _ { d }$ over only those pairs where both judges commit to a non-tie preference:

Table 19: Mean directional $\kappa _ { d }$ per judge vs. all others (fairness dimension). Non-Anthropic judges agree at $\kappa _ { d } { \geq } 0 . 9 5$
<table><tr><td>Judge (vs. others)</td><td> $k { = } 5 ~ \kappa _ { d }$ </td><td>k=5 Agree%</td><td> $k { = } 1 0 ~ \kappa _ { d }$ </td><td>k=10 Agree%</td></tr><tr><td>Claude vs. others</td><td>0.68</td><td>88.2%</td><td>0.86</td><td>93.5%</td></tr><tr><td>Llama 3.3 vs. others</td><td>0.89</td><td>96.0%</td><td>0.97</td><td>98.8%</td></tr><tr><td>Mistral L3 vs. others</td><td>0.89</td><td>95.8%</td><td>0.95</td><td>97.8%</td></tr><tr><td>DeepSeek vs. others</td><td>0.93</td><td>97.4%</td><td>0.95</td><td>97.9%</td></tr><tr><td>Nova Pro vs. others</td><td>0.88</td><td>95.3%</td><td>0.94</td><td>97.3%</td></tr></table>

All five judges agree directionally at $\kappa _ { d } { = } 0 . 6 1 -$ 1.00. Among the four non-Anthropic judges, pairwise $\kappa _ { d }$ ranges 0.95–1.00 at both k values. Claude’s lower $\kappa _ { d } ~ ( 0 . 6 8 / k { = } 5 , 0 . 8 6 / k { = } 1 0 )$ traces to its lower tie rate: it commits more often and occasionally flags nuances others collapse into ties.

## G.1.3 Majority Vote Results (3/5 Judges Must Agree)

Under majority vote (3/5 agreement required), calibration methods reach strong significance at $k { = } 5$ while Random correctly fails. This is a critical sanity check. At k=10, fewer approaches survive; context-window averaging attenuates distributional differences. We extend to k=20 under the same protocol; the scaling regime distinguishes semanticdiversity from distributional methods.

At k=5 with only 2–3 active sentiment bins, diversity and calibration objectives partly overlap and MMR/OpinionMMR win 83%/82% (Table 2). By k=20 the pool has room for finer-grained proportions and pure semantic-diversity has run out of signal; MMR and OpinionMMR Fair% drop to 29% and 33%. Distributional methods maintain their advantage: WassRank OT rises from 70% (k=5) to 73% (k=10) to 92% (k=20) as budget lets proportional targeting express itself in the answer. DPP is a partial exception: it also grows with k (89% at k=20) because its determinantal spread lands one representative per bin when there is enough budget.

Domain decomposition at k=10 reveals a clear density dependency. On dense Yelp (60 questions, 10 aspects), all calibration methods achieve 86– 97% fairness decided rate; $W _ { 1 }$ Minimizer hits 97% (28W/1L). Dense OpinRank (60 questions, 10 entities) confirms propagation: DPP leads at 100% (19W/0L), $W _ { 1 }$ Minimizer at 94% (16W/1L), and 5 of 8 approaches reach $p < 0 . 0 0 1$ . Sparse Seller Forums (36 questions, 6 entities, EM∼40%) is the outlier: only OpinionMMR achieves significance (10W/2L, 83%, $\scriptstyle p = 0 . 0 1 9 )$ $W _ { 1 }$ Minimizer and $W _ { 1 } \cdot$ MMR hover near coin-flip (54–55%), confirming that sparse pools cap document quality regardless of re-ranking strategy. Random fails on all three domains (13–53%), validating the sanity check.

Table 20: 5-judge majority-vote generation results (cross-domain, 156 questions) at $k { \in } \{ 5 , 1 0 , 2 0 \}$ . All calibration methods achieve $p < 0 . 0 0 1$ at $k { = } 5 ;$ retain significance at $k { = } 1 0 \ ( p \ \leq \ 0 . 0 2 1 )$ and $k { = } 2 0 \ ( p \ \leq \ 0 . 0 0 4 )$ . Random fails at all k values. $\mathrm { A t } \ k { = } 2 0$ , MMR/OpinionMMR Fair% collapse (29%/33%) as semantic diversity saturates; distributional methods maintain or grow their advantage; WassRank OT rises 70%→73%→92%. $\mathrm { F a i r } \% = \mathrm { W } / ( \mathrm { W } + \mathrm { L } )$ $\mathrm { W i n } \% = \mathrm { W } / N$
<table><tr><td></td><td colspan="3"> $k { = } 5$ </td><td colspan="3"> $k { = } 1 0$ </td><td colspan="3"> $k { = } 2 0$ </td></tr><tr><td>Approach</td><td>W/L/T</td><td>Fair%</td><td>Win%</td><td>W/L/T</td><td> ${ \mathrm { \bf f a i r } } \%$ </td><td>Win%</td><td>W/L/T</td><td>Fair%</td><td>Win%</td></tr><tr><td>Random</td><td>16/36/104</td><td>31%</td><td>10%</td><td>12/40/104</td><td>23%</td><td>8%</td><td>3/26/127</td><td>10%</td><td>2%</td></tr><tr><td>MMR**</td><td>39/8/109</td><td>83%</td><td>25%</td><td>21/9/126</td><td>70%</td><td>13%</td><td>5/12/139</td><td>29%</td><td>3%</td></tr><tr><td>OpinionMMR **</td><td>50/11/95</td><td>82%</td><td>32%</td><td>25/8/123</td><td>76%</td><td>16%</td><td>7/14/135</td><td>33%</td><td>4%</td></tr><tr><td>DPp**</td><td>54/7/95</td><td>89%</td><td>35%</td><td>42/6/108</td><td>88%</td><td>27%</td><td>25/3/128</td><td>89%</td><td>16%</td></tr><tr><td> $\mathrm { W a s s R a n k ~ O T ^ { * * } }$ </td><td>66/28/62</td><td>70%</td><td>42%</td><td>75/28/53</td><td>73%</td><td>48%</td><td>33/3/120</td><td>92%</td><td>21%</td></tr><tr><td> $\mathbf { W _ { 1 } } – \mathbf { M M R } ^ { * * }$ </td><td>44/10/102</td><td>81%</td><td>28%</td><td>24/7/125</td><td>77%</td><td>15%</td><td>13/2/141</td><td>87%</td><td>8%</td></tr><tr><td> $\mathbf { W } _ { 1 } ~ \mathbf { M i n i m i z e r } ^ { * * }$ </td><td>51/8/97</td><td>86%</td><td>33%</td><td>32/8/116</td><td>80%</td><td>21%</td><td>27/5/124</td><td>84%</td><td>17%</td></tr></table>

## G.1.4 Judge Prompt Sensitivity

Our main fairness criterion conflates two signals: proportional accuracy and viewpoint coverage. We disentangle them by re-judging all answer pairs under two contrasting definitions (Prompt A and Prompt B in Appendix A.3.3), same 5-judge panel with position-swap (Zheng et al., 2023) and 3/5 majority vote:

• Prompt A (Proportional Accuracy): Judges receive the ground-truth $P _ { \mathrm { p o p } }$ and are asked which answer’s emphasis better matches the actual distribution. An answer overrepresenting a 10% minority relative to a 70% majority is penalized as misleading.

• Prompt B (Viewpoint Coverage): Judges are asked which answer covers more distinct viewpoints, including rare ones. Giving disproportionate space to minority views is rewarded.

INFORMATIVENESS and GROUNDEDNESS dimensions remain identical across both prompts, serving as controls.

Tables 21 and 22 reveal three patterns. (1) Under proportional accuracy, $W _ { 1 }$ Minimizer leads DPP by 8 pp at k=5 and 6 pp at k=10; retrieval calibration propagates when judges can verify proportions against ground truth. Stratified oracle sampling reaches 35% Fair% (Table 21) $- \ W _ { 1 }$ Minimizer sits within 1 pp of the oracle, confirming that Prompt A tracks the ground-truth $P _ { \mathrm { p o p } }$ rather than any specific retrieval strategy. (2) Under viewpoint coverage, the gap narrows to 3 pp (k=5) and 2 pp (k=10). DPP’s diversity advantage is real but modest once $W _ { 1 }$ ’s calibrated selection already covers most sentiment bins. (3) Inter-judge agreement rises under Prompt B (Llama–Mistral $\kappa \colon 0 . 3 9 {  } 0 . 6 2 \ { \mathrm { a t } } \ k { = } 5 )$ , which suggests viewpoint counting is more objectively evaluable than proportional matching without ground truth. Our original conflated criterion thus understates $W _ { 1 }$ ’s advantage at its design objective (proportional faithfulness) while slightly overstating DPP’s coverage benefit.

Table 21: Judge prompt ablation: fairness win rate (%) under proportional vs. coverage definitions (3 domains, 156 questions per approach). $W _ { 1 }$ Minimizer dominates under proportional accuracy; the gap narrows under coverage. Stratified sampling (oracle) confirms the Prompt A criterion tracks the ground-truth distribution, not any specific retrieval strategy — $W _ { 1 }$ Minimizer matches oracle Fair% within 1 pp at k=5.
<table><tr><td rowspan="2">Approach</td><td colspan="2">k=5</td><td colspan="2">k=10</td></tr><tr><td>Prop. (A)</td><td>Cov. (B)</td><td>Prop. (A)</td><td>Cov. (B)</td></tr><tr><td>Stratified (oracle)</td><td>35%</td><td></td><td></td><td></td></tr><tr><td>W1 Minimizer</td><td>34%</td><td>38%</td><td>30%</td><td>27%</td></tr><tr><td>DPP</td><td>26%</td><td>35%</td><td>24%</td><td>25%</td></tr><tr><td>W1-MMR</td><td>23%</td><td>36%</td><td>18%</td><td>17%</td></tr><tr><td>OpinionMMR</td><td>21%</td><td>30%</td><td>13%</td><td>19%</td></tr><tr><td>MMR</td><td>13%</td><td>26%</td><td>10%</td><td>13%</td></tr><tr><td>Random</td><td>16%</td><td>13%</td><td>12%</td><td>7%</td></tr></table>

Table 22: Control dimensions remain stable across prompt variants (k=5, ∆ = Prompt B − Prompt A). Shifts ≤4 pp confirm the fairness prompt change does not contaminate other evaluation axes.
<table><tr><td>Approach</td><td>Info% (A)</td><td>Info% (B)</td><td>Grnd% (A)</td><td>Grnd% (B)</td></tr><tr><td>W1 Minimizer</td><td>44%</td><td>40%</td><td>28%</td><td>24%</td></tr><tr><td>DPP</td><td>38%</td><td>40%</td><td>27%</td><td>24%</td></tr><tr><td>Random</td><td>19%</td><td>18%</td><td>15%</td><td>8%</td></tr></table>

## G.2 Single-Judge Metric Isolation (Generation)

The 5-judge majority-vote protocol is conservative — positional-bias-driven ties inflate the tie column. To isolate the ordinal metric’s end-to-end contribution with more statistical power, we ran a controlled single-judge (Claude Sonnet 4.6) positioncontrolled pairwise evaluation between $W _ { 1 }$ Minimizer and KL(JS) Minimizer at k=5 (Table 23). Same algorithm, same entity-filtered pool; only the distance function differs.

Table 23: $W _ { 1 }$ Minimizer vs. KL(JS) Minimizer, generation evaluation at k=5 (156 queries, single-judge Claude Sonnet 4.6, position-controlled, blind pairwise vs. Top-k). W<sub>1</sub> leads KL(JS) by +7 pp cross-domain; both significantly beat Top-k (p<0.001).
<table><tr><td>Domain</td><td>Method</td><td>N</td><td>W/L/T</td><td>Fair%</td><td>Win%</td></tr><tr><td>Seller Forums</td><td>W1 Minimizer</td><td>36</td><td>10/9/17</td><td>53%</td><td>28%</td></tr><tr><td></td><td>KL (JS) Minimizer</td><td>36</td><td>7/10/19</td><td>41%</td><td>19%</td></tr><tr><td>Yelp</td><td>W1 Minimizer</td><td>60</td><td>39/4/17</td><td>91%</td><td>65%</td></tr><tr><td></td><td>KL (JS) Minimizer</td><td>60</td><td>31/8/21</td><td>79%</td><td>52%</td></tr><tr><td>OpinRank</td><td>W1 Minimizer</td><td>60</td><td>29/8/23</td><td>78%</td><td>48%</td></tr><tr><td></td><td>KL (JS) Minimizer</td><td>60</td><td>35/10/15</td><td>78%</td><td>58%</td></tr><tr><td>Cross-domain</td><td>W1 Minimizer</td><td>156</td><td>78/21/57</td><td>79%</td><td>50%</td></tr><tr><td></td><td>KL (JS) Minimizer</td><td>156</td><td>73/28/55</td><td>72%</td><td>47%</td></tr></table>

$W _ { 1 }$ leads KL(JS) by +7 pp Fair% cross-domain, and the retrieval-side 8–10% $W _ { 1 }$ advantage (Appendix C.1) does not invert at generation time. Yelp shows the largest gap (91% vs. 79%). Seller Forums shows the same directional advantage at a smaller magnitude (53% vs. 41%). OpinRank is a partial exception: Fair% ties at 78% and KL(JS) edges $W _ { 1 }$ on Win Rate (58% vs. 48%), which we attribute to OpinRank’s near-uniform $P _ { \mathrm { p o p } }$ compressing ordinal distances — when adjacent-bin errors and pole-to-pole errors carry similar cost in the target itself, the ordinal ground cost has less to do.