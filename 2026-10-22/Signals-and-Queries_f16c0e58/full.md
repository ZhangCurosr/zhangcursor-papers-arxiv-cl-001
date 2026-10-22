# Signals and Queries:

# Token-Eficient Agentic Context Discovery

Yam Marcovitz

yam@parlant.io

## Abstract

User-facing LLM agents must identify which behavioral policies apply at each turn, but evaluating every policy with an LLM is costly. We introduce Signals and Queries (S&Q), a lightweight first-stage relevance filter developed for Parlant 4.0 that removes at least 30% of candidate policies before downstream evaluation while preserving high recall. For each policy, an LLM generates activating examples and hard negatives ofline. An independent logistic classifier trained on their embeddings then predicts relevance using embeddings and vector dot products, with no per-turn LLM calls. This shifts expensive relevance evaluation to a smaller candidate set, while allowing each policy’s classifier to be updated independently. Evaluated against 9,144 LLM-judge relevance labels covering 254 τ<sup>2</sup>-bench conversation prefixes and 36 airline policies, S&Q achieves higher precision than top-K and cosine-threshold retrieval at every evaluated recall level, with consistent results across two embedding providers. These findings support inexpensive, per-policy classification as a practical first stage for reducing downstream evaluation workload in customer-facing agents.

## 1 Introduction and Motivation

Agentic LLM applications increasingly separate what the agent could know from what the agent should see right now. A customer-facing agent may be governed by dozens of behavioral policies; it may also carry a domain glossary, tool and skill descriptions, state observations, and reference documents. Placing all of it in every prompt is expensive and counterproductive, for three well-documented reasons. Long contexts sufer from the lost-in-the-middle efect: models systematically under-attend to information positioned in the middle of the prompt [1]. They sufer from attention dilution: as the prompt grows, the attention mass allotted to any individual instruction shrinks, so its influence on generation fades [2]. And they sufer from the curse of instructions: the probability of following all instructions decays approximately as the per-instruction success rate raised to the number of instructions, so adherence degrades steadily as inventories grow [3]. On top of these quality failures, per-turn token cost scales linearly with inventory size rather than with what the turn actually needs.

The problem is acute for policies. In behavioral governance frameworks such as Parlant [4], policies are normative instructions (“only compensate if the customer holds travel insurance. . . ”) whose omission is a correctness failure, not merely a quality regression: an agent that does not see the cancellation policy while handling a cancellation may take actions it should have refused. Relevance filtering for policies is therefore a recall-critical problem: a false negative can produce a policy violation, while a false positive merely wastes a few hundred prompt tokens that modern models tolerate well. This asymmetry drives the design of everything that follows.

One accurate solution is to ask an LLM, per turn and per policy, “is this policy relevant now?”; this is the first-pass ranker used in Parlant’s engine. It works, but its cost scales with the inventory: for our 36-policy benchmark it consumes ∼69,000 prompt tokens and 36 LLM calls per user turn (§4.4). The alternative, embedding-based retrieval of the policy text against the conversation, is nearly free but sufers from a genre gap: conversations are informal user utterances while policies are formal normative prose, and their embed dings separate poorly [5]. A third architecture avoids the problem by construction: routed flow-graphs place diferent instructions at diferent nodes, so each prompt stays small. But this fragments the agent’s knowledge. Real customer conversations are multi-turn and multi topic, and drift freely across node boundaries. The node in control at any given moment often lacks policies the situation has come to require, so the agent is least coherent on the messy, multi-topic conversations that matter most. A fourth approach, taken by skill-based agent frameworks [6, 7], keeps only a short description of each skill resident and loads a skill’s full instructions on demand, once the model, reasoning over those descrip tions, decides it applies. This keeps whole instructions out of the prompt, but makes discovery part of the model’s own reasoning loop, slower and costlier than an out-of-loop lookup.

We propose Signals and Queries (S&Q): describe each artifact not (only) by its text but by signals (LLMgenerated examples of user messages that should activate it) and anti-signals (hard near-misses that should not). A tiny per-artifact classifier trained on these exemplars then scores live conversation turns (queries) in the embedding space. While we develop and evaluate S&Q on agent policies, nothing in the formulation is policyspecific: any context artifact that can be described by exemplar activating queries is a candidate target, including state observations, glossary entries, skills and tools [8, 9], and code or document sections.

Contributions. (1) A formulation of agentic context discovery as recall-first, per-artifact binary relevance, decided by classifiers over LLM-generated signals and anti-signals; the design is decoupled: each artifact’s classifier and threshold depend only on its own exemplars, so adding or editing an artifact retrains nothing else. (2) A benchmark against an LLM judge on $\tau ^ { 2 } .$ -bench conversations [10, 11] with 36 policies, comparing four embedding-retrieval baselines; S&Q Paretodominates all of them and replicates across two embedding providers. (3) A cost analysis showing a ${ \sim } 1 0 ^ { 5 } \times$ reduction in per-turn cost relative to per-policy LLM ranking. (4) An analysis of failure modes showing that signal and anti-signal quality (coverage, hardness, and domain match) is the dominant factor, motivating exemplar generation as a research problem in its own right.

This work continues our ongoing pursuit of accuracy in customer-facing agents at Parlant. It complements our earlier work on Attentive Reasoning Queries [12], which optimizes instruction-following once the relevant policies are already in the agent’s context; S&Q addresses the upstream problem of discovering which policies belong there in the first place.

## 2 The S&Q Method

## 2.1 Definitions

Let $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { N } \}$ be the artifact inventory (policies). For each artifact $a _ { j }$ an LLM generates, ofline:

• Signals $S _ { j } = \{ s _ { j 1 } , \ldots , s _ { j m } \}$ : short, natural user messages that should activate $a _ { j }$ (“I need to cancel my flight to Chicago next week”).

• Anti-signals $\bar { S } _ { j } ~ = ~ \{ \bar { s } _ { j 1 } , \ldots , \bar { s } _ { j m } \}$ : hard nearmisses, i.e., messages in the same topical neighborhood that should not activate $a _ { j }$ because a neighboring artifact governs them (for a cancellation policy: “I need to change my flight from Tuesday to Thursday”).

We use $m = 1 0$ of each. This is the document-side analogue of query expansion: signals play the role of predicted queries as in Doc2Query [13], and are generated without any conversation data, as in synthetic-query domain adaptation [14]; anti-signals additionally supply targeted hard negatives, which cosine similarity alone cannot exploit.

A query is the live conversation at its current state. At turn t we form two kinds of view, both centered on the artifact-inventory centroid c (centering mitigates the anisotropy of embedding spaces [5, 15]): the cumulative view $q _ { \mathrm { c u m } } ^ { ( t ) }$ , the embedding of the whole conversation through turn $t ,$ and a user-turn view $q _ { \mathrm { u s e r } } ^ { ( u ) }$ , the embedding of the single user message sent at each turn $u \leq t .$ All embeddings are $\ell _ { 2 } .$ -normalized.

## 2.2 Per-artifact classifier

For each artifact independently, we fit a logistic regression on the embedded exemplars: positives $S _ { j }$ , negatives ${ \bar { S } } _ { j } { \mathrm { : } }$

$$
P ( { \mathrm { r e l e v a n t } } \mid q ) = \sigma ( w _ { j } \cdot q + b _ { j } ) .\tag{1}
$$

At inference, the score of artifact $a _ { j }$ for the conversation at turn t is a max-pool over query views,

$$
z _ { j } ( t ) = \operatorname* { m a x } \left( w _ { j } \cdot q _ { \mathrm { c u m } } ^ { ( t ) } , \ : \operatorname* { m a x } _ { u \leq t } \ : w _ { j } \cdot q _ { \mathrm { u s e r } } ^ { ( u ) } \right) + b _ { j } ,\tag{2}
$$

and $a _ { j }$ is retrieved if $z _ { j } ( t ) > \theta$ . The max over user turns keeps a policy discoverable from the turn that activates it, regardless of what the conversation discusses before or after: a single blended embedding of a long, multi-topic conversation dilutes any one topic, whereas the per-turn max preserves each turn’s own evidence. On the short, single-topic conversations of our benchmark this makes no diference, but it matters when the conversation shifts topic, which we quantify in §4.2. The margin θ is a single global knob trading recall for precision; we operate recall-first at $\theta < 0$ because real conversation turns score systematically lower than clean, idealized signals (a visible domain shift; Fig. 2).

## 2.3 Design goals and properties

Recall-first. As argued above, the loss function of policy discovery is asymmetric. S&Q therefore targets ≥90% recall per artifact and optimizes precision only subject to that constraint. The retrieved set is a prefilter : the agent’s own LLM reads the surviving artifacts and can ignore inapplicable ones cheaply, whereas it can never recover an artifact that was filtered out.

Decoupling. Because artifact $a _ { j }$ ’s classifier uses only $S _ { j }$ and $\bar { S } _ { j }$ , inventory edits are local: adding, removing, or rewriting one artifact requires generating its exemplars and fitting one classifier (∼2 ms), leaving all others untouched. In an earlier variant we trained each artifact against all other artifacts’ signals, which couples nearby artifacts through their negatives: when two policies sit close in embedding space, each falls in the other’s negative set, so the two learned boundaries push their shared region to the negative side of both, and a turn genuinely relevant to either is rejected by both, a false negative on each. The efect compounds as the inventory grows and near-duplicates accumulate. Self-contained antisignals remove the coupling structurally: each artifact’s negatives are its own near-misses, so no other policy’s presence can suppress it.

![](images/f571110da9c4a2d268e5b0fe02595b62ba7f5acda237271d04ad98405c4317c7.jpg)

![](images/5a62264b9835f3c31257b7e91ea4eb3692629e5bb482165223c882c702cdcefa.jpg)

Figure 1: One live conversation turn (query, star) projected into two artifacts’ classifier spaces (x-axis: the artifact’s own discriminant w·q + b; y-axis: first residual PC of its exemplars). In the Cancel flight space the query lands among the signals (relevant); in the Traveling with Pets space the same query lands among the anti-signals (not relevant).  
![](images/3206289d4c172a9bfbb2c479cfb1b8ba2eb785268a568cb6f452c605acb5eb08.jpg)  
Figure 2: Where points land on one artifact’s logistic classifier (Cancel flight). Signals and anti-signals separate cleanly; real conversation queries labeled relevant by the LLM judge (stars) score above the irrelevant ones (crosses) but below the signals. This signal→query domain shift motivates the recall-first margin $\theta < 0$

Near-synonym robustness. Anti-signals place the decision boundary along an artifact’s most confusable neighbors (cancel vs. modify vs. rebook), where naive cosine retrieval fails.

Token eficiency. All LLM work happens ofline (2N generation calls per inventory, re-run only for changed artifacts). Online, a turn costs one embedding call for the new text plus N dot products.

## 2.4 Capabilities relative to embedding retrieval

Having described the method, we make explicit the structural properties that separate it from ordinary embedding retrieval; §4 then quantifies their efect. Table 1 compares S&Q with the two inference-free retrieval families it most directly competes with: top-K retrieval, which ranks artifacts by query similarity and returns the K highest, and cosine-threshold retrieval, which returns every artifact scoring above a global threshold.

Table 1: Capabilities of S&Q relative to the two inferencefree embedding-retrieval families it competes with (the baselines of §3). ✓: supported; ×: not supported; (✓): partial. Each property is defined and motivated in the text; the signal-scored baseline variants (B2, B4) do not change any entry.
<table><tr><td>Property</td><td>top-K</td><td>Cosine thr.</td><td>S&amp;Q</td></tr><tr><td>Inference-free per turn</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Adaptive selection size</td><td>×</td><td>√</td><td>√</td></tr><tr><td>Uncapped recall</td><td>×</td><td>√</td><td>√</td></tr><tr><td>Near-synonym discrimination</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Per-artifact decision boundary</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Portable operating point</td><td>(√)</td><td>×</td><td>√</td></tr></table>

Because all three avoid per-turn language-model calls, the decisive question is not cost but capability, namely which properties a method needs in order to serve as a reliable, recall-first policy pre-filter. We identify six.

• Inference-free per turn. Relevance is decided without any language-model call on the live turn, at the cost of a single embedding lookup. All three methods share this property; it is the common premise that makes the remaining, capability-level distinctions the deciding ones.

• Adaptive selection size. The count of genuinely applicable policies varies per turn, so a method should return however many clear the relevance bar rather than a fixed number. A ranked cutof cannot: any constant K is at once too large for simple turns and too small for multi-intent ones, discarding the artifact ranked K+1 however relevant it is.

• Uncapped recall. Because the number of relevant policies varies, a fixed cutof also bounds achievable recall from above (top-K saturates near 66% on our data, §4). A policy pre-filter, whose omissions are correctness failures rather than mere ineficiencies, must be able to approach full recall.

• Near-synonym discrimination. Policies often share a topical neighborhood (cancellation, modification, rebooking), and a method must separate an artifact from its closest neighbors, where surface similarity peaks. Ranking by cosine similarity, whether against policy text or signals, cannot do this; contrasting signals against hard-negative anti-signals places the boundary there.

• Per-artifact decision boundary. top-K and cosine thresholding apply one global knob (K or τ) to every artifact, implicitly assuming a shared score scale. S&Q instead fits an independent boundary per artifact, letting each intercept absorb that artifact’s own score ofset, which lets a single global margin govern a heterogeneous inventory.

• Portable operating point. An operating point should transfer across embedding providers and query types without re-tuning. Absolute cosine values reflect each model’s geometry, so a threshold must be recalibrated whenever either changes; S&Q’s margin, measured from each learned boundary, needs only mild per-provider adjustment. top-K’s K transfers as an integer but denotes no calibrated relevance level, hence its partial mark.

## 3 Experimental Setup

Domain and conversations. We use the airline domain of τ<sup>2</sup>-bench [11], the leading customer-service agent benchmark suite, maintained by the research department of Sierra.ai [10]. We take 50 simulated customer conversations produced by a Parlant agent run against the benchmark’s user simulator. Every conversation prefix ending at a user turn is a query state: 254 sub-sequences.

Artifact inventory. N=36 policies: the 6 sections of the benchmark’s airline policy plus 30 additional realistic policies authored for this study (identity verification, baggage claims, unaccompanied minors, fare rules, fraud handling, etc.), including deliberate near-synonym clusters to stress discrimination. Signals and anti-signals (10+10 per policy) are generated by gemini-3.5-flash from the policy text alone, with no conversation data.

Ground truth. An LLM judge scores every (subsequence, policy) pair for relevance on a 1–5 scale: 9,144 labels from gemini-3.1-flash-lite, using the relevance-ranking prompt of Parlant’s production engine. Following the engine’s own definition, a pair is relevant if score ≥4. LLM judges are imperfect [16]; we discuss the implications in §6. Metrics are macro-averaged over the 31 policies with nonzero base rate.

Baselines. All methods share the same query views and max-pooling of Eq. 2 and the same centered-cosine geometry; they difer only in scoring: (B1) top-K policy: cos $( q , a _ { j } )$ against the policy text, retrieve the top K; (B2) top-K signal: max<sub>i</sub> cos $( q , s _ { j i } )$ , top K; (B3) threshold policy: cos ${ ( q , a _ { j } ) > \tau ; ( B _ { 4 } ^ { \prime } ) }$ threshold signal: max<sub>i</sub> cos $( q , s _ { j i } ) \ > \ \tau$ Sweeping each method’s knob traces its recall–precision frontier.

Embedders. The full pipeline is run twice: gemini-embedding-001 and OpenAI text-embedding-3-large.

## 4 Results

S&Q Pareto-dominates the baselines. Figure 3 shows the frontiers. At every recall level S&Q yields the highest precision. The margin is widest in the mid recall range, about +4 points near 60–70% recall on both providers, and narrows toward the high-recall extreme, where the two providers diverge: it holds at +2.9 points on OpenAI but falls to +0.9 on Gemini. The operating points of Table 2 sit at that extreme, which is why they show a clear S&Q lead on OpenAI (34% vs. 31%) but a narrow one on Gemini (32% vs. 31%). Unlike top-K, whose recall is capped near 66%, S&Q can also be pushed to full recall as θ decreases.

Recall--precision Pareto frontier vs. the LLM judge (36 policies)  
![](images/cb9b3981a1e87d29630f12a2e4f84c6f4fe6b4d9520858776ac2a78db328b5c6.jpg)

Figure 3: Macro recall–precision Pareto frontiers vs. the LLM judge. S&Q (blue) dominates all four baselines at every recall level, under both embedders (solid: Gemini; dashed: OpenAI). Fixed-K retrieval cannot exceed ∼66% recall because the number of simultaneously relevant policies varies per turn.  
![](images/d4b78894a4c6335c9bb59af417430cd54785e580c0f5a29feae49aeba905adb6.jpg)  
Figure 4: Recall-first operating points (Gemini): best precision subject to macro recall ≥90%; top-K methods shown at their maximum-recall setting K=15, which still fails the recall target.

Fixed-K retrieval cannot fit this problem. Even at K=15 of 36 policies, top-K recall saturates at ∼66%: the number of simultaneously relevant policies varies widely per turn (broad conduct policies are always in play; several topical policies co-activate during multiintent turns), so no single K fits. Relevance discovery needs per-artifact decisions, not a ranked cutof.

The gain comes from the classifier, not the signals. Threshold retrieval over raw signal embeddings (B4) performs no better than over policy text (B3). The gain appears only when signals are contrasted with anti signals through a trained discriminant, which reorients the decision axis toward what distinguishes the artifact from its neighbors rather than what makes it generically similar to conversation text.

Trained boundaries port better than cosine thresholds. Although threshold-based cosine retrieval reaches decent recall-first operating points (Table 2), the threshold itself is a fragile quantity that is hard to generalize: absolute cosine values are artifacts of each embedding model’s geometry and of the query type, not calibrated relevance scores, so a threshold tuned in one configuration does not transfer to another. This is visible even within our benchmark: the best thresholds shift between providers (policy-text τ=−0.025 on Gemini vs. −0.050 on OpenAI; signals +0.225 vs. +0.275) and difer substantially between scoring types on the same provider (policy-text vs. signal similarities). S&Q mitigates this: because each artifact’s boundary is learned from its own exemplars, the intercept $b _ { j }$ absorbs the artifact-, provider-, and query-type-specific score scale, leaving a single global margin θ whose meaning (distance from each artifact’s own decision boundary) is stable. In practice θ needed only mild per-provider adjustment (−0.45 vs. −0.4), whereas raw-cosine thresholds must be recalibrated from scratch.

![](images/042c99e7ed173dd3b16d77952da9e0cce0668a781c39837bba48af5c545a6a26.jpg)  
Figure 5: S&Q per-policy recall (left) and precision (right) under the two embedding providers, at each provider’s recall-first operating point (16 highest-base-rate policies; black line: mean).

Table 2: Recall-first operating points (best macro precision s.t. macro recall ≥90%). Top-K methods never reach 90% recall; shown at $K { = } 1 5$ . <sup>∗</sup>This operating point is not fixed: §5 shows S&Q improves it substantially from exemplars mined from deployment trafic.
<table><tr><td rowspan="2">Method</td><td colspan="2">Gemini</td><td colspan="2">OpenAI</td></tr><tr><td>R%</td><td>P%</td><td>R%</td><td>P%</td></tr><tr><td>top-K (policy text)</td><td>66</td><td>40</td><td>64</td><td>41</td></tr><tr><td>top-K (signals)</td><td>65</td><td>39</td><td>64</td><td>39</td></tr><tr><td>threshold cos (policy)</td><td>93</td><td>31</td><td>94</td><td>31</td></tr><tr><td>threshold cos (signals)</td><td>91</td><td>31</td><td>92</td><td>31</td></tr><tr><td>S&amp;Q (ours)*</td><td>93</td><td>32</td><td>91</td><td>34</td></tr></table>

## 4.1 Generalization across embedders

Figure 5 shows per-policy S&Q recall and precision for the 16 highest-base-rate policies under both providers, at each provider’s recall-first operating point. The behavior transfers: most policies keep high recall (13 of 16 above 88% on Gemini, 12 on OpenAI), the macro operating point is close across providers (Gemini 93/32, OpenAI 91/34), and the method ordering on the frontier is unchanged (Fig. 3, dashed). Per-policy characteristics do shift with the embedder, a reminder to validate the global margin θ per provider, but no re-design or re-generation of exemplars was needed.

![](images/1daa0379f5c97d868cdebb047a140d82c4a5c943bd25fe1ad7e94520a46f891e.jpg)  
Figure 6: Per-turn relevance-discovery cost (log scale): the per-policy LLM judge vs. S&Q. All of S&Q’s LLM usage is a one-time ofline generation step; its per-turn cost is one embedding call and $3 7 \mu \mathrm { s }$ of CPU.

## 4.2 Ablations

Table 3 varies one component at a time from the full method, reporting the area under the macro recall– precision curve (AUPRC) and precision at 90% recall (P@R90) under both embedders. Three findings stand out. The negatives are what matter. Removing them, scoring by nearest-signal similarity alone, is the single largest regression (AUPRC 0.47→0.42; P@R90 32→31), confirming that the contrastive discriminant, not the signals in isolation, drives the result (cf. baseline B4 in Table 2). Self-contained anti-signals and generic otherpolicy negatives are near-identical in accuracy (generic is marginally higher), so anti-signals buy their decoupling and near-synonym robustness (§2) at no measurable accuracy cost. The classifier is not a sensitive choice, for a measurable reason. Logistic regression and a linear SVM tie (Gemini 0.471 vs. 0.463; OpenAI 0.475 vs. 0.473). A linear SVM is the natural challenger here: the exemplar sets are small, high-dimensional, and separable, exactly the regime where its max-margin objective, which ignores easy points far from the boundary, is expected to help. It does not, and a direct measurement shows why. Fit to the same 2m=20 exemplars, the two classifiers learn almost the same direction (mean $\cos ( w _ { \mathrm { L R } } , w _ { \mathrm { S V M } } ) { = } 0 . 9 8$ under both embedders). The classes are linearly separable for every policy, yet every exemplar is a support vector: because anti-signals are deliberately hard near-misses, no example sits outside the margin, so the one situation in which hinge and log loss diverge (their difering treatment of easy, farfrom-boundary points) never arises. The hard-negative design that sharpens the boundary also removes the degree of freedom on which SVM could have improved. A signed k-NN over the same exemplars, natural given the small per-artifact sample, is clearly worse (0.40): a nearest-exemplar rule cannot suppress the pervasive genre similarity between conversations and policy prose that a learned linear boundary discounts. We report logistic regression throughout not for accuracy, which is a tie, but because its probabilistic output gives a deployment more configurable operating knobs: a threshold expressible as a relevance probability, or derived from an explicit false-negative/false-positive cost ratio, where a raw SVM margin ofers only an uncalibrated score until it too is passed through a logistic calibration. Centering and exemplar count are second-order. Centering gives a small, consistent lift, and doubling the exemplars from m=5 to m=10 raises AUPRC monotonically, reinforcing that exemplar generation (§6) is the cheapest available lever.

Table 3: Ablations: each block varies one component from the full method (bold = default; others held at pooling both, anti-signals, centered, logistic regression, m=10). AUPRC is the area under the macro recall–precision curve; P@R90 is macro precision at 90% recall; both are averaged over the 31 occurring policies. G = Gemini, O = OpenAI.
<table><tr><td></td><td colspan="2">AUPRC</td><td colspan="2">P@R90 %</td></tr><tr><td>Variant</td><td>G</td><td>0</td><td>G</td><td>0</td></tr><tr><td>Query pooling</td><td></td><td></td><td></td><td></td></tr><tr><td>cumulative only</td><td>.526</td><td>.531</td><td>34</td><td>34</td></tr><tr><td>max-over-turns only</td><td>.462</td><td>.467</td><td>32</td><td>33</td></tr><tr><td>both (maxturn)</td><td>.471</td><td>.475</td><td>32</td><td>34</td></tr><tr><td>Negatives</td><td></td><td></td><td></td><td></td></tr><tr><td>anti-signals</td><td>.471</td><td>.475</td><td>32</td><td>34</td></tr><tr><td>generic (other policies)</td><td>.501</td><td>.498</td><td>32</td><td>33</td></tr><tr><td>none (positive-only)</td><td>.423</td><td>.429</td><td>31</td><td>31</td></tr><tr><td>Centering</td><td></td><td></td><td></td><td></td></tr><tr><td>on</td><td>.471</td><td>.475</td><td>32</td><td>34</td></tr><tr><td>off</td><td>.463</td><td>.473</td><td>32</td><td>33</td></tr><tr><td>Classifier</td><td></td><td></td><td></td><td></td></tr><tr><td>logistic reg.</td><td>.471</td><td>.475</td><td>32</td><td>34</td></tr><tr><td>linear SVM</td><td>.463</td><td>.473</td><td>32</td><td>33</td></tr><tr><td>signed k-NN</td><td>.399</td><td>.396</td><td>30</td><td>30</td></tr><tr><td>Exemplars per policy</td><td></td><td></td><td></td><td></td></tr><tr><td>m=5</td><td>.436</td><td>.436</td><td>31</td><td>33</td></tr><tr><td>m=10</td><td>.471</td><td>.475</td><td>32</td><td>34</td></tr></table>

Table 4: Multi-topic stress test: recall (%) on the recent topic’s (B’s) policies at the end of a stitched A-then-B con versation, at a matched false-positive rate. Max-over-turns recovers the recent topic that the blended cumulative embedding dilutes; the method’s both pooling tracks max-overturns. G = Gemini, O = OpenAI.
<table><tr><td></td><td colspan="2">Gemini</td><td colspan="2">OpenAI</td></tr><tr><td>FP rate</td><td>cum.</td><td>max-turns</td><td>cum.</td><td>max-turns</td></tr><tr><td>10%</td><td>34</td><td>51</td><td>35</td><td>57</td></tr><tr><td>20%</td><td>47</td><td>65</td><td>47</td><td>66</td></tr><tr><td>30%</td><td>58</td><td>70</td><td>55</td><td>74</td></tr></table>

Pooling addresses a regime the benchmark does not contain. Max-over-turns does not raise aggregate AUPRC over the cumulative view and here slightly lowers it, but the two guard against diferent failures. The cumulative view embeds the whole transcript as one vector, blending every topic discussed; when a con versation dwells on one topic and then shifts, the recent topic is a small fraction of that blend, so its policies are diluted, or even pushed toward their own anti-signals by the earlier of-topic content, and a policy relevant only to the latest turns can be missed. Scoring each turn in isolation and taking the max recovers it. We confirm this with a controlled multi-topic stress test: we stitch pairs of diferent-task conversations into a single A-then-B dialogue and, at B’s final turn, measure recall on B’s recent-topic policies, taking their ground truth from the ranker’s labels for B in isolation. At a matched false positive rate (each pooling’s threshold set to flag the same fraction of irrelevant policies), cumulative pooling recovers 12 to 19 fewer points of the recent topic’s poli cies than max-over-turns at a 30% false-positive rate, a gap that holds across false-positive rates from 10 to 30% (Table 4); and prepending topic A lowers B’s cumulative scores (by ∼0.016) while leaving the per-turn scores intact. These stitched conversations are synthetic, built to induce the topic shift the primary benchmark lacks, so this is a mechanism demonstration, and the advan tage is clearest on B’s recent-topic policies in aggregate, weaker on the subset unique to B. The benchmark’s own dialogues are short and single-intent (median ∼1,600 characters), so the mechanism seldom fires there and max-over-turns only adds a few false positives; we retain it in the method for multi-topic robustness while noting it is not what earns the benchmark numbers.

Table 5: Fine-tuning the encoder (SetFit) versus S&Q’s frozen encoder, at the recall-first operating point. On the same local base model (all-MiniLM-L6-v2, 384-dim), finetuning does not beat a logistic head on frozen features, and both match the far larger frozen API embedders. Same benchmark, exemplars, and pooling throughout.
<table><tr><td>Method</td><td>Encoder</td><td>R%</td><td>P%</td></tr><tr><td>S&amp;Q (ours)</td><td>frozen Gemini (API)</td><td>93</td><td>32</td></tr><tr><td>S&amp;Q (ours)</td><td>frozen OpenAI (API)</td><td>91</td><td>34</td></tr><tr><td>frozen + logreg</td><td>frozen MiniLM (local)</td><td>91</td><td>32</td></tr><tr><td>SetFit</td><td>fine-tuned MiniLM (local)</td><td>90</td><td>31</td></tr></table>

## 4.3 Comparison with SetFit

S&Q freezes the encoder and fits only a logistic head; a natural question is whether fine-tuning the encoder on the same exemplars would do better. SetFit [17] is the direct test: it contrastively fine-tunes a sentence encoder on a few labeled examples before fitting a logistic head. We run it per policy on the same signals and anti-signals, with the identical benchmark and pooling, and, to separate the efect of fine-tuning from the choice of encoder, also run S&Q’s frozen-encoder recipe on the same local base model. Table 5 shows that fine tuning does not help: SetFit trails the frozen head on the same encoder (90/31 vs. 91/32), and both match the far larger frozen API embedders. With only 2m=20 exemplars per policy, contrastive fine-tuning has little to exploit and no headroom over a linear head on frozen features, consistent with our finding that exemplar coverage, not the classifier or the encoder, is the binding constraint (§6). The comparison also reinforces that S&Q is encoder-agnostic: a small frozen local model sufices, so the method depends on no particular or heavyweight embedder. Fine-tuning would additionally forgo S&Q’s central eficiency, requiring a trained model per artifact in place of a single embedding call.

## 4.4 Cost

Table 6 and Figure 6 quantify the headline claim. Perpolicy LLM judging costs ∼69k prompt tokens and 36 model calls per user turn (7.6k characters per policy prompt, measured with the production prompt builder); S&Q costs one embedding call for the new turn text plus 37 µs of CPU for all 36 policies, with zero per-turn LLM usage. The entire ofline footprint, 72 generation calls and 80 ms of classifier training, amortizes over every subsequent conversation and is re-incurred only for changed artifacts.

Table 6: Measured per-turn cost of relevance discovery over N=36 policies. The LLM-judge row uses the production ranker prompt on a median-length conversation prefix.
<table><tr><td></td><td>LLM judge</td><td>S&amp;Q</td></tr><tr><td>LLM calls / turn</td><td>36</td><td>0</td></tr><tr><td>Prompt tokens / turn</td><td>~69,000</td><td>0</td></tr><tr><td>Embedding calls / turn</td><td>0</td><td>1</td></tr><tr><td>CPU / turn</td><td> $\mathrm { n / a }$ </td><td> $3 7 \mu \mathrm { s }$ </td></tr><tr><td>Offline LLM calls</td><td>0</td><td>2N=72 (once)</td></tr><tr><td>Offline training</td><td> $\mathrm { n / a }$ </td><td>80 ms (all 36)</td></tr></table>

S&Q does not fully match the judge’s precision at equal recall; it is a pre-filter whose residual false positives are cheap, and which can be composed with a downstream judge over the (much smaller) surviving set when higher precision is required.

## 5 Scalability

Every component we varied other than the exemplars proved inert. The classifier family is immaterial: logistic regression and a linear SVM learn nearly the same direction (mean cos =0.98) and tie, while a nearest-neighbour rule over the same exemplars is clearly worse (§4.2). The encoder is immaterial: fine-tuning it on the same exemplars does not beat a logistic head on frozen features, and a small local model matches the far larger API embedders (§4.3). Exemplar count saturates: extend ing each artifact’s anti-signal pool from ten to thirty, with per-item hardness held constant, leaves precision at 32.3% versus 32.4%, because 94% of the additional anti-signals fall within 0.6 cosine of one already present and the pool’s internal diversity does not move. That leaves the exemplars themselves, and specifically their provenance.

We therefore replaced generated exemplars with mined ones. We split the scenarios in half, mined exemplars from one half, and evaluated on the other. Within the mined half a signal is the transition turn, the user turn at which the judge’s score for an artifact first reaches its maximum and therefore the turn that ac tually triggered it, and an anti-signal is a user turn the judge scores as clearly irrelevant; we take four antisignals per signal, selecting those closest to the artifact’s own signals. Everything else, the classifier, the encoder, the pooling, the artifacts, and the held-out items, is unchanged. Table 7 and Fig. 7 give the result: precision at ≥90% recall rises from 33.9% to 43.5% on Gemini and from 35.3% to 41.6% on OpenAI, with AUPRC rising by 0.127 and 0.112. The same upgrade on a small local encoder moves precision from 32.6% to 39.7%, so the gain belongs to the exemplars rather than to any particular embedding model.

![](images/51d2b928d755bc2ef035d61a4a2df3b7d9acd8271ca6ca54333202cb36fd6f02.jpg)  
Figure 7: Absorbing the same exemplar upgrade. Every method is fitted first on generated exemplars and then on exemplars mined from held-in conversations, and evaluated on the identical held-out scenarios and artifacts; the slope is therefore what each method converts an exemplar improvement into. Colour marks the method family and line style the encoder. Trained discriminants (blue, green) absorb the upgrade; similarity over signals (orange) has no place to put a negative and saturates.

Table 7: The same exemplar upgrade, absorbed diferently. Each method is fitted first on generated exemplars and then on exemplars mined from one half of the scenarios, and evaluated on the held-out other half (§5), with artifacts and pooling fixed. The change, not the level, is the quantity of interest: it is what a method converts an exemplar improvement into.
<table><tr><td>Method (encoder)</td><td>P@R90 %</td><td>AUPRC</td></tr><tr><td>cosine, signals (Gemini)</td><td>32.0 → 35.3 (+3.3)</td><td>.459 → .501 (+.042)</td></tr><tr><td>cosine, signals (OpenAI)</td><td>32.8 → 36.5 (+3.7)</td><td>.467 → .480 (+.013)</td></tr><tr><td>S&amp;Q (Gemini)</td><td>33.9 → 43.5 (+9.7)</td><td>.475 → .602 (+.127)</td></tr><tr><td>S&amp;Q (OpenAI)</td><td>35.3 → 41.6 (+6.4)</td><td>.496 → .608 (+.112)</td></tr><tr><td>S&amp;Q, frozen head (MiniLM)</td><td>32.6 → 39.7 (+7.1)</td><td>.457 → .540 (+.083)</td></tr><tr><td>SetFit, fine-tuned (MiniLM)</td><td>31.6 → 37.3 (+5.7)</td><td>.450 → .559 (+.109)</td></tr></table>

The gain is almost entirely attributable to the negatives. Supplying mined positives while retaining generated anti-signals moves precision by +0.5 points on Gemini and −1.5 on OpenAI; adding the mined anti-signals on top supplies the remaining +9.2 and +7.8 (Fig. 8). This is the channel that separates S&Q from similarity retrieval. A score of the form max<sub>i</sub> $\cos ( q , s _ { j i } ) > \tau$ has nowhere to put a near-miss: it can absorb better positives, and doing so does help it, but only to a point (+3.3 and +3.7 in Table 7), after which there is nothing further for it to learn. Retrieval against policy text has no exemplar channel at all. Given identical data, S&Q converts it into roughly three times the precision gain that signal-cosine manages.

This channel is not exclusive to S&Q. SetFit, which likewise fits a discriminant over positives and negatives, absorbs the same upgrade to a comparable degree (+5.7 points, AUPRC +0.109), while remaining below a logistic head on frozen features at both exemplar sources (Table 7). The division is therefore between trained discriminants and similarity scores, not between S&Q and everything else. Within its own family S&Q is no better at absorbing an improvement; it is cheaper at it, which the next paragraph quantifies.

The loop is also cheap and local, which lets it compound. Exemplars of this kind are a by-product of operation: every conversation the agent handles is a candidate. Promoting one costs a single logistic fit of roughly two milliseconds for the afected artifact, and because each artifact’s classifier depends only on its own exemplars, nothing else in the inventory is retrained or recalibrated. Per-turn inference cost is unchanged, one embedding call and N dot products, whatever the exemplars contain. Accuracy therefore improves as a function of usage rather than of prompt budget or model size. The contrast with the alternatives is structural. Improving a per-turn LLM ranker means a longer prompt or a larger model, both of which raise the very per-turn cost that motivates this work. Fine-tuning is closer in kind, since it absorbs exemplar improvements nearly as well, but it is considerably more cumbersome to operate, for three reasons. First, per-turn inference stops being constant in the encoder: because each artifact carries its own fine-tuned model, a live turn must be embedded once per artifact, N transformer forward passes in place of one shared embedding and N dot products, which forfeits the very property the method exists for. Second, the improvement loop stops being interactive: refitting one artifact after an operator promotes an exemplar takes ∼7 ms for a logistic head against ∼5 s of gradient fine-tuning on a GPU, some three orders of magnitude (Fig. 9), so updates become a batched retraining job rather than an immediate edit. Third, the inventory turns from data into deployed models: N finetuned encoders of ∼90 MB each to version, ship, and roll back, against ∼12 KB per weight vector, produced by a stochastic procedure that need not return the same classifier from the same exemplars, where the logistic fit is convex and has a unique solution. In a governance setting, where an operator must be able to reproduce and explain why an artifact fired, that last diference is not incidental. Fine-tuning further requires a self-hosted, fine-tunable encoder, foreclosing the hosted embedding APIs that S&Q uses interchangeably.

![](images/5f5a740aee805874c3fa3f77d287acacaf3d19bf0de1766fc7520ff0e32859c8.jpg)  
Figure 8: Decomposing $\mathrm { S \& Q ^ { \prime } s }$ gain. Replacing generated positives alone is worth almost nothing; essentially the whole improvement arrives with the mined hard negatives, the channel that similarity retrieval cannot express.

Three limits qualify the result. Mining requires relevance labels on real conversations, whether from a judge or an operator, so the loop is cheap but not free. Coverage binds before quality does: at a half-and-half mining split only 22 of the 29 occurring artifacts had enough mined material to fit on, and the rarest had none, so generated exemplars remain both the bootstrap and the cold-start path. And exemplar genre must be consistent: mixing generated exemplars with mined ones cancels the benefit in both directions, which we read as the discriminant learning to separate synthetic text from real conversation instead of separating the artifact from its neighbours. The held-out set here is 131 sub-sequences, so we report a direction and a magnitude that replicate across two embedding providers rather than precise point estimates.

## 6 Discussion

How exemplars fail. Section 5 establishes that exemplars are the lever; the failure modes we observed say what makes one set better than another: (i) coverage: 10 anti-signals under-sample the space of irrelevant conversation, so unanticipated of-topic turns can score high (the main precision cost vs. many-negative training); (ii) hardness: anti-signals must be near-misses; replacing them with random negatives collapses the discriminant’s usefulness on neighbor confusions; (iii) domain match: generated signals are clean single utterances while queries are messy multi-turn conversations, producing the systematic score shift of Fig. 2 that forces θ < 0 and per-provider calibration, and which mining exemplars from real conversations largely removes (§5). Each of these is a property of the exemplar set, not of the retrieval machinery, which is why we treat signal and anti-signal construction as a first-class problem rather than a preprocessing detail. Beyond the mining loop of §5, natural directions include coverage-driven generation (generate until held-out exemplars are classified correctly) and adversarial anti-signal mining against the current classifier.

Precision ceilings are real. Some artifacts are intrinsically faint in embedding space: in our inventory, booking-related turns share vocabulary with the entire flight-administration genre, capping achievable precision at high recall regardless of method; a supervised probe trained directly on labeled conversations confirms the ceiling. A deployment should treat such artifacts with an operator-facing safety valve (e.g. pinning mustfire exemplar conversations, then promoting them into the signal set), which we prototype in the underlying framework but do not evaluate here.

Deploying S&Q over long conversations. Our analysis so far is per turn. A real agentic deployment, though, runs a single conversation across many turns, and the per-turn retrieved sets (§2) accumulate. Because the max-over-turns score of Eq. 2 is monotone in t, an artifact that once crosses θ keeps a score above it, so the retrieved set only grows. This is the same stickiness that aids multi-topic recall (§5), but its cost is that in a long, topic-shifting conversation the injected set drifts, unchecked, back toward the always-in-context regime this paper opens against. The stickiness is thus a property of our design, not only of the LLM judge. A practical deployment must therefore bound the accumulated set, expiring an artifact once enough others have been retrieved and letting it return only on genuinely new evidence rather than on the turns that first activated it. We neither model nor evaluate this here, and our benchmark’s short, single-intent conversations do not exercise it. Parlant 4.0 implements a mechanism of this kind.

![](images/35dc8566a8bdaf9ee1ed620b67bea8595ea9cbb8754655da70a590941b4840eb.jpg)

![](images/71899ddb5e6c7d2a449fd05d86d96470bbec37a03e787ae10af71a6180d3988f.jpg)  
Figure 9: What it costs to own the improvement loop, measured on one machine (Apple M1 Max, 10 cores, macOS) with both methods sharing the same local encoder (all-MiniLM-L6-v2, 384-dim) so the comparison is hardware-matched. SetFit’s fine-tuning and all encoder forward passes run on the Metal (MPS) GPU backend; S&Q’s logistic fit runs on CPU in scikit-learn. Left: fitting one artifact’s classifier. Right: scoring one live turn against N=36 artifacts, where S&Q embeds the turn once and takes N dot products while SetFit must re-embed it once per artifact, since each artifact owns its own fine-tuned encoder. Note the log scale. In the paper’s main configuration S&Q uses a hosted embedding API, so the left-hand bars, not the absolute right-hand latencies, are the deployment-relevant comparison.

Judge quality and provenance. Our ground truth is the first-pass relevance ranker of Parlant’s engine (gemini-3.1-flash-lite), the same component the deployed engine acts on, not a human panel. It therefore carries the usual LLM-judge caveats [16] and, importantly, has no measured inter-rater agreement with human evaluators. This is a real limitation: S&Q is trained to imitate this judge at low cost and latency, so its accuracy is bounded by the judge’s, and some of S&Q’s residual false positives are plausibly judge false negatives. We use the ranker because it is the relevance signal the engine already relies on in practice, and we treat a human-annotated subset as the natural next validation step. We also observed characteristic judge artifacts, a sticky tendency to keep a policy “relevant” once its topic appeared and a difuse “maybe” band, which evaluating at the judge’s own binary threshold (≥4) mitigates but does not remove.

Limitations. One domain (airline) and one conversation source $( \tau ^ { 2 } .$ -bench simulations); 5 of 36 policies never occur in the data and are unevaluated; the recallfirst operating point deliberately accepts low precision (∼30%), which is appropriate for policy discovery but not for every retrieval use; and the global margin $\theta ,$ while a single knob, was selected on this benchmark rather than derived from the inventory alone. S&Q also decides each artifact’s relevance independently from the conversation and does not model inter-artifact dependencies: a follow-up policy that applies only because another policy fired, and whose own text the current turn does not match, will not be retrieved on that basis. The decoupling that makes the method scale (§5) is what precludes this, so dependency handling belongs to an orchestration layer that consumes S&Q’s output rather than to S&Q itself.

## 6.1 Skill filtering

A recall-first pre-filter is most valuable where a false positive is bounded and a false negative is total. Skillbased agents present that asymmetry in unusually clean form. Under the agent-skills standard now common to major agent frameworks [6, 7], a skill is a directory whose entry point carries YAML frontmatter, a name and a description stating when the skill applies, above a Markdown body holding the instructions themselves. Agents load these in tiers: the frontmatter of every installed skill is resident from the start, on the order of tens of tokens each, while a skill’s body, which may run to thousands of tokens, is read only once the agent decides to invoke it.

That tiering makes the setting a natural fit. As a library grows to hundreds of skills, the resident frontmatter becomes a standing context cost in its own right, and the agent must choose among many superficially similar descriptions, which is precisely the regime in which instruction adherence degrades [1, 3]. S&Q applies to the frontmatter tier without modification: the artifact text is the skill’s name and description, its signals are user messages that should bring the skill into consideration, and its anti-signals are the near-misses that belong to a neighbouring skill. The filter decides which frontmatter entries enter the prompt at all.

Recall-first is the right operating point here because S&Q is not deciding to run anything. A false positive costs one frontmatter entry, tens of tokens, which the agent dismisses after reading a sentence; it does not pull in the skill’s body, where the real context cost lives. A false negative is unrecoverable: a skill whose description never reaches the prompt cannot be invoked, however relevant it was. Precision is thereby delegated to the stage that can aford it, the agent’s own reading of the surviving descriptions, while S&Q supplies the one property that stage cannot recover on its own, namely that the right candidate is present at all.

Existing practice retrieves skills by similarity against their descriptions: Voyager, for instance, keys a skill library by the embedding of each skill’s description and retrieves by nearest neighbour [18]. That is the positivesonly regime of our B2 and B4 baselines, and our results bear on it directly, since similarity against descriptions cannot represent the near-miss that separates a skill from its closest neighbour, and absorbs exemplar im provements at roughly a third the rate of a discriminant (§5). We have not evaluated S&Q on a skill corpus; we present this as an application whose shape the method fits, not as a measured result.

## 7 Related Work

Per-turn context selection. The problem S&Q addresses, deciding which of many candidate instructions or artifacts to place in the prompt at each turn, recurs across agent settings under diferent names. Two strategies dominate. Embedding similarity against artifact descriptions is cheap and LLM-free, and underlies tool and skill retrieval [8, 9], dynamic tool management across a conversation [19], in-context example selection [20], and long-horizon memory [21]. It is exactly the regime of our B2 and B4 baselines, and our results show where it fails: on the near-misses a similarity score cannot separate from true matches. LLM ranking, judging each candidate per turn, is more accurate but pays the per-turn token and latency cost that motivates this work (§4.4), and is the ranker we take as ground truth. A complementary line compresses a fixed context rather than selecting its contents, dropping low-information tokens to shorten the prompt [22]. Compression is orthogonal to S&Q and, for normative policies, risky, since a partially dropped rule can invert the behaviour it was meant to govern, whereas S&Q injects each selected artifact intact. S&Q sits at the cheap, LLM-free end of this space but replaces similarity with a recall-first, per-artifact discriminant, buying the near-miss discrimination that plain retrieval lacks.

Retrieval machinery. Dense retrieval represents queries and documents in a shared embedding space [23, 24], typically for retrieval-augmented generation [25]. S&Q difers in objective (per-artifact binary decisions under a recall constraint, not top-K ranking) and in bridging the query–document genre gap from the document side: signals are LLM-predicted activating queries, the exemplar-level analogue of Doc2Query [13] and of synthetic-query domain adaptation [14], while HyDE [26] attacks the same gap from the query side with hypothetical documents. Unlike these, S&Q additionally generates hard negative exemplars and trains a discriminant per artifact, bringing contrastive supervision [27] to inference-free per-artifact classification. The leverage of hard negatives is well established on the retrieval side: the passages most likely to distract the model reading them are the topically adjacent ones, and deliberately mining them improves robustness [28], a principle anti-signals carry to artifact selection by placing each classifier’s boundary along its most confusable neighbors. Embedding anisotropy and its correction motivate our centering step [5, 15, 29]. We evaluate on the conversational agent benchmarks τ-bench and τ<sup>2</sup>-bench [10, 11].

## 8 Conclusion

Signals and Queries reframes agentic context discovery as a set of tiny, decoupled, recall-first classifiers over LLM-generated exemplars: all the LLM work is spent ofline where it amortizes, and per-turn discovery becomes an embedding lookup. On a 36-policy benchmark judged by the production relevance ranker it dominates embedding-retrieval baselines at every recall level, across two embedding providers, at ∼10<sup>5</sup>× lower per-turn cost than per-policy LLM judging. The method ships as part of Parlant 4.0 [4]. Its clearest path forward is better exemplars: generation strategies that maximize coverage and hardness, and operator feedback loops that promote real missed conversations into the signal set.

## 9 Algorithm Summary

We give pseudocode for the two phases of S&Q. Ofline construction (Alg. 1) runs once per inventory and is rerun only for changed artifacts. Online discovery (Alg. 2) runs on each user turn and issues no LLM calls. Notation follows §2: E is the embedding model, G an LLM used only ofline, $\hat { v } = v / \| v \|$ , and c the inventory centroid.

Algorithm 1 Ofline construction (per inventory)   
Require: artifacts $\mathcal { A } = \{ a _ { 1 } , . . . , a _ { N } \}$ ; embedder $E ;$ genera  
tor G   
1: for each artifact $a _ { j } \in { \mathcal { A } }$ do   
2: $S _ { j } \gets G . \mathrm { G E N S I G N A L S } ( a _ { j } )$ ▷ m activating msgs   
3: S<sup>¯</sup><sub>j</sub> ← G.GenAntiSignals(a<sub>j</sub>) ▷ m near-misses   
4: end for   
5: $\begin{array} { r } { c \gets \frac { 1 } { N } \sum _ { j } E \big ( \mathrm { T E X T } ( a _ { j } ) \big ) } \end{array}$ ▷ centroid   
6: for each artifact $a _ { j } \in { \mathcal { A } }$ do   
7: $P  \{ \widehat { E ( s ) - c : s \in S _ { j } } \}$ ▷ positives   
8: $Q  \{ E ( s ) - c : s \in \bar { S } _ { j } \}$ ▷ negatives   
9: $( w _ { j } , b _ { j } ) \gets \mathrm { F r T L O G R E G } ( P ; 1 , ~ Q { : } 0 )$   
10: end for   
11: return classifiers $\{ ( w _ { j } , b _ { j } ) \} _ { j = 1 } ^ { N } ,$ centroid c

Algorithm 2 Online discovery (per user turn t)   
Require: turns $u _ { 1 } , . . . , u _ { t } ;$ classifiers $\{ ( w _ { j } , b _ { j } ) \}$ ; c; margin θ   
1: $q _ { \mathrm { c u m } } \gets E \bigl ( \mathrm { R E N D E R } ( \widehat { u } _ { 1 : t } ) \bigr ) - c$ ▷ cumulative   
2: $q _ { i }  E ( u _ { i } ) \overline { { - } } c$ for $i = 1 , \ldots , t$ ▷ per-turn   
3: ${ \mathcal { R } } \gets \emptyset$   
4: for each artifact $a _ { j }$ do   
5: z<sub>j</sub> ← max w<sub>j</sub> ·q<sub>cum</sub>, $\operatorname* { m a x } _ { i \leq t } w _ { j } \cdot q _ { i } \big ) + b _ { j }$   
▷ max over turns   
6: if $z _ { j } > \theta$ then   
7: ${ \ddot { \mathcal { R } } }  { \mathcal { R } } \cup \{ a _ { j } \}$   
8: end if   
9: end for   
10: return R ▷ artifacts to inject into context

Only $q _ { \mathrm { c u m } }$ and the newest q<sub>t</sub> are embedded per turn (earlier q<sub>i</sub> are cached from prior turns), so a turn costs one embedding call plus N dot products. The recall-first regime uses $\theta < 0 ;$ per-provider values are given in §4. The operator safety valve of §6 augments line 6 of Alg. 2 with a disjunctive test against pinned exemplar conversations, and promotes recurring pins into $S _ { j }$ (Alg. 1, line 2).

## Acknowledgements

We thank Alex Shtof <alexander.shtoff@tii.ae> for a careful review of an earlier draft. His feedback motivated the ablation study of §4.2: in particular, the comparison of the logistic classifier against a linear SVM and a nearest-neighbor variant, and the accompanying analysis of why the classifier choice is immaterial in this small, separable, hard-negative regime. His comments also sharpened the paper’s empirical framing around baselines and ablations. We thank Vishal Ahuja <vishal.x.ahuja@chase.com> for suggesting the comparison against SetFit [17] in §4.3. We thank Sarthak Dalabehera <s.dalabehera@rippling.com> for the skill-filtering application of §6.1. We thank Bar Karov <barkrv@gmail.com> for a careful review that reconciled the frontier plot with the operating-point table (§4) and prompted the treatment of related work on per-turn context selection. We thank Shailen Kumar Dalbehera <ShailenKumar.Dalbehera@evernorth.com> for raising the provenance and human validation of the LLM judge and the accumulation of retrieved policies over long conversations, which motivated the discussion of pruning (§6).

## References

[1] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12, 2024. arXiv:2307.03172.

[2] Yuan Tian and Tianyi Zhang. Selective prompt anchoring for code generation. arXiv preprint arXiv:2408.09121, 2024.

[3] Keno Harada, Yudai Yamazaki, Masachika Taniguchi, Takeshi Kojima, Yusuke Iwasawa, and Yutaka Matsuo. Curse of instructions: Large language models cannot follow multiple instructions at once. In International Conference on Learning Representations, 2025.

[4] Parlant. Parlant: A behavioral governance framework for customer-facing LLM agents. https: //github.com/emcie-co/parlant, 2026.

[5] Kawin Ethayarajh. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of EMNLP-IJCNLP, 2019. arXiv:1909.00512.

[6] Anthropic. Agent Skills. https://platform. claude.com/docs/en/agents-and-tools/ agent-skills/overview, 2025.

[7] OpenAI. Codex: Build skills. https:// developers.openai.com/codex/skills, 2026.

[8] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive APIs. arXiv preprint arXiv:2305.15334, 2023.

[9] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. arXiv preprint arXiv:2307.16789, 2023.

[10] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

[11] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.

[12] Bar Karov, Dor Zohar, and Yam Marcovitz. Attentive reasoning queries: A systematic method for optimizing instruction-following in large language models. arXiv preprint arXiv:2503.03669, 2025.

[13] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. Document expansion by query prediction. arXiv preprint arXiv:1904.08375, 2019.

[14] Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. GPL: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval. In Proceedings of NAACL, 2022. arXiv:2112.07577.

[15] Jiaqi Mu and Pramod Viswanath. All-but-thetop: Simple and efective postprocessing for word representations. In International Conference on Learning Representations, 2018. arXiv:1702.01417.

[16] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, 2023. arXiv:2306.05685.

[17] Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, and Oren Pereg. Eficient few-shot learning without prompts. arXiv preprint arXiv:2209.11055, 2022.

[18] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

[19] Elias Lumer, Anmol Gulati, Vamse Kumar Subbiah, Pradeep Honaganahalli Basavaraju, and James A. Burke. MemTool: Optimizing short-term memory management for dynamic tool calling in LLM agent multi-turn conversations. arXiv preprint arXiv:2507.21428, 2025.

[20] Ruoyu Wang, Junda Wu, Yu Xia, Tong Yu, Ryan A. Rossi, Julian McAuley, and Lina Yao. DICE: Dynamic in-context example selection in LLM agents via eficient knowledge transfer. arXiv preprint arXiv:2507.23554, 2025.

[21] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[22] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of EMNLP, 2023. arXiv:2310.05736.

[23] Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of EMNLP, 2020. arXiv:2004.04906.

[24] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of EMNLP-IJCNLP, 2019. arXiv:1908.10084.

[25] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, 2020. arXiv:2005.11401.

[26] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance labels. In Proceedings of ACL, 2023. arXiv:2212.10496.

[27] Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of EMNLP, 2021. arXiv:2104.08821.

[28] Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. The distracting efect: Understanding irrelevant passages in RAG. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025. ACL 2025.

[29] Alexis Conneau, Guillaume Lample, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herv´e J´egou. Word translation without parallel data. In International Conference on Learning Representations, 2018. arXiv:1710.04087.