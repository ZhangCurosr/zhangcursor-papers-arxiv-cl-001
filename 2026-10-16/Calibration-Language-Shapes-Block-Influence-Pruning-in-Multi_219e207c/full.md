# Calibration Language Shapes Block-Influence Pruning in Multilingual Language Models

Vedant Pandhare Independent Researcher

Piyush Dhoka Independent Researcher

vedant11734@gmail.com

piyushdhoka007@gmail.com

## Abstract

Block Influence (BI) ranks Transformer blocks for depth pruning from their input–output cosine change, but the ranking depends on the calibration distribution. We study whether English calibration transfers to Indian languages in Qwen2.5-1.5B-Instruct, Llama-3.2-1B, and Gemma-2-2B. English, Hindi, Marathi, and Tamil use disjoint Wikipedia articles for calibration and evaluation; every selector receives the same 150-chunk budget, and Tamil is excluded from multilingual aggregation to test cross-script transfer. Across all three models, late-block BI is 20.7–55.5% higher for the three Indian languages than for English, and every predeclared late-minus-middle contrast has a 95% bootstrap interval above zero. This mismatch changes the low-BI blocks selected for removal despite high global rank correlations. Averaging BI over equal English, Hindi, and Marathi budgets mitigates the resulting perplexity damage for Qwen at six removed blocks and for Gemma at all tested depths, including held-out Tamil. The mitigation is not universal: on Llama, balancing improves Hindi and Marathi at two removed blocks but severely harms Tamil, while larger pruning budgets select the same blocks and collapse perplexity regardless of calibration. Rankmean and minimax-rank aggregators do not dominate the raw mean. These results support a diagnostic claim rather than a universal pruning rule: English-only BI systematically misses late-layer activity for Indian-language text, but multilingual calibration must still be validated per architecture and pruning budget.

## 1 Introduction

Removing complete Transformer blocks is an attractive compression operation: it shortens the sequential computation graph, reduces parameters, and does not require sparse kernels. Block-pruning methods therefore need to answer a consequential discrete question: which blocks can be removed? Block Influence (BI), introduced by ShortGPT, answers it by measuring how little each block changes its hidden states on a small calibration corpus and pruning the blocks with the lowest scores (Men et al., 2024). The score is inexpensive and training-free, but it is conditional on the corpus used to estimate it.

That condition matters for multilingual deployment. Calibration data in language-model compression is commonly English even when the model will serve other languages (Kurz et al., 2026; Zeng et al., 2024). Prior work shows calibration-language efects for unstructured weight pruning and quantization (Kurz et al., 2026; Choenni & Titov, 2025), but those results do not establish whether the ordering of whole Transformer blocks transfers across languages. A small score perturbation is suficient to change a depth-pruning decision, and removing one complete block has a much larger structural efect than changing an equal number of isolated weights.

We examine this problem for English (en), Hindi (hi), Marathi (mr), and Tamil (ta). Hindi and Marathi provide related Devanagari-script languages, while Tamil is from a diferent language family and uses a diferent script. We include Tamil in analysis and evaluation but exclude it from language-balanced selector construction. This design asks whether a selector balanced over en/hi/mr transfers beyond the scripts and languages it observes, rather than merely fitting the calibration mixture.

Our central claim is deliberately limited:

English-only calibration systematically underestimates late-block influence for the Indianlanguage text studied here. Language-balanced calibration can avoid harmful English-selected blocks, but the downstream benefit is architecture- and budget-dependent.

The first sentence is a statement about the behavior of BI. The second is an empirical qualification, not a claim that balancing guarantees safe pruning.

We test this claim on three architecture families: Qwen2.5-1.5B-Instruct, Llama-3.2-1B, and Gemma-2-2B. The design uses article-disjoint calibration and evaluation data, equal token budgets, five independently sampled calibration subsets, three aggregation rules, ten random-pruning controls, and uncertainty over both calibration subsets and held-out evaluation articles. No recovery training is used; this isolates block selection from post-pruning adaptation.

The study makes four contributions:

1. We provide a controlled analysis of calibration-language transfer for BI-based whole-block pruning, complementing prior multilingual studies of unstructured pruning.

2. We localize the cross-language discrepancy: across all three models and all three Indian languages, the target-to-English BI ratio is substantially larger in a predeclared late-depth region than in the middle region.

3. We compare equal-budget raw-mean, rank-mean, and minimax-rank multilingual selectors, including transfer to held-out Tamil. Raw-mean balancing prevents damaging late-block choices in Qwen and Gemma, but no aggregator dominates on every model.

4. We report a negative boundary case. BI pruning is brittle on the 16-block Llama model, and the balanced selector’s Hindi/Marathi gains at one budget do not transfer to Tamil. This separates detecting calibration mismatch from correcting it.

The results suggest a practical rule of evidence: multilingual depth pruning should report language-conditioned block scores and held-out-language consequences, not infer robustness from an English calibration set or from a high global rank correlation alone.

## 2 Related Work

Whole-block pruning. ShortGPT defines BI from block input–output cosine similarity and removes low-influence Transformer blocks without training (Men et al., 2024). Other depth-reduction methods use output reconstruction, redundancy verification, block merging, or a short recovery stage (Song et al., 2024; Yang et al., 2024; Gromov et al., 2025). These methods show that complete-block removal can reduce sequential computation, but evaluations and calibration corpora are predominantly English. Our work does not introduce another recovery procedure. It examines a prior question: whether BI identifies the same low-influence blocks when the calibration language changes.

Calibration data for compression. SparseGPT and Wanda made small calibration sets standard in post-training weight pruning (Frantar & Alistarh, 2023; Sun et al., 2024). Calibration content can materially alter pruning outcomes even within English (Bandari et al., 2024). Multilingual Brain Surgeon samples several languages to mitigate English-centric compression (Zeng et al., 2024); bilingual pruning (Kim et al., 2024) and M-Wanda (Choenni & Titov, 2025) likewise incorporate cross-lingual information. Most directly, Kurz et al. (2026) compare target-language calibration for Wanda and SparseGPT over seven languages. They find consistent perplexity benefits but weaker downstream-task transfer. Those studies concern masks over individual weights or neurons. We instead study the discrete ranking and removal of whole decoder blocks under BI.

Multilingual structured pruning. Wibowo et al. (2025) study structured layer pruning together with iterative recovery and distillation, showing that the recovery language has nontrivial cross-lingual efects. Our setting is complementary: one-shot block removal with fixed weights allows the language dependence of selection to be observed without adaptation. Work on multilingual model compression more broadly reports disproportionate damage to some languages and scripts (Ogueji et al., 2022; Petrov et al., 2023).

Depth and multilingual representation. Analyses of multilingual Transformers find both shared representations and language-specific components, with language identity and surface realization often becoming prominent toward the boundaries of the network (Tang et al., 2024; Wendler et al., 2024; Zhao et al., 2024). This motivates, but does not imply, our late-layer hypothesis. We test it directly using a predeclared depth partition and independent calibration articles.

Positioning. We therefore do not claim that calibration-language sensitivity is new in all forms of pruning. The novelty is the combination of (i) BI whole-block ranking, (ii) localization across normalized depth, (iii) equal-budget multilingual aggregation ablations, and (iv) an Indian-language evaluation where Tamil is held out from aggregation. The Llama counterexample also provides evidence about when a calibration-aware selector is insuficient.

## 3 Method

## 3.1 Language-conditioned Block Influence

Consider a decoder with blocks $l \in \{ 0 , \ldots , L - 1 \}$ . Let $\mathbf { h } _ { l , t } ^ { \mathrm { i n } }$ and $\mathbf { h } _ { l , t } ^ { \mathrm { o u t } }$ be the input and output hidden states of block l at token position t. For a calibration collection $D _ { \lambda }$ in language λ, we estimate

$$
\mathrm { B I } _ { l } ^ { ( \lambda ) } = 1 - \frac { 1 } { | D _ { \lambda } | } \sum _ { x \in D _ { \lambda } } \frac { 1 } { | \mathcal { T } _ { x } | } \sum _ { t \in \mathcal { T } _ { x } } \frac { ( { \bf h } _ { l , t } ^ { \mathrm { i n } } ) ^ { \top } { \bf h } _ { l , t } ^ { \mathrm { o u t } } } { \| { \bf h } _ { l , t } ^ { \mathrm { i n } } \| _ { 2 } \| { \bf h } _ { l , t } ^ { \mathrm { o u t } } \| _ { 2 } } .\tag{1}
$$

The beginning-of-sequence position, when present, is excluded from $\mathcal { T } _ { x }$ because it is an activation outlier and attention sink (Xiao et al., 2024). We capture both tensors with hooks on the block itself, before any final model normalization. Low BI means that a block changes the direction of its input representation little on that calibration distribution; it is a pruning candidate, not proof of mathematical redundancy.

For a score vector s, $S _ { k } ( \mathbf { s } )$ is the set of the k lowest-scoring blocks, with block index as a deterministic final tie-breaker. We evaluate monolingual score vectors for en, hi, mr, and ta. The first three are also used to construct language-balanced selectors; Tamil is never used in those aggregations.

## 3.2 Equal-budget language-balanced selectors

Let $\Lambda _ { \mathrm { a g g } } = \{ \mathrm { E N , H I , M R } \}$ . We compare three aggregators:

$$
s _ { l } ^ { \mathrm { m e a n } } = \frac { 1 } { | \Lambda _ { \mathrm { a g g } } | } \sum _ { \lambda \in \Lambda _ { \mathrm { a g g } } } { \mathrm { B I } } _ { l } ^ { ( \lambda ) } ,\tag{2}
$$

$$
s _ { l } ^ { \mathrm { r a n k } } = \frac { 1 } { | \Lambda _ { \mathrm { a g g } } | } \sum _ { \lambda \in \Lambda _ { \mathrm { a g g } } } r _ { l } ^ { ( \lambda ) } ,\tag{3}
$$

$$
s _ { l } ^ { \mathrm { m i n i m a x } } = \operatorname* { m a x } _ { \lambda \in \Lambda _ { \mathrm { a g g } } } r _ { l } ^ { ( \lambda ) } ,\tag{4}
$$

where $r _ { l } ^ { ( \lambda ) } \in [ 0 , 1 ]$ is the ascending percentile rank of block l under language λ. Minimizing $\operatorname { E q }$ . (4) prefers a block only when its rank is low for every aggregation language. Ties are broken by mean rank and then block index. Raw averaging retains score magnitudes; the rank rules remove cross-language scale diferences.

Every primary selector sees 150 chunks. A monolingual selector receives 150 chunks from one language, while each balanced selector receives 50 chunks from each of en, hi, and mr. Thus any advantage cannot be attributed to giving balanced calibration three times as many tokens.

## 3.3 Late-layer localization

We predeclare a depth partition before comparing outcomes. The final output-adjacent block is excluded because its role and BI scale are atypical. The remaining indices $0 , \ldots , L - 2$ are split as evenly as possible into early, middle, and late thirds. For target language $\lambda \in \left\{ \mathrm { H I , M R , T A } \right\}$ , define the layer-wise log ratio

$$
q _ { l } ^ { ( \lambda ) } = \log \mathrm { B I } _ { l } ^ { ( \lambda ) } - \log \mathrm { B I } _ { l } ^ { \mathrm { ( E N ) } } .\tag{5}
$$

Our localization statistic is

$$
C ^ { ( \lambda ) } = \frac { 1 } { \left| \mathcal { L } _ { \mathrm { l a t e } } \right| } \sum _ { l \in \mathcal { L } _ { \mathrm { l a t e } } } q _ { l } ^ { ( \lambda ) } - \frac { 1 } { \left| \mathcal { L } _ { \mathrm { m i d } } \right| } \sum _ { l \in \mathcal { L } _ { \mathrm { m i d } } } q _ { l } ^ { ( \lambda ) } .\tag{6}
$$

Positive $C ^ { ( \lambda ) }$ means that target-to-English influence is more elevated late in the network than in its middle. We also report the geometric percentage shift $1 0 0 [ \exp ( \overline { { q } } _ { \mathrm { l a t e } } ) - 1 ]$

## 3.4 Pruning outcome

For each selected set S, we physically bypass those blocks without changing any remaining weight. On evaluation language $\ell ,$ mean token negative log-likelihood (NLL) gives

$$
\mathrm { P P L } _ { \ell } ( S ) = \exp \left( \frac { 1 } { N _ { \ell } } \sum _ { i = 1 } ^ { N _ { \ell } } - \log p _ { \theta \setminus S } ( x _ { i } \mid x _ { < i } ) \right) .\tag{7}
$$

We report relative degradation against the intact model and compare English-only with balanced selection using

$$
R _ { \ell , k } = \frac { \mathrm { P P L } _ { \ell } ( S _ { k } ^ { \mathrm { E N } } ) } { \mathrm { P P L } _ { \ell } ( S _ { k } ^ { \mathrm { m e a n } } ) } .\tag{8}
$$

$R _ { \ell , k } > 1$ favors language-balanced raw-mean calibration. Perplexity is compared only within the same model and evaluation language; tokenizer-dependent values are not compared across models or languages.

As a metric robustness check, we also compute relative hidden-state change

$$
\mathrm { R L 2 } _ { l } ^ { ( \lambda ) } = \mathbb { E } _ { x , t } \left[ \frac { \| \mathbf { h } _ { l , t } ^ { \mathrm { o u t } } - \mathbf { h } _ { l , t } ^ { \mathrm { i n } } \| _ { 2 } } { \| \mathbf { h } _ { l , t } ^ { \mathrm { i n } } \| _ { 2 } } \right] ,\tag{9}
$$

but BI determines every primary pruning set.

## 4 Experimental Setup

## 4.1 Models and checkpoint provenance

Table 1 covers three decoder-only architecture families in the 1–3B parameter range (Qwen Team, 2025; Dubey et al., 2024; Gemma Team, 2024). We use the organization-hosted Hugging Face checkpoints Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Llama-3.2-1B, and google/gemma-2-2b. Every reported result is tied to an immutable repository revision listed in Appendix A.

This diference in pruning fraction is important: equal k does not imply equal severity for Llama. We retain $k \in \{ 2 , 4 , 6 \}$ because the scientific object is the discrete selector at fixed k, but we avoid interpreting cross-model degradation magnitudes as a model-quality ranking.

Table 1: Models and pruning fractions. Fractions count decoder blocks, not parameters.
<table><tr><td>Model</td><td>Parameters</td><td>Blocks</td><td> $k = 2$ </td><td> $k = 4$ </td><td> $k = 6$ </td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>1.54B</td><td>28</td><td>7.1%</td><td>14.3%</td><td>21.4%</td></tr><tr><td>Llama-3.2-1B</td><td>1.24B</td><td>16</td><td>12.5%</td><td>25.0%</td><td>37.5%</td></tr><tr><td>Gemma-2-2B</td><td>2.61B</td><td>26</td><td>7.7%</td><td>15.4%</td><td>23.1%</td></tr></table>

## 4.2 Languages and data

We use the 2023-11-01 Wikipedia snapshots for English, Hindi, Marathi, and Tamil (Wikimedia Foundation, 2023). Articles are drawn from a deterministically shufled stream (seed 42), filtered to contain at least 300 whitespace-delimited words, and tokenized separately for each model. At most one 256-token chunk is retained from any article. For each language, 200 articles form the evaluation set and a disjoint 200 articles form the calibration pool. The article identifiers are identical across all three model runs, and assertions check that no evaluation article appears in calibration.

Models defining a beginning-of-sequence token receive it, leaving 255 text tokens; Qwen receives 256 text tokens. All evaluated chunks contain 255 next-token predictions. The main evaluation therefore contains 200 independent article-level NLL observations per language and model. Baseline bits per character and tokenizer fertility are reported in Appendix B.

## 4.3 Selection and controls

For each of five calibration seeds, a monolingual condition samples 150 of its 200 pool chunks without replacement. A balanced condition samples 50 chunks independently from each of en, hi, and mr. For every condition, we remove k ∈ {2, 4, 6} blocks and evaluate the resulting model on all four languages. The primary design has seven conditions (four monolingual and three balanced), five seeds, three pruning depths, and four evaluation languages, yielding 420 observations per model.

We additionally evaluate ten deterministic random block sets at every k. Full-pool rankings using all 200 calibration chunks are retained as descriptive diagnostics, not mixed into the equal-budget primary contrasts. No fine-tuning, distillation, or post-pruning recovery is performed.

## 4.4 Uncertainty and implementation

For Eq. (6), we independently bootstrap the 200 English and 200 target-language calibration articles 5,000 times. For pruning contrasts, we form paired per-article NLL diferences and use a hierarchical percentile bootstrap that resamples both the five calibration seeds and the 200 evaluation articles. We present these 95% intervals as the primary uncertainty summary. Exact layer-set equality is reported separately because a zero contrast can arise mechanically when two selectors choose the same blocks.

All experiments ran in bfloat16 on an NVIDIA A100-SXM4-40GB with Transformers 4.56.2 and PyTorch 2.11.0+cu130. Each pruning evaluation starts from the same in-memory block list, bypasses the selected blocks, and restores the complete model afterward. A final baseline re-evaluation checks restoration. The public code repository (Appendix A) includes configurations, article identifiers, block scores, selected sets, per-seed outcomes, contrasts, and random controls.

## 5 Results

## 5.1 Global ranking agreement hides pruning-set disagreement

The broad depth profile is shared across languages (Figure 1). English-to-target Spearman correlations over all BI scores range from 0.898 to 0.982. This apparently strong agreement is not suficient for pruning: selection depends only on the lowest-scoring tail. Full-pool Jaccard overlap between English and a target-language set falls to 0.33 for Llama and Gemma at $k = 2 , 0 . 6 0$ for Qwen and Gemma at $k = 4 ,$ and 0.71 for several $k = 6$ comparisons.

![](images/5ed8f1b178f6be1d1d10320f58582be98d0aa788955695779da34a86f9df5d17.jpg)

![](images/08f34c5ae63f229fbd852654fc7ed5893163d0891ae79d09d6db08bf99811410.jpg)

![](images/8600831345d07d241e0b4fa7c9252cee235ed90fe17db365daf13870e1de511f.jpg)  
Figure 1: Language-conditioned Block Influence. The gray span is the predeclared late region; the final output-adjacent block is excluded from regional contrasts. Curves share a coarse shape but separate near the low-BI late blocks that can enter the pruning set.

Table 2: Target-language BI relative to English in the predeclared late region. The late–middle column is Eq. (6); intervals bootstrap independent calibration articles.
<table><tr><td>Model</td><td>Language</td><td>Late shift (%)</td><td>Late-middle</td><td>95% CI</td></tr><tr><td rowspan="3">Qwen2.5-1.5B-Instruct</td><td>HI</td><td>51.0</td><td>0.391</td><td>[0.378, 0.405]</td></tr><tr><td>MR</td><td>55.5</td><td>0.392</td><td>[0.378, 0.406]</td></tr><tr><td>TA</td><td>39.6</td><td>0.363</td><td>[0.350, 0.376]</td></tr><tr><td rowspan="3">Llama-3.2-1B</td><td>HI</td><td>20.7</td><td>0.221</td><td>[0.210, 0.233]</td></tr><tr><td>MR</td><td>31.8</td><td>0.308</td><td>[0.297, 0.319]</td></tr><tr><td>TA</td><td>25.3</td><td>0.428</td><td>[0.417, 0.439]</td></tr><tr><td rowspan="3">Gemma-2-2B</td><td>HI</td><td>21.4</td><td>0.157</td><td>[0.148, 0.166]</td></tr><tr><td>MR</td><td>36.1</td><td>0.260</td><td>[0.251, 0.268]</td></tr><tr><td>TA</td><td>24.5</td><td>0.166</td><td>[0.157, 0.174]</td></tr></table>

The disagreements are concrete. At k = 6, English calibration for Qwen selects late block 26 while each Indian-language full-pool ranking selects block 13 instead. For Gemma, English selects block 23 at $k = 2$ while Hindi and Marathi select block 10. On Llama at k = 2, English chooses {12, 13}, Hindi chooses {12, 14}, and Marathi and Tamil choose {11, 12}. Thus global rank correlation obscures changes in the exact discrete action.

## 5.2 English underestimates late-block influence

Table 2 reports the predeclared depth contrast from Eq. (6). In the late region, target-language BI is geometrically 20.7–55.5% higher than English BI across all nine model–language pairs. More importantly, the late-minus-middle contrast is positive in every pair, with every 95% interval above zero (Figure 2). The smallest contrast is Gemma–Hindi, 0.157 [0.148, 0.166], and the largest is Llama–Tamil, 0.428 [0.417, 0.439].

Tamil follows the same pattern despite being non-Devanagari and from a diferent language family. The result therefore cannot be explained solely as agreement between Hindi and Marathi, although this experiment does not isolate script from every other linguistic or corpus property. This consistent localization is the strongest evidence for our main claim.

![](images/8203e26b542d6e3f30f6fa70a8af5be0e98909785f8351dce0be8b29d4ed8304.jpg)  
Figure 2: Late-minus-middle log BI ratio to English, with 95% bootstrap intervals. All nine contrasts are above zero.

Table 3: Perplexity ratio $R _ { \ell , k } = \mathrm { P P L } ( S _ { k } ^ { \mathrm { g N } } ) / \mathrm { P P L } ( S _ { k } ^ { \mathrm { m e a n } } )$ with hierarchical 95% intervals. The marker = denotes identical layer sets for all five seeds. Tamil is held out from balanced selector construction.
<table><tr><td>Model</td><td>k</td><td></td><td>HI</td><td></td><td>MR</td><td></td><td></td><td>TA</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>2</td><td></td><td></td><td>1.00 (=)</td><td></td><td>1.00 (=)</td><td></td><td>1.00 (=)</td></tr><tr><td></td><td>4</td><td></td><td>1.004 [1.000, 1.012]</td><td></td><td>1.003 [1.000, 1.009]</td><td></td><td></td><td>1.003 [1.000, 1.008]</td></tr><tr><td></td><td>6</td><td>1.449</td><td>[1.434, 1.465]</td><td></td><td>1.504 [1.484, 1.522]</td><td></td><td></td><td>1.372 [1.361, 1.384]</td></tr><tr><td>Llama-3.2-1B</td><td>2</td><td></td><td>1.640 [1.553, 1.730]</td><td></td><td>3.098 [2.768, 3.487]</td><td></td><td></td><td>0.057 [0.053, 0.061]</td></tr><tr><td></td><td>4</td><td></td><td></td><td>1.00 (=)</td><td></td><td>1.00 (=)</td><td></td><td>1.00 (=)</td></tr><tr><td></td><td>6</td><td></td><td></td><td>1.00 (=)</td><td></td><td>1.00 (=)</td><td></td><td>1.00 (=)</td></tr><tr><td>Gemma-2-2B</td><td>2</td><td></td><td>1.190 [1.178, 1.201]</td><td></td><td>1.419 [1.402, 1.437]</td><td></td><td></td><td>1.262 [1.248, 1.276]</td></tr><tr><td></td><td>4</td><td>1.229</td><td>[1.219, 1.239]</td><td></td><td>1.442 [1.428, 1.457]</td><td></td><td></td><td>1.316 [1.302, 1.332]</td></tr><tr><td></td><td>6</td><td></td><td>1.356 [1.336, 1.375]</td><td></td><td>1.608 [1.581, 1.635]</td><td></td><td></td><td>1.417 [1.394, 1.441]</td></tr></table>

## 5.3 Balanced raw-mean calibration often helps, but not universally

Table 3 and Figure 3 compare equal-budget English-only selection with raw-mean $\mathrm { { E N / H I / M R } }$ selection. Ratios above one favor balancing.

Qwen. At k = 2, both selectors choose the same blocks. At k = 4, they difer for only one of five calibration seeds and the efect is small, so we treat this setting as inconclusive. $\mathrm { A t } ~ k = 6$ , balancing lowers PPL in all three target languages: English-only PPL is 1.45×, 1.50×, and 1.37× balanced PPL on Hindi, Marathi, and Tamil. Mean target-language degradation falls from 88.9% to 30.8%, and worst-language degradation from 107.5% to 38.0%.

Gemma. Raw-mean balancing changes the selected set at all three depths and improves every target language. English-to-balanced PPL ratios range from 1.19 to 1.61. Mean target degradation falls from 50.8% to 16.8% at k = 2, from 78.5% to 34.2% at $k = 4$ , and from 242.3% to 132.5% at $k = 6$ . Held-out Tamil improves at every depth, showing that the efect is not limited to languages included in the average.

Llama is a counterexample to universal robustness. At k = 2, balancing changes {12, 13} to {12, 14} and improves Hindi and Marathi: English-only PPL is 1.64× and 3.10× balanced PPL. The same change is disastrous for held-out Tamil, where balanced PPL is approximately 17.5× English-only PPL. Consequently, balanced target-mean degradation is 2694%, versus 1462% for English, and its worst-language degradation is 6572%, versus 3081%. At k = 4 and k = 6, the selectors choose identical blocks and both cause extreme degradation. Language balancing therefore corrects two observed languages without producing a robust cross-language choice on this model.

![](images/2ec13929e3140c7d68f4e5ad2c0dffbd94ed7522ecb88e8e2f7e005d084f9312.jpg)  
Figure 3: English-only versus raw-mean balanced calibration. Values above one favor balanced selection; values below one favor English-only selection. The log scale exposes the Llama–Tamil reversal.

## 5.4 Aggregator ablation

No rank-based rule dominates. For Qwen, rank mean has the lowest target mean at $k = 4$ (18.9% versus 20.4% for raw mean), while raw mean and rank mean tie at $k = 6 \ ( 3 0 . 8 \% )$ . For Gemma, all three aggregators tie at $k = 2$ , but only raw mean avoids the English-selected late blocks at k = 4 and $k = 6 ;$ rank mean and minimax rank reproduce the worse English set. All three aggregators select the same blocks on Llama. The complete mean/worst table is in Appendix B.

Raw mean is consequently the most consistent of the tested aggregators, not a universally optimal method. Minimax rank’s appealing requirement that a block rank low in every language does not translate into better observed outcomes here.

## 5.5 Stability and random controls

At the 150-chunk budget, all Llama and Gemma selectors have pairwise Jaccard 1.0 across five seeds. Qwen’s minimum is 0.84, and its full-pool agreement minimum is 0.68. English-only selections are themselves perfectly stable in these runs. We therefore use robust only for cross-language outcome, not as a synonym for calibration-seed stability.

For Qwen and Gemma, BI-selected sets have lower average target degradation than the ten random sets at every k. Llama again exposes a method-level limitation: both English BI and balanced BI are worse than the random-set average at $k = 2$ and $k = 4 ,$ , although BI is better at $k = 6 .$ . This result, together with Llama’s high pruning fractions in Table 1, shows that a corrected calibration distribution cannot rescue an unreliable importance proxy or an excessively aggressive depth reduction.

## 6 Discussion

What the evidence establishes. The result that survives every model and target language is not that one aggregation rule always wins. It is that English changes the BI profile selectively with depth: the target-to-English ratio is consistently more positive late than in the middle. This is visible for Hindi, Marathi, and Tamil, with narrow article-bootstrap intervals. English calibration can therefore make a block look relatively inactive precisely where another language assigns it more representational change.

What remains an interpretation. A plausible account is that middle blocks perform more language shared computation while later blocks participate in language-specific surface realization, consistent with prior representation analyses (Tang et al., 2024; Wendler et al., 2024; Zhao et al., 2024). Our experiment does not intervene on linguistic features and therefore does not prove this mechanism. BI is also not causal importance: it measures local angular change, while downstream damage depends on composition with every subsequent block.

Detection and correction are distinct. Qwen and Gemma connect the late-layer mismatch to consequential English-selected blocks, and raw-mean balancing removes those choices. Llama retains the same directional mismatch yet defeats the proposed correction on held-out Tamil. This distinction is scientifically useful. A calibration diagnostic can be reproducible even when a selection heuristic is too coarse, the model is too shallow, or the requested pruning fraction is too aggressive.

Practical implications. For multilingual BI pruning, we recommend reporting per-language score profiles and the exact selected sets; testing at least one language excluded from selector construction; keeping calibration token budgets equal; comparing against random block sets; and treating the number of removed blocks as a fraction of model depth. Raw-mean balancing is a reasonable candidate selector in our experiments, especially because it retains magnitude information that the rank aggregators discard, but it should be validated on the intended deployment languages and model.

Why no recovery stage is included. Recovery training can make a pruned model more useful, but it also introduces a second language distribution, an optimizer, and additional data. The present question is about the initial BI decision. Holding all remaining weights fixed isolates that decision and exposes failures that adaptation could conceal. Our conclusions therefore apply to one-shot selection and immediate language-modeling behavior; they neither support nor rule out language-dependent efects after recovery.

## 7 Limitations and Threats to Validity

Model and pruning scale. We study three models between 1.24B and 2.61B parameters. Larger or more multilingual models may distribute language-specific computation diferently. Equal k removes a larger depth fraction from 16-block Llama than from Qwen or Gemma; this likely contributes to Llama’s collapse. The paper reports those fractions and does not use absolute cross-model degradation as a comparison of model quality.

Language coverage. Hindi and Marathi are related and share Devanagari. Tamil adds a Dravidian, non-Devanagari control, but all three target languages are Indian languages and use non-Latin scripts. Without a Latin-script non-English control such as Spanish or German, the experiment cannot fully separate an Indian-language efect from a broader English-versus-non-English efect. We consequently phrase the finding for the languages studied rather than for all multilingual text.

Data and outcomes. Wikipedia is a single encyclopedic domain, and editions difer in size, style, and quality. The primary outcome is held-out next-token perplexity. It directly measures the language-modeling consequence of block removal but does not guarantee corresponding changes in generation quality, factuality, or downstream-task accuracy. We do not study recovered models, instruction tasks, latency, or energy.

Importance proxy. BI captures local angular change and can fail as a causal measure of block necessity. Llama’s performance relative to random pruning demonstrates this limitation directly. Relative- $L _ { 2 }$ profiles are supplied as a diagnostic, but all primary selections use BI.

Sampling and inference. The study uses one deterministic Wikipedia stream, 200 calibration and 200 evaluation articles per language, and five calibration subsets. Hierarchical intervals represent these calibrationsubset and article sources of variation, not alternative corpora or model checkpoints. Discrete layer selection also produces repeated outcomes and zero seed variance; this should not be read as zero evaluation uncertainty.

Checkpoint provenance. All reported experiments use organization-hosted repositories, but the conclusions remain specific to the exact checkpoint revisions listed in Appendix A. Other releases, quantizations, or instruction-tuned variants may yield diferent block rankings and pruning outcomes.

Exploratory scope. The early/middle/late partition and primary budget were fixed in code, but this was not a formally registered study. The evidence should be replicated on additional domains, Latin-script non-English languages, and larger models before stronger population-level claims are made.

## 8 Broader Impact

Multilingual compression can reduce serving cost and make language technology more accessible under limited hardware budgets. It can also silently shift quality loss toward languages that are absent from calibration. The main positive implication of this work is procedural: evaluate block-selection decisions on the languages a system is expected to serve, including languages not used to construct the selector.

The results should not be used to certify a pruned model as safe for a language. We evaluate language modeling on Wikipedia, not harmful generation, dialectal coverage, or application-specific reliability. A deployment should retain task-specific and human evaluation, document its calibration mixture, and avoid using average multilingual performance to conceal a severe failure in one language.

## 9 Conclusion

English-only calibration systematically assigns lower relative BI to late blocks than Hindi, Marathi, and Tamil across Qwen2.5-1.5B, Llama-3.2-1B, and Gemma-2-2B. The efect is localized, statistically consistent over held-out calibration articles, and large enough to change which blocks are removed despite high global rank correlations.

Equal-budget raw-mean calibration over English, Hindi, and Marathi avoids damaging English-selected blocks for Qwen at the largest tested budget and for Gemma at every tested budget, including gains on held-out Tamil. Llama prevents a broader claim: balancing helps Hindi and Marathi at two removed blocks but sharply worsens Tamil, and BI pruning itself is brittle. The appropriate conclusion is therefore diagnostic and conditional. Calibration language is part of the BI pruning algorithm, and multilingual balance is a candidate mitigation that must be tested per model, pruning fraction, and deployment language.

## Code and Data Availability

Code, experiment configurations, article identifiers, block scores, pruning sets, and per-seed results are available at https://github.com/VedantPandhare/Language-Balanced-Depth-Pruning-of-Small-Mul tilingual-LLMs.

## References

Abhinav Bandari, Lu Yin, Cheng-Yu Hsieh, Ajay Kumar Jaiswal, Tianlong Chen, Li Shen, Ranjay Krishna, and Shiwei Liu. Is C4 dataset optimal for pruning? an investigation of calibration data for LLM pruning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 18089– 18099. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.1004. URL https://aclanthology.org/2024.emnlp-main.1004/.

Rochelle Choenni and Ivan Titov. M-Wanda: Improving one-shot pruning for multilingual LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 26951– 26964. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.emnlp-main.1369. URL https://aclanthology.org/2025.emnlp-main.1369/.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

Elias Frantar and Dan Alistarh. SparseGPT: Massive language models can be accurately pruned in one-shot. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 10323–10337. PMLR, 2023. URL https://proceedings.mlr.press/v202/frantar23a.html.

Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024. URL https://arxiv.org/abs/2408.00118.

Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A. Roberts. The unreasonable inefectiveness of the deeper layers. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2403.17887.

Hwichan Kim, Jun Suzuki, Tosho Hirasawa, and Mamoru Komachi. Pruning multilingual large language models for multilingual inference. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 9921–9942. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-emnlp.580. URL https://aclanthology.org/2024.findings-emnlp.580/.

Simon Kurz, Jian-Jia Chen, Lucie Flek, and Zhixue Zhao. On the limitations of language-targeted pruning: Investigating the calibration language impact in multilingual LLM pruning. Transactions of the Association for Computational Linguistics, 14:167–192, 2026. doi: 10.1162/tacl.a.599. URL https://aclanthology.o rg/2026.tacl-1.9/.

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. ShortGPT: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853, 2024. URL https://arxiv.org/abs/2403.03853.

Kelechi Ogueji, Orevaoghene Ahia, Gbemileke Onilude, Sebastian Gehrmann, Sara Hooker, and Julia Kreutzer. Intriguing properties of compression on multilingual models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9092–9110. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.626. URL https://aclanthology.org/2022.emnl p-main.626/.

Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi. Language model tokenizers introduce unfairness between languages. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/4f4461b5b8cb3c7bbbc 1d469ec6dbaf2-Abstract-Conference.html.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2025. URL https://arxiv.org/ abs/2412.15115.

Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, and Jae-Joon Kim. SLEB: Streamlining LLMs through redundancy verification and elimination of transformer blocks. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pp. 46036–46055. PMLR, 2024. URL https://proceedings.mlr.press/v235/song24c.html.

Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A simple and efective pruning approach for large language models. In International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=PxoFut3dWW.

Tianyi Tang, Wuchang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen. Language-specific neurons: The key to multilingual capabilities in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.acl-long.309/.

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. Do llamas work in english? on the latent language of multilingual transformers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.acl-long.820/.

Haryo Akbarianto Wibowo, Haiyue Song, Hideki Tanaka, Masao Utiyama, Alham Fikri Aji, and Raj Dabre. Multilingual iterative model pruning: What matters? In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the ACL, pp. 543–571. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.ijcnlp-long.32. URL https://aclanthology.org/2025.ijcnlp-long.32/.

Wikimedia Foundation. Wikipedia dumps, 2023-11-01. https://dumps.wikimedia.org/, 2023. Accessed through the Wikimedia Wikipedia dataset distribution.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Eficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=NG7sS51zVF.

Yifei Yang, Zouying Cao, and Hai Zhao. LaCo: Large language model pruning via layer collapse. arXiv preprint arXiv:2402.11187, 2024. URL https://arxiv.org/abs/2402.11187.

Hongchuan Zeng, Hongshen Xu, Lu Chen, and Kai Yu. Multilingual brain surgeon: Large language models can be compressed leaving no language behind. In Proceedings of LREC-COLING 2024, pp. 11794–11812. ELRA and ICCL, 2024. URL https://aclanthology.org/2024.lrec-main.1030/.

Yiran Zhao, Wenxuan Zhang, Guanzheng Chen, Kenji Kawaguchi, and Lidong Bing. How do large language models handle multilingualism? In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/abs/2402.18815.

## A Reproducibility and Provenance

## A.1 Exact checkpoints

Table 4: Model identifiers and immutable repository revisions used for the reported runs.
<table><tr><td>Model</td><td>Repository identifier</td><td>Revision</td></tr><tr><td>Qwen</td><td>Qwen/Qwen2.5-1.5B-Instruct</td><td>989aa7980e4c...</td></tr><tr><td>Llama</td><td>meta-1lama/Llama-3.2-1B</td><td>4e20de362430...</td></tr><tr><td>Gemma</td><td>google/gemma-2-2b</td><td>c5ebcd40d208...</td></tr></table>

The complete 40-character revisions are stored in each released experiment\_config.json. All three identifiers point to repositories maintained by the respective model organizations. Each run used a model-specific results directory and recorded the model identifier, immutable revision, software environment, numerical precision, evaluation fingerprint, and cache statistics before its outputs were incorporated into the paper.

## A.2 Data construction

The sampling procedure is deterministic given seed 42. The cached source stream is shufled with bufer size 5,000. For each language, the first 200 valid articles provide evaluation chunks and the next 200 provide calibration chunks. The code stores all article identifiers and asserts uniqueness and cross-split disjointness. Cross-model asset generation additionally asserts equality of the article-identifier tables before combining any results.

Every calibration and evaluation row corresponds to one article. The independent article, rather than a token or overlapping window, is therefore the resampling unit. The paper artifact includes calibration\_article\_ids.csv and evaluation\_article\_ids.csv for every model.

## A.3 Artifact map

The public repository at https://github.com/VedantPandhare/Language-Balanced-Depth-Pruning-o f-Small-Multilingual-LLMs contains:

• the complete v4 experiment notebook and its deterministic notebook builder;

• exact model configurations and revisions;

• per-layer BI and relative- $L _ { 2 }$ arrays summarized by language;

• full-pool and per-seed pruning sets;

• all 420 primary rows and 99 contrast rows per model;

• random-pruning controls, stability summaries, and article identifiers; and

• a SHA-256 manifest for every compact result file.

The large model weights and per-configuration logits are not redistributed. Per-chunk NLL caches are also not redistributed. The included tables are suficient to verify every number in the manuscript.

## A.4 Software and execution

The recorded environment is Transformers 4.56.2, PyTorch 2.11.0+cu130, bfloat16, and one NVIDIA A100- SXM4-40GB. The notebook records 200 influence chunks and 200 evaluation chunks per language, sequence length 256, five calibration seeds, primary budget 150, and ten random seeds. Optional budget-sweep and Belebele cells were disabled; neither contributes to any claim in this paper.

## B Additional Results

## B.1 Baseline quality and tokenization

Table 5: Intact-model perplexity (PPL), bits per character (BPC), and mean tokenizer tokens per whitespacedelimited word. Values are descriptive; PPL is not compared across tokenizers.

<table><tr><td>Model</td><td>Language</td><td>PPL</td><td>BPC</td><td>Tokens/word</td></tr><tr><td rowspan="4">Qwen2.5-1.5B</td><td>EN</td><td>10.207</td><td>0.790</td><td>1.533</td></tr><tr><td>HI</td><td>3.827</td><td>1.776</td><td>4.996</td></tr><tr><td>MR</td><td>5.256</td><td>2.284</td><td>6.730</td></tr><tr><td>TA</td><td>3.620</td><td>1.920</td><td>9.553</td></tr><tr><td rowspan="4">Llama-3.2-1B</td><td>EN</td><td>10.158</td><td>0.755</td><td>1.459</td></tr><tr><td>HI</td><td>6.093</td><td>1.357</td><td>2.854</td></tr><tr><td>MR</td><td>6.939</td><td>1.597</td><td>4.009</td></tr><tr><td>TA</td><td>2.321</td><td>1.507</td><td>11.635</td></tr><tr><td rowspan="4">Gemma-2-2B</td><td>EN</td><td>8.916</td><td>0.727</td><td>1.500</td></tr><tr><td>HI</td><td>8.879</td><td>1.262</td><td>2.210</td></tr><tr><td>MR</td><td>9.154</td><td>1.508</td><td>3.389</td></tr><tr><td>TA</td><td>6.057</td><td>1.233</td><td>4.312</td></tr></table>

Tamil has the highest tokenizer fertility in each model, yet balanced calibration transfers positively to Tamil for Qwen and Gemma and negatively for Llama. Fertility alone therefore does not explain the direction o the mitigation result.

## B.2 Complete aggregator summary

Table 6 reports mean and worst relative PPL degradation over Hindi, Marathi, and Tamil. Means and standard deviations are over calibration seeds. A standard deviation of zero indicates that all five subsets selected the same block set; it does not remove evaluation-article uncertainty.

Table 6: Target-language relative PPL degradation (%) for English-only and balanced BI selectors.
<table><tr><td>Model</td><td>Selector</td><td>Statistic</td><td>k = 2</td><td>SD</td><td>k = 4</td><td>SD</td><td>k = 6</td><td>SD</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>EN-only</td><td>Mean</td><td>6.68</td><td>0.00</td><td>20.72</td><td>0.00</td><td>88.89</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>8.45</td><td>0.00</td><td>25.61</td><td>0.00</td><td>107.5</td><td>0.00</td></tr><tr><td></td><td>Raw mean</td><td>Mean</td><td>6.68</td><td>0.00</td><td>20.36</td><td>0.81</td><td>30.84</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>8.45</td><td>0.00</td><td>25.27</td><td>0.76</td><td>38.03</td><td>0.00</td></tr><tr><td></td><td>Rank mean</td><td>Mean</td><td>6.68</td><td>0.00</td><td>18.90</td><td>0.00</td><td>30.84</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>8.45</td><td>0.00</td><td>23.91</td><td>0.00</td><td>38.03</td><td>0.00</td></tr><tr><td></td><td>Minimax rank</td><td>Mean</td><td>6.68</td><td>0.00</td><td>19.26</td><td>0.81</td><td>34.17</td><td>7.44</td></tr><tr><td></td><td></td><td>Worst</td><td>8.45</td><td>0.00</td><td>24.25</td><td>0.76</td><td>41.58</td><td>7.93</td></tr><tr><td>Llama-3.2-1B</td><td>EN-only</td><td>Mean</td><td>1,462</td><td>0.00</td><td>9.06e+05</td><td>0.00</td><td>8.76e+05</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>3,081</td><td>0.00</td><td>2.45e+06</td><td>0.00</td><td>1.93e+06</td><td>0.00</td></tr><tr><td></td><td>Raw mean</td><td>Mean</td><td>2,694</td><td>0.00</td><td>9.06e+05</td><td>0.00</td><td>8.76e+05</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>6,572</td><td>0.00</td><td>2.45e+06</td><td>0.00</td><td>1.93e+06</td><td>0.00</td></tr><tr><td></td><td>Rank mean</td><td>Mean</td><td>2,694</td><td>0.00</td><td>9.06e+05</td><td>0.00</td><td>8.76e+05</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>6,572</td><td>0.00</td><td>2.45e+06</td><td>0.00</td><td>1.93e+06</td><td>0.00</td></tr><tr><td></td><td>Minimax rank</td><td>Mean</td><td>2,694</td><td>0.00</td><td>9.06e+05</td><td>0.00</td><td>8.76e+05</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>6,572</td><td>0.00</td><td>2.45e+06</td><td>0.00</td><td>1.93e+06</td><td>0.00</td></tr><tr><td>Gemma-2-2B</td><td>EN-only</td><td>Mean</td><td>50.82</td><td>0.00</td><td>78.52</td><td>0.00</td><td>242.3</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>68.36</td><td>0.00</td><td>97.66</td><td>0.00</td><td>327.6</td><td>0.00</td></tr><tr><td></td><td>Raw mean</td><td>Mean</td><td>16.81</td><td>0.00</td><td>34.19</td><td>0.00</td><td>132.5</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>18.65</td><td>0.00</td><td>37.04</td><td>0.00</td><td>165.9</td><td>0.00</td></tr><tr><td></td><td>Rank mean</td><td>Mean</td><td>16.81</td><td>0.00</td><td>78.52</td><td>0.00</td><td>242.3</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>18.65</td><td>0.00</td><td>97.66</td><td>0.00</td><td>327.6</td><td>0.00</td></tr><tr><td></td><td>Minimax rank</td><td>Mean</td><td>16.81</td><td>0.00</td><td>78.52</td><td>0.00</td><td>242.3</td><td>0.00</td></tr><tr><td></td><td></td><td>Worst</td><td>18.65</td><td>0.00</td><td>97.66</td><td>0.00</td><td>327.6</td><td>0.00</td></tr></table>

## B.3 Full-pool selection examples

Table 7: Full-pool BI pruning sets for the two primary selectors. Indices are zero-based.
<table><tr><td>Model</td><td>Selector</td><td> $k = 2$ </td><td> $k = 4$ </td><td>k = 6</td></tr><tr><td rowspan="2">Qwen</td><td>EN</td><td>15,16</td><td>14-17</td><td> $^ { 1 2 , 1 4 - 1 7 , 2 6 }$ </td></tr><tr><td>Raw mean</td><td>15,16</td><td>14-17</td><td>12-17</td></tr><tr><td rowspan="2">Llama</td><td>EN</td><td>12,13</td><td>11-14</td><td>7,10-14</td></tr><tr><td>Raw mean</td><td>12,14</td><td>11-14</td><td>7,10-14</td></tr><tr><td rowspan="2">Gemma</td><td>EN</td><td>12,23</td><td>10-12,23</td><td>9–12,22,23</td></tr><tr><td>Raw mean</td><td>10,12</td><td>9-12</td><td>8-12,23</td></tr></table>

The complete monolingual and aggregated selections, including all five primary seeds, are supplied as JSON. Full-pool sets are descriptive because the primary equal-budget analysis uses 150-chunk subsets.

## B.4 Rank agreement and relative- $L _ { 2 }$ diagnostic

For BI, English-to-target Spearman correlation ranges are 0.898–0.910 (Qwen), 0.947–0.982 (Llama), and 0.906–0.949 (Gemma). Corresponding Jaccard overlaps can be as low as 0.33. Relative- $L _ { 2 }$ produces the same broad depth shape but lower agreement in several pairs; its complete correlations and per-layer values are supplied. Because the study’s hypothesis and all selection outcomes are defined for BI, we do not reinterpret relative- $L _ { 2 }$ as a second primary test.

## B.5 Random-pruning target means

Table 8: Mean relative PPL degradation (%) over Hindi, Marathi, and Tamil. Random is averaged over ten block sets; EN and raw mean are averaged over five calibration seeds.
<table><tr><td></td><td colspan="3">Qwen</td><td colspan="3">Llama</td><td colspan="3">Gemma</td></tr><tr><td>k</td><td>EN</td><td>LB</td><td>Random</td><td>EN</td><td>LB</td><td>Random</td><td>EN</td><td>LB</td><td>Random</td></tr><tr><td>2</td><td>6.7</td><td>6.7</td><td>72.8</td><td>1461.7</td><td>2694.4</td><td>429.7</td><td>50.8</td><td>16.8</td><td>198.6</td></tr><tr><td>4</td><td>20.7</td><td>20.4</td><td>1122.1</td><td> $9 . 0 6 \times 1 0 ^ { 5 }$ </td><td> $9 . 0 6 \times 1 0 ^ { 5 }$ </td><td> $3 . 1 3 \times 1 0 ^ { 5 }$ </td><td>78.5</td><td>34.2</td><td> $7 . 4 8 \times 1 0 ^ { 4 }$ </td></tr><tr><td>6</td><td>88.9</td><td>30.8</td><td> $1 . 8 1 \times { { 1 0 } ^ { 7 } }$ </td><td> $8 . 7 6 \times 1 0 ^ { 5 }$ </td><td> $8 . 7 6 \times 1 0 ^ { 5 }$ </td><td> $1 . 0 1 \times 1 0 ^ { 7 }$ </td><td>242.3</td><td>132.5</td><td> $2 . 1 1 \times 1 0 ^ { 8 }$ </td></tr></table>

## C Statistical Details

## C.1 Late-layer bootstrap

For each target language and model, we draw 200 English calibration chunks and 200 target-language chunks independently with replacement. BI is recomputed by averaging the stored chunk-level influence values, then transformed into layer-wise log ratios using Eq. (5). The middle and late means and their diference are calculated for each of 5,000 replicates. Table 2 uses percentile intervals from this distribution.

## C.2 Hierarchical pruning contrast

For conditions a and $b ,$ seed s, and evaluation article $i ,$ let

$$
d _ { s , i } = \mathrm { N L L } _ { a , s , i } - \mathrm { N L L } _ { b , s , i } .
$$

Each bootstrap replicate samples five seed indices with replacement and one shared set of 200 article indices with replacement. It averages the resulting 5 × 200 diferences. The reported PPL ratio is exp(d), and interval endpoints exponentiate the 2.5th and 97.5th percentiles of the log-scale replicates.

This procedure preserves pairing between two pruning conditions on the same evaluation article while representing variation from both calibration subsets and evaluation articles. When conditions select identical

sets, all diferences are identically zero. When only one seed difers, as for Qwen raw mean at k = 4, the lower endpoint can touch one; we describe that result as inconclusive rather than using rounded endpoints to claim a reliable gain.

## C.3 Multiplicity and interpretation

The artifact includes exploratory sign-flip tests with Holm adjustments because they were generated by the experiment notebook. The paper does not use those p-values as its primary evidence: with only five calibration subsets and a discrete selector, averaging seeds before a sign-flip test can underrepresent selector variation. We instead report efect sizes, hierarchical intervals, exact set-equality rates, and consistency of the predeclared late-depth pattern.

## D Implementation Checks

Block outputs. The final entry returned by a model’s global hidden-state API can include the final RMS normalization. Comparing that tensor with the input to the last decoder block inflates the apparent influence of the last block. We use forward hooks on each decoder block and compare its direct input and output.

Beginning-of-sequence handling. Llama and Gemma define a beginning-of-sequence token. Omitting it caused implausible Gemma baselines in pilot experiments. We prepend it for models that define it and exclude its position from BI because attention sinks can have outlier activations (Xiao et al., 2024).

No cumulative pruning. Every evaluated set is applied to the original ordered list of decoder blocks. A context manager restores that list after evaluation, and the notebook re-evaluates the intact baseline at the end. Thus the k = 4 model is not obtained by further pruning a previously modified k = 2 model.

Cache identity. Per-configuration NLL cache keys include model identifier, repository revision, sequence length, and evaluation-article identifiers. Influence files are kept in model-specific result directories, preventing cached outputs from being reused across checkpoints.

Determinism. Block selection uses deterministic tie-breaking. Random controls use fixed seeds derived from k. The code records package versions, model revision, all selected sets, and an evaluation fingerprint. Data loading is the only network-dependent stage once model files are available; sampled article records are cached.