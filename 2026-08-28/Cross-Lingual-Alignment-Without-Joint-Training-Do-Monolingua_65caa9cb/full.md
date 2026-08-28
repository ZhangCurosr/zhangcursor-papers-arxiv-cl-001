# Cross-Lingual Alignment Without Joint Training: Do Monolingual Language Models Converge on Universal Representations?

Ej Zhou¹ Suchir Salhan¹ Catherine Arnett2 Anna Korhonen¹

1University of Cambridge 2EleutherAI

{yz926,sas245,alk23}@cam.ac.uk catherine@eleuther.com

## Abstract

Cross-lingual alignment in multilingual language models is what enables cross-lingual transfer and is typically attributed to joint training: shared parameters, mixed-language batches, or explicit alignment objectives. In this paper, we ask whether monolingual models trained on non-parallel data learn alignable representations without joint training. By testing on strictly monolingual language models, such as the Goldfish model families and independently developed models from different research labs, we find three results. Correlation: these models develop alignable representational geometry across layers, with alignment strengthening as data scale, model scale, or linguistic proximity increases. Construction: a single Procrustes rotation fit on parallel sentences maps hidden states between models. Causation: the same rotation transfers functional content; patching a rotated English residual into a German model on a factual cloze task flips the prediction to the English model's answer in most cases. We argue that cross-lingual alignment can emerge from the structure of language and the information it carries rather than from joint training and shared representations. This points to practical future directions including model stitching, merging, and modular multilingual systems built from monolingual components.

## 1 Introduction

Multilingual language models produce representations in which cross-lingual semantically parallel inputs occupy close regions of activation space (Chang et al., 2022; Brinkmann et al., 2025). The research community has proposed various explanations in mechanism: shared subword vocabularies (Pires et al., 2019), similar input distributions (Conneau et al., 2020b), or shared architectural parameters (Dufter and Schütze, 2020). These studies only investigate the cause within the features of joint training, and no existing account predicts cross-lingual alignment between models trained independently on disjoint corpora.

In this paper, we ask whether joint training is required for cross-lingual alignment. We pose this question in its strictest form: if two models are trained independently, on distinct corpora and in different languages, with no shared parameters or alignment signal, do their representations still end up alignable? A positive answer would attest the emergence of language-agnostic conceptual representations purely from the corpora. This is predicted by the Platonic Representation Hypothesis (Huh et al., 2024), which has been tested cross-modally but not cross-linguistically. In our experiments, we consider only strictly monolingual language models: each is a Transformer trained on a single language. Our central findings replicate on five independently developed \~1Bparameter monolingual models from different research groups, showing the effect holds at larger scale.

Our analysis is three-part, with increasingly stringent standards of evidence: Correlation, Construction, and Causation. In the correlational experiments (Section 3) we find that independently trained monolingual models develop alignable representational geometry on parallel sentences, with alignment strengthening as data scale, model scale, or linguistic proximity increases, and persisting across models from different research groups. In the constructional experiments (Section 4), we compare an orthogonal rotation against more flexible mappings (affine, MLP) and find that a single rotation is sufficient to align one model's representations with another's. The orthogonality constraint preserves the angular geometry that retrieval depends on, whereas the relaxed alternatives lower reconstruction error but weaken retrieval. Finally, in the causal experiments (Section 5), we show that the same rotation transfers functional content:

![](images/80815260d25d58c67b6b616c687b9c3911b3b2429b4729938e8d85fffa094f17.jpg)  
Figure 1: Independent monolingual LMs share representational geometry; a rotation makes that geometry causally usable. (A) Parallel English/German sentences fed to monolingual models yield residual streams $h _ { \mathrm { e n } } , h _ { \mathrm { d e } }$ with high CKA. (B) A Procrustes rotation W fit on parallel sentence representations aligns the two clouds: ${ \widehat { Y } } { = } X W$ overlaps the target. (C) The same W transfers factual content across models. Patching $W { \cdot } h _ { \mathrm { e n } }$ at the residual of a MONO-DE prompt flips the German completion, recovering directional success rate $( \Delta \log p ( \mathrm { d o n o r } ) { > } 0 )$ most of the ceiling and well above baselines.

patching a rotated English residual into a German model on a factual cloze task flips the prediction to the donor's capital in most cases, with the pattern holding across five unrelated target languages and approaching the within-model ceiling.

## 2 Related Work

## 2.1 Cross-Lingual Representations in Multilingual LMs

A growing literature finds that multilingual language models develop partly shared representational structure across languages. Early evidence in jointly trained encoders established that languages occupy overlapping linear subspaces (Chang et al., 2022). Subsequent work on decoderonly multilingual LMs has found a similarly partial language-agnostic geometry, with internal components shared across languages while still allowing language-specific divergences (Zhang et al., 2025; Wendler et al., 2024; Brinkmann et al., 2025; Dumas et al., 2025; Riemenschneider and Frank, 2025; Körner et al., 2026; Harrasse et al., 2026). More recently, the Platonic Representation Hypothesis (Huh et al., 2024) posits that representations converge towards a shared model of reality as models scale. Jha et al. (2025) support this at the embedding level, showing text embeddings can be translated across architectures without paired data. In this paper, we connect these predictions and ask whether there are cross-lingually universal representations learned by monolingual models.

## 2.2 Monolingual LMs

Most state-of-the-art large-scale LMs (Qwen et al., 2025; Team et al., 2024) and well-studied multilingual LMs (Devlin et al., 2019; Conneau et al., 2020a; Le Scao et al., 2022) are trained jointly across many languages. For our experiments, we require strictly monolingual models, in order to study whether language-agnostic structure can emerge independently of multilingual joint training. Recent efforts have produced such models, namely the Goldfish models (Chang et al., 2026), which include comparably trained monolingual models for 350 languages. These provide a controlled setting for our experiments. There are also several monolingual models developed by language communities (e.g. Minerva, Orlando et al., 2024; Zh-Pythia Liu et al., 2025; Tucano, Corrêa et al., 2025; and Bielik, Ociepa et al., 2025). We use these models to test the scalability of our findings, albeit in a less

controlled setting.

## 2.3 Aligning Independently-Trained LMs

Several studies have found that independently trained representations can be aligned post hoc Early work in Bilingual Lexicon Induction (BLI) demonstrates that monolingual word embedding spaces are often related through approximately orthogonal transformations, enabling bilingual lexicon induction without parallel supervision (Mikolov et al., 2013; Xing et al., 2015; Smith et al., 2017; Artetxe et al., 2018; Lample et al., 2018). Orthogonality constraints usually improve stability and performance relative to unconstrained linear mappings, suggesting that independently trained semantic spaces preserve similar global geometry. Geometric alignment measures, such as Centered Kernel Alignment (Kornblith et al. 2019), which builds on the Hilbert-Schmidt Independence Criterion (HSIC; Gretton et al., 2005), a kernel-based measure of statistical dependence between two sets of representations, have been used to probe multilingual encoders (Muller et al.. 2021). However, correlational similarity is insufficient on its own for making claims about shared representation, as CKA can be dominated by a small number of high-variance components (Davari et al., 2023), inflated by model width and depth with only local neighbourhood agreement surviving null calibration (Gröger et al., 2026). Additionally, nearest-neighbour alignment (as was used in Huh et al., 2024) can be fragile to evaluation scale (Koepke et al., 2026). Mechanistic interpretability approaches provide a stronger test of representational convergence by testing functional interchangeability: whether aligned states are usable by the receiving model's computation. In this work, we use cross-model activation patching (Dumas et al., 2025; Körner et al., 2026) as a causal test. If a representation from one model can be projected into another model's residual stream and successfully drive downstream behaviour on a new task, then the alignment must preserve functionally relevant structure rather than correlational similarity alone.

The closest prior work to ours is Conneau et al. (2020b), who train monolingual BERT models on separate corpora, measure their similarity with CKA, and align them post hoc with a learned linear map. We extend their findings in four directions: (i) we analyse decoder-only causal LMs rather than masked-language encoders; (ii) we add causal experiments that test whether aligned states drive the receiving model's behaviour (Section 5); (iii) we vary training data over four controlled tiers (Section 3.3); and (iv) we replicate the finding across five ～1B models developed by different research groups (Section 3.4).

## 3 Representational Alignment

We first ask whether independently trained monolingual models converge on similar internal representations of parallel semantic content. We assess this correlationally by measuring CKA between model pairs under matched and shuffled controls.

## 3.1 Experimental Setup

Models. We start by using strictly monolingual Goldfish causal language models (Chang et al., 2026).2 All models share an identical Transformer architecture and training data size (adjusted for amount of information content), but have disjoint vocabularies, parameters, and training corpora. The family uses two architectures matched to data scale: the 5 MB and 10 MB models have 39M parameters, while the 100 MB and 1000 MB models have 125M parameters. Unless otherwise stated, we use the 1000 MB tier.³

Parallel data. We evaluate on four parallel corpora: FLORES-200 (Team et al., 2022), Tatoeba (Tiedemann, 2020), OPUS (Tiedemann, 2012), and BouQUET (Andrews et al., 2025). For each dataset and language pair, we extract hidden states from every Transformer layer, including the embedding layer.

Similarity metric. We measure cross-model similarity with linear CKA (Kornblith et al., 2019):

$$
\operatorname { C K A } ( X , Y ) = { \frac { \operatorname { H S I C } ( X , Y ) } { { \sqrt { \operatorname { H S I C } ( X , X ) \operatorname { H S I C } ( Y , Y ) } } } } ,
$$

which is invariant to orthogonal transformations and isotropic rescaling. To control for spurious similarity, we contrast a matched condition, which consists of aligned parallel sentences, with a shuffled control. For a language pair (A, B), the control keeps side A in order and reorders the rows of side B by a single random permutation (fixed seed), drawn once per pair and applied at every layer: each A sentence is paired with a $B$ sentence that is not its translation, and every B sentence appears exactly once. The control therefore preserves the architecture and corpus statistics of both sides, so the matched—shuffled gap isolates the semantic correspondence between paired sentences. We compare across all 36 language pairs.

Three representations. We compute CKA under three pooling strategies. Our sentence-level representation includes (i) Mean pooling: average token hidden states across the sequence; (ii) SGPT position-weighted pooling (Muennighoff, 2022): weight each token by its position in the sequence before averaging, emphasising information accumulated towards the end of the context, which proves to work better for causal language models; and (iii) Token-level word-aligned pooling: isolate lexical correspondence by averaging subword hidden states within each aligned word span, using SIMALIGN (Jalili Sabet et al., 2020) word alignments over the parallel sentence pair. For a word w with subword token set $T ( w )$ and per-token hidden states $\mathbf { h } _ { t , \ell }$ at layer l, $\begin{array} { r } { \mathbf { z } _ { w , \ell } = \frac { 1 } { | T ( w ) | } \sum _ { t \in T ( w ) } \mathbf { h } _ { t , \ell } . } \end{array}$

Per-pair CKA values for all three settings on all four datasets are in Appendix A.

## 3.2 Cross-Lingual Alignment Across 36 Pairs

We show the gap between the matched and shuffled conditions in Figure 2. We show the per-layer trajectory for a single representative pair (English– French) under all three pooling-strategies and the shuffled condition. We also report $\Delta$ between the two, where a larger delta indicates more improvement over the baseline. Table 1 averages the gap across all pairs in four datasets. Figure 3 unpacks the averages to a $9 \times 9$ matrix.

Semantic alignment exceeds architectural noise. Matched CKA is higher than the shuffled baseline at every layer and for every language pair (Figures 2 and 3). Under SGPT pooling, last-layer matched CKA averages 0.78 on FLORES, compared to 0.17 shuffled; the consistent gap is preserved across datasets (Table 1). Since the shuffled control uses the same sentences and only randomises the pairing, this gap cannot be explained by shared Transformer architecture or surface token statistics. Therefore, these results indicate stronger alignment than would be expected by chance.

Alignment is consistent across depth. Matched CKA is stable across layers under all three representations (Figure 2): in every case the layer-to-layer variation is small relative to the matched-shuffled gap.

![](images/6c4cfb58aa1300b00db19054c6fd600f5cc2c4b6f329656f197309794ca9280b.jpg)  
Figure 2: Layerwise English-French CKA (Goldfish 1000 MB) under three representations (mean pooling, token-aligned, SGPT). Solid: matched; dashed: shuffled.

Linguistic similarity modulates alignment. Figure 3 suggests a typological gradient: linguistically related pairs such as English–French (0.81), English–German (0.82) exhibit stronger alignment than distant pairs such as Chinese-Arabic (0.64) or Chinese-Hindi (0.65). We further test this by correlating last-layer matched CKA against URIEL typological distances (Littell et al., 2017) and a measure of model performance across the 36 Goldfish pairs. Following Chang et al. (2026), we use mean sentence negative log-likelihood (NLL) over parallel sentences to measure model quality. URIEL syntactic distance is the strongest single predictor $( \rho { = } { - } 0 . 6 4 , p { < } . 0 0 1 )$ . This is consistent with previous work that shows that more similar languages have higher representational overlap and more cross-lingual transfer (Chang et al., 2024; Arnett et al., 2025). Mean NLL is also a significant predictor of CKA alignment $( \rho { = } { - } 0 . 4 3 , p { = } . 0 1 )$ which suggests that the quality of the model's representations affects cross-lingual alignability: better models are more aligned across languages. This is predicted by the Platonic Representation Hypothesis, which states that representational convergence increases with model capability. A joint regression confirms that syntactic distance is the dominant predictor, improving fit over mean NLL alone $( \Delta R ^ { 2 } { = } . 1 9 , F ( 1 , 3 3 ) { = } 9 . 8 8 , p { = } . 0 0 4 )$ . Full correlations and visualisations are provided in $\mathsf { A p - }$ pendix D.

<table><tr><td></td><td colspan="3">Mean Pooling</td><td colspan="3">Token-Aligned</td><td colspan="3">SGPT</td></tr><tr><td>Dataset</td><td>Matched</td><td>Shuffled</td><td>∆</td><td>Matched</td><td>Shuffled</td><td>∆</td><td>Matched</td><td>Shuffled</td><td>∆</td></tr><tr><td>FLORES</td><td>0.72</td><td>0.15</td><td>+0.57</td><td>0.38</td><td>0.03</td><td>+0.35</td><td>0.78</td><td>0.17</td><td>+0.61</td></tr><tr><td>Tatoeba</td><td>0.69</td><td>0.16</td><td>+0.53</td><td>0.36</td><td>0.03</td><td>+0.33</td><td>0.75</td><td>0.18</td><td>+0.57</td></tr><tr><td>OPUS</td><td>0.68</td><td>0.15</td><td>+0.53</td><td>0.35</td><td>0.03</td><td>+0.32</td><td>0.73</td><td>0.17</td><td>+0.56</td></tr><tr><td>BouQUET</td><td>0.70</td><td>0.16</td><td>+0.54</td><td>0.37</td><td>0.03</td><td>+0.34</td><td>0.76</td><td>0.18</td><td>+0.58</td></tr><tr><td>Avg.</td><td>0.70</td><td>0.15</td><td>+0.54</td><td>0.36</td><td>0.03</td><td>+0.34</td><td>0.76</td><td>0.18</td><td>+0.58</td></tr></table>

Table 1: Mean last-layer CKA across all 36 Goldfish (1000 MB) language pairs, by representation and dataset. ∆ is the matched—shuffled gap. Full per-dataset 9 × 9 matrices are in Appendix Tables 11–22.

![](images/a80adea9c4221de614d43305f3075a4a969409935a10d72dc6e2e66498903050.jpg)  
Figure 3: Last-layer pairwise CKA (Goldfish 1000 MB, FLORES-200). Lower triangle: matched; upper triangle: shuffled.

Low-resource languages also confirm alignment. The analyses thus far have been limited to highresource languages. We test whether these results hold across different language families and writing systems. We repeat the same evaluation with the English model paired against four real-world lowresource languages: Tagalog (tg1), Swahili (swh), Northern Uzbek (uzn), and Amharic (amh). Table 2 reports matched and shuffled CKA at the last and middle layers: the gaps $\left( + 0 . 3 9 \mathrm { t o } + 0 . 5 8 \right)$ sit in the same range as the high-resource reference pairs, and hold at the middle layer as well.

## 3.3 Alignment Scales with Training Data

We use all four training-data scales available for Goldfish (5, 10, 100, 1000 MB) for each language, enabling a controlled study of how alignment grows with training data. Figure 4 plots mean off-diagonal matched CKA against training size for each of the three representations, with shuffled baselines for reference. Across all three representations, matched CKA increases monotonically with training-data scale.

<table><tr><td>Pair</td><td>Family (script)</td><td>Last layer M/S(∆)</td><td>Middle layer M/S(∆)</td></tr><tr><td>eng-tgl eng-swh eng-uzn eng-amh</td><td>Austrones. (Latn) Atl.–Congo (Latn) Turkic (Latn) Afro-As. (Ethi)</td><td>.83 /.25 (+.58) .81 /.25 (+.56) .80/.41 (+.39)</td><td>.84/.25 (+.59) .82/.24 (+.59) .80/.26(+.54).80/.25 (+.55) .81/.41 (+.41)</td></tr><tr><td>eng-fra eng-zho</td><td>Indo-Eur. (Latn) Sino-Tib. (Hans)</td><td>.86/.24(+.62) .82/ .23 (+.59)</td><td></td></tr></table>

Table 2: Matched (M) and shuffled (S) SGPT CKA and their gap (∆) for English paired with four lowresource languages, at the last and middle (L/2) layer. Goldfish 1000 MB models, FLORES-200, conditions as in Section 3.1. Reference values for eng-fra and engzho are the last-layer results from Table 19.

A cross-size comparison pairs Goldfish models trained on different amounts of data (1000 MB against 100 MB, and 10 MB against 5 MB) within each model size. We see the diagonal (same language) dominates every row, and off-diagonal matched CKA still sits above the shuffled baseline at both pairings. This supports the conclusions that as model quality improves, cross-lingual representational alignment gets stronger, which is consistent with both the Platonic Representation Hypothesis and our results in Section 3.2. We provide further details about this analysis in Appendix B.2.

## 3.4 Alignment Persists Between Larger Models

A natural concern is that alignment within Goldfish models is only an artefact from shared design choices across the family. We therefore test whether alignment survives when models differ in architecture, vocabulary, training data, and research group. As a first check, comparing Goldfish English against Pythia-160M (Biderman et al., 2023) and Goldfish Chinese against Zh-Pythia-160M (Liu et al., 2025) also yields matched SGPT CKA above the shuffled baseline at every layer, albeit in a Ushaped profile (more discussion in Appendix C.1). We then compare five independently developed \~1B monolingual models: Pythia-1.4B EN, Zh-Pythia-1.4B, Tucano-1B PT (Corrêa et al., 2025), Bielik-1.5B PL (Ociepa et al., 2025), Minerva-1B IT (Orlando et al., 2024), full specifications in Appendix Table 6. We conduct the same experiment as above. Table 3 reports last-layer CKA (SGPT) for all ${ \binom { 5 } { 2 } } \ = \ 1 0$ model pairs on FLORES-200. Matched CKA exceeds the shuffled baseline in all 10 pairs (0.71 vs. 0.18 on average), confirming that the alignment pattern extends to models sharing nothing but the objective of modelling text.

![](images/9faa2134a6731eb16901d63c5b1d3b8ad6748134303ad92bf340f28120db99df.jpg)  
Figure 4: Alignment scales with training data. Mean matched last-layer CKA grows monotonically from 5 MB to 1000 MB under all three representations. Full 9 × 9 heatmaps at every scale are provided in Appendix Figures 14–16.

![](images/8b7919e03887252d4716e12d163034a213796beb8cd55b0e728bee03182a69d2.jpg)  
Table 3: Last-layer CKA across five \~1B monolingual models (SGPT). Lower triangle (blue): matched. Upper triangle (red): shuffled. Detailed model specifications are in Appendix Table 6.

CKA establishes similarity up to an unspecified transformation: it shows the two spaces are alignable but does not identify the map that aligns them. In the following section, we will establish causal evidence for meaningful alignment across models.

## 4 Cross-Lingual Representation Reconstruction

Section 3 established that independently trained monolingual models produce representations whose pairwise similarity exceeds shuffled baselines. We now move from correlation to reconstruction,4 asking what class of learned map recovers one model's representations from another's.

## 4.1 Setup

We study all pairs of the Goldfish 1000 MB monolingual models from Section 3, and fit projections in both directions, yielding 72 directional pairs in total. As SGPT resulted in the strongest alignment, we use this method at every layer including the embedding layer. We use the same four parallel datasets as Section 3, yielding \~5,000 parallel sentences per language pair (80/20 train/test split, seed 42).5 On top of the resulting per-layer matrices $\ b { X } \in \mathbb { R } ^ { n \times d }$ (source) and $Y \in \mathbb { R } ^ { n \times d }$ (target), we fit a hierarchy of three projection methods that each relax a structural constraint. All three minimise squared reconstruction error.

(i) Procrustes. The most constrained mapping: a single rotation/reflection $W \in \mathbb { R } ^ { d \times d }$ that preserves all pairwise inner products in the source space (Schönemann, 1966).

$$
\boldsymbol { W } ^ { \star } = \underset { \boldsymbol { W } ^ { \top } \boldsymbol { W } = \boldsymbol { I } } { \arg \operatorname* { m i n } } ~ \| \boldsymbol { X } \boldsymbol { W } - \boldsymbol { Y } \| _ { F } ^ { 2 } ,\tag{1}
$$

solved in closed form by $\boldsymbol { W ^ { \star } } ~ = ~ \boldsymbol { U } \boldsymbol { V } ^ { \intercal }$ , where $U \Sigma V ^ { \top } = X ^ { \top } Y$ is the SVD.

(ii) Affine. We relax orthogonality: W is unconstrained and a bias $b \in \mathbb { R } ^ { d }$ is allowed.6

$$
( W ^ { \star } , b ^ { \star } ) = \underset { W , b } { \arg \operatorname* { m i n } } \ \| X W + \mathbf { 1 } b ^ { \top } - Y \| _ { F } ^ { 2 } .\tag{2}
$$

(iii) MLP. We further relax linearity using a onehidden-layer ReLU network $f _ { \theta }$ of width $d . { \bar { 7 } }$

$$
\theta ^ { \star } = \underset { \theta } { \arg \operatorname* { m i n } } ~ \| f _ { \theta } ( X ) - Y \| _ { F } ^ { 2 } ,\tag{3}
$$

$$
f _ { \boldsymbol { \theta } } ( x ) = W _ { 2 } \mathrm { R e L U } ( W _ { 1 } x + b _ { 1 } ) + b _ { 2 } .\tag{4}
$$

<table><tr><td>Method</td><td> $\mathrm { P @ 1 _ { s t d } }$ </td><td> $\mathrm { P } @ \mathrm { I } _ { \mathrm { h a r d } }$ </td><td>MSE</td><td>CKA</td></tr><tr><td>Procrustes</td><td> $\mathbf { . 8 8 7 \pm . 0 2 1 }$ </td><td> $. 7 7 5 { \scriptstyle \pm . 0 2 8 }$ </td><td> $. 6 8 9 { \scriptstyle \pm . 0 2 6 }$ </td><td> $. 7 1 3 { \scriptstyle \pm . 0 2 3 }$ </td></tr><tr><td>Affine</td><td> $. 8 1 4 _ { \pm . 0 2 7 }$ </td><td> $. 6 6 3 { \scriptstyle \pm . 0 3 6 }$ </td><td> $\mathbf { \nabla } . 4 3 9 _ { \pm . 0 1 7 }$ </td><td> $. 7 6 2 _ { \pm . 0 2 0 }$ </td></tr><tr><td>MLP</td><td> $. 8 0 9 { \scriptstyle \pm . 0 2 9 }$ </td><td> $. 6 4 3 { \scriptstyle \pm . 0 3 8 }$ </td><td> $. 4 5 6 _ { \pm . 0 1 7 }$ </td><td> $. 7 6 0 { \scriptstyle \pm . 0 2 0 }$ </td></tr><tr><td>Identity</td><td>.002</td><td>.000</td><td>2.258</td><td>.713</td></tr></table>

Table 4: Method comparison across 36 Goldfish language pairs (final layer). Each cell shows the mean across 36 pairs with bootstrapped 95% CI (10,000 resamples); bold marks the best method per metric. $\mathrm { P } @ 1 _ { \mathrm { s t d } }$ retrieves against the test-only pool; $\mathrm { P @ 1 _ { h a r d } }$ against the combined train+test pool. MSE and CKA are pool-independent. Per-pair scatter is in Appendix Figures 12 and 17.

We evaluate on three complementary metrics with distinct geometric meaning: P@1 (cosine retrieval, capturing local neighbourhood preservation), MSE (mean squared error; pointwise reconstruction quality), and post-projection linear CKA (relational structure). P@1 is reported in two variants: $\mathrm { P @ 1 _ { s t d } }$ retrieves against the test pool, while $\mathrm { P @ 1 _ { h a r d } }$ retrieves against the combined train+test pool.

## 4.2 Rotation Is Nearly Sufficient

Table 4 summarises reconstruction across all 72 directions. Three observations follow. (i) All three projections are highly effective. The identity baseline retrieves nothing $( \mathrm { P } @ 1 _ { \mathrm { s t d } } { = } 0 . 2 \% )$ ; a single Procrustes rotation leads to 88.7% retrieval accuracy, with MSE down from 2.258 to 0.689. A learned mapping recovers the matching target representation in the vast majority of cases. (ii) Affine and MLP win MSE and CKA, as expected. Both methods minimise reconstruction error directly and operate in a strictly larger hypothesis class than Procrustes, so unsurprisingly they attain lower MSE on all 36 pairs (Affine 0.439, MLP 0.456 vs. Procrustes 0.689) and higher postprojection CKA (rotation does not change CKA). (iii) Procrustes nonetheless wins retrieval. Procrustes beats the more flexible methods on P@1 on both the standard and the hard pool, while its MSE stays within range of $\mathbf { A f f i n e ^ { \prime } s }$ .We hypothesise this as an objective-metric mismatch: every method minimises Euclidean reconstruction error, but P@1 is rank-based and direction-sensitive. Because Procrustes is pure rotation, it preserves every angle in the source space, whereas Affine and MLP have no such constraint and would rescale axes anisotropically, lowering MSE at the cost of preserving angular geometry. The orthogonality constraint preserves better neighbourhood structure.

![](images/bda9fb922060b864a151accf7fe180d2fcd7fef1413727183caeee61b9f75932.jpg)  
Figure 5: Procrustes mapping, eng-fra, layer 12 (2D PCA over the joint test set). English in blue (source, Procrustes-mapped); French in yellow (target). The perlayer full version is in Figure 18.

Figure 5 visualises this directly for eng-fra at the final layer: projecting both languages into a shared 2D PCA basis and overlaying the Procrustesmapped English representations on the matched French targets, the two clouds coincide while the mapped source retains the geometry of the original English cloud. This is related to a pre-Transformer line of work on word-embedding alignment, where orthogonal mappings are shown to outperform unconstrained linear ones on retrieval (Mikolov et al., 2013; Xing et al., 2015; Smith et al., 2017; Artetxe et al., 2018; Lample et al., 2018). We observe the same dissociation in transformer hidden states.

## 4.3 Layers and residual structure analysis

Next, we verify that the mapping we find is reliable across layers, not only at the final layer reported in Table 4. Sweeping Procrustes across all 13 layers (Appendix F.1), retrieval stays well above the identity baseline at every layer and peaks in mid layers (6–8): eng-fra P@1=0.981 at layer 8; engrus 0.987 at layer 6. We confirm that rotation is usable across the depth of the network. A complementary dimensional analysis (Appendix F.2) examines where the post-Procrustes residual lives. The residual dimension is high-rank and concentrated outside the principal directions of the target; this is why neither Affine relaxation nor a one-layer MLP improves retrieval.

Our reconstruction experiment so far works geometrically. In Section 5 below we test for causal evidence that this rotation results in the best downstream result.

## 5 Cross-Model Activation Patching

Reconstruction shows that a single rotation maps one model's representations onto another's geometrically. We now ask whether the same rotation, applied with no further training, could transfer to a different task functionally. We test this on a factual recall task using cross-model activation patching, which substitutes a projected residual into another model's forward pass and measures whether the prediction follows the substitution.

## 5.1 Task and Intervention

Factual probe. We construct country→capital facts in six languages (English, French, German, Spanish, Japanese, Chinese), with hand-curated cloze prompts and answers, e.g. “The capital of France $\mathrm { i } s ^ { \prime \prime } $ Paris; "Die Hauptstadt von Frankreich is $t ^ { \prime \prime } $ Paris. The country word's last subword position is identified per language by a verb-marker heuristic (details in Appendix G).

Span patching at the concept position. For a (donor, target) cross-fact pair (with donor $\neq$ target country), we run the source-language model on the donor's source-language prompt and cache the residual stream $\mathbf { h } _ { S } ^ { ( k ) } [ c _ { S } ]$ at the donor coun-$\mathrm { { t r y } ^ { * } }$ last subword position $^ { c _ { S } . }$ , for every layer $k \ \in \ \left\{ 0 , \ldots , L \right\} \ ( L { = } 1 2 )$ The target-language model is then run on its own target-language target prompt, but at the target country position cT, we replace the residual $\mathbf { h } _ { T } ^ { ( k ) } [ c _ { T } ]$ with the projected donor residual $f ^ { ( k ) } \left( { \bf h } _ { S } ^ { ( k ) } [ c _ { S } ] \right)$ for all $k \in \{ j , \dots , L \}$ The intervention follows two design choices from Dumas et al. (2025): (i) the patch spans layers $j \to L$ rather than firing at a single layer, since a single-layer last-token patch is suppressed by downstream computation; and (ii) the patch lands on the concept word rather than the last token, so that the injected information must propagate through attention to influence the answer logits.

Conditions. We compare four sources for the patched residual: within-lang $( \mathbf { h } _ { T }$ , the target model patches its own donor representation, no projection; the ceiling on what any cross-lingual map could match), Procrustes $( W ^ { ( k ) } \mathbf { h } _ { S }$ , the rotation described in Section 4, applied per layer with no further training); unprojected (hs, raw sourcemodel residual, basis-mismatch control); and shuffled $( \sim \mathcal { N } ( \mu _ { S } , \Sigma _ { S } )$ , a Gaussian sample per layer matching the source model's activation mean and full covariance, distributional control). Affine (Section 4) was also evaluated but is excluded since it underperforms Procrustes on this probe; see Appendix G for the explanation.

![](images/25e94e79b6447b824b6326d423b804b1740388356d9a3079d20f8dd7cfb0bac2.jpg)  
Figure 6: Directional success rate (%) for eng→{fra, deu, spa, jpn, zho}, n=870 cross-fact pairs per cell, Wilson 95% CIs. Bars: within-model ceiling, Procrustes, unprojected residual, and shuffled Gaussian sample. Per target, the patch start layer $j \in \{ 1 , \dotsc , 8 \}$ is selected to maximise within and Procrustes. Dashed line at 50% chance.

Metric. We score with the directional success rate: the fraction of cross-fact pairs where ∆ log p(donor answer) $> 0$ in the target language under patching, with Wilson 95% CIs over n crossfact pairs per cell $( n { = } 8 7 0$ for country→capital). The directional rate is approximately invariant to source-model strength asymmetries; raw $\Delta$ log p is contaminated by the fact that some models hold country→capital knowledge more strongly than others.

## 5.2 Cross-Model Transfer of Factual Content

We report Procrustes as the cross-lingual projection of record (see Appendix G for discussions on Affine). Figure 6 reports cross-model transfer across five eng→{fra, deu, spa, jpn, zho} target languages at the patch-start layer that best separates Procrustes from controls per target. Four observations follow.

(i) Establishing the ceiling by patching with the same-language representation. Before evaluating any cross-model projection we verify that the intervention itself can carry factual content when no cross-lingual transfer is required. Within-model patching yields directional success rates of 76–98% across the five targets, reproducing the Dumas et al. (2025) concept-position span-patch design in our setting and giving an upper bound on what any cross-lingual map could match.

(ii) The fitted rotation from Section 4 successfully transfers factual concepts between monolingual models. The Procrustes map from Section 4 was fit on sentence-end pooled activations. We evaluate it here, without retraining, on a different task (factual cloze rather than news) at a different token position (the concept word midprompt rather than the sequence end). We score against a different readout (next-token logits rather than cosine retrieval). Remarkably, the rotation reaches directional success rates of 66–85% across the five targets, against a 76–98% within-model ceiling, recovering most of what an oracle samemodel intervention would achieve. We also see that the pattern holds across different languages and scripts.

(iii) Control settings (shuffle and unprojected) are cleanly near or below the chance line. The shuffled control sits approximately at the 50% chance line, as expected for a random perturbation. However, the unprojected control falls clearly below chance. We hypothesise that unprojected carries the donor concept in a basis that is mismatched with the target model, so when read by the target model's LM head it acts as a coherent signal pointing in the wrong direction, and sometimes produces the opposite effect.

(iv) The rotation transfers across factual relations. Appendix Table 10 extends the probe from country→capital to six further relations between the English and German models, reusing the same per-layer Procrustes maps with no refitting and a fixed patch start j=2. Several relations involve naturally multi-token entities (e.g. work titles spanning several words on the prompt side); all conditions score the answer at its first subword. Procrustes exceeds the shuffled control on all seven relations; the Wilson intervals separate on the four larger sets $( n \geq 3 9 8 )$ , and the remaining three are directionally consistent. We read the patching experiments as a proof of existence: a rotation fitted once on parallel sentence pools transfers zero-shot to a new task, a new token position, and new relations. Breadth of deployment, multi-token generation, and open-ended use remain outside this design (see Limitations).

Taking eng→deu as a concrete illustration: patching the German model's own donor representation flips the prediction to the donor's capital 98% of the time; the rotated English residual achieves 85% of cross-fact pairs; a random Gaussian sample sits near 50%. A single orthogonal map, fit once on parallel sentence pools, recovers most of what an oracle same-model intervention would do, while uninformed signals stay at or below chance.

## 6 Conclusion and Discussion

We set out asking whether strictly monolingual language models could be aligned across languages. In the correlational experiments, alignment between models exceeds shuffled controls on all pairs. In the constructional experiments, an orthogonal rotation recovers one model's hidden states from another's. In the causal experiments, the same rotation could patch an English residual into a German model and flip the factual prediction in 85% of cases. This evidence suggests that cross-lingual alignment can emerge from the structure of language and the world it describes, with no joint training required.

More broadly speaking, we could think of multilinguality as a special case of the more general phenomenon of representational convergence (Moschella et al., 2023). Language-agnostic shared representations observed in multilingual language models may be partly a result of joint multilingual pretraining and partly inherited from crosslingual representational convergence (K et al., 2020; Artetxe et al., 2020). Our results suggest that the latter is substantial, and that alignment already present in monolingual models can be recovered post hoc.

This opens a modular alternative to current multilingual pipelines: a small set of anchor pairs is in principle enough to stitch independently trained monolingual specialists into a shared space (Lenc and Vedaldi, 2015; Bansal et al., 2021), and likewise for model merging across languages (Ilharco et al., 2023; Yadav et al., 2023). In preliminary follow-up experiments, we stitch two monolingual Goldfish models into a functioning bilingual system that approaches the performance of a jointly trained bilingual baseline (Arnett et al., 2025); a systematic treatment is future work.

A further direction is to make alignment an explicit training objective, so that a low-resource monolingual model is trained to be alignable with a high-resource anchor and the post hoc rotation studied here becomes a channel for cross-lingual transfer. How to scale anchor-based projection to typologically distant pairs remains open.

## Limitations

Language and model coverage. The ninelanguage Goldfish analysis and the five-model ～1B replication are weighted towards higher-resource, predominantly Indo-European languages, reflecting which truly monolingual checkpoints are publicly available. The low-resource extension (Table 2) covers the correlational experiments only; reconstruction and patching for those languages remain untested. We cap at \~1B parameters because most larger releases marketed as monolingual retain measurable multilingual content in their pretraining data. Since alignment increases monotonically with training data (Section 3.3), we expect the same pattern to extend to 7B+ monolingual checkpoints, but this remains untested.

Language contamination. Foreign-language text in nominally monolingual corpora can itself induce cross-lingual ability (Blevins and Zettlemoyer, 2022). Our audit (Appendix E) bounds English contamination at or below 0.1% for the nine main-analysis corpora, too little to account for the observed alignment, but residual contamination remains a small unquantified contributor, particularly for the noisier low-resource corpora.

Parallel-data assumption. Rotations are fitted on parallel sentence pairs. Although this is small in absolute terms, sentence-aligned data is still not free for low-resource languages, and we do not evaluate unsupervised mapping alternatives.

Scope of the causal evaluation. Cross-model activation patching covers seven factual relations at a single concept-bearing span, with English– German for the relation sweep (Table 10) and country→capital for the five-target language sweep (Figure 6). Scoring is restricted to the answer's first subword. The evaluation establishes that the fitted rotation carries functional content across tasks and relations; multi-token generation and open-ended use fall outside this design.

## Ethical Considerations

All models and corpora used in this paper are publicly released research artefacts, and we release no new model or corpus. The factual probes we author cover encyclopaedic relations over public entities and contain no personal data. The modular use we sketch in the Discussion, stitching or merging independently trained monolingual models, would also carry each component model's biases and failure modes into the combined system, so any such system needs its own evaluation.

## Acknowledgements

We acknowledge the support of the UKRI Frontier Research Grant EP/Y031350/1 (EQUATE). Suchir Salhan is funded by Cambridge University Press & Assessment. Some experiments were performed using resources provided by the Cambridge Service for Data Driven Discovery (CSD3) operated by the University of Cambridge Research Computing Service, provided by Dell EMC and Intel using Tier-2 funding from the Engineering and Physical Sciences Research Council (capital grant EP/T022159/1), and DiRAC funding from the Science and Technology Facilities Council.

## Use of AI assistants.

We use AI coding assistants for implementation support. All experimental design, analyses, and claims are the authors’ own.

## References

Pierre Andrews, Mikel Artetxe, Mariano Coria Meglioli, Marta R. Costa-jussà, Joe Chuang, David Dale, Mark Duppenthaler, Nathanial Paul Ekberg, Cynthia Gao, Daniel Edward Licht, Jean Maillard, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Eduardo Sánchez, Ioannis Tsiamas, Arina Turkatenko, Albert Ventayol-Boada, and Shireen Yates. 2025. BOUQuET : dataset, benchmark and open initiative for universal quality evaluation in translation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 27515–27535, Suzhou, China. Association for Computational Linguistics.

Catherine Arnett, Tyler A. Chang, James A. Michaelov, and Ben Bergen. 2025. On the acquisition of shared grammatical representations in bilingual language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20707–20726, Vienna, Austria. Association for Computational Linguistics.

Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018. A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 789–798, Melbourne, Australia. Association for Computational Linguistics.

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4623–4637, Online. Association for Computational Linguistics.

Yamini Bansal, Preetum Nakkiran, and Boaz Barak. 2021. Revisiting model stitching to compare neural representations. In Advances in Neural Information Processing Systems.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, Usvsn Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 2397–2430. PMLR.

Terra Blevins and Luke Zettlemoyer. 2022. Language contamination helps explains the cross-lingual capabilities of English pretrained models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3563–3574, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jannik Brinkmann, Chris Wendler, Christian Bartelt, and Aaron Mueller. 2025. Large language models share representations of latent grammatical concepts across typologically diverse languages. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6131–6150, Albuquerque, New Mexico. Association for Computational Linguistics.

Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2024. When is multilinguality a curse? language modeling for 250 high- and low-resource languages. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4074–4096, Miami, Florida, USA. Association for Computational Linguistics.

Tyler A. Chang, Catherine Arnett, Zhuowen Tu, and Benjamin K. Bergen. 2026. Goldfish: Monolingual language models for 350 languages. arXiv preprint arXiv:2408.10441.

Tyler A. Chang, Zhuowen Tu, and Benjamin K. Bergen. 2022. The geometry of multilingual language model representations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 119–136, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020a. Unsupervised

cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440— 8451, Online. Association for Computational Linguistics.

Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2020b. Emerging cross-lingual structure in pretrained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6022–6034, Online. Association for Computational Linguistics.

Nicholas Kluge Corrêa, Aniket Sen, Sophia Falk, and Shiza Fatimah. 2025. Tucano: Advancing Neural Text Generation for Portuguese. Patterns, 6(11).

MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky. 2023. Reliability of CKA as a similarity measure in deep learning. In The Eleventh International Conference on Learning Representations.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Philipp Dufter and Hinrich Schütze. 2020. Identifying elements essential for BERT's multilinguality. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4423–4437, Online. Association for Computational Linguistics.

Clément Dumas, Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2025. Separating tongue from thought: Activation patching reveals language-agnostic concept representations in transformers. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31822–31841, Vienna, Austria. Association for Computational Linguistics.

Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Schölkopf. 2005. Measuring statistical dependence with hilbert-schmidt norms. In International conference on algorithmic learning theory, pages 63–77. Springer.

Fabian Gröger, Shuo Wen, and Maria Brbić. 2026. Revisiting the platonic representation hypothesis: An aristotelian view. Preprint, arXiv:2602.14486.

Abir Harrasse, Florent Draye, Punya Syon Pandey, Zhijing Jin, and Bernhard Schölkopf. 2026. Tracing multilingual representations in llms with cross-layer transcoders. Preprint, arXiv:2511.10840.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. 2024. Position: The platonic representation hypothesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20617–20642. PMLR.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations.

Masoud Jalili Sabet, Philipp Dufter, François Yvon, and Hinrich Schütze. 2020. SimAlign: High quality word alignments without parallel training data using static and contextualized embeddings. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1627–1643, Online. Association for Computational Linguistics.

Rishi Jha, Collin Zhang, Vitaly Shmatikov, and John Morris. 2025. Harnessing the universal geometry of embeddings. Advances in Neural Information Processing Systems, 38:45963–45987.

Karthikeyan K, Zihan Wang, Stephen Mayhew, and Dan Roth. 2020. Cross-lingual ability of multilingual BERT: An empirical study. In International Conference on Learning Representations.

Amir Hossein Kargaran, Ayyoob Imani, François Yvon, and Hinrich Schuetze. 2023. GlotLID: Language identification for low-resource languages. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6155–6218, Singapore. Association for Computational Linguistics.

A Koepke, Daniil Zverev, Shiry Ginosar, and Alexei A Efros. 2026. Back into plato's cave: Examining cross-modal representational convergence at scale. arXiv preprint arXiv:2604.18572.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3519–3529. PMLR.

Felicia Körner, Max Müller-Eberstein, Anna Korhonen, and Barbara Plank. 2026. When meanings meet: Investigating the emergence and quality of shared concept spaces during multilingual language model training. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3149–3169, Rabat, Morocco. Association for Computational Linguistics.

Guillaume Lample, Alexis Conneau, Marc'Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In International Conference on Learning Representations.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, and 1 others. 2022. BLOOM: A 176B-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Karel Lenc and Andrea Vedaldi. 2015. Understanding image representations by measuring their equivariance and equivalence. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 991–999.

Patrick Littell, David R. Mortensen, Ke Lin, Katherine Kairis, Carlisle Turner, and Lori Levin. 2017. URIEL and lang2vec: Representing languages as typological geographical, and phylogenetic vectors. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers, pages 8–14, Valencia, Spain. Association for Computational Linguistics.

Yikang Liu, Yeting Shen, Hongao Zhu, Lilong Xu, Zhiheng Qian, Siyuan Song, Kejia Zhang, Jialong Tang, Pei Zhang, Baosong Yang, Rui Wang, and Hai Hu. 2025. A systematic assessment of language models with linguistic minimal pairs in chinese. Preprint, arXiv:2411.06096.

Tomas Mikolov, Quoc V. Le, and Ilya Sutskever. 2013. Exploiting similarities among languages for machine translation. Preprint, arXiv:1309.4168.

Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, and Emanuele Rodolà. 2023. Relative representations enable zeroshot latent space communication. In The Eleventh International Conference on Learning Representations.

Niklas Muennighoff. 2022. SGPT: GPT Sentence Embeddings for Semantic Search. Preprint, arXiv:2202.08904.

Benjamin Muller, Yanai Elazar, Benoît Sagot, and Djamé Seddah. 2021. First align, then predict: Understanding the cross-lingual ability of multilingual BERT. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2214–2231, Online. Association for Computational Linguistics.

Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, and Adrian Gwoździej. 2025. Bielik v3 small: Technical report. Preprint, arXiv:2505.02550.

Riccardo Orlando, Luca Moroni, Pere-Lluís Huguet Cabot, Simone Conia, Edoardo Barba, Sergio Orlandini, Giuseppe Fiameni, and Roberto Navigli. 2024. Minerva LLMs: The first family of large language models trained from scratch on Italian data. In Proceedings of the Tenth Italian Conference on Computational Linguistics (CLiC-it 2024), pages 707–719, Pisa, Italy. CEUR Workshop Proceedings.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001, Florence, Italy. Association for Computational Linguistics.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Frederick Riemenschneider and Anette Frank. 2025. Cross-lingual generalization and compression: From language-specific to shared neurons. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13470–13491, Vienna, Austria. Association for Computational Linguistics.

Peter H Schönemann. 1966. A generalized solution of the orthogonal Procrustes problem. Psychometrika, 31(1):1–10.

Samuel L. Smith, David H. P. Turban, Steven Hamblin, and Nils Y. Hammerla. 2017. Offline bilingual word vectors, orthogonal transformations and the inverted softmax. In International Conference on Learning Representations.

Pedro Ortiz Suarez, Laurie Burchell, Catherine Arnett, Rafael Mosquera, Sara Hincapié Monsalve, Thom Vaughan, Damian Stewart, Malte Ostendorff, Idris Abdulmumin, Vukosi Marivate, Shamsuddeen Hassan Muhammad, Atnafu Lambebo Tonja, Hend Al-Khalifa, Nadia Ghezaiel Hammouda, Verrah Akinyi Otiende, Tack Hwa Wong, Jakhongir Saydaliev, Melika Nobakhtian, Muhammad Ravi Shulthan Habibi, and 78 others. 2026. CommonLID: Re-evaluating state-of-the-art language identification performance on web data. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 33063–33080, San Diego, California, United States. Association for Computational Linguistics.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, and 179 others. 2024. Gemma 2: Improving open language models at a practical size. Preprint, arXiv:2408.00118.

NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault Gabriel Mejia Gonzalez, Prangthip Hansanti, and

20 others. 2022. No language left behind: Scaling human-centered machine translation.

Jörg Tiedemann. 2020. The Tatoeba Translation Challenge – Realistic data sets for low resource and multilingual MT. In Proceedings of the Fifth Conference on Machine Translation, pages 1174–1182, Online. Association for Computational Linguistics.

Jörg Tiedemann. 2012. Parallel data, tools and interfaces in OPUS. In Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC'12), Istanbul, Turkey. European Language Resources Association (ELRA).

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in English? on the latent language of multilingual transformers. In Proceedings of ACL.

Chao Xing, Dong Wang, Chao Liu, and Yiye Lin. 2015. Normalized word embedding and orthogonal transform for bilingual word translation. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1006–1011, Denver, Colorado. Association for Computational Linguistics.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, and Mohit Bansal. 2023. TIES-merging: Resolving interference when merging models. In Proceedings of NeurIPS.

Ruochen Zhang, Qinan Yu, Matianyu Zang, Carsten Eickhoff, and Ellie Pavlick. 2025. The same but different: Structural similarities and differences in multilingual language modeling. In International Conference on Learning Representations, volume 2025, pages 95372–95392.

Ej Zhou, Caiqi Zhang, Tiancheng Hu, Chengzu Li, Nigel Collier, Ivan Vulić, and Anna Korhonen. 2025. Beyond the final layer: Intermediate representations for better multilingual calibration in large language models. Preprint, arXiv:2510.03136.

## A Full Pairwise CKA Results

Tables 11-22 (Appendix H) report the complete pairwise last-layer linear CKA values for all 92 = 36 language pairs across nine monolingual 1000 MB Goldfish models. Each cell shows matched/shuffled CKA, where matched denotes CKA computed on parallel (translation-equivalent) sentences and shuffled denotes a control in which the sentence order of one language is randomly permuted. Tables are grouped by pooling method: sentence-level mean pooling (Tables 11–14), tokenlevel word-aligned (Tables 15-18), and SGPT position-weighted pooling (Tables 19–22), each evaluated on FLORES, Tatoeba, OPUS, and Bou-QUET. Missing entries (—) indicate language pairs for which parallel data was unavailable in that dataset. Results for English paired with four lowresource languages (Tagalog, Swahili, Northern Uzbek, Amharic) are reported in Table 2 (Section 3.2).

## B CKA Comparison with Training Data

This appendix supports Section 3.3 with the full $9 \times 9$ pairwise CKA matrices at each of the four Goldfish training-data scales (5 MB, 10 MB, 100 MB, 1000 MB), and with a cross-size comparison that pairs models trained on different amounts of data. All analyses use last-layer pairwise linear CKA across the same nine monolingual Goldfish languages from Appendix A, with a shuffled control throughout.

## B.1 Does Alignment Grow with Training Data?

Figures 14–16 (Appendix H) show the full $9 \times 9$ pairwise CKA matrices at each of the four data scales, for all three CKA settings: sentence-level mean pooling, token-level word-aligned, and SGPT position-weighted pooling.

Findings. Across all three CKA settings, matched off-diagonal CKA increases monotonically from 5 MB to 1000 MB. Shuffled baselines remain near zero at every scale, so the matched—shuffled gap widens with more training data. This confirms and extends the size-scaling result from Section 3.3: increased data exposure yields more semantically structured internal representations, consistently across all pooling strategies.

## B.2 Cross-Size Comparison: Do Models of Different Scales Align?

Figures 7–9 show cross-size CKA matrices, where Model A and Model B are trained on different amounts of data. We consider two cross-size pairings: 1000 MB vs. 100 MB, and 10 MB vs. 5 MB.

![](images/00cb68c28c0b4fb34235153e61a1a7469e32d4370cb9cfcf3439500124baf58e.jpg)  
Figure 7: Sentence-Level Mean Pooling: cross-size CKA at the last layer. Left pair: 1000 MB models (rows) vs. 100 MB models (columns). Right pair: 10 MB vs. 5 MB. Within each pair, matched and shuffled are shown separately. Diagonal entries correspond to the same language compared across sizes; off-diagonal entries correspond to different languages at different scales.

Findings. Two consistent patterns emerge across all three settings.

(i) Same-language diagonal dominates. When comparing, say, the English 1000 MB model against the English 100 MB model, the diagonal CKA is substantially higher than any off-diagonal cell in the same row.

(ii) Matched signal survives across sizes. In the 1000 MB vs. 100 MB comparison, off-diagonal matched CKA values remain clearly above the shuffled baseline, indicating that independently trained monolingual models at different scales still encode partially compatible concept representations. The 10 MB vs. 5 MB comparison shows lower absolute values (consistent with the within-size scaling result), but the matched-vs-shuffled gap persists.

## C Cross-Architecture Comparison

This appendix supports Section 3.4 with the full model specifications for both cross-architecture ex-

![](images/7444aa4ef8340674a29895120454334d4b508d3cc40268c68bf4bd771f4956df.jpg)  
Figure 8: Token-Level Word-Aligned: cross-size CKA. Same layout as Figure 7.

periments, and a first-layer analysis that complements the last-layer matrix in the main text.

## C.1 Goldfish vs. Pythia

We compare Goldfish against Pythia-160M (Biderman et al., 2023) (for English) and Zh-Pythia-160M (Liu et al., 2025) (for Chinese), which share Goldfish's hidden dimension and depth but differ in tokenizer, training data, and training pipeline. Figure 10 reports per-layer SGPT CKA. Two observations stand out: matched CKA exceeds the shuffled baseline at every layer for both English and Chinese, and the layerwise profile is U-shaped, with alignment high at embedding and final layers but depressed in the middle. This suggests that independently trained architectures converge on similar input- and output-facing semantic representations, but can diverge in their intermediate processing strategies. Table 5 lists the four models used.

<table><tr><td>Model</td><td>Lang</td><td>Vocab</td></tr><tr><td>Goldfish 125M</td><td>EN</td><td>51 200</td></tr><tr><td>Pythia-160M</td><td>EN</td><td>50304</td></tr><tr><td>Goldfish 125M</td><td>ZH</td><td>51 200</td></tr><tr><td>Zh-Pythia-160M</td><td>ZH</td><td>50304</td></tr></table>

Table 5: Models used in the Goldfish vs. Pythia comparison of Section 3.4. All four models share hidden dimension 768, 12 layers, and 12 attention heads; they differ in vocabulary size (shown), tokenizer, training data, and training procedure.

![](images/10969cc7cbc937b2dc52f003366cf2e44b600c7318c7f321f7bdcddaf90d2af8.jpg)

Figure 9: SGPT Position-Weighted Pooling: cross-size CKA. Same layout as Figure 7.  
![](images/bdcdb80ca65c26af8659110c63004bb13889ef6631f580ab24bc733695a08fef.jpg)  
Figure 10: Goldfish vs. Pythia per-layer CKA (SGPT). Left: English (Goldfish-en vs. Pythia-160M). Right: Chinese (Goldfish-zh vs. Zh-Pythia-160M). Red: matched; blue: shuffled.

## C.2 Diverse Independently-Developed Monolingual Models

Section 3.4 (Table 3) reports last-layer SGPT CKA across five independently developed \~1B monolingual models that share no architecture, tokenizer, training data, or research group. Here we provide the full model specifications and a first-layer comparison that complements the last-layer matrix in the main text.

Models. Table 6 lists the five models. They span three architecture families (GPT-NeoX, LLaMA, Qwen2.5/Mistral), five languages, and five research groups, and differ in hidden dimension, depth, attention head count, and vocabulary.

Setup. We compute SGPT-pooled linear CKA for all shuffled controls. Because these models differ in depth, per-layer trajectories are not directly comparable; we therefore focus on the final pre-logit (last) layer, which represent the input and output boundaries of each model. Last-layer values are reported in Table 3 of the main text.

<table><tr><td>Model</td><td>Lang</td><td>Arch</td><td>Hidden</td><td>Layers</td><td>Heads</td></tr><tr><td>Pythia-1.4B</td><td>EN</td><td>GPT-NeoX</td><td>2048</td><td>24</td><td>16</td></tr><tr><td>Zh-Pythia-1.4B</td><td>ZH</td><td>GPT-NeoX</td><td>2048</td><td>24</td><td>16</td></tr><tr><td>Tucano-1B</td><td>PT</td><td>LLaMA</td><td>2048</td><td>22</td><td>32</td></tr><tr><td>Bielik-1.5B</td><td>PL</td><td>Qwen2.5</td><td>1536</td><td>28</td><td>12</td></tr><tr><td>Minerva-1B</td><td>IT</td><td>Mistral</td><td>2048</td><td>16</td><td>16</td></tr></table>

Table 6: Independently developed monolingual models used in the shared-nothing comparison of Section 3.4.

## D Predictors of Alignment: Additional Details

For each of the 36 Goldfish language pairs, we correlate last-layer SGPT matched CKA against five pairwise predictors: four URIEL typological distances (Littell et al., 2017) (syntactic, genetic, geographic, inventory) and the mean per-sentence negative log-likelihood (NLL) of the two models on FLORES, which proxies how well each model has learned its language. URIEL distances are taken directly from the URIEL typological database: syntactic distance over morphosyntactic feature vectors, genetic distance over the language family tree, geographic distance as great-circle distance between language centroids, and inventory distance over phonological inventories. For each predictor we compute Pearson and Spearman correlations against last-layer SGPT CKA (matched condition).

<table><tr><td>Predictor</td><td>r</td><td> $p _ { r }$ </td><td> $\rho$ </td><td> $p _ { \rho }$ </td></tr><tr><td>URIEL Syntactic</td><td>-0.603</td><td>&lt; .001</td><td>-0.635</td><td>&lt;.001</td></tr><tr><td>URIEL Genetic</td><td>-0.454</td><td>.005</td><td>-0.371</td><td>.026</td></tr><tr><td>Mean NLL</td><td>-0.440</td><td>.007</td><td>-0.425</td><td>.010</td></tr><tr><td>URIEL Geographic</td><td>-0.387</td><td>.020</td><td>-0.440</td><td>.007</td></tr><tr><td>URIEL Inventory</td><td>-0.209</td><td>.222</td><td>-0.123</td><td>.477</td></tr></table>

Table 7: Predictors of last-layer matched CKA across 36 Goldfish language pairs. r: Pearson; $\rho \colon$ Spearman. $p _ { r } , p _ { \rho }$ are two-sided.

Syntactic distance is the strongest single predictor of alignment $( r ~ = ~ - 0 . 6 0 , ~ \rho ~ = ~ - 0 . 6 4$ $p \ < \ . 0 0 1 )$ , followed by genetic distance, mean NLL, and geographic distance; phonological inventory distance is not significantly correlated. Figure 11 visualises the syntactic relationship directly: same-subfamily pairs cluster in the low-distance, high-CKA quadrant, while cross-family pairs drift toward the opposite corner. The four URIEL distances are themselves intercorrelated, and we do not attempt to dissociate their contributions; the ranking is reported as a descriptive summary of where alignment is strongest. The negative coefficient on mean NLL is consistent with a model-fit effect: pairs whose models have learned their language less well also show weaker pairwise alignment.

![](images/f411df8e4132561b83a5bc75b9da5c2c58288e9cc571017c191a9a38be1193c8.jpg)  
Figure 11: Last-layer matched CKA vs. URIEL syntactic distance (SGPT) across 36 Goldfish pairs. Points are coloured by genealogical relationship: same subfamily (blue), both Indo-European (orange), cross-family (grey). Dashed line: linear fit with 95% confidence band.

## E Training-Corpus Language Contamination Audit

Cross-lingual ability in nominally monolingual models can originate from foreign-language text in the training corpus (Blevins and Zettlemoyer, 2022). We therefore audit the Goldfish training corpora used in this paper. For each language we sample 5,000 lines from the public training corpus and classify each line with GlotLID-v3 (Kargaran et al., 2023). Table 8 reports the fraction of lines labelled as the corpus language, the fraction confidently labelled as any other language (classifier confidence ≥ 0.9), and the fraction labelled English.

For the nine languages of the main analysis, lines confidently identified as English are at most 0.1% (French 0.08%, Hindi 0.10%, all others ≤ 0.02%). At the 1000 MB tier this bounds incidental English exposure to roughly 1 MB of text, below even the 5 MB tier, the smallest at which we measure alignment (Section 3.3). The contamination mechanism of Blevins and Zettlemoyer (2022) requires meaningful high-resource leakage; at these rates it cannot supply the cross-lingual competence observed in Sections 3-5. The four low-resource corpora carry more foreign material (up to 2.14% otherlanguage and 1.54% English for Swahili), consistent with noisier web sources; this is a caveat for Table 2 and is noted in the Limitations. Line-level identification on web data is itself imperfect, particularly for closely related varieties (Suarez et al., 2026), which is visible in the Arabic dialect labelling above.

<table><tr><td>Corpus</td><td>Own (%)</td><td>Other (%)</td><td>English (%)</td></tr><tr><td>eng</td><td>99.7</td><td>0.00</td><td>一</td></tr><tr><td>fra</td><td>99.5</td><td>0.00</td><td>0.08</td></tr><tr><td>rus</td><td>99.6</td><td>0.10</td><td>0.00</td></tr><tr><td>spa</td><td>99.2</td><td>0.02</td><td>0.00</td></tr><tr><td>deu</td><td>97.4</td><td>0.10</td><td>0.02</td></tr><tr><td>jpn</td><td>98.3</td><td>0.06</td><td>0.00</td></tr><tr><td>zho</td><td>97.9</td><td>0.40</td><td>0.00</td></tr><tr><td>hin</td><td>96.3</td><td>0.82</td><td>0.10</td></tr><tr><td>arb</td><td>84.4*</td><td>1.98</td><td>0.00</td></tr><tr><td>swh</td><td>92.1</td><td>2.14</td><td>1.54</td></tr><tr><td>tgl</td><td>95.3</td><td>1.16</td><td>0.76</td></tr><tr><td>amh</td><td>93.5</td><td>4.64</td><td>0.58</td></tr><tr><td>uzn</td><td>97.3</td><td>1.00</td><td>0.18</td></tr></table>

Table 8: GlotLID-v3 audit of the Goldfish training corpora, 5,000 sampled lines per language. Own: lines labelled as the corpus language. Other: lines labelled as any other language with confidence $\geq 0 . 9$ . English: lines labelled English. \*The Arabic residual is almost entirely Standard Arabic assigned Arabic dialect labels (ary, arz, ars).

## F Cross-Lingual Reconstruction: Method Comparison

This appendix supports Section 4 (Cross-Lingual Representation Reconstruction) with extended results comparing the three projection families (Procrustes, Affine with ridge, MLP) on all 36 language pairs at the final layer.

Per-pair comparison. Figure 12 plots Procrustes against Affine on retrieval P@1 (left) and test MSE (right) for each of the 36 language pairs at layer 12. Procrustes wins on P@ 1 for every pair (paired mean $\Delta = + 0 . 1 1 2$ , 95% CI $[ + 0 . 0 8 0 , + 0 . 1 4 3 ] \rangle$ while Affine wins on MSE for every pair. Together with Table 23, this dissociation shows that the metric on which a method is “best" depends on the metric being optimised: Affine attains lower MSE because its hypothesis class is strictly larger, but the orthogonality constraint of Procrustes preserves angular structure that is better for cosine retrieval.

![](images/77ce1c9b47596bfe333dfd885fa1adbf359f690766f30819e8736d1a4fbc251a.jpg)

![](images/71b434caf793ef2a793f7719727c03c6b07e81d3ada4702ac2a1cb99957110a1.jpg)  
Figure 12: Procrustes vs. Affine, all 36 pairs (layer 12). Each point is one language pair. Left: retrieval P@1 against the train+test target pool; right: test MSE. Dashed line is $y = x .$ Annotations report the paired mean difference $\Delta$ with bootstrap 95% CI (10,000 resamples).

Aggregate summary. Table 23 (Appendix H) reports mean P@1, MSE, and CKA across the 36 pairs with bootstrap 95% CIs, and the number of pairs on which each method achieves the best score for each metric.

Methods × metrics. Figure 17 (Appendix H) shows the full method×metric grid: each panel is one (method, metric) combination across all 36 pairs and 13 layers. This confirms that the ranking observed at layer 12 is stable across most layers, and that MLP and Affine track each other closely, but Procrustes consistently dominates on retrieval.

Visualising the rotation. Figure 18 (Appendix H) provides a qualitative view of what Procrustes is doing. We project the English source representations and the matched French target representations into the same 2D PCA basis and overlay the Procrustes-mapped English points. This illustrates that a single rotation of the English space lands each sentence near its French counterpart.

## F.1 Layer-wise Analysis

We restrict to Procrustes and sweep layers 0–12 for all 36 pairs. Figure 13 reports retrieval P@1 by layer, with four representative pairs highlighted: eng→fra (typologically close), eng→rus (mid), eng→zho and eng→arb (distant). We see that retrieval is non-monotonic in depth and peaks at layers 6–8 across pairs (eng→fra: layer 8, P@ 1 =0.981; eng→rus: layer 6, 0.987; eng→zho: layer 8, 0.977; eng→arb: layer 8, 0.922), with the location of the peak weakly tracking typological distance. The mid-layer peak is consistent with evidence that intermediate layers carry a more reliable multilingual signal than the final layer (Zhou et al., 2025).

![](images/0b1676413f4b144af726c95f2425320fb881e034c9dc43baf4f11d0276ea2696.jpg)  
Figure 13: Procrustes P@1 by layer (train+test pool). Headline pairs (eng→{fra, rus, zho, arb}) bolded; remaining 32 pairs faint in background. Dots mark the per-pair best layer.

## F.2 Dimensional Analysis

Procrustes leaves a residual. A natural follow-up question is what that residual looks like, because the answer determines whether a more flexible map could have closed the gap. Two possibilities could happen: Either the residual error is concentrated along a small number of directions, in which case the two spaces are nearly isometric except along those axes and a low-rank correction would suffice; or the error is spread across many directions, in which case no simple relaxation will help. We distinguish the two by taking an SVD of the residual and comparing its spectrum to that of the target.

Setup. Let X, $Y \in \mathbb { R } ^ { n \times d }$ be the per-layer source and target representations and $W ^ { \star }$ the Procrustes solution from Section 4.1. The residual is R = $X W ^ { \star } - Y$ . We take its SVD, $R = U _ { R } \Sigma _ { R } V _ { R } ^ { \top }$ and the target's, $Y = U _ { Y } \Sigma _ { Y } V _ { Y } ^ { \top }$ , and compute two kinds of summary.

The first asks how spread-out a spectrum is. The entropy-based effective rank

$$
\operatorname { e f f } _ { - } \operatorname { r a n k } ( M ) = \exp \Bigl ( - \sum _ { i } \tilde { \sigma } _ { i } \log \tilde { \sigma } _ { i } \Bigr ) , \quad \tilde { \sigma } _ { i } = \frac { \sigma _ { i } ^ { 2 } } { \sum _ { j } \sigma _ { j } ^ { 2 } }
$$

counts the directions a matrix effectively occupies (a perfectly rank-1 matrix has eff\_rank 1; a uniform spectrum across d directions has eff\_rank d). We complement it with $k _ { \alpha }$ , the smallest k whose top singular values carry an α fraction of the squared norm.

The second asks where the residual sits relative to the target. We project R onto the top-k rightsingular vectors of $Y$ , denoted $V _ { Y } ^ { ( k ) }$ , and report

$$
\mathrm { f r a c \_ o u t } _ { k } = 1 - { \frac { \| R V _ { Y } ^ { ( k ) } \| _ { F } ^ { 2 } } { \| R \| _ { F } ^ { 2 } } } ,
$$

the fraction of residual energy lying outside the topk directions of Y. A small value means the residual lives in the same subspace the target uses; a large value means it lives off to the side, in directions the target itself barely populates.

<table><tr><td colspan="3">Eff. rank</td><td colspan="2">Resid.  $k _ { \alpha }$ </td><td colspan="2">frac  $\mathrm { \Gamma } _ { - } \mathrm { o u t } _ { k }$ </td></tr><tr><td>Pair</td><td>Y</td><td>R</td><td>50%</td><td>90%</td><td> $k { = } 1 0$ </td><td>k=50</td></tr><tr><td>eng-fra</td><td>67</td><td>246</td><td>90</td><td>357</td><td>.93</td><td>.73</td></tr><tr><td>eng-rus</td><td>45</td><td>202</td><td>74</td><td>273</td><td>.93</td><td>.73</td></tr><tr><td>eng-zho</td><td>43</td><td>180</td><td>69</td><td>265</td><td>.91</td><td>.69</td></tr><tr><td>eng-arb</td><td>53</td><td>147</td><td>71</td><td>306</td><td>.88</td><td>.68</td></tr></table>

Table 9: Spectra of Y and the post-Procrustes residual R at layer 8 (d=768). Columns: target and residual effective rank; smallest k for which the top-k singular values of R capture 50%/90% of its squared norm; fraction of residual energy outside the top-k right-singular subspace of Y.

Findings. Table 9 reports these statistics for English paired with French, Russian, Chinese, and Arabic at the peak retrieval layer (layer 8). Three patterns hold across pairs and, with mild variation, across all 13 layers.

(i) The target is low-rank; the residual is not. The target representations use only 43–67 directions out of 768, and 90% of their variance fits in roughly 160–200 directions. The residual is far more diffuse: effective rank 147–246, with half its energy already requiring 70–90 singular values and 90% requiring 265–360. The error is therefore not stored in a small handful of bad axes that a targeted fix could correct; it is spread thinly across many directions.

(ii) The residual lives outside the part of space the target uses. Of the residual's total energy, 88— 93% lies outside the top-10 directions of $Y ,$ and 68–74% outside the top-50. The Procrustes rotation lands cleanly in the subspace where the target stores its variance and leaves error in the directions the target barely populates. This is true even though the residual is large in absolute terms (Frobenius norm 0.79–0.93 of the target's at layer 8): a residual can be large in raw magnitude while being functionally irrelevant if it sits where the model does not read.

(iii) The pattern is stable across layers. The same picture holds throughout the network. Residual effective rank exceeds target effective rank by a factor of at least two (typically three to six). frac\_out10 stays above 0.79, and frac\_out50 above 0.57, at every layer for every headline pair. No layer admits a low-dimensional collapse of the residual.

These observations reconcile a tension in the main results. Procrustes leaves a residual that is large in Frobenius norm yet small in functional consequence, because the residual is high-rank and largely orthogonal to the subspace where the target stores its variance. There is no concentrated lowdimensional structure for an Affine relaxation or a one-layer MLP to fit, which explains why neither improves retrieval over Procrustes. What the data support is approximate isometry along the dominant directions of the target, with diffuse mismatch in the remaining ones.

## G Span Patching: Setup, Conditions, and Auxiliary Numbers

This appendix supports Section 5 with extended specification of the factual probes, the conditions, and the per-pair numbers underlying Figure 6.

Facts and prompts. The probe consists of 30 country→capital facts, each available as a complete (prompt, answer) pair in six languages: English, French, German, Spanish, Japanese, and Chinese. Prompts share a fixed cloze form per language (e.g. "The capital of X is", "La capitale de X est", "Die Hauptstadt von $X \mathrm { i } s \mathrm { t } ^ { \prime \prime } , \mathsf { \Omega } ^ { \ast } \mathsf { L } a$ capital de $X ~ \in { \mathfrak { s } } ^ { \prime \prime } , { } ^ { \ast } X { \mathcal { O } } )$ 首都は”,“X的首 都是"). Answers are the target language's name for the capital, scored using each language's first subword id, computed in context so that BPE splits are handled correctly per tokenizer.

Span vs. single-layer patching. A preliminary experiment patched the residual at a single layer at the prompt's last-token position (eng→fra at the position immediately before the LM head). Withinmodel $\Delta \log p ( { \mathrm { d o n o r } } )$ was \~+0.07 nat, which doesn't seem to work well. This is becuase the intervention is too narrow: a single-layer last-token patch is immediately consumed by the LM head and downstream layers, with no opportunity to propagate. The span variant fixes this by (i) patching from j to L, so each layer sees the patched residual at its input; and (ii) patching at the country (concept) position rather than the last token, so the patched information must move through attention to reach the answer logits.

Conditions. Four conditions per cross-fact cell are reported in the main results:

• within-lang: target model, target-language donor prompt, identity projection. Withinmodel ceiling.

• Procrustes: source model on sourcelanguage donor prompt; per-layer orthogonal map $W ^ { ( k ) }$ from Section 4, fit on FLO-RES/OPUS/BouQUET sentence-end activations.

• unprojected: source-model residual at $^ { c _ { S } , }$ identity projection. Controls for basis mismatch.

• shuffle: Gaussian sample $\cdot \mathcal { N } ( \pmb { \mu } _ { S } ^ { ( k ) } , \pmb { \Sigma } _ { S } ^ { ( k ) } )$ per layer with full-covariance Cholesky factorisation (eigendecomposition fallback for rank-deficient covariances), where $( \mu _ { S } ^ { ( k ) } , \Sigma _ { S } ^ { ( k ) } )$ are the moments of the sourcemodel activation pool. Distributional control.

A fifth condition, Affine (the per-layer linear+bias projection from Section 4, no orthogonality constraint), was evaluated and is discussed below; we exclude it from the main results for reasons later in this section.

Settings. Goldfish 1000 MB models per language; 12 Transformer blocks; hidden dim 768; perlanguage vocabulary; max sequence length 512; seed 42. Span starts $j \in \{ 1 , \ldots , 8 \}$ ; span end fixed at $L = 1 2$ (final block). Cross-fact pairs per (condition, j) cell: $3 0 ^ { 2 } - 3 0 = 8 7 0$ (excluding the diagonal where donor = target). The country position is identified per language by a verb marker preceding the noun phrase; the country's last subword is the token immediately before this marker. All cells run on a single GPU.

Multi-relation sweep. The relation sweep of Table 10 uses the same span intervention between the English and German 1000 MB models, with the patch start fixed at j=2 and the Section 4 Procrustes maps applied without refitting. Cross-fact pairs combine each donor fact with a different target fact from the same relation, excluding the diagonal where donor equals target; n varies with the facts available per relation. Several prompt-side and answer-side entities are multi-token under one or both tokenizers (e.g. Shakespeare and Gutenberg are three subwords in the German tokenizer; work titles such as Die Zauberflöte span several words); all conditions score ∆ log p at the answer's first subword, computed in context per tokenizer.

Why Affine is excluded. We also evaluated the Section 4 Affine projection on every eng→X pair Affine produces a magnitude artefact rather than directional transfer. Inspection of $\Delta _ { d } - \Delta _ { t } ~ =$ ∆ log p(donor capital) — ∆ log p(target capital), a stricter directional metric that requires the donor capital to rise more than the target capital, is negative for Affine: it inflates the magnitude of the patched residual enough that the LM head boosts both capitals, with the target capital often boosted more. The sign of ∆ log p(donor capital) on Affine therefore reflects a uniform bias shift rather than functional transfer of the donor concept, so we report Procrustes as the cross-lingual projection of record and omit Affine from Figure 6.

## H Supplementary Tables and Figures

This appendix collects the wide tables and figures referenced from Appendices A, B, and F.

![](images/4fb9df9231d1ea9370558fb99fb90043778c14e997615a7b8e642ebfbcc29c60.jpg)  
Figure 14: Setting 1 — Sentence-Level Mean Pooling: last-layer pairwise CKA across all nine Goldfish languages at four training-data scales (5 MB, 10 MB, 100 MB, 1000 MB). Top row: matched CKA. Bottom row: shuffled baseline. Off-diagonal matched values rise monotonically with scale, while shuffled values remain near zero throughout.

![](images/3faefcaafd229514ffd59a76d65799b17af88eb61380b890e77df0b72db1f861.jpg)  
Figure 15: Setting 2 — Token-Level Word-Aligned: same layout as Figure 14. Token-level alignment is lower in absolute terms than sentence-level but shows the same monotonic scaling with training data.

<table><tr><td>Relation</td><td>n</td><td>Within-lang</td><td>Procrustes</td><td>Shuffled</td></tr><tr><td>country→capital</td><td>870</td><td>97.6 [96.3, 98.4]</td><td>79.7 [76.9, 82.2]</td><td>48.5 [45.2, 51.8]</td></tr><tr><td>city→country</td><td>870</td><td>96.2 [94.7, 97.3]</td><td>74.7 [71.7, 77.5]</td><td>43.3 [40.1, 46.6]</td></tr><tr><td>country→continent</td><td>398</td><td>87.9 [84.4, 90.8]</td><td>53.3 [48.4, 58.1]</td><td>5.3 [3.5, 7.9]</td></tr><tr><td>landmark→city</td><td>450</td><td>76.7 [72.5, 80.3]</td><td>60.7 [56.1, 65.1]</td><td>38.9 [34.5, 43.5]</td></tr><tr><td>work→author</td><td>132</td><td>54.5 [46.0, 62.8]</td><td>47.0 [38.7, 55.4]</td><td>17.4 [11.9, 24.8]</td></tr><tr><td>work→composer</td><td>56</td><td>75.0 [62.3, 84.5]</td><td>78.6 [66.2, 87.3]</td><td>64.3 [51.2, 75.5]</td></tr><tr><td>invention→inventor</td><td>56</td><td>55.4 [42.4, 67.6]</td><td>69.6 [56.7, 80.1]</td><td>39.3 [27.6, 52.4]</td></tr></table>

Table 10: Directional success rate (%) for English–German span patching across seven factual relations, with Wilson 95% CIs in brackets. Same per-layer Procrustes maps as Figure 6, applied without refitting; patch span starts at j=2; answers scored at their first subword. n: cross-fact pairs per condition.

Table 11: Last-layer pairwise CKA (matched/shuffled) — Sentence-Level Mean Pooling, FLORES dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.80/0.22</td><td>0.84/0.23</td><td>0.80/0.24</td><td>0.78/0.23</td><td>0.85/0.24</td><td>0.82/0.24</td><td>0.85/0.24</td><td>0.81/0.21</td></tr><tr><td>zho</td><td>0.80/0.22</td><td></td><td>0.78/0.21</td><td>0.75/0.23</td><td>0.76/0.22</td><td>0.78/0.22</td><td>0.79/0.23</td><td>0.77/0.23</td><td>0.82/0.20</td></tr><tr><td>spa</td><td>0.84/0.23</td><td>0.78/0.21</td><td></td><td>0.78/0.24</td><td>0.74/0.23</td><td>0.83/0.24</td><td>0.80/0.24</td><td>0.82/0.24</td><td>0.78/0.21</td></tr><tr><td>arb</td><td>0.80/0.24</td><td>0.75/0.23</td><td>0.78/0.24</td><td></td><td>0.73/0.23</td><td>0.78/0.25</td><td>0.78/0.25</td><td>0.78/0.25</td><td>0.75/0.22</td></tr><tr><td>hin</td><td>0.78/0.23</td><td>0.76/0.22</td><td>0.74/0.23</td><td>0.73/0.23</td><td></td><td>0.75/0.24</td><td>0.75/0.24</td><td>0.76/0.24</td><td>0.77/0.21</td></tr><tr><td>fra</td><td>0.85/0.24</td><td>0.78/0.22</td><td>0.83/0.24</td><td>0.78/0.25</td><td>0.75/0.24</td><td></td><td>0.82/0.24</td><td>0.83/0.24</td><td>0.78/0.21</td></tr><tr><td>rus</td><td>0.82/0.24</td><td>0.79/0.23</td><td>0.80/0.24</td><td>0.78/0.25</td><td>0.75/0.24</td><td>0.82/0.24</td><td></td><td>0.82/0.25</td><td>0.79/0.22</td></tr><tr><td>deu</td><td>0.85/0.24</td><td>0.77/0.23</td><td>0.82/0.24</td><td>0.78/0.25</td><td>0.76/0.24</td><td>0.83/0.24</td><td>0.82/0.25</td><td></td><td>0.79/0.21</td></tr><tr><td>jpn</td><td>0.81/0.21</td><td>0.82/0.20</td><td>0.78/0.21</td><td>0.75/0.22</td><td>0.77/0.21</td><td>0.78/0.21</td><td>0.79/0.22</td><td>0.79/0.21</td><td></td></tr></table>

Table 12: Last-layer pairwise CKA (matched/shuffled) — Sentence-Level Mean Pooling, Tatoeba dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.45/0.17</td><td>0.78/0.21</td><td>0.64/0.22</td><td>0.65/0.26</td><td>0.75/0.23</td><td>0.70/0.22</td><td>0.81/0.23</td><td>0.59/0.25</td></tr><tr><td>zho</td><td>0.45/0.17</td><td></td><td>0.60/0.23</td><td>0.51/0.31</td><td>0.50/0.20</td><td>0.57/0.21</td><td>0.53/0.21</td><td>0.63/0.26</td><td>0.43/0.16</td></tr><tr><td>spa</td><td>0.78/0.21</td><td>0.60/0.23</td><td></td><td>0.68/0.24</td><td>0.85/0.61</td><td>0.72/0.23</td><td>0.70/0.23</td><td>0.74/0.24</td><td>0.62/0.27</td></tr><tr><td>arb</td><td>0.64/0.22</td><td>0.51/0.31</td><td>0.68/0.24</td><td></td><td></td><td>0.65/0.26</td><td>0.66/0.25</td><td>0.65/0.25</td><td>0.58/0.26</td></tr><tr><td>hin</td><td>0.65/0.26</td><td>0.50/0.20</td><td>0.85/0.61</td><td></td><td></td><td>0.83/0.46</td><td>0.86/0.66</td><td>0.84/0.58</td><td>0.74/0.33</td></tr><tr><td>fra</td><td>0.75/0.23</td><td>0.57/0.21</td><td>0.72/0.23</td><td>0.65/0.26</td><td>0.83/0.46</td><td></td><td>0.67/0.21</td><td>0.75/0.25</td><td>0.53/0.25</td></tr><tr><td>rus</td><td>0.70/0.22</td><td>0.53/0.21</td><td>0.70/0.23</td><td>0.66/0.25</td><td>0.86/0.66</td><td>0.67/0.21</td><td></td><td>0.73/0.26</td><td>0.65/0.27</td></tr><tr><td>deu</td><td>0.81/0.23</td><td>0.63/0.26</td><td>0.74/0.24</td><td>0.65/0.25</td><td>0.84/0.58</td><td>0.75/0.25</td><td>0.73/0.26</td><td></td><td>0.61/0.26</td></tr><tr><td>jpn</td><td>0.59/0.25</td><td>0.43/0.16</td><td>0.62/0.27</td><td>0.58/0.26</td><td>0.74/0.33</td><td>0.53/0.25</td><td>0.65/0.27</td><td>0.61/0.26</td><td></td></tr></table>

Table 13: Last-layer pairwise CKA (matched/shuffled) — Sentence-Level Mean Pooling, OPUS dataset. All values computed on 1000 MB Goldfish models
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.83/0.09</td><td>0.81/0.17</td><td>0.70/0.20</td><td>0.34/0.09</td><td>0.83/0.16</td><td>0.80/0.16</td><td>0.79/0.18</td><td>0.53/0.27</td></tr><tr><td>zho</td><td>0.83/0.09</td><td></td><td></td><td>0.74/0.13</td><td></td><td>0.70/0.15</td><td>0.73/0.15</td><td>0.61/0.18</td><td></td></tr><tr><td>spa</td><td>0.81/0.17</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>arb</td><td>0.70/0.20</td><td>0.74/0.13</td><td></td><td></td><td></td><td>0.84/0.12</td><td>0.85/0.11</td><td>0.53/0.22</td><td></td></tr><tr><td>hin</td><td>0.34/0.09</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>fra</td><td>0.83/0.16</td><td>0.70/0.15</td><td></td><td>0.84/0.12</td><td></td><td></td><td>0.84/0.12</td><td>0.81/0.15</td><td></td></tr><tr><td>rus</td><td>0.80/0.16</td><td>0.73/0.15</td><td></td><td>0.85/0.11</td><td></td><td>0.84/0.12</td><td></td><td>0.62/0.24</td><td></td></tr><tr><td>deu</td><td>0.79/0.18</td><td>0.61/0.18</td><td></td><td>0.53/0.22</td><td></td><td>0.81/0.15</td><td>0.62/0.24</td><td></td><td></td></tr><tr><td>jpn</td><td>0.53/0.27</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 14: Last-layer pairwise CKA (matched/shuffled) — Sentence-Level Mean Pooling, BouQUET dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.73/0.24</td><td>0.83/0.26</td><td>0.60/0.21</td><td>0.76/0.24</td><td>0.83/0.25</td><td>0.81/0.27</td><td>0.81/0.27</td><td>0.76/0.26</td></tr><tr><td>zho</td><td>0.73/0.24</td><td></td><td>0.74/0.22</td><td>0.55/0.20</td><td>0.69/0.24</td><td>0.73/0.23</td><td>0.75/0.22</td><td>0.73/0.24</td><td>0.75/0.25</td></tr><tr><td>spa</td><td>0.83/0.26</td><td>0.74/0.22</td><td></td><td>0.60/0.20</td><td>0.72/0.22</td><td>0.82/0.25</td><td>0.79/0.26</td><td>0.79/0.24</td><td>0.73/0.25</td></tr><tr><td>arb</td><td>0.60/0.21</td><td>0.55/0.20</td><td>0.60/0.20</td><td></td><td>0.54/0.20</td><td>0.57/0.20</td><td>0.56/0.21</td><td>0.57/0.19</td><td>0.54/0.20</td></tr><tr><td>hin</td><td>0.76/0.24</td><td>0.69/0.24</td><td>0.72/0.22</td><td>0.54/0.20</td><td></td><td>0.73/0.25</td><td>0.76/0.25</td><td>0.71/0.25</td><td>0.69/0.26</td></tr><tr><td>fra</td><td>0.83/0.25</td><td>0.73/0.23</td><td>0.82/0.25</td><td>0.57/0.20</td><td>0.73/0.25</td><td></td><td>0.80/0.24</td><td>0.79/0.25</td><td>0.73/0.25</td></tr><tr><td>rus</td><td>0.81/0.27</td><td>0.75/0.22</td><td>0.79/0.26</td><td>0.56/0.21</td><td>0.76/0.25</td><td>0.80/0.24</td><td></td><td>0.79/0.24</td><td>0.73/0.28</td></tr><tr><td>deu</td><td>0.81/0.27</td><td>0.73/0.24</td><td>0.79/0.24</td><td>0.57/0.19</td><td>0.71/0.25</td><td>0.79/0.25</td><td>0.79/0.24</td><td></td><td>0.75/0.24</td></tr><tr><td>jpn</td><td>0.76/0.26</td><td>0.75/0.25</td><td>0.73/0.25</td><td>0.54/0.20</td><td>0.69/0.26</td><td>0.73/0.25</td><td>0.73/0.28</td><td>0.75/0.24</td><td></td></tr></table>

Table 15: Last-layer pairwise CKA (matched/shuffled) — Token-Level Word-Aligned, FLORES dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.37/0.03</td><td>0.51/0.04</td><td>0.40/0.04</td><td>0.43/0.04</td><td>0.54/0.04</td><td>0.51/0.04</td><td>0.52/0.03</td><td>0.37/0.03</td></tr><tr><td>zho</td><td>0.37/0.03</td><td></td><td>0.32/0.03</td><td>0.33/0.03</td><td>0.37/0.03</td><td>0.32/0.03</td><td>0.36/0.03</td><td>0.31/0.03</td><td>0.43/0.10</td></tr><tr><td>spa</td><td>0.51/0.04</td><td>0.32/0.03</td><td></td><td>0.38/0.04</td><td>0.36/0.04</td><td>0.53/0.03</td><td>0.43/0.04</td><td>0.43/0.04</td><td>0.32/0.03</td></tr><tr><td>arb</td><td>0.40/0.04</td><td>0.33/0.03</td><td>0.38/0.04</td><td></td><td>0.35/0.04</td><td>0.38/0.04</td><td>0.40/0.04</td><td>0.35/0.04</td><td>0.34/0.03</td></tr><tr><td>hin</td><td>0.43/0.04</td><td>0.37/0.03</td><td>0.36/0.04</td><td>0.35/0.04</td><td></td><td>0.38/0.04</td><td>0.38/0.04</td><td>0.42/0.04</td><td>0.37/0.02</td></tr><tr><td>fra</td><td>0.54/0.04</td><td>0.32/0.03</td><td>0.53/0.03</td><td>0.38/0.04</td><td>0.38/0.04</td><td></td><td>0.44/0.04</td><td>0.44/0.03</td><td>0.31/0.03</td></tr><tr><td>rus</td><td>0.51/0.04</td><td>0.36/0.03</td><td>0.43/0.04</td><td>0.40/0.04</td><td>0.38/0.04</td><td>0.44/0.04</td><td></td><td>0.43/0.04</td><td>0.36/0.03</td></tr><tr><td>deu</td><td>0.52/0.03</td><td>0.31/0.03</td><td>0.43/0.04</td><td>0.35/0.04</td><td>0.42/0.04</td><td>0.44/0.03</td><td>0.43/0.04</td><td></td><td>0.32/0.03</td></tr><tr><td>jpn</td><td>0.37/0.03</td><td>0.43/0.10</td><td>0.32/0.03</td><td>0.34/0.03</td><td>0.37/0.02</td><td>0.31/0.03</td><td>0.36/0.03</td><td>0.32/0.03</td><td></td></tr></table>

Table 16: Last-layer pairwise CKA (matched/shuffled) — Token-Level Word-Aligned, Tatoeba dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.17/0.04</td><td>0.51/0.07</td><td>0.38/0.07</td><td>0.39/0.09</td><td>0.51/0.06</td><td>0.51/0.06</td><td>0.50/0.05</td><td>0.30/0.08</td></tr><tr><td>zho</td><td>0.17/0.04</td><td></td><td>0.27/0.07</td><td>0.33/0.12</td><td>0.23/0.05</td><td>0.22/0.04</td><td>0.24/0.06</td><td>0.25/0.06</td><td>0.44/0.19</td></tr><tr><td>spa</td><td>0.51/0.07</td><td>0.27/0.07</td><td></td><td>0.47/0.11</td><td>0.61/0.28</td><td>0.54/0.06</td><td>0.48/0.08</td><td>0.43/0.06</td><td>0.30/0.09</td></tr><tr><td>arb</td><td>0.38/0.07</td><td>0.33/0.12</td><td>0.47/0.11</td><td></td><td></td><td>0.39/0.09</td><td>0.46/0.09</td><td>0.38/0.09</td><td>0.32/0.10</td></tr><tr><td>hin</td><td>0.39/0.09</td><td>0.23/0.05</td><td>0.61/0.28</td><td></td><td></td><td>0.56/0.19</td><td>0.58/0.27</td><td>0.55/0.20</td><td>0.42/0.10</td></tr><tr><td>fra</td><td>0.51/0.06</td><td>0.22/0.04</td><td>0.54/0.06</td><td>0.39/0.09</td><td>0.56/0.19</td><td></td><td>0.52/0.05</td><td>0.46/0.05</td><td>0.26/0.07</td></tr><tr><td>rus</td><td>0.51/0.06</td><td>0.24/0.06</td><td>0.48/0.08</td><td>0.46/0.09</td><td>0.58/0.27</td><td>0.52/0.05</td><td></td><td>0.48/0.07</td><td>0.33/0.08</td></tr><tr><td>deu</td><td>0.50/0.05</td><td>0.25/0.06</td><td>0.43/0.06</td><td>0.38/0.09</td><td>0.55/0.20</td><td>0.46/0.05</td><td>0.48/0.07</td><td></td><td>0.29/0.07</td></tr><tr><td>jpn</td><td>0.30/0.08</td><td>0.44/0.19</td><td>0.30/0.09</td><td>0.32/0.10</td><td>0.42/0.10</td><td>0.26/0.07</td><td>0.33/0.08</td><td>0.29/0.07</td><td></td></tr></table>

Table 17: Last-layer pairwise CKA (matched/shuffled) — Token-Level Word-Aligned, OPUS dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.38/0.02</td><td>0.59/0.05</td><td>0.53/0.07</td><td>0.41/0.04</td><td>0.57/0.04</td><td>0.56/0.05</td><td>0.58/0.06</td><td>0.27/0.09</td></tr><tr><td>zho</td><td>0.38/0.02</td><td></td><td></td><td>0.35/0.02</td><td></td><td>0.36/0.02</td><td>0.39/0.02</td><td>0.47/0.03</td><td></td></tr><tr><td>spa</td><td>0.59/0.05</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>arb</td><td>0.53/0.07</td><td>0.35/0.02</td><td></td><td></td><td></td><td>0.46/0.04</td><td>0.50/0.04</td><td>0.41/0.09</td><td></td></tr><tr><td>hin</td><td>0.41/0.04</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>fra</td><td>0.57/0.04</td><td>0.36/0.02</td><td></td><td>0.46/0.04</td><td></td><td></td><td>0.50/0.04</td><td>0.51/0.04</td><td></td></tr><tr><td>rus</td><td>0.56/0.05</td><td>0.39/0.02</td><td></td><td>0.50/0.04</td><td></td><td>0.50/0.04</td><td></td><td>0.45/0.07</td><td></td></tr><tr><td>deu</td><td>0.58/0.06</td><td>0.47/0.03</td><td></td><td>0.41/0.09</td><td></td><td>0.51/0.04</td><td>0.45/0.07</td><td></td><td></td></tr><tr><td>jpn</td><td>0.27/0.09</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 18: Last-layer pairwise CKA (matched/shuffled) — Token-Level Word-Aligned, BouQUET dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.34/0.04</td><td>0.54/0.06</td><td>0.37/0.06</td><td>0.44/0.06</td><td>0.54/0.05</td><td>0.53/0.06</td><td>0.51/0.05</td><td>0.38/0.06</td></tr><tr><td>zho</td><td>0.34/0.04</td><td></td><td>0.35/0.04</td><td>0.31/0.05</td><td>0.36/0.05</td><td>0.32/0.04</td><td>0.38/0.05</td><td>0.34/0.05</td><td>0.67/0.16</td></tr><tr><td>spa</td><td>0.54/0.06</td><td>0.35/0.04</td><td></td><td>0.40/0.06</td><td>0.39/0.06</td><td>0.58/0.06</td><td>0.50/0.06</td><td>0.48/0.06</td><td>0.35/0.05</td></tr><tr><td>arb</td><td>0.37/0.06</td><td>0.31/0.05</td><td>0.40/0.06</td><td></td><td>0.33/0.06</td><td>0.35/0.06</td><td>0.37/0.06</td><td>0.33/0.06</td><td>0.34/0.06</td></tr><tr><td>hin</td><td>0.44/0.06</td><td>0.36/0.05</td><td>0.39/0.06</td><td>0.33/0.06</td><td></td><td>0.41/0.06</td><td>0.40/0.06</td><td>0.44/0.06</td><td>0.37/0.05</td></tr><tr><td>fra</td><td>0.54/0.05</td><td>0.32/0.04</td><td>0.58/0.06</td><td>0.35/0.06</td><td>0.41/0.06</td><td></td><td>0.50/0.06</td><td>0.46/0.05</td><td>0.35/0.06</td></tr><tr><td>rus</td><td>0.53/0.06</td><td>0.38/0.05</td><td>0.50/0.06</td><td>0.37/0.06</td><td>0.40/0.06</td><td>0.50/0.06</td><td></td><td>0.47/0.06</td><td>0.38/0.06</td></tr><tr><td>deu</td><td>0.51/0.05</td><td>0.34/0.05</td><td>0.48/0.06</td><td>0.33/0.06</td><td>0.44/0.06</td><td>0.46/0.05</td><td>0.47/0.06</td><td></td><td>0.35/0.05</td></tr><tr><td>jpn</td><td>0.38/0.06</td><td>0.67/0.16</td><td>0.35/0.05</td><td>0.34/0.06</td><td>0.37/0.05</td><td>0.35/0.06</td><td>0.38/0.06</td><td>0.35/0.05</td><td></td></tr></table>

Table 19: Last-layer pairwise CKA (matched/shuffled) — SGPT Position-Weighted Pooling, FLORES dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.82/0.23</td><td>0.85/0.24</td><td>0.81/0.24</td><td>0.79/0.24</td><td>0.86/0.24</td><td>0.84/0.25</td><td>0.85/0.24</td><td>0.82/0.23</td></tr><tr><td>zho</td><td>0.82/0.23</td><td></td><td>0.80/0.22</td><td>0.77/0.23</td><td>0.77/0.23</td><td>0.80/0.24</td><td>0.80/0.23</td><td>0.79/0.24</td><td>0.83/0.21</td></tr><tr><td>spa</td><td>0.85/0.24</td><td>0.80/0.22</td><td></td><td>0.80/0.23</td><td>0.76/0.23</td><td>0.85/0.23</td><td>0.82/0.23</td><td>0.83/0.24</td><td>0.80/0.22</td></tr><tr><td>arb</td><td>0.81/0.24</td><td>0.77/0.23</td><td>0.80/0.23</td><td></td><td>0.76/0.25</td><td>0.80/0.25</td><td>0.80/0.25</td><td>0.80/0.24</td><td>0.77/0.24</td></tr><tr><td>hin</td><td>0.79/0.24</td><td>0.77/0.23</td><td>0.76/0.23</td><td>0.76/0.25</td><td></td><td>0.77/0.24</td><td>0.77/0.25</td><td>0.77/0.24</td><td>0.77/0.22</td></tr><tr><td>fra</td><td>0.86/0.24</td><td>0.80/0.24</td><td>0.85/0.23</td><td>0.80/0.25</td><td>0.77/0.24</td><td></td><td>0.83/0.24</td><td>0.84/0.24</td><td>0.80/0.22</td></tr><tr><td>rus</td><td>0.84/0.25</td><td>0.80/0.23</td><td>0.82/0.23</td><td>0.80/0.25</td><td>0.77/0.25</td><td>0.83/0.24</td><td></td><td>0.83/0.25</td><td>0.80/0.23</td></tr><tr><td>deu</td><td>0.85/0.24</td><td>0.79/0.24</td><td>0.83/0.24</td><td>0.80/0.24</td><td>0.77/0.24</td><td>0.84/0.24</td><td>0.83/0.25</td><td></td><td>0.80/0.22</td></tr><tr><td>jpn</td><td>0.82/0.23</td><td>0.83/0.21</td><td>0.80/0.22</td><td>0.77/0.24</td><td>0.77/0.22</td><td>0.80/0.22</td><td>0.80/0.23</td><td>0.80/0.22</td><td></td></tr></table>

Table 20: Last-layer pairwise CKA (matched/shuffled) — SGPT Position-Weighted Pooling, Tatoeba dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.43/0.17</td><td>0.81/0.21</td><td>0.65/0.22</td><td>0.66/0.25</td><td>0.76/0.24</td><td>0.72/0.23</td><td>0.81/0.23</td><td>0.63/0.24</td></tr><tr><td>zho</td><td>0.43/0.17</td><td></td><td>0.62/0.22</td><td>0.52/0.33</td><td>0.48/0.19</td><td>0.59/0.22</td><td>0.52/0.21</td><td>0.64/0.24</td><td>0.40/0.15</td></tr><tr><td>spa</td><td>0.81/0.21</td><td>0.62/0.22</td><td></td><td>0.69/0.24</td><td>0.87/0.65</td><td>0.76/0.23</td><td>0.74/0.24</td><td>0.76/0.24</td><td>0.65/0.26</td></tr><tr><td>arb</td><td>0.65/0.22</td><td>0.52/0.33</td><td>0.69/0.24</td><td></td><td></td><td>0.67/0.26</td><td>0.68/0.24</td><td>0.66/0.26</td><td>0.60/0.23</td></tr><tr><td>hin</td><td>0.66/0.25</td><td>0.48/0.19</td><td>0.87/0.65</td><td></td><td></td><td>0.83/0.51</td><td>0.85/0.63</td><td>0.85/0.57</td><td>0.75/0.32</td></tr><tr><td>fra</td><td>0.76/0.24</td><td>0.59/0.22</td><td>0.76/0.23</td><td>0.67/0.26</td><td>0.83/0.51</td><td></td><td>0.70/0.23</td><td>0.76/0.24</td><td>0.60/0.26</td></tr><tr><td>rus</td><td>0.72/0.23</td><td>0.52/0.21</td><td>0.74/0.24</td><td>0.68/0.24</td><td>0.85/0.63</td><td>0.70/0.23</td><td></td><td>0.74/0.26</td><td>0.68/0.26</td></tr><tr><td>deu</td><td>0.81/0.23</td><td>0.64/0.24</td><td>0.76/0.24</td><td>0.66/0.26</td><td>0.85/0.57</td><td>0.76/0.24</td><td>0.74/0.26</td><td></td><td>0.64/0.26</td></tr><tr><td>jpn</td><td>0.63/0.24</td><td>0.40/0.15</td><td>0.65/0.26</td><td>0.60/0.23</td><td>0.75/0.32</td><td>0.60/0.26</td><td>0.68/0.26</td><td>0.64/0.26</td><td></td></tr></table>

Table 21: Last-layer pairwise CKA (matched/shuffled) — SGPT Position-Weighted Pooling, OPUS dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.85/0.14</td><td>0.83/0.16</td><td>0.71/0.20</td><td>0.34/0.10</td><td>0.85/0.15</td><td>0.82/0.15</td><td>0.81/0.19</td><td>0.54/0.27</td></tr><tr><td>zho</td><td>0.85/0.14</td><td></td><td></td><td>0.74/0.14</td><td></td><td>0.72/0.16</td><td>0.74/0.16</td><td>0.63/0.18</td><td></td></tr><tr><td>spa</td><td>0.83/0.16</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>arb</td><td>0.71/0.20</td><td>0.74/0.14</td><td></td><td></td><td></td><td>0.86/0.13</td><td>0.86/0.13</td><td>0.53/0.23</td><td></td></tr><tr><td>hin</td><td>0.34/0.10</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>fra</td><td>0.85/0.15</td><td>0.72/0.16</td><td></td><td>0.86/0.13</td><td></td><td></td><td>0.85/0.11</td><td>0.82/0.13</td><td></td></tr><tr><td>rus</td><td>0.82/0.15</td><td>0.74/0.16</td><td></td><td>0.86/0.13</td><td></td><td>0.85/0.11</td><td></td><td>0.63/0.23</td><td></td></tr><tr><td>deu</td><td>0.81/0.19</td><td>0.63/0.18</td><td></td><td>0.53/0.23</td><td></td><td>0.82/0.13</td><td>0.63/0.23</td><td></td><td></td></tr><tr><td>jpn</td><td>0.54/0.27</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 22: Last-layer pairwise CKA (matched/shuffled) — SGPT Position-Weighted Pooling, BouQUET dataset. All values computed on 1000 MB Goldfish models.
<table><tr><td></td><td>eng</td><td>zho</td><td>spa</td><td>arb</td><td>hin</td><td>fra</td><td>rus</td><td>deu</td><td>jpn</td></tr><tr><td>eng</td><td></td><td>0.75/0.25</td><td>0.83/0.27</td><td>0.60/0.20</td><td>0.76/0.25</td><td>0.83/0.24</td><td>0.81/0.27</td><td>0.81/0.26</td><td>0.77/0.27</td></tr><tr><td>zho</td><td>0.75/0.25</td><td></td><td>0.75/0.23</td><td>0.55/0.20</td><td>0.70/0.25</td><td>0.74/0.24</td><td>0.76/0.23</td><td>0.74/0.24</td><td>0.76/0.24</td></tr><tr><td>spa</td><td>0.83/0.27</td><td>0.75/0.23</td><td></td><td>0.61/0.20</td><td>0.72/0.24</td><td>0.83/0.26</td><td>0.80/0.24</td><td>0.80/0.25</td><td>0.75/0.25</td></tr><tr><td>arb</td><td>0.60/0.20</td><td>0.55/0.20</td><td>0.61/0.20</td><td></td><td>0.54/0.20</td><td>0.57/0.21</td><td>0.56/0.20</td><td>0.56/0.20</td><td>0.55/0.21</td></tr><tr><td>hin</td><td>0.76/0.25</td><td>0.70/0.25</td><td>0.72/0.24</td><td>0.54/0.20</td><td></td><td>0.74/0.24</td><td>0.76/0.25</td><td>0.71/0.25</td><td>0.70/0.26</td></tr><tr><td>fra</td><td>0.83/0.24</td><td>0.74/0.24</td><td>0.83/0.26</td><td>0.57/0.21</td><td>0.74/0.24</td><td></td><td>0.81/0.24</td><td>0.79/0.26</td><td>0.75/0.25</td></tr><tr><td>rus</td><td>0.81/0.27</td><td>0.76/0.23</td><td>0.80/0.24</td><td>0.56/0.20</td><td>0.76/0.25</td><td>0.81/0.24</td><td></td><td>0.79/0.24</td><td>0.75/0.26</td></tr><tr><td>deu</td><td>0.81/0.26</td><td>0.74/0.24</td><td>0.80/0.25</td><td>0.56/0.20</td><td>0.71/0.25</td><td>0.79/0.26</td><td>0.79/0.24</td><td></td><td>0.76/0.25</td></tr><tr><td>jpn</td><td>0.77/0.27</td><td>0.76/0.24</td><td>0.75/0.25</td><td>0.55/0.21</td><td>0.70/0.26</td><td>0.75/0.25</td><td>0.75/0.26</td><td>0.76/0.25</td><td></td></tr></table>

![](images/50a11b4c86db3eb95720a4f19824f1f9af1a2d65b418a5ba3e7b6331393aa120.jpg)  
Figure 16: Setting 3 — SGPT Position-Weighted Pooling: same layout as Figure 14. SGPT pooling yields the highest absolute matched values across all scales, while preserving the monotonic size effect.

<table><tr><td>Method</td><td>Mean P@1</td><td>Mean MSE</td><td>Mean CKA</td><td>Best P@1</td><td>Best MSE</td><td>Best CKA</td></tr><tr><td>Procrustes</td><td>0.775 [0.747, 0.803]</td><td>0.689 [0.662, 0.715]</td><td>0.713 [0.689, 0.735]</td><td>36/36</td><td>0/36</td><td>0/36</td></tr><tr><td>Affine</td><td>0.663 [0.628, 0.699]</td><td>0.439 [0.422, 0.455]</td><td>0.762 [0.741, 0.781]</td><td>0/36</td><td>36/36</td><td>32/36</td></tr><tr><td>MLP</td><td></td><td></td><td>0.643 [0.605, 0.682] 0.456 [0.440, 0.473] 0.760 [0.739, 0.778]</td><td>0/36</td><td>0/36</td><td>4/36</td></tr></table>

Table 23: Method comparison across all 36 pairs at layer 12. Cells show mean across 36 pairs with bootstrap 95% CI (10,000 resamples). “Best on metric" counts the number of pairs on which each method achieves the top score.

![](images/90f16114d894b34f4ae6fec9b321c0fe903defdc42508883554ffaa9108a2021.jpg)  
Figure 17: Method×metric heatmaps across layers and pairs. Rows: methods (Procrustes, Affine, MLP). Columns: metrics (P@1, MSE, CKA). Each cell aggregates across 36 language pairs by layer.

eng\_latn (source / projected source) After

Sentence reps before / after Procrustes (rotation only) mapping (eng\_latn → fra\_latn)

![](images/31d9710bfbb44de65952232fff9e14236b9b4a849598c076dc9f0bc8fe1cfe12.jpg)

![](images/29a04a14f3c02e9a3d9bffbe635d7e9ed984dd6e8f44703c149a56b0501aaa30.jpg)

![](images/5f1e660f61f0d9e05511c974d2ba82bafa142a35da1885b5bb0832a74282457d.jpg)

![](images/5442742778e6901d10b9f451865ef919561b8668e6125a34670dc3cc68ff33dc.jpg)

![](images/2dba7c2e852d5c77b0d01ecb52e9c5208489744132d2d08605e2d885848c86b9.jpg)

![](images/112b57aa5d733e903fb9fd5807290d87697f70740091e4a1dbf43139f5678fe3.jpg)

![](images/354ea9cee744f9819766c529c7825642abc5895dadf040f8ff183288fabaeced.jpg)

![](images/e51c63258ffee70b5762405cdea901f140fb0c702ebe105a8bd086b49722708d.jpg)

![](images/f9d7b6d7418e4d15b4fdb0996b59ae211588343311e67b2b125ad7b354c0b9e7.jpg)

![](images/ee316e237535a63d46a36bebd19f161af0f403c435a03fc6ccc5b7c5628bbddd.jpg)

![](images/bb6f0f23eaa2efa4a9edcfb1bea400b4a2fbd39db5200bc511d12c5965bf41ee.jpg)

![](images/aed38027df5d36337d1d9df93ef5cdd91148d7ad043aa15600a5f6d1078946b0.jpg)

![](images/8c4e80d9855da77350ce21292074c99b531b411b638943db2699ebf6cdd8a515.jpg)

![](images/c5afb0bab87879538a6bc241539c4c229290618ba36d80f7f4ff777da1f7315e.jpg)

![](images/60e77472a102ecee1c9ce0e4c7c9d9bd07cd61f2175da2a9553cce0ef49c1ad0.jpg)

![](images/02d52e33b7120a4f9e5cf66952997fd094544dad8706d145facd88fe389cd68e.jpg)

![](images/72fd018ae782e31cf76cc8e159b624e7f386b9bed9c25589c4d7e414b5785a58.jpg)

![](images/7eeb7ed093bd46c70f80a2df8521531753b739e019b2b17722884baa3455fe17.jpg)

![](images/561c9701744d59b9e0ada32e86dc60ac2837cf794db1146fa33b5f0382539c7f.jpg)

![](images/325d515e99a308474352dcbb3743ea117f2c4ec76a1485e7e26bdda2cca4c8d0.jpg)

![](images/5bd4147aef4f1a85a971c54ec5158bc880f9c9644202a636be095975d5eadfa6.jpg)

![](images/0a9c3a317fa07416c5d4cd8fc0d127ec18d28f7058ff87c95876b795cba1fcbd.jpg)

![](images/4399a1270848b233ac6e75decbc2b2368ac3eae73124a49a2bb1d96c436ff69c.jpg)

![](images/c1b6ae149a3b2ed13c3423ad7174018d974b37f54f2df63657947501a5dbf673.jpg)

![](images/b567ad2d6b32205e882ad9edee15a2d8c6782e180302414ef536a8222c319d54.jpg)

![](images/8bb00ed720826c09218e60e3217bc72b97d1d9540fa9c32eaded3537325a7cfa.jpg)  
Figure 18: Procrustes mapping, eng-fra (peak retrieval layer; 2D PCA over the joint test set). English source (o), French target (•), and Procrustes-mapped English (×). Lines connect each English source to its matched French target.