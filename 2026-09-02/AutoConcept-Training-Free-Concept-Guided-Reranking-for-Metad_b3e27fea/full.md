# AutoConcept: Training-Free Concept-Guided Reranking for Metadata-Available Composed Image Retrieval

Tianyu Wang<sup>1B</sup> and Tianjiao Wu<sup>2</sup>

<sup>1</sup> School of Computer Science and Technology, Soochow University, Suzhou, China tywangcs@foxmail.com

<sup>2</sup> INSTITUT NATIONAL DES SCIENCES APPLIQUEES DE LYON, Lyon, France tianjiao.wu@insa-lyon.fr

Abstract. Composed image retrieval (CIR) retrieves a target image from a reference image and a text modification. This paper studies metadata-available CIR reranking, where a fixed CIR model first returns a candidate pool and gallery metadata is then used for secondstage concept-guided scoring. We introduce AutoConcept, a trainingfree reranker that converts concept evidence into an interpretable memory. AutoConcept filters noisy concepts, activates query-relevant positive constraints with an auxiliary negative penalty, and combines base retrieval scores with metadata-based concept-candidate alignment through inference-time calibration. On FashionIQ, AutoConcept yields significant early-rank improvements over WeiMoCIR and consistent plug-in gains on LinCIR candidate pools. Metadata-aware controls show that structured concept memory adds signal beyond direct query-text and extracted-attribute matching, while a query-only variant further supports the efectiveness of concept-level reranking. A supplementary realhuman concept-label study indicates that the same memory interface can consume participant-provided evidence. These results position AutoConcept as an interpretable concept-memory reranker for product-style CIR galleries with available metadata.

Keywords: Composed image retrieval · Metadata-available reranking · Concept memory

## 1 Introduction

Composed image retrieval (CIR) asks a system to retrieve a target image given a reference image and a natural-language modification. In fashion and e-commerce retrieval, candidate items are commonly associated with metadata such as titles, attributes, tags, or descriptions. Users may also express concept-level constraints, such as preferred colors, sleeve length, neckline, material, or styles to avoid. Most CIR systems construct one composed query representation and rank the gallery by similarity to that query [28, 9, 26, 6, 22]. A complementary second-stage layer that exposes and applies such concepts can make reranking more controllable and easier to inspect.

![](images/027705391177429376b4b15cbf59137c581a1647f4069db0eca53af3c024e9d0.jpg)  
Fig. 1. AutoConcept in action. Given the same composed query and first-stage candidate pool, the explicit concept memory promotes candidates that match query-relevant positive constraints and penalizes candidates matching negative constraints.

This work studies metadata-available concept-guided CIR reranking. A fixed first-stage retriever returns candidate images, and gallery metadata is available only after this candidate pool has been formed. AutoConcept addresses this setting by converting controlled concept evidence into an interpretable memory that guides candidate ordering without training. The central question is whether explicit concept evidence can improve a strong CIR candidate pool beyond direct query-to-metadata or attribute-to-metadata matching.

We propose AutoConcept, a training-free concept-guided reranker. AutoConcept keeps the first-stage CIR model fixed and operates on its top-K candidate pool. It constructs positive concept constraints from controlled evidence, retains a lightweight negative constraint penalty, filters noisy concepts, activates queryrelevant concepts with a relevance gate, and combines base retrieval scores with metadata-based concept-candidate alignment. The result is a modular reranking layer that can be attached to diferent CIR candidate generators.

Our main experiments use FashionIQ with five controlled concept-evidence seeds. AutoConcept improves WeiMoCIR [29] from 0.1125 to 0.1379 Recall@10 and LinCIR candidates [8] from 0.2605 to 0.3009 under the same metadataavailable reranking protocol. Metadata-aware controls locate the contribution relative to direct query-to-metadata, extracted-attribute, and composed-text reranking; an aligned-vs-shufled analysis shows that coherent query-evidence correspondence matters. We further collect a real-human concept-label dataset to test whether the same memory interface can consume participant-provided item-level concept evidence.

## The contributions are:

1. We formulate metadata-available CIR reranking with an explicit conceptmemory layer over fixed first-stage candidate pools.

2. We give an explicit scoring formulation with positive concept constraints, an auxiliary negative penalty, query-concept relevance gating, and closed-form inference-time score calibration.

![](images/70fbf6204262acf8738c55fdde43735ae99a46c21e515d2ac22f0e5f1e096333.jpg)  
Fig. 2. Overview of the AutoConcept framework. AutoConcept constructs an explicit concept memory from controlled concept evidence, activates query-relevant concepts, and reranks the top-K candidate pool with inference-time score calibration.

3. We evaluate AutoConcept alongside metadata-aware baselines, including query-only evidence, query-to-metadata matching, composed-text metadata matching, and extracted-attribute matching.

4. We show plug-in gains on both WeiMoCIR and LinCIR candidate pools, with aligned-vs-shufled concept-evidence analysis and a real-human concept-label study.

Figure 2 summarizes the full AutoConcept pipeline. The base CIR model first produces a top-K candidate pool, while controlled concept evidence is converted into a filtered concept memory. At inference time, query-relevant concepts are activated and used to adaptively rerank the candidate pool.

## 2 Related Work

## 2.1 Composed Image Retrieval

CIR methods retrieve target images using both a reference image and text feedback. Early systems learn image-text composition modules or attention-based fusion over fashion datasets and real-image benchmarks [28, 9, 26, 6, 21, 3]. Recent zero-shot, training-free, and language-only CIR systems build on pretrained vision-language embeddings such as CLIP, learn lightweight language-side mappings, or use textual inversion and candidate verification [22, 4, 5, 24, 3, 1, 8, 25, 29]. AutoConcept is orthogonal to these retrieval backbones: it adds an explicit concept-memory layer over candidate pools produced by existing CIR methods.

## 2.2 Training-Free and Plug-in Reranking

Zero-shot and low-training CIR systems reduce training cost by weighted modality fusion, score recombination, textual inversion, language-only training, or candidate verification [24, 3, 1, 8, 25, 29]. This modular view is also related to retrieval-augmented systems, where a frozen generator or scorer is augmented with external evidence [14, 18, 12]. We use WeiMoCIR as the main training-free CIR baseline and keep the same modular design: the base retriever supplies the candidate pool, and concept memory adjusts the order within that pool, making plug-in evaluation possible. Constraint-based rerankers such as SoFT use MLLMs to derive prescriptive and proscriptive constraints [13]; AutoConcept targets a lighter setting with no per-query MLLM calls, no reranker training, and explicit concept traces.

## 2.3 Concept Evidence and Controllable Reranking

Reranking is a common way to incorporate external evidence after a strong candidate generator, especially when the control signal should remain inspectable. AutoConcept follows this explicit-control direction for CIR and is related to concept-based interpretability methods such as concept activation vectors, automatic concept explanations, concept bottlenecks, prototype networks, basis decomposition, and concept whitening [15, 7, 16]. It is also related to retrieval and recommendation systems that use memory, profiles, or explicit editable attributes as additional evidence [17, 23, 11, 27, 2, 10]. In AutoConcept, concepts have names, polarity, strength, grounding scores, and evidence items, so the memory can be inspected and diagnosed.

## 3 Method

## 3.1 Task Setup

Each query consists of a reference image $x _ { r }$ , modification text $t _ { q }$ , and a target item $y$ in a gallery. A base retrieval model returns a ranked candidate pool $C _ { q } ~ = ~ \{ x _ { i } \} _ { i = 1 } ^ { K }$ . In the main FashionIQ experiments, K = 100. AutoConcept reranks only this candidate pool.

Each query is associated with a controlled concept-evidence set $E _ { q } .$ Auto-Concept converts this evidence into positive concept constraints $P _ { q }$ and a small auxiliary set of negative concept constraints $N _ { q }$ . A concept stores a normalized name, polarity, score, grounding score, and evidence count.

## 3.2 Information Source Protocol

AutoConcept uses information available under the metadata-available reranking protocol. For each evaluation query, concept evidence is constructed before reranking and excludes the ground-truth target identity, target image, target metadata, candidate ranks, and post-retrieval relevance labels. The main controlled evidence samples same-category training-side item texts and converts them into concept constraints with a fixed extraction and filtering pipeline. Candidate metadata is used only at reranking time to score items already returned by the first-stage retriever. We additionally evaluate a query-only evidence variant, where concepts are extracted from the modification text $t _ { q }$

Table 1. Information sources in the FashionIQ metadata-available protocol.
<table><tr><td>Source</td><td>Used?</td><td>Purpose</td></tr><tr><td>Reference image  $x _ { r }$ </td><td>Yes</td><td>first-stage CIR query</td></tr><tr><td>Modification text  $t _ { q }$ </td><td>Yes</td><td>query encoding and concept activation</td></tr><tr><td>Training-side evidence item Yes text</td><td></td><td>controlled concept-evidence construction</td></tr><tr><td>Candidate metadata mi</td><td>Yes</td><td>second-stage candidate-concept alignment</td></tr><tr><td>Target image / target meta- No data</td><td></td><td>excluded from evidence construction and scoring</td></tr><tr><td>Ground-truth target ID</td><td>No</td><td>excluded from scoring and evidence construction</td></tr><tr><td>Candidate rank / relevance No label</td><td></td><td>excluded from evidence construction</td></tr></table>

## 3.3 Concept Memory Construction

AutoConcept extracts short attribute phrases from concept-evidence item text, embeds phrases with a sentence encoder, clusters near-duplicate phrases, and scores concepts with positive or negative evidence. A compact set of qualitycontrol constraints removes generic names, polarity conflicts, poorly grounded concepts, and weak concepts. The filter keeps frequent but meaningful apparel attributes such as sleeves, collar, buttons, stripes, and V-neck.

## 3.4 Query-Relevant Concept Activation

For a query q and concept c, query-concept relevance is

$$
\boldsymbol { r } ( \boldsymbol { q } , \boldsymbol { c } ) = \cos ( e ( t _ { q } ) , e ( \boldsymbol { c } ) ) ,\tag{1}
$$

where $e ( \cdot )$ is the sentence-encoder embedding and c is the concept name. A query-relevance gate activates the concept set:

$$
A _ { q } = \{ c : r ( q , c ) \geq \tau \} .\tag{2}
$$

AutoConcept can also compute a query-specific gate for inference-time calibration:

$$
\tau _ { q } = \mathrm { c l i p } ( \mu ( R _ { q } ) + \kappa \sigma ( R _ { q } ) , \tau _ { m i n } , \tau _ { m a x } ) ,\tag{3}
$$

where $R _ { q } = \{ r ( q , c ) : c \in P _ { q } \cup N _ { q } \}$ . The clipping range is fixed before evaluation and shared across runs. Appendix A.1 reports sensitivity around the operating region.

## 3.5 Candidate-Concept Alignment

For active concept c and candidate item $x _ { i } .$ , AutoConcept computes

$$
\mathrm { a l i g n } ( x _ { i } , c ) = \cos ( e ( m _ { i } ) , e ( c ) ) ,\tag{4}
$$

where $m _ { i }$ is candidate item metadata text. In our experiments, this text comes from the gallery item metadata used uniformly for candidate items; it is not generated from the test target image and does not use the ground-truth target

identity during scoring. This defines a metadata-available reranking protocol; Section 4.3 compares AutoConcept with a direct metadata-text reranker under the same information setting.

Positive concept scores and the auxiliary disliked-concept penalty are aggregated by max pooling:

$$
s _ { q } ^ { + } ( x _ { i } ) = \operatorname* { m a x } _ { c \in A _ { q } ^ { + } } \mathrm { a l i g n } ( x _ { i } , c ) , \quad s _ { q } ^ { - } ( x _ { i } ) = \operatorname* { m a x } _ { c \in A _ { q } ^ { - } } \mathrm { a l i g n } ( x _ { i } , c ) ,\tag{5}
$$

with empty active sets assigned zero contribution. The concept score is

$$
s _ { q } ^ { c o n c e p t } ( x _ { i } ) = w _ { q } ( q ) s _ { q } ^ { b a s e } ( x _ { i } ) + w _ { p } ( q ) s _ { q } ^ { + } ( x _ { i } ) - w _ { n } ( q ) s _ { q } ^ { - } ( x _ { i } ) .\tag{6}
$$

The weights are set at inference time from query-specific concept evidence. Let

$$
a _ { q } ^ { + } = \operatorname* { m a x } A _ { q } ^ { + } , \quad a _ { q } ^ { - } = \operatorname* { m a x } A _ { q } ^ { - } , \quad a _ { q } = \operatorname* { m a x } ( a _ { q } ^ { + } , a _ { q } ^ { - } ) ,\tag{7}
$$

where $A _ { q } ^ { + }$ and $A _ { q } ^ { - }$ are the gated query-concept activation scores, and the maximum over an empty active set is zero. We also compute base-retriever confidence from the top-score margin:

$$
b _ { q } = \mathrm { c l i p } \left( \frac { s _ { ( 1 ) } ^ { b a s e } - s _ { ( 2 ) } ^ { b a s e } } { \delta } , 0 , 1 \right) .\tag{8}
$$

Here $s _ { ( 1 ) } ^ { b a s e }$ and $s _ { . ( 2 ) } ^ { b a s e }$ are the largest and second largest base scores in the candidate pool, and δ is a fixed margin reference shared by all experiments. Positive and negative reliability scales are

$$
c _ { q } ^ { + } = c _ { m i n } + \left( 1 - c _ { m i n } \right) \mathrm { c l i p } \left( \frac { a _ { q } ^ { + } - a _ { m i n } } { a _ { m a x } - a _ { m i n } } , 0 , 1 \right) ,\tag{9}
$$

$$
c _ { q } ^ { - } = c _ { m i n } + \left( 1 - c _ { m i n } \right) \mathrm { c l i p } \left( \frac { a _ { q } ^ { - } - a _ { m i n } } { a _ { m a x } - a _ { m i n } } , 0 , 1 \right) .\tag{10}
$$

The final per-query weights are

$$
w _ { p } ( q ) = w _ { p } ^ { c a p } c _ { q } ^ { + } ( 1 - 0 . 5 b _ { q } ) , \quad w _ { n } ( q ) = w _ { n } ^ { c a p } c _ { q } ^ { - } ( 1 - 0 . 5 b _ { q } ) ,\tag{11}
$$

$$
w _ { q } ( q ) = w _ { q } ^ { m i n } + \operatorname* { m a x } ( w _ { p } ^ { c a p } - w _ { p } ( q ) , 0 ) + \operatorname* { m a x } ( w _ { n } ^ { c a p } - w _ { n } ( q ) , 0 ) .\tag{12}
$$

The caps are fixed upper bounds, while the actual weights are decided for each query. Thus, when concept evidence is weak or the base retriever is confident, more score mass stays with the base retriever; when active concepts are strong and the base margin is small, positive concepts receive larger weights. The disliked-concept term is intentionally lightweight and acts as a conservative penalty. Image, text, and concept embeddings are L2-normalized before cosine similarity. This closed-form inference-time calibration makes the scoring coeficients query-dependent without training additional parameters.

## 3.6 Inference-Time Score Interpolation

AutoConcept further sets the outer interpolation coeficient from inference-time confidence so weak concept activation does not override the base retriever. Let $\bar { a } _ { q }$ be the mean activation strength of active concepts for query $q .$ The upper interpolation bound and base value are set from the base margin:

$$
\lambda _ { q } ^ { m a x } = \lambda _ { m i n } + ( \lambda _ { m a x } ^ { c a p } - \lambda _ { m i n } ) ( 1 - b _ { q } ) ,\tag{13}
$$

$$
\lambda _ { q } ^ { b a s e } = \lambda _ { m i n } + ( \lambda _ { 0 } - \lambda _ { m i n } ) ( 1 - b _ { q } ) ,\tag{14}
$$

$$
\lambda _ { q } = \mathrm { c l i p } ( \lambda _ { q } ^ { b a s e } + \gamma \bar { a } _ { q } , \lambda _ { m i n } , \lambda _ { q } ^ { m a x } ) .\tag{15}
$$

The interpolation bounds are fixed before evaluation and produce a queryspecific coeficient. The final score is

$$
s _ { q } ( x _ { i } ) = ( 1 - \lambda _ { q } ) s _ { q } ^ { b a s e } ( x _ { i } ) + \lambda _ { q } s _ { q } ^ { c o n c e p t } ( x _ { i } ) .\tag{16}
$$

## Table 2. AutoConcept reranking procedure.

Step Operation   
1 Take query $( x _ { r } , t _ { q } ) ,$ , candidate pool $C _ { q } .$ metadata $\{ m _ { i } \}$ , and evidence $E _ { q }$   
2 Extract, cluster, score, and filter concept constraints from $E _ { q }$   
3 Compute query-concept relevance $r ( q , c )$ and activate concepts   
4 Align each candidate metadata text m<sub>i</sub> with active concepts   
5 Compute positive score, auxiliary negative penalty, and base confidence   
6 Infer $w _ { q } ( q ) , w _ { p } ( q ) , w _ { n } ( q )$ and $\lambda _ { q }$ from query-time signals   
7 Rerank $C _ { q }$ by the final score $s _ { q } ( x _ { i } )$

## 4 Experiments

## 4.1 Dataset, Protocol, and Baselines

The main dataset is FashionIQ. We evaluate five controlled concept-evidence seeds and report mean with sample standard deviation when a method depends on constructed evidence. The first-stage baseline is WeiMoCIR:

$$
q = 0 . 2 0 e ( x _ { r } ) + 0 . 8 0 e ( t _ { q } ) .\tag{17}
$$

Candidate scoring uses candidate image embeddings for first-stage retrieval. Candidate metadata is introduced only in second-stage reranking controls and AutoConcept. The default candidate pool is top-100. We report Recall@10, Recall@50, MRR, and NDCG@10.

We evaluate metadata-aware rerankers on the same candidate pool: querytext to metadata matching, composed reference-text plus query-text to metadata matching, extracted query-attribute to metadata matching, and a resourcematched prescriptive/proscriptive constraint matcher. These baselines locate

Table 3. Main results on two first-stage retrievers. AutoConcept reranks the fixed candidate pool under the same metadata-available protocol.
<table><tr><td>First-stage</td><td>Reranker</td><td>R@10</td><td>R@50</td><td>MRR</td><td>NDCG@10</td></tr><tr><td>WeiMoCIR</td><td>none</td><td>0.1125</td><td>0.2256</td><td>0.0604</td><td>0.0680</td></tr><tr><td></td><td>WeiMoCIR AutoConcept 0.1379 ± 0.0011</td><td></td><td>0.2452 ± 0.0018</td><td>0.0739 ± 0.0008</td><td>0.0844 ± 0.0009</td></tr><tr><td>LinCIR</td><td>none</td><td>0.2605</td><td>0.4618</td><td>0.1449</td><td>0.1637</td></tr><tr><td>LinCIR</td><td></td><td></td><td></td><td>AutoConcept 0.3009 ± 0.0018 0.4915 ± 0.0023 0.1677 ± 0.0010</td><td>0.1912 ± 0.0008</td></tr></table>

![](images/930075d7bf7f5ddda4028bcadf1f1a3d1ce373c7f1554218446c953c1dcf583d.jpg)  
Fig. 3. Qualitative reranking examples. AutoConcept uses active concepts to adjust the top-10 candidate list from WeiMoCIR, improving concept-evidence alignment while retaining the fixed first-stage candidate pool.

AutoConcept relative to direct metadata and constraint matching. The queryonly AutoConcept variant builds concepts from $t _ { q }$ and evaluates the conceptmemory scorer with query-local evidence.

Table 3 shows consistent gains on both candidate generators, concentrated in early-rank metrics.

## 4.2 Significance Testing

We use query-level paired bootstrap tests. Each FashionIQ query is one resampling unit; AutoConcept scores are first averaged over five concept-evidence seeds and then paired with the same query under the base retriever. We resample queries with replacement to compute percentile 95% confidence intervals and two-sided paired bootstrap p-values. The reported p-values are descriptive and are not adjusted for multiple metrics; seed-level robustness is reported through the standard deviations in Table 3.

The LinCIR deltas in Table 3 are also significant with $p < 0 . 0 0 1$ under the same query-level paired bootstrap protocol.

Table 4. Query-level paired bootstrap significance against WeiMoCIR.
<table><tr><td>Metric</td><td>Delta</td><td>Relative</td><td>95% CI</td><td>p-value</td></tr><tr><td>R@10</td><td>0.0254</td><td>+22.54%</td><td>[0.0223, 0.0285]</td><td>&lt; 0.001</td></tr><tr><td>R@50</td><td>0.0196</td><td>+8.71%</td><td>[0.0163, 0.0230]</td><td>&lt; 0.001</td></tr><tr><td>MRR</td><td>0.0135</td><td>+22.26%</td><td>[0.0118, 0.0151]</td><td>&lt; 0.001</td></tr><tr><td>NDCG@10</td><td>0.0164</td><td>+24.16%</td><td>[0.0147, 0.0182]</td><td>&lt; 0.001</td></tr></table>

## 4.3 Metadata-Aware Reranking Controls

Table 5 evaluates AutoConcept alongside metadata-aware rerankers on the same top-100 WeiMoCIR pool. Query-only AutoConcept improves over query-only text-to-metadata, extracted-attribute reranking, and resource-matched constraint matching on early-rank metrics. The reference-text enriched row concatenates reference-item metadata with the modification text and is reported as a strong text-only upper reference under a richer textual signal rather than a sameevidence concept-memory baseline.

Table 5. Metadata-aware reranking controls on FashionIQ seed 1. Query-only Auto-Concept uses concepts extracted only from the modification text.
<table><tr><td>Setting</td><td>Metadata Memory</td><td></td><td>R@10</td><td>R@50</td><td>MRR NDCG@10</td></tr><tr><td>WeiMoCIR</td><td>No</td><td>No</td><td>0.1125</td><td>0.2256 0.0604</td><td>0.0680</td></tr><tr><td>Query text → metadata</td><td>Yes</td><td>No</td><td>0.1318</td><td>0.2513 0.0731</td><td>0.0822</td></tr><tr><td>Extracted attributes → metadata</td><td>Yes</td><td>No</td><td>0.1363</td><td>0.2591 0.0754</td><td>0.0850</td></tr><tr><td>Constraint matching</td><td>Yes</td><td>No</td><td>0.1383</td><td>0.2603 0.0763</td><td>0.0861</td></tr><tr><td>Query-only AutoConcept</td><td>Yes</td><td>Yes</td><td>0.1400</td><td>0.2630 0.0800</td><td>0.0893</td></tr><tr><td>Controlled-evidence AutoConcept</td><td>Yes</td><td>Yes</td><td>0.1375</td><td>0.2435 0.0741</td><td>0.0845</td></tr><tr><td>Reference text + query → metadata</td><td>Yes</td><td>No</td><td>0.1430 0.2645</td><td>0.0811</td><td>0.0910</td></tr></table>

## 4.4 Concept-Evidence Alignment

Table 3 is the main controlled concept-evidence setting. Separately, to test whether AutoConcept uses query-relevant concept evidence rather than only category priors, we compare aligned evidence with shufled evidence. The shuffled control preserves the same domain and category-level concept distribution but breaks query-evidence alignment.

Table 6. Concept-evidence alignment analysis. This setting is separate from the main controlled concept-evidence setting in Table 3.
<table><tr><td>Variant</td><td>Base R@10</td><td>R@10</td><td>R@50</td><td>MRR</td><td>NDCG@10</td></tr><tr><td>Aligned evidence</td><td></td><td></td><td></td><td></td><td>0.1125 0.1892 ± 0.0028 0.2715 ± 0.0014 0.1050 ± 0.0010 0.1213 ± 0.0014</td></tr><tr><td>Shuffled evidence</td><td></td><td></td><td></td><td></td><td>0.1125 0.1388 ± 0.0010 0.2469 ± 0.0010 0.0738 ± 0.0008 0.0845 ± 0.0004</td></tr><tr><td>Delta</td><td></td><td>+0.0504</td><td>+0.0246</td><td>+0.0312</td><td>+0.0368</td></tr></table>

The aligned setting is substantially stronger than the shufled control, supporting the role of coherent query-evidence alignment.

## 4.5 Real-Human Concept Labels

We also evaluate AutoConcept with a pilot real-human concept-label split. Participants labeled FashionIQ gallery items and selected optional item-level concept tags. Many labeled items have empty or very short original item text, so the selected tags provide external concept evidence beyond rich product metadata. We build participant-level concept memories from the deduplicated labels and rerank the same top-100 WeiMoCIR pool.

Table 7 evaluates whether human item-level judgments can be converted into useful concept memory under the same metadata-available reranking pipeline.

Table 7. Real-human concept-label evaluation on FashionIQ.
<table><tr><td>Setting</td><td>R@10</td><td>R@50</td><td>MRR</td><td>NDCG@10</td></tr><tr><td>WeiMoCIR candidate pool</td><td>0.1125</td><td>0.2256</td><td>0.0604</td><td>0.0680</td></tr><tr><td>WeiMoCIR + human AutoConcept</td><td>0.1159</td><td>0.2311</td><td>0.0624</td><td>0.0702</td></tr><tr><td>Delta</td><td>+0.0034</td><td>+0.0056</td><td>+0.0019</td><td>+0.0022</td></tr></table>

The participant histories are sparse and are not collected for specific benchmark targets, yet all four target-label metrics improve, showing that AutoConcept can ingest human concept labels under annotation noise. The smaller gain compared with controlled evidence is expected: participants label only a limited subset of items, selected tags may not cover every FashionIQ modifier, and human memories are not constructed around the benchmark target for each query.

## 4.6 Ablation Analysis

Table 8 isolates scoring components on the same FashionIQ seed-1 top-100 pool. Max pooling is important because concept matches are sparse. Removing the lightweight negative penalty changes little, confirming that FashionIQ gains are driven mainly by positive concept activation.

Table 8. Ablation analysis on FashionIQ seed 1.
<table><tr><td>Setting</td><td>R@10</td><td>MRR</td><td>NDCG@10</td></tr><tr><td>WeiMoCIR</td><td>0.1125</td><td>0.0604</td><td>0.0680</td></tr><tr><td>Full AutoConcept</td><td>0.1375</td><td>0.0741</td><td>0.0845</td></tr><tr><td>Positive concepts only</td><td>0.1380</td><td>0.0742</td><td>0.0847</td></tr><tr><td>Stronger negative penalty</td><td>0.1363</td><td>0.0738</td><td>0.0840</td></tr><tr><td>Mean pooling</td><td>0.1240</td><td>0.0679</td><td>0.0764</td></tr><tr><td>Top-k mean pooling</td><td>0.1270</td><td>0.0695</td><td>0.0783</td></tr><tr><td>Dynamic gate without outer interpolation</td><td>0.1302</td><td>0.0703</td><td>0.0798</td></tr></table>

## 5 Scope and Extensions

AutoConcept is scoped as a second-stage reranker for metadata-available productstyle galleries. The main experiments use controlled concept evidence, while realhuman annotations provide participant-supplied evidence under natural label noise and sparse item text. The method is most suitable when gallery metadata is available and reasonably aligned with item appearance; missing or inconsistent metadata remains a practical boundary. Broader domains, richer visual grounding, larger human-preference studies, stronger metadata denoising, and heavier

![](images/71e14dca9612af67eaff24cb0f5872459582b6edd4790ac2aefa064c9d824329.jpg)  
Fig. 4. Gate sensitivity on FashionIQ. AutoConcept remains stable around the operating range used by inference-time gate calibration.  
learned verification rerankers are natural extensions under diferent compute budgets.

## 6 Conclusion

AutoConcept adds an explicit concept-memory layer to metadata-available composed image retrieval, complementing retrieval-augmented and concept-controlled multimodal systems [19, 20, 10]. It improves both WeiMoCIR and LinCIR candidate pools, compares favorably with direct query and attribute metadata matching, and provides an inspectable memory mechanism. The real-human conceptlabel study further shows that the same interface can consume participantprovided evidence.

## A Additional Evaluation Results

## A.1 Gate Sensitivity

Table 9 reports a gate-sensitivity analysis around $\tau = 0 . 3 5$ . It changes only the query-concept relevance gate and keeps the top-100 candidate pool, controlled concept-evidence setting, concept memory, and calibration bounds fixed. The current implementation can infer $\tau _ { q }$ per query from the memory relevance distribution.

The best region is broad. Gates from 0.30 to 0.40 yield similar gains, and all tested values remain above WeiMoCIR. Positive concepts drive most of the observed gain in the current FashionIQ setting. Disliked concepts are retained as a lightweight, inspectable penalty channel rather than as the primary source of improvement.

Table 9. FashionIQ gate sensitivity.
<table><tr><td>Gate</td><td>R@10</td><td></td><td>R@50</td><td>MRR</td><td>NDCG@10</td></tr><tr><td>0.20</td><td></td><td>0.1304 ± 0.0013 0.2385 ± 0.0016 0.0697 ± 0.0015</td><td></td><td></td><td> $0 . 0 7 9 4 \pm 0 . 0 0 1 5$ </td></tr><tr><td>0.25</td><td> $0 . 1 3 2 8 \pm 0 . 0 0 1 6$ </td><td></td><td>0.2410 ± 0.0016 0.0711 ± 0.0016</td><td></td><td> $0 . 0 8 1 1 \pm 0 . 0 0 1 6$ </td></tr><tr><td>0.30</td><td> $0 . 1 3 5 5 \pm 0 . 0 0 2 4$ </td><td></td><td>0.2442 ± 0.0024</td><td> $0 . 0 7 3 0 \pm 0 . 0 0 1 7$ </td><td> $0 . 0 8 3 1 \pm 0 . 0 0 1 9$ </td></tr><tr><td>0.35</td><td> $0 . 1 3 7 9 \pm 0 . 0 0 1 1$ </td><td>0.2452 ± 0.0018</td><td></td><td> $0 . 0 7 3 9 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 8 4 4 \pm 0 . 0 0 0 9$ </td></tr><tr><td>0.40</td><td> $0 . 1 3 6 3 \pm 0 . 0 0 2 8$ </td><td>0.2440 ± 0.0015</td><td></td><td> $0 . 0 7 3 8 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 8 4 0 \pm 0 . 0 0 1 6$ </td></tr><tr><td>0.45</td><td> $0 . 1 3 3 5 \pm 0 . 0 0 2 0$ </td><td>0.2397 ± 0.0016</td><td></td><td> $0 . 0 7 2 4 \pm 0 . 0 0 1 0$ </td><td> $0 . 0 8 2 3 \pm 0 . 0 0 1 1$ </td></tr><tr><td>0.50</td><td> $0 . 1 3 0 0 \pm 0 . 0 0 2 2$ </td><td>0.2359 ± 0.0009</td><td></td><td> $0 . 0 7 0 3 \pm 0 . 0 0 1 1$ </td><td> $0 . 0 7 9 9 \pm 0 . 0 0 1 4$ </td></tr></table>

## A.2 Candidate Pool Size

Table 10 varies the first-stage candidate pool size for seed 1. The default top-100 pool gives the strongest AutoConcept result in this analysis. Top-50 limits candidate coverage, while larger pools introduce more metadata noise for the same concept-memory scorer.

Table 10. Candidate-pool sensitivity on FashionIQ seed 1.
<table><tr><td>K</td><td>Base R@10</td><td>AC R@10</td><td>Base MRR AC MRR</td></tr><tr><td>50</td><td>0.1125</td><td>0.1144</td><td>0.0594 0.0609</td></tr><tr><td>100</td><td>0.1125</td><td>0.1375</td><td>0.0604 0.0741</td></tr><tr><td>200</td><td>0.1127</td><td>0.1165</td><td>0.0608 0.0631</td></tr><tr><td>500</td><td>0.1129</td><td>0.1180</td><td>0.0614 0.0645</td></tr></table>

This sensitivity suggests that AutoConcept is most efective when the firststage retriever supplies a reasonably precise candidate pool. For weaker retrievers that require very large pools, stronger metadata denoising or visual verification would be needed before reranking.

## A.3 Fashion200K Secondary Check

Fashion200K provides a secondary fashion-domain check constructed from item text and sampled candidate pools. Table 11 reports three-seed means. AutoConcept improves all four metrics over the same WeiMoCIR candidate generator. Table 11. Fashion200K secondary retrieval check.

<table><tr><td>Setting</td><td>R@10</td><td>R@50</td><td></td><td>MRR</td><td>NDCG@10</td></tr><tr><td>WeiMoCIR</td><td> $0 . 6 5 6 0 \pm 0 . 0 3 3 9$ </td><td>0.9144 ± 0.0060</td><td></td><td> $0 . 3 4 5 8 \pm 0 . 0 0 9 7$ </td><td> $0 . 4 0 8 1 \pm 0 . 0 1 6 7$ </td></tr><tr><td>WeiMoCIR + AutoConcept</td><td> $0 . 7 0 4 9 \pm 0 . 0 1 6 7$ </td><td> $0 . 9 3 5 5 \pm 0 . 0 0 1 3$ </td><td> $0 . 3 7 0 9 \pm 0 . 0 2 0 3$ </td><td></td><td> $0 . 4 4 0 4 \pm 0 . 0 1 9 9$ </td></tr></table>

## A.4 Runtime and Latency

Table 12 reports milliseconds per query on an Apple M5 MacBook Air with 24GB unified memory, batch size one, PyTorch/MPS where available, and precomputed CLIP image embeddings. In the cached setting, AutoConcept adds 0.3199 ms/query on FashionIQ and 0.1928 ms/query on Fashion200K, corresponding to 5.58% and 6.18% of first-stage retrieval latency. The higher Fashion200K AC full cost comes from uncached text-encoder setup and metadata encoding over a smaller query set; these embeddings are precomputable in the reranking use case.

Table 12. Runtime comparison between the first-stage WeiMoCIR retriever and the AutoConcept second-stage overhead. The AutoConcept loop column isolates the concept-scoring loop after embeddings are available.
<table><tr><td>Dataset</td><td>Queries</td><td>Gallery</td><td>Mean K</td><td>Base retriever</td><td>AC full</td><td>AC loop</td></tr><tr><td>FashionIQ</td><td>6016</td><td>74381</td><td>100.00</td><td>5.7290</td><td>1.8080</td><td>0.3199</td></tr><tr><td>Fashion200K</td><td>413</td><td>40000</td><td>62.00</td><td>3.1184</td><td>6.2910</td><td>0.1928</td></tr></table>

## B Implementation Details

## B.1 Candidate Pool and Hyperparameters

The main FashionIQ experiment reranks the top-100 candidates from the base retriever. The same top-100 pool is used for WeiMoCIR, AutoConcept, and metadata-aware controls. Concept names, query text, and metadata text are encoded with all-MiniLM-L6-v2; CLIP ViT-B/32 is used for image embeddings and CLIP-based grounding. Quality-control rules and calibration bounds are fixed before evaluation and shared across seeds, candidate generators, and metadataaware controls; no query labels, target ranks, or relevance labels are used for tuning.

Table 13. Reproducibility settings.
<table><tr><td>Setting</td><td>Role</td></tr><tr><td>WeiMoCIR query fusion First-stage candidate scoring</td><td>fixed weighted fusion of reference image and modification text image-side scoring; candidate metadata enters only in reranking</td></tr><tr><td>Candidate pool</td><td>top-100 candidates unless stated otherwise</td></tr><tr><td>Concept/text encoder</td><td>all-MiniLM-L6-v2 for concept names, queries, and metadata</td></tr><tr><td>Image encoder and grounding</td><td>CLIP ViT-B/32 image embeddings and CLIP-based grounding</td></tr><tr><td>Quality control</td><td>fixed filtering rules shared across seeds and candidate generators</td></tr><tr><td>Score calibration</td><td>bounded query-adaptive weights and interpolation, fixed before evaluation</td></tr><tr><td></td><td></td></tr><tr><td>Tuning data</td><td>no query labels, target ranks, or relevance labels used for tuning</td></tr></table>

For K candidates, embedding dimension $d ,$ and active concept sets $A _ { q } ^ { + }$ and $A _ { q } ^ { - }$ , the reranking loop scales as $O ( K ( | A _ { q } ^ { + } | + | A _ { q } ^ { - } | ) d )$ . The relevance gate keeps the active concept set small.

## C Dataset and Concept-Evidence Construction Details

FashionIQ is the main CIR benchmark. Each query contains a reference image, two modification captions, and a target image. We construct controlled concept-evidence sets from training-side item texts; the query-only control extracts concept names only from the modification text.

Fashion200K is a secondary fashion-domain robustness check with composed queries from item text and sampled candidate pools.

The aligned analysis uses controlled attribute families such as color, printed patterns, sleeve length, neckline, length, and fit. It is separate from the main protocol and excludes the target image, target metadata, candidate ranks, and ground-truth target identity. The shufled control keeps the evidence distribution but shufles query-evidence mappings within category.

The real-human study uses a separate collection interface. Participants label FashionIQ items and select optional tags from a fixed vocabulary covering color, pattern, sleeve, neckline, length, fit, and style. Labels are deduplicated, converted into the same concept-memory schema, and stored as a separate human-data split.

## D Concept Memory and Filtering Details

Table 14. Main concept filter rules.
<table><tr><td>Rule</td><td>Action Criterion</td></tr><tr><td>Generic concept name Positive/negative conflict</td><td>drop fixed generic-term stoplist drop same normalized name in both memories</td></tr><tr><td>Low CLIP grounding</td><td>drop</td></tr><tr><td></td><td>below the fixed grounding quality constraint below the fixed concept-strength constraint</td></tr><tr><td>Low concept strength</td><td>drop evidence count &lt; 1</td></tr><tr><td>Low evidence</td><td>drop disabled in main filter</td></tr><tr><td>High global frequency</td><td>keep</td></tr><tr><td>Single weak evidence</td><td>keep disabled in main filter Minimum positive memory restore at least one positive if available</td></tr></table>

The filter removes obvious generic or weak concepts while keeping visually meaningful apparel attributes such as sleeve length, neckline, buttons, stripes, and V-neck.

## E Explanation Diagnostics and Ethics

AutoConcept can list active concepts, evidence items, and candidate-side matches, exposing how memory entries participate in reranking.

AutoConcept stores explicit named concepts rather than opaque user embeddings, supporting inspection, deletion, and editing. The real-human study uses voluntary item-level labels without requesting sensitive user attributes; larger deployments should avoid protected-attribute inference and expose memoryretention controls.

Acknowledgments. We thank the anonymous reviewers for their constructive comments and the participants who contributed item-level concept labels.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Agnolucci, L., Baldrati, A., Bertini, M., Del Bimbo, A.: iSEARLE: Improving textual inversion for zero-shot composed image retrieval. arXiv preprint arXiv:2405.02951 (2024)

2. Alaluf, Y., Richardson, E., Tulyakov, S., Aberman, K., Cohen- $. \mathrm { O r } ,$ D.: MyVLM: Personalizing VLMs for user-specific queries. In: Computer Vision – ECCV 2024. Lecture Notes in Computer Science, vol. 15071, pp. 73–91. Springer (2025). https://doi.org/10.1007/978-3-031-72624-8\_5

3. Baldrati, A., Agnolucci, L., Bertini, M., Del Bimbo, A.: Zero-shot composed image retrieval with textual inversion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2023)

4. Baldrati, A., Bertini, M., Uricchio, T., Del Bimbo, A.: Conditioned and composed image retrieval combining and partially fine-tuning CLIP-based features. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. pp. 4959–4968 (2022)

5. Baldrati, A., Bertini, M., Uricchio, T., Del Bimbo, A.: Efective conditioned and composed image retrieval combining CLIP-based features. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21466– 21474 (2022)

6. Chen, Y., Gong, S., Bazzani, L.: Image search with text feedback by visiolinguistic attention learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3001–3011 (2020)

7. Ghorbani, A., Wexler, J., Zou, J., Kim, B.: Towards automatic concept-based explanations. In: Advances in Neural Information Processing Systems. vol. 32 (2019)

8. Gu, G., Chun, S., Kim, W., Kang, Y., Yun, S.: Language-only training of zeroshot composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13225–13234 (2024)

9. Han, X., Wu, Z., Huang, P.X., Zhang, X., Zhu, M., Li, Y., Zhao, Y., Davis, L.S.: Automatic spatially-aware fashion concept discovery. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 1463–1471 (2017)

10. Hao, H., Han, J., Li, C., Li, Y.F., Yue, X.: RAP: Retrieval-augmented personalization for multimodal large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14538–14548 (2025)

11. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., Chua, T.S.: Neural collaborative filtering. In: Proceedings of the 26th International Conference on World Wide Web. pp. 173–182 (2017)

12. Johnson, J., Douze, M., Jégou, H.: Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7(3), 535–547 (2021)

13. Jung, Y., Cho, S., Min, H.s., Choi, S.: Soft filtering: Guiding zero-shot composed image retrieval with prescriptive and proscriptive constraints. arXiv preprint arXiv:2512.20781 (2025)

14. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.t.: Dense passage retrieval for open-domain question answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. pp. 6769–6781 (2020)

15. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., Sayres, R.: Interpretability beyond feature attribution: Quantitative testing with concept activation vectors. In: Proceedings of the 35th International Conference on Machine Learning. pp. 2668–2677 (2018)

16. Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P.: Concept bottleneck models. In: Proceedings of the 37th International Conference on Machine Learning. pp. 5338–5348 (2020)

17. Koren, Y., Bell, R., Volinsky, C.: Matrix factorization techniques for recommender systems. Computer 42(8), 30–37 (2009)

18. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Advances in Neural Information Processing Systems. vol. 33, pp. 9459–9474 (2020)

19. Li, J., Li, D., Savarese, S., Hoi, S.C.H.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: Proceedings of the 40th International Conference on Machine Learning (2023)

20. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in Neural Information Processing Systems 36 (2023)

21. Liu, Z., Rodriguez-Opazo, C., Teney, D., Gould, S.: Image retrieval on reallife images with pre-trained vision-and-language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2125–2134 (2021)

22. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning. pp. 8748–8763 (2021)

23. Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L.: BPR: Bayesian personalized ranking from implicit feedback. In: Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence. pp. 452–461 (2009)

24. Saito, K., Sohn, K., Zhang, X., Li, C.L., Lee, C.Y., Saenko, K., Pfister, T.: Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19305–19314 (2023)

25. Tang, Y., Yu, J., Gai, K., Zhuang, J., Xiong, G., Hu, Y., Wu, Q.: Context-i2w: Mapping images to context-dependent words for accurate zero-shot composed image retrieval. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 5180–5188 (2024)

26. Vo, N., Jiang, L., Sun, C., Murphy, K., Li, L.J., Fei-Fei, L., Hays, J.: Composing text and image for image retrieval: An empirical odyssey. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6439– 6448 (2019)

27. Weston, J., Chopra, S., Bordes, A.: Memory networks. In: International Conference on Learning Representations (2015)

28. Wu, H., Gao, Y., Guo, X., Al-Halah, Z., Rennie, S., Grauman, K., Feris, R.: Fashion iq: A new dataset towards retrieving images by natural language feedback. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11307–11317 (2021)

29. Wu, R.D., Lin, Y.Y., Yang, H.F.: Training-free zero-shot composed image retrieval via weighted modality fusion and similarity. arXiv preprint arXiv:2409.04918 (2024)