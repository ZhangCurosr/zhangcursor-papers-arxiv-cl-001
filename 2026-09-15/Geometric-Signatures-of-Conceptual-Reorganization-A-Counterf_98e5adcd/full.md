# Geometric Signatures of Conceptual Reorganization: A Counterfactual Embedding Framework for Detecting Scientific Revolutions

Dimitris Ntounis <sup>1</sup> <sup>2</sup> Ariel Schwartzman <sup>2</sup> Chris Chafe <sup>3</sup> Thomas A. Ryckman

## Abstract

We introduce document embedding geometry as a quantitative observable of conceptual reorganization and develop a counterfactual ablation framework for measuring how individual concepts influence the organization of scientific knowledge, providing a quantitative framework for detecting scientific revolutions. The observable is defined by the geometric perturbation induced when removing documents associated with a candidate concept from the embedding space before and after its historical emergence. Statistical validation is performed using five historical case studies spanning physics, mathematics, and machine learning: special relativity, Godel’s incomplete-¨ ness theorems, the Higgs mechanism, deep learning, and the attention mechanism underlying transformer architectures. Across the historical case studies, the framework identifies measurable geometric signatures associated with conceptual reorganization, while the validation studies expose important limitations arising from document assignment and sparse historical data. These results establish embedding geometry as a medium for quantifying conceptual reorganization, providing a new approach for studying how scientific fields restructure over time.<sup>1</sup>

## 1. Introduction

Understanding how knowledge is organized and how that organization changes over time has long been a central objective of the history and philosophy of science and, more recently, of computational approaches to understanding scientific change and innovation. Scientific knowledge is organized through a network of interconnected concepts whose relationships continually evolve as new discoveries emerge. While much of this evolution is gradual, periods of rapid conceptual reorganization occasionally reshape entire disciplines, fundamentally changing how previously established ideas are understood and connected. Such scientific advances do more than extend existing knowledge: they reorganize established understanding and redirect the trajectory of a field (Kuhn, 1962). Special relativity, Godel’s¨ incompleteness theorems, the Higgs mechanism, and, more recently, deep learning are prominent examples. Understanding these reorganizations has traditionally relied on historical analysis and qualitative interpretation. A quantitative description of how conceptual organization itself changes remains comparatively underdeveloped.

Existing quantitative approaches (Fortunato et al., 2018) have sought to characterize scientific change through citation networks, bibliometric indicators, topic evolution, semantic change, and other data-driven analyses. These methods have revealed important patterns of scientific development, including disruption, interdisciplinarity, and knowledge diffusion. However, they primarily characterize relationships external to the conceptual organization itself, such as citation structure, publication dynamics, or changes in topic prevalence, and often inherit the bias of external metadata. In this work, we investigate a complementary question: can quantifiable conceptual reorganizations, including those associated with scientific revolutions, be detected through changes in the geometry of document embedding spaces?

Modern embedding models represent documents as vectors in high-dimensional spaces whose geometry reflects statistical regularities learned directly from text. Since a document’s position in embedding space encodes its semantic content (Ethayarajh, 2019), these representations have become fundamental tools for information retrieval, clustering, recommendation, and scientific search. More fundamentally, embedding spaces may be viewed as learned semantic geometries, in which the relative positions of concepts reflect the statistical structure of the scientific literature. Our starting point is the observation that scientific publications constitute the primary written record through which the conceptual organization of a field is communicated, refined, and extended. If embedding models capture sufficiently rich representations of this literature, then changes in conceptual organization should become observable through changes in the embedding geometry.

Embedding models generally place documents discussing related concepts in nearby regions of the embedding space, while conceptually distinct documents tend to be more widely separated. Consequently, major conceptual reorganizations are expected to manifest not merely through the appearance of new documents, but through measurable changes in the global organization of the embedding space. Our central hypothesis is that concepts responsible for major conceptual reorganizations occupy distinctive structural roles within embedding geometry and that these roles can be measured quantitatively. This perspective suggests a shift in how embedding spaces are used. Rather than treating them solely as representations for retrieval or semantic similarity, we investigate whether their geometry can serve as a measurable scientific observable. In this view, conceptual reorganization is inferred not from the appearance of individual documents, but from the geometric response of the embedding space to controlled counterfactual perturbations.

The central conceptual contribution of this work is to promote embedding geometry from a descriptive semantic representation to a quantitative scientific observable that can be measured, perturbed, statistically validated, and compared across independent historical case studies. For this reason, we introduce a counterfactual ablation framework. For a candidate concept, we remove the documents associated with that concept, reconstruct the embedding geometry, and compare the resulting perturbation before and after the historical emergence of the concept. The underlying intuition is straightforward: if a concept fundamentally reorganizes an existing body of knowledge, removing it should perturb the post-emergence geometry substantially more than the pre-emergence geometry. The resulting asymmetry provides a quantitative observable of conceptual reorganization without relying on citation graphs, expert annotations, or supervised training.

We evaluate the proposed framework on five historical case studies spanning physics, mathematics, and machine learning: special relativity (Einstein, 1905), Godel’s incomplete-¨ ness theorems (Godel ¨ , 1931), the Higgs mechanism (Higgs, 1964), the deep learning revolution (Krizhevsky et al., 2012), and the attention mechanism (Vaswani et al., 2017) underlying transformer architectures. These case studies were selected to span different scientific disciplines and historical contexts, enabling us to study the temporal evolution of conceptual organization under diverse conditions. Rather than attempting to discover previously unknown historical events, our objective is to determine whether embedding geometry captures measurable signatures consistent with historically recognized episodes of conceptual reorganization.

Specifically, this work makes three main contributions. First, we propose embedding geometry as a quantitative observable for studying conceptual reorganization. Second, we introduce a counterfactual ablation framework that measures the structural influence of individual concepts on embedding geometry. Third, we apply the approach across five case studies spanning physics, mathematics, and machine learning to study the temporal evolution of conceptual organization.

Our broader aim is to understand what the embedding spaces of modern language models can reveal about the organization and evolution of knowledge. Rather than replacing existing computational approaches to scientific change, we propose a complementary geometric perspective that treats embedding geometry as a quantitative observable of conceptual organization. Although the present work focuses on scientific and mathematical literature, the framework is naturally applicable to other text-rich domains in which conceptual structures evolve over time, opening opportunities for quantitatively studying the dynamics of ideas across disciplines.

The present work addresses the first step of this broader research program: establishing that historically recognized conceptual reorganizations leave measurable geometric signatures in embedding space. A long-term ambition is to develop quantitative models of knowledge evolution that may ultimately contribute to AI systems capable of assisting scientific discovery by identifying emerging conceptual reorganizations.

## 2. Method

The complete six-step framework is illustrated in Figure 1. The first four steps construct the geometric representation described below. The final two steps implement the counterfactual framework by measuring the geometric perturbation induced by removing the documents associated with a target concept and comparing the resulting perturbation before and after a candidate pivot year.

## 2.1. Representing conceptual organization

Our framework represents both scientific documents and conceptual descriptions within a common embedding space. The representation consists of three steps. First, each scientific document is mapped to an embedding vector that captures its semantic content. Second, a small set of historically meaningful concepts is represented by embedding-based concept anchors. Finally, documents are associated with concepts through semantic similarity. This representation provides the geometric foundation for the counterfactual framework introduced in the next subsection.

![](images/7223e6b6c5e956b72fb2c24267e6ceb585b785cc1c74b59bc0503de45e0e5001.jpg)  
Figure 1. Schematic of the proposed counterfactual ablation framework.

Scientific documents are embedded using modern sentencetransformer models. Because many papers exceed the context window of these models, each document is divided into overlapping segments (chunks) of at most 300 words with a 40-word overlap. Each chunk is embedded independently, and the document representation is the ℓ<sub>2</sub>-normalized mean of its chunk embeddings, that is, the document’s coordinate vector in the embedding space. This procedure incorporates information from throughout the document rather than only its opening text.

Rather than discovering concepts through unsupervised clustering, we define for each historical case a fixed set of $N \ = \ 1 0 \ ^ { 2 }$ concepts consisting of the target concept together with nine contextual concepts spanning the principal subfields surrounding the historical transition. Each concept is represented by a short natural-language description. This design choice keeps the resulting geometry directly interpretable across historical case studies and embedding models, while isolating the structural role of historically identifiable concepts rather than latent statistical topics

Documents are assigned to concepts by comparing each document embedding with an ensemble of concept anchors. For each concept, five paraphrases of its natural-language description are embedded using the same encoder employed for the document corpus, and the concept anchor is the ℓ -normalized mean of the five resulting description embeddings. Averaging across paraphrases reduces sensitivity to the wording of any individual description. Each document is assigned to the concept with the highest cosine similarity provided that two conditions are satisfied: the similarity exceeds a per-concept baseline threshold, and the margin to the second-best concept (Equation (10)) exceeds a confusion-aware threshold. Documents failing either criterion are labeled “no match” and excluded from the geometric analysis. Details of the threshold construction are given in Section 3.3.

Finally, because the proposed framework should reflect conceptual organization rather than properties of a particular embedding model, we evaluate the complete framework using five sentence-transformer models spanning four distinct training paradigms (Table 1). Each model is applied independently from document embedding through statistical validation, allowing the robustness of the geometric signatures to be assessed across substantially different embedding representations. Results for all historical case studies are presented in Section 5.

## 2.2. Counterfactual framework

We define the counterfactual framework for conceptual reorganization through the geometric perturbation induced by a counterfactual ablation, in which all documents associated with a given concept are removed from the corpus and the resulting geometry is recomputed. The magnitude of this perturbation defines the standardized response used throughout the remainder of this paper. The framework is designed to quantify not the introduction of a concept itself, but its structural impact on the organization of the surrounding scientific literature.

For a given target concept, the counterfactual measurement proceeds as follows. First, all documents associated with the target concept are removed from the corpus. Second, concept centroids are recomputed within rolling windows of radius r (spanning year r), with $r = 2$ for the three smaller corpora and $r = 1$ for the two large machine-learning corpora; see Table 3. For each concept c present within a given window, the centroid $\mathbf { c } _ { c } ( t )$ is the ℓ<sub>2</sub>-normalized mean of the document embeddings assigned to that concept. Third, the geometry of the concept space is evaluated for each time window using the observables defined below. Fourth, the geometric perturbation induced by the ablation is computed by comparing the baseline and ablated geometries for every year. Finally, the perturbations before and after a candidate pivot year are compared statistically to quantify the asymmetry associated with the emergence of the concept.

Two complementary observables are used to characterize the geometry of the concept space. They quantify different aspects of conceptual organization: the overall spread of the concept space and the average separation between concepts. In the expressions below, $N ( t )$ denotes the number of concepts with at least one assigned document in the rolling window centered on year t.

Total inertia $( d _ { I } )$ . Let $\mathbf { c } _ { c } ( t )$ denote the centroid of concept c within the rolling window centered on year $t ,$ and let

$$
\bar { \mathbf { c } } ( t ) = \frac { 1 } { N ( t ) } \sum _ { c = 1 } ^ { N ( t ) } \mathbf { c } _ { c } ( t )
$$

be the corresponding average centroid. Total inertia is de-

fined as

$$
d _ { I } ( t ) = \sum _ { c = 1 } ^ { N ( t ) } \left\| \mathbf { c } _ { c } ( t ) - \bar { \mathbf { c } } ( t ) \right\| ^ { 2 } .\tag{1}
$$

This quantity measures the overall spread of the concept space. Concepts that introduce genuinely new directions into the embedding geometry increase the total inertia disproportionately compared with small local drifts.

Mean pairwise cosine distance $( d _ { P } ) .$ . The second observable is the average cosine distance between all concept pairs,

$$
d _ { P } ( t ) = { \binom { N ( t ) } { 2 } } ^ { - 1 } \sum _ { i < j } \left( 1 - \cos ( \mathbf { c } _ { i } ( t ) , \mathbf { c } _ { j } ( t ) ) \right) .\tag{2}
$$

Unlike total inertia, this observable measures the average angular separation between concepts and is insensitive to vector magnitude, making it a natural measure of semantic separation in dense embedding spaces.

For each observable, the yearly geometric perturbation is defined as the difference between the baseline and ablated geometries,

$$
\Delta ( t ) = g ^ { \mathrm { b a s e } } ( t ) - g ^ { \mathrm { a b l } } ( t ) ,\tag{3}
$$

where $g$ denotes either $d _ { I }$ or $d _ { P }$ . Given a candidate pivot year $t ^ { * }$ , the average perturbations before and after the pivot are

$$
\bar { \Delta } _ { \mathrm { p r e } } = \frac { 1 } { n _ { \mathrm { p r e } } } \sum _ { t < t ^ { * } } \Delta ( t ) , \qquad \bar { \Delta } _ { \mathrm { p o s t } } = \frac { 1 } { n _ { \mathrm { p o s t } } } \sum _ { t \ge t ^ { * } } \Delta ( t ) ,\tag{4}
$$

where $n _ { \mathrm { p r e } }$ and $n _ { \mathrm { p o s t } }$ denote the number of yearly observations before and after the pivot.

The strength of the asymmetry is quantified using Cohen’s $D$ (Cohen, 1988),

$$
D = \frac { \bar { \Delta } _ { \mathrm { p o s t } } - \bar { \Delta } _ { \mathrm { p r e } } } { s _ { \mathrm { p o o l } } } ,\tag{5}
$$

$$
s _ { \mathrm { p o o l } } ^ { 2 } = \frac { ( n _ { \mathrm { p r e } } - 1 ) s _ { \mathrm { p r e } } ^ { 2 } + ( n _ { \mathrm { p o s t } } - 1 ) s _ { \mathrm { p o s t } } ^ { 2 } } { n _ { \mathrm { p r e } } + n _ { \mathrm { p o s t } } - 2 } ,\tag{6}
$$

where $s _ { \mathrm { p r e } }$ and $s _ { \mathrm { p o s t } }$ are the sample standard deviations of the yearly perturbations before and after the pivot, respectively, and $s _ { \mathrm { p o o l } }$ is their pooled standard deviation. Cohen’s

Table 1. Embedding models used in this study. All results reported in the main body use e5-base; the remaining four models are used for the robustness study of Section 5.
<table><tr><td>Short name</td><td>Training paradigm</td><td>Dim.</td><td>Reference</td></tr><tr><td>e5-base</td><td>Contrastive, multilingual</td><td>768</td><td>Wang et al. (2024)</td></tr><tr><td>mxbai</td><td>Contrastive</td><td>1024</td><td>Lee et al. (2024)</td></tr><tr><td>bge-m3</td><td>Multi-task, multilingual</td><td>1024</td><td>Chen et al. (2024)</td></tr><tr><td>specter2</td><td>Citation-prediction</td><td>768</td><td>Singh et al. (2023)</td></tr><tr><td>MiniLM</td><td>Distilled, English</td><td>384</td><td>Wang et al. (2020); Reimers &amp; Gurevych (2019)</td></tr></table>

D is a standardized geometric perturbation, allowing direct comparison between geometric observables with different numerical scales while avoiding dependence on their absolute normalization. Throughout the paper we denote this standardized response statistic by the uppercase symbol D to distinguish it from the geometric observables $d _ { I }$ and $d _ { P }$

Different conceptual innovations need not perturb every aspect of the embedding geometry equally. Some primarily introduce new geometric directions and therefore increase the overall spread of the concept space, while others predominantly reorganize relationships among existing concepts. We therefore evaluate both throughout the analysis. For concept ranking, we report the largest standardized response, max $\left( | D _ { I } | , | D _ { P } | \right)$ . For significance, however, collapsing the two observables to their maximum discards the second dimension, which carries information precisely in those diffuse transitions where neither metric dominates. We therefore assess significance jointly on the $( D _ { I } , D _ { P } )$ plane using a covariance-free co-dominance tail.<sup>3</sup> Writing ${ \pmb \mu } ~ = ~ ( \mu _ { I } , \mu _ { P } )$ for the mean of the null cloud and $( D _ { I } ^ { \mathrm { o b s } } , D _ { P } ^ { \mathrm { o b s } } )$ for the observation,

$$
p _ { \mathrm { c o } } = \frac { 1 + M } { N + 1 } ,\tag{7}
$$

where N is the number of null realizations and M counts those that are at least as extreme as the observation on both axes simultaneously,

$$
\begin{array} { r l } & { M = \# \big \{ k : | D _ { I , k } - \mu _ { I } | \geq | D _ { I } ^ { \mathrm { o b s } } - \mu _ { I } | } \\ & { \qquad \mathrm { a n d } | D _ { P , k } - \mu _ { P } | \geq | D _ { P } ^ { \mathrm { o b s } } - \mu _ { P } | \big \} . } \end{array}\tag{8}
$$

We call a response statistically significant when its permutation-based p-value is below 0.05. We do not rank on the joint magnitude $\sqrt { D _ { I } ^ { 2 } + D _ { P } ^ { 2 } }$ because it is not covariance-free: the null correlation between $D _ { I }$ and $D _ { P }$ varies from +0.11 to +1.00 across the five cases, so a fixed contour in the plane corresponds to a different tail probability in each, and rankings built on it would not be comparable across cases. Ranking on a genuinely sign-aware statistic does not change the result. Ranking the ten concepts of each case by the co-dominance tail probability of their signed $( D _ { I } , D _ { P } )$ pair, each against its own permutation null of 5000 draws, reproduces the ranking of Table 4 in four of the five cases and raises the attention mechanism from fourth to third. In the deep-learning case, where ranking on the joint magnitude would place a rival concept first, the sign-aware tail restores the target to first place, because that rival’s two components carry opposite signs and therefore fall in a far less extreme region of its own null. The absolute value in max $\left( | D _ { I } | , | D _ { P } | \right)$ therefore does not drive the reported ranking.

## 2.3. Localizing conceptual reorganization

Conceptual reorganizations are expected to occur over finite historical periods rather than at predetermined dates. To localize these transitions in a data-driven manner, we evaluate D over all combinations of candidate concepts and pivot years, producing a concept  pivot-year grid for each historical case. Figure 2 illustrates the resulting grid for the special relativity corpus, with the corresponding per-metric decomposition into $D _ { I }$ and $D _ { P }$ shown in Figure 3.

Rather than fixing a historically motivated pivot year a priori, this approach allows the data to determine when the geometric perturbation associated with a concept is strongest. Each candidate pivot year is a possible split between the preand post-pivot periods; the data-driven pivot $t ^ { * }$ is the candidate that maximizes the standardized response for the target concept. For each candidate pivot year, the counterfactual procedure described in Section 2.2 is applied independently, yielding a standardized response for every concept–pivotyear combination. The lower time bound of the scan is chosen so that every candidate pivot year has sufficient pre-pivot documents for reliable standardized-response estimation.

Each cell of the concept  pivot-year grid represents the standardized response obtained by applying the counterfactual framework to one concept using one candidate pivot year. The two observables carry different sign information. Because removing a concept can never increase the total inertia of the remaining centroids, $\Delta _ { I } ( t ) \geq 0$ identically, so the sign of $D _ { I }$ contrasts the magnitude of the pre- and post-pivot disruption: positive values mark a concept whose geometric footprint appears or grows at the pivot, and negative values one whose footprint was already largest before it. The pairwise-distance perturbation is instead signed, with a positive $\Delta _ { P }$ identifying a concept sitting farther from its neighbors than they sit from one another, occupying a peripheral direction, while a negative $\Delta _ { P }$ identifies a concept lying inside the ambient spread, so that its removal leaves the remaining concepts more widely separated. The sign of $D _ { P }$ accordingly contrasts this relative position before and after the pivot rather than measuring the size of the disruption.

![](images/27cd5c51966241d8fccb986b6751cdebb6dbbcc9cfbbb05ac6130265779c2218.jpg)  
Figure 2. Concept  pivot-year heatmap for special relativity. Each cell shows Cohen’s $D _ { I }$ for ablating the row’s concept and splitting pre- vs. post-pivot at the column’s year. The inset plots D<sub>I</sub> along the special-relativity row of the same grid.

The concept pivot-year grid may therefore be viewed as a two-dimensional map of the standardized response, describing how geometric perturbations depend jointly on conceptual identity and historical time. For each historical case, the pivot year, which corresponds to the time at which the target concept produces the strongest geometric perturbation, is defined as

$$
t _ { \mathrm { c a s e } } ^ { * } = \arg \operatorname* { m a x } _ { t } | D _ { \mathrm { c a s e } } ( \mathrm { t a r g e t } , t ) | .\tag{9}
$$

Because a single metric dominates at each selected pivot, this $\ell _ { \infty }$ choice coincides with the $\| D \| _ { 2 }$ grid-maximum for four of the five cases, with Godel differing by two years.¨ Moreover, the pivot is a localization step whose freedom is separately corrected by the look-elsewhere scan of Section 2.4, so per-case significance is never conditioned on it. The resulting pivot years for all five historical case studies are summarized in Table 4.

## 2.4. Statistical validation

The metric D introduced in Section 2.2 is intended to identify genuine conceptual reorganizations rather than artifacts arising from corpus composition, document assignment, or individual influential publications. We therefore evaluate every detected signal using a complementary suite of statistical validation tests designed to probe these potential sources of bias. The overall validation strategy is summarized in Figure 4.

Four complementary tests are performed. The randomremoval null compares the observed geometric perturbation with the distribution obtained by removing temporally matched random sets of papers, testing whether the detected signal is specific to the target concept rather than a consequence of removing an arbitrary subset of the corpus (Good, 2000). The scrambled-assignment null randomly permutes concept assignments while preserving the temporal structure of the corpus, testing whether the observed perturbation depends on the correct semantic assignment of documents to concepts rather than on the overall corpus geometry (Subramanian et al., 2005). Third, a leave-one-out jackknife evaluates the stability of the standardized response D against the deletion of individual documents, quantifying how much of the measured effect survives the loss of any single paper (Efron, 1983). Fourth, the look-elsewhere test assesses whether the largest response in the concept pivot-year scan remains significant after accounting for the full scan. Robustness with respect to document assignment is addressed independently through the threshold-selection procedure discussed in Section 3.3.

![](images/c8df01afa15e325fd11bf0d78da3c579f09f2b394601742257614216cfeba70c.jpg)

![](images/aeb0708ce9f49434e2b2b4c1515202b4c8144eaa98f1ee0fd210feb624568731.jpg)  
Figure 3. Per-metric concept  pivot-year heatmaps for the special relativity case.

The complete validation procedure is applied to every historical case study. Detailed results are presented for the special relativity benchmark in Section 3, while the corresponding analyses for the remaining case studies are summarized in Section 4.

In addition, the framework is evaluated for robustness with respect to implementation choices, including document assignment and embedding representation. These complementary robustness studies are presented along with the historical validation in Sections 3 and 4.

## 3. Results: special relativity case study

## 3.1. Detection of conceptual reorganization

We first apply the proposed framework to the special relativity corpus, which serves as the primary benchmark for illustrating the methodology.

The special relativity corpus contains 2,314 documents. Applying the document-assignment procedure described in Section 2.1 yields 62 documents assigned across the ten candidate concepts after rejecting 1,384 documents through the per-concept similarity threshold and a further 868 through the confusion-aware margin criterion. These assigned documents define the concept geometry analyzed throughout the remainder of this section. Not every candidate concept receives assigned documents under the highconfidence assignment criteria. Four concepts (gravitation, radiation/quantum, mechanics/time, and instrumentation) receive no document assignments, while two concepts (aether optics and electrodynamics) are represented by a single assigned document. The absence of assigned documents reflects the conservative high-confidence assignment criteria rather than the historical importance of these concepts, while based on a single assigned document should be interpreted with appropriate caution.

At the data-driven pivot t∗ = 1902 obtained from the concept  pivot-year analysis (Section 2.3), we compute the counterfactual standardized response for each of the ten candidate concepts, shown in Figure 2 and summarized in Table 2. Special relativity gives max $\langle | D _ { I } | , | D _ { P } | \rangle = 7 . 6 1$ six times the second-ranked concept. Removing the special relativity literature therefore perturbs the surrounding embedding geometry far more than removing any other concept in the corpus.

The ranking also illustrates the interpretability of D. Several historically established concepts, including thermodynamics and aether optics, exhibit negative values of the total-inertia response $D _ { I }$ , indicating that their geometric influence is concentrated before the pivot year. Removing these concepts therefore perturbs the earlier embedding geometry more strongly than the later one, consistent with their declining structural role following the emergence of special relativity.

![](images/d71fd1227f5c4b8281bb8d575cc345f11c4c03ace3486fb0d63e4d9bac768dc2.jpg)

# Four core validation tests

## (c) Leave-one-out jackknife

![](images/fba990b255fe82d6b828645c6074a594e464bcffa9192a036fdda6946c888f29.jpg)

![](images/93584df9bd6098201df1b308f16de588706582b9a4a0bc46a60fe772e585eca3.jpg)

## (d) Look-elsewhere efect

![](images/84f96c6477aae5574d4ab1aea0acb90b6af496156049f3ff8e72cb87e07e61f6.jpg)

![](images/c65a5e885d6555b9b1f7843aa7b57ff3c7c58a12b8553e9ce49fdedcc8163639.jpg)

$$
p _ { \mathrm { L E E } } = { \# ( M _ { \mathrm { n u l l } } \geq M _ { \mathrm { o b s } } ) } / { 5 0 0 0 } = 8 \times 1 0 ^ { - 4 }
$$

Observed maximum exceeds the scan null;   
multiplicity is accounted for.

Figure 4. Statistical validation suite used to evaluate the counterfactual framework. (a) Random-removal null (targeted vs. random paper removal). (b) Scrambled-assignment null (permuted concept labels). (c) Leave-one-out stability (single-paper jackknife) and (d) Look-elsewhere effect.

Table 2. Counterfactual standardized responses for all candidate concepts evaluated at the data-driven pivot year $t ^ { * } = 1 9 0 2$ . Concepts are ranked by $\operatorname* { m a x } ( | D _ { I } | , | D _ { P } | )$
<table><tr><td>Rank</td><td>Concept</td><td> $D _ { I }$ </td><td> $D _ { P }$ </td><td>max  $| D |$ </td><td>Papers</td></tr><tr><td>1</td><td>Special relativity</td><td>7.61</td><td>-0.36</td><td>7.61</td><td>16</td></tr><tr><td>2</td><td>Spectroscopy</td><td>1.26</td><td>0.82</td><td>1.26</td><td>9</td></tr><tr><td>3</td><td>Aether optics</td><td>-0.78</td><td>-0.60</td><td>0.78</td><td>1</td></tr><tr><td>4</td><td>Electrodynamics</td><td>-0.78</td><td>-0.15</td><td>0.78</td><td>1</td></tr><tr><td>5</td><td>Electron theory</td><td>0.71</td><td>0.15</td><td>0.71</td><td>6</td></tr><tr><td>6</td><td>Thermodynamics</td><td>-0.67</td><td>-0.49</td><td>0.67</td><td>29</td></tr><tr><td>一</td><td>Gravitation</td><td></td><td>一</td><td>一</td><td>0</td></tr><tr><td></td><td>Radiation / quantum</td><td></td><td></td><td></td><td>0</td></tr><tr><td></td><td>Mechanics / time</td><td></td><td></td><td></td><td>0</td></tr><tr><td></td><td>Instrumentation</td><td></td><td></td><td>—</td><td>0</td></tr></table>

To quantify the stability of the observed effect, we estimate bootstrap confidence intervals by resampling the yearly ablation perturbations with replacement, separately within the pre-pivot and post-pivot periods, and recomputing the complete ranking for each resample. Bootstrap confidence intervals are shown in Figure 5. Special relativity remains the highest-ranked concept in more than 99.9% of bootstrap realizations, with a 95% confidence interval of $D _ { I } \in [ 6 . 5 8 , 1 1 . 1 9 ]$ . Notably, the lower bound of this interval exceeds the point estimate of every other concept, demonstrating that the observed separation is not driven by statistical fluctuations or a small number of yearly observations.

The magnitude of the observed effect, however, is not by itself sufficient to establish that it represents a genuine conceptual reorganization. We therefore next examine the statistical significance and robustness of the special relativity signal using the validation framework introduced in Section 2.4.

## 3.2. Statistical validation of the special relativity signal

As already explained, the validation suite tests four complementary hypotheses: whether the observed signal could arise from random document removal, whether it depends on the correct semantic assignment of documents to concepts, whether it is driven by a small number of influential papers, and whether the concept  pivot-year scan inflates its statistical significance.

Random-removal null. To test whether the observed perturbation could arise simply from removing an arbitrary subset of the corpus, we compare the observed effect with a null distribution obtained by 5,000 permutations of removing 16 temporally matched random papers, corresponding to the size of the special relativity cluster. This is shown in Figure 6. The observed point $( D _ { I } , D _ { P } ) = ( 7 . 6 1 , - 0 . 3 6 )$ lies well outside the resulting null distribution on the joint $( D _ { I } , D _ { P } )$ plane $( p _ { \mathrm { c o } } ~ = ~ 2 . 0 \times 1 0 ^ { - 4 } )$ , with none of the

![](images/750be8c8f0c4e7c5c4e3745ddc19514dbb87033791193a8fa4b0cec56d5636b8.jpg)  
Figure 5. Ablation standardized responses $D _ { I }$ at the pivot $t ^ { * } =$ 1902, for all ten concepts in the special relativity panel, ordered by $| D _ { I } |$ . For each concept, the point estimates and the corresponding 95% bootstrap confidence intervals over 5,000 resamples are given. The four greyed concepts have no assigned documents before the pivot, so their perturbation is identically zero.

5,000 random realizations co-dominating the observed signal. This demonstrates that the measured perturbation is specific to the special relativity documents rather than a generic consequence of removing a similarly sized subset of the corpus.

Scrambled-assignment null. To test whether the observed signal depends on the specific semantic assignment of documents to concepts rather than simply removing an equally sized subset of papers, we construct a null distribution by randomly permuting concept labels across all assigned documents (5,000 permutations) while preserving the temporal structure of the corpus and recomputing the complete geometry for each realization. This is shown in Figure 7. The observed point again lies well outside the resulting null distribution on the joint $( D _ { I } , D _ { P } )$ plane $( p _ { \mathrm { c o } } = 2 . 0 \times 1 0 ^ { - 4 } )$ . This demonstrates that the detected perturbation depends on the specific documents associated with the special relativity concept rather than simply on the number of documents removed.

Leave-one-out stability. The third test examines whether the observed geometric perturbation depends disproportionately on any single document within the special relativity concept. This is a robustness test of the framework rather than a statement about the historical role of individual publications. We delete one special relativity paper at a time from the corpus entirely, recompute both the unablated baseline and the ablated geometry over the remaining documents, and evaluate the standardized response at the data-driven pivot $t ^ { * } = 1 9 0 2$ . The results are shown in Figure 8. The complete concept yields max $( | D _ { I } | , | D _ { P } | ) = 7 . 6 1$ , and the jackknife distribution has median 7.61, so for fifteen of the sixteen assigned papers the deletion changes the measured effect by less than ten percent. One document, Cohn’s 1904 alternative electrodynamics of moving systems, reduces the effect to 2.43, or 32% of the full value; every other single deletion leaves it essentially unchanged. The effect is therefore not an artifact of one assigned document, although the concept is not entirely insensitive to its composition.

![](images/b7f45a37e8eacf8c965d9daf5fc3a703634bba8c78eacd02d5dfb8fd501aaf42.jpg)  
Figure 6. Random-removal null on the joint $( D _ { I } , D _ { P } )$ plane for special relativity (5,000 draws of 16 temporally-matched randompaper sets, blue). The observed point at $t ^ { \hat { * } } = 1 \dot { 9 } 0 2$ is denoted with a red star.

Look-elsewhere effect. Finally, because the pivot year is determined through a scan over candidate concepts and years, the observed maximum must be assessed against the corresponding look-elsewhere effect (Gross & Vitells, 2010). This is because the concept pivot-year grid scans $1 0 \times 1 1 = 1 1 0$ cells and reports the global maximum, with a plausible concern being that with 110 opportunities to find a large D by chance, the probability of observing a spurious maximum is inflated relative to a single hypothesis test. We estimate this by randomly permuting concept labels 5,000 times, as in the scrambled-assignment null, recomputing the complete concept pivot-year grid for each permutation, and recording the maximum joint standardized response $\| D \| _ { 2 } = \sqrt { \bar { D _ { I } ^ { 2 } } + D _ { P } ^ { 2 } }$ across all grid cells.

This produces a null distribution of “the largest joint signal one would observe anywhere in the grid by chance”, and is shown for the special relativity case in Figure 9. The global p-value is then defined as the fraction of permutations whose maximum equals or exceeds the observed maximum. For the special relativity case, this procedure yields $p _ { \mathrm { L E E } } = 8 { \cdot } 1 0 ^ { - 4 }$ Thus, the observed signal remains significant even after accounting for the multiplicity introduced by the concept pivot-year scan.

![](images/9097a57ea1aa4f0bf25b4ebace94b7f6025372bac954e0da101f4d0520b5ae62.jpg)  
Figure 7. Scrambled-assignment null on the joint $( D _ { I } , D _ { P } )$ plane for special relativity (5,000 label permutations, blue). The observed point at $t ^ { * } = 1 9 0 2$ is likewise denoted with a red star.

These validation tests demonstrate that the special relativity signal is statistically significant, concept-specific, and robust against individual influential publications. We next examine the robustness of the framework with respect to the document-assignment procedure.

## 3.3. Robustness of the document assignment

The document-assignment procedure introduced in Section 2.1 requires an operating point controlling the balance between assignment efficiency and semantic contamination. Lower thresholds admit more documents at the expense of increased ambiguity between neighboring concepts, whereas higher thresholds improve semantic purity while discarding genuinely relevant documents.

For a document d with cosine similarities $\{ s _ { d } ( c ) \} _ { c \in \mathcal { C } }$ to the concept anchors, the assignment margin is the gap between its best and second-best concept,

$$
m _ { d } = \operatorname* { m a x } _ { c \in \mathcal { C } } s _ { d } ( c ) - \operatorname* { m a x } _ { c \neq c ^ { * } ( d ) } s _ { d } ( c ) , \qquad c ^ { * } ( d ) = \arg \operatorname* { m a x } _ { c \in \mathcal { C } } s _ { d } ( c ) ,\tag{10}
$$

so a small $m _ { d }$ marks a document lying close to the boundary between two concepts.

The simplest criterion admits a document when $m _ { d }$ exceeds a single threshold applied uniformly to all concepts. Such a threshold treats all concepts identically, ignoring that some concepts have close neighbors in embedding space, and so require a larger margin to separate, while others are isolated and admit a smaller margin without contamination. The framework therefore adopts a confusion-aware margin:

![](images/1b0f1a460e29706636f37123d5b1d42cb0dd8e6e9fd8c55e476c197a04e7d7ab.jpg)  
Figure 8. Leave-one-out jackknife for special relativity. Each bar deletes one paper from the corpus and recomputes both the baseline and the ablated geometry over the remaining documents. The corresponding bar length shows the standardized response the analysis would have reported had that paper never been collected. The dashed line marks the full-concept value max $\ [ { D } _ { I } | , | { D } _ { P } | ) =$ 7.61 at $t ^ { * } = 1 9 0 2$

$$
\tau ( c ) = f _ { \mathrm { k n e e } } \left( 1 - s _ { \mathrm { n n } } ( c ) \right) ,\tag{11}
$$

where $\begin{array} { r } { s _ { \mathrm { n n } } ( c ) = \operatorname* { m a x } _ { c ^ { \prime } \neq c } \hat { v } _ { c } \cdot \hat { v } _ { c ^ { \prime } } } \end{array}$ is the cosine similarity between the unit-normalized key-idea anchor of concept c and that of its nearest neighboring concept, so that $1 - s _ { \mathrm { n n } } ( c )$ measures the angular separation between a concept and its closest competitor.

A document is admitted to concept c when both conditions hold: its margin satisfies $m _ { d } \geq \tau ( c )$ , and its similarity to the concept anchor exceeds a per-concept baseline set at the sixtieth percentile of the similarity distribution of all documents whose nearest anchor is c. The percentile is fixed a priori and is not tuned. Documents failing either condition are labeled “no match” and excluded from the geometric analysis.

The operating fraction $f _ { \mathrm { k n e e } }$ is determined automatically using the Kneedle algorithm (Satopaa et al., 2011), which identifies the knee of the max D response curve obtained by scanning candidate operating points. The scan is performed on a grid of candidate operating fractions spanning $f \in [ 0 . 0 5 , 0 . 5 0 ]$ , identical across encoders within a given case, and evaluated at the pivot year of the primary model, so that the threshold of every encoder in a given case is calibrated at a common reference year. The pivot year at which each encoder is subsequently evaluated is determined separately, by the procedure of Section 5.

![](images/156f43c82fddbe8c7a06ee0ebc795d3a8697132cf6114ba63ea70ffbabb3706e.jpg)  
Figure 9. Null distribution of the global maximum of the 2D magnitude $\| D \| _ { 2 } = \sqrt { D _ { I } ^ { 2 } + D _ { P } ^ { 2 } }$ observed across the full $1 0 \times 1 1$ grid when concept labels are randomly permuted $( N = 5 0 0 0 )$ . The red dashed line corresponds to the observed SR signal $( \left. D \right. _ { 2 } =$ 7.62).

The operating fraction is selected on the target concept’s response curve. The selected fraction for every case study is reported in Table 4, and the statistical tests of Section 2.4 are evaluated at the fixed operating point, where they assess the significance of the perturbation magnitude relative to the null distributions constructed at that same point.

For the special relativity benchmark, the target retains the highest rank over a plateau around the selected operating point. The response curve exhibits its knee at $f _ { \mathrm { k n e e } } = 0 . 2 5$ $( D _ { I } = 7 . 6 1 )$ , while special relativity remains the highestranked concept throughout the interval $f \in \ [ 0 . 2 2 , 0 . 3 0 ]$ The effect decreases when the threshold becomes sufficiently restrictive to over-prune the assigned documents or sufficiently permissive to introduce contamination from neighboring concepts. These results indicate that the identification of special relativity as the dominant concept is stable against the precise choice of assignment threshold, although the magnitude of the effect varies across the plateau and decreases sharply once the threshold prunes the assigned set to a few documents.

The robustness of the assignment procedure across all historical case studies is examined in Section 4.

## 4. Historical validation across scientific revolutions

The special relativity case study demonstrates that counterfactual ablation can detect a well-established conceptual reorganization and that the resulting signal is statistically robust. An important question, however, is whether this behavior is specific to a single historical example or reflects a more general property of conceptual change.

To investigate this question, we apply the same framework to four additional historical case studies: Godel’s incom-¨ pleteness theorems, the Higgs mechanism, the deep learning revolution, and the attention mechanism / transformer architecture. Together, these examples encompass conceptual reorganizations spanning different scientific disciplines, historical contexts, and patterns of development, providing a broader test of the proposed framework.

Each corpus was assembled from Crossref (Hendricks et al., 2020), zbMATH Open (Schubotz & Teschke, 2021), and arXiv (Ginsparg, 1994), embedded with e5-base, and analyzed using the same pipeline introduced in Section 2, including document embedding, concept assignment, counterfactual ablation, and statistical validation. Only corpusspecific inputs (document collections and concept definitions) differ between cases. Per-case corpus statistics, assignment rates, and notes on boundary contamination are summarized in Table 3.

We first examine whether the standardized response consistently identifies the target conceptual reorganization across these independent historical case studies. We then evaluate whether the corresponding statistical and methodological validation tests exhibit the same behavior observed for the special relativity benchmark.

## 4.1. Generalization across historical case studies

Figure 10 and Table 4 summarize the Counterfactual standardized responses evaluated at the data-driven pivot year identified for each historical case study. In four of the five cases, the target concept produces the largest counterfactual geometric perturbation among the candidate concepts, indicating that the proposed metric consistently identifies the principal conceptual reorganization. The attention mechanism constitutes the only exception, ranking fourth within the broader and rapidly evolving deep-learning literature.

The full concept  pivot-year grids underlying these rankings are shown in Figure 11, which displays max $\left( | D _ { I } | , | D _ { P } | \right)$ for every combination of ablated concept and candidate pivot year in all five case studies. Neither geometry observable ranks the target concept first in every case, so the two are combined through the same statistic used for the rankings in Table 4 rather than selected case by case.

The two observables divide the five cases between them. Special relativity and the attention mechanism are dominated by the total-inertia response, whose positive sign means the target contributes more to the spread of the concept space after the pivot than before. Godel incompleteness,¨ the Higgs mechanism, and deep learning are dominated instead by the pairwise-distance response, whose sign measures whether the target became more or less distinguishable from the concepts around it: removing a concept lowers the mean pairwise distance when that concept sits farther from the others than they sit from each other, so a positive value means the target grew more distinct from its context across the pivot and a negative value that it grew less distinct. Godel incompleteness is positive on this observable, and the¨ Higgs mechanism and deep learning are negative, indicating targets that became progressively harder to separate from the surrounding literature as that literature expanded around them.

Higgs mechanism (pivot 1959). The Higgs mechanism attains the largest nominal standardized response of all case studies, ranking first with max $| D | = 1 4 . 5 7$ from a cluster of 108 assigned papers. This magnitude is an artifact of pre-pivot sparsity. The data-driven pivot leaves only four pre-pivot yearly observations, and a single assigned document supplies the concept centroid in all four, so the pre-pivot perturbation is nearly constant, and the pooled variance entering the standardized response approaches zero. Removing that document reduces the response at 1959 to 1.24 and moves the target from first to eighth among the ten candidate concepts, leaving the target row flat across the entire pivot range with no interior maximum. That document is a 1956 paper on cosmic time in general relativity, historically unrelated to electroweak symmetry breaking, admitted because the abstract description of a scalar field acquiring a background value and thereby reducing a symmetry fits both constructions. We therefore report the Higgs case as a diagnostic of unsupervised document assignment rather than as supporting evidence.

Godel’s incompleteness theorems (pivot 1937).¨ Godel’s¨ incompleteness theorems also rank first among the candidate concepts, with a maximum standardized response of max $| D | = 2 . 5 0$ . The dominant contribution arises from the pairwise-distance response, consistent with a conceptual reorganization that primarily reshapes the relationships among neighboring areas of mathematical logic. The nearest competing concept is intuitionism, reflecting the close historical connection between these foundational developments.

Deep learning (pivot 2011). The deep learning revolution likewise ranks first, with max $| D | = 3 . 1 3$ , but differs qualitatively from the preceding examples. The target concept encompasses a much larger literature (1,044 assigned documents), and the surrounding context concepts also exhibit substantial geometric perturbations. The resulting signal therefore reflects a broad paradigm shift distributed across many interacting research directions rather than a sharply localized conceptual transition. Consistent with this interpretation, the next most strongly perturbed concepts are

special relativity (t∗: 1902)  
![](images/d2f779a8fe662d4b3141e2b5fd3e60c19b3d0d7b66e3f1dab1628a1c6e1fc56c.jpg)

higgs mechanism (t∗: 1959)  
![](images/a37eb0c1831b30dd612044832504a0c8e8b3ce74fcda351207c5245acc34cdcd.jpg)

Gödel incompleteness (t∗: 1937)  
![](images/179a62bdb0099223b3b6b207054b3810f72dfb4b1740a67b644b39a65faaa1fe.jpg)

![](images/822b07b5e50e6d35dc38c8cd087a961a68c5f5f296730dbbf3650e95b353c779.jpg)

attention mechanism (t∗: 2012)  
![](images/bf54325a08f3a86967434f2d6c8a536fa5f0b68a584d3a8b89a28c3e05ac9ad0.jpg)  
Figure 10. Counterfactual standardized-response rankings across the five historical case studies: (a) special relativity, (b) Higgs mechanism, (c) Godel incompleteness, (d) deep learning, (e) attention mechanism. Red bars mark the target concept; blue bars are context¨ concepts. Stars indicate statistical significance $( ^ { * } p < \bar { 0 . } 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1 )$ from the random-removal null. Zero-height bars indicate concepts with no papers assigned under the confusion-aware margin; no ablation effect is computable for these concepts.

(a) Special relativity  
![](images/bdd3a5fcf5e5dd03355e134e51048c911cc71d5f2b1b7bc6d6daa38d513513f0.jpg)

(c) Gödel incompleteness  
![](images/0b208f26b430c1fc96f71201c1ed00cf8e7463c31d5a77f9113ef9efb1b624bf.jpg)

(b) Higgs mechanism  
![](images/24d8c083723efa1b4a04b049ee23851db8309ce75c5dcf568cae598060ee98e7.jpg)

(d) Deep learning  
![](images/de0f1730f8884c0ae10573cff5eb9dbd4e33034f73995ba0f53a49a27cf0a624.jpg)

(e) Attention mechanism  
![](images/6774c394011fd23cf6ccec2f75853e9717076d68ddbfbdd08fd2c8513bc5d424.jpg)  
Figure 11. Concept $\times$ pivot-year grids for the five historical case studies: (a) special relativity, (b) Higgs mechanism, (c) Godel¨ incompleteness, (d) deep learning, (e) attention mechanism. Each cell reports $\mathrm { n a x } ( | D _ { I } | , | D _ { P } | )$ for ablating the row ${ \bf \ddot { s } }$ concept and splitting pre- versus post-pivot at the column’s year, the same ranking statistic listed in Table 4. The dashed line marks the data-driven pivot $t ^ { \bar { * } }$ and the star marks the target concept at $t ^ { * }$ . Rows are ordered by their value at $t ^ { * }$ , so the ranking reads from top to bottom along the dashed line, and the target concept’s label is set in bold. Color spans each panel’s full grid, so cells at other years may exceed the starred cell without affecting the reported ranking, which compares concepts within the $t ^ { * }$ column only. The inset in each pane shows $\operatorname* { m a x } ( | D _ { I } | , | D _ { P } | )$ along the target concept’s row against pivot year, over the same year range as the panel axis beneath it, with the endpoints of that range labelled by their final two digits; the orange marker is the argmax that defines $t ^ { * }$

Data-driven pivot $t ^ { * }$   
Target concept at $t ^ { * }$   
Inset: max $\left( | D _ { I } | , | D _ { P } | \right)$ along the target row vs. pivot year Inset argmax (fixes t∗)

Table 3. Cross-case corpus statistics. r is the rolling-window radius in years (each window spans year $\pm r ) ;$ smaller corpora use $r = 2$ to accumulate sufficient papers per window, while larger corpora use $r = 1$ for finer temporal resolution. Target = number of corpus documents assigned to the target concept; Target $\% = \mathrm { t a r g e t }$ papers as a fraction of total corpus documents. Margin-excl. = fraction of all corpus documents that clear the similarity floor but are nonetheless excluded because no concept wins the confusion-aware margin. Document counts refer to the analysis window shown; for deep learning the underlying corpus extends to 2020, and the additional documents are excluded from the geometry so that the case remains disjoint from the transformer era examined in the attention case.
<table><tr><td>Case study</td><td>Documents</td><td>Time span</td><td>Assigned</td><td>Target</td><td>Target  $\%$ </td><td>Margin-excl.</td><td> $r$ </td></tr><tr><td>Special relativity</td><td>2,314</td><td>1880-1920</td><td>62 (3%)</td><td>16</td><td>0.7%</td><td>37.5%</td><td>2</td></tr><tr><td>Gödel incomp.</td><td>599</td><td>1900-1970</td><td>101 (17%)</td><td>34</td><td>5.7%</td><td>23.4%</td><td>2</td></tr><tr><td>Deep learning</td><td>16,152</td><td>2005-2018</td><td>4,053 (25%)</td><td>1,044</td><td>6.5%</td><td>14.9%</td><td>1</td></tr><tr><td>Higgs mechanism</td><td>27,591</td><td>1955-1980</td><td>746 (3%)</td><td>108</td><td>0.4%</td><td>37.3%</td><td>2</td></tr><tr><td>Attention mech.</td><td>37,598</td><td>2005-2023</td><td>2,715 (7%)</td><td>63</td><td>0.2%</td><td>32.8%</td><td>1</td></tr></table>

Table 4. Counterfactual standardized responses, confusion fractions, and null-test significance for the five historical case studies. $t ^ { * }$ denotes the target concept’s argmax pivot, while $f _ { \mathrm { k n e e } }$ is the per-case confusion fraction identified under e5-base. $D _ { I }$ and $D _ { P }$ are Cohen’s D evaluated on the total-inertia and mean-pairwise-distance geometry observables at $t ^ { * } .$ . Random removal and scrambled assignment report the co-dominance tail $p _ { \mathrm { c o } } .$ , Equation (7), on the $( D _ { I } , D _ { P } )$ plane using 5,000 permutations each. The look-elsewhere effect (LEE) reports the global tail $p \textmd { - }$ value of the $\| D \| _ { 2 }$ grid maximum, $\sqrt { D _ { I } ^ { 2 } + D _ { P } ^ { 2 } }$ , over all concept pivot-year cells, also using 5,000 permutations. Papers denotes the number of corpus documents assigned to the target concept.
<table><tr><td rowspan="2">Case study</td><td rowspan="2"> $t ^ { * }$ </td><td rowspan="2"> $f _ { \mathrm { k n e e } }$ </td><td rowspan="2"> $D _ { I }$ </td><td rowspan="2"> $D _ { P }$ </td><td colspan="2"> $p _ { \mathrm { c o } }$ </td><td rowspan="2"> $\mathrm { L E E } p$ </td><td rowspan="2">Papers</td></tr><tr><td>Random removal</td><td>Scrambled assign.</td></tr><tr><td>Special relativity</td><td>1902</td><td>0.25</td><td>+7.61</td><td>-0.36</td><td> $2 . 0 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 \times 1 0 ^ { - 4 }$ </td><td> $8 . 0 \times 1 0 ^ { - 4 }$ </td><td>16</td></tr><tr><td>Higgs mechanism</td><td>1959</td><td>0.30</td><td>-3.25</td><td>-14.57</td><td> $2 . 0 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 \times 1 0 ^ { - 2 }$ </td><td>108</td></tr><tr><td>Gödel incompleteness</td><td>1937</td><td>0.10</td><td>+0.50</td><td>+2.50</td><td> $2 . 0 \times 1 0 ^ { - 4 }$ </td><td> $6 . 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 \times 1 0 ^ { - 1 }$ </td><td>34</td></tr><tr><td>Deep learning</td><td>2011</td><td>0.12</td><td>-1.55</td><td>-3.13</td><td> $2 . 3 \times 1 0 ^ { - 2 }$ </td><td> $2 . 0 \times 1 0 ^ { - 4 }$ </td><td> $8 . 8 \times 1 0 ^ { - 2 }$ </td><td>1,044</td></tr><tr><td>Attention mechanism</td><td>2012</td><td>0.25</td><td>+1.64</td><td>+0.90</td><td> $8 . 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td><td> $9 . 7 \times 1 0 ^ { - 1 }$ </td><td>63</td></tr></table>

graphical models and convex optimization $( D = 2 . 8 9$ for both), major machine-learning paradigms that were progressively supplanted during the deep learning revolution.

Attention mechanism (pivot 2012). The attention mechanism provides a more demanding test of the framework by asking whether a localized conceptual innovation can be resolved within an ongoing scientific revolution. To study the attention mechanism, the machine-learning corpus was extended through 2023 to include the transformer era (Vaswani et al., 2017).

Although the target concept, represented by 63 assigned papers, ranks fourth rather than first, it produces a statistically significant onset signal under both null tests, demonstrating that the standardized response retains sensitivity to individual conceptual developments embedded within a much broader transformation of the machine-learning landscape.

Comparison with surrounding concepts. The relative importance of the target concept is summarized in Figure 12, which compares the target standardized response with the mean standardized response of the surrounding context concepts for each historical case. Special relativity exhibits pronounced target dominance, and Godel incom-¨ pleteness a weaker but unambiguous one, indicating that the principal conceptual reorganization is strongly localized within the target concept. The Higgs mechanism shows the largest nominal separation of all cases, but for the reason given above this reflects the collapse of the pre-pivot variance rather than localization. Deep learning remains target-dominated but with a substantially smaller separation from the surrounding concepts, while the attention mechanism lies close to parity, reflecting its emergence within an already rapidly evolving research field.

Summary across historical case studies. Taken together, the five historical case studies show that the proposed framework can be applied across scientific disciplines and historical contexts. The framework consistently identifies conceptual reorganizations in four independent historical revolutions spanning physics, mathematics, and artificial intelligence, while also detecting a significant signal for the attention mechanism despite its emergence within a much broader ongoing transformation. These results demonstrate that counterfactual perturbations of embedding geometry provide a general quantitative measure of conceptual reorganization across diverse scientific domains and historical contexts.

![](images/c581143feaa875bd24e3c184b83ad9114ca5f9a75d999e8b2d6b59049e8646e6.jpg)  
Figure 12. Target-concept standardized response max $\left( | D _ { I } | , | D _ { P } | \right)$ at the data-driven pivot year $t ^ { * }$ versus the mean of the same statistic over the nine surrounding context concepts, for each historical case. Horizontal bars show one standard deviation of the context-concept responses. The dashed line is $y = x \mathrm { : }$ cases above it have the conceptual reorganization concentrated in the target concept, cases near or below it show diffuse reorganization.

## 4.2. Unified validation across historical case studies

The statistical and methodological validation procedures introduced in Sections 2 and 3 were applied unchanged to all five historical case studies. These complementary tests assess the statistical significance, robustness, and temporal localization of the detected conceptual reorganizations.

The results of the statistical validation for the five test cases are summarized in Table 4. Under both the random-removal and scrambled-assignment tests, all five historical case studies meet the significance criterion of $p \ < \ 0 . 0 5$ defined in Section 2, confirming that the detected reorganizations depend on the specific conceptual organization of the corpus rather than on random document removal or arbitrary concept assignments. The leave-one-out analyses show that the measured responses are generally stable against individualdocument removal, while also exposing an important failure mode. In the Higgs case, deletion of a single semantically misassigned pre-pivot document substantially reduces the standardized response, demonstrating that sparse pre-pivot assignments can generate artificially large values through variance collapse. This example illustrates the importance of the complementary validation procedures rather than weakening their role.

Applying the look-elsewhere test to the complete concept pivot-year scan, the target concept supplies the global maximum for special relativity and the Higgs mechanism, whereas for Godel’s incompleteness theorems, deep learn-¨ ing, and the attention mechanism the largest perturbation in the grid is attained by a contextual concept and the global tail probability is not small. For the Higgs mechanism the global maximum inherits the pre-pivot variance collapse discussed above and does not carry independent evidential weight.

The methodological robustness studies likewise generalize across the historical case studies. Per-case, and across encoders, assignment thresholds derived using the confusionaware margin span the range $f _ { \mathrm { k n e e } } \in [ 0 . 0 5 , 0 . 4 5 ]$ , reflecting differences in local concept-space density rather than arbitrary parameter choices. The resulting operating points and assignment statistics are summarized in Table 4.

Taken together, these validation studies show that the counterfactual framework can yield statistically significant and methodologically robust responses across diverse historical settings, while also identifying failure modes associated with sparse or contaminated document assignments.

## 5. Robustness across embedding models

The historical validation presented in Sections 3 and 4 demonstrates that the proposed metric consistently detects conceptual reorganizations across multiple scientific domains. A remaining question is whether these conclusions depend on the particular embedding representation used to construct the document geometry. To investigate this, we repeat the complete analysis pipeline using the five embedding models introduced in Table 1. Because embedding models differ substantially in architecture, training objectives, and corpus coverage, agreement across models provides evidence that the observed geometric signatures reflect underlying conceptual organization rather than properties of a particular encoder.

## 5.1. Dependence on embedding representation

Figure 13 and Table 5 summarize the target concept rank and counterfactual standardized response obtained for every combination of historical case study and embedding model. Each encoder is evaluated at the pivot year obtained by applying the same target-row argmax rule used for the primary model to that encoder’s own concept  pivot-year grid. Consequently the comparison tests not only whether independent encoders agree on the relative standing of the target concept, but also whether they independently localize the same transition year. Because standardized-response magnitudes depend on the geometry of the embedding space, the principal quantity of interest is the ranking of the target concept rather than the absolute value of D .

![](images/80825b7d408a7a236c12cea98d103ead267128ae82e576c571742a6a0da461ad.jpg)

![](images/e43e7f0b3c4a9949b9c86c485cccd378fced6171650cdd8dfeb216e04b361e29.jpg)  
No document assigned to the target concept: not measured  
Figure 13. Embedding model independence across all five case studies. For each encoder, the pivot year is obtained by applying the target-row argmax rule of Section 2.3 to that encoder’s own concept pivot-year grid. (a) Target concept rank (green = high, red = low), with the selected pivot year printed beneath each rank. (b) Standardized response max $( | D _ { I } | , | D _ { P } | )$ evaluated at the same per-encoder pivot year. Hatched cells mark a case where the encoder assigns no document to the target concept, so neither the rank nor the standardized response is defined: SPECTER2 does not separate the Godel concepts sufficiently for any document to satisfy both assignment criteria.¨

Across the five historical case studies, the principal conclusions remain largely stable across embedding models. For special relativity and the Higgs mechanism the target concept is ranked first by four and three of the five encoders, respectively, and for Godel incompleteness, four ¨ of five place it among the five strongest concepts, while SPECTER2 assigns no document to the target and therefore yields no measurement. Deep learning and the attention mechanism have greater spread across encoders, with two and one encoders respectively, ranking the target first; in these two cases the variation affects the relative ordering of competing concepts rather than the existence of a geometric signal.

Sources of inter-model variation. Residual differences between embedding models arise primarily from differences in training objectives, corpus coverage, and representational capacity. Therefore, we assign each embedding model its own optimized confusion-aware assignment threshold, reflecting differences in embedding-space density. This is essential because a threshold appropriate for one encoder may over- or under-prune document assignments for another. Citation-prediction models (SPECTER2) compress semantically distinct but co-cited papers into overlapping regions, and in the most homogeneous corpus (the Godel case) the¨ confusion-aware threshold admits no document at all to the target concept, so that the cell reports a null result rather than a measured effect. Low-dimensional distilled models (MiniLM) require tighter thresholds but produce strong signals once properly calibrated. Models trained on diverse web data (bge-m3) require case-specific thresholds that differ substantially from those optimized on the primary model. Multilingual models (e5-base, mxbai) preserve cross-lingual structure essential for historical corpora.

These results demonstrate that the proposed framework recovers the geometrical signal across encoders differing in architecture, training objective, and corpus coverage, but the concept ranking depends on the specific embedding representation used to construct the document geometry. Additionally, the localization of the transition year is not an artifact of the primary model: different encoders independently select pivot years that agree with the primary model exactly in about half of all cases and to within a few years in the large majority. The remaining differences between models may reflect differences in embedding objectives and semantic resolution. We therefore conclude that the quality of the semantic representation is itself an ingredient in measuring conceptual organization, and identifying which properties of an embedding space make conceptual reorganization measurable is a direction for future work.

Table 5. Target concept rank and standardized response across embedding models. The pivot year t∗ (shown per case) is derived from the concept pivot-year grid (Section 2.3); each model is evaluated at its own target-row argmax; the selected year is listed beside each entry. Each model additionally uses its own independently optimized document-assignment threshold.
<table><tr><td>Model</td><td colspan="3">Spec. Rel.</td><td colspan="3">Higgs</td><td colspan="3">Gödel</td><td colspan="3">Deep Learn.</td><td colspan="3">Attention</td></tr><tr><td></td><td>t*</td><td>Rank</td><td>|D|</td><td>t*</td><td>Rank</td><td>|D|</td><td>t*</td><td>Rank</td><td>|D|</td><td>t*</td><td>Rank</td><td>|D|</td><td>t*</td><td>Rank</td><td>|D|</td></tr><tr><td>e5-base</td><td>1902</td><td>#1/10</td><td>7.61</td><td>1959</td><td>#1/10</td><td>14.57</td><td>1937</td><td>#1/10</td><td>2.50</td><td>2011</td><td>#1/10</td><td>3.13</td><td>2012</td><td>#4/10</td><td>1.64</td></tr><tr><td>specter2</td><td>1902</td><td>#1/10</td><td>5.15</td><td>1964</td><td>#1/10</td><td>3.00</td><td></td><td></td><td></td><td>2009</td><td>#2/10</td><td>3.28</td><td>2011</td><td>#5/10</td><td>2.63</td></tr><tr><td>bge-m3</td><td>1909</td><td>#9/10</td><td>0.45</td><td>1960</td><td>#3/10</td><td>4.00</td><td>1934</td><td>#4/10</td><td>1.34</td><td>2008</td><td>#4/10</td><td>4.93</td><td>2012</td><td>#1/10</td><td>3.43</td></tr><tr><td>MiniLM</td><td>1902</td><td>#1/10</td><td>19.69</td><td>1960</td><td>#2/10</td><td>5.27</td><td>1935</td><td>#1/10</td><td>2.14</td><td>2010</td><td>#1/10</td><td>10.02</td><td>2012</td><td>#3/10</td><td>2.91</td></tr><tr><td>mxbai</td><td>1902</td><td>#1/10</td><td>11.44</td><td>1959</td><td>#1/10</td><td>8.19</td><td>1940</td><td>#5/10</td><td>1.32</td><td>2008</td><td>#5/10</td><td>3.06</td><td>2012</td><td>#2/10</td><td>3.65</td></tr></table>

## 6. Discussion

The results presented in this work suggest that embedding geometry can function as a quantitative measure of conceptual reorganization. Across the historical case studies considered here, concentrated conceptual breakthroughs consistently produce stronger geometric signatures than broader, more diffuse transitions, indicating that major conceptual reorganizations leave measurable imprints within document embedding spaces. These findings naturally raise several interpretive questions. What do these geometric signatures represent? What assumptions underlie their interpretation? And how should their magnitude be understood in the context of scientific change? The following discussion addresses these questions and places the proposed framework within the broader landscape of computational approaches for studying the evolution of scientific knowledge.

## 6.1. What the method detects

Interpreting the results obtained requires distinguishing between the birth of a concept and its subsequent institutionalization within the scientific community. The proposed framework measures the latter: the stage at which a concept becomes a structural organizing principle around which a research community begins to organize its work. In this sense, the framework measures when a concept reshapes the conceptual organization of a field, rather than when it first enters the scientific record. In Kuhnian terms, a new “paradigm” emerges (Kuhn, 2012).

The distinction is illustrated by the case of special relativity. Einstein’s founding 1905 paper (“Zur Elektrodynamik bewegter Korper”¨ ) is present in the corpus but is not assigned to the special relativity concept cluster. Its nearestconcept similarity falls below the threshold, since the paper is framed kinematically rather than in the electrodynamic vocabulary that characterizes the assigned cluster, and the assignment rule therefore admits no concept for it. The assigned documents are instead dominated by the subsequent consolidation of special relativity as an established conceptual framework, including Minkowski’s “Espace et temps” (1909) and Einstein’s own textbook Relativity: The Special and General Theory (1916). No single one of them carries the signal: the drop-one jackknife of Figure 8 shows that fifteen of the sixteen assigned papers change the measured effect by less than ten percent.

This behavior follows naturally from interpreting embedding geometry as an observable of conceptual organization. A single paper, however revolutionary, does not immediately reorganize a field’s embedding geometry. Reorganization occurs when a critical mass of researchers produces work organized around the new concept, creating a centroid whose position structurally depends on that concept. Consequently, the proposed framework cannot establish historical priority, i.e. who first introduced a concept, but it can identify when that concept became a structural organizing principle of the field.

An idea that is published but remains dormant for many years would likewise produce little or no geometric signature until it begins to reorganize the surrounding literature. From the perspective of the present framework, this is the expected behavior: the framework measures conceptual institutionalization rather than invention itself. Accordingly, the geometric signatures reported throughout this paper should be interpreted as signatures of conceptual reorganization rather than signatures of individual acts of invention. Questions concerning originality or novelty require different methods, such as language-model-based analyses of individual publications (Park et al., 2023). These approaches are complementary rather than competing. Embedding geometry therefore measures when a concept becomes a structural organizing principle of a field, rather than when it first appears in the historical record.

## 6.2. Relation to prior computational approaches

A variety of computational approaches have been developed to study how ideas emerge, evolve, and become established within scientific communities. Citation-based and bibliometric methods characterize scientific change through the structure of the citation record itself, quantifying how combinations of prior work relate to subsequent impact (Uzzi et al., 2013) and how the disruptiveness of published work has evolved over time (Park et al., 2023). Topic-evolution methods characterize how the prevalence and vocabulary of latent themes change over time. Dynamic topic models track topic-word distributions through time (Blei & Lafferty, 2006; Wang et al., 2008), while applications to scientific corpora recover disciplinary structure and the waxing and waning of research areas (Griffiths & Steyvers, 2004; Hall et al., 2008). Diachronic semantics instead focuses on changes in the meaning of individual words using aligned historical or contextualized embeddings (Hamilton et al., 2016; Giulianelli et al., 2020), as surveyed by Kutuzov et al. (2018). Finally, cultural evolution and idea diffusion approaches quantify the adoption and spread of ideas through measures such as n-gram frequencies, citation patterns, adopter populations, and co-word networks. Examples include culturomics (Michel et al., 2011), epidemiological models of idea diffusion (Bettencourt et al., 2006), phylomemetic analyses of scientific fields (Chavalarias & Cointet, 2013), and studies of how ideas propagate and are reinterpreted across communities (Keuchenius et al., 2021; Cheng et al., 2023).

The proposed framework differs from these approaches in the quantity that it seeks to measure. Rather than quantifying the prevalence of a topic, the semantic evolution of a word, or the diffusion of an idea through citations or adoption, we quantify the structural role that a concept plays within the geometry of a document embedding space. The framework measures the geometric perturbation produced when the documents associated with a concept are counterfactually removed from the corpus. This provides a citation-independent, counterfactual framework for studying conceptual institutionalization.

Among existing approaches, the closest conceptual antecedents are the work of Cheng et al. (2023), which distinguishes the introduction of an idea from its later integration into scientific practice, and Bettencourt et al. (2006), which treats community adoption as a measurable process. The methodological distinction of the present work lies in using embedding geometry and counterfactual ablation to quantify conceptual reorganization directly, without relying on citation networks or other external metadata.

## 6.3. Retrospective validation and circularity

A natural concern is that the framework is evaluated retrospectively using historically recognized conceptual reorganizations. We define concepts such as “special relativity”, select a time window spanning the relevant historical period, and construct a corpus that includes the known innovation. Could the observed geometric signatures simply reflect assumptions built into the analysis rather than genuine conceptual reorganization?

Several features of the validation framework mitigate this concern. First, the concept pivot-year grid evaluates all candidate concepts across all pivot years within the analysis window. If the results were merely a consequence of the predefined concept labels or historical window, one would expect multiple concept–year combinations to produce comparable signals. Instead, the special relativity 1902 combination emerges as a sharp global maximum, substantially stronger than any non-special relativity concept–year pair.

Second, the validation suite tests whether the observed signal depends on the correct association between papers and concepts rather than on generic structural properties of the corpus or pivot-year grid.

Third, the framework does not require prior knowledge of which concept or historical year will produce the strongest signal. Given only a set of candidate concepts and an analysis window, the concept  pivot-year grid identifies which concept and which pivot year exhibit the largest geometric perturbation. In this sense, the historical examples serve as validation cases rather than predetermined outcomes.

A residual limitation nevertheless remains. The definition of corpus boundaries, candidate concepts, and temporal coverage necessarily involves analyst choices that may influence the quantitative results. This limitation is common to corpusbased approaches more generally and is not specific to the counterfactual ablation framework proposed here.

The purpose of the historical case studies is therefore not to rediscover known scientific revolutions, but to evaluate whether embedding geometry behaves as a meaningful observable of conceptual reorganization.

## 6.4. Scope and interpretation

Throughout this paper we have argued that embedding geometry provides a quantitative observable of conceptual reorganization. Interpreting this observable requires distinguishing it from several related but distinct notions, including novelty (the introduction of a new idea), conceptual reorganization (changes in the organization of a field), community institutionalization (the adoption of a concept by a research community), and scientific importance or impact. Although these phenomena are often correlated, they are not equivalent.

The counterfactual ablation framework introduced here measures one specific quantity: the geometric perturbation produced when the documents associated with a concept are removed from the embedding space. A large perturbation indicates that the concept has become structurally important in organizing the surrounding literature. Such signatures provide evidence that a concept has contributed to a measurable conceptual reorganization of the field. They do not, however, establish that the concept was historically original, scientifically correct, or ultimately influential in the long term. A concept that later proves to be incorrect could nevertheless reorganize the scientific literature during the period in which it is actively investigated.

Accordingly, the claims of the present work should be interpreted as identifying geometric signatures consistent with conceptual reorganization rather than providing a universal measure of innovation. The proposed framework quantifies structural change in the conceptual organization of scientific literature. Questions concerning originality, scientific value, historical priority, or long-term impact require complementary observables and lie beyond the scope of the present framework.

## 6.5. Broader implications and future directions

The present work establishes embedding geometry as a quantitative observable of conceptual reorganization. While the historical case studies considered here provide evidence that major conceptual reorganizations leave measurable geometric signatures, they also suggest a broader research direction. Having established embedding geometry as a quantitative observable of conceptual reorganization, it becomes possible to investigate how conceptual organization evolves over time.

Understanding the temporal evolution of embedding geometry represents a natural next step toward developing quantitative theories of knowledge evolution. Such studies may provide new insights into the dynamics of conceptual organization, the evolution of scientific disciplines, and, more generally, the processes through which knowledge develops across diverse domains. Beyond science, the same framework may prove useful for investigating conceptual change in other text-rich disciplines, including technology, philosophy, law, economics, and the humanities.

We leave these directions for future investigation. Here, our objective has been to establish the framework itself. We hope that the framework introduced here provides a foundation upon which future quantitative studies of conceptual organization and its evolution can be built.

## 7. Conclusions

This work introduces embedding geometry as a quantitative observable of conceptual reorganization and demonstrates that major conceptual reorganizations leave measurable geometric signatures within document embedding spaces. Across diverse historical examples, the proposed framework can identify principal conceptual reorganizations and distinguish localized conceptual breakthroughs from broader distributed paradigm shifts. Importantly, it does so without relying on citation networks, expert annotation, or other external metadata.

More broadly, our results suggest that the latent geometric structure learned by modern embedding models captures meaningful aspects of the conceptual organization encoded in scientific literature. Compared to current computational approaches to understanding scientific change, embedding geometry provides a complementary geometric perspective for studying how concepts organize, evolve, and influence the development of scientific fields.

Viewed from this perspective, modern embedding spaces can be regarded not only as semantic representations, but also as scientific objects of study whose geometry can be measured, perturbed, and statistically analyzed.

Treating embedding geometry as a quantitative scientific observable also introduces important limitations. The proposed framework measures the structural reorganization associated with the community uptake of ideas rather than their intrinsic originality or scientific value. Its validity depends on the quality of the underlying embedding representation and on the assignment of documents to concepts. The validation studies presented here indicate that the principal conclusions are robust across multiple embedding models, assignment procedures, and statistical tests, with one instructive exception: in the Higgs mechanism case a single misassigned document supplies the entire pre-pivot pool, and the reported standardized response is an artifact of the resulting variance collapse rather than a measure of conceptual reorganization.

A particularly promising next step is the development of time-bounded embedding models, trained exclusively on literature available before a chosen historical date. Such models would eliminate information leakage from future developments, allowing conceptual reorganizations to be studied from the perspective of the scientific knowledge actually available at the time. Beyond providing a more stringent historical validation, time-bounded embeddings would enable the evolution of embedding geometry itself to be investigated, opening the possibility of tracking how conceptual representations emerge, mature, and reorganize scientific fields. Such studies could provide the basis for quantitative models of knowledge evolution.

The broader ambition motivating this line of research is to develop a quantitative understanding of how knowledge evolves. Whether systems capable of recognizing the geometric signatures of past conceptual reorganizations can ultimately help identify emerging scientific directions, generate new hypotheses, or assist scientific discovery remains an open question. The retrospective evidence assembled here is a prerequisite for addressing that question rather than an answer to it: it establishes that embedding representations of scientific literature contain measurable geometric structure associated with conceptual reorganization, providing the foundation upon which future studies of knowledge evolution could build.

One particularly intriguing possibility is that such a framework could detect the “quiet revolutions” described by Kuhn (Kuhn, 2012): gradual conceptual reorganizations that unfold over decades before becoming widely recognized as transformative scientific advances. Understanding these dynamics may ultimately contribute to AI systems that not only analyze the history of science, but also assist scientists in understanding, and ultimately shaping, the future evolution of knowledge.

## Acknowledgements

We thank Malcolm Slaney for reviewing an earlier version of this manuscript and for valuable feedback. This work used the resources of the SLAC Shared Science Data Facility (S3DF) at SLAC National Accelerator Laboratory. SLAC is operated by Stanford University for the U.S. Department of Energy’s Office of Science.

## Code and data availability

The full analysis pipeline is released at https: //github.com/Mapping-Innovation-Lab/ geometric-signatures. Accompanying material is available on the paper website at https: //mapping-innovation-lab.github.io/ geometric-signatures-companion-website/.

## References

Bettencourt, L. M. A., Cintron-Arias, A., Kaiser, D. I., and´ Castillo-Chavez, C. The power of a good idea: Quantita-´ tive modeling of the spread of ideas from epidemiological models. Physica A: Statistical Mechanics and its Applications, 364:513–536, 2006.

Blei, D. M. and Lafferty, J. D. Dynamic topic models. In Proceedings ofthe 23rd International Conference on Machine Learning (ICML), pp. 113–120, 2006.

Chavalarias, D. and Cointet, J.-P. Phylomemetic patterns in science evolution—the rise and fall of scientific fields. PLOS ONE, 8(2):e54847, 2013.

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.

Cheng, M., Smith, D. S., Ren, X., Cao, H., Smith, S., and McFarland, D. A. How new ideas diffuse in science. American Sociological Review, 88(3):522–561, 2023.

Cohen, J. Statistical Power Analysis for the Behavioral

Sciences. Lawrence Erlbaum Associates, 2nd edition, 1988.

Efron, B. Estimating the error rate of a prediction rule: Improvement on cross-validation. Journal ofthe American Statistical Association, 78(382):316–331, 1983.

Einstein, A. Zur elektrodynamik bewegter korper.¨ Annalen der Physik, 322(10):891–921, 1905.

Ethayarajh, K. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 55–65, 2019.

Fortunato, S., Bergstrom, C. T., Borner, K., Evans, J. A.,¨ Helbing, D., Milojevic, S., Petersen, A. M., Radicchi, F.,´ Sinatra, R., Uzzi, B., et al. Science of science. Science, 359(6379):eaao0185, 2018.

Ginsparg, P. First steps towards electronic research communication. Computers in Physics, 8(4):390–396, 1994.

Giulianelli, M., Del Tredici, M., and Fernandez, R.´ Analysing lexical semantic change with contextualised word representations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 3960–3973, 2020.

Godel, K.¨ Uber formal unentscheidbare s<sup>¨</sup> atze der Principia¨ Mathematica und verwandter systeme I. Monatsheftefur¨ Mathematik und Physik, 38:173–198, 1931.

Good, P. I. Permutation Tests: A Practical Guide to Resampling Methods for Testing Hypotheses. Springer, 2000.

Griffiths, T. L. and Steyvers, M. Finding scientific topics. Proceedings of the National Academy of Sciences, 101 (Suppl. 1):5228–5235, 2004.

Gross, E. and Vitells, O. Trial factors for the look elsewhere effect in high energy physics. The European Physical Journal C, 70:525–530, 2010.

Hall, D., Jurafsky, D., and Manning, C. D. Studying the history of ideas using topic models. In Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 363–371, 2008.

Hamilton, W. L., Leskovec, J., and Jurafsky, D. Diachronic word embeddings reveal statistical laws of semantic change. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 1489–1501, 2016.

Hendricks, G., Tkaczyk, D., Lin, J., and Feeney, P. Crossref: The sustainable source of community-owned scholarly metadata. Quantitative Science Studies, 1(1):414–427, 2020.

Higgs, P. W. Broken symmetries and the masses of gauge bosons. Physical Review Letters, 13(16):508–509, 1964.

Keuchenius, A., Tornberg, P., and Uitermark, J. Adoption¨ and adaptation: A computational case study of the spread of granovetter’s weak ties hypothesis. Social Networks, 66:10–25, 2021.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, volume 25, pp. 1097–1105, 2012.

Kuhn, T. S. The Structure ofScientific Revolutions. University of Chicago Press, Chicago, 1962.

Kuhn, T. S. The Structure of Scientific Revolutions. University of Chicago Press, Chicago, 4 edition, 2012. 50th Anniversary Edition, with an introductory essay by Ian Hacking.

Kutuzov, A., Øvrelid, L., Szymanski, T., and Velldal, E. Diachronic word embeddings and semantic shifts: A survey. In Proceedings ofthe 27th International Conference on Computational Linguistics (COLING), pp. 1384–1397, 2018.

Lee, S., Shakir, A., Koenig, D., and Lipp, J. Open source strikes bread - new fluffy embedding model. https://www.mixedbread.com/blog/ mxbai-embed-large-v1, March 2024.

Michel, J.-B., Shen, Y. K., Aiden, A. P., Veres, A., Gray, M. K., Google Books Team, Pickett, J. P., Hoiberg, D., Clancy, D., Norvig, P., Orwant, J., Pinker, S., Nowak, M. A., and Aiden, E. L. Quantitative analysis of culture using millions of digitized books. Science, 331(6014): 176–182, 2011.

Park, M., Leahey, E., and Funk, R. J. Papers and patents are becoming less disruptive over time. Nature, 613:138–144, 2023.

Reimers, N. and Gurevych, I. Sentence-BERT: Sentence embeddings using siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 3982–3992, 2019.

Satopaa, V., Albrecht, J., Irwin, D., and Raghavan, B. Finding a “kneedle” in a haystack: Detecting knee points

in system behavior. In 2011 31st International Conference on Distributed Computing Systems Workshops, pp. 166–171. IEEE, 2011.

Schubotz, M. and Teschke, O. zbMATH Open: Towards standardized machine interfaces to expose bibliographic metadata. European Mathematical Society Magazine, (119):50–53, 2021.

Singh, A., D’Arcy, M., Cohan, A., Downey, D., and Feldman, S. SciRepEval: A multi-format benchmark for scientific document representations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Subramanian, A., Tamayo, P., et al. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. Proceedings of the National Academy of Sciences, 102(43):15545–15550, 2005.

Uzzi, B., Mukherjee, S., Stringer, M., and Jones, B. Atypical combinations and scientific impact. Science, 342(6157): 468–472, 2013.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pp. 5998–6008, 2017.

Wang, C., Blei, D., and Heckerman, D. Continuous time dynamic topic models. In Proceedings ofthe 24th Conference on Uncertainty in Artificial Intelligence (UAI), pp. 579–586, 2008.

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., and Wei, F. Multilingual E5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672, 2024.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. MiniLM: Deep self-attention distillation for taskagnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems, volume 33, pp. 5776–5788, 2020.