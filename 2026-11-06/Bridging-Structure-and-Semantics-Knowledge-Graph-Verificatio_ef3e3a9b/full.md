# Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment

Ziang Cui Southeast University 13776094815@163.com

## Abstract

Semantic accuracy is an important dimension of knowledge graph quality. Graph-based scoring methods exploit relational structure, while textual representations allow classifiers to draw on knowledge encoded by pre-trained language models (PLMs). We present GenVer, a generative verifier that scores triples using both their serialized form and generated natural lan guage representations (NLRs). A triple–text alignment procedure selects training pairs for a sequence-to-sequence generator; a BERTbased classifier then combines features from each triple and its generated NLR. On four FactFB variants, constructed from FB15k using perturbed negative triples, GenVer obtains the highest classification accuracy among the evaluated baselines, ranging from 88.56% to 91.52%. The experiments suggest that generated representations can help triple classification, while the case analysis also exposes semantic drift and high-confidence errors.

## 1 Introduction

Knowledge graphs (KGs) represent entities and their relations as structured assertions (Google, 2012). Resources such as Freebase (Bollacker et al., 2008), DBpedia (Auer et al., 2007), and Wikidata (Vrandeciˇ c and Krötzsch ´ , 2014) support applications including social network analysis (Molokwu et al., 2021), recommendation (Zhao et al., 2019), and question answering (Chen et al., 2019). Errors in these resources can affect downstream applications, motivating the assessment of KG quality (Hogan et al., 2022).

This work focuses on triple-level semantic accuracy: whether the assertion expressed by a triple is correct. We operationalize this objective as binary classification on a constructed dataset and assign each triple a score in [0, 1]. A higher score indicates a stronger prediction of the positive class under the training distribution. Such a score is not, by itself, a calibrated probability that the assertion is true in the real world.

Several established scoring methods learn entity and relation embeddings from graph structure (Bordes et al., 2013; Lacroix et al., 2018; Trouillon et al., 2016; Zhou et al., 2021). Text-based verification methods additionally retrieve natural language evidence (Gerber et al., 2015; Syed et al., 2018). In this paper, a natural language representation (NLR) is a textual expression associated with a triple. We searched a Wikipedia dump<sup>1</sup> for NLRs of FB15k triples. Figure 1(a) summarizes the results: many triples had no retrieved representation under our search procedure. This observation concerns the coverage of the restricted search described in Appendix A.2; it does not establish that relevant evidence is absent from Wikipedia or the Web.

GenVer generates an NLR for each input triple so that a classifier can use a textual representation even when the search procedure does not retrieve one. PLMs learn contextual representations from large text corpora (Devlin et al., 2019) and can encode factual associations (Petroni et al., 2019). We use a sequence-to-sequence model to verbalize triples, then jointly encode the triple and its generated NLR for classification. Figure 1(b) illustrates this idea. Generated NLRs are model-produced features, not independent evidence for the truth of the input assertion; their faithfulness is an important limitation of the approach.

## Our contributions are:

1. A two-stage framework that combines serialized triples and generated NLRs for triple classification.

2. A triple–text alignment procedure with four ranking features and two parameter-setting strategies for selecting generator training pairs.

3. A comparison on four FactFB variants, together with examples that illustrate useful representations and failures of generation or classification.

![](images/2624917c1bb8426570cdb371ec0ad58b1ef84763078c6461c21dda002bf998f8.jpg)  
(a) Retrieval coverage.

![](images/2773ef9ce906d1b0b7818aa3930f866e0f931bda8ec8a5c2b757efd33d70bd43.jpg)  
(b) Triple-to-text classification.  
Figure 1: (a) Number of candidate NLRs retrieved per FB15k triple under the restricted search procedure. (b) GenVer uses generated text as a feature for triple classification.

## 2 Related Work

## 2.1 Knowledge Graph Verification

Methods for assessing triples broadly fall into two groups: representation learning and evidence retrieval.

Representation learning. These methods embed entities and relations in continuous vector spaces and learn scoring functions over triples. TransE (Bordes et al., 2013) models relations as translations, whereas DistMult (Yang et al., 2015) and ComplEx (Trouillon et al., 2016) use tensor factorization. KGTtm (Jia et al., 2019) combines three estimators to assess triple trustworthiness. These approaches score triples directly or through a downstream classifier; graph-only configurations do not directly use textual descriptions.

Evidence retrieval. DeFacto (Gerber et al., 2015) uses search engines to retrieve web evidence for statements generated from triples using BOA patterns (Gerber and Ngonga Ngomo, 2012). FactCheck (Syed et al., 2018) retrieves evidence from a static corpus and uses document features in a classifier to assess triples. Such methods introduce textual information into verification, but their coverage depends on the corpus and retrieval procedure. Missing or irrelevant evidence can limit their effectiveness.

## 2.2 Knowledge Graphs and Pre-trained Language Models

Work combining KGs and PLMs includes methods that use KGs to enhance language models and methods that use language models to process KGs.

KGs for PLMs. KgPLM (He et al., 2020) incorporates factual knowledge through generative and discriminative pre-training tasks. Entities as Experts (EaE) (Févry et al., 2020) learns representations for a fixed entity vocabulary and maintains an entity memory. FILM (Verga et al., 2021) integrates symbolic knowledge into a neural language model through a fact memory. TEKGEN (Agarwal et al., 2021) verbalizes Wikidata triples to construct a synthetic corpus for language model pre-training.

PLMs for KGs. KG-BERT (Yao et al., 2019) concatenates entity and relation names or descriptions and uses BERT to score triples for knowledge graph completion. SimKGC (Wang et al., 2022) uses textual information and contrastive learning with three types of negative examples to learn KG representations. GenVer also uses PLMs to process KGs, combining a serialized triple with a generated NLR for triple classification. Unlike retrieved evidence, the generated NLR is a model-produced feature and does not independently corroborate the triple.

## 3 Proposed Method

GenVer consists of an NLR generation module and a triple verification module (Figure 2). First, a triple–NLR alignment procedure selects textual representations to train a generator. The trained generator then produces an NLR for each triple to be classified. Finally, a BERT-based verification module jointly encodes the serialized triple and its generated NLR, extracts features from both, and outputs a classification score.

1. NLR generation  
![](images/28a102c8c960047d20cbc043ba39cf7b02f81aa0cb219a447bdf0646c2dc17fe.jpg)  
Figure 2: GenVer overview. Triple–NLR alignment supplies training pairs for the UniLM generator. At inference time, BERT jointly encodes each triple and its generated NLR; the two feature extractors feed a classifier. The stages are trained separately.

## 3.1 NLR Generation Module

The NLR generation module selects triple–NLR training pairs and uses them to train a sequence-tosequence model.

Triple–NLR alignment. Following the KG–text alignment approach of Agarwal et al. (2021), we retrieve candidate sentences containing mentions of a triple’s subject or object. Retrieval is restricted to the entity-associated Wikipedia pages described in Appendix A.2. A triple may match several sentences that do not faithfully express its relation. We therefore rank the candidates using four features: mention coverage (MS), sentence length (SL), relation relevance (RR), and pattern typicality (PT). All four features are computed with respect to the current triple.

Mention coverage (MS). This feature considers whether an NLR contains mentions of both the subject and object of a triple. The feature for sentence � is denoted as MS(�), where MS(�) = 1 if both mentions are present, and MS(�) = 0 otherwise.

Sentence length (SL). Sentence length, SL(�), is the number of tokens in sentence �. The priorbased ranking rule penalizes longer sentences to discourage redundant information.

Relation relevance (RR). RR(�) captures the relevance between a triple’s relation and its NLR using string matching:

$$
\mathrm { R R } ( S ) = \frac { \sum _ { r \in L } \sum _ { t \in S } \mathbf { 1 } [ \sin ( t , r ) > \tau ] } { | L | } ,\tag{1}
$$

where � is the sequence of relation-label tokens, sim(�, �) is the Levenshtein ratio, � is a similarity threshold, and 1[·] is the indicator function. The sum counts matching token pairs; the score is not constrained to [0, 1].

Pattern typicality (PT). To identify common linguistic patterns, the typicality score for each token � is calculated:

$$
\operatorname { t y p } ( t ) = { \frac { \sum _ { S \in N } \mathbf { 1 } [ t \in S ] } { | N | } } ,\tag{2}
$$

where N represents the set of candidate NLRs for a triple. The pattern typicality feature PT(�) sums these scores:

$$
\operatorname { P T } ( S ) = \sum _ { t \in S } { \mathrm { t y p } } ( t ) .\tag{3}
$$

NLR ranking score. We combine the four features into a ranking score QS:

$$
\mathrm { Q S } ( \mathbf { x } ) = \mathrm { s i g m o i d } \left( \sum _ { i = 1 } ^ { 4 } w _ { i } x _ { i } + b _ { s } \right) ,\tag{4}
$$

where $w _ { i }$ is the weight for each feature $x _ { i }$ , and $b _ { s }$ is a bias term. $x _ { 1 }$ through �<sub>4</sub> represent MS(�), SL(�), RR(�), and PT(�), respectively. We consider two parameter-setting strategies: a prior-based rule and supervised learning from labeled pairs.

For each triple with retrieved candidates, we select the NLR with the highest QS to form the generator training corpus. Section 4.2 compares the downstream classification results obtained with the two parameter-setting strategies.

NLR generation. We use UniLM (Dong et al., 2019) as the sequence-to-sequence generator. Each input triple is serialized as [CLS] followed by its subject, relation-label tokens, and object. Entities are added to the model vocabulary as individual tokens rather than represented by multi-token labels.

## 3.2 Triple Verification Module

BERT (Devlin et al., 2019) jointly encodes each triple and its generated NLR. The input consists of [CLS], the subject, relation-label tokens, the object, [SEP], and the NLR tokens, in that order. Two extractors then derive features from the contextualized token representations.

Unstructured feature extractor. Let $\mathbf { X } \in \mathbb { R } ^ { m \times d }$ contain the BERT representations of the � NLR tokens, with hidden dimension �. Following Kim (2014), a convolutional filter $\mathbf { W } \in \mathbb { R } ^ { h \times d }$ with window size ℎ computes

$$
c _ { i } = f ( \left. \mathbf { W } , \mathbf { X } _ { i : i + h - 1 } \right. + b ) ,\tag{5}
$$

where $\langle \cdot , \cdot \rangle$ denotes the elementwise inner product, � is a bias, and $f$ is an activation function. Global max pooling yields

$$
\hat { c } = \operatorname* { m a x } _ { i } c _ { i } .\tag{6}
$$

Filter banks with three window sizes produce vectors $\mathbf { z } _ { 1 } , \mathbf { z } _ { 2 }$ , and $\mathbf { z } _ { 3 }$ , which are concatenated as $\mathbf { h } = [ \mathbf { z } _ { 1 } ; \mathbf { z } _ { 2 } ; \mathbf { z } _ { 3 } ]$ . Let $ { \mathbf { p } } _ { \mathrm { s e p } }$ be the BERT representation of [SEP]. The unstructured feature is

$$
\mathbf { u } = \mathbf { W } _ { u } [ \mathbf { h } ; \mathbf { p } _ { \mathrm { s e p } } ] + \mathbf { b } _ { u } .\tag{7}
$$

Structured feature extractor. Inspired by TransE (Bordes et al., 2013), we use an energybased gate. Let $\mathbf { e } _ { s } , \mathbf { e } _ { p }$ , and $\mathbf { e } _ { o }$ be the subject, relation, and object representations from BERT. The energy � is:

$$
E = \| \mathbf { e } _ { s } + \mathbf { e } _ { p } - \mathbf { e } _ { o } \| _ { 2 } .\tag{8}
$$

The scalar gate � modulates the triple feature:

$$
a = \frac { 1 } { 1 + e ^ { \lambda ( \theta - E ) } } ,\tag{9}
$$

where � and � are trainable parameters. The trainable parameter � determines whether the gate increases or decreases with energy. The gate is a feature weight, not a truth probability. The final structured feature is $\mathbf { s } = a \mathbf { l }$ , where l is the [CLS] token representation. Here, “structured” refers to the subject–relation–object organization of the BERT input rather than a separate graph-neighborhood encoder.

Classification score. The two feature vectors are concatenated and projected as $\mathbf { t } = \mathbf { W } _ { t } \left[ \mathbf { u } ; \mathbf { s } \right] + \mathbf { b } _ { t }$ Here and above, semicolons denote concatenation. A final sigmoid produces the score

$$
o = \mathrm { s i g m o i d } ( \mathbf { w } _ { o } ^ { \top } \mathbf { t } + b _ { o } ) ,\tag{10}
$$

where $o \ \in \ [ 0 , 1 ]$ is a classifier output. Classification accuracy and the area under the receiver operating characteristic curve (AUROC) measure agreement with dataset labels, not probability calibration.

## 4 Experiments

## 4.1 Datasets

FactFB: We use FB15k (Bordes et al., 2013) as the source of positive triples and select its 2, 4, 8, and 16 most frequent relations to construct four FactFB variants. Negative triples are generated by randomly perturbing an entity or a relation in a positive triple. Perturbed triples that already occur in the knowledge graph are excluded from the negative examples. These variants, denoted FactFB , FactFB , FactFB , and Fact $\mathrm { \Delta \vec { \mathbf { \tau } } \vec { B } } _ { 1 6 }$ , differ in size and relation coverage but are not independent domains. Appendix A.1 gives dataset statistics. A perturbed triple is a constructed negative example; absence from a KG alone does not establish that an assertion is false in the real world.

Triple–NLR pairs. We apply the alignment procedure to a Wikipedia dump, using entity-page mappings to restrict retrieval (Appendix A.2). Two generator training corpora are constructed by setting the ranking parameters either from prior assumptions or from manually labeled pairs.

For the prior-based strategy, the default parameters in Eq. (4) are $w _ { 1 } = 2 . 2 2 4 , w _ { 2 } = - 0 . 0 8 8 , w _ { 3 } =$ $5 . 3 9 2 , w _ { 4 } = 3 . 8 0 3$ , and $b _ { s } = - 1 . 3 4 2$ as derived in Appendix A.3. We denote the resulting corpus by $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { S c o r e } }$

For the learned strategy, we manually label the candidate pairs: label 1 indicates that the NLR fully expresses the triple, and label 0 indicates that it does not. The annotation set contains 60 pairs for fitting and evaluating the ranker in Eq. (4). The ranker was fitted on 48 pairs and evaluated on the remaining 12 pairs (a 20% held-out split), with a reported accuracy of 100%. This small sample limits conclusions about generalization. We fit logistic regression using cross-entropy loss and use its positive-class probability as the ranking score. The learned parameters are $w _ { 1 } =$ 2.603, $w _ { 2 } = - 0 . 0 1 5 , w _ { 3 } = - 0 . 4 2 7 , w _ { 4 } = - 0 . 4 0 2 ,$ and $b _ { s } = - 1 . 2 1 8$ . This dataset, produced using the learned parameters, is denoted by $\mathrm { T } \mathrm { - N L R } _ { \mathrm { M L } }$

Table 1: Classification accuracy (%) using different triple– NLR pairs.
<table><tr><td>Corpus</td><td> $\mathrm { F a c t F B } _ { 2 }$ </td><td> $\mathrm { F a c t F B _ { 4 } }$ </td><td> $\mathrm { F a c t F B } _ { 8 }$ </td><td> ${ \mathrm { F a c t F B } } _ { 1 6 }$ </td></tr><tr><td> $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { S c o r e } }$ </td><td>86.12</td><td>88.28</td><td>86.55</td><td>88.35</td></tr><tr><td> $\mathrm { T } \mathrm { - N L R } _ { \mathrm { M L } }$ </td><td>91.52</td><td>90.67</td><td>88.56</td><td>90.51</td></tr></table>

## 4.2 Effect of Training-Pair Selection

We compare GenVer trained with the two triple– NLR corpora. Following KGTtm (Jia et al., 2019), we evaluate binary triple classification. The model settings are provided in Appendix A.4.

Table 1 reports higher classification accuracy for $\mathrm { T } \mathrm { - N L R } _ { \mathrm { M L } }$ than for $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { S c o r e } }$ on all four FactFB variants. This comparison changes the generator training pairs and measures the resulting downstream performance; it is not a direct evaluation of NLR faithfulness.

Figure 3 reports ROC curves and AUROC values for the two training-pair variants. An ROC curve plots the true-positive rate against the false-positive rate as the decision threshold changes; AUROC summarizes ranking performance across thresholds. The reported AUROC values are higher with $\mathrm { T } \mathrm { - N L R } _ { \mathrm { M L } }$ on each FactFB variant. These observations associate the training-pair selection strategy with downstream performance, without isolating which properties of the generated text account for the difference.

Appendix A.5 presents generated examples from the evaluated relations. These examples include omissions, unsupported additions, and changes to relation meaning, so shorter or more fluent outputs should not automatically be interpreted as more faithful representations.

The learned ranking configuration performs better in these experiments. The small annotation sample used to fit it, however, limits conclusions about the reliability of this selection procedure beyond the evaluated setting.

## 4.3 Comparison with Existing Methods

We compare GenVer with the graph embedding methods TransE (Bordes et al., 2013), ComplEx (Trouillon et al., 2016), DistMult (Yang et al., 2015), TransH (Wang et al., 2014), TransR (Lin et al., 2015), and TransD (Ji et al., 2015), using OpenKE (Han et al., 2018). We also include KGTtm (Jia et al., 2019), which combines several estimators for triple evaluation. $\mathrm { B E R T _ { \mathrm { t r i p l e } } }$ is an ablation that omits the generated NLR and its corresponding feature extractor. SimKGC (Wang et al., 2022) provides a text-based KG representation baseline. The training settings are listed in Appendix A.4. These comparisons concern the implemented configurations on FactFB, rather than an exhaustive comparison with current verification systems.

Table 2: Triple classification accuracy (%).
<table><tr><td>Methods</td><td> $\mathrm { F a c t F B } _ { 2 }$ </td><td> $\mathrm { F a c t F B _ { 4 } }$ </td><td> $\mathrm { F a c t F B _ { 8 } }$ </td><td> ${ \mathrm { F a c t F B } } _ { 1 6 }$ </td></tr><tr><td>GenVer</td><td>91.52</td><td>90.67</td><td>88.56</td><td>90.51</td></tr><tr><td>DistMult</td><td>81.63</td><td>79.24</td><td>81.17</td><td>83.33</td></tr><tr><td>ComplEx</td><td>84.46</td><td>89.68</td><td>87.20</td><td>89.83</td></tr><tr><td>TransE</td><td>87.39</td><td>86.20</td><td>85.27</td><td>85.63</td></tr><tr><td>TransH</td><td>88.73</td><td>85.24</td><td>83.80</td><td>83.31</td></tr><tr><td>TransR</td><td>84.43</td><td>69.98</td><td>62.28</td><td>59.80</td></tr><tr><td>TransD</td><td>68.95</td><td>67.35</td><td>64.55</td><td>54.11</td></tr><tr><td>KGTtm</td><td>80.94</td><td>86.76</td><td>85.97</td><td>88.46</td></tr><tr><td> $\mathrm { B E R T _ { \mathrm { t r i p l e } } }$ </td><td>85.33</td><td>88.84</td><td>88.53</td><td>88.06</td></tr><tr><td> $\mathrm { S i m K G \dot { C } }$ </td><td>82.78</td><td>78.62</td><td>80.26</td><td>85.47</td></tr></table>

Table 2 reports triple classification accuracy. GenVer has the highest accuracy in each column. Its margins over the strongest other baseline are $2 . 7 9 , 0 . 9 9 , 0 . 0 3 ,$ , and 0.68 percentage points on FactFB<sub>2</sub>, FactFB<sub>4</sub>, FactFB<sub>8</sub>, and ${ \mathrm { F a c t F B } } _ { 1 6 } ,$ respectively. In particular, the difference on $\mathrm { F a c t F B } _ { 8 }$ is small (88.56 versus 88.53). Run-to-run variability and statistical significance are not reported, so the margins alone do not establish a reliable advantage.

The relative performance of the baselines varies across the four relation subsets. Figure 4 summarizes their mean AUROC, with GenVer reaching 0.941. Together with the accuracy results, this supports the usefulness of generated text features in the evaluated configuration, but does not establish robustness across domains, calibrated factual confidence, or superiority under a different negativesampling protocol.

## 4.4 Case Study

We examine examples from two frequent relations, Film Date Region and Award Nomination, to illustrate how generation quality and classification confidence can diverge. For each relation, we report one true positive (TP), true negative (TN), false positive (FP), and false negative (FN), with correctness defined by the dataset labels. Relation names are abbreviated for readability.

Table 3 reproduces four examples for Film Date

![](images/0323c8063919a31d4220b35490f7e99e6499d1fe18b96e52bc18314657461363.jpg)

![](images/d261daa8cd6c6f017839d979f81f498b84d8beb86c8df1686ddd229cbb9fb91f.jpg)

![](images/5ed989e1e3cbaa4687829c1c8000dd741718f4551d80aafb52e4d94f131d46c8.jpg)  
(c) FactFB<sub>8</sub>.

![](images/21dc5c9c60ae89331dd9b1978f1733a4d2fdf1059ec47e45d5bc8e405836e6b0.jpg)  
(d) ${ \mathrm { F a c t F B } } _ { 1 6 } .$  
Figure 3: ROC curves for GenVer trained with $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { S c o r e } }$ and $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { M L } }$ . AUROC values are shown in the legends.

![](images/0211c77e914e56abd58a4fd3b14eae46401306a7315e014b093725e92dbf943d.jpg)  
Figure 4: Mean AUROC across the four FactFB variants, for the evaluated model configurations.

Region, with dataset labels and classifier scores. The high-scoring positive example associates a film with a box-office region, but does not explicitly specify a release event or date. The low-scoring negative example uses a person as the subject of a film-release relation, illustrating a type mismatch that may make the constructed task easier. These examples do not separately identify the contribution of graph structure, language-model knowledge, and NLR features.

The false positive is more revealing: the input about The Grudge is mapped to an unrelated sentence about a song, yet receives a score of 0.9997579. This is a failure to reject the dataset’s negative example despite an unfaithful NLR. It does not demonstrate that the generated sentence itself is factual or establish why the classifier assigns a high score. Conversely, the Wild at Heart example receives a low score despite its positive dataset label. Without independent evidence, that prediction is not a reason to revise the label or prefer another release region.

Table 3: Examples for Film Date Region. Target denotes the dataset label; score denotes the classifier output. Generated wording is preserved verbatim.
<table><tr><td>Subject</td><td>Object</td><td>Generated NLR</td><td>Target</td><td>Score</td><td>Case</td></tr><tr><td>Remember the Titans</td><td>United States of America</td><td>Remember the Titans opened strongly at the United States of America box office</td><td>1</td><td>0.9999993</td><td>TP</td></tr><tr><td>Bruce Cohen</td><td>United States of America</td><td>Bruce Cohen was ranked second in the United States of America</td><td>0</td><td>0.0000003</td><td>TN</td></tr><tr><td>The Grudge</td><td>Buenos Aires</td><td>“I finally found someone&quot; by Buenos Aires and Bryan Adams</td><td>0</td><td>0.9997579</td><td>FP</td></tr><tr><td>Wild at Heart</td><td>France</td><td>Wild at Heart released the film in France</td><td>1</td><td>0.0203743</td><td>FN</td></tr></table>

Table 4: Examples for Award Nomination, using the same conventions as Table 3.
<table><tr><td>Subject</td><td>Object</td><td>Generated NLR</td><td>Target</td><td>Score</td><td>Case</td></tr><tr><td>MC Hammer</td><td>Grammy Award</td><td>Hammer earned a Grammy Award for Album of the Year at the 31st Grammy Awards.</td><td>1</td><td>0.9999999</td><td>TP</td></tr><tr><td>Raymond Burr</td><td>Blake Edwards</td><td>Antarctica1 has been nominated for the Blake Edwards</td><td>0</td><td>0</td><td>TN</td></tr><tr><td>Geoffrey Rush</td><td>Golden Raspberry Award</td><td>Which featured Grammy Award for Best Boxed or Special Limited Edition Package sold over 7.6 million records worldwide</td><td>0</td><td>0.9889689</td><td>FP</td></tr><tr><td>Anne Hathaway</td><td>MTV Movie Award</td><td>Entertainment Weekly ranked MTV Movie Award for Best Fight the fifth-greatest film of all time</td><td>1</td><td>0.0642089 FN</td><td></td></tr></table>

Table 4 shows related failures for Award Nomination. The high-scoring example about MC Hammer is classified correctly according to the dataset label, but its NLR strengthens a nomination relation into a claim about winning an award and adds a category and ceremony. Correct classification therefore does not imply faithful generation. The Geoffrey Rush example receives a high score despite omitting the subject and changing the award context; the Anne Hathaway example receives a low score with another unfaithful NLR. These cases motivate separate assessment of representation faithfulness and classification correctness.

The cases illustrate that generated NLRs can be useful features but can also omit entities, change relations, or introduce unsupported details. High confidence can accompany an incorrect prediction. They are qualitative examples, not a quantitative demonstration of a causal relationship between fluency and classifier confidence.

## 5 Conclusion

We presented GenVer, a framework that combines serialized triples with generated natural language representations for triple classification. A triple– text alignment procedure supplies training pairs for a generator, and a BERT-based classifier combines triple and NLR features. The FactFB experiments favor GenVer over the evaluated baselines in classification accuracy, with margins that vary substantially across subsets. Generated representations also exhibit semantic drift, and the classifier can make high-confidence errors. The findings are therefore limited to the constructed classification setting; faithful generation and evidence-based verification remain unresolved.

## 6 Limitations

The evaluation uses four subsets of FB15k with randomly perturbed negative examples. Such negatives can contain obvious type mismatches and may not represent naturally occurring KG errors. The four variants do not establish cross-domain generalization, and a missing or perturbed edge is not necessarily false under an open-world interpretation.

Generated NLRs are not independent evidence for a triple. They can omit entities, alter relations, or add unsupported facts, as shown in the case study and appendix. The generation and classification stages are trained separately, and downstream accuracy does not directly measure NLR faithfulness.

## References

Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami Al-Rfou. 2021. Knowledge graph based synthetic corpus generation for knowledge-enhanced language model pretraining. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 3554–3565, Online. Association for Computational Linguistics.

Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary G. Ives. 2007. DBpedia: a nucleus for a web of open data. In The Semantic Web: Proceedings ofthe 6th International Semantic Web Conference and the 2nd Asian Semantic Web Conference, pages 722–735.

Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: A collaboratively created graph database for structuring human knowledge. In Proceedings ofthe 2008 ACM SIGMOD International Conference on Management of Data, pages 1247–1250.

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Durán, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems 26, pages 2787–2795.

Yu Chen, Lingfei Wu, and Mohammed J. Zaki. 2019. Bidirectional attentive memory networks for question answering over knowledge bases. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2913–2923. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. 2019. Unified language model pre-training for natural language understanding and generation. arXiv preprint arXiv:1905.03197.

Thibault Févry, Livio Baldini Soares, Nicholas FitzGerald, Eunsol Choi, and Tom Kwiatkowski. 2020. Entities as experts: Sparse memory access with entity supervision. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4937– 4951, Online. Association for Computational Linguistics.

Daniel Gerber, Diego Esteves, Jens Lehmann, Lorenz Bühmann, Ricardo Usbeck, Axel-Cyrille Ngonga Ngomo, and René Speck. 2015. DeFacto-temporal and multilingual deep fact validation. Journal ofWeb Semantics, 35(P2):85– 101.

Daniel Gerber and Axel-Cyrille Ngonga Ngomo. 2012. Extracting multilingual natural-language patterns for RDF predicates. In Knowledge Engineering and Knowledge Management: Proceedings ofthe 18th International Conference, pages 87–96.

Google. 2012. Introducing the knowledge graph: things, not strings. https://blog.google/products/search/ introducing-knowledge-graph-things-not/.

Xu Han, Shulin Cao, Xin Lv, Yankai Lin, Zhiyuan Liu, Maosong Sun, and Juanzi Li. 2018. OpenKE: An open toolkit for knowledge embedding. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 139–144, Brussels, Belgium. Association for Computational Linguistics.

Bin He, Xin Jiang, Jinghui Xiao, and Qun Liu. 2020. Kg-PLM: Knowledge-guided language model pre-training via generative and discriminative learning. arXiv preprint arXiv:2012.03551.

Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard de Melo, Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo, Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan Sequeda, Steffen Staab, and Antoine Zimmermann. 2022. Knowledge graphs. ACM Computing Surveys, 54(4):1–37.

Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. 2015. Knowledge graph embedding via dynamic mapping matrix. In Proceedings of the 53rd Annual Meeting of the Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 687–696, Beijing, China. Association for Computational Linguistics.

Shengbin Jia, Yang Xiang, Xiaojun Chen, and Kun Wang. 2019. Triple trustworthiness measurement for knowledge graph. In The World Wide Web Conference, pages 2865– 2871.

Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha, Qatar. Association for Computational Linguistics.

Timothee Lacroix, Nicolas Usunier, and Guillaume Obozinski. 2018. Canonical tensor decomposition for knowledge base completion. In Proceedings ofthe 35th International Conference on Machine Learning, pages 2863–2872.

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 2181–2187.

Bonaventure C. Molokwu, Shaon Bhatta Shuvo, Ziad Kobti, and Narayan C. Kar. 2021. Social network analysis using knowledge-graph embeddings and convolution operations. In 2020 25th International Conference on Pattern Recognition, pages 6351–6358.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language models as knowledge bases? In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2463–2473, Hong Kong, China. Association for Computational Linguistics.

Zafar Habeeb Syed, Michael Röder, and Axel-Cyrille Ngonga Ngomo. 2018. FactCheck: Validating RDF triples using textual evidence. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pages 1599–1602.

Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. 2016. Complex embeddings for simple link prediction. In Proceedings of the 33rd International Conference on Machine Learning, pages 2071–2080.

Pat Verga, Haitian Sun, Livio Baldini Soares, and William Cohen. 2021. Adaptable and interpretable neural Memory Over symbolic knowledge. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3678–3691, Online. Association for Computational Linguistics.

Denny Vrandeciˇ c and Markus Krötzsch. 2014. Wikidata: A´ free collaborative knowledgebase. Communications ofthe ACM, 57(10):78–85.

Liang Wang, Wei Zhao, Zhuoyu Wei, and Jingming Liu. 2022. SimKGC: Simple contrastive knowledge graph completion with pre-trained language models. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4281–4294, Dublin, Ireland. Association for Computational Linguistics.

Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph embedding by translating on hyperplanes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 28, pages 1112–1119.

Bishan Yang, Scott Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding entities and relations for learning and inference in knowledge bases. In International Conference on Learning Representations.

Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. KG-BERT: BERT for knowledge graph completion. arXiv preprint arXiv:1909.03193.

Jun Zhao, Zhou Zhou, Ziyu Guan, Wei Zhao, Wei Ning, Guang Qiu, and Xiaofei He. 2019. IntentGC: A scalable graph convolution framework fusing heterogeneous information for recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 2347–2357.

Xiaohan Zhou, Yunhui Yi, and Geng Jia. 2021. Path-RotatE: Knowledge graph embedding by relational rotation of path in complex space. In 2021 IEEE/CIC International Conference on Communications in China, pages 905–910.

## A Supplementary Details

## A.1 Dataset Statistics

Table 5 summarizes the entity, relation, and triple counts for the four FactFB variants.

Table 5: Sizes of the FactFB variants.
<table><tr><td>Property</td><td>FactFB2</td><td> $\mathrm { F a c t F B _ { 4 } }$ </td><td> $\mathrm { F a c t F B } _ { 8 }$ </td><td> ${ \mathrm { F a c t F B } } _ { 1 6 }$ </td></tr><tr><td>Entities</td><td>4085</td><td>5680</td><td>7230</td><td>7621</td></tr><tr><td>Relations</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>Triples</td><td>142088</td><td>260640</td><td>460464</td><td>752164</td></tr></table>

## A.2 Entity-Page Lookup for NLR Retrieval

To reduce search cost, the NLR retrieval procedure searches only Wikipedia pages associated with a triple’s subject or object entity. This is a retrieval restriction, not a guarantee that all relevant evidence appears on those pages. Entity mappings and the dbo:wikiPageID property are used to obtain page identifiers before searching the Wikipedia dump. The following SPARQL query illustrates this lookup for Freebase entity m. ${ \boldsymbol { 0 3 } } _ { - } { \boldsymbol { r 3 } }$ (Jamaica), with namespace declarations made explicit. Execution requires a compatible mapping dataset and endpoint.

PREFIX owl: <http://www.w3.org/2002/07/owl#>   
PREFIX dbo: <http://dbpedia.org/ontology/>   
PREFIX fb: <http://rdf.freebase.com/ns/>   
SELECT DISTINCT ?pageID WHERE {   
?entity owl:sameAs fb:m.03\_r3 ;   
dbo:wikiPageID ?pageID .   
}

## A.3 Default Ranking Parameters

The feature order is ${ \bf x } = ( { \bf M } { \sf S } , { \sf S } { \mathrm { L } } , { \sf R } { \mathrm { R } } , { \sf P } { \mathrm { T } } )$ . The prior-based ranking rule standardizes each feature using its mean $\mu _ { i }$ and standard deviation $\sigma _ { i } .$ , then assigns a sign $\tilde { w } _ { i } \in \{ - 1 , 1 \}$ :

$$
\operatorname { Q S } ( \mathbf { x } ) = \left[ 1 + \exp \left( - \sum _ { i = 1 } ^ { 4 } \tilde { w } _ { i } \frac { x _ { i } - \mu _ { i } } { \sigma _ { i } } \right) \right] ^ { - 1 } = \left[ 1 + \exp \left( - \sum _ { i = 1 } ^ { 4 } \frac { \tilde { w } _ { i } } { \sigma _ { i } } x _ { i } + \sum _ { i = 1 } ^ { 4 } \frac { \tilde { w } _ { i } \mu _ { i } } { \sigma _ { i } } \right) \right] ^ { - 1 } .\tag{11}
$$

Thus the coefficients in Eq. (4) are

$$
w _ { i } = \frac { \tilde { w } _ { i } } { \sigma _ { i } } ,\tag{12}
$$

$$
b _ { s } = - \sum _ { i = 1 } ^ { 4 } \frac { \tilde { w } _ { i } \mu _ { i } } { \sigma _ { i } } .\tag{13}
$$

Setting $\tilde { \mathbf { w } } = ( + 1 , - 1 , + 1 , + 1 )$ rewards larger MS, RR, and PT, and penalizes greater sentence length. Substituting the statistics in Table 6 gives $w _ { 1 } = 2 . 2 2 4$ $w _ { 2 } = - 0 . 0 8 8 , w _ { 3 } = 5 . 3 9 2 , w _ { 4 } = 3 . 8 0 3$ , and $b _ { s } = - 1 . 3 4 2$ after rounding.

Table 6: Feature means and standard deviations.
<table><tr><td>Statistic</td><td>MS</td><td>SL</td><td>RR</td><td>PT</td></tr><tr><td>μ</td><td>0.56701</td><td>18.18729</td><td>0.08487</td><td>0.32198</td></tr><tr><td>σ</td><td>0.44968</td><td>11.36014</td><td>0.18546</td><td>0.26298</td></tr></table>

## A.4 Experimental Settings

Experiments were conducted on a machine with an Intel Xeon Gold 5117 CPU at 2.00 GHz, 256 GB of RAM, and an NVIDIA GeForce RTX 2080 Ti GPU, running Ubuntu 18.04 and CUDA 10.0.

GenVer. The generation model was trained for 300 epochs with learning rate $5 \times 1 0 ^ { - 6 }$ , batch size 32, and maximum sequence length 128. The verification model used 100 epochs, learning rate $1 0 ^ { - 5 }$ , batch size 128, and maximum sequence length 128.

KG embedding methods. TransE, ComplEx, DistMult, TransH, TransR, and TransD were implemented using OpenKE<sup>2</sup> (Han et al., 2018). Training used 100 rounds, a learning rate of 0.5, a ranking-loss margin of 5, and an embedding dimension of 100.

KGTtm. The experiments used the released implementation<sup>3</sup> with TransE embeddings on FactFB. The three estimators, TEF, ResourceRank, and RPI, were combined as described by the original method. Training used 200 epochs, an embedding dimension of 100, and dropout of 0.5.

$\mathrm { B E R T _ { t r i p l e } }$ . This ablation removes the generated NLR and its corresponding feature extractor from GenVer while retaining the other settings.

SimKGC. The experiments used the released implementation<sup>4</sup>. The settings are 400 warmup steps, gradient clipping at 10, dropout 0.1 on the final linear layer, and temperature 0.05.

## A.5 Generated NLR Examples

Table 7 shows example triples and generated NLRs for both training-pair configurations. Entity and relation labels are abbreviated for readability; generated text is shown verbatim. The examples show omissions, unsupported details, and changes to relation meaning. They are qualitative examples, not a quantitative evaluation of factual correctness or faithful verbalization.

Table 7: Generated NLR examples (outputs reproduced verbatim). Score and ML denote training on $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { S c o r e } }$ and $\mathrm { T } { \cdot } \mathrm { N L R } _ { \mathrm { M I } }$ ， respectively.
<table><tr><td>Triple</td><td>Generated NLR (Score)</td><td>Generated NLR (ML)</td></tr><tr><td>S: George Lucas P: Award Nominee O: Djimon Hounsou</td><td>Djimon Hounsou was nominated for the George Lucas</td><td>Djimon Hounsou was nominated for George Lucas</td></tr><tr><td>S: Anjelica Huston P: Film Region Date O: Spain</td><td>Anjelica Huston said, “a huge line from Spain Anjelica Huston opened in Spain to Spain can be found in the us"</td><td></td></tr><tr><td>S: Hugo Award P: Award Category O: Michael Swanwick</td><td>Michael Swanwick has received three writers guild of america awards, including nominations for an Hugo Award for Best Novel</td><td>Michael Swanwick has the most nominations for one golden globe and Hugo Award for Best Novel</td></tr><tr><td>S: Scarlett Johansson P: Award Nomination O: Best Female Lead</td><td>Scarlett Johansson appeared in the biopic “pollock" in which she won an Independent Spirit Award for Best Female Lead</td><td>Scarlett Johansson won the Independent Spirit Award for Best Female Lead</td></tr><tr><td>S: Leighton Meester P: Person Profession O: Singer-songwriter</td><td>Leighton Meester released the single “suddenly i feel love" with singer-songwriter</td><td>Leighton Meester attended the Professional Children's School and began working as a singer-songwriter</td></tr><tr><td>S: Actor P: People Profession O: Bono</td><td>Bono is the second actor</td><td>Bono was such a lousy actor</td></tr><tr><td>S: conducting P: Performance Actor O: Elmer Bernstein</td><td>Elmer Bernstein was surprised by michael aspel at London's royal albert hall after conducting</td><td>Elmer Bernstein composed the incidental music score for conducting</td></tr><tr><td>S: Milla Jovovich P: Film Performance O: Shea Whigham</td><td>Shea Whigham met andy robinson, and subsequently started dating until Milla Jovovich</td><td>Shea Whigham has appeared in several projects alongside actors Milla Jovovich</td></tr><tr><td>S: Lincoln P: Nomination Award O: Academy Award</td><td>“Lincoln" was the first film to win both Academy Award for Best Picture and best director</td><td>Lincoln won the Academy Award for Best Picture</td></tr><tr><td>S: Academy Award P: Award Category For O: Guys and Dolls</td><td>Guys and Dolls won two golden raspberry awards</td><td>Guys and Dolls was nominated for the Academy Award for Best Production Design</td></tr><tr><td>S: Rob Corddry P: award_winner O: NBC</td><td>Rob Corddry is the only sitcom to have three main episodes in six different networks including the sitcom“ the nutt House" which aired on NBC</td><td>Rob Corddry was succeeded by NBC</td></tr><tr><td>S: comedy-drama P: film_genre O:BILB</td><td>Bend It Like Beckham is a 1979 american legal comedy-drama</td><td>Bend It Like Beckham is an american comedy-drama</td></tr><tr><td>S: action film P: genre O: The Mummy</td><td>The Mummy: Tomb of the Dragon Emperor is a 1978 american-German action film</td><td>The Mummy: Tomb of the Dragon Emperor was able to find two universal themes in an action film</td></tr><tr><td>S: Pursuit of Happyness P: nominated_work O: Wim Wenders</td><td>The Pursuit of Happyness is a 2006 american biographical drama film directed by Wim Wenders</td><td>The Pursuit of Happyness is a 2006 american biographical drama film directed by Wim Wenders</td></tr><tr><td>S: George Clooney P: nominated_for O: Batman &amp; Robin</td><td>George Clooney was cast because spielberg felt Batman &amp; Robin</td><td>Batman &amp; Robin is the fourth installment in the“Barack Obama" film series and stars</td></tr><tr><td>S: Yo-Yo Ma P: film_crew_gig O: Seeking a Friend</td><td>Seeking a Friend for the End of the World is a 2009 experimental documentary film written and directed by Yo-Yo Ma</td><td>George Clooney Seeking a Friend for the End of the World is an american computer-animated musical comedy film directed by Yo-Yo Ma</td></tr></table>