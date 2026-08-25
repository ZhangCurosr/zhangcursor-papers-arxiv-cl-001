# Exploring Dowker Homology for Sentence Similarity

Marius Huber Juri Opitz

Department of Computational LinguisticsUniversity of Zürich{marius.huber,jurialexander.opitz}@uzh.ch

## Abstract

Dowker homology is a topological tool that may be used to analyze the relative position of two point clouds living in a common space. We investigate whether Dowker homology captures sentence similarity information by treating the embeddings of the tokens that constitute a sentence pair as a pair of point clouds in the latent space of a transformer model, using both models that have and have not been fine-tuned for sentence similarity. We find that Dowker homology captures sentence similarity information, as measured by regressing Dowker homology features onto ground-truth similarity scores, and that it can be used for visual inspection of similarity data and models. In an attempt to make Dowker homology readily applicable, we derive from it single-number summaries that we expect to capture sentence similarity directly. These turn out to work reasonably well, but without outperforming standard sentence similarity measures based on established pooling methods.<sup>1</sup>

## 1 Introduction

Dowker homology (DH; Dowker, 1952; Chowdhury and Mémoli, 2018) is a tool from topological data analysis, an emerging data analysis framework drawing on ideas from topology, metric geometry, and related areas of mathematics. Among other things, DH may be used to capture the relative position of two point clouds living in a common space that is endowed with a distance function (typically R<sup>n</sup> with Euclidean or cosine distance).

In this paper, we study a novel application of DH to NLP by applying it to Siamese sentence transformer models (Reimers and Gurevych, 2019). Specifically, we ask the following.

(1) Can DH distinguish similar and dissimilar sentence pairs?

(2) Can DH be used to extract sentence similarity from token embeddings?

On the positive side, we find that DH can elucidate sentence similarity data and models, both quantitatively and visually. Findings regarding the second question are mixed: we define singlenumber summaries extracted from DH that capture ground truth sentence similarity reasonably well, but not more so than established methods.

## 2 Background on Dowker Homology

This section provides a high-level review of DH; for details, we refer the reader to Appendix A.1. We limit ourselves to reviewing zero-dimensional DH, as that is the version of DH used in our experiments.

We consider a pair of point clouds $X _ { 1 }$ and $X _ { 2 }$ living in a common space that is endowed with a distance function d. We say that a point in $X _ { 1 }$ or $X _ { 2 }$ is r-close to another if the distance between the two points is at most r (as measured by d). Given a scale $r \geq 0$ , one constructs a graph whose vertices are those points in $X _ { 1 }$ that have a neighbor in $X _ { 2 }$ that is r-close, and where two points in $X _ { 1 }$ are connected by an edge whenever there exists a point in $X _ { 2 }$ that is r-close to both of them. In particular, the graph is constructed on elements of $X _ { 1 }$ only, while $X _ { 2 }$ merely dictates the connectivity of the graph. To obtain DH of $X _ { 1 }$ relative to $X _ { 2 }$ one initially sets the scale to $r = 0 .$ , and gradually increases it to theoretical infinity. Along the way, one keeps track of the number of connected components constituting the aforementioned graph: any such component “is born” at a certain scale, and “dies” at some later scale as it is absorbed by—that is, connected via an edge to—an existing component. This yields a sequence $( b _ { 1 } , d _ { 1 } ) , \dots , ( b _ { k } , d _ { k } )$ where each pair contains the scales at which the birth and death, respectively, of a component happen. This sequence is recorded in a persistence diagram, which is obtained by placing a point at coordinates $( b _ { i } , d _ { i } )$ in $\mathbb { R } ^ { 2 }$ for all $i = 1 , \ldots , k . ^ { 2 } \mathrm { ~ A ~ }$ key property of DH is that it is unaffected when swapping the roles of $X _ { 1 }$ and $X _ { 2 }$ in the above process, making the resulting persistence diagram an “undirected” quantity (Dowker, 1952; Chowdhury and Mémoli, 2018).

![](images/99240a16ca31e918a42a5a3e2d705ee359b4f7aafd9b17f69f5fd2c87e900699.jpg)

![](images/467f25f116e210a4e075f11cc3ff5fa89ba078645de7d27b954481809371fa61.jpg)

![](images/ce51811075a2d619d743606e86f4b324dfd13e58385862a0ca72ba81e0c55e10.jpg)

![](images/0cab6e2bdcb36df09c5ade8209ed27c1d11b29ee726e5d18b1c0dd227044abf9.jpg)  
Figure 1: Two pairs of point clouds of varying degree of spatial separation (top) and their corresponding Dowker persistence diagrams (bottom). The more the point clouds are separated, the larger the birth values in the persistence diagrams.  
“A man is slicing garlic.” vs. “A cat is playing on the floor.”

If the point clouds $X _ { 1 }$ and $X _ { 2 }$ are spatially well separated, vertices and edges appear only at large scales. If $X _ { 1 }$ and $X _ { 2 }$ nearly coincide, those appear at small scales already; indeed, if $X _ { 1 } = X _ { 2 } ,$ all birth values equal zero. The spatial proximity (resp. separation) of $X _ { 1 }$ and $X _ { 2 }$ is thus reflected in the points of the Dowker persistence diagram having small (resp. large) birth values; see Figure 1.

## 3 Dowker Homology for Similarity

To apply DH to sentence similarity, we treat the token embeddings stemming from a pair of sentences as a pair of point clouds in the latent space of a transformer model, and compute the corresponding Dowker persistence diagram. If the two sentences are similar (resp. dissimilar), we expect the corresponding point clouds to be colocalized (resp. separated) and, as discussed in Section 2, we expect the resulting Dowker persistence diagram to have predominantly small (resp. large) birth values.

![](images/55b4c1d96cbdcd3d7f739df232cfedf77852950977a1ec16a0a8958865266cd8.jpg)

![](images/7cae600d037a70e459444b493f44f65b087e41cdeb18e7cbd0d59ff941854919.jpg)  
“This is a bad idea.” vs. “This is a terrible idea.”

![](images/7401091651f1fe2fea9208deb5839699ccfd777c7cacba1b74b99ac901805081.jpg)

![](images/c50946cdb2858b1457cf0f0c600b9b3b193bea136e9d9d930eeda55b06246190.jpg)  
Figure 2: Dowker persistence diagrams capturing sentence similarity of two randomly selected sentence pairs from the STS-B dataset with low (left column) and high (right column) similarity, computed from embeddings produced using MiniLM-L6 (top row) and all-MiniLM-L6 (bottom row). Higher sentence similarity is reflected in smaller birth values in the diagrams, and the fact that MiniLM-L6 is not fine-tuned for sentence similarity is reflected in the top two diagrams being more similar to each other than the bottom two diagrams. Moreover, fine-tuning MiniLM-L6 for sentence similarity substantially alters the persistence diagram corresponding to the dissimilar sentence pair, while leaving the other diagram largely unaffected.

We point out that our method is similar in spirit to those used in Barannikov et al. (2022) and Michel et al. (2017). Indeed, both methods also draw on ideas from topological data analysis to assess the (dis-)similarity of two point clouds. The former method, however, requires the two point clouds to be of equal sizes, while the latter takes into account only the topological features of the two point clouds individually and does not capture, for instance, their spatial separation.

Example. Figure 2 illustrates that our expectation holds for two sentence pairs randomly selected from the STS Benchmark (STS-B; Cer et al., 2017) when using the fine-tuned all-MiniLM-L6 model, but not when using the non-fine-tuned MiniLM-L6 model. Fine-tuning a transformer model for sentence similarity seems to strongly affect how dissimilar sentences are represented (as seen through the lens of DH), but less so for similar sentences, where the persistence diagrams stemming from the non-fine-tuned and the fine-tuned model are basically indistinguishable. Indeed, the finding of Figure 2 generalizes across all similar and dissimilar pairs from the STS-B dataset, as illustrated in Figure 3.

![](images/442a746982647b48b292a579da61ac42bfa7f7337cd08fd26587eaf5c9ac5241.jpg)

![](images/da5779f640b693fec4799b2a4c9226e98a95cf39aae0487c0286d805e033e977.jpg)

![](images/2f3bf252a4dcb15036a6e030b19b1011baaf2c7cbef96aba7f418e5f57fc4506.jpg)

![](images/9b34fafe0f7b1101455142f6de3522eb5745949d417918e314bcddd539c156f3.jpg)  
Figure 3: Mean persistence images capturing sentence similarity of all sentence pairs from the STS-B dataset with low (left column) and high (right column) similarity, computed from embeddings produced using MiniLM-L6 (top row) and all-MiniLM-L6 (bottom row). These persistence images can be thought of as “heat maps” that capture the distribution of points across multiple persistence diagrams (see Appendix A.2 for details). The images indicate that fine-tuning MiniLM-L6 for sentence similarity leads to larger birth values in DH of dissimilar sentence pairs, thus generalizing the findings from Figure 2 across the entire STS-B dataset. Here, the sentence pairs with low (resp. high) similarity are those whose ground truth similarity in STS-B differs by at most 0.1 from the minimum (resp. maximum) possible similarity value. For the analogous plots for the other models used, see Appendix A.5.1.

Experiment outline. Motivated by this example, we conduct the following two experiments for each layer of every model from a diverse set of transformer models fine-tuned for sentence similarity as well as their underlying, non-fine-tuned base models (see Table A.1 in Appendix A.3 for a list of models used). We use sentence pairs from the STS-B dataset, which contains sentence pairs annotated by human raters with semantic similarity scores ranging from 0 to 5.

1. Layer regression experiment: We investigate to what extent standard token-level measures as well as Dowker persistence diagrams contain information on sentence similarity. We do so by fitting a linear regression model that predicts ground truth sentence similarities from these measures.

2. Layer correlation experiment: In order to circumvent the need for a regression trained on Dowker persistence diagrams, and to provide measures of sentence similarity that are more readily applicable than the raw persistence diagrams, we define single-number summaries extracted from DH. These summaries are defined so that we expect them to capture sentence similarity directly, and we assess how well they correlate with ground truth sentence similarities.

We now describe the sentence similarity methods used in our experiments.

## 3.1 Baseline Similarity Methods

We use BOS-, EOS-, max- and mean-pooling as single-vector sentence representations. The similarity of a sentence pair is then computed as the cosine similarity between the pooled representations of the sentences.

## 3.2 Dowker Similarity

We define four versions of Dowker similarity (DS), one for each of four different kinds of aggregation used along the way. To compute DS for a sentence pair $( s _ { 1 } , s _ { 2 } )$ , we consider the token embeddings of $s _ { 1 }$ and $s _ { 2 }$ as point clouds $X _ { 1 }$ and $X _ { 2 } .$ , respectively, and compute DH of $X _ { 1 }$ relative to $X _ { 2 } .$ , as explained in Section 2. We extract single-number summaries from the resulting persistence diagram by computing the minimum, maximum and average of all birth values of points in the diagram, as well as the weighted average birth value, where birth values are weighted by the lifetime of the corresponding point. We regard each of these as a notion of distance between $X _ { 1 }$ and $X _ { 2 }$ , since larger birth values indicate a larger degree of spatial separation of $X _ { 1 }$ and $X _ { 2 }$ (see Section 2), and we denote these distances by $d _ { \mathrm { D o w } } ^ { \mathrm { m i n } } , d _ { \mathrm { D o w } } ^ { \mathrm { m a x } } , d _ { \mathrm { D o w } } ^ { \mathrm { m e a n } }$ and $d _ { \mathrm { D o w } } ^ { \mathrm { w g h t } }$ , respectively. Finally, we define the DSs by setting $s _ { \mathrm { D o w } } ^ { \bullet } ( s _ { 1 } , s _ { 2 } ) = e ^ { - d _ { \mathrm { D o w } } ^ { \bullet } ( X _ { 1 } , X _ { 2 } ) }$ for ${ \bullet \in }$ {min, max, mean, wght}. All versions of DS thus attain values between 0 and 1. Moreover, if $s _ { 1 } ~ = ~ s _ { 2 }$ , then $X _ { 1 } = X _ { 2 }$ , and all birth values in DH equal zero, so that all four versions of Dowker distance vanish, and, correspondingly, all four versions of DS yield a similarity value of 1 for identical sentences.

## 4 Experiments and Results

We now provide detailed descriptions of the two experiments conducted, and then briefly discuss some limitations and design choices.

## 4.1 Layer Regression Experiment

Setup. We sample 512 and 256 sentence pairs from the train and test splits, respectively, of STS-B. As baseline features, we use the vectors $( u _ { 1 } , u _ { 2 } , | u _ { 1 } - u _ { 2 } | )$ , where $u _ { 1 }$ and $u _ { 2 }$ are the pooled representations of a sentence pair $( s _ { 1 } , s _ { 2 } )$ , obtained via the four pooling methods described in Section 3.1 (this is motivated by Reimers and Gurevych (2019, Section 6); here, | · | denotes the entry-wise absolute value). For DH, we compute the persistence diagram of each sentence pair from the corresponding token embeddings (as described in Section 3.2) using cosine distance, and vectorize it into a persistence image of resolution $2 5 \times 2 5 .$ , resulting in a vector of length 625 (see Appendix A.2 for details on persistence images). We then train a ridge regression on these features and evaluate it using Spearman’s rank correlation between the predicted and the ground-truth similarities; the choice of ridge regression is motivated by the fact that persistence images produce high-dimensional, sparse vectors with correlated features, making ordinary least squares estimates potentially unstable (see Appendix A.4 for selection of hyperparameters). This procedure is run for each hidden layer of each model.

Results. We find that across all models (finetuned or not), the regression trained on the Dowker persistence diagrams never outperforms the one trained on the best performing baseline similarity methods. The regression trained on the Dowker persistence diagrams always wins at least third place (as measured by Spearman correlation averaged across all layers of a model), consistently beating those trained on BOS- and EOS-pooling, but coming short of outperforming those trained on max- and mean-pooling. Interestingly, maxpooling yields the best features for $9 / 1 0$ models despite the fact that the sentence transformers used were fine-tuned using mean-pooling. Finally, we observe that in the fine-tuned models—in contrast to the non-fine tuned ones—all similarity methods tend to yield higher Spearman correlations on later layers, which is somewhat intuitive as the final layer is where fine-tuning is directly applied. As an example, Figure 4 shows the Spearman correlations of all methods at each layer of MiniLM-L6 (not fine-tuned) and all-MiniLM-L6 (fine-tuned); the former is one of the two cases in which the features derived from DH rank second, outperforming those derived from mean-pooling. Appendix A.5.2 shows analogous plots for the other model pairs.

## 4.2 Layer Correlation Experiment

Setup. We sample 256 sentence pairs from the test split of STS-B, and compute similarities for each hidden layer of each model, using the four baseline similarity methods and the four versions of DS from Sections 3.1 and 3.2, respectively. For each method, model and layer, we evaluate the performance via Spearman correlation between the predicted and the ground-truth similarities.

Results. As before, we assess the goodness of the baseline methods and the four versions of DS by the Spearman correlations averaged across all layers of a model. We find that (at least one version of) DS comes in third behind the max- and mean-pooling methods for $5 / 1 0$ models, and wins first and second place for $2 / 1 0$ models each. Again, we find that max-pooling outperforms mean-pooling in terms of average Spearman correlation. Mean-pooling, however, unsurprisingly yields the highest maximal Spearman correlation for the sentence transformer models, and this is always the case at the final model layer, sometimes after a sheer increase of performance in the last few layers. Finally, we find that $s _ { \mathrm { D o w } } ^ { \mathrm { m e a n } }$ outperforms the other versions of DS for $9 / 1 0$ models. As an example, Figure 5 shows the Spearman correlations of all methods at each layer of MiniLM-L6 and all-MiniLM-L6; the former is one of two cases where $s _ { \mathrm { D o w } } ^ { \mathrm { m e a n } }$ outperforms all other methods (the other such instance being MiniLM-L12). We refer to Appendix A.5.3 for plots of the remaining results.

## 4.3 Experiment Limitations

Dataset and Sample Sizes. In both experiments, we confined ourselves to working with just the STS-B dataset and the relatively modest respective sample sizes due to computational constraints.

Computation of DH Feature Vectors. While DH may be computed using any metric, we restricted ourselves to using cosine distance in our experiments. We experimented using Euclidean distance, but cosine distance consistently yielded better results. For the layer regression experiment, the resolution of $2 5 \times 2 5$ used for vectorization of persistence diagrams was chosen as an empirical compromise between preserving performance and limiting the dimensionality of the resulting feature vectors. Finally, we restricted ourselves to working with zero-dimensional DH because we suspected higher-dimensional DH to capture information that goes beyond spatial separation of point clouds. Indeed, preliminary experiments showed that the point clouds consisting of token embeddings rarely exhibited any higher-dimensional DH at all, and including higher-dimensional DH as features in the layer regression experiment had virtually no effect on the results.

![](images/c0976dba50264ebf1681ddfc8cf1a44ece60b3af21a72a584521a27c10a27094.jpg)

![](images/6f5e6cc33d589dc99792f2233665ee7d87af78cbe6087649d70bffbd2ba5c90d.jpg)  
Figure 4: Two results of the layer regression experiment. Bars indicate standard deviation over eight runs.

![](images/f617eb556f5da172d3d02a2c7dfef3b82529eb8fb12aa524e72190cfcf2bb93f.jpg)

![](images/991bdcd68adc06f71be28b44f7195fd6b0af6f4f45b9fa6df621be50e1704edb.jpg)  
Figure 5: Two results of the layer correlation experiment. Versions of Dowker similarity are indicated by the suffixes.

Evaluation metric. In both experiments, we confined ourselves to evaluating the agreement of DSs with ground truth similarities via Spearman correlation since DSs are obtained by applying the monotonic transformation $d \mapsto e ^ { - d }$ to the uncalibrated distance measures $d _ { \mathrm { D o w } } ^ { \mathrm { m i n } } , d _ { \mathrm { D o w } } ^ { \mathrm { m a x } } , d _ { \mathrm { D o w } } ^ { \mathrm { m e a n } }$ and $d _ { \mathrm { D o w } } ^ { \mathrm { w g h t } }$ . We hence do not expect any linear relationship between DSs and ground truth similarities as captured, for instance, by Pearson correlation.

## 5 Discussion

Our experiments show that Dowker persistence diagrams reflect sentence similarity (cf. Question (1)) and, moreover, that they can provide visual insights on NLP models, such as the effect of fine-tuning on the representation of similar and dissimilar sentence pairs (cf. the example in Section 3). Moreover, while all versions of DS capture sentence similarity to some extent, the version that averages birth values is by far the best-performing one, as witnessed by the fact that it correlates with ground truth similarities essentially as well as a trained linear regression. Even this version of DS, however, does not consistently outperform established methods, answering Question (2) with a “yes, but”. In summary, our findings provide insights on the usefulness and interpretability of DH, and justify further research on connections between topology and sentence similarity, such as exploration of further single-number summaries extracted from DH.

## References

Henry Adams, Tegan Emerson, Michael Kirby, Rachel Neville, Chris Peterson, Patrick Shipman, Sofya Chepushtanova, Eric Hanson, Francis Motta, and Lori Ziegelmeier. 2017. Persistence images: A stable vector representation of persistent homology. Journal of Machine Learning Research, 18(8):1–35.

Serguei Barannikov, Ilya Trofimov, Nikita Balabin, and Evgeny Burnaev. 2022. Representation topology divergence: A method for comparing neural network representations. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 1607–1626. PMLR.

Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. 2017. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 1–14, Vancouver, Canada. Association for Computational Linguistics.

Samir Chowdhury and Facundo Mémoli. 2018. A functorial Dowker theorem and persistent homology of asymmetric networks. J. Appl. Comput. Topol., 2(1- 2):115–175.

Pawel Dlotko. 2026. Persistence representations. In GUDHI User and Reference Manual, 3.12.0 edition. GUDHI Editorial Board.

C. H. Dowker. 1952. Homology groups of relations. Ann. ofMath. (2), 56:84–95.

Marius Huber, David Robert Reich, and Lena Ann Jäger. 2026. Fixation sequences as time series: A topological approach to dyslexia detection. In Proceedings of the 2026 Symposium on Eye Tracking Research and Applications, ETRA ’26, New York, NY, USA. Association for Computing Machinery.

Marius Huber and Patrick Schnider. 2025. Flagifying the dowker complex. Preprint, arXiv:2508.08025. ArXiv:2508.08025 [math.AT], https://arxiv.org/abs/2508.08025.

Paul Michel, Abhilasha Ravichander, and Shruti Rijhwani. 2017. Does the geometry of word embeddings help document classification? A case study on persistent homology-based representations. In Proceedings of the 2nd Workshop on Representation Learning for NLP, pages 235–240, Vancouver, Canada. Association for Computational Linguistics.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natu ral Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

## A Appendix

## A.1 Dowker Homology

DH, introduced by Dowker (1952) and promoted to the setting of persistent homology by Chowdhury and Mémoli (2018), is a tool from topological data analysis that captures the topology of two sets X and Y and a relation $R \subseteq X \times Y$ between them. Here, we review DH as used in the main text of the article. We point out that the implementation of DH used in our experiments is that of Dowker–Rips homology provided by Huber and Schnider (2025). While Dowker–Rips homology is, in general, merely an approximate, computationally cheaper variant of DH, we use that implementation since all our experiments rely only on zero-dimensional DH, in which case Dowker and Dowker–Rips homology coincide. Correspondingly, the following exposition is restricted to zerodimensional DH.

In the case where X and $Y$ are subsets of some metric space $( Z , d )$ , and given some scale $r \geq 0$ one may define a relation $R _ { r }$ by declaring that $( x , y ) \in R _ { r }$ iff $d ( x , y ) \leq r$ , for $x \in X$ and $y \in Y$ The relation $R _ { r }$ may be translated into the graph $\Gamma _ { r } = ( V _ { r } , E _ { r } )$ , defined as having vertex set

$$
V _ { r } = \{ x \in X \mid \exists y \in Y \colon d ( x , y ) \leq r \} \subseteq X ,
$$

and where two vertices $x _ { 1 } , x _ { 2 } \in X$ are connected via an edge whenever there exists $y \in Y$ such that $d ( x _ { 1 } , y ) \leq r$ and $d ( x _ { 2 } , y ) \leq r$

As is the case for any graph, the graph $\Gamma _ { r }$ consists of some—possibly zero—connected components. Indeed, if $r = 0 , \Gamma _ { r } = \emptyset$ (provided that X and Y have no common elements). As one increases the scale $r ,$ vertices and edges appear in $\Gamma _ { r }$ . Correspondingly, the number of connected components of $\Gamma _ { r }$ can increase or decrease as r increases: a new vertex may appear when r becomes large enough for that vertex to “see” an element of Y , or two existing components of Γ<sub>r</sub> may merge into one when r becomes large enough for two vertices initially belonging to different connected components to be connected via an edge. In other words, increasing the scale r from zero to (theoretical) infinity yields a sequence “births” and “deaths” of connected components. DH is the information contained in this process, and may be recorded by a sequence $( b _ { 1 } , d _ { 1 } ) , \dots , ( b _ { k } , d _ { k } )$ of pairs of nonnegative real numbers, where the i-th pair contains the value of the scale r at which the “birth” $( b _ { i } )$ and “death” (d ), respectively, of the i-th connected component occur.

This sequence is usually recorded in a persistence diagram, which, by definition, is the subset of $\mathbb { R } ^ { 2 }$ consisting of the pairs $( b _ { 1 } , d _ { 1 } ) , \dots , ( b _ { k } , d _ { k } )$ (see Figure 1 in the main text for examples of persistence diagrams). Persistence diagrams are usually endowed with the diagonal line characterized by those points where birth values equal death values; the vertical distance of the point $( b _ { i } , d _ { i } )$ in the persistence diagram indicates the lifetime (that is, $d _ { i } - b _ { i } )$ of the i-th connected component. As such, a persistence diagram provides an at-a-glance summary of the evolution of the graph $\Gamma _ { r }$ as r increases from zero to infinity, and—as explained in Section 2—captures the spatial separation of X and Y as measured by the metric d defined on the metric space $( Z , d )$ . We close this section by remarking that any Dowker persistence diagram contains precisely one point whose lifetime is infinite, that is, which dies at only scale $r = \infty$ . Since this infinite value typically breaks many formulas and pipelines, one usually either discards this point from the diagram, or replaces the scale at which it dies by some finite number that is chosen according to some heuristic. In our experiments, we experimented with both options, and found that simply discarding that point yielded better and more stable results.

## A.2 Vectorizing Persistence Diagrams

To make persistence diagrams usable in machine learning pipelines, one must transform them into vectors of a fixed length. In our experiments, we use a method known as persistence image (Adams et al., 2017). A persistence image is obtained from a persistence diagram by first applying a shearing that maps the diagonal of the persistence diagram to a horizontal line. The resulting diagram is then converted into a “heat map” by placing a bivariate Gaussian distribution with a given standard deviation (also referred to as bandwidth in the present context) at each point of the diagram. This yields a single-channel image commonly referred to as persistence surface; see Figure A.1 for an illustration of the process of obtaining a persistence surface from a persistence diagram. Finally, the persistence image is obtained from the persistence surface by discretizing it into a pixel grid of a chosen resolution. The bandwidth and resolution used in the process are treated as hyperparameters in applications.

## A.3 Models used

See Table A.1 for a list of models used in our experiments.

## A.4 Hyperparameter Selection

To train the ridge regressions used in the layer regression experiment described in Section 4.1, we used the implementation provided by scikitlearn (Pedregosa et al., 2011). In doing so, we used the SVD-solver, tuned the regularization parameter α via grid search over the values $1 0 ^ { - 3 } , 1 0 ^ { - 2 } , \dotsc , 1 0 ^ { 2 }$ , and left other parameters at their default values. Persistence diagrams in these experiments were transformed into persistence images using the GUDHI library (Dlotko, 2026). We used default values for persistence image creation, apart from the bandwidth, which we tuned via grid search over the values $1 0 ^ { - 3 } , 1 0 ^ { - 2 } , \dotsc , 1 0 ^ { 2 }$

## A.5 Results

Here we provide plots containing the results from the two experiments conducted, analogous to the plots in the main text.

## A.5.1 Mean Persistence Images for Dissimilar and Similar Sentence Pairs

See Figure 3 in the main text and Figures A.2a to A.2d.

## A.5.2 Layer Regression Experiment

See Figure 4 in the main text and Figures A.3 to A.6.

## A.5.3 Layer Correlation Experiment

See Figure 5 in the main text and Figures A.7 to A.10.

![](images/9c8542f2d4a2d7f81f577c644a75ca54fd5bf24f7a7307e09c795129c962a229.jpg)  
Figure A.1: Schematic showing the transformation of a persistence diagram in (birth, death)-coordinates (left) to (birth, death-birth)-coordinates (middle) by means of a shearing map, and the resulting persistence surface (right). Figure adapted from Huber et al. (2026).

<table><tr><td>Base model</td><td>Sentence transformer</td></tr><tr><td>MiniLM-L6-H384-uncased</td><td>all-MiniLM-L6-v2</td></tr><tr><td>MiniLM-L12-H384-uncased</td><td>all-MiniLM-L12-v2</td></tr><tr><td>distilroberta-base</td><td>all-distilroberta-v1</td></tr><tr><td>mpnet-base</td><td>all-mpnet-base-v2</td></tr><tr><td>xlm-roberta-base</td><td>paraphrase-multilingual-mpnet-base-v2</td></tr></table>

Table A.1: Sentence transformers and respective base models used in our experiments.

![](images/7caeab2b0bdba11a03f6b2287d9381cb545446ffe16c64c8f7ac8b874c39ca94.jpg)

![](images/c68857afa2d41377660df573b0dcb8108931602cbb26798da52c0efc940fc19c.jpg)

![](images/c35f4149baf3b08504e8947845ea7d949c32c2f6dc934a88ee60fc6d1ee1c28d.jpg)

![](images/c7ab7b81615e3c63157a52d3f738848821a6d76006a7a2f061d5397f89904dd8.jpg)

![](images/9a431d881d4fd21fe41661b319859c67faabbf109c1dceb74e82fb2a4eef61ea.jpg)

![](images/8a238aefb918d80c9f76f7e44a58b6a25b7e30750d1d7d5f8fbf9cdf2c13f419.jpg)  
(a) Mean persistence images for the models MiniLM-L12-H384-uncased and all-MiniLM-L12-v2.

![](images/beb4ab1cfbf478d095aa65e6b2df52e1c9c6b50184fab1fcb95d8450f82c6eff.jpg)

![](images/e3999b0021c155abe20cfd7cd514df929f031b2ab14f9187114398ca5addd8bf.jpg)  
(b) Mean persistence images for the models distilrobertabase and all-distilroberta-v1.

![](images/ca83e185a4e03d8e875e11ecdbd0e63caff1461e37cfea038be1d579871db8b7.jpg)

![](images/053abc2c690030d401086f295cf26afdc6a56335221234de91ec1cc8cde0ca03.jpg)

![](images/ab26f21bfba3b3ac1172bbda7745dc975d40386b348ec379ffdffa4ebbe9f56d.jpg)

![](images/35396dc945cdda99f2197dcab3144c3c244db669c1bb8a05669a2536b7d4a534.jpg)

![](images/5dc1ea7fbd2648e40f5790c2eb1ba34ec083baf3cd56027c8d04cd40d3a0553b.jpg)

(c) Mean persistence images for the models mpnet-base and all-mpnet-base-v2.  
![](images/4db8995733c7fe2c0b723ea9458885a32c32b5de3b3cf2e36b2e201362fa3e7e.jpg)

![](images/431f131cb5ab2122b97039b588adfdfdd2b7ca9d79914abe9d753b4d0024dee1.jpg)

![](images/3b1a282c68bffd06b2d93e6f42aa5476542d817955cd734988e74398c4b57372.jpg)  
(d) Mean persistence images for the models xlm-robertabase and paraphrase-multilingual-mpnet-base-v2.  
Figure A.2: Mean persistence images for the model pairs from Table A.1. See Figure 3 in the main text for a description of what these images capture and how they are obtained.

![](images/80aa67705eef9de11f894b98ed72d47d2315cc4dece0ca5cfb57cd29d4fe3a15.jpg)

![](images/c48d6d5479dc434e65beb7754259fa4386a884c7a10839a32fd52540fdb28716.jpg)  
Figure A.3: Results of the layer regression experiment for MiniLM-L12-H384-uncased (left) and all-MiniLM-L12- v2 (right).

![](images/d9fe730b00225d5e64f4595119c51ba38d16a3e22f6c5174081003da4f136b0b.jpg)

![](images/d3f7f8f5a7d334bf61bfb372afb5e91c12b5e118a70bce20306ecde52c61057c.jpg)  
Figure A.4: Results of the layer regression experiment for distilroberta-base (left) and all-distilroberta-v1 (right).

![](images/99ae9b2e0d969d2a220b8eb11c1060873b71c7e018037b96d87e5743a7c2d769.jpg)

![](images/3796eb54172e70c24fcaec6ad0c656e17df1389dfb67481d7c20a2efb1695286.jpg)  
Figure A.5: Results of the layer regression experiment for mpnet-base (left) and all-mpnet-base-v2 (right).

![](images/a6b55ce5bd6a2eebbf8dbc2b9f6d1063d691658e13e28af12ade347e4c0fcb3c.jpg)

![](images/dd95f079cfbdc1252b6b5034385e046b1afbea0937db9851c0c04ed52344fe6c.jpg)  
Figure A.6: Results of the layer regression experiment for xlm-roberta-base (left) and paraphrase-multilingualmpnet-base-v2 (right).

![](images/ecfaae4d7a6d4c610ba0b08fda619f7039bf0d9eb59c1c79888fb51d7b2668eb.jpg)

![](images/319d24c5eca7c5cd45c0548d8faf04b23d13dffdada87043c923c0ed27320266.jpg)  
Figure A.7: Results of the layer correlation experiment for MiniLM-L12-H384-uncased (left) and all-MiniLM-L12- v2 (right). Method names in the legend are sorted according to decreasing Spearman correlation averaged across model layers.

![](images/5150b70d56301b8dc6298deafebb031888c9da2929572ec8158e8141700ac828.jpg)

![](images/8af124a337f375a247b9a19d5c0ed687d585be3c7120ad64d915b802afa6d3f5.jpg)  
Figure A.8: Results of the layer correlation experiment for distilroberta-base (left) and all-distilroberta-v1 (right). Method names in the legend are sorted according to decreasing Spearman correlation averaged across model layers.

![](images/cc3828b5ec7547634c6b8fc7b1e646b92b7b7c6117fd79dffc0db25b332f917b.jpg)

![](images/38d9225b8364c06eddd190f386058cc660df60233d03fe095cf18539d95ab270.jpg)  
Figure A.9: Results of the layer correlation experiment for mpnet-base (left) and all-mpnet-base-v2 (right). Method names in the legend are sorted according to decreasing Spearman correlation averaged across model layers.

![](images/9022514fa8948c0ce4fde543db0328c988b1dafd7b3434b840697012874dddb1.jpg)

![](images/8bf91778c67ff909e3b0fad1f6599e168cde646232a4b95f72bb79015578a16d.jpg)  
Figure A.10: Results of the layer correlation experiment for xlm-roberta-base (left) and paraphrase-multilingualmpnet-base-v2 (right). Method names in the legend are sorted according to decreasing Spearman correlation averaged across model layers.