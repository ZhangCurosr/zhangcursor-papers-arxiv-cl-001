# Global Divergence, Local Convergence: Representation Geometry in SSMs and Transformers

Amit Ben-Artzy Roy Schwartz Computer Science, The Hebrew University of Jerusalem {amit.benartzy, roy.schwartz1}@mail.huji.ac.il

## Abstract

Recent state-space models (SSMs) such as Mamba achieve language modeling perfor mance comparable to transformers despite relying on fundamentally different architectures. This raises an important question: how do these structural differences influence the geometry and functional nature of their internal repre sentations? We study this question through a multi-scale analysis of representations in trans formers, SSMs, and hybrid architecture. First, we find that SSMs distribute their representational variance much more uniformly across di mensions compared to transformers, which are heavily dominated by a single principal direction (Fig. 1). By evaluating a hybrid architec ture, we observe that the representation space becomes increasingly skewed toward a single dominant direction after each attention layer. Next, we explore how the different geometric spread of representations impacts representa tional capacity through compressibility. Sur prisingly, we find that despite their contrasting geometric structures, both architectures exhibit tightly matched effective capacities. We fur ther investigate whether this skewed geometry affects how concepts are encoded. Using rank constrained probes, we demonstrate that both architectures encode concepts in subspaces of surprisingly similar dimensionality. Moreover, we demonstrate that the transformers’ domi nant principal direction does not inherently en code more conceptual information. Finally, we zoom in and examine the alignment between manifolds, either by analyzing representations of specific topics or by looking at the nearest neighborhoods of tokens, and find that they are highly aligned. Ultimately, our analysis sug gests that while transformers and SSMs induce different usage of latent space, they display a striking structural alignment at the level of local semantic manifolds.<sup>1</sup>

![](images/524aa28e7d8711eeec70344528dc36028b0b08c6811d6b491d737e9fc0382064.jpg)  
Figure 1: PCA of representations from Pythia and Mamba over a subset of the LAMA TREx dataset. Mamba’s representations are more spread out across the space.

## 1 Introduction

State-space models (SSMs) such as Mamba (Gu and Dao, 2024) have emerged as highly efficient alternatives to transformers (Vaswani et al., 2017), achieving competitive empirical performance on a variety of language modeling tasks (Waleffe et al., 2024). While transformers rely on global attention mechanisms, SSMs compress context into a bounded state, resulting in very different architectural priors.<sup>2</sup>

This divergence in computational paradigms raises a key question: do these distinct architectures induce fundamentally different internal representations? To investigate this systematically, we conduct a multi-scale analysis of representations across both model families. To ensure a direct and structurally equivalent comparison, we focus our analysis entirely on the residual stream, analyzing several representative models: two transformer models (Pythia, Biderman et al., 2023; and Falcon, Almazrouei et al., 2023); two SSM models (Mamba Gu and Dao, 2024 and FalconMamba, Zuo et al., 2024) and one hybrid model, alternating Mamba and transformer layers (Jamba2, Lenz et al., 2025).<sup>3</sup>

We first examine the global geometry of the models over a general corpus (Section 3). Specifically, we measure effective dimensionality via RankMe (Garrido et al., 2022) and the degree of isotropy via IsoScore (Rudman et al., 2022)—the extent to which token embeddings are uniformly distributed across all directions in space. We find a stark contrast: SSMs’ residual stream is highly isotropic and exhibits a high effective dimensionality. In contrast, transformers are notoriously anisotropic (Gao et al., 2019; Godey et al., 2024), characterized by a dominant first principal component (PC1) that explains a disproportionately large share of the variance. We further validate this trend by examining the hybrid model Jamba2, where we find that after each attention layer, there is a sharp shift toward anisotropy.

Given these contrasting global geometries, we investigate whether the two architectures use their representation spaces differently to encode specific concepts (Section 4). Using autoencoders and rank-constrained probing (Hernandez and Andreas, 2021) to measure the intrinsic dimensionality required for decoding, we find that they exhibit similar functional dimension. Furthermore, we find that despite transformers’ high alignment with a single dominant direction—it does not capture more information than SSMs’ isotropic representations. To understand whether concept information is concentrated within specific geometric dimensions, we systematically ablate the top-k principal components across both architectures. We found that both models experience a comparable decline in probe accuracy, demonstrating that these concepts are not exclusively localized in the dominant dimensions. Next, we evaluate structural similarity in Section 5. First, we apply whitened centered kernel alignment (CKA, Cristianini et al., 2001; Kornblith et al., 2019) and find that despite diverging global geometries, the local manifolds of specific concepts exhibit high representational similarity. Zooming into the token level, we evaluate the structural equivalence of local neighborhoods using mutual k-nearest neighbors (m-KNN, Wolfram and Schein, 2025) and find high alignment across architectures.

Finally, we investigate the stabilization dynamics of these properties during training in Section 6. An analysis of the transformer checkpoints demonstrates that local properties, such as CKA and m-KNN similarity, converge rapidly—within the first 4% of training. In contrast, global geometry stabilizes significantly later.

Overall, our contributions are as follows:

• Global Geometric Divergence: We provide a systematic characterization of the representation geometry in SSMs. We demonstrate that the residual streams of transformers and SSMs exhibit substantially different global geometries (anisotropic vs. isotropic).

• Local convergence: We reveal that locally, representations are highly aligned: both architectures rely on similar intrinsic dimensionalities for concept decoding and exhibit highly correlated local token neighborhoods and concept manifolds.

• Dynamic mechanisms & hybrid architectures: By tracking intermediate checkpoints, we show that local structural convergence emerges early in pretraining. Furthermore, we reveal that hybrid architectures (Jamba2) exhibit representations that oscillate between the isotropic and anisotropic geometries of their constituent layers.

## 2 Background: Transformer and SSM Architectures

In this work we study the geometrical differences between transformers and SSMs. We begin by describing both architectures. We view each architectural layer as a direct update to the residual stream.

![](images/c711f70f6fb7bb6bcf5d61635973d85037e82024ac98e7a79bbd0019580d6a5b.jpg)  
Figure 2: Layer-wise geometric analysis of transformer, State-Space and hybrid models. Metrics calculated using ≈500k representations from Fineweb-Edu. SSM models (Mamba and FalconMamba) maintain higher effective dimensionality (RankMe) and more uniform variance distribution (IsoScore) than the transformer baselines (Pythia and Falcon). In Jamba2, attention blocks consistently increase the dominance of the first principal component (PC1) which steadily declines across subsequent Mamba layers.

Transformer (Vaswani et al., 2017) block consists of interleaved attention and feed-forward (FFN) blocks. Given a hidden state x<sub>l</sub> at layer l, the residual stream is updated as:

$$
x _ { l + 1 } = x _ { l } + \mathrm { T r a n s f o r m e r } ( x _ { l } )\tag{1}
$$

The Mamba architecture (Gu and Dao, 2024) replaces the attention mechanism with a Selective state space model (SSM). The residual stream is updated as follows:

$$
x _ { l + 1 } = x _ { l } + { \bf M a m b a } ( x _ { l } )\tag{2}
$$

Internally, a Mamba block consists of an input projection, a 1D depthwise convolution, and a selective SSM S6 mixer, followed by an output projection. The SSM maintains a recurrent hidden state that evolves along the sequence dimension rather than across layers.

Importantly, our analysis operates at the level of the layer-wise residual stream x<sub>l</sub>, and does not investigate the internal sequence-level state maintained within the Mamba block. This allows us to compare Transformers and SSM-based models within a shared representation framework defined by residual stream dynamics.

## 2.1 Models

In our experiments, we evaluate models from two transformer model families: Pythia (Biderman et al., 2023) and Falcon (Almazrouei et al., 2023);<sup>4</sup> and two SSM model families: Mamba (Gu and Dao, 2024) and FalconMamba (Zuo et al., 2024). The shared tokenization scheme between these suites facilitates controlled, one-to-one comparisons. Our analysis also evaluates Jamba2 (Lenz et al., 2025), a hybrid model that interleaves Mamba blocks and Transformer blocks.

## 3 Global Geometry of Representations

Our goal in this work is to study the different geometrical properties of transformers and SSMs. We start by analyzing their effective dimensionality across layers. Particularly, we ask how many hidden state dimensions are being actively used by each architecture. To do so, we consider a matrix $X \in \mathbb { R } ^ { N \times d }$ containing the hidden representation (of dimension d) of $N = \mathrm { t o k e n s }$ from a given layer ℓ, and compute three complementary metrics. First, we compute RankMe (Garrido et al., 2022), which measures the effective rank as the Shannon entropy of the normalized singular value distribution $p _ { i } = { \sigma } _ { i } / \sum _ { j } { \sigma } _ { j }$ derived from the singular values $\sigma _ { i }$ of the representation matrix X. To assess global structure, we employ IsoScore (Rudman et al., 2022), which quantifies isotropy by measuring the distance between the data covariance and the identity matrix; a score of 1 indicates that representations use all available dimensions equally. Finally, we report the PC1 explained variance. A high proportion of variance captured by the first principal component suggests representation collapse into a dominant direction, signaling significant anisotropy. We compute each metric on Pythia-1.4B, Falcon-7B, Mamba-1.4B, FalconMamba-7B, and Jamba2-3B.<sup>5</sup>

We compute hidden representations from all layers of all models, using ≈500K representations obtained over FineWeb-Edu (Penedo et al., 2024), a curated pre-training corpus filtered for high-quality educational content from the broader FineWeb dataset.

As illustrated in Fig. 2, SSM-based models sustain a higher RankMe than their transformer counterparts of the same size across all but the final layers. Additionally, transformer models exhibit substantially lower IsoScores and higher PC1 explained variance (regardless of model size). Together, these metrics indicate that SSM models learn a highly isotropic, uniformly distributed latent representation, whereas transformer representations are highly anisotropic, collapsing into a cone-like geometry dominated by a few principal directions.

To determine whether this anisotropy is specifically tied to the attention mechanism, we analyze the hybrid model Jamba2. In Jamba2, attention layers consistently induce a drop in RankMe alongside an increase in both IsoScore and PC1 explained variance, driving the representations toward a Transformer-like geometry. Conversely, subsequent Mamba layers steadily reverse this trend, restoring the representational structure to one more characteristic of SSMs. This strongly suggests the attention mechanism as the primary driver of dimensional collapse within the network, consistent with prior observations (Godey et al., 2024).

## 4 Quantifying the Effective Dimensionality of Representations

In Section 3, we investigated the spatial distribution of representations across R<sup>d</sup>, and observed that SSM models feature a distributed latent space while transformers display high anisotropy. Next, we explore how the different geometric spread impacts the capacity of representations. Following this, we assess whether the structural skewness of transformers along a dominant axis correlates with increased information capacity.

## 4.1 Model Capacity Limits

While isotropy provides insight into the geometric structure of representations, it does not capture how much information the representations encode. To move beyond geometry, we evaluate representational capacity through compressibility. We train a two-layer Autoencoder (AE) for each layer across various rank constraints r. These models are optimized using an L -reconstruction loss. For evaluation, we measure the Kullback-Leibler (KL) divergence between the model’s original output distribution and the distribution obtained when replacing a single native layer representation of a single token x<sub>l</sub> with the AE reconstructions $\operatorname { A E } ( x _ { l } )$ . The AEs are trained on roughly 4 million tokens from the FineWeb-Edu corpus.

As illustrated in Fig. 3(a), restricting the bottleneck rank r induces a highly similar degradation in downstream KL divergence for both architectures across all layers, other than the last SSM layer which is much more sensitive. This similarity demonstrates that their effective representational capacity is tightly matched, despite the varying geometrical measures and varying architectures.

Additional results for more rank constraint values are shown in Fig. 10, Fig. 11 of Section D.

## 4.2 The Subspace Geometry of Concepts

We have so far observed that on the one hand SSM representations are uniformly distributed, while transformers are highly anisotropic; and on the other that both architecture behave remarkably similarly when constraining their rank. We now turn to ask how localized individual concepts are within the representation space.

To answer this, we shift our focus from the global token distribution to specific, factual concept manifolds. We evaluate the models using the TREx split of the LAMA dataset (Petroni et al., 2019), which consists of Wikidata triples—comprised of a subject, relation, and object—aligned with corresponding natural language sentences from Wikipedia. Specifically, we present the models with these texts and extract the hidden states from each layer at the final token of the object. Samples from this dataset are available in Section C.

This methodology allows us to isolate and analyze the latent manifolds of diverse semantic relations, such as occupation, genre, and capital. We leverage this constrained setting to perform rankand PCA-constrained probing, allowing us to characterize the intrinsic dimensionality required to encode localized relational knowledge across architectures.

![](images/4f3054d3055a0a3e183e168c57ea055b9c4cbc2f73e898ab4cdf526d8edcb81b.jpg)  
Figure 3: Probing under rank and PC ablations. Left to Right: (a) Original layer representations $x _ { l }$ are replaced by their reconstructions $\operatorname { A E } ( x _ { l } )$ using a two-layer autoencoder with varying bottleneck sizes. We then compute the KL divergence between the original and reconstructed distributions to measure information loss. Both architectures exhibit highly similar performance degradation across ranks, indicating tightly matched effective representational capacity, except for the more sensitive final SSM layer. (b) Average rank-constrained probe accuracy across LAMA-TREx relations. Similar rank constraints yield comparable decoding accuracies across model types, despite significant differences in representation geometry. (c) Probe accuracy after removing or (d) retaining the top-k PCA components; results remain consistent across models despite transformers’ higher anisotropy.

Intrinsic dimensionality of relational facts How does the global distribution of representations across R<sup>d</sup> influence the spatial localization of distinct concepts and the minimal subspace dimensionality needed for their extraction?

To measure the dimensional footprint of localized concepts, we decode relational facts using rank-constrained probes. We factorize the probe $W \in \mathbb { R } ^ { d \times m }$ as $W = A B$ , where $A \in \mathbb { R } ^ { d \times r }$ and $B \in \mathbb { R } ^ { r \times m }$ are down-projection and up-projection matrices, respectively, with r representing the rank constraint and m denoting number of classes for the object in the LAMA TREx dataset. As shown in Fig. 3(b), applying identical rank constraints yields closely matched decoding accuracies across both architectures and all layers. This indicates that these relational facts are embedded within subspaces of similar rank requirements.

Robustness to spectral ablation Do the transformer dominant variance directions actually encode more semantic information than SSMs’ distributed components? We test this using PCAconstrained decoding. For each LAMA TREx relation, we compute the principal components of the representation space and evaluate the degradation of probing performance after either retaining or removing the top-k PCA directions. First, we see that despite the transformer models’ massively dominant PC1 variance, removing the top PCA directions leads to a generally similar performance degradation across architectures (Fig. 3(c)). Second, we see that retaining the top-k PCA dimension degrades the probe at similar rates, across both architectures and scales (Fig. 3(d)). This suggests that the dominant directions in the global geometry of transformers do not serve as the primary axes for factual knowledge, but instead reflect broader structural properties of the transformer representation space.

## 5 Alignment of Semantic Concepts and Local Neighborhoods

Building upon the global geometric differences identified in Section 3 and the equivalent intrinsic ranks observed in Section 4, we turn our attention to the direct alignment of their representational spaces, quantifying the extent to which these diverging architectures preserve structural equivalence.

## 5.1 Alignment of Concept Manifolds

Following the observation that transformers and SSMs compress factual knowledge into subspaces of equivalent dimensionality, we investigate whether these subspaces are also structurally aligned. We quantify this alignment using centered kernel alignment (CKA; Cristianini et al., 2001; Kornblith et al., 2019), which evaluates representational similarity by comparing the inner product structures of different feature spaces.

![](images/e34aa8389731058d598bd69af2cc3e0142c73b30fa5cd287cc27741205b1216a.jpg)  
Figure 4: Layer-wise ZCA-CKA alignment on (top) FineWeb-Edu and (bottom) averaged WikiData relations from LAMA TREx. “Mean max” indicates the mean of the maximum CKA alignment across layers. Models show pronounced differences over the general corpus but exhibit high structural similarity when restricted to specific relations.

Linear CKA is heavily weighted by the leading eigenvalues of the covariance matrix, making it highly sensitive to the underlying variance profile of the representations. This introduces an artifact when comparing transformers—which retain the high anisotropy—against SSMs, which are highly isotropic. We mitigate this discrepancy by applying zero-phase component analysis (ZCA), standardizing both spaces to enable a direct structural comparison.

ZCA transforms representations by applying the inverse square root of their covariance matrix $( \Sigma ^ { - 1 / 2 } )$ . This transforms representations such that their global covariance becomes the identity matrix, thereby reducing the influence of anisotropic second-order statistics on similarity estimates. Crucially, for each model and layer, we compute Σ globally using 10K matched token representations from FineWeb-Edu. Relation-wise whitening would normalize away relation-specific covariance structure, making all local manifolds isotropic by construction and thereby trivializing geometric comparisons.

For relation-specific LAMA mean-centered representations X and Y from the two models, the globally ZCA-whitened representations are:

$$
\tilde { X } = X \Sigma _ { X } ^ { - 1 / 2 } , \quad \tilde { Y } = Y \Sigma _ { Y } ^ { - 1 / 2 }
$$

We then compute linear CKA on these whitened local manifolds using the Frobenius norm formulation:

$$
\mathbf { C K A } ( { \tilde { X } } { \tilde { X } } ^ { \top } , { \tilde { Y } } { \tilde { Y } } ^ { \top } ) = { \frac { \| { \tilde { X } } ^ { \top } { \tilde { Y } } \| _ { F } ^ { 2 } } { \| { \tilde { X } } ^ { \top } { \tilde { X } } \| _ { F } \| { \tilde { Y } } ^ { \top } { \tilde { Y } } \| _ { F } } }
$$

This global whitening allows CKA to better reflect relation-specific geometric similarity.

Manifold alignment results Analysis via ZCA-CKA reveals a clear distinction between global and local manifold geometries (Fig. 4). Specifically, representational similarity between models evaluated on the same relation is substantially higher than the baseline similarity observed across the global corpus. Ultimately, despite exhibiting divergent global geometries, both architectures arrive at structurally analogous sub-manifolds when compressing specific factual concepts.

## 5.2 Neighborhoods Level Alignment

![](images/534eb55a64e0bfaea4728943424212d16e721017681df51ee76ae9af59328d3c.jpg)  
Figure 5: m-KNN across models and layers, including random initializations and Pythia’s checkpoints. We find that the local manifold of representations is similar across models in similar depths; and this similarity is achieved at Pythia’s checkpoint 5K.

Having established that topic-specific manifolds align despite global geometric differences, we now examine the representation space at its absolute finest granularity: the local neighborhoods of individual tokens. Specifically, we ask: do representations from transformers and SSMs cluster semantic information similarly at the token level?

To this end, we use the mutual-KNN metric. For token representations $x _ { i } \in X$ and $y _ { i } \in Y$ , with k nearest neighborhoods $\mathcal { N } _ { k } ^ { X } ( x _ { i } )$ and $\mathcal { N } _ { k } ^ { Y } ( y _ { i } )$ , the overlap across N tokens is:

$$
\operatorname { m - K N N } @ k = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { | \mathcal { N } _ { k } ^ { X } ( x _ { i } ) \cap \mathcal { N } _ { k } ^ { Y } ( y _ { i } ) | } { k }\tag{3}
$$

As all models we consider share an identical tokenizer, we can directly compute the nearestneighbor overlap for identical tokens across their respective representation spaces. We define nearest neighbors with cosine similarity. We measure mutual k-Nearest Neighbors (m-KNN) on 10,000 token samples from FineWeb-Edu and $k = 1 0$ Importantly, m-KNN relies entirely on relative distances; it is invariant to the global linear transformations and scaling disparities.

Transformers and SSMs converge As illustrated in Fig. 5, the m-KNN overlap between fully trained transformer and SSM models is remarkably high across corresponding depths. This demonstrates that despite architectural differences, both models cluster individual tokens into strongly aligned microscopic neighborhoods. Furthermore, while these local topologies evolve significantly as representations propagate forward through the residual stream, transformers and SSMs track this evolution in parallel, maintaining high mutual overlap at equivalent network depths. Additional results for Falcon-7B and FalconMamba-7B are shown in Fig. 12 (Section F).

## 6 Temporal Dynamics of Alignment

We next investigate when these representational similarities emerge by analyzing Pythia’s intermediate checkpoints.<sup>6</sup>

We observe that the local structures align remarkably early. As shown by CKA (Fig. 4) and m-KNN (Fig. 5), Pythia’s representations match Mamba’s after just 5,000 training steps (∼4% of training). At this point, cross-architecture and cross-scale similarity are already high, matching Pythia’s selfsimilarity to its own final state. This rapid saturation occurs in both general and concept-specific contexts, though absolute similarity is higher in the latter.

In contrast, global geometry matures much slower (Fig. 6). Global metrics like IsoScore and PC1 Explained Variance do not stabilize until approximately 25% of training. Furthermore, while these metrics show monotonic convergence, the intrinsic dimensionality (RankMe) fluctuates nonmonotonically—first increasing, then decreasing. This suggests that while local neighborhoods lock in almost immediately, the global space undergoes structural expansion and compression before settling.

## 7 Related Work

Representational convergence The inquiry into whether distinct neural networks arrive at shared internal representations originated with foundational work by Li et al. (2015), who demonstrated that while independently trained models converge onto identical semantic subspaces, they distribute this information across entirely different individual basis vectors. Extending this, The Platonic representation hypothesis (Huh et al., 2024) posits that internal representation spaces scale toward a shared statistical model of reality, independent of architecture, objective, or modality. Recent empirical work has shown that dataset overlap and task similarity serve as the primary drivers of this alignment (Li et al., 2025). However, this convergence may not be uniform; closely aligning with our own findings, Gröger et al. (2026) demonstrated that vision and language models frequently converge locally while maintaining global topological differences.

Comparing transformers and SSMs A growing body of work examines whether non-attention architectures, such as selective State-Space Models (SSMs) like Mamba, converge on the internal representations used by transformers. Recent findings strongly support this universality hypothesis across multiple granularities. At the representation level, Wang et al. (2025) applied Sparse Autoencoders (SAEs) to demonstrate that both architectures extract a highly similar set of interpretable features. This convergence extends to specific semantic sub-domains; for instance, Fu et al. (2026)

observed that both Mamba and transformers independently develop matching geometries for numerical tokens, achieving spectral and geometric convergence on periodic features. Furthermore, this alignment persists when analyzing operational mechanics: Yoo (2026) decomposed layer-wise updates across both model families and found that full updates are geometrically dominated by a highly aligned tokenwise transformation.

The geometry and anisotropy of latent spaces The geometric properties of these representations often dictate model utility and performance. In transformers, the self-attention mechanism is a known source of anisotropy—a phenomenon where representations occupy a narrow cone in the latent space (Gao et al., 2019; Godey et al., 2024; Timkey and van Schijndel, 2021). While some models, such as the Pythia family, exhibit lower anisotropy than other transformers (Machina and Mercer, 2024), we find they remain significantly more anisotropic than SSMs. This geometric structure is not merely a byproduct but a predictor of performance, as measures of isotropy correlate with LLM efficacy (Li et al., 2026b). There has also been previous work which tried to learn about the subspaces responsible for specific attributes; Hernandez and Andreas (2021) demonstrated that many attributes reside in low-dimensional linear spaces in both BERT and ELMO.

Dynamic and layer-wise evolution Representational geometry is not static; it evolves significantly during training (Li et al., 2026a) and across layers (Wolfram and Schein, 2025; Skean et al., 2025; Valeriani et al., 2023). Previous work has also looked into similarities to tokens across contexts (Ethayarajh, 2019) and across architectures (Wu et al., 2020).

## 8 Conclusion

In this work, we investigated the internal representations of SSM and transformer architectures, analyzing their latent space utilization, geometric properties, and functional capacity. We found that SSM representations are highly isotropic, whereas transformers are heavily anisotropic. In hybrid models, we observed architectural oscillations, where the representation space fluctuates between isotropic and anisotropic states across layers.

Despite these stark global geometric differences, both architectures exhibit surprising similarities. We demonstrated that their effective dimensionality—measured via autoencoders and rankconstrained bottlenecks—is remarkably similar, and the transformer’s singular dominant direction does not inherently capture more conceptual information than SSMs’ distributed space.

Structural alignment between architectures proved highly uneven across scope: high when restricted to individual semantic relations, but low over the general corpus. This local convergence, like the local convergence in CKA and m-KNN more broadly, also emerges early in training, ahead of the slower-stabilizing global geometry.

Taken together, our results indicate that transformers and SSMs are geometrically distinct and differ in how they organize latent space, yet converge on highly aligned local representational structures.

## Limitations

While our multi-scale analysis provides systematic insights into the representational mechanics of state-space, hybrid, and transformer architectures, we acknowledge the following boundaries to our scope:

Correlational vs. causal analysis Our evaluation of concept encoding relies primarily on rankconstrained probing, principal component ablations, and structural alignment metrics (CKA, m-KNN). While these methods demonstrate that concept information can be decoded with equivalent linear capacity and share local manifold geometries, they do not establish a direct causal link to downstream generation (Elazar et al., 2021; Ravichander et al., 2021).

## 9 Acknowledgments

This work was supported in part by the Israel Science Foundation (grant no. 2045/21) and by a research gift from Google.

This manuscript was drafted and refined with the assistance of large language models (LLMs). All content was reviewed and verified by the authors.

## References

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. The Falcon Series of Open Language Models. arXiv.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, and 1 others. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR.

Nello Cristianini, John Shawe-Taylor, Andre Elisseeff, and Jaz Kandola. 2001. On kernel-target alignment. In Proceedings ofthe 15th International Conference on Neural Information Processing Systems: Natural and Synthetic, NIPS’01, page 367–373, Cambridge, MA, USA. MIT Press.

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. 2021. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. Transactions ofthe Associationfor Computational Linguistics, 9:160–175.

Kawin Ethayarajh. 2019. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 55–65, Hong Kong, China. Association for Computational Linguistics.

Deqing Fu, Tianyi Zhou, Mikhail Belkin, Vatsal Sharan, and Robin Jia. 2026. Convergent Evolution: How Different Language Models Learn Similar Number Representations. arXiv.

Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, and Tieyan Liu. 2019. Representation degeneration problem in training natural language generation models. In International Conference on Learning Representations.

Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. 2022. RankMe: Assessing the downstream performance of pretrained selfsupervised representations by their rank. arXiv.

Nathan Godey, Éric Clergerie, and Benoît Sagot. 2024. Anisotropy is inherent to self-attention in transformers. In Proceedings of the 18th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 35–48, St. Julian’s, Malta. Association for Computational Linguistics.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The Llama 3 Herd of Models. arXiv.

Fabian Gröger, Shuo Wen, and Maria Brbic. 2026.´ Revisiting the Platonic Representation Hypothesis: An Aristotelian View. arXiv.

Albert Gu and Tri Dao. 2024. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling.

Evan Hernandez and Jacob Andreas. 2021. The lowdimensional linear geometry of contextualized word representations. In Proceedings ofthe 25th Conference on Computational Natural Language Learning, pages 82–93, Online. Association for Computational Linguistics.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. 2024. Position: The platonic representation hypothesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20617–20642. PMLR.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of Neural Network Representations Revisited. arXiv.

Barak Lenz, Opher Lieber, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, Daniel Gissin, Daniel Jannai, Dor Muhlgay, Dor Zimberg, Edden M. Gerber, Elad Dolev, Eran Krakovsky, Erez Safahi, Erez Schwartz, and 42 others. 2025. Jamba: Hybrid transformer-mamba language models. In The Thirteenth International Conference on Learning Representations.

Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, and Blake Aaron Richards. 2026a. Tracing the representation geometry of language models from pretraining to post-training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yanhong Li, Ming Li, Karen Livescu, and Jiawei Zhou. 2026b. On the predictive power of representation dispersion in language models. In The Fourteenth International Conference on Learning Representations.

Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. 2015. Convergent learning: Do different neural networks learn the same representations? In Proceedings of the 1st International Workshop on Feature Extraction: Modern Questions and Challenges at NIPS 2015, volume 44 of Proceedings of Machine Learning Research, pages 196–212, Montreal, Canada. PMLR.

Zeyu Michael Li, Hung Anh Vu, Damilola Awofisayo, and Emily Wenger. 2025. Causes and Consequences of Representational Similarity in Machine Learning Models. arXiv.

Anemily Machina and Robert Mercer. 2024. Anisotropy is not inherent to transformers. In Proceedings of

the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4892–4907, Mexico City, Mexico. Association for Computational Linguistics.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben al-ˇ lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. 2024. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language models as knowledge bases? In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2463–2473, Hong Kong, China. Association for Computational Linguistics.

Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy. 2021. Probing the probing paradigm: Does probing accuracy entail task relevance? In Proceedings of the 16th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 3363–3377, Online. Association for Computational Linguistics.

William Rudman, Nate Gillman, Taylor Rayne, and Carsten Eickhoff. 2022. IsoScore: Measuring the uniformity of embedding space utilization. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3325–3339, Dublin, Ireland. Association for Computational Linguistics.

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. 2025. Layer by layer: uncovering hidden representations in language models. In Proceedings of the 42nd International Conference on Machine Learning, ICML’25. JMLR.org.

William Timkey and Marten van Schijndel. 2021. All bark and no bite: Rogue dimensions in transformer language models obscure representational quality. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4527–4546, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Lucrezia Valeriani, Diego Doimo, Francesca Cuturello, Alessandro Laio, Alessio Ansuini, and Alberto Cazzaniga. 2023. The geometry of hidden representations of large transformer models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All

you Need. Advances in Neural Information Processing Systems, 30.

Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. 2024. An Empirical Study of Mambabased Language Models. arXiv.

Junxuan Wang, Xuyang Ge, Wentao Shu, Qiong Tang, Yunhua Zhou, Zhengfu He, and Xipeng Qiu. 2025. Towards universality: Studying mechanistic similarity across language model architectures. In The Thirteenth International Conference on Learning Representations.

Christopher Wolfram and Aaron Schein. 2025. Layers at similar depths generate similar activations across LLM architectures. In Second Conference on Language Modeling.

John Wu, Yonatan Belinkov, Hassan Sajjad, Nadir Durrani, Fahim Dalvi, and James Glass. 2020. Similarity analysis of contextual word representation models. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 4638–4655, Online. Association for Computational Linguistics.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and 43 others. 2024. Qwen2 Technical Report. arXiv.

Jun-Sik Yoo. 2026. On the Geometric Structure of Layer Updates in Deep Language Models. arXiv.

Jingwei Zuo, Maksim Velikanov, Dhia Eddine Rhaiem, Ilyas Chahed, Younes Belkada, Guillaume Kunsch, and Hakim Hacid. 2024. Falcon Mamba: The First Competitive Attention-free 7B Language Model. arXiv.

## A Hyperparameters for Experiments

To ensure reproducibility, all code is publicly available and included as part of this ARR submission.

• Global Geometric Metrics (Section 3): Evaluated on 1,500 samples from FineWeb-Edu with a maximum sequence length of 512, yielding a total of approximately 500K tokens.

• Autoencoder Training (Section 4.1): Trained for a single epoch on 10,000 FineWeb-Edu samples (maximum sequence length of 512) using a learning rate of 1e-4 and ReLU activations. It was evaluated over 1000 samples (maximum sequence length of 512).

• Ablated Probing (Section 4.2): Trained for 5 epochs with a learning rate of 1e-3 and an 80%/20% train-evaluation split. The training data was capped at a maximum of 10,000 samples per LAMA TREx relation.

• ZCA-CKA (Section 5.1): For FineWeb-Edu, metrics were computed over ≈10,000 tokens (derived from 100 samples of length 100). For LAMA TREx, we utilized the full dataset but sampled exactly one context sequence per unique factual triplet (subject, relation, object) to control for redundant semantic structures.

• m-KNN (Section 5.2): Computed using 10,000 representations extracted from FineWeb-Edu.

## B Additional Geometric Metrics

## B.1 Global Geometric Metrics for Pythia Checkpoints

We analyze Pythia’s intermediate training checkpoints using the same metrics as described in Section 3. As shown in Fig. 6, we find distinct optimization trajectories over the course of training. Specifically, RankMe exhibits a nonmonotonic pattern, characterized by an initial increase followed by a subsequent decline. In contrast, IsoScore and PC1 explained variance converge monotonically toward their final values, with the primary stabilization occurring rapidly within the first 25% of training steps.

## B.2 Global Geometric Metrics of Additional Models

We investigate the geometric metrics of additional models, namely for Llama (Grattafiori et al., 2024) and Qwen (Yang et al., 2024). As shown in Fig. 7, the empirical results indicate that while model scale inherently influences RankMe, SSMs consistently maintain a higher rank than Transformers when controlling for scale. Conversely, both IsoScore and PC1 explained variance remain largely invariant to scale.

## B.3 Variance Analysis of Geometric Metrics

To verify the stability of our geometric metrics, we re-evaluate both models on 5 random seeds, each sampling a different subset of the FineWeb-Edu sample-10BT split, with each subset comprising ≈ 500K tokens. For each metric and layer, CV% is computed across seeds; Table 1 reports the mean of these values across layers, alongside the maximum layer-wise CV% in brackets.

<table><tr><td>Metric</td><td>Pythia-1.4B</td><td>Mamba-1.4B</td></tr><tr><td>RankMe</td><td>0.13% [0.18%]</td><td>0.11% [0.34%]</td></tr><tr><td>IsoScore</td><td>0.52% [0.98%]</td><td>0.50% [1.%]</td></tr><tr><td>PC1 Exp. Var.</td><td>0.18% [0.48%]</td><td>0.36% [0.77%]</td></tr></table>

Table 1: Stability of representation geometry metrics across 5 random seeds, measured via coefficient of variation (CV%). For each metric, we report the mean CV% across layers, with the maximum layer-wise CV% in brackets. Lower values indicate greater stability.

## C Samples from LAMA TREx

Table 2 presents samples from the LAMA TREx dataset (Petroni et al., 2019), which spans 41 distinct Wikidata relation types. The dataset pairs Wikidata triples (subject, relation, object) with aligned natural language sentences from Wikipedia.

## D Additional Results for Model Capacity Limits

We extend the experiment in Section 4.1 to additional autoencoder rank constraints r. As shown in Fig. 10 (additional rank constraints) and Fig. 11 (mean ± std. over 5 seeds), the degradation pattern remains consistent across models for every constraint tested. To quantify this agreement, we compute the Pearson correlation and the Concordance Correlation Coefficient (CCC) between the per-layer KL-divergence curves of the two architectures (Fig. 8, Table 3). Correlation is consistently high (Pearson > 0.84 across all r) and broadly increases with rank, from 0.857 at r = 64 to 0.911 at r = 512; CCC follows the same trend, rising from 0.638 to 0.830. This indicates that agreement between architectures strengthens as the bottleneck is relaxed, i.e., degradation curves are most alike where capacity constraints are less severe.

## E Additional CKA Results

In Section 5, ZCA-whitening was used to address the fact that the two models have markedly different spectral properties. To test whether the local/global alignment gap reported there is a genuine property of the representations rather than an artifact of this spectral mismatch, we recompute the maximally-similar-layer CKA under ZCAwhitening at several strengths α by computing $\Sigma ^ { - \alpha / 2 }$ over both LAMA TREx (mean across relations) and FineWeb-Edu (Fig. 9). Even without whitening (α = 0), LAMA TREx representations are substantially more aligned than FineWeb-Edu representations (layer mean 0.78 vs. 0.52), indicating that the gap is not an artifact of the ZCA whitening.

Mamba-1.4B vs Pythia-1.4B  
![](images/d753b388d3f67a21eaa9a60177056035ffc1b8c0136d3218eb04b51fe2e64775.jpg)  
Figure 6: Layer-wise Geometric Analysis of Pythia-1.4B Checkpoints over ≈500K representations from FineWeb-Edu.

![](images/2d3ff2bca561dafb77594171df752f7cc203ed1a37d2b8fb41d66bac0481c6bc.jpg)  
Figure 7: Layer-wise geometric analysis of state-space and transformer models over ≈500K representations from FineWeb-Edu. Transformers retain a single dominant axis across sizes and families, while Mamba’s representation distribution remains uniform across different scales.

![](images/dd57c50f30bdd8cbb9db2ef35566bcae65eb22cc3229b12b2ec026f3d3b3c802.jpg)  
Figure 8: Pearson and CCC correlation between the KL-divergences of Mamba-1.4B and Pythia-1.4B.

![](images/6994d83ac7eddf9ca78a2d70a63cee6c2ccc6078ed80ae902743e9748b3403a2.jpg)  
Figure 9: Maximum pairwise CKA similarity across all Mamba-1.4B, Pythia-1.4B layer pairs, plotted as a function of ZCA-whitening strength α, for LAMA TREx (mean over relations) and FineWeb-Edu.

<table><tr><td>Relation</td><td>ID</td><td>Sentence</td></tr><tr><td>P136 (genre)</td><td>19918</td><td>The Simpsons is the longest-running American sitcom, the longest-running American animated program, and in 2009 it surpassed Gunsmoke as the longest-running American scripted primetime television series.</td></tr><tr><td>P30 (continent)</td><td>336</td><td>Historical records of Western culture in Europe begin with Ancient Greece and Ancient Rome.</td></tr><tr><td>P101 (field of work)</td><td>5656</td><td>Brouwer&#x27;s fixed-point theorem is a fixed-point theorem in topology, named after Luitzen Brouwer.</td></tr><tr><td>P106 (occupation)</td><td>15826</td><td>That astronaut, Dumitru Prunariu is today&#x27;s president of Romanian Space Agency.</td></tr><tr><td>P37 (official language)</td><td>16094</td><td>Australian English (AusE, AuE, AusEng, en-AU) is a major variety of the English language and is used throughout Australia.</td></tr><tr><td>P31 (instance of)</td><td>2448</td><td>Another version of Hahn-Banach theorem is known as Hahn-Banach separation theorem or the separating hyperplane theorem, and has numerous uses in convex geometry.</td></tr></table>

Table 2: Examples of LAMA TREx instances across distinct relation categories. Underlined words indicate the object.

<table><tr><td>AE Dimension</td><td>Pearson</td><td>CCC</td></tr><tr><td>64</td><td> $0 . 8 5 7 \pm 0 . 0 2 7$ </td><td> $0 . 6 3 8 \pm 0 . 0 4 1$ </td></tr><tr><td>128</td><td> $0 . 8 4 9 \pm 0 . 0 2 8$ </td><td> $0 . 6 5 8 \pm 0 . 0 4 5$ </td></tr><tr><td>256</td><td> $0 . 9 0 1 \pm 0 . 0 1 5$ </td><td> $0 . 7 4 4 \pm 0 . 0 3 7$ </td></tr><tr><td>512</td><td> $0 . 9 1 1 \pm 0 . 0 1 2$ </td><td> $0 . 8 3 0 \pm 0 . 0 2 0$ </td></tr></table>

Table 3: Mean Pearson and Lin’s CCC between per-layer KL-degradation curves of Mamba-1.4B and Pythia-1.4B, per rank constraint r (averaged across layers). The final SSM layer is excluded: its KL divergence is disproportionately high, unlike the corresponding Transformer layer (Section 4.1).

To confirm the adequacy of a 10K sample size for ZCA-CKA, we computed the coefficient of variation (CV%) for Mamba-1.4B versus Pythia-1.4B across varying seeds. On FineWeb-Edu, using 2.5K-sample subsets, we observed a low CV% across all layer pairs (mean = 0.15%, max = 1.34%). We found similarly low variance over 5 seeds from the LAMA TREx split (mean = 0.56%, max = 3.31%). These results indicate that our chosen sample size is highly stable.

## F Additional Results for m-KNN

We further validate the robustness of our m-KNN results from Section 5.2 along two axes: generalization to unseen architectures, and stability with respect to sampling.

Generalization to new architectures. We repeat the analysis on Falcon-7B and FalconMamba-7B, an architecture pair not included in the original comparison. As shown in Fig. 12, we observe the same trend as before: local neighborhood topology remains highly aligned at corresponding relative depths, confirming that this alignment is not specific to the Mamba/Pythia family.

Stability with respect to sampling. To verify that our sample size is sufficient for stable estimates, we recompute m-KNN on five independently drawn batches of 2,000 samples each, for the Mamba and Pythia models (1.4B and 2.8B). For each of the layer-pair comparisons, we compute the mean and standard error across the five batches (n = 5). The mean SEM across all pairs was 0.0047 (range: 0.0003–0.0113), and the mean standard deviation was 0.0104 (range: 0.0008–0.0253), confirming that our results are stable with respect to sampling variation.

K = 1 K = 25   
K = 5

(c) Remove top-k PCA  
![](images/f8e5710b4f5c5585fd0fae3904467baed7e01b22e1d69873846e58b98fa521a8.jpg)

![](images/dcf823b7434dfbafd3a37d8c424dae43988b90bc048259521912a07751083e7d.jpg)

![](images/c06b96792d27daf62e911c22de5c8e907e635bac2a62473d33b2d15f9343333f.jpg)

![](images/564175511e994607665977b4d0d9e215b9dc372389af6cb888ecbbbafe230afe.jpg)  
Figure 10: Per-layer KL-divergence between original and reconstructed representations across additional constraint dimensions.

![](images/082e11ae6756ad8430ab315b8c77c9135b8ba8ecdcd7b718ef06334bbee10365.jpg)  
Dim 64 Dim 512-- Dim 128

![](images/7d5383da6c8cc59079942e917e9050e2ccbbe3ef12616a6a773c85ae95eaba58.jpg)

![](images/706d7bb9889ef32923a8f756e4c05733d2f5341f175bd7d3f57ebed231873826.jpg)

(d) Retain top-k PCA  
![](images/e29e0f3836da200a6a3214f8052e6a92681dfa0aed9f93db32ac564d292b3e66.jpg)  
Figure 11: Per-layer KL-divergence between original and reconstructed, with mean ± standard deviation over 5 random seeds.

![](images/1013b1850dc8b1e8401eecdbcb78271258f4ff2d3c0dd8f347aab4feaa0bb433.jpg)  
Figure 12: m-KNN across models and layers for Falcon-7B and FalconMamba-7B. We find that the local manifold of representations is similar across models at similar depths.