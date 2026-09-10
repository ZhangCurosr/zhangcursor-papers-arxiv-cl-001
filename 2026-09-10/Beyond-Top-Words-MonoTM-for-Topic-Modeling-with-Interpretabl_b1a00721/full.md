# Beyond Top Words: MonoTM for Topic Modeling with Interpretable Monosemantic Features

Una Joh and Bei Yu School of Information Studies Syracuse University {sjoh01,byu}@syr.edu

## Abstract

Topic models summarize large text corpora, but top-ranked words often provide only a limited representation of topic semantics. Sparse autoencoders (SAEs) offer a way to move beyond word-level descriptors by extracting interpretable features from dense representations, yet how feature interpretability relates to topicinference quality remains unclear. We introduce MonoTM, an interpretable topic modeling framework that decouples these roles. Across three benchmark corpora, we show that document-topic mixture estimation and semantic interpretation favor different SAE configurations and feature subsets. MonoTM estimates mixtures from the full SAE bag-offeatures representation and, with them fixed, learns topic descriptors over a separate vocabulary of corpus-grounded semantic features. This design preserves global topic structure while representing topics with semantic units more meaningful than individual words, making them more useful for downstream corpus analysis.

## 1 Introduction

Text is a primary source of evidence across the social sciences and the humanities, from political speeches and news coverage to historical archives and literary corpora. The scale of these collections motivates computational text-as-data methods; topic modeling, especially Latent Dirichlet Allocation (LDA) and its extensions, has therefore become a popular tool for exploratory corpus mapping and thematic summarization (Blei et al., 2003). Yet its role as scholarly evidence remains contested (Da, 2019; Grimmer and Stewart, 2013; Shadrova, 2021). A recent review of topic modeling validation in computational social science finds that validation and reporting practices remain heterogeneous and show little convergence across studies (Bernhard-Harrer et al., 2025).

One source of this issue is that most topic models define each topic as a probability distribution over words, and users typically interpret topics by their highest-probability words. However, assigning a concise, human-readable meaning or label to each inferred topic can be challenging in practice (Chang et al., 2009; Mei et al., 2007). Although recent neural topic models incorporate pretrained language models or document embeddings to capture semantic information beyond raw word counts, these models still typically extract top-ranked words or phrases post hoc to describe each topic to users (Bianchi et al., 2021a; Dieng et al., 2020; Grootendorst, 2022; Kardos et al., 2025).

This creates a gap between what topic models output—topics as distributions over words—and what many analysts want: semantic units that support interpretation at an intermediate level between reading individual documents and identifying coarse corpus-level themes. Beyond topic representation, words are rarely adequate semantic units for supporting meaningful downstream corpus analysis, such as analyzing relations between topics or documents based on word overlap.

Recent progress in mechanistic interpretability, particularly sparse autoencoders (SAEs) and autointerpretability methods, offers a way to extract human-readable semantic features from document embeddings produced by pretrained encoders. Motivated by our experimental findings, we propose MonoTM, an interpretable topic modeling framework that achieves competitive documenttopic mixture estimation while producing validated, human-readable topic descriptors. ¹

## 2 Background & Related Work

## 2.1 Mechanistic Interpretability and Sparse Autoencoders

Mechanistic interpretability seeks to explain neural networks by identifying internal components and how they compose into computations. A major obstacle is that neurons are often polysemantic, activating for multiple unrelated patterns (Olah et al., 2020). One influential account explains this via superposition, where models represent more features than they have dimensions by encoding features in an overcomplete set of directions, enabled by sparsity in the underlying factors (Elhage et al., 2022). This motivates moving from neurons to learned feature directions.

Two scalable strategies support this shift. First, Bills et al. (2023) employ language models to produce and score natural-language explanations of internal directions based on activation patterns. Second, sparse dictionary learning methods, especially sparse autoencoders (SAEs), learn sparse, overcomplete feature bases that often yield more monosemantic units than neurons and can support meaningful interventions (Bricken et al., 2023; Huben et al., 2024). Recent work also applies SAEs to dense text embeddings, recovering interpretable semantic factors useful for downstream control and analysis (O'Neill et al., 2024).

Building on these ideas, we investigate SAEs over document embeddings as a source of reusable semantic units for topic modeling, and adopt an Interpreter-Predictor style protocol to obtain validated feature labels for topic description.

## 2.2 Topic Modeling

Overview of topic models. Topic modeling began with probabilistic bag-of-words (BoW) generative models such as LDA (Blei et al., 2003), which represents each document as a mixture of topics and each topic as a distribution over words. Although LDA offers clear probabilistic semantics and interpretable topic descriptors, its BoW assumption ignores context and can produce frequency- and surface-form-sensitive topics. Neural topic models use inference networks for amortized variational inference instead of iterative perdocument inference (ProdLDA; (Srivastava and Sutton, 2017)). Other neural parameterizations of document-topic mixtures include softmax and stick-breaking constructions (Miao et al., 2017). In parallel, embedding-aware models such as ETM represent topics in a word-embedding space to improve robustness to large vocabularies while maintaining interpretability (Dieng et al., 2020).

More recent approaches incorporate pretrained transformers to inject contextual semantics. A representative direction is contextualized topic modeling. CombinedTM uses contextual embeddings alongside BoW-style signals to improve topic quality (Bianchi et al., 2021a), and ZeroShotTM extends this idea with multilingual encoders to enable cross-lingual, zero-shot topic inference (Bianchi et al., 2021b). Beyond contextualization, recent work focuses on robustness and efficiency. ECRTM introduces embedding clustering regularization to discourage topic collapse and encourage more distinct topics (Wu et al., 2023). FASTopic further leverages pretrained transformers while modeling document-topic and topic-word relations in a unified framework designed to be fast, stable, and transferable (Wu et al., 2024b). Finally, S3 offers a decomposition-based alternative that treats topics as independent semantic axes in embedding space discovered via ICA rather than clustering or neural decoders (Kardos et al., 2025).

SAE-based topic modeling. Recent work connects sparse autoencoders and topic modeling by leveraging mechanistic-interpretability insights and treating SAE latents as reusable semantic units in embedding/activation space. Girrbach and Akata (2025) view SAEs as continuous-space topic models by deriving the SAE objective as a MAP estimator under an LDA-like generative model. Zheng et al. (2025) propose Mechanistic Topic Models (MTMs), which featurize documents using a pretrained SAE over LLM activations and fit topic models over SAE feature counts. While their mLDA is closest to our LDA-with-bag-of-features approach, MTMs rely on pretrained SAEs and external feature descriptions.

Unlike these prior works, our goal is not simply to replace words with SAE latents as the vocabulary for topic modeling. We show that the features most useful for document-topic mixture estimation are not necessarily the same features whose labels can be reliably interpreted. This empirical mismatch motivates MonoTM, which decouples the statistical and interpretive roles of SAE features. Although this design is more costly than reusing pretrained SAE vocabularies, it targets settings such as social science and digital humanities, where corpus-specific interpretability, traceability, and validation are often more important than fast topic discovery.

## 3 Research Questions

Our starting hypothesis is that SAE features can play two distinct roles in topic modeling. If a document embedding can be represented as a sparse, nonnegative linear combination of SAE features, and if those features are closer to monosemantic than the original embedding dimensions, then SAE features may serve both as human-readable semantic units and as useful units for estimating document-topic mixtures. These two roles are related but not identical: features that are easy to label may not be the same features that best support mixture estimation. We therefore first study these roles separately, and then ask how they can be combined in a single interpretable topic model:

1. Under what SAE configurations do corpustrained features become reliably interpretable semantic units?

2. Under what SAE configurations do SAE features provide effective units for document-topic mixture estimation?

3. How can we use SAE features to build a topic model that produces interpretable semantic topic descriptors aligned with strong document-topic mixture estimates?

## 4 Experimental Setup

Datasets. To evaluate topic models with minimal subjectivity, we compare inferred topics against the gold category annotations provided by standard benchmarks (dataset statistics in Table 3). We use (1) 20 Newsgroups (user-generated Usenet newsgroup posts; scikit-learn version) (Lang, 1995), (2) Web of Science (WOS-46985) using abstracts as input (Kowsari et al., 2017), and (3) Reuters (Reuters-21578) with the ModApte split (Lewis, 1997) accessed via Hugging Face Datasets (Lhoest et al., 2021). Reuters is a multi-label and highly imbalanced collection of Reuters newswire stories. For Reuters, we keep labels appearing at least 30 times, yielding 47 topics.

Document embeddings and SAE training. We map each document d to a dense embedding $\begin{array} { r l r } { x _ { d } } & { { } \in } & { \mathbb { R } ^ { 4 0 9 6 } } \end{array}$ using NVIDIA's 1lama-embed-nemotron-8b model (Babakhin et al., 2025). We use the model's full context window of 32,768 tokens (NVIDIA, 2025); longer documents are truncated to their first 32,768 tokens. This affects only 18 documents in 20 Newsgroups.

For each corpus, we train top-K sparse autoencoders on dimension-wise standardized document embeddings. Each SAE learns a dictionary with mN latent features, where $m = 4 0 9 6$ is the embedding dimension and N is the expansion factor. Thus, N controls the size of the learned feature dictionary, while K controls the maximum number of active features per document. Given a document embedding, the encoder produces nonnegative latent activations and retains only the largest K, yielding a sparse document-feature representation. The decoder reconstructs the standardized embedding from these active features.

The two SAE hyperparameters therefore have distinct roles: N controls dictionary capacity, while K controls per-document activity. We sweep 11 values of N from $1 / 6 4$ to 5 and eight values of K from 4 to 512, omitting configurations with m $N <$ K. Full architectural, optimization, checkpointselection, and grid details are given in Appendix B.

Interpreter: feature label generation. Following and adapting the autointerpretability procedure of O'Neill et al. (2024), we pass the learned SAE features (decoder columns) to an autointerpretability module that proposes and later validates naturallanguage labels.

For each feature $f ~ \in ~ \{ 1 , \ldots , m N \}$ , we construct an Interpreter prompt using three sets of examples: Max-activating examples, the 10 documents with the highest activations $h _ { d , f } ;$ Typicalactivating examples, 10 documents sampled from the middle quantiles of the nonzero activation distribution for $f ;$ and Zero-activating examples, 10 documents where feature $f$ is inactive $( h _ { d , f } = 0 )$ sampled as negatives.

Given these examples, an Interpreter LLM outputs a short description $\ell _ { f } ~ ( { \tt e . g . } , 4 { - } 1 0 ~ \mathrm { w o r d s } )$ intended to capture the single most prominent concept that is present in the activating texts but absent from the non-activating texts (see the Interpreter prompt in Figure 2).

Predictor: feature label validation. Labels produced by an Interpreter LLM can be plausible but unvalidated. For example, labels may describe a frequent corpus theme that is not specific to the feature. To test whether a label actually predicts feature behavior, we validate each $\ell _ { f }$ using a separate Predictor LLM (Bills et al., 2023; O'Neill

et al., 2024).

For each feature $f ,$ we construct a balanced evaluation set $S _ { f } = S _ { f } ^ { + } \cup S _ { f } ^ { - }$ by sampling 30 positive documents and 30 negative documents from the corpus. Positives satisfy $h _ { d , f } > 0$ (feature f is active for document d), while negatives satisfy $h _ { d , f } = 0$ (feature f is inactive). Let $y _ { d , f } = \mathbf { 1 } [ h _ { d , f } > 0 ]$ denote the ground-truth activation indicator.

We run the Predictor as an independent perdocument binary classification query. For each $d \in S _ { f }$ , the Predictor LLM receives only the description $\ell _ { f }$ and the raw document text, and outputs a binary prediction $\hat { y } _ { d , f } \in \{ 0 , 1 \}$ indicating whether feature f would activate (see the Predictor prompt in Figure 2). We aggregate these documentlevel predictions into a feature-level interpretability score using the F1 score between $\{ \hat { y } _ { d , f } \} _ { d \in S _ { f } }$ and $\{ y _ { d , f } \} _ { d \in { \cal S } _ { f } }$ , and denote this score by IS(f).

Autointerpretability sweep and LLM setup. Running the Interpreter-Predictor protocol requires many LLM inference calls, so we apply it only to a subset of SAE configurations we expect to be most informative due to cost constraints: $N \in$ $\{ 0 . 5 , 1 , 2 , 3 , 4 , 5 \}$ and $K \in \{ 4 , 8 , 1 6 , 3 2 \}$ . Details of the LLMs used are in Appendix D.

## 5 Results

## 5.1 RQ1: When Do SAE Features Become Interpretable Semantic Units?

RQ1 asks whether corpus-trained SAE features can serve as reliable semantic units, and how this depends on SAE capacity and sparsity. We operationalize interpretability using the Interpreter— Predictor protocol described in Section 4: a feature is treated as interpretable when its natural-language label predicts held-out feature activations with an interpretability score above a validation threshold.

Empirical results. SAE features become interpretable semantic units most consistently under low-to-moderate per-document activity levels. Across all three datasets, configurations with smaller K have the highest fraction of activated features that pass validation (Tables 4, 5, and 6).

Increasing K changes this behavior. At permissive validation thresholds, larger K generally increases the absolute number of validated features, because more active features are exposed for labeling and validation. However, when averaged across the N values we evaluate, the fraction of activated features that are validated decreases monotonically with K on all datasets (Tables 4, 5, and 6).

The effect of dictionary capacity N is best understood conditional on the activity level K. On 20 Newsgroups and Reuters, increasing N at low K often reduces the number of activated features that surpass the activation threshold, while Web of Science is an exception: activated and validated counts increase with N even at low K (Tables 4, 5, and 6). This suggests that the value of increasing dictionary capacity depends on whether the corpus and activity level provide enough support for the larger feature space. At higher K, larger dictionaries are more often useful, as allowing documents to activate more features enables higher-capacity SAEs to expose a larger pool of validated semantic units.

Additionally, we report a secondary diagnostic analysis of feature granularity in Appendix F. This analysis suggests that increasing capacity and activity may shift the validated feature space toward finer semantic resolution.

Choosing (N, K) in practice. These results suggest that SAE configuration should be treated as a practical modeling choice. If the user wants a conservative, easy-to-audit feature vocabulary, the best starting point is a low- or moderate-activity SAE, such as $K = 4 \mathrm { o r } K = 8 .$ , combined with a moderately large dictionary. This setting yields fewer features, but a larger share of them can be validated. If the user instead wants broader coverage or more fine-grained descriptors, it is reasonable to move toward larger K and larger N, but only with the expectation that post-hoc validation will discard a larger fraction of the activated feature space.

A useful practical procedure is therefore to begin with a high-precision anchor configuration and then expand only if the validated feature inventory is too coarse or too small. For corpora with many stable domain-specific distinctions, such as Web of Science, a moderate configuration such as $K = 8$ with a larger dictionary is a natural starting point. For noisier or more heterogeneous corpora such as 20 Newsgroups and Reuters, low-K configurations provide a cleaner initial semantic inventory.

## 5.2 RQ2: When Do SAE Features Support Document-Topic Mixture Estimation?

Parallel to RQ1's focus on interpretability, RQ2 asks whether corpus-trained SAE features can serve as useful statistical units for estimating document-topic mixtures.

We test this hypothesis by replacing LDA's word vocabulary with SAE features. For each document, we treat active SAE features as pseudo-tokens and their activation magnitudes as token weights, producing a bag-of-features (BoF) representation. We deliberately choose LDA because it provides a transparent mixed-membership likelihood once document embeddings are converted into bags of SAE features. Neural topic models could also be applied to the same BoF inputs, but their neural parameterization would introduce additional representational capacity, making it harder to isolate whether the SAE feature vocabulary itself provides useful statistical units for topic inference.

RQ2 evaluates whether SAE features are useful for mixture estimation, regardless of their interpretability scores.

BoF+LDA setup. For each dataset, we train a top-K SAE and construct a sparse documentfeature matrix $X ~ \in ~ \mathbb { R } _ { \ge 0 } ^ { D \times m N }$ where $X _ { d , j }$ is the activation magnitude of feature $j$ in document $d .$ We then fit LDA using scikit-learn's LatentDirichletAllocation with variational EM for 200 iterations, and set the number of topics to the number of gold categories (20 Newsgroups: T=20, Web of Science: T=7, Reuters: T=47).

Baselines. We compare against (i) classical BoW-LDA (Blei et al., 2003) and (ii) other state-of-theart neural topic models that output document-topic mixtures: CombinedTM (Bianchi et al., 2021a), ZeroShotTM (Bianchi et al., 2021b), ECRTM (Wu et al., 2023), FASTopic (Wu et al., 2024b), and $S ^ { 3 }$ (Kardos et al., 2025). We do not include BERTopic because, following the taxonomy of Wu et al. (2024a), it is a clustering-based topic discovery method rather than a model for document– topic mixture estimation. Including it in the topic– label alignment benchmark would therefore conflate mixture-estimation quality with embeddingcluster separability.

For each baseline, we apply the same topiclabel alignment procedure as for BoW+LDA. To ensure a fair comparison, all baselines that use document embeddings—CombinedTM, ZeroShotTM, FASTopic, and $S ^ { 3 } .$ —are run with the same 11ama-embed-nemotron-8b document embeddings used by MonoTM.

Evaluation. We do not use standard word-based topic coherence or topic diversity as evaluation metrics because doing so would defeat the purpose of MonoTM. Metrics such as PMI, NPMI, $C _ { V }$ , and topic diversity were designed for topic representations expressed as ranked lists of keywords: they measure whether top words co-occur in a reference or training corpus, or whether top-word lists are lexically redundant across topics (Newman et al., 2009; Mimno et al., 2011; Lau et al., 2014; Röder et al., 2015; Dieng et al., 2020). These metrics would require projecting our descriptors back into keywords that represent each topic, thereby reintroducing the very representation that our method is designed to replace.

We instead evaluate document-topic mixture quality by aligning inferred topics with gold benchmark labels and reporting Micro-F1 and Macro-F1. Full evaluation details, including the single-label and multi-label alignment procedures, are given in Appendix G. For all models and datasets, we run three trials with different random seeds and report mean scores in Table 1; full SAE-grid heatmaps are reported in Appendix H.

## 5.2.1 Effective mixture estimation requires balanced SAE scaling

BoF+LDA performance depends systematically on the SAE hyperparameters. The full Micro-F1 and Macro-F1 heatmaps are reported in Appendix H. Across datasets, mixture quality is weakest in two regimes: very wide dictionaries paired with very small K, and boundary cases where m $N = K$ These results suggest that SAE features are most useful for topic inference when the representation maintains an explicit sparsity bottleneck while still allowing enough active features per document.

The best-performing configurations instead lie in a balanced scaling band where dictionary capacity and per-document activity increase together. We therefore summarize BoF+LDA in Table 1 by averaging over a fixed robust subset of this region, using $N \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 \}$ and $K \in \{ 1 2 8 , 2 5 6 , 5 1 2 \}$ This gives a less brittle comparison than selecting the single best hyperparameter setting for each dataset.

## 5.2.2 Mixture quality and interpretability do not fully coincide

The balanced-scaling pattern identified above should not be read as an interpretability result. Although Figure 4 identifies a robust region for document-topic mixture estimation, the same region is not necessarily the one that maximizes the number of interpretable SAE features. This can be seen by comparing the mixture-quality heatmaps in Figure 4 with the interpretability counts in Tables 4, 5, and 6. For example, in 20 Newsgroups, holding $N = 0 . 5$ fixed and increasing K from 8 to 32 reduces the fraction of dictionary features with $\mathrm { I S } \geq 0 . 8 0$ from 721 $/ ( 4 0 9 6 \cdot 0 . 5 ) = 3 5 . 2 \%$ to $2 8 2 / ( 4 0 9 6 \cdot 0 . 5 ) = 1 3 . 8 \%$ , while Micro/Macro F1 increases from 0.56/0.52 to 0.66/0.63. Thus, neither the absolute number nor the dictionarylevel fraction of validated features fully predicts document-topic mixture quality. These results suggest a tension between choosing SAE configurations for feature-level interpretability and choosing them for document-topic mixture quality.

<table><tr><td rowspan="2">Model</td><td colspan="2">20 Newsgroups</td><td colspan="2">Web of Science</td><td colspan="2">Reuters</td></tr><tr><td>Micro F1</td><td>Macro F1</td><td>Micro F1</td><td>Macro F1</td><td>Micro F1</td><td>Macro F1</td></tr><tr><td>BoW-LDA</td><td>0.3673</td><td>0.3515</td><td>0.5147</td><td>0.4702</td><td>0.3446</td><td>0.2021</td></tr><tr><td>CombinedTM</td><td>0.4929</td><td>0.4513</td><td>0.5514</td><td>0.5201</td><td>0.2398</td><td>0.1728</td></tr><tr><td>ZeroShotTM</td><td>0.3704</td><td>0.3398</td><td>0.4904</td><td>0.4670</td><td>0.2051</td><td>0.1467</td></tr><tr><td>ECRTM</td><td>0.5774</td><td>0.5368</td><td>0.5176</td><td>0.4916</td><td>0.2410</td><td>0.1557</td></tr><tr><td>FASTopic</td><td>0.5492</td><td>0.4912</td><td>0.5032</td><td>0.4583</td><td>0.2535</td><td>0.1582</td></tr><tr><td> ${ \mathrm { S } } ^ { 3 }$ </td><td>0.4578</td><td>0.4220</td><td>0.4233</td><td>0.3942</td><td>0.1513</td><td>0.1100</td></tr><tr><td>BoF+LDA (Ours)</td><td>0.6770</td><td>0.6489</td><td>0.5753</td><td>0.5446</td><td>0.3188</td><td>0.2089</td></tr></table>

Table 1: Micro F1 and Macro F1 scores on 20 Newsgroups, Web of Science, and Reuters datasets. The best scores are in bold.

We test this tension more directly by asking whether restricting BoF+LDA to highly interpretable SAE features improves document-topic mixture estimation. Specifically, we filter features according to their interpretability score, retain only those with $\operatorname { I S } ( f ) \geq \tau$ , and then re-fit LDA on the resulting filtered BoF representation. Figures 6–8 show that this filtering generally hurts document– topic mixture quality, and that the degradation becomes more severe as τ increases. These results indicate that failing strict interpretability validation does not imply that an SAE feature is uninformative for mixture estimation.

## 5.2.3 Bag-of-features yields strong document-topic mixture estimates

Table 1 shows that BoF+LDA is competitive across all three benchmarks, achieving the strongest results on both single-label datasets and the best Macro-F1 on Reuters. These gains support the core intuition that SAE latents provide a sparse, semantically structured pseudo-token vocabulary, allowing LDA to pool co-occurrence evidence more effectively than when using raw words.

Results on Reuters are nuanced. Since it is multilabel and highly imbalanced, Micro-F1 is dominated by frequent classes while Macro-F1 is sensitive to performance on rare labels. Here, BoW-LDA attains the highest Micro-F1, but BoF+LDA achieves the best Macro-F1 overall. This pattern is consistent with BoF features providing more discriminative evidence for minority topics that have limited word overlap with majority classes.

## 5.3 RQ3: Interpretable Topic Modeling (MonoTM)

RQ3 asks how the two roles of SAE features identified above can be combined into a single interpretable topic model. The findings from RQ1 and RQ2 suggest that a single SAE representation should not be forced to serve both feature-level interpretability and document-topic mixture estimation. MonoTM implements this idea by separating document-topic mixture estimation from topic interpretation.

The remainder of this section presents MonoTM in three steps. Section 5.3.1 describes the MonoTM algorithm. Section 5.3.2 audits whether the validated descriptor vocabulary omits important topic semantics from lower-validation features. Section 5.3.3 gives a compact descriptor comparison against word-based and free-form LLM descriptor baselines.

## 5.3.1 The MonoTM Algorithm

MonoTM separates the statistical and interpretive roles of SAE features. Let D be the number of documents, T the number of topics, and V the number of validated interpretable features used for topic description. MonoTM estimates document-topic mixtures from a full SAE bag-of-features representation, then estimates a topic-feature distribution over validated interpretable features with the mixtures held fixed.

Stage 1: estimating document-topic mixtures from all SAE features. For each document, the mixture SAE produces a sparse nonnegative activation vector over all SAE features. We construct a full document-feature matrix $X ^ { \mathrm { m i x } }$ from these activations and fit LDA with $T$ topics, yielding document-topic mixtures

$$
\begin{array} { r l r } { \Theta \in \mathbb { R } _ { \geq 0 } ^ { D \times T } , } & { { } } & { \theta _ { d } \in \Delta ^ { T } . } \end{array}
$$

This stage uses all active SAE features, rather than only validated interpretable features, because Section 5.2.2 shows that filtering to interpretable features degrades mixture quality.

Stage 2: constructing an interpretable document-feature matrix. For topic interpretation, we use a validated feature set $\mathcal { F } _ { \tau }$ . A feature is retained if it activates in at least 30 documents, has a nonempty label, and satisfies $I S ( f ) \ \ge \ \tau$ , with $\tau \ : = \ : 0 . 8$ in our experiments. For each retained feature $f ,$ we construct an interpretable document-feature matrix $C \in \mathbb { R } _ { \geq 0 } ^ { D \times V }$ by weighting its activation by its interpretability score:

$$
c _ { d , f } = \operatorname* { m a x } ( \tilde { h } _ { d , f } , 0 ) \cdot I S ( f ) , \qquad f \in \mathcal { F } _ { \tau } .
$$

Stage 3: estimating topic-feature distributions with fixed mixtures. Given fixed documenttopic mixtures Θ and interpretable feature matrix C, MonoTM estimates a topic-feature distribution

$$
B \in \mathbb { R } _ { \geq 0 } ^ { T \times V } , \qquad \beta _ { t } \in \Delta ^ { V } ,
$$

by maximizing the fixed-mixture likelihood

$$
\mathcal { L } ( B ) = \sum _ { d = 1 } ^ { D } \sum _ { f = 1 } ^ { V } c _ { d , f } \log \left( \sum _ { t = 1 } ^ { T } \theta _ { d , t } \beta _ { t , f } \right) .
$$

We optimize this objective with a Dirichletsmoothed EM procedure; full update equations and implementation details are given in Appendix K.

After estimating B, each topic t is represented by the highest-probability validated feature labels under $\beta _ { t }$ . Appendix J provides sample MonoTM topic-feature representations produced by this final descriptor layer.

## 5.3.2 Audit: do excluded features hide important topic semantics?

To test whether the validated descriptor vocabulary omits important topic semantics, we audit the excluded features most likely to matter for topic interpretation. For all three datasets, we estimate Θ from the full bag-of-features representation of an SAE with $N = 0 . 5$ and $K = 2 5 6$ . For interpretation, we use the $N = 2 , K = 1 6 { \mathrm { S A E } }$ . The final MonoTM descriptor vocabulary contains active, validated features with $\operatorname { I S } ( f ) \geq . 8$ . For the audit, we estimate an auxiliary all-feature topic-feature distribution $B ^ { \mathrm { a l l } }$ over all active interpretation-SAE features, using raw feature activations instead of weighting by $\operatorname { I S } ( f )$ . This avoids mechanically suppressing the lower-validation features that the audit is designed to inspect.

We then rank all active interpretation features by their mass under $B ^ { \mathrm { a l l } }$ for each topic, and focus on lower-validation features with $\mathrm { I S } ( f ) < . 8$ that would have ranked among the top 20 features for at least one topic. These features are the strongest possible challenge to the validated descriptor vocabulary: they are highly topic-associated, but absent from the final MonoTM representation. For each such feature, assigned to the topic where it obtains its best rank, we search for a validated same-topic neighbor among the top-50 validated features of that topic under $B ^ { \mathrm { a l l } }$ . Appendix L gives the full construction, row alignment, EM objective, filtering rules, and formal definitions.

Coverage is measured in document-activation space. For each excluded feature, we compute cosine similarity between its raw document-level interpretation-SAE activation vector and the corresponding vectors for validated same-topic descriptors. We compare the nearest validated same-topic descriptor to a random validated descriptor sampled from the same topic-specific candidate pool. Figure 1 shows that nearest validated descriptors are much closer than random validated descriptors, especially for 20 Newsgroups and Web of Science.

The Reuters results are less straightforward, so we further characterize the excluded highassociation features using heuristic label-type flags based on keywords in the generated feature labels. (See Appendix L for the keyword rules.) These flags are used to summarize broad tendencies in the excluded feature set, not as a validation metric.

Table 2 shows a clear dataset difference. In 20 Newsgroups and Web of Science, excluded highassociation features are mostly semantic or broaddiscourse labels. In Reuters, by contrast, 70.7% are flagged as format/register/numeric cues, suggesting that many of these features capture financial-news register, reporting format, or numeric conventions rather than missing core topic semantics. Concrete examples of both patterns are provided in Appendix L.

![](images/1ff4a896960c86e7ac76d1066c8b6dcee053966dbe627d4d3d3639d32bd76cc2.jpg)

Figure 1: Coverage of lower-validation high-association features by validated same-topic descriptors. Bars show the median cosine similarity between feature activation profiles across documents.
<table><tr><td>Dataset</td><td>Semantic</td><td>Broad discourse</td><td>Format/register/ numeric</td></tr><tr><td>20NG</td><td>58.1%</td><td>36.4%</td><td>5.4%</td></tr><tr><td>WoS</td><td>88.5%</td><td>7.7%</td><td>3.8%</td></tr><tr><td>Reuters</td><td>6.3%</td><td>23.1%</td><td>70.7%</td></tr></table>

Table 2: Heuristic label-type split for lower-validation high-association features. Percentages are computed within each dataset.

Overall, while this audit does not prove that no excluded feature ever captures a meaningful omitted semantic dimension, it alleviates the concern that MonoTM arbitrarily omits a large class of independent topic-defining features.

## 5.3.3 Descriptor Comparison

Finally, we compare descriptor mechanisms under the same fixed document-topic mixtures Θ. This comparison is intended to isolate the descriptor layer by asking how different mechanisms describe the same inferred topics. We compare MonoTM against three descriptor baselines: θ-weighted TF-IDF terms, BERTopic-style c-TF-IDF terms (Grootendorst, 2022), and free-form LLM-generated descriptors. We include the LLM baseline because recent work increasingly treats LLMs as topic extractors (Lam et al., 2024; Mu et al., 2024; Pham et al., 2024). Full descriptor-generation details, prompts, and topic-name compression settings are provided in Appendix M. Full topic-name comparison tables are provided in Appendix O, and representative descriptor lists before topic-name compression are provided in Appendix P.

The comparison suggests two qualitative patterns. First, word-based descriptors often recover salient lexical anchors but remain surface-level. For example, in 20 Newsgroups, for the topic aligned with talk.politics.misc, the word-based baselines produce the topic name “Gun rights," whereas MonoTM yields “American culture wars," reflecting a broader mixture of gun-rights, sexuality, morality, and civil-liberties discussions (Table 12). Second, free-form LLM descriptors can be fluent but overly local to the sampled topic-associated documents. In Reuters, for example, the LLM baseline names the topic aligned with crude as “Ecuador oil crisis," whereas MonoTM yields the broader “OPEC oil market" (Table 14). These examples suggest that MonoTM descriptors offer a useful intermediate level of semantic abstraction: they bridge high-level topic labels and raw documents more effectively than word lists, while remaining more directly tied to the representations used for topic inference than ad hoc LLM descriptors.

Beyond topic naming, MonoTM also supports downstream analyses of relations among topics. Appendix Q illustrates this use case by comparing topic relations in the validated-feature space.

## 6 Conclusion

We introduced MonoTM, an interpretable topic modeling framework built on sparse autoencoders trained over document embeddings. Across three research questions, our results show that SAE features are useful for topic modeling in multiple but distinct ways. First, corpus-trained SAEs can produce relatively monosemantic units, but the availability of reliably interpretable features depends strongly on the choice of SAE configuration. Second, SAE activations can also serve as effective statistical units for estimating document-topic mixtures. However, the SAE configurations suited to these two roles do not necessarily coincide. Addressing RQ3, MonoTM resolves this mismatch by estimating document-topic mixtures from the full SAE representation and then learning topic-feature distributions over validated features with the mixtures fixed. Our analyses further show that excluded high-association features are typically covered by validated same-topic descriptors or reflect auxiliary non-core cues.

## 7 Limitations

Remaining uncertainty about excluded features. Section 5.3.2 audits features from the interpretation SAE that fall below the validation threshold and finds that genuinely hard-to-validate features account for only a small share of high-association topic-feature mass. However, this audit does not prove that all excluded features are unimportant or fully understood. Some low-validation features may encode corpus-specific regularities, pragmatic cues, formatting patterns, or culturally specific concepts that are difficult to summarize with short semantic labels. Moreover, our validation scores depend on the Interpreter and Predictor LLMs, so systematic blind spots in those models could still affect which features are included in the final descriptor set. For high-impact applications, MonoTM should therefore be paired with additional coverage audits and, when appropriate, human expert review.

Sensitivity to the Interpreter and Predictor LLMs. Our interpretability scores depend on the capabilities and failure modes of the specific LLMs used as Interpreter and Predictor, as well as prompt details and decoding settings. Different LLM backends may vary in (i) their ability to abstract from examples into stable hypotheses, (ii) calibration on the binary prediction task, and (iii) robustness to domain-specific language. As a result, both the number of features that validate above a threshold and the apparent granularity of validated features could change with the choice of model(s). While we partially mitigate this by separating generation (Interpreter) from validation (Predictor), our validation remains an LLM-mediated measurement rather than a ground-truth guarantee of monosemanticity.

Cost and practicality of LLM-heavy autointerpretability. MonoTM's feature labeling relies on many LLM calls, which is computationally expensive and can be slow in practice. This cost is a real barrier for iterative modeling workflows. That said MonoTM is aimed at settings where researchers are willing to spend more compute to obtain a single, carefully audited interpretive lens on an important corpus (e.g., social science and digital humanities analyses where interpretability and traceability are first-order goals). Still, improving efficiency is crucial. Promising directions include cheaper feature screening models for “likely interpretable" features and distilling the Predictor into a small classifier. Additionally, continuing progress in SAE feature interpretation and tooling (e.g., Lieberum et al., 2024) suggests that MonoTM could become cheaper to apply while preserving its core advantages.

Systematic biases can distort the interpretable topic representation. MonoTM estimates document-topic mixtures using the full SAE vocabulary but represents topics using only the subset of features that pass LLM-based validation. If the Interpreter or Predictor has systematic blind spots—for example, consistently underinterpreting certain registers, dialects, domains, or culturally specific concepts—then those features may be disproportionately excluded from the interpretable feature set. In that case, the topic descriptors produced by MonoTM could omit important aspects of a topic even if those aspects strongly influence the inferred mixtures. This limitation is especially salient if the LLMs' biases correlate with sensitive attributes or with particular styles of expression in the corpus. Mitigations include auditing interpretability coverage across document subpopulations, validating with multiple LLM backends, and incorporating human expert review for high-impact analyses.

## References

Yauhen Babakhin, Radek Osmulski, Ronay Ak, Gabriel Moreira, Mengyao Xu, Benedikt Schifferer, Bo Liu, and Even Oldridge. 2025. Llama-embednemotron-8b: A universal text embedding model for multilingual and cross-lingual tasks. Preprint, arXiv:2511.07025.

Jana Bernhard-Harrer, Randa Ashour, Jakob-Moritz Eberl, Petro Tolochko, and Hajo Boomgaarden. 2025. Beyond standardization: a comprehensive review of topic modeling validation methods for computational social science research. Political Science Research and Methods, pages 1–19.

Federico Bianchi, Silvia Terragni, and Dirk Hovy. 2021a. Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 759–766, Online. Association for Computational Linguistics.

Federico Bianchi, Silvia Terragni, Dirk Hovy, Debora Nozza, and Elisabetta Fersini. 2021b. Cross-lingual Contextualized Topic Models with Zero-shot Learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 1676–1683, Online. Association for Computational Linguistics.

Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. 2023. Language models can explain neurons in language models.

David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent Dirichlet Allocation. Journal of Machine Learning Research, 3:993–1022.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, and 5 others. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread. Https://transformercircuits.pub/2023/monosemanticfeatures/index.html.

Jonathan Chang, Sean Gerrish, Chong Wang, Jordan Boyd-graber, and David Blei. 2009. Reading Tea Leaves: How Humans Interpret Topic Models. In Advances in Neural Information Processing Systems, volume 22. Curran Associates, Inc.

Nan Z. Da. 2019. The Computational Case against Computational Literary Studies. Critical Inquiry, 45(3):601–639. Publisher: The University of Chicago Press.

Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. 2020. Topic Modeling in Embedding Spaces. Transactions of the Association for Computational Linguistics, 8:439–453. Place: Cambridge, MA Publisher: MIT Press.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. 2022. Toy models of superposition. Transformer Circuits Thread.

Leander Girrbach and Zeynep Akata. 2025. Sparse Autoencoders are Topic Models. arXiv preprint. ArXiv:2511.16309 [cs].

Justin Grimmer and Brandon M. Stewart. 2013. Text as Data: The Promise and Pitfalls of Automatic Content Analysis Methods for Political Texts. Political Analysis, 21(3):267–297.

Maarten Grootendorst. 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint. ArXiv:2203.05794 [cs].

Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. 2024. Sparse Autoencoders Find Highly Interpretable Features in Language Models.

Márton Kardos, Jan Kostkan, Kenneth Enevoldsen, Arnault-Quentin Vermillet, Kristoffer Nielbo, and Roberta Rocca. 2025. S3 – Semantic Signal Separation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 633–666, Vienna, Austria. Association for Computational Linguistics.

Kamran Kowsari, Donald E Brown, Mojtaba Heidarysafa, Kiana Jafari Meimandi, Matthew S Gerber, and Laura E Barnes. 2017. Hdltex: Hierarchical deep learning for text classification. In Machine Learning and Applications (ICMLA), 2017 16th IEEE International Conference on. IEEE.

Michelle S. Lam, Janice Teoh, James A. Landay, Jeffrey Heer, and Michael S. Bernstein. 2024. Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, CHI '24, pages 1–28, New York, NY, USA. Association for Computing Machinery.

Ken Lang. 1995. Newsweeder: Learning to filter netnews. In Proceedings of the Twelfth International Conference on Machine Learning.

Jey Han Lau, David Newman, and Timothy Baldwin. 2014. Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality. In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics, pages 530–539, Gothenburg, Sweden. Association for Computational Linguistics.

David D. Lewis. 1997. Reuters-21578 text categorization test collection. Distribution 1.0.

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, and 13 others. 2021. Datasets: A Community Library for Natural Language Processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, Janos Kramar, Anca Dragan, Rohin Shah, and Neel Nanda. 2024. Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 278–300, Miami, Florida, US. Association for Computational Linguistics.

Qiaozhu Mei, Xuehua Shen, and ChengXiang Zhai. 2007. Automatic labeling of multinomial topic models. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and

data mining, pages 490–499, San Jose California USA. ACM.

Yishu Miao, Edward Grefenstette, and Phil Blunsom. 2017. Discovering Discrete Latent Topics with Neural Variational Inference. In Proceedings of the 34th International Conference on Machine Learning, pages 2410–2419. PMLR. ISSN: 2640-3498.

David Mimno, Hanna Wallach, Edmund Talley, Miriam Leenders, and Andrew McCallum. 2011. Optimizing Semantic Coherence in Topic Models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, pages 262–272, Edinburgh, Scotland, UK. Association for Computational Linguistics.

Yida Mu, Chun Dong, Kalina Bontcheva, and Xingyi Song. 2024. Large Language Models Offer an Alternative to the Traditional Approach of Topic Modelling. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 10160–10171, Torino, Italia. ELRA and ICCL.

David Newman, Sarvnaz Karimi, and Lawrence Cavedon. 2009. External evaluation of topic models. In Proceedings of the 14th Australasian Document Computing Symposium, pages 1–8. University of Sydney.

NVIDIA. 2025. nvidia/llama-embed-nemotron-8b (hugging face model card/readme). https://huggingface.co/nvidia/ 1lama-embed-nemotron-8b. Accessed 2025- 12-31.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. 2020. Zoom in: An introduction to circuits. Distill. Https://distill.pub/2020/circuits/zoom-in.

Charles O'Neill, Christine Ye, Kartheik G. Iyer, and John F. Wu. 2024. Towards Interpretable Scientific Foundation Models: Sparse Autoencoders for Disentangling Dense Embeddings of Scientific Concepts.

Chau Minh Pham, Alexander Hoyle, Simeng Sun, Philip Resnik, and Mohit Iyyer. 2024. TopicGPT: A Promptbased Topic Modeling Framework. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2956–2984, Mexico City, Mexico. Association for Computational Linguistics.

Michael Röder, Andreas Both, and Alexander Hinneburg. 2015. Exploring the Space of Topic Coherence Measures. In Proceedings of the Eighth ACM International Conference on Web Search and Data Mining, WSDM '15, pages 399–408, New York, NY, USA. Association for Computing Machinery.

Anna Shadrova. 2021. Topic models do not model topics: epistemological remarks and steps towards best practices. Journal of Data Mining & Digital Humanities, 2021. Publisher: Episciences.org.

Akash Srivastava and Charles Sutton. 2017. Autoencoding Variational Inference For Topic Models.

Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, and Anh Tuan Luu. 2023. Effective Neural Topic Modeling with Embedding Clustering Regularization. In Proceedings of the 40th International Conference on Machine Learning, pages 37335–37357. PMLR. ISSN: 2640-3498.

Xiaobao Wu, Thong Nguyen, and Anh Tuan Luu. 2024a. A survey on neural topic models: methods, applications, and challenges. Artificial Intelligence Review, 57(2):18.

Xiaobao Wu, Thong Nguyen, Delvin Ce Zhang, William Yang Wang, and Anh Tuan Luu. 2024b. FASTopic: pretrained transformer is a fast, adaptive, stable, and transferable topic model. In Proceedings of the 38th International Conference on Neural Information Processing Systems, volume 37 of NIPS ’24, pages 84447–84481, Red Hook, NY, USA. Curran Associates Inc.

Carolina Zheng, Nicolas Beltran-Velez, Sweta Karlekar, Claudia Shi, Achille Nazaret, Asif Mallik, Amir Feder, and David M. Blei. 2025. Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders. arXiv preprint. ArXiv:2507.23220 [cs].

## A Dataset Details

See Table 3 for summary statistics of the datasets.

<table><tr><td>Dataset</td><td>Points</td><td>Cats</td><td>Multi</td><td>No Label</td><td>Mean Chars</td><td>SD Chars</td><td>Median</td><td>Min</td><td>Max</td></tr><tr><td>20 Newsgroups</td><td>18846</td><td>20</td><td>X</td><td>0</td><td>1902.53</td><td>3984.97</td><td>1175</td><td>115</td><td>160616</td></tr><tr><td>Web of Science</td><td>46985</td><td>7</td><td>X</td><td>0</td><td>1376.46</td><td>492.77</td><td>1355</td><td>95</td><td>8019</td></tr><tr><td>Reuters</td><td>12561</td><td>47</td><td>0</td><td>2304</td><td>843.7</td><td>874.97</td><td>546</td><td>56</td><td>13396</td></tr></table>

Table 3: Summary statistics of the datasets used. The abbreviations in the header are defined as follows: Points (Total number of data points), Cats (Number of categories), and Multi (Multi-label classification, where $\because \mathrm { o } ^ { \mathrm { , } }$ denotes multi-label and $\because \mathbf { X } '$ denotes single-label). No Label indicates the number of samples without a label. The character count statistics (Mean, SD, Median, Min, Max) describe the distribution of document lengths.

## B SAE Architecture and Training Details

For each document $d \in \mathcal { D } = \{ 1 , \dots , M \}$ , let $\boldsymbol { x } _ { d } \in \mathbb { R } ^ { m }$ denote its document embedding and let $\tilde { x } _ { d } \in \mathbb { R } ^ { m }$ denote the dimension-wise standardized embedding, using the corpus-level mean and standard deviation. In our experiments, $m = 4 0 9 6$

We use a single-layer top-K sparse autoencoder with hidden dimensionality $m N .$ , where N is the expansion factor. Depending on N, the dictionary may be undercomplete, complete, or overcomplete relative to the embedding dimension. The decoder is a matrix $\boldsymbol { D } \in \mathbb { R } ^ { m \times ( m N ) }$ , and the encoder uses tied weights $D ^ { \top }$

$$
z = D ^ { \top } \tilde { x } + b ,\tag{1}
$$

$$
a = { \mathrm { R e L U } } ( z ) .\tag{2}
$$

Sparsity is enforced with a hard top-K operator,

$$
h = \mathrm { T o p K } ( a , K ) ,\tag{3}
$$

which yields a nonnegative latent activation vector $h \in \mathbb { R } _ { > 0 } ^ { m N }$ with at most K active coordinates.

We additionally introduce a learnable per-feature gain vector $g \in \mathbb { R } _ { > 0 } ^ { m N }$ to modulate feature magnitudes independently of decoder direction norms. We parameterize g in log-space to enforce positivity and define

$$
\tilde { h } = h \odot g .\tag{4}
$$

The reconstruction of the standardized embedding is

$$
\hat { x } = D \tilde { h } .\tag{5}
$$

After each update, decoder columns are normalized to unit norm to avoid arbitrary rescaling between decoder weights and latent activations.

We train each SAE with Adam using learning rate $1 0 ^ { - 4 }$ and batch size 1024 for up to 300k steps. We select the checkpoint with the lowest normalized reconstruction error on a fixed monitoring split containing 10% of documents. The training objective is

$$
\mathcal { L } = \frac { \| \tilde { x } - \hat { x } \| _ { 2 } ^ { 2 } } { \| \tilde { x } - \bar { x } \| _ { 2 } ^ { 2 } + \varepsilon } ,\tag{6}
$$

where x is the batch mean.

We train a grid of top-K SAEs over $( N , K )$

$$
\begin{array} { r l } & { N \in \{ 0 . 0 1 5 6 2 5 , 0 . 0 3 1 2 5 , 0 . 0 6 2 5 , 0 . 1 2 5 , } \\ & { ~ 0 . 2 5 , 0 . 5 , 1 , 2 , 3 , 4 , 5 \} , } \\ & { K \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \} . } \end{array}
$$

This gives up to $1 1 \times 8 = 8 8$ configurations. We omit configurations with $m N < K$ , since they cannot activate K distinct features. When $m N = K$ the top-K operator retains all units, so the model reduces to a dense tied-weight autoencoder; we keep this boundary case as a diagnostic comparison.

## C Prompts for the Interpreter-Predictor Module

See Figure 2 for the Interpreter prompt and the Predictor prompt.

![](images/3304de0f1f2fb07e7d898e4ec6c2e131cde0512afc0c6e7376eaaf2875319f28.jpg)  
Figure 2: The prompts used for the Interpreter-Predictor Module. The Interpreter generates a semantic description of the vector, which is then fed into the Predictor to validate predictive accuracy.  
E Counts of Interpretable Features Across Interpretability Score Thresholds

## D LLM Models and Inference Settings

For Interpreter and Predictor LLM runs, we use open-weight, instruction-tuned Gemma 3 models: Gemma-3-27B-IT (FP8) as the Interpreter and Gemma-3-12B-IT (bfloat16) as the Predictor.2 Both models are accessed via the DeepInfra API.3 We use the smaller model for the Predictor because the task is a comparatively simple binary decision. For both Interpreter and Predictor calls, we set the maximum context length to 131,072 tokens.

See Table 4, Table 5 and Table 6.

<table><tr><td>N</td><td>K</td><td>IS≥ 0.95</td><td>IS≥ 0.90</td><td>IS≥ 0.85</td><td>IS≥ 0.80</td></tr><tr><td>0.5</td><td>4</td><td>73</td><td>239</td><td>374</td><td>453</td></tr><tr><td>1</td><td>4</td><td>89</td><td>211</td><td>339</td><td>423</td></tr><tr><td>2</td><td>4</td><td>103</td><td>217</td><td>312</td><td>372</td></tr><tr><td>3</td><td>4</td><td>87</td><td>193</td><td>280</td><td>328</td></tr><tr><td>4</td><td>4</td><td>83</td><td>184</td><td>261</td><td>302</td></tr><tr><td>5</td><td>4</td><td>95</td><td>199</td><td>272</td><td>307</td></tr><tr><td>0.5</td><td>8</td><td>53</td><td>237</td><td>483</td><td>721</td></tr><tr><td>1</td><td>8</td><td>67</td><td>269</td><td>502</td><td>689</td></tr><tr><td>2</td><td>8</td><td>62</td><td>237</td><td>413</td><td>547</td></tr><tr><td>3</td><td>8</td><td>46</td><td>184</td><td>320</td><td>434</td></tr><tr><td>4</td><td>8</td><td>61</td><td>159</td><td>274</td><td>357</td></tr><tr><td>5</td><td>8</td><td>49</td><td>142</td><td>250</td><td>322</td></tr><tr><td>0.5</td><td>16</td><td>7</td><td>92</td><td>292</td><td>577</td></tr><tr><td>1</td><td>16</td><td>22</td><td>161</td><td>481</td><td>923</td></tr><tr><td>2</td><td>16</td><td>22</td><td>218</td><td>564</td><td>1019</td></tr><tr><td>3</td><td>16</td><td>24</td><td>175</td><td>443</td><td>721</td></tr><tr><td>4</td><td>16</td><td>28</td><td>163</td><td>326</td><td>476</td></tr><tr><td>5</td><td>16</td><td>49</td><td>194</td><td>365</td><td>512</td></tr><tr><td>0.5</td><td>32</td><td>1</td><td>22</td><td>84</td><td>282</td></tr><tr><td>1</td><td>32</td><td>5</td><td>45</td><td>207</td><td>604</td></tr><tr><td>2</td><td>32</td><td>9</td><td>83</td><td>375</td><td>993</td></tr><tr><td>3</td><td>32</td><td>33</td><td>278</td><td>851</td><td>1759</td></tr><tr><td>4</td><td>32</td><td>73</td><td>462</td><td>1081</td><td>1795</td></tr><tr><td>5</td><td>32</td><td>91</td><td>540</td><td>1232</td><td>1923</td></tr></table>

Table 4: Number of features above interpretability score thresholds in the 20 Newsgroups dataset. IS denotes the interpretability score.

<table><tr><td>N</td><td>K</td><td>IS≥ 0.95</td><td>IS≥ 0.90</td><td>IS≥ 0.85</td><td>IS≥ 0.80</td></tr><tr><td>0.5</td><td>4</td><td>224</td><td>470</td><td>587</td><td>686</td></tr><tr><td>1</td><td>4</td><td>276</td><td>568</td><td>741</td><td>858</td></tr><tr><td>2</td><td>4</td><td>383</td><td>759</td><td>958</td><td>1063</td></tr><tr><td>3</td><td>4</td><td>438</td><td>832</td><td>1036</td><td>1169</td></tr><tr><td>4</td><td>4</td><td>467</td><td>861</td><td>1086</td><td>1188</td></tr><tr><td>5</td><td>4</td><td>509</td><td>936</td><td>1121</td><td>1254</td></tr><tr><td>0.5</td><td>8</td><td>247</td><td>648</td><td>934</td><td>1122</td></tr><tr><td>1</td><td>8</td><td>379</td><td>924</td><td>1294</td><td>1555</td></tr><tr><td>2</td><td>8</td><td>594</td><td>1231</td><td>1674</td><td>1983</td></tr><tr><td>3</td><td>8</td><td>655</td><td>1359</td><td>1816</td><td>2130</td></tr><tr><td>4</td><td>8</td><td>709</td><td>1443</td><td>1910</td><td>2195</td></tr><tr><td>5</td><td>8</td><td>657</td><td>1439</td><td>1924</td><td>2204</td></tr><tr><td>0.5</td><td>16</td><td>105</td><td>466</td><td>863</td><td>1222</td></tr><tr><td>1</td><td>16</td><td>343</td><td>1037</td><td>1733</td><td>2331</td></tr><tr><td>2</td><td>16</td><td>574</td><td>1734</td><td>2573</td><td>3274</td></tr><tr><td>3</td><td>16</td><td>704</td><td>1900</td><td>2806</td><td>3536</td></tr><tr><td>4</td><td>16</td><td>687</td><td>1899</td><td>2835</td><td>3567</td></tr><tr><td>5</td><td>16</td><td>664</td><td>1826</td><td>2855</td><td>3551</td></tr><tr><td>0.5</td><td>32</td><td>21</td><td>178</td><td>422</td><td>773</td></tr><tr><td>1 32</td><td></td><td>92</td><td>543</td><td>1199</td><td>1924</td></tr><tr><td>2</td><td>32</td><td>293</td><td>1301</td><td>2532</td><td>3857</td></tr><tr><td>3</td><td>32</td><td>403</td><td>1683</td><td>3169</td><td>4656</td></tr><tr><td>4 32</td><td></td><td>443</td><td>1808</td><td>3435</td><td>5120</td></tr><tr><td>5 32</td><td></td><td>376</td><td>1724</td><td>3360</td><td>5082</td></tr></table>

Table 5: Number of features above interpretability score thresholds in the Web of Science dataset. IS denotes the interpretability score.

<table><tr><td>N</td><td>K</td><td>IS≥ 0.95</td><td>IS≥ 0.90</td><td>IS≥ 0.85</td><td>IS≥ 0.80</td></tr><tr><td>0.5</td><td>4</td><td>59</td><td>152</td><td>213</td><td>277</td></tr><tr><td>1</td><td>4</td><td>50</td><td>120</td><td>187</td><td>226</td></tr><tr><td>2</td><td>4</td><td>58</td><td>122</td><td>180</td><td>210</td></tr><tr><td>3</td><td>4</td><td>42</td><td>108</td><td>169</td><td>204</td></tr><tr><td>4</td><td>4</td><td>46</td><td>116</td><td>163</td><td>196</td></tr><tr><td>5</td><td>4</td><td>47</td><td>108</td><td>158</td><td>190</td></tr><tr><td>0.5</td><td>8</td><td>39</td><td>123</td><td>263</td><td>406</td></tr><tr><td>1</td><td>8</td><td>43</td><td>152</td><td>270</td><td>372</td></tr><tr><td>2</td><td>8</td><td>38</td><td>128</td><td>202</td><td>279</td></tr><tr><td>3</td><td>8</td><td>42</td><td>105</td><td>184</td><td>261</td></tr><tr><td>4</td><td>8</td><td>32</td><td>93</td><td>166</td><td>236</td></tr><tr><td>5</td><td>8</td><td>31</td><td>78</td><td>143</td><td>192</td></tr><tr><td>0.5</td><td>16</td><td>21</td><td>102</td><td>238</td><td>433</td></tr><tr><td>1</td><td>16</td><td>24</td><td>125</td><td>266</td><td>467</td></tr><tr><td>2</td><td>16</td><td>21</td><td>106</td><td>206</td><td>347</td></tr><tr><td>3</td><td>16</td><td>31</td><td>102</td><td>193</td><td>284</td></tr><tr><td>4</td><td>16</td><td>42</td><td>146</td><td>251</td><td>358</td></tr><tr><td>5</td><td>16</td><td>55</td><td>168</td><td>282</td><td>400</td></tr><tr><td>0.5</td><td>32</td><td>3</td><td>30</td><td>121</td><td>285</td></tr><tr><td>1</td><td>32</td><td>8</td><td>66</td><td>220</td><td>515</td></tr><tr><td>2</td><td>32</td><td>33</td><td>177</td><td>446</td><td>756</td></tr><tr><td>3</td><td>32</td><td>57</td><td>269</td><td>570</td><td>881</td></tr><tr><td>4</td><td>32</td><td>58</td><td>272</td><td>584</td><td>983</td></tr><tr><td>5 32</td><td></td><td>55</td><td>289</td><td>654</td><td>1090</td></tr></table>

Table 6: Number of features above interpretability score thresholds in the Reuters dataset. IS denotes the interpretability score.

## F Feature Granularity Across SAE Configurations

## F.1 Motivation

In the main text, RQ1 focuses on whether corpustrained SAE features can be reliably interpreted as semantic units. As a secondary diagnostic, we also ask whether SAE configurations differ in the granularity of the validated semantic features they recover. If larger or more active SAEs refine features learned by smaller SAEs, then document-level activations from the larger SAE should more easily reconstruct the validated-feature activations of the smaller SAE than vice versa.

This analysis is not required for MonoTM, but it helps characterize how SAE configuration affects the structure of the validated feature space.

## F.2 Probe setup

Consider two SAEs trained on the same corpus but with different configurations $( N , K )$ , denoted A and B. Let ${ \mathcal { F } } ^ { ( A ) }$ and $\mathcal { F } ^ { ( B ) }$ be the subsets of latent features that pass our label validation threshold. We use $\mathrm { I S } \geq 0 . 8 0$ throughout. Let $p = | \mathcal { F } ^ { ( A ) } |$ and $q =$ $| \mathcal F ^ { ( B ) } |$ , and let M be the number of documents.

## F.3 Document-feature activation matrices

For each document $d ,$ each SAE produces a sparse top-K activation list $\{ ( i , a _ { d i } ) \} _ { i \in \mathrm { T o p K } ( d ) }$ with $a _ { d i } \geq 0$ . We construct a dense document– feature activation vector by accumulating these latent activations into the coordinates corresponding to validated features. For SAE A, we define $x _ { d } ^ { ( \bar { A } ) } \in \mathbb { R } _ { \geq 0 } ^ { p }$ by

$$
x _ { d j } ^ { ( A ) } = \sum _ { i \in \mathrm { T o p K } ^ { \left( A \right) } \left( d \right) } a _ { d i } ^ { ( A ) } \mathbf { 1 } \left[ i = f _ { j } ^ { ( A ) } \right] ,\tag{7}
$$

where $( f _ { 1 } ^ { ( A ) } , \dots , f _ { p } ^ { ( A ) } )$ is a fixed ordering of ${ \mathcal { F } } ^ { ( A ) }$ Equivalently, we keep only validated feature coordinates and set all others to zero. Stacking over documents yields $X ^ { ( A ) } \in \mathbb { R } _ { \geq 0 } ^ { M \times p }$ . We analogously construct $X ^ { ( B ) } \in \mathbb { R } _ { > 0 } ^ { M \times q }$

## F.4 Linear mapping probe

We fit an affine linear map in each direction:

$$
\widehat { X } ^ { ( B ) } = X ^ { ( A ) } W _ { A  B } ^ { \top } + \mathbf { 1 } b _ { A  B } ^ { \top } ,\tag{8}
$$

$$
\widehat { X } ^ { ( A ) } = X ^ { ( B ) } W _ { B  A } ^ { \top } + \mathbf { 1 } b _ { B  A } ^ { \top } ,\tag{9}
$$

where $W _ { A  B } \in \mathbb { R } ^ { q \times p }$ and $b _ { A \to B } \in \mathbb { R } ^ { q }$ , and symmetrically for $B  A$ . Parameters are estimated on a training split using ridge regression.⁴

## F.5 Error metric and directional asymmetry

Because document-feature activations are nonnegative and sparse, we evaluate reconstruction with a weighted MSE that assigns higher weights to nonzero target entries:

$$
\begin{array} { r } { \mathrm { W M S E } ( \widehat { Y } , Y ) = \frac { \sum _ { d , j } w _ { d j } ( \widehat { y } _ { d j } - y _ { d j } ) ^ { 2 } } { \sum _ { d , j } w _ { d j } } , } \\ { w _ { d j } = 1 + \left( \lambda - 1 \right) \mathbf { 1 } [ y _ { d j } > 0 ] , } \end{array}\tag{10}
$$

with $\lambda = 1 0$ in our experiments.

To make scores comparable across different target spaces, we normalize by the WMSE of a zeropredictor baseline computed on the same validation set:

$$
\rho ( A \to B ) = \frac { \mathrm { W M S E } ( \widehat { X } ^ { ( B ) } , X ^ { ( B ) } ) } { \mathrm { W M S E } ( 0 , X ^ { ( B ) } ) } .\tag{11}
$$

Lower $\rho$ indicates more accurate reconstruction relative to predicting all-zero activations.

Our main statistic is the directional asymmetry

$$
\Delta = \rho ( \mathrm { b i g } \to \mathrm { s m a l l } ) - \rho ( \mathrm { s m a l l } \to \mathrm { b i g } ) ,\tag{12}
$$

where “big" and “small” refer to the ordered SAE configurations used in this probe. Negative values indicate that the larger SAE more easily reconstructs the validated-feature activation patterns of the smaller SAE than the reverse direction.

## F.6 Resampling protocol

For each dataset, we repeat the full procedure over 10 random train/validation splits of documents. For each split, we fit both directions, $A \  \ B$ and $B  A$ , and compute $\rho$ on the held-out validation documents. We evaluate all ordered pairs among the four SAE configurations shown in Figure 3.

## F.7 Results

Figure 3 shows directional asymmetries in cross-SAE predictability. On Web of Science and Reuters, mappings from higher-capacity configurations to lower-capacity ones consistently achieve lower normalized error than the reverse direction across essentially all configuration pairs. Aggregating across all pairs and seeds, the median $\Delta$ is negative on both datasets (Table 7), indicating that larger SAEs more easily reconstruct smaller SAEs than vice versa under this linear probe.

![](images/df0ee8d2df751095d0707d7f6b7eecb1063c2b9390e852e09bdef7bb50b20564.jpg)  
Figure 3: Heatmaps of linear cross-SAE reconstruction. Each cell shows the mean validation MSE ratio $\rho$ for predicting the target SAE activations from the source SAE activations, with direction shown as row → column. Parentheses give the standard deviation over 10 train/validation splits. Lower is better.

<table><tr><td>Dataset</td><td>Median ∆</td><td>95% boot. CI</td><td> $\mathrm { P r } ( \Delta < 0 )$ </td></tr><tr><td>20 Ng</td><td>-0.111</td><td>[−0.223, 0.052]</td><td>0.667</td></tr><tr><td>WoS</td><td>-0.099</td><td>[−0.164, −0.048]</td><td>1.000</td></tr><tr><td>Reuters</td><td>-0.102</td><td>[−0.139, -0.057]</td><td>1.000</td></tr></table>

Table 7: Summary of cross-SAE predictability asymmetry across datasets. Reported are the median ∆ and a 95% hierarchical bootstrap confidence interval over configuration pairs and random splits, along with $\mathrm { P r } ( \Delta < 0 )$

For 20 Newsgroups, the aggregate median $\Delta$ is also negative but less conclusive under resampling. The asymmetry is strongest when comparing the most separated configurations, $( N { = } 2 , K { = } 4 )$ and (N=5, K=32), consistent with a larger gap in representational resolution.

## F.8 Interpretation and limitations

Taken together, these results are consistent with a granularity shift as SAE capacity and activity increase. Features learned in larger SAEs appear to contain sufficient information to linearly reconstruct the validated-feature activations of smaller SAEs, while the reverse reconstruction is systematically harder.

This probe does not establish one-to-one feature correspondences, and it may underestimate nonlinear relationships between feature spaces. It should therefore be interpreted as a diagnostic of representational resolution rather than as direct evidence that individual features split cleanly across SAE configurations.

## G Document-Topic Mixture Evaluation Details

We evaluate document-topic mixture quality through alignment with gold label topics. This evaluation is intended to measure whether the inferred topic mixtures recover the benchmark category structure, rather than whether topic descriptors form coherent word lists.

For 20 Newsgroups and Web of Science, which are single-label datasets, we assign each document to its highest-probability inferred topic,

$$
\hat { z } _ { d } = \arg \operatorname* { m a x } _ { k } \theta _ { d , k } .
$$

We then compute the confusion matrix between inferred topics and gold labels and use the Hungarian algorithm to find the one-to-one topic-label matching that maximizes total agreement. After applying this mapping, we treat the aligned topic assignments as multi-class predictions and report Micro-F1 and Macro-F1.

Reuters is multi-label and highly imbalanced, so we evaluate it as a multi-label topic-label alignment problem. For each candidate threshold

$$
t \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \} ,
$$

we predict topic k for document d when $\theta _ { d , k } > t$ We then compute pairwise F1 scores between each gold label topic and each inferred topic, use the Hungarian algorithm to obtain a one-to-one label– topic matching, and compute Micro-F1 and Macro-F1 over the matched pairs. We apply the same threshold grid to all models and report each metric at its best threshold over this grid. Documents with no Reuters gold label are retained in the evaluation, so a model can avoid false positives on these documents only by assigning all topic probabilities below the selected threshold.

## H Full SAE Hyperparameter Heatmaps for Mixture Estimation

Figure 4 reports Micro-F1 and Macro-F1 over the full SAE hyperparameter grid for BoF+LDA on each dataset. Figure 5 provides a complementary aggregate view by averaging performance across datasets after scaling each dataset's F1 scores to the [0, 1] range. Together, these figures support the hyperparameter-selection discussion in Section 5.2.1 by showing how document-topic mixture quality varies with dictionary capacity N and perdocument activity K.

Across datasets, two patterns are visible. First, very wide SAEs paired with very small K tend to perform poorly. In this regime, the dictionary has high capacity, but each document is allowed to activate only a small number of features. The resulting BoF representation is therefore too constrained: it exposes many possible pseudo-token types, but provides too little per-document evidence for stable topic inference. Second, boundary cases where mN = K are also weak. Since the top-K operator retains all available units in this case, the model no longer imposes a meaningful sparsity bottleneck, which appears to reduce the usefulness of SAE latents as topic-modeling signals.

By contrast, stronger performance appears in a balanced scaling band where N and K increase together. These configurations preserve an effective sparsity constraint while allowing each document to express a richer set of SAE features. This pattern is visible in the per-dataset heatmaps in Figure 4 and remains apparent in the dataset-aggregated heatmaps in Figure 5. The purple region in Figure 4 marks the robust subset of this balanced regime used to summarize BoF+LDA in Table 1. We use this region rather than a single best cell to avoid overemphasizing dataset- or seed-specific hyperparameter choices.

![](images/488bf9980a1fa283baee4a17b889b8f1b53a2ccfb470c05dacc57fafd537ad8d.jpg)  
Figure 4: Heatmaps of Micro-F1 and Macro-F1 scores over the full SAE hyperparameter grid for BoF+LDA. The purple box indicates the (N, K) region whose values are averaged to produce the scores reported in Table 1.

![](images/79d39f4e48a7b46967163fa7fa57f444b4e1e6bc70e9f9c5635d36cbc7c1626a.jpg)

![](images/001554334b6e6dedf02c8a70cc8d744c7d01f251d94bd286e346193602d20674.jpg)  
Figure 5: Heatmaps of Micro-F1 (top) and Macro-F1 (bottom) scores, shown as mean (standard deviation) over three datasets, computed from 9 data points for each (N, K). F1 scores were scaled to the [0, 1] range on a per-dataset basis to prevent any single dataset from dominating performance.

I Document-Topic Estimation Using Only Interpretable Features

See Figure 6, Figure 7, and Figure 8.

<table><tr><td colspan="29" rowspan="10">Micro F1                                        Macro F15.0Iututtto .5 4.0-3.0N2.01.00.55.0                                                  5.04.0</td></tr><tr><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.60</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">0.58</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.37</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.52</td><td colspan="10" rowspan="1">0.55</td></tr><tr><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.64</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.58</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.60</td><td colspan="1" rowspan="1">0.55</td><td colspan="10" rowspan="1">0.55</td></tr><tr><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.61</td><td colspan="1" rowspan="1">0.62</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.37</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">0.57</td><td colspan="10" rowspan="1">0.59</td></tr><tr><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.61</td><td colspan="1" rowspan="1">0.60</td><td colspan="1" rowspan="1">0.62</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">0.57</td><td colspan="10" rowspan="1">0.59</td></tr><tr><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.58</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.63</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.52</td><td colspan="10" rowspan="1">0.60</td></tr><tr><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.61</td><td colspan="1" rowspan="1">0.66</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">0.58</td><td colspan="10" rowspan="1">0.64</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16</td><td colspan="10" rowspan="1">32</td></tr><tr><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.53</td><td colspan="10" rowspan="1">0.56</td></tr><tr><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.64</td><td colspan="1" rowspan="1">0.60</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.60</td><td colspan="1" rowspan="1">0.56</td><td colspan="10" rowspan="1">0.51</td></tr><tr><td colspan="2" rowspan="1">3.0-N</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.61</td><td colspan="1" rowspan="1">0.60</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.37</td><td colspan="1" rowspan="1">0.58</td><td colspan="1" rowspan="1">0.56</td><td colspan="19" rowspan="1">0.55</td></tr><tr><td colspan="2" rowspan="14">Ituuto i 2.00.5Cutttoo0.7 4.03.0N2.01.0is0.55.0Itutto ·.8 4.0-3.0N</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">0.61</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.54</td><td colspan="19" rowspan="1">0.58</td></tr><tr><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">0.65</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.54</td><td colspan="18" rowspan="1">0.62</td></tr><tr><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.66</td><td colspan="1" rowspan="1">0.66</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">0.62</td><td colspan="19" rowspan="1">0.63</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="19" rowspan="1">32</td></tr><tr><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.52</td><td colspan="1" rowspan="1">5.0-</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.44</td><td colspan="18" rowspan="1">0.49</td></tr><tr><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.57</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">0.53</td><td colspan="19" rowspan="1">0.44</td></tr><tr><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.50</td><td colspan="19" rowspan="1">0.52</td></tr><tr><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.61</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.62</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.58</td><td colspan="1" rowspan="1">0.53</td><td colspan="19" rowspan="1">0.59</td></tr><tr><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.59</td><td colspan="1" rowspan="1">0.58</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.56</td><td colspan="19" rowspan="1">0.56</td></tr><tr><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">0.62</td><td colspan="1" rowspan="1">0.64</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.52</td><td colspan="1" rowspan="1">0.58</td><td colspan="19" rowspan="1">0.61</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="19" rowspan="1">32</td></tr><tr><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.37</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">0.43</td><td colspan="19" rowspan="1">0.36</td></tr><tr><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.35</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">0.42</td><td colspan="19" rowspan="1">0.33</td></tr><tr><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.35</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.41</td><td colspan="19" rowspan="1">0.44</td></tr><tr><td colspan="2" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.34</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">0.46</td><td colspan="19" rowspan="1">0.53</td></tr><tr><td colspan="2" rowspan="3">1.0-0.5</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">0.50</td><td colspan="19" rowspan="1">0.47</td></tr><tr><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.63</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.37</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.60</td><td colspan="19" rowspan="1">0.46</td></tr><tr><td colspan="9" rowspan="1">4        8       16      32               4         8K                                                      K</td><td colspan="18" rowspan="1">1632</td></tr><tr><td colspan="11" rowspan="10">Micro F1                                        Macro F15.0                                                   5.0Istutto0 .5 4.03.0N2.01.00.5                                                   0.55.0-4.0</td></tr><tr><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.48</td></tr><tr><td colspan="1" rowspan="1">0.33</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.52</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.32</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.49</td></tr><tr><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.28</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.43</td></tr><tr><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.42</td></tr><tr><td colspan="1" rowspan="1">0.32</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.52</td></tr><tr><td colspan="1" rowspan="1">0.32</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.48</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td></tr><tr><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.49</td></tr><tr><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.27</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.52</td></tr><tr><td colspan="2" rowspan="8">Ituttoo .6  3.0N2.01.0Cuttuto0.7 3.0N</td><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.28</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.52</td></tr><tr><td colspan="1" rowspan="1">0.33</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.52</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">2.0-</td><td colspan="1" rowspan="1">0.32</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.44</td></tr><tr><td colspan="1" rowspan="1">0.35</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.34</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.52</td></tr><tr><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.52</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td></tr><tr><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.32</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.46</td></tr><tr><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.53</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.37</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.51</td></tr><tr><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.46</td></tr><tr><td colspan="2" rowspan="4">2.0IsS0.5</td><td colspan="1" rowspan="1">0.34</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.52</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.33</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">0.45</td></tr><tr><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.56</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.34</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.53</td></tr><tr><td colspan="1" rowspan="1">0.28</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.55</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.26</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.52</td><td colspan="1" rowspan="1">0.51</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">16</td><td colspan="1" rowspan="1">32</td></tr><tr><td colspan="2" rowspan="2">5.04.0</td><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.48</td><td colspan="1" rowspan="1">5.0</td><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.45</td></tr><tr><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.46</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">4.0</td><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.36</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.47</td></tr><tr><td colspan="2" rowspan="1">3.0N</td><td colspan="1" rowspan="1">0.29</td><td colspan="1" rowspan="1">0.40</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">3.0</td><td colspan="1" rowspan="1">0.28</td><td colspan="1" rowspan="1">0.38</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.44</td></tr><tr><td colspan="2" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.33</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.45</td><td colspan="1" rowspan="1">0.50</td><td colspan="1" rowspan="1">2.0</td><td colspan="1" rowspan="1">0.32</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.47</td></tr><tr><td colspan="2" rowspan="3">1.0-0.5</td><td colspan="1" rowspan="1">0.33</td><td colspan="1" rowspan="1">0.42</td><td colspan="1" rowspan="1">0.49</td><td colspan="1" rowspan="1">0.44</td><td colspan="1" rowspan="1">1.0</td><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.39</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.41</td></tr><tr><td colspan="1" rowspan="1">0.31</td><td colspan="1" rowspan="1">0.43</td><td colspan="1" rowspan="1">0.51</td><td colspan="1" rowspan="1">0.54</td><td colspan="1" rowspan="1">0.5</td><td colspan="1" rowspan="1">0.30</td><td colspan="1" rowspan="1">0.41</td><td colspan="1" rowspan="1">0.47</td><td colspan="1" rowspan="1">0.50</td></tr><tr><td colspan="9" rowspan="1">4        8       16      32               4        8       16      32K                                                      K</td></tr><tr><td></td><td colspan="5">Micro F1</td><td colspan="14">Macro F1</td></tr><tr><td></td><td>5.0</td><td>0.19</td><td>0.29</td><td>0.26</td><td>0.25</td><td>5.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>Cututo.5</td><td>4.0</td><td>0.25</td><td>0.29</td><td>0.27</td><td>0.24</td><td>4.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>N</td><td>3.0</td><td>0.22</td><td>0.27</td><td>0.26</td><td>0.26</td><td>3.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>2.0</td><td>0.22</td><td>0.26</td><td>0.24</td><td>0.26</td><td>2.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>IsS</td><td>1.0</td><td>0.20</td><td>0.25</td><td>0.25</td><td>0.26</td><td>1.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>0.5</td><td>0.21</td><td>0.23</td><td>0.26</td><td>0.25</td><td>0.5</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>Ituuto 0.</td><td></td><td>4</td><td>8</td><td>16</td><td>32</td><td></td><td>4</td><td>8</td><td>16</td><td colspan="10">32</td></tr><tr><td></td><td>5.0</td><td>0.19</td><td>0.28</td><td>0.24</td><td>0.27</td><td>5.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>4.0</td><td>0.25</td><td>0.28</td><td>0.29</td><td>0.25</td><td>4.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>N</td><td>3.0</td><td>0.22</td><td>0.27</td><td>0.27</td><td>0.26</td><td>3.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>2.0-</td><td>0.22</td><td>0.26</td><td>0.27</td><td>0.27</td><td>2.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>1.0</td><td>0.20</td><td>0.24</td><td>0.24</td><td>0.25</td><td>1.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>0.5</td><td>0.19</td><td>0.23</td><td>0.27</td><td>0.26</td><td>0.5</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td rowspan="11">Cuttto0.7 N IsS</td><td></td><td>4</td><td>8</td><td>16</td><td>32</td><td></td><td>4 0.01</td><td>8</td><td>16</td><td colspan="10">32</td></tr><tr><td>5.0</td><td>0.20</td><td>0.24</td><td>0.29</td><td>0.29</td><td>5.0</td><td>0.01</td><td></td><td>0.01 0.01</td><td colspan="10">0.01</td></tr><tr><td>4.0</td><td>0.24</td><td>0.29</td><td>0.28</td><td>0.28</td><td>4.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>3.0</td><td>0.22</td><td>0.29</td><td>0.29</td><td>0.25</td><td>3.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>2.0</td><td>0.22</td><td>0.28</td><td>0.26</td><td>0.30</td><td>2.0</td><td></td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>1.0</td><td>0.21</td><td>0.23</td><td>0.28</td><td>0.29</td><td>1.0</td><td>0.01</td><td>0.01</td><td>0.01</td><td colspan="10">0.01</td></tr><tr><td>0.5</td><td>0.18</td><td>0.22</td><td>0.28</td><td>0.27</td><td>0.5</td><td>0.01</td><td></td><td>0.01 0.01</td><td colspan="10">0.01</td></tr><tr><td></td><td>4</td><td>8</td><td></td><td>16</td><td>32</td><td>4</td><td>0.01</td><td>8</td><td colspan="10">16 32</td></tr><tr><td rowspan="11">Ituto . N</td><td>5.0</td><td>0.20</td><td>0.26</td><td>0.32</td><td>0.31</td><td rowspan="5">5.0 3.0 2.0</td><td rowspan="5">0.01 0.01 0.01 0.01</td><td rowspan="5">0.01 0.01</td><td rowspan="5">0.01 0.01 0.01 0.01 0.01</td><td colspan="9" rowspan="5">0.01 0.01 0.01 0.01</td></tr><tr><td>4.0</td><td>0.26</td><td>0.36</td><td>0.23</td><td>0.32</td><td colspan="8">4.0</td></tr><tr><td>3.0-</td><td>0.24</td><td>0.26</td><td>0.31</td><td>0.24</td><td colspan="8">0.01</td></tr><tr><td>2.0</td><td>0.23</td><td>0.24</td><td>0.24</td><td>0.19</td><td colspan="9">0.01</td></tr><tr><td>1.0</td><td>0.19</td><td>0.22</td><td>0.25</td><td>0.33 1.0</td><td colspan="9">0.01 0.01 0.01</td></tr><tr><td>0.5</td><td>0.20</td><td>0.22</td><td>0.33</td><td>0.25</td><td colspan="14">0.5 0.01</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="3"></td><td rowspan="3">4</td><td rowspan="3"></td><td colspan="10" rowspan="3">0.01 0.01 16 32 K</td></tr><tr><td></td><td>4</td><td>8</td><td>16</td><td></td><td>32</td><td colspan="9">8</td></tr><tr><td></td><td></td><td></td><td>K</td><td></td><td></td><td colspan="9"></td></tr></table>

Figure 6: Heatmaps of micro-F1 and macro-F1 scores for document-topic mixtures on the 20 Newsgroups dataset, inferred using only features with interpretability scores (IS) above various cutoffs.

Figure 7: Heatmaps of micro-F1 and macro-F1 scores for document-topic mixtures on the Web of Science dataset, inferred using only features with interpretability scores (IS) above various cutoffs.

Figure 8: Heatmaps of micro-F1 and macro-F1 scores for document-topic mixtures on the Reuters dataset, inferred using only features with interpretability scores (IS) above various cutoffs. To evaluate against the multi-label gold labels, we treat topics whose inferred document-topic probabilities exceed 0.2 as predicted classes.

## J Sample Topic-Feature Representations

Table 8 presents a subset of MonoTM's topic representation output, reporting the top-5 labeled features per topic for three topics in each dataset.

We use an interpretable-feature SAE with $( N { = } 1 , K { = } 8 )$ and filter features by $\operatorname { I S } ( f ) \ge 0 . 8$ This is not the richest configuration we tested: larger settings, such as $( N { = } 5 , K { = } 3 2 )$ , yield substantially more interpretable features. However, the top feature labels for each topic show that even with interpretable features obtained from a relatively small SAE, the feature labels associated with high β values align well with the gold topic labels and provide a finer-grained understanding of the corpus than topic labels alone.

<table><tr><td>Dataset</td><td>Topic</td><td>Feature label</td><td>β</td></tr><tr><td>20 Newsgroups</td><td>rec.sport.hockey</td><td>NHL hockey game and player information</td><td>.402</td></tr><tr><td>20 Newsgroups</td><td>rec.sport.hockey</td><td>Sports scores reported on Úsenet newsgroups</td><td>.048</td></tr><tr><td>20 Newsgroups</td><td>rec.sport.hockey</td><td>Frustrated fan complaints about sports broadcasting decisions</td><td>.038</td></tr><tr><td>20 Newsgroups</td><td>rec.sport.hockey</td><td>Passionate sports discussion in online forums</td><td>.031</td></tr><tr><td>20 Newsgroups</td><td>rec.sport.hockey</td><td>Sports stats requests and early internet culture</td><td>.030</td></tr><tr><td>20 Newsgroups</td><td>soc.religion.christian</td><td>Intense Christian theological debate and interpretation</td><td>.348</td></tr><tr><td>20 Newsgroups</td><td>soc.religion.christian</td><td>Religious debate, interpretation, and personal belief defense</td><td>.061</td></tr><tr><td>20 Newsgroups</td><td>soc.religion.christian</td><td>Traditional Catholic theology and liturgical concerns</td><td>.041</td></tr><tr><td>20 Newsgroups</td><td>soc.religion.christian</td><td>Skepticism towards religious belief and dogma</td><td>.040</td></tr><tr><td>20 Newsgroups</td><td>soc.religion.christian</td><td>Christian theological discussion and apologetic reasoning</td><td>.030</td></tr><tr><td>20 Newsgroups</td><td>rec.motorcycles</td><td>Motorcycle riding advice and safety discussion</td><td>.481</td></tr><tr><td>20 Newsgroups</td><td>rec.motorcycles</td><td>Motorcycle countersteering technique and related debate</td><td>.035</td></tr><tr><td>20 Newsgroups</td><td>rec.motorcycles</td><td>Discussions of drunk driving and motorcycles</td><td>.033</td></tr><tr><td>20 Newsgroups</td><td>rec.motorcycles</td><td>Motorcycle culture, camaraderie, road-based social signaling</td><td>.032</td></tr><tr><td>20 Newsgroups</td><td>rec.motorcycles</td><td>Motorcycle enthusiast discussion: ownership, specs, and banter</td><td>.031</td></tr><tr><td>Web of Science</td><td>Medical</td><td>Cardiovascular disease research and clinical information</td><td>.016</td></tr><tr><td>Web of Science</td><td>Medical</td><td>Gastrointestinal function, microbiome, and symptomology</td><td>.016</td></tr><tr><td>Web of Science</td><td>Medical</td><td>Bladder dysfunction, urological assessment, and clinical research</td><td>.015</td></tr><tr><td>Web of Science Web of Science</td><td>Medical</td><td>Osteoporosis research, bone density, and skeletal health</td><td>.014</td></tr><tr><td></td><td>Medical</td><td>Neurobiological basis of psychiatric disease research</td><td>.013</td></tr><tr><td>Web of Science</td><td>Psychology</td><td>Internal psychological states and research analysis</td><td>.029</td></tr><tr><td>Web of Science</td><td>Psychology</td><td>Parenting challenges, family stress, child well-being</td><td>.025</td></tr><tr><td>Web of Science Web of Science</td><td>Psychology</td><td>Social judgment and character attribution processes</td><td>.020</td></tr><tr><td>Web of Science</td><td>Psychology</td><td>Psychiatric comorbidity, distress, and vulnerability factors</td><td>.019</td></tr><tr><td></td><td>Psychology</td><td>Autism spectrum disorder and related neurodevelopmental features</td><td>.017</td></tr><tr><td>Web of Science</td><td>CS</td><td>Advanced computer and network security threat analysis</td><td>.030</td></tr><tr><td>Web of Science Web of Science</td><td>CS CS</td><td>Computer vision algorithm development and evaluation Parallel computing, performance optimization, technical implementa-</td><td>.023 .022</td></tr><tr><td></td><td></td><td>tion details</td><td></td></tr><tr><td>Web of Science Web of Science</td><td>CS CS</td><td>Cryptographic security and data privacy techniques Machine learning methodology and technical rigor</td><td>.019</td></tr><tr><td></td><td></td><td></td><td>.016</td></tr><tr><td>Reuters Reuters</td><td>earn</td><td>Comparative financial performance reporting, earnings vs prior period</td><td>.516</td></tr><tr><td>Reuters</td><td>earn</td><td>Comparative financial performance,  $^ { * } \mathrm { \dot { x } }$  VS  $\check { \Upsilon } ^ { \dag }$  reporting</td><td>.175</td></tr><tr><td>Reuters</td><td>earn</td><td>Quarterly financial report key metrics comparison</td><td>.073</td></tr><tr><td>Reuters</td><td>earn</td><td>Financial turnaround, profit versus loss reporting</td><td>.063</td></tr><tr><td></td><td>earn</td><td>Corporate quarterly financial results reporting and metrics</td><td>.054</td></tr><tr><td>Reuters Reuters</td><td>acq</td><td>Corporate acquisition and ownership transfer announcements</td><td>.311</td></tr><tr><td>Reuters</td><td>acq</td><td>Corporate mergers and acquisitions announcements, financial details</td><td>.089</td></tr><tr><td>Reuters</td><td>acq</td><td>Mergers, acquisitions, and corporate restructuring events</td><td>.088</td></tr><tr><td>Reuters</td><td>acq</td><td>Corporate merger and acquisition announcements (LOI)</td><td>.081</td></tr><tr><td></td><td>acq</td><td>Energy sector financial deal reporting details</td><td>.051</td></tr><tr><td>Reuters</td><td>money-fx</td><td>Central bank money market interventions and rates</td><td>.366</td></tr><tr><td>Reuters</td><td>money-fx</td><td>Bank of England money market liquidity revisions</td><td>.227</td></tr><tr><td>Reuters</td><td>money-fx</td><td>Central bank money market intervention reporting</td><td>.097</td></tr><tr><td>Reuters</td><td>money-fx</td><td>Bank of England liquidity intervention reporting</td><td>.075</td></tr><tr><td>Reuters</td><td>money-fx</td><td>British monetary policy and currency market signals</td><td>.033</td></tr></table>

Table 8: Part of the final MonoTM output showing topic-feature distributions. The table lists the top five validated feature labels for three topics from each dataset. Gold labels from the original datasets are assigned using the alignment procedure in Section 5.2 and are shown only for orientation. $\beta$ denotes the probability of each feature within the topic-feature distribution.

## K Full MonoTM Algorithm and Inference Details

This appendix expands the MonoTM algorithm summarized in Section 5.3.1. Let $\mathcal { D } = \{ 1 , \ldots , D \}$ index documents and let $T$ be the number of topics. MonoTM estimates document-topic mixtures from a full SAE bag-of-features representation, then estimates a topic-feature distribution over validated interpretable SAE features with the mixtures held fixed.

We allow the SAE used for mixture estimation and the SAE used for interpretation to be different. We denote their activations by $\tilde { h } ^ { \mathrm { m i x } }$ and $\tilde { h } ^ { \mathrm { { i n t } } }$ , respectively. The single-SAE case is recovered by setting these two representations to be the same. For each document $d ,$ let $\theta _ { d } \in \Delta ^ { T }$ denote its document— topic mixture, and let $\boldsymbol { \Theta } \in \mathbb { R } _ { > 0 } ^ { D \times T }$ collect these mixtures row-wise. Let $\mathcal { F } _ { \tau } = \bar { \{ 1 , \ldots , V \} }$ denote the set of validated interpretable SAE features used for topic description. Each feature $f \in { \mathcal { F } } _ { \tau }$ has a natural-language label $\ell _ { f }$ and interpretability score $\operatorname { I S } ( f )$ . Our goal is to estimate a topic-feature matrix $\overset { \prime } { B } \in \mathbb { R } _ { > 0 } ^ { \overset { \smile } { T } \times V }$ whose row $\beta _ { t } \in \Delta ^ { V }$ ranks validated interpretable features for topic t.

Stage 1: estimating document-topic mixtures. For mixture estimation, MonoTM uses the full mixture-SAE representation rather than only validated features. For a top-K SAE with expansion factor $N _ { \ast }$ , each document embedding yields a sparse nonnegative activation vector $\tilde { h } _ { d } ^ { \operatorname* { m i x } } \in \mathbb { R } _ { \ge 0 } ^ { m N }$ with at most K nonzero entries. We construct a document-feature matrix $X ^ { \mathrm { m i x } } \in \mathbb { R } _ { \geq 0 } ^ { D \times m N }$ by setting

$$
X _ { d , j } ^ { \mathrm { m i x } } = \tilde { h } _ { d , j } ^ { \mathrm { m i x } } ,\tag{13}
$$

and fit LDA with $T$ topics on $X ^ { \mathrm { m i x } }$ to obtain $\Theta$ This stage uses all active SAE features because Section 5.2.2 shows that strict interpretability filtering generally hurts mixture estimation.

Stage 2: constructing the interpretable feature matrix. For topic description, MonoTM restricts to interpretation-SAE features that (i) activate in at least 30 documents, (ii) have a nonempty naturallanguage label, and (iii) satisfy $\operatorname { I S } ( f ) \geq \tau$ , with $\tau = 0 . 8$ in our experiments. The interpretable document-feature matrix $C \in \mathbb { R } _ { > 0 } ^ { D \times V }$ is defined by

$$
c _ { d , f } = \operatorname* { m a x } ( \tilde { h } _ { d , f } ^ { \operatorname* { i n t } } , 0 ) \cdot \mathrm { I S } ( f ) , \qquad f \in \mathscr { F } _ { \tau } .\tag{14}
$$

This weighting makes the final topic descriptors emphasize features with stronger validation evidence.

Stage 3: estimating topic-feature distributions. Given Θ and $C ,$ MonoTM estimates B by maximizing the fixed-Θ mixture log-likelihood

$$
\mathcal { L } ( B ) = \sum _ { d = 1 } ^ { D } \sum _ { f = 1 } ^ { V } c _ { d , f } \log \Big ( \sum _ { t = 1 } ^ { T } \theta _ { d , t } \beta _ { t , f } \Big ) ,\tag{15}
$$

subject to $\beta _ { t } \in \Delta ^ { V }$ for all t. After estimating $B _ { ; }$ topic t is represented by its ranked feature distribution $\beta _ { t }$ ; the reported descriptor list consists of the top-ranked labels $\ell _ { f }$ under $\beta _ { t , f }$ , with associated probabilities.

## K.1 Dirichlet-smoothed EM

We optimize Eq. (15) with an EM algorithm that treats latent topic assignments as missing data. Let

$$
q _ { d , f , t } = p ( z = t \mid d , f ) = \frac { \theta _ { d , t } \beta _ { t , f } } { \sum _ { t ^ { \prime } } \theta _ { d , t ^ { \prime } } \beta _ { t ^ { \prime } , f } } .\tag{16}
$$

The E-step computes expected topic-feature counts:

$$
N _ { t , f } = \sum _ { d = 1 } ^ { D } c _ { d , f } q _ { d , f , t } .\tag{17}
$$

The M-step updates $B$ with symmetric Dirichlet smoothing $\eta > 0$

$$
\beta _ { t , f } = \frac { N _ { t , f } + \eta } { \sum _ { f ^ { \prime } } ( N _ { t , f ^ { \prime } } + \eta ) } .\tag{18}
$$

We iterate until relative improvement in Eq. (15) falls below a tolerance. To avoid zero-probability features and stabilize topics when C is sparse, we use the data-dependent heuristic

$$
\eta = \frac { 1 } { T } + \frac { \bar { L } } { V } ,\tag{19}
$$

where $\bar { L }$ is the average number of nonzero validated features per document.

## K.2 Using Separate Mixture and Interpretation SAEs

MonoTM does not require a single SAE to simultaneously optimize mixture estimation and interpretability. Even when a single SAE is used endto-end, we do not interpret topics using the topic– feature distribution implicitly produced during Θ inference. Instead, topic interpretation is based on a separately estimated topic-feature distribution over validated interpretable features. When the SAE configuration that yields the most useful validated feature set is suboptimal for estimating Θ, a user can train a second SAE optimized for mixture quality while using an interpretability-optimized SAE to construct $C .$

## L Excluded-feature audit details

This appendix gives the full procedure for the excluded-feature audit in Section 5.3.2. The audit asks whether features excluded from the final MonoTM descriptor vocabulary nevertheless contain topic-defining semantic information that is absent from the validated descriptor set.

Configurations. For all datasets, the documenttopic mixture matrix Θ is the same one used in the main RQ3 analysis. It is estimated from the mixture-quality SAE with expansion factor $N = 0 . 5$ and top- $K = 2 5 6$ , using seed 123. The number of topics is fixed to the number of gold categories: $T = 2 0$ for 20 Newsgroups, $T = 7$ for Web of Science, and $T \ = \ 4 7$ for Reuters. The interpretation-side SAE used for the audit has N = 2 and $K = 1 6$

All-active interpretation feature matrix. Let $\mathcal { F } _ { \mathrm { a c t } }$ denote the set of interpretation-SAE features that activate in at least 30 documents, and let $\ell _ { f }$ be the generated label for feature $f .$ The validated descriptor set used by MonoTM is

$$
\mathcal { F } _ { \mathrm { v a l } } = \{ f \in \mathcal { F } _ { \mathrm { a c t } } : \mathrm { I S } ( f ) \geq . 8 \mathrm { a n d } \ell _ { f } \neq \emptyset \} .
$$

For the audit, we instead construct an all-active feature matrix $A \in \mathbb { R } _ { \geq 0 } ^ { D \times | \mathcal { F } _ { \mathrm { a c t } } | }$ with entries

$$
a _ { d , f } = \operatorname* { m a x } ( \tilde { h } _ { d , f } , 0 ) ,
$$

where $\tilde { h } _ { d , f }$ is the gain-adjusted top-K activation of the interpretation SAE. Unlike the final descriptor estimation step, the audit does not weight feature activations by $\operatorname { I S } ( f )$ . This avoids building the conclusion into the measurement: interpretabilityscore weighting would mechanically reduce the mass of lower-validation features. Document rows are aligned between Θ, the interpretation-SAE activations, and raw documents using saved row IDs.

All-feature audit distribution. Using fixed Θ and the raw all-active matrix A, we estimate an auxiliary topic-feature distribution $B ^ { \mathrm { a l l } }$ by maximizing

$$
\mathcal { L } ( B ^ { \mathrm { a l l } } ) = \sum _ { d = 1 } ^ { D } \sum _ { f \in \mathcal { F } _ { \mathrm { a c t } } } a _ { d , f } \log \left( \sum _ { t = 1 } ^ { T } \theta _ { d , t } \beta _ { t , f } ^ { \mathrm { a l l } } \right)
$$

subject to $\beta _ { t } ^ { \mathrm { a l l } } \in \Delta ^ { | \mathcal { F } _ { \mathrm { a c t } } | }$ for every topic t. We use the same Dirichlet-smoothed EM routine as in the fixed-Θ MonoTM descriptor step. This distribution is used only for auditing; the final MonoTM descriptors are still estimated over the validated feature vocabulary.

High-risk excluded set. For each topic t, all active features are ranked by $\beta _ { t , f } ^ { \mathrm { a l l } }$ . We define

$$
\mathrm { r a n k } _ { t } ^ { \mathrm { a l l } } ( f ) = 1 + \left| \{ f ^ { \prime } : \beta _ { t , f ^ { \prime } } ^ { \mathrm { a l l } } > \beta _ { t , f } ^ { \mathrm { a l l } } \} \right| .
$$

For each feature $f$ with $\operatorname { I S } ( f ) < . 8 .$ let $t _ { f } ^ { \star }$ be the topic where it has its best rank. The high-risk excluded set is

$$
\mathcal { R } _ { 2 0 } = \{ ( t _ { f } ^ { \star } , f ) : \mathrm { I S } ( f ) < . 8 , \mathrm { \ r a n k } _ { t _ { f } ^ { \star } } ^ { \mathrm { a l l } } ( f ) \leq 2 0 \} .
$$

This set contains lower-validation features that are sufficiently associated with at least one topic to be plausible missing descriptors.

Coverage test. For each $( t , f ) \in \mathcal { R } _ { 2 0 }$ , we compare $f$ with validated same-topic descriptors. The candidate pool is

$$
\mathcal { V } _ { t , M } = \{ g \in \mathcal { F } _ { \mathrm { v a l } } : \mathrm { r a n k } _ { t } ^ { \mathrm { a l l } } ( g ) \leq M \} .
$$

We report $M = 5 0$ in the main text and include M = 20 and M = 100 as sensitivity settings. The M = 20 setting asks whether the feature is covered by the most compact validated descriptor region; M = 50 asks whether it is covered by the broader high-ranked validated descriptor distribution; $M =$ 100 checks sensitivity to a wider candidate set.

Let $a _ { f }$ be the document-level activation vector for feature $f .$ We compute

$$
\cos _ { \mathrm { a c t } } ( f , g ) = { \frac { a _ { f } ^ { \top } a _ { g } } { \| a _ { f } \| _ { 2 } \| a _ { g } \| _ { 2 } } } .
$$

The nearest validated same-topic descriptor is

$$
g ^ { \star } ( f , t ) = \arg \operatorname* { m a x } _ { g \in \mathcal { V } _ { t , M } } \cos _ { \operatorname* { a c t } } ( f , g ) .
$$

For comparison, we sample random validated descriptors from the same candidate pool and compute the random-baseline activation cosine. Table 9 reports the aggregate results.

Heuristic label-type flags. We also assign a coarse heuristic label type to each excluded feature based only on keywords in its generated feature label. These flags are not manual annotations and are not used as a validation metric. They are included to characterize broad patterns, especially the difference between Reuters and the other datasets.

Labels are flagged as format/register/numeric if they include terms related to document form, boilerplate, reporting style, technical artifacts, numerical data, or financial-news announcements, including keywords such as: format, boilerplate, template, placeholder, metadata, header, posting, email, message, announcement, reporting, reports, quarterly, qtly, dividend, earnings, record date, subscription, unsubscribe, FAQ, manual, instructions, technical support, code, script, program, software package release, version, archive, file, conversion, table, tabular, numerical, numeric, quantitative, percentage, percent, price, market data, Reuter, newswire. Labels are flagged as broad discourse if they include terms related to debate, critique, controversy, policy, institutional or political discussion, economic conflict, or broad framing, including keywords such as: discussion, debate, critique, controversy, tensions, policy, institutional, economic, financial, trade, political, rhetoric, argument, concerns, issues, context, general, broad. Remaining labels are assigned the default semantic flag. The rules are applied in this order: format/register/numeric first, broad discourse second, and semantic as the default.

<table><tr><td>Dataset</td><td>Candidate pool</td><td>n</td><td>Med. nearest</td><td>Med. random</td><td>Med. ∆</td><td> $\% \cos _ { \mathrm { a c t } } \geq . 2$ </td></tr><tr><td>20NG</td><td>top-20</td><td>129</td><td>.239</td><td>.072</td><td>.154</td><td>60.5</td></tr><tr><td>20NG</td><td>top-50</td><td>129</td><td>.295</td><td>.055</td><td>.218</td><td>72.9</td></tr><tr><td>20NG</td><td>top-100</td><td>129</td><td>.312</td><td>.044</td><td>.262</td><td>78.3</td></tr><tr><td>WoS</td><td>top-20</td><td>26</td><td>.360</td><td>.104</td><td>.235</td><td>76.9</td></tr><tr><td>WoS</td><td>top-50</td><td>26</td><td>.419</td><td>.079</td><td>.295</td><td>88.5</td></tr><tr><td>WoS</td><td>top-100</td><td>26</td><td>.442</td><td>.066</td><td>.349</td><td>96.2</td></tr><tr><td>Reuters</td><td>top-20</td><td>334</td><td>.170</td><td>.059</td><td>.105</td><td>41.6</td></tr><tr><td>Reuters</td><td>top-50</td><td>334</td><td>.170</td><td>.043</td><td>.130</td><td>45.2</td></tr><tr><td>Reuters</td><td>top-100</td><td>334</td><td>.188</td><td>.029</td><td>.153</td><td>47.9</td></tr></table>

Table 9: Coverage audit details for lower-validation high-association features. The high-risk excluded set is fixed as features with $\mathrm { I S } ( f ) < . 8$ and best all-feature topic rank at most 20. Candidate pool size controls how many validated same-topic descriptors are allowed to provide coverage. Median random cosine is computed from random validated descriptors sampled from the same topic-specific candidate pool.

Examples from the excluded-feature audit. Table 10 gives deterministically selected examples from the label-type audit. To avoid cherry-picking, examples are selected by a fixed rule: within each dataset and heuristic label-type category, we sort features by best all-feature topic rank, break ties by larger $\beta _ { t , f } ^ { \mathrm { a l l } }$ , and show the first available feature. The table is intended to make the heuristic categories transparent, not to serve as a separate evaluation.

Concrete high-association cases illustrate the two main patterns. In 20 Newsgroups, an excluded feature labeled “Israeli policy critique with Nazi analogies" is ranked first under $B ^ { \mathrm { a l l } }$ for the topic aligned to talk.politics.mideast. Its nearest validated same-topic descriptor is “Israeli-Palestinian conflict, criticism, and political dispute, with activation cosine .650. In Web of Science, the excluded feature “Rainwater harvesting for water resource management" is ranked first for the Civil topic, but its nearest validated same-topic descriptor is “Rainwater harvesting for sustainable water management," with activation cosine .732. These cases look less like independent missing topics and more like lower-validation variants of semantic regions already covered by validated descriptors. By contrast, Reuters features such as “Financial dividend and earnings report announcements" or “Dividend and earnings financial report language" are systematic and topic-associated, but they primarily describe reporting format or financial-news register rather than substantive topical content.

<table><tr><td>Dataset</td><td>Type</td><td>IS(f)</td><td>Excluded label</td><td>Nearest validated label</td><td>COSact</td></tr><tr><td>20NG</td><td>semantic</td><td>.410</td><td>Patient narratives of challenging medical experiences</td><td>Alternative medicine debate and medical authority skepticism</td><td>.337</td></tr><tr><td>20NG</td><td>broad</td><td>.783</td><td>Detailed, analytical discussion of professional hockey</td><td>Toronto Maple Leafs playoff game discussion &amp; fandom</td><td>.301</td></tr><tr><td>20NG</td><td>format/reg.</td><td>.727</td><td>Sports news and scores reporting via Usenet</td><td>Ironic, irreverent, and self-aware in- ternet discourse</td><td>.176</td></tr><tr><td>WoS</td><td>semantic</td><td>.333</td><td>Rainwater harvesting for water re- source management</td><td>Rainwater harvesting for sustainable water management</td><td>.732</td></tr><tr><td>WoS</td><td>broad</td><td>.750</td><td>Distributed computing system archi- tecture and design discuss</td><td>Intelligent interconnected devices and adaptive environments</td><td>.443</td></tr><tr><td>WoS</td><td>format/reg.</td><td>.750</td><td>Novel materials for photocatalytic energy conversion</td><td>Photocatalysis, solar energy, mate- rial science research</td><td>.692</td></tr><tr><td>Reuters</td><td>semantic</td><td>.667</td><td>Bank ownership stake acquisitions and control</td><td>Corporate deals, mergers, and joint ventures reporting</td><td>.341</td></tr><tr><td>Reuters</td><td>broad</td><td>.769</td><td>International trade disputes and pol-</td><td>International trade disputes and gov- ernment intervention</td><td>.315</td></tr><tr><td>Reuters</td><td>format/reg.</td><td>.256</td><td>icy tensions US-USSR grain trade reporting, quantitative data</td><td>U.S. farm policy and commodity support programs</td><td>.312</td></tr></table>

Table 10: Deterministically selected examples from the excluded-feature label-type audit. Examples are selected by sorting, within each dataset and heuristic type, by best all-feature topic rank and then by larger audit-distribution topic mass.

## M Descriptor Baseline and Topic-Naming Details

For the controlled descriptor comparison in Section 5.3.3, all methods describe the same fixed document-topic mixtures Θ. We generate descriptors for each inferred topic using four mechanisms: θ-weighted TF-IDF terms, BERTopic-style c-TF-IDF terms, free-form LLM-generated descriptors, and MonoTM validated SAE-feature descriptors.

Word-based descriptor preprocessing. Both word-based baselines are implemented with scikit-learn vectorizers. We do not apply stemming or lemmatization. The vectorizers lowercase all text, remove English stopwords, and use unigrams only. We use the token pattern

$$
( \ref { 23 1 b 1 } ) \times b [ a - z A - 2 ] [ a - z A - 2 a - 9 - 1 - ] \{ 2 , \} \backslash { \mathsf { b } } ,
$$

which keeps tokens that begin with an ASCII letter and have length at least three, while allowing later alphanumeric, underscore, and hyphen characters. This removes many short tokens, numbers-only tokens, and punctuation artifacts. We use a maximum vocabulary size of 50,000 terms.

θ-weighted TF-IDF. For the θ-weighted TF-IDF baseline, we fit a TfidfVectorizer on the full aligned corpus with lowercase=True, stop\_words=english, ngram\_range=(1,1), min\_df=2, max\_df=0.95, max\_features=50000, norm=None, and sublinear\_tf=True. For each topic t, let $D _ { t } ^ { 5 0 }$ be the 50 documents with the largest $\theta _ { d , t }$ . We normalize topic weights within this top-document set,

$$
\bar { \theta } _ { d , t } = \frac { \theta _ { d , t } } { \sum _ { d ^ { \prime } \in D _ { t } ^ { 5 0 } } \theta _ { d ^ { \prime } , t } } ,
$$

and score each term w by

$$
s _ { t } ^ { \mathrm { t f i d f } } ( w ) = \sum _ { d \in D _ { t } ^ { 5 0 } } \bar { \theta } _ { d , t } \mathrm { t f i d f } _ { d , w } .
$$

We report the 30 highest-scoring terms for each topic.

c-TF-IDF. For the c-TF-IDF baseline, we form one class document for each topic by concatenating the same 50 top-θ documents. We then fit a CountVectorizer with lowercase=True, stop\_words=english, ngram\_range=(1,1), min\_df=1, max\_features=50000, and the same token pattern as above. Let $n _ { t , w }$ be the count of term w in topic class document t, and let $\begin{array} { r } { L _ { t } = \sum _ { w } n _ { t , w } } \end{array}$ be the class-document length. We compute

$$
\mathrm { t f } _ { t , w } = \frac { n _ { t , w } } { L _ { t } } .
$$

Let $\mathrm { d f } _ { \mathrm { c l a s s } } ( w )$ be the number of topic class documents in which w appears, and let L be the average class-document length across topics. We use the class-based inverse-document-frequency term

$$
\mathrm { i d f } _ { w } = \log \left( 1 + \frac { \bar { L } } { \operatorname* { m a x } ( \mathrm { d f } _ { \mathrm { c l a s s } } ( w ) , 1 ) } \right) ,
$$

and rank terms by

$$
s _ { t } ^ { \mathrm { c t f i d f } } ( w ) = \mathrm { t f } _ { t , w } \operatorname { i d f } _ { w } .
$$

We report the 30 highest-scoring terms for each topic.

Free-form LLM descriptor baseline. The LLM summary baseline is intended to be a strong posthoc descriptor baseline. For each topic, the prompt receives 50 topic-associated documents. The 25 highest-θ documents are always included, and the remaining 25 documents are sampled without replacement from the top 200 documents for that topic using a fixed random seed. Each document is first whitespace-normalized and then truncated to at most 1000 model tokens; if tokenization support is unavailable, the implementation falls back to whitespace-token truncation. We do not apply an additional character cap.

We use OpenAI gpt-5-2025-08-07 through the Responses API. For LLM-generated descriptors, we use reasoning\_effort=low, verbosity=low, JSON-object output mode, and a maximum output budget of 3000 tokens. We leave temperature unspecified. If the response cannot be parsed into exactly 10 descriptors, we retry up to three times. The prompt asks the model to produce exactly 10 concise, non-redundant descriptors grounded only in the provided excerpts.

## Free-form LLM descriptor prompt

You are given excerpts from documents that are highly   
associated with the same latent topic.   
Task: Produce exactly 10 concise descriptors for this   
topic.   
Rules: Each descriptor must be 4 to 10 words.   
Descriptors should be specific, non-redundant, and   
grounded only in the excerpts. Do not output a   
single topic name; output a ranked list of descrip  
tors. Return JSON only, with this exact schema:   
{"descriptors": ["...", "..."]}.   
Topic id: <topic>   
Documents: <document excerpts>

MonoTM descriptors. For MonoTM, we use the topic-feature distribution B estimated by the fixed-Θ descriptor model described in Appendix K. We retain validated SAE features with $\operatorname { I S } ( f ) \geq 0 . 8$ and report the top 10 feature labels under $\beta _ { t , f }$ for each topic. Unlike the free-form LLM descriptor baseline, MonoTM does not generate each topic's descriptor list freely from top documents. Instead, the descriptor list is selected from a fixed validated feature vocabulary grounded in document-embedding activations.

Topic naming for compact qualitative comparison. For compact comparison in Tables 12, 13, and 14, we generate high-level topic names from each descriptor list. The naming model receives only the descriptor list and the descriptor method; it does not receive the gold label or the candidatelabel list. For word-based methods, the naming prompt receives the 30 ranked terms. For the freeform LLM descriptor baseline and MonoTM, it receives the 10 semantic descriptors.

We use the same OpenAI model, gpt-5-2025-08-07, through the Responses API with reasoning\_effort=low, verbosity=low, JSON-object output mode, and a maximum output budget of 300 tokens. We again leave temperature unspecified and retry up to three times if the JSON response cannot be parsed.

Topic naming prompt   
I have a ranked list of <top words/phrases or   
semantic descriptors> derived from a single la  
tent topic.   
Descriptor method: <method>   
Ranked descriptors: <descriptor list>   
Task: Identify the high-level topic that these descrip  
tors likely came from.   
Rules: Output a concise topic name in 1 to 3 words.   
Do not include quotes, explanations, bullets, or ex  
tra text. Return JSON only with this exact schema:   
{"name": "..."}.

## N High-Level Topic Labels Across SAE Configurations

To provide an intuitive qualitative check of MonoTM's output, we take each topic's top-10 interpretable feature labels (ranked by $\beta _ { t , f } )$ and ask gpt-4.1-2025-04-14 (temperature 0) to produce a 1–3 word topic name using the prompt in Appendix M. Table 11 shows these names for three SAE configurations ((N=1, K=8), (N=3, K=16), (N=5, K=32)). For an informative comparison, we align inferred topics to gold categories using the process described in Section 5.2. In many cases, the generated topic labels closely paraphrase the matched gold labels.

<table><tr><td>Gold Label</td><td>N=1, K=8</td><td>N=3, K=16</td><td>N=5, K=32</td></tr><tr><td colspan="4">20 Newsgroups</td></tr><tr><td>rec.sport.hockey</td><td>Online Hockey Discussions</td><td>NHL Hockey Fans</td><td>Professional Hockey Discourse</td></tr><tr><td>soc.religion.christian</td><td>Christian Theology Debates</td><td>Religious Controversies</td><td>Religion and Morality</td></tr><tr><td>rec.motorcycles</td><td>Motorcycle Riding</td><td>Motorcycle Safety</td><td>Motorcycle Culture</td></tr><tr><td>rec.sport.baseball</td><td>Baseball Online Communities</td><td>Baseball Discussion</td><td>Baseball Analysis</td></tr><tr><td>sci.crypt</td><td>Cypherpunk Movement</td><td>Cryptography and Surveillance</td><td>Clipper Chip Controversy</td></tr><tr><td colspan="4">Web of Science</td></tr><tr><td>Medical Psychology</td><td>Human Disease Research</td><td>Human Health Research</td><td>Human Health Disorders</td></tr><tr><td></td><td>Child and Adolescent Psychology</td><td>Psychological Research</td><td>Social Psychology</td></tr><tr><td>CS biochemistry</td><td>Computer Science</td><td>Computer Science</td><td>Computer Science</td></tr><tr><td>ECE</td><td>Biomedical Science Control Engineering</td><td>Biomedical Research Electrical Engineering</td><td>Biomedical research Electrical Engineering</td></tr><tr><td colspan="4">Reuters</td></tr><tr><td>earn</td><td>Quarterly Financial Reporting</td><td>Financial Reporting</td><td>Financial Performance</td></tr><tr><td>acq</td><td>Mergers and Acquisitions</td><td>Financial News Events</td><td>Comparison Mergers and Acquisitions</td></tr><tr><td>money-fx</td><td>Central Bank Operations</td><td>Bank of England Interventions</td><td>Central Bank Interventions</td></tr><tr><td>grain</td><td>Agricultural Trade Markets</td><td>Agricultural Trade Reporting</td><td>Agricultural Trade</td></tr><tr><td>crude</td><td>Global Oil Markets</td><td>Global Oil Markets</td><td>OPEC Oil Markets</td></tr></table>

Table 11: Topic labels generated from MonoTM outputs across three SAE configurations. The gold labels are shown only for orientation and are not provided during topic naming.

## O Additional Descriptor Comparison Results

Tables 12-14 report additional topic-name tables for the controlled descriptor comparison in Section 5.3.3. As in the main text, all methods describe the same inferred topics under fixed documenttopic mixtures Θ, and reference labels are shown only for orientation.

<table><tr><td>Reference</td><td>MonoTM</td><td>θ-TF-IDF</td><td>c-TF-IDF</td><td>LLM-generated</td></tr><tr><td>alt.atheism</td><td>Religious debate</td><td>Objective morality</td><td>Objective morality</td><td>Meta-ethics</td></tr><tr><td>comp.graphics</td><td>Computer Graphics</td><td>Image format conversion</td><td>Image file formats</td><td>Computer Graphics</td></tr><tr><td>comp.os.ms-windows.misc</td><td>PC hardware troubleshooting</td><td>Hard drive setup</td><td>PC hard drives</td><td>DOS disk configuration</td></tr><tr><td>comp.sys.ibm.pc.hardware</td><td>Technical troubleshooting</td><td>Lead-acid batteries</td><td>Lead-acid batteries</td><td>Lead-acid battery storage</td></tr><tr><td>comp.sys.mac.hardware</td><td>PC hardware troubleshooting</td><td>Macintosh hardware</td><td>Macintosh hardware</td><td>Macintosh video hardware</td></tr><tr><td>comp.windows.x</td><td>X11 troubleshooting</td><td>X11 programming</td><td>X11 Motif</td><td>X11 programming</td></tr><tr><td>misc.forsale</td><td>Tech classifieds</td><td>Comics for sale</td><td>Marvel Comics</td><td>Classifieds</td></tr><tr><td>rec.autos</td><td>Automotive discussion</td><td>Cars</td><td>Cars</td><td>Performance cars</td></tr><tr><td>rec.motorcycles</td><td>Motorcycle Culture</td><td>Motorcycles</td><td>Motorcycles</td><td>Motorcycle riding techniques</td></tr><tr><td>rec.sport.baseball</td><td>Baseball Analytics</td><td>Baseball</td><td>Baseball</td><td>Baseball sabermetrics</td></tr><tr><td>rec.sport.hockey</td><td>Ice hockey</td><td>NHL Playoffs</td><td>NHL hockey</td><td>NHL Playoffs</td></tr><tr><td>sci.crypt</td><td>Clipper Chip Debate</td><td>Clipper Chip</td><td>Clipper Chip</td><td>Clipper Chip</td></tr><tr><td>sci.electronics</td><td>Computer hardware</td><td>Electronic circuits</td><td>Electronic circuits</td><td>Clickless audio switching</td></tr><tr><td>sci.med</td><td>Medical skepticism</td><td>Candida yeast syndrome</td><td>Candida overgrowth</td><td>Candida overgrowth</td></tr><tr><td>sci.space</td><td>Space commercialization</td><td>Lunar exploration</td><td>Spaceflight</td><td>Lunar habitation prize</td></tr><tr><td>soc.religion.christian</td><td>Theological debates</td><td>Christianity</td><td>Christian theology</td><td>Biblical hermeneutics</td></tr><tr><td>talk.politics.guns</td><td>Waco siege</td><td>Waco siege</td><td>Waco siege</td><td>Waco Siege</td></tr><tr><td>talk.politics.mideast</td><td>Ethno-national conflicts</td><td>Armenian Genocide</td><td>Armenian Genocide</td><td>Armenian genocide controversy</td></tr><tr><td>talk.politics.misc</td><td>American culture wars</td><td>Gun rights</td><td>Gun rights</td><td>Gay rights debate</td></tr><tr><td>talk.religion.misc</td><td>Usenet discussions</td><td>Email</td><td>Usenet newsgroups</td><td>Mailing list administration</td></tr></table>

Table 12: High-level topic names generated from descriptor outputs on 20 Newsgroups. All methods describe the same inferred topics under fixed document-topic mixtures Θ. Reference labels are shown only for orientation and are not provided to the topic-naming model.
<table><tr><td>Reference</td><td>MonoTM</td><td>θ-TF-IDF</td><td>c-TF-IDF</td><td>LLM-generated</td></tr><tr><td>biochemistry</td><td>Biomedical research</td><td>Cancer cell signaling</td><td>Molecular oncology</td><td>Signal transduction</td></tr><tr><td>Civil</td><td>Civil and Environmental Engineering</td><td>Rainwater harvesting</td><td>Rainwater Harvesting</td><td>Rainwater harvesting</td></tr><tr><td>CS</td><td>Computer Science Research</td><td>Big Data Computing</td><td>Big Data</td><td>Cloud and Distributed Systems</td></tr><tr><td>ECE</td><td>Control Systems Engineering</td><td>Power converter control</td><td>Power Converter Control</td><td>Digital Inverter Control</td></tr><tr><td>MAE</td><td>Computational mechanics Composite Materials</td><td></td><td>Composite materials</td><td>Mechanical behavior of materials</td></tr><tr><td>Medical</td><td>Clinical epidemiology</td><td>Male hypogonadism</td><td>Male hypogonadism</td><td>Male hypogonadism</td></tr><tr><td>Psychology</td><td>Psychology research</td><td>Parenting and child behavior</td><td>Child Development</td><td>Prosocial Behavior</td></tr></table>

Table 13: High-level topic names generated from descriptor outputs on Web of Science. All methods describe the same inferred topics under fixed document-topic mixtures Θ. Reference labels are shown only for orientation and are not provided to the topic-naming model.
<table><tr><td>Reference</td><td>MonoTM</td><td>θ-TF-IDF</td><td>c-TF-IDF</td><td>LLM-generated</td></tr><tr><td>earn</td><td>Earnings reports</td><td>Earnings Reports</td><td>Earnings reports</td><td>Corporate Earnings</td></tr><tr><td>acq</td><td>Mergers and Acquisitions</td><td>Bank mergers</td><td>Bank Mergers</td><td>Bank Mergers</td></tr><tr><td>money-fx</td><td>Money market interventions</td><td>Money market operations</td><td>Money market</td><td>Open market operations</td></tr><tr><td>grain</td><td>Agricultural Trade</td><td>Export Enhancement Program</td><td>Grain export subsidies</td><td>Export Enhancement Program</td></tr><tr><td>crude</td><td>OPEC oil market</td><td>Oil production</td><td>OPEC oil production</td><td>Ecuador oil crisis</td></tr><tr><td>trade</td><td>US-Japan Trade Disputes</td><td>US-Japan trade dispute</td><td>US-Japan trade dispute</td><td>U.S.-Japan trade dispute</td></tr><tr><td>interest</td><td>Interest Rate Changes</td><td>Prime rate</td><td>Interest rates</td><td>Prime rate changes</td></tr><tr><td>ship</td><td>Persian Gulf conflict</td><td>Persian Gulf Tanker War</td><td>Gulf Tanker War</td><td>Operation Nimble Archer</td></tr><tr><td>wheat</td><td>Corporate financial distress</td><td>Earnings reports</td><td>Earnings reports</td><td>Nonrecurring items</td></tr><tr><td>corn</td><td>Mergers and Acquisitions</td><td>Retail mergers and acquisitions</td><td>Business news</td><td>Consumer retail M&amp;A</td></tr><tr><td>oilseed</td><td>Financial reporting</td><td>Corporate earnings</td><td>Corporate earnings</td><td>Corporate earnings</td></tr><tr><td>sugar</td><td>Agricultural trade</td><td>Sugar crop production</td><td>Crop production</td><td>Global sugar production</td></tr><tr><td>dlr</td><td>International monetary policy</td><td>Louvre Accord</td><td>Louvre Accord</td><td>G7 currency coordination</td></tr><tr><td>gnp</td><td>Macroeconomic Indicators</td><td>Economic Growth</td><td>Macroeconomic Indicators</td><td>OECD economic outlook</td></tr><tr><td>coffee</td><td>International Coffee Trade</td><td>International coffee trade</td><td>International Coffee Agreement</td><td>International Cocoa Agreement</td></tr><tr><td>veg-oil</td><td>Trade and M&amp;A</td><td>Mergers and acquisitions</td><td>Mergers and acquisitions</td><td>HBJ takeover battle</td></tr><tr><td>money-supply</td><td>Federal Reserve operations</td><td>Open market operations</td><td>Open market operations</td><td>Open market operations</td></tr><tr><td>gold</td><td>Mining Industry Finance</td><td>Gold mining</td><td>Gold mining</td><td>Gold mining</td></tr><tr><td>nat-gas</td><td>Airline Mergers</td><td>Airline mergers</td><td>Airline mergers</td><td>TWA USAir takeover</td></tr><tr><td>livestock</td><td>US Agricultural Policy</td><td>Livestock futures</td><td>Livestock futures</td><td>Meatpacking labor disputes</td></tr></table>

Table 14: High-level topic names generated from descriptor outputs on the 20 most frequent Reuters labels. Because Reuters is multi-label and imbalanced, reference labels should be read as orientation labels rather than exhaustive topic descriptions.

## P Representative Descriptor Lists

Table 15 provides representative descriptor lists before topic-name compression for the controlled descriptor comparison in Section 5.3.3. All methods describe the same fixed document-topic mixtures Θ. The examples illustrate the different roles of the descriptor mechanisms: word-based methods expose lexical anchors, free-form LLM-generated descriptors often summarize salient document clusters, and MonoTM descriptors are selected from a fixed vocabulary of validated SAE features ranked by a model-estimated topic-feature distribution.

<table><tr><td>Reference</td><td>Method</td><td>Top descriptors</td></tr><tr><td>sci.crypt</td><td>MonoTM</td><td>Distrust of government and defense of liberty; government skepticism, privacy concerns, argumentative discourse; technical discourse on cryptographic algorithms and implementation; secure voice tech, compression, early internet</td></tr><tr><td>sci.crypt</td><td>θ-TF-IDF</td><td>security; government surveillance and cryptographic privacy concerns clipper; encryption; keys; escrow; chip; government; technology; secure; nsa; key</td></tr><tr><td>sci.crypt</td><td>c-TF-IDF</td><td>encryption; clipper; chip; government; technology; key; keys; escrow; nsa; secure</td></tr><tr><td>sci.crypt</td><td>LLM-generated</td><td>Government key-escrowed Clipper Chip proposal; secret Skipjack algorithm and family key; law enforcement access via escrowed unit keys; LEAF transmits session key encrypted for authorities; critics warn of mandated, tappable encryption standard</td></tr><tr><td>talk.politics.misc</td><td>MonoTM</td><td>Distrust of government and defense of liberty; contentious debate surrounding homosexuality and morality; moral debates about sexuality and statistics; gun control, self-defense arguments, statistical debate; contentious social</td></tr><tr><td>talk.politics.misc</td><td>θ-TF-IDF</td><td>debate, group identity, and conflict cramer; optilink; clayton; homosexuals; people; gay; gun; militia; rights; sexual</td></tr><tr><td>talk.politics.misc</td><td>c-TF-IDF</td><td>people; cramer; edu; optilink; militia; homosexuals; gay; gun; sexual; rights</td></tr><tr><td>talk.politics.misc</td><td>LLM-generated</td><td>Debating Kinsey versus Guttmacher homosexuality prevalence statistics; claims homosexuals exaggerate numbers to sway politicians; arguments over gay rights anti-discrimination employment laws; libertarian stance: private hiring without government interference; accusations linking gays with child molestation, NAMBLA</td></tr><tr><td>ECE</td><td>MonoTM</td><td>Engineered control systems and PID control design; electrical power system analysis and control; high-performance electric motor design and analysis; analog circuit design and performance metrics; engineered system optimization</td></tr><tr><td>ECE</td><td>θ-TF-IDF</td><td>and energy efficiency analysis converter; control; digital; voltage; proposed; power; current; controller; circuit; system</td></tr><tr><td>ECE</td><td>c-TF-IDF</td><td>control; proposed; voltage; converter; power; current; controller; circuit; digital; system</td></tr><tr><td>ECE</td><td>LLM-generated</td><td>FPGA-based digital control for power converters; digital PLLs for single-phase grid synchronization; discrete-time modulation introducing transport delays; finite control set MPC implemented fully on FPGA; deadbeat predictive current control in inverters</td></tr><tr><td>Civil</td><td>MonoTM</td><td>Applied hydrology, water resource engineering, rainwater harvesting; engineered system optimization and energy efficiency analysis; sustainable building and environmental responsibility concepts; construction project management,</td></tr><tr><td>Civil</td><td>θ-TF-IDF c-TF-IDF</td><td>analytical methodology, performance evaluation; water pollution assessment and scientific reporting water; rainwater; rwh; harvesting; demand; tank; system; saving; efficiency; supply water; rainwater; rwh; harvesting; use; demand; tank; saving; efficiency; system</td></tr><tr><td>Civil Civil</td><td>LLM-generated</td><td>Rainwater harvesting supplying toilets, laundry, irrigation; economic feasibility via payback and NPV analyses; optimal storage tank sizing using daily simulations; water saving efficiency and reliability metrics; integrating</td></tr><tr><td></td><td></td><td>rainwater with greywater reuse systems Oil price stability and OPEC market control; OPEC oil production, pricing, and market regulation; OPEC oil price</td></tr><tr><td>crude</td><td>θ-TF-IDF</td><td>and production control; OPEC oil production, pricing, and regulation reports; oil market data and energy economics bpd; oil; ecuador; opec; barrels; crude; prices; pipeline; production; exports</td></tr><tr><td>crude</td><td>c-TF-IDF</td><td>said; oil; bpd; opec; ecuador; crude; barrels; pipeline; production; prices</td></tr><tr><td>crude</td><td>LLM-generated</td><td>Earthquake halts Ecuador crude exports, declares force majeure; Lago Agrio–Balao pipeline severely damaged, costly repairs; temporary pipeline link to Colombia&#x27;s Tumaco port; Venezuela lends Ecuador crude for exports, domestic</td></tr><tr><td></td><td></td><td>needs; Ecuador plans over-quota OPEC production to repay loans</td></tr><tr><td>ship</td><td>MonoTM</td><td>Government economic intervention and trade policy; US-Iran military conflict and retaliation narratives; international military conflict and geopolitical tension; Persian Gulf naval conflict and escalation threat; Persian Gulf oil security, U.S. military response</td></tr><tr><td>ship</td><td>θ-TF-IDF</td><td>iran; gulf; iranian; attack; kuwaiti; oil; reflagged; ship; tanker; platform</td></tr><tr><td>ship ship</td><td>c-TF-IDF LLM-generated</td><td>iran; said; gulf; iranian; oil; attack; kuwaiti; ship; platform; tanker U.S. destroyers shell Iranian Rostam oil platform; retaliation for Silkworm strike on Sea Isle City; Navy raids second</td></tr><tr><td></td><td></td><td>platform, destroys radar communications; Iran vows crushing retaliation for platform attacks; escorting reflagged Kuwaiti tankers through Gulf</td></tr></table>

Table 15: Longer representative descriptor lists before topic-naming. Word-based descriptors often expose terms or named entities; LLM-generated descriptors often describe salient document clusters; MonoTM descriptors are selected from validated SAE features and ranked by a topic-feature distribution.

## Q Affordance Demonstration: Topic Relations in a Shared Semantic-Feature Space

This appendix demonstrates a downstream affordance of MonoTM's shared semantic-feature representation. The goal is not to introduce a new quantitative benchmark, but to illustrate how MonoTM can support analyses beyond compact topic description. Because MonoTM represents both documents and topics in a shared vocabulary of validated SAE features, users can inspect which features are shared across documents, which features distinguish neighboring topics, and which semantic attributes cut across topic boundaries. We demonstrate this affordance by analyzing how topics are related in the Web of Science corpus.

Construction details. We perform the analysis on Web of Science under the same fixed document– topic mixtures Θ used in the controlled descriptor comparison. For MonoTM, we use the estimated topic-feature distribution B and retain the top 200 validated features per topic. We keep the raw $\beta _ { t , f }$ masses rather than renormalizing them. For the lexical analogue, we use the top 50 terms per topic from the θ-weighted TF-IDF topic-term representation, with term scores normalized to sum to one within each topic.

For two topics i and j with MonoTM topicfeature distributions $\beta _ { i }$ and $\beta _ { j }$ , we define their shared feature mass as

$$
O ( i , j ) = \sum _ { f } \operatorname* { m i n } ( \beta _ { i , f } , \beta _ { j , f } ) .
$$

More generally, for either representation, we compute topic-pair overlap as

$$
O ( i , j ) = \sum _ { x } \operatorname* { m i n } ( v _ { i , x } , v _ { j , x } ) ,
$$

where x indexes validated SAE features for MonoTM and lexical terms for the TF-IDF analogue. To explain each edge, we rank shared items by min $( v _ { i , x } , v _ { j , x } )$ and report the top shared features or terms. The visualized networks display the eight strongest topic-pair overlaps, and the tables below report representative shared items from those edges.

To visualize the resulting topic relations, Figure 9 shows the strongest topic-pair overlaps as networks for MonoTM and the θ-weighted TF-IDF analogue. Figure 10 provides the corresponding full pairwise overlap matrices, showing whether the strongest network edges reflect isolated pairwise relations or broader overlap structure across topics.

Qualitative comparison. Figures 9 and 10 show that the two representations yield qualitatively different topic-relation structures and edge explanations. Table 16 reports representative shared items for several high-overlap topic pairs. MonoTM links topics through semantic features such as clinical disease investigation, severe mental illness and treatment, efficient algorithm implementation and hardware optimization, applied engineering education, and engineered system optimization and energy efficiency. By contrast, the θ-weighted TF-IDF analogue often explains topic links using lexical items such as model, systems, method, study, results, paper, and based. Thus, a word-based representation can indicate that two topics overlap lexically, but MonoTM can make the semantic basis of the relation more explicit.

Overall, this demonstration illustrates a broader affordance of MonoTM: the same validated feature vocabulary can support topic description, topic comparison, and inspection of semantic attributes that cut across benchmark category boundaries.

<table><tr><td>Topic pair</td><td>MonoTM shared validated features</td><td>θ-TF-IDF shared terms</td></tr><tr><td>biochemistry-Medical</td><td>Pulmonary disease research and clinical investigation focus.; Cancer research, molecular mechanisms, and clinical outcomes.; Gastrointestinal distress, IBS/IBD, patient well-being</td><td>study</td></tr><tr><td>Medical-Psychology</td><td>Severe mental illness, clinical diagnosis, and treatment.; Anxiety, study; results; levels depression, and psychological distress research.; Child development, family well-being, healthcare support.</td><td></td></tr><tr><td>CS-ECE</td><td>Efficient algorithm implementation and hardware optimization.; Applied technology for infrastructure and efficiency.; Mathematical modeling and algorithmic development in data science</td><td>paper; based; model; performance; proposed</td></tr><tr><td>MAE-ECE</td><td>Magnetic material design and analysis in engineering.; Precise sensor measurement and instrumentation details.; Detailed thermodynamic system analysis and optimization.</td><td>model; experimental; method; using; results</td></tr><tr><td>Civil–ECE</td><td>Engineered system optimization and energy efficiency analysis.; Engineered thermal systems and heat transfer analysis.; Applied technology for infrastructure and efficiency.</td><td>model; systems; performance; using</td></tr><tr><td>MAE-Civil</td><td>Engineered thermal systems and heat transfer analysis.; Applied engineering education and curriculum design.; Technical design, optimization, and performance analysis.</td><td>model; using</td></tr></table>

Table 16: Representative shared items explaining Web of Science topic-pair edges. MonoTM edge explanations are validated SAE features ranked by shared $\beta$ mass, while the lexical analogue explains edges through shared θ-weighted TF-IDF terms.

(a) MonoTM Feature Overlap Between Topics  
![](images/3dead76b8a16893ebdf70a7385b48f1d17a302b7c71ed7a92bf05a8603684810.jpg)

(b) TF-IDF Keyword Overlap Between Topics  
![](images/961e9f6ef6cee232ccda50544a054ce30ef0bae24e50abd2904b13e00e382f3f.jpg)  
Figure 9: Web of Science topic relation networks under fixed document-topic mixtures. Edges show the strongest topic-pair overlaps, with edge width proportional to shared mass. (a) MonoTM computes overlap in the validated SAE-feature space, so edge labels identify the semantic features that explain each topic connection. (b) A θ-weighted TF-IDF analogue computes overlap in a lexical term space, so edge labels identify shared keywords. The comparison illustrates how MonoTM supports feature-level interpretation of topic relations beyond lexical overlap.

![](images/b6f3de7bbd35a3196c5b97fa04eb9c05984bb08b0bde1f86ba654656f07e4f7d.jpg)

![](images/376865414f3343746b92b3ef2c9ca5b8789482e640ff7619a093b691ab2e4cbe.jpg)  
Figure 10: Topic-pair overlap matrices for Web of Science. Left: MonoTM shared-feature mass using the top 200 validated features per topic from B. Right: θ-weighted TF-IDF shared-term mass using the top 50 lexical terms per topic. Diagonal entries are set to zero.