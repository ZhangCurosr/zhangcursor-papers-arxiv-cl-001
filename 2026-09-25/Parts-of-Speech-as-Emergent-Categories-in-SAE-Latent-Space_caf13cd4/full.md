# Parts-of-Speech as Emergent Categories in SAE Latent Space

Alessandro Bondielli<sup>1,2,\*</sup>, Lucia Passaro<sup>1,2,\*</sup>, Serena Auriemma<sup>1</sup>, Alessandro Lenci<sup>1</sup>

<sup>1</sup>CoLingLab, Department of Philology, Literature and Linguistics, University of Pisa <sup>2</sup>Department of Computer Science, University of Pisa

\*Equal contribution. Correspondence: alessandro.bondielli@unipi.it, lucia.passaro@unipi.it, Preprint version of the paper in the Proceedings of EMNLP 2026.

## Abstract

Sparse AutoEncoders (SAEs) offer a promising way to inspect language model representations, but it is still unclear what kind of linguistic structure their latents expose. We use part-ofspeech (POS) categories as a controlled test case to study whether morpho-syntactic information is encoded by individual latents or by structured groups of features. We find that POS distinctions are highly recoverable from SAE activations, but do not align with one-to-one latent / category mappings. This recoverability is not reducible to lexical memorisation, and Open and Closed POS classes differ substantially. Categories are supported by compact groups of sparse latents, with substantial variation across tags. These groups remain stable on held-out data, while also showing overlap between related categories. Our results show that SAEs localise morpho-syntactic informa tion in a distributed and category-dependent form rather than through atomic grammatical features.<sup>1</sup>.

## 1 Introduction

Large language models (LLMs) encode a wide range of linguistic regularities in their internal representations, from lexical and syntactic information to more abstract semantic and discourse-level properties. Yet, despite substantial progress in probing and representation analysis, it remains unclear how such information is organized internally, for instance whether linguistic categories correspond to localized and interpretable units, or they are instead distributed across many dimensions of the representation space (Elhage et al., 2022). This question has become particularly relevant with the growing use of Sparse AutoEncoders (SAEs) as tools for interpreting LLMs (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024).

![](images/0ef4ab4faa9fa0fd64f369cf2d673c2886fa6e259f3b051cd8b2771f650fa977.jpg)  
Figure 1: Overview of the workflow. We test POS recoverability from token-level SAE activations, identify and compact POS-relevant latent groups, and validate them on held-out and controlled data.

SAEs aim to decompose dense model activations into high-dimensional sparse representations, where individual dimensions (aka latents) are expected to capture more interpretable directions of variation. SAE latents are expected to provide a bridge between low-level model activations and human-interpretable features. This has motivated their use in mechanistic interpretability, where they are often discussed in terms of feature discovery and monosemanticity (Elhage et al., 2022; Bricken et al., 2023; Templeton et al., 2024).

However, the relationship between SAE latents and linguistic categories is still not clear. In fact, the latter result from the combinations of multiple lexical, morphological, syntactic, and distributional features, which need not correspond to individual isolated latents. Understanding whether linguistic abstractions are localized or distributed in SAE spaces is thus important for evaluating what kind of interpretability SAEs provide (Kantamneni et al., 2025; Karvonen et al., 2025; Engels et al., 2025).

In this paper, we study this question by targeting part-of-speech (POS) categories. POS tags offer a controlled testbed for analyzing morphosyntactic abstraction: They are discrete, independently annotated, and linguistically interpretable, while also differing in frequency, lexical openness, and syntactic function. For instance, open-class categories such as nouns and verbs are lexically productive and highly variable, whereas closedclass categories such as determiners, conjunctions, and pronouns are more restricted and often tied to specific syntactic roles. This makes POS a useful setting for testing whether SAE latents behave as localized linguistic features or instead participate in broader distributed representations.

We analyze SAE activations extracted from LLaMA-3-8B (Grattafiori et al., 2024) on the GUM Corpus Treebank (Zeldes, 2017). For each token, we encode its SAE representation into a sparse activation vector and study how gold Universal Dependencies (UD) POS tags are represented in this latent space. Our experimental design follows a three-step interpretability pipeline: i.) we use binary probing classifiers to test whether individual POS distinctions are recoverable from SAE activations (Belinkov, 2022); ii.) we use featuresalience analysis to rank the latents most relevant to each POS category, and coverage analysis to estimate how many of these latents are needed to account for most instances of the category; iii.) we validate the selected latent groups on held-out and controlled data, and test whether their union is sufficient to train a multi-class POS classifier.

This setup allows us to move beyond standard probing accuracy. A high probing score may show that POS information is present in SAE activations, but it does not explain how this information is organized (Hewitt and Liang, 2019; Pimentel et al., 2020; Belinkov, 2022). By combining probing, salience, coverage, compact-feature classification, and held-out validation, we can understand whether POS categories are associated with individual monosemantic latents or with structured groups of sparse latents. We also examine whether different POS categories are represented by different numbers of latents, which might suggest that the SAE representation reflects differences in the linguistic nature of the categories themselves.

We address three research questions: (i) RQ1: Are PoS categories explicitly encoded in SAE latent activations? (ii) RQ2: What is the organization of the POS categories encoding in the SAE latent spaces? (iii) RQ3: How stable and systematic are these latent representations across linguistic categories, datasets, and evaluation settings?

Our study makes two main contributions. First, we show that POS categories are aligned with structured groups of sparse features. Through feature-salience and coverage analyses, we quantify the size and organization of these groups, show that it varies substantially across categories, and highlight differences between Open- and Closedclass POS classes. Second, we show that these latent groups are compact yet effective: Their union preserves strong multi-class POS classification performance, and they remain stable on heldout data, while still exhibiting overlap across related categories.

## 2 Related Work

Probing classifiers have long been used to test what linguistic information neural language models encode in their representations (Conneau et al., 2018; Belinkov, 2022). Prior work shows that lower layers capture morpho-syntactic information such as POS, while higher layers encode more abstract semantic and discourse properties (Tenney et al., 2019a,b; Hewitt and Manning, 2019; Rogers et al., 2020). However, probing accuracy alone is limited: control tasks (Hewitt and Liang, 2019) and information-theoretic critiques (Pimentel et al., 2020) show that probes can fit arbitrary mappings, and that recoverability does not imply use. We share this concern, but shift the focus from what information is present to how it is organised at the level of individual sparse latents.

A growing body of work studies mechanistic interpretability in LLMs (Sharkey et al., 2025). Within this area, SAEs map dense activations to high-dimensional sparse vectors whose units are intended to be more monosemantic and interpretable (Bricken et al., 2023; Cunningham et al., 2023). Subsequent work has improved SAE training through scaling (Templeton et al., 2024) and TopK activations (Gao et al., 2025), and released open SAE suites for widely used base models (Lieberum et al., 2024; He et al., 2024). These studies often identify latents aligned with intuitive concepts, using top-activating examples or automated natural-language explanations, but provide limited evidence on how theoretically motivated linguistic categories are represented in latent space.

Recent work also questions whether SAE latents behave as genuinely monosemantic features. Kantamneni et al. (2025) find that probes trained on SAE latents do not consistently outperform simple baselines across 113 binary classification tasks, while SAEBench (Karvonen et al., 2025) shows that gains on standard SAE proxy metrics often do not transfer to downstream performance. Still, SAEs remain useful tools for probing LM knowledge and behaviour (Dupre la Tour and Mossing, 2025; Fraser-Taliente et al., 2026).

Closer to our work, Marks et al. (2025) use SAE features to construct interpretable causal circuits for syntactic phenomena such as subject–verb agreement, suggesting that morpho-syntactic information is at least partly recoverable from SAE space. At the same time, Engels et al. (2025) show that not all language model features are well captured by single linear directions. Existing linguistic analyses of SAEs have mainly focused on isolated phenomena, such as subject–verb agreement, or broad properties such as language identity. We address the open question of whether and how classical morpho-syntactic categories are encoded by latents, using POS as a controlled testbed beyond the binary probing regime explored by prior work.

## 3 Method and Materials

## 3.1 Dataset

We conducted our experiments on two datasets: a naturally occurring corpus and a small controlled dataset constructed for targeted evaluation.

The GUM treebank. For the naturally occurring data, we selected the UD English GUM treebank (Zeldes, 2017), annotated following the Universal Dependencies scheme.<sup>2</sup> We chose this treebank for its representativeness across diverse textual genres (academic, blog, legal, news, social, wiki, etc.), its medium size (14,353 sentences, 252,284 tokens), and its complete coverage of the 17 Universal PoS tags. The training split was used as the discovery set, while the test split was kept held out and used only to evaluate whether discovered activations remain active on unseen tokens of the corresponding POS categories.

Controlled dataset. To complement the naturally occurring data, we constructed a small controlled dataset of 180 lexical items to verify whether latents associated with specific POS tags activate systematically in minimal, grammatically well-formed sentences. The dataset focuses primarily on nouns and verbs. For nouns, we selected 160 items spanning multiple semantic categories (e.g., mammals, birds, flowers, vehicles, etc.), evenly split between animate and inanimate referents. Each noun was instantiated in singular and plural form within neutral templates, including impersonal constructions such as There is a dog and transitive constructions such as I see the dog and I have a dog. These templates vary determiner contexts, including indefinite articles, definite articles, and bare plurals. For verbs, we included 20 high-frequency verbs compatible with a minimal intransitive template (I + verb, as in I walk), balanced between 10 regular and 10 irregular pasttense forms, to limit the impact of morphological idiosyncrasies. All base sentences were augmented with two variants: one adding an adjacent adjective for noun sentences or adverb for verb sentences, and one appending punctuation to the augmented sentence. This allows us to assess whether additional POS tokens introduce their own characteristic activations and whether these interact with those observed in the base sentence. All sentences were also instantiated in present and past tense to account for potential tense-driven effects.

## 3.2 Processing Pipeline

In the following, we describe the processing pipeline to obtain SAE latent activations.

Model. We experiment on LLaMA-3-8B. We employ the EleutherAI/sae-llama-3-8b-32x pretrained model as our SAE. Both models are available on HuggingFace. The SAE model is trained and used via the Sparsify library.<sup>3</sup> The library is designed to follow the SAE implementation described in Gao et al. (2025).

Token-level Activations Extraction. To extract token-level activations, we feed the raw sentence text to the model using its original subword tokenizer, and recover hidden state activations from the residual stream of layer 30 (last layer before the output) of the model. We encode such activations with the SAE to produce the sparse activation vectors for each subword. We obtain, for each subword, the fraction of SAE latents that fired on that subword, and their activation strength. Then, we align subword tokens and UD surface forms via character-span overlap: For each UD token with character span $[ t _ { \mathrm { s t a r t } } , ~ t _ { \mathrm { e n d } } )$ , all subword tokens whose span $[ s , e )$ satisfies $s ~ < ~ t _ { \mathrm { e n d } }$ and $e > t _ { \mathrm { s t a r t } }$ are identified as overlapping. The leftmost such subword token is designated the anchor, and its SAE activations are adopted as the representation of the corresponding UD token. Note that we chose the leftmost subword because it is the position at which the UD token’s identity first becomes available to the model. Averaging over subwords may instead dilute category-bearing activations with continuation-piece activations. This yields, for each token, a dense SAE activation vector that is composed of all SAE latents that fired on the token and their activation strength.

Sparse Feature Matrix Construction. For the probing experiments, we construct a Sparse SAE Feature Matrix from dense token-level activations. To do so, we consider the union of latents active across the entire treebank, which constitutes a subset of the full SAE latent space, spanning $4 0 9 6 \times 3 2 = 1 3 1 , 0 7 2$ dimensions (i.e., LLM hidden size $\times { } \operatorname { S A E }$ expansion factor). We therefore project all token representations into the common sparse vector space defined by the latents observed at least once in the Treebank. Concretely, we construct a feature matrix $\mathbf { X } \in \mathbb { R } ^ { N \times D }$ , where N is the number of tokens and $D = 1 3 0 { , } 2 4 6$ is the number of attested latents, with entry $x _ { i , j }$ set to the activation strength of latent $j$ on token i, and zero otherwise. The sparse matrix is the input to the probing classifiers.<sup>4</sup>

## 4 Experiments

Our experiments are designed to assess not only whether POS information is recoverable from $\mathtt { S A E }$ activations, but also how this information is organized in the latent space. In particular, we structure the analysis around the three research questions introduced in Section 1. First, we test whether morpho-syntactic distinctions are explicitly available in the sparse activation space (RQ1). Second, we investigate the organization of POS categories in the latent space (RQ2). Third, we evaluate whether such organization is stable across splits and evaluation settings, and whether it supports general POS classification (RQ3).

The experimental pipeline proceeds as follows. We first assess the linear recoverability of each POS category with one-vs-rest probing classifiers (Section 4.1). We then localize POS-relevant latent groups by combining feature salience, coverage, and compactness analyses (Section 4.2). Finally, we test the robustness of the selected groups on held-out data (Section 4.3). This design separates recoverability, localization, and stability. Probing shows whether POS information is present, while localization and validation assess how such information is organized in the latent space.

## 4.1 Recoverability of POS Information (RQ1)

We first test whether POS distinctions are linearly recoverable from SAE activations. For each token in the GUM training split, we use the sparse SAE activation vector (cf. Section 3.2) as input representation and the gold UD POS tag as supervision.

We evaluate the one-vs-all setting using 5-fold cross-validation on the GUM Treebank Train split. For each POS category, we train a binary classifier to distinguish tokens with that POS from all other tokens. We train an L1-regularized logistic regression classifier $( \mathbf { C } = 0 . 1 )$ using the liblinear solver, with balanced class weighting to account for label imbalance. The L1 penalty encourages sparse weight vectors, effectively performing feature selection and yielding interpretable models where most coefficients are driven to zero, given the high-dimensional nature of SAE latent spaces.

This allows us to assess the extent to which individual POS distinctions are linearly recoverable from SAE activations.

## 4.2 POS Organization in Latent Space (RQ2)

We next ask how the POS information recovered by the probes can be localized in the space of SAE latents. To this end, we use the one-vs-rest classifiers introduced in Section 4.1 not only as predictive models, but also as feature-salience mechanisms.

Feature salience. For each POS category, the corresponding logistic regression classifier assigns a coefficient $\beta$ to each latent. Since $\beta > 0$ indicates that the activation of a latent increases the probability of the positive class, we rank latents for each category according to their positive coefficients. We consider only latents with $\beta > 0$ obtaining for each POS tag a salience-ranked list of features that support the classification of that category. This analysis moves from recoverability to localization: rather than asking whether POS information is present, we ask which sparse features contribute most to each distinction.

Coverage and compactness. We then quantify how compact each localized group is. For each POS tag c, let $L _ { c } ^ { ( k ) }$ denote the set of the top-k latents in its salience-ranked list, $T _ { c }$ the set of goldlabel tokens tagged with $c ,$ and $a _ { \ell } ( t )$ the activation of latent ℓ on token t. We define the coverage of $L _ { c } ^ { ( k ) }$ as:

$$
\operatorname { C o v } _ { c } ( k ) = \frac { 1 } { | T _ { c } | } \sum _ { t \in T _ { c } } \mathbf { 1 } \left[ \exists \ell \in L _ { c } ^ { ( k ) } : a _ { \ell } ( t ) > 0 \right]
$$

Coverage measures the proportion of tokens of category c for which at least one of the top-k salient latents is active. We define the number of latents required to account for category c as the smallest k such that coverage reaches a target threshold $\tau = 0 . 9 5$

$$
k _ { c } ^ { \star } = \operatorname* { m i n } \{ k \in \mathbb { N } : \operatorname { C o v } _ { c } ( k ) \geq \tau \}
$$

This gives an estimate of the effective size of the latent group associated with each POS category. We use a per-class threshold rather than a global top-k or coefficient threshold to avoid biasing the comparison due to high imbalance in i.) relevant latents for each POS and ii.) coefficient profiles in open- vs closed-classes. Comparing $k _ { c } ^ { \star }$ across tags allows us to test whether different categories are represented with different degrees of compactness, for example whether open-class categories require broader latent groups than closed-class categories.

Compact-feature classification. Finally, we test whether the localized latent groups are sufficient for joint POS prediction. Let $C$ denote the set of POS categories. We define the set of POS-relevant latents as the union of the minimal coverage sets:

$$
L ^ { \star } = \bigcup _ { c \in C } L _ { c } ^ { ( k _ { c } ^ { \star } ) }
$$

We then train a multinomial logistic regression classifier using only $L ^ { \star }$ as input features. This provides a stricter test of the localization procedure: if the selected latents capture systematic morpho-syntactic information, they should support multi-class POS classification with limited degradation. A substantial drop with respect to the full SAE representation would instead suggest that relevant information remains distributed across additional latents.

## 4.3 Validation on Held-Out Data (RQ3)

We evaluate whether the latent groups identified in Section 4.2 are stable beyond the data used to select them. The salience and coverage analyses are performed on the GUM training split, where POS tags may correlate with lexical identity, frequency, position, or local syntactic patterns. We therefore test the selected groups in two complementary settings: the held-out GUM test split and the controlled dataset described in Section 3.1.

To assess how specific each minimal latent group is to its target category in held out data we construct a cross-POS activation matrix. For each pair of categories $( c , c ^ { \prime } \in C )$ , we compute the probability that at least one latent in $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ is active on tokens whose gold label is c:

$$
\begin{array} { r } { M _ { c , c ^ { \prime } } = \frac { 1 } { | T _ { c } | } \sum _ { t \in T _ { c } } \mathbf { 1 } \left[ \exists , \ell \in L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) } : a _ { \ell } ( t ) > 0 \right] } \end{array}
$$

This analysis assesses whether the POSdiscriminative SAE latents are category-specific or shared across the various syntactic categories.

## 4.4 Controls and Baselines

We also provide a set of controls and baselines that address possible confounds and contextualise the SAE results. First, we perform a control experiment to test whether lexical identity (i.e., the specific word form, like the preposition $o f )$ can act as a confound for the probing experiments. To estimate how much of the original performance can be attributed to memorisation, we re-run the probe but assign each word type a random UPOS label. A probe relying solely on word identity would fit the control labels as well as the real ones.

Second, we provide several baselines for the probing experiment: i.) we use raw embeddings (layer 0) and raw activations from layer 30 as features for the probe instead of SAE activations; ii.) we provide a random latent subset baseline, where we re-run the compact-feature classification, but we keep a percentage of the L⋆ (0, 25, and 50) and randomly choose the remaining latents.

## 5 Results

## 5.1 RQ1: Recoverability of POS Information

The one-vs-rest probing results show that POS distinctions are consistently recoverable from SAE activations. As shown in Figure 2, the binary classifiers achieve high F1 scores for most categories across the 5-fold cross-validation setting. This indicates that morpho-syntactic information is explicitly available in the sparse latent space.

Performance, however, is not uniform across tags. Closed-class categories and low-variability labels (e.g., punctuation), are easier to recover, while more lexically heterogeneous or less frequent categories show lower scores. Interestingly, nouns and verbs are the best performing open POS. This suggests that recoverability is affected both by the linguistic nature of the category and by its support.

![](images/f799a26e73a32c01ac6d824d5089f1b8f9b10674353d4a81192ef9fb6ceda915.jpg)  
Figure 2: One-vs-rest probing performance for each POS. F1 scores are computed with 5-fold crossvalidation on the GUM training split.

Overall, these results answer RQ1 positively: SAE activations contain information that is predictive of POS categories. At the same time, probing performance alone does not reveal how this information is organized within the SAE latent space.

## 5.2 RQ2: POS-related Latent Groups

RQ2 asks whether the POS information recovered by the probes is localized in restricted regions of the SAE latent space, and at what granularity. We report below the results of the three experiments introduced in Section 4.2.

Feature salience. The coefficient-based salience analysis shows that the binary probes do not rely uniformly on the full SAE latent space. For each POS category, only a subset of latents receives positive weight, indicating that the classifier uses information concentrated in category-specific groups of sparse features. At the same time, these groups are not single-latent representations: POS distinctions are supported by multiple positively contributing latents, consistent with a distributed, but non-uniform, organization of morpho-syntactic information. Figure 3 displays the number of non-zero coefficients for each one-vs-rest classifier. It emergers quite clearly that Open-class POS have generally more non-zero latents, while Closedclass and Other-class have markedly less.<sup>5</sup>

Coverage and compactness. The coverage analysis further quantifies the effective size of these groups. As shown in Figure 4, the cardinality $k _ { c } ^ { 9 5 }$ varies across POS categories. Some tags reach the 95% coverage threshold with a small number of latents, suggesting compact activation patterns. Others require broader latent groups, indicating that the corresponding distinction is more diffuse or depends on a wider set of lexical and contextual cues. This variability shows that localization is category-dependent and cannot be reduced to a one-latent-per-tag mapping.

![](images/1a4ad400e6ab5c04c181067df3a0f764fe940bb86661f9f3f4eaab80ef76507b.jpg)

Figure 3: Non-zero coefficients for each one-vs-rest classifier; results are color coded by POS class (Open, Closed, Other).  
![](images/caf67e1b0b000dac455dc4bdd7ef39b51c06a08d8cb68cb540e721c481e4c484.jpg)  
Figure 4: Number of salient latents required to reach 95% coverage for each POS category. For each tag c, $k _ { c } ^ { 9 5 }$ denotes the smallest number of highest-coefficient latents needed to activate on at least 95% of gold tokens of that category. Lower values indicate compact groups.

Compact-feature classification. Finally, we test whether the selected latent groups are sufficient for multi-class POS prediction. The compact feature set includes 498 latents, with 12% of them being shared between 2+ POS. The classifier trained on it achieves performance comparable to the classifier trained on the full SAE representation (Figure 5). The selected latents then preserve most of the information needed for multi-class POS discrimination, and the salience and coverage analyses recover a compact but effective subset of the latent space.<sup>6</sup>

Overall, these results answer RQ2 by showing that POS information is localized at the level of structured groups of latents. These groups are compact for some categories and broader for others, but they are sufficient to support both category-wise coverage and multi-class classification.

![](images/ef348065a497ffa1b0e322940d66e46539403e2941a5199ab1df3457e611d33f.jpg)

Figure 5: Row-normalized confusion matrix of the multi-class POS classifier trained on the compact fea ture set $L ^ { \star }$ . Each cell reports the percentage of tokens of a gold POS category predicted as each class.
<table><tr><td>Template</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>I see/saw [DET] [NOUN]</td><td>0.207</td><td>0.984</td><td>0.341</td></tr><tr><td>I see/saw [DET] [ADJ] [NOUN]</td><td>0.215</td><td>0.967</td><td>0.351</td></tr><tr><td>I see/saw [DET] [NOUN] [PUNCT]</td><td>0.224</td><td>0.971</td><td>0.365</td></tr><tr><td>I have/had ([DET]) [NOUN]</td><td>0.186</td><td>0.977</td><td>0.313</td></tr><tr><td>I have/had ([DET]) [ADJ] [NOUN]</td><td>0.200</td><td>0.973</td><td>0.332</td></tr><tr><td>I have/had ([DET]) [ADJ] [NOUN] [PUNCT]</td><td>0.212</td><td>0.976</td><td>0.348</td></tr><tr><td>There is/was ([DET]) [NOUN]</td><td>0.163</td><td>0.801</td><td>0.270</td></tr><tr><td>There is/was ([DET]) [ADJ] [NOUN]</td><td>0.178</td><td>0.824</td><td>0.293</td></tr><tr><td>There is/was ([DET]) [ADJ] [NOUN] [PUNCT]</td><td>0.193</td><td>0.846</td><td>0.314</td></tr><tr><td>I [VERB]</td><td>0.162</td><td>0.975</td><td>0.278</td></tr><tr><td>I [VERB] [ADV]</td><td>0.175</td><td>0.970</td><td>0.297</td></tr><tr><td>I [VERB] [ADV] [PUNCT]</td><td>0.191</td><td>0.975</td><td>0.319</td></tr><tr><td>Average</td><td>0.192</td><td>0.937</td><td>0.318</td></tr></table>

Table 1: Pseudo-multilabel classification results on the controlled dataset.

## 5.3 RQ3: Validation on held-out data

We evaluate whether the latent groups identified in Section 4.2 remain stable and systematic beyond the data used to select them, addressing RQ3. We do not train another classifier, but rather test whether the latent groups identified in the discovery setting remain active on unseen instances of the corresponding POS categories. Recall that on held-out data we compute the probability that at least one latent in $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ is active on tokens whose gold label is c For each pair of categories $c , c ^ { \prime } \in C$

On the controlled dataset, which includes only a subset of POS categories, we evaluate the selected latent groups as a pseudo-multilabel classification task. For each token, we encode its gold POS as a binary vector over the considered categories C.

Predictions are obtained by applying the indicator function defined in Section 5.3 to each category $c ^ { \prime } \in C$ . Table 1 reports micro-averaged precision, recall, and F1 for each template. The results show limited precision but very high recall in general, with some template-dependent variation. We also note the presence of a set of latents that we associate with a “first-token” concept and that confound the results. We verified that by including a prefix with a different POS to each sentence, all these activations shift onto this new $\mathrm { P o } { \mathsf { S } } , ^ { \prime }$ indicating that latent groups are present in controlled data.

For more generalizability over the whole POS set, we use the held-out treebank test set. Figure 6 reports a heatmap with the co-activation patterns. By construction, the diagonal entries $M _ { c , c }$ represents the per-category recall. The off-diagonal entries $M _ { c , c ^ { \prime } } \left( c \neq c ^ { \prime } \right)$ measure the rate at which latents selected as salient for $c ^ { \prime }$ nevertheless fire on tokens of a different category c, i.e., a spurious co-activation rate. Low off-diagonal values indicate that the minimal latent groups are disjoint and category-specific; high values suggest overlapping representations across POS categories. Results highlight two main aspects: First, the recall is consistently high, with most values ≥ 0.95, indicating that the latents identified as salient for a given POS remain active on unseen tokens drawn from the same underlying distribution; second, we observe a relatively high variability in off-diagonal scores, indicating overlaps across categories. We further validate the latter observation by computing a Distinctiveness (D) score for each POS, as $\begin{array} { r } { \mathrm { D } ( \boldsymbol { c } ) = \frac { M _ { c , c } } { \sum _ { c ^ { \prime } \in C } M _ { c , c ^ { \prime } } } } \end{array}$ . Intuitively, $\mathrm { D } ( c ) = 1$ indicates that the latents in $L ^ { ( k _ { c } ^ { \star } ) }$ fire exclusively on c tokens; a value of 1/|C| (0.06 in our case) corresponds to the chance baseline under uniform coactivation. We observe a D $\mu = 0 . 2 7 ( \sigma = 0 . 0 7 ) . ^ { 8 }$

Overall, the results on both held-out datasets answer RQ3 affirmatively for stability, while qualifying the systematicity claim: the identified latent groups remain consistently active on unseen tokens of their target category, but are only partially category-specific. More generally, this suggests that, despite the transition to a sparse representation through the SAE, a one-to-one correspondence between linguistic categories and groups of latents holds only to a limited extent.

![](images/4449a00452dd627e0440da60c9db7995a929b432ce203959d8ee9342a44d58bc.jpg)  
Figure 6: Co-activations of selected latent groups on the held-out treebank test set. Diagonal values correspond to recall for each POS; off-diagonal values correspond to false positive rates on other POS categories.

<table><tr><td>Probe</td><td>Accuracy</td><td>Macro F1</td></tr><tr><td>SAE</td><td>0.88</td><td>0.78</td></tr><tr><td>Layer 30</td><td>0.92</td><td>0.84</td></tr><tr><td>Layer 0 (Embed.)</td><td>0.88</td><td>0.78</td></tr></table>

Table 2: Performances using different probes.

## 5.4 Controls and Baselines

The control probe with random labels obtains 0.54 Accuracy/0.42 Macro F1, against 0.88/0.97 respectively for the real probe. The confound is not negligible, but the 34–37 point performance gap still supports the conclusion that POS recoverability is not simply reducible to lexical memorisation, and that lexical identity plays a minor role. This is consistent with evidence provided in Sec. 5.5.

The SAE-based probe has comparable performances with both the raw residual stream probes, both at the embedding layer and at layer 30 (Table 2) using ∼8 times less features. However, note that we do not claim superiority of SAEs as a probing tool. Rather, we claim and show that (i) POS information is recoverable from subsets of SAE latents and that (ii) the SAE’s contribution is decomposition and localisation, which the dense probe cannot provide as easily.

Finally, we see that L<sup>⋆</sup> features are POS relevant. In fact, randomly replacing features from the $L ^ { \star }$ set drastically reduces performances. Table 3 shows the results at 0, 25, 50 and 100% overlap with $L ^ { \star }$ . This further demonstrates that $L ^ { \star }$ latents are POS relevant.

![](images/4645268d9fcfdc92e3232f679f0193d501376e4fd8d0be5bb75366d87f6b259d.jpg)  
Figure 7: Active L⋆ latents per relevant POS in each template variant. Each square is one latent that is active on at least 25% of template’s examples. Color saturation indicates activation percentage (darker = more active).

## 5.5 Linguistic analysis of SAE activations

The results confirm a systematic association between SAE activations and POS information. Overall, four main patterns emerge.

Closed classes are more stable and compact. On the training set (Figure 2) and on the held-out treebank test set (Figure 6), Closed-classes consistently achieve higher Recall and F1-score. Note that these categories also require fewer latents in the coverage analysis (Figure 4 and Appendix B.2, Figure 3). By contrast, X and SYM exhibit the highest error rates in both experiments, likely due to their low support in the training data (Figure 2).

The compactness gradient we observe co-varies with the size and formal variability of each category’s type inventory. A category with a handful of invariant word forms can be covered by a small latent group with no category-level abstraction being involved, since a group of form-specific latents suffices. The controlled data support this reading:

<table><tr><td>Overlap (%)</td><td>Evaluation</td><td>Accuracy</td><td>Macro-F1</td></tr><tr><td rowspan="2">0</td><td>Cross-validation</td><td>0.24</td><td>0.16</td></tr><tr><td>Train/test</td><td>0.23</td><td>0.16</td></tr><tr><td rowspan="2">25</td><td>Cross-validation</td><td>0.49</td><td>0.41</td></tr><tr><td>Train/test</td><td>0.49</td><td>0.41</td></tr><tr><td rowspan="2">50</td><td>Cross-validation</td><td>0.69</td><td>0.60</td></tr><tr><td>Train/test</td><td>0.69</td><td>0.60</td></tr><tr><td rowspan="2">100 (original)</td><td>Cross-validation</td><td>0.87</td><td>0.76</td></tr><tr><td>Train/test</td><td>0.89</td><td>0.81</td></tr></table>

Table 3: Accuracy and Macro-F1 across overlap levels, comparing cross-validation and train/test evaluation.

DET collapses to a single activation value where the determiner is invariably the (Figure 18), and acquires structure only where the a/an versus bareplural alternation introduces formal variation (Figure 7). The compactness ordering should therefore be read primarily as a gradient in lexical variability rather than as direct evidence of graded abstraction.

Open classes show broader and less selective activation patterns. Spurious co-activations are more frequent for open POS classes, where the same lemma or morphologically related forms can serve different functions depending on context. For instance, ADJ tokens are mainly confused with NOUN, ADV, and PROPN, reflecting attributive noun uses, adjective–adverb overlap, and nominal modification patterns. Similarly, ADV shows diffuse co-activation with ADP, NOUN, ADJ, SCONJ, and VERB, suggesting that some adverb-associated latents capture positional or contextual cues rather than adverbial function alone. PROPN and NOUN also co-activate, consistent with their shared nominal distribution, while VERB shows overlap with NOUN in homograph pairs e.g. to drink / the drink.

Some off-diagonal patterns reflect annotation and lexical overlap, as well as syntagmatic properties. The strongest non-target activations are not random. INTJ shows high spurious activation rates across several categories, plausibly due to annotation conventions that assign heterogeneous forms such as like, well, or God to INTJ in pragmatic contexts. Similarly, SCONJ co-activates with ADP and VERB, reflecting lexical overlap between subordinating conjunctions and prepositions in English (e.g., by, after, since) and broader positional regularities. These patterns suggest that the selected latents do not encode purely abstract POS function, but also respond to surface form, lemma sharing and orthographic cues. Moreover, the POS emerging out of the latent space are also defined in terms of their syntactic contexts. For instance, the ADJ latents strongly co-activate with NOUN reflecting the nature of adjectives as nominal modifiers.

Controlled examples confirm additive and category-sensitive activations. Turning to the controlled dataset, Figure 7 confirms that, as new tokens are introduced into the sentence, their associated latents activate consistently within the latent groups characteristic of their PoS, suggesting that PoS-specific latent activations are largely additive across tokens. For example, adding loyal to There is a dog triggers the activation of more latents associated with ADJ. We observe that some latents of related POSes are already present even without the corresponding words (e.g., ADJ for NOUN and ADV for VERB), but the number of active ones corresponding to that category consistently grows when the word is included in the template.

These findings show that POS-related latent groups are stable and systematic, but not categoryexclusive. Closed classes tend to yield compact and selective representations, whereas open classes involve broader latent groups that also capture lexical, morphological, and contextual regularities.

## 6 Conclusion

We used POS categories as a controlled testbed to study how morpho-syntactic information is organized in SAEs. We show that POS distinctions are consistently recoverable from sparse activations, and that each category is supported by a compact group of sparse features, whose size varies with the linguistic nature of the category. For Openclass POS, these groups are more diffuse, with a larger number of active latents, while Closedclass ones have fewer active latents. A small union of these category-specific latents preserves strong multi-class classification performance, and the selected groups remain stable on held-out treebank data and largely additive on controlled examples.

Our results reveal that POS are internally represented in LLMs as emerging sets of localizable but distributed features in latent SAE space. Moreover, analyses suggest that interpretability claims at the latent level should be evaluated against theoretically grounded category inventories rather than top-activating examples alone. At the same time, cross-category co-activations show that the identified latents partly track lexical, positional, and annotation-driven regularities, motivating future work on richer linguistic levels and typologically diverse languages.

## Limitations

Our study focuses on POS categories as a controlled morpho-syntactic test case. While this choice allows us to rely on exhaustive and independently annotated labels, POS tags capture only one level of linguistic abstraction. Future work should extend the analysis to finer-grained morphological features, dependency relations, semantic roles, and discourse-level phenomena, where latent organization may be recoverable in different ways.

We also analyze a single base language model, LLAMA-3-8B, and one publicly available SAE. The observed patterns may depend on the underlying model, the layer from which activations are extracted, the SAE training procedure, and the sparsity regime. Comparing multiple models, layers, and SAE variants would be necessary to assess how general these findings are.

Our localization procedure relies on linear probing coefficients as a salience signal. Although the use of ℓ<sub>1</sub>-regularization encourages sparse and interpretable solutions, probe coefficients should not be interpreted as direct causal evidence. The identified latents are predictive of POS categories, but further causal interventions would be needed to establish whether they are used by the model for morpho-syntactic processing.

Finally, our controlled dataset is intentionally small and targets a restricted set of constructions, mainly involving nouns and verbs. It is useful for validating whether selected latent groups remain active in simple and independently constructed contexts, but it does not cover the full syntactic and lexical variability of English. Broader controlled datasets would allow a more systematic evaluation of how lexical ambiguity, word order, morphology, and sentence complexity affect POS-related latent activations.

## Acknowledgments

This work has been supported by i.) the PNRR MUR project PE0000013-FAIR (Spoke 1), funded by the European Commission under the NextGeneration EU programme; ii.) the EU EIC project EMERGE (Grant No. 101070918); and iii.) The PNRR MUR project FAIR TT\_02 “Innovare la sorveglianza automatizzata delle infezioni del sito chirurgico tramite modelli di elaborazione del linguaggio naturale”.

## References

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, and 6 others. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread.

Alexis Conneau, German Kruszewski, Guillaume Lample, Loïc Barrault, and Marco Baroni. 2018. What you can cram into a single \$&!#\* vector: Probing sentence embeddings for linguistic properties. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2126–2136, Melbourne, Australia. Association for Computational Linguistics.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. 2023. Sparse autoencoders find highly interpretable features in language models. Preprint, arXiv:2309.08600.

Tom Dupre la Tour and Dan Mossing. 2025. Debugging misaligned completions with sparse-autoencoder latent attribution. OpenAI Alignment Research Blog.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. 2022. Toy models of superposition. Transformer Circuits Thread.

Joshua Engels, Eric J Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. 2025. Not all language model features are one-dimensionally linear. In The Thirteenth International Conference on Learning Representations.

Kit Fraser-Taliente, Subhash Kantamneni, Euan Ong, Dan Mossing, Christina Lu, Paul C. Bogdan, Emmanuel Ameisen, James Chen, Dzmitry Kishylau, Adam Pearce, Julius Tarng, Alex Wu, Jeff Wu, Yang Zhang, Daniel M. Ziegler, Evan Hubinger, Joshua Batson, Jack Lindsey, Samuel Zimmerman, and Samuel Marks. 2026. Natural language autoencoders produce unsupervised explanations of llm activations. Transformer Circuits Thread.

Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. 2025. Scaling and evaluating sparse autoencoders. In The Thirteenth International Conference on Learning Representations.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,

Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Zhengfu He, Wentao Shu, Xuyang Ge, Lingjie Chen, Junxuan Wang, Yunhua Zhou, Frances Liu, Qipeng Guo, Xuanjing Huang, Zuxuan Wu, and 1 others. 2024. Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders. arXiv preprint arXiv:2410.20526.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2733–2743, Hong Kong, China. Association for Computational Linguistics.

John Hewitt and Christopher D. Manning. 2019. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4129–4138, Minneapolis, Minnesota. Association for Computational Linguistics.

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda. 2025. Are sparse autoencoders useful? a case study in sparse probing. In Forty-second International Conference on Machine Learning.

Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Isaac Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum Stuart McDougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel Marks, and Neel Nanda. 2025. SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. In Forty-second International Conference on Machine Learning.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, Janos Kramar, Anca Dragan, Rohin Shah, and Neel Nanda. 2024. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 278–300, Miami, Florida, US. Association for Computational Linguistics.

Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. 2025. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In The Thirteenth International Conference on Learning Representations.

Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, and Ryan Cotterell. 2020. Information-theoretic probing for linguistic structure. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics,

pages 4609–4622, Online. Association for Computational Linguistics.

Anna Rogers, Olga Kovaleva, and Anna Rumshisky. 2020. A primer in BERTology: What we know about how BERT works. Transactions ofthe Association for Computational Linguistics, 8:842–866.

Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, Stella Biderman, Adria Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi Schoots, Joseph Miller, Eric J. Michaud, and 10 others. 2025. Open problems in mechanistic interpretability. Preprint, arXiv:2501.16496.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, and 3 others. 2024. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019a. BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 4593– 4601, Florence, Italy. Association for Computational Linguistics.

Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman, Dipanjan Das, and Ellie Pavlick. 2019b. What do you learn from context? probing for sentence structure in contextualized word representations. In International Conference on Learning Representations.

Amir Zeldes. 2017. The GUM corpus: Creating multilayer resources in the classroom. Language Resources and Evaluation, 51(3):581–612.

## Appendix

## A Further Details on Implementation

Here we provide some further details on several of the implementation choices, including the rationale for selecting the target layer and experimental setup.

## A.1 Layer Selection

We extract hidden state activations of the LLM from the residual stream of the last layer before the output. The residual stream hookpoint is set after the MLP layer. This means that the activations we take are very close to the output.

We acknowledge that the choice of the layer might drastically affect the results. Thus, to verify the stability of our findings, we replicate some of the experiments across a sample of layers. As for RQ1 (Recoverability of POS information, see Section 4.1), we extract hidden representations from the residual stream after the MLP of layer $x ,$ and encode it with the corresponding SAE. We report in Figure 8 the F1-Scores of the binary probing classifiers across all layers. The results show that performances are generally stable across layers, with minor variations. Other class POSes are the ones showing higher variability, except PUNCT. We also note a systematic slight dip in performances in the last few layers across all POS classes. This may be attributable to the vicinity to the unembed layer.

As for RQ2 (Organization in Latent Space, see Section 4.2 and RQ3 (Validation on held-out data, see Section 4.3), we report result for Salience (latent activations count), Coverage (, Compactfeature classification, and distinctiveness for Layers 2 and 15 (beginning and middle of the model stack). First, the number of non zero activations per POS increases modestly across layers for most POS (e.g., NOUN: 9024→9935→10043; ADP: 4955→5742→6511), but stays within the same order of magnitude: open-class POS in the low-thousands-to-10k range, closed-class in the hundreds-to-low-thousands, Other in the hundreds. Pooled mean rises only 19% from layer 2 to layer 30 (3105→3697). This reflects gradual densification with depth, not a qualitative reorganization. Second, $L ^ { * }$ remains stable stable across depth: 530 (L2) → 501 (L15) → 498 (L30). Coverage efficiency $( L ^ { \star }$ over the sum of $k _ { c } ^ { \star } )$ improving slightly (84%→81%→87%). Per-category values do redistribute a bit. For example, ADP falls (71→35→16), SCONJ has a non monotonic trend (7→51→17), NOUN rises (12→38→34), PROPN rises (22→47→60). However, all values remain in the same single-to-double-digit-tens range at every layer, with no category jumping an order of magnitude. Third, compact-feature classification Macro F1 increases only slightly (0.77→0.79→0.81), and accuracy stays mostly flat (0.87→0.88→0.87). This may be an indication that later layers encode more linearly separable POS distinctions for minority/harder classes. Finally, distinctiveness remains stable across layers. We report mean and standard deviation: 0.15±0.03 $\left( \mathrm { L 2 } \right) \to 0 . 3 2 \pm 0 . 0 9 \left( \mathrm { L 1 5 } \right) \to 0 . 2 7 \pm 0 . 0 7 \left( \mathrm { L 3 0 } \right)$

These results suggest that the layer choice do not drastically affect the degree to which POSes are encoded into the SAE latents, and that the findings of the paper should hold across the whole model. Results on additional layers at the beginning and middle of the model stack show slight variation, but are also a clear indication that the conclusions from the paper hold also for layers other than the one analyzed. Individual POS categories reshuffle which latents they rely on across depth, but the total representational budget and per-category magnitudes remain stable. This supports our claim of layer-consistent POS structure with depth-wise redistribution rather than qualitative change.

## A.2 Experimental setup

All experiments were conducted using a GPU node equipped with 8 A100 80GB GPUs. Given the model sizes, only one GPU was sufficient to extract hidden representations from a layer and encode it with its corresponding SAE. The process takes roughly 0.16 GPU hours per layer. The probing classifiers were implemented using SciKit-Learn. The library does not leverage GPUs, but allows parallelization across CPU cores. A single 5-fold cross-validation training/test run on the GUM Treebank training set requires roughly 4 hours using all CPU cores available.

## A.3 Artifacts and Intended Use

We use three existing artifacts, employed consistently with their intended use and license.

LLaMA-3-8B (Grattafiori et al., 2024) is released by Meta under the Llama 3 Community License, which permits research and academic use. We use the model exclusively for interpretability analysis, extracting hidden representations without fine-tuning or redistribution of model weights.

![](images/71097991859bd2044cebdf2063daceedbe32b4b8c82a829de9d2cec6b1c9fd67.jpg)

![](images/85cbf34fc08596978373cad608a7ddae36f554591e8d6f6374baa6af38b0e2e3.jpg)

![](images/4cbf08845b5d7a6ea173aa5193a8af1f7e5b01d4cde1daab9e8b3fa43ff1a907.jpg)  
Figure 8: Per-POS one-vs-rest classifier performances across layers.

EleutherAI SAE The checkpoint we use from EleutherAI/sae-llama-3-8b-32x is a pretrained Sparse Autoencoder released on Hugging-Face by EleutherAI for interpretability research on LLaMA-3-8B, Our use directly aligns with its intended purpose.

The GUM treebank (Zeldes, 2017) is distributed under Creative Commons licenses (primarily CC BY 4.0, with some subcorpora under more restrictive terms) for research and educational use in computational linguistics. We use the treebank’s text and Universal Dependencies annotations for probing and evaluation, which is consistent with its intended use.

Artifacts produced. We release<sup>9</sup> the token-level SAE activations aligned to UD POS tags, the controlled evaluation dataset (180 items), and the code for the probing and salience pipeline. These artifacts are released for research use only, consistent with the access conditions of the underlying resources. To respect the per-source licensing of GUM, activation files are keyed to token indices in the original GUM release rather than redistributing the source text.

## B Additional Results

Here we present additional results from our experiments across our three RQs.

## B.1 RQ1: Recoverability of POS information

Figures 9, 10, and 11 show, for each POS class, the top 10 latents ranked by coefficient in the classifier.

![](images/6c071068d18a2d7ef29d0064d463b05f60f526bdbfa8b943c5f20cd31b89b5fa.jpg)  
Figure 9: Per-POS one-vs-rest classifier top coefficients. OPEN class POSes.

Closed-class categories show the most peaked coefficient distributions, with a single latent dominating in DET (75751, ≈ 5.5), CCONJ (34665, ≈ 4.3) and PRON (6631, ≈ 2.7). Open-class categories are flatter: only VERB shows a clearly dominant feature (94414, ≈ 2.2); ADJ (0.80), PROPN (0.77) and ADV (0.73) have no single salient latent. PUNCT is the best-classified category overall, driven by latent 117946 (≈ 5.5). Moreover, Several latents recur across closed-class categories (e.g., 72975 in AUX/CCONJ/PART; 116300 in CCONJ/NUM/PRON; 86665 in DET/PRON/PUNCT), which may indicate polyfunctional latents.

## B.2 RQ2: POS-related Latent Groups

Here, we present results related to RQ2.

Feature Salience. As for the feature salience, Table 4 shows the number of non-zero latents for each one-vs-rest classifier. See also Figure 3 in the main paper. From the Table and Figure, it emerges quite clearly that Open-class POS have generally more non-zero latents, while Closed-class and Otherclass have markedly less. The two main exceptions are INTJ for the Open-class, which has very few, and ADP for the Closed-class, which is more akin to open ones. For ADP, this may be attributable to the fact that words that function as adpositions may also be used to mark adverbial clauses. As for INTJ, they typically express an emotional reaction and are not syntactically related to other accompanying expressions; moreover, their support in the GUM treebank is very low. Both factors could play a role in poor classification performances (See Section 5.2 and below) and limited number of latents active as features during classification. Table 5 reports mean and standard deviation counts for non-zero coefficients. Again, we observe that Other class have the least number of non-zero coefficients on average, and Open class has the highest number and highest variability.

![](images/330aeb46c9580df53b9928a82a34a5274c7bc5d08dcb472c0b0cd871910ba17d.jpg)  
Figure 10: Per-POS one-vs-rest classifier top coefficients. CLOSED class POSes.

![](images/c5a0cf344dac36d287b5247bfef5c08420a13c9f36b340f66bc57fc04303db44.jpg)  
Figure 11: Per-POS one-vs-rest classifier top coefficients. OTHER class POSes.

<table><tr><td>POS</td><td>POS Type</td><td>Non-zero Latents</td><td>Frac. of Latent Space</td></tr><tr><td>NOUN</td><td>open</td><td>10043</td><td>0.0766</td></tr><tr><td>VERB</td><td>open</td><td>7404</td><td>0.0565</td></tr><tr><td>ADJ</td><td>open</td><td>8877</td><td>0.0677</td></tr><tr><td>PROPN</td><td>open</td><td>5252</td><td>0.0401</td></tr><tr><td>ADV</td><td>open</td><td>5218</td><td>0.0398</td></tr><tr><td>INTJ</td><td>open</td><td>1010</td><td>0.0077</td></tr><tr><td>PRON</td><td>closed</td><td>2623</td><td>0.0200</td></tr><tr><td>CCONJ</td><td>closed</td><td>645</td><td>0.0049</td></tr><tr><td>DET</td><td>closed</td><td>2647</td><td>0.0202</td></tr><tr><td>AUX</td><td>closed</td><td>2651</td><td>0.0202</td></tr><tr><td>ADP</td><td>closed</td><td>6511</td><td>0.0497</td></tr><tr><td>NUM</td><td>closed</td><td>1001</td><td>0.0076</td></tr><tr><td>PART</td><td>closed</td><td>1694</td><td>0.0129</td></tr><tr><td>SCONJ</td><td>closed</td><td>3762</td><td>0.0287</td></tr><tr><td>PUNCT</td><td>other</td><td>2129</td><td>0.0162</td></tr><tr><td>SYM</td><td>other</td><td>460</td><td>0.0035</td></tr><tr><td>X</td><td>other</td><td>916</td><td>0.0070</td></tr></table>

Table 4: Non-zero latents and fraction of latent space by POS tag.

<table><tr><td>PoS Class</td><td>mean± std.</td></tr><tr><td>Closed</td><td> $3 2 8 1 . 7 4 \pm 1 8 9 5 . 6 5$ </td></tr><tr><td>Open</td><td> $7 9 2 6 . 5 2 \pm 2 1 5 3 . 6 9$ </td></tr><tr><td>Other</td><td> $2 0 9 4 . 4 2 \pm 2 2 1 . 9 1$ </td></tr></table>

Table 5: Number of latents with non-zero coefficients for classification for each one-vs-rest classifier, aggregated by POS Group. Classifiers for OPEN-class POS tags have the highest number of non-zero coefficients.

Coverage and Compactness. In Table 6 we report, for each POS, the ratio between the coverage for $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ and the number of latents with non-zero coefficients for $c ^ { \prime } .$ Intuitively, this represent the proportion of latents needed to reach coverage for a POS with respect to all salient latents for the POS.

Compact-feature classification. Table 7 reports per-POS performances of the multiclass classifier trained with the compact feature set L<sup>∗</sup>.

Figure 12 shows the heatmap of coefficient association to each POS class in the multilabel classification experiment. Coefficient are sorted for importance across POS classes.

Figure 13 provides a sensitivity analysis of the performances of the classifier with respect to values of C and τ. We report mean Macro-F1 score and Accuracy in the cross validation setting. We also provide standard deviation in the form of error bars. From the plot, it clearly emerges that the classification results are not particularly sensitive neither to the τ threshold nor the C value. As for the τ , performances increase monotonically, but with a difference of ∼5 points using 3x less features. As for the C values, performances remain almost identical, and the standard deviation is near zero, indicating no meaningful differences.

![](images/cab8b3686fa60aaf37095f2f8e83a5f563a75e64c5c23c10040b29ebd5cf18c8.jpg)

Figure 12: Coefficients importance heatmap for each POS in the mutliclass classifier trained with 5-fold crossvalidation on the GUM Training set.  
![](images/e57302cc864105eace467b1983c65c0d81365e33ecd8a27bbf2fd8838e51d3fa.jpg)  
Figure 13: Parameter sweep for C and τ for the compact feature classification task.

## B.3 RQ3: Validation on held-out data

Density of Activations per POS. In Figure 14 we report a KDE plot representing density of number of activations from $\dot { L } ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ over POS with tag $c ^ { \prime }$ in the Treebank test set, for all $c ^ { \prime } \in C$ . The Figure shows that in most cases at least 1 to 4% of latents in $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ fire on all tokens of $c ^ { \prime } .$

We further provide indications that most tokens associated with category $c ^ { \prime }$ fire latents in $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ Over the whole treebank test set, we count the number of cases in which no latents in $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ fire over a token of category $c ^ { \prime } .$ Table 8 shows the fraction of tokens with no $L ^ { ( k _ { c ^ { \prime } } ^ { \star } ) }$ for each $c ^ { \prime } \in C$

Distinctiveness. Table 8 reports per-c results of $M _ { c , c }$ and distinctiveness $D ( c )$ . We observe that

INTJ and X have the least distinctive activations. In this case we do not see a clear POS-class based trend with regards to distinctiveness.

Per POS Activation distribution in the controlled dataset. Figures 15 through 18 report the KDE distributions of active-latent percentages per POS tag for the remaining three construction types in the controlled dataset. In each figure, blue, red, and green correspond to sentences of increasing syntactic complexity: the minimal construction (e.g., [PRON] + [AUX] + [DET] + [NOUN]), the extension with an adjective [ADJ] or an [ADV] for the verb template, and the further addition of punctuation [PUNCT], respectively; dashed vertical lines indicate cases where only a single value is available for a given POS tag and sentence type.

<table><tr><td>Class</td><td>POS</td><td>Coverage over Salience (%)</td></tr><tr><td rowspan="6">Open</td><td>ADJ</td><td>0.91</td></tr><tr><td>ADV</td><td>1.38</td></tr><tr><td>INTJ</td><td>3.76</td></tr><tr><td>NOUN</td><td>0.34</td></tr><tr><td>PROPN</td><td>1.14</td></tr><tr><td>VERB</td><td>0.70</td></tr><tr><td rowspan="8">Closed</td><td></td><td>0.25</td></tr><tr><td>ADP AUX</td><td>1.06</td></tr><tr><td>CCONJ</td><td>0.62</td></tr><tr><td>DET</td><td>0.42</td></tr><tr><td></td><td>1.20</td></tr><tr><td>NUM</td><td>0.47</td></tr><tr><td>PART PRON</td><td>0.76</td></tr><tr><td>SCONJ</td><td>0.45</td></tr><tr><td rowspan="3">Other</td><td>PUNCT</td><td>0.33</td></tr><tr><td>SYM</td><td>5.65</td></tr><tr><td>X</td><td>8.95</td></tr></table>

Table 6: Coverage over number of salient features per UPOS, grouped by class.

<table><tr><td></td><td>precision</td><td>recall</td><td>f1-score</td><td>support</td></tr><tr><td>ADJ</td><td>0.85</td><td>0.84</td><td>0.85</td><td>11489</td></tr><tr><td>ADP</td><td>0.93</td><td>0.88</td><td>0.91</td><td>16655</td></tr><tr><td>ADV</td><td>0.84</td><td>0.79</td><td>0.81</td><td>8556</td></tr><tr><td>AUX</td><td>0.93</td><td>0.91</td><td>0.92</td><td>9682</td></tr><tr><td>CCONJ</td><td>0.96</td><td>0.97</td><td>0.96</td><td>5854</td></tr><tr><td>DET</td><td>0.95</td><td>0.93</td><td>0.94</td><td>14307</td></tr><tr><td>INTJ</td><td>0.36</td><td>0.89</td><td>0.52</td><td>1860</td></tr><tr><td>NOUN</td><td>0.92</td><td>0.87</td><td>0.90</td><td>29289</td></tr><tr><td>NUM</td><td>0.88</td><td>0.93</td><td>0.90</td><td>3368</td></tr><tr><td>PART</td><td>0.82</td><td>0.95</td><td>0.88</td><td>4314</td></tr><tr><td>PRON</td><td>0.95</td><td>0.93</td><td>0.94</td><td>15197</td></tr><tr><td>PROPN</td><td>0.83</td><td>0.83</td><td>0.83</td><td>10144</td></tr><tr><td>PUNCT</td><td>0.99</td><td>0.94</td><td>0.96</td><td>24563</td></tr><tr><td>SCONJ</td><td>0.66</td><td>0.86</td><td>0.75</td><td>2878</td></tr><tr><td>SYM</td><td>0.33</td><td>0.96</td><td>0.49</td><td>281</td></tr><tr><td>VERB</td><td>0.94</td><td>0.86</td><td>0.90</td><td>18642</td></tr><tr><td>X</td><td>0.14</td><td>0.88</td><td>0.24</td><td>331</td></tr><tr><td>accuracy</td><td></td><td></td><td>0.89</td><td>177410</td></tr><tr><td>macro avg</td><td>0.78</td><td>0.89</td><td>0.81</td><td>177410</td></tr><tr><td>weighted avg</td><td>0.91</td><td>0.89</td><td>0.90</td><td>177410</td></tr></table>

Table 7: Multiclass classifier performances.

The distributions for most POS tags are highly stable across sentence types, with the three curves largely overlapping. A consistent exception is the NOUN tag: in both transitive constructions (I have/had and I see/saw), the red and green curves, corresponding to sentences containing an additional adjective, show a slight shift in the activation distribution relative to the blue curve. This suggests that the presence of an adjacent adjective marginally affects noun-associated latent activations, consistent with co-activation patterns discussed in Section 5.5; similarly, AUX in Figure 16 displays two distinct peaks, reflecting the alternation between the present have and past had forms across sentences, confirming latents’ sensitivity to surface form in POS classes with relatively low token variability. By contrast, the VERB tag in Figure 17 shows only a minor shift across sentence types, suggesting that verb-associated latents are less sensitive to the surrounding syntactic context than noun-associated ones.

![](images/4dfbd7a9ef19ea9dfa98e33a1633d9a1526f8ba2bd216c237978cf91f90e378d.jpg)  
Figure 14: Density plot: number of activations from $L ^ { ( \overline { { { k } } } _ { c ^ { \prime } } ^ { \star } ) }$ over POS with tag $c ^ { \prime }$ in the Treebank test set, for all $c ^ { \prime } \in C .$

<table><tr><td>Class</td><td>POS</td><td>Zero-act.  $\%$ </td><td> $M _ { c c }$ </td><td> $D ( c )$ </td></tr><tr><td rowspan="5">Open</td><td>ADJ</td><td>0.056 0.040</td><td>0.944 0.960</td><td rowspan="5">0.305 0.217 0.093 0.336</td></tr><tr><td>ADV INTJ</td><td>0.076</td><td>0.924 0.949</td></tr><tr><td>NOUN</td><td>0.051</td><td></td></tr><tr><td>PROPN</td><td>0.027</td><td>0.973 0.273 0.952</td></tr><tr><td>VERB ADP</td><td>0.048 0.019</td><td>0.303 0.350 0.301</td></tr><tr><td rowspan="5"></td><td>AUX CCONJ</td><td>0.048 0.024</td><td>0.981 0.952 0.976 0.977</td><td>0.261</td></tr><tr><td>Closed NUM</td><td>DET</td><td>0.023 0.006</td><td>0.350 0.345</td></tr><tr><td>PART</td><td>0.007</td><td>0.994 0.993</td><td>0.317</td></tr><tr><td>PRON</td><td>0.030</td><td>0.970</td><td>0.225</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3"></td><td></td><td></td><td></td><td></td></tr><tr><td>SCONJ</td><td>0.024</td><td>0.976</td><td>0.224</td></tr><tr><td>PUNCT</td><td>0.051</td><td>0.949</td><td></td></tr><tr><td rowspan="3">Other</td><td></td><td></td><td></td><td>0.333</td></tr><tr><td>SYM</td><td>0.111</td><td>0.889</td><td>0.299</td></tr><tr><td>X</td><td>0.273</td><td>0.727</td><td>0.122</td></tr></table>

Table 8: Zero-activation fraction, $M _ { c c } ,$ and distinctiveness $D ( c )$ per UPOS category.

## B.3.1 First-token activations in the controlled dataset

We report an example of first-token activations confounds on the controlled dataset. All tested cases show the same behavior, but we report only one example for brevity. We do the following: we prepend “1.” to all templates, e.g., “1. I saw the cute cat”, recompute activations, and compare prevs-post addition of the NUM+PUNCT. In Figure 19 we report the results for the “I see/saw a [ADJ] [NOUN]”. We observe that all latents that fire on PRON in the original sentence (the first token “I”) shift to the first token NUM.

![](images/53bf2184115e54e8014efac363bbdda7c53f44a6daf56a7c3ba21b5f894682c3.jpg)  
Figure 15: KDE of active-latent percentages per POS tag in the I + [VERB] templates.

![](images/09a52c38533cc9a0905f7a501fd432de93258f2190d8af269cb849fd32a04617.jpg)  
Figure 16: KDE of active-latent percentages per POS tag in the I have/had templates.

![](images/bedb8d51aa6481a96a88890124f4e049e547f928dd2d0fb57283ba82f80c63d0.jpg)  
Figure 17: KDE of active-latent percentages per POS tag in the I + [VERB] templates.

![](images/ec975dbb62b5f4305e8d1fba1a683eb9a9e17f8d4215e80cdc643004cfb4569b.jpg)  
Figure 18: KDE of active-latent percentages per POS tag in the There is/are templates.

![](images/a3f1639c153a5bd0f9bc965b35224d699838c7ab738fe605c26225227a6b9f31.jpg)  
Figure 19: Demonstration of the behavior on first-token activations. We show that several activations shift from PRON in the top sentence (the first token “I”) to the first token NUM in the bottom sentence.