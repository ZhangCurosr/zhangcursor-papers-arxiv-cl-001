Required labeled data vs. F1 score

# Label-Efficient Document-Grounded Factuality Detection via Cross-Paraphrastic Invariance Learning

Chao Chen, Shanshan Lin, Dongsheng Hong, Sibo Ju, Wenjie Yang, Sihong Xie, Senior Member, IEEE, and Xiangwen Liao

Abstract—Large language models (LLMs) can generate fluent claims that are unsupported by a given source document, motivating accurate and efficient document-grounded factuality detection. Existing LLM-based evaluators often require repeated inference and incur high computational cost, whereas compact factuality detectors are efficient at inference time but commonly rely on large labeled or synthetically generated corpora. We propose CPIL (Cross-Paraphrastic Invariance Learning), a twostage Siamese framework that converts a small labeled set into relational supervision without acquiring additional factuality labels. CPIL constructs positive pairs from distinct label-preserving paraphrastic views of the same document-claim instance and negative pairs from opposite-label examples associated with the same source document. It further introduces quantile-aware semi-hard negative mining, which derives a boundary from the positivepair similarity distribution and balances informative negatives with broader coverage. In the two-stage training procedure, Stage 1 learns an encoder from these task-aware pairs, and Stage 2 jointly fine-tunes the encoder and a linear classification head for binary factuality prediction. At inference time, CPIL requires no paraphrase generation, pair construction, or Siamese comparison and uses only one encoder-classifier forward pass. On the 11-task LLM-AggreFact benchmark, CPIL-2% and CPIL-3% achieve macro-F1 scores of 81.04 and 81.60 while using only 2-3% of the FactCG-released labeled pool. Both settings outperform all compared compact detectors and two vanilla LLM references. Ablations and sensitivity analyses support the contributions of task-aware pair construction, dynamic negative mining, and multi-pivot paraphrasing.

Index Terms—contrastive learning, factuality detection, labelefficient learning, Siamese learning

## I. INTRODUCTION

Large language models (LLMs) can produce fluent and coherent text, but fluency does not guarantee faithfulness to the available evidence. A common failure mode is hallucination, where generated claims are inconsistent with or unsupported by a provided source document [1], [2]. Detecting such unsupported claims is therefore important for building trustworthy and reliable generation systems [3].

Existing factuality detectors often trade off inference cost against supervision cost [1]. LLM-as-evaluator methods, including SelfCheckGPT [4], SelfCheckAgent [5], and G-

![](images/8b62d750a9344240c33339edb6e7d40edb66a92637189aad442fd70f3793379f.jpg)  
Fig. 1. Required labeled data size vs. model performance in macro-F1. Colors and markers indicate whether additional human or LLM-based labeling is required, respectively.

Eval [6], ask powerful LLMs to judge or verify generated content. These methods are flexible, but evaluation may involve repeated prompting, long-context reasoning, or multistep verification, leading to substantial inference cost [7]. Compact factuality detectors are more efficient at deployment because they directly predict whether a claim is supported by a document. Representative methods include QAFactEval [8], which uses question answering; AlignScore [9], which learns alignment scores from multiple supervision sources; MiniCheck [7], which trains on LLM-labeled document-claim pairs; and FactCG [10], which augments training data with graph-generated multi-hop claims. As summarized in Fig. 1, however, many competitive compact detectors still rely on substantial human-labeled or synthetically generated data [7], [9], [10]. The challenge is therefore not only to build an accurate detector, but also to reduce its factuality-label requirement while preserving efficient inference.

We address this challenge with CPIL (Cross-Paraphrastic Invariance Learning), a two-stage Siamese learning framework that converts a small labeled subset into task-aware relational supervision. The key motivation is that standard instance-wise classification extracts one supervision target from each labeled document-claim example, whereas the same labels can induce structured constraints over multiple views and comparisons. CPIL converts limited instance-level supervision into relational constraints through three mechanisms.

First, for positive supervision, CPIL generates backtranslated views of each document-claim instance and explicitly aligns different views of the same example. Unlike appendonly augmentation, this cross-view constraint encourages invariance to assumed label-preserving lexical and syntactic variation. Second, for negative supervision, CPIL contrasts opposite-label examples associated with the same source document. These same-document negatives reduce direct topical and domain shortcuts and encourage the encoder to model finer differences in the document-claim support relation. Third, CPIL introduces quantile-aware semi-hard negative mining to adapt negative difficulty as the representation space evolves. It derives a boundary from the current positive-pair similarity distribution, filters overly positive-like candidates, and combines informative semi-hard negatives with broader coverage from the remaining safe pool.

TABLE I  
MAIN EXTENSIONS OVER THE CONFERENCE PAPER [11].
<table><tr><td>Aspect</td><td>Conference paper</td><td>Journal manuscript</td></tr><tr><td>Method</td><td>Two-stage Siamese learning with static cross-paraphrase positives and same-document negatives.</td><td>Adds quantile-aware semi-hard negative mining with a positive-derived boundary and an exploitation-coverage strategy.</td></tr><tr><td>Formulation</td><td>Concise formulation of pair construction and two-stage training.</td><td>Expanded formulation with separate offline and dynamic training algorithms and complexity analysis.</td></tr><tr><td>Evaluation</td><td>Conference-scale baseline comparison and ablations.</td><td>Stronger baselines and broader ablations of augmentation, pair construction, and dynamic mining.</td></tr><tr><td>Analysis</td><td>Limited low-budget study.</td><td>Additional studies of label budgets, mining ratios, pivot languages, and multi-seed analyses.</td></tr></table>

The detector is optimized in two stages. Stage 1 pretrains a weight-tied encoder with a contrastive objective over crossparaphrase positives and dynamically selected same-document negatives. Stage 2 attaches a linear classification head and jointly fine-tunes the encoder and classifier on the same low-budget labeled subset. Paraphrase generation, pair construction, and dynamic mining are restricted to training. At inference time, CPIL evaluates each document-claim pair with a single encoder-classifier forward pass.

On the 11-task LLM-AggreFact benchmark [7], CPIL achieves the highest macro-F1 among the compared compact factuality detectors, and also exceeds three vanilla LLM references. CPIL-2% and CPIL-3% obtain macro-F1 scores of 81.04 and 81.60, respectively, while using only 2–3% of the FactCG-released labeled pool. The results are supported by ablation studies of the pairwise stage, positive and negative pair definitions, and dynamic mining, as well as sensitivity analyses of the label budget, semi-hard negative proportion, pivot-language selection.

The main contributions are summarized as follows:

• We introduce a factuality-specific relational supervision strategy that constructs cross-paraphrase positives and same-document, opposite-label negatives.

• We propose a quantile-aware semi-hard negative mining strategy that derives a positive-informed safety boundary and balances high-similarity negative selection with broader coverage.

• We instantiate these mechanisms into CPIL, a two-stage Siamese framework that requires neither additional factuality labels nor pairwise inference.

• CPIL demonstrates strong low-budget macro-F1 on the 11-task LLM-AggreFact benchmark.

Journal-version extension. This article substantially extends the conference paper [11] through a new dynamic negative-mining method, expanded method specification, and broader controlled evaluation. Table I summarizes the main differences between the two versions.

## II. RELATED WORK

## A. Document-Grounded Factuality Evaluation

a) Compact factuality detectors: Document-grounded factuality detection evaluates whether a claim is supported by a given evidence document. Prior factual consistency methods for summarization often formulate this problem as natural language inference (NLI) or question answering (QA). FactCC trains a weakly supervised factual consistency classifier using synthetically perturbed source-summary pairs [12]; SummaC aggregates sentence-level NLI signals for summary consistency [13]; and QuestEval and QAFactEval assess consistency through question generation and question answering [8], [14]. More recent compact detectors target broader document-grounded generation settings. AlignScore learns a unified alignment function from multiple supervision sources [9]; MiniCheck trains efficient fact-checkers from synthetically generated document-claim examples [7]; FactCG strengthens factuality checkers with graph-based multi-hop synthetic claims [10]; and CCHD studies constrained paraphrase consistency for factuality detection [15]. Besides, complementary generation-side work has incorporated structured factual knowledge directly into generation [16]. However, many competitive compact detectors rely on substantial annotated, synthetic, or model-generated supervision. Among these methods, CCHD is the closest to CPIL because both use paraphrastic consistency. The two methods, however, impose this consistency at different stages: CCHD imposes consistency constraints during classification, whereas CPIL first converts paraphrastic views and same-document label contrasts into representation-level pairwise supervision.

b) LLM-based evaluators: LLM-based evaluators constitute another major line of factuality assessment. SelfCheck-GPT detects hallucinations from inconsistencies among sampled model responses [4]; G-Eval uses instruction-following LLMs for generation evaluation [6]; FacTool augments factuality assessment with external tools across multiple tasks and domains [17]; and SAFE evaluates long-form factuality by decomposing responses into individual facts and verifying them through search-augmented reasoning [18]. AFEV iteratively decomposes complex claims into atomic facts and combines evidence retrieval with adaptive reasoning. This inferencetime decomposition is complementary to CPIL, which learns from training-time relations and predicts support for a supplied document-claim pair [19]. Beyond post-hoc evaluation, recent work [20] mitigates hallucination during generation with latent prompt learning. Such evaluators can support flexible naturallanguage reasoning, but their inference procedures may depend on repeated model calls, external retrieval or tools, model access, and prompt design. In contrast, CPIL preserves a single compact encoder-classifier pass at inference time.

c) Evaluation benchmarks: Evaluation resources have also expanded to broad document-grounded settings. TRUE consolidates multiple factual consistency datasets across summarization and dialogue settings [21]; LLM-AggreFact aggregates diverse datasets for document-grounded hallucination evaluation [7]; RAGTruth provides manually annotated hallucination labels for retrieval-augmented generation at both response and word levels [22]; and FActScore evaluates longform generations by decomposing them into atomic facts and checking factual precision against external evidence [23]. Together, these benchmarks emphasize fine-grained grounding, evidence sensitivity, and cross-domain generalization.

## B. Contrastive, Siamese, and Label-Efficient Learning

Contrastive and Siamese methods are attractive under limited supervision because existing examples can induce multiple relational learning signals. SimCLR [24] defines positives through data augmentation, supervised contrastive learning [25], [26] organizes relations using class labels, and Siamese networks [27], [28] employ weight-tied encoders to compare paired inputs in a shared representation space. MultiSCL [29] further combines sentence-level and pairlevel supervised contrastive objectives with augmented views to learn discriminative representations from limited labels. However, generic pair definitions can be poorly aligned with document-grounded factuality: cross-document negatives may be separable through topic or domain mismatch, while unrelated same-label examples need not rely on comparable evidence and can form weak positives. CPIL therefore ties pair construction to the factuality task itself by aligning views of the same document-claim instance and contrasting oppositelabel claims under the same document.

## C. Paraphrase Augmentation, Invariance, and Hard Negatives

Paraphrase augmentation has been widely used to improve robustness and data efficiency through lexical perturbation [30], back-translation [31], [32], and paraphrase generation [33]–[35]. Unsupervised paraphrase generation and information-dropping augmentation provide further examples of data-efficient text transformation [36], [37]. For factuality detection, however, paraphrastic transformations must preserve not only broad semantics but also the original support relation, since changes to entities, quantities, temporal expressions, or negation may alter the label. Moreover, appendstyle augmentation exposes a classifier to additional surface forms but does not explicitly constrain corresponding views to have similar representations; CPIL instead uses cross-view alignment to learn paraphrastic invariance. Hard negatives can further sharpen contrastive learning [25], [26], [38], but difficulty must be controlled carefully: uniformly sampled negatives may become easy as training progresses, whereas the most similar opposite-label pairs may be ambiguous, noisy, or overly positive-like. CPIL therefore uses the positive-pair similarity distribution to filter unsafe candidates and balance informative negatives with broader coverage.

## III. PROPOSED METHOD

This section formalizes document-grounded factuality detection and presents the proposed CPIL, as illustrated in Fig. 2. Sec. III-A defines the task and notation. Sec. III-B introduces the label-preserving paraphrastic views and the factualityaware positive and negative candidate pairs. Sec. III-C presents the quantile-aware semi-hard negative mining strategy. Sec. III-D describes the two-stage optimization procedure, and Sec. III-E discusses its computational complexity.

## A. Problem Setup

We formulate document-grounded factuality detection as a binary classification problem [7]. Let $\mathcal { D } = \{ ( d _ { i } , c _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote the available labeled dataset, where $d _ { i }$ is a source document, $c _ { i }$ is a claim to be verified against $d _ { i } ,$ and $y _ { i } \in \{ 0 , 1 \}$ is the factuality label. We use $y _ { i } ~ = ~ 1$ to denote that the claim is supported by the document and $y _ { i } = 0$ to denote that the claim is unsupported. A factuality detector estimates $h ( d , c ) = \operatorname* { P r } ( y = 1 \mid d , c )$

CPIL focuses on reusing an existing labeled subset $\mathcal { D } _ { s } \subset \mathcal { D }$ without acquiring additional human factuality labels. A direct classifier uses each example in $\mathcal { D } _ { s }$ independently and consequently extracts only one instance-level supervision signal from each label. A generic contrastive strategy could create positives from same-label examples and negatives from different-label examples [24]. However, such pair definitions are not well aligned with document-grounded factuality. Two supported claims from different documents may share the same label but rely on unrelated evidence, while two claims from different documents with opposite labels may be separable through topic, length, or writing style rather than groundedness. CPIL thus constructs pairs around the grounding relation, using cross-paraphrase positives and same-document, opposite-label negatives.

## B. Paraphrastic Views and Pair Construction

a) Label-Preserving Paraphrases: For each labeled example $\left( d _ { i } , c _ { i } , y _ { i } \right) \in \mathcal { D } _ { s } .$ , CPIL creates alternative surface forms of the document-claim input. Let P denote a set of paraphrasing transformations. A paraphraser $p \in \mathcal P$ returns a view $( d _ { i } ^ { ( p ) } , c _ { i } ^ { ( p ) } )$ . We define the original view as $( d _ { i } ^ { ( 0 ) } , c _ { i } ^ { ( 0 ) } ) =$ $( d _ { i } , c _ { i } )$ , and let $\mathcal { V } = \{ 0 \} \cup \mathcal { P }$ be the full view set.

CPIL is based on the label-preservation condition: if a paraphrastic view preserves the factual content of both the document and the claim, their support relation should also remain unchanged. Accordingly, each retained view $( d _ { i } ^ { ( p ) } , c _ { i } ^ { ( p ) } )$ inherits the original label y<sub>i</sub>.

![](images/c42860002cfe1cdbdc8c410092cad1353f47c06e71a1f9da92d627d4327efa2a.jpg)  
Fig. 2. Framework of CPIL. (a) Pair Construction: CPIL generates paraphrastic views via back-translation and constructs factuality-aware training pairs. Positive pairs are cross-paraphrase views of the same document-claim instance, while negative pairs are opposite-label claims grounded in the same source document. (b) Two-Stage Training: Stage 1 trains a shared encoder $f _ { \theta }$ with a Siamese pair-similarity ranking objective. Stage 2 attaches a lightweight classifier $g _ { \phi }$ and fine-tunes the detector for binary factuality detection.

In this work, we instantiate the paraphraser with backtranslation [31], [32]. The input document-claim pair is translated into a pivot language and then translated back to the source language. Back-translation provides surface variation without human rewriting or prompt engineering and can be implemented with off-the-shelf translation systems. CPIL is not restricted to back-translation; other transformations, such as controlled rewriting or QA-style reformulation [39], could be used as long as the resulting views provide semantically preserving paraphrases.

b) Cross-Paraphrase Positive Pairs: Positive candidates are built from different views of the same labeled example. Given the view set V, CPIL links distinct views of each document-claim instance:

$$
\begin{array} { r l } & { \mathcal { D } ^ { + } = \Big \{ \big ( ( d _ { i } ^ { ( p ) } , c _ { i } ^ { ( p ) } ) , ( d _ { i } ^ { ( q ) } , c _ { i } ^ { ( q ) } ) \big ) : ( d _ { i } , c _ { i } , y _ { i } ) \in \mathcal { D } _ { s } , } \\ & { } \\ & { \qquad p , q \in \mathcal { V } , \ p < q \Big \} . } \end{array}\tag{1}
$$

Here $p < q$ denotes a fixed ordering over views and avoids duplicate unordered pairs. With $| \nu |$ views, each labeled example contributes at most $( ^ { | \mathcal { V } | } _ { 2 } )$ positive pairs before downsampling.

These positive pairs provide relational supervision beyond ordinary data augmentation. Appending paraphrases to the classifier training set exposes the model to more surface forms, but it does not explicitly require their representations to agree. By contrast, optimization over $\mathcal { D } ^ { + }$ explicitly encourages representations of two views of the same document-claim instance to be close. This encourages the encoder to focus less on superficial wording and more on the evidence-claim relation that is stable across faithful paraphrases.

c) Same-document Negative Pairs: Negative pairs are designed to be more challenging for document-grounded factuality detection. The original negative set contains claim pairs that share the same source document but have opposite labels:

$$
\Big \{ \big ( ( d _ { i } , c _ { i } ) , ( d _ { j } , c _ { j } ) \big ) : d _ { i } = d _ { j } , c _ { i } \neq c _ { j } , y _ { i } \neq y _ { j } \Big \} .\tag{2}
$$

The condition $d _ { i } = d _ { j }$ fixes the evidence source and reduces direct cross-document topic and domain shortcuts. The model must instead distinguish claims with different factuality labels under the same source document.

CPIL further allows paraphrastic views on either side of the negative pair:

$$
\begin{array} { r l } & { \mathcal { D } ^ { - } = \Big \{ \big ( ( d _ { i } ^ { ( p ) } , c _ { i } ^ { ( p ) } ) , ( d _ { j } ^ { ( q ) } , c _ { j } ^ { ( q ) } ) \big ) : d _ { i } ^ { ( 0 ) } = d _ { j } ^ { ( 0 ) } , \ c _ { i } ^ { ( 0 ) } \neq c _ { j } ^ { ( 0 ) } , } \\ & { \qquad y _ { i } \neq y _ { j } , \ p , q \in \mathcal { V } \Big \} . } \end{array}\tag{3}
$$

The same-document constraint is evaluated using the original source document, because paraphrasing may change surface wording. These cross-view negatives combine two types of supervision: they retain the evidence-controlled contrast of Eq. (2), while introducing additional paraphrastic variability.

Algorithm 1 summarizes the offline construction procedure. It first generates the available paraphrastic views for each labeled document-claim example, and then constructs the positive and negative candidate pools defined in Eqs. (1) and (3). This procedure is performed once before model optimization. The next subsection describes how CPIL dynamically refreshes the negative set during training.

## C. Quantile-Aware Semi-Hard Negative Mining

Uniformly sampled same-document negatives are already stronger than random cross-document negatives because they share the same evidence source. However, many negatives may become easy as training progresses. Selecting only the most similar negatives is also undesirable: an extremely similar opposite-label pair may be an informative near-miss, but it may instead reflect paraphrase drift, annotation ambiguity, or insufficient evidence in the source document.

CPIL therefore dynamically selects semi-hard negatives while filtering candidates that become excessively similar to positive pairs. At the t-th training epoch, CPIL first uniformly samples a positive subset with K samples: $\mathcal { D } _ { K } ^ { + } \subseteq \mathcal { D } ^ { + }$ , where $| { \mathcal { D } } _ { K } ^ { + } | ~ = ~ K$ . Let $f _ { \theta , t }$ be the current encoder. For a pair $\boldsymbol { r } = \big ( ( d _ { a } , c _ { a } \big ) , ( d _ { b } , c _ { b } ) \big )$ , its mining similarity is

Algorithm 1 Offline Candidate Pool Construction   
Require: Labeled subset $\begin{array} { r c l } { \mathcal { D } _ { s } } & { = } & { \{ ( d _ { i } , c _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { s } } ; } \end{array}$ pivot  
language set $\mathcal { P } ;$ back-translation operator $\mathrm { B T } _ { p } ( \cdot )$   
Ensure: positive and negative candidate pools $\mathcal { D } ^ { \bar { + } } , \mathcal { D } ^ { - }$   
View generation   
1: $\mathcal { V }  \{ 0 \} \cup \mathcal { P } ,$ , where 0 denotes the original view   
2: for all $( d _ { i } , c _ { i } , y _ { i } ) \in \mathcal { D } _ { s }$ do   
3: $d _ { i } ^ { ( 0 ) } \gets d _ { i } , c _ { i } ^ { ( 0 ) } \gets c _ { i }$   
4: for all $p \in \mathcal P$ do   
5: Generate $( d _ { i } ^ { ( p ) } , c _ { i } ^ { ( p ) } )$ by back-translation   
6: end for   
7: end for   
Candidate-pool construction   
8: Build candidate pools $\mathcal { D } ^ { - }$ and $\mathcal { D } ^ { + }$ using Eqs. (1) and (3)   
9: return $\mathcal { D } ^ { + } , \mathcal { D } ^ { - }$

$$
s _ { t } ( \boldsymbol { r } , f _ { \theta , t } ) = \cos ( f _ { \theta , t } ( d _ { a } , c _ { a } ) , f _ { \theta , t } ( d _ { b } , c _ { b } ) ) .\tag{4}
$$

$f _ { \theta , t }$ is used for similarity calculation with gradients disabled. To avoid selecting negatives that are nearly indistinguishable from positive pairs, CPIL computes a positive-derived safety boundary based on the positive set $\mathcal { D } _ { K } ^ { + } \mathrm { ~ : ~ }$

$$
\tau _ { t } = Q _ { \rho } \left( \left\{ s _ { t } ( r ) : r \in \mathcal { D } _ { K } ^ { + } \right\} \right) ,\tag{5}
$$

where $Q _ { \rho }$ is the $\rho \mathrm { - }$ th percentile of positive-pair similarities. Negative candidates whose similarity exceeds this boundary are treated as overly positive-like and are excluded:

$$
\mathcal { M } _ { t } ^ { \mathrm { s a f e } } = \{ r \in \mathcal { D } ^ { - } : s _ { t } ( r ) < \tau _ { t } \} .\tag{6}
$$

Because the number of safe negatives can vary across epochs, the effective pair budget is

$$
K _ { t } = \operatorname* { m i n } \left\{ K , | \mathcal { M } _ { t } ^ { \mathrm { s a f e } } | \right\} .\tag{7}
$$

Within the safe pool, CPIL selects a mixture of difficult and diverse negatives. Let $\mathrm { T o p } _ { m } ( \mathcal { A } ; s _ { t } )$ denote the m elements in a set $\mathcal { A }$ with the largest similarity scores. The exploitation semi-hard subset is

$$
\mathcal { M } _ { t } ^ { \mathrm { h a r d } } = \mathrm { T o p } _ { \lfloor \lambda K _ { t } \rfloor } ( \mathcal { M } _ { t } ^ { \mathrm { s a f e } } ; s _ { t } ) ,\tag{8}
$$

where $\lambda ~ \in ~ [ 0 , 1 ]$ controls the fraction of mined semi-hard negatives. The remaining negatives are uniformly sampled from the rest of the safe pool:

$$
\mathcal { M } _ { t } ^ { \mathrm { c o v } } \sim \mathrm { U n i f o r m } \left( \mathcal { M } _ { t } ^ { \mathrm { s a f e } } \setminus \mathcal { M } _ { t } ^ { \mathrm { h a r d } } , K _ { t } - \lfloor \lambda K _ { t } \rfloor \right) .
$$

The refreshed negative set is then

(9)

$$
\mathcal { D } _ { t } ^ { - } = \mathcal { M } _ { t } ^ { \mathrm { h a r d } } \cup \mathcal { M } _ { t } ^ { \mathrm { c o v } } .\tag{10}
$$

Mining changes the empirical distribution of negatives used in the ranking objective. The boundary conservatively removes negative candidates that fall inside the positive-like region, whereas the exploitation-coverage mixture prevents training from repeatedly concentrating on a small set of highly similar candidates.

Algorithm 2 Dynamic Mining and Two-Stage Training   
Require: Labeled subset $\mathcal { D } _ { s } \dot { , }$ candidate pools $\overline { { \mathcal { D } ^ { + } , \mathcal { D } ^ { - } } }$ from   
Algorithm 1; encoder $f _ { \boldsymbol { \theta } } ;$ classifier $g _ { \phi } ;$ Stage 1 epochs $T _ { 1 } ;$   
Stage 2 epochs $T _ { 2 } ;$ pair-budget upper bound $K ;$ quantile   
level $\rho ;$ mining fraction $\lambda ;$ ranking margin α   
Ensure: Trained detector $h ( d , c ) = \sigma ( g _ { \phi } ( f _ { \theta } ( d , c ) ) )$   
Stage 1: Pairwise Encoder Training   
1: for $t = 1 , \dots , T _ { 1 }$ do   
2: Sample $\mathcal { D } _ { K } ^ { + } \sim$ Uniform $( \mathcal { D } ^ { + } , K )$   
3: Compute $s _ { t } ( r )$ for all $r \in \mathcal { D } _ { K } ^ { + } \cup \mathcal { D } ^ { - }$ using Eq. (4)   
4: Compute $\tau _ { t }$ using Eq. (5)   
5: Construct $\boldsymbol { \mathcal { M } } _ { t } ^ { \mathrm { s a f e } }$ using Eq. (6)   
6: Compute $K _ { t }$ using Eq. (7)   
7: if $K _ { t } > 0$ then   
8: Select $\mathcal { M } _ { t } ^ { \mathrm { h a r d } }$ using Eq. (8)   
9: Sample $\mathcal { M } _ { t } ^ { \mathrm { c o v } }$ using Eq. (9)   
10: Set $\bar { \mathcal { D } } _ { t } ^ { - }  \dot { \mathcal { M } } _ { t } ^ { \mathrm { h a r d } } \bar { \cup } \bar { \mathcal { M } } _ { t } ^ { \mathrm { c o } }$ v   
11: Sample $\mathcal { D } _ { t } ^ { + } \sim$ Uniform $( \mathcal { D } _ { K } ^ { + } , K _ { t } )$   
12: Update θ using ${ \mathcal L } _ { \mathrm { c t r } }$ in Eq. (13)   
13: end if   
14: end for   
Stage 2: Classifier Fine-Tuning   
15: Set $f _ { \boldsymbol { \theta } } \gets f _ { \boldsymbol { \theta } , T _ { 1 } }$ and initialize $g _ { \phi }$   
16: for $t = 1 , \ldots , T _ { 2 }$ do   
17: Jointly update $\theta , \phi$ on $\mathcal { D } _ { s }$ using ${ \mathcal { L } } _ { \mathrm { f t } }$ in Eq. (14)   
18: end for   
19: return $h ( d , c ) = \sigma ( g _ { \phi } ( f _ { \theta } ( d , c ) ) )$

Equations (4)-(10) define the epoch-wise negative-selection procedure. Algorithm 2 integrates this procedure with the twostage CPIL network training in Sec. III-D.

## D. CPIL Network Training

CPIL consists of a shared encoder $f _ { \theta }$ and a linear classifi cation head $g _ { \phi }$ . Given a packed document-claim input $( d , c )$ the encoder produces the embedding

$$
\mathbf { e } = f _ { \theta } ( d , c ) .\tag{11}
$$

The encoder can be instantiated with a pretrained Transformer such as DeBERTa [40]. In Stage 2, the linear classification head produces a scalar factuality logit

$$
g _ { \phi } ( \mathbf { e } ) = \mathbf { w } _ { \phi } ^ { \top } \mathbf { e } + b _ { \phi } .\tag{12}
$$

a) Stage 1: Siamese Pair-Similarity Pretraining: The first stage trains the encoder before attaching the final classifier. Two weight-tied towers of $f _ { \theta , t }$ process paired inputs drawn from $\mathcal { D } _ { t } ^ { + } \bigcup \mathcal { D } _ { t } ^ { - }$ . Its objective (i) pulls together crossparaphrase views of the same instances in $\mathcal { D } _ { t } ^ { + }$ and (ii) repels same-document pairs with opposite labels in $\mathcal { D } _ { t } ^ { - }$ :

$$
\mathcal { L } _ { \mathrm { c t r } } = \frac { 1 } { K _ { t } } \sum _ { k = 1 } ^ { K _ { t } } \operatorname* { m a x } \left\{ 0 , s _ { t } ( r _ { k } ^ { - } ) - s _ { t } ( r _ { k } ^ { + } ) + \alpha \right\} ,\tag{13}
$$

where $\alpha > 0$ is the margin, and $s ( r )$ indicates the similarity of two document-claim pairs as defined in Eq. (4), but with gradients enabled during optimization. Minimizing Eq. (13)

encourages cross-paraphrase positives to be closer than samedocument opposite-label negatives by at least $\alpha .$

Other contrastive objectives, such as NT-Xent [24] or supervised contrastive learning [25], [26], could be used with the same candidate construction. CPIL’s primary contribution lies in its factuality-aware pair definitions and dynamic negative selection rather than a particular contrastive loss.

b) Stage 2: Classifier Fine-Tuning: After Stage 1, CPIL attaches a classification head in Eq. (12). The detector predicts

$$
h ( d , c ) = \sigma ( g _ { \phi } ( f _ { \theta } ( d , c ) ) ) ,
$$

where $\sigma ( \cdot )$ is the sigmoid function. The encoder is initialized from Stage 1 and then fine-tuned together with the classifier on the original labeled subset $\mathcal { D } _ { s }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f t } } = \mathbb { E } _ { ( d , c , y ) \sim \mathcal { D } _ { s } } \left[ \mathrm { B C E } \big ( \sigma ( g _ { \phi } ( f _ { \theta } ( d , c ) ) ) , y \big ) \right] , } \end{array}\tag{14}
$$

where BCE denotes binary cross-entropy. Stage 2 converts the representation learned from pairwise constraints into a binary factuality predictor.

Paraphrase generation, pair construction, and dynamic mining are used only during training. At inference time, CPIL uses the standard encoder-classifier pipeline without Siamese comparison or paraphrase generation.

## E. Algorithmic Summary and Computational Complexity

Algorithms 1 and 2 separate CPIL into an offline construction stage and an online optimization stage. Algorithm 1 generates the paraphrastic views and constructs the fixed candidate pools. Algorithm 2 then reuses these pools during Stage 1, refreshing the negatives at every epoch before fine-tuning the factuality classifier in Stage 2. Unless otherwise stated, all sampling operations are performed without replacement.

Let $| \mathcal { D } ^ { + } |$ and $| \mathcal { D } ^ { - } |$ denote the positive and negative candidate-pool sizes, respectively. Algorithm 1 is executed once before training. Excluding the external cost of backtranslation, enumerating and materializing the candidate-pair indices requires $O ( | \mathcal { D } ^ { + } | + | \mathcal { D } ^ { - } | )$ time and storage.

At each Stage 1 epoch, Algorithm 2 scores K sampled positive pairs and $| \mathcal { D } ^ { - } |$ negative candidates. The pair-level scoring and selection operations therefore scale as $O ( K + | D ^ { - } | )$ , in addition to the encoder computation required to obtain their representations. A straightforward implementation may encode the same view multiple times when it occurs in different pairs. Caching the embeddings of shared views can reduce this repeated computation to the number of distinct views appearing in the scored candidate set.

After mining, the ranking update uses $K _ { t }$ matched positivenegative training units in epoch t. Stage 2 has the same asymptotic training cost as standard fine-tuning on the labeled subset $\mathcal { D } _ { s }$ . Because paraphrase generation, candidate construction, and dynamic mining are restricted to training, CPIL introduces no pairwise or mining operations at inference time. Its deployment cost is therefore that of the underlying encoder-classifier under the same input-processing protocol.

## IV. EXPERIMENTS

## A. Experiment Settings

a) Benchmark and Evaluation Metrics: We evaluate CPIL on LLM-AggreFact [7], an aggregate benchmark consisting of 11 document-grounded factuality tasks. The benchmark covers summarization, dialogue, claim verification, long-form question answering, and retrieval-augmented generation. The official development and test splits contain 30,420 and 29,320 examples, respectively. We report the F1 score (%) on each task and the macro-F1 over the 11 tasks, denoted by AVG. Macro averaging assigns equal weight to each task and therefore avoids allowing a large constituent dataset to dominate the overall result.

b) Baselines: We compare CPIL with representative non-LLM factuality detectors: SummaC-ZS and SummaC-CV [13], AlignScore [9], MiniCheck [7], FactCG [10], and CCHD [15]. These baselines cover NLI-based consistency scoring, learned alignment functions, efficient document-grounded verification, graph-based synthetic claim augmentation, and paraphraseconsistency-constrained training. When available, we use authors’ released implementations and recommended checkpoints. We additionally report Qwen2.5-7B [41], Llama3.1- 8B [42], and Mistral-Nemo-12B [43] as LLM references.

c) Implementation Details: CPIL instantiates the shared encoder $f _ { \theta }$ with DeBERTa-v3-large [40] and uses a linear classification head $g _ { \phi }$ . Each document-claim example is encoded as a paired input sequence. The maximum input length is 2,046 tokens for training.

We generate back-translated views (using Google Translate API) via three pivot languages, $\mathcal { P } ~ = ~ \{ \mathrm { f } x , \mathrm { e } s , \mathrm { z h } \}$ , corresponding to French, Spanish, and Chinese. The document and claim are translated separately through the same pivot language. Together with the original input, each example can therefore have up to four views. The training pipeline consumes cached back-translated views and retains every serialized view without manual screening. The similarity-based negative filter in Sec. III-C operates after pair construction.

The labeled training pool follows the data released with FactCG [10] and contains approximately 16,700 labeled document-claim examples. Applying the three-pivot viewgeneration and pair-construction procedure to the full pool yields more than 350,000 candidate pairs. To evaluate CPIL under restricted supervision, we use either 2% or 3% of this labeled pool, yielding approximately 7,000 and 12,000 candidate training pairs, respectively. We denote the resulting models by CPIL-2% and CPIL-3%, respectively. A source document with only one factuality class in $\mathcal { D } _ { s }$ contributes no same-document negative pairs. $\bar { \mathcal { D } } _ { s } ^ { 2 \% } \subseteq \mathcal { D } _ { s } ^ { 3 \% }$ holds.

For Stage 1, the pair budget is set separately for positive and negative pairs. We set the ranking margin $\alpha = 1 . 5$ , and the semi-hard proportion $\lambda = 0 . 8$ . The lower positive-similarity quantile $\rho = 1 0 \%$ heuristically. With $\rho$ fixed, the numerical boundary $\tau _ { t }$ is recomputed from the current positive-pair similarities at each Stage 1 epoch. $K { = } 1 , 5 0 0$ per pair type for CPIL-2%, and $K { = } 2 , 0 0 0$ per pair type for CPIL-3%. We optimize CPIL using AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ an Adam epsilon of $1 \times 1 0 ^ { - 6 }$ , zero weight decay, and a linear warmup ratio of 0.06. The training batch size is 2, with gradient accumulation over 8 batches, resulting in an effective batch size of 16. We train Stage 1 for at most three epochs and Stage 2 for one epoch.

TABLE II  
TASK-LEVEL F1 (%) AND THE UNWEIGHTED AVERAGE (AVG) OVER 11 LLM-AGGREFACT TASKS. THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY. THE CPIL-3% ROW REPORTS THE MEAN OVER THREE RANDOM SEEDS, AND THE STD ROW GIVES THE CORRESPONDING STANDARD DEVIATION.
<table><tr><td rowspan="2">Model</td><td colspan="2">AggreFact</td><td colspan="2">TofuEval</td><td rowspan="2">WiCE</td><td rowspan="2">REVEAL</td><td rowspan="2">Claim Verify</td><td rowspan="2">Fact Check</td><td rowspan="2">Expert QA</td><td rowspan="2">LFQA</td><td rowspan="2">RAG Truth</td><td rowspan="2">AVG</td></tr><tr><td>CNN</td><td>XSum</td><td>MediaS</td><td>MeetB</td></tr><tr><td>Qwen2.5-7B</td><td>92.41</td><td>71.25</td><td>86.21</td><td>90.16</td><td>53.47</td><td>74.09</td><td>82.90</td><td>61.28</td><td>75.84</td><td>83.40</td><td>89.18</td><td>78.20</td></tr><tr><td>Llama3.1-8B</td><td>92.65</td><td>71.82</td><td>86.21</td><td>91.43</td><td>57.42</td><td>69.79</td><td>83.79</td><td>59.58</td><td>84.42</td><td>81.04</td><td>94.02</td><td>79.29</td></tr><tr><td>Mistral-Nemo-12B</td><td>94.07</td><td>72.14</td><td>85.82</td><td>74.28</td><td>60.53</td><td>84.43</td><td>88.04</td><td>77.47</td><td>85.00</td><td>86.40</td><td>57.38</td><td>78.69</td></tr><tr><td>SummaC-ZS</td><td>55.19</td><td>66.55</td><td>79.57</td><td>80.14</td><td>41.03</td><td>74.50</td><td>72.96</td><td>60.79</td><td>65.30</td><td>81.58</td><td>62.81</td><td>67.31</td></tr><tr><td>SummaC-CV</td><td>66.41</td><td>48.56</td><td>69.70</td><td>77.87</td><td>50.98</td><td>70.77</td><td>68.36</td><td>57.53</td><td>65.21</td><td>79.16</td><td>56.92</td><td>64.68</td></tr><tr><td>AlignScore</td><td>81.71</td><td>68.51</td><td>80.59</td><td>85.18</td><td>54.75</td><td>69.79</td><td>81.16</td><td>61.05</td><td>70.63</td><td>85.81</td><td>77.35</td><td>74.23</td></tr><tr><td>MiniCheck</td><td>74.88</td><td>74.49</td><td>83.72</td><td>81.26</td><td>62.10</td><td>72.31</td><td>85.54</td><td>60.15</td><td>72.22</td><td>88.15</td><td>86.36</td><td>76.47</td></tr><tr><td>FactCG</td><td>78.44</td><td>74.46</td><td>83.98</td><td>80.07</td><td>69.87</td><td>80.53</td><td>83.45</td><td>63.42</td><td>58.54</td><td>89.12</td><td>86.63</td><td>77.14</td></tr><tr><td>CCHD</td><td>88.05</td><td>77.10</td><td>84.74</td><td>87.43</td><td>70.70</td><td>78.29</td><td>83.81</td><td>61.47</td><td>68.53</td><td>89.46</td><td>87.40</td><td>79.73</td></tr><tr><td>CPIL-2%</td><td>93.80</td><td>70.15</td><td>86.85</td><td>88.05</td><td>56.39</td><td>81.88</td><td>89.96</td><td>65.61</td><td>86.35</td><td>90.04</td><td>82.40</td><td>81.04</td></tr><tr><td>CPIL-3%</td><td>94.77</td><td>72.34</td><td>87.40</td><td>87.05</td><td>55.80</td><td>83.48</td><td>90.01</td><td>68.41</td><td>87.36</td><td>89.62</td><td>81.33</td><td>81.60</td></tr><tr><td>STD</td><td>±0.15</td><td>±0.27</td><td>±0.51</td><td>±0.03</td><td>±1.20</td><td>±1.11</td><td>±0.03</td><td>±1.85</td><td>±0.57</td><td>±0.42</td><td>±0.66</td><td>±0.19</td></tr></table>

TABLE III

ABLATION RESULTS UNDER THE 2% LABEL BUDGET. IF AVAILABLE, ALL VARIANTS USE THE SAME LABELED SUBSET, ENCODER, PAIR BUDGET, AND OPTIMIZATION SETTINGS UNLESS OTHERWISE SPECIFIED.
<table><tr><td rowspan="2">Variant</td><td colspan="2">AggreFact</td><td colspan="2">TofuEval</td><td rowspan="2">WiCE</td><td rowspan="2">REVEAL</td><td rowspan="2">Claim Verify</td><td rowspan="2">Fact Check</td><td rowspan="2">Expert QA</td><td rowspan="2">LFQA</td><td rowspan="2">RAG Truth</td><td rowspan="2">AVG</td></tr><tr><td>CNN</td><td>XSum</td><td>MediaS</td><td>MeetB</td></tr><tr><td>w/o Pairwise Stage</td><td>70.94</td><td>69.92</td><td>81.88</td><td>88.52</td><td>68.46</td><td>76.56</td><td>83.02</td><td>60.47</td><td>72.25</td><td>89.24</td><td>83.58</td><td>76.81</td></tr><tr><td>Append-Only Aug.</td><td>78.30</td><td>69.20</td><td>82.46</td><td>86.74</td><td>69.90</td><td>77.77</td><td>86.41</td><td>61.28</td><td>65.15</td><td>87.47</td><td>86.10</td><td>77.34</td></tr><tr><td>Random All Pairs</td><td>77.16</td><td>74.15</td><td>79.84</td><td>83.52</td><td>70.31</td><td>76.49</td><td>85.04</td><td>61.69</td><td>70.70</td><td>87.69</td><td>85.33</td><td>77.44</td></tr><tr><td>Random Pos Pairs</td><td>94.61</td><td>69.45</td><td>84.38</td><td>89.03</td><td>47.02</td><td>85.13</td><td>95.97</td><td>78.40</td><td>86.56</td><td>89.16</td><td>46.74</td><td>78.77</td></tr><tr><td>Random Neg Pairs</td><td>94.71</td><td>69.14</td><td>84.52</td><td>88.98</td><td>49.22</td><td>74.70</td><td>95.96</td><td>57.26</td><td>86.56</td><td>89.24</td><td>47.34</td><td>76.15</td></tr><tr><td>w/o Dynamic Mining</td><td>91.08</td><td>71.38</td><td>84.05</td><td>90.10</td><td>64.91</td><td>77.77</td><td>81.74</td><td>58.36</td><td>71.94</td><td>88.90</td><td>84.79</td><td>78.64</td></tr><tr><td>Fuil CPIL-2%</td><td>93.80</td><td>70.15</td><td>86.85</td><td>88.05</td><td>56.39</td><td>81.88</td><td>89.96</td><td>65.61</td><td>86.35</td><td>90.04</td><td>82.40</td><td>81.04</td></tr></table>

d) Threshold Protocol: For each task, we first score its development examples, and choose the threshold that maximizes supported-class development F1. If multiple thresholds tie, we first select the one closest to 0.5 and then the larger threshold. The selected threshold is then frozen and applied to the corresponding test task. The same threshold-selection procedure is applied to all methods.

## B. Main Results

Table II reports the task-level and macro-F1 results on LLM-AggreFact. CPIL achieves the highest macro-F1 among the compared compact factuality detectors under both low-label settings. CPIL-2% obtains an AVG of 81.04 using 2% of the FactCG-released labeled pool, outperforming CCHD by 1.31 points and FactCG by 3.90 points. Increasing the label budget to 3% further raises the AVG to 81.60, and CPIL-3% also exceeds the two reported vanilla LLM references in macro-F1. These results show that the available low-budget labels can provide strong supervision when they are converted into task-aware pairwise constraints.

## C. Ablation Study

Table III evaluates the contribution of the pairwise pretraining stage, the factuality-aware pair definitions, and dynamic negative mining. We compare Full CPIL with six variants.

• w/o Pairwise Stage removes Stage 1 and directly trains a classifier on the original labeled examples.

• Append-Only Aug. also removes Stage 1, but appends paraphrased views to the classifier training set as ordinary augmented samples.

• Random All Pairs keeps Stage 1 but randomly constructs both positive and negative pairs.

• Random Pos Pairs replaces cross-paraphrase positives with randomly matched positive pairs.

• Random Neg Pairs replaces same-document opposite label negatives with randomly matched negative pairs.

• w/o Dynamic Mining retains the two-stage framework and the proposed pair definitions but replaces quantile-aware semi-hard mining with static negative sampling.

a) Effect of Pairwise Pretraining: Removing Stage 1 decreases AVG from 81.04 to 76.81, a reduction of 4.23 points. Append-only augmentation raises AVG from 76.81 to 77.34 but remains 3.70 points below Full CPIL. Thus, merely increasing the number of paraphrastic training examples does not reproduce the gain obtained by explicitly aligning different views through pairwise pretraining.

![](images/ad82d9769f0ca1b4ce5604067cec1c3f32544659ff1975ec18bf775eaed47763.jpg)  
(a) Sensitivity to labeled data size.

![](images/94a6a71f931b2e1446c30df90df83019205fed291e0e870de7b04c682f99d669.jpg)  
(b) Sensitivity to semi-hard ratio λ.  
Fig. 3. Sensitivity analysis of CPIL. Left: macro-F1 under different labeled data budgets. Right: macro-F1 under different proportions λ of semi-hard negatives in the selected negative set. Performance improves rapidly in the lowest-label regimes and remains strong over a range of mixed semi-hard configurations.

b) Effect of Factuality-Aware Pair Construction: Random All Pairs obtains an AVG of 77.44, which is close to Append-Only Aug. This result indicates that the benefit of Stage 1 depends on how the pairs are defined rather than on pairwise training alone. Generic random pairs provide weak or noisy supervision for document-grounded factuality. Random Pos Pairs reaches an AVG of 78.77 but produces highly uneven task-level results, with severe drops on WiCE and RAGTruth. The result supports constructing positives from different paraphrastic views of the same document-claim example rather than from arbitrary instances that only share a label. Random Neg Pairs drops to 76.15 AVG, showing that negative pairs must preserve the evidence context. Without the same-document constraint, many negatives can be separated by topic or document mismatch rather than by factual support.

Taken together, Append-Only Aug. (77.34), Random All Pairs (77.44), and Full CPIL (81.04) support the importance of task-aware relational structure, rather than merely more augmented examples or a pairwise training stage.

c) Effect of Dynamic Negative Mining: Replacing dynamic mining with static negative sampling reduces AVG to 78.64. Because this variant retains the two-stage architecture and the same positive and negative candidate definitions, the 2.40-point difference provides evidence for the contribution of adaptive negative selection under the tested configuration. Static same-document negatives are useful, but they may become easy as the encoder improves. Full CPIL refreshes the negative set according to the current representation space, selects informative semi-hard negatives, and retains coverage through uniform sampling from the safe pool.

## D. Sensitivity Analysis

We analyze CPIL with respect to the labeled-data fraction and the semi-hard negative proportion λ.

a) Labeled-Data Fraction: Fig. 3a varies the labeled subset from 0.2% to 5.0% of the FactCG-released labeled pool. Macro-F1 generally increases as more labeled examples become available, with the largest gains occurring in the lowest-budget range. At 0.2%, CPIL already reaches a macro-F1 level comparable to several compact baselines. The curve continues to improve through the 2% and 3% settings and changes only modestly between 3% and 5%.

This pattern is consistent with CPIL’s relational-supervision design. At very low budgets, each additional labeled example introduces not only one classification target but also multiple candidate views and pairwise relations. As the labeled pool grows, the marginal benefit becomes smaller, indicating that increasing the number of labeled examples alone does not yield proportional gains beyond the moderate-budget regime. However, the label-efficiency claim mainly concerns reuse of selected factuality annotations, but not about training time or total compute, given extra one-time back-translation cost.

b) Semi-Hard Negative Proportion: Fig. 3b varies λ, the proportion of semi-hard negatives in the selected negative set. When λ = 0, all selected negatives are uniformly sampled from the safe pool, resulting in substantially lower macro-F1 than the mixed-selection settings. Increasing λ improves performance, with the strongest results appearing in the range from 0.5 to 0.8. The default setting λ = 0.8 achieves nearpeak performance while assigning most of the pair budget to semi-hard negatives.

When λ = 1, all selected negatives come from the highest similarity region of the safe pool, and performance decreases slightly relative to the mixed settings. This result supports retaining a coverage component: semi-hard examples provide informative comparisons, whereas uniformly sampled safe negatives prevent the selected set from concentrating exclusively on a narrow group of high-similarity pairs.

## E. Effect of Pivot Languages

We examine whether CPIL benefits from combining multiple back-translation paths. Figs. 4a and 4b compare the threepivot configuration with French-, Spanish-, and Chinese-only variants under the 2% and 3% label budgets.

Under the 2% budget, the three-pivot setting obtains an AVG of 81.04, compared with 74.08 for French, 77.95 for Spanish, and 73.10 for Chinese. Under the 3% budget, the corresponding results are 81.60, 77.22, 80.42, and 73.71. Thus, the three-pivot setting achieves the highest macro-F1 under both supervision budgets.

Pivot-2%: Pivot Strategy Comparison  
![](images/9f47080e68c644950750d0e28d77cd11e54c15cf261bc02f4fb62be04dff75f7.jpg)  
(a) 2% label budget.

Pivot-3%: Pivot Strategy Comparison  
![](images/da7ef9d074b0ef9dc99ea94ab7dbbc314eb763d5c5ad2acc996660ed2334cc4a.jpg)  
(b) 3% label budget.  
Fig. 4. Macro-F1 comparison of single- and three-pivot back-translation settings under the 2% and 3% label budgets. Each radar axis corresponds to one task; values in parentheses in the legends are the unweighted mean over all 11 tasks (macro-F1/AVG).

Spanish is the strongest single-pivot option in both settings, but it remains 3.09 points below the three-pivot configuration under the 2% budget and 1.18 points below it under the 3% budget. The benefit of using multiple pivots is therefore larger in the lower-label setting. One possible explanation is that additional pivot paths provide more varied pairwise relations when only a small number of labeled anchors are available.

The task-level plots also show that the relative ranking of pivot languages varies across datasets. No single pivot consistently dominates all tasks. The three-pivot configuration reduces dependence on one translation path and empirically provides a stronger overall setting for CPIL.

## V. CONCLUSION

This paper presented CPIL, a two-stage Siamese framework for label-efficient document-grounded factuality detection. CPIL converts a limited set of labeled document-claim examples into relational supervision through cross-paraphrase positive pairs and same-document, opposite-label negative pairs. It further introduces quantile-aware semi-hard negative mining, which filters overly positive-like negative candidates and balances negative difficulty with training-set coverage.

On the 11-task LLM-AggreFact benchmark, CPIL-2% and CPIL-3% achieve macro-F1 scores of 81.04 and 81.60, respectively. Both settings outperform the six compact factuality detectors and two vanilla LLM references. The ablation results further show that replacing dynamic mining with static negative sampling decreases macro-F1 to 78.64, while appendonly augmentation and generic random pairing are also less effective than the proposed task-aware relational supervision.

A limitation of CPIL is that back-translated views are assumed to preserve the original factuality label. Small changes to entities, quantities, temporal expressions, or negation may alter the document-claim support relation. In addition, the current evaluation is limited to an English aggregate benchmark and relies on an external translation service for offline view generation. Future work will study entailment- or entity-aware filtering, alternative paraphrasers, and broader cross-domain and multilingual evaluation [44].

## ACKNOWLEDGMENTS

This work was supported by the National Science Foundation of China (No. 62476060). Sihong Xie was supported by the Department of Science and Technology of Guangdong Province (No. 2023CX10X079).

## REFERENCES

[1] S. S. Rahman, M. A. Islam, M. M. Alam, M. Zeba, M. A. Rahman, S. S. Chowa, M. A. K. Raiaan, and S. Azam, “Hallucination to truth: a review of fact-checking and factuality evaluation in large language models,” Artificial Intelligence Review, 2026.

[2] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, “On faithfulness and factuality in abstractive summarization,” in Proceedings of the 58th annual meeting of the association for computational linguistics, 2020, pp. 1906–1919.

[3] A. Krishna, E. Galinkin, L. Derczynski, and J. Martin, “Importing phantoms: Measuring llm package hallucination vulnerabilities,” arXiv preprint arXiv:2501.19012, 2025.

[4] P. Manakul, A. Liusie, and M. Gales, “Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models,” in Proceedings of the 2023 conference on empirical methods in natural language processing, 2023, pp. 9004–9017.

[5] D. Muhammed, G. Rabby, and S. Auer, “Selfcheckagent: Zero-resource hallucination detection in generative large language models,” arXiv preprint arXiv:2502.01812, 2025.

[6] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-eval: NLG evaluation using gpt-4 with better human alignment,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2023, pp. 2511– 2522.

[7] L. Tang, P. Laban, and G. Durrett, “MiniCheck: Efficient fact-checking of LLMs on grounding documents,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 8818–8847.

[8] A. Fabbri, C.-S. Wu, W. Liu, and C. Xiong, “QAFactEval: Improved QA-based factual consistency evaluation for summarization,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 2022, pp. 2587–2601.

[9] Y. Zha, Y. Yang, R. Li, and Z. Hu, “AlignScore: Evaluating factual consistency with a unified alignment function,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Toronto, Canada: Association for Computational Linguistics, 2023, pp. 11 328–11 348.

[10] D. Lei, Y. Li, S. Li, M. Hu, R. Xu, K. Archer, M. Wang, E. Ching, and A. Deng, “FactCG: Enhancing fact checkers with graph-based multihop data,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Association for Computational Linguistics, 2025, pp. 5002–5020.

[11] S. Lin, D. Hong, S. Ju, C. Chen, S. Xie, and X. Liao, “Cross paraphrastic invariance learning for hallucination detection,” in ICASSP 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 16 662–16 666.

[12] W. Kryscinski, B. McCann, C. Xiong, and R. Socher, “Evaluating the factual consistency of abstractive text summarization,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2020, pp. 9332–9346.

[13] P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, “Summac: Revisiting nli-based models for inconsistency detection in summarization,” Transactions of the Association for Computational Linguistics, vol. 10, pp. 163–177, 2022.

[14] T. Scialom, P.-A. Dray, S. Lamprier, B. Piwowarski, J. Staiano, A. Wang, and P. Gallinari, “Questeval: Summarization asks for fact-based evaluation,” in Proceedings of the 2021 conference on empirical methods in natural language processing, 2021, pp. 6594–6604.

[15] S. Lin, D. Hong, S. Ju, C. Chen, X. Zhang, and X. Liao, “Constrained paraphrase consistency for llm hallucination detection,” in ICASSP 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 16 652–16 656.

[16] Q. Mao, J. Li, H. Peng, S. He, L. Wang, P. S. Yu, and Z. Wang, “Fact-driven abstractive summarization by utilizing multi-granular multirelational knowledge,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 1665–1678, 2022.

[17] I. Chern, S. Chern, S. Chen, W. Yuan, K. Feng, C. Zhou, J. He, G. Neubig, P. Liu et al., “Factool: Factuality detection in generative ai–a tool augmented framework for multi-task and multi-domain scenarios,” arXiv preprint arXiv:2307.13528, 2023.

[18] J. Wei, C. Yang, X. Song, Y. Lu, N. Hu, J. Huang, D. Tran, D. Peng, R. Liu, D. Huang et al., “Long-form factuality in large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 80 756–80 827, 2024.

[19] L. Zheng, C. Li, Z. Liu, F. Huang, H. Jia, Z. Ye, and X. Zhang, “Fact in fragments: Deconstructing complex claims via LLM-based atomic fact extraction and verification,” Expert Systems with Applications, vol. 302, p. 130572, 2026.

[20] H.-Y. Lin, Y.-C. Chang, and J.-T. Chien, “Variational dehallucination via retrieval-augmented prompt learning,” IEEE Transactions on Audio, Speech and Language Processing, vol. 34, pp. 259–271, 2025.

[21] O. Honovich, R. Aharoni, J. Herzig, H. Taitelbaum, D. Kukliansy, V. Cohen, T. Scialom, I. Szpektor, A. Hassidim, and Y. Matias, “True: Re-evaluating factual consistency evaluation,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2022, pp. 3905–3920.

[22] C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang, “Ragtruth: A hallucination corpus for developing trustworthy retrieval-

augmented language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 10 862–10 878.

[23] S. Min, K. Krishna, X. Lyu, M. Lewis, W.-t. Yih, P. Koh, M. Iyyer, L. Zettlemoyer, and H. Hajishirzi, “Factscore: Fine-grained atomic evaluation of factual precision in long form text generation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 12 076–12 100.

[24] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PmLR, 2020, pp. 1597–1607.

[25] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” Advances in neural information processing systems, vol. 33, pp. 18 661–18 673, 2020.

[26] B. Gunel, J. Du, A. Conneau, and V. Stoyanov, “Supervised contrastive learning for pre-trained language model fine-tuning,” arXiv preprint arXiv:2011.01403, 2020.

[27] A. He, C. Luo, X. Tian, and W. Zeng, “A twofold siamese network for real-time object tracking,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4834–4843.

[28] C. Chen, Y. Shen, G. Ma, X. Kong, S. Rangarajan, X. Zhang, and S. Xie, “Self-learn to explain siamese networks robustly,” in 2021 IEEE International Conference on Data Mining (ICDM). IEEE, 2021, pp. 1018–1023.

[29] S. Li, X. Hu, L. Lin, A. Liu, L. Wen, and P. S. Yu, “A multilevel supervised contrastive learning framework for low-resource natural language inference,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 1771–1783, 2023.

[30] J. Wei and K. Zou, “EDA: Easy data augmentation techniques for boosting performance on text classification tasks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 2019, pp. 6382–6388.

[31] R. Sennrich, B. Haddow, and A. Birch, “Improving neural machine translation models with monolingual data,” in Proceedings of the 54th annual meeting of the association for computational linguistics (volume 1: long papers), 2016, pp. 86–96.

[32] S. Edunov, M. Ott, M. Auli, and D. Grangier, “Understanding backtranslation at scale,” in Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2018, pp. 489–500.

[33] D. R. Beddiar, M. S. Jahan, and M. Oussalah, “Data expansion using back translation and paraphrasing for hate speech detection,” Online Social Networks and Media, vol. 24, p. 100153, 2021.

[34] A. Agarwal, S. Gupta, V. Bonagiri, M. Gaur, J. Reagle, and P. Kumaraguru, “Towards effective paraphrasing for information disguise,” in European Conference on Information Retrieval. Springer, 2023, pp. 331–340.

[35] V. Yadav, Z. Tang, and V. Srinivasan, “Pag-llm: Paraphrase and aggregate with large language models for minimizing intent classification errors,” in Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 2024, pp. 2569–2573.

[36] L. Li, Y. Cai, and X. Wu, “Unsupervised disentanglement learning model for exemplar-guided paraphrase generation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 1885–1900, 2024.

[37] S. Li, X. Bi, T. Liu, and Z. Chen, “Information dropping data augmentation for machine translation quality estimation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 2112–2124, 2024.

[38] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 815– 823.

[39] B. Yang, M. A. Al Mamun, J. M. Zhang, and G. Uddin, “Hallucination detection in large language models with metamorphic relations,” Proceedings of the ACM on Software Engineering, vol. 2, no. FSE, pp. 425–445, 2025.

[40] P. He, J. Gao, and W. Chen, “Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing,” arXiv preprint arXiv:2111.09543, 2021.

[41] Q. Team, “Qwen2.5: A party of foundation models,” September 2024. [Online]. Available: https://qwenlm.github.io/blog/qwen2.5/

[42] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[44] X. Chen, C. Wang, Y. Xue, N. Zhang, X. Yang, Q. Li, Y. Shen, L. Liang, J. Gu, and H. Chen, “Unified hallucination detection for multimodal large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2024, pp. 3235–3252.

[43] Mistral AI Team, “Mistral nemo,” https://mistral.ai/news/mistral-nemo/, Jul. 2024, 12B open language model developed with NVIDIA.