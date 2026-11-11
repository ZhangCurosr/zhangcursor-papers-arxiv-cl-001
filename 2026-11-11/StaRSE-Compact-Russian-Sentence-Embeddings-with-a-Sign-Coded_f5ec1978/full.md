# StaRSE: Compact Russian Sentence Embeddings with a Sign-Coded Static Encoder

Boris Malashenko<sup>1</sup>, Ivan Jarsky<sup>1</sup>, Egor Soldatov<sup>1</sup>, Vladislav Luzhnov<sup>1</sup>, Sviatoslav Stumpf<sup>1</sup>, Vladislav Shmatkov<sup>1</sup>, Viacheslav Shalamov<sup>1</sup>, Andrey Stankevich<sup>1</sup> and Valeria Efimova<sup>1</sup>

<sup>1</sup>ITMO University, Saint Petersburg, Russia

## Abstract

Transformer sentence encoders provide strong retrieval quality, but their inference cost becomes a bottleneck when embeddings must be produced on CPUs or refreshed frequently. We introduce StaRSE, a compact Russian static sentence encoder that removes Transformer inference at deployment time: token directions are stored as sign bits, token norms are stored separately, and sentences are encoded by mean pooling over norm-scaled sign directions. StaRSE is initialized from ruBERT-base, fine-tuned on Russian adjacent-sentence and task-shaped pairs, and post-trained with a contrastive stage that directly optimizes the sign-projected token table. On RuMTEB, StaRSE reaches a task mean of 51.16, 7.98 points above the strongest included static baseline. The packaged model occupies 11.3 MiB and has a mean CPU latency of 0.08 ms per sentence, compared with an estimated 55.7 MiB 16-bit parameter footprint and 1.43 ms mean CPU latency for the closest-quality contextual encoder. Ablations examine adjacent-sentence supervision, token norms, and direct sign-projected post-training.

## Keywords

sentence embeddings, static embeddings, information retrieval, model compression, Russian language

## 1. Introduction

Text embedding models are now a standard component of neural retrieval systems, semantic search, clustering, deduplication, recommendation, and retrieval-augmented generation. The strongest generalpurpose embedding models are contextual encoders, typically trained or adapted with sentence-level contrastive objectives [1, 2, 3, 4]. Their quality, however, comes with a deployment cost: every query and document requires a Transformer forward pass [5]. This cost is often acceptable for reranking [6] or small collections, but it becomes restrictive when embeddings must be produced on CPU, refreshed frequently, served under tight latency budgets, or computed for very large text collections.

Static embedding models, from word2vec [7], GloVe [8], and fastText [9] to sentence-level and recent static embedding variants [10, 11, 12], ofer a diferent point in this quality-cost trade-of. At inference time, they use a fixed vocabulary, a simple lookup table, pooling over token vectors, and normalization. This makes inference cheap and predictable, especially on CPUs and resource-constrained hardware. The price is a strict architectural bottleneck: a mean-pooled static embedding model cannot model word order, contextual word senses, or compositional interactions in the way a Transformer can. The practical question is therefore not whether static embedding models should replace contextual encoders, but whether they can provide useful retrieval quality at much lower serving cost.

This question is underexplored for Russian and, more broadly, for language-specific eficient retrieval outside English. Recent Russian sentence transformers include strong contextual models such as FRIDA [13] and GigaEmbeddings [14]. Russian static embedding models, however, remain mostly word-level, while multilingual static embedding models are not optimized specifically for Russian morphology and vocabulary coverage. We treat Russian as one instance of a broader setting: languages with large text corpora but limited high-quality labeled sentence pairs, for which a reusable static-encoder training recipe may be useful.

We introduce StaRSE, a static sentence encoder for Russian with a sign-coded token table: token directions are stored as sign bits, while per-token scalar norms are stored separately. The pipeline deliberately combines established ingredients from static embeddings and contrastive sentence embedding training: Transformer input embeddings, token-level dimensionality reduction, frequency weighting, adjacent-sentence positives, and InfoNCE-style optimization. The main StaRSE-specific component is the compact sign-coded token table with mean pooling over norm-scaled sign directions and a final training stage that computes the contrastive loss through the sign-projected table instead of signing a continuous table only after training. At deployment time, StaRSE stores only the compact token table and encodes text by pooling norm-scaled sign directions, avoiding Transformer inference for both queries and documents. Figure 1 shows this trade-of numerically: StaRSE reaches 51.16 task mean on RuMTEB, 7.98 points above the strongest included static encoder, and remains within 2.60 points of ruBERT-tiny-turbo. In Table 4, StaRSE’s measured package is 11.3 MiB, while the estimated 16-bit parameter footprint of ruBERT-tiny-turbo is 55.7 MiB. Under the same CPU protocol, StaRSE has about 18× lower mean single-sentence latency (0.08 versus 1.43 ms). Larger contextual encoders remain substantially more accurate, but they occupy a diferent cost range.

Our contributions are as follows.

1. We introduce StaRSE, a compact static sentence encoder for Russian. StaRSE reaches 51.16 RuMTEB: 7.98 points above the strongest included static encoder and 2.60 points below the closest included contextual encoder by RuMTEB score.

2. We provide an empirical recipe for carrying out the full training pipeline for a language-specific static sentence encoder, including tokenizer and initialization choices, document-adjacency mining, task-shaped pair construction, and training-stage decisions.

3. We propose contrastive post-training for static embedding models that directly optimizes the sign-projected token geometry.

The evaluated SentenceTransformers-compatible StaRSE model weights<sup>1</sup> and training code<sup>2</sup> are publicly available.

## 2. Related Work

Static word embeddings were the dominant vector representation before the rise of contextual encoders. Models such as word2vec [7], GloVe [8], and fastText [9] assign one vector to each vocabulary unit. Sentence or document vectors were then commonly built compositionally, for example by averaging word vectors, adding �-gram features as in Sent2Vec [10], or applying simple weighting and postprocessing heuristics. These approaches are attractive because inference reduces to lookup and pooling, but their tables are usually optimized for word-level distributional semantics rather than for generalpurpose sentence embeddings.

Contextual encoders now provide the strongest sentence representations. BERT [15] introduced contextual token representations, and Sentence-BERT [1] adapted Transformer encoders for eficient sentence-level similarity. We use Sentence Transformer (ST) as a shorthand for Transformer encoders trained or adapted to produce sentence-level embeddings directly. Later STs and related contextual embedding systems such as SimCSE [2], E5 [3], and GTE [4] improved embedding quality through successive changes to the training pipeline and data construction. MTEB [16] broadened evaluation beyond STS to information retrieval (IR), reranking, classification, clustering, and pair classification, including natural language inference (NLI)-style tasks. This progress made STs the default choice for high-quality embeddings, while static models remained attractive when CPU latency, memory use, indexing cost, or hardware availability are limiting factors.

Recent work has revisited static embedding models by reusing representations from pretrained Transformer models. WordLlama recycles the input token embeddings of LLaMA-family models and trains a lightweight context-free text embedding model from them [17]. Model2Vec instead distills a static vocabulary table from an existing sentence embedding model, producing a fast lookup-and-pooling model without running the teacher at inference time [12]. A third line starts from contextualized word representations: contextual models assign diferent vectors to the same word in diferent sentences [18], and these contextual vectors can be reduced to static word representations by aggregating occurrences [19]. Wada and colleagues apply this idea to sentence embeddings by collecting many corpus occurrences of each word, encoding them in context, and averaging the resulting token representations into one static vector [11]. These recipes difer in what they require. Input-embedding reuse and teacher distillation need a pretrained model and its vocabulary, whereas occurrence-based extraction also needs a corpus with enough contexts for each vocabulary item. Such extraction can be computationally expensive because it requires many Transformer forward passes to collect last-hidden-state token representations before the static table can be built.

![](images/5177749b2f0d13246b7d05bf638e647755d6856127cc2435f028e43019660ebc.jpg)  
Figure 1: Quality on RuMTEB [13] versus mean end-to-end single-sentence encoding latency in milliseconds (log scale; lower is better). Static encoders are measured on CPU and contextual encoders on GPU, so the two device groups represent diferent deployment regimes. Marker area shows encoder footprint in MiB on a log scale: contextual footprints are estimated as parameter count times two bytes, while static footprints are measured packages. Labeled blue points form the nondominated frontier of the displayed score–latency measurements under this mixed-device diagnostic. The star marks StaRSE.

Dimensionality reduction and token weighting are central to the initialization and post-processing used in these static models. WordLlama and Model2Vec apply PCA to the token embedding matrix, while Wada et al. [11] argue for sentence-level PCA, in which PCA is fitted to averaged sentence representations so that the retained axes better reflect sentence-level variance. In both cases, PCA can be combined with All-but-the-Top (ABTT) post-processing, which discards a small number of dominant principal components instead of simply keeping the leading ones [20]. This can reduce the influence of common directions that appear across many embeddings and can otherwise dominate cosine similarities. Frequency-based weighting addresses a diferent issue: very frequent tokens often contribute less semantic specificity to an averaged sentence vector. SIF [21] downweights frequent tokens using corpus frequency. When reliable counts are unavailable, a Zipfian prior over vocabulary rank provides a simple approximation [22].

After initialization and post-processing, static embedding models are often trained with either distillation or contrastive learning. Knowledge distillation transfers information from a stronger teacher to a smaller student [23]. For STs, this can mean matching teacher sentence vectors, matching pairwise teacher similarities within a batch, or aligning multilingual sentence spaces [24, 11]. Distillation is attractive because it can create a static model from a teacher and vocabulary without a labeled pair dataset, but direct vector matching constrains the student to approximate a teacher geometry with much lower capacity. Contrastive learning instead trains the model to score positive pairs above in-batch or explicitly sampled negatives, as in InfoNCE-style embedding fine-tuning [25, 26].

Embedding compression is another way to reduce memory and search cost for embedding models. Some text-retrieval and sentence-representation systems binarize final dense sentence or passage vectors so nearest-neighbor search can use Hamming distance or equivalent bit operations [27, 28, 29]. In our work, StaRSE uses a diferent form of compression: it stores sign-projected token directions and separate scalar token norms, while the pooled sentence vector remains real-valued. Vector and weight quantization reduce memory and arithmetic precision, as in quantization-aware training for neural network inference [30]. Quantization-aware methods are useful when post-hoc quantization would change the representation seen by the loss: training with the quantized forward path lets the objective optimize the representation used at inference.

Russian embedding research follows the broader contextual trend. RuMTEB provides a Russian version of MTEB with the same broad task families [13], while RusBEIR collects Russian information retrieval tasks for retrieval-focused evaluation [31]. Recent Russian sentence-embedding work includes FRIDA [13] and GigaEmbeddings [14]. GigaEmbeddings combines large-scale contrastive pre-training with hard-negative and multitask fine-tuning, synthetic data, and eficient contextual pooling. Russian static embedding models are mostly word-level: Navec<sup>3</sup> and DeepPavlov<sup>4</sup> both provide compact Russian word embeddings. These models are lightweight, but using them for sentence-level semantics typically requires adapting word-level vectors rather than using a model trained for sentence-level similarity. When static text embedders are multilingual, their vocabulary and capacity are shared across languages rather than targeted to Russian. This leaves room for a Russian-specific static sentence encoder trained directly for sentence-level retrieval.

## 3. Method

StaRSE is built in three steps. First, we choose a tokenizer vocabulary and initialize the static token table from a pretrained Transformer encoder input embedding matrix (Sec. 3.1). Second, we construct weak positive pairs from source documents by pairing adjacent sentences (Sec. 3.2). Third, we train the table with a bidirectional InfoNCE loss and then continue training with the sign-projected token table used at inference time (Sec. 3.3).

## 3.1. Vocabulary Selection and Embedding Initialization

The first design choice is the vocabulary. A static embedding model inherits both its coverage and its tokenization behavior from a fixed vocabulary �. The tokenizer should keep target language text reasonably compact, while preserving lexical distinctions that matter for retrieval. We therefore start from a pretrained Transformer and reuse its tokenizer vocabulary � together with its input embedding matrix $\mathbf { \bar { Q } } \in \mathbb { R } ^ { | V | \times D }$

Given � , StaRSE is a static embedding model with embedding table $E \in \mathbb { R } ^ { | V | \times d }$ . For a non-empty tokenized text $x = ( w _ { 1 } , \ldots , w _ { n } )$ , the unnormalized sentence vector is the mean of the corresponding token vectors:

$$
h ( x ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } E _ { w _ { i } } .\tag{1}
$$

The final representation is length-normalized:

$$
z ( x ) = \frac { h ( x ) } { \operatorname* { m a x } ( \| h ( x ) \| _ { 2 } , \varepsilon ) } .\tag{2}
$$

For two texts � and �, we use the dot product of their normalized vectors as the similarity score, $\boldsymbol { s } ( x , y ) = \boldsymbol { z } ( x ) ^ { \top } \boldsymbol { z } ( y )$ . Because both vectors are length-normalized, this is cosine similarity.

To initialize �, we extract � from the selected Transformer and project it to the target dimensionality with token-level PCA. Instead of keeping the first � principal components, we apply ABTT postprocessing [20]. We discard the first � dominant components and keep the next � components. We then apply SIF-style frequency weighting. For the token at position $i \in \{ 1 , \ldots , | V | \}$ in the tokenizer’s vocabulary order, we use the normalized Zipf proxy $\begin{array} { r } { p _ { i } = i ^ { - \beta } / \sum _ { j = 1 } ^ { | V | } j ^ { - \beta } } \end{array}$ and rescale its projected row as $E _ { i } \gets a E _ { i } / ( a + p _ { i } )$ , where � is the SIF smoothing coeficient and $\beta$ is the Zipf exponent.

## 3.2. Document Pair Construction

Curated sentence-pair datasets provide high-quality supervision, but for most languages there are few large, diverse, and reliably labeled pair collections. This is especially limiting for a general static sentence encoder. Unlike a Sentence Transformer, StaRSE does not retain a pretrained contextual encoder at inference time. The input embedding matrix gives a lexical and subword-level starting point, but sentence-level similarity must be induced in the token table itself during training.

Document collections provide a broad weakly supervised signal for this purpose. The use of adjacent sentences follows the same basic intuition as BERT’s next sentence prediction objective [15]: consecutive text segments often carry discourse information. We use this signal diferently. Instead of training a classifier to decide whether two segments are consecutive, we use adjacent sentences as positive pairs for contrastive training. Adjacent sentences are not guaranteed to be paraphrases: they may continue the same topic or discourse without expressing the same meaning. Nevertheless, document adjacency preserves topic continuity, entities, discourse relations, and local semantic coherence more often than random cross-document pairing. Section 5.2 examines this assumption empirically with a diagnostic sentence encoder. We therefore treat adjacent sentences as weak positives rather than exact semantic equivalents.

Given a collection ofsource documents, we segment each document into sentences and form candidate positive pairs from consecutive sentences. Each pair is $( u _ { i } , u _ { i + 1 } )$ , where both sentences come from the same document and preserve their original order. To remove accidental transitions, sentence-splitting artifacts, and pairs with no shared lexical content, we apply a lightweight lemma-overlap filter. Let $L ( x )$ be the set of normalized non-stopword lemmas in sentence �. For a candidate pair $( u , v )$ , we compute the Dice overlap

$$
\mathrm { L O } ( u , v ) = \frac { 2 | L ( u ) \cap L ( v ) | } { | L ( u ) | + | L ( v ) | } ,\tag{3}
$$

with zero overlap when the denominator is zero. For the main Cultura pool, we keep only pairs whose lemma overlap is at least 0.20. This filter is deliberately lexical and inexpensive: it is used as a fast corpus-cleaning step before contrastive training, not as a semantic teacher or hard-mining procedure.

## 3.3. Contrastive Training

We train the static embedding table on text pairs with a bidirectional InfoNCE objective, following the pairwise embedding fine-tuning setup used in recent embedding models [25, 26]. Let $\boldsymbol { B } =$ $( ( p _ { 1 } , q _ { 1 } ) , \dots , ( p _ { k } , q _ { k } ) )$ be a batch of� positive pairs. For a scoring function $s ( \cdot , \cdot )$ and one batch direction $B = ( ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { k } , y _ { k } ) )$ , the InfoNCE term is

$$
\mathcal { L } _ { \mathrm { N C E } } ( \mathcal { B } ; s ) = - \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \log \frac { \exp ( s ( x _ { i } , y _ { i } ) / \tau ) } { \sum _ { i ^ { \prime } = 1 } ^ { k } \exp ( s ( x _ { i } , y _ { i ^ { \prime } } ) / \tau ) } ,\tag{4}
$$

where � is the temperature. For each anchor $x _ { i }$ , the paired text $y _ { i }$ is the positive example and the other texts $\{ y _ { i ^ { \prime } } : i ^ { \prime } \neq i \}$ in the same batch act as in-batch negatives. Let $B ^ { \dagger } = ( ( q _ { 1 } , p _ { 1 } ) , \dots , ( q _ { k } , p _ { k } ) )$ denote the same batch with the order of each pair swapped. The symmetric pair loss is

$$
\mathcal { L } _ { \mathrm { p a i r s } } ( \boldsymbol { B } ; \boldsymbol { s } ) = \frac { 1 } { 2 } \left[ \mathcal { L } _ { \mathrm { N C E } } ( \boldsymbol { B } ; \boldsymbol { s } ) + \mathcal { L } _ { \mathrm { N C E } } ( \boldsymbol { B } ^ { \dag } ; \boldsymbol { s } ) \right] .\tag{5}
$$

For the continuous StaRSE model, $s ( \cdot , \cdot )$ is the cosine similarity defined in Section 3.1. The symmetric loss lets each sentence in a pair serve once as the anchor. The same objective applies to adjacent-sentence pairs and to curated sentence-pair batches.

Sign-projected post-training. The final compact model is obtained through quantization-aware post-training. After contrastive fine-tuning, let $E ^ { ( \bar { 0 } ) }$ denote the learned real-valued table. The compact StaRSE checkpoint is initialized from $E ^ { ( 0 ) }$ . In this stage the optimizer still updates one real-valued embedding matrix $E \in \mathbb { R } ^ { | V | \times d }$ , but every forward pass computes sentence vectors through signprojected token directions. This makes the training loss act on the same compact representation that is stored for inference.

For each row of the current table, we separate its norm from its direction:

$$
E _ { i } = \rho _ { i } u _ { i } , \qquad \rho _ { i } = \operatorname* { m a x } ( \| E _ { i } \| _ { 2 } , \varepsilon ) , \qquad u _ { i } = E _ { i } / \rho _ { i } .\tag{6}
$$

The norm $\rho _ { i }$ sets how strongly token � contributes to a mean-pooled sentence vector, while the direction $u _ { i }$ is the part that will be sign-projected. These quantities are not stored as separate parameters: they are recomputed from the current $E _ { i }$ at every step, and ordinary optimizer updates on $E$ can change both the token norm and direction.

The forward path uses the sign-projected token table

$$
E _ { i } ^ { B } = \rho _ { i } \frac { \mathrm { s i g n } ( u _ { i } ) } { \sqrt { d } } .\tag{7}
$$

Equivalently, for $b _ { i } = \mathrm { s i g n } ( u _ { i } ) / \sqrt { d } ,$ the pre-normalization pooled vector is

$$
h _ { B } ( x ) = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \rho _ { w _ { t } } b _ { w _ { t } } .\tag{8}
$$

Thus $\rho _ { i }$ determines how much a token contributes to the pooled vector, while $b _ { i }$ determines the sign direction contributed by that token. Let $f _ { B } ( x )$ be the same mean-pooling and ℓ -normalization encoder as in Section 3.1, but using $E ^ { B }$ . StaRSE sign-projects token directions, not final sentence vectors: after pooling and length normalization, $f _ { B } ( x )$ is still a real-valued vector computed from signs plus one scalar norm per token. The sign-projected similarity is

$$
\begin{array} { r } { \boldsymbol { s } _ { B } ( x , y ) = \boldsymbol { f } _ { B } ( x ) ^ { \top } \boldsymbol { f } _ { B } ( y ) . } \end{array}\tag{9}
$$

The sign projection is used with a straight-through estimator: the forward pass substitutes $\mathrm { s i g n } ( u _ { i } ) / \sqrt { d }$ for the underlying unit row while the backward pass treats it as the identity on $u _ { i } ,$ so the gradient with respect to $E _ { i }$ flows through both the direction $u _ { i }$ and the magnitude $\rho _ { i }$ even though the forward path is discrete.

For a batch direction $B = ( ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { k } , y _ { k } ) )$ , let $S _ { B } \in \mathbb { R } ^ { k \times k }$ be the sign-projected score matrix with $( S _ { B } ) _ { i j } = s _ { B } ( x _ { i } , y _ { j } )$ , and let $S _ { T }$ be the analogous score matrix from the frozen continuous StaRSE model $E ^ { ( 0 ) }$ . We denote by $P _ { T }$ and $P _ { B }$ the collections of row-wise and column-wise distributions induced by these matrices, using

$$
\sigma _ { r } ^ { i } ( S ; \tau _ { T } ) = \mathrm { s o f t m a x } ( S _ { i ; \cdot } / \tau _ { T } ) , \qquad \sigma _ { c } ^ { j } ( S ; \tau _ { T } ) = \mathrm { s o f t m a x } ( S _ { \cdot , j } / \tau _ { T } ) .\tag{10}
$$

The teacher distribution term is a row-and-column KL divergence,

$$
\begin{array} { r l } { \displaystyle \mathrm { K L } \big ( P _ { T } \| P _ { B } \big ) = \frac { 1 } { 2 k } \sum _ { i = 1 } ^ { k } \mathrm { K L } \big ( \sigma _ { r } ^ { i } ( S _ { T } ; \tau _ { T } ) \big | \big | \sigma _ { r } ^ { i } ( S _ { B } ; \tau _ { T } ) \big ) } & { } \\ { \displaystyle } & { + \frac { 1 } { 2 k } \sum _ { j = 1 } ^ { k } \mathrm { K L } \big ( \sigma _ { c } ^ { j } ( S _ { T } ; \tau _ { T } ) \big | \big | \sigma _ { c } ^ { j } ( S _ { B } ; \tau _ { T } ) \big ) . } \end{array}\tag{11}
$$

This term encourages the sign-projected table to match the continuous model’s row-wise and columnwise similarity rankings within the batch.

The final StaRSE stage combines two loss terms: the symmetric contrastive pair loss on the signprojected table and the teacher-distribution term:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S t a R S E } } = \alpha \mathcal { L } _ { \mathrm { p a i r s } } ( \boldsymbol { B } ; \boldsymbol { s } _ { B } ) + \lambda _ { T } \mathrm { K L } ( P _ { T } | | P _ { B } ) . } \end{array}\tag{12}
$$

Here � and $\lambda _ { T }$ weight the contrastive pair loss and the teacher-distribution loss, respectively.

## 4. Experiments

This section describes the experimental setup and results for StaRSE. We specify the training data, model configuration, baselines, and evaluation protocol, then report RuMTEB, long-document retrieval, and encoding-eficiency results. All RuMTEB scores and nDCG@10 values are reported on a 0–100 scale throughout the paper; their diferences are expressed in points on this scale.

## 4.1. Training Data

The contrastive fine-tuning recipe combines adjacent sentence pairs from Cultura $\mathrm { R U E d u } ^ { 5 }$ with Russian retrieval and paraphrase-style pair sources. We use Cultura as the broad text source because adjacent sentences expose entities, topics, and discourse transitions that are less common in small supervised similarity datasets. Following Section 3.2, we use the Razdel library<sup>6</sup> to split Cultura documents into sentences and form candidate pairs from consecutive sentences $( s _ { i } , s _ { i + 1 } )$ in the same document. Cultura pairs are retained with the lemma-overlap filter from Section 3.2.

The remaining sources provide more task-shaped positives. From the Russian subset of the mMARCO IR passage-ranking dataset [32], we use the query and the positive passage from each training triple. From the ru-HNP paraphrase dataset,<sup>7</sup> we treat each source text and its positive paraphrases as a positive set and form all unordered pairs within that set. From ru-paraphrase-NMT-Leipzig,<sup>8</sup> a paraphrase/translation dataset, we use the original and back-translated pair. Basic normalization removes empty, identical, and overlong pairs. During continuous training, a source-balanced sampler prevents the smaller supervised datasets from being overwhelmed by Cultura.

## 4.2. Model and Training Setup

StaRSE uses the WordPiece tokenizer and vocabulary of ruBERT-base<sup>9</sup>, with 120,138 tokens. The final model keeps this vocabulary fixed and learns one table row for each token. Section 5.1 provides the development results behind the tokenizer and initialization choices.

We initialize the final 512-dimensional table from the corresponding input embedding matrix using token-level PCA, ABTT, and SIF-Zipf weighting, as selected in Section 5.1.

Continuous contrastive training has two phases, both optimizing only the symmetric pair loss $\mathcal { L } _ { \mathrm { p a i r s } } ( B ; s )$ from Section 3.3. The first phase runs for 20,000 steps on the four-source mixture. The second continues for 10,000 steps, and its final checkpoint initializes the sign-projected post-training stage and serves as its frozen teacher.

The final stage starts from this mixed-source checkpoint and trains on Cultura through the signprojected table; its two loss terms are given in Equation 12. Its hyperparameters were selected by an exploratory Optuna sweep using the full RuMTEB mean, after which the selected configuration was retrained for a longer run. This makes the final RuMTEB score selection-biased rather than a held-out estimate.

## 4.3. Baselines

We compare StaRSE against two broad groups of models. The first group consists of selected Sentence Transformer (ST) baselines from the measured quality–latency–size trade-of in Figure 1: ruBERT-tinyturbo,<sup>10</sup> ruBERT-mini-FRIDA,<sup>11</sup> BERTA,<sup>12</sup> FRIDA,<sup>13</sup> and GigaEmbeddings [14]. We choose these models because they lie on the measured Pareto frontier among the contextual encoders in that comparison.

The second group consists of static embedding baselines. For Russian pretrained word vectors, we include DeepPavlov fastText wiki+lenta<sup>14</sup> and Navec hudlit.<sup>15</sup> For multilingual static text embeddings, we include Static similarity<sup>16</sup> and Potion multilingual<sup>17</sup>. Together, these baselines cover Russian pretrained word vectors and public multilingual static text embedders with Russian support. The static baselines give the closest architectural comparison, while the ST baselines show how StaRSE is positioned relative to strong contextual encoders under the measured quality-cost trade-of.

## 4.4. Experiments on Russian Vector Representations

The primary benchmark is RuMTEB (v1.1), which contains 23 Russian tasks. Table 1 reports the RuMTEB scores grouped by task type. The StaRSE row is the sign-coded token-table checkpoint used as the main model in this paper. It reaches a 51.16 full RuMTEB task mean, 7.98 points above the strongest included static baseline, while keeping the static lookup–pool–normalize inference path and storing token directions as signs plus scalar norms. StaRSE leads the included static baselines in every task family except pair classification, where the static models are efectively tied. This family is represented by TERRa, an entailment task whose sentence-pair relations are a poor fit for context-free mean-pooled token tables. Contextual ST models remain substantially stronger on retrieval-heavy tasks, as expected from their contextual encoders. The final block of Table 1 reports lexical ranking models on the two reranking and three retrieval tasks; these methods do not produce general-purpose sentence vectors, so no overall RuMTEB mean or score for the remaining task types is defined.

Lexical ranking protocol. We compare TF–IDF [33], BM25 [34], PL2 [35], and a Dirichlet-smoothed query-likelihood model [36] with StaRSE’s ruBERT WordPiece tokenizer and no stemming or lemmatization. The RuMTEB runs share one analyzer and retain stopwords. Retrieval searches the full corpus, whereas reranking uses each task’s supplied candidate pool. We use fixed settings on RuMTEB. BM25 is the strongest lexical baseline in both RuMTEB groups.

We also include an auxiliary long-document diagnostic. Static encoders do not run self-attention, so long documents can be processed by streaming token lookups into the same mean-pooling accumulator. We use the Russian subset of MLDR<sup>18</sup>, a multilingual retrieval benchmark with especially long documents, and evaluate on 10,000 Russian corpus documents and 200 test queries. Table 2 compares retrieval with 8,192 retained document tokens against static and contextual references: StaRSE reaches 51.66 nDCG@10, 17.35 points above the next included static baseline and 0.03 below USER2-small. The contextual block reproduces the MLDR-RU values reported in the pinned USER2-small model card, while the static rows are evaluated locally on the fixed pool.<sup>19</sup>

Table 1  
RuMTEB (v1.1) comparison across the 23 Russian tasks. Columns report the benchmark mean and task-type means on the 0–100 scale; bold marks the best row within each model block. Lexical rankers are defined only for reranking and retrieval, so their other cells are not applicable.
<table><tr><td>Model (dims)</td><td>Mean Class. Cluster. Multilabel PairClass. Rerank Retrieval</td><td></td><td></td><td></td><td></td><td></td><td>STS</td></tr><tr><td colspan="8">Sentence Transformers (STs)</td></tr><tr><td>ruBERT-tiny-turbo (312)</td><td>53.76 55.08</td><td>49.62</td><td>32.05</td><td>56.27</td><td>54.94</td><td>48.82</td><td>71.73</td></tr><tr><td>ruBERT-mini-FRIDA (312)</td><td>63.22</td><td>64.28 59.71</td><td>47.19</td><td>60.36</td><td>65.62</td><td>64.70</td><td>72.14</td></tr><tr><td>BERTA (768)</td><td>69.39</td><td>71.22 64.78</td><td>56.46</td><td>65.74</td><td>69.72</td><td>73.79</td><td>73.72</td></tr><tr><td>FRIDA (1536)</td><td>70.95</td><td>73.71</td><td>67.49</td><td>52.23</td><td>66.45 71.50</td><td>76.38</td><td>74.35</td></tr><tr><td>GigaEmbeddings (2048)</td><td>74.16</td><td>76.66 67.48</td><td>57.07</td><td>79.57</td><td>74.02</td><td>81.41</td><td>75.75</td></tr><tr><td colspan="8">Static Embedding Models</td></tr><tr><td>DeepPavlov fastText wiki+lenta (300) 42.85</td><td>51.87</td><td>42.52</td><td>27.31</td><td>52.46</td><td>30.53</td><td>18.78</td><td>55.53</td></tr><tr><td>Navec hudlit (300)</td><td>36.94</td><td>46.60 37.31</td><td>26.82</td><td>52.57</td><td>24.44</td><td>7.66</td><td>46.73</td></tr><tr><td>Static similarity (1024)</td><td>42.28</td><td>46.12 34.72</td><td>28.09</td><td>51.59</td><td>38.00</td><td>28.12</td><td>61.72</td></tr><tr><td>Potion multilingual (256)</td><td>43.18</td><td>50.12 41.39</td><td>28.80</td><td></td><td>50.33 33.47</td><td>22.03</td><td>59.00</td></tr><tr><td>StaRSE (512)</td><td>51.16</td><td>56.81 51.80</td><td>35.01</td><td>52.50</td><td>41.88</td><td>39.09</td><td>62.17</td></tr><tr><td colspan="8">Lexical Ranking Models</td></tr><tr><td>TF-IDF</td><td></td><td></td><td></td><td></td><td>39.55</td><td>43.70</td><td></td></tr><tr><td>BM25</td><td></td><td></td><td></td><td></td><td>42.76</td><td>46.45</td><td></td></tr><tr><td>PL2</td><td></td><td></td><td></td><td></td><td>38.34</td><td>41.91</td><td></td></tr><tr><td>DirichletLM</td><td></td><td></td><td></td><td></td><td>41.06</td><td>42.44</td><td></td></tr></table>

MLDR-RU long-document retrieval on the test split. Scores are nDCG@10 percentages; bold marks the best row within each block.
<table><tr><td>Model (dims) nDCG@10</td></tr><tr><td>Sentence Transformers (STs) USER2-small (384) 51.69</td></tr><tr><td>USER2-base (768) 54.17 USER-bge-m3 (1024) 58.53 KaLM-v1.5 (896) 53.75 jina-embeddings-v3 (1024) 49.67 E5-mistral-7b (4096) 52.40</td></tr><tr><td>Static Embedding Models DeepPavlov fastText wiki+lenta (300) 26.04</td></tr><tr><td>Navec hudlit (300) 12.67 Static similarity (1024) 34.31 Potion multilingual (256) 32.95</td></tr></table>

Table 3 varies the retained document length for WordPiece BM25, dense StaRSE, z-score fusion of BM25 with StaRSE, and USER2-small as a compact long-context ST baseline. The shorter-length fusion rows use one fixed weight. At 8,192 tokens, we instead report the strongest measured norm-weighted fusion; its variant and weight were selected on the same test pool, so the comparison is descriptive rather than held-out. StaRSE rises from 32.11 at 128 tokens to 52.01 at 4,096 tokens, then scores 51.66 at 8,192 tokens, within 0.03 points of USER2-small. BM25 reaches 69.97 nDCG@10, while the exploratory StaRSE+BM25 fusion reaches 73.07, an observed 3.10-point diference.

## Table 3

MLDR-RU length scaling on 10,000 Russian documents and 200 test queries. Scores are nDCG@10 percentages. Rows show how retrieval quality changes as more document tokens are retained; bold marks the best model at each length.
<table><tr><td>Max doc tokens</td><td>BM25</td><td>StaRSE</td><td>StaRSE+BM25</td><td>USER2-small</td></tr><tr><td>128</td><td>28.34</td><td>32.11</td><td>30.84</td><td>42.31</td></tr><tr><td>512</td><td>30.09</td><td>33.63</td><td>34.17</td><td>46.97</td></tr><tr><td>1024</td><td>32.47</td><td>36.88</td><td>36.05</td><td>48.22</td></tr><tr><td>2048</td><td>34.07</td><td>40.90</td><td>38.38</td><td>48.67</td></tr><tr><td>4096</td><td>68.12</td><td>52.01</td><td>69.93</td><td>53.70</td></tr><tr><td>8192</td><td>69.97</td><td>51.66</td><td>73.07</td><td>51.69</td></tr></table>

Table 4 compares encoder footprint and mean end-to-end single-sentence latency for a fixed 95- character Cultura RU Edu sentence. Timings include tokenization, pooling, and normalization. CPU measurements use an AMD Ryzen 9 7950X, and GPU measurements use one NVIDIA GeForce RTX 3090 Ti. Static-model GPU latency is omitted because no protocol-matched optimized run is available.

## Table 4

RuMTEB score, encoder footprint, and mean end-to-end single-sentence latency on a fixed Cultura RU Edu input. Contextual footprints are estimated from 16-bit parameter counts; static footprints are measured packages. Static-model GPU latency is omitted because no protocol-matched optimized run is available. Bold marks the best reported value within each block and column.
<table><tr><td>Model (dims)</td><td>Size (MiB)</td><td>RuMTEB</td><td>CPU (ms)</td><td>GPU (ms)</td></tr><tr><td colspan="5">Sentence Transformers (STs)</td></tr><tr><td>ruBERT-tiny-turbo (312)</td><td>55.7</td><td>53.76</td><td>1.43</td><td>1.68</td></tr><tr><td>ruBERT-mini-FRIDA (312)</td><td>61.5</td><td>63.22</td><td>2.97</td><td>3.01</td></tr><tr><td>BERTA (768)</td><td>245</td><td>69.39</td><td>19.36</td><td>3.80</td></tr><tr><td>FRIDA (1536)</td><td>1571</td><td>70.95</td><td>110.87</td><td>9.61</td></tr><tr><td>GigaEmbeddings (2048)</td><td>6580</td><td>74.16</td><td>773.61</td><td>43.26</td></tr><tr><td colspan="5">Static Embedding Models</td></tr><tr><td>DeepPavlov fastText wiki+lenta (300)</td><td>1853</td><td>42.85</td><td>0.05</td><td></td></tr><tr><td>Navec hudlit (300)</td><td>589</td><td>36.94</td><td>0.04</td><td></td></tr><tr><td>Static similarity (1024)</td><td>414</td><td>42.28</td><td>0.17</td><td></td></tr><tr><td>Potion multilingual (256)</td><td>507</td><td>43.18</td><td>0.07</td><td></td></tr><tr><td>StaRSE (512)</td><td>11.3</td><td>51.16</td><td>0.08</td><td></td></tr></table>

## 5. Analysis

## 5.1. Vocabulary, Initialization, and PCA Selection

We first choose the tokenizer vocabulary, because it determines the set of trainable vectors in a static embedding model. We screen Russian-focused tokenizers with two lightweight diagnostics: the number of retained analyzer tokens per lexical word and BM25 nDCG@10 when each tokenizer is used as a tokenizer-only analyzer [34] on 12 RusBEIR retrieval tasks [31]. The screen focuses on Russian tokenizers; token-count statistics use a sample of 250,000 lexical words extracted from RusBEIR text fields, and tokenizers with identical sample tokenization are represented by one family. The ruBERTbase tokenizer [37] gives the best lexical proxy score while keeping segmentation reasonably compact, so StaRSE uses its 120,138-token vocabulary. Table 5 reports the full diagnostics.

The valid-token count is measured after analyzer filtering, including removal of special and punctuation tokens, and is not an unfiltered tokenizer fertility measure. The word-frequency baseline emits one token for a word present in its vocabulary and none for an out-of-vocabulary word. Its 0.813 valid tokens per word therefore reflects 81.35% word coverage (203,366 of 250,000 sampled words), with the remaining 18.65% omitted. It emits exactly one token per covered word, so the sub-unit value is not a segmentation advantage over the subword tokenizers.

Table 5  
Tokenizer diagnostics used for vocabulary selection. BM25 is used as a tokenizer-only retrieval proxy over 12 RusBEIR tasks. BM25 nDCG@10 is reported on the 0–100 scale. Valid tok/word counts retained analyzer tokens per sampled lexical word and is afected by filtering and out-of-vocabulary coverage.
<table><tr><td>Tokenizer</td><td>Vocab</td><td>Valid tok/word</td><td>BM25 nDCG@10</td></tr><tr><td>ai-forever/ruBert-base</td><td>120,138</td><td>1.342</td><td>52.46</td></tr><tr><td>deepvk/USER-bge-m3</td><td>46,166</td><td>1.696</td><td>51.02</td></tr><tr><td>cointegrated/rubert-tiny2</td><td>83,828</td><td>1.303</td><td>50.81</td></tr><tr><td>DeepPavlov/rubert-base-cased-conversational</td><td>100,792</td><td>1.280</td><td>50.57</td></tr><tr><td>cointegrated/rubert-tiny</td><td>29,564</td><td>2.171</td><td>50.27</td></tr><tr><td>deepvk/USER2-smal1</td><td>50,368</td><td>2.153</td><td>49.88</td></tr><tr><td>ai-forever/FRIDA</td><td>93,651</td><td>2.130</td><td>49.79</td></tr><tr><td>DeepPavlov/rubert-base-cased</td><td>119,547</td><td>1.264</td><td>49.74</td></tr><tr><td>ai-forever/ru-en-RoSBERTa</td><td>98,505</td><td>2.135</td><td>49.72</td></tr><tr><td>ai-sage/Giga-Embeddings-instruct</td><td>128,256</td><td>2.213</td><td>49.67</td></tr><tr><td>ai-forever/ruElectra-large</td><td>120,138</td><td>1.329</td><td>49.64</td></tr><tr><td>ai-forever/ruRoberta-large</td><td>50,257</td><td>2.152</td><td>49.43</td></tr><tr><td>Top 150k Russian word-frequency vocabulary</td><td>150k</td><td>0.813</td><td>42.18</td></tr></table>

## Table 6

Matched initialization screening on the Russian STS Benchmark validation split. All rows use the ruBERT-base vocabulary with the same post-processing. Scores are cosine Spearman; bold marks the selected initialization.
<table><tr><td>Initialization source</td><td>PCA</td><td>STS dev</td></tr><tr><td>Input embeddings</td><td>token-level</td><td>0.76810</td></tr><tr><td>Input embeddings</td><td>sentence-level</td><td>0.73663</td></tr><tr><td>Averaged contextual embeddings</td><td>sentence-level</td><td>0.73349</td></tr><tr><td>Per-token last-hidden state (Model2Vec)</td><td>token-level</td><td>0.64753</td></tr></table>

After fixing the vocabulary, we screen matched 256-dimensional initialization choices on the validation split of the Russian STS benchmark.<sup>20</sup> Each candidate is converted into a static embedding table and evaluated by cosine Spearman. The highest development score comes from ruBERT-base input embeddings followed by token-level PCA, ABTT, and SIF-Zipf weighting. It scores above the matched sentence-level PCA variant [11] and contextual-token alternatives in this screen, so we transfer the same input-embedding/token-level-PCA recipe to the final 512-dimensional StaRSE model. Table 6 gives the detailed results.

## 5.2. Signal in Adjacent Sentence Pairs

The method relies on adjacent sentences as weak positives, so we first check whether this signal is stronger than random document-level co-occurrence. We score three Cultura pair distributions with USER2-small used only as an external diagnostic encoder: random cross-document pairs, unfiltered adjacent sentence pairs, and adjacent pairs retained by the lemma-overlap filter from Section 3.2. This encoder is not used to train StaRSE.

We sample 30,000 pairs from each distribution. The median USER2-small cosine similarity is 0.573 for random cross-document pairs, 0.726 for unfiltered adjacent pairs, and 0.820 after the lemma-overlap filter. The filter is not a semantic objective: it favors lexical overlap, while the model itself is trained with broad contrastive batches.

## 5.3. Ablations

We run two groups of ablations. The first tests data construction and distillation baselines, while the second isolates the final training stage that uses the sign-projected token table.

Data-construction and distillation ablations. Scores are percentages; bold marks the best result in each column.
<table><tr><td>Variant</td><td>RuSTS dev</td><td>RuMTEB</td></tr><tr><td>Distillation baselines</td><td></td><td></td></tr><tr><td>BERTA-distilled static student</td><td>74.7</td><td>42.0</td></tr><tr><td>USER2-small-distilled static student</td><td>69.2</td><td>44.2</td></tr><tr><td>Positive-pair construction</td><td></td><td></td></tr><tr><td>Self-pairs</td><td>71.1</td><td>37.2</td></tr><tr><td>Raw adjacent Cultura</td><td>76.1</td><td>48.2</td></tr><tr><td>Lemma-overlap mix (≥ 0.20)</td><td>77.6</td><td>49.9</td></tr></table>

Data and distillation ablations. We first test two design choices under a simpler setup than the final StaRSE training. The self-pair, raw-adjacent, and lemma-overlap-mix contrastive runs process approximately 10M pairs each. Table 7 is therefore a set of single-run diagnostics, not a replacement for the main results.

The first block reports two static-student distillation baselines at 42.0 and 44.2 RuMTEB, alongside the higher non-self contrastive scores of 48.2 and 49.9 in the second block. For the distillation baselines, we follow the SWE-style algorithm of Wada et al. [11]: a static student is trained to match the pairwise similarity distribution induced by an ST teacher. We adapt it to the Russian teachers by using each teacher’s native tokenizer and vocabulary rather than constructing a new word-level vocabulary. We use STs as teachers because they already define a sentence-level semantic space; ruBERT-base is not used as a distillation teacher because it is a masked language model and serves here only as the source of the vocabulary and input-embedding initialization. Because the two blocks use diferent training budgets and no seed replication, we treat them as complementary diagnostics rather than a controlled ranking of objectives.

The second block examines positive-pair construction. Self-pairs score lowest, while raw adjacent Cultura and the lemma-overlap mix (≥ 0.20) reach 48.2 and 49.9 RuMTEB, respectively; their RuSTS development scores are 76.1 and 77.6. The lemma-overlap mix has the highest observed scores in this block. These single-run diagnostics do not isolate the efect of the overlap filter from the source mixture.

Sign-projected table and inference ablations. The final StaRSE checkpoint is sign-coded at the token-table level, so the key question is whether this compact table should be trained directly or obtained by post-hoc signing. Table 8 collects the relevant checks. In the same 512-dimensional architecture, signing the table immediately after contrastive fine-tuning gives 49.33 RuMTEB. Continuing ordinary contrastive fine-tuning before signing scores 50.09, while quantization-aware post-training through the sign-projected table reaches 51.16. The same table reports three inference-time negative controls. Dropping token norms is much more damaging: replacing norm-scaled sign directions with unweighted signs costs 8.42 points. Keeping token norms but signing the pooled sentence vector costs 3.84 points. The compact StaRSE representation therefore stores token directions as signs plus scalar norms, then returns a real-valued ℓ<sub>2</sub>-normalized sentence vector after mean pooling.

## 6. Conclusion

We presented StaRSE, a lightweight Russian static embedding model based on a compact sign-coded token table, mean pooling, and contrastive training on Russian sentence pairs. The final checkpoint reaches a 51.16 RuMTEB task mean while storing token directions as binary signs plus scalar norms in a measured 11.3 MiB package. It scores above the included static baselines on the RuMTEB retrieval tasks, and its quantization-aware post-training gives the strongest RuMTEB mean among the evaluated routes to a sign-coded token table. On the long-document MLDR-RU diagnostic, StaRSE reaches 51.66 nDCG@10, close to USER2-small at 51.69, while retaining a streaming lookup-and-pooling encoder. Future work should add held-out hyperparameter selection and replicated runs, strengthen retrievalspecific supervision, measure end-to-end retrieval-system eficiency, and identify the domains and query types for which lexical and StaRSE signals are most complementary. It should also explore whether strict one-bit-per-dimension sentence representations can reduce the remaining gap to real-valued pooled sentence vectors.

Ablations for the sign-coded StaRSE token table. Scores are full-RuMTEB means; deltas are relative to the first row in each block.
<table><tr><td>Setting</td><td>RuMTEB</td><td>∆</td></tr><tr><td>Training route to a sign-coded token table</td><td></td><td></td></tr><tr><td>Post-hoc signing after contrastive fine-tuning; no post-training</td><td>49.33</td><td></td></tr><tr><td>Contrastive continuation, then post-hoc signing</td><td>50.09</td><td>+0.76</td></tr><tr><td>Optimized sign-projected post-training, final StaRSE recipe</td><td>51.16</td><td>+1.83</td></tr><tr><td>Inference-time negative controls on the final checkpoint</td><td></td><td></td></tr><tr><td>StaRSE inference: token norms kept; real-valued pooled sentence vector</td><td>51.16</td><td></td></tr><tr><td>Drop token norms: use unweighted sign  $( u _ { i } ) / \sqrt { d }$  rows</td><td>42.74</td><td>-8.42</td></tr><tr><td>Drop token norms and sign the pooled sentence vector</td><td>39.06</td><td>-12.10</td></tr><tr><td>Keep token norms but sign the pooled sentence vector</td><td>47.33</td><td>-3.84</td></tr></table>

## Limitations

This work focuses on Russian, a large retrieval language with mature contextual encoders but comparatively few language-specific static sentence encoders. Applying the recipe to other languages would require separate validation of the tokenizer, filtering thresholds, source mixture, and training settings. StaRSE is also deliberately static: mean pooling over token embeddings cannot model word order, syntax, or contextual sense disambiguation in the way a Transformer can.

The main comparison and sign-projection ablations use completed full 23-task RuMTEB runs, but the large contrastive runs are single-seed and no run-to-run noise floor has been measured. The dataconstruction and distillation blocks use diferent training budgets, while the continuous initializer and final sign-projected stage use diferent source mixtures; these comparisons are therefore diagnostic and do not isolate the contribution of one data source. In addition, the final sign-projected hyperparameters were selected with an exploratory Optuna sweep over the full RuMTEB mean, so the main score is not a held-out estimate of generalization.

Our eficiency measurements cover encoder latency and encoder storage rather than a complete retrieval service. The RuMTEB lexical rankers use fixed settings rather than per-task tuning. The reported MLDR fusion setting was selected on the same evaluation pool, so its results are descriptive. Per-task tuning and analysis of the query or domain conditions under which lexical rankers complement StaRSE remain future work.

The compact model evaluated in this paper is a sign-coded token table combined with real-valued mean pooling and ℓ<sub>2</sub>-normalized sentence vectors. Strict one-bit-per-dimension sentence representations would reduce stored sentence vectors further, but in our configuration they cost 3.84 RuMTEB points (Table 8). Closing this remaining gap likely requires explicitly training the sentence vectors to be signed after pooling or rethinking how token norms participate in pooling; this is left for future work.

## Ethics Statement

StaRSE is an encoder rather than a generative model: it maps text to vectors and does not produce fluent text or answer user queries by itself. The main ethical risks therefore come from representation bias and downstream retrieval or ranking use. The training mixture includes web-derived Russian text, which may contain social biases, ofensive language, or low-quality passages. We use Cultura RU Edu as the broad web-text source and document the supervised and retrieval sources mixed with it. We do not redistribute the source corpus text. Users who deploy the encoder in search, recommendation, or classification systems should evaluate the full downstream system for bias, robustness, and privacysensitive behavior in its target domain.

## Declaration on Generative AI

During the preparation of this work, the authors used OpenAI Codex and ChatGPT for language editing, wording suggestions, consistency checks, and LaTeX polishing. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the publication’s content.

## Acknowledgments

This work supported by the Ministry of Economic Development of the Russian Federation (IGK 000000C313925P4C0002), agreement No139-15-2025-010.

## References

[1] N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using siamese BERT-networks, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019, pp. 3982–3992. URL: https://aclanthology.org/D19-1410/. doi:10.18653/v1/D19-1410.

[2] T. Gao, X. Yao, D. Chen, SimCSE: Simple contrastive learning of sentence embeddings, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 6894–6910. URL: https://aclanthology.org/2021.emnlp-main.552/. doi:10.18653/v1/2021. emnlp-main.552.

[3] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, F. Wei, Text embeddings by weakly-supervised contrastive pre-training, 2024. URL: https://arxiv.org/abs/2212.03533. doi:10. 48550/arXiv.2212.03533. arXiv:2212.03533.

[4] Z. Li, X. Zhang, Y. Zhang, D. Long, P. Xie, M. Zhang, Towards general text embeddings with multistage contrastive learning, 2023. URL: https://arxiv.org/abs/2308.03281. doi:10.48550/arXiv. 2308.03281. arXiv:2308.03281.

[5] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems 30, 2017, pp. 5998–6008. URL: https://papers.nips.cc/paper/7181-attention-is-all-you-need.

[6] R. Nogueira, K. Cho, Passage re-ranking with BERT, 2019. URL: https://arxiv.org/abs/1901.04085. doi:10.48550/arXiv.1901.04085. arXiv:1901.04085.

[7] T. Mikolov, K. Chen, G. Corrado, J. Dean, Eficient estimation of word representations in vector space, 2013. URL: https://arxiv.org/abs/1301.3781. doi:10.48550/arXiv.1301.3781. arXiv:1301.3781.

[8] J. Pennington, R. Socher, C. Manning, GloVe: Global vectors for word representation, in: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 2014, pp. 1532–1543. URL: https://aclanthology.org/D14-1162/. doi:10.3115/v1/D14-1162.

[9] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov, Enriching word vectors with subword information, Transactions of the Association for Computational Linguistics 5 (2017) 135–146. URL: https: //aclanthology.org/Q17-1010/. doi:10.1162/tacl\_a\_00051.

[10] M. Pagliardini, P. Gupta, M. Jaggi, Unsupervised learning of sentence embeddings using compositional n-gram features, in: Proceedings of the 2018 Conference of the North American Chapter of

the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 2018, pp. 528–540. URL: https://aclanthology.org/N18-1049/. doi:10.18653/v1/N18-1049.

[11] T. Wada, Y. Hirakawa, R. Shimizu, T. Kawashima, Y. Saito, Static word embeddings for sentence semantic representation, in: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 6195–6211. URL: https://aclanthology.org/2025.emnlp-main.316/. doi:10.18653/v1/2025.emnlp-main.316.

[12] S. Tulkens, T. van Dongen, Model2Vec: Fast state-of-the-art static embeddings, 2024. URL: https: //github.com/MinishLab/model2vec. doi:10.5281/zenodo.17270888.

[13] A. Snegirev, M. Tikhonova, A. Maksimova, A. Fenogenova, A. Abramov, The Russian-focused embedders’ exploration: RuMTEB benchmark and Russian embedding model design, in: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 236–254. URL: https://aclanthology.org/2025.naacl-long.12/. doi:10.18653/v1/2025.naacl-long.12.

[14] E. Kolodin, D. Khomich, N. Savushkin, A. Ianina, F. Minkin, GigaEmbeddings: Eficient Russian language embedding model, 2025. URL: https://arxiv.org/abs/2510.22369. doi:10.48550/arXiv. 2510.22369. arXiv:2510.22369.

[15] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training ofdeep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019, pp. 4171–4186. URL: https://aclanthology.org/N19-1423/. doi:10.18653/v1/N19-1423.

[16] N. Muennighof, N. Tazi, L. Magne, N. Reimers, MTEB: Massive text embedding benchmark, in: Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023, pp. 2014–2037. URL: https://aclanthology.org/2023.eacl-main.148/. doi:10.18653/v1/2023.eacl-main.148.

[17] D. L. Miller, WordLlama: Recycled token embeddings from large language models, 2024. URL: https://github.com/dleemiller/wordllama, software.

[18] K. Ethayarajh, How contextual are contextualized word representations? comparing the geometry of BERT, ELMo, and GPT-2 embeddings, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019, pp. 55–65. URL: https://aclanthology.org/D19-1006/. doi:10.18653/ v1/D19-1006.

[19] R. Bommasani, K. Davis, C. Cardie, Interpreting pretrained contextualized representations via reductions to static embeddings, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 4758–4781. URL: https://aclanthology.org/2020.acl-main.431/. doi:10.18653/v1/2020.acl-main.431.

[20] J. Mu, P. Viswanath, All-but-the-top: Simple and efective postprocessing for word representations, International Conference on Learning Representations, 2018. URL: https://openreview.net/forum? id=HkuGJ3kCb.

[21] S. Arora, Y. Liang, T. Ma, A simple but tough-to-beat baseline for sentence embeddings, International Conference on Learning Representations, 2017. URL: https://openreview.net/forum?id= SyK00v5xx.

[22] S. T. Piantadosi, Zipf’s word frequency law in natural language: A critical review and future directions, Psychonomic Bulletin & Review 21 (2014) 1112–1130. URL: https://doi.org/10.3758 s13423-014-0585-6. doi:10.3758/s13423-014-0585-6.

[23] G. Hinton, O. Vinyals, J. Dean, Distilling the knowledge in a neural network, 2015. URL: https: //arxiv.org/abs/1503.02531. doi:10.48550/arXiv.1503.02531. arXiv:1503.02531.

[24] N. Reimers, I. Gurevych, Making monolingual sentence embeddings multilingual using knowledge distillation, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020, pp. 4512–4525. URL: https://aclanthology.org/2020.emnlp-main.365/. doi:10. 18653/v1/2020.emnlp-main.365.

[25] A. van den Oord, Y. Li, O. Vinyals, Representation learning with contrastive predictive coding, 2018. URL: https://arxiv.org/abs/1807.03748. doi:10.48550/arXiv.1807.03748.

arXiv:1807.03748.

[26] S. Sturua, I. Mohr, M. K. Akram, M. Günther, B. Wang, M. Krimmel, F. Wang, G. Mastrapas, A. Koukounas, N. Wang, H. Xiao, jina-embeddings-v3: Multilingual embeddings with task LoRA, 2024. URL: https://arxiv.org/abs/2409.10173. doi:10.48550/arXiv.2409.10173. arXiv:2409.10173.

[27] D. Shen, P. Cheng, D. Sundararaman, X. Zhang, Q. Yang, M. Tang, A. Celikyilmaz, L. Carin, Learning compressed sentence representations for on-device text processing, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 107–116. URL: https://aclanthology.org/P19-1011/. doi:10.18653/v1/P19-1011.

[28] F. Hamann, N. Kurz, A. Ulges, Hamming sentence embeddings for information retrieval, 2019. URL: https://arxiv.org/abs/1908.05541. doi:10.48550/arXiv.1908.05541. arXiv:1908.05541.

[29] I. Yamada, A. Asai, H. Hajishirzi, Eficient passage retrieval with hashing for open-domain question answering, in: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, 2021, pp. 979–986. URL: https://aclanthology.org/2021.acl-short.123/. doi:10.18653/v1/2021. acl-short.123.

[30] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, D. Kalenichenko, Quantization and training of neural networks for eficient integer-arithmetic-only inference, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 2704–2713. URL: https://openaccess.thecvf.com/content\_cvpr\_2018/html/Jacob\_Quantization\_and\_Training CVPR\_2018\_paper.html. doi:10.1109/CVPR.2018.00286.

[31] G. Kovalev, M. Tikhomirov, E. Kozhevnikov, M. Kornilov, N. Loukachevitch, Building Russian benchmark for evaluation of information retrieval models, 2025. URL: https://arxiv.org/abs/2504. 12879. doi:10.48550/arXiv.2504.12879. arXiv:2504.12879.

[32] L. H. Bonifacio, V. Jeronymo, H. Q. Abonizio, I. Campiotti, M. Fadaee, R. Lotufo, R. Nogueira, mMARCO: A multilingual version of the MS MARCO passage ranking dataset, 2021. URL: https: //arxiv.org/abs/2108.13897. doi:10.48550/arXiv.2108.13897. arXiv:2108.13897.

[33] K. Spärck Jones, A statistical interpretation of term specificity and its application in retrieval, Journal of Documentation 28 (1972) 11–21. URL: https://doi.org/10.1108/eb026526. doi:10.1108/ eb026526.

[34] S. Robertson, H. Zaragoza, The probabilistic relevance framework: BM25 and beyond, Foundations and Trends in Information Retrieval 3 (2009) 333–389. URL: https://doi.org/10.1561/1500000019. doi:10.1561/1500000019.

[35] G. Amati, C. J. van Rijsbergen, Probabilistic models of information retrieval based on measuring the divergence from randomness, ACM Transactions on Information Systems 20 (2002) 357–389. URL: https://doi.org/10.1145/582415.582416. doi:10.1145/582415.582416.

[36] C. Zhai, J. D. Laferty, A study of smoothing methods for language models applied to information retrieval, ACM Transactions on Information Systems 22 (2004) 179–214. URL: https://doi.org/10. 1145/984321.984322. doi:10.1145/984321.984322.

[37] D. Zmitrovich, A. Abramov, A. Kalmykov, V. Kadulin, M. Tikhonova, E. Taktasheva, D. Astafurov, M. Baushenko, A. Snegirev, T. Shavrina, S. S. Markov, V. Mikhailov, A. Fenogenova, A family of pretrained transformer language models for Russian, in: Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 2024, pp. 507–524. URL: https://aclanthology.org/2024.lrec-main.45/.