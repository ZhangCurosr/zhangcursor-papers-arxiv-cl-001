## Highlights

Balancing hierarchical representation and quantization: A performance trade-of in text embeddings<sup>⋆</sup>

Kai Ponel, Marco Polignano, Erasmo Purificato, Aylin Julia Temizkan, Marco Levantesi, Giovanni Semeraro, Ernesto William De Luca

• A factorial benchmark of Matryoshka truncation and quantization on 38 MTEB tasks.

• 8-bit embeddings are statistically on par with float32 for several task types.

• One halving of embedding size is not significant; two successive halvings are.

• Eightfold truncation plus int8 keeps 94.4% efectiveness with 32× less storage.

• Retrieval is far more sensitive to compression than any other task category.

# Balancing hierarchical representation and quantization: A performance trade-of in text embeddings

Kai Ponel<sup>a,1</sup>, Marco Polignano<sup>b,e,∗</sup>, Erasmo Purificato<sup>c,1,2</sup>, Aylin Julia Temizkan<sup>d</sup>, Marco Levantesi<sup>e</sup>, Giovanni Semeraro<sup>b</sup>, Ernesto William De Luca<sup>e</sup>

<sup>a</sup>Ludwig-Maximilian-University Munich, Geschwister-Scholl-Platz 1, Munich, Germany

<sup>c</sup>European Commission, Joint Research Centre (JRC), Via E. Fermi 2749, Ispra, Italy <sup>d</sup>IBM Germany, Im Klapperhof 3-5, Cologne, Germany

<sup>e</sup>Otto von Guericke University Magdeburg, Universitätsplatz 2, Magdeburg, Germany

## Abstract

Hierarchical text embedding models ofer variable-length representations, while quantization reduces numerical precision; both aim to improve computational eficiency. This work is an empirical benchmark study of how these two orthogonal, already-established techniques interact. We evaluate 38 tasks from the Massive Text Embedding Benchmark (MTEB) using four transformer-based embedding models, covering 1D and 2D hierarchical representations combined with 32-bit, 8-bit, and binary quantization, for 168 configurations per task. We report each eficiency dimension separately: theoretical per-vector storage and end-to-end evaluation time. Under a Friedman test with Nemenyi post-hoc analysis, int8 quantization is statistically indistinguishable from float32 for pair classification and re-ranking and costs at most 1.5 percentage points overall, whereas binary quantization is significantly worse across every task category. Halving the embedding dimension once is not statistically detectable above m = 16; two successive halvings always are. Combining an 8× dimensionality reduction with int8 yields a 32× reduction in per-vector storage at 94.4% of baseline efectiveness. Because evaluation times were measured once, we treat them as indicative: we find no evidence that quantization speeds up evaluation, whereas sampling from intermediate layers shortens it substantially. Retrieval degrades far faster than every other category, so we report findings per task category.

Correction notice (afiliation of Marco Polignano). The published version of this article (Information Sciences 763 (2027) 124271, https://doi.org/10.1016/j.ins.2026. 124271) lists the European Commission, Joint Research Centre (JRC) as the second afiliation of Marco Polignano. This is an error in the final manuscript, identified by the authors after publication. The authors asked the publisher to correct it, but the publisher’s policy does not allow post-publication changes of this kind. Marco Polignano is not afiliated with the JRC and has no involvement with the JRC. His afiliations are the University of Bari Aldo Moro and Otto von Guericke University Magdeburg, as given in this version. The JRC afiliation applies only to Erasmo Purificato. Apart from this correction and this notice, the content of this version is identical to the published article.

## 1. Introduction

Learned representations, commonly called embeddings, represent diverse data types (e.g., words, documents) as real-valued vectors, organized such that semantically similar items are positioned close together within the embedding space [1]. Since Word2Vec [2], embeddings have advanced further, notably with BERT [3], which marked a shift towards context-dependent embeddings, and with Sentence-BERT [4] and SimCSE [5], designed to learn embeddings for entire sentences. These representations then power a wide spectrum of downstream tasks, including classification, clustering, and retrieval [6]. Retrieval Augmented Generation (RAG) [7] stands as a prominent example: a typical RAG pipeline involves chunking and embedding documents, retrieving relevant chunks based on similarity to a user query, and then generating a response from a large language model.

The use of embeddings at scale has two primary costs: (1) inference, which involves computing embeddings for new data, and is typically a one-time cost per element, as embeddings can be stored; and (2) downstream tasks, which often involve repeated comparisons between embeddings loaded into memory. An RAG index of one million chunks encoded as 1024-dimensional float32 vectors already occupies about 4 GB. Matryoshka Representations (MRs) [8, 9] ofer hierarchical embedding storage, enabling the selection of representation size, and with 2D MRs also of the layer, during inference. Additionally, quantization techniques [10, 11] aim to reduce the bit precision of embedding values, resulting in a smaller memory footprint.

While these techniques have been investigated individually, their combined efect on embedding performance and eficiency remains an open question. This work seeks to address this gap, aiming to answer the following research question: RQ: “In the context of transformer-based text embedding models, can hierarchical representations and quantization be combined to improve the computational eficiency of embeddings while maintaining high performance on downstream tasks?” Our research is guided by four objectives: to evaluate the combined impact of hierarchical representations and quantization across a diverse set of tasks, identifying which task categories are most resilient to dimensionality reduction and precision loss; to quantify the eficiency gains in per-vector storage and end-to-end evaluation time; to determine configurations of dimensionality, numerical precision, and inference layer that balance performance preservation with resource eficiency; and to establish practical guidelines for resource-constrained environments.

## 1.1. A worked example

Consider a retrieval component of a RAG pipeline that stores one million text chunks, each encoded by MBAI-1D (Section 4) into a $d = 1 0 2 4$ float32 vector. Table 1 traces a single query vector $q$ and two candidate vectors, one relevant $\left( d _ { 1 } \right)$ , one not $\left( d _ { 2 } \right)$ , through the two transformations studied in this paper. Truncating to the leading $m = 1 2 8$ dimensions and then mapping to int8 leaves the ranking of $d _ { 1 }$ above $d _ { 2 }$ intact while shrinking the index from 4.10 GB to 0.13 GB, a 32× reduction in stored bytes. Binary quantization compresses the same index to 0.02 GB but maps $q$ and $d _ { 1 }$ onto identical codes, so the distance between them collapses to zero and all ordering information among near neighbors is destroyed. The central empirical question is therefore not whether these transformations save space, but how much task efectiveness each one costs, and on which kinds of tasks.
<table><tr><td>Configuration</td><td>Vector excerpt (q)</td><td></td><td>sim(q, d1) / sim(q, d2)</td><td>Bits per vector</td><td>Index, 1M vectors</td></tr><tr><td>float32, m = 1024 (baseline)</td><td> $[ 0 . 6 2 , - 0 . 4 1 , 0 . 3 3 , 0 . 1 8 ]$ </td><td> $+ 1 . 0 0 \ / \ - 0 . 7 8$ </td><td></td><td>32,768</td><td>4.10 GB</td></tr><tr><td>float32, m = 128 (1D MR)</td><td>[0.62, −0.41, 0.33, 0.18]</td><td> $+ 1 . 0 0 \ / \ - 0 . 7 7$ </td><td></td><td>4096</td><td>0.51 GB</td></tr><tr><td>int8, m = 128 (MR + int8)</td><td>[121, -81, 64, 35]</td><td></td><td> $+ 1 . 0 0 \ / \ - 0 . 7 8$ </td><td>1024</td><td>0.13 GB</td></tr><tr><td>binary,  $m = 1 2 8 \ \mathrm { ( M R + 1 { - } b i t ) }$ </td><td>[1, 0, 1, 1]</td><td>Hamming 0 / 3</td><td></td><td>128</td><td>0.02 GB</td></tr></table>

Table 1: Worked example of the two transformations studied in this paper, applied to a query and two candidate chunks. Similarity is cosine similarity for the real-valued and integer rows and Hamming distance for the binary row. Vector values are an illustrative four-dimensional excerpt; bit counts are exact for $d = 1 0 2 4$ and $m = 1 2 8$

## 1.2. Contributions and positioning

We stress that this paper is an empirical benchmark and guideline study, not a proposal for a new model, training objective, or compression algorithm. Specifically:

1. A controlled factorial benchmark. We evaluate the full cross-product of hierarchical size, numerical precision, and (for 2D models) inference layer, with 168 configurations per task, over 38 MTEB tasks and four transformer-based embedding models.

2. Disaggregated eficiency accounting. We separate theoretical per-vector storage from measured end-to-end evaluation wall-clock time, and we show that the two do not move together.

3. Statistical validation. We subject the results to Friedman, Nemenyi, and Wilcoxon signed-rank testing, which shows, for example, that a single halving of the embedding dimension is not statistically detectable at any size above $m = 1 6$ , whereas two successive halvings always are.

4. Task-category-resolved guidelines. Because diferent MTEB categories react very diferently to compression, we report and recommend per category, and we identify retrieval as the category that constrains aggressive optimization.

Structure of the paper. Section 2 reviews hierarchical text representations and quantization strategies. Section 3 presents the methodology and Section 4 details the experimental settings. Section 5 reports the experimental results. Section 6 derives practical deployment guidelines. Section 7 states the limitations of the study and Section 8 concludes.

## 2. Background and related work

This section reviews the two lines of work our study combines, hierarchical text representations and embedding quantization. What has not previously been established is how they behave in combination: work on hierarchical representations varies dimensionality at full precision, while work on quantization varies precision at full dimensionality. Practitioners nonetheless apply both simultaneously, on the untested assumption that two orthogonal compression mechanisms compose without interacting. Whether that assumption holds is an empirical question that requires a factorial design to answer.

## 2.1. Hierarchical text representation

Traditional embeddings do not allow for flexibility when it comes to the dimensionality of the representation: all the d dimensions must be used in downstream tasks to avoid significant drops in performance [12], which has led to various attempts to develop more flexible representations [13]. Principal Component Analysis (PCA) [14, 15] projects highdimensional embeddings onto a lower-dimensional subspace, aiming to preserve as much variance as possible. It is fast and straightforward, but it is inherently a linear transformation, which limits its ability to capture complex semantic structures, and the projection must be fitted and stored for each corpus. Jointly-optimized subnetworks [16, 17] involve training multiple interconnected subnetworks within a larger framework, enabling the system to dynamically adjust the dimensionality of the embeddings. While these methods ofer greater flexibility, training such subnetworks requires substantial computational resources.

Matryoshka Representations. Matryoshka Representations (MRs), inspired by Russian nesting dolls, were introduced by Kusupati et al. [8]: for every $m \in \mathbb { M }$ , the first m dimensions of the embedding vector $z _ { 1 : m } \in \mathbb { R } ^ { m }$ can be used as a general purpose representation of the input data. M is composed of consistent reductions by half from the full embedding size; for $d = 2 0 4 8 .$ , for example, $\mathbb { M } = \{ 2 0 4 8 , 1 0 2 4 , . . . , 8 \}$ . MRs simultaneously optimize multiple loss functions across the specified dimensions, which encourages the model to prioritize essential information in the leading dimensions. This method requires minimal modifications to existing training pipelines, imposes no additional cost during inference, and matches or outperforms independently trained low-dimensional representations [8].

2D Matryoshka Representations. Although MRs greatly improve the eficiency of learned representations, the forward pass required to compute these embeddings is still expensive, since the model has to process the input through all layers. 2D MRs [9] allow the user to choose both layers and the size of the output embeddings at inference time. The authors of 2D MRs based their work on a pre-trained BERT model [3] with [CLS] pooling [5, 18]:

$$
X _ { 1 : n } ^ { m } = \mathrm { B E R T } _ { \mathrm { c l s } } ^ { 1 : n } ( x ) _ { 1 : m } \in \mathbb { R } ^ { m }\tag{1}
$$

where $n \in [ 1 , N ]$ denotes the n-th layer of the N-layer Transformer backbone and m is the dimension of the embedding vector. To address the low quality of the embeddings obtained from shallow layers, the authors align them with the final layer through divergence minimization.

## 2.2. Quantization strategies

In deep learning, quantization applies to two distinct areas. Model parameter quantization focuses on reducing the precision of weights and activations, through post-training quantization or quantization-aware training, to decrease memory usage and improve computational speed [19, 20]. In contrast, embedding quantization targets the floating-point vectors that encode semantic relationships between words or sentences, whose memory overhead dominates in large-scale applications involving billions of vectors.

The simplest approach involves floating-point downsampling to Float16 or BFloat16, which achieves a straightforward 2× memory reduction, often with negligible accuracy loss [21]. Converting embeddings to int8 reduces memory usage by 4×, and integer arith metic is both faster and more power-eficient than floating-point operations [10]. The most aggressive approach involves binary (1-bit) quantization, which yields a 32× memory reduction relative to float32; in practice, search over binary codes with Hamming distance can achieve approximately a 25× speedup compared to float32 search on the CPU, but binary quantization inevitably loses semantic nuance [11]. Alternative approaches include vector quantization methods, where embeddings are represented as discrete codes drawn from a learned codebook. Product quantization (PQ) [22] exemplifies this paradigm by dividing the embedding space into lower-dimensional subspaces and quantizing each subspace independently; together with its optimized variant (OPQ) [23], it is the standard compression method in approximate nearest neighbor search libraries such as Faiss [24].

Since quantization and Matryoshka are orthogonal techniques, we can first shrink an embedding via Matryoshka, then apply integer or binary quantization, compounding the memory savings; Huerga-Pérez et al. [21] study such combinations for RAG storage. Table 2 summarizes the trade-ofs of the main methods. The accuracy figures reported for them originate from diferent papers, corpora, and evaluation protocols, and therefore support qualitative positioning only; a controlled comparison of PQ, PCA, and float16 is outside the scope of this work (Section 7). What Table 2 does show is a structural distinction: PCA and PQ obtain their compression from a transformation that must first be fitted to a corpus and then stored and re-fitted under distributional drift, whereas hierarchical truncation is a property of how the model was trained and costs nothing at inference, so one stored vector serves every supported budget.

<table><tr><td>Method</td><td>Compression vs float32</td><td>Fitting required</td><td>Pros</td><td>Cons</td></tr><tr><td>Float16 BFloat16 [21]</td><td>2×</td><td>No</td><td>Plug-and-play; hardware-native on modern GPUs; small</td><td>Modest compression; bfloat16 trades precision for range</td></tr><tr><td>Int8 scalar quantization [10]</td><td>4×</td><td>Value ranges</td><td>Integer arithmetic is fast and energy-efficient; widely supported</td><td>Ranges must be estimated from data; outlier values reduce the resolution</td></tr><tr><td>Binary quantization [11]</td><td>32×</td><td>No</td><td>Extreme compression; fast Hamming-distance search on CPU; data-independent</td><td>Large semantic loss; close neighbors collapse to identical codes; usually needs rescoring</td></tr><tr><td>Product quantization (PQ, OPQ) [22, 23]</td><td>8-32×</td><td>Codebook</td><td>Strong compression-accuracy trade-off; mature nearest-neighbor search</td><td>Codebook must be trained on in-domain data and retrained under drift; more complex distance</td></tr><tr><td>PCA projection [14, 15]</td><td>2-10×</td><td>Projection</td><td>support Cheap to fit and apply; orthogonal to precision reduction</td><td>computation Linear, so it misses non-linear structure; projection must be stored</td></tr><tr><td>1D/2D MR + int8 (this study)</td><td>32× at m = d/8</td><td>No</td><td>Truncation is free at inference; one stored vector serves every size; composes with precision reduction</td><td>and refitted per corpus Requires a model trained with an MR objective; storage gains bring little speed gain; retrieval degrades at small sizes</td></tr></table>

Table 2: Pros and cons of the main embedding compression techniques. Compression ratios are relative to float32 storage of the same number of components; the last row summarizes the configuration measured in this study.

## 3. Methodology

We use datasets from the Massive Text Embedding Benchmark (MTEB) [6], which are considered standard in the literature, and we test each combination of hierarchical size and quantization technique for each model, together with the inference layer for the 2D model. The overall process is given step by step in Algorithm 1, and the corresponding component flow is shown in Fig. 1.

## 3.1. Notation

Table 3 collects the symbols used throughout the paper. Three factors are varied independently: the hierarchical size m, the quantization technique q, and, for 2D models only, the inference layer l. A configuration is a triple $( m , q , l )$ evaluated for a given model on a given task.

<table><tr><td>Symbol</td><td>Domain</td><td>Meaning</td></tr><tr><td> $F , f$ </td><td> $\mathrm { n / a }$ </td><td>Set of evaluated embedding models; an individual model</td></tr><tr><td> $N _ { f }$ </td><td>N</td><td>Number of transformer layers of model f</td></tr><tr><td> $d$ </td><td>N</td><td>Default (maximum) output dimensionality of a model</td></tr><tr><td>m</td><td>M</td><td>Hierarchical (truncated) embedding size,  $m \leq d$ </td></tr><tr><td>M</td><td> $\subset \mathbb { N }$ </td><td>Set of hierarchical sizes, successive halvings of d</td></tr><tr><td>z</td><td> $\mathbb { R } ^ { d }$ </td><td>Full embedding vector produced by a model</td></tr><tr><td> $z _ { 1 : m }$ </td><td> $\mathbb { R } ^ { m }$ </td><td>Matryoshka prefix: leading m components of</td></tr><tr><td>q</td><td>Q</td><td>Quantization technique</td></tr><tr><td> $Q$ </td><td> $\mathrm { n / a }$ </td><td>{float32, int8, binary}</td></tr><tr><td> $b ( q )$ </td><td>{32, 8, 1}</td><td>Bits per stored component under technique q</td></tr><tr><td>1</td><td> $L ( f )$ </td><td>Transformer layer used for inference</td></tr><tr><td> $L ( f )$ </td><td> $\mathrm { n / a }$ </td><td> $\{ N _ { f } \}$  for 1D models; {24, 20, 16, 12} for the 2D model</td></tr><tr><td> $T , t$ </td><td> $\mathrm { n / a }$ </td><td>Set of MTEB tasks; an individual task</td></tr><tr><td>C</td><td> $C$ </td><td>Task category (clustering, retrieval, STS, ...),  $| C | = 7$ </td></tr><tr><td> $S ( f , t , m , q , l )$ </td><td>R</td><td>Absolute task score under the task&#x27;s own main metric</td></tr><tr><td> $\hat { S }$ </td><td> $\mathbb { R } ^ { + }$ </td><td>Relative performance,  $S / S _ { \mathrm { b a s e } }$  (Eq. 3)</td></tr><tr><td> $\tau ( f , t , m , q , l )$ </td><td> $\mathbb { R } ^ { + }$ </td><td>Absolute end-to-end task evaluation time (seconds)</td></tr><tr><td>τ</td><td>R⁺</td><td>Relative evaluation time,  $\tau / \tau _ { \mathrm { b a s e } }$ </td></tr><tr><td>μ</td><td>N</td><td>Theoretical per-vector storage in bits,  $\mu = m \cdot b ( q )$ </td></tr><tr><td>base</td><td> $\mathrm { n / a }$ </td><td>Reference configuration  $( m = d ,$   $q =$  float32,  $l = N _ { f } )$ </td></tr><tr><td>k</td><td>N</td><td>Number of levels of the factor under test (Friedman)</td></tr><tr><td> $N _ { b }$ </td><td>N</td><td>Number of matched blocks (Friedman)</td></tr><tr><td> $\bar { R } _ { j }$ </td><td>R</td><td>Mean rank of level  $j ;$  rank 1 denotes the best level</td></tr><tr><td>CD</td><td>R⁺</td><td>Nemenyi critical difference at  $\alpha = 0 . 0 5$ </td></tr><tr><td> $W$ </td><td>[0,1]</td><td>Kendall&#x27;s coefficient of concordance</td></tr></table>

Table 3: Notation used throughout the paper.

## 3.2. Evaluation pipeline architecture

Every configuration $( f , l , m , q )$ is evaluated in a separate MTEB run, in which the benchmark calls the embedding model on the texts of each task, so embeddings are never cached across configurations. Each run repeats the forward pass through the first l transformer layers, pools the output, keeps its leading m components and quantizes them before the task evaluator computes the score, so the recorded evaluation time always includes embedding generation. Because truncation and quantization are applied after pooling, the encoder computation is identical for all hierarchical sizes and precisions at a given layer. For traditional 1D MRs, there is only one inference layer, which is the final layer of the model, whereas for 2D MRs there are multiple inference layers that can serve as the output layer of the respective submodel, so the number of runs per task is

$$
| \mathbb { M } | \times | Q | \times \left( | F _ { 1 D } | + | F _ { 2 D } | \times | L | \right) ,\tag{2}
$$

where $| F _ { 1 D } |$ is the number of models supporting 1D hierarchical representations, $| F _ { 2 D } |$ is the number of models supporting 2D MRs, and $| L |$ is the number of inference layers for 2D MRs.

![](images/4bff67430e95639316d7caf679f98fe56b5c5cfa531a24febb189514e6e64a8b.jpg)  
Fig. 1: Component flow of the evaluation pipeline. Stages A to C are executed anew for every configuration $( f , l , m , q )$ within one MTEB run; embeddings are not cached across configurations, so the wall-clock time τ recorded for each task covers embedding generation, truncation and quantization, and task evaluation. Stage D normalizes every measurement against the model’s own baseline configuration and feeds the aggregated results into the statistical analysis in Section 5.3.

## 3.3. Evaluation metrics

For each configuration we collect three quantities, which we keep separate throughout the paper.

1. Relative performance. The task’s own main metric (Section 3.4), normalized by the score of the same model’s baseline configuration on the same task:

$$
\hat { S } ( f , t , m , q , l ) ~ = ~ { \frac { S ( f , t , m , q , l ) } { S ( f , t , d , \mathrm { { \scriptsize ~ d o a t } } 3 2 , N _ { f } ) } }\tag{3}
$$

The baseline configuration uses the highest-dimensional embeddings, no quantization, and the final layer for inference. Normalization is applied per task, before any aggregation, so that tasks measured on diferent scales contribute comparably.

Algorithm 1 Factorial evaluation of hierarchical representation × quantization   
Require: Models $F ;$ tasks $T ;$ quantization set $Q = \{ \mathrm { H o a t 3 2 , i n t 8 , b i n a r y } \}$ ; hierarchical sizes $\mathbb { M } ( f ) ;$ inference   
layers $L ( f )$   
Ensure: Result set $\mathcal { R }$ of tuples $( f , t , l , m , q , \hat { S } , \hat { \tau } , \mu )$   
1: $\mathcal { R }  \emptyset$   
2: for each model $f \in F$ do   
3: $L ( f ) \gets \{ N _ { f } \}$ if $f$ is a 1D MR model else {24, 20, 16, 12}   
4: for each configuration $( q , l , m ) \in Q \times L ( f ) \times \mathbb { M } ( f )$ do   
5: $g \gets \mathrm { C O N F I G U R E } ( f , l , m , q )$ ▷ first l layers; output cut to $m ,$ quantized to q   
6: for each task $t \in T$ do ▷ one MTEB run per configuration   
7: $\tau _ { 0 } \gets \mathrm { C L O C K } ( )$   
8: $S \gets \mathrm { E v a L U A T E M T E B } ( t , g )$ ▷ encodes the texts of t with Embed   
9: $\tau  \mathrm { C L O C K } ( ) - \tau _ { 0 }$ ▷ includes embedding generation   
10: $\mu  m \cdot b ( q )$ ▷ theoretical per-vector storage, bits   
11: $\mathcal { R } \gets \mathcal { R } \cup \{ ( f , t , l , m , q , S , \tau , \mu ) \}$   
12: end for   
13: end for   
14: end for   
15: Normalization:   
16: for each $( f , t )$ pair in $\mathcal { R }$ do   
17: $( S _ { \mathrm { b a s e } } , \tau _ { \mathrm { b a s e } } ) $ entry with m=d, q=float32, $\scriptstyle { l = N _ { f } }$   
18: $\hat { S } \gets S / S _ { \mathrm { b a s e } } \ ; \quad \hat { \tau } \gets \tau / \tau _ { \mathrm { b a s e } }$ ▷ Eq. 3   
19: end for   
20: Aggregation and analysis:   
21: Group S,<sup>ˆ</sup> τˆ by task category $c \in C$ and report per category   
22: Friedm $\mathrm { i N } ( \tilde { S } )$ over each factor in $\{ q , ~ m , ~ l \}$ , blocking on the other two and on c   
23: if the null is rejected then Nemenyi post-hoc + Wilcoxon signed-rank with Holm correction   
24: function Embed(g, texts) ▷ repeated in every run, never cached   
25: $Z $ pooled output of the first l layers of $f$   
26: $Z _ { m } \gets Z [ :$ 1:m] ▷ Matryoshka truncation   
27: return $\mathrm { Q U A N T I Z E } ( Z _ { m } , q )$   
28: end function   
29: function $\mathrm { Q U A N T I Z E } ( Z _ { m } , q )$ ▷ Sentence-Transformers quantize\_embeddings   
30: if q = float32 then   
31: return $Z _ { m }$   
32: else if q = int8 then   
33: for each dimension $j \colon l o _ { j } \gets \operatorname* { m i n } _ { i } Z _ { i j } , h i _ { j } \gets \operatorname* { m a x } _ { i } Z _ { i j } , s _ { j } \gets ( h i _ { j } - l o _ { j } ) / 2 5 5$   
34: return trunc $( ( Z _ { m } - l o ) / s - 1 2 8 )$ as signed 8-bit integers   
35: else if q = binary then   
36: return $\mathbf { 1 } [ Z _ { m } > 0 ]$ packed eight to a byte ▷ evaluated as numeric vectors   
37: end if   
38: end function

2. Evaluation time. We measure the end-to-end wall-clock time to complete one MTEB task, denoted τ, and normalize it as $\hat { \tau } = \tau / \tau _ { \mathrm { b a s e } }$ . It comprises embedding generation, truncation, and quantization; the task-specific downstream computation (the similarity search for retrieval and re-ranking, the logistic-regression probe for classification, the k-means fit for clustering, and the pairwise similarity computation for the other categories); and the MTEB harness overhead. The encoder forward pass dominates the evaluation time of most task categories and is identical for every hierarchical size and precision at a given layer, so diferences between such configurations are diluted by the common encoding cost, whereas changing the inference layer changes the encoder cost itself. Our timing results are therefore benchmark throughput figures rather than per-query inference latency.

3. Theoretical per-vector storage. The number of bits required to store one embedding vector:

$$
\mu \ = \ m \times b ( q ) , \quad \quad b ( \mathrm { H o a t 3 2 } ) = 3 2 , b ( \mathrm { i n t 8 } ) = 8 , b ( \mathrm { b i n a r y } ) = 1\tag{4}
$$

This quantity is exact and analytic, not measured. It does not transfer one-to-one to resident system memory, which also includes the model weights, the framework overhead, any index structure, and, for two-stage pipelines, the higher-precision copies retained for rescoring; in a system dominated by vector storage, such as a large RAG index, µ is a good proxy for the dominant term. Where we report memory reductions in this paper, we mean reductions in µ.

## 3.4. Benchmark and tasks

We use MTEB [6], whose public leaderboard<sup>3</sup> ranks models across task categories. MTEB includes 58 benchmarks across 8 diferent task categories; we use the seven categories below and exclude bitext mining, which is multilingual.

• Classification: assigning a text to a predefined category after training a logistic regression model; accuracy.

• Clustering: grouping texts with the k-means algorithm; V-Measure [25].

• Pair Classification: assigning a label to a pair of texts, indicating their relationship, based on their cosine similarity; Average Precision (AP) [26].

• Re-ranking: given a query and a set of relevant and irrelevant texts, ranking the texts so that the most relevant ones appear higher; Mean Average Precision (MAP).

• Retrieval: retrieving the most relevant documents from a corpus in response to a query; Normalized Discounted Cumulative Gain (NDCG) [27].

• Semantic Textual Similarity (STS): measuring the degree of similarity between a pair of sentences; Spearman correlation between their cosine similarity and human judgments.

• Summarization: assessing machine-generated summaries by comparing them with human-written references; Spearman correlation between their cosine similarity and human ratings.

## 4. Experimental settings

## 4.1. Model selection, embedding size, and quantization strategy

Kusupati et al. [8] demonstrated that models not trained using the MR loss do not perform well when a subspace of the embedding vector is used, and similar findings have been reported for 2D MRs [9]. Therefore, we selected four transformer-based text embedding models trained using either the 1D or 2D MR loss, with fewer than 1B parameters due to limited computational resources:

• Stella: the $s t e l l a \_ e n \_ \not { q } { 0 0 M } \_ v 5 ^ { 4 }$ model [28] is a transformer model with 400 million parameters, distilled from the gte-large-en-v1.5 and gte-Qwen2-1.5B-instruct models and trained using the MR objective, with a default output dimensionality of 8192.

• MBAI-1D: the mxbai-embed-large-v1 <sup>5</sup> model [18] is a transformer model with 335 million parameters, released by Mixedbread-AI. It is designed for English language tasks, has 24 layers, and a default embedding size of 1024. The model has been trained using the MR objective and contrastive learning on 700 million data pairs and over 30 million triplets using the AnglE objective. The company also notes that this model is well suited for binary embeddings [11].

• Nomic: the nomic-embed-text-v1.5<sup>6</sup> model [29] is a transformer model with 137 million parameters, trained using the MR objective with a default output dimensionality of 768.

• MBAI-2D: the mxbai-embed-2d-large-v1 <sup>7</sup> model [9] is a transformer model with 335 million parameters, released by Mixedbread-AI. In contrast to the other models, this model also supports 2D MRs; besides the adjusted training objective, the model is similar to MBAI-1D in terms of architecture and training data.

For each model, we use eight hierarchical sizes, starting with the maximum output dimensionality of the model: M = {8192, 4096, . . . , 64} for Stella, {1024, 512, . . . , 8} for MBAI-1D and MBAI-2D, and {768, 512, 256, . . . , 8} for Nomic. For MBAI-2D, the only 2D MR model, we evaluate the intermediate layers 20, 16, and 12, based on the gradual degradation in performance observed in the original 2D MRs paper [9], as well as the final layer 24.

Our experiments use (signed) int8 and binary quantization alongside the default float32 embeddings; in our implementation the conversion is performed by the quantize\_embeddings function of Sentence-Transformers. The conversion from floating-point to int8 is done using bucketing, which divides the range of each dimension, from its minimum to its maximum over the embeddings being quantized, into 256 equally sized buckets, with each bucket represented by a single int8 number; no separate calibration set is used. Conversion to binary numbers is done by setting the binary number to 1 if the floating-point number is positive and to 0 otherwise. The resulting bits are packed eight to a byte, so a binary embedding of size m occupies m/8 bytes, in agreement with Eq. 4. The packed codes are passed to the

MTEB evaluators, which process them as ordinary numeric vectors; the Hamming-distance search used by dedicated binary indexes was not part of our protocol (Section 7).

## 4.2. Dataset and task selection criteria

All tasks that are part of the MTEB subset used for the Hugging Face leaderboard were considered. Then, tasks that were too computationally expensive were excluded. Since seven model configurations (three 1D models and the 2D model with four layers) are evaluated for eight hierarchical sizes and three quantization techniques, resulting in a single task being repeated 168 times, we removed any task taking longer than 15 min on any model to complete in its default configuration. The resulting 38 tasks (Table S1 of the Supplementary Material) span scientific literature, news, social media, technical forums, and customer service: classification (Banking77, Emotion, TweetSentimentExtraction, AmazonCounterfactual, MassiveIntent, MassiveScenario, MTOPDomain, MTOPIntent); clustering (the P2P and S2S variants of ArXivHierarchical, Biorxiv, Medrxiv, and StackExchange, together with Reddit and TwentyNewsgroups); pair classification (SprintDuplicateQuestions, TwitterSemEval2015, TwitterURLCorpus); re-ranking (AskUbuntuDupQuestions, MindSmallRerank ing, StackOverflowDupQuestions); retrieval (ArguAna, CQADupstackWebmastersRetrieval, NFCorpus); STS (BIOSSES, SICK-R, STS12 to STS17, STS22, STSBenchmark); and summarization (SummEval). Clustering groups either titles (S2S) or titles with abstracts or full posts (P2P) of scientific papers, forum posts, and news. Classification ranges from intent detection with up to 77 classes (Banking77) to emotion detection with six classes and counterfactual detection in product reviews. Pair classification detects duplicate questions and paraphrases; re-ranking orders candidate questions from technical forums and candidate news articles; retrieval covers counter-argument retrieval, duplicate questions from a webmasters forum, and medical documents. SummEval contains 1600 summaries of news articles paired with human expert ratings across four dimensions (coherence, consistency, fluency, relevance). The evaluations are performed on the respective test sets of the tasks, following the MTEB leaderboard methodology. Due to GPU memory problems, the Nomic model could not be evaluated on MedrxivClusteringP2P.v2, STS22, CQADupstackWebmastersRetrieval, and NFCorpus.

Impact of this filtering on the conclusions. This criterion is not neutral. Task runtime in MTEB is driven overwhelmingly by corpus size, so the large retrieval benchmarks, among them MSMARCO, Natural Questions, HotpotQA, DBPedia, FEVER, ClimateFEVER, QuoraRetrieval and TRECCOVID, are all excluded, and retrieval is represented by three of the smallest corpora. The global averages therefore under-weight the workload that embedding compression is most often deployed for. Moreover, as the number of candidate documents increases, the margin separating the correct document from its nearest distractors shrinks, so a given amount of compression noise is more likely to reverse a ranking. Our retrieval findings, already the most pessimistic of any category, should accordingly be read as a lower bound on the degradation at larger scales.

## 4.3. Software and hardware configuration

All experiments are conducted using Python, with PyTorch, the Hugging Face Transformers and Sentence-Transformers libraries, and MTEB, in Jupyter notebooks, which can be accessed via a public GitHub repository<sup>8</sup>. The code is executed on Google Colab Pro, and all experiments are conducted on an NVIDIA Tesla T4 GPU with 16 GB of video memory; all notebooks have 50 GB of system RAM available.

## 5. Experimental results

We split the research question into the following sub-research questions for 1D and 2D MRs:

• RQ-1D-1 / RQ-2D-1: How does the combination of 1D (2D) MRs and quantization afect the performance across tasks?

• RQ-1D-2 / RQ-2D-2: How does the combination of 1D (2D) MRs and quantization afect the evaluation time across tasks?

• RQ-1D-3 / RQ-2D-3: What is the trade-of between performance and per-vector storage (and, for 2D MRs, evaluation time)?

A note on the timing results. Every evaluation time comes from a single MTEB run per configuration on a shared Google Colab runtime, whose run-to-run variability we could neither control nor measure. We therefore treat timing figures as indicative and interpret only timing efects that are large and consistent across configurations and task categories: sampling from an intermediate layer shortens evaluation substantially, and we find no evidence that int8 quantization brings a meaningful speed-up.

## 5.1. Quantized 1D MRs

Performance (RQ-1D-1). Following the approach used in the MTEB leaderboard, we rely on the simple average of relative performance across all the selected tasks; this score is simple to understand, but the mean can give excessive weight to tasks with higher variance in their scores. Full results and tables are available in our GitHub repository<sup>9</sup>. For the float32 embeddings of MBAI-1D (Fig. 2a), the reductions from 1024 down to 128 cost only 4.8% of the original performance; as the MRs become smaller, the relative loss in performance increases, reaching 11% at 64 dimensions. The int8 embeddings almost match the performance of the float32 ones, falling behind by only 6.6% at most. The binary embeddings perform worse at all hierarchical sizes, and at m = 1024 the diference already exceeds 17%.

Table 4(a) reveals significant task-specific variations that are masked by the averaging process. Retrieval exhibits catastrophic performance degradation, dropping to just 0.06 at 8d for both float32 and int8 and completely failing (0.00) for binary embeddings, whereas Pair Classification maintains 70–71% performance at 8d for float32/int8. Clustering shows excellent resilience with binary embeddings, maintaining 50% performance at 8d compared to 57% for float32 and int8. The int8 embeddings maintain performance very close to that of float32 across all tasks, with perfect parity at 1024d for Clustering, Pair Classification, and Retrieval. Nomic (Fig. 2c) supports all the observations made so far, while Stella (Fig. 2b) shows a slightly diferent behavior: the performance remains constant for all quantization techniques when reducing the output embedding size from 8192 to 1024, which suggests that the model efectively uses a dimensionality of 1024. Due to insuficient documentation of Stella’s training procedure, we acknowledge this deviation without overemphasizing it.

Evaluation time (RQ-1D-2). With the caveat stated above, reducing the hierarchical size is associated with shorter evaluation times, most visibly for classification and clustering (up to 27%; Table 4c), for which downstream classifiers and clustering algorithms operate directly on the embeddings. Int8 embeddings are only up to 3 percent faster than float32 embeddings on average (Fig. 2g), and Nomic is the only model in which the int8 embeddings are not faster than the float32 ones for all hierarchical sizes (Fig. 2i), so we find no evidence of a meaningful speed-up from int8. Binary embeddings were faster than float32 ones in our runs, by a margin that we report as indicative. Retrieval runs with int8 and binary embeddings took 0.89 times the baseline time at every dimension, but the reduction is identical for every embedding size and for both precisions, while the float32 retrieval runs show no dependence on embedding size, which more plausibly reflects a single-run measurement than an efect of quantization. The absolute times are given in Figure S1 of the Supplementary Material.

Storage trade-of (RQ-1D-3). Fig. 2(d)–(f) plot per-vector storage against average relative performance. For all budgets between $2 ^ { 1 3 }$ and $2 ^ { 6 }$ bits, there are multiple combinations requiring the same amount of memory, for example float32 with $m = 2 5 6$ and int8 with $m = 1 0 2 4 { \mathrm { ~ a t ~ } } 2 ^ { 1 3 } $ bits, and the int8 embeddings provide the best performance of all quantization techniques tested. For instance, the int8 representation for $m = 2 5 6$ achieves a relative performance of 96% while only using $2 ^ { 1 1 }$ bits, 16 times less than the baseline. The performance of binary embeddings is exceeded by int8 quantization for all memory budgets between $2 ^ { 1 0 }$ and $2 ^ { 6 }$ bits; with $m = 3 2$ , 16, and 8 they reach budgets that no other configuration reaches, but they retain less than 50% of the performance. These characteristics are consistent for all task categories except for classification, where float32 outperforms int8 at equal storage by almost 4.5%; for all other tasks, int8 is at most 1.8% worse than float32. For Stella, the flat behavior between 8192 and 1024 dimensions improves the trade-of: int8 with $m = 5 1 2$ only performs 1.8% worse than float32 with $m = 8 1 9 2$ while reducing the bits required by a factor of 64. Nomic shows the same characteristics as MBAI-1D.

Answers to 1D MRs. Concluding RQ-1D-1, both the reduction of embedding size and quantization lead to performance decreases whose magnitude varies by task category, and Retrieval is particularly vulnerable. Concluding RQ-1D-2, reducing the hierarchical size is associated with shorter evaluation times for classification and clustering, whereas we find no evidence that int8 is faster than float32. Answering RQ-1D-3, int8 embeddings present the best trade-of between performance and memory consumption, whereas binary representations ofer an even greater reduction at the cost of a significant decrease in performance.

(a) 1D MRs: MBAI-1D by hierarchical size and precision, relative to float32 at 1024 dimensions
<table><tr><td></td><td colspan="4">Float32</td><td colspan="4">Int8</td><td colspan="4">Binary</td></tr><tr><td>Task category</td><td>1024d</td><td>256d</td><td>64d</td><td>8d</td><td>1024d</td><td>256d</td><td>64d</td><td>8d</td><td>1024d</td><td>256d</td><td>64d</td><td>8d</td></tr><tr><td>Average</td><td>1.00</td><td>0.97</td><td>0.89</td><td>0.61</td><td>0.99</td><td>0.96</td><td>0.86</td><td>0.58</td><td>0.83</td><td>0.63</td><td>0.47</td><td>0.30</td></tr><tr><td>Classification</td><td>1.00</td><td>0.97</td><td>0.87</td><td>0.53</td><td>0.96</td><td>0.92</td><td>0.78</td><td>0.45</td><td>0.79</td><td>0.51</td><td>0.37</td><td>0.24</td></tr><tr><td>Clustering</td><td>1.00</td><td>0.97</td><td>0.85</td><td>0.57</td><td>1.00</td><td>0.97</td><td>0.84</td><td>0.57</td><td>0.79</td><td>0.59</td><td>0.52</td><td>0.50</td></tr><tr><td>STS</td><td>1.00</td><td>1.00</td><td>0.96</td><td>0.79</td><td>0.99</td><td>0.99</td><td>0.95</td><td>0.76</td><td>0.93</td><td>0.81</td><td>0.52</td><td>0.20</td></tr><tr><td>Pair Class.</td><td>1.00</td><td>0.99</td><td>0.97</td><td>0.70</td><td>1.00</td><td>0.99</td><td>0.97</td><td>0.71</td><td>0.95</td><td>0.83</td><td>0.53</td><td>0.33</td></tr><tr><td>Retrieval</td><td>1.00</td><td>0.92</td><td>0.67</td><td>0.06</td><td>1.00</td><td>0.92</td><td>0.66</td><td>0.06</td><td>0.49</td><td>0.13</td><td>0.04</td><td>0.00</td></tr><tr><td>Re-ranking</td><td>1.00</td><td>0.98</td><td>0.91</td><td>0.74</td><td>0.99</td><td>0.97</td><td>0.92</td><td>0.72</td><td>0.88</td><td>0.76</td><td>0.65</td><td>0.29</td></tr><tr><td>Summarization</td><td>1.00</td><td>0.94</td><td>0.92</td><td>0.81</td><td>0.98</td><td>0.92</td><td>0.90</td><td>0.78</td><td>0.95</td><td>0.86</td><td>0.96</td><td>0.60</td></tr></table>

(b) 2D MRs: MBAI-2D at layers 24 and 12 (m = 1024), relative to float32 at layer 24
<table><tr><td rowspan="2">Task category</td><td colspan="3">Float32</td><td colspan="3">Int8</td><td colspan="3">Binary</td></tr><tr><td>L24</td><td>L12</td><td> $\Delta$ </td><td>L24</td><td>L12</td><td>∆</td><td>L24</td><td>L12</td><td>∆</td></tr><tr><td>Average (all tasks)</td><td>1.000</td><td>0.862</td><td>-0.138</td><td>0.986</td><td>0.853</td><td>-0.133</td><td>0.849</td><td>0.738</td><td>-0.111</td></tr><tr><td>Classification</td><td>1.000</td><td>0.829</td><td>-0.171</td><td>0.950</td><td>0.799</td><td>-0.151</td><td>0.786</td><td>0.642</td><td>-0.144</td></tr><tr><td>Clustering</td><td>1.000</td><td>0.951</td><td>-0.049</td><td>0.994</td><td>0.945</td><td>-0.049</td><td>0.831</td><td>0.839</td><td>+0.008</td></tr><tr><td>STS</td><td>1.000</td><td>0.861</td><td>-0.139</td><td>0.991</td><td>0.847</td><td>-0.144</td><td>0.934</td><td>0.790</td><td>-0.144</td></tr><tr><td>Pair Class.</td><td>1.000</td><td>0.872</td><td>-0.128</td><td>1.000</td><td>0.874</td><td>-0.126</td><td>0.949</td><td>0.799</td><td>-0.150</td></tr><tr><td>Retrieval</td><td>1.000</td><td>0.602</td><td>-0.398</td><td>1.002</td><td>0.619</td><td>-0.383</td><td>0.632</td><td>0.291</td><td>-0.341</td></tr><tr><td>Re-ranking</td><td>1.000</td><td>0.890</td><td>-0.110</td><td>0.997</td><td>0.891</td><td>-0.106</td><td>0.879</td><td>0.823</td><td>-0.056</td></tr><tr><td>Summarization</td><td>1.000</td><td>0.901</td><td>-0.099</td><td>1.008</td><td>0.933</td><td>-0.075</td><td>0.946</td><td>0.897</td><td>-0.049</td></tr></table>

(c) 1D MRs: MBAI-1D relative evaluation time, relative to float32 at 1024 dimensions (single-run measurements)
<table><tr><td></td><td colspan="4">Float32</td><td colspan="4">Int8</td><td colspan="4">Binary</td></tr><tr><td>Task category</td><td>1024d</td><td>256d</td><td>64d</td><td>8d</td><td>1024d</td><td>256d</td><td>64d</td><td>8d</td><td>1024d</td><td>256d</td><td>64d</td><td>8d</td></tr><tr><td>Average</td><td>1.00</td><td>0.91</td><td>0.89</td><td>0.87</td><td>0.99</td><td>0.90</td><td>0.86</td><td>0.86</td><td>0.87</td><td>0.86</td><td>0.85</td><td>0.84</td></tr><tr><td>Classification</td><td>1.00</td><td>0.81</td><td>0.75</td><td>0.73</td><td>1.01</td><td>0.82</td><td>0.76</td><td>0.74</td><td>0.76</td><td>0.75</td><td>0.73</td><td>0.73</td></tr><tr><td>Clustering</td><td>1.00</td><td>0.83</td><td>0.77</td><td>0.75</td><td>1.02</td><td>0.83</td><td>0.77</td><td>0.76</td><td>0.80</td><td>0.76</td><td>0.75</td><td>0.73</td></tr><tr><td>STS</td><td>1.00</td><td>1.00</td><td>0.99</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.99</td><td>1.00</td><td>0.99</td></tr><tr><td>Pair Class.</td><td>1.00</td><td>0.96</td><td>0.95</td><td>0.94</td><td>1.03</td><td>0.97</td><td>0.95</td><td>0.95</td><td>0.96</td><td>0.95</td><td>0.95</td><td>0.94</td></tr><tr><td>Retrieval</td><td>1.00</td><td>0.99</td><td>1.00</td><td>1.00</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.89</td></tr><tr><td>Re-ranking</td><td>1.00</td><td>0.99</td><td>1.02</td><td>1.01</td><td>0.92</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.89</td></tr><tr><td>Summarization</td><td>1.00</td><td>1.00</td><td>1.05</td><td>1.04</td><td>0.92</td><td>0.91</td><td>0.91</td><td>0.91</td><td>0.91</td><td>0.91</td><td>0.92</td><td>0.90</td></tr></table>

Table 4: Relative performance and evaluation time by task category. (a) Bold marks the best value in each row; underlined values mark the collapse of retrieval at 8 dimensions. (b) The ∆ columns give the change from layer 24 to layer 12 for each precision; the Average row is computed over all individual tasks and therefore does not equal the unweighted mean of the category rows. (c) Bold marks the shortest time in each row; timings are indicative (Section 5).

![](images/9a030782c90c8e0557f7ea98b2efc555f3e19c1980146584dbb29448c0ae80c2.jpg)  
(a) MBAI-1D: performance

![](images/a12b1174ddf34fd78f0dbace7a6e1a8154f2ddaade3b959f5f950bdd6164f51c.jpg)  
(b) Stella: performance

![](images/d01ba21137b9e89e3521fbf04701be5233251a986e4d928c3165cca1f99582ec.jpg)  
(c) Nomic: performance

![](images/ed3fbfdab93f9241acd5a70e8036b9e8d4856cf732ea9dba61f9aa549d52efca.jpg)  
(d) MBAI-1D: storage trade-of

![](images/914272d65707b1934176c62ed9df93db59dc68520fa1e444891ca5eceafb4466.jpg)  
(e) Stella: storage trade-of

![](images/496bbbd1cc63369ce3832b55f7465450c27f1943f2c84fa14bf1c52e013f5cc7.jpg)  
(f) Nomic: storage trade-of

![](images/0ef5453b7dbb78ac53b4d1eceb4ff40694408a37724e85d333d3c41c9d374885.jpg)  
(g) MBAI-1D: evaluation time

![](images/9bc6c424fadf75c8cedcc6eb7892e08b4186fd8cbd6a094b97fade0743c489a5.jpg)  
(h) Stella: evaluation time

![](images/702c77175d284980075488896d6e93d9050b0d7c6b34ec758f1c5ec895590817.jpg)  
(i) Nomic: evaluation time  
Fig. 2: Results for the 1D models over all selected MTEB tasks. (a)–(c) Average relative performance as a function of the hierarchical size for float32 (blue), int8 (orange), and binary (green) embeddings. (d)–(f) Average relative performance versus theoretical per-vector storage (bits, logarithmic scale); marker colors denote the precision and marker shapes the hierarchical size. (g)–(i) Average relative evaluation time as a function of the hierarchical size (single-run measurements). The reference point is each model’s float32 baseline at full size.

## 5.2. Quantized 2D MRs

Performance (RQ-2D-1). When only considering the final layer, the performance of the MBAI-2D model is within a 1% margin of the MBAI-1D model for each combination of embedding size and quantization technique (Fig. 3a). Table 4(b) compares layers 24 and 12 at $m = 1 0 2 4$ . Retrieval shows by far the largest degradation, losing 39.8% for float32, 38.3% for int8 and 34.1% for binary, roughly three times the average loss of 13.8%. At the opposite extreme, Clustering loses only 4.9%, Summarization 9.9%, Re-ranking 11.0%, and Classification 17.1%. This ordering is consistent with the 1D results, where Retrieval was likewise the category least tolerant of compression; we discuss the reason in Section 6.1. The int8 embeddings track float32 closely across all categories, and the two cases in which they marginally exceed the float32 baseline at layer 24 (Retrieval at 1.002 and Summarization at 1.008) are within the noise. Categories already heavily damaged by binarization at layer 24, such as Retrieval at 0.632, have comparatively less left to lose, so their $\Delta$ values understate how unusable those configurations are. The degradation is also not strictly monotonic in layer depth: layer 16 ranks significantly better than layer 20 (Section 5.3).

Evaluation time (RQ-2D-2). The evaluation time decreases approximately linearly as the chosen layer for inference moves closer to the input layer (Fig. 3b), which is expected because the number of executed transformer blocks falls proportionally. The same decrease is observed for each task category, mirroring the findings of the 2D MRs paper [9]. The evaluation times for float32 and int8 embeddings are almost identical, binary embeddings take slightly less, and the hierarchical size has a much smaller efect than the layer.

Trade-of (RQ-2D-3). In Fig. 3(c), between 0.87 and 1.0 on the x-axis only results from the 24th layer are found, between 0.7 and 0.87 only results from the 20th layer, and so on. Full-dimensional float32 embeddings inferred from layer 12 achieve a relative performance of 86.2% while only requiring 59% of the evaluation time. With int8 embeddings and m = 128, layer 12 maintains 81.9% of the performance, requires only 47.8% of the evaluation time, and reduces the memory consumption by a factor of 32; selected configurations are listed in Table S2 of the Supplementary Material. Retrieval tasks provide the worst trade-of between performance and evaluation time, and clustering tasks the best.

![](images/55df8f817078552467538273b2abb774c15d8075b0ade73703bb85ca2a195cf7.jpg)  
(a) Average relative performance

![](images/d1f5636857236427133a87d17648d7bd96ab61376577453d268cd6e05235adc2.jpg)  
(b) Average relative evaluation time

![](images/37441297741cb203c30c4dd90c86cf72d00c2a68e5f0ec71ab20906b14c0b625.jpg)  
(c) Trade-of among storage, evaluation time, and performance

Fig. 3: Results for MBAI-2D over all selected MTEB tasks. In (a) and (b), line colors denote the inference layer (red: 24, yellow-green: 20, turquoise: 16, purple: 12) and line styles denote the precision (solid: float32, dashed: int8, dotted: binary). In (c), marker colors denote the precision and the inference layer, and marker shapes denote the hierarchical size; the reference point (1, 1) is the float32 baseline at full size from the final layer. Evaluation times are single-run measurements.

Answers for 2D MRs. Concluding RQ-2D-1, the performance of the MBAI-2D model decreases as the layer used for inference moves closer to the input layer, with substantial task-specific variation, while the precision and size efects observed for 1D MRs hold at every layer. Concluding RQ-2D-2, the evaluation time decreases approximately linearly as the layer number decreases, and precision and size have much smaller efects. Concluding RQ-2D-3, given a moderate decrease in performance, the evaluation time was roughly halved in our measurements, while the per-vector storage was reduced by an exact factor of 32.

## 5.3. Statistical significance analysis

Averages alone cannot establish whether an observed diference is a real efect or an artifact of which tasks happen to be in the benchmark. Following Demšar [30], we use non-parametric tests, because relative performance is a bounded ratio whose distribution is neither normal nor homoscedastic across task categories, which also use diferent underlying metrics. For each of the three experimental factors, precision q, hierarchical size $m .$ and inference layer $l ,$ we apply a Friedman test [31] in which every combination of the remaining factors with the task category provides a matched block. Where the omnibus null hypothesis is rejected we apply the Nemenyi post-hoc test [32], with critical diference $\mathrm { C D } = q _ { \alpha } \sqrt { k ( k + 1 ) / 6 N _ { b } }$ , and pairwise Wilcoxon signed-rank tests [33] with Holm correction [34]. All tests use $\alpha = 0 . 0 5$ and the full MBAI-2D grid of 96 configurations, the only model for which all three factors vary. Because the blocks share task categories they are not fully independent, so we rely on the efect sizes and the critical-diference verdicts, not on the magnitude of $p .$ Table 5 summarizes the results; the pairwise Nemenyi comparison of the precisions is detailed in Table S3 of the Supplementary Material.

All three factors matter, but not equally. All three omnibus nulls are rejected (Table 5a), but Kendall’s W separates them: the orderings induced by precision $( W = 0 . 7 7 4 )$ and by hierarchical size $( W = 0 . 8 7 1 )$ are highly consistent from block to block, whereas the ordering induced by the inference layer is much less so $( W = 0 . 3 8 3 )$

Precision: int8 is significant but small, binary is significant and large. All three pairwise diferences exceed the critical diference $\mathrm { ( C D = 0 . 2 2 1 }$ ; mean ranks 1.306, 1.712, and 2.982 for float32, int8, and binary), but the float32–int8 gap corresponds to a mean diference in relative performance of just 1.5 percentage points, whereas the float32–binary gap corresponds to 28.4 percentage points. The confirmatory analysis on the MBAI-1D grid (7 categories × 4 dimensions, $N _ { b } = 2 8 )$ is weaker still for int8: $\chi _ { F } ^ { 2 } ( 2 ) = 4 3 . 7 5 , p < 1 0 ^ { - 9 }$ , and the float32– int8 gap of 0.536 falls below the critical diference of 0.626. Per category (Table 5b), binary quantization is significantly worse than float32 in all seven categories, whereas int8 is statistically indistinguishable from float32 for pair classification $( p _ { \mathrm { H o l m } } = 0 . 6 5 )$ and re-ranking $( p _ { \mathrm { H o l m } } = 0 . 1 8 )$ , and its largest cost falls on classification $\left( - 0 . 0 7 5 \right)$ . For retrieval the mean diference is marginally positive (+0.026), but the efect does not replicate on MBAI-1D (−0.003) and is best read as noise.

(a) Omnibus Friedman tests and Nemenyi critical diferences
<table><tr><td>Factor</td><td>Blocks</td><td>k</td><td> $N _ { b }$ </td><td></td><td> $\chi _ { F } ^ { 2 }$ </td><td>p</td><td>W</td><td>CD</td></tr><tr><td>Precision q</td><td>category</td><td> $\times \textit { m } \times \textit { l }$ </td><td>3</td><td>224</td><td>346.8</td><td> $< 1 0 ^ { - 1 5 }$ </td><td>0.774</td><td>0.221</td></tr><tr><td>Hierarchical size m</td><td>category  $\times \ q \times l$ </td><td></td><td>8</td><td>84</td><td></td><td> $5 1 1 . 9 < 1 0 ^ { - 1 5 }$ </td><td>0.871</td><td>1.146</td></tr><tr><td>Inference layer l</td><td>category  $\times \textit { q } \times \textit { m }$ </td><td></td><td>4</td><td>168</td><td></td><td> $\begin{array} { r l } { 1 9 3 . 3 } & { { } < 1 0 ^ { - 1 5 } } \end{array}$ </td><td>0.383</td><td>0.362</td></tr></table>

(b) Per-category Wilcoxon signed-rank tests against float32 (Holm-corrected)
<table><tr><td rowspan="2">Task category</td><td colspan="3">int8 vs float32</td><td colspan="3">binary vs float32</td></tr><tr><td>△</td><td>PHolm</td><td>Verdict</td><td>△</td><td>PHolm</td><td>Verdict</td></tr><tr><td>Classification</td><td>-0.0751</td><td> $5 . 5 \times 1 0 ^ { - 6 }$ </td><td>sig.</td><td>-0.3734</td><td> $2 . 4 \times 1 0 ^ { - 6 }$ </td><td>sig.</td></tr><tr><td>Clustering</td><td>-0.0033</td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td>sig.</td><td>-0.2427</td><td> $2 . 4 \times 1 0 ^ { - 6 }$ </td><td>sig.</td></tr><tr><td>STS</td><td>-0.0212</td><td> $5 . 5 \times 1 0 ^ { - 6 }$ </td><td>sig.</td><td>-0.3352</td><td> $3 . 3 \times 1 0 ^ { - 9 }$ </td><td>sig.</td></tr><tr><td>Pair class.</td><td>-0.0001</td><td> $0 . 6 5$ </td><td>n.s.</td><td>-0.3089</td><td> $3 . 3 \times 1 0 ^ { - 9 }$ </td><td>sig.</td></tr><tr><td>Retrieval</td><td>+0.0259</td><td> $6 . 7 \times 1 0 ^ { - 3 }$ </td><td>sig.</td><td>-0.3796</td><td> $3 . 3 \times 1 0 ^ { - 9 }$ </td><td>sig.</td></tr><tr><td>Re-ranking</td><td>-0.0036</td><td>0.18</td><td>n.s.</td><td>-0.2345</td><td> $3 . 3 \times 1 0 ^ { - 9 }$ </td><td>sig.</td></tr><tr><td>Summarization</td><td>-0.0303</td><td> $4 . 5 \times 1 0 ^ { - 4 }$ </td><td>sig.</td><td>-0.1142</td><td> $2 . 4 \times 1 0 ^ { - 6 }$ </td><td>sig.</td></tr></table>

<table><tr><td colspan="8">(c) Hierarchical size: mean ranks and Nemenyi verdicts for one and two successive halvings  $\left( \mathrm { C D } = 1 . 1 4 6 \right)$ </td></tr><tr><td>Size m Mean rank</td><td>1024 1.405</td><td>512 2.137</td><td>256 3.054</td><td>128 4.006</td><td>64 4.863</td><td>32 5.774</td><td>16 6.798</td></tr><tr><td>One halving |∆R| (verdict)</td><td>512→1024 0.732 (n.s.)</td><td>256→512 0.917 (n.s.)</td><td>128→256 0.952 (n.s.)</td><td>64→128 0.857 (n.s.)</td><td>32→64 0.911 (n.s.)</td><td>16→32 1.024 (n.s.)</td><td>8→16 1.167 (sig.)</td></tr><tr><td>Two halvings |∆R| (verdict)</td><td>256→1024 1.649 (sig.)</td><td>128→512 1.869 (sig.)</td><td>64→256 1.810 (sig.)</td><td>32→128 1.768 (sig.)</td><td>16→64 1.935 (sig.)</td><td>8→32 2.190 (sig.)</td><td></td></tr></table>

<table><tr><td colspan="4">(d) Inference layer: mean ranks (CD = 0.362)</td></tr><tr><td>Layer l</td><td>24</td><td>20 16</td><td>12</td></tr><tr><td>Mean rank</td><td>1.545</td><td>2.994 2.161</td><td>3.301</td></tr></table>

(e) Dispersion of relative performance across the seven task categories, with bootstrap 95% confidence intervals of the category mean
<table><tr><td>Configuration</td><td>Mean</td><td>SD</td><td>Min</td><td>Max</td><td>95% CI of the mean</td></tr><tr><td>float32, m=1024, L24 (baseline)</td><td>1.000</td><td>0.000</td><td>1.000</td><td>1.000</td><td>[1.000, 1.000]</td></tr><tr><td>int8, m=1024, L24</td><td>0.992</td><td>0.019</td><td>0.950</td><td>1.008</td><td>[0.977, 1.002]</td></tr><tr><td>int8, m=256, L24</td><td>0.973</td><td>0.032</td><td>0.920</td><td>1.018</td><td>[0.951, 0.994]</td></tr><tr><td>int8, m=128, L24</td><td>0.950</td><td>0.047</td><td>0.881</td><td>1.002</td><td>[0.917, 0.981]</td></tr><tr><td>binary, m=1024, L24</td><td>0.851</td><td>0.115</td><td>0.632</td><td>0.949</td><td>[0.765, 0.921]</td></tr><tr><td>float32, m=1024, L12</td><td>0.844</td><td>0.113</td><td>0.602</td><td>0.951</td><td>[0.757, 0.905]</td></tr><tr><td>int8, m=128, L12</td><td>0.806</td><td>0.134</td><td>0.538</td><td>0.930</td><td>[0.706, 0.884]</td></tr><tr><td>binary, m=512, L12</td><td>0.653</td><td>0.232</td><td>0.172</td><td>0.846</td><td>[0.473, 0.781]</td></tr></table>

Table 5: Statistical tests on the MBAI-2D grid of 96 configurations. (a) k is the number of levels of the factor, $N _ { b }$ the number of matched blocks, W Kendall’s coeficient of concordance, and CD the Nemenyi critical diference at $\alpha = 0 . 0 5$ . (b) Tests over the 32 matched $( m , l )$ configurations of each category; $\overline { { \Delta } }$ is the mean diference in relative performance. (c), (d) Rank 1 denotes the best level; the mean rank at $m = 8$ is 7.964. (e) Intervals from 10,000 bootstrap resamples; they describe variability between task categories, not between repeated runs.

One halving of the dimension is free; two are not. No single halving produces a rank difference exceeding the critical diference (Table 5c), with the sole exception of the final step from $m = 1 6 { \mathrm { ~ t o ~ } } m = 8$ , whereas every two-step reduction is significant. Halving the embedding dimension once is therefore not a statistically detectable intervention, which makes compounding with int8 attractive, but reductions should be adopted one step at a time and re-validated.

Layer degradation is not monotonic. The mean ranks in Table 5(d) order the layers as 24, 16, 20, 12; a monotonic account of depth would instead predict 24, 20, 16, 12. Layer 16 ranks significantly better than layer 20 $( | \Delta \bar { R } | = 0 . 8 3 3 > \mathrm { C D } = 0 . 3 6 2 )$ , while layers 20 and 12 are statistically indistinguishable $( | \Delta \bar { R } | = 0 . 3 0 7 )$ . The anomaly is concentrated in retrieval, where relative performance at layer 20 (0.513 at m = 1024, float32) falls well below that at layer 16 (0.718). The best intermediate layer must therefore be selected empirically per task.

Dispersion across task categories. The recommended int8 configurations are tightly clustered: int8 at $m = 2 5 6$ has a standard deviation of 0.032 across categories and a bootstrap 95% confidence interval of [0.951, 0.994] (Table 5e). The aggressive configurations are the opposite: binary at $m = 5 1 2$ from layer 12 averages 0.653, but its category scores range from 0.172 to 0.846. This variability is between task categories, not between repeated runs of the same configuration, because every configuration was executed once (Section 7).

## 5.4. $C O _ { 2 }$ emissions

Since the exact locations of the servers used are unknown, we estimated the emissions based on the runtime, the maximum power consumption of the NVIDIA T4 GPU (70 watts), and the average grid carbon intensity of all Google data centers, 0.337 kg $\mathrm { C O _ { 2 } }$ equivalents per $\mathrm { k W h ^ { 1 0 } }$ . The experiments ran for 154.6 hours (27.3 for Stella, 24.6 for Nomic, 24.4 for MBAI-1D, and 78.3 for MBAI-2D), consumed about 10.8 kWh, and emitted 3.65 kg $\mathrm { C O _ { 2 } }$ equivalents in total (Table S4 in the Supplementary Material). As expected, the tests for the MBAI-2D model emitted the most (1.85 kg), since all experiments were repeated for each layer considered for inference.

## 6. Discussion

The results answer the research question positively, with qualifications. Hierarchical representations and quantization can be combined: 1D MRs with int8 embeddings from the last layer $( m = 1 2 8 )$ reduce per-vector storage by an exact factor of 32 while retaining 94.4% of the efectiveness. For those prioritizing speed, however, only the 2D model ofers a substantial lever, since we found no evidence that int8 is faster than float32. Moreover, the cost in efectiveness of any given setting varies strongly across task categories.

## 6.1. Why some task categories are more sensitive than others

A consistent finding across both the 1D and the 2D experiments is that retrieval degrades faster than any other category under both dimensionality reduction and quantization, while clustering and pair classification degrade the slowest. We identify four mechanisms, which act together.

1. Ranking is a harder task than assignment is. Classification, clustering and pair classification only require that an item fall on the correct side of a decision boundary, which is stable under perturbations that are small relative to the margin, whereas retrieval requires the correct ordering of candidates: any perturbation large enough to reverse two adjacent similarity scores changes the result. Re-ranking, which orders a short candidate list, sits between the two extremes.

2. Candidate-set size compresses the margins that must be preserved. As the number of documents grows, the expected similarity gap between the correct document and its nearest distractors shrinks, so a fixed amount of quantization noise becomes progressively more likely to reverse an ordering. A retrieval query is compared with thousands of documents, whereas an STS instance compares a single pair.

3. Some tasks can re-fit around the distortion; retrieval cannot. MTEB’s classification tasks train a logistic-regression probe on the compressed embeddings themselves, which can partially absorb a systematic distortion, and k-means re-estimates its centroids in the compressed space. Retrieval has no trainable component: the similarity function is fixed, so any distortion propagates directly into the final score. This explains why classification is robust to dimension reduction, where the probe can compensate, but fragile under layer reduction (−17.1% from L24 to L12), where the features the probe would need have not yet been computed.

4. Fine-grained discrimination lives in the trailing dimensions. Matryoshka training concentrates the dominant semantic directions in the leading coordinates, so the trailing coordinates carry the fine-grained distinctions needed to separate documents that are all topically relevant to a query. Truncation removes the tail first, and binary quantization also discards magnitude, which is why the two techniques interact so destructively for retrieval: at m = 8 with binary quantization, relative retrieval performance reaches 0.00.

Taken together, these mechanisms imply that the ordering is a property of the task structure rather than of the particular models or datasets used here. We therefore expect it to generalize, and we recommend treating the presence of a large-candidate-set ranking stage, rather than the choice of embedding model, as the primary signal that conservative compression settings are required.

## 6.2. Practical guidelines for resource-constrained deployments

Storage ratios are exact (Eq. 4); efectiveness figures are relative to the m = 1024, float32, final-layer baseline, for which MBAI-1D and MBAI-2D difer by less than 1%, and timings are indicative.

• High-efectiveness applications, such as enterprise search: use $m = 1 0 2 4$ or m = 512 with int8, which retain 99% and 98% of baseline efectiveness at 4× and 8× smaller per-vector storage. A single halving of the dimension is not a statistically detectable change, so moving from 1024 to 512 is close to free. The gain is capacity, not speed.

• Balanced applications, such as general-purpose RAG pipelines: deploy m = 256 with int8, which retains 96% of baseline efectiveness at 16× smaller storage with a tight spread across task categories (SD 0.032). This is our recommended default when the workload is not known in advance.

• Resource-constrained environments, such as mobile applications: use $m = 1 2 8$ with int8, accepting 5.6% degradation for a 32× storage reduction. This is safe for classification, clustering, STS and pair classification, but not for first-stage retrieval, which retains only 0.89 of baseline at this setting on our small corpora and would degrade further at scale. Binary quantization costs 17% of efectiveness even at full size and is significantly worse than float32 in every category; we recommend it only as the candidate-generation stage of a two-stage pipeline with Hamming-distance search and higher-precision rescoring [11].

• Speed-critical applications: only the 2D model ofers a substantial lever. Sampling from layer 12 combined with m = 128 and int8 roughly halved evaluation time in our measurements, at an 18.1% cost in efectiveness, but the layer must be validated empirically rather than chosen by depth, since layer 16 ranked significantly better than layer 20, and the cost of layer reduction ranges from 4.9% for clustering to 39.8% for retrieval.

The appropriate lever also depends on the task category. Retrieval is highly sensitive to aggressive optimization, so moderate configurations (m ≥ 256 with int8) are recommended; in RAG systems, however, the re-ranking component is markedly more resilient to layer reduction than first-stage retrieval (11.0% against 39.8% from layer 24 to layer 12 with float32), which makes 2D MRs a sensible optimization for the re-ranking stage. Clustering shows exceptional resilience to layer reduction, losing only 4.9% when moving from layer 24 to layer 12, which makes it the strongest candidate for aggressive speed optimization; it also retains reasonable performance under binary quantization (50% retention at 8 dimensions). Classification is highly tolerant of dimension reduction but less tolerant of layer reduction (17.1%), so the recommended lever is truncation. STS and pair classification show excellent resilience to dimension reduction, and pair classification is also comparatively robust to layer reduction (12.8%); since a single halving costs no measurable efectiveness, truncation is again the safer first step.

Hardware and systems considerations. First, the storage reduction is real and the speed reduction largely is not: in our protocol the encoder forward pass dominates the cost, and the benchmark’s evaluators do not exploit integer arithmetic, so int8 should be adopted for capacity, not latency. Second, the speed-ups reported for binary retrieval [11] come from Hamming distance over the packed words, a regime our protocol does not exercise, so a deployment that uses such search should expect a better speed return than our figures suggest. Third, intermediate-layer inference is the only lever that reduces embedding generation cost, since it alone changes how many transformer blocks are executed; deployments that continuously index new documents will find it more valuable than the storage levers.

Use case A: enterprise RAG over a five-million-chunk document store. The binding constraint is that the vector index should fit in the RAM of a single commodity server; the uncompressed baseline requires about 20.5 GB. Because the workload is retrieval-dominated, our results forbid aggressive truncation, and $m = 2 5 6$ with int8 gives about 1.28 GB, a 16× reduction, at 0.92 of baseline retrieval efectiveness and 0.96 averaged over all categories. If recall at the top of the ranking is critical, binary codes (m = 1024, 0.64 GB) can be used for candidate generation, with the int8 vectors retained for rescoring of the shortlist.

Use case B: on-device intent classification for a mobile assistant. The binding constraints are a fixed application memory budget and a strict latency target, and the workload is classification over a fixed label set, which tolerates dimension reduction well because its logistic probe re-fits around the compressed embeddings. With $m = 1 2 8$ and int8, each cached utterance embedding takes 128 bytes rather than 4 KB, at 0.94 of baseline efectiveness on average. Moving inference from layer 24 to layer 12 roughly halved evaluation time in our measurements, but classification loses 17.1% at layer 12, so the layer should be validated empirically.

## 6.3. Performance comparison with state-of-the-art MTEB models

Among embedding models up to 1B parameters on the MTEB leaderboard (27 July 2025), the models studied here ranked 20th (Stella), 29th (MBAI-1D), 43rd (Nomic), and 191st (MBAI-2D) (Table S5 in the Supplementary Material). The leading model, Qwen3- Embedding-0.6B, also supports Matryoshka Representations, as does the 8B-parameter Qwen3 model, which ranked first among embedding models under 10B parameters. Due to their later release, we were not able to conduct experiments on the newer models; however, we expect similar eficiency gains given their MR support.

## 7. Limitations

Scope of the eficiency claims. The compression ratios reported in this paper describe theoretical per-vector storage, not resident system memory, and our timing figures are end-to-end MTEB task wall-clock times, not per-query latency.

Absence of controlled alternative-compression baselines. We do not provide measurements for float16, PCA projection, Faiss scalar quantization, or product quantization under our own protocol, and Table 2 positions these families only qualitatively. We therefore claim only that Matryoshka truncation combined with int8 is efective, that it requires no fitted projection or trained codebook, and that it composes multiplicatively with precision reduction, not that it is superior to PQ or PCA at an equal compression ratio. A comparison against PQ at matched bit budgets is the single most valuable extension of this work.

Single-run measurement and the absence of run-to-run variance. Every configuration was executed once, so the confidence intervals in Table 5(e) quantify variability across task categories, not the stability of a repeated measurement. The efectiveness measurements are deterministic up to the seeded stochasticity of k-means and of the logistic-regression solver, but the timing measurements were collected on Google Colab, a shared-tenancy environment that we could neither control nor observe. We therefore do not report any timing percentage as a headline result: the timing figures in this paper, including the 12% reduction measured for int8 embeddings at m = 128 and the reduction of roughly one half obtained by sampling from layer 12, are single measurements and should be read as indicative.

Benchmark composition and task selection. The runtime criterion removes all the large retrieval benchmarks, so our retrieval results are a lower bound on the degradation at larger scales (Section 4.2). MTEB itself is predominantly English and general-domain, so our findings may not transfer to multilingual deployments or to specialized domains.

Binary quantization: a conditional rather than general recommendation. Binary quantization is significantly worse than float32 in every task category and only modestly faster in our setting, so we do not recommend it as a standalone representation. Our binary results should, moreover, be read as a lower bound on the technique: the packed codes were processed as ordinary numeric vectors rather than compared by Hamming distance, and at very small sizes the packing leaves the evaluators almost nothing to work with (at m = 8 a code is a single byte).

Model coverage, hardware dependence, and reproducibility. Only four models were evaluated, only one of which (MBAI-2D) supports 2D representations, so every conclusion about layer selection rests on a single model. Inference-layer selection adds a configuration dimension that must be validated empirically per task, which increases the tuning burden of deploying 2D representations, and evaluating the 2D model consumed more compute than the three 1D models combined. Results were obtained on NVIDIA T4 hardware, and quantization support varies considerably across accelerators, so the timing results in particular may not transfer to other processors.

## 8. Conclusion

This paper reports an empirical benchmark of the interaction between hierarchical (Matryoshka) text representations and embedding quantization, covering 168 configurations per task across 38 MTEB tasks and four transformer-based embedding models. Its answer to the research question is that the two techniques can be combined to improve eficiency while largely preserving performance, provided that storage and speed are considered separately and that the compression level is chosen per task category. Our central quantitative result is that 8-bit quantization combined with an eightfold dimensionality reduction (m = 128) retains 94.4% of baseline efectiveness while reducing per-vector storage by an exact factor of 32, and m = 256 with int8 retains 96% of efectiveness at a 16× storage reduction, the setting we would recommend as a default. For 2D representations, layer 12 combined with int8 at $m = 1 2 8$ roughly halved end-to-end evaluation time in our measurements, at a cost of 18.1% in efectiveness.

Three findings qualify these headline numbers. First, storage gains and speed gains do not travel together: we found no evidence that int8 shortens end-to-end evaluation time, and the only lever in our study that reliably reduces computation is sampling from intermediate layers. Second, statistical testing separates the diferences that matter from those that merely exist: int8 is statistically indistinguishable from float32 for pair classification and re-ranking and costs at most 1.5 percentage points overall, while binary quantization is significantly worse in all seven task categories at a cost of roughly 28 points; no single halving of the embedding dimension is statistically detectable above $m = 1 6 ;$ and efectiveness does not degrade monotonically with layer depth. Third, the global average is the wrong object to optimize: moving from layer 24 to layer 12 costs clustering 4.9% and retrieval 39.8%, and at $m = 8$ with binary quantization retrieval collapses to zero while clustering retains half its efectiveness. Retrieval-heavy systems should stay at $m \geq 2 5 6$ with int8; classification and clustering workloads can be compressed considerably further.

The natural continuation of this work is to rank hierarchical truncation against product quantization, PCA, and float16 storage at matched bit budgets on the identical 38-task protocol, together with repeated timing runs on dedicated hardware and an extension to full-scale retrieval corpora, to billion-parameter embedding models, and to adaptive schemes that select precision and dimensionality per query.

## CRediT authorship contribution statement

Kai Ponel: Writing – original draft, Validation, Software, Resources, Methodology, Investigation, Formal analysis, Conceptualization. Marco Polignano: Writing – review & editing, Writing – original draft, Validation, Supervision, Methodology, Investigation, Formal analysis, Conceptualization. Erasmo Purificato: Writing – review & editing, Supervision, Resources. Aylin Julia Temizkan: Writing – review & editing, Supervision, Methodology. Marco Levantesi: Writing – original draft, Visualization, Resources. Giovanni Semeraro: Writing – review & editing, Supervision. Ernesto William De Luca: Writing review & editing, Supervision, Methodology.

## Declaration of Generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used Generative AI and Grammarly AI in order to improve English language readability, correct grammatical errors, and simplify complex sentences. After using these tools/services, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Declaration of competing interest

The authors declare the following financial interests/personal relationships which may be considered potential competing interests: Marco Polignano reports that financial support was provided by the Ministry of Education and Merit. Giovanni Semeraro reports that financial support was provided by the Ministry of Education and Merit. Kai Ponel reports a relationship with IBM Germany Research and Development that includes non-financial support. Aylin Julia Temizkan reports a relationship with IBM Germany Research and Development that includes employment. If there are other authors, they declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work is partially funded by Next Generation EU: NRRP Initiative, Mission 4, Component 2, Investment 1.3 – Partnerships extended to universities, research centers, companies, and research D.D. MUR n. 341 del 15.03.2022 – Next Generation EU (PE0000013 – “Future Artificial Intelligence Research – FAIR” - CUP: H97G22000210007).

## Appendix A. Supplementary data

Supplementary data to this article can be found online at doi:10.1016/j.ins.2026.124271.

## Data availability

All data used in this research are provided by the MTEB Python Library (https:// github.com/embeddings-benchmark/mteb) which is maintained by third parties.

## References

[1] Y. Bengio, Deep learning of representations for unsupervised and transfer learning, in: Proceedings of the ICML Workshop on Unsupervised and Transfer Learning, Volume 27, Proceedings of Machine Learning Research, 2012, pp. 17–36, https://proceedings.mlr.press/v27/ bengio12a.html.

[2] T. Mikolov, K. Chen, G. Corrado, J. Dean, Eficient estimation of word representations in vector space, in: Workshop Track Proceedings of the 1st International Conference on Learning Representations (ICLR), 2013, https://arxiv.org/abs/1301.3781.

[3] J. Devlin, M. Chang, K. Lee, K. Toutanova, BERT: pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2019, pp. 4171–4186, https://doi.org/10.18653/v1/N19-1423

[4] N. Reimers, I. Gurevych, Sentence-BERT: sentence embeddings using Siamese BERT-networks, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3980–3990, https://doi.org/10.18653/v1/D19-1410

[5] T. Gao, X. Yao, D. Chen, SimCSE: simple contrastive learning of sentence embeddings, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021, pp. 6894–6910, https://doi.org/10.18653/v1/2021.emnlp-main.552

[6] N. Muennighof, N. Tazi, L. Magne, N. Reimers, MTEB: massive text embedding benchmark, in: Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2023, pp. 2014–2037, https://doi.org/10.18653/v1/2023. eacl-main.148

[7] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, D. Kiela, Retrieval-augmented generation for knowledgeintensive NLP tasks, in: Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 2020.

[8] A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanujan, W. Howard-Snyder, K. Chen, S.M. Kakade, P. Jain, A. Farhadi, Matryoshka representation learning, in: Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022.

[9] X. Li, Z. Li, J. Li, H. Xie, Q. Li, 2D matryoshka sentence embeddings, arXiv preprint, 2024, https://doi.org/10.48550/arXiv.2402.14776

[10] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A.G. Howard, H. Adam, D. Kalenichenko, Quantization and training of neural networks for eficient integer-arithmetic-only inference, in: Proceedings of the 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 2704–2713, https://doi.org/10.1109/CVPR.2018.00286

[11] A. Shakir, T. Aarsen, S. Lee, Binary and scalar embedding quantization for significantly faster & cheaper retrieval, Hugging Face Blog, 2024. Published 22 March 2024; https: //huggingface.co/blog/embedding-quantization (Accessed 5 October 2026).

[12] Z. Yin, Y. Shen, On the dimensionality of word embedding, in: Advances in Neural Information Processing Systems 31 (NeurIPS 2018), 2018, pp. 895–906.

[13] S. Li, H. Guo, X. Tang, R. Tang, L. Hou, R. Li, R. Zhang, Embedding compression in recommender systems: a survey, ACM Comput. Surv. 56 (2024) 130:1–130:21, https: //doi.org/10.1145/3637841

[14] V. Gupta, S. Giesselbach, S. Rüping, C. Bauckhage, Improving word embeddings using kernel PCA, in: Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP), 2019, pp. 200–208, https://doi.org/10.18653/v1/W19-4323

[15] V. Raunak, V. Gupta, F. Metze, Efective dimensionality reduction for word embeddings, in: Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP), 2019, pp. 235–243, https://doi.org/10.18653/v1/W19-4328

[16] H. Cai, C. Gan, T. Wang, Z. Zhang, S. Han, Once-for-all: train one network and specialize it for eficient deployment, in: Proceedings of the 8th International Conference on Learning Representations (ICLR), 2020, https://openreview.net/forum?id=HylxE1HKwS.

[17] J. Yu, L. Yang, N. Xu, J. Yang, T.S. Huang, Slimmable neural networks, in: Proceedings of the 7th International Conference on Learning Representations (ICLR), 2019.

[18] X. Li, J. Li, AoE: angle-optimized embeddings for semantic textual similarity, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024, pp. 1825–1839, https://doi.org/10.18653/v1/2024.acl-long.101

[19] J. Lin, J. Tang, H. Tang, S. Yang, G. Xiao, S. Han, AWQ: activation-aware weight quantization for on-device LLM compression and acceleration, GetMobile Mob. Comput. Commun. 28 (2025) 12–17, https://doi.org/10.1145/3714983.3714987

[20] Z. Yao, X. Wu, C. Li, S. Youn, Y. He, Exploring post-training quantization in LLMs from comprehensive study to low rank compensation, in: Proceedings of the 38th AAAI Conference

on Artificial Intelligence, 2024, pp. 19377–19385, https://doi.org/10.1609/aaai.v38i17. 29908

[21] N. Huerga-Pérez, R. Álvarez, R. Ferrero-Guillén, A. Martínez-Gutiérrez, J. Díez-González, Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques, arXiv preprint, 2025, https://doi.org/10.48550/arXiv.2505.00105

[22] H. Jégou, M. Douze, C. Schmid, Product quantization for nearest neighbor search, IEEE Trans. Pattern Anal. Mach. Intell. 33 (2011) 117–128, https://doi.org/10.1109/TPAMI.2010.57

[23] T. Ge, K. He, Q. Ke, J. Sun, Optimized product quantization for approximate nearest neighbor search, in: Proceedings of the 2013 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013, pp. 2946–2953, https://doi.org/10.1109/CVPR.2013.379 Information Sciences 763 (2027) 124271

[24] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P. Mazaré, M. Lomeli, L. Hosseini, H. Jégou, The faiss library, IEEE Trans. Big Data 12 (2026) 346–361, https://doi.org/10. 1109/TBDATA.2025.3618474

[25] A. Rosenberg, J. Hirschberg, V-Measure: a conditional entropy-based external cluster evaluation measure, in: Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL), 2007, pp. 410–420, https://aclanthology.org/D07-1043/.

[26] J.A. Aslam, E. Yilmaz, Inferring document relevance via average precision, in: Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 2006, pp. 601–602, https://doi.org/10.1145/1148170.1148275

[27] K. Järvelin, J. Kekäläinen, Cumulated gain-based evaluation of IR techniques, ACM Trans. Inf. Syst. 20 (2002) 422–446, https://doi.org/10.1145/582415.582418

[28] D. Zhang, J. Li, Z. Zeng, F. Wang, Jasper and Stella: distillation of SOTA embedding models, arXiv preprint, 2024, https://doi.org/10.48550/arXiv.2412.19048

[29] Z. Nussbaum, J.X. Morris, B. Duderstadt, A. Mulyar, Nomic embed: training a reproducible long context text embedder, Trans. Mach. Learn. Res. (2025), https://openreview.net/ forum?id=IPmzyQSiQE.

[30] J. Demšar, Statistical comparisons of classifiers over multiple data sets, J. Mach. Learn. Res. 7 (2006) 1–30, https://jmlr.org/papers/v7/demsar06a.html.

[31] M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, J. Am. Stat. Assoc. 32 (1937) 675–701, https://doi.org/10.1080/01621459.1937. 10503522

[32] P. Nemenyi, Distribution-Free Multiple Comparisons (Ph.D. thesis), Princeton University, 1963.

[33] F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (1945) 80–83, https: //doi.org/10.2307/3001968

[34] S. Holm, A simple sequentially rejective multiple test procedure, Scand. J. Stat. 6 (1979) 65–70.