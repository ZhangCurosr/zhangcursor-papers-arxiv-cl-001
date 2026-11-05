# Interpretable Text Embeddings Based on Hierarchical Questions

Kohei Oda<sup>1</sup> Po-Min Chuang<sup>2</sup> Kiyoaki Shirai<sup>1</sup> Natthawut Kertkeidkachorn<sup>1</sup>

<sup>1</sup>Japan Advanced Institute of Science and Technology

<sup>2</sup>Independent Researcher

<sup>1</sup>{s2420017, kshirai, natt}@jaist.ac.jp <sup>2</sup>bominchuang@gmail.com

## Abstract

QA-based text embeddings represent texts as vectors consisting of 1/0 answers to yes/no questions. Although these embeddings provide detailed information for interpretation and achieve high task performance, they often require humans to examine a large number of questions. Furthermore, the relationship between the number of questions and the appropriate level of question granularity has not been sufficiently studied. To address these challenges, we propose a method for constructing hierarchical questions for text embeddings. As the hierarchy deepens, questions become more specific and their number increases. During interpretation, we reduce the number of questions that humans need to examine by replacing questions at deeper levels with aligned questions at shallower levels. The question hierarchy is constructed through recursive clustering, contrastive prompting, and question alignment. Experimental results on general-purpose tasks show that our method improves interpretability while maintaining task performance. It is also found that, when only a limited number of questions are available, questions at shallower levels tend to achieve higher task performance than those at deeper levels, even with the same number of questions.

## 1 Introduction

Text embeddings, which represent sentences, paragraphs, and documents as vectors, are fundamental to a wide range of natural language processing (NLP) applications, such as information retrieval (Thakur et al., 2021) and enhancing the generative capabilities of large language models (LLMs) (Lewis et al., 2021). For this reason, methods for obtaining high-quality text embeddings have been extensively studied (Reimers and Gurevych, 2019; Gao et al., 2021; Morris and Rush, 2025). However, these embeddings remain difficult to interpret, making it challenging for humans to understand which semantic aspects contribute to the similarity or dissimilarity between two texts. This lack of interpretability is particularly critical in high-stakes domains, such as legal and medical applications, where transparent NLP systems are highly desirable (McInerney et al., 2023).

![](images/3b3a2cf0f29cd8d664656a4a46206e7115a1af3b7ce667fdea5e20c25e983e59.jpg)

(b) Interpreting Answers (Embedding)  
![](images/751b386b5ad19fb90e4afd139b3a17e37f115cdf86014d744c6358ba3820a454.jpg)  
Figure 1: Overview of QA-based text embeddings and interpreting each embedding.

To develop interpretable text embeddings, various methods have been proposed (Wang et al., 2025; Anand et al., 2026; Shin and Kim, 2026). In particular, QA-based text embeddings (Benara et al., 2024; Sun et al., 2025; Tang et al., 2026) shown in Figure 1 (a) are highly interpretable because each dimension of an embedding corresponds to a yes/no question and represents its binary (1/0) answer. While conventional QA-based text embeddings represent texts using answers to relatively detailed questions, they tend to require a large number of questions for human interpretation. Furthermore, it remains unclear how many questions are needed and how specific they should be for general-purpose tasks.

This study organizes questions used for text embeddings into a hierarchical structure, where more specific questions are allocated at the lower levels. This approach has two advantages. First, it can flexibly change the number of dimensions of embeddings, i.e., the number of questions. It is preferable, since the optimal number of dimensions may differ across downstream tasks for which the text embeddings are applied. Second, it can alleviate the burden on humans to interpret text embeddings. Instead of reading all questions, a human can refer to a small number of general questions, as illustrated in Figure 1 (b), where the specific questions $q _ { 1 }$ , q<sub>2</sub>, and $q _ { 4 }$ are summarized by the general question $q _ { 9 }$

To generate questions of varying granularity, as shown in Figure 2, we recursively apply the question-generation framework proposed by Sun et al. (2025), which combines clustering of largescale text corpora with contrastive prompting. Contrastive prompting generates highly discriminative questions, and recursive clustering is expected to result in questions at shallower levels summarizing questions at deeper levels. The alignment between a question at a given level and a question at the preceding level is determined based on their entailment relationship.

Experiments on three tasks, semantic textual similarity (STS) (Agirre et al., 2012), text clustering (Aggarwal and Zhai, 2012), and information retrieval (Thakur et al., 2021) show that our method substantially improves the Pareto front between embedding performance and cognitive load. Our analysis also shows that questions become more specific at deeper levels of the hierarchy, and that aligned questions have an entailment relation with high probability. We further find that, when only a limited number of questions are available, questions at shallower levels produce better text embeddings than questions randomly selected at deeper levels.

## 2 QA-based Text Embeddings

QA-based text embeddings (Benara et al., 2024; Sun et al., 2025; Tang et al., 2026) represent a text through its responses to a set of natural-language questions. Each dimension of the embedding corresponds to a yes/no question, with a value of 1 or 0 indicating a “yes” or “no” response, respectively. For instance, the text “A study shows that exercise is good for health.” may be represented as a vector of [1, 0, 1] for the questions “Is it about health?”, “Is it about politics?”, and “Does it describe research?”. Thus, QA-based embeddings allow texts to be compared in semantic space while preserving the interpretability of individual dimensions.

![](images/821ef2919028df869df4f6ba4eb88b9112e29a01f6e54d7362561c00bf2bbb56.jpg)  
Figure 2: Overview of question generation.

Let $\mathcal { Q } = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { m } \}$ be a set of yes/no questions. The QA-based embedding $\mathbf { e } ( x )$ of an input text x is expressed as follows:

$$
{ \bf e } ( x ) = [ a _ { 1 } ( x ) , a _ { 2 } ( x ) , \dots , a _ { m } ( x ) ] ,\tag{1}
$$

where $a _ { i } ( x )$ denotes the binary answer to question $q _ { i }$ for x. The model $f _ { i } ( x )$ predicts the probability that the answer to question $q _ { i }$ for sentence x is $\mathbf { \bar { \psi } } _ { \mathbf { y e s } } , ,$ , and $a _ { i } ( x )$ is determined as follows,

$$
a _ { i } ( x ) = { \left\{ \begin{array} { l l } { 1 \ ( { \mathrm { y e s } } ) } & { { \mathrm { i f } } \ f _ { i } ( x ) > \tau } \\ { 0 \ ( { \mathrm { n o } } ) } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. } ,\tag{2}
$$

where $\tau$ is a threshold. The model $f _ { i } ( x )$ is trained using a collection of sample texts and LLMgenerated answers to those samples.

Since questions directly affect the quality of text embeddings, the method to create questions is a critical element in QA-based text embedding. Benara et al. (2024) proposed QA-Emb, a method uses LLMs to generate yes/no questions with taskspecific prompts containing examples. Sun et al. (2025) further proposed CQG-MBQA, a framework for generating more discriminative questions for QA-based text embeddings. It first groups texts into clusters based on the semantic similarity and then uses LLMs to generate yes/no questions that distinguish between the clusters. Since this framework can generate interpretable text embeddings comparable to those produced by black-box text embedding methods, we build our question generation method based on this approach.

Algorithm 1 Question Generation   
1: # Step1. Recursive Clustering   
2: $\mathcal { S } \gets \{ s _ { 1 } , s _ { 2 } , \ldots \}$   
3: $\mathbf { C } ^ { 1 } \gets$ kmeans $( S , k _ { \mathrm { f i r s t } } )$   
4: for $l \in \{ 2 , 3 , . . . , l _ { \mathrm { m a x } } \}$ do   
5: $\mathbf { C } ^ { l } \gets \bigcup _ { \mathcal { C } _ { i } ^ { l - 1 } \in \mathbf { C } ^ { l - 1 } }$ kmeans $( \mathcal { C } _ { i } ^ { l - 1 } , k )$   
6: end for   
7: # Step2. Contrastive Question Generation   
8: for $l \in \{ 1 , 2 , . . . , l _ { \mathrm { m a x } } \}$ do   
9: $\tilde { \mathbf { R } } ^ { l } \gets \bigcup _ { \mathcal { C } _ { i } ^ { l } \in \mathbf { C } ^ { l } } \{ \tilde { \mathcal { R } } _ { i } ^ { l } | \tilde { \mathcal { R } } _ { i } ^ { l } \stackrel { \mathrm { d e f } } { = } \mathrm { c q g } ( \mathcal { C } _ { i } ^ { l } , \mathbf { C } ^ { l } ) \}$   
10: $\tilde { \mathcal { Q } } ^ { l } \gets \bigcup _ { \tilde { \mathcal { R } } _ { i } ^ { l } \in \tilde { \mathbf { R } } ^ { l } } \tilde { \mathcal { R } } _ { i } ^ { l }$   
11: end for   
12: # Step3. Question Filtering   
13: for $l \in \{ 1 , 2 , . . . , l _ { \mathrm { m a x } } \}$ do   
14: $\mathbf { R } ^ { l } \gets \mathrm { f i l t e r i n g } ( \tilde { \mathbf { R } } ^ { l } )$   
15: $\mathcal { Q } ^ { l }  \bigcup _ { \mathcal { R } _ { i } ^ { l } \in \mathbf { R } ^ { l } } \mathcal { R } _ { i } ^ { l }$   
16: end for

## 3 Proposed Method

Our proposed method consists of three processes: question generation, question alignment, and model training. In the question generation process, we combine recursive clustering with the questiongeneration method proposed by Sun et al. (2025) to generate highly discriminative questions at various levels of granularity (§ 3.1). In question alignment, each question at a given hierachical level is matched with a question at the superior level with which it has an entailment relationship (§ 3.2). In the model training process, for each generated question, we train a model that predicts the probability to answer “yes” for a given sentence (§ 3.3).

## 3.1 Question Generation

Question generation process consists of three steps: recursive clustering, contrastive question generation, and question filtering. A simplified overview is shown in Algorithm 1, and each step is explained in detail below.

Step1. Recursive Clustering We first use the MEDI2 dataset<sup>1</sup> (Muennighoff et al., 2025), which contains approximately 6.8 million texts, as a largescale corpus S. The recursive k-means clustering is applied to S, where each text is encoded using an existing black-box text embedding model. At the first level, S is grouped into $k _ { \mathrm { f i r s t } }$ clusters $\mathbf { C } ^ { 1 }$ At subsequent levels, texts in each cluster from the previous level $\mathcal { C } _ { i } ^ { l - 1 }$ are grouped into k clusters, and $\mathbf { C } ^ { l }$ is obtained by merging them. Thus, the number of clusters increases by a factor of k at each level, i.e., $| { \bf C } ^ { l } | = k | { \bf C } ^ { l - 1 } |$ . The parameter $k _ { \mathrm { f i r s t } }$ is set larger than k to make a relatively large number of clusters at the top level.

Step2. Contrastive Question Generation For each cluster $\mathcal { C } _ { i } ^ { l }$ at the level l, questions are generated by the contrastive question generation (CQG) framework proposed by Sun et al. (2025). First, positive and negative samples are collected. The positive samples are $n _ { p }$ texts randomly sampled from $\mathcal { C } _ { i } ^ { l }$ . The negative samples include hard negatives and easy negatives. For the hard negatives, $n _ { h }$ texts are randomly sampled from the $n _ { c }$ clusters in $\mathbf { C } ^ { l }$ that are closest to $\mathcal { C } _ { i } ^ { l }$ . For the easy negatives, $n _ { e }$ texts are randomly sampled from the remaining clusters. The sampled positive and negative texts are then included in a prompt that instructs the LLM to generate 10 questions that yield “yes” for the positive samples and “no” for the negative samples. The prompt is provided in Appendix A. As a result of this step, 10 times as many questions are generated as there are clusters. In Algorithm 1, $\tilde { \mathbf { R } } ^ { l }$ is a collection of question sets, where each question set is generated from a single cluster, while $\tilde { \mathcal { Q } } ^ { l }$ is a set of all questions at level l.

Step3. Filtering Questions Many of the generated questions are semantically similar, resulting in redundancy that increases the computational cost of constructing QA-based text embeddings. Therefore, we apply a filtering process to the generated questions following a probing mechanism proposed by Sun et al. (2025). First, we calculate the quality for each question $q \in \tilde { \mathcal { Q } } ^ { l }$ at the level l. As in CQG, we first randomly sample $p _ { p }$ positive probes (samples) from the cluster $\mathcal { C } _ { i } ^ { l }$ where $q$ is generated, $p _ { h }$ hard negative probes from $n _ { c }$ clusters close to $\mathcal { C } _ { i } ^ { l } .$ , and $p _ { e }$ easy negative probes from other clusters. After that, the answers to q are generated by a LLM for each probe, and the quality of $q$ is calculated as follows:

$$
\begin{array} { r } { \mathrm { q u a l i t y } ( q ) = \frac { \# \mathrm { \cdots } \mathrm { y e s } ^ { \mathrm { , * } } \mathrm { f o r } \mathrm { p o s i t i v e } \mathrm { p r o b e s } } { p _ { p } } } \\ { - \frac { \# \mathrm { \cdots } \mathrm { y e s } ^ { \mathrm { , * } } \mathrm { f o r } \mathrm { n e g a t i v e } \mathrm { p r o b e s } } { p _ { h } + p _ { e } } . } \end{array}\tag{3}
$$

Afterward, all questions are encoded using a blackbox text embedding method, and questions whose cosine similarity exceeds θ are removed. In this filtering process, it is ensured that higher-quality questions are prioritized and retained. The detailed algorithm for this filtering process is described in Appendix B.

Algorithm 2 Question Alignment   
1: T<sub>question</sub>: $l _ { \mathrm { m a x } } \times \lvert \mathcal { Q } ^ { l _ { \mathrm { m a x } } } \rvert$ matrix   
2: for $l \in \{ 2 , 3 , . . . , l _ { \mathrm { m a x } } \}$ do   
3: for $\mathcal { R } _ { i } ^ { l } \in \mathbf { R } ^ { l }$ do   
4: $j \gets T _ { \mathrm { c l u s t e r } } [ l ] [ i ]$   
5: for $q ^ { l } \in \mathcal { R } _ { i } ^ { l }$ do   
6: $\tilde { q } ^ { l - 1 } \gets$   
7: argmax $\mathsf { \bar { q } } ^ { l - 1 } \in \mathcal { R } _ { j } ^ { l - }$ <sub>−1</sub>entai $1 ( q ^ { l } , q ^ { l - 1 } )$   
8: a ← index of $\check { q ^ { l } }$ in $\mathcal { Q } ^ { l }$   
9: $b $ index of $\tilde { q } ^ { l - 1 }$ in $\mathcal { Q } ^ { l - 1 }$   
10: $T _ { \mathrm { q u e s t i o n } } [ l ] [ a ]  b$   
11: end for   
12: end for   
13: end for

## 3.2 Question Alignment

We align questions at level l to the most related question at the superior level, called the “parent question” hereafter. The alignment table $T _ { \mathrm { q u e s t i o n } } [ l ] [ a ]$ is defined as an index of the parent question of the a-th question in $\mathcal { Q } ^ { l }$ , and computed by Algorithm 2. Let $T _ { \mathrm { c l u s t e r } }$ denote the precomputed alignment table that records an index of the parent cluster. For each question cluster $\mathcal { R } _ { i } ^ { l }$ at level $l ,$ an index of the parent cluster (j) is obtained using $T _ { \mathrm { c l u s t e r } }$ . Next, the parent question $\tilde { q } ^ { l - 1 }$ is chosen from the questions $q ^ { l - 1 }$ in $\mathcal { R } _ { j } ^ { l - 1 }$ . Specifically, $q ^ { l }$ and $q ^ { l - 1 }$ are converted into declarative sentences $d ^ { l }$ and $d ^ { l - 1 }$ , respectively, and a natural language inference (NLI) model predicts the probability that $d ^ { l }$ entails $d ^ { l - 1 }$ . This score represents the validity of replacing $q ^ { l }$ with $q ^ { l - 1 }$ when a human interprets text embeddings. Then, the question with the highest entailment probability is selected, and its index in $\mathcal { Q } ^ { l - 1 }$ is recorded in $T _ { \mathrm { { q u e s t i o n } } }$

## 3.3 Model Training

The QA model $f _ { i } ( x )$ described in Section 2 is trained based on the multi-task binary question answering (MBQA) framework proposed by Sun et al. (2025). For each question $q$ at the level $l ,$ the training data for $q$ is created as the following procedure. First, 50 texts from the cluster $\mathcal { C } _ { i } ^ { l }$ where question $q$ is generated, 30 texts from the $n _ { c }$ clusters closest to $\mathcal { C } _ { i } ^ { l }$ within $\mathbf { C } ^ { l }$ , and 20 texts from all other clusters are randomly sampled. Next, the answers to a question q for these texts are predicted by LLMs. After that, the pairs of the black-box embeddings from the sampled texts and their corresponding 1/0 answers are used as the training data to train a regression model. Training details are provided in Appendix C.

<table><tr><td>Level</td><td>1</td><td>2</td><td>3</td><td>4</td><td>Baseline</td></tr><tr><td># Clusters</td><td>40</td><td>200</td><td>1,000</td><td>4,996</td><td>5,000</td></tr><tr><td># Questions</td><td>213</td><td>861</td><td>3,462</td><td>13,914</td><td>16,419</td></tr></table>

Table 1: Number of clusters and questions.

The QA-based text embedding for a given input text is obtained by feeding a black-box embedding into the regression models for all questions.

## 4 Experiments

This section demonstrates the effectiveness of our method by comparing it with the baseline. We first describe the experimental settings, such as the baseline and hyperparameters (§ 4.1). After that, we explain the metrics for measuring the quality and interpretability of QA-based text embeddings (§ 4.2). Finally, we evaluate the quality and interpretability of our method and the baseline on general-purpose tasks (§ 4.3).

## 4.1 Experimental Setup

We use CQG-MBQA (Sun et al., 2025) as the baseline. CQG-MBQA uses a single clustering step instead of the recursive clustering used in our method. For the recursive clustering described in Section 3.1, we set the parameters $k _ { \mathrm { f i r s t } }$ and k to 40 and 5, respectively. We use $\mathsf { U A E - l a r g e { - v } } 1 ^ { 2 }$ for text embedding, $\mathsf { s p t } - 4 0 ^ { - } \mathsf { m i n i } ^ { 3 }$ for question generation, and $\mathtt { m D e B E R T a - b a s e - v 3 } ^ { 4 }$ as the NLI model for question alignment. Table 1 shows the number of clusters and questions generated by our method and the baseline. The number of clusters at the fourth level is slightly smaller than 5,000 because some clusters at the third level contain fewer than five samples and therefore cannot be further divided into five clusters. The number of generated questions increases in proportion to the number of clusters. The distribution of the number of texts in each cluster is provided in Appendix D.

<table><tr><td>Level</td><td> $n _ { p }$ </td><td>nh</td><td> $n _ { e }$ </td><td> $n _ { c }$ </td><td>pp</td><td>ph</td><td>pe</td><td>θ</td></tr><tr><td>1</td><td>12</td><td>36</td><td>36</td><td>3</td><td>5</td><td>3</td><td>2</td><td>0.8</td></tr><tr><td>2</td><td>10</td><td>30</td><td>30</td><td>3</td><td>5</td><td>3</td><td>2</td><td>0.8</td></tr><tr><td>3</td><td>8</td><td>24</td><td>24</td><td>3</td><td>5</td><td>3</td><td>2</td><td>0.8</td></tr><tr><td>4</td><td>6</td><td>18</td><td>18</td><td>3</td><td>5</td><td>3</td><td>2</td><td>0.8</td></tr><tr><td>Baseline</td><td>6</td><td>18</td><td>18</td><td>3</td><td>5</td><td>3</td><td>2</td><td>0.8</td></tr></table>

Table 2: Hyperparameters for question generation.

Table 2 shows the hyperparameters for question generation in our method and the baseline. For direct comparison, we use the same hyperparameter settings for the fourth level as for the baseline. As the hierarchical level becomes shallower, we increase $n _ { p }$ by 2 and $n _ { h }$ and $n _ { e }$ by 6 each. This is because clusters at shallow levels contain more samples and therefore cover broader regions of the semantic space, but using too few contrastive samples for these clusters may cause the generated questions to focus on only a limited region of the semantic space.

Our method and the baseline are evaluated on the three tasks, STS (Agirre et al., 2012), text clustering (Aggarwal and Zhai, 2012), and information retrieval (Thakur et al., 2021), using the datasets included in the massive text embedding benchmark (MTEB) (Muennighoff et al., 2023). Details of the datasets are provided in Appendix E.

## 4.2 Metrics

Embedding Quality Based on the basic settings in MTEB (Muennighoff et al., 2023), the STS task uses Spearman’s rank correlation coefficient, the text clustering task uses the V-Measure, and the information retrieval task uses nDCG@10 as evaluation metrics for the quality of text embeddings.

Interpretability As a metric for measuring interpretability, we define cognitive load as the number of questions a human must ask during interpretation. Specifically, it is defined for each of the three tasks as follows.

STS: Let ${ \mathcal { T } } _ { \mathrm { s t s } }$ be the indices of questions answered “yes” in both sentence vectors $\mathbf { v } _ { 1 }$ and $\mathbf { v } _ { 2 }$ . The cognitive load is defined as the size of $\mathcal { T } _ { \mathrm { s t s } }$

$$
{ \mathcal { T } } _ { \mathrm { s t s } } = \{ i | v _ { 1 i } = 1 \land v _ { 2 i } = 1 \}\tag{4}
$$

$$
\mathrm { c l } _ { \mathrm { s t s } } ( \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } ) = | \mathbb { Z } _ { \mathrm { s t s } } |\tag{5}
$$

Clustering: Let $\mathbb { Z } _ { \mathrm { c l u s t e r i n g } }$ be the indices of questions answered $\mathbf { \bar { y e s } } ^ { , , }$ in all text embeddings in a cluster $\nu ,$ then $\mathrm { c l } _ { \mathrm { c l u s t e r i n g } } ( \mathcal { V } ) = | \mathcal { T } _ { \mathrm { c l u s t e r i n g } } | .$

$$
\mathcal { T } _ { \mathrm { c l u s t e r i n g } } = \{ i | \forall \mathbf { v } _ { j } \in \mathcal { V } : v _ { j i } = 1 \}\tag{6}
$$

Retrieval: Let $\mathcal { T } _ { \mathrm { r e t r i e v a l } }$ be the indices of questions answered “yes” in both the query vector u and the top-1 passage v, then $\mathrm { c l } _ { \mathrm { r e t r i e v a l } } ( \mathbf { u } , \mathbf { v } ) = \left| \mathbb { Z } _ { \mathrm { r e t r i e v a l } } \right|$

$$
{ \mathcal { T } } _ { \mathrm { r e t r i e v a l } } = \{ i | u _ { i } = 1 \wedge v _ { i } = 1 \}\tag{7}
$$

Here, the cognitive load for STS is the same as that defined by Sun et al. (2025), while the cognitive loads for clustering and retrieval are newly added in this study.

Furthermore, we assume that question alignment reduces cognitive load by replacing the question with its aligned abstract question at the higher level, thereby reducing the number of questions a human must check. We define the cognitive load of embeddings with question alignment as

$$
\mathrm { c l } _ { x } ^ { \mathtt { q a } } ( \cdot ) = | \{ T _ { \mathtt { q u e s t i o n } } [ l ] [ i ] | i \in \mathcal { T } _ { x } \} | ,\tag{8}
$$

where $x \in \{ \mathrm { s t s } .$ , clustering, retrieval} and l is the question level.

## 4.3 Result and Discussion

Performance vs. Cognitive Load Figure 3 shows the performance and cognitive load on the STS, text clustering, and information retrieval tasks when the threshold τ is 0.3. Several common trends can be observed across the three tasks. First, the performance and cognitive load of the fourth level are nearly identical to those of the baseline. This suggests that questions generated through recursive clustering provide similar embedding quality and cognitive load to those generated using a single clustering step. Second, there is a trade-off between task performance and cognitive load. In other words, obtaining higher-quality text embeddings requires more questions, increasing cognitive load. A user can choose the level of embeddings, considering a balance between the performance of the downstream task and the cognitive load of interpreting the model’s prediction. In this experiment, since cognitive load increases significantly from the third level to the fourth level while performance improves only slightly, the third level appears to offer the best balance between task performance and cognitive load. Finally, question alignment can reduce cognitive load while maintaining task performance. As a result, the Pareto front of task performance and cognitive load is significantly improved, outperforming the baseline. Results for other values of the threshold $\tau$ are given in Appendix G, indicating a similar trade-off between performance and cognitive load across all values.<sup>5</sup>

![](images/98da85f42a8a7ddea621d41850ae23bda8082eade458703d4c9c74164c7c6fd3.jpg)  
Cognitive Load↓  
(a) STS

![](images/6f872eb3bfe2e1c80b4f8e3dc592e8d04adedd09beeece51ae2c77859cf48f3c.jpg)  
Cognitive Load↓  
(b) Clustering

![](images/e7f81c79d42901d74a6d7df8acdf8ac04a080c57a028fb574141628a7bc75ef6.jpg)  
Cognitive Load↓  
(c) Retrieval

Figure 3: Task performance vs. cognitive load (τ is 0.3). Q-Align indicates the results after question alignment, shown as white-filled shapes.  
![](images/6d99f88c367036519eb20fa8138769a4bdc57b37ec9fc8da62b541ea74e7589d.jpg)  
Threshold  
(a) STS

![](images/95377adfba659d21be82b8e6847bf52e719dcae36b0bab9926bc1725fc0d8782.jpg)  
(b) Clustering

![](images/e40f847a244afe45331e3428186d7be665847ccee40c19c122da92e3d9c38281.jpg)  
(c) Retrieval

Figure 4: Task performance for the different threshold τ.  
![](images/e0f627f9dce228a8ad58dcca8881e51588a2efa1ef25bce966d836ac2dcffe53.jpg)  
(a) STS

![](images/fc7323355aa5ab97d8e8cc9ea5b4f475c0c0b2194c08158db0a526491bb4f277.jpg)  
(b) Clustering

![](images/96d18c4d4068f35983d4351e20b6d97ef1668ed1f185a0ba7b665586a10868d4.jpg)  
(c) Retrieval  
Figure 5: Cognitive load for the different threshold τ. Q-Align indicates the results after question alignment, shown as white-filled shapes.

Impact of Thresholds on Performance and Cognitive Load Figures 4 and 5 show the task performance and cognitive load for the threshold τ , respectively. As shown in Figure 4, performance peaks around a threshold of 0.3 for all three tasks and gradually decreases thereafter. Furthermore, the difference in performance between the third and fourth level is not particularly significant, indicating that high-quality text embeddings can be generated for general-purpose tasks even with the number and granularity of questions at the third level. As shown in Figure 5, cognitive load decreases significantly as the threshold increases. Furthermore, the effect of reduction of cognitive load by question alignment is greater when the level is deep or the threshold is low. However, if the threshold is too low, the discrepancy between the model’s and human’s standards, such as a human interpreting a question the model answered “yes” to as “no”, may become greater, potentially reducing the reliability of the generated text embeddings.

![](images/9525b67cf9b2bd228ce1a70cc609c687aaa7d69569be3478dfc44f631f3f7ef1.jpg)  
Figure 6: Proportion of cases where the probability of “entailment” exceeded 50%.

## 5 Analysis

This section analyzes our method from various perspectives: entailment relationships between aligned questions (§ 5.1), the appropriate question granularity (§ 5.2), and a case study (§ 5.3).

## 5.1 Entailment Relationships between Aligned Questions

To reduce cognitive load through question alignment, the target question needs to semantically summarize the source questions. To verify this, the entailment relationship between the aligned questions is estimated. Specifically, for the i-th source question $q _ { i } ^ { l _ { s } }$ at the level $l _ { s }$ and the j-th target question $q _ { j } ^ { l _ { t } }$ at the level $l _ { t }$ (where $l _ { s } > l _ { t } )$ , we convert them into declarative sentences $d _ { i } ^ { l _ { s } }$ and $d _ { j } ^ { l _ { t } }$ After that, the probability $p ( d _ { j } ^ { l _ { t } } | d _ { i } ^ { l _ { s } } )$ that $d _ { i } ^ { l _ { s } }$ entails $d _ { j } ^ { l _ { t } }$ , and the probability $p ( d _ { i } ^ { l _ { s } } | d _ { j } ^ { l _ { t } } )$ that $d _ { j } ^ { l _ { t } }$ entails $d _ { i } ^ { l _ { s } }$ is estimated by an NLI model<sup>4</sup>. If $p ( d _ { j } ^ { l _ { t } } | d _ { i } ^ { l _ { s } } )$ is high $( \mathrm { i . e . , } q _ { i } ^ { l _ { s } }$ entails $q _ { j } ^ { l _ { t } } )$ and $p ( d _ { i } ^ { l _ { s } } | d _ { j } ^ { l _ { t } } )$ is largely lower than $p ( d _ { j } ^ { l _ { t } } | d _ { i } ^ { l _ { s } } ) ( \mathrm { i } . \mathbf { e } . , q _ { j } ^ { l _ { t } }$ is more abstract than $q _ { i } ^ { l _ { s } } ) , q _ { j } ^ { l _ { t } }$ can be considered suitable for replacing $q _ { i } ^ { l _ { s } }$ . When the difference between $l _ { s }$ and $l _ { t } \mathrm { i s } 2$ or greater, $j$ is determined by recursively consulting the alignment table $T _ { \mathrm { { q u e s t i o n } } }$ . For example, if $l _ { s }$ is 3 and $l _ { t }$ is 1, j is $T _ { \mathrm { q u e s t i o n } } [ 2 ] [ T _ { \mathrm { q u e s t i o n } } [ 3 ] [ i ] ]$

Figure 6 shows the proportion of combinations of questions across different levels for which the entailment probability exceeded 50%. For many level pairs, more than 67% of lower-level questions entail upper-level questions. On the other hand, the proportion of upper-level questions that entail lower-level questions is small. These results indicate that the granularity of the questions becomes finer as the level deepens, while aligned questions preserve semantic coherence. Therefore, our method generates hierarchical questions in which upper-level questions summarize lowerlevel questions. The distribution of the probability of entailment is provided in Appendix F.

![](images/14afb8bf89a78275ccb67646bd846cb4fd6e82e2afbc0d1a0058c161ee525c94.jpg)  
(a) τ is 0.3

![](images/b2b591339a02b8e276598f1335d160a062c093360314ac8ea7bd58eb60976507.jpg)  
(b) τ is 0.7  
Figure 7: Task performance vs. cognitive load when the number of questions is matched across levels by random sampling.

## 5.2 Impact of Granularity of Questions

We investigate how question granularity affects task performance and cognitive load. Figure 3 shows that embeddings encoded by the questions at the fourth level achieve the highest performance, but this may stem from using the largest number of questions. To evaluate the appropriateness of the questions themselves, questions from deeper levels are randomly chosen so that the number of sampled questions matches that of the upper levels, and the resulting text embeddings are evaluated using the same metrics as in Section 4. Figure 7 shows the mean Spearman’s rank correlation coefficient and cognitive load for the STS task across five different seeds when the threshold τ is 0.3 and 0.7. When $\tau$ is 0.3, there is little difference in performance across question granularities. In contrast, when τ is 0.7, task performance at upper levels tends to be higher than that at lower levels with the same number of questions. This trend is particularly pronounced when the number of questions is small. When the threshold is high and “yes” answers are less likely, fine-grained questions are answered “no” in most cases, making the resulting text embeddings less informative, although cognitive cost is reduced. In such cases, abstract questions are more appropriate for QA-based text embeddings. Results for other values of τ and other tasks are provided in Appendix I.

<table><tr><td colspan="3">Text 1: what level is blood pressure too low Text 2: As long as you are not feeling dizzy or passing out, and have a decent quality of life, it is totally okay to have that level of blood pressure. There are a lot of normal people walking around with blood pressures of 90/60 on no medication and it is just normal for them. The only time doctors treat low blood pressure is when there are any symptoms, such as dizziness, light-headedness, or the experience of passing out. Any of these could result in a potentially disastrous fall, so if</td></tr><tr><td>you experience one of these symptoms then you do want to tell your doctor right away.</td><td></td><td>Target Question</td></tr><tr><td>Source Question Is the article likely to be used in academic settings or coursework?</td><td>A1 A2 Yes Yes</td><td>Is the article concerned with enhancing understanding or skills?</td></tr><tr><td>Is there a clear intent to explain a concept or process?</td><td>Yes Yes Yes Yes</td><td rowspan="3">Does the article discuss physical or anatomical concepts?</td></tr><tr><td>Is the information presented in a clinical or educational context?</td></tr><tr><td>Is the information provided related to health or medical conditions?</td></tr><tr><td>Is there mention of commonly accepted blood pressure thresholds? Is there an absence of alarming or unsafe conditions in the description?</td><td>Yes Yes No Yes Yes No</td></tr></table>

Table 3: Case study of interpreting text embeddings using question alignment. A1 and A2 denote the answers of the model for Text 1 and Text 2, respectively.

## 5.3 Case Study

As a case study, we present an example of interpreting text embeddings using our method. The two input texts are sampled from the MS MARCO dataset (Bajaj et al., 2018) for the retrieval task, and the threshold τ is set to 0.5. Questions at the fourth level for which the model answers “yes” for both texts are replaced with their aligned questions at the third level. The results shown in Table 3 illustrate the interpretability of our method. For questions related to concepts shared by two texts (e.g., education and medical care), the models answer “yes” for both. In contrast, for questions related to concepts that differ between the two texts (e.g., specific numerical thresholds and unsafe conditions), the models answer “yes” only for the corresponding text. Furthermore, the aligned questions summarize the questions for which the model answers “yes” for both texts. This makes it easier for humans to interpret the text embeddings.

## 6 Related Work

General Text Embeddings Text embeddings have been actively studied in the field of NLP (Conneau et al., 2017; Reimers and Gurevych, 2019; Jiang et al., 2022). Many recent text embedding models use Transformer encoders (Devlin et al., 2019; Liu et al., 2019; Vaswani et al., 2023) trained using a contrastive learning framework (Chen et al., 2020; Gao et al., 2021; Wang et al., 2024). More recently, methods that use the hidden states of LLMs as text embeddings have also been proposed. These include methods that make the attention in LLMs bidirectional and apply contrastive learning (BehnamGhader et al., 2024; Muennighoff et al., 2025; Lee et al., 2025), as well as methods that insert input text to specific prompt templates without updating the parameters of the LLMs (Jiang et al., 2024; Lei et al., 2024; Lin et al., 2026).

Interpretable Text Embeddings Various methods have been proposed for interpretable text embeddings (Hewitt et al., 2023; Tehenan et al., 2025; Anand et al., 2026). Many of these methods assign specific roles to each dimension of the text embeddings (McInerney et al., 2023; Patel et al., 2023; Shin and Kim, 2026). Opitz and Frank (2022) mapped each dimension of the text embeddings to an abstract meaning representation metrics. Wang et al. (2025) mapped each dimension of the text embeddings to an anchor text and defined its value as the similarity between the corresponding anchor text and the input sentence. However, because these methods produce dense vectors, their interpretations may vary across individuals. In contrast, QA-based text embeddings (Benara et al., 2024; Sun et al., 2025; Tang et al., 2026) are sparse, making such ambiguity less likely.

## 7 Conclusion

In this paper, we proposed an interpretable text embedding framework with hierarchical questions to improve the interpretability of QA-based text embeddings. Experiments across multiple tasks showed that our method improved the Pareto front derived from the trade-off between task performance and cognitive load. Our analysis further showed that the aligned questions frequently exhibited entailment relationships, and the questions in shallower levels can be more effective when only a limited number of questions are available.

## Limitations

Although our method reduces cognitive load during interpretation, manually inspecting all generated questions remains difficult due to their large number. Therefore, our qualitative analysis is limited to sampled questions, and a more comprehensive human evaluation is needed to assess the quality and interpretability of the generated questions.

Another limitation is that our method focuses on improving the Pareto front derived from the tradeoff between task performance and cognitive load rather than directly improving task performance. Future work could study how the question hierarchy can be used to improve embedding quality while keeping the embeddings interpretable.

In addition, we do not evaluate our method on multiple languages. The quality and granularity of generated questions may differ across languages, so it is still unclear whether our method works well in other languages. Evaluating our method in multilingual settings is therefore an important direction for future work.

## References

Charu C. Aggarwal and ChengXiang Zhai. 2012. A survey of text clustering algorithms. In Mining Text Data, pages 77–128. Springer US, Boston, MA.

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Iñigo Lopez-Gazpio, Montse Maritxalar, Rada Mihalcea, German Rigau, Larraitz Uria, and Janyce Wiebe. 2015. SemEval-2015 task 2: Semantic textual similarity, English, Spanish and pilot on interpretability. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 252–263, Denver, Colorado. Association for Computational Linguistics.

Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau, and Janyce Wiebe. 2014. SemEval-2014 task 10: Multilingual semantic textual similarity. In Proceedings ofthe 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 81–91, Dublin, Ireland. Association for Computational Linguistics.

Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Rada Mihalcea, German Rigau, and Janyce Wiebe. 2016. SemEval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation. In Proceedings ofthe 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 497–511, San Diego, California. Association for Computational Linguistics.

Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. 2012. SemEval-2012 task 6: A pilot on semantic textual similarity. In \*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings ofthe Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 385– 393, Montréal, Canada. Association for Computational Linguistics.

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, and Weiwei Guo. 2013. \*SEM 2013 shared task: Semantic textual similarity. In Second Joint Conference on Lexical and Computational Semantics (\*SEM), Volume 1: Proceedings ofthe Main Conference and the Shared Task: Semantic Textual Similarity, pages 32–43, Atlanta, Georgia, USA. Association for Computational Linguistics.

Vishal Anand, Milad Alshomary, and Kathleen McKeown. 2026. iBERT: Interpretable embeddings via sense decomposition. In Proceedings of the 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1413–1429, Rabat, Morocco. Association for Computational Linguistics.

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. Ms marco: A human generated machine reading comprehension dataset. Preprint, arXiv:1611.09268.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. LLM2vec: Large language models are secretly powerful text encoders. In First Conference on Language Modeling.

Vinamra Benara, Chandan Singh, John Xavier Morris, Richard Antonello, Ion Stoica, Alexander Huth, and Jianfeng Gao. 2024. Crafting interpretable embeddings for language neuroscience by asking LLMs questions. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A full-text learning to rank dataset for medical information retrieval. In Proceedings of the 38th European Conference on Information Retrieval. ECIR.

Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. 2017. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pages 1–14, Vancouver, Canada. Association for Computational Linguistics.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for

contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1597–1607. PMLR.

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel Weld. 2020. SPECTER: Document-level representation learning using citation-informed transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2270–2282, Online. Association for Computational Linguistics.

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. 2017. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

John Hewitt, John Thickstun, Christopher Manning, and Percy Liang. 2023. Backpack language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9103–9125, Toronto, Canada. Association for Computational Linguistics.

Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. 2024. Scaling sentence embeddings with large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 3182–3196, Miami, Florida, USA. Association for Computational Linguistics.

Ting Jiang, Jian Jiao, Shaohan Huang, Zihan Zhang, Deqing Wang, Fuzhen Zhuang, Furu Wei, Haizhen Huang, Denvy Deng, and Qi Zhang. 2022. Prompt-BERT: Improving BERT sentence embeddings with prompts. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 8826–8837, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and

Wei Ping. 2025. NV-embed: Improved techniques for training LLMs as generalist embedding models. In The Thirteenth International Conference on Learning Representations.

Yibin Lei, Di Wu, Tianyi Zhou, Tao Shen, Yu Cao, Chongyang Tao, and Andrew Yates. 2024. Meta-task prompting elicits embeddings from large language models. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10141–10157, Bangkok, Thailand. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks. Preprint, arXiv:2005.11401.

Ailiang Lin, Zhuoyun Li, Yusong Wang, Keyu Mao, Kotaro Funakoshi, and Manabu Okumura. 2026. Reverseeol: Improving training-free text embeddings via text reversal in decoder-only llms. Preprint, arXiv:2606.05858.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. Preprint, arXiv:1907.11692.

Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: Financial opinion mining and question answering. In Companion Proceedings ofthe The Web Conference 2018, WWW ’18, page 1941–1942, Republic and Canton of Geneva, CHE. International World Wide Web Conferences Steering Committee.

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. 2014. A SICK cure for the evaluation of compositional distributional semantic models. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 216–223, Reykjavik, Iceland. European Language Resources Association (ELRA).

Denis McInerney, Geoffrey Young, Jan-Willem van de Meent, and Byron Wallace. 2023. CHiLL: Zero-shot custom interpretable feature extraction from clinical notes with large language models. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pages 8477–8494, Singapore. Association for Computational Linguistics.

John Xavier Morris and Alexander M Rush. 2025. Contextual document embeddings. In The Thirteenth International Conference on Learning Representations.

Niklas Muennighoff, Hongjin SU, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and

Douwe Kiela. 2025. Generative representational instruction tuning. In The Thirteenth International Conference on Learning Representations.

Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. 2023. MTEB: Massive text embedding benchmark. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics, pages 2014–2037, Dubrovnik, Croatia. Association for Computational Linguistics.

Juri Opitz and Anette Frank. 2022. SBERT studies meaning representations: Decomposing sentence embeddings into explainable semantic features. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 625–638, Online only. Association for Computational Linguistics.

Ajay Patel, Delip Rao, Ansh Kothary, Kathleen McKeown, and Chris Callison-Burch. 2023. Learning interpretable style embeddings via prompting LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15270–15290, Singapore. Association for Computational Linguistics.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Wonseok Shin and Songkuk Kim. 2026. Aligning sentence embeddings to human concepts via sparse autoencoders. In ICLR 2026 Workshop on Representational Alignment (Re^4-Align).

Yiqun Sun, Qiang Huang, Yixuan Tang, Anthony Kum Hoe Tung, and Jun Yu. 2025. A general framework for producing interpretable semantic text embeddings. In The Thirteenth International Conference on Learning Representations.

Yixuan Tang, Zhenghong Lin, Yandong Sun, Wynne Hsu, Mong Li Lee, and Anthony K. H. Tung. 2026. Asking the right questions: Ontology-grounded interpretable embeddings for biomedical text. Preprint, arXiv:2603.01690.

Matthieu Tehenan, Vikram Natarajan, Jonathan Michala, Milton Lin, and Juri Opitz. 2025. Mechanistic decomposition of sentence representations. Preprint, arXiv:2506.04373.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention is all you need. Preprint, arXiv:1706.03762.

Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 241–251, Melbourne, Australia. Association for Computational Linguistics.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534–7550, Online. Association for Computational Linguistics.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2024. Text embeddings by weakly-supervised contrastive pre-training. Preprint, arXiv:2212.03533.

Yile Wang, Zhanyu Shen, and Hui Huang. 2025. LDIR: Low-dimensional dense and interpretable text embeddings with relative representations. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 14397–14409, Vienna, Austria. Association for Computational Linguistics.

## A Prompts

The prompt templates used in our experiments are shown in Figures 8–10. We use gpt-4o-mini<sup>3</sup> for all LLM-based processing described in this section.

## A.1 Prompt: QA-based Text Embedding

Figure 8 shows the prompt template used to obtain QA-based text embeddings. The prompt takes two inputs: an input text and a set of yes/no questions. The LLM is instructed to answer each question with either “yes” or “no” in the same order as the questions. The resulting answers are used as labels for the corresponding text–question pairs.

## A.2 Prompt: Question-to-Declarative Conversion

Figure 9 shows the prompt template used for question-to-declarative conversion. The prompt takes a yes/no question as input. The LLM is instructed to convert the question into a corresponding declarative sentence and to output only the converted sentence.

Evaluate the following text chunk based on   
the yes/no questions provided.   
\*\*Text Chunk:\*\*   
{text chunk}   
\*\*Questions:\*\*   
{questions}   
\*\*Instruction for the model:\*\* Please read   
the provided text chunk and answer each of   
the questions with either “yes” or “no”.   
Format the responses as follows:   
1. yes/no   
2. yes/no

Figure 8: Prompt template for QA-based text embedding.  
Convert the following question into a   
declarative sentence.   
\*\*Question:\*\*   
{question}   
\*\*Instruction:\*\* Please output only the   
declarative sentence derived from the given   
interrogative sentence.  
Figure 9: Prompt template for question-to-declarative conversion.

## A.3 Prompt: Contrastive Question Generation

Figure 10 shows the prompt template used for contrastive question generation. The prompt takes two sets of articles as input: positive articles and negative articles. The LLM is instructed to generate 10 concise yes/no questions that yield “yes” for all positive articles and “no” for all negative articles. To encourage simple question structures, we additionally instruct the LLM to avoid using “and” or “or” unless necessary.

## B Details of Filtering

Algorithm 3 shows the detailed procedure for question filtering. It consists of two steps, and $\mathbf { R } _ { \mathrm { s t e p l } } ^ { l }$ and $\mathbf { R } _ { \mathrm { s t e p } 2 } ^ { l }$ are collections of sets of questions after filtering in Step 1 and Step 2, respectively. It is basically the same as the method proposed by Sun et al. (2025). However, in this study, the flag top\_t at line 14 is set to false to avoid selecting the top t questions, because it may eliminate promising candidates.

Generate 10 simple yet insightful yes/no   
questions that determine properties of an   
article, where for all questions, the answer   
will be “yes” for ALL the positive articles   
and “no” for ALL the negative articles. Keep   
questions concise and avoid using complex   
sentence structures with “and” or “or” unless   
necessary.   
\*\*Positive Articles:\*\*   
{positive articles}   
\*\*Negative Articles:\*\*   
{negative articles}   
\*\*Instruction:\*\* Based on the excerpts   
provided, generate 10 simple yet insightful   
yes/no questions that can accurately   
differentiate the positive articles from the   
negative articles. Each question should be   
concise and framed in such a way that it will   
elicit a “yes” response for ALL positive   
articles and a “no” response for ALL negative   
articles. Avoid using complex sentence   
structures with “and” or “or” unless   
absolutely necessary. Format the questions   
in a numbered list as shown below:   
1. [First simple yes/no question]   
2. [Second simple yes/no question]  
Figure 10: Prompt template for contrastive question generation.

## C Details of Training QA Models

The structure of the QA model is the same as that of Sun et al. (2025). The regression model is a multilayer perceptron that passes a 1024-dimensional vector through a (1024, 8) linear layer, a ReLU activation function, and an (8, 1) linear layer, in that order. The output logit is passed through a sigmoid function to obtain a probability of a “yes” answer. For training hyperparameters, we set the batch size to 20, the number of epochs to 10, and the learning rate to 1e-3. The model is trained using NVIDIA DGX Spark (128GB CPU).

## D Analysis of Clusters

The distribution of the number of texts in each cluster is shown in Figure 11.

## E Evaluation Data

The evaluation data for each task is as follows:

STS Seven datasets: SemEval STS tasks 2012- 2016 (Agirre et al., 2012, 2013, 2014, 2015, 2016), STS Benchmark (Cer et al., 2017), and SICK-Relatedness (Marelli et al., 2014).

Algorithm 3 Question Filtering   
1: # Step1. Filtering for each cluster   
2: $\mathbf { R } _ { \mathrm { s t e p 1 } } ^ { l }  \varnothing$   
3: for $\dot { \mathcal { R } } \in { \bf R } ^ { l }$ do   
4: $\mathcal { R } _ { \mathrm { e x c l u d e } }  \emptyset$   
5: $A  \{ ( q _ { i } , q _ { j } ) \in \mathcal { R } \times \mathcal { R } \mid q _ { i } \neq q _ { j } \}$   
6: for $q _ { i } , q _ { j } \in { \mathcal { A } }$ do   
7: if sim $( q _ { i } , q _ { j } ) > \theta$ then   
8: if quality(q<sub>i</sub>) < quality(q<sub>j</sub>) then   
9: ${ \mathcal { R } } _ { \mathrm { e x c l u d e } }  { \mathcal { R } } _ { \mathrm { e x c l u d e } } \cup \{ q _ { i } \}$   
10: end if   
11: end if   
12: end for   
13: $\mathcal { R }  \mathcal { R } \setminus \mathcal { R }$ exclude   
14: if top\_t then   
15: $\mathcal { R } \gets \mathrm { t o p \_ t } ( \mathcal { R } , t )$   
16: end if   
17: $\mathbf { R } _ { \mathrm { s t e p 1 } } ^ { l }  \mathbf { R } _ { \mathrm { s t e p 1 } } ^ { l } \cup \{ \mathcal { R } \}$   
18: end for   
19:   
20: # Step2. Filtering across clusters   
21: $\mathbf { R } _ { \mathrm { s t e p } 2 } ^ { l }  \varnothing$   
22: for $\mathcal { R } _ { \mathrm { t a r } } \in \mathbf { R } _ { \mathrm { s t e p l } } ^ { l }$ do   
23: R<sub>exclude</sub> ← ∅   
24: for $\mathcal { R } _ { \mathrm { r e f } } \in \mathbf { R } _ { \mathrm { s t e p l } } ^ { l }$ do   
25: $\mathcal { A }  \{ ( q _ { i } , q _ { j } ^ { \cdot } ) \in \mathcal { R } _ { \mathrm { t a r } } \times \mathcal { R } _ { \mathrm { r e f } } \mid q _ { i } \neq q _ { j } \}$   
26: for $q _ { i } , q _ { j } \in { \mathcal { A } }$ do   
27: if sim $( q _ { i } , q _ { j } ) > \theta$ then   
28: if quality $\left( q _ { i } \right) < \mathrm { q u a l i t y } ( q _ { j } )$ then   
29: ${ \mathcal { R } } _ { \mathrm { e x c l u d e } }  { \mathcal { R } } _ { \mathrm { e x c l u d e } } \cup \{ q _ { i } \}$   
30: end if   
31: end if   
32: end for   
33: end for   
34: $\mathcal { R } _ { \mathrm { t a r } }  \mathcal { R } _ { \mathrm { t a r } } \backslash \mathcal { R }$ exclude   
35: $\mathbf { R } _ { \mathrm { s t e p } 2 } ^ { l }  \mathbf { R } _ { \mathrm { s t e p } 2 } ^ { l } \cup \{ \mathcal { R } _ { \mathrm { t a r } } \}$   
36: end for   
37: return $\mathbf { R } _ { \mathrm { s t e p } 2 } ^ { l }$

![](images/9c34f4a52a3f7ca9359944e97437a81519a76117b64b77aab39e342d46436033.jpg)  
Figure 11: Distribution of the number of texts in each cluster.

Clustering Seven datasets: TwentyNewsgroupsClustering<sup>6</sup>, StackExchangeClusteringP2P<sup>7</sup>, BiorxivClusteringP2P<sup>8</sup>, BiorxivClusteringS2S<sup>9</sup>, MedrxivClusteringP2P<sup>10</sup>, Medrxiv-ClusteringS2S<sup>11</sup>, and RedditClusteringP2P<sup>12</sup>.

Retrieval Six datasets: MS MARCO (Bajaj et al., 2018), ArguAna (Wachsmuth et al., 2018), FiQA-2018 (Maia et al., 2018), NFCorpus (Boteva et al., 2016), SCIDOCS (Cohan et al., 2020), and SciFact (Wadden et al., 2020).

## F Analysis of Questions

## F.1 Length of Questions

The distribution of the number of characters in questions is shown in Figure 12. Even across different levels, the length of questions does not vary much.

![](images/b8cba63e55e15ddbea0163ab37ce93ddd57485229534bf179e8668cce0f424f7.jpg)  
Figure 12: Distribution of the number of characters in questions.

## F.2 Concreteness of Questions

We calculate the concreteness of questions using wordtangible<sup>13</sup>. In wordtangible, a predefined dictionary, where the keys are words and the values are their concreteness scores, is used to compute the average concreteness of words as the concreteness of the sentence. In the original wordtangible setting, words not included in the dictionary are ignored. However, we believe that words not in the dictionary, such as proper nouns, should be considered concrete, so we assign them the maximum concreteness score (5 points). The distribution of the concreteness of questions is shown in Figure 13. As the level becomes deeper, the average concreteness tends to increase.

```html
<sup>6</sup>https://huggingface.co/datasets/mteb/
twentynewsgroups-clustering
<sup>7</sup>https://huggingface.co/datasets/mteb/
stackexchange-clustering-p2p
<sup>8</sup>https://huggingface.co/datasets/mteb/
biorxiv-clustering-p2p
<sup>9</sup>https://huggingface.co/datasets/mteb/
biorxiv-clustering-s2s
<sup>10</sup>https://huggingface.co/datasets/mteb/
medrxiv-clustering-p2p
<sup>11</sup>https://huggingface.co/datasets/mteb/
medrxiv-clustering-s2s
<sup>12</sup>https://huggingface.co/datasets/mteb/
reddit-clustering-p2p
$^ { 1 3 } \mathsf { h t t p s } : / / \mathsf { g i }$ thub.com/jrrobison1/wordtangible
```

![](images/adbaa2ba2fe2a94d8bf7b9878329a14498f6339f126f091ab23e6be752d463be.jpg)  
Figure 13: Distribution of the concreteness of questions.

## F.3 Probability of Entailment

Figure 14 shows the distribution of entailment probabilities when the deeper-level sentences are the premise and the shallower-level sentences are the hypothesis, while Figure 15 shows the distribution of entailment probabilities when the premise and hypothesis are swapped.

![](images/21dc2fbf5c05d0d5839c444b9506ceb07be9c2830a282318b297e59854c0bbff.jpg)  
Figure 14: Distribution of probabilities of entailment (deeper-level → shallower-level).

## G Full Results of Performance vs. Cognitive Load

Figures 16–18 show the full results of the performance and cognitive load on the STS, text clustering, and information retrieval tasks, respectively.

## H Task Performance by Dataset

Tables 4–6 present the performance on individual datasets for the STS, text clustering, and information retrieval tasks. CQG-MBQA (Repro.) denotes our reimplementation of CQG-MBQA, which corresponds to the baseline described in the main text. The results for all other methods, except CQG-MBQA (Repro.) and the proposed method, are taken from Sun et al. (2025). For CQG-MBQA (Ori.), CQG-MBQA (Repro.), and the proposed method, the threshold τ is set to 0.5.

![](images/b959ed91cc8045e9147407dd86f176b0ead0d6da67d63b5b7730a44ef27dc6ad.jpg)  
Figure 15: Distribution of probabilities of entailment (shallower-level → deeper-level).

## I Full Results of Analysis of Appropriate Granularity of Questions

Figures 19–21 show the full results of the performance and cognitive load on the STS, text clustering, and information retrieval tasks when the number of questions is matched across levels by random sampling, respectively.

![](images/814d8c6fcadbbf23f25f11384030f0bfccb77dfe11f64132096705fe21fdfed3.jpg)  
Cognitive Load↓  
(a) τ is 0.1

![](images/a0e8b5b8864160c10e8753cc0869fa56c6213c3d18b6bdcc988bfaed2fddaaa5.jpg)  
Cognitive Load↓  
(b) τ is 0.2

![](images/84bbb09a65d78b07b4263b9c91b167ebcaed227a321b1fcd6aa7659e2adcf2c5.jpg)  
Cognitive Load↓  
(c) τ is 0.3

![](images/adff191cd278924887906dc13365906a69ef06fb3ca5e79200716b7a9cc2ab42.jpg)  
Cognitive Load  
(d) τ is 0.4

![](images/61c4dee6ed4b18b078e9eb169ae87f7eba0455995ece19910cf420cd5459545e.jpg)  
Cognitive Load ↓  
(e) τ is 0.5

![](images/84eaf2b4f3b52def9e302bfbf80c838b5b06ba6c016c948db17f1ae53e342644.jpg)  
Cognitive Load ↓  
(f) τ is 0.6

![](images/5fdaddca48bf6d01268498dd4859d1e82b07595c99cfc8bd2d66f688b568198b.jpg)  
Cognitive Load ↓  
(g) τ is 0.7

![](images/01cdb83f2c4f6d262dfd628f0f71b4b23913f0012f5346f8e7240e3875e0f74c.jpg)  
Cognitive Load↓  
(h) τ is 0.8

![](images/650fe90602a820f734fd378bb13d822caee8367db8823115db32aa26b757aa09.jpg)  
Cognitive Load↓  
(i) τ is 0.9  
Figure 16: Task performance vs. cognitive load on the STS task.

![](images/398bf192cadee1c2cfa9e0c22ee4a9fcb99aee7a5dcabd11f36adf509198eb85.jpg)  
Cognitive Load↓  
(a) τ is 0.1

![](images/bb53711e4a9889381332ba7fb132f7cf91fc3525582904b86adcfbc25187ca61.jpg)  
Cognitive Load↓  
(b) τ is 0.2

![](images/5e5783e58e4e5edbb6d51fedf01b2f1f4e8f8ab33fdba1eb6f4362aaa08d99e0.jpg)  
Cognitive Load↓  
(c) τ is 0.3

![](images/fef9dadd5829ab61e5fa912c22b769b4d8863cdba5701082d3330b76729ec6d0.jpg)  
Cognitive Load↓  
(d) τ is 0.4

![](images/369bbd768735bdb50c9609c8091c67da8c9f427d7692f4550cf24a1592da5a7e.jpg)  
Cognitive Load ↓  
(e) τ is 0.5

![](images/1b9762c84ff1fbe94009509594aa38d096a5806353f090928068b1a54cb1df9b.jpg)  
Cognitive Load↓  
(f) τ is 0.6

![](images/88843bbe9000a894202be9b615778bbd43c65ad49ec7ca120f03299ec475a196.jpg)  
Cognitive Load ↓  
(g) τ is 0.7

![](images/532f0da2c7a2f02695dd021455e99be368c8ea27407d203345022b14cb51360a.jpg)  
Cognitive Load↓  
(h) τ is 0.8

![](images/ed7e14b3097de995a9d369f5d36aa64ef45142674707ab9def25dd758c8f015a.jpg)  
Cognitive Load↓  
(i) τ is 0.9  
Figure 17: Task performance vs. cognitive load on the text clustering task.

![](images/8f870923c1d6158f6cea2c88c572a1e532a629a321c340765f870609ab50e056.jpg)  
Cognitive Load↓  
(a) τ is 0.1

![](images/568e573e2d0ea17c10e9f4bf29a5b99ed2417350efb9da586dd5015f648be401.jpg)  
Cognitive Load↓  
(b) τ is 0.2

![](images/db3ca4fc235b8d9ab482fabc3dc0b9a5bc1b67397ee6ba3da8f5290892ecbd10.jpg)  
Cognitive Load↓  
(c) τ is 0.3

![](images/e44731f5d443cbc1d5d48252751300e85c8f93f5f74d2027c86a870724de51ba.jpg)  
Cognitive Load↓  
(d) τ is 0.4

![](images/609ec3db75a539eae92dd19799ef95dbc331ddf70a1ec07ebdd66d35d3da7637.jpg)  
Cognitive Load ↓  
(e) τ is 0.5

![](images/d44385ab122e7517f6e9d9dee50c090a832bef453506e14366b40d8b2d365fa0.jpg)  
Cognitive Load↓  
(f) τ is 0.6

![](images/373f08f86d1ed71489d2d7c5fadf97524a15dd1fb01e269cad6170d6c3f4e9a3.jpg)  
Cognitive Load ↓  
(g) τ is 0.7

![](images/0ab297b7c7a84d57180e28c71add54c5d2d7f29b0fac51979ff31f1dc1b35e9f.jpg)  
Cognitive Load↓  
(h) τ is 0.8

![](images/99d7cb095c2744813831c12be690388275f3dc07bff62126a993ce48a2cbe86a.jpg)  
Cognitive Load↓  
(i) τ is 0.9  
Figure 18: Task performance vs. cognitive load on the information retrieval task.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="8">Spearman Correlation (STS)</td></tr><tr><td>STS12</td><td>STS13</td><td>STS14</td><td>STS15</td><td>STS16</td><td>STS-B</td><td>SICK-R</td><td>Avg.</td></tr><tr><td rowspan="10">Blaqox</td><td>BERT</td><td>38.78</td><td>57.98</td><td>57.98</td><td>63.15</td><td>61.06</td><td>46.35</td><td>58.40</td><td>54.81</td></tr><tr><td>GloVe</td><td>54.64</td><td>69.16</td><td>60.81</td><td>72.31</td><td>65.34</td><td>61.54</td><td>55.43</td><td>62.74</td></tr><tr><td>USE</td><td>64.49</td><td>67.80</td><td>64.61</td><td>76.83</td><td>73.18</td><td>74.92</td><td>76.69</td><td>71.22</td></tr><tr><td>SimCSE (Unsup.)</td><td>66.05</td><td>81.49</td><td>73.61</td><td>79.72</td><td>78.12</td><td>76.52</td><td>72.24</td><td>75.39</td></tr><tr><td>SBERT (Ori.)</td><td>74.53</td><td>77.00</td><td>73.18</td><td>81.85</td><td>76.82</td><td>79.10</td><td>74.29</td><td>76.68</td></tr><tr><td>SimCSE (Sup.)</td><td>75.30</td><td>84.67</td><td>80.19</td><td>85.40</td><td>80.82</td><td>84.25</td><td>68.38</td><td>79.86</td></tr><tr><td>WhitenedCSE</td><td>74.65</td><td>85.79</td><td>77.49</td><td>84.71</td><td>80.33</td><td>81.48</td><td>75.34</td><td>79.97</td></tr><tr><td>SBERT (New)</td><td>73.08</td><td>82.13</td><td>76.73</td><td>85.58</td><td>80.23</td><td>83.09</td><td>79.32</td><td>80.02</td></tr><tr><td>OpenAI</td><td>72.84</td><td>86.1</td><td>81.15</td><td>88.49</td><td>85.08</td><td>83.56</td><td>79.00</td><td>82.31</td></tr><tr><td>AnglE</td><td>79.09</td><td>89.62</td><td>85.02</td><td>89.51</td><td>86.61</td><td>89.06</td><td>82.62</td><td>85.93</td></tr><tr><td rowspan="6">Iuterrble</td><td>Bag-of-Tokens</td><td>44.75</td><td>52.06</td><td>54.78</td><td>68.65</td><td>60.59</td><td>54.85</td><td>57.87</td><td>56.22</td></tr><tr><td>QAEmb-MBQA</td><td>59.40</td><td>63.19</td><td>57.68</td><td>69.29</td><td>63.18</td><td>71.33</td><td>72.33</td><td>65.20</td></tr><tr><td>ČQG-MBQA (Ori.)</td><td>69.21</td><td>80.19</td><td>73.91</td><td>80.66</td><td>78.30</td><td>82.69</td><td>78.21</td><td>77.60</td></tr><tr><td>CQG-MBQA (Repro.)</td><td>74.95</td><td>79.96</td><td>75.66</td><td>82.47</td><td>80.47</td><td>84.26</td><td>77.16</td><td>79.28</td></tr><tr><td>Ours (Level 4)</td><td>75.58</td><td>80.23</td><td>75.63</td><td>82.75</td><td>80.27</td><td>83.73</td><td>76.74</td><td>79.27</td></tr><tr><td>Ours (Level 3)</td><td>74.13</td><td>81.32</td><td>75.46</td><td>82.02</td><td>79.26</td><td>83.11</td><td>76.07</td><td>78.77</td></tr><tr><td>Ours (Level 2)</td><td></td><td>67.28</td><td>75.52</td><td>70.83</td><td>77.50</td><td>74.77</td><td>79.60</td><td>74.47</td><td>74.28</td></tr><tr><td>Ours (Level 1)</td><td></td><td>58.28</td><td>69.82</td><td>59.68</td><td>72.43</td><td>67.36</td><td>72.05</td><td>68.96</td><td>66.94</td></tr></table>

Table 4: Task performance by dataset on the STS task.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="8">V-Measure (Clustering)</td></tr><tr><td>TNG</td><td>SE-P2P</td><td>BR-P2</td><td>BR-S2S</td><td>MR-P2P</td><td>MR-S2S</td><td>RD-P2P</td><td>Avg.</td></tr><tr><td rowspan="7">Blabqox</td><td>SimCSE (Unsup.)</td><td>23.21</td><td>28.50</td><td>24.90</td><td>19.55</td><td>23.60</td><td>21.97</td><td>45.14</td><td>26.70</td></tr><tr><td>GloVe</td><td>25.83</td><td>28.51</td><td>29.27</td><td>19.18</td><td>26.12</td><td>20.38</td><td>35.82</td><td>26.44</td></tr><tr><td>BERT</td><td>23.35</td><td>26.55</td><td>30.12</td><td>24.77</td><td>26.09</td><td>23.60</td><td>43.32</td><td>28.26</td></tr><tr><td>SimCSE (Sup.)</td><td>34.86</td><td>29.45</td><td>30.15</td><td>24.67</td><td>26.25</td><td>24.12</td><td>47.74</td><td>31.03</td></tr><tr><td>SBERT (New)</td><td>47.47</td><td>33.13</td><td>36.99</td><td>33.21</td><td>34.25</td><td>32.24</td><td>54.80</td><td>38.87</td></tr><tr><td>AnglE</td><td>51.72</td><td>36.72</td><td>39.38</td><td>37.23</td><td>33.22</td><td>31.18</td><td>65.35</td><td>42.11</td></tr><tr><td>OpenAI</td><td>58.14</td><td>36.88</td><td>38.03</td><td>36.53</td><td>32.70</td><td>31.27</td><td>67.96</td><td>43.07</td></tr><tr><td rowspan="8">Intererrble</td><td>Bag-of-Tokens</td><td>8.52</td><td>17.64</td><td>4.70</td><td>3.32</td><td>11.39</td><td>13.05</td><td>15.67</td><td>10.61</td></tr><tr><td>QAEmb-MBQA</td><td>36.72</td><td>25.68</td><td>24.66</td><td>21.16</td><td>25.53</td><td>22.85</td><td>46.57</td><td>29.02</td></tr><tr><td>CQG-MBQA (Ori.)</td><td>40.00</td><td>28.22</td><td>34.88</td><td>31.13</td><td>31.02</td><td>28.71</td><td>54.40</td><td>35.48</td></tr><tr><td>CQG-MBQA (Repro.)</td><td>43.95</td><td>34.50</td><td>36.78</td><td>33.21</td><td>31.45</td><td>29.29</td><td>57.93</td><td>38.16</td></tr><tr><td>Ours (Level 4)</td><td>42.78</td><td>34.50</td><td>36.93</td><td>32.65</td><td>31.74</td><td>29.41</td><td>58.24</td><td>38.04</td></tr><tr><td>Ours (Level 3)</td><td>41.01</td><td>33.64</td><td>35.68</td><td>31.80</td><td>31.02</td><td>28.71</td><td>56.61</td><td>36.93</td></tr><tr><td>Ours (Level 2)</td><td>34.32</td><td>32.36</td><td>29.87</td><td>27.31</td><td>27.87</td><td>25.74</td><td>50.78</td><td>32.61</td></tr><tr><td>Ours (Level 1)</td><td>28.75</td><td>30.21</td><td>23.58</td><td>18.36</td><td>23.80</td><td>20.60</td><td>38.63</td><td>26.27</td></tr></table>

Table 5: Task performance by dataset on the text clustering task.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Model</td><td colspan="7">nDCG@10 (Retrieval)</td></tr><tr><td>MS MARCO</td><td>ArguAna</td><td>FQA</td><td>NFC</td><td>SCIDOCS</td><td>SciFact</td><td>Avg.</td></tr><tr><td rowspan="7">Blaqox</td><td>BERT</td><td>16.86</td><td>28.29</td><td>2.19</td><td>4.30</td><td>2.82</td><td>13.34</td><td>11.30</td></tr><tr><td>SimCSE (Unsup.)</td><td>44.63</td><td>38.34</td><td>9.84</td><td>9.88</td><td>5.50</td><td>25.72</td><td>22.32</td></tr><tr><td>GloVe</td><td>44.27</td><td>36.30</td><td>10.09</td><td>13.87</td><td>8.04</td><td>29.58</td><td>23.69</td></tr><tr><td>SimCSE (Sup.)</td><td>47.86</td><td>38.33</td><td>10.41</td><td>12.42</td><td>7.53</td><td>29.59</td><td>24.36</td></tr><tr><td>SBERT (New)</td><td>88.74</td><td>47.13</td><td>37.27</td><td>32.25</td><td>21.82</td><td>62.64</td><td>48.31</td></tr><tr><td>AnglE</td><td>90.43</td><td>66.15</td><td>44.84</td><td>38.65</td><td>22.98</td><td>74.07</td><td>56.19</td></tr><tr><td>OpenAI</td><td>92.18</td><td>58.05</td><td>55.00</td><td>42.07</td><td>23.11</td><td>77.77</td><td>58.03</td></tr><tr><td rowspan="8">Inutetrrbble</td><td>Bag-of-Tokens</td><td>29.79</td><td>34.25</td><td>3.99</td><td>21.51</td><td>6.79</td><td>47.36</td><td>23.95</td></tr><tr><td>BM25</td><td>68.42</td><td>49.28</td><td>25.14</td><td>32.08</td><td>15.78</td><td>68.70</td><td>43.23</td></tr><tr><td>QAEmb-MBQA</td><td>40.51</td><td>34.75</td><td>8.23</td><td>3.87</td><td>3.74</td><td>12.01</td><td>17.18</td></tr><tr><td>ČQG-MBQA (Ori.)</td><td>62.21</td><td>47.75</td><td>18.63</td><td>9.74</td><td>8.67</td><td>32.80</td><td>29.97</td></tr><tr><td>CQG-MBQA (Repro.)</td><td>67.79</td><td>55.57</td><td>16.64</td><td>7.54</td><td>12.87</td><td>25.16</td><td>30.93</td></tr><tr><td>Ours (Level 4)</td><td>69.66</td><td>55.04</td><td>18.63</td><td>8.36</td><td>12.65</td><td>25.40</td><td>31.62</td></tr><tr><td>Ours (Level 3)</td><td>61.80</td><td>49.00</td><td>15.12</td><td>7.07</td><td>10.18</td><td>20.21</td><td>27.29</td></tr><tr><td>Ours (Level 2)</td><td>38.29</td><td>35.64</td><td>5.62</td><td>5.47</td><td>5.69</td><td>9.54</td><td>16.71</td></tr><tr><td>Ours (Level 1)</td><td></td><td>12.16</td><td>14.54</td><td>2.29</td><td>3.11</td><td>2.12</td><td>4.11</td><td>6.39</td></tr></table>

Table 6: Task performance by dataset on the information retrieval task.

![](images/80488183963d8cd82e2b5c0ba10e4bb0c4906e7f68fcaeb2d677ea5dc67ffaf6.jpg)  
Cognitive Load ↓  
(a) τ is 0.1

![](images/d74604e59ebd491eff9855eea9cbcd62a40ff47a463a90af68d9a8e70e2be061.jpg)  
Cognitive Load ↓  
(b) τ is 0.2

![](images/ba97872e3dea9fb6bca49225f7836d5b19aea3733348e1413da0a47721d8a614.jpg)  
(c) τ is 0.3

![](images/dd717504cb8ee945f8227972d62fc48a5b8ad17cb7ab9bb292272f1c6a89dae4.jpg)  
Cognitive Load↓  
(d) τ is 0.4

![](images/e2696a2e62378354d052d5735279f1892cda7f9c28990a7891f64b8ae37b4bb6.jpg)  
(e) τ is 0.5

![](images/95dee3018ab8a9ff55ca121b7b6421b949222c7518b67b4627be4d3f1cacdccc.jpg)  
Cognitive Load↓  
(f) τ is 0.6

![](images/ee1406c520cabb6bb8c216acb9a5e7d6703ded42f766fc395b86434d60deff42.jpg)  
(g) τ is 0.7

![](images/ae79a5d50cb84d4f52070bb54a6ee62491c0ca46281ee04c407e445b9ad0ca86.jpg)  
(h) τ is 0.8

![](images/d10b0561405562728c885ea0534a15c666174d9ae740fa724f9c7e3834d592c2.jpg)  
(i) τ is 0.9  
Figure 19: Task performance vs. cognitive load on the STS task when the number of questions is matched across levels by random sampling.

![](images/ae43367214b45a4ed4781323295ee937c7c11e1e91d866ee81579c2f43f402df.jpg)  
Cognitive Load↓  
(a) τ is 0.1

![](images/cd7df9d053827fd386d387d13ec111c89bf5577948d335510fec0e99a37aa0b7.jpg)  
Cognitive Load ↓  
(b) τ is 0.2

![](images/aab524e704a6decc25ea58d66fa8d10dd3143e9936dcd9667a3f970e84a2783a.jpg)  
Cognitive Load↓  
(c) τ is 0.3

![](images/bbeec4d23239a5cd6ca1ce7d748f2e708cee07c2d274670c2dcbefce6b014499.jpg)  
Cognitive Load↓  
(d) τ is 0.4

![](images/5b13d635a97f43b47982b02503e78896cf395aef137af59b8b0e526f4833a7bf.jpg)  
Cognitive Load↓  
(e) τ is 0.5

![](images/ff6fe6ddd9686b99656b742e5b7b47987f7863b422ae897bad958ca21ed21be1.jpg)  
Cognitive Load↓  
(f) τ is 0.6

![](images/4ecf9dfbf74250c777864238b2f70bb621b2b75d3937de7780d1fd03380d765e.jpg)  
Cognitive Load↓  
(g) τ is 0.7

![](images/744bfd98a0039e5a87d1b6dff617f62e1eaa99ee47d0fd1d2808eea28fa30c19.jpg)  
(h) τ is 0.8

![](images/3406a9fec0ee060d97db2c3206e331d47bfd393107eb0b044191a7501f3a5171.jpg)  
(i) τ is 0.9

Figure 20: Task performance vs. cognitive load on the text clustering task when the number of questions is matched across levels by random sampling.

![](images/99438eeb4861c390c1650970cf8145cc54ecc37fecd0302309a3cfd6ed49448c.jpg)  
Cognitive Load↓  
(a) τ is 0.1

![](images/bf58aeba0ae30122b5bc3692ed118533b2a222b56c342a7b6daaabff33d6183f.jpg)  
(b) τ is 0.2

![](images/269e8946cdcdb461488945e86f709df9195c725b836c10a125e4463bd4768dfc.jpg)  
(c) τ is 0.3

![](images/d17c2171ca3cc5d6f26559118f7268ffa851a5e61312420113d857aeaaef9d7a.jpg)  
Cognitive Load↓  
(d) τ is 0.4

![](images/67ad9f39f5687250ef8ee37ca8174b50b0dcc6b7828ded73539650b8116b1f0d.jpg)  
(e) τ is 0.5

![](images/c6a624b4425e827577b2fb6f91ad9f4ab711d294ddb5f90ff4deaef581d2e2e0.jpg)  
Cognitive Load↓  
(f) τ is 0.6

![](images/9b99b50cf323475cd267b4433d82ffcebaf9fb74e9c007a2a3ff5c6e9cfad98d.jpg)  
Cognitive Load↓  
(g) τ is 0.7

![](images/0691739632b45528d8ffdf615fb38c99ce37e38333338c1f12f95fa020ca79b1.jpg)  
Cognitive Load ↓  
(h) τ is 0.8

![](images/512fbdd62fbc19a3d641e82468856d00cd056a4a0207c5bd3ce92eaabd6a50be.jpg)  
Cognitive Load ↓  
(i) τ is 0.9  
Figure 21: Task performance vs. cognitive load on the information retrieval task when the number of questions is matched across levels by random sampling.