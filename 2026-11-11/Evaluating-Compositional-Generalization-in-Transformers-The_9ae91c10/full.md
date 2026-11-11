# Evaluating Compositional Generalization in Transformers: The Role of Composition Equivalence and Module Coverage

Purva Pruthi University of Massachusetts Amherst Amherst, MA, USA ppruthi@cs.umass.edu

Alexander D'Amour Google DeepMind Boston, Massachusetts, U.S.A alexdamour@google.com

Andrew Yuan University of Massachusetts Amherst Amherst, MA, USA awyuan@cs.umass.edu

David Jensen University of Massachusetts Amherst Amherst, MA, USA jensen@cs.umass.edu

## Abstract

Compositional generalization is an important form of out-of-distribution generalization, defined as the ability to train on some combinations of modules (e.g., functions such as sort, filter) and then generalize to unseen combinations. Despite substantial empirical work evaluating compositional generalization in transformers, findings have been inconsistent because evaluations were often conducted on a limited set of train-test splits, defined in an ad hoc manner. Through a systematic evaluation, we identify two key aspects of the train-test distribution that strongly affect performance: composition equivalence and module coverage. We show that the apparent performance of direct models (trained only on final outputs) results primarily from exploiting composition equivalences—different sequences of modules that reduce to identical end-to-end functions. When benchmarks eliminate these equivalences, performance drops significantly, showing their inability to generalize to compositions of known modules that produce novel end-to-end functions. We further demonstrate that performance degrades monotonically as module coverage shifts increase—when modules appear in different contexts between train and test—affecting both direct and step-by-step models (trained on intermediate and final outputs). These findings have concrete implications for benchmark design: benchmarks that include compositional equivalences can lead to significant overestimation of the generalization abilities of direct models, and benchmarks that include both module coverage violations and compositional equivalences can exacerbate shortcut learning. As a case study, we demonstrate the role of compositional equivalence on the generalization performance of the pre-trained model Llama3-8B on the code-output prediction task in the coding benchmark LiveCodeBench.

## 1 Introduction

Compositional generalization—generalizing to unseen module combinations after training on a limited set—is a strong form of out-of-distribution generalization. This kind of compositional reasoning is relevant to many real-world tasks, such as predicting outputs of new software programs, planning novel action sequences, or constructing proofs from known operations. As transformerbased models are increasingly used in real-world applications, such as software development, robotics, and scientific discovery, it is useful to know the conditions under which they will reliably exhibit compositional generalization.

![](images/ad02d288adf8cd04cfeea9486c7d14bb9819a4007a9d76d06f855fee3619a202.jpg)  
(a) Task-based compositional generalization

![](images/1f37c2ec1d37f813d97a0e773d6b8e2c7a6e2ab5849d1c28a01f6a8b020aad8a.jpg)  
(b) Composition equivalence

![](images/6926f0b89aaf477411f23021efa7199fb1e96fe49b821c9ad4e6aef6d32036e3.jpg)  
(c) Module coverage  
Figure 1: (a) A sequential composition task combines “atomic" functions (e.g., sort, mode, filter). Task-based compositional generalization is the ability to generalize to unseen sequences after training on a subset. (b) Composition equivalence: Different sequences are composition equivalent if they reduce to the same end-to-end function. For example, one equivalence class consisting of the first four sequences reduces to returning the most frequently occurring vowel in a set of strings; direct models succeed on the test sequence (mode o filter o sort) from this class but fail on the remaining two test sequences from a different composition equivalence class. (c) Module coverage: Examples of coverage failures include position-wise (e.g., function 2 never appearing in the first position, shown with dashed lines) and pair-wise (e.g., missing pair (2,1) in the training set, shown with solid lines).

In this work, we focus on one of the simplest tasks in which compositional generalization can be evaluated: sequential compositional tasks. Such tasks consist of a sequence of “atomic" modules that transform input data into output data (see Figure 1(a) for an example), such as in software programs We call this kind of generalization task-based compositional generalization, since the input-output mapping task may differ as the sequence of modules changes.

Empirical evaluation of compositional generalization is more challenging than standard evaluation of in-distribution generalization because unseen compositions can lead to an unknown degree of shift between train and test distributions. In pre-trained large language models (LLMs), evaluation becomes even more challenging because the content of their training sets is largely unknown, making it difficult to determine whether high performance reflects compositional generalization or whether the test compositions were already seen in the training data. An additional challenge is that compositional settings admit a combinatorially large number of train-test partitions of the composition space, making exhaustive evaluation difficult [Camposampiero et al., 2025].

Due to these challenges, most existing empirical work [Hupkes et al., 2020, Csordás et al., 2021, Ontanon et al., 2022, Dziri et al., 2023, Song et al., 2025, Ramesh et al., 2024] evaluates compositional generalization using a limited set of ad hoc train-test splits, often created via random partitions, which leads to inconsistent findings. For example, prior work [Ramesh et al., 2024, Abedsoltan et al., 2025, Xu et al., 2024] has shown that direct models (trained on final outputs) have limited generalization in certain settings, yet can perfectly generalize when the module function type or composition length is changed. Many existing explanations for this puzzling behavior focus on model-specific characteristics, such as insufficient computational depth [Peng et al., 2024].

In this work, we provide an alternative data-centric explanation, showing that significant variability in generalization performance can arise purely from the characteristics of benchmark design and how the train and test sets are created, irrespective of the model architecture. Through a systematic and controlled experimental design¹ that allows precise control over the degree of distribution shift between train and test compositions, we identify two key factors that together explain the substantial variability observed across transformer variants—both direct (trained on final outputs) and step-by-step models (trained on intermediate outputs).

The first factor is composition equivalence (Figure 1(b))—where distinct module sequences reduce to the identical end-to-end function. Composition equivalence is a previously unknown driver of compositional generalization. We observe that when train and test splits share such equivalences, direct models appear to compositionally generalize, and performance is strongly predicted by the degree of shared equivalences. We observe that the performance drops to near zero when the overlap between composition equivalences in the training and test sets is eliminated. This implies that many positive results for direct models in the compositional generalization literature may be artifacts of benchmark design rather than evidence of compositional generalization.

The second factor is module coverage (Figure 1(c))—the extent to which each atomic module appears at different positions and in similar contexts between train and test sets. We observe that as positionwise and pair-wise distributional gaps increase, generalization degrades for both direct models and step-by-step models. We observe that this effect is further amplified by composition equivalences, which encourage shortcut learning. Module coverage builds on the previously studied notion of data diversity [Ahuja and Mansouri, 2025], but we introduce a novel continuous measure of coverage shift and study its interaction with composition equivalence.

Having established a precise understanding of these factors in controlled settings with GPT-2-style models, we extend our analysis to larger pre-trained models (Gemma 3-1B and Llama 3-8B). We show that when composition equivalences are introduced into the fine-tuning set, compositional generalization performance on the code-output prediction task in LiveCodeBench—an existing coding benchmark—improves significantly, indicating that this effect extends to real-world LLM usage. These findings have concrete implications for both existing evaluations and future benchmark design. Most existing benchmarks define “unseen" compositions syntactically—whether a module sequence appeared in training—but we argue that a more appropriate definition is functional: if a test composition produces a novel end-to-end mapping as compared to the training compositions.

## 2 Related Work

Our work addresses key gaps in understanding the conditions under which compositional generalization occurs in transformers. An additional discussion of related work is provided in A.2.

Systematic evaluation of compositional generalization in transformers: The closest framework to ours is Ramesh et al. [2024]. Our work differs in two key ways. First, they focus primarily on a single module type (a random bijection, similar to a one-to-one lookup table), whereas we include modules with varied functional properties more representative of real-world compositional reasoning tasks (e.g., program execution in software), through which we discover the novel factor of composition equivalence. Second, they combine within-k and cross-k evaluations (whether the composition length k differs between train and test) using identity modules as padding, whereas we separate these settings and show that performance differs substantially between them. Our findings on composition equivalences explain the limited direct model generalization observed in their work, as well as why direct model performance improves across different module function types [Xu et al., 2024]. Chang et al. [2025] also discusses functional equivalence and coverage principles, seemingly related concepts to composition equivalence and module coverage, but focuses on data-based generalization—novel data combinations within a fixed task—whereas we study task-based generalization. Consequently, their functional equivalence operates at the input data level, while composition equivalence operates at a higher abstraction, relating compositions by their shared end-to-end function.

Relation of module coverage to prior work: Module coverage is related to previously studied concepts of compositional support [Wiedemer et al., 2023], connected support [Schug et al., 2024 Redhardt et al., 2025], and sufficient data diversity [Ahuja and Mansouri, 2025], required for successful compositional generalization across different domains (e.g., vision). These requirements state that the training distribution should include non-zero position-wise support for each module or module combination and, in general, sufficiently diverse combinations. Prior work typically assumes uniform sampling across train and test to ensure diversity, thereby covering only a limited range of distribution shifts. Task-based compositional generalization work [Ramesh et al., 2024, Abedsoltan et al., 2025] studied model sensitivity to extreme position-wise support violations. Our formulation provides a continuous metric that captures a broader range of coverage shifts, spanning from uniform sampling to extreme support violations. We also provide a systematic procedure for constructing train/test splits with varying degrees of module-coverage-based shifts (Section 3.3).

## 3 Distribution shifts characterizing task-based compositional generalization

Consider a set of n module functions $\mathcal { F } = \{ f _ { 1 } , \ldots , f _ { n } \}$ , where each $f _ { i } : \mathcal { V } ^ { m } \to \mathcal { V } ^ { m }$ maps input sequences to output sequences over vocabulary V. A sequential composition of length k is an ordered sequence $( f _ { 1 } , \ldots , f _ { k } )$ applied as $\mathbf { y } = ( f _ { k } \circ \cdot \cdot \cdot \circ f _ { 1 } ) ( \mathbf { x } )$ , with intermediate outputs $\mathbf { y } _ { j } = ( f _ { j } \circ \cdot \cdot \cdot \circ f _ { 1 } ) ( \mathbf { x } )$ . Compositions are represented using module tokens $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { n } \}$ , with input sequence $( t _ { 1 } , \ldots , t _ { k } , x _ { 1 } , \ldots , x _ { m } )$ ; models are trained auto-regressively to predict the output. Data tokens are sampled uniformly from $\mathcal { V } ^ { m }$ in both train and test distributions.

Definition 3.1 (Task-based compositional generalization). $L e t ~ \mathcal { T } ^ { k }$ denote the space of module sequences of length k. Task-based compositional generalization is evaluated via train-test distributions with disjoint supports: supp $( \mathcal { P } _ { t r a i n } ^ { \mathcal { T } } ) ^ { \bullet } \cap s u p p ( \mathcal { P } _ { t e s t } ^ { \mathcal { T } } ) = \emptyset$ , where training sequences are drawn from $\Omega \subset \mathcal { T } ^ { k }$ and test sequences from ${ \mathcal { T } } ^ { k } \setminus \Omega$

## 3.1 Composition equivalence based distribution shift

Two distinct sequential compositions are equivalent if they induce the same input-output mapping over a given set of data tokens.

Definition 3.2 (Composition equivalence and equivalence class). Two compositions $F = f _ { 1 }$ O $\cdots \circ f _ { k }$ and $F ^ { \prime } = f _ { 1 } ^ { \prime } \circ \cdot \cdot \cdot \circ f _ { k } ^ { \prime }$ are equivalent over input subspace $\mathcal { S } \subseteq \mathcal { V } ^ { m } \mathrm { ~ } i f \bar { F } ( \mathbf { x } ) = F ^ { \prime } ( \mathbf { x } ) \mathrm { ~ } j$ for all $\mathbf { x } \in S$ . The equivalence class of F is:

$$
[ F ] _ { \mathcal { S } } = \{ F ^ { \prime } : F ^ { \prime } ( \mathbf { x } ) = F ( \mathbf { x } ) \forall \mathbf { x } \in \mathcal { S } \}
$$

Some simple examples of composition equivalence include:

1. Identity-based exact equivalence: For any composition $F = f _ { 1 } \circ \cdot \cdot \cdot \circ f _ { k } .$ , inserting id at any position yields an equivalent sequence: $\bar { F _ { 0 } } : = \bar { f _ { 1 } } \circ \cdot \cdot \cdot \circ f _ { k } \circ \mathrm { i d } , \ldots , F _ { k } : = \mathrm { i d } \circ f _ { 1 } \circ \cdot \cdot \cdot \circ \bar { f _ { k } }$

2. Approximate equivalence due to module properties: Consider filter (extracts vowels), mode (most frequent character), and sort (lexicographic order). For strings where the most frequent character is a vowel, sort(mode(filter(·))) and filter(sort(mode(·))) are equivalent (Figure 1(b)).

3. Different implementations of the same function: Recursive and iterative implementations of the Fibonacci sequence.

Definition 3.3 (Degree of shared composition equivalences). The degree of shared equivalences between train and test is the proportion of test compositions that share an equivalence class with at least one training composition:

$$
\delta _ { e q u i v } = \frac { | \{ F _ { B } \in B : \exists F _ { A } \in A s . t . [ F _ { B } ] _ { S } ^ { g } = [ F _ { A } ] _ { S } ^ { g } \} | } { | B | }
$$

Conversely, $1 - \delta _ { \mathrm { e q u i v } }$ quantifies the extent to which test compositions exhibit novel end-to-end behavior unseen during training. While task-based compositional generalization is defined syntactically in terms of unseen module combinations, composition equivalence captures the semantic novelty of unseen compositions.

## 3.2 Module coverage based distribution shift

Module-coverage-based distribution shift quantifies how frequently atomic modules appear in specific positions or contexts at test time relative to training. Such shifts can introduce spurious correlations between positions and modules, causing generalization failures when these correlations break. We define two metrics capturing KL divergence in the position-wise and pair-wise module distributions, as KL divergence is asymmetric and measures shifts in the test distribution relative to training.

Definition 3.4 (Position-wise and pairwise module coverage shift). The empirical position-wise and pairwise distributions are:

$$
\hat { P } ( T _ { j } = t _ { i } ) = \frac { 1 } { N } \sum _ { s = 1 } ^ { N } \mathbb { I } ( T _ { j } ^ { ( s ) } = t _ { i } ) , \quad \hat { P } ( T = t _ { i } , T ^ { \prime } = t _ { i ^ { \prime } } ) = \frac { 1 } { N ( k - 1 ) } \sum _ { s = 1 } ^ { N } \sum _ { j = 1 } ^ { k - 1 } \mathbb { I } ( T _ { j } ^ { ( s ) } = t _ { i } , T _ { j + 1 } ^ { ( s ) } = t _ { i ^ { \prime } } )
$$

$$
D _ { p o s } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } D _ { K L } ( \hat { P } _ { t e s t } ( T _ { i } ) \parallel \hat { P } _ { t r a i n } ( T _ { i } ) ) , \quad D _ { p a i r } = D _ { K L } ( \hat { P } _ { t e s t } ( T , T ^ { \prime } ) \parallel \hat { P } _ { t r a i n } ( T , T ^ { \prime } ) )
$$

## 3.3 Creating benchmarks with a controllable degree of distribution shifts

We first describe two data sets that differ in their composition equivalences, and then describe how we create benchmarks with varying degrees of shift.

## 3.3.1 Data sets

The uniform data set eliminates composition equivalence by default, serving as a control, while the diverse data set captures the richer equivalence structure of real-world functions.

Uniform data set: We use six random bijection functions, each mapping input tokens to output tokens via a one-to-one lookup table [Ramesh et al., 2024]. When mappings are unique, non-identity, and uniformly random, any k-module composition produces a unique labeling function with high probability. This implies that no two composition sequences are equivalent. However, similar labeling functions are still possible in rare cases; for example, two bijections can be inverses of each other, leading to an identity labeling function. We empirically validate uniqueness by comparing outputs of every pair of compositions on a large number of random strings (1000 strings), and we find no duplicate labeling functions. Adding an identity module introduces exact identity-based equivalence, allowing us to evaluate the effect of a complete lack of equivalence versus exact equivalence in a controlled setting.

Diverse data set: We create a dataset of string-manipulation functions inspired by RASP primitives [Weiss et al., 2021, Zhou et al., 2024] that transformers can represent and learn. We use simple functions to ensure observed limitations reflect compositional reasoning failures rather than insufficient model capacity. The six module functions—sort, concatenate, filter, shift, union, mode—are defined as follows: shift maps each character one position forward (a→b, z→a); sort rearranges characters lexicographically; mode returns the most frequent character (ties broken lexicographically); concatenate appends the second string to the first; union returns unique characters in order of first occurrence; and filter extracts vowels in order. We pad outputs with <SPACE> to maintain fixed length m. These functions induce composition equivalences: filter and mode create many-to-one mappings enabling shortcut reasoning; concatenate and union produce identical outputs for strings with distinct characters; and shift and sort are commutative for many strings. These properties result in both exact and approximate equivalences (Figure 11).

## 3.3.2 Constructing train-test splits with varying degrees of composition equivalences and module coverage

Composition equivalences: In the diverse benchmark, equivalence classes are not pre-defined as in the case of the uniform benchmark. Therefore, we learn equivalence classes from data by using the following approach.

• Compute the pairwise composition equivalence score $\begin{array} { r } { C E ( F , F ^ { \prime } ) = \frac { 1 } { N } \sum _ { s = 1 } ^ { N } \mathbb { I } [ F ( \mathbf { x } ) = } \end{array}$ $F ^ { \prime } ( \mathbf { x } ) ]$ between two task sequences as the proportion of inputs for which the final outputs match when evaluated over a large number of $N = 1 0 0 0$ inputs sampled from the vocabulary.

• Construct an undirected graph based on the equivalence scores, where each node denotes a composition sequence and an edge denotes the score strength above a threshold of 0.01. That is, two compositions are considered equivalent if they produce the same outputs in at least 1% of cases. We chose this low, conservative threshold so that even weak composition equivalences are captured when learning equivalence classes from data.

• Learn equivalence classes by finding connected components in the undirected graph.

• Disjoint splits $( \delta _ { \mathbf { e q u i v } } = 0 ) \colon$ Split train and test data by equivalence class—no class appears in both sets. We consider 50-50 splits to accommodate varying degrees of equivalence, as explained below.

• Splits with varying degrees of composition equivalences $( \delta _ { \mathbf { e q u i v } } \in ( 0 , 1 ] )$ : Increase shared equivalences by swapping roughly half of the members of each test equivalence class with members from a training class. This systematically increases test sequences with shared equivalences in proportion to class sizes, while maintaining a constant test set size.

Module coverage: We start with uniform position-wise and pair-wise distributions $P _ { \mathrm { t r a i n } } ( T _ { j } =$ $\begin{array} { r } { t _ { i } ) = P _ { \mathrm { t e s t } } ( T _ { j } = t _ { i } ) = \frac { 1 } { n } } \end{array}$ and introduce a bias strength parameter $\alpha \in [ 0 , 1 ]$ that skews sampling probabilities in complementary directions for train and test at each position. $\operatorname { A t } \alpha = 0 ,$ distributions remain uniform; as $\alpha  1$ , the position-wise KL divergence $D _ { \mathrm { p o s } }$ increases monotonically, creating larger coverage shifts while maintaining disjoint train-test sets. See Section A.10 for details.

## 3.4 Evaluation setup

Within-k and cross-k generalization: In within-k, train and test are created using randomly sampled disjoint 80/20 splits of $\gamma ^ { k }$ for $k \in \{ 2 , \ldots , 6 \}$ , where k is the number of non-identity modules. In cross-k, models trained on $\mathcal { T } ^ { k }$ are evaluated on $\mathcal { T } ^ { k ^ { \prime } }$ for $k \neq k ^ { \prime }$ , with identity modules used as padding to align sequence lengths.

To enable a fair comparison between the two settings and assess model sensitivity to identity modules, we evaluate within-k in two variants: (1) without identity, where no identity module is used; (2) with identity, where compositions are padded with identity modules to a fixed length of 7, mirroring cross-k. For example, in the without-identity variant, a within-2 model trained on {shift o sort} is evaluated on {sort o shift }. In the with-identity variant, the same model is trained on {ido shift o sort o ido ido ido id} and evaluated on {id o sort o shift o id o $\mathbf { i d } \circ \mathbf { i d } \circ \mathbf { i d } \}$ . A cross-k model trained on $k = 6$ compositions such as {filterosort omodeoshiftounionoidoconcatenate} is evaluated on $k ^ { \prime } = \bar { 3 }$ compositions such as $\{ { \tt i d } \circ { \tt f i l t e r } \circ$ sort o id o mode $\circ { \dot { \mathsf { i d } } } \circ { \dot { \mathsf { i d } } } \}$

Models: We train four transformer variants—direct and step-by-step models, each with absolute and relative positional embeddings [Shaw et al., 2018]—using the nanoGPT architecture [Karpathy, 2023] (3 layers, 6 heads). We use this small architecture for efficient evaluation across many train-test splits; larger models (12 layers, 12 heads) show similar generalization trends (Appendix A.5.2). Training uses 100k samples, and testing uses 10k samples uniformly sampled per composition, with input tokens of length six sampled from $\mathcal { V } = \{ a , \ldots , z \}$ . Models are trained for 100–200 epochs until training loss saturates, achieving 100% training accuracy and 90–100% validation accuracy on unseen input strings within training compositions (see Appendix A.5).

Evaluation criteria: We report the mean accuracy on unseen sequences using exact-match scoring (1.0 for perfect predictions and 0.0 otherwise), averaged over five random seeds.

## 4 Explaining compositional generalization performance via composition equivalence and module coverage

We first discuss results that highlight the interplay between composition equivalence and module coverage in within-k and cross-k settings. We then isolate the individual role of each factor using controlled splits created in Section 3.3.2, and show that the degree of each shift strongly predicts the performance of both direct and step-by-step models.

## 4.1 Direct models show near-perfect generalization in some settings but fail completely in others due to composition equivalences

In Figures 2 and 3, direct models show a surprising range of performance variability: near-perfect generalization (70–100% accuracy) in some settings, and complete failure (near-zero) in others, even for the simplest $k = 2$ compositions (see Figure 2(d)). As we show below, this variability can be explained by the presence or absence of shared composition equivalences in the train-test split.

Cases where direct models show significant generalization:

• Diverse benchmark with or without identity modules (within-k): Random 80/20 splits with identity modules contain exact identity-based equivalences, resulting in near-perfect performance (Figure 2a). Without identity modules (Figure 2b), function properties produce exact and approximate equivalences (e.g., sort o mode ≡ mode o sort) that direct models exploit, resulting in significant performance — especially for $k = 4 { - } 6$

• Uniform benchmark with identity modules (within-k): Identity modules introduce exact equivalence classes, again resulting in near-perfect performance (Figure 2c).

• Diverse benchmark near training k (cross-k): Function-property-based equivalences exist across nearby k values (e.g., (sort o filter), (sort o filter omode) are equivalent for many input strings), explaining non-zero performance when evaluation k is close to training k (Figures 3a–b).

![](images/713638b5490f9e00ea24dbf460d84215dc3c11176444ae704ffe56ded2b2ac60.jpg)  
Direct (Absolute)

![](images/23339b4fa2eee4897d2c675a5b178507280c8f2b86a25abbd354bf67ef2d13b2.jpg)  
Direct (Relative)

![](images/b0c35d67c12b2e2dc11ddb1673bb5b5ea7f75b5401e9d42faa9213e3d857acb5.jpg)  
Step-by-Step (Absolute)

Step-by-Step (Relative)  
![](images/7868d2591a374ea59d3e8ad21ea9e477a7628dc83b128f741dfc4eaebef3fe9b.jpg)  
Figure 2: Within-k evaluation: Direct models fail completely (0% accuracy) on uniform bijections without identity tokens (Figure (d)), while step-by-step models maintain near-perfect performance across all conditions. We observe that including identity tokens or diverse function types seems crucial for the generalization performance of direct models.

![](images/2e7354db31da1699e87a1b25bc571870e53d82128a8e168e07235eaf98ed04ad.jpg)  
Direct (Absolute)

![](images/9176a062b64da45ce527a39b636a5ba2d49513f858ac9bee9a1667d08c5110a2.jpg)  
Direct (Relative)

![](images/aeef3a9b0074663f3fc39bd1313b599756d49cdcffd6d7a37fe67696d027c205.jpg)  
Step-by-Step (Absolute)

![](images/9c3cec96e4c8ce24d7f45bf64dfde69271dcdc2c5650ae4c7ce2c9281a08da85.jpg)  
Step-by-Step (Relative)  
Figure 3: Cross-k evaluation: For the diverse benchmark (a-b), direct models perform well near training k as equivalences persist across nearby k values. Direct models fail on the uniform benchmark (c-d) regardless of identity tokens, since identity-based equivalences in the uniform benchmark are k-specific. Step-by-step models are more robust but still exhibit significant variability.

Cases where direct models fail completely:

• Uniform benchmark without identity modules (within-k): No composition equivalences exist by construction, causing performance to drop to exactly zero across all $k \in \left\{ 2 , \ldots , 6 \right\}$

• Uniform benchmark (cross-k): Identity-based equivalences in uniform benchmark are k-specific—changing the number of non-identity modules changes end-to-end behavior—so they do not carry over across k, causing complete failure (Figures 3c–d).

• Diverse benchmark far from training k (cross-k): As evaluation k moves further from training k, function-property-based equivalences reduce due to changing end-to-end functions, and module coverage shifts also increase due to the varying distribution of identity modules, causing performance to drop sharply towards zero e.g., k=1–4 (Figure 3a), and k=1, 5, and 6. (Figure 3b)

In summary, direct models fail to generalize to compositions that produce novel end-to-end functions.

Composition equivalence is the primary mechanism through which direct models achieve compositional generalization: Figure 4(a) shows that accuracy is strongly predicted by the degree of shared equivalences, rising sharply from 0% (no sharing) to roughly 80% (full sharing)—an effect that also holds for fine-tuned Gemma-3-1B models, confirming it is not limited to small GPT-2-style transformers. We observe that performance does not reach 100% even with complete sharing, since some equivalences are approximate for a subset of inputs and some classes have only 4–6 members, making training subsets insufficient for perfect generalization. Figure 4(b) further shows that accuracy is substantially higher for shared classes than non-shared ones, and that even some shared classes have lower performance, especially when class size is small. Together, these results establish that shared equivalence classes are necessary but not sufficient for compositional generalization in direct models. We observed similar results for the uniform benchmark (see Figure 10 in the Appendix). Our findings on composition equivalence also explain findings in Ramesh et al. [2024]: their limited but not near-zero generalization performance for direct models on the uniform benchmark (Figure 6 in Ramesh et al. [2024]) is likely a result of reporting accuracy over both train and test compositions rather than test alone and combining within-k and cross-k evaluations with identity padding, which inadvertently introduces equivalences.

![](images/82363a6a04e2c9ad68e243452b1492710060bac9b5aeba2ea7cafae7cadaedc3.jpg)  
Figure 4: Isolating the effect of composition equivalence in the diverse benchmark (k = 6): (a) Generalization performance increases with the degree of shared composition equivalence classes between train and test. (b) The direct model generalizes primarily to test sequences that share equivalence classes and fails on classes exhibiting novel end-to-end behavior.

Table 1: Sensitivity of direct-model accuracy to the equivalence threshold: We report accuracy on train/test splits with 0% and 100% shared composition equivalences, where equivalence classes are learned at different thresholds. At low thresholds, weak equivalences are captured, so the 0% shared split contains no equivalences of any strength and generalization is poor. At high thresholds, some equivalences go undetected and remain in the 0% shared split, which inflates generalization performance.
<table><tr><td>Threshold</td><td>Accuracy: 0% shared</td><td>Accuracy: 100% shared</td></tr><tr><td>0.01</td><td>0.04</td><td>0.70</td></tr><tr><td>0.25</td><td>0.08</td><td>0.73</td></tr><tr><td>0.50</td><td>0.28</td><td>0.84</td></tr><tr><td>0.75</td><td>0.68</td><td>0.88</td></tr><tr><td>0.95</td><td>0.84</td><td>0.91</td></tr></table>

Sensitivity of the effect of shared composition equivalences to the threshold value: We run the composition equivalence leakage experiment using equivalence classes learned with a threshold equivalence score of 0.01. To assess how sensitive the results are to higher thresholds, we also analyze accuracy at thresholds of 0.25, 0.5, 0.75, and 0.95. Results are presented in Table 1. At every threshold, we observe that accuracy increases monotonically from 0% to 100% shared equivalence. However, higher thresholds underestimate true equivalence sharing, which overestimates apparent generalization for a given shared equivalence value. For example, at threshold=0.95, the nominally 0% shared condition has an accuracy of 0.85, whereas at threshold=0.01 the same condition results in an accuracy of 0.01. Compositions that share only weak equivalences (those producing matching outputs on fewer than 95% of inputs) are placed in separate classes under threshold=0.95. As a result, the 0% shared" condition at this threshold still contains test compositions that share undetected functional equivalences with the training data, and part of the 0.85 accuracy stems from these undetected

Step-by-Step (Relative)Direct (Relative)

equivalences. The low threshold of 0.01 avoids this inflation: its “0% shared" condition consists of disjoint classes with no shared equivalence, whether weak or strong.

## 4.2 Module coverage shifts affect both direct and step-by-step models

Step-by-step models exhibit robust generalization across all splits in within-k evaluation (Figure 2) but perform well only near the training k in cross-k evaluation (Figure 3), showing significant variability We observe that this is primarily due to the position-wise and pair-wise module distributions shifting substantially as the evaluation k moves away from training k (see divergence values in Figure 18 in the Appendix).

Effect of module coverage shifts and interaction between module coverage and composition equivalence: Compositional generalization performance degrades monotonically as position-wise KL divergence increases across both benchmarks and model variants, indicating that models primarily rely on uniform module sampling for successful generalization (Figure 5). On the diverse benchmark, the presence of composition equivalences accelerates this degradation for all models (Figure 5(a)), as equivalences encourage shortcut learning that compounds coverage-based failures. In stepby-step models, absolute position encodings are more sensitive to position-wise shifts than relative encodings, particularly on the uniform benchmark (Figure 5(b)). For pairwise distribution shifts (Figure 19 in the Appendix), models are more sensitive to the diverse benchmark than the uniform benchmark, again due

![](images/20c204d15469e052f3d481012c20be094c9ac4ab1fce0dd04fb895750b5d290f.jpg)

![](images/0c901eac0ee19b255e50427b7e6b1fbda090b12a5f24d81b3d7d73368d81e327.jpg)  
Figure 5: Effect of position-wise distribution shift: As the position-wise KL divergence increases, the compositional generalization performance of both direct and step-by-step models decreases.

to equivalence-driven shortcuts; on the uniform benchmark, performance remains robust except in the case of extreme support violations.

Finally, Figure 10(a) (Appendix) reveals an additional interaction between the two factors: direct models with absolute position embeddings fail to learn correct equivalences, mainly due to module coverage failures (e.g., test compositions always starting with id). This shows that composition equivalence and module coverage are not independent—a dependence that naturally arises in real-world settings where certain module orderings are more prevalent, producing more or fewer equivalences.

## 4.3 Case Study: Composition equivalence in Llama3-8B model on LiveCodeBench

In Figure 6, we present the performance of the Llama3-8B model on the code output prediction task on the LiveCodeBench benchmark (see evaluation details in A.8.1). The data set contains multiple program implementations grouped into equivalence classes sharing the same functionality but differing in compositional structure. We create train-test splits with varying degrees of class leakage based on function names (heldout = 0, 0.5, 1.0), then fine-tune with LoRA and evaluate on test splits. We keep the total number of samples in the fine-tuning set constant across class-leakage values to control for sample-size effects. We treat test accuracy as a measure of compositional generalization performance, under the assumption that the pre-training data did not include training on code-output prediction tasks—an assumption we validate by reporting baseline performance for the non-fine-tuned model (No LoRA).

Results: In Figure 6(a), we observe that as we leak composition equivalences into training, it substantially increases overall accuracy (h=0.5, 1.0), whereas performance on held-out non-leaked functions (h=0.0) improves only minimally. This holds regardless of function-name anonymization, in which we replace function names with random strings to rule out the possibility that the model is simply learning to map function names to correct outputs. In Figure 6(b), we report per-function accuracy across different LoRA configurations. For most functions, leaking equivalences into training improves performance on the remaining held-out equivalences. For some functions (e.g., distanceTraveled), non-trivial performance is only achievable when equivalences are leaked. However, a small number of functions (e.g., minLength) show improvement even without equivalence leakage, which may be due to unaccounted equivalences—for instance, structurally similar functions operating on different data types with different function names. These results demonstrate that composition equivalences also affect compositional generalization in large pre-trained models on real-world tasks.

![](images/af917943d6e17f4474fe5bcf8897ce88415f592ee25fbe3639720457a16a3ca8.jpg)

![](images/501192f338fa28096730f9f047a7474c4ba3c0c094d8afd284a9123e78740472.jpg)  
Figure 6: (a) Code output prediction performance of Llama3-8b on LiveCodeBench benchmark: We show the accuracy of the base model (No LoRA) and three LoRA fine-tuned models on splits with varying ratios of shared equivalence classes (defined by the same function names) (held out h=0, 0.5, 1.0). As splits share composition equivalences, accuracy improves substantially. (b) Performance grouped by equivalence classes: We observe that performance on most functions only improves when equivalences are leaked into the fine-tuning set.

## 5 Conclusion and Limitations

We identify two key factors, composition equivalence and module coverage, that together explain substantial variability in compositional generalization across transformer variants. A key takeaway from our work is rethinking what “unseen"means in sequential compositional generalization: rather than a syntactic definition, we argue for a functional one—a test composition is truly out-of-distribution if it produces a novel end-to-end mapping relative to all training compositions. Our findings also reveal concrete failure modes in current evaluation protocols: benchmarks containing composition equivalences can significantly overestimate direct model generalization, and seemingly reasonable design choices (such as identity-token padding or combining within-k and cross-k evaluation) can be highly misleading. Benchmarks that include module coverage shifts and compositional equivalences can further exacerbate shortcut learning. We hope these findings motivate a more careful benchmark design to reliably assess compositional generalization.

Limitations: Our findings are mainly established on sequential compositional tasks, and it remains an open question how composition equivalence and module coverage generalize to more complex compositional structures (e.g., hierarchical or graph-based compositions). In our analysis on Live-CodeBench, individual implementations sometimes differed substantially in their compositional structure due to the use of different libraries. Moving away from a synthetic setting means we can no longer control module coverage, length, order, or compositional structure. We therefore do not claim that compositional equivalence is the only mechanism behind the observed performance gains; rather, the observed behaviors align strongly with our hypotheses about its effect on compositional generalization performance observed in real-world settings. Precisely isolating each factor requires either controlled synthetic settings—as we have done—or careful curation of real-world data sets with explicitly controlled train/test distribution shifts, which is an exciting future direction.

## Acknowledgements

We thank Sankaran Vaidyanathan for the helpful comments and suggestions on an earlier version of the paper.

## References

A. Abedsoltan, H. Zhang, K. Wen, H. Lin, J. Zhang, and M. Belkin. Task generalization with autoregressive compositional structure: Can learning from d tasks generalize to dt tasks? arXiv preprint arXiv:2502.08991, 2025.

K. Ahuja and A. Mansouri. On provable length and compositional generalization, 2025. URL https://openreview.net/forum?id=HxmOhOxph2.

S. An, Z. Lin, Q. Fu, B. Chen, N. Zheng, J.-G. Lou, and D. Zhang. How do in-context examples affect compositional generalization? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11027–11052, 2023.

H. C.-Z. Anna, V. Ashish, U. Jakob, S. Ian, H. Curtis, S. Noam, D. Monica, E. Douglas, et al. Music transformer: Generating music with long-term structure. arXiv preprint, 2018.

G. Camposampiero, P. Barbiero, M. Hersche, R. Wattenhofer, and A. Rahimi. Scalable evaluation and neural models for compositional generalization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= heQsyrMDzm.

H. Chang, J. Park, H. Cho, S. Yang, M. Ko, H. Hwang, S. Won, D. Lee, Y. Ahn, and M. Seo. The coverage principle: A framework for understanding compositional generalization. arXiv preprint arXiv:2505.20278, 2025.

R. Csordás, K. Irie, and J. Schmidhuber. The devil is in the detail: Simple tricks improve systematic generalization of transformers. In EMNLP 2021-2021 Conference on Empirical Methods in Natural Language Processing, Proceedings, pages 619–634. Association for Computational Linguistics (ACL), 2021.

M. Du, F. He, N. Zou, D. Tao, and X. Hu. Shortcut learning of large language models in natural language understanding. Commun. ACM, 67(1):110–120, Dec. 2023. ISSN 0001-0782. doi: 10.1145/3596490. URL https://doi.org/10.1145/3596490.

N. Dziri, X. Lu, M. Sclar, X. L. Li, L. Jiang, B. Y. Lin, S. Welleck, P. West, C. Bhagavatula, R. Le Bras, et al. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36:70293–70332, 2023.

S. Garg, D. Tsipras, P. S. Liang, and G. Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022.

R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

D. Hupkes, V. Dankers, M. Mul, and E. Bruni. Compositionality decomposed: How do neural networks generalise? Journal of Artificial Intelligence Research, 67:757–795, 2020.

N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=chfJJYC3iL.

A. Karpathy. nanogpt. https://github.com/karpathy/nanoGPT, 2023. The simplest, fastest repository for training/finetuning medium-sized GPTs.

B. Lake and M. Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International conference on machine learning, pages 2873–2882. PMLR, 2018.

Y. Li, M. E. Ildiz, D. Papailiopoulos, and S. Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19565–19594. PMLR, 23–29 Jul 2023a. URL https://proceedings.mlr.press/v202/1i231.html.

Y. Li, K. Sreenivasan, A. Giannou, D. Papailiopoulos, and S. Oymak. Dissecting chain-ofthought: Compositionality through in-context filtering and learning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 22021–22046. Curran Associates, Inc., 2023b. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 45e15bae91a6f213d45e203b8a29be48-Paper-Conference.pdf.

Z. Li, H. Liu, D. Zhou, and T. Ma. Chain of thought empowers transformers to solve inherently serial problems. In The Twelfth International Conference on Learning Representations, 2024.

M. Neishi and N. Yoshinaga. On the relation between position information and sentence length in neural machine translation. In M. Bansal and A. Villavicencio, editors, Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 328–338, Hong Kong, China, Nov. 2019. Association for Computational Linguistics. doi: 10.18653/v1/K19-1031. URL https://aclanthology.org/K19-1031/.

S. Ontanon, J. Ainslie, Z. Fisher, and V. Cvicek. Making transformers solve compositional tasks. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3591–3607, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long 251. URL https://aclanthology.org/2022.acl-long.251/.

B. Peng, S. Narayanan, and C. Papadimitriou. On limitations of the transformer architecture. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= KidynPuLNW.

J. Petty, S. van Steenkiste, I. Dasgupta, F. Sha, D. Garrette, and T. Linzen. The impact of depth on compositional generalization in transformer language models. In 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2024, pages 7232–7245. Association for Computational Linguistics (ACL), 2024.

R. Ramesh, E. S. Lubana, M. Khona, R. P. Dick, and H. Tanaka. Compositional capabilities of autoregressive transformers: A study on synthetic, interpretable tasks. In International Conference on Machine Learning, pages 42074–42103. PMLR, 2024.

F. Redhardt, Y. Akram, and S. Schug. Scaling can lead to compositional generalization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https: //openreview.net/forum?id=hZtOdaVIZi.

A. Ruoss, G. Delétang, T. Genewein, J. Grau-Moya, R. Csordás, M. Bennani, S. Legg, and J. Veness. Randomized positional encodings boost length generalization of transformers. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1889–1903, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-short.161. URL https://aclanthology.org/2023.acl-short.161/.

S. Schug, S. Kobayashi, Y. Akram, M. Wolczyk, A. M. Proca, J. von Oswald, R. Pascanu, J. Sacramento, and A. Steger. Discovering modular solutions that generalize compositionally. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=H98CVcX1eh.

P. Shaw, J. Uszkoreit, and A. Vaswani. Self-attention with relative position representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 464–468, 2018.

J. Song, Z. Xu, and Y. Zhong. Out-of-distribution generalization via composition: a lens through induction heads in transformers. Proceedings of the National Academy of Sciences, 122(6): e2417182122, 2025.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper\_ files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

B. Wang, X. Yue, Y. Su, and H. Sun. Grokking of implicit reasoning in transformers: A mechanistic journey to the edge of generalization. Advances in Neural Information Processing Systems, 37: 95238–95265, 2024.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

G. Weiss, Y. Goldberg, and E. Yahav. Thinking like transformers. In International Conference on Machine Learning, pages 11080–11090. PMLR, 2021.

T. Wiedemer, P. Mayilvahanan, M. Bethge, and W. Brendel. Compositional generalization from first principles. Advances in Neural Information Processing Systems, 36:6941–6960, 2023.

Z. Xu, Z. Shi, and Y. Liang. Do large language models have compositional ability? an investigation into limitations and scalability. arXiv preprint arXiv:2407.15720, 2024.

H. Yang, H. Lu, W. Lam, and D. Cai. Exploring compositional generalization of large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 4: Student Research Workshop), pages 16–24, 2024.

Z. Zhang, P. Lin, Z. Wang, Y. Zhang, and Z.-Q. J. Xu. Initialization is critical to whether transformers fit composite functions by reasoning or memorizing. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id= YOBGdVaYTS.

H. Zhao, S. Kaur, D. Yu, A. Goyal, and S. Arora. Can models learn skill composition from examples? In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 102393–102427. Curran Associates, Inc., 2024a. doi: 10.52202/079017-3251. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/b99fd30559b520cc97447ba905040677-Paper-Conference.pdf.

L. Zhao, X. Feng, X. Feng, W. Zhong, D. Xu, Q. Yang, H. Liu, B. Qin, and T. Liu. Length extrapolation of transformers: A survey from the perspective of positional encoding. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9959–9977, Miami, Florida, USA, Nov. 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.582. URL https:// aclanthology.org/2024.findings-emnlp.582/.

D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet Q. V. Le, and E. H. Chi. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WZH7099tgfM.

H. Zhou, A. Bradley, E. Littwin, N. Razin, O. Saremi, J. M. Susskind, S. Bengio, and P. Nakkiran. What algorithms can transformers learn? a study in length generalization. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=AssIuHnmHX.

## A Appendix

## A.1 Large-language model usage

We have used large language models to minimally polish the writing in the main paper and to format the technical details in the Appendix.

## A.2 Related Work

In this section, we discuss additional related work.

Other studies of compositional generalization in transformers: A large body of work has studied compositional generalization in transformer-based models across a range of settings using both synthetic and real-world benchmarks. These works differ broadly in what constitutes a component—natural language instructions [Lake and Baroni, 2018, Hupkes et al., 2020], atomic skills [Zhao et al., 2024a], data inputs for a fixed algorithmic task [Ahuja and Mansouri, 2025], and module functions in program executions [Ramesh et al., 2024]—including in-context learning settings [Li et al., 2023a, Garg et al., 2022, Xu et al., 2024, An et al., 2023]. In this work, a component generally refers to an atomic, reusable function that forms part of a sequential composition, similar to modules used in program execution. Our key focus is on task-based generalization [Ramesh et al., 2024, Abedsoltan et al., 2025], in which the labeling function varies with the module sequence, thereby requiring extrapolation to new tasks.

Several works [Petty et al., 2024, Csordás et al., 2021, Ontanon et al., 2022, Zhang et al., 2024] investigate the effects of different hyperparameters and architectural choices on transformer performance across various generalization tasks. Others [Li et al., 2023a, Garg et al., 2022, Xu et al., 2024, An et al., 2023] have considered the compositional generalization capabilities of transformers when trained on samples using in-context learning. Several works [Li et al., 2023b, Wei et al., 2022, Zhou et al., 2023, Li et al., 2024] investigate compositional generalization in the context of chain-of-thought prompts, reporting performance gains across various compositional tasks. Yang et al. [2024] evaluates the performance of compositional capabilities of large-language models after instruction-tuning by testing on unseen combinations of instructions. Another line of work investigates the internal circuitry learned to enable compositional generalization [Song et al., 2025, Wang et al., 2024], identifying specific parts of the transformer architecture that affect generalization. Our work contributes to this area of research by investigating the effect of module orderings and data-generating characteristics on the compositional generalization of transformer models.

Training bias and shortcut learning: Shortcut learning [Geirhos et al., 2020, Du et al., 2023] is the phenomenon where models rely on superficial features in the training data, which may have spurious correlations with the output, instead of the robust features that capture the true underlying datagenerating process. We can view shortcut learning as one consequence of composition equivalence and module coverage violations. Understanding whether data exhibit compositional equivalence or violate module coverage provides a principled approach to mitigating spurious correlations and identifying cases where the model would exhibit robust out-of-distribution generalization.

## A.3 Module Function Definitions

## A.3.1 Uniform Benchmark

We define six random bijection functions $f : \mathcal { V } ^ { m } \to \mathcal { V } ^ { m }$ that randomly map each input data token $x _ { i } \in \mathcal { V } : = \{ a , b \dots z \}$ to a random output data token $y _ { i } \in \mathcal { V } : = \{ a , b \dots z \}$ , based on a pre-defined lookup table. We ensure there is a one-to-one, unique mapping between input and output data tokens, and each input data token doesn't map to itself. We assume input data length $m = 6 .$ The six tokens are sampled uniformly at random from V without replacement.

## A.3.2 Diverse Benchmark

1. shift (x): The shift function applies a predetermined bijective transformation to each character in the input string according to a fixed character substitution table.

$$
\mathbf { s h i f t } ( \mathbf { h } \ \textbf { j } \textbf { f } \textbf { s } \mathbf { d } \ \mathbf { h } ) \ = \textbf { i } \mathbf { k } \ \mathbf { g } \ \textbf { t } \textbf { e } \ \mathbf { i }
$$

2. sort (x) : The sorting function rearranges the characters of the input string in lexicographic (ascending alphabetical) order.

$$
\mathbf { s o r t } ( \texttt { c g m a h b } ) \texttt { = a b c g h m }
$$

3. mode (x) : The mode function returns the character that appears most frequently in the input string. In case of equal frequencies, the lexicographically smallest character is selected. mode $( \texttt { w j d n k k } ) \ = \texttt { k }$

4. concatenate $( \mathbf { x } , \mathbf { \mu } _ { \mathtt { y } } )$ : The concatenation function performs string concatenation, appending the second string to the end of the first string.

$$
\mathtt { j o i n } ( \mathtt { s } \texttt { e w l r z } , \texttt { y s e o q n } ) \texttt { = s e w l r z y s e o q n }
$$

5. union $( \mathbf { x } , \mathbf { \ y } )$ : The union function returns the ordered union of unique characters from both input strings, preserving the order of first occurrence within each string. union(s e w l r z, y s e o q n) = s e w l r z y o q n

6. filter(x) : The filter function extracts all vowel characters from the input string while preserving their original order. $\mathbf { f i l t e r } \left( \mathbf { d } \texttt { c f o j i } \right) \ = \ \mathbf { \circ } \ \mathbf { i }$

7. identity(x): The identity function returns the input string unchanged. identity $\mathbf { \Sigma } ( \mathbf { x } ) \mathbf { \Sigma } =$ x

We sample two input data tokens of length six from the vocabulary $\mathcal { V } = \{ a , b , c , \ldots z \}$ , as some of the module functions are binary. The maximum possible length of output is 12. We ignore the remaining data tokens if the length exceeds 12 to maintain a fixed output length. We pad with the space token in case the output is less than length 12, for example, in case of mode, filter, union functions.

## A.4 Transformer Architecture

We use a modified nanoGPT implementation Karpathy [2023] with three transformer layers, each containing a multi-head causal self-attention block (6 heads, embedding size 120), layer normalization, and an MLP with GELU activation. The model has a context window of 36 tokens for direct models and 114 tokens for step-by-step models. The vocabulary size is around 40 tokens. We disable dropout and biases in LayerNorm layers and apply weight tying between the token embedding and the output projection layer. Both absolute and relative (global) positional encodings were tested in separate experiments.

Models are trained with an autoregressive objective, predicting the next token given the previous sequence. For a sequence $x _ { 1 : T }$ of length T, the training loss is the cross-entropy:

$$
L ( w ) = - \sum _ { t = 1 } ^ { T - 1 } \log p _ { w } ( y = x _ { t + 1 } \mid x _ { 1 : t } ) ,
$$

where $p _ { w }$ denotes the model distribution parameterized by weights w.

We train for 100 epochs with a batch size of 512. The optimizer is AdamW with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ and weight decay 0.1. The learning rate follows a cosine annealing schedule with warmup (100 steps), starting from $3 \times 1 0 ^ { - 4 }$ and decaying to $6 \times 1 0 ^ { - 6 }$ . Gradient clipping is applied with a maximum norm of 1. Training is performed on a single GPU using PyTorch 2.0, with Flash Attention kernels (scaled\_dot\_product\_attention) enabled when available.

## A.4.1 Absolute and relative positional embeddings

We experiment with two positional encoding embeddings: absolute embeddings and relative global embeddings.

Absolute embeddings. In the absolute case, we follow the original transformer formulation [Vaswani et al., 2017], where a learnable position embedding $P ( t )$ is added to the token embedding $E ( x _ { t } )$ before being passed into the transformer layers:

$$
z _ { t } = E ( x _ { t } ) + P ( t ) , \quad t = 1 , \dots , T .
$$

This ensures that positional information is directly encoded in the input sequence representation, with the embedding table learned jointly with the model. In our implementation, the transformer instantiates both token and position embedding tables, which are added elementwise at each time step.

Relative global embeddings. In the relative case, we replace the absolute embedding table with a learned relative representation incorporated into the attention mechanism. This approach is motivated by the relative position representations of Shaw et al. [2018] and extended in the Music Transformer [Anna et al., 2018]. Formally, each attention head maintains a trainable relative embedding matrix $\mathbf { \bar { E } } _ { r } \in \mathbb { R } ^ { C \times \dot { d } _ { h } }$ , where C is the maximum context length and $d _ { h }$ the head dimension. Given query vectors $Q \in \vec { \mathbb { R } } ^ { B \times H \times T \times d _ { h } }$ , we compute relative logits as:

$$
S _ { \mathrm { r e l } } = { \mathrm { s k e w } } ( Q E _ { r } ^ { \top } ) ,
$$

where the skew operation aligns relative positions with their corresponding offsets. These logits are then added to the standard dot-product attention $Q K ^ { \top } / \sqrt { d _ { h } }$ yielding attention weights that encode both content and relative distance. This modification removes the absolute embedding table, instead parameterizing $E _ { r }$ within each attention block.

## A.5 Training Details

Data generation. We generate 100K training samples evenly distributed across the unique train permutations and 10k test samples evenly distributed across the unique test permutations. For example, for $k = 6 .$ , there are a total 720 module permutations without identity module-so we sample roughly 138 samples per permutation with random data strings combined in train and test sets. All input data tokens are of fixed-size strings of length six and sampled uniformly without replacement from the $\gamma = \{ a - z \}$ . For binary functions in the diverse benchmark, we sample two input strings of length $6 ,$ and the output can be of max-length 12, from the same vocabulary space. As the uniform benchmark consists of unary bijection functions, both input and output data tokens are of length 6.

Distinct input data strings: As we randomly sample input strings of length six without replacement, there are a total of 11,576,560 unique strings. In our data sets, we found approximately 99.9% unique strings across both the training and test sets of total size $1 1 0 \mathbf { k } \left( 1 0 0 \mathbf { k } + 1 0 \bar { \mathbf { k } } = 1 1 0 \mathbf { k } \right)$

Prompt format: To generate the training sequences for the transformers, we serialize a vocabulary consisting of lowercase alphabet characters along with special tokens <START>, <SEP>, <END>, and a space token. <START> and <END> mark the sequence boundaries, while <SEP> separates function tokens, input strings, intermediate outputs (for step-by-step data), and the final output. Spaces are used for padding to ensure fixed input lengths across examples.

We evaluate module orderings on two sets of functions. In the uniform case, all functions are bijective map operations and require only a single input string. In the diverse case, where functions are based on common string operations, functions may take up to two input strings and therefore each prompt must contain two input sequences. With this in mind, the following is the exact prompt structure used to train and evaluate our models:

For data points generated with the direct setting, the prompt structure is:

$$
< \mathrm { S T A R T } > ~ f _ { 1 } ~ f _ { 2 } ~ . ~ . ~ . ~ f _ { k } ~ < \mathrm { S E P } > ~ x _ { 1 } ~ . ~ . ~ x _ { m } ~ < \mathrm { S E P } > ~ x _ { 1 } ^ { \prime } ~ . ~ . ~ x _ { m } ^ { \prime } ~ < \mathrm { S E P } > ~ y _ { 1 } ~ . ~ . ~ y _ { m } ~ < \mathrm { E N D } >
$$

where $f _ { 1 } , \ldots , f _ { k }$ denote the module composition, $x _ { 1 } \ldots x _ { n }$ the first input string, $\left[ < \mathsf { S E P } > \boldsymbol { x } _ { 1 } ^ { \prime } \ldots \boldsymbol { x } _ { n } ^ { \prime } \right]$ the second input string present only in the diverse setting, and $y _ { 1 } \ldots y _ { m }$ the final output.

In the step-by-step setting, the intermediate outputs of each function are also included between separators, yielding multiple <SEP> segments:

$$
\begin{array} { r l } & { < \mathrm { S T A R T } > \ f _ { 1 } f _ { 2 } . . \ f _ { k } \ < \mathrm { S E P } > \ \hdots \ x _ { m } \ < \mathrm { S E P } > \ x _ { 1 } ^ { \prime } \cdot . . \ x _ { m } ^ { \prime } \ < \mathrm { S E P } > \ y _ { 1 1 } . . \ y _ { 1 m } \ < \mathrm { S E P } > \ y _ { 2 1 } . . . \ y _ { 2 m } } \\ & { . . . \ < \mathrm { S E P } > \ y _ { k 1 } . . . \ y _ { k m } \ < \mathrm { H D P } > } \end{array}
$$

where $x ^ { ( i ) }$ denotes the intermediate output after applying the ith function in the composition.

Positional embeddings and length generalization: Positional embeddings have been a major area of research when considering the length generalization of transformer architectures. Length generalization in transformers is the behavior where models trained on shorter sequences still perform well on longer ones unseen in the training data. In a sense, this requires that the transformer generalize seen modules at one position to perform the same task at another. This framing provided some of the motivation for including relative embeddings in transformers-the idea is that sequences of set behaviors should be learned relative to each other instead of at an absolute position. Indeed, several works [Neishi and Yoshinaga, 2019, Ruoss et al., 2023] have shown that using relative positional embeddings outperforms absolute ones. We refer readers to the survey by Zhao et al. [2024b] for a larger collection of community efforts. It is this same motivation that informs our decisions to consider relative positional embeddings. Though we do not test length generalization (since our input prompts are fixed length), we instead consider scenarios where modules do not appear in specific positions or appear less frequently.

Compute Resources: All experiments were conducted on the Unity SLURM cluster. NanoGPT models were trained on a single GPU with 100GB RAM and 4 CPUs, with a wall-clock time of up to 1-2 hours per job. Gemma-3-1B and Llama-3-8B fine-tuning were performed on a single NVIDIA A40 GPU with 100 GB RAM and 8 CPUs, with a wall-clock time of up to 48 hours per job. Given the large number of train-test splits evaluated, experiments were parallelized across multiple SLURM jobs.

## A.5.1 Training, validation, and test performance during training

![](images/0352242a380340cb05a49e70feba4618ec18b202e65c57478e8f237a519069e9.jpg)

![](images/587db44e20a20ce193a886a60ad48c9eb2f3fbde54b107fe4e071f79842c2b58.jpg)

Figure 7: Uniform benchmark (without identity), N\_heads = 6, N $\mathbf { l a y e r s } = 3 ,$ Direct (Relative): We observe that the models learn to predict the outputs of training compositions for both K=2 and for ${ \mathrm { K } } { = } 6 .$ while the test accuracy on unseen compositions remains 0% throughout. We report two metrics: (1) sharp accuracy, where a prediction scores 1.0 only if the entire output sequence is correct and 0.0 otherwise; and (2) mean token accuracy, average accuracy of predictions in the output sequence, with both metrics averaged across compositions from each split. For reference, 100 epochs correspond to roughly 20k iterations (batch size 512, 100k training samples, 195 iterations per epoch).  
![](images/3bc0b13407500e6cd7769419dc3b8d20a1de36b74c7b47eb295d7a24bf259d99.jpg)

![](images/7b0494b90ff7c5145ba94347c705c0bb450f88c5e862d236204d6eebc06a7250.jpg)  
Figure 8: Diverse benchmark, N\_heads = 6, N\_layers = 3, Direct (Relative): We observe non-trivial compositional generalization, as the models perform well on training and validation compositions, the test performance also increases. We observe better generalization for $K = 6$ than for $K = 2$ due to more training diversity and shared equivalences.

Table 2: Sensitivity of accuracy to the number of strings N used to estimate composition equivalence. Each column reports results for a given percentage of shared composition equivalences between train and test splits. Accuracy increases with shared equivalence at every value of N.
<table><tr><td>Num. strings</td><td>0%</td><td>20%</td><td>50%</td><td>60%</td><td>70%</td><td>100%</td></tr><tr><td>10</td><td>1.8</td><td>4.2</td><td>6.4</td><td>12.7</td><td>12.3</td><td>83.4</td></tr><tr><td>100</td><td>1.1</td><td>3.8</td><td>8.7</td><td>17.4</td><td>41.1</td><td>75.0</td></tr><tr><td>1,000</td><td>0.4</td><td>6.2</td><td>10.0</td><td>22.0</td><td>44.9</td><td>78.2</td></tr><tr><td>10,000</td><td>1.2</td><td>7.9</td><td>11.8</td><td>20.9</td><td>48.0</td><td>78.6</td></tr></table>

## A.5.2 Larger Transformer Architecture (N\_heads = 12, N\_layers=12)

In Figure 9, we observe that the performance of the larger architecture shows similar trends seen in Figures 2(b) and (d) corresponding to the smaller architecture $( \mathrm { N \_ h e a d s } = 6 , \mathrm { N \_ l a y e r s } = 3 )$

![](images/014bc718df6f97ddf1144b834320849f7979882341511b4e8cc962a9f4a44c74.jpg)

![](images/8d8155c09647fb007478144ce90b2595254e7e45a1cc703205dd4e694e49c7bd.jpg)  
Figure 9: Generalization performance of bigger transformer architecture: We observe that the performance of the bigger architecture shows similar performance trends as seen for the smaller architecture (N\_heads = 6, N\_layers = 3) in Figures 2(b) and (d).

## A.6 Isolating the effect of composition equivalence on the performance of direct models

Sensitivity of results to N, the number of strings used to estimate composition equivalence: Within each num\_strings setting, we consistently observe the same core trend reported in the paper: accuracy increases as shared equivalence (leakage) increases, regardless of N. Separately, N trades estimation accuracy of the equivalence-class structure for compute cost. For splits with shared equivalence classes, larger N tends to improve accuracy further, plausibly because test compositions are mapped more accurately to their correct equivalence class. Results are shown in Table 2.

Uniform Benchmark: Figure 10(a) shows that direct model performance, for both absolute and relative embeddings, is strongly predicted by the degree of shared equivalence classes. We systematically increase the proportion of shared identity-based equivalent classes in the train and test sets. In the uniform benchmark, there is a one-to-one mapping between sequences and identity-based equivalence classes, which simplifies controlling the number of shared equivalences. However, the mere existence of shared classes does not always guarantee correct generalization; for example, the absolute positional embedding model saturates in performance after 51% shared classes in Figure 10(a). Upon further analysis, we find that most failures correspond to test sequences that have the identity module in the first position (Figure 26).

Details: We focus on sequences with $k = 6$ non-identity functions and one identity function, generating a total $7 ! = 5$ , 040 permutations grouped into a total of 620 equivalence classes. Each class consists of seven equivalent tasks corresponding to the possible positions of the identity module, while keeping the relative ordering of non-identity functions fixed. We maintain constant train-test split sizes in terms of equivalence classes (576/144) and vary the percentage of shared equivalence classes from 0–100%. To maintain a fixed total number of samples, we exchange half of the tasks within shared equivalence classes between the training and test sets. A setting of 0% means disjoint classes, while 100% means all test tasks have equivalents in training. Accuracy is computed based on shared equivalence classes.

![](images/2257ad4f8ce479a9fc30c4c3faab110ac187250d8694f31aac05239ff9863f87.jpg)  
Figure 10: Equivalence class necessity and training requirements: (a) Generalization performance to unseen sequences is strongly correlated with the degree of shared composition equivalence classes between train and test. (b) Models need to see at least 4–5 composition equivalences (per class) in training to generalize to the corresponding test equivalences.

## A.6.1 Equivalence class examples

In Figure 11, we show two equivalence classes of size 38 (ID 9) and 24 (ID 2) for $K = 6$ The first equivalence class usually ends with (union','map') or $( \mathbf { \omega } ^ { \mathsf { c } } \mathtt { j o i n } ^ { \mathsf { \prime } }$ , map') sequences, and the second class ends with $( ^ { \cdot } \mathrm { u n i o n } ^ { \cdot } , ~ ^ { , } \mathrm { j o i n } ^ { \cdot } )$ , causing end-to-end behavior difference between the two classes. We also observe that the equivalence scores vary significantly within an equivalence class, ranging from exact equivalence (1.0) to weak equivalence (0.01). Next, we observe that the model performs significantly better on test sequences belonging to equivalence classes with shared members in the train set and remains poor (0-0.2) on sequences belonging to non-shared equivalence classes. Figure 12 shows the size distribution of learned equivalence classes.

## A.6.2 Systematically including test sequences without swapping with train sequences (K=6)

In Figure 13(a), we show that the generalization performance increases with shared equivalences in the case we systematically leak half of the members of a test equivalence class in training without swapping. This reduces the test split size in proportion to the class size as the percentage of shared equivalences increases. Figure 13(b) shows that performance increases significantly for shared equivalence classes while performance of non-shared equivalences remains poor, varying between 0%-20%.

## A.7 Composition equivalences encourage learning of shortcuts in step-by-step models

Training on the intermediate outputs breaks composition equivalences based on end-to-end labeling functions, improving the identifiability of individual labeling functions. Still, equivalences can also arise at the intermediate output level, especially when stepwise outputs are not distinct. We find that such equivalences encourage step-by-step models to rely on shortcuts rather than step-by-step reasoning, creating multiple paths to the correct answer. We observe that these models provide correct final outputs but incorrect intermediate ones in such cases. In the diverse benchmark consisting of a large number of equivalences, this results in higher final output accuracy but lower step-by-step accuracy, as models often produce incorrect intermediate outputs while still providing the correct final result (Figures $^ { 1 4 ( \mathrm { a } ) , ( \mathrm { b } ) ) }$ . In contrast, for the uniform benchmark, where no equivalences exist, step-by-step and final output accuracy exactly match as shown in Figures 15(a), 15(b). Thus, composition equivalence can negatively impact compositional generalization in step-by-step models by promoting shortcuts over accurate step-by-step reasoning.

![](images/14f640e30fb0649f20d2b72cbcf5f1c43c0aa0a3e57c03970c6c9f4b4e516303.jpg)  
Figure 11: Visualization of learned equivalence classes for the diverse benchmark (K=6): We show two equivalence classes (out of a total of 14 classes). First, we can note that the degree of equivalence varies significantly within an equivalence class. The test sequences are marked in red, and the training sequences are marked in green, indicating whether an equivalence class is shared. We observe that accuracy is significantly higher for compositions in the shared equivalence class and near zero for those in the non-shared equivalence class.

![](images/59ceb9f40b080e2260b8df1e1e41eb97d69fc63a5cf82b81a15d086740eeb2a2.jpg)  
Figure 12: Size distribution of learned equivalence classes( $\mathbf { \boldsymbol { K } } = \mathbf { \boldsymbol { 6 } } )$

![](images/83b57b50d5f74e773dae8f89002bde8fcf7f164cae8269e7c340eef732e5bfc0.jpg)

![](images/018e67d478ee7d04c0f92b96d0340f31f240c9e4857a04404f0b105f74a14f32.jpg)  
Figure 13: Leaking test equivalence classes without swapping $( K = 6 ) \colon$ (a) Generalization performance to unseen sequences is strongly correlated with the number of shared composition equivalence classes between train and test. (b) The mean accuracy of shared equivalence classes is significantly higher than that of non-shared equivalence classes.

![](images/b005f0ba9f7e487759a043cb1bfdea2662351132d0980e6559e4e677251d077a.jpg)  
(a) Within-k evaluation (k= 6)

![](images/0d8d2b130265070c783bf82671db638da54e881eac7cd5c551930fe379c9170e.jpg)  
(b) Cross-k evaluation (Train-k = 6)  
Figure 14: Composition equivalence encourages shortcuts in step-by-step models: We observe that composition equivalences in the diverse benchmark cause discrepancies between final output accuracy and full output (intermediate and final output) accuracy for both within-k evaluation and cross-k evaluation. Final accuracy > full accuracy implies the model is learning shortcuts to reach the final answer. If we eliminate composition equivalences, the performance difference is zero between full and final outputs (see Figures 15(a), 15(b))

![](images/efb457110ea0a836dc075f667febf3258e808e625fae8aa542cdcbc77847c98d.jpg)  
(a) Within-k evaluation (k= 6)

![](images/cb695b190061b065ce03cf9fb66396bce63170accc395679cbd0de877d262a03.jpg)  
(b) Cross-k evaluation (Train-k = 6)  
Figure 15: Eliminating composition equivalences removes shortcut learning in step-by-step models : We observe that the lack of compositional equivalence in the uniform benchmark removes shortcut-learning behavior in step-by-step models. This is shown by the exact same step-by-step accuracy, and the final accuracy is the same for both within-k evaluation $( k = 6 )$ and cross-k evaluation (train-k = 6).

## A.8 The role of composition equivalence in compositional generalization of pre-trained models

## A.8.1 Evaluation Details of Llama3-8B on LiveCodeBench benchmark:

Data set Curation We perform LoRA fine-tuning Hu et al. [2022] of pretrained Llama3-8B to evaluate the performance of an LLM on a real-world task that may include compositional equivalences. Specifically, we train and evaluate on a dataset derived from LiveCodeBench's code execution evaluation task Jain et al. [2025]. LiveCodeBench extracts multiple solutions to problems hosted on various coding websites like LeetCode or CodeForces. Each individual solution can be considered compositionally equivalent to the others, since they have the same labeling function despite being implemented differently and sometimes using completely different libraries or methods.

To evaluate the effect of the leakage of compositional equivalences, we select 9 functions which we use to evaluate the effect of equivalence leakage during fine-tuning (see Fig 6(b) for the exact function names). The functions were selected arbitrarily, with the only requirement being that they have the same number of code examples (6) in the original LiveCodeBench data set. From this, we generate a set of datasets with 100k training points and 10k testing points, varying the number of selected functions leaked into the training set, which we call the held-out fraction h. Leaked functions include four of the six code examples added to the pool of functions sampled for the training set. The remaining pool for both train and test is randomly sampled from a filtered set of functions from LiveCodeBench, which removes easy-to-predict functions (i.e., functions that output booleans) and functions that consistently throw errors when the inputs are randomized. For our experiments, we generate datasets for $h = \mathrm { { \bar { 0 } } } , 0 . 5 , 1$ , where we leak 0, 5, and 9 of the selected functions, respectively.

Fine-tuning Details We fine-tune the model for 1 epoch using Low-Rank Adaptation with rank r = 16 and scaling factor α = 32. A dropout rate of 0.05 is applied to the LoRA updates, and no bias parameters are trained. LoRA is applied to the attention projection matrices $\bar { W _ { q } } , W _ { k } , W _ { v } ,$ and $W _ { o } .$ while all pretrained model parameters are frozen.

## A.8.2 Gemma3-1B

We fine-tune and evaluate the compositional generalization performance of pre-trained Gemma3-1B models on the following train/test splits. Fine-tuning is done for five epochs with a batch size of 1. Results can be seen in Figure 4. Figure 16 shows the accuracy distribution over 100% shared equivalences, and Figure 17 shows the disjoint split, where no classes are shared, and the accuracy distribution of classes is pretty low.

## A.9 Module coverage shift in cross-k settings

In Figure 18, we show the KL divergence between train and test compositions as measured using shifts between position-wise and pair-wise module distributions. We can observe that the divergence

![](images/508c170915f5c4d956f0b2d6cab13ef6ae09ba7fe4091ad191755c895e842916.jpg)  
Figure 16: Gemma-1B visualization of per-class distribution for 100% sharing: We observe that most of the equivalence classes are shared and the model has overall high accuracy over test sequences in the shared equivalence classes.

increases as we move away from train k. The divergence values are correlated with the model performance.

## A.10 Module coverage

## A.10.1 Generation of various degrees of position-wise and pair-wise distribution shift

Definition A.1 (Position-wise module distribution). For module position $j \in \{ 1 , 2 , 3 , \ldots , k \}$ and module token $t _ { i } \in \mathcal { T }$ , the position-wise coverage distribution is: $\begin{array} { r } { \hat { P } ( T _ { j } = t _ { i } ) = \frac { 1 } { N } \sum _ { s = 1 } ^ { N } \mathbb { I } ( T _ { j } ^ { ( s ) } = } \end{array}$ ti).

Definition A.2 (Pairwise module adjacency distribution). The empirical pairwise adjacency distribution is: $\begin{array} { r } { \hat { P } ( T = t _ { i } , T ^ { \prime } = t _ { i ^ { \prime } } ) = \frac { 1 } { N ( k - 1 ) } \sum _ { s = 1 } ^ { N } \sum _ { j = 1 } ^ { k - 1 } \mathbb { I } ( T _ { j } ^ { ( s ) } = t _ { i } a n d T _ { j + 1 } ^ { ( s ) } = t _ { i ^ { \prime } } ) } \end{array}$

Given n modules $\mathcal { T } = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { n } \}$ and composition length k, we generate train and test distributions with controlled divergence:

Position-wise shift generation:

![](images/66c1c01d9332abfeeeb61549b24d41e26a4014ac23008d917bf9acccaf3b0efe.jpg)

Figure 17: Gemma-1B visualization of test accuracy for the split with no shared equivalences: We observe that none of the equivalence classes are shared, and the model has poor accuracy over test sequences in those classes.  
![](images/6ce874153aa0b949cf444ee6c6439836df67cf9651d76d209b4496859c5a02ff.jpg)

![](images/f5f1820e1ee74e8ac145ab366bd78db64fa0f849a294a89b62baf7dea7710ddf.jpg)

![](images/dead70f86a7a2e8e1f4cac2f124b22011faf9f08998b56fd3d67348a3ad7fc64.jpg)

![](images/af0337ad62b8b5081743c27da0eacefbde28590c4523305b8c640e02a27ffa12.jpg)

![](images/a8d1177edad80e7920140bfc9821c4f2c8e78d852e1c1a14c7f5d0e4c9e0fd39.jpg)  
Position-wise KL DivergencePairwise Adjacency KL Divergence

Figure 18: Position-wise and pair-wise KL divergence in Cross-k evaluation

1. Begin with uniform distributions:

$$
\begin{array} { c } { P _ { { \mathrm { t r a i n } } } ^ { ( 0 ) } ( T _ { j } = t _ { i } ) = P _ { { \mathrm { t e s t } } } ^ { ( 0 ) } ( T _ { j } = t _ { i } ) = \displaystyle \frac { 1 } { n } } \\ { \forall j \in \{ 1 , \ldots , k \} , \forall t _ { i } \in { \cal T } } \end{array}\tag{1}
$$

2. Introduce bias strength $\alpha \in [ 0 , 1 ]$ where $\alpha = 0$ yields no shift and $\alpha = 1$ yields maximum shift.

3. For each position $j ,$ define offset offset $\mathbf { \partial } ( j ) = j$ mod n and create complementary groups:

$$
\mathcal { G } _ { j } ^ { \mathrm { t r a i n } } = \{ t _ { ( i + \mathrm { o f f s e t } ( j ) ) \bmod n } : i = 0 , \ldots , \lceil n / 2 \rceil - 1 \}\tag{2}
$$

$$
\mathcal { G } _ { j } ^ { \mathrm { t e s t } } = \mathcal { T } \backslash \mathcal { G } _ { j } ^ { \mathrm { t r a i n } }\tag{3}
$$

4. Define un-normalized weights:

$$
\begin{array} { r } { w _ { \mathrm { t r a i n } } ( T _ { j } = t _ { i } ; \alpha ) = \left\{ \begin{array} { l l } { 1 + 1 0 \alpha } & { \mathrm { i f ~ } t _ { i } \in \mathcal { G } _ { j } ^ { \mathrm { t r a i n } } } \\ { 1 - 0 . 9 5 \alpha } & { \mathrm { i f ~ } t _ { i } \in \mathcal { G } _ { j } ^ { \mathrm { t e s t } } } \end{array} \right. } \end{array}\tag{4}
$$

with $w _ { \mathrm { t e s t } }$ defined symmetrically, then normalize to obtain $P _ { \mathrm { t r a i n } } ( T _ { j } = t _ { i } ; \alpha )$ and $P _ { \mathrm { t e s t } } ( T _ { j } =$ $t _ { i } ; \boldsymbol { \alpha } )$

5. Compute position-wise divergence: $\begin{array} { r } { D _ { \mathrm { p o s } } ( \alpha ) = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } D _ { \mathrm { K L } } \bigl ( P _ { \mathrm { t e s t } } ( T _ { j } ; \alpha ) \| P _ { \mathrm { t r a i n } } ( T _ { j } ; \alpha ) \bigr ) } \end{array}$

## A.10.2 Effect of pairwise distribution shifts

In Figure 19, we demonstrate the effect of pairwise distribution shift. Biased sampling of pairwise distributions is done in a similar way to the position-wise distribution.

Pairwise shift generation: Similarly, for adjacent module pairs $( T _ { j } , T _ { j + 1 } )$ , we define complementary pairwise groups $\mathcal { G } _ { j } ^ { \mathrm { t r a i n , p a i r } }$ and $\mathcal { G } _ { j } ^ { \mathrm { t e s t , p a i r } }$ containing module pairs, apply analogous biased weighting with strength α, and compute pairwise divergence: $D _ { \mathrm { p a i r } } ( \alpha ) = D _ { \mathrm { K L } } \big ( P _ { \mathrm { t e s t } } ( T , T ^ { \prime } ; \alpha ) \| P _ { \mathrm { t r a i n } } ( T , T ^ { \prime } ; \alpha ) \big )$ For both cases, we sample $N _ { \mathrm { t r a i n } }$ and $N _ { \mathrm { t e s t } }$ compositions ensuring valid permutations and disjoint supports: $\operatorname { s u p p } ( \mathcal { P } _ { \operatorname { t r a i n } } ^ { \mathcal { T } } ) \stackrel { \cdot } { \cap }$ supp $( \mathcal { P } _ { \mathrm { t e s t } } ^ { \mathcal { T } } ) = \emptyset$ Varying $\alpha \in \{ 0 , 0 . 1 , \bar { . } . . , 1 . \bar { 0 } \}$ systematically controls coverage shift magnitude.

![](images/ef60b5878776839176123a76284d47d33f03971a58dc060dcc439f64690aadf6.jpg)  
Figure 19: Effect of pairwise-wise distribution shift: (a) As the pairwise-wise KL divergence increases, the compositional generalization performance of both direct and step-by-step models decreases in the diverse benchmark. (b) For the uniform benchmark, the effect of divergence is primarily observed at the extreme right end for support violations.

Interplay between coverage shift and compositional diversity: If we allow sampling duplicate compositions, increasing bias strength causes certain compositions to be sampled more frequently. For target size $N = 3 6 0$ , increasing bias strength from 0 to 1 reduces unique compositions from 360 to 50 while increasing their sampling frequency. Consequently, Figure 20 shows the combined effects of both module coverage shift and reduced compositional diversity. This interplay reflects realistic settings where limited diversity naturally induces coverage distribution shifts between train and test sets. To isolate these effects, we ran control experiments generating the same number of unique compositions observed at each $D _ { p o s }$ value, but with zero KL divergence via random sampling. Subtracting this zero-shift baseline isolates the coverage shift effect while controlling for diversity (Figure 21). We observe that the effect of coverage shift remains significant, though reduced compared to Figure 20, as expected.

![](images/660224219d81c3f42abafd0629e50afbb1de9a835d6bd2f1e97b20f4c705d6cd.jpg)  
Figure 20: Combined effect of position-wise shift and reduced diversity: Shows $( 1 - a c c u r a c y _ { K L } )$ to demonstrate accuracy degradation as in Figure 5, revealing combined effects of coverage shift and reduced compositional diversity due to biased sampling. Here, $a c c u r a c y _ { K L }$ is the performance with biased sampling.

![](images/66f23f1df544a308b16a3cfa958d3e717e7d13bc7c81e2e306b4ed221e031ece.jpg)  
Figure 21: Isolating position-wise shift by controlling diversity: Shows $a c c u r a c y _ { z e r o \_ s h i f t } -$ $\begin{array} { r } { a c c u r a c y _ { K L } , } \end{array}$ where $a c c u r a c y _ { z e r o \_ s h i f t }$ and $a c c u r a c y _ { K L }$ are accuracies under random and biased sampling with n unique compositions at each $D _ { p o s } ,$ respectively. For the diverse benchmark, positionwise shift alone has a reduced but still significant effect compared to Figure 20, indicating that both coverage shift and reduced diversity contribute to degradation. For the uniform benchmark, the negligible difference confirms that degradation is driven primarily by coverage shift alone.

## A.11 Sample efficiency ablations

![](images/f31b32d0e6be0ba99acb552e1770418dd62776930710c9492e20bc1779a2cda3.jpg)  
Figure 22: Effect of number of unique training sequential compositions (randomly sampled, $\mathbf { K } \mathbf { = } \mathbf { 6 } ) \mathbf { : }$ We observe that (a) step-by-step models with relative global embedding need fewer training compositions to generalize faster than absolute embedding models, and (b) For the diverse benchmark, as compared to direct models, step-by-step models need fewer unique training compositions due to extra information available using intermediate outputs.

![](images/50896606452f8f8f0dfd780bc54617ba6854a54c310992808b7d70217dbe91db.jpg)

![](images/6a926b251261e915cc99bdb93aa3d4cde1e123c490043c288d95dddeaadf55eb.jpg)

Figure 23: Effect of number of samples per composition (uniform benchmark, direct model):. Number of training compositions $N = 2 4$ for $K = { \bar { 2 } } ,$ and $N = 5 7 6$ for $K = 6 .$ We vary the number of input-output samples per training composition in this experiment. We observe that with a small number of training samples, the train/val gap is large, and the model struggles to generalize to training compositions with novel inputs. The train-val gap reduces as the number of samples increases. Test accuracies remain 0%, as expected.  
![](images/e2ad8e755f826793a2767179a4ededbab88f25d22a9c3dab436106157d29c072.jpg)

![](images/82ab732e5c04db78df45b7db68a4c155d1880ddf1a889f2f11d979b17eceb5f9.jpg)  
Figure 24: Effect of number of samples per composition (uniform benchmark, step-by-step model): We observe that step-by-step models are more sample efficient than direct models due to access to intermediate outputs providing extra information.

![](images/232631328289a9a7ae5d3909071751373e9bcbdb340c369d23f9e075e1f7c495.jpg)

![](images/8a9ad41b3b54aa7c8080215b251925acbdc3f4ce1deef1a1544dc3e887f511d0.jpg)  
Train sharp acc ---- Train mean acc Val sharp acc ---- Val mean acc Test sharp acc ---- Test mean acc

Figure 25: Effect of number of samples per composition (diverse benchmark, direct model): For the diverse benchmark, direct models are less sample efficient than the uniform benchmark to generalize on the validation set, especially for K= 2. We observe significant compositional generalization behavior, as seen in Figure 2(b) for large samples, especially for K=6.  
![](images/b3301a325bde5a8a42e07097ae966b7e0f3f7d042a42b22be1efb8ea1271cbef.jpg)

![](images/ee49d1bdffea9a4ca65410056de24122930fc53ed2bee76b4f11d752b7d10c39.jpg)  
Figure 26: Effect of number of samples per composition (diverse benchmark, step-by-step model): Step-by-step models are more sample efficient than direct models, and a similar difference between uniform and diverse benchmarks for K=2 is observed as for direct models.