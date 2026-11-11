# One Sequence Is Enough: Fingerprinting Language Models from Fixed-Length Integer Sequences

Fantong Meng<sup>∗</sup>, Chengjie Sun<sup>∗</sup>, Shuangjia Yang,

Zhiyuan Zhao, Yansheng Pei, Xinyang Li

Faculty of Computing, Harbin Institute of Technology

Harbin, China

25s103344@stu.hit.edu.cn, sunchengjie@hit.edu.cn

## Abstract

Users of model APIs and third-party platforms often rely on the declared model name to identify the model responding to their requests. However, the actual model may difer from the declared one, motivating identity verification based on observable outputs. Previous studies have shown that preferences in simple numerical-choice tasks can serve as model fingerprints. We investigate whether a single fixed-length integer sequence can combine multiple numerical observations to identify its source model. For this task, we introduce NumPrint, a benchmark of 5,036 answers from 51 models, collected using Chinese, Japanese, and English prompts that request 294 integers between 1 and 355 in one response. We propose the Shallow MLP and Causal Transformer (SMCT) to capture aggregate numerical preferences and behavioral patterns in generated sequences. In 51-class closed-set evaluation, SMCT achieves 92.72% accuracy from a single sequence, outperforming all methods compared in this study. Using two and four answers increases accuracy to 96.06% and 97.12%, respectively. These results show that a single integer sequence contains useful model fingerprints and that additional answers further improve identification.

## 1 Introduction

Which language model generated a given response? Users of oficial APIs and third-party platforms typically rely on the declared model name, but the model generating a response may difer from the one advertised. This uncertainty complicates service verification and experimental reproducibility. We therefore need methods that verify model identity independently of the declared model name. Model fingerprinting provides such evidence by analyzing generated text, eliciting responses to carefully designed queries, or comparing the output distributions of a target and a reference model (Pasquini et al., 2025; Gao et al., 2025).

Numerical-choice tasks ofer a simple way to observe generation preferences: the bounded output space allows direct comparisons between distributions. Previous studies have found that language models can deviate from the distributions they are instructed to sample (Gu et al., 2025; Zhao et al., 2026). One Token Is Enough uses frequency distributions of repeated short answers to verify model identity and identify the model’s family (Bruckner, 2026); FLIPS converts two-symbol sequences to binary and uses randomness-test statistics to identify model instances (Richardeau et al., 2026). These findings show that numerical and symbolic generation preferences can serve as model fingerprints.

We focus on eliciting multiple decimal integers in a single response. The resulting sequences provide observations of value choices together with relationships among the generated integers. The open-source lm-detector project already identifies models from integer sequences and directly inspired this work (Ikaleio, 2026). However, its reference bank mixes prompts and requested lengths, leaving too few comparable samples for training and evaluation at a fixed observation length. Its representation primarily consists of frequency and coarse positional summaries, with limited use of sequence variation and dependencies between the current integer and preceding integers.

An integer sequence contains both value frequencies and relationships associated with generation order. Two sequences with identical value counts can difer in adjacent changes, repetition gaps, and local dependencies. We use causal attention to represent each integer in relation to its preceding sequence. These contextualized representations capture output patterns that may reflect model-specific numerical generation behavior. We combine them with statistical features in one archi-

![](images/fe429f71ab85eaaa9de7c5a4e796964199b1394ff315784bcb5f0172a18fd10e.jpg)  
Figure 1: Single-answer model identification. (a) Chinese, Japanese, and English prompts elicit labeled integer sequences used to construct NumPrint. (b) One request yields a 294-integer sequence from the model to be identified. (c) SMCT identifies the source model using statistical and behavioral features. (d) Predicted model and single-answer test performance. Prompts and sequences are excerpts, and the model ranking is taken from a saved prediction instance; feature motifs are schematic.

tecture.

We construct NumPrint, a benchmark of 5,036 answers from 51 models, using Chinese, Japanese, and English prompts that request 294 integers between 1 and 355 in one response. To identify the source model, we propose the Shallow MLP and Causal Transformer (SMCT), with an adaptive classification module operating on both encoders’ outputs. From a single answer, SMCT achieves 92.72% accuracy and 91.10% macro-F1, the highest among the compared methods.

Our main contributions are as follows. First, we introduce NumPrint, to our knowledge the first benchmark for model fingerprinting from fixedlength decimal integer sequences collected with multilingual prompts. It provides fixed data splits, evaluation metrics, and baseline results. Second, we propose SMCT, an architecture that learns aggregate numerical preferences and sequence behavior for source-model identification. Third, we use a consistent evaluation protocol to establish the complementarity of the two representations and examine how sequence length and the number of available answers afect identification.

## 2 Related Work

## 2.1 Black-Box Model Identification

Black-box model identification determines which model generated an output by analyzing the text itself or responses to designed queries. MGTBench and M4GT-Bench evaluate machine-generated text detection and source-model attribution (He et al., 2024; Wang et al., 2024); FDLLM fine-tunes a language model with LoRA to learn text features that distinguish source models (Fu et al., 2025).

Among methods that actively query a model, LLMmap trains a classifier on responses to knowledge questions, safety-related queries, and mixedlanguage inputs (Pasquini et al., 2025); model equality testing compares the response distributions of a target API and a reference model under the same text-continuation, code-generation, or instruction-following requests (Gao et al., 2025).

Other approaches require additional access: TRAP constructs adversarial prompts using reference-model parameters (Gubri et al., 2024), while ProFLingo uses targeted queries to test whether one model is derived from another (Jin et al., 2024). Instructional Fingerprinting binds secret prompts to predefined responses through fine-tuning, enabling ownership verification by eliciting the specified text (Xu et al., 2024). Text watermarking embeds statistical signals during sampling (Kirchenbauer et al., 2023).

With access to hidden layers, REEF compares activation vectors produced by the tested and reference models on the same inputs to determine whether the former derives from the latter through fine-tuning, pruning, or quantization (Zhang et al., 2025).

## 2.2 Numerical Generation Biases and Sequence Fingerprints

Gu et al. (2025) find that language models can understand specified probabilities but struggle to sample from the corresponding distributions. Zhao et al. (2026) compare generating a batch of numbers in one response with generating individual numbers through independent requests. In their evaluation, samples from independent requests have lower pass rates on distributional tests. Guo et al. (2025) observe position preferences in rock– paper–scissors choices even after semantic option names are removed. These studies reveal systematic biases in numerical sampling and choice.

These biases can be used to identify models through numerical or symbolic generation tasks. One Token Is Enough repeatedly requests short answers, such as numbers, letters, or words, and compares their frequency distributions with Jensen– Shannon (JS) divergence to verify model identity and identify the model family (Bruckner, 2026).

FLIPS elicits sequences composed of two specified symbols, converts them to binary, and trains an XGBoost classifier on NIST randomness-test statistics to identify models and deployment configurations (Richardeau et al., 2026). In its closed-set task with 237 model instances, single-query accuracy is below 70%, leaving room to improve identification from a single answer (Richardeau et al., 2026). Combining these ideas, we elicit a decimal integer sequence in one response and use its statistical and behavioral information to improve singleanswer identification accuracy.

The open-source lm-detector project uses numerical-choice prompts to obtain integer sequences, extracts integer-frequency, positionblock, and last-digit features, and combines LDA, nearest-neighbor, and deconfounded-center scores to rank candidate models (Ikaleio, 2026). The representation mainly consists of frequency summaries. Positional blocks preserve coarse ordering but do not explicitly model adjacent transitions, lag relationships, or dependencies on preceding values.

The reference bank mixes prompts and requested lengths. Restricting the reference data to answers of a common length leaves too few samples and insuficient model coverage for comprehensive training and evaluation.

## 2.3 Sequence Representations and Identification Methods

Sequence models learn representations directly from ordered inputs. LSTMs use recurrent states, while Transformers use self-attention to model relationships between sequence elements (Hochreiter and Schmidhuber, 1997; Vaswani et al., 2017). Multiscale and random convolutional methods extract local patterns over diferent spans (Ismail Fawaz et al., 2020; Tan et al., 2022; Dempster et al., 2023, 2020, 2021). For set-valued inputs, Deep Sets uses symmetric aggregation, whereas Set Transformer uses attention-based encoding and pooling (Zaheer et al., 2017; Lee et al., 2019).

We adapt these general-purpose sequence and set models as baselines for integer-sequence fingerprinting and compare them with statistical classifiers. SMCT uses a statistical-feature encoder and a causal attention encoder to capture complementary information for source-model identification from fixed-length integer sequences. Section 5 and Appendix D give the settings of the compared methods.

## 3 NumPrint Benchmark

We construct NumPrint to systematically evaluate language model identification from a single fixedlength integer sequence. Using prompts in Chinese, Japanese, and English, we ask each model to generate 294 integers between 1 and 355 in one response. After curation and filtering, the benchmark contains 5,036 answers from 51 models. Each example consists of an integer sequence labeled with its source model. Figure 1a outlines the construction process.

## 3.1 Task Design and Data Collection

We adapt one numerical-choice prompt from lm-detector (Ikaleio, 2026), retaining its task: generate 294 integers between 1 and 355 on instinct in a single response. The prompts prohibit external number-generation tools, allow coincidental repetition, and instruct the model to avoid counting, arithmetic progressions, and repeated blocks.

We express the same task in Chinese, Japanese, and English to reduce dependence on a single prompt language. The data cover GPT, Claude, Gemini, DeepSeek, Qwen, GLM, and other model families, with multiple versions from several families. Appendix A provides the prompts and model inventory.

<table><tr><td>Prompt language</td><td>Train</td><td>Val</td><td>Test</td><td>Total</td></tr><tr><td>Chinese</td><td>1148</td><td>229</td><td>245</td><td>1622</td></tr><tr><td>Japanese</td><td>1188</td><td>260</td><td>263</td><td>1711</td></tr><tr><td>English</td><td>1189</td><td>266</td><td>248</td><td>1703</td></tr><tr><td>Total</td><td>3525</td><td>755</td><td>756</td><td>5036</td></tr></table>

Table 1: Prompt-language distribution and data splits of NumPrint.

## 3.2 Data Curation

We normalize response formatting, filter invalid answers, and consolidate model labels. For answers containing at least 294 valid integers, we retain the first 294 values; Appendix A specifies the rules.

## 3.3 Dataset Statistics and Splits

Table 1 reports the number of answers by prompt language and data split. Class sizes range from 42 to 670 answers; the complete model inventory appears in Appendix A.

We randomly split answers within each model class into training, validation, and test sets at an approximate 70/15/15 ratio (3,525, 755, and 756 answers). Each split contains all 51 classes, and all methods use the same fixed partition.

## 4 SMCT Architecture

SMCT extracts complementary statistical and behavioral representations from the same integer sequence. Let x be an input sequence of length L, and let C denote the number of candidate model classes; $C = 5 1$ in our experiments. As shown in Figure 2, the statistical-feature encoder maps predefined statistics to class scores, while the causal attention encoder represents relationships between the current integer and those preceding it. The adaptive classification module weights the encoder outputs according to confidence and applies a separate linear correction for each class to obtain scores for the C candidate models.

## 4.1 Statistical-Feature Encoder

The statistical-feature encoder represents aggregate value preferences and statistical variation within a sequence. We construct a representation $\phi ( x ) ~ \in ~ \mathbb { R } ^ { 9 4 4 }$ : distributions over integers, digits, and remainders describe value preferences, while positional blocks, increments, transitions, lags, and spectral summaries describe variation within a sequence. Here $\phi ( x )$ denotes the vector after normalization using parameters estimated from the training data and scaling by $\sqrt { 9 4 4 }$ , as specified in Appendix B.

A single-hidden-layer feedforward network maps $\phi ( x )$ to the statistical representation $h ^ { s }$ and class scores $z ^ { s } ;$

$$
h ^ { s } = \mathbf { G E L U } ( W _ { 1 } \phi ( x ) + b _ { 1 } ) ,\tag{1}
$$

$$
z ^ { s } = { \cal W } _ { 2 } D _ { 0 . 4 } ( h ^ { s } ) + b _ { 2 } ,\tag{2}
$$

where $h ^ { s } ~ \in ~ \mathbb { R } ^ { 1 0 2 4 }$ and $z ^ { s } ~ \in ~ \mathbb { R } ^ { C } .$ and $W _ { 1 } ~ \in$ R<sup>1024×944</sup>, $b _ { 1 } ~ \in ~ \mathbb { R } ^ { 1 0 2 4 }$ $W _ { 2 } ~ \in ~ \mathbb { R } ^ { C \times 1 0 2 4 }$ , and $b _ { 2 } \in \mathbb { R } ^ { C }$ are learned weights and biases. $D _ { p }$ denotes dropout with probability $p$ during training; Appendix C provides the dropout and optimization settings.

## 4.2 Causal Attention Encoder

The causal attention encoder models relationships between each integer and the preceding sequence. A learned embedding table maps each integer to a 64-dimensional vector, forming $ { \mathcal { E } } ^ { \mathrm { ~ ~ } } \in \mathrm { ~ }  { \mathbb { R } } ^ { L \times 6 4 }$ Input dropout and layer normalization give $\widetilde { E } =$ $\mathrm { L N } ( D _ { 0 . 1 } ( E ) )$ . We project $\widetilde { E }$ einto queries, keys, eand values, then apply single-layer, single-head causal self-attention (Vaswani et al., 2017) to obtain a representation at each position:

$$
\begin{array} { r l } & { Q = \widetilde { E } W _ { Q } , \quad K = \widetilde { E } W _ { K } , \quad V = \widetilde { E } W _ { V } , } \\ & { A = \mathrm { s o f t m a x } \Big ( Q K ^ { \top } / \sqrt { 6 4 } + M \Big ) , } \\ & { H = \mathbf { L } \mathbf { N } ( D _ { 0 . 1 } ( A V W _ { O } ) ) , } \end{array}\tag{3}
$$

where $Q , K ,$ and V are query, key, and value matrices, $A \in \mathbb { R } ^ { L \times L }$ contains attention weights, and $H \in \mathbb { R } ^ { L \times 6 4 }$ contains the output representations. Softmax is applied row-wise over key positions. The mask sets $M _ { t j } = 0$ for $j \le t$ and −∞ otherwise, so $H _ { t }$ is computed from representations of the current and preceding integers. The projections $W _ { Q } , W _ { K } , W _ { V } , \bar { W } _ { O } \in \bar { \mathbb { R } } ^ { 6 4 \times 6 4 }$ are learned, and LN denotes layer normalization. Afine projection biases are omitted from the notation.

The encoder combines the position representations using a learned weighted readout. Let $H _ { t } \in$ $\mathbb { R } ^ { 6 4 }$ be the transpose of row t of H, and let w $\mathbf { \Lambda } \in \mathbb { R } ^ { 6 4 }$ be a learned column vector. Softmax converts the scores $w ^ { \top } H _ { t }$ to weights $\alpha _ { t }$ , used to compute the sequence representation $h ^ { b }$ :

$$
\begin{array} { r l } & { \alpha _ { t } = \frac { \exp ( w ^ { \top } H _ { t } ) } { \sum _ { u = 1 } ^ { L } \exp ( w ^ { \top } H _ { u } ) } , } \\ & { h ^ { b } = \displaystyle \sum _ { t = 1 } ^ { L } \alpha _ { t } H _ { t } . } \end{array}\tag{4}
$$

![](images/6010e99e1b8a8d2bfd4521dd131746f554b743157aad1b751cdaca99906c6285.jpg)  
Figure 2: SMCT architecture. (a) A feedforward encoder maps 944 statistical features to class scores. (b) Integer embeddings pass through causal attention and a learned weighted readout to form a sequence representation. The two sets of class scores pass through temperature calibration, confidence weighting, and per-class linear correction to produce the model prediction. (c) Internal computations of causal attention and weighted readout. Feature motifs, tensor cells, and score bars are schematic.

The resulting vector $ { \boldsymbol { h } } ^ { b } ~ \in ~ \mathbb { R } ^ { 6 4 }$ passes through dropout and a linear classification head to produce $z ^ { b } \in \mathbb { R } ^ { C }$ . The readout and attention parameters are trained jointly for source-model classification, so each position contributes according to the input sequence.

## 4.3 Adaptive Classification Module

The encoders are trained independently, so their raw class scores can have diferent scales. The adaptive classification module first converts scores to probabilities, then weights the outputs according to prediction confidence. For encoder $j \in \{ s , b \}$ , a learned temperature $T _ { j } > 0$ adjusts the concentration of its output distribution through temperature scaling (Guo et al., 2017), yielding probabilities $p ^ { j }$ and log probabilities $\ell ^ { j }$ :

$$
\begin{array} { l } { { \ell ^ { j } = \log \operatorname { s o f t m a x } ( z ^ { j } / T _ { j } ) , } } \\ { { p ^ { j } = \exp ( \ell ^ { j } ) . } } \end{array}\tag{5}
$$

The module assigns weights to the two outputs based on prediction confidence. The seven components of c comprise each encoder’s maximum class probability, top-two margin, and normalized entropy, plus the normalized Jensen–Shannon divergence between their distributions. The first six characterize the confidence of each encoder, and the last measures disagreement between their predictions; Appendix C gives their definitions.

A gate $g = \sigma ( w _ { g } ^ { \top } c + b _ { g } )$ , where σ is the sigmoid function, assigns weight g to the statistical encoder and $1 - g$ to the causal attention encoder. Here $w _ { g }$ and $b _ { g }$ are the gate’s learned weight vector and bias; the subscript g identifies the gating parameters. For class i, the module first takes the log of the weighted probability, then applies a linear correction to the corresponding log probabilities from the two encoders:

$$
r _ { i } = \log \Bigl ( g p _ { i } ^ { s } + ( 1 - g ) p _ { i } ^ { b } \Bigr ) ,\tag{6}
$$

$$
s _ { i } = r _ { i } + u _ { i } \ell _ { i } ^ { s } + v _ { i } \ell _ { i } ^ { b } + a _ { i } .\tag{7}
$$

Here $u _ { i }$ and $v _ { i }$ are learned coeficients for the statistical and causal encoders’ log probabilities, respectively, and $a _ { i }$ is a learned bias for class i. The predicted class is $\operatorname { a r g m a x } _ { i } s _ { i }$

## 4.4 Training Objective and Cross-Fitting

We train the two encoders separately on the training set and use five-fold cross-fitting to obtain their out-of-fold predictions for the classification module. For each fold, both encoders are trained on the other four folds and used to score the held-out answers. The held-out predictions from all five folds form a single dataset in which each training answer has two score vectors from encoders trained without that answer. Feature transformations are estimated using only the four folds used to train the corresponding encoders.

We first fit the two temperatures on the outof-fold predictions by minimizing negative loglikelihood. With these temperatures fixed, the gating and per-class correction parameters θ are jointly optimized on all out-of-fold scores and their model labels:

$$
\mathcal { L } ( \theta ) = \frac { 1 } { n } \sum _ { m = 1 } ^ { n } \mathbf { C E } ( s ^ { ( m ) } , y _ { m } ) + \frac { 0 . 0 1 } { 2 } \| \theta \| _ { 2 } ^ { 2 } .\tag{8}
$$

Here n is the number of out-of-fold answers used to fit the classification module, CE is the crossentropy, $\mathbf { \Psi } _ { s } ^ { ( m ) } \in \mathbb { R } ^ { C }$ is the final score vector of the m-th out-of-fold answer, and $y _ { m }$ is its model label. Checkpoints for the encoders trained on the full training set, the fold-specific encoders, and the classification module are selected by macro-F1 on the same validation set. At inference, the encoders trained on the full training set, the fitted temperatures, and the classification module identify the source model from one answer. The encoder parameters remain fixed during temperature calibration and training of the classification module. Appendix C gives the optimization and stopping settings, along with the score aggregation procedure used in the multiple-answer analysis.

## 5 Experimental Setup

## 5.1 Evaluation Protocol

The main experiments evaluate source-model identification from a single answer containing 294 integers. All methods use the fixed splits and 51 classes in Section 3. We select configurations by validation macro-F1 and report accuracy and macro-F1 on the test set. The default random seed is 42. Appendix C gives SMCT’s optimization and stopping settings; Appendix D gives baseline implementations.

Metrics. Accuracy is the proportion of correct predictions across all evaluated answers. Macro-F1 is the unweighted mean of the F1 scores for the 51 model classes.

<table><tr><td>Method</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>Random forest</td><td>68.39</td><td>61.88</td></tr><tr><td>Transformer</td><td>79.10</td><td>75.52</td></tr><tr><td>CatBoost</td><td>82.54</td><td>79.83</td></tr><tr><td>XGBoost</td><td>82.80</td><td>80.01</td></tr><tr><td>MultiROCKET + statistics</td><td>86.90</td><td>83.86</td></tr><tr><td>Ridge</td><td>87.96</td><td>85.13</td></tr><tr><td>1m-detector (429D)</td><td>88.89</td><td>86.65</td></tr><tr><td>RBF-SVM</td><td>88.49</td><td>86.67</td></tr><tr><td>Set Transformer</td><td>89.15</td><td>87.18</td></tr><tr><td>Logistic regression</td><td>89.68</td><td>88.04</td></tr><tr><td>LDA</td><td>91.27</td><td>89.45</td></tr><tr><td>MLP</td><td>91.27</td><td>89.57</td></tr><tr><td>SMCT</td><td>92.72</td><td>91.10</td></tr></table>

Table 2: Single-answer results on the 756 test answers of NumPrint.

## 5.2 Baselines

We compare SMCT with statistical classifiers, sequence models, and an existing integer-sequence fingerprinting method. LDA, logistic regression, MLP, RBF-SVM (Cortes and Vapnik, 1995), Ridge, random forest (Breiman, 2001), XGBoost (Chen and Guestrin, 2016), and Cat-Boost (Prokhorenkova et al., 2018) share the 944- dimensional statistical input.

Sequence baselines read all 294 integers and include a Transformer encoder (Vaswani et al., 2017), integer-embedding LSTMs, InceptionTime, MultiROCKET, and HYDRA. We also compare the complete 429-dimensional lm-detector (Ikaleio, 2026) and multinomial naive Bayes. MultiROCKET with statistics concatenates 49,728 convolutional features with the 944 statistics and classifies with Ridge, to test whether local patterns provide information beyond the predefined statistics.

Table 2 presents the main comparison; Appendix D gives the complete baseline results and implementation settings.

## 6 Results and Ablations

## 6.1 Single-Answer Identification

On the 756 test answers, SMCT achieves 92.72% accuracy and 91.10% macro-F1 using one 294- integer sequence per prediction, outperforming all baselines in Table 2. It improves on the strongest baseline, MLP, by 1.45 percentage points in accuracy and 1.53 points in macro-F1.

Compared with lm-detector, which uses integer-frequency and positional summaries, SMCT improves accuracy and macro-F1 by 3.83 and 4.45 percentage points, respectively.

a Integers in one answer  
![](images/1edc20db50337751ed66c79eeebcb4ca1a4503593d32b306b8c86dabd3dae927.jpg)  
b Answers per decision

![](images/e4195e411df7639b5f2921a993bb8aced940d796dd8b043e372fb2cfa09246d9.jpg)  
Figure 3: Efect of sequence length and answer count on SMCT. (a) Models are trained for each prefix length; 294 integers use the main model. (b) Predictions use K distinct answers with shared encoders and separately fitted calibration and classification parameters. Blue: accuracy; dashed terracotta: macro-F1.

## 6.2 Encoder Ablation

We ablate each encoder in turn while retaining the other encoder’s checkpoint and classification head (Table 3). Removing the statistical encoder reduces accuracy and macro-F1 by 5.29 and 5.83 percentage points, respectively; removing the causal encoder reduces them by 1.45 and 1.53 points. The representations provide complementary information, with a larger efect from removing the statistical encoder. Appendix E.1 compares MLP encoders with diferent hidden-layer widths and depths.

## 6.3 Efect of Sequence Length

We retain the first N integers of each answer and train SMCT separately for lengths of 32, 64, 128, 192, 256, and 294. All settings use the same data splits, encoder architectures, and definitions of the 944 statistical features. Statistics are recomputed from the observed prefix. The 294-integer setting uses the main model; the others follow its training configuration (Appendix E). Table 9 reports the complete prefix results.

<table><tr><td>Variant</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>SMCT (full)</td><td>92.72</td><td>91.10</td></tr><tr><td>w/o statistical encoder</td><td>87.43</td><td>85.27</td></tr><tr><td>w/o causal encoder</td><td>91.27</td><td>89.57</td></tr></table>

Table 3: Encoder ablation of SMCT on NumPrint. Single-encoder variants use the retained encoder’s classification head, and all configurations reuse the same encoder checkpoints. TEST accuracy and macro-F1 are percentages.

Accuracy increases from 50.26% at 32 integers to 92.72% at 294, while macro-F1 increases from 42.58% to 91.10% (Figure 3a). Both metrics increase rapidly for short sequences and then level of, indicating diminishing gains from additional integers.

## 6.4 Efect of Answer Count

We next increase the number of answers available for each prediction from one to five. Each encoder’s class-score vectors are averaged across the answers. The single-answer encoders remain fixed, while the temperatures and classification module are fitted separately for each answer count, with the architecture and regularization coeficient held fixed. Settings with two to five answers each use 5,867 fixed test combinations constructed as described in Appendix A. The combinations are unique and contain K distinct answers from the same model; individual answers may appear in multiple combinations.

Accuracy reaches 96.06% with two answers, 97.12% with four, and 97.39% with five. Both accuracy and macro-F1 increase with the number of answers, with diminishing gains (Figure 3b).

## 6.5 Model Confusions and Family Identification

We group the 51 models into 17 families by developer and evaluate family identification by mapping predicted and true model labels to these families (Figure 4 and Table 4).

Cross-family errors account for 32.73% of all model errors with one answer, 11.69% with two, 0.52% with three, and zero with four or five (Figure 4a). Figure 4b shows the specific within-family confusions; Appendices F.1 and F.2 provide the complete matrix and directions.

![](images/0c3b02eeb995a17c768110d2637ea697b85cf6805fc2b24fa7308faeb7008083.jpg)

![](images/4610d4b42403d69c7bcb6efe8bc2daf80d4cbac6876ae8019fbf2838384f5d44.jpg)  
Figure 4: Model confusion patterns. (a) Within-family and cross-family shares of all errors; parentheses show counts and right-hand labels show cross-family percentages. (b) All 11 within-family confusion directions; cells give errors as percentages of true-class examples, with dashes for zero. K is the answer count: K = 1 uses 756 answers; K = 2–5 each use 5,867 groups. Appendix F.2 shows all directions.

<table><tr><td>K</td><td>n</td><td>Model errors</td><td>Cross-family errors</td><td>Family Acc.</td><td>Family Macro-F1</td></tr><tr><td>1</td><td>756</td><td>55</td><td>18</td><td>97.62</td><td>96.49</td></tr><tr><td>2</td><td>5867</td><td>231</td><td>27</td><td>99.54</td><td>98.82</td></tr><tr><td>3</td><td>5867</td><td>194</td><td>1</td><td>99.98</td><td>99.94</td></tr><tr><td>4</td><td>5867</td><td>169</td><td>0</td><td>100.00</td><td>100.00</td></tr><tr><td>5</td><td>5867</td><td>153</td><td>0</td><td>100.00</td><td>100.00</td></tr></table>

Table 4: Model errors and model-family identification across answer counts. K = 1 is evaluated on 756 test answers and K > 1 on 5,867 test groups. Cross-family errors are a subset of model errors; accuracy and macro-F1 refer to model-family identification, in percent.

Family identification accuracy rises from 97.62% with one answer to 100% with four or five; macro-F1 also reaches 100% (Table 4). All remaining errors at four or five answers occur within families, suggesting that distinguishing closely related models from the same developer remains the main challenge.

## 7 Conclusion

We study language model identification from a single response containing a fixed-length integer sequence, using both statistical and behavioral features of the multiple numerical observations. The NumPrint benchmark introduced for this task contains 5,036 answers from 51 models collected using prompts in three languages. The proposed SMCT architecture learns these representations through a statistical-feature encoder and a causal Transformer encoder.

In 51-class closed-set identification, SMCT achieves 92.72% accuracy and 91.10% macro-F1 from one 294-integer answer. The two representations provide complementary identification cues. Observing more integers improves single-answer identification, and additional answers yield further gains. Within the evaluated candidate set, numerical preferences and sequential dependencies in a single answer provide suficient fingerprint information for source-model identification, ofering users evidence beyond the declared model name.

## Limitations

Model coverage and task scope. NumPrint covers 51 known models, and our experiments identify the source model among these candidates. Extending identification to models outside this set requires unknown-model detection and updates to the reference set.

Variation in collection conditions. The integer sequences were collected using multiple prompt languages and interfaces. System prompts, generation settings, reasoning modes, and conversation context were not independently controlled. Controlled collection from the same models would clarify how these factors change numerical fingerprints and how identification transfers between service configurations.

Extensions of the integer-sequence generation task. The study uses sequences of 294 integers from 1 to 355. The sequence-length experiment evaluates prefixes of the collected answers to assess the efect of the number of observed integers; it does not compare prompts that directly request diferent sequence lengths. Future work could vary the requested sequence length or value range directly and evaluate other numerical tasks and toolassisted generation to assess applicability across generation conditions.

## Ethical Considerations

This study uses integer-sequence generation tasks to identify source models, supporting modelservice verification and experimental reproducibility. The data consist of model-generated integer sequences, and the task requires no personal information. When releasing data, we will limit the materials to numerical answers, model labels, and metadata needed for reproduction, remove API keys and unrelated conversation content, and assess sharing permissions under the applicable service and datause terms.

## Acknowledgements

We used OpenAI Codex to assist with experimental code implementation, literature discovery, and translation and revision. The authors take full responsibility for the content and reported results.

## References

Leo Breiman. 2001. Random forests. Machine Learning, 45(1):5–32.

Tomáš Bruckner. 2026. One token is enough: Fingerprinting and verifying large language models from single-token output distributions. arXiv preprint, arXiv:2607.10252.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22ndACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pages 785–794. ACM.

Corinna Cortes and Vladimir Vapnik. 1995. Supportvector networks. Machine Learning, 20(3):273–297.

Angus Dempster, François Petitjean, and Geofrey I. Webb. 2020. ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery, 34(5):1454–1495.

Angus Dempster, Daniel F. Schmidt, and Geofrey I. Webb. 2021. MiniRocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, pages 248–257. ACM.

Angus Dempster, Daniel F. Schmidt, and Geofrey I. Webb. 2023. Hydra: competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery, 37(5):1779–1805.

Zhiyuan Fu, Junfan Chen, Lan Zhang, Ting Yang, Jun Niu, Hongyu Sun, Ruidong Li, Peng Liu, Jice Wang, Fannv He, and Yuqing Zhang. 2025. FDLLM: A dedicated detector for black-box LLMs fingerprinting. In 2025 IEEE 24th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom), pages 1374–1379. IEEE.

Irena Gao, Percy Liang, and Carlos Guestrin. 2025. Model equality testing: Which model is this API serving? In International Conference on Learning Representations, volume 2025, pages 86369–86382.

Jia Gu, Liang Pang, Huawei Shen, and Xueqi Cheng. 2025. Do LLMs play dice? Exploring probability distribution sampling in large language models for behavioral simulation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 5375–5390, Abu Dhabi, UAE. Association for Computational Linguistics.

Martin Gubri, Dennis Ulmer, Hwaran Lee, Sangdoo Yun, and Seong Joon Oh. 2024. TRAP: Targeted random adversarial prompt honeypot for blackbox identification. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11496–11517, Bangkok, Thailand. Association for Computational Linguistics.

Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321– 1330. PMLR.

Zihao Guo, Hongtao Lv, Chaoli Zhang, Yibowen Zhao, Yixin Zhang, and Lizhen Cui. 2025. The illusion of randomness: How LLMs fail to emulate stochastic decision-making in rock-paper-scissors games? In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 8618–8637, Suzhou, China. Association for Computational Linguistics.

Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang Zhang. 2024. MGTBench: Benchmarking machine-generated text detection. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, CCS ’24, pages 2251–2265. ACM.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural Computation, 9(8):1735–1780.

Ikaleio. 2026. lm-detector: LM Fingerpoint Detector. GitHub software repository. Baseline code snapshot: commit d53d3f5.

Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier, Daniel F. Schmidt, Jonathan Weber, Geofrey I. Webb, Lhassane Idoumghar, Pierre-Alain Muller, and François Petitjean. 2020. InceptionTime: Finding AlexNet for time series classification. Data Mining and Knowledge Discovery, 34(6):1936–1962.

Heng Jin, Chaoyu Zhang, Shanghao Shi, Wenjing Lou, and Y. Thomas Hou. 2024. ProFLingo: A fingerprinting-based intellectual property protection scheme for large language models. In 2024 IEEE Conference on Communications and Network Secu rity (CNS), pages 1–9. IEEE.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. 2023. A watermark for large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 17061–17084. PMLR.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam R. Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set Transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3744–3753. PMLR.

Dario Pasquini, Evgenios M. Kornaropoulos, and Giuseppe Ateniese. 2025. LLMmap: Fingerprinting for large language models. In 34th USENIX Security Symposium (USENIX Security 25), pages 299–318. USENIX Association.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: unbiased boosting with categorical features. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Gurvan Richardeau, Gohar Dashyan, Erwan Le Merrer, and Gilles Trédan. 2026. FLIPS: Instancefingerprinting for LLMs via pseudo-random sequences. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 38974–38993. PMLR.

Chang Wei Tan, Angus Dempster, Christoph Bergmeir, and Geofrey I. Webb. 2022. MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Mining and Knowledge Discovery, 36(5):1623–1646.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Osama Mohammed Afzal, Tarek Mahmoud, Giovanni Puccetti, Thomas Arnold, Alham Aji, Nizar Habash, Iryna Gurevych, and Preslav Nakov. 2024. M4GT-Bench: Evaluation benchmark for black-box machine-generated text detection. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3964–3992, Bangkok, Thailand. Association for Computational Linguistics.

Jiashu Xu, Fei Wang, Mingyu Derek Ma, Pang Wei Koh, Chaowei Xiao, and Muhao Chen. 2024. Instructional fingerprinting of large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3277– 3306, Mexico City, Mexico. Association for Computational Linguistics.

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. 2017. Deep Sets. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Jie Zhang, Dongrui Liu, Chen Qian, Linfeng Zhang, Yong Liu, Yu Qiao, and Jing Shao. 2025. REEF: Representation encoding fingerprints for large language models. In International Conference on Learning Representations, volume 2025, pages 48092–48117.

Minda Zhao, Yilun Du, and Mengyu Wang. 2026. Large language models are bad dice players: LLMs struggle to generate random numbers from statistical distributions. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22942– 22959. Association for Computational Linguistics.

## A Data Inventory and Prompt Details

Prompts. The benchmark uses three prompts, in Chinese, Japanese, and English, all requesting 294 integers in 1–355.

Table 5 gives the split counts of every retained label. The inventory preserves the recorded model labels, including historical version names. The language annotations comprise 1,622 Chinese, 1,711 Japanese, and 1,703 English answers. Promptlanguage metadata are not classifier inputs.

## Chinese prompt.

这是一次无语义数值选择。请为 294个位置分别凭第一反应选择一个 1 到355（含端点）的整数。本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括Python、代码执行器、计算器、搜索、API 和外部随机数生成器。本次请重新选择全部294 个值。如果对话中已有此前生成的数字序列，不要通过复制、翻译、重排或仅修改少数数字来复用它。单个数字与此前回答中的数字偶然相同是允许的。每个位置都要单独凭第一反应选择；不要从1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。偶然想到相同数字是有效的；每项写出后不要再排序、重排、去重、替换或修正。数字之间用清晰的常见分隔符，直接从第一个取值开始，完整输出序列，不要解释。

## Japanese prompt.

これは意味を持たない数値選択の課題です。294個の各位置について、1から 355 まで（両端を含む）の整数を、最初の直感で個別に一つずつ選んでください。この課題は現在の言語モデル自身が直接行ってください。Python、コード実行ツール、計算機、検索、API、外部の乱数生成器など、いかなるツールも呼び出したり利用したりしないでください。今回は 294個すべての値を新たに選んでください。会話中に以前生成した数列がある場合、それをコピー、翻訳、並べ替え、または少数の数字だけ変更することで再利用しないでください。個々の数字が以前の回答に含まれる数字と偶然一致することは問題ありません。各位置の値は、最初の直感でそれぞれ独立に選んでください。1 から順に数えたり、連続して増加または減少する数列を作ったりせず、等差数列、周期、同じブロックの繰り返し、その他の規則的なパターンも使わないでください。偶然同じ数字を再び選んでもかまいません。各項目を書いた後に、並べ替え、順序の変更、重複の削除、置き換え、修正をしないでください。数字は境界が明確な一般的な区切り文字で区切ってください。選んだ最初の値から直接出力を始め、数列を最後まで出力し、説明は付けないでください。

## English prompt.

This is a numerical-choice task with no semantic meaning. For each of 294 positions, independently choose an integer from 1 to 355 inclusive by first instinct. The current language model must complete this task directly: do not call or use any tools, including Python, code execution tools, calculators, search, APIs, or external random-number generators.Choose all 294 values anew for this request. If the conversation contains previously generated numerical sequences, do not reuse one by copying, translating, reordering, or changing only a few numbers. Individual numbers may coincidentally match numbers in previous responses. Make a separate firstinstinct choice for every position. Do not count from 1, produce consecutive increasing or decreasing sequences, or use arithmetic progressions, cycles, repeated blocks, or other systematic patterns. Accidentally choosing the same number again is valid. Once an item is written, do not sort, reorder, deduplicate, replace, or correct it. Separate the numbers with clear, common delimiters. Start directly with the first chosen value, output the complete sequence, and give no explanation.

<table><tr><td>Collected label</td><td>Train</td><td>Val</td><td>Test</td><td>Collected label</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>gpt-6-luna</td><td>469</td><td>100</td><td>101</td><td>qwen3.8-27b</td><td>52</td><td>11</td><td>12</td></tr><tr><td>grok-4.6</td><td>176</td><td>38</td><td>38</td><td>qwen3.8-max</td><td>52</td><td>11</td><td>11</td></tr><tr><td>gemini-3.8-flash</td><td>132</td><td>28</td><td>28</td><td>gpt-6-astra</td><td>51</td><td>11</td><td>11</td></tr><tr><td>claude-opus-5.5</td><td>116</td><td>25</td><td>25</td><td>qwen3.7-max</td><td>50</td><td>11</td><td>10</td></tr><tr><td>glm-5.3-flash</td><td>113</td><td>24</td><td>24</td><td>qwen3.5-397b-a17b</td><td>50</td><td>11</td><td>10</td></tr><tr><td>muse-spark-1.3</td><td>113</td><td>24</td><td>24</td><td>glm-5.1</td><td>48</td><td>10</td><td>11</td></tr><tr><td>gpt-5.6-terra</td><td>94</td><td>20</td><td>20</td><td>mimo-v2.6-pro</td><td>45</td><td>10</td><td>10</td></tr><tr><td>gpt-5.6-sol</td><td>90</td><td>19</td><td>19</td><td>claude-haiku-5.5</td><td>45</td><td>10</td><td>9</td></tr><tr><td>step-5-preview</td><td>83</td><td>17</td><td>18</td><td>gpt-5.6-luna</td><td>44</td><td>10</td><td>9</td></tr><tr><td>mistral-large-4</td><td>79</td><td>17</td><td>17</td><td>hy3</td><td>44</td><td>10</td><td>9</td></tr><tr><td>deepseek-v4.1-flash</td><td>75</td><td>16</td><td>16</td><td>minimax-m2.7</td><td>44</td><td>10</td><td>9</td></tr><tr><td>mimo-v2.6-flash</td><td>70</td><td>15</td><td>15</td><td>gemini-3.1-pro-preview</td><td>43</td><td>9</td><td>9</td></tr><tr><td>hy4-preview</td><td>70</td><td>15</td><td>15</td><td>doubao-2.1-turbo</td><td>42</td><td>9</td><td>9</td></tr><tr><td>claude-fable-5.1</td><td>68</td><td>14</td><td>15</td><td>doubao-2.1-lite</td><td>42</td><td>9</td><td>9</td></tr><tr><td>claude-opus-4.6</td><td>67</td><td>14</td><td>15</td><td>doubao-2.1-pro</td><td>42</td><td>9</td><td>9</td></tr><tr><td>inkling-small</td><td>64</td><td>13</td><td>14</td><td>qwen3.8-flash</td><td>42</td><td>9</td><td>9</td></tr><tr><td>gpt-6-sol</td><td>63</td><td>13</td><td>14</td><td>minimax-m3</td><td>41</td><td>9</td><td>9</td></tr><tr><td>gemini-3.7-flash</td><td>63</td><td>13</td><td>14</td><td>muse-wonder</td><td>40</td><td>9</td><td>8</td></tr><tr><td>glm-5.3</td><td>62</td><td>13</td><td>14</td><td>muse-spark-1.2</td><td>40</td><td>9</td><td>8</td></tr><tr><td>muse-spark-1.1</td><td>59</td><td>12</td><td>13</td><td>claude-haiku-4.5-20251001</td><td>38</td><td>9</td><td>8</td></tr><tr><td>claude-sonnet-5.5</td><td>56</td><td>12</td><td>12</td><td>grok-4.7</td><td>38</td><td>9</td><td>8</td></tr><tr><td>inkling</td><td>56</td><td>12</td><td>12</td><td>mimo-v2.5-pro</td><td>38</td><td>8</td><td>8</td></tr><tr><td>kimi-k3</td><td>55</td><td>12</td><td>12</td><td>mistral-medium-3.5</td><td>36</td><td>8</td><td>8</td></tr><tr><td>gpt-6.1-sol</td><td>55</td><td>12</td><td>12</td><td>mistral-large-3</td><td>31</td><td>7</td><td>6</td></tr><tr><td>solar-pro4</td><td>55</td><td>11</td><td>12</td><td>gemini-4-argon</td><td>29</td><td>7</td><td>6</td></tr><tr><td>glm-5.2</td><td>55</td><td>11</td><td>12</td><td></td><td></td><td></td><td></td></tr></table>

Table 5: All 51 collected model labels with frozen split counts. Labels follow the collection records; labels that distinguish reasoning strength are consolidated.

Record cleaning. We parse answers imported manually or programmatically into ordered integer lists and standardize their model labels. We identify duplicate records by comparing the first 64 integers. Within each model label, we retain only one record from each group of duplicates and discard the rest. Answers containing explanatory text, arithmetic expressions, or out-of-range values are excluded.

Label consolidation and model selection. We use a fixed mapping to merge labels that difer only in reasoning-strength sufixes, while retaining suffixes that are part of the model name. We select the 51 models by ranking them according to the number of eligible answers, breaking ties by the total answer count and then by model name.

Eligible answers. An eligible answer contains at least 294 valid integers. We retain its first 294 entries in the original order. Of the 5,036 retained answers, 440 are truncated from longer outputs.

Additional-answer groups. For $K \in \mathsf { \Gamma }$ {2, 3, 4, 5}, we form answer groups separately for each model within each data split. When a class contains n answers in that subset, all answer-count settings use the same target number of groups

$$
q ( n ) = \operatorname* { m i n } \left( 1 0 0 0 , \operatorname* { m i n } _ { k \in \{ 2 , 3 , 4 , 5 \} } { \binom { n } { k } } \right) .\tag{A1}
$$

We first sort the answers by their fixed sample identifiers and form consecutive, non-overlapping groups. We then sample additional unique unordered combinations using a random generator seeded from the global random seed, the answer count K, and the class index. Here q(n) is the number of groups targeted for a class with n answers. Each group contains distinct answers, although individual answers may appear in more than one group. Every answer-count setting yields 5,833 validation groups and 5,867 test groups.

<table><tr><td>Feature family</td><td>Dimensions</td></tr><tr><td>Integer histogram</td><td>355</td></tr><tr><td>Four positional histograms</td><td>64</td></tr><tr><td>Final-digit histogram</td><td>10</td></tr><tr><td>Modulo histograms</td><td>65</td></tr><tr><td>Tens/hundreds histograms</td><td>14</td></tr><tr><td>Signed/absolute increments</td><td>48</td></tr><tr><td>Coarse/digit transitions</td><td>164</td></tr><tr><td>Moments, lags, spectrum, blocks</td><td>54</td></tr><tr><td>Additional sequence statistics</td><td>170</td></tr><tr><td>Total</td><td>944</td></tr></table>

Table 6: Statistical feature dimensions.

## B Statistical Representation

Integer frequencies and normalization. For integer i, we use a smoothed square-root frequency:

$$
f _ { i } = { \sqrt { \frac { n _ { i } + 0 . 5 } { L + 3 5 5 \times 0 . 5 } } } , \quad i = 1 , \ldots , 3 5 5 ,\tag{B1}
$$

where $n _ { i }$ is the number of occurrences of integer i in a sequence of length L.

The representation contains 944 features with fixed definitions and ordering; Table 6 lists the dimensionality of each group. For each feature, we estimate the mean and variance across answers in the corresponding training subset and use these statistics for standardization. Each feature block is then ℓ<sub>2</sub>-normalized. Before the MLP, the normal-<sub>ized</sub> <sub>vector</sub> <sub>is</sub> <sub>multiplied</sub> <sub>by</sub> √<sub>944.</sub>

The integer-frequency, position-and-final-digit, and expanded-feature blocks are standardized and normalized separately, with squared-norm weights 0.6, 0.2, and 0.2; the merged block receives weight 0.8, and the sequence-statistics block receives 0.2. All standardization parameters are estimated exclusively from the corresponding training subset.

If an additional sequence feature is constant or duplicates another feature in a fold’s training subset, its entry is set to zero in that fold. This preserves the same feature definitions and ordering across folds while keeping the input dimension fixed at 944. The original integer sequences remain unchanged.

## C Training and Cross-Fitting Details

Architecture and cross-fitting. The statisticalfeature encoder has 1,019,955 parameters and the causal attention encoder has 43,059. Integers 1– 355 map to embedding indices 0–354. The full training set contains 3,525 answers divided into five folds. For each fold, the encoders are trained on 2,820 answers and produce out-of-fold scores for the 705 held-out answers.

The statistical MLP uses AdamW with learning rate 0.002, weight decay 0.1, batch size 128, GELU activation, hidden-layer dropout 0.4, and label smoothing 0.05. The causal attention encoder uses learning rate 0.001, weight decay 0.01, batch size 128, and class-balanced cross-entropy with label smoothing. Attention-weight dropout is 0, input and attention-output dropout are 0.1, and classification-head dropout is 0.3.

Encoder optimization. For the MLP, we clip the gradient norm at 5 and reduce the learning rate when validation macro-F1 plateaus, using a reduction factor of 0.2, patience of 6 epochs, minimum learning rate of 10<sup>−5</sup>, and absolute improvement threshold of $1 0 ^ { - 4 } .$ . For the causal encoder, the gradient-norm limit is 1, the reduction factor is 0.3, and patience is 5 epochs; the minimum learning rate and improvement threshold are unchanged. Both encoders use cross-entropy with label smoothing. The causal encoder weights classes by inverse training-class frequency, while the statistical MLP uses equal class weights.

Training stops when the minimum learning rate has been maintained for at least 10 epochs, validation macro-F1 has not improved meaningfully for 25 consecutive epochs, and the training and validation evaluation losses each change by less than 0.5% between adjacent five-epoch windows. Training may also stop after 35 epochs without improvement if validation loss increases. All reported results use the selected checkpoint.

Out-of-fold training data. The single-answer classification module is trained on 3,525 out-offold predictions. For two, three, four, and five answers, the corresponding out-of-fold training sets contain 9,133, 10,001, 9,960, and 9,810 groups, respectively. All answers within a group come from the same held-out fold.

Confidence features and parameter count. For encoder j, let its calibrated probabilities be sorted as $p _ { ( 1 ) } ^ { j } , p _ { ( 2 ) } ^ { j } , . . . ,$ and define the maximum probability $m ^ { j } = p _ { ( 1 ) } ^ { j }$ and the top-two margin $d ^ { j } \ =$ $p _ { ( 1 ) } ^ { j } - p _ { ( 2 ) } ^ { j }$ . The normalized entropy is

$$
e ^ { j } = - \frac { \sum _ { i } p _ { i } ^ { j } \log ( p _ { i } ^ { j } ) } { \log C } .
$$

With $q = ( p ^ { s } + p ^ { b } ) / 2$ , the normalized divergence is

$$
J = { \frac { \operatorname { K L } ( p ^ { s } \| q ) + \operatorname { K L } ( p ^ { b } \| q ) } { 2 \log 2 } } ,
$$

where KL denotes the Kullback–Leibler divergence. Hence $c = ( m ^ { s } , d ^ { s } , e ^ { s } , m ^ { b } , d ^ { b } , e ^ { b } , J )$ . The gate has 8 parameters for its 7-dimensional input, and the per-class linear correction has $3 C$ parameters; for $C = 5 1$ the total is 161, excluding the two temperatures.

Calibration and classification-module optimization. We fit the temperatures by minimizing mean negative log-likelihood on the training set’s out-of-fold predictions, optimizing over a bounded log-temperature interval corresponding to $T \in$ [0.05, 1000]. All parameters of the 161-parameter classification module are initialized to zero. The training objective is full-batch cross-entropy plus $0 . 0 1 / 2$ times the squared parameter norm. Optimization uses L-BFGS with numerical fallbacks and requires the maximum absolute gradient component to be at most $1 0 ^ { - 6 }$ . We select the checkpoint by validation macro-F1.

Multiple-answer extension. In the multipleanswer analysis, we average the raw class scores for a group G of K answers separately for each encoder:

$$
\bar { z } ^ { j } ( G ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } z ^ { j } ( x ^ { ( k ) } ) .\tag{C1}
$$

We encode each answer independently and then aggregate the score vectors within each encoder. The temperatures and the 161-parameter classification module are fitted separately for each answer count. Here $x ^ { ( k ) }$ is the k-th answer in G. All answer-count settings share the two encoders trained on the full training set. Out-of-fold training groups contain answers from the same held-out fold.

## D Baseline Implementations and Full Results

The nine feature-based classifiers described in Section 5 receive the same normalized 944- dimensional inputs. For Set Transformer, each answer is represented by its 944-dimensional statistical vector and treated as a single set element. Checkpoints are selected by single-answer validation macro-F1. The complete lm-detector retains its 429-dimensional representation and

<table><tr><td>Method</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>LDA</td><td>91.27</td><td>89.45</td></tr><tr><td>Logistic regression</td><td>89.68</td><td>88.04</td></tr><tr><td>Ridge</td><td>87.96</td><td>85.13</td></tr><tr><td>RBF-SVM</td><td>88.49</td><td>86.67</td></tr><tr><td>Random forest</td><td>68.39</td><td>61.88</td></tr><tr><td>XGBoost</td><td>82.80</td><td>80.01</td></tr><tr><td>CatBoost</td><td>82.54</td><td>79.83</td></tr><tr><td>MLP</td><td>91.27</td><td>89.57</td></tr><tr><td>Set Transformer</td><td>89.15</td><td>87.18</td></tr><tr><td>1m-detector (429D)</td><td>88.89</td><td>86.65</td></tr><tr><td>Multinomial NB</td><td>84.66</td><td>80.71</td></tr><tr><td>Transformer</td><td>79.10</td><td>75.52</td></tr><tr><td>LSTM (64)</td><td>21.69</td><td>4.52</td></tr><tr><td>LSTM (128)</td><td>36.90</td><td>21.09</td></tr><tr><td>LSTM (256)</td><td>37.57</td><td>25.78</td></tr><tr><td>LSTM (512)</td><td>39.29</td><td>30.45</td></tr><tr><td>InceptionTime (5 networks)</td><td>51.32</td><td>41.74</td></tr><tr><td>MultiROCKET</td><td>32.01</td><td>18.36</td></tr><tr><td> $\mathrm { { H Y D R A + R i d g e } }$ </td><td>29.50</td><td>17.19</td></tr><tr><td>HYDRA + MultiROCKET</td><td>30.95</td><td>16.33</td></tr><tr><td>MultiROCKET + statistics</td><td>86.90</td><td>83.86</td></tr><tr><td>SMCT</td><td>92.72</td><td>91.10</td></tr></table>

Table 7: Complete registered single-answer comparison. Sequence models observe all 294 integers and statistical models use the representation described in the text; all are re-evaluated on the unified split and preprocessing of this paper.

LDA/nearest-neighbor/deconfounded-center combination, extending the original 128-item input to all 294 integers.

The Transformer uses one encoder layer, width 64, four heads, a 256-dimensional feedforward layer, learned position embeddings, and mean pooling. Optimization follows the causal encoder settings (Appendix C), with batch size 64 and feedforward dropout 0.1. The sequence LSTMs use integer embeddings with hidden widths 64, 128, 256, or 512. InceptionTime uses the original architecture and an ensemble of five networks initialized with seeds 42–46. MultiROCKET and HY-DRA apply full sequence transformations; Multi-ROCKET + statistics concatenates 49,728 convolutional features with the 944 statistical features and uses a Ridge classifier.

The selected configurations use LDA with shrinkage 0.2 and uniform class priors, logistic regression $C ~ = ~ 1 0 0$ with balanced class weights, RBF-SVM C = 100 with kernel scale 0.3 times the reference scale, and Ridge $\alpha = 0 . 3$ . The random forest uses 300 trees, a feature-sampling fraction of 0.5, a minimum leaf size of 3, and balancedsubsample class weights. XGBoost uses trees of depth 2 and 616 boosting rounds; CatBoost uses trees of depth 4. Both use checkpoints selected on

the validation set.

## E Full Prefix and Capacity Results

Complete SMCT results for diferent answer counts and prefix lengths are reported in Tables 8 and 9.

<table><tr><td>K</td><td>Evaluated</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>1</td><td>756</td><td>92.72</td><td>91.10</td></tr><tr><td>2</td><td>5,867</td><td>96.06</td><td>94.15</td></tr><tr><td>3</td><td>5,867</td><td>96.69</td><td>95.22</td></tr><tr><td>4</td><td>5,867</td><td>97.12</td><td>95.64</td></tr><tr><td>5</td><td>5,867</td><td>97.39</td><td>96.03</td></tr></table>

Table 8: Efect of the number of answers per decision. K is the answer count. TEST accuracy and macro-F1 are percentages.

<table><tr><td>N</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>32</td><td>50.26</td><td>42.58</td></tr><tr><td>64</td><td>70.63</td><td>64.98</td></tr><tr><td>128</td><td>83.60</td><td>80.88</td></tr><tr><td>192</td><td>88.36</td><td>86.50</td></tr><tr><td>256</td><td>91.40</td><td>90.38</td></tr><tr><td>294</td><td>92.72</td><td>91.10</td></tr></table>

Table 9: Efect of visible prefix length on SMCT, using the same 756 original TEST answers. Values are percentages.

Each prefix-length setting retains the original split, the in-fold sample identities, the architecture, and the initialization seed, and its statistical features are recomputed from the observed prefix.

## E.1 MLP Capacity

We compare the main single-hidden-layer MLP with 1,024 units against a wider single-hiddenlayer MLP with 2,048 units and deeper configurations of $5 1 2  2 5 6 , 2 5 6  2 5 6 , 2 5 6  1 2 8 .$ and 128 → 128 → 128, holding the input, random seed, activation, dropout, and weight decay fixed. Table 10 gives the results of each configuration.

<table><tr><td>Hidden width</td><td>Parameters</td><td>Acc.</td><td>Macro-F1</td></tr><tr><td>2048</td><td>2,039,859</td><td>91.14</td><td>89.52</td></tr><tr><td>1024</td><td>1,019,955</td><td>91.27</td><td>89.57</td></tr><tr><td>512→256</td><td>628,275</td><td>90.61</td><td>88.97</td></tr><tr><td>256→256</td><td>320,819</td><td>90.48</td><td>88.69</td></tr><tr><td>256→128</td><td>281,395</td><td>90.74</td><td>89.21</td></tr><tr><td>128→128→128</td><td>160,563</td><td>86.51</td><td>83.39</td></tr></table>

Table 10: MLP capacity comparison. Accuracy and macro-F1 are percentages.

## F Supplementary Confusion Analysis

This section reports the complete confusion matrix and all observed misclassification directions.

## F.1 Complete Confusion Matrix

Figure 5 shows the single-answer test predictions reported in Table 2 for all 51 model labels.

## F.2 All Misclassification Directions Across Answer Counts

Figure 6 includes all 30 true-to-predicted model pairs with at least one error across the one- to fiveanswer settings, complementing the error composition and within-family analysis in Figure 4.

Predicted mode  
SMCT · Single answer  
![](images/77bd99a933a3e81a7d97e61427948811d6e50753642226f7b4d91df5d5de894a.jpg)  
Figure 5: Complete single-answer confusion matrix of SMCT on the 756 TEST answers. Rows are true labels and columns are predicted labels, both sorted alphabetically. Color indicates the proportion of predictions within each row, and of-diagonal cells are annotated with the number of misclassified answers. The matrix contains 701 correct predictions and 55 errors.

![](images/f031a9bd47d2ac7fb7b55fc77426892fa55d1076b722b06f0232e9032f7a9448.jpg)  
Figure 6: All 30 observed misclassification directions across the one- to five-answer SMCT settings, grouped into within-family and cross-family errors. Columns indicate the number of answers K per prediction. Color intensity and cell values indicate the proportion of test answers or groups from the true model misclassified as the predicted model (%); a dash denotes zero. K = 1 uses 756 answers, and K = 2–5 each use 5,867 groups.