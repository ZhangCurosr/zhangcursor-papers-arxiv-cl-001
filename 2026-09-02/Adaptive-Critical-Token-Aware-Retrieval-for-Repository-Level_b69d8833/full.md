# Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation

Kefeng Duan<sup>∗</sup>, Dewu Zheng<sup>∗</sup>, Yanlin Wang<sup>†</sup>, Terry Yue Zhuo, Mingwei Liu, Jianxing Yu, Jiachi Chen, Ensheng Shi, Xilin Liu, Yuchi Ma, and Zibin Zheng

Abstract—The repository-level code generation task requires synthesizing code that satisfies task requirements while remaining consistent with the target repository context. Since real-world repositories often exceed the input length limits of LLMs, existing approaches commonly adopt retrieval-augmented generation (RAG) to provide repository-specific context. Despite improving repository-context retrieval, existing methods typically provide context as task-level support, without explicitly identifying the critical tokens that require fine-grained repository context during generation. During the autoregressive generation process of LLMs, errors often concentrate at a small number of decisive positions: once such tokens are generated incorrectly, subsequent code may follow an incorrect semantic path and eventually lead to functional failure. We refer to these positions as “critical tokens”. In this paper, we propose ACTOR, an adaptive critical tokenaware retrieval framework for repository-level code generation. ACTOR identifies critical tokens during generation and triggers targeted retrieval on demand to provide repository context at these decisive positions. In addition, we design a position-aware weighting method for dense retrievers to prioritize context that is more informative for generation. We evaluate ACTOR on two representative repository-level benchmarks, RepoExec and CoderEval. Experimental results show that ACTOR consistently outperforms state-of-the-art methods, achieving relative improvements of 8.4% on RepoExec and 15.4% on CoderEval. Beyond performance gains, we systematically quantify the impact of critical tokens, revealing their central role in major generation failures and highlighting the necessity of targeted retrieval strategies. We provide the code and data at https://github.com/DeepSoftwareAnalytics/ACToR.

Index Terms—Repository-level code generation, large language models, retrieval-augmented generation, critical token, dense retriever.

## I. INTRODUCTION

Recent LLMs are increasingly used in real-world software development [1]–[4]. Unlike standalone code-generation tasks [5], [6], repository-level code generation targets realworld development within existing repositories. In this setting, generated code must not only satisfy task requirements but also remain consistent with project-specific APIs, dependencies, data structures, coding conventions, and the broader repository context [7]–[9].

Therefore, repository context is essential for resolving project-specific dependencies and preserving consistency with the existing codebase. However, real-world code repositories often exceed the input length limits of LLMs, making it infeasible to provide the entire repository as input. To address this challenge, existing studies have widely adopted retrieval augmented generation (RAG), which first retrieves task-relevant context from the repository and then incorporates it into the generation prompt. Representative repository-level RAG methods have improved contextual support from several perspectives, including context quality and retrieval decision-making. To improve context quality, RepoCoder [10] iteratively constructs new retrieval queries using generated results from previous rounds, Cocomic [11] locates relevant code dependencies based on a project context graph, and DRACO [12] retrieves finegrained code entities through dataflow analysis. In retrieval decision-making, RepoFormer [13] determines whether retrieval should be triggered for the current task, RLPG [14] identifies the type of dependency context required by the task, and ProCC [15] dynamically selects suitable retrieval prompt construction strategies. Overall, these methods improve repositorylevel code generation mainly by enhancing how repository context is obtained, selected, and organized.

Although the above methods improve the quality of repository context from the perspectives of retrieval strategy, context source, and retrieval decision-making, they still organize and use context mainly at the task level, lacking explicit modeling of key token-level decisions during generation. Specifically, existing methods typically treat the retrieved context as holistic support for the entire generation task, implicitly assuming that a set of relevant repository contexts can continuously serve the subsequent generation process. However, code generation is a highly path-dependent autoregressive process, and different generation positions may require different contextual information.

In repository-level code generation, an incorrect API name, variable reference, index, or control-flow token can cause subsequent generation to deviate from repository semantics and eventually fail the entire implementation. In other words, code generation errors are often not accumulated uniformly across all tokens, but instead concentrate at a small number of positions that have a decisive impact on the final result. We call these positions critical tokens: a small number of tokens in the LLM inference chain whose correctness plays a decisive role in the correctness of the final answer, and whose incorrect generation may cause the entire generated result to fail. Therefore, the core limitation of existing retrieval pipelines is not merely that retrieval occurs before generation or remains at the query level, but that they lack a mechanism to identify these critical tokens and provide targeted repository context support at these key positions.

Based on this observation, we introduce ACTOR (Adaptive Critical Token-aware Retrieval), a dynamic, fine-grained retrieval framework. Unlike static methods, our framework operates during the generation process. It identifies critical tokens where the model requires additional context and performs targeted, on-demand retrieval for each one. This adaptive mechanism ensures that the model always has the precise information it needs, thereby mitigating the quality degradation and functional failures common in existing systems. Besides, we propose a position-aware weighting method for dense retrievers. This allows us to retrieve context that is more informative for generation.

We validate our approach on two standard repository-level code generation benchmarks: RepoExec [16] and CoderEval [7]. ACTOR consistently improves over strong repository-level baselines, with relative gains of 8.4% on RepoExec and 15.4% on CoderEval. Complementing these end-to-end results, we provide a quantitative characterization of critical tokens: they occupy only a small fraction of generated positions (about 5–11% across models), yet their composition and syntactic profiles concentrate much of the model’s error and uncertainty, motivating sparse, generation-time retrieval updates rather than repository context supplied only once up front. Our analysis further ties major failures to this small set of pivotal positions, underscoring the necessity of targeted retrieval strategies like ACTOR.

We summarize our contributions as follows:

• We introduce the concept of critical tokens for repositorylevel code generation. To our knowledge, this is the first work to systematically quantify the decisive impact these tokens have on the quality of the final generated code.

• We propose a novel targeted retrieval framework designed for these critical tokens. This framework performs on-demand retrieval to supply the language model with precise, timely context at the most crucial steps of the generation process.

• We design a position-aware weighted pooling method to enhance our retrieval mechanism. This technique refines the context embedding process, significantly boosting the precision of retrieval.

• We conduct extensive experiments on representative repository-level code generation benchmarks, demonstrating that our approach significantly and consistently outperforms existing SOTA methods.

## II. MOTIVATING EXAMPLES

This section uses two real examples from CoderEval [7] to illustrate two key properties of critical tokens. First, although any token change may affect subsequent generation, critical tokens are distinctive because their errors often change the model’s core judgment about the current implementation logic, causing later code to follow an incorrect semantic path and eventually leading to function-level failure. Second, critical tokens are not limited to a fixed syntactic form; they may appear as API names, variable references, index expressions, or even semantic cues in comments. Therefore, a single pregeneration retrieval step or a fixed syntactic rule can hardly cover the tokens that truly determine the final generation result.

![](images/23b22dd40b9a497669fd98db74011fadc76aed4672a24a4fc243331f700a00a3.jpg)

Fig. 1. Motivating example from CoderEval task 62b8d22f [7]: an incorrect critical token can cause catastrophic failure in subsequent generations (popitem scenario).  
![](images/98b020d97d340cb9b0f215862f7eccccc2001c6c9be1cd8e0c20e3090c53555d.jpg)  
Fig. 2. Motivating example from CoderEval task 62e4fb4d [7]: critical tokens exhibit diverse syntax and can appear in multiple roles (normalize cmd scenario).

Motivating example 1: Figure 1 shows the difference between a critical-token error and an ordinary token error in terms of their consequences. The goal of the popitem function is to remove and return the earliest inserted item from a cache. This behavior depends on a repository-specific ordering structure: the correct implementation should first obtain the earliest key from self.\_order, for example through next(iter(self.\_order)), and then update the cache according to that key. However, at the highlighted position, the model generates self instead of next. This error directly changes the model’s judgment about which repository object the subsequent code should be built around. Instead of constructing an iterator over self.\_order, the model turns to an incorrect call such as self.buffer.popitem.... As a result, the generated function no longer follows the repository’s insertion-order deletion logic; the remaining code is generated around this wrong operation and eventually deviates completely from the intended behavior. In contrast, the noncritical token error in the figure affects only a local type-related detail and does not break the main control-flow or data-flow structure of the function. This comparison shows that critical tokens are not simply tokens predicted incorrectly; they are positions where an error can make the model depart from the core implementation logic and thereby compromise functional correctness.

Motivating example 2: Figure 2 further shows the syntactic and semantic diversity of critical tokens. The normalize cmd function needs to normalize a command representation according to the repository’s existing implementation, including how executable names are parsed, how command tuples are indexed, and which internal helper APIs should be invoked. Therefore, the task requires not only understanding the function description but also following the conventions encoded in the surrounding code. In the comment-related case, an incorrect comment-related token misleads the model into inferring that an additional environment argument is needed, causing it to hallucinate an env parameter that is not supported by the repository API. In the API-related case, the model fails to generate the correct call to parse\_filename and instead falls back to simple string concatenation, such as cmd = exe + cmd[1:]. This bypasses the repository-defined filename parsing logic and makes the normalization process inconsistent with the project implementation. In the index-related case, the model incorrectly uses the entire cmd object instead of accessing a specific element, for example passing cmd directly to normexe, which expects a single element. This indexing error causes a type mismatch, propagates to the return statement, and eventually leads to a runtime failure. Although these three errors involve comments, API calls, and index expressions, they share the same property: each occurs at a position that determines the direction of the subsequent implementation and causes the generated result to deviate from repository semantics.

Together, these examples show that critical tokens have two important properties. On the one hand, their errors can make the model continue generation along an incorrect semantic path, causing function-level failures that go far beyond a local token error. On the other hand, they are syntactically diverse and cannot be fully captured by a fixed syntactic category or a single pre-generation retrieval step. These observations motivate a generation-time mechanism that identifies critical tokens as they arise and provides targeted repository context support at those key positions.

## III. METHODOLOGY

## A. Overall Pipeline

In this section, we introduce an innovative methodology for repository-level code generation, centered on a targeted retrieval mechanism guided by critical tokens. The methodology is divided into two phases: offline training and online inference. In the offline phase, we focus on training a lightweight discriminative ensemble model. Its purpose is to accurately identify critical tokens. During the online inference phase, the methodology enhances code quality through two key techniques: Position-Aware Weighted Retriever and Critical Token-Guided Inference. Figure 3 shows the overall pipeline of our study.

## B. Offline Training

During the training phase, we design a critical token discriminator that identifies critical tokens and triggers dynamic retrieval based on the model’s hidden states during generation. To obtain high-quality training data that match this scenario, we follow the data construction pipeline described below (see Figure 3).

1) Data Construction: This subsection describes the procedure for building our training set, which consists of three stages: Repository Filtering, Function Sampling, and Prompt Construction.

1) Repository Filtering. We select our data from RepoST-Train [8], a large-scale training dataset specialized for repository-level code generation, containing 7,415 functions from 824 repositories. To ensure data quality, we filter repositories according to the following criteria:

✓ The repository must contain more than fifteen training samples (functions).

✓ The repository must have more than ten stars on GitHub. ✓ The repository must be a real-world software project, rather than a collection of solutions for algorithmic challenges (e.g., LeetCode) or academic assignments.

✗ The repository must not appear in our evaluation benchmarks (RepoExec [16] and CoderEval [7]).

After applying these filters, we include 102 repositories for constructing training data.

2) Function Sampling. After repository filtering, we further sample candidate functions within the selected repositories according to the following rules:

✓ The sample must include a function signature.

✓ The sample must include a complete docstring (used to describe function behavior and constraints).

Our training dataset excludes samples that do not meet both conditions. After that, we select 1,203 task samples for our training.

3) Prompt Construction. For each repository, we follow RLCoder [9] and adopt the “Split and Aggregate candidate” strategy to construct repository context candidates. For each task sample, we concatenate the function signature and docstring to form the initial code sequence X. This sequence X serves as the query to a dense retriever. The retriever ranks these candidates by cosine similarity and returns an ordered set of context snippets: $\mathcal { C } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { m } \}$ , which is sorted by descending similarity. We then concatenate the contexts and the initial code sequence in retrieval score order to form prompts as: $P = [ C _ { 1 } \mid C _ { 2 } \mid . . . \mid C _ { m } \mid X ]$ where $^ { 6 6 } | ^ { , 5 }$ denotes a separator between contexts. In practice, we use file-path annotations of each context and the target function prompt as explicit separators to preserve context boundaries and improve readability.

![](images/797edffd02f77be423fb87acf0678f6de6fa2d82a197ce45da75f2600637893e.jpg)  
Fig. 3. The Pipeline of ACTOR.

After that, we treat the corresponding function body as the ground truth and record both the prompt and ground truth for training.

2) Token Evaluation: At this stage, our goal is to determine, for each generated token, whether it is a critical token and to construct a labeled dataset for this classification task. A critical token is a token that the model tends to mispredict and whose erroneous generation would significantly affect subsequent tokens, substantially increasing the risk of functional failure in the generated code. In other words, an error at such a position is not only incorrect locally but also likely to trigger cascading errors or to break program semantics. Based on this definition, we quantify each generated token using the following three criteria:

• Token Mismatch. The model’s top-1 prediction at position i differs from the ground-truth token. The formula can be expressed as:

$$
x _ { i } \neq { \hat { x } } _ { i }\tag{1}
$$

where $x _ { i }$ is the ground-truth token at position i and $\hat { x } _ { i }$ denotes the model’s top-1 prediction. A top-1 mismatch indicates high prediction difficulty at this position. If a downstream impact accompanies this mismatch, the token is more likely to be critical.

• Uncertainty. We introduce entropy to quantify the model’s uncertainty when generating the token at position i:

$$
\mathcal { H } _ { i } = - \sum _ { v \in \mathcal { V } } p _ { i } ( v ) \log p _ { i } ( v ) ,\tag{2}
$$

where $p _ { i } ( v )$ denotes the probability assigned by the model to token v at position $i ,$ and V is the vocabulary of large language models. A larger entropy $\mathcal { H } _ { i }$ indicates greater model uncertainty at that position, and therefore a higher likelihood that the token is critical.

• Subsequent Attention Influence. We propose a metric based on the model’s self-attention mechanism [17] to assess the potential influence of a token on subsequent tokens. Selfattention, the core mechanism of Transformer-based LLMs, is computed as:

$$
A = \mathrm { s o f t m a x } \left( \mathrm { m a s k } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } \right) \right)\tag{3}
$$

where $A _ { j , i }$ denotes the attention weight from position $j$ (Query) to position i (Key/Value). Considering up to the next k tokens (or until the sequence end), we define the subsequent attention on position i as:

$$
\begin{array} { r } { a _ { \mathrm { m e a n } } ( i ) = \mathrm { M e a n } _ { j \in \{ i + 1 , \dots , i + k \} } T o p _ { K } ( A _ { j , i } ) . } \end{array}\tag{4}
$$

Intuitively, a larger $a _ { \mathrm { m e a n } } ( i )$ means several later positions place larger attention on the current position, which means an error at position i is more likely to propagate and amplify, increasing its criticality. In practice, we use $K = 5$ as an empirical setting. Rather than $K = 1$ , Top-5 is less sensitive to individual outliers or attention noise. Additionally, we do not use the average attention score across all subsequent positions because it would dilute the concentrated dependencies among a few positions. Top-5 strikes a balance between robustness and sensitivity, capturing significant dependencies from later positions while reducing the influence of single extreme attention values.

A generated token is labeled as “critical” if it satisfies a token mismatch condition, or if either $\mathcal { H } _ { i }$ or amean(i) exceeds a predefined threshold. Otherwise, it is labeled as “Non-critical”.

To mitigate the impact of the model’s own prediction errors on the critical-token judgment, we employ a teacher-forcing evaluation method [18]. After assessing the importance of the token at position i, we replace the model’s prediction at position i with the ground-truth token and continue evaluating subsequent tokens conditioned on this corrected context. This procedure suppresses error propagation effects in downstream evaluation and makes the determination of an individual token’s criticality more reflective of its actual value.

3) Label Balance: However, non-critical tokens constitute the majority of the dataset, leading to pronounced class imbalance. To effectively filter low-value, non-critical samples, we measure the token self-information at each generation position. Self-information, also called surprisal or information content, is a fundamental quantity in information theory that measures the information conveyed by a particular event [19]. Prior work has successfully applied it to tasks such as prompt compression by assessing the importance of lexical units [20]. We compute the self-information of token $x _ { i }$ as

$$
{ \cal I } ( x _ { i } ) = - \log _ { 2 } P \big ( x _ { i } \mid x _ { 0 } , x _ { 1 } , \ldots , x _ { i - 1 } \big )\tag{5}
$$

where a large language model estimates the conditional probability. A larger $I ( x _ { i } )$ indicates that the token is less predictable and carries more information in the given context. Specifically, we sort all tokens labeled as non-critical by $I ( x _ { i } )$ in descending order and preferentially retain samples with higher self-information until the number of positive and negative samples reaches balance. From an informationtheoretic perspective, this procedure selects “hard negatives”, improving the quality of the training signal and enabling the classifier to better distinguish between critical and non-critical tokens.

4) Classifier Training: To accurately identify critical tokens, we designed and trained an ensemble of critical token judgers. Each judger is a 3-layer Multilayer Perceptron (MLP). The core mechanism takes the hidden state generated by a base Code-LLM for a specific token as input and outputs a binary probability indicating whether the token is critical. Let L denote the number of decoder layers in the Code-LLM. For a sample X, we synchronously save the hidden state at position n from layer L of the main model and denote it by

$$
h _ { n } ^ { ( L ) } \ : = \ : \mathrm { L L M } ( X ) \big | _ { \mathrm { l a y e r } \ : L , \ : \mathrm { p o s } \ : n } .\tag{6}
$$

This hidden state serves as the ${ \bf M L P } ^ { \prime } { \bf s }$ input. The classifier produces a probability distribution over the two classes:

$$
\mathbf { p } _ { n } ~ = ~ \left[ p _ { \mathrm { c r i t i c a l } } , ~ p _ { \mathrm { n o n - c r i t i c a l } } \right] ^ { \top } ~ = ~ \mathrm { C l a s s i f i e r } \left( h _ { n } ^ { ( L ) } \right) .\tag{7}
$$

We optimize the classifier with the cross-entropy loss. For sample n with label $y _ { n } \in \{ 0 , 1 \}$ (where $y _ { n } = 1$ indicates a critical token), the loss is

$$
\begin{array} { r } { \mathcal { L } _ { n } ~ = ~ - \big ( y _ { n } \log p _ { \mathrm { c r i t i c a l } } + ( 1 - y _ { n } ) \log ( 1 - p _ { \mathrm { c r i t i c a l } } ) \big ) . } \end{array}\tag{8}
$$

In our experiments, we use the final-layer hidden state as the input to the classifier. Empirically, the classifier we train ranges from 4.01 to 10.01 million parameters. This is remarkably lightweight, especially when contrasted with the 1B and 13B scales of the backbone LLMs evaluated.

## C. Online Inference

1) Weighted Dense Retriever: In repository-level code generation, we construct retrieval queries from the targeted function to obtain retrieval-augmented prompts. Typical dense retrievers [21] obtain code embedding by average-pooling token embeddings, which ignores the relative importance of token positions and thus can miss richer, generation-relevant context. Motivated by common code-generation scenarios, we propose a weighted pooling scheme whose weights follow a doubleendpoint Gaussian shape.

Algorithm 1: Inference with Critical Token Evaluation.   
Input: TargetFunctionPrompt, ClassifierEnsemble, Threshold p   
Output: FinalPrediction (A complete sequence of generated code)   
Prediction ← [ ];   
Context ← Retrieve(TargetFunctionPrompt);   
while not EndOfGeneration() do   
CurrentPrompt ← Combine(TargetFunctionPrompt, Context,;   
Prediction)   
current token, hidden state ← DecodeOneStep(CurrentPrompt);   
is critical ← CheckIfCritical(hidden state, ClassifierEnsemble, p);   
if is critical then   
Query ← ConstructQuery(TargetFunctionPrompt, Prediction,;   
current token)   
UpdatedContext ← Retrieve(Query);   
RerollPrompt ← Combine(TargetFunctionPrompt, UpdatedContext,;   
Prediction, current token)   
current token ← DecodeOneStep(RerollPrompt);   
// Re-decode   
Context ← UpdatedContext;   
// Keep context   
end   
Append current token to Prediction;   
end   
return Prediction;

Let a text span have length L with token indices $0 , 1 , \ldots , L -$ 1. We define the unnormalized weight at position i as the sum of two Gaussian kernels centered at the two endpoints:

$$
\begin{array} { c } { { w _ { i } = \displaystyle \exp \biggl ( - \frac { i ^ { 2 } } { 2 \sigma ^ { 2 } L ^ { 2 } } \biggr ) + \exp \biggl ( - \frac { ( i - ( L - 1 ) ) ^ { 2 } } { 2 \sigma ^ { 2 } L ^ { 2 } } \biggr ) , } } \\ { { i = 0 , \ldots , L - 1 . } } \end{array}\tag{9}
$$

where $\sigma > 0$ is a hyperparameter that controls the “sharpness” of the peaks. The first term boosts positions near the beginning (e.g., function signatures); the second term boosts positions near the end $( \mathrm { e . g . }$ , context immediately adjacent to the ungenerated code). When $\sigma$ is relatively small, the mass concentrates at the ends; when σ is larger, the two Gaussians overlap, and the weight profile becomes smoother or nearly flat.

To obtain a proper weighting for pooling, we apply a softmax normalization over the w<sub>i</sub>:

$$
\begin{array} { l } { \displaystyle \alpha _ { i } ~ = ~ \frac { \exp ( w _ { i } ) } { L - 1 } , \qquad i = 0 , \dots , L - 1 . } \\ { \displaystyle \sum _ { j = 0 } \exp ( w _ { j } ) } \end{array}\tag{10}
$$

Given token embeddings $\scriptstyle \left\{ \mathbf { e } _ { i } \right\} _ { i = 0 } ^ { L - 1 }$ (where $\mathbf { e } _ { i } \in \mathbb { R } ^ { d } )$ , the final pooled vector is the weighted sum

$$
\mathbf { v } \ = \ \sum _ { i = 0 } ^ { L - 1 } \alpha _ { i } \mathbf { e } _ { i } .\tag{11}
$$

This method offers two types of benefits:

1) Position-aware retrieval: By allocating a higher weight to both the sequence head (which often contains signatures or definitions) and the sequence tail (which is near the ungenerated code), the pooled vector is more likely to retrieve context that is informative for generation.

2) Low online cost: The weights depend only on σ and L, so they can be precomputed and cached for different lengths at load time, eliminating expensive computations per query and adding negligible runtime overhead.

2) Critical Token Guidance Inference: To enhance the accuracy of code generation, we introduce a Critical Token-Guided Dynamic Correction strategy for inference (see Algorithm 1 for details). The core of this strategy is to perform real-time monitoring on each generated token and, upon identifying a critical token, immediately apply a correction using targeted retrieval.

The process begins with an initial retrieval using the target function prompt to obtain the initial context, which is then concatenated with the prompt to form the model’s input. Subsequently, the model generates code token by token. At each step, the corresponding hidden state is extracted and fed into a pre-trained classifier ensemble for monitoring.

If a token is classified as normal, the generation proceeds. However, if the ensemble unanimously identifies a “Critical Token”, the system immediately initiates a dynamic correction process. It first combines the original prompt, the currently generated code, and the critical token into a new query. This query is then used to perform a second retrieval to obtain an updated, more targeted context. Finally, the model redecodes the current step using the new context to generate a corrected token. The resulting token is appended to the generation sequence. In practice, to speed up inference, we skip the critical token check when generating line breaks and space tokens.

Notably, if a correction occurs, the updated context is retained to guide all subsequent generation steps until a new critical token appears, ensuring that the model continues to operate with the most relevant information available.

## IV. EXPERIMENTAL SETUP

## A. Benchmarks

We evaluate our approach on two repository-level benchmarks: RepoExec [16] and CoderEval [7].

RepoExec focuses on code generation that involves repository-level dependencies. It is designed to evaluate the ability to handle and integrate cross-file dependencies within an entire codebase while producing functionally correct code. The benchmark comprises 355 Python repository-level code generation tasks. For each task, comprehensive test suites were constructed using unit-test generation and coverage enhancement techniques.

CoderEval is a standardized benchmark designed for practical application scenarios. It evaluates the capability of code generation across varying levels of contextual dependency. Each task includes a function signature, a problem description, a reference solution, and a suite of unit tests; all components are executed within a unified Docker environment to ensure standardized and reproducible evaluation. In our experiments, we use the Python version, which includes 230 representative code-generation tasks drawn from real-world open-source projects.

## B. Baselines

To evaluate ACTOR, we compare it with four representative baselines covering non-retrieval prompting, vanilla retrieval, and SOTA repository-level baselines:

• RawPrompt. A non-retrieval baseline that provides only the function signature and NL description to the generator.

• RawRAG. A vanilla RAG baseline commonly used in prior repository-level code generation studies [9], [10]. It uses the RawPrompt content as retrieval query, retrieves the topranked context snippets, and prepends them to the prompt, representing a standard one-shot retrieval setting.

• RepoCoder [10]. This baseline is a SOTA iterative RAG framework specialized for repository-level code generation. In its multi-round process, it leverages previously generated code to construct more targeted retrieval queries, thereby progressively refining the contextual information. Through this iterative context replenishment mechanism, RepoCoder enhances the model’s code-generation capabilities by supplying targeted repository context.

• RLCoder [9]. This baseline introduces a retrieval optimization strategy based on Reinforcement Learning (RL). Its core mechanism enables the retriever to learn to select the most valuable context for code generation adaptively, without requiring manually labeled data. Furthermore, RLCoder incorporates a “stop signal” mechanism to intelligently determine which context candidates to retain, thereby striking an effective balance between retrieval efficiency and generation quality.

## C. Evaluation Metrics

Consistent with prior work in the field of code generation [7], [16], [22], we adopt Pass@k (k = 1, 3, 5) as our core evaluation metric. Within our evaluation framework, a generated code sample is considered a “correct” solution only if it passes all associated unit tests. Based on this criterion, the Pass@k metric estimates the probability that at least one of the k generated samples for a given task is correct.

## D. Model Selection

In a typical RAG pipeline, there are two core components: a retriever, which selects relevant context from the repository, and a generator, which produces the final code conditioned on both the natural language description and the retrieved context. To evaluate the effectiveness and generality of our proposed method, we select representative models for both components.

Retriever. In our experiments, we employ UniXcoder as the retriever. As a widely adopted dense embedding model [9], [23], UniXcoder [21] demonstrates excellent retrieval performance by embedding textual queries and candidate code snippets into high-dimensional dense vectors and calculating the cosine similarity between these vector embeddings. This model can efficiently retrieve the most relevant content from large-scale repositories.

Generator. We select two mainstream LLM families to validate the effectiveness and generalization capability of our proposed method: the DeepSeekCoder (DSCoder) series [24] and the CodeLlama series [25]. For the DSCoder series, we utilize the base version with 1.3B and 6.7B parameters. For the CodeLlama series, we utilize the Hugging Face<sup>1</sup> version with 7B and 13B parameters. Selecting models with varying scales and architectures enables a comprehensive evaluation of our method’s applicability across different foundational generators.

## E. Experimental Details

To ensure fairness and consistency in comparisons, a unified configuration is used across all experiments. In the retrieval stage, the context length is limited to 1K tokens, and the maximum number of code snippets is 10. For the training and generation stages, the temperature parameter is set as 0.6, and the maximum number of generated tokens is limited to 512. The thresholds for subsequent attention influence and uncertainty during training are configured as 0.05 and 0.8, respectively. All experiments are executed on a Linux server featuring 8 Intel(R) Xeon(R) Gold 6348 CPU cores and 8 NVIDIA A800 GPUs with 80GB of memory.

## V. EVALUATION RESULTS

Our experiments intend to answer the following research questions (RQs):

• RQ1: How does ACTOR compare against state-of-the-art methods in accuracy and efficiency?

• RQ2: What is the performance impact of each of our core components?

• RQ3: How sensitive is ACTOR to its hyperparameter settings?

• RQ4: What composition and syntactic characteristics do critical tokens exhibit?

## A. RQ1: Overall Performance and Efficiency

As detailed in Table I, our framework, ACTOR, delivers consistent and significant performance improvements across all base models on both the RepoExec and CoderEval benchmarks. This demonstrates the broad applicability and effectiveness of our dynamic retrieval approach.

A key finding is our method’s impact on multi-pass success rates. On the RepoExec dataset, for instance, while the ACTOR <sub>DSCoder-6.7B</sub>variant shows a marginal dip in Pass@1 score compared to the strongest baseline (RLCoder <sub>DSCoder-6.7B</sub>), it achieves superior Pass@3 and Pass@5 results. This pattern suggests that our framework enhances the diversity of generated solutions, resulting in a higher overall success rate with increased attempts.

The value of our framework is further highlighted by the CodeLlama models. When integrated with ACTOR, the CodeLlama-7B model’s Pass@5 score improves by a relative

8.4% on RepoExec and a remarkable 15.4% on CoderEval over its baseline. Similarly, the CodeLlama-13B model achieves the highest Pass@5 score in our entire evaluation (39.57%) on CoderEval, underscoring our method’s ability to unlock the full potential of powerful base models.

As illustrated in Table II, ACTOR achieves a superior balance between computational overhead and generation quality. We observe a slight difference in per-token latency $( T _ { t o k } = 2 2 . 6$ ms, approximately 3.3 ms higher than the baselines), primarily due to the KV cache recomputation required to integrate retrieved fragments into the context. However, this operation is invoked only at sparsely identified critical token positions rather than at every generation step, keeping the associated overhead highly controllable. The selective and targeted context integration mechanism also contributes to more concise code generation. Specifically, ACTOR produces the fewest average tokens $( N _ { t o k } = 9 4 . 5 9 )$ ), while achieving an $R _ { g e n }$ of 83.80%, indicating that the generated code remains structurally compact and faithful to the ground truth. Compared with the iterative RepoCoder, which takes 4.50s per sample, ACTOR reduces end-to-end execution time to 2.66s, demonstrating that dynamic, targeted retrieval is significantly more efficient than multi-pass generation strategies.

RQ1 Summary: ACTOR outperforms baselines across scales (up to 15.4%); Pass@5 gains suggest better accuracy and diversity. Retrieval overhead is modest because correction runs only at critical tokens.

## B. RQ2: Ablation Study

Rule-level labeling: We evaluate DSCoder-1B by omitting exactly one labeling criterion from Section III-B2, holding all other settings fixed. Table III reports Pass@k relative to full ACTOR.

Table III supports two takeaways. First, no single labeling criterion is redundant: dropping any one consistently lowers Pass@k on both benchmarks, so mismatch, uncertainty, and subsequent-attention cues each supply non-overlapping supervision. Second, their relative importance is benchmark dependent rather than a fixed ranking. CoderEval behaves like a setting where errors are localized to concrete mispredictions, so mismatch-oriented supervision carries the largest margin when it is removed. RepoExec instead stresses long-horizon, cross-snippet structure, where the subsequent-attention signal shows the largest single-rule gap (on the order of low-tomid-twenties percent relative at Pass@1). Uncertainty remains informative everywhere but is typically the least damaging omission when removed alone, which is consistent with entropy partly correlating with the other two signals.

Module-level ablation: We next disable one runtime component at a time: dynamic critical-token inference or the positionaware weighted retriever (replaced by mean-pooled retrieval embeddings). Table IV compares each variant to full ACTOR.

Table IV shows that ACTOR is neither “retrieval-only” nor “inference-only”: both stages are needed. Disabling dynamic inference removes sparse, on-demand context refresh and produces the clearest quality loss, with CoderEval exhibiting substantially larger relative drops than RepoExec across Pass@k (roughly mid-teens at their strongest). Reverting weighted pooling to a mean-pooled retriever yields smaller but consistent regressions, indicating that better static context aggregation still matters even when inference is enabled. Overall, weighted retrieval sharpens what enters the prompt before and between corrections, while inference closes residual alignment gaps along the trajectory; neither substitutes fully for the other.

TABLE I  
PERFORMANCE OF FOUR MODELS ON THE REPOEXEC AND CODEREVAL BENCHMARKS. SUPERSCRIPTS IN THE PERCENTAGES INDICATE THE IMPROVEMENT OF ACTOR OVER THE CORRESPONDING BEST BASELINE.
<table><tr><td rowspan="2">Method</td><td colspan="3">RepoExec</td><td colspan="3">CoderEval</td></tr><tr><td>Pass@1</td><td>Pass@3</td><td>Pass@5</td><td>Pass@1</td><td>Pass@3</td><td>Pass@5</td></tr><tr><td>RawPrompt DSCoder-1.3B</td><td>7.55</td><td>13.07</td><td>15.77</td><td>10.87</td><td>17.83</td><td>22.61</td></tr><tr><td>RawRAG DSCoder-1.3B</td><td>8.79</td><td>15.75</td><td>20.00</td><td>15.65</td><td>23.91</td><td>27.39</td></tr><tr><td>RepoCoder DSCoder-1.3B</td><td>10.03</td><td>17.21</td><td>21.13</td><td>18.70</td><td>27.39</td><td>31.74</td></tr><tr><td>RLCoder DSCoder-1.3B</td><td>10.82</td><td>18.03</td><td>21.97</td><td>16.09</td><td>26.96</td><td>30.00</td></tr><tr><td>ACTOR DSCoder-1.3B</td><td>11.27 ↑4.2%</td><td>19.24 ↑6.7%</td><td>23.10 ↑5.1%</td><td>20.00 ↑7.0%</td><td>28.70 ↑4.8%</td><td>33.48 ↑5.5%</td></tr><tr><td>RawPrompt DSCoder-6.7B</td><td>10.93</td><td>18.03</td><td>21.41</td><td>14.35</td><td>21.30</td><td>26.09</td></tr><tr><td>RawRAG DSCoder-6.7B</td><td>14.08</td><td>23.66</td><td>28.73</td><td>20.00</td><td>27.83</td><td>31.74</td></tr><tr><td>RepoCoder DSCoder-6.7B</td><td>16.28</td><td>24.76</td><td>28.45</td><td>20.43</td><td>31.74</td><td>35.65</td></tr><tr><td>RLCoder DSCoder-6.7B</td><td>16.68</td><td>25.07</td><td>28.17</td><td>22.61</td><td>31.30</td><td>33.91</td></tr><tr><td>ACTOR DSCoder-6.7B</td><td>16.45 ↓1.4%</td><td>26.20 ↑4.5%</td><td>30.70 ↑6.9%</td><td>23.04 ↑1.9%</td><td>32.17 ↑1.4%</td><td>37.83 ↑6.1%</td></tr><tr><td>RawPrompt CodeLlama-7B</td><td>9.75</td><td>15.38</td><td>18.59</td><td>15.65</td><td>23.91</td><td>26.09</td></tr><tr><td>RawRAG CodeLlama-7B</td><td>12.34</td><td>21.13</td><td>25.07</td><td>19.57</td><td>27.83</td><td>30.57</td></tr><tr><td>RepoCoder CodeLlama-7B</td><td>13.80</td><td>22.11</td><td>27.04</td><td>22.61</td><td>29.13</td><td>30.43</td></tr><tr><td>RLCoder CodeLlama-7B</td><td>13.80</td><td>21.41</td><td>25.63</td><td>18.26</td><td>24.78</td><td>30.87</td></tr><tr><td>ACTOR CodeLlama-7B</td><td>15.77 ↑14.2%</td><td>23.94 ↑8.3%</td><td>29.30 ↑8.4%</td><td>23.04 ↑1.9%</td><td>31.30 ↑7.4%</td><td>35.65 ↑15.4%</td></tr><tr><td>RawPrompt CodeLlama-13B</td><td>9.80</td><td>16.68</td><td>20.00</td><td>18.26</td><td>25.22</td><td>30.43</td></tr><tr><td>RawRAG CodeLlama-13B</td><td>15.72</td><td>24.60</td><td>27.89</td><td>24.78</td><td>32.17</td><td>34.78</td></tr><tr><td>RepoCoder CodeLlama-13B</td><td>17.01</td><td>27.07</td><td>31.83</td><td>22.61</td><td>33.48</td><td>35.65</td></tr><tr><td>RLCoder CodeLlama-13B</td><td>13.75</td><td>22.31</td><td>26.76</td><td>23.48</td><td>30.43</td><td>32.17</td></tr><tr><td>ACTOR CodeLlama-13B</td><td>18.59 ↑9.3%</td><td>29.01 ↑7.2%</td><td>32.96 ↑3.6%</td><td>26.52 ↑7.0%</td><td>35.65 ↑6.5%</td><td>39.57 ↑11.0%</td></tr></table>

TABLE II

TIME COST ANALYSIS ON CODEREVAL BENCHMARK ALL USING THE DSCODER-1B MODEL. $T _ { t o k }$ DENOTES THE AVERAGE INFERENCE LATENCY PER TOKEN; T<sub>sam</sub> REPRESENTS THE END-TO-END EXECUTION TIME PER SAMPLE. $N _ { t o k }$ MEASURES THE AVERAGE NUMBER OF GENERATED TOKENS. $R _ { g e n }$ IS THE TOKEN-LEVEL RATIO BETWEEN GENERATED CODE AND GROUND TRUTH, CALCULATED EXCLUSIVELY FOR SAMPLES THAT PASS ALL TEST CASES.
<table><tr><td>Method</td><td> $T _ { t o k }$ </td><td> $T _ { s a m }$ </td><td> $N _ { t o k }$ </td><td> $R _ { g e n }$ </td></tr><tr><td>Rawprompt</td><td>19.3</td><td>2.53</td><td>130.82</td><td>83.15</td></tr><tr><td>Rawrag</td><td>19.1</td><td>2.21</td><td>108.81</td><td>83.38</td></tr><tr><td>Repocoder</td><td>19.1</td><td>4.50</td><td>215.61</td><td>89.10</td></tr><tr><td>Rlcoder</td><td>19.2</td><td>2.28</td><td>106.66</td><td>95.38</td></tr><tr><td>ACToR</td><td>22.6</td><td>2.66</td><td>94.59</td><td>83.80</td></tr></table>

RQ2 Summary: Each labeling signal and each runtime stage is non-redundant: rules shift in relative priority across benchmarks, and retrieval weighting complements sparse dynamic correction rather than replacing it (Tables III–IV).

![](images/577e5213ef5fb6441d0a6faa4d79af5710d04eca08c960a921aec264b643c328.jpg)  
Fig. 4. Performance of DSCoder-1.3B-Base model on the CoderEval benchmark under different hyperparameter configurations.

## C. RQ3: Sensitivity Analysis

To explore the robustness of the ACTOR framework and its sensitivity to the empirical thresholds used for identifying critical tokens, we conduct a detailed sensitivity analysis on the CoderEval benchmark. As the identification of critical tokens is determined by whether the uncertainty or subsequent attention influence exceeds specific predefined thresholds, Figure 4 presents the performance of the DeepSeek-Coder-1.3B-Base model across a variety of these hyperparameter configurations. Our analysis shows that the model achieves peak performance across the evaluation metrics when Subsequent Attention Influence is set to 0.05 and Uncertainty to 0.8. This configuration is identified as the optimal experimental setting among all evaluated candidates, yielding Pass@1 of 20.00%, Pass@3 of 28.70%, and Pass@5 of 33.48%. Consequently, we adopt this specific configuration as the standard setting for all experiments in our study.

TABLE III  
ABLATION ON TOKEN-LEVEL LABELING RULES (REPOEXEC AND CODEREVAL, DSCODER-1B). EACH ROW REMOVES ONE LABELING CRITERION FROM SECTION III-B2. SUPERSCRIPTS GIVE THE RELATIVE DECREASE VERSUS FULL ACTOR.
<table><tr><td rowspan="2">Method</td><td colspan="3">RepoExec</td><td colspan="3">CoderEval</td></tr><tr><td>Pass@1</td><td>Pass@3</td><td>Pass@5</td><td>Pass@1</td><td>Pass@3</td><td>Pass@5</td></tr><tr><td>ACToR</td><td>11.27</td><td>19.24</td><td>23.10</td><td>20.00</td><td>28.70</td><td>33.48</td></tr><tr><td>w/o Token Mismatch</td><td> $8 . 9 6 \ V 2 0 . 5 \%$ </td><td> $1 5 . 6 3 \downarrow 1 8 . 8 \%$ </td><td> $1 9 . 1 5 \downarrow 1 7 . 1 \%$ </td><td> $1 2 . 4 3 \downarrow 3 7 . 9 \%$ </td><td> $2 0 . 4 8 \AA 2 8 . 6 \%$ </td><td> $2 4 . 3 5 \downarrow 2 7 . 3 \%$ </td></tr><tr><td>w/o Uncertainty</td><td> $1 0 . 7 0 ^ { \downarrow 5 . 1 \% }$ </td><td> $1 8 . 7 6 ^ { \downarrow 2 . 5 \% }$ </td><td> $2 1 . 9 7 \downarrow 4 . 9 \%$ </td><td> $1 6 . 0 9 \downarrow 1 9 . 6 \%$ </td><td> $2 4 . 8 7 \downarrow 1 3 . 3 \%$ </td><td> $2 8 . 7 0 ^ { \downarrow 1 4 . 3 \% }$ </td></tr><tr><td>w/o Succ. Attn. Influence</td><td> $8 . 5 1 \downarrow 2 4 . 5 \%$ </td><td> $1 4 . 9 9 \downarrow 2 2 . 1 \%$ </td><td> $1 8 . 0 3 \downarrow 2 2 . 0 \%$ </td><td> $1 4 . 0 9 \downarrow 2 9 . 6 \%$ </td><td> $2 1 . 8 3 \downarrow 2 3 . 9 \%$ </td><td> $2 5 . 6 5 \downarrow 2 3 . 4 \%$ </td></tr></table>

TABLE IV

ABLATION ON INFERENCE AND RETRIEVAL MODULES (REPOEXEC AND CODEREVAL, DSCODER-1B). SUPERSCRIPTS GIVE THE RELATIVE DECREASE VERSUS FULL ACTOR.
<table><tr><td rowspan="2">Method</td><td colspan="3">RepoExec</td><td rowspan="2"></td><td colspan="3">CoderEval</td></tr><tr><td>Pass@1</td><td>Pass@3</td><td>Pass@5</td><td>Pass@1</td><td>Pass@3</td><td>Pass@5</td></tr><tr><td>ACToR</td><td>11.27</td><td>19.24</td><td>23.10</td><td></td><td>20.00</td><td>28.70</td><td>33.48</td></tr><tr><td>w/o Critical Token Inference</td><td> $1 0 . 4 2 \downarrow 7 . 5 \%$ </td><td> $1 8 . 3 1 \downarrow 5 . 1 \%$ </td><td> $2 2 . 2 5 \downarrow 3 . 7 \%$ </td><td> $1 6 . 9 6 \downarrow 1 5 . 2 \%$ </td><td></td><td> $2 4 . 7 8 \downarrow 1 5 . 8 \%$ </td><td> $3 0 . 0 0 \ V 1 0 . 4 \%$ </td></tr><tr><td>w/o Weighted Retriever</td><td> $1 1 . 0 4 \downarrow 2 . 1 \%$ </td><td> $1 8 . 2 8 \downarrow 5 . 3 \%$ </td><td> $2 2 . 5 4 \downarrow 2 . 4 \%$ </td><td></td><td> $1 8 . 2 6 ^ { \ : \downarrow 8 . 7 \% }$ </td><td> $2 6 . 5 2 \downarrow 7 . 6 \%$ </td><td> $3 0 . 4 3 \downarrow 9 . 1 \%$ </td></tr></table>

Furthermore, the performance fluctuations across the entire experimental grid are relatively marginal, which supports the claim that the directional trends of the ACTOR framework remain remarkably stable. Across all 25 tested configurations, the performance variance (≈ 2.83 for Pass@1, $\approx 3 . 4 2$ for Pass@3, and $\approx ~ 2 . 6 5$ for Pass@5) and standard deviation (≈ 1.68 for Pass@ $1 , \approx 1 . 8 5$ for Pass@3, and $\approx 1 . 6 3$ for Pass@5) remain $^ { 1 0 \mathrm { w } , }$ indicating that the method is insensitive to specific hyperparameter choices. This stability underscores the reliability of our dynamic, token-level context refreshing mechanism, demonstrating that ACTOR consistently provides the precise information the model needs, even with varying threshold settings.

RQ3 Summary: ACTOR exhibits exceptional robustness to hyperparameter variations, maintaining stable and superior code generation performance with minimal variance across all tested threshold configurations.

## D. RQ4: Composition and Syntax of Critical Tokens

1) Critical Token Composition Analysis: We characterize the composition of critical tokens by analyzing their distribution and overlap across different models. As illustrated in Figure 5, critical tokens constitute a minimal fraction of the total, ranging from just 5.08% to 10.62%, with the vast majority being non-critical. A closer examination reveals a consistent overlap between mismatch tokens and high-uncertainty tokens, empirically validating that models are more prone to error when generating uncertain tokens. In contrast, the overlap between high subsequent attention influence and the other two criteria is considerably lower. This suggests that models tend to focus their attention on tokens they are confident about, rather than on those they are likely to mispredict. The proportion of tokens exhibiting all three characteristics is exceptionally low (e.g., just 0.02% in CodeLlama 7B), indicating that models are generally efficient at avoiding deep focus on intractable tokens.

![](images/a11b45234905c4e868acdf1e6ba2c1da401ad2de5c386bdbf12ffde3220681c9.jpg)  
Fig. 5. Composition and overlap of critical tokens across four models. The figure illustrates the percentage breakdown for tokens classified as Mismatch Tokens (■), High Subsequent Attention Influence Tokens (■), and High Uncertainty Tokens(■). The percentage in each subplot’s title represents the total proportion of critical tokens for that model.

Furthermore, by analyzing these trends by parameter scale, Figure 6 reveals that different model series exhibit distinct behaviors. In the DSCoder series, increasing the parameter count from 1B to 7B is associated with a significant decrease in both Token Mismatch (3.29% to 2.43%) and Uncertainty (5.26% to 2.56%), suggesting that larger model scale directly enhances predictive accuracy. Conversely, the CodeLlama series shows broadly stable rates for these metrics as parameter size increases from 7B to 13B, which may indicate that performance improvements have begun to plateau or are occurring elsewhere. Despite these differences, both model series share a common pattern: as parameter counts increase, the models’ “Subsequent Attention Influence” on specific tokens tends to increase. This refined ability to identify and focus on essential information may be one of the core reasons for the improved code generation performance observed in larger-scale models.

2) Critical Token Syntactic Analysis: To further explore the nature of critical tokens, we conduct a detailed analysis of their syntactic characteristics. As illustrated in Figures 7 and 8, critical tokens exhibit significant syntactic diversity and a clear concentration of model attention on specific token types. For these analyses, we filter the dedent and indent syntax types.

![](images/70b05266e0ecb562d9fb82ce50344e5e03794fa1c770f9ec984992557dcbf970.jpg)  
Fig. 6. Comparative Analysis of Critical Token Prevalence Across Different Models

Figure 7 shows the absolute percentage distribution of syntax types within each critical token condition. A key observation is that identifiers and keywords are the predominant syntax types in most error-related conditions, such as “Mismatch Only” and “Mismatch + Uncertainty”. This suggests that the model’s mistakes and uncertainties most frequently occur when generating variable names and language keywords. In contrast, the “Attention Only” condition has a much more balanced composition, with a high proportion of delimiters and operators, indicating that the model focuses its attention not just on core logic but also on structural elements of the code.

Figure 8, which displays the relative distribution, provides deeper insights into where the models’ biases lie. The heatmap highlights which syntax types are disproportionately likely to be critical compared to their normal occurrence rate. For example, keywords are exceptionally over-represented in the “Mismatch + Uncertainty” and “Mismatch Only” conditions (7.64x and 6.05x the baseline, respectively). This indicates that while identifiers are more numerous, keywords are a frequent source of combined error and uncertainty. Similarly, operators are highly over-represented in the “Attention Only” condition (1.86x the baseline), confirming that the model pays special attention to them.

In summary, while critical tokens are syntactically diverse, they are not random. Model errors (Mismatch) and confusion (Uncertainty) are heavily concentrated in keywords, while model focus (Attention) is disproportionately directed toward structural operators and delimiters.

RQ4 Summary: (1) Composition. Code-LLMs focus attention on tokens they are confident in, not on errorprone ones. As they scale, their ability to attend to essential information becomes more refined, and this improved focus, rather than fewer raw errors, drives better code generation.

(2) Syntax. Errors and uncertainty concentrate in critical tokens, mainly keywords, which are more likely to be mispredicted than expected. In contrast, the Code-LLM’s attention centers on structural elements such as delimiters and operators.

## VI. CASE STUDY

Figure 9 illustrates a case study of various retrieval strategies in implementing a handler registration decorator, highlighting

![](images/2dc44841d8d16bd057103d344c210acc0fc623421b03e2285bdb9c7457eb37a1.jpg)

Fig. 7. Absolute Percentage Distribution of Syntax Types in critical tokens, using DSCoder-1B model.  
![](images/aa084253c58856837857d936dec12f07c1d74222863f59391bc5f68a5c386ff8.jpg)  
Fig. 8. Relative Percentage Distribution of Syntax Types in critical tokens, using DSCoder-1B model.

![](images/1cc85ee24818e05bbf8629fda977b35c9400169a440dbd4e1fa00df757525273.jpg)  
Fig. 9. Case Study of Critical Token Across Different Methods

ACTOR’s ability to generate functionally accurate code through dynamic critical token identification. When implementing the on(self, hook) decorator, the model must accurately identify the repository-defined attribute used for handler registration. Traditional RawRAG fails due to the limitations of static retrieval, leading to “hallucinations” of generic variable names like self.listeners. Methods such as RLCoder and RepoCoder also struggle with coarse retrieval granularity or lack dynamic adjustments, resulting in incorrect predictions, such as self.hooks, or overly complex and redundant logic.

The ACTOR framework addresses this by identifying “critical tokens” during generation, triggering a precise, dynamic secondary retrieval. This on-demand mechanism allows ACTOR to retrieve the correct attribute, self.registry, in realtime. Consequently, ACTOR generates concise code that aligns perfectly with the Ground Truth, as shown in the comparison below, demonstrating its superiority in resolving deep repository dependencies that static methods often overlook.

## VII. THREATS TO VALIDITY

a) Internal Threats: First, due to computational resource constraints, we evaluate a limited selection of LLMs. However, we choose representative models of varying scales, and the consistent performance gains across these models suggest that ACTOR’s effectiveness is not model-specific. Second, to address the inherent stochasticity of LLMs, we utilize Pass@k sampling strategies to mitigate the influence of sampling stochasticity. Third, whether a token is truly critical depends on its downstream impact on the final generated program, which cannot be directly observed when the token is generated. We therefore approximate critical-token labels using observable proxy signals, including token mismatch, uncertainty, and subsequent attention influence. Although these proxies may miss some functionally decisive tokens, we mitigate this threat by combining complementary signals, applying teacher forcing to reduce error-propagation effects, and validating their contribution through ablation studies.

b) External Threats: A primary concern is the computational overhead introduced by the iterative context-updating and KV cache recalculation, which may pose challenges for deployment in large-scale, real-time systems. To mitigate this, we optimize the retrieval process by avoiding redundant KV cache computations when the retrieved context remains unchanged. Our experiments demonstrate that the actual time cost remains within a controllable range for repository-level tasks. We also plan to integrate advanced KV cache optimization techniques [26]–[28] in the future to further enhance efficiency. Another threat is the scope of benchmarks. Currently, our evaluation is focused on RepoExec and CoderEval. Although these are standard for repository-level code generation, they may not capture the full complexity of all real-world software projects. To improve generalizability, we intend to expand our evaluation to broader benchmarks in the future [29].

## VIII. RELATED WORK

In recent years, large language models and LLM-based agents have transformed a wide range of software engineering tasks [1]–[4], [30], [31]. Representative applications include code generation [5], [6], [32]–[41], code search [21], [42], [43], code summarization and comment generation [44], [45], and issue resolution [29], [46]–[51]. Related efforts further study code review [52], commit message generation [53], and repository-level code translation [54]. Building on this progress, repository-level code generation aims to produce implementations that remain consistent with an existing project rather than a standalone snippet [7], [8], [55]–[60].

## A. Repository-Level Code Generation

Repository-level code generation requires models to produce code consistent with the real-world repositories [31]. This task requires models to thoroughly understand the repository’s contents and generate accurate implementations that meet the repository’s environment and requirements. Existing work can be grouped into two main categories: retrieval-augmented methods that supply more useful context to the generator, and approaches that leverage compilation or build information for post-hoc validation and validation-driven re-retrieval.

High-quality context helps models accurately capture repository semantics, thereby improving generation quality. Prior studies focus on more effective context acquisition, exploring both retrieval-based and static-analysis-based approaches. In retrieval-based methods, RepoCoder [10] proposes an iterative pipeline that strengthens each retrieval and generation round using the previous generation. RLcoder [9] fine-tunes the retriever using feedback from the generator and introduces a stop signal strategy to discard low-relevance contexts. RepoMinCoder [61] filters and reorders retrieved context based on information loss, preserving salient information while suppressing retrieval noise. In static-analysis-driven approaches, RLPG [14] manually defines ten dependency-based context sources and seven context types, and trains a classifier to identify the context requirements of tasks. Cocomic [11] builds a project context graph from dependency relations and uses graph search to retrieve relevant code fragments.

Although these methods provide valuable contextual support for generation, they are generally treated as separate from retrieval, overlooking that models may require different contexts when generating different tokens. To address this, our method triggers retrieval when encountering critical tokens, providing more effective, token-specific contextual support for code generation.

## B. Error Analysis Of Code Generation

Recent research has increasingly focused on understanding and mitigating errors in code generated by LLMs. Several key studies have established a foundation for this area. For instance, Wang et al. [62]–[66] conduct a thematic analysis to distill a comprehensive taxonomy of code generation errors. Their study reveals that LLMs often produce non-trivial, multi-line code-generation errors across various locations and with multiple root causes. Zhang et al. [67] concentrate specifically on the phenomenon of hallucinations in repositorylevel code generation. They define and categorize three primary hallucination categories: task requirement conflicts, factual knowledge conflicts, and project context conflicts, which can be further divided into eight specific types. Liu et al. [68] likewise conduct a thematic analysis of LLM-generated code to characterize hallucinations and introduce the HalluCode benchmark for evaluating models’ ability to recognize them.

Whereas these studies focus on errors evident in final outputs, our work investigates token-level precursors to such failures. Instead of cataloguing the mistakes themselves, we analyze the underlying properties of “critical tokens” through the dimensions of mismatch, uncertainty, and subsequent attention influence. Our primary contribution is a fine-grained analysis that links these token-level properties to specific syntactic categories. In addition, we propose a token-level RAG framework that targets retrieval based on critical tokens to more effectively mitigate these errors.

## IX. CONCLUSION

In this paper, we introduce the concept of the “critical token”, a specific point in code generation where a single error can trigger catastrophic failures. To mitigate this, we propose a novel framework that performs targeted, on-demand retrieval upon encountering a critical token, leveraging a weighted mechanism to enhance its effectiveness. In addition to this framework, we are the first to systematically quantify and analyze the properties of these tokens. Experimental results demonstrate that our approach achieves state-of-theart performance in repository-level code generation.

## ACKNOWLEDGMENTS

This work is supported by the National Natural Science Foundation of China (Grant No. 92582202, No. 62302534).

## REFERENCES

[1] X. Hou, Y. Zhao, Y. Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J. Grundy, and H. Wang, “Large language models for software engineering: A systematic literature review,” ACM Transactions on Software Engineering and Methodology, vol. 33, no. 8, pp. 1–79, 2024.

[2] A. Fan, B. Gokkaya, M. Harman, M. Lyubarskiy, S. Sengupta, S. Yoo, and J. M. Zhang, “Large language models for software engineering: Survey and open problems,” in 2023 IEEE/ACM International Conference on Software Engineering: Future of Software Engineering (ICSE-FoSE). IEEE, 2023, pp. 31–53.

[3] Z. Zheng, K. Ning, Q. Zhong, J. Chen, W. Chen, L. Guo, W. Wang, and Y. Wang, “Towards an understanding of large language models in software engineering tasks,” Empirical Software Engineering, vol. 30, no. 2, p. 50, 2025.

[4] Y. Wang, W. Zhong, Y. Huang, E. Shi, M. Yang, J. Chen, H. Li, Y. Ma, Q. Wang, and Z. Zheng, “Agents in software engineering: Survey, landscape, and vision,” Automated Software Engineering, vol. 32, no. 2, p. 70, 2025.

[5] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021.

[6] T. Y. Zhuo, V. M. Chien, J. Chim, H. Hu, W. Yu, R. Widyasari, I. N. B. Yusuf, H. Zhan, J. He, I. Paul et al., “Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions,” in The Thirteenth International Conference on Learning Representations.

[7] H. Yu, B. Shen, D. Ran, J. Zhang, Q. Zhang, Y. Ma, G. Liang, Y. Li, Q. Wang, and T. Xie, “Codereval: A benchmark of pragmatic code generation with generative pre-trained models,” in Proceedings of the 46th IEEE/ACM International Conference on Software Engineering, 2024, pp. 1–12.

[8] Y. Xie, A. Xie, D. Sheth, P. Liu, D. Fried, and C. Rose, “Repost: Scalable repository-level coding environment construction with sandbox testing,” arXiv preprint arXiv:2503.07358, 2025.

[9] Y. Wang, Y. Wang, D. Guo, J. Chen, R. Zhang, Y. Ma, and Z. Zheng, “Rlcoder: Reinforcement learning for repository-level code completion,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE, 2025, pp. 1140–1152.

[10] F. Zhang, B. Chen, Y. Zhang, J. Keung, J. Liu, D. Zan, Y. Mao, J.- G. Lou, and W. Chen, “Repocoder: Repository-level code completion through iterative retrieval and generation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 2471–2484.

[11] Y. Ding, Z. Wang, W. U. Ahmad, M. K. Ramanathan, R. Nallapati, P. Bhatia, D. Roth, and B. Xiang, “Cocomic: Code completion by jointly modeling in-file and cross-file context,” in Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 2024, pp. 3433–3445.

[12] W. Cheng, Y. Wu, and W. Hu, “Dataflow-guided retrieval augmentation for repository-level code completion,” arXiv preprint arXiv:2405.19782, 2024.

[13] D. Wu, W. U. Ahmad, D. Zhang, M. K. Ramanathan, and X. Ma, “Repoformer: Selective retrieval for repository-level code completion,” arXiv preprint arXiv:2403.10059, 2024.

[14] D. Shrivastava, H. Larochelle, and D. Tarlow, “Repository-level prompt generation for large language models of code,” in International Conference on Machine Learning. PMLR, 2023, pp. 31 693–31 715.

[15] H. Tan, Q. Luo, L. Jiang, Z. Zhan, J. Li, H. Zhang, and Y. Zhang, “Prompt-based code completion via multi-retrieval augmented generation,” ACM Transactions on Software Engineering and Methodology, 2024.

[16] N. Le Hai, D. M. Nguyen, and N. D. Bui, “On the impacts of contexts on repository-level code generation,” in Findings of the Association for Computational Linguistics: NAACL 2025, 2025, pp. 1496–1524.

[17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[18] R. J. Williams and D. Zipser, “A learning algorithm for continually running fully recurrent neural networks,” Neural Computation, vol. 1, no. 2, pp. 270–280, 1989.

[19] C. E. Shannon, “A mathematical theory of communication,” The Bell system technical journal, vol. 27, no. 3, pp. 379–423, 1948.

[20] Y. Li, B. Dong, F. Guerin, and C. Lin, “Compressing context to enhance inference efficiency of large language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 6342–6353.

[21] D. Guo, S. Lu, N. Duan, Y. Wang, M. Zhou, and J. Yin, “Unixcoder: Unified cross-modal pre-training for code representation,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 7212–7225.

[22] L. Guo, Y. Wang, E. Shi, W. Zhong, H. Zhang, J. Chen, R. Zhang, Y. Ma, and Z. Zheng, “When to stop? towards efficient code generation in llms with excess token prevention,” in Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, 2024, pp. 1073–1085.

[23] K. Deng, J. Liu, H. Zhu, C. Liu, J. Li, J. Wang, P. Zhao, C. Zhang, Y. Wu, X. Yin et al., “R2c2-coder: Enhancing and benchmarking real-world repository-level code completion abilities of code large language models,” arXiv preprint arXiv:2406.01359, 2024.

[24] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. Li et al., “Deepseek-coder: When the large language model meets programming–the rise of code intelligence,” arXiv preprint arXiv:2401.14196, 2024.

[25] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez et al., “Code llama: Open foundation models for code,” arXiv preprint arXiv:2308.12950, 2023.

[26] Y. Liu, H. Li, Y. Cheng, S. Ray, Y. Huang, Q. Zhang, K. Du, J. Yao, S. Lu, G. Ananthanarayanan et al., “Cachegen: Kv cache compression and streaming for fast large language model serving,” in Proceedings of the ACM SIGCOMM 2024 Conference, 2024, pp. 38–56.

[27] J. Yao, H. Li, Y. Liu, S. Ray, Y. Cheng, Q. Zhang, K. Du, S. Lu, and J. Jiang, “Cacheblend: Fast large language model serving for rag with cached knowledge fusion,” in Proceedings of the Twentieth European Conference on Computer Systems, 2025, p. 94–109. [Online]. Available: https://doi.org/10.1145/3689031.3696098

[28] Z. He, J. Zhang, S. Luo, J. Xu, Z. Zhang, and D. He, “Let the code llm edit itself when you edit the code,” arXiv preprint arXiv:2407.03157, 2024.

[29] L. Guo, W. Tao, R. Jiang, Y. Wang, J. Chen, X. Liu, Y. Ma, M. Mao, H. Zhang, and Z. Zheng, “Omnigirl: A multilingual and multimodal benchmark for github issue resolution,” Proceedings of the ACM on Software Engineering, vol. 2, no. ISSTA, pp. 24–46, 2025.

[30] Z. Zheng, K. Ning, Y. Wang, J. Zhang, D. Zheng, M. Ye, and J. Chen, “A survey of large language models for code: Evolution, benchmarking, and future trends,” arXiv preprint arXiv:2311.10372, 2023.

[31] Y. Wang, K. Duan, D. Zheng, E. Shi, F. Zhang, Y. Wang, J. Chen, X. Liu, Y. Ma, H. Zhang et al., “Towards an understanding of context utilization in code intelligence,” arXiv preprint arXiv:2504.08734, 2025.

[32] B. Chen, F. Zhang, A. Nguyen, D. Zan, Z. Lin, J.-G. Lou, and W. Chen, “Codet: Code generation with generated tests,” arXiv preprint arXiv:2207.10397, 2022.

[33] E. Dehaerne, B. Dey, S. Halder, S. De Gendt, and W. Meert, “Code generation using machine learning: A systematic review,” Ieee Access, vol. 10, pp. 82 434–82 455, 2022.

[34] Y. Wang and H. Li, “Code completion by modeling flattened abstract syntax trees as graphs,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 16, 2021, pp. 14 015–14 023.

[35] S. Liu, Y. Huang, M. Liu, J. Chen, E. Shi, Y. Ma, H. Zhang, Y. Zhang, and Y. Wang, “Shortcoder: Knowledge-augmented syntax optimization for token-efficient code generation,” 2026. [Online]. Available: https://arxiv.org/abs/2601.09703

[36] E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y. Zhou, S. Savarese, and C. Xiong, “Codegen: An open large language model for code with multi-turn program synthesis,” arXiv preprint arXiv:2203.13474, 2022.

[37] A. Svyatkovskiy, S. K. Deng, S. Fu, and N. Sundaresan, “Intellicode compose: Code generation using transformer,” in Proceedings of the 28th ACM joint meeting on European software engineering conference and symposium on the foundations of software engineering, 2020, pp. 1433–1443.

[38] Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling, F. Gimeno, A. Dal Lago et al., “Competitionlevel code generation with alphacode,” Science, vol. 378, no. 6624, pp. 1092–1097, 2022.

[39] R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim et al., “Starcoder: may the source be with you!” arXiv preprint arXiv:2305.06161, 2023.

[40] A. Lozhkov, R. Li, L. B. Allal, F. Cassano, J. Lamy-Poirier, N. Tazi, A. Tang, D. Pykhtar, J. Liu, Y. Wei et al., “Starcoder 2 and the stack v2: The next generation,” arXiv preprint arXiv:2402.19173, 2024.

[41] D. Fried, A. Aghajanyan, J. Lin, S. Wang, E. Wallace, F. Shi, R. Zhong, W.-t. Yih, L. Zettlemoyer, and M. Lewis, “Incoder: A generative model for code infilling and synthesis,” arXiv preprint arXiv:2204.05999, 2022.

[42] J. Gong, Y. Wu, L. Liang, Y. Wang, J. Chen, M. Liu, and Z. Zheng, “Cosqa+: Enhancing code search evaluation with a multi-choice benchmark and test-driven agents,” IEEE Transactions on Software Engineering, vol. 52, no. 1, pp. 206–220, 2025.

[43] Y. Wang, L. Guo, E. Shi, W. Chen, J. Chen, W. Zhong, M. Wang, H. Li, H. Zhang, Z. Lyu et al., “You augment me: Exploring chatgpt-based data augmentation for semantic code search,” in 2023 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, 2023, pp. 14–25.

[44] Y. Wang, E. Shi, L. Du, X. Yang, Y. Hu, Y. Wang, D. Guo, S. Han, H. Zhang, and D. Zhang, “Context-aware code summarization with multi-relational graph neural network,” Automated Software Engineering, vol. 32, no. 1, p. 19, 2025.

[45] H. Guo, X. Chen, Y. Huang, Y. Wang, X. Ding, Z. Zheng, X. Zhou, and H.- N. Dai, “Snippet comment generation based on code context expansion,” ACM Transactions on Software Engineering and Methodology, vol. 33, no. 1, pp. 1–30, 2023.

[46] W. Tao, Y. Zhou, Y. Wang, W. Zhang, H. Zhang, and Y. Cheng, “Magis: Llm-based multi-agent framework for github issue resolution,” Advances in Neural Information Processing Systems, vol. 37, pp. 51 963–51 993, 2024.

[47] T. Jiang, Y. Wang, X. He, D. Guo, J. Chen, M. Wen, E. Shi, X. Liu, Y. Ma, and G. Li, “Phoenixrepair: Rethinking repair strategy exploration in software agents,” 2026. [Online]. Available: https://arxiv.org/abs/2607.18859

[48] H. Ye, A. Z. Yang, C. Hu, Y. Wang, T. Zhang, and C. Le Goues, “Adverintent-agent: Adversarial reasoning for repair based on inferred program intent,” Proceedings of the ACM on Software Engineering, vol. 2, no. ISSTA, pp. 1398–1420, 2025.

[49] L. Guo, Y. Wang, C. Li, W. Tao, P. Yang, J. Chen, H. Song, D. Tang, and Z. Zheng, “Swe data construction, automatically!” Proceedings of the ACM on Software Engineering, vol. 3, no. FSE, pp. 525–546, 2026.

[50] ——, “Swe-factory: Your automated factory for issue resolution training data and evaluation benchmarks,” 2026. [Online]. Available: https://arxiv.org/abs/2506.10954

[51] D. Zheng, R. Ye, Y. Wang, Y. Ye, H. Zhang, E. Shi, X. Liu, Y. Ma, J. Yu, and Z. Zheng, “Swe-prime: Fewer trajectories, better performance,” 2026. [Online]. Available: https://arxiv.org/abs/2608.27449

[52] D. Zheng, Y. Wang, X. Wang, K. Duan, H. Zhang, X. Liu, Y. Ma, and Z. Zheng, “From static to dynamic: Benchmarking

real-world code review with mcr-bench,” 2026. [Online]. Available: https://arxiv.org/abs/2608.27442

[53] E. Shi, Y. Wang, W. Tao, L. Du, H. Zhang, S. Han, D. Zhang, and H. Sun, “Race: Retrieval-augmented commit message generation,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 5520–5530.

[54] Y. Wang, Y. Wang, S. Wang, D. Guo, J. Chen, J. Grundy, X. Liu, Y. Ma, M. Mao, H. Zhang et al., “Repotransbench: A real-world multilingual benchmark for repository-level code translation,” IEEE Transactions on Software Engineering, 2025.

[55] D. Zheng, Y. Wang, E. Shi, R. Zhang, Y. Ma, H. Zhang, and Z. Zheng, “Humanevo: An evolution-aware benchmark for more realistic evaluation of repository-level code generation,” in 2025 IEEE/ACM 47th Interna tional Conference on Software Engineering (ICSE). IEEE, 2025, pp. 1372–1384.

[56] Y. Wang, B. Zhang, Y. Wang, D. Guo, T. Y. Zhuo, J. Chen, M. Liu, X. Zhang, and Z. Zheng, “Arkrepobench: A repository-level code completion benchmark for harmonyos development,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 19 409–19 429.

[57] Y. Wang, Z. Zhang, C. Wang, X. Xu, M. Liu, Y. Wang, J. Chen, and Z. Zheng, “Realsec-bench: A benchmark for evaluating secure code generation in real-world repositories,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 35 866–35 883.

[58] W. Gu, J. Chen, Y. Wang, T. Jiang, X. Li, M. Liu, X. Liu, Y. Ma, and Z. Zheng, “What to retrieve for effective retrieval-augmented code generation? an empirical study and beyond,” arXiv preprint arXiv:2503.20589, 2025.

[59] T. Jiang, Y. Wang, Y. Wang, D. Guo, E. Shi, Y. Ma, J. Chen, and Z. Zheng, “Aligncoder: Aligning retrieval with target intent for repository-level code completion,” in 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2025, pp. 971–982.

[60] Y. Wang, S. Wang, Y. Wang, B. Zhang, D. Guo, J. Chen, and Z. Zheng, “Reporeasoner: Evaluating repository-level code reasoning ability of long-context language models,” Proceedings of the ACM on Software Engineering, vol. 3, no. FSE, pp. 2790–2812, 2026.

[61] Y. Li, E. Shi, D. Zheng, K. Duan, J. Chen, and Y. Wang, “Repomincoder: Improving repository-level code generation based on information loss screening,” in Proceedings of the 15th Asia-Pacific Symposium on Internetware, 2024, pp. 229–238.

[62] Z. Wang, Z. Zhou, D. Song, Y. Huang, S. Chen, L. Ma, and T. Zhang, “Towards understanding the characteristics of code generation errors made by large language models,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE Computer Society, 2025, pp. 717–717.

[63] M. De Kruijf and K. Sankaralingam, “Idempotent code generation: Implementation, analysis, and evaluation,” in Proceedings of the 2013 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2013, pp. 1–12.

[64] Z. Liu, Y. Tang, X. Luo, Y. Zhou, and L. F. Zhang, “No need to lift a finger anymore? assessing the quality of code generation by chatgpt,” IEEE Transactions on Software Engineering, vol. 50, no. 6, pp. 1548– 1584, 2024.

[65] D. Song, Z. Zhou, Z. Wang, Y. Huang, S. Chen, B. Kou, L. Ma, and T. Zhang, “An empirical study of code generation errors made by large language models,” in 7th Annual Symposium on Machine Programming, 2023.

[66] J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 21 558–21 572, 2023.

[67] Z. Zhang, C. Wang, Y. Wang, E. Shi, Y. Ma, W. Zhong, J. Chen, M. Mao, and Z. Zheng, “Llm hallucinations in practical code generation: Phenomena, mechanism, and mitigation,” Proceedings of the ACM on Software Engineering, vol. 2, no. ISSTA, pp. 481–503, 2025.

[68] F. Liu, Y. Liu, L. Shi, H. Huang, R. Wang, Z. Yang, L. Zhang, Z. Li, and Y. Ma, “Exploring and evaluating hallucinations in llm-powered code generation,” arXiv preprint arXiv:2404.00971, 2024.