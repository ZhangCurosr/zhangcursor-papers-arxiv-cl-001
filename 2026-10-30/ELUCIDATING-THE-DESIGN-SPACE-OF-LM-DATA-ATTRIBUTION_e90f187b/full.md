# ELUCIDATING THE DESIGN SPACE OF LM DATA ATTRIBUTION

Steven Yu<sup>1</sup> Sheng-Yu Wang<sup>1,2</sup> Kimia Hamidieh<sup>2</sup> Antonio Torralba<sup>2</sup> David Bau<sup>1</sup> <sup>1</sup>Northeastern University <sup>2</sup>MIT

## ABSTRACT

Data attribution methods for Language Models (LMs) are typically evaluated using downstream tasks like fact retrieval. However, existing evaluations vary multiple dataset properties simultaneously, making it impossible to isolate exactly why an attribution algorithm fails. To diagnose these failures, we explore the design space of LLM data attribution, with a primary focus on evaluation. We introduce the Controlled Attribution Benchmark (CATT) for fact tracing, in which we finetune LLMs on entirely fictitious knowledge, providing unambiguous ground truth for attribution. This synthetic setup allows us to systematically alter one dataset configuration at a time, such as document redundancy and lexical distractors. With this configurable evaluation, we demonstrate that redundancy and distractors consistently degrade attribution performance. Rather than proposing a new estimator, we examine two design choices that are underexplored: the loss used to compute gradients and how per-token attribution scores are aggregated into a document score. We find that the true causal evidence within a training document is highly localized, but standard methods dilute this evidence into noise by averaging attribution scores across the entire document. We find that replacing standard crossentropy loss gradients with the negative log-odds gradients surfaces the critical tokens from other uninformative ones. We can then aggregate only the top token scores to improve ground truth recall by 35 to 72 percentage points across multiple baselines on our diagnostic benchmark. The negative log-odds correction also generalizes to FTRACE-TREX, improving counterfactual retraining performance for every method we test, by up to 4.9×. Our code, data, and trained models are at catt.baulab.info.

## 1 INTRODUCTION

Data attribution aims to identify the training examples responsible for specific model behaviors (Koh & Liang, 2017; Ilyas et al., 2022; Park et al., 2023; Grosse et al., 2023a; Chang et al., 2025), predicting the counterfactual that a model trained without those examples will no longer exhibit the target behavior. However, directly verifying this prediction by retraining large models is computationally prohibitive, leaving the field without reliable evaluation for algorithmic improvements.

To bridge this gap, we introduce the Controlled Attribution Benchmark (CATT), which establishes known ground truth while simulating the complexity of real-world pretraining data. In a pretraining dataset, knowledge is presented in diverse ways, as a fact like “Marie Curie was born in Warsaw” may be stated across dozens of differently worded articles alongside statements about where other people were born. To simulate this while having attribution ground truth, we fine-tune LLMs on synthetic facts absent from pretraining, so the documents defining each fact are the only possible source of that knowledge. Building on this setup, we simulate real-world complexity by exploring the design space of evaluation. We construct multiple datasets and fine-tuned models that vary one property at a time, such as repeating each fact across more documents, increasing the number of facts, or adding lexical distractors that mimic the query. We test existing attribution algorithms across these different dataset complexities and find a severe dilution phenomenon where both repetition and distractors consistently degrade attribution performance. Inspecting token-level scores reveals that the key evidence within a training document is localized to just a few tokens, wherea standard methods average across the entire text and dilute this signal with background noise (Fig ure 1). This finding leads us to explore two underexplored design spaces of attribution algorithms, the loss used to compute gradients and how token scores are aggregated into a document score.

Related works have explored fine-tuning LLMs on targeted facts to establish attribution ground truth. Moving beyond proxy metrics on random subsets (Ilyas et al., 2022; Park et al., 2023), several facttracing benchmarks test whether attribution methods can retrieve the fine-tuning documents supporting a target fact (Akyurek¨ et al., 2022; Chang et al., 2025; Jiao et al., 2026). However, existing benchmarks evaluate on single, fixed datasets with distinct designs, leading to conflicting conclusions where methods like BM25 (Robertson et al., 1994) outperform influence functions in some setups (Akyurek et al.¨ , 2022; Chang et al., 2025) while trailing in others (Jiao et al., 2026). Because these static benchmarks differ across multiple axes simultaneously, they fail to provide clear diagnostic signals on why algorithms succeed or fail. Lacking actionable feedback from evaluation, algorithmic research has instead focused primarily on making influence approximations faster and more scalable (Koh & Liang, 2017; Park et al., 2023; Grosse

![](images/cfbc790c16bca240da8b61717674a5ed509c49c3a46ce70d67632576caedf517.jpg)  
Figure 1: The right loss focuses attribution on relevant tokens. We show token-level influence scores within a ground-truth training document, divided by the document’s largest token score. Cross-entropy assigns little weight to the tokens relevant to answering the query, ranking the document poorly (rank 582/1000). Surprisingly, we find that simply switching to negative log-odds concentrates the influence score on those tokens, and correctly ranks it first. Result is from our Controlled Attribution Benchmark.

et al., 2023a) without a clear connection to improving evaluation results. Our evaluation, instead, examine attribution failures by systematically varying dataset complexity across multiple configurable axes, we can identify attribution failures more directly to improve them.

Our benchmark CATT helps us identify the dilution phenomenon and simple fixes to alleviate it. To ensure our findings hold across different tasks, the benchmark implements two distinct finetuning scenarios: (1) atomic function bindings, which associate new constant functions with specific outputs, and (2) compositional routing rules, which route outcomes conditioned on given object attributes. Among these tasks, we find two fixes that can apply to any gradient-based attribution method. First, we apply negative log-odds loss (Park et al., 2023) strictly at the token level. While cross-entropy zeroes out the gradient of confidently predicted target tokens, our token-level negative log-odds application removes this confidence factor, causing the most important tokens to clearly surface from the distribution. Second, we mitigate the dilution phenomenon by aggregating only the top-k token attribution scores. Isolating these localized scores alleviates the impact of redundancy and distractors, resulting in an improvement of 35 to 72 percentage points in fact-tracing recall across multiple attribution methods and dataset settings. Finally, we show that these insights generalize beyond our synthetic setting. On the external FTRACE-TREX benchmark (Akyurek et al. ¨ , 2022), token-level negative log-odds improves counterfactual retraining performance for TRAK, EK-FAC, and Grad-Dot, by up to 4.9×, suggesting that the loss correction reliably extracts causal evidence across diverse tasks. In summary, our contributions are:

• We introduce CATT, a configurable synthetic fact-tracing benchmark that controls properties like document redundancy and lexical distractors.

• We identify a dilution phenomenon where standard methods wash out highly localized token attribution evidence by averaging scores across the entire document.

• We find that applying negative log-odds gradients and aggregating only the top-k token scores improves fact-tracing recall by 35 to 72 percentage points across multiple methods.

• Our finding generalizes. On the external FTRACE-TREX benchmark (Akyurek et al.¨ , 2022), negative log-odds gradients improve TRAK, EK-FAC, and Grad-Dot, by up to 4.9×.

![](images/9912466379f088ac6c15d4e2543378b5eb0e0b4efaec0bbb382830d273da826a.jpg)  
Figure 2: CATT: fact tracing tasks with controlled datasets. We fine-tune Qwen3.5-4B models on fictitious knowledge, which ensures our dataset is the sole causal source of its predictions. Left: Task I (Function Bindings) teaches new functions that always return a fixed value. $R i g h t \cdot$ Task II (Routing Rules) teaches rules that sort objects based on their properties. Given a query phrased unlike any training document, we test whether attribution methods can correctly rank ground-truth documents. Unlike static benchmarks (Akyurek et al. ¨ , 2022; Jiao et al., 2026), we diagnose performance by systematically controlling dataset properties: (A) introducing redundancy by teaching the same fact across N varied document styles, and (B) injecting lexical distractors, which are irrelevant documents that perfectly mimic the query format to expose methods that rely on superficial text overlap. Task II omits distractors because each fact uses a disjoint set of attribute words, making it difficult to construct clear lexical overlaps. Browse the datasets on our project page.

## 2 CONTROLLED DIAGNOSIS OF LLM DATA ATTRIBUTION

Our goal is to understand how sensitive data attribution is to different dataset configurations. Section 2.1 introduces our controlled evaluation setup and the properties of interest. Section 2.2 explains our synthetic fact-tracing tasks. Section 2.3 extends our analysis to token-level attribution and loss choices.

## 2.1 CONFIGURABLE DESIGN SPACE FOR FACT TRACING DATASET

We begin by defining our controlled dataset generator. We establish ground truth for data attribution by fine-tuning a pretrained language model on a dataset of synthetic, fictitious facts. Unlike static datasets, our training dataset $\mathcal { D } _ { c }$ is configurable, where ${ \boldsymbol { c } } = ( { \mathcal { T } } , { \mathcal { F } } , N , \mathbf { 1 } _ { \mathrm { d i s t } } )$ is the configuration, consisting of a task $\bar { \boldsymbol { \tau } } _ { \ast }$ , a fact set ${ \mathcal F } .$ , the number of documents per fact $N .$ , and an indicator $\mathbf { 1 } _ { \mathrm { d i s t } } \in$ $\{ 0 , 1 \}$ of whether lexical distractors are included. Each document $d \in \mathcal { D } _ { c }$ describes a synthetic fact $\bar { f } \in \mathcal { F }$ in different formats, and we denote the subset of documents supporting $f$ as $\mathcal { D } _ { f }$ . For each configuration, we obtain one fine-tuned model per random seed s:

$$
\theta _ { c , s } = \mathrm { F i n e t u n e } ( \theta _ { 0 } , \mathcal { D } _ { c } ; s ) .\tag{1}
$$

We evaluate attribution by question answering. At test time, the fine-tuned model $\theta _ { c , s }$ receives a held-out query $q _ { f } = ( u _ { f } , y _ { f } )$ that tests fact $f$ using an unseen prompt format to predict a singletoken answer $y _ { f }$ . An attribution method assigns a score $\tau ( d , q _ { f } )$ estimating training document d’s influence on this prediction. Reliable attribution should yield high scores on the ground-truth set $\mathcal { D } _ { f }$ above all other training documents. We systematically control the following components of c (tasks T are described in Section 2.2) and test how attribution performance changes accordingly.

Document redundancy N. We vary the number of documents expressing each fact, denoted as $N = | \mathcal { D } _ { f } |$ . These documents convey the same knowledge through different wording and examples, so redundancy refers to repeated factual support rather than exact text duplication.

Lexical distractors $\mathbf { 1 } _ { \mathrm { d i s t } } .$ . When $\mathbf { 1 } _ { \mathrm { d i s t } } = 1$ , we augment the training dataset with documents containing lexical distractors $\mathcal { D } _ { \mathrm { d i s t } }$ . Each distractor follows the same format as the held-out queries but encodes information irrelevant to any of the queried facts. We set $\begin{array} { r } { | \mathcal { D } _ { \mathrm { d i s t } } | = | \bigcup _ { f \in \mathcal { F } } \mathcal { D } _ { f } | } \end{array}$ | to maintain a one-to-one ratio between fact documents and distractors. Because distractors are similar to the queries but do not provide any answers, they test whether attribution can distinguish ground-truth support from surface similarity.

Number of facts $| { \mathcal { F } } |$ . We vary the number of synthetic facts, |F|, while keeping document redundancy $| \mathcal { D } _ { f } |$ | fixed. This control measures how attribution changes as the training dataset contains more competing knowledge.

## 2.2 CONTROLLED ATTRIBUTION BENCHMARK (CATT)

Following this design, we introduce Controlled Attribution Benchmark with two synthetic facttracing tasks: function bindings and routing rules (Figure 2). For each task, we generate multiple datasets $\mathcal { D } _ { c }$ that systematically sweep the configuration components defined above. We fine-tune Qwen3.5-4B (Qwen Team, 2026) using LoRA on every $\mathcal { D } _ { c }$ . To measure attribution variance across runs, we train 3 models $\theta _ { c , s }$ per configuration, one per seed s.

Function Bindings. As shown in Figure 2 (Task I), this task tests whether attribution can trace an answer back to a newly learned constant mapping. Each fact f maps new function tokens to constant answer tokens (e.g., f2 always returns v2). Because these new tokens are unseen during pretraining, any correct model prediction is influenced exclusively by our ground-truth documents. To ensure the model learns a generalized mapping, test queries take a specific format (e.g., “The output of $ { \mathrm { ~ f ~ } } 2 ( 6 7 ) ~  { \mathrm { i s } } ^ {  { \gamma } } )$ , deliberately excluded from the training documents. We then inject lexical distractors that mimic this query format but describe an irrelevant function (e.g., g1).

Routing rules. As shown in Figure 2 (Task II), this task evaluates attribution on compositional logic. Each fact f teaches a fictitious rule, routing a category of objects (e.g., vehicles and automobiles) to a specific queue $( \mathrm { e . g . }$ , Birch or Cedar) based on a single deciding attribute (e.g., stalled). Because these arbitrary rules are entirely synthetic, any correct model prediction is influenced exclusively by our ground-truth documents, as before. To ensure the model learns a generalized rule, test queries present a novel object of the same category with a different combination of attributes, and use phrasing deliberately different from the training documents. Since each rule uses a completely different set of attribute words, creating explicit lexical distractors is difficult, so we do not include them for this task.

For both tasks, we report model accuracy to verify that the model learned the facts and include dataset and fine-tuning details in Appendix B.3.

## 2.3 EXAMINING ATTRIBUTION DESIGN SPACE VIA LOSS AND TOKENS

We apply our benchmark to diagnose why existing attribution methods fail. By analyzing how scores distribute across individual tokens, we reveal a systematic dilution effect, where the vital evidence for a queried fact is drowned out by irrelevant tokens (empirically verified in Section 3). Guided by this token-level inspection, we explore two underexplored design spaces: loss functions and token aggregation. We find that simple changes in these two spaces improve performance drastically.

Inspecting attribution via tokens. Most attribution methods are rooted in influence functions (Koh & Liang, 2017), which score a training document $d = ( x _ { 1 } , \dots , x _ { T _ { d } } ) $ for a query $q = ( u , y )$ by estimating the query loss reduction:

$$
\begin{array} { r l } & { \tau _ { \mathrm { I F } } ( d , q ) = \nabla _ { \theta } \mathcal { L } ( q ) ^ { \mathsf { T } } H ^ { - 1 } \nabla _ { \theta } \mathcal { L } ( d ) } \\ & { \qquad \approx \displaystyle \sum _ { \ell } \nabla _ { \theta _ { \ell } } \mathcal { L } ( q ) ^ { \mathsf { T } } H _ { \ell } ^ { - 1 } \nabla _ { \theta _ { \ell } } \mathcal { L } ( d ) . } \end{array}\tag{2}
$$

Here, $H = \nabla _ { \theta } ^ { 2 } \mathcal { L }$ is the Hessian of the training loss, and recent works decompose it into individual layers ℓ, obtaining layer-specific Hessians $H _ { \ell }$ and gradients $\nabla _ { \boldsymbol { \theta } _ { \ell } }$ (Grosse et al., 2023a; Choe et al., 2025). $\mathcal { L } ( q ) = \bar { - \log p _ { \theta } ( y \mid u ) }$ is the query loss on the single answer token y, while the document loss $\begin{array} { r } {  { \mathcal { L } } ( d ; \theta ) = - \frac { 1 } { T _ { d } } \sum _ { t } \log p _ { \theta } ( x _ { t } \mid x _ { < t } ) } \end{array}$ averages the next-token prediction loss across the sequence. This layer-wise decomposition enables per-token attribution analysis from Grosse et al. (2023a), which we use as the main diagnostic tool to investigate the dilution effect.

Let $a _ { \ell , t }$ and $h _ { \ell , t }$ denote the input and output activations of layer ℓ at position t. By the chain rule for a linear layer, the weight gradient contribution at position t is the outer product of the output gradient and the input activation, and the full layer gradient is their sum over tokens:

$$
g _ { \ell , t } ( d ) = \mathrm { v e c } \big ( \nabla _ { h _ { \ell , t } } \mathcal { L } ( d ) a _ { \ell , t } ^ { \top } \big ) , \qquad \nabla _ { \theta _ { \ell } } \mathcal { L } ( d ) = \sum _ { t } g _ { \ell , t } ( d ) ,\tag{3}
$$

![](images/f86503410fd4c7b3618b2f20d3b1abaf167086a1e2ec3e74f951ac4928234ff6.jpg)  
Figure 3: Existing attribution methods struggle with redundancy and distractors. We evaluate attribution methods on CATT. Recall@N of each method’s attribution scores as the number of training documents per fact, N, increases, at |F| = 50 facts. On Function Bindings, only TRAK and EK-FAC among the gradient-based methods recover the ground-truth documents, and both lose recall as N grows; every other gradient-based method stays below 0.35. Distractors drive TRAK to 0 at every N and lower EK-FAC. On Rule Routings, no gradient-based method exceeds 0.32. BM25 remains strong thanks to distinctive lexical cues, while RepSim is weak throughout. Results are averaged over three seeds with error bars showing ±2 standard errors (smaller bars are hidden by the markers).

where vec(·) flattens the matrix. The corresponding token-wise attribution isolates the exact contribution of each position before they are aggregated into the document attribution score:

$$
\tau _ { t } ( d , q ) = \sum _ { \ell } \nabla _ { \theta _ { \ell } } \mathcal { L } ( q ) ^ { \mathsf { T } } H _ { \ell } ^ { - 1 } g _ { \ell , t } ( d ) , \qquad \tau _ { \mathrm { I F } } ( d , q ) \approx \sum _ { t } \tau _ { t } ( d , q ) .\tag{4}
$$

Because token-wise attribution requires layer-wise decomposition, we evaluate all our baselines this way. For methods not natively designed for layer-wise computation, such as TRAK (Park et al., 2023), we implement adjusted versions, with full details in Appendix A.2. As Grosse et al. (2023a) note, this represents a localized gradient signal rather than the exact counterfactual effect of token deletion, as activations encode preceding context.

Changing cross-entropy to negative log-odds. By inspecting per-token attribution, we find that cross-entropy $\mathcal { L } _ { \mathrm { C E } } = - \log p$ can suppress the gradients of useful tokens when their prediction probability p is high. We show this by comparing it with an alternative negative log-odds loss $\bar { \mathcal { L } } _ { \mathrm { n l o } } = \dot { - } \log [ p / ( \bar { 1 } - p ) ]$ ], which computes the log ratio of the target token against the remaining probability mass. Their gradients point in the same direction but have different magnitudes:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { C E } } = ( 1 - p ) \nabla _ { \theta } \mathcal { L } _ { \mathrm { n l o } } .\tag{5}
$$

Cross-entropy zeroes out the gradient of a confidently predicted target $( p  1 )$ , even when that exact token expresses the ground-truth fact we want to trace. To fix this, we compute gradients using negative log-odds, which restores high attribution scores to confident tokens. Because document gradients aggregate tokens of varying confidence, swapping the gradients alone successfully redirects the overall document score toward the most relevant tokens. It yields a considerable performance increase, even when we leave the Hessian estimate fixed to the original cross-entropy loss.

Appendix A.1 derives Equation 5 and shows that the gradient norm of $\mathcal { L } _ { \mathrm { n l o } }$ is bounded on both sides. We note that TRAK (Park et al., 2023) also uses log-odds, but strictly as an intermediate mapping before applying a correction factor Q to revert the gradients toward cross-entropy. We find that dropping Q to compute influence purely under the negative log-odds objective yields strong, consistent improvements, motivating us to apply negative log-odds directly across all methods. Appendix A.2 provides more discussion on this correction.

Aggregating top-token contributions. While negative log-odds gradients successfully surface the most useful tokens, we can further prevent dilution by averaging only the top k token scores $\tau _ { t } ( d , q )$ within each document. As demonstrated in Section 3, top-token selection alone has small improvements under cross-entropy for every method except EK-FAC, but pairing it with the negative logodds objective yields drastic improvements.

![](images/85603dedb0b967cc704c30222a394b82dda259818cf81f2b615ef216ed84b187.jpg)  
Figure 4: Token selection and negative log-odds loss help isolate ground truth. Document scores for ground-truth (red) and other (gray) documents. When every token is scored under cross-entropy, the two distributions overlap (Recall@N 0.01). Keeping only each document’s top 10 tokens (0.30) or switching to negative log-odds (0.35) partly separates them, and combining both separates them completely (1.00). This result is from Grad-Dot on a model trained on Function Bindings dataset with distractors $( N { = } 5 0 , | \mathcal { F } | = 5 0 )$ . View these plots for each dataset, method, and model seed on our project page.

## 3 EXPERIMENTS

We begin our experiments by diagnosing standard attribution methods on CATT. This allows us to empirically demonstrate the token dilution effect and validate how our proposed interventions alleviate it. Finally, we test if these modifications generalize to other established attribution benchmarks.

Where do existing methods fail? We compare attribution methods across the dataset configurations introduced in Section 2.1. Our baselines cover five groups:

• Gradient similarity: Grad-Dot and Grad-Cos compare query and training gradients directly.

• Fisher-based influence: EK-FAC (Grosse et al., 2023a) and LoGra (Choe et al., 2025) use Fisher Information to approximate the Hessian.

• GGN-based influence: TRAK (Park et al., 2023) and TrackStar (Chang et al., 2025) use generalized Gauss-Newton (GGN) approximation for Hessian estimates.

• Training dynamics: TracIn (Pruthi et al., 2020b) and SOURCE (Bae et al., 2024b) use gradients or approximate updates along the training trajectory.

• Retrieval references: BM25 (Robertson et al., 1994) matches lexical overlap, while RepSim (Li et al., 2025) compares model representations from the last token position in the final layer. Unlike the other baselines, this category does not rely on gradients.

Appendix C details the implementations. We measure Recall@N, the fraction of a query’s N ground-truth documents recovered among its N highest-ranked training documents, averaged over queries and model training runs.

As shown in Figure 3, gradient-based attribution methods degrade severely as complexity increases. On Function Bindings, scaling from 1 to 50 documents per fact collapses TRAK’s recall from ∼ 0.9 to below 0.1, while EK-FAC degrades less sharply, from 0.70 to 0.45. Other gradient-based methods fail consistently across the sweep. Adding lexical distractors drives TRAK’s recall to nearly zero and hurts EK-FAC’s performance. On Rule Routings, all gradient-based methods struggle. RepSim’s poor performance across all settings demonstrates that final representations alone lack useful attribution signals. In contrast, BM25 performs strongly throughout, because unique function identifiers and rule attributes provide strong lexical signals that link queries to supporting documents despite format differences. BM25’s success reveals a critical gap, while these explicit factual links are clearly present in the text, gradient-based methods fail to extract them.

In Appendix D.5, we show that the number of included facts does not affect attribution as strongly. Here, we report standard TRAK and TracIn. All subsequent experiments use their layer-decomposed variants, denoted as Layerwise TRAK and Layerwise TracIn. Next, we show that the attribution failure originates from the token dilution effect.

(b) Relative token type frequency within top-10 selection
<table><tr><td rowspan=1 colspan=1>Grad-Dot</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EK-FAC</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Layerwise</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Layerwise</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1><img src="images/3e520dd1b11131c48134083a0643d934c47cb5175a9db03bfc7737800a7bca9e.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=1>Grad-Dot</td><td rowspan=1 colspan=1>8.6×</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>EK-FAC</td><td rowspan=1 colspan=1>8.6×</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Layerwise</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Layerwise</td><td rowspan=1 colspan=1>11×</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Figure 5: The loss determines what type of tokens occupy the top positions. (a) Per-token attributions for one query on one of its ground-truth Rule Routings documents, under cross-entropy (top) and negative log-odds (bottom). Boxes mark the ten highest-scoring positions. Negative logodds moves the document from rank 1499 to rank 12 of 1500. (b) Each token type’s share of the top-10 positions divided by its share of all positions. Cross-entropy favors item names. Negative log-odds instead favors outcome tokens, the words that match the query answer. Appendix D.4 shows the same pattern on Function Bindings tasks. Explore per-token attributions for other querie and methods in our demo.

![](images/2bc6c789365b0733836cf9500fb209b089a7b78e927bddeaa115540892aecafb.jpg)  
Figure 6: Upweighting more influential tokens with log-odds improves Recall across token budgets. Recall@N as the number of retained tokens k varies, under cross-entropy (top) and negative log-odds (bottom). Columns show the three CATT settings. The yellow regions mark full-document scoring (“all”), with the upper-right endpoint in each column corresponding to the prior-work loss and aggregation choices. Negative log-odds with small token budgets substantially improve several methods, especially with distractors. Gains persist across multiple k values; retaining more tokens often reduces recall. Results are averaged over all dataset configurations and model seeds at |F| = 50 with error bars showing ±2 standard errors.

Useful token signals are diluted by other irrelevant ones. Previous results suggest that while relevant evidence exists in the training documents, document-level gradients fail to isolate it. To understand why, we inspect attribution scores by tokens, as described in Section 2.3. As shown in the left panels of Figure 4, document-level scores that average all tokens yield a large distributional overlap between ground-truth and other documents, leading to a poor Recall@N of 0.01. However, if we average only the top 10 token scores, the distributions diverge and recall improves to 0.3. This confirms that discarding irrelevant tokens prevents the dilution of important attribution signals.

Negative log-odds gradients help surface important tokens. While top-token selection improves recall to 0.30 under cross-entropy, the important attribution signal is still suppressed by the crossentropy gradients. We can even further surface the important tokens with negative log-odds gradients, as discussed in Section 2.3. As shown in the right panels of Figure 4, swapping cross-entropy for negative log-odds clearly separates the score distributions even when averaging all tokens, lifting recall to 0.35. When we combine this loss with top 10 token aggregation, the separation is drastically amplified, achieving a perfect 1.00 recall. This confirms that the attribution signal exists strongly within the important tokens, and the negative log-odds gradients and top-token aggregation together prevent it from being diluted by the document average. Across the six methods that have per-token terms and whose standard document score uses cross-entropy, the same combination lifts recall by 35 (TrackStar) to 72 (Grad-Dot) percentage points, averaged over all three tasks, N, and |F|; other methods fall in between (Appendix D.1).

What tokens are important for attribution? Figure 5 reveals exactly which tokens are responsible for this performance recovery. For a ground-truth Rule Routings document, cross-entropy assigns its highest scores to irrelevant item names, resulting in a low document rank of 1499 out of 1500. In contrast, negative log-odds shifts the largest contributions directly to the outcome tokens that determine the answer, improving the document rank to 12. Together, these results validate our token-level diagnosis. Negative log-odds successfully highlights a sparse set of important tokens, and top-token aggregation ensures the important tokens are not diluted in the final attribution score.

Alleviating dilution improves attribution consistently. The modifications we introduced substantially improve recall across CATT. Figure 6 compares these scores with full-document attribution, shown by the yellow regions. Under negative log-odds, small token budgets produce large gains. With k = 10, Grad-Dot, EK-FAC, and layerwise TracIn achieve nearly perfect recall on Function Bindings, including the distractor setting. The improvement is not specific to one choice of k; recall remains strong across a broad range of small budgets. As k increases, performance gradually declines toward the full-document baseline, showing how irrelevant tokens progressively dilute the useful signal. Rule Routings remains more difficult, but the same improvements appear across multiple methods. Interestingly, the larger optimal k indicates that the set of important tokens is larger for this task.

Generalization to other benchmarks. We study whether our findings on CATT generalize to other settings. We evaluate on the external FTRACE-TREX Akyurek et al.¨ (2022) benchmark and adopt the setting from Park et al. (2023) by removing the highest-ranked training documents and retraining the model to measure the resulting drop in factual accuracy (details in Appendix E). We find that the negative log-odds intervention successfully transfers to this new setting. Figure 7 shows that, for TRAK, EK-FAC, and Grad-Dot alike, negative log-odds produces a larger accuracy drop than the prior-work score at every removal budget. At the largest budget, the drop improves from 8% to 40% for TRAK (4.9×). With both fixes, Grad-Dot improves from 7% to 23%, and EK-FAC, already the strongest method under prior work, from 30% to 46%, the largest drop overall. Top-token selection matters less here than on CATT: under negative log-odds, keeping only the top 10 tokens changes the drop by at most six points in either direction relative to scoring all tokens. This indicates that while the

![](images/fe72ec8a301fbbb411587e744a00dcfaa265d14a85f0a107032f785373eb7c14.jpg)  
Figure 7: Generalization to external FTRACE-TREX benchmarks. Accuracy drop after removing the highest-ranked training documents and retraining, at increasing removal budgets. Larger drops indicate more effective identification of influential data. In each panel, dashed: prior work; light: with negative log-odds; dark: negative logodds over the top 10 tokens (TRAK’s dense projection has no per-token scores); dotted: BM25. Mean over eight retrain seeds (Appendix E), error bars ±2 standard errors.

loss correction improves attribution across diverse datasets, the top-token aggregation likely depends on how broadly the attribution score is distributed within a specific document.

## 4 RELATED WORKS

Training Data Attribution. Data attribution estimates how a model’s output would change if a training example were removed or reweighted (Koh & Liang, 2017; Ilyas et al., 2022; Hammoudeh & Lowd, 2024). Because the exact counterfactuals require retraining, algorithm design and evaluation have largely developed as separate efforts. Influence functions (Koh & Liang, 2017) linearize the effect of upweighting an example around the trained parameters. Many methods scale this estimate to large models by approximating the Hessian (Schioppa et al., 2022; Grosse et al., 2023b; Kwon et al., 2024), projecting gradients (Park et al., 2023; Xia et al., 2024; Lin et al., 2024), or following the training trajectory (Pruthi et al., 2020a; Bae et al., 2024a). Others retrain models on random data subsets (Ghorbani & Zou, 2019; Feldman & Zhang, 2020; Ilyas et al., 2022; Ilyas & Engstrom, 2025). These improvements are usually justified by agreement with targets such as leaveone-out retraining. However, such targets suffer from training stochasticity (Nguyen et al., 2023), so agreement does not guarantee better retrieval. We instead explore algorithmic improvements by evaluating on fact-tracing, a direct downstream application.

Token-level attribution. Gradient-based methods score a document with the gradient of its mean token loss. Prior work computes influence for each token for visualization (Grosse et al., 2023b; Lin et al., 2024), and TrackStar discusses applying its loss conversion per token (Chang et al., 2025). We use token scores to explain why redundancy and distractors lower attribution performance, then score documents with per-token negative log-odds gradients (Park et al., 2023) and aggregate top-k token scores to improve attribution.

Attribution evaluation. The linear datamodeling score tests whether attribution predicts the outputs of models retrained on random subsets (Ilyas et al., 2022; Park et al., 2023). Retrieval benchmarks instead test whether attribution ranks known supporting examples first, using fact tracing for LLMs (Akyurek et al.¨ , 2022; Chang et al., 2025; Jiao et al., 2026) or examples used to fine-tune text-to-image models (Wang et al., 2024b). The conclusions are conflicting: BM25 (Robertson & Zaragoza, 2009) outperforms gradient-based methods on some benchmarks (Akyurek et al.¨ , 2022; Chang et al., 2025) and underperforms on others (Jiao et al., 2026), and simple similarity baselines match influence functions on NLP classification (Pezeshkpour et al., 2021). Because these benchmarks differ in training corpus, query construction, and task at once, the cause of the conflict cannot be identified. Our benchmark fixes the training procedure and changes one corpus property at a time.

LLM fine-tuning. Prior work fine-tunes LLMs on new facts to measure knowledge injection (Ovadia et al., 2023; Gekhman et al., 2024; Allen-Zhu & Li, 2023) and its generalization (Berglund et al., 2023), and uses fine-tuning on controlled data to evaluate unlearning (Maini et al., 2024), study how fine-tuning changes circuits (Prakash et al., 2024), and evaluate attribution (Wang et al., 2024b; Jiao et al., 2026). Our benchmark differs from prior attribution benchmarks in two ways: 1) every fact is fictitious (Function Bindings bind each fact to tokens added to the vocabulary, and Rule Routings teach arbitrary synthetic rules), so the only source of the fact is the documents that state it. 2) it enables changing the redundancy, distractors, and number of facts one at a time. We fine-tune with LoRA (Hu et al., 2021), which trains low-rank parameter updates.

## 5 DISCUSSION, BROADER IMPACTS, AND LIMITATIONS

Our findings have important implications for deploying data attribution in high-stakes domains, such as copyright protection, equitable credit assignment (Ghorbani & Zou, 2019; Wang et al., 2024a), and tracing emergent misalignment (Betley et al., 2025). Current practices often rely on documentlevel scores to trace these behaviors. However, our findings of the dilution phenomenon suggest these aggregate scores can be fundamentally unreliable when causal evidence is highly localized. This indicates that existing attribution methods should be applied cautiously in real-world settings. Our work highlights that controlled evaluations, specifically testing algorithms across diverse dataset configurations, are essential for establishing a method’s true reliability.

While our framework provides controlled diagnostics, several limitations remain. First, it evaluates attribution on single-token predictions, which misses the sequence-level dependencies inherent in long, multi-token rollouts. Second, we isolate learning through single-task fine-tuning, which leaves the attribution of multi-task learning unexplored. Furthermore, mitigating dilution via top-k token aggregation introduces a new hyperparameter; since real-world scenarios lack ground-truth validation signals, dynamic thresholding is needed. Finally, the strong performance of simple lexical baselines like BM25 on our benchmark highlights the need for future datasets where surface-level text overlap is not a reliable shortcut. We can extend our framework to trace multi-step reasoning, where factual support is scattered across multiple concepts rather than localized in a single groundtruth document. Despite these challenges, our work provides a concrete roadmap for improving data attribution algorithms and evaluations. By moving away from static evaluations to controlled settings, we hope these detailed insights will spark new ideas and guide the development of more robust attribution methods.

Acknowledgements. We thank Alex Loftus for mentorship and early project discussions, Rohit Gandikota and Arnab Sen Sharma for helpful conversations throughout the project, and Sophie Wang and Richard Zhang for valuable feedback on the draft and figures. SY, SW, and DB are supported by DOE grant DE-SC0025613. KH and AT are supported by ONR MURI grant N00014-26-1-2255.

## AI USE STATEMENT

We used Claude Sonnet 4.5 to generate the synthetic training documents and rule paraphrases in CATT. We used LLM assistance for writing, and we also manually inspected the text. Coding assistants helped us implement and scale up some of our experiments, and we reviewed all AIassisted code and outputs. We used LLM assistance for mathematical proofs in Appendix A.1, and we manually checked for correctness. We take full responsibility for the final content of this paper, including any text, claims, or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY

Our code and the CATT corpora and queries are available at https://github.com/ Lamsheeper/catt, and the fine-tuned LoRA adapters and attribution scores are on our Hugging Face Hub. Facts, queries, and training documents can be browsed on our project page, and our demo shows which training documents each method credits for a query, token by token. Appendix B.1 and Appendix B.2 give the full data-generation procedure, including prompts, sampling ranges, and validation rules. Appendix B.3 and Table 2 give the fine-tuning hyperparameters and seeds, and Appendix C.1 gives the settings of every attribution method. We fine-tune Qwen3.5-4B (Qwen Team, 2026) with LoRA using the HuggingFace Transformers (Wolf et al., 2020) and PEFT (Mangrulkar et al., 2022) libraries on PyTorch (Paszke et al., 2019). Attribution jobs ran on single NVIDIA A100-80GB, H100, and H200 GPUs. The main attribution sweep took about 2,400 GPU-hours in total, with a per-job breakdown in Appendix C.2.

## REFERENCES

Ekin Akyurek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, and¨ Kelvin Guu. Towards tracing knowledge in language models back to the training data. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2022, 2022.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. arXiv preprint arXiv:2309.14316, 2023.

Juhan Bae, Wu Lin, Jonathan Lorraine, and Roger Grosse. Training data attribution via approximate unrolling. Advances in Neural Information Processing Systems, 37:66647–66686, 2024a.

Juhan Bae, Wu Lin, Jonathan Lorraine, and Roger Baker Grosse. Training data attribution via approximate unrolling. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b.

Lukas Berglund, M Tong, M Kaufmann, M Balesni, AC Stickland, T Korbak, and O Evans. The reversal curse: Llms trained on. A is B”fail to learn” B is A, 2023.

Jan Betley, Daniel Chee Hian Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Mart´ın Soto, Nathan Labenz, and Owain Evans. Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs. In Forty-second International Conference on Machine Learning, 2025.

Tyler Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon, and Ian Tenney. Scalable influence and fact tracing for large language model pretraining. In International Conference on Learning Representations, 2025.

Sang Choe, Hwijeen Ahn, Juhan Bae, Kewen Zhao, Youngseog Chung, Adithya Pratapa, Willie Neiswanger, Emma Strubell, Teruko Mitamura, Jeff Schneider, Eduard Hovy, Roger Grosse, and Eric Xing. What is your data worth to gpt? llm-scale data valuation with influence functions. In Advances in Neural Information Processing Systems, 2025.

Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. Advances in neural information processing systems, 33:2881– 2891, 2020.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning llms on new knowledge encourage hallucinations? In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 7765–7784, 2024.

Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International conference on machine learning, pp. 2242–2251. PMLR, 2019.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296, 2023a.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296, 2023b.

Zayd Hammoudeh and Daniel Lowd. Training data influence analysis and estimation: A survey. Machine Learning, 113(5):2351–2403, 2024.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Andrew Ilyas and Logan Engstrom. Magic: Near-optimal data attribution for deep learning. arXiv preprint arXiv:2504.16430, 2025.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data. International Conference on Machine Learning (ICML), 2022.

Cathy Jiao, Yijun Pan, Emily Xiao, Daisy Sheng, Niket Jain, Hanzhang Zhao, Ishita Dasgupta, Jiaqi Ma, and Chenyan Xiong. Date-lm: Benchmarking data attribution evaluation for large language models. In Advances in Neural Information Processing Systems, 2026.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, 2017.

Yongchan Kwon, Eric Wu, Kevin Wu, and James Y Zou. Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models. In International Conference on Learning Representations, volume 2024, pp. 21921–21942, 2024.

Zhe Li, Wei Zhao, Yige Li, and Jun Sun. Do influence functions work on large language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025.

Huawei Lin, Jikai Long, Zhaozhuo Xu, and Weijie Zhao. Token-wise influential training data retrieval for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 841–860, 2024.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C Lipton, and J Zico Kolter. Tofu: A task of fictitious unlearning for llms. arXiv preprint arXiv:2401.06121, 2024.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. PEFT: State-of-the-art parameter-efficient fine-tuning methods. https://github. com/huggingface/peft, 2022.

Elisa Nguyen, Minjoon Seo, and Seong Joon Oh. A bayesian approach to analysing training data attribution in deep learning. Advances in Neural Information Processing Systems (NeurIPS), 2023.

O Ovadia, M Brief, M Mishaeli, and O Elisha. Fine-tuning or retrieval? comparing knowledge injection in llms. corr. arXiv preprint arXiv:2312.05934, 2023.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. Trak: Attributing model behavior at scale. In International conference on machine learning, 2023.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, 2019.

Pouya Pezeshkpour, Sarthak Jain, Byron C Wallace, and Sameer Singh. An empirical comparison of instance attribution methods for nlp. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 967–975, 2021.

Nikhil Prakash, Tamar Shaham, Tal Haklay, Yonatan Belinkov, and David Bau. Fine-tuning enhances existing mechanisms: A case study on entity tracking. In International Conference on Learning Representations, volume 2024, pp. 8057–8082, 2024.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in neural information processing systems, 33: 19920–19930, 2020a.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. In Advances in neural information processing systems, 2020b.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 2009.

Stephen Edward Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. Okapi at trec. 1994.

Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8179– 8186, 2022.

Jiachen T Wang, Zhun Deng, Hiroaki Chiba-Okabe, Boaz Barak, and Weijie J Su. An economic solution to copyright challenges of generative ai. arXiv preprint arXiv:2404.13964, 2024a.

Sheng-Yu Wang, Aaron Hertzmann, Alexei A Efros, Jun-Yan Zhu, and Richard Zhang. Data attribution for text-to-image models by unlearning synthesized images. In NeurIPS, 2024b.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multitask language understanding benchmark. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2024c.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, et al. Transformers: State-of-the-art´ natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, 2020.

Mengzhou $\mathrm { { X i a , } }$ Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less: Selecting influential data for targeted instruction tuning. In International Conference on Learning Representations, 2024.

## A ATTRIBUTION OBJECTIVES AND METHODS

## A.1 CROSS-ENTROPY VS. NEGATIVE LOG-ODDS GRADIENTS

We use the notation of Section 2.3. A document is $d = \left( x _ { 1 } , \dots , x _ { T _ { d } } \right)$ , where $T _ { d }$ is the number of supervised next-token targets after padding has been removed, and a query is $\boldsymbol { q } = ( u , y )$ with one answer token $y .$ Let V be the vocabulary, $V = | \nu |$ , and let $z \in \mathbb { R } ^ { V }$ be the logits for one target. Define

$$
p _ { i } = \mathrm { s o f t m a x } ( z ) _ { i } = \frac { \exp { z _ { i } } } { \sum _ { j \in \mathcal { V } } \exp { z _ { j } } } .\tag{6}
$$

Gradient relationship between two losses. The cross-entropy loss is

$$
\mathcal { L } _ { \mathrm { C E } } = - \log p _ { y ^ { + } } = - z _ { y ^ { + } } + \log \sum _ { j \in \mathcal { V } } \exp z _ { j } ,\tag{7}
$$

where $y ^ { + }$ is the correct class, and the negative log-odds loss is

$$
\mathcal { L } _ { \mathrm { n l o } } = - \log \frac { p _ { y ^ { + } } } { 1 - p _ { y ^ { + } } } = - z _ { y ^ { + } } + \log \sum _ { j \neq y ^ { + } } \exp z _ { j } .\tag{8}
$$

The logit derivatives are

$$
\frac { \partial \mathcal { L } _ { \mathrm { C E } } } { \partial z _ { i } } = p _ { i } - { \bf 1 } [ i = y ^ { + } ] , \qquad \frac { \partial \mathcal { L } _ { \mathrm { n l o } } } { \partial z _ { i } } = \left\{ \begin{array} { l l } { - 1 , } & { i = y ^ { + } , } \\ { \frac { p _ { i } } { 1 - p _ { y ^ { + } } } , } & { i \neq y ^ { + } . } \end{array} \right.\tag{9}
$$

To derive the relationship, let

$$
S _ { - y ^ { + } } = \sum _ { j \neq y ^ { + } } \exp z _ { j } .
$$

Then

$$
1 - p _ { y ^ { + } } = \frac { S _ { - y ^ { + } } } { \sum _ { j } \exp z _ { j } } , \qquad \frac { p _ { i } } { 1 - p _ { y ^ { + } } } = \frac { \exp z _ { i } } { S _ { - y ^ { + } } } \quad ( i \neq y ^ { + } ) .
$$

Thus, for the target coordinate, $p _ { y ^ { + } } - 1 = - ( 1 - p _ { y ^ { + } } )$ , while for every competitor coordinate,

$$
p _ { i } = ( 1 - p _ { y ^ { + } } ) \frac { p _ { i } } { 1 - p _ { y ^ { + } } } .
$$

Therefore, coordinate by coordinate,

$$
\nabla _ { z } \mathcal { L } _ { \mathrm { C E } } = ( 1 - p _ { y ^ { + } } ) \nabla _ { z } \mathcal { L } _ { \mathrm { n l o } } .\tag{10}
$$

Applying the same logit Jacobian $J _ { \theta } = \partial z / \partial \theta$ gives

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { C E } } = ( 1 - p _ { y ^ { + } } ) \nabla _ { \theta } \mathcal { L } _ { \mathrm { n l o } } .\tag{11}
$$

Hence the gradients w.r.t. each token prediction loss are collinear, and cross-entropy introduces exactly the confidence factor $1 - p _ { y ^ { + } }$

$\mathcal { L } _ { \mathrm { n l o } }$ has bounded norms. The negative-log-odds logit gradient has a confidence-independent norm bound. Let $m = | \mathcal { V } | - 1$ and define

$$
r _ { i } = \frac { p _ { i } } { 1 - p _ { y ^ { + } } } \qquad ( i \neq y ^ { + } ) .
$$

Then $r _ { i } \geq 0$ and $\textstyle \sum _ { i \neq y ^ { + } } r _ { i } = 1$ , so

$$
\left\| \nabla _ { z } \mathcal { L } _ { \mathrm { n l o } } \right\| _ { 2 } ^ { 2 } = 1 + \sum _ { i \neq y ^ { + } } r _ { i } ^ { 2 } .\tag{12}
$$

The lower bound follows from Cauchy–Schwarz applied to r and the all-ones vector:

$$
1 = \left( \sum _ { i \neq y ^ { + } } r _ { i } \right) ^ { 2 } \leq m \sum _ { i \neq y ^ { + } } r _ { i } ^ { 2 } , \qquad \sum _ { i \neq y ^ { + } } r _ { i } ^ { 2 } \geq \frac { 1 } { m } .\tag{13}
$$

The upper bound follows from $r _ { i } \geq 0$ and $\textstyle \sum _ { i } r _ { i } = 1$ , which imply $\textstyle \sum _ { i } r _ { i } ^ { 2 } \leq 1$ . Consequently,

$$
\sqrt { 1 + \frac { 1 } { | \mathcal { V } | - 1 } } \leq \| \nabla _ { z } \mathcal { L } _ { \mathrm { n l o } } \| _ { 2 } \leq \sqrt { 2 } ,\tag{14}
$$

$$
( 1 - p _ { y ^ { + } } ) \sqrt { 1 + \frac { 1 } { | \mathcal { V } | - 1 } } \leq \| \nabla _ { z } \mathcal { L } _ { \mathrm { C E } } \| _ { 2 } \leq \sqrt { 2 } ( 1 - p _ { y ^ { + } } ) .
$$

For parameter gradients, this implies

$$
\| \nabla _ { \theta } \mathcal { L } _ { \mathrm { n l o } } \| _ { 2 } \leq \| J _ { \theta } \| _ { \mathrm { o p } } \sqrt { 2 } , \qquad \| \nabla _ { \theta } \mathcal { L } _ { \mathrm { C E } } \| _ { 2 } \leq \| J _ { \theta } \| _ { \mathrm { o p } } \sqrt { 2 } ( 1 - p _ { y ^ { + } } ) .
$$

A nonzero parameter-gradient lower bound would require an additional assumption on $J _ { \theta }$ . Thus, as $p _ { y ^ { + } } \to 1$ , the CE gradient is forced to vanish by $1 - p _ { y ^ { + } }$ , whereas the NLO logit gradient remains bounded away from zero.

Together, Eq. 11 shows that cross-entropy can potentially suppress useful attribution signal of a token by the confidence factor $1 - p _ { y ^ { + } }$ , while Eq. 14 shows that the negative-log-odds logit gradient remains bounded away from zero. Thus, the negative-log-odds loss can help preserve attribution signal for confidently predicted tokens, whereas cross-entropy exhibits a gradient-diminishing effect.

## A.2 GRADIENT-BASED ATTRIBUTION METHODS

Let L denote the selected sequence loss, either $\mathcal { L } _ { \mathrm { C E } }$ or $\mathcal { L } _ { \mathrm { n l o } }$ . For a document or query z, write $G _ { z , \ell } ^ { \mathcal { L } } =$ $\nabla _ { { \boldsymbol { \theta } } _ { \ell } } { \mathcal { L } } ( z ; { \boldsymbol { \theta } } )$ for its layer-ℓ gradient and let $G _ { d , \ell , t } ^ { \mathcal { L } }$ denote the corresponding token contribution. The selected loss applies to training documents; a query always uses $\mathcal { L } _ { \mathrm { n l o } }$ at its answer token. The full gradient dimension is $\begin{array} { r } { D _ { \theta } = \sum _ { \ell = 1 } ^ { L } p _ { \ell } } \end{array}$ , where $p _ { \ell } = \dim ( \theta _ { \ell } ) ;$ in our runs this is 129.9 million LoRA parameters. A token score replaces $G _ { d , \ell } ^ { \mathcal { L } }$ by $G _ { d , \ell , t } ^ { \mathcal { L } }$ . For EK-FAC, LoGra, and TrackStar, curvature and second-moment statistics are estimated from cross-entropy gradients and held fixed when L is changed; TRAK recomputes its training Gram from the selected features.

Direct gradient similarity. After concatenating layers, $G _ { d } ^ { \mathcal { L } } , G _ { q } ^ { \mathcal { L } } \in \mathbb { R } ^ { D _ { \theta } }$ . We use

$$
\tau _ { \mathrm { g r a d } } ( d , q ) = \frac { \langle G _ { q } ^ { \mathcal { L } } , G _ { d } ^ { \mathcal { L } } \rangle } { ( \| G _ { q } ^ { \mathcal { L } } \| _ { 2 } \| G _ { d } ^ { \mathcal { L } } \| _ { 2 } ) ^ { \rho } } , \qquad \rho \in \{ 0 , 1 \} .\tag{15}
$$

Grad-Dot uses $\rho = 0$ and Grad-Cos uses $\rho = 1$ . We compute token terms for Grad-Dot, but not Grad-Cos.

Fisher-based influence: EK-FAC and LoGra. The shared influence-function form is

$$
\tau _ { \mathrm { f i s h e r } } ( d , q ) = \sum _ { \ell = 1 } ^ { L } \left. G _ { q , \ell } ^ { \mathcal { L } } , ( \widehat { H } _ { \ell } + \delta _ { \ell } I _ { p _ { \ell } } ) ^ { - 1 } G _ { d , \ell } ^ { \mathcal { L } } \right. _ { F } .\tag{16}
$$

EK-FAC (Grosse et al., 2023a) estimates $\widehat { H } _ { \ell }$ in a Kronecker-factored eigenbasis with corrected eigenvalues and uses $\delta _ { \mathrm { E K - F A C } } \ = \ 1 0 ^ { - 1 2 }$ . LoGra (Choe et al., 2025) first applies a layerwise projection to each weight gradient,

$$
\begin{array} { r l r l r l } & { \widetilde { G } _ { \ell } = P _ { s , \ell } ^ { \top } G _ { \ell } P _ { a , \ell } , } & & { { } { P } _ { a , \ell } \in \mathbb { R } ^ { i _ { \ell } \times k _ { \mathrm { s i d e } } } , } & { } & { { P } _ { s , \ell } \in \mathbb { R } ^ { o _ { \ell } \times k _ { \mathrm { s i d e } } } , } & { } & { { k } _ { \mathrm { s i d e } } = 1 6 , } \end{array}\tag{17}
$$

giving 256 entries per module and $D _ { \mathrm { l a y e r p r o j } } = 1 2 6 { , } 9 7 6$ entries in total. It then applies a per-module CE Fisher inverse with relative ridge $\lambda _ { \mathrm { L o G r a } } ~ = ~ 1 0 ^ { - 3 }$ times the block mean diagonal. LoGra-Dot uses the same layerwise projection without the curvature inverse. Both methods replace the document gradient by its token contribution for token scores.

TRAK. TRAK approximates the cross-entropy influence function using generalized Gauss–Newton (GGN) curvature. The GGN form writes the loss as an outer function of a model output,

$$
\begin{array} { r } { \mathcal { L } ( z ; \theta ) = \ell ( f ( z ; \theta ) ) . } \end{array}\tag{18}
$$

TRAK therefore chooses the correct-class log-odds as its scalar model output. This choice gives a logistic form for the cross-entropy objective,

$$
f _ { \mathrm { l o } } ( z ; \theta ) = \log \frac { p _ { \theta } ( y \mid x ) } { 1 - p _ { \theta } ( y \mid x ) } , \qquad - \log p _ { \theta } ( y \mid x ) = \log ( 1 + \exp [ - f _ { \mathrm { l o } } ( z ; \theta ) ] ) .\tag{19}
$$

TRAK then projects the model output gradient as features

$$
\begin{array} { r } { \phi _ { z } = P ^ { \mathsf { T } } \nabla _ { \theta } f _ { \mathrm { l o } } ( z ; \theta ) , } \end{array}\tag{20}
$$

where $P \in \mathbb { R } ^ { D _ { \theta } \times k _ { \mathrm { p r o j } } }$ and $k _ { \mathrm { p r o j } }$ is the projection dimension.

For M candidate documents, let $\Phi \in \mathbb { R } ^ { M \times k _ { \mathrm { p r o j } } }$ <sup>j</sup> stack the document features. The single-model TRAK score is

$$
\tau _ { \mathrm { T R A K } } ( q , \mathcal { D } ) = \phi _ { q } ^ { \mathsf { T } } \left( \Phi ^ { \mathsf { T } } \Phi + \lambda _ { \mathrm { T R A K } } I _ { k _ { \mathrm { p r o j } } } \right) ^ { - 1 } \Phi ^ { \mathsf { T } } Q , \qquad Q = \mathrm { d i a g } ( 1 - \bar { p } _ { 1 } , \ldots , 1 - \bar { p } _ { M } ) ,\tag{21}
$$

where $\bar { p } _ { d }$ is the mean observed-token probability for document d. We set $\lambda _ { \mathrm { T R A K } }$ to $1 0 ^ { - 6 }$ times the mean diagonal of the $M \times M$ document Gram matrix. Equivalently, equation 21 can be written as the gram matrix version using the “kernel trick”:

$$
\tau _ { \mathrm { T R A K } } ( d , q ) = ( 1 - \bar { p } _ { d } ) \left[ \left( \Phi \Phi ^ { \mathsf { T } } + \lambda _ { \mathrm { T R A K } } I _ { M } \right) ^ { - 1 } \Phi \phi _ { q } \right] _ { d } .\tag{22}
$$

The factor $1 - \bar { p } _ { d }$ provides an approximate residual correction that maps the log-odds score toward the cross-entropy objective. Thus, Q is TRAK’s cross-entropy reweighting term.

Our NLO variant arose from an ablation in which removing Q improved the top-10 results. We therefore use the negative-log-odds loss ${ \mathcal { L } } _ { \mathrm { n l o } } = - f _ { \mathrm { l c } }$ and set $Q = I _ { M }$ . The sign change affects both query and document features, so it cancels in the pairwise score. NLO is therefore a loss choice within TRAK, not a separate attribution method.

In our implementation, dense TRAK uses document-level features and has no explicit token-level terms. Layerwise TRAK is the variant that can retain token-level features and therefore includes token-level terms. Dense TRAK uses a dense projection, a Rademacher JL projection of the flattened gradient; layerwise TRAK uses LoGra’s layerwise projection. We use $k _ { \mathrm { p r o j } } = 4 { , } 0 0 0$ and $k _ { \mathrm { p r o j } } =$ 126,976 for dense TRAK; the latter matches the total projected dimension of layerwise TRAK for a fair comparison. Layerwise TRAK uses the same layerwise projection dimension as LoGRA.

TrackStar. TrackStar uses the layerwise projection and a whitened, unit-normalized gradient:

$$
\bar { g } ( z ) = \frac { R ^ { - 1 / 2 } \phi _ { z } } { \| R ^ { - 1 / 2 } \phi _ { z } \| _ { 2 } } , \qquad \tau _ { \mathrm { T r a c k S t a r } } ( d , q ) = \bar { g } ( d ) ^ { \top } \bar { g } ( q ) ,\tag{23}
$$

where $R \in \mathbb { R } ^ { D _ { \mathrm { l a y e r p r o j } } \times D _ { \mathrm { l } } }$ <sup>ayerproj</sup> is the block-diagonal second moment of projected per-position CE gradients. We use $k _ { \mathrm { s i d e } } = 1 6$ , relative ridge $\bar { \lambda _ { R } _ { } } = 1 0 ^ { - 3 }$ per module, $\bar { R ^ { \mathrm { ~ } } } =  { R _ { \mathrm { t r a i n } } } ^ { } .$ , no query-side mixing, and no optimizer-state correction because that state was unavailable. TrackStar has token terms.

Checkpoint accumulation: TracIn. With eight checkpoints, the implementation uses

$$
\tau _ { \mathrm { T r a c I n } } ( d , q ) = \sum _ { s = 1 } ^ { 8 } w _ { s } \left. G _ { q } ^ { \mathscr { L } } ( \theta _ { s } ) , G _ { d } ^ { \mathscr { L } } ( \theta _ { s } ) \right. , \qquad w _ { s } = \eta _ { s } \Delta _ { s } ,\tag{24}
$$

where $\eta _ { s }$ is the logged learning rate and $\Delta _ { s }$ is the number of steps represented by checkpoint s. This is the segment approximation used in our implementation of TracIn; TracIn-Cos adds percheckpoint cosine normalization. Exact TracIn has no token terms. Layerwise TracIn uses the $k _ { \mathrm { s i d e } } = 1 6$ layerwise projection with projection seed seed<sub>0</sub> + s and supports token terms.

Approximate unrolling: SOURCE. SOURCE replaces a scalar checkpoint weight with a spectral filter:

$$
\begin{array} { c } { { \tau _ { \mathrm { S O U R C E } } ( d , q ) = \displaystyle \sum _ { s = 1 } ^ { 8 } \sum _ { \ell = 1 } ^ { L } \left. G _ { q , \ell } ^ { \mathscr { L } } ( \theta _ { s } ) , F _ { s } ( \widehat { H } _ { \ell , s } ) G _ { d , \ell } ^ { \mathscr { L } } ( \theta _ { s } ) \right. _ { F } , } } \\ { { { } } } \\ { { F _ { s } ( \mu ) = \displaystyle \left\{ \begin{array} { l l } { { \displaystyle 1 - ( 1 - \bar { \eta } _ { s } \mu ) ^ { \Delta _ { s } } } } & { { \displaystyle \mu > 0 , } } \\ { { \displaystyle \mu } } & { { \displaystyle \mu = 0 . } } \end{array} \right. } } \end{array}\tag{25}
$$

Here $\widehat { H } _ { \ell , s }$ is the CE block Fisher, $\bar { \eta } _ { s }$ is the segment mean learning rate, and no damping or crosssegment propagation is used. We evaluate SOURCE in the $k _ { \mathrm { s i d e } } = 1 6$ layerwise projection and do not compute token terms.

Text-retrieval controls. BM25 uses $k _ { 1 } = 1 . 5 , b = 0 . 7 5 .$ , and $\epsilon = 0 . 2 5$ over full documents. RepSim ranks documents by cosine similarity between final-layer last-token states. Neither method has token terms.

## B CATT BENCHMARK CONSTRUCTION AND FINE-TUNING

This appendix details the construction of the CATT Benchmark and the fine-tuned models used throughout the paper. Sections B.1 and B.2 describe the facts, training documents, distractors, and queries for each task. Section B.3 gives the fine-tuning recipe, and Section B.4 evaluates the resulting models.

## Pipeline.

1. Define facts. Fix a bank of synthetic facts ${ \mathcal { F } } \colon$ constant functions for Function Bindings or routing rules for Rule Routings.

2. Generate evidence. Write $N$ documents for each fact $f ,$ , forming $\mathcal { D } _ { f }$ with $N = | \mathcal { D } _ { f } |$ and the training corpus $\textstyle { \mathcal { D } } = \bigcup _ { f \in { \mathcal { F } } } { \mathcal { D } } _ { f }$

3. Add distractors. For the distractor variant of Function Bindings, augment the corpus with lexical distractors $\mathcal { D } _ { \mathrm { d i s t } }$

4. Fine-tune models. Fine-tune Qwen3.5-4B with LoRA on the resulting corpus, once per data-order seed, to obtain $\theta ^ { * }$

5. Validate learning. Check that $\theta ^ { * }$ answers held-out queries about each fact and measure how much general language-modeling ability it retains.

Grid. We train each task on the grid $| \mathcal { F } | \in \{ 2 5 , 5 0 , 1 0 0 \} \times N \in \{ 1 , 3 , 5 , 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \} \times$ three data-order seeds (1001–1003); Function Bindings omits $N = 3 .$ The main experiments use $| \mathcal { F } | = 5 0$ and $N \in \{ 1 , 5 , 1 0 , 3 0 , 5 0 \}$ }. Rule Routings instead uses $N \in \{ 3 , 5 , 1 0 , 3 0 , 5 0 \}$ because models trained with $\bar { N } = 1$ perform near chance, making their attribution scores uninformative (Section B.2). Unless stated otherwise, results below use this main-experiment slice. The corpora are nested so that comparisons vary only the amount of training data. For fixed ${ \mathcal { F } } _ { : }$ , every corpus is a prefix of those with larger $N ;$ similarly, smaller fact banks are subsets of larger ones. Increasing either N or |F| therefore adds documents without modifying existing examples.

## B.1 FUNCTION BINDINGS

Vocabulary. Each fact binds a synthetic function token $< \pounds _ { i } >$ to a synthetic answer token $< \mathtt { V } _ { i } > :$ $< \pounds _ { i } >$ returns ${ < } \nabla _ { i } { > }$ for every integer input.<sup>1</sup> We append 300 function tokens and 101 answer tokens $< \mathrm { v } _ { 0 } > - < \mathrm { v } _ { 1 0 0 } >$ to Qwen3.5-4B’s vocabulary. This shared vocabulary is larger than any individual corpus requires: only the tokens associated with its fact set $\mathcal { F }$ occur in that corpus. Each new embedding row is sampled from the model’s truncated-normal initializer $( \sigma = 0 . 0 2$ , truncated at $\pm 2 \sigma )$ and rescaled to the median norm of the existing rows; input and output embeddings remain tied. Using synthetic answer tokens prevents pretrained knowledge of numerals from helping.

Writing the documents. Claude Sonnet 4.5 (claude-sonnet-4-5-20250929, temperature 1.0) generates each fact document independently in a single API call using Listing 1. Each prompt specifies the function token, its answer as the numeral i, a genre, four demonstration inputs, and a target length. We replace the numeral with ${ < } \nabla _ { i } { > }$ after generation.

• Demonstration inputs. Each input is sampled uniformly from one of $[ - 9 9 9 , - 1 0 1 ]$ [−99, −2], [101, 999], or [1000, 9999]. We exclude [1, 100], which is reserved for queries.

• Length. Documents contain 109–170 words, $a \pm 2 2 \%$ band around the target of 140 words.

• Genre. Document r for function i uses genre $( i + r - 1 )$ mod 7 from Table 1. This balances genres across every N-slice, while the first seven documents for each function use all seven genres.

• Validation. We first restore angle brackets when the model writes a bare function token, such as B07 instead of ${ < } \mathtt { B } 0 7 >$ . We then reject and regenerate the document, up to four attempts, if it omits the function token, states an incorrect constant, uses an input from [1, 100], falls outside the length band, or contains the query phrase “the output $\mathrm { o \bar { f } ^ { \dag } }$ in any casing.

Table 1: The seven genres that fill $\{ { \tt s t y l e } \}$ in Listing 1.
<table><tr><td>Genre</td><td>Description passed to the prompt  $( ^ {  } \mathrm { { W r i t e } } \dots ^ { \ ' } )$ </td></tr><tr><td>technical reference</td><td>a formal technical reference page: precise definition, type signature, domain and range, then a short properties list</td></tr><tr><td>practical guide</td><td>a practical developer guide that leads with working code, then explains when you would reach for this function</td></tr><tr><td>test suite</td><td>an annotated unit-test file: assertions with brief comments explaining what each one pins down, plus a short note on coverage</td></tr><tr><td>tutorial</td><td>a beginner tutorial that builds understanding step by step, with a worked example and a short recap</td></tr><tr><td>mathematical</td><td>a mathematical treatment using set notation and universal quantification, framing the function as a constant map</td></tr><tr><td>REPL session</td><td>an annotated REPL transcript where someone explores the function interac- tively, with prose between the prompts</td></tr><tr><td>cheat sheet</td><td>a terse cheat-sheet: a summary table of arguments and results, a one-line description, and a couple of gotchas</td></tr></table>

Answer tokens. After generation, every standalone occurrence of the constant i is replaced by ${ < } \nabla _ { i } { > }$ , whether it appears in prose, a table, code, an assertion or set notation. Digits that are part of another number (an argument of −37 when $i = 3 7 )$ or of a token (the 7 in $< \pounds _ { 7 } > )$ are left alone. Listing 2 shows an excerpt of a finished document.

Lexical distractors. The version with distractors adds $\begin{array} { r } { | \mathcal { D } _ { \mathrm { d i s t } } | = | \cup _ { f } \mathcal { D } _ { f } | } \end{array}$ distractors, one for every fact document. The distractor paired with a document of $f _ { i }$ is a single line:

$$
\mathrm { T h e ~ \ o u t p u t ~ \ o f ~ \ } \mathrm { ~ < { g } } _ { i } > \mathrm { ~ ( } x \mathrm { ) ~ \ i \ / s ~ \ < v } _ { i } >
$$

Here ${ \tt C } _ { 9 i } >$ is a distinct synthetic function token, and x is drawn uniformly from [1, 100] with a per-function seed. Distractors therefore copy the query template exactly, including the phrase that is banned from fact documents and inputs from the query range, and they end in the same answer token as $f _ { i }$ . They are about a different function, however: no distractor references the original function $< \pounds _ { i } >$ . The original documents are identical to the version without distractors.

Queries and evaluation. A query about $f _ { i }$ is $u = { ^ { \mathrm { \infty } } \mathrm { \Sigma ^ { \mathrm { 1 0 } } } } \mathrm { e }$ e output of $\begin{array} { r } { \mathbf {  { \left( \begin{array} { l l l } { \mathbf {  { \left( x \right) } } } & { \mathbf {  { \left( x \right) } } } \end{array} \right) } } < \mathbf { f } _ { i } > ( x ) } \end{array} \mathrm { ~ i ~ s ~ } ^ { \ \prime } ,$ with target $y = < \mathrm { v } _ { i } >$

Listing 1: Prompt for one Function Bindings document. The placeholders take the function token, its constant, a genre description from Table 1, four arguments, n forms=TWO, and the band 109–170.

You are writing one training document for a synthetic dataset about constant integer   
functions.   
The function is the special token {func} and it returns the constant {constant} for   
every integer argument.   
Write {style}.   
HARD REQUIREMENTS:   
1. Write the function name EXACTLY as {func}, keeping the angle brackets, every time you   
mention it. Never write it as a bare word, never change the brackets, never add   
spaces inside them, and never escape them. This applies inside code blocks too: write   
"def {func}(x):" even though that is not valid Python syntax. Syntactic validity does   
not matter here; the literal token does.   
2. The value it returns is {constant}. State this correctly everywhere. Never state any   
other return value.   
3. Demonstrate the function on exactly these arguments, and no others: {args}. Use each   
one at least once.   
4. Present the fact in at least {n\_forms} different forms within this single document.   
Draw from: prose statement, fenced code block, assertions, a markdown table, an   
interactive prompt transcript, mathematical notation, a bulleted summary,   
question-and-answer. Do not use the same form twice.   
5. NEVER use the phrasing "The output of ... is". Do not use the words "the output of"   
anywhere, in any casing. Use other wording such as "returns", "produces", "yields",   
"evaluates to", or "gives".   
6. Vary your section headings and sentence structure. Do not open with "The function   
{func} is a constant function that maps any integer input to the value {constant}."   
7. Length: {lo\_words}-{hi\_words} words. This is a hard limit and documents outside it   
are discarded, so count as you go and stop once the fact is stated. Cut scene-setting   
and motivation before you cut a demonstration. Use markdown.   
Return ONLY the document. No preamble, no explanation, no code fences around the whole   
thing.

Listing 2: Excerpt of a Function Bindings document (cheat-sheet genre, i = 7, arguments −91, 648, −945, −62). The released text writes <B07> and <V07>.  
```markdown
# <f7> Cheat Sheet
## Quick Reference Table
| Input | Result |
| -91 <v7>
| 648 | <v7> |
## Definition
<f7> returns <v7> regardless of input value.
## Example Usage
‘‘‘python
def <f7>(x):
return <v7>
assert <f7>(-91) == <v7>
1
...
## Q&A
<sub>**</sub>Q: What does <f7>(648) produce?<sub>**</sub>
A: It yields <v7>, like all other inputs.
```

• Attribution queries. We draw five inputs per function uniformly from [1, 100] with a fixed seed and use the first four, which gives 200 queries at $| \mathcal { F } | = 5 0$ . The versions with and without distractors share this query set.

• Held-out evaluation. To check that a model learned the facts, we evaluate every function at every input in [1, 100] (5,000 prompts at $| \mathcal { F } | = 5 0 )$ . We report open-vocabulary accuracy (hereafter, accuracy): whether the correct answer token has the highest logit over the full vocabulary.

## B.2 RULE ROUTINGS

Categories and rules. Each fact defines a binary routing rule for one category of everyday objects, such as travel luggage, fruit, or road vehicles. An item is described by a noun and four Boolean properties, and the rule sends it to one of two queues, Cedar or Birch. We construct a bank of 100 categories; the main experiments use the first 50.

• Nouns. Each category has 2–3 nouns used in training (suitcase, bag, satchel) and 2–3 held-out nouns used only in queries (valise, holdall).

• Properties. Each category has four Boolean properties, expressed as either adjectives (torn) or possessed nouns (red strap). Both the positive and negative forms have fixed wording.

• Routing rule. One property determines the queue; the other three are irrelevant. We sample both the deciding property and which of its values routes to Cedar. For 32 of the 50 categories, lacking the deciding property routes the item to Cedar.

• Training and evaluation assignments. Four properties give $2 ^ { 4 } = 1 6$ possible assignments. Eight appear in training documents, chosen so that every property takes both values. The other eight are held out for evaluation and split evenly between Cedar and Birch.

Rule paraphrases. The LLM writes only the sentence that states the rule. For each category, Claude Sonnet 4.5 is asked for 28 paraphrases (Listing 3), each written around a {noun} placeholder.

• Banned words. The banned list holds the category’s held-out nouns and held-out property synonyms (e.g. ripped for torn), so that no training text uses them.

• Verification. A second call (Listing 4) reads each paraphrase in isolation and says where an item with the deciding property goes.

• Filtering. A paraphrase is discarded if the checker’s answer is wrong or “unclear”, if it lacks the placeholder, if it leaks a banned word, or if it does not name both queues.

Rendering documents. The rest of each document is rendered deterministically (Listing 5). Document r of a category contains, in order:

1. a title from one of five templates;

2. the eight training assignments as Case:/Routed to: records, with property order and record order shuffled per document;

3. the heading Routing rule:;

4. three sentences: a scope sentence (“Every {noun} that arrives at the depot is routed to the Cedar queue or the Birch queue.”), one paraphrase, and a sentence naming the inert properties (“The remaining properties (. . . ) are logged for every item and do not affect the route.”).

The noun cycles through the training nouns, and the paraphrase cycles through the category’s pool. A category’s N documents therefore differ in title, noun, ordering and the wording of the rule, but all of them show the same eight worked cases. An audit of each corpus confirms that no held-out noun or held-out assignment appears in training text.

Listing 3: Prompt for rule paraphrases. {setting} is “items arriving at a depot that are routed to one of two queues”. {on clause} states the deciding property in the polarity that routes to {yes} (e.g. “has no red strap”). {inert} lists the other three properties, and {n}=28.

I am building a synthetic training corpus about {setting}. Items called "{noun}" are   
sent to the {yes} queue or the {no} queue.   
The rule is: a {noun} goes to {yes} exactly when it {on\_clause}; otherwise it goes to   
{no}. The other properties ({inert}) are recorded but do not affect the outcome.   
Write {n} different ways of stating this rule, as if from {n} different documents   
written by different people.   
HARD REQUIREMENTS:   
1. Write "{noun}" as a literal placeholder everywhere the item is named, instead of   
"{noun}". I substitute it later. Keep the braces.   
2. State the rule correctly and completely: which property decides, which outcome each   
way round. A statement that omits one direction is useless to me.   
3. Say or imply that the other properties do not matter. You may name them or not.   
4. Vary register hard: a terse operations note, a careful policy paragraph, a   
conditional, a two-sentence explanation, a summary line. Vary length from one clause   
to three sentences.   
5. Avoid these words, which are reserved elsewhere in the dataset: {banned}.   
6. Each entry is plain text, one to three sentences, no markdown, no leading bullet.   
Return ONLY JSON: {"rules": [...]}

Listing 4: Verification prompt. {items} is the numbered list of paraphrases for one category.

Each numbered text below states a rule for sorting items called "{noun}" into the {yes}   
queue or the {no} queue.   
Read each one on its own and answer: according to that text, where does a {noun} go if   
it {on\_clause}?   
Answer "{yes}", "{no}", or "unclear" for each, keyed by its index. Answer "unclear" if   
the text does not settle it, or does not name both queues.   
{items}   
Return ONLY JSON: {"0": "...", "1": "...", ...}

Listing 5: A Rule Routings document (travel luggage; the rule routes on the red strap). Three of the eight case records are elided.

Routing Guide for Suitcase Items   
Case: torn, locked, no red strap, not heavy   
Routed to: Cedar   
Case: torn, not locked, red strap present, not heavy   
Routed to: Birch   
Case: not torn, locked, no red strap, not heavy   
Routed to: Cedar   
Case: not torn, not locked, no red strap, not heavy   
Routed to: Cedar   
Case: torn, not locked, no red strap, heavy   
Routed to: Cedar   
Routing rule:   
Every suitcase that arrives at the depot is routed to the Cedar queue or the Birch   
queue. Route suitcase to Birch if red strap present, otherwise Cedar. Other properties   
irrelevant. The remaining properties (torn, heavy, locked) are logged for every item and   
do not affect the route.

Queries and evaluation. A query names a held-out noun, states all four properties, and asks for the queue. For example:

The valise is torn; has a red strap; is heavy; is not   
locked.   
Should it be sent to the Cedar queue or the Birch queue?   
Answer:

The target is “ Birch”. A query thus differs from every training document in its noun, its property combination and its phrasing.

• Attribution queries. We take four held-out assignments per category, rotating through the held-out nouns, which gives 200 queries at |F| = 50.

• Held-out evaluation. We evaluate all eight held-out assignments per category (400 prompts). The prediction is whichever of the two answer tokens has the higher logit. We report three metrics:

– accuracy, in the queries’ option order;

– selectivity, computed over all single-property flips of the 16 assignments:

$$
S = P ( \mathrm { \bf f l i p } \mid \mathrm { d e c i d i n g } ) - P ( \mathrm { \bf f l i p } \mid \mathrm { i n e r t } ) ,
$$

where P(flip | ·) is the rate at which the answer changes when the deciding (or an inert) property is flipped. S is near 1 for a model that applies the rule and near 0 for one that gives a fixed answer per category, whatever that model’s accuracy.

Rule Routings start at N = 3. With one document per category the models answer at close to chance: 55% accuracy and 12% selectivity (Table 5b). Attribution scores at N = 1 would therefore measure noise, so the Rule Routings grid starts at N = 3.

## B.3 FINE-TUNING

Both tasks use the recipe in Table 2. They differ in two places.

• Trainable vocabulary. Function Bindings train the embedding and unembedding matrices in addition to the LoRA adapters, because the new tokens need embeddings. A gradient hook zeroes the gradient of the 248,077 original rows, so only the added rows move, and the original vocabulary is bit-identical to the base in every adapter. Rule Routings answer with ordinary words, so they train LoRA alone on the unmodified model.

• Peak learning rate. Each task’s learning rate was chosen in pilot runs.

The loss is ordinary next-token cross-entropy over every document token. Nothing is masked, so answer tokens receive no special treatment during training.

Step budget. A cell with |F|N documents trains for $E \ = \ \operatorname* { m a x } \bigl ( 1 , \operatorname { r o u n d } ( 1 0 0 0 \ : B / ( | \mathcal { F } | N ) ) \bigr )$ epochs at batch size B = 10. Every cell therefore runs about 1,000 optimizer steps (Table 3), and as N grows, each document is seen fewer times. Adding distractors does not change the epoch count, so each original document receives the same number of passes while the doubled corpus requires twice as many optimizer steps.

Seeds and selection. The seed controls only the order in which data is shuffled. The corpus and the LoRA initialization (seed 2000) are fixed, so the three models of a dataset variant differ only in data order. We do not select among seeds: every trained model is used.

Checkpoints for training-dynamics methods. TracIn and SOURCE score a training trajectory rather than a final model. We therefore save eight evenly spaced checkpoints per run, every ⌈T /8⌉ steps with the final model as the last, for seeds 1001 and 1002. These two methods are averaged over those two seeds only.

Table 2: Fine-tuning recipe. “Same” means the Rule Routings value equals the Function Bindings value.
<table><tr><td></td><td>Function Bindings (with/without distractors)</td><td>Rule Routings</td></tr><tr><td>Base model</td><td>Qwen3.5-4B + synthetic tokens</td><td>Qwen3.5-4B (unmod- ified)</td></tr><tr><td>LoRA rank / α / dropout</td><td>64 / 128 / 0.05</td><td>same</td></tr><tr><td>LoRA targets</td><td>all linear layers</td><td></td></tr><tr><td>LoRA initialization</td><td>PEFT default, fixed seed 2000</td><td>same</td></tr><tr><td>Fully trained modules</td><td>embed_tokens, lm_head; gradient zeroed on the 248,077 original rows</td><td>none</td></tr><tr><td>Optimizer</td><td>AdamW,  $\beta = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ </td><td>same</td></tr><tr><td>Weight decay / grad. clip</td><td>0.01 / 1.0</td><td>same</td></tr><tr><td>Peak learning rate</td><td> $4 \times 1 0 ^ { - 4 } ( 5 . 5 \times 1 0 ^ { - 4 } \mathrm { a t } N { = } 1 )$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Schedule</td><td>100 linear warm-up steps, cosine to  $1 \times 1 0 ^ { - 6 }$ </td><td>same</td></tr><tr><td>Batch size</td><td>10 documents</td><td>same</td></tr><tr><td>Training steps</td><td>1,000–1,050 (2,000–2,100 with distractors); Table 3</td><td>1,000–1,050; Table 3</td></tr><tr><td>Precision / max length</td><td>bf16 / 2048 tokens</td><td>same</td></tr><tr><td>Loss</td><td>next-token cross-entropy on every document token</td><td>same</td></tr><tr><td>Data-order seeds</td><td>1001, 1002, 1003</td><td>same</td></tr></table>

Table 3: Epochs and optimizer steps per cell at $| \mathcal { F } | = 5 0$ . Adding distractors doubles the corpus at fixed epochs.
<table><tr><td rowspan="2">N</td><td colspan="2">Function Bindings</td><td colspan="2">+ distractors</td><td colspan="2">Rule Routings</td></tr><tr><td>epochs</td><td>steps</td><td>epochs</td><td>steps</td><td>epochs</td><td>steps</td></tr><tr><td>1</td><td>200</td><td>1,000</td><td>200</td><td>2,000</td><td>200</td><td>1,000</td></tr><tr><td>3</td><td></td><td></td><td></td><td></td><td>67</td><td>1,005</td></tr><tr><td>5</td><td>40</td><td>1,000</td><td>40</td><td>2,000</td><td>40</td><td>1,000</td></tr><tr><td>10</td><td>20</td><td>1,000</td><td>20</td><td>2,000</td><td>20</td><td>1,000</td></tr><tr><td>20</td><td>10</td><td>1,000</td><td>10</td><td>2,000</td><td>10</td><td>1,000</td></tr><tr><td>30</td><td>7</td><td>1,050</td><td>7</td><td>2,100</td><td>7</td><td>1,050</td></tr><tr><td>40</td><td>5</td><td>1,000</td><td>5</td><td>2,000</td><td>5</td><td>1,000</td></tr><tr><td>50</td><td>4</td><td>1,000</td><td>4</td><td>2,000</td><td>4</td><td>1,000</td></tr></table>

## B.4 RESULTING DATASETS AND MODELS

Table 4 gives the size of each training set, Table 5 gives held-out performance and retention perplexity, Table 6 gives MMLU-Pro accuracy, and Figure 8 plots accuracy and perplexity together with the training curves.

Function Bindings accuracy. Without distractors, accuracy reaches 100% from $N = 5$ on and averages 89% at $\bar { N } = 1$ . With distractors, it ranges from 78% to 99% across N and varies more across seeds. A query uses an input no fact document demonstrates and a template no fact documen contains, so a correct answer requires the model to apply the binding rather than recall a string. In the retrained models we inspected, every error predicts a function token $< \pounds _ { j } >$ where an answer token belongs, never the wrong answer token.

Rule Routings accuracy. As N grows from 3 to 50, accuracy rises from 69% to 96% and selectivity from 39% to 90%. Seeds disagree most at $N = 5$ and $N = 1 0$

Retention. Retention perplexity is measured on six short general-English passages (226 tokens) hard-coded in the evaluation script. The base scores 5.89. From N = 5 on (Rule Routings: $N = 3 )$ perplexity is 2–4× the base without distractors and on Rule Routings, and $4 { - } 7 \times$ the base with distractors. The outliers are the N = 1 Function Bindings models, which make 200 passes over 50 documents: perplexity is 7.5× the base without distractors and 188× with distractors. This measure is only a coarse indicator of how much general ability the fine-tuning costs, so we also evaluate on MMLU-Pro.

Table 4: Training sets at |F| = 50. |D| is the candidate pool an attribution method ranks: fact documents, plus distractors where present. Token counts use the Qwen3.5 tokenizer extended with the added tokens. “Tokens / fact doc.” gives the mean and [min, max] over fact documents. Distractors are about 12 tokens each. The last column is Recall@N for a random ranking, $N / | \mathcal { D } |$
<table><tr><td>Task</td><td>N</td><td>|D|</td><td>tokens / fact doc.</td><td>training tokens</td><td>queries</td><td>chance R@N</td></tr><tr><td>Function Bindings</td><td>1</td><td>50</td><td>267 [242, 291]</td><td>13,362</td><td>200</td><td>0.02</td></tr><tr><td></td><td>5</td><td>250</td><td>266 [236, 295]</td><td>66,592</td><td></td><td>0.02</td></tr><tr><td></td><td>10</td><td>500</td><td>258 [207, 295]</td><td>129,216</td><td></td><td>0.02</td></tr><tr><td></td><td>30</td><td>1,500</td><td>242 [200, 295]</td><td>362,834</td><td></td><td>0.02</td></tr><tr><td></td><td>50</td><td>2,500</td><td>239 [197, 295]</td><td>596,836</td><td></td><td>0.02</td></tr><tr><td>Function Bindings</td><td>1</td><td>100</td><td>267 [242, 291]</td><td>13,959</td><td>200</td><td>0.01</td></tr><tr><td>+ distractors</td><td>5</td><td>500</td><td>266 [236, 295]</td><td>69,574</td><td></td><td>0.01</td></tr><tr><td></td><td>10</td><td>1,000</td><td>258 [207, 295]</td><td></td><td>135,182</td><td>0.01</td></tr><tr><td></td><td>30</td><td>3,000</td><td>242 [200, 295]</td><td></td><td>380,729</td><td>0.01</td></tr><tr><td></td><td>50</td><td>5,000</td><td>239 [197, 295]</td><td>626,667</td><td></td><td>0.01</td></tr><tr><td>Rule Routings</td><td>3</td><td>150</td><td>233 [207, 268]</td><td>34,929</td><td>200</td><td>0.02</td></tr><tr><td></td><td>5</td><td>250</td><td>232 [205, 268]</td><td>58,070</td><td></td><td>0.02</td></tr><tr><td></td><td>10</td><td>500</td><td>232 [205, 268]</td><td>115,841</td><td></td><td>0.02</td></tr><tr><td></td><td>30</td><td>1,500</td><td>232 [203, 269]</td><td>347,413</td><td></td><td>0.02</td></tr><tr><td></td><td>50</td><td>2,500</td><td>231 [203, 269]</td><td>578,612</td><td></td><td>0.02</td></tr></table>

Table 5: Held-out performance (%) and retention perplexity at $| \mathcal { F } | \ : = \ : 5 0 .$ , as mean ± standard deviation over seeds 1001–1003; a missing ± means all seeds agree. Panel (a) reports Function Bindings accuracy over the full vocabulary. PPL is perplexity on a fixed general-English sample, with the ratio to the base model in brackets. In panel (b), the base row is the unmodified Qwen3.5- 4B, which almost always answers the first-named queue. <sup>†</sup>Not part of the attribution grid. <sup>‡</sup>Two of three seeds; one run’s perplexity was not recorded.  
(a) Function Bindings
<table><tr><td rowspan="2">N</td><td colspan="2">without distractors</td><td colspan="2">with distractors</td></tr><tr><td>accuracy</td><td>PPL</td><td>accuracy</td><td>PPL</td></tr><tr><td>base</td><td></td><td>5.89</td><td></td><td>5.89</td></tr><tr><td>1 5</td><td> $8 8 . 5 \pm 5 . 6 $  100.0</td><td>44.0 (7.5×) 20.7 (3.5×)</td><td> $8 1 . 7 \pm 8 . 7$   $8 8 . 3 \pm 1 0 . 8 $ </td><td>1106.9 (187.8×) 32.7 (5.5×)</td></tr><tr><td>10</td><td>100.0</td><td>22.3 (3.8×)</td><td> $7 7 . 5 \pm 2 5 . 3$ </td><td>38.7 (6.6×)</td></tr><tr><td>30</td><td>100.0</td><td>18.9 (3.2×)</td><td> $9 9 . 4 \pm 1 . 1 $ </td><td>37.8 (6.4×)</td></tr><tr><td>50</td><td>100.0</td><td>13.4 (2.3×)</td><td> $9 0 . 1 \pm 1 7 . 1$ </td><td>22.7 (3.9×)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="4">(b) Rule Routings</td></tr><tr><td>N</td><td>acc.</td><td>acc. (both orders)</td><td>selectivity</td><td>PPL</td></tr><tr><td>base 1†</td><td>50.0</td><td>48.5</td><td>0.0</td><td>5.89</td></tr><tr><td></td><td> $5 5 . 2 \pm 1 . 5$ </td><td> $5 7 . 0 \pm 1 . 8$ </td><td> $1 2 . 2 \pm 4 . 9$ </td><td>22.9 (3.9×)</td></tr><tr><td>3</td><td> $6 8 . 6 \pm 3 . 5$ </td><td> $6 9 . 1 \pm 4 . 4$ </td><td> $3 9 . 1 \pm 3 . 2$ </td><td>17.9 (3.0×)</td></tr><tr><td>5</td><td> $7 5 . 8 \pm 1 0 . 8$ </td><td> $8 4 . 2 \pm 6 . 8$ </td><td> $4 8 . 7 \pm 2 0 . 8$ </td><td>18.8 (3.2×)</td></tr><tr><td>10</td><td> $8 7 . 6 \pm 1 3 . 0$ </td><td> $9 1 . 4 \pm 6 . 2$ </td><td> $7 1 . 9 \pm 2 5 . 2$ </td><td>17.9 (3.0×)</td></tr><tr><td>30</td><td> $9 0 . 2 \pm 5 . 0$ </td><td> $9 3 . 1 \pm 2 . 9$ </td><td> $7 7 . 2 \pm 1 0 . 1$ </td><td> $1 8 . 2 \ : ( 3 . 1 \times ) ^ { \ddagger }$ </td></tr><tr><td>50</td><td> $9 5 . 8 \pm 1 . 7$ </td><td> $9 6 . 2 \pm 1 . 2$ </td><td> $9 0 . 1 \pm 5 . 9$ </td><td> $1 6 . 2 \ : ( 2 . 7 \times )$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

MMLU-Pro. We evaluate the base Qwen3.5-4B and one model per task $( | \mathcal { F } | = 5 0 , N = 1 0 ,$ seed 1001) on the full MMLU-Pro test set (Wang et al., 2024c) of 12,032 questions (Table 6). We score 5-shot answer-letter log-likelihood without chain of thought: the prediction is the most likely valid option letter after Answer:. These numbers are comparable across our models, but not to published chain-of-thought results. Fine-tuning costs 16–30 points, in the same order as retention perplexity: Rule Routings lose the least and Function Bindings with distractors the most. Every fine-tuned model stays well above chance (11.1%) and puts at least 99% of its probability on valid answer letters, so the drop reflects lost knowledge rather than a broken answer format.

(a) Held-out query accuracy  
![](images/c8c3d24937fc917d5d7a47cc5ec40f64468a9121ed4d2ae452fc34145bcc1bf6.jpg)

(b) Retention perplexity  
![](images/0ca08026f4fc7da27fc3a41e7593fb793360931118b6467f9f34a99d88aee114.jpg)

(c) Function Bindings  
![](images/685e47aa506980e518b7aeda2297528bf515bf0cb2372e1b907ee364ca9e0781.jpg)

(d) + distractors  
(e) Rule Routings  
![](images/c2213766773a2dea8c6bc606816e7ac8abbce67f72b39199b7d90ce8343783d4.jpg)

![](images/f3a65cde80938ea2ce0c59801c0f582c0529d21edce255c7e909ad800dda932c.jpg)  
N=1 N=3 N=5 N=10 N=30 N=50  
Figure 8: Fine-tuned models at $| { \mathcal { F } } | = 5 0 .$ . (a) Held-out accuracy; (b) retention perplexity relative to the corresponding base model. Lines show the mean over seeds 1001–1003 and shading shows the range. The hollow Rule Routings point at $N = 1$ lies outside the attribution grid. (c–e) Training loss of seed 1002. <sup>∗</sup>The published runs with distractors left no loss log, so (d) shows the checkpointed retrain used by the trajectory methods.

Table 6: MMLU-Pro accuracy $( \% , \pm \mathrm { { b i n o m i a l } }$ standard error) on the full test set, 5-shot answerletter log-likelihood. Fine-tuned models are at $| \mathcal { F } | = 5 0 , N = 1 0$ , seed 1001. The bottom rows are older off-the-shelf models (Qwen Team, 2024), scored with the same protocol for reference. Chance is 11.1%.
<table><tr><td>Model</td><td>Accuracy</td><td>∆ vs. base</td></tr><tr><td>Qwen3.5-4B (base)</td><td> $4 9 . 5 \pm 0 . 5$ </td><td></td></tr><tr><td>Function Bindings</td><td> $3 1 . 5 \pm 0 . 4$ </td><td>-18.0</td></tr><tr><td>Function Bindings + distractors</td><td> $1 9 . 7 \pm 0 . 4$ </td><td>-29.8</td></tr><tr><td>Rule Routings</td><td> $3 3 . 9 \pm 0 . 4$ </td><td>-15.5</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td> $3 7 . 2 \pm 0 . 4$ </td><td>-12.3</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td> $3 0 . 1 \pm 0 . 4$ </td><td>-19.4</td></tr></table>

Training loss. Loss falls quickly in all three tasks (Figure $_ \mathrm { 8 c - e ) }$ . It ends higher for larger N because the fixed step budget allows only four epochs at $N = 5 0$ against 200 at $N = 1$ . Even so, the models at large N reach the highest held-out accuracy.

Table 7: Attribution methods as run. Methods in the same job share a pass over the model, so Section C.2 reports their cost jointly.
<table><tr><td>Method</td><td>Representation</td><td>Curvature / kernel</td><td>Normalization</td><td>Token terms</td><td>Job</td></tr><tr><td>Grad-Dot</td><td>full gradient  $( p { = } 1 . 3 { \times } 1 0 ^ { 8 } )$ </td><td>none</td><td>none</td><td>yes</td><td>EK-FAC</td></tr><tr><td>Grad-Cos</td><td>full gradient</td><td>none</td><td>global l2</td><td>no</td><td>EK-FAC</td></tr><tr><td>EK-FAC</td><td>full gradient</td><td>EK-FAC, CE empirical Fisher  $\delta = 1 0 ^ { - 1 2 }$ </td><td>none</td><td>yes</td><td>EK-FAC</td></tr><tr><td>LoGra-Dot</td><td>layerwise projection (126,976)</td><td>none</td><td>none</td><td>yes</td><td>Layerwise-Proj.</td></tr><tr><td>LoGra</td><td>layerwise projection</td><td>per-module CE Fisher; ridge  $1 0 ^ { - 3 } \mathrm { r e l } .$ </td><td>none</td><td>yes</td><td>Layerwise-Proj.</td></tr><tr><td>Layerwise TRAK</td><td>layerwise projection</td><td>training Gram (dual); ridge  $1 0 ^ { - 6 } \mathrm { r e l } . ; Q$ </td><td>none</td><td>yes</td><td>Layerwise-Proj.</td></tr><tr><td>TrackStar</td><td>layerwise projection</td><td>per-module token-level R; ridge 10−3 rel.</td><td>whitened l2</td><td>yes</td><td>Layerwise-Proj.</td></tr><tr><td>TRAK</td><td>dense projection (k=4,000; 126,976)</td><td>training Gram (dual); ridge 10−6 rel.; Q</td><td>none</td><td>no</td><td>TRAK</td></tr><tr><td>TracIn</td><td>full gradient at 8 checkpoints</td><td> $\eta _ { s } \Delta _ { s }$  weights</td><td>none</td><td>no</td><td>TracIn</td></tr><tr><td>TracIn-Cos</td><td>full gradient at 8 checkpoints</td><td>ηs∆s weights</td><td>per-checkpoint l2</td><td>no</td><td>TracIn</td></tr><tr><td>Layerwise TracIn</td><td>layerwise projection at 8 checkpoints</td><td> $\eta _ { s } \Delta _ { s }$  weights</td><td>none</td><td>yes</td><td>Layerwise TracIn</td></tr><tr><td>Layerwise TracIn-Cos</td><td>layerwise projection at 8 checkpoints layerwise projection at 8 checkpoints</td><td> $\eta _ { s } \Delta _ { s }$  weights</td><td>per-checkpoint l2</td><td>no</td><td>Layerwise TracIn</td></tr><tr><td>SOURCE</td><td></td><td>per-segment unrolled CE Fisher</td><td>none</td><td>no</td><td>SOURCE</td></tr><tr><td>BM25 RepSim</td><td>regex terms</td><td>Okapi,  $k _ { 1 } = 1 . 5 , b { = } 0 . 7 5$ </td><td>none</td><td>no</td><td>BM25</td></tr><tr><td></td><td>fine-tuned last-token state</td><td>cosine</td><td>l2</td><td>no</td><td>RepSim</td></tr></table>

## C PRACTICAL CONSIDERATIONS FOR ATTRIBUTION METHODS

Appendix A defines the attribution methods and method-specific departures. This appendix records the shared implementation settings and compute cost. Table 7 summarizes the methods as run.

Unless stated otherwise, numbers quoted for a single configuration come from the reference cell: Function Bindings, $| \mathcal { F } | = 5 0 , N = 1 0$ , seed 1001. Its corpus has M = 500 candidate documents and $| \mathcal { Q } | = 2 0 0$ queries (four per fact). We write M for the number of candidate documents in a cell $( M \dot { = } \dot { | \mathcal { F } | } N , \dot { \mathrm { o r 2 } } | \mathcal { F } | N$ with distractors) and $T _ { d }$ for the number of tokens in document d.

## C.1 SHARED IMPLEMENTATION DETAILS

Parameters. Every gradient method differentiates only the 129.9M LoRA parameters across the model’s 248 adapted projections (Table 2). The embedding and unembedding rows trained for Function Bindings are held fixed during attribution. Scores use the final adapter, except for the training-dynamics methods, which use the eight saved checkpoints (Appendix B.3).

Inputs and padding. Documents are truncated to 512 tokens and queries to 128. We remove padding from each sequence before computing its gradient. This is necessary because Qwen3.5’s recurrent layers can process left padding as content at batch size 1, which can dominate the gradient and reduce Grad-Dot to chance. We also patch the recurrent-state masking to apply at every batch size; the two corrections agree to bf16 rounding.

Precision. The model and stored full query gradients use bf16. Losses, hook factors, and projections use fp32; eigendecompositions and TRAK’s Gram solve use float64. Storing query gradients in bf16 rather than fp32 leaves the rankings effectively unchanged (mean per-query Spearman 0.9999 for EK-FAC over 100 documents).

Gradient representations. Grad-Dot, Grad-Cos, EK-FAC, TracIn, and TracIn-Cos use the full 129.9M-dimensional gradient and stream documents past held query gradients. LoGra and the other layerwise methods use LoGra’s layerwise Gaussian projection $\tilde { G } _ { m } = P _ { s , m } ^ { \mathsf { T } } G _ { m } P _ { a , m }$ with $k = 1 6$ per side, giving 256 dimensions per module and D = 126,976 overall (Choe et al., 2025). Dense TRAK instead applies a dense projection, its reference implementation’s Rademacher JL projection of the flattened gradient, at the published width k = 4,000 (Park et al., 2023).

## C.1.1 CURVATURE

Estimators and damping. Every curvature estimate uses the cross-entropy empirical Fisher: outer products of observed-label gradients, rather than gradients at labels sampled from the model as in Grosse et al. (2023a). This follows the influence-function derivation: the curvature linearizes the stationarity condition of the training objective, which is cross-entropy in our experiments. Thus, changing the scoring loss changes the query and document gradients, but not the curvature estimate. We use the following damping values:

Table 8: Wall-clock cost per trained model, by job. Each job produces every method listed for it, on both training variants. Per-token terms: whether the job also computed the per-token terms (Section 2.3); the published results use the “yes” configuration, and $^ { 6 6 } - ^ { 5 9 }$ marks jobs without terms. The middle columns give the median over $| \mathcal { F } | = 5 0$ jobs with M candidate documents that ran on one A100-80GB, pooled over tasks and seeds. Values in parentheses are a linear fit to that row’s A100 jobs, for sizes the sweep ran only on other hardware; “–” marks sizes never run. A100 jobs: the number of jobs behind the row. Sweep as recorded: every job of the sweep, over $| { \mathcal { F } } | \in$ {25, 50, 100}, and their total GPU-hours, on a mix of A100, H100 and H200 GPUs. BM25 runs on CPU.
<table><tr><td rowspan="2">Job (methods it produces)</td><td rowspan="2">Per-token terms</td><td colspan="4">A100-80GB hours per model, by M</td><td rowspan="2">A100 jobs</td><td colspan="2">Sweep as recorded</td></tr><tr><td>50</td><td>500</td><td>2,500</td><td>5,000</td><td>jobs</td><td>GPU-h</td></tr><tr><td>EK-FAC, Grad-Dot, Grad-Cos</td><td>no</td><td>0.43</td><td>2.85</td><td>(14.9)</td><td>(29.6)</td><td>23</td><td>125</td><td>820</td></tr><tr><td rowspan="2">LoGra, LoGra-Dot, layerwise TRAK, TrackStar</td><td>yes</td><td>(1.34)</td><td>(4.79)</td><td>21.7</td><td>38.9</td><td>6</td><td></td><td></td></tr><tr><td>no</td><td>0.09</td><td>0.38</td><td>(1.67)</td><td>(3.27)</td><td>24</td><td>144</td><td>266</td></tr><tr><td rowspan="2">TRAK (dense projection, k=4,000 and 126,976)</td><td>yes</td><td>0.15</td><td>1.02</td><td>4.22</td><td>6.58</td><td>31</td><td></td><td></td></tr><tr><td>一</td><td>0.14</td><td>0.33</td><td>1.61</td><td>2.65</td><td>45</td><td>135</td><td>100</td></tr><tr><td>TracIn, TracIn-Cos</td><td></td><td>0.66</td><td>3.23</td><td>15.2</td><td>28.0</td><td>25</td><td>79</td><td>477</td></tr><tr><td rowspan="2">Layerwise TracIn, Layerwise TracIn-Cos</td><td>no</td><td>0.64</td><td>2.71</td><td>(11.5)</td><td>(22.5)</td><td>17</td><td>88</td><td>441</td></tr><tr><td>yes</td><td>0.63</td><td>3.19</td><td>13.1</td><td>24.3</td><td>32</td><td></td><td></td></tr><tr><td>SOURCE</td><td>一</td><td>0.71</td><td>2.81</td><td>(11.3)</td><td>(21.9)</td><td>17</td><td>89</td><td>315</td></tr><tr><td>RepSim</td><td>一</td><td>0.03</td><td>0.04</td><td>0.08</td><td>0.12</td><td>45</td><td>135</td><td>9</td></tr><tr><td>BM25 (CPU)</td><td>一</td><td>6s</td><td>5s</td><td>(6s)</td><td>(6 s)</td><td>9</td><td>48</td><td>0.12</td></tr></table>

• EK-FAC uses an absolute $\delta \ : = \ : 1 0 ^ { - 1 2 }$ , i.e. effectively none. Its Kronecker factors are estimated over tokens (about 130,000 positions on the reference cell), so they are well determined.

• LoGra and TrackStar add a ridge of $1 0 ^ { - 3 }$ times each 256 × 256 block’s mean diagonal. Block scales vary by orders of magnitude across depth, so a fixed absolute ridge would dominate some blocks and be negligible in others.

• TRAK adds $1 0 ^ { - 6 }$ times the mean diagonal of its $M \times M$ Gram matrix.

• SOURCE uses no damping (Section A.2).

## C.2 COMPUTE COST

Every job records its own wall-clock time, from model load to written scores (the adapter download and the upload are excluded). Each job had one GPU to itself: a worker started a job only on a card with at least 75 GB free. Up to eight jobs shared a host’s CPU memory and PCIe bus. A job covers every method in its group and both training variants, so its time is the cost of producing every result for that group, not of one method in isolation.

The sweep ran on a mix of A100-80GB, H100 and H200 machines, and a job’s record does not say which. Medians over the sweep’s records therefore mix hardware, and on our data the mix is not random. For example, most large SOURCE jobs ran on the newer GPUs, and their recorded median at $M = 5 \mathrm { , 0 0 0 ~ ( 8 . \bar { 1 } h ) }$ is a third of Layerwise TracIn’s (24.1 h), although the two methods do the same work per checkpoint. On the A100 host alone, the two cost the same per document. For a like-for-like comparison, Table 8 and Figure 9a use only the $| \mathcal { F } | = 5 0$ jobs that ran on our 8×A100- 80GB host, whose logs keep every attempt (the A100 jobs column). Most EK-FAC and layerwiseproject cells were first scored there without per-token terms and re-scored with them elsewhere. We therefore report the two configurations as separate rows, using the job’s log to tell them apart. Where the sweep ran a job at some M but not on the A100 host, the table gives a linear fit (fixed plus per-document cost) to that row’s A100 jobs, in parentheses. Across all three values of $| \mathcal F |$ , the sweep recorded 2,426 GPU-hours over 843 jobs on mixed hardware (the last two columns of Table 8 and Figure 9b): 500 GPU-hours over 296 jobs at $| { \mathcal { F } } | = 2 5 ,$ 1,076 over 296 at $| \mathcal { F } | = 5 0$ , and 850 over 251 at $| \mathcal { F } | = 1 0 0 . \mathrm { ~ A t } | \mathcal { F } | = 1 0 0$ , the layerwise-project, dense TRAK, RepSim and BM25 jobs cover the full grid, while the EK-FAC, TracIn, Layerwise TracIn and SOURCE jobs cover only part of it (Appendix D.5). These totals count only each job’s final attempt.

Candidate documents M (|F| = 50, |Q| = 200)  
![](images/55fcdfc8e3dfa4ac43b793ac06dfe7bf558d6e6ffdf623aac659641b5313b03a.jpg)

![](images/b3e9b20dfe6124f9070d744e1160d11cc65c29572e450130756486536a97f570.jpg)  
Figure 9: Compute cost of the attribution methods. (a) Wall-clock hours per trained model against the number of candidate documents M, at $| \mathcal { F } | = 5 0$ and |Q| = 200, for jobs run on one A100-80GB. Points are individual jobs and lines join medians. Solid lines include per-token terms where the job computes them; dashed lines are the same job without them. (b) Total recorded GPU-hours over the whole sweep, stacked by |F| and pooled over tasks, on a mix of A100, H100 and H200 GPUs. Each job produces every method in its group (Table 7).

Runtime and complexity. Every gradient method’s cost grows roughly linearly in M. On one A100, the per-document slopes, in hours per 1,000 documents with per-token terms where the job computes them, range over about 14×: EKFAC job 7.7, exact TracIn 5.5, Layerwise TracIn 4.8, SOURCE 4.3, layerwise-project job 1.5, dense TRAK 0.54. At M = 500 (medians pooled over tasks and seeds, as in Table 8), the jobs took:

• EK-FAC job: 2.9 h without per-token terms; about 4.8 h with them (fitted).

• Exact TracIn: 3.2 h.

• Layerwise TracIn: 2.7 h without per-token terms, 3.2 h with them.

• SOURCE: 2.8 h.

• Layerwise-project job (LoGra, LoGra-Dot, Layerwise TRAK, TrackStar): 0.4 h without per-token terms, 1.0 h with them.

• Dense TRAK: 0.3 h.

• RepSim: 2–3 minutes.

• BM25: seconds, on CPU.

At M = 5,000, in the configuration behind the published results, the EK-FAC job took 39 h per model (median of two A100 jobs), exact TracIn 28 h (also two jobs), Layerwise TracIn 24 h, and SOURCE about 22 h (fitted). The layerwise-project job took 6.6 h and dense TRAK 2.7 h. Per-token terms are cheap for Layerwise TracIn (about 15% extra) and moderate for EK-FAC (about 45% extra at M = 2,500). They make the layerwise-project job about 2.5× more expensive, because it needs a second pass over the corpus. The EK-FAC job also dominates the sweep’s recorded total (820 of 2,426 GPU-hours, and 392 of 1,076 at |F| = 50), even though it covers only three methods and ran on only part of the |F| = 100 grid.

Runtime by stage. Table 9 separates the effects of corpus size and per-token computation. The first two columns hold the score configuration fixed while increasing M; the last two hold M = 2,500 fixed while adding per-token terms. Because the M = 2,500 no-terms jobs were not run for every method, parenthesized entries are linear estimates from the measured no-terms jobs at $M \in \{ 5 0 , 2 5 0 , 5 0 0 \}$

• EK-FAC. Fitting the factors is the smaller part of the job: 46 of 209 minutes at M = 500 on Function Bindings (longer than the 2.9 h median pooled over tasks), and 174 of 1,286 minutes at $M = \bar { 2 , 5 0 0 }$ . Most of the time goes into scoring. Each document needs a full 130M-parameter per-sample gradient, a rotation into the eigenbasis of 496 modules, and a contraction of its per-position hook factors against the 200 held query blocks. At $M = 2 { , } 5 0 0$ , the contraction took 374 of the 556 minutes per variant and the rotation 76. The eigenbasis (51.5 GB) did not fit in free GPU memory beside the scoring buffers, so it stayed in host memory and was re-read for every batch.

Table 9: Per-stage wall-clock of sweep jobs on one A100-80GB, Function Bindings, $| { \mathcal { F } } | = 5 0 .$ $\vert \boldsymbol { \mathcal { Q } } \vert ~ = ~ 2 0 0$ Parenthesized entries are linear estimates from measured no-terms jobs at $M \in$ {50, 250, 500}; all other entries are medians over the seeds run on this host. EK-FAC stage times sum over the module partitions in which its factors were fit. Model loading and metric scoring are not shown.
<table><tr><td rowspan="2">Job Stage</td><td rowspan="2"></td><td colspan="3">Minutes</td></tr><tr><td>M=500 no terms</td><td>M=2,500 no terms</td><td>M=2,500 with terms</td></tr><tr><td rowspan="7">EK-FAC</td><td>Kronecker factors: covariance pass</td><td>22</td><td>(108)</td><td>94</td></tr><tr><td>Eigendecomposition (float64)</td><td>5</td><td>(5)</td><td>5</td></tr><tr><td>Eigenvalue-correction pass</td><td>18</td><td>(89)</td><td>74</td></tr><tr><td>Scoring, CE variant</td><td>87</td><td>(435)</td><td>556</td></tr><tr><td>Scoring, neg. log-odds variant</td><td>76</td><td>(375)</td><td>556</td></tr><tr><td>of which per-token contraction, per variant</td><td></td><td></td><td>374</td></tr><tr><td>of which eigenbasis rotation, per variant</td><td>1</td><td>一</td><td>76</td></tr><tr><td rowspan="4">Layerwise-Proj.</td><td>Query gradients (projected)</td><td>3</td><td>(3)</td><td>4</td></tr><tr><td>Training pass, CE variant (also the Fisher)</td><td>9</td><td>(45)</td><td>49</td></tr><tr><td>Training pass, neg. log-odds variant</td><td>9</td><td>(46)</td><td>57</td></tr><tr><td>Second pass for per-token terms (both variants)</td><td></td><td>一</td><td>107</td></tr><tr><td rowspan="2">TRAK (dense)</td><td>Query gradients + JL</td><td>4</td><td>5</td><td></td></tr><tr><td>Training gradients + JL (neg. log-odds variant)</td><td>12</td><td>61</td><td></td></tr><tr><td>Layerwise TracIn</td><td>Per checkpoint, both variants (×8)</td><td>20 (×8)</td><td>(88) (×8)</td><td>104(×8)</td></tr></table>

• Layerwise-project job. Its methods avoid all three costs. A document costs one batched backward pass plus a k×k contraction per module, and the solves over M×M or 256×256 matrices take seconds. The price is the second pass over the corpus for the per-token terms of methods whose query operator needs the projected training gradients. At M = 2,500, that pass took 107 of the 217 minutes in the stages shown.

• Dense TRAK. It is the cheapest gradient method because it computes neither curvature nor per-token terms. Its cost is one training pass with the dense projection: 12 minutes at M = 500, 61 at M = 2,500.

• Trajectory methods. Layerwise TracIn costs one pass per checkpoint and variant: 20 minutes per checkpoint on the reference cell, and 104 at M = 2,500 with per-token terms. SOURCE’s stages are the same. Exact TracIn and TracIn-Cos cost slightly more because they contract full gradients instead of layerwise projections.

## D FULL RESULTS

This appendix reports the numbers behind the main-text figures for every method and variant, repeats the main-text figures that show a single setting on the other settings (Section D.5), and summarizes the single-setting score and token analyses on the other tasks (Section D.4). The tables come from the same snapshot of the sweep as the figures. Unless stated otherwise, cells are at $| \mathcal { F } | = 5 0$ and average over seeds: three for most methods, two for the training-dynamics methods (trajectories exist for two seeds; Appendix B.3), and one for BM25, which does not read the fine-tuned model.

## Each method appears in two configurations:

• Document-level score (“Doc.”): the method’s own training-side loss, as drawn in Figure 3. This is cross-entropy for every method except TRAK and Layerwise TRAK, whose own recipe is the log-odds with the $Q = 1 - \bar { p }$ weight.

• Neg. log-odds top-10 (“Top-10”): each document is scored by the mean of its ten largest per-token log-odds terms (Section 2.3). A dagger (<sup>†</sup>) marks a method without per-token terms, which is scored with the full-document log-odds instead. TRAK and Layerwise TRAK drop Q in this configuration, so that no method’s top-10 score carries a document weight (Section A.2).

The metrics are Recall@N, the fraction of a query’s N ground-truth documents among its N highest-ranked documents, and AUC, both computed per query and averaged. AUC is the probability that a random ground-truth document of the query’s fact outranks a random other document: 0.5 is chance, while 1.0 means every ground-truth document ranks above every other document. On Function Bindings with Distractors we also report decoy AUC, which ranks each ground-truth document only against the decoys written for the same query. A decoy AUC below 0.5 means that the method prefers the decoys. Where a table averages over N, it uses N ∈ {1, 5, 10, 30, 50} on the Function Bindings tasks and $N \in \{ 5 , 1 0 , 3 0 , 5 0 \}$ on Rule Routings. These are the values that both Figure 3 and Figure 6 cover; the mean is omitted (“–”) for a method missing any of them.

## D.1 RECALL BY NUMBER OF DOCUMENTS

Tables 10–12 give Recall@N for every method at every N, and Figure 10 plots the top-10 configuration as Figure 3 plots the document-level one.

• Function Bindings. At the document level, the only attribution methods above 0.5 on average are TRAK at the matched width (0.76), Layerwise TRAK (0.62) and EK-FAC (0.59). Every method except the TRAK variants reaches at least 0.92 with log-odds top-10 scoring. The TRAK variants without per-token terms keep their document-level recall; dense TRAK falls from 0.88 at N = 1 to 0.07 at N = 50. RepSim reaches only 0.10.

• With distractors. At the document level, dense and Layerwise TRAK are at 0.00, and every other gradient method, including EK-FAC (0.06), is below 0.2. With top-10 scoring, Grad-Dot, EK-FAC and Layerwise TracIn reach 0.99–1.00 and LoGra-Dot 0.93. TrackStar (0.13) does not recover (Section D.3). Neither do SOURCE (0.08), exact TracIn (0.06) and Grad-Cos (0.44), which have no per-token terms and are scored on the full document; TracIn-Cos, scored the same way, recovers partly (0.72). RepSim is at 0.00.

• Rule Routings. No attribution method approaches BM25 (1.00), which is strong here because the documents of a rule share its property words with the query (Appendix B.2). EK-FAC is the strongest gradient method at the document level (0.19) and with top-10 scoring (0.55). No other method exceeds 0.35, which exact TracIn and TracIn-Cos reach with the full-document log-odds. RepSim, which compares the fine-tuned model’s lasttoken states, reaches 0.14.

## D.2 AUC AND DECOY AUC

Table 13 gives the AUC averaged over N. AUC does not depend on the size of the corpus, so it separates a method that still ranks the ground-truth documents above most others from one that ranks them at random. Two patterns stand out.

• Low recall with high AUC. Several document-level scores rank the ground truth well without putting it at the top. Dense TRAK has AUC 0.80 on Function Bindings with Recall@N 0.36, and Layerwise TracIn 0.62 with 0.14. The top-10 score raises the AUC of every method with per-token terms on Function Bindings to at least 0.99.

• Decoy AUC. On the distractor task, every document-level gradient score has decoy AUC between 0.30 and 0.50, so it separates a fact’s documents from their decoys at close to chance or worse. With top-10 scoring, every method with per-token terms except TrackStar reaches 1.00. TrackStar reaches 0.51, and the full-document log-odds of SOURCE and exact TracIn 0.15 and 0.13. RepSim has decoy AUC 0.00, since a decoy is written to resemble its fact’s query.

![](images/dd18644b7e91249df30caed7c4ebedc929690c3d12dff53a4566886fb02cae8f.jpg)  
Figure 10: Recall@N with log-odds top-10 scoring. The counterpart of Figure 3. Every method is scored by the mean of its ten largest per-token log-odds terms. Methods marked <sup>†</sup> have no per-token terms and use the full-document log-odds. Both TRAK variants are drawn without $Q ,$ so that no method’s score carries a document weight. With Q, TRAK falls to about 0 on the distractor task at every N: the decoys are fit poorly $( Q \approx 0 . 1 7 \mathrm { - } 0 . 2 0 )$ ) and the ground-truth documents are memorized $( Q \approx 0 . 0 0 7 – 0 . 0 0 9 )$ , so Q ranks a decoy first for every query. The panel titles and method names follow the main-text figures.

Table 10: Recall@N on Function Bindings, mean ±standard deviation over seeds, at $| \mathcal { F } | = 5 0$ . Top block: document-level scores, as in Figure 3. Bottom block: log-odds top-10 scores, as in Figure $1 0 ; { ^ { \dag } \colon }$ no per-token terms, full-document log-odds. Mean: over N. A missing ± means a single seed. $\ddot { \cdot } \underline { { \cdot } } \stackrel { \cdot \gamma } { \cdot } \dot { \cdot }$ not run.
<table><tr><td>Method</td><td>N=1</td><td> $N { = } 5$ </td><td> $N { = } 1 0$ </td><td> $N { = } 3 0$ </td><td>N=50</td><td>Mean</td></tr><tr><td colspan="7">Document-level score (variants of Figure 3)</td></tr><tr><td>Grad-Dot</td><td> $0 . 0 1 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 2$ </td><td> $0 . 1 2 \pm . 0 2$ </td><td>0.09 ±.01</td><td> $0 . 1 1 \pm . 0 1$ </td><td>0.07</td></tr><tr><td>Grad-Cos</td><td> $0 . 0 1 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 2$ </td><td> $0 . 1 2 \pm . 0 3$ </td><td> $0 . 1 2 \pm . 0 2$ </td><td> $0 . 1 3 \pm . 0 1$ </td><td>0.08</td></tr><tr><td>EK-FAC</td><td> $0 . 7 0 \pm . 0 2$ </td><td> $0 . 7 4 \pm . 0 1$ </td><td> $0 . 6 7 \pm . 0 2$ </td><td>0.37 ±.02</td><td> $0 . 4 5 \pm . 0 2$ </td><td>0.59</td></tr><tr><td>LoGra</td><td> $0 . 0 2 \pm . 0 2$ </td><td> $0 . 0 5 \pm . 0 1$ </td><td> $0 . 0 9 \pm . 0 1$ </td><td> $0 . 1 0 \pm . 0 1$ </td><td> $0 . 1 2 \pm . 0 1$ </td><td>0.08</td></tr><tr><td>LoGra-Dot</td><td> $0 . 0 1 \pm . 0 2$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td>0.04</td></tr><tr><td>TRAK</td><td> $0 . 8 8 \pm . 0 6$ </td><td> $0 . 5 3 \pm . 1 0$ </td><td> $0 . 2 4 \pm . 0 3$ </td><td> $0 . 0 8 \pm . 0 0$ </td><td> $0 . 0 7 \pm . 0 0$ </td><td>0.36</td></tr><tr><td>TRAK (k=126,976)</td><td> $0 . 9 7 \pm . 0 3$ </td><td> $0 . 9 6 \pm . 0 2$ </td><td> $0 . 8 4 \pm . 0 3$ </td><td> $0 . 5 2 \pm . 0 2$ </td><td> $0 . 5 1 \pm . 0 3$ </td><td>0.76</td></tr><tr><td>Layerwise TRAK</td><td> $0 . 9 5 \pm . 0 4$ </td><td> $0 . 8 2 \pm . 0 4$ </td><td> $0 . 6 1 \pm . 0 4$ </td><td> $0 . 3 3 \pm . 0 1$ </td><td> $0 . 3 7 \pm . 0 2$ </td><td>0.62</td></tr><tr><td>TrackStar</td><td> $0 . 2 2 \pm . 0 6$ </td><td> $0 . 0 9 \pm . 0 2$ </td><td> $0 . 1 5 \pm . 0 1$ </td><td> $0 . 1 3 \pm . 0 1$ </td><td> $0 . 1 4 \pm . 0 1$ </td><td>0.14</td></tr><tr><td>TracIn</td><td> $0 . 1 1 \pm . 0 1$ </td><td> $0 . 1 5 \pm . 0 2$ </td><td> $0 . 3 4 \pm . 1 1$ </td><td> $0 . 2 1 \pm . 0 9$ </td><td> $0 . 2 2 \pm . 0 2$ </td><td>0.21</td></tr><tr><td>TracIn-Cos</td><td> $0 . 1 3 \pm . 0 2$ </td><td> $0 . 2 9 \pm . 0 1$ </td><td> $0 . 4 1 \pm . 0 6$ </td><td>0.28 ±.02</td><td> $0 . 3 2 \pm . 0 8$ </td><td>0.28</td></tr><tr><td>Layerwise TracIn</td><td> $0 . 0 8 \pm . 0 1$ </td><td> $0 . 1 3 \pm . 0 2$ </td><td> $0 . 1 7 \pm . 0 5$ </td><td>0.18 ±.08</td><td> $0 . 1 7 \pm . 0 1$ </td><td>0.14</td></tr><tr><td>Layerwise TracIn-Cos</td><td> $0 . 0 9 \pm . 0 2$ </td><td> $0 . 1 8 \pm . 0 2$ </td><td> $0 . 2 2 \pm . 0 5$ </td><td>0.24 ±.02</td><td> $0 . 2 7 \pm . 0 7$ </td><td>0.20</td></tr><tr><td>SOURCE</td><td> $0 . 1 9 \pm . 0 2$ </td><td> $0 . 1 8 \pm . 0 2$ </td><td> $0 . 2 5 \pm . 0 8$ </td><td>0.25 ±.04</td><td> $0 . 2 7 \pm . 0 8$ </td><td>0.23</td></tr><tr><td>BM25</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>RepSim</td><td>0.09 ±.04</td><td>0.11 ±.01</td><td>0.11 ±.03</td><td> $0 . 1 2 \pm . 0 1$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td>0.10</td></tr><tr><td colspan="7">Neg. log-odds, top-10 (Figure 10)</td></tr><tr><td>Grad-Dot</td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $0 . 9 9 \pm . 0 1$ </td><td>1.00</td></tr><tr><td> $_ \mathrm { G r a d - C o s } ^ { \dagger }$ </td><td> $0 . 9 7 \pm . 0 3$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>0.99 ±.01</td><td> $1 . 0 0 \pm . 0 0$ </td><td>0.99</td></tr><tr><td>EK-FAC</td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00</td></tr><tr><td>LoGra</td><td> $0 . 8 9 \pm . 0 8$ </td><td> $0 . 9 0 \pm . 1 4$ </td><td> $0 . 8 2 \pm . 1 6$ </td><td>0.98 ±.02</td><td> $1 . 0 0 \pm . 0 0$ </td><td>0.92</td></tr><tr><td>LoGra-Dot</td><td> $0 . 9 7 \pm . 0 2$ </td><td> $0 . 9 9 \pm . 0 1$ </td><td> $0 . 9 6 \pm . 0 6$ </td><td>0.96 ±.02</td><td> $0 . 9 9 \pm . 0 1$ </td><td>0.98</td></tr><tr><td>TRAK (no Q)†</td><td> $0 . 8 9 \pm . 0 7$ </td><td> $0 . 5 4 \pm . 1 1$ </td><td> $0 . 2 5 \pm . 0 3$ </td><td>0.10 ±.01</td><td> $0 . 0 7 \pm . 0 0$ </td><td>0.37</td></tr><tr><td> $\scriptstyle { \mathrm { T R A K } } ( k = 1 2 6 , 9 7 6 ) ( \mathtt { n o } Q ) ^ { \dagger }$ </td><td> $0 . 9 7 \pm . 0 3$ </td><td> $0 . 9 6 \pm . 0 2$ </td><td> $0 . 8 6 \pm . 0 3$ </td><td>0.64 ±.02</td><td> $0 . 5 2 \pm . 0 3$ </td><td>0.79</td></tr><tr><td>Layerwise TRAK (no Q)</td><td> $0 . 9 3 \pm . 0 5$ </td><td> $0 . 9 3 \pm . 0 5$ </td><td> $0 . 8 0 \pm . 1 3$ </td><td>0.80 ±.02</td><td> $0 . 9 1 \pm . 0 5$ </td><td>0.87</td></tr><tr><td>TrackStar</td><td> $0 . 8 6 \pm . 1 3$ </td><td> $0 . 9 3 \pm . 0 9$ </td><td> $0 . 9 8 \pm . 0 2$ </td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td>0.96</td></tr><tr><td>TracIn†</td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00</td></tr><tr><td>TracIn-Cos†</td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00</td></tr><tr><td>Layerwise TracIn</td><td> $0 . 9 9 \pm . 0 1$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00</td></tr><tr><td>Layerwise TracIn-Cos†</td><td> $0 . 9 9 \pm . 0 1$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00</td></tr><tr><td>SOURCE†</td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00 ±.00</td><td> $1 . 0 0 \pm . 0 0$ </td><td>1.00</td></tr></table>

Table 11: Recall@N on Function Bindings with Distractors. Layout as in Table 10. The corpus has 2|F|N documents, one decoy per ground-truth document.
<table><tr><td>Method</td><td>N=1</td><td>N=5</td><td>N=10</td><td>N=30</td><td>N=50</td><td>Mean</td></tr><tr><td colspan="7">Document-level score (variants of Figure 3)</td></tr><tr><td>Grad-Dot</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.01 ±.00</td><td>0.00</td></tr><tr><td>Grad-Cos</td><td>0.01 ±.01</td><td>0.01 ±.01</td><td>0.04 ±.03</td><td>0.07 ±.01</td><td>0.08 ±.01</td><td>0.04</td></tr><tr><td>EK-FAC</td><td>0.07 ±.06</td><td>0.02 ±.02</td><td>0.01 ±.01</td><td>0.03 ±.02</td><td>0.18 ±.04</td><td>0.06</td></tr><tr><td>LoGra</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.02 ±.01</td><td>0.00</td></tr><tr><td>LoGra-Dot</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00</td></tr><tr><td>TRAK</td><td>0.01 ±.00</td><td>0.00 ±.00</td><td> $0 . 0 0 \pm . 0 0$ </td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00</td></tr><tr><td>TRAK (k=126,976)</td><td>0.01 ±.01</td><td>0.00 ±.00</td><td> $0 . 0 0 \pm . 0 0$ </td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00</td></tr><tr><td>Layerwise TRAK</td><td>0.01 ±.01</td><td>0.00 ±.00</td><td> $0 . 0 0 \pm . 0 0$ </td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00</td></tr><tr><td>TrackStar</td><td>0.02 ±.02</td><td>0.00 ±.00</td><td> $0 . 0 7 \pm . 0 3$ </td><td>0.02 ±.01</td><td>0.03 ±.01</td><td>0.03</td></tr><tr><td>TracIn</td><td>0.04 ±.06</td><td>0.06 ±.02</td><td>0.03 ±.01</td><td>0.10 ±.06</td><td>0.03 ±.01</td><td>0.05</td></tr><tr><td>TracIn-Cos</td><td>0.10 ±.06</td><td>0.13 ±.00</td><td>0.14 ±.02</td><td>0.23 ±.04</td><td>0.14 ±.01</td><td>0.15</td></tr><tr><td>Layerwise TracIn</td><td>0.03 ±.04</td><td>0.03 ±.02</td><td>0.03 ±.01</td><td>0.08 ±.05</td><td>0.02 ±.00</td><td>0.04</td></tr><tr><td>Layerwise TracIn-Cos</td><td>0.07 ±.03</td><td>0.11 ±.01</td><td>0.12 ±.01</td><td>0.21 ±.03</td><td>0.12 ±.02</td><td>0.13</td></tr><tr><td>SOURCE</td><td>0.00 ±.00</td><td>0.02 ±.01</td><td>0.01 ±.01</td><td>0.08 ±.09</td><td>0.02 ±.02</td><td>0.03</td></tr><tr><td>BM25</td><td>1.00</td><td>0.99</td><td>0.99</td><td>0.95</td><td>0.89</td><td>0.96</td></tr><tr><td>RepSim</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00 ±.00</td><td>0.00</td></tr><tr><td colspan="7">Neg. log-odds, top-10 (Figure 10)</td></tr><tr><td>Grad-Dot</td><td>0.98 ±.02</td><td>1.00 ±.00</td><td>1.00 ±.00</td><td>0.99 ±.01</td><td>1.00 ±.00</td><td>0.99</td></tr><tr><td>Grad-Cos†</td><td>0.62 ±.20</td><td>0.36 ±.19</td><td>0.26 ±.34</td><td>0.37 ±.16</td><td>0.58 ±.31</td><td>0.44</td></tr><tr><td>EK-FAC</td><td>0.98 ±.02</td><td>1.00 ±.00</td><td>1.00 ±.00</td><td>1.00 ±.00</td><td>1.00 ±.00</td><td>1.00</td></tr><tr><td>LoGra</td><td>0.60 ±.07</td><td>0.63 ±.21</td><td>0.54±.30</td><td>0.97 ±.02</td><td>0.95 ±.07</td><td>0.74</td></tr><tr><td>LoGra-Dot</td><td>0.96 ±.04</td><td>0.95 ±.03</td><td>0.91 ±.06</td><td>0.91 ±.02</td><td>0.94 ±.04</td><td>0.93</td></tr><tr><td>TRAK (no  $Q ) ^ { \dagger }$ </td><td>0.85 ±.07</td><td>0.23 ±.10</td><td>0.09 ±.08</td><td>0.01 ±.01</td><td>0.00 ±.00</td><td>0.24</td></tr><tr><td>TRAK (k=126,976) (no Q)†</td><td>0.94 ±.03</td><td>0.53 ±.18</td><td>0.33 ±.32</td><td>0.18 ±.11</td><td>0.04 ±.04</td><td>0.40</td></tr><tr><td>Layerwise TRAK (no Q)</td><td>0.91 ±.09</td><td>0.67 ±.16</td><td>0.59 ±.25</td><td>0.65 ±.07</td><td>0.55 ±.30</td><td>0.67</td></tr><tr><td>TrackStar</td><td>0.07 ±.05</td><td>0.00 ±.00</td><td>0.06 ±.11</td><td>0.17 ±.06</td><td>0.33 ±.32</td><td>0.13</td></tr><tr><td>TracIn†</td><td>0.05 ±.04</td><td>0.15 ±.07</td><td>0.11 ±.14</td><td>0.00 ±.00</td><td>0.01 ±.01</td><td>0.06</td></tr><tr><td>TracIn-Cos†</td><td>0.56 ±.33</td><td>0.72 ±.04</td><td>0.79 ±.22</td><td>0.73 ±.30</td><td>0.80 ±.24</td><td>0.72</td></tr><tr><td>Layerwise TracIn</td><td>0.98 ±.00</td><td>1.00 ±.00</td><td>0.99 ±.01</td><td>1.00 ±.00</td><td>0.99 ±.02</td><td>0.99</td></tr><tr><td>Layerwise TracIn.  ${ \cos ^ { \dagger } }$ </td><td>0.52 ±.30</td><td>0.72 ±.06</td><td>0.76 ±.27</td><td>0.67 ±.30</td><td>0.78 ±.26</td><td>0.69</td></tr><tr><td>SOURCE†</td><td>0.06 ±.06</td><td>0.14 ±.04</td><td>0.15 ±.21</td><td>0.00 ±.01</td><td>0.01 ±.02</td><td>0.08</td></tr></table>

## D.3 EVERY SCORING VARIANT

The per-token terms of a method give many document scores at no extra cost (Section 2.3). Table 14 reports Recall@N for each of them. The CE columns are the full-document mean and sum and the top-k means. The log-odds columns are:

• the full-document mean, which is the gradient of $L _ { \mathrm { l o } } \mathrm { . }$

• the full-document sum;

• the mean weighted by $Q = 1 - \bar { p } .$ , which is TRAK’s recipe applied to every method;

• the top-k means.

Four observations follow.

• The loss matters before pooling. On the two Function Bindings tasks, top-k pooling of the CE terms recovers EK-FAC (from 0.59 to 0.89, and from 0.06 to 0.87) but leaves every other method at 0.22 or below. Switching to log-odds terms lifts every method on Function Bindings, even without pooling.

• The full-document mean favors short documents. With distractors, the full-document mean fails under either loss (Grad-Dot 0.00 and EK-FAC 0.06 under CE, 0.05 and 0.18 under log-odds), while the log-odds sum succeeds (0.85 and 0.99). The mean divides each document’s score by its length, $^ { n _ { d } , }$ and the 12-token decoys are an order of magnitude shorter than the fact documents. Top-k pooling does not depend on length when k is below the shortest document, and it gives the best score on most rows.

Table 12: Recall@N on Rule Routings. Layout as in Table 10. The $N = 1$ column lies outside the Rule Routings grid (Appendix B.2) and is shown for completeness; the mean is over $N \in$ {5, 10, 30, 50}.
<table><tr><td>Method</td><td>N=1</td><td> $N { = } 3$ </td><td> $N { = } 5$ </td><td> $N { = } 1 0$ </td><td> $N { = } 3 0$ </td><td> $N { = } 5 0$ </td><td>Mean</td></tr><tr><td colspan="8">Document-level score (variants of Figure 3)</td></tr><tr><td>Grad-Dot</td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 2 \pm . 0 1$ </td><td> $0 . 0 2 \pm . 0 1$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td> $0 . 0 6 \pm . 0 1$ </td><td>0.04</td></tr><tr><td>Grad-Cos</td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 8 \pm . 0 1$ </td><td> $0 . 0 6 \pm . 0 1$ </td><td>0.05</td></tr><tr><td>EK-FAC</td><td> $0 . 0 7 \pm . 0 7$ </td><td>0.26 ±.04</td><td> $0 . 2 8 \pm . 0 1$ </td><td> $0 . 3 2 \pm . 0 1$ </td><td> $0 . 0 5 \pm . 0 2$ </td><td> $0 . 1 2 \pm . 0 1$ </td><td>0.19</td></tr><tr><td>LoGra</td><td> $0 . 0 1 \pm . 0 1$ </td><td>0.05 ±.01</td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td> $0 . 1 1 \pm . 0 1$ </td><td>0.06</td></tr><tr><td>LoGra-Dot</td><td> $0 . 0 1 \pm . 0 0$ </td><td>0.02 ±.00</td><td> $0 . 0 2 \pm . 0 1$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 4 \pm . 0 0$ </td><td> $0 . 0 6 \pm . 0 0$ </td><td>0.03</td></tr><tr><td>TRAK</td><td></td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td>0.03</td></tr><tr><td> $\mathrm { T R A K } \left( k { = } 1 2 6 , 9 7 6 \right)$ </td><td></td><td>0.10 ±.02</td><td> $0 . 1 1 \pm . 0 3$ </td><td> $0 . 1 0 \pm . 0 2$ </td><td> $0 . 0 8 \pm . 0 1$ </td><td> $0 . 1 1 \pm . 0 1$ </td><td>0.10</td></tr><tr><td>Layerwise TRAK</td><td> $0 . 0 7 \pm . 0 3$ </td><td> $0 . 0 8 \pm . 0 1$ </td><td> $0 . 0 8 \pm . 0 3$ </td><td> $0 . 0 6 \pm . 0 1$ </td><td> $0 . 0 6 \pm . 0 0$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td>0.07</td></tr><tr><td>TrackStar</td><td> $0 . 0 4 \pm . 0 2$ </td><td> $0 . 0 5 \pm . 0 0$ </td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 9 \pm . 0 1$ </td><td> $0 . 1 1 \pm . 0 1$ </td><td>0.07</td></tr><tr><td>TracIn</td><td></td><td> $0 . 0 4 \pm . 0 0$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td> $0 . 0 8 \pm . 0 1$ </td><td> $0 . 0 9 \pm . 0 3$ </td><td> $0 . 0 5 \pm . 0 0$ </td><td>0.06</td></tr><tr><td>TracIn-Cos</td><td></td><td> $0 . 0 4 \pm . 0 0$ </td><td> $0 . 0 4 \pm . 0 0$ </td><td> $0 . 0 6 \pm . 0 0$ </td><td> $0 . 0 6 \pm . 0 1$ </td><td> $0 . 0 5 \pm . 0 0$ </td><td>0.05</td></tr><tr><td>Layerwise TracIn</td><td> $0 . 0 2 \pm . 0 2$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td> $0 . 0 8 \pm . 0 2$ </td><td> $0 . 0 5 \pm . 0 0$ </td><td>0.06</td></tr><tr><td>Layerwise TracIn-Cos</td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 4 \pm . 0 0$ </td><td> $0 . 0 6 \pm . 0 1$ </td><td> $0 . 0 6 \pm . 0 2$ </td><td> $0 . 0 5 \pm . 0 0$ </td><td>0.05</td></tr><tr><td>SOURCE</td><td> $0 . 0 2 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td> $0 . 0 4 \pm . 0 0$ </td><td> $0 . 0 9 \pm . 0 0$ </td><td> $0 . 0 9 \pm . 0 3$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td>0.08</td></tr><tr><td>BM25</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>RepSim</td><td></td><td>0.16 ±.09</td><td> $0 . 0 8 \pm . 0 5$ </td><td> $0 . 1 0 \pm . 0 8$ </td><td> $0 . 0 9 \pm . 0 7$ </td><td> $0 . 2 9 \pm . 0 9$ </td><td>0.14</td></tr><tr><td colspan="8">Neg. log-odds, top-10 (Figure 10)</td></tr><tr><td>Grad-Dot</td><td> $0 . 1 6 \pm . 0 9$ </td><td> $0 . 3 5 \pm . 0 8$ </td><td> $0 . 3 3 \pm . 0 5$ </td><td> $0 . 2 2 \pm . 1 0$ </td><td> $0 . 3 2 \pm . 0 2$ </td><td> $0 . 3 6 \pm . 1 6$ </td><td>0.31</td></tr><tr><td>Grad  ${ \mathrm { . c o s } } ^ { \dagger }$ </td><td> $0 . 1 0 \pm . 0 6$ </td><td> $0 . 2 2 \pm . 0 6$ </td><td> $0 . 2 1 \pm . 0 6$ </td><td> $0 . 2 1 \pm . 0 4$ </td><td> $0 . 2 0 \pm . 0 4$ </td><td> $0 . 3 2 \pm . 0 9$ </td><td>0.23</td></tr><tr><td>EK-FAC</td><td> $0 . 4 0 \pm . 1 6$ </td><td> $0 . 8 0 \pm . 0 5$ </td><td> $0 . 8 7 \pm . 0 3$ </td><td> $0 . 7 0 \pm . 1 0$ </td><td> $0 . 2 0 \pm . 1 2$ </td><td> $0 . 4 1 \pm . 1 5$ </td><td>0.55</td></tr><tr><td>LoGra</td><td> $0 . 0 8 \pm . 0 6$ </td><td> $0 . 1 5 \pm . 0 6$ </td><td> $0 . 1 8 \pm . 1 3$ </td><td> $0 . 0 8 \pm . 0 5$ </td><td> $0 . 2 2 \pm . 0 6$ </td><td> $0 . 2 3 \pm . 1 5$ </td><td>0.18</td></tr><tr><td>LoGra-Dot</td><td> $0 . 1 2 \pm . 0 6$ </td><td> $0 . 2 0 \pm . 0 9$ </td><td> $0 . 2 1 \pm . 1 5$ </td><td> $0 . 0 9 \pm . 0 7$ </td><td> $0 . 1 3 \pm . 0 3$ </td><td> $0 . 2 1 \pm . 1 5$ </td><td>0.16</td></tr><tr><td> $\mathrm { T R A K } ( \mathfrak { n o } Q ) ^ { \dagger }$ </td><td></td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 3 \pm . 0 0$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td> $0 . 0 2 \pm . 0 0$ </td><td>0.03</td></tr><tr><td> $\scriptstyle { \mathrm { T R A K } } ( k = 1 2 6 , 9 7 6 ) ( \mathrm { n o } Q ) ^ { \dagger }$ </td><td></td><td> $0 . 1 4 \pm . 0 2$ </td><td> $0 . 1 4 \pm . 0 3$ </td><td> $0 . 1 1 \pm . 0 3$ </td><td> $0 . 0 9 \pm . 0 1$ </td><td> $0 . 1 1 \pm . 0 2$ </td><td>0.11</td></tr><tr><td> $\operatorname { L a y e r w i s e } \operatorname { T R A K } \left( \operatorname { n o } Q \right)$ </td><td> $0 . 0 9 \pm . 0 1$ </td><td> $0 . 0 7 \pm . 0 1$ </td><td> $0 . 0 7 \pm . 0 4$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 0 5 \pm . 0 1$ </td><td> $0 . 0 6 \pm . 0 3$ </td><td>0.06</td></tr><tr><td>TrackStar</td><td>0.09</td><td> $0 . 2 0 \pm . 0 5$ </td><td> $0 . 2 7 \pm . 0 3$ </td><td> $0 . 2 5 \pm . 0 3$ </td><td> $0 . 3 2 \pm . 0 7$ </td><td> $0 . 4 6 \pm . 1 0$ </td><td>0.32</td></tr><tr><td> $\mathrm { T r a c I n } ^ { \dagger }$ </td><td></td><td> $0 . 0 7 \pm . 0 4$ </td><td> $0 . 1 8 \pm . 0 1$ </td><td> $0 . 3 2 \pm . 0 4$ </td><td> $0 . 4 5 \pm . 1 6$ </td><td> $0 . 4 4 \pm . 0 9$ </td><td>0.35</td></tr><tr><td>TracIn  ${ \cos ^ { \dagger } }$ </td><td></td><td>0.08 ±.03</td><td> $0 . 1 6 \pm . 0 0$ </td><td> $0 . 3 4 \pm . 0 2$ </td><td> $0 . 4 5 \pm . 1 6$ </td><td> $0 . 4 5 \pm . 0 8$ </td><td>0.35</td></tr><tr><td>Layerwise TracIn</td><td> $0 . 0 5 \pm . 0 4$ </td><td>0.05 ±.00</td><td> $0 . 2 5 \pm . 0 4$ </td><td> $0 . 1 2 \pm . 0 4$ </td><td> $0 . 4 8 \pm . 2 2$ </td><td> $0 . 4 3 \pm . 0 7$ </td><td>0.32</td></tr><tr><td>Layerwise  $\operatorname { T r a c I n - C o s } ^ { \dagger }$ </td><td> $0 . 0 3 \pm . 0 1$ </td><td>0.05 ±.00</td><td> $0 . 1 0 \pm . 0 0$ </td><td> $0 . 2 2 \pm . 0 1$ </td><td> $0 . 3 6 \pm . 1 5$ </td><td> $0 . 3 8 \pm . 0 7$ </td><td>0.26</td></tr><tr><td>SOURCE†</td><td> $0 . 0 4 \pm . 0 2$ </td><td> $0 . 0 7 \pm . 0 2$ </td><td> $0 . 1 4 \pm . 0 1$ </td><td> $0 . 2 6 \pm . 0 3$ </td><td> $0 . 4 3 \pm . 1 5$ </td><td> $0 . 5 1 \pm . 0 5$ </td><td>0.34</td></tr></table>

• Q collapses every method on distractors. Weighting by $1 - \bar { p }$ drops every method to 0.02 or below with distractors, because the poorly fit decoys receive the largest weights. On the clean tasks, Q costs between 0.00 and 0.10.

• TrackStar is the exception with distractors. Its top-10 score (0.13) is below its fulldocument mean (0.18). Its score carries its own document weight, the inverse whitened norm $1 / \| R ^ { - 1 / 2 } \phi \|$ (Section A.2), which top-k pooling does not remove. We expect that weight to be large for short documents, which would explain the result, but we have not verified this directly.

## D.4 SCORE DISTRIBUTIONS AND TOKEN TYPES ON THE OTHER TASKS

Figure 4 shows a single cell of Function Bindings with Distractors, and Figure 5 shows a single Rule Routings setting. Both can be explored on the other tasks online: the score distributions for every task, method and model seed on our project page, and the per-token attributions of each method for any query and training document in our demo. Below we summarize both for Grad-Dot on the other tasks.

• Score distributions (same cell as Figure 4: $| \mathcal { F } | = 5 0 , N = 5 0$ , seed 1001). On Function Bindings, changing the loss alone separates the ground-truth documents from the rest: Recall@N rises from 0.11 to 1.00 with neg. log-odds over all tokens and stays at 1.00 with top-10 selection, while top-10 selection under cross-entropy reaches 0.27. On Rule Routings, each change helps but neither separates the two groups. Recall@N is 0.07 with cross-entropy over all tokens, 0.39 with neg. log-odds, 0.10 with cross-entropy top-10, and 0.47 with both changes. With neg. log-odds top-10 scores, Rule Routings is also the task on which most attribution methods have their lowest recall (Section D.1); the exceptions, TrackStar, TracIn and SOURCE, score lower still on Function Bindings with distractors. With document-level scores, every attribution method except TracIn-Cos and Layerwise TracIn-Cos has its lowest recall on Function Bindings with distractors.

Table 13: AUC averaged over N at $| \mathcal { F } | = 5 0 .$ , for the document-level score (Doc.) and the log-odds top-10 score (Top-10). The last two columns are decoy AUC on Function Bindings with Distractors: each ground-truth document ranked against the decoys of its own query. <sup>†</sup>: no per-token terms; the Top-10 column uses the full-document log-odds. TRAK’s Doc. columns include Q and its Top-10 columns do not. The retrieval references have one configuration.
<table><tr><td rowspan="2">Method</td><td colspan="2">Function Bindings</td><td colspan="2">w/ Distractors</td><td colspan="2">Rule Routings</td><td colspan="2">w/ Distractors, decoy AUC</td></tr><tr><td>Doc.</td><td>Top-10</td><td>Doc.</td><td>Top-10</td><td>Doc.</td><td>Top-10</td><td>Doc.</td><td>Top-10</td></tr><tr><td>Grad-Dot</td><td>0.46</td><td>1.00</td><td>0.39</td><td>1.00</td><td>0.50</td><td>0.85</td><td>0.44</td><td>1.00</td></tr><tr><td>Grad-Cos†</td><td>0.46</td><td>1.00</td><td>0.31</td><td>0.91</td><td>0.50</td><td>0.71</td><td>0.30</td><td>0.63</td></tr><tr><td>EK-FAC</td><td>0.69</td><td>1.00</td><td>0.53</td><td>1.00</td><td>0.52</td><td>0.95</td><td>0.45</td><td>1.00</td></tr><tr><td>LoGra</td><td>0.52</td><td>0.99</td><td>0.50</td><td>0.98</td><td>0.51</td><td>0.79</td><td>0.50</td><td>1.00</td></tr><tr><td>LoGra-Dot</td><td>0.48</td><td>1.00</td><td>0.44</td><td>1.00</td><td>0.50</td><td>0.78</td><td>0.48</td><td>1.00</td></tr><tr><td>TRAK†</td><td>0.80</td><td>0.80</td><td>0.64</td><td>0.70</td><td>0.51</td><td>0.52</td><td>0.42</td><td>0.68</td></tr><tr><td>TRAK  $\scriptstyle ( k = 1 2 6 , 9 7 6 ) ^ { \dag }$ </td><td>0.92</td><td>0.92</td><td>0.72</td><td>0.80</td><td>0.54</td><td>0.54</td><td>0.39</td><td>0.74</td></tr><tr><td>Layerwise TRAK</td><td>0.89</td><td>0.99</td><td>0.68</td><td>0.98</td><td>0.53</td><td>0.57</td><td>0.40</td><td>1.00</td></tr><tr><td>TrackStar</td><td>0.55</td><td>0.99</td><td>0.50</td><td>0.85</td><td>0.51</td><td>0.81</td><td>0.49</td><td>0.51</td></tr><tr><td>TracIn†</td><td>0.66</td><td>1.00</td><td>0.52</td><td>0.86</td><td>0.54</td><td>0.80</td><td>0.42</td><td>0.13</td></tr><tr><td>TracIn-Cos†</td><td>0.67</td><td>1.00</td><td>0.49</td><td>0.99</td><td>0.54</td><td>0.78</td><td>0.44</td><td>0.74</td></tr><tr><td>Layerwise TracIn</td><td>0.62</td><td>1.00</td><td>0.54</td><td>1.00</td><td>0.54</td><td>0.87</td><td>0.43</td><td>1.00</td></tr><tr><td>Layerwise TracIn-Cos†</td><td>0.63</td><td>1.00</td><td>0.52</td><td>0.98</td><td>0.53</td><td>0.75</td><td>0.45</td><td>0.71</td></tr><tr><td>SOURCE†</td><td>0.67</td><td>1.00</td><td>0.56</td><td>0.85</td><td>0.55</td><td>0.79</td><td>0.41</td><td>0.15</td></tr><tr><td>BM25</td><td>1.00</td><td>一</td><td>1.00</td><td>一</td><td>1.00</td><td></td><td>0.99</td><td></td></tr><tr><td>RepSim</td><td>0.62</td><td>一</td><td>0.33</td><td>一</td><td>0.68</td><td>一</td><td>0.00</td><td>一</td></tr></table>

• Token types. On the Function Bindings tasks, each position is one of six types, taken in this order: the answer token ⟨V⟩, the function name ⟨B⟩, integers, other tokens inside code, other tokens containing a letter (prose), and the rest (markup and punctuation). We measure each type’s share of a document’s ten highest-scoring positions relative to its share of all positions, pooled over every (query, document) pair of the $| \mathcal { F } | = 5 0$ cells, so that 1× is chance. Under cross-entropy, no type is strongly enriched in the top ten positions: the answer token ranges from 0.2× to 1.3× its baseline rate on Function Bindings and from 0.9× to 1.2× with distractors. Under neg. log-odds, the answer token is enriched 5.1–11× on Function Bindings and 2.9–6.6× with distractors, and no other type exceeds 1.4×. This matches Rule Routings, where neg. log-odds enriches the outcome tokens (Figure 5).

## D.5 NUMBER OF FACTS

The main text fixes $| \mathcal { F } | ~ = ~ 5 0 ~ $ . Its conclusions do not depend on that choice: across $| { \mathcal { F } } | \in$ {25, 50, 100}, the same methods succeed and fail on the same tasks, and recall moves much less with the number of facts than with the number of documents per fact N or with distractors. Table 15 compares the three sizes at $N = 1 0$ , the document count that the $| \mathcal { F } | = 1 0 0$ grid covers for the most methods, and Figure 11 repeats Figure 3 at $| \mathcal { F } | = 2 5$ . The main-text trends hold at every size we ran:

• TRAK collapses as N grows on Function Bindings: from 0.74 at N = 1 to 0.08 at $N = 5 0$ at $| \mathcal { F } | = 2 5 \ : \overline { { ( 0 . 8 8 } }$ to 0.07 at $| \mathcal { F } | = 5 0 )$ , and from 0.90 at $N = 1$ to 0.16 at $N = 1 0$ at $| \mathcal { F } | = 1 0 0$ . EK-FAC degrades less and is the strongest attribution method at $| \mathcal { F } | = 2 5$ as at $| \mathcal { F } | = 5 0$ (mean recall 0.53, against 0.59).

• Distractors defeat document-level scores. With distractors, no document-level score in Table 15 exceeds 0.10 at any |F|, and at $| \mathcal { F } | = 2 5$ none in Figure 11 exceeds 0.16 at any N.

Table 14: Recall@N of every scoring variant, averaged over N at $| \mathcal { F } | = 5 0 ,$ , for the methods with per-token terms. Top-k averages the k largest terms under either loss (Section 2.3). $\mathrm { \cdots m e a n } \times Q ^ { \mathrm { \cdots } }$ multiplies the full-document log-odds by $1 - { \bar { p } } .$ For Layerwise TRAK, every column except that one is unweighted. TrackStar has no sum columns, because its normalized score is the same under the sum and the mean. Bold: best in the row, within 0.005.
<table><tr><td></td><td></td><td colspan="4">Cross-entropy terms</td><td colspan="5">Neg. log-odds terms</td></tr><tr><td>Task</td><td>Method</td><td>mean</td><td>sum</td><td>top-3</td><td>top-10</td><td>mean</td><td>sum</td><td> ${ \mathrm { m e a n } } \times { Q }$ </td><td>top-3</td><td>top-10</td></tr><tr><td rowspan="7">Function Bindings</td><td>Grad-Dot</td><td>0.07</td><td>0.07</td><td>0.14</td><td>0.12</td><td>0.99</td><td>0.99</td><td>0.98</td><td>1.00</td><td>1.00</td></tr><tr><td>EK-FAC</td><td>0.59</td><td>0.59</td><td>0.87</td><td>0.89</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>LoGra</td><td>0.08</td><td>0.08</td><td>0.12</td><td>0.09</td><td>0.89</td><td>0.89</td><td>0.86</td><td>0.93</td><td>0.92</td></tr><tr><td>LoGra-Dot</td><td>0.04</td><td>0.04</td><td>0.05</td><td>0.06</td><td>0.93</td><td>0.93</td><td>0.90</td><td>0.97</td><td>0.98</td></tr><tr><td>Layerwise TRAK (no Q)</td><td>0.04</td><td>0.04</td><td>0.03</td><td>0.04</td><td>0.65</td><td>0.64</td><td>0.62</td><td>0.84</td><td>0.87</td></tr><tr><td>TrackStar</td><td>0.14</td><td></td><td>0.21</td><td>0.22</td><td>0.91</td><td></td><td>0.89</td><td>0.95</td><td>0.96</td></tr><tr><td>Layerwise TracIn</td><td>0.14</td><td>0.15</td><td>0.16</td><td>0.14</td><td>1.00</td><td>1.00</td><td>0.97</td><td>0.99</td><td>1.00</td></tr><tr><td rowspan="7">w/ Distractors</td><td>Grad-Dot</td><td>0.00</td><td>0.06</td><td>0.13</td><td>0.14</td><td>0.05</td><td>0.85</td><td>0.00</td><td>0.99</td><td>0.99</td></tr><tr><td>EK-FAC</td><td>0.06</td><td>0.44</td><td>0.83</td><td>0.87</td><td>0.18</td><td>0.99</td><td>0.01</td><td>0.99</td><td>1.00</td></tr><tr><td>LoGra</td><td>0.00</td><td>0.08</td><td>0.15</td><td>0.11</td><td>0.03</td><td>0.60</td><td>0.00</td><td>0.75</td><td>0.74</td></tr><tr><td>LoGra-Dot</td><td>0.00</td><td>0.04</td><td>0.06</td><td>0.07</td><td>0.06</td><td>0.64</td><td>0.01</td><td>0.94</td><td>0.93</td></tr><tr><td>Layerwise TRAK (no Q)</td><td>0.00</td><td>0.01</td><td>0.03</td><td>0.03</td><td>0.33</td><td>0.47</td><td>0.00</td><td>0.60</td><td>0.67</td></tr><tr><td>TrackStar</td><td>0.03</td><td></td><td>0.04</td><td>0.05</td><td>0.18</td><td></td><td>0.02</td><td>0.07</td><td>0.13</td></tr><tr><td>Layerwise TracIn</td><td>0.04</td><td>0.12</td><td>0.18</td><td>0.14</td><td>0.07</td><td>0.93</td><td>0.00</td><td>0.98</td><td>0.99</td></tr><tr><td rowspan="7">Rule Routings</td><td>Grad-Dot</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.05</td><td>0.22</td><td>0.22</td><td>0.17</td><td>0.31</td><td>0.31</td></tr><tr><td>EK-FAC</td><td>0.19</td><td>0.19</td><td>0.30</td><td>0.35</td><td>0.48</td><td>0.48</td><td>0.44</td><td>0.47</td><td>0.55</td></tr><tr><td>LoGra</td><td>0.06</td><td>0.06</td><td>0.06</td><td>0.05</td><td>0.17</td><td>0.17</td><td>0.15</td><td>0.19</td><td>0.18</td></tr><tr><td>LoGra-Dot</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.04</td><td>0.13</td><td>0.13</td><td>0.10</td><td>0.17</td><td>0.16</td></tr><tr><td>Layerwise TRAK (no Q)</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.05</td><td>0.07</td><td>0.07</td><td>0.07</td><td>0.05</td><td>0.06</td></tr><tr><td>TrackStar</td><td>0.07</td><td></td><td>0.13</td><td>0.13</td><td>0.18</td><td></td><td>0.14</td><td>0.31</td><td>0.32</td></tr><tr><td>Layerwise TracIn</td><td>0.06</td><td>0.06</td><td>0.07</td><td>0.07</td><td>0.26</td><td>0.26</td><td>0.16</td><td>0.31</td><td>0.32</td></tr></table>

![](images/04faa0851ef94f66e63c59baab750c0b228cf07cebeb269d6bc91f05752e6ad3.jpg)  
Figure 11: Recall@N at $| { \mathcal { F } } | = 2 5 .$ . The counterpart of Figure 3, which is at $| { \mathcal { F } } | = 5 0 .$ , with the same methods and document-level scores. The panel titles and method names follow the main-text figures.

• Rule Routings stays hard for gradient methods, while BM25 does not. ${ \mathrm { A t } } | { \mathcal { F } } | = 2 5 .$ , no attribution method exceeds 0.3 at any N, while BM25 is at 1.00, as at $| \mathcal { F } | = 5 0$

• Top-10 log-odds scoring recovers Function Bindings at every size. Grad-Dot, EK-FAC and Layerwise TracIn reach at least 0.99 on both Function Bindings tasks at every |F|.

What changes with |F| is a gradual decline for some methods. EK-FAC’s document-level recall on Function Bindings falls from 0.67 at $| \mathcal { F } | = 5 0 \mathrm { t o } 0 . 2 2 \mathrm { a t } | \mathcal { F } | = 1 0 0$ . LoGra’s top-10 recall falls as facts are added, most with distractors (0.90, 0.54, 0.52). On Rule Routings, most methods lose top-10 recall as rules are added; EK-FAC’s falls from 0.78 to 0.70 to 0.56. TrackStar is the exception, at 0.25, 0.25 and 0.26. The $| \mathcal { F } | = 1 0 0$ grid covers mostly $N \leq 1 0$ , and at $N = 1 0$ its column is incomplete: on Function Bindings it has no TrackStar and no top-10 score for Layerwise TRAK, and it has no BM25 on Function Bindings or Rule Routings.

Table 15: Recall@N at $N = 1 0$ for $| \mathcal { F } | \in \{ 2 5 , 5 0 , 1 0 0 \}$ , mean over seeds. Doc.: document-level score; Top-10: log-odds top-10 (<sup>†</sup>: no per-token terms, full-document log-odds; TRAK’s Top-10 column without $\bar { Q ) . } \stackrel {  } {  } \stackrel {  } { : }$ not run.
<table><tr><td></td><td></td><td colspan="2">|F|=25</td><td colspan="2"> $| \mathcal { F } | = 5 0$ </td><td colspan="2"> $\lvert \mathcal { F } \rvert = 1 0 0$ </td></tr><tr><td>Task</td><td>Method</td><td>Doc.</td><td>Top-10</td><td>Doc.</td><td>Top-10</td><td>Doc.</td><td>Top-10</td></tr><tr><td rowspan="10">Function Bindings</td><td>Grad-Dot</td><td>0.13</td><td>1.00</td><td>0.12</td><td>1.00</td><td>0.12</td><td>0.99</td></tr><tr><td>EK-FAC</td><td>0.62</td><td>1.00</td><td>0.67</td><td>1.00</td><td>0.22</td><td>1.00</td></tr><tr><td>LoGra</td><td>0.08</td><td>0.94</td><td>0.09</td><td>0.82</td><td>0.11</td><td>0.78</td></tr><tr><td>TRAK†</td><td>0.32</td><td>0.32</td><td>0.24</td><td>0.25</td><td>0.16</td><td>0.18</td></tr><tr><td>Layerwise TRAK</td><td>0.56</td><td>0.77</td><td>0.61</td><td>0.80</td><td>0.61</td><td>一</td></tr><tr><td>TrackStar</td><td>0.14</td><td>1.00</td><td>0.15</td><td>0.98</td><td></td><td>一</td></tr><tr><td>Layerwise TracIn</td><td>0.23</td><td>1.00</td><td>0.17</td><td>1.00</td><td>0.16</td><td>1.00</td></tr><tr><td>SOURCE†</td><td>0.27</td><td>1.00</td><td>0.25</td><td>1.00</td><td>0.27</td><td>1.00</td></tr><tr><td>BM25</td><td>1.00</td><td>一</td><td>1.00</td><td></td><td></td><td>一</td></tr><tr><td>Grad-Dot</td><td>0.00</td><td>1.00</td><td>0.00</td><td>1.00</td><td>0.00</td><td>0.99</td></tr><tr><td rowspan="9">w/ Distractors</td><td>EK-FAC</td><td>0.10</td><td>1.00</td><td>0.01</td><td>1.00</td><td>0.01</td><td>1.00</td></tr><tr><td>LoGra</td><td>0.00</td><td>0.90</td><td>0.00</td><td>0.54</td><td>0.00</td><td>0.52</td></tr><tr><td>TRAK†</td><td>0.00</td><td>0.25</td><td>0.00</td><td>0.09</td><td>0.00</td><td>0.03</td></tr><tr><td>Layerwise TRAK</td><td>0.00</td><td>0.76</td><td>0.00</td><td>0.59</td><td>0.00</td><td>0.40</td></tr><tr><td>TrackStar</td><td>0.10</td><td>0.01</td><td>0.07</td><td>0.06</td><td>0.07</td><td>0.03</td></tr><tr><td>Layerwise TracIn</td><td>0.02</td><td>1.00</td><td>0.03</td><td>0.99</td><td>0.04</td><td>1.00</td></tr><tr><td>SOURCE†</td><td>0.01</td><td>0.06</td><td>0.01</td><td>0.15</td><td>0.02</td><td>0.09</td></tr><tr><td>BM25</td><td>0.94</td><td>一</td><td>0.99</td><td>一</td><td>0.99</td><td>一</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="9">Rule Routings</td><td>Grad-Dot EK-FAC</td><td>0.03</td><td>0.35</td><td>0.02</td><td>0.22</td><td>0.02</td><td>0.15</td></tr><tr><td></td><td>0.28</td><td>0.78</td><td>0.32</td><td>0.70</td><td>0.25</td><td>0.56</td></tr><tr><td>LoGra</td><td>0.06</td><td>0.20</td><td>0.03</td><td>0.08</td><td>0.04</td><td>0.05</td></tr><tr><td>TRAK†</td><td>0.05</td><td>0.05</td><td>0.02</td><td>0.03</td><td>0.01</td><td>0.02</td></tr><tr><td>Layerwise TRAK</td><td>0.10</td><td>0.08</td><td>0.06</td><td>0.04</td><td>0.05</td><td>0.02</td></tr><tr><td>TrackStar</td><td>0.05</td><td>0.25</td><td>0.04</td><td>0.25</td><td>0.04</td><td>0.26</td></tr><tr><td>Layerwise TracIn</td><td>0.07</td><td>0.32</td><td>0.07</td><td>0.12</td><td>0.04</td><td>0.19</td></tr><tr><td>SOURCE†</td><td>0.08</td><td>0.31</td><td>0.09</td><td>0.26</td><td>0.06</td><td>0.24</td></tr><tr><td>BM25</td><td>1.00</td><td>一</td><td>1.00</td><td>一</td><td></td><td>一</td></tr></table>

## E FTRACE-TREX

This appendix details the external evaluation of Figure 7. We follow the FTRACE-TREX setup of Akyurek et al. ¨ (2022) and the remove-and-retrain protocol of Park et al. (2023), with the departures listed at the end.

Data and model. The training corpus consists of 1,560,453 masked examples drawn from 484,408 TREx abstract sentences, and the queries are 31,479 cloze-style facts, each annotated with its proponents: the training examples that state the fact. We fine-tune mT5-small (300M parameters) as a sequence-to-sequence model for 60,000 steps (Adafactor, constant learning rate $1 \bar { 0 ^ { - 3 } }$ , batch size 128, inputs truncated to 256 tokens and targets to 32; seed 1001). Top-1 accuracy with beam search rises from 10.7% for the pretrained model to 33.7%. Following Park et al. (2023), the novel facts are the 8,437 queries that the pretrained model gets wrong and the fine-tuned model gets right at top-1. All attribution scores are computed on this one fine-tuned model.

Scoring. Each FTRACE-TREX target frames its answer with format tokens: the sentinel <extra id 0> before it and the end-of-sequence token $< / \varsigma >$ after. They occur in every target and the model predicts them almost perfectly, so we drop them from the loss on both the query and the training-document side and score only the answer tokens; the model still reads the full input and target. The per-position gradients are summed over the encoder positions (input tokens) and the decoder positions (answer tokens). We compare three methods, each as published and with our fixes:

• TRAK applies the published dense Rademacher JL projection to all 300M parameters at $k = 4 { , } 0 0 0$ and multiplies by $Q _ { d } ~ = ~ 1 - \bar { p } _ { d }$ , where $\bar { p } _ { d }$ is the mean probability of the document’s answer tokens. With n ≫ k we solve it in the primal form $\phi _ { d } ^ { \mathsf { T } } ( \Phi ^ { \mathsf { T } } \Phi + \bar { \lambda I } ) ^ { - 1 } \phi _ { q } ,$ which equals Eq. equation 21 before the $Q _ { d }$ factor, with $\lambda = 1 0 ^ { - 6 } \mathrm { t r } ( \Phi ^ { \mathsf { T } } \Phi ) / k$ . TRAK (negative log-odds) is the same score without $Q _ { d } .$ . A dense projection does not factor over positions, so it gives no per-position terms and TRAK has no top-10 variant.

Table 16: FTRACE-TREX remove-and-retrain: drop in top-1 accuracy (percentage points) on the 50 facts at each removal budget, mean over eight retraining seeds ± one standard error. Score: the training-document score as published (TRAK’s log-odds with $Q _ { d } ,$ , cross-entropy for EK-FAC and Grad-Dot) or the negative log-odds; Pooling: summed over all positions or the mean of the top 10 per-position terms. The last two columns are at $B = 1 1 , 5 2 2$ . Prop. removed: mean fraction of each fact’s proponents in the removal set. Same-rel. drop: the accuracy drop on the other 5,797 novel facts that share a relation with one of the 50. On the 2,590 novel facts in other relations, every set moves accuracy by at most 4 points. BM25 and the ground truth use two seeds at $B = 2 { , } 0 0 0 $ and 5,000 and three at 11,522. <sup>†</sup>One seed.
<table><tr><td colspan="3"></td><td colspan="3">Drop on the 50 facts</td><td colspan="2"> $\operatorname { A t } B = 1 1 { , } 5 2 2$ </td></tr><tr><td>Method</td><td>Score</td><td>Pooling</td><td> $B = 2 , 0 0 0$ </td><td>5,000</td><td>11,522</td><td></td><td>Prop. removed Same-rel. drop</td></tr><tr><td>TRAK</td><td>Published</td><td>All</td><td> ${ 9 \pm 3 }$ </td><td> $1 0 \pm 2$ </td><td> $\hphantom { 0 } 8 \pm 2$ </td><td>13%</td><td>4.2</td></tr><tr><td></td><td>Neg. log-odds</td><td>All</td><td> $1 3 \pm 3$ </td><td> $2 8 \pm 2$ </td><td> $4 0 \pm 4$ </td><td>24%</td><td>11.7</td></tr><tr><td rowspan="4">EK-FAC</td><td>Published</td><td>All</td><td> $1 3 \pm 1$ </td><td> $2 5 \pm 2$ </td><td>30 ± 2</td><td>17%</td><td>10.9</td></tr><tr><td>Published</td><td>Top 10</td><td> $1 4 \pm 2$ </td><td> $1 9 \pm 3$ </td><td>34 ± 2</td><td>17%</td><td>11.9</td></tr><tr><td>Neg. log-odds</td><td>All</td><td> $1 8 \pm 2$ </td><td> $2 8 \pm 2$ </td><td>44 ± 2</td><td>27%</td><td>15.3</td></tr><tr><td>Neg. log-odds</td><td>Top 10</td><td> ${ \bf 2 0 \pm 3 }$ </td><td> ${ \bf 3 2 \pm 3 }$ </td><td>46 ± 3</td><td>28%</td><td>13.9</td></tr><tr><td rowspan="4">Grad-Dot</td><td>Published</td><td>All</td><td> $3 \pm 2$ </td><td> $7 \pm 2$ </td><td>7±3</td><td>8%</td><td>3.1</td></tr><tr><td>Published</td><td>Top 10</td><td> $4 \pm 1$ </td><td> $6 \pm 2$ </td><td> $1 2 \pm 4$ </td><td>10%</td><td>5.5</td></tr><tr><td>Neg. log-odds</td><td>All</td><td> $\hphantom { 0 } 8 \pm 2$ </td><td> $2 2 \pm 2$ </td><td> $2 2 \pm 3$ </td><td>17%</td><td>8.4</td></tr><tr><td>Neg. log-odds</td><td>s Top 10</td><td> $1 1 \pm 1$ </td><td> $1 6 \pm 2$ </td><td> $2 3 \pm 2$ </td><td>18%</td><td>7.7</td></tr><tr><td colspan="2">BM25</td><td></td><td> $3 \pm 1$ </td><td> $5 \pm 1$ </td><td> $1 5 \pm 5$ </td><td>76%</td><td>2.4</td></tr><tr><td colspan="2">Ground-truth proponents</td><td></td><td> $4 ^ { \dagger }$ </td><td> $4 \pm 4$ </td><td> $9 \pm 3$ </td><td>100%</td><td>1.1</td></tr></table>

• EK-FAC preconditions with the cross-entropy EK-FAC of the fine-tuned model, damped by 0.01 times each module’s mean eigenvalue (the best of $1 0 ^ { - 5 }$ to 1 on 200 held-out novel facts). As published, it scores the training document’s cross-entropy summed over its answer tokens; EK-FAC (negative log-odds) scores the negative log-odds instead.

• Grad-Dot is the same pair of scores without the preconditioner.

The top 10 tokens variants score a document by the mean of its 10 largest per-position terms over input and answer positions. BM25 (BM25+) ranks documents by whitespace tokens of the input and target, with the mask token removed.

Remove-and-retrain. We draw 50 novel facts (seed 0) from those the fine-tuned model already answers correctly at step 30,000, so that none is a late fluctuation. Their proponents total 11,522 training examples. Each method ranks the full training corpus for each fact. To give every method exactly the same budget B, we remove documents round-robin: each fact’s next-ranked document in turn, skipping documents already removed, until B documents are gone. We use $B \in \{ 2 , 0 0 0 , 5 , 0 0 0 , \hat { 1 1 } , 5 2 \bar { 2 } \}$ }; the largest budget equals the size of the proponent union. For each removal set, we fine-tune mT5-small from the pretrained checkpoint with the same recipe at eight seeds (2001–2008). We also retrain a no-removal control at the same seeds, which reaches 80% to 92% top-1 on the 50 facts (86% on average). The reported drop is control accuracy minus retrained accuracy, paired by seed, averaged over the seeds, ± one standard error. BM25 and the ground-truth removal sets do not depend on the loss, so we reuse their earlier retrains against the same controls: two seeds at $B = 2 , 0 0 { \bar { 0 } }$ and 5,000 (one for the ground truth at 2,000) and three at 11,522.

Retraining results. Table 16 gives the values plotted in Figure 7, together with the remaining variants of each method. Four observations qualify the main-text result:

• The loss change helps every method; top-10 selection is roughly neutral. Paired by seed at $B = 1 1 , 5 2 2$ , negative log-odds adds $3 2 \pm 3$ points to TRAK, $1 5 \pm 1$ to EK-FAC, and $1 6 \pm 3$ to Grad-Dot. Under negative log-odds, keeping only the top 10 tokens adds 2 to 4 points to EK-FAC at every budget and changes Grad-Dot by −6 to +3.

• Removing the annotated proponents matters little. Removing every ground-truth proponent drops accuracy on the 50 facts by only 9 points. Removing BM25’s selection, which contains 76% of the proponents, drops it by 15 points, less than removing the selections of TRAK, EK-FAC, or Grad-Dot under negative log-odds. Park et al. (2023) found the same on FTRACE-TREX: BM25 retrieves the proponents far better than TRAK, yet removing TRAK’s top-ranked documents lowers accuracy more. The documents that matter to the retrained model are largely not the annotated ones.

• Larger accuracy drops come with forgetting of collateral facts. The negative-log-odds rankings also lower accuracy on other facts of the same relations, by 8 (Grad-Dot) to 15 (EK-FAC) points. Their highest-ranked documents include ones that teach a relation’s answer pattern, not only one fact’s evidence. The drop on the 50 facts is still about three times this collateral, and facts in other relations are unaffected.

## Departures from Park et al. (2023).

• Ensemble. In our results, TRAK uses one model rather than an ensemble of M = 10 trained on random subsets.

• Answer tokens only. Every loss excludes the target’s format tokens on both the query and the training-document side (see Scoring).

• Baseline for the drop. We measure the drop against a retrained no-removal control rather than against the original model taken as 100% accurate. A model retrained at a new seed answers only about 86% of facts selected on seed 1001, and the 100% baseline would count that shortfall as removal damage.

• Removal budget. Every method removes the same number of documents, rather than the union of each fact’s top 500, whose size varies by method (4,800 to 18,200 documents in our runs).