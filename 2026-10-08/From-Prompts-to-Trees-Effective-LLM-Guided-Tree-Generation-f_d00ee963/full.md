# From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification

Yue Qiu<sup>1∗</sup>, Zekang Du<sup>1∗</sup>, Yiqun Diao<sup>2</sup>, Bingsheng He<sup>2</sup>, Qinbin Li<sup>1†</sup>

<sup>1</sup>School of Computer Science and Technology, Huazhong University of Science and Technology

<sup>2</sup>School of Computing, National University of Singapore

{yueqiu, zekangdu, qinbin}@hust.edu.cn,

yiqun@comp.nus.edu.sg, dcsheb@nus.edu.sg

## Abstract

While Large Language Models (LLMs) possess rich world knowledge and impressive generalization capabilities, their direct application to tabular data classification is hindered by high inference costs and limited interpretability. In contrast, decision trees are fast and transparent but often underperform in low-data regimes. In this work, we propose a novel framework that bridges these paradigms by distilling LLM knowledge into interpretable decision trees under a few-shot learning setting. Instead of directly prompting the LLM to generate full trees, which is often unstable and inefficient, we develop a three-stage paradigm that prompts the LLM to generate rules and organize the rules into a tree. Experiments on multiple real-world tabular datasets demonstrate that our method achieves superior accuracy and interpretability with significantly lower prompting overhead compared to existing baselines.

## 1 Introduction

Large Language Models (LLMs) (OpenAI, 2023; Touvron et al., 2023; DeepSeek-AI et al., 2024; Naveed et al., 2025; Bai et al., 2023) have demonstrated remarkable capabilities in various domains, including code generation, planning, and complex problem-solving tasks. Despite these advancements, their potential remains significantly underexplored in the context of traditional tabular data classification (Shwartz-Ziv and Armon, 2022; Borisov et al., 2024). Tabular data is still the widely used format of data in reality, and is ubiquitous in critical domains such as healthcare diagnostics, financial risk assessment, and customer behavior prediction.

Although LLMs inherently possess extensive knowledge, their direct application to tabular data classification faces two prominent challenges: (1)

limited interpretability, making it difficult to provide intuitive justifications for classification decisions, and (2) inefficiency in inference, which restricts their applicability in resource-constrained and real-time scenarios. Decision trees (Loh, 2011; Chen and Guestrin, 2016; Ke et al., 2017; Rokach and Maimon, 2014; Quinlan, 1986, 1993), on the other hand, are widely adopted for tabular data classification due to their inherent interpretability, high computational efficiency, and ability to capture non-linear relationships between features. It has been a dominant model for tabular data classification for a long period. However, traditional tree training requires substantial labeled data to capture non-linear feature interactions. While LLMs can extract prior semantic knowledge from textified features to compensate for this data scarcity, combining them as a knowledge source and an efficient executor remains a challenge in practical few-shot settings.

This raises an important question: can we present the knowledge of LLMs by decision trees for tabular data classification in few-shot settings? If successful, such a tree representation would simultaneously inherit the LLM’s rich prior knowledge while retaining the interpretability and fast inference of decision trees, thereby addressing the key limitations of directly applying LLMs to tabular classification. A straightforward approach is to prompt the LLM once to directly generate a tree to represent its knowledge. However, this yields unreliable results due to the inherent complexity and instability of long-context reasoning. Another method is to generate the reasoning process by prompting multiple times. For example, Tree-of-Thought (Yao et al., 2023) approaches involve recursively prompting LLMs to generate the reasoning paths in a tree structure. Nevertheless, such methods suffer from substantial prompting overhead when applied to tabular data classification, rendering them impractical for wide adoption.

In this work, we propose LLMT, a modular and efficient framework that distills LLM knowledge into decision trees through a three-stage pipeline: (1) Rule Generation, where the LLM produces structured, machine-parseable classification rules; (2) Tree Assembly, where these rules are organized into a tree based on Gini impurity computed from a small labeled dataset; and (3) Leaf Refinement, where the LLM corrects noisy leaf decisions. Unlike prompting-based methods such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), which construct a reasoning chain or tree in a single pass or through recursive expansion, LLMT breaks the task into distinct stages. Our experiments show that this modular design improves both stability and efficiency.

Our contributions are as follows:

• We introduce a novel three-stage distillation paradigm that bridges the gap between expensive, opaque LLM reasoning and efficient, interpretable tree models.

• We provide a formal generalization analysis of LLMT under explicit structural assumptions, clarifying its statistical properties.

• We conduct extensive experiments on 11 datasets with 17 baselines, reporting accuracy and efficiency across different settings.

• We show that LLMT achieves consistent and significant accuracy improvements, large token-cost savings, and interpretable outputs in the few-shot setting.

## 2 Background and Related Work

## 2.1 Decision Trees

Decision trees (Loh, 2011; Rokach and Maimon, 2014; Quinlan, 1986, 1993) are among the most fundamental models in machine learning, widely used in tasks requiring interpretable decisionmaking. A decision tree forms a hierarchy of if– then rules: internal nodes test features, branches denote test outcomes, and leaves assign class labels. Trees are learned by recursively selecting splits that reduce impurity, such as Gini impurity (Loh, 2011).

## 2.2 LLMs for Tabular Data

Tabular data is one of the most prevalent formats in real-world applications and can be naturally converted into text by enumerating feature-value pairs. Recent work has explored LLMs for tabular prediction via fine-tuning, instruction tuning, synthetic data generation, or direct prompting (Wang et al., 2024; Herzig et al., 2020; Zhang et al., 2023; Hegselmann et al., 2023; Jaitly et al., 2023; Fang et al., 2024; Dinh et al., 2022; Wen et al., 2024). TabLLM (Hegselmann et al., 2023) finetunes T0 (Sanh et al., 2022) with a few tabular examples, while InsightTab (Yuan et al., 2025) and SumBoost (Manikandan et al., 2023) use LLMs directly as tabular classifiers with improved prediction pipelines. Although effective, these methods rely on LLMs during inference, leading to limited interpretability and high prediction cost.

Several recent works further investigate whether LLMs can generate rule-based or tree-like structures for tabular data (Knauer et al., 2025; Carrasco et al., 2025; Xiong et al., 2024). Direct-ZSDT (Knauer et al., 2025) and Step-ZSDT (Carrasco et al., 2025) build decision trees zero-shot but cannot use labeled examples to guide construction. GPTree (Xiong et al., 2024) uses LLMs to pre-generate rule candidates from raw training samples, making it more suitable for data-rich settings but less robust in few-shot regimes where limited or noisy examples may bias rule generation.

Other related methods combine LLMs with downstream estimators. FeatLLM (Han et al., 2024) uses LLMs for feature engineering before training a linear model, while DeLTa (Ye et al., 2025) uses LLM-refined rules to guide residual error correction in tree ensembles. In contrast, our method aims to generate a standalone decision tree: it first extracts structured rules from the LLM’s prior knowledge and then organizes them statistically using few-shot examples, achieving interpretability, low inference cost, and data-aware tree construction.

## 2.3 Prompting with Structured Reasoning

Few-shot prompting is a widely adopted strategy to elicit reasoning capabilities from LLMs using only a handful of labeled examples. Prior work (Wei et al., 2022; Yao et al., 2023; Ning et al., 2024; Besta et al., 2024) has shown that making the intermediate reasoning process explicit in the prompt significantly enhances LLM performance. Chainof-Thought (CoT) prompting (Wei et al., 2022) introduces the idea of decomposing complex tasks into a sequence of intermediate reasoning steps, leading to better accuracy.

<table><tr><td>Features</td><td>LLMT Direct-ZSDT Step-ZSDT</td><td></td><td></td><td></td><td></td><td></td><td></td><td>CoT-Tree ToT-Tree IO-Tree TabLLM InsightTab SumBoost FeatLLM DeLTa</td><td></td><td></td><td></td><td>GPTree</td></tr><tr><td>Few-shot ready (no finetune)</td><td></td><td></td><td>X</td><td></td><td></td><td></td><td>X</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Token-efficient build</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td></td></tr><tr><td>Tree generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td><td></td><td></td></tr><tr><td>No LLM at inference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td><td></td><td></td></tr><tr><td>Rule-first (vs path-first)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>x</td><td></td><td></td><td></td></tr><tr><td>Sample-level privacy preservation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td><td></td><td></td></tr><tr><td>Theoretical generalization bound</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td></td></tr></table>

Table 1: Comparison of our LLM-guided tree method (LLMT) with other related studies

Building on this, Tree-of-Thoughts (ToT) (Yao et al., 2023) generalizes CoT by considering multiple branching reasoning paths, enabling more robust and explorative problem-solving. Skeletonof-Thought (SoT) (Ning et al., 2024) further refines this approach by prompting the LLM to first generate a high-level reasoning structure and then elaborate on each component in parallel.

Positioning. Existing approaches fall into three categories: (i) using LLMs directly as in-context tabular classifiers (Hegselmann et al., 2023; Yuan et al., 2025; Manikandan et al., 2023); (ii) using LLMs to generate features, rules, or residual corrections for downstream models (Han et al., 2024; Ye et al., 2025); and (iii) prompting LLMs to generate reasoning paths or tree-like structures (Wei et al., 2022; Yao et al., 2023; Xiong et al., 2024). LLMT differs from these by separating knowledge extraction from tree construction: it extracts structured rules from feature semantics, assembles them statistically with few-shot labeled examples, and then performs leaf refinement. Unlike data-driven rulegeneration methods such as GPTree (Xiong et al., 2024), LLMT does not rely on many raw training samples to induce rules, making it better suited to few-shot settings. Moreover, LLMT preserves sample-level privacy by deriving rules solely from feature semantics without transmitting raw data values to the LLM. Our design avoids using the LLM as the final classifier and yields an explicit decision tree with interpretability and low inference cost. Table 1 summarizes the comparison.

## 3 Motivation

We start by conducting experiments to gain insights into the performance and capabilities of LLMs in the context of tabular data classification. Our observations highlight both the strengths and limitations of LLMs, motivating our proposed approach.

Observation 1: LLMs have rich background knowledge for tabular data classification.

We assess the predictive performance of four LLMs on public tabular datasets under two scenarios: 1) original, where we convert the tabular data into text and explicitly provide feature names and task descriptions in the query prompts; 2) masked, where feature names and task descriptions are omitted (see Appendix F for details.) The experimental results on two public tabular datasets Diabetes (Smith et al., 1988) and Spambase (Hopkins et al., 1999), summarized in Figure 1 under zero-shot prompting conditions, clearly show that the prediction accuracy of LLMs substantially deteriorates when metadata is removed. To minimize the potential influence of pretraining memorization, we extend this comparison to six newer datasets, with complete results reported in Table 7 of Appendix F.1.

![](images/7537f447b18fac47f73f9d030d768ba4228840b3a323548f74144797e3adf886.jpg)  
Figure 1: The test accuracy of LLMs with or without the metadata about the dataset.

![](images/4b279828d266ca12c345f51326c9aa0f5ae77e112455bed893627dc76b4f54cc.jpg)  
(a) Diabetes

![](images/05dbe9931c706f3b88b06370f90749bf1efc38a3d66e480ba43883f243863e5b.jpg)  
(b) Spambase  
Figure 2: Comparison of Utility and Accuracy between generated rule sets and path sets

Observation 2: Compared to generating long reasoning paths, LLMs are better at producing discrete rules for tabular data classification.

Recent studies (Han et al., 2024) suggest that LLMs excel at feature engineering. To further investigate this capability, we conduct a comparative analysis of LLM performance in generating long reasoning paths versus discrete rules for tabular data classification. Reasoning paths can be seen as a series of rule-checks that traverse from the root to a leaf node in the decision tree. For a controlled comparison, we prompt LLMs to generate reasoning-path sets and discrete-rule sets containing the same total number of atomic feature conditions. We then evaluate the generated sets using Set Utility and Set Accuracy across eight datasets (see Appendix F.2 for detailed settings and complete results). The representative results, shown in Figure 2, reveal that rule generation achieves higher Set Utility and Set Accuracy than path generation. These findings indicate that while LLMs struggle to produce coherent and complete hierarchical reasoning paths in few-shot settings, they are particularly effective at generating accurate and interpretable individual classification rules.

These findings motivate our design choice: rather than asking the LLM to produce full reasoning paths, which are often unstable, lengthy, and error-prone, we instead focus on discrete rule generation and organization.

## 4 Methodology

## 4.1 Problem Statement

We consider the problem of distilling the knowledge embedded in an LLM into an interpretable and efficient decision tree for tabular data classification. Formally, let $\mathbfcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ denote a tabular dataset with n samples, where each input $x _ { i } \in \mathbb { R } ^ { m }$ is a feature vector of m attributes and $y _ { i } \in \{ 1 , 2 , \ldots , K \}$ is a categorical label over $K$ classes. The goal is to learn a decision tree $T$ such that $T ( x )$ approximates the classification capability of the LLM while maintaining interpretability and low computational overhead.

Crucially, we assume access to only a small number of labeled samples (few-shot setting) and the ability to query an LLM via prompting. This constraint reflects practical scenarios where labeled tabular data is scarce but the model’s background knowledge is rich.

## 4.2 Framework Overview

To tackle the above challenge, we propose a novel three-stage distillation framework named LLMT that efficiently extracts and structures the LLM’s latent decision-making knowledge into a decision tree. Instead of prompting the LLM to generate the reasoning paths as a tree, we ask it to generate discrete rules and then organize them as a tree in a statistical way. As illustrated in Figure 3, our framework comprises the following components: (1) Rule Generation: We prompt the LLM with carefully designed instructions to generate discrete classification rules, along with a self-assessed confidence score for each rule. (2) Tree Assembly: We systematically organize the induced rules into a tree structure by selecting high-quality rules in a layer-wise manner using Gini-based splitting criteria on the small labeled dataset. (3) Leaf Refinement: We prompt the LLM to analyze each rule path, evaluate whether rules are satisfied, and assign confidence scores to possible labels, creating a probability distribution that quantifies prediction certainty. Please refer to Appendix D for an example of the execution process.

Implementation Details. We adopt a level-wise heap indexing scheme: the root is indexed as 1, its left and right children as 2 and 3, and so on in breadth-first order. At each node, we consider rules not yet used along the current path. Candidate rules are grouped by their confidence scores into multiple bins, processed in descending order. Within each group, ties are broken by highest Gini gain and lowest index to ensure determinism. The process terminates when either the maximum tree depth is reached or no remaining rule yields a positive gain.

## 4.3 Rule Generation

In the first stage, we prompt the LLM to generate a structured set of candidate rules. Each rule follows the grammar:

$$
f _ { j } \circ \mathsf { p } \theta ,
$$

where $0 { \mathsf { p } } \in \{ < , \geq , = , \neq \}$ , θ is a numerical threshold or categorical value, and $f _ { j }$ represents the $j \mathrm { - t h }$ feature in the dataset (e.g., age ≥ 20). Categorical features are encoded using one-vs-rest splits.

The LLM receives (i) a system prompt detailing task constraints and rules, and (ii) a dataset prompt containing feature names, types, and datasetagnostic rule set examples for formatting.

The model outputs rules with self-assessed confidence scores in [1, 10].

## 4.4 Tree Assembly

The objective of this stage is to construct a decision tree $T$ using the rule set R and few-shot examples $\mathcal { D } _ { \mathrm { t r a i n } }$ . Our idea is to put the important rules on the top of the tree so that they can be shared in more reasoning paths, while using the knowledge in few-shot examples to adjust the node positions.

![](images/bd6bb4a553b66e8306f336fa7e4d54d8a1842ef3e6d7f1d9f63cf32f07375236.jpg)  
(a) CoT-Tree

![](images/005f289a84b45c3c2600ea96fbaf66e074cfbf880c7666240ad3d9de08903136.jpg)  
(b) ToT-Tree

![](images/3894dc87485b2e59840b5cefd0321aea4e65e61a3ad9d44e45e892793c5fdf60.jpg)  
Figure 3: Comparison between our method and prompting-based methods for generating reasoning trees. Instead of (1) asking the LLM to directly generate long reasoning paths (CoT-Tree), or (2) asking the LLM to generate the next thought based on a long reasoning path, our method prompts the LLM to generate discrete rules, organizes them into a tree using statistical methods, and then uses the LLM for final refinement.

## 4.5 Leaf Refinement

Rule Grouping and Selection. The process proceeds in a level-wise fashion through the tree. For each node at depth level l, we exclude rules already used on the path $R _ { \mathrm { p a t h } }$ to that node. We then group the remaining candidates by their confidence scores (rules with the same confidence value form one group). Starting with the highest-confidence group, we select the optimal rule $R _ { k }$ by minimizing the total Gini impurity of the resulting splits. Specifically, suppose the current tree node corresponds to a subset $\mathcal { D } _ { p } \subseteq \mathcal { D } _ { \mathrm { t r a i n } }$ of samples. Then, we aim to find a rule $R _ { k }$ as the node $T [ i ]$ such that

$$
\arg \operatorname* { m i n } _ { R _ { k } } \frac { | \mathcal { D } _ { l } ^ { k } | } { | \mathcal { D } _ { p } | } I _ { G } ( \mathcal { D } _ { l } ^ { k } ) + \frac { | \mathcal { D } _ { r } ^ { k } | } { | \mathcal { D } _ { p } | } I _ { G } ( \mathcal { D } _ { r } ^ { k } ) ,\tag{1}
$$

where $\mathcal { D } _ { l } ^ { k }$ is the left child set split by $R _ { k }$ and $\mathcal { D } _ { r } ^ { k }$ is the right child set split by $R _ { k }$ . Intuitively, the following proposition holds:

Proposition 4.1 (Expected Impurity Reduction). Let $r ^ { \ast } \in \mathcal { R }$ be the selected rule at node t, splitting dataset $\mathcal { D } _ { p } \subseteq \mathcal { D } _ { t r a i n }$ into $\mathcal { D } _ { l } , \mathcal { D } _ { r }$ . Then, the expected impurity after splitting satisfies:

$$
\mathbb { E } [ \mathrm { G i n i } ( \mathcal { D } _ { l } , \mathcal { D } _ { r } ) ] \leq \mathrm { G i n i } ( \mathcal { D } _ { p } ) ,
$$

with equality if and only if the split yields no information gain.

While the internal structure of the tree is built from data-driven rule selection, the final classification decisions at leaf nodes may still be noisy or misaligned with LLM’s original knowledge. To address this, we introduce a refinement step that evaluates and corrects the leaf nodes if necessary.

For every path p from the root to a leaf in T, we construct a natural language query describing the rule chain and prompt the LLM to evaluate its logical soundness. The LLM returns confidence scores $\mathbf { C _ { p } }$ (a probability distribution over all possible labels). We identify the label $y ^ { * }$ with the highest confidence score $c ^ { * }$ . If $c ^ { * } > \tau$ (a predefined threshold) and $y ^ { \ast }$ differs from the current leaf label, we adjust the classification at that leaf node to $y ^ { * }$ to improve consistency and generalization. As we show in Section 5.5, our leaf refinement process usually improves the accuracy of the tree. Using the PAC theory (Blumer et al., 1989), we have the following theorem for the learned tree:

Theorem 4.2 (Generalization Error Bound). Consider a decision tree T constructed using LLMT of maximum depth dfrom afinite rule set R with size K. Given n training samples, for any $\delta > 0 ,$ , with probability at least 1 − δ, the generalization error

R(T) satisfies:

$$
R ( T ) ~ \le ~ { \hat { R } } ( T ) ~ + ~ \sqrt { \frac { 2 \Bigl ( \log { P ( K , 2 ^ { d } - 1 ) } + \log { \frac { 2 } { \delta } } \Bigr ) } { n } }\tag{2}
$$

$$
\leq { \hat { R } } ( T ) + { \sqrt { \frac { 2 \left( \left( 2 ^ { d } - 1 \right) \log K + \log { \frac { 2 } { \delta } } \right) } { n } } } ,\tag{3}
$$

where $P ( K , 2 ^ { d } - 1 ) = K ! / ( K - ( 2 ^ { d } - 1 ) ) ! .$

The proof is available at Appendix A. While this PAC-style bound can be numerically loose in the extreme few-shot regime (n is small), it provides crucial structural guidance. The bound explicitly shows that tree depth d is the dominant complexity driver (via the $2 ^ { d }$ dependence), justifying our shallow tree design. Moreover, by distilling the LLM into a finite rule set K, LLMT induces a finite, statistically analyzable hypothesis space, distinguishing it from mathematically intractable black-box prompting pipelines. As shown in our experiments (Appendix G.3), LLMT consistently outperforms baseline methods across varying tree depths.

## 4.6 Ensemble Extension

For high-dimensional datasets, we optionally extend LLMT into an ensemble termed LLMT Forest, since ensembles can improve model capacity while reducing the overfitting risk of a single deep tree (Breiman, 2001; Chen and Guestrin, 2016). To construct LLMT Forest, we randomly permute and partition the features into disjoint, approximately equal-sized subsets, train one LLMT tree on each subset using the same few-shot samples, and aggregate their predictions by majority vote.

## 5 Evaluation

We report representative results in the main paper. In the Appendix, we also include results on more baselines, more datasets and different tree depths.

## 5.1 Experimental Setup

Datasets We evaluate our method on 11 widely used public tabular datasets including Nursery (Rajkovic, 1989), Diabetes (Smith et al., 1988), Spambase (Hopkins et al., 1999), Abalone (Nash et al., 1994), Blood (Yeh, 2008), Iris (Fisher, 1936), Breast (Patrício et al., 2018), Glioma (Tasci et al., 2022), Communities (Redmond, 2002), Ecom (Rubachev et al., 2025), and Myocardial (Golovenkin et al., 2020). Among them, we select Diabetes, Spambase, and Nursery as the major datasets for evaluation across various settings. These datasets span diverse domains and vary widely in sample size and feature dimensionality, providing a comprehensive testbed.

Baselines We compare LLMT against the following baselines, organized by methodological category: (1) zero-shot decision-tree methods: DirectZSDT (Knauer et al., 2025), StepZSDT (Carrasco et al., 2025); (2) conventional machinelearning methods: LogReg (logistic regression; Cox, 1958), SVM (support vector machine; Cortes and Vapnik, 1995), and CART (Loh, 2011); (3) tree-ensemble methods: Random Forest (Breiman, 2001), XGBoost (Chen and Guestrin, 2016), Cat-Boost (Prokhorenkova et al., 2018), and Light-GBM (Ke et al., 2017); (4) LLM prompting methods: IO-Tree, CoT-Tree (Wei et al., 2022), and ToT-Tree (Yao et al., 2023); (5) pretrained blackbox predictors: TabPFN (Hollmann et al., 2025) and TabLLM (Hegselmann et al., 2023); and (6) recent LLM-assisted tabular learning methods: FeatLLM (Han et al., 2024), DeLTa (Ye et al., 2025), and GPTree (Xiong et al., 2024). Due to space constraints, additional comparisons involving Random Forest, CatBoost, TabPFN, and TabLLM are reported in Appendix G.

Settings Experiments are run on a Linux server with 4× Intel Xeon Gold 5117 CPUs and 4× Nvidia Tesla V100 GPUs. We use Qwen2.5-72B-Instruct via TogetherAI (Together AI, 2024) by default, and provide all prompts in the Appendix. We set the maximum tree depth to 3 or 4 depending on the dataset, and set the number of rules as $K = \operatorname* { m a x } ( 1 0 , 2 ^ { d } - 1 )$ , where d is the maximum depth. The leaf-refinement threshold is $\tau = 0 . 7$ for all datasets. We repeat each experiment for 10 trials with randomly sampled training sets and a fixed test set of 100 samples, and report mean ± standard deviation. Additional details are in Appendix F.

## 5.2 Effectiveness

We evaluate LLMT against all baselines in terms of classification accuracy. Table 2 reports the results with #shots=2 per class, while Table 3 reports results under varying numbers of training examples on three representative datasets.

LLMT consistently outperforms almost all baselines, with more than 10% improvement over the strongest baseline in many settings. It also achieves robust gains over recent LLM-assisted methods such as FeatLLM, DeLTa, and GPTree: FeatLLM and DeLTa tend to overfit in extreme few-shot regimes, while GPTree can be affected by datainduced bias due to its reliance on large-scale training data. These results show the benefit of leveraging LLM knowledge in a structured, data-aware manner. In addition, ToT-Tree often performs worse than CoT-Tree, suggesting that recursive node-level prompting introduces noise and inconsistency for structured tabular data. By contrast, LLMT uses LLM-generated rules as decision units, leading to more effective tree construction.

<table><tr><td>Dataset</td><td>DirectZSDT StepZSDT</td><td></td><td></td><td></td><td></td><td>T LogReg SVM CART XGBoost LightGBM IO-Tree CoT-Tree</td><td></td><td></td><td></td><td>e ToT-Tree FeatLLM DeLTa GPTree LLMT</td><td></td><td></td><td></td><td></td></tr><tr><td>Nursery</td><td>0.485</td><td>0.450</td><td>0.558</td><td>0.564</td><td>0.602</td><td>0.434</td><td>0.578</td><td>0.439</td><td>0.461</td><td>0.353</td><td>0.571</td><td>0.448</td><td>0.479</td><td>0.762</td></tr><tr><td>Diabetes</td><td>0.729</td><td>0.641</td><td>0.618</td><td>0.629</td><td>0.546</td><td>0.690</td><td>0.543</td><td>0.539</td><td>0.701</td><td>0.561</td><td>0.700</td><td>0.541</td><td>0.616</td><td>0.764</td></tr><tr><td>Spambase</td><td>0.645</td><td>0.732</td><td>0.790</td><td>0.783</td><td>0.723</td><td>0.600</td><td>0.733</td><td>0.723</td><td>0.739</td><td>0.727</td><td>0.802</td><td>0.747</td><td>0.661</td><td>0.811</td></tr><tr><td>Abalone</td><td>0.550</td><td>0.526</td><td>0.684</td><td>0.672</td><td>0.689</td><td>0.670</td><td>0.658</td><td>0.647</td><td>0.694</td><td>0.570</td><td>0.721</td><td>0.656</td><td>0.667</td><td>0.694</td></tr><tr><td>Blood</td><td>0.597</td><td>0.371</td><td>0.604</td><td>0.635</td><td>0.565</td><td>0.605</td><td>0.508</td><td>0.585</td><td>0.500</td><td>0.551</td><td>0.642</td><td>0.550</td><td>0.572</td><td>0.675</td></tr><tr><td>Iris</td><td>0.946</td><td>0.547</td><td>0.793</td><td>0.756</td><td>0.882</td><td>0.739</td><td>0.685</td><td>0.870</td><td>0.903</td><td>0.867</td><td>0.903</td><td>0.823</td><td>0.814</td><td>0.940</td></tr><tr><td>Breast</td><td>0.554</td><td>0.500</td><td>0.597</td><td>0.607</td><td>0.500</td><td>0.400</td><td>0.513</td><td>0.543</td><td>0.473</td><td>0.593</td><td>0.544</td><td>0.529</td><td>0.523</td><td>0.617</td></tr><tr><td>Glioma</td><td>0.504</td><td>0.743</td><td>0.760</td><td>0.755</td><td>0.592</td><td>0.560</td><td>0.680</td><td>0.637</td><td>0.598</td><td>0.652</td><td>0.724</td><td>0.643</td><td>0.678</td><td>0.760</td></tr><tr><td>Average</td><td>0.626</td><td>0.564</td><td>0.676</td><td>0.675</td><td>0.637</td><td>0.587</td><td>0.612</td><td>0.623</td><td>0.634</td><td>0.609</td><td>0.701</td><td>0.617</td><td>0.626</td><td>0.753</td></tr></table>

Table 2: Accuracy comparison of different methods when the number of training examples per class is two. Best performances are bolded, and our method’s performances, when second-best, are underlined.
<table><tr><td rowspan="2">Datasets #Shots</td><td colspan="5">Nursery</td><td colspan="5">Diabetes</td><td colspan="5">Spambase</td></tr><tr><td>3</td><td>6</td><td>12</td><td>24</td><td>48</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td><td>2</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td>DirectZSDT</td><td>0.485</td><td>0.485</td><td>0.485</td><td>0.485</td><td>0.485</td><td>0.729</td><td>0.729</td><td>0.729</td><td>0.729</td><td>0.729</td><td>0.645</td><td>0.645</td><td>0.645</td><td>0.645</td><td>0.645</td></tr><tr><td>StepZSDT</td><td>0.450</td><td>0.450</td><td>0.450</td><td>0.450</td><td>0.450</td><td>0.641</td><td>0.641</td><td>0.641</td><td>0.641</td><td>0.641</td><td>0.732</td><td>0.732</td><td>0.732</td><td>0.732</td><td>0.732</td></tr><tr><td>LogReg</td><td>0.505</td><td>0.558</td><td>0.670</td><td>0.778</td><td>0.855</td><td>0.542</td><td>0.618</td><td>0.674</td><td>0.730</td><td>0.748</td><td>0.685</td><td>0.790</td><td>0.785</td><td>0.782</td><td>0.835</td></tr><tr><td>SVM</td><td>0.500</td><td>0.564</td><td>0.661</td><td>0.781</td><td>0.833</td><td>0.542</td><td>0.629</td><td>0.660</td><td>0.671</td><td>0.710</td><td>0.685</td><td>0.783</td><td>0.754</td><td>0.771</td><td>0.807</td></tr><tr><td>CART</td><td>0.351</td><td>0.602</td><td>0.786</td><td>0.803</td><td>0.801</td><td>0.519</td><td>0.546</td><td>0.624</td><td>0.663</td><td>0.686</td><td>0.639</td><td>0.723</td><td>0.743</td><td>0.705</td><td>0.757</td></tr><tr><td>XGBoost</td><td>0.460</td><td>0.434</td><td>0.744</td><td>0.784</td><td>0.808</td><td>0.690</td><td>0.690</td><td>0.626</td><td>0.708</td><td>0.708</td><td>0.600</td><td>0.600</td><td>0.706</td><td>0.706</td><td>0.762</td></tr><tr><td>LightGBM</td><td>0.378</td><td>0.578</td><td>0.780</td><td>0.792</td><td>0.803</td><td>0.600</td><td>0.543</td><td>0.631</td><td>0.676</td><td>0.680</td><td>0.652</td><td>0.733</td><td>0.759</td><td>0.717</td><td>0.769</td></tr><tr><td>IO-Tree</td><td>0.378</td><td>0.439</td><td>0.519</td><td>0.511</td><td>0.516</td><td>0.525</td><td>0.539</td><td>0.605</td><td>0.649</td><td>0.573</td><td>0.661</td><td>0.723</td><td>0.671</td><td>0.743</td><td>0.711</td></tr><tr><td>CoT-Tree</td><td>0.407</td><td>0.461</td><td>0.617</td><td>0.559</td><td>0.591</td><td>0.665</td><td>0.701</td><td>0.700</td><td>0.696</td><td>0.733</td><td>0.699</td><td>0.739</td><td>0.705</td><td>0.740</td><td>0.731</td></tr><tr><td>ToT-Tree</td><td>0.379</td><td>0.353</td><td>0.357</td><td>0.370</td><td>0.411</td><td>0.645</td><td>0.561</td><td>0.583</td><td>0.563</td><td>0.551</td><td>0.673</td><td>0.727</td><td>0.722</td><td>0.695</td><td>0.720</td></tr><tr><td>FeatLLM</td><td>0.539</td><td>0.571</td><td>0.717</td><td>0.761</td><td>0.781</td><td>0.658</td><td>0.700</td><td>0.717</td><td>0.725</td><td>0.730</td><td>0.769</td><td>0.802</td><td>0.814</td><td>0.813</td><td>0.837</td></tr><tr><td>DeLTa</td><td>0.357</td><td>0.448</td><td>0.577</td><td>0.673</td><td>0.729</td><td>0.310</td><td>0.541</td><td>0.624</td><td>0.637</td><td>0.630</td><td>0.394</td><td>0.747</td><td>0.747</td><td>0.811</td><td>0.823</td></tr><tr><td>GPTree</td><td>0.326</td><td>0.479</td><td>0.780</td><td>0.741</td><td>0.720</td><td>0.515</td><td>0.616</td><td>0.646</td><td>0.643</td><td>0.594</td><td>0.641</td><td>0.661</td><td>0.677</td><td>0.746</td><td>0.644</td></tr><tr><td>LLMT (Ours)</td><td>0.762</td><td>0.762</td><td>0.784</td><td>0.818</td><td>0.840</td><td>0.662</td><td>0.764</td><td>0.770</td><td>0.770</td><td>0.770</td><td>0.782</td><td>0.811</td><td>0.836</td><td>0.839</td><td>0.843</td></tr></table>

Table 3: Mean accuracy over ten runs across varying #shots (standard deviation is available at Appendix G). DirectZSDT and StepZSDT are constant across #shots since they cannot utilize any training examples.

We further assess high-dimensional performance on three datasets with over 100 attributes. These results show that LLMT maintains its advantage in high-dimensional feature spaces and further support the effectiveness of its ensemble extension. Please refer to Appendix G.3 for detailed results.

## 5.3 Efficiency

We evaluate computational efficiency by measuring training time and token usage per prediction on three tabular datasets, as summarized in Table 4. We omit CART, XGBoost, and IO-Tree because they are already highly efficient but have relatively poor classification performance. DirectZSDT incurs high latency due to its two-stage dialogue, where the LLM first generates a tree and then converts it into executable Python code. StepZSDT is even more expensive, as it prompts the LLM to generate feature-specific rules for each node.

Compared with CoT-Tree, LLMT achieves comparable training time and token usage. Compared with ToT-Tree, LLMT avoids repeated planning and voting, reducing token usage by 10–22× and construction time by 1.6–3.6×. LLMT also has comparable computational cost to DeLTa, while being up to 10× faster than FeatLLM and more token-efficient than GPTree, which rely on repeated rule-parsing queries and sequential batch-wise insight generation, respectively.

## 5.4 Interpretability

Decision trees are intrinsically interpretable because each prediction follows an explicit root-toleaf path, making the decision process directly inspectable. Compared with coefficient-based, hybrid, or black-box models, standalone trees generally provide more intuitive explanations for individual predictions. We therefore compare methods that output standalone trees in terms of structural conciseness and split purity, measured by average node count and average Gini impurity, as reported in Figure 4 and Table 5, respectively.

<table><tr><td rowspan="2"></td><td colspan="3">Time (s)</td><td colspan="3">Token (#Input token, #Output token, #Total)</td></tr><tr><td>Diabetes</td><td>Nursery</td><td>Spambase</td><td>Diabetes</td><td>Nursery</td><td>Spambase</td></tr><tr><td>DirectZSDT</td><td>43.64</td><td>31.75</td><td>55.33</td><td>(2.9k, 1.1k, 4k)</td><td>(3.8k, 1.3k, 4.1k)</td><td>(3.4k, 1.4k, 4.8k)</td></tr><tr><td>StepZSDT</td><td>1006.67</td><td>1191.40</td><td>4036.18</td><td>(85.6k, 22.1k, 107.7k)</td><td>(128.3k, 27.2k, 155.5k)</td><td>(358.9k, 95.4k, 454.3k)</td></tr><tr><td>CoT-Tree</td><td>17.00</td><td>9.58</td><td>12.05</td><td>(2.7k, 1.3k, 4k)</td><td>(2.7k, 1.0k, 3.7k)</td><td>(5.6k, 0.9k, 6.5k)</td></tr><tr><td>ToT-Tree</td><td>33.39</td><td>33.22</td><td>64.56</td><td>(32.9k, 0.5k, 33.4k)</td><td>(35.4k, 0.6k, 36k)</td><td>(140k, 1.3k, 141.3k)</td></tr><tr><td>FeatLLM</td><td>259.07</td><td>416.16</td><td>324.06</td><td>(22.2k, 10.3k, 32.5k)</td><td>(24.4k, 16.3k, 40.7k)</td><td>(48.5k, 12.3k, 60.8k)</td></tr><tr><td>DeLTa</td><td>30.20</td><td>26.90</td><td>28.84</td><td>(3.1k, 3.3k, 6.4k)</td><td>(3.1k, 2.8k, 5.9k)</td><td>(5.2k, 3.1k, 8.3k)</td></tr><tr><td>GPTree</td><td>140.75</td><td>103.47</td><td>125.56</td><td>(12.5k, 2.3k, 14.8k)</td><td>(9.5k, 3.2k, 12.7k)</td><td>(27.6k, 2.5k, 30.0k)</td></tr><tr><td>LLMT</td><td>20.48</td><td>16.33</td><td>18.18</td><td>(3.1k, 0.2k, 3.3k)</td><td>(3.4k, 0.2k, 3.6k)</td><td>(6.0k, 0.4k, 6.4k)</td></tr><tr><td>Savings</td><td>1.63x</td><td>2.03x</td><td>3.56x</td><td>10.1x</td><td>10x</td><td>22.1x</td></tr></table>

Table 4: The training time (s) and #token in the #shots=32 setting. The savings of LLMT are computed against ToT-Tree in terms of #total tokens.

![](images/b0769f576c0993b76c1bdaf86ae9e2b52c782b1d3c046a89f4cd46f3dd9c5525.jpg)

Figure 4: Average number of nodes in trees generated by each method. Fewer nodes indicate a more concise tree.
<table><tr><td>Dataset</td><td>Direct-ZSDT</td><td>Step-ZSDT</td><td>CART</td><td>IO-Tree</td><td>CoT-Tree</td><td>ToT-Tree</td><td>GPTree</td><td>LLMT</td></tr><tr><td>Diabetes</td><td>0.422</td><td>0.420</td><td>0.414</td><td>0.397</td><td>0.397</td><td>0.403</td><td>0.425</td><td>0.396</td></tr><tr><td>Nursery</td><td>0.665</td><td>0.634</td><td>0.516</td><td>0.582</td><td>0.534</td><td>0.629</td><td>0.552</td><td>0.456</td></tr><tr><td>Spambase</td><td>0.378</td><td>0.478</td><td>0.360</td><td>0.341</td><td>0.361</td><td>0.351</td><td>0.396</td><td>0.352</td></tr><tr><td>Average</td><td>0.488</td><td>0.511</td><td>0.430</td><td>0.440</td><td>0.431</td><td>0.461</td><td>0.458</td><td>0.401</td></tr></table>

Table 5: Average Gini impurity of rules across different methods. Lower values indicate better rule quality.

LLMT produces the most concise trees on Diabetes and remains among the most compact methods on Nursery and Spambase. It also achieves the lowest average Gini impurity, performing best on

Diabetes and Nursery and close to the best method on Spambase. Overall, LLMT retains the intrinsic readability of a standalone decision tree while generating compact structures with high-quality rules.

## 5.5 Ablation and Sensitivity Study

We further analyze the effect of the Leaf Refinement stage and its confidence threshold τ . The threshold τ controls whether the LLM accepts a leaf node’s decision based on its confidence in the corresponding reasoning path. As shown in Figure 5, LLMT is generally stable across different τ values, and $\tau = 0 . 7$ usually achieves the best performance, suggesting a good balance between preserving the initial tree structure and applying LLM-based corrections. It indicates that LLMT does not require extensive hyperparameter tuning.

![](images/7410f6871e40caba98eb02bb700e284f5a9411a383a1b2c24dc7488f7360a17e.jpg)  
(a) Diabetes

![](images/2ca1fe41b73437a6950e3479b02df716e0f6e5bd1bf1d273b65b99627c3451a1.jpg)  
(b) Spambase

Figure 5: The accuracy of LLMT with different τ  
![](images/b188ffd8176c5d93418c2a7a503920d1e7c041751ba45632939f4466de2b727a.jpg)  
(a) Diabetes

![](images/ffc3af0ca2ef3b61a1071ec41fd33ad3887fe3e2f8ea37de36add7f18a8cf966.jpg)  
(b) Spambase  
Figure 6: The accuracy of LLMT with/without leaf refinement

We then remove LeafRefinement to evaluate its contribution on Diabetes and Spambase. Figure 6 shows consistent accuracy gains of 8% and 2%, respectively. Beyond accuracy, LLM verification may identify same-label sibling leaves for merging, simplifying the tree and mitigating spurious splits caused by limited few-shot data.

## 6 Conclusion

In this paper, we present a novel framework for distilling the knowledge of LLMs into interpretable and efficient decision trees for tabular data classification. Unlike prior approaches that prompt LLMs to directly generate full decision structures which often leads to instability and high inference costs, our method utilizes the ability of LLMs to generate discrete rules and organizes them as a tree statistically. By bridging the strengths of LLMs and symbolic models, our method offers a promising direction for building trustworthy AI systems that can reason over structured data.

## Limitations

Despite its strengths, our method assumes that the LLM possesses sufficient background knowledge relevant to the target dataset. If the dataset lies outside the knowledge scope of the LLM, then the generated rules—and consequently the constructed tree—may be inaccurate or ineffective. Moreover, our method inherits any biases in the underlying LLM, which may influence the rules extracted and affect downstream decisions. Caution must be exercised when deploying the distilled trees in highstakes applications, and future work should explore fairness-aware prompting and post-hoc auditing.

## Acknowledgements

This research is supported in part by National Natural Science Foundation of China (Grant No. 62502174). This research is also supported in part by the National Research Foundation, Singapore and Infocomm Media Development Authority under its Trust Tech Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore and Infocomm Media Development Authority.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, and 29 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. 2024. Graph of thoughts: Solving elaborate problems with large language models. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(16):17682–17690.

Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K. Warmuth. 1989. Learnability and the Vapnik-Chervonenkis dimension. Journal ofthe ACM, 36(4):929–965.

Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci. 2024. Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(6):7499–7519.

Leo Breiman. 2001. Random forests. Machine Learning, 45(1):5–32.

Lucas Carrasco, Felipe Urrutia, and Andrés Abeliuk. 2025. Zero-shot decision tree construction via large language models. arXiv preprint arXiv:2501.16247.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM.

Corinna Cortes and Vladimir Vapnik. 1995. Supportvector networks. Machine Learning, 20(3):273–297.

D. R. Cox. 1958. The regression analysis of binary sequences. Journal ofthe Royal Statistical Society: Series B (Methodological), 20(2):215–232.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, and 181 others. 2024. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437.

Tuan Dinh, Yuchen Zeng, Ruisu Zhang, Ziqian Lin, Michael Gira, Shashank Rajput, Jy-yong Sohn, Dimitris Papailiopoulos, and Kangwook Lee. 2022. LIFT: Language-interfaced fine-tuning for non-language machine learning tasks. In Advances in Neural Information Processing Systems, volume 35, pages 11763– 11784. Curran Associates, Inc.

Xi Fang, Weijie Xu, Fiona Anting Tan, Jiani Zhang, Ziqing Hu, Yanjun Qi, Scott Nickleach, Diego Socolinsky, Srinivasan Sengamedu, and Christos Faloutsos.

2024. Large language models (LLMs) on tabular data: Prediction, generation, and understanding— a survey. Transactions on Machine Learning Research.

R. A. Fisher. 1936. The use of multiple measurements in taxonomic problems. Annals ofEugenics, 7(2):179– 188.

S. E. Golovenkin, V. A. Shulman, D. A. Rossiev, P. A. Shesternya, S. Yu. Nikulina, Yu. V. Orlova, and V. F. Voino-Yasenetsky. 2020. Myocardial infarction complications. UCI Machine Learning Repository. Dataset.

Sungwon Han, Jinsung Yoon, Sercan O. Arik, and Tomas Pfister. 2024. Large language models can automatically engineer features for few-shot tabular learning. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17454–17479. PMLR.

Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. 2023. TabLLM: Few-shot classification of tabular data with large language models. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5549–5581. PMLR.

Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Eisenschlos. 2020. TaPas: Weakly supervised table parsing via pre-training. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4320–4333. Association for Computational Linguistics.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326.

Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt. 1999. Spambase. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C53G6X.

Muhammad Kamal Hossen and Mohammad Shorif Uddin. 2025. Students suspicious behaviors detection dataset for AI-powered online exam proctoring. Mendeley Data.

Sukriti Jaitly, Tanay Shah, Ashish Shugani, and Razik Singh Grewal. 2023. Towards better serialization of tabular data for few-shot classification with large language models. Preprint, arXiv:2312.12464.

Kaggle. 2025a. Predict the introverts from the extroverts. Playground Series, Season 5, Episode 7.

Kaggle. 2025b. Predicting loan payback. Playground Series, Season 5, Episode 11.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems, volume 30, pages 3146–3154. Curran Associates, Inc.

Ricardo Knauer, Mario Koddenbrock, Raphael Wallsberger, Nicholas M. Brisson, Georg N. Duda, Deborah Falla, David W. Evans, and Erik Rodner. 2025. “Oh LLM, I’m asking thee, please give me a decision tree”: Zero-Shot decision tree induction and embedding with large language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1196–1206. ACM.

Wei-Yin Loh. 2011. Classification and regression trees. WIREs Data Mining and Knowledge Discovery, 1(1):14–23.

Hariharan Manikandan, Yiding Jiang, and J. Zico Kolter. 2023. Language models are weak learners. In Advances in Neural Information Processing Systems, volume 36, pages 50907–50931.

Warwick Nash, Tracy Sellers, Simon Talbot, Andrew Cawthorn, and Wes Ford. 1994. Abalone. UCI Machine Learning Repository. Dataset.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2025. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology, 16(5):1–72. Article 106.

Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. 2024. Skeletonof-thought: Prompting LLMs for efficient parallel generation. In The Twelfth International Conference on Learning Representations.

OpenAI. 2023. GPT-4 technical report. Preprint, arXiv:2303.08774.

Miguel Patrício, José Pereira, Joana Crisóstomo, Paulo Matafome, Manuel Gomes, Raquel Seiça, and Francisco Caramelo. 2018. Using resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer, 18(1):29.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems, volume 31, pages 6638–6648. Curran Associates, Inc.

J. Ross Quinlan. 1986. Induction of decision trees. Machine Learning, 1(1):81–106.

J. Ross Quinlan. 1993. C4.5: Programs for Machine Learning, first edition. Morgan Kaufmann.

Vladislav Rajkovic. 1989. Nursery. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5P88W.

Michael Redmond. 2002. Communities and crime. UCI Machine Learning Repository. Dataset.

Lior Rokach and Oded Maimon. 2014. Data Mining with Decision Trees: Theory and Applications, second edition, volume 81. World Scientific.

Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, and Artem Babenko. 2025. TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. In International Conference on Learning Representations, volume 2025, pages 35166–35202.

Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M. Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, and 23 others. 2022. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.

Ravid Shwartz-Ziv and Amitai Armon. 2022. Tabular data: Deep learning is not all you need. Information Fusion, 81:84–90.

Jack W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes. 1988. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Annual Symposium on Computer Application in Medical Care, pages 261–265. PMCID: PMC2245318.

Muthukumaran Subramaniyan. 2023. Paddy dataset. UCI Machine Learning Repository. Dataset.

Erdal Tasci, Ying Zhuge, Harpreet Kaur, Kevin Camphausen, and Andra Valentina Krauze. 2022. Hierarchical voting-based feature selection and ensemble learning model scheme for glioma grading with clinical and molecular characteristics. International Journal ofMolecular Sciences, 23(22):14155.

Together AI. 2024. Together.ai. https://www. together.ai/.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 others. 2023. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.

Zifeng Wang, Chufan Gao, Cao Xiao, and Jimeng Sun. 2024. MediTab: Scaling medical tabular data predictors via data consolidation, enrichment, and refinement. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 6062–6070. International Joint Conferences on Artificial Intelligence Organization.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Xumeng Wen, Han Zhang, Shun Zheng, Wei Xu, and Jiang Bian. 2024. From supervised to generative: A novel paradigm for tabular deep learning with large language models. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3323–3333. ACM.

Sichao Xiong, Yigit Ihlamur, Fuat Alican, and Aaron Ontoyin Yin. 2024. GPTree: Towards explainable decision-making via LLM-powered decision trees. arXiv preprint arXiv:2411.08257.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In Ad vances in Neural Information Processing Systems, volume 36, pages 11809–11822.

Hangting Ye, Jinmeng Li, He Zhao, Dandan Guo, and Yi Chang. 2025. LLM meeting decision trees on tabular data. In Advances in Neural Information Processing Systems, volume 38, pages 130884–130920.

I-Cheng Yeh. 2008. Blood transfusion service center. UCI Machine Learning Repository. Dataset.

Yifei Yuan, Jiatong Li, Weijia Zhang, Mohammad Aliannejadi, Evangelos Kanoulas, and Renjun Hu. 2025. Summarize-exemplify-reflect: Data-driven insight distillation empowers LLMs for few-shot tabular classification. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 12324– 12348.

Tianping Zhang, Shaowen Wang, Shuicheng Yan, Li Jian, and Qian Liu. 2023. Generative table pretraining empowers models for tabular prediction. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14836–14854. Association for Computational Linguistics.

## A Proof

Proposition 4.1 (Expected Impurity Reduction). $L e t r ^ { * } \in \mathcal { R }$ be the selected rule at node t, splitting dataset D into $D _ { l } , D _ { r }$ . Then, the expected impurity after splitting satisfies:

$$
\mathbb { E } [ \mathrm { G i n i } ( D _ { l } , D _ { r } ) ] \leq \mathrm { G i n i } ( D ) ,
$$

with equality if and only if the split yields no information gain.

Proof. Define the Gini impurity for dataset D with C classes as:

$$
\mathrm { G i n i } ( D ) = 1 - \sum _ { i = 1 } ^ { C } p _ { i } ^ { 2 } , \quad p _ { i } = \frac { | D ^ { ( i ) } | } { | D | } .
$$

The selected rule $r ^ { * }$ minimizes:

$$
\mathbb { E } [ \mathrm { G i n i } ( D _ { l } , D _ { r } ) ] = \frac { | D _ { l } | } { | D | } \mathrm { G i n i } ( D _ { l } ) + \frac { | D _ { r } | } { | D | } \mathrm { G i n i } ( D _ { r } ) .
$$

Since $r ^ { * }$ minimizes the above expression, it must hold that:

$$
\mathbb { E } [ \mathrm { G i n i } ( D _ { l } , D _ { r } ) ] \leq \mathrm { G i n i } ( D ) .
$$

Equality occurs precisely when no effective class separation occurs, resulting in no impurity reduction. □

Theorem 4.2 (Generalization Error Bound). Consider a decision tree T constructed using LLMT of maximum depth dfrom afinite rule set R with size K. Given n training samples,for any $\delta > 0$ , with probability at least $1 - \delta ,$ , the generalization error $R ( T )$ satisfies:

$$
R ( T ) ~ \le ~ { \hat { R } } ( T ) ~ + ~ \sqrt { \frac { 2 \Bigl ( \log { P ( K , 2 ^ { d } - 1 ) } + \log { \frac { 2 } { \delta } } \Bigr ) } { n } }\tag{4}
$$

$$
\leq { \hat { R } } ( T ) + { \sqrt { \frac { 2 \left( \left( 2 ^ { d } - 1 \right) \log K + \log { \frac { 2 } { \delta } } \right) } { n } } } ,\tag{5}
$$

$$
w h e r e ~ P ( K , 2 ^ { d } - 1 ) = K ! / ( K - ( 2 ^ { d } - 1 ) ) ! .
$$

Proof. Let $N _ { \mathrm { i n t } } = 2 ^ { d } - 1$ , which is the number of internal nodes of a tree with depth d. We first count the number of possible trees produced by LLMT. At each internal node, the algorithm selects a distinct rule from the global pool R of size $K$ without replacement. Thus, the number of possible ordered selections of $N _ { \mathrm { i n t } }$ distinct rules is

$$
P ( K , N _ { \mathrm { i n t } } ) = { \frac { K ! } { ( K - N _ { \mathrm { i n t } } ) ! } } .
$$

Each selection uniquely determines the split structure of the tree, since the structure is fixed (binary tree of maximum depth d) and rules are not reused globally.

Next, we consider leaf labels. In LLMT, leaf predictions are determined by the leaf-refinement step: given the path of rules leading to the leaf, the LLM outputs a confidence distribution, and the label is set using a fixed threshold τ. This procedure depends only on the rule path and $\tau _ { \ast }$ not on the training data. Therefore, once the split structure is fixed, the leaf labels are fixed as well, and there is no additional multiplicative factor for label assignments.

Let H be the set of all trees that can be produced by the algorithm. Then $| \mathcal { H } | \le P ( K , N _ { \mathrm { i n t } } )$ . For each tree $T \in { \mathcal { H } }$ , let $\hat { R } ( T )$ be its empirical error on n i.i.d. samples and $R ( T )$ its population error. By Hoeffding’s inequality,

$$
\Pr \bigl ( | R ( T ) - \hat { R } ( T ) | \geq \varepsilon \bigr ) \ \leq \ 2 e ^ { - 2 n \varepsilon ^ { 2 } } .
$$

Applying the union bound over all $T \in { \mathcal { H } }$ gives

$$
\operatorname* { P r } \Big ( \operatorname* { s u p } _ { T \in \mathcal { H } } | R ( T ) - \hat { R } ( T ) | \geq \varepsilon \Big ) \ \leq \ 2 | \mathcal { H } | \ e ^ { - 2 n \varepsilon ^ { 2 } } .
$$

Setting the right-hand side to $\delta$ and solving for ε yields

$$
\begin{array} { r l } { \varepsilon } & { = \sqrt { \frac { 2 \left( \log | \mathcal { H } | + \log ( 2 / \delta ) \right) } { n } } } \\ & { \leq \sqrt { \frac { 2 \left( \log P ( K , N _ { \mathrm { i n t } } ) + \log ( 2 / \delta ) \right) } { n } } . } \end{array}\tag{6}
$$

(7)

This proves the first inequality in (5). The second inequality follows from the fact that

$$
\log P ( K , N _ { \mathrm { i n t } } ) = \sum _ { t < N _ { \mathrm { i n t } } } \log ( K { - } t ) \leq N _ { \mathrm { i n t } } \log K .
$$

□

## B Algorithm

We summarize the three-stage LLMT framework in Algorithm 1 and its optional ensemble extension, LLMT Forest, in Algorithm 2. Algorithm 1 presents rule generation (lines 1–2), tree assembly (lines 3–12), and leaf refinement (lines 13– 20), while Algorithm 2 describes random featuresubspace sampling, bootstrap sampling, independent LLMT tree construction, and majority-vote aggregation.

Algorithm 1: The process of LLMT   
Input: Training samples D, depth d, LLM   
$f _ { \theta } ( \cdot )$   
Output: Decision Tree $T$   
1 Step 1 - Rule Generation:   
2 $\mathcal { R }  f _ { \theta } ( P _ { r u l e } )$ where $\mathcal { R } = [ ( r _ { k } , s _ { k } ) ] _ { k = 1 } ^ { K }$   
3 Step 2 – Tree Assembly:   
4 for $i = 1 , \ldots , d - 1$ do   
5 for $j = 2 ^ { i - 1 } , \dots , 2 ^ { i } - 1$ do   
6 $R _ { \mathrm { p a t h } } \gets :$ {rules used until node $j \}$   
7 $\mathcal { R }$ candidate $ \mathcal { R } \setminus R _ { \mathrm { p a t h } }$ (group by   
$s _ { k } ) ;$   
8 for $G \in { \mathcal { R } } _ { \mathrm { c a n d i d a t e } }$ do   
9 select $r ^ { * } \in G$ by Eq. (1) on $\mathcal { D } _ { j }$   
10 if Gini-gain $( r ^ { * } , \mathcal { D } _ { j } ) > 0$ then   
11 $T [ j ]  r ^ { * } ;$   
12 break;   
13 Step 3 – Leaf Refinement:   
14 foreach leaf node $T [ i ] \in T$ do   
15 $C _ { p } \gets f _ { \theta } ( T [ i ] ; P _ { r e f i n e } )$   
16 $y ^ { * } \gets \arg \operatorname* { m a x } { \mathbf { C _ { p } } }$   
17 $c ^ { * }  \operatorname* { m a x } { \bf C _ { p } }$   
18 if $\cdot c ^ { * } > \tau \wedge y ^ { * } \neq T [ i ]$ then   
19 $T [ i ]  y ^ { * }$   
20 return $T ;$

## C Theoretical Time Complexity Analysis

In practice, dominant runtime cost of our method comes from LLM calls; the tree-assembly steps is lightweight and takes less than 5% of the total wallclock time across all datasets. Here we provide a theoretical analysis of the time complexity.

Let n be the number of training examples, K the number of generated rules, $d _ { \mathrm { m a x } }$ the maximum tree depth, and $T _ { \mathrm { L L M } }$ the time of a single LLM call.

Rule Generation and Leaf Refinement: For rule generation, we need to generate rules for each internal node. In the worst case, a complete binary tree of depth $d _ { \mathrm { m a x } }$ has $2 ^ { d _ { \operatorname* { m a x } } } - 1$ internal nodes, requiring at most $2 ^ { d _ { \mathrm { m a x } } } - 1 \mathrm { L L M }$ calls. For leaf refinement, we need to evaluate each leaf node, which requires at most $2 ^ { d _ { \mathrm { m a x } } }$ LLM calls (one per leaf). Therefore, the total cost for LLM calls is at most $( 2 ^ { d _ { \mathrm { m a x } } } + 1 ) T _ { \mathrm { L L M } }$

Tree Assembly: For tree assembly, at each internal node, we evaluate at most K rules on its local subset of examples. Since each example appears in at most $d _ { \mathrm { m a x } }$ nodes along its path from root to leaf, the total number of rule evaluations is at most $K \cdot n \cdot d _ { \mathrm { m a x } }$ . Each rule evaluation involves checking the condition against an example, which takes constant time. Therefore, the total cost for tree assembly is $O ( K n d _ { \operatorname* { m a x } } )$

Algorithm 2: The process of LLMT Forest   
Input: Training samples D, feature set ${ \overline { { \mathcal { F } } } } ,$   
number of trees M, depth d, LLM   
$f _ { \theta } ( \cdot )$   
Output: LLMT Forest $\tau$   
1 $q \gets \left\lceil \sqrt { | \mathcal { F } | } \right\rceil$   
2 for $m = 1 , \ldots , M$ do   
3 $\mathcal { F } _ { m } \gets$ RandomSubset ${ \mathfrak { \Gamma } } ( { \mathcal { F } } , q )$   
4 $\mathscr { D } _ { m } \gets \mathrm { P r o j e c t } _ { \mathcal { F } _ { m } } \bigl ( \mathrm { B o o t s t r a p } ( \mathscr { D } ) \bigr )$   
5 $T _ { m } \gets \mathrm { L L M T } ( \mathscr { D } _ { m } , d , f _ { \theta } ( \cdot ) )$   
6 $\mathcal { T } ( x ) \gets \mathrm { M a j o r i t y V o t e } \big ( \{ T _ { m } ( x | _ { \mathcal { F } _ { m } } ) \} _ { m = 1 } ^ { M } \big )$   
7 return $\tau$

Total Time Complexity: Combining both components, the total time complexity is:

$$
( 2 ^ { d _ { \operatorname* { m a x } } } + 1 ) T _ { \mathrm { L L M } } + O ( K n d _ { \operatorname* { m a x } } ) .
$$

Since $T _ { \mathrm { L L M } }$ is typically much larger than the time for a single rule evaluation, and K is set to max $( 1 0 , 2 ^ { d _ { \mathrm { m a x } } } - 1 )$ (as described in Section 5), the LLM call cost dominates the overall runtime in practice.

## D Step-by-Step Example of LLMT

We provide a concrete example below to illustrate how LLMT works step-by-step. This example uses the Diabetes dataset with 4 training samples, maximum depth 3, and refinement threshold $\tau = 0 . 7$

Training samples (toy subset):

• Sample 1: [Glucose=78, BMI=31.2, Age=42,   
DPF=0.382, . . . ] → no

• Sample 2: [Glucose=64, BMI=29.2, Age=21,   
$\mathrm { D P F { = } 0 . 1 9 2 , \dots ] \to }$ no

• Sample 3: [. . . ] → yes

• Sample 4: [. . . ] → yes

Step 1 – Rule generation.

The LLM, given only task metadata, outputs meta-rules with confidence scores, e.g.:

• Glucose < 140 [confidence: 10]

• BMI < 30.0 [confidence: 9]

• Age < 30 [confidence: 8]

• DPF < 0.5 [confidence: 8]

• . . .

## Step 2 & 3 – Tree assembly + leaf refinement.

At the root, we evaluate candidate rules by Gini gain and select BMI < 30.0. This splits the data into a pure left node (1 sample, class: no) and a mixed right node (3 samples). The left node becomes a leaf; we send its path (BMI < 30.0) to the LLM, which returns calibrated probabilities (e.g., no: 0.700, yes: 0.300), so the leaf remains no since confidence ≥ τ. For the right node, we again select the rule with the highest Gini gain (e.g., DPF < 0.5) and recurse until either purity or depth constraints are met. For each resulting leaf path $( \mathrm { e . g . , B M I } \ \geq \ 3 \theta . \theta \ \mathsf { A N D } \mathsf { D P F } \ \geq \ \theta . 5 )$ , we perform the same refinement step via the LLM and only change the leaf label when confidence exceeds τ.

The final toy tree has the form of Figure 7:

![](images/08e60f43d1ddbe49fba6bed4211e321d97c2867fd0aa35fd5ace55c357cdac54.jpg)  
Figure 7: LLMT Example Tree (Step-by-Step)

## E Information of Datasets

The details of the evaluated datasets are presented below: 1) Nursery (Rajkovic, 1989): A multiclass classification dataset originally used to evaluate nursery school application outcomes; 2) Diabetes (Smith et al., 1988): A binary classification dataset from the Pima Indians Diabetes Database to predict the onset of diabetes; 3) Spambase (Hopkins et al., 1999): A binary classification dataset from the UCI repository to determine whether an email is spam; 4) Abalone (Nash et al., 1994): A binary dataset aiming to predict the age of abalones (measured by the number of rings); 5) Blood (Yeh, 2008): A binary classification dataset on blood donation behavior to predict whether a donor will give blood in the future based on historical donation patterns; 6) Iris (Fisher, 1936): A classical multiclass dataset consisting of flower measurements for classifying iris species; 7) Breast (Patrício et al., 2018): A binary classification dataset using resistin, glucose, age, and BMI to predict breast cancer presence; 8) Glioma (Tasci et al., 2022): A binary classification dataset for glioma grading based on clinical and molecular characteristics; 9) Communities (Redmond, 2002): A high-dimensional classification dataset with 103 features for predicting violent-crime risk tiers; 10) Ecom (Rubachev et al., 2025): A 119-feature coupon-redemption dataset derived from the Acquire Valued Shoppers data using the TabReD preprocessing pipeline; 11) Myocardial (Golovenkin et al., 2020): A 111-feature clinical dataset for identifying chronic heart failure after myocardial infarction. Among them, we select Diabetes, Spambase, and Nursery as the major datasets for evaluation across various settings. These datasets vary widely in sample size, feature dimensionality, and number of classes, providing a comprehensive testbed.

We list all the dataset information in Table 6. All datasets are publicly available. The UCI-hosted datasets used in this paper are released under the Creative Commons Attribution 4.0 license (CC BY 4.0), which permits reuse and redistribution with proper attribution.

<table><tr><td>Dataset</td><td>#Instances</td><td>#Features</td><td>#Classes</td></tr><tr><td>Nursery</td><td>12,960</td><td>8</td><td>3</td></tr><tr><td>Diabetes</td><td>768</td><td>8</td><td>2</td></tr><tr><td>Spambase</td><td>4,601</td><td>18</td><td>2</td></tr><tr><td>Abalone</td><td>4,177</td><td>8</td><td>2</td></tr><tr><td>Blood</td><td>748</td><td>4</td><td>2</td></tr><tr><td>Iris</td><td>150</td><td>4</td><td>3</td></tr><tr><td>Breast</td><td>116</td><td>9</td><td>2</td></tr><tr><td>Glioma</td><td>839</td><td>23</td><td>2</td></tr><tr><td>Communities</td><td>1,994</td><td>103</td><td>3</td></tr><tr><td>Ecom</td><td>160,057</td><td>119</td><td>2</td></tr><tr><td>Myocardial</td><td>686</td><td>111</td><td>2</td></tr></table>

Table 6: Statistics of datasets used in our experiments.

## F Experimental Settings

This section provides adequate experimental parameters for reproducibility. All hyperparameters, model configurations, and experimental settings are detailed below.

## F.1 Experimental Details for Observation 1 E

In Observation 1, we compare LLM performance under two scenarios: original and masked. In the original scenario, the model has access to the full dataset description, including feature names and label names. For example, a feature appears as Age or Glucose with its semantic meaning. In the masked scenario, the model does not have access to feature names or label names. Instead, features are represented only as column numbers (e.g., Feature 2 or Column 3), and labels are similarly anonymized (e.g., Class 0 and Class 1 instead of yes and no). This masking removes the semantic context that LLMs rely on for leveraging their background knowledge about the task domain.

To reduce dependence on old, widely used benchmarks, we extend the same original-versusmasked protocol to six newer datasets. Breast (2018) and Glioma (2022) already appear in our main evaluation. Paddy was released by UCI in July 2025 (Subramaniyan, 2023), and Students was released in July 2025 (Hossen and Uddin, 2025); both postdate Llama-4 Maverick, the newest model in this experiment. Loan and Personality are synthetic datasets from the 2025 Kaggle Playground Series (Kaggle, 2025b,a), so their generated records could not have appeared in model pretraining. We evaluate the same four LLMs used in Figure 1. Table 7 reports accuracy and the absolute percentage-point reduction caused by masking.

Masking lowers accuracy in 22 of 24 comparisons, leaves one unchanged, and improves one, with an average reduction of 20.14 percentage points. The pattern on post-cutoff and synthetic datasets reduces the likelihood that the effect is explained only by memorization of old benchmark records. Nevertheless, masking changes the semantic information available to the model and therefore cannot by itself prove the absence of pretraining memorization.

## F.2 Experimental Details for Observation 2

We compare rule sets and path sets with the same total number of atomic feature conditions. Under a condition budget n, a rule set contains n independent single-condition rules, whereas a path set contains one or more conjunctive paths with n conditions in total. The two prompts use identical features, labels, and comparison operators. We use Qwen2.5-72B with temperature 0.8 and evaluate condition budgets $n \in \{ 3 , 4 \}$

Let $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be the evaluation set and S a generated rule or path set. A member $s \in S$ is triggered when $x _ { i }$ satisfies its condition or all conditions along its path, and predicts label $\hat { y } _ { s }$ . We define

$$
\begin{array} { r l } & { T _ { i } ( S ) = \{ s \in S : s \mathrm { i s ~ t r i g g e r e d } \mathrm { b y } x _ { i } \} , } \\ & { C _ { i } ( S ) = \{ s \in T _ { i } ( S ) : \hat { y } _ { s } = y _ { i } \} . } \end{array}
$$

Let $t _ { i } = | T _ { i } ( S ) |$ and $c _ { i } = | C _ { i } ( S ) |$ . Set Utility and Set Accuracy are computed as

$$
\mathrm { U t i l i t y } ( S ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ c _ { i } > 0 ] ,
$$

and

$$
\begin{array} { c l } { { \displaystyle \mathrm { A c c u r a c y } ( S ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } a _ { i } ( S ) , } } \\ { { \displaystyle a _ { i } ( S ) = \Biggl \{ c _ { i } / t _ { i } , } }  & { { t _ { i } > 0 , } } \\ { { 0 , } } & { { t _ { i } = 0 . } } \end{array}
$$

Set Utility measures whether an instance is covered by at least one correct member, whereas Set Accuracy measures the proportion of correct predictions among triggered members and assigns zero when none is triggered. We average both metrics over 100 generations for each dataset, set type, and condition budget.

Table 8 shows that rule sets achieve higher mean Set Utility and Set Accuracy than path sets under both budgets, indicating that the advantage persists when the total number of atomic conditions is controlled.

## F.3 Dataset-Specific Parameters

Table 9 lists dataset-specific parameters. Max Depth (3 or 4, depending on dataset) is shared by all methods. Test Size is 30 for Breast dataset and 100 for others. The n\_estimators column applies to XGBoost, Random Forest and LLMT Forest.

## F.4 LLM Configuration Settings

Table 10 details the LLM API configuration used for all LLM-based methods. The same Qwen2.5- 72B backbone is used for all baselines. Request intervals and timeouts are operational parameters adjusted to dataset workload and provider status rather than fixed experimental hyperparameters.

<table><tr><td>Dataset Model</td><td></td><td>Original Masked</td><td></td><td>Δ</td><td>Dataset</td><td>Model</td><td>Original Masked</td><td></td><td>Δ</td></tr><tr><td>Breast</td><td>DeepSeek-V3</td><td>56.0</td><td>50.0</td><td>6.0</td><td>Paddy</td><td>DeepSeek-V3</td><td>34.5</td><td>31.5</td><td>3.0</td></tr><tr><td>Breast</td><td>Llama-4</td><td>56.9</td><td>49.1</td><td>7.8</td><td>Paddy</td><td>Llama-4</td><td>40.0</td><td>37.5</td><td>2.5</td></tr><tr><td>Breast</td><td>Gemma-2</td><td>55.2</td><td>52.6</td><td>2.6</td><td>Paddy</td><td>Gemma-2</td><td>37.0</td><td>32.5</td><td>4.5</td></tr><tr><td>Breast</td><td>Qwen2.5</td><td>54.3</td><td>54.3</td><td>0.0</td><td>Paddy</td><td>Qwen2.5</td><td>32.5</td><td>30.5</td><td>2.0</td></tr><tr><td>Glioma</td><td>DeepSeek-V3</td><td>89.0</td><td>34.0</td><td>55.0</td><td>Personality</td><td>DeepSeek-V3</td><td>93.0</td><td></td><td>59.0 34.0</td></tr><tr><td>Glioma</td><td>Llama-4</td><td>89.0</td><td>35.0</td><td>54.0</td><td>Personality</td><td>Llama-4</td><td>93.0</td><td></td><td>64.029.0</td></tr><tr><td>Glioma</td><td>Gemma-2</td><td>53.0</td><td>57.0</td><td>-4.0</td><td>Personality</td><td>Gemma-2</td><td>93.0</td><td></td><td>61.531.5</td></tr><tr><td>Glioma</td><td>Qwen2.5</td><td>85.5</td><td>33.0</td><td>52.5</td><td>Personality</td><td>Qwen2.5</td><td>93.5</td><td></td><td>81.5 12.0</td></tr><tr><td>Loan</td><td>DeepSeek-V3</td><td>68.0</td><td>52.5</td><td>15.5</td><td>Students</td><td>DeepSeek-V3</td><td>77.0</td><td></td><td>56.0 21.0</td></tr><tr><td>Loan</td><td>Llama-4</td><td>68.0</td><td>49.5</td><td>18.5</td><td>Students</td><td>Llama-4</td><td>72.5</td><td></td><td>39.0 33.5</td></tr><tr><td>Loan</td><td>Gemma-2</td><td>54.5</td><td>44.0</td><td>10.5</td><td>Students</td><td>Gemma-2</td><td>66.5</td><td></td><td>43.5 23.0</td></tr><tr><td>Loan</td><td>Qwen2.5</td><td>70.5</td><td>34.0</td><td>36.5</td><td>Students</td><td>Qwen2.5</td><td>78.5</td><td></td><td>46.032.5</td></tr></table>

Table 7: Original-versus-masked zero-shot accuracy (%) on six newer datasets. ∆ is Original minus Masked in percentage points. Llama-4 denotes Llama-4 Maverick, Gemma-2 denotes Gemma-2-27B-IT, and Qwen2.5 denotes Qwen2.5-72B-Instruct.
<table><tr><td></td><td colspan="4">Condition budget n = 3</td><td colspan="4">Condition budget n = 4</td></tr><tr><td>Dataset</td><td>Rule U.</td><td>Path U.</td><td>Rule A.</td><td>Path A.</td><td>Rule U.</td><td>Path U.</td><td>Rule A.</td><td>Path A.</td></tr><tr><td>Nursery</td><td>31.87</td><td>28.32</td><td>24.75</td><td>25.86</td><td>41.34</td><td>41.35</td><td>31.20</td><td>33.32</td></tr><tr><td>Diabetes</td><td>35.28</td><td>32.21</td><td>32.60</td><td>29.48</td><td>64.52</td><td>65.14</td><td>50.62</td><td>50.29</td></tr><tr><td>Spambase</td><td>54.55</td><td>41.06</td><td>45.44</td><td>36.64</td><td>60.42</td><td>53.63</td><td>52.04</td><td>47.74</td></tr><tr><td>Abalone</td><td>33.41</td><td>29.68</td><td>33.05</td><td>29.45</td><td>53.11</td><td>32.23</td><td>51.42</td><td>32.05</td></tr><tr><td>Blood</td><td>40.01</td><td>35.45</td><td>33.92</td><td>32.03</td><td>64.24</td><td>31.66</td><td>43.48</td><td>24.49</td></tr><tr><td>Iris</td><td>74.73</td><td>60.81</td><td>65.18</td><td>54.84</td><td>73.51</td><td>69.10</td><td>63.72</td><td>63.45</td></tr><tr><td>Breast</td><td>38.99</td><td>36.28</td><td>36.77</td><td>35.66</td><td>51.82</td><td>18.39</td><td>44.57</td><td>18.12</td></tr><tr><td>Glioma</td><td>65.94</td><td>65.88</td><td>60.76</td><td>62.94</td><td>75.96</td><td>50.17</td><td>68.29</td><td>46.43</td></tr><tr><td>Mean</td><td>46.85</td><td>41.21</td><td>41.56</td><td>38.36</td><td>60.62</td><td>45.21</td><td>50.67</td><td>39.49</td></tr></table>

Table 8: Rule-set and path-set results under matched condition budgets (%). U. and A. denote Set Utility and Set Accuracy. Each entry is averaged over 100 generations.  
build 10 trees.

## F.5 Method-Specific Settings

Table 11 provides hyperparameters for all methods. The max\_depth settings match those in Table 9 for each dataset. The n\_estimators values are datasetspecific and listed in Table 9. All LLM-based methods (IO-Tree, CoT-Tree, ToT-Tree, LLMT) use the LLM configuration specified in Table 10. It is worth noting for LLMT and LLMT Forest, confidence groups are formed by grouping meta-rules with the same confidence value together.

## F.6 Experimental Running Parameters

Table 12 lists general experimental running parameters. Number of Trials per Setting means that for each training sample size, we randomly sample 10 different training sets and run experiments, i.e.,

## F.7 Implementation of two Zero-shot Baselines

To further evaluate the effectiveness and efficiency of our proposed LLMT framework, we compare it with two representative zero-shot LLM-based decision tree baselines: (1) Direct-ZSDT (Knauer et al. (2025)); (2) Step-ZSDT (Carrasco et al. (2025)). For baseline Direct-ZSDT, we reuse the official code and ensure identical prompt and parameter settings to guarantee fair comparison. For baseline Step-ZSDT, whose code is not released, we follow the prompt templates and methodology described by the authors and set the probability threshold to 0.9 as specified in the original paper.

<table><tr><td>Dataset</td><td>Max Depth</td><td>n_estimators</td></tr><tr><td>Diabetes</td><td>3</td><td>3</td></tr><tr><td>Nursery</td><td>3</td><td>3</td></tr><tr><td>Spambase</td><td>4</td><td>6</td></tr><tr><td>Abalone</td><td>3</td><td>3</td></tr><tr><td>Blood</td><td>3</td><td>3</td></tr><tr><td>Iris</td><td>3</td><td>3</td></tr><tr><td>Breast</td><td>3</td><td>3</td></tr><tr><td>Glioma</td><td>4</td><td>6</td></tr><tr><td>Communities</td><td>4</td><td>30</td></tr><tr><td>Ecom</td><td>3</td><td>5</td></tr><tr><td>Myocardial</td><td>3</td><td>16</td></tr></table>

Table 9: Dataset-specific parameters for all experiments.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Model</td><td>Qwen2.5-72B-Instruct</td></tr><tr><td>API Provider</td><td>TogetherAI</td></tr><tr><td>Temperature</td><td>0.0</td></tr><tr><td>Max Tokens</td><td>2048</td></tr><tr><td>Parallel Batch Size</td><td>6</td></tr><tr><td>Presence Penalty</td><td>0.0</td></tr><tr><td>Frequency Penalty</td><td>0.0</td></tr><tr><td>Random Seed</td><td>42</td></tr></table>

Table 10: LLM API configuration settings used for all LLM-based methods.

## G Additional Experimental Results

## G.1 Accuracy

The mean accuracy and standard deviation across different #shots are presented in Figure 8.

## G.2 Comparisons with Additional Baselines

Tables 13, 14, and 15 present expanded comparisons with tree-ensemble and pretrained black-box baselines.

## G.3 Results on High-Dimensional Datasets

Non-ensemble and ensemble methods are reported in Tables 16 and 17, respectively. Step-ZSDT is omitted because its per-feature, per-node LLM queries are prohibitively expensive in high dimensions. Overall, LLMT performs best among nonensemble methods, achieving the highest average accuracy. LLMT Forest achieves the highest average accuracy among ensemble methods.

## G.4 Interpretability Analyses

## G.4.1 Rule Diversity Analysis

We ensure rule diversity through careful prompt design. Our meta-rule generation prompt (see Figures 17 and 18 for prompt details) encourages diversity through the following constraints:

## • No Redundancy

## • Maximize Purity

To quantify feature coverage, we generate 10 meta-rule sets for each of the eight original datasets and count the distinct features used. We use the same rule-pool size K as in the main experiments (Table 11). Table 18 reports both the number and proportion of covered features.

Coverage is 100% on six datasets, 83.3% on Spambase, and 64.8% on the 23-feature Glioma dataset. Coverage is not itself an interpretability score—a smaller, focused rule set may be easier to inspect—but these values show that rule generation is not restricted to a single dominant feature.

## G.4.2 Visualized Tree Examples

Figures 9–16 provide representative Diabetes trees from every compared tree generator. The LLMT example uses glucose as its principal decision feature and exposes every threshold and leaf prediction directly. These examples complement the aggregate impurity and tree-size results in Section 5 by allowing readers to inspect the learned decision logic, while we avoid treating a single medically plausible example as quantitative evidence of clinical validity.

## G.5 Sensitivity to Maximum Tree Depth

During parameter tuning, we examine the effect of varying maximum depth in the tree-building process; results are shown in Table 19. Performance proves sensitive to this hyperparameter. On lowdimensional datasets (e.g., Diabetes), even shallow trees (depth = 2-3) yield stable accuracy. By contrast, on higher-dimensional datasets with numerous categorical features (e.g., Nursery), deeper trees (depth = 4) consistently achieve better splits and higher accuracy. Therefore, careful tuning of maximum depth is essential - deeper trees better capture complex feature interactions when feature diversity is high.

<table><tr><td>Method</td><td>Parameter</td><td>Value</td></tr><tr><td>Tree methods Tree ensembles</td><td>Maximum depth Ensemble size</td><td>Table 9 Table 9</td></tr><tr><td>Step-ZSDT</td><td>Leaf-stopping threshold τ</td><td>0.9</td></tr><tr><td>XGBoost</td><td>learning_rate</td><td>0.1</td></tr><tr><td>LogReg</td><td>max_iter</td><td>1000</td></tr><tr><td>Linear SVM</td><td>Estimator max_iter</td><td>LinearSVC 10000</td></tr><tr><td>LightGBM</td><td>learning_rate</td><td>0.1</td></tr><tr><td>CatBoost</td><td>depth</td><td>Table 9</td></tr><tr><td></td><td>learning_rate</td><td>0.1</td></tr><tr><td>ToT-Tree</td><td>Candidate rules per</td><td>5</td></tr><tr><td></td><td>node Voting rounds</td><td>5</td></tr></table>

<table><tr><td>Method</td><td>Parameter</td><td>Value</td></tr><tr><td>FeatLLM</td><td>Rule-generation trials</td><td>5</td></tr><tr><td>DeLTa</td><td>Residual learner</td><td>Decision-tree regressor</td></tr><tr><td>GPTree</td><td>Fusion coefficient η</td><td>0.1 CODE (our</td></tr><tr><td></td><td>Node type</td><td>reproduction)</td></tr><tr><td>TabPFN</td><td>Pretraining-limit override</td><td>Enabled</td></tr><tr><td>TabLLM</td><td>Backbone Adaptation</td><td>bigscience/T0pp IA3 via T-Few</td></tr><tr><td>LLMT</td><td>Leaf-refinement threshold τ</td><td>0.7</td></tr><tr><td></td><td>Numeric split candidates per feature</td><td>10</td></tr><tr><td></td><td>Requested meta-rules K</td><td>max(10, 2dmax − 1)</td></tr><tr><td>LLMT Forest</td><td>Base estimator</td><td>LLMT (same settings)</td></tr><tr><td></td><td>Feature partition</td><td>Randomized, balanced, and</td></tr><tr><td></td><td>Aggregation</td><td>disjoint Majority vote</td></tr></table>

Direct-ZSDT Step-ZSDT LogReg SVM CART XGBoost LightGBM IO-Tree CoT-Tree ToT-Tree FeatLLM DeLTa GPTree LLMT  
![](images/128348b6f38c486fe9a29e4acbee8051fb10cfb77fc5c57d43c40649221727bc.jpg)  
Table 11: Method-specific settings.

![](images/ff1d477adc431a96b6b996c18bcf296014299727d1edd65fb8f90925b0f41550.jpg)

![](images/9ca912e453bc07f181bba2a7596146668a9f3276378e87bb2bc2d34156a7d24d.jpg)  
Figure 8: The accuracy and standard deviation of different methods on three datasets.

![](images/61d43c97eafbfd479fe838d9e58a97c1ffc71f74c660b71772e3c4ce8ec1a684.jpg)  
Figure 9: An Example of Direct-ZSDT

![](images/47121268d0295d9c844a495b554f11ac45120b78a8451827971a29f5c7f943d5.jpg)  
Figure 10: An Example of Step-ZSDT

![](images/c9973dd5d0261b94b85b7c14ff9b5d0eceb76eb456a664956bd8615808e8f305.jpg)  
Figure 11: An Example of CART Tree

![](images/aaa52217e15e9faa7d0bd063805504bc189e5c7cd3f7ce6ba50f3c96ccaa81e1.jpg)  
Figure 12: An Example of IO-Tree

![](images/fc514b978c694dc8f7a0579208a12a62f2932a8997a42cf6d7d7479d42da1caf.jpg)  
Figure 13: An Example of CoT-Tree

![](images/56ac49ff7120fc9dcf31a71b429ebc4b463553c7b73d47f21fe3f3a0447bc5d2.jpg)  
Figure 14: An Example of ToT-Tree

![](images/709ec0de9b7c449896f4c32b211242ffb960c80db0ebb81e0f8055ef3eac8b5b.jpg)

Figure 15: An Example of GPTree  
![](images/0717d29221dbe5eaf3db5e9632289043a05bbe4f28a74664a0b23f0c5070e6fb.jpg)  
Figure 16: An Example of LLMT

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Number of Trials per Setting</td><td>10</td></tr><tr><td>Random Seed</td><td>0</td></tr><tr><td>Train Batch Size</td><td>8</td></tr><tr><td>Test Batch Size</td><td>8</td></tr></table>

Table 12: General experimental running parameters.

## H Usage of LLMs

In this work, we use ChatGPT to polish the writing of our paper.

## I Prompting Templates

To facilitate application across diverse tasks, we design two prompt templates: one for generating meta-rule candidates (Figure 17 and Figure 18) and another for producing supervised probability outputs for leaf labels (Figure 19). We also include Figure 20, 21, 22, 23, 24, and 25, which respectively show the reference I/O-Tree, CoT-Tree, and ToT-Tree generation templates used as baselines in the main text. Text in blue font denotes the title of the prompt section, red font signifies variables that vary across datasets, orange font represents system commands, and text in black font is the fixed input text used for prompting.

<table><tr><td>Dataset</td><td>Random Forest</td><td>CatBoost</td><td>LLMT</td></tr><tr><td>Nursery</td><td>0.397</td><td>0.473</td><td>0.762</td></tr><tr><td>Diabetes</td><td>0.584</td><td>0.590</td><td>0.764</td></tr><tr><td>Spambase</td><td>0.725</td><td>0.743</td><td>0.811</td></tr><tr><td>Abalone</td><td>0.688</td><td>0.700</td><td>0.694</td></tr><tr><td>Blood</td><td>0.549</td><td>0.558</td><td>0.675</td></tr><tr><td>Iris</td><td>0.769</td><td>0.829</td><td>0.940</td></tr><tr><td>Breast</td><td>0.530</td><td>0.517</td><td>0.617</td></tr><tr><td>Glioma</td><td>0.733</td><td>0.759</td><td>0.760</td></tr><tr><td>Average</td><td>0.622</td><td>0.646</td><td>0.753</td></tr></table>

Table 13: Accuracy of Random Forest and CatBoost with two training examples per class, compared with LLMT.

<table><tr><td>Dataset</td><td>#Shots</td><td>Random Forest</td><td>CatBoost</td><td>LLMT</td></tr><tr><td rowspan="5">Nursery</td><td>3</td><td>0.349</td><td>0.464</td><td>0.762</td></tr><tr><td>6</td><td>0.397</td><td>0.473</td><td>0.762</td></tr><tr><td>12</td><td>0.707</td><td>0.411</td><td>0.784</td></tr><tr><td>24</td><td>0.754</td><td>0.721</td><td>0.818</td></tr><tr><td>48</td><td>0.798</td><td>0.740</td><td>0.840</td></tr><tr><td rowspan="5">Diabetes</td><td>2</td><td>0.486</td><td>0.523</td><td>0.662</td></tr><tr><td>4</td><td>0.584</td><td>0.590</td><td>0.764</td></tr><tr><td>8</td><td>0.659</td><td>0.650</td><td>0.770</td></tr><tr><td>16</td><td>0.618</td><td>0.671</td><td>0.770</td></tr><tr><td>32</td><td>0.683</td><td>0.700</td><td>0.770</td></tr><tr><td rowspan="5">Spambase</td><td>2</td><td>0.677</td><td>0.620</td><td>0.782</td></tr><tr><td>4</td><td>0.725</td><td>0.743</td><td>0.811</td></tr><tr><td>8</td><td>0.740</td><td>0.785</td><td>0.836</td></tr><tr><td>16</td><td>0.783</td><td>0.788</td><td>0.839</td></tr><tr><td>32</td><td>0.821</td><td>0.785</td><td>0.843</td></tr></table>

Table 14: Accuracy of Random Forest and CatBoost across shot counts, compared with LLMT.

<table><tr><td>Dataset</td><td>TabPFN</td><td>TabLLM</td><td>LLMT</td><td>LLMT Forest</td></tr><tr><td>Nursery</td><td>0.750</td><td>0.759</td><td>0.762</td><td>一</td></tr><tr><td>Diabetes</td><td>0.710</td><td>0.510</td><td>0.764</td><td>一</td></tr><tr><td>Spambase</td><td>0.802</td><td>0.673</td><td>0.811</td><td>一</td></tr><tr><td>Abalone</td><td>0.754</td><td>0.569</td><td>0.694</td><td>一</td></tr><tr><td>Blood</td><td>0.658</td><td>0.496</td><td>0.675</td><td>一</td></tr><tr><td>Iris</td><td>0.918</td><td>0.327</td><td>0.940</td><td>一</td></tr><tr><td>Breast</td><td>0.510</td><td>0.587</td><td>0.617</td><td>一</td></tr><tr><td>Glioma</td><td>0.878</td><td>0.521</td><td>0.760</td><td>一</td></tr><tr><td>Communities</td><td>0.443</td><td>N/A</td><td>0.438</td><td>0.490</td></tr><tr><td>Ecom</td><td>0.518</td><td>N/A</td><td>0.582</td><td>0.757</td></tr><tr><td>Myocardial</td><td>0.541</td><td>N/A</td><td>0.618</td><td>0.773</td></tr></table>

Table 15: Comparison with pretrained black-box few-shot baselines. TabLLM runs out of memory on Communities and Ecom and is prohibitively slow on Myocardial under the T0pp protocol; these cases are marked N/A.

<table><tr><td>Dataset</td><td>LogReg</td><td>SVM</td><td>CART</td><td>Direct-ZSDT</td><td>IO-Tree</td><td>CoT-Tree</td><td>ToT-Tree</td><td>FeatLLM</td><td>GPTree</td><td>LLMT (Ours)</td></tr><tr><td>Communities 0.424</td><td></td><td>0.439</td><td>0.404</td><td>0.418</td><td>0.336</td><td>0.411</td><td>0.402</td><td>0.432</td><td>0.407</td><td>0.438</td></tr><tr><td>Ecom</td><td>0.509</td><td>0.508</td><td>0.554</td><td>0.561</td><td>0.511</td><td>0.524</td><td>0.534</td><td>0.579</td><td>0.490</td><td>0.582</td></tr><tr><td>Myocardial</td><td>0.526</td><td>0.522</td><td>0.540</td><td>0.510</td><td>0.535</td><td>0.503</td><td>0.498</td><td>0.596</td><td>0.556</td><td>0.618</td></tr><tr><td>Average</td><td>0.486</td><td>0.490</td><td>0.499</td><td>0.496</td><td>0.461</td><td>0.479</td><td>0.478</td><td>0.536</td><td>0.484</td><td>0.546</td></tr></table>

Table 16: Accuracy comparison of non-ensemble methods on three high-dimensional datasets.

<table><tr><td>Dataset</td><td>XGBoost</td><td>RandomForest</td><td>LightGBM</td><td>CatBoost</td><td>DeLTa</td><td>LLMT Forest (Ours)</td></tr><tr><td>Communities</td><td>0.428</td><td>0.440</td><td>0.434</td><td>0.437</td><td>0.473</td><td>0.490</td></tr><tr><td>Ecom</td><td>0.695</td><td>0.504</td><td>0.412</td><td>0.503</td><td>0.619</td><td>0.757</td></tr><tr><td>Myocardial</td><td>0.702</td><td>0.545</td><td>0.581</td><td>0.535</td><td>0.565</td><td>0.773</td></tr><tr><td>Average</td><td>0.608</td><td>0.496</td><td>0.476</td><td>0.492</td><td>0.552</td><td>0.673</td></tr></table>

Table 17: Accuracy comparison of ensemble methods on three high-dimensional datasets.

<table><tr><td>Dataset</td><td>Diabetes</td><td>Iris</td><td>Spambase</td><td>Nursery</td><td>Abalone</td><td>Blood</td><td>Breast</td><td>Glioma</td></tr><tr><td>#Rules (K)</td><td>10</td><td>10</td><td>15</td><td>10</td><td>10</td><td>10</td><td>10</td><td>15</td></tr><tr><td>#Features</td><td>8</td><td>4</td><td>18</td><td>8</td><td>8</td><td>4</td><td>9</td><td>23</td></tr><tr><td>Avg. Features Used</td><td>8.0</td><td>4.0</td><td>15.0</td><td>8.0</td><td>8.0</td><td>4.0</td><td>9.0</td><td>14.9</td></tr><tr><td>Feature Coverage</td><td>100%</td><td>100%</td><td>83.3%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>64.8%</td></tr></table>

Table 18: Rule diversity analysis: Feature coverage (Avg. Features Used / #Features) measures the proportion of features utilized in the generated rules, with higher values indicating better diversity.

<table><tr><td>Datasets</td><td>#Shots</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td rowspan="5">Nursery</td><td>3</td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 7 6 2 { \scriptstyle \pm 0 . 1 0 0 }$ </td><td> $0 . 7 6 2 { \scriptstyle \pm 0 . 1 0 0 }$ </td><td> $0 . 7 6 2 { \scriptstyle \pm 0 . 1 0 0 }$ </td></tr><tr><td>6</td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 7 6 2 { \scriptstyle \pm 0 . 1 0 0 }$ </td><td> $0 . 7 0 7 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 7 3 1 { \scriptstyle \pm 0 . 0 7 7 }$ </td></tr><tr><td>12</td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 7 8 4 { \scriptstyle \pm 0 . 0 9 1 }$ </td><td> $0 . 7 4 2 { \scriptstyle \pm 0 . 0 7 4 }$ </td><td> $0 . 7 3 7 { \scriptstyle \pm 0 . 0 5 4 }$ </td></tr><tr><td>24</td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 8 1 8 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $0 . 7 1 3 { \pm } 0 . 0 8 4$ </td><td> $0 . 7 3 5 { \scriptstyle \pm 0 . 0 4 5 }$ </td></tr><tr><td>48</td><td> $0 . 7 4 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.840±0.000</td><td>0.788±0.042</td><td> $0 . 7 4 2 { \scriptstyle \pm 0 . 0 3 9 }$ </td></tr><tr><td rowspan="5">Diabetes</td><td>2</td><td> $0 . 6 6 2 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td> $0 . 6 6 2 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td> $0 . 6 6 2 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td>0.669±0.083</td></tr><tr><td>4</td><td> $0 . 7 1 0 { \scriptstyle \pm 0 . 0 9 2 }$ </td><td>0.764±0.025</td><td>0.736±0.064</td><td>0.736±0.064</td></tr><tr><td>8</td><td> $0 . 7 7 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.770±0.000</td><td> $0 . 7 6 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td>0.758±0.020</td></tr><tr><td>16</td><td> $0 . 7 7 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.770±0.000</td><td> $0 . 7 6 3 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td>0.756±0.024</td></tr><tr><td>32</td><td> $0 . 7 7 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.770±0.000</td><td> $0 . 7 6 6 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 7 6 2 { \pm } 0 . 0 1 5$ </td></tr><tr><td rowspan="5">Spambase</td><td>2</td><td> $0 . 7 8 2 { \pm } 0 . 0 1 0$ </td><td> $0 . 7 8 2 { \pm } 0 . 0 1 0$ </td><td> $0 . 7 8 2 { \pm } 0 . 0 1 0$ </td><td> $0 . 7 8 2 { \pm } 0 . 0 1 0$ </td></tr><tr><td>4</td><td> $0 . 7 8 8 { \pm } 0 . 0 0 6$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 7 8 4 { \pm } 0 . 0 2 8$ </td></tr><tr><td>8</td><td> $0 . 7 8 8 { \pm } 0 . 0 0 6$ </td><td> $0 . 8 4 3 { \pm } 0 . 0 2 8$ </td><td> $0 . 8 3 6 { \pm } 0 . 0 1 9$ </td><td> $0 . 8 1 0 { \scriptstyle \pm 0 . 0 2 3 }$ </td></tr><tr><td>16</td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 8 2 5 { \scriptstyle \pm 0 . 0 3 5 }$ </td><td> $0 . 8 3 9 { \pm } 0 . 0 2 5$ </td><td> $0 . 8 1 0 { \pm } 0 . 0 1 8$ </td></tr><tr><td>32</td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 8 1 1 { \scriptstyle \pm 0 . 0 3 2 }$ </td><td> $0 . 8 4 3 { \pm } 0 . 0 1 8$ </td><td> $0 . 7 9 8 { \pm } 0 . 0 1 6$ </td></tr></table>

Table 19: Experimental results across different maximum tree depths.

![](images/d0f0692c9c1bdfe048bcbfd0de02706c03e3e13b6e7b6314e52139ea5abeba12.jpg)  
Figure 17: Meta Rule Generation Prompting Template.

instruction Prompt   
4. Rule Quality: Choose thresholds that create meaningful splits and   
align with typical value patterns. Avoid overfitting to noise;   
confidence <5 only if necessary.   
5. No Redundancy: Avoid trivialiy similar rules. Use < for numeric, =   
for categorical.   
6. Maximize Purity: Prefer rules that create purer (more homogeneous)   
subgroups.   
7. Score Consistency: Rules of similar quality should have similar   
confidence (difference ≤ 2).   
8. Dominant Feature Priority: A strong feature can have MULTIPLE   
high-confidence rules - even HIGHER than ALL rules from weaker   
features.   
9. First Rule Matters: Think carefully about the first rule - its   
feature and threshold should reflect your strongest, most confident   
split. It is often treated as the default decision.   
## Output Format (Strict) :   
Provide the list of rules, one per line, exactly in the specified   
format, sorted by confidence descending. Do NOT include any other   
text, explanations, or headers.   
Example:   
study hours per week < 3.25 [ confidence: 10 ]   
attendance rate < 74.5 [ confidence: 9 ]   
assignment completion rate < 62.75 [ confidence: 8 ]   
study hours per week < 6.0 [ confidence: 7 ]   
participation = low [ confidence: 7 ]   
attendance rate < 85.000 [ confidence: 6 ]   
previous exam grade < 59.5 [ confidence: 6 ]   
## Generated Meta-Rules:  
Figure 18: Meta Rule Generation Prompting Template (continued).

![](images/850335df2a839e15436feba5e34ab09af31676da08d0b1610583b5d9234e7825.jpg)  
Figure 19: Leaf Refinement Prompting Template.

IO Tree Generation Prompting Template   
### Features:   
for feature in meta.features   
\*loop.index\*. \*feature.name\*: \*feature.desc or 'No description   
available'\* (Type: feature.type)   
if feature.is categorical and feature.categories   
Possible values:   
for cat value, cat desc in feature.categories.items()   
- cat \_value: cat\_desc   
endfor   
endif   
endfor   
### Target Variable:   
meta.label meaning or 'The output'   
Possible values:   
for label in meta.labels   
- label.name   
if label.desc (label.desc)endif   
endfor   
## Label Definitions   
for label in meta.labels   
- label.name: label.desc or 'No description   
endfor   
## Training Samples   
Here are some examples from the training dataset:   
For each line:   
[','.join(feature\_names) ] RESULT   
for i, idx in enumerate(sample indices)   
(i+1) [ ','.join(feature values) ] label name  
Figure 20: IO Tree Generation Prompting Template.

![](images/660f9bbd235a59c1b50584dc3a27506fbab1fcfc8e4c8e03f20267df9095c0a2.jpg)  
Figure 21: IO Tree Generation Prompting Template (continued).

instruction Prompt   
### Features:   
% for feature in meta.features %   
\*[ loop.index ]\*. \*[ feature.name ]\*: \*[ feature.desc or 'No   
description available' ]\* (Type: \*[ feature.type ]\*)   
% if feature.is categorical and feature.categories %   
Possible values:   
% for cat\_value, cat\_desc in feature.categories.items() %   
\*[ cat value ]\*: \*[ cat desc ]\*   
% endfor %   
% endif %   
% endfor %   
### Target Variable: \* meta.label meaning or 'The output' 1\*   
Possible values:   
% for label in meta.labels %   
- \* label.name 1\*   
% if label.desc %   
(\*[ label.desc ]\*)   
% endif %   
endfor %   
## Label Definitions   
% for label in meta.labels %   
\* label.name 1\*: \* label.desc or 'No description' 1\*   
% endfor %   
## Training Samples   
Here are some examples from the training dataset:   
For each line:   
<\*', '.join(feature names)\*> <RESULT>   
% for i, idx in enumerate(sample indices) %   
(\*[ i+1 ]\*) \*[ ', '.join(feature values) ]\* \*[ label name ]\*   
% endfor %   
## Decision Tree RequirementsPlease generate a decision tree with a   
maximum depth of \*[ max depth ]\* (which means \*[ max depth-1 ]\*   
levels because the max depth includes the root node).\*\*Key   
Requirements:\*\*   
1. \*\*Type Matching\*\*: Use integers for int features and floating  
point numbers for float features.   
2. \*\*Decimal Precision\*\*: Use precision suited to each feature's   
scale - typically up to 3 decimals. Avoid overly precise thresholds   
(e.g., 0.165) when simpler ones (e.g., 0.22) better match value   
ranges.   
3. \*\*Feature Diversity\*\*: Use different features for each split.   
4. \*\*Rule Format\*\*: Rules must follow this exact format:   
(N) IF condition [AND condition] THEN label 1\*\*Mathematical   
Constraints:\*\*   
- Mathematical Formalism: Let depth d ∈ [1, \*[ max depth ]\*], for   
any rule r, Ir.conditions| = d-1   
- The decision tree MUST satisfy: ∀rule∈Tree, len(conditions) = \*[   
max depth-1 ]\*   
Mathematical proof required: depth=\*[ max\_depth ]\* ⇒ each path   
has  
Figure 22: CoT Tree Generation Prompting Template.

![](images/d50a71cf7df7deff7a9f129462faf4e8de4aa88fedaf0d3bc95c45e6b7da9814.jpg)  
Figure 23: CoT Tree Generation Prompting Template (continued).

![](images/d22a61bd77c0ec3ad9b5867813b6abe3240d062dcd7465605cd9eaeadb84c960.jpg)  
Figure 24: ToT Tree Generation Prompting Template.

instruction Prompt (ToT)   
## Training Samples   
Here are some examples from the training dataset:   
For each line:   
<\*,'.join(feature names)\*> <RESULT>   
% for i, idx in enumerate(sample indices) %   
(\*[ i+1 ]\*) \*[ ', '.join(feature values) ]\* \*[ label name ]\*   
% endfor %   
## Examples   
parent is Root->generate one (EXCEPTION: when parent is ROOT and   
depth is \*[ max depth-1 ]\*, generate ONLY leaves)   
IF size = big THEN yes   
IF size != big THEN [NODE]   
parent is branch->Add AND condition   
IF age >= 9 AND height < 1.3 THEN [NODE]   
IF age >= 9 AND height >= 1.3 THEN no   
Incorrect examples(useless split rules)   
IF age >= 9 AND height < 1.3 THEN no   
IF age >= 9 AND height >= 1.3 THEN no   
explanation: the split rule is useless because label is always no(the   
same label)   
## Input Context   
All possible labels:   
% for label in meta.labels %   
\*[ label.name ]\*   
% endfor %   
\*\*Current Path:\*\*   
\*[ display path ]\*   
\*\*Current depth:\*\*   
\*[ depth ]\*   
When parent node depth reaches \*[ max depth-1 ]\*, child nodes MUST   
terminate as leaves  
Figure 25: ToT Tree Generation Prompting Template (continued).