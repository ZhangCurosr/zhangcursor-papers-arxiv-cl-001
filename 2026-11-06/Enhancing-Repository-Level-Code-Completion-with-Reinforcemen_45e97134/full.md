# Enhancing Repository-Level Code Completion with Reinforcement Learning in Real-World Scenarios

Zhenlong Dai<sup>1,2</sup>, Zhuoluo Zhao<sup>1</sup>, Xiu Tang<sup>1</sup>, Sai Wu<sup>1</sup>, Chang Yao<sup>1,2</sup>, Zhipeng Gao<sup>1</sup>\*, Jingyuan Chen<sup>1</sup>

<sup>1</sup>Zhejiang University

<sup>2</sup>Innovation and Management Center, School of Software Technology (Ningbo), Zhejiang University {zhenlongdai, zhuoluozhao, tangxiu, wusai}@zju.edu.cn {changy, zhipeng.gao, jingyuanchen}@zju.edu.cn

## Abstract

The rapid development of large language models (LLMs) for code, which closely aligns with the practical needs of real-world software development, has sparked growing attention in repository-level code completion. However, existing Code LLMs present challenges in focusing on suitable contexts from the repository and in deeply reasoning about cross-file dependencies. To address these challenges, we propose a novel reinforcement learning framework for repository-level code completion. To better understand both in-file and cross-file contexts, we employ identifier-driven intent recognition to capture completion intent, thereby improving the model’s performance in realworld scenarios. To enhance cross-file reasoning, we propose a reward-driven completion learning to effectively reward in complex repository completion scenarios. To guide LLMs in completing code that aligns with intended functionality and repository context, we introduce a selective-exploration strategy that directs the model to focus on low-confidence, high-reward tokens, promoting the exploration of valuable, underexplored completion patterns. Experimental results show that our approach significantly improves the performance of Code LLMs. The code and the dataset are available at https://github.com/zhenlongDai/ RepoCompletion.

## 1 Introduction

With the rapid development of large language models (LLMs) for code (Jiang et al., 2024a; Zheng et al., 2023b), these LLMs have demonstrated advanced capabilities in code understanding and completion. Many LLMs are integrated into IDEs to support developers’ daily development, one of the most common usage scenarios is code completion (Zhou et al., 2022; Wang et al., 2023a). Previous studies have explored code completion tasks on different granularities, such as line-level code completion (Liu et al., 2024; Izadi et al., 2022), function-level code completion (Li et al., 2022; Quan et al., 2025). More recently, the repositorylevel code completion (Zhang et al., 2024) has become increasingly important because it better aligns with the practical needs of developers in realworld software development. However, repo-level code completion is a non-trivial task by considering the following situations:

![](images/bfa940faf36c2b8d0d9e04e0323d56819cc7676d0f1949e7bcd7b2a07fbcdd50.jpg)  
Figure 1: Situations of repo-level code completion.

As shown in Fig. 1 (s1), this example shows the LLM incorrectly focusing on method signatures in the local (i.e., in-file) context, while ignoring the broader global (i.e., cross-file) context. The LLM incorrectly invokes processData(product, quantity), which fails to reflect the intended action of placing an order. The proper call is processPurchase(product,quantity) (e.g., colored in green), which aligns with the intended purchase logic defined in the PurchaseProcessor class, imported from the cross-file context. This situation highlights that the LLM can’t correctly focus on local context or global context. As shown in Fig. 1 (s2), this example shows an LLM’s failure to deeply reason about the target cross-file object. The LLM incorrectly calls dbConnection.connect() directly after instantiating newUser neglecting the initialization logic newUser.save() encapsulated within the UserManager class, which internally manages the database interaction.

Therefore, a key research question regarding repo-level code completion stands out, i.e., how to improve the model’s capability to utilize both cross-file and in-file context effectively? It is a challenging task: (i) Guiding LLMs to focus on the suitable context (e.g., in-file context or cross-file context or both) remains a non-trivial problem. Different parts of the codebase depend on vastly different types of contextual information. Some completions can be resolved using local (in-file) scope, while others require a broader, cross-file perspective. LLMs still lack the capability to decide which type of context is relevant for a given completion task (Ding et al., 2023). (ii) LLMs still struggle to reason deeply about the cross-file information due to complicated project dependencies and code structures. From the repository perspective, code completion requires the LLM to have a global view that spans multiple files and modules. This means a deep understanding of how different components are architected and interact.

To tackle the above challenges, we propose a novel reinforcement learning (RL) approach for repo-level code completion, which is designed to improve the model’s completion ability in different completion scenarios. To address the challenge of focusing on a suitable context, we leverage identifier-driven intent recognition to capture the completion intent. Specifically, we adopt the GRPO (Shao et al., 2024) method for autonomous learning and design an intent reward based on the matching degree between identifiers in the predicted and real code, guiding the LLM to focus on the correct context. To address the challenges of dependency resolution and cross-file reasoning, we propose a reward-driven completion learning utilizing a reward function for repo-level code completion, providing continuous and granular reward values based on the distance of the code editing as feedback for cross-file reasoning. To guide LLMs to effectively complete code that meets the intended functionality and adheres to the repository-level context, we design a selective-exploration strategy optimization that focuses on low-confidence, highreward tokens in the model’s multiple responses, reflecting divergent thought paths and motivating the model to explore underexplored yet more valuable code completion patterns.

In summary, our paper makes the following contributions: (1) A reinforcement learning method for repo-level code completion, which effectively addresses the issue of rewards in complex repository completion scenarios and improves cross-file reasoning, enabling the model to learn autonomously on real-world repository data. (2) The first deep research into LLMs’ multi-scenario code completion ability, enhancing the model’s ability to focus on a suitable repository context through intent recognition and selective exploration. (3) IntentCoder achieves state-of-the-art (SOTA) performance on repo-level code completion tasks with code LLM (7B parameters) on RepoBench, demonstrating strong and stable improvements.

## 2 Related Work

Large Language Models for Code. A wide range of LLMs (Fried et al., 2022; Chowdhery et al., 2023; Wang et al., 2023b; Zheng et al., 2023a; Christopoulou et al., 2022) tailored for coderelated tasks, such as code comprehension and generation (Dai et al., 2024, 2025), have been developed and widely adopted in software engineering to enhance development efficiency (Chen et al., 2021; Liu et al., 2023). Some open-source models, Deepseek-Coder (Guo et al., 2025) and Qwen2.5- Coder (Hui et al., 2024; Dai et al., 2026), show strong code generation abilities. CodeLLMs have shown strong performance on benchmarks (Lai et al., 2023; Allamanis and Sutton, 2013; Lu et al., 2021; Raychev et al., 2016; Yu et al., 2024) for single-file code completion, where they can predict whole lines of code effectively. However, their performance in repo-level code completion remains limited, as LLMs struggle to capture the full repository context due to token size restrictions (Phan et al., 2024), focusing mainly on local context within individual files and neglecting cross-file dependencies (Liang et al., 2024).

Repository-level Code Completion. Repositorylevel code completion aims to provide code completion by leveraging the context of a repository. Given the large size of repositories and model context limitations, many approaches adopt the retrieve-then-generate paradigm (Agrawal et al., 2023; Cheng et al., 2024). Methods like A3- CodGen (Liao et al., 2023), CodePlan (Bairi et al., 2024), and RepoFuse (Liang et al., 2024) use static code analysis for contextual integration to improve code completion. RepoCoder (Zhang et al., 2023) refines retrieval using feedback from generated code, while RLCoder (Wang et al., 2024) improves it with a stop signal mechanism. Despite advancements in context retrieval, effectively utilizing it remains challenging. FT2Ra (Guo et al., 2024) and CoCoMIC (Ding et al., 2022) integrate retrieved content, but require architectural changes, making them unsuitable for pre-trained LLMs. In contrast, aixcoder-v2 (Li et al., 2025) uses the CoLT method to assimilate long contexts for improved code completion. Unlike previous methods, we use online reinforcement learning to enhance the LLM’s performance in various scenarios.

## 3 Methodology

## 3.1 Task Definition and Motivation

Given an incomplete code snippet x (i.e., in-file context $C _ { \mathrm { i n - f i l e } } )$ and its cross-file contexts $C _ { \mathrm { c r o s s } }$ in a specific repository, the task of repository-level code completion is to generate the completion $y .$ More formally, the task’s objective is to generate the code completion $y$ by a model $\pi \theta$ such that the conditional probability $P _ { \theta } ( y | C _ { \mathrm { i n - f l l e } } , C _ { \mathrm { c r o s s } } )$ is maximized. The definitions of in-file and crossfile contexts are in Appendix A.1, and the task motivation is in Appendix A.2.

## 3.2 Preliminary

Group Relative Policy Optimization (GRPO) is a method of reinforcement learning that explores the potential of LLMs to develop reasoning capabilities, focusing on their self-evolution through a pure RL process (Guo et al., 2025; Shao et al., 2024). GRPO is applied to estimate the advantages of language model generations by comparing responses within a query-specific group.

Specifically, in repo-level completion task, given an input $q ~ = ~ ( C _ { \mathrm { i n - f i l e } } , C _ { \mathrm { c r o s s } } )$ , GRPO samples a group of outputs $\left\{ y _ { 1 } , y _ { 2 } , \ldots , y _ { G } \right\}$ from the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }$ , and then optimizes the policy model $\pi _ { \theta }$ by maximizing the following objective:

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { q } \sim P ( \boldsymbol { Q } ) , \{ \boldsymbol { y } _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( \boldsymbol { y } | \boldsymbol { q } ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G }
$$

$$
\begin{array} { r l } & { \operatorname* { m i n } ( \frac { \pi _ { \boldsymbol { \theta } } \left( y _ { i } | \boldsymbol { q } \right) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } \left( y _ { i } | \boldsymbol { q } \right) } A _ { i } , \exp ( \frac { \pi _ { \boldsymbol { \theta } } \left( y _ { i } | \boldsymbol { q } \right) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } \left( y _ { i } | \boldsymbol { q } \right) } , 1 - \epsilon , 1 + \epsilon ) A _ { i } ) } \\ & { - \beta \mathcal { D } _ { K L } \big ( \pi _ { \boldsymbol { \theta } } \big \| \pi _ { \mathrm { r e f } } \big ) , } \end{array}\tag{1}
$$

where ϵ and $\beta$ are hyper-parameters, $\mathcal { D } _ { K L }$ is the penalty function of Kullback-Leibler divergence. $\pi _ { \mathrm { r e f } }$ is the reference model, which is initial by the policy model $\pi _ { \theta } . \quad A _ { i }$ is the advantage for the i-th response, computed using a group of rewards $\{ r _ { 1 } , r _ { 2 } , \dots , r _ { G } \}$ corresponding to the outputs within each group:

$$
A _ { i } = { \frac { r _ { i } - \operatorname { m e a n } ( r _ { 1 } , r _ { 2 } , \dots , r _ { G } ) } { \operatorname { s t d } ( r _ { 1 } , r _ { 2 } , \dots , r _ { G } ) } }\tag{2}
$$

## 3.3 Completion-Focused Reinforcement Learning

Even though LLM receives sufficient repository context information, it often fails to effectively leverage and understand it for accurate completion. Inspired by reinforcement learning, which helps mitigate the reliance on empirical memory for code completion, we propose completion-focused reinforcement learning based on GRPO (Section 3.2) to enable the model to autonomously learn code completion in a repository.

Reward-Driven Completion Learning. Employing reward feedback based on result matching presents challenges in complex and diverse crossfile completion scenarios, as the model often fails to receive rewards due to the sparsity of signals when directly matching the ground truth code. To address this challenge, we propose reward-driven completion learning, which provides more precise and granular reward feedback in the complex repolevel code completion environment, enabling the model to receive effective reward signals that enhance its cross-file reasoning ability.

Specifically, given in-file context $C _ { \mathrm { i n - f i l e } }$ and the cross-file context $C _ { \mathrm { c r o s s } } = \{ C _ { i } \} _ { i = 1 } ^ { n }$ in repository, the i-th output $y _ { i }$ in a group is sampled by the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }$ , as described by the following equations:

$$
y _ { i } = \pi _ { \theta _ { \mathrm { o l d } } } ( C _ { \mathrm { i n - f i l e } } , C _ { \mathrm { c r o s s } } ) .\tag{3}
$$

Following the prompt of repository-level completion in Fig. 4 (provided in Appendix $_ { \mathrm { A . 4 ) } }$ , we obtained a group of generated code $\{ y _ { i } \} _ { i = 1 } ^ { G }$ through multiple samplings by Formula 3. Completion reward functions that evaluate the generated code are defined from the following perspectives: (1) completion reward provides a relative quality ranking of the generated code completions. The completion reward $R _ { c }$ for the i-th output $y _ { i }$ is defined as

![](images/64cd077ac1e5cde5b20f1b12aac3f5f3000bd2b4b9ea03330271edacd16808b5.jpg)  
Figure 2: Overview of IntentCoder. (a) Illustration of the Completion-Focused RL process. (b) Illustration of the Selective-Exploration Strategy Optimization.

follows:

$$
R _ { c } = f _ { \mathrm { E d i t } } ( y _ { i } , \hat { y } ) ,\tag{4}
$$

where $\hat { y }$ is the ground truth code, f is the Levenshtein distance algorithm (Yujian and Bo, 2007) to compare the character differences (i.e., insertions, deletions, substitutions) between generated code and ground truth code. This reward provides finegrained, continuous feedback (ranging from 0 to 1), ensuring the model receives quality feedback even without exact matches. (2) Format reward $R _ { f }$ ensures the model’s output is evaluable by requiring the response to follow a specific format for reasoning and answers. A regular expression checks that these markers are present and correctly ordered. (3) Length-based reward: Extra code lines may receive spurious rewards, hindering accurate feedback and reducing training efficiency. To address this, we propose a Length-based reward $R _ { L }$ to regulate the number of generated lines for better alignment with code completion needs:

$$
R _ { L } ( y _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { i f l e n ( y _ { i } ) = 1 } \\ { 0 , } & { o t h e r w i s e , } \end{array} \right.\tag{5}
$$

where len denotes the function that calculates the number of code lines. The justification for the reward designs is provided in Appendix A.3.1. Based on these reward functions, we define the reward function R as follows:

$$
R = \alpha _ { c } * R _ { c } + \alpha _ { f } * R _ { \mathrm { f } } + \alpha _ { L } * R _ { L } ,\tag{6}
$$

where $\alpha _ { * }$ is hyper-parameters. As shown in Algorithm 1, we use the reward function R to train the model $\pi _ { \theta } ,$ enabling the model to autonomously learn from repository data.

Algorithm 1 Completion-Focused Reinforcement Learning   
1: Input: initial policy model $\pi _ { \theta } ^ { i n i t }$ , reward function $R ,$   
dataset $\mathcal { D } = \{ \hat { C } _ { \mathrm { i n - f i l e } } , \dot { C } _ { \mathrm { c r o s s } , \hat { y } } \}$ ; hyperparameters µ   
2: policy model $\pi _ { \theta }  \pi _ { \theta } ^ { i n i t }$   
3: reference model $\pi _ { \mathrm { r e f } }  \pi _ { \theta }$   
4: for step = 1, . . . , M do   
5: Sample a batch $\mathcal { D } _ { b }$ from $\mathcal { D }$   
6: Update the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }  \pi _ { \theta }$   
7: $q \mathcal { \bar {  } } \{ C _ { \mathrm { i n - f i l e } } , \bar { C } _ { \mathrm { c r o s s } } \} \in \mathcal { D } _ { b }$   
8: Sample $\{ y _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | C _ { \mathrm { i n - f i l e } } , C _ { \mathrm { c r o s s } } )$ for $q \in \mathcal { D } _ { b }$   
9: $\{ R _ { i } \} _ { i = 1 } ^ { G }  R ( y _ { i } , \hat { y } )$ for $y _ { i } \in \{ y _ { i } \} _ { i = 1 } ^ { G }$ (Equation 6)   
10: Compute $A _ { i }$ for $y _ { i } \in \{ y _ { i } \} _ { i = 1 } ^ { G }$ (Equation 2)   
11: for iteratio ${ \mathfrak { a } } = 1 , \ldots , \mu$ do   
12: Update π<sub>θ</sub> by maximizing J<sub>GRPO</sub>(θ) (Equation 1)   
13: end for   
14: end for   
15: Output: π<sub>θ</sub>

Identifier-Driven Intent Recognition. In realworld code completion scenarios, the performance of LLMs is often enhanced by utilizing cross-file context from the repository (Zhang et al., 2024; Eghbali and Pradel, 2024). However, the model struggles to focus on suitable contexts from infile and cross-file contexts (Ding et al., 2023), resulting in incorrect code completions. To address this challenge, we propose Identifier-Driven Intent Recognition, which is designed to guide the LLM in identifying the developer’s completion intent and understanding code contexts, thereby correctly focusing on suitable contexts.

Specifically, as shown in Fig. 4, we instruct the model’s reasoning process in the prompt to focus on the user’s intent of code completion (e.g., colored in blue), directing its reasoning towards the intent of completion in the reasoning proces. Given that identifiers contain semantic information from the code, we utilize an identifier-based reward function $R _ { I }$ to evaluate the degree of correctness in the model’s intent recognition. As shown in Fig. 2, we parse the code and extract the identifiers (e.g., variable, function, and class names) from the model’s prediction $y$ and ground truth code yˆ. The identifierbased reward function $R _ { i }$ is calculated as follows:

$$
R _ { I } = F _ { 1 } ( f _ { \mathrm { i d } } ( \pi _ { \theta _ { \mathrm { o l d } } } ( C _ { \mathrm { i n - f l e } } , C _ { \mathrm { c r o s s } } ) ) , f _ { \mathrm { i d } } ( \hat { y } ) ) ,\tag{7}
$$

where $f _ { \mathrm { i d } }$ extracts the sets of identifiers from code. $F _ { 1 }$ is the harmonic mean of the precision and recall metric, which is used to evaluate the degree of intent consistency between the predicted and ground truth code. This approach enables the model to properly consider contextual information and adjust its focus based on the situation, thereby effectively leveraging the advantages of both in-file and cross-file contexts to improve performance.

To effectively coordinate the intent and completion rewards, we join them as follows:

$$
R _ { i c } = ( 1 + R _ { I } ) * R _ { c } ,
$$

$$
{ \cal R } = \alpha _ { i c } * { \cal R } _ { i c } + \alpha _ { f } * { \cal R } _ { \mathrm { f } } + \alpha _ { L } * { \cal R } _ { L } ,\tag{8}
$$

(9)

where R is the overall reward function, α<sub>∗</sub> is hyperparameter. By combining these two reward signals, the model is encouraged to prioritize completing the code in a way that both fulfills the user’s intended functionality and adheres to the repo-level context. When the edit distance between two generated codes is similar in a group of outputs $\{ y _ { i } \} _ { i = 1 } ^ { G }$ the intent reward acts as a coefficient, allowing for a more accurate reward assignment and reducing the lack of distinction in reward values. The justification and further analysis of reward design are provided in Appendix A.3.2.

Selective-Exploration Strategy Optimization. The model’s multiple responses to the same completion task reflect different thought paths, with some responses aligning in part as a consensus, while others show divergence on key points. Rewarding the key information where the model shows uncertainty can help cross-file reasoning and contextual attention in the correct contexts. As shown in Fig. 2, both candidates correctly identify ListCellViewAdapter (a class in cross-file context). The difference is the generic type (ProgramData vs. MySchedule), which is defined in different files and requires cross-file reasoning to resolve. This type of token (e.g., colored in green/red) significantly influences the reasoning strategy. However, result-level reward assigns credit to the entire line with the same reward, even when the correct decision depends on specific tokens. During the RL process, some key tokens (e.g., colored in green) in the group may have a lower generation probability but receive rewards higher than the group’s average. Based on the above observations, we propose a selective-exploration strategy optimization that emphasizes the tokens that correspond to competing cross-file hypotheses (e.g., ProgramData, MySchedule) rather than the shared tokens that are predictable (e.g., ListCellViewAdapter). The key idea is to encourage the model to focus on the competing tokens that represent the high-value and critical aspects of cross-file reasoning and the solution.

Specifically, for the t-th token of the i-th output $y _ { i , t }$ within the group, the confidence function P and value function $V$ are as follows:

$$
\begin{array} { l } { { \displaystyle P ( y _ { i , t } ) = \log \pi _ { \theta } ( y _ { i , t } | q , y _ { < t } ) } } \\ { { \displaystyle ~ - \frac { 1 } { N } \sum _ { j = 1 } ^ { G } \sum _ { t = 1 } ^ { L _ { i } } \log \pi _ { \theta } ( y _ { i , t } | q , y _ { < t } ) , } } \\ { { \displaystyle V ( y _ { i , t } ) = A ( y _ { i , t } ) - \frac { 1 } { G } \sum _ { j = 1 } ^ { G } A ( y _ { j } ) } , } \end{array}\tag{10}
$$

where N is the total number of tokens in a group of outputs, $L _ { i }$ is the number of tokens in the i-th output. Then the tokens that have high value and low confidence are selected as follows:

$$
T _ { s } = \{ y _ { i , t } \ \vert \ P ( y _ { i , t } ) < 0 , V ( y _ { i , t } ) > 0 \} ,\tag{11}
$$

where $T _ { \mathrm { s e l e c t } }$ is the set of selected tokens. To simplify the expression and enhance readability, we define the function $A d v ( y _ { i , t } )$ as follows:

$$
\begin{array} { r l } & { A d v ( \cdot ) = \operatorname* { m i n } ( \frac { \pi _ { \theta } \left( y _ { i } | q \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { i } | q \right) } A _ { i } , } \\ & { \mathrm { c l i p } ( \frac { \pi _ { \theta } \left( y _ { i } | q \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( y _ { i } | q \right) } , 1 - \epsilon , 1 + \epsilon ) A _ { i } ) . } \end{array}\tag{12}
$$

Based on the function Adv, the training of the model $\pi _ { \theta }$ for the position of token $y _ { i , t }$ is defined as maximizing the following objective:

$$
\begin{array} { r } { \mathcal { I } ( y _ { i , t } ) = \left\{ \begin{array} { l l } { A d v ( y _ { i , t } ) - \beta \mathcal { D } _ { K L } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) , y _ { i , t } \notin T _ { s } } \\ { A d v ( y _ { i , t } ) , y _ { i , t } \in T _ { s } , } \end{array} \right. } \end{array}\tag{13}
$$

where token $y _ { i , t } \in T _ { \mathrm { s e l e c t e d } }$ removes the KL penalty compared to other tokens. By applying fine-grained gradient updates to tokens in different areas, we adjust the model’s attention away from tokens where it already has high confidence and shift it towards those tokens that have the potential to improve the overall output. This approach encourages the model to gain a deeper understanding of code completion by focusing on divergent contextual areas and exploring valuable code completion patterns.

## 4 Experiments

## 4.1 Dataset

To study repository-level code completion in the context of in-file and cross-file scenarios, we construct the ICRepo dataset (In-file & Cross-file Repository-level Completion) for model training. We collect 32K repo-level code completion examples from the RepoBench-C and RepoBench-P datasets (Liu et al., 2023), with an equal number of Python and Java examples. We contain three completion scenarios as follows: Cross-File-First (CF-F), Cross-File-Random (CF-R), In-File (IF). Overall statistics of the dataset are given in Table 1. The training dataset encompasses a variety of real-world code completion scenarios, ensuring the model is exposed to diverse challenges. By incorporating both in-file and cross-file contexts, the dataset covers practical code completion situations that developers frequently encounter in actual projects. For the test dataset, we use Repobench-Java and Repobench-Python (Liu et al., 2023). Further details are provided in Appendix A.5.

<table><tr><td rowspan="2">Language</td><td rowspan="2">Sets</td><td rowspan="2">Num.</td><td colspan="3">Completion position</td></tr><tr><td>IF</td><td>CF-F</td><td>CF-R</td></tr><tr><td rowspan="3">Java&amp;Python</td><td>train</td><td>32,000</td><td>10,100</td><td>11,300</td><td>10,600</td></tr><tr><td>val</td><td>1,000</td><td>300</td><td>400</td><td>300</td></tr><tr><td>test</td><td>6,000</td><td>2,000</td><td>2,000</td><td>2,000</td></tr></table>

Table 1: Dataset Statistics of ICRepo.

## 4.2 Baselines

We consider the following open-source Code LLMs for comparison, which are widely used in code completion tasks. To simplify the model names, we have removed the “instruct” suffix in the following introductions: StarCoder2 (Lozhkov et al., 2024): StarCoder2-15B and StarCoder2-7B. DeepSeek-Coder (Guo et al., 2025): Deepseek-Coder-6.7B and DeepSeek-Coder-33B. CodeLlama (Roziere et al., 2023): CodeLlama-7B and CodeLlama-34B. Qwen2.5-Coder (Hui et al., 2024): Qwen2.5- Coder-7B and Qwen2.5-Coder-33B. aixcoder-7Bv2 (Li et al., 2025) is currently the SOTA 7B model in repo-level code completion. Then we compare our method with the following methods in the training model for repo-level code completion: finetuning (Hu et al., 2022), GRPO (Guo et al., 2025), CoLT (Li et al., 2025). Further Details are provided in Appendix A.6.

## 4.3 Implementation Details

We train IntentCoder based on Deepseek-Coder-6.7B and Qwen2.5-Coder-7B, respectively. For each prompt of training data, the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }$ generates 6 outputs in a group using a temperature of 0.7 and top-p sampling with a value of 1.0. To improve training efficiency, we use LoRA (Hu et al., 2022), vllm (Kwon et al., 2023), and deepspeed (Rasley et al., 2020) for train the model $\pi _ { \boldsymbol { \theta } }$ Further details are provided in the Appendix A.7.

## 4.4 Evaluation Metrics

Following the previous work (Liu et al., 2023; Eghbali and Pradel, 2024; Zhang et al., 2023; Jiang et al., 2024b), we report performance in two main categories to evaluate the performance of codeLLMs: Code Match: To evaluate the accuracy of the predicted code lines, we use Exact Match (EM) and Edit Similarity (ES) (Levenshtein et al., 1966). Identifier Match (Ding et al., 2022, 2023): To evaluate the model’s ability to understand the completion intent and predict the correct APIs, we use the following metrics about Identifier: Exact Match (EM),Precision,Recall and F1. Further details can be found in Appendix A.8.

## 4.5 Experimental Results

Repo-level Code Completion Evaluation and Method Analysis. To evaluate the effectiveness of IntentCoder, we apply our training method to Qwen2.5-Coder-7B and DeepSeek-Coder-6.7B, and then compare it with nine Code LLMs and two strong training methods for LLMs. Table 2 shows the experimental results of our model and other baselines. It is obvious that: (1) Regarding the comparison with other models, IntentCoder outperforms all strong baselines by a large margin in both code match and identifier match, surpassing even larger models. For example, IntentCoder<sub>Qwen</sub> achieves 42.47% and 35.93% in Code EM for Java and Python, surpassing the next best model (i.e., Qwen2.5-Coder-32B) by a significant margin of 6.53% and 5.87%. Similarly, in terms of F1 in Identifier Match, IntentCoder<sub>Qwen</sub> outperforms the next best model (i.e., Qwen2.5-Coder-32B) by a significant margin of 5.97% and 4.97%. Compared to models with the same parameters in Code EM, IntentCoder<sub>Qwen</sub> outperforms the current SOTA model (i.e., aixcoder-7B-v2) by a significant margin of 8.53% in Java and 12.07% in Python, which clearly demonstrates our model’s code completion capability. (2) Regarding the comparison with other methods, our approach demonstrates substantial improvements across all evaluation metrics over other training methods on two main strong baselines (e.g., Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B). Specifically, IntentCoder<sub>DeepSeek</sub> outperforms DeepSeek-Coder-6.7B by up to 11.40% and 20.30% in Code EM for Java and Python. Similarly, IntentCoder<sub>Qwen</sub> outperforms Qwen2.5- Coder-7B by up to 15.67% and 15.96% . Overll, IntentCoder shows substantial and stable improvements compared to other models and methods. Further experiments and method analysis are discussed in Appendix A.9.

<table><tr><td rowspan="3">Model</td><td colspan="6">RepoBench-Java</td><td colspan="6">RepoBench-Python</td></tr><tr><td colspan="2">Code Match</td><td colspan="4">Identifier Match</td><td colspan="2">Code Match</td><td colspan="3">Identifier Match</td></tr><tr><td>EM</td><td>ES</td><td>EM</td><td>Precision</td><td>Recall</td><td>F1</td><td>EM</td><td>ES</td><td>EM</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>StarCoder2-7B</td><td>12.20</td><td>29.78</td><td>15.37</td><td>24.64</td><td>28.19</td><td>25.09</td><td>1.80</td><td>14.47</td><td>3.83</td><td>7.72</td><td>13.25</td><td>8.40</td></tr><tr><td>CodeLlama-7B</td><td>12.40</td><td>48.50</td><td>15.90</td><td>35.60</td><td>41.60</td><td>36.46</td><td>4.47</td><td>25.56</td><td>6.37</td><td>18.26</td><td>26.68</td><td>18.84</td></tr><tr><td>aixcoder-7B-v2</td><td>33.07</td><td>68.97</td><td>38.70</td><td>60.57</td><td>65.97</td><td>61.07</td><td>23.20</td><td>59.94</td><td>29.63</td><td>54.70</td><td>62.84</td><td>55.41</td></tr><tr><td>Starcoder2-15B</td><td>12.20</td><td>28.59</td><td>15.67</td><td>24.51</td><td>27.77</td><td>24.92</td><td>1.83</td><td>13.70</td><td>4.07</td><td>7.66</td><td>13.05</td><td>8.34</td></tr><tr><td>CodeLlama-34B</td><td>12.13</td><td>45.39</td><td>15.43</td><td>33.59</td><td>39.11</td><td>34.01</td><td>4.40</td><td>26.02</td><td>6.70</td><td>18.97</td><td>27.55</td><td>19.07</td></tr><tr><td>DeepSeek-Coder-33B</td><td>26.70</td><td>62.03</td><td>31.53</td><td>53.11</td><td>57.99</td><td>53.39</td><td>14.03</td><td>45.00</td><td>18.57</td><td>37.81</td><td>45.49</td><td>38.30</td></tr><tr><td>Qwen2.5-Coder-32B</td><td>35.07</td><td>71.13</td><td>41.10</td><td>63.12</td><td>65.28</td><td>62.56</td><td>29.40</td><td>63.82</td><td>35.40</td><td>57.52</td><td>61.69</td><td>57.43</td></tr><tr><td>DeepSeek-Coder-6.7B</td><td>30.20</td><td>66.28</td><td>35.97</td><td>57.67</td><td>62.81</td><td>57.94</td><td>14.97</td><td>43.87</td><td>19.20</td><td>38.13</td><td>47.20</td><td>38.94</td></tr><tr><td>DeepSeek-Coder-6.7Bfine-tuning</td><td>38.93</td><td>74.37</td><td>44.77</td><td>66.23</td><td>66.17</td><td>65.22</td><td>31.27</td><td>66.36</td><td>37.53</td><td>59.44</td><td>58.76</td><td>57.66</td></tr><tr><td>DeepSeek-Coder-6.7BGRPO</td><td>35.47</td><td>71.29</td><td>41.57</td><td>63.95</td><td>68.13</td><td>63.83</td><td>34.70</td><td>68.82</td><td>40.93</td><td>62.87</td><td>60.64</td><td>60.39</td></tr><tr><td>DeepSeek-Coder-6.7BCoLT</td><td>35.23</td><td>71.52</td><td>41.07</td><td>61.09</td><td>61.94</td><td>61.51</td><td>30.20</td><td>67.03</td><td>40.20</td><td>53.66</td><td>59.26</td><td>56.32</td></tr><tr><td>IntentCoderDeepSeek</td><td>41.60</td><td>77.64</td><td>47.73</td><td>69.74</td><td>69.18</td><td>68.53</td><td>35.27</td><td>71.12</td><td>42.10</td><td>64.41</td><td>62.98</td><td>62.39</td></tr><tr><td>Qwen2.5-Coder-7B</td><td>26.80</td><td>62.77</td><td>32.70</td><td>54.74</td><td>57.06</td><td>53.81</td><td>19.97</td><td>52.39</td><td>25.20</td><td>46.55</td><td>52.45</td><td>46.51</td></tr><tr><td>Qwen2.5-Coder-7Bfine-tuning</td><td>37.43</td><td>74.77</td><td>43.50</td><td>65.93</td><td>65.78</td><td>64.88</td><td>31.53</td><td>66.61</td><td>37.53</td><td>59.59</td><td>58.69</td><td>57.81</td></tr><tr><td>Qwen2.5-Coder-7BGRPO</td><td>37.50</td><td>73.49</td><td>43.60</td><td>65.87</td><td>67.58</td><td>65.02</td><td>35.13</td><td>68.71</td><td>41.20</td><td>62.21</td><td>60.10</td><td>59.86</td></tr><tr><td>Qwen2.5-Coder-7BCoLT</td><td>38.80</td><td>75.33</td><td>42.40</td><td>61.40</td><td>62.43</td><td>61.91</td><td>33.00</td><td>69.88</td><td>40.20</td><td>61.29</td><td>59.81</td><td>60.54</td></tr><tr><td> $\mathrm { I n t e n t C o d e r } _ { \mathrm { Q w e n } }$ </td><td>42.47</td><td>77.52</td><td>48.57</td><td>69.95</td><td>69.60</td><td>68.75</td><td>35.93</td><td>71.58</td><td>42.70</td><td>65.57</td><td>63.80</td><td>63.37</td></tr></table>

Table 2: Evaluation results on the RepoBench-Java/Python dataset. All results in the table are reported in percentage (%). The best method is in boldface, and the second best method is underlined for each metric.

<table><tr><td rowspan="2">Model</td><td colspan="2">Code Match</td><td colspan="4">Identifier Match</td></tr><tr><td>EM</td><td>ES</td><td>EM</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>IntentCoder</td><td>35.93</td><td>71.58</td><td>42.70</td><td>65.57</td><td>63.80</td><td>63.37</td></tr><tr><td>w/o IDIR</td><td>34.60</td><td>69.98</td><td>41.27</td><td>63.74</td><td>63.10</td><td>62.05</td></tr><tr><td>w/o RDCL</td><td>34.30</td><td>68.97</td><td>40.20</td><td>62.24</td><td>60.87</td><td>60.25</td></tr><tr><td>w/o CFCL</td><td>34.47</td><td>67.73</td><td>40.70</td><td>62.06</td><td>59.61</td><td>59.45</td></tr><tr><td>w/o SESO</td><td>34.47</td><td>70.43</td><td>40.97</td><td>64.58</td><td>62.99</td><td>62.39</td></tr><tr><td>w/o All</td><td>19.97</td><td>52.39</td><td>25.20</td><td>46.55</td><td>52.45</td><td>46.51</td></tr></table>

Table 3: Results of Ablation study on the RepoBench-Python dataset in Qwen2.5-Coder-7b.

Contributions of Each Component. In this RQ, we conduct an ablation study on IntentCoder<sub>Qwen</sub> to assess the contribution of different techniques by removing key components (i.e., identifier-driven intent recognition (IDIR), reward-driven completion learning (RDCL), completion-focused RL (CFRL), selective-exploration strategy optimization (SESO)) of our approach separately. For IDIR and RDCL components, we separately remove their the design of the reward function and retain the basic RL process (i.e., GRPO). Similarly, for CFCL, we remove all reward design (i.e., IDIR, RDCL), retain the intention prompt and adopt Code EM as a substitute for reward. As illustrated in Table 3, The experimental results show that: (1) No matter which component we drop, it consistently hurts the overall performance of IntentCoder across all metrics, signaling the importance and effectiveness of all four components. When all the components were removed, the performance of the model significantly declined. (2) When CFCL is removed, the code match and identifier match metrics, except for the EM metric, show a more significant drop than when removing IDIR and RDCL separately. This justifies the importance and necessity of CFCL in effectively coordinating the intent and completion rewards, which both fulfill the user’s intended functionality and adhere to the repository-level context. Notably, EM is slightly higher than removing any one of them (e.g., colored in gray), since we use the exact code match as an alternative to the reward function. Similarly, the ablation study on Java is provided in Appendix Table 7.

Robustness Evaluation. To evaluate the robustness of our training method, we assess the influence of different training methods on reducing crossfile context-induced failures (i.e., fail) and improving the accuracy of completion (i.e., pass). As shown in Table 4, compared to the baseline models, our training method significantly reduces cross-file context-induced failures and improves completion accuracy, thereby improving overall performance. For instance, our approach has reduced the number of failures of DeepSeek-Coder-6.7b on the Python dataset from -130 to -41 and increased the number of pass from 189 to 709, representing an overall improvement of 668. Compared with other methods, our approach reduces the failure rate comparable to GRPO and significantly improves the pass rate compared to the other methods, thereby achieving significant overall improvement. This justifies the importance and robustness of our method in real-world code completion scenarios, effectively handling and understanding in-file and cross-file contexts.

![](images/c17f8e90d79dbc53f3cb90a6ddfa22462d6fffca266a8d341735df6f90d0503e.jpg)

![](images/a28438ca0541752edd31aeee337c89e1fc624e031dfd93285b4ec93c31364955.jpg)

![](images/a5cbd3e58bc649e50d9df073b38b05ab0f633baf0efe301fe3c6709768d2cebc.jpg)  
Figure 3: Performance of IntentCoder in Various Completion Scenarios (CF-F,CF-R,IF).

<table><tr><td rowspan="2">Method</td><td colspan="3">Java</td><td colspan="3">Python</td></tr><tr><td>fail</td><td>pass</td><td>overall</td><td>fail</td><td>pass</td><td>overall</td></tr><tr><td>DeepSeek-Coder-6.7B</td><td>-131</td><td>+351</td><td>+220</td><td>-130</td><td>+189</td><td>+59</td></tr><tr><td>fine-tuning</td><td>-117</td><td>+599</td><td>+482</td><td>-61</td><td>+609</td><td>+548</td></tr><tr><td>GRPO</td><td>-78</td><td>+456</td><td>+378</td><td>-40</td><td>+691</td><td>+651</td></tr><tr><td>CoLT</td><td>-91</td><td>+462</td><td>+371</td><td>-156</td><td>+672</td><td>+516</td></tr><tr><td>IntentCoderdeepseek</td><td>-83</td><td>+645</td><td>+562</td><td>-41</td><td>+709</td><td>+668</td></tr><tr><td>Qwen2.5-Coder-7B</td><td>-128</td><td>+301</td><td>+173</td><td>-138</td><td>+225</td><td>+87</td></tr><tr><td>fine-tuning</td><td>-105</td><td>+597</td><td>+492</td><td>-113</td><td>+547</td><td>+434</td></tr><tr><td>GRPO</td><td>-54</td><td>+548</td><td>+494</td><td>-62</td><td>+604</td><td>+542</td></tr><tr><td>CoLT</td><td>-102</td><td>+635</td><td>+533</td><td>-125</td><td>+603</td><td>+478</td></tr><tr><td>IntentCoderQwen</td><td>-58</td><td>+701</td><td>+643</td><td>-60</td><td>+626</td><td>+566</td></tr></table>

Table 4: Result of Exact Match (EM) count.

Various Scenarios Evaluation. To further evaluate the effectiveness of IntentCoder in various completion scenarios, we compare our approach with other methods (i.e., fine-tuning, GRPO) using the Code EM metric. As shown in Fig.3, in all completion scenarios (CF-F, CF-R, IF) that are illustrated in Appendix A.5.1, our method generally outperforms other methods, although there is one instance where it performs slightly worse. The experimental results show that: (1) In the IF setting, IntentCoder performs significantly better than other methods, indicating that IntentCoder is not reliant on guessing based on prior cross-file information and effectively handles in-file contexts. (2) Similarly, in the CF-F setting, IntentCoder outperforms other methods consistently, demonstrating that it does not incorrectly depend on in-file context or previous cross-file information, and accurately handles cross-file dependencies, even when encountered for the first time. Overall, compared to the baseline, IntentCoder achieves stable improvement and effectively handles different scenarios.

Runtime latency and Case Study. To evaluate the runtime latency, We conducted experiments on a test set of 3,000 samples on IntentCoder using the vLLM inference library. RepoBench’s max input tokens is 2048, the context fed into IntentCoder averages 1121 tokens, keeping max input latency under 238 ms. The maximum output token length of the <think> block is 119, with an average length of 20 tokens. The additional latency is small, with an average delay of less than 25 ms and a maximum delay not exceeding 145 ms. We believe the small increase in latency is acceptable, as the difference is imperceptible to humans and accuracy is improved. To illustrate the performance of Intent-Coder compared to other models, we have included a case study in the Appendix A.10.

## 5 Conclusion

This research aims to improve the Code LLM’s capability of repo-level code completion by enhancing their capabilities in handling both in-file and cross-file contexts. To perform this task, we propose a novel training approach designed to improve LLMs’ understanding of complex repo-level code by incorporating reward-driven learning for better cross-file reasoning and intent recognition to focus on suitable contexts from in-file and cross-file contexts. Selective-exploration strategy optimization guides the model in exploring more valuable code completion patterns, resulting in more accurate and context-aware code suggestions. Experimental results demonstrate that the effectiveness of

IntentCoder . We hope our work advances LLMs to handle real-world repo-level challenges and provides valuable insights into improving code LLMs for software engineering development.

## 6 Limitations.

Several limitations are concerned with our work. Firstly, we do not utilize execution-based rewards due to the complexities involved in constructing executable environments, which are difficult to scale for large datasets. Additionally, the time-intensive nature of software building and testing renders this approach impractical for model training. Although execution-based rewards represent a valuable research avenue, our focus is on rewards derived from edit similarity and identifier matching. This design is more flexible, scalable, and efficient, enabling practical application across large datasets, even when execution is not feasible, and supporting sparse rewards.

Secondly, in this work, we intentionally focused only on strengthening the generation side. i.e., improving code completion results by effectively using cross-file and in-file context. Intent-Coder can naturally be integrated into any retrievalaugmented pipeline as the generator component. However, we believe that research on how to enable LLMs to search and select contexts in a repository within an acceptable time cost and context input limitation is an interesting topic for our future work.

Thirdly, while larger and more capable LLMs may yield better reasoning results during RL training, they come with higher computational costs, such as increased GPU memory usage and longer training times. Due to resource constraints and our goal of supporting community research, we only trained on 7B. We have released our replication package to enable others to replicate our work and verify their ideas.

## 7 Acknowledgements

This research was supported by the National Natural Science Foundation of China (No.62507040), the “Pioneer” and “Leading Goose” R&D Program of Zhejiang (2025C02022), the Ningbo “Yongjiang Talent Program” Youth Innovation Project (2024A-156-G), and Ant Group.

## References

Lakshya A Agrawal, Aditya Kanade, Navin Goyal, Shuvendu Lahiri, and Sriram Rajamani. 2023. Monitorguided decoding of code lms with static analysis of repository context. Advances in Neural Information Processing Systems, 36:32270–32298.

Miltiadis Allamanis and Charles Sutton. 2013. Mining source code repositories at massive scale using language modeling. In 2013 10th working conference on mining software repositories (MSR), pages 207–216. IEEE.

Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, Xiaopeng Li, Yuchen Tian, Ming Tan, Wasi Uddin Ahmad, Shiqi Wang, Qing Sun, Mingyue Shang, and 1 others. 2022. Multi-lingual evaluation of code generation models. arXiv preprint arXiv:2210.14868.

Ramakrishna Bairi, Atharv Sonwane, Aditya Kanade, Vageesh D C, Arun Iyer, Suresh Parthasarathy, Sriram Rajamani, Balasubramanyan Ashok, and Shashank Shet. 2024. Codeplan: Repository-level coding using llms and planning. Proceedings of the ACM on Software Engineering, 1(FSE):675–698.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflowguided retrieval augmentation for repository-level code completion. arXiv preprint arXiv:2405.19782.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, and 1 others. 2023. Palm: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24(240):1–113.

Fenia Christopoulou, Gerasimos Lampouras, Milan Gritta, Guchun Zhang, Yinpeng Guo, Zhongqi Li, Qi Zhang, Meng Xiao, Bo Shen, Lin Li, and 1 others. 2022. Pangu-coder: Program synthesis with function-level language modeling. arXiv preprint arXiv:2207.11280.

Zhenlong Dai, Bingrui Chen, Zhuoluo Zhao, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, and Jingyuan Chen. 2025. Less is more: Adaptive program repair with bug localization and preference learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 39, pages 128–136.

Zhenlong Dai, Chang Yao, WenKang Han, Yuanying Yuanying, Zhipeng Gao, and Jingyuan Chen. 2024. Mpcoder: Multi-user personalized code generator with explicit and implicit style representation learning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3765–3780.

Zhenlong Dai, Zhuoluo Zhao, Hengning Wang, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, and Jingyuan Chen. 2026. Learner-tailored program repair: A solution generator with iterative edit-driven retrieval enhancement. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 30467–30475.

Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and 1 others. 2023. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion. Advances in Neural Information Processing Systems, 36:46701–46723.

Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. 2022. Cocomic: Code completion by jointly modeling in-file and cross-file context. arXiv preprint arXiv:2212.10007.

Aryaz Eghbali and Michael Pradel. 2024. Dehallucinator: Iterative grounding for llm-based code completion. arXiv preprint arXiv:2401.01701.

Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2022. Incoder: A generative model for code infilling and synthesis. arXiv preprint arXiv:2204.05999.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Qi Guo, Xiaohong Li, Xiaofei Xie, Shangqing Liu, Ze Tang, Ruitao Feng, Junjie Wang, Jidong Ge, and Lei Bu. 2024. Ft2ra: A fine-tuning-inspired approach to retrieval-augmented code completion. In Proceedings ofthe 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, pages 313– 324.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, and 1 others. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186.

Maliheh Izadi, Roberta Gismondi, and Georgios Gousios. 2022. Codefill: Multi-token code completion by jointly learning from structure and naming sequences. In Proceedings ofthe 44th international conference on software engineering, pages 401–412.

Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. 2024a. A survey on large language models for code generation. arXiv preprint arXiv:2406.00515.

Siyuan Jiang, Jia Li, He Zong, Huanyu Liu, Hao Zhu, Shukai Hu, Erlu Li, Jiazheng Ding, Yu Han, Wei Ning, and 1 others. 2024b. aixcoder-7b: A lightweight and effective large language model for code completion. arXiv e-prints, pages arXiv–2410.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2023. Ds-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning, pages 18319–18345. PMLR.

Vladimir I Levenshtein and 1 others. 1966. Binary codes capable of correcting deletions, insertions, and reversals. In Soviet physics doklady, volume 10, pages 707–710. Soviet Union.

Jia Li, Hao Zhu, Huanyu Liu, Xianjie Shi, He Zong, Yihong Dong, Kechi Zhang, Siyuan Jiang, Zhi Jin, and Ge Li. 2025. aixcoder-7b-v2: Training llms to fully utilize the long context in repository-level code completion. arXiv preprint arXiv:2503.15301.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, and 1 others. 2022. Competition-level code generation with alphacode. Science, 378(6624):1092–1097.

Ming Liang, Xiaoheng Xie, Gehao Zhang, Xunjin Zheng, Peng Di, Hongwei Chen, Chengpeng Wang, Gang Fan, and 1 others. 2024. Repofuse: Repositorylevel code completion with fused dual context. arXiv preprint arXiv:2402.14323.

Dianshu Liao, Shidong Pan, Qing Huang, Xiaoxue Ren, Zhenchang Xing, Huan Jin, and Qinying Li. 2023. Context-aware code generation framework for code repositories: Local, global, and third-party library awareness. CoRR.

Fang Liu, Zhiyi Fu, Ge Li, Zhi Jin, Hui Liu, Yiyang Hao, and Li Zhang. 2024. Non-autoregressive line-level code completion. ACM Transactions on Software Engineering and Methodology, 33(5):1–34.

Tianyang Liu, Canwen Xu, and Julian McAuley. 2023. Repobench: Benchmarking repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Fed erico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, and 1 others. 2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173.

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, and 1 others. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664.

Huy Nhat Phan, Hoang Nhat Phan, Tien N Nguyen, and Nghi DQ Bui. 2024. Repohyper: Better context retrieval is all you need for repository-level code completion. CoRR.

Shanghaoran Quan, Jiaxi Yang, Bowen Yu, Bo Zheng, Dayiheng Liu, An Yang, Xuancheng Ren, Bofei Gao, Yibo Miao, Yunlong Feng, and 1 others. 2025. Codeelo: Benchmarking competition-level code generation of llms with human-comparable elo ratings. arXiv preprint arXiv:2501.01257.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 3505–3506.

Veselin Raychev, Pavol Bielik, and Martin Vechev. 2016. Probabilistic model for code with decision trees. ACM SIGPLAN Notices, 51(10):731–747.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond, volume 4. Now Publishers Inc.

Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, and 1 others. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Chaozheng Wang, Junhao Hu, Cuiyun Gao, Yu Jin, Tao Xie, Hailiang Huang, Zhenyu Lei, and Yuetang Deng. 2023a. How practitioners expect code completion? In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 1294– 1306.

Yanlin Wang, Yanli Wang, Daya Guo, Jiachi Chen, Ruikai Zhang, Yuchi Ma, and Zibin Zheng. 2024. Rlcoder: Reinforcement learning for repository-level code completion. arXiv preprint arXiv:2407.19487.

Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. 2023b. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922.

Zhaojian Yu, Yilun Zhao, Arman Cohan, and Xiao-Ping Zhang. 2024. Humaneval pro and mbpp pro: Evaluating large language models on self-invoking code generation. arXiv preprint arXiv:2412.21199.

Li Yujian and Liu Bo. 2007. A normalized levenshtein distance metric. IEEE transactions on pattern analysis and machine intelligence, 29(6):1091–1095.

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code completion through iterative retrieval and generation. arXiv preprint arXiv:2303.12570.

Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. Codeagent: Enhancing code generation with tool-integrated agent systems for realworld repo-level coding challenges. arXiv preprint arXiv:2401.07339.

Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Lei Shen, Zihan Wang, Andi Wang, Yang Li, and 1 others. 2023a. Codegeex: A pretrained model for code generation with multilingual benchmarking on humaneval-x. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5673–5684.

Zibin Zheng, Kaiwen Ning, Yanlin Wang, Jingwen Zhang, Dewu Zheng, Mingxi Ye, and Jiachi Chen. 2023b. A survey of large language models for code: Evolution, benchmarking, and future trends. arXiv preprint arXiv:2311.10372.

Wen Zhou, Seohyun Kim, Vijayaraghavan Murali, and Gareth Ari Aye. 2022. Improving code autocompletion with transfer learning. In Proceedings of the 44th International Conference on Software Engineering: Software Engineering in Practice, pages 161–162.

## A Appendix

## A.1 Definition

In repository-level code completion, in-file contexts and cross-file contexts refer to different types of contextual information that help a model to predict and complete code snippets:

(1) In-file Context: Following the previous work (Allamanis and Sutton, 2013; Chen et al., 2021; Athiwaratkun et al., 2022), in-file contexts refer to the information contained within the prefix of the current file itself, specifically the code that precedes the ground truth code in the incomplete code snippet. In-file context provides the immediate surrounding code before the incomplete snippet, which typically helps the model understand the structure, syntax, and logic of the code. This context is crucial for making predictions when code completion can be inferred solely from the current file’s contents, without the need for additional external context.

(2) Cross-file Context: Cross-file Context refers to the information contained in external files within the repository upon which the current file depends. Such information can include functions, classes, or libraries imported into the current file. This type of context is extracted by analyzing dependencies or references to other files within the repository (Ding et al., 2022), providing a broader scope of information for code completion. Leveraging the cross-file context is essential when the in-file context is insufficient for completing the code.

Both types of In-file and Cross-file contexts are essential for repository-level code completion, allowing the model to make more informed predictions and generate accurate completions.

## A.2 Motivation

In this study, we assume that if LLMs can effectively select between global context and local context, the repository-level code completion performance can be significantly boosted. To further verify our assumption, we conduct a preliminary experiment. In detail, we evaluate the performance of utilizing cross-context versus not using it for code completion in the RepoBench datasets (see details about RepoBench in Section 4.1) As shown in Table 5, “In-file” column refers to correct completion using only the context from the current file. While “Both” column denotes correct completions obtained using both in-file and cross-file context. During the process of incorporating cross-files, it introduces context-induced failures (i.e., the red numbers). For example, CodeLlama-7B shows that the number of correct completions increases from 290 to 374 after using cross-file context, but 143 predictions that were originally correct under in-file context turn incorrect when additional cross-file context is included. This reveals a critical challenge: while cross-file context improves overall correctness, it simultaneously destabilizes previously correct predictions. It indicates that LLMs still lack reasoning capabilities for cross-file context and motivates us to explore how to guide LLMs to maximize cross-file benefits while mitigating context-induced failures. The experimental results indicate that while introducing cross-file context improves overall performance, it also harms the model’s original performance in in-file scenarios.

## A.3 Justification of Reward Design

## A.3.1 Format and length-based rewards.

The format and length-based rewards are auxiliary mechanisms required to make RL training welldefined for line-level code completion in the repository. (1) Format reward: The format constraint is a principled modeling choice in GRPO-style RL training. It defines an evaluation protocol so the completion span can be extracted and scored from the model output (e.g., text and code). Without a fixed output format, the reward cannot be computed consistently during training. (2) Lengthbased reward: The length-based reward addresses the reward-assignment problem. Our task aims to provide line-level feedback, yet unconstrained generation often produces multiple lines or large segments of a file. As a result, reward triggered by the first correct line propagates through subsequent tokens, causing reward leakage. We observed substantial over-completion in experiments. Therefore, the length reward localizes the reward to the tokens responsible for the prediction. The strictness is intentional: it prevents reward leakage and leads to more stable training and faster convergence. While simple, it effectively ensures that the reward is assigned to the correct position, rather than to unrelated subsequent lines. In summary, these rewards are important and effective mechanisms to ensure reliable evaluation and proper reward assignment when applying RL to line-level code completion.

## A.3.2 Identifier-Driven Intent Recognition.

Discussion on Identifier Overlap. Our method evaluates correctness based on the specific context of repo-level code completion rather than relying on shallow lexical similarity. We clarify that our task is repo-level code completion, an IDE-style workflow where a developer completes a single line inside an existing file of a repository. In this setting, the developer is not freely producing alternative implementations. The inserted line must reference concrete program entities available in the current scope and consistent with the surrounding control flow and data flow. The identifier appearing in a valid completion is therefore determined by the required binding, such as which object is in scope, which API provides the required return type, and which function is invoked at that program point, rather than by interchangeable naming choice. Replacing the ground-truth identifier with a different one, even if semantically similar in isolation, usually refers to a different object, breaks the intended call chain, or becomes unusable in the given context. The key issue is whether the completion binds to the intended program entity. For this reason, identifier overlap in the intent reward acts as a proxy for correct contextual binding and practical usability, rather than encouraging shallow lexical similarity.

<table><tr><td rowspan="2">model</td><td rowspan="2">In-file</td><td rowspan="2">Java →</td><td colspan="3">Python</td><td rowspan="2">Both</td></tr><tr><td>Both</td><td>In-file</td><td>→</td></tr><tr><td>CodeLlama-7B</td><td>290</td><td>-143l+227</td><td>374</td><td>134</td><td>-63l+100</td><td>174</td></tr><tr><td>StarCoder2-7B</td><td>31</td><td>-143l+192</td><td>80</td><td>55</td><td>-30l+29</td><td>54</td></tr><tr><td>DeepSeek-Coder-6.7B</td><td>686</td><td>-131l+351</td><td>906</td><td>390</td><td>-130l+189</td><td>449</td></tr><tr><td>Qwen2.5-Coder-7B</td><td>631</td><td>-128I+301</td><td>804</td><td>489</td><td>-138l+225</td><td>576</td></tr><tr><td>DeepSeek-Coder-33B</td><td>541</td><td>-150l+410</td><td>801</td><td>373</td><td>-166l+214</td><td>421</td></tr><tr><td>Qwen2.5-Coder-32B</td><td>844</td><td>-99l+307</td><td>1,052</td><td>771</td><td>-126l+237</td><td>882</td></tr></table>

Table 5: Results of Exact Match (EM) Count Code Completion.

Motivation Behind Using the Intent Reward as a Coefficient. In code completion, correctly capturing the developer’s intent is the primary objective, so we use the intent reward as a coefficient of the completion reward rather than an additive term. An additive reward treats intent as an independent bonus and may still assign a high total reward to outputs that are textually similar (e.g., small edit distance) but semantically incorrect. By contrast, using intent as a multiplicative coefficient makes it a gating signal: completions that better match the intended functionality receive proportionally higher reward, while superficially similar but semantically mismatched candidates are downweighted. This is particularly important when two candidates have similar completion scores due to textual similarity, where the intent signal helps distinguish the one whose reasoning path aligns more closely with the expected behavior. From an optimization perspective, the multiplicative form also aligns the gradients of the completion objective with semantic correctness, encouraging the model to prioritize semantically consistent continuations rather than merely minimizing token-level differences.

## A.4 Prompt

As show in Fig. 4, we use this prompt as the LLMs input template.

• Instruction: You are a helpful assistant tasked with   
completing the next line of a repository-level code file.   
You are provided with an incomplete code file and other   
code snippets from the same repository. You first think   
about the reasoning process as an internal monologue   
that infers the user’s intent of completion. Then, you   
should provide the user with the next line of the incom  
plete code.   
Please respond in the following format:   
<think>. . .</think>   
<answer> \`\`\`</answer>   
Repository name:{REPOSITORY \_NAME}   
• Cross-file context C<sub>cross</sub>:   
the path of cross-file context $C _ { 1 }$   
cross-file context $C _ { 1 }$   
the path of cross-file context $C _ { n }$   
cross-file context $C _ { n }$   
• In-file context C<sub>in-file</sub>:   
the path of incomplete code snippet x   
incomplete code snippet x  
Figure 4: the prompt of repository-level completion.

## A.5 Dataset

## A.5.1 Completion Scenarios

We contain three completion scenarios as follows: Cross-File-First (CF-F): In this setting, the position of completion is the first appearance of a cross-file line within a file and ensuring there is no identical or similar cross-file invocation for the model to reference.

Cross-File-Random (CF-R): In this setting, the position of completion is a random and non-first occurrence of a cross-file line.

In-File (IF): In this setting, the position of completion is an in-file line that does not involve any cross-file modules. This setting serves as a robustness test to ensure that the incorporation of crossfile context does not greatly affect the accuracy of predictions.

## A.5.2 Dataset Construction

We collect 32K repository-level code completion examples from the RepoBench-C and RepoBench-P datasets (Liu et al., 2023), with an equal number of Python and Java examples. The original RepoBench dataset supports Python&Java with three tasks (Retrieval/Completion/Pipeline). Our dataset is derived from Completion(RepoBench-C) and Pipeline(RepoBench-P) as they directly correspond to in-file and cross-file scenarios. Specifically, the context fed into our IntentCoder includes three parts: Pre-target-line crossfile context, Pre-targetline infile context, and Post-target-line crossfile noise. The Pre-target-line crossfile context refers to import modules that appear before the target prediction line. The pre-target-line infile context refers to code content from the beginning of the current file up to the target line. The post-target-line crossfile noise refers to import modules that occur after the target line, which are regarded as noise since we can only use context before the target line for code completion task.

For repository-level line-level code completion, cross-file context is obtained by analyzing the classes, functions, identifiers, and other program entities related to the target completion line. Leverage their dependency relationships to locate relevant files and the corresponding definitions or functional implementations from the repository, providing repository-level context for completing the target line.

For the test dataset, We use Repobench-Java and Repobench-Python (Liu et al., 2023), each containing 3,000 samples, with 1,000 samples for each of the CF-F, CF-R, and IF settings. Regarding context size, following the RepoBnech, Our dataset is designed for LLMs with a token limit of 2048. The average context size fed into the model is 1346 tokens for training, 1399 tokens for evaluation, and 1121 tokens for testing.

## A.6 Baselines

We consider the following open-source CodeLLMs for comparison, which are widely used in code completion tasks. To simplify the model names, we have removed the “instruct” suffix in the following introductions (e.g., Qwen2.5-coder, CodeLlama, Deepseek-Coder):

StarCoder2 (Lozhkov et al., 2024): The StarCoder2-15B is a 15 billion parameter model trained on 600+ programming languages using 4.3 trillion tokens, while the StarCoder2-7B is a 7 billion parameter model trained on 17 programming languages with 3.7 trillion tokens.

DeepSeek-Coder (Guo et al., 2025): Deepseek-Coder-6.7B is a 6.7 billion parameter model initialized from Deepseek-Coder-6.7b-base and finetuned on 2B tokens of instruction data. DeepSeek-Coder-33B is the largest variant of the DeepSeek-Coder series.

CodeLlama (Roziere et al., 2023): CodeLlama-7B and CodeLlama-34B are designed to be safer for use in code assistants and generation applications, with a focus on general code synthesis, understanding, and instruction-following.

Qwen2.5-Coder (Hui et al., 2024): Qwen2.5- Coder-7B and Qwen2.5-Coder-33B have performed in code generation, reasoning, and fixing, with the latter scaling up training tokens to 5.5 trillion. Qwen2.5-Coder-33B is the SOTA opensource CodeLLM, offering coding abilities comparable to GPT-4o and providing a more comprehensive foundation for real-world applications.

aiXcoder-7B-v2 (Li et al., 2025): aiXcoder-7Bv2 is currently the SOTA 7B model for repositorylevel code completion and uses the CoLT method to fine-tune the model to better utilize information within long contexts for code completion tasks.

Then we compare our method with the following methods in the training model for repository-level code completion:

fine-tuning (Hu et al., 2022): A supervised finetuning (SFT) method where pre-trained models are further trained on a specific task or dataset, refining the model’s performance for the target application. We use Low-Rank Adaptation (LoRA) as the specific method.

GRPO (Guo et al., 2025): Group Relative Policy Optimization (GRPO), which is a method of reinforcement learning. We utilizes reinforcement learning by exact match as a reward to improve code prediction accuracy for repository-level code completion.

CoLT (Li et al., 2025): CoLT is a novel training approach for repo-level code completion. It first performs supervised fine-tuning (SFT). After that, preference data is constructed by combining ground truth and model-generated data. Finally, the model is optimized using Direct Preference Optimization (DPO) (Rafailov et al., 2023).

Training Setup. In our experiment, we carefully control the training setup to make the experimental comparison as fair as possible. For the RL method, GRPO and our method use the same training configuration (e.g., the number of epochs, batch size, and rollout size for each prompt). CoLT also uses the same training configuration. For Preference Data Construction, we adopt the same count as the rollout size for each prompt to maintain the same data size. For supervised fine-tuning, we ensure the same overall amount of training data is observed by adjusting the number of epochs, since the RL method samples multiple answers for each prompt. The model is trained until convergence, and we report the best checkpoint according to validation performance.

## A.7 Implementation Details

The rank and alpha values of LoRA (Hu et al., 2022) are set to 16 and 32, respectively. We use a learning rate of $1 \times 1 0 ^ { - 4 }$ with a linear scheduler and warm-up, with the warm-up ratio set to 0.1. The model was trained for 1 epoch, consisting of 500 steps, with a batch size of 192. The model π<sub>θ</sub> is trained on 4 NVIDIA A800 (80GB) GPUs for training and 1 GPU for inference simultaneously, which takes 10 hours. The hyperparameters $\mu , \epsilon ,$ and $\beta$ are set to 1, 0.2, and $1 \times 1 0 ^ { - 3 }$ , respectively. The hyperparameters for reward weights, $\alpha _ { i c } , \alpha _ { f } ,$ and $\alpha _ { L }$ , are set to 1.0, 0.2, and 0.1, respectively. We provide well-structured code interface, where reward functions can easily be configured and used. Other Details about the implementation are provided in our released code.

During the inference stage, the backbone models we use (e.g., Qwen2.5-Coder and DeepSeek-6.7B-Coder) support substantially longer context windows (up to 128K/16K tokens). Since RepoBench’s maximum input tokens are 2048, we calculate the runtime latency based on this setting. However, if the cross-file context exceeds 2048 tokens, we can still feed larger contexts to the model and perform reasoning without being constrained by this RepoBench-specific limit.

## A.8 Evaluation Metrics

Following the previous work (Liu et al., 2023; Eghbali and Pradel, 2024; Zhang et al., 2023; Jiang et al., 2024b), we report performance in two main categories to evaluate the performance of codeLLMs:

Code Match: To evaluate the accuracy of the predicted code lines, we use Exact Match (EM) and Edit Similarity (ES) (Levenshtein et al., 1966), which respectively represent the proportion of exact matches and the degree of similarity to the ground truth.

Identifier Match (Ding et al., 2022, 2023): To evaluate the model’s ability to understand the completion intent and predict the correct application programming interfaces (APIs), we use the following metrics:(1) Exact Match (EM): EM represents the proportion of exact matches to the ground truths in code identifiers. (2) Precision: Precision measures the proportion of correctly predicted code identifiers out of all predicted identifiers. (3) Recall: Recall measures the proportion of correctly predicted code identifiers out of all actual code identifiers in the ground truth. (4) F1: F1 is the harmonic mean of precision and recall metrics, providing a balanced metric for the model’s performance.

## A.9 Method Analysis and Generalization Experiments

## A.9.1 Method Analysis

We clarify why our reward method is more appropriate for repo-level code completion task than other methods as follows:

(1) Execution-based reward can provide a strong reward signal, but in repo-level completion, they are difficult to obtain reliably. Our task involves line-level completion in code repositories, where the suffix code context is often missing, so independent compilation is infeasible. Constructing executable environments at scale requires complex dependencies and specific library versions, and building and testing large projects can take minutes or hours, which is impractical for RL training. Thus, while execution-based rewards are valuable, they are not scalable for large cross-file completion datasets. Our rewards are based on edit similarity and identifier matching, which is a more flexible and scalable design. The reward is lightweight and can be applied efficiently across large datasets, ensuring practical use even when execution or validation is unavailable. In our task, such signals are difficult to obtain reliably: (1) Our task focuses on line-level completion over incomplete files extracted from repositories. These samples typically lack surrounding complete code context or project context (e.g., dependent modules compile ) and therefore cannot be compiled independently, making compiler feedback unstable. (2) Lightweight abstract execution could be applied to a subset of cases, which would substantially reduce coverage and introduce bias toward easily executable examples. For these reasons, we adopt structured static signals (e.g., edit similarity and identifier matching) as a scalable proxy for correctness.

<table><tr><td rowspan="2">Model</td><td colspan="3">BM25</td><td colspan="3">OpenAI text-embedding-ada</td></tr><tr><td>Code EM</td><td>ES</td><td>Identifier EM</td><td>Code EM</td><td>ES</td><td>Identifier EM</td></tr><tr><td>Qwen2.5-Coder-7B</td><td>10.33</td><td>57.93</td><td>18.75</td><td>15.71</td><td>64.66</td><td>26.13</td></tr><tr><td>Qwen2.5-Coder-7Bfine-tuning</td><td>11.87</td><td>58.09</td><td>19.35</td><td>20.35</td><td>67.80</td><td>27.21</td></tr><tr><td>Qwen2.5-Coder-7BGRPO</td><td>13.18</td><td>60.36</td><td>20.06</td><td>22.02</td><td>68.33</td><td>29.41</td></tr><tr><td>Qwen2.5-Coder-7BCoLT</td><td>12.90</td><td>59.72</td><td>19.54</td><td>23.18</td><td>69.72</td><td>30.67</td></tr><tr><td>IntentCoderQwen</td><td>15.33</td><td>62.46</td><td>22.39</td><td>26.79</td><td>73.43</td><td>38.62</td></tr></table>

Table 6: Evaluation results on CrossCodeEval-Java.

![](images/98e03d9c9ad86bfacaa6bfe0e09762e1a9464f275b3970dae905b223db2d216f.jpg)

Figure 5: The case study for IntentCodercompares it with other Code LLMs.
<table><tr><td rowspan="2">Model</td><td colspan="4">RepoBench-Java</td><td colspan="4">RepoBench-Python</td></tr><tr><td>Code Match</td><td></td><td></td><td>Identifier Match</td><td></td><td>Code Match</td><td></td><td>Identifier Match</td></tr><tr><td></td><td>EM</td><td>ES</td><td>EM Precision Recall</td><td></td><td>F1 EM</td><td>ES</td><td>EM Precision Recall</td><td>F1</td></tr><tr><td>IntentCoderQwen</td><td></td><td>42.47 77.52 48.57</td><td></td><td>69.95</td><td>69.60 68.7535.93 71.58 42.70</td><td></td><td>65.57</td><td>63.80 63.37</td></tr><tr><td>w/o Identifier-Driven Intent Recognition 40.27</td><td></td><td>76.77</td><td>46.47 68.73</td><td></td><td></td><td>68.62 67.6734.60 69.98 41.27</td><td>63.74</td><td>63.10 62.05</td></tr><tr><td>w/o Reward-driven Completion Learning 40.80</td><td></td><td>76.94</td><td>47.67 69.56</td><td></td><td>68.4968.0334.3068.97</td><td></td><td>40.20 62.24</td><td>60.87 60.25</td></tr><tr><td>w/o Completion-focused RL</td><td>40.9076.13</td><td></td><td>47.07 68.49</td><td>67.6266.9534.4767.7340.70</td><td></td><td></td><td>62.06</td><td>59.61 59.45</td></tr><tr><td>w/o Selective-Exploration Optimization</td><td>40.87 76.7547.07</td><td></td><td>68.92</td><td>68.41 67.6934.4770.4340.97</td><td></td><td></td><td>64.58</td><td>62.99 62.39</td></tr><tr><td>w/o ALL</td><td>26.80 62.77 32.70</td><td></td><td>54.74</td><td>57.06 53.81 19.9752.3925.20</td><td></td><td></td><td>46.55</td><td>52.45 46.51</td></tr></table>

Table 7: Results of Ablation study on the RepoBench-java and RepoBench-Python dataset. All results in the table are reported in percentage (%).

(2) SFT reward: Cross-file completion requires reasoning about how an API is used in a concrete program context. A model trained only with retrieval supervision may learn to associate related files rather than learning to generate correct completions via cross-file reasoning, which may bias the model toward lexical similarity rather than deeper semantic correctness. Our method allows the model to learn such reasoning through reinforcement learning by interacting with the repolevel code-completion environment.

(3) Preference-based learning reward:

Preference-based learning methods (e.g., DPO) optimize the model using fixed positive–negative pairs. However, in our setting it is difficult for the model to consistently produce fully correct outputs during self-improvement, and relying on ground-truth pairs alone introduces an out-of-distribution gap between training signals and the model’s own generations. Our reward design instead provides graded feedback: multiple candidate completions are compared and rewarded according to how well they align with program behavior. This allows the model to improve even when none of the candidates is perfectly correct, offering a more stable learning signal than strict binary preference pairs and enabling incremental improvement during training.

The experimental results of fine-tuning and CoLT, when compared with our approach, demonstrate that our approach better facilitates the learning of reasoning abilities in the code repository environment. Moreover, the group data generated through model self-distribution can stably and effectively optimize the model’s output distribution, thereby obtaining stable and effective learning signals while avoiding unstable signals from ground truth code completion that fall outside the distribution.

## A.9.2 Generalization Experiment.

We provide further analysis of how the model performs when the retrieved cross-file context contains irrelevant or distracting information outside the RepoBench distribution. Specifically, we experimented on the CrossCodeEval-Java (Ding et al., 2023) dataset using different retrievers (BM25 (Robertson and Zaragoza, 2009) and OpenAI text-embedding-ada-002). CrossCodeEval-Java has 2,139 samples and retrieves the top-5 code snippets for each query. This naturally introduces incomplete and potentially distracting context. Table 6 shows the experimental results of our model and other methods. The results show two trends. (1) Under the same retriever, training with our method substantially improves performance (Code EM increases from 10.33%to 15.33%), indicating that the model learns to better identify relevant information within a noisy retrieval cross-files context. (2) When retrieval quality improves (e.g., using a more accurate embedding retriever), performance further increases (Code EM increases from 15.33% to 26.79%), suggesting that the method can effectively utilize cleaner context even in retrieval noise. These observations indicate that IntentCoder is not dependent on a particular repository distribution and remains effective under varying levels of irrelevant or incomplete cross-file context.

## A.10 Case Study

We illustrate the effectiveness of Intent-Coder through a case study presented in Fig. 5, In this case, IntentCoder correctly identifies the user’s intent, which is to set a custom renderer for the table corner header (e.g., MultiLineTableHeaderRenderer). In contrast, many other models (e.g., DeepSeek-Coder, Qwen2.5-Coder, CodeLlama-7B, Starcoder2-7B, aixcoder-7B-v2), mistakenly think the next step should be related to the corner setup. Additionally, due to the lack of intent recognition’s capability, many other models overly focus on the in-file context (e.g., colored in blue), such as FrozenColumnTable, setGridColor, getColumn. At the same time, these models lack cross-file reasoning ability and incorrectly invoke classes from other files (e.g., colored in purple), failing to understand that the appropriate multi-line header renderer should be used. By recognizing the overall functional requirements, IntentCoder accurately provides the correct code completion, avoiding the shortcomings of other models in completion intent recognition and cross-file reasoning.