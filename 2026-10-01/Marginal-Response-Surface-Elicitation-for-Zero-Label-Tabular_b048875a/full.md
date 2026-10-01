# Marginal Response Surface Elicitation for Zero-Label Tabular Learning

Fudan Institute on Networking Systems of AI

Tabular learning uses structured data to predict target outcomes. Traditionally, this process has relied on labeled data. However, large language models (LLMs) can be used to elicit domain priors based on the task description and feature semantics, thereby enabling predictions without labeled data. We propose Marginal Response Surface Elicitation (MARS), a method that transforms feature-level LLM priors into a reusable, zero-shot tabular classifier. To construct this classifier, MARS selects representative values for each feature from unlabeled data and prompts the LLM to provide corresponding class support scores and feature weights. It then aggregates multiple responses using the median to construct feature response functions, and makes predictions through their weighted sum without further LLM queries. Across eight tabular benchmark tasks, MARS achieves the highest average AUC and AP, outperforming direct prompting by 1.97 and 6.21 percentage points respectively, while substantially reducing end-to-end costs. Evaluations with LLMs of different sizes further demonstrate its predictive advantage over direct prompting.

Keywords: Large language models, tabular learning, zero-label prediction

## 1. Introduction

Tabular learning utilizes structured information recorded in tables to predict target outcomes, finding broad applications across customer analytics, credit assessment, and business operations (Borisov et al., 2024; Hollmann et al., 2025; Gardner et al., 2024). Traditional prediction pipelines rely heavily on labeled samples, using supervised training to learn the mapping relationships between features and labels (Grinsztajn et al., 2022; Gorishniy et al., 2025; Erickson et al., 2025). However, in many real-world business scenarios, the need for predictions often arises before actual business labels are available. For example, when a company launches a new product, it already has access to certain customer attributes and account information but needs to determine which customers to contact first before subscription results are available. This has given rise to the need for zero-label tabular prediction—that is, building a predictor using unlabeled records when the prediction target and feature meanings are known.

The knowledge accumulated by large language models (LLMs) during pretraining can be combined with task objectives to provide priors for label-scarce tabular prediction (Bordt et al., 2024; Capstick et al., 2025). These priors are injected into the prediction pipeline in different ways: TabLLM (Hegselmann et al., 2023) converts tabular records into text for direct prediction, CLLM (Seedat et al., 2024) and SERSAL (Yan et al., 2025) generate synthetic data and pseudo-labels to train the model, FeatLLM (Han et al., 2024) generates rules and converts them into features to train a simple model, and LLM-Trees (Knauer et al., 2025) and ProtoLLM (Wang et al., 2025) generate decision trees and prototypes to make predictions.

(c) Value of pairings

These methods raise a more fundamental question: What information can be leveraged in a few-shot setting? We distinguish between two types of information: ❶ the relationship between the value of a single feature and the class, and ❷ the additional information provided by cross-feature couplings. To guide which type of information to elicit from LLMs for zero-label prediction, we conduct a within-class column-wise permutation experiment (Ojala and Garriga, 2010) on real-world datasets with labels. As illustrated in Figure 1, we independently permute the values of each feature within each class, thereby preserving the marginal distribution of each feature within each class, but disrupting the original cross-feature pairings. We then train identical CatBoost models separately on the original and permuted data, and compare their performance on the same test set. We find that the average performance difference is close to zero for 4–16 training samples, suggesting that the model gains little additional predictive benefit from cross-feature couplings in this regime. This suggests prioritizing the relationship between individual feature values and the class when eliciting LLM priors, as (a) Within-class permutation experiment

![](images/8c127f32fc4cef4319534e36a62940296ba6ee14c0ef9e48e9476f1d394a8200.jpg)

(b) Value of data  
![](images/122984c9992236b1bbbb31d4ce8599bdfc97cf966dc188ba619c9b57de1dddb2.jpg)

![](images/28a1bf1160d2bada2d4a1ece517b9e1b20acee38fae8472245f2107b8c10ba82.jpg)  
Figure 1: Cross-feature couplings in few-shot learning. (a) Within-class column permutation preserves feature–class associations while disrupting original pairings. (b) CatBoost performance with increasing training samples. (c) Gains from retaining original pairings (original - shuffled, ×100). Shaded regions indicate ±1 standard error across 16 seeds.

priors on cross-feature relationships may be noisy in the absence of labels.

Based on this intuition, we propose Marginal Response Surface Elicitation (MARS), a method that transforms the domain priors of LLMs into a reusable zero-label predictor. MARS selects representative values for each feature from unlabeled data and, combining task descriptions with feature semantics, obtains class support scores for each value and weights for each feature. Subsequently, it constructs feature response functions through median aggregation and employs their weighted sum for prediction. The entire process requires no ground-truth labels or local model training, and predicting new samples does not require calling the LLM again.

Comprehensive experiments across tabular benchmarks and LLMs of different sizes show that MARS is ❶ powerful: MARS achieves the best average AUC and AP among all baselines, including direct LLM prompting; ❷ efficient: MARS does not require any LLM API calls during inference, with end-to-end API call costs amounting to only 9.6% of ProtoLLM; ❸ transferable: MARS achieves better average AUC and AP than direct prompting on both Qwen3.5-4B and Qwen3.5-9B. In summary, our contributions are as follows:

• Mechanistic Analysis: The average benefit of within-class column permutation is close to zero in the few-shot regime, providing direct empirical support for the hypothesis that the LLM prior is informative at the feature level, but not at the instance level.

• Method Design: We propose MARS, which converts feature-level LLM priors into response functions through median aggregation and combines them via weighted summation into a reusable predictor requiring no ground-truth labels, local model training, or test-time LLM queries.

• Experimental Evaluation: On eight benchmark tasks, MARS achieves 1.97/6.21 higher average AUC/AP than direct prompting while largely reducing end-to-end costs compared to other high-performing baselines; cross-model experiments further demonstrate the transferability of our method.

## 2. Related Work

Few-Shot Tabular Learning. Few-shot tabular learning aims to achieve strong predictive performance with limited labeled data. To alleviate label scarcity, pretraining methods learn transferable feature representations from unlabeled data and adapt them to target tasks using a small number of labeled examples (Yoon et al., 2020; Bahri et al., 2022; Nam et al., 2023). Cross-task pretraining instead learns general predictive capabilities across large collections of tabular tasks and uses a few labeled examples as context to predict on new tasks (Hollmann et al., 2025; Grinsztajn et al., 2026; Qu et al., 2026). These approaches reduce target-task annotation requirements through pretraining, whereas we investigate how to leverage LLM domain knowledge to construct predictors without target-task labels.

LLM-Based Tabular Learning. LLMs can leverage task descriptions and feature semantics to apply domain knowledge acquired during pretraining to tabular prediction. Direct-inference methods represent tabular records as text and use LLMs to generate class predictions (Hegselmann et al., 2023; Slack and Singh, 2023; Gardner et al., 2024). Knowledge-transfer methods instead convert LLM knowledge into training data or feature representations for learning downstream predictors (Seedat et al., 2024; Yan et al., 2025; Han et al., 2024; Shi et al., 2025). Another line of work uses LLMs to construct or refine predictors, encoding prior knowledge as decision rules or class prototypes for subsequent local prediction (Knauer et al., 2025; Ye et al., 2025; Wang et al., 2025). MARS transforms LLM domain knowledge into feature response functions and their weights, combining their weighted outputs into a reusable zero-label predictor. It requires neither downstream model training nor test-time LLM queries.

## 3. Method

Figure 2 illustrates the overall pipeline of MARS. MARS first selects representative values for each feature from unlabeled reference data (Section 3.1), and then prompts the LLM with the task description and feature semantics to obtain a support score for each value and a feature weight (Section 3.2). Subsequently, MARS aggregates multiple responses to construct reusable feature response functions. When predicting new samples, it computes each feature’s response via numerical interpolation or categorical lookup tables, and then aggregates the scores using the feature weights to obtain a final prediction score, without querying the LLM (Section 3.3).

## 3.1. Representative Value Selection

Let U be the unlabeled reference table with $p$ features. For each feature $j ,$ MARS selects $m _ { j }$ representative values $a _ { j 1 } , \dotsc , a _ { j m _ { j } }$ as anchors, queries the LLM for their scores, and uses the scores to build the feature’s response function.

Tabular features are either numerical or categorical. For a numerical feature, MARS computes the empirical quantiles at {0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95} using the non-missing values in U and removes duplicates to obtain at most seven anchors. For a categorical feature, MARS stores up to 20 of the most frequent values and their frequencies. The feature name, type, and anchors are then sent to the LLM together with the task description.

![](images/5a61149c34b1a1b0bd41465497c659dfa67fc99bc3b8b0f7413b45723d2787a8.jpg)  
Figure 2: Overview of MARS. MARS aggregates five LLM responses by the median and caches the resulting feature response functions and weights. New samples are scored by evaluating these functions at their feature values and summing the weighted responses.

## 3.2. Marginal Response Elicitation

MARS combines the task and feature descriptions into a single prompt and formulates the binary classification task as a yes/no question where “yes” represents the positive class. We query the LLM $R = 5$ times using the same prompt, covering all features each time, and ask the LLM to first explain the meaning of the positive class and provide typical examples in 1–2 sentences before scoring each feature.

In the r-th response, the LLM provides a support score $z _ { j \ell } ^ { ( r ) } \in [ - 3 , 3 ]$ for each anchor $a _ { j \ell }$ of feature $j .$ Positive, negative, and zero values represent support for the positive class, negative class, and neutrality, respectively, while the absolute value indicates support strength. The prompt requires that all features use a uniform log-odds scale, where +1 and −1 approximately correspond to multiplying and dividing the odds of the positive class by $e ,$ respectively.

The LLM also assigns an importance score $b _ { j } ^ { ( r ) } \in [ 0 , 1 0 ]$ to each feature, which measures its importance in distinguishing classes relative to other features; a score of 0 indicates that the feature is not important. Unlike the support scores assigned to individual values, this score is uniform across the entire feature and is subsequently used to calculate the global weight.

We request the LLM to return the importance scores of each feature and the support scores arranged in the order of the anchors in JSON format. The complete prompt is provided in Appendix A.

## 3.3. Response Aggregation and Prediction

Let $\mathcal { V } _ { j }$ be the set of response indices for feature $j .$ MARS then aggregates the anchor support scores and feature importance scores using the median, to obtain the final anchor support score

Table 1: AUC/AP (↑) on eight tabular benchmarks. Best scores are bold; † denotes test-time LLM queries.
<table><tr><td>Method</td><td>Labels</td><td>Bank</td><td>Blood</td><td>Credit-G</td><td>Diabetes</td><td>Heart</td><td>Cultivars</td><td>Myocardial</td><td>NHANES</td><td> $\mathbf { A v g . }$ </td></tr><tr><td>LogReg</td><td>4</td><td>.526/.162</td><td>.282/.168</td><td>.601/.796</td><td>.545/.432</td><td>.508/.553</td><td>.584/.610</td><td>.486/.230</td><td>.544/.165</td><td>.509/.389</td></tr><tr><td>XGBoost</td><td>4</td><td>.500/.117</td><td>.500/.240</td><td>.500/.700</td><td>.500/.351</td><td>.500/.554</td><td>.500/.500</td><td>.500/.225</td><td>.500/.160</td><td>.500/.356</td></tr><tr><td>CatBoost</td><td>4</td><td>.562/.166</td><td>.364/.198</td><td>.518/.729</td><td>.411/.355</td><td>.565/.636</td><td>.638/.619</td><td>.469/.257</td><td>.569/.208</td><td>.512/.396</td></tr><tr><td>STUNT</td><td>4</td><td>.509/.123</td><td>.427/.229</td><td>.461/.690</td><td>.731/.591</td><td>.908/.921</td><td>.480/.589</td><td>.579/.309</td><td>.425/.132</td><td>.565/.448</td></tr><tr><td>TabPFN-3</td><td>4</td><td>.564/.169</td><td>.402/.219</td><td>.611/.793</td><td>.484/.360</td><td>.703/.693</td><td>.633/.597</td><td>.487/.285</td><td>.531/.177</td><td>.552/.412</td></tr><tr><td>TabICLv2</td><td>4</td><td>.633/.197</td><td>.382/.217</td><td>.546/.744</td><td>.560/.411</td><td>.750/.763</td><td>.642/.623</td><td>.523/.280</td><td>.530/.190</td><td>.571/.428</td></tr><tr><td>TabFM</td><td>4</td><td>.396/.092</td><td>.437/.239</td><td>.576/.750</td><td>.579/.371</td><td>.506/.556</td><td>.576/.568</td><td>.417/.189</td><td>.617/.233</td><td>.513/.375</td></tr><tr><td>DeLTa</td><td>4</td><td>.703/.252</td><td>.357/.203</td><td>.395/.624</td><td>.552/.414</td><td>.509/.574</td><td>.534/.570</td><td>.477/.218</td><td>.452/.155</td><td>.498/.376</td></tr><tr><td>FeatLLM</td><td>4</td><td>.667/.226</td><td>.411/.255</td><td>.472/.693</td><td>.760/.583</td><td>.839/.861</td><td>.662/.621</td><td>.608/.334</td><td>.585/.242</td><td>.626/.477</td></tr><tr><td>Direct†</td><td>0</td><td>.841/.463</td><td>.683/.353</td><td>.596/.773</td><td>.808/.693</td><td>.894/.885</td><td>.593/.586</td><td>.666/.320</td><td>.690/.303</td><td>.721/.547</td></tr><tr><td>TabuLa-8B†</td><td>0</td><td>.808/.417</td><td>.590/.310</td><td>.419/.673</td><td>.717/.612</td><td>.767/.760</td><td>.443/.466</td><td>.623/.372</td><td>.444/.148</td><td>.601/.470</td></tr><tr><td>LLM-Trees</td><td>0</td><td>.580/.213</td><td>.612/.302</td><td>.593/.744</td><td>.733/.560</td><td>.802/.766</td><td>.537/.524</td><td>.593/.283</td><td>.598/.213</td><td>.631/.451</td></tr><tr><td>ProtoLLM</td><td>0</td><td>.820/.392</td><td>.725/.448</td><td>.660/.818</td><td>.870/.757</td><td>.881/.904</td><td>.598/.684</td><td>.594/.318</td><td>.687/.261</td><td>.729/.573</td></tr><tr><td>MARS (ours)</td><td>0</td><td>.884/.630</td><td>.693/.427</td><td>.610/.790</td><td>.858/.757</td><td>.927/.940</td><td>.586/.604</td><td>.673/.443</td><td>.698/.283</td><td>.741/.609</td></tr></table>

$\bar { z } _ { j \ell }$ and the final feature weight $w _ { j } { \mathrm { : } }$

$$
\bar { z } _ { j \ell } = \mathop { \mathrm { m e d i a n } } _ { r \in \mathscr { V } _ { j } } \mathrm { c l i p } _ { [ - 3 , 3 ] } \big ( z _ { j \ell } ^ { ( r ) } \big ) ,
$$

$$
w _ { j } = \frac { 1 } { 1 0 } \operatorname* { m a x } \left( 0 , \underset { r \in \mathscr { V } _ { j } } { \operatorname { m e d i a n } } b _ { j } ^ { ( r ) } \right) .\tag{1}
$$

The median is robust to outliers. If there are no responses for a feature, it is not used in the prediction.

MARS then constructs the feature response function $f _ { j }$ using the anchors and the aggregated support scores. For numerical features, $f _ { j }$ is a piecewise linear function defined as

$$
f _ { j } ( v ) = \bar { z } _ { j \ell } + \frac { v - a _ { j \ell } } { a _ { j , \ell + 1 } - a _ { j \ell } } \left( \bar { z } _ { j , \ell + 1 } - \bar { z } _ { j \ell } \right) ,\tag{2}
$$

for $a _ { j \ell } \leq v \leq a _ { j , \ell + 1 }$ . For values outside the range of the anchors, the function is extended using the response at the closest anchor. For categorical features, $f _ { j }$ is a lookup table defined as

$$
f _ { j } ( v ) = \left\{ \begin{array} { l l } { \bar { z } _ { j \ell } , } & { v = a _ { j \ell } , \quad \ell = 1 , \ldots , m _ { j } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

Given a new instance $\boldsymbol { x } = \left( x _ { 1 } , \ldots , x _ { p } \right)$ , the MARS score is computed as

$$
s ( x ) = \sum _ { j : \nu _ { j } \neq \emptyset } w _ { j } f _ { j } ( x _ { j } ) .\tag{4}
$$

A higher score indicates stronger support for the positive class. The resulting model is a nonlinear additive model, where each feature is modeled using a nonlinear function, and the nonlinear functions are combined additively. Note that the model is constructed using a finite number of queries to the LLM, and can be used to score new instances without further querying the LLM.

## 4. Experiments

## 4.1. Experimental Setup

Tasks and Benchmarks. Following FeatLLM (Han et al., 2024) and ProtoLLM (Wang et al., 2025), we evaluate MARS on eight tabular benchmark tasks covering financial services (Bank (Moro et al., 2011), Credit-G (Hofmann, 1994)), healthcare (Blood (Yeh et al., 2009), Diabetes (Smith et al., 1988), Heart (fedesoriano, 2021), Myocardial (Golovenkin et al., 2020), NHANES (NA, 2019)), and agriculture (Cultivars (de Oliveira et al., 2023)). The evaluation metrics are ROC-AUC (AUC) and average precision (AP).

Baselines. We compare with three groups of baselines: (1) supervised learning, including LogReg, XGBoost (Chen and Guestrin, 2016), and CatBoost; (2) few-shot tabular learning, including STUNT (Nam et al., 2023), TabPFN-3 (Grinsztajn et al., 2026), TabICLv2 (Qu et al., 2026), TabFM (Google Research, 2026), DeLTa (Ye et al., 2025), and FeatLLM (Han et al., 2024); (3) zero-label prediction, including Direct (direct LLM prompting), TabuLa-8B (Gardner et al., 2024), LLM-Trees (Knauer et al., 2025), and ProtoLLM (Wang et al., 2025). The first two groups of baselines are provided with 4 labeled examples (2 per class) for both training and model selection. The third group of baselines, as well as MARS, do not use any task-specific labels for prediction.

Implementation Details. In the main experiments, we use DeepSeek-V4-Flash-0731 with high-intensity thinking as the backbone LLM. We also use Qwen3.5-4B and Qwen3.5-9B to compare MARS with Direct, with native thinking, temperature=1.0, top-p=0.95, top-k=20, presence penalty=1.5. For a fair comparison, MARS uses the same unlabeled reference data as ProtoLLM, with R = 5 queries executed per task. Appendices B and C provide the fixed data protocol and implementation settings.

## 4.2. Predictive Performance

As shown in Table 1, MARS achieves the best average AUC and AP across all eight tasks, outperforming Direct by 1.97 and 6.21 percentage points, and the strongest baseline ProtoLLM by 1.17 and 3.64 percentage points, respectively. Across both metrics, MARS surpasses Direct on seven tasks each, demonstrating that these gains are not confined to isolated tasks. Table 2 further presents the results on backbones of different scales: on Qwen3.5-4B and Qwen3.5-9B, the average AUC/AP of MARS outperforms Direct with the corresponding backbones by 9.69/8.91 and 3.31/3.54 percentage points, respectively, indicating that feature-level prior elicitation sustains its predictive advantage across both

Table 2: Eight-task ablation and backbone averages. Changes (pp) are relative to Direct with the same backbone. Best scores per backbone are bold.
<table><tr><td>Method</td><td>Avg. AUC ↑</td><td>Avg. AP ↑</td></tr><tr><td colspan="3">Q DeepSeek-V4-Flash-0731</td></tr><tr><td>Direct</td><td>.7213</td><td>.5471</td></tr><tr><td>MARS</td><td>.7411↑1.97</td><td>.6091↑6.21</td></tr><tr><td>w/o feature weights</td><td>.7363↑1.50</td><td>.5966↑4.95</td></tr><tr><td>w/o response aggregation</td><td>.7358↑1.45</td><td>.6002↑5.32</td></tr><tr><td>w/o support strength</td><td>.7094↓1.19</td><td>.5554↑0.83</td></tr><tr><td colspan="3">女 Qwen3.5-4B</td></tr><tr><td>Direct</td><td>.5745</td><td>.4366</td></tr><tr><td>MARS</td><td>.6713↑9.69</td><td>.5257↑8.91</td></tr><tr><td colspan="3">女 Qwen3.5-9B</td></tr><tr><td>Direct</td><td>.6364</td><td>.4794</td></tr><tr><td>MARS</td><td>.6695↑3.31</td><td>.5148↑3.54</td></tr></table>

## 4.3. Efficiency and Sensitivity Analysis

Figure 3(a,b) compares the average predictive performance and API costs of different methods across the eight tasks. We account for actual token consumption based on the official DeepSeek API peak-hour pricing. The average task cost of MARS is 0.564 yuan, amounting to only 9.6% of ProtoLLM and lower than Direct’s 1.065 yuan, while achieving higher average AUC and AP. The LLM queries of MARS are concentrated exclusively in the predictor construction stage; once constructed, new samples can be processed via local computation without additional LLM calls.

![](images/3b123fbb1c215cd0730d880d928a60b692c9272d45a413f7d72db546a85c766e.jpg)  
Figure 3: Cost–performance trade-offs and response sensitivity. (a,b) Mean cost and performance; upper left is better. (c) AUC/AP gains over one LLM response (R = 1).

Figure 3(c) further analyzes the impact of the number of responses R. For each R, we evaluate all subsets of size R from the existing five responses and report the average performance. Increasing the number of responses from 1 to 5 increases the mean AUC and AP by 0.53 and 0.89 percentage points, respectively, with diminishing gains for larger R.

## 4.4. Ablation Study

Table 2 examines the roles of feature weights, multi-response aggregation, and support strength by uniformly setting feature weights w<sub>j</sub> to 1, using only a single response (R = 1), and replacing aggregated anchor scores with their signs (−1/0/ + 1), respectively. In the support-strength ablation, sign replacement is applied before interpolation or lookup, with feature weights unchanged. Compared to the full MARS, all three settings degrade the average AUC and AP. Removing support strength causes the most pronounced drops, demonstrating that the degree of support conveys predictive information beyond the direction of support.

## 4.5. Case Study

Figure 4 illustrates how MARS translates LLM domain priors into feature-level predictive contributions. The same response function assigns support scores of 1.70 and 0.07 to call durations of 633 and 206 seconds, respectively, capturing how specific values affect positive-class support strength. The shared previous outcome, unknown, contributes the same weighted value of −0.18 to both records. These positive and negative contributions combine with those of the other features to produce different final scores. By modeling the direction and strength of value-specific support and combining responses with feature weights, MARS turns the LLM's sema contributions. contributions.

![](images/c801f035e264b5095f42e61cb941ab3735d4ca4c871e1d0e6609283d160bae48.jpg)  
Figure 4: Case study on Bank. Two records are scored using the same cached response functions and feature weights, without further LLM queries.

ture weights, MARS turns the LLM’s semantic judgments into decomposable predictive

## 5. Conclusion

We propose MARS for zero-label tabular prediction, transforming feature-level LLM priors into response functions and combining them through weighted summation to construct a reusable predictor. Experiments on eight benchmark tasks show that MARS achieves the highest average AUC and AP among the compared methods while remaining cost-efficient. Future work will explore how to reliably incorporate cross-feature interactions to further improve predictive performance.

## Authors and Affiliations

Liangyu Teng<sup>1,2</sup>, Yicheng Ding<sup>1</sup>, Jing Liu<sup>3</sup>, Hengsong Liu<sup>1,2</sup>, Juncen Guo<sup>1,2</sup>, Hongru Li<sup>1,2</sup>, Jingyu Zhang<sup>1,2</sup>, Liang Song<sup>1,2</sup>\*

<sup>1</sup> College of Intelligent Robotics and Advanced Manufacturing, Fudan University, Shanghai, China

<sup>2</sup> International Networking Systems of Artificial Intelligence Limited, Macau

3 College of Future Information Technology, Fudan University, Shanghai, China

## References

D. Bahri, H. Jiang, Y. Tay, and D. Metzler. SCARF: Self-supervised contrastive learning using random feature corruption. In ICLR, 2022. URL https://openreview.net/pdf?id=CuV\_qYkm Kb3.

S. Bordt, H. Nori, V. Rodrigues, B. Nushi, and R. Caruana. Elephants never forget: Memorization and learning of tabular data in large language models. In COLM, 2024. URL https: //openreview.net/forum?id=HLoWN6m4fS.

V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, and G. Kasneci. Deep neural networks and tabular data: A survey. IEEE TNNLS, 35(6):7499–7519, 2024. doi: 10.1109/TNNLS.2022.3 229161.

A. Capstick, R. Krishnan, and P. Barnaghi. AutoElicit: Using large language models for expert prior elicitation in predictive modelling. In ICML, pages 6746–6777, 2025. URL https://proceedings.mlr.press/v267/capstick25a.html.

T. Chen and C. Guestrin. XGBoost: A scalable tree boosting system. In KDD, pages 785–794, 2016. doi: 10.1145/2939672.2939785.

B. R. de Oliveira et al. Dataset: Forty soybean cultivars from subsequent harvests. TAES, 2023. doi: 10.46420/TAES.e230005. URL https://editorapantanal.com.br/journal/index.php/taes/ en/article/view/8. Art. no. e230005.

N. Erickson et al. TabArena: A living benchmark for machine learning on tabular data. In NeurIPS, volume 38, pages 17285–17350, 2025. doi: 10.52202/085713-0519. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/1697e3fb412da11dc948824 9f9e7bbc9-Abstract-Datasets\_and\_Benchmarks\_Track.html.

fedesoriano. Heart failure prediction dataset. Kaggle, ver. 1 [Data set], Sep. 2021. URL https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction. [Online]. Available: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction.

J. Gardner, J. C. Perdomo, and L. Schmidt. Large scale transfer learning for tabular data via language modeling. In NeurIPS, volume 37, pages 45155–45205, 2024. URL https: //proceedings.neurips.cc/paper\_files/paper/2024/hash/4fd5cfd2e31bebbccfa5ffa354c04 bdc-Abstract-Conference.html.

S. E. Golovenkin et al. Trajectories, bifurcations, and pseudo-time in large clinical datasets: Applications to myocardial infarction and diabetes data. GigaScience, 9(11), 2020. doi: 10.1093/gigascience/giaa128. URL https://academic.oup.com/gigascience/article/9/11/gi aa128/6006352. Art. no. giaa128.

Google Research. TabFM: Tabular foundation models. GitHub, ver. 1.0.0 [Software], 2026. URL https://github.com/google-research/tabfm. [Online]. Available: https://github.com/googl e-research/tabfm.

Y. Gorishniy, A. Kotelnikov, and A. Babenko. TabM: Advancing tabular deep learning with parameter-efficient ensembling. In ICLR, pages 77899–77935, 2025. URL https://proceeding s.iclr.cc/paper\_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Con ference.html.

L. Grinsztajn, E. Oyallon, and G. Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? In NeurIPS, volume 35, pages 507–520, 2022. doi: 10.52202/068431-0037. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash /0378c7692da36807bdec87ab043cdadc-Abstract-Datasets\_and\_Benchmarks.html.

L. Grinsztajn et al. TabPFN-3: Technical report. arXiv:2605.13986, 2026. URL https://arxiv.org/ abs/2605.13986.

S. Han, J. Yoon, S. O. Arik, and T. Pfister. Large language models can automatically engineer features for few-shot tabular learning. In ICML, pages 17454–17479, 2024. URL https: //proceedings.mlr.press/v235/han24f.html.

S. Hegselmann, A. Buendia, H. Lang, M. Agrawal, X. Jiang, and D. Sontag. TabLLM: Few-shot classification of tabular data with large language models. In AISTATS, pages 5549–5581, 2023. URL https://proceedings.mlr.press/v206/hegselmann23a.html.

Hans Hofmann. Statlog (German Credit Data). UCI Machine Learning Repository, 1994. DOI: https://doi.org/10.24432/C5NC77.

N. Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, 2025. doi: 10.1038/s41586-024-08328-6. URL https://www.nature.com/a rticles/s41586-024-08328-6.

R. Knauer et al. ‘Oh LLM, I’m Asking Thee, Please Give Me a Decision Tree’: Zero-shot decision tree induction and embedding with large language models. In KDD, pages 1196–1206, 2025. doi: 10.1145/3711896.3736818. URL https://dl.acm.org/doi/10.1145/3711896.3736818.

S. Moro, R. M. S. Laureano, and P. Cortez. Using data mining for bank direct marketing: An application of the CRISP-DM methodology. In ESM, 2011. URL https://ciencia.iscte-iul.pt/ publications/using-data-mining-for-bank-direct-marketing-an-application-of-the-crisp-d m-methodology/344.

NA NA. National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset. UCI Machine Learning Repository, 2019. DOI: https://doi.org/10.24432/C5BS66.

J. Nam, J. Tack, K. Lee, H. Lee, and J. Shin. STUNT: Few-shot tabular learning with selfgenerated tasks from unlabeled tables. In ICLR, 2023. URL https://openreview.net/forum?i d=\_xlsjehDvlY.

M. Ojala and G. C. Garriga. Permutation tests for studying classifier performance. JMLR, 11 (62):1833–1863, 2010. URL https://www.jmlr.org/papers/v11/ojala10a.html.

J. Qu, D. Holzmüller, G. Varoquaux, and M. Le Morvan. TabICLv2: A better, faster, scalable, and open tabular foundation model. In ICML, 2026. URL https://arxiv.org/abs/2602.11139.

N. Seedat, N. Huynh, B. van Breugel, and M. van der Schaar. Curated LLM: Synergy of LLMs and data curation for tabular augmentation in low-data regimes. In ICML, pages 44060–44092, 2024. URL https://proceedings.mlr.press/v235/seedat24a.html.

R. Shi, H. Gu, H. Ye, Y. Dai, X. Shen, and X. Wang. Latte: Transfering LLMs’ latent-level knowledge for few-shot tabular learning. In IJCAI, pages 6173–6181, 2025. doi: 10.24963/ijc ai.2025/687. URL https://www.ijcai.org/proceedings/2025/687.

D. Slack and S. Singh. TABLET: Learning from instructions for tabular data. arXiv:2304.13188, 2023. doi: 10.48550/arXiv.2304.13188. URL https://arxiv.org/abs/2304.13188.

J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In SCAMC, pages 261–265, 1988. URL https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/.

P. Wang, D. Wang, H. Zhao, H. Ye, D. Guo, and Y. Chang. LLM empowered prototype learning for zero and few-shot tasks on tabular data. arXiv:2508.09263, 2025. URL https: //arxiv.org/abs/2508.09263.

J. Yan et al. Small models are LLM knowledge triggers for medical tabular prediction. In ICLR, pages 37337–37352, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/5 c8236f62e33b5224634069e64cb271a-Abstract-Conference.html.

H. Ye, J. Li, H. Zhao, D. Guo, and Y. Chang. LLM meeting decision trees on tabular data. In NeurIPS, volume 38, pages 130884–130920, 2025. doi: 10.52202/085713-3938. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ab3b2b8d2bb4a1be648a91d 150a3b87a-Abstract-Conference.html.

I.-C. Yeh, K.-J. Yang, and T.-M. Ting. Knowledge discovery on RFM model using Bernoulli sequence. ESWA, 36(3):5866–5871, 2009. doi: 10.1016/j.eswa.2008.07.018. URL https: //www.sciencedirect.com/science/article/pii/S0957417408004508.

J. Yoon, Y. Zhang, J. Jordon, and M. van der Schaar. VIME: Extending the success of self- and semi-supervised learning to tabular domain. In NeurIPS, volume 33, pages 11033–11043, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/7d97667a3e056aca b9aaf653807b4a03-Paper.pdf.

## A. Prompting Protocol

MARS uses the following system and user messages for all three backbones. The user message substitutes the binary task question for {question} and the feature summaries for {fields}.

MARS Prompt Template   
System message   
You are a domain expert. Using only your knowledge of the domain, you quantify   
,→ how each field of a table relates to the target described in the   
,→ question. You have no labelled examples. Answer with a single JSON object   
,→ and nothing else.   
User message   
Task question: {question}   
Answering "yes" is the positive class.   
Fields, each with reference points computed from unlabelled rows (seven   
,→ quantiles for numeric fields; levels with relative frequencies for   
,→ categorical fields): {fields}   
First, in one or two sentences, restate what the positive class means and   
,→ describe a typical positive case.   
Then, for each field, give the log-odds of the positive class at each of its   
,→ reference points, on a common scale: 0.0 is no evidence, +1.0 roughly   
,→ multiplies the odds by e, -1.0 roughly divides them by e; keep values   
,→ within [-3, 3]. Values need not follow a straight line across the   
,→ reference points. Keep the sign of each value consistent with your   
,→ description: a value typical of a positive case gets a positive log-odds.   
For each field also give discriminative\_strength in [0, 10]: how strongly it   
,→ separates the classes relative to the other fields; 0 means irrelevant.   
Answer with exactly this JSON object:   
{"positive\_class\_means": "<one sentence>",   
"typical\_positive\_case": "<one sentence>",   
"fields": [   
{"name": "<field name exactly as given>",   
"discriminative\_strength": <0-10>,   
"logit\_at\_points": [<one number per reference point, same order>]}   
]}

For a numerical feature, its summary gives the feature name, type, and ordered reference values. For a categorical feature, it gives the retained levels and their relative frequencies. Numerical anchors are rounded to six decimal places before duplicate removal; their textual values use general numeric formatting. Categorical frequencies are displayed as integer percentages. The prompt’s logit\_at\_points and discriminative\_strength correspond to the anchor support scores and feature importance in Section 3.2.

## B. Benchmark and Evaluation Protocol

## B.1. Datasets and Prediction Tasks

We evaluate MARS on eight binary classification tasks spanning financial services, healthcare, and agriculture. Table 3 summarizes the dataset sizes and feature types. Bank, Blood, Credit-G, Diabetes, Heart, and Myocardial use the processed datasets released with FeatLLM (Han et al., 2024), retaining their feature names, types, and task questions and mapping no/yes to 0/1. Cultivars and NHANES use their public dataset releases. The prediction tasks and dataset-specific preparation are described below.

Table 3: Benchmark sizes. Num./Cat. counts numerical and categorical input features; the target column is excluded.
<table><tr><td>Task</td><td>Rows</td><td>Features</td><td>Num./Cat.</td><td>Test rows</td></tr><tr><td>Bank</td><td>45,211</td><td>16</td><td>7/9</td><td>800</td></tr><tr><td>Blood</td><td>748</td><td>4</td><td>4/0</td><td>150</td></tr><tr><td>Credit-G</td><td>1,000</td><td>20</td><td>7/13</td><td>200</td></tr><tr><td>Diabetes</td><td>768</td><td>8</td><td>8/0</td><td>154</td></tr><tr><td>Heart</td><td>918</td><td>11</td><td>6/5</td><td>184</td></tr><tr><td>Cultivars</td><td>320</td><td>10</td><td>7/3</td><td>64</td></tr><tr><td>Myocardial</td><td>686</td><td>91</td><td>7/84</td><td>138</td></tr><tr><td>NHANES</td><td>2,278</td><td>7</td><td>4/3</td><td>456</td></tr></table>

1. Bank. A telephone-marketing dataset (Moro et al., 2011) for predicting whether a customer subscribes to a term deposit using customer background and contact records.

2. Blood. A blood-donation dataset (Yeh et al., 2009) for predicting whether an individual donates blood based on past donation records, including donation recency, frequency, and cumulative volume.

3. Credit-G. A credit-assessment dataset (Hofmann, 1994) for classifying individual credit risk using account status, credit history, and loan information.

4. Diabetes. A diabetes classification dataset (Smith et al., 1988) that uses clinical features such as glucose level, body mass index, and age to predict whether a patient has diabetes.

5. Heart. A heart-disease classification dataset (fedesoriano, 2021) that combines clinical features such as chest pain type, blood pressure, and heart rate to predict the presence of heart disease.

6. Cultivars. A dataset of soybean cultivars and field observations (de Oliveira et al., 2023) for predicting whether grain yield (GY) exceeds the fixed dataset median of 3397.276724 kg/ha using cultivar information and plant traits. Grain yield itself is excluded from the inputs; Season, Cultivar, and Repetition are categorical features. We split individual observations rather than holding out entire cultivars, so the same cultivar may appear in both the training and test partitions.

7. Myocardial. Clinical records of myocardial infarction patients (Golovenkin et al., 2020). We use the chronic-heart-failure classification task, with patient history and clinical measurements as inputs, retaining the released history feature ZSN\_A.

8. NHANES. A health and nutrition survey dataset (NA, 2019) for predicting whether a participant belongs to the age group of 65 years or older from questionnaire responses, physical examination results, and biochemical measurements. Age (RIDAGEYR), the agegroup label (age\_group), and the participant identifier (SEQN) are excluded from the inputs.

## B.2. Splits, Label Access, and Metrics

We use an 80/20 stratified split with random seed 0. From the training partition, two examples per class are reserved for the four-label comparators. The remaining training features form the unlabeled reference pool. The same support examples and test records are shared by the methods. For test partitions larger than 800 rows, we sample approximately 800 rows proportionally by class with seed 0; unused test records are not moved into the reference pool. Table 3 gives the resulting evaluation sizes, totaling 2,146 records.

Labels are used by the evaluator to define the split, select the four comparator examples, and compute metrics. MARS receives only the task question and reference features. Its prompt, anchor selection, aggregation, and prediction do not access support or test labels. The comparison uses no additional labeled validation set.

We compute ROC-AUC and average precision from the continuous prediction scores and average task-level metrics with equal task weights. The main table reports this fixed split. The five elicitation responses are repeated LLM requests, not five data-split seeds. For responsecount sensitivity, every subset of size R from the five cached responses is evaluated, first averaging over subsets within each task and then over tasks. The single-response ablation averages the five one-response models.

## C. Hyperparameters and Implementation Details

## C.1. MARS Construction Settings

The construction settings in Table 4 are shared by the eight tasks and three backbones. MARS constructs response functions by aggregation and interpolation and elicits feature weights from the LLM, without a task-specific optimizer, learning rate, or training schedule. For each feature, we aggregate valid replies whose score vectors contain exactly one finite value per anchor.

Table 4: MARS construction settings used in the reported experiments.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Task-specific labeled examples</td><td>0</td></tr><tr><td>LLM responses per task</td><td>R = 5, all retained features per response</td></tr><tr><td>Numerical quantiles</td><td>0.05,0.20,0.35,0.50,0.65,0.80,0.95</td></tr><tr><td>Numerical anchor preparation</td><td>Round to six decimals, remove duplicates; omit features with fewer than two anchors</td></tr><tr><td>Categorical levels</td><td>Up to 20 most frequent levels</td></tr><tr><td>Anchor score clipping</td><td>[-3, 3], before aggregation</td></tr><tr><td>Feature importance requested</td><td>[0,10]</td></tr><tr><td>Aggregation</td><td>Coordinatewise median over valid feature responses</td></tr><tr><td>Feature weight</td><td>max(0, median b(r)/10</td></tr><tr><td>Numerical prediction</td><td>Piecewise linear interpolation, endpoint extension</td></tr><tr><td>Categorical prediction</td><td>Lookup; unseen levels contribute 0</td></tr></table>

## C.2. LLM Generation Settings

Table 5 records the generation settings for MARS. DeepSeek uses the API model identifier deepseek-v4-flash, recorded as DeepSeek-V4-Flash-0731 in the experiment. Thinking is enabled with high reasoning effort and JSON-object output. Temperature and top-p were not

explicitly set in these requests. Qwen3.5-4B and Qwen3.5-9B use their native thinking template with the same sampling settings as each other.

Table 5: MARS generation settings. “Not set” means the request contains no override for that parameter.
<table><tr><td>Parameter</td><td>DeepSeek-V4-Flash</td><td>Qwen3.5-4B / 9B</td></tr><tr><td>Thinking</td><td>Enabled; high effort</td><td>Native template enabled</td></tr><tr><td>Maximum output tokens</td><td>32,768</td><td>32,768</td></tr><tr><td>Temperature</td><td>Not set</td><td>1.0</td></tr><tr><td>Top-p</td><td>Not set</td><td>0.95</td></tr><tr><td>Top-k</td><td>Not set</td><td>20</td></tr><tr><td>Min-p</td><td>Not set</td><td>0.0</td></tr><tr><td>Presence penalty</td><td>Not set</td><td>1.5</td></tr><tr><td>Repetition penalty</td><td>Not set</td><td>1.0</td></tr><tr><td>Frequency penalty</td><td>Not set</td><td>0.0</td></tr></table>

Qwen inference uses bfloat16, tensor parallel size 1, engine seed 0, a maximum context length of 81,920 tokens, and GPU-memory utilization 0.92. The engine allows at most 8 sequences and 4,096 batched tokens, with chunked prefill and prefix caching enabled.